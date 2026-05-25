"""
Term Structure Factor Engine for Futures Trading
=================================================
Comprehensive term structure analysis that converts raw curve JSON files
into factor DataFrame for backtesting and strategy development.

Factors computed per symbol per date:
  - basis_spread: annualized near-far spread %
  - basis_zscore: z-score vs 60-day rolling window
  - structure_state: contango(+1) / flat(0) / backwardation(-1)
  - curve_slope: linear regression slope across full curve
  - roll_yield_proxy: annualized roll yield estimate
  - spread_momentum_5d / 20d: change in spread over N days
  - extreme_signal: spread beyond 2σ from mean

Cross-commodity factors:
  - group_basis_avg: average basis for same commodity group
  - basis_rank: percentile rank within group

Commodity groups:
  BLACK: rbfi, hcfi, ifi, jfi, jmfi
  METAL: cufi, alfi, znfi, pbfi, nifi, snfi, aufi, agfi
  ENERGY: scfi, bufi, fufi, tafi, mafi
  AGRI: mfi, yfi, ofi, rmfi, cfi, srfi, cffi
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TS_DIR = '/Users/chengming/home/futures_platform/data/futures_term_structure/'
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

COMMODITY_GROUPS = {
    'BLACK':  ['rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi'],
    'METAL':  ['cufi', 'alfi', 'znfi', 'pbfi', 'nifi', 'snfi', 'aufi', 'agfi'],
    'ENERGY': ['scfi', 'bufi', 'fufi', 'tafi', 'mafi'],
    'AGRI':   ['mfi', 'yfi', 'ofi', 'rmfi', 'cfi', 'srfi', 'cffi'],
}

# Build reverse lookup: symbol -> group name
_SYM_TO_GROUP = {}
for _grp, _syms in COMMODITY_GROUPS.items():
    for _s in _syms:
        _SYM_TO_GROUP[_s] = _grp

# Signal parameters
ZSCORE_WINDOW = 60          # rolling window for z-score
SPREAD_MOM_WINDOWS = [5, 20]  # momentum lookback periods
EXTREME_SIGMA = 2.0         # σ threshold for extreme signal
FLAT_THRESHOLD = 0.003      # |spread_pct| below this = flat structure


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------
def load_term_structure_data(
    ts_dir: str = TS_DIR,
    symbols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load all term structure JSON files into a unified DataFrame.

    Parameters
    ----------
    ts_dir : str
        Path to directory containing JSON files.
    symbols : list[str] or None
        If provided, only load these symbols. Otherwise load everything.

    Returns
    -------
    pd.DataFrame with columns:
        symbol, date, structure, near_contract, near_price,
        far_contract, far_price, total_spread, total_spread_pct,
        curve (list of dicts), n_contracts
    """
    print(f"[TS] Loading term structure data from {ts_dir} ...", flush=True)
    t0 = time.time()

    sym_set = set(symbols) if symbols else None

    # Collect raw records
    records = []
    errors = 0
    skipped = 0

    filenames = sorted(os.listdir(ts_dir))
    for fname in filenames:
        if not fname.endswith('.json'):
            continue

        parts = fname.rsplit('_', 1)
        if len(parts) != 2:
            continue
        sym, date_part = parts
        date_part = date_part.replace('.json', '')

        if sym_set and sym not in sym_set:
            skipped += 1
            continue

        fpath = os.path.join(ts_dir, fname)
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
        except Exception:
            errors += 1
            continue

        # Validate minimal fields
        if not data.get('near_price') or not data.get('far_price'):
            errors += 1
            continue
        if data['near_price'] <= 0:
            errors += 1
            continue

        curve = data.get('curve', [])
        records.append({
            'symbol': sym,
            'date': pd.Timestamp(data['date']),
            'structure_raw': data.get('structure', ''),
            'near_contract': data.get('near_contract', ''),
            'near_price': float(data['near_price']),
            'far_contract': data.get('far_contract', ''),
            'far_price': float(data['far_price']),
            'total_spread': float(data.get('total_spread', 0)),
            'total_spread_pct': float(data.get('total_spread_pct', 0)),
            'curve': curve,
            'n_contracts': len(curve),
        })

    df = pd.DataFrame(records)
    if len(df) == 0:
        print(f"[TS] WARNING: No records loaded.")
        return df

    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    unique_syms = df['symbol'].nunique()
    date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
    print(f"  Loaded {len(df):,} records for {unique_syms} symbols")
    print(f"  Date range: {date_range}")
    print(f"  Skipped: {skipped:,}, errors: {errors}")
    print(f"  Done in {time.time() - t0:.1f}s", flush=True)

    return df


# ---------------------------------------------------------------------------
# 2. Curve-level feature extraction (applied per row)
# ---------------------------------------------------------------------------
def _extract_curve_features(curve: List[dict]) -> Dict[str, float]:
    """Extract quantitative features from a single day's curve.

    Returns dict with:
      - curve_slope : OLS slope of price ~ contract_index (normalized)
      - front_avg   : average of front-half contracts
      - back_avg    : average of back-half contracts
      - spread_front_back_pct : (front_avg - back_avg) / front_avg
    """
    if not curve or len(curve) < 2:
        return {
            'curve_slope': np.nan,
            'front_avg': np.nan,
            'back_avg': np.nan,
            'spread_front_back_pct': np.nan,
        }

    # Sort by year, month
    sorted_c = sorted(curve, key=lambda x: (x.get('year', 0), x.get('month', 0)))
    prices = [c['price'] for c in sorted_c if c.get('price') and c['price'] > 0]

    if len(prices) < 2:
        return {
            'curve_slope': np.nan,
            'front_avg': np.nan,
            'back_avg': np.nan,
            'spread_front_back_pct': np.nan,
        }

    n = len(prices)
    prices_arr = np.array(prices, dtype=float)

    # Linear regression: price = slope * index + intercept
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = prices_arr.mean()
    slope = np.sum((x - x_mean) * (prices_arr - y_mean)) / (np.sum((x - x_mean) ** 2) + 1e-12)
    # Normalize slope as % of mean price
    slope_pct = slope / (y_mean + 1e-12)

    # Front vs back averages
    mid = n // 2
    front_avg = prices_arr[:mid].mean() if mid > 0 else prices_arr[0]
    back_avg = prices_arr[mid:].mean()

    spread_fb_pct = (front_avg - back_avg) / (front_avg + 1e-12)

    return {
        'curve_slope': slope_pct,
        'front_avg': front_avg,
        'back_avg': back_avg,
        'spread_front_back_pct': spread_fb_pct,
    }


def _estimate_months_to_expiry(near_contract: str, far_contract: str) -> float:
    """Rough estimate of months between near and far contract.

    Parses contract codes like RB2101 -> year=21, month=01.
    Falls back to 3 months if parsing fails.
    """
    try:
        def parse_ym(code):
            # Strip letter prefix, get numeric part e.g. RB2101 -> 2101
            num = ''.join(c for c in code if c.isdigit())
            if len(num) >= 4:
                yr = int(num[:2])
                mo = int(num[2:4])
                return yr, mo
            return None, None

        ny, nm = parse_ym(near_contract)
        fy, fm = parse_ym(far_contract)
        if ny is not None and fy is not None:
            return (fy - ny) * 12 + (fm - nm)
    except Exception:
        pass
    return 3.0


# ---------------------------------------------------------------------------
# 3. Factor Computation
# ---------------------------------------------------------------------------
def compute_factors(
    raw_df: pd.DataFrame,
    zscore_window: int = ZSCORE_WINDOW,
    mom_windows: List[int] = None,
    extreme_sigma: float = EXTREME_SIGMA,
) -> pd.DataFrame:
    """Compute all term structure factors from raw loaded data.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Output from load_term_structure_data().
    zscore_window : int
        Rolling window for z-score computation (default 60).
    mom_windows : list[int]
        Lookback periods for spread momentum (default [5, 20]).
    extreme_sigma : float
        Number of standard deviations for extreme signal (default 2.0).

    Returns
    -------
    pd.DataFrame with all factor columns added.
    """
    if mom_windows is None:
        mom_windows = SPREAD_MOM_WINDOWS

    print("[TS] Computing factors ...", flush=True)
    t0 = time.time()
    df = raw_df.copy()

    # ------------------------------------------------------------------
    # 3a. Basic basis features (vectorized)
    # ------------------------------------------------------------------
    # basis_spread: (near - far) / near, annualized
    #   positive = backwardation (near > far)
    #   negative = contango (near < far)
    df['basis_pct'] = (df['near_price'] - df['far_price']) / df['near_price']

    # Annualize: estimate months between contracts, scale to 12 months
    df['months_gap'] = df.apply(
        lambda r: _estimate_months_to_expiry(
            r.get('near_contract', ''), r.get('far_contract', '')
        ),
        axis=1,
    )
    df['basis_spread_ann'] = df['basis_pct'] / (df['months_gap'] / 12.0 + 1e-6)

    # structure_state: +1 backwardation, -1 contango, 0 flat
    def _structure_state(row):
        pct = row['basis_pct']
        if pd.isna(pct):
            return 0
        if pct > FLAT_THRESHOLD:
            return 1   # backwardation
        elif pct < -FLAT_THRESHOLD:
            return -1  # contango
        return 0

    df['structure_state'] = df.apply(_structure_state, axis=1)

    # ------------------------------------------------------------------
    # 3b. Curve-level features (per-row apply)
    # ------------------------------------------------------------------
    print("  Computing curve features ...", flush=True)
    curve_feats = df['curve'].apply(_extract_curve_features)
    curve_df = pd.DataFrame(curve_feats.tolist(), index=df.index)
    df = pd.concat([df, curve_df], axis=1)

    # roll_yield_proxy: annualized roll yield
    #   In backwardation (basis > 0), long position earns positive roll yield
    #   In contango (basis < 0), long position pays negative roll yield
    #   Formula: basis_pct * (12 / months_gap)
    df['roll_yield_proxy'] = df['basis_pct'] * (12.0 / (df['months_gap'] + 1e-6))

    # ------------------------------------------------------------------
    # 3c. Time-series factors (per-symbol rolling)
    # ------------------------------------------------------------------
    print("  Computing rolling z-score and momentum ...", flush=True)
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    # Z-score of basis_pct over rolling window
    df['basis_mean'] = df.groupby('symbol')['basis_pct'].transform(
        lambda s: s.rolling(zscore_window, min_periods=10).mean()
    )
    df['basis_std'] = df.groupby('symbol')['basis_pct'].transform(
        lambda s: s.rolling(zscore_window, min_periods=10).std()
    )
    df['basis_zscore'] = (df['basis_pct'] - df['basis_mean']) / (df['basis_std'] + 1e-10)

    # Spread momentum
    for w in mom_windows:
        col = f'spread_momentum_{w}d'
        df[col] = df.groupby('symbol')['basis_pct'].transform(
            lambda s: s.diff(w)
        )

    # Extreme signal: |z-score| > extreme_sigma
    df['extreme_signal'] = 0
    df.loc[df['basis_zscore'] > extreme_sigma, 'extreme_signal'] = 1    # extreme backwardation
    df.loc[df['basis_zscore'] < -extreme_sigma, 'extreme_signal'] = -1  # extreme contango

    # Drop helper columns
    df.drop(columns=['basis_mean', 'basis_std'], inplace=True)

    print(f"  Factor computation done ({time.time() - t0:.1f}s)", flush=True)
    return df


# ---------------------------------------------------------------------------
# 4. Cross-Commodity Factors
# ---------------------------------------------------------------------------
def compute_cross_commodity_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Add group-level factors: group_basis_avg, basis_rank.

    Parameters
    ----------
    df : pd.DataFrame
        Output from compute_factors(), must have 'basis_pct' column.

    Returns
    -------
    pd.DataFrame with added cross-commodity columns.
    """
    print("[TS] Computing cross-commodity factors ...", flush=True)
    t0 = time.time()

    df = df.copy()

    # Assign group
    df['commodity_group'] = df['symbol'].map(_SYM_TO_GROUP).fillna('OTHER')

    # Group average basis (same group, same date)
    df['group_basis_avg'] = df.groupby(['commodity_group', 'date'])['basis_pct'].transform(
        'mean'
    )

    # Basis rank within group on each date (0 = most contango, 1 = most backwardation)
    def _rank_in_group(group):
        if len(group) <= 1:
            return pd.Series(0.5, index=group.index)
        return group['basis_pct'].rank(pct=True)

    df['basis_rank'] = df.groupby(['commodity_group', 'date']).apply(
        _rank_in_group
    ).reset_index(level=[0, 1], drop=True)

    # Group-level stats
    n_grouped = df[df['commodity_group'] != 'OTHER']['commodity_group'].nunique()
    n_other = (df['commodity_group'] == 'OTHER').sum()
    print(f"  {n_grouped} groups mapped, {n_other:,} rows in OTHER")
    print(f"  Cross-commodity factors done ({time.time() - t0:.1f}s)", flush=True)

    return df


# ---------------------------------------------------------------------------
# 5. Full Pipeline
# ---------------------------------------------------------------------------
def build_term_structure_factors(
    ts_dir: str = TS_DIR,
    symbols: Optional[List[str]] = None,
    zscore_window: int = ZSCORE_WINDOW,
    mom_windows: List[int] = None,
    extreme_sigma: float = EXTREME_SIGMA,
    save_csv: bool = True,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Full pipeline: load -> compute factors -> cross-commodity -> export.

    Parameters
    ----------
    ts_dir : str
        Path to term structure JSON directory.
    symbols : list[str] or None
        Symbols to load. None = all.
    zscore_window : int
        Rolling z-score window.
    mom_windows : list[int]
        Momentum lookback periods.
    extreme_sigma : float
        Sigma threshold for extreme signal.
    save_csv : bool
        Whether to save result to CSV.
    output_path : str or None
        Custom CSV output path. Default: term_structure_factors.csv in same dir.

    Returns
    -------
    pd.DataFrame with all factor columns.
    """
    print("=" * 80)
    print("Term Structure Factor Engine")
    print("=" * 80)

    t_total = time.time()

    # Step 1: Load raw data
    raw_df = load_term_structure_data(ts_dir, symbols)
    if raw_df.empty:
        return raw_df

    # Step 2: Compute per-symbol per-date factors
    factor_df = compute_factors(raw_df, zscore_window, mom_windows, extreme_sigma)

    # Step 3: Cross-commodity factors
    factor_df = compute_cross_commodity_factors(factor_df)

    # Step 4: Clean up - drop curve column (not CSV-friendly) before saving
    export_df = factor_df.drop(columns=['curve'], errors='ignore')

    if save_csv:
        if output_path is None:
            output_path = os.path.join(OUTPUT_DIR, 'term_structure_factors.csv')
        print(f"[TS] Saving to {output_path} ...", flush=True)
        export_df.to_csv(output_path, index=False)
        print(f"  Saved {len(export_df):,} rows x {len(export_df.columns)} columns")

    elapsed = time.time() - t_total
    print("=" * 80)
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"  Output: {len(export_df):,} rows, {export_df['symbol'].nunique()} symbols")
    print(f"  Date range: {export_df['date'].min()} to {export_df['date'].max()}")
    print(f"  Factor columns: {sorted([c for c in export_df.columns if c not in ('symbol', 'date', 'structure_raw', 'near_contract', 'far_contract', 'commodity_group')])}")
    print("=" * 80)

    return factor_df


# ---------------------------------------------------------------------------
# Convenience: load factors from CSV (for backtesting use)
# ---------------------------------------------------------------------------
def load_factors(
    csv_path: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load pre-computed factor CSV with filtering.

    Parameters
    ----------
    csv_path : str or None
        Path to CSV. Default: term_structure_factors.csv in same directory.
    symbols : list[str] or None
        Filter to these symbols.
    start_date, end_date : str or None
        Date range filter (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame with parsed dates.
    """
    if csv_path is None:
        csv_path = os.path.join(OUTPUT_DIR, 'term_structure_factors.csv')

    df = pd.read_csv(csv_path, parse_dates=['date'])

    if symbols:
        df = df[df['symbol'].isin(symbols)]
    if start_date:
        df = df[df['date'] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df['date'] <= pd.Timestamp(end_date)]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Quick summary stats for a factor
# ---------------------------------------------------------------------------
def factor_summary(df: pd.DataFrame, factor_col: str) -> pd.DataFrame:
    """Per-symbol summary statistics for a given factor column."""
    if factor_col not in df.columns:
        raise ValueError(f"Column '{factor_col}' not found. Available: {list(df.columns)}")

    return df.groupby('symbol')[factor_col].agg(
        count='count',
        mean='mean',
        std='std',
        min='min',
        q25=lambda x: x.quantile(0.25),
        median='median',
        q75=lambda x: x.quantile(0.75),
        max='max',
    ).round(6)


# ---------------------------------------------------------------------------
# CLI / Test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Quick test: load and print sample factors for rbfi
    print("\n" + "=" * 80)
    print("Quick Test: Loading rbfi term structure factors")
    print("=" * 80)

    # Build factors for rbfi only (fast)
    df = build_term_structure_factors(
        symbols=['rbfi'],
        save_csv=False,
    )

    if df.empty:
        print("ERROR: No data loaded for rbfi")
    else:
        print(f"\nTotal rows for rbfi: {len(df)}")
        print(f"\nColumns ({len(df.columns)}):")
        for c in df.columns:
            print(f"  {c}")

        # Show most recent 10 rows of key factors
        key_cols = [
            'symbol', 'date', 'structure_state', 'basis_pct',
            'basis_spread_ann', 'basis_zscore', 'curve_slope',
            'roll_yield_proxy', 'spread_momentum_5d', 'spread_momentum_20d',
            'extreme_signal', 'commodity_group', 'group_basis_avg', 'basis_rank',
        ]
        available_cols = [c for c in key_cols if c in df.columns]
        print(f"\nMost recent 10 rows (key factors):")
        print(df[available_cols].tail(10).to_string(index=False))

        # Summary stats
        print(f"\nFactor summary for 'basis_pct':")
        print(factor_summary(df, 'basis_pct').to_string())

        print(f"\nFactor summary for 'basis_zscore':")
        print(factor_summary(df, 'basis_zscore').to_string())

        print(f"\nFactor summary for 'curve_slope':")
        print(factor_summary(df, 'curve_slope').to_string())

        # Structure distribution
        print(f"\nStructure state distribution:")
        print(df['structure_state'].value_counts().sort_index().to_string())

        # Extreme signal frequency
        print(f"\nExtreme signal frequency:")
        print(df['extreme_signal'].value_counts().sort_index().to_string())

    print("\nDone.")
