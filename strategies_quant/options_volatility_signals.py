"""
Options Volatility Analysis Module for Futures Trading Signal Generation

Reads options chain data from two sources:
  - tq_options/     : raw TQSDK option chain snapshots (IV in percent, e.g. 24.64)
  - options_calculated/ : pre-calculated Greeks with cleaner data (IV as decimal, e.g. 0.1497)

Primary analysis uses options_calculated/ (cleaner Greeks, consistent decimal IV).
Falls back to tq_options/ when calculated data is missing.

Outputs a DataFrame of daily volatility signals per product, saved as CSV.
"""

import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("/Users/chengming/home/futures_platform/data")
TQ_OPTIONS_DIR = DATA_DIR / "tq_options"
CALC_OPTIONS_DIR = DATA_DIR / "options_calculated"
OUTPUT_DIR = Path("/Users/chengming/home/futures_platform/strategies_quant/output")

# IV percentile lookback window (trading days)
IV_PERCENTILE_WINDOW = 60

# Signal thresholds
IV_HIGH_PCT = 90          # percentile threshold for "extreme high"
IV_LOW_PCT = 10           # percentile threshold for "extreme low"
IV_RISING_DAYS = 5        # consecutive rising days to trigger signal
SKEW_EXTREME_PCT = 90     # percentile for extreme skew


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def _parse_product_date_from_filename(filename: str):
    """Extract (product, date_str) from filename patterns.

    Handles both naming conventions:
      - 'DCE_a_20260520.json' or 'a_20260520.json'  (options_calculated)
      - 'CZCE_AP_20260520.json'                       (tq_options)
    """
    stem = Path(filename).stem
    # Find the date portion: last 8 digits following an underscore
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 8:
        return parts[0], parts[1]
    return stem, None


def load_calculated_options(directory: Path = CALC_OPTIONS_DIR) -> pd.DataFrame:
    """Load all options_calculated JSON files into a single DataFrame.

    Each row = one option contract on one date.
    Uses the internal ``product`` field for normalised product identification.
    """
    records = []
    if not directory.exists():
        return pd.DataFrame()

    for fp in sorted(directory.glob("*.json")):
        file_product, date_str = _parse_product_date_from_filename(fp.name)
        if date_str is None:
            continue
        # Skip known bad data: 20260516 files all contain CFFEX HO data
        if date_str == "20260516":
            continue
        try:
            with open(fp) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            item["_date"] = date_str
            records.append(item)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Standardise column names
    col_map = {
        "option_type": "option_class",
        "implied_vol": "implied_volatility",
        "strike": "strike_price",
    }
    df.rename(columns=col_map, inplace=True)

    # Use the internal product field, not the filename prefix
    df["_product_key"] = df["product"].astype(str).str.strip()

    # Ensure option_class is uppercase CALL/PUT
    df["option_class"] = df["option_class"].str.upper().str.strip()

    # IV in options_calculated is already decimal (0.15 = 15%)
    # Replace 0 / NaN with NaN so they are ignored in aggregation
    df["implied_volatility"] = pd.to_numeric(df["implied_volatility"], errors="coerce")
    df.loc[df["implied_volatility"] <= 0, "implied_volatility"] = np.nan
    # Cap unreasonably high IV (> 300% = 3.0 in decimal)
    df.loc[df["implied_volatility"] > 3.0, "implied_volatility"] = np.nan

    df["delta"] = pd.to_numeric(df.get("delta"), errors="coerce")
    df["gamma"] = pd.to_numeric(df.get("gamma"), errors="coerce")
    df["theta"] = pd.to_numeric(df.get("theta"), errors="coerce")
    df["vega"] = pd.to_numeric(df.get("vega"), errors="coerce")
    df["strike_price"] = pd.to_numeric(df["strike_price"], errors="coerce")
    df["underlying_price"] = pd.to_numeric(df["underlying_price"], errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
    df["open_interest"] = pd.to_numeric(df.get("open_interest", 0), errors="coerce").fillna(0)
    df["days_to_expiry"] = pd.to_numeric(df.get("days_to_expiry"), errors="coerce")

    df["_date"] = pd.to_datetime(df["_date"], format="%Y%m%d")
    df["_source"] = "calculated"

    return df


def load_tq_options(directory: Path = TQ_OPTIONS_DIR) -> pd.DataFrame:
    """Load all tq_options JSON files into a single DataFrame.

    IV in tq_options is in percent (24.64 = 24.64%).  We convert to decimal.
    Uses the internal ``product`` field for normalised product identification.
    """
    records = []
    if not directory.exists():
        return pd.DataFrame()

    for fp in sorted(directory.glob("*.json")):
        _, date_str = _parse_product_date_from_filename(fp.name)
        if date_str is None:
            continue
        try:
            with open(fp) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            item["_date"] = date_str
            records.append(item)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Use the internal product field, not the filename prefix
    df["_product_key"] = df["product"].astype(str).str.strip()

    df["option_class"] = df["option_class"].str.upper().str.strip()

    # Convert IV from percent to decimal
    df["implied_volatility"] = pd.to_numeric(df["implied_volatility"], errors="coerce")
    # 0 means no calculation available -> NaN
    df.loc[df["implied_volatility"] <= 0, "implied_volatility"] = np.nan
    # If value > 1 it is in percent; convert to decimal
    mask_pct = df["implied_volatility"] > 1
    df.loc[mask_pct, "implied_volatility"] = df.loc[mask_pct, "implied_volatility"] / 100.0
    # Cap unreasonably high IV (> 300% = 3.0 in decimal)
    df.loc[df["implied_volatility"] > 3.0, "implied_volatility"] = np.nan

    df["delta"] = pd.to_numeric(df.get("delta"), errors="coerce")
    df.loc[df["delta"].abs() > 2, "delta"] = np.nan  # bad data guard
    df["strike_price"] = pd.to_numeric(df["strike_price"], errors="coerce")
    df["underlying_price"] = pd.to_numeric(df["underlying_price"], errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
    df["open_interest"] = pd.to_numeric(df.get("open_interest", 0), errors="coerce").fillna(0)

    # Compute days_to_expiry from expire_datetime
    if "expire_datetime" in df.columns:
        df["expire_datetime"] = pd.to_numeric(df["expire_datetime"], errors="coerce")
        df["_date_dt"] = pd.to_datetime(df["_date"], format="%Y%m%d")
        df["expire_date"] = pd.to_datetime(df["expire_datetime"], unit="s", errors="coerce")
        df["days_to_expiry"] = (df["expire_date"] - df["_date_dt"]).dt.days
        df.drop(columns=["_date_dt", "expire_date"], inplace=True)

    df["_date"] = pd.to_datetime(df["_date"], format="%Y%m%d")
    df["_source"] = "tq_options"

    return df


def load_all_options() -> pd.DataFrame:
    """Load options data from both sources, preferring calculated data.

    The ``_product_key`` column is normalised to the internal product field
    from each data source so that both sources can be compared side-by-side.
    Calculated-source data is preferred where it overlaps with tq_options.
    """
    df_calc = load_calculated_options()
    df_tq = load_tq_options()

    common_cols = [
        "_product_key", "_date", "_source", "option_class",
        "strike_price", "underlying_price", "implied_volatility",
        "delta", "volume", "open_interest", "days_to_expiry",
    ]

    def _select_cols(df):
        existing = [c for c in common_cols if c in df.columns]
        return df[existing]

    if df_calc.empty and df_tq.empty:
        return pd.DataFrame(columns=common_cols)
    if df_calc.empty:
        return _select_cols(df_tq)
    if df_tq.empty:
        return _select_cols(df_calc)

    # Merge: keep calculated rows, add tq_options rows for (product, date) combos
    # not already covered.
    df_c = _select_cols(df_calc)
    df_t = _select_cols(df_tq)
    merged = pd.concat([df_c, df_t], ignore_index=True)
    # Deduplicate: prefer calculated source
    merged.sort_values("_source", ascending=True, inplace=True)  # calculated first
    merged.drop_duplicates(subset=["_product_key", "_date", "option_class",
                                    "strike_price", "days_to_expiry"],
                           keep="first", inplace=True)
    return merged


# ---------------------------------------------------------------------------
# IV Analysis  (per product, per date)
# ---------------------------------------------------------------------------

def _atm_iv(group: pd.DataFrame) -> Optional[float]:
    """ATM IV: IV of the option whose strike is closest to the underlying price.

    Looks at options with reasonable delta (|delta| between 0.3 and 0.7) first;
    if none found, falls back to absolute closest strike.
    Requires at least 3 contracts with valid IV to produce a result.
    """
    valid = group.dropna(subset=["implied_volatility"])
    if len(valid) < 3:
        return np.nan

    near = valid.copy()
    near["_strike_dist"] = (near["strike_price"] - near["underlying_price"]).abs()

    # Prefer options in the 0.3-0.7 delta band
    if "delta" in near.columns:
        near_delta = near[(near["delta"].abs() >= 0.3) & (near["delta"].abs() <= 0.7)]
        if not near_delta.empty:
            near_delta = near_delta.nsmallest(5, "_strike_dist")
            return near_delta["implied_volatility"].median()

    # Fallback: just closest strike
    closest = near.nsmallest(5, "_strike_dist")
    return closest["implied_volatility"].median()


def _delta_target_iv(group: pd.DataFrame, target_delta: float,
                     option_class: str) -> Optional[float]:
    """IV of the option nearest to `target_delta` for a given class (CALL/PUT).

    If no options with valid delta are found, falls back to strike-based
    moneyness: select OTM options within 5-15% of underlying price.
    """
    subset = group[(group["option_class"] == option_class)]
    valid = subset.dropna(subset=["implied_volatility", "delta"])
    if not valid.empty:
        valid = valid.copy()
        valid["_delta_dist"] = (valid["delta"].abs() - abs(target_delta)).abs()
        closest = valid.nsmallest(3, "_delta_dist")
        if not closest.empty:
            return closest["implied_volatility"].median()

    # Fallback: use strike-based moneyness for 25-delta equivalent
    # ~25-delta CALL: strike ~5-10% above underlying
    # ~25-delta PUT:  strike ~5-10% below underlying
    valid2 = subset.dropna(subset=["implied_volatility"]).copy()
    if valid2.empty:
        return np.nan
    up = valid2["underlying_price"].iloc[0]
    if up <= 0:
        return np.nan
    valid2["_moneyness"] = valid2["strike_price"] / up
    if option_class == "CALL":
        # OTM call: strike > underlying, target ~1.05-1.10 moneyness
        otm = valid2[valid2["strike_price"] > up]
        if otm.empty:
            otm = valid2
        otm = otm.copy()
        otm["_dist"] = (otm["_moneyness"] - 1.075).abs()
    else:
        # OTM put: strike < underlying, target ~0.90-0.95 moneyness
        otm = valid2[valid2["strike_price"] < up]
        if otm.empty:
            otm = valid2
        otm = otm.copy()
        otm["_dist"] = (otm["_moneyness"] - 0.925).abs()
    closest = otm.nsmallest(3, "_dist")
    if closest.empty:
        return np.nan
    return closest["implied_volatility"].median()


def _iv_skew(group: pd.DataFrame) -> Optional[float]:
    """Put-call skew: (25d_put_iv - 25d_call_iv) / atm_iv.

    A positive skew means puts are more expensive (demand for downside protection).
    Returns NaN if the result is unreasonable (>5 in absolute value).
    """
    atm = _atm_iv(group)
    if pd.isna(atm) or atm == 0:
        return np.nan
    put_25d = _delta_target_iv(group, -0.25, "PUT")
    call_25d = _delta_target_iv(group, 0.25, "CALL")
    if pd.isna(put_25d) or pd.isna(call_25d):
        return np.nan
    skew = (put_25d - call_25d) / atm
    # Clamp unreasonable values (data quality issues)
    if abs(skew) > 5:
        return np.nan
    return skew


def _iv_term_slope(group: pd.DataFrame) -> Optional[float]:
    """IV term structure slope: near-term IV minus far-term IV.

    Positive = backwardation (near term more expensive = fear / event risk).
    Uses median IV per expiration bucket.
    """
    if "days_to_expiry" not in group.columns:
        return np.nan
    valid = group.dropna(subset=["implied_volatility", "days_to_expiry"])
    if valid.empty:
        return np.nan
    # Filter out invalid DTE values
    valid = valid[valid["days_to_expiry"] > 0]
    if valid.empty:
        return np.nan

    near = valid[valid["days_to_expiry"] <= 30]
    far = valid[valid["days_to_expiry"] > 30]
    if near.empty or far.empty:
        return np.nan

    near_iv = near.groupby("days_to_expiry")["implied_volatility"].median().median()
    far_iv = far.groupby("days_to_expiry")["implied_volatility"].median().median()
    return near_iv - far_iv


def _oi_weighted_iv(group: pd.DataFrame) -> Optional[float]:
    """Open-interest weighted average IV across the chain."""
    valid = group.dropna(subset=["implied_volatility"])
    valid = valid[valid["open_interest"] > 0]
    if valid.empty:
        return np.nan
    weights = valid["open_interest"]
    return np.average(valid["implied_volatility"], weights=weights)


def _put_call_oi_ratio(group: pd.DataFrame) -> Optional[float]:
    """Total PUT open interest / total CALL open interest."""
    put_oi = group.loc[group["option_class"] == "PUT", "open_interest"].sum()
    call_oi = group.loc[group["option_class"] == "CALL", "open_interest"].sum()
    if call_oi == 0:
        return np.nan
    return put_oi / call_oi


def compute_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all IV metrics for each (product, date) combination.

    Returns a DataFrame with one row per (product, date).
    """
    if df.empty:
        return pd.DataFrame()

    # Pre-filter: remove expired contracts (DTE <= 0) and extremely far expiries
    if "days_to_expiry" in df.columns:
        df = df[df["days_to_expiry"] > 0].copy()
        df = df[df["days_to_expiry"] <= 365].copy()

    # Remove contracts with zero/invalid underlying price
    df = df[df["underlying_price"] > 0].copy()

    results = []
    for (product, date), group in df.groupby(["_product_key", "_date"]):
        # Focus on the most liquid expiry for ATM/skew calculations:
        # pick the expiry bucket with the highest volume.
        # But keep the full group for term structure analysis.
        if "days_to_expiry" in group.columns and group["days_to_expiry"].notna().any():
            # Group by expiry, find the most active one
            expiry_activity = group.groupby("days_to_expiry").agg(
                activity=("volume", "sum")
            )
            if expiry_activity.empty:
                continue
            best_dte = expiry_activity["activity"].idxmax()
            # For ATM/skew: use contracts near the most active expiry
            dte_range = max(10, best_dte * 0.5)
            active = group[
                (group["days_to_expiry"] >= max(1, best_dte - dte_range)) &
                (group["days_to_expiry"] <= best_dte + dte_range)
            ]
            if active.empty:
                active = group
        else:
            active = group

        # Use the underlying price from the most active slice (median)
        up = active["underlying_price"].median()

        row = {
            "product": product,
            "date": date,
            "underlying_price": up,
            "atm_iv": _atm_iv(active),
            "iv_skew": _iv_skew(active),
            "iv_term_slope": _iv_term_slope(group),  # full chain for term structure
            "oi_weighted_iv": _oi_weighted_iv(active),
            "put_call_oi_ratio": _put_call_oi_ratio(active),
            "total_option_volume": group["volume"].sum(),
            "total_option_oi": group["open_interest"].sum(),
            "n_contracts": len(group),
            "n_with_iv": group["implied_volatility"].notna().sum(),
        }
        results.append(row)

    metrics = pd.DataFrame(results)
    metrics.sort_values(["product", "date"], inplace=True)
    metrics.reset_index(drop=True, inplace=True)
    return metrics


# ---------------------------------------------------------------------------
# IV Percentile (rolling)
# ---------------------------------------------------------------------------

def compute_iv_percentile(metrics: pd.DataFrame,
                         window: int = IV_PERCENTILE_WINDOW) -> pd.DataFrame:
    """Add a rolling IV percentile column per product.

    The percentile represents where the current ATM IV sits within the
    recent lookback window.  With few data points (as in this dataset with
    only 3-5 dates) the percentile is of limited value but still computed.
    """
    if metrics.empty:
        return metrics

    results = []
    for product, group in metrics.groupby("product"):
        group = group.sort_values("date").copy()
        iv_values = group["atm_iv"].values
        pcts = []
        for i, iv in enumerate(iv_values):
            if pd.isna(iv):
                pcts.append(np.nan)
                continue
            # Collect all non-NaN IVs up to current position
            historical = [v for v in iv_values[:i+1] if not np.isnan(v)]
            if len(historical) < 2:
                pcts.append(np.nan)
                continue
            # Percentile: fraction of values <= current
            rank = sum(1 for v in historical if v <= iv)
            pct = rank / len(historical) * 100
            pcts.append(pct)
        group["iv_percentile"] = pcts
        results.append(group)

    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# Signal Generation
# ---------------------------------------------------------------------------

def generate_signals(metrics: pd.DataFrame) -> pd.DataFrame:
    """Generate trading signals from daily IV metrics.

    Adds boolean flag columns and categorical regime columns.
    """
    df = metrics.copy()

    # --- IV extreme signals ---
    df["iv_extreme_high"] = df["iv_percentile"] >= IV_HIGH_PCT
    df["iv_extreme_low"] = df["iv_percentile"] <= IV_LOW_PCT

    # --- Skew extreme (using rolling percentile per product) ---
    skew_results = []
    for product, group in df.groupby("product"):
        group = group.sort_values("date").copy()
        skew_valid = group["iv_skew"].dropna()
        if len(skew_valid) < 10:
            group["skew_extreme"] = False
            skew_results.append(group)
            continue
        abs_skew = group["iv_skew"].abs()
        group["skew_percentile"] = abs_skew.expanding(min_periods=5).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
        )
        group["skew_extreme"] = group["skew_percentile"] >= SKEW_EXTREME_PCT
        skew_results.append(group)
    df = pd.concat(skew_results, ignore_index=True)

    # --- IV rising (consecutive days of rising ATM IV) ---
    rising_results = []
    for product, group in df.groupby("product"):
        group = group.sort_values("date").copy()
        iv_diff = group["atm_iv"].diff()
        # Count consecutive positive differences
        rising_streak = []
        streak = 0
        for val in iv_diff:
            if pd.notna(val) and val > 0:
                streak += 1
            else:
                streak = 0
            rising_streak.append(streak)
        group["iv_rising_days"] = rising_streak
        group["iv_rising"] = group["iv_rising_days"] >= IV_RISING_DAYS
        rising_results.append(group)
    df = pd.concat(rising_results, ignore_index=True)

    # --- Term structure signal ---
    # Positive iv_term_slope = near IV > far IV = backwardation = fear
    df["term_structure_signal"] = df["iv_term_slope"].apply(
        lambda x: "backwardation" if pd.notna(x) and x > 0.01
        else ("contango" if pd.notna(x) and x < -0.01 else "flat")
    )

    # --- Volatility regime ---
    def _vol_regime(row):
        pct = row["iv_percentile"]
        if pd.isna(pct):
            return "unknown"
        if pct >= 90:
            return "extreme"
        if pct >= 70:
            return "high"
        if pct >= 30:
            return "normal"
        return "low"

    df["vol_regime"] = df.apply(_vol_regime, axis=1)

    # --- Directional bias ---
    def _directional_bias(row):
        skew = row["iv_skew"]
        oi_ratio = row["put_call_oi_ratio"]
        if pd.isna(skew) and pd.isna(oi_ratio):
            return "neutral"
        score = 0.0
        if pd.notna(skew):
            # Positive skew = puts expensive = bearish sentiment
            score += -1.0 if skew > 0.05 else (1.0 if skew < -0.05 else 0.0)
        if pd.notna(oi_ratio):
            # High put/call OI ratio = bearish positioning
            score += -1.0 if oi_ratio > 1.5 else (1.0 if oi_ratio < 0.67 else 0.0)
        if score > 0.5:
            return "bullish"
        if score < -0.5:
            return "bearish"
        return "neutral"

    df["directional_bias"] = df.apply(_directional_bias, axis=1)

    # --- Entry filter ---
    df["entry_filter"] = df["vol_regime"] != "extreme"

    # --- Composite signal: a single-direction indicator combining all ---
    def _composite_signal(row):
        """Combine multiple signals into a single score [-3, +3]."""
        score = 0
        # IV extreme high -> expect mean reversion (short vol -> fade move)
        if row.get("iv_extreme_high"):
            score -= 1
        # IV extreme low -> expect expansion (uncertainty ahead)
        if row.get("iv_extreme_low"):
            score += 1
        # Rising IV -> growing uncertainty, often bearish for risky assets
        if row.get("iv_rising"):
            score -= 1
        # Backwardation -> fear
        if row.get("term_structure_signal") == "backwardation":
            score -= 1
        # Directional bias
        bias = row.get("directional_bias", "neutral")
        if bias == "bullish":
            score += 1
        elif bias == "bearish":
            score -= 1
        return score

    df["signal_score"] = df.apply(_composite_signal, axis=1)

    return df


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(output_dir: Path = OUTPUT_DIR,
                 products: Optional[list] = None) -> pd.DataFrame:
    """End-to-end pipeline: load -> metrics -> signals -> export.

    Parameters
    ----------
    output_dir : directory for CSV output
    products : optional list of product keys to filter (e.g. ['DCE_a', 'SHFE_cu'])

    Returns
    -------
    DataFrame with all signals
    """
    print("=" * 60)
    print("Options Volatility Signal Generation Pipeline")
    print("=" * 60)

    # 1. Load
    print("\n[1/4] Loading options data...")
    df = load_all_options()
    if df.empty:
        print("ERROR: No options data loaded.")
        return pd.DataFrame()

    if products:
        df = df[df["_product_key"].isin(products)]
    else:
        # Exclude financial-index options (not commodity futures)
        exclude = {"options", "HO", "IO", "MO"}
        df = df[~df["_product_key"].isin(exclude)]

    print(f"  Loaded {len(df):,} option contracts")
    print(f"  Products: {df['_product_key'].nunique()}")
    print(f"  Date range: {df['_date'].min().date()} to {df['_date'].max().date()}")
    print(f"  Contracts with valid IV: {df['implied_volatility'].notna().sum():,} "
          f"({df['implied_volatility'].notna().mean() * 100:.1f}%)")

    # 2. Compute daily metrics
    print("\n[2/4] Computing daily IV metrics...")
    metrics = compute_daily_metrics(df)
    print(f"  Generated metrics for {len(metrics):,} (product, date) pairs")

    # 3. IV percentile
    print("\n[3/4] Computing rolling IV percentiles...")
    metrics = compute_iv_percentile(metrics)

    # 4. Generate signals
    print("\n[4/4] Generating trading signals...")
    signals = generate_signals(metrics)

    # Summary
    print("\n" + "=" * 60)
    print("Signal Summary")
    print("=" * 60)
    vol_counts = signals["vol_regime"].value_counts()
    print(f"\nVolatility Regime Distribution:")
    for regime, count in vol_counts.items():
        print(f"  {regime:>10s}: {count:>4d}")

    bias_counts = signals["directional_bias"].value_counts()
    print(f"\nDirectional Bias Distribution:")
    for bias, count in bias_counts.items():
        print(f"  {bias:>10s}: {count:>4d}")

    print(f"\nIV Extreme High: {signals['iv_extreme_high'].sum()}")
    print(f"IV Extreme Low:  {signals['iv_extreme_low'].sum()}")
    print(f"IV Rising:       {signals['iv_rising'].sum()}")
    print(f"Skew Extreme:    {signals['skew_extreme'].sum()}")

    term_counts = signals["term_structure_signal"].value_counts()
    print(f"\nTerm Structure:")
    for ts, count in term_counts.items():
        print(f"  {ts:>15s}: {count:>4d}")

    # Export
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "options_volatility_signals.csv"
    signals.to_csv(out_path, index=False)
    print(f"\nSignals exported to: {out_path}")
    print(f"  {len(signals)} rows x {len(signals.columns)} columns")

    return signals


# ---------------------------------------------------------------------------
# Quick test / demo
# ---------------------------------------------------------------------------

def demo_single_product(product_key: str = "b"):
    """Load and display signals for a single product.

    Default product 'b' (DCE soybean meal) has good data coverage.
    Product keys use the bare product code (e.g. 'b', 'au', 'AP', 'CF').
    """
    print(f"\n{'=' * 60}")
    print(f"Demo: {product_key}")
    print(f"{'=' * 60}")

    df = load_all_options()
    df = df[df["_product_key"] == product_key]

    if df.empty:
        print(f"No data found for {product_key}")
        return

    print(f"\nLoaded {len(df)} contracts across {df['_date'].nunique()} dates")

    metrics = compute_daily_metrics(df)
    metrics = compute_iv_percentile(metrics)
    signals = generate_signals(metrics)

    # Display key columns
    display_cols = [
        "date", "underlying_price", "atm_iv", "iv_percentile",
        "iv_skew", "iv_term_slope", "vol_regime", "directional_bias",
        "iv_extreme_high", "iv_extreme_low", "iv_rising",
        "term_structure_signal", "signal_score", "entry_filter",
    ]
    existing = [c for c in display_cols if c in signals.columns]
    print("\nSample signals (last 10 dates):")
    print(signals[existing].tail(10).to_string(index=False))

    return signals


if __name__ == "__main__":
    # Run full pipeline
    signals = run_pipeline()

    # Demo a single product with good data coverage
    demo_single_product("b")    # DCE soybean meal
    demo_single_product("au")   # SHFE gold
    demo_single_product("AP")   # CZCE apple
