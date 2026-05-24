#!/usr/bin/env python3
"""
Term Structure FACTOR MINING for Chinese Commodity Futures
==========================================================
Computes term structure factors and tests their predictive power via IC (Information Coefficient).

Output:
  - Factor ranking table sorted by ICIR
  - Per-factor decile analysis
  - Factor correlation matrix
"""

import json, os, sys, warnings, time
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 250)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

BASE = Path('/Users/chengming/home/futures_platform')
TS_DIR = BASE / 'data' / 'futures_term_structure'
FD_DIR = BASE / 'data' / 'futures_weighted'


def normalize_symbol(sym):
    """Normalize symbol name: lowercase, strip 'fi' suffix, so 'agfi' -> 'ag', 'AG' -> 'ag'"""
    s = sym.lower().strip()
    if s.endswith('fi'):
        s = s[:-2]
    return s


# ── 1. Load all term structure JSON files ──────────────────────────────────────
def load_term_structure():
    print("=" * 80)
    print("STEP 1: Loading term structure JSON files ...")
    t0 = time.time()

    rows = []
    files = sorted(TS_DIR.glob('*.json'))
    for i, fp in enumerate(files):
        if i % 10000 == 0 and i > 0:
            print(f"  ... loaded {i}/{len(files)} files")
        try:
            with open(fp) as f:
                d = json.load(f)

            curve = d.get('curve', [])
            # Extract prices and volumes for first few contracts
            prices = []
            volumes = []
            holds = []
            for c in curve:
                p = c.get('price') or c.get('close')
                v = c.get('volume', 0) or 0
                h = c.get('hold', 0) or c.get('oi', 0) or 0
                if p is not None and p > 0:
                    prices.append(p)
                    volumes.append(v)
                    holds.append(h)

            row = {
                'symbol': normalize_symbol(d['symbol']),
                'date': d['date'],
                'structure': d.get('structure', ''),
                'near_price': d.get('near_price'),
                'far_price': d.get('far_price'),
                'total_spread': d.get('total_spread'),
                'total_spread_pct': d.get('total_spread_pct'),
                'n_contracts': len(prices),
                'p1': prices[0] if len(prices) >= 1 else None,
                'p2': prices[1] if len(prices) >= 2 else None,
                'p3': prices[2] if len(prices) >= 3 else None,
                'v1': volumes[0] if len(volumes) >= 1 else 0,
                'v2': volumes[1] if len(volumes) >= 2 else 0,
                'v_far': volumes[-1] if len(volumes) >= 2 else 0,
                'h1': holds[0] if len(holds) >= 1 else 0,
                'h2': holds[1] if len(holds) >= 2 else 0,
                'h_far': holds[-1] if len(holds) >= 2 else 0,
            }

            # mid price (3rd contract from end or middle of curve)
            if len(prices) >= 3:
                mid_idx = len(prices) // 2
                row['p_mid'] = prices[mid_idx]
                row['v_mid'] = volumes[mid_idx]
                row['h_mid'] = holds[mid_idx]
            else:
                row['p_mid'] = None
                row['v_mid'] = None
                row['h_mid'] = None

            rows.append(row)
        except Exception as e:
            continue

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    # Remove garbage symbols
    df = df[df['symbol'].str.len() >= 2]

    print(f"  Loaded {len(rows)} records from {len(files)} files")
    print(f"  Symbols: {df['symbol'].nunique()}")
    print(f"  Date range: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  Time: {time.time()-t0:.1f}s")
    return df


# ── 2. Compute factors ────────────────────────────────────────────────────────
def compute_factors(df):
    print("\n" + "=" * 80)
    print("STEP 2: Computing term structure factors ...")
    t0 = time.time()

    # --- Factor 1: Roll Yield (展期收益率) ---
    # total_spread_pct is already (far-near)/near or similar; recompute consistently
    df['roll_yield'] = np.where(
        df['near_price'].notna() & df['far_price'].notna() & (df['near_price'] != 0),
        (df['near_price'] - df['far_price']) / df['near_price'] * 100,
        np.nan
    )

    # --- Factor 2: Curve Slope (曲线斜率) ---
    df['curve_slope'] = np.where(
        df['p1'].notna() & df['p2'].notna() & (df['p1'] != 0),
        (df['p2'] - df['p1']) / df['p1'] * 100,
        np.nan
    )

    # --- Factor 3: Curvature (曲率) ---
    df['curvature'] = np.where(
        df['p1'].notna() & df['p2'].notna() & df['p3'].notna(),
        df['p2'] - (df['p1'] + df['p3']) / 2,
        np.nan
    )
    # Normalize curvature by near price
    df['curvature_norm'] = np.where(
        df['curvature'].notna() & (df['p1'] != 0),
        df['curvature'] / df['p1'] * 100,
        np.nan
    )

    # --- Factor 4: Structure State (结构状态) ---
    df['struct_state'] = np.where(df['structure'] == 'backwardation', 1,
                         np.where(df['structure'] == 'contango', -1, 0))

    # --- Factor 5: Structure Flip (结构转换) ---
    df = df.sort_values(['symbol', 'date'])
    df['prev_structure'] = df.groupby('symbol')['struct_state'].shift(1)
    df['struct_flip'] = (df['struct_state'] != df['prev_structure']).astype(int)
    df.loc[df['prev_structure'].isna(), 'struct_flip'] = np.nan

    # --- Factor 6: Flip Frequency (结构转换频率) ---
    df['flip_freq_20d'] = df.groupby('symbol')['struct_flip'].transform(
        lambda x: x.rolling(20, min_periods=5).sum()
    )
    df['flip_freq_60d'] = df.groupby('symbol')['struct_flip'].transform(
        lambda x: x.rolling(60, min_periods=15).sum()
    )

    # --- Factor 7: Carry Change (Carry变化) ---
    df['carry_change_5d'] = df.groupby('symbol')['roll_yield'].transform(
        lambda x: x.diff(5)
    )
    df['carry_change_20d'] = df.groupby('symbol')['roll_yield'].transform(
        lambda x: x.diff(20)
    )

    # --- Factor 8: Near Premium (近月溢价) ---
    df['near_premium'] = np.where(
        df['p1'].notna() & df['p2'].notna() & (df['p1'] != 0),
        (df['p1'] - df['p2']) / df['p1'] * 100,
        np.nan
    )

    # --- Factor 9: Curve Convexity (曲线凹凸性) ---
    df['curve_convexity'] = np.where(
        df['p1'].notna() & df['p_mid'].notna() & df['far_price'].notna() & (df['p1'] != 0),
        (df['far_price'] - 2 * df['p_mid'] + df['p1']) / df['p1'] * 100,
        np.nan
    )

    # --- Factor 10: Volume Ratio (liquidity concentration) ---
    df['vol_ratio'] = np.where(
        (df['v1'].notna()) & (df['v_far'].notna()) & (df['v_far'] != 0),
        df['v1'] / df['v_far'],
        np.nan
    )
    df['vol_ratio'] = df['vol_ratio'].clip(upper=100)

    # --- Factor 11: OI Ratio ---
    df['oi_ratio'] = np.where(
        (df['h_far'].notna()) & (df['h_far'] != 0),
        df['h1'] / df['h_far'],
        np.nan
    )
    df['oi_ratio'] = df['oi_ratio'].clip(upper=100)

    # --- Additional factors ---
    df['roll_yield_zscore'] = df.groupby('symbol')['roll_yield'].transform(
        lambda x: (x - x.rolling(60, min_periods=20).mean()) / x.rolling(60, min_periods=20).std()
    )

    df['carry_mom_5d'] = df.groupby('symbol')['roll_yield'].transform(
        lambda x: x.diff(5)
    )
    df['carry_mom_20d'] = df.groupby('symbol')['roll_yield'].transform(
        lambda x: x.diff(20)
    )

    factor_cols = [
        'roll_yield', 'curve_slope', 'curvature_norm',
        'struct_state', 'struct_flip', 'flip_freq_20d', 'flip_freq_60d',
        'carry_change_5d', 'carry_change_20d', 'near_premium',
        'curve_convexity', 'vol_ratio', 'oi_ratio',
        'roll_yield_zscore', 'carry_mom_5d', 'carry_mom_20d'
    ]

    print(f"  Computed {len(factor_cols)} factors")
    print(f"  Time: {time.time()-t0:.1f}s")
    return df, factor_cols


# ── 3. Load futures daily & merge forward returns ─────────────────────────────
def load_futures_daily_and_merge(df):
    print("\n" + "=" * 80)
    print("STEP 3: Loading futures daily data & computing forward returns ...")
    t0 = time.time()

    daily_list = []
    for fp in sorted(FD_DIR.glob('*.csv')):
        try:
            d = pd.read_csv(fp)
            d['symbol'] = d['ts_code'].apply(normalize_symbol)

            # Handle mixed date formats: some are YYYYMMDD, some are YYYY-MM-DD
            def parse_date(s):
                s = str(s).strip()
                if '-' in s:
                    return pd.Timestamp(s)
                elif len(s) == 8 and s.isdigit():
                    return pd.Timestamp(f'{s[:4]}-{s[4:6]}-{s[6:8]}')
                else:
                    return pd.NaT
            d['trade_date'] = d['trade_date'].apply(parse_date)
            d = d.dropna(subset=['trade_date'])

            # Filter out garbage rows (close=0)
            d = d[d['close'] > 0]

            daily_list.append(d[['symbol', 'trade_date', 'close', 'vol', 'oi']])
        except Exception as e:
            continue

    daily = pd.concat(daily_list, ignore_index=True)
    daily = daily.sort_values(['symbol', 'trade_date']).reset_index(drop=True)

    print(f"  Loaded daily data: {len(daily)} rows, {daily['symbol'].nunique()} symbols")
    print(f"  Daily date range: {daily['trade_date'].min()} ~ {daily['trade_date'].max()}")
    print(f"  Daily symbols sample: {sorted(daily['symbol'].unique())[:10]}")

    # Compute forward returns
    for n in [1, 3, 5, 10, 20]:
        daily[f'fwd_{n}d'] = daily.groupby('symbol')['close'].transform(
            lambda x: x.shift(-n) / x - 1
        )

    # Merge with term structure factors
    # Only keep rows that overlap with term structure dates
    ts_date_min = df['date'].min()
    ts_date_max = df['date'].max()

    merge_df = daily[
        (daily['trade_date'] >= ts_date_min) & (daily['trade_date'] <= ts_date_max)
    ].copy()

    print(f"  Daily data in TS date range: {len(merge_df)} rows")

    df = df.merge(
        merge_df[['symbol', 'trade_date', 'fwd_1d', 'fwd_3d', 'fwd_5d', 'fwd_10d', 'fwd_20d']],
        left_on=['symbol', 'date'],
        right_on=['symbol', 'trade_date'],
        how='left'
    )
    df.drop(columns=['trade_date'], inplace=True, errors='ignore')

    # Also get OI from daily for OI-weighted carry
    df = df.merge(
        merge_df[['symbol', 'trade_date', 'oi']].rename(columns={'oi': 'daily_oi'}),
        left_on=['symbol', 'date'],
        right_on=['symbol', 'trade_date'],
        how='left'
    )
    df.drop(columns=['trade_date'], inplace=True, errors='ignore')

    valid = df['fwd_5d'].notna().sum()
    print(f"  Merged: {len(df)} rows, {valid} with valid fwd_5d")

    # Diagnostic: check symbol overlap
    ts_syms = set(df['symbol'].unique())
    daily_syms = set(daily['symbol'].unique())
    overlap = ts_syms & daily_syms
    print(f"  Symbol overlap: {len(overlap)} / TS:{len(ts_syms)} / Daily:{len(daily_syms)}")
    if len(overlap) < len(ts_syms):
        missing = ts_syms - daily_syms
        if len(missing) <= 20:
            print(f"  Missing in daily: {sorted(missing)}")

    print(f"  Time: {time.time()-t0:.1f}s")
    return df


# ── 4. IC Test (Information Coefficient) ──────────────────────────────────────
def compute_ic_analysis(df, factor_cols, return_cols=['fwd_1d', 'fwd_3d', 'fwd_5d', 'fwd_10d', 'fwd_20d']):
    print("\n" + "=" * 80)
    print("STEP 4: Computing IC (Information Coefficient) ...")
    t0 = time.time()

    # Need cross-sectional data: at least 10 symbols per day for meaningful IC
    # Filter to rows with valid forward returns
    df_ic = df[df['fwd_5d'].notna()].copy()
    date_counts = df_ic.groupby('date')['symbol'].nunique()
    valid_dates = date_counts[date_counts >= 10].index
    df_ic = df_ic[df_ic['date'].isin(valid_dates)].copy()
    print(f"  Valid dates (>=10 symbols with fwd_5d): {len(valid_dates)}")
    print(f"  Total observations for IC: {len(df_ic)}")

    if len(valid_dates) < 10:
        print("  WARNING: Very few valid dates. IC may not be reliable.")

    # Also compute IC for the full panel (pooled cross-section) as backup
    results = {}
    for fi, factor in enumerate(factor_cols):
        if factor not in df_ic.columns:
            continue

        if fi % 5 == 0:
            print(f"  Processing factor {fi+1}/{len(factor_cols)}: {factor}")

        ic_data = {}
        for ret_col in return_cols:
            # Daily cross-sectional Spearman IC
            daily_ics = []
            for date, group in df_ic.groupby('date'):
                valid = group[[factor, ret_col]].dropna()
                if len(valid) < 8:
                    continue
                try:
                    ic_val, _ = stats.spearmanr(valid[factor], valid[ret_col])
                    if not np.isnan(ic_val):
                        daily_ics.append({'date': date, 'ic': ic_val})
                except:
                    continue

            if len(daily_ics) < 20:
                ic_data[ret_col] = {'ic_mean': np.nan, 'ic_std': np.nan, 'icir': np.nan,
                                     't_stat': np.nan, 'hit_rate': np.nan, 'n_days': len(daily_ics)}
                continue

            ic_series = pd.DataFrame(daily_ics).set_index('date')['ic']
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            n = len(ic_series)
            icir = ic_mean / ic_std if ic_std != 0 else 0
            t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std != 0 else 0
            hit_rate = (ic_series > 0).mean()

            ic_data[ret_col] = {
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'icir': icir,
                't_stat': t_stat,
                'hit_rate': hit_rate,
                'n_days': n
            }

        results[factor] = ic_data

    print(f"  Time: {time.time()-t0:.1f}s")
    return results


# ── 5. Print Factor Ranking Table ─────────────────────────────────────────────
def print_factor_ranking(ic_results):
    print("\n" + "=" * 80)
    print("FACTOR RANKING TABLE (sorted by |ICIR(5d)|)")
    print("=" * 80)

    # Build table
    rows = []
    for factor, ret_data in ic_results.items():
        row = {
            'Factor': factor,
        }
        has_data = False
        for ret_col in ['fwd_1d', 'fwd_3d', 'fwd_5d', 'fwd_10d', 'fwd_20d']:
            if ret_col in ret_data and not np.isnan(ret_data[ret_col].get('ic_mean', np.nan)):
                row[f'IC({ret_col})'] = ret_data[ret_col]['ic_mean']
                row[f'ICIR({ret_col})'] = ret_data[ret_col]['icir']
                row[f't({ret_col})'] = ret_data[ret_col]['t_stat']
                has_data = True
            else:
                row[f'IC({ret_col})'] = np.nan
                row[f'ICIR({ret_col})'] = np.nan
                row[f't({ret_col})'] = np.nan

        if 'fwd_5d' in ret_data and not np.isnan(ret_data['fwd_5d'].get('hit_rate', np.nan)):
            row['HitRate(5d)'] = ret_data['fwd_5d']['hit_rate']
            row['N_days'] = ret_data['fwd_5d']['n_days']
        else:
            row['HitRate(5d)'] = np.nan
            row['N_days'] = 0

        if has_data:
            rows.append(row)

    if not rows:
        print("  No valid IC results to display.")
        return pd.DataFrame()

    table = pd.DataFrame(rows)
    table = table.sort_values('ICIR(fwd_5d)', key=lambda x: x.abs(), ascending=False).reset_index(drop=True)

    # Print formatted table
    header = (f"{'Factor':<25} {'IC(1d)':>8} {'IC(3d)':>8} {'IC(5d)':>8} {'IC(10d)':>8} {'IC(20d)':>8}"
              f"  {'ICIR(5d)':>9} {'ICIR(10d)':>10} {'ICIR(20d)':>10}  {'HitRate(5d)':>11} {'t(5d)':>8} {'N':>5}")
    sep = "-" * len(header)
    print(header)
    print(sep)

    for _, r in table.iterrows():
        ic1 = f"{r['IC(fwd_1d)']:>8.4f}" if not np.isnan(r.get('IC(fwd_1d)', np.nan)) else f"{'':>8}"
        ic3 = f"{r['IC(fwd_3d)']:>8.4f}" if not np.isnan(r.get('IC(fwd_3d)', np.nan)) else f"{'':>8}"
        ic5 = f"{r['IC(fwd_5d)']:>8.4f}" if not np.isnan(r.get('IC(fwd_5d)', np.nan)) else f"{'':>8}"
        ic10 = f"{r['IC(fwd_10d)']:>8.4f}" if not np.isnan(r.get('IC(fwd_10d)', np.nan)) else f"{'':>8}"
        ic20 = f"{r['IC(fwd_20d)']:>8.4f}" if not np.isnan(r.get('IC(fwd_20d)', np.nan)) else f"{'':>8}"
        icir5 = f"{r['ICIR(fwd_5d)']:>9.3f}" if not np.isnan(r.get('ICIR(fwd_5d)', np.nan)) else f"{'':>9}"
        icir10 = f"{r['ICIR(fwd_10d)']:>10.3f}" if not np.isnan(r.get('ICIR(fwd_10d)', np.nan)) else f"{'':>10}"
        icir20 = f"{r['ICIR(fwd_20d)']:>10.3f}" if not np.isnan(r.get('ICIR(fwd_20d)', np.nan)) else f"{'':>10}"
        hr = f"{r['HitRate(5d)']:>10.1%}" if not np.isnan(r.get('HitRate(5d)', np.nan)) else f"{'':>10}"
        t5 = f"{r['t(fwd_5d)']:>8.2f}" if not np.isnan(r.get('t(fwd_5d)', np.nan)) else f"{'':>8}"
        nd = f"{int(r['N_days']):>5}" if r['N_days'] > 0 else f"{'':>5}"
        print(f"{r['Factor']:<25} {ic1} {ic3} {ic5} {ic10} {ic20}  {icir5} {icir10} {icir20}  {hr} {t5} {nd}")

    return table


# ── 6. Decile Analysis ───────────────────────────────────────────────────────
def decile_analysis(df, top_factors, ret_col='fwd_5d'):
    print("\n" + "=" * 80)
    print(f"DECILE ANALYSIS (Top {len(top_factors)} factors, target={ret_col})")
    print("=" * 80)

    for factor in top_factors:
        print(f"\n--- {factor} ---")
        if factor not in df.columns:
            print("  Factor not in dataframe")
            continue

        valid = df[['symbol', 'date', factor, ret_col]].dropna()

        if len(valid) < 100:
            print(f"  Insufficient data ({len(valid)} rows)")
            continue

        # Cross-sectional decile assignment per date
        valid = valid.copy()
        try:
            valid['decile'] = valid.groupby('date')[factor].transform(
                lambda x: pd.qcut(x.rank(method='first'), 10, labels=False, duplicates='drop') + 1
            )
        except Exception as e:
            print(f"  Decile assignment failed: {e}")
            continue

        decile_stats = valid.groupby('decile').agg(
            count=(ret_col, 'count'),
            avg_ret=(ret_col, 'mean'),
            med_ret=(ret_col, 'median'),
            win_rate=(ret_col, lambda x: (x > 0).mean()),
            avg_factor=(factor, 'mean'),
        )

        print(f"  {'Decile':>6} {'Count':>8} {'AvgRet':>10} {'MedRet':>10} {'WinRate':>10} {'AvgFactor':>12}")
        for d, row in decile_stats.iterrows():
            print(f"  {int(d):>6} {int(row['count']):>8} {row['avg_ret']:>10.4%} {row['med_ret']:>10.4%} {row['win_rate']:>10.1%} {row['avg_factor']:>12.4f}")

        # Long-short: decile 10 - decile 1
        d1 = valid[valid['decile'] == 1].groupby('date')[ret_col].mean()
        d10 = valid[valid['decile'] == 10].groupby('date')[ret_col].mean()
        common_dates = d1.index.intersection(d10.index)
        if len(common_dates) > 20:
            ls_ret = d10.loc[common_dates] - d1.loc[common_dates]
            ann_ret = ls_ret.mean() * 252
            sharpe = ls_ret.mean() / ls_ret.std() * np.sqrt(252) if ls_ret.std() > 0 else 0
            print(f"  Long-Short (D10-D1): mean={ls_ret.mean():.4%}, ann={ann_ret:.2%}, Sharpe~={sharpe:.3f}, N_days={len(common_dates)}")


# ── 7. Factor Correlation Matrix ──────────────────────────────────────────────
def factor_correlation(df, top_factors):
    print("\n" + "=" * 80)
    print("FACTOR CORRELATION MATRIX (top factors, panel-level)")
    print("=" * 80)

    valid_factors = [f for f in top_factors if f in df.columns]
    if len(valid_factors) < 2:
        print("  Not enough factors for correlation")
        return

    corr = df[valid_factors].corr()
    print(corr.round(3).to_string())

    # Identify highly correlated pairs
    print("\n  Highly correlated pairs (|corr| > 0.6):")
    found = False
    for i in range(len(valid_factors)):
        for j in range(i+1, len(valid_factors)):
            c = corr.iloc[i, j]
            if abs(c) > 0.6:
                print(f"    {valid_factors[i]:<25} <-> {valid_factors[j]:<25} : {c:.3f}")
                found = True
    if not found:
        print("    None found - factors are largely independent")


# ── 8. Top factors summary ────────────────────────────────────────────────────
def top_factors_summary(ic_results, n=5):
    print("\n" + "=" * 80)
    print(f"TOP {n} FACTORS by |ICIR| across all horizons")
    print("=" * 80)

    factor_best = {}
    for factor, ret_data in ic_results.items():
        best_icir = 0
        best_period = None
        for period, vals in ret_data.items():
            if not np.isnan(vals.get('icir', np.nan)):
                if abs(vals['icir']) > abs(best_icir):
                    best_icir = vals['icir']
                    best_period = period
        if best_period is not None:
            factor_best[factor] = {
                'best_icir': best_icir,
                'best_period': best_period,
                'best_ic': ret_data[best_period]['ic_mean'],
                'best_t': ret_data[best_period]['t_stat'],
                'best_hit': ret_data[best_period]['hit_rate'],
            }

    ranked = sorted(factor_best.items(), key=lambda x: abs(x[1]['best_icir']), reverse=True)

    for rank, (factor, data) in enumerate(ranked[:n], 1):
        print(f"\n  #{rank}: {factor}")
        print(f"       Best horizon: {data['best_period']}, ICIR={data['best_icir']:.3f}, "
              f"IC={data['best_ic']:.4f}, t={data['best_t']:.2f}, HitRate={data['best_hit']:.1%}")

        all_horizons = ic_results.get(factor, {})
        for period in ['fwd_1d', 'fwd_3d', 'fwd_5d', 'fwd_10d', 'fwd_20d']:
            if period in all_horizons and not np.isnan(all_horizons[period].get('icir', np.nan)):
                p = all_horizons[period]
                marker = " <<< BEST" if period == data['best_period'] else ""
                print(f"         {period}: IC={p['ic_mean']:.4f}, ICIR={p['icir']:.3f}, "
                      f"t={p['t_stat']:.2f}, Hit={p['hit_rate']:.1%}{marker}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    total_t0 = time.time()

    # Step 1: Load term structure
    df = load_term_structure()

    # Step 2: Compute factors
    df, factor_cols = compute_factors(df)

    # Step 3: Merge with futures daily + forward returns
    df = load_futures_daily_and_merge(df)

    # Data summary
    print(f"\n  Final dataset: {len(df)} rows, {df['symbol'].nunique()} symbols")
    print(f"  Date range: {df['date'].min().date()} ~ {df['date'].max().date()}")
    for ret in ['fwd_1d', 'fwd_3d', 'fwd_5d', 'fwd_10d', 'fwd_20d']:
        n_valid = df[ret].notna().sum()
        n_syms = df[df[ret].notna()]['symbol'].nunique()
        print(f"  {ret}: {n_valid} valid observations ({n_syms} symbols)")

    # Step 4: IC analysis
    ic_results = compute_ic_analysis(df, factor_cols)

    # Step 5: Factor ranking table
    ranking_table = print_factor_ranking(ic_results)

    if len(ranking_table) > 0:
        top_n = min(8, len(ranking_table))
        top_factor_names = ranking_table.head(top_n)['Factor'].tolist()

        # Step 6: Decile analysis
        decile_analysis(df, top_factor_names, ret_col='fwd_5d')

        # Step 7: Factor correlation
        factor_correlation(df, top_factor_names)

        # Step 8: Top factors summary
        top_factors_summary(ic_results, n=5)
    else:
        print("\n  WARNING: No valid IC results produced.")
        print("  Diagnosing data issues ...")
        # Show sample of what we have
        sample = df[df['fwd_5d'].notna()].head(20)
        print(sample[['symbol', 'date', 'roll_yield', 'fwd_5d']].to_string())

    print("\n" + "=" * 80)
    print(f"TOTAL TIME: {time.time()-total_t0:.1f}s")
    print("=" * 80)


if __name__ == '__main__':
    main()
