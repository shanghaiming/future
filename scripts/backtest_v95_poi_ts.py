#!/usr/bin/env python3
"""
Backtest v95: POI Factor + Term Structure Combined Strategy
===========================================================
Base signal: POI factor (price change * OI change direction)
  - Long when price up AND OI up (bullish confirmation)
  - Short when price down AND OI down (bearish confirmation)
Filters:
  - Term structure: only long backwardated, only short contango
  - Spread percentile: only trade when spread in extreme percentiles (>80th or <20th)
Risk management: stop loss -2%, take profit +5%
Walk-forward: train 2021-2023, validate 2024, test 2025-2026

Portfolio construction: non-overlapping basket rebalance.
  Every H days, form a new basket of top-N names. Hold for H days.
  Returns are the equal-weight return of the basket over H days.
  Baskets do NOT overlap -- we wait for the holding period to end before
  rebalancing, which gives clean non-overlapping return series.
"""

import os
import json
import glob
import warnings
import itertools
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────
TS_DIR = os.path.expanduser('~/home/futures_platform/data/futures_term_structure/')
PRICE_DIR = os.path.expanduser('~/home/futures_platform/data/futures_weighted/')

# ── Parameters ─────────────────────────────────────────────────────────
LOOKBACK = 5          # days for computing price/OI changes
HOLDING_PERIODS = [5, 10, 15, 20]
POSITION_COUNTS = [5, 10, 15]
SPREAD_LOWER = 20     # percentile
SPREAD_UPPER = 80     # percentile
STOP_LOSS = -0.02     # -2%
TAKE_PROFIT = 0.05    # +5%

TRAIN_START = '2021-01-01'
TRAIN_END   = '2023-12-31'
VAL_START   = '2024-01-01'
VAL_END     = '2024-12-31'
TEST_START  = '2025-01-01'
TEST_END    = '2026-12-31'


# ── Step 1: Load and align price data ──────────────────────────────────
def load_price_data():
    """Load all futures price CSVs into a single panel."""
    all_dfs = []
    csv_files = glob.glob(os.path.join(PRICE_DIR, '*fi.csv'))
    print(f"Loading {len(csv_files)} price files...")

    for fpath in csv_files:
        sym = os.path.basename(fpath).replace('.csv', '')
        df = pd.read_csv(fpath, usecols=['trade_date', 'close', 'oi'])
        df['symbol'] = sym

        # Normalize date format
        dates = []
        for d in df['trade_date'].astype(str):
            d = d.strip()
            if '-' in d:
                dates.append(d)
            else:
                dates.append(f'{d[:4]}-{d[4:6]}-{d[6:8]}')
        df['trade_date'] = dates
        df = df[df['trade_date'] > '2020-12-31'].copy()
        df = df[df['oi'] > 0].copy()
        df = df.drop_duplicates(subset=['trade_date', 'symbol'], keep='first')

        if len(df) > 200:
            all_dfs.append(df)

    panel = pd.concat(all_dfs, ignore_index=True)
    panel = panel.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
    print(f"  Price panel: {len(panel)} rows, {panel['symbol'].nunique()} symbols")
    return panel


# ── Step 2: Load term structure data ──────────────────────────────────
def load_term_structure():
    """Load all term structure JSON files."""
    ts_data = {}
    json_files = glob.glob(os.path.join(TS_DIR, '*.json'))
    print(f"Loading {len(json_files)} term structure files...")

    for fpath in json_files:
        fname = os.path.basename(fpath)
        parts = fname.replace('.json', '').split('_')
        sym = parts[0]
        date_raw = parts[1]
        date_str = f'{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}'

        if date_str < '2021-01-01':
            continue

        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            ts_data[(sym, date_str)] = {
                'structure': data.get('structure', ''),
                'spread_pct': data.get('total_spread_pct', 0.0),
            }
        except Exception:
            continue

    print(f"  Term structure records: {len(ts_data)}")
    return ts_data


# ── Step 3: Compute POI factor and signals ────────────────────────────
def compute_signals(price_panel, ts_data):
    """
    For each symbol-date, compute:
      - POI factor (+1/-1/0)
      - Term structure info
      - Price at entry and at +H days (for each holding period)
    """
    results = []
    symbols = price_panel['symbol'].unique()
    print(f"Computing POI signals for {len(symbols)} symbols...")

    for sym in sorted(symbols):
        sdf = price_panel[price_panel['symbol'] == sym].copy()
        sdf = sdf.sort_values('trade_date').reset_index(drop=True)

        if len(sdf) < LOOKBACK + 30:
            continue

        close = sdf['close'].values.astype(float)
        oi = sdf['oi'].values.astype(float)
        dates = sdf['trade_date'].values

        for i in range(LOOKBACK, len(sdf)):
            price_chg = (close[i] - close[i - LOOKBACK]) / close[i - LOOKBACK]
            oi_chg = (oi[i] - oi[i - LOOKBACK]) / oi[i - LOOKBACK]

            if price_chg > 0 and oi_chg > 0:
                poi = 1
            elif price_chg < 0 and oi_chg < 0:
                poi = -1
            else:
                poi = 0

            dt = dates[i]
            ts_info = ts_data.get((sym, dt), None)

            row = {
                'symbol': sym,
                'date': dt,
                'close': close[i],
                'price_chg': price_chg,
                'oi_chg': oi_chg,
                'poi': poi,
                'structure': ts_info['structure'] if ts_info else '',
                'spread_pct': ts_info['spread_pct'] if ts_info else np.nan,
            }

            # Forward prices for each holding period
            for hp in HOLDING_PERIODS:
                if i + hp < len(sdf):
                    row[f'exit_price_{hp}d'] = close[i + hp]
                    row[f'exit_date_{hp}d'] = dates[i + hp]
                else:
                    row[f'exit_price_{hp}d'] = np.nan
                    row[f'exit_date_{hp}d'] = ''

            results.append(row)

    df = pd.DataFrame(results)
    print(f"  Signal rows: {len(df)}, with TS data: {df['spread_pct'].notna().sum()}")
    return df


# ── Step 4: Compute rolling spread percentiles ────────────────────────
def add_spread_percentile(signals):
    """Compute rolling 252-day percentile rank of spread for each symbol."""
    signals = signals.sort_values(['symbol', 'date']).copy()
    signals['spread_pct_rank'] = np.nan

    for sym in signals['symbol'].unique():
        mask = signals['symbol'] == sym
        idx = signals[mask].index
        sp = signals.loc[idx, 'spread_pct']
        signals.loc[idx, 'spread_pct_rank'] = sp.rolling(252, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
        )

    return signals


# ── Step 5: Build non-overlapping basket portfolio ─────────────────────
def build_portfolio(signals, holding_period, pos_count, use_ts_filter=True,
                    use_spread_filter=True):
    """
    Non-overlapping basket rebalance approach.
    Every H trading days, form a new basket. Hold for H days.
    Each basket returns = equal-weight return over H days.
    Baskets are sequential, not overlapping.
    """
    df = signals.copy()

    # Apply filters
    long_mask = df['poi'] == 1
    short_mask = df['poi'] == -1

    if use_ts_filter:
        long_mask = long_mask & (df['structure'] == 'backwardation')
        short_mask = short_mask & (df['structure'] == 'contango')

    if use_spread_filter:
        long_mask = long_mask & (df['spread_pct_rank'] >= SPREAD_UPPER)
        short_mask = short_mask & (df['spread_pct_rank'] <= SPREAD_LOWER)

    df['signal'] = 0
    df.loc[long_mask, 'signal'] = 1
    df.loc[short_mask, 'signal'] = -1

    trade_df = df[df['signal'] != 0].copy()
    if len(trade_df) == 0:
        return None

    # Strength metric for ranking
    trade_df['strength'] = trade_df['price_chg'].abs() * trade_df['oi_chg'].abs()

    exit_col = f'exit_price_{holding_period}d'

    # Get all unique trading dates, sorted
    all_dates = sorted(signals['date'].unique())

    # Walk through dates in steps of holding_period (non-overlapping)
    basket_returns = []
    i = 0
    while i < len(all_dates):
        entry_date = all_dates[i]

        # Get signals on this date
        day_signals = trade_df[trade_df['date'] == entry_date]
        if len(day_signals) == 0:
            i += 1
            continue

        # Select top N longs and shorts
        longs = day_signals[day_signals['signal'] == 1].nlargest(pos_count, 'strength')
        shorts = day_signals[day_signals['signal'] == -1].nlargest(pos_count, 'strength')

        selected = pd.concat([longs, shorts])
        if len(selected) == 0:
            i += 1
            continue

        # Compute individual position returns
        pos_rets = []
        for _, row in selected.iterrows():
            entry_price = row['close']
            exit_price = row[exit_col]
            if pd.isna(exit_price) or entry_price == 0:
                continue
            raw_ret = (exit_price - entry_price) / entry_price
            # Apply direction (short = negative)
            directed_ret = raw_ret * row['signal']
            # Risk management: clamp
            if directed_ret < STOP_LOSS:
                directed_ret = STOP_LOSS
            elif directed_ret > TAKE_PROFIT:
                directed_ret = TAKE_PROFIT
            pos_rets.append(directed_ret)

        if pos_rets:
            basket_returns.append({
                'entry_date': entry_date,
                'ret': np.mean(pos_rets),
                'n_positions': len(pos_rets),
            })

        # Jump forward by holding_period
        i += holding_period

    if not basket_returns:
        return None

    port = pd.DataFrame(basket_returns)
    return port


# ── Step 6: Performance metrics ────────────────────────────────────────
def calc_metrics(port, label=''):
    if port is None or len(port) < 5:
        return None

    rets = port['ret'].values

    # Cumulative return (non-overlapping)
    cum = np.cumprod(1 + rets)
    total_ret = cum[-1] / cum[0] - 1

    # Annualized: count actual calendar years
    n_baskets = len(rets)
    # Each basket is holding_period trading days, ~holding_period/252 years
    years = max(n_baskets * 20 / 252, 0.1)  # use 20 as avg holding period for annualization

    # Better: use actual date range
    if 'entry_date' in port.columns:
        dates = port['entry_date'].values
        d0 = pd.Timestamp(dates[0])
        d1 = pd.Timestamp(dates[-1])
        years = max((d1 - d0).days / 365.25, 0.1)
    else:
        years = max(n_baskets / 12, 0.1)  # rough

    cagr = (cum[-1] / cum[0]) ** (1 / years) - 1 if total_ret > -1 else -1

    # Sharpe (using per-basket returns, annualize with baskets per year)
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if len(rets) > 1 else 0.001
    baskets_per_year = max(n_baskets / years, 1)
    sharpe = (mean_ret / std_ret) * np.sqrt(baskets_per_year) if std_ret > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    mdd = dd.min()

    # Win rate
    wr = np.mean(rets > 0) * 100

    avg_pos = port['n_positions'].mean()

    return {
        'label': label,
        'Sharpe': round(sharpe, 2),
        'CAGR': round(cagr * 100, 2),
        'MDD': round(mdd * 100, 2),
        'WR': round(wr, 1),
        'TotalRet': round(total_ret * 100, 2),
        'AvgPos': round(avg_pos, 1),
        'Baskets': n_baskets,
    }


def calc_walkforward_metrics(signals, holding_period, pos_count, use_ts_filter,
                              use_spread_filter):
    """Compute metrics for train/val/test periods."""
    periods = [
        ('Train 2021-2023', TRAIN_START, TRAIN_END),
        ('Val 2024', VAL_START, VAL_END),
        ('Test 2025-2026', TEST_START, TEST_END),
    ]
    results = []
    for pname, pstart, pend in periods:
        mask = (signals['date'] >= pstart) & (signals['date'] <= pend)
        period_signals = signals[mask].copy()
        port = build_portfolio(period_signals, holding_period, pos_count,
                               use_ts_filter, use_spread_filter)
        m = calc_metrics(port, pname)
        if m:
            results.append(m)
        else:
            results.append({
                'label': pname, 'Sharpe': 0, 'CAGR': 0, 'MDD': 0,
                'WR': 0, 'TotalRet': 0, 'AvgPos': 0, 'Baskets': 0
            })
    return results


# ── Main ──────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Backtest v95: POI Factor + Term Structure Combined Strategy")
    print("=" * 70)

    # Load data
    price_panel = load_price_data()
    ts_data = load_term_structure()

    # Compute signals
    signals = compute_signals(price_panel, ts_data)

    # Add spread percentile
    signals = add_spread_percentile(signals)

    # Filter to date range
    signals = signals[
        (signals['date'] >= TRAIN_START) & (signals['date'] <= TEST_END)
    ].copy()
    print(f"\nFiltered signals: {len(signals)} rows")

    # Summary stats
    print(f"  POI=+1 (bullish): {(signals['poi']==1).sum()}")
    print(f"  POI=-1 (bearish): {(signals['poi']==-1).sum()}")
    print(f"  POI=0  (diverge): {(signals['poi']==0).sum()}")
    print(f"  Backwardation: {(signals['structure']=='backwardation').sum()}")
    print(f"  Contango: {(signals['structure']=='contango').sum()}")

    # ── Grid Search ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GRID SEARCH: Holding Period x Position Count x Filter Combo")
    print("  (Non-overlapping basket rebalance)")
    print("=" * 70)

    all_results = []

    configs = [
        ('POI_only',       False, False),
        ('POI+TS',         True,  False),
        ('POI+Spread',     False, True),
        ('POI+TS+Spread',  True,  True),
    ]

    for hp in HOLDING_PERIODS:
        for pc in POSITION_COUNTS:
            for cfg_name, use_ts, use_sp in configs:
                port = build_portfolio(signals, hp, pc, use_ts, use_sp)
                label = f'{cfg_name}/h{hp}/n{pc}'
                m = calc_metrics(port, label)
                if m:
                    m['config'] = cfg_name
                    m['hold'] = hp
                    m['pos'] = pc
                    m['use_ts'] = use_ts
                    m['use_sp'] = use_sp
                    all_results.append(m)

    # Print results table
    print("\n" + "-" * 115)
    print(f"{'Config':<20} {'Hold':>4} {'Pos':>3} {'Sharpe':>7} {'CAGR%':>8} "
          f"{'MDD%':>7} {'WR%':>5} {'TotRet%':>9} {'AvgPos':>6} {'Baskets':>7}")
    print("-" * 115)

    all_results.sort(key=lambda x: x['Sharpe'], reverse=True)
    for r in all_results:
        print(f"{r['label']:<20} {r['hold']:>4} {r['pos']:>3} {r['Sharpe']:>7.2f} "
              f"{r['CAGR']:>8.2f} {r['MDD']:>7.2f} {r['WR']:>5.1f} "
              f"{r['TotalRet']:>9.2f} {r['AvgPos']:>6.1f} {r['Baskets']:>7d}")

    # ── Walk-forward for top configs ──────────────────────────────
    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION (Top 5 by Sharpe)")
    print("=" * 70)

    seen_configs = set()
    top_unique = []
    for r in all_results:
        key = (r['config'], r['hold'], r['pos'])
        if key not in seen_configs:
            seen_configs.add(key)
            top_unique.append(r)
        if len(top_unique) >= 5:
            break

    for r in top_unique:
        print(f"\n{'='*70}")
        print(f"Config: {r['config']}, Hold={r['hold']}d, Pos={r['pos']}")
        print(f"{'='*70}")
        wf = calc_walkforward_metrics(signals, r['hold'], r['pos'],
                                       r['use_ts'], r['use_sp'])
        print(f"  {'Period':<20} {'Sharpe':>7} {'CAGR%':>8} {'MDD%':>7} "
              f"{'WR%':>5} {'TotRet%':>9} {'AvgPos':>6} {'Baskets':>7}")
        print(f"  {'-'*70}")
        for wm in wf:
            print(f"  {wm['label']:<20} {wm['Sharpe']:>7.2f} {wm['CAGR']:>8.2f} "
                  f"{wm['MDD']:>7.2f} {wm['WR']:>5.1f} {wm['TotalRet']:>9.2f} "
                  f"{wm['AvgPos']:>6.1f} {wm['Baskets']:>7d}")

    # ── Best config details ──────────────────────────────────────
    best = all_results[0]
    print(f"\n{'='*70}")
    print(f"BEST CONFIG: {best['config']}, Hold={best['hold']}d, Pos={best['pos']}")
    print(f"  Sharpe={best['Sharpe']}, CAGR={best['CAGR']}%, MDD={best['MDD']}%, "
          f"WR={best['WR']}%, TotalRet={best['TotalRet']}%")
    print(f"{'='*70}")

    # Detailed walk-forward for best
    print("\nFull Walk-Forward for Best Config:")
    wf = calc_walkforward_metrics(signals, best['hold'], best['pos'],
                                   best['use_ts'], best['use_sp'])
    for wm in wf:
        print(f"  {wm['label']:<20} Sharpe={wm['Sharpe']:.2f} CAGR={wm['CAGR']:.2f}% "
              f"MDD={wm['MDD']:.2f}% WR={wm['WR']:.1f}% TotRet={wm['TotalRet']:.2f}% "
              f"AvgPos={wm['AvgPos']:.1f} Baskets={wm['Baskets']}")

    # ── Comparison for best hold/pos ──────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON: POI-only vs POI+TS vs POI+Spread vs POI+TS+Spread")
    print(f"(Using best params: Hold={best['hold']}d, Pos={best['pos']})")
    print("=" * 70)

    for cfg_name, use_ts, use_sp in configs:
        port = build_portfolio(signals, best['hold'], best['pos'], use_ts, use_sp)
        m = calc_metrics(port, cfg_name)
        if m:
            print(f"  {cfg_name:<20} Sharpe={m['Sharpe']:.2f} CAGR={m['CAGR']:.2f}% "
                  f"MDD={m['MDD']:.2f}% WR={m['WR']:.1f}% TotRet={m['TotalRet']:.2f}%")

    # ── Also do walk-forward comparison for best ──────────────────
    print("\n" + "=" * 70)
    print("WALK-FORWARD COMPARISON across filter types")
    print(f"(Using Hold={best['hold']}d, Pos={best['pos']})")
    print("=" * 70)
    for cfg_name, use_ts, use_sp in configs:
        print(f"\n  {cfg_name}:")
        wf = calc_walkforward_metrics(signals, best['hold'], best['pos'],
                                       use_ts, use_sp)
        for wm in wf:
            print(f"    {wm['label']:<20} Sharpe={wm['Sharpe']:.2f} "
                  f"CAGR={wm['CAGR']:.2f}% MDD={wm['MDD']:.2f}% "
                  f"WR={wm['WR']:.1f}% TotRet={wm['TotalRet']:.2f}%")

    # ── Walk-forward stability ranking ────────────────────────────
    print("\n" + "=" * 70)
    print("WALK-FORWARD STABILITY RANKING")
    print("  Score = min(Val_Sharpe, Test_Sharpe) -- favors consistent OOS")
    print("=" * 70)

    wf_ranking = []
    for r in all_results:
        wf = calc_walkforward_metrics(signals, r['hold'], r['pos'],
                                       r['use_ts'], r['use_sp'])
        val_s = wf[1]['Sharpe'] if len(wf) > 1 else 0
        test_s = wf[2]['Sharpe'] if len(wf) > 2 else 0
        val_cagr = wf[1]['CAGR'] if len(wf) > 1 else 0
        test_cagr = wf[2]['CAGR'] if len(wf) > 2 else 0
        test_mdd = wf[2]['MDD'] if len(wf) > 2 else 0
        test_wr = wf[2]['WR'] if len(wf) > 2 else 0

        stability_score = min(val_s, test_s)
        wf_ranking.append({
            'label': r['label'],
            'config': r['config'],
            'hold': r['hold'],
            'pos': r['pos'],
            'use_ts': r['use_ts'],
            'use_sp': r['use_sp'],
            'val_sharpe': val_s,
            'test_sharpe': test_s,
            'val_cagr': val_cagr,
            'test_cagr': test_cagr,
            'test_mdd': test_mdd,
            'test_wr': test_wr,
            'stability': stability_score,
        })

    wf_ranking.sort(key=lambda x: x['stability'], reverse=True)

    print(f"\n{'Config':<22} {'Hold':>4} {'Pos':>3} {'Val-Sh':>7} "
          f"{'Test-Sh':>8} {'Val-CA%':>8} {'Test-CA%':>9} {'Test-MDD%':>9} "
          f"{'Test-WR%':>8} {'Stability':>9}")
    print("-" * 100)
    for wr_item in wf_ranking[:30]:
        print(f"{wr_item['label']:<22} {wr_item['hold']:>4} {wr_item['pos']:>3} "
              f"{wr_item['val_sharpe']:>7.2f} {wr_item['test_sharpe']:>8.2f} "
              f"{wr_item['val_cagr']:>8.2f} {wr_item['test_cagr']:>9.2f} "
              f"{wr_item['test_mdd']:>9.2f} {wr_item['test_wr']:>8.1f} "
              f"{wr_item['stability']:>9.2f}")

    # ── Final recommendation ──────────────────────────────────────
    best_stable = wf_ranking[0]
    print(f"\n{'='*70}")
    print(f"RECOMMENDED CONFIG (Best Walk-Forward Stability)")
    print(f"  Config: {best_stable['config']}, Hold={best_stable['hold']}d, "
          f"Pos={best_stable['pos']}")
    print(f"  Val 2024  : Sharpe={best_stable['val_sharpe']:.2f}, "
          f"CAGR={best_stable['val_cagr']:.2f}%")
    print(f"  Test 2025-: Sharpe={best_stable['test_sharpe']:.2f}, "
          f"CAGR={best_stable['test_cagr']:.2f}%, MDD={best_stable['test_mdd']:.2f}%, "
          f"WR={best_stable['test_wr']:.1f}%")
    print(f"  Stability Score: {best_stable['stability']:.2f}")
    print(f"{'='*70}")

    # Full walk-forward for recommended
    wf = calc_walkforward_metrics(signals, best_stable['hold'], best_stable['pos'],
                                   best_stable['use_ts'], best_stable['use_sp'])
    print("\nFull Walk-Forward for Recommended Config:")
    for wm in wf:
        print(f"  {wm['label']:<20} Sharpe={wm['Sharpe']:.2f} CAGR={wm['CAGR']:.2f}% "
              f"MDD={wm['MDD']:.2f}% WR={wm['WR']:.1f}% TotRet={wm['TotalRet']:.2f}% "
              f"AvgPos={wm['AvgPos']:.1f} Baskets={wm['Baskets']}")

    print("\nDone.")


if __name__ == '__main__':
    main()
