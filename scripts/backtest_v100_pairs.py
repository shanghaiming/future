#!/usr/bin/env python3
"""
V100: Pairs Trading with Cointegration - Chinese Commodity Futures
===================================================================
Core idea: Trade mean-reverting spreads between cointegrated commodity pairs.

Strategy:
  1. Formation: 120-day rolling window Engle-Granger cointegration test
  2. Selection: pairs with p-value < 0.05
  3. Spread: OLS hedge ratio, z-score with 20-day rolling window
  4. Entry: z-score > threshold -> short outperformer + long underperformer
  5. Exit: z-score crosses 0, or max holding period reached
  6. Dynamic re-selection: re-run cointegration every 60 days

Test matrix:
  - Z-score thresholds: 1.5, 2.0, 2.5
  - Holding periods: 5d, 10d, 15d, 20d
  - Pair types: within-sector only, cross-sector, all
  - Position sizing: equal weight, hedge-ratio weighted

Risk: -2% SL per pair, +5% TP. Walk-forward: train 2021-2023, validate 2024, test 2025-2026.
"""
import os, sys, glob, time, warnings
import numpy as np
import pandas as pd
from itertools import combinations, product
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAILY_DIR = os.path.join(BASE_DIR, 'data', 'futures_weighted')

INITIAL_CAPITAL = 500_000
LEVERAGE = 3

# ──────────────────────────────────────────────────────────────────────
# SECTOR DEFINITIONS
# ──────────────────────────────────────────────────────────────────────
SECTORS = {
    'metals': ['cufi', 'alfi', 'znfi', 'pbfi', 'nifi', 'snfi', 'aufi', 'agfi'],
    'ferrous': ['rbfi', 'hcfi', 'jfi', 'jmfi', 'zcfi', 'sffi', 'ifi'],
    'energy':  ['scfi', 'fufi', 'bufi', 'pgfi', 'tafi'],
    'chemicals': ['mafi', 'egfi', 'fgfi', 'safi', 'ebfi', 'ppfi', 'lfi', 'vfi'],
    'agri': ['afi', 'bfi', 'mfi', 'yfi', 'pfi', 'cffi', 'srfi',
             'rifi', 'rmfi', 'rsfi', 'oifi'],
}

# Build symbol -> sector map
SYMBOL_SECTOR = {}
for sec, syms in SECTORS.items():
    for s in syms:
        SYMBOL_SECTOR[s] = sec


# ──────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────
def load_all_data():
    """Load daily price data for all commodities."""
    print("  Loading daily price data...")
    daily_data = {}
    for f in sorted(glob.glob(os.path.join(DAILY_DIR, '*.csv'))):
        sym = os.path.basename(f).replace('.csv', '')
        df = pd.read_csv(f)
        if len(df) < 200:
            continue
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        df = df.sort_values('trade_date').reset_index(drop=True)
        df = df[df['close'].notna() & (df['close'] > 0)].reset_index(drop=True)
        if len(df) < 100:
            continue
        df['ret_1d'] = df['close'].pct_change(1) * 100
        df['log_close'] = np.log(df['close'])
        daily_data[sym] = df

    print(f"    {len(daily_data)} symbols loaded")
    return daily_data


def build_aligned_matrix(daily_data, start_date='2019-01-01', end_date='2026-12-31'):
    """Build a unified price matrix aligned on trade dates."""
    all_dates = set()
    all_syms = []
    for sym, df in daily_data.items():
        sub = df[(df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)]
        if len(sub) < 100:
            continue
        all_dates |= set(sub['trade_date'].values)
        all_syms.append(sym)

    date_index = pd.DatetimeIndex(sorted(all_dates))
    close_mat = pd.DataFrame(np.nan, index=date_index, columns=all_syms)
    log_mat = pd.DataFrame(np.nan, index=date_index, columns=all_syms)
    ret_mat = pd.DataFrame(np.nan, index=date_index, columns=all_syms)

    for sym in all_syms:
        df = daily_data[sym]
        mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
        sub = df[mask].set_index('trade_date')
        close_mat.loc[sub.index, sym] = sub['close']
        log_mat.loc[sub.index, sym] = sub['log_close']
        ret_mat.loc[sub.index, sym] = sub['ret_1d']

    close_mat = close_mat.ffill().bfill()
    log_mat = log_mat.ffill().bfill()
    ret_mat = ret_mat.ffill().fillna(0)

    print(f"    Price matrix: {close_mat.shape[0]} dates x {close_mat.shape[1]} symbols")
    # Convert to numpy for speed
    log_arr = log_mat.values  # (T, N)
    ret_arr = ret_mat.values
    close_arr = close_mat.values
    return close_mat, log_mat, ret_mat, close_arr, log_arr, ret_arr


# ──────────────────────────────────────────────────────────────────────
# FAST COINTEGRATION TEST (numpy-based, no pandas overhead)
# ──────────────────────────────────────────────────────────────────────
def test_coint_fast(y_arr, x_arr):
    """
    Fast cointegration test using numpy.
    Returns (p_value, hedge_ratio, half_life).
    y_arr, x_arr are 1D numpy arrays (same length, no NaN).
    """
    n = len(y_arr)
    if n < 40:
        return 1.0, 0.0, 999

    try:
        # OLS: y = alpha + beta * x + eps
        ones = np.ones(n)
        X = np.column_stack([ones, x_arr])
        # Normal equation: beta = (X'X)^-1 X'y
        XtX = X.T @ X
        Xty = X.T @ y_arr
        beta = np.linalg.solve(XtX, Xty)
        hedge_ratio = beta[1]
        residuals = y_arr - X @ beta

        # Half-life from OU process: delta_resid = lambda * resid_lag + eps
        resid_lag = residuals[:-1]
        delta_resid = residuals[1:] - residuals[:-1]
        # OLS for half-life
        X_hl = np.column_stack([np.ones(n - 1), resid_lag])
        beta_hl = np.linalg.lstsq(X_hl, delta_resid, rcond=None)[0]
        lam = beta_hl[1]
        if lam >= -0.001:
            half_life = 999
        else:
            half_life = -np.log(2) / lam

        # Use statsmodels coint for proper p-value (faster than manual ADF)
        score, pvalue, crit = coint(y_arr, x_arr, autolag='AIC')
        return pvalue, hedge_ratio, half_life
    except Exception:
        return 1.0, 0.0, 999


# ──────────────────────────────────────────────────────────────────────
# PRECOMPUTE COINTEGRATION FOR ALL RE-SELECTION DATES
# ──────────────────────────────────────────────────────────────────────
def precompute_cointegration(log_arr, log_mat, ret_mat, symbols, sym_to_col,
                             formation=120, reselect_freq=60,
                             start_date=None, end_date=None,
                             pair_filter='all'):
    """
    Precompute cointegration results at each re-selection date.
    Returns dict: date_idx -> list of (sym_y, sym_x, p_val, hedge_ratio, half_life)
    """
    dates = log_mat.index
    if start_date:
        mask = dates >= pd.Timestamp(start_date)
        start_i = mask.argmax() if mask.any() else 0
    else:
        start_i = 0
    if end_date:
        mask = dates <= pd.Timestamp(end_date)
        end_i = len(dates) - 1 - mask[::-1].argmax() if mask.any() else len(dates) - 1
    else:
        end_i = len(dates) - 1

    # Build pair candidate indices
    col_indices = {s: sym_to_col[s] for s in symbols if s in sym_to_col}
    cand_pairs = []
    syms_list = list(col_indices.keys())
    for i in range(len(syms_list)):
        for j in range(i + 1, len(syms_list)):
            sa, sb = syms_list[i], syms_list[j]
            sec_a = SYMBOL_SECTOR.get(sa, 'unknown')
            sec_b = SYMBOL_SECTOR.get(sb, 'unknown')
            if pair_filter == 'within_sector':
                if sec_a == 'unknown' or sec_a != sec_b:
                    continue
            elif pair_filter == 'cross_sector':
                if sec_a == 'unknown' or sec_b == 'unknown' or sec_a == sec_b:
                    continue
            cand_pairs.append((sa, sb, col_indices[sa], col_indices[sb]))

    print(f"    Precomputing cointegration for {len(cand_pairs)} candidate pairs...")

    # Determine re-selection dates
    reselect_dates = list(range(start_i, end_i + 1, reselect_freq))
    if not reselect_dates or reselect_dates[0] != start_i:
        reselect_dates.insert(0, start_i)

    results = {}
    total_rs = len(reselect_dates)

    for rs_idx, date_idx in enumerate(reselect_dates):
        if date_idx < formation:
            continue
        f_start = date_idx - formation
        pairs_found = []

        for (sa, sb, ca, cb) in cand_pairs:
            y_p = log_arr[f_start:date_idx + 1, ca]
            x_p = log_arr[f_start:date_idx + 1, cb]
            # Check enough non-NaN overlap
            valid = np.isfinite(y_p) & np.isfinite(x_p)
            n_valid = valid.sum()
            if n_valid < formation * 0.7:
                continue
            y_v, x_v = y_p[valid], x_p[valid]
            p_val, hr, hl = test_coint_fast(y_v, x_v)
            if p_val < 0.05 and 3 <= hl <= 60:
                pairs_found.append((sa, sb, p_val, hr, hl))

        pairs_found.sort(key=lambda x: x[2])
        results[date_idx] = pairs_found

        if rs_idx % 5 == 0 or rs_idx == total_rs - 1:
            print(f"      [{rs_idx + 1}/{total_rs}] idx={date_idx} "
                  f"({dates[date_idx].strftime('%Y-%m-%d') if date_idx < len(dates) else 'end'}): "
                  f"{len(pairs_found)} cointegrated pairs")

    return results


# ──────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE (using precomputed cointegration)
# ──────────────────────────────────────────────────────────────────────
def run_pairs_backtest(log_mat, log_arr, ret_arr,
                       coint_precomputed, sym_to_col, col_to_sym,
                       formation=120, z_window=20,
                       z_entry=2.0, max_hold=20,
                       sl_pct=-2.0, tp_pct=5.0,
                       reselect_freq=60,
                       position_sizing='equal',
                       start=None, end=None,
                       max_pairs=10,
                       lev=LEVERAGE):
    """
    Pairs trading backtest using precomputed cointegration results.
    """
    dates = log_mat.index
    mask = pd.Series(True, index=dates)
    if start:
        mask &= (dates >= pd.Timestamp(start))
    if end:
        mask &= (dates <= pd.Timestamp(end))
    trade_dates = dates[mask]
    if len(trade_dates) == 0:
        return None, []

    capital = INITIAL_CAPITAL
    equity_curve = []
    all_trades = []
    active_pairs = []

    # Determine re-selection dates from precomputed keys
    sorted_rs_dates = sorted(coint_precomputed.keys())
    current_rs_idx = -1
    current_coint = []

    for dt in trade_dates:
        idx = dates.get_loc(dt)
        if idx < formation:
            continue

        daily_pnl = 0.0

        # ── Check if we need to re-select ──
        # Find the most recent re-selection date <= current idx
        new_rs = None
        for rs_d in sorted_rs_dates:
            if rs_d <= idx and rs_d > (current_rs_idx if current_rs_idx >= 0 else -1):
                new_rs = rs_d

        if new_rs is not None and new_rs != current_rs_idx:
            current_rs_idx = new_rs
            current_coint = coint_precomputed.get(new_rs, [])

        # ── Exit existing positions ──
        surviving = []
        for pos in active_pairs:
            ca, cb = pos['col_y'], pos['col_x']
            hr = pos['hedge_ratio']

            # Z-score
            z_start = max(0, idx - z_window)
            sp_slice = log_arr[z_start:idx + 1, ca] - hr * log_arr[z_start:idx + 1, cb]
            sp_mean = np.mean(sp_slice)
            sp_std = np.std(sp_slice)
            current_z = (sp_slice[-1] - sp_mean) / sp_std if sp_std > 1e-10 else 0

            # PnL
            ret_y = ret_arr[idx, ca]
            ret_x = ret_arr[idx, cb]
            if pos['direction'] == 1:
                pair_ret = -ret_y * pos['weight_y'] + ret_x * pos['weight_x']
            else:
                pair_ret = ret_y * pos['weight_y'] - ret_x * pos['weight_x']

            pos['cum_ret'] += pair_ret
            pos['hold_days'] += 1
            daily_pnl += pair_ret / 100 * pos['notional']

            # Exit checks
            exit_reason = None
            if pos['direction'] == 1 and current_z <= 0:
                exit_reason = 'z_cross'
            elif pos['direction'] == -1 and current_z >= 0:
                exit_reason = 'z_cross'
            if pos['hold_days'] >= max_hold:
                exit_reason = 'max_hold'
            if pos['cum_ret'] <= sl_pct:
                exit_reason = 'stop_loss'
            if pos['cum_ret'] >= tp_pct:
                exit_reason = 'take_profit'

            if exit_reason:
                sy, sx = pos['sym_y'], pos['sym_x']
                all_trades.append({
                    'sym_y': sy, 'sym_x': sx,
                    'sector_y': SYMBOL_SECTOR.get(sy, '?'),
                    'sector_x': SYMBOL_SECTOR.get(sx, '?'),
                    'direction': pos['direction'],
                    'entry_date': pos['entry_date'],
                    'exit_date': dt,
                    'hold_days': pos['hold_days'],
                    'cum_ret': pos['cum_ret'],
                    'exit_reason': exit_reason,
                    'hedge_ratio': hr,
                    'p_val': pos['p_val'],
                    'half_life': pos['half_life'],
                })
            else:
                surviving.append(pos)

        active_pairs = surviving

        # ── New entries ──
        n_active = len(active_pairs)
        if n_active < max_pairs and current_coint:
            used_syms = set()
            for p in active_pairs:
                used_syms.add(p['sym_y'])
                used_syms.add(p['sym_x'])

            for (sy, sx, pv, hr, hl) in current_coint:
                if n_active >= max_pairs:
                    break
                if sy in used_syms or sx in used_syms:
                    continue
                ca, cb = sym_to_col[sy], sym_to_col[sx]

                z_start = max(0, idx - z_window)
                sp_slice = log_arr[z_start:idx + 1, ca] - hr * log_arr[z_start:idx + 1, cb]
                sp_mean = np.mean(sp_slice)
                sp_std = np.std(sp_slice)
                if sp_std < 1e-10:
                    continue
                current_z = (sp_slice[-1] - sp_mean) / sp_std

                direction = 0
                if current_z > z_entry:
                    direction = 1
                elif current_z < -z_entry:
                    direction = -1
                if direction == 0:
                    continue

                notional = (capital / max_pairs) * lev
                if position_sizing == 'hedge_ratio':
                    tw = 1 + abs(hr)
                    wy, wx = 1.0 / tw, abs(hr) / tw
                else:
                    wy, wx = 0.5, 0.5

                active_pairs.append({
                    'sym_y': sy, 'sym_x': sx,
                    'col_y': ca, 'col_x': cb,
                    'direction': direction,
                    'hedge_ratio': hr,
                    'entry_date': dt,
                    'hold_days': 0,
                    'cum_ret': 0.0,
                    'notional': notional,
                    'weight_y': wy, 'weight_x': wx,
                    'p_val': pv, 'half_life': hl,
                })
                n_active += 1

        capital += daily_pnl
        equity_curve.append({'date': dt, 'equity': capital, 'pnl': daily_pnl,
                             'n_positions': len(active_pairs)})

    return equity_curve, all_trades


# ──────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────
def compute_metrics(equity_curve, trades, label=""):
    if not equity_curve:
        return {}
    eq = pd.DataFrame(equity_curve)
    eq['ret'] = eq['equity'].pct_change() * 100
    eq = eq.dropna()
    if len(eq) == 0:
        return {}

    total_ret = (eq['equity'].iloc[-1] / INITIAL_CAPITAL - 1) * 100
    n_days = (eq['date'].iloc[-1] - eq['date'].iloc[0]).days
    ann_ret = ((eq['equity'].iloc[-1] / INITIAL_CAPITAL) ** (252 / max(n_days, 1)) - 1) * 100
    daily_rets = eq['ret'].values
    sharpe = np.mean(daily_rets) / (np.std(daily_rets) + 1e-10) * np.sqrt(252)
    cum_max = eq['equity'].cummax()
    dd = (eq['equity'] - cum_max) / cum_max * 100
    mdd = dd.min()

    if trades:
        tdf = pd.DataFrame(trades)
        win_trades = tdf[tdf['cum_ret'] > 0]
        lose_trades = tdf[tdf['cum_ret'] <= 0]
        win_rate = len(win_trades) / len(tdf) * 100
        avg_ret = tdf['cum_ret'].mean()
        avg_win = win_trades['cum_ret'].mean() if len(win_trades) > 0 else 0
        avg_loss = lose_trades['cum_ret'].mean() if len(lose_trades) > 0 else 0
        profit_factor = abs(win_trades['cum_ret'].sum()) / (abs(lose_trades['cum_ret'].sum()) + 1e-10)
        avg_hold = tdf['hold_days'].mean()
        exit_counts = tdf['exit_reason'].value_counts().to_dict()
    else:
        win_rate = avg_ret = avg_win = avg_loss = profit_factor = avg_hold = 0
        exit_counts = {}

    return {
        'label': label,
        'total_return': round(total_ret, 2),
        'annual_return': round(ann_ret, 2),
        'sharpe': round(sharpe, 2),
        'max_drawdown': round(mdd, 2),
        'num_trades': len(trades) if trades else 0,
        'win_rate': round(win_rate, 2),
        'avg_return': round(avg_ret, 4),
        'avg_win': round(avg_win, 4),
        'avg_loss': round(avg_loss, 4),
        'profit_factor': round(profit_factor, 2),
        'avg_hold_days': round(avg_hold, 1),
        'exit_reasons': exit_counts,
        'final_equity': round(eq['equity'].iloc[-1], 0),
    }


def print_metrics(m):
    print(f"\n  {'='*60}")
    print(f"  {m['label']}")
    print(f"  {'='*60}")
    print(f"  Total Return:    {m['total_return']:>8.2f}%")
    print(f"  Annual Return:   {m['annual_return']:>8.2f}%")
    print(f"  Sharpe Ratio:    {m['sharpe']:>8.2f}")
    print(f"  Max Drawdown:    {m['max_drawdown']:>8.2f}%")
    print(f"  # Trades:        {m['num_trades']:>8d}")
    print(f"  Win Rate:        {m['win_rate']:>8.2f}%")
    print(f"  Avg Return:      {m['avg_return']:>8.4f}%")
    print(f"  Avg Win:         {m['avg_win']:>8.4f}%")
    print(f"  Avg Loss:        {m['avg_loss']:>8.4f}%")
    print(f"  Profit Factor:   {m['profit_factor']:>8.2f}")
    print(f"  Avg Hold Days:   {m['avg_hold_days']:>8.1f}")
    print(f"  Final Equity:    {m['final_equity']:>12.0f}")
    if m['exit_reasons']:
        print(f"  Exit Reasons:")
        for reason, cnt in sorted(m['exit_reasons'].items()):
            print(f"    {reason:>15s}: {cnt}")


# ──────────────────────────────────────────────────────────────────────
# TOP PAIRS ANALYSIS
# ──────────────────────────────────────────────────────────────────────
def analyze_top_pairs(all_trades, top_n=20):
    if not all_trades:
        print("\n  No trades to analyze.")
        return None

    tdf = pd.DataFrame(all_trades)
    pair_stats = tdf.groupby(['sym_y', 'sym_x']).agg(
        sector_y=('sector_y', 'first'),
        sector_x=('sector_x', 'first'),
        n_trades=('cum_ret', 'count'),
        total_ret=('cum_ret', 'sum'),
        avg_ret=('cum_ret', 'mean'),
        win_rate=('cum_ret', lambda x: (x > 0).mean() * 100),
        avg_hold=('hold_days', 'mean'),
        avg_half_life=('half_life', 'mean'),
    ).reset_index()
    pair_stats = pair_stats.sort_values('total_ret', ascending=False)

    print(f"\n  {'='*100}")
    print(f"  TOP {top_n} MOST PROFITABLE PAIRS")
    print(f"  {'='*100}")
    print(f"  {'Rank':>4} {'Pair':>14} {'Sector':>20} {'Trades':>7} {'Total%':>8} "
          f"{'Avg%':>8} {'WR%':>6} {'HoldD':>6} {'HL':>5}")
    print(f"  {'-'*4} {'-'*14} {'-'*20} {'-'*7} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*5}")

    for rank, (i, row) in enumerate(pair_stats.head(top_n).iterrows(), 1):
        pair = f"{row['sym_y']}/{row['sym_x']}"
        sec = f"{row['sector_y']}-{row['sector_x']}"
        print(f"  {rank:>4} {pair:>14} {sec:>20} {row['n_trades']:>7} "
              f"{row['total_ret']:>8.2f} {row['avg_ret']:>8.4f} "
              f"{row['win_rate']:>6.1f} {row['avg_hold']:>6.1f} "
              f"{row['avg_half_life']:>5.1f}")

    # Sector analysis
    print(f"\n  {'='*60}")
    print(f"  SECTOR COMBINATION ANALYSIS")
    print(f"  {'='*60}")
    tdf['sector_pair'] = tdf.apply(
        lambda r: tuple(sorted([r['sector_y'], r['sector_x']])), axis=1)
    tdf['sector_pair_str'] = tdf['sector_pair'].apply(lambda x: f"{x[0]}-{x[1]}")

    sec_stats = tdf.groupby('sector_pair_str').agg(
        n_trades=('cum_ret', 'count'),
        total_ret=('cum_ret', 'sum'),
        avg_ret=('cum_ret', 'mean'),
        win_rate=('cum_ret', lambda x: (x > 0).mean() * 100),
    ).sort_values('total_ret', ascending=False)

    print(f"  {'Sector Combo':>25} {'Trades':>7} {'Total%':>8} {'Avg%':>8} {'WR%':>6}")
    print(f"  {'-'*25} {'-'*7} {'-'*8} {'-'*8} {'-'*6}")
    for idx2, row in sec_stats.iterrows():
        print(f"  {idx2:>25} {row['n_trades']:>7} {row['total_ret']:>8.2f} "
              f"{row['avg_ret']:>8.4f} {row['win_rate']:>6.1f}")

    return pair_stats


# ──────────────────────────────────────────────────────────────────────
# PARAMETER GRID SEARCH
# ──────────────────────────────────────────────────────────────────────
def grid_search(log_mat, log_arr, ret_arr, sym_to_col, col_to_sym,
                usable_syms, start, end, label_prefix=""):
    """Run parameter grid search with precomputed cointegration."""
    z_thresholds = [1.5, 2.0, 2.5]
    hold_periods = [5, 10, 15, 20]
    pair_types = ['all', 'within_sector', 'cross_sector']
    sizing_methods = ['equal', 'hedge_ratio']

    total_combos = len(z_thresholds) * len(hold_periods) * len(pair_types) * len(sizing_methods)
    combo_idx = 0
    results = []

    for z_t, hold, pt, ps in product(z_thresholds, hold_periods, pair_types, sizing_methods):
        combo_idx += 1
        cfg_label = f"z={z_t}_h={hold}_{pt}_{ps}"
        full_label = f"{label_prefix}{cfg_label}" if label_prefix else cfg_label

        print(f"  [{combo_idx}/{total_combos}] {cfg_label}...", end=' ')

        try:
            # Precompute cointegration for this pair type
            coint_pc = precompute_cointegration(
                log_arr, log_mat, None, usable_syms, sym_to_col,
                formation=120, reselect_freq=60,
                start_date=start, end_date=end,
                pair_filter=pt
            )

            eq, trades = run_pairs_backtest(
                log_mat, log_arr, ret_arr,
                coint_pc, sym_to_col, col_to_sym,
                z_entry=z_t, max_hold=hold,
                position_sizing=ps,
                start=start, end=end,
                max_pairs=10,
            )
            m = compute_metrics(eq, trades, full_label)
            m['z_threshold'] = z_t
            m['hold_period'] = hold
            m['pair_type'] = pt
            m['sizing'] = ps
            results.append(m)
            print(f"Sharpe={m['sharpe']:.2f} Ret={m['total_return']:.2f}% "
                  f"Trades={m['num_trades']}")
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                'label': full_label, 'total_return': -999,
                'sharpe': -999, 'max_drawdown': -999,
                'num_trades': 0, 'z_threshold': z_t,
                'hold_period': hold, 'pair_type': pt,
                'sizing': ps, 'win_rate': 0,
            })

    return pd.DataFrame(results)


def print_grid_results(df, title=""):
    print(f"\n  {'='*100}")
    print(f"  {title}")
    print(f"  {'='*100}")
    if df.empty:
        print("  No results.")
        return

    df = df.sort_values('sharpe', ascending=False)
    print(f"  {'Rank':>4} {'Z':>4} {'Hold':>5} {'Type':>15} {'Sizing':>12} "
          f"{'Return%':>8} {'Sharpe':>7} {'MDD%':>7} {'Trades':>7} {'WR%':>6}")
    print(f"  {'-'*4} {'-'*4} {'-'*5} {'-'*15} {'-'*12} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")
    for rank, (_, row) in enumerate(df.head(30).iterrows(), 1):
        print(f"  {rank:>4} {row['z_threshold']:>4} {row['hold_period']:>5} "
              f"{row['pair_type']:>15} {row['sizing']:>12} "
              f"{row['total_return']:>8.2f} {row['sharpe']:>7.2f} "
              f"{row['max_drawdown']:>7.2f} {row['num_trades']:>7} "
              f"{row.get('win_rate', 0):>6.1f}")


# ──────────────────────────────────────────────────────────────────────
# WALK-FORWARD ANALYSIS
# ──────────────────────────────────────────────────────────────────────
def walk_forward_analysis(log_mat, log_arr, ret_arr, sym_to_col, col_to_sym,
                          usable_syms, best_params):
    """Walk-forward: train 2021-2023, validate 2024, test 2025-2026."""
    periods = {
        'TRAIN (2021-2023)': ('2021-01-01', '2023-12-31'),
        'VALIDATE (2024)':   ('2024-01-01', '2024-12-31'),
        'TEST (2025-2026)':  ('2025-01-01', '2026-12-31'),
    }

    wf_results = []
    all_period_trades = []

    for period_name, (pstart, pend) in periods.items():
        print(f"\n  >>> {period_name}: {pstart} to {pend}")
        try:
            coint_pc = precompute_cointegration(
                log_arr, log_mat, None, usable_syms, sym_to_col,
                formation=120, reselect_freq=60,
                start_date=pstart, end_date=pend,
                pair_filter=best_params['pair_type']
            )
            eq, trades = run_pairs_backtest(
                log_mat, log_arr, ret_arr,
                coint_pc, sym_to_col, col_to_sym,
                z_entry=best_params['z_threshold'],
                max_hold=best_params['hold_period'],
                position_sizing=best_params['sizing'],
                start=pstart, end=pend,
                max_pairs=10,
            )
            m = compute_metrics(eq, trades, f"{period_name}")
            m['period'] = period_name
            wf_results.append(m)
            all_period_trades.extend(trades if trades else [])
            print_metrics(m)
        except Exception as e:
            print(f"    Error: {e}")
            wf_results.append({'period': period_name, 'total_return': -999,
                               'sharpe': -999, 'max_drawdown': -999,
                               'num_trades': 0, 'win_rate': 0})

    print(f"\n  {'='*80}")
    print(f"  WALK-FORWARD SUMMARY")
    print(f"  {'='*80}")
    print(f"  Best params: z={best_params['z_threshold']}, hold={best_params['hold_period']}, "
          f"type={best_params['pair_type']}, sizing={best_params['sizing']}")
    print(f"  {'Period':>22} {'Return%':>8} {'Sharpe':>7} {'MDD%':>7} {'Trades':>7} {'WR%':>6}")
    print(f"  {'-'*22} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")
    for m in wf_results:
        print(f"  {m['period']:>22} {m.get('total_return',-999):>8.2f} "
              f"{m.get('sharpe',-999):>7.2f} {m.get('max_drawdown',-999):>7.2f} "
              f"{m.get('num_trades',0):>7} {m.get('win_rate',0):>6.1f}")

    return all_period_trades


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 70)
    print("V100: PAIRS TRADING WITH COINTEGRATION")
    print("Chinese Commodity Futures")
    print("=" * 70)

    # ── Load data ──
    print("\n[1] Loading data...")
    daily_data = load_all_data()
    close_mat, log_mat, ret_mat, close_arr, log_arr, ret_arr = build_aligned_matrix(
        daily_data, start_date='2019-01-01', end_date='2026-12-31')

    # Build symbol <-> column mapping
    sym_to_col = {s: i for i, s in enumerate(log_mat.columns)}
    col_to_sym = {i: s for s, i in sym_to_col.items()}

    # Usable symbols (must have data before 2021)
    usable_syms = []
    for s in log_mat.columns:
        first_valid_idx = log_mat[s].first_valid_index()
        if first_valid_idx and first_valid_idx <= pd.Timestamp('2020-06-01'):
            usable_syms.append(s)
    print(f"    Usable symbols (data before 2020-06): {len(usable_syms)}")

    # ── Initial cointegration scan ──
    print("\n[2] Initial cointegration scan (recent 120d)...")
    scan_coint = precompute_cointegration(
        log_arr, log_mat, None, usable_syms, sym_to_col,
        formation=120, reselect_freq=99999,
        start_date='2026-01-01', end_date='2026-12-31',
        pair_filter='all'
    )
    total_pairs = []
    for k, v in scan_coint.items():
        total_pairs.extend(v)
    total_pairs.sort(key=lambda x: x[2])
    print(f"    Found {len(total_pairs)} cointegrated pairs in recent window")
    if total_pairs:
        print(f"\n    Top 15 by p-value:")
        for rank, (sy, sx, pv, hr, hl) in enumerate(total_pairs[:15], 1):
            sec_y = SYMBOL_SECTOR.get(sy, '?')
            sec_x = SYMBOL_SECTOR.get(sx, '?')
            within = "YES" if sec_y == sec_x else "no"
            print(f"    {rank:>3}. {sy:>5}/{sx:>5}  p={pv:.4f}  "
                  f"HR={hr:>6.3f}  HL={hl:>5.1f}d  "
                  f"sectors={sec_y}/{sec_x}  within={within}")

    # ── Grid search on training period ──
    print("\n[3] Grid search on TRAIN period (2021-2023)...")
    grid_df = grid_search(
        log_mat, log_arr, ret_arr, sym_to_col, col_to_sym,
        usable_syms, start='2021-01-01', end='2023-12-31',
        label_prefix="TRAIN_")

    print_grid_results(grid_df, "GRID SEARCH RESULTS (TRAIN 2021-2023, sorted by Sharpe)")

    if grid_df.empty or grid_df['sharpe'].max() == -999:
        print("\n  No valid results. Using defaults.")
        best_params = {'z_threshold': 2.0, 'hold_period': 10,
                       'pair_type': 'all', 'sizing': 'equal'}
    else:
        best_row = grid_df.sort_values('sharpe', ascending=False).iloc[0]
        best_params = {
            'z_threshold': best_row['z_threshold'],
            'hold_period': best_row['hold_period'],
            'pair_type': best_row['pair_type'],
            'sizing': best_row['sizing'],
        }
        print(f"\n  BEST PARAMS (by Sharpe): z={best_params['z_threshold']}, "
              f"hold={best_params['hold_period']}, type={best_params['pair_type']}, "
              f"sizing={best_params['sizing']}")

    # ── Walk-forward analysis ──
    print("\n[4] Walk-forward analysis...")
    wf_trades = walk_forward_analysis(
        log_mat, log_arr, ret_arr, sym_to_col, col_to_sym,
        usable_syms, best_params)

    # ── Top pairs analysis on test period ──
    print("\n[5] Top pairs analysis (test 2025-2026)...")
    try:
        coint_test = precompute_cointegration(
            log_arr, log_mat, None, usable_syms, sym_to_col,
            formation=120, reselect_freq=60,
            start_date='2025-01-01', end_date='2026-12-31',
            pair_filter=best_params['pair_type']
        )
        eq_test, trades_test = run_pairs_backtest(
            log_mat, log_arr, ret_arr,
            coint_test, sym_to_col, col_to_sym,
            z_entry=best_params['z_threshold'],
            max_hold=best_params['hold_period'],
            position_sizing=best_params['sizing'],
            start='2025-01-01', end='2026-12-31',
            max_pairs=10,
        )
        analyze_top_pairs(trades_test, top_n=20)
    except Exception as e:
        print(f"  Test error: {e}")

    # ── Within-sector vs cross-sector comparison ──
    print("\n[6] Within-sector vs Cross-sector comparison (test 2025-2026)...")
    for pt in ['within_sector', 'cross_sector', 'all']:
        try:
            coint_cmp = precompute_cointegration(
                log_arr, log_mat, None, usable_syms, sym_to_col,
                formation=120, reselect_freq=60,
                start_date='2025-01-01', end_date='2026-12-31',
                pair_filter=pt
            )
            eq_cmp, trades_cmp = run_pairs_backtest(
                log_mat, log_arr, ret_arr,
                coint_cmp, sym_to_col, col_to_sym,
                z_entry=best_params['z_threshold'],
                max_hold=best_params['hold_period'],
                position_sizing=best_params['sizing'],
                start='2025-01-01', end='2026-12-31',
                max_pairs=10,
            )
            m_cmp = compute_metrics(eq_cmp, trades_cmp, f"TEST {pt}")
            print_metrics(m_cmp)
        except Exception as e:
            print(f"  {pt} error: {e}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Total runtime: {elapsed:.1f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
