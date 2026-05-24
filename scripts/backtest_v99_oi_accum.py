#!/usr/bin/env python3
"""
V99: Open Interest Accumulation/Distribution Strategy
======================================================
Core idea: OI + Price classify market regime
  - OI up + Price up = "Accumulation Long" -> GO LONG
  - OI up + Price down = "Accumulation Short" -> GO SHORT
  - OI down + Price up = "Short Covering" -> AVOID
  - OI down + Price down = "Long Liquidation" -> AVOID

Performance: Pre-compute signals into daily-ranked panels.
Walk-forward: train 2021-2023, validate 2024, test 2025-2026.
"""
import os, sys, glob, time
import numpy as np
import pandas as pd
import warnings
from itertools import product

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAILY_DIR = os.path.join(BASE_DIR, 'data', 'futures_weighted')

INITIAL_CAPITAL = 500_000
LEVERAGE = 3

TRAIN_START, TRAIN_END = '2021-01-01', '2023-12-31'
VALID_START, VALID_END = '2024-01-01', '2024-12-31'
TEST_START, TEST_END = '2025-01-01', '2026-05-20'


# ======================================================================
# DATA LOADING
# ======================================================================
def load_panel():
    """Load all futures data into a panel with pre-computed features."""
    print("Loading futures daily data...", flush=True)
    t0 = time.time()

    dfs = []
    for f in sorted(glob.glob(os.path.join(DAILY_DIR, '*.csv'))):
        sym = os.path.basename(f).replace('.csv', '')
        df = pd.read_csv(f, usecols=['trade_date','open','high','low','close','vol','oi'])
        if len(df) < 200:
            continue
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        df = df.sort_values('trade_date').reset_index(drop=True)
        df = df[df['close'].notna() & (df['close'] > 0)].reset_index(drop=True)
        if len(df) < 100:
            continue
        df = df[df['trade_date'] >= '2019-01-01'].reset_index(drop=True)
        if len(df) < 100:
            continue
        df['symbol'] = sym
        dfs.append(df[['symbol','trade_date','open','high','low','close','vol','oi']])

    panel = pd.concat(dfs, ignore_index=True)
    panel = panel.sort_values(['symbol','trade_date']).reset_index(drop=True)

    # Pre-compute OI features per symbol
    print("  Computing OI features...", flush=True)
    for w in [3, 5, 10]:
        panel[f'oi_chg_{w}d'] = panel.groupby('symbol')['oi'].transform(
            lambda x: (x - x.shift(w)) / x.shift(w).replace(0, np.nan) * 100
        )
        panel[f'price_chg_{w}d'] = panel.groupby('symbol')['close'].transform(
            lambda x: (x - x.shift(w)) / x.shift(w) * 100
        )

    # OI velocity
    for w in [5, 10]:
        panel[f'oi_vel_{w}d'] = panel.groupby('symbol')[f'oi_chg_{w}d'].transform(
            lambda x: x - x.shift(w)
        )

    # OI vs Volume ratio
    vol_chg_5d = panel.groupby('symbol')['vol'].transform(
        lambda x: (x - x.shift(5)) / x.shift(5).replace(0, np.nan) * 100
    )
    panel['oi_vol_ratio_5d'] = panel['oi_chg_5d'].abs() / vol_chg_5d.abs().replace(0, np.nan)

    # Persistent OI streak
    panel['oi_up'] = (panel.groupby('symbol')['oi'].transform(
        lambda x: x > x.shift(1))).astype(int)
    panel['oi_persist'] = panel.groupby('symbol')['oi_up'].transform(
        lambda s: s.groupby((s != s.shift()).cumsum()).cumcount() + 1
    )
    # Fix: the above counts any consecutive same-value streak, we want consecutive OI up
    def streak_fn(s):
        result = []
        count = 0
        for v in s:
            if v == 1:
                count += 1
            else:
                count = 0
            result.append(count)
        return pd.Series(result, index=s.index)
    panel['oi_persist'] = panel.groupby('symbol')['oi_up'].transform(streak_fn)

    # Set trade_date as index for fast lookups
    panel = panel.set_index('trade_date')

    # Build sym -> sub-dataframe for quick access
    print("  Building symbol panels...", flush=True)
    sym_panels = {sym: group for sym, group in panel.groupby('symbol')}

    # Group by date once
    print("  Building date groups...", flush=True)
    date_groups = {dt: group for dt, group in panel.groupby(panel.index)}

    elapsed = time.time() - t0
    print(f"  Panel: {len(panel):,} rows, {panel['symbol'].nunique()} symbols, {elapsed:.1f}s", flush=True)

    return panel, date_groups, sym_panels


# ======================================================================
# BACKTEST ENGINE
# ======================================================================
def run_backtest(panel, date_groups, sym_panels, start, end,
                 window=5, oi_threshold=5.0,
                 top_k=10, hold=10,
                 sl_pct=-2.0, tp_pct=5.0,
                 filter_type='none',
                 lev=LEVERAGE):
    """
    Backtest with pre-computed signals applied inline.
    panel: original panel (trade_date indexed) with oi_chg, price_chg, velocity, etc.
    date_groups: dict of date -> DataFrame (all symbols for that date)
    sym_panels: dict of symbol -> DataFrame (all dates for that symbol)
    """
    oi_chg_col = f'oi_chg_{window}d'
    price_chg_col = f'price_chg_{window}d'
    vel_col = f'oi_vel_{window}d'
    use_vel = filter_type == 'velocity'
    use_oi_vol = filter_type == 'oi_vol_ratio'
    use_persist = filter_type == 'persistent'

    dates = pd.bdate_range(start=start, end=end)
    # Filter to dates that actually exist in our data
    available_dates = set(date_groups.keys())
    dates = [d for d in dates if d in available_dates]

    cap = INITIAL_CAPITAL
    equity_curve = []
    all_trades = []
    positions = {}
    n_slots = top_k * 2

    for dt in dates:
        if dt not in date_groups:
            continue

        pnl = 0.0
        to_remove = []

        # ── EXIT LOGIC ──
        for sym, pos in list(positions.items()):
            sp = sym_panels.get(sym)
            if sp is None:
                continue
            if dt not in sp.index:
                continue

            row = sp.loc[dt]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            cur_c = row['close']
            cur_h = row['high']
            cur_l = row['low']
            if np.isnan(cur_c) or cur_c == 0:
                continue

            days_held = (dt - pos['entry_date']).days
            direction = pos['direction']

            slippage = 0.001
            if direction == 1:
                raw_ret = (cur_c / pos['entry_price'] - 1) * 100
                sl_trigger = (cur_l / pos['entry_price'] - 1) * 100
                tp_trigger = (cur_h / pos['entry_price'] - 1) * 100
            else:
                raw_ret = (1 - cur_c / pos['entry_price']) * 100
                # For short: SL when intraday gain is negative (price rose)
                # Using low as worst intraday marker (conservative for short)
                sl_trigger = (1 - cur_l / pos['entry_price']) * 100
                # TP when intraday gain is large (price fell intraday)
                tp_trigger = (1 - cur_h / pos['entry_price']) * 100

            triggered = False
            exit_reason = None
            actual_ret = raw_ret

            if sl_trigger <= sl_pct:
                actual_ret = sl_pct - slippage
                triggered = True
                exit_reason = 'SL'
            elif tp_trigger >= tp_pct:
                actual_ret = tp_pct - slippage
                triggered = True
                exit_reason = 'TP'
            elif days_held >= hold:
                actual_ret = raw_ret - slippage
                triggered = True
                exit_reason = 'HOLD'

            if triggered:
                trade_pnl = actual_ret / 100.0 * pos['notional'] * lev
                pnl += trade_pnl
                all_trades.append({
                    'symbol': sym, 'entry_date': pos['entry_date'],
                    'exit_date': dt, 'direction': direction,
                    'entry_price': pos['entry_price'], 'exit_price': cur_c,
                    'return_pct': actual_ret, 'pnl': trade_pnl,
                    'exit_reason': exit_reason,
                    'oi_chg': pos.get('oi_chg', np.nan),
                    'regime': pos.get('regime', ''),
                    'days_held': days_held,
                })
                to_remove.append(sym)

        for sym in to_remove:
            del positions[sym]

        # ── ENTRY LOGIC (vectorized signal computation on day's data) ──
        n_open = len(positions)
        if n_open < n_slots:
            day_df = date_groups[dt]

            # Compute signals for today
            oi_chg = day_df[oi_chg_col].values
            price_chg = day_df[price_chg_col].values

            # Regime classification
            direction = np.zeros(len(day_df), dtype=int)
            oi_up = oi_chg > 0
            price_up = price_chg > 0
            direction[oi_up & price_up] = 1      # AccumLong
            direction[oi_up & ~price_up] = -1     # AccumShort

            # Threshold filter
            direction[np.abs(oi_chg) < oi_threshold] = 0

            # Additional filters
            if use_vel and vel_col in day_df.columns:
                vel = day_df[vel_col].values
                direction[(direction == 1) & (vel <= 0)] = 0
                direction[(direction == -1) & (vel >= 0)] = 0
            elif use_oi_vol and 'oi_vol_ratio_5d' in day_df.columns:
                r = day_df['oi_vol_ratio_5d'].values
                direction[r < 1.0] = 0
            elif use_persist:
                persist = day_df['oi_persist'].values
                direction[persist < 3] = 0

            # Filter to active signals
            active_mask = direction != 0
            if active_mask.any():
                active = day_df[active_mask].copy()
                active_dir = direction[active_mask]
                active_score = np.abs(oi_chg[active_mask])
                active['_direction'] = active_dir
                active['_score'] = active_score

                long_cands = active[active['_direction'] == 1].nlargest(top_k, '_score')
                short_cands = active[active['_direction'] == -1].nlargest(top_k, '_score')

                n_long_open = sum(1 for p in positions.values() if p['direction'] == 1)
                n_short_open = sum(1 for p in positions.values() if p['direction'] == -1)

                for _, row in pd.concat([long_cands, short_cands]).iterrows():
                    sym = row['symbol']
                    if sym in positions:
                        continue
                    d = int(row['_direction'])
                    if d == 1 and n_long_open >= top_k:
                        continue
                    if d == -1 and n_short_open >= top_k:
                        continue

                    price = row['close']
                    if np.isnan(price) or price == 0:
                        continue

                    regime = 'AccumLong' if d == 1 else 'AccumShort'
                    notional = cap / n_slots
                    positions[sym] = {
                        'direction': d, 'entry_date': dt,
                        'entry_price': price, 'notional': notional,
                        'oi_chg': float(row[oi_chg_col]),
                        'regime': regime,
                    }
                    if d == 1:
                        n_long_open += 1
                    else:
                        n_short_open += 1
                    if n_long_open + n_short_open >= n_slots:
                        break

        # Update equity (mark-to-market)
        day_pnl = pnl
        for sym, pos in positions.items():
            sp = sym_panels.get(sym)
            if sp is None:
                continue
            if dt not in sp.index:
                continue
            row = sp.loc[dt]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            cur_c = row['close']
            if np.isnan(cur_c) or cur_c == 0:
                continue
            if pos['direction'] == 1:
                unreal = (cur_c / pos['entry_price'] - 1)
            else:
                unreal = (1 - cur_c / pos['entry_price'])
            day_pnl += unreal * pos['notional'] * lev

        cap += pnl
        equity_curve.append({'date': dt, 'equity': cap, 'daily_pnl': pnl})

    # Close remaining positions
    end_ts = pd.Timestamp(end)
    for sym, pos in list(positions.items()):
        sp = sym_panels.get(sym)
        if sp is None:
            continue
        valid = sp.loc[sp.index <= end_ts]
        if len(valid) == 0:
            continue
        row = valid.iloc[-1]
        cur_c = row['close']
        if np.isnan(cur_c) or cur_c == 0:
            continue
        if pos['direction'] == 1:
            ret = (cur_c / pos['entry_price'] - 1) * 100 - 0.001
        else:
            ret = (1 - cur_c / pos['entry_price']) * 100 - 0.001
        trade_pnl = ret / 100.0 * pos['notional'] * lev
        all_trades.append({
            'symbol': sym, 'entry_date': pos['entry_date'],
            'exit_date': end_ts, 'direction': pos['direction'],
            'entry_price': pos['entry_price'], 'exit_price': cur_c,
            'return_pct': ret, 'pnl': trade_pnl, 'exit_reason': 'END',
            'oi_chg': pos.get('oi_chg', np.nan),
            'regime': pos.get('regime', ''),
            'days_held': (end_ts - pos['entry_date']).days,
        })

    eq_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
    return eq_df, trades_df


# ======================================================================
# METRICS
# ======================================================================
def calc_metrics(eq_df, trades_df):
    if eq_df.empty or len(eq_df) < 10:
        return None
    eq = eq_df['equity'].values
    total_return = (eq[-1] / eq[0] - 1) * 100

    daily_ret = eq_df['daily_pnl'].values / INITIAL_CAPITAL
    ann_ret = np.mean(daily_ret) * 252 * 100
    ann_vol = np.std(daily_ret) * np.sqrt(252) * 100
    sharpe = (ann_ret / ann_vol) if ann_vol > 0 else 0

    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    mdd = dd.min()

    n_trades = len(trades_df)
    if n_trades > 0:
        win_t = trades_df[trades_df['pnl'] > 0]
        lose_t = trades_df[trades_df['pnl'] <= 0]
        wr = len(win_t) / n_trades * 100
        avg_win = win_t['return_pct'].mean() if len(win_t) > 0 else 0
        avg_loss = lose_t['return_pct'].mean() if len(lose_t) > 0 else 0
        avg_ret = trades_df['return_pct'].mean()
        long_t = trades_df[trades_df['direction'] == 1]
        short_t = trades_df[trades_df['direction'] == -1]
        long_wr = (len(long_t[long_t['pnl'] > 0]) / len(long_t) * 100) if len(long_t) > 0 else 0
        short_wr = (len(short_t[short_t['pnl'] > 0]) / len(short_t) * 100) if len(short_t) > 0 else 0
        long_avg = long_t['return_pct'].mean() if len(long_t) > 0 else 0
        short_avg = short_t['return_pct'].mean() if len(short_t) > 0 else 0
        sl_count = len(trades_df[trades_df['exit_reason'] == 'SL'])
        tp_count = len(trades_df[trades_df['exit_reason'] == 'TP'])
        hold_count = len(trades_df[trades_df['exit_reason'] == 'HOLD'])
        accum_l = trades_df[trades_df['regime'] == 'AccumLong']
        accum_s = trades_df[trades_df['regime'] == 'AccumShort']
        accum_l_wr = (len(accum_l[accum_l['pnl'] > 0]) / len(accum_l) * 100) if len(accum_l) > 0 else 0
        accum_s_wr = (len(accum_s[accum_s['pnl'] > 0]) / len(accum_s) * 100) if len(accum_s) > 0 else 0
        accum_l_n = len(accum_l)
        accum_s_n = len(accum_s)
    else:
        wr = avg_win = avg_loss = avg_ret = 0
        long_wr = short_wr = long_avg = short_avg = 0
        sl_count = tp_count = hold_count = 0
        accum_l_wr = accum_s_wr = 0
        accum_l_n = accum_s_n = 0

    return {
        'total_return': total_return,
        'ann_return': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'mdd': mdd,
        'n_trades': n_trades,
        'win_rate': wr,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_ret': avg_ret,
        'long_wr': long_wr,
        'short_wr': short_wr,
        'long_avg': long_avg,
        'short_avg': short_avg,
        'sl_count': sl_count,
        'tp_count': tp_count,
        'hold_count': hold_count,
        'accum_l_n': accum_l_n,
        'accum_l_wr': accum_l_wr,
        'accum_s_n': accum_s_n,
        'accum_s_wr': accum_s_wr,
    }


def print_metrics(m, label=""):
    if m is None:
        print(f"  {label}: NO DATA", flush=True)
        return
    print(f"  {label}", flush=True)
    print(f"    Return: {m['total_return']:+.2f}% (ann {m['ann_return']:+.2f}%)  "
          f"Vol: {m['ann_vol']:.2f}%  Sharpe: {m['sharpe']:.2f}", flush=True)
    print(f"    MDD: {m['mdd']:.2f}%  Trades: {m['n_trades']}  WR: {m['win_rate']:.1f}%  "
          f"AvgWin: {m['avg_win']:+.2f}%  AvgLoss: {m['avg_loss']:+.2f}%", flush=True)
    print(f"    Long WR: {m['long_wr']:.1f}% (avg {m['long_avg']:+.3f}%)  "
          f"Short WR: {m['short_wr']:.1f}% (avg {m['short_avg']:+.3f}%)", flush=True)
    print(f"    Regimes -> AccumLong: {m['accum_l_n']} (WR {m['accum_l_wr']:.1f}%)  "
          f"AccumShort: {m['accum_s_n']} (WR {m['accum_s_wr']:.1f}%)", flush=True)
    print(f"    Exits -> SL: {m['sl_count']}  TP: {m['tp_count']}  Hold: {m['hold_count']}", flush=True)


# ======================================================================
# PARAMETER SWEEP
# ======================================================================
def parameter_sweep(panel, date_groups, sym_panels, start, end, label="TRAIN"):
    """
    Efficient two-stage sweep:
      Stage 1: Find best window and filter with moderate params (36 combos)
      Stage 2: Full sweep on best window across thresholds/K/hold (48 combos)
    """
    print(f"\n{'='*70}", flush=True)
    print(f"  PARAMETER SWEEP: {label} ({start} to {end})", flush=True)
    print(f"{'='*70}", flush=True)

    t0 = time.time()

    # Stage 1: Find best window x filter with moderate params
    print(f"\n  Stage 1: Coarse sweep (36 combos)...", flush=True)
    stage1 = []
    for w in [3, 5, 10]:
        for ft in ['none', 'velocity', 'oi_vol_ratio', 'persistent']:
            for th in [3.0, 5.0, 8.0]:
                eq_df, trades_df = run_backtest(
                    panel, date_groups, sym_panels, start, end,
                    window=w, oi_threshold=th, top_k=10, hold=10,
                    filter_type=ft
                )
                m = calc_metrics(eq_df, trades_df)
                if m is not None and m['n_trades'] >= 20:
                    params = {'window': w, 'threshold': th, 'top_k': 10,
                              'hold': 10, 'filter': ft}
                    stage1.append((params, m))

    stage1.sort(key=lambda x: x[1]['sharpe'], reverse=True)

    # Get top-3 unique windows and top-2 unique filters
    seen_w = set()
    seen_f = set()
    best_windows = []
    best_filters = []
    for p, m in stage1:
        if p['window'] not in seen_w and len(best_windows) < 2:
            best_windows.append(p['window'])
            seen_w.add(p['window'])
        if p['filter'] not in seen_f and len(best_filters) < 2:
            best_filters.append(p['filter'])
            seen_f.add(p['filter'])

    print(f"\n  Stage 1 top config:", flush=True)
    for i, (p, m) in enumerate(stage1[:5]):
        print(f"    #{i+1}: w={p['window']} th={p['threshold']}% filter={p['filter']} "
              f"Sharpe={m['sharpe']:.2f} WR={m['win_rate']:.1f}%", flush=True)
    print(f"  Selected windows: {best_windows}, filters: {best_filters}", flush=True)

    # Stage 2: Fine sweep on selected windows and filters
    print(f"\n  Stage 2: Fine sweep...", flush=True)
    results = list(stage1)
    for w in best_windows:
        for ft in best_filters:
            for th in [3.0, 5.0, 8.0, 10.0]:
                for k in [5, 10, 15]:
                    for h in [5, 10, 15, 20]:
                        # Skip configs already in stage1
                        dup = any(p['window'] == w and p['threshold'] == th and
                                 p['top_k'] == k and p['hold'] == h and
                                 p['filter'] == ft for p, _ in stage1)
                        if dup:
                            continue

                        eq_df, trades_df = run_backtest(
                            panel, date_groups, sym_panels, start, end,
                            window=w, oi_threshold=th, top_k=k, hold=h,
                            filter_type=ft
                        )
                        m = calc_metrics(eq_df, trades_df)
                        if m is not None and m['n_trades'] >= 20:
                            params = {'window': w, 'threshold': th, 'top_k': k,
                                      'hold': h, 'filter': ft}
                            results.append((params, m))

    results.sort(key=lambda x: x[1]['sharpe'], reverse=True)
    elapsed = time.time() - t0
    print(f"\n  Sweep done: {len(results)} configs in {elapsed:.0f}s", flush=True)

    return results


# ======================================================================
# WALK-FORWARD
# ======================================================================
def walk_forward_test(panel, date_groups, sym_panels, train_results):
    print(f"\n{'='*70}", flush=True)
    print(f"  WALK-FORWARD ANALYSIS", flush=True)
    print(f"{'='*70}", flush=True)

    top_n = min(10, len(train_results))
    print(f"\nTop {top_n} configs from training (by Sharpe):", flush=True)
    for i, (p, m) in enumerate(train_results[:top_n]):
        print(f"  #{i+1}: w={p['window']} th={p['threshold']}% K={p['top_k']} "
              f"hold={p['hold']} filter={p['filter']} -> "
              f"Sharpe={m['sharpe']:.2f} WR={m['win_rate']:.1f}% "
              f"Ret={m['total_return']:+.2f}% MDD={m['mdd']:.2f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"  VALIDATION: {VALID_START} to {VALID_END}", flush=True)
    print(f"{'='*70}", flush=True)

    valid_results = []
    for p, _ in train_results[:top_n]:
        eq_df, trades_df = run_backtest(
            panel, date_groups, sym_panels, VALID_START, VALID_END,
            window=p['window'], oi_threshold=p['threshold'],
            top_k=p['top_k'], hold=p['hold'],
            filter_type=p['filter']
        )
        m = calc_metrics(eq_df, trades_df)
        if m is not None:
            valid_results.append((p, m))
            print_metrics(m, f"w={p['window']} th={p['threshold']}% K={p['top_k']} "
                          f"hold={p['hold']} filter={p['filter']}")

    if not valid_results:
        print("  No valid results in validation!", flush=True)
        return None

    valid_results.sort(key=lambda x: x[1]['sharpe'], reverse=True)
    best_params, best_valid = valid_results[0]
    print(f"\n  >>> BEST VALIDATION <<<", flush=True)
    print_metrics(best_valid, f"w={best_params['window']} th={best_params['threshold']}% "
                  f"K={best_params['top_k']} hold={best_params['hold']} filter={best_params['filter']}")

    # Robustness check
    train_p2s = {str(p): m['sharpe'] for p, m in train_results[:top_n]}
    valid_p2s = {str(p): m['sharpe'] for p, m in valid_results}
    print(f"\n  Train/Valid Sharpe correlation:", flush=True)
    for p_str, ts in sorted(train_p2s.items(), key=lambda x: -x[1])[:5]:
        vs = valid_p2s.get(p_str, None)
        vs_str = f"{vs:.2f}" if vs is not None else 'N/A'
        print(f"    {p_str}: train={ts:.2f}  valid={vs_str}", flush=True)

    # OOS Test
    print(f"\n{'='*70}", flush=True)
    print(f"  OUT-OF-SAMPLE TEST: {TEST_START} to {TEST_END}", flush=True)
    print(f"{'='*70}", flush=True)

    eq_df, trades_df = run_backtest(
        panel, date_groups, sym_panels, TEST_START, TEST_END,
        window=best_params['window'], oi_threshold=best_params['threshold'],
        top_k=best_params['top_k'], hold=best_params['hold'],
        filter_type=best_params['filter']
    )
    m = calc_metrics(eq_df, trades_df)
    print_metrics(m, f"BEST: w={best_params['window']} th={best_params['threshold']}% "
                  f"K={best_params['top_k']} hold={best_params['hold']} filter={best_params['filter']}")

    print(f"\n  Testing top-3 validation configs OOS:", flush=True)
    for i, (p, vm) in enumerate(valid_results[:3]):
        eq_df, trades_df = run_backtest(
            panel, date_groups, sym_panels, TEST_START, TEST_END,
            window=p['window'], oi_threshold=p['threshold'],
            top_k=p['top_k'], hold=p['hold'],
            filter_type=p['filter']
        )
        m = calc_metrics(eq_df, trades_df)
        print_metrics(m, f"  Valid#{i+1}: w={p['window']} th={p['threshold']}% "
                      f"K={p['top_k']} hold={p['hold']} filter={p['filter']}")

    return best_params, best_valid, m


# ======================================================================
# DETAILED ANALYSIS
# ======================================================================
def detailed_analysis(panel, date_groups, sym_panels, params, period_start, period_end, label=""):
    print(f"\n{'='*70}", flush=True)
    print(f"  DETAILED ANALYSIS: {label} ({period_start} to {period_end})", flush=True)
    print(f"  Config: w={params['window']} th={params['threshold']}% "
          f"K={params['top_k']} hold={params['hold']} filter={params['filter']}", flush=True)
    print(f"{'='*70}", flush=True)

    eq_df, trades_df = run_backtest(
        panel, date_groups, sym_panels, period_start, period_end,
        window=params['window'], oi_threshold=params['threshold'],
        top_k=params['top_k'], hold=params['hold'],
        filter_type=params['filter']
    )
    m = calc_metrics(eq_df, trades_df)
    print_metrics(m, "Overall")

    if trades_df.empty:
        return m

    # Monthly
    trades_df['exit_month'] = pd.to_datetime(trades_df['exit_date']).dt.to_period('M')
    monthly = trades_df.groupby('exit_month').agg(
        n_trades=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).sum() / len(x) * 100),
        total_pnl=('pnl', 'sum'),
        avg_ret=('return_pct', 'mean'),
    )
    print(f"\n  Monthly:", flush=True)
    print(f"  {'Month':<10} {'N':>5} {'WR%':>7} {'PnL':>12} {'AvgRet%':>9}", flush=True)
    for month, row in monthly.iterrows():
        print(f"  {str(month):<10} {int(row['n_trades']):>5} {row['win_rate']:>7.1f} "
              f"{row['total_pnl']:>12,.0f} {row['avg_ret']:>+9.3f}", flush=True)

    # By commodity
    sym_stats = trades_df.groupby('symbol').agg(
        n_trades=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).sum() / len(x) * 100),
        avg_ret=('return_pct', 'mean'),
        total_pnl=('pnl', 'sum'),
    ).sort_values('total_pnl', ascending=False)
    print(f"\n  Top 15 commodities:", flush=True)
    print(f"  {'Symbol':<8} {'N':>5} {'WR%':>7} {'AvgRet%':>9} {'PnL':>12}", flush=True)
    for sym, row in sym_stats.head(15).iterrows():
        print(f"  {sym:<8} {int(row['n_trades']):>5} {row['win_rate']:>7.1f} "
              f"{row['avg_ret']:>+9.3f} {row['total_pnl']:>12,.0f}", flush=True)

    # By regime
    regime_stats = trades_df.groupby('regime').agg(
        n_trades=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).sum() / len(x) * 100),
        avg_ret=('return_pct', 'mean'),
        total_pnl=('pnl', 'sum'),
    )
    print(f"\n  By regime:", flush=True)
    for regime, row in regime_stats.iterrows():
        print(f"    {regime:<15} N={int(row['n_trades']):>5} WR={row['win_rate']:.1f}% "
              f"AvgRet={row['avg_ret']:+.3f}% PnL={row['total_pnl']:,.0f}", flush=True)

    # Exit reason
    exit_stats = trades_df.groupby('exit_reason').agg(
        n_trades=('pnl', 'count'),
        avg_ret=('return_pct', 'mean'),
        win_rate=('pnl', lambda x: (x > 0).sum() / len(x) * 100),
    )
    print(f"\n  By exit:", flush=True)
    for reason, row in exit_stats.iterrows():
        print(f"    {reason:<6} N={int(row['n_trades']):>5} WR={row['win_rate']:.1f}% "
              f"AvgRet={row['avg_ret']:+.3f}%", flush=True)

    # Drawdowns
    if not eq_df.empty:
        eq = eq_df['equity'].values
        peak_arr = np.maximum.accumulate(eq)
        dd = (eq - peak_arr) / peak_arr * 100
        in_dd = dd < 0
        dd_periods = []
        dd_start = None
        for j in range(len(dd)):
            if in_dd[j] and dd_start is None:
                dd_start = j
            elif not in_dd[j] and dd_start is not None:
                dd_periods.append((dd_start, j - 1))
                dd_start = None
        if dd_start is not None:
            dd_periods.append((dd_start, len(dd) - 1))

        dd_info = []
        for s, e in dd_periods:
            worst_idx = s + np.argmin(dd[s:e + 1])
            dd_info.append({
                'start': eq_df['date'].iloc[s],
                'trough': eq_df['date'].iloc[worst_idx],
                'end': eq_df['date'].iloc[e],
                'depth': dd[worst_idx],
            })
        dd_info.sort(key=lambda x: x['depth'])

        print(f"\n  Top 5 drawdowns:", flush=True)
        for i, d in enumerate(dd_info[:5]):
            print(f"    #{i+1}: {d['depth']:.2f}%  "
                  f"{d['start'].strftime('%Y-%m-%d')} -> "
                  f"{d['trough'].strftime('%Y-%m-%d')} -> "
                  f"{d['end'].strftime('%Y-%m-%d')}", flush=True)

    return m


# ======================================================================
# QUICK COMPARISON
# ======================================================================
def quick_compare(panel, date_groups, sym_panels, start, end, label=""):
    print(f"\n{'='*70}", flush=True)
    print(f"  QUICK COMPARISON: {label} ({start} to {end})", flush=True)
    print(f"{'='*70}", flush=True)

    configs = [
        {'window': 5, 'threshold': 5.0, 'top_k': 10, 'hold': 10, 'filter': 'none'},
        {'window': 5, 'threshold': 5.0, 'top_k': 10, 'hold': 10, 'filter': 'velocity'},
        {'window': 5, 'threshold': 5.0, 'top_k': 10, 'hold': 10, 'filter': 'oi_vol_ratio'},
        {'window': 5, 'threshold': 5.0, 'top_k': 10, 'hold': 10, 'filter': 'persistent'},
        {'window': 3, 'threshold': 5.0, 'top_k': 10, 'hold': 10, 'filter': 'none'},
        {'window': 10, 'threshold': 5.0, 'top_k': 10, 'hold': 10, 'filter': 'none'},
        {'window': 5, 'threshold': 3.0, 'top_k': 10, 'hold': 10, 'filter': 'none'},
        {'window': 5, 'threshold': 8.0, 'top_k': 10, 'hold': 10, 'filter': 'none'},
    ]

    for cfg in configs:
        t0 = time.time()
        eq_df, trades_df = run_backtest(
            panel, date_groups, sym_panels, start, end,
            window=cfg['window'], oi_threshold=cfg['threshold'],
            top_k=cfg['top_k'], hold=cfg['hold'],
            filter_type=cfg['filter']
        )
        elapsed = time.time() - t0
        m = calc_metrics(eq_df, trades_df)
        if m is not None:
            print(f"  w={cfg['window']} th={cfg['threshold']}% K={cfg['top_k']} "
                  f"hold={cfg['hold']} filter={cfg['filter']:<14} -> "
                  f"Sharpe={m['sharpe']:.2f} WR={m['win_rate']:.1f}% "
                  f"Ret={m['ann_return']:+.2f}% MDD={m['mdd']:.2f}% "
                  f"Trades={m['n_trades']} ({elapsed:.1f}s)", flush=True)
        else:
            print(f"  w={cfg['window']} th={cfg['threshold']}% K={cfg['top_k']} "
                  f"hold={cfg['hold']} filter={cfg['filter']:<14} -> NO DATA", flush=True)


# ======================================================================
# MAIN
# ======================================================================
def main():
    print("=" * 70, flush=True)
    print("  V99: Open Interest Accumulation/Distribution Strategy", flush=True)
    print("=" * 70, flush=True)

    panel, date_groups, sym_panels = load_panel()

    # Phase 1: Quick comparison
    quick_compare(panel, date_groups, sym_panels, TRAIN_START, TRAIN_END, "TRAIN")
    quick_compare(panel, date_groups, sym_panels, VALID_START, VALID_END, "VALID")

    # Phase 2: Full parameter sweep
    t0 = time.time()
    train_results = parameter_sweep(panel, date_groups, sym_panels, TRAIN_START, TRAIN_END, "TRAIN")
    elapsed = time.time() - t0
    print(f"\n  Sweep completed in {elapsed:.0f}s", flush=True)

    if not train_results:
        print("  No valid results!", flush=True)
        return

    print(f"\n  Top 20 training configs (by Sharpe):", flush=True)
    print(f"  {'#':>3} {'w':>3} {'th%':>5} {'K':>3} {'hold':>5} {'filter':<14} "
          f"{'Sharpe':>8} {'WR%':>6} {'AnnRet%':>9} {'MDD%':>7} {'Trades':>7}", flush=True)
    for i, (p, m) in enumerate(train_results[:20]):
        print(f"  {i+1:>3} {p['window']:>3} {p['threshold']:>5.0f} {p['top_k']:>3} "
              f"{p['hold']:>5} {p['filter']:<14} "
              f"{m['sharpe']:>8.2f} {m['win_rate']:>6.1f} "
              f"{m['ann_return']:>+9.2f} {m['mdd']:>7.2f} {m['n_trades']:>7}", flush=True)

    # Phase 3: Walk-forward
    wf_result = walk_forward_test(panel, date_groups, sym_panels, train_results)

    # Phase 4: Detailed analysis
    if wf_result:
        best_params, _, _ = wf_result
        detailed_analysis(panel, date_groups, sym_panels, best_params,
                         TRAIN_START, TEST_END, "FULL PERIOD")
        detailed_analysis(panel, date_groups, sym_panels, best_params,
                         TEST_START, TEST_END, "TEST ONLY (2025-2026)")

    print(f"\n{'='*70}", flush=True)
    print(f"  V99 BACKTEST COMPLETE", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == '__main__':
    main()
