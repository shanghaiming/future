#!/usr/bin/env python3
"""
V98: Term Structure Spread Mean-Reversion Strategy
===================================================
Core idea: Spread percentile rank mean-reversion
  - When spread percentile > threshold (e.g. 90th): spread historically wide -> SHORT
  - When spread percentile < (100 - threshold): spread historically tight -> LONG

Test matrix:
  - Percentile thresholds: 80/20, 85/15, 90/10, 95/5
  - Hold periods: 5d, 10d, 15d, 20d
  - Position counts: 5, 10 per side
  - Filters: none, momentum, volume, momentum+volume

Risk: -2% SL, +5% TP. Walk-forward: train 2021-2023, validate 2024, test 2025-2026.
"""
import os, sys, json, glob, time
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
from itertools import product

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS_DIR = os.path.join(BASE_DIR, 'data', 'futures_term_structure')
DAILY_DIR = os.path.join(BASE_DIR, 'data', 'futures_weighted')

INITIAL_CAPITAL = 500_000
LEVERAGE = 3


# ──────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────
def load_all_data():
    """Load term structure and daily price data with optimized lookups."""
    print("  Loading term structure data...", flush=True)
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(TS_DIR, '*.json')))
    records = []
    for f in files:
        try:
            with open(f, 'r') as fh:
                d = json.load(fh)
            records.append({
                'symbol': d['symbol'],
                'date': d['date'],
                'structure': d.get('structure', ''),
                'total_spread_pct': d.get('total_spread_pct', np.nan),
            })
        except Exception:
            continue
    ts_df = pd.DataFrame(records)
    ts_df['date'] = pd.to_datetime(ts_df['date'])
    ts_df = ts_df.sort_values(['symbol', 'date']).reset_index(drop=True)
    print(f"    {len(ts_df):,} TS records loaded in {time.time()-t0:.1f}s", flush=True)

    # Rolling percentile rank: vectorized per symbol
    print("  Computing rolling percentile ranks...", flush=True)
    t1 = time.time()
    spread_pct = ts_df['total_spread_pct'].values
    symbols = ts_df['symbol'].values
    result = np.full(len(ts_df), np.nan)

    # Identify group boundaries
    prev_sym = None
    start_idx = 0
    groups = []
    for i in range(len(symbols)):
        if symbols[i] != prev_sym:
            if prev_sym is not None:
                groups.append((start_idx, i))
            start_idx = i
            prev_sym = symbols[i]
    groups.append((start_idx, len(symbols)))

    window = 252
    min_periods = 63
    for gs, ge in groups:
        vals = spread_pct[gs:ge]
        n = len(vals)
        for i in range(n):
            if i < min_periods - 1:
                continue
            s = max(0, i - window + 1)
            w = vals[s:i+1]
            valid_mask = ~np.isnan(w)
            nv = valid_mask.sum()
            if nv < min_periods or np.isnan(vals[i]):
                continue
            result[gs + i] = np.sum(w[valid_mask] <= vals[i]) / nv * 100

    ts_df['spread_pct_rank'] = result
    print(f"    Percentile ranks computed in {time.time()-t1:.1f}s", flush=True)

    print("  Loading daily price data...", flush=True)
    t2 = time.time()
    daily_data = {}
    # Pre-build: for each symbol, a dict mapping date->row dict for O(1) lookup
    price_lookup = {}  # sym -> {date_str: (close, high, low, ret_5d, vol_ratio)}
    for f in sorted(glob.glob(os.path.join(DAILY_DIR, '*.csv'))):
        sym = os.path.basename(f).replace('.csv', '')
        df = pd.read_csv(f)
        if len(df) < 100:
            continue
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        df = df.sort_values('trade_date').reset_index(drop=True)
        df = df[df['close'].notna() & (df['close'] > 0)].reset_index(drop=True)
        if len(df) < 50:
            continue
        df['ret_5d'] = df['close'].pct_change(5) * 100
        df['vol_ma20'] = df['vol'].rolling(20).mean()
        df['vol_ratio'] = df['vol'] / df['vol_ma20'].replace(0, np.nan)
        df['ret_1d'] = df['close'].pct_change(1) * 100
        daily_data[sym] = df

        # Build fast lookup
        lookup = {}
        for _, row in df.iterrows():
            dt = row['trade_date']
            lookup[dt] = (
                float(row['close']),
                float(row.get('high', row['close'])),
                float(row.get('low', row['close'])),
                float(row.get('ret_5d', np.nan)),
                float(row.get('vol_ratio', np.nan)),
                float(row.get('ret_1d', 0)),
            )
        price_lookup[sym] = lookup

    print(f"    {len(daily_data)} symbols loaded in {time.time()-t2:.1f}s", flush=True)

    common = set(ts_df['symbol'].unique()) & set(daily_data.keys())
    ts_df = ts_df[ts_df['symbol'].isin(common)].reset_index(drop=True)
    daily_data = {k: v for k, v in daily_data.items() if k in common}
    price_lookup = {k: v for k, v in price_lookup.items() if k in common}
    print(f"    {len(common)} common symbols with both datasets", flush=True)

    # Build TS lookup: date -> {sym: spread_pct_rank}
    ts_lookup = defaultdict(dict)
    for _, row in ts_df.iterrows():
        if not np.isnan(row['spread_pct_rank']):
            ts_lookup[row['date']][row['symbol']] = row['spread_pct_rank']

    return ts_df, daily_data, price_lookup, ts_lookup


# ──────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE (optimized with pre-built lookups)
# ──────────────────────────────────────────────────────────────────────
def run_backtest(price_lookup, ts_lookup, start, end,
                 pct_high=90, pct_low=10,
                 n_pos=5, hold=10,
                 sl_pct=-2.0, tp_pct=5.0,
                 use_momentum_filter=False,
                 use_volume_filter=False,
                 lev=LEVERAGE):
    """
    Spread mean-reversion backtest with O(1) lookups.
    """
    dates = pd.bdate_range(start=start, end=end)
    cap = INITIAL_CAPITAL
    equity_curve = []
    all_trades = []
    positions = []  # list of dicts

    for dt in dates:
        pnl = 0.0
        keep = []

        # ── EXIT LOGIC ──
        for p in positions:
            plookup = price_lookup.get(p['sym'], {})
            pdata = plookup.get(dt)
            if pdata is None:
                keep.append(p)
                continue

            cur_c, cur_h, cur_l, _, _, _ = pdata
            if cur_c == 0:
                keep.append(p)
                continue

            days_held = (dt - p['entry_date']).days
            direction = p['direction']

            slippage = 0.001
            if direction == 1:
                raw_ret = (cur_c / p['entry_price'] - 1) * 100
                sl_trigger = (cur_l / p['entry_price'] - 1) * 100
                tp_trigger = (cur_h / p['entry_price'] - 1) * 100
            else:
                raw_ret = (1 - cur_c / p['entry_price']) * 100
                sl_trigger = (1 - cur_h / p['entry_price']) * 100
                tp_trigger = (1 - cur_l / p['entry_price']) * 100

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
                trade_pnl = actual_ret / 100.0 * p['notional'] * lev
                pnl += trade_pnl
                all_trades.append({
                    'symbol': p['sym'],
                    'entry_date': p['entry_date'],
                    'exit_date': dt,
                    'direction': direction,
                    'entry_price': p['entry_price'],
                    'exit_price': cur_c,
                    'return_pct': actual_ret,
                    'pnl': trade_pnl,
                    'exit_reason': exit_reason,
                    'spread_pct_rank': p.get('spread_rank', np.nan),
                    'days_held': days_held,
                })
            else:
                keep.append(p)

        positions = keep

        # ── ENTRY LOGIC ──
        current_syms = {p['sym'] for p in positions}
        n_open = len(positions)
        n_slots = n_pos * 2

        if n_open < n_slots:
            ts_today = ts_lookup.get(dt, {})
            if ts_today:
                candidates = []
                for sym, spr in ts_today.items():
                    if sym in current_syms:
                        continue
                    plookup = price_lookup.get(sym, {})
                    pdata = plookup.get(dt)
                    if pdata is None:
                        continue
                    cur_c, cur_h, cur_l, ret5, vol_ratio, ret1d = pdata
                    if cur_c == 0:
                        continue

                    # Determine direction
                    direction = 0
                    if spr >= pct_high:
                        direction = -1  # SHORT
                    elif spr <= pct_low:
                        direction = 1   # LONG

                    if direction == 0:
                        continue

                    # Momentum filter
                    if use_momentum_filter:
                        if np.isnan(ret5):
                            continue
                        if direction == -1 and ret5 >= 0:
                            continue
                        if direction == 1 and ret5 <= 0:
                            continue

                    # Volume filter
                    if use_volume_filter:
                        if np.isnan(vol_ratio) or vol_ratio < 1.0:
                            continue

                    # Score by extremity
                    if direction == -1:
                        score = spr
                    else:
                        score = 100 - spr

                    candidates.append({
                        'sym': sym,
                        'direction': direction,
                        'price': cur_c,
                        'score': score,
                        'spread_rank': spr,
                    })

                # Sort and select
                long_cands = sorted([c for c in candidates if c['direction'] == 1],
                                    key=lambda x: x['score'], reverse=True)
                short_cands = sorted([c for c in candidates if c['direction'] == -1],
                                     key=lambda x: x['score'], reverse=True)

                n_long_open = sum(1 for p in positions if p['direction'] == 1)
                n_short_open = sum(1 for p in positions if p['direction'] == -1)

                for c in long_cands[:max(0, n_pos - n_long_open)]:
                    positions.append({
                        'sym': c['sym'],
                        'direction': c['direction'],
                        'entry_date': dt,
                        'entry_price': c['price'],
                        'notional': cap / n_slots,
                        'spread_rank': c['spread_rank'],
                    })
                for c in short_cands[:max(0, n_pos - n_short_open)]:
                    positions.append({
                        'sym': c['sym'],
                        'direction': c['direction'],
                        'entry_date': dt,
                        'entry_price': c['price'],
                        'notional': cap / n_slots,
                        'spread_rank': c['spread_rank'],
                    })

        # Update equity
        cap += pnl
        equity_curve.append({'date': dt, 'equity': cap, 'daily_pnl': pnl})

    # Close remaining
    end_ts = pd.Timestamp(end)
    for p in positions:
        plookup = price_lookup.get(p['sym'], {})
        # Find last available price before end
        best_dt = None
        best_pdata = None
        for d in pd.bdate_range(end=end, periods=5):
            pdata = plookup.get(d)
            if pdata:
                best_dt = d
                best_pdata = pdata
                break
        if best_pdata is None:
            continue
        cur_c = best_pdata[0]
        if p['direction'] == 1:
            ret = (cur_c / p['entry_price'] - 1) * 100 - 0.001
        else:
            ret = (1 - cur_c / p['entry_price']) * 100 - 0.001
        trade_pnl = ret / 100.0 * p['notional'] * lev
        all_trades.append({
            'symbol': p['sym'],
            'entry_date': p['entry_date'],
            'exit_date': end_ts,
            'direction': p['direction'],
            'entry_price': p['entry_price'],
            'exit_price': cur_c,
            'return_pct': ret,
            'pnl': trade_pnl,
            'exit_reason': 'END',
            'spread_pct_rank': p.get('spread_rank', np.nan),
            'days_held': (end_ts - p['entry_date']).days,
        })

    eq_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
    return eq_df, trades_df


# ──────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────
def calc_metrics(eq_df, trades_df):
    """Calculate performance metrics."""
    if eq_df.empty or len(eq_df) < 10:
        return None
    eq = eq_df['equity'].values
    total_return = (eq[-1] / eq[0] - 1) * 100

    daily_ret = eq_df['daily_pnl'].values / INITIAL_CAPITAL
    # Remove zeros for sharper Sharpe (non-trading days)
    active_ret = daily_ret[daily_ret != 0] if np.any(daily_ret != 0) else daily_ret
    ann_ret = np.mean(daily_ret) * 252 * 100
    ann_vol = np.std(daily_ret) * np.sqrt(252) * 100
    sharpe = (ann_ret / ann_vol) if ann_vol > 0 else 0

    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    mdd = dd.min()

    n_trades = len(trades_df)
    if n_trades > 0:
        win_trades = trades_df[trades_df['pnl'] > 0]
        lose_trades = trades_df[trades_df['pnl'] <= 0]
        wr = len(win_trades) / n_trades * 100
        avg_win = win_trades['return_pct'].mean() if len(win_trades) > 0 else 0
        avg_loss = lose_trades['return_pct'].mean() if len(lose_trades) > 0 else 0
        avg_ret = trades_df['return_pct'].mean()
        long_trades = trades_df[trades_df['direction'] == 1]
        short_trades = trades_df[trades_df['direction'] == -1]
        long_wr = (len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) * 100) if len(long_trades) > 0 else 0
        short_wr = (len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) * 100) if len(short_trades) > 0 else 0
        sl_count = len(trades_df[trades_df['exit_reason'] == 'SL'])
        tp_count = len(trades_df[trades_df['exit_reason'] == 'TP'])
        hold_count = len(trades_df[trades_df['exit_reason'] == 'HOLD'])
    else:
        wr = avg_win = avg_loss = avg_ret = 0
        long_wr = short_wr = 0
        sl_count = tp_count = hold_count = 0

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
        'sl_count': sl_count,
        'tp_count': tp_count,
        'hold_count': hold_count,
    }


def print_metrics(m, label=""):
    if m is None:
        print(f"  {label}: NO DATA")
        return
    print(f"  {label}")
    print(f"    Return: {m['total_return']:+.2f}% (ann {m['ann_return']:+.2f}%)  "
          f"Vol: {m['ann_vol']:.2f}%  Sharpe: {m['sharpe']:.2f}")
    print(f"    MDD: {m['mdd']:.2f}%  Trades: {m['n_trades']}  WR: {m['win_rate']:.1f}%  "
          f"AvgWin: {m['avg_win']:+.2f}%  AvgLoss: {m['avg_loss']:+.2f}%")
    print(f"    Long WR: {m['long_wr']:.1f}%  Short WR: {m['short_wr']:.1f}%  "
          f"AvgRet: {m['avg_ret']:+.3f}%")
    print(f"    Exits -> SL: {m['sl_count']}  TP: {m['tp_count']}  Hold: {m['hold_count']}")


# ──────────────────────────────────────────────────────────────────────
# PARAMETER SWEEP
# ──────────────────────────────────────────────────────────────────────
def parameter_sweep(price_lookup, ts_lookup, start, end, label="TRAIN"):
    thresholds = [(80, 20), (85, 15), (90, 10), (95, 5)]
    holds = [5, 10, 15, 20]
    n_positions = [5, 10]
    filters = [
        ('none', False, False),
        ('mom', True, False),
        ('vol', False, True),
        ('mom+vol', True, True),
    ]

    results = []
    total_combos = len(thresholds) * len(holds) * len(n_positions) * len(filters)
    count = 0

    for (pct_h, pct_l), hold, n_pos, (filt_name, use_mom, use_vol) in product(
            thresholds, holds, n_positions, filters):
        count += 1
        if count % 20 == 0 or count == total_combos:
            print(f"    [{label}] {count}/{total_combos} combinations...", flush=True)

        eq_df, trades_df = run_backtest(
            price_lookup, ts_lookup, start, end,
            pct_high=pct_h, pct_low=pct_l,
            n_pos=n_pos, hold=hold,
            sl_pct=-2.0, tp_pct=5.0,
            use_momentum_filter=use_mom,
            use_volume_filter=use_vol,
        )
        m = calc_metrics(eq_df, trades_df)
        if m is not None:
            m['pct_high'] = pct_h
            m['pct_low'] = pct_l
            m['hold'] = hold
            m['n_pos'] = n_pos
            m['filter'] = filt_name
            m['use_mom'] = use_mom
            m['use_vol'] = use_vol
            results.append(m)

    print(f"    [{label}] Done. {total_combos} combinations tested.", flush=True)
    return sorted(results, key=lambda x: x['sharpe'], reverse=True)


# ──────────────────────────────────────────────────────────────────────
# WALK-FORWARD
# ──────────────────────────────────────────────────────────────────────
def walk_forward(price_lookup, ts_lookup):
    print("\n" + "=" * 80)
    print("WALK-FORWARD ANALYSIS")
    print("=" * 80)

    # TRAINING
    print("\n[1/3] TRAINING: 2021-01-01 to 2023-12-31")
    print("-" * 60)
    train_results = parameter_sweep(price_lookup, ts_lookup, '2021-01-01', '2023-12-31', 'TRAIN')

    print(f"\n  Top 10 Training Configurations:")
    print(f"  {'Rank':<5} {'PctH':>5} {'PctL':>5} {'Hold':>5} {'NPos':>5} {'Filter':<10} "
          f"{'Sharpe':>7} {'Ret':>8} {'MDD':>7} {'WR':>6} {'Trades':>7}")
    print("  " + "-" * 85)

    top_n = min(10, len(train_results))
    for i, m in enumerate(train_results[:top_n]):
        print(f"  {i+1:<5} {m['pct_high']:>5} {m['pct_low']:>5} {m['hold']:>5} {m['n_pos']:>5} "
              f"{m['filter']:<10} {m['sharpe']:>7.2f} {m['ann_return']:>+7.2f}% "
              f"{m['mdd']:>7.2f} {m['win_rate']:>5.1f}% {m['n_trades']:>7}")

    # VALIDATION
    print(f"\n[2/3] VALIDATION: 2024-01-01 to 2024-12-31")
    print("-" * 60)
    print("  Testing top 10 training configs on validation set...", flush=True)

    val_results = []
    for m in train_results[:top_n]:
        eq_df, trades_df = run_backtest(
            price_lookup, ts_lookup, '2024-01-01', '2024-12-31',
            pct_high=m['pct_high'], pct_low=m['pct_low'],
            n_pos=m['n_pos'], hold=m['hold'],
            sl_pct=-2.0, tp_pct=5.0,
            use_momentum_filter=m['use_mom'],
            use_volume_filter=m['use_vol'],
        )
        vm = calc_metrics(eq_df, trades_df)
        if vm is not None:
            vm['pct_high'] = m['pct_high']
            vm['pct_low'] = m['pct_low']
            vm['hold'] = m['hold']
            vm['n_pos'] = m['n_pos']
            vm['filter'] = m['filter']
            vm['use_mom'] = m['use_mom']
            vm['use_vol'] = m['use_vol']
            vm['train_sharpe'] = m['sharpe']
            val_results.append(vm)

    val_results.sort(key=lambda x: x['sharpe'], reverse=True)

    print(f"\n  Validation Results (sorted by Sharpe):")
    print(f"  {'Rank':<5} {'PctH':>5} {'PctL':>5} {'Hold':>5} {'NPos':>5} {'Filter':<10} "
          f"{'ValShr':>7} {'TrnShr':>7} {'Ret':>8} {'MDD':>7} {'WR':>6} {'Trades':>7}")
    print("  " + "-" * 100)
    for i, m in enumerate(val_results[:10]):
        print(f"  {i+1:<5} {m['pct_high']:>5} {m['pct_low']:>5} {m['hold']:>5} {m['n_pos']:>5} "
              f"{m['filter']:<10} {m['sharpe']:>7.2f} {m['train_sharpe']:>7.2f} "
              f"{m['ann_return']:>+7.2f}% {m['mdd']:>7.2f} {m['win_rate']:>5.1f}% {m['n_trades']:>7}")

    # TEST
    if not val_results:
        print("\n  No valid configurations found. Aborting.")
        return None, None

    best = val_results[0]
    print(f"\n[3/3] OUT-OF-SAMPLE TEST: 2025-01-01 to 2026-05-21")
    print("-" * 60)
    print(f"  Selected config: PctH={best['pct_high']} PctL={best['pct_low']} "
          f"Hold={best['hold']} NPos={best['n_pos']} Filter={best['filter']}")

    eq_df, trades_df = run_backtest(
        price_lookup, ts_lookup, '2025-01-01', '2026-05-21',
        pct_high=best['pct_high'], pct_low=best['pct_low'],
        n_pos=best['n_pos'], hold=best['hold'],
        sl_pct=-2.0, tp_pct=5.0,
        use_momentum_filter=best['use_mom'],
        use_volume_filter=best['use_vol'],
    )
    test_m = calc_metrics(eq_df, trades_df)
    print(f"\n  OUT-OF-SAMPLE RESULTS:")
    print_metrics(test_m, "TEST 2025-2026")

    # Top 3 OOS comparison
    if len(val_results) >= 3:
        print(f"\n  --- Top 3 Configs OOS Comparison ---")
        print(f"  {'#':<3} {'PctH':>5} {'PctL':>5} {'H':>3} {'N':>3} {'Filter':<10} "
              f"{'Sharpe':>7} {'Ret':>8} {'MDD':>7} {'WR':>6} {'Trades':>7}")
        print("  " + "-" * 80)
        for i, cfg in enumerate(val_results[:3]):
            eq_t, tr_t = run_backtest(
                price_lookup, ts_lookup, '2025-01-01', '2026-05-21',
                pct_high=cfg['pct_high'], pct_low=cfg['pct_low'],
                n_pos=cfg['n_pos'], hold=cfg['hold'],
                sl_pct=-2.0, tp_pct=5.0,
                use_momentum_filter=cfg['use_mom'],
                use_volume_filter=cfg['use_vol'],
            )
            tm = calc_metrics(eq_t, tr_t)
            if tm:
                print(f"  {i+1:<3} {cfg['pct_high']:>5} {cfg['pct_low']:>5} {cfg['hold']:>3} "
                      f"{cfg['n_pos']:>3} {cfg['filter']:<10} {tm['sharpe']:>7.2f} "
                      f"{tm['ann_return']:>+7.2f}% {tm['mdd']:>7.2f} "
                      f"{tm['win_rate']:>5.1f}% {tm['n_trades']:>7}")

    # Detailed analysis
    print(f"\n  --- Detailed Analysis of Best OOS Config ---")
    if trades_df is not None and len(trades_df) > 0:
        long_t = trades_df[trades_df['direction'] == 1]
        short_t = trades_df[trades_df['direction'] == -1]
        print(f"  Direction breakdown:")
        print(f"    LONG:  {len(long_t):>4} trades  "
              f"WR: {(len(long_t[long_t['pnl']>0])/max(len(long_t),1))*100:.1f}%  "
              f"AvgRet: {long_t['return_pct'].mean():+.3f}%")
        print(f"    SHORT: {len(short_t):>4} trades  "
              f"WR: {(len(short_t[short_t['pnl']>0])/max(len(short_t),1))*100:.1f}%  "
              f"AvgRet: {short_t['return_pct'].mean():+.3f}%")

        print(f"\n  Exit reason breakdown:")
        for reason in ['SL', 'TP', 'HOLD', 'END']:
            rt = trades_df[trades_df['exit_reason'] == reason]
            if len(rt) > 0:
                print(f"    {reason:5s}: {len(rt):>4} trades  AvgRet: {rt['return_pct'].mean():+.3f}%")

        # Monthly
        trades_df['exit_month'] = pd.to_datetime(trades_df['exit_date']).dt.to_period('M')
        monthly = trades_df.groupby('exit_month').agg(
            n_trades=('pnl', 'count'),
            total_pnl=('pnl', 'sum'),
            avg_ret=('return_pct', 'mean'),
        )
        print(f"\n  Monthly performance:")
        print(f"  {'Month':<10} {'Trades':>7} {'PnL':>12} {'AvgRet':>8}")
        for month, row in monthly.iterrows():
            print(f"  {str(month):<10} {row['n_trades']:>7} {row['total_pnl']:>12,.0f} {row['avg_ret']:>+7.3f}%")

    return val_results, test_m


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 80)
    print("V98: TERM STRUCTURE SPREAD MEAN-REVERSION STRATEGY")
    print("    Spread percentile rank -> expect mean reversion")
    print("=" * 80, flush=True)

    ts_df, daily_data, price_lookup, ts_lookup = load_all_data()
    t1 = time.time()
    print(f"\n  Data loading + prep: {t1-t0:.1f}s\n", flush=True)

    # BASELINE
    print("=" * 60)
    print("BASELINE: 90/10 threshold, hold=10, n_pos=5 (Full 2021-2026)")
    print("=" * 60)
    for filt_name, use_mom, use_vol in [('none', False, False),
                                         ('momentum', True, False),
                                         ('volume', False, True),
                                         ('mom+vol', True, True)]:
        eq_df, trades_df = run_backtest(
            price_lookup, ts_lookup, '2021-01-01', '2026-05-21',
            pct_high=90, pct_low=10,
            n_pos=5, hold=10,
            sl_pct=-2.0, tp_pct=5.0,
            use_momentum_filter=use_mom,
            use_volume_filter=use_vol,
        )
        m = calc_metrics(eq_df, trades_df)
        print_metrics(m, f"Filter={filt_name}")

    # WALK-FORWARD
    val_results, test_m = walk_forward(price_lookup, ts_lookup)

    # SUMMARY
    t2 = time.time()
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n  Total runtime: {t2-t0:.1f}s")
    print(f"\n  Strategy: Spread Mean-Reversion")
    print(f"  Signal: Rolling 252d percentile rank of total_spread_pct")
    print(f"  Risk: -2% SL, +5% TP, {LEVERAGE}x leverage")
    print(f"\n  Walk-forward periods:")
    print(f"    Train:      2021-01-01 to 2023-12-31")
    print(f"    Validate:   2024-01-01 to 2024-12-31")
    print(f"    Test (OOS): 2025-01-01 to 2026-05-21")

    if val_results:
        best = val_results[0]
        print(f"\n  Best validated config:")
        print(f"    Thresholds: {best['pct_high']}/{best['pct_low']}")
        print(f"    Hold: {best['hold']}d  Positions: {best['n_pos']}/side")
        print(f"    Filter: {best['filter']}")
        print(f"    Train Sharpe: {best['train_sharpe']:.2f}")
        print(f"    Valid Sharpe: {best['sharpe']:.2f}")
        if test_m:
            print(f"    Test Sharpe:  {test_m['sharpe']:.2f}")
            print(f"    Test Return:  {test_m['ann_return']:+.2f}%")
            print(f"    Test MDD:     {test_m['mdd']:.2f}%")
            print(f"    Test WR:      {test_m['win_rate']:.1f}%")
            print(f"    Test Trades:  {test_m['n_trades']}")


if __name__ == '__main__':
    main()
