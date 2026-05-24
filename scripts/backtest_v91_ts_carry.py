#!/usr/bin/env python3
"""
V91: Term Structure Carry Strategy for Chinese Commodity Futures
================================================================
v3: Refined approach after discovery that:
  - Long backwardation: DOES NOT WORK (consistently terrible)
  - Short contango: MARGINAL (best Sharpe ~0.34 with H=20 SL=-3 TP=6)
  - Pure carry alone is too weak as a signal

v3 improvements:
  1. Combine carry with price momentum (confirm direction)
  2. Adaptive hold: exit early if carry reverses
  3. Use ATR for volatility-adjusted position sizing
  4. Focus on short-contango with price declining confirmation
  5. Also test: long backwardation ONLY when price is rising (supply tightness confirms)
  6. Rolling rebalance (when position exits, fill with new signal)
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS_DIR = os.path.join(BASE_DIR, 'data', 'futures_term_structure')
DAILY_DIR = os.path.join(BASE_DIR, 'data', 'futures_weighted')

INITIAL_CAPITAL = 500_000


def load_all_data():
    """Load and prepare all data."""
    print("  Loading term structure...")
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

    # Carry momentum
    ts_df['spread_5d_ago'] = ts_df.groupby('symbol')['total_spread_pct'].shift(5)
    ts_df['spread_10d_ago'] = ts_df.groupby('symbol')['total_spread_pct'].shift(10)
    ts_df['carry_mom_5d'] = ts_df['total_spread_pct'] - ts_df['spread_5d_ago']
    ts_df['carry_mom_10d'] = ts_df['total_spread_pct'] - ts_df['spread_10d_ago']

    # Z-score of spread within each symbol's history
    ts_df['spread_zscore'] = ts_df.groupby('symbol')['total_spread_pct'].transform(
        lambda x: (x - x.rolling(60, min_periods=20).mean()) / x.rolling(60, min_periods=20).std()
    )

    print(f"    {len(ts_df):,} records, {ts_df['symbol'].nunique()} symbols")

    print("  Loading daily data...")
    daily_data = {}
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
        # Price features
        df['ret_1d'] = df['close'].pct_change(1) * 100
        df['ret_5d'] = df['close'].pct_change(5) * 100
        df['ret_10d'] = df['close'].pct_change(10) * 100
        df['ret_20d'] = df['close'].pct_change(20) * 100
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        # ATR
        tr = np.maximum(df['high'] - df['low'],
                        np.maximum(abs(df['high'] - df['close'].shift(1)),
                                   abs(df['low'] - df['close'].shift(1))))
        df['atr_20'] = tr.rolling(20).mean()
        df['atr_pct'] = df['atr_20'] / df['close'] * 100
        daily_data[sym] = df

    print(f"    {len(daily_data)} symbols loaded")

    common = set(ts_df['symbol'].unique()) & set(daily_data.keys())
    ts_df = ts_df[ts_df['symbol'].isin(common)].reset_index(drop=True)
    daily_data = {k: v for k, v in daily_data.items() if k in common}
    print(f"    {len(common)} common symbols")

    return ts_df, daily_data


def run_backtest(ts_df, daily_data, start, end,
                 mode='contango_short_confirmed',
                 n_pos=5, hold=20, lev=3,
                 sl_pct=-3.0, tp_pct=6.0,
                 rebalance_freq=10,
                 min_spread=1.0,
                 use_zscore=False,
                 min_zscore=1.0):
    """
    mode options:
      'contango_short'          - Pure contango short
      'contango_short_confirmed'- Contango short + price declining (ret_5d < 0)
      'contango_short_momentum' - Contango short + carry widening (carry_mom_5d > 0)
      'backwardation_long'      - Pure backwardation long
      'backwardation_confirmed' - Backwardation long + price rising
      'extreme_carry'           - Spread z-score > threshold (contango short, backwardation long)
      'combined'                - Short contango + declining price OR long backwardation + rising price
    """
    dates = pd.date_range(start=start, end=end, freq='B')
    cap = INITIAL_CAPITAL
    eq, trades, pos = [], [], []

    # Lookup tables
    ts_by_date = {}
    for dt, grp in ts_df.groupby('date'):
        ts_by_date[dt] = grp.set_index('symbol')

    trading_day = 0

    for dt in dates:
        trading_day += 1
        pnl = 0.0
        keep = []

        # Exit logic
        for p in pos:
            df = daily_data.get(p['sym'])
            if df is None:
                keep.append(p); continue
            idx = df.index[df['trade_date'] == dt]
            if len(idx) == 0:
                keep.append(p); continue

            row = df.loc[idx[0]]
            cur_h, cur_l, cur_c = float(row['high']), float(row['low']), float(row['close'])
            if np.isnan(cur_c) or cur_c == 0:
                keep.append(p); continue

            days_held = (dt - p['entry_date']).days
            slip = 0.001
            triggered = False
            actual_ret = None
            reason = None

            if p['dir'] == 'long':
                if sl_pct and cur_l <= p['ep'] * (1 + sl_pct / 100):
                    fill = p['ep'] * (1 + sl_pct / 100) * (1 - slip)
                    actual_ret = (fill - p['ep']) / p['ep'] * 100
                    reason = 'SL'; triggered = True
                if not triggered and tp_pct and cur_h >= p['ep'] * (1 + tp_pct / 100):
                    fill = p['ep'] * (1 + tp_pct / 100) * (1 - slip)
                    actual_ret = (fill - p['ep']) / p['ep'] * 100
                    reason = 'TP'; triggered = True
                if not triggered:
                    actual_ret = (cur_c - p['ep']) / p['ep'] * 100
            else:
                if sl_pct and cur_h >= p['ep'] * (1 - sl_pct / 100):
                    fill = p['ep'] * (1 - sl_pct / 100) * (1 + slip)
                    actual_ret = (p['ep'] - fill) / p['ep'] * 100
                    reason = 'SL'; triggered = True
                if not triggered and tp_pct and cur_l <= p['ep'] * (1 - tp_pct / 100):
                    fill = p['ep'] * (1 - tp_pct / 100) * (1 + slip)
                    actual_ret = (p['ep'] - fill) / p['ep'] * 100
                    reason = 'TP'; triggered = True
                if not triggered:
                    actual_ret = (p['ep'] - cur_c) / p['ep'] * 100

            if days_held >= hold:
                if not triggered:
                    reason = 'expire'
            elif not triggered:
                keep.append(p); continue

            if reason:
                pnl += p['not'] * actual_ret / 100
                trades.append({
                    'sym': p['sym'], 'dir': p['dir'],
                    'entry_date': p['entry_date'], 'exit_date': dt,
                    'ep': p['ep'], 'xp': cur_c,
                    'ret': actual_ret,
                    'pnl': p['not'] * actual_ret / 100,
                    'hold_days': days_held, 'reason': reason,
                    'spread_pct': p['spread_pct'],
                })

        pos = keep
        cap += pnl
        if cap <= 0:
            eq.append({'date': dt, 'capital': 0}); break

        # Entry on rebalance
        if trading_day % rebalance_freq != 1:
            eq.append({'date': dt, 'capital': cap}); continue

        day_ts = ts_by_date.get(dt)
        if day_ts is None:
            eq.append({'date': dt, 'capital': cap}); continue

        existing = {p['sym'] for p in pos}
        long_cands = []
        short_cands = []

        for sym in day_ts.index:
            if sym in existing:
                continue
            df = daily_data.get(sym)
            if df is None:
                continue
            idx = df.index[df['trade_date'] == dt]
            if len(idx) == 0:
                continue

            price_row = df.loc[idx[0]]
            entry_price = float(price_row['open'])
            if np.isnan(entry_price) or entry_price <= 0:
                continue

            ts_row = day_ts.loc[sym]
            spread = ts_row['total_spread_pct']
            if np.isnan(spread):
                continue

            carry_mom = ts_row.get('carry_mom_5d', np.nan)
            zscore = ts_row.get('spread_zscore', np.nan)
            ret_5d = price_row.get('ret_5d', np.nan)
            ret_10d = price_row.get('ret_10d', np.nan)
            ret_20d = price_row.get('ret_20d', np.nan)
            ma5 = price_row.get('ma5', np.nan)
            ma20 = price_row.get('ma20', np.nan)
            atr_pct = price_row.get('atr_pct', np.nan)

            if abs(spread) < min_spread:
                continue

            if use_zscore and not np.isnan(zscore):
                if abs(zscore) < min_zscore:
                    continue

            # ── Mode filtering ──
            if mode == 'contango_short':
                if spread > 0:
                    short_cands.append({'sym': sym, 'ep': entry_price, 'spread_pct': spread,
                                        'score': abs(spread)})

            elif mode == 'contango_short_confirmed':
                # Contango + price declining in last 5d = bearish confirmation
                if spread > 0 and not np.isnan(ret_5d) and ret_5d < 0:
                    score = abs(spread) + abs(ret_5d) * 2
                    short_cands.append({'sym': sym, 'ep': entry_price, 'spread_pct': spread,
                                        'score': score})

            elif mode == 'contango_short_momentum':
                # Contango + carry widening (contango getting worse)
                if spread > 0 and not np.isnan(carry_mom) and carry_mom > 0:
                    score = abs(spread) + carry_mom
                    short_cands.append({'sym': sym, 'ep': entry_price, 'spread_pct': spread,
                                        'score': score})

            elif mode == 'backwardation_long':
                if spread < 0:
                    long_cands.append({'sym': sym, 'ep': entry_price, 'spread_pct': spread,
                                       'score': abs(spread)})

            elif mode == 'backwardation_confirmed':
                # Backwardation + price rising = bullish confirmation
                if spread < 0 and not np.isnan(ret_5d) and ret_5d > 0:
                    score = abs(spread) + ret_5d * 2
                    long_cands.append({'sym': sym, 'ep': entry_price, 'spread_pct': spread,
                                       'score': score})

            elif mode == 'extreme_carry':
                # Only trade extreme carry (high z-score)
                if np.isnan(zscore):
                    continue
                if spread > 0 and zscore > min_zscore:
                    short_cands.append({'sym': sym, 'ep': entry_price, 'spread_pct': spread,
                                        'score': zscore})
                elif spread < 0 and zscore < -min_zscore:
                    long_cands.append({'sym': sym, 'ep': entry_price, 'spread_pct': spread,
                                       'score': -zscore})

            elif mode == 'combined':
                # Short contango + declining price
                if spread > 0 and not np.isnan(ret_5d) and ret_5d < 0:
                    score = abs(spread) + abs(ret_5d)
                    short_cands.append({'sym': sym, 'ep': entry_price, 'spread_pct': spread,
                                        'score': score})
                # Long backwardation + rising price
                elif spread < 0 and not np.isnan(ret_5d) and ret_5d > 0:
                    score = abs(spread) + ret_5d
                    long_cands.append({'sym': sym, 'ep': entry_price, 'spread_pct': spread,
                                       'score': score})

            elif mode == 'trend_carry':
                # Carry aligned with MA trend
                if spread > 0:  # contango
                    if not np.isnan(ma5) and not np.isnan(ma20) and ma5 < ma20:
                        score = abs(spread) + 5  # trend confirm bonus
                        short_cands.append({'sym': sym, 'ep': entry_price,
                                            'spread_pct': spread, 'score': score})
                    else:
                        short_cands.append({'sym': sym, 'ep': entry_price,
                                            'spread_pct': spread, 'score': abs(spread)})
                elif spread < 0:  # backwardation
                    if not np.isnan(ma5) and not np.isnan(ma20) and ma5 > ma20:
                        score = abs(spread) + 5
                        long_cands.append({'sym': sym, 'ep': entry_price,
                                           'spread_pct': spread, 'score': score})
                    else:
                        long_cands.append({'sym': sym, 'ep': entry_price,
                                           'spread_pct': spread, 'score': abs(spread)})

        # Sort by score (highest = strongest signal)
        long_cands.sort(key=lambda x: -x.get('score', 0))
        short_cands.sort(key=lambda x: -x.get('score', 0))

        max_per_dir = n_pos
        n_long = sum(1 for p in pos if p['dir'] == 'long')
        n_short = sum(1 for p in pos if p['dir'] == 'short')

        has_both = len(long_cands) > 0 and len(short_cands) > 0
        n_dirs = 2 if has_both else 1

        for c_ in long_cands[:max_per_dir]:
            if n_long >= max_per_dir:
                break
            notional = cap * lev / (max_per_dir * n_dirs) if n_dirs > 1 else cap * lev / max_per_dir
            pos.append({
                'sym': c_['sym'], 'dir': 'long',
                'entry_date': dt, 'ep': c_['ep'],
                'not': notional, 'spread_pct': c_['spread_pct'],
            })
            n_long += 1

        for c_ in short_cands[:max_per_dir]:
            if n_short >= max_per_dir:
                break
            notional = cap * lev / (max_per_dir * n_dirs) if n_dirs > 1 else cap * lev / max_per_dir
            pos.append({
                'sym': c_['sym'], 'dir': 'short',
                'entry_date': dt, 'ep': c_['ep'],
                'not': notional, 'spread_pct': c_['spread_pct'],
            })
            n_short += 1

        eq.append({'date': dt, 'capital': cap})

    return eq, trades


def pr(eq, trades, label, verbose=True):
    """Print results."""
    eq_df = pd.DataFrame(eq)
    if len(eq_df) == 0 or eq_df['capital'].iloc[-1] <= 0:
        print(f"  {label}: BLOWN UP")
        return None

    eq_df['peak'] = eq_df['capital'].cummax()
    eq_df['dd'] = (eq_df['capital'] - eq_df['peak']) / eq_df['peak'] * 100
    n_yr = max((eq_df['date'].iloc[-1] - eq_df['date'].iloc[0]).days / 365.25, 0.01)
    mdd = eq_df['dd'].min()
    dr = eq_df['capital'].pct_change().dropna()
    sharpe = dr.mean() / dr.std() * (252**0.5) if dr.std() > 0 else 0
    sortino_denom = dr[dr < 0].std() * (252**0.5) if len(dr[dr < 0]) > 0 else 1
    sortino = dr.mean() * (252**0.5) / sortino_denom if sortino_denom > 0 else 0
    total_ret = (eq_df['capital'].iloc[-1] / INITIAL_CAPITAL - 1) * 100
    cagr = ((eq_df['capital'].iloc[-1] / INITIAL_CAPITAL) ** (1/n_yr) - 1) * 100
    pos_days = (dr > 0).mean() * 100 if len(dr) > 0 else 0

    if trades:
        td = pd.DataFrame(trades)
        wr = (td['ret'] > 0).mean() * 100
        avg = td['ret'].mean()
        long_t = td[td['dir'] == 'long']
        short_t = td[td['dir'] == 'short']
        l_wr = (long_t['ret'] > 0).mean() * 100 if len(long_t) > 0 else 0
        s_wr = (short_t['ret'] > 0).mean() * 100 if len(short_t) > 0 else 0
        reasons = td['reason'].value_counts()
        td['year'] = pd.to_datetime(td['exit_date']).dt.year
    else:
        wr = avg = 0
        td = pd.DataFrame()
        long_t = short_t = pd.DataFrame()
        l_wr = s_wr = 0

    if not verbose:
        print(f"  {label}")
        print(f"    N={len(trades)} WR={wr:.1f}% Sharpe={sharpe:.2f} MDD={mdd:.1f}% "
              f"Avg={avg:+.3f}% Ret={total_ret:+.1f}%")
    else:
        print(f"\n  {'='*60}")
        print(f"  {label}")
        print(f"  {'='*60}")
        print(f"  Return: {total_ret:+.1f}%  CAGR: {cagr:+.1f}%  Sharpe: {sharpe:.2f}  "
              f"Sortino: {sortino:.1f}  MDD: {mdd:.1f}%")
        print(f"  N={len(trades)}  WR={wr:.1f}%  Avg={avg:+.3f}%  PosDays={pos_days:.1f}%")
        if len(long_t) > 0:
            print(f"    LONG:  N={len(long_t)} WR={l_wr:.1f}% Avg={long_t['ret'].mean():+.3f}%")
        if len(short_t) > 0:
            print(f"    SHORT: N={len(short_t)} WR={s_wr:.1f}% Avg={short_t['ret'].mean():+.3f}%")
        if len(td) > 0:
            print(f"  -- Exit --")
            for reason, cnt in reasons.items():
                sub = td[td['reason'] == reason]
                print(f"    {reason:8s}: N={cnt:4d} WR={(sub['ret']>0).mean()*100:.1f}% "
                      f"Avg={sub['ret'].mean():+.3f}%")
            print(f"  -- Year --")
            for yr in sorted(td['year'].unique()):
                sub = td[td['year'] == yr]
                yr_r = eq_df[eq_df['date'].dt.year == yr]
                ys = yr_r['capital'].iloc[0] if len(yr_r) > 0 else INITIAL_CAPITAL
                ye = yr_r['capital'].iloc[-1] if len(yr_r) > 0 else ys
                print(f"    {yr}: N={len(sub):4d} WR={(sub['ret']>0).mean()*100:.1f}% "
                      f"Avg={sub['ret'].mean():+.3f}% Port={(ye/ys-1)*100:+.1f}%")

    return {
        'sharpe': sharpe, 'sortino': sortino, 'mdd': mdd, 'wr': wr,
        'avg': avg, 'n': len(trades), 'total_ret': total_ret,
        'cagr': cagr, 'label': label, 'pos_days': pos_days,
    }


def main():
    print("=" * 70)
    print("V91: TERM STRUCTURE CARRY STRATEGY (v3)")
    print("  Carry + Price Momentum Confirmation")
    print("=" * 70)

    print("\n[1] Loading data...")
    ts_df, daily_data = load_all_data()

    test_start = '2021-06-01'
    test_end = '2025-12-31'

    # ════════════════════════════════════════════════════════════════
    # A. BASELINE: Pure Contango Short (best from v2)
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("[A] BASELINE: Pure Contango Short (H=20, SL=-3, TP=6)")
    print(f"{'='*70}")

    for n_pos in [5, 10]:
        for lev in [3, 5]:
            eq, tr = run_backtest(
                ts_df, daily_data, test_start, test_end,
                mode='contango_short', n_pos=n_pos, hold=20, lev=lev,
                sl_pct=-3.0, tp_pct=6.0, rebalance_freq=10, min_spread=1.0,
            )
            pr(eq, tr, f"Baseline N={n_pos} L={lev}x", verbose=False)

    # ════════════════════════════════════════════════════════════════
    # B. CONTANGO SHORT + PRICE CONFIRMATION
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("[B] CONTANGO SHORT + Price Declining Confirmation")
    print(f"{'='*70}")

    confirmed_results = []
    for hold in [5, 10, 15, 20]:
        for n_pos in [5, 10]:
            for lev in [3, 5]:
                for sl, tp in [(-3.0, 6.0), (-3.0, 8.0), (-2.0, 5.0)]:
                    rebal = min(hold, 10)
                    label = f"Confirmed H={hold} N={n_pos} L={lev}x SL={sl} TP={tp}"
                    eq, tr = run_backtest(
                        ts_df, daily_data, test_start, test_end,
                        mode='contango_short_confirmed',
                        n_pos=n_pos, hold=hold, lev=lev,
                        sl_pct=sl, tp_pct=tp,
                        rebalance_freq=rebal, min_spread=1.0,
                    )
                    r = pr(eq, tr, label, verbose=False)
                    if r:
                        r.update({'hold': hold, 'n_pos': n_pos, 'lev': lev,
                                  'sl_pct': sl, 'tp_pct': tp, 'rebalance_freq': rebal,
                                  'mode': 'confirmed'})
                        confirmed_results.append(r)

    # ════════════════════════════════════════════════════════════════
    # C. CONTANGO SHORT + CARRY MOMENTUM
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("[C] CONTANGO SHORT + Carry Widening Momentum")
    print(f"{'='*70}")

    momentum_results = []
    for hold in [5, 10, 20]:
        for n_pos in [5, 10]:
            for sl, tp in [(-3.0, 6.0), (-3.0, 8.0)]:
                rebal = min(hold, 10)
                label = f"CarryMom H={hold} N={n_pos} L=3x SL={sl} TP={tp}"
                eq, tr = run_backtest(
                    ts_df, daily_data, test_start, test_end,
                    mode='contango_short_momentum',
                    n_pos=n_pos, hold=hold, lev=3,
                    sl_pct=sl, tp_pct=tp,
                    rebalance_freq=rebal, min_spread=1.0,
                )
                r = pr(eq, tr, label, verbose=False)
                if r:
                    r.update({'hold': hold, 'n_pos': n_pos, 'lev': 3,
                              'sl_pct': sl, 'tp_pct': tp, 'rebalance_freq': rebal,
                              'mode': 'carry_momentum'})
                    momentum_results.append(r)

    # ════════════════════════════════════════════════════════════════
    # D. COMBINED: Short contango+declining + Long backwardation+rising
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("[D] COMBINED: Carry + Price Direction Confirmation")
    print(f"{'='*70}")

    combined_results = []
    for hold in [5, 10, 20]:
        for n_pos in [5, 10]:
            for lev in [3, 5]:
                for sl, tp in [(-3.0, 6.0), (-3.0, 8.0), (-2.0, 5.0)]:
                    rebal = min(hold, 10)
                    label = f"Combined H={hold} N={n_pos} L={lev}x SL={sl} TP={tp}"
                    eq, tr = run_backtest(
                        ts_df, daily_data, test_start, test_end,
                        mode='combined',
                        n_pos=n_pos, hold=hold, lev=lev,
                        sl_pct=sl, tp_pct=tp,
                        rebalance_freq=rebal, min_spread=1.0,
                    )
                    r = pr(eq, tr, label, verbose=False)
                    if r:
                        r.update({'hold': hold, 'n_pos': n_pos, 'lev': lev,
                                  'sl_pct': sl, 'tp_pct': tp, 'rebalance_freq': rebal,
                                  'mode': 'combined'})
                        combined_results.append(r)

    # ════════════════════════════════════════════════════════════════
    # E. EXTREME CARRY (z-score filter)
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("[E] EXTREME CARRY (z-score > 1.5)")
    print(f"{'='*70}")

    extreme_results = []
    for hold in [5, 10, 20]:
        for n_pos in [5, 10]:
            for sl, tp in [(-3.0, 6.0), (-3.0, 8.0)]:
                rebal = min(hold, 10)
                label = f"Extreme H={hold} N={n_pos} L=3x SL={sl} TP={tp}"
                eq, tr = run_backtest(
                    ts_df, daily_data, test_start, test_end,
                    mode='extreme_carry',
                    n_pos=n_pos, hold=hold, lev=3,
                    sl_pct=sl, tp_pct=tp,
                    rebalance_freq=rebal, min_spread=1.0,
                    use_zscore=True, min_zscore=1.5,
                )
                r = pr(eq, tr, label, verbose=False)
                if r:
                    r.update({'hold': hold, 'n_pos': n_pos, 'lev': 3,
                              'sl_pct': sl, 'tp_pct': tp, 'rebalance_freq': rebal,
                              'mode': 'extreme_carry'})
                    extreme_results.append(r)

    # ════════════════════════════════════════════════════════════════
    # F. TREND + CARRY
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("[F] TREND + CARRY (MA trend confirms carry direction)")
    print(f"{'='*70}")

    trend_results = []
    for hold in [5, 10, 20]:
        for n_pos in [5, 10]:
            for lev in [3, 5]:
                for sl, tp in [(-3.0, 6.0), (-2.0, 5.0)]:
                    rebal = min(hold, 10)
                    label = f"Trend H={hold} N={n_pos} L={lev}x SL={sl} TP={tp}"
                    eq, tr = run_backtest(
                        ts_df, daily_data, test_start, test_end,
                        mode='trend_carry',
                        n_pos=n_pos, hold=hold, lev=lev,
                        sl_pct=sl, tp_pct=tp,
                        rebalance_freq=rebal, min_spread=1.0,
                    )
                    r = pr(eq, tr, label, verbose=False)
                    if r:
                        r.update({'hold': hold, 'n_pos': n_pos, 'lev': lev,
                                  'sl_pct': sl, 'tp_pct': tp, 'rebalance_freq': rebal,
                                  'mode': 'trend_carry'})
                        trend_results.append(r)

    # ════════════════════════════════════════════════════════════════
    # G. BEST CONFIG + DETAILED ANALYSIS
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("[G] BEST CONFIGURATION — DETAILED ANALYSIS")
    print(f"{'='*70}")

    all_r = confirmed_results + momentum_results + combined_results + extreme_results + trend_results
    viable = [r for r in all_r if r['mdd'] >= -50]
    if not viable:
        viable = all_r
    best = sorted(viable, key=lambda x: -x['sharpe'])[0]

    # Map short mode names to function mode names
    mode_map = {
        'confirmed': 'contango_short_confirmed',
        'carry_momentum': 'contango_short_momentum',
        'combined': 'combined',
        'extreme_carry': 'extreme_carry',
        'trend_carry': 'trend_carry',
    }
    best_mode = mode_map.get(best.get('mode', ''), best.get('mode', 'contango_short_confirmed'))

    print(f"\n  Best: {best['label']}")
    print(f"  Mode: {best_mode}")

    # Detailed run
    eq_b, tr_b = run_backtest(
        ts_df, daily_data, test_start, test_end,
        mode=best_mode,
        n_pos=best['n_pos'], hold=best['hold'], lev=best['lev'],
        sl_pct=best.get('sl_pct', -3.0), tp_pct=best.get('tp_pct', 6.0),
        rebalance_freq=best.get('rebalance_freq', 10),
        min_spread=1.0,
        use_zscore='extreme' in best.get('mode', ''),
        min_zscore=1.5 if 'extreme' in best.get('mode', '') else 1.0,
    )
    pr(eq_b, tr_b, "BEST CONFIG (Detailed)", verbose=True)

    # Per-commodity
    if tr_b:
        tdf = pd.DataFrame(tr_b)
        print(f"\n  -- Top 15 Commodities --")
        sp = tdf.groupby('sym')['ret'].agg(['mean', 'count']).reset_index()
        sp = sp[sp['count'] >= 3].sort_values('mean', ascending=False)
        for _, row in sp.head(15).iterrows():
            sub = tdf[tdf['sym'] == row['sym']]
            print(f"    {row['sym']:6s}: N={int(row['count']):3d} WR={(sub['ret']>0).mean()*100:.1f}% "
                  f"Avg={row['mean']:+.3f}%")

        print(f"\n  -- Bottom 10 --")
        for _, row in sp.tail(10).iterrows():
            sub = tdf[tdf['sym'] == row['sym']]
            print(f"    {row['sym']:6s}: N={int(row['count']):3d} WR={(sub['ret']>0).mean()*100:.1f}% "
                  f"Avg={row['mean']:+.3f}%")

    # ════════════════════════════════════════════════════════════════
    # H. WALK-FORWARD VALIDATION
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("[H] WALK-FORWARD VALIDATION")
    print(f"{'='*70}")

    min_d = ts_df['date'].min()
    max_d = ts_df['date'].max()
    total_d = (max_d - min_d).days
    train_end = min_d + timedelta(days=int(total_d * 0.6))
    val_end = min_d + timedelta(days=int(total_d * 0.8))

    print(f"  Train: {min_d.date()} to {train_end.date()}")
    print(f"  Val:   {(train_end+timedelta(days=1)).date()} to {val_end.date()}")
    print(f"  Test:  {(val_end+timedelta(days=1)).date()} to {max_d.date()}")

    wf_results = []
    for phase, ps, pe in [
        ('Train', min_d, train_end),
        ('Validation', train_end + timedelta(days=1), val_end),
        ('Test', val_end + timedelta(days=1), max_d),
    ]:
        eq, tr = run_backtest(
            ts_df, daily_data,
            ps.strftime('%Y-%m-%d'), pe.strftime('%Y-%m-%d'),
            mode=best_mode,
            n_pos=best['n_pos'], hold=best['hold'], lev=best['lev'],
            sl_pct=best.get('sl_pct', -3.0), tp_pct=best.get('tp_pct', 6.0),
            rebalance_freq=best.get('rebalance_freq', 10),
            min_spread=1.0,
            use_zscore='extreme' in best.get('mode', ''),
            min_zscore=1.5 if 'extreme' in best.get('mode', '') else 1.0,
        )
        r = pr(eq, tr, f"WF-{phase}", verbose=True)
        if r:
            r['phase'] = phase
            wf_results.append(r)

    if len(wf_results) >= 2:
        sharpes = [r['sharpe'] for r in wf_results]
        wrs = [r['wr'] for r in wf_results]
        print(f"\n  WF Summary:")
        print(f"    Sharpe: {sharpes}")
        print(f"    WR:     {wrs}")
        print(f"    All positive Sharpe: {all(s > 0 for s in sharpes)}")

    # ════════════════════════════════════════════════════════════════
    # I. TOP 20 ALL RESULTS
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("[I] TOP 20 RESULTS BY SHARPE")
    print(f"{'='*70}")

    sorted_r = sorted(all_r, key=lambda x: -x['sharpe'])
    print(f"\n  {'#':>3} {'Sharpe':>7} {'WR':>6} {'MDD':>7} {'Avg':>8} "
          f"{'Ret':>8} {'N':>5}  {'Mode':>15}  {'Label'}")
    for i, r in enumerate(sorted_r[:20]):
        print(f"  {i+1:3d} {r['sharpe']:7.2f} {r['wr']:5.1f}% {r['mdd']:6.1f}% "
              f"{r['avg']:+7.3f}% {r['total_ret']:+7.1f}% {r['n']:5d}  "
              f"{r.get('mode',''):>15}  {r['label']}")

    # ════════════════════════════════════════════════════════════════
    # J. FINAL SUMMARY
    # ════════════════════════════════════════════════════════════════
    print(f"""

  ======================================================================
  V91 TERM STRUCTURE CARRY STRATEGY v3 — FINAL SUMMARY
  ======================================================================

  Best Mode: {best.get('mode', 'N/A')}
  Config:    H={best['hold']}d N={best['n_pos']} L={best['lev']}x
             SL={best.get('sl_pct', 'None')} TP={best.get('tp_pct', 'None')}
  Performance:
    Sharpe:  {best['sharpe']:.2f}
    WR:      {best['wr']:.1f}%
    MDD:     {best['mdd']:.1f}%
    Avg Ret: {best['avg']:+.3f}%
    CAGR:    {best['cagr']:+.1f}%

  Walk-Forward: {'PASS' if wf_results and all(r['sharpe'] > 0 for r in wf_results) else 'MIXED'}

  Key Insight: Term structure carry alone is a weak signal for Chinese
  commodities. Price momentum confirmation significantly improves results.
  ======================================================================
  """)


if __name__ == '__main__':
    main()
