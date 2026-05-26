"""
V324: Mean Reversion + Risk Management — V323 Signal + V318 Risk Engine
=======================================================================
V323 combo (consec+ret5d+gap) is the best mean-reversion signal:
  54.4% WR, 23.5% annual, Sharpe 1.23 (2019-2026)

V324 adds:
  1. Volatility-targeted position sizing (Moreira & Muir 2017)
  2. Kelly criterion with drawdown breaker
  3. Trend filter (only buy dips in uptrends)
  4. Dynamic hold period based on conviction

Target: improve risk-adjusted returns, test 5+ years.
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
from collections import defaultdict
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_v301 import load_all_data

CASH0 = 1_000_000
COMM = 0.0005


def compute_signals(C, O, H, L, V, NS, ND):
    """Compute all signals for V324."""
    # Consecutive down days
    consec_dn = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        consec = 0
        for di in range(1, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                if C[si, di] < C[si, di-1]:
                    consec += 1
                else:
                    consec = 0
            else:
                consec = 0
            consec_dn[si, di] = consec

    # 5d return
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di-5]) and C[si, di-5] > 0:
                ret_5d[si, di] = C[si, di] / C[si, di-5] - 1

    # 20d return (for trend filter)
    ret_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di-20]) and C[si, di-20] > 0:
                ret_20d[si, di] = C[si, di] / C[si, di-20] - 1

    # Overnight gap
    gap = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if not np.isnan(O[si, di]) and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                gap[si, di] = O[si, di] / C[si, di-1] - 1

    # 20d volatility
    vol_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            rets = []
            for j in range(di - 20, di):
                if not np.isnan(C[si, j]) and not np.isnan(C[si, j-1]) and C[si, j-1] > 0:
                    rets.append(C[si, j] / C[si, j-1] - 1)
            if len(rets) >= 10:
                vol_20d[si, di] = np.std(rets) * np.sqrt(252)

    # Combo oversold score (cross-sectional rank)
    combo_rank = np.full((NS, ND), np.nan)
    for di in range(ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            s = 0.0
            # Consecutive down days (normalized)
            s += min(consec_dn[si, di] / 5.0, 1.0) * 0.4
            # 5d return (negative = oversold, negate for ranking)
            if not np.isnan(ret_5d[si, di]):
                s += min(max(-ret_5d[si, di] / 0.1, 0), 1.0) * 0.4
            # Gap down
            if not np.isnan(gap[si, di]):
                s += min(max(-gap[si, di] / 0.03, 0), 1.0) * 0.2
            scores[si] = s

        valid = ~np.isnan(scores)
        if valid.sum() >= 5:
            combo_rank[:, di] = pd.Series(scores).rank(pct=True, na_option='keep').values

    # Trend filter: 20d return > 0 means uptrend
    trend_up = np.full((NS, ND), False)
    for si in range(NS):
        for di in range(ND):
            if not np.isnan(ret_20d[si, di]):
                trend_up[si, di] = ret_20d[si, di] > 0

    return combo_rank, consec_dn, vol_20d, trend_up, ret_20d


def backtest_v324(C, O, H, L, NS, ND, dates, syms,
                  combo_rank, consec_dn, vol_20d, trend_up,
                  top_n=1, hold_days=5,
                  min_rank=0.7, atr_stop=2.5,
                  use_trend_filter=False,
                  vol_target=None,
                  kelly_fraction=None,
                  dd_breaker=None,  # reduce position when DD > threshold
                  leverage=1.0,
                  start_di=60, end_di=None):
    """
    Mean reversion with risk management.
    Signal at close[di], enter at open[di+1]. No look-ahead.
    """
    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions = []
    trades = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0
        new_positions = []

        # Exit positions
        for si, edi, ep, sp, alloc in positions:
            c = C[si, di]
            if np.isnan(c):
                new_positions.append((si, edi, ep, sp, alloc))
                continue
            exit_r = None
            if c < sp:
                exit_r = 'stop'
            elif di >= edi + hold_days:
                exit_r = 'hold'
            if exit_r:
                pnl = (c - ep) / ep - COMM
                profit = equity * alloc * leverage * pnl
                daily_pnl += profit
                trades.append({
                    'pnl_abs': profit, 'pnl_pct': pnl * 100 * leverage,
                    'days': di - edi + 1, 'di': di, 'year': d.year,
                    'sym': syms[si], 'reason': exit_r,
                })
            else:
                new_positions.append((si, edi, ep, sp, alloc))

        positions = new_positions
        equity += daily_pnl
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        if equity <= 0:
            break

        held = {p[0] for p in positions}
        if len(positions) >= top_n:
            continue

        # Drawdown breaker: reduce position size in drawdown
        pos_multiplier = 1.0
        if dd_breaker and peak > 0:
            current_dd = (peak - equity) / peak
            if current_dd > dd_breaker:
                pos_multiplier = max(0.1, 1.0 - current_dd / 0.5)

        # Entry
        candidates = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(combo_rank[si, di]):
                continue
            if combo_rank[si, di] < min_rank:
                continue

            # Trend filter
            if use_trend_filter and not trend_up[si, di]:
                continue

            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue

            # Position sizing
            alloc = pos_multiplier / max(top_n, 1)

            # Vol-targeted sizing
            if vol_target and not np.isnan(vol_20d[si, di]) and vol_20d[si, di] > 0:
                vol_adj = min(vol_target / vol_20d[si, di], 2.0)
                alloc *= vol_adj

            # Kelly sizing (simplified: f = p - q/b where p=WR, q=1-p, b=avg_win/avg_loss)
            if kelly_fraction:
                alloc *= kelly_fraction

            candidates.append((combo_rank[si, di], si, alloc))

        if not candidates:
            continue
        candidates.sort(key=lambda x: -x[0])

        for rank, si, alloc in candidates[:top_n]:
            if len(positions) >= top_n or si in held:
                break
            if di + 1 >= ND:
                break
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            # ATR stop
            atr_v = []
            for j in range(max(start_di, di - 14), di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
            if not atr_v:
                continue
            atr = np.mean(atr_v)
            positions.append((si, di + 1, ep, ep - atr_stop * atr, alloc))
            held.add(si)

    for si, edi, ep, sp, alloc in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * leverage * pnl

    return trades, equity, max_dd


def analyze(trades, equity, max_dd, label=""):
    if not trades:
        print(f"  {label}: no trades")
        return None
    nw = sum(1 for t in trades if t['pnl_pct'] > 0)
    wr = nw / len(trades) * 100
    n_days = max(1, trades[-1]['di'] - trades[0]['di'])
    ann = ((equity / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
    ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
    rets = np.array(ap) / CASH0
    sh = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0

    print(f"  {label}: {len(trades)}t WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
          f"Sh={sh:.2f} eq={equity:,.0f}")

    yr = {}
    for t in trades:
        y = t['year']
        if y not in yr:
            yr[y] = {'n': 0, 'w': 0, 'pnl': []}
        yr[y]['n'] += 1
        if t['pnl_pct'] > 0:
            yr[y]['w'] += 1
        yr[y]['pnl'].append(t['pnl_pct'])
    for y in sorted(yr.keys()):
        ys = yr[y]
        cum = np.prod([1 + p / 100 for p in ys['pnl']]) - 1
        print(f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% cum={cum:+.1%}")

    return {'n': len(trades), 'wr': wr, 'dd': max_dd, 'ann': ann, 'sh': sh, 'eq': equity}


def main():
    t0 = time.time()
    print("=" * 70)
    print("  V324: MEAN REVERSION + RISK MANAGEMENT")
    print("  V323 signal + V318 risk engine")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    print(f"  {NS} sym, {ND} days, "
          f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    print("[V324] Computing signals...", flush=True)
    combo_rank, consec_dn, vol_20d, trend_up, ret_20d = \
        compute_signals(C, O, H, L, V, NS, ND)
    print("  Done.", flush=True)

    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2019-01-01'):
            bt_2019 = i
            break

    # === Ablation: risk management features ===
    print("\n" + "=" * 70)
    print("  RISK MANAGEMENT ABLATION (tn=1 hd=5 lev=1, 2019-2026)")
    print("=" * 70)

    configs = [
        (False, None, None, None, "baseline (V323)"),
        (False, 0.15, None, None, "+vol target 15%"),
        (False, 0.20, None, None, "+vol target 20%"),
        (False, 0.25, None, None, "+vol target 25%"),
        (False, None, 0.5, None, "+half Kelly"),
        (False, None, 0.3, None, "+quarter Kelly"),
        (False, None, None, 0.10, "+dd breaker 10%"),
        (False, None, None, 0.15, "+dd breaker 15%"),
        (False, None, None, 0.20, "+dd breaker 20%"),
        (True, None, None, None, "+trend filter"),
        (True, 0.20, None, None, "+trend+vol 20%"),
        (True, 0.20, 0.5, None, "+trend+vol+kelly"),
        (True, 0.20, 0.5, 0.15, "+trend+vol+kelly+dd"),
        (True, 0.15, 0.5, 0.15, "+all (vt=15%)"),
        (True, 0.25, 0.5, 0.15, "+all (vt=25%)"),
    ]

    for tf, vt, kf, db, label in configs:
        trades, eq, dd = backtest_v324(
            C, O, H, L, NS, ND, dates, syms,
            combo_rank, consec_dn, vol_20d, trend_up,
            top_n=1, hold_days=5, min_rank=0.7,
            use_trend_filter=tf, vol_target=vt,
            kelly_fraction=kf, dd_breaker=db,
            leverage=1, start_di=bt_2019)
        analyze(trades, eq, dd, label)

    # === Full parameter sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    results = []
    for tn in [1, 2, 3]:
        for hd in [3, 5]:
            for lev in [1, 2, 3]:
                for tf in [False, True]:
                    for vt in [None, 0.20]:
                        trades, eq, dd = backtest_v324(
                            C, O, H, L, NS, ND, dates, syms,
                            combo_rank, consec_dn, vol_20d, trend_up,
                            top_n=tn, hold_days=hd, min_rank=0.7,
                            use_trend_filter=tf, vol_target=vt,
                            leverage=lev, start_di=bt_2019)
                        if len(trades) < 10:
                            continue
                        nw = sum(1 for t in trades if t['pnl_pct'] > 0)
                        wr = nw / len(trades) * 100
                        n_days = max(1, trades[-1]['di'] - trades[0]['di'])
                        ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
                        ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
                        rets_arr = np.array(ap) / CASH0
                        sh = np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252) if np.std(rets_arr) > 0 else 0
                        results.append({
                            'tn': tn, 'hd': hd, 'lev': lev, 'tf': tf, 'vt': vt,
                            'n': len(trades), 'wr': wr, 'ann': ann, 'dd': dd, 'sh': sh,
                        })

    results.sort(key=lambda x: -x['sh'])
    print(f"\n{'TN':>3} {'HD':>3} {'L':>3} {'TF':>3} {'VT':>5} {'N':>5} {'WR':>5} "
          f"{'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 60)
    for r in results[:20]:
        vt_s = f"{r['vt']:.2f}" if r['vt'] else "None"
        print(f"{r['tn']:>3} {r['hd']:>3} {r['lev']:>3} "
              f"{'Y' if r['tf'] else 'N':>3} {vt_s:>5} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sh']:>5.2f}")

    # === Best configs yearly ===
    print("\n" + "=" * 70)
    print("  BEST CONFIGS — YEARLY (2019-2026)")
    print("=" * 70)

    best = results[:5]
    for r in best:
        trades, eq, dd = backtest_v324(
            C, O, H, L, NS, ND, dates, syms,
            combo_rank, consec_dn, vol_20d, trend_up,
            top_n=r['tn'], hold_days=r['hd'], min_rank=0.7,
            use_trend_filter=r['tf'], vol_target=r['vt'],
            leverage=r['lev'], start_di=bt_2019)
        tf_s = "T" if r['tf'] else "NT"
        vt_s = f"vt={r['vt']}" if r['vt'] else "no-vt"
        print(f"\n  --- {tf_s} {vt_s} tn={r['tn']} hd={r['hd']} lev={r['lev']} ---")
        analyze(trades, eq, dd, f"{tf_s} {vt_s} tn={r['tn']} hd={r['hd']}")

    # === Full 10-year ===
    print("\n" + "=" * 70)
    print("  FULL 2016-2026 (10 years)")
    print("=" * 70)

    for r in best[:3]:
        trades, eq, dd = backtest_v324(
            C, O, H, L, NS, ND, dates, syms,
            combo_rank, consec_dn, vol_20d, trend_up,
            top_n=r['tn'], hold_days=r['hd'], min_rank=0.7,
            use_trend_filter=r['tf'], vol_target=r['vt'],
            leverage=r['lev'], start_di=60)
        tf_s = "T" if r['tf'] else "NT"
        vt_s = f"vt={r['vt']}" if r['vt'] else "no-vt"
        print(f"\n  FULL {tf_s} {vt_s} tn={r['tn']} hd={r['hd']} lev={r['lev']}")
        analyze(trades, eq, dd, f"full {tf_s} {vt_s} tn={r['tn']} hd={r['hd']}")

    print(f"\n[V324] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
