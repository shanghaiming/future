"""
V323: Pure Rank-Based Mean Reversion — Simplest Possible Strategy
==================================================================
V320 found: 5 consecutive down days → +0.15%/day avg (37.8% annualized)
V322 found: momentum is regime-dependent, unstable over 5+ years

Approach: SIMPLE rank-based mean reversion
1. Each day, rank all commodities by "oversold" score
2. Buy the MOST oversold (most consecutive down days, biggest recent drop)
3. Hold for N days
4. No complex multi-factor combination

Backtest: Full 2016-2026, 5+ year validation from 2019
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
from collections import defaultdict
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_v301 import load_all_data

CASH0 = 1_000_000
COMM = 0.0005


def backtest_rank_reversion(C, O, H, L, NS, ND, dates, syms,
                            rank_mode='consec',  # 'consec', 'ret5d', 'ret10d', 'gap', 'combo'
                            top_n=1, hold_days=2,
                            min_rank=0.7,         # minimum rank to qualify
                            atr_stop=2.5,
                            stop_loss_pct=None,   # percentage stop loss
                            take_profit_pct=None,  # percentage take profit
                            leverage=1.0,
                            start_di=60, end_di=None):
    """
    Pure rank-based mean reversion backtest.
    Rank commodities by oversold metric, buy most oversold.
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
        for si, edi, ep, sp, tp, alloc in positions:
            c = C[si, di]
            if np.isnan(c):
                new_positions.append((si, edi, ep, sp, tp, alloc))
                continue
            exit_r = None
            if sp is not None and c < sp:
                exit_r = 'stop'
            elif tp is not None and c > tp:
                exit_r = 'profit'
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
                new_positions.append((si, edi, ep, sp, tp, alloc))

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

        # Compute oversold rank
        scores = np.full(NS, np.nan)
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue

            score = 0.0
            valid = True

            if rank_mode == 'consec':
                # Consecutive down days
                consec = 0
                for j in range(di, max(0, di - 20), -1):
                    if j < 1:
                        break
                    if np.isnan(C[si, j]) or np.isnan(C[si, j-1]) or C[si, j-1] <= 0:
                        consec = 0
                        break
                    if C[si, j] < C[si, j-1]:
                        consec += 1
                    else:
                        break
                score = consec  # higher = more oversold

            elif rank_mode == 'ret5d':
                # 5-day return (more negative = more oversold)
                if di >= 5 and not np.isnan(C[si, di-5]) and C[si, di-5] > 0:
                    score = -(C[si, di] / C[si, di-5] - 1)  # negate: more negative ret → higher score
                else:
                    valid = False

            elif rank_mode == 'ret10d':
                if di >= 10 and not np.isnan(C[si, di-10]) and C[si, di-10] > 0:
                    score = -(C[si, di] / C[si, di-10] - 1)
                else:
                    valid = False

            elif rank_mode == 'gap':
                # Overnight gap (bigger gap down = more oversold)
                if di >= 1 and not np.isnan(O[si, di]) and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                    score = -(O[si, di] / C[si, di-1] - 1)
                else:
                    valid = False

            elif rank_mode == 'combo':
                # Weighted combo: consec + ret5d + gap
                cs = 0.0
                consec = 0
                for j in range(di, max(0, di - 20), -1):
                    if j < 1:
                        break
                    if np.isnan(C[si, j]) or np.isnan(C[si, j-1]) or C[si, j-1] <= 0:
                        consec = 0
                        break
                    if C[si, j] < C[si, j-1]:
                        consec += 1
                    else:
                        break
                cs += min(consec / 5.0, 1.0) * 0.4

                if di >= 5 and not np.isnan(C[si, di-5]) and C[si, di-5] > 0:
                    ret5 = -(C[si, di] / C[si, di-5] - 1)
                    cs += min(max(ret5 / 0.1, 0), 1.0) * 0.4  # normalize 10% drop = 1.0

                if di >= 1 and not np.isnan(O[si, di]) and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                    gap = -(O[si, di] / C[si, di-1] - 1)
                    cs += min(max(gap / 0.03, 0), 1.0) * 0.2  # normalize 3% gap = 1.0

                score = cs

            if not valid or np.isnan(score):
                continue
            scores[si] = score

        # Rank and select
        valid_mask = ~np.isnan(scores)
        if valid_mask.sum() < 5:
            continue

        ranks = pd.Series(scores).rank(pct=True, na_option='keep').values

        candidates = []
        for si in range(NS):
            if np.isnan(ranks[si]):
                continue
            if ranks[si] < min_rank:
                continue
            if si in held:
                continue
            candidates.append((ranks[si], si))

        if not candidates:
            continue
        candidates.sort(key=lambda x: -x[0])  # highest rank = most oversold

        alloc = 1.0 / max(top_n, 1)
        for rank, si in candidates[:top_n]:
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
            atr = np.mean(atr_v) if atr_v else 0

            sp = ep - atr_stop * atr if atr > 0 else None
            if stop_loss_pct:
                sp2 = ep * (1 - stop_loss_pct)
                sp = max(sp, sp2) if sp else sp2

            tp = None
            if take_profit_pct:
                tp = ep * (1 + take_profit_pct)

            positions.append((si, di + 1, ep, sp, tp, alloc))
            held.add(si)

    # Close remaining
    for si, edi, ep, sp, tp, alloc in positions:
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
    print("  V323: PURE RANK-BASED MEAN REVERSION")
    print("  Simplest possible: rank by oversold metric, buy most oversold")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    print(f"  {NS} sym, {ND} days, "
          f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    # Backtest periods
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2019-01-01'):
            bt_2019 = i
            break

    # === Compare ranking modes ===
    print("\n" + "=" * 70)
    print("  RANKING MODE COMPARISON (tn=1, hd=2, lev=1, 2019-2026)")
    print("=" * 70)

    for mode in ['consec', 'ret5d', 'ret10d', 'gap', 'combo']:
        for min_r in [0.5, 0.7, 0.9]:
            trades, eq, dd = backtest_rank_reversion(
                C, O, H, L, NS, ND, dates, syms,
                rank_mode=mode, top_n=1, hold_days=2,
                min_rank=min_r, start_di=bt_2019)
            if len(trades) < 10:
                continue
            nw = sum(1 for t in trades if t['pnl_pct'] > 0)
            wr = nw / len(trades) * 100
            n_days = max(1, trades[-1]['di'] - trades[0]['di'])
            ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
            ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
            rets_arr = np.array(ap) / CASH0
            sh = np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252) if np.std(rets_arr) > 0 else 0
            print(f"  {mode:>6} min_r={min_r:.1f}: {len(trades)}t WR={wr:.1f}% "
                  f"ann={ann:+.1f}% DD={dd:.1f}% Sh={sh:.2f}")

    # === Full parameter sweep for best modes ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (all modes, 2019-2026)")
    print("=" * 70)

    results = []
    for mode in ['consec', 'ret5d', 'ret10d', 'gap', 'combo']:
        for tn in [1, 2, 3]:
            for hd in [1, 2, 3, 5]:
                for lev in [1, 2, 3]:
                    trades, eq, dd = backtest_rank_reversion(
                        C, O, H, L, NS, ND, dates, syms,
                        rank_mode=mode, top_n=tn, hold_days=hd,
                        min_rank=0.7, leverage=lev,
                        start_di=bt_2019)
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
                        'mode': mode, 'tn': tn, 'hd': hd, 'lev': lev,
                        'n': len(trades), 'wr': wr, 'ann': ann, 'dd': dd, 'sh': sh,
                    })

    results.sort(key=lambda x: -x['sh'])
    print(f"\n{'Mode':>6} {'TN':>3} {'HD':>3} {'L':>3} {'N':>5} {'WR':>5} "
          f"{'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 60)
    for r in results[:25]:
        print(f"{r['mode']:>6} {r['tn']:>3} {r['hd']:>3} {r['lev']:>3} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sh']:>5.2f}")

    results.sort(key=lambda x: -x['ann'])
    print(f"\n--- By Annual Return ---")
    for r in results[:15]:
        print(f"  {r['mode']:>6} tn={r['tn']} hd={r['hd']} lev={r['lev']}: "
              f"{r['n']}t WR={r['wr']:.1f}% ann={r['ann']:+.1f}% "
              f"DD={r['dd']:.1f}% Sh={r['sh']:.2f}")

    # === Best configs detailed ===
    print("\n" + "=" * 70)
    print("  BEST CONFIGS — YEARLY DETAIL (2019-2026)")
    print("=" * 70)

    # Find top configs by Sharpe
    top_configs = results[:5]
    for r in top_configs:
        trades, eq, dd = backtest_rank_reversion(
            C, O, H, L, NS, ND, dates, syms,
            rank_mode=r['mode'], top_n=r['tn'], hold_days=r['hd'],
            min_rank=0.7, leverage=r['lev'],
            start_di=bt_2019)
        print(f"\n  --- {r['mode']} tn={r['tn']} hd={r['hd']} lev={r['lev']} ---")
        analyze(trades, eq, dd, f"{r['mode']} tn={r['tn']} hd={r['hd']}")

    # === Full 10-year ===
    print("\n" + "=" * 70)
    print("  FULL 2016-2026 (10 years)")
    print("=" * 70)

    for r in top_configs[:3]:
        trades, eq, dd = backtest_rank_reversion(
            C, O, H, L, NS, ND, dates, syms,
            rank_mode=r['mode'], top_n=r['tn'], hold_days=r['hd'],
            min_rank=0.7, leverage=r['lev'],
            start_di=60)
        print(f"\n  FULL {r['mode']} tn={r['tn']} hd={r['hd']} lev={r['lev']}")
        analyze(trades, eq, dd, f"full {r['mode']} tn={r['tn']} hd={r['hd']}")

    # === Stop-loss/take-profit variants ===
    print("\n" + "=" * 70)
    print("  STOP/PROFIT VARIANTS (best mode, 2019-2026)")
    print("=" * 70)

    best_mode = top_configs[0]['mode'] if top_configs else 'combo'
    best_tn = top_configs[0]['tn'] if top_configs else 1
    best_hd = top_configs[0]['hd'] if top_configs else 2

    for sl in [None, 0.03, 0.05, 0.08]:
        for tp in [None, 0.03, 0.05, 0.08]:
            trades, eq, dd = backtest_rank_reversion(
                C, O, H, L, NS, ND, dates, syms,
                rank_mode=best_mode, top_n=best_tn, hold_days=best_hd,
                min_rank=0.7, leverage=1,
                stop_loss_pct=sl, take_profit_pct=tp,
                start_di=bt_2019)
            if len(trades) < 10:
                continue
            nw = sum(1 for t in trades if t['pnl_pct'] > 0)
            wr = nw / len(trades) * 100
            n_days = max(1, trades[-1]['di'] - trades[0]['di'])
            ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
            ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
            rets_arr = np.array(ap) / CASH0
            sh = np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252) if np.std(rets_arr) > 0 else 0
            sl_s = f"{sl*100:.0f}%" if sl else "None"
            tp_s = f"{tp*100:.0f}%" if tp else "None"
            print(f"  SL={sl_s:>5} TP={tp_s:>5}: {len(trades)}t WR={wr:.1f}% "
                  f"ann={ann:+.1f}% DD={dd:.1f}% Sh={sh:.2f}")

    print(f"\n[V323] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
