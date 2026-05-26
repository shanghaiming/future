"""
V327: Same-Day Gap Reversal — Exploit Intraday Mean Reversion
=============================================================
V326 confirmed: Gap DN <-3% → +0.65% intraday reversal (53.5% positive)
Gap DN <-2% → +0.44% intraday reversal (52.9% positive)

KEY DIFFERENCE: Enter at TODAY'S open (observable gap), exit at TODAY'S close.
This is NOT look-ahead: we observe the gap at the open and enter immediately.

Combine with:
1. Same-day gap reversal (enter at open, exit at close)
2. Cross-sectional gap ranking (buy the biggest gap down)
3. Plus the mean-reversion combo for overnight holding

Backtest 5+ years.
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
from collections import defaultdict
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_v301 import load_all_data

CASH0 = 1_000_000
COMM = 0.0005


def backtest_sameday_gap(C, O, H, L, NS, ND, dates, syms,
                         min_gap=0.02,
                         top_n=1,
                         atr_stop_pct=None,
                         leverage=1.0,
                         start_di=60, end_di=None):
    """
    Same-day gap reversal.
    At open[di], observe gap = O[di]/C[di-1] - 1.
    If gap down > min_gap, buy at O[di], exit at C[di].
    This is clean: signal observable at open time.
    """
    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    trades = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        # Compute gaps for all commodities
        candidates = []
        for si in range(NS):
            c_prev = C[si, di-1]
            o_now = O[si, di]
            c_now = C[si, di]
            if np.isnan(c_prev) or np.isnan(o_now) or np.isnan(c_now):
                continue
            if c_prev <= 0 or o_now <= 0:
                continue
            gap = o_now / c_prev - 1
            if gap < -min_gap:
                # Score by gap magnitude (bigger gap = more oversold)
                candidates.append((-gap, si, o_now, c_now))

        if not candidates:
            continue

        candidates.sort(key=lambda x: -x[0])  # biggest gap first

        daily_pnl = 0
        for i, (score, si, entry_price, exit_price) in enumerate(candidates[:top_n]):
            alloc = 1.0 / max(top_n, 1)

            # Check stop loss
            if atr_stop_pct:
                sl = entry_price * (1 - atr_stop_pct)
                # Use low of the day to check stop
                low = L[si, di]
                if not np.isnan(low) and low < sl:
                    exit_price = sl

            pnl = (exit_price - entry_price) / entry_price - COMM
            profit = equity * alloc * leverage * pnl
            daily_pnl += profit
            trades.append({
                'pnl_abs': profit, 'pnl_pct': pnl * 100 * leverage,
                'days': 1, 'di': di, 'year': d.year,
                'sym': syms[si], 'reason': 'sameday',
            })

        equity += daily_pnl
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        if equity <= 0:
            break

    return trades, equity, max_dd


def backtest_combined(C, O, H, L, NS, ND, dates, syms,
                      min_gap=0.02,
                      gap_alloc=0.4,      # allocation to same-day gap
                      mr_alloc=0.6,       # allocation to overnight MR
                      mr_hold=5,
                      atr_stop=2.5,
                      leverage=1.0,
                      start_di=60, end_di=None):
    """
    Combined strategy:
    - Same-day gap reversal (40% capital)
    - Overnight mean reversion combo (60% capital)
    """
    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    mr_positions = []  # [(si, edi, ep, sp, alloc)]
    trades = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0

        # === Exit MR positions ===
        new_mr = []
        for si, edi, ep, sp, alloc in mr_positions:
            c = C[si, di]
            if np.isnan(c):
                new_mr.append((si, edi, ep, sp, alloc))
                continue
            exit_r = None
            if c < sp:
                exit_r = 'stop'
            elif di >= edi + mr_hold:
                exit_r = 'hold'
            if exit_r:
                pnl = (c - ep) / ep - COMM
                profit = equity * alloc * leverage * pnl
                daily_pnl += profit
                trades.append({
                    'pnl_abs': profit, 'pnl_pct': pnl * 100 * leverage,
                    'days': di - edi + 1, 'di': di, 'year': d.year,
                    'sym': syms[si], 'reason': f'mr_{exit_r}',
                })
            else:
                new_mr.append((si, edi, ep, sp, alloc))
        mr_positions = new_mr

        # === Same-day gap reversal ===
        gap_candidates = []
        for si in range(NS):
            c_prev = C[si, di-1]
            o_now = O[si, di]
            c_now = C[si, di]
            if np.isnan(c_prev) or np.isnan(o_now) or np.isnan(c_now):
                continue
            if c_prev <= 0 or o_now <= 0:
                continue
            gap = o_now / c_prev - 1
            if gap < -min_gap:
                gap_candidates.append((-gap, si, o_now, c_now))

        if gap_candidates:
            gap_candidates.sort(key=lambda x: -x[0])
            score, si, ep, ex = gap_candidates[0]
            pnl = (ex - ep) / ep - COMM
            profit = equity * gap_alloc * leverage * pnl
            daily_pnl += profit
            trades.append({
                'pnl_abs': profit, 'pnl_pct': pnl * 100 * leverage,
                'days': 1, 'di': di, 'year': d.year,
                'sym': syms[si], 'reason': 'gap_sameday',
            })

        equity += daily_pnl
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        if equity <= 0:
            break

        # === MR entry ===
        if len(mr_positions) < 1:
            mr_candidates = []
            mr_held = {p[0] for p in mr_positions}
            # Gap candidates already used
            gap_si = gap_candidates[0][1] if gap_candidates else -1

            for si in range(NS):
                if si in mr_held or si == gap_si:
                    continue
                if di + 1 >= ND or np.isnan(O[si, di+1]):
                    continue
                if np.isnan(C[si, di]) or C[si, di] <= 0:
                    continue

                # Combo oversold score
                score = 0.0
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
                score += min(consec / 5.0, 1.0) * 0.4

                if di >= 5 and not np.isnan(C[si, di-5]) and C[si, di-5] > 0:
                    ret5 = -(C[si, di] / C[si, di-5] - 1)
                    score += min(max(ret5 / 0.1, 0), 1.0) * 0.4

                if di >= 1 and not np.isnan(O[si, di]) and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                    g = -(O[si, di] / C[si, di-1] - 1)
                    score += min(max(g / 0.03, 0), 1.0) * 0.2

                if score > 0.3:
                    mr_candidates.append((score, si))

            if mr_candidates:
                mr_candidates.sort(key=lambda x: -x[0])
                score, si = mr_candidates[0]
                ep = O[si, di+1]
                if not np.isnan(ep) and ep > 0:
                    atr_v = []
                    for j in range(max(start_di, di - 14), di):
                        hh, ll, cc = H[si, j], L[si, j], C[si, j]
                        if not any(np.isnan([hh, ll, cc])):
                            atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
                    if atr_v:
                        atr = np.mean(atr_v)
                        mr_positions.append((si, di+1, ep, ep - atr_stop * atr, mr_alloc))

    # Close MR positions
    for si, edi, ep, sp, alloc in mr_positions:
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
    print("  V327: SAME-DAY GAP REVERSAL + OVERNIGHT MR")
    print("  Exploit intraday gap reversal (observable at open)")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    print(f"  {NS} sym, {ND} days, "
          f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2019-01-01'):
            bt_2019 = i
            break

    # === Same-day gap reversal only ===
    print("\n" + "=" * 70)
    print("  SAME-DAY GAP REVERSAL ONLY (2019-2026)")
    print("=" * 70)

    for mg in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]:
        for tn in [1, 2, 3]:
            for lev in [1, 2, 3]:
                trades, eq, dd = backtest_sameday_gap(
                    C, O, H, L, NS, ND, dates, syms,
                    min_gap=mg, top_n=tn, leverage=lev,
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
                if sh > 0.3 or ann > 5:
                    print(f"  gap>{mg*100:.1f}% tn={tn} lev={lev}: "
                          f"{len(trades)}t WR={wr:.1f}% ann={ann:+.1f}% DD={dd:.1f}% Sh={sh:.2f}")

    # Best same-day detail
    print("\n  === BEST SAME-DAY DETAIL ===")
    for mg in [0.015, 0.02, 0.03]:
        for lev in [1, 2, 3]:
            trades, eq, dd = backtest_sameday_gap(
                C, O, H, L, NS, ND, dates, syms,
                min_gap=mg, top_n=1, leverage=lev,
                start_di=bt_2019)
            print(f"\n  gap>{mg*100:.1f}% lev={lev}")
            analyze(trades, eq, dd, f"gap>{mg*100:.0f}% lev={lev}")

    # === Combined: same-day gap + overnight MR ===
    print("\n" + "=" * 70)
    print("  COMBINED: SAME-DAY GAP + OVERNIGHT MR (2019-2026)")
    print("=" * 70)

    for gap_a, mr_a, mg, hd in [(0.3, 0.7, 0.02, 5),
                                  (0.4, 0.6, 0.02, 5),
                                  (0.5, 0.5, 0.02, 5),
                                  (0.3, 0.7, 0.015, 5),
                                  (0.4, 0.6, 0.015, 5),
                                  (0.3, 0.7, 0.02, 3),
                                  (0.4, 0.6, 0.02, 3)]:
        for lev in [1, 2, 3]:
            trades, eq, dd = backtest_combined(
                C, O, H, L, NS, ND, dates, syms,
                min_gap=mg, gap_alloc=gap_a, mr_alloc=mr_a,
                mr_hold=hd, leverage=lev,
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
            print(f"  gap={gap_a:.0%}/{mr_a:.0%} mg={mg*100:.1f}% hd={hd} lev={lev}: "
                  f"{len(trades)}t WR={wr:.1f}% ann={ann:+.1f}% DD={dd:.1f}% Sh={sh:.2f}")

    # Best combined detail
    print("\n  === BEST COMBINED DETAIL ===")
    for gap_a, mr_a, mg, hd in [(0.4, 0.6, 0.02, 5), (0.3, 0.7, 0.02, 5)]:
        for lev in [1, 2]:
            trades, eq, dd = backtest_combined(
                C, O, H, L, NS, ND, dates, syms,
                min_gap=mg, gap_alloc=gap_a, mr_alloc=mr_a,
                mr_hold=hd, leverage=lev,
                start_di=bt_2019)
            print(f"\n  gap={gap_a:.0%} mr={mr_a:.0%} mg={mg*100:.1f}% hd={hd} lev={lev}")
            analyze(trades, eq, dd, f"combined lev={lev}")

            # Per-strategy breakdown
            gap_t = [t for t in trades if t['reason'] == 'gap_sameday']
            mr_t = [t for t in trades if t['reason'].startswith('mr_')]
            for name, st in [('gap_sameday', gap_t), ('mr_overnight', mr_t)]:
                if st:
                    nw = sum(1 for t in st if t['pnl_pct'] > 0)
                    print(f"      {name:>15}: {len(st)}t WR={nw/len(st)*100:.1f}%")

    # Full 10-year
    print("\n" + "=" * 70)
    print("  FULL 2016-2026")
    print("=" * 70)

    for gap_a, mr_a, mg, hd in [(0.4, 0.6, 0.02, 5)]:
        for lev in [1, 2]:
            trades, eq, dd = backtest_combined(
                C, O, H, L, NS, ND, dates, syms,
                min_gap=mg, gap_alloc=gap_a, mr_alloc=mr_a,
                mr_hold=hd, leverage=lev,
                start_di=60)
            print(f"\n  FULL gap={gap_a:.0%} mr={mr_a:.0%} lev={lev}")
            analyze(trades, eq, dd, f"full lev={lev}")

    print(f"\n[V327] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
