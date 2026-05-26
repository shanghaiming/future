"""
V326: Calendar Effects + Intraday Pattern Exploitation
======================================================
Exploit calendar and structural anomalies:
1. Day-of-week effects (buy dips on specific days)
2. Month-of-year seasonality (agricultural cycles)
3. Pre/post-holiday effects
4. Overnight gap with intraday exit (hold overnight only)
5. OI surge detection (capital flowing in → follow the money)

Combined with the best mean-reversion signal from V323/V324.
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


def analyze_calendar_effects(C, O, NS, ND, dates, syms):
    """Analyze calendar anomalies before building strategy."""
    print("\n  === CALENDAR EFFECT ANALYSIS ===\n")

    # Day of week
    print("  Day-of-week average next-day returns:")
    dow_rets = defaultdict(list)
    for di in range(1, ND - 1):
        d = dates[di]
        dow = d.dayofweek  # 0=Mon, 4=Fri
        for si in range(NS):
            if not np.isnan(C[si, di+1]) and not np.isnan(C[si, di]) and C[si, di] > 0:
                ret = C[si, di+1] / C[si, di] - 1
                dow_rets[dow].append(ret)
    for dow in sorted(dow_rets.keys()):
        rets = np.array(dow_rets[dow])
        names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        print(f"    {names[dow]}: n={len(rets)}, avg={np.mean(rets)*100:+.4f}%, "
              f"positive={np.mean(rets>0)*100:.1f}%")

    # Month of year
    print("\n  Month-of-year average daily returns:")
    moy_rets = defaultdict(list)
    for di in range(1, ND - 1):
        d = dates[di]
        moy = d.month
        for si in range(NS):
            if not np.isnan(C[si, di+1]) and not np.isnan(C[si, di]) and C[si, di] > 0:
                ret = C[si, di+1] / C[si, di] - 1
                moy_rets[moy].append(ret)
    for moy in sorted(moy_rets.keys()):
        rets = np.array(moy_rets[moy])
        print(f"    {moy:>2}月: n={len(rets)}, avg={np.mean(rets)*100:+.4f}%, "
              f"positive={np.mean(rets>0)*100:.1f}%")

    # Day of month (first half vs second half)
    print("\n  First-half vs second-half of month:")
    fh_rets, sh_rets = [], []
    for di in range(1, ND - 1):
        d = dates[di]
        for si in range(NS):
            if not np.isnan(C[si, di+1]) and not np.isnan(C[si, di]) and C[si, di] > 0:
                ret = C[si, di+1] / C[si, di] - 1
                if d.day <= 15:
                    fh_rets.append(ret)
                else:
                    sh_rets.append(ret)
    fh = np.array(fh_rets)
    sh = np.array(sh_rets)
    print(f"    1-15日: avg={np.mean(fh)*100:+.4f}%, positive={np.mean(fh>0)*100:.1f}%")
    print(f"    16-31日: avg={np.mean(sh)*100:+.4f}%, positive={np.mean(sh>0)*100:.1f}%")

    # Consecutive down days by day of week
    print("\n  Mean reversion by day of week:")
    for dow in range(5):
        names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        rets_after_dn = []
        for si in range(NS):
            consec = 0
            for di in range(1, ND - 1):
                if np.isnan(C[si, di]) or np.isnan(C[si, di-1]) or C[si, di-1] <= 0:
                    consec = 0
                    continue
                ret = C[si, di] / C[si, di-1] - 1
                if ret < 0:
                    consec += 1
                else:
                    consec = 0
                if consec >= 2 and dates[di].dayofweek == dow:
                    if not np.isnan(C[si, di+1]) and C[si, di] > 0:
                        fwd = C[si, di+1] / C[si, di] - 1
                        rets_after_dn.append(fwd)
        if rets_after_dn:
            arr = np.array(rets_after_dn)
            print(f"    {names[dow]} after 2dn: n={len(arr)}, avg={np.mean(arr)*100:+.4f}%, "
                  f"positive={np.mean(arr>0)*100:.1f}%")

    # Overnight gap → intraday reversal by gap size
    print("\n  Gap reversal by gap magnitude:")
    for thresh in [0.005, 0.01, 0.015, 0.02, 0.03]:
        oc_up, oc_dn = [], []
        for si in range(NS):
            for di in range(1, ND):
                if np.isnan(O[si, di]) or np.isnan(C[si, di-1]) or C[si, di-1] <= 0:
                    continue
                if np.isnan(C[si, di]):
                    continue
                co = O[si, di] / C[si, di-1] - 1  # overnight
                oc = C[si, di] / O[si, di] - 1     # intraday
                if co > thresh:
                    oc_up.append(oc)
                elif co < -thresh:
                    oc_dn.append(oc)
        if oc_up:
            arr_up = np.array(oc_up)
            print(f"    Gap UP >{thresh*100:.1f}%: n={len(arr_up)}, "
                  f"avg intraday={np.mean(arr_up)*100:+.4f}% ({np.mean(arr_up>0)*100:.1f}% pos)")
        if oc_dn:
            arr_dn = np.array(oc_dn)
            print(f"    Gap DN <-{thresh*100:.1f}%: n={len(arr_dn)}, "
                  f"avg intraday={np.mean(arr_dn)*100:+.4f}% ({np.mean(arr_dn>0)*100:.1f}% pos)")


def backtest_v326(C, O, H, L, NS, ND, dates, syms,
                  top_n=1, hold_days=5,
                  signal_mode='combo',  # 'combo', 'gap_only', 'consec_dow'
                  min_gap=0.01,
                  min_consec=2,
                  prefer_dow=None,  # list of preferred day-of-week (0-4)
                  avoid_dow=None,   # list of avoided day-of-week
                  prefer_month=None, # list of preferred months
                  atr_stop=2.5,
                  dd_breaker=None,
                  leverage=1.0,
                  start_di=60, end_di=None):
    """Calendar-enhanced mean-reversion strategy."""

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

        pos_mult = 1.0
        if dd_breaker and peak > 0:
            current_dd = (peak - equity) / peak
            if current_dd > dd_breaker:
                pos_mult = max(0.1, 1.0 - current_dd / 0.5)

        # Calendar filters
        dow = d.dayofweek
        if avoid_dow and dow in avoid_dow:
            continue
        if prefer_dow and dow not in prefer_dow:
            continue
        if prefer_month and d.month not in prefer_month:
            continue

        # Score candidates
        candidates = []
        for si in range(NS):
            if si in held:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue

            score = 0.0
            valid = True

            if signal_mode == 'combo':
                # Oversold combo
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

            elif signal_mode == 'gap_only':
                if di >= 1 and not np.isnan(O[si, di]) and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                    g = O[si, di] / C[si, di-1] - 1
                    if g < -min_gap:
                        score = -g  # bigger gap down = higher score
                    else:
                        valid = False
                else:
                    valid = False

            elif signal_mode == 'gap_intraday':
                # Enter at close (today), exit at close (tomorrow)
                # This captures intraday reversal after gap
                if di >= 1 and not np.isnan(O[si, di]) and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                    g = O[si, di] / C[si, di-1] - 1
                    if g < -min_gap:
                        score = -g
                    else:
                        valid = False
                else:
                    valid = False

            elif signal_mode == 'consec_dow':
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
                if consec >= min_consec:
                    score = consec
                else:
                    valid = False

            if not valid or score <= 0:
                continue

            alloc = pos_mult / max(top_n, 1)
            candidates.append((score, si, alloc))

        if not candidates:
            continue
        candidates.sort(key=lambda x: -x[0])

        for score, si, alloc in candidates[:top_n]:
            if len(positions) >= top_n or si in held:
                break
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
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
    print("  V326: CALENDAR EFFECTS + STRUCTURAL PATTERNS")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    print(f"  {NS} sym, {ND} days, "
          f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    # Calendar effect analysis
    analyze_calendar_effects(C, O, NS, ND, dates, syms)

    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2019-01-01'):
            bt_2019 = i
            break

    # === Day-of-week filtered mean reversion ===
    print("\n" + "=" * 70)
    print("  DAY-OF-WEEK FILTERED (2019-2026)")
    print("=" * 70)

    for hd in [2, 5]:
        for dow_set, label in [
            (None, "all days"),
            ([0, 1, 2], "Mon-Wed only"),
            ([1, 2, 3], "Tue-Thu only"),
            ([2, 3, 4], "Wed-Fri only"),
            ([0, 4], "Mon+Fri only"),
            ([0], "Mon only"),
            ([1], "Tue only"),
            ([2], "Wed only"),
            ([3], "Thu only"),
            ([4], "Fri only"),
        ]:
            trades, eq, dd = backtest_v326(
                C, O, H, L, NS, ND, dates, syms,
                signal_mode='combo', top_n=1, hold_days=hd,
                prefer_dow=dow_set, leverage=1,
                start_di=bt_2019)
            if len(trades) < 5:
                continue
            nw = sum(1 for t in trades if t['pnl_pct'] > 0)
            wr = nw / len(trades) * 100
            n_days = max(1, trades[-1]['di'] - trades[0]['di'])
            ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
            ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
            rets_arr = np.array(ap) / CASH0
            sh = np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252) if np.std(rets_arr) > 0 else 0
            print(f"  hd={hd} {label:>15}: {len(trades):>4}t WR={wr:.1f}% "
                  f"ann={ann:+.1f}% DD={dd:.1f}% Sh={sh:.2f}")

    # === Gap-only intraday strategy ===
    print("\n" + "=" * 70)
    print("  GAP REVERSAL STRATEGY (2019-2026)")
    print("=" * 70)

    for mg in [0.005, 0.01, 0.015, 0.02]:
        for hd in [1, 2, 3]:
            for lev in [1, 2, 3]:
                trades, eq, dd = backtest_v326(
                    C, O, H, L, NS, ND, dates, syms,
                    signal_mode='gap_only', min_gap=mg,
                    top_n=1, hold_days=hd, leverage=lev,
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
                if sh > 0.5 or ann > 10:
                    print(f"  gap>{mg*100:.1f}% hd={hd} lev={lev}: "
                          f"{len(trades)}t WR={wr:.1f}% ann={ann:+.1f}% DD={dd:.1f}% Sh={sh:.2f}")

    # === Best config with full yearly detail ===
    print("\n" + "=" * 70)
    print("  BEST CONFIGS — FULL DETAIL (2019-2026)")
    print("=" * 70)

    best_configs = [
        ('combo', None, None, 1, 5, 1, "combo all days"),
        ('combo', [0, 1, 2], None, 1, 5, 1, "combo Mon-Wed"),
        ('combo', [1, 2, 3], None, 1, 5, 1, "combo Tue-Thu"),
        ('gap_only', None, None, 1, 2, 1, "gap>1% hd=2"),
        ('gap_only', None, None, 1, 2, 2, "gap>1% hd=2 lev=2"),
    ]
    for mode, pdow, pmo, tn, hd, lev, label in best_configs:
        mg = 0.01 if mode == 'gap_only' else 0
        trades, eq, dd = backtest_v326(
            C, O, H, L, NS, ND, dates, syms,
            signal_mode=mode, min_gap=mg,
            prefer_dow=pdow, prefer_month=pmo,
            top_n=tn, hold_days=hd, leverage=lev,
            start_di=bt_2019)
        print(f"\n  --- {label} ---")
        analyze(trades, eq, dd, label)

    # === Full 10-year ===
    print("\n" + "=" * 70)
    print("  FULL 2016-2026")
    print("=" * 70)

    for mode, pdow, pmo, tn, hd, lev, label in best_configs[:3]:
        mg = 0.01 if mode == 'gap_only' else 0
        trades, eq, dd = backtest_v326(
            C, O, H, L, NS, ND, dates, syms,
            signal_mode=mode, min_gap=mg,
            prefer_dow=pdow, prefer_month=pmo,
            top_n=tn, hold_days=hd, leverage=lev,
            start_di=60)
        print(f"\n  FULL {label}")
        analyze(trades, eq, dd, f"full {label}")

    print(f"\n[V326] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
