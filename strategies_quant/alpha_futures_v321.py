"""
V321: Mean Reversion + OI Capitulation — Based on Market Structure Analysis
=============================================================================
V320 revealed the TRUE market essence:
  - Consecutive down days → strong mean reversion (+0.15%/day at 5d streak)
  - OI down + Price down (capitulation) → +70% annualized forward return
  - Cross-sectional: worst performers bounce back most
  - AC1 is negative on average (-0.02) → mean-reverting market

This strategy exploits:
1. Consecutive down days (primary signal)
2. OI capitulation (confirmation signal)
3. Cross-sectional recent-return rank (buy worst recent performers)
4. Gap reversal (overnight gap downs)
5. Volatility targeting for position sizing

Backtest: 5+ years (2019-2026), no look-ahead bias.
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
from collections import defaultdict
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_v301 import load_all_data

CASH0 = 1_000_000
COMM = 0.0005  # round-trip commission


def compute_mean_reversion_signals(C, O, V, OI, NS, ND):
    """Compute mean-reversion signals without look-ahead."""
    # Signal arrays: higher = stronger buy signal
    consec_dn = np.zeros((NS, ND), dtype=int)  # consecutive down days
    recent_ret = np.full((NS, ND), np.nan)      # 5d return rank (low = oversold)
    oi_capitulation = np.full((NS, ND), np.nan) # OI declining + price declining
    gap_signal = np.full((NS, ND), np.nan)       # overnight gap down rank
    vol_20d = np.full((NS, ND), np.nan)          # 20d volatility

    for si in range(NS):
        consec = 0
        for di in range(1, ND):
            # Consecutive down days
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                ret = C[si, di] / C[si, di-1] - 1
                if ret < 0:
                    consec += 1
                else:
                    consec = 0
                consec_dn[si, di] = consec
            else:
                consec = 0

            # 5d return
            if di >= 5 and not np.isnan(C[si, di]) and not np.isnan(C[si, di-5]) and C[si, di-5] > 0:
                recent_ret[si, di] = C[si, di] / C[si, di-5] - 1

            # OI 5d change + price 5d change
            if di >= 5:
                oi_now = OI[si, di]
                oi_5 = OI[si, di-5]
                c_now = C[si, di]
                c_5 = C[si, di-5]
                if (not np.isnan(oi_now) and not np.isnan(oi_5) and
                    not np.isnan(c_now) and not np.isnan(c_5) and c_5 > 0):
                    oi_chg = (oi_now - oi_5) / max(abs(oi_5), 1)
                    p_chg = (c_now - c_5) / c_5
                    # Capitulation: both OI and price declining
                    if oi_chg < 0 and p_chg < 0:
                        oi_capitulation[si, di] = abs(p_chg)  # stronger decline = stronger signal
                    else:
                        oi_capitulation[si, di] = 0

            # Overnight gap
            if di >= 1 and not np.isnan(O[si, di]) and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                gap_signal[si, di] = (O[si, di] / C[si, di-1] - 1)  # negative = gap down

            # 20d vol
            if di >= 20:
                rets = []
                for j in range(di - 20, di):
                    if not np.isnan(C[si, j]) and not np.isnan(C[si, j-1]) and C[si, j-1] > 0:
                        rets.append(C[si, j] / C[si, j-1] - 1)
                if len(rets) >= 10:
                    vol_20d[si, di] = np.std(rets) * np.sqrt(252)

    # Cross-sectional rank of 5d returns (low rank = oversold = buy)
    ret_rank = np.full((NS, ND), np.nan)
    for di in range(ND):
        valid = ~np.isnan(recent_ret[:, di])
        if valid.sum() >= 10:
            # Rank: lowest return gets highest score (1.0), highest return gets 0.0
            ret_rank[:, di] = 1.0 - pd.Series(recent_ret[:, di]).rank(pct=True, na_option='keep').values

    # Cross-sectional rank of gap (biggest gap down = highest score)
    gap_rank = np.full((NS, ND), np.nan)
    for di in range(ND):
        valid = ~np.isnan(gap_signal[:, di])
        if valid.sum() >= 10:
            gap_rank[:, di] = 1.0 - pd.Series(gap_signal[:, di]).rank(pct=True, na_option='keep').values

    return consec_dn, ret_rank, oi_capitulation, gap_rank, vol_20d


def backtest_v321(C, O, H, L, V, OI, NS, ND, dates, syms,
                  consec_dn, ret_rank, oi_capitulation, gap_rank, vol_20d,
                  min_consec=2, top_n=1, hold_days=2,
                  use_oi=True, use_gap=True, use_ret_rank=True,
                  min_score=0.5, atr_stop=2.5,
                  vol_target=None, leverage=1.0,
                  start_di=60, end_di=None):
    """
    Mean-reversion backtest with clean execution.
    Signal computed at close[di], enter at open[di+1], no look-ahead.
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

        # Exit existing positions
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

        # Entry: compute combined mean-reversion score
        held = {p[0] for p in positions}
        if len(positions) >= top_n:
            continue

        candidates = []
        for si in range(NS):
            if si in held:
                continue

            # Primary: consecutive down days
            cd = consec_dn[si, di]
            if cd < min_consec:
                continue

            score = 0.0
            n_signals = 0

            # Consecutive down bonus (core signal)
            score += min(cd / 5.0, 1.0) * 0.3
            n_signals += 1

            # Recent return rank (oversold)
            if use_ret_rank and not np.isnan(ret_rank[si, di]):
                score += ret_rank[si, di] * 0.3
                n_signals += 1

            # OI capitulation
            if use_oi and not np.isnan(oi_capitulation[si, di]):
                oi_cap = oi_capitulation[si, di]
                if oi_cap > 0:
                    score += min(oi_cap * 10, 1.0) * 0.25
                    n_signals += 1

            # Gap reversal
            if use_gap and not np.isnan(gap_rank[si, di]):
                score += gap_rank[si, di] * 0.15
                n_signals += 1

            if n_signals < 2:
                continue
            if score < min_score:
                continue

            # Check next day tradable
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue

            # Vol-targeted position sizing
            alloc = 1.0 / max(top_n, 1)
            if vol_target and not np.isnan(vol_20d[si, di]) and vol_20d[si, di] > 0:
                vol_adj = min(vol_target / vol_20d[si, di], 2.0)
                alloc *= vol_adj

            candidates.append((score, si, alloc))

        if not candidates:
            continue
        candidates.sort(key=lambda x: -x[0])

        for score, si, alloc in candidates[:top_n]:
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

    # Close remaining
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

    # Yearly breakdown
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
        print(f"    {y}: {ys['n']}t WR={ys['w'] / ys['n'] * 100:.1f}% cum={cum:+.1%}")

    return {'n': len(trades), 'wr': wr, 'dd': max_dd, 'ann': ann, 'sh': sh, 'eq': equity}


def main():
    t0 = time.time()
    print("=" * 70)
    print("  V321: MEAN REVERSION + OI CAPITULATION")
    print("  Based on V320 Market Structure Analysis")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    print(f"  {NS} commodities, {ND} days, "
          f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    print("[V321] Computing mean-reversion signals...", flush=True)
    consec_dn, ret_rank, oi_capitulation, gap_rank, vol_20d = \
        compute_mean_reversion_signals(C, O, V, OI, NS, ND)
    print("  Signals done.", flush=True)

    # 5+ year backtest: 2019-2026
    bt_start = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2019-01-01'):
            bt_start = i
            break
    print(f"  Backtest start: {dates[bt_start].strftime('%Y-%m-%d')}")

    # === Signal component ablation ===
    print("\n" + "=" * 70)
    print("  SIGNAL COMPONENT ABLATION (5+ year, tn=1, hd=2, lev=1)")
    print("=" * 70)

    configs = [
        (2, True, True, True, "all signals"),
        (2, True, False, False, "consec+OI only"),
        (2, False, True, False, "consec+ret_rank only"),
        (2, False, False, True, "consec+gap only"),
        (2, True, True, False, "consec+OI+ret"),
        (2, False, False, False, "consec only"),
        (3, True, True, True, "consec>=3 all"),
        (3, False, False, False, "consec>=3 only"),
        (4, True, True, True, "consec>=4 all"),
        (5, True, True, True, "consec>=5 all"),
    ]

    for min_c, uoi, urr, ug, label in configs:
        trades, eq, dd = backtest_v321(
            C, O, H, L, V, OI, NS, ND, dates, syms,
            consec_dn, ret_rank, oi_capitulation, gap_rank, vol_20d,
            min_consec=min_c, top_n=1, hold_days=2,
            use_oi=uoi, use_gap=ug, use_ret_rank=urr,
            min_score=0.3, start_di=bt_start)
        analyze(trades, eq, dd, f"mc={min_c} {label}")

    # === Parameter sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (top_n x hold_days x leverage)")
    print("=" * 70)

    results = []
    for tn in [1, 2, 3]:
        for hd in [1, 2, 3, 5]:
            for mc in [2, 3]:
                for lev in [1, 2, 3]:
                    trades, eq, dd = backtest_v321(
                        C, O, H, L, V, OI, NS, ND, dates, syms,
                        consec_dn, ret_rank, oi_capitulation, gap_rank, vol_20d,
                        min_consec=mc, top_n=tn, hold_days=hd,
                        use_oi=True, use_gap=True, use_ret_rank=True,
                        min_score=0.3, leverage=lev,
                        start_di=bt_start)
                    if len(trades) < 5:
                        continue
                    nw = sum(1 for t in trades if t['pnl_pct'] > 0)
                    wr = nw / len(trades) * 100
                    n_days = max(1, trades[-1]['di'] - trades[0]['di'])
                    ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
                    ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
                    rets_arr = np.array(ap) / CASH0
                    sh = np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252) \
                        if np.std(rets_arr) > 0 else 0
                    results.append({
                        'tn': tn, 'hd': hd, 'mc': mc, 'lev': lev,
                        'n': len(trades), 'wr': wr, 'ann': ann, 'dd': dd, 'sh': sh,
                    })

    results.sort(key=lambda x: -x['sh'])
    print(f"\n{'TN':>3} {'HD':>3} {'MC':>3} {'L':>3} {'N':>5} {'WR':>5} "
          f"{'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 55)
    for r in results[:25]:
        print(f"{r['tn']:>3} {r['hd']:>3} {r['mc']:>3} {r['lev']:>3} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sh']:>5.2f}")

    results.sort(key=lambda x: -x['ann'])
    print(f"\n--- By Annual Return ---")
    for r in results[:15]:
        print(f"  tn={r['tn']} hd={r['hd']} mc={r['mc']} lev={r['lev']}: "
              f"{r['n']}t WR={r['wr']:.1f}% ann={r['ann']:+.1f}% "
              f"DD={r['dd']:.1f}% Sh={r['sh']:.2f}")

    # === Best config detailed yearly ===
    print("\n" + "=" * 70)
    print("  BEST CONFIG YEARLY DETAIL")
    print("=" * 70)

    best_configs = [
        (1, 2, 2, 1, "Best Sharpe candidate"),
        (1, 2, 2, 3, "Best return candidate"),
        (1, 3, 2, 1, "hd=3 variant"),
        (1, 2, 3, 1, "mc=3 variant"),
    ]
    for tn, hd, mc, lev, label in best_configs:
        trades, eq, dd = backtest_v321(
            C, O, H, L, V, OI, NS, ND, dates, syms,
            consec_dn, ret_rank, oi_capitulation, gap_rank, vol_20d,
            min_consec=mc, top_n=tn, hold_days=hd,
            use_oi=True, use_gap=True, use_ret_rank=True,
            min_score=0.3, leverage=lev,
            start_di=bt_start)
        print(f"\n  --- {label} (tn={tn} hd={hd} mc={mc} lev={lev}) ---")
        analyze(trades, eq, dd, label)

    # === Full period detail (2016-2026) ===
    print("\n" + "=" * 70)
    print("  FULL 2016-2026 PERIOD (10 years)")
    print("=" * 70)

    for tn, hd, mc, lev in [(1, 2, 2, 1), (1, 2, 2, 2), (1, 2, 2, 3)]:
        trades, eq, dd = backtest_v321(
            C, O, H, L, V, OI, NS, ND, dates, syms,
            consec_dn, ret_rank, oi_capitulation, gap_rank, vol_20d,
            min_consec=mc, top_n=tn, hold_days=hd,
            use_oi=True, use_gap=True, use_ret_rank=True,
            min_score=0.3, leverage=lev,
            start_di=60)
        print(f"\n  FULL tn={tn} hd={hd} mc={mc} lev={lev}")
        analyze(trades, eq, dd, f"full lev={lev}")

    # === Vol-targeted variant ===
    print("\n" + "=" * 70)
    print("  VOL-TARGETED VARIANT")
    print("=" * 70)

    for vt in [0.15, 0.20, 0.25, 0.30]:
        for lev in [1, 2, 3]:
            trades, eq, dd = backtest_v321(
                C, O, H, L, V, OI, NS, ND, dates, syms,
                consec_dn, ret_rank, oi_capitulation, gap_rank, vol_20d,
                min_consec=2, top_n=1, hold_days=2,
                use_oi=True, use_gap=True, use_ret_rank=True,
                min_score=0.3, leverage=lev, vol_target=vt,
                start_di=bt_start)
            if len(trades) < 5:
                continue
            nw = sum(1 for t in trades if t['pnl_pct'] > 0)
            wr = nw / len(trades) * 100
            n_days = max(1, trades[-1]['di'] - trades[0]['di'])
            ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
            ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
            rets_arr = np.array(ap) / CASH0
            sh = np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252) \
                if np.std(rets_arr) > 0 else 0
            print(f"  vt={vt:.2f} lev={lev}: {len(trades)}t WR={wr:.1f}% "
                  f"ann={ann:+.1f}% DD={dd:.1f}% Sh={sh:.2f}")

    print(f"\n[V321] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
