"""
V322: Momentum Selection + Mean-Reversion Timing — Best of Both Worlds
======================================================================
V320 showed: market is mean-reverting at daily level (AC1=-0.02, consecutive down → bounce)
V318 showed: cross-sectional momentum ranking is the dominant alpha (57% ann, Sharpe 4.18)
V321 showed: pure mean reversion alone is weak (31% ann best)

COMBINATION: Use momentum to identify WHAT to buy, mean reversion for WHEN to buy.
- Cross-sectional momentum rank → identify strongest commodities
- Only enter on pullbacks (consecutive down days in an uptrend)
- OI capitulation as bonus signal
- Volatility-targeted sizing

Backtest: Full 2016-2026 (10 years), WF 2019-2026 (7+ years)
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
from collections import defaultdict
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_v301 import load_all_data
from alpha_futures_v311 import compute_all_signals

CASH0 = 1_000_000
COMM = 0.0005


def compute_combined_signals(C, O, H, L, V, OI, NS, ND):
    """Compute momentum rank + mean-reversion timing signals."""
    # Momentum scores from V311
    mom_scores = compute_all_signals(C, O, H, L, V, NS, ND)

    # Consecutive down days
    consec_dn = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        consec = 0
        for di in range(1, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                ret = C[si, di] / C[si, di-1] - 1
                if ret < 0:
                    consec += 1
                else:
                    consec = 0
            else:
                consec = 0
            consec_dn[si, di] = consec

    # 5d return (for oversold detection)
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di-5]) and C[si, di-5] > 0:
                ret_5d[si, di] = C[si, di] / C[si, di-5] - 1

    # OI 5d change
    oi_chg = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(OI[si, di]) and not np.isnan(OI[si, di-5]):
                oi_chg[si, di] = OI[si, di] - OI[si, di-5]

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

    # Cross-sectional ret_5d rank (low = oversold)
    oversold_rank = np.full((NS, ND), np.nan)
    for di in range(ND):
        valid = ~np.isnan(ret_5d[:, di])
        if valid.sum() >= 10:
            oversold_rank[:, di] = 1.0 - pd.Series(ret_5d[:, di]).rank(
                pct=True, na_option='keep').values

    return mom_scores, consec_dn, oversold_rank, oi_chg, vol_20d


def backtest_v322(C, O, H, L, V, OI, NS, ND, dates, syms,
                  mom_scores, consec_dn, oversold_rank, oi_chg, vol_20d,
                  top_n=1, hold_days=2,
                  min_mom=0.5, min_consec=0, use_oversold=False,
                  use_oi_cap=False, atr_stop=2.5,
                  vol_target=None, leverage=1.0,
                  start_di=60, end_di=None):
    """
    Combined momentum + mean-reversion backtest.
    Momentum ranking selects WHAT, pullback timing selects WHEN.
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

        candidates = []
        for si in range(NS):
            if si in held:
                continue

            mom = mom_scores[si, di]
            if np.isnan(mom):
                continue
            if mom < min_mom:
                continue

            # Mean-reversion timing: pullback required
            cd = consec_dn[si, di]
            if cd < min_consec:
                continue

            score = mom  # base score is momentum

            # Oversold bonus (5d return rank)
            if use_oversold and not np.isnan(oversold_rank[si, di]):
                score += oversold_rank[si, di] * 0.3

            # OI capitulation bonus
            if use_oi_cap and not np.isnan(oi_chg[si, di]):
                if oi_chg[si, di] < 0 and cd >= min_consec:
                    score += 0.1

            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue

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
    print("  V322: MOMENTUM SELECTION + MEAN-REVERSION TIMING")
    print("  Best of both worlds: momentum for WHAT, pullback for WHEN")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    print(f"  {NS} commodities, {ND} days, "
          f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    print("[V322] Computing combined signals...", flush=True)
    mom_scores, consec_dn, oversold_rank, oi_chg, vol_20d = \
        compute_combined_signals(C, O, H, L, V, OI, NS, ND)
    print("  Signals done.", flush=True)

    # 5+ year backtest start
    bt_start = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2019-01-01'):
            bt_start = i
            break
    print(f"  Backtest start: {dates[bt_start].strftime('%Y-%m-%d')}")

    # === Approach comparison ===
    print("\n" + "=" * 70)
    print("  APPROACH COMPARISON (tn=1, lev=1)")
    print("=" * 70)

    configs = [
        # (min_mom, min_consec, use_oversold, use_oi_cap, hold_days, label)
        (0.5, 0, False, False, 2, "Pure momentum (V318 baseline)"),
        (0.5, 1, False, False, 2, "Mom+1d pullback"),
        (0.5, 2, False, False, 2, "Mom+2d pullback"),
        (0.5, 3, False, False, 2, "Mom+3d pullback"),
        (0.5, 0, True, False, 2, "Mom+oversold rank"),
        (0.5, 1, True, False, 2, "Mom+1dpull+oversold"),
        (0.5, 2, True, False, 2, "Mom+2dpull+oversold"),
        (0.5, 1, True, True, 2, "Mom+1dpull+oversold+OI"),
        (0.5, 2, True, True, 2, "Mom+2dpull+oversold+OI"),
        (0.6, 0, False, False, 2, "Pure mom min=0.6"),
        (0.6, 1, True, True, 2, "Mom0.6+1dpull+all"),
        (0.6, 2, True, True, 2, "Mom0.6+2dpull+all"),
        (0.5, 0, False, False, 1, "Pure mom hd=1"),
        (0.5, 1, True, True, 1, "Mom+1dpull+all hd=1"),
        (0.5, 0, False, False, 3, "Pure mom hd=3"),
        (0.5, 1, True, True, 3, "Mom+1dpull+all hd=3"),
        (0.5, 0, False, False, 5, "Pure mom hd=5"),
        (0.5, 1, True, True, 5, "Mom+1dpull+all hd=5"),
    ]

    for mm, mc, uo, uoi, hd, label in configs:
        trades, eq, dd = backtest_v322(
            C, O, H, L, V, OI, NS, ND, dates, syms,
            mom_scores, consec_dn, oversold_rank, oi_chg, vol_20d,
            min_mom=mm, min_consec=mc, use_oversold=uo,
            use_oi_cap=uoi, top_n=1, hold_days=hd,
            start_di=bt_start)
        analyze(trades, eq, dd, label)

    # === Full parameter sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP")
    print("=" * 70)

    results = []
    for tn in [1, 2, 3]:
        for hd in [1, 2, 3, 5]:
            for mc in [0, 1, 2]:
                for lev in [1, 2, 3]:
                    for uo in [False, True]:
                        trades, eq, dd = backtest_v322(
                            C, O, H, L, V, OI, NS, ND, dates, syms,
                            mom_scores, consec_dn, oversold_rank, oi_chg, vol_20d,
                            min_mom=0.5, min_consec=mc, use_oversold=uo,
                            use_oi_cap=False, top_n=tn, hold_days=hd,
                            leverage=lev, start_di=bt_start)
                        if len(trades) < 10:
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
                            'tn': tn, 'hd': hd, 'mc': mc, 'lev': lev, 'uo': uo,
                            'n': len(trades), 'wr': wr, 'ann': ann, 'dd': dd, 'sh': sh,
                        })

    results.sort(key=lambda x: -x['sh'])
    print(f"\n{'TN':>3} {'HD':>3} {'MC':>3} {'L':>3} {'UO':>3} {'N':>5} {'WR':>5} "
          f"{'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 60)
    for r in results[:25]:
        print(f"{r['tn']:>3} {r['hd']:>3} {r['mc']:>3} {r['lev']:>3} "
              f"{'Y' if r['uo'] else 'N':>3} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sh']:>5.2f}")

    results.sort(key=lambda x: -x['ann'])
    print(f"\n--- By Annual Return ---")
    for r in results[:15]:
        print(f"  tn={r['tn']} hd={r['hd']} mc={r['mc']} lev={r['lev']} uo={'Y' if r['uo'] else 'N'}: "
              f"{r['n']}t WR={r['wr']:.1f}% ann={r['ann']:+.1f}% "
              f"DD={r['dd']:.1f}% Sh={r['sh']:.2f}")

    # === Best configs yearly detail ===
    print("\n" + "=" * 70)
    print("  BEST CONFIGS — YEARLY DETAIL (2019-2026)")
    print("=" * 70)

    best_configs = [
        (1, 2, 0, False, 1, "V318 baseline"),
        (1, 2, 1, True, 1, "Mom+1dpull+oversold"),
        (1, 2, 2, True, 1, "Mom+2dpull+oversold"),
        (1, 1, 1, True, 2, "hd=1 variant lev=2"),
        (1, 5, 0, False, 1, "hd=5 pure mom"),
    ]
    for hd, mc, uo, lev_yn, lev, label in best_configs:
        trades, eq, dd = backtest_v322(
            C, O, H, L, V, OI, NS, ND, dates, syms,
            mom_scores, consec_dn, oversold_rank, oi_chg, vol_20d,
            min_mom=0.5, min_consec=mc, use_oversold=uo,
            use_oi_cap=False, top_n=1, hold_days=hd,
            leverage=lev, start_di=bt_start)
        print(f"\n  --- {label} (hd={hd} mc={mc} uo={uo} lev={lev}) ---")
        analyze(trades, eq, dd, label)

    # === Full 2016-2026 ===
    print("\n" + "=" * 70)
    print("  FULL 2016-2026 (10 years)")
    print("=" * 70)

    for hd, mc, uo, lev in [(2, 0, False, 1), (2, 1, True, 1), (2, 0, False, 2),
                             (2, 1, True, 2), (2, 0, False, 3), (2, 1, True, 3)]:
        trades, eq, dd = backtest_v322(
            C, O, H, L, V, OI, NS, ND, dates, syms,
            mom_scores, consec_dn, oversold_rank, oi_chg, vol_20d,
            min_mom=0.5, min_consec=mc, use_oversold=uo,
            use_oi_cap=False, top_n=1, hold_days=hd,
            leverage=lev, start_di=60)
        print(f"\n  FULL hd={hd} mc={mc} uo={uo} lev={lev}")
        analyze(trades, eq, dd, f"full mc={mc} lev={lev}")

    print(f"\n[V322] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
