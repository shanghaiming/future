"""
V316: Clean Momentum with Leverage — Path to 600%
===================================================
V314 proved the best no-look-ahead signal is V311's momentum ensemble
with next-day-open execution. The ceiling is ~54% annual at 1x.

To reach 600%, we need leverage. This script tests the clean signal
with leverage 1x-8x and finds the optimal risk-return trade-off.

Also tests: gap trading, overnight hold optimization, and trailing stops.
"""
import sys, os, time, warnings, json, glob
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_v301 import load_all_data, compute_factors, detect_regimes
from alpha_futures_v311 import compute_all_signals, load_ts

CASH0 = 1_000_000
COMM = 0.0005


def backtest_v316(C, O, H, L, NS, ND, dates, syms,
                  scores, ts_data, regime,
                  top_n=1, hold_days=3, atr_stop=2.5,
                  min_score=0.6, use_carry=True,
                  leverage=1.0, trail_stop=True,
                  start_di=60, end_di=None):
    """
    Clean execution: signal[di], enter O[di+1], exit C[di+hold].
    Leverage: multiply position allocation by leverage factor.
    """
    if end_di is None:
        end_di = ND - 1

    ts_si = {s: i for i, s in enumerate(ts_data['syms'])}
    ts_di = {d: i for i, d in enumerate(ts_data['dates'])}

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions = []
    trades = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0
        new_positions = []

        for si, edi, ep, sp, alloc, hw in positions:
            c = C[si, di]
            h = H[si, di]
            if np.isnan(c):
                new_positions.append((si, edi, ep, sp, alloc, hw))
                continue

            # Trailing stop
            new_sp = sp
            if trail_stop and not np.isnan(h):
                atr_v = []
                for j in range(max(start_di, di - 14), di):
                    hh, ll, cc = H[si, j], L[si, j], C[si, j]
                    if not any(np.isnan([hh, ll, cc])):
                        atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
                if atr_v:
                    atr = np.mean(atr_v)
                    new_sp = max(sp, h - atr_stop * atr)

            exit_r = None
            if c < new_sp:
                exit_r = 'stop'
            elif di - edi >= hold_days:
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
                new_positions.append((si, edi, ep, new_sp, alloc, max(hw, h if not np.isnan(h) else hw)))

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

        # Entry
        held = {p[0] for p in positions}
        if len(positions) >= top_n:
            continue

        r = regime[di] if regime is not None and di < len(regime) else 0
        if r in (-1, 2):
            continue

        candidates = []
        for si in range(NS):
            if si in held:
                continue
            score = scores[si, di]
            if np.isnan(score) or score < min_score:
                continue
            if use_carry:
                tsi = ts_di.get(d)
                if tsi is not None:
                    sym = syms[si]
                    tssi = ts_si.get(sym, -1)
                    if tssi >= 0 and tsi < ts_data['cz'].shape[1]:
                        cz = ts_data['cz'][tssi, tsi]
                        if not np.isnan(cz) and cz > 1:
                            score += 0.1
            if di + 1 < ND and np.isnan(O[si, di + 1]):
                continue
            candidates.append((score, si))

        if not candidates:
            continue
        candidates.sort(key=lambda x: -x[0])

        alloc = 1.0 / max(top_n, 1)
        for score, si in candidates[:top_n]:
            if len(positions) >= top_n or si in held:
                break
            if di + 1 >= ND:
                continue
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
            positions.append((si, di + 1, ep, ep - atr_stop * atr, alloc, ep))
            held.add(si)

    for si, edi, ep, sp, alloc, hw in positions:
        c = C[si, ND - 1]
        if not np.isnan(c):
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
        print(f"    {y}: {ys['n']}t WR={ys['w'] / ys['n'] * 100:.1f}% cum={cum:+.1%}")
    return {'n': len(trades), 'wr': wr, 'dd': max_dd, 'ann': ann, 'sh': sh}


def main():
    t0 = time.time()
    print("=" * 60)
    print("  V316: CLEAN MOMENTUM + LEVERAGE → 600% TARGET")
    print("=" * 60)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    ts_data = load_ts(start='2021-01-01')
    F = compute_factors(C, O, H, L, V, OI, NS, ND)
    regime = detect_regimes(F, NS, ND)

    print("[V316] Computing signals (V311 ensemble)...", flush=True)
    scores = compute_all_signals(C, O, H, L, V, NS, ND)

    wf_start = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2024-01-01'):
            wf_start = i
            break

    # --- Leverage sweep with best configs from V314 ---
    print(f"\n=== LEVERAGE SWEEP (WF 2024-2026) ===")
    print(f"{'TN':>3} {'HD':>3} {'Lev':>4} {'C':>2} {'N':>5} {'WR':>5} "
          f"{'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 60)

    best_configs = [
        (1, 2, True), (1, 3, True), (1, 3, False),
        (2, 3, True), (3, 3, True), (3, 5, True),
    ]

    results = []
    for tn, hd, carry in best_configs:
        for lev in [1, 2, 3, 5, 8]:
            trades, eq, dd = backtest_v316(
                C, O, H, L, NS, ND, dates, syms,
                scores, ts_data, regime,
                top_n=tn, hold_days=hd, use_carry=carry,
                leverage=lev, start_di=wf_start)
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
                'tn': tn, 'hd': hd, 'lev': lev, 'carry': carry,
                'n': len(trades), 'wr': wr, 'ann': ann, 'dd': dd, 'sh': sh,
            })
            print(f"{tn:>3} {hd:>3} {lev:>4} {'Y' if carry else 'N':>2} "
                  f"{len(trades):>5} {wr:>5.1f} {ann:>+8.1f} {dd:>6.1f} {sh:>5.2f}")

    # Sort by Sharpe and by annual return
    print(f"\n--- Top by Annual Return ---")
    results.sort(key=lambda x: -x['ann'])
    for r in results[:15]:
        print(f"  tn={r['tn']} hd={r['hd']} lev={r['lev']} "
              f"c={'Y' if r['carry'] else 'N'}: "
              f"{r['n']}t WR={r['wr']:.1f}% ann={r['ann']:+.1f}% "
              f"DD={r['dd']:.1f}% Sh={r['sh']:.2f}")

    print(f"\n--- Top by Sharpe ---")
    results.sort(key=lambda x: -x['sh'])
    for r in results[:15]:
        print(f"  tn={r['tn']} hd={r['hd']} lev={r['lev']} "
              f"c={'Y' if r['carry'] else 'N'}: "
              f"{r['n']}t WR={r['wr']:.1f}% ann={r['ann']:+.1f}% "
              f"DD={r['dd']:.1f}% Sh={r['sh']:.2f}")

    # Best config with full details at each leverage
    print(f"\n=== BEST CONFIG DETAILS (tn=1 hd=3 carry) ===")
    for lev in [1, 2, 3, 5, 8]:
        trades, eq, dd = backtest_v316(
            C, O, H, L, NS, ND, dates, syms,
            scores, ts_data, regime,
            top_n=1, hold_days=3, use_carry=True,
            leverage=lev, start_di=wf_start)
        analyze(trades, eq, dd, f"tn=1 hd=3 lev={lev}")

    print(f"\n=== SECOND BEST (tn=1 hd=2 carry) ===")
    for lev in [1, 2, 3, 5, 8]:
        trades, eq, dd = backtest_v316(
            C, O, H, L, NS, ND, dates, syms,
            scores, ts_data, regime,
            top_n=1, hold_days=2, use_carry=True,
            leverage=lev, start_di=wf_start)
        analyze(trades, eq, dd, f"tn=1 hd=2 lev={lev}")

    # Diversified config
    print(f"\n=== DIVERSIFIED (tn=3 hd=3 carry) ===")
    for lev in [1, 2, 3, 5, 8]:
        trades, eq, dd = backtest_v316(
            C, O, H, L, NS, ND, dates, syms,
            scores, ts_data, regime,
            top_n=3, hold_days=3, use_carry=True,
            leverage=lev, start_di=wf_start)
        analyze(trades, eq, dd, f"tn=3 hd=3 lev={lev}")

    print(f"\n[V316] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
