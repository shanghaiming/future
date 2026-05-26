"""
V314: Look-Ahead Bias Audit on V311
====================================
V311 uses scores computed from C[si,di] (today's close) but enters at O[si,di] (today's open).
Since close > open chronologically, this is LOOK-AHEAD BIAS.

This script tests:
1. V311 original (with look-ahead) — expect 236% WF annual
2. V311 corrected: signal from di-1, enter at O[di] — removes look-ahead
3. V311 corrected: signal from di, enter at O[di+1] — next-day execution
4. V311 corrected: signal from di-1, enter at C[di] — close-to-close with 1-day lag

If the 236% drops to near zero, V311's edge was entirely look-ahead.
If it stays strong, the signal genuinely predicts future returns.
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_v301 import load_all_data, compute_factors, detect_regimes
from alpha_futures_v311 import compute_all_signals, load_ts, analyze

CASH0 = 1_000_000
COMM = 0.0005


def backtest_v314(C, O, H, L, NS, ND, dates, syms,
                  scores, ts_data, regime,
                  top_n=1, hold_days=1, atr_stop=2.5,
                  min_score=0.6, use_carry_boost=False,
                  signal_lag=0, entry_mode='open',
                  start_di=60, end_di=None):
    """
    signal_lag: 0 = use scores[di] (original V311), 1 = use scores[di-1]
    entry_mode: 'open' = O[di+lag], 'close' = C[di+lag], 'next_open' = O[di+1]
    """
    if end_di is None:
        end_di = ND

    ts_si = {s: i for i, s in enumerate(ts_data['syms'])}
    ts_di = {d: i for i, d in enumerate(ts_data['dates'])}

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions = []
    trades = []

    for di in range(max(start_di + signal_lag, 1), end_di):
        d = dates[di]
        daily_pnl = 0
        new_positions = []

        # Exit logic
        for si, edi, ep, sp, alloc in positions:
            c = C[si, di]
            if np.isnan(c):
                new_positions.append((si, edi, ep, sp, alloc))
                continue
            exit_r = None
            if c < sp:
                exit_r = 'stop'
            elif di - edi >= hold_days:
                exit_r = 'hold'
            if exit_r:
                pnl = (c - ep) / ep - COMM
                profit = equity * alloc * pnl
                daily_pnl += profit
                trades.append({
                    'pnl_abs': profit, 'pnl_pct': pnl * 100,
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

        # Entry logic
        held = {p[0] for p in positions}
        if len(positions) >= top_n:
            continue

        # Regime filter
        r = regime[di] if regime is not None and di < len(regime) else 0
        if r in (-1, 2):
            continue

        # Use lagged signal
        signal_di = di - signal_lag
        if signal_di < start_di:
            continue

        candidates = []
        for si in range(NS):
            if si in held:
                continue
            score = scores[si, signal_di]
            if np.isnan(score) or score < min_score:
                continue

            if use_carry_boost:
                tsi = ts_di.get(d)
                if tsi is not None:
                    sym = syms[si]
                    tssi = ts_si.get(sym, -1)
                    if tssi >= 0 and tsi < ts_data['cz'].shape[1]:
                        cz = ts_data['cz'][tssi, tsi]
                        if not np.isnan(cz) and cz > 1:
                            score += 0.1

            candidates.append((score, si))

        if not candidates:
            continue
        candidates.sort(key=lambda x: -x[0])

        alloc = 1.0 / max(top_n, 1)
        for score, si in candidates[:top_n]:
            if len(positions) >= top_n or si in held:
                break

            # Entry price depends on mode
            if entry_mode == 'open':
                ep = O[si, di]
            elif entry_mode == 'close':
                ep = C[si, di]
            elif entry_mode == 'next_open':
                if di + 1 >= ND:
                    continue
                ep = O[si, di + 1]
            else:
                ep = O[si, di]

            if np.isnan(ep) or ep <= 0:
                continue

            # ATR stop (use data up to signal_di to avoid look-ahead)
            atr_v = []
            for j in range(max(start_di, signal_di - 14), signal_di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
            if not atr_v:
                continue
            atr = np.mean(atr_v)
            positions.append((si, di, ep, ep - atr_stop * atr, alloc))
            held.add(si)

    # Close remaining
    for si, edi, ep, sp, alloc in positions:
        c = C[si, ND - 1]
        if not np.isnan(c):
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl

    return trades, equity, max_dd


def main():
    t0 = time.time()
    print("=" * 70)
    print("  V314: LOOK-AHEAD BIAS AUDIT ON V311")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    ts_data = load_ts(start='2021-01-01')
    F = compute_factors(C, O, H, L, V, OI, NS, ND)
    regime = detect_regimes(F, NS, ND)

    print("[V314] Computing signals...", flush=True)
    scores = compute_all_signals(C, O, H, L, V, NS, ND)

    wf_start = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2024-01-01'):
            wf_start = i
            break

    # --- Test different signal/entry combinations ---
    configs = [
        # (label, signal_lag, entry_mode, description)
        ("V311-ORIG", 0, 'open', "Original: score[di] enter O[di] (LOOK-AHEAD)"),
        ("LAG1-O", 1, 'open', "Fixed: score[di-1] enter O[di] (no look-ahead)"),
        ("LAG1-C", 1, 'close', "Fixed: score[di-1] enter C[di] (close-to-close)"),
        ("LAG0-NO", 0, 'next_open', "score[di] enter O[di+1] (next-day open)"),
        ("LAG1-NO", 1, 'next_open', "score[di-1] enter O[di+1] (conservative)"),
    ]

    for tn in [1, 3]:
        for hd in [1, 2, 3]:
            print(f"\n{'='*60}")
            print(f"  Config: top_n={tn} hold_days={hd}")
            print(f"{'='*60}")

            for label, lag, mode, desc in configs:
                print(f"\n  {label}: {desc}")

                # Full in-sample
                trades, eq, dd = backtest_v314(
                    C, O, H, L, NS, ND, dates, syms,
                    scores, ts_data, regime,
                    top_n=tn, hold_days=hd,
                    signal_lag=lag, entry_mode=mode)
                analyze(trades, eq, dd, "  IS")

                # Walk-forward
                if wf_start:
                    trades, eq, dd = backtest_v314(
                        C, O, H, L, NS, ND, dates, syms,
                        scores, ts_data, regime,
                        top_n=tn, hold_days=hd,
                        signal_lag=lag, entry_mode=mode,
                        start_di=wf_start)
                    analyze(trades, eq, dd, "  WF")

    print(f"\n[V314] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
