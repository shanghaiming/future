"""
V317: Multi-Strategy Fusion — Maximum Alpha Without Look-Ahead
================================================================
Instead of trying to improve one signal, combine multiple independent alpha sources:
1. Momentum: cross-sectional rank (V311 signal, clean execution)
2. Carry: term structure backwardation (from V308/V309)
3. Gap reversal: buy overnight gap downs, sell gap ups
4. Volatility breakout: buy when price exceeds Bollinger upper band

Run them as parallel strategies with separate capital.
Combined alpha = sum of individual alphas if uncorrelated.
"""
import sys, os, time, warnings, json, glob
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_v301 import load_all_data, compute_factors, detect_regimes
from alpha_futures_v311 import compute_all_signals, load_ts

CASH0 = 1_000_000
COMM = 0.0005


def compute_gap_signals(C, O, NS, ND):
    """
    Gap signal: today's open vs yesterday's close.
    Gap down = buy signal (mean reversion).
    Return: gap z-score rank (higher = bigger gap down).
    """
    gap_rank = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        gaps = np.full(NS, np.nan)
        for si in range(NS):
            c_prev = C[si, di - 1]
            o_now = O[si, di]
            if not np.isnan(c_prev) and not np.isnan(o_now) and c_prev > 0:
                gaps[si] = (o_now - c_prev) / c_prev
        valid = ~np.isnan(gaps)
        if valid.sum() >= 5:
            # Rank: bigger gap DOWN = higher rank (more oversold)
            gap_rank[:, di] = 1 - pd.Series(gaps).rank(pct=True, na_option='keep').values
    return gap_rank


def compute_breakout_signals(C, H, L, NS, ND, window=20, num_std=2.0):
    """
    Bollinger Band breakout: price above upper band = bullish breakout.
    """
    bb_rank = np.full((NS, ND), np.nan)
    for di in range(window, ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            prices = C[si, di - window:di]
            c = C[si, di]
            if np.isnan(c) or np.isnan(prices).any():
                continue
            mean_p = np.nanmean(prices)
            std_p = np.nanstd(prices)
            if std_p > 0:
                scores[si] = (c - mean_p) / (num_std * std_p)
        valid = ~np.isnan(scores)
        if valid.sum() >= 5:
            bb_rank[:, di] = pd.Series(scores).rank(pct=True, na_option='keep').values
    return bb_rank


def backtest_multi(C, O, H, L, NS, ND, dates, syms,
                   mom_scores, gap_scores, bb_scores,
                   ts_data, regime,
                   w_mom=0.5, w_gap=0.25, w_bb=0.25,
                   top_n=1, hold_days=3, atr_stop=2.5,
                   min_score=0.6, use_carry=True,
                   leverage=1.0, start_di=60, end_di=None):
    """Multi-strategy fusion backtest with clean execution."""
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
                profit = equity * alloc * leverage * pnl
                daily_pnl += profit
                trades.append({
                    'pnl_abs': profit, 'pnl_pct': pnl * 100 * leverage,
                    'days': di - edi + 1, 'di': di, 'year': d.year,
                    'sym': syms[si], 'reason': exit_r, 'strat': 'multi',
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

        r = regime[di] if regime is not None and di < len(regime) else 0
        if r in (-1, 2):
            continue

        # Combine scores
        candidates = []
        for si in range(NS):
            if si in held:
                continue
            m = mom_scores[si, di]
            g = gap_scores[si, di]
            b = bb_scores[si, di]
            if np.isnan(m):
                continue

            score = w_mom * np.nan_to_num(m, nan=0.5)
            if not np.isnan(g):
                score += w_gap * g
            if not np.isnan(b):
                score += w_bb * b

            if use_carry:
                tsi = ts_di.get(d)
                if tsi is not None:
                    sym = syms[si]
                    tssi = ts_si.get(sym, -1)
                    if tssi >= 0 and tsi < ts_data['cz'].shape[1]:
                        cz = ts_data['cz'][tssi, tsi]
                        if not np.isnan(cz) and cz > 1:
                            score += 0.1

            if score < min_score:
                continue
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
            positions.append((si, di + 1, ep, ep - atr_stop * atr, alloc))
            held.add(si)

    for si, edi, ep, sp, alloc in positions:
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
    print("  V317: MULTI-STRATEGY FUSION")
    print("=" * 60)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    ts_data = load_ts(start='2021-01-01')
    F = compute_factors(C, O, H, L, V, OI, NS, ND)
    regime = detect_regimes(F, NS, ND)

    print("[V317] Computing signals...", flush=True)
    mom_scores = compute_all_signals(C, O, H, L, V, NS, ND)
    gap_scores = compute_gap_signals(C, O, NS, ND)
    bb_scores = compute_breakout_signals(C, H, L, NS, ND)
    print("  Signals done.", flush=True)

    wf_start = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2024-01-01'):
            wf_start = i
            break

    # --- Weight sweep ---
    print(f"\n=== Weight Sweep (WF 2024-2026) ===")
    weight_configs = [
        (1.0, 0.0, 0.0, "mom-only"),
        (0.0, 1.0, 0.0, "gap-only"),
        (0.0, 0.0, 1.0, "bb-only"),
        (0.5, 0.25, 0.25, "balanced"),
        (0.6, 0.2, 0.2, "mom-heavy"),
        (0.4, 0.3, 0.3, "diverse"),
        (0.7, 0.15, 0.15, "mom-dom"),
        (0.5, 0.5, 0.0, "mom+gap"),
        (0.5, 0.0, 0.5, "mom+bb"),
    ]

    results = []
    for tn in [1, 2, 3]:
        for hd in [2, 3, 5]:
            for wm, wg, wb, wlabel in weight_configs:
                for lev in [1, 2, 3]:
                    trades, eq, dd = backtest_multi(
                        C, O, H, L, NS, ND, dates, syms,
                        mom_scores, gap_scores, bb_scores,
                        ts_data, regime,
                        w_mom=wm, w_gap=wg, w_bb=wb,
                        top_n=tn, hold_days=hd, leverage=lev,
                        start_di=wf_start)
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
                        'tn': tn, 'hd': hd, 'wm': wm, 'wg': wg, 'wb': wb,
                        'wlabel': wlabel, 'lev': lev,
                        'n': len(trades), 'wr': wr, 'ann': ann, 'dd': dd, 'sh': sh,
                    })

    results.sort(key=lambda x: -x['ann'])
    print(f"\n{'TN':>3} {'HD':>3} {'W':>12} {'L':>3} {'N':>5} {'WR':>5} "
          f"{'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 65)
    for r in results[:25]:
        print(f"{r['tn']:>3} {r['hd']:>3} {r['wlabel']:>12} {r['lev']:>3} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sh']:>5.2f}")

    results.sort(key=lambda x: -x['sh'])
    print(f"\n--- By Sharpe ---")
    for r in results[:15]:
        print(f"  {r['wlabel']:>12} tn={r['tn']} hd={r['hd']} lev={r['lev']}: "
              f"{r['n']}t WR={r['wr']:.1f}% ann={r['ann']:+.1f}% "
              f"DD={r['dd']:.1f}% Sh={r['sh']:.2f}")

    # Best detail runs
    print(f"\n=== MOM-ONLY DETAIL ===")
    for lev in [1, 2, 3, 5]:
        trades, eq, dd = backtest_multi(
            C, O, H, L, NS, ND, dates, syms,
            mom_scores, gap_scores, bb_scores,
            ts_data, regime,
            w_mom=1.0, w_gap=0.0, w_bb=0.0,
            top_n=1, hold_days=2, leverage=lev,
            start_di=wf_start)
        analyze(trades, eq, dd, f"mom-only lev={lev}")

    print(f"\n=== MOM+GAP DETAIL ===")
    for lev in [1, 2, 3, 5]:
        trades, eq, dd = backtest_multi(
            C, O, H, L, NS, ND, dates, syms,
            mom_scores, gap_scores, bb_scores,
            ts_data, regime,
            w_mom=0.5, w_gap=0.5, w_bb=0.0,
            top_n=1, hold_days=2, leverage=lev,
            start_di=wf_start)
        analyze(trades, eq, dd, f"mom+gap lev={lev}")

    print(f"\n[V317] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
