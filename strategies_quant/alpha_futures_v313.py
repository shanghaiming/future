"""
V313: Momentum Acceleration + Volatility Scaled (600% target, no leverage)
==========================================================================
V312 showed IC-adaptive weighting fails because cross-sectional IC is near zero.
V311's simple momentum ranking works because momentum PERSISTS (underreaction).

Key innovations for V313:
1. Momentum acceleration (2nd derivative) — momentum that's getting stronger
2. Volatility-scaled returns — prefer commodities with bigger expected moves
3. Trend persistence — momentum with low choppiness = more likely to continue
4. Close-to-close execution (proven by V311)
5. Dynamic hold via trailing signal
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_v301 import load_all_data, compute_factors, detect_regimes

CASH0 = 1_000_000
COMM = 0.0005


def compute_accel_signals(C, O, H, L, V, NS, ND):
    """
    Compute momentum + acceleration + persistence signals.
    Return scores where higher = more likely to continue rising.
    """
    scores = np.full((NS, ND), np.nan)

    # --- Component 1: Multi-period momentum rank ---
    mom_ranks = {}
    for period in [3, 5, 10, 20]:
        r = np.full((NS, ND), np.nan)
        for di in range(period, ND):
            rets = np.full(NS, np.nan)
            for si in range(NS):
                c0, c1 = C[si, di - period], C[si, di]
                if not np.isnan(c0) and not np.isnan(c1) and c0 > 0:
                    rets[si] = (c1 - c0) / c0
            valid = ~np.isnan(rets)
            if valid.sum() >= 5:
                r[:, di] = pd.Series(rets).rank(pct=True, na_option='keep').values
        mom_ranks[period] = r

    # --- Component 2: Momentum ACCELERATION rank ---
    # acceleration = current momentum minus previous period's momentum
    accel_ranks = {}
    for period in [5, 10, 20]:
        accel = np.full((NS, ND), np.nan)
        for di in range(period + 3, ND):
            deltas = np.full(NS, np.nan)
            for si in range(NS):
                # Current momentum
                c_now = C[si, di]
                c_prev = C[si, di - period]
                c_3ago = C[si, di - 3]
                if np.isnan(c_now) or np.isnan(c_prev) or np.isnan(c_3ago):
                    continue
                if c_prev <= 0 or c_3ago <= 0:
                    continue
                mom_now = (c_now - c_prev) / c_prev
                mom_3ago = (c_3ago - C[si, di - period - 3]) / C[si, di - period - 3] \
                    if not np.isnan(C[si, di - period - 3]) and C[si, di - period - 3] > 0 else np.nan
                if not np.isnan(mom_3ago):
                    deltas[si] = mom_now - mom_3ago
            valid = ~np.isnan(deltas)
            if valid.sum() >= 5:
                accel[:, di] = pd.Series(deltas).rank(pct=True, na_option='keep').values
        accel_ranks[period] = accel

    # --- Component 3: Trend persistence (low choppiness) ---
    # Choppiness index: lower = more trending, higher = choppier
    # We want low choppiness (trending) commodities with high momentum
    persistence_rank = np.full((NS, ND), np.nan)
    chop_window = 14
    for di in range(chop_window, ND):
        chops = np.full(NS, np.nan)
        for si in range(NS):
            highs = H[si, di - chop_window:di]
            lows = L[si, di - chop_window:di]
            closes = C[si, di - chop_window:di]
            if np.isnan(highs).any() or np.isnan(lows).any() or np.isnan(closes).any():
                continue
            atr_sum = np.sum(np.maximum(highs - lows, 0))
            hh = np.max(highs)
            ll = np.min(lows)
            if hh - ll > 0 and atr_sum > 0:
                # Choppiness: 100 * LOG10(sum(ATR) / (HH - LL)) / LOG10(period)
                chops[si] = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(chop_window)
        valid = ~np.isnan(chops)
        if valid.sum() >= 5:
            # Lower choppiness = better, so rank ascending (1 - rank)
            persistence_rank[:, di] = 1 - pd.Series(chops).rank(pct=True, na_option='keep').values

    # --- Component 4: Volatility-scaled momentum ---
    # Higher vol commodities have bigger moves when momentum is correct
    voladj_mom = np.full((NS, ND), np.nan)
    for di in range(20, ND):
        vals = np.full(NS, np.nan)
        for si in range(NS):
            # 5-day return / 20-day vol = risk-adjusted momentum
            c0, c1 = C[si, di - 5], C[si, di]
            if np.isnan(c0) or np.isnan(c1) or c0 <= 0:
                continue
            ret = (c1 - c0) / c0
            # 20-day vol
            rets = []
            for j in range(max(1, di - 20), di):
                if not np.isnan(C[si, j]) and not np.isnan(C[si, j - 1]) and C[si, j - 1] > 0:
                    rets.append(C[si, j] / C[si, j - 1] - 1)
            if len(rets) >= 10:
                vol = np.std(rets)
                if vol > 1e-10:
                    vals[si] = ret / vol  # Sharpe-like measure
        valid = ~np.isnan(vals)
        if valid.sum() >= 5:
            voladj_mom[:, di] = pd.Series(vals).rank(pct=True, na_option='keep').values

    # --- Combine ---
    # Weights: momentum is primary, acceleration and persistence are secondary
    w_mom = [0.15, 0.20, 0.15, 0.10]  # mom3, mom5, mom10, mom20
    w_accel = [0.10, 0.10, 0.05]       # accel5, accel10, accel20
    w_persist = 0.10                     # persistence
    w_voladj = 0.05                      # vol-adj mom

    for di in range(25, ND):
        components = []
        weights = []

        for i, period in enumerate([3, 5, 10, 20]):
            v = mom_ranks[period][:, di]
            if not np.isnan(v).all():
                components.append(np.nan_to_num(v, nan=0.5))
                weights.append(w_mom[i])

        for i, period in enumerate([5, 10, 20]):
            v = accel_ranks[period][:, di]
            if not np.isnan(v).all():
                components.append(np.nan_to_num(v, nan=0.5))
                weights.append(w_accel[i])

        v = persistence_rank[:, di]
        if not np.isnan(v).all():
            components.append(np.nan_to_num(v, nan=0.5))
            weights.append(w_persist)

        v = voladj_mom[:, di]
        if not np.isnan(v).all():
            components.append(np.nan_to_num(v, nan=0.5))
            weights.append(w_voladj)

        if components:
            total_w = sum(weights)
            scores[:, di] = sum(c * w for c, w in zip(components, weights)) / total_w

    return scores, {'mom_ranks': mom_ranks, 'accel_ranks': accel_ranks,
                    'persistence': persistence_rank, 'voladj': voladj_mom}


def backtest_v313(C, O, H, L, NS, ND, dates, syms,
                  scores, regime,
                  top_n=1, hold_days=1, atr_stop=2.5,
                  min_score=0.6, trail_signal=True,
                  start_di=60, end_di=None):
    if end_di is None:
        end_di = ND

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions = []
    trades = []

    for di in range(max(start_di, 1), end_di):
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
            elif trail_signal and not np.isnan(scores[si, di]):
                # Exit early if signal decayed significantly
                if scores[si, di] < 0.3:
                    exit_r = 'signal'
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

        candidates = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(C[si, di]) or np.isnan(C[si, di - 1]):
                continue
            score = scores[si, di]
            if np.isnan(score) or score < min_score:
                continue
            candidates.append((score, si))

        if not candidates:
            continue
        candidates.sort(key=lambda x: -x[0])

        alloc = 1.0 / max(top_n, 1)
        for score, si in candidates[:top_n]:
            if len(positions) >= top_n or si in held:
                break
            # Enter at today's close price (signal from today's close)
            ep = C[si, di]
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
            positions.append((si, di, ep, ep - atr_stop * atr, alloc))
            held.add(si)

    # Close remaining
    for si, edi, ep, sp, alloc in positions:
        c = C[si, ND - 1]
        if not np.isnan(c):
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl

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
    print("  V313: MOMENTUM ACCELERATION + VOL SCALED")
    print("=" * 60)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    F = compute_factors(C, O, H, L, V, OI, NS, ND)
    regime = detect_regimes(F, NS, ND)

    print("[V313] Computing acceleration signals...", flush=True)
    scores, components = compute_accel_signals(C, O, H, L, V, NS, ND)

    # Find WF start
    wf_start = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2024-01-01'):
            wf_start = i
            break

    # --- Full in-sample ---
    print(f"\n--- Full In-Sample ---")
    for tn, hd, ms, trail in [(1, 1, 0.5, True), (1, 1, 0.6, True),
                               (1, 1, 0.6, False), (1, 2, 0.6, True),
                               (1, 3, 0.6, True), (3, 1, 0.5, True),
                               (3, 3, 0.6, True)]:
        trades, eq, dd = backtest_v313(
            C, O, H, L, NS, ND, dates, syms, scores, regime,
            top_n=tn, hold_days=hd, min_score=ms, trail_signal=trail)
        analyze(trades, eq, dd, f"IS tn={tn} hd={hd} ms={ms} {'TR' if trail else 'FX'}")

    # --- Walk-forward ---
    print(f"\n--- Walk-Forward 2024-2026 ---")
    for tn, hd, ms, trail in [(1, 1, 0.5, True), (1, 1, 0.6, True),
                               (1, 1, 0.6, False), (1, 2, 0.6, True),
                               (1, 3, 0.6, True), (3, 1, 0.5, True),
                               (3, 3, 0.6, True)]:
        trades, eq, dd = backtest_v313(
            C, O, H, L, NS, ND, dates, syms, scores, regime,
            top_n=tn, hold_days=hd, min_score=ms, trail_signal=trail,
            start_di=wf_start)
        analyze(trades, eq, dd, f"WF tn={tn} hd={hd} ms={ms} {'TR' if trail else 'FX'}")

    # --- Sweep ---
    print(f"\n--- Parameter Sweep (WF 2024-2026) ---")
    results = []
    for tn in [1, 2, 3]:
        for hd in [1, 2, 3, 5]:
            for ms in [0.5, 0.6, 0.7]:
                for trail in [True, False]:
                    trades, eq, dd = backtest_v313(
                        C, O, H, L, NS, ND, dates, syms, scores, regime,
                        top_n=tn, hold_days=hd, min_score=ms,
                        trail_signal=trail, start_di=wf_start)
                    if len(trades) < 5:
                        continue
                    nw = sum(1 for t in trades if t['pnl_pct'] > 0)
                    wr = nw / len(trades) * 100
                    n_days = max(1, trades[-1]['di'] - trades[0]['di'])
                    ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
                    ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
                    rets_arr = np.array(ap) / CASH0
                    sh = np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252) if np.std(rets_arr) > 0 else 0
                    results.append({
                        'tn': tn, 'hd': hd, 'ms': ms, 'trail': trail,
                        'n': len(trades), 'wr': wr, 'ann': ann, 'dd': dd, 'sh': sh,
                    })

    results.sort(key=lambda x: -x['ann'])
    print(f"\n{'TN':>3} {'HD':>3} {'MS':>4} {'TR':>3} {'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 55)
    for r in results[:20]:
        print(f"{r['tn']:>3} {r['hd']:>3} {r['ms']:>4.1f} {'Y' if r['trail'] else 'N':>3} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} {r['dd']:>6.1f} {r['sh']:>5.2f}")

    results.sort(key=lambda x: -x['sh'])
    print(f"\n--- By Sharpe ---")
    for r in results[:10]:
        print(f"  tn={r['tn']} hd={r['hd']} ms={r['ms']} tr={'Y' if r['trail'] else 'N'}: "
              f"{r['n']}t WR={r['wr']:.1f}% ann={r['ann']:+.1f}% DD={r['dd']:.1f}% Sh={r['sh']:.2f}")

    print(f"\n[V313] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
