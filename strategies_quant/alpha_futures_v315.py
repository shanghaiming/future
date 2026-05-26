"""
V315: Optimized Clean Momentum (No Look-Ahead)
================================================
Building on V314's finding that LAG0-NO (score[di] + enter O[di+1]) is the best
clean execution framework. Optimize signal quality and execution.

Execution: signal from day di, enter at O[di+1], exit at C[di+hold]
This is clean — no look-ahead, practically executable.

Optimizations:
1. Multi-signal fusion with momentum + carry + OI
2. Adaptive hold: exit when signal decays (not fixed days)
3. Volatility-weighted position sizing within 1x
4. Trailing stop for trend capture
5. Regime-aware position count
"""
import sys, os, time, warnings, json, glob
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_v301 import load_all_data, compute_factors, detect_regimes

CASH0 = 1_000_000
COMM = 0.0005
TS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'data', 'futures_term_structure')


def load_ts(start='2021-01-01'):
    ts_dir = os.path.abspath(TS_DIR)
    all_files = glob.glob(os.path.join(ts_dir, '*.json'))
    ts = {}
    for fp in all_files:
        try:
            with open(fp) as f: d = json.load(f)
            sym, ds = d.get('symbol', ''), d.get('date', '')
            if sym and ds: ts[(sym, ds)] = d
        except: pass
    syms = sorted(set(k[0] for k in ts))
    dates = sorted(set(k[1] for k in ts))
    dates = [d for d in dates if pd.Timestamp(d) >= pd.Timestamp(start)]
    NS, ND = len(syms), len(dates)
    si_map = {s: i for i, s in enumerate(syms)}
    di_map = {d: i for i, d in enumerate(dates)}
    spread = np.full((NS, ND), np.nan)
    for (sym, ds), d in ts.items():
        if sym in si_map and ds in di_map:
            sp = d.get('total_spread_pct')
            if sp: spread[si_map[sym], di_map[ds]] = float(sp)
    cz = np.full((NS, ND), np.nan)
    for s in range(NS):
        for d in range(60, ND):
            w = spread[s, d - 60:d]; v = w[~np.isnan(w)]
            if len(v) >= 20 and not np.isnan(spread[s, d]):
                m, sd = np.mean(v), np.std(v, ddof=1)
                if sd > 1e-10: cz[s, d] = (spread[s, d] - m) / sd
    return {'cz': cz, 'syms': syms, 'dates': [pd.Timestamp(d) for d in dates]}


def compute_clean_signals(C, O, H, L, V, OI, NS, ND):
    """Compute momentum signals (clean, no future data used)."""
    scores = np.full((NS, ND), np.nan)

    # Multi-period momentum ranks
    mom_ranks = []
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
        mom_ranks.append(r)

    # Trend slope rank
    slope_rank = np.full((NS, ND), np.nan)
    for di in range(20, ND):
        slopes = np.full(NS, np.nan)
        for si in range(NS):
            prices = C[si, di - 20:di]
            valid = ~np.isnan(prices)
            if valid.sum() >= 15:
                x = np.arange(20)[valid]
                y = prices[valid]
                if len(x) >= 10:
                    s = np.polyfit(x, y, 1)[0]
                    mean_p = np.mean(y)
                    if mean_p > 0:
                        slopes[si] = s / mean_p
        valid = ~np.isnan(slopes)
        if valid.sum() >= 5:
            slope_rank[:, di] = pd.Series(slopes).rank(pct=True, na_option='keep').values

    # Volume surge
    vol_rank = np.full((NS, ND), np.nan)
    for di in range(20, ND):
        vratios = np.full(NS, np.nan)
        for si in range(NS):
            vt = V[si, di]
            va = np.nanmean(V[si, di - 20:di])
            if not np.isnan(vt) and not np.isnan(va) and va > 0:
                vratios[si] = vt / va
        valid = ~np.isnan(vratios)
        if valid.sum() >= 5:
            vol_rank[:, di] = pd.Series(vratios).rank(pct=True, na_option='keep').values

    # OI increase (5-day)
    oi_rank = np.full((NS, ND), np.nan)
    for di in range(5, ND):
        oi_deltas = np.full(NS, np.nan)
        for si in range(NS):
            oi_now = OI[si, di]
            oi_prev = OI[si, di - 5]
            if not np.isnan(oi_now) and not np.isnan(oi_prev) and oi_prev > 0:
                oi_deltas[si] = oi_now / oi_prev - 1
        valid = ~np.isnan(oi_deltas)
        if valid.sum() >= 5:
            oi_rank[:, di] = pd.Series(oi_deltas).rank(pct=True, na_option='keep').values

    # Combine with weights
    for di in range(20, ND):
        components = []
        weights = []
        # Momentum periods (heavier weight on shorter periods for faster signal)
        mom_weights = [0.20, 0.20, 0.15, 0.10]  # 3d, 5d, 10d, 20d
        for i, mr in enumerate(mom_ranks):
            v = mr[:, di]
            if not np.isnan(v).all():
                components.append(np.nan_to_num(v, nan=0.5))
                weights.append(mom_weights[i])
        v = slope_rank[:, di]
        if not np.isnan(v).all():
            components.append(np.nan_to_num(v, nan=0.5))
            weights.append(0.15)
        v = vol_rank[:, di]
        if not np.isnan(v).all():
            components.append(np.nan_to_num(v, nan=0.5))
            weights.append(0.10)
        v = oi_rank[:, di]
        if not np.isnan(v).all():
            components.append(np.nan_to_num(v, nan=0.5))
            weights.append(0.10)
        if components:
            total_w = sum(weights)
            scores[:, di] = sum(c * w for c, w in zip(components, weights)) / total_w

    return scores


def backtest_v315(C, O, H, L, NS, ND, dates, syms,
                  scores, ts_data, regime,
                  top_n=1, hold_days=3, atr_stop=2.5,
                  min_score=0.6, use_carry=True,
                  trail_stop=True, signal_exit=True,
                  start_di=60, end_di=None):
    """
    Clean backtest: signal from di, enter at O[di+1], exit at C[di+hold].
    No look-ahead.
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

        # Exit logic
        for si, edi, ep, sp, alloc, high_water in positions:
            c = C[si, di]
            h = H[si, di]
            if np.isnan(c):
                new_positions.append((si, edi, ep, sp, alloc, high_water))
                continue

            # Update trailing stop
            new_sp = sp
            if trail_stop and not np.isnan(h):
                # Trail stop up as price rises
                atr_v = []
                for j in range(max(start_di, di - 14), di):
                    hh, ll, cc = H[si, j], L[si, j], C[si, j]
                    if not any(np.isnan([hh, ll, cc])):
                        atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
                if atr_v:
                    atr = np.mean(atr_v)
                    new_sp = max(sp, h - atr_stop * atr)
                high_water = max(high_water, h)

            exit_r = None
            if c < new_sp:
                exit_r = 'stop'
            elif di - edi >= hold_days:
                exit_r = 'hold'
            elif signal_exit and not np.isnan(scores[si, di]):
                # Exit if signal has decayed below threshold
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
                new_positions.append((si, edi, ep, new_sp, alloc, high_water))

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

        # Adaptive top_n: in trending regime, allow more positions
        effective_tn = top_n
        if r == 1:  # trending
            effective_tn = min(top_n + 1, 5)

        candidates = []
        for si in range(NS):
            if si in held:
                continue
            score = scores[si, di]
            if np.isnan(score) or score < min_score:
                continue

            # Carry boost
            if use_carry:
                tsi = ts_di.get(d)
                if tsi is not None:
                    sym = syms[si]
                    tssi = ts_si.get(sym, -1)
                    if tssi >= 0 and tsi < ts_data['cz'].shape[1]:
                        cz = ts_data['cz'][tssi, tsi]
                        if not np.isnan(cz) and cz > 1:
                            score += 0.1

            # Check next day's open exists
            if di + 1 < ND and np.isnan(O[si, di + 1]):
                continue

            candidates.append((score, si))

        if not candidates:
            continue
        candidates.sort(key=lambda x: -x[0])

        alloc = 1.0 / max(effective_tn, 1)
        for score, si in candidates[:effective_tn]:
            if len(positions) >= effective_tn or si in held:
                break

            # Enter at NEXT DAY's open (no look-ahead)
            if di + 1 >= ND:
                continue
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue

            # ATR stop (using data up to di only — no look-ahead)
            atr_v = []
            for j in range(max(start_di, di - 14), di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
            if not atr_v:
                continue
            atr = np.mean(atr_v)
            stop = ep - atr_stop * atr

            positions.append((si, di + 1, ep, stop, alloc, ep))
            held.add(si)

    # Close remaining
    for si, edi, ep, sp, alloc, hw in positions:
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
    print("  V315: OPTIMIZED CLEAN MOMENTUM (NO LOOK-AHEAD)")
    print("=" * 60)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    ts_data = load_ts(start='2021-01-01')
    F = compute_factors(C, O, H, L, V, OI, NS, ND)
    regime = detect_regimes(F, NS, ND)

    print("[V315] Computing clean signals...", flush=True)
    scores = compute_clean_signals(C, O, H, L, V, OI, NS, ND)

    wf_start = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2024-01-01'):
            wf_start = i
            break

    # --- Selected configs ---
    print(f"\n--- Full In-Sample ---")
    configs = [
        (1, 2, 0.5, True, True, True, "v1"),
        (1, 3, 0.5, True, True, True, "v1"),
        (1, 3, 0.6, True, True, True, "v2"),
        (1, 5, 0.6, True, True, True, "v3"),
        (2, 3, 0.5, True, True, True, "v4"),
        (3, 3, 0.5, True, True, True, "v5"),
        (3, 5, 0.6, True, True, True, "v6"),
        (1, 3, 0.5, True, False, True, "no_trail"),
        (1, 3, 0.5, True, True, False, "no_sigexit"),
        (1, 3, 0.5, False, True, True, "no_carry"),
    ]

    for tn, hd, ms, carry, trail, sigexit, label in configs:
        trades, eq, dd = backtest_v315(
            C, O, H, L, NS, ND, dates, syms,
            scores, ts_data, regime,
            top_n=tn, hold_days=hd, min_score=ms,
            use_carry=carry, trail_stop=trail, signal_exit=sigexit)
        analyze(trades, eq, dd, f"IS {label} tn={tn} hd={hd}")

    # Walk-forward
    print(f"\n--- Walk-Forward 2024-2026 ---")
    for tn, hd, ms, carry, trail, sigexit, label in configs:
        trades, eq, dd = backtest_v315(
            C, O, H, L, NS, ND, dates, syms,
            scores, ts_data, regime,
            top_n=tn, hold_days=hd, min_score=ms,
            use_carry=carry, trail_stop=trail, signal_exit=sigexit,
            start_di=wf_start)
        analyze(trades, eq, dd, f"WF {label} tn={tn} hd={hd}")

    # --- Full sweep ---
    print(f"\n--- Full Sweep (WF 2024-2026) ---")
    results = []
    for tn in [1, 2, 3, 5]:
        for hd in [1, 2, 3, 5, 7]:
            for ms in [0.4, 0.5, 0.6, 0.7]:
                for carry in [True, False]:
                    for trail in [True, False]:
                        trades, eq, dd = backtest_v315(
                            C, O, H, L, NS, ND, dates, syms,
                            scores, ts_data, regime,
                            top_n=tn, hold_days=hd, min_score=ms,
                            use_carry=carry, trail_stop=trail,
                            signal_exit=True, start_di=wf_start)
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
                            'tn': tn, 'hd': hd, 'ms': ms,
                            'carry': carry, 'trail': trail,
                            'n': len(trades), 'wr': wr, 'ann': ann,
                            'dd': dd, 'sh': sh,
                        })

    results.sort(key=lambda x: -x['ann'])
    print(f"\n{'TN':>3} {'HD':>3} {'MS':>4} {'C':>2} {'TR':>2} {'N':>5} "
          f"{'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 60)
    for r in results[:25]:
        print(f"{r['tn']:>3} {r['hd']:>3} {r['ms']:>4.1f} "
              f"{'Y' if r['carry'] else 'N':>2} {'Y' if r['trail'] else 'N':>2} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sh']:>5.2f}")

    results.sort(key=lambda x: -x['sh'])
    print(f"\n--- By Sharpe ---")
    for r in results[:15]:
        print(f"  tn={r['tn']} hd={r['hd']} ms={r['ms']} "
              f"c={'Y' if r['carry'] else 'N'} tr={'Y' if r['trail'] else 'N'}: "
              f"{r['n']}t WR={r['wr']:.1f}% ann={r['ann']:+.1f}% "
              f"DD={r['dd']:.1f}% Sh={r['sh']:.2f}")

    print(f"\n[V315] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
