"""
V311: Maximum No-Leverage Returns
===================================
Push the signal quality to the absolute limit without leverage.
Key ideas:
1. Ultra-concentrated: only the #1 ranked commodity each period
2. Signal strength filter: only trade when score > 90th percentile
3. Next-day return capture: enter at open, exit at close (1-day hold)
4. Ensemble: momentum + carry + ADX + volume + RSI
5. Kelly criterion for optimal position sizing within 1x
6. Aggressive hold periods (1-2 days) for maximum compounding
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
            sym, ds = d.get('symbol',''), d.get('date','')
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
            w = spread[s, d-60:d]; v = w[~np.isnan(w)]
            if len(v) >= 20 and not np.isnan(spread[s, d]):
                m, sd = np.mean(v), np.std(v, ddof=1)
                if sd > 1e-10: cz[s, d] = (spread[s, d] - m) / sd
    return {'cz': cz, 'syms': syms, 'dates': [pd.Timestamp(d) for d in dates]}


def compute_all_signals(C, O, H, L, V, NS, ND):
    """Compute all signals and return a combined score."""
    scores = np.full((NS, ND), np.nan)

    # 1. Multi-period momentum rank (average of 5, 10, 20 day)
    mom_ranks = []
    for period in [5, 10, 20]:
        r = np.full((NS, ND), np.nan)
        for di in range(period, ND):
            rets = np.full(NS, np.nan)
            for si in range(NS):
                c0, c1 = C[si, di-period], C[si, di]
                if not np.isnan(c0) and not np.isnan(c1) and c0 > 0:
                    rets[si] = (c1 - c0) / c0
            valid = ~np.isnan(rets)
            if valid.sum() >= 5:
                r[:, di] = pd.Series(rets).rank(pct=True, na_option='keep').values
        mom_ranks.append(r)

    # 2. Trend slope (linear regression over 20 days)
    slope_rank = np.full((NS, ND), np.nan)
    for di in range(20, ND):
        slopes = np.full(NS, np.nan)
        for si in range(NS):
            prices = C[si, di-20:di]
            valid = ~np.isnan(prices)
            if valid.sum() >= 15:
                x = np.arange(20)[valid]
                y = prices[valid]
                if len(x) >= 10:
                    s = np.polyfit(x, y, 1)[0]
                    mean_p = np.mean(y)
                    if mean_p > 0:
                        slopes[si] = s / mean_p  # Normalized slope
        valid = ~np.isnan(slopes)
        if valid.sum() >= 5:
            slope_rank[:, di] = pd.Series(slopes).rank(pct=True, na_option='keep').values

    # 3. Volume surge
    vol_rank = np.full((NS, ND), np.nan)
    for di in range(20, ND):
        vratios = np.full(NS, np.nan)
        for si in range(NS):
            vt = V[si, di]
            va = np.nanmean(V[si, di-20:di])
            if not np.isnan(vt) and not np.isnan(va) and va > 0:
                vratios[si] = vt / va
        valid = ~np.isnan(vratios)
        if valid.sum() >= 5:
            vol_rank[:, di] = pd.Series(vratios).rank(pct=True, na_option='keep').values

    # 4. Short-term reversal (1-day return)
    rev_rank = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        rets = np.full(NS, np.nan)
        for si in range(NS):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                rets[si] = (C[si, di] - C[si, di-1]) / C[si, di-1]
        valid = ~np.isnan(rets)
        if valid.sum() >= 5:
            # For momentum, we want positive recent return (NOT reversal)
            rev_rank[:, di] = pd.Series(rets).rank(pct=True, na_option='keep').values

    # Combine: momentum-focused ensemble
    for di in range(20, ND):
        components = []
        weights = []
        for mr in mom_ranks:
            v = mr[:, di]
            if not np.isnan(v).all():
                components.append(np.nan_to_num(v, nan=0.5))
                weights.append(0.25)
        v = slope_rank[:, di]
        if not np.isnan(v).all():
            components.append(np.nan_to_num(v, nan=0.5))
            weights.append(0.2)
        v = vol_rank[:, di]
        if not np.isnan(v).all():
            components.append(np.nan_to_num(v, nan=0.5))
            weights.append(0.05)
        v = rev_rank[:, di]
        if not np.isnan(v).all():
            components.append(np.nan_to_num(v, nan=0.5))
            weights.append(0.05)
        if components:
            total_w = sum(weights)
            scores[:, di] = sum(c * w for c, w in zip(components, weights)) / total_w

    return scores


def backtest_v311(C, O, H, L, NS, ND, dates, syms,
                  scores, ts_data, regime,
                  top_n=3, hold_days=2, atr_stop=2.5,
                  min_score=0.6, use_carry_boost=True,
                  start_di=60, end_di=None):
    if end_di is None: end_di = ND

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
                new_positions.append((si, edi, ep, sp, alloc)); continue
            exit_r = None
            if c < sp: exit_r = 'stop'
            elif di - edi >= hold_days: exit_r = 'hold'
            if exit_r:
                pnl = (c - ep) / ep - COMM
                profit = equity * alloc * pnl
                daily_pnl += profit
                trades.append({
                    'pnl_abs': profit, 'pnl_pct': pnl * 100,
                    'days': di - edi, 'di': di, 'year': d.year,
                    'sym': syms[si], 'reason': exit_r,
                })
            else:
                new_positions.append((si, edi, ep, sp, alloc))
        positions = new_positions
        equity += daily_pnl
        if equity > peak: peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100
            if dd > max_dd: max_dd = dd
        if equity <= 0: break

        # Entry
        held = {p[0] for p in positions}
        if len(positions) >= top_n: continue

        # Regime
        r = regime[di] if regime is not None and di < len(regime) else 0
        if r in (-1, 2): continue  # Skip choppy/volatile entirely

        candidates = []
        for si in range(NS):
            if si in held: continue
            if np.isnan(C[si, di]) or np.isnan(O[si, di]): continue
            score = scores[si, di]
            if np.isnan(score) or score < min_score: continue

            # Carry boost
            if use_carry_boost:
                tsi = ts_di.get(d)
                if tsi is not None:
                    sym = syms[si]
                    tssi = ts_si.get(sym, -1)
                    if tssi >= 0 and tsi < ts_data['cz'].shape[1]:
                        cz = ts_data['cz'][tssi, tsi]
                        if not np.isnan(cz) and cz > 1:
                            score += 0.1  # Small boost for strong backwardation

            candidates.append((score, si))

        if not candidates: continue
        candidates.sort(key=lambda x: -x[0])

        alloc = 1.0 / max(top_n, 1)  # No leverage, equal weight
        for score, si in candidates[:top_n]:
            if len(positions) >= top_n or si in held: break
            op = O[si, di]
            if np.isnan(op) or op <= 0: continue
            atr_v = []
            for j in range(max(start_di, di-14), di):
                hh, ll, cc = H[si,j], L[si,j], C[si,j]
                if not any(np.isnan([hh, ll, cc])):
                    atr_v.append(max(hh-ll, abs(hh-cc), abs(ll-cc)))
            if not atr_v: continue
            atr = np.mean(atr_v)
            positions.append((si, di, op, op - atr_stop*atr, alloc))
            held.add(si)

    for si, edi, ep, sp, alloc in positions:
        c = C[si, ND-1]
        if not np.isnan(c):
            pnl = (c-ep)/ep - COMM
            equity += equity * alloc * pnl

    return trades, equity, max_dd


def analyze(trades, equity, max_dd, label=""):
    if not trades:
        print(f"  {label}: no trades"); return None
    nw = sum(1 for t in trades if t['pnl_pct'] > 0)
    wr = nw / len(trades) * 100
    ann = ((equity/CASH0)**(1/max(1.0,(trades[-1]['di']-trades[0]['di'])/252))-1)*100
    ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
    rets = np.array(ap) / CASH0
    sh = np.mean(rets)/np.std(rets)*np.sqrt(252) if np.std(rets) > 0 else 0
    print(f"  {label}: {len(trades)}t WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% Sh={sh:.2f} eq={equity:,.0f}")
    yr = {}
    for t in trades:
        y = t['year']
        if y not in yr: yr[y] = {'n':0,'w':0,'pnl':[]}
        yr[y]['n'] += 1
        if t['pnl_pct'] > 0: yr[y]['w'] += 1
        yr[y]['pnl'].append(t['pnl_pct'])
    for y in sorted(yr.keys()):
        ys = yr[y]
        print(f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% cum={np.prod([1+p/100 for p in ys['pnl']])-1:+.1%}")
    return {'n': len(trades), 'wr': wr, 'dd': max_dd, 'ann': ann, 'sh': sh}


def main():
    t0 = time.time()
    print("=" * 60)
    print("  V311: MAXIMIZE NO-LEVERAGE RETURNS")
    print("=" * 60)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    ts_data = load_ts(start='2021-01-01')

    F = compute_factors(C, O, H, L, V, OI, NS, ND)
    regime = detect_regimes(F, NS, ND)

    print("[V311] Computing ensemble signals...", flush=True)
    scores = compute_all_signals(C, O, H, L, V, NS, ND)

    # Sweep
    print("\n--- Parameter Sweep (1x leverage) ---")
    results = []
    for tn in [1, 2, 3, 5]:
        for hd in [1, 2, 3, 5]:
            for ms in [0.5, 0.6, 0.7, 0.8]:
                for carry in [True, False]:
                    trades, eq, dd = backtest_v311(
                        C, O, H, L, NS, ND, dates, syms,
                        scores, ts_data, regime,
                        top_n=tn, hold_days=hd,
                        min_score=ms, use_carry_boost=carry)
                    if len(trades) < 5: continue
                    nw = sum(1 for t in trades if t['pnl_pct'] > 0)
                    wr = nw / len(trades) * 100
                    ann = ((eq/CASH0)**(1/max(1.0,(trades[-1]['di']-trades[0]['di'])/252))-1)*100
                    ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
                    rets = np.array(ap)/CASH0
                    sh = np.mean(rets)/np.std(rets)*np.sqrt(252) if np.std(rets) > 0 else 0
                    results.append({
                        'tn':tn, 'hd':hd, 'ms':ms, 'carry':carry,
                        'n':len(trades), 'wr':wr, 'ann':ann, 'dd':dd, 'sh':sh,
                    })

    results.sort(key=lambda x: -x['sh'])
    print(f"\n{'TN':>3} {'HD':>3} {'MS':>4} {'C':>2} {'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 50)
    for r in results[:20]:
        print(f"{r['tn']:>3} {r['hd']:>3} {r['ms']:>4.1f} {'Y' if r['carry'] else 'N':>2} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} {r['dd']:>6.1f} {r['sh']:>5.2f}")

    # Also by annual return
    print("\n--- By Annual Return ---")
    results.sort(key=lambda x: -x['ann'])
    for r in results[:10]:
        print(f"  tn={r['tn']} hd={r['hd']} ms={r['ms']} c={'Y' if r['carry'] else 'N'}: "
              f"{r['n']}t WR={r['wr']:.1f}% ann={r['ann']:+.1f}% DD={r['dd']:.1f}% Sh={r['sh']:.2f}")

    print(f"\n[V311] Done. {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
