"""
Alpha V44 — Score Filter + Hybrid Stop
=======================================
V43发现: 58%交易1-3天被止损(WR=21%), 但ATR=0.8仍是最优。
V44探索:
  1. 最低分数过滤: composite > 阈值才买，否则空仓
  2. 混合止损: 前2天用宽止损(1.5x), 之后收紧(0.8x)
  3. 纯分数阈值回测: 只看composite本身作为因子

只计算V41需要的5个因子，大幅加速。
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7 import COMMISSION, STAMP_DUTY, CASH0


def compute_v41_factors_only(NS, ND, C, O, H, L, V):
    """Only compute the 5 factors needed by V41.
    Need V7 base + V10 interactions (for BWP_BNW) + V14 (for HAR-RV).
    """
    from alpha_v7 import compute_all_factors
    from alpha_v7b import compute_interaction_factors
    from alpha_v7d import compute_extra_factors
    from alpha_v7e import compute_v7e_factors
    from alpha_v7f import compute_advanced_interactions
    from alpha_v10 import compute_v10_factors, compute_v10_interactions
    from alpha_v14 import compute_v14_factors, compute_v14_interactions

    t0 = time.time()
    base = compute_all_factors(NS, ND, C, O, H, L, V)
    inter = compute_interaction_factors(base, NS, ND, C, O, H, L, V)
    extra = compute_extra_factors(NS, ND, C, O, H, L, V)
    v7e = compute_v7e_factors(NS, ND, C, O, H, L, V)
    adv = compute_advanced_interactions({**base, **inter, **extra, **v7e}, NS, ND)

    # V10 for BWP_BNW (needs BB_WIDTH_PCT_INV × BODY_NW)
    v7_all = {**base, **inter, **extra, **v7e, **adv}
    v10f = compute_v10_factors(NS, ND, C, O, H, L, V)
    v10_all = {**v7_all, **v10f}
    v10_inter = compute_v10_interactions(v10_all, NS, ND)
    v10_all.update(v10_inter)

    # V14 for HAR-RV
    v14f = compute_v14_factors(NS, ND, C, O, H, L, V)
    v14_all = {**v10_all, **v14f}
    v14_inter = compute_v14_interactions(v14_all, NS, ND)
    v14_all.update(v14_inter)

    # Only return the 5 needed factors
    needed = ['R_BWP_BNW', 'R_TENSION', 'R_R_SQUARED', 'R_SMA_DEV', 'R_HAR_RV_RATIO_INV']
    factors = {}
    for f in needed:
        if f in v14_all:
            factors[f] = v14_all[f]
    print(f"  Factors computed: {len(factors)} needed ({time.time()-t0:.0f}s)", flush=True)
    return factors


def backtest_v44(factor_weights, factors, NS, ND, dates, C, O, H, L, V,
                top_n=1, rebalance_days=5, atr_stop_mult=0.8,
                min_score=None, hybrid_stop=False, wide_mult=1.5, wide_days=2):
    """Backtest with score filter and/or hybrid stop.

    min_score: only buy if composite score > this threshold (0-100)
    hybrid_stop: use wider stop for first `wide_days` days
    """
    factor_names = list(factor_weights.keys())
    weights = np.array([factor_weights[f] for f in factor_names])

    cash = float(CASH0)
    holdings = []
    trades = []
    last_rebalance = -999
    year_stats = {}

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # Stop loss check
        for pos in list(holdings):
            si = pos['si']
            stopped_out = False
            days_held = (dates[di] - pos['ed']).days

            # Choose stop multiplier based on days held
            if hybrid_stop and days_held <= wide_days:
                current_mult = wide_mult
            else:
                current_mult = atr_stop_mult

            if current_mult > 0:
                atr = 0
                atr_count = 0
                for dd in range(max(di - 14, 1), di):
                    if not np.isnan(H[si, dd]) and not np.isnan(L[si, dd]):
                        tr = H[si, dd] - L[si, dd]
                        if not np.isnan(C[si, dd - 1]):
                            tr = max(tr, abs(H[si, dd] - C[si, dd - 1]),
                                     abs(L[si, dd] - C[si, dd - 1]))
                        atr += tr
                        atr_count += 1
                if atr_count > 0:
                    atr /= atr_count

                if atr > 0:
                    stop = pos['hw'] - current_mult * atr
                    today_low = L[si, di]
                    today_open = O[si, di]

                    if not np.isnan(today_low) and today_low <= stop:
                        if not np.isnan(today_open) and today_open < stop:
                            sp = today_open
                        else:
                            sp = stop
                        pnl = (sp - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * sp * (1 - COMMISSION - STAMP_DUTY)
                        trades.append({'pnl': pnl, 'days': days_held,
                                       'di': di, 'reason': 'stop', 'year': year})
                        holdings.remove(pos)
                        stopped_out = True

            if not stopped_out:
                today_high = H[si, di]
                if not np.isnan(today_high) and today_high > 0:
                    pos['hw'] = max(pos['hw'], today_high)

            if pos in holdings:
                if days_held >= 60:
                    sp = O[si, di] if not np.isnan(O[si, di]) else C[si, di]
                    if not np.isnan(sp) and sp > 0:
                        pnl = (sp - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * sp * (1 - COMMISSION - STAMP_DUTY)
                        trades.append({'pnl': pnl, 'days': days_held,
                                       'di': di, 'reason': 'time_stop', 'year': year})
                        holdings.remove(pos)

        # Rebalance
        if di - last_rebalance >= rebalance_days:
            # Use di-1 to avoid look-ahead bias
            composite = np.zeros(NS)
            count = np.zeros(NS)
            for fname, w in zip(factor_names, weights):
                if fname not in factors:
                    continue
                arr = factors[fname]
                vals = arr[:, di - 1]
                valid = ~np.isnan(vals)
                if valid.sum() < 50:
                    continue
                composite[valid] += w * vals[valid]
                count[valid] += abs(w)

            mask = count > 0
            if mask.sum() < top_n * 2:
                last_rebalance = di
                continue
            composite[mask] /= count[mask]
            composite[~mask] = -9999

            # Score filter: check if best stock meets minimum score
            sorted_idx = np.argsort(-composite)
            best_score = composite[sorted_idx[0]]

            if min_score is not None and best_score < min_score:
                # Score too low, sell all and wait
                for pos in list(holdings):
                    p = O[pos['si'], di]
                    if np.isnan(p) or p <= 0:
                        p = C[pos['si'], di]
                    if not np.isnan(p) and p > 0:
                        pnl = (p - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
                        trades.append({'pnl': pnl, 'days': (dates[di] - pos['ed']).days,
                                       'di': di, 'reason': 'score_skip', 'year': year})
                holdings = []
                last_rebalance = di
                continue

            top_indices = set(sorted_idx[:top_n])
            current_indices = set(h['si'] for h in holdings)

            to_sell = current_indices - top_indices
            for pos in list(holdings):
                if pos['si'] in to_sell:
                    p = O[pos['si'], di]
                    if np.isnan(p) or p <= 0:
                        p = C[pos['si'], di]
                    if not np.isnan(p) and p > 0:
                        pnl = (p - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
                        trades.append({'pnl': pnl, 'days': (dates[di] - pos['ed']).days,
                                       'di': di, 'reason': 'rebalance', 'year': year})
                        holdings.remove(pos)

            current_indices = set(h['si'] for h in holdings)
            to_buy = top_indices - current_indices
            n_to_buy = len(to_buy)
            if n_to_buy > 0 and cash > 10000:
                alloc = cash / n_to_buy
                for si in to_buy:
                    p = O[si, di]
                    if np.isnan(p) or p <= 0:
                        p = C[si, di]
                    if not np.isnan(p) and p > 0:
                        shares = int(alloc / (1 + COMMISSION) / p)
                        if shares > 0:
                            cost = shares * p * (1 + COMMISSION)
                            if cost <= cash:
                                cash -= cost
                                holdings.append({
                                    'si': si, 'shares': shares, 'entry': p,
                                    'ed': dates[di], 'hw': p, 'score': composite[si]
                                })
            last_rebalance = di

    # Close remaining
    for pos in holdings:
        p = C[pos['si'], ND - 1]
        if not np.isnan(p) and p > 0:
            pnl = (p - pos['entry']) / pos['entry'] * 100
            cash += pos['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
            trades.append({'pnl': pnl, 'days': 999, 'di': ND - 1, 'reason': 'end',
                           'year': dates[ND - 1].year})

    if cash <= 0 or not trades:
        return None

    days_total = (dates[ND - 1] - dates[MIN_TRAIN]).days
    yr = max(days_total / 365.25, 0.01)
    ann = ((cash / CASH0) ** (1 / yr) - 1) * 100
    nw = sum(1 for t in trades if t['pnl'] > 0)
    wr = nw / max(len(trades), 1) * 100
    avg_w = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if nw > 0 else 0
    avg_l = np.mean([abs(t['pnl']) for t in trades if t['pnl'] <= 0]) if nw < len(trades) else 0

    for t in trades:
        y = t.get('year', 'unknown')
        if y not in year_stats:
            year_stats[y] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
        year_stats[y]['trades'] += 1
        if t['pnl'] > 0:
            year_stats[y]['wins'] += 1
        year_stats[y]['total_pnl'] += t['pnl']

    equity = float(CASH0)
    peak = float(CASH0)
    max_dd = 0
    for t in sorted(trades, key=lambda x: x['di']):
        equity *= (1 + t['pnl'] / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    return {
        'ann': round(ann, 1), 'n': len(trades), 'wr': round(wr, 1),
        'avg_w': round(avg_w, 1), 'avg_l': round(avg_l, 1),
        'edge': round((nw / max(len(trades), 1)) * avg_w - (1 - nw / max(len(trades), 1)) * avg_l, 2),
        'max_dd': round(max_dd, 1), 'tpy': round(len(trades) / yr, 1),
        'final': round(cash, 0), 'year_stats': year_stats,
    }


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V44 — Score Filter + Hybrid Stop", flush=True)
    print("  Target: beat V41 V15B_EQUAL_A0.8 = +342.0%", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()
    factors = compute_v41_factors_only(NS, ND, C, O, H, L, V)

    v41_weights = {'R_BWP_BNW': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2,
                   'R_SMA_DEV': 0.2, 'R_HAR_RV_RATIO_INV': 0.2}

    results = []

    # =====================================================================
    # TEST 1: Baseline (V41 config)
    # =====================================================================
    print("\n  Test 1: V41 baseline...", flush=True)
    r = backtest_v44(v41_weights, factors, NS, ND, dates, C, O, H, L, V,
                    top_n=1, rebalance_days=5, atr_stop_mult=0.8)
    if r:
        r['test'] = 'BASELINE_A0.8'
        results.append(r)
        print(f"  Baseline: {r['ann']:+.1f}% DD={r['max_dd']:.1f}%", flush=True)

    # =====================================================================
    # TEST 2: Score filter sweep
    # =====================================================================
    print("\n  Test 2: Score filter...", flush=True)
    for min_s in [85, 88, 90, 92, 94, 95]:
        for atr in [0.8, 1.0]:
            r = backtest_v44(v41_weights, factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr,
                            min_score=min_s)
            if r:
                r['test'] = f'SF{min_s}_A{atr}'
                results.append(r)
    print(f"  Score filter done: {len(results)}", flush=True)

    # =====================================================================
    # TEST 3: Hybrid stop (wide first 1-3 days, then tight)
    # =====================================================================
    print("\n  Test 3: Hybrid stop...", flush=True)
    for wide_d in [1, 2, 3]:
        for wide_m in [1.2, 1.5, 2.0]:
            for tight_m in [0.6, 0.8, 1.0]:
                r = backtest_v44(v41_weights, factors, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=5, atr_stop_mult=tight_m,
                                hybrid_stop=True, wide_mult=wide_m, wide_days=wide_d)
                if r:
                    r['test'] = f'HY_W{wide_d}M{wide_m}_T{tight_m}'
                    results.append(r)
    print(f"  Hybrid stop done: {len(results)}", flush=True)

    # =====================================================================
    # TEST 4: Score filter + hybrid stop combos
    # =====================================================================
    print("\n  Test 4: Best combos...", flush=True)
    for min_s in [90, 92, 94]:
        for wide_d in [2]:
            for wide_m in [1.5]:
                for tight_m in [0.6, 0.8]:
                    r = backtest_v44(v41_weights, factors, NS, ND, dates, C, O, H, L, V,
                                    top_n=1, rebalance_days=5, atr_stop_mult=tight_m,
                                    min_score=min_s, hybrid_stop=True,
                                    wide_mult=wide_m, wide_days=wide_d)
                    if r:
                        r['test'] = f'CMB_SF{min_s}_W{wide_d}M{wide_m}_T{tight_m}'
                        results.append(r)
    print(f"  Combos done: {len(results)}", flush=True)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  ALL RESULTS (V44 SCORE FILTER + HYBRID STOP)", flush=True)
    print(f"  {'Test':<35s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*80}", flush=True)
    for r in results[:40]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<35s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    # Best by category
    for cat_name, prefix in [('Score Filter', 'SF'), ('Hybrid Stop', 'HY_'), ('Combo', 'CMB_')]:
        cat = [r for r in results if r['test'].startswith(prefix)]
        if cat:
            best = max(cat, key=lambda x: x['ann'])
            pos = " ALL+" if all_positive(best) else ""
            print(f"\n  Best {cat_name}: {best['test']} → {best['ann']:+.1f}%DD={best['max_dd']:.1f}%{pos}", flush=True)

    # Top 5 year-by-year
    for i, r in enumerate(results[:5]):
        print(f"\n  Year-by-year #{i+1}: {r['test']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    if results:
        best = results[0]
        print(f"\n  === V44 BEST vs V41 RECORD ===", flush=True)
        print(f"  V44: {best['test']} = {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)
        print(f"  V41: V15B_EQUAL_A0.8 = +342.0% DD=53.7%", flush=True)
        delta = best['ann'] - 342.0
        print(f"  Delta: {delta:+.1f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
