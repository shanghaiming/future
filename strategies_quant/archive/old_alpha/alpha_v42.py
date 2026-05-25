"""
Alpha V42 — Rank Momentum & Stability Factors
==============================================
V41 established +342.0% with equal-weight V15-B factors.
V42 explores a NEW signal dimension: rank dynamics over time.

Instead of static factor ranks, use:
1. RANK_MOMENTUM: How fast a stock's composite rank is improving (rank Δ over 3/5/10 days)
2. RANK_STABILITY: How consistently a stock stays in top ranks (std of rank over rolling window)
3. RANK_ACCELERATION: Second derivative of rank change (is improvement accelerating?)
4. COMPOSITE_RANK_SHIFT: Today's rank minus N-day average rank (mean reversion in rank space)

Hypothesis: Stocks with improving and stable high ranks outperform those
with high but volatile ranks. This captures "smart money accumulating" patterns.

Uses V41's optimal config: equal weights, ATR=0.8, rebalance=5d, top_n=1.
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7c import backtest_v7c


def compute_rank_dynamics_factors(all_factors, NS, ND, weights):
    """Compute rank-based dynamic factors.

    First compute composite score, then analyze rank dynamics.
    """
    t0 = time.time()
    results = {}

    # Step 1: Compute composite score
    print("  Computing composite score...", flush=True)
    score = np.zeros((NS, ND))
    weight_sum = 0
    for fname, w in weights.items():
        if fname in all_factors:
            arr = all_factors[fname]
            valid = ~np.isnan(arr)
            score[valid] += arr[valid] * w
            weight_sum += w
    if weight_sum > 0:
        score /= weight_sum

    # Step 2: Rank stocks by composite score each day
    print("  Computing daily ranks...", flush=True)
    daily_rank = np.full((NS, ND), np.nan)
    for di in range(ND):
        vals = score[:, di]
        valid = ~np.isnan(vals)
        n = valid.sum()
        if n < 50:
            continue
        order = np.argsort(vals[valid])
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1)
        daily_rank[valid, di] = ranks

    # Step 3: Rank momentum (rank improvement over N days)
    print("  Computing rank momentum...", flush=True)
    for window, wname in [(3, 'RM3'), (5, 'RM5'), (10, 'RM10')]:
        rank_mom = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(window + 1, ND):
                cur = daily_rank[si, di]
                prev = daily_rank[si, di - window]
                if np.isnan(cur) or np.isnan(prev):
                    continue
                # Negative = rank improved (lower rank number = better)
                rank_mom[si, di] = prev - cur  # positive = improved
        # Normalize
        r = np.full_like(rank_mom, np.nan)
        for di in range(ND):
            vals = rank_mom[:, di]
            valid = ~np.isnan(vals)
            n = valid.sum()
            if n < 50:
                continue
            order = np.argsort(vals[valid])
            ranks = np.empty(n)
            ranks[order] = np.arange(1, n + 1)
            r[valid, di] = ranks / n * 100
        results[f'R_RANK_MOM_{wname}'] = r

    # Step 4: Rank stability (std of rank over rolling window)
    print("  Computing rank stability...", flush=True)
    for window, wname in [(5, 'RS5'), (10, 'RS10'), (20, 'RS20')]:
        rank_stab = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(window, ND):
                vals = daily_rank[si, di - window + 1:di + 1]
                n_valid = (~np.isnan(vals)).sum()
                if n_valid < window * 0.7:
                    continue
                rank_stab[si, di] = -np.nanstd(vals)  # negative std = more stable = better
        # Normalize
        r = np.full_like(rank_stab, np.nan)
        for di in range(ND):
            vals = rank_stab[:, di]
            valid = ~np.isnan(vals)
            n = valid.sum()
            if n < 50:
                continue
            order = np.argsort(vals[valid])
            ranks = np.empty(n)
            ranks[order] = np.arange(1, n + 1)
            r[valid, di] = ranks / n * 100
        results[f'R_RANK_STAB_{wname}'] = r

    # Step 5: Rank acceleration (change in rank momentum)
    print("  Computing rank acceleration...", flush=True)
    # Use RM5 for acceleration
    rm5 = results.get('R_RANK_MOM_RM5')
    if rm5 is not None:
        rank_accel = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(6, ND):
                cur = rm5[si, di]
                prev = rm5[si, di - 3]
                if np.isnan(cur) or np.isnan(prev):
                    continue
                rank_accel[si, di] = cur - prev
        # Normalize
        r = np.full_like(rank_accel, np.nan)
        for di in range(ND):
            vals = rank_accel[:, di]
            valid = ~np.isnan(vals)
            n = valid.sum()
            if n < 50:
                continue
            order = np.argsort(vals[valid])
            ranks = np.empty(n)
            ranks[order] = np.arange(1, n + 1)
            r[valid, di] = ranks / n * 100
        results['R_RANK_ACCEL'] = r

    # Step 6: Composite rank shift (today vs 10-day average rank)
    print("  Computing rank shift...", flush=True)
    rank_shift = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(11, ND):
            cur = daily_rank[si, di]
            window_ranks = daily_rank[si, di - 10:di]
            avg_rank = np.nanmean(window_ranks)
            if np.isnan(cur) or np.isnan(avg_rank):
                continue
            rank_shift[si, di] = avg_rank - cur  # positive = rank improved vs average
    # Normalize
    r = np.full_like(rank_shift, np.nan)
    for di in range(ND):
        vals = rank_shift[:, di]
        valid = ~np.isnan(vals)
        n = valid.sum()
        if n < 50:
            continue
        order = np.argsort(vals[valid])
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1)
        r[valid, di] = ranks / n * 100
    results['R_RANK_SHIFT'] = r

    # Step 7: High-rank consistency (% of days in top 20% over rolling window)
    print("  Computing high-rank consistency...", flush=True)
    for window, wname in [(5, 'HC5'), (10, 'HC10')]:
        hrc = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(window, ND):
                window_ranks = daily_rank[si, di - window + 1:di + 1]
                n_valid = (~np.isnan(window_ranks)).sum()
                if n_valid < window * 0.7:
                    continue
                # Total stocks on those days
                total = np.nanmax(daily_rank[:, di]) if not np.isnan(daily_rank[:, di]).all() else 500
                threshold = total * 0.2  # top 20%
                pct_top = np.nanmean(window_ranks <= threshold) * 100
                hrc[si, di] = pct_top
        # Normalize
        r = np.full_like(hrc, np.nan)
        for di in range(ND):
            vals = hrc[:, di]
            valid = ~np.isnan(vals)
            n = valid.sum()
            if n < 50:
                continue
            order = np.argsort(vals[valid])
            ranks = np.empty(n)
            ranks[order] = np.arange(1, n + 1)
            r[valid, di] = ranks / n * 100
        results[f'R_RANK_HRC_{wname}'] = r

    print(f"  Rank dynamics done: {len(results)} factors ({time.time()-t0:.0f}s)", flush=True)
    return results


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V42 — Rank Momentum & Stability Factors", flush=True)
    print("  Target: beat V41 V15B_EQUAL_A0.8 = +342.0%", flush=True)
    print("=" * 70, flush=True)

    # Try loading from cache first
    cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'factor_cache', 'all_factors.pkl')
    if os.path.exists(cache_file):
        import pickle
        print("  Loading cached factors...", flush=True)
        t0 = time.time()
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        all_factors = data['factors']
        NS, ND = data['NS'], data['ND']
        dates = data['dates']
        C, O, H, L, V = data['C'], data['O'], data['H'], data['L'], data['V']
        syms, sym_set = data['syms'], data['sym_set']
        print(f"  Loaded {len(all_factors)} factors in {time.time()-t0:.1f}s", flush=True)
    else:
        print("  No cache found, loading data...", flush=True)
        from alpha_v7 import compute_all_factors
        from alpha_v7b import compute_interaction_factors
        from alpha_v7d import compute_extra_factors
        from alpha_v7e import compute_v7e_factors
        from alpha_v7f import compute_advanced_interactions
        from alpha_v8 import compute_v8_factors, compute_v8_interactions
        from alpha_v9 import compute_v9_factors, compute_v9_interactions
        from alpha_v10 import compute_v10_factors, compute_v10_interactions
        from alpha_v11 import compute_v11_factors, compute_v11_interactions
        from alpha_v14 import compute_v14_factors, compute_v14_interactions

        NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()
        base_factors = compute_all_factors(NS, ND, C, O, H, L, V)
        inter_factors = compute_interaction_factors(base_factors, NS, ND, C, O, H, L, V)
        extra_factors = compute_extra_factors(NS, ND, C, O, H, L, V)
        v7e_factors = compute_v7e_factors(NS, ND, C, O, H, L, V)
        adv_inter = compute_advanced_interactions(
            {**base_factors, **inter_factors, **extra_factors, **v7e_factors}, NS, ND)
        v8_factors = compute_v8_factors(NS, ND, C, O, H, L, V)
        v8_all = {**base_factors, **inter_factors, **extra_factors,
                  **v7e_factors, **adv_inter, **v8_factors}
        v8_inter = compute_v8_interactions(v8_all, NS, ND)
        v8_all.update(v8_inter)
        v9_factors = compute_v9_factors(NS, ND, C, O, H, L, V)
        v9_all = {**v8_all, **v9_factors}
        v9_inter = compute_v9_interactions(v9_all, NS, ND)
        v9_all.update(v9_inter)
        v10_factors = compute_v10_factors(NS, ND, C, O, H, L, V)
        v10_all = {**v9_all, **v10_factors}
        v10_inter = compute_v10_interactions(v10_all, NS, ND)
        v10_all.update(v10_inter)
        v11_factors = compute_v11_factors(NS, ND, C, O, H, L, V)
        v11_all = {**v10_all, **v11_factors}
        v11_inter = compute_v11_interactions(v11_all, NS, ND)
        v11_all.update(v11_inter)
        v14_factors = compute_v14_factors(NS, ND, C, O, H, L, V)
        all_factors = {**v11_all, **v14_factors}
        v14_inter = compute_v14_interactions(all_factors, NS, ND)
        all_factors.update(v14_inter)

    # V41 optimal weights (equal)
    v41_weights = {'R_BWP_BNW': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2,
                   'R_SMA_DEV': 0.2, 'R_HAR_RV_RATIO_INV': 0.2}

    # Compute rank dynamics
    print("\n  Computing rank dynamics factors...", flush=True)
    rank_factors = compute_rank_dynamics_factors(all_factors, NS, ND, v41_weights)
    all_factors.update(rank_factors)

    print(f"\n  Total factors: {len(all_factors)}", flush=True)

    results = []

    # =====================================================================
    # TEST 1: Each rank dynamic factor alone
    # =====================================================================
    print("\n  Test 1: Single rank dynamics factors...", flush=True)
    for fname in sorted(rank_factors.keys()):
        r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=1, rebalance_days=5, atr_stop_mult=0.8)
        if r:
            r['test'] = f'{fname}_A0.8'
            results.append(r)
    print(f"  Singles done: {len(results)}", flush=True)

    # =====================================================================
    # TEST 2: V41 optimal + each rank dynamics factor
    # =====================================================================
    print("\n  Test 2: V41 + rank dynamics...", flush=True)
    for fname in sorted(rank_factors.keys()):
        for weight in [0.1, 0.2, 0.3]:
            weights = {**v41_weights, fname: weight}
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=0.8)
            if r:
                r['test'] = f'V41+{fname}_W{weight}'
                results.append(r)
    print(f"  V41 combos done: {len(results)}", flush=True)

    # =====================================================================
    # TEST 3: Rank dynamics ONLY (no V15 factors)
    # =====================================================================
    print("\n  Test 3: Rank dynamics combos...", flush=True)
    combos = [
        {'R_RANK_MOM_RM5': 0.5, 'R_RANK_STAB_RS10': 0.5},
        {'R_RANK_MOM_RM3': 0.4, 'R_RANK_MOM_RM10': 0.3, 'R_RANK_STAB_RS5': 0.3},
        {'R_RANK_SHIFT': 0.5, 'R_RANK_STAB_RS10': 0.5},
        {'R_RANK_HRC_HC10': 0.5, 'R_RANK_MOM_RM5': 0.5},
        {'R_RANK_ACCEL': 0.3, 'R_RANK_MOM_RM5': 0.3, 'R_RANK_STAB_RS10': 0.4},
    ]
    for weights in combos:
        r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=1, rebalance_days=5, atr_stop_mult=0.8)
        if r:
            tag = '_'.join(k[7:10] for k in sorted(weights.keys()))
            r['test'] = f'RD_{tag}'
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
    print(f"  ALL RESULTS (V42 RANK DYNAMICS)", flush=True)
    print(f"  {'Test':<40s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*85}", flush=True)
    for r in results:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<40s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    # Best by category
    for cat_name, filter_fn in [
        ('Single factors', lambda t: not t.startswith('V41+') and not t.startswith('RD_')),
        ('V41 combos', lambda t: t.startswith('V41+')),
        ('Rank-only combos', lambda t: t.startswith('RD_')),
    ]:
        cat = [r for r in results if filter_fn(r['test'])]
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
        print(f"\n  === V42 BEST vs V41 RECORD ===", flush=True)
        print(f"  V42: {best['test']} = {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)
        print(f"  V41: V15B_EQUAL_A0.8 = +342.0% DD=53.7%", flush=True)
        delta = best['ann'] - 342.0
        print(f"  Delta: {delta:+.1f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
