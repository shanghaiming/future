"""
Alpha V35 — Engine Optimization Sweep
======================================
V15 found the best FACTOR combo (HAR-RV + BWP_BNW = +235.6%).
But we never optimized the TRADING ENGINE parameters!

V35 tests engine parameters with V15's best factors:
  1. Rebalance period: 3, 5, 7, 10, 15, 20 days
  2. ATR multiplier: 0.6, 0.8, 1.0, 1.2, 1.5, 2.0
  3. Top_n: 1, 2
  4. Time stop: 30, 45, 60, 90 days (or none)
  5. Combined best params

Using V7c bug-fixed engine (no look-ahead).

Factor combo from V15 best:
  BWP_BNW (0.3) + HAR_RV_RATIO_INV (0.3) + R_SQUARED (0.2) + SMA_DEV (0.2)

But V15 code actually used:
  BWP_BNW (0.25) + TENSION (0.25) + R_SQUARED (0.2) + SMA_DEV (0.15) + HAR_RV (0.15)

Test BOTH combos to see which responds better to engine optimization.

NO LOOK-AHEAD: Same V7c engine, just parameter sweep.
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN
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
from alpha_v7c import backtest_v7c


def backtest_v35(factor_weights, factors, NS, ND, dates, C, O, H, L, V,
                 top_n=1, rebalance_days=10, atr_stop_mult=1.0,
                 max_hold_days=60):
    """V35 backtest with configurable time stop.

    Same as V7c but with configurable max_hold_days.
    NO LOOK-AHEAD: all data uses di-1, trades at O[si,di].
    """
    from alpha_v7 import COMMISSION, STAMP_DUTY, CASH0

    factor_names = list(factor_weights.keys())
    weights = np.array([factor_weights[f] for f in factor_names])

    cash = float(CASH0)
    holdings = []
    trades = []
    last_rebalance = -999
    year_stats = {}

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # ATR stop loss — FIXED: no look-ahead
        for pos in list(holdings):
            si = pos['si']
            stopped_out = False

            if atr_stop_mult > 0:
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
                else:
                    atr = 0

                if atr > 0:
                    stop = pos['hw'] - atr_stop_mult * atr
                    today_low = L[si, di]
                    today_open = O[si, di]

                    if not np.isnan(today_low) and today_low <= stop:
                        if not np.isnan(today_open) and today_open < stop:
                            sp = today_open
                        else:
                            sp = stop
                        pnl = (sp - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * sp * (1 - COMMISSION - STAMP_DUTY)
                        trades.append({'pnl': pnl, 'days': (dates[di] - pos['ed']).days,
                                       'di': di, 'reason': 'stop', 'year': year})
                        holdings.remove(pos)
                        stopped_out = True

            if not stopped_out:
                today_high = H[si, di]
                if not np.isnan(today_high) and today_high > 0:
                    pos['hw'] = max(pos['hw'], today_high)

            # Time stop: configurable max hold
            if pos in holdings:
                days_held = (dates[di] - pos['ed']).days
                if max_hold_days > 0 and days_held >= max_hold_days:
                    sp = O[si, di] if not np.isnan(O[si, di]) else C[si, di]
                    if not np.isnan(sp) and sp > 0:
                        pnl = (sp - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * sp * (1 - COMMISSION - STAMP_DUTY)
                        trades.append({'pnl': pnl, 'days': days_held,
                                       'di': di, 'reason': 'time_stop', 'year': year})
                        holdings.remove(pos)

        # Rebalance
        if di - last_rebalance < rebalance_days:
            continue

        # Composite score
        composite = np.zeros(NS)
        count = np.zeros(NS)
        for fname, w in zip(factor_names, weights):
            if fname not in factors:
                continue
            arr = factors[fname]
            vals = arr[:, di]
            valid = ~np.isnan(vals)
            if valid.sum() < 50:
                continue
            composite[valid] += w * vals[valid]
            count[valid] += abs(w)

        mask = count > 0
        if mask.sum() < top_n * 2:
            continue
        composite[mask] /= count[mask]
        composite[~mask] = -9999

        top_indices = set(np.argsort(-composite)[:top_n])
        current_indices = set(h['si'] for h in holdings)

        # Sell
        to_sell = current_indices - top_indices
        for pos in list(holdings):
            if pos['si'] in to_sell:
                sp = O[pos['si'], di]
                if np.isnan(sp) or sp <= 0:
                    sp = C[pos['si'], di]
                if not np.isnan(sp) and sp > 0:
                    pnl = (sp - pos['entry']) / pos['entry'] * 100
                    cash += pos['shares'] * sp * (1 - COMMISSION - STAMP_DUTY)
                    trades.append({'pnl': pnl, 'days': (dates[di] - pos['ed']).days,
                                   'di': di, 'reason': 'rebalance', 'year': year})
                    holdings.remove(pos)

        # Buy
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
                                'ed': dates[di], 'hw': p
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

    # Exit reason analysis
    exit_reasons = {}
    for t in trades:
        r = t.get('reason', 'unknown')
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

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
        'exit_reasons': exit_reasons,
    }


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V35 — Engine Optimization Sweep", flush=True)
    print("  Rebalance × ATR × TopN × TimeStop", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Load all factors
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

    print(f"\n  Total factors: {len(all_factors)}", flush=True)

    # V15 best factor combos (both versions)
    # Version A: from V15 research notes
    weights_A = {'R_BWP_BNW': 0.3, 'R_HAR_RV_RATIO_INV': 0.3,
                 'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2}
    # Version B: from V15 actual code
    weights_B = {'R_BWP_BNW': 0.25, 'R_TENSION': 0.25,
                 'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.15, 'R_HAR_RV_RATIO_INV': 0.15}

    results = []

    # =====================================================================
    # PHASE 1: Rebalance period sweep (fixed ATR=1.0, top_n=1)
    # =====================================================================
    print("\n  Phase 1: Rebalance period sweep...", flush=True)
    for rebal in [3, 5, 7, 10, 15, 20]:
        for wa, tag in [(weights_A, 'A'), (weights_B, 'B')]:
            r = backtest_v35(wa, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=rebal, atr_stop_mult=1.0,
                            max_hold_days=60)
            if r:
                r['test'] = f'R{rebal}_A1.0_{tag}'
                results.append(r)
    print(f"  Phase 1 done: {len(results)} results", flush=True)

    # =====================================================================
    # PHASE 2: ATR multiplier sweep (fixed rebal=10, top_n=1)
    # =====================================================================
    print("\n  Phase 2: ATR multiplier sweep...", flush=True)
    phase2_start = len(results)
    for atr in [0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
        for wa, tag in [(weights_A, 'A'), (weights_B, 'B')]:
            r = backtest_v35(wa, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=10, atr_stop_mult=atr,
                            max_hold_days=60)
            if r:
                r['test'] = f'ATR{atr}_R10_{tag}'
                results.append(r)
    print(f"  Phase 2 done: +{len(results) - phase2_start} results", flush=True)

    # =====================================================================
    # PHASE 3: Time stop sweep (fixed rebal=10, ATR=1.0, top_n=1)
    # =====================================================================
    print("\n  Phase 3: Time stop sweep...", flush=True)
    phase3_start = len(results)
    for ts in [0, 30, 45, 60, 90, 120]:
        for wa, tag in [(weights_A, 'A'), (weights_B, 'B')]:
            r = backtest_v35(wa, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=10, atr_stop_mult=1.0,
                            max_hold_days=ts)
            if r:
                r['test'] = f'TS{ts}_R10_A1.0_{tag}'
                results.append(r)
    print(f"  Phase 3 done: +{len(results) - phase3_start} results", flush=True)

    # =====================================================================
    # PHASE 4: Top_n=2 with best params
    # =====================================================================
    print("\n  Phase 4: Top_n=2 sweep...", flush=True)
    phase4_start = len(results)
    for rebal in [5, 7, 10, 15]:
        for atr in [0.8, 1.0, 1.2, 1.5]:
            for wa, tag in [(weights_A, 'A'), (weights_B, 'B')]:
                r = backtest_v35(wa, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=2, rebalance_days=rebal, atr_stop_mult=atr,
                                max_hold_days=60)
                if r:
                    r['test'] = f'T2_R{rebal}_A{atr}_{tag}'
                    results.append(r)
    print(f"  Phase 4 done: +{len(results) - phase4_start} results", flush=True)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  TOP 50 RESULTS (V35 ENGINE SWEEP)", flush=True)
    print(f"  {'Test':<35s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*80}", flush=True)
    for r in results[:50]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<35s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    # Phase analysis
    print(f"\n  === PHASE ANALYSIS ===", flush=True)

    # Phase 1: Best rebalance
    p1 = [r for r in results if r['test'].startswith('R') and '_A1.0_' in r['test']]
    if p1:
        print(f"\n  Phase 1 — Rebalance (ATR=1.0, T1):", flush=True)
        for r in sorted(p1, key=lambda x: -x['ann'])[:6]:
            pos = " ALL+" if all_positive(r) else ""
            print(f"    {r['test']:<35s} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}%{pos}", flush=True)

    # Phase 2: Best ATR
    p2 = [r for r in results if r['test'].startswith('ATR') and '_R10_' in r['test']]
    if p2:
        print(f"\n  Phase 2 — ATR multiplier (R10, T1):", flush=True)
        for r in sorted(p2, key=lambda x: -x['ann'])[:8]:
            pos = " ALL+" if all_positive(r) else ""
            print(f"    {r['test']:<35s} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}%{pos}", flush=True)

    # Phase 3: Time stop
    p3 = [r for r in results if r['test'].startswith('TS') and '_R10_A1.0_' in r['test']]
    if p3:
        print(f"\n  Phase 3 — Time stop (R10, A1.0, T1):", flush=True)
        for r in sorted(p3, key=lambda x: -x['ann'])[:8]:
            pos = " ALL+" if all_positive(r) else ""
            print(f"    {r['test']:<35s} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}%{pos}", flush=True)

    # Phase 4: Top_n=2
    p4 = [r for r in results if r['test'].startswith('T2_')]
    if p4:
        print(f"\n  Phase 4 — Top_n=2:", flush=True)
        for r in sorted(p4, key=lambda x: -x['ann'])[:10]:
            pos = " ALL+" if all_positive(r) else ""
            print(f"    {r['test']:<35s} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}%{pos}", flush=True)

    # Top 5 year-by-year
    for i, r in enumerate(results[:5]):
        print(f"\n  Year-by-year #{i+1}: {r['test']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        if 'exit_reasons' in r:
            print(f"    Exit reasons: {r['exit_reasons']}", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    # Compare best V35 vs V15 baseline
    if results:
        best = results[0]
        print(f"\n  === V35 BEST vs V15 BASELINE ===", flush=True)
        print(f"  V35: {best['test']} = {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)
        print(f"  V15: HAR_RV_T1_A1.0 = +235.6% DD=32.4%", flush=True)
        delta = best['ann'] - 235.6
        print(f"  Delta: {delta:+.1f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
