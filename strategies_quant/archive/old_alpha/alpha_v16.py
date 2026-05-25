"""
Alpha V16 — Confluence Breakthrough: Epanechnikov + Momentum Confirmation
=========================================================================
V14 engine failed (+2.8%), V15 testing gates and factor combos.

V16 takes a completely different approach: instead of modifying the engine,
modify the FACTOR SCORING to be non-linear.

Key insight from 260-strategy study:
  - BB_WIDTH_PCT_INV × BODY_NW works because it captures compression+quality
  - But it treats rank=51 and rank=99 the same way (linear weight)
  - In reality, only rank>80 matters — top quintile is where alpha lives

V16 Innovation: Epanechnikov-Weighted Factor Scoring
  - Apply Epanechnikov kernel to EACH factor before combining
  - K(u) = 0.75 × (1 - u²) where u = (100 - rank) / 50
  - rank=100 → K=0.75, rank=75 → K=0.56, rank=50 → K=0, rank<50 → K=0
  - This means: only stocks in TOP HALF of each factor contribute
  - Stocks strong across ALL factors get exponentially boosted

Additional ideas:
  - Momentum confirmation: only buy when MOM5 rank > 50
  - Volume confirmation: only buy when VOL_ANOMALY rank > 50
  - Squeeze + Body + Momentum triple convergence

LOOK-AHEAD SELF-CHECK:
  [x] All factors use ONLY data up to d=di-1
  [x] Results stored at index di
  [x] No same-day data used
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7 import compute_all_factors, COMMISSION, STAMP_DUTY, CASH0
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


def compute_epanechnikov_factors(all_factors, NS, ND, factor_names):
    """Apply Epanechnikov kernel to create non-linear factors.

    For each factor X with rank in [0, 100]:
      Epan(X) = K(u) × X where u = (100 - X) / 50, K(u) = 0.75×(1-u²) for u≤1

    rank=100 → Epan = 0.75 × 100 = 75
    rank=80  → u=0.4, K=0.75×(1-0.16)=0.63, Epan = 0.63 × 80 = 50.4
    rank=60  → u=0.8, K=0.75×(1-0.64)=0.27, Epan = 0.27 × 60 = 16.2
    rank=50  → u=1.0, K=0, Epan = 0
    rank<50  → K=0, Epan = 0

    This creates a "soft AND" gate: only stocks in the top half survive.
    """
    new = {}
    for fname in factor_names:
        if fname not in all_factors:
            continue
        arr = all_factors[fname]
        epan = np.full_like(arr, np.nan)

        for di in range(60, ND):
            for si in range(NS):
                v = arr[si, di]
                if np.isnan(v):
                    continue
                u = (100.0 - v) / 50.0  # 0 for rank=100, 1 for rank=50, 2 for rank=0
                if u <= 1.0:
                    k = 0.75 * (1.0 - u * u)
                    epan[si, di] = k * v
                else:
                    epan[si, di] = 0.0  # Below threshold = zero

        new[f'E_{fname}'] = epan

    return new


def backtest_v16(factor_weights, factors, NS, ND, dates, C, O, H, L, V,
                 top_n=1, rebalance_days=10, atr_stop_mult=1.5,
                 use_epanechnikov=False, epan_factor_names=None,
                 momentum_confirm=False, mom_factor='R_MOM5', mom_threshold=50,
                 volume_confirm=False, vol_factor='R_VOL_ANOMALY', vol_threshold=50):
    """V16 Backtest — Epanechnikov scoring + confirmation filters.

    LOOK-AHEAD SELF-CHECK:
      [x] All factors at di use only data up to di-1
      [x] Epanechnikov applied to ranked factors (no look-ahead)
      [x] Confirmation filters use only factor values at di
      [x] ATR stop: same BUG-FIXED logic as v7c
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

        # === ATR stop loss (BUG-FIXED) ===
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

            if pos in holdings:
                days_held = (dates[di] - pos['ed']).days
                if days_held >= 60:
                    sp = O[si, di] if not np.isnan(O[si, di]) else C[si, di]
                    if not np.isnan(sp) and sp > 0:
                        pnl = (sp - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * sp * (1 - COMMISSION - STAMP_DUTY)
                        trades.append({'pnl': pnl, 'days': days_held,
                                       'di': di, 'reason': 'time_stop', 'year': year})
                        holdings.remove(pos)

        # === Rebalance ===
        if di - last_rebalance < rebalance_days:
            continue

        # === Composite Score ===
        composite = np.zeros(NS)
        count = np.zeros(NS)

        if use_epanechnikov and epan_factor_names:
            # Use Epanechnikov-weighted versions of specified factors
            for fname, w in zip(factor_names, weights):
                # Check if Epanechnikov version exists
                epan_name = f'E_{fname}'
                if fname in epan_factor_names and epan_name in factors:
                    arr = factors[epan_name]
                    vals = arr[:, di]
                    valid = ~np.isnan(vals)
                    composite[valid] += w * vals[valid]
                    count[valid] += abs(w)
                elif fname in factors:
                    arr = factors[fname]
                    vals = arr[:, di]
                    valid = ~np.isnan(vals)
                    composite[valid] += w * vals[valid]
                    count[valid] += abs(w)
        else:
            # Standard linear combination
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

        # === Momentum Confirmation ===
        if momentum_confirm and mom_factor in factors:
            mom_arr = factors[mom_factor][:, di]
            for si in range(NS):
                if np.isnan(mom_arr[si]) or mom_arr[si] < mom_threshold:
                    composite[si] = -9999  # Filter out negative momentum

        # === Volume Confirmation ===
        if volume_confirm and vol_factor in factors:
            vol_arr = factors[vol_factor][:, di]
            for si in range(NS):
                if np.isnan(vol_arr[si]) or vol_arr[si] < vol_threshold:
                    composite[si] = -9999  # Filter out low volume

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
    print("  Alpha V16 — Epanechnikov + Confirmation", flush=True)
    print("  Non-linear scoring + momentum/volume filters", flush=True)
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

    # Create Epanechnikov-weighted factors
    epan_names = ['R_BWP_BNW', 'R_TENSION', 'R_R_SQUARED', 'R_SMA_DEV',
                  'R_BODY_NW', 'R_BB_WIDTH_PCT_INV', 'R_MOM5', 'R_KER',
                  'R_HAR_RV_RATIO_INV', 'R_LOG_PRESSURE', 'R_ATR_TERRAIN']
    epan_factors = compute_epanechnikov_factors(all_factors, NS, ND, epan_names)
    all_factors.update(epan_factors)

    print(f"\n  Total factors: {len(all_factors)} (including {len(epan_factors)} Epan)", flush=True)

    results = []

    # =====================================================================
    # TEST 1: Baseline BwpBNW
    # =====================================================================
    bwp_weights = {'R_BWP_BNW': 0.3, 'R_TENSION': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2}
    for top_n in [1, 2]:
        for atr in [1.0, 1.2, 1.5]:
            r = backtest_v7c(bwp_weights, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=top_n, rebalance_days=10, atr_stop_mult=atr)
            if r:
                r['test'] = f'Base_T{top_n}_A{atr}'
                results.append(r)
    print(f"  Baseline done", flush=True)

    # =====================================================================
    # TEST 2: Epanechnikov scoring on BwpBNW factors
    # =====================================================================
    for top_n in [1, 2]:
        for atr in [1.0, 1.2, 1.5]:
            r = backtest_v16(bwp_weights, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=top_n, atr_stop_mult=atr,
                            use_epanechnikov=True,
                            epan_factor_names=['R_BWP_BNW', 'R_TENSION', 'R_R_SQUARED', 'R_SMA_DEV'])
            if r:
                r['test'] = f'Epan_T{top_n}_A{atr}'
                results.append(r)
    print(f"  Epanechnikov done", flush=True)

    # =====================================================================
    # TEST 3: Epanechnikov + Momentum confirmation
    # =====================================================================
    for top_n in [1, 2]:
        for atr in [1.0, 1.2, 1.5]:
            for mom_th in [40, 50, 60]:
                r = backtest_v16(bwp_weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, atr_stop_mult=atr,
                                use_epanechnikov=True,
                                epan_factor_names=['R_BWP_BNW', 'R_TENSION', 'R_R_SQUARED', 'R_SMA_DEV'],
                                momentum_confirm=True, mom_threshold=mom_th)
                if r:
                    r['test'] = f'Epan_M{mom_th}_T{top_n}_A{atr}'
                    results.append(r)
    print(f"  Epan+Mom done", flush=True)

    # =====================================================================
    # TEST 4: Epanechnikov + Volume confirmation
    # =====================================================================
    for top_n in [1, 2]:
        for atr in [1.0, 1.2, 1.5]:
            for vol_th in [40, 50, 60]:
                r = backtest_v16(bwp_weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, atr_stop_mult=atr,
                                use_epanechnikov=True,
                                epan_factor_names=['R_BWP_BNW', 'R_TENSION', 'R_R_SQUARED', 'R_SMA_DEV'],
                                volume_confirm=True, vol_threshold=vol_th)
                if r:
                    r['test'] = f'Epan_V{vol_th}_T{top_n}_A{atr}'
                    results.append(r)
    print(f"  Epan+Vol done", flush=True)

    # =====================================================================
    # TEST 5: Momentum confirmation only (no Epan)
    # =====================================================================
    for top_n in [1, 2]:
        for atr in [1.0, 1.2, 1.5]:
            for mom_th in [30, 40, 50]:
                r = backtest_v16(bwp_weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, atr_stop_mult=atr,
                                momentum_confirm=True, mom_threshold=mom_th)
                if r:
                    r['test'] = f'Mom{mom_th}_T{top_n}_A{atr}'
                    results.append(r)
    print(f"  Momentum confirm done", flush=True)

    # =====================================================================
    # TEST 6: Volume confirmation only (no Epan)
    # =====================================================================
    for top_n in [1, 2]:
        for atr in [1.0, 1.2, 1.5]:
            for vol_th in [30, 40, 50]:
                r = backtest_v16(bwp_weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, atr_stop_mult=atr,
                                volume_confirm=True, vol_threshold=vol_th)
                if r:
                    r['test'] = f'Vol{vol_th}_T{top_n}_A{atr}'
                    results.append(r)
    print(f"  Volume confirm done", flush=True)

    # =====================================================================
    # TEST 7: Epan on expanded factor set (5-6 factors)
    # =====================================================================
    expanded_weights = {
        'R_BWP_BNW': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2,
        'R_SMA_DEV': 0.1, 'R_MOM5': 0.15, 'R_HAR_RV_RATIO_INV': 0.15,
    }
    expanded_epan = ['R_BWP_BNW', 'R_TENSION', 'R_R_SQUARED', 'R_MOM5', 'R_HAR_RV_RATIO_INV']
    for top_n in [1, 2]:
        for atr in [1.0, 1.2, 1.5]:
            r = backtest_v16(expanded_weights, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=top_n, atr_stop_mult=atr,
                            use_epanechnikov=True, epan_factor_names=expanded_epan)
            if r:
                r['test'] = f'ExpEpan_T{top_n}_A{atr}'
                results.append(r)
    print(f"  Expanded Epan done", flush=True)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  TOP 40 RESULTS (V16 EPANECHNIKOV + CONFIRMATION)", flush=True)
    print(f"  {'Test':<30s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*80}", flush=True)
    for r in results[:40]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<30s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    # Top 5 year-by-year
    for i, r in enumerate(results[:5]):
        print(f"\n  Year-by-year #{i+1}: {r['test']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
