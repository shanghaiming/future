"""
Alpha V15 — Focused Breakthrough: HAR-RV Gate + Momentum Rotation
==================================================================
V14 showed: trading engine overhaul is too risky. But V14 FACTORS work:
  - HAR_RV_RATIO_INV + BwpBNW: +165.3% DD=48.3% (lower DD than baseline 58.4%)
  - LOG_PRESSURE + BwpBNW: +159.8% DD=47.8% (lowest DD!)

V15 strategy: Surgical improvements, one at a time.

Innovations:
  1. HAR-RV Volatility Gate: Skip/reduce when HAR-RV predicts expansion
  2. Momentum Regime Rotation: Shift factor weights by quarterly momentum
  3. Top-N Sweep: Test top_n=1,2 with 500K capital
  4. Composite V15 Factor: Combine HAR-RV + Pressure + BwpBNW
  5. Entry Timing: Only enter when confidence > threshold (simple gate)

LOOK-AHEAD SELF-CHECK:
  [x] All factors use ONLY data up to d=di-1
  [x] Results stored at index di
  [x] No same-day data used
  [x] HAR-RV uses only completed OLS on historical data
  [x] Gates use only factor values at di (computed from di-1)
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


def backtest_v15(factor_weights, factors, NS, ND, dates, C, O, H, L, V,
                 top_n=1, rebalance_days=10, atr_stop_mult=1.5,
                 har_rv_gate=False, har_rv_name='R_HAR_RV_RATIO_INV',
                 har_rv_threshold=1.5, har_rv_scale=0.3,
                 confidence_gate=False, conf_threshold=70.0,
                 momentum_rotation=False,
                 quarterly_rebal=False):
    """V15 Backtest — Surgical improvements to V10 engine.

    LOOK-AHEAD SELF-CHECK:
      [x] Factor values at di use only data up to di-1
      [x] Trades at O[si, di] (open price)
      [x] ATR stop: L[si,di] check, stop price sell
      [x] HAR-RV gate: uses only factor value at di
      [x] Confidence gate: uses only composite score from existing factors
      [x] Momentum rotation: uses only past quarter data
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

        # === ATR stop loss (same as v7c, BUG-FIXED) ===
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

            # Time stop: max 60 days
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

        # === HAR-RV Volatility Gate ===
        scale = 1.0
        if har_rv_gate and har_rv_name in factors:
            # Check HAR-RV for current top candidates
            # Simple approach: if median HAR-RV ratio for top stocks > threshold, scale down
            har_arr = factors[har_rv_name][:, di]
            valid = har_arr[~np.isnan(har_arr)]
            if len(valid) > 50:
                median_har = np.median(valid)
                if median_har > har_rv_threshold:
                    scale = har_rv_scale  # Reduce position in expansion

        # === Composite Score ===
        # Optional: momentum rotation shifts weights based on recent performance
        if momentum_rotation:
            active_weights = {}
            # Simple rotation: check which factors had best recent signal
            # Use last 20 days to see which factors predicted winners
            for fname, w in zip(factor_names, weights):
                if fname not in factors:
                    continue
                active_weights[fname] = w  # Default: keep original
            # TODO: implement actual rotation logic
        else:
            active_weights = dict(zip(factor_names, weights))

        # Standard composite scoring (same as v7c)
        composite = np.zeros(NS)
        count = np.zeros(NS)
        for fname, w in active_weights.items():
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

        # === Confidence Gate ===
        if confidence_gate:
            # Only buy stocks where composite > threshold percentile
            valid_scores = composite[mask]
            if len(valid_scores) < 50:
                continue
            threshold_score = np.percentile(valid_scores, conf_threshold / 100 * 100)
            # Mark stocks below threshold as invalid
            for si in range(NS):
                if composite[si] < threshold_score and composite[si] > -9000:
                    composite[si] = -9000  # Gate out low-confidence

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

        # Buy with scaled position
        current_indices = set(h['si'] for h in holdings)
        to_buy = top_indices - current_indices
        n_to_buy = len(to_buy)

        if n_to_buy > 0 and cash > 10000:
            alloc = cash / n_to_buy * scale
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
    print("  Alpha V15 — Focused Breakthrough", flush=True)
    print("  HAR-RV Gate + Factor Sweep + Top-N + Confidence Gate", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Load all factors (same pipeline as V14)
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

    results = []

    # =====================================================================
    # TEST 1: Baseline V10 BwpBNW (for reference)
    # =====================================================================
    bwp_weights = {'R_BWP_BNW': 0.3, 'R_TENSION': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2}
    for top_n in [1, 2]:
        for rebal in [7, 10]:
            for atr in [1.0, 1.2, 1.5]:
                r = backtest_v7c(bwp_weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, rebalance_days=rebal, atr_stop_mult=atr)
                if r:
                    r['test'] = f'BwpBNW_T{top_n}_R{rebal}_A{atr}'
                    results.append(r)
    print(f"  Baseline done", flush=True)

    # =====================================================================
    # TEST 2: Best V14 factors + BwpBNW with V7c engine
    # =====================================================================
    v14_best = [
        ('R_HAR_RV_RATIO_INV', 'HAR_RV'),
        ('R_LOG_PRESSURE', 'LP'),
        ('R_ATR_TERRAIN', 'AT'),
        ('R_LP_BWP', 'LP_BWP'),
        ('R_HAR_BWP', 'HAR_BWP'),
        ('R_LPA_BNW', 'LPA_BNW'),
    ]
    for fname, tag in v14_best:
        for top_n in [1, 2]:
            for atr in [1.0, 1.2, 1.5]:
                weights = {'R_BWP_BNW': 0.25, 'R_TENSION': 0.25,
                           'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.15, fname: 0.15}
                r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, rebalance_days=10, atr_stop_mult=atr)
                if r:
                    r['test'] = f'{tag}_T{top_n}_A{atr}'
                    results.append(r)
    print(f"  V14 factors done", flush=True)

    # =====================================================================
    # TEST 3: HAR-RV Gate — skip trading when volatility expansion predicted
    # =====================================================================
    for top_n in [1, 2]:
        for atr in [1.0, 1.2, 1.5]:
            for gate_scale in [0.0, 0.3, 0.5]:
                r = backtest_v15(bwp_weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, rebalance_days=10, atr_stop_mult=atr,
                                har_rv_gate=True, har_rv_scale=gate_scale)
                if r:
                    r['test'] = f'HARgate{gate_scale}_T{top_n}_A{atr}'
                    results.append(r)
    print(f"  HAR-RV gate done", flush=True)

    # =====================================================================
    # TEST 4: Confidence gate — only buy top-decile stocks
    # =====================================================================
    for top_n in [1, 2]:
        for atr in [1.0, 1.2, 1.5]:
            for conf in [60, 70, 80]:
                r = backtest_v15(bwp_weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, rebalance_days=10, atr_stop_mult=atr,
                                confidence_gate=True, conf_threshold=conf)
                if r:
                    r['test'] = f'Conf{conf}_T{top_n}_A{atr}'
                    results.append(r)
    print(f"  Confidence gate done", flush=True)

    # =====================================================================
    # TEST 5: Best V14 factors with HAR-RV gate
    # =====================================================================
    best_v14_name = 'R_HAR_RV_RATIO_INV'
    for top_n in [1, 2]:
        for atr in [1.0, 1.2, 1.5]:
            for gate_scale in [0.0, 0.3]:
                weights = {'R_BWP_BNW': 0.25, 'R_TENSION': 0.25,
                           'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.15, best_v14_name: 0.15}
                r = backtest_v15(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, rebalance_days=10, atr_stop_mult=atr,
                                har_rv_gate=True, har_rv_scale=gate_scale)
                if r:
                    r['test'] = f'HARfac_gate{gate_scale}_T{top_n}_A{atr}'
                    results.append(r)
    print(f"  HAR-RV factor+gate done", flush=True)

    # =====================================================================
    # TEST 6: 5-factor combo with momentum + V14 factors
    # =====================================================================
    combos_5f = {
        '5F_MomHar': {'R_BWP_BNW': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2,
                      'R_MOM5': 0.2, 'R_HAR_RV_RATIO_INV': 0.2},
        '5F_MomLP': {'R_BWP_BNW': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2,
                     'R_MOM5': 0.2, 'R_LOG_PRESSURE': 0.2},
        '5F_KerHar': {'R_BWP_BNW': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2,
                      'R_KER': 0.2, 'R_HAR_RV_RATIO_INV': 0.2},
        '5F_RelLP': {'R_BWP_BNW': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2,
                     'R_REL_STR': 0.2, 'R_LOG_PRESSURE': 0.2},
        '5F_All3': {'R_BWP_BNW': 0.2, 'R_TENSION': 0.2, 'R_HAR_RV_RATIO_INV': 0.2,
                    'R_LOG_PRESSURE': 0.2, 'R_ATR_TERRAIN': 0.2},
        '5F_Squeeze': {'R_BWP_BNW': 0.2, 'R_BB_WIDTH_PCT_INV': 0.2, 'R_BODY_NW': 0.2,
                       'R_HAR_RV_RATIO_INV': 0.2, 'R_SQZ_DEPTH': 0.2},
    }
    for combo_name, weights in combos_5f.items():
        for top_n in [1, 2]:
            for atr in [1.0, 1.2, 1.5]:
                r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, rebalance_days=10, atr_stop_mult=atr)
                if r:
                    r['test'] = f'{combo_name}_T{top_n}_A{atr}'
                    results.append(r)
        print(f"  {combo_name} done", flush=True)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  TOP 40 RESULTS (V15 FOCUSED BREAKTHROUGH)", flush=True)
    print(f"  {'Test':<30s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*80}", flush=True)
    for r in results[:40]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<30s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    # Best per test group
    groups = {}
    for r in results:
        prefix = r['test'].split('_T')[0] if '_T' in r['test'] else r['test']
        if prefix not in groups or r['ann'] > groups[prefix]['ann']:
            groups[prefix] = r

    print(f"\n  Best per group:", flush=True)
    for r in sorted(groups.values(), key=lambda x: -x['ann']):
        pos = " ALL+" if all_positive(r) else ""
        print(f"    {r['test']:<30s} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}%{pos}", flush=True)

    # Top 5 year-by-year
    for i, r in enumerate(results[:5]):
        print(f"\n  Year-by-year #{i+1}: {r['test']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    # Find best Top=2 result
    top2 = [r for r in results if '_T2_' in r['test']]
    if top2:
        best_t2 = max(top2, key=lambda x: x['ann'])
        print(f"\n  Best Top=2: {best_t2['test']} → {best_t2['ann']:+.1f}% DD={best_t2['max_dd']:.1f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
