"""
Alpha V37 — Factor Confidence Position Sizing
==============================================
V29 showed that gating (skip trades) doesn't work — it's either too loose
or too tight. V36 tests adaptive stops. V37 tests a different approach:

Scale position size based on FACTOR CONFIDENCE:
  - If composite score is top 5%: Full allocation (100%)
  - If top 10-20%: 80% allocation
  - If top 20-30%: 60% allocation
  - If top 30-50%: 40% allocation
  - Below top 50%: Don't buy (implicit gate)

Also test:
  - Step function sizing vs linear sizing
  - Kelly-inspired sizing: f* = (p×b - q) / b where p = win rate proxy
  - Composite score as continuous position scaling

Factor combo: V15 winner (BWP_BNW + HAR_RV + R_SQUARED + SMA_DEV)

NO LOOK-AHEAD: Factor scores use di-1 data, position sizing is deterministic.
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


def backtest_v37(factor_weights, factors, NS, ND, dates, C, O, H, L, V,
                 top_n=1, rebalance_days=10, atr_stop_mult=1.0,
                 sizing='uniform', min_pct_rank=50):
    """V37 backtest with confidence-based position sizing.

    Sizing modes:
      'uniform': Standard equal allocation (baseline)
      'step': Step function based on percentile rank
      'linear': Continuous scaling from min_pct_rank to 100
      'kelly': Kelly-inspired based on historical win rate
      'composite': Use composite score directly as position fraction

    NO LOOK-AHEAD: All data uses di-1, position sizing at rebalance time.
    """
    factor_names = list(factor_weights.keys())
    weights = np.array([factor_weights[f] for f in factor_names])

    cash = float(CASH0)
    holdings = []
    trades = []
    last_rebalance = -999
    year_stats = {}
    # For Kelly sizing: track running win rate
    kelly_wins = 0
    kelly_total = 0

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # ATR stop loss — same as V7c
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

        # Compute percentile rank for each stock
        valid_scores = composite[mask]
        if len(valid_scores) < 50:
            continue

        pct_rank = np.zeros(NS)
        for si in range(NS):
            if composite[si] > -9000:
                pct_rank[si] = np.sum(valid_scores <= composite[si]) / len(valid_scores) * 100

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
                    # Update Kelly tracking
                    kelly_total += 1
                    if pnl > 0:
                        kelly_wins += 1

        # Buy with position sizing
        current_indices = set(h['si'] for h in holdings)
        to_buy = top_indices - current_indices
        n_to_buy = len(to_buy)
        if n_to_buy > 0 and cash > 10000:
            base_alloc = cash / n_to_buy

            for si in to_buy:
                # Determine position scale based on sizing mode
                if sizing == 'uniform':
                    scale = 1.0
                elif sizing == 'step':
                    pr = pct_rank[si]
                    if pr >= 95:
                        scale = 1.0
                    elif pr >= 90:
                        scale = 0.8
                    elif pr >= 80:
                        scale = 0.6
                    elif pr >= 70:
                        scale = 0.5
                    elif pr >= min_pct_rank:
                        scale = 0.4
                    else:
                        continue  # Skip low-confidence
                elif sizing == 'linear':
                    pr = pct_rank[si]
                    if pr < min_pct_rank:
                        continue
                    scale = (pr - min_pct_rank) / (100 - min_pct_rank)
                    scale = max(scale, 0.2)  # Minimum 20%
                elif sizing == 'kelly':
                    if kelly_total > 20:
                        p = kelly_wins / kelly_total  # Win probability
                        avg_w_val = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if any(t['pnl'] > 0 for t in trades) else 5
                        avg_l_val = abs(np.mean([t['pnl'] for t in trades if t['pnl'] <= 0])) if any(t['pnl'] <= 0 for t in trades) else 3
                        b = avg_w_val / max(avg_l_val, 0.1)  # Win/loss ratio
                        q = 1 - p
                        f_star = max((p * b - q) / max(b, 0.1), 0.1)
                        f_star = min(f_star, 1.5)  # Cap at 150%
                        scale = f_star
                    else:
                        scale = 1.0  # Not enough data yet
                elif sizing == 'composite':
                    # Scale by composite score percentile
                    pr = pct_rank[si]
                    scale = pr / 100.0
                    if scale < 0.3:
                        continue
                else:
                    scale = 1.0

                alloc = base_alloc * scale
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
    print("  Alpha V37 — Factor Confidence Position Sizing", flush=True)
    print("  Step/Linear/Kelly/Composite position scaling", flush=True)
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

    weights_A = {'R_BWP_BNW': 0.3, 'R_HAR_RV_RATIO_INV': 0.3,
                 'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2}
    weights_B = {'R_BWP_BNW': 0.25, 'R_TENSION': 0.25,
                 'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.15, 'R_HAR_RV_RATIO_INV': 0.15}

    results = []

    # =====================================================================
    # TEST 1: Baseline uniform
    # =====================================================================
    print("\n  Test 1: Baseline uniform...", flush=True)
    for wa, tag in [(weights_A, 'A'), (weights_B, 'B')]:
        for atr in [0.8, 1.0, 1.2]:
            r = backtest_v37(wa, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=10, atr_stop_mult=atr,
                            sizing='uniform')
            if r:
                r['test'] = f'UNI_A{atr}_{tag}'
                results.append(r)

    # =====================================================================
    # TEST 2: Step function sizing
    # =====================================================================
    print("  Test 2: Step sizing...", flush=True)
    for min_pr in [30, 40, 50, 60]:
        for atr in [0.8, 1.0, 1.2]:
            for wa, tag in [(weights_A, 'A')]:
                r = backtest_v37(wa, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=10, atr_stop_mult=atr,
                                sizing='step', min_pct_rank=min_pr)
                if r:
                    r['test'] = f'STEP_p{min_pr}_A{atr}_{tag}'
                    results.append(r)

    # =====================================================================
    # TEST 3: Linear sizing
    # =====================================================================
    print("  Test 3: Linear sizing...", flush=True)
    for min_pr in [30, 40, 50, 60, 70]:
        for atr in [0.8, 1.0, 1.2]:
            for wa, tag in [(weights_A, 'A')]:
                r = backtest_v37(wa, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=10, atr_stop_mult=atr,
                                sizing='linear', min_pct_rank=min_pr)
                if r:
                    r['test'] = f'LIN_p{min_pr}_A{atr}_{tag}'
                    results.append(r)

    # =====================================================================
    # TEST 4: Kelly sizing
    # =====================================================================
    print("  Test 4: Kelly sizing...", flush=True)
    for atr in [0.8, 1.0, 1.2, 1.5]:
        for wa, tag in [(weights_A, 'A'), (weights_B, 'B')]:
            r = backtest_v37(wa, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=10, atr_stop_mult=atr,
                            sizing='kelly')
            if r:
                r['test'] = f'KELLY_A{atr}_{tag}'
                results.append(r)

    # =====================================================================
    # TEST 5: Composite score sizing
    # =====================================================================
    print("  Test 5: Composite sizing...", flush=True)
    for atr in [0.8, 1.0, 1.2]:
        for wa, tag in [(weights_A, 'A')]:
            r = backtest_v37(wa, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=10, atr_stop_mult=atr,
                            sizing='composite')
            if r:
                r['test'] = f'COMP_A{atr}_{tag}'
                results.append(r)

    # =====================================================================
    # TEST 6: Top_n=2 with best sizing
    # =====================================================================
    print("  Test 6: T2 sizing...", flush=True)
    for sizing in ['uniform', 'step', 'linear', 'kelly']:
        for atr in [0.8, 1.0, 1.2]:
            r = backtest_v37(weights_A, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=2, rebalance_days=10, atr_stop_mult=atr,
                            sizing=sizing, min_pct_rank=50)
            if r:
                r['test'] = f'T2_{sizing[:3].upper()}_A{atr}'
                results.append(r)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  TOP 40 RESULTS (V37 POSITION SIZING)", flush=True)
    print(f"  {'Test':<35s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*80}", flush=True)
    for r in results[:40]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<35s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    # Best per sizing mode
    modes = {}
    for r in results:
        for mode in ['UNI', 'STEP', 'LIN', 'KELLY', 'COMP', 'T2']:
            if r['test'].startswith(mode):
                if mode not in modes or r['ann'] > modes[mode]['ann']:
                    modes[mode] = r

    print(f"\n  Best per sizing mode:", flush=True)
    for mode, r in sorted(modes.items(), key=lambda x: -x[1]['ann']):
        pos = " ALL+" if all_positive(r) else ""
        print(f"    {r['test']:<35s} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}%{pos}", flush=True)

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
        print(f"\n  === V37 BEST vs V15 BASELINE ===", flush=True)
        print(f"  V37: {best['test']} = {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)
        print(f"  V15: HAR_RV_T1_A1.0 = +235.6% DD=32.4%", flush=True)
        print(f"  Delta: {best['ann'] - 235.6:+.1f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
