"""
Alpha V36 — Adaptive Stop-Loss + Position Sizing
=================================================
V29 showed gates don't work. V14 showed adaptive rebalancing kills.
But what about ADAPTIVE STOP-LOSS?

Key idea: Don't gate (skip trades). Instead, adjust stop-loss width
based on market volatility regime:
  - Low vol (HAR-RV < 25th pct): Tighter stop (0.8× ATR) — mean reversion mode
  - Normal vol: Standard stop (1.0× ATR)
  - High vol (HAR-RV > 75th pct): Wider stop (1.5× ATR) — ride trends

Also test:
  - Position scaling: Reduce position when HAR-RV is high
  - Combined: Adaptive stop + position scale
  - Per-stock HAR-RV (instead of market-wide)

Factor combo: V15 winner (BWP_BNW + HAR_RV + R_SQUARED + SMA_DEV)

NO LOOK-AHEAD: HAR-RV uses di-1 data, stop uses L[si,di] check.
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


def backtest_v36(factor_weights, factors, NS, ND, dates, C, O, H, L, V,
                 top_n=1, rebalance_days=10, base_atr_mult=1.0,
                 adaptive_stop=False, atr_mult_low=0.8, atr_mult_high=1.5,
                 position_scale=False, pos_scale_factor=0.5,
                 per_stock_har=False,
                 har_rv_name='R_HAR_RV_RATIO_INV'):
    """V36 backtest with adaptive stop-loss and/or position scaling.

    NO LOOK-AHEAD:
      - Factor values at di use only data up to di-1
      - ATR uses data up to di-1
      - Stop check uses L[si,di] (realistic intraday)
      - HAR-RV regime from factor value at di (computed from di-1)
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

        # === Compute market-wide HAR-RV regime ===
        market_har_pctl = 50  # Default: normal
        if (adaptive_stop or position_scale) and har_rv_name in factors:
            har_vals = factors[har_rv_name][:, di]  # 1D array for all stocks at di
            valid = har_vals[~np.isnan(har_vals)]
            if len(valid) > 50:
                med = np.median(valid)
                p25 = np.percentile(valid, 25)
                p75 = np.percentile(valid, 75)
                if med > p75:
                    market_har_pctl = 75  # High vol
                elif med < p25:
                    market_har_pctl = 25  # Low vol

        # ATR stop loss
        for pos in list(holdings):
            si = pos['si']
            stopped_out = False

            # Determine ATR multiplier
            if adaptive_stop:
                if per_stock_har and har_rv_name in factors:
                    # Per-stock HAR-RV regime
                    stock_har = factors[har_rv_name][si, di]
                    if not np.isnan(stock_har):
                        har_vals = factors[har_rv_name][:, di]
                        valid_h = har_vals[~np.isnan(har_vals)]
                        if len(valid_h) > 50:
                            p75 = np.percentile(valid_h, 75)
                            p25 = np.percentile(valid_h, 25)
                            if stock_har > p75:
                                use_mult = atr_mult_high
                            elif stock_har < p25:
                                use_mult = atr_mult_low
                            else:
                                use_mult = base_atr_mult
                        else:
                            use_mult = base_atr_mult
                    else:
                        use_mult = base_atr_mult
                else:
                    # Market-wide regime
                    if market_har_pctl >= 75:
                        use_mult = atr_mult_high
                    elif market_har_pctl <= 25:
                        use_mult = atr_mult_low
                    else:
                        use_mult = base_atr_mult
            else:
                use_mult = base_atr_mult

            if use_mult > 0:
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
                    stop = pos['hw'] - use_mult * atr
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
            # Position scaling based on market regime
            if position_scale and market_har_pctl >= 75:
                alloc *= pos_scale_factor  # Reduce position in high vol
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
    print("  Alpha V36 — Adaptive Stop-Loss + Position Sizing", flush=True)
    print("  Market-adaptive ATR multiplier + regime position scaling", flush=True)
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

    # V15 best factor combos
    weights_A = {'R_BWP_BNW': 0.3, 'R_HAR_RV_RATIO_INV': 0.3,
                 'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2}
    weights_B = {'R_BWP_BNW': 0.25, 'R_TENSION': 0.25,
                 'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.15, 'R_HAR_RV_RATIO_INV': 0.15}

    results = []

    # =====================================================================
    # TEST 1: Baseline (no adaptation) — should match V15
    # =====================================================================
    print("\n  Test 1: Baseline...", flush=True)
    for wa, tag in [(weights_A, 'A'), (weights_B, 'B')]:
        for atr in [0.8, 1.0, 1.2, 1.5]:
            r = backtest_v36(wa, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=10, base_atr_mult=atr)
            if r:
                r['test'] = f'BL_A{atr}_{tag}'
                results.append(r)

    # =====================================================================
    # TEST 2: Market-wide adaptive stop
    # =====================================================================
    print("  Test 2: Market adaptive stop...", flush=True)
    for atr_low, atr_high in [(0.6, 1.5), (0.8, 1.5), (0.8, 2.0), (1.0, 2.0), (0.6, 2.0)]:
        for base_atr in [0.8, 1.0, 1.2]:
            for wa, tag in [(weights_A, 'A')]:
                r = backtest_v36(wa, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=10, base_atr_mult=base_atr,
                                adaptive_stop=True, atr_mult_low=atr_low, atr_mult_high=atr_high)
                if r:
                    r['test'] = f'AS_b{base_atr}l{atr_low}h{atr_high}_{tag}'
                    results.append(r)

    # =====================================================================
    # TEST 3: Per-stock adaptive stop
    # =====================================================================
    print("  Test 3: Per-stock adaptive stop...", flush=True)
    for atr_low, atr_high in [(0.6, 1.5), (0.8, 2.0), (1.0, 2.0)]:
        for base_atr in [0.8, 1.0, 1.2]:
            for wa, tag in [(weights_A, 'A')]:
                r = backtest_v36(wa, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=10, base_atr_mult=base_atr,
                                adaptive_stop=True, atr_mult_low=atr_low, atr_mult_high=atr_high,
                                per_stock_har=True)
                if r:
                    r['test'] = f'PS_b{base_atr}l{atr_low}h{atr_high}_{tag}'
                    results.append(r)

    # =====================================================================
    # TEST 4: Position scaling (no adaptive stop)
    # =====================================================================
    print("  Test 4: Position scaling...", flush=True)
    for scale_f in [0.3, 0.5, 0.7]:
        for atr in [0.8, 1.0, 1.2]:
            for wa, tag in [(weights_A, 'A')]:
                r = backtest_v36(wa, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=10, base_atr_mult=atr,
                                position_scale=True, pos_scale_factor=scale_f)
                if r:
                    r['test'] = f'PS_s{scale_f}_A{atr}_{tag}'
                    results.append(r)

    # =====================================================================
    # TEST 5: Combined: Adaptive stop + Position scale
    # =====================================================================
    print("  Test 5: Combined adaptive stop + position scale...", flush=True)
    for atr_low, atr_high in [(0.6, 1.5), (0.8, 2.0)]:
        for scale_f in [0.5, 0.7]:
            for base_atr in [0.8, 1.0]:
                for wa, tag in [(weights_A, 'A')]:
                    r = backtest_v36(wa, all_factors, NS, ND, dates, C, O, H, L, V,
                                    top_n=1, rebalance_days=10, base_atr_mult=base_atr,
                                    adaptive_stop=True, atr_mult_low=atr_low, atr_mult_high=atr_high,
                                    position_scale=True, pos_scale_factor=scale_f)
                    if r:
                        r['test'] = f'CB_b{base_atr}l{atr_low}h{atr_high}s{scale_f}_{tag}'
                        results.append(r)

    # =====================================================================
    # TEST 6: Top_n=2 with best adaptive configs
    # =====================================================================
    print("  Test 6: Top_n=2 adaptive...", flush=True)
    for atr_low, atr_high in [(0.6, 1.5), (0.8, 2.0)]:
        for base_atr in [0.8, 1.0, 1.2]:
            for wa, tag in [(weights_A, 'A')]:
                r = backtest_v36(wa, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=2, rebalance_days=10, base_atr_mult=base_atr,
                                adaptive_stop=True, atr_mult_low=atr_low, atr_mult_high=atr_high)
                if r:
                    r['test'] = f'T2_AS_b{base_atr}l{atr_low}h{atr_high}_{tag}'
                    results.append(r)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  TOP 40 RESULTS (V36 ADAPTIVE STOP)", flush=True)
    print(f"  {'Test':<40s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*80}", flush=True)
    for r in results[:40]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<40s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    # Best per test group
    groups = {}
    for r in results:
        prefix = r['test'].split('_')[0]
        if prefix not in groups or r['ann'] > groups[prefix]['ann']:
            groups[prefix] = r

    print(f"\n  Best per group:", flush=True)
    for r in sorted(groups.values(), key=lambda x: -x['ann']):
        pos = " ALL+" if all_positive(r) else ""
        print(f"    {r['test']:<40s} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}%{pos}", flush=True)

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

    # V15 baseline comparison
    if results:
        best = results[0]
        print(f"\n  === V36 BEST vs V15 BASELINE ===", flush=True)
        print(f"  V36: {best['test']} = {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)
        print(f"  V15: HAR_RV_T1_A1.0 = +235.6% DD=32.4%", flush=True)
        print(f"  Delta: {best['ann'] - 235.6:+.1f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
