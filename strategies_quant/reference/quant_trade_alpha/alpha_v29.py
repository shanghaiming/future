"""
Alpha V29 — Conditional Gate Strategy (Non-Linear Factor Combination)
=====================================================================
KEY INSIGHT from V15-V26 results:
  - Linear rank() + weighted sum has a ceiling (~235% from HAR-RV + BWP_BNW)
  - All advanced algorithm factors (HMM/DMD/FFT/Markov) are correlated with existing factors
  - Real alpha comes from CONDITIONAL GATING: only buy when multiple independent conditions align

V29 uses GATE logic instead of linear combination:
  Gate 1 (Volatility): BB_WIDTH_PCT_INV > 70 → squeeze confirmed
  Gate 2 (Trend): MOM5 rank > 50 → upward momentum
  Gate 3 (Structure): TENSION rank > 50 → price above key levels
  Gate 4 (Vol Regime): HAR-RV ratio favorable → not expanding

  SCORE = SUM(rank × weight) ONLY for stocks passing ALL gates
  → This is non-linear: a stock must pass ALL gates to be eligible
  → Stocks failing any gate get score = 0

Also tests:
  - Multi-gate AND (all 4 gates must pass)
  - Multi-gate OR (any 2 of 4 gates)
  - Weighted gate (soft gate: score × min(gate_scores))
  - Momentum exit: sell when MOM5 rank drops below threshold
  - Adaptive rebalance: only rebalance when top stocks change

LOOK-AHEAD SELF-CHECK:
  [x] All factors use ONLY data up to d=di-1
  [x] Results stored at index di
  [x] No same-day data used
  [x] Gate checks use only factor values at di (computed from di-1)
  [x] Momentum exit uses MOM5 at di (computed from di-1)
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


def backtest_v29(factor_weights, factors, NS, ND, dates, C, O, H, L, V,
                 top_n=1, rebalance_days=10, atr_stop_mult=1.5,
                 gates=None, gate_mode='AND',
                 momentum_exit=False, mom_exit_threshold=30.0,
                 adaptive_rebalance=False, adaptive_threshold=0.5):
    """V29 Backtest — Conditional Gate + Momentum Exit.

    gates: dict of {factor_name: threshold} — stock must have factor >= threshold
    gate_mode: 'AND' (all gates), 'OR' (any gate), 'SOFT' (multiply by min gate score)
    momentum_exit: sell when MOM5 rank drops below threshold
    adaptive_rebalance: only rebalance when top stocks change by > threshold

    LOOK-AHEAD SELF-CHECK:
      [x] Factor values at di use only data up to di-1
      [x] Trades at O[si, di] (open price)
      [x] ATR stop: L[si,di] check, stop price sell
      [x] Gates use only factor values at di
      [x] Momentum exit uses only factor value at di
    """
    factor_names = list(factor_weights.keys())
    weights = np.array([factor_weights[f] for f in factor_names])

    cash = float(CASH0)
    holdings = []
    trades = []
    last_rebalance = -999
    year_stats = {}
    prev_top = set()

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

            # === Momentum Exit ===
            if momentum_exit and pos in holdings:
                if 'R_MOM5' in factors:
                    mom5_val = factors['R_MOM5'][si, di]
                    if not np.isnan(mom5_val) and mom5_val < mom_exit_threshold:
                        sp = O[si, di] if not np.isnan(O[si, di]) else C[si, di]
                        if not np.isnan(sp) and sp > 0:
                            pnl = (sp - pos['entry']) / pos['entry'] * 100
                            cash += pos['shares'] * sp * (1 - COMMISSION - STAMP_DUTY)
                            trades.append({'pnl': pnl, 'days': (dates[di] - pos['ed']).days,
                                           'di': di, 'reason': 'mom_exit', 'year': year})
                            holdings.remove(pos)

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

        # === Compute composite score ===
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

        # === Apply GATES ===
        if gates:
            gate_scores = np.ones(NS)
            n_gates_passed = np.zeros(NS, dtype=int)

            for gate_name, gate_threshold in gates.items():
                if gate_name not in factors:
                    continue
                gate_vals = factors[gate_name][:, di]
                for si in range(NS):
                    if np.isnan(gate_vals[si]):
                        gate_scores[si] = 0.0
                    elif gate_vals[si] >= gate_threshold:
                        n_gates_passed[si] += 1
                        # Soft score: how far above threshold (0-1 range)
                        gate_scores[si] = min(gate_scores[si],
                                              gate_vals[si] / 100.0)
                    else:
                        # Failed this gate
                        if gate_mode == 'AND':
                            gate_scores[si] = 0.0
                        elif gate_mode == 'SOFT':
                            gate_scores[si] *= (gate_vals[si] / gate_threshold)

            if gate_mode == 'AND':
                # Zero out stocks that failed ANY gate
                composite *= gate_scores
            elif gate_mode == 'OR':
                # Zero out stocks that failed ALL gates
                for si in range(NS):
                    if n_gates_passed[si] == 0:
                        composite[si] = 0.0
            elif gate_mode == 'SOFT':
                composite *= gate_scores

        composite[~mask] = -9999

        # === Adaptive Rebalance ===
        top_indices = set(np.argsort(-composite)[:top_n])
        if adaptive_rebalance and prev_top:
            # Only rebalance if top stocks changed significantly
            overlap = len(top_indices & prev_top) / max(len(prev_top), 1)
            if overlap >= adaptive_threshold:
                # Not enough change, skip this rebalance
                continue

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
            alloc = cash / n_to_buy * 0.98  # 2% buffer
            for si in to_buy:
                price = O[si, di]
                if np.isnan(price) or price <= 0:
                    price = C[si, di]
                if np.isnan(price) or price <= 0:
                    continue
                shares = int(alloc / price / 100) * 100
                if shares > 0 and cash >= shares * price * (1 + COMMISSION + STAMP_DUTY):
                    cost = shares * price * (1 + COMMISSION + STAMP_DUTY)
                    cash -= cost
                    holdings.append({
                        'si': si, 'entry': price, 'shares': shares,
                        'ed': dates[di], 'hw': price,
                        'buy_di': di
                    })

        last_rebalance = di
        prev_top = top_indices

        # Year-end tracking
        if year not in year_stats:
            year_stats[year] = {'trades': 0, 'wins': 0, 'total_pnl': 0.0}

    # Force-sell remaining
    for pos in list(holdings):
        sp = C[pos['si'], ND - 1]
        if not np.isnan(sp) and sp > 0:
            pnl = (sp - pos['entry']) / pos['entry'] * 100
            cash += pos['shares'] * sp * (1 - COMMISSION - STAMP_DUTY)
            trades.append({'pnl': pnl, 'days': (dates[ND - 1] - pos['ed']).days,
                           'di': ND - 1, 'reason': 'end', 'year': dates[ND - 1].year})

    if not trades:
        return None

    # Stats
    for t in trades:
        y = t['year']
        if y not in year_stats:
            year_stats[y] = {'trades': 0, 'wins': 0, 'total_pnl': 0.0}
        year_stats[y]['trades'] += 1
        if t['pnl'] > 0:
            year_stats[y]['wins'] += 1
        year_stats[y]['total_pnl'] += t['pnl']

    n = len(trades)
    wins = sum(1 for t in trades if t['pnl'] > 0)
    total_pnl = sum(t['pnl'] for t in trades)
    avg_pnl = total_pnl / n if n > 0 else 0
    wr = wins / n * 100 if n > 0 else 0
    edge = avg_pnl - COMMISSION * 200  # rough

    # Max drawdown
    equity = [CASH0]
    for t in sorted(trades, key=lambda x: x['di']):
        equity.append(equity[-1] * (1 + t['pnl'] / 100))
    peak = equity[0]
    max_dd = 0
    for e in equity:
        peak = max(peak, e)
        dd = (peak - e) / peak * 100
        max_dd = max(max_dd, dd)

    years = (dates[ND - 1] - dates[MIN_TRAIN]).days / 365.25
    ann = ((cash / CASH0) ** (1 / years) - 1) * 100 if years > 0 and cash > 0 else -100

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        r = t['reason']
        if r not in exit_reasons:
            exit_reasons[r] = {'count': 0, 'pnl': 0.0}
        exit_reasons[r]['count'] += 1
        exit_reasons[r]['pnl'] += t['pnl']

    return {
        'ann': ann, 'n': n, 'wr': wr, 'edge': edge,
        'max_dd': max_dd, 'year_stats': year_stats,
        'total_pnl': total_pnl, 'cash': cash,
        'exit_reasons': exit_reasons,
    }


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V29 — Conditional Gate Strategy", flush=True)
    print("  Non-linear factor combination via entry/exit gating", flush=True)
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
    v14_inter = compute_v14_interactions({**v11_all, **v14_factors}, NS, ND)
    all_factors = {**v11_all, **v14_factors, **v14_inter}

    print(f"\n  Total factors: {len(all_factors)}", flush=True)

    results = []

    # =====================================================================
    # BASELINE: V7c standard (no gates)
    # =====================================================================
    print(f"\n  === BASELINE ===", flush=True)
    bwp = {'R_BWP_BNW': 0.3, 'R_TENSION': 0.3,
            'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2}
    for top_n in [1, 2]:
        for atr in [1.0, 1.2, 1.5]:
            r = backtest_v7c(bwp, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=top_n, rebalance_days=10, atr_stop_mult=atr)
            if r:
                r['test'] = f'Baseline_T{top_n}_A{atr}'
                results.append(r)
    print(f"  Baseline done", flush=True)

    # =====================================================================
    # V15 BEST: HAR-RV + BWP_BNW (proven +235.6%)
    # =====================================================================
    print(f"\n  === V15 BEST REPRODUCTION ===", flush=True)
    har_weights = {
        'R_HAR_RV_RATIO_INV': 0.3, 'R_BWP_BNW': 0.3,
        'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2
    }
    for top_n in [1]:
        for atr in [1.0, 1.2]:
            r = backtest_v7c(har_weights, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=top_n, rebalance_days=10, atr_stop_mult=atr)
            if r:
                r['test'] = f'HAR_RV_T{top_n}_A{atr}'
                results.append(r)
    print(f"  V15 reproduction done", flush=True)

    # =====================================================================
    # TEST 1: AND Gate — squeeze + momentum + tension
    # =====================================================================
    print(f"\n  === TEST 1: AND GATE ===", flush=True)
    gate_configs = [
        ('SqueezeGate_60', {'R_BB_WIDTH_PCT_INV': 60}),
        ('SqueezeGate_70', {'R_BB_WIDTH_PCT_INV': 70}),
        ('SqueezeGate_80', {'R_BB_WIDTH_PCT_INV': 80}),
        ('MomGate_40', {'R_MOM5': 40}),
        ('MomGate_50', {'R_MOM5': 50}),
        ('MomGate_60', {'R_MOM5': 60}),
        ('DoubleAND_60', {'R_BB_WIDTH_PCT_INV': 60, 'R_MOM5': 50}),
        ('DoubleAND_70', {'R_BB_WIDTH_PCT_INV': 70, 'R_MOM5': 60}),
        ('TripleAND', {'R_BB_WIDTH_PCT_INV': 60, 'R_MOM5': 40, 'R_TENSION': 50}),
        ('QuadAND', {'R_BB_WIDTH_PCT_INV': 60, 'R_MOM5': 40,
                     'R_TENSION': 50, 'R_HAR_RV_RATIO_INV': 50}),
    ]

    for gname, gates in gate_configs:
        for top_n in [1]:
            for atr in [1.0, 1.2]:
                r = backtest_v29(bwp, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, rebalance_days=10, atr_stop_mult=atr,
                                gates=gates, gate_mode='AND')
                if r:
                    r['test'] = f'{gname}_T{top_n}_A{atr}'
                    results.append(r)
        print(f"  {gname} done", flush=True)

    # =====================================================================
    # TEST 2: OR Gate — any 2 of 4 conditions
    # =====================================================================
    print(f"\n  === TEST 2: OR GATE ===", flush=True)
    or_gates = {
        'OR_Quad': {'R_BB_WIDTH_PCT_INV': 60, 'R_MOM5': 50,
                    'R_TENSION': 50, 'R_HAR_RV_RATIO_INV': 50},
    }
    for gname, gates in or_gates.items():
        for top_n in [1]:
            for atr in [1.0, 1.2]:
                r = backtest_v29(bwp, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, rebalance_days=10, atr_stop_mult=atr,
                                gates=gates, gate_mode='OR')
                if r:
                    r['test'] = f'{gname}_T{top_n}_A{atr}'
                    results.append(r)
        print(f"  {gname} done", flush=True)

    # =====================================================================
    # TEST 3: SOFT Gate — multiply by gate confidence
    # =====================================================================
    print(f"\n  === TEST 3: SOFT GATE ===", flush=True)
    soft_gates = [
        ('SoftDouble', {'R_BB_WIDTH_PCT_INV': 50, 'R_MOM5': 40}),
        ('SoftTriple', {'R_BB_WIDTH_PCT_INV': 50, 'R_MOM5': 40, 'R_TENSION': 40}),
        ('SoftQuad', {'R_BB_WIDTH_PCT_INV': 50, 'R_MOM5': 40,
                      'R_TENSION': 40, 'R_HAR_RV_RATIO_INV': 40}),
    ]
    for gname, gates in soft_gates:
        for top_n in [1]:
            for atr in [1.0, 1.2]:
                r = backtest_v29(bwp, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, rebalance_days=10, atr_stop_mult=atr,
                                gates=gates, gate_mode='SOFT')
                if r:
                    r['test'] = f'{gname}_T{top_n}_A{atr}'
                    results.append(r)
        print(f"  {gname} done", flush=True)

    # =====================================================================
    # TEST 4: Momentum Exit — sell when MOM5 drops
    # =====================================================================
    print(f"\n  === TEST 4: MOMENTUM EXIT ===", flush=True)
    for mom_th in [20, 30, 40, 50]:
        for atr in [1.0, 1.2]:
            r = backtest_v29(bwp, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=10, atr_stop_mult=atr,
                            momentum_exit=True, mom_exit_threshold=mom_th)
            if r:
                r['test'] = f'MomExit{mom_th}_T1_A{atr}'
                results.append(r)
        print(f"  MomExit{mom_th} done", flush=True)

    # =====================================================================
    # TEST 5: AND Gate + Momentum Exit (best combo)
    # =====================================================================
    print(f"\n  === TEST 5: GATE + MOMENTUM EXIT ===", flush=True)
    best_gates = {'R_BB_WIDTH_PCT_INV': 60, 'R_MOM5': 40}
    for mom_th in [20, 30, 40]:
        for atr in [1.0, 1.2]:
            r = backtest_v29(bwp, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=10, atr_stop_mult=atr,
                            gates=best_gates, gate_mode='AND',
                            momentum_exit=True, mom_exit_threshold=mom_th)
            if r:
                r['test'] = f'GateMom{mom_th}_T1_A{atr}'
                results.append(r)
        print(f"  GateMom{mom_th} done", flush=True)

    # =====================================================================
    # TEST 6: HAR-RV factor + AND Gate (the winner combo)
    # =====================================================================
    print(f"\n  === TEST 6: HAR-RV + AND GATE ===", flush=True)
    har_weights_gate = {
        'R_HAR_RV_RATIO_INV': 0.25, 'R_BWP_BNW': 0.25,
        'R_TENSION': 0.25, 'R_R_SQUARED': 0.25
    }
    har_gates = {'R_BB_WIDTH_PCT_INV': 60, 'R_MOM5': 40}
    for top_n in [1]:
        for atr in [1.0, 1.2]:
            r = backtest_v29(har_weights_gate, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=top_n, rebalance_days=10, atr_stop_mult=atr,
                            gates=har_gates, gate_mode='AND')
            if r:
                r['test'] = f'HAR_Gate_T{top_n}_A{atr}'
                results.append(r)
            # With momentum exit
            r = backtest_v29(har_weights_gate, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=top_n, rebalance_days=10, atr_stop_mult=atr,
                            gates=har_gates, gate_mode='AND',
                            momentum_exit=True, mom_exit_threshold=30)
            if r:
                r['test'] = f'HAR_GateMom_T{top_n}_A{atr}'
                results.append(r)
    print(f"  HAR+Gate done", flush=True)

    # =====================================================================
    # TEST 7: Adaptive Rebalance + Gate
    # =====================================================================
    print(f"\n  === TEST 7: ADAPTIVE REBALANCE + GATE ===", flush=True)
    for adapt_th in [0.3, 0.5]:
        r = backtest_v29(bwp, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=1, rebalance_days=10, atr_stop_mult=1.0,
                        gates={'R_BB_WIDTH_PCT_INV': 60}, gate_mode='AND',
                        adaptive_rebalance=True, adaptive_threshold=adapt_th)
        if r:
            r['test'] = f'Adapt{adapt_th}_T1_A1.0'
            results.append(r)
    print(f"  Adaptive done", flush=True)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  TOP 50 RESULTS (V29 CONDITIONAL GATE)", flush=True)
    print(f"  {'Test':<30s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*80}", flush=True)
    for r in results[:50]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<30s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    # Top 5 year-by-year
    for i, r in enumerate(results[:5]):
        print(f"\n  Year-by-year #{i+1}: {r['test']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        # Exit reasons
        if 'exit_reasons' in r:
            for reason, stats in r['exit_reasons'].items():
                avg = stats['pnl'] / max(stats['count'], 1)
                print(f"    Exit {reason}: {stats['count']}x, avg={avg:+.1f}%", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    # Best per strategy group
    groups = {}
    for r in results:
        # Extract group name (before _T)
        prefix = r['test'].rsplit('_T', 1)[0] if '_T' in r['test'] else r['test']
        if prefix not in groups or r['ann'] > groups[prefix]['ann']:
            groups[prefix] = r
    print(f"\n  Best per group:", flush=True)
    for r in sorted(groups.values(), key=lambda x: -x['ann']):
        pos = " ALL+" if all_positive(r) else ""
        print(f"    {r['test']:<30s} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}%{pos}", flush=True)

    print(f"\n{'='*70}", flush=True)
