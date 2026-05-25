"""
Alpha V38 — Regime-Adaptive Factor Weights (Isolated Test)
==========================================================
V14 tried regime-adaptive weights but changed 5 things → failed.
V29 tried gates → failed.
V38 isolates ONLY regime-adaptive factor weights, using ATR_TERRAIN
as the regime signal (proven independent by V21 Kendall τ analysis).

Key idea:
  ATR_TERRAIN = 100 (Squeeze): Market compressing → use momentum weights
    - BWP_BNW × 0.4 + MOM5 × 0.3 + SMA_DEV × 0.3
  ATR_TERRAIN = 75 (Fading): Compression easing → balanced weights
    - BWP_BNW × 0.3 + HAR_RV × 0.3 + R_SQUARED × 0.2 + SMA_DEV × 0.2
  ATR_TERRAIN = 50 (Normal): Standard regime → V15 baseline weights
    - BWP_BNW × 0.3 + HAR_RV × 0.3 + R_SQUARED × 0.2 + SMA_DEV × 0.2
  ATR_TERRAIN = 25 (Expansion): High vol → defensive weights
    - HAR_RV × 0.4 + R_SQUARED × 0.3 + SMA_DEV × 0.3

This is ONLY factor weight changes per regime, nothing else.

Also test:
  - Per-stock regime vs market-median regime
  - Different weight configurations per regime
  - Simple 2-regime (trend/mean-rev) split

NO LOOK-AHEAD: ATR_TERRAIN uses di-1 data, weights applied at di.
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


def backtest_v38(regime_configs, factors, NS, ND, dates, C, O, H, L, V,
                 regime_factor='R_ATR_TERRAIN', regime_mode='market',
                 top_n=1, rebalance_days=10, atr_stop_mult=1.0,
                 default_weights=None):
    """V38 backtest with regime-adaptive factor weights.

    regime_configs: dict mapping regime_state → factor_weights dict
      - Keys are regime names or numeric thresholds
      - Values are factor weight dicts like {'R_BWP_BNW': 0.3, ...}
    regime_factor: which factor to use for regime detection
    regime_mode: 'market' (median), 'stock' (per-stock), 'binary' (above/below median)
    default_weights: fallback if no regime matches

    NO LOOK-AHEAD: regime_factor at di uses di-1 data.
    """
    factor_names = set()
    for cfg in regime_configs.values():
        factor_names.update(cfg.keys())
    if default_weights:
        factor_names.update(default_weights.keys())

    cash = float(CASH0)
    holdings = []
    trades = []
    last_rebalance = -999
    year_stats = {}
    regime_counts = {}

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

        # === Regime detection ===
        # Compute market-wide regime from regime_factor
        market_regime = 'normal'
        if regime_factor in factors:
            rf = factors[regime_factor][:, di]
            valid_rf = rf[~np.isnan(rf)]
            if len(valid_rf) > 50:
                med = np.median(valid_rf)
                # ATR_TERRAIN: 100=Squeeze, 75=Fading, 50=Normal, 25=Expansion
                if med >= 87.5:
                    market_regime = 'squeeze'
                elif med >= 62.5:
                    market_regime = 'fading'
                elif med >= 37.5:
                    market_regime = 'normal'
                else:
                    market_regime = 'expansion'

        # Select weights based on regime
        if regime_mode == 'market':
            active_weights = regime_configs.get(market_regime, default_weights or regime_configs.get('normal', {}))
        elif regime_mode == 'binary':
            # Simple 2-regime: squeeze+fading vs normal+expansion
            if market_regime in ('squeeze', 'fading'):
                active_weights = regime_configs.get('trend', default_weights or regime_configs.get('normal', {}))
            else:
                active_weights = regime_configs.get('defensive', default_weights or regime_configs.get('normal', {}))
        elif regime_mode == 'stock':
            # Per-stock: determine regime for each stock individually
            # Handled below in composite scoring
            active_weights = None
        else:
            active_weights = default_weights or regime_configs.get('normal', {})

        regime_counts[market_regime] = regime_counts.get(market_regime, 0) + 1

        # === Composite score ===
        if regime_mode == 'stock' and regime_factor in factors:
            # Per-stock regime: different weights per stock
            composite = np.zeros(NS)
            count = np.zeros(NS)
            rf_vals = factors[regime_factor][:, di]
            for si in range(NS):
                if np.isnan(rf_vals[si]):
                    continue
                stock_regime = 'normal'
                rv = rf_vals[si]
                if rv >= 87.5:
                    stock_regime = 'squeeze'
                elif rv >= 62.5:
                    stock_regime = 'fading'
                elif rv >= 37.5:
                    stock_regime = 'normal'
                else:
                    stock_regime = 'expansion'
                w = regime_configs.get(stock_regime, default_weights or regime_configs.get('normal', {}))
                for fname, wt in w.items():
                    if fname in factors:
                        v = factors[fname][si, di]
                        if not np.isnan(v):
                            composite[si] += wt * v
                            count[si] += abs(wt)
        else:
            # Market-wide: same weights for all stocks
            w = active_weights or {}
            composite = np.zeros(NS)
            count = np.zeros(NS)
            for fname, wt in w.items():
                if fname not in factors:
                    continue
                arr = factors[fname]
                vals = arr[:, di]
                valid = ~np.isnan(vals)
                if valid.sum() < 50:
                    continue
                composite[valid] += wt * vals[valid]
                count[valid] += abs(wt)

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
        'regime_counts': regime_counts,
    }


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V38 — Regime-Adaptive Factor Weights", flush=True)
    print("  ATR_TERRAIN regime signal, isolated single-variable test", flush=True)
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

    # V15 baseline weights
    v15_weights = {'R_BWP_BNW': 0.3, 'R_HAR_RV_RATIO_INV': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2}

    results = []

    # =====================================================================
    # TEST 1: Baseline (uniform weights, no regime adaptation)
    # =====================================================================
    print("\n  Test 1: Baseline...", flush=True)
    for atr in [0.8, 1.0, 1.2]:
        r = backtest_v38({'normal': v15_weights}, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=1, rebalance_days=10, atr_stop_mult=atr,
                        default_weights=v15_weights)
        if r:
            r['test'] = f'BL_A{atr}'
            results.append(r)

    # =====================================================================
    # TEST 2: 4-regime market-wide adaptation
    # =====================================================================
    print("  Test 2: 4-regime market...", flush=True)
    configs_4regime = {
        'squeeze': {'R_BWP_BNW': 0.4, 'R_MOM5': 0.3, 'R_SMA_DEV': 0.3},
        'fading': {'R_BWP_BNW': 0.3, 'R_HAR_RV_RATIO_INV': 0.3, 'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        'normal': v15_weights,
        'expansion': {'R_HAR_RV_RATIO_INV': 0.4, 'R_R_SQUARED': 0.3, 'R_SMA_DEV': 0.3},
    }
    for atr in [0.8, 1.0, 1.2]:
        r = backtest_v38(configs_4regime, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=1, rebalance_days=10, atr_stop_mult=atr,
                        regime_mode='market', default_weights=v15_weights)
        if r:
            r['test'] = f'4R_mkt_A{atr}'
            results.append(r)

    # =====================================================================
    # TEST 3: 2-regime binary (trend vs defensive)
    # =====================================================================
    print("  Test 3: 2-regime binary...", flush=True)
    configs_2regime = {
        'trend': {'R_BWP_BNW': 0.4, 'R_MOM5': 0.3, 'R_R_SQUARED': 0.3},
        'defensive': {'R_HAR_RV_RATIO_INV': 0.4, 'R_R_SQUARED': 0.3, 'R_SMA_DEV': 0.3},
        'normal': v15_weights,
    }
    for atr in [0.8, 1.0, 1.2]:
        r = backtest_v38(configs_2regime, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=1, rebalance_days=10, atr_stop_mult=atr,
                        regime_mode='binary', default_weights=v15_weights)
        if r:
            r['test'] = f'2R_bin_A{atr}'
            results.append(r)

    # =====================================================================
    # TEST 4: Per-stock regime adaptation
    # =====================================================================
    print("  Test 4: Per-stock regime...", flush=True)
    for atr in [0.8, 1.0, 1.2]:
        r = backtest_v38(configs_4regime, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=1, rebalance_days=10, atr_stop_mult=atr,
                        regime_mode='stock', default_weights=v15_weights)
        if r:
            r['test'] = f'4R_stk_A{atr}'
            results.append(r)

    # =====================================================================
    # TEST 5: Different weight configs per regime
    # =====================================================================
    print("  Test 5: Alternative weight configs...", flush=True)
    # Config A: Momentum-heavy in squeeze, volatility-heavy in expansion
    configs_A = {
        'squeeze': {'R_MOM5': 0.4, 'R_REL_STR': 0.3, 'R_BWP_BNW': 0.3},
        'fading': {'R_BWP_BNW': 0.35, 'R_HAR_RV_RATIO_INV': 0.35, 'R_SMA_DEV': 0.3},
        'normal': v15_weights,
        'expansion': {'R_HAR_RV_RATIO_INV': 0.5, 'R_R_SQUARED': 0.3, 'R_SMA_DEV': 0.2},
    }
    # Config B: Squeeze focus on BWP_BNW, Expansion defensive
    configs_B = {
        'squeeze': {'R_BWP_BNW': 0.5, 'R_HAR_RV_RATIO_INV': 0.3, 'R_R_SQUARED': 0.2},
        'fading': v15_weights,
        'normal': v15_weights,
        'expansion': {'R_HAR_RV_RATIO_INV': 0.5, 'R_KER': 0.3, 'R_R_SQUARED': 0.2},
    }
    # Config C: Use LOG_PRESSURE in expansion (independent dimension)
    configs_C = {
        'squeeze': {'R_BWP_BNW': 0.4, 'R_MOM5': 0.3, 'R_SMA_DEV': 0.3},
        'fading': {'R_BWP_BNW': 0.3, 'R_HAR_RV_RATIO_INV': 0.3, 'R_LOG_PRESSURE': 0.2, 'R_R_SQUARED': 0.2},
        'normal': v15_weights,
        'expansion': {'R_HAR_RV_RATIO_INV': 0.3, 'R_LOG_PRESSURE': 0.3, 'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
    }
    for cfg_name, cfg in [('A', configs_A), ('B', configs_B), ('C', configs_C)]:
        for atr in [0.8, 1.0, 1.2]:
            r = backtest_v38(cfg, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=10, atr_stop_mult=atr,
                            regime_mode='market', default_weights=v15_weights)
            if r:
                r['test'] = f'C{cfg_name}_A{atr}'
                results.append(r)

    # =====================================================================
    # TEST 6: Top_n=2 with regime adaptation
    # =====================================================================
    print("  Test 6: T2 regime...", flush=True)
    for cfg_name, cfg in [('4R', configs_4regime), ('C', configs_C)]:
        for atr in [0.8, 1.0, 1.2]:
            r = backtest_v38(cfg, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=2, rebalance_days=10, atr_stop_mult=atr,
                            regime_mode='market', default_weights=v15_weights)
            if r:
                r['test'] = f'T2_{cfg_name}_A{atr}'
                results.append(r)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  ALL RESULTS (V38 REGIME-ADAPTIVE WEIGHTS)", flush=True)
    print(f"  {'Test':<25s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*70}", flush=True)
    for r in results:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<25s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    # Top 5 year-by-year
    for i, r in enumerate(results[:5]):
        print(f"\n  Year-by-year #{i+1}: {r['test']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        if 'regime_counts' in r:
            print(f"    Regime dist: {r['regime_counts']}", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    if results:
        best = results[0]
        print(f"\n  === V38 BEST vs V15 BASELINE ===", flush=True)
        print(f"  V38: {best['test']} = {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)
        print(f"  V15: HAR_RV_T1_A1.0 = +235.6% DD=32.4%", flush=True)
        print(f"  Delta: {best['ann'] - 235.6:+.1f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
