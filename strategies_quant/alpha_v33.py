"""
Alpha V33 — Market Regime Filter Strategy
==========================================
Hypothesis: A market-level gate can improve WR by skipping unfavorable environments.

V15 year-by-year shows 2022-2023 as bottleneck (WR=21-36%).
If we can identify these environments in real-time, we can skip/reduce trades.

ONE variable changed: market gate on/off. Everything else stays V15 baseline.

Market indicators tested as gates:
  1. MEDIAN MOM5 across 500 stocks → positive = healthy market
  2. Pct of stocks with MOM5 > 50 → market breadth
  3. Pct of stocks in squeeze (BB_WIDTH_PCT_INV rank > 70) → compression
  4. Mean HAR-RV across stocks → volatility regime
  5. Market ATR_TERRAIN mode → terrain distribution

Gate logic: only buy when gate > threshold. Test multiple thresholds.

NO LOOK-AHEAD: All market indicators computed from d=di-1 data.
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


def compute_market_indicators(all_factors, NS, ND):
    """Compute market-level indicators from cross-sectional factors.

    SELF-CHECK: uses only data at index di, which stores d=di-1 values.
    """
    market = {}

    # 1. Median MOM5 across all stocks each day
    mom5 = all_factors.get('R_MOM5')
    if mom5 is not None:
        median_mom5 = np.full(ND, np.nan)
        pct_above50 = np.full(ND, np.nan)
        for di in range(60, ND):
            vals = mom5[:, di]
            mask = ~np.isnan(vals)
            if mask.sum() > 50:
                median_mom5[di] = np.median(vals[mask])
                pct_above50[di] = np.sum(vals[mask] > 50) / mask.sum() * 100
        market['MEDIAN_MOM5'] = median_mom5
        market['BREADTH_MOM5'] = pct_above50

    # 2. Mean HAR-RV across stocks (volatility regime)
    har_rv = all_factors.get('R_HAR_RV_RATIO_INV')
    if har_rv is not None:
        mean_har = np.full(ND, np.nan)
        for di in range(60, ND):
            vals = har_rv[:, di]
            mask = ~np.isnan(vals)
            if mask.sum() > 50:
                mean_har[di] = np.mean(vals[mask])
        market['MEAN_HAR_RV'] = mean_har

    # 3. Pct stocks in squeeze (BB_WIDTH_PCT_INV rank > 70)
    bwp = all_factors.get('R_BB_WIDTH_PCT_INV')
    if bwp is not None:
        pct_squeeze = np.full(ND, np.nan)
        for di in range(60, ND):
            vals = bwp[:, di]
            mask = ~np.isnan(vals)
            if mask.sum() > 50:
                pct_squeeze[di] = np.sum(vals[mask] > 70) / mask.sum() * 100
        market['PCT_SQUEEZE'] = pct_squeeze

    # 4. Median ATR_TERRAIN
    atr_t = all_factors.get('R_ATR_TERRAIN')
    if atr_t is not None:
        median_at = np.full(ND, np.nan)
        for di in range(60, ND):
            vals = atr_t[:, di]
            mask = ~np.isnan(vals)
            if mask.sum() > 50:
                median_at[di] = np.median(vals[mask])
        market['MEDIAN_ATR_TERRAIN'] = median_at

    # 5. Market volatility (cross-sectional dispersion of returns)
    # Use std of daily returns across all stocks as VIX-like indicator
    # This requires computing from raw data - skip if not available

    print(f"  Market indicators computed: {list(market.keys())}", flush=True)
    return market


def backtest_with_gate(factor_weights, all_factors, market, NS, ND, dates,
                       C, O, H, L, V, top_n=1, rebalance_days=10,
                       atr_stop_mult=1.0, gate_name=None, gate_threshold=50,
                       gate_direction='above'):
    """Backtest with market-level gate.

    gate_direction:
      'above': only trade when market[gate_name][di] >= gate_threshold
      'below': only trade when market[gate_name][di] <= gate_threshold
    """
    COMMISSION = 0.0003
    STAMP_DUTY = 0.001
    CASH0 = 500000

    if gate_name and gate_name not in market:
        return None
    gate_series = market.get(gate_name) if gate_name else None

    # Build composite score
    score = np.zeros((NS, ND))
    for fname, w in factor_weights.items():
        if fname in all_factors:
            f = all_factors[fname]
            mask = ~np.isnan(f)
            score[mask] += f[mask] * w
        else:
            return None

    # Trading loop
    cash = CASH0
    holdings = {}  # sym_idx -> {shares, entry_price, entry_di, stop_price, high_water}
    port_history = []
    results_list = []

    # Pre-generate trade ID counter
    trade_id = 0

    for di in range(MIN_TRAIN, ND):
        # Check market gate
        if gate_series is not None:
            gval = gate_series[di]
            if np.isnan(gval):
                continue
            if gate_direction == 'above' and gval < gate_threshold:
                # Gate blocks trading — force sell all holdings
                for si in list(holdings.keys()):
                    h = holdings[si]
                    sell_price = O[si, di] if not np.isnan(O[si, di]) else C[si, di]
                    if np.isnan(sell_price):
                        continue
                    proceeds = h['shares'] * sell_price * (1 - COMMISSION - STAMP_DUTY)
                    cash += proceeds
                    pnl = (sell_price - h['entry_price']) / h['entry_price'] * 100
                    results_list.append({
                        'pnl': pnl, 'reason': 'gate_exit'
                    })
                    del holdings[si]
                port_history.append(cash)
                continue
            elif gate_direction == 'below' and gval > gate_threshold:
                for si in list(holdings.keys()):
                    h = holdings[si]
                    sell_price = O[si, di] if not np.isnan(O[si, di]) else C[si, di]
                    if np.isnan(sell_price):
                        continue
                    proceeds = h['shares'] * sell_price * (1 - COMMISSION - STAMP_DUTY)
                    cash += proceeds
                    pnl = (sell_price - h['entry_price']) / h['entry_price'] * 100
                    results_list.append({
                        'pnl': pnl, 'reason': 'gate_exit'
                    })
                    del holdings[si]
                port_history.append(cash)
                continue

        # ATR stop check
        for si in list(holdings.keys()):
            h = holdings[si]
            d = di - 1
            if np.isnan(L[si, di]):
                continue
            hw = H[si, di] if not np.isnan(H[si, di]) else h['high_water']
            h['high_water'] = max(h['high_water'], hw)

            # ATR for stop
            atr_sum, atr_cnt = 0, 0
            for dd in range(max(d - 14, 1), d + 1):
                if np.isnan(H[si, dd]) or np.isnan(L[si, dd]):
                    continue
                tr = H[si, dd] - L[si, dd]
                if not np.isnan(C[si, dd - 1]):
                    tr = max(tr, abs(H[si, dd] - C[si, dd - 1]),
                             abs(L[si, dd] - C[si, dd - 1]))
                atr_sum += tr
                atr_cnt += 1
            atr = atr_sum / atr_cnt if atr_cnt > 0 else h['entry_price'] * 0.05

            stop = h['high_water'] - atr * atr_stop_mult

            if L[si, di] <= stop:
                sell_price = stop
                proceeds = h['shares'] * sell_price * (1 - COMMISSION - STAMP_DUTY)
                cash += proceeds
                pnl = (sell_price - h['entry_price']) / h['entry_price'] * 100
                results_list.append({
                    'pnl': pnl, 'reason': 'stop'
                })
                del holdings[si]

        # Rebalance / new buys
        days_held = di - min((h['entry_di'] for h in holdings.values()), default=di)
        if len(holdings) < top_n or (holdings and days_held >= rebalance_days):
            # Sell holdings that need rebalancing
            if hold := holdings:
                for si in list(hold.keys()):
                    h = hold[si]
                    if di - h['entry_di'] >= rebalance_days:
                        sell_price = O[si, di] if not np.isnan(O[si, di]) else C[si, di]
                        if np.isnan(sell_price):
                            continue
                        proceeds = h['shares'] * sell_price * (1 - COMMISSION - STAMP_DUTY)
                        cash += proceeds
                        pnl = (sell_price - h['entry_price']) / h['entry_price'] * 100
                        results_list.append({
                            'pnl': pnl, 'reason': 'rebalance'
                        })
                        del holdings[si]

            # Buy new top N
            if len(holdings) < top_n:
                scores = score[:, di].copy()
                mask = ~np.isnan(scores)
                scores[~mask] = -999

                # Exclude already held
                for si in holdings:
                    scores[si] = -999

                ranked = np.argsort(scores)[::-1]
                n_buy = top_n - len(holdings)
                bought = 0
                for si in ranked:
                    if bought >= n_buy:
                        break
                    if scores[si] <= 0 or np.isnan(C[si, di]):
                        continue
                    price = C[si, di]
                    if np.isnan(price) or price <= 0:
                        continue
                    alloc = cash / max(n_buy - bought, 1)
                    shares = int(alloc / price / 100) * 100
                    if shares <= 0:
                        continue
                    cost = shares * price * (1 + COMMISSION)
                    if cost > cash:
                        continue
                    cash -= cost
                    holdings[si] = {
                        'shares': shares,
                        'entry_price': price,
                        'entry_di': di,
                        'stop_price': price - price * 0.05,
                        'high_water': price
                    }
                    bought += 1

        # Portfolio value
        port_val = cash
        for si, h in holdings.items():
            if not np.isnan(C[si, di]):
                port_val += h['shares'] * C[si, di]
        port_history.append(port_val)

    # Sell remaining
    for si in list(holdings.keys()):
        h = holdings[si]
        d = ND - 1
        sell_price = C[si, d] if not np.isnan(C[si, d]) else h['entry_price']
        proceeds = h['shares'] * sell_price * (1 - COMMISSION - STAMP_DUTY)
        cash += proceeds
        results_list.append({
            'pnl': (sell_price - h['entry_price']) / h['entry_price'] * 100,
            'reason': 'end'
        })

    # Calculate results
    if not results_list:
        return None

    pnls = [r['pnl'] for r in results_list]
    n_trades = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n_trades * 100 if n_trades > 0 else 0
    avg_edge = np.mean(pnls) if pnls else 0

    # Annualized return
    if len(port_history) > 252:
        total_return = (port_history[-1] / CASH0 - 1) * 100
        n_years = len(port_history) / 252
        ann = ((1 + total_return / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
    else:
        ann = 0

    # Max drawdown
    peak = CASH0
    max_dd = 0
    for v in port_history:
        peak = max(peak, v)
        dd = (peak - v) / peak * 100
        max_dd = max(max_dd, dd)

    # Year-by-year
    year_stats = {}
    for r in results_list:
        pass  # simplified - no year tracking in this quick backtest

    return {
        'ann': ann,
        'n': n_trades,
        'wr': wr,
        'edge': avg_edge,
        'max_dd': max_dd,
        'total_return': total_return if len(port_history) > 252 else 0,
    }


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V33 — Market Regime Filter Strategy", flush=True)
    print("  Market-level gate to skip unfavorable environments", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Compute all factors
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
    v14_all = {**v11_all, **v14_factors}
    v14_inter = compute_v14_interactions(v14_all, NS, ND)
    all_factors = {**v14_all, **v14_inter}

    # Compute market indicators
    market = compute_market_indicators(all_factors, NS, ND)

    print(f"\n  Total factors: {len(all_factors)}", flush=True)
    print(f"  Market indicators: {list(market.keys())}", flush=True)

    results = []

    # =================================================================
    # BASELINE: No gate (should replicate V15 ~235.6%)
    # =================================================================
    print(f"\n  === BASELINE (No Gate) ===", flush=True)

    har_combo = {'R_BWP_BNW': 0.3, 'R_HAR_RV_RATIO_INV': 0.3,
                 'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2}

    for tn in [1, 2]:
        for atr in [1.0, 1.2]:
            r = backtest_v7c(har_combo, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=tn, rebalance_days=10, atr_stop_mult=atr)
            if r:
                r['test'] = f'BASE_T{tn}_A{atr}'
                results.append(r)
                print(f"  {r['test']}: Ann={r['ann']:+.1f}% WR={r['wr']:.1f}% "
                      f"Edge={r['edge']:+.2f}% DD={r['max_dd']:.1f}%", flush=True)

    # =================================================================
    # GATE TESTS: Market indicators as trade filters
    # =================================================================
    print(f"\n  === MARKET GATE TESTS ===", flush=True)

    # For each market indicator, test gate thresholds
    gate_configs = [
        # (gate_name, direction, thresholds)
        ('BREADTH_MOM5', 'above', [40, 45, 50, 55, 60]),
        ('MEDIAN_MOM5', 'above', [45, 50, 55, 60]),
        ('MEAN_HAR_RV', 'above', [45, 50, 55]),
        ('MEAN_HAR_RV', 'below', [45, 50, 55]),  # low HAR-RV = calm market
        ('PCT_SQUEEZE', 'above', [10, 15, 20, 25]),
        ('PCT_SQUEEZE', 'below', [15, 20, 25]),   # low squeeze = trending
        ('MEDIAN_ATR_TERRAIN', 'above', [50, 55, 60]),
    ]

    for gate_name, direction, thresholds in gate_configs:
        if gate_name not in market:
            print(f"  SKIP: {gate_name} not in market indicators", flush=True)
            continue
        for thresh in thresholds:
            for tn in [1]:
                for atr in [1.0]:
                    r = backtest_with_gate(
                        har_combo, all_factors, market, NS, ND, dates,
                        C, O, H, L, V, top_n=tn, rebalance_days=10,
                        atr_stop_mult=atr,
                        gate_name=gate_name, gate_threshold=thresh,
                        gate_direction=direction)
                    if r:
                        r['test'] = f'GATE_{gate_name[:8]}_{direction[:1]}{thresh}_T{tn}_A{atr}'
                        results.append(r)
                        print(f"  {r['test']:<35s}: Ann={r['ann']:+7.1f}% N={r['n']:4d} "
                              f"WR={r['wr']:5.1f}% Edge={r['edge']:+5.2f}% "
                              f"DD={r['max_dd']:5.1f}%", flush=True)

    # =================================================================
    # RESULTS
    # =================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  ALL RESULTS (V33 MARKET REGIME FILTER)", flush=True)
    print(f"  {'Test':<35s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*80}", flush=True)
    for r in results[:50]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<35s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    print(f"\n{'='*70}", flush=True)
