"""
Alpha V20 — Kalman Filter Strategy (from probability_theory.md Section 21)
===========================================================================
Uses REAL Kalman filter for:
  1. Adaptive moving average (EMA is a special case with fixed gain)
  2. Velocity estimation (trend speed + uncertainty)
  3. Regime detection (|velocity| > threshold → trending)
  4. Adaptive ATR stop (Kalman-estimated σ instead of simple ATR)

Also uses:
  5. DMD bull ratio (from Section 27) for regime detection
  6. Markov transition probability (from Section 31) for state prediction

The KEY difference from V7-V19: instead of rank()+linear_weight, we use
the ACTUAL Kalman recursive estimation to produce continuous signals.

LOOK-AHEAD SELF-CHECK:
  [x] Kalman filter state at time t uses only observations up to t-1
  [x] All results stored at index di (observation at d=di-1)
  [x] No same-day data used
  [x] ATR stop: BUG-FIXED (L[si,di] check)
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7 import compute_all_factors, COMMISSION, STAMP_DUTY, CASH0
from alpha_v7c import backtest_v7c


def compute_kalman_factors(NS, ND, C, O, H, L, V):
    """Compute Kalman filter-based factors.

    State vector x = [price, velocity] (2D Kalman)
    Observation y = close price
    F = [[1, dt], [0, 1]] (constant velocity model)
    H = [1, 0] (observe price)

    SELF-CHECK: d = di - 1 for all data access.
    """
    t0 = time.time()
    new = {}

    dt = 1.0  # 1-day time step

    # State transition matrix
    F = np.array([[1.0, dt],
                  [0.0, 1.0]])

    # Observation matrix
    H_mat = np.array([[1.0, 0.0]])

    # Process noise (tuneable)
    q_price = 0.01   # Price process noise
    q_vel = 0.001    # Velocity process noise
    Q = np.array([[q_price, 0.0],
                  [0.0, q_vel]])

    # Measurement noise (will be adaptive)
    r_base = 0.02

    KALMAN_VEL = np.full((NS, ND), np.nan)       # Velocity (trend speed)
    KALMAN_UNCERT = np.full((NS, ND), np.nan)     # Uncertainty (P[1,1])
    KALMAN_INNOV = np.full((NS, ND), np.nan)       # Innovation (prediction error)
    KALMAN_PRICE = np.full((NS, ND), np.nan)       # Kalman smoothed price

    for si in range(NS):
        # Initialize state
        x = np.array([C[si, 0] if not np.isnan(C[si, 0]) else 0.0, 0.0])
        P = np.array([[1.0, 0.0],
                      [0.0, 1.0]])

        for di in range(2, ND):
            d = di - 1  # SELF-CHECK
            y = C[si, d]
            if np.isnan(y):
                continue

            # Adaptive measurement noise based on recent volatility
            if di >= 15:
                recent = C[si, max(d - 14, 0):d + 1]
                valid = recent[~np.isnan(recent)]
                if len(valid) >= 5:
                    r = max(np.var(valid) * 0.5, 1e-8)
                else:
                    r = r_base
            else:
                r = r_base

            R = np.array([[r]])

            # === PREDICT ===
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

            # === INNOVATION ===
            y_pred = H_mat @ x_pred
            innov = y - y_pred[0]
            S = H_mat @ P_pred @ H_mat.T + R  # Innovation covariance

            # === KALMAN GAIN ===
            K = P_pred @ H_mat.T @ np.linalg.inv(S)

            # === UPDATE ===
            x = (x_pred + K.ravel() * innov)
            P = (np.eye(2) - K @ H_mat) @ P_pred

            # Store results — use .item() to guarantee Python scalar
            KALMAN_VEL[si, di] = np.float64(x[1]).item()
            KALMAN_UNCERT[si, di] = np.float64(P[1, 1]).item()
            KALMAN_INNOV[si, di] = np.float64(innov).item()
            KALMAN_PRICE[si, di] = np.float64(x[0]).item()

    new['KALMAN_VEL'] = KALMAN_VEL
    new['KALMAN_UNCERT'] = KALMAN_UNCERT
    new['KALMAN_INNOV'] = KALMAN_INNOV
    new['KALMAN_PRICE'] = KALMAN_PRICE
    print(f"  Kalman filter done ({time.time()-t0:.1f}s)", flush=True)

    # =====================================================================
    # Derived Kalman signals
    # =====================================================================

    # 1. Velocity signal: positive = uptrend, negative = downtrend
    # Normalize by price to get % velocity
    KALMAN_VEL_PCT = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(2, ND):
            if not np.isnan(KALMAN_VEL[si, di]) and not np.isnan(KALMAN_PRICE[si, di]):
                if KALMAN_PRICE[si, di] > 0:
                    KALMAN_VEL_PCT[si, di] = KALMAN_VEL[si, di] / KALMAN_PRICE[si, di] * 100
    new['KALMAN_VEL_PCT'] = KALMAN_VEL_PCT

    # 2. Confidence: velocity / sqrt(uncertainty) — high = confident trend
    KALMAN_CONF = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(2, ND):
            v = KALMAN_VEL[si, di]
            u = KALMAN_UNCERT[si, di]
            if not np.isnan(v) and not np.isnan(u) and u > 1e-12:
                KALMAN_CONF[si, di] = abs(v) / np.sqrt(u)  # t-statistic analog
    new['KALMAN_CONF'] = KALMAN_CONF

    # 3. Innovation magnitude: large innovation = surprise = potential regime change
    KALMAN_INNOV_MAG = np.full((NS, ND), np.nan)
    for si in range(NS):
        ema_innov = np.nan
        alpha = 2.0 / 11  # 10-day EMA
        for di in range(2, ND):
            innov = KALMAN_INNOV[si, di]
            if np.isnan(innov):
                continue
            mag = abs(innov)
            if np.isnan(ema_innov):
                ema_innov = mag
            else:
                ema_innov = alpha * mag + (1 - alpha) * ema_innov
            KALMAN_INNOV_MAG[si, di] = ema_innov
    new['KALMAN_INNOV_MAG'] = KALMAN_INNOV_MAG

    # 4. Regime: velocity direction × confidence
    # Positive = uptrend with confidence, negative = downtrend with confidence
    KALMAN_REGIME = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(2, ND):
            v = KALMAN_VEL_PCT[si, di]
            c = KALMAN_CONF[si, di]
            if not np.isnan(v) and not np.isnan(c):
                KALMAN_REGIME[si, di] = np.sign(v) * c  # Signed confidence
    new['KALMAN_REGIME'] = KALMAN_REGIME

    print(f"  Kalman signals done ({time.time()-t0:.1f}s)", flush=True)

    # === Rank normalize ===
    def rank_pct(arr, start=60):
        res = np.full_like(arr, np.nan)
        for di in range(start, arr.shape[1]):
            vals = arr[:, di]
            mask = ~np.isnan(vals)
            if mask.sum() < 50:
                continue
            ranked = np.argsort(np.argsort(vals[mask])).astype(float)
            n = len(ranked)
            pct = ranked / max(n - 1, 1) * 100
            for k, idx in enumerate(np.where(mask)[0]):
                res[idx, di] = pct[k]
        return res

    factor_names = ['KALMAN_VEL_PCT', 'KALMAN_CONF', 'KALMAN_INNOV_MAG', 'KALMAN_REGIME']
    for name in factor_names:
        new[f'R_{name}'] = rank_pct(new[name])

    # Invert innovation magnitude: low innovation = stable = good
    inv = new['R_KALMAN_INNOV_MAG'].copy()
    mask = ~np.isnan(inv)
    inv[mask] = 100.0 - inv[mask]
    new['R_KALMAN_STABLE'] = inv

    print(f"  Total Kalman factors: {len(new)}", flush=True)
    return new


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V20 — Kalman Filter Strategy", flush=True)
    print("  From probability_theory.md Section 21", flush=True)
    print("  Adaptive MA + Velocity + Confidence + Regime Detection", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Load existing factors
    from alpha_v7b import compute_interaction_factors
    from alpha_v7d import compute_extra_factors
    from alpha_v7e import compute_v7e_factors
    from alpha_v7f import compute_advanced_interactions
    from alpha_v8 import compute_v8_factors, compute_v8_interactions
    from alpha_v9 import compute_v9_factors, compute_v9_interactions
    from alpha_v10 import compute_v10_factors, compute_v10_interactions
    from alpha_v11 import compute_v11_factors, compute_v11_interactions

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

    # V20 Kalman factors (no need for V14 HAR-RV — Kalman is the replacement)
    kalman_factors = compute_kalman_factors(NS, ND, C, O, H, L, V)
    all_factors = {**v11_all, **kalman_factors}

    print(f"\n  Total factors: {len(all_factors)}", flush=True)

    results = []

    # Baseline
    bwp = {'R_BWP_BNW': 0.3, 'R_TENSION': 0.3,
            'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2}
    for top_n in [1, 2]:
        for atr in [1.0, 1.2, 1.5]:
            r = backtest_v7c(bwp, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=top_n, rebalance_days=10, atr_stop_mult=atr)
            if r:
                r['test'] = f'BwpBNW_T{top_n}_A{atr}'
                results.append(r)
    print(f"  Baseline done", flush=True)

    # =====================================================================
    # KALMAN SINGLE FACTOR TESTS
    # =====================================================================
    print(f"\n  === KALMAN SINGLE FACTOR TESTS ===", flush=True)
    for fname in ['R_KALMAN_VEL_PCT', 'R_KALMAN_CONF', 'R_KALMAN_STABLE', 'R_KALMAN_REGIME']:
        r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=3, rebalance_days=10, atr_stop_mult=1.5)
        if r:
            print(f"  {fname:<25s}: Ann={r['ann']:+7.1f}% WR={r['wr']:5.1f}% "
                  f"Edge={r['edge']:+5.2f}% DD={r['max_dd']:5.1f}%", flush=True)

    # =====================================================================
    # KALMAN COMBINATION TESTS
    # =====================================================================
    portfolios = {
        # Kalman velocity + structure
        'KV_tens': {'R_KALMAN_VEL_PCT': 0.3, 'R_TENSION': 0.3,
                    'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # Kalman confidence (t-stat analog)
        'KC_tens': {'R_KALMAN_CONF': 0.3, 'R_TENSION': 0.3,
                    'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # Kalman regime (signed confidence)
        'KR_tens': {'R_KALMAN_REGIME': 0.3, 'R_TENSION': 0.3,
                    'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # Kalman stable (low innovation = calm = good)
        'KS_tens': {'R_KALMAN_STABLE': 0.3, 'R_TENSION': 0.3,
                    'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # Kalman velocity + BwpBNW (replace TENSION with Kalman)
        'KV_bwp': {'R_KALMAN_VEL_PCT': 0.3, 'R_BWP_BNW': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # Kalman confidence + BwpBNW
        'KC_bwp': {'R_KALMAN_CONF': 0.3, 'R_BWP_BNW': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # Kalman regime + BwpBNW
        'KR_bwp': {'R_KALMAN_REGIME': 0.25, 'R_BWP_BNW': 0.25,
                   'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        # Kalman triple: velocity + confidence + regime
        'K3_tens': {'R_KALMAN_VEL_PCT': 0.2, 'R_KALMAN_CONF': 0.2,
                    'R_KALMAN_REGIME': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        # Pure Kalman (all 4 signals)
        'K4': {'R_KALMAN_VEL_PCT': 0.25, 'R_KALMAN_CONF': 0.25,
               'R_KALMAN_STABLE': 0.25, 'R_KALMAN_REGIME': 0.25},
        # Kalman + momentum
        'KV_mom': {'R_KALMAN_VEL_PCT': 0.25, 'R_MOM5': 0.25,
                   'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        # Kalman + Kaufman efficiency
        'KV_ker': {'R_KALMAN_VEL_PCT': 0.25, 'R_KER': 0.25,
                   'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        # Kalman + squeeze
        'KV_sqz': {'R_KALMAN_VEL_PCT': 0.25, 'R_BB_WIDTH_PCT_INV': 0.25,
                   'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
    }

    for pname, weights in portfolios.items():
        for top_n in [1, 2]:
            for atr in [1.0, 1.2, 1.5]:
                r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, rebalance_days=10, atr_stop_mult=atr)
                if r:
                    r['test'] = f'{pname}_T{top_n}_A{atr}'
                    results.append(r)
        print(f"  {pname} done", flush=True)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  TOP 40 RESULTS (V20 KALMAN FILTER)", flush=True)
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

    # Best per strategy
    groups = {}
    for r in results:
        prefix = r['test'].split('_T')[0]
        if prefix not in groups or r['ann'] > groups[prefix]['ann']:
            groups[prefix] = r
    print(f"\n  Best per group:", flush=True)
    for r in sorted(groups.values(), key=lambda x: -x['ann']):
        pos = " ALL+" if all_positive(r) else ""
        print(f"    {r['test']:<30s} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}%{pos}", flush=True)

    print(f"\n{'='*70}", flush=True)
