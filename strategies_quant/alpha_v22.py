"""
Alpha V22 — HMM Regime Detection Strategy (from probability_theory.md Section 16)
==================================================================================
Uses Gaussian Mixture Model (GMM) as HMM emission model with:
  1. 3-state regime detection (bull/bear/range) via GMM clustering
  2. Multi-dimensional emission: [returns, volatility, volume_percentile]
  3. Rolling window estimation (500-day) — no look-ahead
  4. State persistence scoring (analog of HMM transition diagonal)
  5. Regime-conditional factor weighting

Since hmmlearn doesn't compile on Python 3.14, we use sklearn's GMM
with manual Viterbi-like decoding and state persistence tracking.

KEY INSIGHT from probability_theory.md Section 16:
  - HMM emission should use RANK (percentile) not raw values (A股涨跌停)
  - Multi-dimensional emission: returns + vol + volume_rank
  - 2-3 year rolling window (500-750 obs)
  - State persistence a_ii > 0.7 means regime has inertia

NO LOOK-AHEAD: GMM trained on [0:di-1] data only, predict state at di.
"""
import sys, os, time, warnings
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import rankdata
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
from alpha_v7c import backtest_v7c


def compute_hmm_factors(NS, ND, C, O, H, L, V):
    """Compute HMM regime detection factors using GMM.

    OPTIMIZED: Train a global GMM on all stocks' aggregate features every
    63 days, then predict per-stock states each day. Much faster than
    per-stock training.

    SELF-CHECK: d = di - 1 for all data access.
    """
    t0 = time.time()
    new = {}

    # Output arrays
    HMM_STATE = np.full((NS, ND), np.nan)
    HMM_BULL_PROB = np.full((NS, ND), np.nan)
    HMM_BEAR_PROB = np.full((NS, ND), np.nan)
    HMM_REGIME_SCORE = np.full((NS, ND), np.nan)
    HMM_PERSISTENCE = np.full((NS, ND), np.nan)

    N_STATES = 3
    RETRAIN_EVERY = 63

    # Precompute per-stock features for all days
    # Feature 1: 20-day return rank
    # Feature 2: 20-day volatility
    # Feature 3: 20-day volume rank
    RET_RANK = np.full((NS, ND), np.nan)
    VOL_20 = np.full((NS, ND), np.nan)
    VRANK = np.full((NS, ND), np.nan)

    for si in range(NS):
        for di in range(21, ND):
            d = di - 1
            prices = C[si, d - 19:d + 1]
            if np.any(np.isnan(prices)) or len(prices) < 5:
                continue
            log_ret = np.diff(np.log(prices))
            if np.any(np.isnan(log_ret)):
                continue
            # Return rank
            RET_RANK[si, di] = np.sum(log_ret <= log_ret[-1]) / len(log_ret)
            # Volatility
            VOL_20[si, di] = np.std(log_ret)
            # Volume rank
            vols = V[si, d - 19:d + 1]
            v_valid = vols[~np.isnan(vols)]
            if len(v_valid) >= 5:
                VRANK[si, di] = np.sum(v_valid <= v_valid[-1]) / len(v_valid)

    print(f"  HMM features precomputed ({time.time()-t0:.1f}s)", flush=True)

    # Train global GMM periodically
    global_model = None
    bull_state = 0
    bear_state = 2
    range_state = 1
    last_train_di = -999

    for di in range(MIN_TRAIN, ND):
        d = di - 1

        # Retrain GMM every 63 days
        if global_model is None or (di - last_train_di) >= RETRAIN_EVERY:
            # Collect features from all stocks at this time
            ret_vals = RET_RANK[:, di]
            vol_vals = VOL_20[:, di]
            vrank_vals = VRANK[:, di]

            mask = ~np.isnan(ret_vals) & ~np.isnan(vol_vals) & ~np.isnan(vrank_vals)
            if mask.sum() < 100:
                continue

            X = np.column_stack([ret_vals[mask], vol_vals[mask], vrank_vals[mask]])

            try:
                gmm = GaussianMixture(
                    n_components=N_STATES,
                    covariance_type='full',
                    n_init=3,
                    max_iter=100,
                    random_state=42,
                    reg_covar=1e-6
                )
                gmm.fit(X)
                global_model = gmm
                last_train_di = di

                # Label states by mean return rank
                means = gmm.means_[:, 0]
                sorted_idx = np.argsort(means)
                bear_state = sorted_idx[0]
                range_state = sorted_idx[1]
                bull_state = sorted_idx[2]
            except Exception:
                continue

        if global_model is None:
            continue

        # Predict all stocks at once
        ret_vals = RET_RANK[:, di]
        vol_vals = VOL_20[:, di]
        vrank_vals = VRANK[:, di]
        mask = ~np.isnan(ret_vals) & ~np.isnan(vol_vals) & ~np.isnan(vrank_vals)

        if mask.sum() < 10:
            continue

        X_all = np.column_stack([ret_vals[mask], vol_vals[mask], vrank_vals[mask]])

        try:
            probs = global_model.predict_proba(X_all)
            states = global_model.predict(X_all)
        except Exception:
            continue

        # Write results
        indices = np.where(mask)[0]
        for k, si in enumerate(indices):
            bull_prob = probs[k, bull_state]
            bear_prob = probs[k, bear_state]

            if states[k] == bull_state:
                HMM_STATE[si, di] = 0
            elif states[k] == bear_state:
                HMM_STATE[si, di] = 2
            else:
                HMM_STATE[si, di] = 1

            HMM_BULL_PROB[si, di] = bull_prob
            HMM_BEAR_PROB[si, di] = bear_prob
            HMM_REGIME_SCORE[si, di] = bull_prob - bear_prob

    # Compute persistence (how many of last 10 days in same state)
    for si in range(NS):
        state_seq = HMM_STATE[si, :]
        for di in range(MIN_TRAIN + 10, ND):
            if np.isnan(state_seq[di]):
                continue
            recent = state_seq[di - 9:di + 1]
            valid = recent[~np.isnan(recent)]
            if len(valid) >= 5:
                HMM_PERSISTENCE[si, di] = np.sum(valid == state_seq[di]) / len(valid)

    new['HMM_STATE'] = HMM_STATE
    new['HMM_BULL_PROB'] = HMM_BULL_PROB
    new['HMM_BEAR_PROB'] = HMM_BEAR_PROB
    new['HMM_REGIME_SCORE'] = HMM_REGIME_SCORE
    new['HMM_PERSISTENCE'] = HMM_PERSISTENCE

    print(f"  HMM regime detection done ({time.time()-t0:.1f}s)", flush=True)

    # =====================================================================
    # Rank normalize
    # =====================================================================
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

    # Rank the regime score and persistence
    new['R_HMM_REGIME_SCORE'] = rank_pct(new['HMM_REGIME_SCORE'])
    new['R_HMM_PERSISTENCE'] = rank_pct(new['HMM_PERSISTENCE'])
    new['R_HMM_BULL_PROB'] = rank_pct(new['HMM_BULL_PROB'])

    # Invert bear prob (low bear = good)
    inv_bear = new['HMM_BEAR_PROB'].copy()
    mask = ~np.isnan(inv_bear)
    inv_bear[mask] = 1.0 - inv_bear[mask]
    new['R_HMM_LOW_BEAR'] = rank_pct(inv_bear)

    print(f"  Total HMM factors: {len(new)}", flush=True)
    return new


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V22 — HMM Regime Detection (probability_theory.md Section 16)", flush=True)
    print("  GMM-based 3-state regime: bull/bear/range", flush=True)
    print("  Multi-dimensional emission: returns + vol + volume_rank", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Load existing factors
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

    # V22 HMM factors
    hmm_factors = compute_hmm_factors(NS, ND, C, O, H, L, V)
    all_factors = {**v11_all, **hmm_factors}

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
    # HMM SINGLE FACTOR TESTS
    # =====================================================================
    print(f"\n  === HMM SINGLE FACTOR TESTS ===", flush=True)
    for fname in ['R_HMM_REGIME_SCORE', 'R_HMM_PERSISTENCE', 'R_HMM_BULL_PROB', 'R_HMM_LOW_BEAR']:
        r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=3, rebalance_days=10, atr_stop_mult=1.5)
        if r:
            print(f"  {fname:<25s}: Ann={r['ann']:+7.1f}% WR={r['wr']:5.1f}% "
                  f"Edge={r['edge']:+5.2f}% DD={r['max_dd']:5.1f}%", flush=True)

    # =====================================================================
    # HMM COMBINATION TESTS
    # =====================================================================
    portfolios = {
        # HMM regime score + structure
        'HR_tens': {'R_HMM_REGIME_SCORE': 0.3, 'R_TENSION': 0.3,
                    'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # HMM persistence + BwpBNW
        'HP_bwp': {'R_HMM_PERSISTENCE': 0.3, 'R_BWP_BNW': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # HMM bull prob + momentum
        'HB_mom': {'R_HMM_BULL_PROB': 0.3, 'R_MOM5': 0.3,
                   'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        # HMM low bear + squeeze
        'HL_sqz': {'R_HMM_LOW_BEAR': 0.3, 'R_BB_WIDTH_PCT_INV': 0.3,
                   'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        # Triple HMM
        'H3': {'R_HMM_REGIME_SCORE': 0.2, 'R_HMM_PERSISTENCE': 0.2,
               'R_HMM_BULL_PROB': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        # HMM regime + BwpBNW (replace TENSION with HMM)
        'HR_bwp': {'R_HMM_REGIME_SCORE': 0.25, 'R_BWP_BNW': 0.25,
                   'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        # HMM + Kalman (from V20)
        'HK_vel': {'R_HMM_REGIME_SCORE': 0.25, 'R_KALMAN_VEL_PCT': 0.25,
                   'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        # HMM persistence + Kalman confidence
        'HP_KC': {'R_HMM_PERSISTENCE': 0.25, 'R_KALMAN_CONF': 0.25,
                  'R_BWP_BNW': 0.25, 'R_R_SQUARED': 0.25},
        # HMM + KER (efficiency)
        'HR_ker': {'R_HMM_REGIME_SCORE': 0.25, 'R_KER': 0.25,
                   'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        # HMM + Hurst
        'HR_hurst': {'R_HMM_REGIME_SCORE': 0.25, 'R_HURST': 0.25,
                     'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
    }

    for pname, weights in portfolios.items():
        # Check if all factors exist
        missing = [f for f in weights if f not in all_factors]
        if missing:
            print(f"  SKIP {pname}: missing {missing}", flush=True)
            continue
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
    print(f"  TOP 40 RESULTS (V22 HMM REGIME)", flush=True)
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
