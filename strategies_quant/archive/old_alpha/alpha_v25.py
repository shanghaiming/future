"""
Alpha V25 — Markov State Machine + KL Divergence Regime Detection
=================================================================
From probability_theory.md Sections 29 & 31:

Section 31: Markov State Machines
  - KMeans/GMM clustering into K states
  - Transition matrix P_ij = N(i→j) / Σ_j N(i→j)
  - P_ii > 0.7 = state persistence (momentum)
  - Transition entropy H(i) = -Σ_j P_ij log2(P_ij)

Section 29: Information Theory
  - KL divergence D_KL(P||Q) detects regime change
  - Low Shannon entropy = ordered market (trending)
  - High Shannon entropy = chaotic market (range)

Strategy:
  1. KMeans cluster price/vol features into 4 states
  2. Build transition matrix from rolling window
  3. Compute transition entropy (predictability)
  4. Compute KL divergence between recent and historical distributions
  5. Shannon entropy of returns distribution
  6. Combine into regime-aware factor

NO LOOK-AHEAD: All computations use data up to di-1 only.
"""
import sys, os, time, warnings
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.stats import entropy as scipy_entropy
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
from alpha_v7c import backtest_v7c


def compute_markov_kl_factors(NS, ND, C, O, H, L, V):
    """Compute Markov state machine + KL divergence factors.

    For each stock:
    1. Compute daily features: return, vol_ratio, range_pct, body_pct
    2. KMeans cluster into 4 states using rolling window
    3. Build transition matrix
    4. Compute transition entropy and persistence
    5. Compute Shannon entropy of returns distribution
    6. Compute KL divergence between recent and historical

    SELF-CHECK: d = di - 1. All data up to d only.
    """
    t0 = time.time()
    new = {}

    WINDOW = 200     # ~10 months
    N_STATES = 4
    ENTROPY_BINS = 10

    MK_PERSISTENCE = np.full((NS, ND), np.nan)     # State persistence P_ii
    MK_TRANS_ENTROPY = np.full((NS, ND), np.nan)    # Transition entropy H(P_i)
    MK_SHANNON_ENTROPY = np.full((NS, ND), np.nan)  # Return distribution entropy
    MK_KL_DIVERGENCE = np.full((NS, ND), np.nan)    # KL divergence
    MK_STATE_SCORE = np.full((NS, ND), np.nan)      # Current state's average return
    MK_BULL_TRANS = np.full((NS, ND), np.nan)        # P(any_state → bull_state)

    for si in range(NS):
        last_kmeans = None
        last_train_end = -999
        state_history = []
        state_returns = {}  # state → list of returns

        for di in range(MIN_TRAIN, ND):
            d = di - 1  # SELF-CHECK

            start = max(0, d - WINDOW + 1)
            closes = C[si, start:d + 1]
            highs = H[si, start:d + 1]
            lows = L[si, start:d + 1]
            opens = O[si, start:d + 1]
            volumes = V[si, start:d + 1]

            valid_mask = ~np.isnan(closes) & ~np.isnan(highs) & ~np.isnan(lows)
            valid_idx = np.where(valid_mask)[0]
            if len(valid_idx) < 80:
                continue

            closes_v = closes[valid_idx]
            highs_v = highs[valid_idx]
            lows_v = lows[valid_idx]
            opens_v = opens[valid_idx]
            volumes_v = volumes[valid_idx]

            # Daily features
            returns = np.diff(closes_v) / closes_v[:-1]
            ranges = (highs_v[1:] - lows_v[1:]) / closes_v[:-1]
            bodies = np.abs(closes_v[1:] - opens_v[1:]) / closes_v[:-1]

            if len(returns) < 50:
                continue

            # Feature matrix for clustering
            # Rank-transform features (非参数 — from probability_theory.md Section 16)
            n_feat = len(returns)
            ret_rank = np.argsort(np.argsort(returns)).astype(float) / max(n_feat - 1, 1)
            range_rank = np.argsort(np.argsort(ranges)).astype(float) / max(n_feat - 1, 1)
            body_rank = np.argsort(np.argsort(bodies)).astype(float) / max(n_feat - 1, 1)

            features = np.column_stack([ret_rank, range_rank, body_rank])

            # Remove NaN
            valid_feat = ~np.any(np.isnan(features), axis=1)
            features = features[valid_feat]
            returns_v = returns[valid_feat]

            if len(features) < 50:
                continue

            # Retrain KMeans every 63 days
            if last_kmeans is None or (d - last_train_end) >= 63:
                try:
                    km = MiniBatchKMeans(n_clusters=N_STATES, random_state=42,
                                        n_init=5, batch_size=min(100, len(features)))
                    km.fit(features)
                    last_kmeans = km
                    last_train_end = d

                    # Label states by mean return
                    state_means = {}
                    for s in range(N_STATES):
                        mask_s = km.labels_ == s
                        if mask_s.sum() > 0:
                            state_means[s] = np.mean(returns_v[mask_s])
                        else:
                            state_means[s] = 0.0

                    # Sort states: 0=bull, 1=mild_bull, 2=mild_bear, 3=bear
                    sorted_states = sorted(state_means.keys(), key=lambda x: -state_means[x])
                    state_order = {sorted_states[i]: i for i in range(len(sorted_states))}

                    # Store return mapping
                    state_returns = {}
                    for s in range(N_STATES):
                        mask_s = km.labels_ == s
                        if mask_s.sum() > 0:
                            state_returns[s] = returns_v[mask_s]

                except Exception:
                    continue

            # Predict current state
            if last_kmeans is None:
                continue

            # Current feature (last day)
            curr_feat = features[-1:]
            try:
                curr_state = last_kmeans.predict(curr_feat)[0]
            except Exception:
                continue

            # Track state history
            state_history.append(curr_state)

            # =====================================================================
            # Transition matrix
            # =====================================================================
            if len(state_history) >= 30:
                # Build transition counts
                trans = np.zeros((N_STATES, N_STATES))
                for k in range(len(state_history) - 1):
                    from_s = state_history[k]
                    to_s = state_history[k + 1]
                    if from_s < N_STATES and to_s < N_STATES:
                        trans[from_s, to_s] += 1

                # Normalize
                row_sums = trans.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                trans_prob = trans / row_sums

                # Persistence: P(curr_state → curr_state)
                persistence = trans_prob[curr_state, curr_state]
                MK_PERSISTENCE[si, di] = persistence

                # Transition entropy for current state
                p_row = trans_prob[curr_state]
                p_row = p_row[p_row > 0]
                if len(p_row) > 0:
                    trans_ent = -np.sum(p_row * np.log2(p_row))
                    MK_TRANS_ENTROPY[si, di] = trans_ent

                # Bull transition: P(any → bull_state=sorted_states[0])
                bull_state = sorted_states[0]
                avg_bull_trans = np.mean(trans_prob[:, bull_state])
                MK_BULL_TRANS[si, di] = avg_bull_trans

            # =====================================================================
            # Current state score
            # =====================================================================
            if curr_state in state_returns and len(state_returns[curr_state]) > 0:
                MK_STATE_SCORE[si, di] = np.mean(state_returns[curr_state][-20:])

            # =====================================================================
            # Shannon entropy of returns distribution
            # =====================================================================
            if len(returns_v) >= 30:
                recent_returns = returns_v[-60:] if len(returns_v) >= 60 else returns_v
                counts, _ = np.histogram(recent_returns, bins=ENTROPY_BINS, density=False)
                total = counts.sum()
                if total > 0:
                    probs = counts[counts > 0] / total
                    shannon_ent = -np.sum(probs * np.log2(probs))
                    MK_SHANNON_ENTROPY[si, di] = shannon_ent

            # =====================================================================
            # KL divergence: recent vs historical
            # =====================================================================
            if len(returns_v) >= 100:
                hist = returns_v[:-30]
                recent = returns_v[-30:]

                hist_counts, bin_edges = np.histogram(hist, bins=ENTROPY_BINS, density=False)
                recent_counts, _ = np.histogram(recent, bins=bin_edges, density=False)

                hist_total = hist_counts.sum()
                recent_total = recent_counts.sum()

                if hist_total > 0 and recent_total > 0:
                    p = (hist_counts + 1) / (hist_total + ENTROPY_BINS)  # Laplace smoothing
                    q = (recent_counts + 1) / (recent_total + ENTROPY_BINS)

                    kl_div = np.sum(q * np.log(q / p))
                    MK_KL_DIVERGENCE[si, di] = kl_div

    new['MK_PERSISTENCE'] = MK_PERSISTENCE
    new['MK_TRANS_ENTROPY'] = MK_TRANS_ENTROPY
    new['MK_SHANNON_ENTROPY'] = MK_SHANNON_ENTROPY
    new['MK_KL_DIVERGENCE'] = MK_KL_DIVERGENCE
    new['MK_STATE_SCORE'] = MK_STATE_SCORE
    new['MK_BULL_TRANS'] = MK_BULL_TRANS

    print(f"  Markov + KL factors done ({time.time()-t0:.1f}s)", flush=True)

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

    new['R_MK_PERSISTENCE'] = rank_pct(new['MK_PERSISTENCE'])
    new['R_MK_BULL_TRANS'] = rank_pct(new['MK_BULL_TRANS'])
    new['R_MK_STATE_SCORE'] = rank_pct(new['MK_STATE_SCORE'])

    # Invert entropy (low entropy = ordered = good = trending)
    inv_trans_ent = new['MK_TRANS_ENTROPY'].copy()
    mask = ~np.isnan(inv_trans_ent)
    if mask.any():
        mn, mx = np.nanmin(inv_trans_ent), np.nanmax(inv_trans_ent)
        if mx > mn:
            inv_trans_ent[mask] = mx - inv_trans_ent[mask] + mn
    new['R_MK_LOW_TRANS_ENT'] = rank_pct(inv_trans_ent)

    inv_shannon = new['MK_SHANNON_ENTROPY'].copy()
    mask = ~np.isnan(inv_shannon)
    if mask.any():
        mn, mx = np.nanmin(inv_shannon), np.nanmax(inv_shannon)
        if mx > mn:
            inv_shannon[mask] = mx - inv_shannon[mask] + mn
    new['R_MK_LOW_SHANNON'] = rank_pct(inv_shannon)

    # KL divergence: high = regime change = could be opportunity
    new['R_MK_KL_DIV'] = rank_pct(new['MK_KL_DIVERGENCE'])

    print(f"  Total Markov+KL factors: {len(new)}", flush=True)
    return new


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V25 — Markov State Machine + KL Divergence", flush=True)
    print("  (probability_theory.md Sections 29 & 31)", flush=True)
    print("  KMeans 4-state + transition matrix + Shannon entropy + KL div", flush=True)
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

    # V25 Markov+KL factors
    mk_factors = compute_markov_kl_factors(NS, ND, C, O, H, L, V)
    all_factors = {**v11_all, **mk_factors}

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
    # SINGLE FACTOR TESTS
    # =====================================================================
    print(f"\n  === MARKOV+KL SINGLE FACTOR TESTS ===", flush=True)
    for fname in ['R_MK_PERSISTENCE', 'R_MK_BULL_TRANS', 'R_MK_STATE_SCORE',
                  'R_MK_LOW_TRANS_ENT', 'R_MK_LOW_SHANNON', 'R_MK_KL_DIV']:
        r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=3, rebalance_days=10, atr_stop_mult=1.5)
        if r:
            print(f"  {fname:<25s}: Ann={r['ann']:+7.1f}% WR={r['wr']:5.1f}% "
                  f"Edge={r['edge']:+5.2f}% DD={r['max_dd']:5.1f}%", flush=True)

    # =====================================================================
    # COMBINATION TESTS
    # =====================================================================
    portfolios = {
        # Persistence + structure
        'MP_tens': {'R_MK_PERSISTENCE': 0.3, 'R_TENSION': 0.3,
                    'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # Bull transition + BwpBNW
        'MB_bwp': {'R_MK_BULL_TRANS': 0.3, 'R_BWP_BNW': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # Low Shannon entropy (ordered) + momentum
        'MS_mom': {'R_MK_LOW_SHANNON': 0.3, 'R_MOM5': 0.3,
                   'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        # Low transition entropy + squeeze
        'ME_sqz': {'R_MK_LOW_TRANS_ENT': 0.3, 'R_BB_WIDTH_PCT_INV': 0.3,
                   'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        # Triple Markov
        'M3': {'R_MK_PERSISTENCE': 0.2, 'R_MK_BULL_TRANS': 0.2,
               'R_MK_LOW_SHANNON': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        # KL divergence + Kalman (regime change + adaptive)
        'MK_KV': {'R_MK_KL_DIV': 0.25, 'R_KALMAN_VEL_PCT': 0.25,
                  'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        # State score + Hurst (both measure trend quality)
        'MS_hurst': {'R_MK_STATE_SCORE': 0.25, 'R_HURST': 0.25,
                     'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        # Persistence + DMD bull ratio (both measure trend continuation)
        'MP_DMD': {'R_MK_PERSISTENCE': 0.25, 'R_DMD_BULL_RATIO': 0.25,
                   'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        # KL divergence + wavelet Hurst
        'MK_WH': {'R_MK_KL_DIV': 0.25, 'R_WAV_HURST': 0.25,
                  'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        # Full Markov + entropy
        'MF': {'R_MK_PERSISTENCE': 0.15, 'R_MK_BULL_TRANS': 0.15,
               'R_MK_LOW_SHANNON': 0.15, 'R_MK_KL_DIV': 0.15,
               'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
    }

    for pname, weights in portfolios.items():
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
    print(f"  TOP 40 RESULTS (V25 MARKOV+KL)", flush=True)
    print(f"  {'Test':<30s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*80}", flush=True)
    for r in results[:40]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<30s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    for i, r in enumerate(results[:5]):
        print(f"\n  Year-by-year #{i+1}: {r['test']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
