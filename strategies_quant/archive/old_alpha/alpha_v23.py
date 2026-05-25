"""
Alpha V23 — DMD Dynamic Mode Decomposition (probability_theory.md Section 27)
===============================================================================
Implements DMD from scratch using SVD. For each stock:
  1. Build time-delay embedding snapshot matrix X from closing prices
  2. SVD decomposition → reduced-order linear operator Ã
  3. Eigen-decomposition of Ã → DMD modes Φ, eigenvalues Λ, amplitudes b
  4. Compute bull energy ratio: ρ_bull = Σ(|λ_k|>1)|b_k|² / Σ|b_k|²
  5. Detect cycle periods from complex eigenvalues

From probability_theory.md:
  - |λ|>1 = growing mode (bull), |λ|<1 = decaying (bear), |λ|≈1 = oscillating
  - ρ_bull > 0.6 = trend market, ρ_bull < 0.3 = range market
  - DMD captures GLOBAL oscillation patterns (complementary to NW local trends)

NO LOOK-AHEAD: Snapshot matrix uses data up to di-1 only.
"""
import sys, os, time, warnings
import numpy as np
from numpy.linalg import svd, eig, pinv, norm
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


def compute_dmd_factors(NS, ND, C, O, H, L, V):
    """Compute DMD-based factors for each stock.

    Uses rolling window of ~100 trading days.
    Time-delay embedding with n_delays=10.

    SELF-CHECK: d = di - 1. Snapshot matrix uses prices[di-window:di].
    """
    t0 = time.time()
    new = {}

    DMD_BULL_RATIO = np.full((NS, ND), np.nan)
    DMD_N_MODES = np.full((NS, ND), np.nan)
    DMD_DOMINANT_PERIOD = np.full((NS, ND), np.nan)
    DMD_GROWTH_STRENGTH = np.full((NS, ND), np.nan)   # Σ(|λ|-1)+ for |λ|>1
    DMD_DECAY_STRENGTH = np.full((NS, ND), np.nan)     # Σ(1-|λ|) for |λ|<1

    N_DELAYS = 10
    WINDOW = 120     # ~6 months
    SVD_RANK = 8     # Reduced order

    for si in range(NS):
        for di in range(MIN_TRAIN + WINDOW, ND):
            d = di - 1  # SELF-CHECK

            # Get price window
            start = max(0, d - WINDOW + 1)
            prices = C[si, start:d + 1]
            valid_mask = ~np.isnan(prices)
            prices = prices[valid_mask]

            if len(prices) < N_DELAYS + 10:
                continue

            # Normalize prices (demean + divide by std to avoid numerical issues)
            p_mean = np.mean(prices)
            p_std = np.std(prices)
            if p_std < 1e-10:
                continue
            prices_norm = (prices - p_mean) / p_std

            # Build time-delay embedding snapshot matrix
            # X = [x_1, ..., x_{m-1}], X' = [x_2, ..., x_m]
            n_obs = len(prices_norm)
            m = n_obs - N_DELAYS + 1  # number of snapshots

            if m < SVD_RANK + 5:
                continue

            # X[i,j] = prices_norm[j + i] for i=0..N_DELAYS-1, j=0..m-2
            X = np.zeros((N_DELAYS, m - 1))
            Xp = np.zeros((N_DELAYS, m - 1))
            for i in range(N_DELAYS):
                X[i] = prices_norm[i:m - 1 + i]
                Xp[i] = prices_norm[i + 1:m + i]

            # SVD of X
            try:
                U, S, Vt = svd(X, full_matrices=False)
            except Exception:
                continue

            # Truncate to SVD_RANK
            r = min(SVD_RANK, len(S))
            Ur = U[:, :r]
            Sr = np.diag(S[:r])
            Vr = Vt[:r, :].T

            # Reduced-order operator Ã = Ur' * X' * Vr * Sr^(-1)
            try:
                A_tilde = Ur.T @ Xp @ Vr @ np.linalg.inv(Sr)
            except Exception:
                continue

            # Eigen-decomposition of Ã
            try:
                eigenvalues, W = eig(A_tilde)
            except Exception:
                continue

            # DMD modes: Φ = X' * Vr * Σr^(-1) * W
            try:
                Phi = Xp @ Vr @ np.linalg.inv(Sr) @ W
            except Exception:
                continue

            # Amplitudes: b = Φ† * x_1
            x1 = X[:, 0]
            try:
                b = pinv(Phi) @ x1
            except Exception:
                continue

            # Compute bull energy ratio
            abs_eigs = np.abs(eigenvalues)
            abs_b = np.abs(b)

            total_energy = np.sum(abs_b ** 2)
            if total_energy < 1e-10:
                continue

            growing_mask = abs_eigs > 1.0
            bull_energy = np.sum(abs_b[growing_mask] ** 2)
            bull_ratio = bull_energy / total_energy

            # Growth strength: how much the growing modes exceed 1
            growth_strength = np.sum(np.maximum(abs_eigs[growing_mask] - 1.0, 0) * abs_b[growing_mask] ** 2)

            # Decay strength: how much the decaying modes are below 1
            decay_mask = abs_eigs < 1.0
            decay_strength = np.sum(np.maximum(1.0 - abs_eigs[decay_mask], 0) * abs_b[decay_mask] ** 2)

            # Dominant cycle from complex eigenvalues
            complex_mask = np.abs(np.imag(eigenvalues)) > 1e-6
            dominant_period = np.nan
            if np.any(complex_mask):
                # Pick the complex eigenvalue with largest amplitude
                complex_idx = np.where(complex_mask)[0]
                best_c = complex_idx[np.argmax(abs_b[complex_idx])]
                omega = np.abs(np.angle(eigenvalues[best_c]))
                if omega > 1e-6:
                    dominant_period = 2 * np.pi / omega

            DMD_BULL_RATIO[si, di] = bull_ratio
            DMD_N_MODES[si, di] = np.sum(growing_mask)
            DMD_GROWTH_STRENGTH[si, di] = growth_strength / total_energy
            DMD_DECAY_STRENGTH[si, di] = decay_strength / total_energy
            if not np.isnan(dominant_period):
                DMD_DOMINANT_PERIOD[si, di] = min(dominant_period, 200)  # Cap at 200 days

    new['DMD_BULL_RATIO'] = DMD_BULL_RATIO
    new['DMD_N_MODES'] = DMD_N_MODES
    new['DMD_GROWTH_STRENGTH'] = DMD_GROWTH_STRENGTH
    new['DMD_DECAY_STRENGTH'] = DMD_DECAY_STRENGTH
    new['DMD_DOMINANT_PERIOD'] = DMD_DOMINANT_PERIOD

    print(f"  DMD decomposition done ({time.time()-t0:.1f}s)", flush=True)

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

    new['R_DMD_BULL_RATIO'] = rank_pct(new['DMD_BULL_RATIO'])
    new['R_DMD_GROWTH_STR'] = rank_pct(new['DMD_GROWTH_STRENGTH'])
    new['R_DMD_N_MODES'] = rank_pct(new['DMD_N_MODES'])

    # Invert decay (low decay = good = less bearish)
    inv_decay = new['DMD_DECAY_STRENGTH'].copy()
    mask = ~np.isnan(inv_decay)
    if mask.any():
        mn = np.nanmin(inv_decay)
        mx = np.nanmax(inv_decay)
        if mx > mn:
            inv_decay[mask] = 1.0 - (inv_decay[mask] - mn) / (mx - mn)
    new['R_DMD_LOW_DECAY'] = rank_pct(inv_decay)

    # Dominant period as a factor (longer period = more stable trend)
    new['R_DMD_PERIOD'] = rank_pct(new['DMD_DOMINANT_PERIOD'])

    print(f"  Total DMD factors: {len(new)}", flush=True)
    return new


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V23 — DMD Dynamic Mode Decomposition", flush=True)
    print("  (probability_theory.md Section 27)", flush=True)
    print("  Bull energy ratio + growth/decay strength + cycle detection", flush=True)
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

    # V23 DMD factors
    dmd_factors = compute_dmd_factors(NS, ND, C, O, H, L, V)
    all_factors = {**v11_all, **dmd_factors}

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
    # DMD SINGLE FACTOR TESTS
    # =====================================================================
    print(f"\n  === DMD SINGLE FACTOR TESTS ===", flush=True)
    for fname in ['R_DMD_BULL_RATIO', 'R_DMD_GROWTH_STR', 'R_DMD_LOW_DECAY',
                  'R_DMD_N_MODES', 'R_DMD_PERIOD']:
        r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=3, rebalance_days=10, atr_stop_mult=1.5)
        if r:
            print(f"  {fname:<25s}: Ann={r['ann']:+7.1f}% WR={r['wr']:5.1f}% "
                  f"Edge={r['edge']:+5.2f}% DD={r['max_dd']:5.1f}%", flush=True)

    # =====================================================================
    # DMD COMBINATION TESTS
    # =====================================================================
    portfolios = {
        # DMD bull ratio + structure
        'DB_tens': {'R_DMD_BULL_RATIO': 0.3, 'R_TENSION': 0.3,
                    'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # DMD growth strength + BwpBNW
        'DG_bwp': {'R_DMD_GROWTH_STR': 0.3, 'R_BWP_BNW': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # DMD bull + squeeze
        'DB_sqz': {'R_DMD_BULL_RATIO': 0.3, 'R_BB_WIDTH_PCT_INV': 0.3,
                   'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        # DMD growth + momentum
        'DG_mom': {'R_DMD_GROWTH_STR': 0.3, 'R_MOM5': 0.3,
                   'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        # DMD low decay + Kalman
        'DL_KV': {'R_DMD_LOW_DECAY': 0.25, 'R_KALMAN_VEL_PCT': 0.25,
                  'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        # Triple DMD
        'D3': {'R_DMD_BULL_RATIO': 0.2, 'R_DMD_GROWTH_STR': 0.2,
               'R_DMD_LOW_DECAY': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        # DMD + HMM (from V22)
        'DH_regime': {'R_DMD_BULL_RATIO': 0.25, 'R_HMM_REGIME_SCORE': 0.25,
                      'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        # DMD + Hurst (both measure trend structure)
        'DH_hurst': {'R_DMD_BULL_RATIO': 0.25, 'R_HURST': 0.25,
                     'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        # DMD + KER
        'DG_ker': {'R_DMD_GROWTH_STR': 0.25, 'R_KER': 0.25,
                   'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        # DMD period (cycle) + NW slope (local trend) — complementary
        'DP_NW': {'R_DMD_PERIOD': 0.25, 'R_NW_SLOPE': 0.25,
                  'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
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
    print(f"  TOP 40 RESULTS (V23 DMD)", flush=True)
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
