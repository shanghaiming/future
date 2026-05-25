"""
Alpha V27 — Algorithm Fusion: Best Factor from Each Algorithm Family
=====================================================================
Combines the best factor from each algorithmic dimension into one score.

From probability_theory.md, we have 7 independent signal dimensions:
  1. Price geometry: R_TENSION, R_R_SQUARED, R_BWP_BNW
  2. Kalman adaptive: R_KALMAN_VEL_PCT, R_KALMAN_CONF, R_KALMAN_REGIME
  3. HMM regime: R_HMM_REGIME_SCORE, R_HMM_PERSISTENCE
  4. DMD spectral: R_DMD_BULL_RATIO, R_DMD_GROWTH_STR
  5. Wavelet multi-scale: R_WAV_HURST, R_WAV_TREND_STR
  6. Markov+Entropy: R_MK_PERSISTENCE, R_MK_LOW_SHANNON
  7. FFT+CUSUM: R_FFT_LOW_FREQ, R_CUSUM_UP

Key principle from Section 10 Copula:
  "组合的关键不是哪些好，而是哪些独立"
  Independent signals agreeing → Bayesian posterior ↑↑↑

This script:
1. Computes factors from V20 (Kalman), V22 (HMM), V23 (DMD), V24 (Wavelet),
   V25 (Markov+KL), V26 (FFT+CUSUM)
2. Tests each factor individually
3. Tests all 2-way, 3-way combinations within each algorithm family
4. Tests cross-algorithm combinations (one from each family)
5. Finds the most independent algorithm signals via Kendall τ

NO LOOK-AHEAD: All factors use d = di - 1.
"""
import sys, os, time, warnings
import numpy as np
from scipy.stats import kendalltau
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


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V27 — Algorithm Fusion", flush=True)
    print("  Best factor from each algorithm family → combined signal", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # =====================================================================
    # LOAD ALL FACTORS
    # =====================================================================
    print("\n  Loading base factors...", flush=True)
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

    # =====================================================================
    # ALGORITHM FACTORS (lazy import to avoid circular deps)
    # =====================================================================
    print("\n  Computing Kalman factors (V20)...", flush=True)
    from alpha_v20 import compute_kalman_factors
    kalman_factors = compute_kalman_factors(NS, ND, C, O, H, L, V)

    print("  Computing HMM factors (V22)...", flush=True)
    from alpha_v22 import compute_hmm_factors
    hmm_factors = compute_hmm_factors(NS, ND, C, O, H, L, V)

    print("  Computing DMD factors (V23)...", flush=True)
    from alpha_v23 import compute_dmd_factors
    dmd_factors = compute_dmd_factors(NS, ND, C, O, H, L, V)

    print("  Computing Wavelet factors (V24)...", flush=True)
    from alpha_v24 import compute_wavelet_factors
    wav_factors = compute_wavelet_factors(NS, ND, C, O, H, L, V)

    print("  Computing Markov+KL factors (V25)...", flush=True)
    from alpha_v25 import compute_markov_kl_factors
    mk_factors = compute_markov_kl_factors(NS, ND, C, O, H, L, V)

    print("  Computing FFT+CUSUM factors (V26)...", flush=True)
    from alpha_v26 import compute_fft_cusum_factors
    fft_factors = compute_fft_cusum_factors(NS, ND, C, O, H, L, V)

    # Merge all
    all_factors = {**v11_all, **kalman_factors, **hmm_factors,
                   **dmd_factors, **wav_factors, **mk_factors, **fft_factors}

    print(f"\n  Total factors: {len(all_factors)}", flush=True)

    # =====================================================================
    # DEFINE ALGORITHM FAMILIES
    # =====================================================================
    families = {
        'Geometry': ['R_TENSION', 'R_R_SQUARED', 'R_BWP_BNW', 'R_SMA_DEV',
                     'R_MOM5', 'R_KINETIC', 'R_BODY_RATIO'],
        'Kalman': ['R_KALMAN_VEL_PCT', 'R_KALMAN_CONF', 'R_KALMAN_REGIME',
                   'R_KALMAN_STABLE'],
        'HMM': ['R_HMM_REGIME_SCORE', 'R_HMM_PERSISTENCE',
                'R_HMM_BULL_PROB', 'R_HMM_LOW_BEAR'],
        'DMD': ['R_DMD_BULL_RATIO', 'R_DMD_GROWTH_STR', 'R_DMD_LOW_DECAY',
                'R_DMD_N_MODES', 'R_DMD_PERIOD'],
        'Wavelet': ['R_WAV_HURST', 'R_WAV_TREND_STR', 'R_WAV_LONG_MOM',
                    'R_WAV_DENOISED_MOM', 'R_WAV_COMPOSITE', 'R_WAV_ENERGY_RATIO'],
        'Markov': ['R_MK_PERSISTENCE', 'R_MK_BULL_TRANS', 'R_MK_STATE_SCORE',
                   'R_MK_LOW_TRANS_ENT', 'R_MK_LOW_SHANNON', 'R_MK_KL_DIV'],
        'FFT_CUSUM': ['R_FFT_LOW_FREQ', 'R_FFT_TREND_NOISE', 'R_FFT_DOM_AMP',
                      'R_CUSUM_UP', 'R_CUSUM_VOL', 'R_CUSUM_LOW_DOWN'],
    }

    # Filter to existing factors
    for fam in families:
        families[fam] = [f for f in families[fam] if f in all_factors]

    # =====================================================================
    # PHASE 1: SINGLE FACTOR TEST (best from each family)
    # =====================================================================
    print(f"\n  === PHASE 1: SINGLE FACTOR TEST ===", flush=True)
    single_results = {}

    for fam_name, factors in families.items():
        print(f"\n  --- {fam_name} Family ---", flush=True)
        best_r = None
        best_f = None
        for fname in factors:
            r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=3, rebalance_days=10, atr_stop_mult=1.5)
            if r:
                ann = r['ann']
                print(f"    {fname:<30s}: Ann={ann:+7.1f}% DD={r['max_dd']:5.1f}%", flush=True)
                if best_r is None or ann > best_r['ann']:
                    best_r = r
                    best_f = fname
        if best_f:
            single_results[fam_name] = (best_f, best_r)
            print(f"    → Best: {best_f} ({best_r['ann']:+.1f}%)", flush=True)

    # =====================================================================
    # PHASE 2: INTRA-FAMILY COMBOS (top 2 from each family)
    # =====================================================================
    print(f"\n  === PHASE 2: INTRA-FAMILY COMBOS ===", flush=True)
    family_top2 = {}
    for fam_name, factors in families.items():
        fam_results = []
        for fname in factors:
            r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=10, atr_stop_mult=1.2)
            if r:
                fam_results.append((fname, r['ann']))
        fam_results.sort(key=lambda x: -x[1])
        family_top2[fam_name] = [f[0] for f in fam_results[:2]]
        print(f"  {fam_name}: top2 = {family_top2[fam_name]}", flush=True)

    # =====================================================================
    # PHASE 3: CROSS-FAMILY COMBOS
    # =====================================================================
    print(f"\n  === PHASE 3: CROSS-FAMILY COMBOS ===", flush=True)
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

    # One-from-each-family combos
    # Geometry: TENSION, Kalman: KALMAN_VEL_PCT, HMM: REGIME_SCORE, etc.
    cross_combos = [
        # 2-family
        {'R_TENSION': 0.5, 'R_KALMAN_VEL_PCT': 0.5},
        {'R_TENSION': 0.5, 'R_HMM_REGIME_SCORE': 0.5},
        {'R_TENSION': 0.5, 'R_DMD_BULL_RATIO': 0.5},
        {'R_TENSION': 0.5, 'R_WAV_HURST': 0.5},
        {'R_BWP_BNW': 0.5, 'R_KALMAN_CONF': 0.5},
        {'R_BWP_BNW': 0.5, 'R_MK_PERSISTENCE': 0.5},
        {'R_BWP_BNW': 0.5, 'R_FFT_LOW_FREQ': 0.5},
        # 3-family
        {'R_TENSION': 0.34, 'R_KALMAN_VEL_PCT': 0.33, 'R_HMM_REGIME_SCORE': 0.33},
        {'R_TENSION': 0.34, 'R_DMD_BULL_RATIO': 0.33, 'R_WAV_HURST': 0.33},
        {'R_BWP_BNW': 0.34, 'R_KALMAN_CONF': 0.33, 'R_MK_PERSISTENCE': 0.33},
        {'R_R_SQUARED': 0.34, 'R_CUSUM_UP': 0.33, 'R_WAV_TREND_STR': 0.33},
        # 4-family
        {'R_TENSION': 0.25, 'R_KALMAN_VEL_PCT': 0.25,
         'R_DMD_BULL_RATIO': 0.25, 'R_WAV_HURST': 0.25},
        {'R_BWP_BNW': 0.25, 'R_HMM_REGIME_SCORE': 0.25,
         'R_MK_PERSISTENCE': 0.25, 'R_CUSUM_UP': 0.25},
        {'R_TENSION': 0.25, 'R_KALMAN_REGIME': 0.25,
         'R_FFT_LOW_FREQ': 0.25, 'R_MK_LOW_SHANNON': 0.25},
        # 5-family (each from different algorithm)
        {'R_TENSION': 0.2, 'R_KALMAN_VEL_PCT': 0.2,
         'R_DMD_BULL_RATIO': 0.2, 'R_WAV_TREND_STR': 0.2,
         'R_MK_PERSISTENCE': 0.2},
        {'R_BWP_BNW': 0.2, 'R_KALMAN_CONF': 0.2,
         'R_HMM_REGIME_SCORE': 0.2, 'R_FFT_LOW_FREQ': 0.2,
         'R_MK_LOW_SHANNON': 0.2},
        # 6-family
        {'R_TENSION': 0.17, 'R_KALMAN_VEL_PCT': 0.17,
         'R_HMM_REGIME_SCORE': 0.17, 'R_DMD_BULL_RATIO': 0.17,
         'R_WAV_HURST': 0.16, 'R_CUSUM_UP': 0.16},
        # 7-family (all algorithms)
        {'R_TENSION': 0.14, 'R_KALMAN_VEL_PCT': 0.14,
         'R_HMM_REGIME_SCORE': 0.14, 'R_DMD_BULL_RATIO': 0.14,
         'R_WAV_HURST': 0.14, 'R_MK_PERSISTENCE': 0.15,
         'R_FFT_LOW_FREQ': 0.15},
        # Best-from-each (use single_results)
    ]

    # Add best-from-each-family combo
    if len(single_results) >= 5:
        best_combo = {}
        for fam_name, (fname, r) in single_results.items():
            best_combo[fname] = 1.0 / len(single_results)
        cross_combos.append(best_combo)

    for i, weights in enumerate(cross_combos):
        missing = [f for f in weights if f not in all_factors]
        if missing:
            continue
        n_fam = len(weights)
        for top_n in [1, 2]:
            for atr in [1.0, 1.2, 1.5]:
                r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, rebalance_days=10, atr_stop_mult=atr)
                if r:
                    r['test'] = f'C{i}_{n_fam}fam_T{top_n}_A{atr}'
                    results.append(r)
        print(f"  Combo {i} ({n_fam} families) done", flush=True)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  TOP 50 RESULTS (V27 ALGORITHM FUSION)", flush=True)
    print(f"  {'Test':<30s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*80}", flush=True)
    for r in results[:50]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<30s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    # Best per family count
    fam_groups = {}
    for r in results:
        # Extract number of families from test name
        parts = r['test'].split('_')
        for p in parts:
            if p.endswith('fam'):
                n_fam = p.replace('fam', '')
                if n_fam not in fam_groups or r['ann'] > fam_groups[n_fam]['ann']:
                    fam_groups[n_fam] = r
                break

    print(f"\n  Best per family count:", flush=True)
    for n_fam in sorted(fam_groups.keys()):
        r = fam_groups[n_fam]
        print(f"    {n_fam} families: {r['test']:<30s} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}%", flush=True)

    # Top 5 year-by-year
    for i, r in enumerate(results[:5]):
        print(f"\n  Year-by-year #{i+1}: {r['test']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    # =====================================================================
    # KENDALL τ INDEPENDENCE CHECK between top algorithm factors
    # =====================================================================
    print(f"\n  === KENDALL τ INDEPENDENCE CHECK ===", flush=True)
    algo_factors = ['R_TENSION', 'R_KALMAN_VEL_PCT', 'R_HMM_REGIME_SCORE',
                    'R_DMD_BULL_RATIO', 'R_WAV_HURST', 'R_MK_PERSISTENCE',
                    'R_FFT_LOW_FREQ', 'R_CUSUM_UP', 'R_BWP_BNW']
    algo_factors = [f for f in algo_factors if f in all_factors]

    sample_di = min(1500, ND - 100)
    print(f"  Cross-sectional τ at di={sample_di}:", flush=True)
    for i, fa in enumerate(algo_factors):
        for j, fb in enumerate(algo_factors):
            if j <= i:
                continue
            va = all_factors[fa][:, sample_di]
            vb = all_factors[fb][:, sample_di]
            mask = ~np.isnan(va) & ~np.isnan(vb)
            if mask.sum() > 50:
                tau, p = kendalltau(va[mask], vb[mask])
                if abs(tau) > 0.15:
                    print(f"    {fa:<25s} vs {fb:<25s}: τ={tau:+.3f} p={p:.4f} {'***' if abs(tau)>0.3 else ''}", flush=True)

    print(f"\n{'='*70}", flush=True)
