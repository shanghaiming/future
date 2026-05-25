"""
Alpha V24 — Wavelet Multi-Scale Decomposition (probability_theory.md Section 23)
================================================================================
Uses pywt for wavelet decomposition. Key insight from probability_theory.md:

  "EMA是带滞后模糊镜, 小波是无滞后多倍显微镜 — 正交分解, 每层独立"

  - MRA (Multi-Resolution Analysis): cA_j (approx) + cD_j (detail) per scale
  - cA_j ≈ EMA(2^j) but NO LAG (orthogonal projection)
  - Wavelet variance: σ_j² = Var(cD_j) → Hurst exponent
  - Wavelet momentum: trend in different time scales
  - Energy ratio: which scale dominates

NO LOOK-AHEAD: Wavelet decomposition uses data up to di-1 only.
"""
import sys, os, time, warnings
import numpy as np
import pywt
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


def compute_wavelet_factors(NS, ND, C, O, H, L, V):
    """Compute wavelet multi-scale decomposition factors.

    Uses Daubechies-4 wavelet with 5 levels of decomposition.
    Rolling window of 128 trading days (~6 months).

    SELF-CHECK: d = di - 1. Decomposition uses prices[start:d+1] only.
    """
    t0 = time.time()
    new = {}

    WAVELET = 'db4'
    LEVEL = 5
    WINDOW = 128

    # Output arrays
    WAV_HURST = np.full((NS, ND), np.nan)         # Wavelet Hurst exponent
    WAV_TREND_STR = np.full((NS, ND), np.nan)      # Trend strength (cA energy)
    WAV_SHORT_MOM = np.full((NS, ND), np.nan)      # Short-scale momentum (cD1 slope)
    WAV_MED_MOM = np.full((NS, ND), np.nan)        # Medium-scale momentum (cD3 slope)
    WAV_LONG_MOM = np.full((NS, ND), np.nan)       # Long-scale momentum (cA slope)
    WAV_ENERGY_RATIO = np.full((NS, ND), np.nan)    # Trend vs noise energy ratio
    WAV_DENOISED_MOM = np.full((NS, ND), np.nan)    # Momentum from denoised signal

    for si in range(NS):
        for di in range(MIN_TRAIN, ND):
            d = di - 1  # SELF-CHECK

            start = max(0, d - WINDOW + 1)
            prices = C[si, start:d + 1]
            valid_mask = ~np.isnan(prices)
            prices = prices[valid_mask]

            if len(prices) < 64:  # Need enough data for 5-level decomposition
                continue

            # Normalize
            p_mean = np.mean(prices)
            p_std = np.std(prices)
            if p_std < 1e-10:
                continue
            prices_norm = (prices - p_mean) / p_std

            # Wavelet decomposition
            try:
                coeffs = pywt.wavedec(prices_norm, WAVELET, level=LEVEL)
                # coeffs = [cA5, cD5, cD4, cD3, cD2, cD1]
                cA = coeffs[0]  # Approximation (trend)
                cDs = coeffs[1:]  # Details at each scale
            except Exception:
                continue

            # 1. Wavelet Hurst exponent
            # log(σ_j²) ≈ (2H-1) * j * log(2)
            if len(cDs) >= 3:
                variances = []
                for j, cD in enumerate(cDs):
                    var_j = np.var(cD)
                    if var_j > 1e-10:
                        variances.append((j + 1, np.log2(var_j)))

                if len(variances) >= 3:
                    js = np.array([v[0] for v in variances])
                    log_vars = np.array([v[1] for v in variances])
                    # Linear regression: log_var = (2H-1) * j * log2 + const
                    if len(js) >= 2:
                        # Simple OLS
                        n = len(js)
                        x = js - np.mean(js)
                        y = log_vars - np.mean(log_vars)
                        ss_xx = np.sum(x ** 2)
                        if ss_xx > 1e-10:
                            slope = np.sum(x * y) / ss_xx
                            # slope ≈ (2H - 1) in log2 scale
                            H_est = (slope / 1.0 + 1) / 2.0
                            H_est = np.clip(H_est, 0.0, 1.0)
                            WAV_HURST[si, di] = H_est

            # 2. Trend strength: energy in approximation vs total
            total_energy = np.sum(cA ** 2) + sum(np.sum(cD ** 2) for cD in cDs)
            if total_energy > 1e-10:
                trend_energy = np.sum(cA ** 2)
                WAV_TREND_STR[si, di] = trend_energy / total_energy

                # Energy ratio: detail energy at each scale
                detail_energy = total_energy - trend_energy
                if detail_energy > 1e-10:
                    WAV_ENERGY_RATIO[si, di] = trend_energy / max(detail_energy, 1e-10)

            # 3. Scale-specific momentum (slope of wavelet coefficients)
            # cDs = [cD5, cD4, cD3, cD2, cD1] — index from finest to coarsest
            # cD1 = cDs[-1] (shortest ~2-4 days), cD3 = cDs[-3] (medium ~8-16 days)
            try:
                cD1 = np.atleast_1d(cDs[-1]).ravel().astype(float)
                if len(cD1) >= 5:
                    recent = cD1[-5:]
                    WAV_SHORT_MOM[si, di] = float(recent[-1] - recent[0]) / len(recent)
            except Exception:
                pass

            try:
                if len(cDs) >= 3:
                    cD3 = np.atleast_1d(cDs[-3]).ravel().astype(float)
                    if len(cD3) >= 3:
                        recent = cD3[-3:]
                        WAV_MED_MOM[si, di] = float(recent[-1] - recent[0]) / len(recent)
            except Exception:
                pass

            try:
                cA_arr = np.atleast_1d(cA).ravel().astype(float)
                if len(cA_arr) >= 3:
                    recent = cA_arr[-3:]
                    WAV_LONG_MOM[si, di] = float(recent[-1] - recent[0]) / len(recent)
            except Exception:
                pass

            # 4. Denoised momentum: reconstruct from cA + cD5 + cD4 only
            # (remove high-frequency noise cD1, cD2, cD3)
            try:
                denoised_coeffs = [cA]  # cA5
                for j in range(len(cDs)):
                    if j >= len(cDs) - 3:  # Keep cD5, cD4 (longer scales)
                        denoised_coeffs.append(cDs[j])
                    else:
                        denoised_coeffs.append(np.zeros_like(cDs[j]))

                denoised = pywt.waverec(denoised_coeffs, WAVELET)
                denoised = np.atleast_1d(denoised).ravel().astype(float)
                if len(denoised) >= 5:
                    WAV_DENOISED_MOM[si, di] = float(denoised[-1] - denoised[-5]) / 5
            except Exception:
                pass

    new['WAV_HURST'] = WAV_HURST
    new['WAV_TREND_STR'] = WAV_TREND_STR
    new['WAV_SHORT_MOM'] = WAV_SHORT_MOM
    new['WAV_MED_MOM'] = WAV_MED_MOM
    new['WAV_LONG_MOM'] = WAV_LONG_MOM
    new['WAV_ENERGY_RATIO'] = WAV_ENERGY_RATIO
    new['WAV_DENOISED_MOM'] = WAV_DENOISED_MOM

    print(f"  Wavelet decomposition done ({time.time()-t0:.1f}s)", flush=True)

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

    new['R_WAV_HURST'] = rank_pct(new['WAV_HURST'])
    new['R_WAV_TREND_STR'] = rank_pct(new['WAV_TREND_STR'])
    new['R_WAV_SHORT_MOM'] = rank_pct(new['WAV_SHORT_MOM'])
    new['R_WAV_MED_MOM'] = rank_pct(new['WAV_MED_MOM'])
    new['R_WAV_LONG_MOM'] = rank_pct(new['WAV_LONG_MOM'])
    new['R_WAV_ENERGY_RATIO'] = rank_pct(new['WAV_ENERGY_RATIO'])
    new['R_WAV_DENOISED_MOM'] = rank_pct(new['WAV_DENOISED_MOM'])

    # Composite: Hurst × Trend strength × Long momentum
    composite = np.full((NS, ND), np.nan)
    for di in range(MIN_TRAIN, ND):
        h = new['R_WAV_HURST'][:, di]
        t = new['R_WAV_TREND_STR'][:, di]
        m = new['R_WAV_LONG_MOM'][:, di]
        mask = ~np.isnan(h) & ~np.isnan(t) & ~np.isnan(m)
        if mask.sum() >= 50:
            composite[mask, di] = (h[mask] + t[mask] + m[mask]) / 3.0
    new['R_WAV_COMPOSITE'] = rank_pct(composite)

    print(f"  Total Wavelet factors: {len(new)}", flush=True)
    return new


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V24 — Wavelet Multi-Scale Decomposition", flush=True)
    print("  (probability_theory.md Section 23)", flush=True)
    print("  db4 wavelet, 5-level MRA, Hurst + trend + momentum", flush=True)
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

    # V24 Wavelet factors
    wav_factors = compute_wavelet_factors(NS, ND, C, O, H, L, V)
    all_factors = {**v11_all, **wav_factors}

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
    # WAVELET SINGLE FACTOR TESTS
    # =====================================================================
    print(f"\n  === WAVELET SINGLE FACTOR TESTS ===", flush=True)
    for fname in ['R_WAV_HURST', 'R_WAV_TREND_STR', 'R_WAV_LONG_MOM',
                  'R_WAV_DENOISED_MOM', 'R_WAV_COMPOSITE', 'R_WAV_ENERGY_RATIO']:
        r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=3, rebalance_days=10, atr_stop_mult=1.5)
        if r:
            print(f"  {fname:<25s}: Ann={r['ann']:+7.1f}% WR={r['wr']:5.1f}% "
                  f"Edge={r['edge']:+5.2f}% DD={r['max_dd']:5.1f}%", flush=True)

    # =====================================================================
    # WAVELET COMBINATION TESTS
    # =====================================================================
    portfolios = {
        # Wavelet composite + structure
        'WC_tens': {'R_WAV_COMPOSITE': 0.3, 'R_TENSION': 0.3,
                    'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # Wavelet Hurst + BwpBNW
        'WH_bwp': {'R_WAV_HURST': 0.3, 'R_BWP_BNW': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # Wavelet denoised momentum + momentum
        'WD_mom': {'R_WAV_DENOISED_MOM': 0.3, 'R_MOM5': 0.3,
                   'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        # Wavelet trend strength + squeeze
        'WT_sqz': {'R_WAV_TREND_STR': 0.3, 'R_BB_WIDTH_PCT_INV': 0.3,
                   'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        # Wavelet energy ratio + Kalman
        'WE_KV': {'R_WAV_ENERGY_RATIO': 0.25, 'R_KALMAN_VEL_PCT': 0.25,
                  'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        # Wavelet Hurst + standard Hurst (cross-validation)
        'WH_hurst': {'R_WAV_HURST': 0.25, 'R_HURST': 0.25,
                     'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        # Wavelet long momentum + NW slope (trend from different methods)
        'WL_NW': {'R_WAV_LONG_MOM': 0.25, 'R_NW_SLOPE': 0.25,
                  'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        # Triple Wavelet
        'W3': {'R_WAV_HURST': 0.2, 'R_WAV_TREND_STR': 0.2,
               'R_WAV_LONG_MOM': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        # Wavelet + DMD (both spectral methods)
        'WD_DMD': {'R_WAV_COMPOSITE': 0.25, 'R_DMD_BULL_RATIO': 0.25,
                   'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        # Wavelet + KER
        'WK_ker': {'R_WAV_TREND_STR': 0.25, 'R_KER': 0.25,
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
    print(f"  TOP 40 RESULTS (V24 WAVELET)", flush=True)
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
