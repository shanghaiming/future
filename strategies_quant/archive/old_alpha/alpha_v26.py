"""
Alpha V26 — FFT Spectral Analysis + CUSUM Signal Detection
============================================================
From probability_theory.md Sections 9 & 30:

Section 30: FFT Spectral Analysis
  - DFT: X_k = Σ x_n * exp(-i2πkn/N)
  - Power spectrum PSD_k = |X_k|²/N
  - Dominant period detection
  - Low/high frequency energy ratio (trend vs noise)

Section 9: Sequential Analysis
  - CUSUM: S_t = max(0, S_{t-1} + (x_t - μ_0 - k)), signal when S_t > h
  - Cumulative evidence accumulation — don't act on single bar noise
  - "耐心是概率优势的必要条件: 累积足够的证据才行动"

Strategy:
  1. FFT dominant period + amplitude per stock
  2. Low-frequency energy ratio (trend strength)
  3. CUSUM momentum detection (cumulative positive returns)
  4. CUSUM volatility shift detection

NO LOOK-AHEAD: All computations use data up to di-1 only.
"""
import sys, os, time, warnings
import numpy as np
from scipy import signal as scipy_signal
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


def compute_fft_cusum_factors(NS, ND, C, O, H, L, V):
    """Compute FFT spectral + CUSUM sequential detection factors.

    SELF-CHECK: d = di - 1. All data up to d only.
    """
    t0 = time.time()
    new = {}

    FFT_WINDOW = 120   # ~6 months
    CUSUM_WINDOW = 60  # ~3 months

    # Output arrays
    FFT_DOMINANT_PERIOD = np.full((NS, ND), np.nan)
    FFT_DOMINANT_AMP = np.full((NS, ND), np.nan)
    FFT_LOW_FREQ_RATIO = np.full((NS, ND), np.nan)
    FFT_TREND_NOISE = np.full((NS, ND), np.nan)
    CUSUM_UP = np.full((NS, ND), np.nan)
    CUSUM_DOWN = np.full((NS, ND), np.nan)
    CUSUM_VOL_SHIFT = np.full((NS, ND), np.nan)

    for si in range(NS):
        cusum_up = 0.0
        cusum_down = 0.0
        cusum_vol = 0.0

        for di in range(MIN_TRAIN, ND):
            d = di - 1  # SELF-CHECK

            prices = C[si, max(0, d - FFT_WINDOW + 1):d + 1]
            valid_mask = ~np.isnan(prices)
            prices = prices[valid_mask]

            if len(prices) < 64:
                continue

            # =====================================================================
            # FFT Spectral Analysis
            # =====================================================================
            # Detrend + window
            detrended = scipy_signal.detrend(prices, type='linear')
            demeaned = detrended - np.mean(detrended)

            w = len(demeaned)
            hann = scipy_signal.windows.hann(w)
            windowed = demeaned * hann

            # FFT
            fft_result = np.fft.rfft(windowed)
            freqs = np.fft.rfftfreq(w, d=1)
            psd = np.abs(fft_result) ** 2 / w

            # Dominant period (skip DC component at index 0)
            if len(psd) > 2 and np.max(psd[1:]) > 0:
                # Find peaks
                peaks, props = scipy_signal.find_peaks(
                    psd[1:], prominence=np.quantile(psd[1:], 0.75))
                if len(peaks) > 0:
                    dominant_idx = peaks[np.argmax(psd[1:][peaks])] + 1
                    period = 1.0 / freqs[dominant_idx]
                    amp = np.abs(fft_result[dominant_idx]) * 2 / w
                    FFT_DOMINANT_PERIOD[si, di] = min(period, 200)
                    FFT_DOMINANT_AMP[si, di] = amp

            # Low-frequency energy ratio
            # Split at period ~20 days (freq = 0.05)
            total_psd = np.sum(psd[1:])
            if total_psd > 1e-10:
                low_freq_mask = (freqs > 0) & (freqs <= 0.05)  # Period >= 20 days
                low_psd = np.sum(psd[low_freq_mask])
                FFT_LOW_FREQ_RATIO[si, di] = low_psd / total_psd

                # Trend/noise: ratio of lowest 3 frequencies to highest 3
                n_freq = len(psd[1:])
                if n_freq >= 6:
                    sorted_psd = np.sort(psd[1:])[::-1]  # Descending
                    top_3 = np.sum(sorted_psd[:3])
                    bottom_3 = np.sum(sorted_psd[-3:])
                    if bottom_3 > 1e-10:
                        FFT_TREND_NOISE[si, di] = top_3 / bottom_3

            # =====================================================================
            # CUSUM Sequential Detection
            # =====================================================================
            returns_all = np.diff(prices) / prices[:-1]
            if len(returns_all) < 10:
                continue

            # Use last CUSUM_WINDOW days
            ret_recent = returns_all[-CUSUM_WINDOW:]
            mu_0 = np.mean(returns_all[:30]) if len(returns_all) >= 30 else 0.0

            # CUSUM for positive shift (buy signal)
            k = 0.005  # Minimum detectable shift
            cusum_up_val = 0.0
            max_cusum_up = 0.0
            for r in ret_recent:
                cusum_up_val = max(0, cusum_up_val + (r - mu_0 - k))
                max_cusum_up = max(max_cusum_up, cusum_up_val)
            CUSUM_UP[si, di] = max_cusum_up

            # CUSUM for negative shift (sell signal)
            cusum_down_val = 0.0
            max_cusum_down = 0.0
            for r in ret_recent:
                cusum_down_val = max(0, cusum_down_val - (r - mu_0 + k))
                max_cusum_down = max(max_cusum_down, cusum_down_val)
            CUSUM_DOWN[si, di] = max_cusum_down

            # CUSUM for volatility shift
            if len(returns_all) >= 30:
                vol_baseline = np.std(returns_all[:30])
                vol_recent = np.abs(ret_recent - np.mean(ret_recent))

                cusum_vol_val = 0.0
                max_cusum_vol = 0.0
                k_vol = vol_baseline * 0.5
                for v in vol_recent:
                    cusum_vol_val = max(0, cusum_vol_val + (v - vol_baseline - k_vol))
                    max_cusum_vol = max(max_cusum_vol, cusum_vol_val)
                CUSUM_VOL_SHIFT[si, di] = max_cusum_vol

    new['FFT_DOMINANT_PERIOD'] = FFT_DOMINANT_PERIOD
    new['FFT_DOMINANT_AMP'] = FFT_DOMINANT_AMP
    new['FFT_LOW_FREQ_RATIO'] = FFT_LOW_FREQ_RATIO
    new['FFT_TREND_NOISE'] = FFT_TREND_NOISE
    new['CUSUM_UP'] = CUSUM_UP
    new['CUSUM_DOWN'] = CUSUM_DOWN
    new['CUSUM_VOL_SHIFT'] = CUSUM_VOL_SHIFT

    print(f"  FFT + CUSUM factors done ({time.time()-t0:.1f}s)", flush=True)

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

    new['R_FFT_LOW_FREQ'] = rank_pct(new['FFT_LOW_FREQ_RATIO'])
    new['R_FFT_TREND_NOISE'] = rank_pct(new['FFT_TREND_NOISE'])
    new['R_FFT_DOM_AMP'] = rank_pct(new['FFT_DOMINANT_AMP'])
    new['R_FFT_PERIOD'] = rank_pct(new['FFT_DOMINANT_PERIOD'])
    new['R_CUSUM_UP'] = rank_pct(new['CUSUM_UP'])
    new['R_CUSUM_VOL'] = rank_pct(new['CUSUM_VOL_SHIFT'])

    # Invert CUSUM_DOWN (low = good = no bearish signal)
    inv_down = new['CUSUM_DOWN'].copy()
    mask = ~np.isnan(inv_down)
    if mask.any():
        mn, mx = np.nanmin(inv_down), np.nanmax(inv_down)
        if mx > mn:
            inv_down[mask] = mx - inv_down[mask] + mn
    new['R_CUSUM_LOW_DOWN'] = rank_pct(inv_down)

    print(f"  Total FFT+CUSUM factors: {len(new)}", flush=True)
    return new


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V26 — FFT Spectral + CUSUM Sequential Detection", flush=True)
    print("  (probability_theory.md Sections 9 & 30)", flush=True)
    print("  FFT dominant period + CUSUM momentum/vol shift", flush=True)
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

    # V26 FFT+CUSUM factors
    fft_factors = compute_fft_cusum_factors(NS, ND, C, O, H, L, V)
    all_factors = {**v11_all, **fft_factors}

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
    print(f"\n  === FFT+CUSUM SINGLE FACTOR TESTS ===", flush=True)
    for fname in ['R_FFT_LOW_FREQ', 'R_FFT_TREND_NOISE', 'R_FFT_DOM_AMP',
                  'R_CUSUM_UP', 'R_CUSUM_VOL', 'R_CUSUM_LOW_DOWN']:
        r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=3, rebalance_days=10, atr_stop_mult=1.5)
        if r:
            print(f"  {fname:<25s}: Ann={r['ann']:+7.1f}% WR={r['wr']:5.1f}% "
                  f"Edge={r['edge']:+5.2f}% DD={r['max_dd']:5.1f}%", flush=True)

    # =====================================================================
    # COMBINATION TESTS
    # =====================================================================
    portfolios = {
        # FFT low freq + structure
        'FL_tens': {'R_FFT_LOW_FREQ': 0.3, 'R_TENSION': 0.3,
                    'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # CUSUM up + BwpBNW
        'FC_bwp': {'R_CUSUM_UP': 0.3, 'R_BWP_BNW': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # FFT trend/noise + momentum
        'FT_mom': {'R_FFT_TREND_NOISE': 0.3, 'R_MOM5': 0.3,
                   'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        # CUSUM up + Kalman (sequential evidence + adaptive)
        'FK_vel': {'R_CUSUM_UP': 0.25, 'R_KALMAN_VEL_PCT': 0.25,
                   'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        # FFT + DMD (both spectral)
        'FD_DMD': {'R_FFT_LOW_FREQ': 0.25, 'R_DMD_BULL_RATIO': 0.25,
                   'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        # CUSUM + Wavelet (sequential + multi-scale)
        'FC_WAV': {'R_CUSUM_UP': 0.25, 'R_WAV_TREND_STR': 0.25,
                   'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        # Triple FFT+CUSUM
        'F3': {'R_FFT_LOW_FREQ': 0.2, 'R_CUSUM_UP': 0.2,
               'R_FFT_TREND_NOISE': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        # CUSUM low down + squeeze (no bearish signal + squeeze)
        'FS_sqz': {'R_CUSUM_LOW_DOWN': 0.3, 'R_BB_WIDTH_PCT_INV': 0.3,
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
    print(f"  TOP 40 RESULTS (V26 FFT+CUSUM)", flush=True)
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
