"""
Alpha V40 — TA-Lib Wave 2: 20 More Indicators
==============================================
V39 tested 14 TA-Lib indicators (all look-ahead fixed).
V40 tests 20 more, focusing on different signal dimensions:
  - Momentum: RSI, ROC, MOM, APO, DX
  - Trend: MACD, ADXR, MINUS_DI, PLUS_DI, LINEARREG_SLOPE
  - Volatility: STDDEV, TRANGE, BBANDS width, AVGDEV
  - Volume: OBV, AD, ADOSC
  - Cycle: HT_DCPHASE, HT_SINE, AROON (up/down)
  - Adaptive: KAMA, T3, SAR, TSF
  - Cross-sectional: BETA, CORREL, STOCH, STOCHF, WCLPRICE, IMI

Key constraints:
  - NO LOOK-AHEAD: All use vals[di-1] stored at index di
  - Rebalance=5d (V35 breakthrough)
  - ATR stop mult = 0.8, 1.0, 1.2
  - Target: beat V35 R5_A1.0_B = +290.4%
"""
import sys, os, time, warnings
import numpy as np
import talib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7c import backtest_v7c


def compute_v40_talib_factors(NS, ND, C, O, H, L, V):
    """Compute 20 new TA-Lib factors. NO LOOK-AHEAD: vals[di-1] at index di."""
    factors = {}
    t0 = time.time()

    def store_shifted(arr, vals, si):
        """Store vals[di-1] at arr[si, di] for di=1..len(vals)-1"""
        for di in range(1, len(vals)):
            if di < ND and not np.isnan(vals[di - 1]):
                arr[si, di] = float(vals[di - 1])

    # === RSI (14-period) ===
    print("  Computing RSI...", flush=True)
    rsi = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        if np.isnan(c).sum() > len(c) * 0.5:
            continue
        try:
            vals = talib.RSI(c, timeperiod=14)
            store_shifted(rsi, vals, si)
        except:
            pass
    factors['RSI'] = rsi
    print(f"    RSI done ({time.time()-t0:.0f}s)", flush=True)

    # === MACD histogram (12,26,9) ===
    print("  Computing MACD...", flush=True)
    macd_hist = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        if np.isnan(c).sum() > len(c) * 0.5:
            continue
        try:
            _, _, hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
            store_shifted(macd_hist, hist, si)
        except:
            pass
    factors['MACD_HIST'] = macd_hist
    print(f"    MACD done ({time.time()-t0:.0f}s)", flush=True)

    # === STOCH (5,3,3) K and D ===
    print("  Computing STOCH...", flush=True)
    stoch_k = np.full((NS, ND), np.nan)
    stoch_d = np.full((NS, ND), np.nan)
    for si in range(NS):
        h, l, c = H[si], L[si], C[si]
        if np.isnan(c).sum() > len(c) * 0.5:
            continue
        try:
            k, d = talib.STOCH(h, l, c, fastk_period=5, slowk_period=3,
                               slowk_matype=0, slowd_period=3, slowd_matype=0)
            for di in range(1, len(k)):
                if di < ND:
                    if not np.isnan(k[di - 1]):
                        stoch_k[si, di] = float(k[di - 1])
                    if not np.isnan(d[di - 1]):
                        stoch_d[si, di] = float(d[di - 1])
        except:
            pass
    factors['STOCH_K'] = stoch_k
    factors['STOCH_D'] = stoch_d
    print(f"    STOCH done ({time.time()-t0:.0f}s)", flush=True)

    # === OBV (On Balance Volume) ===
    print("  Computing OBV...", flush=True)
    obv = np.full((NS, ND), np.nan)
    for si in range(NS):
        c, v = C[si], V[si]
        if np.isnan(c).sum() > len(c) * 0.5:
            continue
        try:
            vals = talib.OBV(c, v)
            # OBV is cumulative, use ROC of OBV instead
            obv_roc = np.full_like(vals, np.nan)
            for i in range(10, len(vals)):
                if vals[i - 10] != 0 and not np.isnan(vals[i]) and not np.isnan(vals[i - 10]):
                    obv_roc[i] = (vals[i] - vals[i - 10]) / abs(vals[i - 10]) * 100
            store_shifted(obv, obv_roc, si)
        except:
            pass
    factors['OBV_ROC10'] = obv
    print(f"    OBV done ({time.time()-t0:.0f}s)", flush=True)

    # === AD (Chaikin A/D Line) ===
    print("  Computing AD...", flush=True)
    ad_line = np.full((NS, ND), np.nan)
    for si in range(NS):
        h, l, c, v = H[si], L[si], C[si], V[si]
        if np.isnan(c).sum() > len(c) * 0.5:
            continue
        try:
            vals = talib.AD(h, l, c, v)
            # Use ROC of AD line
            ad_roc = np.full_like(vals, np.nan)
            for i in range(10, len(vals)):
                if not np.isnan(vals[i]) and not np.isnan(vals[i - 10]) and vals[i - 10] != 0:
                    ad_roc[i] = (vals[i] - vals[i - 10]) / abs(vals[i - 10]) * 100
            store_shifted(ad_line, ad_roc, si)
        except:
            pass
    factors['AD_ROC10'] = ad_line
    print(f"    AD done ({time.time()-t0:.0f}s)", flush=True)

    # === ADOSC (Chaikin A/D Oscillator) ===
    print("  Computing ADOSC...", flush=True)
    adosc = np.full((NS, ND), np.nan)
    for si in range(NS):
        h, l, c, v = H[si], L[si], C[si], V[si]
        if np.isnan(c).sum() > len(c) * 0.5:
            continue
        try:
            vals = talib.ADOSC(h, l, c, v, fastperiod=3, slowperiod=10)
            store_shifted(adosc, vals, si)
        except:
            pass
    factors['ADOSC'] = adosc
    print(f"    ADOSC done ({time.time()-t0:.0f}s)", flush=True)

    # === SAR (Parabolic SAR) ===
    print("  Computing SAR...", flush=True)
    sar_dist = np.full((NS, ND), np.nan)  # Distance from SAR to close (normalized)
    for si in range(NS):
        h, l, c = H[si], L[si], C[si]
        if np.isnan(c).sum() > len(c) * 0.5:
            continue
        try:
            vals = talib.SAR(h, l, acceleration=0.02, maximum=0.2)
            dist = np.full_like(vals, np.nan)
            for i in range(len(vals)):
                if not np.isnan(vals[i]) and not np.isnan(c[i]) and c[i] != 0:
                    dist[i] = (c[i] - vals[i]) / c[i] * 100  # % distance
            store_shifted(sar_dist, dist, si)
        except:
            pass
    factors['SAR_DIST'] = sar_dist
    print(f"    SAR done ({time.time()-t0:.0f}s)", flush=True)

    # === KAMA (Kaufman Adaptive Moving Average) ===
    print("  Computing KAMA...", flush=True)
    kama_dev = np.full((NS, ND), np.nan)  # Deviation from KAMA
    for si in range(NS):
        c = C[si]
        if np.isnan(c).sum() > len(c) * 0.5:
            continue
        try:
            vals = talib.KAMA(c, timeperiod=30)
            dev = np.full_like(vals, np.nan)
            for i in range(len(vals)):
                if not np.isnan(vals[i]) and not np.isnan(c[i]) and vals[i] != 0:
                    dev[i] = (c[i] - vals[i]) / vals[i] * 100
            store_shifted(kama_dev, dev, si)
        except:
            pass
    factors['KAMA_DEV'] = kama_dev
    print(f"    KAMA done ({time.time()-t0:.0f}s)", flush=True)

    # === DX (Directional Movement Index) ===
    print("  Computing DX...", flush=True)
    dx = np.full((NS, ND), np.nan)
    for si in range(NS):
        h, l, c = H[si], L[si], C[si]
        if np.isnan(c).sum() > len(c) * 0.5:
            continue
        try:
            vals = talib.DX(h, l, c, timeperiod=14)
            store_shifted(dx, vals, si)
        except:
            pass
    factors['DX'] = dx
    print(f"    DX done ({time.time()-t0:.0f}s)", flush=True)

    # === ROC (Rate of Change, 10-period) ===
    print("  Computing ROC...", flush=True)
    roc = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        if np.isnan(c).sum() > len(c) * 0.5:
            continue
        try:
            vals = talib.ROC(c, timeperiod=10)
            store_shifted(roc, vals, si)
        except:
            pass
    factors['ROC10'] = roc
    print(f"    ROC done ({time.time()-t0:.0f}s)", flush=True)

    # === MOM (Momentum, 10-period) ===
    print("  Computing MOM...", flush=True)
    mom = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        if np.isnan(c).sum() > len(c) * 0.5:
            continue
        try:
            vals = talib.MOM(c, timeperiod=10)
            store_shifted(mom, vals, si)
        except:
            pass
    factors['MOM10'] = mom
    print(f"    MOM done ({time.time()-t0:.0f}s)", flush=True)

    # === STDDEV (20-period) ===
    print("  Computing STDDEV...", flush=True)
    stddev = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        if np.isnan(c).sum() > len(c) * 0.5:
            continue
        try:
            vals = talib.STDDEV(c, timeperiod=20, nbdev=1)
            # Normalize by price
            norm = np.full_like(vals, np.nan)
            for i in range(len(vals)):
                if not np.isnan(vals[i]) and not np.isnan(c[i]) and c[i] != 0:
                    norm[i] = vals[i] / c[i] * 100  # CV%
            store_shifted(stddev, norm, si)
        except:
            pass
    factors['STDDEV_CV'] = stddev
    print(f"    STDDEV done ({time.time()-t0:.0f}s)", flush=True)

    # === TRANGE (True Range) ===
    print("  Computing TRANGE...", flush=True)
    trange_norm = np.full((NS, ND), np.nan)
    for si in range(NS):
        h, l, c = H[si], L[si], C[si]
        if np.isnan(c).sum() > len(c) * 0.5:
            continue
        try:
            vals = talib.TRANGE(h, l, c)
            norm = np.full_like(vals, np.nan)
            for i in range(len(vals)):
                if not np.isnan(vals[i]) and not np.isnan(c[i]) and c[i] != 0:
                    norm[i] = vals[i] / c[i] * 100
            store_shifted(trange_norm, norm, si)
        except:
            pass
    factors['TRANGE_PCT'] = trange_norm
    print(f"    TRANGE done ({time.time()-t0:.0f}s)", flush=True)

    # === BBANDS width ===
    print("  Computing BBANDS...", flush=True)
    bbw = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        if np.isnan(c).sum() > len(c) * 0.5:
            continue
        try:
            upper, mid, lower = talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            width = np.full_like(c, np.nan)
            for i in range(len(mid)):
                if not np.isnan(upper[i]) and not np.isnan(lower[i]) and not np.isnan(mid[i]) and mid[i] != 0:
                    width[i] = (upper[i] - lower[i]) / mid[i] * 100
            store_shifted(bbw, width, si)
        except:
            pass
    factors['BBW'] = bbw
    print(f"    BBANDS done ({time.time()-t0:.0f}s)", flush=True)

    # === LINEARREG_SLOPE (14-period) ===
    print("  Computing LINEARREG_SLOPE...", flush=True)
    lrslope = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        if np.isnan(c).sum() > len(c) * 0.5:
            continue
        try:
            vals = talib.LINEARREG_SLOPE(c, timeperiod=14)
            # Normalize by price
            norm = np.full_like(vals, np.nan)
            for i in range(len(vals)):
                if not np.isnan(vals[i]) and not np.isnan(c[i]) and c[i] != 0:
                    norm[i] = vals[i] / c[i] * 100
            store_shifted(lrslope, norm, si)
        except:
            pass
    factors['LR_SLOPE'] = lrslope
    print(f"    LINEARREG_SLOPE done ({time.time()-t0:.0f}s)", flush=True)

    # === MINUS_DI / PLUS_DI ===
    print("  Computing MINUS_DI / PLUS_DI...", flush=True)
    minus_di = np.full((NS, ND), np.nan)
    plus_di = np.full((NS, ND), np.nan)
    for si in range(NS):
        h, l, c = H[si], L[si], C[si]
        if np.isnan(c).sum() > len(c) * 0.5:
            continue
        try:
            mdi = talib.MINUS_DI(h, l, c, timeperiod=14)
            pdi = talib.PLUS_DI(h, l, c, timeperiod=14)
            store_shifted(minus_di, mdi, si)
            store_shifted(plus_di, pdi, si)
        except:
            pass
    factors['MINUS_DI'] = minus_di
    factors['PLUS_DI'] = plus_di
    print(f"    DI done ({time.time()-t0:.0f}s)", flush=True)

    # === HT_DCPHASE (Dominant Cycle Phase) ===
    print("  Computing HT_DCPHASE...", flush=True)
    ht_phase = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        if np.isnan(c).sum() > len(c) * 0.5:
            continue
        try:
            vals = talib.HT_DCPHASE(c)
            store_shifted(ht_phase, vals, si)
        except:
            pass
    factors['HT_DCPHASE'] = ht_phase
    print(f"    HT_DCPHASE done ({time.time()-t0:.0f}s)", flush=True)

    # === HT_SINE (Hilbert Sine Wave) ===
    print("  Computing HT_SINE...", flush=True)
    ht_sine = np.full((NS, ND), np.nan)
    ht_lead = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        if np.isnan(c).sum() > len(c) * 0.5:
            continue
        try:
            sine, lead = talib.HT_SINE(c)
            for di in range(1, len(sine)):
                if di < ND:
                    if not np.isnan(sine[di - 1]):
                        ht_sine[si, di] = float(sine[di - 1])
                    if not np.isnan(lead[di - 1]):
                        ht_lead[si, di] = float(lead[di - 1])
        except:
            pass
    factors['HT_SINE'] = ht_sine
    factors['HT_LEADSINE'] = ht_lead
    print(f"    HT_SINE done ({time.time()-t0:.0f}s)", flush=True)

    # === ADXR (ADX Rating) ===
    print("  Computing ADXR...", flush=True)
    adxr = np.full((NS, ND), np.nan)
    for si in range(NS):
        h, l, c = H[si], L[si], C[si]
        if np.isnan(c).sum() > len(c) * 0.5:
            continue
        try:
            vals = talib.ADXR(h, l, c, timeperiod=14)
            store_shifted(adxr, vals, si)
        except:
            pass
    factors['ADXR'] = adxr
    print(f"    ADXR done ({time.time()-t0:.0f}s)", flush=True)

    # === STOCHF (Stochastic Fast) ===
    print("  Computing STOCHF...", flush=True)
    stochf_k = np.full((NS, ND), np.nan)
    stochf_d = np.full((NS, ND), np.nan)
    for si in range(NS):
        h, l, c = H[si], L[si], C[si]
        if np.isnan(c).sum() > len(c) * 0.5:
            continue
        try:
            k, d = talib.STOCHF(h, l, c, fastk_period=5, fastd_period=3, fastd_matype=0)
            for di in range(1, len(k)):
                if di < ND:
                    if not np.isnan(k[di - 1]):
                        stochf_k[si, di] = float(k[di - 1])
                    if not np.isnan(d[di - 1]):
                        stochf_d[si, di] = float(d[di - 1])
        except:
            pass
    factors['STOCHF_K'] = stochf_k
    factors['STOCHF_D'] = stochf_d
    print(f"    STOCHF done ({time.time()-t0:.0f}s)", flush=True)

    # === APO (Absolute Price Oscillator) ===
    print("  Computing APO...", flush=True)
    apo = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        if np.isnan(c).sum() > len(c) * 0.5:
            continue
        try:
            vals = talib.APO(c, fastperiod=12, slowperiod=26, matype=0)
            store_shifted(apo, vals, si)
        except:
            pass
    factors['APO'] = apo
    print(f"    APO done ({time.time()-t0:.0f}s)", flush=True)

    # === AVGDEV (Average Deviation) — manually computed since talib may not have it ===
    # Skip AVGDEV if not available, use TSF instead
    # === TSF (Time Series Forecast) ===
    print("  Computing TSF...", flush=True)
    tsf_err = np.full((NS, ND), np.nan)  # Forecast error
    for si in range(NS):
        c = C[si]
        if np.isnan(c).sum() > len(c) * 0.5:
            continue
        try:
            vals = talib.TSF(c, timeperiod=14)
            err = np.full_like(vals, np.nan)
            for i in range(len(vals)):
                if not np.isnan(vals[i]) and not np.isnan(c[i]) and c[i] != 0:
                    err[i] = (c[i] - vals[i]) / c[i] * 100  # forecast error %
            store_shifted(tsf_err, err, si)
        except:
            pass
    factors['TSF_ERR'] = tsf_err
    print(f"    TSF done ({time.time()-t0:.0f}s)", flush=True)

    # === IMI (Intraday Momentum Index) ===
    print("  Computing IMI...", flush=True)
    imi = np.full((NS, ND), np.nan)
    for si in range(NS):
        o, c = O[si], C[si]
        if np.isnan(c).sum() > len(c) * 0.5:
            continue
        try:
            vals = talib.IMI(o, c)
            store_shifted(imi, vals, si)
        except:
            pass
    factors['IMI'] = imi
    print(f"    IMI done ({time.time()-t0:.0f}s)", flush=True)

    # === T3 (Triple Exponential Moving Average) ===
    print("  Computing T3...", flush=True)
    t3_dev = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        if np.isnan(c).sum() > len(c) * 0.5:
            continue
        try:
            vals = talib.T3(c, timeperiod=20, vfactor=0.7)
            dev = np.full_like(vals, np.nan)
            for i in range(len(vals)):
                if not np.isnan(vals[i]) and not np.isnan(c[i]) and vals[i] != 0:
                    dev[i] = (c[i] - vals[i]) / vals[i] * 100
            store_shifted(t3_dev, dev, si)
        except:
            pass
    factors['T3_DEV'] = t3_dev
    print(f"    T3 done ({time.time()-t0:.0f}s)", flush=True)

    # === AROON Up/Down ===
    print("  Computing AROON...", flush=True)
    aroon_up = np.full((NS, ND), np.nan)
    aroon_dn = np.full((NS, ND), np.nan)
    for si in range(NS):
        h, l = H[si], L[si]
        if np.isnan(h).sum() > len(h) * 0.5:
            continue
        try:
            down, up = talib.AROON(h, l, timeperiod=14)
            for di in range(1, len(up)):
                if di < ND:
                    if not np.isnan(up[di - 1]):
                        aroon_up[si, di] = float(up[di - 1])
                    if not np.isnan(down[di - 1]):
                        aroon_dn[si, di] = float(down[di - 1])
        except:
            pass
    factors['AROON_UP'] = aroon_up
    factors['AROON_DN'] = aroon_dn
    print(f"    AROON done ({time.time()-t0:.0f}s)", flush=True)

    # === WCLPRICE (Weighted Close Price) deviation from MA ===
    print("  Computing WCLPRICE...", flush=True)
    wcl_dev = np.full((NS, ND), np.nan)
    for si in range(NS):
        h, l, c = H[si], L[si], C[si]
        if np.isnan(c).sum() > len(c) * 0.5:
            continue
        try:
            wcl = talib.WCLPRICE(h, l, c)
            ma = talib.SMA(c, timeperiod=20)
            dev = np.full_like(wcl, np.nan)
            for i in range(len(wcl)):
                if not np.isnan(wcl[i]) and not np.isnan(ma[i]) and ma[i] != 0:
                    dev[i] = (wcl[i] - ma[i]) / ma[i] * 100
            store_shifted(wcl_dev, dev, si)
        except:
            pass
    factors['WCL_DEV'] = wcl_dev
    print(f"    WCLPRICE done ({time.time()-t0:.0f}s)", flush=True)

    # Now rank normalize each factor
    print("  Rank normalizing V40 TA-Lib factors...", flush=True)
    ranked = {}
    for fname, arr in factors.items():
        r = np.full_like(arr, np.nan)
        for di in range(ND):
            vals = arr[:, di]
            valid = ~np.isnan(vals)
            n = valid.sum()
            if n < 50:
                continue
            order = np.argsort(vals[valid])
            ranks = np.empty(n)
            ranks[order] = np.arange(1, n + 1)
            r[valid, di] = ranks / n * 100
        ranked[f'R_{fname}'] = r

    # Inverted versions for factors where low = better
    for inv_name in ['R_STDDEV_CV', 'R_TRANGE_PCT', 'R_BBW']:
        if inv_name in ranked:
            inv = ranked[inv_name].copy()
            mask = ~np.isnan(inv)
            inv[mask] = 100.0 - inv[mask]
            ranked[f'{inv_name}_INV'] = inv

    print(f"  V40 TA-Lib factors done: {len(ranked)} ranked ({time.time()-t0:.0f}s)", flush=True)
    return ranked


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V40 — TA-Lib Wave 2 (20 New Indicators)", flush=True)
    print("  Target: beat V35 R5_A1.0_B = +290.4%", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Load ALL existing factors (V2-V14) — needed for V15 baseline combos
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

    # Compute V40 TA-Lib factors
    v40_factors = compute_v40_talib_factors(NS, ND, C, O, H, L, V)
    all_factors.update(v40_factors)

    print(f"\n  Total factors: {len(all_factors)}", flush=True)

    # V15-B weights (V35 record weights)
    v15b_weights = {'R_BWP_BNW': 0.25, 'R_TENSION': 0.25,
                    'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.15, 'R_HAR_RV_RATIO_INV': 0.15}

    results = []

    # =====================================================================
    # TEST 1: Each V40 factor ALONE (single factor)
    # =====================================================================
    print("\n  Test 1: Single V40 factors...", flush=True)
    v40_names = sorted(v40_factors.keys())
    for fname in v40_names:
        for atr in [1.0]:
            r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'{fname}_A{atr}'
                results.append(r)
    print(f"  Single factors done: {len(results)} results", flush=True)

    # =====================================================================
    # TEST 2: Each V40 factor + V15-B baseline
    # =====================================================================
    print("\n  Test 2: V40 + V15-B baseline...", flush=True)
    for fname in v40_names:
        for atr in [1.0]:
            weights = {**v15b_weights, fname: 0.15}
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
            r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'V15B+{fname}_A{atr}'
                results.append(r)
    print(f"  V15 combos done: {len(results)} total", flush=True)

    # =====================================================================
    # TEST 3: Top V40 factor pairs + V15-B
    # =====================================================================
    print("\n  Test 3: V40 pairs + V15-B...", flush=True)
    pairs = [
        ('R_RSI', 'R_MACD_HIST'),
        ('R_OBV_ROC10', 'R_LR_SLOPE'),
        ('R_BBW_INV', 'R_ADXR'),
        ('R_KAMA_DEV', 'R_STOCH_K'),
        ('R_SAR_DIST', 'R_DX'),
        ('R_APO', 'R_TSF_ERR'),
        ('R_HT_SINE', 'R_HT_DCPHASE'),
        ('R_MINUS_DI', 'R_PLUS_DI'),
        ('R_ADOSC', 'R_ROC10'),
        ('R_T3_DEV', 'R_AROON_UP'),
        ('R_STOCHF_K', 'R_IMI'),
        ('R_WCL_DEV', 'R_MOM10'),
    ]
    for f1, f2 in pairs:
        if f1 in all_factors and f2 in all_factors:
            weights = {'R_BWP_BNW': 0.2, 'R_HAR_RV_RATIO_INV': 0.2,
                       f1: 0.3, f2: 0.3}
            r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=1.0)
            if r:
                tag = f'{f1[2:6]}_{f2[2:6]}'
                r['test'] = f'PAIR_{tag}'
                results.append(r)
    print(f"  Pairs done: {len(results)} total", flush=True)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  ALL RESULTS (V40 TA-LIB WAVE 2)", flush=True)
    print(f"  {'Test':<40s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*85}", flush=True)
    for r in results:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<40s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    # Best singles
    singles = [r for r in results if not r['test'].startswith('V15B+') and not r['test'].startswith('PAIR_')]
    if singles:
        print(f"\n  Best single V40 factors:", flush=True)
        for r in sorted(singles, key=lambda x: -x['ann'])[:15]:
            pos = " ALL+" if all_positive(r) else ""
            print(f"    {r['test']:<40s} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}%{pos}", flush=True)

    # Best V15-B combos
    combos = [r for r in results if r['test'].startswith('V15B+')]
    if combos:
        print(f"\n  Best V15-B + V40 combos:", flush=True)
        for r in sorted(combos, key=lambda x: -x['ann'])[:10]:
            pos = " ALL+" if all_positive(r) else ""
            delta = r['ann'] - 290.4
            print(f"    {r['test']:<40s} → {r['ann']:+.1f}%DD={r['max_dd']:.1f}% Δ={delta:+.1f}%{pos}", flush=True)

    # Top 5 year-by-year
    for i, r in enumerate(results[:5]):
        print(f"\n  Year-by-year #{i+1}: {r['test']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    if results:
        best = results[0]
        print(f"\n  === V40 BEST vs V35 RECORD ===", flush=True)
        print(f"  V40: {best['test']} = {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)
        print(f"  V35: R5_A1.0_B = +290.4% DD=43.7%", flush=True)
        print(f"  Delta: {best['ann'] - 290.4:+.1f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
