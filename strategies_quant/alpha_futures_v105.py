"""
Alpha Futures V105 -- TA-Lib Volume + Volatility + Cycle Indicators
====================================================================
Current best next-open: +49.6% (50-day breakout).

V105 FOCUS: Systematically test TA-Lib volume, volatility, and cycle indicators
with NEXT-OPEN execution (signal at close di, entry at O[si, di+1]).

Indicators tested:
A) MFI (Money Flow Index) - volume-weighted RSI
B) OBV (On Balance Volume) trend
C) AD/ADOSC (Accumulation/Distribution)
D) NATR (Normalized ATR) - volatility breakout
E) TRANGE (True Range) - volatility
F) BOP (Balance of Power)
G) HT_DCPERIOD + HT_TRENDMODE (Hilbert Transform)
H) HT_SINE (Hilbert Sine Wave)
I) MAMA/FAMA (Mesa Adaptive Moving Average)
J) KAMA (Kaufman Adaptive Moving Average)
K) T3 (Triple Exponential Moving Average)
L) APO/PPO (Price Oscillator)
M) DX (Directional Movement Index)
N) MULTI-SIGNAL: NATR spike + MFI oversold recovery + bullish close

All computed with talib directly on numpy arrays.
Long-only. COMM=0.0003. Walk-forward 2020-2025.
"""
import sys, os, time, warnings
import numpy as np
import talib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

# ============================================================
# CONSTANTS
# ============================================================
MULT = {'agfi': 15, 'alfi': 5, 'aufi': 1000, 'bufi': 10, 'cufi': 5, 'fufi': 10,
        'rbfi': 10, 'znfi': 5, 'nifi': 1, 'hcfi': 10, 'spfi': 10, 'ssfi': 5,
        'sffi': 5, 'smfi': 5, 'pbfi': 5, 'snfi': 1, 'rufi': 10, 'wrffi': 10,
        'afi': 10, 'bfi': 10, 'bbfi': 500, 'cffi': 5, 'cfi': 10, 'csfi': 10,
        'ebfi': 5, 'egfi': 10, 'fbfi': 500, 'ifi': 100, 'jfi': 100, 'jmfi': 60,
        'lfi': 5, 'mfi': 10, 'pgfi': 20, 'ppfi': 5, 'vfi': 5, 'yfi': 10,
        'pfi': 10, 'jdfi': 5, 'lhfi': 16, 'pkfi': 5, 'rrfi': 20, 'lrfi': 20,
        'jrfi': 20, 'pmfi': 20, 'whfi': 20, 'rsfi': 20, 'cjfi': 10, 'mafi': 10,
        'apfi': 10, 'cyfi': 5, 'fgfi': 20, 'oifi': 10, 'pfifi': 5, 'rmfi': 10,
        'srfi': 10, 'tafi': 5, 'safi': 20, 'urfi': 20, 'scfi': 1000, 'lufi': 10,
        'bcfi': 5, 'nrfi': 1, 'lgfi': 20, 'brfi': 5, 'lcfi': 1, 'sifi': 5,
        'ni': 1, 'tai': 5}
DEF_MULT = 10
COMM = 0.0003


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def compute_indicator_per_sym(indicator_func):
    """Compute a talib indicator for each symbol, handling NaN arrays.
    Returns array of shape (NS, ND) with NaN where input was invalid."""
    pass  # placeholder, each indicator handled inline


def main():
    print("=" * 180)
    print("Alpha Futures V105 -- TA-Lib Volume + Volatility + Cycle Indicators")
    print("=" * 180)
    print("\n  ALL signals computed at close di, entry at O[si, di+1] (NEXT DAY OPEN)")
    print("  Testing 14 indicator groups with 3 hold periods each.")

    # -- Load data -------------------------------------------------
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ================================================================
    # PRECOMPUTE ALL TA-LIB INDICATORS
    # ================================================================
    print("\n[Indicators] Computing TA-Lib indicators...", flush=True)
    t0 = time.time()

    # We store indicators in dicts keyed by si, with arrays of length ND
    # Using full arrays (NS, ND) filled with NaN

    # ---- A) MFI (Money Flow Index) ----
    MFI = np.full((NS, ND), np.nan)
    for si in range(NS):
        h = H[si]; l = L[si]; c = C[si]; v = V[si]
        valid = ~(np.isnan(h) | np.isnan(l) | np.isnan(c) | np.isnan(v))
        if np.sum(valid) < 30:
            continue
        idx = np.where(valid)[0]
        try:
            mfi_vals = talib.MFI(h[idx], l[idx], c[idx], v[idx], timeperiod=14)
            MFI[si, idx] = mfi_vals
        except:
            pass
    print(f"  A) MFI computed ({time.time()-t0:.1f}s)")

    # ---- B) OBV (On Balance Volume) ----
    OBV = np.full((NS, ND), np.nan)
    OBV_SMA20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]; v = V[si]
        valid = ~(np.isnan(c) | np.isnan(v))
        if np.sum(valid) < 30:
            continue
        idx = np.where(valid)[0]
        try:
            obv_vals = talib.OBV(c[idx], v[idx])
            OBV[si, idx] = obv_vals
            # SMA of OBV
            sma = talib.SMA(obv_vals, timeperiod=20)
            OBV_SMA20[si, idx] = sma
        except:
            pass
    print(f"  B) OBV + OBV_SMA20 computed ({time.time()-t0:.1f}s)")

    # ---- C) AD and ADOSC ----
    AD = np.full((NS, ND), np.nan)
    AD_SMA20 = np.full((NS, ND), np.nan)
    ADOSC_arr = np.full((NS, ND), np.nan)
    for si in range(NS):
        h = H[si]; l = L[si]; c = C[si]; v = V[si]
        valid = ~(np.isnan(h) | np.isnan(l) | np.isnan(c) | np.isnan(v))
        if np.sum(valid) < 30:
            continue
        idx = np.where(valid)[0]
        try:
            ad_vals = talib.AD(h[idx], l[idx], c[idx], v[idx])
            AD[si, idx] = ad_vals
            sma = talib.SMA(ad_vals, timeperiod=20)
            AD_SMA20[si, idx] = sma
            adosc_vals = talib.ADOSC(h[idx], l[idx], c[idx], v[idx], fastperiod=3, slowperiod=10)
            ADOSC_arr[si, idx] = adosc_vals
        except:
            pass
    print(f"  C) AD + ADOSC computed ({time.time()-t0:.1f}s)")

    # ---- D) NATR (Normalized ATR) ----
    NATR = np.full((NS, ND), np.nan)
    NATR_SMA20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        h = H[si]; l = L[si]; c = C[si]
        valid = ~(np.isnan(h) | np.isnan(l) | np.isnan(c))
        if np.sum(valid) < 30:
            continue
        idx = np.where(valid)[0]
        try:
            natr_vals = talib.NATR(h[idx], l[idx], c[idx], timeperiod=14)
            NATR[si, idx] = natr_vals
            sma = talib.SMA(natr_vals, timeperiod=20)
            NATR_SMA20[si, idx] = sma
        except:
            pass
    print(f"  D) NATR + NATR_SMA20 computed ({time.time()-t0:.1f}s)")

    # ---- E) TRANGE (True Range) ----
    TRANGE = np.full((NS, ND), np.nan)
    TRANGE_SMA20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        h = H[si]; l = L[si]; c = C[si]
        valid = ~(np.isnan(h) | np.isnan(l) | np.isnan(c))
        if np.sum(valid) < 30:
            continue
        idx = np.where(valid)[0]
        try:
            tr_vals = talib.TRANGE(h[idx], l[idx], c[idx])
            TRANGE[si, idx] = tr_vals
            sma = talib.SMA(tr_vals, timeperiod=20)
            TRANGE_SMA20[si, idx] = sma
        except:
            pass
    print(f"  E) TRANGE + TRANGE_SMA20 computed ({time.time()-t0:.1f}s)")

    # ---- F) BOP (Balance of Power) ----
    BOP = np.full((NS, ND), np.nan)
    for si in range(NS):
        o = O[si]; h = H[si]; l = L[si]; c = C[si]
        valid = ~(np.isnan(o) | np.isnan(h) | np.isnan(l) | np.isnan(c))
        # Also need h != l
        idx = np.where(valid)[0]
        if len(idx) < 10:
            continue
        try:
            bop_vals = talib.BOP(o[idx], h[idx], l[idx], c[idx])
            BOP[si, idx] = bop_vals
        except:
            pass
    print(f"  F) BOP computed ({time.time()-t0:.1f}s)")

    # ---- G) HT_DCPERIOD + HT_TRENDMODE ----
    HT_DCPERIOD = np.full((NS, ND), np.nan)
    HT_TRENDMODE = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        valid = ~np.isnan(c)
        idx = np.where(valid)[0]
        if len(idx) < 64:  # Hilbert needs ~64 bars
            continue
        try:
            dcp = talib.HT_DCPERIOD(c[idx])
            HT_DCPERIOD[si, idx] = dcp
            tm = talib.HT_TRENDMODE(c[idx])
            HT_TRENDMODE[si, idx] = tm
        except:
            pass
    print(f"  G) HT_DCPERIOD + HT_TRENDMODE computed ({time.time()-t0:.1f}s)")

    # ---- H) HT_SINE (Hilbert Sine Wave) ----
    HT_SINE = np.full((NS, ND), np.nan)
    HT_LEADSINE = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        valid = ~np.isnan(c)
        idx = np.where(valid)[0]
        if len(idx) < 64:
            continue
        try:
            sine, leadsine = talib.HT_SINE(c[idx])
            HT_SINE[si, idx] = sine
            HT_LEADSINE[si, idx] = leadsine
        except:
            pass
    print(f"  H) HT_SINE + HT_LEADSINE computed ({time.time()-t0:.1f}s)")

    # ---- I) MAMA/FAMA ----
    MAMA = np.full((NS, ND), np.nan)
    FAMA = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        valid = ~np.isnan(c)
        idx = np.where(valid)[0]
        if len(idx) < 50:
            continue
        try:
            mama, fama = talib.MAMA(c[idx], fastlimit=0.5, slowlimit=0.05)
            MAMA[si, idx] = mama
            FAMA[si, idx] = fama
        except:
            pass
    print(f"  I) MAMA/FAMA computed ({time.time()-t0:.1f}s)")

    # ---- J) KAMA ----
    KAMA = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        valid = ~np.isnan(c)
        idx = np.where(valid)[0]
        if len(idx) < 30:
            continue
        try:
            kama_vals = talib.KAMA(c[idx], timeperiod=30)
            KAMA[si, idx] = kama_vals
        except:
            pass
    print(f"  J) KAMA computed ({time.time()-t0:.1f}s)")

    # ---- K) T3 ----
    T3 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        valid = ~np.isnan(c)
        idx = np.where(valid)[0]
        if len(idx) < 50:
            continue
        try:
            t3_vals = talib.T3(c[idx], timeperiod=20, vfactor=0.7)
            T3[si, idx] = t3_vals
        except:
            pass
    print(f"  K) T3 computed ({time.time()-t0:.1f}s)")

    # ---- L) PPO ----
    PPO_arr = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        valid = ~np.isnan(c)
        idx = np.where(valid)[0]
        if len(idx) < 30:
            continue
        try:
            ppo_vals = talib.PPO(c[idx], fastperiod=12, slowperiod=26, matype=0)
            PPO_arr[si, idx] = ppo_vals
        except:
            pass
    print(f"  L) PPO computed ({time.time()-t0:.1f}s)")

    # ---- M) DX + PLUS_DI + MINUS_DI ----
    DX_arr = np.full((NS, ND), np.nan)
    PLUS_DI = np.full((NS, ND), np.nan)
    MINUS_DI = np.full((NS, ND), np.nan)
    for si in range(NS):
        h = H[si]; l = L[si]; c = C[si]
        valid = ~(np.isnan(h) | np.isnan(l) | np.isnan(c))
        idx = np.where(valid)[0]
        if len(idx) < 30:
            continue
        try:
            dx = talib.DX(h[idx], l[idx], c[idx], timeperiod=14)
            DX_arr[si, idx] = dx
            pdi = talib.PLUS_DI(h[idx], l[idx], c[idx], timeperiod=14)
            PLUS_DI[si, idx] = pdi
            mdi = talib.MINUS_DI(h[idx], l[idx], c[idx], timeperiod=14)
            MINUS_DI[si, idx] = mdi
        except:
            pass
    print(f"  M) DX + PLUS_DI + MINUS_DI computed ({time.time()-t0:.1f}s)")

    print(f"\n  All indicators computed ({time.time()-t_start:.1f}s total)")

    # ================================================================
    # SIGNAL GENERATION (at close of day di)
    # ================================================================
    print("\n[Signals] Computing all signals...", flush=True)
    t1 = time.time()

    # Dict to store signal arrays: key -> (signal_bool_array, score_array)
    signals = {}

    # ---- A) MFI signals ----
    # A1: MFI crosses above 20 (oversold recovery)
    sig_mfi_cross20 = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(1, ND):
            m0 = MFI[si, di]
            m1 = MFI[si, di - 1]
            if np.isnan(m0) or np.isnan(m1):
                continue
            if m0 > 20 and m1 <= 20:
                sig_mfi_cross20[si, di] = True
    signals['mfi_cross20'] = sig_mfi_cross20
    print(f"  A1) MFI cross above 20: {np.sum(sig_mfi_cross20)} signals")

    # A2: MFI < 30 (oversold level)
    sig_mfi_oversold = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(ND):
            m = MFI[si, di]
            if np.isnan(m):
                continue
            if m < 30:
                sig_mfi_oversold[si, di] = True
    signals['mfi_oversold'] = sig_mfi_oversold
    print(f"  A2) MFI < 30 oversold: {np.sum(sig_mfi_oversold)} signals")

    # ---- B) OBV crosses above SMA20 ----
    sig_obv_cross = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(1, ND):
            o0 = OBV[si, di]; o1 = OBV[si, di - 1]
            s0 = OBV_SMA20[si, di]; s1 = OBV_SMA20[si, di - 1]
            if np.isnan(o0) or np.isnan(o1) or np.isnan(s0) or np.isnan(s1):
                continue
            if o0 > s0 and o1 <= s1:
                sig_obv_cross[si, di] = True
    signals['obv_cross'] = sig_obv_cross
    print(f"  B) OBV cross above SMA20: {np.sum(sig_obv_cross)} signals")

    # ---- C1) AD crosses above SMA20 ----
    sig_ad_cross = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(1, ND):
            a0 = AD[si, di]; a1 = AD[si, di - 1]
            s0 = AD_SMA20[si, di]; s1 = AD_SMA20[si, di - 1]
            if np.isnan(a0) or np.isnan(a1) or np.isnan(s0) or np.isnan(s1):
                continue
            if a0 > s0 and a1 <= s1:
                sig_ad_cross[si, di] = True
    signals['ad_cross'] = sig_ad_cross
    print(f"  C1) AD cross above SMA20: {np.sum(sig_ad_cross)} signals")

    # ---- C2) ADOSC crosses above 0 ----
    sig_adosc_cross = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(1, ND):
            a0 = ADOSC_arr[si, di]; a1 = ADOSC_arr[si, di - 1]
            if np.isnan(a0) or np.isnan(a1):
                continue
            if a0 > 0 and a1 <= 0:
                sig_adosc_cross[si, di] = True
    signals['adosc_cross'] = sig_adosc_cross
    print(f"  C2) ADOSC cross above 0: {np.sum(sig_adosc_cross)} signals")

    # ---- D) NATR spike: NATR > 2*SMA(NATR,20) AND C > O ----
    sig_natr_spike = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(ND):
            n_val = NATR[si, di]; s_val = NATR_SMA20[si, di]
            c = C[si, di]; o = O[si, di]
            if np.isnan(n_val) or np.isnan(s_val) or np.isnan(c) or np.isnan(o):
                continue
            if s_val > 0 and n_val > 2 * s_val and c > o:
                sig_natr_spike[si, di] = True
    signals['natr_spike'] = sig_natr_spike
    print(f"  D) NATR spike + bullish: {np.sum(sig_natr_spike)} signals")

    # ---- E) TRANGE spike: TRANGE > 2*SMA(TRANGE,20) AND C > O ----
    sig_trange_spike = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(ND):
            tr = TRANGE[si, di]; ts = TRANGE_SMA20[si, di]
            c = C[si, di]; o = O[si, di]
            if np.isnan(tr) or np.isnan(ts) or np.isnan(c) or np.isnan(o):
                continue
            if ts > 0 and tr > 2 * ts and c > o:
                sig_trange_spike[si, di] = True
    signals['trange_spike'] = sig_trange_spike
    print(f"  E) TRANGE spike + bullish: {np.sum(sig_trange_spike)} signals")

    # ---- F) BOP thresholds: 0.3, 0.5, 0.7 ----
    for threshold in [0.3, 0.5, 0.7]:
        sig_bop = np.zeros((NS, ND), dtype=bool)
        for si in range(NS):
            for di in range(ND):
                b = BOP[si, di]
                if np.isnan(b):
                    continue
                if b > threshold:
                    sig_bop[si, di] = True
        key = f'bop_{int(threshold*10)}'
        signals[key] = sig_bop
        print(f"  F) BOP > {threshold}: {np.sum(sig_bop)} signals")

    # ---- G) HT_DCPERIOD < 20 AND HT_TRENDMODE == 1 ----
    sig_ht_trend = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(ND):
            dcp = HT_DCPERIOD[si, di]; tm = HT_TRENDMODE[si, di]
            if np.isnan(dcp) or np.isnan(tm):
                continue
            if dcp < 20 and tm == 1:
                sig_ht_trend[si, di] = True
    signals['ht_trend'] = sig_ht_trend
    print(f"  G) HT_DCPERIOD<20 + TRENDMODE=1: {np.sum(sig_ht_trend)} signals")

    # ---- H) HT_SINE crosses above HT_LEADSINE ----
    sig_ht_sine = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(1, ND):
            s0 = HT_SINE[si, di]; s1 = HT_SINE[si, di - 1]
            l0 = HT_LEADSINE[si, di]; l1 = HT_LEADSINE[si, di - 1]
            if np.isnan(s0) or np.isnan(s1) or np.isnan(l0) or np.isnan(l1):
                continue
            if s0 > l0 and s1 <= l1:
                sig_ht_sine[si, di] = True
    signals['ht_sine'] = sig_ht_sine
    print(f"  H) HT_SINE cross above LeadSine: {np.sum(sig_ht_sine)} signals")

    # ---- I) MAMA crosses above FAMA ----
    sig_mama = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(1, ND):
            m0 = MAMA[si, di]; m1 = MAMA[si, di - 1]
            f0 = FAMA[si, di]; f1 = FAMA[si, di - 1]
            if np.isnan(m0) or np.isnan(m1) or np.isnan(f0) or np.isnan(f1):
                continue
            if m0 > f0 and m1 <= f1:
                sig_mama[si, di] = True
    signals['mama_cross'] = sig_mama
    print(f"  I) MAMA cross above FAMA: {np.sum(sig_mama)} signals")

    # ---- J) Price crosses above KAMA ----
    sig_kama = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(1, ND):
            c0 = C[si, di]; c1 = C[si, di - 1]
            k0 = KAMA[si, di]; k1 = KAMA[si, di - 1]
            if np.isnan(c0) or np.isnan(c1) or np.isnan(k0) or np.isnan(k1):
                continue
            if c0 > k0 and c1 <= k1:
                sig_kama[si, di] = True
    signals['kama_cross'] = sig_kama
    print(f"  J) Price cross above KAMA: {np.sum(sig_kama)} signals")

    # ---- K) Price crosses above T3 ----
    sig_t3 = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(1, ND):
            c0 = C[si, di]; c1 = C[si, di - 1]
            t0 = T3[si, di]; t1 = T3[si, di - 1]
            if np.isnan(c0) or np.isnan(c1) or np.isnan(t0) or np.isnan(t1):
                continue
            if c0 > t0 and c1 <= t1:
                sig_t3[si, di] = True
    signals['t3_cross'] = sig_t3
    print(f"  K) Price cross above T3: {np.sum(sig_t3)} signals")

    # ---- L) PPO crosses above 0 ----
    sig_ppo = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(1, ND):
            p0 = PPO_arr[si, di]; p1 = PPO_arr[si, di - 1]
            if np.isnan(p0) or np.isnan(p1):
                continue
            if p0 > 0 and p1 <= 0:
                sig_ppo[si, di] = True
    signals['ppo_cross'] = sig_ppo
    print(f"  L) PPO cross above 0: {np.sum(sig_ppo)} signals")

    # ---- M) DX > 30 AND PLUS_DI > MINUS_DI ----
    sig_dx = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(ND):
            dx = DX_arr[si, di]; pdi = PLUS_DI[si, di]; mdi = MINUS_DI[si, di]
            if np.isnan(dx) or np.isnan(pdi) or np.isnan(mdi):
                continue
            if dx > 30 and pdi > mdi:
                sig_dx[si, di] = True
    signals['dx_strong'] = sig_dx
    print(f"  M) DX>30 + PLUS_DI>MINUS_DI: {np.sum(sig_dx)} signals")

    # ---- N) MULTI-SIGNAL: NATR spike + MFI oversold recovery + C > O ----
    sig_multi = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(ND):
            # NATR > 1.5x average
            n_val = NATR[si, di]; s_val = NATR_SMA20[si, di]
            if np.isnan(n_val) or np.isnan(s_val) or s_val <= 0:
                continue
            if n_val <= 1.5 * s_val:
                continue
            # MFI oversold recovery: MFI crossed above 20 recently (within 3 days)
            mfi_recovered = False
            if di >= 2:
                m_now = MFI[si, di]
                m_prev = MFI[si, di - 1]
                m_prev2 = MFI[si, di - 2]
                if (not np.isnan(m_now) and m_now > 20 and
                    not np.isnan(m_prev) and
                    (m_prev <= 20 or (not np.isnan(m_prev2) and m_prev2 <= 20))):
                    mfi_recovered = True
            elif di >= 1:
                m_now = MFI[si, di]
                m_prev = MFI[si, di - 1]
                if not np.isnan(m_now) and m_now > 20 and not np.isnan(m_prev) and m_prev <= 20:
                    mfi_recovered = True
            if not mfi_recovered:
                continue
            # Bullish close
            c = C[si, di]; o = O[si, di]
            if np.isnan(c) or np.isnan(o) or c <= o:
                continue
            sig_multi[si, di] = True
    signals['multi_vol'] = sig_multi
    print(f"  N) MULTI (NATR+MFI+bullish): {np.sum(sig_multi)} signals")

    print(f"\n  All signals computed ({time.time()-t1:.1f}s)")

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(config, wf_test_year=None):
        """
        Config:
            signal: str (key into signals dict)
            hold_days: int
            top_n: int (max concurrent positions)
            comm: float
        """
        sig_type = config['signal']
        hold_days = config['hold_days']
        top_n = config.get('top_n', 3)
        comm = config.get('comm', COMM)

        sig_arr = signals.get(sig_type)
        if sig_arr is None:
            return None

        # Date boundaries
        if wf_test_year is not None:
            test_start_di = None
            test_end_di = None
            for di in range(ND):
                if dates[di].year == wf_test_year and test_start_di is None:
                    test_start_di = di
                if dates[di].year == wf_test_year + 1 and test_end_di is None:
                    test_end_di = di
            if test_start_di is None:
                return None
            if test_end_di is None:
                test_end_di = ND
            start_di = MIN_TRAIN
            end_di = test_end_di
        else:
            test_start_di = MIN_TRAIN
            start_di = MIN_TRAIN
            end_di = ND
            test_end_di = ND

        if end_di < start_di + hold_days + 2:
            return None

        cash = float(CASH0)
        positions = []
        trades = []

        for di in range(start_di, end_di - 1):
            # Reset cash at test window start (WF mode)
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                positions = []

            # -- Close positions -----------------------------------------
            closed = []
            for pos in positions:
                days_held = di - pos['entry_di']
                if days_held >= pos['hold_days']:
                    exit_price = C[pos['si'], di]
                    if np.isnan(exit_price) or exit_price <= 0:
                        exit_price = pos['entry_price']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    mkt_val = exit_price * mult * abs(pos['lots'])
                    cash += mkt_val - mkt_val * comm
                    pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
                    invested = pos['entry_price'] * mult * abs(pos['lots'])
                    pnl_pct = pnl / invested * 100 if invested > 0 else 0
                    trades.append({
                        'pnl_pct': pnl_pct,
                        'entry_di': pos['entry_di'],
                        'exit_di': di,
                        'year': dates[di].year if di < ND else dates[-1].year,
                        'sym': pos.get('sym', ''),
                        'days_held': days_held,
                    })
                    closed.append(pos)

            for pos in closed:
                positions.remove(pos)

            # If we have max positions, skip new entries
            if len(positions) >= top_n:
                continue

            # -- Generate signals at day di --------------------------------
            entry_di = di + 1
            if entry_di >= end_di:
                continue

            candidates = []
            for si in range(NS):
                if not sig_arr[si, di]:
                    continue
                if any(p['si'] == si for p in positions):
                    continue
                ep = O[si, entry_di]
                if np.isnan(ep) or ep <= 0:
                    continue
                # Score: use close price momentum as tiebreaker
                c0 = C[si, di]
                c1 = C[si, di - 5] if di >= 5 else np.nan
                score = 0
                if not np.isnan(c0) and not np.isnan(c1) and c1 > 0:
                    score = (c0 - c1) / c1 * 100
                candidates.append((score, {
                    'si': si, 'sym': syms[si], 'entry_price': ep,
                }))

            if not candidates:
                continue

            # Sort by score descending
            candidates.sort(key=lambda x: -x[0])

            # Open positions
            n_slots = top_n - len(positions)
            for sc, info in candidates[:max(0, n_slots)]:
                si = info['si']
                sym = info['sym']
                price = info['entry_price']
                mult = MULT.get(sym, DEF_MULT)
                notional = price * mult

                # Equal allocation across top_n slots
                alloc = cash / max(1, top_n - len(positions))
                lots = max(1, int(alloc / (notional * (1 + comm))))
                if lots <= 0:
                    continue
                cost_in = notional * lots * (1 + comm)
                if cost_in > cash:
                    lots = max(1, int(cash * 0.9 / (notional * (1 + comm))))
                    cost_in = notional * lots * (1 + comm) if lots > 0 else 0
                if lots <= 0 or cost_in <= 0 or cost_in > cash:
                    continue

                cash -= cost_in

                positions.append({
                    'si': si, 'entry_price': price, 'entry_di': entry_di,
                    'lots': lots, 'dir': 1, 'sym': sym,
                    'hold_days': hold_days,
                })

        # Close remaining positions at end
        for pos in positions:
            ae = end_di - 1 if end_di < ND else ND - 1
            exit_price = C[pos['si'], ae]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos['entry_price']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = exit_price * mult * abs(pos['lots'])
            cash += mkt_val - mkt_val * comm
            pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
            invested = pos['entry_price'] * mult * abs(pos['lots'])
            pnl_pct = pnl / invested * 100 if invested > 0 else 0
            trades.append({
                'pnl_pct': pnl_pct,
                'entry_di': pos['entry_di'],
                'exit_di': ae,
                'year': dates[ae].year if ae < ND else dates[-1].year,
                'sym': pos.get('sym', ''),
                'days_held': ae - pos['entry_di'],
            })

        # Results
        wf_mode = wf_test_year is not None
        n_days_test = (test_end_di - test_start_di) if wf_mode else (end_di - start_di)
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0
        avg_hold = np.mean([t['days_held'] for t in trades]) if trades else 0

        # Max drawdown from equity curve
        eq = float(CASH0)
        peak = eq
        mdd = 0.0
        for t in trades:
            eq *= (1 + t['pnl_pct'] / 100)
            if eq > peak:
                peak = eq
            dd = (eq - peak) / peak * 100
            if dd < mdd:
                mdd = dd

        freq_per_yr = n_trades / (n_days_test / 252) if n_days_test > 0 else 0

        return {
            'ann': ann, 'wr': wr, 'n': n_trades, 'avg_pnl': avg_pnl,
            'final_cash': cash, 'n_days': n_days_test, 'mdd': mdd,
            'avg_hold': avg_hold, 'freq': freq_per_yr,
        }

    # ================================================================
    # BUILD CONFIGURATIONS
    # ================================================================
    print("\n[Sweep] Building configurations...", flush=True)
    configs = []
    cid = 0

    signal_keys = list(signals.keys())
    for sig_key in signal_keys:
        for hd in [5, 10, 20]:
            for tn in [3]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': sig_key,
                    'hold_days': hd, 'top_n': tn, 'comm': COMM,
                    'label': f"{sig_key}_H{hd}_TN{tn}",
                })

    # Also test top_n=1 for select signals
    select_signals = ['mfi_cross20', 'natr_spike', 'mama_cross', 'kama_cross',
                      'ppo_cross', 'dx_strong', 'multi_vol', 'bop_5']
    for sig_key in select_signals:
        if sig_key in signals:
            for hd in [5, 10, 20]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': sig_key,
                    'hold_days': hd, 'top_n': 1, 'comm': COMM,
                    'label': f"{sig_key}_H{hd}_TN1",
                })

    print(f"  Total configs: {len(configs)}")

    # ================================================================
    # RUN FULL-PERIOD BACKTEST
    # ================================================================
    print("\n[Backtest] Running full-period sweep...", flush=True)
    results = []
    for i, cfg in enumerate(configs):
        r = run_backtest(cfg)
        if r:
            r['config'] = cfg
            r['label'] = cfg['label']
            results.append(r)
        if (i + 1) % 20 == 0 or i == len(configs) - 1:
            print(f"  ... {i+1}/{len(configs)} done ({time.time()-t_start:.0f}s)", flush=True)

    results.sort(key=lambda x: -x['ann'])

    # ================================================================
    # SIGNAL GROUP LABELS
    # ================================================================
    group_labels = {
        'mfi_cross20': 'A1) MFI cross above 20',
        'mfi_oversold': 'A2) MFI < 30',
        'obv_cross': 'B) OBV cross above SMA20',
        'ad_cross': 'C1) AD cross above SMA20',
        'adosc_cross': 'C2) ADOSC cross above 0',
        'natr_spike': 'D) NATR spike + bullish',
        'trange_spike': 'E) TRANGE spike + bullish',
        'bop_3': 'F1) BOP > 0.3',
        'bop_5': 'F2) BOP > 0.5',
        'bop_7': 'F3) BOP > 0.7',
        'ht_trend': 'G) HT_DCPERIOD<20 + TRENDMODE=1',
        'ht_sine': 'H) HT_SINE cross above LeadSine',
        'mama_cross': 'I) MAMA cross above FAMA',
        'kama_cross': 'J) Price cross above KAMA',
        't3_cross': 'K) Price cross above T3',
        'ppo_cross': 'L) PPO cross above 0',
        'dx_strong': 'M) DX>30 + PLUS_DI>MINUS_DI',
        'multi_vol': 'N) MULTI (NATR+MFI+bullish)',
    }

    # ================================================================
    # FULL-PERIOD RESULTS
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  FULL-PERIOD RESULTS (All configs) -- ALL NEXT-OPEN EXECUTION")
    print(f"{'=' * 180}")
    print(f"  {'#':>3} | {'Label':<30} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'AvgHold':>7} | {'Freq/Yr':>7} | {'Final':>14}")
    print("-" * 170)
    for i, r in enumerate(results):
        print(f"  {i+1:>3} | {r['label']:<30} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}% | {r['avg_hold']:>6.1f}d | {r['freq']:>6.1f}/yr | {r['final_cash']:>13,.0f}")

    # ================================================================
    # BEST PER SIGNAL GROUP
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  BEST PER SIGNAL GROUP (Full Period)")
    print(f"{'=' * 180}")
    print(f"  {'Signal':<42} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'AvgHold':>7} | {'Freq/Yr':>7} | Best Config")
    print("-" * 180)

    # Group by signal base name (strip _H\d+_TN\d)
    def base_signal(label):
        parts = label.rsplit('_H', 1)
        return parts[0]

    best_per_sig = {}
    for r in results:
        key = r['config']['signal']
        if key not in best_per_sig:
            best_per_sig[key] = r

    for sig_key in signal_keys:
        if sig_key in best_per_sig:
            b = best_per_sig[sig_key]
            label = group_labels.get(sig_key, sig_key)
            print(f"  {label:<42} | {b['ann']:>+8.1f}% | {b['wr']:>5.1f}% | {b['n']:>5} | {b['avg_pnl']:>+6.3f}% | {b['mdd']:>6.1f}% | {b['avg_hold']:>6.1f}d | {b['freq']:>6.1f}/yr | {b['label']}")

    # ================================================================
    # SIGNAL GROUP SUMMARY
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  SIGNAL GROUP SUMMARY (Average of all configs per group)")
    print(f"{'=' * 180}")
    print(f"  {'Signal':<42} | {'Avg Ann':>9} | {'Avg WR':>7} | {'Avg N':>7} | {'Avg PnL':>8} | {'Avg MDD':>8} | {'#Positive':>9}")
    print("-" * 160)

    for sig_key in signal_keys:
        sub = [r for r in results if r['config']['signal'] == sig_key]
        if not sub:
            continue
        avg_ann = np.mean([r['ann'] for r in sub])
        avg_wr = np.mean([r['wr'] for r in sub])
        avg_n = np.mean([r['n'] for r in sub])
        avg_pnl = np.mean([r['avg_pnl'] for r in sub])
        avg_mdd = np.mean([r['mdd'] for r in sub])
        n_pos = sum(1 for r in sub if r['ann'] > 0)
        label = group_labels.get(sig_key, sig_key)
        print(f"  {label:<42} | {avg_ann:>+8.1f}% | {avg_wr:>6.1f}% | {avg_n:>7.0f} | {avg_pnl:>+7.3f}% | {avg_mdd:>7.1f}% | {n_pos:>5}/{len(sub)}")

    # ================================================================
    # WALK-FORWARD (Top 20 configs)
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Collect top 20 overall + best per signal group
    wf_configs = list(results[:20])
    for sig_key in signal_keys:
        if sig_key in best_per_sig:
            r = best_per_sig[sig_key]
            if r['config'] not in [w['config'] for w in wf_configs]:
                wf_configs.append(r)

    print(f"\n{'=' * 210}")
    print(f"  WALK-FORWARD ({len(wf_configs)} configs)")
    print(f"{'=' * 210}")

    header = f"  {'#':>3} | {'Config':<30} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>7} |"
    header += f" {'Pos':>4} | {'MDD':>7} | {'WR':>6}"
    print(header)
    print("-" * 210)

    wf_rows = []
    for i, r in enumerate(wf_configs):
        cfg = r['config']
        wf_row = {'label': cfg['label'], 'signal': cfg['signal'],
                  'entry': 'next_open', 'windows': {}, 'mdd': {}, 'wr': {}}
        for yr in wf_years:
            wr = run_backtest(cfg, wf_test_year=yr)
            if wr:
                wf_row['windows'][yr] = wr['ann']
                wf_row['mdd'][yr] = wr['mdd']
                wf_row['wr'][yr] = wr['wr']
        wf_rows.append(wf_row)

        vals = [wf_row['windows'].get(yr, 0) for yr in wf_years]
        avg = np.mean(vals) if vals else 0
        pos = sum(1 for v in vals if v > 0)
        avg_mdd = np.mean(list(wf_row['mdd'].values())) if wf_row['mdd'] else 0
        avg_wr = np.mean(list(wf_row['wr'].values())) if wf_row['wr'] else 0

        row_str = f"  {i+1:>3} | {wf_row['label']:<30} | {avg:>+7.1f}% |"
        for v in vals:
            row_str += f" {v:>+7.1f}% |"
        row_str += f" {pos}/6 | {avg_mdd:>6.1f}% | {avg_wr:>5.1f}%"
        print(row_str)

    # ================================================================
    # WF COMPARISON PER SIGNAL GROUP
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  WALK-FORWARD COMPARISON (Best per signal group)")
    print(f"{'=' * 180}")
    header2 = f"  {'Signal':<42} | {'WF Avg':>8} |"
    for yr in wf_years:
        header2 += f" {yr:>7} |"
    header2 += f" {'Pos':>4} | Avg MDD | Avg WR"
    print(header2)
    print("-" * 180)

    for sig_key in signal_keys:
        wf_match = [w for w in wf_rows if w['signal'] == sig_key]
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            avg = np.mean(vals) if vals else 0
            pos = sum(1 for v in vals if v > 0)
            avg_mdd = np.mean(list(wf['mdd'].values())) if wf['mdd'] else 0
            avg_wr = np.mean(list(wf['wr'].values())) if wf['wr'] else 0
            label = group_labels.get(sig_key, sig_key)
            row_str = f"  {label:<42} | {avg:>+7.1f}% |"
            for v in vals:
                row_str += f" {v:>+7.1f}% |"
            row_str += f" {pos}/6 | {avg_mdd:>6.1f}% | {avg_wr:>5.1f}%"
            print(row_str)

    # ================================================================
    # HOLD PERIOD ANALYSIS
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  HOLD PERIOD ANALYSIS (Average across all signal groups)")
    print(f"{'=' * 180}")
    print(f"  {'Hold':>6} | {'Avg Ann':>9} | {'Avg WR':>7} | {'Avg N':>7} | {'Avg MDD':>8} | {'#Positive':>9}")
    print("-" * 80)

    for hd in [5, 10, 20]:
        sub = [r for r in results if r['config']['hold_days'] == hd]
        if not sub:
            continue
        avg_ann = np.mean([r['ann'] for r in sub])
        avg_wr = np.mean([r['wr'] for r in sub])
        avg_n = np.mean([r['n'] for r in sub])
        avg_mdd = np.mean([r['mdd'] for r in sub])
        n_pos = sum(1 for r in sub if r['ann'] > 0)
        print(f"  {hd:>5}d | {avg_ann:>+8.1f}% | {avg_wr:>6.1f}% | {avg_n:>7.0f} | {avg_mdd:>7.1f}% | {n_pos:>5}/{len(sub)}")

    # ================================================================
    # TOP 5 CONFIGS THAT BEAT +49.6%
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  CONFIGS THAT BEAT +49.6% (Current Best: 50-day Breakout)")
    print(f"{'=' * 180}")
    beaters = [r for r in results if r['ann'] > 49.6]
    if beaters:
        print(f"  {'#':>3} | {'Label':<30} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'AvgHold':>7} | {'Freq/Yr':>7}")
        print("-" * 120)
        for i, r in enumerate(beaters[:20]):
            print(f"  {i+1:>3} | {r['label']:<30} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}% | {r['avg_hold']:>6.1f}d | {r['freq']:>6.1f}/yr")
    else:
        print("  No configs beat +49.6% annual return.")
        # Show top 5 anyway
        print(f"\n  Top 5 configs:")
        print(f"  {'#':>3} | {'Label':<30} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'AvgHold':>7}")
        print("-" * 120)
        for i, r in enumerate(results[:5]):
            print(f"  {i+1:>3} | {r['label']:<30} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}% | {r['avg_hold']:>6.1f}d")

    # ================================================================
    # TOP 5 DETAILED + WF
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  TOP 5 BEST CONFIGS -- DETAILED + WALK-FORWARD")
    print(f"{'=' * 180}")
    for i, r in enumerate(results[:5]):
        label = group_labels.get(r['config']['signal'], r['config']['signal'])
        print(f"\n  #{i+1}: {r['label']}  ({label})")
        print(f"       Annual: {r['ann']:>+8.1f}%  |  WR: {r['wr']:>5.1f}%  |  Trades: {r['n']}  |  MDD: {r['mdd']:>6.1f}%  |  AvgHold: {r['avg_hold']:.1f}d")
        print(f"       AvgPnL: {r['avg_pnl']:>+6.3f}%  |  Freq: {r['freq']:.1f}/yr  |  Final: {r['final_cash']:>13,.0f}")

        # WF for this config
        cfg = r['config']
        wf_str = "       WF: "
        for yr in wf_years:
            wr = run_backtest(cfg, wf_test_year=yr)
            if wr:
                wf_str += f"{yr}:{wr['ann']:>+6.1f}%  "
            else:
                wf_str += f"{yr}: N/A  "
        print(wf_str)

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  FINAL SUMMARY: TA-Lib VOLUME + VOLATILITY + CYCLE INDICATORS")
    print(f"{'=' * 180}")
    print()

    # Rank by avg WF
    sig_wf_avg = {}
    for sig_key in signal_keys:
        wf_match = [w for w in wf_rows if w['signal'] == sig_key]
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            sig_wf_avg[sig_key] = np.mean(vals) if vals else -999

    ranked_by_wf = sorted(sig_wf_avg.items(), key=lambda x: -x[1])

    print("  Ranked by Walk-Forward Average Annual Return:")
    print(f"  {'#':>3} | {'Signal':<42} | {'WF Avg':>8} | {'Best Ann':>9} | {'Best MDD':>8}")
    print("-" * 100)
    for i, (sig_key, wf_avg) in enumerate(ranked_by_wf):
        b = best_per_sig.get(sig_key)
        label = group_labels.get(sig_key, sig_key)
        best_ann = b['ann'] if b else 0
        best_mdd = b['mdd'] if b else 0
        print(f"  {i+1:>3} | {label:<42} | {wf_avg:>+7.1f}% | {best_ann:>+8.1f}% | {best_mdd:>7.1f}%")

    # Category summary
    print(f"\n  CATEGORY PERFORMANCE:")
    categories = {
        'Volume': ['mfi_cross20', 'mfi_oversold', 'obv_cross', 'ad_cross', 'adosc_cross'],
        'Volatility': ['natr_spike', 'trange_spike'],
        'Momentum': ['bop_3', 'bop_5', 'bop_7', 'ppo_cross', 'dx_strong'],
        'Cycle/Hilbert': ['ht_trend', 'ht_sine', 'mama_cross'],
        'Adaptive MA': ['kama_cross', 't3_cross'],
        'Multi-Signal': ['multi_vol'],
    }
    for cat_name, cat_keys in categories.items():
        sub = [r for r in results if r['config']['signal'] in cat_keys]
        if not sub:
            continue
        avg_ann = np.mean([r['ann'] for r in sub])
        best_ann = max(r['ann'] for r in sub)
        avg_wr = np.mean([r['wr'] for r in sub])
        avg_mdd = np.mean([r['mdd'] for r in sub])
        n_pos = sum(1 for r in sub if r['ann'] > 0)
        # WF avg
        wf_vals = []
        for k in cat_keys:
            if k in sig_wf_avg:
                wf_vals.append(sig_wf_avg[k])
        wf_cat_avg = np.mean(wf_vals) if wf_vals else 0
        print(f"  {cat_name:<15}: AvgAnn={avg_ann:>+7.1f}%  Best={best_ann:>+7.1f}%  WR={avg_wr:>5.1f}%  MDD={avg_mdd:>6.1f}%  Pos={n_pos}/{len(sub)}  WF_Avg={wf_cat_avg:>+7.1f}%")

    print(f"\n  Total configs tested: {len(configs)}")
    print(f"  Total runtime: {time.time()-t_start:.0f}s")
    print(f"\n{'=' * 180}")


if __name__ == '__main__':
    main()
