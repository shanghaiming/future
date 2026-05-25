"""
Alpha Futures V107 -- MULTI-SIGNAL COMBINATIONS using TA-Lib
=============================================================
Current best practical: 50-day Breakout +49.6% annual.

V107 IDEA: Combining 2-3 confirming signals from DIFFERENT indicator categories
(trend, momentum, volatility, volume) should give stronger, more reliable signals
that survive the 1-day execution delay.

ALL signals computed at close of day di using data up to and including di.
Entry at O[si, di+1] (NEXT DAY OPEN).
Exit at C[si, di+hold] (close price hold days later).

12 signal combos (A-L) x 3 holds (5/10/20) x 2 top_n (1/3) = ~72 configs.
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


def main():
    print("=" * 180)
    print("Alpha Futures V107 -- MULTI-SIGNAL COMBINATIONS using TA-Lib")
    print("=" * 180)
    print("\n  Combining 2-3 confirming signals from DIFFERENT indicator categories.")
    print("  ALL signals at close di, entry at O[si, di+1] (NEXT DAY OPEN)")

    # -- Load data -------------------------------------------------
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ================================================================
    # PRECOMPUTE ALL TA-LIB INDICATORS
    # ================================================================
    print("\n[Indicators] Computing ALL TA-Lib indicators...", flush=True)
    t0 = time.time()

    # -- Trend --
    ADX = np.full((NS, ND), np.nan)
    PLUS_DI = np.full((NS, ND), np.nan)
    MINUS_DI = np.full((NS, ND), np.nan)
    SAR_arr = np.full((NS, ND), np.nan)
    AROON_UP = np.full((NS, ND), np.nan)
    AROON_DOWN = np.full((NS, ND), np.nan)
    AROONOSC = np.full((NS, ND), np.nan)

    # -- Momentum --
    RSI = np.full((NS, ND), np.nan)
    CCI = np.full((NS, ND), np.nan)
    CMO = np.full((NS, ND), np.nan)
    WILLR = np.full((NS, ND), np.nan)
    ROC10 = np.full((NS, ND), np.nan)
    MOM10 = np.full((NS, ND), np.nan)
    MACD_hist = np.full((NS, ND), np.nan)
    STOCH_K = np.full((NS, ND), np.nan)
    STOCH_D = np.full((NS, ND), np.nan)
    ULTOSC = np.full((NS, ND), np.nan)
    PPO = np.full((NS, ND), np.nan)

    # -- Volatility --
    ATR = np.full((NS, ND), np.nan)
    NATR = np.full((NS, ND), np.nan)
    TRANGE = np.full((NS, ND), np.nan)
    BBANDS_upper = np.full((NS, ND), np.nan)
    BBANDS_middle = np.full((NS, ND), np.nan)
    BBANDS_lower = np.full((NS, ND), np.nan)

    # -- Volume --
    OBV = np.full((NS, ND), np.nan)
    AD_line = np.full((NS, ND), np.nan)
    ADOSC = np.full((NS, ND), np.nan)
    MFI = np.full((NS, ND), np.nan)

    # -- Moving Averages --
    SMA10 = np.full((NS, ND), np.nan)
    SMA20 = np.full((NS, ND), np.nan)
    SMA50 = np.full((NS, ND), np.nan)
    EMA12 = np.full((NS, ND), np.nan)
    EMA26 = np.full((NS, ND), np.nan)
    KAMA30 = np.full((NS, ND), np.nan)
    T3_20 = np.full((NS, ND), np.nan)

    # -- Cycle --
    HT_TRENDMODE = np.full((NS, ND), np.nan)
    HT_DCPERIOD = np.full((NS, ND), np.nan)

    # -- Adaptive --
    MAMA = np.full((NS, ND), np.nan)
    FAMA = np.full((NS, ND), np.nan)

    for si in range(NS):
        c = C[si].astype(np.float64)
        o = O[si].astype(np.float64)
        h = H[si].astype(np.float64)
        l = L[si].astype(np.float64)
        v = V[si].astype(np.float64)

        # Handle NaN: talib needs NaN-filled arrays, it handles NaN internally
        # Trend
        adx = talib.ADX(h, l, c, timeperiod=14)
        pdi = talib.PLUS_DI(h, l, c, timeperiod=14)
        mdi = talib.MINUS_DI(h, l, c, timeperiod=14)
        sar = talib.SAR(h, l, acceleration=0.02, maximum=0.2)
        aroon_d, aroon_u = talib.AROON(h, l, timeperiod=14)
        aroonosc = talib.AROONOSC(h, l, timeperiod=14)

        ADX[si] = adx
        PLUS_DI[si] = pdi
        MINUS_DI[si] = mdi
        SAR_arr[si] = sar
        AROON_UP[si] = aroon_u
        AROON_DOWN[si] = aroon_d
        AROONOSC[si] = aroonosc

        # Momentum
        rsi = talib.RSI(c, timeperiod=14)
        cci = talib.CCI(h, l, c, timeperiod=14)
        cmo = talib.CMO(c, timeperiod=14)
        willr = talib.WILLR(h, l, c, timeperiod=14)
        roc10 = talib.ROC(c, timeperiod=10)
        mom10 = talib.MOM(c, timeperiod=10)
        macd, macd_signal, macd_hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
        stoch_k, stoch_d = talib.STOCH(h, l, c,
                                         fastk_period=14, slowk_period=3,
                                         slowk_matype=0, slowd_period=3, slowd_matype=0)
        ultosc = talib.ULTOSC(h, l, c, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        ppo = talib.PPO(c, fastperiod=12, slowperiod=26, matype=0)

        RSI[si] = rsi
        CCI[si] = cci
        CMO[si] = cmo
        WILLR[si] = willr
        ROC10[si] = roc10
        MOM10[si] = mom10
        MACD_hist[si] = macd_hist
        STOCH_K[si] = stoch_k
        STOCH_D[si] = stoch_d
        ULTOSC[si] = ultosc
        PPO[si] = ppo

        # Volatility
        atr = talib.ATR(h, l, c, timeperiod=14)
        natr = talib.NATR(h, l, c, timeperiod=14)
        trange = talib.TRANGE(h, l, c)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

        ATR[si] = atr
        NATR[si] = natr
        TRANGE[si] = trange
        BBANDS_upper[si] = bb_upper
        BBANDS_middle[si] = bb_middle
        BBANDS_lower[si] = bb_lower

        # Volume
        obv = talib.OBV(c, v)
        ad_line = talib.AD(h, l, c, v)
        adosc = talib.ADOSC(h, l, c, v, fastperiod=3, slowperiod=10)
        mfi = talib.MFI(h, l, c, v, timeperiod=14)

        OBV[si] = obv
        AD_line[si] = ad_line
        ADOSC[si] = adosc
        MFI[si] = mfi

        # Moving Averages
        SMA10[si] = talib.SMA(c, timeperiod=10)
        SMA20[si] = talib.SMA(c, timeperiod=20)
        SMA50[si] = talib.SMA(c, timeperiod=50)
        EMA12[si] = talib.EMA(c, timeperiod=12)
        EMA26[si] = talib.EMA(c, timeperiod=26)
        KAMA30[si] = talib.KAMA(c, timeperiod=30)
        T3_20[si] = talib.T3(c, timeperiod=20)

        # Cycle
        HT_TRENDMODE[si] = talib.HT_TRENDMODE(c)
        HT_DCPERIOD[si] = talib.HT_DCPERIOD(c)

        # Adaptive
        mama, fama = talib.MAMA(c, fastlimit=0.5, slowlimit=0.05)
        MAMA[si] = mama
        FAMA[si] = fama

        if (si + 1) % 10 == 0 or si == NS - 1:
            print(f"  ... {si+1}/{NS} commodities done ({time.time()-t0:.1f}s)", flush=True)

    print(f"  All TA-Lib indicators computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # COMPUTE DERIVED ARRAYS (SMA of volume, SMA of NATR, SMA of OBV)
    # ================================================================
    print("\n[Derived] Computing SMAs of volume, NATR, OBV...", flush=True)

    vol_sma20 = np.full((NS, ND), np.nan)
    natr_sma20 = np.full((NS, ND), np.nan)
    obv_sma20 = np.full((NS, ND), np.nan)

    for si in range(NS):
        # SMA of volume
        v = V[si].astype(np.float64)
        vol_sma20[si] = talib.SMA(v, timeperiod=20)

        # SMA of NATR
        n = NATR[si].astype(np.float64)
        natr_sma20[si] = talib.SMA(n, timeperiod=20)

        # SMA of OBV
        ob = OBV[si].astype(np.float64)
        obv_sma20[si] = talib.SMA(ob, timeperiod=20)

    print(f"  Derived arrays computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # SIGNAL GENERATION
    # ================================================================
    print("\n[Signals] Computing all 12 signal combos...", flush=True)

    # -- A) TREND+MOMENTUM CONFIRMATION --
    sig_A = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(50, ND):
            adx_v = ADX[si, di]
            rsi_v = RSI[si, di]
            c_v = C[si, di]
            sma50_v = SMA50[si, di]
            if np.isnan(adx_v) or np.isnan(rsi_v) or np.isnan(c_v) or np.isnan(sma50_v):
                continue
            if adx_v > 25 and 50 <= rsi_v <= 70 and c_v > sma50_v:
                sig_A[si, di] = True
    print(f"  A) TREND+MOMENTUM: {np.sum(sig_A)} signals")

    # -- B) BREAKOUT+MOMENTUM+VOLUME --
    sig_B = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(50, ND):
            c_v = C[si, di]
            bb_up = BBANDS_upper[si, di]
            mh = MACD_hist[si, di]
            mh_prev = MACD_hist[si, di - 1]
            v_v = V[si, di]
            vs20 = vol_sma20[si, di]
            if np.isnan(c_v) or np.isnan(bb_up) or np.isnan(mh) or np.isnan(v_v) or np.isnan(vs20):
                continue
            if di < 1 or np.isnan(mh_prev):
                continue
            if c_v > bb_up and mh > 0 and mh > mh_prev and vs20 > 0 and v_v > 1.5 * vs20:
                sig_B[si, di] = True
    print(f"  B) BREAKOUT+MOMENTUM+VOL: {np.sum(sig_B)} signals")

    # -- C) ADX+CCI+VOLUME --
    sig_C = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(50, ND):
            adx_v = ADX[si, di]
            pdi_v = PLUS_DI[si, di]
            mdi_v = MINUS_DI[si, di]
            cci_v = CCI[si, di]
            v_v = V[si, di]
            vs20 = vol_sma20[si, di]
            if np.isnan(adx_v) or np.isnan(pdi_v) or np.isnan(mdi_v) or np.isnan(cci_v) or np.isnan(v_v) or np.isnan(vs20):
                continue
            if adx_v > 25 and pdi_v > mdi_v and cci_v > 100 and vs20 > 0 and v_v > 1.2 * vs20:
                sig_C[si, di] = True
    print(f"  C) ADX+CCI+VOL: {np.sum(sig_C)} signals")

    # -- D) SAR+ADX+MOMENTUM --
    sig_D = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(50, ND):
            sar_v = SAR_arr[si, di]
            c_v = C[si, di]
            adx_v = ADX[si, di]
            roc_v = ROC10[si, di]
            if np.isnan(sar_v) or np.isnan(c_v) or np.isnan(adx_v) or np.isnan(roc_v):
                continue
            if sar_v < c_v and adx_v > 20 and roc_v > 0:
                sig_D[si, di] = True
    print(f"  D) SAR+ADX+MOMENTUM: {np.sum(sig_D)} signals")

    # -- E) KAMA_CROSS+RSI --
    sig_E = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(50, ND):
            c_v = C[si, di]
            c_prev = C[si, di - 1]
            kama_v = KAMA30[si, di]
            kama_prev = KAMA30[si, di - 1]
            rsi_v = RSI[si, di]
            if np.isnan(c_v) or np.isnan(c_prev) or np.isnan(kama_v) or np.isnan(kama_prev) or np.isnan(rsi_v):
                continue
            if c_v > kama_v and c_prev <= kama_prev and rsi_v > 50:
                sig_E[si, di] = True
    print(f"  E) KAMA_CROSS+RSI: {np.sum(sig_E)} signals")

    # -- F) AROON+MACD+CCI --
    sig_F = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(50, ND):
            au = AROON_UP[si, di]
            ad_v = AROON_DOWN[si, di]
            mh = MACD_hist[si, di]
            cci_v = CCI[si, di]
            if np.isnan(au) or np.isnan(ad_v) or np.isnan(mh) or np.isnan(cci_v):
                continue
            if au > 70 and ad_v < 30 and mh > 0 and cci_v > 0:
                sig_F[si, di] = True
    print(f"  F) AROON+MACD+CCI: {np.sum(sig_F)} signals")

    # -- G) ULTIMATE OSCILLATOR+STOCH+RSI (triple oversold reversal) --
    sig_G = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(50, ND):
            ult = ULTOSC[si, di]
            ult_prev = ULTOSC[si, di - 1]
            sk = STOCH_K[si, di]
            sk_prev = STOCH_K[si, di - 1]
            sd = STOCH_D[si, di]
            sd_prev = STOCH_D[si, di - 1]
            rsi_v = RSI[si, di]
            if np.isnan(ult) or np.isnan(ult_prev) or np.isnan(sk) or np.isnan(sk_prev) or np.isnan(sd) or np.isnan(sd_prev) or np.isnan(rsi_v):
                continue
            # ULTOSC crossing up from oversold
            if ult > 30 and ult_prev <= 30:
                # STOCH_K crossing above STOCH_D
                if sk > sd and sk_prev <= sd_prev:
                    # RSI recovering
                    if rsi_v > 40:
                        sig_G[si, di] = True
    print(f"  G) ULTOSC+STOCH+RSI reversal: {np.sum(sig_G)} signals")

    # -- H) TRENDMODE+NATR_SPIKE+ADX --
    sig_H = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(50, ND):
            tm = HT_TRENDMODE[si, di]
            natr_v = NATR[si, di]
            ns20 = natr_sma20[si, di]
            adx_v = ADX[si, di]
            c_v = C[si, di]
            o_v = O[si, di]
            if np.isnan(tm) or np.isnan(natr_v) or np.isnan(ns20) or np.isnan(adx_v) or np.isnan(c_v) or np.isnan(o_v):
                continue
            if tm == 1 and ns20 > 0 and natr_v > 1.5 * ns20 and adx_v > 20 and c_v > o_v:
                sig_H[si, di] = True
    print(f"  H) TRENDMODE+NATR_SPIKE+ADX: {np.sum(sig_H)} signals")

    # -- I) MAMA_FAMA+PPO+OBV --
    sig_I = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(50, ND):
            mama_v = MAMA[si, di]
            fama_v = FAMA[si, di]
            ppo_v = PPO[si, di]
            obv_v = OBV[si, di]
            os20 = obv_sma20[si, di]
            if np.isnan(mama_v) or np.isnan(fama_v) or np.isnan(ppo_v) or np.isnan(obv_v) or np.isnan(os20):
                continue
            if mama_v > fama_v and ppo_v > 0 and obv_v > os20:
                sig_I[si, di] = True
    print(f"  I) MAMA_FAMA+PPO+OBV: {np.sum(sig_I)} signals")

    # -- J) MULTI-INDICATOR SCORING (score >= 4, top_n=3) --
    sig_J = np.zeros((NS, ND), dtype=bool)
    score_J = np.zeros((NS, ND), dtype=np.int32)
    for si in range(NS):
        for di in range(50, ND):
            c_v = C[si, di]
            v_v = V[si, di]
            vs20 = vol_sma20[si, di]
            natr_v = NATR[si, di]
            ns20 = natr_sma20[si, di]

            adx_v = ADX[si, di]
            pdi_v = PLUS_DI[si, di]
            mdi_v = MINUS_DI[si, di]
            rsi_v = RSI[si, di]
            mh = MACD_hist[si, di]
            sma50_v = SMA50[si, di]
            cci_v = CCI[si, di]

            if np.isnan(c_v) or np.isnan(v_v) or np.isnan(adx_v) or np.isnan(rsi_v):
                continue

            s = 0
            # +1: ADX > 25 AND PLUS_DI > MINUS_DI
            if not np.isnan(adx_v) and not np.isnan(pdi_v) and not np.isnan(mdi_v):
                if adx_v > 25 and pdi_v > mdi_v:
                    s += 1
            # +1: RSI in 50-70
            if not np.isnan(rsi_v) and 50 <= rsi_v <= 70:
                s += 1
            # +1: MACD_hist > 0
            if not np.isnan(mh) and mh > 0:
                s += 1
            # +1: C > SMA_50
            if not np.isnan(sma50_v) and c_v > sma50_v:
                s += 1
            # +1: V > 1.2 * SMA(V,20)
            if not np.isnan(vs20) and vs20 > 0 and v_v > 1.2 * vs20:
                s += 1
            # +1: CCI > 0
            if not np.isnan(cci_v) and cci_v > 0:
                s += 1
            # +1: NATR > SMA(NATR,20)
            if not np.isnan(natr_v) and not np.isnan(ns20) and ns20 > 0 and natr_v > ns20:
                s += 1

            score_J[si, di] = s
            if s >= 4:
                sig_J[si, di] = True
    print(f"  J) SCORING >=4 (7-factor): {np.sum(sig_J)} signals")

    # -- K) CONCENTRATED HIGH-SCORE (score >= 5, top_n=1) --
    sig_K = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(50, ND):
            if score_J[si, di] >= 5:
                sig_K[si, di] = True
    print(f"  K) SCORING >=5 (concentrated): {np.sum(sig_K)} signals")

    # -- L) WEIGHTED SCORING + BREAKOUT COMBO --
    sig_L = np.zeros((NS, ND), dtype=bool)
    score_L = np.zeros((NS, ND), dtype=np.int32)
    for si in range(NS):
        for di in range(50, ND):
            c_v = C[si, di]
            v_v = V[si, di]
            vs20 = vol_sma20[si, di]
            adx_v = ADX[si, di]
            rsi_v = RSI[si, di]
            roc_v = ROC10[si, di]
            sma20_v = SMA20[si, di]
            mh = MACD_hist[si, di]

            if np.isnan(c_v) or np.isnan(v_v) or np.isnan(adx_v) or np.isnan(rsi_v):
                continue

            s = 0
            # ADX trend (weight 2): +2 if ADX>25
            if adx_v > 25:
                s += 2
            # Momentum (weight 2): +2 if RSI>50 AND ROC>0
            if not np.isnan(rsi_v) and not np.isnan(roc_v) and rsi_v > 50 and roc_v > 0:
                s += 2
            # Volume (weight 1): +1 if V > 1.5*SMA(V,20)
            if not np.isnan(vs20) and vs20 > 0 and v_v > 1.5 * vs20:
                s += 1
            # Price position (weight 1): +1 if C > SMA_20
            if not np.isnan(sma20_v) and c_v > sma20_v:
                s += 1
            # MACD (weight 1): +1 if MACD_hist > 0
            if not np.isnan(mh) and mh > 0:
                s += 1

            score_L[si, di] = s
            if s >= 5:
                sig_L[si, di] = True
    print(f"  L) WEIGHTED SCORING >=5: {np.sum(sig_L)} signals")

    print(f"\n  All signals computed ({time.time()-t_start:.1f}s total)")

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(config, wf_test_year=None):
        sig_type = config['signal']
        hold_days = config['hold_days']
        top_n = config['top_n']
        comm = config.get('comm', COMM)

        # Map signal type to signal array and score array
        sig_map = {
            'A': sig_A, 'B': sig_B, 'C': sig_C, 'D': sig_D,
            'E': sig_E, 'F': sig_F, 'G': sig_G, 'H': sig_H,
            'I': sig_I, 'J': sig_J, 'K': sig_K, 'L': sig_L,
        }
        score_map = {
            'A': score_J, 'B': score_J, 'C': score_J, 'D': score_J,
            'E': score_J, 'F': score_J, 'G': score_J, 'H': score_J,
            'I': score_J, 'J': score_J, 'K': score_J, 'L': score_L,
        }
        sig_arr = sig_map[sig_type]
        sc_arr = score_map[sig_type]

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
                sc = sc_arr[si, di] if not np.isnan(sc_arr[si, di]) else 0
                candidates.append((sc, {
                    'si': si, 'sym': syms[si], 'entry_price': ep,
                }))

            if not candidates:
                continue

            # Sort by score descending
            candidates.sort(key=lambda x: -x[0])

            # Open positions
            n_slots = top_n - len(positions)
            for sc_val, info in candidates[:max(0, n_slots)]:
                si = info['si']
                sym = info['sym']
                price = info['entry_price']
                mult = MULT.get(sym, DEF_MULT)
                contracts = max(1, int(cash / (price * mult)))
                cost_in = price * mult * contracts * (1 + comm)
                if cost_in > cash:
                    contracts = int(cash * 0.9 / (price * mult * (1 + comm)))
                    cost_in = price * mult * contracts * (1 + comm) if contracts > 0 else 0
                if contracts <= 0 or cost_in <= 0 or cost_in > cash:
                    continue

                cash -= cost_in
                positions.append({
                    'si': si, 'entry_price': price, 'entry_di': entry_di,
                    'lots': contracts, 'dir': 1, 'sym': sym,
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

    sig_labels = {
        'A': 'Trend+Mom',
        'B': 'Brkout+Mom+Vol',
        'C': 'ADX+CCI+Vol',
        'D': 'SAR+ADX+Mom',
        'E': 'KAMA_X+RSI',
        'F': 'Aroon+MACD+CCI',
        'G': 'UltOsc+Stch+RSI',
        'H': 'TrendMode+NATR+ADX',
        'I': 'MAMA+PPO+OBV',
        'J': 'Score>=4_7f',
        'K': 'Score>=5_conc',
        'L': 'WtdScore>=5',
    }

    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
        for hd in [5, 10, 20]:
            if sig_key == 'K':
                tn = 1  # concentrated, single best
            elif sig_key == 'J':
                tn = 3
            else:
                for tn in [1, 3]:
                    cid += 1
                    configs.append({
                        'id': cid, 'signal': sig_key,
                        'hold_days': hd, 'top_n': tn, 'comm': COMM,
                        'label': f"{sig_key}_{sig_labels[sig_key]}_H{hd}_TN{tn}",
                    })
                continue  # skip the code below for this sig_key iteration
            # For J and K, just one top_n value
            cid += 1
            configs.append({
                'id': cid, 'signal': sig_key,
                'hold_days': hd, 'top_n': tn, 'comm': COMM,
                'label': f"{sig_key}_{sig_labels[sig_key]}_H{hd}_TN{tn}",
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
        if (i + 1) % 10 == 0 or i == len(configs) - 1:
            print(f"  ... {i+1}/{len(configs)} done ({time.time()-t_start:.0f}s)", flush=True)

    results.sort(key=lambda x: -x['ann'])

    # ================================================================
    # FULL-PERIOD RESULTS
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  FULL-PERIOD RESULTS (All configs) -- NEXT-OPEN EXECUTION, MULTI-SIGNAL COMBINATIONS")
    print(f"{'=' * 180}")
    print(f"  {'#':>3} | {'Label':<32} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'AvgHold':>7} | {'Freq/Yr':>7} | {'Final':>14}")
    print("-" * 170)
    for i, r in enumerate(results):
        print(f"  {i+1:>3} | {r['label']:<32} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}% | {r['avg_hold']:>6.1f}d | {r['freq']:>6.1f}/yr | {r['final_cash']:>13,.0f}")

    # ================================================================
    # BEST PER SIGNAL TYPE
    # ================================================================
    sig_names = {
        'A': 'A) TREND+MOMENTUM (ADX+RSI+SMA50)',
        'B': 'B) BREAKOUT+MOM+VOL (BBands+MACD+Vol)',
        'C': 'C) ADX+CCI+VOLUME',
        'D': 'D) SAR+ADX+ROC',
        'E': 'E) KAMA_CROSS+RSI',
        'F': 'F) AROON+MACD+CCI',
        'G': 'G) ULTOSC+STOCH+RSI reversal',
        'H': 'H) TRENDMODE+NATR_SPIKE+ADX',
        'I': 'I) MAMA_FAMA+PPO+OBV',
        'J': 'J) SCORE>=4 (7-factor ensemble)',
        'K': 'K) SCORE>=5 (concentrated)',
        'L': 'L) WEIGHTED SCORE>=5',
    }

    print(f"\n{'=' * 180}")
    print("  BEST PER SIGNAL TYPE (Full Period)")
    print(f"{'=' * 180}")
    print(f"  {'Signal':<45} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'AvgHold':>7} | {'Freq/Yr':>7} | Best Config")
    print("-" * 180)

    best_per_sig = {}
    for r in results:
        key = r['config']['signal']
        if key not in best_per_sig:
            best_per_sig[key] = r

    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
        if sig_key in best_per_sig:
            b = best_per_sig[sig_key]
            print(f"  {sig_names.get(sig_key, sig_key):<45} | {b['ann']:>+8.1f}% | {b['wr']:>5.1f}% | {b['n']:>5} | {b['avg_pnl']:>+6.3f}% | {b['mdd']:>6.1f}% | {b['avg_hold']:>6.1f}d | {b['freq']:>6.1f}/yr | {b['label']}")

    # ================================================================
    # SIGNAL TYPE SUMMARY
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  SIGNAL TYPE SUMMARY (Average of all configs per type)")
    print(f"{'=' * 180}")
    print(f"  {'Signal':<45} | {'Avg Ann':>9} | {'Avg WR':>7} | {'Avg N':>7} | {'Avg PnL':>8} | {'Avg MDD':>8} | {'#Positive':>9}")
    print("-" * 160)

    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
        sub = [r for r in results if r['config']['signal'] == sig_key]
        if not sub:
            continue
        avg_ann = np.mean([r['ann'] for r in sub])
        avg_wr = np.mean([r['wr'] for r in sub])
        avg_n = np.mean([r['n'] for r in sub])
        avg_pnl = np.mean([r['avg_pnl'] for r in sub])
        avg_mdd = np.mean([r['mdd'] for r in sub])
        n_pos = sum(1 for r in sub if r['ann'] > 0)
        print(f"  {sig_names.get(sig_key, sig_key):<45} | {avg_ann:>+8.1f}% | {avg_wr:>6.1f}% | {avg_n:>7.0f} | {avg_pnl:>+7.3f}% | {avg_mdd:>7.1f}% | {n_pos:>5}/{len(sub)}")

    # ================================================================
    # WALK-FORWARD (Top 15 configs)
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Collect top 15 overall + best per signal type
    wf_configs = list(results[:15])
    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
        if sig_key in best_per_sig:
            r = best_per_sig[sig_key]
            if r['config'] not in [w['config'] for w in wf_configs]:
                wf_configs.append(r)

    print(f"\n{'=' * 210}")
    print(f"  WALK-FORWARD ({len(wf_configs)} configs)")
    print(f"{'=' * 210}")

    header = f"  {'#':>3} | {'Config':<32} | {'Avg':>8} |"
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

        row_str = f"  {i+1:>3} | {wf_row['label']:<32} | {avg:>+7.1f}% |"
        for v in vals:
            row_str += f" {v:>+7.1f}% |"
        row_str += f" {pos}/6 | {avg_mdd:>6.1f}% | {avg_wr:>5.1f}%"
        print(row_str)

    # ================================================================
    # WF COMPARISON PER SIGNAL
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  WALK-FORWARD COMPARISON (Best per signal type)")
    print(f"{'=' * 180}")
    header2 = f"  {'Signal':<45} | {'WF Avg':>8} |"
    for yr in wf_years:
        header2 += f" {yr:>7} |"
    header2 += f" {'Pos':>4} | Avg MDD | Avg WR"
    print(header2)
    print("-" * 180)

    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
        wf_match = [w for w in wf_rows if w['signal'] == sig_key]
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            avg = np.mean(vals) if vals else 0
            pos = sum(1 for v in vals if v > 0)
            avg_mdd = np.mean(list(wf['mdd'].values())) if wf['mdd'] else 0
            avg_wr = np.mean(list(wf['wr'].values())) if wf['wr'] else 0
            row_str = f"  {sig_names.get(sig_key, sig_key):<45} | {avg:>+7.1f}% |"
            for v in vals:
                row_str += f" {v:>+7.1f}% |"
            row_str += f" {pos}/6 | {avg_mdd:>6.1f}% | {avg_wr:>5.1f}%"
            print(row_str)

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  FINAL VERDICT: MULTI-SIGNAL COMBINATIONS")
    print(f"{'=' * 180}")
    print()
    print("  KEY QUESTION: Can combining 2-3 confirming signals from different")
    print("  indicator categories produce reliable alpha with next-open execution?")
    print()

    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
        sub = [r for r in results if r['config']['signal'] == sig_key]
        if not sub:
            continue
        best = sub[0]
        n_pos = sum(1 for r in sub if r['ann'] > 0)

        wf_match = [w for w in wf_rows if w['signal'] == sig_key]
        wf_pos = 0
        wf_avg = 0
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            wf_pos = sum(1 for v in vals if v > 0)
            wf_avg = np.mean(vals)

        verdict = "POSITIVE" if best['ann'] > 0 else "NEGATIVE"
        genuine = "GENUINE ALPHA" if wf_pos >= 4 and best['ann'] > 0 else ("MARGINAL" if wf_pos >= 3 and best['ann'] > 0 else "NO ALPHA")
        beats = "BEATS +49.6%" if best['ann'] > 49.6 else ("CLOSE" if best['ann'] > 30 else "INSUFFICIENT")

        print(f"  {sig_names.get(sig_key, sig_key)}")
        print(f"    Best annual: {best['ann']:>+8.1f}%  |  {n_pos}/{len(sub)} positive configs")
        print(f"    Walk-forward: {wf_pos}/6 positive  |  WF avg: {wf_avg:>+8.1f}%")
        print(f"    Trade freq: {best['freq']:>5.1f}/yr  |  Avg hold: {best['avg_hold']:>5.1f}d  |  Avg PnL: {best['avg_pnl']:>+6.3f}%")
        print(f"    VERDICT: {verdict}  -->  {genuine}  -->  {beats}")
        print()

    # Absolute best
    if results:
        champ = results[0]
        print(f"  {'='*70}")
        print(f"  CHAMPION: {champ['label']}")
        print(f"    Annual: {champ['ann']:>+8.1f}%  |  WR: {champ['wr']:>5.1f}%  |  N: {champ['n']:>4}  |  MDD: {champ['mdd']:>6.1f}%")
        print(f"    Avg PnL/trade: {champ['avg_pnl']:>+6.3f}%  |  Avg Hold: {champ['avg_hold']:>5.1f}d  |  Freq: {champ['freq']:>5.1f}/yr")
        champ_wf = [w for w in wf_rows if w['label'] == champ['label']]
        if champ_wf:
            cw = champ_wf[0]
            vals = [cw['windows'].get(yr, 0) for yr in wf_years]
            print(f"    WF: {[f'{v:>+7.1f}%' for v in vals]}  |  {sum(1 for v in vals if v > 0)}/6 positive")
        print(f"  {'='*70}")

    # Top 5 summary
    print(f"\n  TOP 5 CONFIGS:")
    print(f"  {'#':>3} | {'Label':<32} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'MDD':>7} | WF_Avg | WF_Pos")
    print("-" * 130)
    for i, r in enumerate(results[:5]):
        wf_match = [w for w in wf_rows if w['label'] == r['label']]
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            wf_avg = np.mean(vals)
            wf_pos = sum(1 for v in vals if v > 0)
        else:
            wf_avg = 0
            wf_pos = 0
        print(f"  {i+1:>3} | {r['label']:<32} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['mdd']:>6.1f}% | {wf_avg:>+7.1f}% | {wf_pos}/6")

    print(f"\n  Total runtime: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
