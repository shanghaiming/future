"""
Alpha Futures V184 — Intraday Price Pattern (K-Line Shape) Strategy
==============================================================================
V178 champion: short_mirror, atr<10%, c=0.5, top_n=3 → +187% annual, R/M=11.41

V184 explores K-line / candlestick pattern analysis for futures:
Unlike stocks, futures have T+0 and higher leverage, making daily bar patterns
more meaningful for short-term trading signals.

Factors tested:
  1. Body Ratio: |C-O|/(H-L) — measures trend strength
  2. Upper Shadow Ratio: (H-max(C,O))/(H-L) — selling pressure
  3. Lower Shadow Ratio: (min(C,O)-L)/(H-L) — buying support
  4. Intraday Range: (H-L)/O normalized daily range — volatility expansion
  5. Hammer/Hanging Man: long lower shadow + small body at extremes
  6. Engulfing Pattern: today's body engulfs yesterday's body
  7. Volume-Price Confirmation: large body + high volume = genuine breakout

Signal logic:
  - Long: Strong bullish bar (body ratio > 0.7, C > O) + volume > 1.5x 20d avg
  - Combined with 5-day momentum filter (ROC5 > 0)
  - Short: Mirror logic (bearish bar + volume confirmation)

Walk-forward: Train 2019-2023, Test 2024-2026.
Baseline to beat: R/M = 11.41.
"""
import sys, os, time, warnings
import numpy as np
import talib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

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
    if final <= 0 or initial <= 0 or n_days <= 0: return -100.0
    return (final / initial) ** (1.0 / (n_days / 252)) * 100 - 100


def main():
    print("=" * 130)
    print("  V184 — Intraday Price Pattern (K-Line Shape) Strategy for Futures")
    print("  Baseline: V178 best config → R/M = 11.41")
    print("=" * 130)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    print(f"  {NS} commodities, {ND} days")

    # ===================== PRECOMPUTE =====================
    print("\n[Precompute] K-line pattern factors...", flush=True)
    t0 = time.time()

    # --- Basic returns and momentum ---
    RET = np.full((NS, ND), np.nan)
    ROC5 = np.full((NS, ND), np.nan)
    ROC10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100
        ROC5[si] = talib.ROC(c, timeperiod=5)
        ROC10[si] = talib.ROC(c, timeperiod=10)

    # --- ATR ---
    ATR14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        ATR14[si] = talib.ATR(H[si].astype(np.float64), L[si].astype(np.float64),
                               C[si].astype(np.float64), timeperiod=14)

    ATR_NORM = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            atr = ATR14[si, di]
            cp = C[si, di]
            if not np.isnan(atr) and not np.isnan(cp) and cp > 0:
                ATR_NORM[si, di] = atr / cp * 100

    # --- Z-Score of daily returns ---
    ZSCORE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = RET[si, di-20:di]
            v = rets[~np.isnan(rets)]
            if len(v) < 10: continue
            s = np.std(v, ddof=1)
            if s > 0 and not np.isnan(RET[si, di]):
                ZSCORE[si, di] = (RET[si, di] - np.mean(v)) / s

    # ===================== K-LINE PATTERN FACTORS =====================
    # All factors computed per (symbol, day)
    # Range: H - L (avoid division by zero later)

    HL_RANGE = np.full((NS, ND), np.nan)   # H - L
    BODY = np.full((NS, ND), np.nan)        # |C - O|
    BODY_RATIO = np.full((NS, ND), np.nan)  # |C-O| / (H-L)
    UPPER_SHADOW = np.full((NS, ND), np.nan)  # (H - max(C,O)) / (H-L)
    LOWER_SHADOW = np.full((NS, ND), np.nan)  # (min(C,O) - L) / (H-L)
    ID_RANGE = np.full((NS, ND), np.nan)    # (H-L) / O normalized

    for si in range(NS):
        for di in range(ND):
            o = O[si, di]; c = C[si, di]; h = H[si, di]; l = L[si, di]
            if any(np.isnan(x) for x in [o, c, h, l]): continue
            hl = h - l
            if hl <= 0: continue
            HL_RANGE[si, di] = hl
            BODY[si, di] = abs(c - o)
            BODY_RATIO[si, di] = abs(c - o) / hl
            UPPER_SHADOW[si, di] = (h - max(c, o)) / hl
            LOWER_SHADOW[si, di] = (min(c, o) - l) / hl
            if o > 0:
                ID_RANGE[si, di] = hl / o * 100  # as percentage

    # --- Volume ratio vs 20-day average ---
    VOL_RATIO = np.full((NS, ND), np.nan)
    for si in range(NS):
        v = V[si].astype(np.float64)
        vol_ma = talib.SMA(v, timeperiod=20)
        for di in range(20, ND):
            if not np.isnan(vol_ma[di]) and vol_ma[di] > 0 and not np.isnan(v[di]):
                VOL_RATIO[si, di] = v[di] / vol_ma[di]

    # --- Engulfing pattern ---
    # Bullish engulfing: today's body engulfs yesterday's body
    #   i.e., today O < yesterday's body AND today C > yesterday's body
    # Bearish engulfing: mirror
    ENGULF_BULL = np.full((NS, ND), np.nan)
    ENGULF_BEAR = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            o0 = O[si, di-1]; c0 = C[si, di-1]
            o1 = O[si, di];   c1 = C[si, di]
            if any(np.isnan(x) for x in [o0, c0, o1, c1]): continue
            body_top0 = max(o0, c0); body_bot0 = min(o0, c0)
            body_top1 = max(o1, c1); body_bot1 = min(o1, c1)
            # Bullish engulfing: prev was bearish (c0 < o0), today bullish (c1 > o1)
            # and today's body engulfs yesterday's body
            if c0 < o0 and c1 > o1 and o1 <= body_bot0 and c1 >= body_top0:
                ENGULF_BULL[si, di] = 1.0
            # Bearish engulfing: prev was bullish (c0 > o0), today bearish (c1 < o1)
            if c0 > o0 and c1 < o1 and o1 >= body_top0 and c1 <= body_bot0:
                ENGULF_BEAR[si, di] = 1.0

    # --- Hammer pattern ---
    # Hammer: small body at top, long lower shadow (>= 2x body), little/no upper shadow
    # Hanging man: same shape but at top of uptrend (bearish)
    HAMMER = np.full((NS, ND), np.nan)
    HANGING_MAN = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            br = BODY_RATIO[si, di]
            us = UPPER_SHADOW[si, di]
            ls = LOWER_SHADOW[si, di]
            if np.isnan(br) or np.isnan(us) or np.isnan(ls): continue
            body_val = BODY[si, di]
            hl = HL_RANGE[si, di]
            if np.isnan(body_val) or np.isnan(hl) or body_val <= 0 or hl <= 0: continue
            # Hammer: lower shadow >= 2x body, upper shadow <= 0.1 * range
            if ls >= 2.0 * (body_val / hl) and us <= 0.1 and br <= 0.3:
                # Check if in downtrend (for hammer = bullish reversal)
                roc5 = ROC5[si, di]
                if not np.isnan(roc5) and roc5 < 0:
                    HAMMER[si, di] = 1.0
                else:
                    HANGING_MAN[si, di] = 1.0

    # --- Composite K-line score ---
    # Combines multiple pattern factors into a single directional score
    # Positive = bullish, Negative = bearish
    KSCORE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            br = BODY_RATIO[si, di]
            ls = LOWER_SHADOW[si, di]
            us = UPPER_SHADOW[si, di]
            vr = VOL_RATIO[si, di]
            ir = ID_RANGE[si, di]
            o = O[si, di]; c = C[si, di]
            if any(np.isnan(x) for x in [br, ls, us, vr, ir, o, c]): continue
            if o <= 0: continue

            # Direction: +1 if bullish (C > O), -1 if bearish
            direction = 1.0 if c > o else -1.0

            # Score components (each roughly 0-1 scale):
            # 1. Body ratio: high body ratio = strong trend
            score_br = br  # 0 to 1
            # 2. Volume confirmation: >1 = above average
            score_vol = min(vr / 2.0, 1.0)  # cap at 1.0
            # 3. Shadow alignment: bullish bar with low upper shadow is better
            score_shadow = 1.0 - us if direction > 0 else 1.0 - ls
            # 4. Range expansion: bigger range = more conviction
            score_range = min(ir / 3.0, 1.0)  # normalize, cap

            # Weighted composite
            kscore = direction * (0.35 * score_br + 0.30 * score_vol +
                                  0.15 * score_shadow + 0.20 * score_range)
            KSCORE[si, di] = kscore

    # ===================== REGIME INDICATORS =====================
    print("  Computing regime indicators...", flush=True)

    BREADTH = np.full(ND, np.nan)
    for di in range(5, ND):
        pos_count = 0; total = 0
        for si in range(NS):
            r = ROC5[si, di]
            if not np.isnan(r):
                total += 1
                if r > 0: pos_count += 1
        if total > 0:
            BREADTH[di] = pos_count / total

    MKT_RET = np.full(ND, np.nan)
    for di in range(ND):
        rets_day = RET[:, di]
        valid = rets_day[~np.isnan(rets_day)]
        if len(valid) > 10:
            MKT_RET[di] = np.mean(valid)

    MKT_VOL = np.full(ND, np.nan)
    for di in range(20, ND):
        window = MKT_RET[di-20:di]
        valid = window[~np.isnan(window)]
        if len(valid) >= 10:
            MKT_VOL[di] = np.std(valid, ddof=1)

    valid_vols = MKT_VOL[~np.isnan(MKT_VOL)]
    VOL_MEDIAN = np.median(valid_vols) if len(valid_vols) > 0 else 1.0
    print(f"  Market vol median={VOL_MEDIAN:.4f}%")
    print(f"  Precompute done ({time.time()-t0:.1f}s)")

    # ===================== SIGNAL DEFINITIONS =====================
    # Each signal function returns list of (score, symbol_index, entry_price, label)

    def sig_kline_strong_bar(di, edi):
        """Strong bullish/bearish bar: high body ratio + volume confirmation.
        Long: body_ratio > 0.7, C > O, vol > 1.5x avg, ROC5 > 0
        """
        cands = []
        for s in range(NS):
            br = BODY_RATIO[s, di]
            o = O[s, di]; c = C[s, di]; vr = VOL_RATIO[s, di]
            roc5 = ROC5[s, di]
            if any(np.isnan(x) for x in [br, o, c, vr, roc5]): continue
            if br <= 0.7 or c <= o or vr <= 1.5 or roc5 <= 0: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = br * vr * roc5
            cands.append((score, s, ep, 'kline_strong_bull'))
        return cands

    def sig_kline_strong_bar_short(di, edi):
        """Strong bearish bar: high body ratio + volume confirmation (short side)."""
        cands = []
        for s in range(NS):
            br = BODY_RATIO[s, di]
            o = O[s, di]; c = C[s, di]; vr = VOL_RATIO[s, di]
            roc5 = ROC5[s, di]
            if any(np.isnan(x) for x in [br, o, c, vr, roc5]): continue
            if br <= 0.7 or c >= o or vr <= 1.5 or roc5 >= 0: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = br * vr * abs(roc5)
            cands.append((score, s, ep, 'kline_strong_bear'))
        return cands

    def sig_kline_engulf_vol(di, edi):
        """Bullish engulfing + volume confirmation + momentum filter."""
        cands = []
        for s in range(NS):
            eb = ENGULF_BULL[s, di]
            if np.isnan(eb): continue
            vr = VOL_RATIO[s, di]
            roc5 = ROC5[s, di]
            if any(np.isnan(x) for x in [vr, roc5]): continue
            if vr <= 1.2 or roc5 <= -1.0: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = vr * (roc5 + 2)  # boost by momentum
            cands.append((score, s, ep, 'engulf_bull_vol'))
        return cands

    def sig_kline_engulf_vol_short(di, edi):
        """Bearish engulfing + volume confirmation + momentum filter."""
        cands = []
        for s in range(NS):
            eb = ENGULF_BEAR[s, di]
            if np.isnan(eb): continue
            vr = VOL_RATIO[s, di]
            roc5 = ROC5[s, di]
            if any(np.isnan(x) for x in [vr, roc5]): continue
            if vr <= 1.2 or roc5 >= 1.0: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = vr * (abs(roc5) + 2)
            cands.append((score, s, ep, 'engulf_bear_vol'))
        return cands

    def sig_kline_hammer(di, edi):
        """Hammer reversal pattern (bullish) + volume confirmation."""
        cands = []
        for s in range(NS):
            hm = HAMMER[s, di]
            if np.isnan(hm): continue
            vr = VOL_RATIO[s, di]
            roc5 = ROC5[s, di]
            if any(np.isnan(x) for x in [vr, roc5]): continue
            if vr <= 1.0: continue  # at least average volume
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = vr
            cands.append((score, s, ep, 'hammer_bull'))
        return cands

    def sig_kline_hanging_man_short(di, edi):
        """Hanging man pattern (bearish) + volume confirmation."""
        cands = []
        for s in range(NS):
            hm = HANGING_MAN[s, di]
            if np.isnan(hm): continue
            vr = VOL_RATIO[s, di]
            roc5 = ROC5[s, di]
            if any(np.isnan(x) for x in [vr, roc5]): continue
            if vr <= 1.0: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = vr
            cands.append((score, s, ep, 'hanging_man_bear'))
        return cands

    def sig_kline_composite(di, edi):
        """Composite K-score: combine all pattern factors.
        Long when KSCORE > threshold + volume + momentum."""
        cands = []
        for s in range(NS):
            ks = KSCORE[s, di]
            vr = VOL_RATIO[s, di]
            roc5 = ROC5[s, di]
            if any(np.isnan(x) for x in [ks, vr, roc5]): continue
            if ks <= 0.5 or vr <= 1.2 or roc5 <= 0: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = ks * vr * roc5
            cands.append((score, s, ep, 'kscore_long'))
        return cands

    def sig_kline_composite_short(di, edi):
        """Composite K-score short: KSCORE < -threshold + volume + momentum."""
        cands = []
        for s in range(NS):
            ks = KSCORE[s, di]
            vr = VOL_RATIO[s, di]
            roc5 = ROC5[s, di]
            if any(np.isnan(x) for x in [ks, vr, roc5]): continue
            if ks >= -0.5 or vr <= 1.2 or roc5 >= 0: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = abs(ks) * vr * abs(roc5)
            cands.append((score, s, ep, 'kscore_short'))
        return cands

    def sig_kline_range_expansion(di, edi):
        """Range expansion + directional body + volume.
        Futures-specific: expanding range = breakout in progress."""
        cands = []
        for s in range(NS):
            ir = ID_RANGE[s, di]
            br = BODY_RATIO[s, di]
            vr = VOL_RATIO[s, di]
            roc5 = ROC5[s, di]
            o = O[s, di]; c = C[s, di]
            if any(np.isnan(x) for x in [ir, br, vr, roc5, o, c]): continue
            if c <= o: continue  # bullish only
            if ir <= 1.5 or br <= 0.6 or vr <= 1.3 or roc5 <= 0: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = ir * br * vr * roc5
            cands.append((score, s, ep, 'range_exp_long'))
        return cands

    def sig_kline_range_expansion_short(di, edi):
        """Range expansion + bearish body + volume (short)."""
        cands = []
        for s in range(NS):
            ir = ID_RANGE[s, di]
            br = BODY_RATIO[s, di]
            vr = VOL_RATIO[s, di]
            roc5 = ROC5[s, di]
            o = O[s, di]; c = C[s, di]
            if any(np.isnan(x) for x in [ir, br, vr, roc5, o, c]): continue
            if c >= o: continue  # bearish only
            if ir <= 1.5 or br <= 0.6 or vr <= 1.3 or roc5 >= 0: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = ir * br * vr * abs(roc5)
            cands.append((score, s, ep, 'range_exp_short'))
        return cands

    def sig_kline_low_shadow_bull(di, edi):
        """Long lower shadow = buying support. Works as reversal from oversold.
        Lower shadow > 0.5, small upper shadow < 0.15, ROC5 was negative but recovering."""
        cands = []
        for s in range(NS):
            ls = LOWER_SHADOW[s, di]
            us = UPPER_SHADOW[s, di]
            vr = VOL_RATIO[s, di]
            roc5 = ROC5[s, di]
            br = BODY_RATIO[s, di]
            o = O[s, di]; c = C[s, di]
            if any(np.isnan(x) for x in [ls, us, vr, roc5, br, o, c]): continue
            if ls <= 0.5 or us >= 0.15 or c <= o: continue
            if vr <= 1.0: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = ls * vr * br
            cands.append((score, s, ep, 'low_shadow_bull'))
        return cands

    def sig_kline_high_shadow_short(di, edi):
        """Long upper shadow = selling pressure at highs. Bearish reversal signal.
        Upper shadow > 0.5, small lower shadow < 0.15, ROC5 was positive but weakening."""
        cands = []
        for s in range(NS):
            us = UPPER_SHADOW[s, di]
            ls = LOWER_SHADOW[s, di]
            vr = VOL_RATIO[s, di]
            roc5 = ROC5[s, di]
            br = BODY_RATIO[s, di]
            o = O[s, di]; c = C[s, di]
            if any(np.isnan(x) for x in [us, ls, vr, roc5, br, o, c]): continue
            if us <= 0.5 or ls >= 0.15 or c >= o: continue
            if vr <= 1.0: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = us * vr * br
            cands.append((score, s, ep, 'high_shadow_bear'))
        return cands

    def sig_kline_union(di, edi):
        """Union of all long K-line pattern signals, aggregated by symbol."""
        all_sigs = {}
        signal_fns = [
            sig_kline_strong_bar, sig_kline_engulf_vol, sig_kline_hammer,
            sig_kline_composite, sig_kline_range_expansion, sig_kline_low_shadow_bull
        ]
        for fn in signal_fns:
            for item in fn(di, edi):
                sc, s, ep, st = item
                if s not in all_sigs:
                    all_sigs[s] = [0, ep, []]
                all_sigs[s][0] += sc  # sum scores
                all_sigs[s][2].append(st)
        return [(sc, s, ep, '+'.join(sigs)) for s, (sc, ep, sigs) in all_sigs.items()]

    def sig_kline_union_short(di, edi):
        """Union of all short K-line pattern signals."""
        all_sigs = {}
        signal_fns = [
            sig_kline_strong_bar_short, sig_kline_engulf_vol_short,
            sig_kline_hanging_man_short, sig_kline_composite_short,
            sig_kline_range_expansion_short, sig_kline_high_shadow_short
        ]
        for fn in signal_fns:
            for item in fn(di, edi):
                sc, s, ep, st = item
                if s not in all_sigs:
                    all_sigs[s] = [0, ep, []]
                all_sigs[s][0] += sc
                all_sigs[s][2].append(st)
        return [(sc, s, ep, '+'.join(sigs)) for s, (sc, ep, sigs) in all_sigs.items()]

    # ===================== HELPERS =====================
    def compute_composite(di, daily_eq, high_water, perf_window=20):
        scores = []
        bth = BREADTH[di]
        if not np.isnan(bth):
            scores.append(np.clip((bth - 0.4) / (0.7 - 0.4), 0, 1))
        vol = MKT_VOL[di]
        if not np.isnan(vol) and VOL_MEDIAN > 0:
            vol_ratio = vol / VOL_MEDIAN
            scores.append(np.clip((1.5 - vol_ratio) / (1.5 - 0.8), 0, 1))
        if len(daily_eq) >= perf_window:
            eq_window = np.array(daily_eq[-perf_window:])
            x = np.arange(perf_window)
            try:
                slope = np.polyfit(x, eq_window, 1)[0]
                eq_mean = np.mean(eq_window)
                norm_slope = slope / eq_mean * 100 if eq_mean > 0 else 0
                eq_rets = np.diff(eq_window) / eq_window[:-1] * 100
                eq_rets = eq_rets[np.isfinite(eq_rets)]
                eq_std = np.std(eq_rets) if len(eq_rets) > 5 else 1.0
                z = norm_slope / eq_std if eq_std > 0 else 0
                scores.append(np.clip((z + 1.0) / 2.0, 0, 1))
            except Exception:
                pass
        if high_water > 0:
            cur_dd = (daily_eq[-1] - high_water) / high_water
        else:
            cur_dd = 0
        scores.append(np.clip(1.0 + cur_dd / 0.3, 0, 1))
        return np.mean(scores) if scores else 0.5

    def dd_size(pv, high_water, tiers):
        if high_water <= 0: return tiers[0][1]
        dd = (pv - high_water) / high_water
        for dd_thresh, size_frac in tiers:
            if dd >= -dd_thresh: return size_frac
        return tiers[-1][1]

    # ===================== BACKTEST ENGINE =====================
    def backtest(start_di=MIN_TRAIN, end_di=None,
                 atr_norm_max=10.0,
                 dd_tiers=None,
                 regime_lo=0.5, regime_hi=1.5,
                 top_n=3, short_mode='short_mirror',
                 hold=1,
                 sig_long_fn=None, sig_short_fn=None):
        if end_di is None: end_di = ND
        if dd_tiers is None:
            dd_tiers = [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)]
        if sig_long_fn is None:
            sig_long_fn = sig_kline_strong_bar
        if sig_short_fn is None:
            sig_short_fn = sig_kline_strong_bar_short

        cash = float(CASH0)
        positions = []
        trades = []
        daily_eq = []
        high_water = float(CASH0)
        trade_pnls = []

        for di in range(start_di, end_di - 1):
            # Mark-to-market
            pv = cash
            for p in positions:
                cp = C[p['si'], di]
                if not np.isnan(cp) and cp > 0:
                    m = MULT.get(p['sym'], DEF_MULT)
                    d = p.get('dir', 1)
                    unrealized = (cp - p['entry_price']) * m * p['lots'] * d
                    pv += p['entry_price'] * m * abs(p['lots']) + unrealized - cp * m * abs(p['lots']) * COMM
            daily_eq.append(pv)
            if pv > high_water:
                high_water = pv

            # --- Exit logic (fixed hold) ---
            cl = []
            for p in positions:
                si = p['si']; d = p.get('dir', 1)
                days_held = di - p['entry_di']
                cp = C[si, di]
                if np.isnan(cp) or cp <= 0: continue

                if days_held >= hold:
                    m = MULT.get(p['sym'], DEF_MULT)
                    pnl = (cp - p['entry_price']) * m * p['lots'] * d
                    inv = p['entry_price'] * m * abs(p['lots'])
                    pp = pnl / inv * 100 if inv > 0 else 0
                    if d == 1:
                        cash += cp * m * abs(p['lots']) * (1 - COMM)
                    else:
                        margin = p['entry_price'] * m * abs(p['lots'])
                        cash += margin + pnl - cp * m * abs(p['lots']) * COMM
                    trades.append(pp)
                    trade_pnls.append(pnl)
                    cl.append(p)

            for p in cl: positions.remove(p)

            # --- Position sizing ---
            dd_sz = dd_size(pv, high_water, dd_tiers)
            composite = compute_composite(di, daily_eq, high_water)
            regime_mult = regime_lo + composite * (regime_hi - regime_lo)
            pos_size = max(0.05, min(0.99, dd_sz * regime_mult))

            # --- Enter positions ---
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue

            held_si = set(p['si'] for p in positions)

            # Long signals
            cands_long = sig_long_fn(di, edi)
            cands_long_f = [c for c in cands_long
                           if not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < atr_norm_max]
            cands_long_f.sort(key=lambda x: -x[0])

            best_long = None
            for c in cands_long_f:
                if c[1] not in held_si:
                    best_long = c
                    break

            entries = []
            if best_long:
                entries.append((best_long[0], best_long[1], best_long[2], 'long', pos_size, 1))

            # Short signals
            if short_mode != 'long_only' and len(positions) + len(entries) < top_n:
                cands_short = sig_short_fn(di, edi)
                cands_short_f = [c for c in cands_short
                                if not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < atr_norm_max]
                cands_short_f.sort(key=lambda x: -x[0])

                held_si = set(p['si'] for p in positions) | set(e[1] for e in entries)
                best_short = None
                for c in cands_short_f:
                    if c[1] not in held_si:
                        best_short = c
                        break

                if best_short:
                    entries.append((best_short[0], best_short[1], best_short[2], 'short', pos_size, -1))

            cash_snapshot = cash
            n_planned = len(entries)
            for sc, s, pr, sig_str, pct, d in entries:
                if s in set(p['si'] for p in positions): continue
                if len(positions) >= top_n: break
                cap = cash_snapshot * pct / max(n_planned, 1)
                sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                ct = max(1, int(cap / (pr * m * (1 + COMM))))
                ci = pr * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                    ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                if ct <= 0 or ci <= 0 or ci > cash: continue
                cash -= ci
                pos = {'si': s, 'entry_price': pr, 'entry_di': edi,
                       'lots': ct, 'dir': d, 'sym': sym,
                       'hold_days': hold, 'sig': sig_str, 'score': sc}
                positions.append(pos)

        # Close remaining
        for p in positions:
            ae = end_di - 1
            ep = C[p['si'], min(ae, ND-1)]
            if np.isnan(ep) or ep <= 0: ep = p['entry_price']
            m = MULT.get(p['sym'], DEF_MULT)
            d = p.get('dir', 1)
            if d == 1:
                cash += ep * m * abs(p['lots']) * (1 - COMM)
            else:
                pnl = (ep - p['entry_price']) * m * p['lots'] * d
                margin = p['entry_price'] * m * abs(p['lots'])
                cash += margin + pnl - ep * m * abs(p['lots']) * COMM

        nd = end_di - start_di
        ann = annual_return(cash, CASH0, nd)
        wr = np.mean([1 if t > 0 else 0 for t in trades]) * 100 if trades else 0
        nt = len(trades)
        if daily_eq:
            eq = np.array(daily_eq); pk = np.maximum.accumulate(eq)
            mdd = np.min((eq - pk) / pk * 100)
            r = np.diff(eq) / eq[:-1]
            r = np.where(np.isfinite(r), r, 0)
            sh = np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0
        else:
            mdd = 0; sh = 0

        avg_pnl = np.mean(trade_pnls) if trade_pnls else 0

        return {'ann': ann, 'wr': wr, 'n': nt, 'mdd': mdd, 'sharpe': sh,
                'final': cash, 'avg_pnl': avg_pnl}

    # ===================== PRINTING HELPERS =====================
    def pr(r, label=""):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  {label:85s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:6.2f} | N={r['n']:4d} | AvgPnL={r['avg_pnl']:>8.0f}")

    def walk_forward(label="", **kwargs):
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue
            r = backtest(start_di=ys, end_di=ye, **kwargs)
            res[yr] = r
        return res

    def print_wf(wf_res, label=""):
        pos = sum(1 for r in wf_res.values() if r['ann'] > 0)
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"    {label}")
        print(f"      {pos}/6 pos | Avg={avg_ann:>+7.0f}% | WorstWfMDD={worst_mdd:>5.0f}%")
        print(f"      {ws}")

    # ===================== CONFIG =====================
    DD_AGGR100 = [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)]
    all_results = []

    # ===================== SECTION 0: FACTOR DISTRIBUTION ANALYSIS =====================
    print("\n" + "=" * 130)
    print("  SECTION 0: K-Line Factor Distribution Analysis")
    print("=" * 130)

    for name, arr in [('BODY_RATIO', BODY_RATIO), ('UPPER_SHADOW', UPPER_SHADOW),
                      ('LOWER_SHADOW', LOWER_SHADOW), ('ID_RANGE', ID_RANGE),
                      ('VOL_RATIO', VOL_RATIO), ('KSCORE', KSCORE)]:
        vals = arr[~np.isnan(arr)]
        if len(vals) > 0:
            print(f"    {name:20s}: mean={np.mean(vals):.3f} | std={np.std(vals):.3f} | "
                  f"P25={np.percentile(vals,25):.3f} | P50={np.percentile(vals,50):.3f} | "
                  f"P75={np.percentile(vals,75):.3f} | P90={np.percentile(vals,90):.3f} | "
                  f"N={len(vals)}")

    # Pattern counts
    n_engulf_bull = np.sum(ENGULF_BULL[~np.isnan(ENGULF_BULL)])
    n_engulf_bear = np.sum(ENGULF_BEAR[~np.isnan(ENGULF_BEAR)])
    n_hammer = np.sum(HAMMER[~np.isnan(HAMMER)])
    n_hanging = np.sum(HANGING_MAN[~np.isnan(HANGING_MAN)])
    print(f"\n    Pattern counts: ENGULF_BULL={n_engulf_bull:.0f} | ENGULF_BEAR={n_engulf_bear:.0f} | "
          f"HAMMER={n_hammer:.0f} | HANGING_MAN={n_hanging:.0f}")

    # ===================== SECTION 1: INDIVIDUAL PATTERN SIGNALS =====================
    print("\n" + "=" * 130)
    print("  SECTION 1: Individual K-Line Pattern Signals (long+short mirror)")
    print("=" * 130)

    signal_configs = [
        ('strong_bar',      sig_kline_strong_bar,           sig_kline_strong_bar_short),
        ('engulf_vol',      sig_kline_engulf_vol,           sig_kline_engulf_vol_short),
        ('hammer',          sig_kline_hammer,               sig_kline_hanging_man_short),
        ('composite',       sig_kline_composite,            sig_kline_composite_short),
        ('range_expansion', sig_kline_range_expansion,      sig_kline_range_expansion_short),
        ('low_shadow',      sig_kline_low_shadow_bull,      sig_kline_high_shadow_short),
    ]

    for sig_name, sig_long, sig_short in signal_configs:
        r = backtest(sig_long_fn=sig_long, sig_short_fn=sig_short,
                     short_mode='short_mirror', hold=1, top_n=3,
                     atr_norm_max=10.0)
        pr(r, f"KLINE {sig_name} (hold=1)")
        all_results.append({**r, 'label': f'kline_{sig_name}_h1',
                            'signal': sig_name, 'hold': 1})

    # ===================== SECTION 2: HOLD PERIOD VARIANTS =====================
    print("\n" + "=" * 130)
    print("  SECTION 2: Hold Period Variants for Top Signals")
    print("=" * 130)

    for h in [1, 2, 3]:
        for sig_name, sig_long, sig_short in signal_configs:
            r = backtest(sig_long_fn=sig_long, sig_short_fn=sig_short,
                         short_mode='short_mirror', hold=h, top_n=3,
                         atr_norm_max=10.0)
            pr(r, f"KLINE {sig_name} hold={h}")
            all_results.append({**r, 'label': f'kline_{sig_name}_h{h}',
                                'signal': sig_name, 'hold': h})

    # ===================== SECTION 3: UNION SIGNAL (all patterns combined) =====================
    print("\n" + "=" * 130)
    print("  SECTION 3: Union Signal (all patterns combined)")
    print("=" * 130)

    for h in [1, 2, 3]:
        r = backtest(sig_long_fn=sig_kline_union, sig_short_fn=sig_kline_union_short,
                     short_mode='short_mirror', hold=h, top_n=3,
                     atr_norm_max=10.0)
        pr(r, f"UNION all patterns hold={h}")
        all_results.append({**r, 'label': f'kline_union_h{h}',
                            'signal': 'union', 'hold': h})

    # ===================== SECTION 4: ATR NORM FILTER VARIANTS =====================
    print("\n" + "=" * 130)
    print("  SECTION 4: ATR Norm Filter Variants")
    print("=" * 130)

    for atr_max in [5.0, 7.0, 10.0, 15.0]:
        for sig_name, sig_long, sig_short in [('strong_bar', sig_kline_strong_bar, sig_kline_strong_bar_short),
                                               ('composite', sig_kline_composite, sig_kline_composite_short)]:
            r = backtest(sig_long_fn=sig_long, sig_short_fn=sig_short,
                         short_mode='short_mirror', hold=1, top_n=3,
                         atr_norm_max=atr_max)
            pr(r, f"{sig_name} atr_max={atr_max}%")
            all_results.append({**r, 'label': f'kline_{sig_name}_atr{atr_max}',
                                'signal': sig_name, 'atr_norm_max': atr_max})

    # ===================== SECTION 5: TOP_N VARIANTS =====================
    print("\n" + "=" * 130)
    print("  SECTION 5: top_n Variants")
    print("=" * 130)

    for tn in [1, 2, 3, 5]:
        r = backtest(sig_long_fn=sig_kline_union, sig_short_fn=sig_kline_union_short,
                     short_mode='short_mirror', hold=1, top_n=tn,
                     atr_norm_max=10.0)
        pr(r, f"UNION top_n={tn}")
        all_results.append({**r, 'label': f'kline_union_tn{tn}',
                            'signal': 'union', 'top_n': tn})

    # ===================== SECTION 6: LONG-ONLY VS SHORT MIRROR =====================
    print("\n" + "=" * 130)
    print("  SECTION 6: Long-only vs Short Mirror")
    print("=" * 130)

    for sm in ['long_only', 'short_mirror']:
        for sig_name, sig_long, sig_short in [('strong_bar', sig_kline_strong_bar, sig_kline_strong_bar_short),
                                               ('composite', sig_kline_composite, sig_kline_composite_short),
                                               ('union', sig_kline_union, sig_kline_union_short)]:
            r = backtest(sig_long_fn=sig_long, sig_short_fn=sig_short,
                         short_mode=sm, hold=1, top_n=3,
                         atr_norm_max=10.0)
            pr(r, f"{sig_name} {sm}")
            all_results.append({**r, 'label': f'kline_{sig_name}_{sm}',
                                'signal': sig_name, 'short_mode': sm})

    # ===================== SECTION 7: PAIR COMBINATIONS =====================
    print("\n" + "=" * 130)
    print("  SECTION 7: Best Long Signal + Best Short Signal Cross-Combinations")
    print("=" * 130)

    # Cross all long signals with all short signals
    long_fns = [(n, fn) for n, fn, _ in signal_configs]
    short_fns = [(n, fn) for n, _, fn in signal_configs]

    for ln, lfn in long_fns:
        for sn, sfn in short_fns:
            r = backtest(sig_long_fn=lfn, sig_short_fn=sfn,
                         short_mode='short_mirror', hold=1, top_n=3,
                         atr_norm_max=10.0)
            pr(r, f"L={ln} + S={sn}")
            all_results.append({**r, 'label': f'kline_L{ln}_S{sn}',
                                'signal': f'L={ln}+S={sn}'})

    # ===================== WALK-FORWARD =====================
    print("\n" + "=" * 130)
    print("  WALK-FORWARD: Top Configurations")
    print("=" * 130)

    # Rank all results by R/M
    ranked = sorted(all_results, key=lambda x: abs(x.get('ann', 0) / x.get('mdd', -1)), reverse=True)

    # Walk-forward for top 5 unique configs
    seen_signals = set()
    wf_count = 0
    for r in ranked:
        sig = r.get('signal', '')
        h = r.get('hold', 1)
        key = f"{sig}_h{h}"
        if key in seen_signals:
            continue
        seen_signals.add(key)
        wf_count += 1
        if wf_count > 5:
            break

        # Find the matching signal functions
        sig_l = sig_s = None
        for n, fn_l, fn_s in signal_configs:
            if n == sig:
                sig_l = fn_l; sig_s = fn_s
                break
        if sig == 'union':
            sig_l = sig_kline_union; sig_s = sig_kline_union_short

        if sig_l is None:
            continue

        print(f"\n  Walk-forward: {r['label']}")
        wf = walk_forward(label=r['label'],
                          sig_long_fn=sig_l, sig_short_fn=sig_s,
                          short_mode='short_mirror', hold=h, top_n=3,
                          atr_norm_max=10.0)
        print_wf(wf, r['label'])

    # Also walk-forward the union signal
    for h in [1, 2]:
        print(f"\n  Walk-forward: UNION hold={h}")
        wf = walk_forward(label=f"UNION h={h}",
                          sig_long_fn=sig_kline_union, sig_short_fn=sig_kline_union_short,
                          short_mode='short_mirror', hold=h, top_n=3,
                          atr_norm_max=10.0)
        print_wf(wf, f"UNION h={h}")

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 130)
    print("  V184 FINAL SUMMARY: K-Line Pattern Analysis")
    print("=" * 130)

    # Re-rank final
    ranked = sorted(all_results, key=lambda x: abs(x.get('ann', 0) / x.get('mdd', -1)), reverse=True)
    base_rm = 11.41  # V178 baseline R/M

    print(f"\n  {'Config':50s} | {'Ann':>7s} | {'MDD':>5s} | {'R/M':>6s} | {'WR':>5s} | {'N':>4s} | {'Sh':>5s} | vs 11.41")
    print(f"  {'-'*50}-+-{'-'*7}-+-{'-'*5}-+-{'-'*6}-+-{'-'*5}-+-{'-'*4}-+-{'-'*5}-+-{'-'*8}")
    print(f"  {'V178 BASELINE (R/M=11.41)':50s} |       |       | {base_rm:>6.2f} |       |       |       |    ---")

    for r in ranked[:25]:
        ann = r['ann']; mdd = r['mdd']
        rm = abs(ann / mdd) if mdd != 0 else 0
        delta = rm - base_rm
        marker = " ***" if delta > 2.0 else (" **" if delta > 1.0 else "")
        print(f"  {r['label']:50s} | {ann:>+7.0f}% | {mdd:>5.0f}% | {rm:>6.2f} | {r['wr']:>5.1f}% | {r['n']:>4d} | {r['sharpe']:>5.2f} | {delta:>+8.2f}{marker}")

    # Best by category
    print(f"\n  --- Best by signal type ---")
    for sig_name in ['strong_bar', 'engulf_vol', 'hammer', 'composite',
                     'range_expansion', 'low_shadow', 'union']:
        cat = [r for r in all_results if r.get('signal') == sig_name]
        if cat:
            best = max(cat, key=lambda x: abs(x.get('ann', 0) / x.get('mdd', -1)))
            rm = abs(best['ann'] / best['mdd']) if best['mdd'] != 0 else 0
            print(f"    {sig_name:20s}: Ann={best['ann']:+.0f}% MDD={best['mdd']:.0f}% R/M={rm:.2f} ({best['label']})")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
