"""
Alpha Futures V116 — COMBINED ROC+Z-SIGNAL SYSTEMS
====================================================
V116 FOCUS: Combine the two strongest single signals found:
- ROC(5) > 2%: +89.8% (best single signal)
- Z-score(today_return, 20) > 2.0: +73.9% (best mathematical signal)

Test 10 combination approaches (A-J):
  A) ROC(5)>2% AND Z-score>1.5 (momentum + statistical)
  B) ROC(5)>1% AND Z-score>2.0 (moderate + extreme)
  C) ROC(5)>2% AND Z-score>2.0 (both extreme, very selective)
  D) ROC(5)>2% OR Z-score>2.0 (union, ranked by combined score)
  E) MULTI-FACTOR SCORING (5 factors, score 0-7)
  F) ROC(5)>2% AND Hurst>0.55 (momentum in trending regime)
  G) ROC(5)>2% AND Volume/OI ratio anomaly
  H) ROC(5)>2% AND C>SMA50 AND RSI>50 (triple filter)
  I) RANKED COMBINATION (cross-sectional rank sum)
  J) WEIGHTED SIGNAL SCORE (4-factor weighted)

ALL signals use NEXT-OPEN execution: signal at close di, entry at O[si, di+1].
Walk-forward by year (2020-2025).
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


def compute_hurst(ts, max_lag=20):
    """Compute Hurst exponent via R/S analysis on a 1D array."""
    ts = ts[~np.isnan(ts)]
    if len(ts) < max_lag + 1:
        return np.nan
    lags = range(2, max_lag + 1)
    rs_vals = []
    for lag in lags:
        segments = [ts[i:i+lag] for i in range(0, len(ts) - lag, lag)]
        if not segments:
            continue
        rs_seg = []
        for seg in segments:
            if len(seg) < 2:
                continue
            mean_seg = np.mean(seg)
            deviations = np.cumsum(seg - mean_seg)
            R = np.max(deviations) - np.min(deviations)
            S = np.std(seg, ddof=1)
            if S > 0:
                rs_seg.append(R / S)
        if rs_seg:
            rs_vals.append(np.mean(rs_seg))
    if len(rs_vals) < 3:
        return np.nan
    try:
        log_lags = np.log(list(range(2, 2 + len(rs_vals))))
        log_rs = np.log(rs_vals)
        H = np.polyfit(log_lags, log_rs, 1)[0]
        return H
    except:
        return np.nan


def main():
    print("=" * 220)
    print("  Alpha Futures V116 — COMBINED ROC+Z-SCORE SYSTEMS")
    print("=" * 220)
    print("\n  10 combination approaches (A-J), walk-forward 2020-2025")
    print("  ALL signals at close di, entry at O[si, di+1] (NEXT DAY OPEN)")
    print("  Baseline: ROC(5)>2% = +89.8%, Z-score>2.0 = +73.9%")

    # -- Load data -------------------------------------------------
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ================================================================
    # PRECOMPUTE ALL INDICATORS
    # ================================================================
    print("\n[Precompute] ROC, Z-score, SMA, ADX, RSI, Hurst, Volume/OI...", flush=True)
    t0 = time.time()

    # -- Daily returns --
    RET = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100  # percentage return
    print(f"  Daily returns computed ({time.time()-t0:.1f}s)")

    # -- ROC(5) --
    ROC5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        ROC5[si] = talib.ROC(c, timeperiod=5)
    print(f"  ROC(5) computed ({time.time()-t0:.1f}s)")

    # -- Z-score of daily returns (20-day rolling) --
    ZSCORE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = RET[si, di-20:di]
            valid = rets[~np.isnan(rets)]
            if len(valid) < 10:
                continue
            mean_r = np.mean(valid)
            std_r = np.std(valid, ddof=1)
            if std_r > 0 and not np.isnan(RET[si, di]):
                ZSCORE[si, di] = (RET[si, di] - mean_r) / std_r
    print(f"  Z-score(20) computed ({time.time()-t0:.1f}s)")

    # -- SMA(50) --
    SMA50 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        SMA50[si] = talib.SMA(c, timeperiod=50)
    print(f"  SMA(50) computed ({time.time()-t0:.1f}s)")

    # -- ADX(14) --
    ADX14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        h = H[si].astype(np.float64)
        l = L[si].astype(np.float64)
        c = C[si].astype(np.float64)
        ADX14[si] = talib.ADX(h, l, c, timeperiod=14)
    print(f"  ADX(14) computed ({time.time()-t0:.1f}s)")

    # -- RSI(14) --
    RSI14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        RSI14[si] = talib.RSI(c, timeperiod=14)
    print(f"  RSI(14) computed ({time.time()-t0:.1f}s)")

    # -- Hurst exponent (50-day rolling window, computed every 5 days for speed) --
    HURST = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(55, ND, 5):
            window = c[max(0, di-50):di]
            h_val = compute_hurst(np.diff(window) / window[:-1] * 100, max_lag=15)
            if not np.isnan(h_val):
                HURST[si, di] = h_val
                # Fill next 4 days with same value
                for d2 in range(di+1, min(di+5, ND)):
                    HURST[si, d2] = h_val
    print(f"  Hurst(50) computed ({time.time()-t0:.1f}s)")

    # -- Volume/OI ratio and SMA of V/OI --
    VOI_RATIO = np.full((NS, ND), np.nan)
    VOI_SMA20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            v = V[si, di]
            oi = OI[si, di]
            if not np.isnan(v) and not np.isnan(oi) and oi > 0:
                VOI_RATIO[si, di] = v / oi
        # SMA of V/OI over 20 days
        for di in range(21, ND):
            vals = VOI_RATIO[si, di-20:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 5:
                VOI_SMA20[si, di] = np.mean(valid)
    print(f"  Volume/OI ratio computed ({time.time()-t0:.1f}s)")

    # -- OI 5-day change rate --
    OI_CHANGE5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(6, ND):
            oi_now = OI[si, di]
            oi_5ago = OI[si, di-5]
            if not np.isnan(oi_now) and not np.isnan(oi_5ago) and oi_5ago > 0:
                OI_CHANGE5[si, di] = (oi_now / oi_5ago - 1) * 100
    print(f"  OI 5d change computed ({time.time()-t0:.1f}s)")

    print(f"\n  All indicators computed ({time.time()-t_start:.1f}s total)")

    # ================================================================
    # SIGNAL GENERATION — 10 combination systems
    # ================================================================
    print("\n[Signals] Computing 10 combination systems...", flush=True)
    t_sig = time.time()

    # ------------------------------------------------------------------
    # A) ROC(5)>2% AND Z-score(today_return, 20)>1.5
    # Both momentum and statistical extreme confirm
    # ------------------------------------------------------------------
    sig_A = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(25, ND):
            roc = ROC5[si, di]
            zs = ZSCORE[si, di]
            if np.isnan(roc) or np.isnan(zs):
                continue
            if roc > 2.0 and zs > 1.5:
                sig_A[si, di] = True
    print(f"  A) ROC>2% AND Z>1.5: {np.sum(sig_A)} signals")

    # ------------------------------------------------------------------
    # B) ROC(5)>1% AND Z-score>2.0
    # Moderate momentum + extreme day
    # ------------------------------------------------------------------
    sig_B = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(25, ND):
            roc = ROC5[si, di]
            zs = ZSCORE[si, di]
            if np.isnan(roc) or np.isnan(zs):
                continue
            if roc > 1.0 and zs > 2.0:
                sig_B[si, di] = True
    print(f"  B) ROC>1% AND Z>2.0: {np.sum(sig_B)} signals")

    # ------------------------------------------------------------------
    # C) ROC(5)>2% AND Z-score>2.0
    # Both extreme (very selective)
    # ------------------------------------------------------------------
    sig_C = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(25, ND):
            roc = ROC5[si, di]
            zs = ZSCORE[si, di]
            if np.isnan(roc) or np.isnan(zs):
                continue
            if roc > 2.0 and zs > 2.0:
                sig_C[si, di] = True
    print(f"  C) ROC>2% AND Z>2.0: {np.sum(sig_C)} signals")

    # ------------------------------------------------------------------
    # D) ROC(5)>2% OR Z-score>2.0 (union)
    # Rank by combined score: ROC(5) magnitude * Z-score magnitude
    # ------------------------------------------------------------------
    sig_D = np.zeros((NS, ND), dtype=bool)
    COMBINED_SCORE_D = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(25, ND):
            roc = ROC5[si, di]
            zs = ZSCORE[si, di]
            if np.isnan(roc) or np.isnan(zs):
                continue
            if roc > 2.0 or zs > 2.0:
                sig_D[si, di] = True
                COMBINED_SCORE_D[si, di] = abs(roc) * abs(zs)
    print(f"  D) ROC>2% OR Z>2.0: {np.sum(sig_D)} signals")

    # ------------------------------------------------------------------
    # E) MULTI-FACTOR SCORING (5 factors, score 0-7)
    # +2 if ROC(5) > 2%
    # +1 if ROC(5) > 0%
    # +2 if Z-score > 2.0
    # +1 if C > SMA(C, 50)
    # +1 if ADX > 25
    # Test thresholds: >=4, >=5, >=6
    # ------------------------------------------------------------------
    MF_SCORE = np.zeros((NS, ND), dtype=np.int8)
    for si in range(NS):
        for di in range(55, ND):
            score = 0
            roc = ROC5[si, di]
            zs = ZSCORE[si, di]
            c_now = C[si, di]
            sma50 = SMA50[si, di]
            adx = ADX14[si, di]

            if not np.isnan(roc) and roc > 2.0:
                score += 2
            elif not np.isnan(roc) and roc > 0:
                score += 1

            if not np.isnan(zs) and zs > 2.0:
                score += 2

            if not np.isnan(c_now) and not np.isnan(sma50) and c_now > sma50:
                score += 1

            if not np.isnan(adx) and adx > 25:
                score += 1

            MF_SCORE[si, di] = score

    sig_E4 = np.zeros((NS, ND), dtype=bool)
    sig_E5 = np.zeros((NS, ND), dtype=bool)
    sig_E6 = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(55, ND):
            s = MF_SCORE[si, di]
            if s >= 4:
                sig_E4[si, di] = True
            if s >= 5:
                sig_E5[si, di] = True
            if s >= 6:
                sig_E6[si, di] = True
    print(f"  E4) Multi-factor score>=4: {np.sum(sig_E4)} signals")
    print(f"  E5) Multi-factor score>=5: {np.sum(sig_E5)} signals")
    print(f"  E6) Multi-factor score>=6: {np.sum(sig_E6)} signals")

    # ------------------------------------------------------------------
    # F) ROC(5)>2% AND Hurst>0.55
    # Strong momentum in trending regime
    # ------------------------------------------------------------------
    sig_F = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(55, ND):
            roc = ROC5[si, di]
            hurst = HURST[si, di]
            if np.isnan(roc) or np.isnan(hurst):
                continue
            if roc > 2.0 and hurst > 0.55:
                sig_F[si, di] = True
    print(f"  F) ROC>2% AND Hurst>0.55: {np.sum(sig_F)} signals")

    # ------------------------------------------------------------------
    # G) ROC(5)>2% AND Volume/OI Ratio anomaly
    # V/OI > 1.5 * SMA(V/OI, 20)
    # ------------------------------------------------------------------
    sig_G = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(25, ND):
            roc = ROC5[si, di]
            voi = VOI_RATIO[si, di]
            voi_sma = VOI_SMA20[si, di]
            if np.isnan(roc) or np.isnan(voi) or np.isnan(voi_sma):
                continue
            if voi_sma <= 0:
                continue
            if roc > 2.0 and voi > 1.5 * voi_sma:
                sig_G[si, di] = True
    print(f"  G) ROC>2% AND V/OI anomaly: {np.sum(sig_G)} signals")

    # ------------------------------------------------------------------
    # H) ROC(5)>2% AND C>SMA50 AND RSI>50
    # Triple filter (V108's best risk-adjusted + ROC filter)
    # ------------------------------------------------------------------
    sig_H = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(55, ND):
            roc = ROC5[si, di]
            c_now = C[si, di]
            sma50 = SMA50[si, di]
            rsi = RSI14[si, di]
            if np.isnan(roc) or np.isnan(c_now) or np.isnan(sma50) or np.isnan(rsi):
                continue
            if roc > 2.0 and c_now > sma50 and rsi > 50:
                sig_H[si, di] = True
    print(f"  H) ROC>2% AND C>SMA50 AND RSI>50: {np.sum(sig_H)} signals")

    # ------------------------------------------------------------------
    # I) RANKED COMBINATION
    # Compute ROC(5) rank across all 68 commodities
    # Compute Z-score rank across all 68
    # Combined rank = ROC_rank + Z_rank (lower = better)
    # Buy top_n by combined rank when ROC(5)>0 AND Z-score>0
    # ------------------------------------------------------------------
    sig_I = np.zeros((NS, ND), dtype=bool)
    RANK_SCORE_I = np.full((NS, ND), np.nan)
    for di in range(25, ND):
        roc_vals = ROC5[:, di]
        zs_vals = ZSCORE[:, di]

        # Build valid mask
        valid = np.array([not np.isnan(roc_vals[si]) and not np.isnan(zs_vals[si])
                          and roc_vals[si] > 0 and zs_vals[si] > 0
                          for si in range(NS)])

        if np.sum(valid) < 1:
            continue

        # Rank by ROC5 (higher = better = lower rank number)
        roc_ranks = np.full(NS, 999.0)
        roc_valid_vals = roc_vals.copy()
        roc_valid_vals[~valid] = np.nan
        order_roc = np.argsort(-np.where(valid, roc_vals, -999))
        for rank_idx, si in enumerate(order_roc):
            if valid[si]:
                roc_ranks[si] = rank_idx + 1

        # Rank by Z-score (higher = better = lower rank number)
        zs_ranks = np.full(NS, 999.0)
        order_zs = np.argsort(-np.where(valid, zs_vals, -999))
        for rank_idx, si in enumerate(order_zs):
            if valid[si]:
                zs_ranks[si] = rank_idx + 1

        # Combined rank
        for si in range(NS):
            if valid[si]:
                combined_rank = roc_ranks[si] + zs_ranks[si]
                RANK_SCORE_I[si, di] = -combined_rank  # negative so higher = better for sorting
                sig_I[si, di] = True

    print(f"  I) Ranked combination (ROC>0 AND Z>0): {np.sum(sig_I)} signals")

    # ------------------------------------------------------------------
    # J) WEIGHTED SIGNAL SCORE
    # score = 0.4 * normalize(ROC(5)) + 0.3 * normalize(Z-score)
    #       + 0.2 * normalize(ADX/50) + 0.1 * normalize(OI_5d_change)
    # Buy when score > 70th percentile (top 30% of all signals)
    # ------------------------------------------------------------------
    RAW_SCORE_J = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(25, ND):
            roc = ROC5[si, di]
            zs = ZSCORE[si, di]
            adx = ADX14[si, di]
            oi_chg = OI_CHANGE5[si, di]
            if np.isnan(roc) or np.isnan(zs):
                continue
            # Normalize each to [0, 1] range using simple clipping
            n_roc = np.clip(roc / 10.0, -1, 1)  # +/-10% maps to +/-1
            n_zs = np.clip(zs / 4.0, -1, 1)     # +/-4 sigma maps to +/-1
            n_adx = np.clip((adx / 50.0 if not np.isnan(adx) else 0.5), 0, 1)
            n_oi = np.clip((oi_chg / 20.0 if not np.isnan(oi_chg) else 0), -1, 1)
            score = 0.4 * n_roc + 0.3 * n_zs + 0.2 * n_adx + 0.1 * n_oi
            RAW_SCORE_J[si, di] = score

    sig_J = np.zeros((NS, ND), dtype=bool)
    SCORE_J = np.full((NS, ND), np.nan)
    for di in range(25, ND):
        scores = RAW_SCORE_J[:, di]
        valid = ~np.isnan(scores)
        if np.sum(valid) < 5:
            continue
        pct70 = np.percentile(scores[valid], 70)
        for si in range(NS):
            if valid[si] and scores[si] > pct70:
                sig_J[si, di] = True
                SCORE_J[si, di] = scores[si]
    print(f"  J) Weighted score > 70th pct: {np.sum(sig_J)} signals")

    print(f"  All signals computed ({time.time()-t_sig:.1f}s)")

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(sig_arr, hold_days, top_n, wf_test_year=None,
                     score_arr=None, rank_desc=True):
        """Generic backtest for a signal array.
        score_arr: if provided, use for ranking (higher = better when rank_desc=True).
        """
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
                    cash += mkt_val - mkt_val * COMM
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
                # Score for ranking
                if score_arr is not None:
                    sc = score_arr[si, di]
                    if np.isnan(sc):
                        sc = 0
                else:
                    sc = ROC5[si, di] if not np.isnan(ROC5[si, di]) else 0
                candidates.append((sc, si, ep))

            if not candidates:
                continue

            # Sort by score
            if rank_desc:
                candidates.sort(key=lambda x: -x[0])
            else:
                candidates.sort(key=lambda x: x[0])

            # Open positions
            n_slots = top_n - len(positions)
            cap_per_slot = cash / max(1, n_slots)
            for sc_val, si, price in candidates[:max(0, n_slots)]:
                sym = syms[si]
                mult = MULT.get(sym, DEF_MULT)
                contracts = max(1, int(cap_per_slot / (price * mult)))
                cost_in = price * mult * contracts * (1 + COMM)
                if cost_in > cash:
                    contracts = int(cash * 0.9 / (price * mult * (1 + COMM)))
                    cost_in = price * mult * contracts * (1 + COMM) if contracts > 0 else 0
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
            cash += mkt_val - mkt_val * COMM
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

        # Max drawdown from trade-based equity curve
        eq = float(CASH0)
        peak = eq
        mdd = 0.0
        for t in sorted(trades, key=lambda x: x['entry_di']):
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

    # A) ROC>2% AND Z>1.5
    for hd in [3, 5, 7, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'A',
                'hold_days': hd, 'top_n': tn,
                'label': f"A_ROC2_AND_Z1.5_H{hd}_TN{tn}",
                'sig_arr': sig_A,
                'score_arr': None,
            })

    # B) ROC>1% AND Z>2.0
    for hd in [3, 5, 7, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'B',
                'hold_days': hd, 'top_n': tn,
                'label': f"B_ROC1_AND_Z2.0_H{hd}_TN{tn}",
                'sig_arr': sig_B,
                'score_arr': None,
            })

    # C) ROC>2% AND Z>2.0
    for hd in [3, 5, 7, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'C',
                'hold_days': hd, 'top_n': tn,
                'label': f"C_ROC2_AND_Z2.0_H{hd}_TN{tn}",
                'sig_arr': sig_C,
                'score_arr': None,
            })

    # D) ROC>2% OR Z>2.0 (union, ranked by ROC*Z)
    for hd in [3, 5, 7]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'D',
                'hold_days': hd, 'top_n': tn,
                'label': f"D_ROC2_OR_Z2_H{hd}_TN{tn}",
                'sig_arr': sig_D,
                'score_arr': COMBINED_SCORE_D,
            })

    # E4) Multi-factor score >= 4
    for tn in [1, 3]:
        cid += 1
        configs.append({
            'id': cid, 'signal': 'E4',
            'hold_days': 5, 'top_n': tn,
            'label': f"E4_MF>=4_H5_TN{tn}",
            'sig_arr': sig_E4,
            'score_arr': MF_SCORE.astype(np.float64),
        })

    # E5) Multi-factor score >= 5
    for tn in [1, 3]:
        cid += 1
        configs.append({
            'id': cid, 'signal': 'E5',
            'hold_days': 5, 'top_n': tn,
            'label': f"E5_MF>=5_H5_TN{tn}",
            'sig_arr': sig_E5,
            'score_arr': MF_SCORE.astype(np.float64),
        })

    # E6) Multi-factor score >= 6
    for tn in [1, 3]:
        cid += 1
        configs.append({
            'id': cid, 'signal': 'E6',
            'hold_days': 5, 'top_n': tn,
            'label': f"E6_MF>=6_H5_TN{tn}",
            'sig_arr': sig_E6,
            'score_arr': MF_SCORE.astype(np.float64),
        })

    # F) ROC>2% AND Hurst>0.55
    for hd in [5, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'F',
                'hold_days': hd, 'top_n': tn,
                'label': f"F_ROC2_Hurst0.55_H{hd}_TN{tn}",
                'sig_arr': sig_F,
                'score_arr': None,
            })

    # G) ROC>2% AND V/OI anomaly
    for hd in [5, 10]:
        cid += 1
        configs.append({
            'id': cid, 'signal': 'G',
            'hold_days': hd, 'top_n': 1,
            'label': f"G_ROC2_VOI_H{hd}_TN1",
            'sig_arr': sig_G,
            'score_arr': None,
        })

    # H) ROC>2% AND C>SMA50 AND RSI>50
    for hd in [5, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'H',
                'hold_days': hd, 'top_n': tn,
                'label': f"H_ROC2_SMA50_RSI50_H{hd}_TN{tn}",
                'sig_arr': sig_H,
                'score_arr': None,
            })

    # I) Ranked combination (ROC>0 AND Z>0), rank by combined rank
    for tn in [1, 3]:
        cid += 1
        configs.append({
            'id': cid, 'signal': 'I',
            'hold_days': 5, 'top_n': tn,
            'label': f"I_RankedCombo_H5_TN{tn}",
            'sig_arr': sig_I,
            'score_arr': RANK_SCORE_I,
        })

    # J) Weighted signal score > 70th percentile
    for tn in [1, 3, 5]:
        cid += 1
        configs.append({
            'id': cid, 'signal': 'J',
            'hold_days': 5, 'top_n': tn,
            'label': f"J_WeightedScore_H5_TN{tn}",
            'sig_arr': sig_J,
            'score_arr': SCORE_J,
        })

    total = len(configs)
    print(f"  Total configs: {total}")

    # ================================================================
    # RUN ALL CONFIGS (full backtest)
    # ================================================================
    print("\n[Backtest] Running all configs...", flush=True)
    t1 = time.time()
    results = []

    for ci, cfg in enumerate(configs):
        if ci % 10 == 0:
            print(f"  Config {ci}/{total} ({len(results)} done, {time.time()-t1:.0f}s)", flush=True)
        r = run_backtest(cfg['sig_arr'], cfg['hold_days'], cfg['top_n'],
                         score_arr=cfg.get('score_arr'))
        if r and r['n'] >= 3:
            r['config'] = cfg
            r['label'] = cfg['label']
            r['signal'] = cfg['signal']
            results.append(r)

    print(f"\n  Done ({time.time()-t1:.0f}s, {len(results)} configs with >= 3 trades)")
    results.sort(key=lambda x: -x['ann'])

    # ================================================================
    # PRINT TOP 30
    # ================================================================
    print(f"\n{'=' * 180}")
    print(f"  TOP 30 RESULTS (sorted by annual return)")
    print(f"{'=' * 180}")
    print(f"  {'#':>3} | {'Label':<35} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'MDD':>7} | {'AvgPnL':>8} | {'AvgHold':>7} | {'Freq':>6}")
    print("-" * 180)
    for i, r in enumerate(results[:30]):
        print(f"  {i+1:>3} | {r['label']:<35} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['mdd']:>6.1f}% | {r['avg_pnl']:>+7.3f}% | {r['avg_hold']:>6.1f}d | {r['freq']:>5.1f}/yr")

    # ================================================================
    # BEST PER SIGNAL TYPE
    # ================================================================
    sig_keys = ['A', 'B', 'C', 'D', 'E4', 'E5', 'E6', 'F', 'G', 'H', 'I', 'J']
    sig_names = {
        'A': 'A) ROC>2% AND Z>1.5',
        'B': 'B) ROC>1% AND Z>2.0',
        'C': 'C) ROC>2% AND Z>2.0',
        'D': 'D) ROC>2% OR Z>2.0',
        'E4': 'E4) Multi-factor >=4',
        'E5': 'E5) Multi-factor >=5',
        'E6': 'E6) Multi-factor >=6',
        'F': 'F) ROC>2% AND Hurst>0.55',
        'G': 'G) ROC>2% AND V/OI anomaly',
        'H': 'H) ROC>2% AND C>SMA50 AND RSI>50',
        'I': 'I) Ranked combo',
        'J': 'J) Weighted score >70th pct',
    }

    print(f"\n  BEST PER SIGNAL TYPE:")
    print(f"  {'Signal':<38} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'MDD':>7} | {'AvgPnL':>8} | {'Configs':>7}")
    print("-" * 130)

    best_per_sig = {}
    for sig_key in sig_keys:
        sub = [r for r in results if r['signal'] == sig_key]
        if not sub:
            print(f"  {sig_names.get(sig_key, sig_key):<38} | NO RESULTS")
            continue
        best = sub[0]
        best_per_sig[sig_key] = best
        n_pos = sum(1 for r in sub if r['ann'] > 0)
        print(f"  {sig_names.get(sig_key, sig_key):<38} | {best['ann']:>+8.1f}% | {best['wr']:>5.1f}% | {best['n']:>5} | {best['mdd']:>6.1f}% | {best['avg_pnl']:>+7.3f}% | {n_pos}/{len(sub)} pos")

    # ================================================================
    # KEY COMPARISON: Does combining ROC+Z beat +89.8%?
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  KEY COMPARISON: Does combining ROC+Z-score beat +89.8%?")
    print(f"{'=' * 160}")
    baseline = 89.8
    beats = [r for r in results if r['ann'] > baseline]
    print(f"  Baseline: ROC(5)>2% alone = +{baseline}%")
    print(f"  Configs beating baseline: {len(beats)}")
    for r in beats[:10]:
        print(f"    {r['label']:<40} | {r['ann']:>+8.1f}% | WR={r['wr']:>5.1f}% | N={r['n']:>4} | MDD={r['mdd']:>6.1f}%")

    # Multi-factor best threshold
    print(f"\n  MULTI-FACTOR SCORING THRESHOLDS:")
    for threshold, sk in [('>=4', 'E4'), ('>=5', 'E5'), ('>=6', 'E6')]:
        sub = [r for r in results if r['signal'] == sk]
        if sub:
            best = sub[0]
            n_pos = sum(1 for r in sub if r['ann'] > 0)
            print(f"    Score {threshold}: Best Ann={best['ann']:>+8.1f}% | WR={best['wr']:>5.1f}% | N={best['n']:>4} | MDD={best['mdd']:>6.1f}% | {n_pos}/{len(sub)} positive")

    # ================================================================
    # WALK-FORWARD
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Collect top 15 + best per signal type
    wf_configs = list(results[:15])
    for sig_key in sig_keys:
        if sig_key in best_per_sig:
            r = best_per_sig[sig_key]
            if r not in wf_configs:
                wf_configs.append(r)

    print(f"\n{'=' * 240}")
    print(f"  WALK-FORWARD ({len(wf_configs)} configs, years 2020-2025)")
    print(f"{'=' * 240}")

    header = f"  {'#':>3} | {'Config':<37} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>7} |"
    header += f" {'Pos':>4} | {'MDD':>7} | {'WR':>6}"
    print(header)
    print("-" * 240)

    wf_rows = []
    for i, r in enumerate(wf_configs):
        cfg = r['config']
        wf_row = {'label': cfg['label'], 'signal': cfg['signal'],
                  'entry': 'next_open', 'windows': {}, 'mdd': {}, 'wr': {}}
        for yr in wf_years:
            wr = run_backtest(cfg['sig_arr'], cfg['hold_days'], cfg['top_n'],
                              wf_test_year=yr, score_arr=cfg.get('score_arr'))
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

        row_str = f"  {i+1:>3} | {wf_row['label']:<37} | {avg:>+7.1f}% |"
        for v in vals:
            row_str += f" {v:>+7.1f}% |"
        row_str += f" {pos}/6 | {avg_mdd:>6.1f}% | {avg_wr:>5.1f}%"
        print(row_str)

    # ================================================================
    # WF COMPARISON PER SIGNAL TYPE
    # ================================================================
    print(f"\n{'=' * 220}")
    print("  WALK-FORWARD COMPARISON (Best per signal type)")
    print(f"{'=' * 220}")
    header2 = f"  {'Signal':<38} | {'WF Avg':>8} |"
    for yr in wf_years:
        header2 += f" {yr:>7} |"
    header2 += f" {'Pos':>4} | Avg MDD | Avg WR"
    print(header2)
    print("-" * 220)

    for sig_key in sig_keys:
        wf_match = [w for w in wf_rows if w['signal'] == sig_key]
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            avg = np.mean(vals) if vals else 0
            pos = sum(1 for v in vals if v > 0)
            avg_mdd = np.mean(list(wf['mdd'].values())) if wf['mdd'] else 0
            avg_wr = np.mean(list(wf['wr'].values())) if wf['wr'] else 0
            row_str = f"  {sig_names.get(sig_key, sig_key):<38} | {avg:>+7.1f}% |"
            for v in vals:
                row_str += f" {v:>+7.1f}% |"
            row_str += f" {pos}/6 | {avg_mdd:>6.1f}% | {avg_wr:>5.1f}%"
            print(row_str)
        else:
            print(f"  {sig_names.get(sig_key, sig_key):<38} | NO DATA")

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print(f"\n{'=' * 220}")
    print("  FINAL VERDICT: COMBINED ROC+Z-SCORE SYSTEMS")
    print(f"{'=' * 220}")
    print()
    print("  KEY QUESTIONS:")
    print("  1. Top 5 configs by annual return?")
    print("  2. Does combining ROC+Z-score beat +89.8%?")
    print("  3. Best multi-factor scoring threshold?")
    print("  4. Walk-forward consistency?")
    print()

    beats_best = []
    for sig_key in sig_keys:
        sub = [r for r in results if r['signal'] == sig_key]
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
            wf_avg = np.mean(vals) if vals else 0

        verdict = "POSITIVE" if best['ann'] > 0 else "NEGATIVE"
        genuine = "GENUINE ALPHA" if wf_pos >= 4 and best['ann'] > 0 else ("MARGINAL" if wf_pos >= 3 and best['ann'] > 0 else "NO ALPHA")
        beats = f"BEATS +{baseline}%" if best['ann'] > baseline else ("CLOSE" if best['ann'] > 50 else "INSUFFICIENT")

        if best['ann'] > baseline:
            beats_best.append((sig_key, best))

        print(f"  {sig_names.get(sig_key, sig_key)}")
        print(f"    Best annual: {best['ann']:>+8.1f}%  |  {n_pos}/{len(sub)} positive configs")
        print(f"    Walk-forward: {wf_pos}/6 positive  |  WF avg: {wf_avg:>+8.1f}%")
        print(f"    Trade freq: {best['freq']:>5.1f}/yr  |  Avg hold: {best['avg_hold']:>5.1f}d  |  Avg PnL: {best['avg_pnl']:>+6.3f}%")
        print(f"    VERDICT: {verdict}  -->  {genuine}  -->  {beats}")
        print()

    # Absolute best
    if results:
        best = results[0]
        print(f"  OVERALL BEST: {best['label']}")
        print(f"    Annual: {best['ann']:>+8.1f}%  |  WR: {best['wr']:>5.1f}%  |  N: {best['n']}  |  MDD: {best['mdd']:>6.1f}%")
        wf_match = [w for w in wf_rows if w['label'] == best['label']]
        if wf_match:
            vals = [wf_match[0]['windows'].get(yr, 0) for yr in wf_years]
            avg = np.mean(vals)
            pos = sum(1 for v in vals if v > 0)
            print(f"    Walk-forward: {pos}/6 positive  |  WF avg: {avg:>+8.1f}%")

    # Top 5 summary
    print(f"\n  TOP 5 CONFIGS BY ANNUAL RETURN:")
    for i, r in enumerate(results[:5]):
        wf_match = [w for w in wf_rows if w['label'] == r['label']]
        wf_info = ""
        if wf_match:
            vals = [wf_match[0]['windows'].get(yr, 0) for yr in wf_years]
            wf_pos = sum(1 for v in vals if v > 0)
            wf_avg = np.mean(vals)
            wf_info = f"WF {wf_pos}/6 pos, avg {wf_avg:>+7.1f}%"
        print(f"    {i+1}. {r['label']:<40} | {r['ann']:>+8.1f}% | WR={r['wr']:>5.1f}% | N={r['n']:>4} | MDD={r['mdd']:>6.1f}% | {wf_info}")

    print(f"\n  TOTAL TIME: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
