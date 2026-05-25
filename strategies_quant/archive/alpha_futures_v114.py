"""
Alpha Futures V114 -- REGIME-ADAPTIVE STRATEGY with Next-Open Execution
========================================================================
Current best: ROC(5) cross +81.9%, 6/6 WF.

V114 IDEA: Instead of one signal for all conditions, detect the market regime
per commodity and apply the BEST signal for that regime.

Regime Detection (per commodity):
  1. ADX(14) -- trend strength
  2. NATR(14) -- normalized ATR (volatility regime)
  3. ROC(20) -- medium-term momentum direction
  4. Market breadth -- % of 68 commodities with ROC(5) > 0

Regime-based signals (A-I):
  A) Strong Trend (ADX>30, ROC20>0) -> ROC(5) cross, hold 5d
  B) Moderate Trend (ADX 20-30, ROC20>0) -> T3 cross, hold 5d
  C) Volatility Breakout (NATR squeeze -> expansion) -> 20d breakout, hold 5d
  D) High Vol Momentum (ADX>25 + NATR high) -> ROC(5) + ADX, hold 5d, 75% size
  E) Broad Market Strength (breadth>60%) -> ROC(5) cross, hold 5d
  F) Regime-Switching: ADX>25 -> ROC(5), ADX<=25 -> T3 cross, hold 5d
  G) Dynamic Hold: ADX>35->10d, ADX25-35->5d, ADX<25->3d, ROC(5) entry
  H) Composite Score: ADX+ROC20+NATR+Breadth >=3 -> ROC(5) entry, hold 5d
  I) Per-Commodity Trending Filter: ADX>25 AND ROC20>0 -> ROC(5) cross, hold 5d

Walk-forward by year (2020-2025) for all configs.
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
    print("=" * 200)
    print("Alpha Futures V114 -- REGIME-ADAPTIVE STRATEGY with Next-Open Execution")
    print("=" * 200)
    print("\n  Detect market regime per commodity, apply best signal for that regime.")
    print("  ALL signals at close di, entry at O[si, di+1] (NEXT DAY OPEN)")
    print("  9 regime-adaptive configurations (A-I)")

    # -- Load data -------------------------------------------------
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ================================================================
    # PRECOMPUTE REGIME INDICATORS
    # ================================================================
    print("\n[Precompute] Regime indicators...", flush=True)
    t0 = time.time()

    # --- ROC(5) and ROC(20) ---
    ROC5 = np.full((NS, ND), np.nan)
    ROC20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        ROC5[si] = talib.ROC(c, timeperiod=5)
        ROC20[si] = talib.ROC(c, timeperiod=20)

    ROC5_prev = np.roll(ROC5, 1, axis=1)
    ROC5_prev[:, 0] = np.nan

    # --- ADX(14) ---
    ADX14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        h = H[si].astype(np.float64)
        l = L[si].astype(np.float64)
        c = C[si].astype(np.float64)
        ADX14[si] = talib.ADX(h, l, c, timeperiod=14)

    # --- NATR(14) (Normalized ATR) ---
    NATR14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        h = H[si].astype(np.float64)
        l = L[si].astype(np.float64)
        c = C[si].astype(np.float64)
        NATR14[si] = talib.NATR(h, l, c, timeperiod=14)

    # --- T3 (Triple Exponential Moving Average) ---
    T3 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        T3[si] = talib.T3(c, timeperiod=20, vfactor=0.7)

    T3_prev = np.roll(T3, 1, axis=1)
    T3_prev[:, 0] = np.nan

    # --- 20-day highest high / lowest low for breakout ---
    HIGH20 = np.full((NS, ND), np.nan)
    LOW20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        h = H[si].astype(np.float64)
        l = L[si].astype(np.float64)
        HIGH20[si] = talib.MAX(h, timeperiod=20)
        LOW20[si] = talib.MIN(l, timeperiod=20)

    # --- NATR percentile ranks (100-day rolling window) ---
    NATR_pctl = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(100, ND):
            window = NATR14[si, di-99:di+1]
            valid = window[~np.isnan(window)]
            if len(valid) < 20:
                continue
            natr_now = NATR14[si, di]
            if np.isnan(natr_now):
                continue
            NATR_pctl[si, di] = np.sum(valid <= natr_now) / len(valid) * 100

    NATR_pctl_prev = np.roll(NATR_pctl, 1, axis=1)
    NATR_pctl_prev[:, 0] = np.nan

    # --- Market breadth: % of commodities with ROC(5) > 0 ---
    BREADTH = np.full(ND, np.nan)
    for di in range(10, ND):
        vals = ROC5[:, di]
        valid = vals[~np.isnan(vals)]
        if len(valid) < 10:
            continue
        BREADTH[di] = np.sum(valid > 0) / len(valid) * 100

    print(f"  Regime indicators computed ({time.time()-t0:.1f}s)")
    print(f"    ADX range: {np.nanmin(ADX14):.1f} - {np.nanmax(ADX14):.1f}")
    print(f"    NATR range: {np.nanmin(NATR14):.2f} - {np.nanmax(NATR14):.2f}")
    print(f"    Breadth range: {np.nanmin(BREADTH):.1f} - {np.nanmax(BREADTH):.1f}")

    # ================================================================
    # BUILD SIGNAL ARRAYS (A-I)
    # ================================================================
    print("\n[Signals] Building regime-adaptive signals...", flush=True)

    # A) STRONG TREND: ADX>30, ROC20>0 -> ROC(5) cross (prev<=0, now>0)
    sig_A = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(1, ND):
            adx = ADX14[si, di]
            roc20 = ROC20[si, di]
            roc5 = ROC5[si, di]
            roc5_p = ROC5_prev[si, di]
            if np.isnan(adx) or np.isnan(roc20) or np.isnan(roc5) or np.isnan(roc5_p):
                continue
            if adx > 30 and roc20 > 0 and roc5 > 0 and roc5_p <= 0:
                sig_A[si, di] = True
    print(f"  A) Strong Trend: {np.sum(sig_A)} signals")

    # B) MODERATE TREND: ADX 20-30, ROC20>0 -> T3 cross (close crosses above T3)
    sig_B = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(1, ND):
            adx = ADX14[si, di]
            roc20 = ROC20[si, di]
            c = C[si, di]
            t3 = T3[si, di]
            t3_p = T3_prev[si, di]
            c_prev = C[si, di-1]
            if np.isnan(adx) or np.isnan(roc20) or np.isnan(c) or np.isnan(t3) or np.isnan(t3_p) or np.isnan(c_prev):
                continue
            if 20 <= adx <= 30 and roc20 > 0 and c > t3 and c_prev <= t3_p:
                sig_B[si, di] = True
    print(f"  B) Moderate Trend (T3 cross): {np.sum(sig_B)} signals")

    # C) VOLATILITY BREAKOUT: NATR was below 25th pctl, now crosses above 50th
    #    + 20-day breakout
    sig_C = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(1, ND):
            pctl = NATR_pctl[si, di]
            pctl_prev = NATR_pctl_prev[si, di]
            c = C[si, di]
            high20 = HIGH20[si, di]
            if np.isnan(pctl) or np.isnan(pctl_prev) or np.isnan(c) or np.isnan(high20):
                continue
            # NATR squeeze expansion: was below 25th, now above 50th
            if pctl_prev < 25 and pctl >= 50:
                # Confirm with 20-day breakout
                if c >= high20 * 0.995:  # within 0.5% of 20-day high
                    sig_C[si, di] = True
    print(f"  C) Volatility Breakout: {np.sum(sig_C)} signals")

    # D) HIGH VOL MOMENTUM: ADX>25 + NATR pctl > 75 + ROC5 cross
    sig_D = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(1, ND):
            adx = ADX14[si, di]
            pctl = NATR_pctl[si, di]
            roc5 = ROC5[si, di]
            roc5_p = ROC5_prev[si, di]
            if np.isnan(adx) or np.isnan(pctl) or np.isnan(roc5) or np.isnan(roc5_p):
                continue
            if adx > 25 and pctl > 75 and roc5 > 0 and roc5_p <= 0:
                sig_D[si, di] = True
    print(f"  D) High Vol Momentum: {np.sum(sig_D)} signals")

    # E) BROAD MARKET STRENGTH: breadth > 60% + ROC5 cross
    sig_E = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(1, ND):
            breadth = BREADTH[di]
            roc5 = ROC5[si, di]
            roc5_p = ROC5_prev[si, di]
            if np.isnan(breadth) or np.isnan(roc5) or np.isnan(roc5_p):
                continue
            if breadth > 60 and roc5 > 0 and roc5_p <= 0:
                sig_E[si, di] = True
    print(f"  E) Broad Market Strength: {np.sum(sig_E)} signals")

    # F) REGIME-SWITCHING: ADX>25 -> ROC5 cross, ADX<=25 -> T3 cross
    sig_F_roc5 = np.zeros((NS, ND), dtype=bool)
    sig_F_t3 = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(1, ND):
            adx = ADX14[si, di]
            roc5 = ROC5[si, di]
            roc5_p = ROC5_prev[si, di]
            c = C[si, di]
            t3 = T3[si, di]
            t3_p = T3_prev[si, di]
            c_prev = C[si, di-1]
            if np.isnan(adx):
                continue
            if adx > 25:
                if not np.isnan(roc5) and not np.isnan(roc5_p) and roc5 > 0 and roc5_p <= 0:
                    sig_F_roc5[si, di] = True
            else:
                if not np.isnan(c) and not np.isnan(t3) and not np.isnan(t3_p) and not np.isnan(c_prev):
                    if c > t3 and c_prev <= t3_p:
                        sig_F_t3[si, di] = True
    sig_F = sig_F_roc5 | sig_F_t3
    print(f"  F) Regime-Switching: {np.sum(sig_F)} signals (ROC5:{np.sum(sig_F_roc5)} T3:{np.sum(sig_F_t3)})")

    # G) DYNAMIC HOLD: ROC5 cross, hold depends on ADX
    sig_G = np.zeros((NS, ND), dtype=bool)
    hold_G = np.full((NS, ND), 5, dtype=int)  # default 5 days
    for si in range(NS):
        for di in range(1, ND):
            adx = ADX14[si, di]
            roc5 = ROC5[si, di]
            roc5_p = ROC5_prev[si, di]
            if np.isnan(adx) or np.isnan(roc5) or np.isnan(roc5_p):
                continue
            if roc5 > 0 and roc5_p <= 0:
                sig_G[si, di] = True
                if adx > 35:
                    hold_G[si, di] = 10
                elif adx >= 25:
                    hold_G[si, di] = 5
                else:
                    hold_G[si, di] = 3
    print(f"  G) Dynamic Hold: {np.sum(sig_G)} signals")

    # H) COMPOSITE SCORE: ADX>25 + ROC20>0 + NATR<75pctl + Breadth>50%
    #    Score >= 3 (of 4) -> ROC5 cross
    sig_H_thresholds = {}
    for min_score in [2, 3]:
        sig_H = np.zeros((NS, ND), dtype=bool)
        for si in range(NS):
            for di in range(1, ND):
                adx = ADX14[si, di]
                roc20 = ROC20[si, di]
                pctl = NATR_pctl[si, di]
                breadth = BREADTH[di]
                roc5 = ROC5[si, di]
                roc5_p = ROC5_prev[si, di]
                if np.isnan(roc5) or np.isnan(roc5_p):
                    continue
                score = 0
                if not np.isnan(adx) and adx > 25:
                    score += 1
                if not np.isnan(roc20) and roc20 > 0:
                    score += 1
                if not np.isnan(pctl) and pctl < 75:
                    score += 1
                if not np.isnan(breadth) and breadth > 50:
                    score += 1
                if score >= min_score and roc5 > 0 and roc5_p <= 0:
                    sig_H[si, di] = True
        sig_H_thresholds[min_score] = sig_H
        print(f"  H) Composite Score >= {min_score}: {np.sum(sig_H)} signals")

    # I) PER-COMMODITY TRENDING FILTER: ADX>25 AND ROC20>0 -> ROC5 cross
    #    When NOT trending -> skip
    sig_I = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(1, ND):
            adx = ADX14[si, di]
            roc20 = ROC20[si, di]
            roc5 = ROC5[si, di]
            roc5_p = ROC5_prev[si, di]
            if np.isnan(adx) or np.isnan(roc20) or np.isnan(roc5) or np.isnan(roc5_p):
                continue
            if adx > 25 and roc20 > 0 and roc5 > 0 and roc5_p <= 0:
                sig_I[si, di] = True
    print(f"  I) Per-Commodity Trending: {np.sum(sig_I)} signals")

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(sig_arr, hold_days, top_n, wf_test_year=None,
                     hold_arr=None, size_pct=1.0):
        """Generic backtest for a signal array.
        sig_arr: (NS, ND) bool array of signal days
        hold_days: int default hold period
        top_n: max concurrent positions
        wf_test_year: if set, WF test for that year
        hold_arr: (NS, ND) int array of per-signal hold periods (overrides hold_days)
        size_pct: position size fraction (e.g. 0.75 for 75%)
        """
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

        if end_di < start_di + (hold_days or 3) + 2:
            return None

        cash = float(CASH0)
        positions = []
        trades = []

        for di in range(start_di, end_di - 1):
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                positions = []

            # -- Close positions -----------------------------------------
            closed = []
            for pos in positions:
                days_held = di - pos['entry_di']
                hd = pos['hold_days']
                if days_held >= hd:
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
                sc = ROC5[si, di] if not np.isnan(ROC5[si, di]) else 0
                candidates.append((sc, {
                    'si': si, 'sym': syms[si], 'entry_price': ep,
                }))

            if not candidates:
                continue

            candidates.sort(key=lambda x: -x[0])

            n_slots = top_n - len(positions)
            for sc_val, info in candidates[:max(0, n_slots)]:
                si = info['si']
                sym = info['sym']
                price = info['entry_price']
                mult = MULT.get(sym, DEF_MULT)
                alloc_cash = cash * size_pct
                contracts = max(1, int(alloc_cash / (price * mult)))
                cost_in = price * mult * contracts * (1 + COMM)
                if cost_in > cash:
                    contracts = int(cash * 0.9 / (price * mult * (1 + COMM)))
                    cost_in = price * mult * contracts * (1 + COMM) if contracts > 0 else 0
                if contracts <= 0 or cost_in <= 0 or cost_in > cash:
                    continue

                cash -= cost_in
                # Determine hold period
                if hold_arr is not None:
                    hd = int(hold_arr[si, di])
                else:
                    hd = hold_days

                positions.append({
                    'si': si, 'entry_price': price, 'entry_di': entry_di,
                    'lots': contracts, 'dir': 1, 'sym': sym,
                    'hold_days': hd,
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

    # A) Strong Trend: hold x top_n
    for hd in [5, 10, 20]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'A', 'hold_days': hd, 'top_n': tn,
                'size_pct': 1.0,
                'label': f"A_StrongTrend_ADX30_H{hd}_TN{tn}",
                'sig_arr': sig_A, 'hold_arr': None,
            })

    # B) Moderate Trend (T3 cross): hold x top_n
    for hd in [5, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'B', 'hold_days': hd, 'top_n': tn,
                'size_pct': 1.0,
                'label': f"B_ModTrend_T3_ADX20-30_H{hd}_TN{tn}",
                'sig_arr': sig_B, 'hold_arr': None,
            })

    # C) Volatility Breakout: hold x top_n
    for hd in [5, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'C', 'hold_days': hd, 'top_n': tn,
                'size_pct': 1.0,
                'label': f"C_VolBreakout_NATR_H{hd}_TN{tn}",
                'sig_arr': sig_C, 'hold_arr': None,
            })

    # D) High Vol Momentum: hold x top_n x size_pct
    for hd in [5, 10]:
        for tn in [1, 3]:
            for sp in [0.75, 1.0]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': 'D', 'hold_days': hd, 'top_n': tn,
                    'size_pct': sp,
                    'label': f"D_HighVolMom_ADX25_NATR75_H{hd}_TN{tn}_S{sp:.0%}",
                    'sig_arr': sig_D, 'hold_arr': None,
                })

    # E) Broad Market Strength: hold x top_n
    for hd in [5, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'E', 'hold_days': hd, 'top_n': tn,
                'size_pct': 1.0,
                'label': f"E_Breadth60_H{hd}_TN{tn}",
                'sig_arr': sig_E, 'hold_arr': None,
            })

    # F) Regime-Switching: hold x top_n
    for hd in [5, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'F', 'hold_days': hd, 'top_n': tn,
                'size_pct': 1.0,
                'label': f"F_RegimeSwitch_ADX25_H{hd}_TN{tn}",
                'sig_arr': sig_F, 'hold_arr': None,
            })

    # G) Dynamic Hold (ADX-dependent): top_n only (hold is per-signal)
    for tn in [1, 3]:
        cid += 1
        configs.append({
            'id': cid, 'signal': 'G', 'hold_days': 5, 'top_n': tn,
            'size_pct': 1.0,
            'label': f"G_DynamicHold_ADX_TN{tn}",
            'sig_arr': sig_G, 'hold_arr': hold_G,
        })

    # H) Composite Score: threshold x hold x top_n
    for min_score in [2, 3]:
        for hd in [5, 10]:
            for tn in [1, 3]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': 'H', 'hold_days': hd, 'top_n': tn,
                    'size_pct': 1.0,
                    'label': f"H_Composite_S{min_score}_H{hd}_TN{tn}",
                    'sig_arr': sig_H_thresholds[min_score], 'hold_arr': None,
                })

    # I) Per-Commodity Trending Filter: hold x top_n
    for hd in [5, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'I', 'hold_days': hd, 'top_n': tn,
                'size_pct': 1.0,
                'label': f"I_Trending_ADX25_ROC20_H{hd}_TN{tn}",
                'sig_arr': sig_I, 'hold_arr': None,
            })

    print(f"  Total configs: {len(configs)}")

    # ================================================================
    # RUN FULL-PERIOD BACKTEST
    # ================================================================
    print("\n[Backtest] Running full-period sweep...", flush=True)
    results = []
    for i, cfg in enumerate(configs):
        r = run_backtest(cfg['sig_arr'], cfg['hold_days'], cfg['top_n'],
                         hold_arr=cfg.get('hold_arr'), size_pct=cfg.get('size_pct', 1.0))
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
    print(f"\n{'=' * 200}")
    print("  FULL-PERIOD RESULTS -- REGIME-ADAPTIVE STRATEGY, NEXT-OPEN EXECUTION")
    print(f"{'=' * 200}")
    print(f"  {'#':>3} | {'Label':<46} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'AvgHold':>7} | {'Freq/Yr':>7} | {'Final':>14}")
    print("-" * 200)
    for i, r in enumerate(results):
        print(f"  {i+1:>3} | {r['label']:<46} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}% | {r['avg_hold']:>6.1f}d | {r['freq']:>6.1f}/yr | {r['final_cash']:>13,.0f}")

    # ================================================================
    # BEST PER SIGNAL TYPE
    # ================================================================
    sig_names = {
        'A': 'A) STRONG TREND (ADX>30, ROC20>0)',
        'B': 'B) MODERATE TREND (ADX 20-30, T3 cross)',
        'C': 'C) VOLATILITY BREAKOUT (NATR squeeze)',
        'D': 'D) HIGH VOL MOMENTUM (ADX>25+NATR high)',
        'E': 'E) BROAD MARKET STRENGTH (breadth>60%)',
        'F': 'F) REGIME-SWITCHING (ADX threshold)',
        'G': 'G) DYNAMIC HOLD (ADX-dependent hold)',
        'H': 'H) COMPOSITE SCORE (4-factor filter)',
        'I': 'I) PER-COMMODITY TRENDING FILTER',
    }

    print(f"\n{'=' * 200}")
    print("  BEST PER SIGNAL TYPE (Full Period)")
    print(f"{'=' * 200}")
    print(f"  {'Signal':<50} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'AvgHold':>7} | {'Freq/Yr':>7} | Best Config")
    print("-" * 200)

    best_per_sig = {}
    for r in results:
        key = r['config']['signal']
        if key not in best_per_sig:
            best_per_sig[key] = r

    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
        if sig_key in best_per_sig:
            b = best_per_sig[sig_key]
            print(f"  {sig_names.get(sig_key, sig_key):<50} | {b['ann']:>+8.1f}% | {b['wr']:>5.1f}% | {b['n']:>5} | {b['avg_pnl']:>+6.3f}% | {b['mdd']:>6.1f}% | {b['avg_hold']:>6.1f}d | {b['freq']:>6.1f}/yr | {b['label']}")

    # ================================================================
    # SIGNAL TYPE SUMMARY
    # ================================================================
    print(f"\n{'=' * 200}")
    print("  SIGNAL TYPE SUMMARY (Average of all configs per type)")
    print(f"{'=' * 200}")
    print(f"  {'Signal':<50} | {'Avg Ann':>9} | {'Avg WR':>7} | {'Avg N':>7} | {'Avg PnL':>8} | {'Avg MDD':>8} | {'#Positive':>9}")
    print("-" * 160)

    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
        sub = [r for r in results if r['config']['signal'] == sig_key]
        if not sub:
            continue
        avg_ann = np.mean([r['ann'] for r in sub])
        avg_wr = np.mean([r['wr'] for r in sub])
        avg_n = np.mean([r['n'] for r in sub])
        avg_pnl = np.mean([r['avg_pnl'] for r in sub])
        avg_mdd = np.mean([r['mdd'] for r in sub])
        n_pos = sum(1 for r in sub if r['ann'] > 0)
        print(f"  {sig_names.get(sig_key, sig_key):<50} | {avg_ann:>+8.1f}% | {avg_wr:>6.1f}% | {avg_n:>7.0f} | {avg_pnl:>+7.3f}% | {avg_mdd:>7.1f}% | {n_pos:>5}/{len(sub)}")

    # ================================================================
    # WALK-FORWARD (Top configs + best per signal type)
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Collect top 15 overall + best per signal type
    wf_configs = list(results[:15])
    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
        if sig_key in best_per_sig:
            r = best_per_sig[sig_key]
            if r['config'] not in [w['config'] for w in wf_configs]:
                wf_configs.append(r)

    print(f"\n{'=' * 220}")
    print(f"  WALK-FORWARD ({len(wf_configs)} configs)")
    print(f"{'=' * 220}")

    header = f"  {'#':>3} | {'Config':<46} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>7} |"
    header += f" {'Pos':>4} | {'MDD':>7} | {'WR':>6}"
    print(header)
    print("-" * 220)

    wf_rows = []
    for i, r in enumerate(wf_configs):
        cfg = r['config']
        wf_row = {'label': cfg['label'], 'signal': cfg['signal'],
                  'entry': 'next_open', 'windows': {}, 'mdd': {}, 'wr': {}}
        for yr in wf_years:
            wr = run_backtest(cfg['sig_arr'], cfg['hold_days'], cfg['top_n'],
                              wf_test_year=yr, hold_arr=cfg.get('hold_arr'),
                              size_pct=cfg.get('size_pct', 1.0))
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

        row_str = f"  {i+1:>3} | {wf_row['label']:<46} | {avg:>+7.1f}% |"
        for v in vals:
            row_str += f" {v:>+7.1f}% |"
        row_str += f" {pos}/6 | {avg_mdd:>6.1f}% | {avg_wr:>5.1f}%"
        print(row_str)

    # ================================================================
    # WF COMPARISON PER SIGNAL
    # ================================================================
    print(f"\n{'=' * 200}")
    print("  WALK-FORWARD COMPARISON (Best per signal type)")
    print(f"{'=' * 200}")
    header2 = f"  {'Signal':<50} | {'WF Avg':>8} |"
    for yr in wf_years:
        header2 += f" {yr:>7} |"
    header2 += f" {'Pos':>4} | Avg MDD | Avg WR"
    print(header2)
    print("-" * 200)

    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
        wf_match = [w for w in wf_rows if w['signal'] == sig_key]
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            avg = np.mean(vals) if vals else 0
            pos = sum(1 for v in vals if v > 0)
            avg_mdd = np.mean(list(wf['mdd'].values())) if wf['mdd'] else 0
            avg_wr = np.mean(list(wf['wr'].values())) if wf['wr'] else 0
            row_str = f"  {sig_names.get(sig_key, sig_key):<50} | {avg:>+7.1f}% |"
            for v in vals:
                row_str += f" {v:>+7.1f}% |"
            row_str += f" {pos}/6 | {avg_mdd:>6.1f}% | {avg_wr:>5.1f}%"
            print(row_str)

    # ================================================================
    # REGIME DISTRIBUTION ANALYSIS
    # ================================================================
    print(f"\n{'=' * 200}")
    print("  REGIME DISTRIBUTION ANALYSIS")
    print(f"{'=' * 200}")

    # How often is each regime active?
    total_obs = 0
    adx_strong = 0
    adx_moderate = 0
    adx_none = 0
    natr_high = 0
    natr_normal = 0
    natr_low = 0
    roc20_pos = 0
    roc20_neg = 0
    breadth_riskon = 0
    breadth_neutral = 0
    breadth_riskoff = 0
    trending_count = 0  # ADX>25 AND ROC20>0

    for si in range(NS):
        for di in range(100, ND):
            adx = ADX14[si, di]
            roc20 = ROC20[si, di]
            pctl = NATR_pctl[si, di]
            breadth = BREADTH[di]

            if np.isnan(adx) or np.isnan(roc20):
                continue
            total_obs += 1

            if adx > 30: adx_strong += 1
            elif adx > 20: adx_moderate += 1
            else: adx_none += 1

            if roc20 > 0: roc20_pos += 1
            else: roc20_neg += 1

            if not np.isnan(pctl):
                if pctl > 75: natr_high += 1
                elif pctl >= 25: natr_normal += 1
                else: natr_low += 1

            if not np.isnan(breadth):
                if breadth > 60: breadth_riskon += 1
                elif breadth >= 40: breadth_neutral += 1
                else: breadth_riskoff += 1

            if adx > 25 and roc20 > 0:
                trending_count += 1

    if total_obs > 0:
        print(f"  Total observations: {total_obs:,}")
        print(f"  ADX Regime:")
        print(f"    Strong (ADX>30):  {adx_strong:>8,} ({adx_strong/total_obs*100:.1f}%)")
        print(f"    Moderate (20-30): {adx_moderate:>8,} ({adx_moderate/total_obs*100:.1f}%)")
        print(f"    No trend (<20):   {adx_none:>8,} ({adx_none/total_obs*100:.1f}%)")
        print(f"  ROC(20) Direction:")
        print(f"    Positive:         {roc20_pos:>8,} ({roc20_pos/total_obs*100:.1f}%)")
        print(f"    Negative:         {roc20_neg:>8,} ({roc20_neg/total_obs*100:.1f}%)")
        print(f"  NATR Regime:")
        print(f"    High (>75pctl):   {natr_high:>8,} ({natr_high/total_obs*100:.1f}%)")
        print(f"    Normal (25-75):   {natr_normal:>8,} ({natr_normal/total_obs*100:.1f}%)")
        print(f"    Low (<25pctl):    {natr_low:>8,} ({natr_low/total_obs*100:.1f}%)")
        print(f"  Market Breadth:")
        print(f"    Risk-on (>60%):   {breadth_riskon:>8,} ({breadth_riskon/total_obs*100:.1f}%)")
        print(f"    Neutral (40-60%): {breadth_neutral:>8,} ({breadth_neutral/total_obs*100:.1f}%)")
        print(f"    Risk-off (<40%):  {breadth_riskoff:>8,} ({breadth_riskoff/total_obs*100:.1f}%)")
        print(f"  Trending (ADX>25 & ROC20>0): {trending_count:>8,} ({trending_count/total_obs*100:.1f}%)")

    # ================================================================
    # PER-COMMODITY REGIME ANALYSIS (for signal I)
    # ================================================================
    print(f"\n{'=' * 200}")
    print("  PER-COMMODITY REGIME ANALYSIS (Trending %)")
    print(f"{'=' * 200}")

    commodity_trending_pct = []
    for si in range(NS):
        total_di = 0
        trending_di = 0
        for di in range(100, ND):
            adx = ADX14[si, di]
            roc20 = ROC20[si, di]
            if np.isnan(adx) or np.isnan(roc20):
                continue
            total_di += 1
            if adx > 25 and roc20 > 0:
                trending_di += 1
        pct = trending_di / max(total_di, 1) * 100
        commodity_trending_pct.append((syms[si], pct, trending_di, total_di))

    commodity_trending_pct.sort(key=lambda x: -x[1])
    print(f"  {'Symbol':<10} | {'Trending%':>9} | {'Trending Days':>13} | {'Total Days':>10}")
    print("-" * 60)
    for sym, pct, t_days, tot_days in commodity_trending_pct[:20]:
        print(f"  {sym:<10} | {pct:>8.1f}% | {t_days:>13,} | {tot_days:>10,}")
    print("  ...")
    for sym, pct, t_days, tot_days in commodity_trending_pct[-5:]:
        print(f"  {sym:<10} | {pct:>8.1f}% | {t_days:>13,} | {tot_days:>10,}")

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print(f"\n{'=' * 200}")
    print("  FINAL VERDICT: REGIME-ADAPTIVE STRATEGY with Next-Open Execution")
    print(f"{'=' * 200}")
    print()
    print("  KEY QUESTIONS:")
    print("  1. Does regime detection improve over simple ROC(5)?")
    print("  2. Which regime filter is most valuable?")
    print("  3. Per-commodity regime filtering results")
    print("  4. Best configs by annual return")
    print("  5. Any config beating +81.9%?")
    print()

    beats_best = []
    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
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
            wf_avg = np.mean(vals) if vals else 0

        verdict = "POSITIVE" if best['ann'] > 0 else "NEGATIVE"
        genuine = "GENUINE ALPHA" if wf_pos >= 4 and best['ann'] > 0 else ("MARGINAL" if wf_pos >= 3 and best['ann'] > 0 else "NO ALPHA")
        beats = "BEATS +81.9%" if best['ann'] > 81.9 else ("CLOSE" if best['ann'] > 50 else "INSUFFICIENT")

        if best['ann'] > 81.9:
            beats_best.append((sig_key, best))

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

    # Top 10 summary
    print(f"\n  TOP 10 CONFIGS:")
    print(f"  {'#':>3} | {'Label':<46} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'MDD':>7} | WF_Avg | WF_Pos")
    print("-" * 140)
    for i, r in enumerate(results[:10]):
        wf_match = [w for w in wf_rows if w['label'] == r['label']]
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            wf_avg = np.mean(vals)
            wf_pos = sum(1 for v in vals if v > 0)
        else:
            wf_avg = 0
            wf_pos = 0
        print(f"  {i+1:>3} | {r['label']:<46} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['mdd']:>6.1f}% | {wf_avg:>+7.1f}% | {wf_pos}/6")

    # Signal comparison table
    if beats_best:
        print(f"\n  CONFIGS BEATING +81.9% (ROC(5) standalone):")
        for sig_key, best in beats_best:
            wf_match = [w for w in wf_rows if w['signal'] == sig_key]
            wf_pos = 0
            wf_avg = 0
            if wf_match:
                vals = [wf_match[0]['windows'].get(yr, 0) for yr in wf_years]
                wf_pos = sum(1 for v in vals if v > 0)
                wf_avg = np.mean(vals)
            print(f"    {sig_names.get(sig_key)}: {best['ann']:>+8.1f}%  |  WF: {wf_pos}/6 avg {wf_avg:>+.1f}%  |  {best['label']}")
    else:
        print(f"\n  NO config beats +81.9% (ROC(5) standalone)")

    print(f"\n  Total runtime: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
