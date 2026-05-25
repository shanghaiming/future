"""
Alpha Futures V115 — MULTI-TIMEFRAME SYSTEMS with Next-Open Execution
======================================================================
V115 FOCUS: Use information from multiple timeframes to make better decisions.

Build synthetic weekly/monthly bars from daily data. Test 10 multi-TF systems:

A) WEEKLY ROC + DAILY ROC ALIGNMENT
B) MONTHLY TREND + DAILY BREAKOUT
C) TRIPLE TIMEFRAME (5d/10d/20d ROC)
D) WEEKLY SMA + DAILY ROC CROSS
E) DONCHIAN CHANNEL SYSTEM (turtle-style)
F) MOVING AVERAGE RIBBON (perfect order)
G) MOMENTUM DIVERGENCE ACROSS TIMEFRAMES (accelerating only)
H) MULTI-TIMEFRAME ATR SCALING
I) ICHIMOKU-STYLE SYSTEM
J) MULTI-TIMEFRAME SCORING (0-4 score)

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


def main():
    print("=" * 220)
    print("  Alpha Futures V115 — MULTI-TIMEFRAME SYSTEMS with Next-Open Execution")
    print("=" * 220)
    print("\n  10 multi-timeframe signal types (A-J), walk-forward 2020-2025")
    print("  ALL signals at close di, entry at O[si, di+1] (NEXT DAY OPEN)")

    # -- Load data -------------------------------------------------
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ================================================================
    # PRECOMPUTE ALL TIMEFRAMES AND INDICATORS
    # ================================================================
    print("\n[Precompute] ROC, SMA, ATR, Donchian, Ichimoku...", flush=True)
    t0 = time.time()

    # ROC at multiple timeframes
    ROC5  = np.full((NS, ND), np.nan)
    ROC10 = np.full((NS, ND), np.nan)
    ROC20 = np.full((NS, ND), np.nan)
    ROC5_prev  = np.full((NS, ND), np.nan)
    ROC10_prev = np.full((NS, ND), np.nan)
    ROC20_prev = np.full((NS, ND), np.nan)

    # SMA at multiple periods
    SMA5  = np.full((NS, ND), np.nan)
    SMA10 = np.full((NS, ND), np.nan)
    SMA20 = np.full((NS, ND), np.nan)
    SMA50 = np.full((NS, ND), np.nan)

    # ATR daily and weekly
    ATR10 = np.full((NS, ND), np.nan)
    ATR20 = np.full((NS, ND), np.nan)

    # Donchian channels
    DONCH_HIGH10 = np.full((NS, ND), np.nan)   # 10-day high
    DONCH_LOW10  = np.full((NS, ND), np.nan)   # 10-day low
    DONCH_HIGH20 = np.full((NS, ND), np.nan)   # 20-day high
    DONCH_LOW20  = np.full((NS, ND), np.nan)   # 20-day low

    # Ichimoku
    TENKAN = np.full((NS, ND), np.nan)  # (Highest_H_9 + Lowest_L_9) / 2
    KIJUN  = np.full((NS, ND), np.nan)  # (Highest_H_26 + Lowest_L_26) / 2
    TENKAN_prev = np.full((NS, ND), np.nan)
    KIJUN_prev  = np.full((NS, ND), np.nan)

    # Weekly ATR (5-day)
    WEEKLY_ATR = np.full((NS, ND), np.nan)

    for si in range(NS):
        c = C[si].astype(np.float64)
        h = H[si].astype(np.float64)
        l = L[si].astype(np.float64)

        # ROC
        ROC5[si]  = talib.ROC(c, timeperiod=5)
        ROC10[si] = talib.ROC(c, timeperiod=10)
        ROC20[si] = talib.ROC(c, timeperiod=20)

        # Previous-day ROC (shifted by 1)
        ROC5_prev[si]  = np.roll(ROC5[si], 1);  ROC5_prev[si, 0]  = np.nan
        ROC10_prev[si] = np.roll(ROC10[si], 1); ROC10_prev[si, 0] = np.nan
        ROC20_prev[si] = np.roll(ROC20[si], 1); ROC20_prev[si, 0] = np.nan

        # SMA
        SMA5[si]  = talib.SMA(c, timeperiod=5)
        SMA10[si] = talib.SMA(c, timeperiod=10)
        SMA20[si] = talib.SMA(c, timeperiod=20)
        SMA50[si] = talib.SMA(c, timeperiod=50)

        # ATR
        ATR10[si] = talib.ATR(h, l, c, timeperiod=10)
        ATR20[si] = talib.ATR(h, l, c, timeperiod=20)

        # Donchian channels: max/min of PREVIOUS N days (exclude signal day)
        for di in range(11, ND):
            if not np.isnan(h[di-1]):
                DONCH_HIGH10[si, di] = np.nanmax(h[max(0,di-11):di-1])
                DONCH_LOW10[si, di]  = np.nanmin(l[max(0,di-11):di-1])
        for di in range(21, ND):
            if not np.isnan(h[di-1]):
                DONCH_HIGH20[si, di] = np.nanmax(h[max(0,di-21):di-1])
                DONCH_LOW20[si, di]  = np.nanmin(l[max(0,di-21):di-1])

        # Ichimoku-style: Tenkan(9) and Kijun(26)
        for di in range(9, ND):
            TENKAN[si, di] = (np.nanmax(h[max(0,di-9):di]) + np.nanmin(l[max(0,di-9):di])) / 2
        for di in range(26, ND):
            KIJUN[si, di] = (np.nanmax(h[max(0,di-26):di]) + np.nanmin(l[max(0,di-26):di])) / 2
        TENKAN_prev[si] = np.roll(TENKAN[si], 1); TENKAN_prev[si, 0] = np.nan
        KIJUN_prev[si]  = np.roll(KIJUN[si], 1);  KIJUN_prev[si, 0]  = np.nan

        # Weekly ATR: ATR over 5-day windows
        for di in range(25, ND):
            # Weekly range over last 5 days
            h5 = h[max(0,di-5):di]
            l5 = l[max(0,di-5):di]
            c5 = c[max(0,di-5):di]
            if len(h5) >= 5:
                weekly_range = np.nanmax(h5) - np.nanmin(l5)
                WEEKLY_ATR[si, di] = weekly_range

    print(f"  All indicators computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # SIGNAL GENERATION — 10 systems
    # ================================================================
    print("\n[Signals] Computing 10 multi-timeframe systems...", flush=True)
    t0 = time.time()

    # ------------------------------------------------------------------
    # A) WEEKLY ROC + DAILY ROC ALIGNMENT
    # Weekly ROC(5) > 0 AND Daily ROC(5) crosses above 0
    # ------------------------------------------------------------------
    sig_A = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(10, ND):
            w_roc = ROC20[si, di]   # ~monthly proxy for weekly trend (using 20-day)
            d_roc = ROC5[si, di]
            d_roc_prev = ROC5_prev[si, di]
            if np.isnan(w_roc) or np.isnan(d_roc) or np.isnan(d_roc_prev):
                continue
            # Weekly uptrend + daily ROC just crossed above 0
            if w_roc > 0 and d_roc > 0 and d_roc_prev <= 0:
                sig_A[si, di] = True
    print(f"  A) Weekly+Daily ROC alignment: {np.sum(sig_A)} signals")

    # ------------------------------------------------------------------
    # B) MONTHLY TREND + DAILY BREAKOUT
    # Monthly ROC > 0 AND C > max(C[di-10:di]) (10-day breakout)
    # ------------------------------------------------------------------
    sig_B = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(25, ND):
            m_roc = ROC20[si, di]
            if np.isnan(m_roc) or m_roc <= 0:
                continue
            # 10-day close breakout
            c_now = C[si, di-1]  # yesterday's close (signal day)
            if np.isnan(c_now):
                continue
            c_lookback = C[si, max(0,di-11):di-1]
            c_lookback = c_lookback[~np.isnan(c_lookback)]
            if len(c_lookback) >= 5 and c_now > np.max(c_lookback):
                sig_B[si, di] = True
    print(f"  B) Monthly trend + daily breakout: {np.sum(sig_B)} signals")

    # ------------------------------------------------------------------
    # C) TRIPLE TIMEFRAME (5d/10d/20d ROC all positive + fresh turn)
    # ------------------------------------------------------------------
    sig_C = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(25, ND):
            r5  = ROC5[si, di]
            r10 = ROC10[si, di]
            r20 = ROC20[si, di]
            if np.isnan(r5) or np.isnan(r10) or np.isnan(r20):
                continue
            if r5 > 0 and r10 > 0 and r20 > 0:
                # At least one just turned positive
                r5p  = ROC5_prev[si, di]
                r10p = ROC10_prev[si, di]
                r20p = ROC20_prev[si, di]
                fresh = False
                if not np.isnan(r5p) and r5p <= 0:
                    fresh = True
                if not np.isnan(r10p) and r10p <= 0:
                    fresh = True
                if not np.isnan(r20p) and r20p <= 0:
                    fresh = True
                if fresh:
                    sig_C[si, di] = True
    print(f"  C) Triple TF ROC alignment + fresh turn: {np.sum(sig_C)} signals")

    # ------------------------------------------------------------------
    # D) WEEKLY SMA + DAILY ROC CROSS
    # C > SMA(20) AND ROC(5) crosses above 0
    # ------------------------------------------------------------------
    sig_D = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(25, ND):
            sma = SMA20[si, di]
            d_roc = ROC5[si, di]
            d_roc_prev = ROC5_prev[si, di]
            if np.isnan(sma) or np.isnan(d_roc) or np.isnan(d_roc_prev):
                continue
            c_now = C[si, di-1]
            if np.isnan(c_now):
                continue
            if c_now > sma and d_roc > 0 and d_roc_prev <= 0:
                sig_D[si, di] = True
    print(f"  D) Weekly SMA + daily ROC cross: {np.sum(sig_D)} signals")

    # ------------------------------------------------------------------
    # E) DONCHIAN CHANNEL SYSTEM (turtle-style)
    # Entry: C[di-1] > max(H[di-21:di-1])  (20-day high breakout)
    # Exit tracked via hold_days parameter (fixed hold for backtest)
    # ------------------------------------------------------------------
    sig_E = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(25, ND):
            c_now = C[si, di-1]
            if np.isnan(c_now):
                continue
            h20 = DONCH_HIGH20[si, di]
            if np.isnan(h20):
                continue
            if c_now > h20:
                sig_E[si, di] = True
    print(f"  E) Donchian 20-day breakout: {np.sum(sig_E)} signals")

    # ------------------------------------------------------------------
    # F) MOVING AVERAGE RIBBON (perfect order just established)
    # SMA5 > SMA10 > SMA20 > SMA50, and was NOT perfect order yesterday
    # ------------------------------------------------------------------
    def is_perfect_order(si, di):
        s5  = SMA5[si, di]
        s10 = SMA10[si, di]
        s20 = SMA20[si, di]
        s50 = SMA50[si, di]
        if np.isnan(s5) or np.isnan(s10) or np.isnan(s20) or np.isnan(s50):
            return False
        return s5 > s10 > s20 > s50

    sig_F = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(55, ND):
            if is_perfect_order(si, di) and not is_perfect_order(si, di-1):
                sig_F[si, di] = True
    print(f"  F) MA ribbon perfect order (new): {np.sum(sig_F)} signals")

    # ------------------------------------------------------------------
    # G) MOMENTUM ACCELERATION (ROC5>0 AND ROC20>ROC10>ROC5)
    # ------------------------------------------------------------------
    sig_G_accel = np.zeros((NS, ND), dtype=bool)
    sig_G_decel = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(25, ND):
            r5  = ROC5[si, di]
            r10 = ROC10[si, di]
            r20 = ROC20[si, di]
            if np.isnan(r5) or np.isnan(r10) or np.isnan(r20):
                continue
            # Accelerating: momentum positive and increasing across TF
            if r5 > 0 and r20 > r10 > r5:
                sig_G_accel[si, di] = True
            # Decelerating: positive but weakening
            if r5 > 0 and r20 < r10 < r5:
                sig_G_decel[si, di] = True
    print(f"  G) Momentum accelerating: {np.sum(sig_G_accel)} signals")
    print(f"  G) Momentum decelerating: {np.sum(sig_G_decel)} signals")

    # ------------------------------------------------------------------
    # H) MULTI-TIMEFRAME ATR SCALING
    # Weekly ATR > Daily ATR * 4 AND ROC(5) > 0
    # ------------------------------------------------------------------
    sig_H = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(25, ND):
            w_atr = WEEKLY_ATR[si, di]
            d_atr = ATR10[si, di]
            r5 = ROC5[si, di]
            if np.isnan(w_atr) or np.isnan(d_atr) or np.isnan(r5):
                continue
            if w_atr > d_atr * 4 and r5 > 0:
                sig_H[si, di] = True
    print(f"  H) Weekly ATR scaling + momentum: {np.sum(sig_H)} signals")

    # ------------------------------------------------------------------
    # I) ICHIMOKU-STYLE SYSTEM
    # C > Tenkan AND Tenkan > Kijun AND Tenkan just crossed above Kijun
    # ------------------------------------------------------------------
    sig_I = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(30, ND):
            tk = TENKAN[si, di]
            kj = KIJUN[si, di]
            tk_p = TENKAN_prev[si, di]
            kj_p = KIJUN_prev[si, di]
            if np.isnan(tk) or np.isnan(kj) or np.isnan(tk_p) or np.isnan(kj_p):
                continue
            c_now = C[si, di-1]
            if np.isnan(c_now):
                continue
            # Price above both, Tenkan > Kijun, and cross just happened
            if c_now > tk and tk > kj and tk_p <= kj_p:
                sig_I[si, di] = True
    print(f"  I) Ichimoku-style Tenkan/Kijun cross: {np.sum(sig_I)} signals")

    # ------------------------------------------------------------------
    # J) MULTI-TIMEFRAME SCORING (0-4)
    # +1: ROC(5)>0, +1: ROC(10)>0, +1: ROC(20)>0, +1: C>SMA(50)
    # Only trade when score = 4 (perfect alignment)
    # ------------------------------------------------------------------
    sig_J = np.zeros((NS, ND), dtype=bool)
    MTF_SCORE = np.zeros((NS, ND), dtype=np.int8)
    for si in range(NS):
        for di in range(55, ND):
            score = 0
            if not np.isnan(ROC5[si, di]) and ROC5[si, di] > 0:
                score += 1
            if not np.isnan(ROC10[si, di]) and ROC10[si, di] > 0:
                score += 1
            if not np.isnan(ROC20[si, di]) and ROC20[si, di] > 0:
                score += 1
            c_now = C[si, di-1]
            sma50 = SMA50[si, di]
            if not np.isnan(c_now) and not np.isnan(sma50) and c_now > sma50:
                score += 1
            MTF_SCORE[si, di] = score
            if score == 4:
                sig_J[si, di] = True
    print(f"  J) MTF score=4 (perfect alignment): {np.sum(sig_J)} signals")

    # Also build score >= 3 variant
    sig_J3 = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(55, ND):
            if MTF_SCORE[si, di] >= 3:
                sig_J3[si, di] = True
    print(f"  J3) MTF score>=3: {np.sum(sig_J3)} signals")

    print(f"  All signals computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # BACKTEST ENGINE (same pattern as V110)
    # ================================================================
    def run_backtest(sig_arr, hold_days, top_n, wf_test_year=None,
                     score_arr=None, use_donch_exit=False):
        """Generic backtest for a signal array.
        score_arr: if provided, use this for ranking (higher = better).
        use_donch_exit: if True, exit on 10-day low breakdown (Donchian E).
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

                # Check Donchian exit (10-day low breakdown)
                donch_exit = False
                if use_donch_exit and days_held >= 1:
                    c_now = C[pos['si'], di]
                    dl = DONCH_LOW10[si, di] if 'si' in pos else np.nan
                    # Actually compute inline
                    dl_val = DONCH_LOW10[pos['si'], di]
                    if not np.isnan(c_now) and not np.isnan(dl_val) and c_now < dl_val:
                        donch_exit = True

                if days_held >= pos['hold_days'] or donch_exit:
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

            # Sort by score descending
            candidates.sort(key=lambda x: -x[0])

            # Open positions
            n_slots = top_n - len(positions)
            for sc_val, si, price in candidates[:max(0, n_slots)]:
                sym = syms[si]
                mult = MULT.get(sym, DEF_MULT)
                contracts = max(1, int(cash / (price * mult)))
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

        # Max drawdown from equity curve
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

    # A) Weekly ROC + Daily ROC alignment
    for hd in [5, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'A',
                'hold_days': hd, 'top_n': tn,
                'label': f"A_WkDlyROCAlign_H{hd}_TN{tn}",
                'sig_arr': sig_A,
            })

    # B) Monthly trend + daily breakout
    for hd in [5, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'B',
                'hold_days': hd, 'top_n': tn,
                'label': f"B_MoTrendDlyBO_H{hd}_TN{tn}",
                'sig_arr': sig_B,
            })

    # C) Triple TF alignment + fresh turn
    for hd in [5, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'C',
                'hold_days': hd, 'top_n': tn,
                'label': f"C_TripleTFFresh_H{hd}_TN{tn}",
                'sig_arr': sig_C,
            })

    # D) Weekly SMA + daily ROC cross
    for hd in [5, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'D',
                'hold_days': hd, 'top_n': tn,
                'label': f"D_WkSmaDlyRocX_H{hd}_TN{tn}",
                'sig_arr': sig_D,
            })

    # E) Donchian 20-day breakout
    for hd in [5, 10, 15, 20]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'E',
                'hold_days': hd, 'top_n': tn,
                'label': f"E_Donchian20_H{hd}_TN{tn}",
                'sig_arr': sig_E,
            })

    # F) MA ribbon perfect order (new)
    for hd in [10, 15, 20]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'F',
                'hold_days': hd, 'top_n': tn,
                'label': f"F_MARibbon_H{hd}_TN{tn}",
                'sig_arr': sig_F,
            })

    # G) Momentum acceleration
    for hd in [5, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'G_accel',
                'hold_days': hd, 'top_n': tn,
                'label': f"G_Accel_H{hd}_TN{tn}",
                'sig_arr': sig_G_accel,
            })

    # G_decel) Momentum deceleration (for comparison)
    for hd in [5, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'G_decel',
                'hold_days': hd, 'top_n': tn,
                'label': f"G_Decel_H{hd}_TN{tn}",
                'sig_arr': sig_G_decel,
            })

    # H) Weekly ATR scaling + momentum
    for hd in [5, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'H',
                'hold_days': hd, 'top_n': tn,
                'label': f"H_WkAtrScale_H{hd}_TN{tn}",
                'sig_arr': sig_H,
            })

    # I) Ichimoku-style
    for hd in [5, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'I',
                'hold_days': hd, 'top_n': tn,
                'label': f"I_Ichimoku_H{hd}_TN{tn}",
                'sig_arr': sig_I,
            })

    # J) MTF scoring = 4 (perfect alignment), rank by ROC5
    for hd in [5, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'J',
                'hold_days': hd, 'top_n': tn,
                'label': f"J_MTFScore4_H{hd}_TN{tn}",
                'sig_arr': sig_J,
            })

    # J3) MTF scoring >= 3
    for hd in [5, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'J3',
                'hold_days': hd, 'top_n': tn,
                'label': f"J3_MTFScore3_H{hd}_TN{tn}",
                'sig_arr': sig_J3,
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
        r = run_backtest(cfg['sig_arr'], cfg['hold_days'], cfg['top_n'])
        if r and r['n'] >= 5:
            r['config'] = cfg
            r['label'] = cfg['label']
            r['signal'] = cfg['signal']
            results.append(r)

    print(f"\n  Done ({time.time()-t1:.0f}s, {len(results)} configs with >= 5 trades)")
    results.sort(key=lambda x: -x['ann'])

    # ================================================================
    # PRINT TOP 30
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  TOP 30 RESULTS (sorted by annual return)")
    print(f"{'=' * 160}")
    print(f"  {'#':>3} | {'Label':<30} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'MDD':>7} | {'AvgPnL':>8} | {'AvgHold':>7} | {'Freq':>6}")
    print("-" * 160)
    for i, r in enumerate(results[:30]):
        print(f"  {i+1:>3} | {r['label']:<30} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['mdd']:>6.1f}% | {r['avg_pnl']:>+7.3f}% | {r['avg_hold']:>6.1f}d | {r['freq']:>5.1f}/yr")

    # ================================================================
    # BEST PER SIGNAL TYPE
    # ================================================================
    sig_names = {
        'A': 'A) Weekly+Daily ROC align',
        'B': 'B) Monthly trend+Daily BO',
        'C': 'C) Triple TF fresh turn',
        'D': 'D) Weekly SMA+Daily ROC',
        'E': 'E) Donchian 20d breakout',
        'F': 'F) MA ribbon perfect',
        'G_accel': 'G) Momentum accel',
        'G_decel': 'G) Momentum decel',
        'H': 'H) Weekly ATR scale',
        'I': 'I) Ichimoku-style',
        'J': 'J) MTF score=4',
        'J3': 'J3) MTF score>=3',
    }

    print(f"\n  BEST PER SIGNAL TYPE:")
    print(f"  {'Signal':<35} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'MDD':>7} | {'AvgPnL':>8} | Positive")
    print("-" * 120)

    best_per_sig = {}
    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G_accel', 'G_decel', 'H', 'I', 'J', 'J3']:
        sub = [r for r in results if r['signal'] == sig_key]
        if not sub:
            print(f"  {sig_names.get(sig_key, sig_key):<35} | NO RESULTS")
            continue
        best = sub[0]
        best_per_sig[sig_key] = best
        n_pos = sum(1 for r in sub if r['ann'] > 0)
        print(f"  {sig_names.get(sig_key, sig_key):<35} | {best['ann']:>+8.1f}% | {best['wr']:>5.1f}% | {best['n']:>5} | {best['mdd']:>6.1f}% | {best['avg_pnl']:>+7.3f}% | {n_pos}/{len(sub)}")

    # ================================================================
    # WALK-FORWARD
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Collect top 15 + best per signal type
    wf_configs = list(results[:15])
    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G_accel', 'G_decel', 'H', 'I', 'J', 'J3']:
        if sig_key in best_per_sig:
            r = best_per_sig[sig_key]
            if r not in wf_configs:
                wf_configs.append(r)

    print(f"\n{'=' * 220}")
    print(f"  WALK-FORWARD ({len(wf_configs)} configs, years 2020-2025)")
    print(f"{'=' * 220}")

    header = f"  {'#':>3} | {'Config':<32} | {'Avg':>8} |"
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
            wr = run_backtest(cfg['sig_arr'], cfg['hold_days'], cfg['top_n'], wf_test_year=yr)
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
    print(f"\n{'=' * 200}")
    print("  WALK-FORWARD COMPARISON (Best per signal type)")
    print(f"{'=' * 200}")
    header2 = f"  {'Signal':<35} | {'WF Avg':>8} |"
    for yr in wf_years:
        header2 += f" {yr:>7} |"
    header2 += f" {'Pos':>4} | Avg MDD | Avg WR"
    print(header2)
    print("-" * 200)

    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G_accel', 'G_decel', 'H', 'I', 'J', 'J3']:
        wf_match = [w for w in wf_rows if w['signal'] == sig_key]
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            avg = np.mean(vals) if vals else 0
            pos = sum(1 for v in vals if v > 0)
            avg_mdd = np.mean(list(wf['mdd'].values())) if wf['mdd'] else 0
            avg_wr = np.mean(list(wf['wr'].values())) if wf['wr'] else 0
            row_str = f"  {sig_names.get(sig_key, sig_key):<35} | {avg:>+7.1f}% |"
            for v in vals:
                row_str += f" {v:>+7.1f}% |"
            row_str += f" {pos}/6 | {avg_mdd:>6.1f}% | {avg_wr:>5.1f}%"
            print(row_str)
        else:
            print(f"  {sig_names.get(sig_key, sig_key):<35} | NO DATA")

    # ================================================================
    # ACCELERATION vs DECELERATION COMPARISON
    # ================================================================
    print(f"\n{'=' * 100}")
    print("  KEY COMPARISON: Momentum ACCELERATION vs DECELERATION")
    print(f"{'=' * 100}")
    for sig_key, label in [('G_accel', 'Accelerating'), ('G_decel', 'Decelerating')]:
        sub = [r for r in results if r['signal'] == sig_key]
        if sub:
            best = sub[0]
            wf_match = [w for w in wf_rows if w['signal'] == sig_key]
            if wf_match:
                vals = [wf_match[0]['windows'].get(yr, 0) for yr in wf_years]
                wf_avg = np.mean(vals)
                wf_pos = sum(1 for v in vals if v > 0)
            else:
                wf_avg = 0; wf_pos = 0
            print(f"  {label:20s} | Ann={best['ann']:>+8.1f}% | WR={best['wr']:>5.1f}% | N={best['n']:>4} | WF_Avg={wf_avg:>+7.1f}% | WF_Pos={wf_pos}/6")

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print(f"\n{'=' * 200}")
    print("  FINAL VERDICT: MULTI-TIMEFRAME SYSTEMS")
    print(f"{'=' * 200}")
    print()
    print("  KEY QUESTIONS:")
    print("  1. Which multi-timeframe system works best?")
    print("  2. Does multi-TF alignment improve over single ROC(5)?")
    print("  3. Ichimoku-style system results?")
    print("  4. MA ribbon / Donchian system results?")
    print("  5. Best configs by annual return, any beating +81.9%?")
    print()

    beats_best = []
    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G_accel', 'G_decel', 'H', 'I', 'J', 'J3']:
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
    print(f"  {'#':>3} | {'Label':<32} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'MDD':>7} | WF_Avg | WF_Pos")
    print("-" * 130)
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
        print(f"  {i+1:>3} | {r['label']:<32} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['mdd']:>6.1f}% | {wf_avg:>+7.1f}% | {wf_pos}/6")

    # Configs beating baseline
    if beats_best:
        print(f"\n  CONFIGS BEATING +81.9% (ROC(5) standalone baseline):")
        for sig_key, best in beats_best:
            wf_match = [w for w in wf_rows if w['signal'] == sig_key]
            wf_pos = 0
            if wf_match:
                vals = [wf_match[0]['windows'].get(yr, 0) for yr in wf_years]
                wf_pos = sum(1 for v in vals if v > 0)
            print(f"    {sig_names.get(sig_key)}: {best['ann']:>+8.1f}%  |  WF: {wf_pos}/6  |  {best['label']}")
    else:
        print(f"\n  NO config beats +81.9% (ROC(5) standalone baseline)")

    print(f"\n  Total runtime: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
