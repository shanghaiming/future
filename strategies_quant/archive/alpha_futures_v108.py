"""
Alpha Futures V108 -- Deep Optimization of ROC(5) Signal
========================================================
Current practical champion: ROC(5) crossover at +81.9% annual.

V108 IDEA: Systematically sweep all ROC parameters and filters to find
the truly optimal configuration. 9 sweep dimensions (A-I).

ROC crossover signal: ROC(period)[di] > 0 AND ROC(period)[di-1] <= 0
(crossing from negative to positive), buy at next open.

ALL signals computed at close of day di using data up to and including di.
Entry at O[si, di+1] (NEXT DAY OPEN).
Exit at C[si, di+hold] (close price hold days later).
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
    print("Alpha Futures V108 -- Deep Optimization of ROC(5) Signal")
    print("=" * 200)
    print("\n  Systematic sweep of ROC parameters, filters, and combinations.")
    print("  ALL signals at close di, entry at O[si, di+1] (NEXT DAY OPEN)")

    # -- Load data -------------------------------------------------
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ================================================================
    # PRECOMPUTE TA-LIB INDICATORS
    # ================================================================
    print("\n[Indicators] Computing TA-Lib indicators...", flush=True)
    t0 = time.time()

    # ROC for all periods we need
    roc_periods = [2, 3, 4, 5, 7, 10, 14, 20]
    ROC = {}
    for p in roc_periods:
        ROC[p] = np.full((NS, ND), np.nan)

    # Other indicators needed for filters
    ADX = np.full((NS, ND), np.nan)
    PLUS_DI = np.full((NS, ND), np.nan)
    MINUS_DI = np.full((NS, ND), np.nan)
    RSI = np.full((NS, ND), np.nan)
    MACD_hist = np.full((NS, ND), np.nan)
    ATR = np.full((NS, ND), np.nan)
    SMA20 = np.full((NS, ND), np.nan)
    SMA50 = np.full((NS, ND), np.nan)
    vol_sma20 = np.full((NS, ND), np.nan)

    for si in range(NS):
        c = C[si].astype(np.float64)
        o = O[si].astype(np.float64)
        h = H[si].astype(np.float64)
        l = L[si].astype(np.float64)
        v = V[si].astype(np.float64)

        # ROC for all periods
        for p in roc_periods:
            ROC[p][si] = talib.ROC(c, timeperiod=p)

        # Trend indicators
        ADX[si] = talib.ADX(h, l, c, timeperiod=14)
        PLUS_DI[si] = talib.PLUS_DI(h, l, c, timeperiod=14)
        MINUS_DI[si] = talib.MINUS_DI(h, l, c, timeperiod=14)

        # Momentum indicators
        RSI[si] = talib.RSI(c, timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
        MACD_hist[si] = macd_hist

        # Volatility
        ATR[si] = talib.ATR(h, l, c, timeperiod=14)

        # Moving averages
        SMA20[si] = talib.SMA(c, timeperiod=20)
        SMA50[si] = talib.SMA(c, timeperiod=50)

        # Volume SMA
        vol_sma20[si] = talib.SMA(v, timeperiod=20)

        if (si + 1) % 10 == 0 or si == NS - 1:
            print(f"  ... {si+1}/{NS} commodities done ({time.time()-t0:.1f}s)", flush=True)

    print(f"  All indicators computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # BACKTEST ENGINE (vectorized signal, supports trailing stop)
    # ================================================================
    def run_backtest(signal_arr, hold_days, top_n, score_arr=None,
                     trailing_stop=False, trail_atr_mult=2.0, max_hold=None,
                     wf_test_year=None, label=""):

        if max_hold is None:
            max_hold = hold_days

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

        if end_di < start_di + max_hold + 2:
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

                if trailing_stop:
                    # Check trailing stop: 2*ATR below highest close since entry
                    cur_close = C[pos['si'], di]
                    cur_atr = ATR[pos['si'], di]
                    if not np.isnan(cur_close):
                        if cur_close > pos['highest_close']:
                            pos['highest_close'] = cur_close
                        if not np.isnan(cur_atr) and pos['highest_close'] > 0:
                            trail_price = pos['highest_close'] - trail_atr_mult * cur_atr
                            if cur_close < trail_price:
                                # Trailing stop hit
                                exit_price = cur_close
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
                                continue

                # Max hold exit
                if days_held >= max_hold:
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
                if not signal_arr[si, di]:
                    continue
                if any(p['si'] == si for p in positions):
                    continue
                ep = O[si, entry_di]
                if np.isnan(ep) or ep <= 0:
                    continue
                sc = score_arr[si, di] if score_arr is not None and not np.isnan(score_arr[si, di]) else 0
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
                cost_in = price * mult * contracts * (1 + COMM)
                if cost_in > cash:
                    contracts = int(cash * 0.9 / (price * mult * (1 + COMM)))
                    cost_in = price * mult * contracts * (1 + COMM) if contracts > 0 else 0
                if contracts <= 0 or cost_in <= 0 or cost_in > cash:
                    continue

                cash -= cost_in
                pos = {
                    'si': si, 'entry_price': price, 'entry_di': entry_di,
                    'lots': contracts, 'dir': 1, 'sym': sym,
                    'hold_days': hold_days, 'highest_close': price,
                }
                positions.append(pos)

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
    # HELPER: Generate crossover signal for a given ROC period
    # ================================================================
    def make_roc_cross_signal(roc_period):
        """ROC(period)[di] > 0 AND ROC(period)[di-1] <= 0"""
        sig = np.zeros((NS, ND), dtype=bool)
        roc = ROC[roc_period]
        for si in range(NS):
            for di in range(1, ND):
                cur = roc[si, di]
                prev = roc[si, di - 1]
                if np.isnan(cur) or np.isnan(prev):
                    continue
                if cur > 0 and prev <= 0:
                    sig[si, di] = True
        return sig

    def make_roc_cross_signal_with_filter(roc_period, filter_fn):
        """ROC cross + additional filter function(si, di) -> bool"""
        sig = np.zeros((NS, ND), dtype=bool)
        roc = ROC[roc_period]
        for si in range(NS):
            for di in range(1, ND):
                cur = roc[si, di]
                prev = roc[si, di - 1]
                if np.isnan(cur) or np.isnan(prev):
                    continue
                if cur > 0 and prev <= 0 and filter_fn(si, di):
                    sig[si, di] = True
        return sig

    # ================================================================
    # SWEEP A) ROC PERIOD: periods x hold x top_n
    # ================================================================
    print("\n" + "=" * 200)
    print("  SWEEP A) ROC PERIOD: Which period is truly optimal?")
    print("=" * 200)

    sig_A = {}  # key: period -> signal array
    for p in roc_periods:
        t1 = time.time()
        sig = make_roc_cross_signal(p)
        sig_A[p] = sig
        print(f"  ROC({p}) crossover: {np.sum(sig)} signals ({time.time()-t1:.1f}s)", flush=True)

    configs_A = []
    for p in roc_periods:
        for hd in [3, 5, 7, 10, 15, 20]:
            for tn in [1, 3]:
                configs_A.append({
                    'label': f'A_ROC({p})_H{hd}_TN{tn}',
                    'signal': sig_A[p],
                    'hold_days': hd,
                    'top_n': tn,
                    'score': ROC[p],  # score by ROC magnitude
                    'group': 'A',
                    'roc_period': p,
                })
    print(f"  Total A configs: {len(configs_A)}")

    # ================================================================
    # SWEEP B) ROC MAGNITUDE FILTER
    # ================================================================
    print("\n" + "=" * 200)
    print("  SWEEP B) ROC MAGNITUDE FILTER: Filtering weak crossovers")
    print("=" * 200)

    sig_B = {}
    # B1: ROC(5) > 0.5%
    sig_B['>0.5'] = make_roc_cross_signal_with_filter(5, lambda si, di: ROC[5][si, di] > 0.5)
    print(f"  B) ROC(5)>0.5% cross: {np.sum(sig_B['>0.5'])} signals", flush=True)

    # B2: ROC(5) > 1.0%
    sig_B['>1.0'] = make_roc_cross_signal_with_filter(5, lambda si, di: ROC[5][si, di] > 1.0)
    print(f"  B) ROC(5)>1.0% cross: {np.sum(sig_B['>1.0'])} signals", flush=True)

    # B3: ROC(5) > 2.0%
    sig_B['>2.0'] = make_roc_cross_signal_with_filter(5, lambda si, di: ROC[5][si, di] > 2.0)
    print(f"  B) ROC(5)>2.0% cross: {np.sum(sig_B['>2.0'])} signals", flush=True)

    # B4: ROC(5) > 1*STD(ROC_20) -- above 1 standard deviation of ROC over 20 days
    sig_B_std = np.zeros((NS, ND), dtype=bool)
    roc5 = ROC[5]
    for si in range(NS):
        for di in range(20, ND):
            cur = roc5[si, di]
            prev = roc5[si, di - 1]
            if np.isnan(cur) or np.isnan(prev):
                continue
            if cur > 0 and prev <= 0:
                window = roc5[si, max(0, di-20):di]
                window = window[~np.isnan(window)]
                if len(window) >= 10:
                    std_val = np.std(window)
                    if std_val > 0 and cur > std_val:
                        sig_B_std[si, di] = True
    sig_B['>1std'] = sig_B_std
    print(f"  B) ROC(5)>1*STD(20) cross: {np.sum(sig_B_std)} signals", flush=True)

    # B baseline
    sig_B['>0'] = sig_A[5]

    configs_B = []
    mag_labels = {'>0': '>0(baseline)', '>0.5': '>0.5%', '>1.0': '>1.0%', '>2.0': '>2.0%', '>1std': '>1*STD20'}
    for key in ['>0', '>0.5', '>1.0', '>2.0', '>1std']:
        for hd in [5]:
            for tn in [1, 3]:
                configs_B.append({
                    'label': f'B_ROC5{mag_labels[key]}_H{hd}_TN{tn}',
                    'signal': sig_B[key],
                    'hold_days': hd,
                    'top_n': tn,
                    'score': ROC[5],
                    'group': 'B',
                    'mag_filter': key,
                })
    print(f"  Total B configs: {len(configs_B)}")

    # ================================================================
    # SWEEP C) ROC + CONFIRMATION FILTERS
    # ================================================================
    print("\n" + "=" * 200)
    print("  SWEEP C) ROC + CONFIRMATION FILTERS")
    print("=" * 200)

    # C1: ROC(5) cross AND ADX > 20
    sig_C1 = make_roc_cross_signal_with_filter(5, lambda si, di: (not np.isnan(ADX[si, di])) and ADX[si, di] > 20)
    print(f"  C1) ROC(5)cross AND ADX>20: {np.sum(sig_C1)} signals", flush=True)

    # C2: ROC(5) cross AND ADX > 25
    sig_C2 = make_roc_cross_signal_with_filter(5, lambda si, di: (not np.isnan(ADX[si, di])) and ADX[si, di] > 25)
    print(f"  C2) ROC(5)cross AND ADX>25: {np.sum(sig_C2)} signals", flush=True)

    # C3: ROC(5) cross AND C > SMA(C, 20)
    sig_C3 = make_roc_cross_signal_with_filter(5, lambda si, di: (not np.isnan(C[si, di])) and (not np.isnan(SMA20[si, di])) and C[si, di] > SMA20[si, di])
    print(f"  C3) ROC(5)cross AND C>SMA20: {np.sum(sig_C3)} signals", flush=True)

    # C4: ROC(5) cross AND C > SMA(C, 50)
    sig_C4 = make_roc_cross_signal_with_filter(5, lambda si, di: (not np.isnan(C[si, di])) and (not np.isnan(SMA50[si, di])) and C[si, di] > SMA50[si, di])
    print(f"  C4) ROC(5)cross AND C>SMA50: {np.sum(sig_C4)} signals", flush=True)

    # C5: ROC(5) cross AND V > 1.2 * SMA(V, 20)
    sig_C5 = make_roc_cross_signal_with_filter(5, lambda si, di: (not np.isnan(V[si, di])) and (not np.isnan(vol_sma20[si, di])) and vol_sma20[si, di] > 0 and V[si, di] > 1.2 * vol_sma20[si, di])
    print(f"  C5) ROC(5)cross AND V>1.2*SMA(V,20): {np.sum(sig_C5)} signals", flush=True)

    # C6: ROC(5) cross AND MACD_hist > 0
    sig_C6 = make_roc_cross_signal_with_filter(5, lambda si, di: (not np.isnan(MACD_hist[si, di])) and MACD_hist[si, di] > 0)
    print(f"  C6) ROC(5)cross AND MACD_hist>0: {np.sum(sig_C6)} signals", flush=True)

    # C7: ROC(5) cross AND RSI > 50
    sig_C7 = make_roc_cross_signal_with_filter(5, lambda si, di: (not np.isnan(RSI[si, di])) and RSI[si, di] > 50)
    print(f"  C7) ROC(5)cross AND RSI>50: {np.sum(sig_C7)} signals", flush=True)

    # C8: ROC(5) cross AND PLUS_DI > MINUS_DI
    sig_C8 = make_roc_cross_signal_with_filter(5, lambda si, di: (not np.isnan(PLUS_DI[si, di])) and (not np.isnan(MINUS_DI[si, di])) and PLUS_DI[si, di] > MINUS_DI[si, di])
    print(f"  C8) ROC(5)cross AND PLUS_DI>MINUS_DI: {np.sum(sig_C8)} signals", flush=True)

    configs_C = []
    c_sigs = [
        ('C1', 'ADX>20', sig_C1),
        ('C2', 'ADX>25', sig_C2),
        ('C3', 'C>SMA20', sig_C3),
        ('C4', 'C>SMA50', sig_C4),
        ('C5', 'V>1.2*SMA(V,20)', sig_C5),
        ('C6', 'MACD_hist>0', sig_C6),
        ('C7', 'RSI>50', sig_C7),
        ('C8', 'PLUS_DI>MINUS_DI', sig_C8),
    ]
    for cid, cdesc, csig in c_sigs:
        for hd in [5]:
            for tn in [1, 3]:
                configs_C.append({
                    'label': f'C_{cid}_{cdesc}_H{hd}_TN{tn}',
                    'signal': csig,
                    'hold_days': hd,
                    'top_n': tn,
                    'score': ROC[5],
                    'group': 'C',
                    'filter': cid,
                })
    print(f"  Total C configs: {len(configs_C)}")

    # ================================================================
    # SWEEP D) ROC + DOUBLE CONFIRMATION
    # ================================================================
    print("\n" + "=" * 200)
    print("  SWEEP D) ROC + DOUBLE CONFIRMATION (ROC cross + 2 filters)")
    print("=" * 200)

    # D1: ROC(5) cross AND ADX>25 AND C>SMA20
    sig_D1 = make_roc_cross_signal_with_filter(5,
        lambda si, di: (not np.isnan(ADX[si, di])) and ADX[si, di] > 25
                     and (not np.isnan(C[si, di])) and (not np.isnan(SMA20[si, di]))
                     and C[si, di] > SMA20[si, di])
    print(f"  D1) ROC(5)cross AND ADX>25 AND C>SMA20: {np.sum(sig_D1)} signals", flush=True)

    # D2: ROC(5) cross AND ADX>25 AND V>1.2*SMA(V,20)
    sig_D2 = make_roc_cross_signal_with_filter(5,
        lambda si, di: (not np.isnan(ADX[si, di])) and ADX[si, di] > 25
                     and (not np.isnan(V[si, di])) and (not np.isnan(vol_sma20[si, di]))
                     and vol_sma20[si, di] > 0 and V[si, di] > 1.2 * vol_sma20[si, di])
    print(f"  D2) ROC(5)cross AND ADX>25 AND V>1.2*SMA(V,20): {np.sum(sig_D2)} signals", flush=True)

    # D3: ROC(5) cross AND C>SMA50 AND RSI>50
    sig_D3 = make_roc_cross_signal_with_filter(5,
        lambda si, di: (not np.isnan(C[si, di])) and (not np.isnan(SMA50[si, di]))
                     and C[si, di] > SMA50[si, di]
                     and (not np.isnan(RSI[si, di])) and RSI[si, di] > 50)
    print(f"  D3) ROC(5)cross AND C>SMA50 AND RSI>50: {np.sum(sig_D3)} signals", flush=True)

    # D4: ROC(5) cross AND ADX>25 AND MACD_hist>0
    sig_D4 = make_roc_cross_signal_with_filter(5,
        lambda si, di: (not np.isnan(ADX[si, di])) and ADX[si, di] > 25
                     and (not np.isnan(MACD_hist[si, di])) and MACD_hist[si, di] > 0)
    print(f"  D4) ROC(5)cross AND ADX>25 AND MACD_hist>0: {np.sum(sig_D4)} signals", flush=True)

    configs_D = []
    d_sigs = [
        ('D1', 'ADX>25+C>SMA20', sig_D1),
        ('D2', 'ADX>25+V>1.2*SMA20V', sig_D2),
        ('D3', 'C>SMA50+RSI>50', sig_D3),
        ('D4', 'ADX>25+MACD_hist>0', sig_D4),
    ]
    for did, ddesc, dsig in d_sigs:
        for hd in [5]:
            for tn in [1]:
                configs_D.append({
                    'label': f'D_{did}_{ddesc}_H{hd}_TN{tn}',
                    'signal': dsig,
                    'hold_days': hd,
                    'top_n': tn,
                    'score': ROC[5],
                    'group': 'D',
                    'filter': did,
                })
    print(f"  Total D configs: {len(configs_D)}")

    # ================================================================
    # SWEEP E) ROC + CONCENTRATION (rank by ROC magnitude)
    # ================================================================
    print("\n" + "=" * 200)
    print("  SWEEP E) ROC CONCENTRATION: Buy top_n=1 by ROC magnitude")
    print("=" * 200)

    # Signal is any ROC(5) > 0 (not necessarily crossing), score by ROC magnitude
    sig_E = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(1, ND):
            cur = ROC[5][si, di]
            prev = ROC[5][si, di - 1]
            if np.isnan(cur) or np.isnan(prev):
                continue
            if cur > 0 and prev <= 0:
                sig_E[si, di] = True
    print(f"  E) ROC(5) crossover for ranking: {np.sum(sig_E)} signals", flush=True)

    configs_E = []
    for hd in [3, 5, 7, 10]:
        configs_E.append({
            'label': f'E_ROC5_rank_mag_H{hd}_TN1',
            'signal': sig_E,
            'hold_days': hd,
            'top_n': 1,
            'score': ROC[5],
            'group': 'E',
        })
    print(f"  Total E configs: {len(configs_E)}")

    # ================================================================
    # SWEEP F) ROC + TRAILING STOP
    # ================================================================
    print("\n" + "=" * 200)
    print("  SWEEP F) ROC + TRAILING STOP: 2*ATR trailing stop")
    print("=" * 200)

    sig_F = sig_A[5]  # baseline ROC(5) crossover
    print(f"  F) ROC(5) crossover (same as baseline): {np.sum(sig_F)} signals")

    configs_F = []
    for max_h in [10, 20, 30]:
        configs_F.append({
            'label': f'F_ROC5_trail2ATR_maxH{max_h}_TN1',
            'signal': sig_F,
            'hold_days': max_h,  # not used for fixed hold, max_hold handles it
            'top_n': 1,
            'score': ROC[5],
            'group': 'F',
            'trailing_stop': True,
            'trail_atr_mult': 2.0,
            'max_hold': max_h,
        })
    print(f"  Total F configs: {len(configs_F)}")

    # ================================================================
    # SWEEP G) ROC MULTI-PERIOD
    # ================================================================
    print("\n" + "=" * 200)
    print("  SWEEP G) ROC MULTI-PERIOD: ROC(3)>0 AND ROC(5)>0 AND ROC(10)>0")
    print("=" * 200)

    sig_G = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(1, ND):
            r3 = ROC[3][si, di]
            r5 = ROC[5][si, di]
            r10 = ROC[10][si, di]
            r3_prev = ROC[3][si, di - 1]
            r5_prev = ROC[5][si, di - 1]
            r10_prev = ROC[10][si, di - 1]
            if np.isnan(r3) or np.isnan(r5) or np.isnan(r10):
                continue
            if np.isnan(r3_prev) or np.isnan(r5_prev) or np.isnan(r10_prev):
                continue
            # All positive
            if r3 > 0 and r5 > 0 and r10 > 0:
                # At least one just crossed from negative
                crossed = (r3 > 0 and r3_prev <= 0) or (r5 > 0 and r5_prev <= 0) or (r10 > 0 and r10_prev <= 0)
                if crossed:
                    sig_G[si, di] = True
    print(f"  G) Multi-period ROC cross: {np.sum(sig_G)} signals", flush=True)

    configs_G = []
    for hd in [5]:
        for tn in [1, 3]:
            configs_G.append({
                'label': f'G_MultiROC_H{hd}_TN{tn}',
                'signal': sig_G,
                'hold_days': hd,
                'top_n': tn,
                'score': ROC[5],
                'group': 'G',
            })
    print(f"  Total G configs: {len(configs_G)}")

    # ================================================================
    # SWEEP H) ROC DIVERGENCE
    # ================================================================
    print("\n" + "=" * 200)
    print("  SWEEP H) ROC DIVERGENCE: Price 20-day low + ROC higher low")
    print("=" * 200)

    sig_H = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(25, ND):
            # Current close is 20-day low
            c_cur = C[si, di]
            if np.isnan(c_cur):
                continue
            window_c = C[si, max(0, di-19):di+1]
            if np.any(np.isnan(window_c)):
                continue
            if c_cur != np.min(window_c):
                continue

            # Find previous 20-day low (look back another 20 days)
            prev_window_c = C[si, max(0, di-39):di-19]
            if len(prev_window_c) < 10 or np.any(np.isnan(prev_window_c)):
                continue
            prev_low = np.min(prev_window_c)
            prev_low_di_local = np.argmin(prev_window_c)
            prev_low_di = max(0, di-39) + prev_low_di_local

            # Price made lower low previously (price declining)
            # Check if ROC is making HIGHER low (divergence)
            roc_cur = ROC[5][si, di]
            roc_prev_low = ROC[5][si, prev_low_di]
            if np.isnan(roc_cur) or np.isnan(roc_prev_low):
                continue

            # Divergence: price at new low but ROC higher than at previous low
            if roc_cur > roc_prev_low:
                sig_H[si, di] = True
    print(f"  H) ROC Divergence: {np.sum(sig_H)} signals", flush=True)

    configs_H = []
    for hd in [5, 10]:
        for tn in [1, 3]:
            configs_H.append({
                'label': f'H_Divergence_H{hd}_TN{tn}',
                'signal': sig_H,
                'hold_days': hd,
                'top_n': tn,
                'score': ROC[5],
                'group': 'H',
            })
    print(f"  Total H configs: {len(configs_H)}")

    # ================================================================
    # SWEEP I) ROC ACCELERATION
    # ================================================================
    print("\n" + "=" * 200)
    print("  SWEEP I) ROC ACCELERATION: 3 days increasing momentum")
    print("=" * 200)

    sig_I = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(3, ND):
            r0 = ROC[5][si, di]
            r1 = ROC[5][si, di - 1]
            r2 = ROC[5][si, di - 2]
            if np.isnan(r0) or np.isnan(r1) or np.isnan(r2):
                continue
            # 3 consecutive days of increasing momentum AND ROC > 0
            if r0 > r1 and r1 > r2 and r0 > 0:
                sig_I[si, di] = True
    print(f"  I) ROC Acceleration: {np.sum(sig_I)} signals", flush=True)

    configs_I = []
    for hd in [5]:
        for tn in [1, 3]:
            configs_I.append({
                'label': f'I_Accel_H{hd}_TN{tn}',
                'signal': sig_I,
                'hold_days': hd,
                'top_n': tn,
                'score': ROC[5],
                'group': 'I',
            })
    print(f"  Total I configs: {len(configs_I)}")

    # ================================================================
    # COLLECT ALL CONFIGS
    # ================================================================
    all_configs = configs_A + configs_B + configs_C + configs_D + configs_E + configs_F + configs_G + configs_H + configs_I
    print(f"\n{'=' * 200}")
    print(f"  TOTAL CONFIGS: {len(all_configs)}")
    print(f"{'=' * 200}")

    # ================================================================
    # RUN FULL-PERIOD BACKTEST
    # ================================================================
    print("\n[Backtest] Running full-period sweep...", flush=True)
    results = []
    for i, cfg in enumerate(all_configs):
        is_trail = cfg.get('trailing_stop', False)
        r = run_backtest(
            signal_arr=cfg['signal'],
            hold_days=cfg['hold_days'],
            top_n=cfg['top_n'],
            score_arr=cfg.get('score'),
            trailing_stop=is_trail,
            trail_atr_mult=cfg.get('trail_atr_mult', 2.0),
            max_hold=cfg.get('max_hold', cfg['hold_days']),
        )
        if r:
            r['config'] = cfg
            r['label'] = cfg['label']
            r['group'] = cfg['group']
            results.append(r)
        if (i + 1) % 20 == 0 or i == len(all_configs) - 1:
            print(f"  ... {i+1}/{len(all_configs)} done ({time.time()-t_start:.0f}s)", flush=True)

    results.sort(key=lambda x: -x['ann'])

    # ================================================================
    # FULL-PERIOD RESULTS TABLE
    # ================================================================
    print(f"\n{'=' * 200}")
    print("  FULL-PERIOD RESULTS (All configs sorted by annual return)")
    print(f"{'=' * 200}")
    print(f"  {'#':>3} | {'Grp':>3} | {'Label':<40} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'AvgHold':>7} | {'Freq/Yr':>7} | {'Final':>14}")
    print("-" * 180)
    for i, r in enumerate(results[:60]):
        print(f"  {i+1:>3} | {r['group']:>3} | {r['label']:<40} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}% | {r['avg_hold']:>6.1f}d | {r['freq']:>6.1f}/yr | {r['final_cash']:>13,.0f}")

    # ================================================================
    # BEST PER GROUP
    # ================================================================
    group_names = {
        'A': 'A) ROC PERIOD sweep',
        'B': 'B) ROC MAGNITUDE filter',
        'C': 'C) ROC + CONFIRMATION filter',
        'D': 'D) ROC + DOUBLE CONFIRMATION',
        'E': 'E) ROC CONCENTRATION',
        'F': 'F) ROC + TRAILING STOP',
        'G': 'G) ROC MULTI-PERIOD',
        'H': 'H) ROC DIVERGENCE',
        'I': 'I) ROC ACCELERATION',
    }

    print(f"\n{'=' * 200}")
    print("  BEST PER GROUP (Full Period)")
    print(f"{'=' * 200}")
    print(f"  {'Group':<45} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'AvgHold':>7} | {'Freq/Yr':>7} | Best Config")
    print("-" * 180)

    best_per_group = {}
    for r in results:
        key = r['group']
        if key not in best_per_group:
            best_per_group[key] = r

    for gkey in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
        if gkey in best_per_group:
            b = best_per_group[gkey]
            print(f"  {group_names.get(gkey, gkey):<45} | {b['ann']:>+8.1f}% | {b['wr']:>5.1f}% | {b['n']:>5} | {b['avg_pnl']:>+6.3f}% | {b['mdd']:>6.1f}% | {b['avg_hold']:>6.1f}d | {b['freq']:>6.1f}/yr | {b['label']}")

    # ================================================================
    # SWEEP A DEEP DIVE: ROC Period Sensitivity
    # ================================================================
    print(f"\n{'=' * 200}")
    print("  SWEEP A DEEP DIVE: ROC Period x Hold x TopN")
    print(f"{'=' * 200}")
    print(f"  {'Period':>6} | {'Hold':>4} | {'TN':>2} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'Freq/Yr':>7}")
    print("-" * 130)
    for p in roc_periods:
        for hd in [3, 5, 7, 10, 15, 20]:
            for tn in [1, 3]:
                sub = [r for r in results if r['config'].get('roc_period') == p
                       and r['config']['hold_days'] == hd and r['config']['top_n'] == tn]
                if sub:
                    s = sub[0]
                    print(f"  ROC({p:>2}) | H{hd:>2} | {tn:>2} | {s['ann']:>+8.1f}% | {s['wr']:>5.1f}% | {s['n']:>5} | {s['avg_pnl']:>+6.3f}% | {s['mdd']:>6.1f}% | {s['freq']:>6.1f}/yr")

    # Best period per hold/top_n
    print(f"\n  BEST ROC PERIOD per Hold x TopN:")
    print(f"  {'Hold':>4} | {'TN':>2} | {'Best Period':>11} | {'Ann':>9} | {'N':>5} | {'WR':>6}")
    print("-" * 70)
    for hd in [3, 5, 7, 10, 15, 20]:
        for tn in [1, 3]:
            best_p = None
            best_ann = -999
            for p in roc_periods:
                sub = [r for r in results if r['config'].get('roc_period') == p
                       and r['config']['hold_days'] == hd and r['config']['top_n'] == tn]
                if sub and sub[0]['ann'] > best_ann:
                    best_ann = sub[0]['ann']
                    best_p = p
            if best_p is not None:
                sub = [r for r in results if r['config'].get('roc_period') == best_p
                       and r['config']['hold_days'] == hd and r['config']['top_n'] == tn]
                if sub:
                    s = sub[0]
                    print(f"  H{hd:>2} | {tn:>2} | ROC({best_p:>2})      | {s['ann']:>+8.1f}% | {s['n']:>5} | {s['wr']:>5.1f}%")

    # ================================================================
    # SWEEP B DEEP DIVE: Magnitude Filter Comparison
    # ================================================================
    print(f"\n{'=' * 200}")
    print("  SWEEP B DEEP DIVE: ROC Magnitude Filter Comparison")
    print(f"{'=' * 200}")
    print(f"  {'Filter':<25} | {'TN':>2} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7}")
    print("-" * 100)
    for key, lab in mag_labels.items():
        for tn in [1, 3]:
            sub = [r for r in results if r['config'].get('mag_filter') == key
                   and r['config']['top_n'] == tn]
            if sub:
                s = sub[0]
                print(f"  ROC(5){lab:<18} | {tn:>2} | {s['ann']:>+8.1f}% | {s['wr']:>5.1f}% | {s['n']:>5} | {s['avg_pnl']:>+6.3f}% | {s['mdd']:>6.1f}%")

    # ================================================================
    # SWEEP C DEEP DIVE: Confirmation Filter Comparison
    # ================================================================
    print(f"\n{'=' * 200}")
    print("  SWEEP C DEEP DIVE: ROC + Confirmation Filter Comparison")
    print(f"{'=' * 200}")
    print(f"  {'Filter':<25} | {'TN':>2} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7}")
    print("-" * 100)
    for cid, cdesc, _ in c_sigs:
        for tn in [1, 3]:
            sub = [r for r in results if r['config'].get('filter') == cid
                   and r['config']['top_n'] == tn]
            if sub:
                s = sub[0]
                print(f"  {cid}: {cdesc:<21} | {tn:>2} | {s['ann']:>+8.1f}% | {s['wr']:>5.1f}% | {s['n']:>5} | {s['avg_pnl']:>+6.3f}% | {s['mdd']:>6.1f}%")

    # ================================================================
    # WALK-FORWARD (Top 20 + best per group)
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Collect top 20 overall + best per group
    wf_configs = list(results[:20])
    for gkey in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
        if gkey in best_per_group:
            r = best_per_group[gkey]
            if r['config'] not in [w['config'] for w in wf_configs]:
                wf_configs.append(r)

    print(f"\n{'=' * 220}")
    print(f"  WALK-FORWARD ({len(wf_configs)} configs, years 2020-2025)")
    print(f"{'=' * 220}")

    header = f"  {'#':>3} | {'Grp':>3} | {'Config':<40} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>7} |"
    header += f" {'Pos':>4} | {'MDD':>7} | {'WR':>6}"
    print(header)
    print("-" * 220)

    wf_rows = []
    for i, r in enumerate(wf_configs):
        cfg = r['config']
        wf_row = {'label': cfg['label'], 'group': cfg['group'],
                  'entry': 'next_open', 'windows': {}, 'mdd': {}, 'wr': {}}

        is_trail = cfg.get('trailing_stop', False)
        for yr in wf_years:
            wr_result = run_backtest(
                signal_arr=cfg['signal'],
                hold_days=cfg['hold_days'],
                top_n=cfg['top_n'],
                score_arr=cfg.get('score'),
                trailing_stop=is_trail,
                trail_atr_mult=cfg.get('trail_atr_mult', 2.0),
                max_hold=cfg.get('max_hold', cfg['hold_days']),
                wf_test_year=yr,
            )
            if wr_result:
                wf_row['windows'][yr] = wr_result['ann']
                wf_row['mdd'][yr] = wr_result['mdd']
                wf_row['wr'][yr] = wr_result['wr']
        wf_rows.append(wf_row)

        vals = [wf_row['windows'].get(yr, 0) for yr in wf_years]
        avg = np.mean(vals) if vals else 0
        pos = sum(1 for v in vals if v > 0)
        avg_mdd = np.mean(list(wf_row['mdd'].values())) if wf_row['mdd'] else 0
        avg_wr = np.mean(list(wf_row['wr'].values())) if wf_row['wr'] else 0

        row_str = f"  {i+1:>3} | {wf_row['group']:>3} | {wf_row['label']:<40} | {avg:>+7.1f}% |"
        for v in vals:
            row_str += f" {v:>+7.1f}% |"
        row_str += f" {pos}/6 | {avg_mdd:>6.1f}% | {avg_wr:>5.1f}%"
        print(row_str)

        if (i + 1) % 5 == 0:
            print(f"  ... {i+1}/{len(wf_configs)} WF done ({time.time()-t_start:.0f}s)", flush=True)

    # ================================================================
    # WF COMPARISON PER GROUP
    # ================================================================
    print(f"\n{'=' * 200}")
    print("  WALK-FORWARD COMPARISON (Best per group)")
    print(f"{'=' * 200}")
    header2 = f"  {'Group':<45} | {'WF Avg':>8} |"
    for yr in wf_years:
        header2 += f" {yr:>7} |"
    header2 += f" {'Pos':>4} | Avg MDD | Avg WR"
    print(header2)
    print("-" * 180)

    for gkey in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
        wf_match = [w for w in wf_rows if w['group'] == gkey]
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            avg = np.mean(vals) if vals else 0
            pos = sum(1 for v in vals if v > 0)
            avg_mdd = np.mean(list(wf['mdd'].values())) if wf['mdd'] else 0
            avg_wr = np.mean(list(wf['wr'].values())) if wf['wr'] else 0
            row_str = f"  {group_names.get(gkey, gkey):<45} | {avg:>+7.1f}% |"
            for v in vals:
                row_str += f" {v:>+7.1f}% |"
            row_str += f" {pos}/6 | {avg_mdd:>6.1f}% | {avg_wr:>5.1f}%"
            print(row_str)

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print(f"\n{'=' * 200}")
    print("  FINAL VERDICT: ROC DEEP OPTIMIZATION")
    print(f"{'=' * 200}")
    print()
    print("  KEY QUESTIONS:")
    print("  1. Which ROC period is truly optimal?")
    print("  2. Which filter combination improves ROC the most?")
    print("  3. Does any config beat the baseline +81.9%?")
    print()

    # Baseline reference
    baseline = [r for r in results if r['label'] == 'A_ROC(5)_H5_TN1']
    if baseline:
        print(f"  BASELINE: ROC(5)_H5_TN1 = {baseline[0]['ann']:>+8.1f}% annual")
    print()

    # Question 1: Best ROC period
    print("  1) OPTIMAL ROC PERIOD:")
    for p in roc_periods:
        sub = [r for r in results if r['config'].get('roc_period') == p]
        if sub:
            best = sub[0]
            avg_ann = np.mean([r['ann'] for r in sub])
            n_pos = sum(1 for r in sub if r['ann'] > 0)
            print(f"     ROC({p:>2}): Best={best['ann']:>+8.1f}% ({best['label']}), Avg={avg_ann:>+8.1f}%, {n_pos}/{len(sub)} positive")
    print()

    # Question 2: Best filter
    print("  2) BEST FILTER COMBINATION:")
    non_A = [r for r in results if r['group'] != 'A']
    for r in non_A[:10]:
        print(f"     {r['label']:<45} | {r['ann']:>+8.1f}% | WR={r['wr']:>5.1f}% | N={r['n']:>5} | MDD={r['mdd']:>6.1f}%")
    print()

    # Question 3: Beats baseline?
    print("  3) CONFIGS BEATING +81.9%:")
    baseline_val = baseline[0]['ann'] if baseline else 81.9
    beaters = [r for r in results if r['ann'] > baseline_val]
    if beaters:
        for r in beaters:
            wf_match = [w for w in wf_rows if w['label'] == r['label']]
            wf_pos = 0
            wf_avg = 0
            if wf_match:
                wf = wf_match[0]
                vals = [wf['windows'].get(yr, 0) for yr in wf_years]
                wf_pos = sum(1 for v in vals if v > 0)
                wf_avg = np.mean(vals)
            print(f"     {r['label']:<45} | {r['ann']:>+8.1f}% | WR={r['wr']:>5.1f}% | MDD={r['mdd']:>6.1f}% | WF_Avg={wf_avg:>+7.1f}% | WF_Pos={wf_pos}/6")
    else:
        print(f"     None found -- baseline ROC(5)_H5_TN1 at {baseline_val:+.1f}% is hard to beat!")
    print()

    # Absolute champion
    if results:
        champ = results[0]
        print(f"  {'='*80}")
        print(f"  CHAMPION: {champ['label']}")
        print(f"    Annual: {champ['ann']:>+8.1f}%  |  WR: {champ['wr']:>5.1f}%  |  N: {champ['n']:>4}  |  MDD: {champ['mdd']:>6.1f}%")
        print(f"    Avg PnL/trade: {champ['avg_pnl']:>+6.3f}%  |  Avg Hold: {champ['avg_hold']:>5.1f}d  |  Freq: {champ['freq']:>5.1f}/yr")
        champ_wf = [w for w in wf_rows if w['label'] == champ['label']]
        if champ_wf:
            cw = champ_wf[0]
            vals = [cw['windows'].get(yr, 0) for yr in wf_years]
            avg_wf = np.mean(vals)
            wf_pos = sum(1 for v in vals if v > 0)
            print(f"    WF: {[f'{v:>+7.1f}%' for v in vals]}  |  {wf_pos}/6 positive  |  WF Avg: {avg_wf:>+7.1f}%")
        print(f"  {'='*80}")

    # Top 10 summary with WF
    print(f"\n  TOP 10 CONFIGS (with Walk-Forward):")
    print(f"  {'#':>3} | {'Grp':>3} | {'Label':<40} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'MDD':>7} | {'WF_Avg':>8} | {'WF_Pos':>6}")
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
        print(f"  {i+1:>3} | {r['group']:>3} | {r['label']:<40} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['mdd']:>6.1f}% | {wf_avg:>+7.1f}% | {wf_pos}/6")

    print(f"\n  Total runtime: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
