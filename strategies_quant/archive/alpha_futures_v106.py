"""
Alpha Futures V106 -- TA-Lib Candlestick Pattern Strategy
==========================================================
Test ALL 61 TA-Lib candlestick patterns on 68 commodity futures.

TA-Lib candlestick functions return:
  +100 = bullish pattern detected
  -100 = bearish pattern detected
   0   = no pattern

Signal at close of day di -> entry at O[si, di+1] (NEXT-OPEN execution).
Exit at C[si, di+hold] (close price hold days later).

TEST PLAN:
A) Individual patterns: bullish(100) -> hold 3/5/10 days
B) Pattern count: 2+/3+/4+ bullish patterns same day -> hold 3/5/10
C) Top patterns + volume confirmation (V > 1.5*SMA20(V))
D) Top patterns + trend filter (C > SMA50(C))
E) Top patterns + OI confirmation (OI increasing)
F) Bearish reversal contrarian (bearish pattern + RSI<30)
G) Morning Star / Morning Doji Star specifically
H) Engulfing + Hammer combined

top_n=3 for all tests. Long-only. COMM=0.0003.
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

# All 61 candlestick pattern function names from TA-Lib
ALL_CDL_NAMES = sorted([f for f in dir(talib) if f.startswith('CDL')])

# Categorize: some patterns are primarily bearish (skip for long-only bullish)
BEARISH_ONLY = {'CDL2CROWS', 'CDL3BLACKCROWS', 'CDLADVANCEBLOCK',
                'CDLCONCEALBABYSWALL', 'CDLDARKCLOUDCOVER', 'CDLEVENINGDOJISTAR',
                'CDLEVENINGSTAR', 'CDLHANGINGMAN', 'CDLIDENTICAL3CROWS',
                'CDLINNECK', 'CDLONNECK', 'CDLSHOOTINGSTAR',
                'CDLSTALLEDPATTERN', 'CDLTHRUSTING', 'CDLUPSIDEGAP2CROWS'}


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def main():
    print("=" * 180)
    print("Alpha Futures V106 -- TA-Lib Candlestick Pattern Strategy")
    print("=" * 180)
    print(f"\n  Testing ALL {len(ALL_CDL_NAMES)} candlestick patterns on 68 commodity futures")
    print("  ALL signals computed at close di, entry at O[si, di+1] (NEXT DAY OPEN)")

    # -- Load data -------------------------------------------------
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ================================================================
    # BATCH COMPUTE ALL CANDLESTICK PATTERNS
    # ================================================================
    print(f"\n[Candlestick] Computing ALL {len(ALL_CDL_NAMES)} patterns for {NS} commodities...", flush=True)
    t0 = time.time()

    # cdl_results[pattern_name] = array (NS, ND), values: +100, 0, -100
    cdl_results = {}

    for pi, pname in enumerate(ALL_CDL_NAMES):
        cdl_func = getattr(talib, pname)
        arr = np.zeros((NS, ND), dtype=np.int32)

        for si in range(NS):
            valid = ~np.isnan(C[si])
            if np.sum(valid) < 20:
                continue
            o_valid = np.where(valid, O[si], np.nan)
            h_valid = np.where(valid, H[si], np.nan)
            l_valid = np.where(valid, L[si], np.nan)
            c_valid = np.where(valid, C[si], np.nan)

            # TA-Lib needs float64 arrays and does NOT handle NaN well
            # Build clean arrays for valid regions
            idx_valid = np.where(valid)[0]
            if len(idx_valid) < 20:
                continue
            o_clean = np.asarray(o_valid[idx_valid], dtype=np.float64)
            h_clean = np.asarray(h_valid[idx_valid], dtype=np.float64)
            l_clean = np.asarray(l_valid[idx_valid], dtype=np.float64)
            c_clean = np.asarray(c_valid[idx_valid], dtype=np.float64)

            # Replace any remaining NaN with 0 for safety
            o_clean = np.nan_to_num(o_clean, nan=0.0)
            h_clean = np.nan_to_num(h_clean, nan=0.0)
            l_clean = np.nan_to_num(l_clean, nan=0.0)
            c_clean = np.nan_to_num(c_clean, nan=0.0)

            try:
                result = cdl_func(o_clean, h_clean, l_clean, c_clean)
                arr[si, idx_valid] = result.astype(np.int32)
            except Exception:
                pass

        cdl_results[pname] = arr
        if (pi + 1) % 10 == 0 or pi == len(ALL_CDL_NAMES) - 1:
            print(f"  ... {pi+1}/{len(ALL_CDL_NAMES)} patterns done ({time.time()-t0:.1f}s)", flush=True)

    print(f"  All candlestick patterns computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # COMPUTE AUXILIARY INDICATORS
    # ================================================================
    print("\n[Aux] Computing SMA, RSI, Volume MA...", flush=True)
    t0 = time.time()

    # 50-day SMA of close
    sma50 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(50, ND):
            cw = C[si, di-50:di]
            valid = cw[~np.isnan(cw)]
            if len(valid) >= 25:
                sma50[si, di] = np.mean(valid)

    # 20-day volume MA
    vol_ma20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            vw = V[si, di-20:di]
            valid = vw[~np.isnan(vw)]
            if len(valid) >= 10:
                vol_ma20[si, di] = np.mean(valid)

    # RSI-14 using talib
    rsi14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        valid = ~np.isnan(C[si])
        idx_valid = np.where(valid)[0]
        if len(idx_valid) < 30:
            continue
        c_clean = np.nan_to_num(C[si, idx_valid], nan=0.0).astype(np.float64)
        try:
            r = talib.RSI(c_clean, timeperiod=14)
            rsi14[si, idx_valid] = r
        except Exception:
            pass

    print(f"  Aux indicators computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # SUMMARY: Pattern frequency
    # ================================================================
    print(f"\n[Summary] Pattern frequencies (bullish=+100, bearish=-100):")
    print(f"  {'Pattern':<30} | {'Bullish':>8} | {'Bearish':>8} | {'Total':>8} | {'Bullish/Day':>12}")
    print("  " + "-" * 80)
    pattern_stats = []
    for pname in ALL_CDL_NAMES:
        arr = cdl_results[pname]
        n_bull = int(np.sum(arr == 100))
        n_bear = int(np.sum(arr == -100))
        n_total = n_bull + n_bear
        bull_per_day = n_bull / ND if ND > 0 else 0
        pattern_stats.append((pname, n_bull, n_bear, n_total, bull_per_day))
    pattern_stats.sort(key=lambda x: -x[1])  # sort by bullish count
    for pname, n_bull, n_bear, n_total, bpd in pattern_stats:
        bear_mark = " *" if pname in BEARISH_ONLY else ""
        print(f"  {pname + bear_mark:<30} | {n_bull:>8} | {n_bear:>8} | {n_total:>8} | {bpd:>11.2f}")

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(signal_func, hold_days, top_n=3, comm=COMM, wf_test_year=None,
                     score_func=None, label=""):
        """
        Generic backtest engine.
        signal_func(si, di) -> bool: whether to trigger buy signal
        score_func(si, di) -> float: signal strength for ranking (higher = better)
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
        positions = []  # list of {si, entry_price, entry_di, lots, sym, hold_days}
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

            # If we have enough positions, skip
            if len(positions) >= top_n:
                continue

            # -- Generate signals at day di --------------------------------
            entry_di = di + 1
            if entry_di >= end_di:
                continue

            candidates = []
            for si in range(NS):
                if any(p['si'] == si for p in positions):
                    continue
                if not signal_func(si, di):
                    continue
                ep = O[si, entry_di]
                if np.isnan(ep) or ep <= 0:
                    continue
                sc = score_func(si, di) if score_func else 0.0
                candidates.append((sc, si, ep))

            if not candidates:
                continue

            # Sort by score descending
            candidates.sort(key=lambda x: -x[0])

            # Open positions
            n_slots = top_n - len(positions)
            for sc, si, price in candidates[:max(0, n_slots)]:
                sym = syms[si]
                mult = MULT.get(sym, DEF_MULT)
                notional = price * mult
                lots = max(1, int(cash / (notional * (1 + comm))))
                if lots <= 0:
                    continue
                cost_in = notional * lots * (1 + comm)
                if cost_in > cash:
                    lots = max(1, int(cash * 0.85 / (notional * (1 + comm))))
                    cost_in = notional * lots * (1 + comm) if lots > 0 else 0
                if lots <= 0 or cost_in <= 0 or cost_in > cash:
                    continue

                cash -= cost_in
                positions.append({
                    'si': si, 'entry_price': price, 'entry_di': entry_di,
                    'lots': lots, 'sym': sym, 'hold_days': hold_days,
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
        n_days_test = (test_end_di - test_start_di) if wf_test_year is not None else (end_di - start_di)
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
    # TEST A: INDIVIDUAL PATTERNS
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  TEST A: INDIVIDUAL BULLISH CANDLESTICK PATTERNS (hold 3/5/10 days)")
    print(f"{'=' * 180}")

    individual_results = []
    for pname in ALL_CDL_NAMES:
        arr = cdl_results[pname]
        for hold in [3, 5, 10]:
            # Test bullish signals (result == 100)
            n_bull = int(np.sum(arr == 100))
            if n_bull < 50:
                # Skip patterns with too few signals
                continue

            def make_signal(cd_arr, cd_val):
                def signal_func(si, di):
                    return cd_arr[si, di] == cd_val
                return signal_func

            r = run_backtest(
                signal_func=make_signal(arr, 100),
                hold_days=hold,
                top_n=3,
                label=f"{pname}_bull_H{hold}",
            )
            if r:
                r['pattern'] = pname
                r['direction'] = 'bull'
                r['hold'] = hold
                r['label'] = f"{pname}_bull_H{hold}"
                individual_results.append(r)

            # Also test bearish signals (result == -100) for non-bearish-only patterns
            # and for bearish patterns as contrarian
            n_bear = int(np.sum(arr == -100))
            if n_bear < 50:
                continue

            r2 = run_backtest(
                signal_func=make_signal(arr, -100),
                hold_days=hold,
                top_n=3,
                label=f"{pname}_bear_H{hold}",
            )
            if r2:
                r2['pattern'] = pname
                r2['direction'] = 'bear'
                r2['hold'] = hold
                r2['label'] = f"{pname}_bear_H{hold}"
                individual_results.append(r2)

    # Sort by annual return
    individual_results.sort(key=lambda x: -x['ann'])

    # Print top 20
    print(f"\n  {'#':>3} | {'Pattern':<30} | {'Dir':>4} | {'Hold':>4} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'AvgHold':>7} | {'Freq/Yr':>7}")
    print("  " + "-" * 140)
    for i, r in enumerate(individual_results[:40]):
        print(f"  {i+1:>3} | {r['pattern']:<30} | {r['direction']:>4} | {r['hold']:>4}d | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}% | {r['avg_hold']:>6.1f}d | {r['freq']:>6.1f}/yr")

    # Top 10 individual patterns
    print(f"\n  TOP 10 INDIVIDUAL PATTERNS (by annual return):")
    print(f"  {'#':>3} | {'Pattern':<30} | {'Dir':>4} | {'Hold':>4} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7}")
    print("  " + "-" * 110)
    for i, r in enumerate(individual_results[:10]):
        print(f"  {i+1:>3} | {r['pattern']:<30} | {r['direction']:>4} | {r['hold']:>4}d | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}%")

    # Identify top patterns for further tests
    # Get unique top patterns (by best annual return)
    top_patterns = []
    seen = set()
    for r in individual_results:
        p = r['pattern']
        d = r['direction']
        if (p, d) not in seen:
            top_patterns.append((p, d, r))
            seen.add((p, d))
        if len(top_patterns) >= 10:
            break
    print(f"\n  Top 10 unique patterns for further testing:")
    for i, (p, d, r) in enumerate(top_patterns):
        print(f"    {i+1}. {p} ({d}): Ann={r['ann']:>+8.1f}%, WR={r['wr']:>5.1f}%, N={r['n']}")

    # ================================================================
    # TEST B: PATTERN COUNT (multiple patterns firing same day)
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  TEST B: PATTERN COUNT (multiple bullish patterns same day)")
    print(f"{'=' * 180}")

    # Pre-compute bullish pattern count per (si, di)
    bullish_count = np.zeros((NS, ND), dtype=np.int32)
    bullish_patterns_list = [p for p in ALL_CDL_NAMES if p not in BEARISH_ONLY]
    for pname in bullish_patterns_list:
        bullish_count += (cdl_results[pname] == 100).astype(np.int32)

    # Also pre-compute bearish count
    bearish_count = np.zeros((NS, ND), dtype=np.int32)
    for pname in ALL_CDL_NAMES:
        bearish_count += (cdl_results[pname] == -100).astype(np.int32)

    print(f"  Max bullish patterns on a single day: {np.max(bullish_count)}")
    print(f"  Max bearish patterns on a single day: {np.max(bearish_count)}")

    count_results = []
    for min_count in [2, 3, 4, 5]:
        for hold in [3, 5, 10]:
            bc = bullish_count.copy()
            mc = min_count

            def make_count_signal(bc_arr, min_c):
                def signal_func(si, di):
                    return bc_arr[si, di] >= min_c
                return signal_func

            def count_score(bc_arr):
                def score_func(si, di):
                    return float(bc_arr[si, di])
                return score_func

            n_sig = int(np.sum(bc >= mc))
            r = run_backtest(
                signal_func=make_count_signal(bc, mc),
                hold_days=hold,
                top_n=3,
                score_func=count_score(bc),
                label=f"BullCount>={mc}_H{hold}",
            )
            if r:
                r['label'] = f"BullCount>={mc}_H{hold}"
                r['test'] = 'B'
                r['min_count'] = mc
                r['hold'] = hold
                count_results.append(r)
                print(f"  BullCount>={mc}_H{hold}: Ann={r['ann']:>+8.1f}%, WR={r['wr']:>5.1f}%, N={r['n']}, MDD={r['mdd']:>6.1f}%  (signals={n_sig})")

    # Also test: bearish count as contrarian
    print(f"\n  Bearish Count Contrarian (bearish patterns + RSI<30):")
    bear_count_results = []
    for min_count in [2, 3]:
        for hold in [3, 5, 10]:
            def make_bear_contrarian(bc_arr, rsi_arr, min_c):
                def signal_func(si, di):
                    if bc_arr[si, di] < min_c:
                        return False
                    r = rsi_arr[si, di]
                    if np.isnan(r):
                        return False
                    return r < 30
                return signal_func

            def bear_score(bc_arr):
                def score_func(si, di):
                    return float(bc_arr[si, di])
                return score_func

            n_sig = int(np.sum((bearish_count >= min_count) & (~np.isnan(rsi14)) & (rsi14 < 30)))
            r = run_backtest(
                signal_func=make_bear_contrarian(bearish_count, rsi14, min_count),
                hold_days=hold,
                top_n=3,
                score_func=bear_score(bearish_count),
                label=f"BearContr>={min_count}_H{hold}",
            )
            if r:
                r['label'] = f"BearContr>={min_count}_H{hold}"
                r['test'] = 'B_bear'
                bear_count_results.append(r)
                print(f"  BearContr>={min_count}_H{hold}: Ann={r['ann']:>+8.1f}%, WR={r['wr']:>5.1f}%, N={r['n']}, MDD={r['mdd']:>6.1f}%  (signals={n_sig})")

    # ================================================================
    # TEST C: TOP PATTERNS + VOLUME CONFIRMATION
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  TEST C: TOP PATTERNS + VOLUME CONFIRMATION (V > 1.5 * SMA20(V))")
    print(f"{'=' * 180}")

    vol_results = []
    for pname, direction, best_r in top_patterns[:8]:
        arr = cdl_results[pname]
        cd_val = 100 if direction == 'bull' else -100
        for hold in [3, 5, 10]:
            def make_vol_signal(cd_arr, cd_val, vma_arr, v_arr):
                def signal_func(si, di):
                    if cd_arr[si, di] != cd_val:
                        return False
                    v = v_arr[si, di]
                    vma = vma_arr[si, di]
                    if np.isnan(v) or np.isnan(vma) or vma <= 0:
                        return False
                    return v > 1.5 * vma
                return signal_func

            r = run_backtest(
                signal_func=make_vol_signal(arr, cd_val, vol_ma20, V),
                hold_days=hold,
                top_n=3,
                label=f"{pname}_{direction}_Vol_H{hold}",
            )
            if r:
                r['label'] = f"{pname}_{direction}_Vol_H{hold}"
                r['test'] = 'C'
                r['pattern'] = pname
                r['direction'] = direction
                r['hold'] = hold
                vol_results.append(r)
                print(f"  {pname}_{direction}_Vol_H{hold}: Ann={r['ann']:>+8.1f}%, WR={r['wr']:>5.1f}%, N={r['n']}, MDD={r['mdd']:>6.1f}%")

    # ================================================================
    # TEST D: TOP PATTERNS + TREND FILTER (C > SMA50)
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  TEST D: TOP PATTERNS + TREND FILTER (C > SMA50(C))")
    print(f"{'=' * 180}")

    trend_results = []
    for pname, direction, best_r in top_patterns[:8]:
        arr = cdl_results[pname]
        cd_val = 100 if direction == 'bull' else -100
        for hold in [3, 5, 10]:
            def make_trend_signal(cd_arr, cd_val, sma_arr, c_arr):
                def signal_func(si, di):
                    if cd_arr[si, di] != cd_val:
                        return False
                    c = c_arr[si, di]
                    s = sma_arr[si, di]
                    if np.isnan(c) or np.isnan(s):
                        return False
                    return c > s
                return signal_func

            r = run_backtest(
                signal_func=make_trend_signal(arr, cd_val, sma50, C),
                hold_days=hold,
                top_n=3,
                label=f"{pname}_{direction}_Trend_H{hold}",
            )
            if r:
                r['label'] = f"{pname}_{direction}_Trend_H{hold}"
                r['test'] = 'D'
                r['pattern'] = pname
                r['direction'] = direction
                r['hold'] = hold
                trend_results.append(r)
                print(f"  {pname}_{direction}_Trend_H{hold}: Ann={r['ann']:>+8.1f}%, WR={r['wr']:>5.1f}%, N={r['n']}, MDD={r['mdd']:>6.1f}%")

    # ================================================================
    # TEST E: TOP PATTERNS + OI CONFIRMATION
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  TEST E: TOP PATTERNS + OI CONFIRMATION (OI[di] > OI[di-5])")
    print(f"{'=' * 180}")

    oi_results = []
    for pname, direction, best_r in top_patterns[:8]:
        arr = cdl_results[pname]
        cd_val = 100 if direction == 'bull' else -100
        for hold in [3, 5, 10]:
            def make_oi_signal(cd_arr, cd_val, oi_arr):
                def signal_func(si, di):
                    if cd_arr[si, di] != cd_val:
                        return False
                    if di < 5:
                        return False
                    oi_now = oi_arr[si, di]
                    oi_ago = oi_arr[si, di - 5]
                    if np.isnan(oi_now) or np.isnan(oi_ago):
                        return False
                    return oi_now > oi_ago
                return signal_func

            r = run_backtest(
                signal_func=make_oi_signal(arr, cd_val, OI),
                hold_days=hold,
                top_n=3,
                label=f"{pname}_{direction}_OI_H{hold}",
            )
            if r:
                r['label'] = f"{pname}_{direction}_OI_H{hold}"
                r['test'] = 'E'
                r['pattern'] = pname
                r['direction'] = direction
                r['hold'] = hold
                oi_results.append(r)
                print(f"  {pname}_{direction}_OI_H{hold}: Ann={r['ann']:>+8.1f}%, WR={r['wr']:>5.1f}%, N={r['n']}, MDD={r['mdd']:>6.1f}%")

    # ================================================================
    # TEST F: BEARISH REVERSAL CONTRARIAN (bearish pattern + RSI < 30)
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  TEST F: BEARISH REVERSAL CONTRARIAN (bearish pattern + RSI<30 -> buy)")
    print(f"{'=' * 180}")

    contrarian_results = []
    for pname in ALL_CDL_NAMES:
        arr = cdl_results[pname]
        n_bear = int(np.sum(arr == -100))
        if n_bear < 30:
            continue
        for hold in [3, 5, 10]:
            def make_contrarian(cd_arr, rsi_arr):
                def signal_func(si, di):
                    if cd_arr[si, di] != -100:
                        return False
                    r = rsi_arr[si, di]
                    if np.isnan(r):
                        return False
                    return r < 30
                return signal_func

            n_sig = int(np.sum((arr == -100) & (~np.isnan(rsi14)) & (rsi14 < 30)))
            if n_sig < 10:
                continue

            r = run_backtest(
                signal_func=make_contrarian(arr, rsi14),
                hold_days=hold,
                top_n=3,
                label=f"{pname}_Contrarian_H{hold}",
            )
            if r:
                r['label'] = f"{pname}_Contrarian_H{hold}"
                r['test'] = 'F'
                r['pattern'] = pname
                r['direction'] = 'contrarian'
                r['hold'] = hold
                contrarian_results.append(r)
                if r['ann'] > 10:
                    print(f"  {pname}_Contrarian_H{hold}: Ann={r['ann']:>+8.1f}%, WR={r['wr']:>5.1f}%, N={r['n']}, MDD={r['mdd']:>6.1f}%  (signals={n_sig})")

    contrarian_results.sort(key=lambda x: -x['ann'])
    print(f"\n  Top 10 Contrarian results:")
    print(f"  {'#':>3} | {'Pattern':<30} | {'Hold':>4} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7}")
    print("  " + "-" * 110)
    for i, r in enumerate(contrarian_results[:10]):
        print(f"  {i+1:>3} | {r['pattern']:<30} | {r['hold']:>4}d | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}%")

    # ================================================================
    # TEST G: MORNING STAR / MORNING DOJI STAR
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  TEST G: MORNING STAR / MORNING DOJI STAR (reversal patterns)")
    print(f"{'=' * 180}")

    g_results = []
    # G1: CDLMORNINGSTAR bullish
    for hold in [3, 5, 10]:
        arr_ms = cdl_results['CDLMORNINGSTAR']
        def make_ms_signal(cd_arr):
            def signal_func(si, di):
                return cd_arr[si, di] == 100
            return signal_func

        r = run_backtest(
            signal_func=make_ms_signal(arr_ms),
            hold_days=hold,
            top_n=3,
            label=f"MorningStar_H{hold}",
        )
        if r:
            r['label'] = f"MorningStar_bull_H{hold}"
            r['test'] = 'G'
            r['pattern'] = 'CDLMORNINGSTAR'
            r['direction'] = 'bull'
            r['hold'] = hold
            g_results.append(r)
            print(f"  MorningStar_bull_H{hold}: Ann={r['ann']:>+8.1f}%, WR={r['wr']:>5.1f}%, N={r['n']}, MDD={r['mdd']:>6.1f}%")

    # G2: CDLMORNINGDOJISTAR bullish
    for hold in [3, 5, 10]:
        arr_mds = cdl_results['CDLMORNINGDOJISTAR']
        def make_mds_signal(cd_arr):
            def signal_func(si, di):
                return cd_arr[si, di] == 100
            return signal_func

        r = run_backtest(
            signal_func=make_mds_signal(arr_mds),
            hold_days=hold,
            top_n=3,
            label=f"MorningDojiStar_H{hold}",
        )
        if r:
            r['label'] = f"MorningDojiStar_bull_H{hold}"
            r['test'] = 'G'
            r['pattern'] = 'CDLMORNINGDOJISTAR'
            r['direction'] = 'bull'
            r['hold'] = hold
            g_results.append(r)
            print(f"  MorningDojiStar_bull_H{hold}: Ann={r['ann']:>+8.1f}%, WR={r['wr']:>5.1f}%, N={r['n']}, MDD={r['mdd']:>6.1f}%")

    # G3: Morning Star + Morning Doji Star combined
    for hold in [3, 5, 10]:
        def make_ms_combined_signal(cd_arr1, cd_arr2):
            def signal_func(si, di):
                return cd_arr1[si, di] == 100 or cd_arr2[si, di] == 100
            return signal_func

        r = run_backtest(
            signal_func=make_ms_combined_signal(arr_ms, arr_mds),
            hold_days=hold,
            top_n=3,
            label=f"MorningStar_Combined_H{hold}",
        )
        if r:
            r['label'] = f"MorningStar_Combined_H{hold}"
            r['test'] = 'G'
            r['pattern'] = 'CDLMORNINGSTAR_COMBINED'
            r['direction'] = 'bull'
            r['hold'] = hold
            g_results.append(r)
            print(f"  MorningStar_Combined_H{hold}: Ann={r['ann']:>+8.1f}%, WR={r['wr']:>5.1f}%, N={r['n']}, MDD={r['mdd']:>6.1f}%")

    # G4: Morning Star + Volume confirmation
    for hold in [3, 5, 10]:
        def make_ms_vol_signal(cd_arr, vma_arr, v_arr):
            def signal_func(si, di):
                if cd_arr[si, di] != 100:
                    return False
                v = v_arr[si, di]
                vma = vma_arr[si, di]
                if np.isnan(v) or np.isnan(vma) or vma <= 0:
                    return False
                return v > 1.5 * vma
            return signal_func

        r = run_backtest(
            signal_func=make_ms_vol_signal(arr_ms, vol_ma20, V),
            hold_days=hold,
            top_n=3,
            label=f"MorningStar_Vol_H{hold}",
        )
        if r:
            r['label'] = f"MorningStar_Vol_H{hold}"
            r['test'] = 'G'
            r['pattern'] = 'CDLMORNINGSTAR_VOL'
            r['direction'] = 'bull'
            r['hold'] = hold
            g_results.append(r)
            print(f"  MorningStar_Vol_H{hold}: Ann={r['ann']:>+8.1f}%, WR={r['wr']:>5.1f}%, N={r['n']}, MDD={r['mdd']:>6.1f}%")

    # ================================================================
    # TEST H: ENGULFING + HAMMER (most reliable patterns)
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  TEST H: ENGULFING + HAMMER (most reliable patterns)")
    print(f"{'=' * 180}")

    h_results = []
    arr_eng = cdl_results['CDLENGULFING']
    arr_ham = cdl_results['CDLHAMMER']

    # H1: Engulfing bullish
    for hold in [3, 5, 10]:
        def make_eng_signal(cd_arr):
            def signal_func(si, di):
                return cd_arr[si, di] == 100
            return signal_func

        r = run_backtest(
            signal_func=make_eng_signal(arr_eng),
            hold_days=hold,
            top_n=3,
            label=f"Engulfing_bull_H{hold}",
        )
        if r:
            r['label'] = f"Engulfing_bull_H{hold}"
            r['test'] = 'H'
            r['pattern'] = 'CDLENGULFING'
            r['direction'] = 'bull'
            r['hold'] = hold
            h_results.append(r)
            print(f"  Engulfing_bull_H{hold}: Ann={r['ann']:>+8.1f}%, WR={r['wr']:>5.1f}%, N={r['n']}, MDD={r['mdd']:>6.1f}%")

    # H2: Hammer
    for hold in [3, 5, 10]:
        def make_ham_signal(cd_arr):
            def signal_func(si, di):
                return cd_arr[si, di] == 100
            return signal_func

        r = run_backtest(
            signal_func=make_ham_signal(arr_ham),
            hold_days=hold,
            top_n=3,
            label=f"Hammer_bull_H{hold}",
        )
        if r:
            r['label'] = f"Hammer_bull_H{hold}"
            r['test'] = 'H'
            r['pattern'] = 'CDLHAMMER'
            r['direction'] = 'bull'
            r['hold'] = hold
            h_results.append(r)
            print(f"  Hammer_bull_H{hold}: Ann={r['ann']:>+8.1f}%, WR={r['wr']:>5.1f}%, N={r['n']}, MDD={r['mdd']:>6.1f}%")

    # H3: Engulfing OR Hammer
    for hold in [3, 5, 10]:
        def make_eng_ham_signal(cd_arr1, cd_arr2):
            def signal_func(si, di):
                return cd_arr1[si, di] == 100 or cd_arr2[si, di] == 100
            return signal_func

        r = run_backtest(
            signal_func=make_eng_ham_signal(arr_eng, arr_ham),
            hold_days=hold,
            top_n=3,
            label=f"EngOrHam_H{hold}",
        )
        if r:
            r['label'] = f"EngOrHam_H{hold}"
            r['test'] = 'H'
            r['pattern'] = 'CDLENGULFING_OR_HAMMER'
            r['direction'] = 'bull'
            r['hold'] = hold
            h_results.append(r)
            print(f"  Engulfing_OR_Hammer_H{hold}: Ann={r['ann']:>+8.1f}%, WR={r['wr']:>5.1f}%, N={r['n']}, MDD={r['mdd']:>6.1f}%")

    # H4: Engulfing + Hammer + Volume
    for hold in [3, 5, 10]:
        def make_eng_ham_vol_signal(cd_arr1, cd_arr2, vma_arr, v_arr):
            def signal_func(si, di):
                if cd_arr1[si, di] != 100 and cd_arr2[si, di] != 100:
                    return False
                v = v_arr[si, di]
                vma = vma_arr[si, di]
                if np.isnan(v) or np.isnan(vma) or vma <= 0:
                    return False
                return v > 1.5 * vma
            return signal_func

        r = run_backtest(
            signal_func=make_eng_ham_vol_signal(arr_eng, arr_ham, vol_ma20, V),
            hold_days=hold,
            top_n=3,
            label=f"EngHam_Vol_H{hold}",
        )
        if r:
            r['label'] = f"EngHam_Vol_H{hold}"
            r['test'] = 'H'
            r['pattern'] = 'CDLENGULFING_HAMMER_VOL'
            r['direction'] = 'bull'
            r['hold'] = hold
            h_results.append(r)
            print(f"  EngHam_Vol_H{hold}: Ann={r['ann']:>+8.1f}%, WR={r['wr']:>5.1f}%, N={r['n']}, MDD={r['mdd']:>6.1f}%")

    # H5: Engulfing + Hammer + Trend
    for hold in [3, 5, 10]:
        def make_eng_ham_trend_signal(cd_arr1, cd_arr2, sma_arr, c_arr):
            def signal_func(si, di):
                if cd_arr1[si, di] != 100 and cd_arr2[si, di] != 100:
                    return False
                c = c_arr[si, di]
                s = sma_arr[si, di]
                if np.isnan(c) or np.isnan(s):
                    return False
                return c > s
            return signal_func

        r = run_backtest(
            signal_func=make_eng_ham_trend_signal(arr_eng, arr_ham, sma50, C),
            hold_days=hold,
            top_n=3,
            label=f"EngHam_Trend_H{hold}",
        )
        if r:
            r['label'] = f"EngHam_Trend_H{hold}"
            r['test'] = 'H'
            r['pattern'] = 'CDLENGULFING_HAMMER_TREND'
            r['direction'] = 'bull'
            r['hold'] = hold
            h_results.append(r)
            print(f"  EngHam_Trend_H{hold}: Ann={r['ann']:>+8.1f}%, WR={r['wr']:>5.1f}%, N={r['n']}, MDD={r['mdd']:>6.1f}%")

    # ================================================================
    # COMBINE ALL RESULTS
    # ================================================================
    all_results = []
    for r in individual_results:
        r['test'] = 'A'
        all_results.append(r)
    for r in count_results:
        all_results.append(r)
    for r in bear_count_results:
        all_results.append(r)
    for r in vol_results:
        all_results.append(r)
    for r in trend_results:
        all_results.append(r)
    for r in oi_results:
        all_results.append(r)
    for r in contrarian_results:
        all_results.append(r)
    for r in g_results:
        all_results.append(r)
    for r in h_results:
        all_results.append(r)

    all_results.sort(key=lambda x: -x['ann'])

    # ================================================================
    # MASTER RESULTS TABLE
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  MASTER RESULTS TABLE (ALL tests combined, top 50 by annual return)")
    print(f"{'=' * 180}")
    print(f"  {'#':>3} | {'Label':<35} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'AvgHold':>7} | {'Freq/Yr':>7}")
    print("  " + "-" * 130)
    for i, r in enumerate(all_results[:50]):
        print(f"  {i+1:>3} | {r.get('label', '?'):<35} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}% | {r['avg_hold']:>6.1f}d | {r['freq']:>6.1f}/yr")

    # ================================================================
    # RESULTS BEATING +49.6%
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  CONFIGS BEATING +49.6% (current best practical)")
    print(f"{'=' * 180}")
    beating = [r for r in all_results if r['ann'] > 49.6]
    if beating:
        print(f"  {'#':>3} | {'Label':<35} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'AvgHold':>7}")
        print("  " + "-" * 120)
        for i, r in enumerate(beating[:30]):
            print(f"  {i+1:>3} | {r.get('label', '?'):<35} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}% | {r['avg_hold']:>6.1f}d")
    else:
        print("  No configs beat +49.6%")

    # ================================================================
    # WALK-FORWARD (Top 15 configs)
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Select top 15 configs for WF
    wf_candidates = list(all_results[:15])
    # Ensure at least one from each test
    for test_key in ['B', 'C', 'D', 'E', 'F', 'G', 'H']:
        sub = [r for r in all_results if r.get('test') == test_key and r['ann'] > 0]
        if sub and sub[0] not in wf_candidates:
            wf_candidates.append(sub[0])

    print(f"\n{'=' * 200}")
    print(f"  WALK-FORWARD ({len(wf_candidates)} configs)")
    print(f"{'=' * 200}")

    header = f"  {'#':>3} | {'Config':<35} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>7} |"
    header += f" {'Pos':>4} | {'MDD':>7} | {'WR':>6}"
    print(header)
    print("-" * 200)

    wf_rows = []
    for i, r in enumerate(wf_candidates):
        label = r.get('label', '?')

        # Reconstruct signal function for this config
        # We need to re-derive it from the label
        sig_func, sc_func = reconstruct_signal(r, cdl_results, bullish_count, bearish_count,
                                                vol_ma20, sma50, V, C, OI, rsi14)
        if sig_func is None:
            continue

        hold_days = r.get('hold', r.get('config', {}).get('hold_days', 5))

        wf_row = {'label': label, 'test': r.get('test', '?'), 'windows': {}, 'mdd': {}, 'wr': {}}
        for yr in wf_years:
            wr = run_backtest(
                signal_func=sig_func,
                hold_days=hold_days,
                top_n=3,
                wf_test_year=yr,
                score_func=sc_func,
            )
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

        row_str = f"  {i+1:>3} | {wf_row['label']:<35} | {avg:>+7.1f}% |"
        for v in vals:
            row_str += f" {v:>+7.1f}% |"
        row_str += f" {pos}/6 | {avg_mdd:>6.1f}% | {avg_wr:>5.1f}%"
        print(row_str)

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  FINAL VERDICT: TA-Lib CANDLESTICK PATTERNS ON FUTURES")
    print(f"{'=' * 180}")

    print(f"\n  1. TOP 10 INDIVIDUAL CANDLESTICK PATTERNS BY ANNUAL RETURN:")
    for i, r in enumerate(individual_results[:10]):
        print(f"    {i+1}. {r['pattern']} ({r['direction']}, H{r['hold']}): Ann={r['ann']:>+8.1f}%, WR={r['wr']:>5.1f}%, N={r['n']}, MDD={r['mdd']:>6.1f}%")

    print(f"\n  2. BEST COMBINED PATTERN CONFIGS:")
    combined = [r for r in all_results if r.get('test') in ('B', 'C', 'D', 'E', 'F', 'G', 'H')]
    for i, r in enumerate(combined[:10]):
        print(f"    {i+1}. {r.get('label', '?')}: Ann={r['ann']:>+8.1f}%, WR={r['wr']:>5.1f}%, N={r['n']}, MDD={r['mdd']:>6.1f}%")

    print(f"\n  3. CONFIGS BEATING +49.6%: {len(beating)}")
    if beating:
        for r in beating[:5]:
            print(f"     - {r.get('label', '?')}: Ann={r['ann']:>+8.1f}%")

    print(f"\n  4. WHICH CANDLESTICK PATTERNS WORK WITH NEXT-OPEN EXECUTION:")
    # Aggregate by pattern
    pattern_agg = {}
    for r in individual_results:
        key = (r['pattern'], r['direction'])
        if key not in pattern_agg:
            pattern_agg[key] = {'anns': [], 'wrs': [], 'ns': []}
        pattern_agg[key]['anns'].append(r['ann'])
        pattern_agg[key]['wrs'].append(r['wr'])
        pattern_agg[key]['ns'].append(r['n'])

    # Sort by average annual return
    pattern_list = [(k, v) for k, v in pattern_agg.items()]
    pattern_list.sort(key=lambda x: -np.mean(x[1]['anns']))

    print(f"     {'Pattern':<30} | {'Dir':>4} | {'Avg Ann':>9} | {'Avg WR':>7} | {'Total N':>8} | Verdict")
    print("     " + "-" * 100)
    for (pname, direction), stats in pattern_list[:20]:
        avg_ann = np.mean(stats['anns'])
        avg_wr = np.mean(stats['wrs'])
        total_n = sum(stats['ns'])
        if avg_ann > 20 and avg_wr > 50:
            verdict = "STRONG"
        elif avg_ann > 5 and avg_wr > 48:
            verdict = "MODERATE"
        elif avg_ann > 0:
            verdict = "WEAK"
        else:
            verdict = "NEGATIVE"
        print(f"     {pname:<30} | {direction:>4} | {avg_ann:>+8.1f}% | {avg_wr:>6.1f}% | {total_n:>8} | {verdict}")

    # Walk-forward summary for top configs
    print(f"\n  5. WALK-FORWARD SUMMARY (Top 15):")
    wf_summary = {}
    for wf in wf_rows:
        vals = [wf['windows'].get(yr, 0) for yr in wf_years]
        wf_summary[wf['label']] = {
            'avg': np.mean(vals) if vals else 0,
            'pos': sum(1 for v in vals if v > 0),
            'vals': vals,
        }
    wf_sorted = sorted(wf_summary.items(), key=lambda x: -x[1]['avg'])
    for label, ws in wf_sorted[:15]:
        vals_str = " ".join([f"{v:>+7.1f}%" for v in ws['vals']])
        print(f"     {label:<35}: WF Avg={ws['avg']:>+7.1f}%  Pos={ws['pos']}/6  [{vals_str}]")

    # Champion
    if all_results:
        champ = all_results[0]
        print(f"\n  {'='*70}")
        print(f"  CHAMPION: {champ.get('label', '?')}")
        print(f"    Annual: {champ['ann']:>+8.1f}%  |  WR: {champ['wr']:>5.1f}%  |  N: {champ['n']:>4}  |  MDD: {champ['mdd']:>6.1f}%")
        print(f"    Avg PnL/trade: {champ['avg_pnl']:>+6.3f}%  |  Avg Hold: {champ['avg_hold']:>5.1f}d  |  Freq: {champ['freq']:>5.1f}/yr")
        # WF for champion
        champ_wf = [w for w in wf_rows if w['label'] == champ.get('label', '')]
        if champ_wf:
            cw = champ_wf[0]
            vals = [cw['windows'].get(yr, 0) for yr in wf_years]
            print(f"    WF: {['{:>+7.1f}%'.format(v) for v in vals]}  |  {sum(1 for v in vals if v > 0)}/6 positive")
        print(f"  {'='*70}")

    print(f"\n  Total runtime: {time.time()-t_start:.0f}s")


def reconstruct_signal(r, cdl_results, bullish_count, bearish_count,
                       vol_ma20, sma50, V, C, OI, rsi14):
    """Reconstruct signal function from a result dict for walk-forward testing."""
    label = r.get('label', '')
    hold = r.get('hold', 5)

    # Test A: Individual patterns
    if r.get('test') == 'A':
        pname = r.get('pattern', '')
        direction = r.get('direction', 'bull')
        if pname not in cdl_results:
            return None, None
        arr = cdl_results[pname]
        cd_val = 100 if direction == 'bull' else -100
        def make_sig(cd_arr, cd_v):
            def sig(si, di):
                return cd_arr[si, di] == cd_v
            return sig
        return make_sig(arr, cd_val), None

    # Test B: Pattern count
    if r.get('test') == 'B':
        mc = r.get('min_count', 2)
        def make_sig_bc(bc, min_c):
            def sig(si, di):
                return bc[si, di] >= min_c
            return sig
        def sc_bc(bc):
            def sc(si, di):
                return float(bc[si, di])
            return sc
        return make_sig_bc(bullish_count, mc), sc_bc(bullish_count)

    # Test B_bear: Bearish count contrarian
    if r.get('test') == 'B_bear':
        mc = r.get('min_count', 2)
        def make_sig_bc_contr(bc, rsi, min_c):
            def sig(si, di):
                if bc[si, di] < min_c:
                    return False
                rv = rsi[si, di]
                if np.isnan(rv):
                    return False
                return rv < 30
            return sig
        return make_sig_bc_contr(bearish_count, rsi14, mc), None

    # Test C: Volume confirmation
    if r.get('test') == 'C':
        pname = r.get('pattern', '')
        direction = 'bull'
        if pname not in cdl_results:
            return None, None
        arr = cdl_results[pname]
        cd_val = 100
        def make_sig_vol(cd_arr, cd_v, vma, v):
            def sig(si, di):
                if cd_arr[si, di] != cd_v:
                    return False
                vv = v[si, di]
                vm = vma[si, di]
                if np.isnan(vv) or np.isnan(vm) or vm <= 0:
                    return False
                return vv > 1.5 * vm
            return sig
        return make_sig_vol(arr, cd_val, vol_ma20, V), None

    # Test D: Trend filter
    if r.get('test') == 'D':
        pname = r.get('pattern', '')
        if pname not in cdl_results:
            return None, None
        arr = cdl_results[pname]
        def make_sig_trend(cd_arr, sma, c):
            def sig(si, di):
                if cd_arr[si, di] != 100:
                    return False
                cc = c[si, di]
                ss = sma[si, di]
                if np.isnan(cc) or np.isnan(ss):
                    return False
                return cc > ss
            return sig
        return make_sig_trend(arr, sma50, C), None

    # Test E: OI confirmation
    if r.get('test') == 'E':
        pname = r.get('pattern', '')
        if pname not in cdl_results:
            return None, None
        arr = cdl_results[pname]
        def make_sig_oi(cd_arr, oi):
            def sig(si, di):
                if cd_arr[si, di] != 100:
                    return False
                if di < 5:
                    return False
                oi_now = oi[si, di]
                oi_ago = oi[si, di - 5]
                if np.isnan(oi_now) or np.isnan(oi_ago):
                    return False
                return oi_now > oi_ago
            return sig
        return make_sig_oi(arr, OI), None

    # Test F: Contrarian
    if r.get('test') == 'F':
        pname = r.get('pattern', '')
        if pname not in cdl_results:
            return None, None
        arr = cdl_results[pname]
        def make_sig_contr(cd_arr, rsi):
            def sig(si, di):
                if cd_arr[si, di] != -100:
                    return False
                rv = rsi[si, di]
                if np.isnan(rv):
                    return False
                return rv < 30
            return sig
        return make_sig_contr(arr, rsi14), None

    # Test G: Morning Star variants
    if r.get('test') == 'G':
        if 'MorningDojiStar' in label and 'Combined' not in label and 'Vol' not in label:
            arr = cdl_results['CDLMORNINGDOJISTAR']
            def make_sig(cd_arr):
                def sig(si, di):
                    return cd_arr[si, di] == 100
                return sig
            return make_sig(arr), None
        elif 'Combined' in label:
            arr1 = cdl_results['CDLMORNINGSTAR']
            arr2 = cdl_results['CDLMORNINGDOJISTAR']
            def make_sig(cd1, cd2):
                def sig(si, di):
                    return cd1[si, di] == 100 or cd2[si, di] == 100
                return sig
            return make_sig(arr1, arr2), None
        elif 'MorningStar_Vol' in label:
            arr = cdl_results['CDLMORNINGSTAR']
            def make_sig(cd_arr, vma, v):
                def sig(si, di):
                    if cd_arr[si, di] != 100:
                        return False
                    vv = v[si, di]
                    vm = vma[si, di]
                    if np.isnan(vv) or np.isnan(vm) or vm <= 0:
                        return False
                    return vv > 1.5 * vm
                return sig
            return make_sig(arr, vol_ma20, V), None
        else:
            arr = cdl_results['CDLMORNINGSTAR']
            def make_sig(cd_arr):
                def sig(si, di):
                    return cd_arr[si, di] == 100
                return sig
            return make_sig(arr), None

    # Test H: Engulfing + Hammer
    if r.get('test') == 'H':
        arr_eng = cdl_results['CDLENGULFING']
        arr_ham = cdl_results['CDLHAMMER']

        if 'EngOrHam' in label or 'EngHam_' in label:
            # Check if it has additional filters
            if 'Vol' in label:
                def make_sig(cd1, cd2, vma, v):
                    def sig(si, di):
                        if cd1[si, di] != 100 and cd2[si, di] != 100:
                            return False
                        vv = v[si, di]
                        vm = vma[si, di]
                        if np.isnan(vv) or np.isnan(vm) or vm <= 0:
                            return False
                        return vv > 1.5 * vm
                    return sig
                return make_sig(arr_eng, arr_ham, vol_ma20, V), None
            elif 'Trend' in label:
                def make_sig(cd1, cd2, sma, c):
                    def sig(si, di):
                        if cd1[si, di] != 100 and cd2[si, di] != 100:
                            return False
                        cc = c[si, di]
                        ss = sma[si, di]
                        if np.isnan(cc) or np.isnan(ss):
                            return False
                        return cc > ss
                    return sig
                return make_sig(arr_eng, arr_ham, sma50, C), None
            else:
                def make_sig(cd1, cd2):
                    def sig(si, di):
                        return cd1[si, di] == 100 or cd2[si, di] == 100
                    return sig
                return make_sig(arr_eng, arr_ham), None
        elif 'Engulfing' in label:
            def make_sig(cd_arr):
                def sig(si, di):
                    return cd_arr[si, di] == 100
                return sig
            return make_sig(arr_eng), None
        elif 'Hammer' in label:
            def make_sig(cd_arr):
                def sig(si, di):
                    return cd_arr[si, di] == 100
                return sig
            return make_sig(arr_ham), None

    return None, None


if __name__ == '__main__':
    main()
