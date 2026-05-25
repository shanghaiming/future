"""
Alpha Futures V109 -- PORTFOLIO OF INDEPENDENT STRATEGIES
=========================================================
Run multiple uncorrelated strategies simultaneously with allocated capital.
Improve risk-adjusted returns by diversifying across strategy types.

STRATEGIES:
  S1: ROC(5) cross zero -> buy, hold 5 days, top_n=1
  S2: T3 cross -> buy, hold 5 days, top_n=1
  S3: 50-day breakout -> buy, hold 10 days, top_n=1
  S4: Volatility breakout (range>2*ATR + bullish close) -> buy, hold 5 days, top_n=1
  S5: KAMA cross -> buy, hold 5 days, top_n=1

PORTFOLIO ALLOCATIONS:
  A) Equal weight (20% each)
  B) Best 3 only (33% each)
  C) Concentrated best 2 (50% each)
  D) Risk-parity weighted
  E) Dynamic allocation (monthly momentum-based)
  F) Independent full capital (non-exclusive signals)
  G) Signal overlap analysis

ALL signals computed at close di, entry at O[si, di+1] (NEXT DAY OPEN).
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
    print("Alpha Futures V109 -- PORTFOLIO OF INDEPENDENT STRATEGIES")
    print("=" * 180)
    print("\n  5 independent strategies, 7 portfolio allocation schemes.")
    print("  ALL signals at close di, entry at O[si, di+1] (NEXT DAY OPEN)")

    # -- Load data -------------------------------------------------
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ================================================================
    # PRECOMPUTE INDICATORS
    # ================================================================
    print("\n[Indicators] Computing...", flush=True)
    t0 = time.time()

    # -- ROC(5) --
    ROC5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        ROC5[si] = talib.ROC(c, timeperiod=5)
    print(f"  ROC(5) computed ({time.time()-t0:.1f}s)")

    # -- ROC(5) previous day (for cross-zero detection) --
    ROC5_prev = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            ROC5_prev[si, di] = ROC5[si, di - 1]

    # -- T3 (20-period) --
    T3_20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        T3_20[si] = talib.T3(c, timeperiod=20)
    print(f"  T3(20) computed ({time.time()-t0:.1f}s)")

    # -- T3 previous day --
    T3_prev = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            T3_prev[si, di] = T3_20[si, di - 1]

    # -- 50-day high (lookback window, not including today) --
    high50 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(50, ND):
            h50 = np.nanmax(C[si, di - 50:di])
            if not np.isnan(h50):
                high50[si, di] = h50
    print(f"  50-day high computed ({time.time()-t0:.1f}s)")

    # -- ATR(14) --
    ATR14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        h = H[si].astype(np.float64)
        l = L[si].astype(np.float64)
        c = C[si].astype(np.float64)
        ATR14[si] = talib.ATR(h, l, c, timeperiod=14)
    print(f"  ATR(14) computed ({time.time()-t0:.1f}s)")

    # -- Day range (H - L) --
    day_range = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            h = H[si, di]
            l = L[si, di]
            if not np.isnan(h) and not np.isnan(l):
                day_range[si, di] = h - l

    # -- KAMA(30) --
    KAMA30 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        KAMA30[si] = talib.KAMA(c, timeperiod=30)
    print(f"  KAMA(30) computed ({time.time()-t0:.1f}s)")

    # -- KAMA previous day --
    KAMA_prev = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            KAMA_prev[si, di] = KAMA30[si, di - 1]

    print(f"\n  All indicators computed ({time.time()-t_start:.1f}s total)")

    # ================================================================
    # SIGNAL GENERATION
    # ================================================================
    print("\n[Signals] Computing all 5 strategy signals...", flush=True)

    # S1: ROC(5) cross zero -> buy
    # Signal: ROC5[di] > 0 and ROC5_prev[di] <= 0 (crossed above zero)
    sig_s1 = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(10, ND):
            roc = ROC5[si, di]
            roc_p = ROC5_prev[si, di]
            if np.isnan(roc) or np.isnan(roc_p):
                continue
            if roc > 0 and roc_p <= 0:
                sig_s1[si, di] = True
    print(f"  S1) ROC(5) cross zero: {np.sum(sig_s1)} signals")

    # S2: T3 cross -> buy (price crosses above T3)
    # Signal: C[di] > T3[di] and C[di-1] <= T3[di-1]
    sig_s2 = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(25, ND):
            c = C[si, di]
            c_prev = C[si, di - 1]
            t3 = T3_20[si, di]
            t3_p = T3_prev[si, di]
            if np.isnan(c) or np.isnan(c_prev) or np.isnan(t3) or np.isnan(t3_p):
                continue
            if c > t3 and c_prev <= t3_p:
                sig_s2[si, di] = True
    print(f"  S2) T3 cross: {np.sum(sig_s2)} signals")

    # S3: 50-day breakout -> buy (close > 50-day high)
    sig_s3 = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(50, ND):
            c = C[si, di]
            h50 = high50[si, di]
            if np.isnan(c) or np.isnan(h50) or h50 <= 0:
                continue
            if c > h50:
                sig_s3[si, di] = True
    print(f"  S3) 50-day breakout: {np.sum(sig_s3)} signals")

    # S4: Volatility breakout (range > 2*ATR + bullish close)
    sig_s4 = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(20, ND):
            c = C[si, di]
            o = O[si, di]
            rng = day_range[si, di]
            atr = ATR14[si, di]
            if np.isnan(c) or np.isnan(o) or np.isnan(rng) or np.isnan(atr):
                continue
            if atr <= 0:
                continue
            if rng <= 2 * atr:
                continue
            if c <= o:  # not bullish
                continue
            sig_s4[si, di] = True
    print(f"  S4) Vol breakout: {np.sum(sig_s4)} signals")

    # S5: KAMA cross -> buy (price crosses above KAMA)
    # Signal: C[di] > KAMA[di] and C[di-1] <= KAMA[di-1]
    sig_s5 = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(35, ND):
            c = C[si, di]
            c_prev = C[si, di - 1]
            kama = KAMA30[si, di]
            kama_p = KAMA_prev[si, di]
            if np.isnan(c) or np.isnan(c_prev) or np.isnan(kama) or np.isnan(kama_p):
                continue
            if c > kama and c_prev <= kama_p:
                sig_s5[si, di] = True
    print(f"  S5) KAMA cross: {np.sum(sig_s5)} signals")

    signals = {
        'S1_ROC5':   {'sig': sig_s1, 'hold': 5,  'label': 'ROC(5) cross zero'},
        'S2_T3':     {'sig': sig_s2, 'hold': 5,  'label': 'T3 cross'},
        'S3_50day':  {'sig': sig_s3, 'hold': 10, 'label': '50-day breakout'},
        'S4_VolBrk': {'sig': sig_s4, 'hold': 5,  'label': 'Vol breakout'},
        'S5_KAMA':   {'sig': sig_s5, 'hold': 5,  'label': 'KAMA cross'},
    }
    strategy_keys = list(signals.keys())

    print(f"\n  All signals computed ({time.time()-t_start:.1f}s total)")

    # ================================================================
    # SINGLE-STRATEGY BACKTEST ENGINE
    # ================================================================
    def run_single_strategy(sig_arr, hold_days, allocated_capital,
                            start_di=MIN_TRAIN, end_di=None,
                            wf_test_year=None):
        """
        Run a single strategy with allocated_capital.
        Returns: dict with equity_curve (daily), trades list, final stats.
        equity_curve is indexed by di, value = total portfolio value at that di.
        """
        if end_di is None:
            end_di = ND

        # Walk-forward year boundaries
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
        else:
            test_start_di = start_di
            test_end_di = end_di

        cash = float(allocated_capital)
        positions = []
        trades = []

        # Build daily equity curve
        n_days_test = test_end_di - test_start_di
        equity = np.full(ND, np.nan)

        for di in range(test_start_di, test_end_di):
            # Reset at WF window start
            if wf_test_year is not None and di == test_start_di:
                cash = float(allocated_capital)
                positions = []

            # -- Close positions at end of day di --
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
                        'si': pos['si'],
                    })
                    closed.append(pos)

            for pos in closed:
                positions.remove(pos)

            # Compute equity at this day
            mkt_val_pos = 0.0
            for pos in positions:
                c_now = C[pos['si'], di]
                if np.isnan(c_now) or c_now <= 0:
                    c_now = pos['entry_price']
                mult = MULT.get(pos['sym'], DEF_MULT)
                mkt_val_pos += c_now * mult * abs(pos['lots'])
            equity[di] = cash + mkt_val_pos

            # If we have a position, don't open new one (top_n=1)
            if len(positions) >= 1:
                continue

            # -- Generate signal at day di, entry at di+1 --
            entry_di = di + 1
            if entry_di >= test_end_di:
                continue

            candidates = []
            for si in range(NS):
                if not sig_arr[si, di]:
                    continue
                ep = O[si, entry_di]
                if np.isnan(ep) or ep <= 0:
                    continue
                # Score: use ROC5 momentum as tiebreaker
                score = ROC5[si, di] if not np.isnan(ROC5[si, di]) else 0
                candidates.append((score, si, syms[si], ep))

            if not candidates:
                continue

            # Sort by score descending, take top 1
            candidates.sort(key=lambda x: -x[0])
            score, si, sym, price = candidates[0]
            mult = MULT.get(sym, DEF_MULT)
            notional = price * mult
            lots = max(1, int(cash / (notional * (1 + COMM))))
            cost_in = notional * lots * (1 + COMM)
            if cost_in > cash:
                lots = int(cash * 0.9 / (notional * (1 + COMM)))
                cost_in = notional * lots * (1 + COMM) if lots > 0 else 0
            if lots <= 0 or cost_in <= 0 or cost_in > cash:
                continue

            cash -= cost_in
            positions.append({
                'si': si, 'entry_price': price, 'entry_di': entry_di,
                'lots': lots, 'dir': 1, 'sym': sym,
                'hold_days': hold_days,
            })

        # Close remaining at end
        for pos in positions:
            ae = test_end_di - 1 if test_end_di < ND else ND - 1
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
                'si': pos['si'],
            })
        positions = []

        # Final equity
        final_eq = cash

        # Compute MDD from equity curve
        eq_valid = equity[~np.isnan(equity)]
        mdd = 0.0
        if len(eq_valid) > 0:
            peak = eq_valid[0]
            for ev in eq_valid:
                if ev > peak:
                    peak = ev
                dd = (ev - peak) / peak * 100
                if dd < mdd:
                    mdd = dd

        # WR
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0
        avg_hold = np.mean([t['days_held'] for t in trades]) if trades else 0

        return {
            'final_cash': final_eq,
            'equity': equity,
            'trades': trades,
            'mdd': mdd,
            'wr': wr,
            'n_trades': n_trades,
            'avg_pnl': avg_pnl,
            'avg_hold': avg_hold,
            'test_start_di': test_start_di,
            'test_end_di': test_end_di,
        }

    # ================================================================
    # STANDALONE RESULTS (each with full 500K capital)
    # ================================================================
    print("\n" + "=" * 180)
    print("  STANDALONE RESULTS (each strategy with full 500K capital)")
    print("=" * 180)
    print(f"  {'Strategy':<25} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'AvgHold':>7} | {'Final':>14}")
    print("-" * 130)

    standalone = {}
    for key in strategy_keys:
        s = signals[key]
        r = run_single_strategy(s['sig'], s['hold'], CASH0)
        if r is None:
            continue
        n_days_test = r['test_end_di'] - r['test_start_di']
        ann = annual_return(r['final_cash'], CASH0, n_days_test)
        r['ann'] = ann
        standalone[key] = r
        print(f"  {s['label']:<25} | {ann:>+8.1f}% | {r['wr']:>5.1f}% | {r['n_trades']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}% | {r['avg_hold']:>6.1f}d | {r['final_cash']:>13,.0f}")

    # ================================================================
    # PORTFOLIO ALLOCATION: A) EQUAL WEIGHT
    # ================================================================
    print("\n" + "=" * 180)
    print("  PORTFOLIO A) EQUAL WEIGHT (20% each, 100K per strategy)")
    print("=" * 180)

    def run_portfolio_alloc(weights, label, wf_test_year=None):
        """
        Run portfolio with given weight dict {strategy_key: weight_fraction}.
        Returns combined results.
        """
        cap = CASH0 if wf_test_year is None else CASH0

        all_equity = np.full(ND, 0.0)
        all_trades = []
        strat_results = {}

        for key, w in weights.items():
            s = signals[key]
            allocated = cap * w
            r = run_single_strategy(s['sig'], s['hold'], allocated, wf_test_year=wf_test_year)
            if r is None:
                strat_results[key] = None
                continue
            strat_results[key] = r
            # Add equity curve
            for di in range(ND):
                if not np.isnan(r['equity'][di]):
                    all_equity[di] += r['equity'][di]
            all_trades.extend(r['trades'])

        # Find valid range
        test_start = ND
        test_end = 0
        for key, r in strat_results.items():
            if r is not None:
                test_start = min(test_start, r['test_start_di'])
                test_end = max(test_end, r['test_end_di'])

        if test_start >= test_end:
            return None

        # Fill equity for days where some strategies have nan
        # (use initial allocation as placeholder)
        for di in range(test_start, test_end):
            if all_equity[di] == 0.0:
                # No data yet, use initial capital
                all_equity[di] = cap

        final = all_equity[test_end - 1] if test_end > 0 else cap
        n_days_test = test_end - test_start
        ann = annual_return(final, cap, n_days_test)

        # MDD from combined equity
        mdd = 0.0
        peak = float(cap)
        for di in range(test_start, test_end):
            ev = all_equity[di]
            if ev > peak:
                peak = ev
            dd = (ev - peak) / peak * 100
            if dd < mdd:
                mdd = dd

        # WR
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in all_trades]) * 100 if all_trades else 0
        n_trades = len(all_trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in all_trades]) if all_trades else 0

        # Daily returns for Sharpe
        daily_rets = []
        for di in range(test_start + 1, test_end):
            prev = all_equity[di - 1]
            cur = all_equity[di]
            if prev > 0 and not np.isnan(prev) and not np.isnan(cur):
                daily_rets.append((cur - prev) / prev)
        sharpe = 0.0
        if len(daily_rets) > 20:
            mean_r = np.mean(daily_rets)
            std_r = np.std(daily_rets)
            if std_r > 0:
                sharpe = mean_r / std_r * np.sqrt(252)

        return {
            'ann': ann, 'mdd': mdd, 'wr': wr, 'n_trades': n_trades,
            'avg_pnl': avg_pnl, 'sharpe': sharpe, 'final': final,
            'equity': all_equity, 'trades': all_trades,
            'test_start': test_start, 'test_end': test_end,
            'strat_results': strat_results,
            'label': label,
            'daily_rets': daily_rets,
        }

    # ================================================================
    # RUN ALL PORTFOLIO ALLOCATIONS
    # ================================================================
    portfolio_results = {}

    # --- A) EQUAL WEIGHT ---
    eq_weights = {k: 0.2 for k in strategy_keys}
    rA = run_portfolio_alloc(eq_weights, "A) Equal Weight (20%)")
    portfolio_results['A'] = rA

    # --- B) BEST 3 ONLY ---
    # ROC(5), T3, 50-day breakout (top 3 by standalone return)
    sorted_strats = sorted(standalone.items(), key=lambda x: -x[1]['ann'])
    best3_keys = [k for k, _ in sorted_strats[:3]]
    b3_weights = {k: 1.0/3 for k in best3_keys}
    rB = run_portfolio_alloc(b3_weights, "B) Best 3 (33%)")
    portfolio_results['B'] = rB

    # --- C) CONCENTRATED BEST 2 ---
    best2_keys = [k for k, _ in sorted_strats[:2]]
    c2_weights = {k: 0.5 for k in best2_keys}
    rC = run_portfolio_alloc(c2_weights, "C) Concentrated Best 2 (50%)")
    portfolio_results['C'] = rC

    # --- D) RISK-PARITY ---
    # Weight inversely proportional to MDD
    mdd_values = {
        'S1_ROC5': 41.7, 'S2_T3': 49.5, 'S3_50day': 16.4,
        'S4_VolBrk': 22.4, 'S5_KAMA': 56.2,
    }
    inv_mdd = {k: 1.0 / abs(v) for k, v in mdd_values.items()}
    total_inv = sum(inv_mdd.values())
    rp_weights = {k: v / total_inv for k, v in inv_mdd.items()}
    rD = run_portfolio_alloc(rp_weights, "D) Risk-Parity")
    portfolio_results['D'] = rD

    # --- E) DYNAMIC ALLOCATION ---
    print("\n[Portfolio E] Computing dynamic allocation (monthly momentum)...")

    def run_dynamic_portfolio(wf_test_year=None):
        """Monthly rebalance: allocate more to strategies with higher 30-day return."""
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
        else:
            test_start_di = MIN_TRAIN
            test_end_di = ND

        # For each strategy, run standalone to get per-day equity
        strat_equity = {}
        for key in strategy_keys:
            s = signals[key]
            allocated = CASH0
            r = run_single_strategy(s['sig'], s['hold'], allocated,
                                    start_di=test_start_di, end_di=test_end_di)
            if r is None:
                strat_equity[key] = np.full(ND, np.nan)
                continue
            strat_equity[key] = r['equity']

        # Now simulate dynamic allocation
        combined_equity = np.full(ND, np.nan)
        combined_equity[test_start_di] = CASH0
        last_rebalance_di = test_start_di
        current_weights = {k: 1.0 / len(strategy_keys) for k in strategy_keys}

        for di in range(test_start_di + 1, test_end_di):
            # Rebalance monthly (~21 trading days)
            if di - last_rebalance_di >= 21:
                last_rebalance_di = di
                # Compute 30-day return for each strategy
                strat_rets = {}
                for key in strategy_keys:
                    eq = strat_equity[key]
                    # Find 30 days ago
                    lookback = max(test_start_di, di - 30)
                    eq_now = eq[di - 1] if not np.isnan(eq[di - 1]) else None
                    eq_then = eq[lookback] if not np.isnan(eq[lookback]) else None
                    if eq_now and eq_then and eq_then > 0:
                        strat_rets[key] = (eq_now - eq_then) / eq_then
                    else:
                        strat_rets[key] = 0.0

                # Momentum-based weights: proportional to recent return
                # Shift to positive range for weighting
                min_ret = min(strat_rets.values())
                shifted = {k: v - min_ret + 0.01 for k, v in strat_rets.items()}
                total = sum(shifted.values())
                if total > 0:
                    current_weights = {k: v / total for k, v in shifted.items()}
                else:
                    current_weights = {k: 1.0 / len(strategy_keys) for k in strategy_keys}

            # Combined equity = weighted sum of individual equity curves
            # Normalize each equity to start from 1.0, then weight
            total_eq = 0.0
            for key in strategy_keys:
                eq = strat_equity[key]
                if not np.isnan(eq[di]) and not np.isnan(eq[test_start_di]) and eq[test_start_di] > 0:
                    # Normalized return * weight * capital
                    norm_ret = eq[di] / eq[test_start_di]
                    total_eq += current_weights[key] * norm_ret * CASH0
                else:
                    total_eq += current_weights[key] * CASH0
            combined_equity[di] = total_eq

        final = combined_equity[test_end_di - 1] if not np.isnan(combined_equity[test_end_di - 1]) else CASH0
        n_days_test = test_end_di - test_start_di
        ann = annual_return(final, CASH0, n_days_test)

        # MDD
        mdd = 0.0
        peak = float(CASH0)
        for di in range(test_start_di, test_end_di):
            ev = combined_equity[di]
            if np.isnan(ev):
                continue
            if ev > peak:
                peak = ev
            dd = (ev - peak) / peak * 100
            if dd < mdd:
                mdd = dd

        # Sharpe
        daily_rets = []
        for di in range(test_start_di + 1, test_end_di):
            prev = combined_equity[di - 1]
            cur = combined_equity[di]
            if not np.isnan(prev) and not np.isnan(cur) and prev > 0:
                daily_rets.append((cur - prev) / prev)
        sharpe = 0.0
        if len(daily_rets) > 20:
            sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)

        return {
            'ann': ann, 'mdd': mdd, 'sharpe': sharpe, 'final': final,
            'equity': combined_equity, 'n_days': n_days_test,
            'test_start': test_start_di, 'test_end': test_end_di,
            'label': 'E) Dynamic Allocation',
            'daily_rets': daily_rets, 'n_trades': 0, 'wr': 0, 'avg_pnl': 0,
        }

    rE = run_dynamic_portfolio()
    portfolio_results['E'] = rE

    # --- F) INDEPENDENT FULL CAPITAL ---
    print("\n[Portfolio F] Computing independent full capital...")

    def run_full_capital_independent(wf_test_year=None):
        """
        Each strategy uses 100% capital independently.
        When multiple signals fire on same day for DIFFERENT commodities, diversify.
        When same commodity gets multiple signals, concentrate (100%).
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
        else:
            test_start_di = MIN_TRAIN
            test_end_di = ND

        cash = float(CASH0)
        positions = []  # list of position dicts
        trades = []
        equity = np.full(ND, np.nan)

        for di in range(test_start_di, test_end_di):
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                positions = []

            # -- Close positions --
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
                        'pnl_pct': pnl_pct, 'entry_di': pos['entry_di'],
                        'exit_di': di, 'year': dates[di].year if di < ND else dates[-1].year,
                        'sym': pos.get('sym', ''), 'days_held': days_held,
                        'si': pos['si'], 'strategy': pos.get('strategy', ''),
                    })
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            # Compute equity
            mkt_val_pos = 0.0
            for pos in positions:
                c_now = C[pos['si'], di]
                if np.isnan(c_now) or c_now <= 0:
                    c_now = pos['entry_price']
                mult = MULT.get(pos['sym'], DEF_MULT)
                mkt_val_pos += c_now * mult * abs(pos['lots'])
            equity[di] = cash + mkt_val_pos

            # -- Collect all signals for this day --
            all_signals = []
            for key in strategy_keys:
                s = signals[key]
                for si in range(NS):
                    if not s['sig'][si, di]:
                        continue
                    if any(p['si'] == si for p in positions):
                        continue
                    ep = O[si, di + 1] if di + 1 < test_end_di else np.nan
                    if np.isnan(ep) or ep <= 0:
                        continue
                    score = ROC5[si, di] if not np.isnan(ROC5[si, di]) else 0
                    all_signals.append((score, si, syms[si], ep, key, s['hold']))

            if not all_signals:
                continue

            # Check: same commodity multiple signals -> concentrate
            sym_signal_count = {}
            for score, si, sym, ep, key, hold in all_signals:
                sym_signal_count[sym] = sym_signal_count.get(sym, 0) + 1

            # Unique commodities
            unique_syms = {}
            for score, si, sym, ep, key, hold in all_signals:
                if sym not in unique_syms or score > unique_syms[sym][0]:
                    unique_syms[sym] = (score, si, sym, ep, key, hold)

            candidates = list(unique_syms.values())
            candidates.sort(key=lambda x: -x[0])

            # Open positions -- diversify across different commodities
            n_open = len(positions)
            for score, si, sym, ep, key, hold in candidates:
                if n_open >= 5:  # max 5 concurrent
                    break
                mult = MULT.get(sym, DEF_MULT)
                notional = ep * mult
                # Allocate fraction of cash
                alloc = cash / max(1, min(len(candidates), 5 - n_open))
                lots = max(1, int(alloc / (notional * (1 + COMM))))
                cost_in = notional * lots * (1 + COMM)
                if cost_in > cash:
                    lots = int(cash * 0.9 / (notional * (1 + COMM)))
                    cost_in = notional * lots * (1 + COMM) if lots > 0 else 0
                if lots <= 0 or cost_in <= 0 or cost_in > cash:
                    continue

                cash -= cost_in
                positions.append({
                    'si': si, 'entry_price': ep, 'entry_di': di + 1,
                    'lots': lots, 'dir': 1, 'sym': sym,
                    'hold_days': hold, 'strategy': key,
                })
                n_open += 1

        # Close remaining
        for pos in positions:
            ae = test_end_di - 1 if test_end_di < ND else ND - 1
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
                'pnl_pct': pnl_pct, 'entry_di': pos['entry_di'],
                'exit_di': ae, 'year': dates[ae].year if ae < ND else dates[-1].year,
                'sym': pos.get('sym', ''), 'days_held': ae - pos['entry_di'],
                'si': pos['si'], 'strategy': pos.get('strategy', ''),
            })

        final = cash
        n_days_test = test_end_di - test_start_di
        ann = annual_return(final, CASH0, n_days_test)

        # MDD
        eq_valid = equity[~np.isnan(equity)]
        mdd = 0.0
        if len(eq_valid) > 0:
            peak = eq_valid[0]
            for ev in eq_valid:
                if ev > peak:
                    peak = ev
                dd = (ev - peak) / peak * 100
                if dd < mdd:
                    mdd = dd

        # Sharpe
        daily_rets = []
        for di in range(test_start_di + 1, test_end_di):
            prev = equity[di - 1]
            cur = equity[di]
            if not np.isnan(prev) and not np.isnan(cur) and prev > 0:
                daily_rets.append((cur - prev) / prev)
        sharpe = 0.0
        if len(daily_rets) > 20:
            sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)

        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0

        # Avg concurrent positions
        concurrent_counts = []
        for di in range(test_start_di, test_end_di):
            # Count positions that are active at this day
            count = sum(1 for t in trades if t['entry_di'] <= di <= t['exit_di'])
            concurrent_counts.append(count)
        avg_concurrent = np.mean(concurrent_counts) if concurrent_counts else 0

        return {
            'ann': ann, 'mdd': mdd, 'sharpe': sharpe, 'final': final,
            'equity': equity, 'trades': trades, 'wr': wr,
            'n_trades': n_trades, 'avg_pnl': avg_pnl,
            'test_start': test_start_di, 'test_end': test_end_di,
            'label': 'F) Full Capital Independent',
            'daily_rets': daily_rets,
            'avg_concurrent': avg_concurrent,
        }

    rF = run_full_capital_independent()
    portfolio_results['F'] = rF

    # ================================================================
    # G) SIGNAL OVERLAP ANALYSIS
    # ================================================================
    print("\n" + "=" * 180)
    print("  G) SIGNAL OVERLAP ANALYSIS")
    print("=" * 180)

    # For each pair, count overlap
    sig_names = {k: signals[k]['label'] for k in strategy_keys}

    # Count total signals per strategy
    sig_counts = {}
    for k in strategy_keys:
        sig_counts[k] = int(np.sum(signals[k]['sig']))
    print(f"\n  Signal counts:")
    for k in strategy_keys:
        print(f"    {sig_names[k]:<25}: {sig_counts[k]:>6} total signals")

    # Pairwise overlap
    print(f"\n  Pairwise overlap (same commodity, same day):")
    header = f"  {'':25} |"
    for k2 in strategy_keys:
        header += f" {sig_names[k2][:8]:>8} |"
    print(header)
    print("-" * (30 + 11 * len(strategy_keys)))

    overlap_matrix = {}
    for k1 in strategy_keys:
        row_str = f"  {sig_names[k1]:25} |"
        for k2 in strategy_keys:
            overlap = int(np.sum(signals[k1]['sig'] & signals[k2]['sig']))
            min_count = min(sig_counts[k1], sig_counts[k2]) if sig_counts[k1] > 0 and sig_counts[k2] > 0 else 1
            ratio = overlap / min_count * 100
            overlap_matrix[(k1, k2)] = (overlap, ratio)
            if k1 == k2:
                row_str += f" {'---':>8} |"
            else:
                row_str += f" {overlap:>5}/{ratio:>2.0f}% |"
        print(row_str)

    # Daily return correlation between strategies
    print(f"\n  Daily return correlation (from standalone runs):")
    strat_daily_rets = {}
    for key in strategy_keys:
        r = standalone[key]
        rets = []
        for di in range(r['test_start_di'] + 1, r['test_end_di']):
            prev = r['equity'][di - 1]
            cur = r['equity'][di]
            if not np.isnan(prev) and not np.isnan(cur) and prev > 0:
                rets.append((cur - prev) / prev)
            else:
                rets.append(0.0)
        strat_daily_rets[key] = rets

    # Make same length
    min_len = min(len(v) for v in strat_daily_rets.values())
    for key in strat_daily_rets:
        strat_daily_rets[key] = strat_daily_rets[key][:min_len]

    header = f"  {'':25} |"
    for k2 in strategy_keys:
        header += f" {sig_names[k2][:8]:>8} |"
    print(header)
    print("-" * (30 + 11 * len(strategy_keys)))

    corr_matrix = {}
    for k1 in strategy_keys:
        row_str = f"  {sig_names[k1]:25} |"
        for k2 in strategy_keys:
            r1 = np.array(strat_daily_rets[k1])
            r2 = np.array(strat_daily_rets[k2])
            if len(r1) > 20 and np.std(r1) > 0 and np.std(r2) > 0:
                corr = np.corrcoef(r1, r2)[0, 1]
            else:
                corr = 0.0
            corr_matrix[(k1, k2)] = corr
            if k1 == k2:
                row_str += f" {'1.00':>8} |"
            else:
                row_str += f" {corr:>+7.2f} |"
        print(row_str)

    # ================================================================
    # PORTFOLIO RESULTS SUMMARY
    # ================================================================
    print("\n" + "=" * 180)
    print("  PORTFOLIO ALLOCATION RESULTS (Full Period)")
    print("=" * 180)
    print(f"  {'Allocation':<35} | {'Ann':>9} | {'MDD':>7} | {'Sharpe':>7} | {'Final':>14} | {'N Trades':>9} | {'WR':>6} | {'AvgPnL':>7}")
    print("-" * 150)

    for pkey in ['A', 'B', 'C', 'D', 'E', 'F']:
        r = portfolio_results.get(pkey)
        if r is None:
            print(f"  {pkey}) {'N/A':<33} | ---")
            continue
        print(f"  {r['label']:<35} | {r['ann']:>+8.1f}% | {r['mdd']:>6.1f}% | {r['sharpe']:>6.2f} | {r['final']:>13,.0f} | {r['n_trades']:>9} | {r['wr']:>5.1f}% | {r['avg_pnl']:>+6.3f}%")

    # Also show standalone best
    print(f"\n  --- Standalone comparison ---")
    best_standalone_ann = -999
    best_standalone_key = None
    for key in strategy_keys:
        r = standalone[key]
        if r['ann'] > best_standalone_ann:
            best_standalone_ann = r['ann']
            best_standalone_key = key
    print(f"  Best standalone: {signals[best_standalone_key]['label']:<25} | {best_standalone_ann:>+8.1f}%")

    # ================================================================
    # WALK-FORWARD BY YEAR
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    print(f"\n" + "=" * 200)
    print(f"  WALK-FORWARD ANALYSIS (by year)")
    print(f"{'=' * 200}")

    # WF for each portfolio allocation
    print(f"\n  Portfolio Allocations Walk-Forward:")
    header = f"  {'Allocation':<35} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>7} |"
    header += f" {'Pos':>4} | {'MDD':>7} | {'Sharpe':>7}"
    print(header)
    print("-" * 200)

    for pkey in ['A', 'B', 'C', 'D', 'E', 'F']:
        if pkey in ['A', 'B', 'C', 'D']:
            # Use run_portfolio_alloc for WF
            if pkey == 'A':
                wf_weights = eq_weights
            elif pkey == 'B':
                wf_weights = b3_weights
            elif pkey == 'C':
                wf_weights = c2_weights
            elif pkey == 'D':
                wf_weights = rp_weights

            label = portfolio_results[pkey]['label'] if portfolio_results.get(pkey) else f"{pkey})"
            vals = []
            for yr in wf_years:
                wr = run_portfolio_alloc(wf_weights, label, wf_test_year=yr)
                if wr:
                    vals.append(wr['ann'])
                else:
                    vals.append(0)
            avg = np.mean(vals)
            pos = sum(1 for v in vals if v > 0)
            avg_mdd = portfolio_results[pkey]['mdd'] if portfolio_results.get(pkey) else 0
            avg_sharpe = portfolio_results[pkey]['sharpe'] if portfolio_results.get(pkey) else 0

        elif pkey == 'E':
            label = 'E) Dynamic Allocation'
            vals = []
            for yr in wf_years:
                wr = run_dynamic_portfolio(wf_test_year=yr)
                if wr:
                    vals.append(wr['ann'])
                else:
                    vals.append(0)
            avg = np.mean(vals)
            pos = sum(1 for v in vals if v > 0)
            avg_mdd = portfolio_results['E']['mdd'] if portfolio_results.get('E') else 0
            avg_sharpe = portfolio_results['E']['sharpe'] if portfolio_results.get('E') else 0

        elif pkey == 'F':
            label = 'F) Full Capital Independent'
            vals = []
            for yr in wf_years:
                wr = run_full_capital_independent(wf_test_year=yr)
                if wr:
                    vals.append(wr['ann'])
                else:
                    vals.append(0)
            avg = np.mean(vals)
            pos = sum(1 for v in vals if v > 0)
            avg_mdd = portfolio_results['F']['mdd'] if portfolio_results.get('F') else 0
            avg_sharpe = portfolio_results['F']['sharpe'] if portfolio_results.get('F') else 0

        row_str = f"  {label:<35} | {avg:>+7.1f}% |"
        for v in vals:
            row_str += f" {v:>+7.1f}% |"
        row_str += f" {pos}/6 | {avg_mdd:>6.1f}% | {avg_sharpe:>6.2f}"
        print(row_str)

    # WF for standalone strategies
    print(f"\n  Standalone Strategies Walk-Forward:")
    header = f"  {'Strategy':<35} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>7} |"
    header += f" {'Pos':>4} | {'MDD':>7}"
    print(header)
    print("-" * 200)

    for key in strategy_keys:
        s = signals[key]
        vals = []
        for yr in wf_years:
            wr = run_single_strategy(s['sig'], s['hold'], CASH0, wf_test_year=yr)
            if wr:
                n_days_test = wr['test_end_di'] - wr['test_start_di']
                ann = annual_return(wr['final_cash'], CASH0, n_days_test)
                vals.append(ann)
            else:
                vals.append(0)
        avg = np.mean(vals)
        pos = sum(1 for v in vals if v > 0)
        mdd = standalone[key]['mdd']
        row_str = f"  {s['label']:<35} | {avg:>+7.1f}% |"
        for v in vals:
            row_str += f" {v:>+7.1f}% |"
        row_str += f" {pos}/6 | {mdd:>6.1f}%"
        print(row_str)

    # ================================================================
    # AVERAGE CONCURRENT POSITIONS
    # ================================================================
    print(f"\n" + "=" * 180)
    print("  AVERAGE CONCURRENT POSITIONS")
    print("=" * 180)

    for pkey in ['A', 'B', 'C', 'D', 'F']:
        r = portfolio_results.get(pkey)
        if r is None:
            continue
        # Count concurrent positions from trades
        if r.get('trades'):
            concurrent_counts = []
            for di in range(r['test_start'], r['test_end']):
                count = sum(1 for t in r['trades'] if t['entry_di'] <= di <= t['exit_di'])
                concurrent_counts.append(count)
            avg_conc = np.mean(concurrent_counts) if concurrent_counts else 0
            max_conc = max(concurrent_counts) if concurrent_counts else 0
        else:
            avg_conc = 0
            max_conc = 0
        print(f"  {r['label']:<35} | Avg concurrent: {avg_conc:.2f} | Max concurrent: {max_conc}")

    if portfolio_results.get('F') and 'avg_concurrent' in portfolio_results['F']:
        print(f"  F) Full Capital (tracked)           | Avg: {portfolio_results['F']['avg_concurrent']:.2f}")

    # ================================================================
    # RISK-PARITY WEIGHTS DETAIL
    # ================================================================
    print(f"\n" + "=" * 180)
    print("  RISK-PARITY WEIGHTS (D)")
    print("=" * 180)
    for k, w in rp_weights.items():
        mdd_v = mdd_values[k]
        print(f"  {sig_names[k]:<25} | MDD: {mdd_v:>5.1f}% | Weight: {w*100:>5.1f}%")

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print(f"\n" + "=" * 180)
    print("  FINAL VERDICT")
    print("=" * 180)

    # Best by annual return
    best_ann = -999
    best_ann_label = ""
    for pkey in ['A', 'B', 'C', 'D', 'E', 'F']:
        r = portfolio_results.get(pkey)
        if r and r['ann'] > best_ann:
            best_ann = r['ann']
            best_ann_label = r['label']

    # Best by Sharpe
    best_sharpe = -999
    best_sharpe_label = ""
    for pkey in ['A', 'B', 'C', 'D', 'E', 'F']:
        r = portfolio_results.get(pkey)
        if r and r['sharpe'] > best_sharpe:
            best_sharpe = r['sharpe']
            best_sharpe_label = r['label']

    print(f"\n  1. Best Portfolio by Annual Return:  {best_ann_label} | {best_ann:>+.1f}%")
    print(f"  2. Best Portfolio by Sharpe:         {best_sharpe_label} | {best_sharpe:>.2f}")

    print(f"\n  3. Signal Overlap Summary:")
    low_overlap_pairs = []
    high_overlap_pairs = []
    for k1 in strategy_keys:
        for k2 in strategy_keys:
            if k1 >= k2:
                continue
            overlap, ratio = overlap_matrix[(k1, k2)]
            corr = corr_matrix[(k1, k2)]
            if ratio < 20:
                low_overlap_pairs.append((k1, k2, overlap, ratio, corr))
            elif ratio > 50:
                high_overlap_pairs.append((k1, k2, overlap, ratio, corr))

    if low_overlap_pairs:
        print(f"\n     LOW OVERLAP pairs (<20% overlap, good for diversification):")
        for k1, k2, ov, rat, corr in low_overlap_pairs:
            print(f"       {sig_names[k1]:<20} + {sig_names[k2]:<20} | overlap={ov:>5} ({rat:>4.1f}%) | corr={corr:>+.2f}")
    if high_overlap_pairs:
        print(f"\n     HIGH OVERLAP pairs (>50% overlap, redundant):")
        for k1, k2, ov, rat, corr in high_overlap_pairs:
            print(f"       {sig_names[k1]:<20} + {sig_names[k2]:<20} | overlap={ov:>5} ({rat:>4.1f}%) | corr={corr:>+.2f}")

    # All pairwise correlations
    print(f"\n     All pairwise correlations:")
    for k1 in strategy_keys:
        for k2 in strategy_keys:
            if k1 >= k2:
                continue
            corr = corr_matrix[(k1, k2)]
            print(f"       {sig_names[k1]:<20} vs {sig_names[k2]:<20} | corr = {corr:>+.3f}")

    # 4. Does diversification beat concentrated?
    print(f"\n  4. Diversification vs Concentrated ROC(5):")
    roc5_ann = standalone['S1_ROC5']['ann']
    print(f"     Standalone ROC(5): {roc5_ann:>+.1f}%")
    for pkey in ['A', 'B', 'C', 'D', 'E', 'F']:
        r = portfolio_results.get(pkey)
        if r:
            beats = "BEATS" if r['ann'] > roc5_ann else "LOSES TO"
            print(f"     {r['label']:<35} | {r['ann']:>+8.1f}% | {beats} ROC(5)")

    print(f"\n  Total runtime: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
