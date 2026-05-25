"""
Alpha Futures V125 -- MULTI-STRATEGY PORTFOLIO (Next-Open Execution)
====================================================================
Run 5 INDEPENDENT strategies simultaneously, combining signals to improve
risk-adjusted returns. V109 found correlations of 0.01-0.16 between strategies.

STRATEGIES:
  S1 - CHAMPION:      ROC(5)>1% + Z>1.5 + ROC improving, rank by ROC*Z, hold=1, top_n=1 (+333.5%)
  S2 - T3 CROSS:      price crosses above T3(20), hold=5, top_n=1 (+50.7%)
  S3 - VOL BREAKOUT:  range > 2*ATR AND C>O, hold=5, top_n=1 (+37.2%)
  S4 - Z-SCORE EXTREME: Z(today return, 20d) > 2.0, hold=3, top_n=3 (+73.9%)
  S5 - BREAKOUT STR:  50-day breakout ranked by strength, hold=5, top_n=1 (+60.6%)

PORTFOLIO APPROACHES A-F:
  A) Independent Full Capital -- take all signals, split if multiple
  B) Rotation -- best signal wins each day
  C) Signal Count Weighting -- conviction-based sizing
  D) Diversified Portfolio -- 5 independent equity curves
  E) Dynamic Allocation -- momentum-based monthly rebalancing
  F) Champion + Satellite -- 80/20 split

ALL signals computed at close di, entry at O[si, di+1] (NEXT DAY OPEN).
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


def sharpe_ratio(equity_curve, n_days):
    """Annualized Sharpe ratio from daily equity curve."""
    if len(equity_curve) < 20:
        return 0.0
    eq = np.array(equity_curve, dtype=float)
    rets = np.diff(eq) / eq[:-1]
    rets = rets[~np.isnan(rets)]
    if len(rets) < 10:
        return 0.0
    mean_r = np.mean(rets)
    std_r = np.std(rets, ddof=1)
    if std_r < 1e-10:
        return 0.0
    return mean_r / std_r * np.sqrt(252)


def compute_mdd(equity_curve):
    """Compute max drawdown from equity curve."""
    if len(equity_curve) < 2:
        return 0.0
    eq = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    return float(np.min(dd))


def main():
    print("=" * 180)
    print("  Alpha Futures V125 -- MULTI-STRATEGY PORTFOLIO (Next-Open Execution)")
    print("=" * 180)
    print("  5 independent strategies, 6 portfolio approaches, walk-forward validation")
    print("  ALL signals at close di, entry at O[si, di+1]")

    # -- Load data -------------------------------------------------
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")
    print(f"  MIN_TRAIN={MIN_TRAIN}, CASH0={CASH0:,}")

    # ================================================================
    # PRECOMPUTE INDICATORS
    # ================================================================
    print("\n[Indicators] Computing all indicators...", flush=True)
    t0 = time.time()

    # -- Daily returns --
    RET = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100

    # -- ROC(5) --
    ROC5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        ROC5[si] = talib.ROC(c, timeperiod=5)
    print(f"  ROC(5) ({time.time()-t0:.1f}s)", flush=True)

    # -- Z-score of daily returns (20-day) --
    ZSCORE_20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = RET[si, di-20:di]
            valid = rets[~np.isnan(rets)]
            if len(valid) < 10:
                continue
            mean_r = np.mean(valid)
            std_r = np.std(valid, ddof=1)
            if std_r > 0 and not np.isnan(RET[si, di]):
                ZSCORE_20[si, di] = (RET[si, di] - mean_r) / std_r
    print(f"  Z-score(20d) ({time.time()-t0:.1f}s)", flush=True)

    # -- T3(20) --
    T3_20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        T3_20[si] = talib.T3(c, timeperiod=20)
    print(f"  T3(20) ({time.time()-t0:.1f}s)", flush=True)

    # -- T3 previous day --
    T3_prev = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            T3_prev[si, di] = T3_20[si, di - 1]

    # -- ATR(14) --
    ATR14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        h = H[si].astype(np.float64)
        l = L[si].astype(np.float64)
        c = C[si].astype(np.float64)
        ATR14[si] = talib.ATR(h, l, c, timeperiod=14)
    print(f"  ATR(14) ({time.time()-t0:.1f}s)", flush=True)

    # -- Day range (H - L) --
    day_range = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            hv = H[si, di]
            lv = L[si, di]
            if not np.isnan(hv) and not np.isnan(lv):
                day_range[si, di] = hv - lv

    # -- 50-day high --
    high50 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(50, ND):
            h50 = np.nanmax(C[si, di-50:di])
            if not np.isnan(h50):
                high50[si, di] = h50
    print(f"  50-day high ({time.time()-t0:.1f}s)", flush=True)

    # -- Breakout strength (how far above 50d high, as % of ATR) --
    breakout_strength = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(50, ND):
            c = C[si, di]
            h50 = high50[si, di]
            atr = ATR14[si, di]
            if np.isnan(c) or np.isnan(h50) or h50 <= 0:
                continue
            if c > h50:
                if not np.isnan(atr) and atr > 0:
                    breakout_strength[si, di] = (c - h50) / atr
                else:
                    breakout_strength[si, di] = (c - h50) / h50 * 100
    print(f"  Breakout strength ({time.time()-t0:.1f}s)", flush=True)

    print(f"  All indicators computed ({time.time()-t_start:.1f}s total)")

    # ================================================================
    # SIGNAL GENERATION FOR 5 STRATEGIES
    # ================================================================
    print("\n[Signals] Computing 5 strategy signals...", flush=True)

    # S1 - CHAMPION: ROC(5)>1% + Z>1.5 + ROC improving, rank by ROC*Z
    sig_s1 = np.zeros((NS, ND), dtype=bool)
    s1_score = np.full((NS, ND), np.nan)  # ROC*Z score for ranking
    for si in range(NS):
        for di in range(22, ND):
            roc = ROC5[si, di]
            roc_prev = ROC5[si, di-1]
            z = ZSCORE_20[si, di]
            if np.isnan(roc) or np.isnan(z):
                continue
            if roc <= 1.0:
                continue
            if z <= 1.5:
                continue
            # ROC improving
            if np.isnan(roc_prev) or roc <= roc_prev:
                continue
            sig_s1[si, di] = True
            s1_score[si, di] = roc * z
    print(f"  S1) CHAMPION (ROC>1%+Z>1.5+improving): {int(np.sum(sig_s1))} signals")

    # S2 - T3 CROSS: price crosses above T3(20)
    sig_s2 = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(25, ND):
            c = C[si, di]
            c_prev = C[si, di-1]
            t3 = T3_20[si, di]
            t3_p = T3_prev[si, di]
            if np.isnan(c) or np.isnan(c_prev) or np.isnan(t3) or np.isnan(t3_p):
                continue
            if c > t3 and c_prev <= t3_p:
                sig_s2[si, di] = True
    print(f"  S2) T3 CROSS (price x above T3):        {int(np.sum(sig_s2))} signals")

    # S3 - VOL BREAKOUT: range > 2*ATR AND C>O
    sig_s3 = np.zeros((NS, ND), dtype=bool)
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
            if c <= o:
                continue
            sig_s3[si, di] = True
    print(f"  S3) VOL BREAKOUT (range>2*ATR + bull):   {int(np.sum(sig_s3))} signals")

    # S4 - Z-SCORE EXTREME: Z(today return, 20d) > 2.0
    sig_s4 = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(21, ND):
            z = ZSCORE_20[si, di]
            if np.isnan(z):
                continue
            if z > 2.0:
                sig_s4[si, di] = True
    print(f"  S4) Z-SCORE EXTREME (Z>2.0):            {int(np.sum(sig_s4))} signals")

    # S5 - BREAKOUT STRENGTH: 50-day breakout ranked by strength
    sig_s5 = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(50, ND):
            c = C[si, di]
            h50 = high50[si, di]
            if np.isnan(c) or np.isnan(h50):
                continue
            if c > h50:
                sig_s5[si, di] = True
    print(f"  S5) BREAKOUT 50-day (C > 50d high):     {int(np.sum(sig_s5))} signals")

    # Strategy metadata
    strategies = {
        'S1_CHAMP': {'sig': sig_s1, 'hold': 1, 'top_n': 1, 'score': s1_score,
                     'label': 'S1 CHAMPION (ROC+Z+imp)'},
        'S2_T3':    {'sig': sig_s2, 'hold': 5, 'top_n': 1, 'score': None,
                     'label': 'S2 T3 CROSS'},
        'S3_VOL':   {'sig': sig_s3, 'hold': 5, 'top_n': 1, 'score': None,
                     'label': 'S3 VOL BREAKOUT'},
        'S4_ZEXT':  {'sig': sig_s4, 'hold': 3, 'top_n': 3, 'score': None,
                     'label': 'S4 Z-SCORE EXTREME'},
        'S5_BRK':   {'sig': sig_s5, 'hold': 5, 'top_n': 1, 'score': breakout_strength,
                     'label': 'S5 BREAKOUT 50d'},
    }
    strat_keys = list(strategies.keys())

    # ================================================================
    # SIGNAL OVERLAP ANALYSIS
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  SIGNAL OVERLAP ANALYSIS")
    print(f"{'=' * 180}")

    # Count signals per strategy per day
    min_di = 51  # minimum di where all strategies can fire
    overlap_counts = {n: 0 for n in range(6)}  # 0-5 strategies agree on same commodity
    pair_overlap = {}
    for k1 in strat_keys:
        for k2 in strat_keys:
            if k1 < k2:
                pair_overlap[(k1, k2)] = 0

    total_signal_days = 0
    for di in range(min_di, ND):
        has_signal = False
        for si in range(NS):
            n_strats = sum(1 for k in strat_keys if strategies[k]['sig'][si, di])
            if n_strats > 0:
                has_signal = True
                overlap_counts[n_strats] += 1
            for k1 in strat_keys:
                for k2 in strat_keys:
                    if k1 < k2:
                        if strategies[k1]['sig'][si, di] and strategies[k2]['sig'][si, di]:
                            pair_overlap[(k1, k2)] += 1
        if has_signal:
            total_signal_days += 1

    print(f"\n  Total signal-commodity-day instances: {sum(overlap_counts.values())}")
    print(f"  Days with at least one signal: {total_signal_days}")
    print(f"\n  Distribution: # strategies agreeing on same commodity")
    for n in range(1, 6):
        pct = overlap_counts[n] / max(1, sum(overlap_counts.values())) * 100
        print(f"    {n} strategy(ies): {overlap_counts[n]:>6} ({pct:>5.1f}%)")

    print(f"\n  Pairwise overlap (commodity-days where both signal):")
    for (k1, k2), cnt in sorted(pair_overlap.items(), key=lambda x: -x[1]):
        s1_total = int(np.sum(strategies[k1]['sig']))
        s2_total = int(np.sum(strategies[k2]['sig']))
        min_total = max(1, min(s1_total, s2_total))
        print(f"    {strategies[k1]['label']:<30} x {strategies[k2]['label']:<30}: {cnt:>5} "
              f"({cnt/max(1,s1_total)*100:.1f}% of S1, {cnt/max(1,s2_total)*100:.1f}% of S2)")

    # ================================================================
    # HELPER: get candidates for a strategy on day di
    # ================================================================
    def get_candidates(strat_key, di, entry_di, exclude_si=None):
        """Return list of (score, si, sym, price) for a strategy."""
        s = strategies[strat_key]
        candidates = []
        for si in range(NS):
            if not s['sig'][si, di]:
                continue
            ep = O[si, entry_di]
            if np.isnan(ep) or ep <= 0:
                continue
            if exclude_si and si in exclude_si:
                continue
            # Score
            if s['score'] is not None:
                sc = s['score'][si, di]
                if np.isnan(sc):
                    sc = 0
            else:
                sc = ROC5[si, di] if not np.isnan(ROC5[si, di]) else 0
            candidates.append((sc, si, syms[si], ep))
        candidates.sort(key=lambda x: -x[0])
        return candidates

    # ================================================================
    # SINGLE-STRATEGY BACKTEST ENGINE (for standalone and approach D)
    # ================================================================
    def run_single_strategy(strat_key, allocated_capital, start_di=MIN_TRAIN, end_di=None):
        """Run a single strategy with allocated_capital. Returns equity list."""
        if end_di is None:
            end_di = ND
        s = strategies[strat_key]
        hold = s['hold']
        top_n = s['top_n']

        cash = float(allocated_capital)
        positions = []
        trades = []
        equity = []

        for di in range(start_di, end_di):
            # Track equity
            port_val = cash
            for pos in positions:
                cp = C[pos['si'], di]
                if not np.isnan(cp) and cp > 0:
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    port_val += cp * mult * pos['lots'] - cp * mult * abs(pos['lots']) * COMM
            equity.append(port_val)

            # Close positions whose hold is up
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
                    trades.append({'pnl': pnl, 'pnl_pct': pnl_pct,
                                   'entry_di': pos['entry_di'], 'exit_di': di,
                                   'sym': pos['sym'], 'strat': strat_key})
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            if len(positions) >= top_n:
                continue

            # Generate signals
            entry_di = di + 1
            if entry_di >= end_di:
                continue

            candidates = get_candidates(strat_key, di, entry_di)
            if not candidates:
                continue

            n_slots = top_n - len(positions)
            cap_per_slot = cash / max(1, n_slots)

            for sc_val, si, sym, price in candidates[:max(0, n_slots)]:
                mult = MULT.get(sym, DEF_MULT)
                contracts = max(1, int(cap_per_slot * 0.95 / (price * mult * (1 + COMM))))
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
                    'hold_days': hold,
                })

        # Close remaining
        for pos in positions:
            ae = end_di - 1
            exit_price = C[pos['si'], min(ae, ND-1)]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos['entry_price']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = exit_price * mult * abs(pos['lots'])
            cash += mkt_val - mkt_val * COMM
            pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
            invested = pos['entry_price'] * mult * abs(pos['lots'])
            pnl_pct = pnl / invested * 100 if invested > 0 else 0
            trades.append({'pnl': pnl, 'pnl_pct': pnl_pct,
                           'entry_di': pos['entry_di'], 'exit_di': ae,
                           'sym': pos['sym'], 'strat': strat_key})

        n_days_test = end_di - start_di
        ann = annual_return(cash, allocated_capital, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        mdd = compute_mdd(equity)
        sh = sharpe_ratio(equity, n_days_test)

        return {
            'ann': ann, 'wr': wr, 'n': len(trades), 'mdd': mdd, 'sharpe': sh,
            'final_cash': cash, 'equity': equity, 'trades': trades,
            'n_days': n_days_test,
        }

    # ================================================================
    # STANDALONE RESULTS (each with full CASH0)
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  STANDALONE RESULTS (each strategy with full CASH0={:,})".format(CASH0))
    print(f"{'=' * 180}")
    print(f"  {'Strategy':<35} | {'Ann':>10} | {'WR':>6} | {'N':>5} | {'Sharpe':>8} | {'MDD':>8} | {'Final':>14}")
    print("-" * 120)

    standalone = {}
    for key in strat_keys:
        s = strategies[key]
        r = run_single_strategy(key, CASH0)
        standalone[key] = r
        print(f"  {s['label']:<35} | {r['ann']:>+9.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} "
              f"| {r['sharpe']:>7.2f} | {r['mdd']:>+7.1f}% | {r['final_cash']:>13,.0f}")

    # ================================================================
    # WALK-FORWARD HELPER
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    def get_year_boundaries(yr):
        ts = te = None
        for di in range(ND):
            if dates[di].year == yr and ts is None:
                ts = di
            if dates[di].year == yr + 1 and te is None:
                te = di
        if ts is None:
            return None, None
        if te is None:
            te = ND
        return ts, te

    def run_single_wf(strat_key, capital, yr):
        ts, te = get_year_boundaries(yr)
        if ts is None:
            return None
        return run_single_strategy(strat_key, capital, start_di=ts, end_di=te)

    # ================================================================
    # APPROACH A: INDEPENDENT FULL CAPITAL
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  APPROACH A: INDEPENDENT FULL CAPITAL")
    print("  Each strategy gets 100% of capital. When multiple fire same day, take ALL (split cap).")
    print(f"{'=' * 180}")

    def run_approach_A(start_di=MIN_TRAIN, end_di=None):
        if end_di is None:
            end_di = ND
        cash = float(CASH0)
        positions = []
        trades = []
        equity = []

        for di in range(start_di, end_di):
            # Track equity
            port_val = cash
            for pos in positions:
                cp = C[pos['si'], di]
                if not np.isnan(cp) and cp > 0:
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    port_val += cp * mult * pos['lots'] - cp * mult * abs(pos['lots']) * COMM
            equity.append(port_val)

            # Close positions
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
                    trades.append({'pnl_pct': pnl_pct, 'sym': pos['sym'],
                                   'strat': pos['strat'], 'entry_di': pos['entry_di']})
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            # Collect ALL signals from all strategies
            entry_di = di + 1
            if entry_di >= end_di:
                continue

            # Gather unique commodity signals
            all_signals = {}  # si -> (best_score, strat_key)
            for key in strat_keys:
                s = strategies[key]
                for si in range(NS):
                    if not s['sig'][si, di]:
                        continue
                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    if any(p['si'] == si for p in positions):
                        continue
                    if s['score'] is not None:
                        sc = s['score'][si, di]
                        if np.isnan(sc):
                            sc = 0
                    else:
                        sc = ROC5[si, di] if not np.isnan(ROC5[si, di]) else 0
                    # Keep best score per commodity
                    if si not in all_signals or sc > all_signals[si][0]:
                        all_signals[si] = (sc, key)

            if not all_signals:
                continue

            # Sort by score
            sorted_sigs = sorted(all_signals.items(), key=lambda x: -x[1][0])
            # Take all signals (split capital equally)
            n_sigs = len(sorted_sigs)
            cap_per_sig = cash / n_sigs if n_sigs > 0 else 0

            for si, (sc, strat_key) in sorted_sigs:
                sym = syms[si]
                price = O[si, entry_di]
                mult = MULT.get(sym, DEF_MULT)
                contracts = max(1, int(cap_per_sig * 0.9 / (price * mult * (1 + COMM))))
                cost_in = price * mult * contracts * (1 + COMM)
                if cost_in > cash:
                    contracts = int(cash * 0.8 / (price * mult * (1 + COMM)))
                    cost_in = price * mult * contracts * (1 + COMM) if contracts > 0 else 0
                if contracts <= 0 or cost_in <= 0 or cost_in > cash:
                    continue
                cash -= cost_in
                positions.append({
                    'si': si, 'entry_price': price, 'entry_di': entry_di,
                    'lots': contracts, 'dir': 1, 'sym': sym,
                    'hold_days': strategies[strat_key]['hold'],
                    'strat': strat_key,
                })

        # Close remaining
        for pos in positions:
            ae = end_di - 1
            exit_price = C[pos['si'], min(ae, ND-1)]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos['entry_price']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = exit_price * mult * abs(pos['lots'])
            cash += mkt_val - mkt_val * COMM

        n_days_test = end_di - start_di
        ann = annual_return(cash, CASH0, n_days_test)
        mdd = compute_mdd(equity)
        sh = sharpe_ratio(equity, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        return {'ann': ann, 'wr': wr, 'n': len(trades), 'mdd': mdd, 'sharpe': sh,
                'final_cash': cash, 'equity': equity, 'n_days': n_days_test}

    rA = run_approach_A()
    print(f"  Annual: {rA['ann']:>+.1f}%  WR: {rA['wr']:.1f}%  Trades: {rA['n']}  "
          f"Sharpe: {rA['sharpe']:.2f}  MDD: {rA['mdd']:>+.1f}%  Final: {rA['final_cash']:,.0f}")

    # ================================================================
    # APPROACH B: ROTATION (best signal wins)
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  APPROACH B: ROTATION (best signal wins)")
    print("  All strategies vote. Pick highest historical WR strategy's signal.")
    print(f"{'=' * 180}")

    def run_approach_B(start_di=MIN_TRAIN, end_di=None):
        if end_di is None:
            end_di = ND
        cash = float(CASH0)
        positions = []
        trades = []
        equity = []

        # Historical WR ranking for tie-breaking (S1 > S4 > S5 > S2 > S3)
        strat_priority = {k: i for i, k in enumerate(strat_keys)}

        for di in range(start_di, end_di):
            # Track equity
            port_val = cash
            for pos in positions:
                cp = C[pos['si'], di]
                if not np.isnan(cp) and cp > 0:
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    port_val += cp * mult * pos['lots'] - cp * mult * abs(pos['lots']) * COMM
            equity.append(port_val)

            # Close positions
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
                    trades.append({'pnl_pct': pnl_pct, 'sym': pos['sym'], 'strat': pos['strat']})
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            if len(positions) >= 1:
                continue

            entry_di = di + 1
            if entry_di >= end_di:
                continue

            # Collect best signal from each strategy
            best_per_strat = {}
            for key in strat_keys:
                cands = get_candidates(key, di, entry_di)
                if cands:
                    best_per_strat[key] = cands[0]  # (score, si, sym, price)

            if not best_per_strat:
                continue

            # Pick highest priority strategy's best signal
            best_key = min(best_per_strat.keys(), key=lambda k: strat_priority[k])
            sc_val, si, sym, price = best_per_strat[best_key]
            mult = MULT.get(sym, DEF_MULT)
            contracts = max(1, int(cash * 0.95 / (price * mult * (1 + COMM))))
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
                'hold_days': strategies[best_key]['hold'],
                'strat': best_key,
            })

        # Close remaining
        for pos in positions:
            ae = end_di - 1
            exit_price = C[pos['si'], min(ae, ND-1)]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos['entry_price']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = exit_price * mult * abs(pos['lots'])
            cash += mkt_val - mkt_val * COMM

        n_days_test = end_di - start_di
        ann = annual_return(cash, CASH0, n_days_test)
        mdd = compute_mdd(equity)
        sh = sharpe_ratio(equity, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        return {'ann': ann, 'wr': wr, 'n': len(trades), 'mdd': mdd, 'sharpe': sh,
                'final_cash': cash, 'equity': equity, 'n_days': n_days_test}

    rB = run_approach_B()
    print(f"  Annual: {rB['ann']:>+.1f}%  WR: {rB['wr']:.1f}%  Trades: {rB['n']}  "
          f"Sharpe: {rB['sharpe']:.2f}  MDD: {rB['mdd']:>+.1f}%  Final: {rB['final_cash']:,.0f}")

    # ================================================================
    # APPROACH C: SIGNAL COUNT WEIGHTING (conviction-based sizing)
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  APPROACH C: SIGNAL COUNT WEIGHTING")
    print("  5/5 signals -> 50% cap, 4/5 -> 30%, 3/5 -> 20%, <3 -> skip")
    print(f"{'=' * 180}")

    def run_approach_C(start_di=MIN_TRAIN, end_di=None):
        if end_di is None:
            end_di = ND
        cash = float(CASH0)
        positions = []
        trades = []
        equity = []

        for di in range(start_di, end_di):
            # Track equity
            port_val = cash
            for pos in positions:
                cp = C[pos['si'], di]
                if not np.isnan(cp) and cp > 0:
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    port_val += cp * mult * pos['lots'] - cp * mult * abs(pos['lots']) * COMM
            equity.append(port_val)

            # Close positions
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
                    trades.append({'pnl_pct': pnl_pct, 'sym': pos['sym'], 'strat': pos['strat']})
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            if len(positions) >= 3:
                continue

            entry_di = di + 1
            if entry_di >= end_di:
                continue

            # Count signals per commodity
            commodity_votes = {}  # si -> (count, score)
            for key in strat_keys:
                s = strategies[key]
                for si in range(NS):
                    if not s['sig'][si, di]:
                        continue
                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    if any(p['si'] == si for p in positions):
                        continue
                    if s['score'] is not None:
                        sc = s['score'][si, di]
                        if np.isnan(sc):
                            sc = 0
                    else:
                        sc = ROC5[si, di] if not np.isnan(ROC5[si, di]) else 0
                    if si not in commodity_votes:
                        commodity_votes[si] = [0, 0.0]
                    commodity_votes[si][0] += 1
                    commodity_votes[si][1] += sc

            # Filter by conviction
            conviction_sizing = {5: 0.50, 4: 0.30, 3: 0.20}
            candidates = []
            for si, (cnt, total_sc) in commodity_votes.items():
                if cnt < 3:
                    continue
                size_frac = conviction_sizing.get(cnt, 0)
                if size_frac <= 0:
                    continue
                sym = syms[si]
                price = O[si, entry_di]
                candidates.append((cnt, total_sc, si, sym, price, size_frac))

            if not candidates:
                continue

            # Sort by conviction (count desc), then score
            candidates.sort(key=lambda x: (-x[0], -x[1]))

            n_slots = 3 - len(positions)
            for cnt, total_sc, si, sym, price, size_frac in candidates[:max(0, n_slots)]:
                mult = MULT.get(sym, DEF_MULT)
                alloc = cash * size_frac
                contracts = max(1, int(alloc / (price * mult * (1 + COMM))))
                cost_in = price * mult * contracts * (1 + COMM)
                if cost_in > cash:
                    contracts = int(cash * 0.8 / (price * mult * (1 + COMM)))
                    cost_in = price * mult * contracts * (1 + COMM) if contracts > 0 else 0
                if contracts <= 0 or cost_in <= 0 or cost_in > cash:
                    continue
                cash -= cost_in
                # Use max hold of the strategies that voted
                max_hold = max(strategies[k]['hold'] for k in strat_keys if strategies[k]['sig'][si, di])
                positions.append({
                    'si': si, 'entry_price': price, 'entry_di': entry_di,
                    'lots': contracts, 'dir': 1, 'sym': sym,
                    'hold_days': max_hold,
                    'strat': f"CNT{cnt}",
                })

        # Close remaining
        for pos in positions:
            ae = end_di - 1
            exit_price = C[pos['si'], min(ae, ND-1)]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos['entry_price']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = exit_price * mult * abs(pos['lots'])
            cash += mkt_val - mkt_val * COMM

        n_days_test = end_di - start_di
        ann = annual_return(cash, CASH0, n_days_test)
        mdd = compute_mdd(equity)
        sh = sharpe_ratio(equity, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        return {'ann': ann, 'wr': wr, 'n': len(trades), 'mdd': mdd, 'sharpe': sh,
                'final_cash': cash, 'equity': equity, 'n_days': n_days_test}

    rC = run_approach_C()
    print(f"  Annual: {rC['ann']:>+.1f}%  WR: {rC['wr']:.1f}%  Trades: {rC['n']}  "
          f"Sharpe: {rC['sharpe']:.2f}  MDD: {rC['mdd']:>+.1f}%  Final: {rC['final_cash']:,.0f}")

    # ================================================================
    # APPROACH D: DIVERSIFIED PORTFOLIO (1 position per strategy)
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  APPROACH D: DIVERSIFIED PORTFOLIO (5 independent equity curves)")
    print("  Capital split equally at start (100K each). No rebalancing.")
    print(f"{'=' * 180}")

    def run_approach_D(start_di=MIN_TRAIN, end_di=None):
        if end_di is None:
            end_di = ND
        cap_each = CASH0 / 5
        strat_results = {}
        for key in strat_keys:
            r = run_single_strategy(key, cap_each, start_di, end_di)
            strat_results[key] = r

        # Combine equity curves
        min_len = min(len(strat_results[k]['equity']) for k in strat_keys)
        combined_equity = []
        for i in range(min_len):
            total = sum(strat_results[k]['equity'][i] for k in strat_keys)
            combined_equity.append(total)

        total_trades = sum(strat_results[k]['n'] for k in strat_keys)
        total_final = sum(strat_results[k]['final_cash'] for k in strat_keys)
        total_wr_list = []
        for k in strat_keys:
            for t in strat_results[k]['trades']:
                total_wr_list.append(1 if t['pnl_pct'] > 0 else 0)
        total_wr = np.mean(total_wr_list) * 100 if total_wr_list else 0

        n_days_test = end_di - start_di
        ann = annual_return(total_final, CASH0, n_days_test)
        mdd = compute_mdd(combined_equity)
        sh = sharpe_ratio(combined_equity, n_days_test)

        return {
            'ann': ann, 'wr': total_wr, 'n': total_trades, 'mdd': mdd, 'sharpe': sh,
            'final_cash': total_final, 'equity': combined_equity, 'n_days': n_days_test,
            'strat_results': strat_results,
        }

    rD = run_approach_D()
    print(f"  Portfolio total:")
    print(f"    Annual: {rD['ann']:>+.1f}%  WR: {rD['wr']:.1f}%  Trades: {rD['n']}  "
          f"Sharpe: {rD['sharpe']:.2f}  MDD: {rD['mdd']:>+.1f}%  Final: {rD['final_cash']:,.0f}")
    print(f"  Individual strategy results within portfolio:")
    for key in strat_keys:
        sr = rD['strat_results'][key]
        print(f"    {strategies[key]['label']:<35}: Ann={sr['ann']:>+7.1f}%  Final={sr['final_cash']:>10,.0f}  "
              f"WR={sr['wr']:>5.1f}%  N={sr['n']:>4}  Sharpe={sr['sharpe']:.2f}")

    # ================================================================
    # APPROACH E: DYNAMIC ALLOCATION (momentum-based monthly rebalancing)
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  APPROACH E: DYNAMIC ALLOCATION")
    print("  Monthly rebalancing: last 3 months' return determines allocation.")
    print("  Min 10% per strategy, max 50%.")
    print(f"{'=' * 180}")

    def run_approach_E(start_di=MIN_TRAIN, end_di=None):
        if end_di is None:
            end_di = ND
        cash = float(CASH0)
        positions = []
        trades = []
        equity = []

        # Equal initial weights
        weights = {k: 0.2 for k in strat_keys}

        # Track monthly performance per strategy
        strat_monthly_ret = {k: [] for k in strat_keys}
        strat_month_start_eq = {k: CASH0 * 0.2 for k in strat_keys}
        current_month = dates[start_di].month if start_di < ND else 1
        current_year = dates[start_di].year if start_di < ND else 2020

        for di in range(start_di, end_di):
            # Track equity
            port_val = cash
            for pos in positions:
                cp = C[pos['si'], di]
                if not np.isnan(cp) and cp > 0:
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    port_val += cp * mult * pos['lots'] - cp * mult * abs(pos['lots']) * COMM
            equity.append(port_val)

            # Monthly rebalancing
            if di < ND and dates[di].month != current_month:
                # Record last month's performance per strategy
                for key in strat_keys:
                    strat_eq = strat_month_start_eq.get(key, 0)
                    if strat_eq > 0:
                        # Approximate: use proportional allocation of total
                        strat_ret = (port_val / CASH0 - 1) * weights[key] / 0.2
                        strat_monthly_ret[key].append(strat_ret)
                strat_month_start_eq = {k: port_val * weights[k] for k in strat_keys}

                # Rebalance weights based on last 3 months
                new_weights = {}
                perf_scores = {}
                for key in strat_keys:
                    recent = strat_monthly_ret[key][-3:] if strat_monthly_ret[key] else [0]
                    perf_scores[key] = np.mean(recent) if recent else 0

                # Normalize to [0.10, 0.50] range
                total_score = sum(max(0.01, v) for v in perf_scores.values())
                if total_score <= 0:
                    new_weights = {k: 0.2 for k in strat_keys}
                else:
                    raw = {k: max(0.01, v) / total_score for k, v in perf_scores.items()}
                    # Enforce min 10% and max 50%
                    for k in strat_keys:
                        raw[k] = max(0.10, min(0.50, raw[k]))
                    # Renormalize
                    tw = sum(raw.values())
                    new_weights = {k: raw[k] / tw for k in strat_keys}

                weights = new_weights
                current_month = dates[di].month
                current_year = dates[di].year

            # Close positions
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
                    trades.append({'pnl_pct': pnl_pct, 'sym': pos['sym'], 'strat': pos['strat']})
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            if len(positions) >= 3:
                continue

            entry_di = di + 1
            if entry_di >= end_di:
                continue

            # Collect signals weighted by strategy allocation
            weighted_candidates = []
            for key in strat_keys:
                s = strategies[key]
                w = weights.get(key, 0.2)
                cands = get_candidates(key, di, entry_di,
                                       exclude_si=set(p['si'] for p in positions))
                for sc_val, si, sym, price in cands[:1]:
                    weighted_candidates.append((sc_val * w, si, sym, price, key, w))

            if not weighted_candidates:
                continue

            weighted_candidates.sort(key=lambda x: -x[0])
            n_slots = 3 - len(positions)

            for wsc, si, sym, price, strat_key, w in weighted_candidates[:max(0, n_slots)]:
                alloc = cash * min(w * 2, 0.5)  # Scale up allocation, cap at 50%
                mult = MULT.get(sym, DEF_MULT)
                contracts = max(1, int(alloc / (price * mult * (1 + COMM))))
                cost_in = price * mult * contracts * (1 + COMM)
                if cost_in > cash:
                    contracts = int(cash * 0.8 / (price * mult * (1 + COMM)))
                    cost_in = price * mult * contracts * (1 + COMM) if contracts > 0 else 0
                if contracts <= 0 or cost_in <= 0 or cost_in > cash:
                    continue
                cash -= cost_in
                positions.append({
                    'si': si, 'entry_price': price, 'entry_di': entry_di,
                    'lots': contracts, 'dir': 1, 'sym': sym,
                    'hold_days': strategies[strat_key]['hold'],
                    'strat': strat_key,
                })

        # Close remaining
        for pos in positions:
            ae = end_di - 1
            exit_price = C[pos['si'], min(ae, ND-1)]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos['entry_price']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = exit_price * mult * abs(pos['lots'])
            cash += mkt_val - mkt_val * COMM

        n_days_test = end_di - start_di
        ann = annual_return(cash, CASH0, n_days_test)
        mdd = compute_mdd(equity)
        sh = sharpe_ratio(equity, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        return {'ann': ann, 'wr': wr, 'n': len(trades), 'mdd': mdd, 'sharpe': sh,
                'final_cash': cash, 'equity': equity, 'n_days': n_days_test}

    rE = run_approach_E()
    print(f"  Annual: {rE['ann']:>+.1f}%  WR: {rE['wr']:.1f}%  Trades: {rE['n']}  "
          f"Sharpe: {rE['sharpe']:.2f}  MDD: {rE['mdd']:>+.1f}%  Final: {rE['final_cash']:,.0f}")

    # ================================================================
    # APPROACH F: CHAMPION + SATELLITE (80/20)
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  APPROACH F: CHAMPION + SATELLITE (80% champion / 20% satellites)")
    print(f"{'=' * 180}")

    def run_approach_F(start_di=MIN_TRAIN, end_di=None):
        if end_di is None:
            end_di = ND

        # Champion gets 80%
        champ_cap = CASH0 * 0.80
        r_champ = run_single_strategy('S1_CHAMP', champ_cap, start_di, end_di)

        # 20% split equally among 4 satellites
        sat_cap = CASH0 * 0.05  # 5% each
        sat_results = {}
        for key in strat_keys:
            if key == 'S1_CHAMP':
                continue
            sat_results[key] = run_single_strategy(key, sat_cap, start_di, end_di)

        # Combine equity
        min_len = min(len(r_champ['equity']), *(len(sat_results[k]['equity']) for k in sat_results))
        combined_equity = []
        for i in range(min_len):
            total = r_champ['equity'][i] + sum(sat_results[k]['equity'][i] for k in sat_results)
            combined_equity.append(total)

        total_trades = r_champ['n'] + sum(sat_results[k]['n'] for k in sat_results)
        total_final = r_champ['final_cash'] + sum(sat_results[k]['final_cash'] for k in sat_results)
        total_wr_list = []
        for t in r_champ['trades']:
            total_wr_list.append(1 if t['pnl_pct'] > 0 else 0)
        for k in sat_results:
            for t in sat_results[k]['trades']:
                total_wr_list.append(1 if t['pnl_pct'] > 0 else 0)
        total_wr = np.mean(total_wr_list) * 100 if total_wr_list else 0

        n_days_test = end_di - start_di
        ann = annual_return(total_final, CASH0, n_days_test)
        mdd = compute_mdd(combined_equity)
        sh = sharpe_ratio(combined_equity, n_days_test)

        return {
            'ann': ann, 'wr': total_wr, 'n': total_trades, 'mdd': mdd, 'sharpe': sh,
            'final_cash': total_final, 'equity': combined_equity, 'n_days': n_days_test,
            'champ_result': r_champ, 'sat_results': sat_results,
        }

    rF = run_approach_F()
    print(f"  Portfolio total:")
    print(f"    Annual: {rF['ann']:>+.1f}%  WR: {rF['wr']:.1f}%  Trades: {rF['n']}  "
          f"Sharpe: {rF['sharpe']:.2f}  MDD: {rF['mdd']:>+.1f}%  Final: {rF['final_cash']:,.0f}")
    print(f"  Champion (80% capital):")
    cr = rF['champ_result']
    print(f"    S1 CHAMPION: Ann={cr['ann']:>+7.1f}%  Final={cr['final_cash']:>10,.0f}  "
          f"WR={cr['wr']:>5.1f}%  Sharpe={cr['sharpe']:.2f}")
    print(f"  Satellites (5% capital each):")
    for key in strat_keys:
        if key == 'S1_CHAMP':
            continue
        sr = rF['sat_results'][key]
        print(f"    {strategies[key]['label']:<35}: Final={sr['final_cash']:>10,.0f}  "
              f"Ann={sr['ann']:>+7.1f}%  WR={sr['wr']:>5.1f}%")

    # ================================================================
    # WALK-FORWARD FOR ALL APPROACHES
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  WALK-FORWARD VALIDATION (2020-2025)")
    print(f"{'=' * 180}")

    approach_funcs = {
        'A) Indep Full Cap': run_approach_A,
        'B) Rotation':       run_approach_B,
        'C) Signal Count':   run_approach_C,
        'D) Diversified':    run_approach_D,
        'E) Dynamic Alloc':  run_approach_E,
        'F) Champ+Sat(80/20)': run_approach_F,
    }

    # Also standalone champion for comparison
    def run_champ_standalone(start_di=MIN_TRAIN, end_di=None):
        return run_single_strategy('S1_CHAMP', CASH0, start_di, end_di)

    wf_results = {}
    for ap_name, ap_func in list(approach_funcs.items()) + [('S1 STANDALONE', run_champ_standalone)]:
        wf_results[ap_name] = {}
        for yr in wf_years:
            ts, te = get_year_boundaries(yr)
            if ts is None:
                wf_results[ap_name][yr] = None
                continue
            r = ap_func(start_di=ts, end_di=te)
            wf_results[ap_name][yr] = r

    # Print WF table
    print(f"\n  {'Approach':<25} | {'AvgAnn':>8} |", end="")
    for yr in wf_years:
        print(f" {yr:>8} |", end="")
    print(f" {'WF+':>4} | {'AvgMDD':>8} | {'AvgSh':>6}")
    print("-" * 170)

    wf_summary = {}
    for ap_name in list(approach_funcs.keys()) + ['S1 STANDALONE']:
        vals = {}
        mdds = []
        sharpes = []
        for yr in wf_years:
            r = wf_results[ap_name].get(yr)
            if r is not None:
                vals[yr] = r['ann']
                mdds.append(r['mdd'])
                sharpes.append(r.get('sharpe', 0))
            else:
                vals[yr] = 0
        avg_ann = np.mean(list(vals.values()))
        pos = sum(1 for v in vals.values() if v > 0)
        avg_mdd = np.mean(mdds) if mdds else 0
        avg_sh = np.mean(sharpes) if sharpes else 0

        wf_summary[ap_name] = {
            'avg_ann': avg_ann, 'pos': pos, 'avg_mdd': avg_mdd, 'avg_sh': avg_sh,
            'vals': vals,
        }

        row = f"  {ap_name:<25} | {avg_ann:>+7.1f}% |"
        for yr in wf_years:
            row += f" {vals[yr]:>+7.1f}% |"
        row += f" {pos}/6 | {avg_mdd:>+7.1f}% | {avg_sh:>5.2f}"
        print(row)

    # ================================================================
    # COMPREHENSIVE COMPARISON TABLE
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  FULL-PERIOD COMPARISON (all approaches)")
    print(f"{'=' * 180}")
    print(f"  {'Approach':<25} | {'Ann':>10} | {'WR':>6} | {'N':>5} | {'Sharpe':>8} | {'MDD':>8} | {'WF Avg':>8} | {'WF+':>4} | {'Final':>14}")
    print("-" * 140)

    all_approach_results = {
        'A) Indep Full Cap': rA,
        'B) Rotation': rB,
        'C) Signal Count': rC,
        'D) Diversified': rD,
        'E) Dynamic Alloc': rE,
        'F) Champ+Sat(80/20)': rF,
    }

    # Add standalone champion
    champ_standalone = standalone['S1_CHAMP']
    all_approach_results['S1 STANDALONE'] = champ_standalone

    for ap_name, r in all_approach_results.items():
        ws = wf_summary.get(ap_name, {})
        wf_avg = ws.get('avg_ann', 0)
        wf_pos = ws.get('pos', 0)
        print(f"  {ap_name:<25} | {r['ann']:>+9.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} "
              f"| {r.get('sharpe',0):>7.2f} | {r['mdd']:>+7.1f}% | {wf_avg:>+7.1f}% | {wf_pos}/6 "
              f"| {r['final_cash']:>13,.0f}")

    # ================================================================
    # FINAL REPORT
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  FINAL REPORT")
    print(f"{'=' * 180}")

    # 1. Best by annual return
    best_ann_name = max(all_approach_results.keys(), key=lambda k: all_approach_results[k]['ann'])
    best_ann = all_approach_results[best_ann_name]
    print(f"\n  1. BEST BY ANNUAL RETURN:")
    print(f"     {best_ann_name}: {best_ann['ann']:>+.1f}% (Sharpe={best_ann.get('sharpe',0):.2f}, MDD={best_ann['mdd']:>+.1f}%)")

    # 2. Best by Sharpe
    best_sh_name = max(all_approach_results.keys(), key=lambda k: all_approach_results[k].get('sharpe', 0))
    best_sh = all_approach_results[best_sh_name]
    print(f"\n  2. BEST BY SHARPE RATIO:")
    print(f"     {best_sh_name}: Sharpe={best_sh.get('sharpe',0):.2f} (Ann={best_sh['ann']:>+.1f}%, MDD={best_sh['mdd']:>+.1f}%)")

    # 3. Does multi-strategy beat single champion?
    champ_ann = champ_standalone['ann']
    multi_beats = {k: v for k, v in all_approach_results.items()
                   if k != 'S1 STANDALONE' and v['ann'] > champ_ann}
    print(f"\n  3. DOES MULTI-STRATEGY BEAT SINGLE CHAMPION (+{champ_ann:.1f}%)?")
    if multi_beats:
        for name, r in sorted(multi_beats.items(), key=lambda x: -x[1]['ann']):
            diff = r['ann'] - champ_ann
            print(f"     YES - {name}: {r['ann']:>+.1f}% (+{diff:.1f}pp above champion)")
    else:
        best_multi = max((v for k, v in all_approach_results.items() if k != 'S1 STANDALONE'),
                         key=lambda x: x['ann'])
        best_multi_name = [k for k, v in all_approach_results.items()
                          if k != 'S1 STANDALONE' and v['ann'] == best_multi['ann']][0]
        gap = champ_ann - best_multi['ann']
        print(f"     NO - Best multi ({best_multi_name}): {best_multi['ann']:>+.1f}% "
              f"({gap:.1f}pp below champion)")
        # But check Sharpe
        if best_multi.get('sharpe', 0) > champ_standalone.get('sharpe', 0):
            sh_diff = best_multi['sharpe'] - champ_standalone['sharpe']
            print(f"     However, Sharpe improvement: {best_multi.get('sharpe',0):.2f} vs {champ_standalone.get('sharpe',0):.2f} (+{sh_diff:.2f})")

    # 4. Signal overlap summary
    print(f"\n  4. SIGNAL OVERLAP SUMMARY:")
    total_signals = sum(int(np.sum(strategies[k]['sig'])) for k in strat_keys)
    multi_agree = sum(overlap_counts[n] for n in range(2, 6))
    print(f"     Total signals across all 5 strategies: {total_signals:,}")
    print(f"     Commodity-days with 2+ strategies agreeing: {multi_agree:,} ({multi_agree/max(1,total_signals)*100:.1f}% of signals)")
    print(f"     Commodity-days with 3+ strategies agreeing: {sum(overlap_counts[n] for n in range(3,6)):,}")

    # 5. Optimal allocation weights
    print(f"\n  5. OPTIMAL ALLOCATION WEIGHTS:")
    # Based on standalone performance, compute risk-adjusted scores
    strat_scores = {}
    for key in strat_keys:
        r = standalone[key]
        # Score = return / abs(MDD) if both positive
        if r['ann'] > 0 and abs(r['mdd']) > 0.1:
            strat_scores[key] = r['ann'] / abs(r['mdd'])
        else:
            strat_scores[key] = 0
    total_score = sum(strat_scores.values())
    if total_score > 0:
        print(f"     Risk-adjusted allocation (return/|MDD|):")
        for key in strat_keys:
            w = strat_scores[key] / total_score
            print(f"       {strategies[key]['label']:<35}: {w*100:.1f}%")
    else:
        print(f"     No valid risk-adjusted allocation (all negative returns)")

    # Also Sharpe-based allocation
    sharpe_scores = {}
    for key in strat_keys:
        r = standalone[key]
        sharpe_scores[key] = max(0.01, r.get('sharpe', 0))
    total_sh = sum(sharpe_scores.values())
    if total_sh > 0:
        print(f"\n     Sharpe-based allocation:")
        for key in strat_keys:
            w = sharpe_scores[key] / total_sh
            print(f"       {strategies[key]['label']:<35}: {w*100:.1f}%")

    # ================================================================
    # OVERALL VERDICT
    # ================================================================
    print(f"\n{'=' * 180}")
    print("  OVERALL VERDICT")
    print(f"{'=' * 180}")

    # Find best WF performer
    best_wf_name = max(wf_summary.keys(), key=lambda k: wf_summary[k]['avg_ann'])
    best_wf = wf_summary[best_wf_name]
    print(f"\n  Best walk-forward approach: {best_wf_name}")
    print(f"    WF Avg Annual: {best_wf['avg_ann']:>+.1f}%")
    print(f"    WF Positive years: {best_wf['pos']}/6")
    print(f"    WF Avg MDD: {best_wf['avg_mdd']:>+.1f}%")
    print(f"    WF Avg Sharpe: {best_wf['avg_sh']:.2f}")

    # Most robust (best combination of return, MDD, and WF consistency)
    robustness_score = {}
    for ap_name in all_approach_results:
        r = all_approach_results[ap_name]
        ws = wf_summary.get(ap_name, {})
        score = (ws.get('avg_ann', 0) * 0.4 +
                 ws.get('pos', 0) / 6 * 100 * 0.3 +
                 r.get('sharpe', 0) * 10 * 0.2 +
                 (1 - abs(ws.get('avg_mdd', 0)) / 100) * 50 * 0.1)
        robustness_score[ap_name] = score

    most_robust = max(robustness_score.keys(), key=lambda k: robustness_score[k])
    print(f"\n  Most robust approach: {most_robust} (score={robustness_score[most_robust]:.1f})")

    # Key insight
    print(f"\n  KEY INSIGHT:")
    champ_wf_avg = wf_summary.get('S1 STANDALONE', {}).get('avg_ann', 0)
    best_multi_wf = max((wf_summary[k]['avg_ann'] for k in wf_summary if k != 'S1 STANDALONE'),
                        default=0)
    if best_multi_wf > champ_wf_avg:
        print(f"    Multi-strategy DOES improve walk-forward performance:")
        print(f"    Best multi WF avg ({best_multi_wf:+.1f}%) vs Champion WF avg ({champ_wf_avg:+.1f}%)")
    else:
        print(f"    Champion standalone remains superior in walk-forward:")
        print(f"    Champion WF avg ({champ_wf_avg:+.1f}%) vs Best multi WF avg ({best_multi_wf:+.1f}%)")
        # Check if any multi approach has better Sharpe
        champ_sh = champ_standalone.get('sharpe', 0)
        for ap_name in approach_funcs:
            r = all_approach_results[ap_name]
            if r.get('sharpe', 0) > champ_sh:
                print(f"    BUT {ap_name} has better Sharpe ({r['sharpe']:.2f} vs {champ_sh:.2f})")

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed: {elapsed:.1f}s")
    print("=" * 180)


if __name__ == '__main__':
    main()
