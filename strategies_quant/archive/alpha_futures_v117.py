"""
Alpha Futures V117 -- Universe Selection + Time-Based Optimization
==================================================================
Base signal: ROC(5) > 2% (best from V108: +89.8%).

Tests A-I:
  A) Universe filtering by liquidity (top N commodities by volume)
  B) Universe filtering by volatility band
  C) Per-commodity performance analysis
  D) Day-of-week filter
  E) Month-of-year filter
  F) Intraday / overnight return filter
  G) Hold period fine-tuning (1-20 days)
  H) Re-entry delay (cooldown after exit)
  I) Overlapping position management (sequential vs concurrent)

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
    print("Alpha Futures V117 -- Universe Selection + Time-Based Optimization")
    print("=" * 200)
    print("\n  Base signal: ROC(5) > 2% (V108 champion)")
    print("  Tests A-I: universe, time, hold, re-entry, overlap")

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

    ROC5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        ROC5[si] = talib.ROC(c, timeperiod=5)

    print(f"  ROC(5) computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # BASE SIGNAL: ROC(5) > 2%
    # ================================================================
    base_signal = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(1, ND):
            if not np.isnan(ROC5[si, di]) and ROC5[si, di] > 2.0:
                base_signal[si, di] = True
    print(f"  Base signal (ROC5>2%): {np.sum(base_signal)} signals across all commodities")

    # ================================================================
    # BACKTEST ENGINE (supports universe filter, day filter, re-entry delay)
    # ================================================================
    def run_backtest(signal_arr, hold_days=5, top_n=1, score_arr=None,
                     universe_set=None,      # set of si indices allowed
                     day_filter_set=None,    # set of di indices allowed for entry
                     reentry_delay=0,        # days before re-entering same commodity
                     sequential=False,       # if True, wait for exit before new entry
                     wf_test_year=None,
                     label=""):
        """Core backtest with next-open execution."""
        # Universe filter mask
        uni_mask = np.ones(NS, dtype=bool)
        if universe_set is not None:
            for si in range(NS):
                if si not in universe_set:
                    uni_mask[si] = False

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
        # Track last exit day per commodity for re-entry delay
        last_exit_di = {}

        for di in range(start_di, end_di - 1):
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                positions = []
                last_exit_di = {}

            # -- Close positions at max hold --------------------------
            closed = []
            for pos in positions:
                days_held = di - pos['entry_di']
                if days_held >= hold_days:
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
                        'sym': pos['sym'],
                        'si': pos['si'],
                        'days_held': days_held,
                    })
                    last_exit_di[pos['si']] = di
                    closed.append(pos)

            for pos in closed:
                positions.remove(pos)

            if sequential and len(positions) > 0:
                continue  # Wait for all positions to close before opening new

            if len(positions) >= top_n:
                continue

            # -- Day filter -------------------------------------------
            if day_filter_set is not None and di not in day_filter_set:
                continue

            # -- Generate signals -------------------------------------
            entry_di = di + 1
            if entry_di >= end_di:
                continue

            candidates = []
            for si in range(NS):
                if not uni_mask[si]:
                    continue
                if not signal_arr[si, di]:
                    continue
                if any(p['si'] == si for p in positions):
                    continue
                # Re-entry delay check
                if reentry_delay > 0 and si in last_exit_di:
                    if di - last_exit_di[si] < reentry_delay:
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

            candidates.sort(key=lambda x: -x[0])

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
                    'hold_days': hold_days,
                }
                positions.append(pos)

        # Close remaining at end
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
                'sym': pos['sym'],
                'si': pos['si'],
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
            'label': label,
        }

    # ================================================================
    # HELPER: walk-forward for a single config
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    def run_wf(signal_arr, hold_days, top_n, score_arr=None,
               universe_set=None, day_filter_set=None,
               reentry_delay=0, sequential=False, label=""):
        """Run walk-forward and return summary dict."""
        windows = {}
        for yr in wf_years:
            r = run_backtest(signal_arr, hold_days, top_n, score_arr,
                             universe_set=universe_set,
                             day_filter_set=day_filter_set,
                             reentry_delay=reentry_delay,
                             sequential=sequential,
                             wf_test_year=yr, label=label)
            if r:
                windows[yr] = r['ann']
        vals = [windows.get(yr, 0) for yr in wf_years]
        avg = np.mean(vals) if vals else 0
        pos = sum(1 for v in vals if v > 0)
        return {'windows': windows, 'avg': avg, 'pos': pos, 'vals': vals}

    # ================================================================
    # PRECOMPUTE: per-commodity stats for filtering
    # ================================================================
    print("\n[Precompute] Per-commodity liquidity and volatility...", flush=True)

    # Average daily volume
    avg_vol = np.zeros(NS)
    for si in range(NS):
        v = V[si]
        valid = v[~np.isnan(v)]
        avg_vol[si] = np.mean(valid) if len(valid) > 0 else 0

    # Average daily volatility (std of daily returns)
    avg_volatility = np.zeros(NS)
    for si in range(NS):
        c = C[si]
        valid_mask = ~np.isnan(c) & (c > 0)
        if np.sum(valid_mask) > 20:
            rets = np.diff(c[valid_mask]) / c[valid_mask][:-1]
            avg_volatility[si] = np.std(rets) * 100  # in percent

    # Liquidity ranking (top N)
    vol_rank = np.argsort(-avg_vol)  # descending

    # Day-of-week sets
    dow_sets = {}
    for dow in range(5):  # 0=Mon ... 4=Fri
        dow_sets[dow] = set()
        for di in range(ND):
            if dates[di].weekday() == dow:
                dow_sets[dow].add(di)

    # Month sets
    month_sets = {}
    for m in range(1, 13):
        month_sets[m] = set()
        for di in range(ND):
            if dates[di].month == m:
                month_sets[m].add(di)

    print(f"  Liquidity & volatility computed")
    print(f"  Day-of-week sets: {[f'{dow}:{len(dow_sets[dow])}' for dow in range(5)]}")
    print(f"  Month sets: {[f'{m}:{len(month_sets[m])}' for m in range(1, 13)]}")

    # ================================================================
    # BASELINE: ROC(5)>2%, all 68, hold 5, top_n=1
    # ================================================================
    print("\n" + "=" * 200)
    print("  BASELINE: ROC(5)>2%, all 68 commodities, hold=5, top_n=1")
    print("=" * 200)

    baseline_full = run_backtest(base_signal, hold_days=5, top_n=1, score_arr=ROC5, label="BASELINE")
    print(f"  Full period: Ann={baseline_full['ann']:+.1f}%  WR={baseline_full['wr']:.1f}%  N={baseline_full['n']}  MDD={baseline_full['mdd']:.1f}%")

    baseline_wf = run_wf(base_signal, 5, 1, ROC5, label="BASELINE")
    print(f"  Walk-forward: Avg={baseline_wf['avg']:+.1f}%  Pos={baseline_wf['pos']}/6  {[f'{v:+.1f}%' for v in baseline_wf['vals']]}")

    # ================================================================
    # A) UNIVERSE FILTERING BY LIQUIDITY
    # ================================================================
    print("\n" + "=" * 200)
    print("  A) UNIVERSE FILTERING BY LIQUIDITY: Top N most liquid commodities")
    print("=" * 200)

    liquidity_results = []
    for top_n_uni in [10, 20, 30, 40, 50, 68]:
        uni_set = set(vol_rank[:top_n_uni])
        r = run_backtest(base_signal, hold_days=5, top_n=1, score_arr=ROC5,
                         universe_set=uni_set, label=f"A_Top{top_n_uni}")
        if r:
            wf = run_wf(base_signal, 5, 1, ROC5, universe_set=uni_set, label=f"A_Top{top_n_uni}")
            r['wf_avg'] = wf['avg']
            r['wf_pos'] = wf['pos']
            r['wf_vals'] = wf['vals']
            liquidity_results.append(r)
            print(f"  Top {top_n_uni:>2} liquid: Ann={r['ann']:+8.1f}%  WR={r['wr']:5.1f}%  N={r['n']:>4}  MDD={r['mdd']:6.1f}%  WF_Avg={wf['avg']:+7.1f}%  WF_Pos={wf['pos']}/6")

    # ================================================================
    # B) UNIVERSE FILTERING BY VOLATILITY
    # ================================================================
    print("\n" + "=" * 200)
    print("  B) UNIVERSE FILTERING BY VOLATILITY BAND")
    print("=" * 200)

    vol_bands = [
        ("0.5-2%", 0.5, 2.0),
        ("1.0-3%", 1.0, 3.0),
        ("2.0-5%", 2.0, 5.0),
        ("0.5-3%", 0.5, 3.0),
        ("1.0-5%", 1.0, 5.0),
    ]
    vol_results = []
    for name, lo, hi in vol_bands:
        uni_set = set(si for si in range(NS) if lo <= avg_volatility[si] <= hi)
        if len(uni_set) < 5:
            print(f"  {name}: only {len(uni_set)} commodities in range, skipping")
            continue
        r = run_backtest(base_signal, hold_days=5, top_n=1, score_arr=ROC5,
                         universe_set=uni_set, label=f"B_Vol_{name}")
        if r:
            wf = run_wf(base_signal, 5, 1, ROC5, universe_set=uni_set, label=f"B_Vol_{name}")
            r['wf_avg'] = wf['avg']
            r['wf_pos'] = wf['pos']
            r['wf_vals'] = wf['vals']
            vol_results.append(r)
            print(f"  Vol {name} ({len(uni_set):>2} comm): Ann={r['ann']:+8.1f}%  WR={r['wr']:5.1f}%  N={r['n']:>4}  MDD={r['mdd']:6.1f}%  WF_Avg={wf['avg']:+7.1f}%  WF_Pos={wf['pos']}/6")

    # ================================================================
    # C) PER-COMMODITY PERFORMANCE ANALYSIS
    # ================================================================
    print("\n" + "=" * 200)
    print("  C) PER-COMMODITY PERFORMANCE ANALYSIS")
    print("=" * 200)

    comm_results = []
    for si in range(NS):
        # Create signal for single commodity
        sig_single = np.zeros((NS, ND), dtype=bool)
        sig_single[si, :] = base_signal[si, :]
        uni_set = {si}
        r = run_backtest(sig_single, hold_days=5, top_n=1, score_arr=ROC5,
                         universe_set=uni_set, label=f"C_{syms[si]}")
        if r:
            r['si'] = si
            r['sym'] = syms[si]
            comm_results.append(r)

    comm_results.sort(key=lambda x: -x['ann'])

    print(f"\n  {'#':>3} | {'Sym':<8} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7}")
    print("-" * 80)
    for i, r in enumerate(comm_results):
        print(f"  {i+1:>3} | {r['sym']:<8} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}%")

    # Champion universe: only profitable commodities
    profitable_sis = set(r['si'] for r in comm_results if r['ann'] > 0)
    top20_sis = set(r['si'] for r in comm_results[:20])
    top30_sis = set(r['si'] for r in comm_results[:30])

    print(f"\n  Profitable commodities: {len(profitable_sis)}/{NS}")
    print(f"  Unprofitable: {[syms[r['si']] for r in comm_results if r['ann'] <= 0]}")

    # Test champion universes
    for name, uset in [("Champion_Profitable", profitable_sis), ("Champion_Top20", top20_sis), ("Champion_Top30", top30_sis)]:
        r = run_backtest(base_signal, hold_days=5, top_n=1, score_arr=ROC5,
                         universe_set=uset, label=f"C_{name}")
        if r:
            wf = run_wf(base_signal, 5, 1, ROC5, universe_set=uset, label=f"C_{name}")
            print(f"  {name} ({len(uset)} comm): Ann={r['ann']:+8.1f}%  WR={r['wr']:5.1f}%  N={r['n']:>4}  MDD={r['mdd']:6.1f}%  WF_Avg={wf['avg']:+7.1f}%  WF_Pos={wf['pos']}/6")

    # ================================================================
    # D) DAY-OF-WEEK ANALYSIS
    # ================================================================
    print("\n" + "=" * 200)
    print("  D) DAY-OF-WEEK FILTER")
    print("=" * 200)

    dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    dow_results = {}

    # First: run per-day stats
    for dow in range(5):
        r = run_backtest(base_signal, hold_days=5, top_n=1, score_arr=ROC5,
                         day_filter_set=dow_sets[dow], label=f"D_{dow_names[dow]}")
        if r:
            wf = run_wf(base_signal, 5, 1, ROC5, day_filter_set=dow_sets[dow], label=f"D_{dow_names[dow]}")
            r['wf_avg'] = wf['avg']
            r['wf_pos'] = wf['pos']
            r['wf_vals'] = wf['vals']
            dow_results[dow] = r
            print(f"  {dow_names[dow]:>3} only: Ann={r['ann']:+8.1f}%  WR={r['wr']:5.1f}%  N={r['n']:>4}  MDD={r['mdd']:6.1f}%  WF_Avg={wf['avg']:+7.1f}%  WF_Pos={wf['pos']}/6")

    # Now test: skip certain days (combine remaining)
    print(f"\n  SKIP-DAY TESTS:")
    for skip_dow in range(5):
        keep_set = set()
        for di in range(ND):
            if dates[di].weekday() != skip_dow:
                keep_set.add(di)
        r = run_backtest(base_signal, hold_days=5, top_n=1, score_arr=ROC5,
                         day_filter_set=keep_set, label=f"D_Skip_{dow_names[skip_dow]}")
        if r:
            wf = run_wf(base_signal, 5, 1, ROC5, day_filter_set=keep_set, label=f"D_Skip_{dow_names[skip_dow]}")
            print(f"  Skip {dow_names[skip_dow]:>3}: Ann={r['ann']:+8.1f}%  WR={r['wr']:5.1f}%  N={r['n']:>4}  MDD={r['mdd']:6.1f}%  WF_Avg={wf['avg']:+7.1f}%  WF_Pos={wf['pos']}/6")

    # Best combo: only best 2 days
    sorted_dow = sorted(dow_results.keys(), key=lambda k: -dow_results[k]['ann'])
    best2_set = dow_sets[sorted_dow[0]] | dow_sets[sorted_dow[1]]
    r = run_backtest(base_signal, hold_days=5, top_n=1, score_arr=ROC5,
                     day_filter_set=best2_set, label=f"D_Best2")
    if r:
        wf = run_wf(base_signal, 5, 1, ROC5, day_filter_set=best2_set, label=f"D_Best2")
        print(f"  Best 2 days ({dow_names[sorted_dow[0]]}+{dow_names[sorted_dow[1]]}): Ann={r['ann']:+8.1f}%  WR={r['wr']:5.1f}%  N={r['n']:>4}  WF_Avg={wf['avg']:+7.1f}%  WF_Pos={wf['pos']}/6")

    # ================================================================
    # E) MONTH-OF-YEAR ANALYSIS
    # ================================================================
    print("\n" + "=" * 200)
    print("  E) MONTH-OF-YEAR ANALYSIS")
    print("=" * 200)

    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

    # Per-month performance (only trade in that month)
    for m in range(1, 13):
        r = run_backtest(base_signal, hold_days=5, top_n=1, score_arr=ROC5,
                         day_filter_set=month_sets[m], label=f"E_{month_names[m]}")
        if r:
            wf = run_wf(base_signal, 5, 1, ROC5, day_filter_set=month_sets[m], label=f"E_{month_names[m]}")
            print(f"  {month_names[m]:>3} only: Ann={r['ann']:+8.1f}%  WR={r['wr']:5.1f}%  N={r['n']:>4}  MDD={r['mdd']:6.1f}%  WF_Avg={wf['avg']:+7.1f}%")

    # Skip months
    print(f"\n  SKIP-MONTH TESTS:")
    for skip_m in range(1, 13):
        keep_set = set()
        for di in range(ND):
            if dates[di].month != skip_m:
                keep_set.add(di)
        r = run_backtest(base_signal, hold_days=5, top_n=1, score_arr=ROC5,
                         day_filter_set=keep_set, label=f"E_Skip_{month_names[skip_m]}")
        if r:
            wf = run_wf(base_signal, 5, 1, ROC5, day_filter_set=keep_set, label=f"E_Skip_{month_names[skip_m]}")
            print(f"  Skip {month_names[skip_m]:>3}: Ann={r['ann']:+8.1f}%  WR={r['wr']:5.1f}%  N={r['n']:>4}  MDD={r['mdd']:6.1f}%  WF_Avg={wf['avg']:+7.1f}%")

    # ================================================================
    # F) INTRADAY RETURN FILTER
    # ================================================================
    print("\n" + "=" * 200)
    print("  F) INTRADAY / OVERNIGHT RETURN FILTER")
    print("=" * 200)

    # Precompute intraday and overnight returns
    intraday_ret = np.full((NS, ND), np.nan)
    overnight_ret = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            o = O[si, di]
            c = C[si, di]
            c_prev = C[si, di - 1]
            if not np.isnan(o) and not np.isnan(c) and o > 0:
                intraday_ret[si, di] = (c - o) / o * 100
            if not np.isnan(o) and not np.isnan(c_prev) and c_prev > 0:
                overnight_ret[si, di] = (o - c_prev) / c_prev * 100

    # F1: ROC(5)>2% AND intraday return > 0 (close above open)
    sig_F1 = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(ND):
            if base_signal[si, di]:
                if not np.isnan(intraday_ret[si, di]) and intraday_ret[si, di] > 0:
                    sig_F1[si, di] = True
    print(f"  F1) ROC5>2% AND intraday>0: {np.sum(sig_F1)} signals", flush=True)

    # F2: ROC(5)>2% AND overnight return > 0 (positive gap)
    sig_F2 = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(ND):
            if base_signal[si, di]:
                if not np.isnan(overnight_ret[si, di]) and overnight_ret[si, di] > 0:
                    sig_F2[si, di] = True
    print(f"  F2) ROC5>2% AND overnight>0: {np.sum(sig_F2)} signals", flush=True)

    # F3: ROC(5)>2% AND intraday > 0 AND overnight > 0 (both bullish)
    sig_F3 = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(ND):
            if base_signal[si, di]:
                id_ok = not np.isnan(intraday_ret[si, di]) and intraday_ret[si, di] > 0
                on_ok = not np.isnan(overnight_ret[si, di]) and overnight_ret[si, di] > 0
                if id_ok and on_ok:
                    sig_F3[si, di] = True
    print(f"  F3) ROC5>2% AND intraday>0 AND overnight>0: {np.sum(sig_F3)} signals", flush=True)

    # F4: ROC(5)>2% AND intraday < 0 (bearish intraday, reversal)
    sig_F4 = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(ND):
            if base_signal[si, di]:
                if not np.isnan(intraday_ret[si, di]) and intraday_ret[si, di] < 0:
                    sig_F4[si, di] = True
    print(f"  F4) ROC5>2% AND intraday<0 (reversal): {np.sum(sig_F4)} signals", flush=True)

    for sig, lab in [(sig_F1, "F1_Intraday>0"), (sig_F2, "F2_Overnight>0"),
                     (sig_F3, "F3_Both>0"), (sig_F4, "F4_Intraday<0")]:
        r = run_backtest(sig, hold_days=5, top_n=1, score_arr=ROC5, label=lab)
        if r:
            wf = run_wf(sig, 5, 1, ROC5, label=lab)
            print(f"  {lab:<25}: Ann={r['ann']:+8.1f}%  WR={r['wr']:5.1f}%  N={r['n']:>4}  MDD={r['mdd']:6.1f}%  WF_Avg={wf['avg']:+7.1f}%  WF_Pos={wf['pos']}/6")

    # ================================================================
    # G) HOLD PERIOD FINE-TUNING (1-20 days)
    # ================================================================
    print("\n" + "=" * 200)
    print("  G) HOLD PERIOD FINE-TUNING (1-20 days)")
    print("=" * 200)

    hold_results = []
    for hd in range(1, 21):
        r = run_backtest(base_signal, hold_days=hd, top_n=1, score_arr=ROC5, label=f"G_Hold{hd}")
        if r:
            wf = run_wf(base_signal, hd, 1, ROC5, label=f"G_Hold{hd}")
            r['wf_avg'] = wf['avg']
            r['wf_pos'] = wf['pos']
            r['wf_vals'] = wf['vals']
            hold_results.append(r)
            print(f"  Hold {hd:>2}d: Ann={r['ann']:+8.1f}%  WR={r['wr']:5.1f}%  N={r['n']:>4}  AvgHold={r['avg_hold']:5.1f}d  MDD={r['mdd']:6.1f}%  WF_Avg={wf['avg']:+7.1f}%  WF_Pos={wf['pos']}/6")

    # Best hold period
    if hold_results:
        best_hold = max(hold_results, key=lambda x: x['ann'])
        print(f"\n  BEST HOLD: {best_hold['label']} -> {best_hold['ann']:+.1f}% annual")

    # ================================================================
    # H) RE-ENTRY DELAY
    # ================================================================
    print("\n" + "=" * 200)
    print("  H) RE-ENTRY DELAY (cooldown after exit)")
    print("=" * 200)

    for delay in [0, 1, 2, 3, 5]:
        r = run_backtest(base_signal, hold_days=5, top_n=1, score_arr=ROC5,
                         reentry_delay=delay, label=f"H_Delay{delay}")
        if r:
            wf = run_wf(base_signal, 5, 1, ROC5, reentry_delay=delay, label=f"H_Delay{delay}")
            print(f"  Delay {delay}d: Ann={r['ann']:+8.1f}%  WR={r['wr']:5.1f}%  N={r['n']:>4}  MDD={r['mdd']:6.1f}%  WF_Avg={wf['avg']:+7.1f}%  WF_Pos={wf['pos']}/6")

    # ================================================================
    # I) OVERLAPPING POSITION MANAGEMENT
    # ================================================================
    print("\n" + "=" * 200)
    print("  I) OVERLAPPING POSITION MANAGEMENT")
    print("=" * 200)

    # I1: Concurrent (baseline) -- top_n=1 but already holding, skip same commodity
    # This is the default behavior (already tested as baseline)

    # I2: Sequential -- wait for exit before new entry (any commodity)
    r_seq = run_backtest(base_signal, hold_days=5, top_n=1, score_arr=ROC5,
                         sequential=True, label="I_Sequential")
    if r_seq:
        wf_seq = run_wf(base_signal, 5, 1, ROC5, sequential=True, label="I_Sequential")
        print(f"  Sequential (1 pos): Ann={r_seq['ann']:+8.1f}%  WR={r_seq['wr']:5.1f}%  N={r_seq['n']:>4}  MDD={r_seq['mdd']:6.1f}%  WF_Avg={wf_seq['avg']:+7.1f}%  WF_Pos={wf_seq['pos']}/6")

    # I3: Concurrent top_n=3
    r_c3 = run_backtest(base_signal, hold_days=5, top_n=3, score_arr=ROC5, label="I_Conc3")
    if r_c3:
        wf_c3 = run_wf(base_signal, 5, 3, ROC5, label="I_Conc3")
        print(f"  Concurrent (3 pos):  Ann={r_c3['ann']:+8.1f}%  WR={r_c3['wr']:5.1f}%  N={r_c3['n']:>4}  MDD={r_c3['mdd']:6.1f}%  WF_Avg={wf_c3['avg']:+7.1f}%  WF_Pos={wf_c3['pos']}/6")

    # I4: Concurrent top_n=5
    r_c5 = run_backtest(base_signal, hold_days=5, top_n=5, score_arr=ROC5, label="I_Conc5")
    if r_c5:
        wf_c5 = run_wf(base_signal, 5, 5, ROC5, label="I_Conc5")
        print(f"  Concurrent (5 pos):  Ann={r_c5['ann']:+8.1f}%  WR={r_c5['wr']:5.1f}%  N={r_c5['n']:>4}  MDD={r_c5['mdd']:6.1f}%  WF_Avg={wf_c5['avg']:+7.1f}%  WF_Pos={wf_c5['pos']}/6")

    # I5: Concurrent top_n=3, sequential=False (allow overlapping same commodity? No, already blocked)
    # I6: Different approach -- allow re-entry for same commodity when already in position (skip is default)
    # Let's test: top_n=3 with different hold periods
    for hd in [3, 5, 7]:
        r = run_backtest(base_signal, hold_days=hd, top_n=3, score_arr=ROC5, label=f"I_Conc3_H{hd}")
        if r:
            wf = run_wf(base_signal, hd, 3, ROC5, label=f"I_Conc3_H{hd}")
            print(f"  Conc3 Hold{hd}d: Ann={r['ann']:+8.1f}%  WR={r['wr']:5.1f}%  N={r['n']:>4}  MDD={r['mdd']:6.1f}%  WF_Avg={wf['avg']:+7.1f}%  WF_Pos={wf['pos']}/6")

    # ================================================================
    # COMBINED BEST CONFIG
    # ================================================================
    print("\n" + "=" * 200)
    print("  COMBINED BEST CONFIGS: Testing top universe + best hold + best filters")
    print("=" * 200)

    # Determine best universe, hold, and test combos
    # Use champion universe from C + best hold from G
    best_hold_val = 5
    if hold_results:
        best_hold_val = int(max(hold_results, key=lambda x: x['ann'])['label'].replace('G_Hold', ''))

    combos = []

    # Combo 1: Champion profitable + best hold
    combos.append(("Combo_Profitable_HoldBest", profitable_sis, best_hold_val, None))

    # Combo 2: Top 20 champion + best hold
    combos.append(("Combo_Top20_HoldBest", top20_sis, best_hold_val, None))

    # Combo 3: Champion profitable + best hold + skip worst day
    if sorted_dow:
        skip_worst = sorted_dow[-1]
        keep_days = set()
        for di in range(ND):
            if dates[di].weekday() != skip_worst:
                keep_days.add(di)
        combos.append((f"Combo_Prof_HoldBest_Skip{dow_names[skip_worst]}", profitable_sis, best_hold_val, keep_days))

    # Combo 4: Top 20 + best hold + re-entry delay 1
    combos.append(("Combo_Top20_HoldBest_D1", top20_sis, best_hold_val, None))

    # Combo 5: Top 20 + best hold + concurrent 3
    combos.append(("Combo_Top20_Conc3", top20_sis, best_hold_val, None))

    for name, uset, hd, dfilter in combos:
        if "Conc3" in name:
            tn = 3
        else:
            tn = 1
        r = run_backtest(base_signal, hold_days=hd, top_n=tn, score_arr=ROC5,
                         universe_set=uset, day_filter_set=dfilter,
                         reentry_delay=1 if "D1" in name else 0,
                         label=name)
        if r:
            wf = run_wf(base_signal, hd, tn, ROC5,
                         universe_set=uset, day_filter_set=dfilter,
                         reentry_delay=1 if "D1" in name else 0,
                         label=name)
            print(f"  {name:<35}: Ann={r['ann']:+8.1f}%  WR={r['wr']:5.1f}%  N={r['n']:>4}  MDD={r['mdd']:6.1f}%  WF_Avg={wf['avg']:+7.1f}%  WF_Pos={wf['pos']}/6  Years={[f'{v:+.0f}%' for v in wf['vals']]}")

    # ================================================================
    # WALK-FORWARD COMPARISON TABLE
    # ================================================================
    print(f"\n{'=' * 220}")
    print("  WALK-FORWARD COMPARISON: Key configurations (2020-2025)")
    print(f"{'=' * 220}")

    # Collect key configs for WF comparison
    key_configs = [
        ("BASELINE (all 68, H5, TN1)", base_signal, 5, 1, None, None, 0, False),
        ("BASELINE (all 68, H5, TN3)", base_signal, 5, 3, None, None, 0, False),
    ]

    # Add best liquidity
    if liquidity_results:
        best_liq = max(liquidity_results, key=lambda x: x.get('wf_avg', x['ann']))
        top_n_uni = int(best_liq['label'].replace('A_Top', ''))
        best_liq_set = set(vol_rank[:top_n_uni])
        key_configs.append((f"Best Liquidity (Top{top_n_uni})", base_signal, 5, 1, best_liq_set, None, 0, False))

    # Add champion universe
    key_configs.append(("Champion Profitable", base_signal, 5, 1, profitable_sis, None, 0, False))
    key_configs.append(("Champion Top20", base_signal, 5, 1, top20_sis, None, 0, False))

    # Add best hold
    if hold_results:
        bh = max(hold_results, key=lambda x: x.get('wf_avg', x['ann']))
        bh_val = int(bh['label'].replace('G_Hold', ''))
        key_configs.append((f"Best Hold ({bh_val}d)", base_signal, bh_val, 1, None, None, 0, False))

    # Add sequential
    key_configs.append(("Sequential", base_signal, 5, 1, None, None, 0, True))
    key_configs.append(("Concurrent TN=3", base_signal, 5, 3, None, None, 0, False))

    header = f"  {'Config':<40} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>7} |"
    header += f" {'Pos':>4} |"
    print(header)
    print("-" * 220)

    for label, sig, hd, tn, uset, dfilter, delay, seq in key_configs:
        wf = run_wf(sig, hd, tn, ROC5, universe_set=uset,
                     day_filter_set=dfilter, reentry_delay=delay,
                     sequential=seq, label=label)
        vals = wf['vals']
        row = f"  {label:<40} | {wf['avg']:>+7.1f}% |"
        for v in vals:
            row += f" {v:>+7.1f}% |"
        row += f" {wf['pos']}/6 |"
        print(row)

    # ================================================================
    # FINAL REPORT
    # ================================================================
    print(f"\n{'=' * 200}")
    print("  FINAL REPORT: V117 Universe Selection + Time-Based Optimization")
    print(f"{'=' * 200}")

    print(f"\n  BASELINE (V108 ROC5>2%): Ann={baseline_full['ann']:+.1f}%  N={baseline_full['n']}  WR={baseline_full['wr']:.1f}%  MDD={baseline_full['mdd']:.1f}%")
    print(f"  BASELINE WF: Avg={baseline_wf['avg']:+.1f}%  Pos={baseline_wf['pos']}/6")

    print(f"\n  1) BEST UNIVERSE:")
    if liquidity_results:
        best_liq_full = max(liquidity_results, key=lambda x: x['ann'])
        best_liq_wf = max(liquidity_results, key=lambda x: x.get('wf_avg', -999))
        print(f"     By full-period: {best_liq_full['label']} = {best_liq_full['ann']:+.1f}%")
        print(f"     By WF average:  {best_liq_wf['label']} = {best_liq_wf.get('wf_avg', 0):+.1f}%")
    if comm_results:
        top5 = [f"{r['sym']}({r['ann']:+.0f}%)" for r in comm_results[:5]]
        bottom5 = [f"{r['sym']}({r['ann']:+.0f}%)" for r in comm_results[-5:]]
        print(f"     Top 5 commodities: {', '.join(top5)}")
        print(f"     Bottom 5: {', '.join(bottom5)}")

    print(f"\n  2) BEST DAY-OF-WEEK FILTER:")
    if dow_results:
        best_dow = max(dow_results.values(), key=lambda x: x['ann'])
        print(f"     Best single day: {best_dow['label']} = {best_dow['ann']:+.1f}%")

    print(f"\n  3) OPTIMAL HOLD PERIOD:")
    if hold_results:
        best_h = max(hold_results, key=lambda x: x['ann'])
        best_h_wf = max(hold_results, key=lambda x: x.get('wf_avg', -999))
        print(f"     By full-period: {best_h['label']} = {best_h['ann']:+.1f}%")
        print(f"     By WF average:  {best_h_wf['label']} = {best_h_wf.get('wf_avg', 0):+.1f}%")
        print(f"     Hold period ranking:")
        for r in sorted(hold_results, key=lambda x: -x['ann']):
            print(f"       {r['label']}: {r['ann']:+.1f}% (WF={r.get('wf_avg', 0):+.1f}%)")

    print(f"\n  4) PER-COMMODITY PROFITABILITY:")
    if comm_results:
        n_pos = sum(1 for r in comm_results if r['ann'] > 0)
        n_neg = sum(1 for r in comm_results if r['ann'] <= 0)
        print(f"     Profitable: {n_pos}/{NS}  |  Unprofitable: {n_neg}/{NS}")
        print(f"     Top 10: {', '.join([r['sym'] for r in comm_results[:10]])}")
        print(f"     Skip: {', '.join([r['sym'] for r in comm_results if r['ann'] <= 0])}")

    print(f"\n  5) DOES UNIVERSE FILTERING IMPROVE +89.8%?")
    baseline_val = baseline_full['ann']
    if liquidity_results:
        for r in liquidity_results:
            if r['ann'] > baseline_val:
                print(f"     YES: {r['label']} = {r['ann']:+.1f}% > baseline {baseline_val:+.1f}%")
    if comm_results:
        for uset_name, uset in [("Champion Profitable", profitable_sis), ("Champion Top20", top20_sis)]:
            r = run_backtest(base_signal, hold_days=5, top_n=1, score_arr=ROC5,
                             universe_set=uset, label=f"Final_{uset_name}")
            if r and r['ann'] > baseline_val:
                print(f"     YES: {uset_name} ({len(uset)} comm) = {r['ann']:+.1f}% > baseline {baseline_val:+.1f}%")

    print(f"\n  Total runtime: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
