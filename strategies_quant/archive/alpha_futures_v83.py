"""
Alpha Futures V83 — Hold Period x Lookback Sweep
=================================================
V74 champion: +2185% with LB=1, 1-day hold. But we've ONLY tested 1-day hold with LB=1.
Maybe different lookbacks work better with different hold periods?

Sweep: 4 LB x 4 hold x 3 thresh x 2 top_n = 96 configs
Walk-forward: 6 windows (2020-2025) for top 10 configs.

Key question: Is 1-day hold really optimal? What LB x hold combo is best?
"""
import sys, os, time, warnings
import numpy as np
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

# Extended group map (same as V74 champion) — 44 commodities in 8 groups
GROUP_MAP = {}
# Ferrous
for s in ['rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi']:
    GROUP_MAP[s] = 'ferrous'
# Nonferrous
for s in ['cufi', 'alfi', 'znfi', 'nifi', 'pbfi', 'snfi', 'ssfi', 'sffi']:
    GROUP_MAP[s] = 'nonferrous'
# Precious
for s in ['aufi', 'agfi']:
    GROUP_MAP[s] = 'precious'
# Oils
for s in ['afi', 'mfi', 'yfi', 'pfi', 'cfi', 'csfi', 'rrfi', 'lrfi']:
    GROUP_MAP[s] = 'oils'
# Energy
for s in ['scfi', 'mafi', 'bfi', 'fufi', 'pgfi', 'ebfi', 'fbfi']:
    GROUP_MAP[s] = 'energy'
# Chemical
for s in ['ppfi', 'vfi', 'egfi', 'srfi', 'tafi', 'fgfi', 'lfi']:
    GROUP_MAP[s] = 'chemical'
# Soft
for s in ['whfi', 'apfi', 'cjfi', 'oifi', 'rmfi', 'srfi', 'cffi']:
    GROUP_MAP[s] = 'soft'
# Livestock
for s in ['jdfi', 'lhfi', 'pkfi']:
    GROUP_MAP[s] = 'livestock'


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def main():
    print("=" * 100)
    print("Alpha Futures V83 -- Hold Period x Lookback Sweep")
    print("=" * 100)

    # Load data
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ================================================================
    # PRECOMPUTE MOMENTUM for each lookback
    # ================================================================
    print("\n[Signals] Computing momentum...", flush=True)
    t0 = time.time()

    all_lbs = [1, 2, 3, 5]
    mom = {}
    for lag in all_lbs:
        m = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(lag, ND):
                cn = C[si, di]
                cp = C[si, di - lag]
                if not np.isnan(cn) and not np.isnan(cp) and cp > 0:
                    m[si, di] = (cn - cp) / cp
        mom[lag] = m

    # Compute group momentum (leave-one-out average)
    gm_map = {}
    for si in range(NS):
        g = GROUP_MAP.get(syms[si])
        if g:
            gm_map.setdefault(g, []).append(si)

    grp = {}
    for lag in all_lbs:
        gm = np.full((NS, ND), np.nan)
        for grp_name, members in gm_map.items():
            for di in range(lag, ND):
                for sj in members:
                    vals = [mom[lag][sk, di] for sk in members
                            if sk != sj and not np.isnan(mom[lag][sk, di])]
                    if vals:
                        gm[sj, di] = np.mean(vals)
        grp[lag] = gm

    # Tradeable si indices (those with a group)
    trade_sis = [si for si in range(NS) if GROUP_MAP.get(syms[si])]
    print(f"  Momentum computed ({time.time()-t0:.1f}s)")
    print(f"  Groups: {len(gm_map)} groups, {sum(len(v) for v in gm_map.values())} total members")
    print(f"  Tradeable commodities: {len(trade_sis)}")

    # ================================================================
    # BACKTEST ENGINE with multi-day hold
    # ================================================================
    def run_backtest(lb, hold, threshold, top_n, wf_test_year=None):
        """
        lb: lookback days for momentum signal
        hold: hold period in days (1=exit next day, 2=exit 2 days later, etc.)
        threshold: minimum divergence to trade
        top_n: max concurrent positions
        wf_test_year: if set, run walk-forward for this year
        """
        # Date range
        start_di = MIN_TRAIN
        wf_mode = wf_test_year is not None
        if wf_mode:
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
            end_di = test_end_di
        else:
            test_start_di = start_di
            end_di = ND

        cash = float(CASH0)
        positions = []  # list of dicts
        trades = []

        for di in range(start_di, end_di):
            # Reset cash at start of test window (WF only)
            if wf_mode and di == test_start_di:
                cash = float(CASH0)
                positions = []

            # --- Close positions that have been held for 'hold' days ---
            closed = []
            for pos in positions:
                if di - pos['entry_di'] >= hold:
                    cn = C[pos['si'], di]
                    if np.isnan(cn) or cn <= 0:
                        cn = pos['entry']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    mkt_val = cn * mult * pos['lots']
                    cash += mkt_val - mkt_val * COMM
                    pnl = (cn - pos['entry']) * mult * pos['lots'] * pos['dir']
                    invested = pos['entry'] * mult * pos['lots']
                    pnl_pct = pnl / invested * 100 if invested > 0 else 0
                    trades.append({
                        'pnl_pct': pnl_pct,
                        'entry_di': pos['entry_di'],
                        'exit_di': di,
                        'year': dates[di].year if di < ND else dates[-1].year,
                        'dir': pos['dir'],
                        'sym': pos['sym'],
                    })
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            # --- Score candidates (long only: buy laggards) ---
            candidates = []
            for si in trade_sis:
                sym = syms[si]
                if np.isnan(C[si, di]) or C[si, di] <= 0:
                    continue
                # Skip if already holding this commodity
                if any(p['si'] == si for p in positions):
                    continue

                own = mom[lb][si, di]
                grp_avg = grp[lb][si, di]
                if np.isnan(own) or np.isnan(grp_avg):
                    continue

                div = grp_avg - own  # positive = group ahead, commodity lagging
                if div > threshold:
                    candidates.append((si, div, 1))  # long

            if not candidates:
                continue

            # Sort by divergence (highest first)
            candidates.sort(key=lambda x: -x[1])

            # Open positions (up to top_n slots available)
            n_slots = top_n - len(positions)
            for si, score, direction in candidates[:n_slots]:
                c = C[si, di]
                if np.isnan(c) or c <= 0:
                    continue
                mult = MULT.get(syms[si], DEF_MULT)
                notional = c * mult
                lots = int(cash / (notional * (1 + COMM)))
                if lots <= 0:
                    continue
                cost_in = notional * lots * (1 + COMM)
                if cost_in > cash:
                    lots = int(cash * 0.95 / (notional * (1 + COMM)))
                    cost_in = notional * lots * (1 + COMM) if lots > 0 else 0
                if lots <= 0 or cost_in <= 0 or cost_in > cash:
                    continue

                cash -= cost_in
                positions.append({
                    'si': si, 'entry': c, 'entry_di': di,
                    'lots': lots, 'dir': direction, 'sym': syms[si],
                })

        # Close remaining positions at end
        for pos in positions:
            ae = end_di - 1
            cn = C[pos['si'], ae]
            if np.isnan(cn) or cn <= 0:
                cn = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = cn * mult * pos['lots']
            cash += mkt_val - mkt_val * COMM

        # Calculate results
        if wf_mode:
            n_days_test = test_end_di - test_start_di
        else:
            n_days_test = ND - start_di

        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)

        return {
            'ann': ann, 'wr': wr, 'n': n_trades,
            'final_cash': cash, 'n_days': n_days_test,
        }

    # ================================================================
    # SWEEP CONFIGURATIONS: LB x Hold x Threshold x TopN
    # ================================================================
    print("\n[Sweep] Testing LB x Hold x Threshold x TopN configurations...", flush=True)

    lookbacks = [1, 2, 3, 5]
    holds = [1, 2, 3, 5]
    thresholds = [0.003, 0.005, 0.01]
    top_ns = [1, 3]

    total = len(lookbacks) * len(holds) * len(thresholds) * len(top_ns)
    print(f"  Total configs: {total} (LB: {lookbacks}, Hold: {holds}, Thresh: {thresholds}, TopN: {top_ns})")

    results = []
    count = 0
    for lb in lookbacks:
        for hold in holds:
            for thresh in thresholds:
                for tn in top_ns:
                    count += 1
                    r = run_backtest(lb, hold, thresh, tn)
                    if r:
                        label = f"LB{lb}_H{hold}_T{thresh}_TN{tn}"
                        r['config'] = {'lb': lb, 'hold': hold, 'threshold': thresh, 'top_n': tn}
                        r['label'] = label
                        r['lb'] = lb
                        r['hold'] = hold
                        r['thresh'] = thresh
                        r['tn'] = tn
                        results.append(r)
                    if count % 20 == 0:
                        print(f"  ... {count}/{total} done ({time.time()-t0:.0f}s)", flush=True)

    results.sort(key=lambda x: -x['ann'])

    # ================================================================
    # REPORT: Top 20
    # ================================================================
    print("\n" + "=" * 100)
    print("  FULL-PERIOD RESULTS (Top 20)")
    print("=" * 100)
    print(f"  {'#':>3} | {'Label':<30} | {'Ann':>10} | {'WR':>6} | {'N':>5}")
    print("-" * 70)
    for i, r in enumerate(results[:20]):
        print(f"  {i+1:>3} | {r['label']:<30} | {r['ann']:>+9.1f}% | {r['wr']:>5.1f}% | {r['n']:>5}")

    # ================================================================
    # HEATMAP: LB x Hold (best threshold/top_n for each cell)
    # ================================================================
    print("\n" + "=" * 100)
    print("  LB x HOLD HEATMAP (best annual % for each LB x Hold pair)")
    print("=" * 100)

    # For each (lb, hold) find the best result across threshold and top_n
    heatmap = {}
    for r in results:
        key = (r['lb'], r['hold'])
        if key not in heatmap or r['ann'] > heatmap[key]['ann']:
            heatmap[key] = r

    print(f"\n  {'':>6}", end="")
    for h in holds:
        print(f" | Hold={h:>4}", end="")
    print()
    print("  " + "-" * (8 + 11 * len(holds)))
    for lb in lookbacks:
        print(f"  LB={lb:>2}", end="")
        for h in holds:
            r = heatmap.get((lb, h))
            if r:
                val = r['ann']
                # Color-code: green for best values
                marker = ""
                if val == max(hr['ann'] for hr in heatmap.values()):
                    marker = " **"
                elif val > 1000:
                    marker = " *"
                print(f" | {val:>+8.0f}%{marker}", end="")
            else:
                print(f" | {'N/A':>10}", end="")
        print()

    # Also show the best config details for each cell
    print("\n  Best config details per LB x Hold cell:")
    for lb in lookbacks:
        for h in holds:
            r = heatmap.get((lb, h))
            if r:
                print(f"    LB={lb} Hold={h}: Ann={r['ann']:>+.1f}% WR={r['wr']:.1f}% "
                      f"T={r['thresh']} TN={r['tn']} N={r['n']}")

    # ================================================================
    # HOLD PERIOD ANALYSIS
    # ================================================================
    print("\n" + "=" * 100)
    print("  HOLD PERIOD ANALYSIS")
    print("=" * 100)

    for h in holds:
        hold_results = [r for r in results if r['hold'] == h]
        hold_results.sort(key=lambda x: -x['ann'])
        if hold_results:
            best = hold_results[0]
            avg_ann = np.mean([r['ann'] for r in hold_results])
            avg_wr = np.mean([r['wr'] for r in hold_results])
            print(f"  Hold={h}: Best={best['ann']:>+.1f}% ({best['label']}), "
                  f"Avg(all configs)={avg_ann:>+.1f}%, Avg WR={avg_wr:.1f}%")

    # ================================================================
    # LOOKBACK ANALYSIS
    # ================================================================
    print("\n" + "=" * 100)
    print("  LOOKBACK ANALYSIS")
    print("=" * 100)

    for lb in lookbacks:
        lb_results = [r for r in results if r['lb'] == lb]
        lb_results.sort(key=lambda x: -x['ann'])
        if lb_results:
            best = lb_results[0]
            avg_ann = np.mean([r['ann'] for r in lb_results])
            print(f"  LB={lb}: Best={best['ann']:>+.1f}% ({best['label']}), "
                  f"Avg(all configs)={avg_ann:>+.1f}%")

    # ================================================================
    # WALK-FORWARD for Top 10
    # ================================================================
    print("\n" + "=" * 100)
    print("  WALK-FORWARD (Top 10 configs)")
    print("=" * 100)

    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]
    wf_results = []

    for i, r in enumerate(results[:10]):
        cfg = r['config']
        wf_row = {'label': r['label'], 'lb': cfg['lb'], 'hold': cfg['hold'],
                  'full_ann': r['ann'], 'windows': {}}
        for yr in wf_years:
            wr = run_backtest(cfg['lb'], cfg['hold'], cfg['threshold'], cfg['top_n'],
                              wf_test_year=yr)
            if wr:
                wf_row['windows'][yr] = wr['ann']
        wf_results.append(wf_row)

    # Print WF table
    print(f"\n  {'#':>3} | {'Config':<30} | {'Full':>8} | {'Avg':>8} |", end="")
    for yr in wf_years:
        print(f"  {yr:>7} |", end="")
    print(f"  {'Pos':>4}")
    print("-" * 140)
    for i, wf in enumerate(wf_results):
        vals = [wf['windows'].get(yr, 0) for yr in wf_years]
        avg = np.mean(vals) if vals else 0
        pos = sum(1 for v in vals if v > 0)
        print(f"  {i+1:>3} | {wf['label']:<30} | {wf['full_ann']:>+7.0f}% | {avg:>+7.0f}% |", end="")
        for v in vals:
            print(f"  {v:>+7.0f}% |", end="")
        print(f"  {pos}/6")

    # ================================================================
    # KEY QUESTION: Is 1-day hold really the best?
    # ================================================================
    print("\n" + "=" * 100)
    print("  KEY FINDINGS")
    print("=" * 100)

    # Best overall
    best_all = results[0]
    print(f"\n  Best overall: {best_all['label']} => {best_all['ann']:>+.1f}% annual")

    # Is hold=1 best?
    best_per_hold = {}
    for h in holds:
        hr = [r for r in results if r['hold'] == h]
        hr.sort(key=lambda x: -x['ann'])
        if hr:
            best_per_hold[h] = hr[0]

    print(f"\n  Best by hold period:")
    for h in holds:
        if h in best_per_hold:
            b = best_per_hold[h]
            print(f"    Hold={h}: {b['ann']:>+.1f}% ({b['label']})")

    # Is LB=1 best?
    best_per_lb = {}
    for lb in lookbacks:
        lr = [r for r in results if r['lb'] == lb]
        lr.sort(key=lambda x: -x['ann'])
        if lr:
            best_per_lb[lb] = lr[0]

    print(f"\n  Best by lookback:")
    for lb in lookbacks:
        if lb in best_per_lb:
            b = best_per_lb[lb]
            print(f"    LB={lb}: {b['ann']:>+.1f}% ({b['label']})")

    # WF avg for top 10
    print(f"\n  Walk-forward average for top 10:")
    for i, wf in enumerate(wf_results):
        vals = [wf['windows'].get(yr, 0) for yr in wf_years]
        avg = np.mean(vals)
        pos = sum(1 for v in vals if v > 0)
        print(f"    {i+1}. {wf['label']}: Avg WF={avg:>+.1f}%, {pos}/6 positive")

    print(f"\n  Total time: {time.time()-t_start:.1f}s")
    print("=" * 100)


if __name__ == '__main__':
    main()
