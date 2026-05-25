"""
Alpha Futures V74 — Push V69 Further: Extended Groups + Multi-LB + Filters
===========================================================================
V69 at LB=1 gives +1023% annual. Can we push higher?

New ideas:
  1. Extended group map: assign ALL 68 commodities to groups (currently only 25)
  2. Multi-LB ensemble: combine LB=1,2,3 signals for each commodity
  3. Trade both directions: long laggards AND short leaders (short when own > group)
  4. Intraday range filter: skip trades on very low volatility days
  5. Confirm with OI: only trade when OI is rising (capital flowing in)

Testing approach: sweep configs, 6-window WF for top 10.
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

# Original groups (25 commodities)
GROUP_MAP_ORIG = {
    'rbfi': 'ferrous', 'hcfi': 'ferrous', 'ifi': 'ferrous', 'jfi': 'ferrous', 'jmfi': 'ferrous',
    'cufi': 'nonferrous', 'alfi': 'nonferrous', 'znfi': 'nonferrous', 'nifi': 'nonferrous',
    'aufi': 'precious', 'agfi': 'precious',
    'afi': 'oils', 'mfi': 'oils', 'yfi': 'oils', 'pfi': 'oils', 'cfi': 'oils',
    'scfi': 'energy', 'mafi': 'energy', 'bfi': 'energy', 'fufi': 'energy', 'pgfi': 'energy',
    'ppfi': 'chemical', 'vfi': 'chemical', 'egfi': 'chemical', 'srfi': 'chemical',
}

# Extended groups (all 68 commodities)
GROUP_MAP_EXT = {
    # Ferrous (黑色)
    'rbfi': 'ferrous', 'hcfi': 'ferrous', 'ifi': 'ferrous', 'jfi': 'ferrous', 'jmfi': 'ferrous',
    # Nonferrous (有色金属)
    'cufi': 'nonferrous', 'alfi': 'nonferrous', 'znfi': 'nonferrous', 'nifi': 'nonferrous',
    'pbfi': 'nonferrous', 'snfi': 'nonferrous', 'ssfi': 'nonferrous', 'sffi': 'nonferrous',
    # Precious (贵金属)
    'aufi': 'precious', 'agfi': 'precious',
    # Oils/Oilseeds (油脂油料)
    'afi': 'oils', 'mfi': 'oils', 'yfi': 'oils', 'pfi': 'oils', 'cfi': 'oils',
    'csfi': 'oils', 'rrfi': 'oils', 'lrfi': 'oils',
    # Energy (能源)
    'scfi': 'energy', 'mafi': 'energy', 'bfi': 'energy', 'fufi': 'energy', 'pgfi': 'energy',
    'ebfi': 'energy', 'fbfi': 'energy',
    # Chemical (化工)
    'ppfi': 'chemical', 'vfi': 'chemical', 'egfi': 'chemical', 'srfi': 'chemical',
    'pta_fi': 'chemical', 'mfi': 'chemical',  # mfi already in oils
    # Soft commodities (软商品)
    'srfi': 'soft', 'cffi': 'soft', 'whfi': 'soft', 'apfi': 'soft',
    'cjfi': 'soft', 'oifi': 'soft',
    # Agriculture (农产品)
    'srfi': 'agri', 'whfi': 'agri', 'cfi': 'agri',
    # Livestock (畜牧)
    'jdfi': 'livestock', 'lhfi': 'livestock', 'pkfi': 'livestock',
}

# Actually, let me use a cleaner extended group map
GROUP_MAP_FULL = {}
# Ferrous
for s in ['rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi']:
    GROUP_MAP_FULL[s] = 'ferrous'
# Nonferrous
for s in ['cufi', 'alfi', 'znfi', 'nifi', 'pbfi', 'snfi', 'ssfi', 'sffi']:
    GROUP_MAP_FULL[s] = 'nonferrous'
# Precious
for s in ['aufi', 'agfi']:
    GROUP_MAP_FULL[s] = 'precious'
# Oils
for s in ['afi', 'mfi', 'yfi', 'pfi', 'cfi', 'csfi', 'rrfi', 'lrfi']:
    GROUP_MAP_FULL[s] = 'oils'
# Energy
for s in ['scfi', 'mafi', 'bfi', 'fufi', 'pgfi', 'ebfi', 'fbfi']:
    GROUP_MAP_FULL[s] = 'energy'
# Chemical
for s in ['ppfi', 'vfi', 'egfi', 'srfi', 'tafi', 'fgfi', 'lfi']:
    GROUP_MAP_FULL[s] = 'chemical'
# Soft/Agri
for s in ['whfi', 'apfi', 'cjfi', 'oifi', 'rmfi', 'srfi', 'cffi']:
    GROUP_MAP_FULL[s] = 'soft'
# Livestock
for s in ['jdfi', 'lhfi', 'pkfi']:
    GROUP_MAP_FULL[s] = 'livestock'


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def main():
    print("=" * 98)
    print("Alpha Futures V74 -- Push V69 Further: Extended Groups + Multi-LB")
    print("=" * 98)

    # Load data
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ================================================================
    # PRECOMPUTE MOMENTUM
    # ================================================================
    print("\n[Signals] Computing momentum...", flush=True)
    t0 = time.time()

    all_lbs = [1, 2, 3]
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

    # Group momentum for multiple group maps
    def compute_grp_mom(gmap):
        # Build group members (si indices)
        gm_map = {}
        for si in range(NS):
            g = gmap.get(syms[si])
            if g:
                gm_map.setdefault(g, []).append(si)

        result = {}
        for lag in all_lbs:
            gm = np.full((NS, ND), np.nan)
            for grp, members in gm_map.items():
                for di in range(lag, ND):
                    for sj in members:
                        ms = [mom[lag][sk, di] for sk in members
                              if sk != sj and not np.isnan(mom[lag][sk, di])]
                        if ms:
                            gm[sj, di] = np.mean(ms)
            result[lag] = gm
        return result, gm_map

    grp_orig, gm_orig = compute_grp_mom(GROUP_MAP_ORIG)
    grp_full, gm_full = compute_grp_mom(GROUP_MAP_FULL)
    print(f"  Momentum computed ({time.time()-t0:.1f}s)")
    print(f"  Groups (orig): {len(gm_orig)} groups, {sum(len(v) for v in gm_orig.values())} members")
    print(f"  Groups (full): {len(gm_full)} groups, {sum(len(v) for v in gm_full.values())} members")

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(config, wf_test_year=None):
        """
        Config dict: {
            group_map: 'orig' | 'full',
            lbs: [1] | [1,2] | [1,2,3],  # lookbacks to combine
            threshold: float,
            top_n: 1 | 3,
            short_too: True | False,  # also short leaders
            comm: 0.0003 | 0.0001,
            min_range_pct: 0.0,  # skip if (H-L)/C < this
            require_oi_rise: False | True,
        }
        """
        gmap = GROUP_MAP_ORIG if config['group_map'] == 'orig' else GROUP_MAP_FULL
        grp = grp_orig if config['group_map'] == 'orig' else grp_full
        lbs = config['lbs']
        threshold = config['threshold']
        top_n = config['top_n']
        short_too = config.get('short_too', False)
        comm = config.get('comm', COMM)
        min_range = config.get('min_range_pct', 0.0)
        req_oi = config.get('require_oi_rise', False)

        # Get tradeable si indices (those with group)
        trade_sis = []
        for si in range(NS):
            if gmap.get(syms[si]):
                trade_sis.append(si)

        # Date range
        if wf_test_year:
            start_di = MIN_TRAIN
            # Find first day of test year
            test_start = None
            test_end = None
            for di in range(ND):
                if dates[di].year == wf_test_year and test_start is None:
                    test_start = di
                if dates[di].year == wf_test_year + 1 and test_end is None:
                    test_end = di
            if test_start is None:
                return None
            if test_end is None:
                test_end = ND
            # Only count results in test window
        else:
            start_di = MIN_TRAIN
            test_start = start_di
            test_end = ND

        cash = float(CASH0)
        positions = []
        trades = []

        # For WF, reset cash to CASH0 at start of test year, stop at end of test year
        wf_mode = wf_test_year is not None
        start_di = MIN_TRAIN
        if wf_mode:
            # Find test year boundaries
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

        for di in range(start_di, end_di):
            # Reset cash to CASH0 at start of test window (WF only)
            if wf_mode and di == test_start_di:
                cash = float(CASH0)
                positions = []

            # --- Close positions entered yesterday ---
            closed = []
            for pos in positions:
                if di - pos['entry_di'] >= 1:
                    cn = C[pos['si'], di]
                    if np.isnan(cn) or cn <= 0:
                        cn = pos['entry']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    mkt_val = cn * mult * pos['lots']
                    cash += mkt_val - mkt_val * comm
                    pnl = (cn - pos['entry']) * mult * pos['lots'] * pos['dir']
                    invested = pos['entry'] * mult * pos['lots']
                    pnl_pct = pnl / invested * 100 if invested > 0 else 0
                    trades.append({
                        'pnl_pct': pnl_pct, 'di': pos['entry_di'],
                        'year': dates[di].year if di < ND else dates[-1].year,
                        'dir': pos['dir'],
                    })
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            # --- Score candidates ---
            candidates = []
            for si in trade_sis:
                sym = syms[si]
                if np.isnan(C[si, di]) or C[si, di] <= 0:
                    continue
                if any(p['si'] == si for p in positions):
                    continue

                # Compute combined divergence across multiple LBs
                scores_long = []
                scores_short = []
                for lb in lbs:
                    own = mom[lb][si, di]
                    grp_avg = grp[lb][si, di]
                    if np.isnan(own) or np.isnan(grp_avg):
                        continue
                    div = grp_avg - own  # positive = group ahead, commodity lagging
                    scores_long.append(div)
                    if short_too:
                        scores_short.append(-div)

                if not scores_long:
                    continue

                # Use mean divergence across LBs
                avg_div = np.mean(scores_long)
                if avg_div > threshold:
                    # Optional: intraday range filter
                    if min_range > 0:
                        h = H[si, di]
                        l = L[si, di]
                        c = C[si, di]
                        if not np.isnan(h) and not np.isnan(l) and c > 0:
                            rng = (h - l) / c
                            if rng < min_range:
                                continue

                    # Optional: OI filter
                    if req_oi:
                        oi_now = OI[si, di]
                        oi_20 = np.nanmean(OI[si, max(0, di-20):di])
                        if np.isnan(oi_now) or np.isnan(oi_20) or oi_now < oi_20:
                            continue

                    candidates.append((si, avg_div, 1))  # long

                if short_too and scores_short:
                    avg_div_s = np.mean(scores_short)
                    if avg_div_s > threshold:
                        candidates.append((si, avg_div_s, -1))  # short

            if not candidates:
                continue

            # Sort by score (highest divergence first)
            candidates.sort(key=lambda x: -x[1])

            # Open positions
            n_slots = top_n - len(positions)
            for si, score, direction in candidates[:n_slots]:
                c = C[si, di]
                if np.isnan(c) or c <= 0:
                    continue
                mult = MULT.get(syms[si], DEF_MULT)
                notional = c * mult
                lots = int(cash / (notional * (1 + comm)))
                if lots <= 0:
                    continue
                cost_in = notional * lots * (1 + comm)
                if cost_in > cash:
                    lots = int(cash * 0.95 / (notional * (1 + comm)))
                    cost_in = notional * lots * (1 + comm) if lots > 0 else 0
                if lots <= 0 or cost_in <= 0 or cost_in > cash:
                    continue

                cash -= cost_in
                positions.append({
                    'si': si, 'entry': c, 'entry_di': di,
                    'lots': lots, 'dir': direction, 'sym': syms[si],
                })

        # Close remaining
        for pos in positions:
            ae = ND - 1
            cn = C[pos['si'], ae]
            if np.isnan(cn) or cn <= 0:
                cn = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = cn * mult * pos['lots']
            cash += mkt_val - mkt_val * comm

        # Calculate results
        if wf_mode:
            test_trades = trades
            n_days_test = test_end_di - test_start_di
            ann = annual_return(cash, CASH0, n_days_test)
        else:
            test_trades = trades
            n_days_test = ND - start_di
            ann = annual_return(cash, CASH0, n_days_test)

        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in test_trades]) * 100 if test_trades else 0
        n_trades = len(test_trades)

        return {
            'ann': ann, 'wr': wr, 'n': n_trades,
            'final_cash': cash, 'n_days': n_days_test,
        }

    # ================================================================
    # SWEEP CONFIGURATIONS
    # ================================================================
    print("\n[Sweep] Testing configurations...", flush=True)

    configs = []
    config_id = 0

    # V69 baseline variations
    for gmap in ['orig', 'full']:
        for lbs in [[1], [1, 2], [1, 2, 3]]:
            for thresh in [0.001, 0.003, 0.005, 0.01]:
                for tn in [1, 3]:
                    for short in [False, True]:
                        config_id += 1
                        configs.append({
                            'id': config_id,
                            'group_map': gmap,
                            'lbs': lbs,
                            'threshold': thresh,
                            'top_n': tn,
                            'short_too': short,
                            'comm': 0.0003,
                            'min_range_pct': 0.0,
                            'require_oi_rise': False,
                            'label': f"G{gmap[0].upper()}_LB{'_'.join(map(str,lbs))}_T{thresh}_TN{tn}_{'LS' if short else 'L'}",
                        })

    # With filters
    for gmap in ['orig', 'full']:
        for thresh in [0.003, 0.01]:
            for filt in [{'min_range_pct': 0.01, 'label': 'RNG'},
                         {'require_oi_rise': True, 'label': 'OI'},
                         {'min_range_pct': 0.01, 'require_oi_rise': True, 'label': 'BOTH'}]:
                config_id += 1
                cfg = {
                    'id': config_id,
                    'group_map': gmap,
                    'lbs': [1],
                    'threshold': thresh,
                    'top_n': 3,
                    'short_too': False,
                    'comm': 0.0003,
                    'label': f"G{gmap[0].upper()}_LB1_T{thresh}_TN3_L_{filt['label']}",
                }
                cfg.update({k: v for k, v in filt.items() if k != 'label'})
                configs.append(cfg)

    print(f"  Total configs: {len(configs)}")

    # Run all configs (full period)
    results = []
    for i, cfg in enumerate(configs):
        r = run_backtest(cfg)
        if r:
            r['config'] = cfg
            r['label'] = cfg['label']
            results.append(r)
        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(configs)} done", flush=True)

    # Sort by annual return
    results.sort(key=lambda x: -x['ann'])

    # Print top 20
    print("\n" + "=" * 98)
    print("  FULL-PERIOD RESULTS (Top 20)")
    print("=" * 98)
    print(f"  {'#':>3} | {'Label':<45} | {'Ann':>8} | {'WR':>5} | {'N':>5}")
    print("-" * 80)
    for i, r in enumerate(results[:20]):
        print(f"  {i+1:>3} | {r['label']:<45} | {r['ann']:>+7.1f}% | {r['wr']:>4.1f}% | {r['n']:>5}")

    # WF for top 10
    print("\n" + "=" * 98)
    print("  WALK-FORWARD (Top 10 configs)")
    print("=" * 98)

    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]
    wf_results = []
    for i, r in enumerate(results[:10]):
        cfg = r['config']
        wf_row = {'label': cfg['label'], 'windows': {}}
        for yr in wf_years:
            wr = run_backtest(cfg, wf_test_year=yr)
            if wr:
                wf_row['windows'][yr] = wr['ann']
        wf_results.append(wf_row)

    # Print WF table
    print(f"  {'#':>3} | {'Config':<45} | {'Avg':>8} | ", end="")
    for yr in wf_years:
        print(f"  {yr:>7} |", end="")
    print(f"  {'Pos':>4}")
    print("-" * 130)
    for i, wf in enumerate(wf_results):
        vals = [wf['windows'].get(yr, 0) for yr in wf_years]
        avg = np.mean(vals) if vals else 0
        pos = sum(1 for v in vals if v > 0)
        print(f"  {i+1:>3} | {wf['label']:<45} | {avg:>+7.1f}% |", end="")
        for v in vals:
            print(f"  {v:>+7.1f}% |", end="")
        print(f"  {pos}/6")

    # Key comparison
    print("\n" + "=" * 98)
    print("  KEY COMPARISONS")
    print("=" * 98)

    # orig vs full groups
    orig_best = [r for r in results if r['config']['group_map'] == 'orig']
    full_best = [r for r in results if r['config']['group_map'] == 'full']
    if orig_best:
        print(f"  Best ORIG groups:  {orig_best[0]['ann']:>+8.1f}%  {orig_best[0]['label']}")
    if full_best:
        print(f"  Best FULL groups:  {full_best[0]['ann']:>+8.1f}%  {full_best[0]['label']}")

    # LB=1 vs multi-LB
    lb1_best = [r for r in results if r['config']['lbs'] == [1]]
    multi_best = [r for r in results if len(r['config']['lbs']) > 1]
    if lb1_best:
        print(f"  Best LB=1 only:    {lb1_best[0]['ann']:>+8.1f}%  {lb1_best[0]['label']}")
    if multi_best:
        print(f"  Best Multi-LB:     {multi_best[0]['ann']:>+8.1f}%  {multi_best[0]['label']}")

    # Long only vs long+short
    long_only = [r for r in results if not r['config'].get('short_too', False)]
    long_short = [r for r in results if r['config'].get('short_too', False)]
    if long_only:
        print(f"  Best Long-only:    {long_only[0]['ann']:>+8.1f}%  {long_only[0]['label']}")
    if long_short:
        print(f"  Best Long+Short:   {long_short[0]['ann']:>+8.1f}%  {long_short[0]['label']}")

    # V69 baseline comparison
    v69_baseline = [r for r in results if r['config']['group_map'] == 'orig'
                    and r['config']['lbs'] == [1]
                    and not r['config'].get('short_too', False)
                    and r['config']['threshold'] == 0.01
                    and r['config']['top_n'] == 3]
    if v69_baseline:
        print(f"\n  V69 baseline (reproduced): {v69_baseline[0]['ann']:>+8.1f}%  {v69_baseline[0]['label']}")

    print(f"\n  Total time: {time.time()-t_start:.1f}s")
    print("=" * 98)


if __name__ == '__main__':
    main()
