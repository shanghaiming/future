"""
Alpha Futures V89 -- Group Structure Granularity Sweep
======================================================
CHAMPION V82: +3305% annual with cross-group z-score (Signal D).
V82 uses 8 groups: ferrous(5), nonferrous(8), precious(2), oils(8),
                   energy(7), chemical(7), soft(7), livestock(3) = 47 syms.

V89 tests 5 DIFFERENT group structures to find optimal granularity
for z-score computation:

  A) 4 SUPER-GROUPS  -- broader, more data per z-score
  B) 8 ORIGINAL      -- V82 baseline (champion)
  C) 12 DETAILED     -- finer sector splits
  D) 16 GROUPS       -- even more granular
  E) LEAVE-ONE-OUT   -- z-score against all commodities directly (no groups)

Signal: Same as V82 Signal D -- z = (own - group_avg) / group_std; z < -thresh -> buy
Sweep: z_threshold=[0.3, 0.5, 0.7], top_n=[1, 3]
Walk-forward: top 10 configs across 2020-2025
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

# ── Multipliers ──────────────────────────────────────────────────────
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

# ══════════════════════════════════════════════════════════════════════
# GROUP STRUCTURES (5 variants)
# ══════════════════════════════════════════════════════════════════════

# B) 8 ORIGINAL -- V82 baseline
GROUPS_B = {
    'ferrous':    ['rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi'],
    'nonferrous': ['cufi', 'alfi', 'znfi', 'nifi', 'pbfi', 'snfi', 'ssfi', 'sffi'],
    'precious':   ['aufi', 'agfi'],
    'oils':       ['afi', 'mfi', 'yfi', 'pfi', 'cfi', 'csfi', 'rrfi', 'lrfi'],
    'energy':     ['scfi', 'mafi', 'bfi', 'fufi', 'pgfi', 'ebfi', 'fbfi'],
    'chemical':   ['ppfi', 'vfi', 'egfi', 'tafi', 'fgfi', 'lfi', 'srfi'],
    'soft':       ['whfi', 'apfi', 'cjfi', 'oifi', 'rmfi', 'cffi'],
    'livestock':  ['jdfi', 'lhfi', 'pkfi'],
}

# A) 4 SUPER-GROUPS (broader)
GROUPS_A = {
    'metals':     ['rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi',
                   'cufi', 'alfi', 'znfi', 'nifi', 'pbfi', 'snfi', 'ssfi', 'sffi',
                   'aufi', 'agfi'],
    'energy':     ['scfi', 'mafi', 'bfi', 'fufi', 'pgfi', 'ebfi', 'fbfi'],
    'agriculture': ['afi', 'mfi', 'yfi', 'pfi', 'cfi', 'csfi', 'rrfi', 'lrfi',
                    'whfi', 'apfi', 'cjfi', 'oifi', 'rmfi', 'cffi', 'srfi'],
    'industrial':  ['ppfi', 'vfi', 'egfi', 'tafi', 'fgfi', 'lfi',
                    'jdfi', 'lhfi', 'pkfi'],
}

# C) 12 DETAILED (more granular)
GROUPS_C = {
    'ferrous':       ['rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi'],
    'copper_group':  ['cufi', 'znfi'],
    'aluminum_group': ['alfi', 'nifi', 'pbfi'],
    'precious':      ['aufi', 'agfi'],
    'oilseed':       ['afi', 'mfi', 'yfi', 'pfi'],
    'grain':         ['cfi', 'csfi', 'rrfi', 'lrfi'],
    'crude_energy':  ['scfi', 'mafi', 'bfi', 'fufi'],
    'gas_energy':    ['pgfi', 'ebfi', 'fbfi'],
    'plastic_chem':  ['ppfi', 'vfi', 'egfi', 'tafi'],
    'other_chem':    ['fgfi', 'lfi', 'srfi'],
    'soft':          ['whfi', 'apfi', 'cjfi', 'oifi', 'rmfi', 'cffi'],
    'livestock':     ['jdfi', 'lhfi', 'pkfi'],
    # minor metals not in any group: ssfi, sffi, snfi -> treated as ungrouped
}

# D) 16 GROUPS (most granular)
GROUPS_D = {
    'ferrous_steel': ['rbfi', 'hcfi', 'ifi'],
    'ferrous_coal':  ['jfi', 'jmfi'],
    'copper':        ['cufi', 'znfi'],
    'aluminum':      ['alfi', 'nifi', 'pbfi'],
    'precious':      ['aufi', 'agfi'],
    'soybean':       ['afi', 'mfi', 'yfi', 'pfi'],
    'grain':         ['cfi', 'csfi', 'rrfi', 'lrfi'],
    'crude':         ['scfi', 'bfi'],
    'refined_oil':   ['mafi', 'fufi', 'pgfi'],
    'gas':           ['ebfi', 'fbfi'],
    'plastic':       ['ppfi', 'vfi', 'egfi'],
    'chemical':      ['tafi', 'fgfi', 'lfi'],
    'soft_agri':     ['whfi', 'apfi', 'cjfi', 'oifi', 'rmfi', 'cffi'],
    'sugar_minor':   ['srfi'],
    'livestock':     ['jdfi', 'lhfi', 'pkfi'],
    'minor_metals':  ['ssfi', 'sffi', 'snfi'],
}

# E) LEAVE-ONE-OUT -- no groups, z-score vs all commodities directly
# (handled specially in code -- single pseudo-group containing all commodities)

ALL_GROUP_STRUCTURES = {
    'A_4super':   GROUPS_A,
    'B_8orig':    GROUPS_B,
    'C_12detail': GROUPS_C,
    'D_16fine':   GROUPS_D,
    # E is handled separately
}


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def main():
    print("=" * 120)
    print("Alpha Futures V89 -- Group Structure Granularity Sweep")
    print("Testing 5 group structures for z-score signal quality")
    print("A) 4 super-groups  B) 8 original (V82)  C) 12 detailed  D) 16 fine  E) leave-one-out")
    print("=" * 120)

    # ── Load data ────────────────────────────────────────────────────
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    sym_to_si = {syms[si]: si for si in range(NS)}

    # ── Precompute 1-day returns ─────────────────────────────────────
    print("\n[Signals] Computing 1-day returns...", flush=True)
    t0 = time.time()
    ret1 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            cn = C[si, di]
            cp = C[si, di - 1]
            if not np.isnan(cn) and not np.isnan(cp) and cp > 0:
                ret1[si, di] = (cn - cp) / cp
    print(f"  Returns computed ({time.time()-t0:.1f}s)")

    # ══════════════════════════════════════════════════════════════════
    # BUILD GROUP INDEX MAPS FOR EACH STRUCTURE
    # ══════════════════════════════════════════════════════════════════
    print("\n[Groups] Building group index maps...", flush=True)

    # For each structure, produce:
    #   gm_map: {group_name -> [si, ...]}
    #   si_group: {si -> group_name}
    #   trade_sis: [si, ...] -- commodities that belong to at least one group
    #   group_names: sorted list of group names

    structures = {}

    for sname, gmap_sym in ALL_GROUP_STRUCTURES.items():
        gm = {}
        sg = {}
        for gname, members in gmap_sym.items():
            sis = []
            for s in members:
                if s in sym_to_si:
                    si = sym_to_si[s]
                    sis.append(si)
                    sg[si] = gname
            if sis:
                gm[gname] = sorted(set(sis))
        ts = sorted(sg.keys())
        gn = sorted(gm.keys())
        n_total = sum(len(v) for v in gm.values())
        structures[sname] = {
            'gm_map': gm, 'si_group': sg,
            'trade_sis': ts, 'group_names': gn,
            'n_groups': len(gn), 'n_commodities': n_total,
            'n_unique': len(ts),
        }

    # E) Leave-one-out: single group with all known commodities
    all_known = set()
    for gmap_sym in ALL_GROUP_STRUCTURES.values():
        for members in gmap_sym.values():
            all_known.update(members)
    all_sis = sorted([sym_to_si[s] for s in all_known if s in sym_to_si])
    structures['E_loo'] = {
        'gm_map': {'all': all_sis},
        'si_group': {si: 'all' for si in all_sis},
        'trade_sis': all_sis,
        'group_names': ['all'],
        'n_groups': 1,
        'n_commodities': len(all_sis),
        'n_unique': len(all_sis),
    }

    for sname, sd in structures.items():
        print(f"  {sname}: {sd['n_groups']} groups, {sd['n_unique']} commodities")
        for gn in sd['group_names']:
            print(f"    {gn}({len(sd['gm_map'][gn])})", end="")
        print()

    # ══════════════════════════════════════════════════════════════════
    # PRECOMPUTE GROUP-LEVEL SIGNALS PER STRUCTURE
    # ══════════════════════════════════════════════════════════════════
    print("\n[Signals] Computing group-level signals per structure...", flush=True)
    t0 = time.time()

    # For each structure, compute:
    #   grp_total[group_name] -> array[ND] = avg return of that group
    #   all_groups_avg[ND] = grand mean of all group averages
    #   all_groups_std[ND] = std of group averages

    struct_signals = {}

    for sname, sd in structures.items():
        gm = sd['gm_map']
        gnames = sd['group_names']

        # Compute group total average return per day
        grp_total = {}
        for gname in gnames:
            arr = np.full(ND, np.nan)
            members = gm[gname]
            for di in range(1, ND):
                vals = [ret1[sk, di] for sk in members if not np.isnan(ret1[sk, di])]
                if vals:
                    arr[di] = np.mean(vals)
            grp_total[gname] = arr

        # Grand mean and std of group averages
        all_avg = np.full(ND, np.nan)
        all_std = np.full(ND, np.nan)
        for di in range(1, ND):
            vals = [grp_total[g][di] for g in gnames if not np.isnan(grp_total[g][di])]
            if len(vals) >= 1:
                all_avg[di] = np.mean(vals)
                all_std[di] = np.std(vals) if len(vals) >= 2 else 0.0

        struct_signals[sname] = {
            'grp_total': grp_total,
            'all_avg': all_avg,
            'all_std': all_std,
        }

    print(f"  Signals computed ({time.time()-t0:.1f}s)")

    # ══════════════════════════════════════════════════════════════════
    # BACKTEST ENGINE
    # ══════════════════════════════════════════════════════════════════
    def run_backtest(struct_name, threshold, top_n, wf_test_year=None):
        """
        Signal D (z-score): z = (own_return - all_groups_avg) / all_groups_std
        When z < -threshold -> buy (unusually weak vs all groups)
        1-day hold, long-only, close-to-close.
        """
        sd = structures[struct_name]
        ss = struct_signals[struct_name]
        trade_sis = sd['trade_sis']
        all_avg = ss['all_avg']
        all_std = ss['all_std']

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

        cash = float(CASH0)
        positions = []
        trades = []

        for di in range(start_di, end_di):
            # Reset cash at test window start (WF mode)
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                positions = []

            # ── Close positions held 1 day ───────────────────────────
            closed = []
            for pos in positions:
                if di - pos['entry_di'] >= 1:
                    cn = C[pos['si'], di]
                    if np.isnan(cn) or cn <= 0:
                        cn = pos['entry']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    mkt_val = cn * mult * abs(pos['lots'])
                    cash += mkt_val - mkt_val * COMM
                    pnl = (cn - pos['entry']) * mult * pos['lots']
                    invested = pos['entry'] * mult * abs(pos['lots'])
                    pnl_pct = pnl / invested * 100 if invested > 0 else 0
                    trades.append({
                        'pnl_pct': pnl_pct,
                        'di': pos['entry_di'],
                        'year': dates[di].year if di < ND else dates[-1].year,
                    })
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            # ── Generate signals (Signal D: z-score) ─────────────────
            aga = all_avg[di]
            ags = all_std[di]
            if np.isnan(aga) or np.isnan(ags) or ags < 1e-8:
                continue

            candidates = []
            for si in trade_sis:
                own = ret1[si, di]
                if np.isnan(own):
                    continue
                cc = C[si, di]
                if np.isnan(cc) or cc <= 0:
                    continue
                if any(p['si'] == si for p in positions):
                    continue
                z = (own - aga) / ags
                if z < -threshold:
                    candidates.append((si, -z, syms[si]))

            if not candidates:
                continue

            # Sort by score descending (most negative z = strongest signal)
            candidates.sort(key=lambda x: -x[1])

            # Open positions (up to top_n slots)
            n_slots = top_n - len(positions)
            for si, score, sym in candidates[:max(0, n_slots)]:
                c = C[si, di]
                if np.isnan(c) or c <= 0:
                    continue
                mult = MULT.get(sym, DEF_MULT)
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
                    'lots': lots, 'dir': 1, 'sym': sym,
                })

        # Close remaining
        for pos in positions:
            ae = ND - 1
            cn = C[pos['si'], ae]
            if np.isnan(cn) or cn <= 0:
                cn = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = cn * mult * abs(pos['lots'])
            cash += mkt_val - mkt_val * COMM

        # Results
        wf_mode = wf_test_year is not None
        n_days_test = (test_end_di - test_start_di) if wf_mode else (end_di - start_di)
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0

        # Max drawdown
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

        return {
            'ann': ann, 'wr': wr, 'n': n_trades, 'avg_pnl': avg_pnl,
            'final_cash': cash, 'n_days': n_days_test, 'mdd': mdd,
        }

    # ══════════════════════════════════════════════════════════════════
    # BUILD CONFIGURATIONS
    # ══════════════════════════════════════════════════════════════════
    print("\n[Sweep] Building configurations...", flush=True)
    configs = []
    struct_order = ['A_4super', 'B_8orig', 'C_12detail', 'D_16fine', 'E_loo']

    for sname in struct_order:
        for zt in [0.3, 0.5, 0.7]:
            for tn in [1, 3]:
                label = f"{sname}_Z{zt}_TN{tn}"
                configs.append({
                    'struct': sname,
                    'threshold': zt,
                    'top_n': tn,
                    'label': label,
                })

    print(f"  Total configs: {len(configs)}")

    # ══════════════════════════════════════════════════════════════════
    # RUN FULL-PERIOD BACKTEST
    # ══════════════════════════════════════════════════════════════════
    print("\n[Backtest] Running full-period sweep...", flush=True)
    results = []
    for i, cfg in enumerate(configs):
        r = run_backtest(cfg['struct'], cfg['threshold'], cfg['top_n'])
        if r:
            r['config'] = cfg
            r['label'] = cfg['label']
            r['struct'] = cfg['struct']
            results.append(r)
        if (i + 1) % 5 == 0 or i == len(configs) - 1:
            print(f"  ... {i+1}/{len(configs)} done", flush=True)

    results.sort(key=lambda x: -x['ann'])

    # Print top 30
    print(f"\n{'=' * 130}")
    print("  FULL-PERIOD RESULTS (Top 30)")
    print(f"{'=' * 130}")
    print(f"  {'#':>3} | {'Label':<35} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7}")
    print("-" * 110)
    for i, r in enumerate(results[:30]):
        print(f"  {i+1:>3} | {r['label']:<35} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}%")

    # ══════════════════════════════════════════════════════════════════
    # STRUCTURE COMPARISON (best per structure, full period)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 130}")
    print("  GROUP STRUCTURE COMPARISON (Best config per structure, full period)")
    print(f"{'=' * 130}")
    print(f"  {'Structure':<15} | {'Groups':>6} | {'Commods':>8} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | Best Config")
    print("-" * 130)

    best_per_struct = {}
    for r in results:
        s = r['struct']
        if s not in best_per_struct:
            best_per_struct[s] = r

    for sname in struct_order:
        if sname in best_per_struct:
            b = best_per_struct[sname]
            sd = structures[sname]
            print(f"  {sname:<15} | {sd['n_groups']:>6} | {sd['n_unique']:>8} | {b['ann']:>+8.1f}% | {b['wr']:>5.1f}% | {b['n']:>5} | {b['avg_pnl']:>+6.3f}% | {b['mdd']:>6.1f}% | {b['label']}")

    # ══════════════════════════════════════════════════════════════════
    # THRESHOLD/TOP_N BREAKDOWN PER STRUCTURE
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 130}")
    print("  THRESHOLD x TOP_N BREAKDOWN PER STRUCTURE")
    print(f"{'=' * 130}")
    for sname in struct_order:
        print(f"\n  --- {sname} ({structures[sname]['n_groups']} groups) ---")
        print(f"  {'Z_thresh':>10} | {'TN':>3} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7}")
        print("  " + "-" * 70)
        for r in results:
            if r['struct'] == sname:
                cfg = r['config']
                print(f"  {cfg['threshold']:>10.1f} | {cfg['top_n']:>3} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}%")

    # ══════════════════════════════════════════════════════════════════
    # WALK-FORWARD (Top 10 configs)
    # ══════════════════════════════════════════════════════════════════
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Take top 10 by full-period return, plus best per structure if not already included
    wf_configs = list(results[:10])
    for sname in struct_order:
        if sname in best_per_struct:
            r = best_per_struct[sname]
            if r['label'] not in [w['label'] for w in wf_configs]:
                wf_configs.append(r)

    print(f"\n{'=' * 160}")
    print(f"  WALK-FORWARD ({len(wf_configs)} configs)")
    print(f"{'=' * 160}")

    header = f"  {'#':>3} | {'Config':<35} | {'Grps':>4} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>7} |"
    header += f" {'Pos':>4} | {'MDD':>7}"
    print(header)
    print("-" * 160)

    wf_rows = []
    for i, r in enumerate(wf_configs):
        cfg = r['config']
        sd = structures[cfg['struct']]
        wf_row = {'label': cfg['label'], 'struct': cfg['struct'],
                  'n_groups': sd['n_groups'], 'windows': {}, 'mdd': {}}
        for yr in wf_years:
            wr = run_backtest(cfg['struct'], cfg['threshold'], cfg['top_n'], wf_test_year=yr)
            if wr:
                wf_row['windows'][yr] = wr['ann']
                wf_row['mdd'][yr] = wr['mdd']
        wf_rows.append(wf_row)

        vals = [wf_row['windows'].get(yr, 0) for yr in wf_years]
        avg = np.mean(vals) if vals else 0
        pos = sum(1 for v in vals if v > 0)
        avg_mdd = np.mean(list(wf_row['mdd'].values())) if wf_row['mdd'] else 0

        row_str = f"  {i+1:>3} | {wf_row['label']:<35} | {wf_row['n_groups']:>4} | {avg:>+7.1f}% |"
        for v in vals:
            row_str += f" {v:>+7.1f}% |"
        row_str += f" {pos}/6 | {avg_mdd:>6.1f}%"
        print(row_str)

    # ══════════════════════════════════════════════════════════════════
    # WF COMPARISON PER STRUCTURE (best config for each)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 140}")
    print("  WALK-FORWARD COMPARISON (Best config per group structure)")
    print(f"{'=' * 140}")
    header2 = f"  {'Structure':<15} | {'Grps':>4} | {'WF Avg':>8} |"
    for yr in wf_years:
        header2 += f" {yr:>7} |"
    header2 += f" {'Pos':>4} | {'MDD':>7} | Config"
    print(header2)
    print("-" * 140)

    for sname in struct_order:
        wf_match = [w for w in wf_rows if w['struct'] == sname]
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            avg = np.mean(vals) if vals else 0
            pos = sum(1 for v in vals if v > 0)
            avg_mdd = np.mean(list(wf['mdd'].values())) if wf['mdd'] else 0
            row_str = f"  {sname:<15} | {wf['n_groups']:>4} | {avg:>+7.1f}% |"
            for v in vals:
                row_str += f" {v:>+7.1f}% |"
            row_str += f" {pos}/6 | {avg_mdd:>6.1f}% | {wf['label']}"
            print(row_str)

    # ══════════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 130}")
    print("  FINAL VERDICT")
    print(f"{'=' * 130}")

    # Full-period winner
    if results:
        best = results[0]
        print(f"\n  FULL-PERIOD CHAMPION:")
        print(f"    {best['label']}: {best['ann']:>+8.1f}%  WR {best['wr']:>5.1f}%  N {best['n']:>5}  MDD {best['mdd']:>6.1f}%")

    # WF winner (highest avg WF return with >=4/6 positive)
    best_wf = None
    for wf in wf_rows:
        vals = [wf['windows'].get(yr, 0) for yr in wf_years]
        avg = np.mean(vals) if vals else 0
        pos = sum(1 for v in vals if v > 0)
        wf['wf_avg'] = avg
        wf['wf_pos'] = pos
        if pos >= 4:
            if best_wf is None or avg > best_wf['wf_avg']:
                best_wf = wf

    if best_wf:
        print(f"\n  WALK-FORWARD CHAMPION (>= 4/6 positive windows):")
        vals = [best_wf['windows'].get(yr, 0) for yr in wf_years]
        print(f"    {best_wf['label']}: avg {best_wf['wf_avg']:>+7.1f}%  pos {best_wf['wf_pos']}/6")
        for yr in wf_years:
            v = best_wf['windows'].get(yr, 0)
            print(f"      {yr}: {v:>+8.1f}%")
    else:
        # Fall back to best avg regardless
        wf_sorted = sorted([w for w in wf_rows], key=lambda x: -x.get('wf_avg', 0))
        if wf_sorted:
            bw = wf_sorted[0]
            print(f"\n  WALK-FORWARD BEST (no config with >=4/6 positive):")
            vals = [bw['windows'].get(yr, 0) for yr in wf_years]
            print(f"    {bw['label']}: avg {bw['wf_avg']:>+7.1f}%  pos {bw['wf_pos']}/6")

    # Comparison summary
    print(f"\n  GROUP GRANULARITY RANKING (by full-period annual return):")
    rank = 1
    for sname in struct_order:
        if sname in best_per_struct:
            b = best_per_struct[sname]
            sd = structures[sname]
            tag = " <-- CHAMPION" if b == results[0] else ""
            print(f"    {rank}. {sname:<15} ({sd['n_groups']:>2} groups, {sd['n_unique']:>2} comms): {b['ann']:>+8.1f}%{tag}")
            rank += 1

    print(f"\n  Total time: {time.time()-t_start:.1f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
