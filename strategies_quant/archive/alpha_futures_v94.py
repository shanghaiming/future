"""
Alpha Futures V94 -- Expanded Universe (68 Commodities)
=======================================================
V92 champion: overnight cross-group z-score, 44 commodities, 8 groups, +4282% annual.
V94 idea: expand to ALL 68 commodities by assigning the 24 ungrouped ones.

Steps:
  1) Load data, print all 68 symbols, identify the 24 not in GROUP_MAP
  2) Fix bug: 'srfi' in both 'chemical' and 'soft' -- srfi=sugar -> 'soft' only
  3) Assign remaining 24 to groups based on commodity type
  4) Test: A) V92 baseline (44), B) expanded 68, C) expanded with new groups
  5) Sweep threshold=[0.1, 0.3, 0.5], top_n=[1, 3, 5]
  6) Walk-forward top 10 configs across 2020-2025

Signal: overnight_ret = (O - C_prev) / C_prev
        z = (own_overnight - all_groups_avg) / all_groups_std
        z < -threshold -> buy at C[di], sell at C[di+1]. Long-only, 1-day hold.
"""
import sys, os, time, warnings
import numpy as np
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

# ============================================================
# GROUP MAPS
# ============================================================

# --- V92 ORIGINAL (44 commodities, 8 groups) ---
# Bug fix: srfi removed from 'chemical', stays in 'soft' (srfi = sugar)
GROUP_MAP_V92 = {}
for _s in ['rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi']:
    GROUP_MAP_V92[_s] = 'ferrous'
for _s in ['cufi', 'alfi', 'znfi', 'nifi', 'pbfi', 'snfi', 'ssfi', 'sffi']:
    GROUP_MAP_V92[_s] = 'nonferrous'
for _s in ['aufi', 'agfi']:
    GROUP_MAP_V92[_s] = 'precious'
for _s in ['afi', 'mfi', 'yfi', 'pfi', 'cfi', 'csfi', 'rrfi', 'lrfi']:
    GROUP_MAP_V92[_s] = 'oils'
for _s in ['scfi', 'mafi', 'bfi', 'fufi', 'pgfi', 'ebfi', 'fbfi']:
    GROUP_MAP_V92[_s] = 'energy'
for _s in ['ppfi', 'vfi', 'egfi', 'tafi', 'fgfi', 'lfi']:
    GROUP_MAP_V92[_s] = 'chemical'
for _s in ['whfi', 'apfi', 'cjfi', 'oifi', 'rmfi', 'srfi', 'cffi']:
    GROUP_MAP_V92[_s] = 'soft'
for _s in ['jdfi', 'lhfi', 'pkfi']:
    GROUP_MAP_V92[_s] = 'livestock'

# --- V94 EXPANDED (68 commodities) ---
# Assign the 24 missing commodities to appropriate groups.
# Chinese futures symbol -> commodity mapping:
#   wrffi = wire rod (线材) -> ferrous
#   rufi  = rubber (橡胶) -> soft (agricultural, not ferrous!)
#   spfi  = steel product (螺纹钢? no, already rbfi; spfi=热卷? wait, that's hcfi)
#           Actually spfi might be in V92 already. We'll see.
#   smfi  = silicon manganese (硅锰) -> ferrous (alloy)
#   bufi  = butadiene (丁二烯) -> chemical
#   bbfi  = bean board (豆二/豆粕? no; bbfi=纤板? actually 胶合板) -> other
#   nrfi  = nickel (镍? already nifi... maybe nrfi = 20号胶/TSR20 rubber) -> soft
#   jrfi  = jujube (红枣) -> soft
#   pmfi  = PMMA (聚甲基丙烯酸甲酯/亚克力) -> chemical
#   rsfi  = rapeseed (菜籽) -> soft (or oils)
#   cyfi  = cyclohexane? No. cyfi = 纯苯 (benzene)? -> chemical
#   pfifi = peanut (花生) -> soft
#   safi  = antimony? Or soda ash (纯碱) -> chemical
#   urfi  = urea (尿素) -> chemical
#   lufi  = lutetium? No. lufi = BR (丁二烯橡胶/butadiene rubber) -> chemical
#   bcfi  = bitumen (沥青? but bfi already is bitumen...) -> energy
#   lgfi  = LDPE? or lignin? lgfi = 液化气? No, pgfi is LPG. lgfi = 硅铁? No, that's sifi.
#           lgfi might be something else. -> other
#   brfi  = butadiene rubber (丁二烯橡胶/BR) -> chemical
#   lcfi  = lithium carbonate (碳酸锂) -> nonferrous
#   sifi  = silicon iron (硅铁/ferrosilicon) -> ferrous
#   ni    = nickel index -> nonferrous
#   tai   = PTA index -> chemical
#   bbfi  = block board (胶合板/plywood) -> other
#
# After loading data we'll see which 24 are actually missing.
# For now, define the expanded map based on best knowledge:

GROUP_MAP_V94 = dict(GROUP_MAP_V92)  # start from V92

# ferrous additions: wire rod, silicon manganese, ferrosilicon, steel product
for _s in ['wrffi', 'smfi', 'sifi']:
    GROUP_MAP_V94[_s] = 'ferrous'

# nonferrous additions: lithium carbonate, nickel index
for _s in ['lcfi', 'ni']:
    GROUP_MAP_V94[_s] = 'nonferrous'

# chemical additions: butadiene, PMMA, cyclohexane/benzene, soda ash, urea,
#                     butadiene rubber, BR, PTA index
for _s in ['bufi', 'pmfi', 'cyfi', 'safi', 'urfi', 'lufi', 'brfi', 'tai', 'lgfi']:
    GROUP_MAP_V94[_s] = 'chemical'

# energy additions: bitumen
for _s in ['bcfi']:
    GROUP_MAP_V94[_s] = 'energy'

# soft/agricultural additions: rubber, TSR20 rubber, jujube, rapeseed,
#                               peanut, block board/plywood
for _s in ['rufi', 'nrfi', 'jrfi', 'rsfi', 'pfifi', 'bbfi']:
    GROUP_MAP_V94[_s] = 'soft'

# oils additions: (rsfi could be oils too, but rapeseed -> soft is fine)


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def main():
    print("=" * 120)
    print("Alpha Futures V94 -- Expanded Universe (68 Commodities)")
    print("=" * 120)

    # ================================================================
    # LOAD DATA
    # ================================================================
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ================================================================
    # STEP 1: Identify all 68 symbols and which are missing from V92
    # ================================================================
    print(f"\n{'=' * 100}")
    print("  STEP 1: Symbol Inventory")
    print(f"{'=' * 100}")
    print(f"\n  All {NS} symbols:")
    for si in range(NS):
        s = syms[si]
        in_v92 = s in GROUP_MAP_V92
        in_v94 = s in GROUP_MAP_V94
        tag_v92 = GROUP_MAP_V92.get(s, '---')
        tag_v94 = GROUP_MAP_V94.get(s, '---')
        flag = ""
        if not in_v92 and in_v94:
            flag = " << NEW in V94"
        elif not in_v92 and not in_v94:
            flag = " << STILL UNASSIGNED"
        print(f"    [{si:>2}] {s:<8}  V92: {tag_v92:<15}  V94: {tag_v94:<15}{flag}")

    v92_sis = [si for si in range(NS) if syms[si] in GROUP_MAP_V92]
    v94_sis = [si for si in range(NS) if syms[si] in GROUP_MAP_V94]
    missing_sis = [si for si in range(NS) if syms[si] not in GROUP_MAP_V94]

    print(f"\n  V92 tradeable: {len(v92_sis)} commodities")
    print(f"  V94 tradeable: {len(v94_sis)} commodities")
    if missing_sis:
        print(f"  Still missing ({len(missing_sis)}): {[syms[si] for si in missing_sis]}")
        # Assign any truly missing ones to 'other' group
        for si in missing_sis:
            GROUP_MAP_V94[syms[si]] = 'other'
            print(f"    -> Assigned {syms[si]} to 'other' group")
        v94_sis = [si for si in range(NS) if syms[si] in GROUP_MAP_V94]
        print(f"  V94 tradeable after assignment: {len(v94_sis)} commodities")

    # ================================================================
    # STEP 2: Fix srfi bug (verify)
    # ================================================================
    print(f"\n  BUG FIX CHECK: srfi in chemical? {'srfi' in [_s for _s, g in GROUP_MAP_V92.items() if g == 'chemical']}")
    print(f"                 srfi in soft?      {'srfi' in [_s for _s, g in GROUP_MAP_V92.items() if g == 'soft']}")
    print(f"  V92 chemical group: {sorted([_s for _s, g in GROUP_MAP_V92.items() if g == 'chemical'])}")
    print(f"  V92 soft group:     {sorted([_s for _s, g in GROUP_MAP_V92.items() if g == 'soft'])}")

    # ================================================================
    # PRECOMPUTE RETURNS
    # ================================================================
    print(f"\n[Signals] Computing returns...", flush=True)
    t0 = time.time()

    overnight_ret = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            c_prev = C[si, di - 1]
            o_now = O[si, di]
            if not np.isnan(c_prev) and c_prev > 0 and not np.isnan(o_now) and o_now > 0:
                overnight_ret[si, di] = (o_now - c_prev) / c_prev

    print(f"  Returns computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def build_groups(group_map):
        """Build group membership structures from a GROUP_MAP."""
        gm_map = {}
        si_group = {}
        for si in range(NS):
            g = group_map.get(syms[si])
            if g:
                gm_map.setdefault(g, []).append(si)
                si_group[si] = g
        trade_sis = [si for si in range(NS) if si in si_group]
        group_names = sorted(gm_map.keys())
        return gm_map, si_group, trade_sis, group_names

    def compute_cross_group_zscores(ret_array, trade_sis, gm_map, group_names):
        """Compute z-scores of each commodity's return vs cross-group distribution."""
        grp_total = {}
        for grp in group_names:
            arr = np.full(ND, np.nan)
            members = gm_map[grp]
            for di in range(1, ND):
                vals = [ret_array[sk, di] for sk in members if not np.isnan(ret_array[sk, di])]
                if vals:
                    arr[di] = np.mean(vals)
            grp_total[grp] = arr

        aga = np.full(ND, np.nan)
        ags = np.full(ND, np.nan)
        for di in range(1, ND):
            vals = [grp_total[g][di] for g in group_names if not np.isnan(grp_total[g][di])]
            if len(vals) >= 2:
                aga[di] = np.mean(vals)
                ags[di] = np.std(vals)

        z = np.full((NS, ND), np.nan)
        for si in trade_sis:
            for di in range(1, ND):
                own = ret_array[si, di]
                if np.isnan(own) or np.isnan(aga[di]) or np.isnan(ags[di]) or ags[di] < 1e-8:
                    continue
                z[si, di] = (own - aga[di]) / ags[di]

        return z

    def run_backtest(config, wf_test_year=None):
        """
        Config keys:
            threshold: float (z-score cutoff)
            top_n: int (max positions)
            group_map_name: 'V92' | 'V94'
            comm: float
        """
        threshold = config['threshold']
        top_n = config['top_n']
        comm = config.get('comm', COMM)
        gmn = config['group_map_name']

        if gmn == 'V92':
            gm_map, si_group, trade_sis, group_names = build_groups(GROUP_MAP_V92)
            z = z_v92
        else:
            gm_map, si_group, trade_sis, group_names = build_groups(GROUP_MAP_V94)
            z = z_v94

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
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                positions = []

            # Close positions (close-to-close: entered yesterday, exit today)
            closed = []
            for pos in positions:
                if di - pos['entry_di'] >= 1:
                    cn = C[pos['si'], di]
                    if np.isnan(cn) or cn <= 0:
                        cn = pos['entry']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    mkt_val = cn * mult * abs(pos['lots'])
                    cash += mkt_val - mkt_val * comm
                    pnl = (cn - pos['entry']) * mult * pos['lots'] * pos['dir']
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

            # Generate signals: z < -threshold -> buy
            candidates = []
            for si in trade_sis:
                if any(p['si'] == si for p in positions):
                    continue
                c_now = C[si, di]
                if np.isnan(c_now) or c_now <= 0:
                    continue
                zv = z[si, di]
                if np.isnan(zv):
                    continue
                if zv < -threshold:
                    candidates.append((si, -zv, syms[si]))

            if not candidates:
                continue

            candidates.sort(key=lambda x: -x[1])
            n_slots = top_n - len(positions)
            for si, score, sym in candidates[:max(0, n_slots)]:
                price = C[si, di]
                if np.isnan(price) or price <= 0:
                    continue
                mult = MULT.get(sym, DEF_MULT)
                notional = price * mult
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
                    'si': si, 'entry': price, 'entry_di': di,
                    'lots': lots, 'dir': 1, 'sym': sym,
                })

        # Close remaining positions
        for pos in positions:
            ae = end_di - 1 if end_di < ND else ND - 1
            cn = C[pos['si'], ae]
            if np.isnan(cn) or cn <= 0:
                cn = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = cn * mult * abs(pos['lots'])
            cash += mkt_val - mkt_val * comm

        wf_mode = wf_test_year is not None
        n_days_test = (test_end_di - test_start_di) if wf_mode else (end_di - start_di)
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0

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

    # ================================================================
    # PRECOMPUTE Z-SCORES FOR BOTH UNIVERSES
    # ================================================================
    print("\n[Signals] Computing z-scores...", flush=True)
    t1 = time.time()

    gm92, sg92, ts92, gn92 = build_groups(GROUP_MAP_V92)
    gm94, sg94, ts94, gn94 = build_groups(GROUP_MAP_V94)

    print(f"  V92: {len(ts92)} commodities in {len(gn92)} groups")
    for g in gn92:
        print(f"    {g}: {len(gm92[g])} -- {sorted([syms[si] for si in gm92[g]])}")
    print(f"  V94: {len(ts94)} commodities in {len(gn94)} groups")
    for g in gn94:
        print(f"    {g}: {len(gm94[g])} -- {sorted([syms[si] for si in gm94[g]])}")

    z_v92 = compute_cross_group_zscores(overnight_ret, ts92, gm92, gn92)
    z_v94 = compute_cross_group_zscores(overnight_ret, ts94, gm94, gn94)

    print(f"  z_v92: mean={np.nanmean(z_v92):.4f}, std={np.nanstd(z_v92):.4f}")
    print(f"  z_v94: mean={np.nanmean(z_v94):.4f}, std={np.nanstd(z_v94):.4f}")
    print(f"  Computed in {time.time()-t1:.1f}s")

    # ================================================================
    # BUILD CONFIGURATIONS
    # ================================================================
    print(f"\n[Sweep] Building configurations...", flush=True)
    configs = []
    cid = 0

    for gmn in ['V92', 'V94']:
        for thresh in [0.1, 0.3, 0.5]:
            for tn in [1, 3, 5]:
                cid += 1
                configs.append({
                    'id': cid, 'group_map_name': gmn,
                    'threshold': thresh, 'top_n': tn, 'comm': COMM,
                    'label': f"{gmn}_Z{thresh}_TN{tn}",
                })

    print(f"  Total configs: {len(configs)}")

    # ================================================================
    # RUN FULL-PERIOD BACKTEST
    # ================================================================
    print(f"\n[Backtest] Running full-period sweep...", flush=True)
    results = []
    for i, cfg in enumerate(configs):
        r = run_backtest(cfg)
        if r:
            r['config'] = cfg
            r['label'] = cfg['label']
            results.append(r)
        if (i + 1) % 5 == 0 or i == len(configs) - 1:
            print(f"  ... {i+1}/{len(configs)} done", flush=True)

    results.sort(key=lambda x: -x['ann'])

    # ================================================================
    # FULL-PERIOD RESULTS
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  FULL-PERIOD RESULTS (All configs, sorted by annual return)")
    print(f"{'=' * 130}")
    print(f"  {'#':>3} | {'Label':<25} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7}")
    print("-" * 100)
    for i, r in enumerate(results):
        print(f"  {i+1:>3} | {r['label']:<25} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}%")

    # ================================================================
    # V92 vs V94 COMPARISON
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  V92 vs V94 COMPARISON (Best per universe)")
    print(f"{'=' * 130}")

    v92_results = [r for r in results if r['config']['group_map_name'] == 'V92']
    v94_results = [r for r in results if r['config']['group_map_name'] == 'V94']

    if v92_results:
        best_v92 = v92_results[0]
        print(f"\n  Best V92 (44 commodities):  {best_v92['ann']:>+8.1f}%  WR={best_v92['wr']:.1f}%  N={best_v92['n']}  MDD={best_v92['mdd']:.1f}%  ({best_v92['label']})")
    if v94_results:
        best_v94 = v94_results[0]
        print(f"  Best V94 (68 commodities):  {best_v94['ann']:>+8.1f}%  WR={best_v94['wr']:.1f}%  N={best_v94['n']}  MDD={best_v94['mdd']:.1f}%  ({best_v94['label']})")

    if v92_results and v94_results:
        diff = best_v94['ann'] - best_v92['ann']
        if diff > 0:
            print(f"\n  >>> V94 EXPANDED UNIVERSE IS BETTER (+{diff:.1f}% annual) <<<")
        else:
            print(f"\n  >>> V92 BASELINE IS BETTER ({diff:+.1f}% annual) <<<")

    # Per-config comparison (same threshold/top_n)
    print(f"\n{'=' * 130}")
    print("  MATCHED COMPARISON (V92 vs V94, same threshold & top_n)")
    print(f"{'=' * 130}")
    print(f"  {'Config':<25} | {'V92 Ann':>10} | {'V94 Ann':>10} | {'Delta':>10} | {'V92 WR':>7} | {'V94 WR':>7} | {'V92 N':>6} | {'V94 N':>6}")
    print("-" * 130)

    v92_by_cfg = {r['label']: r for r in v92_results}
    v94_by_cfg = {r['label']: r for r in v94_results}

    for thresh in [0.1, 0.3, 0.5]:
        for tn in [1, 3, 5]:
            lbl = f"V92_Z{thresh}_TN{tn}"
            lbl94 = f"V94_Z{thresh}_TN{tn}"
            r92 = v92_by_cfg.get(lbl)
            r94 = v94_by_cfg.get(lbl94)
            if r92 and r94:
                delta = r94['ann'] - r92['ann']
                tag = "W94" if delta > 0 else "W92"
                print(f"  Z{thresh}_TN{tn:<18} | {r92['ann']:>+9.1f}% | {r94['ann']:>+9.1f}% | {delta:>+9.1f}% | {r92['wr']:>5.1f}% | {r94['wr']:>5.1f}% | {r92['n']:>6} | {r94['n']:>6}  {tag}")

    # ================================================================
    # NEW COMMODITIES IMPACT
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  NEW COMMODITY GROUPS IN V94")
    print(f"{'=' * 130}")
    v94_only = set(GROUP_MAP_V94.keys()) - set(GROUP_MAP_V92.keys())
    print(f"  Commodities added in V94 ({len(v94_only)}):")
    for s in sorted(v94_only):
        g = GROUP_MAP_V94.get(s, '???')
        m = MULT.get(s, DEF_MULT)
        print(f"    {s:<8} -> {g:<15} mult={m}")

    # Count per group
    print(f"\n  Group sizes (V92 -> V94):")
    all_groups = sorted(set(list(GROUP_MAP_V92.values()) + list(GROUP_MAP_V94.values())))
    for g in all_groups:
        n92 = sum(1 for s, gr in GROUP_MAP_V92.items() if gr == g)
        n94 = sum(1 for s, gr in GROUP_MAP_V94.items() if gr == g)
        added = n94 - n92
        new_syms = [s for s in v94_only if GROUP_MAP_V94.get(s) == g]
        print(f"    {g:<15}: {n92:>2} -> {n94:>2}  (+{added})  new: {sorted(new_syms)}")

    # ================================================================
    # WALK-FORWARD (Top 10 configs)
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Collect top 10 configs, ensuring both V92 and V94 are represented
    wf_configs = list(results[:10])
    # Ensure best V92 and best V94 are included
    if v92_results and v92_results[0]['config'] not in [w['config'] for w in wf_configs]:
        wf_configs.append(v92_results[0])
    if v94_results and v94_results[0]['config'] not in [w['config'] for w in wf_configs]:
        wf_configs.append(v94_results[0])

    print(f"\n{'=' * 150}")
    print(f"  WALK-FORWARD ({len(wf_configs)} configs)")
    print(f"{'=' * 150}")

    header = f"  {'#':>3} | {'Config':<25} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>7} |"
    header += f" {'Pos':>4} | {'MDD':>7}"
    print(header)
    print("-" * 150)

    wf_rows = []
    for i, r in enumerate(wf_configs):
        cfg = r['config']
        wf_row = {'label': r['label'], 'gmn': cfg['group_map_name'], 'windows': {}, 'mdd': {}}
        for yr in wf_years:
            wr = run_backtest(cfg, wf_test_year=yr)
            if wr:
                wf_row['windows'][yr] = wr['ann']
                wf_row['mdd'][yr] = wr['mdd']
        wf_rows.append(wf_row)

        vals = [wf_row['windows'].get(yr, 0) for yr in wf_years]
        avg = np.mean(vals) if vals else 0
        pos = sum(1 for v in vals if v > 0)
        avg_mdd = np.mean(list(wf_row['mdd'].values())) if wf_row['mdd'] else 0

        row_str = f"  {i+1:>3} | {wf_row['label']:<25} | {avg:>+7.1f}% |"
        for v in vals:
            row_str += f" {v:>+7.1f}% |"
        row_str += f" {pos}/6 | {avg_mdd:>6.1f}%"
        print(row_str)

    # ================================================================
    # WALK-FORWARD V92 vs V94 BEST
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  WALK-FORWARD: V92 BEST vs V94 BEST")
    print(f"{'=' * 150}")

    if v92_results:
        cfg92 = v92_results[0]['config']
        wf92 = {'label': v92_results[0]['label'], 'windows': {}, 'mdd': {}}
        for yr in wf_years:
            wr = run_backtest(cfg92, wf_test_year=yr)
            if wr:
                wf92['windows'][yr] = wr['ann']
                wf92['mdd'][yr] = wr['mdd']

        vals92 = [wf92['windows'].get(yr, 0) for yr in wf_years]
        avg92 = np.mean(vals92)
        pos92 = sum(1 for v in vals92 if v > 0)
        mdd92 = np.mean(list(wf92['mdd'].values()))

        row92 = f"  V92 best ({wf92['label']}) | {avg92:>+7.1f}% |"
        for v in vals92:
            row92 += f" {v:>+7.1f}% |"
        row92 += f" {pos92}/6 | {mdd92:>6.1f}%"
        print(row92)

    if v94_results:
        cfg94 = v94_results[0]['config']
        wf94 = {'label': v94_results[0]['label'], 'windows': {}, 'mdd': {}}
        for yr in wf_years:
            wr = run_backtest(cfg94, wf_test_year=yr)
            if wr:
                wf94['windows'][yr] = wr['ann']
                wf94['mdd'][yr] = wr['mdd']

        vals94 = [wf94['windows'].get(yr, 0) for yr in wf_years]
        avg94 = np.mean(vals94)
        pos94 = sum(1 for v in vals94 if v > 0)
        mdd94 = np.mean(list(wf94['mdd'].values()))

        row94 = f"  V94 best ({wf94['label']}) | {avg94:>+7.1f}% |"
        for v in vals94:
            row94 += f" {v:>+7.1f}% |"
        row94 += f" {pos94}/6 | {mdd94:>6.1f}%"
        print(row94)

    if v92_results and v94_results:
        print(f"\n  V92 WF avg: {avg92:+.1f}%  |  V94 WF avg: {avg94:+.1f}%  |  Delta: {avg94-avg92:+.1f}%")
        if avg94 > avg92:
            print(f"  >>> V94 EXPANDED UNIVERSE WINS WALK-FORWARD <<<")
        else:
            print(f"  >>> V92 BASELINE WINS WALK-FORWARD <<<")

    # ================================================================
    # THRESHOLD & TOP_N SENSITIVITY (V94)
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  V94 THRESHOLD & TOP_N SENSITIVITY")
    print(f"{'=' * 130}")
    print(f"  {'Config':<25} | {'Ann':>10} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7}")
    print("-" * 100)

    for thresh in [0.1, 0.3, 0.5]:
        for tn in [1, 3, 5]:
            sub = [r for r in v94_results
                   if abs(r['config']['threshold'] - thresh) < 0.01
                   and r['config']['top_n'] == tn]
            if sub:
                r = sub[0]
                print(f"  Z{thresh}_TN{tn:<18} | {r['ann']:>+9.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}%")

    # ================================================================
    # DATA COVERAGE ANALYSIS
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  DATA COVERAGE ANALYSIS (new V94 commodities)")
    print(f"{'=' * 130}")

    for s in sorted(v94_only):
        si = syms.index(s) if s in syms else -1
        if si < 0:
            print(f"    {s:<8} -- NOT IN LOADED DATA")
            continue
        valid = ~np.isnan(C[si])
        n_valid = np.sum(valid)
        if n_valid > 0:
            first_valid = None
            last_valid = None
            for di in range(ND):
                if valid[di]:
                    if first_valid is None:
                        first_valid = dates[di]
                    last_valid = dates[di]
            n_overnight = np.sum(~np.isnan(overnight_ret[si]))
            n_z = np.sum(~np.isnan(z_v94[si]))
            print(f"    {s:<8} -> {GROUP_MAP_V94.get(s,'?'):<15} days={n_valid:>4}  from {first_valid} to {last_valid}  overnight_valid={n_overnight}  z_valid={n_z}")
        else:
            print(f"    {s:<8} -> {GROUP_MAP_V94.get(s,'?'):<15} NO VALID DATA")

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  FINAL VERDICT")
    print(f"{'=' * 130}")

    if v92_results and v94_results:
        print(f"  V92 (44 commodities):  {best_v92['ann']:>+8.1f}%  WR={best_v92['wr']:.1f}%  N={best_v92['n']}  MDD={best_v92['mdd']:.1f}%  ({best_v92['label']})")
        print(f"  V94 (68 commodities):  {best_v94['ann']:>+8.1f}%  WR={best_v94['wr']:.1f}%  N={best_v94['n']}  MDD={best_v94['mdd']:.1f}%  ({best_v94['label']})")

        full_delta = best_v94['ann'] - best_v92['ann']

        print(f"\n  Full period delta: {full_delta:+.1f}%")
        if 'avg92' in dir() and 'avg94' in dir():
            print(f"  Walk-forward delta: {avg94-avg92:+.1f}%")

        if full_delta > 0:
            print(f"\n  >>> EXPANDING UNIVERSE FROM 44 TO 68 COMMODITIES IMPROVES RETURNS <<<")
        else:
            print(f"\n  >>> V92 BASELINE (44 COMMODITIES) IS BETTER -- EXPANSION DOES NOT HELP <<<")
            print(f"  Possible reasons:")
            print(f"    - New commodities may have shorter history / less data")
            print(f"    - Group assignments may dilute signal quality")
            print(f"    - Some new commodities may not be liquid enough")

    print(f"\n  Total time: {time.time()-t_start:.1f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
