"""
Alpha Futures V77 -- Multi-Group Overlapping Memberships
========================================================
V74 champion: +2185% with extended groups (44 commodities, 8 groups), LB=1, 1-day hold.
All vol-adaptive and risk-management enhancements FAILED (V75, V76).

V77 IDEA: Instead of single group membership, give each commodity MULTIPLE overlapping
group memberships. When the same commodity gets a strong signal from MULTIPLE groups
simultaneously, it's a stronger signal.

Example: rbfi belongs to 'ferrous' (primary) AND 'steel_chain' (production chain).
If ferrous group says rbfi should catch up AND steel_chain also says rbfi should catch up
=> double confirmation => stronger signal.

Configurations:
  A: Single group (V74 baseline -- 44 commodities, 8 primary groups)
  B: Multi-group mean score (average all group divergences)
  C: Multi-group agreement (ALL groups must agree, all positive divergence)
  D: Multi-group weighted (weight by inverse group size -- smaller groups get more weight)

Sweep: threshold x top_n x mode, WF 6 windows.
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

MULT = {'agfi': 15, 'alfi': 5, 'aufi': 1000, 'bufi': 10, 'cufi': 5, 'fufi': 10,
        'rbfi': 10, 'znfi': 5, 'nifi': 1, 'hcfi': 10, 'spfi': 10, 'ssfi': 5,
        'sffi': 5, 'smfi': 5, 'pbfi': 5, 'snfi': 1, 'afi': 10, 'bfi': 10,
        'cffi': 5, 'cfi': 10, 'csfi': 10, 'ebfi': 5, 'egfi': 10, 'fbfi': 500,
        'ifi': 100, 'jfi': 100, 'jmfi': 60, 'lfi': 5, 'mfi': 10, 'pgfi': 20,
        'ppfi': 5, 'vfi': 5, 'yfi': 10, 'pfi': 10, 'jdfi': 5, 'lhfi': 16,
        'pkfi': 5, 'rrfi': 20, 'lrfi': 20, 'whfi': 20, 'cjfi': 10, 'mafi': 10,
        'apfi': 10, 'fgfi': 20, 'oifi': 10, 'rmfi': 10, 'srfi': 10, 'tafi': 5,
        'scfi': 1000, 'lufi': 10, 'bcfi': 5, 'nrfi': 1, 'lgfi': 20, 'brfi': 5}
DEF_MULT = 10
COMM = 0.0003

# ── V74 Champion: Primary industry groups (44 commodities, 8 groups) ───
GROUPS_PRIMARY = {
    'ferrous':    ['rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi'],              # 5
    'nonferrous': ['cufi', 'alfi', 'znfi', 'nifi', 'pbfi', 'snfi',
                   'ssfi', 'sffi'],                                      # 8
    'precious':   ['aufi', 'agfi'],                                      # 2
    'oils':       ['afi', 'mfi', 'yfi', 'pfi', 'cfi', 'csfi',
                   'rrfi', 'lrfi'],                                      # 8
    'energy':     ['scfi', 'mafi', 'bfi', 'fufi', 'pgfi',
                   'ebfi', 'fbfi'],                                      # 7
    'chemical':   ['ppfi', 'vfi', 'egfi', 'srfi', 'tafi',
                   'fgfi', 'lfi'],                                       # 7
    'soft':       ['whfi', 'apfi', 'cjfi', 'oifi', 'rmfi',
                   'cffi', 'srfi'],                                      # 7
    'livestock':  ['jdfi', 'lhfi', 'pkfi'],                             # 3
}

# ── Cross-industry supply chain groups (overlapping) ───────────────────
GROUPS_CHAIN = {
    'steel_chain': ['rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi', 'mfi'],  # steel production chain
    'oil_chain':   ['scfi', 'bfi', 'fufi', 'mafi', 'ppfi', 'egfi',
                    'pgfi', 'yfi', 'afi', 'pfi'],                     # petroleum chain
    'feed_chain':  ['mfi', 'yfi', 'afi', 'pfi', 'cfi', 'csfi'],      # feed/meal chain
}

# ── Volatility groups (correlated movement, overlapping) ───────────────
GROUPS_VOL = {
    'high_vol': ['ifi', 'jfi', 'jmfi', 'scfi', 'egfi', 'srfi', 'fufi'],
    'low_vol':  ['cfi', 'csfi', 'alfi', 'znfi', 'lfi', 'nifi'],
}

# All overlapping groups combined (primary + chain + vol)
GROUPS_MULTI = {}
GROUPS_MULTI.update(GROUPS_PRIMARY)
GROUPS_MULTI.update(GROUPS_CHAIN)
GROUPS_MULTI.update(GROUPS_VOL)

# Single-group map for V74 baseline (one membership per commodity, first match wins)
GROUP_MAP_SINGLE = {}
for gname, members in GROUPS_PRIMARY.items():
    for s in members:
        if s not in GROUP_MAP_SINGLE:
            GROUP_MAP_SINGLE[s] = gname


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def main():
    print("=" * 100)
    print("Alpha Futures V77 -- Multi-Group Overlapping Memberships")
    print("=" * 100)

    # ── Load data ──────────────────────────────────────────────────
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    sym_to_si = {syms[si]: si for si in range(NS)}

    # ── Precompute momentum (LB=1) ─────────────────────────────────
    print("\n[Signals] Computing momentum (LB=1)...", flush=True)
    t0 = time.time()

    mom = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            cn = C[si, di]
            cp = C[si, di - 1]
            if not np.isnan(cn) and not np.isnan(cp) and cp > 0:
                mom[si, di] = (cn - cp) / cp

    # ── Build group index maps ─────────────────────────────────────
    def build_group_indices(groups_dict):
        grp_idx = {}
        for gname, members in groups_dict.items():
            sis = [sym_to_si[s] for s in members if s in sym_to_si]
            if len(sis) >= 2:
                grp_idx[gname] = sis
        return grp_idx

    grp_idx_multi = build_group_indices(GROUPS_MULTI)
    grp_idx_primary = build_group_indices(GROUPS_PRIMARY)

    # Multi-membership map: si -> list of group names
    si_to_groups = {}
    for si in range(NS):
        s = syms[si]
        for gname, members in GROUPS_MULTI.items():
            if s in members and gname in grp_idx_multi:
                si_to_groups.setdefault(si, []).append(gname)

    # Single-membership map for V74 baseline
    si_to_group_single = {}
    for si in range(NS):
        g = GROUP_MAP_SINGLE.get(syms[si])
        if g and g in grp_idx_primary:
            si_to_group_single[si] = g

    # Count unique commodities in primary groups
    primary_comms = set()
    for members in GROUPS_PRIMARY.values():
        for s in members:
            if s in sym_to_si:
                primary_comms.add(s)

    print(f"  Momentum computed ({time.time()-t0:.1f}s)")
    print(f"  Primary groups: {len(grp_idx_primary)} groups, {len(primary_comms)} commodities")
    for gn, sis in grp_idx_primary.items():
        print(f"    {gn}: {[syms[s] for s in sis]}")
    print(f"  Multi-groups: {len(grp_idx_multi)} groups total")
    for gn in ['steel_chain', 'oil_chain', 'feed_chain', 'high_vol', 'low_vol']:
        if gn in grp_idx_multi:
            print(f"    {gn}: {[syms[s] for s in grp_idx_multi[gn]]}")
    print(f"  Commodities with multi-group membership: {len(si_to_groups)}")
    # Show commodities with 3+ memberships
    multi3 = [(syms[si], gnames) for si, gnames in si_to_groups.items() if len(gnames) >= 3]
    if multi3:
        print(f"  Commodities with 3+ group memberships ({len(multi3)}):")
        for sym, gnames in multi3:
            print(f"    {sym}: {gnames}")

    # ── Precompute group momentum for each group ───────────────────
    print("\n[Signals] Computing group momentum for all groups...", flush=True)
    t0 = time.time()

    # grp_mom[gname] is array [NS, ND] — group average excluding each member
    grp_mom = {}
    for gname, sis in grp_idx_multi.items():
        gm = np.full((NS, ND), np.nan)
        # Vectorized: for each day, compute mean of all members, then adjust per-member
        for di in range(1, ND):
            # Collect valid momenta for all members
            n_members = len(sis)
            valid_vals = []
            valid_si = []
            for si in sis:
                v = mom[si, di]
                if not np.isnan(v):
                    valid_vals.append(v)
                    valid_si.append(si)
            if len(valid_vals) < 2:
                continue
            total = sum(valid_vals)
            cnt = len(valid_vals)
            for idx, si in enumerate(valid_si):
                # Group mean excluding this member
                gm[si, di] = (total - valid_vals[idx]) / (cnt - 1)
        grp_mom[gname] = gm

    print(f"  Group momentum computed ({time.time()-t0:.1f}s)")

    # ── Scoring modes ──────────────────────────────────────────────
    def score_single(si, di):
        """Mode A: V74 baseline -- single primary group divergence."""
        g = si_to_group_single.get(si)
        if g is None:
            return np.nan
        own = mom[si, di]
        gm = grp_mom[g][si, di]
        if np.isnan(own) or np.isnan(gm):
            return np.nan
        return gm - own

    def score_multi_mean(si, di):
        """Mode B: mean divergence across ALL groups the commodity belongs to."""
        groups = si_to_groups.get(si)
        if not groups:
            return np.nan
        own = mom[si, di]
        if np.isnan(own):
            return np.nan
        divs = []
        for g in groups:
            gm = grp_mom[g][si, di]
            if not np.isnan(gm):
                divs.append(gm - own)
        if not divs:
            return np.nan
        return np.mean(divs)

    def score_multi_agree(si, di):
        """Mode C: ALL groups must agree (all positive divergence).
        Return mean divergence if all agree, else 0."""
        groups = si_to_groups.get(si)
        if not groups:
            return np.nan
        own = mom[si, di]
        if np.isnan(own):
            return np.nan
        divs = []
        for g in groups:
            gm = grp_mom[g][si, di]
            if np.isnan(gm):
                continue
            divs.append(gm - own)
        if not divs:
            return np.nan
        # All must be positive (agree that commodity is lagging)
        if all(d > 0 for d in divs):
            return np.mean(divs)
        return 0.0

    def score_multi_weighted(si, di):
        """Mode D: weighted by inverse group size (smaller groups get more weight)."""
        groups = si_to_groups.get(si)
        if not groups:
            return np.nan
        own = mom[si, di]
        if np.isnan(own):
            return np.nan
        w_divs = []
        w_total = 0
        for g in groups:
            gm = grp_mom[g][si, di]
            if np.isnan(gm):
                continue
            gsize = len(grp_idx_multi[g])
            w = 1.0 / gsize
            w_divs.append(w * (gm - own))
            w_total += w
        if not w_divs or w_total == 0:
            return np.nan
        return sum(w_divs) / w_total

    scoring_fns = {
        'A_single':   score_single,
        'B_mean':     score_multi_mean,
        'C_agree':    score_multi_agree,
        'D_weighted': score_multi_weighted,
    }

    # ── Backtest engine ────────────────────────────────────────────
    def run_backtest(mode, threshold, top_n, wf_test_year=None):
        score_fn = scoring_fns[mode]
        if mode == 'A_single':
            tradeable_sis = list(si_to_group_single.keys())
        else:
            tradeable_sis = list(si_to_groups.keys())

        # Date range setup
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
            end_di = test_end_di
        else:
            test_start_di = MIN_TRAIN
            end_di = ND

        cash = float(CASH0)
        positions = []
        trades = []

        for di in range(MIN_TRAIN, end_di):
            # Reset cash at start of test window (WF only)
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                positions = []

            # ── Close positions entered yesterday (1-day hold) ─────
            closed = []
            for pos in positions:
                if di - pos['entry_di'] >= 1:
                    cn = C[pos['si'], di]
                    if np.isnan(cn) or cn <= 0:
                        cn = pos['entry']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    mkt_val = cn * mult * pos['lots']
                    cash += mkt_val - mkt_val * COMM
                    pnl = (cn - pos['entry']) * mult * pos['lots']
                    invested = pos['entry'] * mult * pos['lots']
                    pnl_pct = pnl / invested * 100 if invested > 0 else 0
                    trades.append({
                        'pnl_pct': pnl_pct,
                        'di': pos['entry_di'],
                        'year': dates[di].year if di < ND else dates[-1].year,
                    })
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            # ── Score candidates ───────────────────────────────────
            candidates = []
            for si in tradeable_sis:
                if np.isnan(C[si, di]) or C[si, di] <= 0:
                    continue
                if any(p['si'] == si for p in positions):
                    continue
                score = score_fn(si, di)
                if np.isnan(score) or score <= threshold:
                    continue
                candidates.append((si, score))

            if not candidates:
                continue

            # Sort by score descending
            candidates.sort(key=lambda x: -x[1])

            # Open positions
            n_slots = top_n - len(positions)
            for si, score in candidates[:n_slots]:
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
                    'lots': lots, 'sym': syms[si],
                })

        # Close remaining positions at end
        for pos in positions:
            ae = ND - 1
            cn = C[pos['si'], ae]
            if np.isnan(cn) or cn <= 0:
                cn = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = cn * mult * pos['lots']
            cash += mkt_val - mkt_val * COMM

        # Results
        if wf_test_year is not None:
            n_days_test = test_end_di - test_start_di
        else:
            n_days_test = ND - MIN_TRAIN

        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)

        return {
            'ann': ann, 'wr': wr, 'n': n_trades,
            'final_cash': cash, 'n_days': n_days_test,
        }

    # ── Sweep configurations ───────────────────────────────────────
    print("\n[Sweep] Testing configurations...", flush=True)

    configs = []
    for mode in ['A_single', 'B_mean', 'C_agree', 'D_weighted']:
        for thresh in [0.001, 0.003, 0.005, 0.01]:
            for tn in [1, 3]:
                label = f"{mode}_T{thresh}_TN{tn}"
                configs.append({
                    'mode': mode,
                    'threshold': thresh,
                    'top_n': tn,
                    'label': label,
                })

    print(f"  Total configs: {len(configs)}")

    results = []
    t_sweep_start = time.time()
    for i, cfg in enumerate(configs):
        r = run_backtest(cfg['mode'], cfg['threshold'], cfg['top_n'])
        if r:
            r['config'] = cfg
            r['label'] = cfg['label']
            results.append(r)
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{len(configs)} done ({time.time()-t_sweep_start:.1f}s)", flush=True)

    results.sort(key=lambda x: -x['ann'])

    # ── Print full-period top 15 ───────────────────────────────────
    print("\n" + "=" * 100)
    print("  FULL-PERIOD RESULTS (Top 15)")
    print("=" * 100)
    print(f"  {'#':>3} | {'Label':<40} | {'Ann':>10} | {'WR':>6} | {'N':>5}")
    print("-" * 80)
    for i, r in enumerate(results[:15]):
        print(f"  {i+1:>3} | {r['label']:<40} | {r['ann']:>+9.1f}% | {r['wr']:>5.1f}% | {r['n']:>5}")

    # ── Mode comparison ────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  MODE COMPARISON (Best config per mode)")
    print("=" * 100)
    for mode in ['A_single', 'B_mean', 'C_agree', 'D_weighted']:
        mode_results = [r for r in results if r['config']['mode'] == mode]
        if mode_results:
            best = mode_results[0]
            print(f"  {mode:<15} | Best Ann: {best['ann']:>+9.1f}% | WR: {best['wr']:>5.1f}% | {best['label']}")
            for j, r in enumerate(mode_results[:3]):
                print(f"    #{j+1}: {r['ann']:>+9.1f}%  WR={r['wr']:.1f}%  N={r['n']}  {r['label']}")

    # ── Walk-forward for top 10 ────────────────────────────────────
    print("\n" + "=" * 100)
    print("  WALK-FORWARD (Top 10 configs)")
    print("=" * 100)

    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]
    wf_results = []
    for i, r in enumerate(results[:10]):
        cfg = r['config']
        wf_row = {'label': cfg['label'], 'mode': cfg['mode'], 'windows': {}}
        for yr in wf_years:
            wr = run_backtest(cfg['mode'], cfg['threshold'], cfg['top_n'], wf_test_year=yr)
            if wr:
                wf_row['windows'][yr] = wr['ann']
        wf_results.append(wf_row)
        print(f"  WF {i+1}/10 done: {cfg['label']}", flush=True)

    # Print WF table
    print(f"\n  {'#':>3} | {'Config':<40} | {'Avg':>8} |", end="")
    for yr in wf_years:
        print(f"  {yr:>7} |", end="")
    print(f"  {'Pos':>4}")
    print("-" * 130)
    for i, wf in enumerate(wf_results):
        vals = [wf['windows'].get(yr, 0) for yr in wf_years]
        avg = np.mean(vals) if vals else 0
        pos = sum(1 for v in vals if v > 0)
        print(f"  {i+1:>3} | {wf['label']:<40} | {avg:>+7.1f}% |", end="")
        for v in vals:
            print(f"  {v:>+7.1f}% |", end="")
        print(f"  {pos}/6")

    # ── Key findings ───────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  KEY FINDINGS")
    print("=" * 100)

    best_single = [r for r in results if r['config']['mode'] == 'A_single']
    best_multi  = [r for r in results if r['config']['mode'] != 'A_single']

    if best_single:
        print(f"  V74 baseline (A_single best): {best_single[0]['ann']:>+9.1f}%  {best_single[0]['label']}")
    if best_multi:
        print(f"  Best multi-group mode:         {best_multi[0]['ann']:>+9.1f}%  {best_multi[0]['label']}")

    if best_single and best_multi:
        diff = best_multi[0]['ann'] - best_single[0]['ann']
        verdict = "BEATS" if diff > 0 else "DOES NOT BEAT"
        print(f"\n  Multi-group {verdict} V74 baseline by {diff:+.1f}% annual")

    # Per-mode best vs V74
    if best_single:
        v74_ann = best_single[0]['ann']
        for mode in ['B_mean', 'C_agree', 'D_weighted']:
            mode_best = [r for r in results if r['config']['mode'] == mode]
            if mode_best:
                diff = mode_best[0]['ann'] - v74_ann
                tag = "WIN" if diff > 0 else "LOSS"
                print(f"  {mode:<15} vs V74: {diff:>+9.1f}%  [{tag}]  {mode_best[0]['label']}")

    # WF comparison
    if len(wf_results) >= 2:
        print("\n  Walk-forward comparison (top 5):")
        for wf in wf_results[:5]:
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            avg = np.mean(vals) if vals else 0
            pos = sum(1 for v in vals if v > 0)
            print(f"    {wf['label']:<40} WF-Avg={avg:>+8.1f}%  Pos={pos}/6")

    print(f"\n  Total time: {time.time()-t_start:.1f}s")
    print("=" * 100)


if __name__ == '__main__':
    main()
