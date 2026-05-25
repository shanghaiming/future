"""
Alpha Futures V88 -- OI-Confirmed Cross-Group Z-Score Momentum
==============================================================
V82 champion: D_zscore signal gives +3305% annual.
  Signal: z = (own_return - all_groups_avg) / all_groups_std; z < -0.5 -> buy.

V88 tests whether OI (open interest) CONFIRMS the z-score signal.

Hypotheses:
  1. z < -0.5 AND OI declining -> short covering / capitulation -> STRONGER buy
  2. z < -0.5 AND OI surging   -> new shorts entering        -> WEAKER signal
  3. OI change relative to group average OI change adds predictive power

Signals:
  A) z_score_only:          Baseline V82 (z < threshold, long-only, 1-day hold)
  B) z_and_oi_decline:      z < threshold AND OI_change < 0
  C) z_and_oi_extreme_decl: z < threshold AND OI_change < -2*std(OI_change_20d)
  D) z_and_vol_oi_ratio:    z < threshold AND (volume / OI) > median of 20d
  E) z_score_oi_combined:   Score = -z * (1 + OI_decline_factor)

Walk-forward: 6 windows (2020-2025), reset cash at test year start.
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

# -- Multipliers ---------------------------------------------------------
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

# -- Group map (same 44 commodities, 8 groups as V82) --------------------
GROUP_MAP = {}
for _s in ['rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi']:
    GROUP_MAP[_s] = 'ferrous'
for _s in ['cufi', 'alfi', 'znfi', 'nifi', 'pbfi', 'snfi', 'ssfi', 'sffi']:
    GROUP_MAP[_s] = 'nonferrous'
for _s in ['aufi', 'agfi']:
    GROUP_MAP[_s] = 'precious'
for _s in ['afi', 'mfi', 'yfi', 'pfi', 'cfi', 'csfi', 'rrfi', 'lrfi']:
    GROUP_MAP[_s] = 'oils'
for _s in ['scfi', 'mafi', 'bfi', 'fufi', 'pgfi', 'ebfi', 'fbfi']:
    GROUP_MAP[_s] = 'energy'
for _s in ['ppfi', 'vfi', 'egfi', 'srfi', 'tafi', 'fgfi', 'lfi']:
    GROUP_MAP[_s] = 'chemical'
for _s in ['whfi', 'apfi', 'cjfi', 'oifi', 'rmfi', 'srfi', 'cffi']:
    GROUP_MAP[_s] = 'soft'
for _s in ['jdfi', 'lhfi', 'pkfi']:
    GROUP_MAP[_s] = 'livestock'


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def main():
    print("=" * 120)
    print("Alpha Futures V88 -- OI-Confirmed Cross-Group Z-Score Momentum")
    print("=" * 120)

    # -- Load data --------------------------------------------------------
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # -- Build group membership -------------------------------------------
    gm_map = {}
    si_group = {}
    for si in range(NS):
        g = GROUP_MAP.get(syms[si])
        if g:
            gm_map.setdefault(g, []).append(si)
            si_group[si] = g

    trade_sis = [si for si in range(NS) if si in si_group]
    group_names = sorted(gm_map.keys())
    print(f"  Tradeable: {len(trade_sis)} commodities in {len(group_names)} groups")
    for gn in group_names:
        print(f"    {gn}: {len(gm_map[gn])} commodities")

    # -- Precompute 1-day returns -----------------------------------------
    print("\n[Signals] Computing 1-day returns...", flush=True)
    t0 = time.time()
    ret1 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            cn = C[si, di]
            cp = C[si, di - 1]
            if not np.isnan(cn) and not np.isnan(cp) and cp > 0:
                ret1[si, di] = (cn - cp) / cp

    # -- Precompute group-level signals -----------------------------------
    print("[Signals] Computing group-level aggregates...", flush=True)

    grp_total = {}
    for grp in group_names:
        arr = np.full(ND, np.nan)
        members = gm_map[grp]
        for di in range(1, ND):
            vals = [ret1[sk, di] for sk in members if not np.isnan(ret1[sk, di])]
            if vals:
                arr[di] = np.mean(vals)
        grp_total[grp] = arr

    all_groups_avg = np.full(ND, np.nan)
    all_groups_std = np.full(ND, np.nan)
    for di in range(1, ND):
        vals = [grp_total[g][di] for g in group_names if not np.isnan(grp_total[g][di])]
        if len(vals) >= 2:
            all_groups_avg[di] = np.mean(vals)
            all_groups_std[di] = np.std(vals)

    # -- Precompute z-score -----------------------------------------------
    print("[Signals] Computing z-scores...", flush=True)
    z_score = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        aga = all_groups_avg[di]
        ags = all_groups_std[di]
        if np.isnan(aga) or np.isnan(ags) or ags < 1e-8:
            continue
        for si in trade_sis:
            own = ret1[si, di]
            if np.isnan(own):
                continue
            z_score[si, di] = (own - aga) / ags

    # -- Precompute OI signals --------------------------------------------
    print("[Signals] Computing OI signals...", flush=True)

    # OI percentage change
    oi_change = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            oi_now = OI[si, di]
            oi_prev = OI[si, di - 1]
            if (not np.isnan(oi_now) and not np.isnan(oi_prev)
                    and oi_prev > 0):
                oi_change[si, di] = (oi_now - oi_prev) / oi_prev

    # Rolling 20d std of OI change
    oi_change_20d_std = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            window = oi_change[si, di - 20:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 5:
                oi_change_20d_std[si, di] = np.std(valid)

    # Volume / OI ratio
    vol_oi_ratio = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            v = V[si, di]
            oi = OI[si, di]
            if (not np.isnan(v) and not np.isnan(oi)
                    and oi > 0 and v > 0):
                vol_oi_ratio[si, di] = v / oi

    # Rolling 20d median of vol/OI ratio (for threshold in signal D)
    vol_oi_median_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            window = vol_oi_ratio[si, di - 20:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 5:
                vol_oi_median_20d[si, di] = np.median(valid)

    # OI decline factor for signal E: normalised OI change relative to 20d std
    oi_decline_factor = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            oc = oi_change[si, di]
            ocs = oi_change_20d_std[si, di]
            if not np.isnan(oc) and not np.isnan(ocs) and ocs > 1e-8:
                # Negative oi_change -> positive factor (short covering)
                oi_decline_factor[si, di] = -oc / ocs

    print(f"  Signals computed ({time.time()-t0:.1f}s)")

    # =====================================================================
    # BACKTEST ENGINE
    # =====================================================================
    def run_backtest(config, wf_test_year=None):
        """
        Config keys:
            signal: 'A_zscore_only' | 'B_z_oi_decline' | 'C_z_oi_extreme_decl'
                    | 'D_z_vol_oi' | 'E_z_oi_combined'
            threshold: float  (z-score cutoff, e.g. 0.3, 0.5, 0.7)
            top_n: 1 | 3
            comm: float
        """
        sig_type = config['signal']
        threshold = config['threshold']
        top_n = config['top_n']
        comm = config.get('comm', COMM)

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

            # -- Close positions held 1 day --------------------------------
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
                        'dir': pos['dir'],
                    })
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            # -- Generate signals ------------------------------------------
            candidates = []

            for si in trade_sis:
                z = z_score[si, di]
                if np.isnan(z):
                    continue
                cc = C[si, di]
                if np.isnan(cc) or cc <= 0:
                    continue
                if any(p['si'] == si for p in positions):
                    continue

                # All signals require z < -threshold
                if z >= -threshold:
                    continue

                if sig_type == 'A_zscore_only':
                    # Baseline V82: pure z-score
                    score = -z
                    candidates.append((si, score, 1, syms[si]))

                elif sig_type == 'B_z_oi_decline':
                    # z < threshold AND OI declining (short covering)
                    oc = oi_change[si, di]
                    if np.isnan(oc) or oc >= 0:
                        continue
                    score = -z  # pure z-score ranking; OI just filters
                    candidates.append((si, score, 1, syms[si]))

                elif sig_type == 'C_z_oi_extreme_decl':
                    # z < threshold AND OI change < -2*std(20d)
                    oc = oi_change[si, di]
                    ocs = oi_change_20d_std[si, di]
                    if np.isnan(oc) or np.isnan(ocs) or ocs < 1e-8:
                        continue
                    if oc >= -2.0 * ocs:
                        continue
                    score = -z
                    candidates.append((si, score, 1, syms[si]))

                elif sig_type == 'D_z_vol_oi':
                    # z < threshold AND vol/OI > 20d median (high activity)
                    vor = vol_oi_ratio[si, di]
                    vom = vol_oi_median_20d[si, di]
                    if np.isnan(vor) or np.isnan(vom):
                        continue
                    if vor <= vom:
                        continue
                    score = -z
                    candidates.append((si, score, 1, syms[si]))

                elif sig_type == 'E_z_oi_combined':
                    # Combined score = -z * (1 + OI_decline_factor)
                    odf = oi_decline_factor[si, di]
                    if np.isnan(odf):
                        odf = 0.0
                    # Clamp OI factor so it doesn't dominate
                    odf_clamped = max(-2.0, min(2.0, odf))
                    score = -z * (1.0 + odf_clamped)
                    candidates.append((si, score, 1, syms[si]))

            if not candidates:
                continue

            # Sort by score descending
            candidates.sort(key=lambda x: -x[1])

            # Open positions (up to top_n slots)
            n_slots = top_n - len(positions)
            for si, score, direction, sym in candidates[:max(0, n_slots)]:
                c = C[si, di]
                if np.isnan(c) or c <= 0:
                    continue
                mult = MULT.get(sym, DEF_MULT)
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
                    'lots': lots, 'dir': direction, 'sym': sym,
                })

        # Close remaining
        for pos in positions:
            ae = ND - 1
            cn = C[pos['si'], ae]
            if np.isnan(cn) or cn <= 0:
                cn = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = cn * mult * abs(pos['lots'])
            cash += mkt_val - mkt_val * comm

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

    # =====================================================================
    # BUILD CONFIGURATIONS
    # =====================================================================
    print("\n[Sweep] Building configurations...", flush=True)
    configs = []
    cid = 0

    sig_types = ['A_zscore_only', 'B_z_oi_decline', 'C_z_oi_extreme_decl',
                 'D_z_vol_oi', 'E_z_oi_combined']
    thresholds = [0.3, 0.5, 0.7]
    top_ns = [1, 3]

    for sig in sig_types:
        for thresh in thresholds:
            for tn in top_ns:
                cid += 1
                label = f"{sig}_Z{thresh}_TN{tn}"
                configs.append({
                    'id': cid, 'signal': sig,
                    'threshold': thresh, 'top_n': tn, 'comm': COMM,
                    'label': label,
                })

    print(f"  Total configs: {len(configs)}")

    # =====================================================================
    # RUN FULL-PERIOD BACKTEST
    # =====================================================================
    print("\n[Backtest] Running full-period sweep...", flush=True)
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

    # Print top 25
    print(f"\n{'=' * 120}")
    print("  FULL-PERIOD RESULTS (Top 25)")
    print(f"{'=' * 120}")
    print(f"  {'#':>3} | {'Label':<40} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7}")
    print("-" * 100)
    for i, r in enumerate(results[:25]):
        print(f"  {i+1:>3} | {r['label']:<40} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}%")

    # =====================================================================
    # SIGNAL COMPARISON (full period)
    # =====================================================================
    print(f"\n{'=' * 120}")
    print("  SIGNAL COMPARISON (Best per signal type, full period)")
    print(f"{'=' * 120}")
    print(f"  {'Signal':<25} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | Best Config")
    print("-" * 120)

    best_per_sig = {}
    for r in results:
        s = r['config']['signal']
        if s not in best_per_sig:
            best_per_sig[s] = r

    for sig in sig_types:
        if sig in best_per_sig:
            b = best_per_sig[sig]
            print(f"  {sig:<25} | {b['ann']:>+8.1f}% | {b['wr']:>5.1f}% | {b['n']:>5} | {b['avg_pnl']:>+6.3f}% | {b['mdd']:>6.1f}% | {b['label']}")

    # Alpha vs baseline A (pure z-score)
    base_a = best_per_sig.get('A_zscore_only')
    if base_a:
        print(f"\n  A_zscore_only Baseline: {base_a['ann']:>+8.1f}%")
        for sig in ['B_z_oi_decline', 'C_z_oi_extreme_decl', 'D_z_vol_oi', 'E_z_oi_combined']:
            if sig in best_per_sig:
                diff = best_per_sig[sig]['ann'] - base_a['ann']
                tag = "BETTER" if diff > 0 else "WORSE"
                print(f"  {sig:<25} {best_per_sig[sig]['ann']:>+8.1f}%  ({tag} {diff:>+.1f}% vs baseline)")

    # =====================================================================
    # WALK-FORWARD (Top 10 + best per signal)
    # =====================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    wf_configs = list(results[:10])
    for sig in sig_types:
        if sig in best_per_sig:
            r = best_per_sig[sig]
            if r['config'] not in [w['config'] for w in wf_configs]:
                wf_configs.append(r)

    print(f"\n{'=' * 150}")
    print(f"  WALK-FORWARD ({len(wf_configs)} configs)")
    print(f"{'=' * 150}")

    header = f"  {'#':>3} | {'Config':<40} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>7} |"
    header += f" {'Pos':>4} | {'MDD':>7}"
    print(header)
    print("-" * 150)

    wf_rows = []
    for i, r in enumerate(wf_configs):
        cfg = r['config']
        wf_row = {'label': cfg['label'], 'signal': cfg['signal'], 'windows': {}, 'mdd': {}}
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

        row_str = f"  {i+1:>3} | {wf_row['label']:<40} | {avg:>+7.1f}% |"
        for v in vals:
            row_str += f" {v:>+7.1f}% |"
        row_str += f" {pos}/6 | {avg_mdd:>6.1f}%"
        print(row_str)

    # =====================================================================
    # WF COMPARISON PER SIGNAL
    # =====================================================================
    print(f"\n{'=' * 130}")
    print("  WALK-FORWARD COMPARISON (Best per signal type)")
    print(f"{'=' * 130}")
    header2 = f"  {'Signal':<25} | {'WF Avg':>8} |"
    for yr in wf_years:
        header2 += f" {yr:>7} |"
    header2 += f" {'Pos':>4}"
    print(header2)
    print("-" * 130)

    for sig in sig_types:
        wf_match = [w for w in wf_rows if w['signal'] == sig]
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            avg = np.mean(vals) if vals else 0
            pos = sum(1 for v in vals if v > 0)
            row_str = f"  {sig:<25} | {avg:>+7.1f}% |"
            for v in vals:
                row_str += f" {v:>+7.1f}% |"
            row_str += f" {pos}/6"
            print(row_str)

    # =====================================================================
    # FINAL VERDICT
    # =====================================================================
    print(f"\n{'=' * 120}")
    print("  FINAL VERDICT -- Does OI Confirmation Improve Z-Score?")
    print(f"{'=' * 120}")

    if base_a:
        print(f"  Baseline A (z-score only):    {base_a['ann']:>+8.1f}%  WR {base_a['wr']:>5.1f}%  N {base_a['n']:>5}  | {base_a['label']}")

        # Find best OI-confirmed signal
        oi_sigs = ['B_z_oi_decline', 'C_z_oi_extreme_decl', 'D_z_vol_oi', 'E_z_oi_combined']
        best_oi = None
        for sig in oi_sigs:
            if sig in best_per_sig:
                b = best_per_sig[sig]
                if best_oi is None or b['ann'] > best_oi['ann']:
                    best_oi = b

        if best_oi:
            diff = best_oi['ann'] - base_a['ann']
            print(f"  Best OI-confirmed signal:     {best_oi['ann']:>+8.1f}%  WR {best_oi['wr']:>5.1f}%  N {best_oi['n']:>5}  | {best_oi['label']}")
            if diff > 0:
                print(f"  >>> OI CONFIRMATION ADDS +{diff:.1f}% ANNUAL ALPHA <<<")
            else:
                print(f"  >>> OI CONFIRMATION DOES NOT ADD ALPHA ({diff:+.1f}%) <<<")
                print(f"  >>> PURE Z-SCORE REMAINS STRONGEST <<<")

        # Per-signal verdict
        print(f"\n  Per-signal breakdown:")
        for sig in oi_sigs:
            if sig in best_per_sig:
                b = best_per_sig[sig]
                diff = b['ann'] - base_a['ann']
                tag = "ADD" if diff > 0 else "NO "
                # Count trades to show filtering effect
                print(f"    {tag} {sig:<25} {b['ann']:>+8.1f}%  (N={b['n']:>5} vs {base_a['n']:>5}, {diff:+.1f}%)")

    print(f"\n  Total time: {time.time()-t_start:.1f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
