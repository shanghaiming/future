"""
Alpha Futures V87 -- Fine-Grained Sweep of V82 Signal D (Cross-Group Z-Score)
=============================================================================
V82 champion: D_zscore_Z0.5_TN3 gave +3305% annual, 6/6 WF positive.

V87 does a fine-grained sweep of z_threshold and top_n to find the true
optimum for Signal D (cross-group z-score momentum).

Signal D (same as V82):
  - Compute 1-day close-to-close returns
  - Compute group averages (8 groups, 44 commodities)
  - z = (own_return - all_groups_avg) / all_groups_std
  - When z < -threshold -> buy (long-only)
  - 1-day hold, close-to-close entry/exit

Sweep:
  z_threshold: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
  top_n: [1, 2, 3, 4, 5]
  = 65 configs

Walk-forward: top 15 configs across 6 years (2020-2025).
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

# ── Group map (same as V82) ─────────────────────────────────────────
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
    print("Alpha Futures V87 -- Fine-Grained Sweep of V82 Signal D (Cross-Group Z-Score)")
    print("=" * 120)

    # ── Load data ────────────────────────────────────────────────────
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ── Build group membership ───────────────────────────────────────
    gm_map = {}           # group_name -> [si, ...]
    si_group = {}         # si -> group_name
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

    # ── Precompute group-level signals ───────────────────────────────
    print("[Signals] Computing group-level aggregates...", flush=True)

    # group_total_avg[group_name] -> array[ND] = average return of that group
    grp_total = {}
    for grp in group_names:
        arr = np.full(ND, np.nan)
        members = gm_map[grp]
        for di in range(1, ND):
            vals = [ret1[sk, di] for sk in members if not np.isnan(ret1[sk, di])]
            if vals:
                arr[di] = np.mean(vals)
        grp_total[grp] = arr

    # all_groups_avg[di] = grand mean of all group averages
    all_groups_avg = np.full(ND, np.nan)
    all_groups_std = np.full(ND, np.nan)
    for di in range(1, ND):
        vals = [grp_total[g][di] for g in group_names if not np.isnan(grp_total[g][di])]
        if len(vals) >= 2:
            all_groups_avg[di] = np.mean(vals)
            all_groups_std[di] = np.std(vals)

    print(f"  Signals computed ({time.time()-t0:.1f}s)")

    # ── Precompute z-scores for all commodities and days ─────────────
    print("[Signals] Precomputing z-scores...", flush=True)
    t1 = time.time()
    # z_score[si, di] for tradeable commodities
    z_score = np.full((NS, ND), np.nan)
    for si in trade_sis:
        for di in range(1, ND):
            aga = all_groups_avg[di]
            ags = all_groups_std[di]
            own = ret1[si, di]
            if np.isnan(aga) or np.isnan(ags) or ags < 1e-8 or np.isnan(own):
                continue
            z_score[si, di] = (own - aga) / ags
    print(f"  Z-scores computed ({time.time()-t1:.1f}s)")

    # ══════════════════════════════════════════════════════════════════
    # BACKTEST ENGINE (Signal D only)
    # ══════════════════════════════════════════════════════════════════
    def run_backtest(z_threshold, top_n, wf_test_year=None):
        """
        Signal D: z = (own_return - all_groups_avg) / all_groups_std
        When z < -threshold -> buy (long-only)
        1-day hold, close-to-close, long-only
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

            # ── Signal D: z-score ────────────────────────────────────
            aga = all_groups_avg[di]
            ags = all_groups_std[di]
            if np.isnan(aga) or np.isnan(ags) or ags < 1e-8:
                continue

            candidates = []  # (si, score, sym)
            for si in trade_sis:
                z = z_score[si, di]
                if np.isnan(z):
                    continue
                cc = C[si, di]
                if np.isnan(cc) or cc <= 0:
                    continue
                if any(p['si'] == si for p in positions):
                    continue
                if z < -z_threshold:
                    score = -z
                    candidates.append((si, score, syms[si]))

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
    # SWEEP CONFIGURATIONS
    # ══════════════════════════════════════════════════════════════════
    z_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
    top_ns = [1, 2, 3, 4, 5]

    configs = []
    for zt in z_thresholds:
        for tn in top_ns:
            label = f"D_zscore_Z{zt}_TN{tn}"
            configs.append({
                'z_threshold': zt,
                'top_n': tn,
                'label': label,
            })

    print(f"\n[Sweep] {len(configs)} configurations: z_threshold in {z_thresholds}, top_n in {top_ns}")

    # ══════════════════════════════════════════════════════════════════
    # RUN FULL-PERIOD BACKTEST
    # ══════════════════════════════════════════════════════════════════
    print("\n[Backtest] Running full-period sweep...", flush=True)
    t_sweep_start = time.time()
    results = []
    for i, cfg in enumerate(configs):
        r = run_backtest(cfg['z_threshold'], cfg['top_n'])
        if r:
            r['config'] = cfg
            r['label'] = cfg['label']
            results.append(r)
        if (i + 1) % 10 == 0 or i == len(configs) - 1:
            print(f"  ... {i+1}/{len(configs)} done ({time.time()-t_sweep_start:.1f}s)", flush=True)

    results.sort(key=lambda x: -x['ann'])

    # Print top 30
    print(f"\n{'=' * 140}")
    print("  FULL-PERIOD RESULTS (Top 30 of 65 configs)")
    print(f"{'=' * 140}")
    print(f"  {'#':>3} | {'Label':<30} | {'Z_thr':>6} | {'TopN':>5} | {'Ann':>10} | {'WR':>6} | {'N':>6} | {'AvgPnL':>8} | {'MDD':>8}")
    print("-" * 140)
    for i, r in enumerate(results[:30]):
        cfg = r['config']
        print(f"  {i+1:>3} | {r['label']:<30} | {cfg['z_threshold']:>6.1f} | {cfg['top_n']:>5} | {r['ann']:>+9.1f}% | {r['wr']:>5.1f}% | {r['n']:>6} | {r['avg_pnl']:>+7.3f}% | {r['mdd']:>7.1f}%")

    # ── Heatmap-style table: z_threshold x top_n ─────────────────────
    print(f"\n{'=' * 120}")
    print("  HEATMAP: Annual Return by z_threshold (rows) x top_n (columns)")
    print(f"{'=' * 120}")

    # Build lookup
    heatmap = {}
    for r in results:
        cfg = r['config']
        heatmap[(cfg['z_threshold'], cfg['top_n'])] = r['ann']

    header = f"  {'Z_thr':>6} |"
    for tn in top_ns:
        header += f" {'TN='+str(tn):>10} |"
    print(header)
    print("-" * (8 + 13 * len(top_ns)))
    for zt in z_thresholds:
        row = f"  {zt:>6.1f} |"
        for tn in top_ns:
            ann = heatmap.get((zt, tn), -999)
            row += f" {ann:>+9.1f}% |"
        print(row)

    # ── Best z_threshold for each top_n ──────────────────────────────
    print(f"\n{'=' * 100}")
    print("  BEST z_threshold FOR EACH top_n")
    print(f"{'=' * 100}")
    for tn in top_ns:
        tn_results = [r for r in results if r['config']['top_n'] == tn]
        if tn_results:
            best = tn_results[0]
            print(f"  top_n={tn}: Best z_threshold={best['config']['z_threshold']:.1f}  "
                  f"Ann={best['ann']:>+9.1f}%  WR={best['wr']:>5.1f}%  N={best['n']:>5}  MDD={best['mdd']:>6.1f}%")

    # ── Best top_n for each z_threshold ──────────────────────────────
    print(f"\n{'=' * 100}")
    print("  BEST top_n FOR EACH z_threshold")
    print(f"{'=' * 100}")
    for zt in z_thresholds:
        zt_results = [r for r in results if r['config']['z_threshold'] == zt]
        if zt_results:
            best = zt_results[0]
            print(f"  z_threshold={zt:<4.1f}: Best top_n={best['config']['top_n']}  "
                  f"Ann={best['ann']:>+9.1f}%  WR={best['wr']:>5.1f}%  N={best['n']:>5}  MDD={best['mdd']:>6.1f}%")

    # ══════════════════════════════════════════════════════════════════
    # WALK-FORWARD (Top 15 configs)
    # ══════════════════════════════════════════════════════════════════
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]
    wf_top = 15

    wf_configs = results[:wf_top]

    print(f"\n{'=' * 160}")
    print(f"  WALK-FORWARD (Top {wf_top} configs, {len(wf_years)} years)")
    print(f"{'=' * 160}")

    header = f"  {'#':>3} | {'Config':<30} | {'Avg WF':>9} |"
    for yr in wf_years:
        header += f" {yr:>8} |"
    header += f" {'Pos':>4} | {'WF_MDD':>8}"
    print(header)
    print("-" * 160)

    wf_rows = []
    for i, r in enumerate(wf_configs):
        cfg = r['config']
        label = r['label']
        wf_row = {'label': label, 'z_threshold': cfg['z_threshold'],
                  'top_n': cfg['top_n'], 'windows': {}, 'mdd': {}}
        for yr in wf_years:
            wr = run_backtest(cfg['z_threshold'], cfg['top_n'], wf_test_year=yr)
            if wr:
                wf_row['windows'][yr] = wr['ann']
                wf_row['mdd'][yr] = wr['mdd']
        wf_rows.append(wf_row)

        vals = [wf_row['windows'].get(yr, 0) for yr in wf_years]
        avg = np.mean(vals) if vals else 0
        pos = sum(1 for v in vals if v > 0)
        avg_mdd = np.mean(list(wf_row['mdd'].values())) if wf_row['mdd'] else 0

        row_str = f"  {i+1:>3} | {label:<30} | {avg:>+8.1f}% |"
        for v in vals:
            row_str += f" {v:>+7.1f}% |"
        row_str += f" {pos}/6 | {avg_mdd:>7.1f}%"
        print(row_str)

    # ── Walk-forward summary ─────────────────────────────────────────
    print(f"\n{'=' * 160}")
    print("  WALK-FORWARD SUMMARY")
    print(f"{'=' * 160}")

    # Rank by average WF return
    wf_ranked = []
    for wf in wf_rows:
        vals = [wf['windows'].get(yr, 0) for yr in wf_years]
        avg = np.mean(vals) if vals else 0
        pos = sum(1 for v in vals if v > 0)
        avg_mdd = np.mean(list(wf['mdd'].values())) if wf['mdd'] else 0
        wf_ranked.append({
            'label': wf['label'],
            'z_threshold': wf['z_threshold'],
            'top_n': wf['top_n'],
            'avg_wf': avg,
            'pos': pos,
            'avg_mdd': avg_mdd,
            'vals': vals,
        })

    wf_ranked.sort(key=lambda x: -x['avg_wf'])

    print(f"  {'Rank':>4} | {'Config':<30} | {'Avg WF':>9} | {'Pos':>4} | {'WF_MDD':>8} | Z_thr | TopN")
    print("-" * 120)
    for i, wf in enumerate(wf_ranked):
        tag = " <<< CHAMPION" if i == 0 else ""
        print(f"  {i+1:>4} | {wf['label']:<30} | {wf['avg_wf']:>+8.1f}% | {wf['pos']:>4}/6 | {wf['avg_mdd']:>7.1f}% | "
              f"{wf['z_threshold']:<5.1f} | {wf['top_n']}{tag}")

    # ── Heatmap of WF average ────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print("  WF HEATMAP: Average Walk-Forward Return by z_threshold x top_n")
    print(f"{'=' * 120}")

    # Build WF lookup
    wf_lookup = {}
    for wf in wf_ranked:
        wf_lookup[(wf['z_threshold'], wf['top_n'])] = wf['avg_wf']

    # Also compute WF for all configs not in top 15
    print("  (Computing WF for all configs...)", flush=True)
    wf_all = {}
    for cfg in configs:
        key = (cfg['z_threshold'], cfg['top_n'])
        if key in wf_lookup:
            wf_all[key] = wf_lookup[key]
        else:
            # Quick compute
            vals = []
            for yr in wf_years:
                wr = run_backtest(cfg['z_threshold'], cfg['top_n'], wf_test_year=yr)
                if wr:
                    vals.append(wr['ann'])
            wf_all[key] = np.mean(vals) if vals else -999

    header = f"  {'Z_thr':>6} |"
    for tn in top_ns:
        header += f" {'TN='+str(tn):>10} |"
    print(header)
    print("-" * (8 + 13 * len(top_ns)))
    best_wf_overall = (-999, None, None)
    for zt in z_thresholds:
        row = f"  {zt:>6.1f} |"
        for tn in top_ns:
            avg = wf_all.get((zt, tn), -999)
            row += f" {avg:>+9.1f}% |"
            if avg > best_wf_overall[0]:
                best_wf_overall = (avg, zt, tn)
        print(row)

    print(f"\n  BEST WF COMBINATION: z_threshold={best_wf_overall[1]}, top_n={best_wf_overall[2]}, "
          f"avg WF return={best_wf_overall[0]:>+.1f}%")

    # ══════════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 130}")
    print("  FINAL VERDICT")
    print(f"{'=' * 130}")

    best_full = results[0]
    bf_cfg = best_full['config']
    print(f"  V82 champion reference: D_zscore_Z0.5_TN3 = +3305%")
    print(f"  Best full-period config: {best_full['label']}")
    print(f"    z_threshold={bf_cfg['z_threshold']}, top_n={bf_cfg['top_n']}")
    print(f"    Annual return: {best_full['ann']:>+.1f}%")
    print(f"    Win rate: {best_full['wr']:>5.1f}%")
    print(f"    Trades: {best_full['n']}")
    print(f"    Avg PnL: {best_full['avg_pnl']:>+.3f}%")
    print(f"    Max DD: {best_full['mdd']:>7.1f}%")

    wf_champ = wf_ranked[0]
    print(f"\n  Best walk-forward config: {wf_champ['label']}")
    print(f"    z_threshold={wf_champ['z_threshold']}, top_n={wf_champ['top_n']}")
    print(f"    Avg WF return: {wf_champ['avg_wf']:>+.1f}%")
    print(f"    Positive years: {wf_champ['pos']}/6")
    print(f"    Avg WF MDD: {wf_champ['avg_mdd']:>7.1f}%")

    # Compare with V82 original
    v82_zt, v82_tn = 0.5, 3
    v82_full = None
    for r in results:
        if r['config']['z_threshold'] == v82_zt and r['config']['top_n'] == v82_tn:
            v82_full = r
            break
    if v82_full:
        print(f"\n  V82 original (Z0.5_TN3) full-period: {v82_full['ann']:>+.1f}%")
        if best_full['label'] != v82_full['label']:
            diff = best_full['ann'] - v82_full['ann']
            print(f"  Improvement over V82: {diff:>+.1f}%")
        else:
            print(f"  V82 original config IS the optimum!")

    print(f"\n  Total time: {time.time()-t_start:.1f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
