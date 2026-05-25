"""
Alpha Futures V90 -- Multi-Day Z-Score vs 1-Day Z-Score (V82 Champion Extension)
=================================================================================
V82 champion: z = (own_1day_return - all_groups_avg_1day) / all_groups_std_1day
              z < -0.5 -> buy. 1-day hold. +3305% annual.

V90 tests whether using MULTI-DAY z-scores beats single-day:
  A) z1_baseline:  V82 exact copy -- z of 1-day return (control)
  B) z2_zscore:    z of 2-day cumulative return
  C) z3_zscore:    z of 3-day cumulative return
  D) z5_zscore:    z of 5-day cumulative return
  E) z_sum:        Sum of z1 over past 3 days (persistent weakness)
  F) z_min:        Min z1 over past 3 days (extreme weakness on any day)
  G) z_avg:        Average z1 over past 3 days (smoother)

All signals: 1-day hold, long-only, close-to-close.
Walk-forward: top 15 configs across 2020-2025.
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

# ── Group map (same as V82 champion) ────────────────────────────────
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
    print("Alpha Futures V90 -- Multi-Day Z-Score vs 1-Day Z-Score (V82 Champion Extension)")
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
    print("\n[Signals] Computing returns...", flush=True)
    t0 = time.time()

    ret1 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            cn = C[si, di]
            cp = C[si, di - 1]
            if not np.isnan(cn) and not np.isnan(cp) and cp > 0:
                ret1[si, di] = (cn - cp) / cp

    # ── Precompute N-day cumulative returns ──────────────────────────
    # ret2: (C[di] - C[di-2]) / C[di-2]
    # ret3: (C[di] - C[di-3]) / C[di-3]
    # ret5: (C[di] - C[di-5]) / C[di-5]
    ret2 = np.full((NS, ND), np.nan)
    ret3 = np.full((NS, ND), np.nan)
    ret5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(2, ND):
            cn = C[si, di]
            c2 = C[si, di - 2]
            if not np.isnan(cn) and not np.isnan(c2) and c2 > 0:
                ret2[si, di] = (cn - c2) / c2
        for di in range(3, ND):
            cn = C[si, di]
            c3 = C[si, di - 3]
            if not np.isnan(cn) and not np.isnan(c3) and c3 > 0:
                ret3[si, di] = (cn - c3) / c3
        for di in range(5, ND):
            cn = C[si, di]
            c5 = C[si, di - 5]
            if not np.isnan(cn) and not np.isnan(c5) and c5 > 0:
                ret5[si, di] = (cn - c5) / c5

    # ── Precompute group-level aggregates for each return window ─────
    # For a given return array `reta`, compute:
    #   grp_total_reta[group_name][di] = mean of reta[si,di] for si in group
    #   all_groups_avg_reta[di], all_groups_std_reta[di]
    # Then z_reta[si, di] = (reta[si,di] - all_groups_avg_reta[di]) / all_groups_std_reta[di]

    def compute_group_zscores(reta, label):
        """Compute z-score of each commodity's return vs all-group distribution."""
        print(f"  Computing group z-scores for {label}...", flush=True)

        # group_total[grp][di] = average reta of that group on day di
        grp_total = {}
        for grp in group_names:
            arr = np.full(ND, np.nan)
            members = gm_map[grp]
            for di in range(ND):
                vals = [reta[sk, di] for sk in members if not np.isnan(reta[sk, di])]
                if vals:
                    arr[di] = np.mean(vals)
            grp_total[grp] = arr

        # all_groups_avg[di], all_groups_std[di]
        aga = np.full(ND, np.nan)
        ags = np.full(ND, np.nan)
        for di in range(ND):
            vals = [grp_total[g][di] for g in group_names if not np.isnan(grp_total[g][di])]
            if len(vals) >= 2:
                aga[di] = np.mean(vals)
                ags[di] = np.std(vals)

        # z[si, di]
        z = np.full((NS, ND), np.nan)
        for si in trade_sis:
            for di in range(ND):
                rv = reta[si, di]
                if np.isnan(rv) or np.isnan(aga[di]) or np.isnan(ags[di]) or ags[di] < 1e-8:
                    continue
                z[si, di] = (rv - aga[di]) / ags[di]

        return z, aga, ags

    z1, aga1, ags1 = compute_group_zscores(ret1, "ret1 (1-day)")
    z2, aga2, ags2 = compute_group_zscores(ret2, "ret2 (2-day)")
    z3, aga3, ags3 = compute_group_zscores(ret3, "ret3 (3-day)")
    z5, aga5, ags5 = compute_group_zscores(ret5, "ret5 (5-day)")

    # ── Precompute composite z-scores (E, F, G) ─────────────────────
    # These are all based on z1 (1-day z-score) rolled over past 3 days

    # E) z_sum: z1[di] + z1[di-1] + z1[di-2]
    z_sum = np.full((NS, ND), np.nan)
    for si in trade_sis:
        for di in range(2, ND):
            v0 = z1[si, di]
            v1 = z1[si, di - 1]
            v2 = z1[si, di - 2]
            if not np.isnan(v0) and not np.isnan(v1) and not np.isnan(v2):
                z_sum[si, di] = v0 + v1 + v2

    # F) z_min: min(z1[di], z1[di-1], z1[di-2])
    z_min = np.full((NS, ND), np.nan)
    for si in trade_sis:
        for di in range(2, ND):
            v0 = z1[si, di]
            v1 = z1[si, di - 1]
            v2 = z1[si, di - 2]
            if not np.isnan(v0) and not np.isnan(v1) and not np.isnan(v2):
                z_min[si, di] = min(v0, v1, v2)

    # G) z_avg: mean(z1[di], z1[di-1], z1[di-2])
    z_avg = np.full((NS, ND), np.nan)
    for si in trade_sis:
        for di in range(2, ND):
            v0 = z1[si, di]
            v1 = z1[si, di - 1]
            v2 = z1[si, di - 2]
            if not np.isnan(v0) and not np.isnan(v1) and not np.isnan(v2):
                z_avg[si, di] = (v0 + v1 + v2) / 3.0

    print(f"  All signals computed ({time.time()-t0:.1f}s)")

    # ══════════════════════════════════════════════════════════════════
    # BACKTEST ENGINE
    # ══════════════════════════════════════════════════════════════════
    # Map signal name -> z-score array
    SIGNAL_Z = {
        'A_z1_baseline': z1,
        'B_z2_zscore':   z2,
        'C_z3_zscore':   z3,
        'D_z5_zscore':   z5,
        'E_z_sum':       z_sum,
        'F_z_min':       z_min,
        'G_z_avg':       z_avg,
    }

    def run_backtest(config, wf_test_year=None):
        """
        Config:
            signal: one of SIGNAL_Z keys
            threshold: z-score threshold (buy when z < -threshold)
            top_n: 1 | 3
            comm: float
        """
        sig_type = config['signal']
        threshold = config['threshold']
        top_n = config['top_n']
        comm = config.get('comm', COMM)

        z_arr = SIGNAL_Z[sig_type]

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

            # ── Generate signals ─────────────────────────────────────
            # Universal: buy when z < -threshold (unusually weak vs all groups)
            candidates = []  # (si, score, direction, sym)

            for si in trade_sis:
                zv = z_arr[si, di]
                if np.isnan(zv):
                    continue
                if zv >= -threshold:
                    continue
                cc = C[si, di]
                if np.isnan(cc) or cc <= 0:
                    continue
                if any(p['si'] == si for p in positions):
                    continue
                score = -zv  # more negative z = higher score
                candidates.append((si, score, 1, syms[si]))

            if not candidates:
                continue

            # Sort by score descending (most extreme weakness first)
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

    # ══════════════════════════════════════════════════════════════════
    # BUILD CONFIGURATIONS
    # ══════════════════════════════════════════════════════════════════
    print("\n[Sweep] Building configurations...", flush=True)
    configs = []
    cid = 0

    signal_keys = ['A_z1_baseline', 'B_z2_zscore', 'C_z3_zscore', 'D_z5_zscore',
                   'E_z_sum', 'F_z_min', 'G_z_avg']
    thresholds = [0.3, 0.5, 0.7, 1.0]
    top_ns = [1, 3]

    for sig in signal_keys:
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

    # ══════════════════════════════════════════════════════════════════
    # RUN FULL-PERIOD BACKTEST
    # ══════════════════════════════════════════════════════════════════
    print("\n[Backtest] Running full-period sweep...", flush=True)
    results = []
    for i, cfg in enumerate(configs):
        r = run_backtest(cfg)
        if r:
            r['config'] = cfg
            r['label'] = cfg['label']
            results.append(r)
        if (i + 1) % 10 == 0 or i == len(configs) - 1:
            print(f"  ... {i+1}/{len(configs)} done", flush=True)

    results.sort(key=lambda x: -x['ann'])

    # Print top 25
    print(f"\n{'=' * 130}")
    print("  FULL-PERIOD RESULTS (Top 25)")
    print(f"{'=' * 130}")
    print(f"  {'#':>3} | {'Label':<40} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7}")
    print("-" * 110)
    for i, r in enumerate(results[:25]):
        print(f"  {i+1:>3} | {r['label']:<40} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}%")

    # ══════════════════════════════════════════════════════════════════
    # SIGNAL COMPARISON (full period)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 140}")
    print("  SIGNAL COMPARISON (Best per signal type, full period)")
    print(f"{'=' * 140}")
    print(f"  {'Signal':<25} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | Best Config")
    print("-" * 140)

    best_per_sig = {}
    for r in results:
        s = r['config']['signal']
        if s not in best_per_sig:
            best_per_sig[s] = r

    for sig in signal_keys:
        if sig in best_per_sig:
            b = best_per_sig[sig]
            print(f"  {sig:<25} | {b['ann']:>+8.1f}% | {b['wr']:>5.1f}% | {b['n']:>5} | {b['avg_pnl']:>+6.3f}% | {b['mdd']:>6.1f}% | {b['label']}")

    # Delta vs baseline (A_z1)
    baseline = best_per_sig.get('A_z1_baseline')
    if baseline:
        print(f"\n  {'='*100}")
        print(f"  ALPHA vs BASELINE (A_z1_baseline = {baseline['ann']:>+8.1f}%)")
        print(f"  {'='*100}")
        for sig in signal_keys[1:]:
            if sig in best_per_sig:
                diff = best_per_sig[sig]['ann'] - baseline['ann']
                tag = "BEATS" if diff > 0 else "LOSES"
                b = best_per_sig[sig]
                print(f"  {sig:<25} {b['ann']:>+8.1f}%  ({tag} by {diff:>+.1f}%)  WR {b['wr']:>5.1f}%  MDD {b['mdd']:>6.1f}%")

    # ── Per-threshold breakdown ──────────────────────────────────────
    print(f"\n{'=' * 140}")
    print("  THRESHOLD BREAKDOWN (by signal type, all thresholds)")
    print(f"{'=' * 140}")
    for sig in signal_keys:
        sig_label = sig.replace('_zscore', '').replace('_z1_', ' z1 ').replace('_', ' ')
        print(f"\n  --- {sig} ---")
        print(f"  {'Threshold':>10} | {'TopN':>4} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7}")
        sig_results = [r for r in results if r['config']['signal'] == sig]
        sig_results.sort(key=lambda x: -x['ann'])
        for r in sig_results:
            cfg = r['config']
            print(f"  {cfg['threshold']:>10.1f} | TN{cfg['top_n']:>2} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}%")

    # ══════════════════════════════════════════════════════════════════
    # WALK-FORWARD (Top 15)
    # ══════════════════════════════════════════════════════════════════
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Take top 15 configs; also add best per signal type if not already in top 15
    wf_configs = list(results[:15])
    for sig in signal_keys:
        if sig in best_per_sig:
            r = best_per_sig[sig]
            if r['config'] not in [w['config'] for w in wf_configs]:
                wf_configs.append(r)

    print(f"\n{'=' * 160}")
    print(f"  WALK-FORWARD ({len(wf_configs)} configs, 6 windows 2020-2025)")
    print(f"{'=' * 160}")

    header = f"  {'#':>3} | {'Config':<40} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>7} |"
    header += f" {'Pos':>4} | {'MDD':>7}"
    print(header)
    print("-" * 160)

    wf_rows = []
    for i, r in enumerate(wf_configs):
        cfg = r['config']
        wf_row = {'label': cfg['label'], 'signal': cfg['signal'],
                  'threshold': cfg['threshold'], 'top_n': cfg['top_n'],
                  'windows': {}, 'mdd': {}}
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

    # ══════════════════════════════════════════════════════════════════
    # WF COMPARISON PER SIGNAL (best per signal)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 140}")
    print("  WALK-FORWARD COMPARISON (Best per signal type)")
    print(f"{'=' * 140}")
    header2 = f"  {'Signal':<25} | {'WF Avg':>8} |"
    for yr in wf_years:
        header2 += f" {yr:>7} |"
    header2 += f" {'Pos':>4} | {'MDD':>7}"
    print(header2)
    print("-" * 140)

    for sig in signal_keys:
        wf_match = [w for w in wf_rows if w['signal'] == sig]
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            avg = np.mean(vals) if vals else 0
            pos = sum(1 for v in vals if v > 0)
            avg_mdd = np.mean([wf['mdd'].get(yr, 0) for yr in wf_years])
            row_str = f"  {sig:<25} | {avg:>+7.1f}% |"
            for v in vals:
                row_str += f" {v:>+7.1f}% |"
            row_str += f" {pos}/6 | {avg_mdd:>6.1f}%"
            print(row_str)

    # ══════════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 140}")
    print("  FINAL VERDICT")
    print(f"{'=' * 140}")

    if baseline:
        print(f"  A) z1_baseline (V82 copy):  {baseline['ann']:>+8.1f}%  WR {baseline['wr']:>5.1f}%  MDD {baseline['mdd']:>6.1f}%")

        # Find best overall
        best_all = results[0]
        print(f"\n  BEST OVERALL: {best_all['label']}")
        print(f"    Ann: {best_all['ann']:>+8.1f}%  WR: {best_all['wr']:>5.1f}%  MDD: {best_all['mdd']:>6.1f}%")

        if best_all['config']['signal'] == 'A_z1_baseline':
            print(f"\n  >>> BASELINE 1-DAY Z-SCORE REMAINS CHAMPION <<<")
            print(f"  >>> Multi-day windows do NOT add alpha <<<")
        else:
            diff = best_all['ann'] - baseline['ann']
            print(f"\n  >>> {best_all['config']['signal']} BEATS BASELINE by {diff:+.1f}% <<<")

        # Category verdict
        print(f"\n  Category analysis:")
        cat_results = {}
        for sig in signal_keys:
            if sig in best_per_sig:
                cat_results[sig] = best_per_sig[sig]

        # Multi-day return z-scores (B, C, D) vs baseline
        print(f"\n  Multi-day return z-scores:")
        for sig, label in [('B_z2_zscore', '2-day'), ('C_z3_zscore', '3-day'), ('D_z5_zscore', '5-day')]:
            if sig in cat_results:
                diff = cat_results[sig]['ann'] - baseline['ann']
                tag = "BETTER" if diff > 0 else "WORSE"
                print(f"    {label:>6}: {cat_results[sig]['ann']:>+8.1f}% ({tag} {diff:>+.1f}%)")

        # Composite z-scores (E, F, G) vs baseline
        print(f"\n  Composite z-scores (3-day window of z1):")
        for sig, label in [('E_z_sum', 'z_sum'), ('F_z_min', 'z_min'), ('G_z_avg', 'z_avg')]:
            if sig in cat_results:
                diff = cat_results[sig]['ann'] - baseline['ann']
                tag = "BETTER" if diff > 0 else "WORSE"
                print(f"    {label:>6}: {cat_results[sig]['ann']:>+8.1f}% ({tag} {diff:>+.1f}%)")

        # Walk-forward verdict
        print(f"\n  Walk-forward robustness:")
        for sig in signal_keys:
            wf_match = [w for w in wf_rows if w['signal'] == sig]
            if wf_match:
                wf = wf_match[0]
                vals = [wf['windows'].get(yr, 0) for yr in wf_years]
                pos = sum(1 for v in vals if v > 0)
                avg = np.mean(vals)
                print(f"    {sig:<25} WF Avg {avg:>+8.1f}%  {pos}/6 positive")

    print(f"\n  Total time: {time.time()-t_start:.1f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
