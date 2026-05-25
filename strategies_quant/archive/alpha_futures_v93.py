"""
Alpha Futures V93 -- Overnight Z-Score Parameter Optimization & Signal Enhancements
====================================================================================
V92 champion: overnight cross-group z-score gives +4282% annual.
Signal: z = (own_overnight - all_groups_avg_overnight) / all_groups_std_overnight
        when z < -threshold -> buy at C[si,di], sell at C[si,di+1]. Long-only, 1-day hold.

V93 experiments:
  A) THRESHOLD SWEEP:        z_threshold 0.05..1.50 (step 0.05) x top_n 1..7 = 210 configs
  B) Z-MAGNITUDE WEIGHTING:  allocate lots proportional to |z| instead of equal weight
  C) INTRADAY CONFIRMATION:  only buy when C[si,di] < O[si,di] (bearish candle confirms)
  D) DAY-OF-WEEK FILTER:     test Mon-Tue, Wed-Fri, exclude Mon, exclude Fri
  E) GROUP-STRENGTH FILTER:  only buy when own group avg overnight return > 0

Walk-forward: top 15 configs across 2020-2025.
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

# ── Group map (same as V82) ──────────────────────────────────────────
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
    print("Alpha Futures V93 -- Overnight Z-Score Optimization & Enhancements")
    print("=" * 120)

    # ── Load data ────────────────────────────────────────────────────
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ── Build group membership ───────────────────────────────────────
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

    # ================================================================
    # PRECOMPUTE OVERNIGHT RETURNS AND CROSS-GROUP Z-SCORES
    # ================================================================
    print("\n[Signals] Computing overnight returns and cross-group z-scores...", flush=True)
    t0 = time.time()

    # overnight_ret[si, di] = (O[si,di] - C[si,di-1]) / C[si,di-1]
    overnight_ret = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            c_prev = C[si, di - 1]
            o_now = O[si, di]
            if not np.isnan(c_prev) and c_prev > 0 and not np.isnan(o_now) and o_now > 0:
                overnight_ret[si, di] = (o_now - c_prev) / c_prev

    # Group-level overnight averages
    # grp_on_avg[group_name][di] = average overnight return of that group's members
    grp_on_avg = {}
    for grp in group_names:
        arr = np.full(ND, np.nan)
        members = gm_map[grp]
        for di in range(1, ND):
            vals = [overnight_ret[sk, di] for sk in members if not np.isnan(overnight_ret[sk, di])]
            if vals:
                arr[di] = np.mean(vals)
        grp_on_avg[grp] = arr

    # Cross-group aggregates
    all_groups_on_avg = np.full(ND, np.nan)
    all_groups_on_std = np.full(ND, np.nan)
    for di in range(1, ND):
        vals = [grp_on_avg[g][di] for g in group_names if not np.isnan(grp_on_avg[g][di])]
        if len(vals) >= 2:
            all_groups_on_avg[di] = np.mean(vals)
            all_groups_on_std[di] = np.std(vals)

    # z-score per commodity: z[si,di] = (own_overnight - all_groups_avg) / all_groups_std
    z_overnight = np.full((NS, ND), np.nan)
    for si in trade_sis:
        for di in range(1, ND):
            own = overnight_ret[si, di]
            aga = all_groups_on_avg[di]
            ags = all_groups_on_std[di]
            if np.isnan(own) or np.isnan(aga) or np.isnan(ags) or ags < 1e-8:
                continue
            z_overnight[si, di] = (own - aga) / ags

    # Precompute intraday candle direction for Experiment C
    # bearish[si, di] = True if C[si,di] < O[si,di]
    bearish = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(1, ND):
            c_now = C[si, di]
            o_now = O[si, di]
            if not np.isnan(c_now) and not np.isnan(o_now) and c_now < o_now:
                bearish[si, di] = True

    # Precompute day of week for Experiment D
    # dates is a list of date objects
    dow = np.array([d.weekday() for d in dates])  # 0=Mon, 1=Tue, ..., 4=Fri

    print(f"  Signals computed ({time.time()-t0:.1f}s)")
    print(f"  z_overnight: mean={np.nanmean(z_overnight):.4f}, std={np.nanstd(z_overnight):.4f}")

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(config, wf_test_year=None):
        """
        Config keys:
            signal: 'A_sweep' | 'B_zweight' | 'C_intraday' | 'D_dow' | 'E_grpstr'
            threshold: float (z < -threshold -> buy)
            top_n: int
            comm: float
            dow_filter: str ('all'|'mon_tue'|'wed_fri'|'no_mon'|'no_fri')  [for D]
            weighted: bool  [for B]
            intraday_confirm: bool  [for C]
            group_filter: bool  [for E]
        """
        sig_type = config['signal']
        threshold = config['threshold']
        top_n = config['top_n']
        comm = config.get('comm', COMM)
        weighted = config.get('weighted', False)
        intraday_confirm = config.get('intraday_confirm', False)
        group_filter = config.get('group_filter', False)
        dow_filter = config.get('dow_filter', 'all')

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

            # ── Day-of-week filter (Experiment D) ────────────────────
            if dow_filter != 'all':
                d = dow[di]
                if dow_filter == 'mon_tue' and d not in (0, 1):
                    continue
                elif dow_filter == 'wed_fri' and d not in (2, 3, 4):
                    continue
                elif dow_filter == 'no_mon' and d == 0:
                    continue
                elif dow_filter == 'no_fri' and d == 4:
                    continue

            # ── Generate signals ─────────────────────────────────────
            candidates = []  # (si, score, direction, sym)

            aga = all_groups_on_avg[di]
            ags = all_groups_on_std[di]
            if np.isnan(aga) or np.isnan(ags) or ags < 1e-8:
                continue

            for si in trade_sis:
                if any(p['si'] == si for p in positions):
                    continue
                c_now = C[si, di]
                if np.isnan(c_now) or c_now <= 0:
                    continue

                z = z_overnight[si, di]
                if np.isnan(z):
                    continue
                if z >= -threshold:
                    continue

                # Experiment C: intraday confirmation filter
                if intraday_confirm and not bearish[si, di]:
                    continue

                # Experiment E: group-strength filter
                # Only buy when own group average overnight return > 0
                if group_filter:
                    own_grp = si_group[si]
                    grp_avg_on = grp_on_avg[own_grp][di]
                    if np.isnan(grp_avg_on) or grp_avg_on <= 0:
                        continue

                score = -z  # higher score = more negative z = stronger signal
                candidates.append((si, score, 1, syms[si]))

            if not candidates:
                continue

            # Sort by score descending
            candidates.sort(key=lambda x: -x[1])

            # Select top_n
            selected = candidates[:top_n]

            if not selected:
                continue

            # Experiment B: Z-magnitude weighting
            if weighted and len(selected) > 1:
                total_score = sum(abs(s[1]) for s in selected)
                if total_score < 1e-10:
                    total_score = 1.0
                # Allocate cash proportionally to |z|
                for si, score, direction, sym in selected:
                    weight = abs(score) / total_score
                    alloc_cash = cash * weight
                    mult = MULT.get(sym, DEF_MULT)
                    notional = c_now * mult if not np.isnan(c_now := C[si, di]) and c_now > 0 else 0
                    if notional <= 0:
                        continue
                    lots = int(alloc_cash / (notional * (1 + comm)))
                    if lots <= 0:
                        continue
                    cost_in = notional * lots * (1 + comm)
                    if cost_in > alloc_cash:
                        lots = int(alloc_cash * 0.95 / (notional * (1 + comm)))
                        cost_in = notional * lots * (1 + comm) if lots > 0 else 0
                    if lots <= 0 or cost_in <= 0 or cost_in > cash:
                        continue

                    cash -= cost_in
                    positions.append({
                        'si': si, 'entry': C[si, di], 'entry_di': di,
                        'lots': lots, 'dir': direction, 'sym': sym,
                    })
            else:
                # Equal-weight allocation (standard)
                n_slots = top_n - len(positions)
                for si, score, direction, sym in selected[:max(0, n_slots)]:
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

        # Close remaining positions
        for pos in positions:
            ae = end_di - 1 if end_di < ND else ND - 1
            cn = C[pos['si'], ae]
            if np.isnan(cn) or cn <= 0:
                cn = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = cn * mult * abs(pos['lots'])
            cash += mkt_val - mkt_val * comm

        # Calculate results
        wf_mode = wf_test_year is not None
        n_days_test = (test_end_di - test_start_di) if wf_mode else (end_di - start_di)
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0

        # Max drawdown from equity curve
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
    # BUILD CONFIGURATIONS
    # ================================================================
    print("\n[Sweep] Building configurations...", flush=True)
    configs = []
    cid = 0

    # ── Experiment A: Threshold Sweep ────────────────────────────────
    # z_threshold from 0.05 to 1.50 in 0.05 steps, top_n from 1 to 7
    # = 30 thresholds x 7 top_n = 210 configs
    for thresh_i in range(1, 31):
        thresh = round(thresh_i * 0.05, 2)
        for tn in range(1, 8):
            cid += 1
            configs.append({
                'id': cid, 'signal': 'A_sweep',
                'threshold': thresh, 'top_n': tn, 'comm': COMM,
                'label': f"A_sweep_Z{thresh:.2f}_TN{tn}",
            })

    # ── Experiment B: Z-Magnitude Weighting ──────────────────────────
    # Use promising thresholds from A plus a range; test with top_n 3,5,7
    for thresh in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80]:
        for tn in [3, 5, 7]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'B_zweight',
                'threshold': thresh, 'top_n': tn, 'comm': COMM,
                'weighted': True,
                'label': f"B_zwt_Z{thresh:.2f}_TN{tn}",
            })

    # ── Experiment C: Intraday Confirmation ──────────────────────────
    # Same threshold sweep but require bearish candle (C < O)
    for thresh in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80]:
        for tn in [3, 5, 7]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'C_intraday',
                'threshold': thresh, 'top_n': tn, 'comm': COMM,
                'intraday_confirm': True,
                'label': f"C_intra_Z{thresh:.2f}_TN{tn}",
            })

    # ── Experiment D: Day-of-Week Filter ─────────────────────────────
    for dow_name in ['mon_tue', 'wed_fri', 'no_mon', 'no_fri']:
        for thresh in [0.15, 0.25, 0.30, 0.40, 0.50]:
            for tn in [3, 5]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': 'D_dow',
                    'threshold': thresh, 'top_n': tn, 'comm': COMM,
                    'dow_filter': dow_name,
                    'label': f"D_{dow_name}_Z{thresh:.2f}_TN{tn}",
                })

    # ── Experiment E: Group-Strength Filter ──────────────────────────
    for thresh in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80]:
        for tn in [3, 5, 7]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'E_grpstr',
                'threshold': thresh, 'top_n': tn, 'comm': COMM,
                'group_filter': True,
                'label': f"E_grpstr_Z{thresh:.2f}_TN{tn}",
            })

    print(f"  Total configs: {len(configs)}")

    # ================================================================
    # RUN FULL-PERIOD BACKTEST
    # ================================================================
    print("\n[Backtest] Running full-period sweep...", flush=True)
    results = []
    for i, cfg in enumerate(configs):
        r = run_backtest(cfg)
        if r:
            r['config'] = cfg
            r['label'] = cfg['label']
            results.append(r)
        if (i + 1) % 50 == 0 or i == len(configs) - 1:
            print(f"  ... {i+1}/{len(configs)} done ({time.time()-t_start:.1f}s)", flush=True)

    results.sort(key=lambda x: -x['ann'])

    # ================================================================
    # FULL-PERIOD RESULTS
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  FULL-PERIOD RESULTS (Top 30)")
    print(f"{'=' * 130}")
    print(f"  {'#':>3} | {'Label':<40} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7}")
    print("-" * 110)
    for i, r in enumerate(results[:30]):
        print(f"  {i+1:>3} | {r['label']:<40} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}%")

    # ================================================================
    # EXPERIMENT A: THRESHOLD SWEEP HEATMAP
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  EXPERIMENT A: THRESHOLD x TOP_N HEATMAP (Annual Return %)")
    print(f"{'=' * 130}")

    sweep_results = [r for r in results if r['config']['signal'] == 'A_sweep']

    # Build lookup: (thresh, top_n) -> ann
    sweep_map = {}
    for r in sweep_results:
        key = (r['config']['threshold'], r['config']['top_n'])
        sweep_map[key] = r

    # Print header
    header = f"  {'Z-thresh':>8} |"
    for tn in range(1, 8):
        header += f" {'TN=' + str(tn):>10} |"
    print(header)
    print("-" * 95)

    best_sweep_ann = -1e9
    best_sweep_label = ""
    for thresh_i in range(1, 31):
        thresh = round(thresh_i * 0.05, 2)
        row = f"  {thresh:>8.2f} |"
        for tn in range(1, 8):
            key = (thresh, tn)
            r = sweep_map.get(key)
            if r:
                ann_str = f"{r['ann']:>+9.1f}%"
                row += f" {ann_str:>10} |"
                if r['ann'] > best_sweep_ann:
                    best_sweep_ann = r['ann']
                    best_sweep_label = r['label']
            else:
                row += f" {'---':>10} |"
        print(row)

    print(f"\n  Best sweep config: {best_sweep_label} -> {best_sweep_ann:>+8.1f}%")

    # ── Best per top_n ───────────────────────────────────────────────
    print(f"\n  Best threshold per top_n (Experiment A):")
    for tn in range(1, 8):
        sub = [r for r in sweep_results if r['config']['top_n'] == tn]
        if sub:
            best = max(sub, key=lambda x: x['ann'])
            print(f"    top_n={tn}: Z={best['config']['threshold']:.2f} -> {best['ann']:>+8.1f}%  WR={best['wr']:.1f}%  N={best['n']}  MDD={best['mdd']:.1f}%")

    # ── Best per threshold (aggregated) ──────────────────────────────
    print(f"\n  Best top_n per threshold (Experiment A, thresholds with best results):")
    for thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80, 1.00, 1.50]:
        sub = [r for r in sweep_results if abs(r['config']['threshold'] - thresh) < 0.001]
        if sub:
            best = max(sub, key=lambda x: x['ann'])
            print(f"    Z={thresh:.2f}: TN={best['config']['top_n']} -> {best['ann']:>+8.1f}%  WR={best['wr']:.1f}%  N={best['n']}  MDD={best['mdd']:.1f}%")

    # ================================================================
    # EXPERIMENT B: Z-MAGNITUDE WEIGHTING RESULTS
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  EXPERIMENT B: Z-MAGNITUDE WEIGHTING vs EQUAL WEIGHT")
    print(f"{'=' * 130}")

    b_results = [r for r in results if r['config']['signal'] == 'B_zweight']
    a_baseline_for_b = {}
    for r in sweep_results:
        key = (round(r['config']['threshold'], 2), r['config']['top_n'])
        a_baseline_for_b[key] = r

    print(f"  {'Label':<40} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'MDD':>7} | {'Baseline':>9} | {'Delta':>8}")
    print("-" * 130)
    for r in sorted(b_results, key=lambda x: -x['ann'])[:20]:
        key = (round(r['config']['threshold'], 2), r['config']['top_n'])
        baseline = a_baseline_for_b.get(key)
        base_ann = baseline['ann'] if baseline else 0
        delta = r['ann'] - base_ann
        base_str = f"{base_ann:>+8.1f}%" if baseline else "N/A"
        print(f"  {r['label']:<40} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['mdd']:>6.1f}% | {base_str:>9} | {delta:>+7.1f}%")

    # ================================================================
    # EXPERIMENT C: INTRADAY CONFIRMATION RESULTS
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  EXPERIMENT C: INTRADAY CONFIRMATION (C < O required)")
    print(f"{'=' * 130}")

    c_results = [r for r in results if r['config']['signal'] == 'C_intraday']

    print(f"  {'Label':<40} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'MDD':>7} | {'Baseline':>9} | {'Delta':>8}")
    print("-" * 130)
    for r in sorted(c_results, key=lambda x: -x['ann'])[:15]:
        key = (round(r['config']['threshold'], 2), r['config']['top_n'])
        baseline = a_baseline_for_b.get(key)
        base_ann = baseline['ann'] if baseline else 0
        delta = r['ann'] - base_ann
        base_str = f"{base_ann:>+8.1f}%" if baseline else "N/A"
        print(f"  {r['label']:<40} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['mdd']:>6.1f}% | {base_str:>9} | {delta:>+7.1f}%")

    # Count how many C configs beat baseline
    c_beats = 0
    c_total = 0
    for r in c_results:
        key = (round(r['config']['threshold'], 2), r['config']['top_n'])
        baseline = a_baseline_for_b.get(key)
        if baseline:
            c_total += 1
            if r['ann'] > baseline['ann']:
                c_beats += 1
    print(f"\n  Intraday confirmation beats baseline: {c_beats}/{c_total} configs")

    # ================================================================
    # EXPERIMENT D: DAY-OF-WEEK FILTER RESULTS
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  EXPERIMENT D: DAY-OF-WEEK FILTER")
    print(f"{'=' * 130}")

    d_results = [r for r in results if r['config']['signal'] == 'D_dow']

    for dow_name in ['mon_tue', 'wed_fri', 'no_mon', 'no_fri']:
        sub = [r for r in d_results if r['config']['dow_filter'] == dow_name]
        sub.sort(key=lambda x: -x['ann'])
        print(f"\n  --- {dow_name.upper()} (best 5) ---")
        print(f"  {'Label':<40} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'MDD':>7}")
        print(f"  {'-'*80}")
        for r in sub[:5]:
            print(f"  {r['label']:<40} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['mdd']:>6.1f}%")
        if sub:
            best = sub[0]
            # Compare to all-days baseline at same thresh/tn
            key = (round(best['config']['threshold'], 2), best['config']['top_n'])
            baseline = a_baseline_for_b.get(key)
            base_ann = baseline['ann'] if baseline else 0
            delta = best['ann'] - base_ann
            print(f"  Best {dow_name}: {best['ann']:>+8.1f}% vs all-days baseline {base_ann:>+8.1f}% (delta {delta:>+.1f}%)")

    # ================================================================
    # EXPERIMENT E: GROUP-STRENGTH FILTER RESULTS
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  EXPERIMENT E: GROUP-STRENGTH FILTER (own group avg overnight > 0)")
    print(f"{'=' * 130}")

    e_results = [r for r in results if r['config']['signal'] == 'E_grpstr']

    print(f"  {'Label':<40} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'MDD':>7} | {'Baseline':>9} | {'Delta':>8}")
    print("-" * 130)
    for r in sorted(e_results, key=lambda x: -x['ann'])[:15]:
        key = (round(r['config']['threshold'], 2), r['config']['top_n'])
        baseline = a_baseline_for_b.get(key)
        base_ann = baseline['ann'] if baseline else 0
        delta = r['ann'] - base_ann
        base_str = f"{base_ann:>+8.1f}%" if baseline else "N/A"
        print(f"  {r['label']:<40} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['mdd']:>6.1f}% | {base_str:>9} | {delta:>+7.1f}%")

    # Count how many E configs beat baseline
    e_beats = 0
    e_total = 0
    for r in e_results:
        key = (round(r['config']['threshold'], 2), r['config']['top_n'])
        baseline = a_baseline_for_b.get(key)
        if baseline:
            e_total += 1
            if r['ann'] > baseline['ann']:
                e_beats += 1
    print(f"\n  Group-strength filter beats baseline: {e_beats}/{e_total} configs")

    # ================================================================
    # SIGNAL COMPARISON (best per experiment, full period)
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  SIGNAL COMPARISON (Best per experiment, full period)")
    print(f"{'=' * 130}")
    print(f"  {'Experiment':<25} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | Best Config")
    print("-" * 130)

    best_per_exp = {}
    exp_order = ['A_sweep', 'B_zweight', 'C_intraday', 'D_dow', 'E_grpstr']
    for r in results:
        s = r['config']['signal']
        if s not in best_per_exp:
            best_per_exp[s] = r

    for sig in exp_order:
        if sig in best_per_exp:
            b = best_per_exp[sig]
            print(f"  {sig:<25} | {b['ann']:>+8.1f}% | {b['wr']:>5.1f}% | {b['n']:>5} | {b['avg_pnl']:>+6.3f}% | {b['mdd']:>6.1f}% | {b['label']}")

    # ================================================================
    # WALK-FORWARD (Top 15 configs)
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Collect WF configs: top 15 overall + best per experiment
    wf_configs = list(results[:15])
    for sig in exp_order:
        if sig in best_per_exp:
            r = best_per_exp[sig]
            if r['config'] not in [w['config'] for w in wf_configs]:
                wf_configs.append(r)

    print(f"\n{'=' * 160}")
    print(f"  WALK-FORWARD ({len(wf_configs)} configs)")
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

    # ================================================================
    # WF COMPARISON PER EXPERIMENT
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  WALK-FORWARD COMPARISON (Best per experiment)")
    print(f"{'=' * 130}")
    header2 = f"  {'Experiment':<25} | {'WF Avg':>8} |"
    for yr in wf_years:
        header2 += f" {yr:>7} |"
    header2 += f" {'Pos':>4} | Avg MDD"
    print(header2)
    print("-" * 130)

    for sig in exp_order:
        wf_match = [w for w in wf_rows if w['signal'] == sig]
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            avg = np.mean(vals) if vals else 0
            pos = sum(1 for v in vals if v > 0)
            avg_mdd = np.mean(list(wf['mdd'].values())) if wf['mdd'] else 0
            row_str = f"  {sig:<25} | {avg:>+7.1f}% |"
            for v in vals:
                row_str += f" {v:>+7.1f}% |"
            row_str += f" {pos}/6 | {avg_mdd:>6.1f}%"
            print(row_str)

    # ================================================================
    # V92 BASELINE COMPARISON
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  V92 BASELINE COMPARISON")
    print(f"{'=' * 130}")

    # V92 baseline: Z=-0.3, TN=3, close-to-close
    v92_cfg = {
        'id': 0, 'signal': 'A_sweep',
        'threshold': 0.3, 'top_n': 3, 'comm': COMM,
        'label': "V92_baseline_Z0.30_TN3",
    }
    v92_result = run_backtest(v92_cfg)
    if v92_result:
        print(f"  V92 Baseline (Z=0.30, TN=3):  {v92_result['ann']:>+8.1f}%  WR={v92_result['wr']:.1f}%  N={v92_result['n']}  MDD={v92_result['mdd']:.1f}%")

        # V92 WF
        v92_wf = {}
        for yr in wf_years:
            wr = run_backtest(v92_cfg, wf_test_year=yr)
            if wr:
                v92_wf[yr] = wr['ann']
        v92_vals = [v92_wf.get(yr, 0) for yr in wf_years]
        v92_avg = np.mean(v92_vals) if v92_vals else 0
        v92_pos = sum(1 for v in v92_vals if v > 0)
        v92_str = f"  V92 WF: Avg={v92_avg:>+7.1f}% |"
        for yr in wf_years:
            v92_str += f" {v92_wf.get(yr, 0):>+7.1f}% |"
        v92_str += f" {v92_pos}/6"
        print(v92_str)

        # Best V93 vs V92
        if results:
            best = results[0]
            diff = best['ann'] - v92_result['ann']
            print(f"\n  Best V93: {best['ann']:>+8.1f}%  ({best['label']})")
            print(f"  Delta vs V92: {diff:>+.1f}%")
            if diff > 0:
                print(f"  >>> V93 IMPROVES ON V92 BY +{diff:.1f}% <<<")
            else:
                print(f"  >>> V93 DOES NOT IMPROVE ON V92 ({diff:+.1f}%) <<<")

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print(f"\n{'=' * 130}")
    print("  FINAL VERDICT")
    print(f"{'=' * 130}")

    if v92_result:
        print(f"  V92 Baseline:  {v92_result['ann']:>+8.1f}%  (Z=0.30, TN=3)")

    # Best sweep config
    if sweep_results:
        best_sweep = max(sweep_results, key=lambda x: x['ann'])
        print(f"  Best Sweep:    {best_sweep['ann']:>+8.1f}%  (Z={best_sweep['config']['threshold']:.2f}, TN={best_sweep['config']['top_n']})")

    # Best enhanced signal
    enhanced_sigs = [r for r in results if r['config']['signal'] in ('B_zweight', 'C_intraday', 'D_dow', 'E_grpstr')]
    if enhanced_sigs:
        best_enhanced = max(enhanced_sigs, key=lambda x: x['ann'])
        print(f"  Best Enhanced: {best_enhanced['ann']:>+8.1f}%  ({best_enhanced['label']})")

    # Best overall
    if results:
        best = results[0]
        print(f"  Best Overall:  {best['ann']:>+8.1f}%  ({best['label']})")

    # Which filters help?
    print(f"\n  Filter effectiveness summary:")
    for exp_name, exp_key in [('B: Z-weighting', 'B_zweight'),
                               ('C: Intraday confirm', 'C_intraday'),
                               ('D: Day-of-week', 'D_dow'),
                               ('E: Group-strength', 'E_grpstr')]:
        sub = [r for r in results if r['config']['signal'] == exp_key]
        if sub:
            best_sub = max(sub, key=lambda x: x['ann'])
            # Compare to best sweep at same threshold/top_n
            key = (round(best_sub['config']['threshold'], 2), best_sub['config']['top_n'])
            baseline = a_baseline_for_b.get(key)
            base_ann = baseline['ann'] if baseline else 0
            delta = best_sub['ann'] - base_ann
            tag = "HELPS" if delta > 0 else "HURTS"
            print(f"    {exp_name:<22} Best={best_sub['ann']:>+8.1f}%  vs baseline={base_ann:>+8.1f}%  ({tag} {delta:>+.1f}%)")

    print(f"\n  Total time: {time.time()-t_start:.1f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
