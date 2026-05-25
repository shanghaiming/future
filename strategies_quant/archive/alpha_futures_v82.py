"""
Alpha Futures V82 -- Cross-Group Sector Rotation vs Within-Group V74
=====================================================================
V74 champion: within-group catch-up (LB=1) gives +2185% with 44 commodities.
V82 tests whether CROSS-GROUP sector rotation adds alpha.

Signals:
  A_rotation:   Group ranked by today's avg return; trade weakest commodity
                in strongest group (momentum catch-up in leading sector) or
                weakest commodity in weakest group (mean-reversion).
  B_combined:   V74 within-group divergence AND cross-group direction agree.
  C_skip_own:   Compare commodity return to average of ALL OTHER groups.
  D_zscore:     Z-score of own return vs all-group distribution.
  V74_baseline: Reproduce V74 within-group signal as control.

Walk-forward: 6 windows (2020-2025), reset cash at test year start.
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

# ── Extended group map (same as V74 champion) ───────────────────────
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
    print("=" * 110)
    print("Alpha Futures V82 -- Cross-Group Sector Rotation vs Within-Group V74")
    print("=" * 110)

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

    # group_avg_ret[si, di] = average return of own group (excluding self)
    grp_avg = np.full((NS, ND), np.nan)
    for grp, members in gm_map.items():
        for di in range(1, ND):
            for sj in members:
                vals = [ret1[sk, di] for sk in members
                        if sk != sj and not np.isnan(ret1[sk, di])]
                if vals:
                    grp_avg[sj, di] = np.mean(vals)

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

    # Precompute: for each si, the average of ALL OTHER groups
    # other_groups_avg[si, di] = mean of group_total_avg for groups != own_group
    other_groups_avg = np.full((NS, ND), np.nan)
    for si in trade_sis:
        own_grp = si_group[si]
        for di in range(1, ND):
            vals = [grp_total[g][di] for g in group_names
                    if g != own_grp and not np.isnan(grp_total[g][di])]
            if vals:
                other_groups_avg[si, di] = np.mean(vals)

    # all_groups_avg[di] = grand mean of all group averages
    all_groups_avg = np.full(ND, np.nan)
    all_groups_std = np.full(ND, np.nan)
    for di in range(1, ND):
        vals = [grp_total[g][di] for g in group_names if not np.isnan(grp_total[g][di])]
        if len(vals) >= 2:
            all_groups_avg[di] = np.mean(vals)
            all_groups_std[di] = np.std(vals)

    # Group rank (1 = strongest/highest return, N = weakest)
    grp_rank_arr = np.full(ND, None, dtype=object)
    for di in range(1, ND):
        ranked = sorted(group_names, key=lambda g: -(grp_total[g][di] if not np.isnan(grp_total[g][di]) else -999))
        grp_rank_arr[di] = {g: r + 1 for r, g in enumerate(ranked)}

    print(f"  Signals computed ({time.time()-t0:.1f}s)")

    # ══════════════════════════════════════════════════════════════════
    # BACKTEST ENGINE
    # ══════════════════════════════════════════════════════════════════
    def run_backtest(config, wf_test_year=None):
        """
        Config:
            signal: 'A_rotation' | 'B_combined' | 'C_skip_own' | 'D_zscore' | 'V74_baseline'
            rotation_dir: 'strong_weak' | 'weak_weak' | 'strong_any'
                          (only for A_rotation)
            threshold: float
            top_n: 1 | 3
            comm: float
        """
        sig_type = config['signal']
        threshold = config['threshold']
        top_n = config['top_n']
        comm = config.get('comm', COMM)
        rot_dir = config.get('rotation_dir', 'strong_weak')

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
            candidates = []  # (si, score, direction, sym)

            # ── Signal A: Cross-group rotation ───────────────────────
            # Rank groups by today's average return.
            # strong_weak: buy the WEAKEST commodity in the STRONGEST group
            # weak_weak:   buy the WEAKEST commodity in the WEAKEST group (mean-reversion)
            # strong_any:  buy any commodity in the STRONGEST group that lags its group
            if sig_type == 'A_rotation':
                ranks = grp_rank_arr[di]
                if ranks is None:
                    continue

                # Pick target group(s) based on rotation direction
                if rot_dir == 'strong_weak':
                    # Target = strongest group (rank 1)
                    target_group = [g for g, r in ranks.items() if r == 1]
                elif rot_dir == 'weak_weak':
                    # Target = weakest group (max rank)
                    max_rank = max(ranks.values())
                    target_group = [g for g, r in ranks.items() if r == max_rank]
                elif rot_dir == 'strong_any':
                    target_group = [g for g, r in ranks.items() if r == 1]
                else:
                    target_group = []

                for tgt_grp in target_group:
                    members = gm_map[tgt_grp]
                    if rot_dir in ('strong_weak', 'weak_weak'):
                        # Find weakest commodity in target group
                        member_rets = []
                        for si in members:
                            r = ret1[si, di]
                            if np.isnan(r):
                                continue
                            cc = C[si, di]
                            if np.isnan(cc) or cc <= 0:
                                continue
                            if any(p['si'] == si for p in positions):
                                continue
                            member_rets.append((si, r, syms[si]))
                        # Sort ascending by return (weakest first)
                        member_rets.sort(key=lambda x: x[1])
                        # Weakest must be negative enough (lagging)
                        for si, r, sym in member_rets:
                            if r < -threshold:
                                score = grp_total[tgt_grp][di] - r  # group strong, self weak
                                candidates.append((si, score, 1, sym))
                                break  # only the weakest
                    elif rot_dir == 'strong_any':
                        # Buy any commodity in strongest group that lags group average
                        for si in members:
                            own = ret1[si, di]
                            ga = grp_avg[si, di]
                            if np.isnan(own) or np.isnan(ga):
                                continue
                            cc = C[si, di]
                            if np.isnan(cc) or cc <= 0:
                                continue
                            if any(p['si'] == si for p in positions):
                                continue
                            div = ga - own  # within-group catch-up
                            if div > threshold:
                                candidates.append((si, div, 1, syms[si]))

            # ── Signal B: Combined within-group + cross-group ────────
            # V74 within-group divergence AND group is strong (top half)
            elif sig_type == 'B_combined':
                ranks = grp_rank_arr[di]
                if ranks is None:
                    continue
                n_groups = len(ranks)
                half = n_groups // 2

                for si in trade_sis:
                    own = ret1[si, di]
                    ga = grp_avg[si, di]
                    if np.isnan(own) or np.isnan(ga):
                        continue
                    cc = C[si, di]
                    if np.isnan(cc) or cc <= 0:
                        continue
                    if any(p['si'] == si for p in positions):
                        continue

                    # V74 within-group signal
                    div = ga - own
                    if abs(div) <= threshold:
                        continue

                    # Cross-group: is this group strong or weak?
                    g = si_group[si]
                    g_rank = ranks.get(g, n_groups)

                    # Strong signal: within-group buy AND group is leading
                    if div > threshold and g_rank <= half:
                        score = div * (1 + (n_groups - g_rank) / n_groups)
                        candidates.append((si, score, 1, syms[si]))
                    # Weaker signal: within-group buy BUT group is lagging
                    elif div > threshold and g_rank > half:
                        score = div * 0.5  # penalize
                        candidates.append((si, score, 1, syms[si]))

            # ── Signal C: Skip-own-group market comparison ───────────
            # If all other groups avg > own return AND > own group avg -> buy
            elif sig_type == 'C_skip_own':
                for si in trade_sis:
                    own = ret1[si, di]
                    oga = other_groups_avg[si, di]
                    own_grp = si_group[si]
                    own_grp_avg = grp_total[own_grp][di]
                    if np.isnan(own) or np.isnan(oga) or np.isnan(own_grp_avg):
                        continue
                    cc = C[si, di]
                    if np.isnan(cc) or cc <= 0:
                        continue
                    if any(p['si'] == si for p in positions):
                        continue

                    # Commodity lags the entire market
                    if oga > own and oga - own > threshold:
                        score = oga - own
                        candidates.append((si, score, 1, syms[si]))

            # ── Signal D: Group-relative z-score ─────────────────────
            # z = (own_return - all_groups_avg) / std(group_returns)
            # If z < -threshold -> buy (unusually weak vs all groups)
            elif sig_type == 'D_zscore':
                aga = all_groups_avg[di]
                ags = all_groups_std[di]
                if np.isnan(aga) or np.isnan(ags) or ags < 1e-8:
                    continue
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
                        score = -z
                        candidates.append((si, score, 1, syms[si]))

            # ── V74 baseline: within-group catch-up ──────────────────
            elif sig_type == 'V74_baseline':
                for si in trade_sis:
                    own = ret1[si, di]
                    ga = grp_avg[si, di]
                    if np.isnan(own) or np.isnan(ga):
                        continue
                    cc = C[si, di]
                    if np.isnan(cc) or cc <= 0:
                        continue
                    if any(p['si'] == si for p in positions):
                        continue
                    div = ga - own
                    if div > threshold:
                        candidates.append((si, div, 1, syms[si]))

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

    # ══════════════════════════════════════════════════════════════════
    # BUILD CONFIGURATIONS
    # ══════════════════════════════════════════════════════════════════
    print("\n[Sweep] Building configurations...", flush=True)
    configs = []
    cid = 0

    for sig in ['A_rotation', 'B_combined', 'C_skip_own', 'D_zscore', 'V74_baseline']:
        for thresh in [0.003, 0.005, 0.01]:
            for tn in [1, 3]:
                if sig == 'A_rotation':
                    for rd in ['strong_weak', 'weak_weak', 'strong_any']:
                        cid += 1
                        label = f"A_{rd}_T{thresh}_TN{tn}"
                        configs.append({
                            'id': cid, 'signal': sig, 'rotation_dir': rd,
                            'threshold': thresh, 'top_n': tn, 'comm': COMM,
                            'label': label,
                        })
                elif sig == 'D_zscore':
                    # For D, threshold is z-score cutoff, use different values
                    for zt in [0.5, 1.0, 1.5]:
                        cid += 1
                        label = f"D_zscore_Z{zt}_TN{tn}"
                        configs.append({
                            'id': cid, 'signal': sig,
                            'threshold': zt, 'top_n': tn, 'comm': COMM,
                            'label': label,
                        })
                else:
                    cid += 1
                    label = f"{sig}_T{thresh}_TN{tn}"
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
    print(f"\n{'=' * 130}")
    print("  SIGNAL COMPARISON (Best per signal type, full period)")
    print(f"{'=' * 130}")
    print(f"  {'Signal':<25} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | Best Config")
    print("-" * 130)

    best_per_sig = {}
    for r in results:
        s = r['config']['signal']
        if s not in best_per_sig:
            best_per_sig[s] = r

    sig_order = ['V74_baseline', 'A_rotation', 'B_combined', 'C_skip_own', 'D_zscore']
    for sig in sig_order:
        if sig in best_per_sig:
            b = best_per_sig[sig]
            print(f"  {sig:<25} | {b['ann']:>+8.1f}% | {b['wr']:>5.1f}% | {b['n']:>5} | {b['avg_pnl']:>+6.3f}% | {b['mdd']:>6.1f}% | {b['label']}")

    # Alpha vs V74
    v74_base = best_per_sig.get('V74_baseline')
    if v74_base:
        print(f"\n  V74 Baseline: {v74_base['ann']:>+8.1f}%")
        for sig in ['A_rotation', 'B_combined', 'C_skip_own', 'D_zscore']:
            if sig in best_per_sig:
                diff = best_per_sig[sig]['ann'] - v74_base['ann']
                tag = "ADD" if diff > 0 else "NO"
                print(f"  {sig:<25} {best_per_sig[sig]['ann']:>+8.1f}%  ({tag} {diff:>+.1f}% alpha)")

    # ══════════════════════════════════════════════════════════════════
    # WALK-FORWARD (Top 10 + best per signal)
    # ══════════════════════════════════════════════════════════════════
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    wf_configs = list(results[:10])
    for sig in sig_order:
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

    # ══════════════════════════════════════════════════════════════════
    # WF COMPARISON PER SIGNAL
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 130}")
    print("  WALK-FORWARD COMPARISON (Best per signal type)")
    print(f"{'=' * 130}")
    header2 = f"  {'Signal':<25} | {'WF Avg':>8} |"
    for yr in wf_years:
        header2 += f" {yr:>7} |"
    header2 += f" {'Pos':>4}"
    print(header2)
    print("-" * 130)

    for sig in sig_order:
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

    # ══════════════════════════════════════════════════════════════════
    # A_ROTATION DETAILED BREAKDOWN
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 130}")
    print("  A_ROTATION BREAKDOWN (by rotation direction)")
    print(f"{'=' * 130}")
    for rd in ['strong_weak', 'weak_weak', 'strong_any']:
        rd_results = [r for r in results if r['config']['signal'] == 'A_rotation'
                      and r['config'].get('rotation_dir') == rd]
        if rd_results:
            best = rd_results[0]
            print(f"  {rd:<15} | Best: {best['ann']:>+8.1f}%  WR {best['wr']:>5.1f}%  N {best['n']:>5}  | {best['label']}")

    # ══════════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 130}")
    print("  FINAL VERDICT")
    print(f"{'=' * 130}")

    if v74_base:
        v74_ann = v74_base['ann']
        print(f"  V74 Baseline (within-group):  {v74_ann:>+8.1f}%")

        cross_sigs = ['A_rotation', 'B_combined', 'C_skip_own', 'D_zscore']
        best_cross = None
        for sig in cross_sigs:
            if sig in best_per_sig:
                b = best_per_sig[sig]
                if best_cross is None or b['ann'] > best_cross['ann']:
                    best_cross = b

        if best_cross:
            diff = best_cross['ann'] - v74_ann
            print(f"  Best cross-group signal:      {best_cross['ann']:>+8.1f}%  ({best_cross['label']})")
            if diff > 0:
                print(f"  >>> CROSS-GROUP ROTATION ADDS +{diff:.1f}% ALPHA OVER V74 <<<")
            else:
                print(f"  >>> CROSS-GROUP ROTATION DOES NOT ADD ALPHA ({diff:+.1f}%) <<<")
                print(f"  >>> V74 WITHIN-GROUP SIGNAL REMAINS CHAMPION <<<")

        # Check combined (B) which uses both signals
        if 'B_combined' in best_per_sig:
            b_ann = best_per_sig['B_combined']['ann']
            diff_b = b_ann - v74_ann
            print(f"  B_combined (within+cross):    {b_ann:>+8.1f}%  ({diff_b:+.1f}% vs V74)")
            if diff_b > 0:
                print(f"  >>> COMBINED SIGNAL IS SUPERIOR <<<")

    print(f"\n  Total time: {time.time()-t_start:.1f}s")
    print("=" * 110)


if __name__ == '__main__':
    main()
