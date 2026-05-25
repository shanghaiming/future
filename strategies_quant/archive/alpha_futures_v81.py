"""
Alpha Futures V81 — Volume-Weighted Group Momentum vs Equal Weight
===================================================================
V74 champion: extended groups (44 commodities, 8 groups), LB=1, 1-day hold,
EQUAL weight for all group members. +2185% annual.

V81 idea: Weight group members by volume/OI instead of equal weight.
Higher-volume commodities should lead the group more. When the volume-weighted
group average moves but a commodity doesn't, the divergence is more meaningful
because high-volume members "confirmed" the group move.

Weighting schemes:
  A: Equal weight (V74 baseline)
  B: Volume-weighted
  C: OI-weighted
  D: Vol*Price weighted
  E: Inverse volume (contrarian within group)

Volume filters:
  none:  No filter
  F:     Only trade when signal commodity volume > 20-day avg (confirmation)
  G:     Only trade when signal commodity volume < 20-day avg (contrarian)

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
        'sffi': 5, 'smfi': 5, 'pbfi': 5, 'snfi': 1, 'afi': 10, 'bfi': 10,
        'cffi': 5, 'cfi': 10, 'csfi': 10, 'ebfi': 5, 'egfi': 10, 'fbfi': 500,
        'ifi': 100, 'jfi': 100, 'jmfi': 60, 'lfi': 5, 'mfi': 10, 'pgfi': 20,
        'ppfi': 5, 'vfi': 5, 'yfi': 10, 'pfi': 10, 'jdfi': 5, 'lhfi': 16,
        'pkfi': 5, 'rrfi': 20, 'lrfi': 20, 'whfi': 20, 'cjfi': 10, 'mafi': 10,
        'apfi': 10, 'fgfi': 20, 'oifi': 10, 'rmfi': 10, 'srfi': 10, 'tafi': 5,
        'scfi': 1000, 'lufi': 10, 'bcfi': 5, 'nrfi': 1, 'lgfi': 20, 'brfi': 5}
DEF_MULT = 10
COMM = 0.0003

# ── Extended group map (same as V74) ────────────────────────────────
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
    print("=" * 110)
    print("Alpha Futures V81 -- Volume-Weighted Group Momentum vs Equal Weight")
    print("=" * 110)

    # ── Load data ────────────────────────────────────────────────────
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ── Build group membership ───────────────────────────────────────
    gm_map = {}  # group_name -> list of si
    for si in range(NS):
        g = GROUP_MAP.get(syms[si])
        if g:
            gm_map.setdefault(g, []).append(si)

    trade_sis = []
    for si in range(NS):
        if GROUP_MAP.get(syms[si]):
            trade_sis.append(si)

    print(f"  Groups: {len(gm_map)} groups, {len(trade_sis)} tradeable commodities")
    for grp, members in sorted(gm_map.items()):
        print(f"    {grp:<12s}: {', '.join(syms[m] for m in members)}")

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
    print(f"  1-day returns done ({time.time()-t0:.1f}s)")

    # ── Precompute volume 20-day average ─────────────────────────────
    print("[Signals] Computing 20-day volume averages...", flush=True)
    t0 = time.time()
    vol_avg20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            vals = V[si, max(0, di-20):di]
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                vol_avg20[si, di] = np.mean(valid)
    print(f"  Volume averages done ({time.time()-t0:.1f}s)")

    # ── Precompute weighted group averages for each scheme ───────────
    # Schemes: A_equal, B_vol, C_oi, D_vol_price, E_inverse_vol
    # For each scheme, compute: grp_avg_w[si, di] = weighted avg of ret1[group_members\{si}, di]
    print("[Signals] Computing weighted group averages...", flush=True)
    t0 = time.time()

    grp_avg_w = {}  # scheme -> np.array (NS, ND)
    schemes = ['A_equal', 'B_vol', 'C_oi', 'D_vol_price', 'E_inverse_vol']

    for scheme in schemes:
        ga = np.full((NS, ND), np.nan)
        for grp, members in gm_map.items():
            nm = len(members)
            for di in range(1, ND):
                # Compute weight for each member (indexed same as members list)
                ws = np.zeros(nm)
                rs_valid = np.full(nm, np.nan)

                for idx, sk in enumerate(members):
                    r = ret1[sk, di]
                    if np.isnan(r):
                        continue
                    rs_valid[idx] = r
                    v = V[sk, di]
                    oi = OI[sk, di]
                    c = C[sk, di]
                    if scheme == 'A_equal':
                        ws[idx] = 1.0
                    elif scheme == 'B_vol':
                        ws[idx] = v if not np.isnan(v) and v > 0 else 0.0
                    elif scheme == 'C_oi':
                        ws[idx] = oi if not np.isnan(oi) and oi > 0 else 0.0
                    elif scheme == 'D_vol_price':
                        if not np.isnan(v) and not np.isnan(c) and v > 0 and c > 0:
                            ws[idx] = v * c
                        else:
                            ws[idx] = 0.0
                    elif scheme == 'E_inverse_vol':
                        ws[idx] = (1.0 / v) if not np.isnan(v) and v > 0 else 0.0

                # For each member, compute leave-one-out weighted average
                for idx, sj in enumerate(members):
                    own_r = rs_valid[idx]
                    if np.isnan(own_r):
                        continue
                    # Leave-one-out: sum weights and weighted returns excluding sj
                    loo_w = 0.0
                    loo_wr = 0.0
                    for k in range(nm):
                        if k == idx:
                            continue
                        rk = rs_valid[k]
                        if np.isnan(rk):
                            continue
                        loo_w += ws[k]
                        loo_wr += ws[k] * rk
                    if loo_w > 0:
                        ga[sj, di] = loo_wr / loo_w

        grp_avg_w[scheme] = ga
        print(f"    {scheme}: done")

    print(f"  Weighted group averages done ({time.time()-t0:.1f}s)")

    # ══════════════════════════════════════════════════════════════════
    # BACKTEST ENGINE
    # ══════════════════════════════════════════════════════════════════
    def run_backtest(config, wf_test_year=None):
        """
        Config dict:
            scheme: 'A_equal' | 'B_vol' | 'C_oi' | 'D_vol_price' | 'E_inverse_vol'
            vol_filter: 'none' | 'F_above_avg' | 'G_below_avg'
            threshold: float
            top_n: 1 | 3
            comm: float
        """
        scheme = config['scheme']
        vol_filter = config.get('vol_filter', 'none')
        threshold = config['threshold']
        top_n = config['top_n']
        comm = config.get('comm', COMM)

        ga = grp_avg_w[scheme]

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
                    mkt_val = cn * mult * pos['lots']
                    cash += mkt_val - mkt_val * comm
                    pnl = (cn - pos['entry']) * mult * pos['lots'] * pos['dir']
                    invested = pos['entry'] * mult * pos['lots']
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

            # ── Score candidates ─────────────────────────────────────
            candidates = []
            for si in trade_sis:
                sym = syms[si]
                if np.isnan(C[si, di]) or C[si, di] <= 0:
                    continue
                if any(p['si'] == si for p in positions):
                    continue

                own = ret1[si, di]
                gavg = ga[si, di]
                if np.isnan(own) or np.isnan(gavg):
                    continue

                div = gavg - own  # positive = group ahead, commodity lagging

                if div <= threshold:
                    continue

                # Volume filter
                if vol_filter == 'F_above_avg':
                    v_now = V[si, di]
                    v_avg = vol_avg20[si, di]
                    if np.isnan(v_now) or np.isnan(v_avg) or v_now <= v_avg:
                        continue
                elif vol_filter == 'G_below_avg':
                    v_now = V[si, di]
                    v_avg = vol_avg20[si, di]
                    if np.isnan(v_now) or np.isnan(v_avg) or v_now >= v_avg:
                        continue

                candidates.append((si, div, 1))  # long laggards

            if not candidates:
                continue

            # Sort by score (highest divergence first)
            candidates.sort(key=lambda x: -x[1])

            # Open positions
            n_slots = top_n - len(positions)
            for si, score, direction in candidates[:max(0, n_slots)]:
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

        # Results
        wf_mode = wf_test_year is not None
        n_days_test = (test_end_di - test_start_di) if wf_mode else (ND - start_di)
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

    for scheme in ['A_equal', 'B_vol', 'C_oi', 'D_vol_price', 'E_inverse_vol']:
        for vf in ['none', 'F_above_avg', 'G_below_avg']:
            for thresh in [0.003, 0.005, 0.01]:
                for tn in [1, 3]:
                    cid += 1
                    label = f"{scheme}_VF{vf}_T{thresh}_TN{tn}"
                    configs.append({
                        'id': cid,
                        'scheme': scheme,
                        'vol_filter': vf,
                        'threshold': thresh,
                        'top_n': tn,
                        'comm': COMM,
                        'label': label,
                    })

    print(f"  Total configs: {len(configs)}")

    # ══════════════════════════════════════════════════════════════════
    # RUN FULL-PERIOD BACKTEST
    # ══════════════════════════════════════════════════════════════════
    print("\n[Backtest] Running full-period sweep...", flush=True)
    t_bt = time.time()
    results = []
    for i, cfg in enumerate(configs):
        r = run_backtest(cfg)
        if r:
            r['config'] = cfg
            r['label'] = cfg['label']
            results.append(r)
        if (i + 1) % 15 == 0 or i == len(configs) - 1:
            print(f"  ... {i+1}/{len(configs)} done", flush=True)

    results.sort(key=lambda x: -x['ann'])
    print(f"  Full-period sweep done ({time.time()-t_bt:.1f}s)")

    # Print top 20
    print("\n" + "=" * 130)
    print("  FULL-PERIOD RESULTS (Top 20)")
    print("=" * 130)
    print(f"  {'#':>3} | {'Label':<55} | {'Ann':>8} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7}")
    print("-" * 120)
    for i, r in enumerate(results[:20]):
        print(f"  {i+1:>3} | {r['label']:<55} | {r['ann']:>+7.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}%")

    # ══════════════════════════════════════════════════════════════════
    # WALK-FORWARD (Top 10)
    # ══════════════════════════════════════════════════════════════════
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Ensure at least 1 config per scheme in WF
    best_per_scheme = {}
    for r in results:
        s = r['config']['scheme']
        if s not in best_per_scheme:
            best_per_scheme[s] = r

    wf_configs = list(results[:10])
    for s, r in best_per_scheme.items():
        if r not in wf_configs:
            wf_configs.append(r)

    print(f"\n{'=' * 150}")
    print(f"  WALK-FORWARD ({len(wf_configs)} configs)")
    print(f"{'=' * 150}")

    header = f"  {'#':>3} | {'Config':<55} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>7} |"
    header += f" {'Pos':>4} | {'MDD':>7}"
    print(header)
    print("-" * 150)

    wf_rows = []
    for i, r in enumerate(wf_configs):
        cfg = r['config']
        wf_row = {'label': cfg['label'], 'scheme': cfg['scheme'], 'windows': {}, 'mdd': {}}
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

        row_str = f"  {i+1:>3} | {wf_row['label']:<55} | {avg:>+7.1f}% |"
        for v in vals:
            row_str += f" {v:>+7.1f}% |"
        row_str += f" {pos}/6 | {avg_mdd:>6.1f}%"
        print(row_str)

    # ══════════════════════════════════════════════════════════════════
    # KEY COMPARISONS
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 110}")
    print("  KEY COMPARISON: Weighting Scheme (best per scheme, full period)")
    print(f"{'=' * 110}")
    print(f"  {'Scheme':<20} | {'Ann':>8} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | Best Config")
    print("-" * 110)

    for scheme in ['A_equal', 'B_vol', 'C_oi', 'D_vol_price', 'E_inverse_vol']:
        scheme_results = [r for r in results if r['config']['scheme'] == scheme]
        if scheme_results:
            best = scheme_results[0]
            marker = " <<< BASELINE" if scheme == 'A_equal' else ""
            print(f"  {scheme:<20} | {best['ann']:>+7.1f}% | {best['wr']:>5.1f}% | {best['n']:>5} | {best['avg_pnl']:>+6.3f}% | {best['mdd']:>6.1f}% | {best['label']}{marker}")

    # WF comparison per scheme
    print(f"\n{'=' * 110}")
    print("  WALK-FORWARD COMPARISON (Best per scheme)")
    print(f"{'=' * 110}")
    header2 = f"  {'Scheme':<20} | {'WF Avg':>8} |"
    for yr in wf_years:
        header2 += f" {yr:>7} |"
    header2 += f" {'Pos':>4}"
    print(header2)
    print("-" * 110)

    for scheme in ['A_equal', 'B_vol', 'C_oi', 'D_vol_price', 'E_inverse_vol']:
        wf_match = [w for w in wf_rows if w['scheme'] == scheme]
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            avg = np.mean(vals) if vals else 0
            pos = sum(1 for v in vals if v > 0)
            row_str = f"  {scheme:<20} | {avg:>+7.1f}% |"
            for v in vals:
                row_str += f" {v:>+7.1f}% |"
            row_str += f" {pos}/6"
            marker = " <<< BASELINE" if scheme == 'A_equal' else ""
            print(f"{row_str}{marker}")

    # Volume filter comparison
    print(f"\n{'=' * 110}")
    print("  VOLUME FILTER COMPARISON (best per filter, full period)")
    print(f"{'=' * 110}")
    print(f"  {'Filter':<20} | {'Ann':>8} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | Best Config")
    print("-" * 110)

    for vf in ['none', 'F_above_avg', 'G_below_avg']:
        vf_results = [r for r in results if r['config']['vol_filter'] == vf]
        if vf_results:
            best = vf_results[0]
            print(f"  {vf:<20} | {best['ann']:>+7.1f}% | {best['wr']:>5.1f}% | {best['n']:>5} | {best['avg_pnl']:>+6.3f}% | {best['mdd']:>6.1f}% | {best['label']}")

    # WF volume filter comparison
    print(f"\n{'=' * 110}")
    print("  WALK-FORWARD VOLUME FILTER COMPARISON (Best per filter)")
    print(f"{'=' * 110}")

    # Gather best WF per vol_filter
    vf_wf_rows = {}
    for vf in ['none', 'F_above_avg', 'G_below_avg']:
        vf_results = [r for r in results if r['config']['vol_filter'] == vf]
        if vf_results:
            best_cfg = vf_results[0]['config']
            wf_row = {'vol_filter': vf, 'windows': {}}
            for yr in wf_years:
                wr = run_backtest(best_cfg, wf_test_year=yr)
                if wr:
                    wf_row['windows'][yr] = wr['ann']
            vf_wf_rows[vf] = wf_row

    header3 = f"  {'Filter':<20} | {'WF Avg':>8} |"
    for yr in wf_years:
        header3 += f" {yr:>7} |"
    header3 += f" {'Pos':>4}"
    print(header3)
    print("-" * 110)

    for vf in ['none', 'F_above_avg', 'G_below_avg']:
        if vf in vf_wf_rows:
            wf = vf_wf_rows[vf]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            avg = np.mean(vals) if vals else 0
            pos = sum(1 for v in vals if v > 0)
            row_str = f"  {vf:<20} | {avg:>+7.1f}% |"
            for v in vals:
                row_str += f" {v:>+7.1f}% |"
            row_str += f" {pos}/6"
            print(row_str)

    # ── Champion analysis ────────────────────────────────────────────
    print(f"\n{'=' * 110}")
    print("  CHAMPION ANALYSIS")
    print(f"{'=' * 110}")
    if results:
        champ = results[0]
        print(f"  Best full-period: {champ['label']}")
        print(f"    Annual return: {champ['ann']:>+.1f}%")
        print(f"    Win rate:      {champ['wr']:>.1f}%")
        print(f"    Trades:        {champ['n']}")
        print(f"    Avg trade PnL: {champ['avg_pnl']:>+.3f}%")
        print(f"    Max DD:        {champ['mdd']:>.1f}%")

        # Compare equal weight (baseline) vs best non-equal
        equal_best = [r for r in results if r['config']['scheme'] == 'A_equal']
        non_equal_best = [r for r in results if r['config']['scheme'] != 'A_equal']
        if equal_best and non_equal_best:
            eq_r = equal_best[0]
            ne_r = non_equal_best[0]
            print(f"\n  V74 baseline (A_equal best):  {eq_r['ann']:>+8.1f}%  {eq_r['label']}")
            print(f"  Best non-equal weight:         {ne_r['ann']:>+8.1f}%  {ne_r['label']}")
            diff = ne_r['ann'] - eq_r['ann']
            if diff > 0:
                print(f"  >>> Volume weighting ADDS +{diff:.1f}% alpha over equal weight <<<")
            else:
                print(f"  >>> Volume weighting does NOT add alpha ({diff:+.1f}%) <<<")

        # WF champion vs WF baseline
        eq_wf = [w for w in wf_rows if w['scheme'] == 'A_equal']
        ne_wf = [w for w in wf_rows if w['scheme'] != 'A_equal']
        if eq_wf and ne_wf:
            eq_vals = [eq_wf[0]['windows'].get(yr, 0) for yr in wf_years]
            ne_vals = [ne_wf[0]['windows'].get(yr, 0) for yr in wf_years]
            eq_avg = np.mean(eq_vals)
            ne_avg = np.mean(ne_vals)
            print(f"\n  WF Avg (A_equal best):  {eq_avg:>+8.1f}%")
            print(f"  WF Avg (best non-equal, {ne_wf[0]['scheme']}): {ne_avg:>+8.1f}%  {ne_wf[0]['label']}")
            diff = ne_avg - eq_avg
            if diff > 0:
                print(f"  >>> Volume weighting ADDS +{diff:.1f}% WF alpha <<<")
            else:
                print(f"  >>> Volume weighting does NOT add WF alpha ({diff:+.1f}%) <<<")

    print(f"\n  Total time: {time.time()-t_start:.1f}s")
    print("=" * 110)


if __name__ == '__main__':
    main()
