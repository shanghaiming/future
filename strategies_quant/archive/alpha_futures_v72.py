"""
Alpha Futures V72 -- Group Momentum Lag + Dynamic Position Sizing
=================================================================
V63 discovered that 1-day hold group momentum lag at LB=3 gives +448.9% annual
with FIXED position sizing (buy max lots with available cash).

V72 QUESTION: Can DYNAMIC position sizing push returns even higher?
Allocate MORE capital when the signal is stronger, LESS when weaker.

3 SIZING MODES:
  Mode A: Signal-Strength Sizing
    - score = grp_mom - own_mom
    - cash_fraction = min(score * scale_factor, 1.0)
    - lots = int(cash_fraction * cash / (price * mult))
  Mode B: Top-3 Equal Split
    - Take TOP-3 commodities by score, split cash equally (1/3 each)
  Mode C: Kelly Criterion (rolling 60-day)
    - Estimate WR and avg win/loss from history
    - Kelly fraction = WR - (1-WR)/(avg_win/avg_loss)
    - lots = int(kelly_fraction * cash / (price * mult))
  Baseline: fixed_full (buy max lots, as in V63)

Signal: group momentum lag (grp_mom - own_mom) > threshold
  - own_mom = price change over LB days for this commodity
  - grp_mom = average own_mom of OTHER commodities in same group
  - score > 0 means the group is moving but this one has lagged -> expect catch-up

Configs: LB x threshold x sizing_mode x (scale for Mode A)
  - LB: [2, 3, 5]
  - Threshold: [0.003, 0.005]
  - Sizing: [fixed_full, signal_strength, top3_equal, kelly_60]
  - Scale: [50, 100, 200] (Mode A only)
  - COMM: 0.0003

Walk-forward: 6 expanding windows (test years 2020-2025)
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

GROUP_MAP = {
    'rbfi': 'ferrous', 'hcfi': 'ferrous', 'ifi': 'ferrous', 'jfi': 'ferrous', 'jmfi': 'ferrous',
    'cufi': 'nonferrous', 'alfi': 'nonferrous', 'znfi': 'nonferrous', 'nifi': 'nonferrous',
    'aufi': 'precious', 'agfi': 'precious',
    'afi': 'oils', 'mfi': 'oils', 'yfi': 'oils', 'pfi': 'oils', 'cfi': 'oils',
    'scfi': 'energy', 'mafi': 'energy', 'bfi': 'energy', 'fufi': 'energy', 'pgfi': 'energy',
    'ppfi': 'chemical', 'vfi': 'chemical', 'egfi': 'chemical', 'srfi': 'chemical',
}


def main():
    t_start = time.time()
    print("=" * 110)
    print("V72 -- Momentum + Dynamic Position Sizing")
    print("=" * 110)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    year_start_di = {}
    year_end_di = {}
    for di in range(ND):
        y = dates[di].year
        if y not in year_start_di:
            year_start_di[y] = di
        year_end_di[y] = di
    print(f"  {NS} commodities, {ND} days, years in data: {sorted(year_start_di.keys())}")

    # Build group membership
    group_members = {}
    group_sis = set()
    for si in range(NS):
        grp = GROUP_MAP.get(syms[si])
        if grp is None:
            continue
        group_sis.add(si)
        if grp not in group_members:
            group_members[grp] = []
        group_members[grp].append(si)
    print(f"  Groups: {len(group_members)}, commodities in groups: {len(group_sis)}")

    # ================================================================
    # PRECOMPUTE MOMENTUM SIGNALS
    # ================================================================
    print("\n[Signals] Precomputing momentum signals...", flush=True)
    t0 = time.time()

    mom = {}
    for lag in [2, 3, 5]:
        m = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(lag, ND):
                c_now = C[si, di]
                c_prev = C[si, di - lag]
                if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                    m[si, di] = (c_now - c_prev) / c_prev
        mom[lag] = m

    grp_mom = {}
    for lag in [2, 3, 5]:
        gm = np.full((NS, ND), np.nan)
        for grp, members in group_members.items():
            for di in range(lag, ND):
                for sj in members:
                    ms = []
                    for sk in members:
                        if sk == sj:
                            continue
                        mv = mom[lag][sk, di]
                        if not np.isnan(mv):
                            ms.append(mv)
                    if ms:
                        gm[sj, di] = np.mean(ms)
        grp_mom[lag] = gm

    # Precompute 1-day forward returns (close-to-close) for each commodity
    fwd_ret = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND - 1):
            c0 = C[si, di]
            c1 = C[si, di + 1]
            if not np.isnan(c0) and not np.isnan(c1) and c0 > 0:
                fwd_ret[si, di] = (c1 - c0) / c0

    print(f"  Momentum signals precomputed ({time.time() - t0:.1f}s)", flush=True)

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(lb=3, threshold=0.003, sizing_mode='fixed_full',
                     signal_scale=100, start_year=None, end_year=None,
                     config_name=""):
        """
        Run backtest with specified position sizing mode.

        sizing_mode:
          'fixed_full'      - baseline: buy max lots with all cash
          'signal_strength' - lots = int(min(score * scale, 1.0) * cash / (price * mult))
          'top3_equal'      - take top 3 scored commodities, each gets cash/3
          'kelly_60'        - Kelly criterion with rolling 60-day estimation
        """
        cash = float(CASH0)
        trades = []

        start_di = MIN_TRAIN
        end_di = ND
        if start_year is not None:
            if start_year in year_start_di:
                start_di = year_start_di[start_year]
            else:
                return None
        if end_year is not None:
            if end_year in year_end_di:
                end_di = year_end_di[end_year] + 1
            else:
                return None

        # For Kelly: track rolling trade history per commodity
        kelly_history = {}  # si -> list of (pnl_pct) from last 60 days
        kelly_stats = {}    # si -> (wr, avg_win, avg_loss, kelly_fraction)

        # For multi-position tracking: positions that need to be closed next day
        # Each position: {'si', 'sym', 'entry', 'lots', 'cash_used', 'entry_di'}
        active_positions = []

        for di in range(start_di, end_di):
            year = dates[di].year

            # --- Close all positions from previous day ---
            closed_positions = []
            remaining = []
            for pos in active_positions:
                if di - pos['entry_di'] >= 1:
                    # Close this position
                    cn = C[pos['si'], di]
                    if np.isnan(cn) or cn <= 0:
                        cn = pos['entry']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    exit_value = cn * mult * pos['lots']
                    entry_value = pos['entry'] * mult * pos['lots']
                    pnl_abs = exit_value - entry_value
                    cost = (entry_value + exit_value) * COMM
                    pnl_abs -= cost
                    pnl_pct = pnl_abs / entry_value * 100 if entry_value > 0 else 0
                    cash += exit_value - exit_value * COMM
                    trades.append({
                        'pnl_abs': pnl_abs, 'pnl_pct': pnl_pct,
                        'di': di, 'year': year, 'sym': pos['sym'],
                        'si': pos['si'], 'score': pos.get('score', 0),
                        'lots': pos['lots'],
                    })
                    closed_positions.append(pos)
                    # Update Kelly history
                    if sizing_mode == 'kelly_60':
                        si_k = pos['si']
                        if si_k not in kelly_history:
                            kelly_history[si_k] = []
                        kelly_history[si_k].append((di, pnl_pct))
                else:
                    remaining.append(pos)
            active_positions = remaining

            # Prune Kelly history to last 60 days
            if sizing_mode == 'kelly_60' and di % 20 == 0:
                for si_k in kelly_history:
                    kelly_history[si_k] = [(d, p) for d, p in kelly_history[si_k] if di - d <= 60]

            # --- Score all group commodities ---
            scored = []
            for si in group_sis:
                own = mom[lb][si, di]
                grp = grp_mom[lb][si, di]
                if np.isnan(own) or np.isnan(grp):
                    continue
                score = grp - own
                if score > threshold:
                    scored.append((si, score))

            if not scored:
                continue

            scored.sort(key=lambda x: -x[1])

            # --- Open new positions based on sizing mode ---
            if sizing_mode == 'fixed_full':
                # Baseline: buy max lots with all cash on top-1 commodity
                best_si, best_score = scored[0]
                c = C[best_si, di]
                if np.isnan(c) or c <= 0:
                    continue
                sym = syms[best_si]
                mult = MULT.get(sym, DEF_MULT)
                notional = c * mult
                lots = int(cash / (notional * (1 + COMM)))
                if lots > 0:
                    cost_in = notional * lots * (1 + COMM)
                    if cost_in <= cash:
                        cash -= cost_in
                        active_positions.append({
                            'si': best_si, 'sym': sym, 'entry': c,
                            'lots': lots, 'entry_di': di, 'score': best_score,
                        })

            elif sizing_mode == 'signal_strength':
                # Mode A: size proportional to signal strength
                best_si, best_score = scored[0]
                c = C[best_si, di]
                if np.isnan(c) or c <= 0:
                    continue
                sym = syms[best_si]
                mult = MULT.get(sym, DEF_MULT)
                cash_fraction = min(best_score * signal_scale, 1.0)
                lots = int(cash_fraction * cash / (c * mult * (1 + COMM)))
                if lots > 0:
                    cost_in = c * mult * lots * (1 + COMM)
                    if cost_in <= cash:
                        cash -= cost_in
                        active_positions.append({
                            'si': best_si, 'sym': sym, 'entry': c,
                            'lots': lots, 'entry_di': di, 'score': best_score,
                        })

            elif sizing_mode == 'top3_equal':
                # Mode B: take top-3, split cash equally
                top3 = scored[:3]
                per_slot = cash / len(top3) if top3 else 0
                for si, score in top3:
                    c = C[si, di]
                    if np.isnan(c) or c <= 0:
                        continue
                    sym = syms[si]
                    mult = MULT.get(sym, DEF_MULT)
                    lots = int(per_slot / (c * mult * (1 + COMM)))
                    if lots > 0:
                        cost_in = c * mult * lots * (1 + COMM)
                        if cost_in <= cash:
                            cash -= cost_in
                            active_positions.append({
                                'si': si, 'sym': sym, 'entry': c,
                                'lots': lots, 'entry_di': di, 'score': score,
                            })

            elif sizing_mode == 'kelly_60':
                # Mode C: Kelly criterion sizing
                # Recompute Kelly stats periodically
                if di % 5 == 0:
                    kelly_stats = {}
                    for si_k in group_sis:
                        hist = kelly_history.get(si_k, [])
                        if len(hist) < 10:
                            kelly_stats[si_k] = 0.25  # default fraction
                            continue
                        pnls = [p for _, p in hist]
                        nw = sum(1 for p in pnls if p > 0)
                        nl = len(pnls) - nw
                        if nl == 0 or nw == 0:
                            kelly_stats[si_k] = 0.5 if nw > nl else 0.1
                            continue
                        wr = nw / len(pnls)
                        avg_win = np.mean([p for p in pnls if p > 0])
                        avg_loss = abs(np.mean([p for p in pnls if p <= 0]))
                        if avg_loss > 0:
                            kelly_f = wr - (1 - wr) / (avg_win / avg_loss)
                        else:
                            kelly_f = 0.5
                        # Half-Kelly for safety, cap at 0.8
                        kelly_f = max(0.05, min(kelly_f * 0.5, 0.8))
                        kelly_stats[si_k] = kelly_f

                best_si, best_score = scored[0]
                c = C[best_si, di]
                if np.isnan(c) or c <= 0:
                    continue
                sym = syms[best_si]
                mult = MULT.get(sym, DEF_MULT)
                kelly_f = kelly_stats.get(best_si, 0.25)
                lots = int(kelly_f * cash / (c * mult * (1 + COMM)))
                if lots > 0:
                    cost_in = c * mult * lots * (1 + COMM)
                    if cost_in <= cash:
                        cash -= cost_in
                        active_positions.append({
                            'si': best_si, 'sym': sym, 'entry': c,
                            'lots': lots, 'entry_di': di, 'score': best_score,
                        })

        # Close remaining positions at end
        for pos in active_positions:
            ae = min(end_di, ND) - 1
            cn = C[pos['si'], ae]
            if np.isnan(cn) or cn <= 0:
                cn = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            exit_value = cn * mult * pos['lots']
            entry_value = pos['entry'] * mult * pos['lots']
            pnl_abs = exit_value - entry_value
            cost = (entry_value + exit_value) * COMM
            pnl_abs -= cost
            pnl_pct = pnl_abs / entry_value * 100 if entry_value > 0 else 0
            cash += exit_value - exit_value * COMM
            trades.append({
                'pnl_abs': pnl_abs, 'pnl_pct': pnl_pct,
                'di': ae, 'year': dates[ae].year, 'sym': pos['sym'],
                'si': pos['si'], 'score': pos.get('score', 0),
                'lots': pos['lots'],
            })

        if len(trades) < 3:
            return None

        # ================================================================
        # COMPUTE STATS
        # ================================================================
        equity = float(CASH0)
        peak = float(CASH0)
        max_dd = 0.0
        for t in sorted(trades, key=lambda x: x['di']):
            equity += t['pnl_abs']
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (peak - equity) / peak * 100
                if dd > max_dd:
                    max_dd = dd

        nw = sum(1 for t in trades if t['pnl_abs'] > 0)
        n_trades = len(trades)
        wr = nw / n_trades * 100
        avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_abs'] > 0]) if nw > 0 else 0
        avg_loss = np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_abs'] <= 0]) if nw < n_trades else 0
        pf = (sum(t['pnl_abs'] for t in trades if t['pnl_abs'] > 0) /
              max(abs(sum(t['pnl_abs'] for t in trades if t['pnl_abs'] < 0)), 1))

        first_di = min(t['di'] for t in trades)
        last_di = max(t['di'] for t in trades)
        days_total = (dates[last_di] - dates[first_di]).days if last_di > first_di else 365
        yr = max(days_total / 365.25, 0.01)
        ann = ((cash / CASH0) ** (1 / yr) - 1) * 100

        tp = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
        sharpe = np.mean(tp) / np.std(tp) * np.sqrt(252) / float(CASH0) if len(tp) > 1 and np.std(tp) > 0 else 0

        year_stats = {}
        for t in trades:
            y = t['year']
            if y not in year_stats:
                year_stats[y] = {'n': 0, 'w': 0, 'pnl': 0.0, 'pnl_abs_sum': 0.0}
            year_stats[y]['n'] += 1
            if t['pnl_abs'] > 0:
                year_stats[y]['w'] += 1
            year_stats[y]['pnl'] += t['pnl_pct']
            year_stats[y]['pnl_abs_sum'] += t['pnl_abs']

        return {
            'name': config_name, 'ann': round(ann, 1), 'n': n_trades,
            'wr': round(wr, 1), 'dd': round(max_dd, 1),
            'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
            'pf': round(pf, 2), 'sharpe': round(sharpe, 2),
            'cash': round(cash, 0), 'yearly': year_stats,
        }

    # ================================================================
    # BUILD CONFIGURATIONS
    # ================================================================
    print("\n[Configs] Building configurations...", flush=True)
    configs = []

    # Baseline: fixed_full (V63 style)
    for lb in [2, 3, 5]:
        for th in [0.003, 0.005]:
            configs.append({
                'lb': lb, 'threshold': th, 'sizing_mode': 'fixed_full',
                'signal_scale': 100,
                'config_name': f"FIXED_LB{lb}_TH{th*1000:.0f}",
            })

    # Mode A: Signal-Strength Sizing
    for lb in [2, 3, 5]:
        for th in [0.003, 0.005]:
            for sc in [50, 100, 200]:
                configs.append({
                    'lb': lb, 'threshold': th, 'sizing_mode': 'signal_strength',
                    'signal_scale': sc,
                    'config_name': f"SIGSTR_LB{lb}_TH{th*1000:.0f}_SC{sc}",
                })

    # Mode B: Top-3 Equal
    for lb in [2, 3, 5]:
        for th in [0.003, 0.005]:
            configs.append({
                'lb': lb, 'threshold': th, 'sizing_mode': 'top3_equal',
                'signal_scale': 100,
                'config_name': f"TOP3_LB{lb}_TH{th*1000:.0f}",
            })

    # Mode C: Kelly 60-day
    for lb in [2, 3, 5]:
        for th in [0.003, 0.005]:
            configs.append({
                'lb': lb, 'threshold': th, 'sizing_mode': 'kelly_60',
                'signal_scale': 100,
                'config_name': f"KELLY_LB{lb}_TH{th*1000:.0f}",
            })

    total_combos = len(configs)
    print(f"  {total_combos} configurations")

    # ================================================================
    # RUN SWEEP
    # ================================================================
    print(f"\n{'=' * 110}")
    print(f"  FULL-PERIOD PARAMETER SWEEP ({total_combos} configs)")
    print(f"{'=' * 110}")

    results = []
    t_sw = time.time()
    for ci, cfg in enumerate(configs):
        r = run_backtest(**cfg)
        if r is not None:
            results.append(r)
        if (ci + 1) % 10 == 0:
            print(f"  [{ci+1}/{total_combos}] {len(results)} with results ({time.time()-t_sw:.1f}s)", flush=True)

    results.sort(key=lambda x: -x['ann'])
    print(f"\n  Sweep complete: {len(results)}/{total_combos} configs ({time.time()-t_sw:.1f}s)", flush=True)

    # ================================================================
    # TOP 20 FULL-PERIOD RESULTS
    # ================================================================
    print(f"\n{'=' * 110}")
    print(f"  TOP 20 FULL-PERIOD RESULTS")
    print(f"{'=' * 110}")
    hdr = f"  {'#':>2s} | {'Config':40s} | {'Ann':>8s} | {'WR':>5s} | {'N':>5s} | {'DD':>6s} | {'PF':>5s} | {'Sharpe':>7s} | {'Cash':>12s}"
    print(hdr)
    print(f"  {'-' * 105}")
    for i, r in enumerate(results[:20]):
        print(f"  {i+1:2d} | {r['name']:40s} | {r['ann']:+7.1f}% | {r['wr']:4.1f}% | {r['n']:5d} | {r['dd']:5.1f}% | {r['pf']:4.2f} | {r['sharpe']:6.2f} | {r['cash']:11.0f}")

    # ================================================================
    # SIZING MODE COMPARISON
    # ================================================================
    print(f"\n{'=' * 110}")
    print(f"  SIZING MODE COMPARISON (best per mode)")
    print(f"{'=' * 110}")

    mode_results = {}
    for r in results:
        mode = r['name'].split('_')[0]
        if mode not in mode_results or r['ann'] > mode_results[mode]['ann']:
            mode_results[mode] = r

    mode_order = ['FIXED', 'SIGSTR', 'TOP3', 'KELLY']
    mode_label = {
        'FIXED': 'Fixed (baseline)',
        'SIGSTR': 'Signal Strength',
        'TOP3': 'Top-3 Equal',
        'KELLY': 'Kelly 60-day',
    }
    for mode in mode_order:
        if mode in mode_results:
            r = mode_results[mode]
            label = mode_label.get(mode, mode)
            print(f"  {label:25s}: Ann={r['ann']:+8.1f}%  WR={r['wr']:5.1f}%  N={r['n']:5d}  DD={r['dd']:5.1f}%  PF={r['pf']:4.2f}  Sharpe={r['sharpe']:5.2f}  ({r['name']})")

    # Average ann per mode
    print(f"\n  Average Annual Return per Mode:")
    mode_avgs = {}
    for r in results:
        mode = r['name'].split('_')[0]
        mode_avgs.setdefault(mode, []).append(r['ann'])
    for mode in mode_order:
        if mode in mode_avgs:
            avg = np.mean(mode_avgs[mode])
            print(f"    {mode_label.get(mode, mode):25s}: avg={avg:+8.1f}%  ({len(mode_avgs[mode])} configs)")

    # ================================================================
    # WALK-FORWARD TOP 5
    # ================================================================
    top5 = results[:5]
    print(f"\n{'=' * 110}")
    print(f"  WALK-FORWARD VALIDATION (Top 5, test years 2020-2025)")
    print(f"{'=' * 110}")

    wf_all = []
    for rank, cfg_result in enumerate(top5):
        cn = cfg_result['name']
        matching = [c for c in configs if c['config_name'] == cn]
        if not matching:
            continue
        bc = matching[0]
        print(f"\n  [{rank+1}] {cn}  (full-period Ann={cfg_result['ann']:+.1f}%)")
        for ty in [2020, 2021, 2022, 2023, 2024, 2025]:
            if ty not in year_start_di:
                continue
            wc = dict(bc)
            wc['start_year'] = ty
            wc['end_year'] = ty
            wc['config_name'] = f"WF_{ty}_{cn}"
            r = run_backtest(**wc)
            if r is not None:
                wf_all.append((cn, ty, r))
                print(f"    {ty}: Ann={r['ann']:+7.1f}%  WR={r['wr']:5.1f}%  N={r['n']:4d}  DD={r['dd']:5.1f}%  PF={r['pf']:4.2f}")
            else:
                print(f"    {ty}: insufficient trades")

    # WF summary table
    if wf_all:
        print(f"\n{'=' * 110}")
        print(f"  WALK-FORWARD SUMMARY TABLE")
        print(f"{'=' * 110}")
        wf_by_config = {}
        for cn, ty, r in wf_all:
            wf_by_config.setdefault(cn, []).append((ty, r))

        wf_summary = []
        for cn, wr_list in wf_by_config.items():
            anns = [r['ann'] for _, r in wr_list]
            aby = {ty: r['ann'] for ty, r in wr_list}
            wf_summary.append({
                'name': cn, 'avg_ann': np.mean(anns),
                'anns_by_year': aby,
                'avg_wr': np.mean([r['wr'] for _, r in wr_list]),
                'avg_dd': np.mean([r['dd'] for _, r in wr_list]),
                'n_positive': sum(1 for a in anns if a > 0),
                'n_windows': len(wr_list),
            })
        wf_summary.sort(key=lambda x: -x['avg_ann'])

        hdr_cols = "Config | WF2020 | WF2021 | WF2022 | WF2023 | WF2024 | WF2025 | Avg"
        print(f"  {hdr_cols}")
        print(f"  {'-' * 105}")
        for i, w in enumerate(wf_summary):
            a20 = f"{w['anns_by_year'].get(2020, float('nan')):+7.1f}%" if 2020 in w['anns_by_year'] else "   N/A "
            a21 = f"{w['anns_by_year'].get(2021, float('nan')):+7.1f}%" if 2021 in w['anns_by_year'] else "   N/A "
            a22 = f"{w['anns_by_year'].get(2022, float('nan')):+7.1f}%" if 2022 in w['anns_by_year'] else "   N/A "
            a23 = f"{w['anns_by_year'].get(2023, float('nan')):+7.1f}%" if 2023 in w['anns_by_year'] else "   N/A "
            a24 = f"{w['anns_by_year'].get(2024, float('nan')):+7.1f}%" if 2024 in w['anns_by_year'] else "   N/A "
            a25 = f"{w['anns_by_year'].get(2025, float('nan')):+7.1f}%" if 2025 in w['anns_by_year'] else "   N/A "
            print(f"  {w['name']:40s} | {a20} | {a21} | {a22} | {a23} | {a24} | {a25} | {w['avg_ann']:+7.1f}%")

    # ================================================================
    # YEARLY BREAKDOWN TOP 5
    # ================================================================
    if len(results) >= 2:
        print(f"\n{'=' * 110}")
        print(f"  YEARLY BREAKDOWN FOR TOP 5")
        print(f"{'=' * 110}")
        for idx, r in enumerate(results[:5]):
            print(f"\n  #{idx+1}: {r['name']}")
            print(f"    Ann={r['ann']:+.1f}%  WR={r['wr']:.1f}%  DD={r['dd']:.1f}%  Sharpe={r['sharpe']:.2f}  N={r['n']}")
            print(f"    {'Year':>6s} | {'N':>5s} | {'WR':>5s} | {'PnL%':>8s} | {'PnL Abs':>12s}")
            print(f"    {'-' * 50}")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / max(ys['n'], 1) * 100
                print(f"    {y:6d} | {ys['n']:5d} | {wr_y:4.1f}% | {ys['pnl']:+7.1f}% | {ys['pnl_abs_sum']:+11.0f}")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print(f"\n{'=' * 110}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 110}")

    if results:
        b = results[0]
        print(f"\n  Best full-period: {b['name']}")
        print(f"    Ann={b['ann']:+.1f}%  WR={b['wr']:.1f}%  N={b['n']}  DD={b['dd']:.1f}%  PF={b['pf']:.2f}  Sharpe={b['sharpe']:.2f}")

    # Compare best of each mode
    fixed_best = mode_results.get('FIXED')
    sig_best = mode_results.get('SIGSTR')
    top3_best = mode_results.get('TOP3')
    kelly_best = mode_results.get('KELLY')

    if fixed_best:
        print(f"\n  Baseline (Fixed):  {fixed_best['ann']:+.1f}%  ({fixed_best['name']})")
    if sig_best:
        diff = sig_best['ann'] - fixed_best['ann'] if fixed_best else 0
        print(f"  Signal Strength:   {sig_best['ann']:+.1f}%  ({sig_best['name']})  delta={diff:+.1f}%")
    if top3_best:
        diff = top3_best['ann'] - fixed_best['ann'] if fixed_best else 0
        print(f"  Top-3 Equal:       {top3_best['ann']:+.1f}%  ({top3_best['name']})  delta={diff:+.1f}%")
    if kelly_best:
        diff = kelly_best['ann'] - fixed_best['ann'] if fixed_best else 0
        print(f"  Kelly 60-day:      {kelly_best['ann']:+.1f}%  ({kelly_best['name']})  delta={diff:+.1f}%")

    if wf_all:
        wf_by_config2 = {}
        for cn, ty, r in wf_all:
            wf_by_config2.setdefault(cn, []).append(r['ann'])
        best_wf_name = max(wf_by_config2.items(), key=lambda x: np.mean(x[1]))[0]
        best_wf_avg = np.mean(wf_by_config2[best_wf_name])
        print(f"\n  Best WF avg: {best_wf_name}  Avg Ann={best_wf_avg:+.1f}%")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 110)


if __name__ == '__main__':
    main()
