"""
Alpha Futures V100 -- Term Structure Carry Signal
==================================================
Use term structure (contango/backwardation) as a trading signal.

Term structure is one of the most proven signals in commodity futures:
- Backwardation (near > far) indicates tight supply -> bullish
- Contango (near < far) indicates ample supply -> bearish/neutral

Signals tested (all with next-open execution):
A) Carry Trade: rank by basis, buy most backwardated, rebalance R days
B) Basis Momentum: buy when basis is increasing (more backwardated)
C) Extreme Backwardation: basis in bottom 10th pct of 1yr -> buy
D) Structure Reversal: contango->backwardation switch -> buy
E) Basis + Momentum Combo: backwardated AND price momentum > 0
F) Curve Shape: steep front backwardation from full curve data

Walk-forward validation across 2021-2025 (term structure data starts 2021).
"""
import sys, os, time, warnings, json
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

WF_YEARS = [2021, 2022, 2023, 2024, 2025]

TS_DIR = '/Users/chengming/home/futures_platform/data/futures_term_structure/'


def load_term_structure(NS, ND, dates, syms):
    """Load term structure data from JSON files into arrays.

    Returns:
        basis: [NS, ND] array - (near_price - far_price) / near_price
                  positive = backwardation, negative = contango
        structure: [NS, ND] array - 1=backwardation, -1=contango, nan=missing
        curve_slope: [NS, ND] array - slope of front 3 vs back 3 contracts
        has_ts: [NS] bool - whether this symbol has any term structure data
    """
    print("[TS] Loading term structure data...", flush=True)
    t0 = time.time()

    basis = np.full((NS, ND), np.nan)
    structure = np.full((NS, ND), np.nan)
    curve_slope = np.full((NS, ND), np.nan)

    # Build date string -> di mapping
    date_str_map = {}
    for di in range(ND):
        date_str_map[dates[di].strftime('%Y%m%d')] = di

    sym_to_si = {syms[si]: si for si in range(NS)}

    # List all files and group by symbol
    loaded_count = 0
    missing_count = 0

    for fname in sorted(os.listdir(TS_DIR)):
        if not fname.endswith('.json'):
            continue
        parts = fname.rsplit('_', 1)
        if len(parts) != 2:
            continue
        sym, date_part = parts
        date_part = date_part.replace('.json', '')

        si = sym_to_si.get(sym)
        if si is None:
            continue

        di = date_str_map.get(date_part)
        if di is None:
            continue

        try:
            with open(os.path.join(TS_DIR, fname), 'r') as f:
                data = json.load(f)
        except:
            missing_count += 1
            continue

        near_p = data.get('near_price')
        far_p = data.get('far_price')
        if near_p and far_p and near_p > 0:
            basis[si, di] = (near_p - far_p) / near_p

        struct = data.get('structure', '')
        if struct == 'backwardation':
            structure[si, di] = 1.0
        elif struct == 'contango':
            structure[si, di] = -1.0

        # Curve slope: front 3 vs back 3
        curve = data.get('curve', [])
        if len(curve) >= 4:
            # Sort by year,month to ensure order
            sorted_curve = sorted(curve, key=lambda x: (x.get('year', 0), x.get('month', 0)))
            front_prices = [c['price'] for c in sorted_curve[:3] if c.get('price')]
            back_prices = [c['price'] for c in sorted_curve[-3:] if c.get('price')]
            if len(front_prices) >= 2 and len(back_prices) >= 2:
                front_avg = np.mean(front_prices)
                back_avg = np.mean(back_prices)
                if front_avg > 0:
                    curve_slope[si, di] = (front_avg - back_avg) / front_avg

        loaded_count += 1

    has_ts = np.array([not np.all(np.isnan(basis[si, :])) for si in range(NS)])

    # Count coverage
    ts_sis = np.where(has_ts)[0]
    print(f"  Loaded {loaded_count} TS files ({missing_count} errors)")
    print(f"  {len(ts_sis)} commodities have TS data")
    print(f"  Date range with TS: ", end='')

    # Find first and last di with any TS data
    first_di = ND
    last_di = 0
    for si in ts_sis:
        valid = np.where(~np.isnan(basis[si, :]))[0]
        if len(valid) > 0:
            first_di = min(first_di, valid[0])
            last_di = max(last_di, valid[-1])
    if first_di < ND:
        print(f"{dates[first_di].strftime('%Y-%m-%d')} to {dates[last_di].strftime('%Y-%m-%d')}")
    else:
        print("none")

    # Coverage by year
    for yr in range(2021, 2027):
        yr_dis = [di for di in range(ND) if dates[di].year == yr]
        if not yr_dis:
            continue
        n_valid = sum(1 for si in ts_sis for di in yr_dis if not np.isnan(basis[si, di]))
        n_total = len(ts_sis) * len(yr_dis)
        pct = n_valid / n_total * 100 if n_total > 0 else 0
        print(f"  {yr}: {pct:.1f}% coverage ({n_valid}/{n_total})")

    print(f"  TS loading done ({time.time()-t0:.1f}s)", flush=True)
    return basis, structure, curve_slope, has_ts


def main():
    t_start = time.time()
    print("=" * 130)
    print("Alpha Futures V100 -- Term Structure Carry Signal")
    print("Classic commodity alpha: backwardation = tight supply = bullish")
    print("=" * 130)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    sym_to_si = {syms[si]: si for si in range(NS)}

    # Year boundaries
    year_start_di = {}
    year_end_di = {}
    for di in range(ND):
        y = dates[di].year
        if y not in year_start_di:
            year_start_di[y] = di
        year_end_di[y] = di

    print(f"  {NS} commodities, {ND} days, years: {sorted(year_start_di.keys())}")

    # Load term structure data
    basis, structure, curve_slope, has_ts = load_term_structure(NS, ND, dates, syms)
    ts_sis = np.where(has_ts)[0]
    print(f"  {len(ts_sis)} commodities with TS data")

    # ================================================================
    # PRECOMPUTE DERIVED SIGNALS
    # ================================================================
    print("\n[Signals] Computing derived signals...", flush=True)
    t0 = time.time()

    # B) Basis momentum: change in basis over past N days
    basis_mom20 = np.full((NS, ND), np.nan)
    for si in ts_sis:
        for di in range(20, ND):
            cur = basis[si, di]
            prev = basis[si, di - 20]
            if not np.isnan(cur) and not np.isnan(prev):
                basis_mom20[si, di] = cur - prev

    # Price momentum at various lookbacks
    mom20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            c_now = C[si, di]
            c_prev = C[si, di - 20]
            if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                mom20[si, di] = (c_now - c_prev) / c_prev

    # C) Basis percentile rank over past 252 days (1 year)
    basis_pct = np.full((NS, ND), np.nan)
    for si in ts_sis:
        for di in range(252, ND):
            window = basis[si, di-252:di]
            valid = window[~np.isnan(window)]
            if len(valid) < 50:
                continue
            cur = basis[si, di]
            if not np.isnan(cur):
                basis_pct[si, di] = np.mean(valid <= cur)  # 0=lowest, 1=highest

    # D) Structure reversal: detect contango->backwardation switches
    struct_prev5 = np.full((NS, ND), np.nan)  # structure 5 days ago
    for si in ts_sis:
        for di in range(5, ND):
            # Majority structure over 5-day window ending 5 days ago
            window = structure[si, di-5:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 3:
                struct_prev5[si, di] = np.sign(np.mean(valid))

    # switch_to_back = was contango, now backwardation
    switch_to_back = np.full((NS, ND), False)
    switch_to_cont = np.full((NS, ND), False)
    for si in ts_sis:
        for di in range(5, ND):
            cur_s = structure[si, di]
            prev_s = struct_prev5[si, di]
            if np.isnan(cur_s) or np.isnan(prev_s):
                continue
            if prev_s < 0 and cur_s > 0:
                switch_to_back[si, di] = True
            if prev_s > 0 and cur_s < 0:
                switch_to_cont[si, di] = True

    print(f"  Derived signals done ({time.time()-t0:.1f}s)", flush=True)

    # ================================================================
    # BACKTEST ENGINE - Rebalancing style
    # ================================================================
    def run_backtest_rebalance(
        signal_type='carry',
        K=5,
        rebalance_days=10,
        comm=COMM,
        start_year=None,
        end_year=None,
        config_name="",
    ):
        """
        Rebalancing backtest: hold top-K commodities, rebalance every R days.
        Long-only, next-open execution.
        """
        # Determine start/end di
        start_di = None
        end_di = ND
        if start_year is not None:
            if start_year in year_start_di:
                start_di = year_start_di[start_year]
            else:
                return None
        else:
            # Find first day with TS data
            for di in range(ND):
                if np.any(~np.isnan(basis[ts_sis, di])):
                    start_di = di
                    break
            if start_di is None:
                return None

        if end_year is not None:
            if end_year in year_end_di:
                end_di = year_end_di[end_year] + 1
            else:
                return None

        # Warmup: need at least 252 days for percentile signals
        if start_di is None or start_di < 252:
            start_di = 252

        cash = float(CASH0)
        positions = {}  # si -> {entry, lots, dir, entry_di}
        trades = []
        last_rebal_di = -999

        for di in range(start_di, end_di):
            # Check if we can enter (need next day's open)
            if di + 1 >= ND:
                break

            year = dates[di].year
            should_rebalance = (di - last_rebal_di >= rebalance_days)

            if not should_rebalance and positions:
                continue

            # --- Close existing positions at today's close ---
            for si in list(positions.keys()):
                pos = positions[si]
                cn = C[si, di]
                if np.isnan(cn) or cn <= 0:
                    cn = pos['entry']
                mult = MULT.get(syms[si], DEF_MULT)
                pnl = (cn - pos['entry']) * mult * pos['lots'] * pos['dir']
                invested = pos['entry'] * mult * pos['lots']
                pnl_pct = pnl / invested * 100 if invested > 0 else 0
                cost = cn * mult * pos['lots'] * comm
                cash += cn * mult * pos['lots'] - cost
                hold_days = di - pos['entry_di']
                trades.append({
                    'pnl_abs': pnl, 'pnl_pct': pnl_pct,
                    'days': hold_days, 'di': di, 'year': year,
                    'sym': syms[si], 'dir': pos['dir'], 'reason': 'rebal',
                })
            positions = {}

            # --- Score candidates ---
            candidates = []
            for si in ts_sis:
                sym = syms[si]

                # Need valid data
                if np.isnan(C[si, di]) or C[si, di] <= 0:
                    continue
                if np.isnan(O[si, di+1]) or O[si, di+1] <= 0:
                    continue
                if np.isnan(basis[si, di]):
                    continue

                if signal_type == 'carry':
                    # Classic carry: buy most backwardated (highest basis)
                    score = basis[si, di]

                elif signal_type == 'basis_mom':
                    # Buy when basis is increasing
                    bm = basis_mom20[si, di]
                    if np.isnan(bm) or bm <= 0:
                        continue
                    score = bm

                elif signal_type == 'extreme_back':
                    # Extreme backwardation: basis percentile < 10%
                    bp = basis_pct[si, di]
                    if np.isnan(bp) or bp > 0.1:
                        continue
                    score = basis[si, di]

                elif signal_type == 'struct_rev':
                    # Structure reversal: contango -> backwardation
                    if not switch_to_back[si, di]:
                        continue
                    score = basis[si, di]

                elif signal_type == 'basis_momentum_combo':
                    # Backwardated AND positive price momentum
                    b = basis[si, di]
                    m = mom20[si, di]
                    if np.isnan(b) or b <= 0:
                        continue
                    if np.isnan(m) or m <= 0:
                        continue
                    score = b + m  # Combined score

                elif signal_type == 'curve_shape':
                    # Steep front backwardation
                    cs = curve_slope[si, di]
                    if np.isnan(cs) or cs <= 0:
                        continue
                    score = cs

                else:
                    continue

                if np.isnan(score):
                    continue
                candidates.append((si, score))

            if not candidates:
                last_rebal_di = di
                continue

            # Sort by score descending, take top K
            candidates.sort(key=lambda x: -x[1])
            top_k = candidates[:K]

            # Allocate capital equally
            if len(top_k) > 0 and cash > 0:
                alloc = cash / len(top_k)
                for si, score in top_k:
                    entry = O[si, di+1]
                    if np.isnan(entry) or entry <= 0:
                        continue
                    mult = MULT.get(syms[si], DEF_MULT)
                    lots = max(1, int(alloc / (entry * mult)))
                    cost = entry * mult * lots
                    if cost > cash:
                        lots = max(1, int(cash / (entry * mult)))
                        cost = entry * mult * lots
                    if lots <= 0 or cost > cash:
                        continue
                    cash -= cost + cost * comm
                    positions[si] = {
                        'entry': entry, 'lots': lots, 'dir': 1,
                        'entry_di': di+1, 'score': score,
                    }

            last_rebal_di = di

        # --- Close remaining positions at end ---
        for si in list(positions.keys()):
            pos = positions[si]
            cn = C[si, end_di - 1] if end_di <= ND else C[si, -1]
            if np.isnan(cn) or cn <= 0:
                cn = pos['entry']
            mult = MULT.get(syms[si], DEF_MULT)
            pnl = (cn - pos['entry']) * mult * pos['lots'] * pos['dir']
            invested = pos['entry'] * mult * pos['lots']
            pnl_pct = pnl / invested * 100 if invested > 0 else 0
            cost = cn * mult * pos['lots'] * comm
            cash += cn * mult * pos['lots'] - cost
            hold_days = (end_di - 1) - pos['entry_di']
            trades.append({
                'pnl_abs': pnl, 'pnl_pct': pnl_pct,
                'days': max(1, hold_days), 'di': end_di - 1,
                'year': dates[min(end_di-1, ND-1)].year,
                'sym': syms[si], 'dir': pos['dir'], 'reason': 'end',
            })

        # --- Compute stats ---
        if not trades:
            return None

        df_t = trades
        pnls = [t['pnl_abs'] for t in df_t]
        pnls_pct = [t['pnl_pct'] for t in df_t]
        total_pnl = sum(pnls)
        n_trades = len(df_t)
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / n_trades * 100 if n_trades > 0 else 0

        # Equity curve for drawdown
        eq = [float(CASH0)]
        for t in df_t:
            eq.append(eq[-1] + t['pnl_abs'])
        eq = np.array(eq)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak * 100
        mdd = dd.min()

        # Annual returns
        yrly = {}
        for t in df_t:
            yrly.setdefault(t['year'], []).append(t['pnl_abs'])
        ann_ret = {}
        for yr, pl in sorted(yrly.items()):
            ann_ret[yr] = sum(pl)

        # Determine years covered
        if start_year and end_year:
            n_years = end_year - start_year + 1
        elif start_year:
            n_years = max(1, dates[end_di-1].year - start_year + 1)
        else:
            n_years = max(1, dates[end_di-1].year - dates[start_di].year + 1)

        total_ret = (cash - CASH0) / CASH0 * 100
        annual_ret = total_ret / n_years if n_years > 0 else 0

        avg_hold = np.mean([t['days'] for t in df_t])

        return {
            'name': config_name,
            'cash': cash,
            'total_ret': total_ret,
            'annual_ret': annual_ret,
            'n_trades': n_trades,
            'wr': wr,
            'mdd': mdd,
            'avg_hold': avg_hold,
            'ann_ret_detail': ann_ret,
            'trades': df_t,
        }

    # ================================================================
    # RUN ALL CONFIGURATIONS - FULL PERIOD
    # ================================================================
    print("\n" + "=" * 130)
    print("PHASE 1: Full period backtest (2021-2025)")
    print("=" * 130)

    configs = []

    # A) Carry Trade: K x rebalance_days
    for K in [3, 5, 10]:
        for R in [10, 20]:
            configs.append(('carry', K, R, f"A_Carry_K{K}_R{R}"))

    # B) Basis Momentum
    for K in [3, 5, 10]:
        for R in [10, 20]:
            configs.append(('basis_mom', K, R, f"B_BasisMom_K{K}_R{R}"))

    # C) Extreme Backwardation
    for K in [3, 5]:
        for R in [10, 20]:
            configs.append(('extreme_back', K, R, f"C_ExtremeBack_K{K}_R{R}"))

    # D) Structure Reversal
    for K in [3, 5]:
        for R in [10, 20]:
            configs.append(('struct_rev', K, R, f"D_StructRev_K{K}_R{R}"))

    # E) Basis + Momentum Combo
    for K in [3, 5, 10]:
        for R in [10, 20]:
            configs.append(('basis_momentum_combo', K, R, f"E_Combo_K{K}_R{R}"))

    # F) Curve Shape
    for K in [3, 5, 10]:
        for R in [10, 20]:
            configs.append(('curve_shape', K, R, f"F_CurveShape_K{K}_R{R}"))

    full_results = []
    for sig_type, K, R, name in configs:
        res = run_backtest_rebalance(
            signal_type=sig_type, K=K, rebalance_days=R,
            config_name=name,
        )
        if res:
            full_results.append(res)
            print(f"  {name:30s}  Ann={res['annual_ret']:+8.1f}%  "
                  f"Tot={res['total_ret']:+8.1f}%  WR={res['wr']:5.1f}%  "
                  f"MDD={res['mdd']:+6.1f}%  Trades={res['n_trades']:4d}  "
                  f"AvgHold={res['avg_hold']:.1f}d")

    # ================================================================
    # RANK BY ANNUAL RETURN
    # ================================================================
    print("\n" + "-" * 130)
    print("FULL PERIOD RANKING (by annual return):")
    print("-" * 130)
    full_results.sort(key=lambda x: -x['annual_ret'])
    for rank, r in enumerate(full_results, 1):
        ann_detail = '  '.join(f"{yr}:{v/CASH0*100:+.1f}%" for yr, v in sorted(r['ann_ret_detail'].items()))
        print(f"  #{rank:2d} {r['name']:30s}  Ann={r['annual_ret']:+8.1f}%  "
              f"Tot={r['total_ret']:+8.1f}%  WR={r['wr']:5.1f}%  "
              f"MDD={r['mdd']:+6.1f}%  Trades={r['n_trades']:4d}")
        print(f"      Annual: {ann_detail}")

    # ================================================================
    # PHASE 2: WALK-FORWARD VALIDATION
    # ================================================================
    print("\n" + "=" * 130)
    print("PHASE 2: Walk-Forward Validation (2021-2025)")
    print("=" * 130)

    # Take top 15 configs for walk-forward
    top15 = full_results[:15]
    print(f"\nWalk-forward for top {len(top15)} configs:")

    wf_all = []
    for r in top15:
        name = r['name']
        # Parse config from name
        parts = name.split('_')
        sig_part = parts[0]
        K_part = parts[-2]  # K3
        R_part = parts[-1]  # R10

        sig_type_map = {
            'A': 'carry', 'B': 'basis_mom', 'C': 'extreme_back',
            'D': 'struct_rev', 'E': 'basis_momentum_combo', 'F': 'curve_shape'
        }
        sig_type = sig_type_map.get(sig_part, 'carry')
        K = int(K_part[1:])
        R_val = int(R_part[1:])

        wf_results = {}
        for yr in WF_YEARS:
            res = run_backtest_rebalance(
                signal_type=sig_type, K=K, rebalance_days=R_val,
                start_year=yr, end_year=yr,
                config_name=f"{name}_{yr}",
            )
            if res:
                wf_results[yr] = res

        n_positive = sum(1 for yr, res in wf_results.items() if res['annual_ret'] > 0)
        avg_wf = np.mean([res['annual_ret'] for res in wf_results.values()]) if wf_results else 0
        min_wf = min((res['annual_ret'] for res in wf_results.values()), default=0)
        max_wf = max((res['annual_ret'] for res in wf_results.values()), default=0)

        wf_row = {
            'name': name,
            'sig_type': sig_type,
            'K': K,
            'R': R_val,
            'full_ann': r['annual_ret'],
            'n_wf': len(wf_results),
            'n_positive': n_positive,
            'avg_wf': avg_wf,
            'min_wf': min_wf,
            'max_wf': max_wf,
            'wf_results': wf_results,
        }
        wf_all.append(wf_row)

        wf_str = '  '.join(
            f"{yr}:{wf_results[yr]['annual_ret']:+.1f}%" if yr in wf_results else f"{yr}:N/A"
            for yr in WF_YEARS
        )
        print(f"  {name:30s}  WF+:{n_positive}/{len(wf_results)}  "
              f"AvgWF={avg_wf:+8.1f}%  Range=[{min_wf:+.1f}, {max_wf:+.1f}%]")
        print(f"      {wf_str}")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print("\n" + "=" * 130)
    print("WALK-FORWARD RANKING (by avg WF return):")
    print("=" * 130)
    wf_all.sort(key=lambda x: -x['avg_wf'])
    for rank, w in enumerate(wf_all, 1):
        wf_str = '  '.join(
            f"{yr}:{w['wf_results'][yr]['annual_ret']:+.1f}%" if yr in w['wf_results'] else f"{yr}:N/A"
            for yr in WF_YEARS
        )
        print(f"  #{rank:2d} {w['name']:30s}  "
              f"Full={w['full_ann']:+8.1f}%  WF+:{w['n_positive']}/{w['n_wf']}  "
              f"AvgWF={w['avg_wf']:+8.1f}%  Range=[{w['min_wf']:+.1f}, {w['max_wf']:+.1f}%]")
        print(f"      {wf_str}")

    # ================================================================
    # SIGNAL QUALITY ANALYSIS
    # ================================================================
    print("\n" + "=" * 130)
    print("SIGNAL QUALITY ANALYSIS")
    print("=" * 130)

    # For best WF config, analyze trade-level stats
    if wf_all:
        best_wf = wf_all[0]
        best_full = next((r for r in full_results if r['name'] == best_wf['name']), None)
        if best_full:
            trades = best_full['trades']
            pnls = [t['pnl_pct'] for t in trades]
            if pnls:
                print(f"\n  Best WF config: {best_wf['name']}")
                print(f"    Mean trade PnL: {np.mean(pnls):+.3f}%")
                print(f"    Median trade PnL: {np.median(pnls):+.3f}%")
                print(f"    Std trade PnL: {np.std(pnls):.3f}%")
                print(f"    Sharpe per trade: {np.mean(pnls)/np.std(pnls):.3f}" if np.std(pnls) > 0 else "")
                print(f"    Skewness: {float(np.mean(((np.array(pnls) - np.mean(pnls)) / np.std(pnls))**3)):.3f}" if np.std(pnls) > 0 else "")

                # By signal type breakdown
                by_sym = {}
                for t in trades:
                    by_sym.setdefault(t['sym'], []).append(t['pnl_pct'])
                sym_stats = [(sym, len(pls), np.mean(pls), np.sum(pls))
                             for sym, pls in by_sym.items()]
                sym_stats.sort(key=lambda x: -x[3])
                print(f"\n    Top 10 symbols by total PnL:")
                for sym, n, avg_p, tot_p in sym_stats[:10]:
                    print(f"      {sym:8s}  trades={n:3d}  avg={avg_p:+.3f}%  total={tot_p:+.1f}%")

    # ================================================================
    # CROSS-SIGNAL COMPARISON
    # ================================================================
    print("\n" + "=" * 130)
    print("CROSS-SIGNAL COMPARISON (best of each type, full period):")
    print("=" * 130)

    signal_types = {'A': 'Carry', 'B': 'BasisMom', 'C': 'ExtremeBack',
                    'D': 'StructRev', 'E': 'Combo', 'F': 'CurveShape'}
    for sig_prefix, sig_name in signal_types.items():
        matching = [r for r in full_results if r['name'].startswith(sig_prefix + '_')]
        if matching:
            best = max(matching, key=lambda x: x['annual_ret'])
            wf_match = next((w for w in wf_all if w['name'] == best['name']), None)
            wf_info = f"WF+:{wf_match['n_positive']}/{wf_match['n_wf']} AvgWF={wf_match['avg_wf']:+.1f}%" if wf_match else "no WF"
            print(f"  {sig_name:15s} {best['name']:25s}  "
                  f"Ann={best['annual_ret']:+8.1f}%  WR={best['wr']:5.1f}%  "
                  f"MDD={best['mdd']:+6.1f}%  {wf_info}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
