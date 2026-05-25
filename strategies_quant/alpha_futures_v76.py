"""
Alpha Futures V76 -- Tail Risk + OI Confirmation on V74 Base
=============================================================
V74 (extended groups, LB=1, 1-day hold) gives +2185% annual but no risk management.

Enhancements tested:
  A: Tail Risk Filter (CAViaR-X) -- skip all trades when core commodities breach VaR
  B: OI Confirmation -- only trade when OI is rising (capital flowing in)
  C: Vol-Scaled Sizing -- reduce position when vol is elevated
  D: Max Drawdown Circuit Breaker -- reduce/halt trading at large drawdowns

Tests: A, B, C, D, AB, ABC, ABCD, none (V74 baseline)
Walk-forward: 6 windows (2020-2025)
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

# Extended group map (same as V74)
GROUP_MAP = {}
for s in ['rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi']:
    GROUP_MAP[s] = 'ferrous'
for s in ['cufi', 'alfi', 'znfi', 'nifi', 'pbfi', 'snfi', 'ssfi', 'sffi']:
    GROUP_MAP[s] = 'nonferrous'
for s in ['aufi', 'agfi']:
    GROUP_MAP[s] = 'precious'
for s in ['afi', 'mfi', 'yfi', 'pfi', 'cfi', 'csfi', 'rrfi', 'lrfi']:
    GROUP_MAP[s] = 'oils'
for s in ['scfi', 'mafi', 'bfi', 'fufi', 'pgfi', 'ebfi', 'fbfi']:
    GROUP_MAP[s] = 'energy'
for s in ['ppfi', 'vfi', 'egfi', 'srfi', 'tafi', 'fgfi', 'lfi']:
    GROUP_MAP[s] = 'chemical'
for s in ['whfi', 'apfi', 'cjfi', 'oifi', 'rmfi', 'srfi', 'cffi']:
    GROUP_MAP[s] = 'soft'
for s in ['jdfi', 'lhfi', 'pkfi']:
    GROUP_MAP[s] = 'livestock'

# Core commodities for tail risk (systemically important)
CORE_COMMODITIES = ['ifi', 'scfi', 'cufi']

# Enhancement flags
ENHANCEMENTS = {
    'none':        {'tail_risk': False, 'oi_confirm': False, 'vol_scale': False, 'circuit_brk': False},
    'tail_risk':   {'tail_risk': True,  'oi_confirm': False, 'vol_scale': False, 'circuit_brk': False},
    'oi_confirm':  {'tail_risk': False, 'oi_confirm': True,  'vol_scale': False, 'circuit_brk': False},
    'vol_scaled':  {'tail_risk': False, 'oi_confirm': False, 'vol_scale': True,  'circuit_brk': False},
    'circuit_brk': {'tail_risk': False, 'oi_confirm': False, 'vol_scale': False, 'circuit_brk': True},
    'tail+oi':     {'tail_risk': True,  'oi_confirm': True,  'vol_scale': False, 'circuit_brk': False},
    'tail+oi+vol': {'tail_risk': True,  'oi_confirm': True,  'vol_scale': True,  'circuit_brk': False},
    'all':         {'tail_risk': True,  'oi_confirm': True,  'vol_scale': True,  'circuit_brk': True},
}


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def main():
    print("=" * 98)
    print("V76 -- Tail Risk + OI Confirmation")
    print("=" * 98)

    # ================================================================
    # LOAD DATA
    # ================================================================
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # Build symbol -> index map
    sym2idx = {s: i for i, s in enumerate(syms)}

    # ================================================================
    # PRECOMPUTE: Returns, Momentum, Group Momentum, VaR, OI change, Vol
    # ================================================================
    print("\n[Signals] Computing...", flush=True)
    t0 = time.time()

    # Daily returns
    ret = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            cp = C[si, di - 1]
            cn = C[si, di]
            if not np.isnan(cn) and not np.isnan(cp) and cp > 0:
                ret[si, di] = (cn - cp) / cp

    # 1-day momentum (LB=1, same as V74)
    mom1 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            cn = C[si, di]
            cp = C[si, di - 1]
            if not np.isnan(cn) and not np.isnan(cp) and cp > 0:
                mom1[si, di] = (cn - cp) / cp

    # Group momentum: for each commodity, avg momentum of group peers
    grp_mom1 = np.full((NS, ND), np.nan)
    gm_map = {}
    for si in range(NS):
        g = GROUP_MAP.get(syms[si])
        if g:
            gm_map.setdefault(g, []).append(si)

    for grp, members in gm_map.items():
        for di in range(1, ND):
            for sj in members:
                ms = [mom1[sk, di] for sk in members
                      if sk != sj and not np.isnan(mom1[sk, di])]
                if ms:
                    grp_mom1[sj, di] = np.mean(ms)

    # Divergence = group_mom - own_mom (positive = commodity lagging group)
    divergence = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            own = mom1[si, di]
            grp = grp_mom1[si, di]
            if not np.isnan(own) and not np.isnan(grp):
                divergence[si, di] = grp - own

    # --- Enhancement A: Rolling 20-day VaR (5th percentile) for core commodities ---
    VAR_WINDOW = 20
    core_si = [sym2idx[s] for s in CORE_COMMODITIES if s in sym2idx]
    # var_breach[di] = True if any core commodity breached VaR yesterday
    var_breach = np.zeros(ND, dtype=bool)
    core_var = {}
    for si in core_si:
        var_arr = np.full(ND, np.nan)
        for di in range(VAR_WINDOW, ND):
            window = ret[si, di - VAR_WINDOW:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 10:
                var_arr[di] = np.percentile(valid, 5)
        core_var[si] = var_arr
        for di in range(VAR_WINDOW + 1, ND):
            if not np.isnan(var_arr[di - 1]) and not np.isnan(ret[si, di - 1]):
                if ret[si, di - 1] < var_arr[di - 1]:
                    var_breach[di] = True

    breach_count = int(np.sum(var_breach))
    print(f"  VaR breach days: {breach_count}/{ND} ({100*breach_count/ND:.1f}%)")

    # --- Enhancement B: OI 5-day change rate ---
    OI_WINDOW = 5
    oi_change = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(OI_WINDOW, ND):
            oi_now = OI[si, di]
            oi_prev = OI[si, di - OI_WINDOW]
            if not np.isnan(oi_now) and not np.isnan(oi_prev) and oi_prev > 0:
                oi_change[si, di] = (oi_now - oi_prev) / oi_prev

    # --- Enhancement C: Vol scaling ---
    VOL_SHORT = 20
    VOL_LONG = 60
    vol_ratio = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(VOL_LONG, ND):
            short_w = ret[si, di - VOL_SHORT:di]
            long_w = ret[si, di - VOL_LONG:di]
            sv = short_w[~np.isnan(short_w)]
            lv = long_w[~np.isnan(long_w)]
            if len(sv) >= 10 and len(lv) >= 20:
                vol_s = np.std(sv)
                vol_l = np.median(np.abs(lv))
                if vol_l > 0:
                    vol_ratio[si, di] = vol_s / vol_l

    # Tradeable symbols (those in GROUP_MAP)
    trade_sis = [si for si in range(NS) if GROUP_MAP.get(syms[si])]

    print(f"  Signals computed ({time.time()-t0:.1f}s)")
    print(f"  Tradeable: {len(trade_sis)} commodities in {len(gm_map)} groups")

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(enh_name, enh_flags, wf_test_year=None,
                     threshold=0.003, top_n=3):
        """
        Run backtest with given enhancement flags.
        Base signal: V74 extended groups, LB=1, 1-day hold.
        """
        use_tail_risk = enh_flags['tail_risk']
        use_oi_confirm = enh_flags['oi_confirm']
        use_vol_scale = enh_flags['vol_scale']
        use_circuit_brk = enh_flags['circuit_brk']

        # Date range
        wf_mode = wf_test_year is not None
        if wf_mode:
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
            test_end_di = ND
            end_di = ND

        cash = float(CASH0)
        positions = []
        trades = []
        equity_curve = []  # track daily equity for DD calc

        for di in range(MIN_TRAIN, end_di):
            # Reset cash for WF mode
            if wf_mode and di == test_start_di:
                cash = float(CASH0)
                positions = []
                trades = []
                equity_curve = []

            # --- Compute current equity for circuit breaker ---
            cur_equity = cash
            for pos in positions:
                cn = C[pos['si'], di]
                if np.isnan(cn) or cn <= 0:
                    cn = pos['entry']
                mult = MULT.get(pos['sym'], DEF_MULT)
                cur_equity += cn * mult * pos['lots']
            equity_curve.append(cur_equity)

            # --- Enhancement D: Circuit breaker ---
            size_mult = 1.0
            if use_circuit_brk and len(equity_curve) > 1:
                peak = max(equity_curve)
                if peak > 0:
                    dd_pct = (peak - cur_equity) / peak
                    if dd_pct > 0.25:
                        size_mult = 0.0  # go to cash
                    elif dd_pct > 0.15:
                        size_mult = 0.5  # half size

            # --- Close positions entered yesterday ---
            closed = []
            for pos in positions:
                if di - pos['entry_di'] >= 1:
                    cn = C[pos['si'], di]
                    if np.isnan(cn) or cn <= 0:
                        cn = pos['entry']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    mkt_val = cn * mult * pos['lots']
                    cash += mkt_val - mkt_val * COMM
                    pnl = (cn - pos['entry']) * mult * pos['lots'] * pos['dir']
                    invested = pos['entry'] * mult * pos['lots']
                    pnl_pct = pnl / invested * 100 if invested > 0 else 0
                    if wf_mode:
                        in_test = test_start_di <= pos['entry_di'] < test_end_di
                    else:
                        in_test = True
                    if in_test:
                        trades.append({
                            'pnl_pct': pnl_pct,
                            'di': pos['entry_di'],
                            'year': dates[di].year if di < ND else dates[-1].year,
                            'dir': pos['dir'],
                        })
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            # --- Enhancement A: Tail risk filter ---
            if use_tail_risk and var_breach[di]:
                continue  # skip all trades today

            # --- Enhancement D: If circuit breaker at 0, skip ---
            if use_circuit_brk and size_mult <= 0:
                continue

            # --- Score candidates ---
            candidates = []
            for si in trade_sis:
                sym = syms[si]
                if np.isnan(C[si, di]) or C[si, di] <= 0:
                    continue
                if any(p['si'] == si for p in positions):
                    continue

                div = divergence[si, di]
                if np.isnan(div) or div <= threshold:
                    continue

                # --- Enhancement B: OI confirmation ---
                if use_oi_confirm:
                    oc = oi_change[si, di]
                    if np.isnan(oc):
                        continue
                    if oc < -0.02:
                        continue  # OI dropping fast, skip
                    if oc <= 0:
                        continue  # OI not rising, skip long

                candidates.append((si, div))

            if not candidates:
                continue

            # Sort by divergence (highest first)
            candidates.sort(key=lambda x: -x[1])

            # Open positions
            n_slots = top_n - len(positions)
            for si, score in candidates[:n_slots]:
                c = C[si, di]
                if np.isnan(c) or c <= 0:
                    continue
                mult = MULT.get(syms[si], DEF_MULT)
                notional = c * mult

                # Base allocation
                alloc_lots = int(cash / (notional * (1 + COMM)))

                # --- Enhancement C: Vol-scaled sizing ---
                if use_vol_scale:
                    vr = vol_ratio[si, di]
                    if not np.isnan(vr) and vr > 0:
                        vol_adj = 1.0 / (1.0 + vr)
                        alloc_lots = int(alloc_lots * vol_adj)

                # --- Enhancement D: Circuit breaker sizing ---
                if use_circuit_brk and size_mult < 1.0:
                    alloc_lots = int(alloc_lots * size_mult)

                if alloc_lots <= 0:
                    continue
                cost_in = notional * alloc_lots * (1 + COMM)
                if cost_in > cash:
                    alloc_lots = int(cash * 0.95 / (notional * (1 + COMM)))
                    cost_in = notional * alloc_lots * (1 + COMM) if alloc_lots > 0 else 0
                if alloc_lots <= 0 or cost_in <= 0 or cost_in > cash:
                    continue

                cash -= cost_in
                positions.append({
                    'si': si, 'entry': c, 'entry_di': di,
                    'lots': alloc_lots, 'dir': 1, 'sym': syms[si],
                })

        # Close remaining positions
        for pos in positions:
            ae = ND - 1
            cn = C[pos['si'], ae]
            if np.isnan(cn) or cn <= 0:
                cn = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = cn * mult * pos['lots']
            cash += mkt_val - mkt_val * COMM
            if wf_mode:
                in_test = test_start_di <= pos['entry_di'] < test_end_di
            else:
                in_test = True
            if in_test:
                pnl = (cn - pos['entry']) * mult * pos['lots'] * pos['dir']
                invested = pos['entry'] * mult * pos['lots']
                pnl_pct = pnl / invested * 100 if invested > 0 else 0
                trades.append({
                    'pnl_pct': pnl_pct,
                    'di': pos['entry_di'],
                    'year': dates[ae].year,
                    'dir': pos['dir'],
                })

        # Compute max drawdown from equity curve
        # Reconstruct equity curve properly from trades
        if wf_mode:
            eq_start = test_start_di
            eq_end = test_end_di
        else:
            eq_start = MIN_TRAIN
            eq_end = ND

        # Build daily equity for DD calculation
        daily_equity = []
        running_cash = float(CASH0)
        running_positions = []
        for di in range(eq_start, eq_end):
            if wf_mode and di == test_start_di:
                running_cash = float(CASH0)
                running_positions = []

            # Close yesterday's positions
            closed_here = []
            for pos in running_positions:
                if di - pos['entry_di'] >= 1:
                    cn = C[pos['si'], di]
                    if np.isnan(cn) or cn <= 0:
                        cn = pos['entry']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    running_cash += cn * mult * pos['lots'] - cn * mult * pos['lots'] * COMM
                    closed_here.append(pos)
            for p in closed_here:
                running_positions.remove(p)

            # Compute equity
            eq = running_cash
            for pos in running_positions:
                cn = C[pos['si'], di]
                if np.isnan(cn) or cn <= 0:
                    cn = pos['entry']
                mult = MULT.get(pos['sym'], DEF_MULT)
                eq += cn * mult * pos['lots']
            daily_equity.append(eq)

            # Entry logic (same as above but simpler for equity tracking)
            if use_tail_risk and var_breach[di]:
                continue
            if use_circuit_brk and len(daily_equity) > 1:
                pk = max(daily_equity)
                if pk > 0:
                    dd_p = (pk - daily_equity[-1]) / pk
                    if dd_p > 0.25:
                        continue

            # Find candidates and enter
            cands = []
            for si in trade_sis:
                sym = syms[si]
                if np.isnan(C[si, di]) or C[si, di] <= 0:
                    continue
                if any(p['si'] == si for p in running_positions):
                    continue
                div = divergence[si, di]
                if np.isnan(div) or div <= threshold:
                    continue
                if use_oi_confirm:
                    oc = oi_change[si, di]
                    if np.isnan(oc) or oc <= 0:
                        continue
                cands.append((si, div))

            if cands:
                cands.sort(key=lambda x: -x[1])
                n_slots = top_n - len(running_positions)
                for si, score in cands[:n_slots]:
                    c = C[si, di]
                    if np.isnan(c) or c <= 0:
                        continue
                    mult = MULT.get(syms[si], DEF_MULT)
                    notional = c * mult
                    lots = int(running_cash / (notional * (1 + COMM)))
                    if use_vol_scale:
                        vr = vol_ratio[si, di]
                        if not np.isnan(vr) and vr > 0:
                            lots = int(lots / (1.0 + vr))
                    if use_circuit_brk and len(daily_equity) > 1:
                        pk = max(daily_equity)
                        if pk > 0:
                            dd_p = (pk - daily_equity[-1]) / pk
                            if dd_p > 0.15:
                                lots = int(lots * 0.5)
                    if lots <= 0:
                        continue
                    cost = notional * lots * (1 + COMM)
                    if cost > running_cash:
                        lots = int(running_cash * 0.95 / (notional * (1 + COMM)))
                        cost = notional * lots * (1 + COMM) if lots > 0 else 0
                    if lots <= 0 or cost <= 0 or cost > running_cash:
                        continue
                    running_cash -= cost
                    running_positions.append({
                        'si': si, 'entry': c, 'entry_di': di,
                        'lots': lots, 'dir': 1, 'sym': syms[si],
                    })

        # Calculate max drawdown
        max_dd = 0.0
        if daily_equity:
            peak = daily_equity[0]
            for eq in daily_equity:
                if eq > peak:
                    peak = eq
                if peak > 0:
                    dd = (peak - eq) / peak * 100
                    if dd > max_dd:
                        max_dd = dd

        # Results
        if wf_mode:
            n_days_test = test_end_di - test_start_di
        else:
            n_days_test = ND - MIN_TRAIN

        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)

        # Profit factor
        wins = [t['pnl_pct'] for t in trades if t['pnl_pct'] > 0]
        losses = [-t['pnl_pct'] for t in trades if t['pnl_pct'] < 0]
        gross_profit = sum(wins) if wins else 0
        gross_loss = sum(losses) if losses else 0.001
        pf = gross_profit / gross_loss if gross_loss > 0 else 0

        return {
            'ann': ann, 'wr': wr, 'n': n_trades,
            'dd': max_dd, 'pf': pf,
            'final_cash': cash, 'n_days': n_days_test,
        }

    # ================================================================
    # ENHANCEMENT COMPARISON
    # ================================================================
    print("\n--- Enhancement Comparison ---")
    print(f"  {'Enhancement':<14} | {'Ann':>9} | {'WR':>5} | {'DD':>6} | {'PF':>5} | {'N':>5} | {'Description'}")
    print("-" * 90)

    comp_results = {}
    for enh_name, enh_flags in ENHANCEMENTS.items():
        t1 = time.time()
        r = run_backtest(enh_name, enh_flags, threshold=0.003, top_n=3)
        if r:
            desc = {
                'none': 'baseline (V74)',
                'tail_risk': 'VaR breach skip',
                'oi_confirm': 'OI rising only',
                'vol_scaled': 'vol-adjusted size',
                'circuit_brk': 'DD circuit breaker',
                'tail+oi': 'tail + OI',
                'tail+oi+vol': 'tail + OI + vol',
                'all': 'all enhancements',
            }.get(enh_name, '')
            print(f"  {enh_name:<14} | {r['ann']:>+8.1f}% | {r['wr']:>4.1f}% | {r['dd']:>5.1f}% | {r['pf']:>5.2f} | {r['n']:>5} | {desc}")
            comp_results[enh_name] = r
        else:
            print(f"  {enh_name:<14} | FAILED")

    # Also test with top_n=1
    print("\n--- Enhancement Comparison (top_n=1) ---")
    print(f"  {'Enhancement':<14} | {'Ann':>9} | {'WR':>5} | {'DD':>6} | {'PF':>5} | {'N':>5} | {'Description'}")
    print("-" * 90)
    comp_results_tn1 = {}
    for enh_name, enh_flags in ENHANCEMENTS.items():
        r = run_backtest(enh_name, enh_flags, threshold=0.003, top_n=1)
        if r:
            print(f"  {enh_name:<14} | {r['ann']:>+8.1f}% | {r['wr']:>4.1f}% | {r['dd']:>5.1f}% | {r['pf']:>5.2f} | {r['n']:>5}")
            comp_results_tn1[enh_name] = r

    # ================================================================
    # THRESHOLD SENSITIVITY (for best enhancement combo)
    # ================================================================
    print("\n--- Threshold Sensitivity (all enhancements, top_n=3) ---")
    print(f"  {'Threshold':>10} | {'Ann':>9} | {'WR':>5} | {'DD':>6} | {'PF':>5} | {'N':>5}")
    print("-" * 60)
    for thresh in [0.001, 0.003, 0.005, 0.01, 0.02]:
        r = run_backtest('all', ENHANCEMENTS['all'], threshold=thresh, top_n=3)
        if r:
            print(f"  {thresh:>10.3f} | {r['ann']:>+8.1f}% | {r['wr']:>4.1f}% | {r['dd']:>5.1f}% | {r['pf']:>5.2f} | {r['n']:>5}")

    # ================================================================
    # WALK-FORWARD: Test best configs
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Pick top 5 enhancement combos for WF based on full-period results
    ranked = sorted(comp_results.items(), key=lambda x: (-x[1]['ann']))
    top_5 = [name for name, _ in ranked[:5]]
    # Also ensure 'none' (baseline) is included
    if 'none' not in top_5:
        top_5 = top_5[:4] + ['none']

    print(f"\n--- Walk-Forward (Best 5 configs, top_n=3) ---")
    print(f"  {'Enhancement':<14} | {'Avg':>9} | ", end="")
    for yr in wf_years:
        print(f" {yr:>7} |", end="")
    print(f" {'Pos':>4} | {'AvgDD':>6}")
    print("-" * 110)

    for enh_name in top_5:
        enh_flags = ENHANCEMENTS[enh_name]
        wf_row = {}
        for yr in wf_years:
            wr = run_backtest(enh_name, enh_flags, wf_test_year=yr, threshold=0.003, top_n=3)
            if wr:
                wf_row[yr] = wr

        vals = [wf_row.get(yr, {}).get('ann', 0) for yr in wf_years]
        dds = [wf_row.get(yr, {}).get('dd', 0) for yr in wf_years]
        avg = np.mean(vals) if vals else 0
        avg_dd = np.mean(dds) if dds else 0
        pos = sum(1 for v in vals if v > 0)

        print(f"  {enh_name:<14} | {avg:>+8.1f}% |", end="")
        for v in vals:
            print(f" {v:>+7.1f}% |", end="")
        print(f" {pos}/6 | {avg_dd:>5.1f}%")

    # ================================================================
    # WALK-FORWARD with top_n=1
    # ================================================================
    print(f"\n--- Walk-Forward (Best 5 configs, top_n=1) ---")
    print(f"  {'Enhancement':<14} | {'Avg':>9} | ", end="")
    for yr in wf_years:
        print(f" {yr:>7} |", end="")
    print(f" {'Pos':>4} | {'AvgDD':>6}")
    print("-" * 110)

    # Rank by top_n=1 results
    ranked_tn1 = sorted(comp_results_tn1.items(), key=lambda x: (-x[1]['ann']))
    top_5_tn1 = [name for name, _ in ranked_tn1[:5]]
    if 'none' not in top_5_tn1:
        top_5_tn1 = top_5_tn1[:4] + ['none']

    for enh_name in top_5_tn1:
        enh_flags = ENHANCEMENTS[enh_name]
        wf_row = {}
        for yr in wf_years:
            wr = run_backtest(enh_name, enh_flags, wf_test_year=yr, threshold=0.003, top_n=1)
            if wr:
                wf_row[yr] = wr

        vals = [wf_row.get(yr, {}).get('ann', 0) for yr in wf_years]
        dds = [wf_row.get(yr, {}).get('dd', 0) for yr in wf_years]
        avg = np.mean(vals) if vals else 0
        avg_dd = np.mean(dds) if dds else 0
        pos = sum(1 for v in vals if v > 0)

        print(f"  {enh_name:<14} | {avg:>+8.1f}% |", end="")
        for v in vals:
            print(f" {v:>+7.1f}% |", end="")
        print(f" {pos}/6 | {avg_dd:>5.1f}%")

    # ================================================================
    # KEY FINDINGS
    # ================================================================
    print("\n--- Key Findings ---")

    if 'none' in comp_results and 'tail_risk' in comp_results:
        base_dd = comp_results['none']['dd']
        tail_dd = comp_results['tail_risk']['dd']
        base_ann = comp_results['none']['ann']
        tail_ann = comp_results['tail_risk']['ann']
        dd_change = tail_dd - base_dd
        ann_change = tail_ann - base_ann
        if dd_change < 0:
            print(f"- Tail risk filter REDUCES DD by {abs(dd_change):.1f}pp ({base_dd:.1f}% -> {tail_dd:.1f}%), "
                  f"annual changes {ann_change:+.1f}pp ({base_ann:+.1f}% -> {tail_ann:+.1f}%)")
        else:
            print(f"- Tail risk filter does NOT reduce DD ({base_dd:.1f}% -> {tail_dd:.1f}%), "
                  f"annual changes {ann_change:+.1f}pp")

    if 'none' in comp_results and 'oi_confirm' in comp_results:
        base_wr = comp_results['none']['wr']
        oi_wr = comp_results['oi_confirm']['wr']
        wr_change = oi_wr - base_wr
        print(f"- OI confirmation changes WR by {wr_change:+.1f}pp ({base_wr:.1f}% -> {oi_wr:.1f}%)")

    if 'none' in comp_results and 'circuit_brk' in comp_results:
        base_dd = comp_results['none']['dd']
        cb_dd = comp_results['circuit_brk']['dd']
        base_ann = comp_results['none']['ann']
        cb_ann = comp_results['circuit_brk']['ann']
        print(f"- Circuit breaker: DD {base_dd:.1f}% -> {cb_dd:.1f}%, "
              f"annual {base_ann:+.1f}% -> {cb_ann:+.1f}%")

    if 'none' in comp_results and 'all' in comp_results:
        print(f"- Full package (all): DD {comp_results['none']['dd']:.1f}% -> {comp_results['all']['dd']:.1f}%, "
              f"annual {comp_results['none']['ann']:+.1f}% -> {comp_results['all']['ann']:+.1f}%, "
              f"WR {comp_results['none']['wr']:.1f}% -> {comp_results['all']['wr']:.1f}%")

    # Best overall
    best_name = max(comp_results, key=lambda k: comp_results[k]['ann'])
    best = comp_results[best_name]
    print(f"- Best enhancement: '{best_name}' with {best['ann']:+.1f}% annual, "
          f"{best['wr']:.1f}% WR, {best['dd']:.1f}% DD, {best['pf']:.2f} PF")

    # Best risk-adjusted (lowest DD with decent return)
    if comp_results:
        best_dd_name = min(comp_results, key=lambda k: comp_results[k]['dd'])
        best_dd = comp_results[best_dd_name]
        print(f"- Lowest DD: '{best_dd_name}' with {best_dd['dd']:.1f}% DD, "
              f"{best_dd['ann']:+.1f}% annual")

    print(f"\n  Total time: {time.time()-t_start:.1f}s")
    print("=" * 98)


if __name__ == '__main__':
    main()
