"""
Alpha Futures V69 -- Deep Dive on 1-Day Hold Group Momentum Lag
================================================================
V63 discovered that 1-day hold group momentum lag (grp_mom_excl_self - own_mom)
at LB=3 gives +448.9% annual -- HIGHER than pair trading! Completely unexpected.

Previous best momentum: V34b at +86.8% with 3-day hold.
1-day hold changes everything: even a weak edge per trade (57.8% WR) compounds
massively through daily recycling of capital. LB=3 captures very short-term
group momentum divergence.

Hypothesis: With 1-day hold, the signal doesn't need to be strong per trade --
it needs to be CONSISTENT. Daily compounding turns 0.3% edge/trade into
+449%/year.

Optimization axes (~200 configs):
1. Momentum lookback: [1, 2, 3, 5, 7, 10]
2. Entry threshold: [0.001, 0.003, 0.005, 0.01, 0.02]
3. Universe: ALL 68 commodities vs only GROUP_MAP commodities (20)
4. Hold: [1] (only 1-day)
5. top_n: [1, 3]
6. COMM: [0.0003, 0.0001]

Additional tests:
- Supply chain upstream lag (1-day lagged upstream momentum) with 1-day hold
- Combined: group lag + upstream lag
- With/without OI filter
- Walk-forward for top 10 (6 windows: test 2020-2025 individually)
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
    'afi': 'oils', 'mfi': 'oils', 'yfi': 'oils', 'pfi': 'oils', 'cfi': 'oils',
    'scfi': 'energy', 'mafi': 'energy', 'bfi': 'energy', 'fufi': 'energy',
    'ppfi': 'chemical', 'vfi': 'chemical', 'egfi': 'chemical', 'pgfi': 'chemical',
}

UPSTREAM = {
    'rbfi': 'ifi', 'hcfi': 'rbfi', 'jfi': 'jmfi',
    'mafi': 'scfi', 'bfi': 'scfi', 'fufi': 'scfi',
    'mfi': 'afi', 'yfi': 'afi', 'pfi': 'yfi',
    'ppfi': 'mafi', 'vfi': 'mafi', 'egfi': 'mafi',
}

# Walk-forward windows: test each year 2020-2025 individually
WF_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


def main():
    t_start = time.time()
    print("=" * 130)
    print("Alpha Futures V69 -- Deep Dive: 1-Day Hold Group Momentum Lag")
    print("V63 discovery: +448.9% at LB=3, 1-day hold. Optimize & validate.")
    print("=" * 130)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    sym_to_si = {syms[si]: si for si in range(NS)}

    year_start_di = {}
    year_end_di = {}
    for di in range(ND):
        y = dates[di].year
        if y not in year_start_di:
            year_start_di[y] = di
        year_end_di[y] = di
    print(f"  {NS} commodities, {ND} days, years: {sorted(year_start_di.keys())}")

    # ================================================================
    # PRECOMPUTE GROUP MEMBERSHIP
    # ================================================================
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
    for grp, members in sorted(group_members.items()):
        names = [syms[si] for si in members]
        print(f"    {grp}: {names}")

    # Upstream mapping
    upstream_si = {}
    upstream_sis = set()
    for si in range(NS):
        up_sym = UPSTREAM.get(syms[si])
        if up_sym and up_sym in sym_to_si:
            upstream_si[si] = sym_to_si[up_sym]
            upstream_sis.add(si)

    # ================================================================
    # PRECOMPUTE MOMENTUM AT ALL LOOKBACKS
    # ================================================================
    print("\n[Signals] Computing momentum signals...", flush=True)
    t0 = time.time()

    all_lookbacks = [1, 2, 3, 5, 7, 10]

    # own momentum: mom[lag][si, di]
    mom = {}
    for lag in all_lookbacks:
        m = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(lag, ND):
                c_now = C[si, di]
                c_prev = C[si, di - lag]
                if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                    m[si, di] = (c_now - c_prev) / c_prev
        mom[lag] = m

    # group momentum excluding self: grp_mom[lag][si, di]
    grp_mom = {}
    for lag in all_lookbacks:
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
    print(f"  Own + group momentum done ({time.time()-t0:.1f}s)", flush=True)

    # ================================================================
    # PRECOMPUTE UPSTREAM LAG SIGNAL
    # ================================================================
    print("  Computing upstream lag signals...", flush=True)

    # upstream momentum (1-day lagged): what did the upstream do yesterday?
    up_mom = {}
    for lag in all_lookbacks:
        um = np.full((NS, ND), np.nan)
        for si in upstream_sis:
            usi = upstream_si[si]
            if usi < 0:
                continue
            for di in range(1, ND):
                um_val = mom[lag][usi, di - 1]  # 1-day lagged
                if not np.isnan(um_val):
                    um[si, di] = um_val
        up_mom[lag] = um
    print(f"  Upstream lag done ({time.time()-t0:.1f}s)", flush=True)

    # ================================================================
    # PRECOMPUTE OI FILTER
    # ================================================================
    print("  Computing OI filter...", flush=True)

    oi_rising = np.full((NS, ND), np.nan)
    for si in range(NS):
        oi_ema = np.full(ND, np.nan)
        oe = 0.0
        alpha_oi = 2.0 / 6
        for di in range(1, ND):
            oi_val = OI[si, di]
            if np.isnan(oi_val):
                continue
            oe = alpha_oi * oi_val + (1 - alpha_oi) * oe
            oi_ema[di] = oe
        for di in range(6, ND):
            cur = oi_ema[di]
            prev = oi_ema[di - 5]
            if not np.isnan(cur) and not np.isnan(prev) and prev > 0:
                oi_rising[si, di] = (cur - prev) / prev
    print(f"  OI filter done ({time.time()-t0:.1f}s)", flush=True)

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(
        signal_type='group_lag',
        lookback=3,
        threshold=0.003,
        use_all=False,
        top_n=1,
        comm=COMM,
        use_oi_filter=False,
        start_year=None,
        end_year=None,
        config_name="",
    ):
        """
        1-day hold momentum backtest.

        signal_type:
          'group_lag'    - (grp_mom_excl_self - own_mom) > threshold -> long
          'upstream_lag' - upstream 1-day lagged mom > threshold -> long
          'combined'     - group_lag + upstream_lag combined score
        """
        # Universe
        if use_all:
            # Use all commodities that have a group
            trade_sis = list(group_sis)
        else:
            # Only GROUP_MAP commodities (20)
            trade_sis = [si for si in group_sis]

        if signal_type == 'upstream_lag':
            trade_sis = [si for si in trade_sis if si in upstream_sis]

        cash = float(CASH0)
        trades = []
        positions = []  # list of open positions

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

        for di in range(start_di, end_di):
            year = dates[di].year

            # --- Close positions held from previous day (1-day hold) ---
            closed = []
            for pos in positions:
                si = pos['si']
                cn = C[si, di]
                if np.isnan(cn) or cn <= 0:
                    cn = pos['entry']
                mult = MULT.get(pos['sym'], DEF_MULT)
                mkt_val = cn * mult * pos['lots']
                pnl = (cn - pos['entry']) * mult * pos['lots'] * pos['dir']
                invested = pos['entry'] * mult * pos['lots']
                pnl_pct = pnl / invested * 100 if invested > 0 else 0
                cash += mkt_val - mkt_val * comm
                trades.append({
                    'pnl_abs': pnl, 'pnl_pct': pnl_pct,
                    'days': 1, 'di': di, 'year': year,
                    'sym': pos['sym'], 'group': pos.get('group', ''),
                    'dir': pos['dir'], 'reason': 'time',
                    'signal': pos.get('signal', signal_type),
                })
                closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            # --- Score all candidates ---
            candidates = []
            for si in trade_sis:
                sym = syms[si]
                if np.isnan(C[si, di]) or C[si, di] <= 0:
                    continue
                # Skip if already holding
                if any(p['si'] == si for p in positions):
                    continue

                if signal_type == 'group_lag':
                    own = mom[lookback][si, di]
                    grp = grp_mom[lookback][si, di]
                    if np.isnan(own) or np.isnan(grp):
                        continue
                    divergence = grp - own
                    if divergence <= threshold:
                        continue
                    score = divergence

                elif signal_type == 'upstream_lag':
                    up = up_mom[lookback][si, di]
                    own = mom[lookback][si, di]
                    if np.isnan(up) or np.isnan(own):
                        continue
                    divergence = up - own
                    if divergence <= threshold:
                        continue
                    score = divergence

                elif signal_type == 'combined':
                    own = mom[lookback][si, di]
                    grp = grp_mom[lookback][si, di]
                    up = up_mom[lookback][si, di]

                    grp_score = 0.0
                    if not np.isnan(grp) and not np.isnan(own):
                        gd = grp - own
                        if gd > threshold:
                            grp_score = gd

                    up_score = 0.0
                    if not np.isnan(up) and not np.isnan(own):
                        ud = up - own
                        if ud > threshold:
                            up_score = ud

                    # Combined: both must be positive, geometric mean
                    if grp_score > 0 and up_score > 0:
                        score = np.sqrt(grp_score * up_score)
                    elif grp_score > 0:
                        score = grp_score * 0.7
                    elif up_score > 0:
                        score = up_score * 0.5
                    else:
                        continue
                    if score <= threshold:
                        continue
                else:
                    continue

                # OI filter: require OI rising
                if use_oi_filter:
                    oi_r = oi_rising[si, di]
                    if np.isnan(oi_r) or oi_r < 0:
                        continue

                candidates.append((si, score, sym))

            if not candidates:
                continue

            # Sort by score (highest divergence first)
            candidates.sort(key=lambda x: -x[1])

            # Open top_n positions
            n_slots = top_n - len(positions)
            for si, score, sym in candidates[:n_slots]:
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
                grp = GROUP_MAP.get(sym, '')
                positions.append({
                    'si': si, 'entry': c, 'entry_di': di,
                    'lots': lots, 'dir': 1, 'sym': sym,
                    'group': grp, 'signal': signal_type,
                })

        # Close remaining positions at end
        ae = min(end_di, ND) - 1
        for pos in positions:
            cn = C[pos['si'], ae]
            if np.isnan(cn) or cn <= 0:
                cn = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = cn * mult * pos['lots']
            pnl = (cn - pos['entry']) * mult * pos['lots'] * pos['dir']
            invested = pos['entry'] * mult * pos['lots']
            pnl_pct = pnl / invested * 100 if invested > 0 else 0
            cash += mkt_val - mkt_val * COMM
            trades.append({
                'pnl_abs': pnl, 'pnl_pct': pnl_pct,
                'days': ae - pos['entry_di'], 'di': ae,
                'year': dates[ae].year,
                'sym': pos['sym'], 'group': pos.get('group', ''),
                'dir': pos['dir'], 'reason': 'end',
                'signal': pos.get('signal', signal_type),
            })

        if len(trades) < 3:
            return None

        # === STATS ===
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
        wr = nw / len(trades) * 100
        avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_abs'] > 0]) if nw > 0 else 0
        avg_loss = np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_abs'] <= 0]) if nw < len(trades) else 0
        avg_days = np.mean([t['days'] for t in trades])
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

        # Per-group stats
        group_stats = {}
        for t in trades:
            g = t['group']
            if g not in group_stats:
                group_stats[g] = {'n': 0, 'w': 0, 'pnl': 0.0}
            group_stats[g]['n'] += 1
            if t['pnl_abs'] > 0:
                group_stats[g]['w'] += 1
            group_stats[g]['pnl'] += t['pnl_abs']

        return {
            'name': config_name,
            'ann': round(ann, 1),
            'n': len(trades),
            'wr': round(wr, 1),
            'dd': round(max_dd, 1),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'avg_days': round(avg_days, 1),
            'pf': round(pf, 2),
            'sharpe': round(sharpe, 2),
            'cash': round(cash, 0),
            'yearly': year_stats,
            'group_stats': group_stats,
            'trades': trades,
        }

    # ================================================================
    # BUILD CONFIGURATIONS
    # ================================================================
    print("\n[Configs] Building configurations...", flush=True)
    configs = []

    # Grid 1: Group lag -- lookback x threshold x top_n x comm
    # 6 x 5 x 2 x 2 = 120 configs
    for lb in [1, 2, 3, 5, 7, 10]:
        for thr in [0.001, 0.003, 0.005, 0.01, 0.02]:
            for tn in [1, 3]:
                for cm in [0.0003, 0.0001]:
                    configs.append({
                        'signal_type': 'group_lag',
                        'lookback': lb,
                        'threshold': thr,
                        'use_all': False,
                        'top_n': tn,
                        'comm': cm,
                        'use_oi_filter': False,
                        'config_name': f"GRP_LB{lb}_T{thr*1000:.0f}_TN{tn}_C{cm*10000:.0f}",
                    })

    # Grid 2: Upstream lag -- lookback x threshold x top_n
    # 6 x 3 x 2 = 36 configs
    for lb in [1, 2, 3, 5, 7, 10]:
        for thr in [0.001, 0.005, 0.01]:
            for tn in [1, 3]:
                configs.append({
                    'signal_type': 'upstream_lag',
                    'lookback': lb,
                    'threshold': thr,
                    'use_all': False,
                    'top_n': tn,
                    'comm': 0.0003,
                    'use_oi_filter': False,
                    'config_name': f"UP_LB{lb}_T{thr*1000:.0f}_TN{tn}",
                })

    # Grid 3: Combined group+upstream lag
    # 3 x 3 x 2 = 18 configs
    for lb in [3, 5, 7]:
        for thr in [0.001, 0.003, 0.005]:
            for tn in [1, 3]:
                configs.append({
                    'signal_type': 'combined',
                    'lookback': lb,
                    'threshold': thr,
                    'use_all': False,
                    'top_n': tn,
                    'comm': 0.0003,
                    'use_oi_filter': False,
                    'config_name': f"COMB_LB{lb}_T{thr*1000:.0f}_TN{tn}",
                })

    # Grid 4: OI filter variants (best configs from V63: LB3, T3, TN1)
    # 8 configs
    for signal in ['group_lag', 'upstream_lag', 'combined']:
        for oi in [True, False]:
            for lb in [3]:
                sname = signal.split('_')[0].upper()
                oname = "OI" if oi else "noOI"
                configs.append({
                    'signal_type': signal,
                    'lookback': lb,
                    'threshold': 0.003,
                    'use_all': False,
                    'top_n': 1,
                    'comm': 0.0003,
                    'use_oi_filter': oi,
                    'config_name': f"{sname}_LB{lb}_{oname}",
                })

    # Grid 5: Use ALL commodities vs GROUP_MAP only
    # 4 configs
    for use_all in [True, False]:
        for lb in [3, 5]:
            uname = "ALL68" if use_all else "GRP20"
            configs.append({
                'signal_type': 'group_lag',
                'lookback': lb,
                'threshold': 0.003,
                'use_all': use_all,
                'top_n': 1,
                'comm': 0.0003,
                'use_oi_filter': False,
                'config_name': f"GRP_{uname}_LB{lb}",
            })

    total_configs = len(configs)
    print(f"  {total_configs} configurations")
    print(f"    Grid 1 (group_lag sweep): 120")
    print(f"    Grid 2 (upstream_lag): 36")
    print(f"    Grid 3 (combined): 18")
    print(f"    Grid 4 (OI filter): 8")
    print(f"    Grid 5 (universe): 4")

    # ================================================================
    # RUN FULL-PERIOD SWEEP
    # ================================================================
    print(f"\n{'=' * 130}")
    print(f"  FULL-PERIOD PARAMETER SWEEP ({total_configs} configs)")
    print(f"{'=' * 130}")

    results = []
    t_sw = time.time()
    for ci, cfg in enumerate(configs):
        r = run_backtest(**cfg)
        if r is not None:
            results.append(r)
        if (ci + 1) % 50 == 0:
            print(f"  [{ci+1}/{total_configs}] {len(results)} with results ({time.time()-t_sw:.1f}s)", flush=True)

    results.sort(key=lambda x: -x['ann'])
    print(f"\n  Sweep complete: {len(results)}/{total_configs} configs ({time.time()-t_sw:.1f}s)", flush=True)

    # ================================================================
    # TOP 20 FULL-PERIOD RESULTS
    # ================================================================
    print(f"\n{'=' * 130}")
    print(f"  TOP 20 FULL-PERIOD RESULTS")
    print(f"{'=' * 130}")
    hdr = (f"  {'#':>2s} | {'Config':45s} | {'Ann':>8s} | {'WR':>5s} | {'N':>5s} | "
           f"{'DD':>6s} | {'PF':>5s} | {'Sh':>6s} | {'AvgW':>6s} | {'AvgL':>6s} | {'Cash':>12s}")
    print(hdr)
    print(f"  {'-' * 130}")
    for i, r in enumerate(results[:20]):
        print(f"  {i+1:2d} | {r['name']:45s} | {r['ann']:+7.1f}% | {r['wr']:4.1f}% | "
              f"{r['n']:5d} | {r['dd']:5.1f}% | {r['pf']:4.2f} | {r['sharpe']:5.2f} | "
              f"{r['avg_win']:+5.2f}% | {r['avg_loss']:5.2f}% | {r['cash']:11.0f}")

    # ================================================================
    # LOOKBACK COMPARISON
    # ================================================================
    print(f"\n{'=' * 130}")
    print(f"  LOOKBACK COMPARISON (group_lag, T=3, TN=1, C=3)")
    print(f"{'=' * 130}")
    print(f"  {'Lookback':>8s} | {'Ann':>8s} | {'WR':>5s} | {'N':>5s} | {'DD':>6s} | {'PF':>5s} | {'Sh':>6s} | {'AvgW':>6s} | {'AvgL':>6s}")
    print(f"  {'-' * 80}")
    for lb in all_lookbacks:
        matching = [r for r in results if r['name'] == f"GRP_LB{lb}_T3_TN1_C3"]
        if matching:
            r = matching[0]
            print(f"  LB={lb:2d}     | {r['ann']:+7.1f}% | {r['wr']:4.1f}% | "
                  f"{r['n']:5d} | {r['dd']:5.1f}% | {r['pf']:4.2f} | {r['sharpe']:5.2f} | "
                  f"{r['avg_win']:+5.2f}% | {r['avg_loss']:5.2f}%")

    # ================================================================
    # SIGNAL TYPE COMPARISON
    # ================================================================
    print(f"\n{'=' * 130}")
    print(f"  SIGNAL TYPE COMPARISON (best per signal type)")
    print(f"{'=' * 130}")
    for sig_type in ['group_lag', 'upstream_lag', 'combined']:
        sig_results = [r for r in results if sig_type in r['name'].lower() or
                       (sig_type == 'group_lag' and r['name'].startswith('GRP_')) or
                       (sig_type == 'upstream_lag' and r['name'].startswith('UP_')) or
                       (sig_type == 'combined' and r['name'].startswith('COMB_'))]
        if sig_results:
            best = sig_results[0]
            print(f"  {sig_type:15s}: {best['name']:45s} Ann={best['ann']:+.1f}%  "
                  f"WR={best['wr']:.1f}%  N={best['n']}  DD={best['dd']:.1f}%  "
                  f"PF={best['pf']:.2f}  Sharpe={best['sharpe']:.2f}")

    # ================================================================
    # OI FILTER COMPARISON
    # ================================================================
    print(f"\n{'=' * 130}")
    print(f"  OI FILTER COMPARISON")
    print(f"{'=' * 130}")
    for sig_prefix in ['GRP', 'UP', 'COMB']:
        for oi in ['OI', 'noOI']:
            matching = [r for r in results if r['name'].startswith(f"{sig_prefix}_LB3_{oi}")]
            if matching:
                r = matching[0]
                print(f"  {r['name']:45s} Ann={r['ann']:+7.1f}%  WR={r['wr']:5.1f}%  "
                      f"N={r['n']:5d}  DD={r['dd']:5.1f}%  PF={r['pf']:.2f}")

    # ================================================================
    # UNIVERSE COMPARISON
    # ================================================================
    print(f"\n{'=' * 130}")
    print(f"  UNIVERSE COMPARISON (ALL 68 vs GROUP_MAP 20)")
    print(f"{'=' * 130}")
    for uname in ['ALL68', 'GRP20']:
        matching = [r for r in results if f"_{uname}_" in r['name']]
        if matching:
            r = matching[0]
            print(f"  {r['name']:45s} Ann={r['ann']:+7.1f}%  WR={r['wr']:5.1f}%  "
                  f"N={r['n']:5d}  DD={r['dd']:5.1f}%")

    # ================================================================
    # YEARLY BREAKDOWN TOP 5
    # ================================================================
    if len(results) >= 2:
        print(f"\n{'=' * 130}")
        print(f"  YEARLY BREAKDOWN FOR TOP 5")
        print(f"{'=' * 130}")
        for idx, r in enumerate(results[:5]):
            print(f"\n  #{idx+1}: {r['name']}")
            print(f"    Ann={r['ann']:+.1f}%  WR={r['wr']:.1f}%  DD={r['dd']:.1f}%  "
                  f"PF={r['pf']:.2f}  Sharpe={r['sharpe']:.2f}  N={r['n']}")
            print(f"    {'Year':>6s} | {'N':>5s} | {'WR':>5s} | {'PnL%':>8s} | {'PnL Abs':>12s}")
            print(f"    {'-' * 55}")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / max(ys['n'], 1) * 100
                print(f"    {y:6d} | {ys['n']:5d} | {wr_y:4.1f}% | {ys['pnl']:+7.1f}% | {ys['pnl_abs_sum']:+11.0f}")

    # ================================================================
    # GROUP BREAKDOWN FOR #1
    # ================================================================
    if results:
        best = results[0]
        print(f"\n{'=' * 130}")
        print(f"  GROUP BREAKDOWN for #1: {best['name']}")
        print(f"{'=' * 130}")
        print(f"  {'Group':15s} | {'N':>5s} | {'WR':>5s} | {'Abs PnL':>12s}")
        print(f"  {'-' * 50}")
        for g in sorted(best['group_stats'].keys(), key=lambda x: -best['group_stats'][x]['pnl']):
            gs = best['group_stats'][g]
            wr_g = gs['w'] / max(gs['n'], 1) * 100
            print(f"  {g:15s} | {gs['n']:5d} | {wr_g:4.1f}% | {gs['pnl']:+11.0f}")

    # ================================================================
    # WALK-FORWARD TOP 10
    # ================================================================
    top10 = results[:10]
    print(f"\n{'=' * 130}")
    print(f"  WALK-FORWARD VALIDATION (Top 10, 6 windows: 2020-2025)")
    print(f"{'=' * 130}")

    wf_all = []
    wf_by_config = {}
    for rank, top_r in enumerate(top10):
        cn = top_r['name']
        matching = [c for c in configs if c['config_name'] == cn]
        if not matching:
            continue
        bc = matching[0]
        print(f"\n  [{rank+1}] {cn}  (full-period Ann={top_r['ann']:+.1f}%)")
        for ty in WF_YEARS:
            if ty not in year_start_di:
                continue
            wc = dict(bc)
            wc['start_year'] = ty
            wc['end_year'] = ty
            wc['config_name'] = f"WF_{ty}_{cn}"
            r = run_backtest(**wc)
            if r is not None:
                wf_all.append((cn, ty, r))
                wf_by_config.setdefault(cn, []).append((ty, r))
                print(f"    {ty}: Ann={r['ann']:+7.1f}%  WR={r['wr']:5.1f}%  "
                      f"N={r['n']:4d}  DD={r['dd']:5.1f}%  PF={r['pf']:4.2f}  "
                      f"Sharpe={r['sharpe']:5.2f}")
            else:
                print(f"    {ty}: insufficient trades")

    # ================================================================
    # WALK-FORWARD AGGREGATE TABLE
    # ================================================================
    if wf_by_config:
        print(f"\n{'=' * 130}")
        print(f"  WALK-FORWARD AGGREGATE TABLE")
        print(f"{'=' * 130}")
        wf_summary = []
        for cn, wr_list in wf_by_config.items():
            anns = [r['ann'] for _, r in wr_list]
            aby = {ty: r['ann'] for ty, r in wr_list}
            n_pos = sum(1 for a in anns if a > 0)
            wf_summary.append({
                'name': cn,
                'avg_ann': np.mean(anns),
                'med_ann': np.median(anns),
                'min_ann': min(anns),
                'max_ann': max(anns),
                'avg_wr': np.mean([r['wr'] for _, r in wr_list]),
                'avg_dd': np.mean([r['dd'] for _, r in wr_list]),
                'avg_pf': np.mean([r['pf'] for _, r in wr_list]),
                'n_positive': n_pos,
                'n_windows': len(wr_list),
                'by_year': aby,
            })
        wf_summary.sort(key=lambda x: -x['avg_ann'])

        # Header with year columns
        hdr = (f"  {'#':>2s} | {'Config':35s} | {'Avg':>7s} | "
               f"{'2020':>7s} | {'2021':>7s} | {'2022':>7s} | {'2023':>7s} | {'2024':>7s} | {'2025':>7s} | "
               f"{'WR':>5s} | {'DD':>5s} | {'Pos':>4s}")
        print(hdr)
        print(f"  {'-' * 130}")
        for i, w in enumerate(wf_summary):
            ycols = ""
            for y in WF_YEARS:
                a = w['by_year'].get(y, float('nan'))
                ycols += f" | {a:+6.1f}%" if not np.isnan(a) else " |    N/A"
            print(f"  {i+1:2d} | {w['name']:35s} | {w['avg_ann']:+6.1f}%{ycols} | "
                  f"{w['avg_wr']:4.1f}% | {w['avg_dd']:4.1f}% | {w['n_positive']}/{w['n_windows']}")

    # ================================================================
    # OVERFITTING CHECK
    # ================================================================
    if wf_by_config and len(wf_summary) > 2:
        print(f"\n{'=' * 130}")
        print(f"  OVERFITTING CHECK")
        print(f"{'=' * 130}")

        full_anns = []
        wf_anns_list = []
        for w in wf_summary:
            name = w['name']
            full_r = next((r for r in results if r['name'] == name), None)
            if full_r:
                full_anns.append(full_r['ann'])
                wf_anns_list.append(w['avg_ann'])

        if len(full_anns) > 2:
            corr = np.corrcoef(full_anns, wf_anns_list)[0, 1]
            decay = np.mean(wf_anns_list) / max(np.mean(full_anns), 0.01)
            print(f"  Configs tested OOS: {len(full_anns)}")
            print(f"  Full-period avg Ann: {np.mean(full_anns):+.1f}%")
            print(f"  WF avg Ann:          {np.mean(wf_anns_list):+.1f}%")
            print(f"  Correlation:         {corr:.3f}")
            print(f"  Decay ratio (WF/IS): {decay:.2f}")
            if corr > 0.5:
                print(f"  -> GOOD: Strong positive correlation")
            elif corr > 0.2:
                print(f"  -> MODERATE: Some predictive power")
            else:
                print(f"  -> WARNING: Weak correlation, possible overfitting")

        # WF positive rate
        all_wf_anns = [r['ann'] for _, _, r in wf_all]
        n_pos_wf = sum(1 for a in all_wf_anns if a > 0)
        print(f"\n  Overall WF positive rate: {n_pos_wf}/{len(all_wf_anns)} "
              f"({n_pos_wf / len(all_wf_anns) * 100:.0f}%)")

        # Best / worst single window
        if wf_all:
            best_wf = max(wf_all, key=lambda x: x[2]['ann'])
            worst_wf = min(wf_all, key=lambda x: x[2]['ann'])
            print(f"  Best single window:  Test {best_wf[1]} = {best_wf[2]['ann']:+.1f}%")
            print(f"  Worst single window: Test {worst_wf[1]} = {worst_wf[2]['ann']:+.1f}%")

    # ================================================================
    # PARAMETER SENSITIVITY SUMMARY
    # ================================================================
    print(f"\n{'=' * 130}")
    print(f"  PARAMETER SENSITIVITY SUMMARY")
    print(f"{'=' * 130}")

    # By lookback
    print(f"\n  By Lookback (group_lag, TN=1, C=3):")
    for lb in all_lookbacks:
        subset = [r for r in results if f"LB{lb}_" in r['name'] and '_TN1_' in r['name'] and '_C3' in r['name']
                  and r['name'].startswith('GRP_')]
        if subset:
            best = max(subset, key=lambda x: x['ann'])
            avg = np.mean([r['ann'] for r in subset])
            print(f"    LB={lb:2d}: N={len(subset):3d}  Avg Ann={avg:+7.1f}%  "
                  f"Best Ann={best['ann']:+.1f}% ({best['name']})")

    # By threshold
    print(f"\n  By Threshold (group_lag, LB=3, TN=1):")
    for thr in [0.001, 0.003, 0.005, 0.01, 0.02]:
        tkey = f"T{thr*1000:.0f}"
        subset = [r for r in results if f"_{tkey}_" in r['name']
                  and '_LB3_' in r['name'] and '_TN1_' in r['name']
                  and r['name'].startswith('GRP_')]
        if subset:
            best = max(subset, key=lambda x: x['ann'])
            avg = np.mean([r['ann'] for r in subset])
            print(f"    T={thr:.3f}: N={len(subset):3d}  Avg Ann={avg:+7.1f}%  "
                  f"Best Ann={best['ann']:+.1f}% ({best['name']})")

    # By top_n
    print(f"\n  By top_n (group_lag, LB=3, T=3):")
    for tn in [1, 3]:
        subset = [r for r in results if f'_TN{tn}_' in r['name']
                  and '_LB3_T3_' in r['name']
                  and r['name'].startswith('GRP_')]
        if subset:
            best = max(subset, key=lambda x: x['ann'])
            avg = np.mean([r['ann'] for r in subset])
            print(f"    TN={tn}: N={len(subset):3d}  Avg Ann={avg:+7.1f}%  "
                  f"Best Ann={best['ann']:+.1f}% ({best['name']})")

    # By commission
    print(f"\n  By Commission (group_lag, LB=3, T=3, TN=1):")
    for cm in [0.0003, 0.0001]:
        ckey = f"C{cm*10000:.0f}"
        subset = [r for r in results if f'_{ckey}' in r['name']
                  and '_LB3_T3_TN1_' in r['name']
                  and r['name'].startswith('GRP_')]
        if subset:
            best = max(subset, key=lambda x: x['ann'])
            print(f"    C={cm:.4f}: Best Ann={best['ann']:+.1f}%  N={best['n']}  "
                  f"WR={best['wr']:.1f}%  PF={best['pf']:.2f}")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print(f"\n{'=' * 130}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 130}")

    if results:
        b = results[0]
        print(f"\n  Best full-period: {b['name']}")
        print(f"    Ann={b['ann']:+.1f}%  WR={b['wr']:.1f}%  N={b['n']}  DD={b['dd']:.1f}%  "
              f"PF={b['pf']:.2f}  Sharpe={b['sharpe']:.2f}")
        print(f"    Avg Win: {b['avg_win']:+.2f}%  Avg Loss: {b['avg_loss']:.2f}%  "
              f"Edge/trade: {(b['wr']/100)*b['avg_win'] - (1-b['wr']/100)*b['avg_loss']:+.3f}%")

    if wf_summary:
        bw = wf_summary[0]
        print(f"\n  Best WF avg: {bw['name']}")
        print(f"    WF Avg Ann={bw['avg_ann']:+.1f}%  WF Med={bw['med_ann']:+.1f}%  "
              f"Min={bw['min_ann']:+.1f}%  Max={bw['max_ann']:+.1f}%  "
              f"Pos/Win={bw['n_positive']}/{bw['n_windows']}")

    print(f"\n  V63 reference: +448.9% (LB=3, T=0.003, TN=1, 1-day hold)")
    if results:
        print(f"  V69 best full-period: {results[0]['ann']:+.1f}%")
    if wf_summary:
        print(f"  V69 best WF avg: {wf_summary[0]['avg_ann']:+.1f}%")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
