"""
Alpha Futures V71 -- Pure Momentum + Pair Confirmation
=======================================================
KEY INSIGHT: V63 discovered that 1-day hold group momentum lag (LB=3)
gives +448.9% annual -- HIGHER than pair trading! But the combo
(pairs+momentum) only gave +369.1%, meaning pair trades REPLACED some
profitable momentum trades. This suggests momentum should be the PRIMARY
signal, with pairs used as CONFIRMATION only.

V71 HYPOTHESIS: Instead of priority-based combo (pairs first, momentum
fallback), use momentum as primary and pair z-score as confirmation
filter. When momentum fires AND pair z-score agrees with direction, take
the trade. When momentum fires without pair confirmation, still take it
but with different sizing.

APPROACH:
  1. Compute group momentum lag (grp_mom_excl_self - own_mom) for all
     commodities with GROUP_MAP. ONLY LONG side: buy when group > self
     (expect catch-up). No shorting.
  2. Compute pair z-scores (LOG spread, LB=15, all 14 pairs)
  3. For each day:
     a. Find the top-ranked commodity by positive momentum divergence
     b. If score > threshold:
        - Check if any pair z-score CONFIRMS the direction
        - CONFIRMED trade: take it
        - UNCONFIRMED trade: take it (momentum alone is strong enough)
     c. Hold 1 day, exit at next close

CONFIGURATIONS (~150):
  - Momentum LB: [2, 3, 5]
  - Momentum threshold: [0.001, 0.003, 0.005, 0.01]
  - Pair z-score threshold for confirmation: [1.0, 1.5, 2.0]
  - Modes: pure_momentum, confirmed_only, anti_confirmed
  - COMM: [0.0003, 0.0001]
  - Universe: group_map commodities only (25 with group assignments)

Walk-forward: 6 expanding windows (test years 2020-2025 individually)
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

GROUP_MAP = {
    'rbfi': 'ferrous', 'hcfi': 'ferrous', 'ifi': 'ferrous', 'jfi': 'ferrous', 'jmfi': 'ferrous',
    'cufi': 'nonferrous', 'alfi': 'nonferrous', 'znfi': 'nonferrous', 'nifi': 'nonferrous',
    'aufi': 'precious', 'agfi': 'precious',
    'afi': 'oils', 'mfi': 'oils', 'yfi': 'oils', 'pfi': 'oils', 'cfi': 'oils',
    'scfi': 'energy', 'mafi': 'energy', 'bfi': 'energy', 'fufi': 'energy', 'pgfi': 'energy',
    'ppfi': 'chemical', 'vfi': 'chemical', 'egfi': 'chemical', 'srfi': 'chemical',
}

PAIRS_14 = [
    ('rbfi', 'ifi'), ('hcfi', 'ifi'), ('hcfi', 'rbfi'),
    ('jfi', 'jmfi'), ('mafi', 'scfi'), ('fufi', 'scfi'),
    ('bfi', 'scfi'), ('mfi', 'afi'), ('yfi', 'afi'),
    ('pfi', 'yfi'), ('ppfi', 'mafi'), ('vfi', 'mafi'),
    ('egfi', 'mafi'), ('cfi', 'csfi'),
]

PAIR_LABEL = {
    ('rbfi', 'ifi'):  'rebar/iron_ore', ('hcfi', 'ifi'):  'hotcoil/iron_ore',
    ('hcfi', 'rbfi'): 'hotcoil/rebar',  ('jfi', 'jmfi'):  'coke/coal',
    ('mafi', 'scfi'): 'methanol/crude', ('fufi', 'scfi'): 'fueloil/crude',
    ('bfi', 'scfi'):  'bitumen/crude',  ('mfi', 'afi'):   'meal/soybean',
    ('yfi', 'afi'):   'soyoil/soybean', ('pfi', 'yfi'):   'palm/soyoil',
    ('ppfi', 'mafi'): 'PP/methanol',    ('vfi', 'mafi'):  'PVC/methanol',
    ('egfi', 'mafi'): 'EG/methanol',    ('cfi', 'csfi'):  'corn/cornstarch',
}

SPREAD_LOG = 'log'

WF_WINDOWS = [
    (2019, 2020),
    (2020, 2021),
    (2021, 2022),
    (2022, 2023),
    (2023, 2024),
    (2024, 2025),
]


def main():
    t_start = time.time()
    print("=" * 120)
    print("V71 -- Pure Momentum + Pair Confirmation")
    print("=" * 120)
    print("[Data] Loading...", flush=True)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    sym_to_si = {syms[si]: si for si in range(NS)}

    year_start_di = {}
    year_end_di = {}
    for di in range(ND):
        y = dates[di].year
        if y not in year_start_di:
            year_start_di[y] = di
        year_end_di[y] = di
    print(f"  {NS} commodities, {ND} days, years in data: {sorted(year_start_di.keys())}")

    # ================================================================
    # BUILD GROUP MEMBERSHIP
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

    # ================================================================
    # BUILD PAIR INDICES
    # ================================================================
    pair_indices = []
    for down_sym, up_sym in PAIRS_14:
        down_si = sym_to_si.get(down_sym, -1)
        up_si = sym_to_si.get(up_sym, -1)
        if down_si >= 0 and up_si >= 0:
            pair_indices.append((down_si, up_si, down_sym, up_sym))
    print(f"  Pairs: {len(pair_indices)}")

    # Build a reverse map: si -> list of (down_si, up_si, down_sym, up_sym)
    # For each pair the si appears in, also store whether it's the down or up leg
    si_pair_map = {}  # si -> [(pair_down_si, pair_up_si, pair_down_sym, pair_up_sym, is_down_leg)]
    for down_si, up_si, down_sym, up_sym in pair_indices:
        for si, is_down in [(down_si, True), (up_si, False)]:
            if si not in si_pair_map:
                si_pair_map[si] = []
            si_pair_map[si].append((down_si, up_si, down_sym, up_sym, is_down))

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

    # Momentum divergence score: grp_mom - own_mom
    # Positive = group rising faster than self -> buy (expect catch-up)
    # This is LONG ONLY -- consistent with V63's approach
    mom_div = {}
    for lag in [2, 3, 5]:
        div = np.full((NS, ND), np.nan)
        for si in group_sis:
            for di in range(lag, ND):
                own = mom[lag][si, di]
                grp = grp_mom[lag][si, di]
                if not np.isnan(own) and not np.isnan(grp):
                    div[si, di] = grp - own
        mom_div[lag] = div
    print(f"  Momentum signals precomputed ({time.time() - t0:.1f}s)", flush=True)

    # ================================================================
    # PRECOMPUTE PAIR Z-SCORES (LOG spread, multiple lookbacks)
    # ================================================================
    print("\n[Signals] Precomputing pair z-scores...", flush=True)
    t1 = time.time()

    PAIR_LBS = [10, 15, 20]
    pair_z = {}  # (down_si, up_si) -> {lb: z_array}

    for down_si, up_si, down_sym, up_sym in pair_indices:
        key = (down_si, up_si)
        spread = np.full(ND, np.nan)
        for di in range(ND):
            pd_val = C[down_si, di]
            pu = C[up_si, di]
            if np.isnan(pd_val) or np.isnan(pu) or pu <= 0 or pd_val <= 0:
                continue
            spread[di] = np.log(pd_val) - np.log(pu)

        pair_z[key] = {}
        for lb in PAIR_LBS:
            z = np.full(ND, np.nan)
            for di in range(lb, ND):
                window = spread[di - lb:di]
                valid = window[~np.isnan(window)]
                if len(valid) >= max(3, int(lb * 0.8)):
                    m_val = np.mean(valid)
                    s_val = np.std(valid, ddof=1)
                    if s_val > 1e-10:
                        z[di] = (spread[di] - m_val) / s_val
            pair_z[key][lb] = z

    print(f"  Pair z-scores precomputed ({time.time() - t1:.1f}s)", flush=True)

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(mom_lb=3, mom_thresh=0.003, pair_z_thresh=1.5,
                     pair_lb=15, mode='pure_momentum', comm=0.0003,
                     start_year=None, end_year=None, config_name=""):
        """
        Backtest a single configuration.

        mode:
          'pure_momentum'   - trade on momentum divergence regardless of pair z
          'confirmed_only'  - only trade when pair z-score confirms direction
          'anti_confirmed'  - trade all momentum signals BUT skip when pair
                              z-score actively DISAGREES with momentum direction
        """
        cash = float(CASH0)
        trades = []
        current_position = None

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

            # --- Close existing position (1-day hold) ---
            if current_position is not None:
                pos = current_position
                cn = C[pos['si'], di]
                if np.isnan(cn) or cn <= 0:
                    cn = pos['entry']
                mult = MULT.get(pos['sym'], DEF_MULT)
                mkt_val = cn * mult * pos['lots']
                pnl = (cn - pos['entry']) * mult * pos['lots']  # always long, dir=1
                invested = pos['entry'] * mult * pos['lots']
                pnl_pct = pnl / invested * 100 if invested > 0 else 0
                cost_exit = mkt_val * comm
                cash += mkt_val - cost_exit
                trades.append({
                    'pnl_abs': pnl - pos['cost_entry'] - cost_exit,
                    'pnl_pct': pnl_pct,
                    'days': di - pos['entry_di'],
                    'di': di,
                    'year': year,
                    'sym': pos['sym'],
                    'group': pos['group'],
                    'confirmed': pos['confirmed'],
                    'score': pos['score'],
                })
                current_position = None

            if current_position is not None:
                continue

            # --- Find best LONG momentum signal ---
            # Score = grp_mom - own_mom. Positive = group outperforming = buy laggard
            best_score = 0.0
            best_si = -1
            for si in group_sis:
                score = mom_div[mom_lb][si, di]
                if np.isnan(score):
                    continue
                if score > best_score:
                    best_score = score
                    best_si = si

            if best_si < 0 or best_score < mom_thresh:
                continue

            # --- Check pair confirmation ---
            # We're going LONG on `best_si`.
            # A pair z-score "confirms" if:
            #   - This symbol is the DOWN leg: z < -threshold means it's cheap relative
            #     to its pair partner, expecting mean reversion upward -> CONFIRMS LONG
            #   - This symbol is the UP leg: z > threshold means the DOWN leg is expensive,
            #     meaning UP should also rise (pairs move together) -> CONFIRMS LONG
            confirmed = False
            si = best_si
            sym = syms[si]
            related = si_pair_map.get(si, [])

            for dsi, usi, dsym, usym, is_down_leg in related:
                z_arr = pair_z.get((dsi, usi), {}).get(pair_lb)
                if z_arr is None:
                    continue
                zv = z_arr[di] if di < len(z_arr) else np.nan
                if np.isnan(zv):
                    continue

                if is_down_leg:
                    # We're buying the down leg. Confirmed if z < -threshold
                    # (down is cheap relative to up, expecting up mean reversion)
                    if zv < -pair_z_thresh:
                        confirmed = True
                        break
                else:
                    # We're buying the up leg. Confirmed if z > threshold
                    # (down is expensive relative to up, up should follow)
                    if zv > pair_z_thresh:
                        confirmed = True
                        break

            # --- Apply mode filter ---
            if mode == 'confirmed_only' and not confirmed:
                continue

            if mode == 'anti_confirmed':
                # Take all momentum signals EXCEPT when pair actively disagrees
                pair_disagrees = False
                for dsi, usi, dsym, usym, is_down_leg in related:
                    z_arr = pair_z.get((dsi, usi), {}).get(pair_lb)
                    if z_arr is None:
                        continue
                    zv = z_arr[di] if di < len(z_arr) else np.nan
                    if np.isnan(zv):
                        continue
                    # Pair disagrees if: we want to buy but the pair says SELL
                    if is_down_leg:
                        if zv > pair_z_thresh:
                            pair_disagrees = True
                            break
                    else:
                        if zv < -pair_z_thresh:
                            pair_disagrees = True
                            break
                if pair_disagrees:
                    continue

            # --- Open LONG position ---
            c = C[si, di]
            if np.isnan(c) or c <= 0:
                continue
            mult = MULT.get(sym, DEF_MULT)
            notional = c * mult
            lots = int(cash / (notional * (1 + comm)))
            if lots <= 0:
                continue
            cost_entry = notional * lots * comm
            cost_in = notional * lots + cost_entry
            if cost_in > cash:
                lots = int(cash * 0.95 / (notional * (1 + comm)))
                if lots <= 0:
                    continue
                cost_entry = notional * lots * comm
                cost_in = notional * lots + cost_entry
            if lots <= 0 or cost_in <= 0 or cost_in > cash:
                continue

            cash -= cost_in
            current_position = {
                'si': si, 'entry': c, 'entry_di': di,
                'lots': lots, 'sym': sym,
                'group': GROUP_MAP.get(sym, 'unknown'),
                'confirmed': confirmed,
                'score': best_score,
                'cost_entry': cost_entry,
            }

        # Close remaining position at end
        if current_position is not None:
            pos = current_position
            ae = min(end_di, ND) - 1
            cn = C[pos['si'], ae]
            if np.isnan(cn) or cn <= 0:
                cn = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = cn * mult * pos['lots']
            pnl = (cn - pos['entry']) * mult * pos['lots']
            invested = pos['entry'] * mult * pos['lots']
            pnl_pct = pnl / invested * 100 if invested > 0 else 0
            cost_exit = mkt_val * comm
            cash += mkt_val - cost_exit
            trades.append({
                'pnl_abs': pnl - pos['cost_entry'] - cost_exit,
                'pnl_pct': pnl_pct,
                'days': ae - pos['entry_di'],
                'di': ae,
                'year': dates[ae].year,
                'sym': pos['sym'],
                'group': pos['group'],
                'confirmed': pos['confirmed'],
                'score': pos['score'],
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
        pf = (sum(t['pnl_abs'] for t in trades if t['pnl_abs'] > 0) /
              max(abs(sum(t['pnl_abs'] for t in trades if t['pnl_abs'] < 0)), 1))

        first_di = min(t['di'] for t in trades)
        last_di = max(t['di'] for t in trades)
        days_total = (dates[last_di] - dates[first_di]).days if last_di > first_di else 365
        yr = max(days_total / 365.25, 0.01)
        ann = ((cash / CASH0) ** (1 / yr) - 1) * 100

        tp = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
        sharpe = np.mean(tp) / np.std(tp) * np.sqrt(252) if len(tp) > 1 and np.std(tp) > 0 else 0

        n_conf = sum(1 for t in trades if t['confirmed'])
        conf_pnl = sum(t['pnl_abs'] for t in trades if t['confirmed'])
        n_unconf = sum(1 for t in trades if not t['confirmed'])
        unconf_pnl = sum(t['pnl_abs'] for t in trades if not t['confirmed'])

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
            'name': config_name,
            'ann': round(ann, 1),
            'n': len(trades),
            'wr': round(wr, 1),
            'dd': round(max_dd, 1),
            'pf': round(pf, 2),
            'sharpe': round(sharpe, 2),
            'cash': round(cash, 0),
            'yearly': year_stats,
            'n_conf': n_conf,
            'n_unconf': n_unconf,
            'conf_pnl': conf_pnl,
            'unconf_pnl': unconf_pnl,
        }

    # ================================================================
    # BUILD CONFIGURATIONS
    # ================================================================
    print("\n[Configs] Building configurations...", flush=True)
    configs = []

    for ml in [2, 3, 5]:
        for mt in [0.001, 0.003, 0.005, 0.01]:
            for mode in ['pure_momentum', 'confirmed_only', 'anti_confirmed']:
                for comm in [0.0003, 0.0001]:
                    for pzt in [1.0, 1.5, 2.0]:
                        for plb in [10, 15, 20]:
                            # pure_momentum doesn't need pair params, avoid duplicates
                            if mode == 'pure_momentum' and (comm != 0.0003 or pzt != 1.0 or plb != 15):
                                continue
                            name = (f"ML{ml}_MT{mt*1000:.0f}_{mode}"
                                    f"_C{int(comm*10000)}_PZT{pzt:.1f}_PLB{plb}")
                            configs.append({
                                'mom_lb': ml,
                                'mom_thresh': mt,
                                'pair_z_thresh': pzt,
                                'pair_lb': plb,
                                'mode': mode,
                                'comm': comm,
                                'start_year': None,
                                'end_year': None,
                                'config_name': name,
                            })

    total_combos = len(configs)
    print(f"  {total_combos} configurations")

    # ================================================================
    # RUN SWEEP
    # ================================================================
    print(f"\n{'=' * 120}")
    print(f"  FULL-PERIOD PARAMETER SWEEP ({total_combos} configs)")
    print(f"{'=' * 120}")

    results = []
    t_sw = time.time()
    for ci, cfg in enumerate(configs):
        r = run_backtest(**cfg)
        if r is not None:
            results.append(r)
        if (ci + 1) % 50 == 0:
            print(f"  [{ci+1}/{total_combos}] {len(results)} with results ({time.time()-t_sw:.1f}s)", flush=True)

    results.sort(key=lambda x: -x['ann'])
    print(f"\n  Sweep complete: {len(results)}/{total_combos} configs ({time.time()-t_sw:.1f}s)", flush=True)

    # ================================================================
    # FULL PERIOD RESULTS
    # ================================================================
    print(f"\n{'=' * 120}")
    print(f"  --- Full Period Results ---")
    print(f"  Config | Ann% | WR% | DD% | PF | Sharpe | Trades | Conf | Mode")
    print(f"{'=' * 120}")

    if results:
        b = results[0]
        cp = b['n_conf'] / max(b['n'], 1) * 100
        print(f"  BEST: {b['name']} | {b['ann']:+.1f}% | {b['wr']:.1f}% | {b['dd']:.1f}% | "
              f"{b['pf']:.2f} | {b['sharpe']:.2f} | {b['n']} | {b['n_conf']}({cp:.0f}%)")

    print(f"\n  --- Top 10 by Annual Return ---")
    hdr = f"  {'#':>2s} | {'Config':65s} | {'Ann':>8s} | {'WR':>5s} | {'DD':>6s} | {'PF':>5s} | {'Sh':>6s} | {'N':>5s} | {'Conf':>6s}"
    print(hdr)
    print(f"  {'-' * 130}")
    for i, r in enumerate(results[:10]):
        cp = r['n_conf'] / max(r['n'], 1) * 100
        print(f"  {i+1:2d} | {r['name']:65s} | {r['ann']:+7.1f}% | {r['wr']:4.1f}% | "
              f"{r['dd']:5.1f}% | {r['pf']:4.2f} | {r['sharpe']:5.2f} | {r['n']:5d} | "
              f"{r['n_conf']:3d}({cp:.0f}%)")

    # ================================================================
    # MODE COMPARISON
    # ================================================================
    print(f"\n{'=' * 120}")
    print(f"  --- Mode Comparison ---")
    print(f"{'=' * 120}")

    for mode_label in ['pure_momentum', 'confirmed_only', 'anti_confirmed']:
        subset = [r for r in results if f'_{mode_label}_' in r['name']]
        if subset:
            best = max(subset, key=lambda x: x['ann'])
            avg = np.mean([r['ann'] for r in subset])
            n_pos = sum(1 for r in subset if r['ann'] > 0)
            print(f"\n  {mode_label} ({len(subset)} configs):")
            print(f"    Avg Ann: {avg:+.1f}%  Positive: {n_pos}/{len(subset)}")
            print(f"    Best: {best['name']}")
            print(f"    Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  DD={best['dd']:.1f}%  "
                  f"PF={best['pf']:.2f}  Sharpe={best['sharpe']:.2f}  N={best['n']}")
            cp = best['n_conf'] / max(best['n'], 1) * 100
            print(f"    Confirmed: {best['n_conf']}({cp:.0f}%)  Unconfirmed: {best['n_unconf']}")

    # ================================================================
    # KEY FINDINGS
    # ================================================================
    print(f"\n{'=' * 120}")
    print(f"  --- Key Findings ---")
    print(f"{'=' * 120}")

    pure_mom = [r for r in results if '_pure_momentum_' in r['name']]
    conf_only = [r for r in results if '_confirmed_only_' in r['name']]
    anti_conf = [r for r in results if '_anti_confirmed_' in r['name']]

    if pure_mom:
        best_pure = max(pure_mom, key=lambda x: x['ann'])
        print(f"  Pure momentum best: {best_pure['ann']:+.1f}% ({best_pure['name']})")
    if conf_only:
        best_conf = max(conf_only, key=lambda x: x['ann'])
        print(f"  Confirmed only best: {best_conf['ann']:+.1f}% ({best_conf['name']})")
    if anti_conf:
        best_anti = max(anti_conf, key=lambda x: x['ann'])
        print(f"  Anti-confirmed best: {best_anti['ann']:+.1f}% ({best_anti['name']})")

    # Confirmation rate analysis for top results
    if results:
        print(f"\n  Confirmation rate in top 20:")
        for i, r in enumerate(results[:20]):
            cp = r['n_conf'] / max(r['n'], 1) * 100
            print(f"    {i+1}. {r['name'][:65]}  Conf={r['n_conf']}({cp:.0f}%) "
                  f"ConfPnL={r['conf_pnl']:+.0f} UnconfPnL={r['unconf_pnl']:+.0f}")

    # ================================================================
    # MOMENTUM LOOKBACK COMPARISON
    # ================================================================
    print(f"\n{'=' * 120}")
    print(f"  --- Momentum Lookback Comparison ---")
    print(f"{'=' * 120}")
    for ml in [2, 3, 5]:
        subset = [r for r in results if f'ML{ml}_' in r['name']]
        if subset:
            best = max(subset, key=lambda x: x['ann'])
            avg = np.mean([r['ann'] for r in subset])
            print(f"  LB={ml}: N={len(subset)}  Avg={avg:+.1f}%  Best={best['ann']:+.1f}% ({best['name']})")

    # ================================================================
    # COMMISSION IMPACT
    # ================================================================
    print(f"\n{'=' * 120}")
    print(f"  --- Commission Impact ---")
    print(f"{'=' * 120}")
    for comm_val in [0.0003, 0.0001]:
        tag = f'_C{int(comm_val*10000)}_'
        subset = [r for r in results if tag in r['name']]
        if subset:
            best = max(subset, key=lambda x: x['ann'])
            avg = np.mean([r['ann'] for r in subset])
            print(f"  COMM={comm_val:.4f}: N={len(subset)}  Avg={avg:+.1f}%  Best={best['ann']:+.1f}%")

    # ================================================================
    # WALK-FORWARD TOP 5
    # ================================================================
    top5 = results[:5]
    print(f"\n{'=' * 120}")
    print(f"  --- Walk-Forward (Top 5 configs) ---")
    print(f"  Config | WF2020 | WF2021 | WF2022 | WF2023 | WF2024 | WF2025 | Avg WF")
    print(f"{'=' * 120}")

    wf_all = []
    wf_by_config = {}
    for rank, cfg in enumerate(top5):
        cn = cfg['name']
        matching = [c for c in configs if c['config_name'] == cn]
        if not matching:
            continue
        bc = matching[0]
        print(f"\n  [{rank+1}] {cn}  (full-period Ann={cfg['ann']:+.1f}%)")

        for train_end, test_year in WF_WINDOWS:
            if test_year not in year_start_di:
                continue
            wc = dict(bc)
            wc['start_year'] = test_year
            wc['end_year'] = test_year
            wc['config_name'] = f"WF_{test_year}_{cn}"
            r = run_backtest(**wc)
            if r is not None:
                wf_all.append((cn, test_year, r))
                if cn not in wf_by_config:
                    wf_by_config[cn] = []
                wf_by_config[cn].append((test_year, r))
                print(f"    {test_year}: Ann={r['ann']:+7.1f}%  WR={r['wr']:5.1f}%  "
                      f"N={r['n']:4d}  DD={r['dd']:5.1f}%  PF={r['pf']:4.2f}")
            else:
                print(f"    {test_year}: insufficient trades")

    # WF Summary Table
    if wf_by_config:
        print(f"\n{'=' * 120}")
        print(f"  Walk-Forward Results Table")
        print(f"{'=' * 120}")
        wf_summary = []
        for cn, wr_list in wf_by_config.items():
            anns = [r['ann'] for _, r in wr_list]
            aby = {ty: r['ann'] for ty, r in wr_list}
            wf_summary.append({
                'name': cn,
                'avg_ann': np.mean(anns),
                'ann_2020': aby.get(2020, float('nan')),
                'ann_2021': aby.get(2021, float('nan')),
                'ann_2022': aby.get(2022, float('nan')),
                'ann_2023': aby.get(2023, float('nan')),
                'ann_2024': aby.get(2024, float('nan')),
                'ann_2025': aby.get(2025, float('nan')),
                'avg_wr': np.mean([r['wr'] for _, r in wr_list]),
                'avg_dd': np.mean([r['dd'] for _, r in wr_list]),
                'n_positive': sum(1 for a in anns if a > 0),
                'n_windows': len(wr_list),
            })
        wf_summary.sort(key=lambda x: -x['avg_ann'])

        hdr = (f"  {'#':>2s} | {'Config':60s} | {'Avg WF':>8s} | "
               f"{'WF2020':>8s} | {'WF2021':>8s} | {'WF2022':>8s} | "
               f"{'WF2023':>8s} | {'WF2024':>8s} | {'WF2025':>8s} | {'Pos':>4s}")
        print(hdr)
        print(f"  {'-' * 170}")
        for i, w in enumerate(wf_summary):
            def fmt_ann(v):
                return f"{v:+7.1f}%" if not np.isnan(v) else "  N/A  "
            print(f"  {i+1:2d} | {w['name']:60s} | {w['avg_ann']:+7.1f}% | "
                  f"{fmt_ann(w['ann_2020'])} | {fmt_ann(w['ann_2021'])} | "
                  f"{fmt_ann(w['ann_2022'])} | {fmt_ann(w['ann_2023'])} | "
                  f"{fmt_ann(w['ann_2024'])} | {fmt_ann(w['ann_2025'])} | "
                  f"{w['n_positive']}/{w['n_windows']}")

    # ================================================================
    # YEARLY BREAKDOWN TOP 3
    # ================================================================
    if len(results) >= 2:
        print(f"\n{'=' * 120}")
        print(f"  Yearly Breakdown for Top 3")
        print(f"{'=' * 120}")
        for idx, r in enumerate(results[:3]):
            print(f"\n  #{idx+1}: {r['name']}")
            print(f"    Ann={r['ann']:+.1f}%  WR={r['wr']:.1f}%  DD={r['dd']:.1f}%  "
                  f"PF={r['pf']:.2f}  Sharpe={r['sharpe']:.2f}  N={r['n']}")
            print(f"    {'Year':>6s} | {'N':>5s} | {'WR':>5s} | {'PnL%':>8s} | {'PnL Abs':>12s}")
            print(f"    {'-' * 50}")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / max(ys['n'], 1) * 100
                print(f"    {y:6d} | {ys['n']:5d} | {wr_y:4.1f}% | {ys['pnl']:+7.1f}% | "
                      f"{ys['pnl_abs_sum']:+11.0f}")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print(f"\n{'=' * 120}")
    print(f"  --- Final Summary ---")
    print(f"{'=' * 120}")

    if results:
        b = results[0]
        print(f"\n  Best full-period: {b['name']}")
        print(f"    Ann={b['ann']:+.1f}%  WR={b['wr']:.1f}%  DD={b['dd']:.1f}%  "
              f"PF={b['pf']:.2f}  Sharpe={b['sharpe']:.2f}  N={b['n']}")
        cp = b['n_conf'] / max(b['n'], 1) * 100
        print(f"    Confirmed: {b['n_conf']}({cp:.0f}%) PnL={b['conf_pnl']:+.0f}  "
              f"Unconfirmed: {b['n_unconf']} PnL={b['unconf_pnl']:+.0f}")

    if pure_mom:
        print(f"\n  Pure momentum best: {max(pure_mom, key=lambda x: x['ann'])['ann']:+.1f}%")
    if conf_only:
        print(f"  Confirmed only best: {max(conf_only, key=lambda x: x['ann'])['ann']:+.1f}%")
    if anti_conf:
        print(f"  Anti-confirmed best: {max(anti_conf, key=lambda x: x['ann'])['ann']:+.1f}%")

    if wf_by_config:
        bw = max(wf_by_config.items(), key=lambda x: np.mean([r['ann'] for _, r in x[1]]))
        avg_wf = np.mean([r['ann'] for _, r in bw[1]])
        n_pos = sum(1 for _, r in bw[1] if r['ann'] > 0)
        print(f"\n  Best WF config: {bw[0]}")
        print(f"    Avg WF Ann={avg_wf:+.1f}%  Positive windows: {n_pos}/{len(bw[1])}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
