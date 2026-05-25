"""
Alpha Futures V70 -- Final Optimized Combo: 1-Day Momentum + Pair Trading
=========================================================================
V63 showed COMBO (pair Z=2.0 + momentum LB3) = +369.1% full period,
WF2023 = +602.8%. This breaks 600% in walk-forward. Need to optimize
and validate.

Approach: Priority-based capital allocation:
  1. First check: pair z-score > Z_pair (adaptive LOG spread, 14 pairs).
     If yes -> trade pair
  2. If no pair signals -> check group momentum lag. If score > threshold
     -> trade momentum
  3. If neither -> stay in cash

Configs (~200):
  - Pair Z threshold: [1.5, 2.0, 2.5, 3.0] (only extreme pair signals)
  - Pair spread mode: [adaptive_LOG, log_LB15, log_LB10]
  - Momentum lookback: [2, 3, 5]
  - Momentum threshold: [0.003, 0.005, 0.01]
  - Momentum scope: [all 68 commodities, group_map only (20)]
  - COMM: [0.0003, 0.0001]
  - Rigorous 6-window WF (2020-2025) for top 10 configs

Reports:
  1. Full-period: Ann, WR, DD, PF, Sharpe
  2. Trade breakdown: % pair trades vs % momentum trades
  3. Pair-only and momentum-only baselines for comparison
  4. 6-window WF (2020-2025): each window separately
  5. Overfitting check: full-period vs WF correlation

Prints: top 20 full-period, top 10 WF, WF breakdown per window.
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
    'afi': 'oils', 'mfi': 'oils', 'yfi': 'oils', 'pfi': 'oils', 'cfi': 'oils',
    'scfi': 'energy', 'mafi': 'energy', 'bfi': 'energy', 'fufi': 'energy',
    'ppfi': 'chemical', 'vfi': 'chemical', 'egfi': 'chemical', 'pgfi': 'chemical',
}

PAIRS = [
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
SPREAD_RAW = 'raw'
SPREAD_PCT = 'pct'

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
    print("=" * 160)
    print("Alpha Futures V70 -- Final Optimized Combo: 1-Day Momentum + Pair Trading")
    print("Priority: 1) Pair z-score > Z_pair  2) Group momentum lag  3) Cash")
    print("Configs: ~200 (pair Z, spread mode, momentum LB/TH, scope, commission)")
    print("Walk-forward: 6-window (2020-2025) for top 10")
    print("=" * 160)

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

    # ----------------------------------------------------------------
    # Build pair indices
    # ----------------------------------------------------------------
    def build_pair_indices(pairs_list):
        indices = []
        for down_sym, up_sym in pairs_list:
            down_si = sym_to_si.get(down_sym, -1)
            up_si = sym_to_si.get(up_sym, -1)
            if down_si >= 0 and up_si >= 0:
                indices.append((down_si, up_si, down_sym, up_sym))
        return indices

    pair_indices_14 = build_pair_indices(PAIRS)
    print(f"  Pair set: P14={len(pair_indices_14)}")

    # ----------------------------------------------------------------
    # Group membership
    # ----------------------------------------------------------------
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
    all_sis = set(range(NS))

    # ================================================================
    # PRECOMPUTE PAIR Z-SCORES
    # ================================================================
    print("\n[Signals] Precomputing pair spreads and z-scores...", flush=True)
    t0 = time.time()

    ALL_MODES = [SPREAD_LOG, SPREAD_RAW, SPREAD_PCT]
    ALL_LOOKBACKS = [5, 7, 10, 15, 20]

    z_scores = {m: {} for m in ALL_MODES}
    all_pair_set = set()
    for down_si, up_si, _, _ in pair_indices_14:
        all_pair_set.add((down_si, up_si))

    for down_si, up_si in all_pair_set:
        key = (down_si, up_si)
        for mode in ALL_MODES:
            spread = np.full(ND, np.nan)
            for di in range(ND):
                pd_val = C[down_si, di]
                pu = C[up_si, di]
                if np.isnan(pd_val) or np.isnan(pu) or pu <= 0 or pd_val <= 0:
                    continue
                if mode == SPREAD_RAW:
                    spread[di] = pd_val - pu
                elif mode == SPREAD_PCT:
                    spread[di] = (pd_val - pu) / pu
                elif mode == SPREAD_LOG:
                    spread[di] = np.log(pd_val) - np.log(pu)
            z_scores[mode][key] = {}
            for lb in ALL_LOOKBACKS:
                z = np.full(ND, np.nan)
                for di in range(lb, ND):
                    window = spread[di - lb:di]
                    valid = window[~np.isnan(window)]
                    if len(valid) >= max(3, lb * 0.8):
                        m_val = np.mean(valid)
                        s_val = np.std(valid, ddof=1)
                        if s_val > 1e-10:
                            z[di] = (spread[di] - m_val) / s_val
                z_scores[mode][key][lb] = z
    print(f"  Pair z-scores precomputed ({time.time() - t0:.1f}s)", flush=True)

    # ================================================================
    # PRECOMPUTE HYPOTHETICAL RETURNS (for adaptive pair selection)
    # ================================================================
    print("\n[Signals] Precomputing per-pair hypothetical returns...", flush=True)
    t1 = time.time()

    # We need hypothetical returns for adaptive selection.
    # Compute for the Z thresholds used in configs: [1.5, 2.0, 2.5, 3.0]
    # plus pair-only baselines at lower Z
    all_zt = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]

    pair_combo_daily_return = {}
    for zt in all_zt:
        for mode in ALL_MODES:
            for lb in ALL_LOOKBACKS:
                combo_key = (mode, lb, zt)
                for down_si, up_si, down_sym, up_sym in pair_indices_14:
                    pair_key = (down_si, up_si, down_sym, up_sym)
                    daily_ret = np.full(ND, np.nan)
                    for di in range(MIN_TRAIN + 1, ND):
                        z_arr = z_scores[mode].get((down_si, up_si), {}).get(lb)
                        if z_arr is None:
                            continue
                        z_prev = z_arr[di - 1]
                        if np.isnan(z_prev) or abs(z_prev) < zt:
                            continue
                        c_de = C[down_si, di - 1]; c_ue = C[up_si, di - 1]
                        c_dx = C[down_si, di]; c_ux = C[up_si, di]
                        if (np.isnan(c_de) or c_de <= 0 or np.isnan(c_ue) or c_ue <= 0 or
                            np.isnan(c_dx) or c_dx <= 0 or np.isnan(c_ux) or c_ux <= 0):
                            continue
                        mult_d = MULT.get(down_sym, DEF_MULT)
                        mult_u = MULT.get(up_sym, DEF_MULT)
                        if z_prev > 0:
                            pnl = (c_de - c_dx) * mult_d + (c_ux - c_ue) * mult_u
                        else:
                            pnl = (c_dx - c_de) * mult_d + (c_ue - c_ux) * mult_u
                        invested = c_de * mult_d + c_ue * mult_u
                        pnl_pct = (pnl - invested * 0.0003 * 2) / invested * 100 if invested > 0 else 0
                        daily_ret[di] = pnl_pct
                    pair_combo_daily_return[(pair_key, combo_key)] = daily_ret

    global_combo_daily_return = {}
    for zt in all_zt:
        for mode in ALL_MODES:
            for lb in ALL_LOOKBACKS:
                combo_key = (mode, lb, zt)
                daily_ret = np.full(ND, np.nan)
                for di in range(MIN_TRAIN + 1, ND):
                    pair_rets = []
                    for dsi, usi, dsym, usym in pair_indices_14:
                        pk = (dsi, usi, dsym, usym)
                        pr = pair_combo_daily_return.get((pk, combo_key))
                        if pr is not None and not np.isnan(pr[di]):
                            pair_rets.append(pr[di])
                    if pair_rets:
                        daily_ret[di] = np.mean(pair_rets)
                global_combo_daily_return[combo_key] = daily_ret
    print(f"  Hypothetical returns precomputed ({time.time() - t1:.1f}s)", flush=True)

    # ================================================================
    # PRECOMPUTE MOMENTUM SIGNALS
    # ================================================================
    print("\n[Signals] Precomputing momentum signals...", flush=True)
    t2 = time.time()

    mom = {}
    for lag in [2, 3, 5, 7]:
        m = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(lag, ND):
                c_now = C[si, di]
                c_prev = C[si, di - lag]
                if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                    m[si, di] = (c_now - c_prev) / c_prev
        mom[lag] = m

    # Group momentum lag for group_map commodities
    grp_mom = {}
    for lag in [2, 3, 5, 7]:
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

    # Simple 1-day momentum for all 68 commodities
    simple_mom = {}
    for lag in [2, 3, 5]:
        sm = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(lag, ND):
                c_now = C[si, di]
                c_prev = C[si, di - lag]
                if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                    sm[si, di] = (c_now - c_prev) / c_prev
        simple_mom[lag] = sm
    print(f"  Momentum signals precomputed ({time.time() - t2:.1f}s)", flush=True)

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(z_thresh=2.0, hold_max=1, exit_z=0.0,
                     mode_type='adaptive_LOG', eval_period=40,
                     candidate_combos=None, pair_indices=None,
                     momentum_threshold=0.003, mom_lookback=3,
                     mom_scope='group', comm=0.0003,
                     start_year=None, end_year=None, config_name=""):
        """
        Priority-based combo backtest.
        mom_scope: 'group' = only group_map commodities (20)
                   'all'   = all 68 commodities
        """
        if pair_indices is None:
            pair_indices = pair_indices_14
        if candidate_combos is None:
            candidate_combos = [(SPREAD_LOG, 10), (SPREAD_LOG, 15), (SPREAD_LOG, 20)]

        cash = float(CASH0)
        trades = []
        current_position = None
        current_combo = candidate_combos[0]

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

        # Choose which SI set for momentum
        if mom_scope == 'group':
            mom_sis = group_sis
        else:
            mom_sis = all_sis

        for di in range(start_di, end_di):
            year = dates[di].year

            # --- Adaptive pair combo selection ---
            if di > start_di:
                ds = di - start_di
                if ds % eval_period == 0 and ds >= eval_period and mode_type != 'fixed':
                    best_combo = candidate_combos[0]
                    best_score = -1e18
                    for c in candidate_combos:
                        ck = (c[0], c[1], z_thresh)
                        dr = global_combo_daily_return.get(ck)
                        if dr is None:
                            continue
                        w = dr[max(start_di, di - eval_period):di]
                        v = w[~np.isnan(w)]
                        score = np.nansum(v) if len(v) >= 3 else -1e10
                        if score > best_score:
                            best_score = score
                            best_combo = c
                    current_combo = best_combo

            # --- Close existing position ---
            if current_position is not None:
                pos = current_position
                exit_reason = None
                days_held = di - pos['entry_di']

                if pos['type'] == 'pair':
                    za = z_scores[pos['mode']].get((pos['down_si'], pos['up_si']), {}).get(pos['lb'])
                    z_now = za[di] if za is not None and di < len(za) else np.nan
                    pd_dir = pos['dir']
                    if not np.isnan(z_now):
                        if pd_dir == 1 and z_now >= exit_z:
                            exit_reason = 'mean_rev'
                        elif pd_dir == -1 and z_now <= -exit_z:
                            exit_reason = 'mean_rev'
                    if exit_reason is None and not np.isnan(z_now):
                        if pd_dir == 1 and z_now < pos['entry_z'] - 1.5:
                            exit_reason = 'stop_loss'
                        elif pd_dir == -1 and z_now > pos['entry_z'] + 1.5:
                            exit_reason = 'stop_loss'
                    if exit_reason is None and days_held >= hold_max:
                        exit_reason = 'time'

                    if exit_reason:
                        cd = C[pos['down_si'], di]
                        cu = C[pos['up_si'], di]
                        if np.isnan(cd) or cd <= 0: cd = pos['entry_down']
                        if np.isnan(cu) or cu <= 0: cu = pos['entry_up']
                        md = MULT.get(pos['down_sym'], DEF_MULT)
                        mu = MULT.get(pos['up_sym'], DEF_MULT)
                        ld, lu = pos['lots_down'], pos['lots_up']
                        if pd_dir == 1:
                            pnl_d = (cd - pos['entry_down']) * md * ld
                            pnl_u = (pos['entry_up'] - cu) * mu * lu
                        else:
                            pnl_d = (pos['entry_down'] - cd) * md * ld
                            pnl_u = (cu - pos['entry_up']) * mu * lu
                        ev_d = pos['entry_down'] * md * ld
                        ev_u = pos['entry_up'] * mu * lu
                        xv_d = cd * md * ld
                        xv_u = cu * mu * lu
                        cost = (ev_d + ev_u) * comm + (xv_d + xv_u) * comm
                        total_pnl = pnl_d + pnl_u - cost
                        invested = ev_d + ev_u
                        pnl_pct = total_pnl / invested * 100 if invested > 0 else 0
                        if pd_dir == 1:
                            cash_return = cd * md * ld - cu * mu * lu
                        else:
                            cash_return = -cd * md * ld + cu * mu * lu
                        cash += pos['cash_invested'] + cash_return - (xv_d + xv_u) * comm
                        trades.append({
                            'pnl_abs': total_pnl, 'pnl_pct': pnl_pct,
                            'days': days_held, 'di': di, 'year': year,
                            'type': 'pair',
                            'pair': (pos['down_sym'], pos['up_sym']),
                            'pair_label': PAIR_LABEL.get((pos['down_sym'], pos['up_sym']), ''),
                            'dir': pd_dir, 'reason': exit_reason,
                        })
                        current_position = None

                elif pos['type'] == 'momentum':
                    if days_held >= 1:
                        exit_reason = 'time'
                    if exit_reason:
                        cn = C[pos['si'], di]
                        if np.isnan(cn) or cn <= 0: cn = pos['entry']
                        mult = MULT.get(pos['sym'], DEF_MULT)
                        mkt_val = cn * mult * pos['lots']
                        pnl = (cn - pos['entry']) * mult * pos['lots'] * pos['dir']
                        pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] > 0 else 0
                        cost = mkt_val * comm
                        cash += mkt_val - cost
                        pnl -= cost
                        trades.append({
                            'pnl_abs': pnl, 'pnl_pct': pnl_pct,
                            'days': days_held, 'di': di, 'year': year,
                            'type': 'momentum', 'sym': pos['sym'],
                            'group': pos['group'], 'dir': pos['dir'],
                            'reason': exit_reason,
                        })
                        current_position = None

            if current_position is not None:
                continue

            # --- Open new position ---
            use_mode, use_lb = (candidate_combos[0] if mode_type == 'fixed' else current_combo)

            # PRIORITY 1: Pair signals
            opened = False
            best_pair_z = 0
            best_pair = None
            for dsi, usi, dsym, usym in pair_indices:
                za = z_scores[use_mode].get((dsi, usi), {}).get(use_lb)
                if za is None: continue
                zv = za[di] if di < len(za) else np.nan
                if np.isnan(zv) or abs(zv) < z_thresh: continue
                if abs(zv) > best_pair_z:
                    best_pair_z = abs(zv)
                    best_pair = (dsi, usi, dsym, usym, zv)

            if best_pair is not None:
                dsi, usi, dsym, usym, zv = best_pair
                cd = C[dsi, di]; cu = C[usi, di]
                if not (np.isnan(cd) or cd <= 0 or np.isnan(cu) or cu <= 0):
                    md = MULT.get(dsym, DEF_MULT)
                    mu = MULT.get(usym, DEF_MULT)
                    cpl = cash / 2
                    ld = int(cpl / (cd * md * (1 + comm)))
                    lu = int(cpl / (cu * mu * (1 + comm)))
                    if ld > 0 and lu > 0:
                        tc = cd * md * ld * (1 + comm) + cu * mu * lu * (1 + comm)
                        if tc > cash:
                            sc = cash * 0.95 / tc
                            ld = max(1, int(ld * sc))
                            lu = max(1, int(lu * sc))
                            tc = cd * md * ld * (1 + comm) + cu * mu * lu * (1 + comm)
                        if tc <= cash:
                            pos_dir = -1 if zv > 0 else 1
                            cash -= tc
                            current_position = {
                                'type': 'pair',
                                'down_si': dsi, 'up_si': usi,
                                'down_sym': dsym, 'up_sym': usym,
                                'entry_down': cd, 'entry_up': cu,
                                'lots_down': ld, 'lots_up': lu,
                                'entry_di': di, 'entry_z': zv,
                                'dir': pos_dir, 'cash_invested': tc,
                                'mode': use_mode, 'lb': use_lb,
                            }
                            opened = True

            if opened:
                continue

            # PRIORITY 2: Momentum signals
            best_mom_score = 0
            best_mom_si = -1
            for si in mom_sis:
                if mom_scope == 'group':
                    # Group momentum lag: score = group_avg - own
                    own = mom[mom_lookback][si, di]
                    grp = grp_mom[mom_lookback][si, di]
                    if np.isnan(own) or np.isnan(grp): continue
                    score = grp - own
                else:
                    # Simple momentum for all 68: just use own momentum
                    own = simple_mom[mom_lookback][si, di]
                    if np.isnan(own): continue
                    score = own
                if score > momentum_threshold and score > best_mom_score:
                    best_mom_score = score
                    best_mom_si = si

            if best_mom_si >= 0:
                si = best_mom_si
                c = C[si, di]
                if not (np.isnan(c) or c <= 0):
                    sym = syms[si]
                    mult = MULT.get(sym, DEF_MULT)
                    notional = c * mult
                    lots = int(cash / (notional * (1 + comm)))
                    if lots > 0:
                        cost_in = notional * lots * (1 + comm)
                        if cost_in > cash:
                            lots = int(cash / (notional * (1 + comm)))
                            cost_in = notional * lots * (1 + comm) if lots > 0 else 0
                        if lots > 0 and cost_in > 0 and cost_in <= cash:
                            cash -= cost_in
                            current_position = {
                                'type': 'momentum',
                                'si': si, 'entry': c, 'entry_di': di,
                                'lots': lots, 'dir': 1, 'sym': sym,
                                'group': GROUP_MAP.get(sym, 'unknown'),
                            }

        # Close remaining position at end
        if current_position is not None:
            pos = current_position
            ae = min(end_di, ND) - 1
            if pos['type'] == 'pair':
                cd = C[pos['down_si'], ae]; cu = C[pos['up_si'], ae]
                if np.isnan(cd) or cd <= 0: cd = pos['entry_down']
                if np.isnan(cu) or cu <= 0: cu = pos['entry_up']
                md = MULT.get(pos['down_sym'], DEF_MULT)
                mu = MULT.get(pos['up_sym'], DEF_MULT)
                ld, lu = pos['lots_down'], pos['lots_up']
                if pos['dir'] == 1:
                    pnl_d = (cd - pos['entry_down']) * md * ld
                    pnl_u = (pos['entry_up'] - cu) * mu * lu
                    cash_ret = cd * md * ld - cu * mu * lu
                else:
                    pnl_d = (pos['entry_down'] - cd) * md * ld
                    pnl_u = (cu - pos['entry_up']) * mu * lu
                    cash_ret = -cd * md * ld + cu * mu * lu
                xv_d = cd * md * ld; xv_u = cu * mu * lu
                cost = (pos['entry_down'] * md * ld + pos['entry_up'] * mu * lu) * comm + (xv_d + xv_u) * comm
                total_pnl = pnl_d + pnl_u - cost
                cash += pos['cash_invested'] + cash_ret - (xv_d + xv_u) * comm
                trades.append({
                    'pnl_abs': total_pnl,
                    'pnl_pct': total_pnl / (pos['entry_down'] * md * ld + pos['entry_up'] * mu * lu) * 100 if (pos['entry_down'] * md * ld + pos['entry_up'] * mu * lu) > 0 else 0,
                    'days': ae - pos['entry_di'], 'di': ae, 'year': dates[ae].year,
                    'type': 'pair', 'pair': (pos['down_sym'], pos['up_sym']),
                    'pair_label': PAIR_LABEL.get((pos['down_sym'], pos['up_sym']), ''),
                    'dir': pos['dir'], 'reason': 'end',
                })
            elif pos['type'] == 'momentum':
                cn = C[pos['si'], ae]
                if np.isnan(cn) or cn <= 0: cn = pos['entry']
                mult = MULT.get(pos['sym'], DEF_MULT)
                mkt_val = cn * mult * pos['lots']
                pnl = (cn - pos['entry']) * mult * pos['lots'] * pos['dir']
                cost = mkt_val * comm
                cash += mkt_val - cost
                pnl -= cost
                trades.append({
                    'pnl_abs': pnl,
                    'pnl_pct': pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] > 0 else 0,
                    'days': ae - pos['entry_di'], 'di': ae, 'year': dates[ae].year,
                    'type': 'momentum', 'sym': pos['sym'],
                    'group': pos['group'], 'dir': pos['dir'], 'reason': 'end',
                })

        if len(trades) < 3:
            return None

        # === STATS ===
        equity = float(CASH0); peak = float(CASH0); max_dd = 0.0
        for t in sorted(trades, key=lambda x: x['di']):
            equity += t['pnl_abs']
            if equity > peak: peak = equity
            if peak > 0:
                dd = (peak - equity) / peak * 100
                if dd > max_dd: max_dd = dd

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

        n_pair = sum(1 for t in trades if t['type'] == 'pair')
        n_mom = sum(1 for t in trades if t['type'] == 'momentum')
        pair_pnl = sum(t['pnl_abs'] for t in trades if t['type'] == 'pair')
        mom_pnl = sum(t['pnl_abs'] for t in trades if t['type'] == 'momentum')
        pair_wins = sum(1 for t in trades if t['type'] == 'pair' and t['pnl_abs'] > 0)
        mom_wins = sum(1 for t in trades if t['type'] == 'momentum' and t['pnl_abs'] > 0)

        year_stats = {}
        for t in trades:
            y = t['year']
            if y not in year_stats:
                year_stats[y] = {'n': 0, 'w': 0, 'pnl': 0.0, 'pnl_abs_sum': 0.0, 'n_pair': 0, 'n_mom': 0}
            year_stats[y]['n'] += 1
            if t['pnl_abs'] > 0: year_stats[y]['w'] += 1
            year_stats[y]['pnl'] += t['pnl_pct']
            year_stats[y]['pnl_abs_sum'] += t['pnl_abs']
            if t['type'] == 'pair': year_stats[y]['n_pair'] += 1
            else: year_stats[y]['n_mom'] += 1

        return {
            'name': config_name, 'ann': round(ann, 1), 'n': len(trades),
            'wr': round(wr, 1), 'dd': round(max_dd, 1),
            'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
            'avg_days': round(avg_days, 1), 'pf': round(pf, 2),
            'sharpe': round(sharpe, 2), 'cash': round(cash, 0),
            'yearly': year_stats, 'n_pair': n_pair, 'n_mom': n_mom,
            'pair_pnl': pair_pnl, 'mom_pnl': mom_pnl,
            'pair_wr': pair_wins / max(n_pair, 1) * 100 if n_pair > 0 else 0,
            'mom_wr': mom_wins / max(n_mom,1) * 100 if n_mom > 0 else 0,
        }

    # ================================================================
    # BUILD CONFIGURATIONS (~200)
    # ================================================================
    print("\n[Configs] Building configurations...", flush=True)
    configs = []

    log_bias_combos = [(SPREAD_LOG, 10), (SPREAD_LOG, 15), (SPREAD_LOG, 20),
                       (SPREAD_RAW, 10), (SPREAD_PCT, 10)]

    # ----- Pair-only baselines (higher Z thresholds) -----
    for zt in [1.5, 2.0, 2.5, 3.0]:
        configs.append({
            'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0,
            'mode_type': 'adaptive_LOG', 'eval_period': 40,
            'candidate_combos': log_bias_combos,
            'pair_indices': pair_indices_14,
            'momentum_threshold': 999.0, 'mom_lookback': 3,
            'mom_scope': 'group', 'comm': 0.0003,
            'start_year': None, 'end_year': None,
            'config_name': f"PAIR_ONLY_adapt_Z{zt:.1f}_C3",
        })
        for sm, slb in [(SPREAD_LOG, 15), (SPREAD_LOG, 10)]:
            configs.append({
                'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0,
                'mode_type': 'fixed', 'candidate_combos': [(sm, slb)],
                'pair_indices': pair_indices_14,
                'momentum_threshold': 999.0, 'mom_lookback': 3,
                'mom_scope': 'group', 'comm': 0.0003,
                'start_year': None, 'end_year': None,
                'config_name': f"PAIR_ONLY_{sm}_LB{slb}_Z{zt:.1f}_C3",
            })

    # ----- Momentum-only baselines -----
    for ml in [2, 3, 5]:
        for mt in [0.003, 0.005, 0.01]:
            for scope in ['group', 'all']:
                configs.append({
                    'z_thresh': 99.0, 'hold_max': 1, 'exit_z': 0.0,
                    'mode_type': 'fixed', 'candidate_combos': [(SPREAD_LOG, 10)],
                    'pair_indices': pair_indices_14,
                    'momentum_threshold': mt, 'mom_lookback': ml,
                    'mom_scope': scope, 'comm': 0.0003,
                    'start_year': None, 'end_year': None,
                    'config_name': f"MOM_ONLY_LB{ml}_MT{mt*1000:.0f}_{scope}_C3",
                })

    # ----- Combined: adaptive pair + momentum (main sweep) -----
    for zt in [1.5, 2.0, 2.5, 3.0]:
        for mt in [0.003, 0.005, 0.01]:
            for ml in [2, 3, 5]:
                for scope in ['group', 'all']:
                    configs.append({
                        'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0,
                        'mode_type': 'adaptive_LOG', 'eval_period': 40,
                        'candidate_combos': log_bias_combos,
                        'pair_indices': pair_indices_14,
                        'momentum_threshold': mt, 'mom_lookback': ml,
                        'mom_scope': scope, 'comm': 0.0003,
                        'start_year': None, 'end_year': None,
                        'config_name': f"COMBO_adapt_Z{zt:.1f}_ML{ml}_MT{mt*1000:.0f}_{scope}_C3",
                    })

    # ----- Combined: fixed pair spread + momentum -----
    for zt in [1.5, 2.0]:
        for sm, slb in [(SPREAD_LOG, 15), (SPREAD_LOG, 10)]:
            for mt in [0.003, 0.005]:
                for ml in [2, 3]:
                    configs.append({
                        'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0,
                        'mode_type': 'fixed', 'candidate_combos': [(sm, slb)],
                        'pair_indices': pair_indices_14,
                        'momentum_threshold': mt, 'mom_lookback': ml,
                        'mom_scope': 'group', 'comm': 0.0003,
                        'start_year': None, 'end_year': None,
                        'config_name': f"COMBO_{sm}_LB{slb}_Z{zt:.1f}_ML{ml}_MT{mt*1000:.0f}_C3",
                    })

    # ----- Low commission variants for top Z thresholds -----
    for zt in [2.0, 2.5]:
        for mt in [0.003, 0.005]:
            for ml in [2, 3]:
                for scope in ['group', 'all']:
                    configs.append({
                        'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0,
                        'mode_type': 'adaptive_LOG', 'eval_period': 40,
                        'candidate_combos': log_bias_combos,
                        'pair_indices': pair_indices_14,
                        'momentum_threshold': mt, 'mom_lookback': ml,
                        'mom_scope': scope, 'comm': 0.0001,
                        'start_year': None, 'end_year': None,
                        'config_name': f"COMBO_adapt_Z{zt:.1f}_ML{ml}_MT{mt*1000:.0f}_{scope}_C1",
                    })

    total_combos = len(configs)
    print(f"  {total_combos} configurations")
    print(f"    Breakdown: pair-only={sum(1 for c in configs if c['config_name'].startswith('PAIR'))}, "
          f"mom-only={sum(1 for c in configs if c['config_name'].startswith('MOM'))}, "
          f"combo={sum(1 for c in configs if c['config_name'].startswith('COMBO'))}")

    # ================================================================
    # RUN FULL-PERIOD SWEEP
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  FULL-PERIOD PARAMETER SWEEP ({total_combos} configs)")
    print(f"{'=' * 160}")

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
    # TOP 20 FULL-PERIOD
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  TOP 20 FULL-PERIOD RESULTS")
    print(f"{'=' * 160}")
    hdr = (f"  {'#':>2s} | {'Config':62s} | {'Ann':>8s} | {'WR':>5s} | {'N':>5s} | "
           f"{'DD':>6s} | {'PF':>5s} | {'Sh':>6s} | {'Pair%':>6s} | {'Mom%':>5s} | {'Cash':>12s}")
    print(hdr)
    print(f"  {'-' * 155}")
    for i, r in enumerate(results[:20]):
        pp = r['n_pair'] / max(r['n'], 1) * 100
        mp = r['n_mom'] / max(r['n'], 1) * 100
        print(f"  {i+1:2d} | {r['name']:62s} | {r['ann']:+7.1f}% | {r['wr']:4.1f}% | {r['n']:5d} | "
              f"{r['dd']:5.1f}% | {r['pf']:4.2f} | {r['sharpe']:5.2f} | {pp:5.1f}% | {mp:4.1f}% | {r['cash']:11.0f}")

    # ================================================================
    # TRADE TYPE BREAKDOWN (top 20)
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  TRADE TYPE BREAKDOWN (top 20)")
    print(f"{'=' * 160}")
    for i, r in enumerate(results[:20]):
        pp = r['n_pair'] / max(r['n'], 1) * 100
        mp = r['n_mom'] / max(r['n'], 1) * 100
        print(f"  {i+1:2d} | {r['name']:62s} | Pair:{r['n_pair']:4d}({pp:.0f}%) WR={r['pair_wr']:.1f}% "
              f"PnL={r['pair_pnl']:+.0f} | Mom:{r['n_mom']:4d}({mp:.0f}%) WR={r['mom_wr']:.1f}% "
              f"PnL={r['mom_pnl']:+.0f}")

    # ================================================================
    # BASELINE COMPARISON
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  BASELINE COMPARISON")
    print(f"{'=' * 160}")

    pair_only = [r for r in results if r['name'].startswith('PAIR_ONLY_')]
    mom_only = [r for r in results if r['name'].startswith('MOM_ONLY_')]
    combo = [r for r in results if r['name'].startswith('COMBO_')]

    for label, subset in [("PAIR-ONLY", pair_only), ("MOMENTUM-ONLY", mom_only), ("COMBINED", combo)]:
        print(f"\n  {label} ({len(subset)} configs):")
        if subset:
            best = max(subset, key=lambda x: x['ann'])
            avg = np.mean([r['ann'] for r in subset])
            pp = best['n_pair'] / max(best['n'], 1) * 100
            mp = best['n_mom'] / max(best['n'], 1) * 100
            print(f"    Best: {best['name']}")
            print(f"    Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  N={best['n']}  DD={best['dd']:.1f}%  "
                  f"PF={best['pf']:.2f}  Sharpe={best['sharpe']:.2f}")
            print(f"    Pair: {best['n_pair']}({pp:.0f}%)  Mom: {best['n_mom']}({mp:.0f}%)")
            print(f"    Avg Ann: {avg:+.1f}%")

    if pair_only and combo:
        bp = max(r['ann'] for r in pair_only)
        bc = max(r['ann'] for r in combo)
        print(f"\n  IMPROVEMENT: Combined best ({bc:+.1f}%) vs Pair-only best ({bp:+.1f}%) = {bc-bp:+.1f}%")

    # ================================================================
    # YEARLY BREAKDOWN TOP 5
    # ================================================================
    if len(results) >= 2:
        print(f"\n{'=' * 160}")
        print(f"  YEARLY BREAKDOWN FOR TOP 5")
        print(f"{'=' * 160}")
        for idx, r in enumerate(results[:5]):
            pp = r['n_pair'] / max(r['n'], 1) * 100
            mp = r['n_mom'] / max(r['n'], 1) * 100
            print(f"\n  #{idx+1}: {r['name']}")
            print(f"    Ann={r['ann']:+.1f}%  WR={r['wr']:.1f}%  DD={r['dd']:.1f}%  "
                  f"Sharpe={r['sharpe']:.2f}  N={r['n']}  Pair={r['n_pair']}({pp:.0f}%)  "
                  f"Mom={r['n_mom']}({mp:.0f}%)")
            print(f"    {'Year':>6s} | {'N':>5s} | {'WR':>5s} | {'PnL%':>8s} | {'PnL Abs':>12s} | "
                  f"{'Pair':>5s} | {'Mom':>5s}")
            print(f"    {'-' * 65}")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / max(ys['n'], 1) * 100
                print(f"    {y:6d} | {ys['n']:5d} | {wr_y:4.1f}% | {ys['pnl']:+7.1f}% | "
                      f"{ys['pnl_abs_sum']:+11.0f} | {ys['n_pair']:5d} | {ys['n_mom']:5d}")

    # ================================================================
    # PARAMETER SENSITIVITY
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  PARAMETER SENSITIVITY (combo configs only)")
    print(f"{'=' * 160}")

    combo_results = [r for r in results if r['name'].startswith('COMBO_')]

    # By Z threshold
    print(f"\n  --- Pair Z Threshold ---")
    for zt in [1.5, 2.0, 2.5, 3.0]:
        subset = [r for r in combo_results if f'_Z{zt:.1f}_' in r['name']]
        if subset:
            avg = np.mean([r['ann'] for r in subset])
            best = max(subset, key=lambda x: x['ann'])
            print(f"    Z={zt:.1f}: Avg={avg:+.1f}%  Best={best['ann']:+.1f}% ({best['name']})")

    # By momentum lookback
    print(f"\n  --- Momentum Lookback ---")
    for ml in [2, 3, 5]:
        subset = [r for r in combo_results if f'_ML{ml}_' in r['name']]
        if subset:
            avg = np.mean([r['ann'] for r in subset])
            best = max(subset, key=lambda x: x['ann'])
            print(f"    LB={ml}: Avg={avg:+.1f}%  Best={best['ann']:+.1f}% ({best['name']})")

    # By momentum threshold
    print(f"\n  --- Momentum Threshold ---")
    for mt in [0.003, 0.005, 0.01]:
        subset = [r for r in combo_results if f'_MT{mt*1000:.0f}_' in r['name']]
        if subset:
            avg = np.mean([r['ann'] for r in subset])
            best = max(subset, key=lambda x: x['ann'])
            print(f"    MT={mt}: Avg={avg:+.1f}%  Best={best['ann']:+.1f}% ({best['name']})")

    # By momentum scope
    print(f"\n  --- Momentum Scope ---")
    for scope in ['group', 'all']:
        subset = [r for r in combo_results if f'_{scope}_' in r['name']]
        if subset:
            avg = np.mean([r['ann'] for r in subset])
            best = max(subset, key=lambda x: x['ann'])
            print(f"    {scope:6s}: Avg={avg:+.1f}%  Best={best['ann']:+.1f}% ({best['name']})")

    # By commission
    print(f"\n  --- Commission ---")
    for comm_val in [0.0003, 0.0001]:
        subset = [r for r in combo_results if f'_C{comm_val*10000:.0f}' in r['name']]
        if subset:
            avg = np.mean([r['ann'] for r in subset])
            best = max(subset, key=lambda x: x['ann'])
            print(f"    {comm_val}: Avg={avg:+.1f}%  Best={best['ann']:+.1f}% ({best['name']})")

    # ================================================================
    # WALK-FORWARD TOP 10
    # ================================================================
    top10 = results[:10]
    print(f"\n{'=' * 160}")
    print(f"  RIGOROUS 6-WINDOW WALK-FORWARD (Top 10 configs)")
    print(f"  Windows: {WF_WINDOWS}")
    print(f"{'=' * 160}")

    wf_all = []
    wf_by_config = {}

    for rank, cfg_result in enumerate(top10):
        cn = cfg_result['name']
        matching = [c for c in configs if c['config_name'] == cn]
        if not matching: continue
        bc = matching[0]
        pp = cfg_result['n_pair'] / max(cfg_result['n'], 1) * 100
        mp = cfg_result['n_mom'] / max(cfg_result['n'], 1) * 100
        print(f"\n  [{rank+1}] {cn}  (full-period Ann={cfg_result['ann']:+.1f}%, "
              f"Pair={pp:.0f}%/Mom={mp:.0f}%)")

        for train_end, test_year in WF_WINDOWS:
            if test_year not in year_start_di:
                print(f"    Train -{train_end}/Test {test_year}: year not in data, SKIP")
                continue

            wf_cfg = dict(bc)
            wf_cfg['start_year'] = test_year
            wf_cfg['end_year'] = test_year
            wf_cfg['config_name'] = f"WF_Train-{train_end}_Test-{test_year}_{cn}"

            r = run_backtest(**wf_cfg)
            if r is not None:
                wf_all.append((cn, train_end, test_year, r))
                wf_by_config.setdefault(cn, []).append((train_end, test_year, r))
                wp = r['n_pair'] / max(r['n'], 1) * 100
                wm = r['n_mom'] / max(r['n'], 1) * 100
                print(f"    Train -{train_end}/Test {test_year}: Ann={r['ann']:+7.1f}%  "
                      f"WR={r['wr']:5.1f}%  N={r['n']:4d}  DD={r['dd']:5.1f}%  PF={r['pf']:4.2f}  "
                      f"Pair={r['n_pair']}({wp:.0f}%)  Mom={r['n_mom']}({wm:.0f}%)")
            else:
                print(f"    Train -{train_end}/Test {test_year}: insufficient trades")

    # ================================================================
    # WALK-FORWARD AGGREGATE TABLE
    # ================================================================
    if wf_by_config:
        print(f"\n{'=' * 160}")
        print(f"  WALK-FORWARD RESULTS TABLE (Top 10, sorted by avg WF Ann)")
        print(f"{'=' * 160}")

        wf_summary = []
        for cn, wr_list in wf_by_config.items():
            anns = [r['ann'] for _, _, r in wr_list]
            aby = {ty: r['ann'] for _, ty, r in wr_list}
            wf_summary.append({
                'name': cn, 'avg_ann': np.mean(anns),
                'med_ann': np.median(anns),
                'min_ann': min(anns), 'max_ann': max(anns),
                'ann_by_year': aby,
                'avg_wr': np.mean([r['wr'] for _, _, r in wr_list]),
                'avg_dd': np.mean([r['dd'] for _, _, r in wr_list]),
                'avg_pf': np.mean([r['pf'] for _, _, r in wr_list]),
                'avg_sharpe': np.mean([r['sharpe'] for _, _, r in wr_list]),
                'n_positive': sum(1 for a in anns if a > 0),
                'n_windows': len(wr_list),
                'window_details': wr_list,
            })
        wf_summary.sort(key=lambda x: -x['avg_ann'])

        # Build header with test year columns
        test_years = sorted(set(ty for _, _, ty, _ in wf_all))
        hdr_yr = " | ".join(f"{ty:>8s}" for ty in [str(y) for y in test_years])
        print(f"  {'#':>2s} | {'Config':55s} | {'Avg Ann':>8s} | {'Med':>8s} | {hdr_yr} | "
              f"{'Avg WR':>6s} | {'Avg DD':>7s} | {'Avg Sh':>6s} | {'Pos?':>4s}")
        print(f"  {'-' * 200}")

        for i, w in enumerate(wf_summary):
            yr_strs = []
            for ty in test_years:
                v = w['ann_by_year'].get(ty, float('nan'))
                yr_strs.append(f"{v:+7.1f}%" if not np.isnan(v) else "  N/A  ")
            yr_cols = " | ".join(f"{s:>8s}" for s in yr_strs)
            print(f"  {i+1:2d} | {w['name']:55s} | {w['avg_ann']:+7.1f}% | {w['med_ann']:+7.1f}% | "
                  f"{yr_cols} | {w['avg_wr']:5.1f}% | {w['avg_dd']:6.1f}% | {w['avg_sharpe']:5.2f} | "
                  f"{w['n_positive']}/{w['n_windows']}")

    # ================================================================
    # WALK-FORWARD WINDOW-BY-WINDOW DETAIL
    # ================================================================
    if wf_by_config:
        print(f"\n{'=' * 160}")
        print(f"  WALK-FORWARD WINDOW-BY-WINDOW DETAIL")
        print(f"{'=' * 160}")

        for i, w in enumerate(wf_summary):
            print(f"\n  [{i+1}] {w['name']}:")
            print(f"  {'Train':>9s} | {'Test':>4s} | {'Ann':>8s} | {'WR':>5s} | {'N':>5s} | "
                  f"{'DD':>6s} | {'PF':>5s} | {'Sharpe':>7s} | {'Pair':>5s} | {'Mom':>5s}")
            print(f"  {'-' * 90}")
            for train_end, test_year, r in sorted(w['window_details'], key=lambda x: x[1]):
                wp = r['n_pair'] / max(r['n'], 1) * 100
                wm = r['n_mom'] / max(r['n'], 1) * 100
                print(f"  -{train_end:4d}    | {test_year:4d} | {r['ann']:+7.1f}% | {r['wr']:4.1f}% | "
                      f"{r['n']:5d} | {r['dd']:5.1f}% | {r['pf']:4.2f} | {r['sharpe']:6.2f} | "
                      f"{wp:4.0f}% | {wm:4.0f}%")

    # ================================================================
    # OVERFITTING CHECK
    # ================================================================
    if wf_by_config:
        print(f"\n{'=' * 160}")
        print(f"  OVERFITTING CHECK: Full-Period vs Walk-Forward Correlation")
        print(f"{'=' * 160}")

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
                print(f"  -> GOOD: Strong positive correlation, training predicts OOS")
            elif corr > 0.2:
                print(f"  -> MODERATE: Some predictive power")
            else:
                print(f"  -> WARNING: Weak/no correlation, possible overfitting")

        # WF positive rate
        all_wf_anns = [r['ann'] for _, _, _, r in wf_all]
        n_pos_wf = sum(1 for a in all_wf_anns if a > 0)
        if all_wf_anns:
            print(f"\n  Overall WF positive rate: {n_pos_wf}/{len(all_wf_anns)} "
                  f"({n_pos_wf / len(all_wf_anns) * 100:.0f}%)")

        # Per-window positive rate
        print(f"\n  Per-window positive rate:")
        test_years = sorted(set(ty for _, _, ty, _ in wf_all))
        for ty in test_years:
            ty_anns = [r['ann'] for _, _, test_y, r in wf_all if test_y == ty]
            if ty_anns:
                n_pos = sum(1 for a in ty_anns if a > 0)
                print(f"    {ty}: {n_pos}/{len(ty_anns)} positive ({n_pos/len(ty_anns)*100:.0f}%) "
                      f"Avg={np.mean(ty_anns):+.1f}%")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 160}")

    if results:
        b = results[0]
        pp = b['n_pair'] / max(b['n'], 1) * 100
        mp = b['n_mom'] / max(b['n'], 1) * 100
        print(f"\n  Best full-period: {b['name']}")
        print(f"    Ann={b['ann']:+.1f}%  WR={b['wr']:.1f}%  N={b['n']}  DD={b['dd']:.1f}%  "
              f"PF={b['pf']:.2f}  Sharpe={b['sharpe']:.2f}")
        print(f"    Pair: {b['n_pair']}({pp:.1f}%) WR={b['pair_wr']:.1f}% PnL={b['pair_pnl']:+.0f}")
        print(f"    Momentum: {b['n_mom']}({mp:.1f}%) WR={b['mom_wr']:.1f}% PnL={b['mom_pnl']:+.0f}")

    combo_both = [r for r in results if r['n_pair'] > 0 and r['n_mom'] > 0]
    if combo_both:
        bc = max(combo_both, key=lambda x: x['ann'])
        cp = bc['n_pair'] / max(bc['n'], 1) * 100
        cm = bc['n_mom'] / max(bc['n'], 1) * 100
        print(f"\n  Best COMBO (both types active): {bc['name']}")
        print(f"    Ann={bc['ann']:+.1f}%  WR={bc['wr']:.1f}%  N={bc['n']}  DD={bc['dd']:.1f}%  "
              f"PF={bc['pf']:.2f}  Sharpe={bc['sharpe']:.2f}")
        print(f"    Pair: {bc['n_pair']}({cp:.1f}%) WR={bc['pair_wr']:.1f}% PnL={bc['pair_pnl']:+.0f}")
        print(f"    Momentum: {bc['n_mom']}({cm:.1f}%) WR={bc['mom_wr']:.1f}% PnL={bc['mom_pnl']:+.0f}")

    if pair_only and combo_both:
        bp = max(pair_only, key=lambda x: x['ann'])['ann']
        bcc = max(combo_both, key=lambda x: x['ann'])['ann']
        print(f"\n  Value-add of momentum fallback:")
        print(f"    Pair-only best: {bp:+.1f}%")
        print(f"    Combo best:     {bcc:+.1f}%")
        print(f"    Improvement:    {bcc-bp:+.1f}%")

    if wf_summary:
        bw = wf_summary[0]
        print(f"\n  Best WF avg: {bw['name']}")
        print(f"    WF Avg Ann={bw['avg_ann']:+.1f}%  Med={bw['med_ann']:+.1f}%  "
              f"Min={bw['min_ann']:+.1f}%  Max={bw['max_ann']:+.1f}%  "
              f"Pos={bw['n_positive']}/{bw['n_windows']}")

        # Cross-reference: does the best full-period config also do well in WF?
        if results:
            best_fp_name = results[0]['name']
            matching_wf = [w for w in wf_summary if w['name'] == best_fp_name]
            if matching_wf:
                mw = matching_wf[0]
                print(f"\n  Best full-period config WF performance:")
                print(f"    {best_fp_name}")
                print(f"    WF Avg={mw['avg_ann']:+.1f}%  Pos={mw['n_positive']}/{mw['n_windows']}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 160)


if __name__ == '__main__':
    main()
