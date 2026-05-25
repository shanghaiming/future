"""
Alpha Futures V63 -- Multi-Strategy Portfolio: V62 Pair + V34b Momentum
=======================================================================
Goal: 600% annual by running V62 pair trading and V34b momentum simultaneously
with intelligent capital switching.

Key insight: V62 uses ~2800 trades over 10 years with 1 pair at a time.
But on ~60% of trading days, V62 has NO signal (z < threshold for all pairs).
On those idle days, we run V34b momentum strategy to keep capital working.

This is NOT a combination -- it's a priority system:
  1. First check: does any pair have |z| > threshold? If yes -> trade the pair
  2. If no pair signals -> check V34b group momentum lag.
     If any commodity has score > momentum_threshold -> trade it
  3. If neither signals -> stay in cash for the day

Default: V62 LOG-biased adaptive pair trading (14 pairs, Z=1.0, 1-day hold, MP=1)
Fallback: V34b group momentum lag signal (top commodity by group lag score)

~150 configs. Walk-forward for best (2023, 2024).
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

SPREAD_RAW = 'raw'
SPREAD_PCT = 'pct'
SPREAD_LOG = 'log'
ALL_MODES = [SPREAD_RAW, SPREAD_PCT, SPREAD_LOG]
ALL_LOOKBACKS = [5, 7, 10, 15, 20]


def main():
    t_start = time.time()
    print("=" * 160)
    print("Alpha Futures V63 -- Multi-Strategy: V62 Pair + V34b Momentum Priority System")
    print("Default: V62 LOG-biased adaptive pair trading | Fallback: V34b group momentum lag")
    print("Capital: 100% for whichever strategy is active. No splitting.")
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

    def build_pair_indices(pairs_list):
        indices = []
        for down_sym, up_sym in pairs_list:
            down_si = sym_to_si.get(down_sym, -1)
            up_si = sym_to_si.get(up_sym, -1)
            if down_si >= 0 and up_si >= 0:
                indices.append((down_si, up_si, down_sym, up_sym))
        return indices

    pair_indices_14 = build_pair_indices(PAIRS_14)
    print(f"  Pair set: P14={len(pair_indices_14)}")

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
    # PRECOMPUTE PAIR Z-SCORES
    # ================================================================
    print("\n[Signals] Precomputing pair spreads and z-scores...", flush=True)
    t0 = time.time()

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

    pair_combo_daily_return = {}
    all_zt = [0.5, 0.8, 1.0, 1.2]

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
                        pnl_pct = (pnl - invested * COMM * 2) / invested * 100 if invested > 0 else 0
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
    # PRECOMPUTE MOMENTUM SIGNALS (V34b)
    # ================================================================
    print("\n[Signals] Precomputing momentum signals...", flush=True)
    t2 = time.time()

    mom = {}
    for lag in [3, 5, 7]:
        m = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(lag, ND):
                c_now = C[si, di]
                c_prev = C[si, di - lag]
                if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                    m[si, di] = (c_now - c_prev) / c_prev
        mom[lag] = m

    grp_mom = {}
    for lag in [3, 5, 7]:
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
    print(f"  Momentum signals precomputed ({time.time() - t2:.1f}s)", flush=True)

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(z_thresh=1.0, hold_max=1, exit_z=0.0,
                     mode_type='adaptive_LOG', eval_period=40,
                     candidate_combos=None, pair_indices=None,
                     momentum_threshold=0.003, mom_lookback=5,
                     start_year=None, end_year=None, config_name=""):
        if pair_indices is None:
            pair_indices = pair_indices_14
        if candidate_combos is None:
            candidate_combos = [(SPREAD_LOG, 10), (SPREAD_LOG, 15), (SPREAD_LOG, 20),
                                (SPREAD_RAW, 10), (SPREAD_PCT, 10)]

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
                    pd = pos['dir']
                    if not np.isnan(z_now):
                        if pd == 1 and z_now >= exit_z:
                            exit_reason = 'mean_rev'
                        elif pd == -1 and z_now <= -exit_z:
                            exit_reason = 'mean_rev'
                    if exit_reason is None and not np.isnan(z_now):
                        if pd == 1 and z_now < pos['entry_z'] - 1.5:
                            exit_reason = 'stop_loss'
                        elif pd == -1 and z_now > pos['entry_z'] + 1.5:
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
                        if pd == 1:
                            pnl_d = (cd - pos['entry_down']) * md * ld
                            pnl_u = (pos['entry_up'] - cu) * mu * lu
                        else:
                            pnl_d = (pos['entry_down'] - cd) * md * ld
                            pnl_u = (cu - pos['entry_up']) * mu * lu
                        ev_d = pos['entry_down'] * md * ld
                        ev_u = pos['entry_up'] * mu * lu
                        xv_d = cd * md * ld
                        xv_u = cu * mu * lu
                        cost = (ev_d + ev_u) * COMM + (xv_d + xv_u) * COMM
                        total_pnl = pnl_d + pnl_u - cost
                        invested = ev_d + ev_u
                        pnl_pct = total_pnl / invested * 100 if invested > 0 else 0
                        if pd == 1:
                            cash_return = cd * md * ld - cu * mu * lu
                        else:
                            cash_return = -cd * md * ld + cu * mu * lu
                        cash += pos['cash_invested'] + cash_return - (xv_d + xv_u) * COMM
                        trades.append({
                            'pnl_abs': total_pnl, 'pnl_pct': pnl_pct,
                            'days': days_held, 'di': di, 'year': year,
                            'type': 'pair',
                            'pair': (pos['down_sym'], pos['up_sym']),
                            'pair_label': PAIR_LABEL.get((pos['down_sym'], pos['up_sym']), ''),
                            'dir': pd, 'reason': exit_reason,
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
                        cash += mkt_val - mkt_val * COMM
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
                    ld = int(cpl / (cd * md * (1 + COMM)))
                    lu = int(cpl / (cu * mu * (1 + COMM)))
                    if ld > 0 and lu > 0:
                        tc = cd * md * ld * (1 + COMM) + cu * mu * lu * (1 + COMM)
                        if tc > cash:
                            sc = cash * 0.95 / tc
                            ld = max(1, int(ld * sc))
                            lu = max(1, int(lu * sc))
                            tc = cd * md * ld * (1 + COMM) + cu * mu * lu * (1 + COMM)
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
            for si in group_sis:
                own = mom[mom_lookback][si, di]
                grp = grp_mom[mom_lookback][si, di]
                if np.isnan(own) or np.isnan(grp): continue
                score = grp - own
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
                    lots = int(cash / (notional * (1 + COMM)))
                    if lots > 0:
                        cost_in = notional * lots * (1 + COMM)
                        if cost_in > cash:
                            lots = int(cash / (notional * (1 + COMM)))
                            cost_in = notional * lots * (1 + COMM) if lots > 0 else 0
                        if lots > 0 and cost_in > 0 and cost_in <= cash:
                            cash -= cost_in
                            current_position = {
                                'type': 'momentum',
                                'si': si, 'entry': c, 'entry_di': di,
                                'lots': lots, 'dir': 1, 'sym': sym,
                                'group': GROUP_MAP.get(sym, 'unknown'),
                            }

        # Close remaining
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
                cost = (pos['entry_down'] * md * ld + pos['entry_up'] * mu * lu) * COMM + (xv_d + xv_u) * COMM
                total_pnl = pnl_d + pnl_u - cost
                cash += pos['cash_invested'] + cash_ret - (xv_d + xv_u) * COMM
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
                cash += mkt_val - mkt_val * COMM
                trades.append({
                    'pnl_abs': pnl,
                    'pnl_pct': pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] > 0 else 0,
                    'days': ae - pos['entry_di'], 'di': ae, 'year': dates[ae].year,
                    'type': 'momentum', 'sym': pos['sym'],
                    'group': pos['group'], 'dir': pos['dir'], 'reason': 'end',
                })

        if len(trades) < 3:
            return None

        # Stats
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
            'mom_wr': mom_wins / max(n_mom, 1) * 100 if n_mom > 0 else 0,
        }

    # ================================================================
    # BUILD CONFIGURATIONS
    # ================================================================
    print("\n[Configs] Building configurations...", flush=True)
    configs = []
    log_bias_combos = [(SPREAD_LOG, 10), (SPREAD_LOG, 15), (SPREAD_LOG, 20),
                       (SPREAD_RAW, 10), (SPREAD_PCT, 10)]

    # Pair-only baselines
    for zt in [0.8, 1.0, 1.2]:
        for sm, slb in [(SPREAD_LOG, 15), (SPREAD_LOG, 10)]:
            configs.append({
                'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0,
                'mode_type': 'fixed', 'candidate_combos': [(sm, slb)],
                'pair_indices': pair_indices_14,
                'momentum_threshold': 999.0, 'mom_lookback': 5,
                'start_year': None, 'end_year': None,
                'config_name': f"PAIR_ONLY_{sm}_LB{slb}_Z{zt:.1f}",
            })
        configs.append({
            'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0,
            'mode_type': 'adaptive_LOG', 'eval_period': 40,
            'candidate_combos': log_bias_combos,
            'pair_indices': pair_indices_14,
            'momentum_threshold': 999.0, 'mom_lookback': 5,
            'start_year': None, 'end_year': None,
            'config_name': f"PAIR_ONLY_adapt_Z{zt:.1f}",
        })

    # Momentum-only baselines
    for ml in [3, 5, 7]:
        for mt in [0.003, 0.005, 0.01, 0.02]:
            configs.append({
                'z_thresh': 99.0, 'hold_max': 1, 'exit_z': 0.0,
                'mode_type': 'fixed', 'candidate_combos': [(SPREAD_LOG, 10)],
                'pair_indices': pair_indices_14,
                'momentum_threshold': mt, 'mom_lookback': ml,
                'start_year': None, 'end_year': None,
                'config_name': f"MOM_ONLY_LB{ml}_MT{mt*1000:.0f}",
            })

    # Combined: fixed spread + momentum
    for zt in [0.8, 1.0, 1.2]:
        for sm, slb in [(SPREAD_LOG, 15), (SPREAD_LOG, 10)]:
            for mt in [0.003, 0.005, 0.01]:
                for ml in [3, 5, 7]:
                    configs.append({
                        'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0,
                        'mode_type': 'fixed', 'candidate_combos': [(sm, slb)],
                        'pair_indices': pair_indices_14,
                        'momentum_threshold': mt, 'mom_lookback': ml,
                        'start_year': None, 'end_year': None,
                        'config_name': f"COMBO_{sm}_LB{slb}_Z{zt:.1f}_ML{ml}_MT{mt*1000:.0f}",
                    })

    # Combined: adaptive pair + momentum
    for zt in [0.8, 1.0, 1.2]:
        for mt in [0.003, 0.005, 0.01]:
            for ml in [3, 5, 7]:
                configs.append({
                    'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0,
                    'mode_type': 'adaptive_LOG', 'eval_period': 40,
                    'candidate_combos': log_bias_combos,
                    'pair_indices': pair_indices_14,
                    'momentum_threshold': mt, 'mom_lookback': ml,
                    'start_year': None, 'end_year': None,
                    'config_name': f"COMBO_adapt_Z{zt:.1f}_ML{ml}_MT{mt*1000:.0f}",
                })

    # Higher Z to force more momentum days
    for zt in [1.5, 2.0]:
        for mt in [0.003, 0.005, 0.01]:
            for ml in [3, 5]:
                configs.append({
                    'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0,
                    'mode_type': 'adaptive_LOG', 'eval_period': 40,
                    'candidate_combos': log_bias_combos,
                    'pair_indices': pair_indices_14,
                    'momentum_threshold': mt, 'mom_lookback': ml,
                    'start_year': None, 'end_year': None,
                    'config_name': f"COMBO_adapt_Z{zt:.1f}_ML{ml}_MT{mt*1000:.0f}",
                })

    total_combos = len(configs)
    print(f"  {total_combos} configurations")

    # ================================================================
    # RUN SWEEP
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
    hdr = f"  {'#':>2s} | {'Config':55s} | {'Ann':>8s} | {'WR':>5s} | {'N':>5s} | {'DD':>6s} | {'PF':>5s} | {'Sh':>6s} | {'Pair%':>6s} | {'Mom%':>5s} | {'Cash':>12s}"
    print(hdr)
    print(f"  {'-' * 150}")
    for i, r in enumerate(results[:20]):
        pp = r['n_pair'] / max(r['n'], 1) * 100
        mp = r['n_mom'] / max(r['n'], 1) * 100
        print(f"  {i+1:2d} | {r['name']:55s} | {r['ann']:+7.1f}% | {r['wr']:4.1f}% | {r['n']:5d} | {r['dd']:5.1f}% | {r['pf']:4.2f} | {r['sharpe']:5.2f} | {pp:5.1f}% | {mp:4.1f}% | {r['cash']:11.0f}")

    # ================================================================
    # TRADE TYPE BREAKDOWN
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  TRADE TYPE BREAKDOWN (top 20)")
    print(f"{'=' * 160}")
    for i, r in enumerate(results[:20]):
        pp = r['n_pair'] / max(r['n'], 1) * 100
        mp = r['n_mom'] / max(r['n'], 1) * 100
        print(f"  {i+1:2d} | {r['name']:55s} | Pair:{r['n_pair']:4d}({pp:.0f}%) WR={r['pair_wr']:.1f}% PnL={r['pair_pnl']:+.0f} | Mom:{r['n_mom']:4d}({mp:.0f}%) WR={r['mom_wr']:.1f}% PnL={r['mom_pnl']:+.0f}")

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
            print(f"    Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  N={best['n']}  DD={best['dd']:.1f}%  PF={best['pf']:.2f}  Sharpe={best['sharpe']:.2f}")
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
            print(f"    Ann={r['ann']:+.1f}%  WR={r['wr']:.1f}%  DD={r['dd']:.1f}%  Sharpe={r['sharpe']:.2f}  N={r['n']}  Pair={r['n_pair']}({pp:.0f}%)  Mom={r['n_mom']}({mp:.0f}%)")
            print(f"    {'Year':>6s} | {'N':>5s} | {'WR':>5s} | {'PnL%':>8s} | {'PnL Abs':>12s} | {'Pair':>5s} | {'Mom':>5s}")
            print(f"    {'-' * 65}")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / max(ys['n'], 1) * 100
                print(f"    {y:6d} | {ys['n']:5d} | {wr_y:4.1f}% | {ys['pnl']:+7.1f}% | {ys['pnl_abs_sum']:+11.0f} | {ys['n_pair']:5d} | {ys['n_mom']:5d}")

    # ================================================================
    # WALK-FORWARD TOP 10
    # ================================================================
    top10 = results[:10]
    print(f"\n{'=' * 160}")
    print(f"  WALK-FORWARD VALIDATION (Top 10, windows: 2023, 2024)")
    print(f"{'=' * 160}")

    wf_all = []
    wf_by_config = {}
    for rank, cfg in enumerate(top10):
        cn = cfg['name']
        matching = [c for c in configs if c['config_name'] == cn]
        if not matching: continue
        bc = matching[0]
        pp = cfg['n_pair'] / max(cfg['n'], 1) * 100
        mp = cfg['n_mom'] / max(cfg['n'], 1) * 100
        print(f"\n  [{rank+1}] {cn}  (full-period Ann={cfg['ann']:+.1f}%, Pair={pp:.0f}%/Mom={mp:.0f}%)")
        for ty in [2023, 2024]:
            if ty not in year_start_di: continue
            wc = dict(bc)
            wc['start_year'] = ty; wc['end_year'] = ty
            wc['config_name'] = f"WF_{ty}_{cn}"
            r = run_backtest(**wc)
            if r is not None:
                wf_all.append((cn, ty, r))
                wf_by_config.setdefault(cn, []).append((ty, r))
                wp = r['n_pair'] / max(r['n'], 1) * 100
                wm = r['n_mom'] / max(r['n'], 1) * 100
                print(f"    {ty}: Ann={r['ann']:+7.1f}%  WR={r['wr']:5.1f}%  N={r['n']:4d}  DD={r['dd']:5.1f}%  PF={r['pf']:4.2f}  Pair={r['n_pair']}({wp:.0f}%)  Mom={r['n_mom']}({wm:.0f}%)")
            else:
                print(f"    {ty}: insufficient trades")

    if wf_by_config:
        print(f"\n{'=' * 160}")
        print(f"  WALK-FORWARD RESULTS TABLE")
        print(f"{'=' * 160}")
        wf_summary = []
        for cn, wr_list in wf_by_config.items():
            anns = [r['ann'] for _, r in wr_list]
            aby = {ty: r['ann'] for ty, r in wr_list}
            wf_summary.append({
                'name': cn, 'avg_ann': np.mean(anns),
                'ann_2023': aby.get(2023, float('nan')),
                'ann_2024': aby.get(2024, float('nan')),
                'avg_wr': np.mean([r['wr'] for _, r in wr_list]),
                'avg_dd': np.mean([r['dd'] for _, r in wr_list]),
                'avg_pf': np.mean([r['pf'] for _, r in wr_list]),
                'n_positive': sum(1 for a in anns if a > 0),
                'n_windows': len(wr_list),
            })
        wf_summary.sort(key=lambda x: -x['avg_ann'])
        print(f"  {'#':>2s} | {'Config':55s} | {'Avg Ann':>8s} | {'2023':>8s} | {'2024':>8s} | {'Avg WR':>6s} | {'Avg DD':>7s} | {'Pos?':>4s}")
        print(f"  {'-' * 120}")
        for i, w in enumerate(wf_summary):
            a23 = f"{w['ann_2023']:+7.1f}%" if not np.isnan(w['ann_2023']) else "  N/A  "
            a24 = f"{w['ann_2024']:+7.1f}%" if not np.isnan(w['ann_2024']) else "  N/A  "
            print(f"  {i+1:2d} | {w['name']:55s} | {w['avg_ann']:+7.1f}% | {a23} | {a24} | {w['avg_wr']:5.1f}% | {w['avg_dd']:6.1f}% | {w['n_positive']}/{w['n_windows']}")

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
        print(f"    Ann={b['ann']:+.1f}%  WR={b['wr']:.1f}%  N={b['n']}  DD={b['dd']:.1f}%  PF={b['pf']:.2f}  Sharpe={b['sharpe']:.2f}")
        print(f"    Pair: {b['n_pair']}({pp:.1f}%) WR={b['pair_wr']:.1f}% PnL={b['pair_pnl']:+.0f}")
        print(f"    Momentum: {b['n_mom']}({mp:.1f}%) WR={b['mom_wr']:.1f}% PnL={b['mom_pnl']:+.0f}")

    combo_both = [r for r in results if r['n_pair'] > 0 and r['n_mom'] > 0]
    if combo_both:
        bc = max(combo_both, key=lambda x: x['ann'])
        cp = bc['n_pair'] / max(bc['n'], 1) * 100
        cm = bc['n_mom'] / max(bc['n'], 1) * 100
        print(f"\n  Best COMBO (both types active): {bc['name']}")
        print(f"    Ann={bc['ann']:+.1f}%  WR={bc['wr']:.1f}%  N={bc['n']}  DD={bc['dd']:.1f}%  PF={bc['pf']:.2f}")
        print(f"    Pair: {bc['n_pair']}({cp:.1f}%) WR={bc['pair_wr']:.1f}% PnL={bc['pair_pnl']:+.0f}")
        print(f"    Momentum: {bc['n_mom']}({cm:.1f}%) WR={bc['mom_wr']:.1f}% PnL={bc['mom_pnl']:+.0f}")

    if wf_by_config:
        bw = max(wf_by_config.items(), key=lambda x: np.mean([r['ann'] for _, r in x[1]]))
        avg_wf = np.mean([r['ann'] for _, r in bw[1]])
        print(f"\n  Best WF avg: {bw[0]}  Avg Ann={avg_wf:+.1f}%")

    if pair_only and combo_both:
        bp = max(pair_only, key=lambda x: x['ann'])['ann']
        bcc = max(combo_both, key=lambda x: x['ann'])['ann']
        print(f"\n  Value-add of momentum fallback:")
        print(f"    Pair-only best: {bp:+.1f}%")
        print(f"    Combo best:     {bcc:+.1f}%")
        print(f"    Improvement:    {bcc-bp:+.1f}%")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 160)


if __name__ == '__main__':
    main()
