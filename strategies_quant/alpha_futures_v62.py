"""
Alpha Futures V62 -- Final Push Toward 600%
=============================================
V61 champion: +324.3% annual, 6/6 WF positive, avg +394.9%.
Config: LOG-biased adaptive, EP40, Z=0.8, MP1, 14 pairs.

New axes of exploration for 600%:
  1. Add more pairs beyond 14 (same-group ferrous, metals, soy products)
  2. Intra-day re-entry: if z still extreme after close, re-enter immediately
  3. Z-score weighted position sizing: scale by z magnitude, reserve cash
  4. Multi-day compounding: allow overlapping entries across days
  5. Adaptive Z threshold: lower Z when vol low, higher when vol high
  6. Combine best individual improvements

~200-250 configs. Rigorous 6-window WF for top 5.

Walk-forward windows:
  Train 2016-2019, Test 2020
  Train 2016-2020, Test 2021
  Train 2016-2021, Test 2022
  Train 2016-2022, Test 2023
  Train 2016-2023, Test 2024
  Train 2016-2024, Test 2025
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

# V61's 14 pairs (13 original + cfi/csfi)
PAIRS_13 = [
    ('rbfi', 'ifi'), ('hcfi', 'ifi'), ('hcfi', 'rbfi'),
    ('jfi', 'jmfi'), ('mafi', 'scfi'), ('fufi', 'scfi'),
    ('bfi', 'scfi'), ('mfi', 'afi'), ('yfi', 'afi'),
    ('pfi', 'yfi'), ('ppfi', 'mafi'), ('vfi', 'mafi'),
    ('egfi', 'mafi'),
]
PAIRS_14 = PAIRS_13 + [('cfi', 'csfi')]

# Extra pairs for testing
PAIRS_14_P2 = PAIRS_14 + [('jfi', 'ifi'), ('cufi', 'znfi')]
PAIRS_14_P4 = PAIRS_14_P2 + [('alfi', 'znfi'), ('mfi', 'yfi')]

PAIR_LABEL = {
    ('rbfi', 'ifi'):  'rebar/iron_ore',
    ('hcfi', 'ifi'):  'hotcoil/iron_ore',
    ('hcfi', 'rbfi'): 'hotcoil/rebar',
    ('jfi', 'jmfi'):  'coke/coal',
    ('mafi', 'scfi'): 'methanol/crude',
    ('fufi', 'scfi'): 'fueloil/crude',
    ('bfi', 'scfi'):  'bitumen/crude',
    ('mfi', 'afi'):   'meal/soybean',
    ('yfi', 'afi'):   'soyoil/soybean',
    ('pfi', 'yfi'):   'palm/soyoil',
    ('ppfi', 'mafi'): 'PP/methanol',
    ('vfi', 'mafi'):  'PVC/methanol',
    ('egfi', 'mafi'): 'EG/methanol',
    ('cfi', 'csfi'):  'corn/cornstarch',
    ('jfi', 'ifi'):   'coke/iron_ore',
    ('cufi', 'znfi'): 'copper/zinc',
    ('alfi', 'znfi'): 'aluminum/zinc',
    ('mfi', 'yfi'):   'meal/soyoil',
}

SPREAD_RAW = 'raw'
SPREAD_PCT = 'pct'
SPREAD_LOG = 'log'
ALL_MODES = [SPREAD_RAW, SPREAD_PCT, SPREAD_LOG]
ALL_LOOKBACKS = [5, 7, 10, 15, 20]

# Walk-forward windows: (train_end_year, test_year)
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
    print("Alpha Futures V62 -- Final Push Toward 600%")
    print("V61 champion: +324.3% annual, 6/6 WF positive, avg +394.9%")
    print("New: more pairs, re-entry, z-weighted sizing, multi-day, adaptive Z, combos")
    print("=" * 160)

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
    print(f"  {NS} commodities, {ND} days, years in data: {sorted(year_start_di.keys())}")

    # Build pair index mapping for all pair sets
    def build_pair_indices(pairs_list):
        indices = []
        for down_sym, up_sym in pairs_list:
            down_si = sym_to_si.get(down_sym, -1)
            up_si = sym_to_si.get(up_sym, -1)
            if down_si >= 0 and up_si >= 0:
                indices.append((down_si, up_si, down_sym, up_sym))
            else:
                print(f"  WARNING: pair ({down_sym}, {up_sym}) not found")
        return indices

    pair_indices_14 = build_pair_indices(PAIRS_14)
    pair_indices_p2 = build_pair_indices(PAIRS_14_P2)
    pair_indices_p4 = build_pair_indices(PAIRS_14_P4)

    print(f"  Pair sets: P14={len(pair_indices_14)}, P14+2={len(pair_indices_p2)}, P14+4={len(pair_indices_p4)}")

    # ================================================================
    # PRECOMPUTE SPREADS AND Z-SCORES FOR ALL MODES x LOOKBACKS
    # ================================================================
    print("\n[Signals] Precomputing spreads and z-scores...", flush=True)
    t0 = time.time()

    z_scores = {m: {} for m in ALL_MODES}

    all_pair_set = set()
    for pidx in [pair_indices_14, pair_indices_p2, pair_indices_p4]:
        for down_si, up_si, down_sym, up_sym in pidx:
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

    print(f"  Z-scores precomputed ({time.time() - t0:.1f}s)", flush=True)

    # ================================================================
    # PRECOMPUTE SPREAD VOLATILITY (for adaptive Z threshold)
    # ================================================================
    print("\n[Signals] Precomputing spread volatility for adaptive Z...", flush=True)
    t_vol = time.time()

    # For each pair, compute rolling std of log spread over a lookback
    spread_vol = {}  # (down_si, up_si) -> {lb: vol_array}
    for down_si, up_si in all_pair_set:
        key = (down_si, up_si)
        spread_vol[key] = {}
        # Use log spread for vol measurement
        z_log = z_scores[SPREAD_LOG].get(key, {})
        for lb in ALL_LOOKBACKS:
            z_arr = z_log.get(lb)
            if z_arr is None:
                continue
            # Rolling volatility of the z-score itself (shorter window)
            vol_window = 20
            vol_arr = np.full(ND, np.nan)
            for di in range(vol_window, ND):
                w = z_arr[di - vol_window:di]
                valid = w[~np.isnan(w)]
                if len(valid) >= 5:
                    vol_arr[di] = np.std(valid, ddof=1)
            spread_vol[key][lb] = vol_arr

    print(f"  Spread volatility precomputed ({time.time() - t_vol:.1f}s)", flush=True)

    # ================================================================
    # PRECOMPUTE PER-PAIR HYPOTHETICAL RETURNS
    # ================================================================
    print("\n[Signals] Precomputing per-pair hypothetical returns...", flush=True)
    t1 = time.time()

    pair_combo_daily_return = {}
    all_zt = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5]
    all_pair_indices = pair_indices_p4  # use largest set for precomputation

    for zt in all_zt:
        for mode in ALL_MODES:
            for lb in ALL_LOOKBACKS:
                combo_key = (mode, lb, zt)
                for down_si, up_si, down_sym, up_sym in all_pair_indices:
                    pair_key = (down_si, up_si, down_sym, up_sym)
                    daily_ret = np.full(ND, np.nan)

                    for di in range(MIN_TRAIN + 1, ND):
                        z_arr = z_scores[mode].get((down_si, up_si), {}).get(lb)
                        if z_arr is None:
                            continue
                        z_prev = z_arr[di - 1]
                        if np.isnan(z_prev) or abs(z_prev) < zt:
                            continue

                        c_down_entry = C[down_si, di - 1]
                        c_up_entry = C[up_si, di - 1]
                        c_down_exit = C[down_si, di]
                        c_up_exit = C[up_si, di]
                        if (np.isnan(c_down_entry) or c_down_entry <= 0 or
                            np.isnan(c_up_entry) or c_up_entry <= 0 or
                            np.isnan(c_down_exit) or c_down_exit <= 0 or
                            np.isnan(c_up_exit) or c_up_exit <= 0):
                            continue

                        mult_down = MULT.get(down_sym, DEF_MULT)
                        mult_up = MULT.get(up_sym, DEF_MULT)

                        if z_prev > 0:
                            pnl_down = (c_down_entry - c_down_exit) * mult_down
                            pnl_up = (c_up_exit - c_up_entry) * mult_up
                        else:
                            pnl_down = (c_down_exit - c_down_entry) * mult_down
                            pnl_up = (c_up_entry - c_up_exit) * mult_up

                        invested = c_down_entry * mult_down + c_up_entry * mult_up
                        cost = invested * COMM * 2
                        pnl_pct = (pnl_down + pnl_up - cost) / invested * 100 if invested > 0 else 0
                        daily_ret[di] = pnl_pct

                    pair_combo_daily_return[(pair_key, combo_key)] = daily_ret

    # Global combo daily returns (average across all pairs in use)
    global_combo_daily_return = {}
    for zt in all_zt:
        for mode in ALL_MODES:
            for lb in ALL_LOOKBACKS:
                combo_key = (mode, lb, zt)
                daily_ret = np.full(ND, np.nan)
                for di in range(MIN_TRAIN + 1, ND):
                    pair_rets = []
                    for down_si, up_si, down_sym, up_sym in all_pair_indices:
                        pk = (down_si, up_si, down_sym, up_sym)
                        pr = pair_combo_daily_return.get((pk, combo_key))
                        if pr is not None and not np.isnan(pr[di]):
                            pair_rets.append(pr[di])
                    if pair_rets:
                        daily_ret[di] = np.mean(pair_rets)
                global_combo_daily_return[combo_key] = daily_ret

    print(f"  Hypothetical returns precomputed ({time.time() - t1:.1f}s)", flush=True)

    # ================================================================
    # BACKTEST ENGINE (enhanced with re-entry, z-weighted sizing,
    #   multi-day compounding, adaptive Z)
    # ================================================================
    def run_backtest(z_thresh=0.8, hold_max=1, exit_z=0.0, max_pairs=1,
                     mode_type='adaptive_log_bias',
                     eval_period=40,
                     candidate_combos=None,
                     pair_indices=None,
                     start_year=None, end_year=None,
                     reentry=False,
                     z_weighted_sizing=False,
                     multi_day=False,
                     adaptive_z=False,
                     config_name=""):
        """
        Enhanced backtest engine with:
          reentry: if True, after closing a pair at day end, immediately re-enter
                   if z is still extreme on the same pair (no cooldown)
          z_weighted_sizing: if True, scale position size by z-score magnitude
                   z < 1.0 -> 60% of capital, z 1.0-1.5 -> 80%, z > 1.5 -> 100%
          multi_day: if True, allow overlapping positions from different days
                   (close yesterday's + open new one independently)
          adaptive_z: if True, adapt z_thresh based on recent spread vol
                   low vol -> use lower Z, high vol -> use higher Z
        """
        if pair_indices is None:
            pair_indices = pair_indices_14
        if candidate_combos is None:
            candidate_combos = [(SPREAD_LOG, 10), (SPREAD_LOG, 15), (SPREAD_LOG, 20),
                                (SPREAD_RAW, 10), (SPREAD_PCT, 10)]

        cash = float(CASH0)
        trades = []
        pair_positions = []

        current_combo = candidate_combos[0]

        # Date range
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

            # --- Adaptive evaluation every eval_period days ---
            if di > start_di:
                days_since_start = di - start_di
                if days_since_start % eval_period == 0 and days_since_start >= eval_period:
                    if mode_type == 'fixed':
                        pass
                    else:
                        best_combo = candidate_combos[0]
                        best_score = -1e18
                        for c in candidate_combos:
                            combo_key = (c[0], c[1], z_thresh)
                            daily_ret = global_combo_daily_return.get(combo_key)
                            if daily_ret is None:
                                continue
                            window = daily_ret[max(start_di, di - eval_period):di]
                            valid = window[~np.isnan(window)]
                            if len(valid) >= 3:
                                score = np.nansum(valid)
                            else:
                                score = -1e10
                            if score > best_score:
                                best_score = score
                                best_combo = c
                        current_combo = best_combo

            # Determine effective z threshold (possibly adaptive)
            effective_z = z_thresh
            if adaptive_z:
                # Compute average spread vol across all pairs today
                vol_vals = []
                use_mode, use_lb_ad = current_combo if mode_type != 'fixed' else candidate_combos[0]
                for down_si, up_si, _, _ in pair_indices:
                    v = spread_vol.get((down_si, up_si), {}).get(use_lb_ad)
                    if v is not None and di < len(v) and not np.isnan(v[di]):
                        vol_vals.append(v[di])
                if vol_vals:
                    avg_vol = np.mean(vol_vals)
                    # Scale z_thresh: if vol < 0.8 (calm), use lower; if > 1.2 (turbulent), use higher
                    vol_ratio = avg_vol / 1.0  # 1.0 is "normal"
                    effective_z = z_thresh * (0.7 + 0.3 * vol_ratio)
                    effective_z = max(0.3, min(effective_z, 2.0))  # clamp

            # --- Manage existing pair positions ---
            new_positions = []
            reentry_candidates = []
            for pos in pair_positions:
                p_down_si = pos['down_si']
                p_up_si = pos['up_si']
                p_mode = pos['mode']
                p_lb = pos['lb']
                z_arr = z_scores[p_mode].get((p_down_si, p_up_si), {}).get(p_lb)
                if z_arr is None:
                    new_positions.append(pos)
                    continue
                z_now = z_arr[di] if di < len(z_arr) else np.nan
                days_held = di - pos['entry_di']
                entry_z = pos['entry_z']
                pos_dir = pos['dir']

                exit_reason = None

                # Mean reversion exit
                if not np.isnan(z_now):
                    if pos_dir == 1 and z_now >= exit_z:
                        exit_reason = 'mean_rev'
                    elif pos_dir == -1 and z_now <= -exit_z:
                        exit_reason = 'mean_rev'

                # Stop loss: z moved further against us
                if exit_reason is None and not np.isnan(z_now):
                    if pos_dir == 1 and z_now < entry_z - 1.5:
                        exit_reason = 'stop_loss'
                    elif pos_dir == -1 and z_now > entry_z + 1.5:
                        exit_reason = 'stop_loss'

                # Time exit
                if exit_reason is None and days_held >= hold_max:
                    exit_reason = 'time'

                if exit_reason:
                    c_down = C[p_down_si, di]
                    c_up = C[p_up_si, di]
                    if np.isnan(c_down) or c_down <= 0:
                        c_down = pos['entry_down']
                    if np.isnan(c_up) or c_up <= 0:
                        c_up = pos['entry_up']

                    mult_down = MULT.get(pos['down_sym'], DEF_MULT)
                    mult_up = MULT.get(pos['up_sym'], DEF_MULT)
                    lots_down = pos['lots_down']
                    lots_up = pos['lots_up']

                    if pos_dir == 1:
                        pnl_down = (c_down - pos['entry_down']) * mult_down * lots_down
                        pnl_up = (pos['entry_up'] - c_up) * mult_up * lots_up
                    else:
                        pnl_down = (pos['entry_down'] - c_down) * mult_down * lots_down
                        pnl_up = (c_up - pos['entry_up']) * mult_up * lots_up

                    entry_val_down = pos['entry_down'] * mult_down * lots_down
                    entry_val_up = pos['entry_up'] * mult_up * lots_up
                    exit_val_down = c_down * mult_down * lots_down
                    exit_val_up = c_up * mult_up * lots_up
                    cost = (entry_val_down + entry_val_up) * COMM + \
                           (exit_val_down + exit_val_up) * COMM

                    total_pnl = pnl_down + pnl_up - cost
                    invested = entry_val_down + entry_val_up
                    pnl_pct = total_pnl / invested * 100 if invested > 0 else 0

                    if pos_dir == 1:
                        cash_return = c_down * mult_down * lots_down - c_up * mult_up * lots_up
                    else:
                        cash_return = -c_down * mult_down * lots_down + c_up * mult_up * lots_up

                    cash += pos['cash_invested'] + cash_return - (exit_val_down + exit_val_up) * COMM

                    trades.append({
                        'pnl_abs': total_pnl,
                        'pnl_pct': pnl_pct,
                        'days': days_held,
                        'di': di,
                        'year': year,
                        'pair': (pos['down_sym'], pos['up_sym']),
                        'pair_label': PAIR_LABEL.get((pos['down_sym'], pos['up_sym']), ''),
                        'dir': pos_dir,
                        'reason': exit_reason,
                        'mode': p_mode,
                        'lb': p_lb,
                    })

                    # Re-entry: if this pair closed and z still extreme, flag for re-entry
                    if reentry and not np.isnan(z_now) and abs(z_now) >= effective_z:
                        reentry_candidates.append((p_down_si, p_up_si,
                                                   pos['down_sym'], pos['up_sym'], z_now))
                else:
                    new_positions.append(pos)

            pair_positions = new_positions

            # --- Check occupied commodities ---
            occupied = set()
            for pos in pair_positions:
                occupied.add(pos['down_si'])
                occupied.add(pos['up_si'])

            # --- Open new pair positions ---
            n_can_open = max_pairs - len(pair_positions)
            if n_can_open <= 0 and not multi_day and not reentry:
                continue

            # Multi-day: if True, allow up to max_pairs total, including from reentry
            if multi_day:
                n_can_open = max(1, max_pairs)  # always check for at least 1

            # Determine which combo to use
            if mode_type == 'fixed':
                use_mode, use_lb = candidate_combos[0]
            else:
                use_mode, use_lb = current_combo

            candidates = []

            # Add re-entry candidates first (high priority)
            if reentry:
                for rc_down_si, rc_up_si, rc_down_sym, rc_up_sym, rc_z in reentry_candidates:
                    if rc_down_si not in occupied and rc_up_si not in occupied:
                        candidates.append((abs(rc_z), rc_down_si, rc_up_si,
                                           rc_down_sym, rc_up_sym, rc_z, True))

            # Then scan all pairs for new signals
            for down_si, up_si, down_sym, up_sym in pair_indices:
                if down_si in occupied or up_si in occupied:
                    continue
                # Skip pairs already in reentry_candidates
                if reentry and any(r[0] == down_si and r[1] == up_si for r in reentry_candidates):
                    continue

                z_arr = z_scores[use_mode].get((down_si, up_si), {}).get(use_lb)
                if z_arr is None:
                    continue
                z_val = z_arr[di] if di < len(z_arr) else np.nan
                if np.isnan(z_val):
                    continue
                if abs(z_val) < effective_z:
                    continue

                candidates.append((abs(z_val), down_si, up_si, down_sym, up_sym, z_val, False))

            if not candidates:
                continue

            candidates.sort(key=lambda x: -x[0])

            # Determine capital per pair based on z-weighted sizing
            opened = 0
            for _, down_si, up_si, down_sym, up_sym, z_val, is_reentry in candidates:
                if not multi_day and opened >= n_can_open:
                    break
                if multi_day and opened >= max_pairs:
                    break
                if down_si in occupied or up_si in occupied:
                    continue

                c_down = C[down_si, di]
                c_up = C[up_si, di]
                if np.isnan(c_down) or c_down <= 0 or np.isnan(c_up) or c_up <= 0:
                    continue

                mult_down = MULT.get(down_sym, DEF_MULT)
                mult_up = MULT.get(up_sym, DEF_MULT)

                # Z-weighted position sizing
                if z_weighted_sizing:
                    abs_z = abs(z_val)
                    if abs_z < 1.0:
                        size_frac = 0.6
                    elif abs_z < 1.5:
                        size_frac = 0.8
                    else:
                        size_frac = 1.0
                else:
                    size_frac = 1.0

                capital_for_pair = cash * size_frac / max(1, max_pairs)
                cash_per_leg = capital_for_pair / 2

                lots_down = int(cash_per_leg / (c_down * mult_down * (1 + COMM)))
                lots_up = int(cash_per_leg / (c_up * mult_up * (1 + COMM)))
                if lots_down <= 0 or lots_up <= 0:
                    continue

                cost_down = c_down * mult_down * lots_down * (1 + COMM)
                cost_up = c_up * mult_up * lots_up * (1 + COMM)
                total_cost = cost_down + cost_up
                if total_cost > cash:
                    scale = cash * 0.95 / total_cost
                    lots_down = max(1, int(lots_down * scale))
                    lots_up = max(1, int(lots_up * scale))
                    cost_down = c_down * mult_down * lots_down * (1 + COMM)
                    cost_up = c_up * mult_up * lots_up * (1 + COMM)
                    total_cost = cost_down + cost_up
                    if total_cost > cash:
                        continue

                if z_val > 0:
                    pos_dir = -1
                else:
                    pos_dir = 1

                cash -= total_cost
                pair_positions.append({
                    'down_si': down_si,
                    'up_si': up_si,
                    'down_sym': down_sym,
                    'up_sym': up_sym,
                    'entry_down': c_down,
                    'entry_up': c_up,
                    'lots_down': lots_down,
                    'lots_up': lots_up,
                    'entry_di': di,
                    'entry_z': z_val,
                    'dir': pos_dir,
                    'cash_invested': total_cost,
                    'mode': use_mode,
                    'lb': use_lb,
                })
                occupied.add(down_si)
                occupied.add(up_si)
                opened += 1

        # Close remaining positions at end
        actual_end = min(end_di, ND) - 1
        for pos in pair_positions:
            p_down_si = pos['down_si']
            p_up_si = pos['up_si']
            c_down = C[p_down_si, actual_end]
            c_up = C[p_up_si, actual_end]
            if np.isnan(c_down) or c_down <= 0:
                c_down = pos['entry_down']
            if np.isnan(c_up) or c_up <= 0:
                c_up = pos['entry_up']

            mult_down = MULT.get(pos['down_sym'], DEF_MULT)
            mult_up = MULT.get(pos['up_sym'], DEF_MULT)
            lots_down = pos['lots_down']
            lots_up = pos['lots_up']

            if pos['dir'] == 1:
                pnl_down = (c_down - pos['entry_down']) * mult_down * lots_down
                pnl_up = (pos['entry_up'] - c_up) * mult_up * lots_up
            else:
                pnl_down = (pos['entry_down'] - c_down) * mult_down * lots_down
                pnl_up = (c_up - pos['entry_up']) * mult_up * lots_up

            entry_val_down = pos['entry_down'] * mult_down * lots_down
            entry_val_up = pos['entry_up'] * mult_up * lots_up
            exit_val_down = c_down * mult_down * lots_down
            exit_val_up = c_up * mult_up * lots_up
            cost = (entry_val_down + entry_val_up) * COMM + \
                   (exit_val_down + exit_val_up) * COMM

            total_pnl = pnl_down + pnl_up - cost
            invested = entry_val_down + entry_val_up
            pnl_pct = total_pnl / invested * 100 if invested > 0 else 0

            if pos['dir'] == 1:
                cash_return = c_down * mult_down * lots_down - c_up * mult_up * lots_up
            else:
                cash_return = -c_down * mult_down * lots_down + c_up * mult_up * lots_up

            cash += pos['cash_invested'] + cash_return - (exit_val_down + exit_val_up) * COMM

            trades.append({
                'pnl_abs': total_pnl,
                'pnl_pct': pnl_pct,
                'days': actual_end - pos['entry_di'],
                'di': actual_end,
                'year': dates[actual_end].year,
                'pair': (pos['down_sym'], pos['up_sym']),
                'pair_label': PAIR_LABEL.get((pos['down_sym'], pos['up_sym']), ''),
                'dir': pos['dir'],
                'reason': 'end',
                'mode': pos['mode'],
                'lb': pos['lb'],
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
        if last_di > first_di:
            days_total = (dates[last_di] - dates[first_di]).days
        else:
            days_total = 365
        yr = max(days_total / 365.25, 0.01)
        ann = ((cash / CASH0) ** (1 / yr) - 1) * 100

        trade_pnls = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
        if len(trade_pnls) > 1:
            rets = np.array(trade_pnls) / float(CASH0)
            sharpe_approx = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
        else:
            sharpe_approx = 0

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

        pair_stats = {}
        for t in trades:
            p = t['pair_label']
            if p not in pair_stats:
                pair_stats[p] = {'n': 0, 'w': 0, 'pnl': 0.0}
            pair_stats[p]['n'] += 1
            if t['pnl_abs'] > 0:
                pair_stats[p]['w'] += 1
            pair_stats[p]['pnl'] += t['pnl_abs']

        mode_usage = {}
        for t in trades:
            m = t.get('mode', '?')
            lb = t.get('lb', '?')
            key = f"{m}_LB{lb}"
            if key not in mode_usage:
                mode_usage[key] = {'n': 0, 'w': 0, 'pnl': 0.0}
            mode_usage[key]['n'] += 1
            if t['pnl_abs'] > 0:
                mode_usage[key]['w'] += 1
            mode_usage[key]['pnl'] += t['pnl_abs']

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
            'sharpe': round(sharpe_approx, 2),
            'cash': round(cash, 0),
            'yearly': year_stats,
            'pair_stats': pair_stats,
            'mode_usage': mode_usage,
            'trades': trades,
        }

    # ================================================================
    # BUILD FULL-PERIOD CONFIGURATIONS
    # ================================================================
    configs = []

    # V61 champion baseline: LOG-biased adaptive, EP40, Z=0.8, MP1, P14
    log_bias_combos = [(SPREAD_LOG, 10), (SPREAD_LOG, 15), (SPREAD_LOG, 20),
                       (SPREAD_RAW, 10), (SPREAD_PCT, 10)]

    # --- Test 1: More pairs (baseline config, varying pair set) ---
    for pidx, pname in [(pair_indices_14, 'P14'), (pair_indices_p2, 'P16'), (pair_indices_p4, 'P18')]:
        for zt in [0.8, 1.0]:
            for mp in [1, 2]:
                name = f"T1_PAIRS_{pname}_Z{zt:.1f}_MP{mp}"
                configs.append({
                    'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0, 'max_pairs': mp,
                    'mode_type': 'adaptive_log_bias', 'eval_period': 40,
                    'candidate_combos': log_bias_combos,
                    'pair_indices': pidx, 'start_year': None, 'end_year': None,
                    'reentry': False, 'z_weighted_sizing': False,
                    'multi_day': False, 'adaptive_z': False,
                    'config_name': name,
                })

    # --- Test 2: Intra-day re-entry ---
    for pidx, pname in [(pair_indices_14, 'P14'), (pair_indices_p4, 'P18')]:
        for zt in [0.8, 1.0, 1.2]:
            for mp in [1, 2]:
                for re in [True, False]:
                    name = f"T2_RE{int(re)}_{pname}_Z{zt:.1f}_MP{mp}"
                    configs.append({
                        'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0, 'max_pairs': mp,
                        'mode_type': 'adaptive_log_bias', 'eval_period': 40,
                        'candidate_combos': log_bias_combos,
                        'pair_indices': pidx, 'start_year': None, 'end_year': None,
                        'reentry': re, 'z_weighted_sizing': False,
                        'multi_day': False, 'adaptive_z': False,
                        'config_name': name,
                    })

    # --- Test 3: Z-score weighted position sizing ---
    for pidx, pname in [(pair_indices_14, 'P14'), (pair_indices_p4, 'P18')]:
        for zt in [0.8, 1.0, 1.2]:
            for mp in [1, 2]:
                for zw in [True, False]:
                    name = f"T3_ZW{int(zw)}_{pname}_Z{zt:.1f}_MP{mp}"
                    configs.append({
                        'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0, 'max_pairs': mp,
                        'mode_type': 'adaptive_log_bias', 'eval_period': 40,
                        'candidate_combos': log_bias_combos,
                        'pair_indices': pidx, 'start_year': None, 'end_year': None,
                        'reentry': False, 'z_weighted_sizing': zw,
                        'multi_day': False, 'adaptive_z': False,
                        'config_name': name,
                    })

    # --- Test 4: Multi-day compounding ---
    for pidx, pname in [(pair_indices_14, 'P14'), (pair_indices_p4, 'P18')]:
        for zt in [0.8, 1.0]:
            for mp in [1, 2]:
                for md in [True, False]:
                    name = f"T4_MD{int(md)}_{pname}_Z{zt:.1f}_MP{mp}"
                    configs.append({
                        'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0, 'max_pairs': mp,
                        'mode_type': 'adaptive_log_bias', 'eval_period': 40,
                        'candidate_combos': log_bias_combos,
                        'pair_indices': pidx, 'start_year': None, 'end_year': None,
                        'reentry': False, 'z_weighted_sizing': False,
                        'multi_day': md, 'adaptive_z': False,
                        'config_name': name,
                    })

    # --- Test 5: Adaptive Z threshold ---
    for pidx, pname in [(pair_indices_14, 'P14'), (pair_indices_p4, 'P18')]:
        for zt in [0.8, 1.0, 1.2]:
            for mp in [1, 2]:
                for az in [True, False]:
                    name = f"T5_AZ{int(az)}_{pname}_Z{zt:.1f}_MP{mp}"
                    configs.append({
                        'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0, 'max_pairs': mp,
                        'mode_type': 'adaptive_log_bias', 'eval_period': 40,
                        'candidate_combos': log_bias_combos,
                        'pair_indices': pidx, 'start_year': None, 'end_year': None,
                        'reentry': False, 'z_weighted_sizing': False,
                        'multi_day': False, 'adaptive_z': az,
                        'config_name': name,
                    })

    # --- Test 6: Combined best improvements ---
    # Combine top features: reentry + z_weighted + more pairs
    for pidx, pname in [(pair_indices_14, 'P14'), (pair_indices_p2, 'P16'), (pair_indices_p4, 'P18')]:
        for zt in [0.8, 1.0]:
            for mp in [1, 2]:
                for re in [True, False]:
                    for zw in [True, False]:
                        # Skip all-false (already tested in T1)
                        if not re and not zw:
                            continue
                        name = f"T6_RE{int(re)}_ZW{int(zw)}_{pname}_Z{zt:.1f}_MP{mp}"
                        configs.append({
                            'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0, 'max_pairs': mp,
                            'mode_type': 'adaptive_log_bias', 'eval_period': 40,
                            'candidate_combos': log_bias_combos,
                            'pair_indices': pidx, 'start_year': None, 'end_year': None,
                            'reentry': re, 'z_weighted_sizing': zw,
                            'multi_day': False, 'adaptive_z': False,
                            'config_name': name,
                        })

    # --- Test 7: Kitchen sink -- combine ALL enhancements ---
    for pidx, pname in [(pair_indices_14, 'P14'), (pair_indices_p4, 'P18')]:
        for zt in [0.8, 1.0]:
            for mp in [1, 2]:
                name = f"T7_ALL_{pname}_Z{zt:.1f}_MP{mp}"
                configs.append({
                    'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0, 'max_pairs': mp,
                    'mode_type': 'adaptive_log_bias', 'eval_period': 40,
                    'candidate_combos': log_bias_combos,
                    'pair_indices': pidx, 'start_year': None, 'end_year': None,
                    'reentry': True, 'z_weighted_sizing': True,
                    'multi_day': True, 'adaptive_z': True,
                    'config_name': name,
                })

    # --- Test 8: Vary eval period with best enhancements ---
    for ep in [30, 40, 60, 80]:
        for pidx, pname in [(pair_indices_14, 'P14'), (pair_indices_p4, 'P18')]:
            for zt in [0.8, 1.0]:
                name = f"T8_EP{ep}_{pname}_Z{zt:.1f}"
                configs.append({
                    'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0, 'max_pairs': 1,
                    'mode_type': 'adaptive_log_bias', 'eval_period': ep,
                    'candidate_combos': log_bias_combos,
                    'pair_indices': pidx, 'start_year': None, 'end_year': None,
                    'reentry': True, 'z_weighted_sizing': True,
                    'multi_day': False, 'adaptive_z': False,
                    'config_name': name,
                })

    # --- Test 9: MP=2 focused -- reentry+zweight to exploit 2 slots ---
    # With MP=2, both reentry AND z-weighted sizing matter more
    for pidx, pname in [(pair_indices_14, 'P14'), (pair_indices_p4, 'P18')]:
        for zt in [0.5, 0.8, 1.0, 1.2]:
            for ep in [40, 60]:
                for md in [True, False]:
                    name = f"T9_MD{int(md)}_{pname}_Z{zt:.1f}_EP{ep}_MP2"
                    configs.append({
                        'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0, 'max_pairs': 2,
                        'mode_type': 'adaptive_log_bias', 'eval_period': ep,
                        'candidate_combos': log_bias_combos,
                        'pair_indices': pidx, 'start_year': None, 'end_year': None,
                        'reentry': True, 'z_weighted_sizing': True,
                        'multi_day': md, 'adaptive_z': False,
                        'config_name': name,
                    })

    # --- Test 10: Hold 2 days with reentry ---
    # Longer hold might let winners run, reentry catches new signals
    for hold in [2, 3]:
        for pidx, pname in [(pair_indices_14, 'P14'), (pair_indices_p4, 'P18')]:
            for zt in [0.8, 1.0]:
                for mp in [1, 2]:
                    name = f"T10_H{hold}_{pname}_Z{zt:.1f}_MP{mp}"
                    configs.append({
                        'z_thresh': zt, 'hold_max': hold, 'exit_z': 0.0, 'max_pairs': mp,
                        'mode_type': 'adaptive_log_bias', 'eval_period': 40,
                        'candidate_combos': log_bias_combos,
                        'pair_indices': pidx, 'start_year': None, 'end_year': None,
                        'reentry': True, 'z_weighted_sizing': False,
                        'multi_day': False, 'adaptive_z': False,
                        'config_name': name,
                    })

    # --- Test 11: Pure LOG fixed with reentry+zweight ---
    # Go back to fixed LOG with the best combos, add enhancements
    for lb in [5, 7, 10, 15]:
        for zt in [0.5, 0.8, 1.0]:
            for re in [True, False]:
                for zw in [True, False]:
                    if not re and not zw:
                        continue
                    name = f"T11_LOG_LB{lb}_Z{zt:.1f}_RE{int(re)}_ZW{int(zw)}"
                    configs.append({
                        'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0, 'max_pairs': 1,
                        'mode_type': 'fixed', 'eval_period': 40,
                        'candidate_combos': [(SPREAD_LOG, lb)],
                        'pair_indices': pair_indices_14, 'start_year': None, 'end_year': None,
                        'reentry': re, 'z_weighted_sizing': zw,
                        'multi_day': False, 'adaptive_z': False,
                        'config_name': name,
                    })

    total_combos = len(configs)
    print(f"\n{'=' * 160}")
    print(f"  FULL-PERIOD PARAMETER SWEEP ({total_combos} configs)")
    print(f"  T1:  More pairs (P14/P16/P18 x Z[0.8,1.0] x MP[1,2])")
    print(f"  T2:  Intra-day re-entry (RE x P14/P18 x Z[0.8,1.0,1.2] x MP[1,2])")
    print(f"  T3:  Z-score weighted sizing (ZW x P14/P18 x Z[0.8,1.0,1.2] x MP[1,2])")
    print(f"  T4:  Multi-day compounding (MD x P14/P18 x Z[0.8,1.0] x MP[1,2])")
    print(f"  T5:  Adaptive Z threshold (AZ x P14/P18 x Z[0.8,1.0,1.2] x MP[1,2])")
    print(f"  T6:  Combined RE+ZW (P14/P16/P18 x Z[0.8,1.0] x MP[1,2])")
    print(f"  T7:  Kitchen sink ALL (P14/P18 x Z[0.8,1.0] x MP[1,2])")
    print(f"  T8:  Eval period sweep with RE+ZW (EP[30,40,60,80] x P14/P18 x Z[0.8,1.0])")
    print(f"  T9:  MP=2 focused (P14/P18 x Z[0.5,0.8,1.0,1.2] x EP[40,60] x MD[0,1])")
    print(f"  T10: Hold 2-3 days with reentry (P14/P18 x Z[0.8,1.0] x MP[1,2])")
    print(f"  T11: Fixed LOG with RE/ZW (LB[5,7,10,15] x Z[0.5,0.8,1.0] x RE/ZW)")
    print(f"{'=' * 160}")

    results = []
    t_sweep_start = time.time()

    for ci, cfg in enumerate(configs):
        r = run_backtest(**cfg)
        if r is not None:
            results.append(r)

        if (ci + 1) % 50 == 0:
            elapsed = time.time() - t_sweep_start
            print(f"  [{ci + 1}/{total_combos}] {len(results)} with results ({elapsed:.1f}s)", flush=True)

    results.sort(key=lambda x: -x['ann'])
    print(f"\n  Sweep complete: {len(results)}/{total_combos} configs ({time.time() - t_sweep_start:.1f}s)",
          flush=True)

    # ================================================================
    # TOP 20 FULL-PERIOD RESULTS
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  TOP 20 FULL-PERIOD RESULTS")
    print(f"{'=' * 160}")
    print(f"  {'#':>2s} | {'Config':50s} | {'Ann':>8s} | {'WR':>5s} | {'N':>5s} | "
          f"{'DD':>6s} | {'PF':>5s} | {'Sharpe':>7s} | {'AvgW':>6s} | {'AvgL':>6s} | "
          f"{'AvgD':>5s} | {'Cash':>12s}")
    print(f"  {'-' * 160}")

    for i, r in enumerate(results[:20]):
        print(f"  {i + 1:2d} | {r['name']:50s} | {r['ann']:+7.1f}% | {r['wr']:4.1f}% | "
              f"{r['n']:5d} | {r['dd']:5.1f}% | {r['pf']:4.2f} | {r['sharpe']:6.2f} | "
              f"{r['avg_win']:+5.2f}% | {r['avg_loss']:5.2f}% | {r['avg_days']:4.1f} | "
              f"{r['cash']:11.0f}")

    # ================================================================
    # PAIR COUNT COMPARISON
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  PAIR COUNT COMPARISON")
    print(f"{'=' * 160}")

    for pset_name in ['P14', 'P16', 'P18']:
        subset = [r for r in results if f'_{pset_name}_' in r['name']]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            best = max(subset, key=lambda x: x['ann'])
            n_pos = sum(1 for r in subset if r['ann'] > 0)
            print(f"  {pset_name}: N={len(subset)}  Avg Ann={avg_ann:+.1f}%  "
                  f"Best Ann={best['ann']:+.1f}%  Positive={n_pos}/{len(subset)}")
            print(f"    Best: {best['name']}  Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  "
                  f"DD={best['dd']:.1f}%  PF={best['pf']:.2f}  Sharpe={best['sharpe']:.2f}")

    # ================================================================
    # REENTRY COMPARISON
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  REENTRY COMPARISON")
    print(f"{'=' * 160}")

    for re_val, re_label in [(True, 'RE_ON'), (False, 'RE_OFF')]:
        subset = [r for r in results if f'_RE{int(re_val)}_' in r['name']]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            best = max(subset, key=lambda x: x['ann'])
            avg_n = np.mean([r['n'] for r in subset])
            print(f"  {re_label}: N={len(subset)}  Avg Ann={avg_ann:+.1f}%  Avg Trades={avg_n:.0f}  "
                  f"Best Ann={best['ann']:+.1f}%")
            print(f"    Best: {best['name']}")

    # ================================================================
    # Z-WEIGHTED SIZING COMPARISON
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  Z-WEIGHTED SIZING COMPARISON")
    print(f"{'=' * 160}")

    for zw_val, zw_label in [(True, 'ZW_ON'), (False, 'ZW_OFF')]:
        subset = [r for r in results if f'_ZW{int(zw_val)}_' in r['name']]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            best = max(subset, key=lambda x: x['ann'])
            print(f"  {zw_label}: N={len(subset)}  Avg Ann={avg_ann:+.1f}%  "
                  f"Best Ann={best['ann']:+.1f}%")
            print(f"    Best: {best['name']}")

    # ================================================================
    # MULTI-DAY COMPARISON
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  MULTI-DAY COMPOUNDING COMPARISON")
    print(f"{'=' * 160}")

    for md_val, md_label in [(True, 'MD_ON'), (False, 'MD_OFF')]:
        subset = [r for r in results if f'_MD{int(md_val)}_' in r['name']]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            best = max(subset, key=lambda x: x['ann'])
            print(f"  {md_label}: N={len(subset)}  Avg Ann={avg_ann:+.1f}%  "
                  f"Best Ann={best['ann']:+.1f}%")
            print(f"    Best: {best['name']}")

    # ================================================================
    # ADAPTIVE Z COMPARISON
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  ADAPTIVE Z THRESHOLD COMPARISON")
    print(f"{'=' * 160}")

    for az_val, az_label in [(True, 'AZ_ON'), (False, 'AZ_OFF')]:
        subset = [r for r in results if f'_AZ{int(az_val)}_' in r['name']]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            best = max(subset, key=lambda x: x['ann'])
            print(f"  {az_label}: N={len(subset)}  Avg Ann={avg_ann:+.1f}%  "
                  f"Best Ann={best['ann']:+.1f}%")
            print(f"    Best: {best['name']}")

    # ================================================================
    # TEST GROUP COMPARISON
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  TEST GROUP COMPARISON (best per test)")
    print(f"{'=' * 160}")

    for test_id in ['T1_', 'T2_', 'T3_', 'T4_', 'T5_', 'T6_', 'T7_', 'T8_', 'T9_', 'T10_', 'T11_']:
        subset = [r for r in results if r['name'].startswith(test_id)]
        if subset:
            best = max(subset, key=lambda x: x['ann'])
            avg_ann = np.mean([r['ann'] for r in subset])
            print(f"  {test_id.strip('_'):3s}: N={len(subset):3d}  Avg={avg_ann:+7.1f}%  "
                  f"Best={best['ann']:+7.1f}%  | {best['name']}")
        else:
            print(f"  {test_id.strip('_'):3s}: no results")

    # ================================================================
    # PER-PAIR STATS (for #1 overall config)
    # ================================================================
    if results:
        best_overall = results[0]
        print(f"\n{'=' * 160}")
        print(f"  PER-PAIR STATS for #1 Config: {best_overall['name']}")
        print(f"  Ann={best_overall['ann']:+.1f}%  WR={best_overall['wr']:.1f}%  "
              f"N={best_overall['n']}  DD={best_overall['dd']:.1f}%  PF={best_overall['pf']:.2f}  "
              f"Sharpe={best_overall['sharpe']:.2f}")
        print(f"{'=' * 160}")
        print(f"  {'Pair':25s} | {'N':>5s} | {'WR':>5s} | {'Abs PnL':>12s} | {'Avg PnL':>10s}")
        print(f"  {'-' * 70}")

        for p in sorted(best_overall['pair_stats'].keys(),
                        key=lambda x: -best_overall['pair_stats'][x]['pnl']):
            ps = best_overall['pair_stats'][p]
            wr_p = ps['w'] / max(ps['n'], 1) * 100
            avg_pnl = ps['pnl'] / max(ps['n'], 1)
            print(f"  {p:25s} | {ps['n']:5d} | {wr_p:4.1f}% | {ps['pnl']:+11.0f} | "
                  f"{avg_pnl:+9.0f}")

        # Mode usage for #1 config
        if best_overall.get('mode_usage'):
            print(f"\n  SPREAD MODE USAGE for #1 config:")
            for mode_key in sorted(best_overall['mode_usage'].keys(),
                                   key=lambda x: -best_overall['mode_usage'][x]['n']):
                mu = best_overall['mode_usage'][mode_key]
                wr_m = mu['w'] / max(mu['n'], 1) * 100
                print(f"    {mode_key:15s}: {mu['n']:5d} trades  WR={wr_m:5.1f}%  "
                      f"PnL={mu['pnl']:+12.0f}")

        # Year-by-year for #1 config
        print(f"\n  Year-by-year breakdown for #1 config:")
        print(f"  {'Year':>6s} | {'N':>5s} | {'WR':>5s} | {'PnL Abs':>12s} | {'PnL %':>8s}")
        print(f"  {'-' * 50}")
        for y in sorted(best_overall['yearly'].keys()):
            ys = best_overall['yearly'][y]
            wr_y = ys['w'] / max(ys['n'], 1) * 100
            print(f"  {y:6d} | {ys['n']:5d} | {wr_y:4.1f}% | {ys['pnl_abs_sum']:+11.0f} | "
                  f"{ys['pnl']:+7.1f}%")

    # ================================================================
    # YEARLY FOR TOP 5
    # ================================================================
    if len(results) >= 2:
        print(f"\n  YEARLY BREAKDOWN FOR TOP 5 CONFIGS:")
        for idx, r in enumerate(results[:5]):
            print(f"\n  #{idx + 1}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.1f}%, "
                  f"DD={r['dd']:.1f}%, Sharpe={r['sharpe']:.2f}, N={r['n']})")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:4d}t  WR={wr_y:5.1f}%  PnL={ys['pnl']:+.1f}%")

    # ================================================================
    # RIGOROUS WALK-FORWARD FOR TOP 5 CONFIGS
    # ================================================================
    top5_for_wf = results[:5]

    print(f"\n{'=' * 160}")
    print(f"  RIGOROUS 6-WINDOW WALK-FORWARD (Top 5 configs)")
    print(f"  Windows: {WF_WINDOWS}")
    print(f"{'=' * 160}")

    wf_all = []
    wf_by_config = {}

    for rank, cfg in enumerate(top5_for_wf):
        cfg_name = cfg['name']
        matching = [c for c in configs if c['config_name'] == cfg_name]
        if not matching:
            print(f"  [{rank + 1}] {cfg_name} -- config not found, SKIP")
            continue

        base_cfg = matching[0]
        print(f"\n  [{rank + 1}] {cfg_name}  (full-period Ann={cfg['ann']:+.1f}%)")

        for train_end, test_year in WF_WINDOWS:
            if test_year not in year_start_di:
                print(f"    Train -{train_end}/Test {test_year}: year not in data, SKIP")
                continue

            wf_name = f"WF_Train-{train_end}_Test-{test_year}_{cfg_name}"
            wf_cfg = dict(base_cfg)
            wf_cfg['start_year'] = test_year
            wf_cfg['end_year'] = test_year
            wf_cfg['config_name'] = wf_name

            r = run_backtest(**wf_cfg)
            if r is not None:
                wf_all.append((cfg_name, train_end, test_year, r))
                if cfg_name not in wf_by_config:
                    wf_by_config[cfg_name] = []
                wf_by_config[cfg_name].append((train_end, test_year, r))
                print(f"    Train -{train_end}/Test {test_year}: Ann={r['ann']:+7.1f}%  "
                      f"WR={r['wr']:5.1f}%  N={r['n']:4d}  DD={r['dd']:5.1f}%  "
                      f"PF={r['pf']:4.2f}  Sharpe={r['sharpe']:6.2f}")
            else:
                print(f"    Train -{train_end}/Test {test_year}: insufficient trades")

    # ================================================================
    # WALK-FORWARD AGGREGATE
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  WALK-FORWARD AGGREGATE (average across all windows per config)")
    print(f"{'=' * 160}")

    wf_avg = []
    for cfg_name, window_results in wf_by_config.items():
        anns = [r['ann'] for _, _, r in window_results]
        wrs = [r['wr'] for _, _, r in window_results]
        ns = [r['n'] for _, _, r in window_results]
        dds = [r['dd'] for _, _, r in window_results]
        pfs = [r['pf'] for _, _, r in window_results]
        sharpe_vals = [r['sharpe'] for _, _, r in window_results]
        n_positive = sum(1 for a in anns if a > 0)

        wf_avg.append({
            'name': cfg_name,
            'avg_ann': np.mean(anns),
            'med_ann': np.median(anns),
            'min_ann': min(anns),
            'max_ann': max(anns),
            'avg_wr': np.mean(wrs),
            'avg_n': np.mean(ns),
            'avg_dd': np.mean(dds),
            'avg_pf': np.mean(pfs),
            'avg_sharpe': np.mean(sharpe_vals),
            'n_positive': n_positive,
            'n_windows': len(window_results),
            'window_details': window_results,
        })

    wf_avg.sort(key=lambda x: -x['avg_ann'])

    print(f"  {'#':>2s} | {'Config':50s} | {'Avg Ann':>8s} | {'Med Ann':>8s} | {'Min Ann':>8s} | "
          f"{'Max Ann':>8s} | {'Avg WR':>6s} | {'Avg N':>6s} | {'Avg DD':>7s} | {'Avg PF':>6s} | "
          f"{'Avg Sh':>6s} | {'Pos/Win':>7s}")
    print(f"  {'-' * 170}")

    for i, w in enumerate(wf_avg):
        print(f"  {i + 1:2d} | {w['name']:50s} | {w['avg_ann']:+7.1f}% | {w['med_ann']:+7.1f}% | "
              f"{w['min_ann']:+7.1f}% | {w['max_ann']:+7.1f}% | {w['avg_wr']:5.1f}% | "
              f"{w['avg_n']:5.0f} | {w['avg_dd']:6.1f}% | {w['avg_pf']:5.2f} | "
              f"{w['avg_sharpe']:5.2f} | {w['n_positive']}/{w['n_windows']}")

    # ================================================================
    # WALK-FORWARD WINDOW-BY-WINDOW DETAIL
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  WALK-FORWARD WINDOW-BY-WINDOW DETAIL")
    print(f"{'=' * 160}")

    for i, w in enumerate(wf_avg):
        print(f"\n  [{i + 1}] {w['name']}:")
        print(f"  {'Train':>9s} | {'Test':>4s} | {'Ann':>8s} | {'WR':>5s} | {'N':>5s} | "
              f"{'DD':>6s} | {'PF':>5s} | {'Sharpe':>7s} | {'AvgW':>6s} | {'AvgL':>6s}")
        print(f"  {'-' * 85}")
        for train_end, test_year, r in sorted(w['window_details'], key=lambda x: x[1]):
            print(f"  -{train_end:4d}    | {test_year:4d} | {r['ann']:+7.1f}% | {r['wr']:4.1f}% | "
                  f"{r['n']:5d} | {r['dd']:5.1f}% | {r['pf']:4.2f} | {r['sharpe']:6.2f} | "
                  f"{r['avg_win']:+5.2f}% | {r['avg_loss']:5.2f}%")

    # ================================================================
    # OVERFITTING CHECK
    # ================================================================
    if wf_avg:
        print(f"\n{'=' * 160}")
        print(f"  OVERFITTING CHECK: Full-Period vs Walk-Forward Correlation")
        print(f"{'=' * 160}")

        full_anns = []
        wf_anns_list = []
        for w in wf_avg:
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
            print(f"  Decay ratio:         {decay:.2f}")

            if corr > 0.5:
                print(f"  -> GOOD: Strong positive correlation, training predicts OOS")
            elif corr > 0.2:
                print(f"  -> MODERATE: Some predictive power")
            else:
                print(f"  -> WARNING: Weak/no correlation, possible overfitting")

        # WF positive rate
        all_wf_anns = [r['ann'] for _, _, _, r in wf_all]
        n_pos_wf = sum(1 for a in all_wf_anns if a > 0)
        print(f"\n  Overall WF positive rate: {n_pos_wf}/{len(all_wf_anns)} "
              f"({n_pos_wf / len(all_wf_anns) * 100:.0f}%)")

        if wf_all:
            best_single = max(wf_all, key=lambda x: x[3]['ann'])
            worst_single = min(wf_all, key=lambda x: x[3]['ann'])
            print(f"  Best single window OOS:  Test {best_single[2]} = "
                  f"{best_single[3]['ann']:+.1f}% ({best_single[0][:50]})")
            print(f"  Worst single window OOS: Test {worst_single[2]} = "
                  f"{worst_single[3]['ann']:+.1f}% ({worst_single[0][:50]})")

    # ================================================================
    # PAIR PROFITABILITY ACROSS TOP 20
    # ================================================================
    if results:
        print(f"\n{'=' * 160}")
        print(f"  PAIR PROFITABILITY ACROSS TOP 20 CONFIGS")
        print(f"{'=' * 160}")

        pair_summary = {}
        for r in results[:20]:
            for p, ps in r['pair_stats'].items():
                if p not in pair_summary:
                    pair_summary[p] = {'n': 0, 'w': 0, 'pnl': 0.0}
                pair_summary[p]['n'] += ps['n']
                pair_summary[p]['w'] += ps['w']
                pair_summary[p]['pnl'] += ps['pnl']

        for p in sorted(pair_summary.keys(), key=lambda x: -pair_summary[x]['pnl']):
            ps = pair_summary[p]
            wr_p = ps['w'] / max(ps['n'], 1) * 100
            print(f"  {p:25s}: {ps['n']:5d} trades  WR={wr_p:5.1f}%  Total Abs={ps['pnl']:+12.0f}")

    # ================================================================
    # FEATURE CONTRIBUTION ANALYSIS
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  FEATURE CONTRIBUTION ANALYSIS")
    print(f"{'=' * 160}")

    # Compare each feature ON vs OFF (controlled: same base config)
    features = [
        ('Reentry', '_RE1_', '_RE0_'),
        ('Z-Weight', '_ZW1_', '_ZW0_'),
        ('Multi-Day', '_MD1_', '_MD0_'),
        ('Adaptive-Z', '_AZ1_', '_AZ0_'),
    ]

    for feat_name, on_tag, off_tag in features:
        on_results = [r for r in results if on_tag in r['name']]
        off_results = [r for r in results if off_tag in r['name']]
        if on_results and off_results:
            on_avg = np.mean([r['ann'] for r in on_results])
            off_avg = np.mean([r['ann'] for r in off_results])
            on_best = max(r['ann'] for r in on_results)
            off_best = max(r['ann'] for r in off_results)
            delta_avg = on_avg - off_avg
            delta_best = on_best - off_best
            print(f"  {feat_name:12s}: ON avg={on_avg:+7.1f}% (best={on_best:+7.1f}%)  "
                  f"OFF avg={off_avg:+7.1f}% (best={off_best:+7.1f}%)  "
                  f"Delta avg={delta_avg:+7.1f}%  Delta best={delta_best:+7.1f}%")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 160}")

    if results:
        print(f"\n  Full-period best: {results[0]['name']}")
        print(f"    Ann={results[0]['ann']:+.1f}%  WR={results[0]['wr']:.1f}%  N={results[0]['n']}  "
              f"DD={results[0]['dd']:.1f}%  PF={results[0]['pf']:.2f}  Sharpe={results[0]['sharpe']:.2f}")

    if wf_avg:
        print(f"\n  Walk-forward best: {wf_avg[0]['name']}")
        print(f"    WF Avg Ann={wf_avg[0]['avg_ann']:+.1f}%  WF Med Ann={wf_avg[0]['med_ann']:+.1f}%  "
              f"Min={wf_avg[0]['min_ann']:+.1f}%  Max={wf_avg[0]['max_ann']:+.1f}%  "
              f"Pos/Win={wf_avg[0]['n_positive']}/{wf_avg[0]['n_windows']}")

        n_all_positive = sum(1 for w in wf_avg if w['n_positive'] == w['n_windows'])
        print(f"\n  Of top 5 WF configs, {n_all_positive} are positive in ALL test windows")

    # Comparison to baselines
    print(f"\n  Baseline comparison:")
    print(f"    V61 champion:         +324.3% (LOG-biased adaptive, EP40, Z=0.8, MP1, 14 pairs)")
    if results:
        print(f"    V62 best full-period: {results[0]['ann']:+.1f}%")
    if wf_avg:
        print(f"    V62 best WF avg:      {wf_avg[0]['avg_ann']:+.1f}%")
        print(f"    V62 best WF min:      {wf_avg[0]['min_ann']:+.1f}% (worst single window)")
        print(f"    V62 best WF max:      {wf_avg[0]['max_ann']:+.1f}% (best single window)")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 160)


if __name__ == '__main__':
    main()
