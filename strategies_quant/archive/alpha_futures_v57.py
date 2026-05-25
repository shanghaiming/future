"""
Alpha Futures V57 -- Per-Pair Adaptive Spread Mode + Parameter Optimization
============================================================================
V55 champion: +307.3%, Sharpe 2.17, WF avg +355.7%
V55 uses GLOBAL mode selection -- same spread mode for all pairs.
Hypothesis: Each pair may prefer different modes/lookbacks.

Tests:
  1. per_pair_adaptive  -- each pair independently selects best (mode, LB) combo
  2. per_pair_fixed     -- offline backtest finds fixed best mode per pair
  3. per_pair_lookback  -- each pair selects its own lookback (mode global)
  4. hybrid             -- global adaptive mode + per-pair lookback
  5. wr_weighted        -- weight pairs by recent WR for signal priority

Configs: ~150
  mode_type:     [global_adaptive, per_pair_adaptive, per_pair_fixed, hybrid]
  eval_period:   [40, 60]
  z_threshold:   [0.8, 1.0, 1.2]
  lookback_opts: [[7,10], [5,7,10], [7,10,15], [5,7,10,15,20]]
  Walk-forward for best (2022, 2023, 2024)

Print: top 20 full-period, top 10 walk-forward, per-pair mode/lookback stats.
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

PAIRS = [
    ('rbfi', 'ifi'), ('hcfi', 'ifi'), ('hcfi', 'rbfi'),
    ('jfi', 'jmfi'), ('mafi', 'scfi'), ('fufi', 'scfi'),
    ('bfi', 'scfi'), ('mfi', 'afi'), ('yfi', 'afi'),
    ('pfi', 'yfi'), ('ppfi', 'mafi'), ('vfi', 'mafi'),
    ('egfi', 'mafi'), ('cfi', 'csfi'),
]

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
}

SPREAD_RAW = 'raw'
SPREAD_PCT = 'pct'
SPREAD_LOG = 'log'
ALL_MODES = [SPREAD_RAW, SPREAD_PCT, SPREAD_LOG]
ALL_LOOKBACKS = [5, 7, 10, 15, 20]


def main():
    t_start = time.time()
    print("=" * 150)
    print("Alpha Futures V57 -- Per-Pair Adaptive Spread Mode + Parameter Optimization")
    print("V55 champion: +307.3%, Sharpe 2.17, WF avg +355.7% (global adaptive mode)")
    print("Hypothesis: Per-pair mode/LB selection outperforms global mode selection")
    print("=" * 150)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()
    sym_to_si = {syms[si]: si for si in range(NS)}

    # Build pair indices
    pair_indices = []
    for down_sym, up_sym in PAIRS:
        down_si = sym_to_si.get(down_sym, -1)
        up_si = sym_to_si.get(up_sym, -1)
        if down_si >= 0 and up_si >= 0:
            pair_indices.append((down_si, up_si, down_sym, up_sym))
        else:
            print(f"  WARNING: pair ({down_sym}, {up_sym}) not found "
                  f"(down_si={down_si}, up_si={up_si})")

    print(f"  {NS} commodities, {ND} days, {len(pair_indices)} active pairs")

    # ================================================================
    # PRECOMPUTE SPREADS AND Z-SCORES FOR ALL MODES x LOOKBACKS
    # ================================================================
    print("\n[Signals] Precomputing spreads and z-scores for all modes x lookbacks...", flush=True)
    t0 = time.time()

    z_scores = {m: {} for m in ALL_MODES}
    spreads_all = {m: {} for m in ALL_MODES}

    for down_si, up_si, down_sym, up_sym in pair_indices:
        for mode in ALL_MODES:
            key = (down_si, up_si)
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

            spreads_all[mode][key] = spread
            z_scores[mode][key] = {}

            for lb in ALL_LOOKBACKS:
                z = np.full(ND, np.nan)
                for di in range(lb, ND):
                    window = spread[di - lb:di]
                    valid = window[~np.isnan(window)]
                    if len(valid) >= lb * 0.8:
                        m_val = np.mean(valid)
                        s_val = np.std(valid, ddof=1)
                        if s_val > 1e-10:
                            z[di] = (spread[di] - m_val) / s_val
                z_scores[mode][key][lb] = z

    print(f"  All z-scores precomputed ({time.time() - t0:.1f}s)", flush=True)

    # ================================================================
    # PRECOMPUTE HYPOTHETICAL PER-PAIR COMBO RETURNS
    # ================================================================
    # For each pair, independently compute hypothetical returns for each
    # (mode, lb, z_thresh) combo. This enables per-pair adaptive selection.
    print("\n[Signals] Precomputing per-pair hypothetical combo returns...", flush=True)
    t1 = time.time()

    # pair_combo_daily_return[(pair_key, mode, lb, zt)][di] = pnl_pct or nan
    pair_combo_daily_return = {}
    pair_combo_keys = []

    for zt in [0.8, 1.0, 1.2]:
        for mode in ALL_MODES:
            for lb in ALL_LOOKBACKS:
                combo_key = (mode, lb, zt)
                pair_combo_keys.append(combo_key)
                # Per-pair daily returns
                for down_si, up_si, down_sym, up_sym in pair_indices:
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

    # Also compute global combo daily returns (average across pairs, same as V55)
    global_combo_daily_return = {}
    for zt in [0.8, 1.0, 1.2]:
        for mode in ALL_MODES:
            for lb in ALL_LOOKBACKS:
                combo_key = (mode, lb, zt)
                daily_ret = np.full(ND, np.nan)
                for di in range(MIN_TRAIN + 1, ND):
                    pair_rets = []
                    for down_si, up_si, down_sym, up_sym in pair_indices:
                        pk = (down_si, up_si, down_sym, up_sym)
                        pr = pair_combo_daily_return.get((pk, combo_key))
                        if pr is not None and not np.isnan(pr[di]):
                            pair_rets.append(pr[di])
                    if pair_rets:
                        daily_ret[di] = np.mean(pair_rets)
                global_combo_daily_return[combo_key] = daily_ret

    print(f"  Per-pair hypothetical returns precomputed ({time.time() - t1:.1f}s)", flush=True)

    # ================================================================
    # PRECOMPUTE PER-PAIR FIXED BEST MODE
    # ================================================================
    # For each pair, find the fixed best (mode, lb) combo based on full-sample return.
    # This is used by 'per_pair_fixed' mode_type.
    print("\n[Signals] Computing per-pair fixed best modes...", flush=True)
    t2 = time.time()

    # fixed_best_per_pair[pair_key] = (mode, lb, zt) based on full-sample sum return
    # We compute this for each z_thresh separately
    fixed_best_per_pair = {}  # zt -> pair_key -> (mode, lb)
    for zt in [0.8, 1.0, 1.2]:
        fixed_best_per_pair[zt] = {}
        for down_si, up_si, down_sym, up_sym in pair_indices:
            pk = (down_si, up_si, down_sym, up_sym)
            best_combo = None
            best_score = -1e18
            for mode in ALL_MODES:
                for lb in ALL_LOOKBACKS:
                    combo_key = (mode, lb, zt)
                    pr = pair_combo_daily_return.get((pk, combo_key))
                    if pr is None:
                        continue
                    valid = pr[~np.isnan(pr)]
                    if len(valid) >= 5:
                        score = np.nansum(valid)
                    else:
                        score = -1e10
                    if score > best_score:
                        best_score = score
                        best_combo = (mode, lb)
            if best_combo:
                fixed_best_per_pair[zt][pk] = best_combo

    print(f"  Per-pair fixed best modes computed ({time.time() - t2:.1f}s)", flush=True)

    # Print per-pair fixed best mode stats
    print("\n  Per-pair fixed best modes (by z_thresh):")
    for zt in [0.8, 1.0, 1.2]:
        print(f"\n    Z={zt:.1f}:")
        for down_si, up_si, down_sym, up_sym in pair_indices:
            pk = (down_si, up_si, down_sym, up_sym)
            best = fixed_best_per_pair[zt].get(pk)
            if best:
                label = PAIR_LABEL.get((down_sym, up_sym), f"{down_sym}/{up_sym}")
                print(f"      {label:25s}: {best[0]:4s} LB{best[1]:2d}")

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(z_thresh=1.0, hold_max=1, exit_z=0.0, max_pairs=1,
                     mode_type='global_adaptive',
                     eval_period=60,
                     lookback_options=None,
                     weight_type='equal',
                     wf_split_year=None,
                     wf_end_year=None,
                     config_name=""):
        """
        mode_type controls how spread mode/lookback is selected:
          'global_adaptive'  -- V55: all pairs use same (mode, LB), selected globally
          'per_pair_adaptive'-- each pair independently selects its best (mode, LB)
          'per_pair_fixed'   -- each pair uses its pre-computed fixed best (mode, LB)
          'hybrid'           -- global adaptive mode, per-pair adaptive lookback

        weight_type controls pair priority when multiple pairs signal:
          'equal'    -- sort by z-score magnitude (current behavior)
          'wr_weighted' -- sort by recent WR, then z-score
        """
        if lookback_options is None:
            lookback_options = [7, 10]

        pidx = pair_indices

        # Determine which combos to consider
        combos = [(m, lb) for m in ALL_MODES for lb in lookback_options]

        cash = float(CASH0)
        trades = []
        pair_positions = []

        # State for adaptive selection
        current_global_combo = combos[0]

        # Per-pair state
        pair_combo_state = {}
        for down_si, up_si, down_sym, up_sym in pidx:
            pk = (down_si, up_si, down_sym, up_sym)
            pair_combo_state[pk] = combos[0]

        # WR tracking for wr_weighted: rolling WR per pair
        pair_wr_tracker = {}
        for down_si, up_si, down_sym, up_sym in pidx:
            pk = (down_si, up_si, down_sym, up_sym)
            pair_wr_tracker[pk] = {'wins': 0, 'total': 0}

        # Date range for walk-forward
        start_di = MIN_TRAIN
        end_di = ND
        if wf_split_year is not None:
            for di in range(MIN_TRAIN, ND):
                if dates[di].year >= wf_split_year:
                    start_di = di
                    break
        if wf_end_year is not None:
            for di in range(start_di, ND):
                if dates[di].year > wf_end_year:
                    end_di = di
                    break

        for di in range(start_di, end_di):
            year = dates[di].year

            # --- Adaptive evaluation every eval_period days ---
            if di > start_di:
                days_since_start = di - start_di
                if days_since_start % eval_period == 0 and days_since_start >= eval_period:
                    if mode_type == 'global_adaptive':
                        # Global: pick best combo across all pairs
                        best_combo = combos[0]
                        best_score = -1e18
                        for c in combos:
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
                        current_global_combo = best_combo

                    elif mode_type == 'per_pair_adaptive':
                        # Per-pair: each pair picks its own best combo
                        for down_si, up_si, down_sym, up_sym in pidx:
                            pk = (down_si, up_si, down_sym, up_sym)
                            best_combo = combos[0]
                            best_score = -1e18
                            for c in combos:
                                combo_key = (c[0], c[1], z_thresh)
                                pr = pair_combo_daily_return.get((pk, combo_key))
                                if pr is None:
                                    continue
                                window = pr[max(start_di, di - eval_period):di]
                                valid = window[~np.isnan(window)]
                                if len(valid) >= 2:
                                    score = np.nansum(valid)
                                else:
                                    score = -1e10
                                if score > best_score:
                                    best_score = score
                                    best_combo = c
                            pair_combo_state[pk] = best_combo

                    elif mode_type == 'hybrid':
                        # Global mode selection first
                        best_mode = ALL_MODES[0]
                        best_mode_score = -1e18
                        for mode in ALL_MODES:
                            mode_score = 0
                            for lb in lookback_options:
                                combo_key = (mode, lb, z_thresh)
                                daily_ret = global_combo_daily_return.get(combo_key)
                                if daily_ret is None:
                                    continue
                                window = daily_ret[max(start_di, di - eval_period):di]
                                valid = window[~np.isnan(window)]
                                if len(valid) >= 3:
                                    mode_score += np.nansum(valid)
                            if mode_score > best_mode_score:
                                best_mode_score = mode_score
                                best_mode = mode

                        # Per-pair lookback selection within the chosen mode
                        for down_si, up_si, down_sym, up_sym in pidx:
                            pk = (down_si, up_si, down_sym, up_sym)
                            best_lb = lookback_options[0]
                            best_lb_score = -1e18
                            for lb in lookback_options:
                                combo_key = (best_mode, lb, z_thresh)
                                pr = pair_combo_daily_return.get((pk, combo_key))
                                if pr is None:
                                    continue
                                window = pr[max(start_di, di - eval_period):di]
                                valid = window[~np.isnan(window)]
                                if len(valid) >= 2:
                                    score = np.nansum(valid)
                                else:
                                    score = -1e10
                                if score > best_lb_score:
                                    best_lb_score = score
                                    best_lb = lb
                            pair_combo_state[pk] = (best_mode, best_lb)

            # --- Manage existing pair positions ---
            new_positions = []
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

                if not np.isnan(z_now):
                    if pos_dir == 1 and z_now >= exit_z:
                        exit_reason = 'mean_rev'
                    elif pos_dir == -1 and z_now <= -exit_z:
                        exit_reason = 'mean_rev'

                if exit_reason is None and not np.isnan(z_now):
                    if pos_dir == 1 and z_now < entry_z - 1.5:
                        exit_reason = 'stop_loss'
                    elif pos_dir == -1 and z_now > entry_z + 1.5:
                        exit_reason = 'stop_loss'

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

                    # Track WR for wr_weighted
                    pk = (pos['down_si'], pos['up_si'], pos['down_sym'], pos['up_sym'])
                    pair_wr_tracker[pk]['total'] += 1
                    if total_pnl > 0:
                        pair_wr_tracker[pk]['wins'] += 1

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
            if n_can_open <= 0:
                continue

            capital_per_pair = cash / max(1, max_pairs)

            candidates = []
            for down_si, up_si, down_sym, up_sym in pidx:
                if down_si in occupied or up_si in occupied:
                    continue

                pk = (down_si, up_si, down_sym, up_sym)

                # Determine which combo to use for this pair
                if mode_type == 'global_adaptive':
                    use_combo = current_global_combo
                elif mode_type == 'per_pair_adaptive':
                    use_combo = pair_combo_state.get(pk, combos[0])
                elif mode_type == 'per_pair_fixed':
                    use_combo = fixed_best_per_pair.get(z_thresh, {}).get(pk, ('raw', 10))
                elif mode_type == 'hybrid':
                    use_combo = pair_combo_state.get(pk, combos[0])
                else:
                    use_combo = current_global_combo

                use_mode, use_lb = use_combo
                z_arr = z_scores[use_mode].get((down_si, up_si), {}).get(use_lb)
                if z_arr is None:
                    continue
                z_val = z_arr[di] if di < len(z_arr) else np.nan
                if np.isnan(z_val):
                    continue
                if abs(z_val) < z_thresh:
                    continue

                # Compute priority score
                if weight_type == 'wr_weighted':
                    wr_t = pair_wr_tracker[pk]
                    recent_wr = wr_t['wins'] / max(wr_t['total'], 1)
                    # Priority: WR-weighted, with z-score as tiebreaker
                    priority = recent_wr * 100 + abs(z_val) * 0.1
                else:
                    priority = abs(z_val)

                candidates.append((priority, down_si, up_si, down_sym, up_sym,
                                   z_val, use_mode, use_lb))

            if not candidates:
                continue

            candidates.sort(key=lambda x: -x[0])

            for _, down_si, up_si, down_sym, up_sym, z_val, use_mode, use_lb in candidates[:n_can_open]:
                c_down = C[down_si, di]
                c_up = C[up_si, di]
                if np.isnan(c_down) or c_down <= 0 or np.isnan(c_up) or c_up <= 0:
                    continue

                mult_down = MULT.get(down_sym, DEF_MULT)
                mult_up = MULT.get(up_sym, DEF_MULT)

                cash_per_leg = capital_per_pair / 2
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

        # Close remaining positions at end
        for pos in pair_positions:
            p_down_si = pos['down_si']
            p_up_si = pos['up_si']
            c_down = C[p_down_si, end_di - 1]
            c_up = C[p_up_si, end_di - 1]
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
                'days': (end_di - 1) - pos['entry_di'],
                'di': end_di - 1,
                'year': dates[end_di - 1].year,
                'pair': (pos['down_sym'], pos['up_sym']),
                'pair_label': PAIR_LABEL.get((pos['down_sym'], pos['up_sym']), ''),
                'dir': pos['dir'],
                'reason': 'end',
                'mode': pos['mode'],
                'lb': pos['lb'],
            })

        if len(trades) < 5:
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

        reasons = {}
        for t in trades:
            r = t['reason']
            if r not in reasons:
                reasons[r] = {'n': 0, 'w': 0, 'pnl': 0.0, 'pnl_pct_sum': 0.0}
            reasons[r]['n'] += 1
            if t['pnl_abs'] > 0:
                reasons[r]['w'] += 1
            reasons[r]['pnl'] += t['pnl_abs']
            reasons[r]['pnl_pct_sum'] += t['pnl_pct']

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

        # Per-pair mode breakdown for detailed analysis
        pair_mode_usage = {}
        for t in trades:
            p = t['pair_label']
            m = t.get('mode', '?')
            lb = t.get('lb', '?')
            mk = f"{m}_LB{lb}"
            if p not in pair_mode_usage:
                pair_mode_usage[p] = {}
            if mk not in pair_mode_usage[p]:
                pair_mode_usage[p][mk] = {'n': 0, 'w': 0, 'pnl': 0.0}
            pair_mode_usage[p][mk]['n'] += 1
            if t['pnl_abs'] > 0:
                pair_mode_usage[p][mk]['w'] += 1
            pair_mode_usage[p][mk]['pnl'] += t['pnl_abs']

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
            'reasons': reasons,
            'yearly': year_stats,
            'pair_stats': pair_stats,
            'mode_usage': mode_usage,
            'pair_mode_usage': pair_mode_usage,
            'trades': trades,
        }

    # ================================================================
    # PARAMETER SWEEP (~150 configs)
    # ================================================================
    print("\n[Backtest] Running parameter sweep...", flush=True)
    results = []
    configs = []

    mode_types = ['global_adaptive', 'per_pair_adaptive', 'per_pair_fixed', 'hybrid']
    eval_periods = [40, 60]
    z_thresholds = [0.8, 1.0, 1.2]
    lb_options_list = [
        ('LB7_10', [7, 10]),
        ('LB5_7_10', [5, 7, 10]),
        ('LB7_10_15', [7, 10, 15]),
        ('LB5_7_10_15_20', [5, 7, 10, 15, 20]),
    ]
    weight_types = ['equal', 'wr_weighted']

    for mt in mode_types:
        for ep in eval_periods:
            for zt in z_thresholds:
                for lb_name, lb_opts in lb_options_list:
                    for wt in weight_types:
                        name = f"MT_{mt}_EP{ep}_Z{zt:.1f}_{lb_name}_W_{wt}"
                        configs.append({
                            'z_thresh': zt,
                            'hold_max': 1,
                            'exit_z': 0.0,
                            'max_pairs': 1,
                            'mode_type': mt,
                            'eval_period': ep,
                            'lookback_options': lb_opts,
                            'weight_type': wt,
                            'wf_split_year': None,
                            'wf_end_year': None,
                            'config_name': name,
                        })

    print(f"  {len(configs)} full-period configurations", flush=True)

    for ci, cfg in enumerate(configs):
        r = run_backtest(**cfg)
        if r is not None:
            results.append(r)
            if r['ann'] > 100:
                print(f"  {r['name']:60s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                      f"N {r['n']:5d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                      f"Sh {r['sharpe']:5.2f} | AvgD {r['avg_days']:.1f}")
        if (ci + 1) % 50 == 0:
            print(f"  [{ci + 1}/{len(configs)}] {len(results)} configs with results", flush=True)

    print(f"  Full-period sweep done: {len(results)} configs with results", flush=True)

    # ================================================================
    # WALK-FORWARD FOR TOP CONFIGS
    # ================================================================
    print(f"\n[Walk-Forward] Rigorous walk-forward for top configs...", flush=True)
    full_results = sorted([r for r in results], key=lambda x: -x['ann'])

    # Select top 10 unique configs for walk-forward
    wf_candidates = []
    seen_params = set()
    for r in full_results:
        if r['name'] in seen_params:
            continue
        seen_params.add(r['name'])
        wf_candidates.append(r)
        if len(wf_candidates) >= 10:
            break

    wf_results = []
    wf_configs_run = 0

    for r in wf_candidates:
        matching = [c for c in configs if c['config_name'] == r['name']]
        if not matching:
            continue
        cfg = matching[0]

        for wf_test_year, wf_label in [(2022, 'WF2022'), (2023, 'WF2023'), (2024, 'WF2024')]:
            wf_name = f"{r['name']}_{wf_label}"
            wf_cfg = dict(cfg)
            wf_cfg['wf_split_year'] = wf_test_year
            wf_cfg['wf_end_year'] = wf_test_year + 1 if wf_test_year < 2024 else None
            wf_cfg['config_name'] = wf_name
            wr = run_backtest(**wf_cfg)
            if wr is not None:
                wf_results.append(wr)
            wf_configs_run += 1

        if wf_configs_run % 10 == 0:
            print(f"  [WF {wf_configs_run} configs tested] {len(wf_results)} with results", flush=True)

    print(f"  {wf_configs_run} walk-forward configs tested, {len(wf_results)} with results", flush=True)

    # ================================================================
    # RESULTS
    # ================================================================
    all_results = results + wf_results
    full_r = [r for r in all_results if '_WF' not in r['name']]
    wf_only = [r for r in all_results if '_WF' in r['name']]
    full_r.sort(key=lambda x: -x['ann'])
    wf_only.sort(key=lambda x: -x['ann'])

    # --- TOP 20 FULL-PERIOD ---
    print(f"\n{'=' * 160}")
    print(f"  TOP 20 FULL-PERIOD RESULTS (sorted by annual return)")
    print(f"{'=' * 160}")
    hdr = (f"  {'Config':60s} | {'Ann':>7s} | {'WR':>5s} | {'N':>5s} | {'DD':>6s} | "
           f"{'PF':>4s} | {'Sharpe':>6s} | {'AvgW':>7s} | {'AvgL':>6s} | {'AvgD':>4s}")
    print(hdr)
    print(f"  {'-' * 155}")
    for r in full_r[:20]:
        print(f"  {r['name']:60s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:5d} | {r['dd']:6.1f}% | {r['pf']:4.2f} | "
              f"{r['sharpe']:6.2f} | {r['avg_win']:+6.2f}% | {r['avg_loss']:5.2f}% | "
              f"{r['avg_days']:4.1f}")

    # --- TOP 10 WALK-FORWARD ---
    if wf_only:
        print(f"\n{'=' * 160}")
        print(f"  TOP 10 WALK-FORWARD RESULTS (out-of-sample)")
        print(f"{'=' * 160}")
        for r in wf_only[:10]:
            print(f"  {r['name']:70s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                  f"Sh {r['sharpe']:6.2f} | AvgD {r['avg_days']:.1f}")

    # --- WALK-FORWARD YEARLY SUMMARY ---
    if wf_only:
        print(f"\n  WALK-FORWARD YEARLY SUMMARY:")
        print(f"  {'Config':60s} | {'WF2022':>8s} | {'WF2023':>8s} | {'WF2024':>8s} | {'Avg WF':>8s}")
        print(f"  {'-' * 120}")

        wf_by_config = {}
        for r in wf_only:
            base_name = r['name'].rsplit('_WF', 1)[0]
            wf_year = r['name'].split('_WF')[-1]
            if base_name not in wf_by_config:
                wf_by_config[base_name] = {}
            wf_by_config[base_name][wf_year] = r

        wf_summary = []
        for base_name, years in wf_by_config.items():
            anns = [years[y]['ann'] for y in years if years.get(y)]
            avg_ann = np.mean(anns) if anns else 0
            wf_summary.append((base_name, years, avg_ann))
        wf_summary.sort(key=lambda x: -x[2])

        for base_name, years, avg_ann in wf_summary[:20]:
            parts = []
            for y_label in ['WF2022', 'WF2023', 'WF2024']:
                if y_label in years:
                    parts.append(f"{years[y_label]['ann']:+7.1f}%")
                else:
                    parts.append(f"{'N/A':>8s}")
            print(f"  {base_name:60s} | {' | '.join(parts)} | {avg_ann:+7.1f}%")

    # --- MODE TYPE COMPARISON ---
    print(f"\n{'=' * 160}")
    print(f"  MODE TYPE COMPARISON")
    print(f"{'=' * 160}")
    print(f"\n  {'Mode Type':25s} | {'N Cfgs':>6s} | {'Avg Ann':>8s} | {'Best Ann':>9s} | "
          f"{'Avg WR':>7s} | {'Avg Sharpe':>10s} | {'Avg DD':>7s} | {'Avg PF':>6s} | "
          f"Best Config")
    print(f"  {'-' * 145}")

    for mt in mode_types:
        subset = [r for r in full_r if f'MT_{mt}_' in r['name']]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            best_ann = max(r['ann'] for r in subset)
            avg_wr = np.mean([r['wr'] for r in subset])
            avg_sh = np.mean([r['sharpe'] for r in subset])
            avg_dd = np.mean([r['dd'] for r in subset])
            avg_pf = np.mean([r['pf'] for r in subset])
            best = max(subset, key=lambda x: x['ann'])
            print(f"  {mt:25s} | {len(subset):6d} | {avg_ann:+7.1f}% | {best_ann:+8.1f}% | "
                  f"{avg_wr:5.1f}% | {avg_sh:10.2f} | {avg_dd:6.1f}% | {avg_pf:5.2f} | "
                  f"{best['name'][:50]}")

    # --- WEIGHT TYPE COMPARISON ---
    print(f"\n  WEIGHT TYPE COMPARISON:")
    print(f"  {'Weight':15s} | {'N':>5s} | {'Avg Ann':>8s} | {'Best Ann':>9s} | {'Avg WR':>7s} | {'Avg Sharpe':>10s}")
    print(f"  {'-' * 70}")
    for wt in weight_types:
        subset = [r for r in full_r if f'_W_{wt}' in r['name']]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            best_ann = max(r['ann'] for r in subset)
            avg_wr = np.mean([r['wr'] for r in subset])
            avg_sh = np.mean([r['sharpe'] for r in subset])
            print(f"  {wt:15s} | {len(subset):5d} | {avg_ann:+7.1f}% | {best_ann:+8.1f}% | "
                  f"{avg_wr:5.1f}% | {avg_sh:10.2f}")

    # --- EVAL PERIOD COMPARISON ---
    print(f"\n  EVAL PERIOD COMPARISON:")
    print(f"  {'EP':>4s} | {'N':>5s} | {'Avg Ann':>7s} | {'Best Ann':>8s} | {'Avg WR':>6s} | {'Avg Sharpe':>10s}")
    print(f"  {'-' * 60}")
    for ep in eval_periods:
        subset = [r for r in full_r if f'_EP{ep}_' in r['name']]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            best_ann = max(r['ann'] for r in subset)
            avg_wr = np.mean([r['wr'] for r in subset])
            avg_sh = np.mean([r['sharpe'] for r in subset])
            print(f"  {ep:4d} | {len(subset):5d} | {avg_ann:+6.1f}% | {best_ann:+7.1f}% | "
                  f"{avg_wr:5.1f}% | {avg_sh:10.2f}")

    # --- LOOKBACK OPTIONS COMPARISON ---
    print(f"\n  LOOKBACK OPTIONS COMPARISON:")
    print(f"  {'LB Set':15s} | {'N':>5s} | {'Avg Ann':>8s} | {'Best Ann':>9s} | {'Avg WR':>7s} | {'Avg Sharpe':>10s}")
    print(f"  {'-' * 70}")
    for lb_name, lb_opts in lb_options_list:
        subset = [r for r in full_r if f'_{lb_name}_' in r['name']]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            best_ann = max(r['ann'] for r in subset)
            avg_wr = np.mean([r['wr'] for r in subset])
            avg_sh = np.mean([r['sharpe'] for r in subset])
            print(f"  {lb_name:15s} | {len(subset):5d} | {avg_ann:+7.1f}% | {best_ann:+8.1f}% | "
                  f"{avg_wr:5.1f}% | {avg_sh:10.2f}")

    # --- Z-THRESHOLD COMPARISON ---
    print(f"\n  Z-THRESHOLD COMPARISON:")
    print(f"  {'Z':>4s} | {'N':>5s} | {'Avg Ann':>7s} | {'Best Ann':>8s} | {'Avg WR':>6s} | {'Avg Sharpe':>10s}")
    print(f"  {'-' * 60}")
    for zt in z_thresholds:
        subset = [r for r in full_r if f'_Z{zt:.1f}_' in r['name']]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            best_ann = max(r['ann'] for r in subset)
            avg_wr = np.mean([r['wr'] for r in subset])
            avg_sh = np.mean([r['sharpe'] for r in subset])
            print(f"  {zt:4.1f} | {len(subset):5d} | {avg_ann:+6.1f}% | {best_ann:+7.1f}% | "
                  f"{avg_wr:5.1f}% | {avg_sh:10.2f}")

    # --- INTERACTION: MODE TYPE x WEIGHT ---
    print(f"\n  MODE TYPE x WEIGHT INTERACTION:")
    print(f"  {'Mode Type':25s} | {'Weight':15s} | {'N':>5s} | {'Avg Ann':>8s} | {'Best Ann':>9s} | {'Avg Sharpe':>10s}")
    print(f"  {'-' * 100}")
    for mt in mode_types:
        for wt in weight_types:
            subset = [r for r in full_r if f'MT_{mt}_' in r['name'] and f'_W_{wt}' in r['name']]
            if subset:
                avg_ann = np.mean([r['ann'] for r in subset])
                best_ann = max(r['ann'] for r in subset)
                avg_sh = np.mean([r['sharpe'] for r in subset])
                print(f"  {mt:25s} | {wt:15s} | {len(subset):5d} | {avg_ann:+7.1f}% | "
                      f"{best_ann:+8.1f}% | {avg_sh:10.2f}")

    # --- BEST CONFIG DETAIL ---
    if full_r:
        best = full_r[0]
        print(f"\n{'=' * 160}")
        print(f"  BEST CONFIG DETAIL: {best['name']}")
        print(f"  Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  N={best['n']}  "
              f"DD={best['dd']:.1f}%  PF={best['pf']:.2f}  Sharpe={best['sharpe']:.2f}  "
              f"Final={best['cash']:.0f}")
        print(f"{'=' * 160}")

        print(f"\n  PER-PAIR BREAKDOWN:")
        for p in sorted(best['pair_stats'].keys(), key=lambda x: -best['pair_stats'][x]['n']):
            ps = best['pair_stats'][p]
            wr_p = ps['w'] / max(ps['n'], 1) * 100
            print(f"    {p:25s}: {ps['n']:4d} trades  WR={wr_p:5.1f}%  Abs PnL={ps['pnl']:+12.0f}")

        print(f"\n  YEARLY BREAKDOWN:")
        for y in sorted(best['yearly'].keys()):
            s = best['yearly'][y]
            wr_y = s['w'] / max(s['n'], 1) * 100
            print(f"    {y}: {s['n']:4d} trades  WR={wr_y:5.1f}%  PnL={s['pnl']:+.1f}%  "
                  f"Abs={s['pnl_abs_sum']:+.0f}")

        print(f"\n  EXIT REASON BREAKDOWN:")
        for reason, s in sorted(best['reasons'].items(), key=lambda x: -x[1]['n']):
            rwr = s['w'] / max(s['n'], 1) * 100
            print(f"    {reason:15s}: {s['n']:4d} trades  WR={rwr:5.1f}%  "
                  f"PnL={s['pnl_pct_sum']:+.1f}%  Abs={s['pnl']:+.0f}")

        if best.get('mode_usage') and len(best['mode_usage']) > 1:
            print(f"\n  SPREAD MODE USAGE:")
            for mode_key in sorted(best['mode_usage'].keys(),
                                   key=lambda x: -best['mode_usage'][x]['n']):
                mu = best['mode_usage'][mode_key]
                wr_m = mu['w'] / max(mu['n'], 1) * 100
                print(f"    {mode_key:15s}: {mu['n']:5d} trades  WR={wr_m:5.1f}%  "
                      f"PnL={mu['pnl']:+12.0f}")

        # Per-pair mode usage for best config
        if best.get('pair_mode_usage'):
            print(f"\n  PER-PAIR MODE/LB SELECTION (best config):")
            for p in sorted(best['pair_mode_usage'].keys()):
                pmu = best['pair_mode_usage'][p]
                modes_str = " | ".join(
                    f"{mk}: {v['n']}t WR={v['w']/max(v['n'],1)*100:.0f}% PnL={v['pnl']:+.0f}"
                    for mk, v in sorted(pmu.items(), key=lambda x: -x[1]['n'])
                )
                print(f"    {p:25s}: {modes_str}")

    # --- YEARLY FOR TOP 5 ---
    if len(full_r) >= 2:
        print(f"\n  YEARLY BREAKDOWN FOR TOP 5 CONFIGS:")
        for idx, r in enumerate(full_r[:5]):
            print(f"\n  #{idx + 1}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.1f}%, "
                  f"DD={r['dd']:.1f}%, Sharpe={r['sharpe']:.2f}, N={r['n']})")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:4d}t  WR={wr_y:5.1f}%  PnL={ys['pnl']:+.1f}%")

    # --- PAIR PROFITABILITY ACROSS TOP 20 ---
    if full_r:
        print(f"\n  PAIR PROFITABILITY ACROSS TOP 20 CONFIGS:")
        pair_summary = {}
        for r in full_r[:20]:
            for p, ps in r['pair_stats'].items():
                if p not in pair_summary:
                    pair_summary[p] = {'n': 0, 'w': 0, 'pnl': 0.0}
                pair_summary[p]['n'] += ps['n']
                pair_summary[p]['w'] += ps['w']
                pair_summary[p]['pnl'] += ps['pnl']

        for p in sorted(pair_summary.keys(), key=lambda x: -pair_summary[x]['pnl']):
            ps = pair_summary[p]
            wr_p = ps['w'] / max(ps['n'], 1) * 100
            print(f"    {p:25s}: {ps['n']:5d} trades  WR={wr_p:5.1f}%  Total Abs={ps['pnl']:+12.0f}")

    # --- PER-PAIR MODE/LB SELECTION STATS ACROSS TOP 20 ---
    if full_r:
        print(f"\n  PER-PAIR MODE PREFERENCES ACROSS TOP 20 CONFIGS:")
        pair_mode_pref = {}
        for r in full_r[:20]:
            for p, pmu in r.get('pair_mode_usage', {}).items():
                if p not in pair_mode_pref:
                    pair_mode_pref[p] = {}
                for mk, v in pmu.items():
                    if mk not in pair_mode_pref[p]:
                        pair_mode_pref[p][mk] = {'n': 0, 'w': 0, 'pnl': 0.0}
                    pair_mode_pref[p][mk]['n'] += v['n']
                    pair_mode_pref[p][mk]['w'] += v['w']
                    pair_mode_pref[p][mk]['pnl'] += v['pnl']

        for p in sorted(pair_mode_pref.keys()):
            modes = pair_mode_pref[p]
            total_n = sum(m['n'] for m in modes.values())
            if total_n < 5:
                continue
            top_mode = max(modes.keys(), key=lambda x: modes[x]['n'])
            top_pct = modes[top_mode]['n'] / total_n * 100
            all_str = " | ".join(
                f"{mk}: {v['n']}t ({v['n']/total_n*100:.0f}%)"
                for mk, v in sorted(modes.items(), key=lambda x: -x[1]['n'])[:3]
            )
            print(f"    {p:25s}: dominant={top_mode} ({top_pct:.0f}%)  [{all_str}]")

    # --- V55 COMPARISON ---
    print(f"\n  === V57 vs V55 BASELINE ===")
    print(f"  V55 best: +307.3%, Sharpe 2.17, WF avg +355.7%")
    if full_r:
        print(f"  V57 best: {full_r[0]['name']}")
        print(f"    Ann={full_r[0]['ann']:+.1f}%  N={full_r[0]['n']}  "
              f"WR={full_r[0]['wr']:.1f}%  DD={full_r[0]['dd']:.1f}%  "
              f"Sharpe={full_r[0]['sharpe']:.2f}")
        delta = full_r[0]['ann'] - 307.3
        print(f"    Delta vs V55: {delta:+.1f}%")

        beating_v55 = sum(1 for r in full_r if r['ann'] > 307.3)
        beating_400 = sum(1 for r in full_r if r['ann'] > 400)
        beating_500 = sum(1 for r in full_r if r['ann'] > 500)
        print(f"    Configs beating V55 (+307.3%): {beating_v55}/{len(full_r)}")
        print(f"    Configs > 400% annual: {beating_400}/{len(full_r)}")
        print(f"    Configs > 500% annual: {beating_500}/{len(full_r)}")

        # Best walk-forward average
        if wf_by_config:
            best_wf_base = wf_summary[0][0]
            best_wf_avg = wf_summary[0][2]
            best_wf_years = wf_summary[0][1]
            print(f"\n  BEST WALK-FORWARD AVERAGE:")
            print(f"    Config: {best_wf_base}")
            print(f"    Avg WF Ann: {best_wf_avg:+.1f}% (V55 WF avg: +355.7%)")
            for y_label in ['WF2022', 'WF2023', 'WF2024']:
                if y_label in best_wf_years:
                    yr = best_wf_years[y_label]
                    print(f"      {y_label}: Ann={yr['ann']:+.1f}%  WR={yr['wr']:.1f}%  "
                          f"N={yr['n']}  DD={yr['dd']:.1f}%")

    # --- CONFIG SUMMARY ---
    print(f"\n  CONFIG SUMMARY BY CATEGORY:")
    categories = {
        'Global Adaptive': len([r for r in full_r if 'MT_global_adaptive_' in r['name']]),
        'Per-Pair Adaptive': len([r for r in full_r if 'MT_per_pair_adaptive_' in r['name']]),
        'Per-Pair Fixed': len([r for r in full_r if 'MT_per_pair_fixed_' in r['name']]),
        'Hybrid (Global Mode + Per-Pair LB)': len([r for r in full_r if 'MT_hybrid_' in r['name']]),
        'Walk-Forward': len(wf_only),
    }
    for cat, count in categories.items():
        print(f"    {cat:40s}: {count:4d} configs")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 150)


if __name__ == '__main__':
    main()
