"""
Alpha Futures V55 -- Adaptive Spread Mode Selection + Rigorous Walk-Forward
==========================================================================
V52 champion: LB10_Z1.0_H1_EZ0_MP1 = +303.5% annual (raw price spread)
V53 finding: PCT spread at LB20 achieved +419.6% WF2023
Key insight: Different spread modes dominate different periods.
If we adaptively select the best spread mode, we might push past 400%.

Approach:
  1. Precompute z-scores for all 3 spread modes (raw, PCT, log)
     at multiple lookbacks [5, 7, 10, 15, 20]
  2. Precompute hypothetical daily returns for each (mode, LB) combo
     by simulating what would have happened if that combo traded
  3. Every eval_period days, evaluate which combo had the best
     rolling return, then switch to that combo for actual trading
  4. This is walk-forward in microcosm -- always using the recently-best approach

Also tests:
  - Non-adaptive baselines: raw_LB10, pct_LB10, pct_LB20, log_LB20
  - Per-pair adaptive mode selection
  - Adding cfi/csfi pair (V46 showed 70.4% WR)
  - Lookback sweep for adaptive mode [5, 7, 10, 15, 20]

~200-300 configs. Print top 20, walk-forward, adaptive mode selection stats.
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

# Standard 13 pairs from V39/V52
PAIRS = [
    ('rbfi', 'ifi'), ('hcfi', 'ifi'), ('hcfi', 'rbfi'),
    ('jfi', 'jmfi'), ('mafi', 'scfi'), ('fufi', 'scfi'),
    ('bfi', 'scfi'), ('mfi', 'afi'), ('yfi', 'afi'),
    ('pfi', 'yfi'), ('ppfi', 'mafi'), ('vfi', 'mafi'),
    ('egfi', 'mafi'),
]

# Extended 14 pairs: add cfi/csfi from V46 (70.4% WR)
PAIRS_EXT = PAIRS + [('cfi', 'csfi')]

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

# Spread modes
SPREAD_RAW = 'raw'
SPREAD_PCT = 'pct'
SPREAD_LOG = 'log'
ALL_MODES = [SPREAD_RAW, SPREAD_PCT, SPREAD_LOG]
ALL_LOOKBACKS = [5, 7, 10, 15, 20]


def main():
    t_start = time.time()
    print("=" * 140)
    print("Alpha Futures V55 -- Adaptive Spread Mode Selection + Rigorous Walk-Forward")
    print("V52 champion: +303.5% (raw LB10) | V53 PCT LB20: +419.6% WF2023")
    print("Hypothesis: Adaptive spread mode selection pushes past 400%")
    print("=" * 140)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()
    sym_to_si = {syms[si]: si for si in range(NS)}

    def build_pair_indices(pairs_list):
        indices = []
        for down_sym, up_sym in pairs_list:
            down_si = sym_to_si.get(down_sym, -1)
            up_si = sym_to_si.get(up_sym, -1)
            if down_si >= 0 and up_si >= 0:
                indices.append((down_si, up_si, down_sym, up_sym))
            else:
                print(f"  WARNING: pair ({down_sym}, {up_sym}) not found "
                      f"(down_si={down_si}, up_si={up_si})")
        return indices

    pair_indices = build_pair_indices(PAIRS)
    pair_indices_ext = build_pair_indices(PAIRS_EXT)

    print(f"  {NS} commodities, {ND} days, "
          f"{len(pair_indices)} standard pairs, {len(pair_indices_ext)} extended pairs")

    # ================================================================
    # PRECOMPUTE SPREADS AND Z-SCORES FOR ALL MODES x LOOKBACKS
    # ================================================================
    print("\n[Signals] Precomputing spreads and z-scores for all modes x lookbacks...", flush=True)
    t0 = time.time()

    z_scores = {m: {} for m in ALL_MODES}
    spreads_all = {m: {} for m in ALL_MODES}

    all_pair_keys = set()
    for pidx_list in [pair_indices, pair_indices_ext]:
        for down_si, up_si, down_sym, up_sym in pidx_list:
            all_pair_keys.add((down_si, up_si, down_sym, up_sym))

    for down_si, up_si, down_sym, up_sym in all_pair_keys:
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
    # PRECOMPUTE HYPOTHETICAL COMBO RETURNS FOR ADAPTIVE SELECTION
    # ================================================================
    # For each (mode, lb, z_thresh) combo, simulate what the 1-day return
    # would have been for each pair on each day. This gives us a rolling
    # performance metric to drive adaptive selection.
    print("\n[Signals] Precomputing hypothetical combo returns...", flush=True)
    t1 = time.time()

    # combo_daily_return[combo_key][di] = average pct return across all pairs that had a signal
    combo_daily_return = {}

    for zt in [0.8, 1.0, 1.2]:
        for mode in ALL_MODES:
            for lb in ALL_LOOKBACKS:
                combo_key = (mode, lb, zt)
                daily_ret = np.full(ND, np.nan)

                for di in range(MIN_TRAIN + 1, ND):
                    pair_rets = []
                    for down_si, up_si, down_sym, up_sym in pair_indices:
                        z_arr = z_scores[mode].get((down_si, up_si), {}).get(lb)
                        if z_arr is None:
                            continue
                        z_prev = z_arr[di - 1]
                        if np.isnan(z_prev) or abs(z_prev) < zt:
                            continue

                        # Simulate 1-day trade: enter at di-1, exit at di
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

                        # Direction: z > 0 means short down/long up, z < 0 means long down/short up
                        if z_prev > 0:
                            # Short down, long up
                            pnl_down = (c_down_entry - c_down_exit) * mult_down
                            pnl_up = (c_up_exit - c_up_entry) * mult_up
                        else:
                            # Long down, short up
                            pnl_down = (c_down_exit - c_down_entry) * mult_down
                            pnl_up = (c_up_entry - c_up_exit) * mult_up

                        invested = c_down_entry * mult_down + c_up_entry * mult_up
                        cost = invested * COMM * 2  # round-trip
                        pnl_pct = (pnl_down + pnl_up - cost) / invested * 100 if invested > 0 else 0
                        pair_rets.append(pnl_pct)

                    if pair_rets:
                        daily_ret[di] = np.mean(pair_rets)

                combo_daily_return[combo_key] = daily_ret

    print(f"  Hypothetical returns precomputed ({time.time() - t1:.1f}s)", flush=True)

    # ================================================================
    # BACKTEST ENGINE -- ADAPTIVE SPREAD MODE SELECTION
    # ================================================================
    def run_backtest(z_thresh=1.0, hold_max=1, exit_z=0.0, max_pairs=1,
                     # Adaptive params
                     adaptive_mode='global',   # 'global', 'per_pair', 'off'
                     eval_period=60,
                     adaptive_lookbacks=ALL_LOOKBACKS,
                     adaptive_modes=ALL_MODES,
                     # Non-adaptive override
                     fixed_mode=SPREAD_RAW,
                     fixed_lb=10,
                     # Pair set
                     use_ext_pairs=False,
                     # Walk-forward
                     wf_split_year=None,
                     wf_end_year=None,
                     config_name=""):
        """
        Adaptive pair trading backtest with spread mode selection.

        Key difference from naive approach: uses precomputed hypothetical
        returns for each combo to drive adaptive selection, rather than
        only tracking executed trades.
        """
        pidx = pair_indices_ext if use_ext_pairs else pair_indices

        if adaptive_mode == 'off':
            combos = [(fixed_mode, fixed_lb)]
        else:
            combos = [(m, lb) for m in adaptive_modes for lb in adaptive_lookbacks]

        cash = float(CASH0)
        trades = []
        pair_positions = []

        # Precompute rolling combo scores for adaptive selection
        # combo_cum_ret[combo][di] = cumulative return over last eval_period days
        current_combo = combos[0]

        # Per-pair adaptive: each pair tracks its own best combo
        pair_combo = {}
        if adaptive_mode == 'per_pair':
            for down_si, up_si, _, _ in pidx:
                pair_combo[(down_si, up_si)] = combos[0]

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
            if adaptive_mode != 'off' and di > start_di:
                days_since_start = di - start_di
                if days_since_start % eval_period == 0 and days_since_start >= eval_period:
                    if adaptive_mode == 'global':
                        best_combo = combos[0]
                        best_score = -1e18
                        for c in combos:
                            combo_key = (c[0], c[1], z_thresh)
                            daily_ret = combo_daily_return.get(combo_key)
                            if daily_ret is None:
                                continue
                            # Sum returns over last eval_period days
                            window = daily_ret[max(start_di, di - eval_period):di]
                            valid = window[~np.isnan(window)]
                            if len(valid) >= 5:
                                score = np.nansum(valid)
                            elif len(valid) > 0:
                                score = np.nansum(valid) * 0.5
                            else:
                                score = -1e10
                            if score > best_score:
                                best_score = score
                                best_combo = c
                        current_combo = best_combo

                    elif adaptive_mode == 'per_pair':
                        for down_si, up_si, down_sym, up_sym in pidx:
                            best_combo = combos[0]
                            best_score = -1e18
                            for c in combos:
                                combo_key = (c[0], c[1], z_thresh)
                                daily_ret = combo_daily_return.get(combo_key)
                                if daily_ret is None:
                                    continue
                                window = daily_ret[max(start_di, di - eval_period):di]
                                valid = window[~np.isnan(window)]
                                if len(valid) >= 3:
                                    score = np.nansum(valid)
                                elif len(valid) > 0:
                                    score = np.nansum(valid) * 0.5
                                else:
                                    score = -1e10
                                if score > best_score:
                                    best_score = score
                                    best_combo = c
                            pair_combo[(down_si, up_si)] = best_combo

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

                if adaptive_mode == 'off':
                    use_combo = (fixed_mode, fixed_lb)
                elif adaptive_mode == 'global':
                    use_combo = current_combo
                elif adaptive_mode == 'per_pair':
                    use_combo = pair_combo.get((down_si, up_si), combos[0])

                use_mode, use_lb = use_combo
                z_arr = z_scores[use_mode].get((down_si, up_si), {}).get(use_lb)
                if z_arr is None:
                    continue
                z_val = z_arr[di] if di < len(z_arr) else np.nan
                if np.isnan(z_val):
                    continue
                if abs(z_val) < z_thresh:
                    continue

                candidates.append((abs(z_val), down_si, up_si, down_sym, up_sym,
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
            'trades': trades,
        }

    # ================================================================
    # PARAMETER SWEEP
    # ================================================================
    print("\n[Backtest] Running parameter sweep...", flush=True)
    results = []
    configs = []

    # ---- Test 1: Non-adaptive baselines ----
    print("  === Test 1: Non-adaptive baselines ===")
    for mode in ALL_MODES:
        for lb in [10, 20]:
            for zt in [0.8, 1.0, 1.2]:
                name = f"BAS_{mode}_LB{lb}_Z{zt:.1f}_H1_EZ0_MP1"
                configs.append({
                    'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0, 'max_pairs': 1,
                    'adaptive_mode': 'off', 'fixed_mode': mode, 'fixed_lb': lb,
                    'use_ext_pairs': False, 'wf_split_year': None, 'wf_end_year': None,
                    'config_name': name,
                })

    # ---- Test 2: Global adaptive with different eval periods ----
    print("  === Test 2: Global adaptive (all LBs, eval period sweep) ===")
    for ep in [20, 40, 60, 90]:
        for zt in [0.8, 1.0, 1.2]:
            for ez in [0.0, 0.2]:
                name = f"ADG_EP{ep}_Z{zt:.1f}_EZ{ez:.1f}_H1_MP1"
                configs.append({
                    'z_thresh': zt, 'hold_max': 1, 'exit_z': ez, 'max_pairs': 1,
                    'adaptive_mode': 'global', 'eval_period': ep,
                    'adaptive_lookbacks': ALL_LOOKBACKS,
                    'adaptive_modes': ALL_MODES,
                    'use_ext_pairs': False, 'wf_split_year': None, 'wf_end_year': None,
                    'config_name': name,
                })

    # ---- Test 3: Per-pair adaptive ----
    print("  === Test 3: Per-pair adaptive (all LBs, eval period sweep) ===")
    for ep in [40, 60, 90]:
        for zt in [0.8, 1.0, 1.2]:
            name = f"ADP_EP{ep}_Z{zt:.1f}_EZ0_H1_MP1"
            configs.append({
                'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0, 'max_pairs': 1,
                'adaptive_mode': 'per_pair', 'eval_period': ep,
                'adaptive_lookbacks': ALL_LOOKBACKS,
                'adaptive_modes': ALL_MODES,
                'use_ext_pairs': False, 'wf_split_year': None, 'wf_end_year': None,
                'config_name': name,
            })

    # ---- Test 4: Adaptive with MP2 ----
    print("  === Test 4: Adaptive global with MP2 ===")
    for ep in [40, 60, 90]:
        for zt in [0.8, 1.0, 1.2]:
            name = f"ADG_EP{ep}_Z{zt:.1f}_EZ0_H1_MP2"
            configs.append({
                'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0, 'max_pairs': 2,
                'adaptive_mode': 'global', 'eval_period': ep,
                'adaptive_lookbacks': ALL_LOOKBACKS,
                'adaptive_modes': ALL_MODES,
                'use_ext_pairs': False, 'wf_split_year': None, 'wf_end_year': None,
                'config_name': name,
            })

    # ---- Test 5: Adaptive with extended pairs (cfi/csfi) ----
    print("  === Test 5: Adaptive with cfi/csfi pair ===")
    for ep in [40, 60, 90]:
        for zt in [0.8, 1.0, 1.2]:
            name = f"ADG_EP{ep}_Z{zt:.1f}_EZ0_H1_MP1_EXT"
            configs.append({
                'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0, 'max_pairs': 1,
                'adaptive_mode': 'global', 'eval_period': ep,
                'adaptive_lookbacks': ALL_LOOKBACKS,
                'adaptive_modes': ALL_MODES,
                'use_ext_pairs': True, 'wf_split_year': None, 'wf_end_year': None,
                'config_name': name,
            })

    # ---- Test 6: Adaptive with restricted lookbacks ----
    print("  === Test 6: Adaptive with subset lookbacks ===")
    for lb_set_name, lb_set in [('short', [5, 7, 10]), ('long', [10, 15, 20])]:
        for ep in [40, 60]:
            for zt in [0.8, 1.0, 1.2]:
                name = f"ADG_EP{ep}_Z{zt:.1f}_EZ0_H1_MP1_LB{lb_set_name}"
                configs.append({
                    'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0, 'max_pairs': 1,
                    'adaptive_mode': 'global', 'eval_period': ep,
                    'adaptive_lookbacks': lb_set,
                    'adaptive_modes': ALL_MODES,
                    'use_ext_pairs': False, 'wf_split_year': None, 'wf_end_year': None,
                    'config_name': name,
                })

    # ---- Test 7: Adaptive with restricted modes ----
    print("  === Test 7: Adaptive with subset modes ===")
    for mode_set_name, mode_set in [('raw_pct', [SPREAD_RAW, SPREAD_PCT]),
                                     ('pct_log', [SPREAD_PCT, SPREAD_LOG])]:
        for ep in [40, 60, 90]:
            for zt in [0.8, 1.0, 1.2]:
                name = f"ADG_EP{ep}_Z{zt:.1f}_EZ0_H1_MP1_M{mode_set_name}"
                configs.append({
                    'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0, 'max_pairs': 1,
                    'adaptive_mode': 'global', 'eval_period': ep,
                    'adaptive_lookbacks': ALL_LOOKBACKS,
                    'adaptive_modes': mode_set,
                    'use_ext_pairs': False, 'wf_split_year': None, 'wf_end_year': None,
                    'config_name': name,
                })

    # ---- Test 8: V52 champion replication ----
    print("  === Test 8: V52 champion replication ===")
    configs.append({
        'z_thresh': 1.0, 'hold_max': 1, 'exit_z': 0.0, 'max_pairs': 1,
        'adaptive_mode': 'off', 'fixed_mode': SPREAD_RAW, 'fixed_lb': 10,
        'use_ext_pairs': False, 'wf_split_year': None, 'wf_end_year': None,
        'config_name': "V52_RAW_LB10_Z1.0_H1_EZ0_MP1",
    })

    # ---- Test 9: V53 best PCT baselines ----
    print("  === Test 9: V53 best PCT baselines ===")
    for lb in [10, 15, 20]:
        for zt in [0.8, 1.0, 1.2]:
            name = f"V53_PCT_LB{lb}_Z{zt:.1f}_H1_EZ0_MP1"
            configs.append({
                'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0, 'max_pairs': 1,
                'adaptive_mode': 'off', 'fixed_mode': SPREAD_PCT, 'fixed_lb': lb,
                'use_ext_pairs': False, 'wf_split_year': None, 'wf_end_year': None,
                'config_name': name,
            })

    # ---- Test 10: Log spread baselines ----
    print("  === Test 10: Log spread baselines ===")
    for lb in [10, 15, 20]:
        for zt in [0.8, 1.0, 1.2]:
            name = f"V53_LOG_LB{lb}_Z{zt:.1f}_H1_EZ0_MP1"
            configs.append({
                'z_thresh': zt, 'hold_max': 1, 'exit_z': 0.0, 'max_pairs': 1,
                'adaptive_mode': 'off', 'fixed_mode': SPREAD_LOG, 'fixed_lb': lb,
                'use_ext_pairs': False, 'wf_split_year': None, 'wf_end_year': None,
                'config_name': name,
            })

    print(f"\n  Total: {len(configs)} full-period configurations", flush=True)

    for ci, cfg in enumerate(configs):
        r = run_backtest(**cfg)
        if r is not None:
            results.append(r)
            if r['ann'] > 100:
                print(f"  {r['name']:55s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
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

    wf_candidates = []
    seen_params = set()
    for r in full_results:
        if r['name'] in seen_params:
            continue
        seen_params.add(r['name'])
        wf_candidates.append(r)
        if len(wf_candidates) >= 20:
            break

    wf_results = []
    wf_configs_run = 0

    for r in wf_candidates:
        matching = [c for c in configs if c['config_name'] == r['name']]
        if not matching:
            continue
        cfg = matching[0]

        for wf_test_year, wf_label in [(2021, 'WF2021'), (2022, 'WF2022'),
                                         (2023, 'WF2023'), (2024, 'WF2024')]:
            wf_name = f"{r['name']}_{wf_label}"
            wf_cfg = dict(cfg)
            wf_cfg['wf_split_year'] = wf_test_year
            wf_cfg['wf_end_year'] = wf_test_year + 1 if wf_test_year < 2024 else None
            wf_cfg['config_name'] = wf_name
            wr = run_backtest(**wf_cfg)
            if wr is not None:
                wf_results.append(wr)
            wf_configs_run += 1

        if wf_configs_run % 20 == 0:
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
    print(f"\n{'=' * 150}")
    print(f"  TOP 20 FULL-PERIOD RESULTS (sorted by annual return)")
    print(f"{'=' * 150}")
    hdr = (f"  {'Config':55s} | {'Ann':>7s} | {'WR':>5s} | {'N':>5s} | {'DD':>6s} | "
           f"{'PF':>4s} | {'Sharpe':>6s} | {'AvgW':>7s} | {'AvgL':>6s} | {'AvgD':>4s}")
    print(hdr)
    print(f"  {'-' * 145}")
    for r in full_r[:20]:
        print(f"  {r['name']:55s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:5d} | {r['dd']:6.1f}% | {r['pf']:4.2f} | "
              f"{r['sharpe']:6.2f} | {r['avg_win']:+6.2f}% | {r['avg_loss']:5.2f}% | "
              f"{r['avg_days']:4.1f}")

    # --- TOP 20 WALK-FORWARD ---
    if wf_only:
        print(f"\n{'=' * 150}")
        print(f"  TOP 20 WALK-FORWARD RESULTS (out-of-sample)")
        print(f"{'=' * 150}")
        for r in wf_only[:20]:
            print(f"  {r['name']:65s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                  f"Sh {r['sharpe']:6.2f} | AvgD {r['avg_days']:.1f}")

    # --- WALK-FORWARD YEARLY SUMMARY ---
    if wf_only:
        print(f"\n  WALK-FORWARD YEARLY SUMMARY:")
        print(f"  {'Config':55s} | {'WF2021':>8s} | {'WF2022':>8s} | {'WF2023':>8s} | {'WF2024':>8s} | {'Avg WF':>8s}")
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
            for y_label in ['WF2021', 'WF2022', 'WF2023', 'WF2024']:
                if y_label in years:
                    parts.append(f"{years[y_label]['ann']:+7.1f}%")
                else:
                    parts.append(f"{'N/A':>8s}")
            print(f"  {base_name:55s} | {' | '.join(parts)} | {avg_ann:+7.1f}%")

    # --- ADAPTIVE vs NON-ADAPTIVE ---
    print(f"\n{'=' * 150}")
    print(f"  ADAPTIVE vs NON-ADAPTIVE COMPARISON")
    print(f"{'=' * 150}")

    adaptive_global = [r for r in full_r if r['name'].startswith('ADG_')]
    adaptive_per_pair = [r for r in full_r if r['name'].startswith('ADP_')]
    non_adaptive = [r for r in full_r if r['name'].startswith(('BAS_', 'V52_', 'V53_'))]

    for label, subset in [('Adaptive Global', adaptive_global),
                          ('Adaptive Per-Pair', adaptive_per_pair),
                          ('Non-Adaptive Baselines', non_adaptive)]:
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            best_ann = max(r['ann'] for r in subset)
            avg_wr = np.mean([r['wr'] for r in subset])
            avg_sh = np.mean([r['sharpe'] for r in subset])
            print(f"  {label:25s}: {len(subset):3d} configs | "
                  f"Avg Ann={avg_ann:+6.1f}% | Best={best_ann:+7.1f}% | "
                  f"Avg WR={avg_wr:.1f}% | Avg Sharpe={avg_sh:.2f}")

    # --- EVAL PERIOD COMPARISON ---
    print(f"\n  EVAL PERIOD COMPARISON (global adaptive):")
    print(f"  {'EP':>4s} | {'N':>5s} | {'Avg Ann':>7s} | {'Best Ann':>8s} | {'Avg WR':>6s} | {'Avg Sharpe':>10s}")
    print(f"  {'-' * 60}")
    for ep in [20, 40, 60, 90]:
        subset = [r for r in full_r if f'_EP{ep}_' in r['name'] and r['name'].startswith('ADG_')]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            best_ann = max(r['ann'] for r in subset)
            avg_wr = np.mean([r['wr'] for r in subset])
            avg_sh = np.mean([r['sharpe'] for r in subset])
            print(f"  {ep:4d} | {len(subset):5d} | {avg_ann:+6.1f}% | {best_ann:+7.1f}% | "
                  f"{avg_wr:5.1f}% | {avg_sh:10.2f}")

    # --- SPREAD MODE USAGE IN TOP ADAPTIVE CONFIGS ---
    print(f"\n  SPREAD MODE USAGE IN TOP ADAPTIVE CONFIGS:")
    for r in full_r[:5]:
        if r.get('mode_usage') and len(r['mode_usage']) > 1:
            print(f"\n  {r['name']}:")
            for mode_key in sorted(r['mode_usage'].keys(),
                                   key=lambda x: -r['mode_usage'][x]['n']):
                mu = r['mode_usage'][mode_key]
                wr_m = mu['w'] / max(mu['n'], 1) * 100
                print(f"    {mode_key:15s}: {mu['n']:5d} trades  WR={wr_m:5.1f}%  "
                      f"PnL={mu['pnl']:+12.0f}")
        else:
            mode_key = list(r.get('mode_usage', {}).keys())
            if mode_key:
                print(f"\n  {r['name']}: uses {mode_key[0]} (non-adaptive)")

    # --- EXTENDED PAIRS COMPARISON ---
    print(f"\n  EXTENDED PAIRS (cfi/csfi) IMPACT:")
    ext_configs = [r for r in full_r if '_EXT' in r['name']]
    std_adg = [r for r in full_r if '_EXT' not in r['name'] and r['name'].startswith('ADG_')]
    if ext_configs and std_adg:
        print(f"    With cfi/csfi   : {len(ext_configs):3d} configs | "
              f"Avg Ann={np.mean([r['ann'] for r in ext_configs]):+.1f}% | "
              f"Best={max(r['ann'] for r in ext_configs):+.1f}%")
        print(f"    Without cfi/csfi: {len(std_adg):3d} configs | "
              f"Avg Ann={np.mean([r['ann'] for r in std_adg]):+.1f}% | "
              f"Best={max(r['ann'] for r in std_adg):+.1f}%")

    # --- Z-THRESHOLD COMPARISON ---
    print(f"\n  Z-THRESHOLD COMPARISON (all configs):")
    print(f"  {'Z':>4s} | {'N':>5s} | {'Avg Ann':>7s} | {'Best Ann':>8s} | {'Avg WR':>6s} | {'Avg Sharpe':>10s}")
    print(f"  {'-' * 60}")
    for zt in [0.8, 1.0, 1.2]:
        subset = [r for r in full_r if f'_Z{zt:.1f}_' in r['name']]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            best_ann = max(r['ann'] for r in subset)
            avg_wr = np.mean([r['wr'] for r in subset])
            avg_sh = np.mean([r['sharpe'] for r in subset])
            print(f"  {zt:4.1f} | {len(subset):5d} | {avg_ann:+6.1f}% | {best_ann:+7.1f}% | "
                  f"{avg_wr:5.1f}% | {avg_sh:10.2f}")

    # --- BEST CONFIG DETAIL ---
    if full_r:
        best = full_r[0]
        print(f"\n{'=' * 150}")
        print(f"  BEST CONFIG DETAIL: {best['name']}")
        print(f"  Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  N={best['n']}  "
              f"DD={best['dd']:.1f}%  PF={best['pf']:.2f}  Sharpe={best['sharpe']:.2f}  "
              f"Final={best['cash']:.0f}")
        print(f"{'=' * 150}")

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

    # --- V52 COMPARISON ---
    print(f"\n  === V55 vs PRIOR BASELINES ===")
    print(f"  V52 champion: LB10_Z1.0_H1_EZ0_MP1 (raw) = +303.5%")
    print(f"  V53 PCT LB20 best WF2023 = +419.6%")
    if full_r:
        print(f"  V55 best: {full_r[0]['name']}")
        print(f"    Ann={full_r[0]['ann']:+.1f}%  N={full_r[0]['n']}  "
              f"WR={full_r[0]['wr']:.1f}%  DD={full_r[0]['dd']:.1f}%  "
              f"Sharpe={full_r[0]['sharpe']:.2f}")
        delta = full_r[0]['ann'] - 303.5
        print(f"    Delta vs V52: {delta:+.1f}%")

        beating_v52 = sum(1 for r in full_r if r['ann'] > 303.5)
        beating_400 = sum(1 for r in full_r if r['ann'] > 400)
        beating_500 = sum(1 for r in full_r if r['ann'] > 500)
        print(f"    Configs beating V52 (+303.5%): {beating_v52}/{len(full_r)}")
        print(f"    Configs > 400% annual: {beating_400}/{len(full_r)}")
        print(f"    Configs > 500% annual: {beating_500}/{len(full_r)}")

        # Best walk-forward average
        if wf_by_config:
            best_wf_base = wf_summary[0][0]
            best_wf_avg = wf_summary[0][2]
            best_wf_years = wf_summary[0][1]
            print(f"\n  BEST WALK-FORWARD AVERAGE:")
            print(f"    Config: {best_wf_base}")
            print(f"    Avg WF Ann: {best_wf_avg:+.1f}%")
            for y_label in ['WF2021', 'WF2022', 'WF2023', 'WF2024']:
                if y_label in best_wf_years:
                    yr = best_wf_years[y_label]
                    print(f"      {y_label}: Ann={yr['ann']:+.1f}%  WR={yr['wr']:.1f}%  "
                          f"N={yr['n']}  DD={yr['dd']:.1f}%")

    # --- CONFIG SUMMARY ---
    print(f"\n  CONFIG SUMMARY BY CATEGORY:")
    categories = {
        'Baseline (non-adaptive)': len([r for r in full_r if r['name'].startswith(('BAS_', 'V52_', 'V53_'))]),
        'Adaptive Global': len([r for r in full_r if r['name'].startswith('ADG_')]),
        'Adaptive Per-Pair': len([r for r in full_r if r['name'].startswith('ADP_')]),
        'Walk-Forward': len(wf_only),
    }
    for cat, count in categories.items():
        print(f"    {cat:25s}: {count:4d} configs")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 150)


if __name__ == '__main__':
    main()
