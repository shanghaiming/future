"""
Alpha Futures V61 -- Adaptive LOG-Biased Framework + Rigorous Walk-Forward
==========================================================================
V57: Global adaptive (raw/pct/log auto-switch, LB[5,7,10]) = +315.3%
V60: LOG spread fixed LB15_Z1.5 = +312.9%, PF=3.46, 100% WF positive

Idea: Use V57's adaptive framework but bias toward LOG spread.
If LOG is genuinely better in recent data, an adaptive system that includes
LOG in the candidate set should naturally select it more often.

Tests:
  1. Adaptive with LOG bias: candidates [raw_LB10, pct_LB10, log_LB10, log_LB15, log_LB20]
  2. Pure LOG adaptive lookback: always LOG, adaptively choose LB from [5,7,10,15,20]
  3. LOG + cfi/csfi pair: 13 pairs vs 14 pairs
  4. LOG with aggressive Z: Z in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5]
  5. Final champion: top 5 configs get rigorous 6-window WF

~200 full-period configs + rigorous WF for top 5.

Walk-forward windows (like V58):
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

PAIRS_13 = [
    ('rbfi', 'ifi'), ('hcfi', 'ifi'), ('hcfi', 'rbfi'),
    ('jfi', 'jmfi'), ('mafi', 'scfi'), ('fufi', 'scfi'),
    ('bfi', 'scfi'), ('mfi', 'afi'), ('yfi', 'afi'),
    ('pfi', 'yfi'), ('ppfi', 'mafi'), ('vfi', 'mafi'),
    ('egfi', 'mafi'),
]
PAIRS_14 = PAIRS_13 + [('cfi', 'csfi')]

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
    print("=" * 150)
    print("Alpha Futures V61 -- Adaptive LOG-Biased Framework + Rigorous Walk-Forward")
    print("V57: Global adaptive +315.3% | V60: LOG LB15_Z1.5 +312.9%, PF=3.46, 100% WF positive")
    print("Idea: Adaptive framework biased toward LOG. If LOG dominates, adaptive should select it.")
    print("=" * 150)

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

    # Build pair index mapping for both pair sets
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

    pair_indices_13 = build_pair_indices(PAIRS_13)
    pair_indices_14 = build_pair_indices(PAIRS_14)

    # ================================================================
    # PRECOMPUTE SPREADS AND Z-SCORES FOR ALL MODES x LOOKBACKS
    # ================================================================
    print("\n[Signals] Precomputing spreads and z-scores for all modes x lookbacks...", flush=True)
    t0 = time.time()

    z_scores = {m: {} for m in ALL_MODES}

    # Use union of all pairs to precompute
    all_pair_set = set()
    for down_si, up_si, down_sym, up_sym in pair_indices_14:
        all_pair_set.add((down_si, up_si))

    for down_si, up_si in all_pair_set:
        key = (down_si, up_si)
        # Compute spreads for each mode
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

    print(f"  All z-scores precomputed ({time.time() - t0:.1f}s)", flush=True)

    # ================================================================
    # PRECOMPUTE PER-PAIR HYPOTHETICAL RETURNS FOR ADAPTIVE SELECTION
    # ================================================================
    print("\n[Signals] Precomputing per-pair hypothetical combo returns...", flush=True)
    t1 = time.time()

    # (pair_key, (mode, lb)) -> daily_ret[di]
    pair_combo_daily_return = {}

    all_zt = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5]
    all_pair_indices = pair_indices_14

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
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(z_thresh=1.0, hold_max=1, exit_z=0.0, max_pairs=1,
                     mode_type='adaptive_log_bias',
                     eval_period=60,
                     candidate_combos=None,
                     pair_indices=None,
                     start_year=None, end_year=None,
                     config_name=""):
        """
        mode_type controls spread mode/lookback selection:
          'adaptive_log_bias'    -- adaptive selection from candidate set (3/5 LOG slots)
          'pure_log_adaptive_lb' -- always LOG, adaptively choose lookback
          'fixed'               -- single fixed (mode, lb, z_thresh)

        candidate_combos: list of (mode, lb) tuples for adaptive selection.
        pair_indices: which pair set to use.
        start_year/end_year: restrict trading to this date range (inclusive).
        """
        if pair_indices is None:
            pair_indices = pair_indices_14
        if candidate_combos is None:
            candidate_combos = [(SPREAD_LOG, 10), (SPREAD_LOG, 15), (SPREAD_LOG, 20),
                                (SPREAD_RAW, 10), (SPREAD_PCT, 10)]

        cash = float(CASH0)
        trades = []
        pair_positions = []

        # Current adaptive combo
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
                        pass  # no adaptation
                    else:
                        # Select best combo from candidates
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

            # Determine which combo to use
            if mode_type == 'fixed':
                use_mode, use_lb = candidate_combos[0]
            else:
                use_mode, use_lb = current_combo

            candidates = []
            for down_si, up_si, down_sym, up_sym in pair_indices:
                if down_si in occupied or up_si in occupied:
                    continue

                z_arr = z_scores[use_mode].get((down_si, up_si), {}).get(use_lb)
                if z_arr is None:
                    continue
                z_val = z_arr[di] if di < len(z_arr) else np.nan
                if np.isnan(z_val):
                    continue
                if abs(z_val) < z_thresh:
                    continue

                candidates.append((abs(z_val), down_si, up_si, down_sym, up_sym, z_val))

            if not candidates:
                continue

            candidates.sort(key=lambda x: -x[0])

            for _, down_si, up_si, down_sym, up_sym, z_val in candidates[:n_can_open]:
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
                    pos_dir = -1  # spread too high -> short down, long up
                else:
                    pos_dir = 1   # spread too low -> long down, short up

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

    # --- Test 1: Adaptive with LOG bias ---
    # Candidates: [raw_LB10, pct_LB10, log_LB10, log_LB15, log_LB20]
    log_bias_combos = [(SPREAD_LOG, 10), (SPREAD_LOG, 15), (SPREAD_LOG, 20),
                       (SPREAD_RAW, 10), (SPREAD_PCT, 10)]
    for ep in [40, 60]:
        for zt in [0.8, 1.0, 1.2, 1.5]:
            for mp in [1, 2]:
                for pidx, pairs_name in [(pair_indices_13, 'P13'), (pair_indices_14, 'P14')]:
                    name = f"T1_LOGBIAS_EP{ep}_Z{zt:.1f}_MP{mp}_{pairs_name}"
                    configs.append({
                        'z_thresh': zt,
                        'hold_max': 1,
                        'exit_z': 0.0,
                        'max_pairs': mp,
                        'mode_type': 'adaptive_log_bias',
                        'eval_period': ep,
                        'candidate_combos': log_bias_combos,
                        'pair_indices': pidx,
                        'start_year': None,
                        'end_year': None,
                        'config_name': name,
                    })

    # --- Test 2: Pure LOG adaptive lookback ---
    for ep in [40, 60]:
        for zt in [0.8, 1.0, 1.2, 1.5]:
            for mp in [1, 2]:
                for pidx, pairs_name in [(pair_indices_13, 'P13'), (pair_indices_14, 'P14')]:
                    name = f"T2_PURELOG_EP{ep}_Z{zt:.1f}_MP{mp}_{pairs_name}"
                    configs.append({
                        'z_thresh': zt,
                        'hold_max': 1,
                        'exit_z': 0.0,
                        'max_pairs': mp,
                        'mode_type': 'pure_log_adaptive_lb',
                        'eval_period': ep,
                        'candidate_combos': [(SPREAD_LOG, 5), (SPREAD_LOG, 7),
                                             (SPREAD_LOG, 10), (SPREAD_LOG, 15),
                                             (SPREAD_LOG, 20)],
                        'pair_indices': pidx,
                        'start_year': None,
                        'end_year': None,
                        'config_name': name,
                    })

    # --- Test 3: LOG + cfi/csfi pair (fixed LOG, compare 13 vs 14 pairs) ---
    for lb in [5, 7, 10, 15, 20]:
        for zt in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5]:
            for mp in [1, 2]:
                for pidx, pairs_name in [(pair_indices_13, 'P13'), (pair_indices_14, 'P14')]:
                    name = f"T3_LOG_LB{lb}_Z{zt:.1f}_MP{mp}_{pairs_name}"
                    configs.append({
                        'z_thresh': zt,
                        'hold_max': 1,
                        'exit_z': 0.0,
                        'max_pairs': mp,
                        'mode_type': 'fixed',
                        'eval_period': 60,
                        'candidate_combos': [(SPREAD_LOG, lb)],
                        'pair_indices': pidx,
                        'start_year': None,
                        'end_year': None,
                        'config_name': name,
                    })

    total_combos = len(configs)
    print(f"\n{'=' * 150}")
    print(f"  FULL-PERIOD PARAMETER SWEEP ({total_combos} configs)")
    print(f"  Test 1: Adaptive LOG bias (EP[40,60] x Z[0.8,1.0,1.2,1.5] x MP[1,2] x P[13,14])")
    print(f"  Test 2: Pure LOG adaptive LB (EP[40,60] x Z[0.8,1.0,1.2,1.5] x MP[1,2] x P[13,14])")
    print(f"  Test 3: Fixed LOG (LB[5,7,10,15,20] x Z[0.3,0.5,0.8,1.0,1.2,1.5] x MP[1,2] x P[13,14])")
    print(f"{'=' * 150}")

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
    print(f"\n{'=' * 150}")
    print(f"  TOP 20 FULL-PERIOD RESULTS")
    print(f"{'=' * 150}")
    print(f"  {'#':>2s} | {'Config':40s} | {'Ann':>8s} | {'WR':>5s} | {'N':>5s} | "
          f"{'DD':>6s} | {'PF':>5s} | {'Sharpe':>7s} | {'AvgW':>6s} | {'AvgL':>6s} | "
          f"{'AvgD':>5s} | {'Cash':>12s}")
    print(f"  {'-' * 150}")

    for i, r in enumerate(results[:20]):
        print(f"  {i + 1:2d} | {r['name']:40s} | {r['ann']:+7.1f}% | {r['wr']:4.1f}% | "
              f"{r['n']:5d} | {r['dd']:5.1f}% | {r['pf']:4.2f} | {r['sharpe']:6.2f} | "
              f"{r['avg_win']:+5.2f}% | {r['avg_loss']:5.2f}% | {r['avg_days']:4.1f} | "
              f"{r['cash']:11.0f}")

    # ================================================================
    # TEST 1 vs TEST 2 COMPARISON
    # ================================================================
    print(f"\n{'=' * 150}")
    print(f"  TEST COMPARISON: Adaptive LOG Bias vs Pure LOG Adaptive vs Fixed LOG")
    print(f"{'=' * 150}")

    test_sets = {
        'T1_Adaptive_LOG_Bias': [r for r in results if r['name'].startswith('T1_')],
        'T2_Pure_LOG_Adaptive': [r for r in results if r['name'].startswith('T2_')],
        'T3_Fixed_LOG': [r for r in results if r['name'].startswith('T3_')],
    }

    for test_name, subset in test_sets.items():
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            best_ann = max(r['ann'] for r in subset)
            avg_wr = np.mean([r['wr'] for r in subset])
            avg_sh = np.mean([r['sharpe'] for r in subset])
            avg_dd = np.mean([r['dd'] for r in subset])
            avg_pf = np.mean([r['pf'] for r in subset])
            n_pos = sum(1 for r in subset if r['ann'] > 0)
            best = max(subset, key=lambda x: x['ann'])
            print(f"\n  {test_name}:")
            print(f"    N={len(subset)}  Avg Ann={avg_ann:+.1f}%  Best Ann={best_ann:+.1f}%  "
                  f"Avg WR={avg_wr:.1f}%  Avg Sharpe={avg_sh:.2f}  Avg DD={avg_dd:.1f}%  "
                  f"Avg PF={avg_pf:.2f}  Positive={n_pos}/{len(subset)}")
            print(f"    Best: {best['name']}")
            print(f"      Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  N={best['n']}  "
                  f"DD={best['dd']:.1f}%  PF={best['pf']:.2f}  Sharpe={best['sharpe']:.2f}")

    # ================================================================
    # 13 PAIRS vs 14 PAIRS COMPARISON
    # ================================================================
    print(f"\n{'=' * 150}")
    print(f"  PAIR SET COMPARISON: 13 pairs vs 14 pairs (with cfi/csfi)")
    print(f"{'=' * 150}")

    for pset_name, pset_filter in [('P13', '_P13'), ('P14', '_P14')]:
        subset = [r for r in results if r['name'].endswith(pset_filter)]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            best = max(subset, key=lambda x: x['ann'])
            n_pos = sum(1 for r in subset if r['ann'] > 0)
            print(f"  {pset_name}: N={len(subset)}  Avg Ann={avg_ann:+.1f}%  "
                  f"Best Ann={best['ann']:+.1f}%  Positive={n_pos}/{len(subset)}")
            print(f"    Best: {best['name']}  Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  "
                  f"DD={best['dd']:.1f}%  PF={best['pf']:.2f}")

    # ================================================================
    # Z-THRESHOLD COMPARISON (full-period)
    # ================================================================
    print(f"\n{'=' * 150}")
    print(f"  Z-THRESHOLD COMPARISON (best per threshold)")
    print(f"{'=' * 150}")
    print(f"  {'Z':>4s} | {'Best Config':40s} | {'Ann':>8s} | {'WR':>5s} | {'N':>5s} | "
          f"{'DD':>6s} | {'PF':>5s} | {'Sharpe':>7s}")
    print(f"  {'-' * 110}")

    z_thresholds_tested = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5]
    for zt in z_thresholds_tested:
        zt_results = [r for r in results if f'_Z{zt:.1f}_' in r['name']]
        if zt_results:
            best = zt_results[0]  # already sorted by ann
            print(f"  {zt:4.1f} | {best['name']:40s} | {best['ann']:+7.1f}% | {best['wr']:4.1f}% | "
                  f"{best['n']:5d} | {best['dd']:5.1f}% | {best['pf']:4.2f} | {best['sharpe']:6.2f}")

    # Z threshold aggregate
    print(f"\n  Z-Threshold aggregate:")
    print(f"  {'Z':>4s} | {'#Configs':>8s} | {'Avg Ann':>8s} | {'Med Ann':>8s} | "
          f"{'Best Ann':>8s} | {'%Positive':>9s} | {'Avg Sharpe':>10s}")
    print(f"  {'-' * 80}")

    for zt in z_thresholds_tested:
        zt_results = [r for r in results if f'_Z{zt:.1f}_' in r['name']]
        if zt_results:
            anns = [r['ann'] for r in zt_results]
            sharpe_vals = [r['sharpe'] for r in zt_results]
            n_pos = sum(1 for a in anns if a > 0)
            print(f"  {zt:4.1f} | {len(zt_results):8d} | {np.mean(anns):+7.1f}% | "
                  f"{np.median(anns):+7.1f}% | {max(anns):+7.1f}% | "
                  f"{n_pos}/{len(anns)} ({n_pos / len(anns) * 100:.0f}%) | "
                  f"{np.mean(sharpe_vals):9.2f}")

    # ================================================================
    # LOOKBACK COMPARISON (for fixed LOG configs only)
    # ================================================================
    print(f"\n{'=' * 150}")
    print(f"  LOOKBACK COMPARISON (fixed LOG configs, best per lookback)")
    print(f"{'=' * 150}")
    print(f"  {'LB':>4s} | {'Best Config':40s} | {'Ann':>8s} | {'WR':>5s} | {'N':>5s} | "
          f"{'DD':>6s} | {'PF':>5s} | {'Sharpe':>7s}")
    print(f"  {'-' * 110}")

    for lb in ALL_LOOKBACKS:
        lb_results = [r for r in results if f'_LB{lb}_' in r['name']]
        if lb_results:
            best = lb_results[0]
            print(f"  {lb:4d} | {best['name']:40s} | {best['ann']:+7.1f}% | {best['wr']:4.1f}% | "
                  f"{best['n']:5d} | {best['dd']:5.1f}% | {best['pf']:4.2f} | {best['sharpe']:6.2f}")

    # ================================================================
    # INTERACTION HEATMAP: LOOKBACK x Z_THRESHOLD (fixed LOG, best ann per cell)
    # ================================================================
    print(f"\n{'=' * 150}")
    print(f"  INTERACTION: LOOKBACK x Z_THRESHOLD (fixed LOG, best Ann per cell)")
    print(f"{'=' * 150}")

    header = f"  {'LB \\ Z':>7s} |"
    for zt in z_thresholds_tested:
        header += f" {'Z=' + str(zt):>10s} |"
    print(header)
    print(f"  {'-' * (9 + 13 * len(z_thresholds_tested))}")

    for lb in ALL_LOOKBACKS:
        row = f"  {lb:7d} |"
        for zt in z_thresholds_tested:
            cell_results = [r for r in results
                            if f'_LB{lb}_' in r['name'] and f'_Z{zt:.1f}_' in r['name']]
            if cell_results:
                best = cell_results[0]
                row += f" {best['ann']:+9.1f}% |"
            else:
                row += f" {'N/A':>10s} |"
        print(row)

    # ================================================================
    # PER-PAIR STATS (for #1 overall config)
    # ================================================================
    if results:
        best_overall = results[0]
        print(f"\n{'=' * 150}")
        print(f"  PER-PAIR STATS for #1 Config: {best_overall['name']}")
        print(f"  Ann={best_overall['ann']:+.1f}%  WR={best_overall['wr']:.1f}%  "
              f"N={best_overall['n']}  DD={best_overall['dd']:.1f}%  PF={best_overall['pf']:.2f}  "
              f"Sharpe={best_overall['sharpe']:.2f}")
        print(f"{'=' * 150}")
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

    print(f"\n{'=' * 150}")
    print(f"  RIGOROUS 6-WINDOW WALK-FORWARD (Top 5 configs)")
    print(f"  Windows: {WF_WINDOWS}")
    print(f"{'=' * 150}")

    wf_all = []  # (config_name, train_end, test_year, result)
    wf_by_config = {}

    for rank, cfg in enumerate(top5_for_wf):
        cfg_name = cfg['name']
        # Find matching config definition
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
    print(f"\n{'=' * 150}")
    print(f"  WALK-FORWARD AGGREGATE (average across all windows per config)")
    print(f"{'=' * 150}")

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

    print(f"  {'#':>2s} | {'Config':40s} | {'Avg Ann':>8s} | {'Med Ann':>8s} | {'Min Ann':>8s} | "
          f"{'Max Ann':>8s} | {'Avg WR':>6s} | {'Avg N':>6s} | {'Avg DD':>7s} | {'Avg PF':>6s} | "
          f"{'Avg Sh':>6s} | {'Pos/Win':>7s}")
    print(f"  {'-' * 160}")

    for i, w in enumerate(wf_avg):
        print(f"  {i + 1:2d} | {w['name']:40s} | {w['avg_ann']:+7.1f}% | {w['med_ann']:+7.1f}% | "
              f"{w['min_ann']:+7.1f}% | {w['max_ann']:+7.1f}% | {w['avg_wr']:5.1f}% | "
              f"{w['avg_n']:5.0f} | {w['avg_dd']:6.1f}% | {w['avg_pf']:5.2f} | "
              f"{w['avg_sharpe']:5.2f} | {w['n_positive']}/{w['n_windows']}")

    # ================================================================
    # WALK-FORWARD WINDOW-BY-WINDOW DETAIL
    # ================================================================
    print(f"\n{'=' * 150}")
    print(f"  WALK-FORWARD WINDOW-BY-WINDOW DETAIL")
    print(f"{'=' * 150}")

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
    # MODE SELECTION STATS ACROSS WF (for adaptive configs)
    # ================================================================
    print(f"\n{'=' * 150}")
    print(f"  MODE SELECTION STATS ACROSS WALK-FORWARD (for adaptive configs)")
    print(f"{'=' * 150}")

    for i, w in enumerate(wf_avg):
        cfg_name = w['name']
        # Only for adaptive configs
        if not cfg_name.startswith('T1_') and not cfg_name.startswith('T2_'):
            continue

        # Collect mode usage across all WF windows
        total_mode_usage = {}
        for _, _, r in w['window_details']:
            for mode_key, mu in r.get('mode_usage', {}).items():
                if mode_key not in total_mode_usage:
                    total_mode_usage[mode_key] = {'n': 0, 'w': 0, 'pnl': 0.0}
                total_mode_usage[mode_key]['n'] += mu['n']
                total_mode_usage[mode_key]['w'] += mu['w']
                total_mode_usage[mode_key]['pnl'] += mu['pnl']

        if total_mode_usage:
            total_trades = sum(mu['n'] for mu in total_mode_usage.values())
            print(f"\n  [{i + 1}] {cfg_name} (WF Avg Ann={w['avg_ann']:+.1f}%):")
            for mode_key in sorted(total_mode_usage.keys(),
                                   key=lambda x: -total_mode_usage[x]['n']):
                mu = total_mode_usage[mode_key]
                pct = mu['n'] / max(total_trades, 1) * 100
                wr_m = mu['w'] / max(mu['n'], 1) * 100
                print(f"    {mode_key:15s}: {mu['n']:5d} trades ({pct:5.1f}%)  "
                      f"WR={wr_m:5.1f}%  PnL={mu['pnl']:+12.0f}")

    # ================================================================
    # OVERFITTING CHECK
    # ================================================================
    if wf_avg:
        print(f"\n{'=' * 150}")
        print(f"  OVERFITTING CHECK: Full-Period vs Walk-Forward Correlation")
        print(f"{'=' * 150}")

        full_anns = []
        wf_anns = []
        for w in wf_avg:
            name = w['name']
            full_r = next((r for r in results if r['name'] == name), None)
            if full_r:
                full_anns.append(full_r['ann'])
                wf_anns.append(w['avg_ann'])

        if len(full_anns) > 2:
            corr = np.corrcoef(full_anns, wf_anns)[0, 1]
            decay = np.mean(wf_anns) / max(np.mean(full_anns), 0.01)
            print(f"  Configs tested OOS: {len(full_anns)}")
            print(f"  Full-period avg Ann: {np.mean(full_anns):+.1f}%")
            print(f"  WF avg Ann:          {np.mean(wf_anns):+.1f}%")
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

        # Best/worst single window
        if wf_all:
            best_single = max(wf_all, key=lambda x: x[3]['ann'])
            worst_single = min(wf_all, key=lambda x: x[3]['ann'])
            print(f"  Best single window OOS:  Test {best_single[2]} = "
                  f"{best_single[3]['ann']:+.1f}% ({best_single[0][:40]})")
            print(f"  Worst single window OOS: Test {worst_single[2]} = "
                  f"{worst_single[3]['ann']:+.1f}% ({worst_single[0][:40]})")

    # ================================================================
    # PAIR PROFITABILITY ACROSS TOP 20
    # ================================================================
    if results:
        print(f"\n{'=' * 150}")
        print(f"  PAIR PROFITABILITY ACROSS TOP 20 CONFIGS")
        print(f"{'=' * 150}")

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
    # FINAL SUMMARY
    # ================================================================
    print(f"\n{'=' * 150}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 150}")

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
    print(f"    V57 (global adaptive, raw/pct/log auto-switch): +315.3%")
    print(f"    V60 (LOG spread fixed LB15_Z1.5):              +312.9%")
    if results:
        print(f"    V61 best full-period:                          {results[0]['ann']:+.1f}%")
    if wf_avg:
        print(f"    V61 best WF avg:                               {wf_avg[0]['avg_ann']:+.1f}%")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 150)


if __name__ == '__main__':
    main()
