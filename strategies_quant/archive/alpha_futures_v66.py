"""
Alpha Futures V66 -- Commission Drag Analysis
==============================================
V62 trades ~2800 times over 10 years. Each trade pays COMM=0.0003 on both legs
(open + close) = 0.06% round-trip. With 1-day hold, that's ~280 round-trips/year
x 0.06% = ~16.8% annual drag from commissions alone.

This version tests what happens with lower commission rates:
  - COMM = 0      (theoretical ceiling)
  - COMM = 0.0001 (institutional/negotiated rate)
  - COMM = 0.0002 (mid-tier rate)
  - COMM = 0.0003 (current standard)
  - COMM = 0.0005 (high/retail rate)

Also tests minimum holding period of 2 days to reduce commission drag
(fewer trades but less cost per unit of return).

Strategy: V62's exact logic -- LOG-biased adaptive, 14 pairs, Z=1.0, EP40, MP=1.
Grid: COMM x [0, 0.0001, 0.0002, 0.0003, 0.0005] x hold [1, 2] x Z [0.8, 1.0, 1.2]
= 30 configs. Walk-forward for best.

Key implementation change: precompute raw PnL WITHOUT commission, then apply
commission as a parameter at backtest time.

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

# V62's 14 pairs (13 original + cfi/csfi)
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

# Walk-forward windows: (train_end_year, test_year)
WF_WINDOWS = [
    (2019, 2020),
    (2020, 2021),
    (2021, 2022),
    (2022, 2023),
    (2023, 2024),
    (2024, 2025),
]

# Commission rates to test
COMM_LEVELS = [0, 0.0001, 0.0002, 0.0003, 0.0005]
HOLD_PERIODS = [1, 2]
Z_THRESHOLDS = [0.8, 1.0, 1.2]


def main():
    t_start = time.time()
    print("=" * 160)
    print("Alpha Futures V66 -- Commission Drag Analysis")
    print("V62 baseline: LOG-biased adaptive, 14 pairs, Z=1.0, EP40, MP=1, COMM=0.0003")
    print("Grid: COMM x [0, 0.0001, 0.0002, 0.0003, 0.0005] x hold [1, 2] x Z [0.8, 1.0, 1.2]")
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

    # Build pair index mapping
    pair_indices = []
    for down_sym, up_sym in PAIRS:
        down_si = sym_to_si.get(down_sym, -1)
        up_si = sym_to_si.get(up_sym, -1)
        if down_si >= 0 and up_si >= 0:
            pair_indices.append((down_si, up_si, down_sym, up_sym))
        else:
            print(f"  WARNING: pair ({down_sym}, {up_sym}) not found")
    print(f"  Active pairs: {len(pair_indices)}")

    # ================================================================
    # PRECOMPUTE SPREADS AND Z-SCORES FOR ALL MODES x LOOKBACKS
    # ================================================================
    print("\n[Signals] Precomputing spreads and z-scores...", flush=True)
    t0 = time.time()

    z_scores = {m: {} for m in ALL_MODES}

    all_pair_set = set()
    for down_si, up_si, down_sym, up_sym in pair_indices:
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
    # PRECOMPUTE PER-PAIR HYPOTHETICAL RETURNS (NO COMMISSION)
    # ================================================================
    # Key difference from V62: we store raw PnL without commission so we can
    # apply different COMM rates at backtest time.
    print("\n[Signals] Precomputing per-pair raw returns (no commission)...", flush=True)
    t1 = time.time()

    # For adaptive mode selection, we still need a score -- use COMM=0 for scoring
    # since the relative ranking of combos doesn't change with commission.
    pair_raw_daily_return = {}  # (pair_key, (mode, lb, zt)) -> daily_ret (raw, no comm)
    all_zt = Z_THRESHOLDS
    log_bias_combos = [(SPREAD_LOG, 10), (SPREAD_LOG, 15), (SPREAD_LOG, 20),
                       (SPREAD_RAW, 10), (SPREAD_PCT, 10)]

    for zt in all_zt:
        for mode in ALL_MODES:
            for lb in ALL_LOOKBACKS:
                combo_key = (mode, lb, zt)
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
                        # Raw PnL % without commission
                        pnl_pct = (pnl_down + pnl_up) / invested * 100 if invested > 0 else 0
                        daily_ret[di] = pnl_pct

                    pair_raw_daily_return[(pair_key, combo_key)] = daily_ret

    # Global combo daily returns (average across all pairs, no commission)
    global_raw_daily_return = {}
    for zt in all_zt:
        for mode in ALL_MODES:
            for lb in ALL_LOOKBACKS:
                combo_key = (mode, lb, zt)
                daily_ret = np.full(ND, np.nan)
                for di in range(MIN_TRAIN + 1, ND):
                    pair_rets = []
                    for down_si, up_si, down_sym, up_sym in pair_indices:
                        pk = (down_si, up_si, down_sym, up_sym)
                        pr = pair_raw_daily_return.get((pk, combo_key))
                        if pr is not None and not np.isnan(pr[di]):
                            pair_rets.append(pr[di])
                    if pair_rets:
                        daily_ret[di] = np.mean(pair_rets)
                global_raw_daily_return[combo_key] = daily_ret

    print(f"  Raw returns precomputed ({time.time() - t1:.1f}s)", flush=True)

    # ================================================================
    # BACKTEST ENGINE (COMM as parameter, hold_max as parameter)
    # ================================================================
    def run_backtest(z_thresh=1.0, hold_max=1, exit_z=0.0, max_pairs=1,
                     mode_type='adaptive_log_bias',
                     eval_period=40,
                     candidate_combos=None,
                     pair_indices_arg=None,
                     start_year=None, end_year=None,
                     comm=0.0003,
                     config_name=""):
        """
        Backtest engine with COMM and hold_max as parameters.
        Commission is applied per trade at open and close.
        """
        if pair_indices_arg is None:
            pair_indices_arg = pair_indices
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
                            daily_ret = global_raw_daily_return.get(combo_key)
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
                    cost = (entry_val_down + entry_val_up) * comm + \
                           (exit_val_down + exit_val_up) * comm

                    total_pnl = pnl_down + pnl_up - cost
                    invested = entry_val_down + entry_val_up
                    pnl_pct = total_pnl / invested * 100 if invested > 0 else 0

                    if pos_dir == 1:
                        cash_return = c_down * mult_down * lots_down - c_up * mult_up * lots_up
                    else:
                        cash_return = -c_down * mult_down * lots_down + c_up * mult_up * lots_up

                    cash += pos['cash_invested'] + cash_return - (exit_val_down + exit_val_up) * comm

                    trades.append({
                        'pnl_abs': total_pnl,
                        'pnl_pct': pnl_pct,
                        'pnl_raw': pnl_down + pnl_up,  # before commission
                        'cost': cost,
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

            # Determine which combo to use
            if mode_type == 'fixed':
                use_mode, use_lb = candidate_combos[0]
            else:
                use_mode, use_lb = current_combo

            candidates = []
            for down_si, up_si, down_sym, up_sym in pair_indices_arg:
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

            opened = 0
            for _, down_si, up_si, down_sym, up_sym, z_val in candidates:
                if opened >= n_can_open:
                    break

                c_down = C[down_si, di]
                c_up = C[up_si, di]
                if np.isnan(c_down) or c_down <= 0 or np.isnan(c_up) or c_up <= 0:
                    continue

                mult_down = MULT.get(down_sym, DEF_MULT)
                mult_up = MULT.get(up_sym, DEF_MULT)

                capital_for_pair = cash / max(1, max_pairs)
                cash_per_leg = capital_for_pair / 2

                lots_down = int(cash_per_leg / (c_down * mult_down * (1 + comm)))
                lots_up = int(cash_per_leg / (c_up * mult_up * (1 + comm)))
                if lots_down <= 0 or lots_up <= 0:
                    continue

                cost_down = c_down * mult_down * lots_down * (1 + comm)
                cost_up = c_up * mult_up * lots_up * (1 + comm)
                total_cost = cost_down + cost_up
                if total_cost > cash:
                    scale = cash * 0.95 / total_cost
                    lots_down = max(1, int(lots_down * scale))
                    lots_up = max(1, int(lots_up * scale))
                    cost_down = c_down * mult_down * lots_down * (1 + comm)
                    cost_up = c_up * mult_up * lots_up * (1 + comm)
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
            cost = (entry_val_down + entry_val_up) * comm + \
                   (exit_val_down + exit_val_up) * comm

            total_pnl = pnl_down + pnl_up - cost
            invested = entry_val_down + entry_val_up
            pnl_pct = total_pnl / invested * 100 if invested > 0 else 0

            if pos['dir'] == 1:
                cash_return = c_down * mult_down * lots_down - c_up * mult_up * lots_up
            else:
                cash_return = -c_down * mult_down * lots_down + c_up * mult_up * lots_up

            cash += pos['cash_invested'] + cash_return - (exit_val_down + exit_val_up) * comm

            trades.append({
                'pnl_abs': total_pnl,
                'pnl_pct': pnl_pct,
                'pnl_raw': pnl_down + pnl_up,
                'cost': cost,
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

        # Commission cost analysis
        total_commission = sum(t['cost'] for t in trades)
        total_raw_pnl = sum(t['pnl_raw'] for t in trades)
        total_net_pnl = sum(t['pnl_abs'] for t in trades)

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
                year_stats[y] = {'n': 0, 'w': 0, 'pnl': 0.0, 'pnl_abs_sum': 0.0,
                                 'raw_pnl': 0.0, 'commission': 0.0}
            year_stats[y]['n'] += 1
            if t['pnl_abs'] > 0:
                year_stats[y]['w'] += 1
            year_stats[y]['pnl'] += t['pnl_pct']
            year_stats[y]['pnl_abs_sum'] += t['pnl_abs']
            year_stats[y]['raw_pnl'] += t['pnl_raw']
            year_stats[y]['commission'] += t['cost']

        pair_stats = {}
        for t in trades:
            p = t['pair_label']
            if p not in pair_stats:
                pair_stats[p] = {'n': 0, 'w': 0, 'pnl': 0.0, 'cost': 0.0}
            pair_stats[p]['n'] += 1
            if t['pnl_abs'] > 0:
                pair_stats[p]['w'] += 1
            pair_stats[p]['pnl'] += t['pnl_abs']
            pair_stats[p]['cost'] += t['cost']

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
            'total_commission': round(total_commission, 0),
            'total_raw_pnl': round(total_raw_pnl, 0),
            'total_net_pnl': round(total_net_pnl, 0),
            'comm_drag_pct': round(total_commission / max(total_raw_pnl, 1) * 100, 1),
            'yearly': year_stats,
            'pair_stats': pair_stats,
            'trades': trades,
        }

    # ================================================================
    # BUILD CONFIGURATION GRID
    # ================================================================
    configs = []

    log_bias_combos = [(SPREAD_LOG, 10), (SPREAD_LOG, 15), (SPREAD_LOG, 20),
                       (SPREAD_RAW, 10), (SPREAD_PCT, 10)]

    for comm in COMM_LEVELS:
        for hold in HOLD_PERIODS:
            for zt in Z_THRESHOLDS:
                name = f"COMM{comm:.4f}_H{hold}_Z{zt:.1f}"
                configs.append({
                    'z_thresh': zt, 'hold_max': hold, 'exit_z': 0.0, 'max_pairs': 1,
                    'mode_type': 'adaptive_log_bias', 'eval_period': 40,
                    'candidate_combos': log_bias_combos,
                    'pair_indices_arg': pair_indices, 'start_year': None, 'end_year': None,
                    'comm': comm,
                    'config_name': name,
                })

    total_combos = len(configs)
    print(f"\n{'=' * 160}")
    print(f"  FULL-PERIOD PARAMETER SWEEP ({total_combos} configs)")
    print(f"  COMM x {COMM_LEVELS}")
    print(f"  Hold x {HOLD_PERIODS}")
    print(f"  Z x {Z_THRESHOLDS}")
    print(f"{'=' * 160}")

    results = []
    t_sweep_start = time.time()

    for ci, cfg in enumerate(configs):
        r = run_backtest(**cfg)
        if r is not None:
            results.append(r)

        if (ci + 1) % 10 == 0:
            elapsed = time.time() - t_sweep_start
            print(f"  [{ci + 1}/{total_combos}] {len(results)} with results ({elapsed:.1f}s)", flush=True)

    results.sort(key=lambda x: -x['ann'])
    print(f"\n  Sweep complete: {len(results)}/{total_combos} configs ({time.time() - t_sweep_start:.1f}s)",
          flush=True)

    # ================================================================
    # ALL RESULTS TABLE
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  ALL RESULTS (sorted by annual return)")
    print(f"{'=' * 160}")
    print(f"  {'#':>2s} | {'Config':35s} | {'Ann':>8s} | {'WR':>5s} | {'N':>5s} | "
          f"{'DD':>6s} | {'PF':>5s} | {'Sharpe':>7s} | {'AvgW':>6s} | {'AvgL':>6s} | "
          f"{'AvgD':>5s} | {'Cash':>12s} | {'TotComm':>10s} | {'CommDrag':>8s}")
    print(f"  {'-' * 175}")

    for i, r in enumerate(results):
        print(f"  {i + 1:2d} | {r['name']:35s} | {r['ann']:+7.1f}% | {r['wr']:4.1f}% | "
              f"{r['n']:5d} | {r['dd']:5.1f}% | {r['pf']:4.2f} | {r['sharpe']:6.2f} | "
              f"{r['avg_win']:+5.2f}% | {r['avg_loss']:5.2f}% | {r['avg_days']:4.1f} | "
              f"{r['cash']:11.0f} | {r['total_commission']:9.0f} | {r['comm_drag_pct']:6.1f}%")

    # ================================================================
    # COMMISSION IMPACT ANALYSIS
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  COMMISSION IMPACT ANALYSIS")
    print(f"  (For each COMM level, average across all hold/Z configs)")
    print(f"{'=' * 160}")

    for comm_val in COMM_LEVELS:
        subset = [r for r in results if f"COMM{comm_val:.4f}" in r['name']]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            avg_wr = np.mean([r['wr'] for r in subset])
            avg_n = np.mean([r['n'] for r in subset])
            avg_dd = np.mean([r['dd'] for r in subset])
            avg_sharpe = np.mean([r['sharpe'] for r in subset])
            avg_comm = np.mean([r['total_commission'] for r in subset])
            avg_raw_pnl = np.mean([r['total_raw_pnl'] for r in subset])
            avg_net_pnl = np.mean([r['total_net_pnl'] for r in subset])
            best = max(subset, key=lambda x: x['ann'])
            rt_cost_pct = comm_val * 2 * 100  # round-trip cost as % of notional
            print(f"\n  COMM={comm_val:.4f} (round-trip={rt_cost_pct:.3f}%):")
            print(f"    Avg Ann={avg_ann:+7.1f}%  Avg WR={avg_wr:5.1f}%  Avg N={avg_n:5.0f}  "
                  f"Avg DD={avg_dd:5.1f}%  Avg Sharpe={avg_sharpe:5.2f}")
            print(f"    Avg Total Commission={avg_comm:10.0f}  Avg Raw PnL={avg_raw_pnl:+12.0f}  "
                  f"Avg Net PnL={avg_net_pnl:+12.0f}")
            print(f"    Best: {best['name']}  Ann={best['ann']:+.1f}%  N={best['n']}  "
                  f"WR={best['wr']:.1f}%  DD={best['dd']:.1f}%  Sharpe={best['sharpe']:.2f}")

    # ================================================================
    # HOLD PERIOD COMPARISON
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  HOLD PERIOD COMPARISON")
    print(f"{'=' * 160}")

    for hold_val in HOLD_PERIODS:
        subset = [r for r in results if f"_H{hold_val}_" in r['name']]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            avg_n = np.mean([r['n'] for r in subset])
            avg_wr = np.mean([r['wr'] for r in subset])
            avg_dd = np.mean([r['dd'] for r in subset])
            avg_comm = np.mean([r['total_commission'] for r in subset])
            best = max(subset, key=lambda x: x['ann'])
            print(f"\n  Hold={hold_val} day(s):")
            print(f"    Avg Ann={avg_ann:+7.1f}%  Avg N={avg_n:5.0f}  Avg WR={avg_wr:5.1f}%  "
                  f"Avg DD={avg_dd:5.1f}%  Avg Comm={avg_comm:10.0f}")
            print(f"    Best: {best['name']}  Ann={best['ann']:+.1f}%  N={best['n']}  "
                  f"WR={best['wr']:.1f}%  DD={best['dd']:.1f}%")

    # ================================================================
    # Z-THRESHOLD COMPARISON
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  Z-THRESHOLD COMPARISON")
    print(f"{'=' * 160}")

    for zt_val in Z_THRESHOLDS:
        subset = [r for r in results if f"_Z{zt_val:.1f}" in r['name']]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            avg_n = np.mean([r['n'] for r in subset])
            avg_wr = np.mean([r['wr'] for r in subset])
            avg_dd = np.mean([r['dd'] for r in subset])
            best = max(subset, key=lambda x: x['ann'])
            print(f"\n  Z={zt_val:.1f}:")
            print(f"    Avg Ann={avg_ann:+7.1f}%  Avg N={avg_n:5.0f}  Avg WR={avg_wr:5.1f}%  "
                  f"Avg DD={avg_dd:5.1f}%")
            print(f"    Best: {best['name']}  Ann={best['ann']:+.1f}%  N={best['n']}  "
                  f"WR={best['wr']:.1f}%  DD={best['dd']:.1f}%")

    # ================================================================
    # COMMISSION DRAG SENSITIVITY: Same Z/hold, varying COMM
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  COMMISSION DRAG SENSITIVITY (same strategy, different COMM)")
    print(f"  Baseline: adaptive LOG, Z=1.0, Hold=1, EP40, MP=1, 14 pairs")
    print(f"{'=' * 160}")
    print(f"  {'COMM':>8s} | {'RoundTrip':>10s} | {'Ann':>8s} | {'WR':>5s} | {'N':>5s} | "
          f"{'DD':>6s} | {'Sharpe':>7s} | {'TotComm':>10s} | {'RawPnL':>12s} | "
          f"{'NetPnL':>12s} | {'Drag%':>6s} | {'AnnDelta':>9s}")
    print(f"  {'-' * 140}")

    baseline_key = "_H1_Z1.0"
    baseline_results = [r for r in results if baseline_key in r['name']]
    baseline_results.sort(key=lambda x: x['total_commission'])  # sort by comm = 0 first

    ref_ann = None
    for r in baseline_results:
        rt = r['total_commission'] / max(r['n'], 1) / max(r['total_raw_pnl'] / r['n'], 1) * 100 if r['n'] > 0 else 0
        # Extract comm from name
        comm_str = r['name'].split('_')[0]  # e.g. "COMM0.0000"
        comm_val = float(comm_str.replace('COMM', ''))
        round_trip_pct = comm_val * 2 * 100

        if ref_ann is None and comm_val == 0:
            ref_ann = r['ann']

        ann_delta = r['ann'] - (ref_ann if ref_ann is not None else r['ann'])

        # drag as % of raw PnL
        drag_of_raw = r['total_commission'] / max(r['total_raw_pnl'], 1) * 100

        print(f"  {comm_val:8.4f} | {round_trip_pct:9.3f}% | {r['ann']:+7.1f}% | {r['wr']:4.1f}% | "
              f"{r['n']:5d} | {r['dd']:5.1f}% | {r['sharpe']:6.2f} | {r['total_commission']:9.0f} | "
              f"{r['total_raw_pnl']:+11.0f} | {r['total_net_pnl']:+11.0f} | {drag_of_raw:5.1f}% | "
              f"{ann_delta:+7.1f}%")

    if ref_ann is not None:
        print(f"\n  -> Zero-commission ceiling: {ref_ann:+.1f}% annual")
        for r in baseline_results:
            comm_val = float(r['name'].split('_')[0].replace('COMM', ''))
            if comm_val > 0:
                pct_lost = (ref_ann - r['ann']) / max(abs(ref_ann), 0.01) * 100
                print(f"  -> COMM={comm_val:.4f}: loses {ref_ann - r['ann']:+.1f}% annual "
                      f"({pct_lost:.1f}% of theoretical ceiling) to commission drag")

    # ================================================================
    # HOLD=1 vs HOLD=2 AT EACH COMM LEVEL
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  HOLD=1 vs HOLD=2 AT EACH COMM LEVEL (Z=1.0)")
    print(f"{'=' * 160}")

    for comm_val in COMM_LEVELS:
        print(f"\n  COMM={comm_val:.4f}:")
        for hold_val in HOLD_PERIODS:
            subset = [r for r in results if f"COMM{comm_val:.4f}_H{hold_val}_" in r['name']]
            if subset:
                # Average across Z thresholds
                avg_ann = np.mean([r['ann'] for r in subset])
                avg_n = np.mean([r['n'] for r in subset])
                avg_comm = np.mean([r['total_commission'] for r in subset])
                best = max(subset, key=lambda x: x['ann'])
                print(f"    Hold={hold_val}: Avg Ann={avg_ann:+7.1f}%  Avg N={avg_n:5.0f}  "
                      f"Avg Comm={avg_comm:10.0f}")
                print(f"      Best: {best['name']}  Ann={best['ann']:+.1f}%  N={best['n']}  "
                      f"WR={best['wr']:.1f}%  DD={best['dd']:.1f}%  Sharpe={best['sharpe']:.2f}")

    # ================================================================
    # YEARLY BREAKDOWN FOR BEST CONFIG
    # ================================================================
    if results:
        best_overall = results[0]
        print(f"\n{'=' * 160}")
        print(f"  YEARLY BREAKDOWN FOR #1 Config: {best_overall['name']}")
        print(f"  Ann={best_overall['ann']:+.1f}%  WR={best_overall['wr']:.1f}%  "
              f"N={best_overall['n']}  DD={best_overall['dd']:.1f}%  PF={best_overall['pf']:.2f}  "
              f"Sharpe={best_overall['sharpe']:.2f}")
        print(f"{'=' * 160}")
        print(f"  {'Year':>6s} | {'N':>5s} | {'WR':>5s} | {'Net PnL':>12s} | {'Raw PnL':>12s} | "
              f"{'Commission':>12s} | {'Comm%':>6s}")
        print(f"  {'-' * 75}")

        for y in sorted(best_overall['yearly'].keys()):
            ys = best_overall['yearly'][y]
            wr_y = ys['w'] / max(ys['n'], 1) * 100
            comm_pct = ys['commission'] / max(ys['raw_pnl'], 1) * 100
            print(f"  {y:6d} | {ys['n']:5d} | {wr_y:4.1f}% | {ys['pnl_abs_sum']:+11.0f} | "
                  f"{ys['raw_pnl']:+11.0f} | {ys['commission']:11.0f} | {comm_pct:5.1f}%")

        # Per-pair breakdown
        print(f"\n  PER-PAIR STATS for #1 config:")
        print(f"  {'Pair':25s} | {'N':>5s} | {'WR':>5s} | {'Abs PnL':>12s} | {'Commission':>12s} | "
              f"{'Comm%':>6s}")
        print(f"  {'-' * 85}")

        for p in sorted(best_overall['pair_stats'].keys(),
                        key=lambda x: -best_overall['pair_stats'][x]['pnl']):
            ps = best_overall['pair_stats'][p]
            wr_p = ps['w'] / max(ps['n'], 1) * 100
            comm_pct = ps['cost'] / max(abs(ps['pnl'] + ps['cost']), 1) * 100
            print(f"  {p:25s} | {ps['n']:5d} | {wr_p:4.1f}% | {ps['pnl']:+11.0f} | "
                  f"{ps['cost']:11.0f} | {comm_pct:5.1f}%")

    # ================================================================
    # YEARLY FOR TOP 5
    # ================================================================
    if len(results) >= 2:
        print(f"\n  YEARLY BREAKDOWN FOR TOP 5 CONFIGS:")
        for idx, r in enumerate(results[:5]):
            print(f"\n  #{idx + 1}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.1f}%, "
                  f"DD={r['dd']:.1f}%, Sharpe={r['sharpe']:.2f}, N={r['n']}, "
                  f"TotalComm={r['total_commission']:.0f})")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:4d}t  WR={wr_y:5.1f}%  NetPnL={ys['pnl_abs_sum']:+.0f}  "
                      f"RawPnL={ys['raw_pnl']:+.0f}  Comm={ys['commission']:.0f}")

    # ================================================================
    # COMMISSION SAVINGS TABLE: How much extra return from lower COMM?
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  COMMISSION SAVINGS TABLE (Z=1.0, Hold=1)")
    print(f"  Delta vs COMM=0.0003 baseline")
    print(f"{'=' * 160}")

    # Find the COMM=0.0003, H=1, Z=1.0 result as reference
    ref_result = next((r for r in results if r['name'] == "COMM0.0003_H1_Z1.0"), None)
    if ref_result:
        ref_ann_val = ref_result['ann']
        ref_cash = ref_result['cash']
        print(f"\n  Baseline (COMM=0.0003): Ann={ref_ann_val:+.1f}%  Cash={ref_cash:.0f}")
        print(f"\n  {'COMM':>8s} | {'Ann':>8s} | {'Ann Delta':>9s} | {'Cash':>12s} | "
              f"{'Cash Delta':>12s} | {'TotalComm':>10s} | {'Comm Saved':>12s}")
        print(f"  {'-' * 90}")

        for comm_val in COMM_LEVELS:
            r = next((x for x in results if x['name'] == f"COMM{comm_val:.4f}_H1_Z1.0"), None)
            if r:
                ann_delta = r['ann'] - ref_ann_val
                cash_delta = r['cash'] - ref_cash
                comm_saved = ref_result['total_commission'] - r['total_commission']
                print(f"  {comm_val:8.4f} | {r['ann']:+7.1f}% | {ann_delta:+7.1f}% | "
                      f"{r['cash']:11.0f} | {cash_delta:+11.0f} | {r['total_commission']:9.0f} | "
                      f"{comm_saved:+11.0f}")

    # Also for Hold=2
    ref_result_h2 = next((r for r in results if r['name'] == "COMM0.0003_H2_Z1.0"), None)
    if ref_result_h2:
        ref_ann_val_h2 = ref_result_h2['ann']
        ref_cash_h2 = ref_result_h2['cash']
        print(f"\n  Baseline Hold=2 (COMM=0.0003): Ann={ref_ann_val_h2:+.1f}%  Cash={ref_cash_h2:.0f}")
        print(f"\n  {'COMM':>8s} | {'Ann':>8s} | {'Ann Delta':>9s} | {'Cash':>12s} | "
              f"{'Cash Delta':>12s} | {'TotalComm':>10s} | {'Comm Saved':>12s}")
        print(f"  {'-' * 90}")

        for comm_val in COMM_LEVELS:
            r = next((x for x in results if x['name'] == f"COMM{comm_val:.4f}_H2_Z1.0"), None)
            if r:
                ann_delta = r['ann'] - ref_ann_val_h2
                cash_delta = r['cash'] - ref_cash_h2
                comm_saved = ref_result_h2['total_commission'] - r['total_commission']
                print(f"  {comm_val:8.4f} | {r['ann']:+7.1f}% | {ann_delta:+7.1f}% | "
                      f"{r['cash']:11.0f} | {cash_delta:+11.0f} | {r['total_commission']:9.0f} | "
                      f"{comm_saved:+11.0f}")

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
                      f"PF={r['pf']:4.2f}  Sharpe={r['sharpe']:6.2f}  "
                      f"Comm={r['total_commission']:.0f}")
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
        comms = [r['total_commission'] for _, _, r in window_results]
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
            'avg_comm': np.mean(comms),
            'n_positive': n_positive,
            'n_windows': len(window_results),
            'window_details': window_results,
        })

    wf_avg.sort(key=lambda x: -x['avg_ann'])

    print(f"  {'#':>2s} | {'Config':35s} | {'Avg Ann':>8s} | {'Med Ann':>8s} | {'Min Ann':>8s} | "
          f"{'Max Ann':>8s} | {'Avg WR':>6s} | {'Avg N':>6s} | {'Avg DD':>7s} | {'Avg PF':>6s} | "
          f"{'Avg Sh':>6s} | {'Avg Comm':>10s} | {'Pos/Win':>7s}")
    print(f"  {'-' * 180}")

    for i, w in enumerate(wf_avg):
        print(f"  {i + 1:2d} | {w['name']:35s} | {w['avg_ann']:+7.1f}% | {w['med_ann']:+7.1f}% | "
              f"{w['min_ann']:+7.1f}% | {w['max_ann']:+7.1f}% | {w['avg_wr']:5.1f}% | "
              f"{w['avg_n']:5.0f} | {w['avg_dd']:6.1f}% | {w['avg_pf']:5.2f} | "
              f"{w['avg_sharpe']:5.2f} | {w['avg_comm']:9.0f} | {w['n_positive']}/{w['n_windows']}")

    # ================================================================
    # WALK-FORWARD WINDOW-BY-WINDOW DETAIL
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  WALK-FORWARD WINDOW-BY-WINDOW DETAIL")
    print(f"{'=' * 160}")

    for i, w in enumerate(wf_avg):
        print(f"\n  [{i + 1}] {w['name']}:")
        print(f"  {'Train':>9s} | {'Test':>4s} | {'Ann':>8s} | {'WR':>5s} | {'N':>5s} | "
              f"{'DD':>6s} | {'PF':>5s} | {'Sharpe':>7s} | {'Comm':>10s}")
        print(f"  {'-' * 85}")
        for train_end, test_year, r in sorted(w['window_details'], key=lambda x: x[1]):
            print(f"  -{train_end:4d}    | {test_year:4d} | {r['ann']:+7.1f}% | {r['wr']:4.1f}% | "
                  f"{r['n']:5d} | {r['dd']:5.1f}% | {r['pf']:4.2f} | {r['sharpe']:6.2f} | "
                  f"{r['total_commission']:9.0f}")

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
    # COMMISSION IMPACT ON WALK-FORWARD
    # ================================================================
    if wf_avg:
        print(f"\n{'=' * 160}")
        print(f"  COMMISSION IMPACT ON WALK-FORWARD (avg across all windows)")
        print(f"{'=' * 160}")

        # Group WF results by COMM level
        for comm_val in COMM_LEVELS:
            wf_subset = [w for w in wf_avg if f"COMM{comm_val:.4f}" in w['name']]
            if wf_subset:
                avg_wf_ann = np.mean([w['avg_ann'] for w in wf_subset])
                avg_wf_dd = np.mean([w['avg_dd'] for w in wf_subset])
                avg_wf_sharpe = np.mean([w['avg_sharpe'] for w in wf_subset])
                avg_wf_comm = np.mean([w['avg_comm'] for w in wf_subset])
                best_wf = max(wf_subset, key=lambda x: x['avg_ann'])
                print(f"\n  COMM={comm_val:.4f}:")
                print(f"    Avg WF Ann={avg_wf_ann:+7.1f}%  Avg WF DD={avg_wf_dd:5.1f}%  "
                      f"Avg WF Sharpe={avg_wf_sharpe:5.2f}  Avg Comm={avg_wf_comm:10.0f}")
                print(f"    Best WF: {best_wf['name']}  Avg Ann={best_wf['avg_ann']:+.1f}%  "
                      f"Pos/Win={best_wf['n_positive']}/{best_wf['n_windows']}")

    # ================================================================
    # KEY INSIGHTS SUMMARY
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  KEY INSIGHTS SUMMARY")
    print(f"{'=' * 160}")

    # Find zero-comm and standard-comm results for comparison
    zero_h1_z1 = next((r for r in results if r['name'] == "COMM0.0000_H1_Z1.0"), None)
    std_h1_z1 = next((r for r in results if r['name'] == "COMM0.0003_H1_Z1.0"), None)
    inst_h1_z1 = next((r for r in results if r['name'] == "COMM0.0001_H1_Z1.0"), None)

    if zero_h1_z1 and std_h1_z1:
        print(f"\n  1. COMMISSION DRAG (Hold=1, Z=1.0):")
        print(f"     Zero commission:   Ann={zero_h1_z1['ann']:+.1f}%  Cash={zero_h1_z1['cash']:.0f}  "
              f"Raw PnL={zero_h1_z1['total_raw_pnl']:+.0f}")
        print(f"     Standard (0.03%):  Ann={std_h1_z1['ann']:+.1f}%  Cash={std_h1_z1['cash']:.0f}  "
              f"Net PnL={std_h1_z1['total_net_pnl']:+.0f}  Commission={std_h1_z1['total_commission']:.0f}")
        delta_ann = zero_h1_z1['ann'] - std_h1_z1['ann']
        delta_cash = zero_h1_z1['cash'] - std_h1_z1['cash']
        print(f"     Annual drag:       {delta_ann:+.1f}% annual return lost to commission")
        print(f"     Cash drag:         {delta_cash:+.0f} over the period")

    if inst_h1_z1 and std_h1_z1:
        print(f"\n  2. INSTITUTIONAL RATE BENEFIT (Hold=1, Z=1.0):")
        print(f"     Institutional (0.01%): Ann={inst_h1_z1['ann']:+.1f}%  Cash={inst_h1_z1['cash']:.0f}")
        print(f"     Standard (0.03%):     Ann={std_h1_z1['ann']:+.1f}%  Cash={std_h1_z1['cash']:.0f}")
        print(f"     Extra return:         {inst_h1_z1['ann'] - std_h1_z1['ann']:+.1f}% annual "
              f"from negotiated rates")

    # Hold period comparison
    std_h2_z1 = next((r for r in results if r['name'] == "COMM0.0003_H2_Z1.0"), None)
    zero_h2_z1 = next((r for r in results if r['name'] == "COMM0.0000_H2_Z1.0"), None)

    if std_h1_z1 and std_h2_z1:
        print(f"\n  3. HOLD PERIOD EFFECT (COMM=0.0003, Z=1.0):")
        print(f"     Hold=1: Ann={std_h1_z1['ann']:+.1f}%  N={std_h1_z1['n']}  "
              f"Comm={std_h1_z1['total_commission']:.0f}")
        print(f"     Hold=2: Ann={std_h2_z1['ann']:+.1f}%  N={std_h2_z1['n']}  "
              f"Comm={std_h2_z1['total_commission']:.0f}")
        print(f"     Trade reduction:     {std_h1_z1['n'] - std_h2_z1['n']} fewer trades")
        print(f"     Commission savings:  {std_h1_z1['total_commission'] - std_h2_z1['total_commission']:+.0f}")

    if zero_h1_z1 and zero_h2_z1:
        print(f"\n  4. HOLD PERIOD WITHOUT COMMISSION (Z=1.0):")
        print(f"     Hold=1: Ann={zero_h1_z1['ann']:+.1f}%  N={zero_h1_z1['n']}")
        print(f"     Hold=2: Ann={zero_h2_z1['ann']:+.1f}%  N={zero_h2_z1['n']}")
        print(f"     Pure alpha comparison (no commission noise)")

    # Best overall finding
    if results:
        print(f"\n  5. BEST OVERALL CONFIG:")
        print(f"     {results[0]['name']}")
        print(f"     Ann={results[0]['ann']:+.1f}%  WR={results[0]['wr']:.1f}%  N={results[0]['n']}  "
              f"DD={results[0]['dd']:.1f}%  PF={results[0]['pf']:.2f}  Sharpe={results[0]['sharpe']:.2f}")

    if wf_avg:
        print(f"\n  6. BEST WALK-FORWARD CONFIG:")
        print(f"     {wf_avg[0]['name']}")
        print(f"     WF Avg Ann={wf_avg[0]['avg_ann']:+.1f}%  WF Med={wf_avg[0]['med_ann']:+.1f}%  "
              f"Min={wf_avg[0]['min_ann']:+.1f}%  Max={wf_avg[0]['max_ann']:+.1f}%  "
              f"Pos/Win={wf_avg[0]['n_positive']}/{wf_avg[0]['n_windows']}")

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

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 160)


if __name__ == '__main__':
    main()
