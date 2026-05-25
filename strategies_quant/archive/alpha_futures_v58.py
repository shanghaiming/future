"""
Alpha Futures V58 -- Rigorous Rolling Walk-Forward Validation of V55 Champion
=============================================================================
V55 claimed +307.3% full period and WF2022 +859.6%. The WF2022 number is
suspiciously high. This script performs a proper rolling-origin walk-forward
test to verify that V55's out-of-sample performance is genuine.

Method: Rolling origin walk-forward with expanding training windows.
  Window 1: Train 2016-2019, Test 2020
  Window 2: Train 2016-2020, Test 2021
  Window 3: Train 2016-2021, Test 2022
  Window 4: Train 2016-2022, Test 2023
  Window 5: Train 2016-2023, Test 2024
  Window 6: Train 2016-2024, Test 2025

For each window:
  1. On training data, find the best parameters using V55's adaptive approach
     (eval_period x z_threshold x spread_mode x lookback grid)
  2. Apply the best training parameters to the TEST period
  3. Report test-period performance

Also tests V52's config (LB10_Z1.0_H1_EZ0_MP1_raw) as a fixed baseline.

The key question: Is V55's adaptive approach genuinely adding value OOS,
or is it just overfitting to the full-period data?
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

# Standard 13 pairs
PAIRS = [
    ('rbfi', 'ifi'), ('hcfi', 'ifi'), ('hcfi', 'rbfi'),
    ('jfi', 'jmfi'), ('mafi', 'scfi'), ('fufi', 'scfi'),
    ('bfi', 'scfi'), ('mfi', 'afi'), ('yfi', 'afi'),
    ('pfi', 'yfi'), ('ppfi', 'mafi'), ('vfi', 'mafi'),
    ('egfi', 'mafi'),
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
}

# Spread modes
SPREAD_RAW = 'raw'
SPREAD_PCT = 'pct'
SPREAD_LOG = 'log'
SPREAD_ADAPTIVE = 'adaptive'
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
    print("=" * 140)
    print("Alpha Futures V58 -- Rigorous Rolling Walk-Forward Validation")
    print("V55 champion: +307.3% full period, WF2022 +859.6% (SUSPICIOUS)")
    print("Method: Rolling origin walk-forward, train on expanding window, test on next year")
    print("=" * 140)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()
    sym_to_si = {syms[si]: si for si in range(NS)}

    # Build pair index mapping
    pair_indices = []
    for down_sym, up_sym in PAIRS:
        down_si = sym_to_si.get(down_sym, -1)
        up_si = sym_to_si.get(up_sym, -1)
        if down_si >= 0 and up_si >= 0:
            pair_indices.append((down_si, up_si, down_sym, up_sym))
        else:
            print(f"  WARNING: pair ({down_sym}, {up_sym}) not found")

    print(f"  {NS} commodities, {ND} days, {len(pair_indices)} active pairs")
    print(f"  Date range: {dates[0]} to {dates[-1]}")

    # Find day indices for each year boundary
    year_start_di = {}
    year_end_di = {}
    for di in range(ND):
        y = dates[di].year
        if y not in year_start_di:
            year_start_di[y] = di
        year_end_di[y] = di

    print(f"  Years in data: {sorted(year_start_di.keys())}")

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
    # BACKTEST ENGINE -- Supports date-range restriction for train/test
    # ================================================================
    def run_backtest(mode, lookback, z_thresh, hold_max=1, exit_z=0.0, max_pairs=1,
                     # Date range
                     start_year=None, end_year=None,
                     config_name=""):
        """
        Run pair trading backtest with a single fixed mode+lookback.

        Args:
            mode: spread mode (raw/pct/log)
            lookback: z-score lookback
            z_thresh: z-score entry threshold
            start_year: if set, only trade from this year onward
            end_year: if set, only trade up to (inclusive) this year
        """
        cash = float(CASH0)
        trades = []
        pair_positions = []

        # Precompute per-pair z-scores for the given mode+lookback
        pair_z = {}
        for down_si, up_si, down_sym, up_sym in pair_indices:
            pair_z[(down_si, up_si)] = z_scores[mode].get((down_si, up_si), {}).get(lookback)

        # Determine date range
        start_di = MIN_TRAIN
        end_di = ND
        if start_year is not None:
            if start_year in year_start_di:
                start_di = year_start_di[start_year]
            else:
                # Year not in data
                return None
        if end_year is not None:
            if end_year in year_end_di:
                end_di = year_end_di[end_year] + 1  # inclusive
            else:
                return None

        for di in range(start_di, end_di):
            year = dates[di].year

            # --- Manage existing pair positions ---
            new_positions = []
            for pos in pair_positions:
                p_down_si = pos['down_si']
                p_up_si = pos['up_si']
                z_arr = pair_z.get((p_down_si, p_up_si))
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
            for down_si, up_si, down_sym, up_sym in pair_indices:
                if down_si in occupied or up_si in occupied:
                    continue

                z_arr = pair_z.get((down_si, up_si))
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

        # Compute annualized return for the actual test period
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
            'trades': trades,
        }

    # ================================================================
    # PARAMETER GRID FOR TRAINING
    # ================================================================
    eval_periods = [20, 40, 60]
    z_thresholds = [0.8, 1.0, 1.2, 1.5]
    spread_modes = [SPREAD_RAW, SPREAD_PCT, SPREAD_LOG]
    lookbacks = [5, 7, 10, 15, 20]

    # V52 baseline config: LB10_Z1.0_H1_EZ0_MP1_raw
    V52_CONFIG = {
        'mode': SPREAD_RAW,
        'lookback': 10,
        'z_thresh': 1.0,
        'hold_max': 1,
        'exit_z': 0.0,
        'max_pairs': 1,
    }

    # ================================================================
    # ROLLING WALK-FORWARD
    # ================================================================
    print("\n" + "=" * 140)
    print("  ROLLING WALK-FORWARD VALIDATION")
    print("=" * 140)
    print(f"\n  Training grid: {len(eval_periods)} eval_periods x "
          f"{len(z_thresholds)} z_thresholds x {len(spread_modes)} modes x "
          f"{len(lookbacks)} lookbacks = "
          f"{len(eval_periods) * len(z_thresholds) * len(spread_modes) * len(lookbacks)} combos")
    print(f"  V52 baseline: {V52_CONFIG['mode']}_LB{V52_CONFIG['lookback']}_Z{V52_CONFIG['z_thresh']}")
    print(f"  Windows: {len(WF_WINDOWS)}")

    wf_results = []  # Overall results storage
    v52_test_results = []  # V52 baseline on each test year

    for win_idx, (train_end, test_year) in enumerate(WF_WINDOWS):
        print(f"\n{'=' * 140}")
        print(f"  WINDOW {win_idx + 1}/{len(WF_WINDOWS)}: "
              f"Train 2016-{train_end}, Test {test_year}")
        print(f"{'=' * 140}")

        # Check if test year exists in data
        if test_year not in year_start_di:
            print(f"  SKIP: Test year {test_year} not in data")
            continue

        # ---- STEP 1: Train all parameter combos on training period ----
        print(f"\n  [Training] Testing {len(eval_periods) * len(z_thresholds) * len(spread_modes) * len(lookbacks)} "
              f"configs on 2016-{train_end}...", flush=True)
        t_train_start = time.time()

        train_results = []
        total_combos = len(eval_periods) * len(z_thresholds) * len(spread_modes) * len(lookbacks)
        combo_count = 0

        for ep in eval_periods:
            for zt in z_thresholds:
                for mode in spread_modes:
                    for lb in lookbacks:
                        combo_count += 1
                        name = f"EP{ep}_{mode}_LB{lb}_Z{zt:.1f}"
                        r = run_backtest(
                            mode=mode, lookback=lb, z_thresh=zt,
                            hold_max=1, exit_z=0.0, max_pairs=1,
                            start_year=None,  # use full train from start
                            end_year=train_end,
                            config_name=name,
                        )
                        if r is not None:
                            train_results.append(r)

                        if combo_count % 30 == 0:
                            print(f"    [{combo_count}/{total_combos}] {len(train_results)} "
                                  f"with results ({time.time() - t_train_start:.1f}s)", flush=True)

        train_results.sort(key=lambda x: -x['ann'])
        print(f"  Training done: {len(train_results)} configs evaluated "
              f"({time.time() - t_train_start:.1f}s)", flush=True)

        # Show top 5 training results
        print(f"\n  TOP 5 TRAINING RESULTS (2016-{train_end}):")
        print(f"  {'Config':35s} | {'Ann':>7s} | {'WR':>5s} | {'N':>5s} | {'DD':>6s} | {'PF':>4s} | {'Sharpe':>6s}")
        print(f"  {'-' * 85}")
        for r in train_results[:5]:
            print(f"  {r['name']:35s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
                  f"{r['n']:5d} | {r['dd']:6.1f}% | {r['pf']:4.2f} | {r['sharpe']:6.2f}")

        if not train_results:
            print(f"  WARNING: No training results for window {win_idx + 1}, skipping test")
            continue

        # ---- STEP 2: Apply top 5 training configs to test period ----
        print(f"\n  [Testing] Applying top 5 training configs to test year {test_year}...", flush=True)

        top5_train = train_results[:5]
        test_results_window = []

        for r_train in top5_train:
            # Parse config name: EP{ep}_{mode}_LB{lb}_Z{zt}
            name = r_train['name']
            parts = name.split('_')
            # EP is first part
            ep_val = parts[0]  # just for reference
            mode_val = parts[1]
            lb_val = int(parts[2][2:])
            zt_val = float(parts[3][1:])

            test_name = f"Train{train_end}_Test{test_year}_{name}"
            r_test = run_backtest(
                mode=mode_val, lookback=lb_val, z_thresh=zt_val,
                hold_max=1, exit_z=0.0, max_pairs=1,
                start_year=test_year, end_year=test_year,
                config_name=test_name,
            )
            if r_test is not None:
                test_results_window.append({
                    'train_result': r_train,
                    'test_result': r_test,
                    'train_end': train_end,
                    'test_year': test_year,
                    'config_name': name,
                })

        # ---- STEP 2b: Run V52 baseline on test period ----
        v52_test_name = f"Train{train_end}_Test{test_year}_V52_baseline"
        r_v52 = run_backtest(
            mode=V52_CONFIG['mode'],
            lookback=V52_CONFIG['lookback'],
            z_thresh=V52_CONFIG['z_thresh'],
            hold_max=V52_CONFIG['hold_max'],
            exit_z=V52_CONFIG['exit_z'],
            max_pairs=V52_CONFIG['max_pairs'],
            start_year=test_year,
            end_year=test_year,
            config_name=v52_test_name,
        )
        if r_v52 is not None:
            v52_test_results.append({
                'test_result': r_v52,
                'train_end': train_end,
                'test_year': test_year,
            })

        # ---- STEP 3: Report results for this window ----
        best_train = train_results[0]
        print(f"\n  Window: Train 2016-{train_end}, Test {test_year}")

        if test_results_window:
            best_test = test_results_window[0]
            print(f"  Best train config: {best_test['config_name']} "
                  f"(Ann={best_train['ann']:+.1f}%, WR={best_train['wr']:.1f}%)")
            print(f"  Test result (best train applied): "
                  f"Ann={best_test['test_result']['ann']:+.1f}%, "
                  f"WR={best_test['test_result']['wr']:.1f}%, "
                  f"DD={best_test['test_result']['dd']:.1f}%, "
                  f"N={best_test['test_result']['n']}, "
                  f"PF={best_test['test_result']['pf']:.2f}, "
                  f"Sharpe={best_test['test_result']['sharpe']:.2f}")

            # Show all top 5 test results
            print(f"\n  Top 5 train configs applied to test {test_year}:")
            print(f"  {'#':>2s} | {'Config':35s} | {'Train Ann':>9s} | {'Test Ann':>8s} | "
                  f"{'Test WR':>7s} | {'Test DD':>7s} | {'Test N':>6s} | {'Test PF':>7s}")
            print(f"  {'-' * 110}")
            for i, tr in enumerate(test_results_window):
                print(f"  {i+1:2d} | {tr['config_name']:35s} | "
                      f"{tr['train_result']['ann']:+8.1f}% | "
                      f"{tr['test_result']['ann']:+7.1f}% | "
                      f"{tr['test_result']['wr']:6.1f}% | "
                      f"{tr['test_result']['dd']:6.1f}% | "
                      f"{tr['test_result']['n']:5d} | "
                      f"{tr['test_result']['pf']:6.2f}")

            wf_results.extend(test_results_window)
        else:
            print(f"  WARNING: No test results for window {win_idx + 1}")

        # V52 baseline comparison
        if r_v52 is not None:
            print(f"\n  V52 baseline test ({test_year}): "
                  f"Ann={r_v52['ann']:+.1f}%, WR={r_v52['wr']:.1f}%, "
                  f"DD={r_v52['dd']:.1f}%, N={r_v52['n']}")
        else:
            print(f"\n  V52 baseline test ({test_year}): No trades")

    # ================================================================
    # AGGREGATE WALK-FORWARD RESULTS
    # ================================================================
    print(f"\n\n{'=' * 140}")
    print(f"  AGGREGATE WALK-FORWARD RESULTS")
    print(f"{'=' * 140}")

    if not wf_results:
        print("  No walk-forward results to aggregate.")
        return

    # Group results by window
    by_window = {}
    for tr in wf_results:
        ty = tr['test_year']
        if ty not in by_window:
            by_window[ty] = []
        by_window[ty].append(tr)

    # For each window, pick the best test result (from top-1 training config)
    print(f"\n  SUMMARY: Best training config applied to each test year")
    print(f"  {'Window':20s} | {'Best Train Config':35s} | {'Train Ann':>9s} | {'Test Ann':>8s} | "
          f"{'Test WR':>7s} | {'Test DD':>7s} | {'Test N':>6s} | {'V52 Ann':>8s}")
    print(f"  {'-' * 140}")

    best_per_window = []
    v52_anns = {}
    for v52r in v52_test_results:
        v52_anns[v52r['test_year']] = v52r['test_result']['ann']

    for test_year in sorted(by_window.keys()):
        results_for_year = by_window[test_year]
        # Take the best test result (from the #1 training config)
        best = results_for_year[0]
        v52_ann = v52_anns.get(test_year, float('nan'))
        print(f"  Test {test_year}            | {best['config_name']:35s} | "
              f"{best['train_result']['ann']:+8.1f}% | "
              f"{best['test_result']['ann']:+7.1f}% | "
              f"{best['test_result']['wr']:6.1f}% | "
              f"{best['test_result']['dd']:6.1f}% | "
              f"{best['test_result']['n']:5d} | "
              f"{v52_ann:+7.1f}%")
        best_per_window.append(best)

    # ================================================================
    # KEY METRICS
    # ================================================================
    print(f"\n{'=' * 140}")
    print(f"  KEY VALIDATION METRICS")
    print(f"{'=' * 140}")

    # Average test return across all windows
    test_anns = [b['test_result']['ann'] for b in best_per_window]
    avg_test_ann = np.mean(test_anns)
    median_test_ann = np.median(test_anns)

    # Consistency: how many windows were positive
    n_positive = sum(1 for a in test_anns if a > 0)
    n_total = len(test_anns)

    # V52 comparison
    v52_test_anns = [v52_anns[ty] for ty in sorted(v52_anns.keys()) if ty in v52_anns]
    v52_avg_ann = np.mean(v52_test_anns) if v52_test_anns else float('nan')
    v52_n_positive = sum(1 for a in v52_test_anns if a > 0)

    print(f"\n  V55 Adaptive Approach (best train config applied OOS):")
    print(f"    Average test-period annual return: {avg_test_ann:+.1f}%")
    print(f"    Median test-period annual return:  {median_test_ann:+.1f}%")
    print(f"    Consistency: {n_positive}/{n_total} windows positive ({n_positive/n_total*100:.0f}%)")
    print(f"    Test returns by year: {[f'{a:+.1f}%' for a in test_anns]}")

    print(f"\n  V52 Fixed Baseline (LB10_Z1.0_H1_EZ0_MP1_raw):")
    print(f"    Average test-period annual return: {v52_avg_ann:+.1f}%")
    print(f"    Consistency: {v52_n_positive}/{len(v52_test_anns)} windows positive ({v52_n_positive/len(v52_test_anns)*100:.0f}%)" if v52_test_anns else "    No V52 results")
    print(f"    Test returns by year: {[f'{a:+.1f}%' for a in v52_test_anns]}")

    # V55 vs V52 head-to-head
    print(f"\n  V55 vs V52 HEAD-TO-HEAD (per test year):")
    print(f"  {'Year':>6s} | {'V55 Ann':>8s} | {'V52 Ann':>8s} | {'Winner':>6s} | {'Delta':>8s}")
    print(f"  {'-' * 50}")
    v55_wins = 0
    v52_wins = 0
    for b in best_per_window:
        ty = b['test_year']
        v55_ann = b['test_result']['ann']
        v52_ann = v52_anns.get(ty, float('nan'))
        if np.isnan(v52_ann):
            print(f"  {ty:6d} | {v55_ann:+7.1f}% | {'N/A':>8s} | {'V55':>6s} | {'N/A':>8s}")
            v55_wins += 1
        else:
            winner = 'V55' if v55_ann > v52_ann else 'V52'
            delta = v55_ann - v52_ann
            print(f"  {ty:6d} | {v55_ann:+7.1f}% | {v52_ann:+7.1f}% | {winner:>6s} | {delta:+7.1f}%")
            if v55_ann > v52_ann:
                v55_wins += 1
            else:
                v52_wins += 1

    print(f"\n  V55 wins: {v55_wins}, V52 wins: {v52_wins}")

    # ================================================================
    # TOP-5 AVERAGE: Average of top 5 train configs applied OOS
    # ================================================================
    print(f"\n{'=' * 140}")
    print(f"  ROBUSTNESS CHECK: Average of Top 5 Train Configs Applied OOS")
    print(f"{'=' * 140}")

    print(f"\n  {'Year':>6s} | {'Avg Top5 Ann':>11s} | {'Best Ann':>8s} | {'Worst Ann':>9s} | "
          f"{'Avg WR':>6s} | {'Pos/Total':>9s}")
    print(f"  {'-' * 70}")

    all_top5_anns = []
    for test_year in sorted(by_window.keys()):
        results_for_year = by_window[test_year]
        top5_anns = [r['test_result']['ann'] for r in results_for_year[:5]]
        top5_wrs = [r['test_result']['wr'] for r in results_for_year[:5]]
        top5_positive = sum(1 for a in top5_anns if a > 0)
        avg_top5_ann = np.mean(top5_anns)
        all_top5_anns.extend(top5_anns)

        print(f"  {test_year:6d} | {avg_top5_ann:+10.1f}% | {max(top5_anns):+7.1f}% | "
              f"{min(top5_anns):+8.1f}% | {np.mean(top5_wrs):5.1f}% | "
              f"{top5_positive}/{len(top5_anns)}")

    if all_top5_anns:
        print(f"\n  Grand average across all top5 OOS tests: {np.mean(all_top5_anns):+.1f}%")
        print(f"  Grand median: {np.median(all_top5_anns):+.1f}%")
        print(f"  Positive rate: {sum(1 for a in all_top5_anns if a > 0)}/{len(all_top5_anns)} "
              f"({sum(1 for a in all_top5_anns if a > 0)/len(all_top5_anns)*100:.0f}%)")

    # ================================================================
    # OVERFITTING ANALYSIS: Train rank vs Test rank
    # ================================================================
    print(f"\n{'=' * 140}")
    print(f"  OVERFITTING ANALYSIS: Train Performance vs Test Performance")
    print(f"{'=' * 140}")

    train_anns_all = []
    test_anns_all = []
    for tr in wf_results:
        train_anns_all.append(tr['train_result']['ann'])
        test_anns_all.append(tr['test_result']['ann'])

    if train_anns_all and test_anns_all:
        corr = np.corrcoef(train_anns_all, test_anns_all)[0, 1]
        print(f"\n  Train-Test correlation (all configs): {corr:.3f}")
        if corr > 0.5:
            print(f"  -> Strong positive correlation: good sign, training predicts OOS")
        elif corr > 0.2:
            print(f"  -> Moderate correlation: some predictive power in training")
        elif corr > 0:
            print(f"  -> Weak correlation: limited predictive power, possible overfitting")
        else:
            print(f"  -> NEGATIVE correlation: SEVERE overfitting detected!")

        # Decay ratio: average test / average train
        avg_train = np.mean(train_anns_all)
        avg_test = np.mean(test_anns_all)
        decay = avg_test / avg_train if avg_train > 0 else float('nan')
        print(f"\n  Average train Ann: {avg_train:+.1f}%")
        print(f"  Average test Ann:  {avg_test:+.1f}%")
        print(f"  Decay ratio (test/train): {decay:.2f}")
        if decay < 0.3:
            print(f"  -> CRITICAL: Less than 30% of train performance survives OOS")
        elif decay < 0.5:
            print(f"  -> WARNING: Only {decay*100:.0f}% of train performance survives OOS")
        elif decay < 0.7:
            print(f"  -> MODERATE: {decay*100:.0f}% retention, some overfitting")
        else:
            print(f"  -> GOOD: {decay*100:.0f}% retention, genuine signal")

    # ================================================================
    # PER-YEAR DETAIL FOR ALL TOP-1 CONFIGS
    # ================================================================
    print(f"\n{'=' * 140}")
    print(f"  DETAILED PER-YEAR BREAKDOWN (Top-1 train config applied OOS)")
    print(f"{'=' * 140}")

    for b in best_per_window:
        tr = b['test_result']
        print(f"\n  Test {b['test_year']} with config {b['config_name']}:")
        print(f"    Ann={tr['ann']:+.1f}%  WR={tr['wr']:.1f}%  N={tr['n']}  "
              f"DD={tr['dd']:.1f}%  PF={tr['pf']:.2f}  Sharpe={tr['sharpe']:.2f}")
        print(f"    Avg Win={tr['avg_win']:+.2f}%  Avg Loss={tr['avg_loss']:.2f}%  "
              f"Avg Days={tr['avg_days']:.1f}")

        if tr['pair_stats']:
            print(f"    Per-pair breakdown:")
            for p in sorted(tr['pair_stats'].keys(), key=lambda x: -tr['pair_stats'][x]['pnl']):
                ps = tr['pair_stats'][p]
                wr_p = ps['w'] / max(ps['n'], 1) * 100
                print(f"      {p:25s}: {ps['n']:3d} trades  WR={wr_p:5.1f}%  Abs={ps['pnl']:+10.0f}")

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print(f"\n{'=' * 140}")
    print(f"  FINAL VERDICT")
    print(f"{'=' * 140}")

    print(f"\n  V55 Adaptive Walk-Forward Summary:")
    print(f"    {n_total} test windows evaluated")
    print(f"    Average OOS annual return: {avg_test_ann:+.1f}%")
    print(f"    Median OOS annual return:  {median_test_ann:+.1f}%")
    print(f"    Positive windows: {n_positive}/{n_total} ({n_positive/n_total*100:.0f}%)")
    print(f"    Best OOS year: {max(test_anns):+.1f}%")
    print(f"    Worst OOS year: {min(test_anns):+.1f}%")

    print(f"\n  V52 Fixed Baseline Summary:")
    print(f"    Average OOS annual return: {v52_avg_ann:+.1f}%")
    print(f"    Positive windows: {v52_n_positive}/{len(v52_test_anns)} "
          f"({v52_n_positive/len(v52_test_anns)*100:.0f}%)" if v52_test_anns else "")

    if v52_test_anns:
        print(f"\n  V55 vs V52:")
        print(f"    V55 average: {avg_test_ann:+.1f}%")
        print(f"    V52 average: {v52_avg_ann:+.1f}%")
        delta = avg_test_ann - v52_avg_ann
        if delta > 50:
            print(f"    V55 WINS by {delta:+.1f}% -- adaptive approach adds significant value")
        elif delta > 10:
            print(f"    V55 edges out V52 by {delta:+.1f}% -- marginal improvement")
        elif delta > -10:
            print(f"    Roughly TIED (delta {delta:+.1f}%) -- no clear winner")
        else:
            print(f"    V52 WINS by {-delta:+.1f}% -- adaptive approach hurts OOS!")

    # Check for WF2022 anomaly
    wf2022_result = None
    for b in best_per_window:
        if b['test_year'] == 2022:
            wf2022_result = b
            break

    if wf2022_result:
        wf2022_ann = wf2022_result['test_result']['ann']
        print(f"\n  WF2022 Specific Check:")
        print(f"    V55 original claim: WF2022 = +859.6%")
        print(f"    This rigorous test: WF2022 = {wf2022_ann:+.1f}%")
        if wf2022_ann > 500:
            print(f"    VERDICT: WF2022 is genuinely exceptional (though possibly lucky)")
        elif wf2022_ann > 100:
            print(f"    VERDICT: WF2022 is strong but the +859.6% claim was inflated")
        elif wf2022_ann > 0:
            print(f"    VERDICT: WF2022 is positive but far below the +859.6% claim")
        else:
            print(f"    VERDICT: WF2022 is NEGATIVE -- the +859.6% claim was overfitting")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 140)


if __name__ == '__main__':
    main()
