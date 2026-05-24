"""
Alpha Futures V51 — Push V39 Pair Trading to Its Absolute Limits
=================================================================
V39 supply chain pair trading champion: +188.1% full period, +376.7% WF2024.
Config: LB10_Z1.5_H3_MP2 (lookback=10, z_threshold=1.5, hold=3 days, max 2 pairs).

V51 tests every dimension to find the ceiling:
  1. Capital utilization: dynamic sizing (1 pair = 100%, 2 pairs = 50%)
     + z-score magnitude weighting
  2. Pair-specific parameters (per-pair lookback/z_threshold)
  3. Aggressive Z thresholds: 0.8, 1.0, 1.2, 1.5
  4. Asymmetric entry/exit: enter at Z, exit when Z crosses threshold
  5. Max pairs sweep: 1..5, 99
  6. Exit optimization: z-cross [0, 0.2, 0.3, 0.5], time [2,3,5,7], z-only exit
  7. Cooldown after exit: [0, 2, 3, 5] days

~300 configs, walk-forward top 20 (2022, 2023, 2024).
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

MULT = {'agfi': 15, 'alfi': 5, 'aufi': 1000, 'bufi': 10, 'cufi': 5, 'fufi': 10, 'rbfi': 10, 'znfi': 5, 'nifi': 1, 'hcfi': 10, 'spfi': 10, 'ssfi': 5, 'sffi': 5, 'smfi': 5, 'pbfi': 5, 'snfi': 1, 'rufi': 10, 'wrffi': 10, 'afi': 10, 'bfi': 10, 'bbfi': 500, 'cffi': 5, 'cfi': 10, 'csfi': 10, 'ebfi': 5, 'egfi': 10, 'fbfi': 500, 'ifi': 100, 'jfi': 100, 'jmfi': 60, 'lfi': 5, 'mfi': 10, 'pgfi': 20, 'ppfi': 5, 'vfi': 5, 'yfi': 10, 'pfi': 10, 'jdfi': 5, 'lhfi': 16, 'pkfi': 5, 'rrfi': 20, 'lrfi': 20, 'jrfi': 20, 'pmfi': 20, 'whfi': 20, 'rsfi': 20, 'cjfi': 10, 'mafi': 10, 'apfi': 10, 'cyfi': 5, 'fgfi': 20, 'oifi': 10, 'pfifi': 5, 'rmfi': 10, 'srfi': 10, 'tafi': 5, 'safi': 20, 'urfi': 20, 'scfi': 1000, 'lufi': 10, 'bcfi': 5, 'nrfi': 1, 'lgfi': 20, 'brfi': 5, 'lcfi': 1, 'sifi': 5, 'ni': 1, 'tai': 5}
DEF_MULT = 10
COMM = 0.0003

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

# Per-pair best parameters from V41 per-pair data
PAIR_PARAMS = {
    ('jfi', 'jmfi'):  {'lb': 10, 'z': 1.5},
    ('bfi', 'scfi'):  {'lb': 10, 'z': 2.0},
    ('fufi', 'scfi'): {'lb': 10, 'z': 1.5},
    ('mafi', 'scfi'): {'lb': 10, 'z': 2.0},
    ('hcfi', 'rbfi'): {'lb': 10, 'z': 1.5},
    ('rbfi', 'ifi'):  {'lb': 10, 'z': 1.5},
    ('hcfi', 'ifi'):  {'lb': 10, 'z': 2.0},
    ('yfi', 'afi'):   {'lb': 10, 'z': 1.5},
    ('mfi', 'afi'):   {'lb': 10, 'z': 1.5},
    ('pfi', 'yfi'):   {'lb': 10, 'z': 1.5},
    ('ppfi', 'mafi'): {'lb': 15, 'z': 2.0},
    ('vfi', 'mafi'):  {'lb': 15, 'z': 2.0},
    ('egfi', 'mafi'): {'lb': 15, 'z': 2.0},
}


def main():
    t_start = time.time()
    print("=" * 130)
    print("Alpha Futures V51 — Push V39 Pair Trading to Absolute Limits")
    print("Testing: capital utilization, per-pair params, asymmetric exit, cooldown, dynamic sizing")
    print("=" * 130)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()
    sym_to_si = {syms[si]: si for si in range(NS)}

    # Build pair index mapping
    pair_indices = []
    for down_sym, up_sym in PAIRS:
        down_si = sym_to_si.get(down_sym, -1)
        up_si = sym_to_si.get(up_sym, -1)
        if down_si >= 0 and up_si >= 0:
            pair_indices.append((down_si, up_si, down_sym, up_sym))

    print(f"  {NS} commodities, {ND} days, {len(pair_indices)} active pairs")

    # ========================================
    # PRECOMPUTE SPREADS AND Z-SCORES
    # ========================================
    print("\n[Signals] Computing spreads and z-scores...", flush=True)
    t0 = time.time()

    # Precompute spreads
    spreads = {}
    for down_si, up_si, down_sym, up_sym in pair_indices:
        spread = np.full(ND, np.nan)
        for di in range(ND):
            pd = C[down_si, di]
            pu = C[up_si, di]
            if not np.isnan(pd) and not np.isnan(pu):
                spread[di] = pd - pu
        spreads[(down_si, up_si)] = spread

    # Precompute z-scores for multiple lookbacks
    LOOKBACKS = [5, 8, 10, 12, 15, 20]
    z_cache = {}  # (down_si, up_si, lookback) -> z array

    for down_si, up_si, down_sym, up_sym in pair_indices:
        sp = spreads[(down_si, up_si)]
        for lb in LOOKBACKS:
            z = np.full(ND, np.nan)
            for di in range(lb, ND):
                window = sp[di - lb:di]
                valid = window[~np.isnan(window)]
                if len(valid) >= lb * 0.8:
                    mean_v = np.mean(valid)
                    std_v = np.std(valid, ddof=1)
                    if std_v > 1e-10:
                        z[di] = (sp[di] - mean_v) / std_v
            z_cache[(down_si, up_si, lb)] = z

    print(f"  Spreads + z-scores computed ({time.time()-t0:.1f}s)", flush=True)

    # ========================================
    # BACKTEST ENGINE (enhanced)
    # ========================================
    def run_backtest(
        lookback, z_entry, hold_max, max_pairs,
        z_exit=0.0,
        dynamic_cap=False,
        z_weight=False,
        per_pair_params=False,
        cooldown=0,
        z_only_exit=False,
        wf_split_year=None,
        config_name="",
    ):
        """
        Enhanced pair trading backtest.

        New features vs V39:
          - dynamic_cap: 1 pair = 100% cash, 2 pairs = 50% each (not fixed 50%)
          - z_weight: size positions by z-score magnitude
          - per_pair_params: use PAIR_PARAMS for per-pair lookback/z
          - cooldown: min days between exit and re-entry for same pair
          - z_exit: exit when z crosses this threshold (0, 0.2, 0.3, 0.5)
          - z_only_exit: no time exit, only exit on z crossing z_exit
        """
        cash = float(CASH0)
        trades = []
        pair_positions = []
        pair_cooldown = {}  # (down_si, up_si) -> earliest re-entry di

        # Build per-pair data
        pair_data = {}
        for down_si, up_si, down_sym, up_sym in pair_indices:
            if per_pair_params:
                pp = PAIR_PARAMS.get((down_sym, up_sym), {'lb': lookback, 'z': z_entry})
                lb = pp['lb']
                z_ent = pp['z']
            else:
                lb = lookback
                z_ent = z_entry

            pair_data[(down_si, up_si)] = {
                'spread': spreads[(down_si, up_si)],
                'z': z_cache[(down_si, up_si, lb)],
                'z_entry': z_ent,
                'down_sym': down_sym,
                'up_sym': up_sym,
            }

        for di in range(MIN_TRAIN, ND):
            year = dates[di].year
            if wf_split_year is not None and year < wf_split_year:
                continue

            # --- Manage existing pair positions ---
            new_positions = []
            for pos in pair_positions:
                p_key = (pos['down_si'], pos['up_si'])
                z_now = pair_data[p_key]['z'][di]
                days_held = di - pos['entry_di']
                entry_z = pos['entry_z']
                pos_dir = pos['dir']

                exit_reason = None

                # Exit 1: Z crosses exit threshold (mean reversion complete)
                if not np.isnan(z_now):
                    if pos_dir == 1 and z_now >= z_exit:
                        exit_reason = 'z_cross'
                    elif pos_dir == -1 and z_now <= -z_exit:
                        exit_reason = 'z_cross'

                # Exit 2: Stop loss (z moves further from entry by 1.5)
                if exit_reason is None and not np.isnan(z_now):
                    if pos_dir == 1 and z_now < entry_z - 1.5:
                        exit_reason = 'stop_loss'
                    elif pos_dir == -1 and z_now > entry_z + 1.5:
                        exit_reason = 'stop_loss'

                # Exit 3: Time exit (unless z_only_exit)
                if exit_reason is None and not z_only_exit and days_held >= hold_max:
                    exit_reason = 'time'

                if exit_reason:
                    c_down = C[pos['down_si'], di]
                    c_up = C[pos['up_si'], di]
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

                    # Set cooldown
                    if cooldown > 0:
                        pair_cooldown[p_key] = di + cooldown
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

            # Determine cash allocation
            n_total_will_have = len(pair_positions) + n_can_open

            candidates = []
            for down_si, up_si, down_sym, up_sym in pair_indices:
                if down_si in occupied or up_si in occupied:
                    continue
                # Cooldown check
                p_key = (down_si, up_si)
                if p_key in pair_cooldown and di < pair_cooldown[p_key]:
                    continue

                pd_data = pair_data[p_key]
                z_val = pd_data['z'][di]
                if np.isnan(z_val):
                    continue
                if abs(z_val) < pd_data['z_entry']:
                    continue
                candidates.append((abs(z_val), down_si, up_si, down_sym, up_sym, z_val))

            if not candidates:
                continue

            # Sort by strongest deviation
            candidates.sort(key=lambda x: -x[0])

            for _, down_si, up_si, down_sym, up_sym, z_val in candidates[:n_can_open]:
                c_down = C[down_si, di]
                c_up = C[up_si, di]
                if np.isnan(c_down) or c_down <= 0 or np.isnan(c_up) or c_up <= 0:
                    continue

                mult_down = MULT.get(down_sym, DEF_MULT)
                mult_up = MULT.get(up_sym, DEF_MULT)

                # Dynamic capital allocation
                if dynamic_cap:
                    # All available pairs share cash equally
                    cash_available = cash
                    # How many total positions after this one opens
                    n_after = len(pair_positions) + 1
                    # But we need to reserve for remaining open slots
                    n_remaining_slots = max_pairs - n_after
                    # Cash for this pair: if more slots may open, use 1/max_pairs
                    # Otherwise use all remaining cash
                    if n_remaining_slots > 0 and len(candidates) > 1:
                        cash_for_pair = cash_available / max_pairs
                    else:
                        # Last slot or only candidate -> use all remaining / slots
                        cash_for_pair = cash_available / max(1, n_after)
                else:
                    cash_for_pair = cash / max_pairs

                # Z-score magnitude weighting
                if z_weight:
                    # Scale by |z| / z_entry (stronger signal = more capital)
                    z_scale = min(abs(z_val) / pair_data[(down_si, up_si)]['z_entry'], 2.0)
                    # Clamp to [0.5, 2.0] range
                    z_scale = max(z_scale, 0.5)
                    cash_for_pair *= z_scale
                    # Don't exceed available cash
                    cash_for_pair = min(cash_for_pair, cash * 0.95)

                cash_per_leg = cash_for_pair / 2
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
        for pos in pair_positions:
            p_down_si = pos['down_si']
            p_up_si = pos['up_si']
            c_down = C[p_down_si, ND - 1]
            c_up = C[p_up_si, ND - 1]
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
                'days': ND - 1 - pos['entry_di'],
                'di': ND - 1,
                'year': dates[ND - 1].year,
                'pair': (pos['down_sym'], pos['up_sym']),
                'pair_label': PAIR_LABEL.get((pos['down_sym'], pos['up_sym']), ''),
                'dir': pos['dir'],
                'reason': 'end',
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

        days_total = (dates[ND - 1] - dates[MIN_TRAIN]).days
        yr = max(days_total / 365.25, 0.01)
        if wf_split_year:
            first_test_di = None
            for di in range(MIN_TRAIN, ND):
                if dates[di].year >= wf_split_year:
                    first_test_di = di
                    break
            if first_test_di:
                days_total = (dates[ND - 1] - dates[first_test_di]).days
                yr = max(days_total / 365.25, 0.01)

        ann = ((cash / CASH0) ** (1 / yr) - 1) * 100

        trade_pnls = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
        if len(trade_pnls) > 1:
            rets = np.array(trade_pnls) / float(CASH0)
            mean_ret = np.mean(rets)
            std_ret = np.std(rets)
            sharpe_approx = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
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
            'trades': trades,
        }

    # ========================================
    # CONFIG GENERATION
    # ========================================
    print("\n[Backtest] Generating configs...", flush=True)

    results = []
    configs = []

    # --- Baseline: replicate V39 champion ---
    configs.append({
        'lookback': 10, 'z_entry': 1.5, 'hold_max': 3, 'max_pairs': 2,
        'z_exit': 0.0, 'dynamic_cap': False, 'z_weight': False,
        'per_pair_params': False, 'cooldown': 0, 'z_only_exit': False,
        'wf_split_year': None, 'config_name': 'V39_BASE_LB10_Z1.5_H3_MP2',
    })

    # --- TEST 1: Capital utilization optimization ---
    # Dynamic cap + z-weight on V39 champion params
    for dyn in [True, False]:
        for zw in [True, False]:
            if not dyn and not zw:
                continue  # already have baseline
            configs.append({
                'lookback': 10, 'z_entry': 1.5, 'hold_max': 3, 'max_pairs': 2,
                'z_exit': 0.0, 'dynamic_cap': dyn, 'z_weight': zw,
                'per_pair_params': False, 'cooldown': 0, 'z_only_exit': False,
                'wf_split_year': None,
                'config_name': f'CAP_DYN{dyn}_ZW{zw}_LB10_Z1.5_H3_MP2',
            })

    # Dynamic cap across different max_pairs
    for mp in [1, 2, 3, 4, 5]:
        configs.append({
            'lookback': 10, 'z_entry': 1.5, 'hold_max': 3, 'max_pairs': mp,
            'z_exit': 0.0, 'dynamic_cap': True, 'z_weight': True,
            'per_pair_params': False, 'cooldown': 0, 'z_only_exit': False,
            'wf_split_year': None,
            'config_name': f'CAP_DYN_ZW_MP{mp}_LB10_Z1.5_H3',
        })

    # --- TEST 2: Per-pair parameters ---
    for mp in [1, 2, 3]:
        for dyn in [False, True]:
            configs.append({
                'lookback': 10, 'z_entry': 1.5, 'hold_max': 3, 'max_pairs': mp,
                'z_exit': 0.0, 'dynamic_cap': dyn, 'z_weight': False,
                'per_pair_params': True, 'cooldown': 0, 'z_only_exit': False,
                'wf_split_year': None,
                'config_name': f'PPP_MP{mp}_DYN{dyn}_H3',
            })
        # Per-pair + dynamic cap + z-weight
        configs.append({
            'lookback': 10, 'z_entry': 1.5, 'hold_max': 3, 'max_pairs': mp,
            'z_exit': 0.0, 'dynamic_cap': True, 'z_weight': True,
            'per_pair_params': True, 'cooldown': 0, 'z_only_exit': False,
            'wf_split_year': None,
            'config_name': f'PPP_MP{mp}_FULL_H3',
        })

    # --- TEST 3: Aggressive Z thresholds ---
    for z_ent in [0.8, 1.0, 1.2, 1.5]:
        for mp in [1, 2, 3]:
            configs.append({
                'lookback': 10, 'z_entry': z_ent, 'hold_max': 3, 'max_pairs': mp,
                'z_exit': 0.0, 'dynamic_cap': False, 'z_weight': False,
                'per_pair_params': False, 'cooldown': 0, 'z_only_exit': False,
                'wf_split_year': None,
                'config_name': f'Z_AGG_Z{z_ent}_MP{mp}_LB10_H3',
            })
            # With dynamic cap
            configs.append({
                'lookback': 10, 'z_entry': z_ent, 'hold_max': 3, 'max_pairs': mp,
                'z_exit': 0.0, 'dynamic_cap': True, 'z_weight': True,
                'per_pair_params': False, 'cooldown': 0, 'z_only_exit': False,
                'wf_split_year': None,
                'config_name': f'Z_AGG_Z{z_ent}_MP{mp}_DYN_ZW',
            })

    # --- TEST 4: Max pairs sweep with best parameters ---
    for mp in [1, 2, 3, 4, 5, 99]:
        for lb in [10]:
            for z_ent in [1.0, 1.2, 1.5]:
                configs.append({
                    'lookback': lb, 'z_entry': z_ent, 'hold_max': 3, 'max_pairs': mp,
                    'z_exit': 0.0, 'dynamic_cap': True, 'z_weight': True,
                    'per_pair_params': False, 'cooldown': 0, 'z_only_exit': False,
                    'wf_split_year': None,
                    'config_name': f'MPSWP_MP{mp}_Z{z_ent}_LB{lb}_DYN',
                })

    # --- TEST 5: Exit optimization ---
    # Z-exit threshold sweep
    for z_exit in [0.0, 0.2, 0.3, 0.5]:
        for hold in [2, 3, 5, 7]:
            for dyn in [False, True]:
                configs.append({
                    'lookback': 10, 'z_entry': 1.5, 'hold_max': hold, 'max_pairs': 2,
                    'z_exit': z_exit, 'dynamic_cap': dyn, 'z_weight': dyn,
                    'per_pair_params': False, 'cooldown': 0, 'z_only_exit': False,
                    'wf_split_year': None,
                    'config_name': f'EXIT_ZE{z_exit}_H{hold}_DYN{dyn}',
                })

    # Z-only exit (no time limit, let mean reversion complete naturally)
    for z_exit in [0.0, 0.2, 0.3]:
        for z_ent in [1.0, 1.2, 1.5]:
            configs.append({
                'lookback': 10, 'z_entry': z_ent, 'hold_max': 999, 'max_pairs': 2,
                'z_exit': z_exit, 'dynamic_cap': True, 'z_weight': True,
                'per_pair_params': False, 'cooldown': 0, 'z_only_exit': True,
                'wf_split_year': None,
                'config_name': f'ZONLY_ZE{z_exit}_ZE{z_ent}_DYN',
            })

    # Asymmetric: enter at Z=1.2, exit at z_cross=0.3
    for z_ent in [1.0, 1.2]:
        for z_exit in [0.2, 0.3]:
            for hold in [3, 5, 7]:
                configs.append({
                    'lookback': 10, 'z_entry': z_ent, 'hold_max': hold, 'max_pairs': 2,
                    'z_exit': z_exit, 'dynamic_cap': True, 'z_weight': True,
                    'per_pair_params': False, 'cooldown': 0, 'z_only_exit': False,
                    'wf_split_year': None,
                    'config_name': f'ASYM_E{z_ent}X{z_exit}_H{hold}',
                })

    # --- TEST 6: Re-entry cooldown ---
    for cd in [0, 2, 3, 5]:
        for z_ent in [1.0, 1.2, 1.5]:
            configs.append({
                'lookback': 10, 'z_entry': z_ent, 'hold_max': 3, 'max_pairs': 2,
                'z_exit': 0.0, 'dynamic_cap': True, 'z_weight': True,
                'per_pair_params': False, 'cooldown': cd, 'z_only_exit': False,
                'wf_split_year': None,
                'config_name': f'CD{cd}_Z{z_ent}_DYN',
            })

    # --- Combined best guesses ---
    # Per-pair params + dynamic cap + z-weight + asymmetric exit + cooldown
    for z_exit in [0.0, 0.2, 0.3]:
        for cd in [0, 2, 3]:
            configs.append({
                'lookback': 10, 'z_entry': 1.5, 'hold_max': 3, 'max_pairs': 2,
                'z_exit': z_exit, 'dynamic_cap': True, 'z_weight': True,
                'per_pair_params': True, 'cooldown': cd, 'z_only_exit': False,
                'wf_split_year': None,
                'config_name': f'COMBO_ZE{z_exit}_CD{cd}_PPP',
            })

    # Per-pair + aggressive Z + dynamic
    for mp in [2, 3]:
        configs.append({
            'lookback': 10, 'z_entry': 1.2, 'hold_max': 3, 'max_pairs': mp,
            'z_exit': 0.0, 'dynamic_cap': True, 'z_weight': True,
            'per_pair_params': True, 'cooldown': 0, 'z_only_exit': False,
            'wf_split_year': None,
            'config_name': f'COMBO_AGG_MP{mp}_PPP',
        })

    # Best V39 params with different lookbacks
    for lb in [5, 8, 10, 12, 15, 20]:
        configs.append({
            'lookback': lb, 'z_entry': 1.5, 'hold_max': 3, 'max_pairs': 2,
            'z_exit': 0.0, 'dynamic_cap': True, 'z_weight': True,
            'per_pair_params': False, 'cooldown': 0, 'z_only_exit': False,
            'wf_split_year': None,
            'config_name': f'LB_LB{lb}_Z1.5_DYN',
        })

    print(f"  {len(configs)} full-period configurations", flush=True)

    # --- Run all full-period configs ---
    print("\n[Backtest] Running full-period configs...", flush=True)
    for ci, cfg in enumerate(configs):
        r = run_backtest(**cfg)
        if r is not None:
            results.append(r)
            if r['ann'] > 50:
                print(f"  {r['name']:45s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                      f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                      f"Sh {r['sharpe']:5.2f} | AvgD {r['avg_days']:.1f}")
        if (ci + 1) % 50 == 0:
            print(f"  [{ci+1}/{len(configs)}] {len(results)} configs with results", flush=True)

    # --- Walk-forward for top 20 ---
    results.sort(key=lambda x: -x['ann'])
    full_results = [r for r in results]
    top20_for_wf = full_results[:20]

    print(f"\n[Walk-Forward] Running WF for top 20 configs...", flush=True)
    wf_results = []
    for r in top20_for_wf:
        # Find the config that produced this result
        cfg = None
        for c in configs:
            if c['config_name'] == r['name']:
                cfg = c.copy()
                break
        if cfg is None:
            continue

        for wf_year in [2022, 2023, 2024]:
            cfg['wf_split_year'] = wf_year
            cfg['config_name'] = f"{r['name']}_WF{wf_year}"
            wr = run_backtest(**cfg)
            if wr is not None:
                wf_results.append(wr)
                print(f"  {wr['name']:55s} | Ann {wr['ann']:+7.1f}% | WR {wr['wr']:5.1f}% | "
                      f"N {wr['n']:4d} | DD {wr['dd']:6.1f}% | PF {wr['pf']:4.2f}")

    # ========================================
    # RESULTS
    # ========================================
    results.sort(key=lambda x: -x['ann'])

    print(f"\n{'=' * 130}")
    print(f"  TOP 20 FULL-PERIOD RESULTS (sorted by annual return)")
    print(f"{'=' * 130}")
    hdr = (f"  {'Config':45s} | {'Ann':>7s} | {'WR':>5s} | {'N':>4s} | {'DD':>6s} | "
           f"{'PF':>4s} | {'Sharpe':>6s} | {'AvgW':>7s} | {'AvgL':>6s} | {'AvgD':>4s}")
    print(hdr)
    print(f"  {'-' * 125}")
    for r in results[:20]:
        print(f"  {r['name']:45s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:4.2f} | "
              f"{r['sharpe']:6.2f} | {r['avg_win']:+6.2f}% | {r['avg_loss']:5.2f}% | "
              f"{r['avg_days']:4.1f}")

    # Walk-forward results
    if wf_results:
        wf_results.sort(key=lambda x: -x['ann'])
        print(f"\n{'=' * 130}")
        print(f"  TOP 10 WALK-FORWARD RESULTS (out-of-sample)")
        print(f"{'=' * 130}")
        for r in wf_results[:10]:
            print(f"  {r['name']:55s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                  f"Sh {r['sharpe']:6.2f}")

    # Best result detailed breakdown
    if results:
        best = results[0]
        print(f"\n{'=' * 130}")
        print(f"  BEST CONFIG DETAIL: {best['name']}")
        print(f"  Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  N={best['n']}  "
              f"DD={best['dd']:.1f}%  PF={best['pf']:.2f}  Sharpe={best['sharpe']:.2f}  "
              f"Final={best['cash']:.0f}")
        print(f"{'=' * 130}")

        print(f"\n  PER-PAIR BREAKDOWN:")
        for p in sorted(best['pair_stats'].keys(), key=lambda x: -best['pair_stats'][x]['n']):
            ps = best['pair_stats'][p]
            wr_p = ps['w'] / max(ps['n'], 1) * 100
            print(f"    {p:25s}: {ps['n']:3d} trades  WR={wr_p:5.1f}%  Abs PnL={ps['pnl']:+12.0f}")

        print(f"\n  YEARLY BREAKDOWN:")
        for y in sorted(best['yearly'].keys()):
            s = best['yearly'][y]
            wr_y = s['w'] / max(s['n'], 1) * 100
            print(f"    {y}: {s['n']:3d} trades  WR={wr_y:5.1f}%  PnL={s['pnl']:+.1f}%  "
                  f"Abs={s['pnl_abs_sum']:+.0f}")

        print(f"\n  EXIT REASON BREAKDOWN:")
        for reason, s in sorted(best['reasons'].items(), key=lambda x: -x[1]['n']):
            rwr = s['w'] / max(s['n'], 1) * 100
            print(f"    {reason:12s}: {s['n']:4d} trades  WR={rwr:5.1f}%  "
                  f"PnL={s['pnl_pct_sum']:+.1f}%  Abs={s['pnl']:+.0f}")

    # Yearly for top 5 full-period
    if len(results) >= 2:
        print(f"\n  YEARLY BREAKDOWN FOR TOP 5 CONFIGS:")
        for idx, r in enumerate(results[:5]):
            print(f"\n  #{idx+1}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.1f}%, "
                  f"DD={r['dd']:.1f}%, Sharpe={r['sharpe']:.2f})")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:3d}t  WR={wr_y:5.1f}%  PnL={ys['pnl']:+.1f}%")

    # Per-pair summary across top 20
    if results:
        print(f"\n  PAIR PROFITABILITY ACROSS TOP 20 CONFIGS:")
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
            print(f"    {p:25s}: {ps['n']:4d} trades  WR={wr_p:5.1f}%  Total Abs={ps['pnl']:+12.0f}")

    # Category winners
    print(f"\n  CATEGORY WINNERS:")
    categories = {
        'Capital Utilization': 'CAP_',
        'Per-Pair Params': 'PPP_',
        'Aggressive Z': 'Z_AGG_',
        'Max Pairs Sweep': 'MPSWP_',
        'Exit Optimization': 'EXIT_',
        'Z-Only Exit': 'ZONLY_',
        'Asymmetric Exit': 'ASYM_',
        'Cooldown': 'CD',
        'Combined': 'COMBO_',
        'Lookback': 'LB_',
        'V39 Baseline': 'V39_',
    }
    for cat_name, prefix in categories.items():
        cat = [r for r in results if r['name'].startswith(prefix)]
        if cat:
            best_cat = max(cat, key=lambda x: x['ann'])
            print(f"    {cat_name:25s}: {best_cat['name']:45s}  Ann={best_cat['ann']:+.1f}%  "
                  f"WR={best_cat['wr']:.1f}%  DD={best_cat['dd']:.1f}%  Sh={best_cat['sharpe']:.2f}")

    # Comparison with V39 champion
    v39_base = [r for r in results if r['name'] == 'V39_BASE_LB10_Z1.5_H3_MP2']
    if v39_base and results:
        v39_ann = v39_base[0]['ann']
        best_ann = results[0]['ann']
        print(f"\n  === V51 BEST vs V39 CHAMPION ===")
        print(f"  V39 Baseline: {v39_base[0]['name']} = {v39_ann:+.1f}% DD={v39_base[0]['dd']:.1f}%")
        print(f"  V51 Best:     {results[0]['name']} = {best_ann:+.1f}% DD={results[0]['dd']:.1f}%")
        print(f"  Delta:        {best_ann - v39_ann:+.1f}%")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
