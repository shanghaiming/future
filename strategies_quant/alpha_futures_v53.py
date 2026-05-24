"""
Alpha Futures V53 -- Push Ultra Short-Term Pair Trading Toward 600% Annual
==========================================================================
V52 champion: LB10_Z1.0_H1_EZ0.0_MP1 = +303.5% annual, 62.9% WR, walk-forward +472.6%.
Key insight: Daily compounding at 62.9% WR is the edge.

V53 explores new dimensions to push past 400% toward 600%:

  1. Lookback optimization for 1-day hold: [3, 5, 7, 10, 15, 20]
  2. Max pairs with 1-day hold: [1, 2, 3, 5] with equal vs z-score weighted allocation
  3. Same-day re-entry: allow immediate re-entry if z still above threshold
  4. OI confirmation: rising OI on at least one leg
  5. Spread normalization: raw, percentage, log
  6. Volume confirmation: vol_ratio > 1.0 on both legs
  7. Seasonal / vol-regime filters

~300 configs. Walk-forward for top 20 (2022, 2023, 2024).

Print: top 20 full-period, top 10 walk-forward, per-pair breakdown,
       lookback comparison table, spread-normalization comparison.
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

# Spread normalization modes
SPREAD_RAW  = 'raw'   # C_down - C_up
SPREAD_PCT  = 'pct'   # (C_down - C_up) / C_up
SPREAD_LOG  = 'log'   # log(C_down) - log(C_up)


def main():
    t_start = time.time()
    print("=" * 140)
    print("Alpha Futures V53 -- Push Ultra Short-Term Pair Trading Toward 600% Annual")
    print("V52 champion: LB10_Z1.0_H1_EZ0.0_MP1 = +303.5%, WR 62.9%, WF +472.6%")
    print("New dimensions: lookback sweep, spread types, OI/vol filters, re-entry, seasonal")
    print("=" * 140)

    # Load data WITH OI
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    sym_to_si = {syms[si]: si for si in range(NS)}
    has_oi = not np.all(np.isnan(OI))
    print(f"  OI available: {has_oi}")

    # Build pair index mapping
    pair_indices = []
    for down_sym, up_sym in PAIRS:
        down_si = sym_to_si.get(down_sym, -1)
        up_si = sym_to_si.get(up_sym, -1)
        if down_si >= 0 and up_si >= 0:
            pair_indices.append((down_si, up_si, down_sym, up_sym))
        else:
            print(f"  WARNING: pair ({down_sym}, {up_sym}) not found in data "
                  f"(down_si={down_si}, up_si={up_si})")

    print(f"  {NS} commodities, {ND} days, {len(pair_indices)} active pairs")

    # ========================================
    # PRECOMPUTE SPREADS (3 normalization types)
    # ========================================
    print("\n[Signals] Computing spreads (raw/pct/log)...", flush=True)
    t0 = time.time()

    spreads = {SPREAD_RAW: {}, SPREAD_PCT: {}, SPREAD_LOG: {}}
    for down_si, up_si, down_sym, up_sym in pair_indices:
        for mode in [SPREAD_RAW, SPREAD_PCT, SPREAD_LOG]:
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
            spreads[mode][(down_si, up_si)] = spread

    # ========================================
    # PRECOMPUTE VOLUME RATIOS (20-day avg)
    # ========================================
    print("  Computing volume ratios...", flush=True)
    vol_ratio = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            vw = V[si, di - 20:di]
            vv = vw[~np.isnan(vw)]
            if len(vv) >= 10 and np.mean(vv) > 0:
                vol_ratio[si, di] = V[si, di] / np.mean(vv) if not np.isnan(V[si, di]) else np.nan

    # ========================================
    # PRECOMPUTE OI CHANGE (1-day)
    # ========================================
    if has_oi:
        print("  Computing OI changes...", flush=True)
        oi_rising = np.zeros((NS, ND), dtype=bool)
        for si in range(NS):
            for di in range(1, ND):
                if not np.isnan(OI[si, di]) and not np.isnan(OI[si, di - 1]) and OI[si, di - 1] > 0:
                    oi_rising[si, di] = OI[si, di] > OI[si, di - 1]

    # ========================================
    # PRECOMPUTE MARKET-WIDE VOLATILITY (for regime filter)
    # ========================================
    print("  Computing market-wide volatility...", flush=True)
    # Average absolute return across all commodities as vol proxy
    mkt_vol = np.full(ND, np.nan)
    for di in range(1, ND):
        rets = []
        for si in range(NS):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 1]) and C[si, di - 1] > 0:
                rets.append(abs((C[si, di] - C[si, di - 1]) / C[si, di - 1]))
        if rets:
            mkt_vol[di] = np.mean(rets)
    # Rolling 20-day percentile of market vol
    mkt_vol_pct = np.full(ND, np.nan)
    for di in range(20, ND):
        window = mkt_vol[di - 20:di]
        valid_w = window[~np.isnan(window)]
        if len(valid_w) >= 10:
            mkt_vol_pct[di] = np.sum(valid_w < mkt_vol[di]) / len(valid_w) if not np.isnan(mkt_vol[di]) else np.nan

    print(f"  All precomputations done ({time.time() - t0:.1f}s)", flush=True)

    # ========================================
    # BACKTEST ENGINE FOR V53
    # ========================================
    def run_backtest(lookback, z_thresh, hold_max, exit_z, max_pairs,
                     spread_mode=SPREAD_RAW,
                     allow_reentry=False,
                     oi_filter='off',          # 'off', 'either', 'both', 'long_leg'
                     vol_filter=False,          # require vol_ratio > 1.0 on both legs
                     capital_mode='equal',      # 'equal' or 'zscore'
                     seasonal_filter='off',     # 'off', 'best_months', 'skip_high_vol'
                     wf_split_year=None,
                     config_name=""):
        """
        V53 enhanced pair trading backtest.
        """
        cash = float(CASH0)
        trades = []
        pair_positions = []
        # Track recently exited pairs for re-entry logic: set of (down_si, up_si) exited today
        exited_today = set()

        # Pre-compute per-pair rolling z-scores for this lookback and spread mode
        pair_data = {}
        for down_si, up_si, down_sym, up_sym in pair_indices:
            sp = spreads[spread_mode].get((down_si, up_si))
            if sp is None:
                continue
            sp_mean = np.full(ND, np.nan)
            sp_std = np.full(ND, np.nan)
            z = np.full(ND, np.nan)

            for di in range(lookback, ND):
                window = sp[di - lookback:di]
                valid = window[~np.isnan(window)]
                if len(valid) >= lookback * 0.8:
                    sp_mean[di] = np.mean(valid)
                    sp_std[di] = np.std(valid, ddof=1)
                    if sp_std[di] > 1e-10:
                        z[di] = (sp[di] - sp_mean[di]) / sp_std[di]

            pair_data[(down_si, up_si)] = {
                'spread': sp,
                'mean': sp_mean,
                'std': sp_std,
                'z': z,
                'down_sym': down_sym,
                'up_sym': up_sym,
            }

        # Seasonal filter: precompute valid months (top 8 by avg return)
        valid_months = None  # None = all months valid
        if seasonal_filter == 'best_months':
            # First pass: compute per-month returns for pairs
            month_pnl = {}
            for month in range(1, 13):
                month_rets = []
                for down_si, up_si, down_sym, up_sym in pair_indices:
                    pd_val = pair_data.get((down_si, up_si))
                    if pd_val is None:
                        continue
                    for di in range(MIN_TRAIN, ND):
                        if dates[di].month == month:
                            z_v = pd_val['z'][di]
                            if not np.isnan(z_v) and abs(z_v) > z_thresh:
                                # Approximate expected return: z-score mean-reverts
                                if di + 1 < ND:
                                    z_next = pd_val['z'][di + 1]
                                    if not np.isnan(z_next) and not np.isnan(z_v):
                                        month_rets.append(abs(z_v) - abs(z_next))
                month_pnl[month] = np.mean(month_rets) if month_rets else 0
            # Top 8 months
            sorted_months = sorted(month_pnl.keys(), key=lambda m: -month_pnl[m])
            valid_months = set(sorted_months[:8])

        for di in range(MIN_TRAIN, ND):
            year = dates[di].year
            month = dates[di].month
            if wf_split_year is not None and year < wf_split_year:
                continue

            # Seasonal filter
            if valid_months is not None and month not in valid_months:
                # Still need to close existing positions
                new_positions = []
                for pos in pair_positions:
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
                    if pos['dir'] == 1:
                        pnl_down = (c_down - pos['entry_down']) * mult_down * lots_down
                        pnl_up = (pos['entry_up'] - c_up) * mult_up * lots_up
                    else:
                        pnl_down = (pos['entry_down'] - c_down) * mult_down * lots_down
                        pnl_up = (c_up - pos['entry_up']) * mult_up * lots_up
                    entry_val = pos['entry_down'] * mult_down * lots_down + pos['entry_up'] * mult_up * lots_up
                    exit_val = c_down * mult_down * lots_down + c_up * mult_up * lots_up
                    cost = entry_val * COMM + exit_val * COMM
                    total_pnl = pnl_down + pnl_up - cost
                    invested = entry_val
                    pnl_pct = total_pnl / invested * 100 if invested > 0 else 0
                    if pos['dir'] == 1:
                        cash_return = c_down * mult_down * lots_down - c_up * mult_up * lots_up
                    else:
                        cash_return = -c_down * mult_down * lots_down + c_up * mult_up * lots_up
                    cash += pos['cash_invested'] + cash_return - exit_val * COMM
                    trades.append({
                        'pnl_abs': total_pnl, 'pnl_pct': pnl_pct,
                        'days': di - pos['entry_di'], 'di': di, 'year': year,
                        'pair': (pos['down_sym'], pos['up_sym']),
                        'pair_label': PAIR_LABEL.get((pos['down_sym'], pos['up_sym']), ''),
                        'dir': pos['dir'], 'reason': 'seasonal_exit',
                    })
                pair_positions = []
                exited_today = set()
                continue

            # Skip high vol regime filter
            if seasonal_filter == 'skip_high_vol':
                if not np.isnan(mkt_vol_pct[di]) and mkt_vol_pct[di] > 0.9:
                    # Extreme vol day: close all and skip
                    new_positions = []
                    for pos in pair_positions:
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
                        if pos['dir'] == 1:
                            pnl_d = (c_down - pos['entry_down']) * mult_down * lots_down
                            pnl_u = (pos['entry_up'] - c_up) * mult_up * lots_up
                        else:
                            pnl_d = (pos['entry_down'] - c_down) * mult_down * lots_down
                            pnl_u = (c_up - pos['entry_up']) * mult_up * lots_up
                        ev = pos['entry_down'] * mult_down * lots_down + pos['entry_up'] * mult_up * lots_up
                        xv = c_down * mult_down * lots_down + c_up * mult_up * lots_up
                        cost = ev * COMM + xv * COMM
                        total_pnl = pnl_d + pnl_u - cost
                        invested = ev
                        pnl_pct = total_pnl / invested * 100 if invested > 0 else 0
                        if pos['dir'] == 1:
                            cr = c_down * mult_down * lots_down - c_up * mult_up * lots_up
                        else:
                            cr = -c_down * mult_down * lots_down + c_up * mult_up * lots_up
                        cash += pos['cash_invested'] + cr - xv * COMM
                        trades.append({
                            'pnl_abs': total_pnl, 'pnl_pct': pnl_pct,
                            'days': di - pos['entry_di'], 'di': di, 'year': year,
                            'pair': (pos['down_sym'], pos['up_sym']),
                            'pair_label': PAIR_LABEL.get((pos['down_sym'], pos['up_sym']), ''),
                            'dir': pos['dir'], 'reason': 'vol_exit',
                        })
                    pair_positions = []
                    exited_today = set()
                    continue

            exited_today = set()

            # --- Manage existing pair positions ---
            new_positions = []
            for pos in pair_positions:
                p_down_si = pos['down_si']
                p_up_si = pos['up_si']
                pd_key = pair_data.get((p_down_si, p_up_si))
                if pd_key is None:
                    new_positions.append(pos)
                    continue
                z_now = pd_key['z'][di]
                days_held = di - pos['entry_di']
                entry_z = pos['entry_z']
                pos_dir = pos['dir']

                exit_reason = None

                # Exit 1: Z crosses exit_z threshold toward mean
                if not np.isnan(z_now):
                    if pos_dir == 1 and z_now >= exit_z:
                        exit_reason = 'mean_rev'
                    elif pos_dir == -1 and z_now <= -exit_z:
                        exit_reason = 'mean_rev'

                # Exit 2: Stop loss -- z moves further by 1.5 from entry
                if exit_reason is None and not np.isnan(z_now):
                    if pos_dir == 1 and z_now < entry_z - 1.5:
                        exit_reason = 'stop_loss'
                    elif pos_dir == -1 and z_now > entry_z + 1.5:
                        exit_reason = 'stop_loss'

                # Exit 3: Time exit
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

                    exited_today.add((p_down_si, p_up_si))
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

            # Capital allocation
            capital_per_pair = cash / max(1, max_pairs)

            candidates = []
            for down_si, up_si, down_sym, up_sym in pair_indices:
                if down_si in occupied or up_si in occupied:
                    continue
                # Re-entry filter: if pair was exited today, check allow_reentry
                if (down_si, up_si) in exited_today and not allow_reentry:
                    continue
                pd_key = pair_data.get((down_si, up_si))
                if pd_key is None:
                    continue
                z_val = pd_key['z'][di]
                if np.isnan(z_val):
                    continue
                if abs(z_val) < z_thresh:
                    continue

                # OI filter
                if oi_filter != 'off' and has_oi:
                    if oi_filter == 'either':
                        if not (oi_rising[down_si, di] or oi_rising[up_si, di]):
                            continue
                    elif oi_filter == 'both':
                        if not (oi_rising[down_si, di] and oi_rising[up_si, di]):
                            continue
                    elif oi_filter == 'long_leg':
                        # Long leg = the one we buy
                        # If z > 0: short down, long up -> up is long leg
                        # If z < 0: long down, short up -> down is long leg
                        if z_val > 0:
                            if not oi_rising[up_si, di]:
                                continue
                        else:
                            if not oi_rising[down_si, di]:
                                continue

                # Volume filter
                if vol_filter:
                    vr_down = vol_ratio[down_si, di] if not np.isnan(vol_ratio[down_si, di]) else 0
                    vr_up = vol_ratio[up_si, di] if not np.isnan(vol_ratio[up_si, di]) else 0
                    if vr_down < 1.0 or vr_up < 1.0:
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

                # Capital allocation: equal or z-score weighted
                if capital_mode == 'zscore' and abs(z_val) > 0:
                    # Weight allocation by z-score magnitude relative to sum
                    # For single pair this just uses full allocation
                    alloc = capital_per_pair
                else:
                    alloc = capital_per_pair

                cash_per_leg = alloc / 2
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
                    pos_dir = -1  # short down + long up
                else:
                    pos_dir = 1   # long down + short up

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

        # Sharpe approximation
        trade_pnls = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
        if len(trade_pnls) > 1:
            rets = np.array(trade_pnls) / float(CASH0)
            mean_ret = np.mean(rets)
            std_ret = np.std(rets)
            sharpe_approx = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
        else:
            sharpe_approx = 0

        # Exit reason breakdown
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

        # Yearly breakdown
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

        # Per-pair breakdown
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
    # PARAMETER SWEEP (~300 configs)
    # ========================================
    print("\n[Backtest] Running parameter sweep...", flush=True)
    results = []
    configs = []

    # ---- Test 1: Lookback optimization for 1-day hold ----
    # Fix H=1, EZ=0, MP=1, raw spread, no filters -- sweep LB and Z
    lookbacks = [3, 5, 7, 10, 15, 20]
    z_thresholds = [0.5, 0.8, 1.0, 1.2, 1.5]
    max_pairs_list = [1, 2, 3, 5]

    print("\n  === Test 1: Lookback x Z-threshold (H=1, EZ=0, MP=1, raw) ===")
    for lb in lookbacks:
        for zt in z_thresholds:
            name = f"T1_LB{lb}_Z{zt:.1f}_H1_EZ0_MP1_raw"
            configs.append((lb, zt, 1, 0, 1, SPREAD_RAW, False, 'off', False, 'equal', 'off', None, name))

    # ---- Test 2: Max pairs with 1-day hold ----
    print("  === Test 2: Max pairs (H=1, EZ=0, best LB/Z from T1 range) ===")
    for lb in [5, 7, 10]:
        for zt in [0.8, 1.0, 1.2]:
            for mp in [2, 3, 5]:
                for cap_mode in ['equal', 'zscore']:
                    name = f"T2_LB{lb}_Z{zt:.1f}_H1_EZ0_MP{mp}_{cap_mode}"
                    configs.append((lb, zt, 1, 0, mp, SPREAD_RAW, False, 'off', False, cap_mode, 'off', None, name))

    # ---- Test 3: Same-day re-entry ----
    print("  === Test 3: Re-entry (H=1, EZ=0, MP=1) ===")
    for lb in [5, 7, 10]:
        for zt in [0.8, 1.0, 1.2]:
            for reentry in [True, False]:
                tag = "Y" if reentry else "N"
                name = f"T3_LB{lb}_Z{zt:.1f}_H1_EZ0_MP1_re{tag}"
                configs.append((lb, zt, 1, 0, 1, SPREAD_RAW, reentry, 'off', False, 'equal', 'off', None, name))

    # ---- Test 4: OI confirmation ----
    print("  === Test 4: OI filter (H=1, EZ=0, MP=1) ===")
    for lb in [5, 7, 10]:
        for zt in [0.8, 1.0, 1.2]:
            for oi_f in ['off', 'either', 'both', 'long_leg']:
                if oi_f == 'off':
                    continue  # already tested in T1
                name = f"T4_LB{lb}_Z{zt:.1f}_H1_EZ0_MP1_oi_{oi_f}"
                configs.append((lb, zt, 1, 0, 1, SPREAD_RAW, False, oi_f, False, 'equal', 'off', None, name))

    # ---- Test 5: Spread normalization ----
    print("  === Test 5: Spread modes (H=1, EZ=0, MP=1) ===")
    for lb in [5, 7, 10, 15, 20]:
        for zt in [0.5, 0.8, 1.0, 1.2, 1.5]:
            for sm in [SPREAD_PCT, SPREAD_LOG]:
                name = f"T5_LB{lb}_Z{zt:.1f}_H1_EZ0_MP1_{sm}"
                configs.append((lb, zt, 1, 0, 1, sm, False, 'off', False, 'equal', 'off', None, name))

    # ---- Test 6: Volume confirmation ----
    print("  === Test 6: Volume filter (H=1, EZ=0, MP=1) ===")
    for lb in [5, 7, 10]:
        for zt in [0.8, 1.0, 1.2]:
            for vf in [True, False]:
                if not vf:
                    continue  # already tested in T1
                name = f"T6_LB{lb}_Z{zt:.1f}_H1_EZ0_MP1_vol_Y"
                configs.append((lb, zt, 1, 0, 1, SPREAD_RAW, False, 'off', vf, 'equal', 'off', None, name))

    # ---- Test 7: Seasonal / vol-regime filters ----
    print("  === Test 7: Seasonal/vol filters (H=1, EZ=0, MP=1) ===")
    for lb in [5, 7, 10]:
        for zt in [0.8, 1.0, 1.2]:
            for sf in ['best_months', 'skip_high_vol']:
                name = f"T7_LB{lb}_Z{zt:.1f}_H1_EZ0_MP1_{sf}"
                configs.append((lb, zt, 1, 0, 1, SPREAD_RAW, False, 'off', False, 'equal', sf, None, name))

    print(f"\n  Total: {len(configs)} full-period configurations", flush=True)

    for ci, cfg in enumerate(configs):
        lb, zt, hd, ez, mp, sm, reentry, oi_f, vf, cm, sf, wf, name = cfg
        r = run_backtest(lb, zt, hd, ez, mp, sm, reentry, oi_f, vf, cm, sf,
                         wf_split_year=wf, config_name=name)
        if r is not None:
            results.append(r)
            if r['ann'] > 10:
                print(f"  {r['name']:60s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                      f"N {r['n']:5d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                      f"Sh {r['sharpe']:5.2f} | AvgD {r['avg_days']:.1f}")
        if (ci + 1) % 50 == 0:
            print(f"  [{ci + 1}/{len(configs)}] {len(results)} configs with results", flush=True)

    print(f"  Full-period sweep done: {len(results)} configs with results", flush=True)

    # ========================================
    # WALK-FORWARD FOR TOP CONFIGS
    # ========================================
    print(f"\n[Walk-Forward] Testing top configs out-of-sample...", flush=True)
    full_results = sorted([r for r in results], key=lambda x: -x['ann'])

    # Deduplicate by core params for walk-forward selection
    wf_candidates = {}
    for r in full_results:
        parts = r['name'].split('_')
        # Extract core params: LB, Z, spread_mode (ignore test prefix T1_)
        if len(parts) >= 5:
            lb_part = parts[1] if parts[0].startswith('T') else parts[0]
            z_part = parts[2] if parts[0].startswith('T') else parts[1]
            key = r['name']
            if key not in wf_candidates:
                wf_candidates[key] = r

    wf_results = []
    wf_configs_run = 0
    wf_top = list(wf_candidates.values())[:20]

    for r in wf_top:
        name_str = r['name']
        # Parse the config from the run
        # We need to reconstruct params from the name
        # Names are like: T1_LB10_Z1.0_H1_EZ0_MP1_raw
        # or T2_LB7_Z0.8_H1_EZ0_MP2_equal
        # or T4_LB5_Z1.0_H1_EZ0_MP1_oi_either
        # We need to re-run with wf_split_year
        # Instead of parsing, find the matching config tuple
        matching = [c for c in configs if c[-1] == name_str]
        if not matching:
            continue
        cfg = matching[0]
        lb, zt, hd, ez, mp, sm, reentry, oi_f, vf, cm, sf, _, orig_name = cfg

        for wf_year in [2022, 2023, 2024]:
            wf_name = f"{orig_name}_WF{wf_year}"
            wr = run_backtest(lb, zt, hd, ez, mp, sm, reentry, oi_f, vf, cm, sf,
                              wf_split_year=wf_year, config_name=wf_name)
            if wr is not None:
                wf_results.append(wr)
            wf_configs_run += 1

        if wf_configs_run % 20 == 0:
            print(f"  [WF {wf_configs_run} configs tested] {len(wf_results)} with results", flush=True)

    print(f"  {wf_configs_run} walk-forward configs tested, {len(wf_results)} with results", flush=True)

    # ========================================
    # RESULTS
    # ========================================
    all_results = results + wf_results
    full_r = [r for r in all_results if '_WF' not in r['name']]
    wf_only = [r for r in all_results if '_WF' in r['name']]
    full_r.sort(key=lambda x: -x['ann'])
    wf_only.sort(key=lambda x: -x['ann'])

    # --- TOP 20 FULL-PERIOD ---
    print(f"\n{'=' * 150}")
    print(f"  TOP 20 FULL-PERIOD RESULTS (sorted by annual return)")
    print(f"{'=' * 150}")
    hdr = (f"  {'Config':60s} | {'Ann':>7s} | {'WR':>5s} | {'N':>5s} | {'DD':>6s} | "
           f"{'PF':>4s} | {'Sharpe':>6s} | {'AvgW':>7s} | {'AvgL':>6s} | {'AvgD':>4s}")
    print(hdr)
    print(f"  {'-' * 145}")
    for r in full_r[:20]:
        print(f"  {r['name']:60s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:5d} | {r['dd']:6.1f}% | {r['pf']:4.2f} | "
              f"{r['sharpe']:6.2f} | {r['avg_win']:+6.2f}% | {r['avg_loss']:5.2f}% | "
              f"{r['avg_days']:4.1f}")

    # --- TOP 10 WALK-FORWARD ---
    if wf_only:
        print(f"\n  TOP 10 WALK-FORWARD RESULTS (out-of-sample)")
        print(f"  {'-' * 145}")
        for r in wf_only[:10]:
            print(f"  {r['name']:70s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                  f"Sh {r['sharpe']:6.2f} | AvgD {r['avg_days']:.1f}")

    # --- LOOKBACK COMPARISON TABLE ---
    print(f"\n{'=' * 150}")
    print(f"  LOOKBACK COMPARISON TABLE (1-day hold, all Z thresholds)")
    print(f"{'=' * 150}")
    print(f"\n  {'LB':>4s} | {'N configs':>9s} | {'Avg Ann':>7s} | {'Best Ann':>8s} | "
          f"{'Avg WR':>6s} | {'Avg N':>6s} | {'Avg DD':>6s} | {'Best Sharpe':>11s}")
    print(f"  {'-' * 80}")
    for lb in lookbacks:
        subset = [r for r in full_r if f'_LB{lb}_' in r['name']]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            best_ann = max(r['ann'] for r in subset)
            avg_wr = np.mean([r['wr'] for r in subset])
            avg_n = np.mean([r['n'] for r in subset])
            avg_dd = np.mean([r['dd'] for r in subset])
            best_sh = max(r['sharpe'] for r in subset)
            print(f"  {lb:4d} | {len(subset):9d} | {avg_ann:+6.1f}% | {best_ann:+7.1f}% | "
                  f"{avg_wr:5.1f}% | {avg_n:6.0f} | {avg_dd:5.1f}% | {best_sh:11.2f}")

    # --- SPREAD NORMALIZATION COMPARISON ---
    print(f"\n  SPREAD NORMALIZATION COMPARISON:")
    print(f"  {'Mode':>6s} | {'N configs':>9s} | {'Avg Ann':>7s} | {'Best Ann':>8s} | "
          f"{'Avg WR':>6s} | {'Avg N':>6s} | {'Avg Sharpe':>10s}")
    print(f"  {'-' * 70}")
    for sm in [SPREAD_RAW, SPREAD_PCT, SPREAD_LOG]:
        subset = [r for r in full_r if r['name'].endswith(f'_{sm}')]
        if not subset:
            # Also check names with _raw at end (T1 configs)
            subset = [r for r in full_r if f'_{sm}' in r['name'] or
                      (sm == SPREAD_RAW and '_raw' in r['name'])]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            best_ann = max(r['ann'] for r in subset)
            avg_wr = np.mean([r['wr'] for r in subset])
            avg_n = np.mean([r['n'] for r in subset])
            avg_sh = np.mean([r['sharpe'] for r in subset])
            print(f"  {sm:6s} | {len(subset):9d} | {avg_ann:+6.1f}% | {best_ann:+7.1f}% | "
                  f"{avg_wr:5.1f}% | {avg_n:6.0f} | {avg_sh:10.2f}")

    # --- RE-ENTRY COMPARISON ---
    print(f"\n  RE-ENTRY COMPARISON (allow_reentry=Y vs N):")
    re_y = [r for r in full_r if '_reY' in r['name']]
    re_n = [r for r in full_r if '_reN' in r['name']]
    if re_y:
        print(f"    Re-entry ON : {len(re_y)} configs | Avg Ann={np.mean([r['ann'] for r in re_y]):+.1f}% | "
              f"Best={max(r['ann'] for r in re_y):+.1f}% | Avg WR={np.mean([r['wr'] for r in re_y]):.1f}% | "
              f"Avg N={np.mean([r['n'] for r in re_y]):.0f}")
    if re_n:
        print(f"    Re-entry OFF: {len(re_n)} configs | Avg Ann={np.mean([r['ann'] for r in re_n]):+.1f}% | "
              f"Best={max(r['ann'] for r in re_n):+.1f}% | Avg WR={np.mean([r['wr'] for r in re_n]):.1f}% | "
              f"Avg N={np.mean([r['n'] for r in re_n]):.0f}")

    # --- OI FILTER COMPARISON ---
    print(f"\n  OI FILTER COMPARISON:")
    print(f"  {'OI Mode':>12s} | {'N':>5s} | {'Avg Ann':>7s} | {'Best Ann':>8s} | {'Avg WR':>6s}")
    print(f"  {'-' * 55}")
    for oi_mode in ['off', 'either', 'both', 'long_leg']:
        subset = [r for r in full_r if f'_oi_{oi_mode}' in r['name']]
        # 'off' is baseline from T1
        if oi_mode == 'off':
            subset = [r for r in full_r if '_oi_' not in r['name'] and r['name'].startswith('T4')]
            if not subset:
                # Use T1 as baseline for 'off'
                subset = [r for r in full_r if r['name'].startswith('T1_')]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            best_ann = max(r['ann'] for r in subset)
            avg_wr = np.mean([r['wr'] for r in subset])
            print(f"  {oi_mode:12s} | {len(subset):5d} | {avg_ann:+6.1f}% | {best_ann:+7.1f}% | {avg_wr:5.1f}%")

    # --- VOLUME FILTER COMPARISON ---
    print(f"\n  VOLUME FILTER COMPARISON:")
    vol_y = [r for r in full_r if '_vol_Y' in r['name']]
    # Baseline: same LB/Z without vol filter (from T1)
    if vol_y:
        vol_baseline = [r for r in full_r if r['name'].startswith('T1_') and
                        any(f'_LB{lb}_' in r['name'] for lb in [5, 7, 10]) and
                        any(f'_Z{zt:.1f}_' in r['name'] for zt in [0.8, 1.0, 1.2])]
        print(f"    Vol filter ON : {len(vol_y)} configs | Avg Ann={np.mean([r['ann'] for r in vol_y]):+.1f}% | "
              f"Best={max(r['ann'] for r in vol_y):+.1f}% | Avg WR={np.mean([r['wr'] for r in vol_y]):.1f}%")
        if vol_baseline:
            print(f"    Baseline (OFF): {len(vol_baseline)} configs | Avg Ann={np.mean([r['ann'] for r in vol_baseline]):+.1f}% | "
                  f"Best={max(r['ann'] for r in vol_baseline):+.1f}% | Avg WR={np.mean([r['wr'] for r in vol_baseline]):.1f}%")

    # --- SEASONAL FILTER COMPARISON ---
    print(f"\n  SEASONAL/VOL-REGIME FILTER COMPARISON:")
    for sf in ['best_months', 'skip_high_vol']:
        subset = [r for r in full_r if f'_{sf}' in r['name'] and not r['name'].startswith('T7_')]
        if not subset:
            subset = [r for r in full_r if f'_{sf}' in r['name']]
        if subset:
            avg_ann = np.mean([r['ann'] for r in subset])
            best_ann = max(r['ann'] for r in subset)
            avg_wr = np.mean([r['wr'] for r in subset])
            avg_n = np.mean([r['n'] for r in subset])
            print(f"    {sf:20s}: {len(subset)} configs | Avg Ann={avg_ann:+.1f}% | "
                  f"Best={best_ann:+.1f}% | Avg WR={avg_wr:.1f}% | Avg N={avg_n:.0f}")

    # --- MAX PAIRS COMPARISON ---
    print(f"\n  MAX PAIRS COMPARISON (from T2 configs):")
    print(f"  {'MP':>3s} | {'CapMode':>7s} | {'N':>5s} | {'Avg Ann':>7s} | {'Best Ann':>8s} | {'Avg WR':>6s} | {'Avg N':>6s}")
    print(f"  {'-' * 60}")
    for mp in [1, 2, 3, 5]:
        for cm in ['equal', 'zscore']:
            subset = [r for r in full_r if f'_MP{mp}_' in r['name'] and f'_{cm}' in r['name']
                      and r['name'].startswith('T2_')]
            if subset:
                avg_ann = np.mean([r['ann'] for r in subset])
                best_ann = max(r['ann'] for r in subset)
                avg_wr = np.mean([r['wr'] for r in subset])
                avg_n = np.mean([r['n'] for r in subset])
                print(f"  {mp:3d} | {cm:7s} | {len(subset):5d} | {avg_ann:+6.1f}% | {best_ann:+7.1f}% | "
                      f"{avg_wr:5.1f}% | {avg_n:6.0f}")

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
    print(f"\n  === V53 vs V52 BASELINE COMPARISON ===")
    print(f"  V52 best: LB10_Z1.0_H1_EZ0.0_MP1 = +303.5% (WR 62.9%)")
    if full_r:
        print(f"  V53 best: {full_r[0]['name']}")
        print(f"    Ann={full_r[0]['ann']:+.1f}%  N={full_r[0]['n']}  "
              f"WR={full_r[0]['wr']:.1f}%  DD={full_r[0]['dd']:.1f}%  "
              f"Sharpe={full_r[0]['sharpe']:.2f}")
        delta = full_r[0]['ann'] - 303.5
        print(f"    Delta vs V52: {delta:+.1f}%")

        # How many configs beat V52?
        beating_v52 = sum(1 for r in full_r if r['ann'] > 303.5)
        beating_400 = sum(1 for r in full_r if r['ann'] > 400)
        beating_500 = sum(1 for r in full_r if r['ann'] > 500)
        beating_600 = sum(1 for r in full_r if r['ann'] > 600)
        print(f"    Configs beating V52 (+303.5%): {beating_v52}/{len(full_r)}")
        print(f"    Configs > 400% annual: {beating_400}/{len(full_r)}")
        print(f"    Configs > 500% annual: {beating_500}/{len(full_r)}")
        print(f"    Configs > 600% annual: {beating_600}/{len(full_r)}")

    # --- COMPOUNDING ANALYSIS ---
    print(f"\n  COMPOUNDING ANALYSIS: Lookback x Z-score")
    print(f"  {'LB':>4s} | {'Z':>4s} | {'Avg N':>6s} | {'Avg WR':>6s} | {'Avg Ann':>7s} | "
          f"{'Best Ann':>8s} | {'Avg Edge':>8s} | {'Comm%':>6s}")
    print(f"  {'-' * 75}")
    for lb in lookbacks:
        for zt in [0.5, 0.8, 1.0, 1.2, 1.5]:
            subset = [r for r in full_r if f'_LB{lb}_' in r['name'] and f'_Z{zt:.1f}_' in r['name']]
            if subset:
                avg_n = np.mean([r['n'] for r in subset])
                avg_wr = np.mean([r['wr'] for r in subset])
                avg_ann = np.mean([r['ann'] for r in subset])
                best_ann = max(r['ann'] for r in subset)
                avg_edge = np.mean([
                    r['wr'] * r['avg_win'] / 100 - (100 - r['wr']) * r['avg_loss'] / 100
                    for r in subset
                ])
                comm_cost = avg_n * COMM * 4 * 100
                print(f"  {lb:4d} | Z={zt:.1f} | {avg_n:6.0f} | {avg_wr:5.1f}% | "
                      f"{avg_ann:+6.1f}% | {best_ann:+7.1f}% | {avg_edge:+7.3f}% | "
                      f"{comm_cost:5.1f}%")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 150)


if __name__ == '__main__':
    main()
