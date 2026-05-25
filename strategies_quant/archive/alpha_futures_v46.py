"""
Alpha Futures V46 — Refined Supply Chain Pair Trading (Best Pairs Only)
=======================================================================
Based on V39 (+188.1% annual with 13 pairs, champion) and V41 findings:
  - V41 showed adding more pairs HURT (2 new pairs were net losers)
  - Best pair: jfi/jmfi (coke/coal) — 69.9% WR, +677M total
  - Good new pair: cfi/csfi (corn/corn starch) — 70.4% WR, +167M
  - Worst new pairs: agfi/aufi (-56M), srfi/cfi (-21M)
  - Most profitable originals: jfi/jmfi, bfi/scfi, fufi/scfi, mafi/scfi

Goal: Test whether removing weak pairs and keeping only the strongest
      pairs with more aggressive parameters boosts returns.

Approach:
  1. Test with ONLY top N pairs by historical WR/profitability
  2. Pair sets: top 5, top 8, top 10, original 13, original 13 + cfi/csfi
  3. Aggressive parameter sweep:
     - lookback: [5, 7, 10, 15, 20]
     - z_threshold: [1.0, 1.2, 1.5, 2.0]
     - hold: [2, 3, 5]
     - max_pairs: [1, 2, 3, 4]
     - exit_z: [0, 0.2, 0.5] (exit when z crosses this level)

Walk-forward validation for best configs on 2022, 2023, 2024.
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

MULT = {
    'agfi': 15, 'alfi': 5, 'aufi': 1000, 'bufi': 10, 'cufi': 5,
    'fufi': 10, 'rbfi': 10, 'znfi': 5, 'nifi': 1, 'hcfi': 10,
    'spfi': 10, 'ssfi': 5, 'sffi': 5, 'smfi': 5, 'pbfi': 5,
    'snfi': 1, 'rufi': 10, 'wrffi': 10,
    'afi': 10, 'bfi': 10, 'bbfi': 500, 'cffi': 5, 'cfi': 10,
    'csfi': 10, 'ebfi': 5, 'egfi': 10, 'fbfi': 500,
    'ifi': 100, 'jfi': 100, 'jmfi': 60, 'lfi': 5, 'mfi': 10,
    'pgfi': 20, 'ppfi': 5, 'vfi': 5, 'yfi': 10, 'pfi': 10,
    'jdfi': 5, 'lhfi': 16, 'pkfi': 5, 'rrfi': 20, 'lrfi': 20,
    'jrfi': 20, 'pmfi': 20, 'whfi': 20, 'rsfi': 20, 'cjfi': 10,
    'mafi': 10, 'apfi': 10, 'cyfi': 5, 'fgfi': 20, 'oifi': 10,
    'pfifi': 5, 'rmfi': 10, 'srfi': 10, 'tafi': 5, 'safi': 20,
    'urfi': 20, 'scfi': 1000, 'lufi': 10, 'bcfi': 5, 'nrfi': 1,
    'lgfi': 20, 'brfi': 5, 'lcfi': 1, 'sifi': 5,
    'ni': 1, 'tai': 5,
}
DEF_MULT = 10
COMM = 0.0003

# ============================================================
# PAIR SETS — ranked by approximate historical profitability
# from V39/V41 results
# ============================================================

# Tier 1 (must include): highest WR and absolute PnL
TIER1 = [
    ('jfi', 'jmfi'),   # coke/coal — 69.9% WR, +677M (best pair)
    ('bfi', 'scfi'),   # bitumen/crude
    ('fufi', 'scfi'),  # fueloil/crude
    ('mafi', 'scfi'),  # methanol/crude
    ('hcfi', 'rbfi'),  # hotcoil/rebar
]

# Tier 2: solid profitability
TIER2 = [
    ('rbfi', 'ifi'),   # rebar/iron_ore
    ('hcfi', 'ifi'),   # hotcoil/iron_ore
    ('yfi', 'afi'),    # soyoil/soybean
    ('mfi', 'afi'),    # meal/soybean
    ('pfi', 'yfi'),    # palm/soyoil
]

# Tier 3: lower profitability
TIER3 = [
    ('ppfi', 'mafi'),  # PP/methanol
    ('vfi', 'mafi'),   # PVC/methanol
    ('egfi', 'mafi'),  # EG/methanol
]

# Good new pair from V41
NEW_GOOD = [
    ('cfi', 'csfi'),   # corn/corn_starch — 70.4% WR, +167M
]

# Build pair sets
PAIRS_T5  = TIER1                                                 # top 5
PAIRS_T8  = TIER1 + TIER2[:3]                                     # top 8
PAIRS_T10 = TIER1 + TIER2                                         # top 10
PAIRS_13  = TIER1 + TIER2 + TIER3                                 # original 13
PAIRS_14  = TIER1 + TIER2 + TIER3 + NEW_GOOD                      # original 13 + cfi/csfi

PAIR_SETS = {
    'T5':  PAIRS_T5,
    'T8':  PAIRS_T8,
    'T10': PAIRS_T10,
    'P13': PAIRS_13,
    'P14': PAIRS_14,
}

PAIR_LABEL = {}
for _pair in PAIRS_14:
    PAIR_LABEL[_pair] = f"{_pair[0]}/{_pair[1]}"


def main():
    t_start = time.time()
    print("=" * 130)
    print("Alpha Futures V46 — Refined Supply Chain Pair Trading (Best Pairs Only)")
    print("Test whether removing weak pairs + aggressive params boosts returns")
    print("Pair sets: T5 (top 5), T8 (top 8), T10 (top 10), P13 (original 13), P14 (13 + cfi/csfi)")
    print("=" * 130)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    sym_to_si = {syms[si]: si for si in range(NS)}

    print(f"  {NS} commodities, {ND} days")
    for set_name, pairs in PAIR_SETS.items():
        active = sum(1 for d, u in pairs if d in sym_to_si and u in sym_to_si)
        print(f"  Pair set {set_name}: {len(pairs)} pairs ({active} active in data)")

    # ============================================================
    # PRECOMPUTE SPREADS (for all possible pairs across all sets)
    # ============================================================
    all_pairs = set()
    for pairs in PAIR_SETS.values():
        for p in pairs:
            all_pairs.add(p)

    print("\n[Signals] Computing spreads...", flush=True)
    t0 = time.time()

    spreads = {}
    for down_sym, up_sym in all_pairs:
        down_si = sym_to_si.get(down_sym, -1)
        up_si = sym_to_si.get(up_sym, -1)
        if down_si < 0 or up_si < 0:
            continue
        spread = np.full(ND, np.nan)
        for di in range(ND):
            pd = C[down_si, di]
            pu = C[up_si, di]
            if not np.isnan(pd) and not np.isnan(pu):
                spread[di] = pd - pu
        spreads[(down_si, up_si)] = spread

    print(f"  {len(spreads)} spreads computed ({time.time()-t0:.1f}s)", flush=True)

    # ============================================================
    # BACKTEST ENGINE (V39-style, simplified from V41)
    # ============================================================
    def run_backtest(pair_set_name, active_pairs, lookback, z_thresh, hold_max,
                     max_pairs, exit_z, wf_split_year=None, config_name=""):
        """
        Pair trading backtest with exit_z parameter.

        Parameters
        ----------
        pair_set_name : str — which pair set is being used
        active_pairs : list of (down_si, up_si, down_sym, up_sym)
        lookback : int — rolling window for spread mean/std
        z_thresh : float — entry threshold
        hold_max : int — max holding days
        max_pairs : int — max concurrent pair positions
        exit_z : float — exit when z crosses this level (0 = standard mean reversion)
        wf_split_year : int or None — walk-forward split year
        config_name : str — label for this config
        """
        cash = float(CASH0)
        trades = []
        pair_positions = []

        # Pre-compute per-pair z-score arrays
        pd_cache = {}
        for down_si, up_si, down_sym, up_sym in active_pairs:
            sp = spreads.get((down_si, up_si))
            if sp is None:
                continue
            sp_mean = np.full(ND, np.nan)
            sp_std = np.full(ND, np.nan)
            z_arr = np.full(ND, np.nan)

            for di in range(lookback, ND):
                window = sp[di - lookback:di]
                valid = window[~np.isnan(window)]
                if len(valid) >= lookback * 0.8:
                    sp_mean[di] = np.mean(valid)
                    sp_std[di] = np.std(valid, ddof=1)
                    if sp_std[di] > 1e-10:
                        z_arr[di] = (sp[di] - sp_mean[di]) / sp_std[di]

            pd_cache[(down_si, up_si)] = {
                'spread': sp,
                'mean': sp_mean,
                'std': sp_std,
                'z': z_arr,
                'down_sym': down_sym,
                'up_sym': up_sym,
            }

        for di in range(MIN_TRAIN, ND):
            year = dates[di].year
            if wf_split_year is not None and year < wf_split_year:
                continue

            # --- Manage existing positions ---
            new_positions = []
            for pos in pair_positions:
                p_down_si = pos['down_si']
                p_up_si = pos['up_si']
                pc = pd_cache.get((p_down_si, p_up_si))
                if pc is None:
                    new_positions.append(pos)
                    continue
                z_now = pc['z'][di]
                days_held = di - pos['entry_di']
                entry_z = pos['entry_z']
                pos_dir = pos['dir']

                exit_reason = None

                # Exit 1: Mean reversion — z crosses exit_z level
                if not np.isnan(z_now):
                    if pos_dir == 1 and z_now <= exit_z:
                        exit_reason = 'mean_rev'
                    elif pos_dir == -1 and z_now >= -exit_z:
                        exit_reason = 'mean_rev'

                # Exit 2: Stop loss — z moves further by 1.0 from entry
                if exit_reason is None and not np.isnan(z_now):
                    if pos_dir == 1 and z_now < entry_z - 1.0:
                        exit_reason = 'stop_loss'
                    elif pos_dir == -1 and z_now > entry_z + 1.0:
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
                        'pair_set': pair_set_name,
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

            candidates = []
            for down_si, up_si, down_sym, up_sym in active_pairs:
                if down_si in occupied or up_si in occupied:
                    continue
                pc = pd_cache.get((down_si, up_si))
                if pc is None:
                    continue
                z_val = pc['z'][di]
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

                cash_per_leg = cash / 2
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
                'pair_set': pair_set_name,
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
            'pair_set': pair_set_name,
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

    # ============================================================
    # BUILD CONFIGURATIONS
    # ============================================================
    print("\n[Backtest] Building configurations...", flush=True)

    configs = []

    lookbacks = [5, 7, 10, 15, 20]
    z_thresholds = [1.0, 1.2, 1.5, 2.0]
    hold_days_list = [2, 3, 5]
    max_pairs_list = [1, 2, 3, 4]
    exit_z_list = [0, 0.2, 0.5]

    for set_name, pairs in PAIR_SETS.items():
        # Build active pair indices for this set
        active_pairs = []
        for down_sym, up_sym in pairs:
            down_si = sym_to_si.get(down_sym, -1)
            up_si = sym_to_si.get(up_sym, -1)
            if down_si >= 0 and up_si >= 0:
                active_pairs.append((down_si, up_si, down_sym, up_sym))

        for lb in lookbacks:
            for zt in z_thresholds:
                for hd in hold_days_list:
                    for mp in max_pairs_list:
                        for ez in exit_z_list:
                            name = f"{set_name}_LB{lb}_Z{zt:.1f}_H{hd}_MP{mp}_EZ{ez:.1f}"
                            configs.append({
                                'set_name': set_name,
                                'active_pairs': active_pairs,
                                'lookback': lb,
                                'z_thresh': zt,
                                'hold_max': hd,
                                'max_pairs': mp,
                                'exit_z': ez,
                                'wf_split_year': None,
                                'config_name': name,
                            })

    print(f"  {len(configs)} full-period configurations", flush=True)

    # ============================================================
    # RUN SWEEP
    # ============================================================
    print("\n[Backtest] Running parameter sweep...", flush=True)
    results = []

    for ci, cfg in enumerate(configs):
        r = run_backtest(
            pair_set_name=cfg['set_name'],
            active_pairs=cfg['active_pairs'],
            lookback=cfg['lookback'],
            z_thresh=cfg['z_thresh'],
            hold_max=cfg['hold_max'],
            max_pairs=cfg['max_pairs'],
            exit_z=cfg['exit_z'],
            wf_split_year=cfg['wf_split_year'],
            config_name=cfg['config_name'],
        )
        if r is not None:
            results.append(r)
            if r['ann'] > 10:
                print(f"  {r['name']:50s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                      f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                      f"Sh {r['sharpe']:5.2f} | AvgD {r['avg_days']:.1f}")
        if (ci + 1) % 200 == 0:
            print(f"  [{ci+1}/{len(configs)}] {len(results)} configs with results", flush=True)

    results.sort(key=lambda x: -x['ann'])

    # Separate full-period from walk-forward
    full_results = [r for r in results if r is not None]

    # ============================================================
    # WALK-FORWARD FOR TOP CONFIGS
    # ============================================================
    wf_results = []
    if full_results:
        # Pick top 20 configs for walk-forward
        # Ensure we get at least 1 from each pair set
        top_for_wf = full_results[:20]

        # Also pick best from each pair set
        seen_sets = set()
        for r in full_results:
            sn = r['pair_set']
            if sn not in seen_sets:
                if r not in top_for_wf:
                    top_for_wf.append(r)
                seen_sets.add(sn)
            if len(seen_sets) == len(PAIR_SETS):
                break

        wf_configs = []
        for r in top_for_wf:
            # Find original config
            orig = None
            for c in configs:
                if c['config_name'] == r['name']:
                    orig = c
                    break
            if orig is None:
                continue
            for wf_year in [2022, 2023, 2024]:
                wf_name = f"{orig['config_name']}_WF{wf_year}"
                wf_configs.append({**orig, 'wf_split_year': wf_year, 'config_name': wf_name})

        print(f"\n  Running {len(wf_configs)} walk-forward configs...", flush=True)
        for ci, cfg in enumerate(wf_configs):
            r = run_backtest(
                pair_set_name=cfg['set_name'],
                active_pairs=cfg['active_pairs'],
                lookback=cfg['lookback'],
                z_thresh=cfg['z_thresh'],
                hold_max=cfg['hold_max'],
                max_pairs=cfg['max_pairs'],
                exit_z=cfg['exit_z'],
                wf_split_year=cfg['wf_split_year'],
                config_name=cfg['config_name'],
            )
            if r is not None:
                wf_results.append(r)
            if (ci + 1) % 30 == 0:
                print(f"    [{ci+1}/{len(wf_configs)}] {len(wf_results)} WF results", flush=True)

        if wf_results:
            wf_results.sort(key=lambda x: -x['ann'])

    # ============================================================
    # RESULTS OUTPUT
    # ============================================================
    print(f"\n{'=' * 140}")
    print(f"  TOP 20 FULL-PERIOD RESULTS (sorted by annual return)")
    print(f"{'=' * 140}")
    hdr = (f"  {'Config':50s} | {'Ann':>7s} | {'WR':>5s} | {'N':>4s} | {'DD':>6s} | "
           f"{'PF':>4s} | {'Sharpe':>6s} | {'AvgW':>7s} | {'AvgL':>6s} | {'AvgD':>4s}")
    print(hdr)
    print(f"  {'-' * 135}")
    for r in full_results[:20]:
        print(f"  {r['name']:50s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:4.2f} | "
              f"{r['sharpe']:6.2f} | {r['avg_win']:+6.2f}% | {r['avg_loss']:5.2f}% | "
              f"{r['avg_days']:4.1f}")

    # Walk-forward top 10
    if wf_results:
        print(f"\n{'=' * 140}")
        print(f"  TOP 10 WALK-FORWARD RESULTS (out-of-sample)")
        print(f"{'=' * 140}")
        for r in wf_results[:10]:
            print(f"  {r['name']:60s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                  f"Sh {r['sharpe']:6.2f}")

    # Best config detail
    if full_results:
        best = full_results[0]
        print(f"\n{'=' * 140}")
        print(f"  BEST CONFIG DETAIL: {best['name']}")
        print(f"  Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  N={best['n']}  "
              f"DD={best['dd']:.1f}%  PF={best['pf']:.2f}  Sharpe={best['sharpe']:.2f}  "
              f"Final={best['cash']:.0f}")
        print(f"{'=' * 140}")

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

    # ============================================================
    # BEST PER PAIR SET (top 3 from each)
    # ============================================================
    print(f"\n{'=' * 140}")
    print(f"  BEST PER PAIR SET (top 3 from each)")
    print(f"{'=' * 140}")
    for set_name in PAIR_SETS.keys():
        set_results = [r for r in full_results if r['pair_set'] == set_name]
        if not set_results:
            print(f"\n  {set_name}: (no results)")
            continue
        print(f"\n  {set_name} ({len(PAIR_SETS[set_name])} pairs):")
        for r in set_results[:3]:
            print(f"    {r['name']:50s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                  f"Sh {r['sharpe']:6.2f}")

    # ============================================================
    # PAIR SET COMPARISON (best config from each set, side by side)
    # ============================================================
    print(f"\n{'=' * 140}")
    print(f"  PAIR SET COMPARISON (best annual from each set)")
    print(f"{'=' * 140}")
    comparison = []
    for set_name in PAIR_SETS.keys():
        set_results = [r for r in full_results if r['pair_set'] == set_name]
        if set_results:
            best_set = set_results[0]
            comparison.append((set_name, best_set))
    comparison.sort(key=lambda x: -x[1]['ann'])
    for set_name, r in comparison:
        print(f"  {set_name:4s} ({len(PAIR_SETS[set_name]):2d} pairs): "
              f"Ann={r['ann']:+7.1f}%  WR={r['wr']:5.1f}%  N={r['n']:4d}  "
              f"DD={r['dd']:6.1f}%  PF={r['pf']:4.2f}  Sh={r['sharpe']:5.2f}  "
              f"AvgD={r['avg_days']:.1f}  | {r['name']}")

    # ============================================================
    # PARAMETER SENSITIVITY (top 20)
    # ============================================================
    if full_results:
        print(f"\n{'=' * 140}")
        print(f"  PARAMETER SENSITIVITY ANALYSIS (top 20 configs)")
        print(f"{'=' * 140}")

        # Lookback distribution
        lb_counts = {}
        for r in full_results[:20]:
            parts = r['name'].split('_')
            lb_val = None
            for p in parts:
                if p.startswith('LB'):
                    lb_val = p
                    break
            if lb_val:
                lb_counts[lb_val] = lb_counts.get(lb_val, 0) + 1
        print(f"\n  Lookback distribution in top 20:")
        for k in sorted(lb_counts.keys()):
            print(f"    {k}: {lb_counts[k]} configs")

        # Z threshold distribution
        zt_counts = {}
        for r in full_results[:20]:
            parts = r['name'].split('_')
            zt_val = None
            for p in parts:
                if p.startswith('Z'):
                    zt_val = p
                    break
            if zt_val:
                zt_counts[zt_val] = zt_counts.get(zt_val, 0) + 1
        print(f"\n  Z threshold distribution in top 20:")
        for k in sorted(zt_counts.keys()):
            print(f"    {k}: {zt_counts[k]} configs")

        # Hold distribution
        hd_counts = {}
        for r in full_results[:20]:
            parts = r['name'].split('_')
            hd_val = None
            for p in parts:
                if p.startswith('H') and not p.startswith('HCF'):
                    hd_val = p
                    break
            if hd_val:
                hd_counts[hd_val] = hd_counts.get(hd_val, 0) + 1
        print(f"\n  Hold days distribution in top 20:")
        for k in sorted(hd_counts.keys()):
            print(f"    {k}: {hd_counts[k]} configs")

        # Max pairs distribution
        mp_counts = {}
        for r in full_results[:20]:
            parts = r['name'].split('_')
            mp_val = None
            for p in parts:
                if p.startswith('MP'):
                    mp_val = p
                    break
            if mp_val:
                mp_counts[mp_val] = mp_counts.get(mp_val, 0) + 1
        print(f"\n  Max pairs distribution in top 20:")
        for k in sorted(mp_counts.keys()):
            print(f"    {k}: {mp_counts[k]} configs")

        # Exit Z distribution
        ez_counts = {}
        for r in full_results[:20]:
            parts = r['name'].split('_')
            ez_val = None
            for p in parts:
                if p.startswith('EZ'):
                    ez_val = p
                    break
            if ez_val:
                ez_counts[ez_val] = ez_counts.get(ez_val, 0) + 1
        print(f"\n  Exit Z distribution in top 20:")
        for k in sorted(ez_counts.keys()):
            print(f"    {k}: {ez_counts[k]} configs")

        # Pair set distribution
        ps_counts = {}
        for r in full_results[:20]:
            ps_counts[r['pair_set']] = ps_counts.get(r['pair_set'], 0) + 1
        print(f"\n  Pair set distribution in top 20:")
        for k in sorted(ps_counts.keys()):
            print(f"    {k} ({len(PAIR_SETS[k])} pairs): {ps_counts[k]} configs")

    # ============================================================
    # TOP 5 YEARLY BREAKDOWN
    # ============================================================
    if len(full_results) >= 2:
        print(f"\n  YEARLY BREAKDOWN FOR TOP 5 CONFIGS:")
        for idx, r in enumerate(full_results[:5]):
            print(f"\n  #{idx+1}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.1f}%, "
                  f"DD={r['dd']:.1f}%, Sharpe={r['sharpe']:.2f})")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:3d}t  WR={wr_y:5.1f}%  PnL={ys['pnl']:+.1f}%  "
                      f"Abs={ys['pnl_abs_sum']:+.0f}")

    # ============================================================
    # PER-PAIR PROFITABILITY ACROSS TOP 20
    # ============================================================
    if full_results:
        print(f"\n  PAIR PROFITABILITY ACROSS TOP 20 CONFIGS:")
        pair_summary = {}
        for r in full_results[:20]:
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

    # ============================================================
    # WALK-FORWARD YEARLY COMPARISON
    # ============================================================
    if wf_results:
        print(f"\n{'=' * 140}")
        print(f"  WALK-FORWARD YEARLY COMPARISON FOR BEST CONFIGS")
        print(f"{'=' * 140}")
        from collections import defaultdict
        wf_by_config = defaultdict(dict)
        for r in wf_results:
            base = r['name'].rsplit('_WF', 1)[0]
            year = r['name'].rsplit('_WF', 1)[1]
            wf_by_config[base][year] = r

        # Show top 5 base configs that have WF data
        shown = 0
        for r in full_results:
            if r['name'] in wf_by_config:
                base = r['name']
                wf_data = wf_by_config[base]
                print(f"\n  {base} (full: Ann={r['ann']:+.1f}%, WR={r['wr']:.1f}%, "
                      f"DD={r['dd']:.1f}%, Sh={r['sharpe']:.2f}):")
                for yr in ['2022', '2023', '2024']:
                    if yr in wf_data:
                        wr = wf_data[yr]
                        print(f"    WF{yr}: Ann={wr['ann']:+7.1f}%  WR={wr['wr']:5.1f}%  "
                              f"N={wr['n']:3d}  DD={wr['dd']:5.1f}%  PF={wr['pf']:.2f}  "
                              f"Sh={wr['sharpe']:.2f}")
                    else:
                        print(f"    WF{yr}: (no data)")
                shown += 1
                if shown >= 5:
                    break

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 140)


if __name__ == '__main__':
    main()
