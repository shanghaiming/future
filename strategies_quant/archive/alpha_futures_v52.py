"""
Alpha Futures V52 — Ultra Short-Term Pair Trading (1-2 Day Hold)
================================================================
Key hypothesis: V39 uses 3-day hold and Z=1.5 entry, producing ~2790 trades.
If we lower Z threshold and hold period, we get more trades.
More trades = more compounding opportunities.

The question: Does WR hold up at higher frequency?
Even if WR drops from 60.6% to 55%, if trade count triples from 2790 to 8370,
the extra compounding could push returns past +188%.

Parameter sweep:
  Z threshold: [0.5, 0.8, 1.0, 1.2, 1.5]
  Lookback: [5, 7, 10]
  Hold days: [1, 2, 3]
  Exit Z: [0, 0.2]
  Max pairs: [1, 2, 3, 5]
  Capital: dynamic (100% when 1 pair, 50% when 2)
  Walk-forward for best (2023, 2024)

~200 configs. Print: top 20 full-period, top 10 walk-forward,
trade count analysis, per-hold-period comparison.
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


def main():
    t_start = time.time()
    print("=" * 130)
    print("Alpha Futures V52 -- Ultra Short-Term Pair Trading (1-2 Day Hold)")
    print("Core: Lower Z threshold + shorter hold = more trades = more compounding")
    print("Hypothesis: Even if WR drops from 60.6% to 55%, triple trade count wins")
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
        else:
            print(f"  WARNING: pair ({down_sym}, {up_sym}) not found in data "
                  f"(down_si={down_si}, up_si={up_si})")

    print(f"  {NS} commodities, {ND} days, {len(pair_indices)} active pairs")

    # ========================================
    # PRECOMPUTE SPREADS
    # ========================================
    print("\n[Signals] Computing spreads...", flush=True)
    t0 = time.time()

    spreads = {}
    for down_si, up_si, down_sym, up_sym in pair_indices:
        spread = np.full(ND, np.nan)
        for di in range(ND):
            pd = C[down_si, di]
            pu = C[up_si, di]
            if not np.isnan(pd) and not np.isnan(pu):
                spread[di] = pd - pu
        spreads[(down_si, up_si)] = spread

    print(f"  Spreads computed ({time.time()-t0:.1f}s)", flush=True)

    # ========================================
    # BACKTEST ENGINE FOR ULTRA SHORT-TERM PAIR TRADING
    # ========================================
    def run_backtest(lookback, z_thresh, hold_max, exit_z, max_pairs,
                     wf_split_year=None, config_name=""):
        """
        Ultra short-term pair trading backtest.

        Key differences from V39:
        - exit_z: exit when z-score crosses this threshold (not just 0)
        - Shorter hold periods (1-2 days)
        - Lower z thresholds (0.5-1.5)
        - Dynamic capital allocation based on number of pairs
        """
        cash = float(CASH0)
        trades = []
        pair_positions = []

        # Pre-compute per-pair rolling z-scores
        pair_data = {}
        for down_si, up_si, down_sym, up_sym in pair_indices:
            sp = spreads[(down_si, up_si)]
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

        for di in range(MIN_TRAIN, ND):
            year = dates[di].year
            if wf_split_year is not None and year < wf_split_year:
                continue

            # --- Manage existing pair positions ---
            new_positions = []
            for pos in pair_positions:
                p_down_si = pos['down_si']
                p_up_si = pos['up_si']
                z_now = pair_data[(p_down_si, p_up_si)]['z'][di]
                days_held = di - pos['entry_di']
                entry_z = pos['entry_z']
                pos_dir = pos['dir']  # +1 = long down/short up, -1 = short down/long up

                exit_reason = None

                # Exit 1: Z crosses exit_z threshold toward mean
                # For pos_dir=1 (long down, short up): entered when z < -thresh, exit when z >= exit_z
                # For pos_dir=-1 (short down, long up): entered when z > thresh, exit when z <= exit_z
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

            # Dynamic capital: allocate 100%/max_pairs of cash to each position
            # This ensures full capital utilization regardless of pair count
            capital_per_pair = cash / max(1, max_pairs)

            candidates = []
            for down_si, up_si, down_sym, up_sym in pair_indices:
                if down_si in occupied or up_si in occupied:
                    continue
                pd = pair_data[(down_si, up_si)]
                z_val = pd['z'][di]
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

                # Use dynamic capital per pair
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

        # Sharpe approximation from per-trade PnLs
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
    # PARAMETER SWEEP (~200 configs)
    # ========================================
    print("\n[Backtest] Running parameter sweep...", flush=True)
    results = []
    configs = []

    lookbacks = [5, 7, 10]
    z_thresholds = [0.5, 0.8, 1.0, 1.2, 1.5]
    hold_days_list = [1, 2, 3]
    exit_z_list = [0, 0.2]
    max_pairs_list = [1, 2, 3, 5]

    # Full-period configs
    for lb in lookbacks:
        for zt in z_thresholds:
            for hd in hold_days_list:
                for ez in exit_z_list:
                    for mp in max_pairs_list:
                        name = f"LB{lb}_Z{zt:.1f}_H{hd}_EZ{ez:.1f}_MP{mp}"
                        configs.append((lb, zt, hd, ez, mp, None, name))

    print(f"  {len(configs)} full-period configurations", flush=True)

    for ci, (lb, zt, hd, ez, mp, wf, name) in enumerate(configs):
        r = run_backtest(lb, zt, hd, ez, mp, wf_split_year=wf, config_name=name)
        if r is not None:
            results.append(r)
            if r['ann'] > 10:
                print(f"  {r['name']:45s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                      f"N {r['n']:5d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                      f"Sh {r['sharpe']:5.2f} | AvgD {r['avg_days']:.1f}")
        if (ci + 1) % 60 == 0:
            print(f"  [{ci+1}/{len(configs)}] {len(results)} configs with results", flush=True)

    # Walk-forward for top configs
    print(f"\n[Walk-Forward] Testing top configs out-of-sample...", flush=True)
    full_results = [r for r in results]
    full_results.sort(key=lambda x: -x['ann'])

    # Select top configs for walk-forward: top 20 by ann, but deduplicate by hold period
    wf_candidates = {}
    for r in full_results:
        parts = r['name'].split('_')
        lb_part = parts[0]   # LB
        z_part = parts[1]    # Z
        h_part = parts[2]    # H
        ez_part = parts[3]   # EZ
        key = f"{lb_part}_{z_part}_{h_part}_{ez_part}"
        if key not in wf_candidates:
            wf_candidates[key] = r

    wf_results = []
    wf_configs_run = 0
    for key, r in list(wf_candidates.items())[:30]:
        parts = r['name'].split('_')
        lb = int(parts[0][2:])
        zt = float(parts[1][1:])
        hd = int(parts[2][1:])
        ez = float(parts[3][2:])
        # Run all max_pairs variants for the best params
        for mp in max_pairs_list:
            for wf_year in [2023, 2024]:
                name = f"{r['name']}_WF{wf_year}"
                # Reconstruct the config matching the original mp
                orig_mp = int(parts[4][2:])
                if mp != orig_mp:
                    name = f"LB{lb}_Z{zt:.1f}_H{hd}_EZ{ez:.1f}_MP{mp}_WF{wf_year}"
                wr = run_backtest(lb, zt, hd, ez, mp, wf_split_year=wf_year, config_name=name)
                if wr is not None:
                    wf_results.append(wr)
                wf_configs_run += 1
        if wf_configs_run % 60 == 0:
            print(f"  [WF {wf_configs_run} configs tested] {len(wf_results)} with results", flush=True)

    print(f"  {wf_configs_run} walk-forward configs tested, {len(wf_results)} with results", flush=True)

    # ========================================
    # RESULTS
    # ========================================
    all_results = results + wf_results
    full_results = [r for r in all_results if '_WF' not in r['name']]
    wf_only = [r for r in all_results if '_WF' in r['name']]
    full_results.sort(key=lambda x: -x['ann'])
    wf_only.sort(key=lambda x: -x['ann'])

    # --- TOP 20 FULL-PERIOD ---
    print(f"\n{'=' * 140}")
    print(f"  TOP 20 FULL-PERIOD RESULTS (sorted by annual return)")
    print(f"{'=' * 140}")
    hdr = (f"  {'Config':45s} | {'Ann':>7s} | {'WR':>5s} | {'N':>5s} | {'DD':>6s} | "
           f"{'PF':>4s} | {'Sharpe':>6s} | {'AvgW':>7s} | {'AvgL':>6s} | {'AvgD':>4s}")
    print(hdr)
    print(f"  {'-' * 135}")
    for r in full_results[:20]:
        print(f"  {r['name']:45s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:5d} | {r['dd']:6.1f}% | {r['pf']:4.2f} | "
              f"{r['sharpe']:6.2f} | {r['avg_win']:+6.2f}% | {r['avg_loss']:5.2f}% | "
              f"{r['avg_days']:4.1f}")

    # --- TOP 10 WALK-FORWARD ---
    if wf_only:
        print(f"\n  TOP 10 WALK-FORWARD RESULTS (out-of-sample)")
        print(f"  {'-' * 135}")
        for r in wf_only[:10]:
            print(f"  {r['name']:55s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                  f"Sh {r['sharpe']:6.2f} | AvgD {r['avg_days']:.1f}")

    # --- TRADE COUNT ANALYSIS ---
    print(f"\n{'=' * 140}")
    print(f"  TRADE COUNT ANALYSIS")
    print(f"{'=' * 140}")
    print(f"\n  Key question: Does more trades at lower WR compound better?")
    print(f"\n  {'Config':45s} | {'N':>5s} | {'WR':>5s} | {'Ann':>7s} | {'Edge':>6s} | {'CommImp':>7s}")
    print(f"  {'-' * 90}")

    # Sort by trade count descending for the trade count analysis
    for r in sorted(full_results, key=lambda x: -x['n'])[:20]:
        edge = r['wr'] * r['avg_win'] / 100 - (100 - r['wr']) * r['avg_loss'] / 100
        # Estimate commission impact: COMM per leg per side * 4 sides * N trades
        comm_impact = r['n'] * COMM * 4 * 100  # rough % of capital
        print(f"  {r['name']:45s} | {r['n']:5d} | {r['wr']:5.1f}% | "
              f"{r['ann']:+7.1f}% | {edge:+6.2f}% | {comm_impact:6.1f}%")

    # --- PER-HOLD-PERIOD COMPARISON ---
    print(f"\n{'=' * 140}")
    print(f"  PER-HOLD-PERIOD COMPARISON (best config per hold period)")
    print(f"{'=' * 140}")

    for hd in hold_days_list:
        hd_results = [r for r in full_results if f"_H{hd}_" in r['name']]
        if hd_results:
            hd_results.sort(key=lambda x: -x['ann'])
            best = hd_results[0]
            avg_n = np.mean([r['n'] for r in hd_results])
            avg_wr = np.mean([r['wr'] for r in hd_results])
            avg_ann = np.mean([r['ann'] for r in hd_results])
            print(f"\n  Hold={hd} day(s):")
            print(f"    Best: {best['name']}")
            print(f"      Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  N={best['n']}  "
                  f"DD={best['dd']:.1f}%  PF={best['pf']:.2f}  Sharpe={best['sharpe']:.2f}")
            print(f"    Averages across {len(hd_results)} configs: "
                  f"Ann={avg_ann:+.1f}%  WR={avg_wr:.1f}%  N={avg_n:.0f}")

    # --- PER Z-THRESHOLD COMPARISON ---
    print(f"\n  PER Z-THRESHOLD COMPARISON:")
    for zt in z_thresholds:
        zt_results = [r for r in full_results if f"_Z{zt:.1f}_" in r['name']]
        if zt_results:
            avg_n = np.mean([r['n'] for r in zt_results])
            avg_wr = np.mean([r['wr'] for r in zt_results])
            avg_ann = np.mean([r['ann'] for r in zt_results])
            best = max(zt_results, key=lambda x: x['ann'])
            print(f"    Z={zt:.1f}: {len(zt_results)} configs | "
                  f"Avg Ann={avg_ann:+.1f}%  Avg WR={avg_wr:.1f}%  Avg N={avg_n:.0f} | "
                  f"Best Ann={best['ann']:+.1f}% (N={best['n']})")

    # --- PER EXIT-Z COMPARISON ---
    print(f"\n  PER EXIT-Z COMPARISON:")
    for ez in exit_z_list:
        ez_results = [r for r in full_results if f"_EZ{ez:.1f}_" in r['name']]
        if ez_results:
            avg_n = np.mean([r['n'] for r in ez_results])
            avg_wr = np.mean([r['wr'] for r in ez_results])
            avg_ann = np.mean([r['ann'] for r in ez_results])
            best = max(ez_results, key=lambda x: x['ann'])
            print(f"    ExitZ={ez:.1f}: {len(ez_results)} configs | "
                  f"Avg Ann={avg_ann:+.1f}%  Avg WR={avg_wr:.1f}%  Avg N={avg_n:.0f} | "
                  f"Best Ann={best['ann']:+.1f}% (N={best['n']})")

    # --- BEST CONFIG DETAIL ---
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
            print(f"    {reason:12s}: {s['n']:4d} trades  WR={rwr:5.1f}%  "
                  f"PnL={s['pnl_pct_sum']:+.1f}%  Abs={s['pnl']:+.0f}")

    # --- YEARLY FOR TOP 5 ---
    if len(full_results) >= 2:
        print(f"\n  YEARLY BREAKDOWN FOR TOP 5 CONFIGS:")
        for idx, r in enumerate(full_results[:5]):
            print(f"\n  #{idx+1}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.1f}%, "
                  f"DD={r['dd']:.1f}%, Sharpe={r['sharpe']:.2f}, N={r['n']})")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:4d}t  WR={wr_y:5.1f}%  PnL={ys['pnl']:+.1f}%")

    # --- PAIR PROFITABILITY ACROSS TOP 20 ---
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
            print(f"    {p:25s}: {ps['n']:5d} trades  WR={wr_p:5.1f}%  Total Abs={ps['pnl']:+12.0f}")

    # --- V39 COMPARISON ---
    print(f"\n  === V52 vs V39 BASELINE COMPARISON ===")
    print(f"  V39 best: LB10_Z1.5_H3_MP2 = +188.1% (2790 trades, WR 60.6%)")
    if full_results:
        print(f"  V52 best: {full_results[0]['name']}")
        print(f"    Ann={full_results[0]['ann']:+.1f}%  N={full_results[0]['n']}  "
              f"WR={full_results[0]['wr']:.1f}%  DD={full_results[0]['dd']:.1f}%  "
              f"Sharpe={full_results[0]['sharpe']:.2f}")
        delta = full_results[0]['ann'] - 188.1
        print(f"    Delta vs V39: {delta:+.1f}%")

        # How many configs beat V39?
        beating_v39 = sum(1 for r in full_results if r['ann'] > 188.1)
        print(f"    Configs beating V39 (+188.1%): {beating_v39}/{len(full_results)}")

    # --- COMPOUNDING ANALYSIS ---
    print(f"\n  COMPOUNDING ANALYSIS: Trade count vs Returns")
    print(f"  {'Hold':>4s} | {'Z':>4s} | {'Avg N':>6s} | {'Avg WR':>6s} | {'Avg Ann':>7s} | "
          f"{'Best Ann':>8s} | {'Comm Cost':>9s} | {'Net Edge':>8s}")
    print(f"  {'-' * 70}")
    for hd in hold_days_list:
        for zt in [0.5, 0.8, 1.0, 1.5]:
            subset = [r for r in full_results if f"_H{hd}_" in r['name'] and f"_Z{zt:.1f}_" in r['name']]
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
                print(f"  H={hd:d}  | Z={zt:.1f} | {avg_n:6.0f} | {avg_wr:5.1f}% | "
                      f"{avg_ann:+6.1f}% | {best_ann:+7.1f}% | {comm_cost:8.1f}% | "
                      f"{avg_edge:+7.3f}%")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 140)


if __name__ == '__main__':
    main()
