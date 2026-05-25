"""
Alpha Futures V60 -- Deep Dive on LOG Spread Mode
==================================================
V58 rigorous walk-forward showed LOG spread was selected by training in
ALL recent windows (2023-2025). This script tests LOG spread exclusively
with an extensive parameter sweep to find the LOG-specific optimum.

Parameter sweep (~180 configs):
  lookback:   [3, 5, 7, 10, 15, 20]
  z_threshold:[0.5, 0.8, 1.0, 1.2, 1.5]
  hold:       [1, 2]
  exit_z:     [0, 0.2, 0.5]
  max_pairs:  [1, 2]

Total: 6 x 5 x 2 x 3 x 2 = 360 combos (filter to ~250 viable ones)

Walk-forward for top 20 configs on years 2022, 2023, 2024.

Print: top 20 full-period, top 10 walk-forward, lookback comparison,
       per-pair stats.
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

# Standard 13 pairs + cfi/csfi (V46 showed it was profitable)
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

# Walk-forward test years
WF_YEARS = [2022, 2023, 2024]

# Parameter sweep space
LOOKBACKS = [3, 5, 7, 10, 15, 20]
Z_THRESHOLDS = [0.5, 0.8, 1.0, 1.2, 1.5]
HOLD_MAX = [1, 2]
EXIT_Z = [0, 0.2, 0.5]
MAX_PAIRS = [1, 2]


def main():
    t_start = time.time()
    print("=" * 150)
    print("Alpha Futures V60 -- LOG Spread Deep Dive")
    print("V58 showed LOG spread dominates in recent walk-forward windows (2023-2025)")
    print("This script sweeps LOG parameters exhaustively to find the LOG-specific optimum")
    print("=" * 150)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    sym_to_si = {syms[si]: si for si in range(NS)}

    # Build pair index mapping
    pair_indices = []
    for down_sym, up_sym in PAIRS:
        down_si = sym_to_si.get(down_sym, -1)
        up_si = sym_to_si.get(up_sym, -1)
        if down_si >= 0 and up_si >= 0:
            pair_indices.append((down_si, up_si, down_sym, up_sym))
        else:
            print(f"  WARNING: pair ({down_sym}, {up_sym}) not found in data")

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
    # PRECOMPUTE LOG SPREADS AND Z-SCORES FOR ALL LOOKBACKS
    # ================================================================
    print("\n[Signals] Precomputing LOG spreads and z-scores for all lookbacks...", flush=True)
    t0 = time.time()

    log_spreads = {}
    z_scores = {}

    for down_si, up_si, down_sym, up_sym in pair_indices:
        key = (down_si, up_si)
        spread = np.full(ND, np.nan)
        for di in range(ND):
            pd_val = C[down_si, di]
            pu = C[up_si, di]
            if np.isnan(pd_val) or np.isnan(pu) or pu <= 0 or pd_val <= 0:
                continue
            spread[di] = np.log(pd_val) - np.log(pu)

        log_spreads[key] = spread
        z_scores[key] = {}

        for lb in LOOKBACKS:
            z = np.full(ND, np.nan)
            for di in range(lb, ND):
                window = spread[di - lb:di]
                valid = window[~np.isnan(window)]
                if len(valid) >= max(3, lb * 0.8):
                    m_val = np.mean(valid)
                    s_val = np.std(valid, ddof=1)
                    if s_val > 1e-10:
                        z[di] = (spread[di] - m_val) / s_val
            z_scores[key][lb] = z

    print(f"  LOG spreads + z-scores precomputed for {len(pair_indices)} pairs x "
          f"{len(LOOKBACKS)} lookbacks ({time.time() - t0:.1f}s)", flush=True)

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(lookback, z_thresh, hold_max=1, exit_z=0.0, max_pairs=1,
                     start_year=None, end_year=None, config_name=""):
        """
        Run pair trading backtest using LOG spread with a single lookback.
        Optionally restrict trading to a date range.
        """
        cash = float(CASH0)
        trades = []
        pair_positions = []

        # Determine date range
        start_di = MIN_TRAIN
        end_di = ND
        if start_year is not None:
            if start_year in year_start_di:
                start_di = year_start_di[start_year]
            else:
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
                z_arr = z_scores.get((p_down_si, p_up_si), {}).get(lookback)
                if z_arr is None:
                    new_positions.append(pos)
                    continue
                z_now = z_arr[di] if di < len(z_arr) else np.nan
                days_held = di - pos['entry_di']
                entry_z = pos['entry_z']
                pos_dir = pos['dir']

                exit_reason = None

                # Mean reversion exit: z crossed back toward zero
                if not np.isnan(z_now):
                    if pos_dir == 1 and z_now >= exit_z:
                        exit_reason = 'mean_rev'
                    elif pos_dir == -1 and z_now <= -exit_z:
                        exit_reason = 'mean_rev'

                # Stop loss: z moved further against us by 1.5 sigma
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

                z_arr = z_scores.get((down_si, up_si), {}).get(lookback)
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

            # Sort by absolute z-score (strongest signal first)
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

                # Trade in the direction of mean reversion
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

        # Annualized return
        first_di = min(t['di'] for t in trades)
        last_di = max(t['di'] for t in trades)
        if last_di > first_di:
            days_total = (dates[last_di] - dates[first_di]).days
        else:
            days_total = 365
        yr = max(days_total / 365.25, 0.01)
        ann = ((cash / CASH0) ** (1 / yr) - 1) * 100

        # Sharpe approximation
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
    # FULL PARAMETER SWEEP
    # ================================================================
    total_combos = len(LOOKBACKS) * len(Z_THRESHOLDS) * len(HOLD_MAX) * len(EXIT_Z) * len(MAX_PAIRS)
    print(f"\n{'=' * 150}")
    print(f"  FULL-PERIOD PARAMETER SWEEP ({total_combos} configs)")
    print(f"  LOG spread only | lookbacks: {LOOKBACKS} | z_thresholds: {Z_THRESHOLDS}")
    print(f"  hold_max: {HOLD_MAX} | exit_z: {EXIT_Z} | max_pairs: {MAX_PAIRS}")
    print(f"{'=' * 150}")

    results = []
    combo_count = 0
    t_sweep_start = time.time()

    for lb in LOOKBACKS:
        for zt in Z_THRESHOLDS:
            for hm in HOLD_MAX:
                for ez in EXIT_Z:
                    for mp in MAX_PAIRS:
                        combo_count += 1
                        name = f"LOG_LB{lb}_Z{zt:.1f}_H{hm}_EZ{ez}_MP{mp}"
                        r = run_backtest(
                            lookback=lb, z_thresh=zt,
                            hold_max=hm, exit_z=ez, max_pairs=mp,
                            config_name=name,
                        )
                        if r is not None:
                            results.append(r)

                        if combo_count % 50 == 0:
                            elapsed = time.time() - t_sweep_start
                            print(f"  [{combo_count}/{total_combos}] {len(results)} with results "
                                  f"({elapsed:.1f}s)", flush=True)

    results.sort(key=lambda x: -x['ann'])
    print(f"\n  Sweep complete: {len(results)}/{total_combos} configs produced results "
          f"({time.time() - t_sweep_start:.1f}s)", flush=True)

    # ================================================================
    # TOP 20 FULL-PERIOD RESULTS
    # ================================================================
    print(f"\n{'=' * 150}")
    print(f"  TOP 20 FULL-PERIOD RESULTS (LOG Spread)")
    print(f"{'=' * 150}")
    print(f"  {'#':>2s} | {'Config':30s} | {'Ann':>8s} | {'WR':>5s} | {'N':>5s} | "
          f"{'DD':>6s} | {'PF':>5s} | {'Sharpe':>7s} | {'AvgW':>6s} | {'AvgL':>6s} | "
          f"{'AvgD':>5s} | {'Cash':>12s}")
    print(f"  {'-' * 145}")

    for i, r in enumerate(results[:20]):
        print(f"  {i+1:2d} | {r['name']:30s} | {r['ann']:+7.1f}% | {r['wr']:4.1f}% | "
              f"{r['n']:5d} | {r['dd']:5.1f}% | {r['pf']:4.2f} | {r['sharpe']:6.2f} | "
              f"{r['avg_win']:+5.2f}% | {r['avg_loss']:5.2f}% | {r['avg_days']:4.1f} | "
              f"{r['cash']:11.0f}")

    # ================================================================
    # LOOKBACK COMPARISON
    # ================================================================
    print(f"\n{'=' * 150}")
    print(f"  LOOKBACK COMPARISON (best config per lookback)")
    print(f"{'=' * 150}")
    print(f"  {'Lookback':>8s} | {'Best Config':30s} | {'Ann':>8s} | {'WR':>5s} | {'N':>5s} | "
          f"{'DD':>6s} | {'PF':>5s} | {'Sharpe':>7s}")
    print(f"  {'-' * 100}")

    for lb in LOOKBACKS:
        lb_results = [r for r in results if f'_LB{lb}_' in r['name']]
        if lb_results:
            best = lb_results[0]  # already sorted by ann
            print(f"  {lb:8d} | {best['name']:30s} | {best['ann']:+7.1f}% | {best['wr']:4.1f}% | "
                  f"{best['n']:5d} | {best['dd']:5.1f}% | {best['pf']:4.2f} | {best['sharpe']:6.2f}")

    # Lookback aggregate stats
    print(f"\n  Lookback aggregate (all configs per lookback):")
    print(f"  {'Lookback':>8s} | {'#Configs':>8s} | {'Avg Ann':>8s} | {'Med Ann':>8s} | "
          f"{'Best Ann':>8s} | {'%Positive':>9s} | {'Avg Sharpe':>10s}")
    print(f"  {'-' * 80}")

    for lb in LOOKBACKS:
        lb_results = [r for r in results if f'_LB{lb}_' in r['name']]
        if lb_results:
            anns = [r['ann'] for r in lb_results]
            sharpe_vals = [r['sharpe'] for r in lb_results]
            n_pos = sum(1 for a in anns if a > 0)
            print(f"  {lb:8d} | {len(lb_results):8d} | {np.mean(anns):+7.1f}% | "
                  f"{np.median(anns):+7.1f}% | {max(anns):+7.1f}% | "
                  f"{n_pos}/{len(anns)} ({n_pos/len(anns)*100:.0f}%) | "
                  f"{np.mean(sharpe_vals):9.2f}")

    # ================================================================
    # Z-THRESHOLD COMPARISON
    # ================================================================
    print(f"\n{'=' * 150}")
    print(f"  Z-THRESHOLD COMPARISON (best config per threshold)")
    print(f"{'=' * 150}")
    print(f"  {'Z-Thresh':>8s} | {'Best Config':30s} | {'Ann':>8s} | {'WR':>5s} | {'N':>5s} | "
          f"{'DD':>6s} | {'PF':>5s} | {'Sharpe':>7s}")
    print(f"  {'-' * 100}")

    for zt in Z_THRESHOLDS:
        zt_results = [r for r in results if f'_Z{zt:.1f}_' in r['name']]
        if zt_results:
            best = zt_results[0]
            print(f"  {zt:8.1f} | {best['name']:30s} | {best['ann']:+7.1f}% | {best['wr']:4.1f}% | "
                  f"{best['n']:5d} | {best['dd']:5.1f}% | {best['pf']:4.2f} | {best['sharpe']:6.2f}")

    # ================================================================
    # EXIT_Z COMPARISON
    # ================================================================
    print(f"\n{'=' * 150}")
    print(f"  EXIT_Z COMPARISON (best config per exit_z)")
    print(f"{'=' * 150}")
    print(f"  {'Exit_Z':>7s} | {'Best Config':30s} | {'Ann':>8s} | {'WR':>5s} | {'N':>5s} | "
          f"{'DD':>6s} | {'PF':>5s} | {'Sharpe':>7s}")
    print(f"  {'-' * 100}")

    for ez in EXIT_Z:
        ez_results = [r for r in results if f'_EZ{ez}_' in r['name']]
        if ez_results:
            best = ez_results[0]
            print(f"  {ez:7.1f} | {best['name']:30s} | {best['ann']:+7.1f}% | {best['wr']:4.1f}% | "
                  f"{best['n']:5d} | {best['dd']:5.1f}% | {best['pf']:4.2f} | {best['sharpe']:6.2f}")

    # ================================================================
    # HOLD_MAX COMPARISON
    # ================================================================
    print(f"\n{'=' * 150}")
    print(f"  HOLD_MAX COMPARISON (best config per hold)")
    print(f"{'=' * 150}")
    print(f"  {'Hold':>4s} | {'Best Config':30s} | {'Ann':>8s} | {'WR':>5s} | {'N':>5s} | "
          f"{'DD':>6s} | {'PF':>5s} | {'Sharpe':>7s}")
    print(f"  {'-' * 100}")

    for hm in HOLD_MAX:
        hm_results = [r for r in results if f'_H{hm}_' in r['name']]
        if hm_results:
            best = hm_results[0]
            print(f"  {hm:4d} | {best['name']:30s} | {best['ann']:+7.1f}% | {best['wr']:4.1f}% | "
                  f"{best['n']:5d} | {best['dd']:5.1f}% | {best['pf']:4.2f} | {best['sharpe']:6.2f}")

    # ================================================================
    # MAX_PAIRS COMPARISON
    # ================================================================
    print(f"\n{'=' * 150}")
    print(f"  MAX_PAIRS COMPARISON (best config per max_pairs)")
    print(f"{'=' * 150}")
    print(f"  {'MP':>2s} | {'Best Config':30s} | {'Ann':>8s} | {'WR':>5s} | {'N':>5s} | "
          f"{'DD':>6s} | {'PF':>5s} | {'Sharpe':>7s}")
    print(f"  {'-' * 100}")

    for mp in MAX_PAIRS:
        mp_results = [r for r in results if r['name'].endswith(f'_MP{mp}')]
        if mp_results:
            best = mp_results[0]
            print(f"  {mp:2d} | {best['name']:30s} | {best['ann']:+7.1f}% | {best['wr']:4.1f}% | "
                  f"{best['n']:5d} | {best['dd']:5.1f}% | {best['pf']:4.2f} | {best['sharpe']:6.2f}")

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
    # WALK-FORWARD FOR TOP 20 CONFIGS
    # ================================================================
    top20_for_wf = results[:20]

    print(f"\n{'=' * 150}")
    print(f"  WALK-FORWARD VALIDATION (Top 20 configs, test years: {WF_YEARS})")
    print(f"{'=' * 150}")

    wf_all = []  # (config_name, test_year, result)

    for rank, cfg in enumerate(top20_for_wf):
        name = cfg['name']
        print(f"\n  [{rank+1}] {name}  (full-period Ann={cfg['ann']:+.1f}%)")

        # Parse config name: LOG_LB{lb}_Z{zt}_H{hm}_EZ{ez}_MP{mp}
        parts = name.split('_')
        lb_val = int(parts[1][2:])
        zt_val = float(parts[2][1:])
        hm_val = int(parts[3][1:])
        ez_val = float(parts[4][2:])
        mp_val = int(parts[5][2:])

        for test_year in WF_YEARS:
            if test_year not in year_start_di:
                print(f"    Test {test_year}: year not in data, SKIP")
                continue

            r = run_backtest(
                lookback=lb_val, z_thresh=zt_val,
                hold_max=hm_val, exit_z=ez_val, max_pairs=mp_val,
                start_year=test_year, end_year=test_year,
                config_name=f"WF_{test_year}_{name}",
            )
            if r is not None:
                wf_all.append((name, test_year, r))
                print(f"    Test {test_year}: Ann={r['ann']:+7.1f}%  WR={r['wr']:5.1f}%  "
                      f"N={r['n']:4d}  DD={r['dd']:5.1f}%  PF={r['pf']:4.2f}  "
                      f"Sharpe={r['sharpe']:6.2f}")
            else:
                print(f"    Test {test_year}: insufficient trades")

    # ================================================================
    # TOP 10 WALK-FORWARD AGGREGATE
    # ================================================================
    print(f"\n{'=' * 150}")
    print(f"  TOP 10 WALK-FORWARD CONFIGS (average across test years)")
    print(f"{'=' * 150}")

    # Compute average OOS performance per config
    wf_by_config = {}
    for name, test_year, r in wf_all:
        if name not in wf_by_config:
            wf_by_config[name] = []
        wf_by_config[name].append((test_year, r))

    wf_avg = []
    for name, year_results in wf_by_config.items():
        anns = [r['ann'] for _, r in year_results]
        wrs = [r['wr'] for _, r in year_results]
        ns = [r['n'] for _, r in year_results]
        dds = [r['dd'] for _, r in year_results]
        pfs = [r['pf'] for _, r in year_results]
        sharpe_vals = [r['sharpe'] for _, r in year_results]
        n_positive = sum(1 for a in anns if a > 0)

        wf_avg.append({
            'name': name,
            'avg_ann': np.mean(anns),
            'med_ann': np.median(anns),
            'avg_wr': np.mean(wrs),
            'avg_n': np.mean(ns),
            'avg_dd': np.mean(dds),
            'avg_pf': np.mean(pfs),
            'avg_sharpe': np.mean(sharpe_vals),
            'n_positive': n_positive,
            'n_years': len(year_results),
            'year_details': year_results,
        })

    wf_avg.sort(key=lambda x: -x['avg_ann'])

    print(f"  {'#':>2s} | {'Config':30s} | {'Avg Ann':>8s} | {'Med Ann':>8s} | {'Avg WR':>6s} | "
          f"{'Avg N':>6s} | {'Avg DD':>7s} | {'Avg PF':>6s} | {'Avg Sharpe':>10s} | "
          f"{'Pos/Yr':>7s}")
    print(f"  {'-' * 130}")

    for i, w in enumerate(wf_avg[:10]):
        print(f"  {i+1:2d} | {w['name']:30s} | {w['avg_ann']:+7.1f}% | {w['med_ann']:+7.1f}% | "
              f"{w['avg_wr']:5.1f}% | {w['avg_n']:5.0f} | {w['avg_dd']:6.1f}% | "
              f"{w['avg_pf']:5.2f} | {w['avg_sharpe']:9.2f} | "
              f"{w['n_positive']}/{w['n_years']}")

    # ================================================================
    # WALK-FORWARD YEAR-BY-YEAR DETAIL (top 5 WF configs)
    # ================================================================
    print(f"\n{'=' * 150}")
    print(f"  WALK-FORWARD YEAR-BY-YEAR DETAIL (Top 5 WF configs)")
    print(f"{'=' * 150}")

    for i, w in enumerate(wf_avg[:5]):
        print(f"\n  [{i+1}] {w['name']}:")
        print(f"  {'Year':>6s} | {'Ann':>8s} | {'WR':>5s} | {'N':>5s} | "
              f"{'DD':>6s} | {'PF':>5s} | {'Sharpe':>7s} | {'AvgW':>6s} | {'AvgL':>6s}")
        print(f"  {'-' * 80}")
        for test_year, r in sorted(w['year_details'], key=lambda x: x[0]):
            print(f"  {test_year:6d} | {r['ann']:+7.1f}% | {r['wr']:4.1f}% | "
                  f"{r['n']:5d} | {r['dd']:5.1f}% | {r['pf']:4.2f} | {r['sharpe']:6.2f} | "
                  f"{r['avg_win']:+5.2f}% | {r['avg_loss']:5.2f}%")

    # ================================================================
    # FULL-PERIOD vs WALK-FORWARD CORRELATION
    # ================================================================
    if wf_avg:
        print(f"\n{'=' * 150}")
        print(f"  OVERFITTING CHECK: Full-Period vs Walk-Forward Correlation")
        print(f"{'=' * 150}")

        full_anns = []
        wf_anns = []
        # Match full-period config name to WF average
        for w in wf_avg:
            name = w['name']
            # Find matching full-period result
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

    # ================================================================
    # PER-PAIR WALK-FORWARD STATS (for #1 WF config)
    # ================================================================
    if wf_avg:
        best_wf = wf_avg[0]
        best_wf_name = best_wf['name']

        # Get the full-period result for the best WF config
        full_r = next((r for r in results if r['name'] == best_wf_name), None)
        if full_r and full_r['pair_stats']:
            print(f"\n{'=' * 150}")
            print(f"  PER-PAIR STATS for #1 WF Config: {best_wf_name}")
            print(f"  Full-period: Ann={full_r['ann']:+.1f}%  WF Avg Ann={best_wf['avg_ann']:+.1f}%  "
                  f"Pos/Yr={best_wf['n_positive']}/{best_wf['n_years']}")
            print(f"{'=' * 150}")
            print(f"  {'Pair':25s} | {'N':>5s} | {'WR':>5s} | {'Abs PnL':>12s}")
            print(f"  {'-' * 60}")

            for p in sorted(full_r['pair_stats'].keys(),
                            key=lambda x: -full_r['pair_stats'][x]['pnl']):
                ps = full_r['pair_stats'][p]
                wr_p = ps['w'] / max(ps['n'], 1) * 100
                print(f"  {p:25s} | {ps['n']:5d} | {wr_p:4.1f}% | {ps['pnl']:+11.0f}")

    # ================================================================
    # INTERACTION HEATMAP: LOOKBACK x Z_THRESHOLD (best ann per cell)
    # ================================================================
    print(f"\n{'=' * 150}")
    print(f"  INTERACTION: LOOKBACK x Z_THRESHOLD (best Ann per cell, full period)")
    print(f"{'=' * 150}")

    header = f"  {'LB \\ Z':>7s} |"
    for zt in Z_THRESHOLDS:
        header += f" {'Z='+str(zt):>10s} |"
    print(header)
    print(f"  {'-' * (9 + 13 * len(Z_THRESHOLDS))}")

    for lb in LOOKBACKS:
        row = f"  {lb:7d} |"
        for zt in Z_THRESHOLDS:
            cell_results = [r for r in results
                            if f'_LB{lb}_' in r['name'] and f'_Z{zt:.1f}_' in r['name']]
            if cell_results:
                best = cell_results[0]
                row += f" {best['ann']:+9.1f}% |"
            else:
                row += f" {'N/A':>10s} |"
        print(row)

    # ================================================================
    # INTERACTION HEATMAP: LOOKBACK x EXIT_Z
    # ================================================================
    print(f"\n{'=' * 150}")
    print(f"  INTERACTION: LOOKBACK x EXIT_Z (best Ann per cell, full period)")
    print(f"{'=' * 150}")

    header = f"  {'LB \\ EZ':>7s} |"
    for ez in EXIT_Z:
        header += f" {'EZ='+str(ez):>10s} |"
    print(header)
    print(f"  {'-' * (9 + 13 * len(EXIT_Z))}")

    for lb in LOOKBACKS:
        row = f"  {lb:7d} |"
        for ez in EXIT_Z:
            cell_results = [r for r in results
                            if f'_LB{lb}_' in r['name'] and f'_EZ{ez}_' in r['name']]
            if cell_results:
                best = cell_results[0]
                row += f" {best['ann']:+9.1f}% |"
            else:
                row += f" {'N/A':>10s} |"
        print(row)

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
              f"Pos/Yr={wf_avg[0]['n_positive']}/{wf_avg[0]['n_years']}")

        # Consistency check
        n_all_positive = sum(1 for w in wf_avg[:10] if w['n_positive'] == w['n_years'])
        print(f"\n  Of top 10 WF configs, {n_all_positive} are positive in ALL test years")

        # Best single year performance
        all_test_anns = [(name, ty, r['ann']) for name, ty, r in wf_all]
        best_single = max(all_test_anns, key=lambda x: x[2])
        worst_single = min(all_test_anns, key=lambda x: x[2])
        print(f"  Best single-year OOS:  {best_single[1]} = {best_single[2]:+.1f}% ({best_single[0][:30]})")
        print(f"  Worst single-year OOS: {worst_single[1]} = {worst_single[2]:+.1f}% ({worst_single[0][:30]})")

        # Overall WF positive rate
        all_wf_anns = [r['ann'] for _, _, r in wf_all]
        n_pos_wf = sum(1 for a in all_wf_anns if a > 0)
        print(f"  Overall WF positive rate: {n_pos_wf}/{len(all_wf_anns)} "
              f"({n_pos_wf/len(all_wf_anns)*100:.0f}%)")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 150)


if __name__ == '__main__':
    main()
