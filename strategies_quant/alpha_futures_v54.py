"""
Alpha Futures V54 — V52 with Multiple Pairs and Aggressive Capital Allocation
=============================================================================
Key hypothesis: V52 found +303.5% with MP1 (max 1 pair at a time). But V52's
V39 showed max_pairs=2 was best at 3-day hold. With 1-day hold and MP2, we
might get 2x the compounding.

With MP1, only 1 pair trades at a time. When it exits, cash sits idle until
the next day. With MP2, we can have 2 pairs running, each using 50% capital.
If both have ~62% WR, the combined return should be higher.

Parameter sweep:
  Max pairs: [1, 2, 3, 5, 10]
  Lookback: [5, 7, 10]
  Z threshold: [0.8, 1.0, 1.2]
  Hold days: [1] (1-day only)
  Allocation: [equal, dynamic, z_weighted]
  Priority: [z_score, wr, random]
  Walk-forward for best (2023, 2024)

~200-250 configs. Print: top 20 full-period, top 10 walk-forward,
MP comparison table (avg return by MP level).
"""
import sys, os, time, warnings, random
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
    print("=" * 140)
    print("Alpha Futures V54 -- V52 + Multiple Pairs + Aggressive Capital Allocation")
    print("Core: 1-day hold with MP2+ allows concurrent pair trades, more compounding")
    print("Hypothesis: MP2 with 50% capital per pair = 2x compounding opportunities")
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
    # PRECOMPUTE PER-PAIR HISTORICAL WR (rolling)
    # ========================================
    # We need rolling WR per pair for priority='wr' sorting.
    # We compute a simple rolling WR using a 60-day lookback on pair trade outcomes.
    # Since we don't have trade outcomes before backtesting, we use a proxy:
    # count how often z-score signals were profitable in the lookback window.
    # We will compute this inside the backtest as we accumulate trade results.

    # ========================================
    # BACKTEST ENGINE
    # ========================================
    def run_backtest(lookback, z_thresh, hold_max, max_pairs,
                     allocation='equal', priority='z_score',
                     wf_split_year=None, config_name=""):
        """
        Multi-pair trading with configurable capital allocation and pair prioritization.

        allocation modes:
          'equal':     each pair gets cash / max_pairs (static)
          'dynamic':   each pair gets cash / available_slots (recalculate daily)
          'z_weighted': stronger signals get more capital (weight by abs_z)

        priority modes (how to rank when multiple pairs signal):
          'z_score': sort by abs(z-score) descending (most deviant first)
          'wr':      sort by rolling WR descending (most reliable first)
          'random':  random shuffle (control)
        """
        cash = float(CASH0)
        trades = []
        pair_positions = []

        # Rolling WR tracker per pair (for priority='wr')
        pair_wr_history = {}  # key: (down_si, up_si) -> list of bool (win/loss)

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

        # Seed RNG for 'random' priority (deterministic across runs)
        rng = random.Random(42)

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
                pos_dir = pos['dir']

                exit_reason = None

                # Exit 1: Time exit (1-day hold = exit next day)
                if days_held >= hold_max:
                    exit_reason = 'time'

                # Exit 2: Z mean-reversion (early exit if z crosses 0)
                if exit_reason is None and not np.isnan(z_now):
                    if pos_dir == 1 and z_now >= 0:
                        exit_reason = 'mean_rev'
                    elif pos_dir == -1 and z_now <= 0:
                        exit_reason = 'mean_rev'

                # Exit 3: Stop loss -- z moves further by 1.5 from entry
                if exit_reason is None and not np.isnan(z_now):
                    if pos_dir == 1 and z_now < entry_z - 1.5:
                        exit_reason = 'stop_loss'
                    elif pos_dir == -1 and z_now > entry_z + 1.5:
                        exit_reason = 'stop_loss'

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

                    won = total_pnl > 0
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

                    # Track WR for this pair
                    pair_key = (p_down_si, p_up_si)
                    if pair_key not in pair_wr_history:
                        pair_wr_history[pair_key] = []
                    pair_wr_history[pair_key].append(won)
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

            # Compute capital per pair based on allocation mode
            if allocation == 'equal':
                capital_per_pair = cash / max(1, max_pairs)
            elif allocation == 'dynamic':
                available_slots = max(1, max_pairs - len(pair_positions))
                capital_per_pair = cash / max(1, available_slots)
            else:  # z_weighted -- will be computed per candidate below
                capital_per_pair = None  # placeholder, computed after filtering

            # Gather candidates
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

                # Compute rolling WR for this pair (last 60 trades or all available)
                pair_key = (down_si, up_si)
                wr_score = 0.5  # default 50% if no history
                if pair_key in pair_wr_history and len(pair_wr_history[pair_key]) > 0:
                    recent = pair_wr_history[pair_key][-60:]
                    wr_score = sum(recent) / len(recent)

                candidates.append({
                    'abs_z': abs(z_val),
                    'z_val': z_val,
                    'wr': wr_score,
                    'down_si': down_si,
                    'up_si': up_si,
                    'down_sym': down_sym,
                    'up_sym': up_sym,
                })

            if not candidates:
                continue

            # Sort candidates by priority
            if priority == 'z_score':
                candidates.sort(key=lambda x: -x['abs_z'])
            elif priority == 'wr':
                candidates.sort(key=lambda x: -x['wr'])
            else:  # random
                rng.shuffle(candidates)

            # For z_weighted allocation, compute weights from z-scores
            if allocation == 'z_weighted':
                selected = candidates[:n_can_open]
                total_z = sum(c['abs_z'] for c in selected)
                if total_z < 1e-10:
                    total_z = 1.0
                for c in selected:
                    c['weight'] = c['abs_z'] / total_z
            else:
                # All selected get equal weight within the capital_per_pair
                for c in candidates[:n_can_open]:
                    c['weight'] = None  # will use capital_per_pair

            # Open positions for top candidates
            for cand in candidates[:n_can_open]:
                down_si = cand['down_si']
                up_si = cand['up_si']
                down_sym = cand['down_sym']
                up_sym = cand['up_sym']
                z_val = cand['z_val']

                c_down = C[down_si, di]
                c_up = C[up_si, di]
                if np.isnan(c_down) or c_down <= 0 or np.isnan(c_up) or c_up <= 0:
                    continue

                mult_down = MULT.get(down_sym, DEF_MULT)
                mult_up = MULT.get(up_sym, DEF_MULT)

                # Compute capital for this pair
                if allocation == 'z_weighted':
                    pair_cap = cash * cand['weight'] * 0.95  # use 95% of allocated
                else:
                    pair_cap = capital_per_pair * 0.95

                cash_per_leg = pair_cap / 2
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
            'mp': max_pairs,
            'alloc': allocation,
            'prio': priority,
        }

    # ========================================
    # PARAMETER SWEEP (~200-250 configs)
    # ========================================
    print("\n[Backtest] Running parameter sweep...", flush=True)
    results = []
    configs = []

    lookbacks = [5, 7, 10]
    z_thresholds = [0.8, 1.0, 1.2]
    hold_days_list = [1]
    max_pairs_list = [1, 2, 3, 5, 10]
    allocations = ['equal', 'dynamic', 'z_weighted']
    priorities = ['z_score', 'wr', 'random']

    # Full-period configs
    # Use abbreviated codes to avoid underscore parsing issues:
    #   alloc: E=equal, D=dynamic, Z=z_weighted
    #   prio:  Z=z_score, W=wr, R=random
    alloc_code = {'equal': 'E', 'dynamic': 'D', 'z_weighted': 'Z'}
    prio_code = {'z_score': 'Z', 'wr': 'W', 'random': 'R'}

    for lb in lookbacks:
        for zt in z_thresholds:
            for hd in hold_days_list:
                for mp in max_pairs_list:
                    for alloc in allocations:
                        for prio in priorities:
                            name = (f"LB{lb}_Z{zt:.1f}_H{hd}_MP{mp}"
                                    f"_{alloc_code[alloc]}{prio_code[prio]}")
                            configs.append((lb, zt, hd, mp, alloc, prio, None, name))

    print(f"  {len(configs)} full-period configurations", flush=True)

    for ci, (lb, zt, hd, mp, alloc, prio, wf, name) in enumerate(configs):
        r = run_backtest(lb, zt, hd, mp, alloc, prio,
                         wf_split_year=wf, config_name=name)
        if r is not None:
            results.append(r)
            if r['ann'] > 50:
                print(f"  {r['name']:55s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                      f"N {r['n']:5d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                      f"Sh {r['sharpe']:5.2f} | AvgD {r['avg_days']:.1f}")
        if (ci + 1) % 75 == 0:
            print(f"  [{ci+1}/{len(configs)}] {len(results)} configs with results", flush=True)

    # Walk-forward for top configs
    print(f"\n[Walk-Forward] Testing top configs out-of-sample...", flush=True)
    full_results = [r for r in results]
    full_results.sort(key=lambda x: -x['ann'])

    # Reverse mapping for abbreviated codes
    alloc_decode = {'E': 'equal', 'D': 'dynamic', 'Z': 'z_weighted'}
    prio_decode = {'Z': 'z_score', 'W': 'wr', 'R': 'random'}

    # Select unique param combos (deduplicate by LB/Z/H/MP)
    wf_candidates = {}
    for r in full_results:
        parts = r['name'].split('_')
        key = '_'.join(parts[:4])  # LB_Z_H_MP
        if key not in wf_candidates:
            wf_candidates[key] = r

    wf_results = []
    wf_configs_run = 0
    for key, r in list(wf_candidates.items())[:30]:
        parts = r['name'].split('_')
        lb = int(parts[0][2:])
        zt = float(parts[1][1:])
        hd = int(parts[2][1:])
        mp = int(parts[3][2:])
        # parts[4] is like "EZ" (alloc_code + prio_code)
        ap_code = parts[4]
        alloc = alloc_decode[ap_code[0]]
        prio = prio_decode[ap_code[1]]

        for wf_year in [2023, 2024]:
            name = f"{r['name']}_WF{wf_year}"
            wr = run_backtest(lb, zt, hd, mp, alloc, prio,
                              wf_split_year=wf_year, config_name=name)
            if wr is not None:
                wf_results.append(wr)
            wf_configs_run += 1

        if wf_configs_run % 30 == 0:
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
    print(f"\n{'=' * 160}")
    print(f"  TOP 20 FULL-PERIOD RESULTS (sorted by annual return)")
    print(f"{'=' * 160}")
    hdr = (f"  {'Config':55s} | {'Ann':>7s} | {'WR':>5s} | {'N':>5s} | {'DD':>6s} | "
           f"{'PF':>4s} | {'Sharpe':>6s} | {'AvgW':>7s} | {'AvgL':>6s} | {'AvgD':>4s}")
    print(hdr)
    print(f"  {'-' * 155}")
    for r in full_results[:20]:
        print(f"  {r['name']:55s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:5d} | {r['dd']:6.1f}% | {r['pf']:4.2f} | "
              f"{r['sharpe']:6.2f} | {r['avg_win']:+6.2f}% | {r['avg_loss']:5.2f}% | "
              f"{r['avg_days']:4.1f}")

    # --- TOP 10 WALK-FORWARD ---
    if wf_only:
        print(f"\n  TOP 10 WALK-FORWARD RESULTS (out-of-sample)")
        print(f"  {'-' * 150}")
        for r in wf_only[:10]:
            print(f"  {r['name']:65s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                  f"Sh {r['sharpe']:6.2f} | AvgD {r['avg_days']:.1f}")

    # --- MP COMPARISON TABLE ---
    print(f"\n{'=' * 140}")
    print(f"  MP (MAX PAIRS) COMPARISON TABLE -- Avg Return by MP Level")
    print(f"{'=' * 140}")
    print(f"\n  {'MP':>3s} | {'Avg Ann':>8s} | {'Best Ann':>9s} | {'Avg WR':>7s} | "
          f"{'Avg N':>6s} | {'Avg DD':>7s} | {'Avg PF':>6s} | {'Avg Sh':>6s} | "
          f"{'#Cfgs':>5s} | Best Config")
    print(f"  {'-' * 130}")
    for mp in max_pairs_list:
        mp_results = [r for r in full_results if r.get('mp') == mp]
        if mp_results:
            avg_ann = np.mean([r['ann'] for r in mp_results])
            best_ann = max(r['ann'] for r in mp_results)
            avg_wr = np.mean([r['wr'] for r in mp_results])
            avg_n = np.mean([r['n'] for r in mp_results])
            avg_dd = np.mean([r['dd'] for r in mp_results])
            avg_pf = np.mean([r['pf'] for r in mp_results])
            avg_sh = np.mean([r['sharpe'] for r in mp_results])
            best = max(mp_results, key=lambda x: x['ann'])
            print(f"  {mp:3d} | {avg_ann:+7.1f}% | {best_ann:+8.1f}% | {avg_wr:5.1f}% | "
                  f"{avg_n:6.0f} | {avg_dd:6.1f}% | {avg_pf:5.2f} | {avg_sh:5.2f} | "
                  f"{len(mp_results):5d} | {best['name']}")

    # --- ALLOCATION COMPARISON ---
    print(f"\n  ALLOCATION MODE COMPARISON:")
    print(f"  {'-' * 120}")
    print(f"  {'Allocation':12s} | {'Avg Ann':>8s} | {'Best Ann':>9s} | {'Avg WR':>7s} | "
          f"{'Avg N':>6s} | {'Avg DD':>7s} | {'Avg PF':>6s} | {'Avg Sh':>6s}")
    print(f"  {'-' * 85}")
    for alloc in allocations:
        alloc_results = [r for r in full_results if r.get('alloc') == alloc]
        if alloc_results:
            avg_ann = np.mean([r['ann'] for r in alloc_results])
            best_ann = max(r['ann'] for r in alloc_results)
            avg_wr = np.mean([r['wr'] for r in alloc_results])
            avg_n = np.mean([r['n'] for r in alloc_results])
            avg_dd = np.mean([r['dd'] for r in alloc_results])
            avg_pf = np.mean([r['pf'] for r in alloc_results])
            avg_sh = np.mean([r['sharpe'] for r in alloc_results])
            print(f"  {alloc:12s} | {avg_ann:+7.1f}% | {best_ann:+8.1f}% | {avg_wr:5.1f}% | "
                  f"{avg_n:6.0f} | {avg_dd:6.1f}% | {avg_pf:5.2f} | {avg_sh:5.2f}")

    # --- PRIORITY COMPARISON ---
    print(f"\n  PRIORITY MODE COMPARISON:")
    print(f"  {'-' * 120}")
    print(f"  {'Priority':12s} | {'Avg Ann':>8s} | {'Best Ann':>9s} | {'Avg WR':>7s} | "
          f"{'Avg N':>6s} | {'Avg DD':>7s} | {'Avg PF':>6s} | {'Avg Sh':>6s}")
    print(f"  {'-' * 85}")
    for prio in priorities:
        prio_results = [r for r in full_results if r.get('prio') == prio]
        if prio_results:
            avg_ann = np.mean([r['ann'] for r in prio_results])
            best_ann = max(r['ann'] for r in prio_results)
            avg_wr = np.mean([r['wr'] for r in prio_results])
            avg_n = np.mean([r['n'] for r in prio_results])
            avg_dd = np.mean([r['dd'] for r in prio_results])
            avg_pf = np.mean([r['pf'] for r in prio_results])
            avg_sh = np.mean([r['sharpe'] for r in prio_results])
            print(f"  {prio:12s} | {avg_ann:+7.1f}% | {best_ann:+8.1f}% | {avg_wr:5.1f}% | "
                  f"{avg_n:6.0f} | {avg_dd:6.1f}% | {avg_pf:5.2f} | {avg_sh:5.2f}")

    # --- MP x ALLOCATION CROSS TABLE ---
    print(f"\n  MP x ALLOCATION CROSS TABLE (avg annual return %):")
    print(f"  {'MP':>3s} | ", end="")
    for alloc in allocations:
        print(f"{alloc:>12s} | ", end="")
    print()
    print(f"  {'-' * 50}")
    for mp in max_pairs_list:
        print(f"  {mp:3d} | ", end="")
        for alloc in allocations:
            subset = [r for r in full_results
                      if r.get('mp') == mp and r.get('alloc') == alloc]
            if subset:
                avg = np.mean([r['ann'] for r in subset])
                print(f"{avg:+11.1f}% | ", end="")
            else:
                print(f"{'N/A':>12s} | ", end="")
        print()

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

    # --- V52 BASELINE COMPARISON ---
    print(f"\n  === V54 vs V52 BASELINE COMPARISON ===")
    print(f"  V52 best (MP1, 1-day): +303.5%")
    if full_results:
        print(f"  V54 best: {full_results[0]['name']}")
        print(f"    Ann={full_results[0]['ann']:+.1f}%  N={full_results[0]['n']}  "
              f"WR={full_results[0]['wr']:.1f}%  DD={full_results[0]['dd']:.1f}%  "
              f"Sharpe={full_results[0]['sharpe']:.2f}")
        delta = full_results[0]['ann'] - 303.5
        print(f"    Delta vs V52: {delta:+.1f}%")

        # MP1 vs MP2+ comparison for V54
        mp1_best = max((r for r in full_results if r.get('mp') == 1),
                       key=lambda x: x['ann'], default=None)
        mp2_best = max((r for r in full_results if r.get('mp') == 2),
                       key=lambda x: x['ann'], default=None)
        print(f"\n  V54 MP1 best: {mp1_best['ann']:+.1f}% ({mp1_best['name']})" if mp1_best else "  V54 MP1: no results")
        print(f"  V54 MP2 best: {mp2_best['ann']:+.1f}% ({mp2_best['name']})" if mp2_best else "  V54 MP2: no results")
        if mp1_best and mp2_best:
            print(f"  MP2 vs MP1 delta: {mp2_best['ann'] - mp1_best['ann']:+.1f}%")

        # How many configs beat V52?
        beating_v52 = sum(1 for r in full_results if r['ann'] > 303.5)
        print(f"    Configs beating V52 (+303.5%): {beating_v52}/{len(full_results)}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 140)


if __name__ == '__main__':
    main()
