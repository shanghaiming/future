"""
V305: Adaptive Pair Trading on Chinese Futures
================================================
Ports V62's proven pair trading architecture to the futures data infrastructure.
V62 achieved 94.6% on stocks; this rebuilds it for futures with regime filtering.

Key features:
1. 3 spread modes × 5 lookbacks with adaptive selection every 40 days
2. 1-day hold (captures immediate mean reversion)
3. Cash-based P&L with contract multipliers and lot sizing
4. Z-score stop loss (1.5 sigma against entry)
5. Occupied commodity tracking (no overlapping pairs on same commodity)
6. V301 regime filter (only trade in favorable regimes)
7. Walk-forward validation (4-year train, 1-year test)
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_v301 import load_all_data, compute_factors, detect_regimes

# ============================================================
# CONSTANTS
# ============================================================
CASH0 = 1_000_000
COMM = 0.0003  # Commission rate (lower for pairs)

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

# Pair definitions (supply-chain related commodity pairs)
PAIRS_14 = [
    ('rbfi', 'ifi'),   # rebar / iron ore
    ('hcfi', 'ifi'),   # hot coil / iron ore
    ('hcfi', 'rbfi'),  # hot coil / rebar
    ('jfi', 'jmfi'),   # coke / coking coal
    ('mafi', 'scfi'),  # methanol / crude
    ('fufi', 'scfi'),  # fuel oil / crude
    ('bfi', 'scfi'),   # bitumen / crude
    ('mfi', 'afi'),    # meal / soybean
    ('yfi', 'afi'),    # soy oil / soybean
    ('pfi', 'yfi'),    # palm oil / soy oil
    ('ppfi', 'mafi'),  # PP / methanol
    ('vfi', 'mafi'),   # PVC / methanol
    ('egfi', 'mafi'),  # EG / methanol
    ('cfi', 'csfi'),   # corn / corn starch
]
PAIRS_16 = PAIRS_14 + [('jfi', 'ifi'), ('cufi', 'znfi')]
PAIRS_18 = PAIRS_16 + [('alfi', 'znfi'), ('mfi', 'yfi')]

# Spread modes
SPREAD_RAW = 'raw'
SPREAD_PCT = 'pct'
SPREAD_LOG = 'log'
ALL_MODES = [SPREAD_RAW, SPREAD_PCT, SPREAD_LOG]
ALL_LOOKBACKS = [5, 7, 10, 15, 20]

# Adaptive candidate combos (LOG-biased like V62)
CANDIDATE_LOG_BIAS = [
    (SPREAD_LOG, 10), (SPREAD_LOG, 15), (SPREAD_LOG, 20),
    (SPREAD_RAW, 10), (SPREAD_PCT, 10),
]


# ============================================================
# PRECOMPUTATION
# ============================================================
def precompute_spreads_and_zscores(C, NS, ND, pair_indices, min_train=60):
    """Precompute spreads and z-scores for all modes × lookbacks × pairs."""
    print("[V305] Precomputing spreads and z-scores...", flush=True)
    t0 = time.time()

    # Collect all unique pair index pairs
    all_pair_keys = set()
    for down_si, up_si, _, _ in pair_indices:
        all_pair_keys.add((down_si, up_si))

    z_scores = {m: {} for m in ALL_MODES}

    for down_si, up_si in all_pair_keys:
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
                    if len(valid) >= max(3, int(lb * 0.8)):
                        m_val = np.mean(valid)
                        s_val = np.std(valid, ddof=1)
                        if s_val > 1e-10:
                            z[di] = (spread[di] - m_val) / s_val
                z_scores[mode][key][lb] = z

    print(f"  Z-scores: {len(all_pair_keys)} pairs × {len(ALL_MODES)} modes × {len(ALL_LOOKBACKS)} lookbacks ({time.time()-t0:.1f}s)", flush=True)
    return z_scores


def precompute_hypothetical_returns(C, NS, ND, pair_indices, z_scores,
                                     all_zt=None):
    """Precompute per-combo daily returns for adaptive selection."""
    if all_zt is None:
        all_zt = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5]

    print("[V305] Precomputing hypothetical returns...", flush=True)
    t0 = time.time()

    global_combo_daily_return = {}
    min_train = 60

    for zt in all_zt:
        for mode in ALL_MODES:
            for lb in ALL_LOOKBACKS:
                combo_key = (mode, lb, zt)
                daily_ret = np.full(ND, np.nan)

                for di in range(min_train + 1, ND):
                    pair_rets = []
                    for down_si, up_si, down_sym, up_sym in pair_indices:
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
                        pair_rets.append(pnl_pct)

                    if pair_rets:
                        daily_ret[di] = np.mean(pair_rets)

                global_combo_daily_return[combo_key] = daily_ret

    print(f"  Hypothetical returns: {len(global_combo_daily_return)} combos ({time.time()-t0:.1f}s)", flush=True)
    return global_combo_daily_return


# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_pair_backtest(C, ND, dates, syms, pair_indices, z_scores,
                      global_combo_daily_return,
                      z_thresh=0.8, hold_max=1, exit_z=0.0, max_pairs=1,
                      eval_period=40, candidate_combos=None,
                      start_di=60, end_di=None,
                      regime=None, regime_filter=False,
                      config_name=""):
    """
    Pair trading backtest engine.
    regime_filter: if True, only enter pairs when regime is favorable (0 or 1)
    """
    if candidate_combos is None:
        candidate_combos = CANDIDATE_LOG_BIAS
    if end_di is None:
        end_di = ND

    cash = float(CASH0)
    trades = []
    pair_positions = []
    current_combo = candidate_combos[0]

    # Year boundaries for stats
    year_start_di = {}
    year_end_di = {}
    for di in range(ND):
        y = dates[di].year
        if y not in year_start_di:
            year_start_di[y] = di
        year_end_di[y] = di

    for di in range(start_di, end_di):
        year = dates[di].year

        # --- Adaptive evaluation every eval_period days ---
        if di > start_di:
            days_since = di - start_di
            if days_since % eval_period == 0 and days_since >= eval_period:
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

        # --- Manage existing positions ---
        new_positions = []
        for pos in pair_positions:
            p_down_si = pos['down_si']
            p_up_si = pos['up_si']
            z_arr = z_scores[pos['mode']].get((p_down_si, p_up_si), {}).get(pos['lb'])
            if z_arr is None:
                new_positions.append(pos)
                continue
            z_now = z_arr[di] if di < len(z_arr) else np.nan
            days_held = di - pos['entry_di']
            entry_z = pos['entry_z']
            pos_dir = pos['dir']

            exit_reason = None

            # Mean reversion exit: z crosses zero
            if not np.isnan(z_now):
                if pos_dir == 1 and z_now >= exit_z:
                    exit_reason = 'mean_rev'
                elif pos_dir == -1 and z_now <= -exit_z:
                    exit_reason = 'mean_rev'

            # Stop loss: z moved 1.5 sigma against
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

                entry_val = (pos['entry_down'] * mult_down * lots_down +
                             pos['entry_up'] * mult_up * lots_up)
                exit_val = (c_down * mult_down * lots_down +
                            c_up * mult_up * lots_up)
                cost = entry_val * COMM + exit_val * COMM
                total_pnl = pnl_down + pnl_up - cost
                pnl_pct = total_pnl / entry_val * 100 if entry_val > 0 else 0

                if pos_dir == 1:
                    cash_return = c_down * mult_down * lots_down - c_up * mult_up * lots_up
                else:
                    cash_return = -c_down * mult_down * lots_down + c_up * mult_up * lots_up

                cash += pos['cash_invested'] + cash_return - exit_val * COMM

                trades.append({
                    'pnl_abs': total_pnl,
                    'pnl_pct': pnl_pct,
                    'days': days_held,
                    'di': di,
                    'year': year,
                    'pair': (pos['down_sym'], pos['up_sym']),
                    'dir': pos_dir,
                    'reason': exit_reason,
                    'mode': pos['mode'],
                    'lb': pos['lb'],
                })
            else:
                new_positions.append(pos)

        pair_positions = new_positions

        # --- Check occupied commodities ---
        occupied = set()
        for pos in pair_positions:
            occupied.add(pos['down_si'])
            occupied.add(pos['up_si'])

        # --- Open new positions ---
        n_can_open = max_pairs - len(pair_positions)
        if n_can_open <= 0:
            continue

        # Regime filter
        if regime_filter and regime is not None:
            r = regime[di] if di < len(regime) else 0
            if r == -1 or r == 2:  # choppy or volatile — skip
                continue

        use_mode, use_lb = current_combo

        candidates = []
        for down_si, up_si, down_sym, up_sym in pair_indices:
            if down_si in occupied or up_si in occupied:
                continue
            z_arr = z_scores[use_mode].get((down_si, up_si), {}).get(use_lb)
            if z_arr is None:
                continue
            z_val = z_arr[di] if di < len(z_arr) else np.nan
            if np.isnan(z_val) or abs(z_val) < z_thresh:
                continue
            candidates.append((abs(z_val), down_si, up_si, down_sym, up_sym, z_val))

        if not candidates:
            continue
        candidates.sort(key=lambda x: -x[0])

        opened = 0
        for _, down_si, up_si, down_sym, up_sym, z_val in candidates:
            if opened >= n_can_open:
                break
            if down_si in occupied or up_si in occupied:
                continue

            c_down = C[down_si, di]
            c_up = C[up_si, di]
            if np.isnan(c_down) or c_down <= 0 or np.isnan(c_up) or c_up <= 0:
                continue

            mult_down = MULT.get(down_sym, DEF_MULT)
            mult_up = MULT.get(up_sym, DEF_MULT)

            capital_for_pair = cash * 0.9 / max(1, max_pairs)
            cash_per_leg = capital_for_pair / 2

            lots_down = int(cash_per_leg / (c_down * mult_down * (1 + COMM)))
            lots_up = int(cash_per_leg / (c_up * mult_up * (1 + COMM)))
            if lots_down <= 0 or lots_up <= 0:
                continue

            cost_down = c_down * mult_down * lots_down * (1 + COMM)
            cost_up = c_up * mult_up * lots_up * (1 + COMM)
            total_cost = cost_down + cost_up
            if total_cost > cash:
                scale = cash * 0.9 / total_cost
                lots_down = max(1, int(lots_down * scale))
                lots_up = max(1, int(lots_up * scale))
                cost_down = c_down * mult_down * lots_down * (1 + COMM)
                cost_up = c_up * mult_up * lots_up * (1 + COMM)
                total_cost = cost_down + cost_up
                if total_cost > cash:
                    continue

            # Direction: z > 0 means spread is wide → expect reversion → short spread
            if z_val > 0:
                pos_dir = -1  # short downstream, long upstream
            else:
                pos_dir = 1   # long downstream, short upstream

            cash -= total_cost
            pair_positions.append({
                'down_si': down_si, 'up_si': up_si,
                'down_sym': down_sym, 'up_sym': up_sym,
                'entry_down': c_down, 'entry_up': c_up,
                'lots_down': lots_down, 'lots_up': lots_up,
                'entry_di': di, 'entry_z': z_val,
                'dir': pos_dir, 'cash_invested': total_cost,
                'mode': use_mode, 'lb': use_lb,
            })
            occupied.add(down_si)
            occupied.add(up_si)
            opened += 1

    # Close remaining at end
    actual_end = min(end_di, ND) - 1
    for pos in pair_positions:
        c_down = C[pos['down_si'], actual_end]
        c_up = C[pos['up_si'], actual_end]
        if np.isnan(c_down) or c_down <= 0: c_down = pos['entry_down']
        if np.isnan(c_up) or c_up <= 0: c_up = pos['entry_up']

        mult_down = MULT.get(pos['down_sym'], DEF_MULT)
        mult_up = MULT.get(pos['up_sym'], DEF_MULT)
        lots_down, lots_up = pos['lots_down'], pos['lots_up']

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
        pnl_pct = total_pnl / entry_val * 100 if entry_val > 0 else 0

        cash += pos['cash_invested'] + ((-1)**(pos['dir'] == -1)) * (
            c_down * mult_down * lots_down - c_up * mult_up * lots_up
        ) - exit_val * COMM

        trades.append({
            'pnl_abs': total_pnl, 'pnl_pct': pnl_pct,
            'days': actual_end - pos['entry_di'], 'di': actual_end,
            'year': dates[actual_end].year,
            'pair': (pos['down_sym'], pos['up_sym']),
            'dir': pos['dir'], 'reason': 'end',
            'mode': pos['mode'], 'lb': pos['lb'],
        })

    if len(trades) < 3:
        return None

    # Compute stats
    equity = float(CASH0)
    peak = float(CASH0)
    max_dd = 0.0
    eq_curve = []
    for t in sorted(trades, key=lambda x: x['di']):
        equity += t['pnl_abs']
        eq_curve.append(equity)
        if equity > peak: peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100
            if dd > max_dd: max_dd = dd

    nw = sum(1 for t in trades if t['pnl_abs'] > 0)
    wr = nw / len(trades) * 100

    first_di = min(t['di'] for t in trades)
    last_di = max(t['di'] for t in trades)
    days_total = max((dates[last_di] - dates[first_di]).days, 365)
    yr = max(days_total / 365.25, 0.01)
    ann = ((cash / CASH0) ** (1 / yr) - 1) * 100

    trade_pnls = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
    if len(trade_pnls) > 1:
        rets = np.array(trade_pnls) / float(CASH0)
        sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
    else:
        sharpe = 0

    # Year breakdown
    year_stats = {}
    for t in trades:
        y = t['year']
        if y not in year_stats:
            year_stats[y] = {'n': 0, 'w': 0, 'pnl_abs': 0.0}
        year_stats[y]['n'] += 1
        if t['pnl_abs'] > 0: year_stats[y]['w'] += 1
        year_stats[y]['pnl_abs'] += t['pnl_abs']

    return {
        'name': config_name,
        'ann': round(ann, 1),
        'n': len(trades),
        'wr': round(wr, 1),
        'dd': round(max_dd, 1),
        'sharpe': round(sharpe, 2),
        'cash': round(cash, 0),
        'avg_days': round(np.mean([t['days'] for t in trades]), 1),
        'year_stats': year_stats,
        'trades': trades,
        'eq_curve': eq_curve,
    }


# ============================================================
# ANALYSIS
# ============================================================
def print_result(r, label=""):
    if r is None:
        print(f"  {label}: no trades")
        return
    print(f"  {label}: ann={r['ann']:+.1f}% n={r['n']} WR={r['wr']:.1f}% "
          f"DD={r['dd']:.1f}% Sh={r['sharpe']:.2f} avg_d={r['avg_days']:.1f}")
    for y in sorted(r['year_stats'].keys()):
        ys = r['year_stats'][y]
        wr_y = ys['w']/ys['n']*100 if ys['n'] > 0 else 0
        print(f"    {y}: {ys['n']} trades WR={wr_y:.1f}% pnl={ys['pnl_abs']:+,.0f}")


# ============================================================
# WALK-FORWARD
# ============================================================
def walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms, pair_indices,
                 z_thresh=0.8, hold_max=1, max_pairs=1, eval_period=40,
                 train_years=4, test_years=1,
                 regime_filter=False):
    years = sorted(set(d.year for d in dates))
    all_trades = []
    start_yr = years[0]

    print(f"\nWF: zt={z_thresh} hm={hold_max} mp={max_pairs} ep={eval_period} "
          f"train={train_years}y test={test_years}y rf={regime_filter}")

    while True:
        train_end = start_yr + train_years - 1
        test_yr = train_end + 1
        if test_yr > years[-1]:
            break

        train_m = np.array([d.year <= train_end for d in dates])
        test_m = np.array([d.year == test_yr for d in dates])
        t0 = max(0, np.where(train_m)[0][0] - 60)
        sl = slice(t0, np.where(test_m)[0][-1] + 1)

        C_s = C[:, sl]
        ND_s = C_s.shape[1]
        d_s = dates[sl]

        # Build pair indices for this slice
        sym_to_si = {syms[si]: si for si in range(NS)}
        pi_slice = []
        for down_sym, up_sym in [(p[2], p[3]) for p in pair_indices]:
            down_si = sym_to_si.get(down_sym, -1)
            up_si = sym_to_si.get(up_sym, -1)
            if down_si >= 0 and up_si >= 0:
                pi_slice.append((down_si, up_si, down_sym, up_sym))

        z_scores_s = precompute_spreads_and_zscores(C_s, NS, ND_s, pi_slice, min_train=60)
        hyp_ret_s = precompute_hypothetical_returns(C_s, NS, ND_s, pi_slice, z_scores_s,
                                                     all_zt=[z_thresh])

        # Regime for this slice
        regime_s = None
        if regime_filter:
            O_s = O[:, sl]; H_s = H[:, sl]; L_s = L[:, sl]
            V_s = V[:, sl]; OI_s = OI[:, sl]
            F = compute_factors(C_s, O_s, H_s, L_s, V_s, OI_s, NS, ND_s)
            regime_s = detect_regimes(F, NS, ND_s)

        test_start_di = np.where(np.array([d.year == test_yr for d in d_s]))[0][0]

        r = run_pair_backtest(C_s, ND_s, d_s, syms, pi_slice, z_scores_s, hyp_ret_s,
                              z_thresh=z_thresh, hold_max=hold_max,
                              max_pairs=max_pairs, eval_period=eval_period,
                              start_di=test_start_di,
                              regime=regime_s, regime_filter=regime_filter,
                              config_name=f"WF-{test_yr}")

        if r and r['trades']:
            yr_trades = [t for t in r['trades'] if t['year'] == test_yr]
            if yr_trades:
                nw = sum(1 for t in yr_trades if t['pnl_abs'] > 0)
                wr = nw / len(yr_trades) * 100
                avg_pnl = np.mean([t['pnl_pct'] for t in yr_trades])
                print(f"  {test_yr}: {len(yr_trades)} trades WR={wr:.1f}% avg={avg_pnl:+.3f}%",
                      flush=True)
                all_trades.extend(yr_trades)
            else:
                print(f"  {test_yr}: no trades in test year", flush=True)
        else:
            print(f"  {test_yr}: no trades", flush=True)

        start_yr += 1

    if all_trades:
        nw = sum(1 for t in all_trades if t['pnl_abs'] > 0)
        wr = nw / len(all_trades) * 100
        pnls = [t['pnl_pct'] for t in all_trades]
        avg = np.mean(pnls)
        cum = np.prod([1 + p/100 for p in pnls]) - 1
        print(f"\n  WF Total: {len(all_trades)} trades WR={wr:.1f}% avg={avg:+.3f}% cum={cum*100:+.1f}%")

        # Year breakdown
        yr_break = {}
        for t in all_trades:
            y = t['year']
            if y not in yr_break: yr_break[y] = {'n':0,'w':0,'pnl':[]}
            yr_break[y]['n'] += 1
            if t['pnl_abs'] > 0: yr_break[y]['w'] += 1
            yr_break[y]['pnl'].append(t['pnl_pct'])
        for y in sorted(yr_break.keys()):
            ys = yr_break[y]
            wr_y = ys['w']/ys['n']*100
            cum_y = np.prod([1+p/100 for p in ys['pnl']])-1
            print(f"    {y}: {ys['n']} trades WR={wr_y:.1f}% cum={cum_y*100:+.1f}%")

    return all_trades


# ============================================================
# MAIN
# ============================================================
def main():
    t_start = time.time()
    print("=" * 60)
    print("  V305: ADAPTIVE PAIR TRADING ON CHINESE FUTURES")
    print("=" * 60)

    # Load data
    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    sym_to_si = {syms[si]: si for si in range(NS)}

    # Build pair indices
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

    pi_14 = build_pair_indices(PAIRS_14)
    pi_16 = build_pair_indices(PAIRS_16)
    pi_18 = build_pair_indices(PAIRS_18)
    print(f"  Pairs: P14={len(pi_14)}, P16={len(pi_16)}, P18={len(pi_18)}")

    # Precompute
    z_scores = precompute_spreads_and_zscores(C, NS, ND, pi_18)
    hyp_ret = precompute_hypothetical_returns(C, NS, ND, pi_18, z_scores)

    # Regime detection for filter
    print("[V305] Computing regime...", flush=True)
    F = compute_factors(C, O, H, L, V, OI, NS, ND)
    regime = detect_regimes(F, NS, ND)

    # ================================================================
    # PARAMETER SWEEP (full backtest)
    # ================================================================
    print("\n" + "=" * 60)
    print("  PARAMETER SWEEP (Full Backtest)")
    print("=" * 60)

    results = []
    for zt in [0.5, 0.8, 1.0, 1.2]:
        for hm in [1, 2, 3]:
            for mp in [1, 2]:
                for pi_set, pi_name in [(pi_14, 'P14'), (pi_16, 'P16'), (pi_18, 'P18')]:
                    for rf in [False, True]:
                        r = run_pair_backtest(C, ND, dates, syms, pi_set, z_scores, hyp_ret,
                                              z_thresh=zt, hold_max=hm, max_pairs=mp,
                                              eval_period=40,
                                              regime=regime, regime_filter=rf,
                                              config_name=f"zt{zt}_hm{hm}_mp{mp}_{pi_name}_rf{rf}")
                        if r:
                            results.append(r)

    results.sort(key=lambda x: -x['sharpe'])
    print(f"\n{'Config':<35} {'Ann':>7} {'N':>5} {'WR':>5} {'DD':>6} {'Sh':>5}")
    print("-" * 70)
    for r in results[:20]:
        print(f"  {r['name']:<33} {r['ann']:>+7.1f} {r['n']:>5} {r['wr']:>5.1f} "
              f"{r['dd']:>6.1f} {r['sharpe']:>5.2f}")

    # ================================================================
    # WALK-FORWARD for top configs
    # ================================================================
    print("\n" + "=" * 60)
    print("  WALK-FORWARD VALIDATION")
    print("=" * 60)

    # Pick top configs by Sharpe for WF
    seen = set()
    wf_count = 0
    for r in results:
        if wf_count >= 5:
            break
        parts = r['name'].split('_')
        # Parse config
        zt = float(parts[0][2:])
        hm = int(parts[1][2:])
        mp = int(parts[2][2:])
        pi_name = parts[3]
        rf = parts[4][2:] == 'True'
        pi_set = pi_14 if pi_name == 'P14' else (pi_16 if pi_name == 'P16' else pi_18)

        key = (zt, hm, mp, pi_name, rf)
        if key in seen:
            continue
        seen.add(key)

        print(f"\n--- WF: {r['name']} (IS Sharpe={r['sharpe']:.2f}) ---")
        walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms, pi_set,
                     z_thresh=zt, hold_max=hm, max_pairs=mp, eval_period=40,
                     train_years=4, test_years=1, regime_filter=rf)
        wf_count += 1

    # ================================================================
    # ADDITIONAL: Multi-pair concurrent + higher hold
    # ================================================================
    print("\n" + "=" * 60)
    print("  ADDITIONAL: Multi-pair + longer hold")
    print("=" * 60)
    for mp in [2, 3, 5]:
        for hm in [1, 2]:
            for zt in [0.8, 1.0]:
                r = run_pair_backtest(C, ND, dates, syms, pi_18, z_scores, hyp_ret,
                                      z_thresh=zt, hold_max=hm, max_pairs=mp,
                                      eval_period=40,
                                      regime=regime, regime_filter=True,
                                      config_name=f"multi_mp{mp}_hm{hm}_zt{zt}_rf")
                if r:
                    print_result(r, f"mp={mp} hm={hm} zt={zt} rf")

    print(f"\n[V305] Done. Total: {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
