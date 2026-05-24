"""
Alpha Futures V41 — Deep Optimization of Supply Chain Pair Trading
==================================================================
Based on v39 best config: LB10_Z1.5_H3_MP2 (+188.1% annual)
Goal: Push to 400%+ annual via 7 optimizations:

1. Dynamic Z-Score Threshold (volatility-adaptive)
2. Asymmetric Entry/Exit (let winners run)
3. Weighted Pairs (Sharpe/WR-based allocation)
4. Max Pairs + Capital Allocation (equal vs inverse-vol)
5. Hold Period Refinement (fixed, adaptive, signal-based)
6. More Pairs (20 pairs, +7 new same-group pairs)
7. OI Confirmation (institutional conviction filter)

~300 configurations tested. Walk-forward for best configs.
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

# Extended pair set (20 pairs: 13 original + 7 new)
PAIRS = [
    # Original v39 pairs
    ('rbfi', 'ifi'),   # rebar vs iron ore
    ('hcfi', 'ifi'),   # hot coil vs iron ore
    ('hcfi', 'rbfi'),  # hot coil vs rebar
    ('jfi', 'jmfi'),   # coke vs coal
    ('mafi', 'scfi'),  # methanol vs crude
    ('fufi', 'scfi'),  # fuel oil vs crude
    ('bfi', 'scfi'),   # bitumen vs crude
    ('mfi', 'afi'),    # soybean meal vs soybean
    ('yfi', 'afi'),    # soybean oil vs soybean
    ('pfi', 'yfi'),    # palm oil vs soybean oil
    ('ppfi', 'mafi'),  # PP vs methanol
    ('vfi', 'mafi'),   # PVC vs methanol
    ('egfi', 'mafi'),  # EG vs methanol
    # NEW: same-group pairs
    ('jfi', 'ifi'),    # coke vs iron ore (both ferrous)
    ('agfi', 'aufi'),  # silver vs gold (both precious metals)
    ('cufi', 'znfi'),  # copper vs zinc (both base metals)
    ('alfi', 'znfi'),  # aluminum vs zinc
    ('mfi', 'yfi'),    # soybean meal vs soybean oil (both soy products)
    ('cfi', 'csfi'),   # corn vs corn starch
    ('srfi', 'cfi'),   # sugar vs corn (both agricultural)
]

PAIR_LABEL = {p: f"{p[0]}/{p[1]}" for p in PAIRS}


def main():
    t_start = time.time()
    print("=" * 130)
    print("Alpha Futures V41 — Deep Optimization of Supply Chain Pair Trading")
    print("7 optimizations over v39 base: dynamic Z, asymmetric exit, weighted pairs,")
    print("  capital allocation, hold refinement, more pairs, OI confirmation")
    print("=" * 130)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    has_oi = not np.all(np.isnan(OI))
    sym_to_si = {syms[si]: si for si in range(NS)}

    # Build pair index mapping
    pair_indices = []
    for down_sym, up_sym in PAIRS:
        down_si = sym_to_si.get(down_sym, -1)
        up_si = sym_to_si.get(up_sym, -1)
        if down_si >= 0 and up_si >= 0:
            pair_indices.append((down_si, up_si, down_sym, up_sym))
        else:
            print(f"  WARNING: pair ({down_sym}, {up_sym}) not in data "
                  f"(down_si={down_si}, up_si={up_si})")

    print(f"  {NS} commodities, {ND} days, {len(pair_indices)} active pairs, OI={'yes' if has_oi else 'no'}")

    # ========================================================
    # PRECOMPUTE SPREADS + ROLLING STATS
    # ========================================================
    print("\n[Signals] Computing spreads and rolling stats...", flush=True)
    t0 = time.time()

    MAX_LB = 60  # max lookback needed for rolling vol of spread_std
    spreads = {}
    pair_data = {}

    for down_si, up_si, down_sym, up_sym in pair_indices:
        sp = np.full(ND, np.nan)
        for di in range(ND):
            pd = C[down_si, di]
            pu = C[up_si, di]
            if not np.isnan(pd) and not np.isnan(pu):
                sp[di] = pd - pu
        spreads[(down_si, up_si)] = sp
        pair_data[(down_si, up_si)] = {'spread': sp}

    print(f"  Spreads computed ({time.time()-t0:.1f}s)", flush=True)

    # ========================================================
    # BACKTEST ENGINE
    # ========================================================
    def run_backtest(lookback, z_mode, z_thresh_base, exit_mode, exit_z,
                     hold_mode, hold_max, max_pairs, weight_mode, cap_alloc,
                     oi_filter, wf_split_year=None, config_name=""):
        """
        Pair trading backtest with all v41 optimizations.

        Parameters
        ----------
        lookback : int — rolling window for spread mean/std
        z_mode : str — 'fixed' or 'dynamic'
        z_thresh_base : float — base z threshold (used for fixed, center for dynamic)
        exit_mode : str — 'zero' (exit at z=0), 'asymmetric' (exit at exit_z)
        exit_z : float — if asymmetric, exit when z crosses this in profit direction
        hold_mode : str — 'fixed', 'adaptive', 'signal'
            fixed: exit after hold_max days
            adaptive: if |z| > 1.0 at hold_max, extend by 2 days (max once)
            signal: exit when z crosses 0 OR hold_max reached
        max_pairs : int — max concurrent pair positions
        weight_mode : str — 'equal', 'sharpe', 'wr'
        cap_alloc : str — 'equal' or 'inv_vol'
        oi_filter : bool — require OI confirmation on leading leg
        """
        cash = float(CASH0)
        trades = []
        pair_positions = []

        # Pre-compute per-pair: z-score arrays for all needed lookbacks
        # We'll compute on-the-fly per lookback
        pd_cache = {}
        for down_si, up_si, down_sym, up_sym in pair_indices:
            sp = spreads[(down_si, up_si)]
            sp_mean = np.full(ND, np.nan)
            sp_std = np.full(ND, np.nan)
            z_arr = np.full(ND, np.nan)

            # Rolling mean/std of spread
            for di in range(lookback, ND):
                window = sp[di - lookback:di]
                valid = window[~np.isnan(window)]
                if len(valid) >= lookback * 0.8:
                    sp_mean[di] = np.mean(valid)
                    sp_std[di] = np.std(valid, ddof=1)
                    if sp_std[di] > 1e-10:
                        z_arr[di] = (sp[di] - sp_mean[di]) / sp_std[di]

            # For dynamic z: compute rolling mean of sp_std over 60 days
            sp_std_rolling_mean = np.full(ND, np.nan)
            vol_window = 60
            for di in range(vol_window, ND):
                w = sp_std[di - vol_window:di]
                valid = w[~np.isnan(w)]
                if len(valid) >= vol_window * 0.5:
                    sp_std_rolling_mean[di] = np.mean(valid)

            # Dynamic threshold per day
            if z_mode == 'dynamic':
                dyn_thresh = np.full(ND, np.nan)
                for di in range(ND):
                    if np.isnan(sp_std[di]) or np.isnan(sp_std_rolling_mean[di]):
                        continue
                    if sp_std_rolling_mean[di] < 1e-10:
                        continue
                    vol_ratio = sp_std[di] / sp_std_rolling_mean[di]
                    if vol_ratio < 0.7:
                        dyn_thresh[di] = 1.0
                    elif vol_ratio > 1.3:
                        dyn_thresh[di] = 2.0
                    else:
                        dyn_thresh[di] = 1.5
            else:
                dyn_thresh = None

            # For inverse-vol capital allocation: store spread std
            # For weighted pairs: compute rolling 60-day Sharpe per pair
            # We'll do this lazily based on trade history

            # OI data for both legs
            oi_down = OI[down_si] if has_oi else None
            oi_up = OI[up_si] if has_oi else None

            pd_cache[(down_si, up_si)] = {
                'spread': sp,
                'mean': sp_mean,
                'std': sp_std,
                'z': z_arr,
                'dyn_thresh': dyn_thresh,
                'oi_down': oi_down,
                'oi_up': oi_up,
                'down_sym': down_sym,
                'up_sym': up_sym,
            }

        # Rolling pair quality (for weight_mode sharpe/wr)
        # Track rolling PnL per pair over last 60 trades
        pair_history = {}  # (down_si, up_si) -> list of (di, pnl_abs)

        for di in range(MIN_TRAIN, ND):
            year = dates[di].year
            if wf_split_year is not None and year < wf_split_year:
                continue

            # --- Manage existing positions ---
            new_positions = []
            for pos in pair_positions:
                p_down_si = pos['down_si']
                p_up_si = pos['up_si']
                pc = pd_cache[(p_down_si, p_up_si)]
                z_now = pc['z'][di]
                days_held = di - pos['entry_di']
                entry_z = pos['entry_z']
                pos_dir = pos['dir']
                extended = pos.get('extended', False)

                exit_reason = None

                # Exit 1: Mean reversion (z crosses exit threshold)
                if not np.isnan(z_now):
                    if exit_mode == 'asymmetric' and exit_z > 0:
                        # Exit when z crosses exit_z in the profit direction
                        if pos_dir == 1 and z_now <= exit_z:
                            exit_reason = 'mean_rev'
                        elif pos_dir == -1 and z_now >= -exit_z:
                            exit_reason = 'mean_rev'
                    else:
                        # Standard: exit at z=0
                        if pos_dir == 1 and z_now <= 0:
                            exit_reason = 'mean_rev'
                        elif pos_dir == -1 and z_now >= 0:
                            exit_reason = 'mean_rev'

                # Exit 2: Stop loss
                if exit_reason is None and not np.isnan(z_now):
                    if pos_dir == 1 and z_now < entry_z - 1.0:
                        exit_reason = 'stop_loss'
                    elif pos_dir == -1 and z_now > entry_z + 1.0:
                        exit_reason = 'stop_loss'

                # Exit 3: Time-based
                if exit_reason is None:
                    if hold_mode == 'fixed':
                        if days_held >= hold_max:
                            exit_reason = 'time'
                    elif hold_mode == 'adaptive':
                        if days_held >= hold_max and not extended:
                            # Check if z still > 1.0 from entry direction
                            if not np.isnan(z_now):
                                if pos_dir == 1 and z_now > 1.0:
                                    pos['extended'] = True
                                    pos['extend_di'] = di
                                elif pos_dir == -1 and z_now < -1.0:
                                    pos['extended'] = True
                                    pos['extend_di'] = di
                                else:
                                    exit_reason = 'time'
                            else:
                                exit_reason = 'time'
                        elif extended:
                            max_after_extend = hold_max + 2
                            if days_held >= max_after_extend:
                                exit_reason = 'time'
                    elif hold_mode == 'signal':
                        # signal mode: exit at z=0 already handled above
                        # Fall back to max hold
                        if days_held >= hold_max:
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

                    pair_key = (pos['down_si'], pos['up_si'])
                    if pair_key not in pair_history:
                        pair_history[pair_key] = []
                    pair_history[pair_key].append((di, total_pnl))

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

            # --- Open new positions ---
            occupied = set()
            for pos in pair_positions:
                occupied.add(pos['down_si'])
                occupied.add(pos['up_si'])

            n_can_open = max_pairs - len(pair_positions)
            if n_can_open <= 0:
                continue

            # Build candidate list with scoring
            candidates = []
            for down_si, up_si, down_sym, up_sym in pair_indices:
                if down_si in occupied or up_si in occupied:
                    continue
                pc = pd_cache[(down_si, up_si)]
                z_val = pc['z'][di]
                if np.isnan(z_val):
                    continue

                # Determine effective threshold
                if z_mode == 'dynamic':
                    if pc['dyn_thresh'] is not None and not np.isnan(pc['dyn_thresh'][di]):
                        eff_thresh = pc['dyn_thresh'][di]
                    else:
                        eff_thresh = z_thresh_base
                else:
                    eff_thresh = z_thresh_base

                if abs(z_val) < eff_thresh:
                    continue

                # OI filter
                if oi_filter and has_oi:
                    # Determine which leg is "leading" (the one we go long)
                    if z_val > 0:
                        # Short down, Long up -> leading leg = up
                        lead_oi = pc['oi_up']
                    else:
                        # Long down, Short up -> leading leg = down
                        lead_oi = pc['oi_down']

                    if lead_oi is not None:
                        oi_now = lead_oi[di]
                        oi_prev = lead_oi[di - 1] if di > 0 else np.nan
                        if np.isnan(oi_now) or np.isnan(oi_prev):
                            continue
                        if oi_now <= oi_prev:
                            # OI not rising on leading leg -> skip
                            continue

                # Compute pair weight for capital allocation
                pair_key = (down_si, up_si)
                pair_wt = 1.0  # default equal
                if weight_mode == 'sharpe':
                    hist = pair_history.get(pair_key, [])
                    recent = [p for d, p in hist if di - d < 60]
                    if len(recent) >= 5:
                        arr = np.array(recent)
                        mean_p = np.mean(arr)
                        std_p = np.std(arr)
                        if std_p > 0:
                            pair_wt = max(0.1, mean_p / std_p)  # Sharpe-like
                elif weight_mode == 'wr':
                    hist = pair_history.get(pair_key, [])
                    recent = [p for d, p in hist if di - d < 60]
                    if len(recent) >= 5:
                        pair_wt = max(0.1, sum(1 for p in recent if p > 0) / len(recent))

                # Volatility for inv_vol capital allocation
                sp_std_val = pc['std'][di] if not np.isnan(pc['std'][di]) else 1.0
                inv_vol = 1.0 / max(sp_std_val, 1e-10)

                candidates.append({
                    'z_abs': abs(z_val),
                    'z_val': z_val,
                    'down_si': down_si,
                    'up_si': up_si,
                    'down_sym': down_sym,
                    'up_sym': up_sym,
                    'pair_wt': pair_wt,
                    'inv_vol': inv_vol,
                })

            if not candidates:
                continue

            # Sort by |z| descending (strongest deviation first)
            candidates.sort(key=lambda x: -x['z_abs'])

            # Compute weight sums for allocation
            chosen = candidates[:n_can_open]

            # Capital allocation weights
            if cap_alloc == 'inv_vol':
                total_iv = sum(c['inv_vol'] * c['pair_wt'] for c in chosen)
                alloc_weights = [(c['inv_vol'] * c['pair_wt']) / total_iv for c in chosen] if total_iv > 0 else [1.0 / len(chosen)] * len(chosen)
            elif weight_mode in ('sharpe', 'wr'):
                total_w = sum(c['pair_wt'] for c in chosen)
                alloc_weights = [c['pair_wt'] / total_w for c in chosen] if total_w > 0 else [1.0 / len(chosen)] * len(chosen)
            else:
                alloc_weights = [1.0 / len(chosen)] * len(chosen)

            for ci_idx, cand in enumerate(chosen):
                down_si = cand['down_si']
                up_si = cand['up_si']
                down_sym = cand['down_sym']
                up_sym = cand['up_sym']
                z_val = cand['z_val']
                alloc = alloc_weights[ci_idx]

                c_down = C[down_si, di]
                c_up = C[up_si, di]
                if np.isnan(c_down) or c_down <= 0 or np.isnan(c_up) or c_up <= 0:
                    continue

                mult_down = MULT.get(down_sym, DEF_MULT)
                mult_up = MULT.get(up_sym, DEF_MULT)

                # Allocate capital proportional to weight
                cash_for_pair = cash * alloc
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
                    pos_dir = -1  # short down, long up
                else:
                    pos_dir = 1   # long down, short up

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
                    'extended': False,
                })

        # Close remaining at end
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

    # ========================================================
    # CONFIGURATION GENERATION
    # ========================================================
    print("\n[Backtest] Building ~300 configurations...", flush=True)

    configs = []

    # --- Phase 1: Baseline sweep (v39 params as anchor) ---
    # lookback=10, z=1.5, hold=3, max_pairs=2 was best in v39
    # Test variations around that anchor
    base_lbs = [10, 15, 20]
    base_zs = [1.0, 1.5, 2.0]
    base_hds = [3, 5]
    base_mps = [2, 3, 4]

    for lb in base_lbs:
        for zt in base_zs:
            for hd in base_hds:
                for mp in base_mps:
                    configs.append({
                        'lookback': lb, 'z_mode': 'fixed', 'z_thresh_base': zt,
                        'exit_mode': 'zero', 'exit_z': 0.0,
                        'hold_mode': 'fixed', 'hold_max': hd,
                        'max_pairs': mp, 'weight_mode': 'equal', 'cap_alloc': 'equal',
                        'oi_filter': False, 'wf_split_year': None,
                        'config_name': f"F_LB{lb}_Z{zt:.1f}_H{hd}_MP{mp}",
                    })

    # --- Phase 2: Dynamic Z threshold ---
    for lb in [10, 15, 20]:
        for zt in [1.5]:  # center threshold
            for hd in [3, 5]:
                for mp in [2, 3, 4]:
                    configs.append({
                        'lookback': lb, 'z_mode': 'dynamic', 'z_thresh_base': zt,
                        'exit_mode': 'zero', 'exit_z': 0.0,
                        'hold_mode': 'fixed', 'hold_max': hd,
                        'max_pairs': mp, 'weight_mode': 'equal', 'cap_alloc': 'equal',
                        'oi_filter': False, 'wf_split_year': None,
                        'config_name': f"DYN_LB{lb}_Z{zt:.1f}_H{hd}_MP{mp}",
                    })

    # --- Phase 3: Asymmetric exit ---
    for ez in [0.2, 0.3, 0.5]:
        for lb in [10, 15]:
            for hd in [3, 5]:
                for mp in [2, 3]:
                    configs.append({
                        'lookback': lb, 'z_mode': 'fixed', 'z_thresh_base': 1.5,
                        'exit_mode': 'asymmetric', 'exit_z': ez,
                        'hold_mode': 'fixed', 'hold_max': hd,
                        'max_pairs': mp, 'weight_mode': 'equal', 'cap_alloc': 'equal',
                        'oi_filter': False, 'wf_split_year': None,
                        'config_name': f"ASYM_EZ{ez:.1f}_LB{lb}_Z1.5_H{hd}_MP{mp}",
                    })

    # --- Phase 4: Dynamic Z + Asymmetric exit (combo) ---
    for ez in [0.2, 0.3]:
        for lb in [10, 15]:
            for hd in [3, 5]:
                for mp in [2, 3, 4]:
                    configs.append({
                        'lookback': lb, 'z_mode': 'dynamic', 'z_thresh_base': 1.5,
                        'exit_mode': 'asymmetric', 'exit_z': ez,
                        'hold_mode': 'fixed', 'hold_max': hd,
                        'max_pairs': mp, 'weight_mode': 'equal', 'cap_alloc': 'equal',
                        'oi_filter': False, 'wf_split_year': None,
                        'config_name': f"DASYM_EZ{ez:.1f}_LB{lb}_H{hd}_MP{mp}",
                    })

    # --- Phase 5: Adaptive / signal hold ---
    for hold_m in ['adaptive', 'signal']:
        for lb in [10, 15]:
            for zt in [1.5]:
                for mp in [2, 3, 4]:
                    configs.append({
                        'lookback': lb, 'z_mode': 'fixed', 'z_thresh_base': zt,
                        'exit_mode': 'zero', 'exit_z': 0.0,
                        'hold_mode': hold_m, 'hold_max': 3,
                        'max_pairs': mp, 'weight_mode': 'equal', 'cap_alloc': 'equal',
                        'oi_filter': False, 'wf_split_year': None,
                        'config_name': f"{hold_m[:4].upper()}_LB{lb}_Z{zt:.1f}_H3_MP{mp}",
                    })

    # --- Phase 6: Weighted pairs ---
    for wm in ['sharpe', 'wr']:
        for lb in [10, 15]:
            for mp in [2, 3, 4]:
                configs.append({
                    'lookback': lb, 'z_mode': 'dynamic', 'z_thresh_base': 1.5,
                    'exit_mode': 'asymmetric', 'exit_z': 0.3,
                    'hold_mode': 'fixed', 'hold_max': 3,
                    'max_pairs': mp, 'weight_mode': wm, 'cap_alloc': 'equal',
                    'oi_filter': False, 'wf_split_year': None,
                    'config_name': f"W-{wm[:3].upper()}_LB{lb}_H3_MP{mp}",
                })

    # --- Phase 7: Capital allocation ---
    for ca in ['inv_vol']:
        for lb in [10, 15]:
            for mp in [2, 3, 4]:
                configs.append({
                    'lookback': lb, 'z_mode': 'dynamic', 'z_thresh_base': 1.5,
                    'exit_mode': 'asymmetric', 'exit_z': 0.3,
                    'hold_mode': 'fixed', 'hold_max': 3,
                    'max_pairs': mp, 'weight_mode': 'equal', 'cap_alloc': ca,
                    'oi_filter': False, 'wf_split_year': None,
                    'config_name': f"IVOL_LB{lb}_H3_MP{mp}",
                })

    # --- Phase 8: OI filter ---
    for lb in [10, 15]:
        for mp in [2, 3, 4]:
            configs.append({
                'lookback': lb, 'z_mode': 'dynamic', 'z_thresh_base': 1.5,
                'exit_mode': 'asymmetric', 'exit_z': 0.3,
                'hold_mode': 'fixed', 'hold_max': 3,
                'max_pairs': mp, 'weight_mode': 'equal', 'cap_alloc': 'equal',
                'oi_filter': True, 'wf_split_year': None,
                'config_name': f"OI_LB{lb}_H3_MP{mp}",
            })

    # --- Phase 9: Max pairs sweep (5) with best settings ---
    for mp in [1, 5]:
        for lb in [10]:
            for wm in ['equal', 'sharpe']:
                configs.append({
                    'lookback': lb, 'z_mode': 'dynamic', 'z_thresh_base': 1.5,
                    'exit_mode': 'asymmetric', 'exit_z': 0.3,
                    'hold_mode': 'fixed', 'hold_max': 3,
                    'max_pairs': mp, 'weight_mode': wm, 'cap_alloc': 'equal',
                    'oi_filter': False, 'wf_split_year': None,
                    'config_name': f"MP{mp}_{wm[:3].upper()}_LB{lb}_H3",
                })

    # --- Phase 10: Hold days extended sweep ---
    for hd in [2, 4, 5]:
        for lb in [10, 15]:
            for mp in [2, 3]:
                configs.append({
                    'lookback': lb, 'z_mode': 'dynamic', 'z_thresh_base': 1.5,
                    'exit_mode': 'asymmetric', 'exit_z': 0.3,
                    'hold_mode': 'fixed', 'hold_max': hd,
                    'max_pairs': mp, 'weight_mode': 'equal', 'cap_alloc': 'equal',
                    'oi_filter': False, 'wf_split_year': None,
                    'config_name': f"H{hd}_DYN_ASYM_LB{lb}_MP{mp}",
                })

    # --- Phase 11: Z threshold sweep with dynamic + asymmetric ---
    for zt in [1.0, 2.0]:
        for lb in [10, 15]:
            for mp in [2, 3, 4]:
                configs.append({
                    'lookback': lb, 'z_mode': 'dynamic', 'z_thresh_base': zt,
                    'exit_mode': 'asymmetric', 'exit_z': 0.3,
                    'hold_mode': 'fixed', 'hold_max': 3,
                    'max_pairs': mp, 'weight_mode': 'equal', 'cap_alloc': 'equal',
                    'oi_filter': False, 'wf_split_year': None,
                    'config_name': f"DASYM_Z{zt:.1f}_LB{lb}_MP{mp}",
                })

    # --- Phase 12: Best combos with adaptive hold ---
    for lb in [10, 15]:
        for mp in [2, 3, 4]:
            configs.append({
                'lookback': lb, 'z_mode': 'dynamic', 'z_thresh_base': 1.5,
                'exit_mode': 'asymmetric', 'exit_z': 0.3,
                'hold_mode': 'adaptive', 'hold_max': 3,
                'max_pairs': mp, 'weight_mode': 'equal', 'cap_alloc': 'equal',
                'oi_filter': False, 'wf_split_year': None,
                'config_name': f"ALL_LB{lb}_ADAP_MP{mp}",
            })

    print(f"  {len(configs)} full-period configurations", flush=True)

    # === Run full-period configs ===
    results = []
    for ci, cfg in enumerate(configs):
        r = run_backtest(
            lookback=cfg['lookback'],
            z_mode=cfg['z_mode'],
            z_thresh_base=cfg['z_thresh_base'],
            exit_mode=cfg['exit_mode'],
            exit_z=cfg['exit_z'],
            hold_mode=cfg['hold_mode'],
            hold_max=cfg['hold_max'],
            max_pairs=cfg['max_pairs'],
            weight_mode=cfg['weight_mode'],
            cap_alloc=cfg['cap_alloc'],
            oi_filter=cfg['oi_filter'],
            wf_split_year=cfg['wf_split_year'],
            config_name=cfg['config_name'],
        )
        if r is not None:
            results.append(r)
        if (ci + 1) % 50 == 0:
            print(f"  [{ci+1}/{len(configs)}] {len(results)} configs with results", flush=True)

    results.sort(key=lambda x: -x['ann'])
    full_results = results  # all are full-period here

    # === Walk-forward for top configs ===
    # Pick top 15 configs and run WF on 2022, 2023, 2024
    wf_results = []
    if full_results:
        top_for_wf = full_results[:15]
        wf_configs = []
        for r in top_for_wf:
            # Parse the original config from name — find it in configs list
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

        print(f"\n  Running {len(wf_configs)} walk-forward configs for top 15...", flush=True)
        for ci, cfg in enumerate(wf_configs):
            r = run_backtest(
                lookback=cfg['lookback'],
                z_mode=cfg['z_mode'],
                z_thresh_base=cfg['z_thresh_base'],
                exit_mode=cfg['exit_mode'],
                exit_z=cfg['exit_z'],
                hold_mode=cfg['hold_mode'],
                hold_max=cfg['hold_max'],
                max_pairs=cfg['max_pairs'],
                weight_mode=cfg['weight_mode'],
                cap_alloc=cfg['cap_alloc'],
                oi_filter=cfg['oi_filter'],
                wf_split_year=cfg['wf_split_year'],
                config_name=cfg['config_name'],
            )
            if r is not None:
                wf_results.append(r)
            if (ci + 1) % 20 == 0:
                print(f"    [{ci+1}/{len(wf_configs)}] {len(wf_results)} WF results", flush=True)

    # ========================================================
    # RESULTS OUTPUT
    # ========================================================
    print(f"\n{'=' * 140}")
    print(f"  TOP 20 FULL-PERIOD RESULTS (sorted by annual return)")
    print(f"{'=' * 140}")
    hdr = (f"  {'Config':45s} | {'Ann':>7s} | {'WR':>5s} | {'N':>4s} | {'DD':>6s} | "
           f"{'PF':>4s} | {'Sharpe':>6s} | {'AvgW':>7s} | {'AvgL':>6s} | {'AvgD':>4s}")
    print(hdr)
    print(f"  {'-' * 135}")
    for r in full_results[:20]:
        print(f"  {r['name']:45s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:4.2f} | "
              f"{r['sharpe']:6.2f} | {r['avg_win']:+6.2f}% | {r['avg_loss']:5.2f}% | "
              f"{r['avg_days']:4.1f}")

    # Walk-forward results
    if wf_results:
        wf_results.sort(key=lambda x: -x['ann'])
        print(f"\n{'=' * 140}")
        print(f"  TOP 10 WALK-FORWARD RESULTS (out-of-sample)")
        print(f"{'=' * 140}")
        for r in wf_results[:10]:
            print(f"  {r['name']:55s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                  f"Sh {r['sharpe']:6.2f}")

    # Best config detailed breakdown
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

    # Top 5 yearly breakdown
    if len(full_results) >= 2:
        print(f"\n  YEARLY BREAKDOWN FOR TOP 5 CONFIGS:")
        for idx, r in enumerate(full_results[:5]):
            print(f"\n  #{idx+1}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.1f}%, "
                  f"DD={r['dd']:.1f}%, Sharpe={r['sharpe']:.2f})")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:3d}t  WR={wr_y:5.1f}%  PnL={ys['pnl']:+.1f}%")

    # Best config yearly comparison (walk-forward)
    if wf_results:
        print(f"\n  WALK-FORWARD YEARLY COMPARISON FOR BEST CONFIG:")
        # Group by base config
        from collections import defaultdict
        wf_by_config = defaultdict(dict)
        for r in wf_results:
            base = r['name'].rsplit('_WF', 1)[0]
            year = r['name'].rsplit('_WF', 1)[1]
            wf_by_config[base][year] = r

        # Show top 5 base configs
        base_order = []
        for r in full_results[:15]:
            if r['name'] in wf_by_config:
                base_order.append(r['name'])

        for base in base_order[:5]:
            wf_data = wf_by_config.get(base, {})
            print(f"\n  {base}:")
            for yr in ['2022', '2023', '2024']:
                if yr in wf_data:
                    wr = wf_data[yr]
                    print(f"    WF{yr}: Ann={wr['ann']:+7.1f}%  WR={wr['wr']:5.1f}%  "
                          f"N={wr['n']:3d}  DD={wr['dd']:5.1f}%  PF={wr['pf']:.2f}  "
                          f"Sh={wr['sharpe']:.2f}")
                else:
                    print(f"    WF{yr}: (no data)")

    # Per-pair summary across top 20
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

    # Optimization dimension analysis
    if full_results:
        print(f"\n  OPTIMIZATION DIMENSION ANALYSIS (top 20 configs):")
        dims = {
            'z_mode': {}, 'exit_mode': {}, 'hold_mode': {},
            'weight_mode': {}, 'cap_alloc': {}, 'oi_filter': {},
        }
        for r in full_results[:20]:
            name = r['name']
            # Parse dimension from name prefix
            for prefix, dim in [('DYN_', 'z_mode'), ('DASYM_', 'combo_dyn_asym'),
                                ('ASYM_', 'exit_asym'), ('ADAP_', 'hold_adaptive'),
                                ('ADPT_', 'hold_adaptive'), ('SIG_', 'hold_signal'),
                                ('W-SHA_', 'weight_sharpe'), ('W-WR_', 'weight_wr'),
                                ('IVOL_', 'cap_inv_vol'), ('OI_', 'oi_filter'),
                                ('ALL_', 'combo_all'), ('F_', 'fixed_baseline'),
                                ('MP1_', 'mp_sweep'), ('MP5_', 'mp_sweep'),
                                ('H2_', 'hold_2'), ('H4_', 'hold_4'), ('H5_', 'hold_5')]:
                if name.startswith(prefix):
                    key = prefix.rstrip('_')
                    if key not in dims['z_mode']:
                        dims['z_mode'][key] = {'n': 0, 'ann_sum': 0.0}
                    dims['z_mode'][key]['n'] += 1
                    dims['z_mode'][key]['ann_sum'] += r['ann']
                    break

        for dim_name, vals in dims.items():
            if vals:
                print(f"\n    {dim_name}:")
                for k, v in sorted(vals.items(), key=lambda x: -x[1]['ann_sum'] / max(x[1]['n'], 1)):
                    avg_ann = v['ann_sum'] / max(v['n'], 1)
                    print(f"      {k:20s}: {v['n']:2d} configs  Avg Ann={avg_ann:+7.1f}%")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 140)


if __name__ == '__main__':
    main()
