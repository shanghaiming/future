"""
V306: Fusion Strategy — Corrected Pairs + Factor + Leverage
============================================================
1. Corrected pair trading with i0 (iron ore) and all ferrous pairs
2. V301 regime-aware factor signals (long-only)
3. Capital split: adaptive allocation between pairs and factors
4. Leverage parameter for concentrated bets
5. Walk-forward validation
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_v301 import load_all_data, compute_factors, detect_regimes, generate_signals

# ============================================================
# CONSTANTS
# ============================================================
CASH0 = 1_000_000
COMM = 0.0003
COMM_FACTOR = 0.0005  # Higher commission for factor strategy

MULT = {'agfi': 15, 'alfi': 5, 'aufi': 1000, 'bufi': 10, 'cufi': 5, 'fufi': 10,
        'rbfi': 10, 'znfi': 5, 'nifi': 1, 'hcfi': 10, 'spfi': 10, 'ssfi': 5,
        'sffi': 5, 'smfi': 5, 'pbfi': 5, 'snfi': 1, 'rufi': 10, 'wrffi': 10,
        'afi': 10, 'bfi': 10, 'bbfi': 500, 'cffi': 5, 'cfi': 10, 'csfi': 10,
        'ebfi': 5, 'egfi': 10, 'fbfi': 500, 'i0': 100, 'jmfi': 60,
        'lfi': 5, 'mfi': 10, 'pgfi': 20, 'ppfi': 5, 'vfi': 5, 'yfi': 10,
        'pfi': 10, 'jdfi': 5, 'lhfi': 16, 'pkfi': 5, 'rrfi': 20, 'lrfi': 20,
        'jrfi': 20, 'pmfi': 20, 'whfi': 20, 'rsfi': 20, 'cjfi': 10, 'mafi': 10,
        'apfi': 10, 'cyfi': 5, 'fgfi': 20, 'oifi': 10, 'pfifi': 5, 'rmfi': 10,
        'srfi': 10, 'tafi': 5, 'safi': 20, 'urfi': 20, 'scfi': 1000, 'lufi': 10,
        'bcfi': 5, 'nrfi': 1, 'lgfi': 20, 'brfi': 5, 'lcfi': 1, 'sifi': 5,
        'im0': 100, 'aofi': 10, 'shfi': 10}
DEF_MULT = 10

# Corrected pairs using i0 for iron ore
PAIRS_CORE = [
    ('rbfi', 'i0'),    # rebar / iron ore  ← KEY pair
    ('hcfi', 'i0'),    # hot coil / iron ore
    ('hcfi', 'rbfi'),  # hot coil / rebar
    ('jmfi', 'i0'),    # coking coal / iron ore
    ('hcfi', 'jmfi'),  # hot coil / coking coal
    ('rbfi', 'jmfi'),  # rebar / coking coal
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
    ('cufi', 'znfi'),  # copper / zinc
    ('alfi', 'znfi'),  # aluminum / zinc
    ('mfi', 'yfi'),    # meal / soy oil
]

SPREAD_RAW, SPREAD_PCT, SPREAD_LOG = 'raw', 'pct', 'log'
ALL_MODES = [SPREAD_RAW, SPREAD_PCT, SPREAD_LOG]
ALL_LOOKBACKS = [5, 7, 10, 15, 20]
CANDIDATE_LOG_BIAS = [
    (SPREAD_LOG, 10), (SPREAD_LOG, 15), (SPREAD_LOG, 20),
    (SPREAD_RAW, 10), (SPREAD_PCT, 10),
]


# ============================================================
# PAIR TRADING ENGINE
# ============================================================
def precompute_pair_signals(C, NS, ND, pair_indices, min_train=60):
    """Precompute spreads, z-scores, and hypothetical returns."""
    t0 = time.time()
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
                if mode == SPREAD_RAW: spread[di] = pd_val - pu
                elif mode == SPREAD_PCT: spread[di] = (pd_val - pu) / pu
                elif mode == SPREAD_LOG: spread[di] = np.log(pd_val) - np.log(pu)

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

    # Hypothetical returns for adaptive selection
    all_zt = [0.5, 0.8, 1.0]
    hyp_ret = {}
    for zt in all_zt:
        for mode in ALL_MODES:
            for lb in ALL_LOOKBACKS:
                combo_key = (mode, lb, zt)
                daily_ret = np.full(ND, np.nan)
                for di in range(min_train + 1, ND):
                    pair_rets = []
                    for down_si, up_si, down_sym, up_sym in pair_indices:
                        z_arr = z_scores[mode].get((down_si, up_si), {}).get(lb)
                        if z_arr is None: continue
                        z_prev = z_arr[di - 1]
                        if np.isnan(z_prev) or abs(z_prev) < zt: continue
                        c_de = C[down_si, di-1]; c_ue = C[up_si, di-1]
                        c_dx = C[down_si, di]; c_ux = C[up_si, di]
                        if any(np.isnan(x) or x <= 0 for x in [c_de, c_ue, c_dx, c_ux]):
                            continue
                        mult_d = MULT.get(down_sym, DEF_MULT)
                        mult_u = MULT.get(up_sym, DEF_MULT)
                        if z_prev > 0:
                            pnl = (c_de - c_dx)*mult_d + (c_ux - c_ue)*mult_u
                        else:
                            pnl = (c_dx - c_de)*mult_d + (c_ue - c_ux)*mult_u
                        invested = c_de*mult_d + c_ue*mult_u
                        cost = invested * COMM * 2
                        pair_rets.append((pnl - cost) / invested * 100 if invested > 0 else 0)
                    if pair_rets:
                        daily_ret[di] = np.mean(pair_rets)
                hyp_ret[combo_key] = daily_ret

    print(f"  Pair signals: {len(all_pair_keys)} pairs × 3 modes × 5 lb ({time.time()-t0:.1f}s)", flush=True)
    return z_scores, hyp_ret


def backtest_pairs(C, ND, dates, syms, pair_indices, z_scores, hyp_ret,
                   z_thresh=0.8, hold_max=3, max_pairs=1, eval_period=40,
                   start_di=60, end_di=None, regime=None, regime_filter=True,
                   capital_fraction=0.5):
    """Pair trading backtest with separate capital tracking."""
    if end_di is None: end_di = ND
    cash = CASH0 * capital_fraction
    trades = []
    positions = []
    current_combo = CANDIDATE_LOG_BIAS[0]

    for di in range(start_di, end_di):
        # Adaptive mode selection
        if di > start_di and (di - start_di) % eval_period == 0:
            best_combo, best_score = CANDIDATE_LOG_BIAS[0], -1e18
            for c in CANDIDATE_LOG_BIAS:
                dr = hyp_ret.get((c[0], c[1], z_thresh))
                if dr is None: continue
                w = dr[max(start_di, di-eval_period):di]
                v = w[~np.isnan(w)]
                s = np.nansum(v) if len(v) >= 3 else -1e10
                if s > best_score: best_score, best_combo = s, c
            current_combo = best_combo

        # Regime filter
        if regime_filter and regime is not None:
            r = regime[di] if di < len(regime) else 0
            if r in (-1, 2): continue  # Skip choppy/volatile

        # Exit management
        new_pos = []
        for pos in positions:
            z_arr = z_scores[pos['mode']].get((pos['dsi'], pos['usi']), {}).get(pos['lb'])
            z_now = z_arr[di] if z_arr is not None and di < len(z_arr) else np.nan
            days_held = di - pos['edi']
            exit_r = None

            if not np.isnan(z_now):
                if pos['dir'] == 1 and z_now >= 0: exit_r = 'mean_rev'
                elif pos['dir'] == -1 and z_now <= 0: exit_r = 'mean_rev'
            if exit_r is None and not np.isnan(z_now):
                if pos['dir'] == 1 and z_now < pos['ez'] - 1.5: exit_r = 'stop'
                elif pos['dir'] == -1 and z_now > pos['ez'] + 1.5: exit_r = 'stop'
            if exit_r is None and days_held >= hold_max: exit_r = 'time'

            if exit_r:
                cd = C[pos['dsi'], di]; cu = C[pos['usi'], di]
                if np.isnan(cd) or cd <= 0: cd = pos['cd']
                if np.isnan(cu) or cu <= 0: cu = pos['cu']
                md = MULT.get(pos['dsym'], DEF_MULT); mu = MULT.get(pos['usym'], DEF_MULT)
                ld, lu = pos['ld'], pos['lu']
                if pos['dir'] == 1:
                    pnl = (cd - pos['cd'])*md*ld + (pos['cu'] - cu)*mu*lu
                else:
                    pnl = (pos['cd'] - cd)*md*ld + (cu - pos['cu'])*mu*lu
                ev = pos['cd']*md*ld + pos['cu']*mu*lu
                xv = cd*md*ld + cu*mu*lu
                cost = ev * COMM + xv * COMM
                total_pnl = pnl - cost
                pnl_pct = total_pnl / ev * 100 if ev > 0 else 0

                if pos['dir'] == 1:
                    cash_ret = cd*md*ld - cu*mu*lu
                else:
                    cash_ret = -cd*md*ld + cu*mu*lu
                cash += pos['ci'] + cash_ret - xv * COMM

                trades.append({
                    'pnl_abs': total_pnl, 'pnl_pct': pnl_pct,
                    'days': days_held, 'di': di, 'year': dates[di].year,
                    'type': 'pair', 'pair': f"{pos['dsym']}/{pos['usym']}",
                    'dir': pos['dir'], 'reason': exit_r,
                })
            else:
                new_pos.append(pos)
        positions = new_pos

        # Entry
        occupied = set()
        for p in positions:
            occupied.add(p['dsi']); occupied.add(p['usi'])
        n_open = max_pairs - len(positions)
        if n_open <= 0: continue

        use_mode, use_lb = current_combo
        candidates = []
        for dsi, usi, dsym, usym in pair_indices:
            if dsi in occupied or usi in occupied: continue
            z_arr = z_scores[use_mode].get((dsi, usi), {}).get(use_lb)
            if z_arr is None: continue
            z_val = z_arr[di] if di < len(z_arr) else np.nan
            if np.isnan(z_val) or abs(z_val) < z_thresh: continue
            candidates.append((abs(z_val), dsi, usi, dsym, usym, z_val))

        if not candidates: continue
        candidates.sort(key=lambda x: -x[0])

        opened = 0
        for _, dsi, usi, dsym, usym, z_val in candidates:
            if opened >= n_open: break
            if dsi in occupied or usi in occupied: continue
            cd = C[dsi, di]; cu = C[usi, di]
            if np.isnan(cd) or cd <= 0 or np.isnan(cu) or cu <= 0: continue

            md = MULT.get(dsym, DEF_MULT); mu = MULT.get(usym, DEF_MULT)
            cap = cash * 0.9 / max(1, max_pairs)
            per_leg = cap / 2
            ld = int(per_leg / (cd * md * (1 + COMM)))
            lu = int(per_leg / (cu * mu * (1 + COMM)))
            if ld <= 0 or lu <= 0: continue

            tc = cd*md*ld*(1+COMM) + cu*mu*lu*(1+COMM)
            if tc > cash:
                scale = cash * 0.9 / tc
                ld = max(1, int(ld * scale)); lu = max(1, int(lu * scale))
                tc = cd*md*ld*(1+COMM) + cu*mu*lu*(1+COMM)
                if tc > cash: continue

            pos_dir = -1 if z_val > 0 else 1
            cash -= tc
            positions.append({
                'dsi': dsi, 'usi': usi, 'dsym': dsym, 'usym': usym,
                'cd': cd, 'cu': cu, 'ld': ld, 'lu': lu,
                'edi': di, 'ez': z_val, 'dir': pos_dir,
                'ci': tc, 'mode': use_mode, 'lb': use_lb,
            })
            occupied.add(dsi); occupied.add(usi)
            opened += 1

    # Close remaining
    actual_end = min(end_di, ND) - 1
    for pos in positions:
        cd = C[pos['dsi'], actual_end]; cu = C[pos['usi'], actual_end]
        if np.isnan(cd) or cd <= 0: cd = pos['cd']
        if np.isnan(cu) or cu <= 0: cu = pos['cu']
        md = MULT.get(pos['dsym'], DEF_MULT); mu = MULT.get(pos['usym'], DEF_MULT)
        ld, lu = pos['ld'], pos['lu']
        if pos['dir'] == 1:
            pnl = (cd - pos['cd'])*md*ld + (pos['cu'] - cu)*mu*lu
        else:
            pnl = (pos['cd'] - cd)*md*ld + (cu - pos['cu'])*mu*lu
        ev = pos['cd']*md*ld + pos['cu']*mu*lu
        xv = cd*md*ld + cu*mu*lu
        total_pnl = pnl - (ev + xv) * COMM
        cash += pos['ci'] + ((-1)**(pos['dir']==-1))*(cd*md*ld - cu*mu*lu) - xv*COMM
        trades.append({
            'pnl_abs': total_pnl, 'pnl_pct': total_pnl/ev*100 if ev > 0 else 0,
            'days': actual_end-pos['edi'], 'di': actual_end,
            'year': dates[actual_end].year, 'type': 'pair',
            'pair': f"{pos['dsym']}/{pos['usym']}",
            'dir': pos['dir'], 'reason': 'end',
        })

    return trades, cash


# ============================================================
# FACTOR STRATEGY (Long-only with leverage)
# ============================================================
def backtest_factor(signal, C, O, H, L, NS, ND, dates, syms, regime,
                    top_n=5, hold_days=3, atr_stop=2.5, leverage=1.0,
                    start_di=60, end_di=None, capital_fraction=0.5):
    """V303-style leveraged long-only factor backtest."""
    if end_di is None: end_di = ND
    equity = CASH0 * capital_fraction
    peak = equity
    positions = []
    trades = []
    max_pos = top_n
    pos_alloc = leverage / max_pos

    for di in range(max(start_di, 1), end_di):
        daily_pnl = 0
        new_positions = []
        for si, edi, ep, sp, d, a in positions:
            c = C[si, di]
            if np.isnan(c):
                new_positions.append((si, edi, ep, sp, d, a)); continue
            exit_r = None
            if d > 0 and c < sp: exit_r = 'stop'
            elif di - edi >= hold_days: exit_r = 'hold'
            if exit_r:
                pnl = d * (c - ep) / ep - COMM_FACTOR
                profit = equity * a * pnl
                daily_pnl += profit
                trades.append({
                    'pnl_abs': profit, 'pnl_pct': pnl * 100,
                    'days': di - edi, 'di': di, 'year': dates[di].year,
                    'type': 'factor', 'sym': syms[si],
                    'dir': d, 'reason': exit_r,
                })
            else:
                new_positions.append((si, edi, ep, sp, d, a))
        positions = new_positions
        equity += daily_pnl
        if equity > peak: peak = equity
        if equity <= 0: break
        if di < start_di: continue

        held = {p[0] for p in positions}
        if len(positions) >= max_pos: continue

        r = regime[di] if di < len(regime) else 0
        size_mult = {1: 1.0, 0: 0.8, -1: 0.6, 2: 0.3}.get(r, 0.5)
        alloc = pos_alloc * size_mult

        sig_vals = [(signal[si, di], si) for si in range(NS)
                    if not np.isnan(signal[si, di]) and si not in held
                    and not np.isnan(C[si, di]) and not np.isnan(O[si, di])
                    and signal[si, di] > 0.15]
        if not sig_vals: continue
        sig_vals.sort(key=lambda x: x[0], reverse=True)

        for score, si in sig_vals[:top_n]:
            if len(positions) >= max_pos or si in held: break
            op = O[si, di]
            if np.isnan(op) or op <= 0: continue
            atr_v = []
            for j in range(max(start_di, di - 14), di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
            if not atr_v: continue
            atr = np.mean(atr_v)
            positions.append((si, di, op, op - atr_stop * atr, 1, alloc))
            held.add(si)

    # Close remaining
    for si, edi, ep, sp, d, a in positions:
        c = C[si, ND - 1]
        if not np.isnan(c):
            pnl = d * (c - ep) / ep - COMM_FACTOR
            profit = equity * a * pnl
            trades.append({
                'pnl_abs': profit, 'pnl_pct': pnl * 100,
                'days': ND - 1 - edi, 'di': ND - 1,
                'year': dates[ND - 1].year, 'type': 'factor',
                'sym': syms[si], 'dir': d, 'reason': 'end',
            })

    return trades, equity


# ============================================================
# FUSION BACKTEST
# ============================================================
def backtest_fusion(C, O, H, L, V, OI, NS, ND, dates, syms,
                    pair_indices, regime, signal,
                    pair_params=None, factor_params=None):
    """Run fusion of pair trading + factor strategy."""
    if pair_params is None:
        pair_params = {'z_thresh': 0.8, 'hold_max': 3, 'max_pairs': 1}
    if factor_params is None:
        factor_params = {'top_n': 5, 'hold_days': 3, 'atr_stop': 2.5, 'leverage': 2.0}

    # Precompute pair signals
    z_scores, hyp_ret = precompute_pair_signals(C, NS, ND, pair_indices)

    # Run pair trading (50% capital)
    pair_trades, pair_cash = backtest_pairs(
        C, ND, dates, syms, pair_indices, z_scores, hyp_ret,
        regime=regime, regime_filter=True,
        capital_fraction=0.5, **pair_params)

    # Run factor strategy (50% capital)
    factor_trades, factor_eq = backtest_factor(
        signal, C, O, H, L, NS, ND, dates, syms, regime,
        capital_fraction=0.5, **factor_params)

    all_trades = pair_trades + factor_trades
    all_trades.sort(key=lambda x: x['di'])

    # Combined equity curve
    pair_final = pair_cash
    factor_final = factor_eq
    total_final = pair_final + factor_final

    return all_trades, pair_final, factor_final, total_final


# ============================================================
# ANALYSIS
# ============================================================
def analyze_trades(trades, label="", cash_init=CASH0):
    if not trades:
        print(f"  {label}: no trades")
        return None

    nw = sum(1 for t in trades if t['pnl_abs'] > 0)
    wr = nw / len(trades) * 100
    pnls = [t['pnl_pct'] for t in trades]
    abs_pnls = [t['pnl_abs'] for t in trades]

    equity = cash_init
    peak = cash_init
    max_dd = 0
    for t in sorted(trades, key=lambda x: x['di']):
        equity += t['pnl_abs']
        if equity > peak: peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100
            if dd > max_dd: max_dd = dd

    cum_pnl = sum(abs_pnls)
    ann_ret = ((equity / cash_init) ** (1 / max((trades[-1]['di'] - trades[0]['di']) / 252, 0.1)) - 1) * 100 if len(trades) > 1 else 0

    # Sharpe
    if len(abs_pnls) > 1:
        rets = np.array(abs_pnls) / cash_init
        sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
    else:
        sharpe = 0

    print(f"  {label}: {len(trades)} trades WR={wr:.1f}% cum={cum_pnl:+,.0f} "
          f"DD={max_dd:.1f}% ann={ann_ret:+.1f}% Sh={sharpe:.2f}")

    # Year breakdown
    yr_stats = {}
    for t in trades:
        y = t['year']
        if y not in yr_stats: yr_stats[y] = {'n': 0, 'w': 0, 'pnl': 0.0}
        yr_stats[y]['n'] += 1
        if t['pnl_abs'] > 0: yr_stats[y]['w'] += 1
        yr_stats[y]['pnl'] += t['pnl_abs']
    for y in sorted(yr_stats.keys()):
        ys = yr_stats[y]
        wr_y = ys['w'] / ys['n'] * 100
        print(f"    {y}: {ys['n']}t WR={wr_y:.1f}% pnl={ys['pnl']:+,.0f}")

    return {'n': len(trades), 'wr': wr, 'dd': max_dd, 'ann': ann_ret,
            'sharpe': sharpe, 'cum_pnl': cum_pnl}


# ============================================================
# WALK-FORWARD
# ============================================================
def walk_forward_fusion(C, O, H, L, V, OI, NS, ND, dates, syms,
                        pair_indices, train_years=4, test_years=1,
                        pair_params=None, factor_params=None):
    years = sorted(set(d.year for d in dates))
    all_trades = []
    start_yr = years[0]
    pf = pair_params or {'z_thresh': 0.8, 'hold_max': 3, 'max_pairs': 1}
    ff = factor_params or {'top_n': 5, 'hold_days': 3, 'atr_stop': 2.5, 'leverage': 2.0}

    print(f"\nWF Fusion: pair={pf} factor={ff}", flush=True)

    while True:
        train_end = start_yr + train_years - 1
        test_yr = train_end + 1
        if test_yr > years[-1]: break

        train_m = np.array([d.year <= train_end for d in dates])
        test_m = np.array([d.year == test_yr for d in dates])
        t0 = max(0, np.where(train_m)[0][0] - 60)
        sl = slice(t0, np.where(test_m)[0][-1] + 1)

        C_s, O_s, H_s, L_s = C[:, sl], O[:, sl], H[:, sl], L[:, sl]
        V_s, OI_s = V[:, sl], OI[:, sl]
        d_s = dates[sl]
        ND_s = len(d_s)

        # Compute signals on full slice
        F = compute_factors(C_s, O_s, H_s, L_s, V_s, OI_s, NS, ND_s)
        regime_s = detect_regimes(F, NS, ND_s)
        signal_s, _, _, _ = generate_signals(F, regime_s, C_s, NS, ND_s, syms)

        test_start = np.where(np.array([d.year == test_yr for d in d_s]))[0][0]

        trades, pf_cash, ft_eq, total = backtest_fusion(
            C_s, O_s, H_s, L_s, V_s, OI_s, NS, ND_s, d_s, syms,
            pair_indices, regime_s, signal_s,
            pair_params=pf, factor_params=ff)

        yr_trades = [t for t in trades if t['year'] == test_yr]
        if yr_trades:
            nw = sum(1 for t in yr_trades if t['pnl_abs'] > 0)
            wr = nw / len(yr_trades) * 100
            pairs_n = sum(1 for t in yr_trades if t['type'] == 'pair')
            factor_n = sum(1 for t in yr_trades if t['type'] == 'factor')
            print(f"  {test_yr}: {len(yr_trades)}t (P:{pairs_n} F:{factor_n}) "
                  f"WR={wr:.1f}% total={total:+,.0f}", flush=True)
            all_trades.extend(yr_trades)
        else:
            print(f"  {test_yr}: no trades", flush=True)

        start_yr += 1

    if all_trades:
        print(f"\n  --- WF Summary ---")
        analyze_trades(all_trades, "Combined WF")
        pair_t = [t for t in all_trades if t['type'] == 'pair']
        fact_t = [t for t in all_trades if t['type'] == 'factor']
        if pair_t: analyze_trades(pair_t, "Pairs WF")
        if fact_t: analyze_trades(fact_t, "Factor WF")

    return all_trades


# ============================================================
# MAIN
# ============================================================
def main():
    t_start = time.time()
    print("=" * 60)
    print("  V306: FUSION — CORRECTED PAIRS + FACTOR + LEVERAGE")
    print("=" * 60)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    sym_to_si = {syms[si]: si for si in range(NS)}

    # Build pair indices with corrected symbols
    pair_indices = []
    for down_sym, up_sym in PAIRS_CORE:
        down_si = sym_to_si.get(down_sym, -1)
        up_si = sym_to_si.get(up_sym, -1)
        if down_si >= 0 and up_si >= 0:
            pair_indices.append((down_si, up_si, down_sym, up_sym))
        else:
            print(f"  SKIP: ({down_sym}, {up_sym}) not in data")
    print(f"  Active pairs: {len(pair_indices)}/{len(PAIRS_CORE)}")

    # Compute regime and signals
    print("[V306] Computing regime and factor signals...", flush=True)
    F = compute_factors(C, O, H, L, V, OI, NS, ND)
    regime = detect_regimes(F, NS, ND)
    signal, _, _, _ = generate_signals(F, regime, C, NS, ND, syms)

    # ================================================================
    # FUSION SWEEP (Full Backtest)
    # ================================================================
    print("\n--- Full Fusion Sweep ---")
    fusion_results = []

    for zt in [0.5, 0.8, 1.0]:
        for hm in [2, 3]:
            for lev in [1.0, 2.0, 3.0]:
                pp = {'z_thresh': zt, 'hold_max': hm, 'max_pairs': 1}
                fp = {'top_n': 5, 'hold_days': 3, 'atr_stop': 2.5, 'leverage': lev}
                trades, pf, ff, total = backtest_fusion(
                    C, O, H, L, V, OI, NS, ND, dates, syms,
                    pair_indices, regime, signal,
                    pair_params=pp, factor_params=fp)

                nw = sum(1 for t in trades if t['pnl_abs'] > 0)
                wr = nw / len(trades) * 100 if trades else 0
                ann = ((total / CASH0) ** (1 / max((dates[-1]-dates[0]).days/365.25, 0.1)) - 1) * 100
                pair_n = sum(1 for t in trades if t['type'] == 'pair')
                fact_n = sum(1 for t in trades if t['type'] == 'factor')

                fusion_results.append({
                    'zt': zt, 'hm': hm, 'lev': lev,
                    'n': len(trades), 'wr': wr, 'ann': ann,
                    'total': total, 'pair_n': pair_n, 'fact_n': fact_n,
                    'pp': pp, 'fp': fp,
                })

    fusion_results.sort(key=lambda x: -x['ann'])
    print(f"\n{'ZT':>4} {'HM':>3} {'LEV':>4} {'N':>5} {'WR':>5} {'Ann':>8} {'Final':>12}")
    print("-" * 50)
    for r in fusion_results[:15]:
        print(f"{r['zt']:>4.1f} {r['hm']:>3} {r['lev']:>4.1f} {r['n']:>5} "
              f"{r['wr']:>5.1f} {r['ann']:>+8.1f} {r['total']:>12,.0f}")

    # ================================================================
    # Detailed analysis of top config
    # ================================================================
    best = fusion_results[0]
    print(f"\n--- Best Config: zt={best['zt']} hm={best['hm']} lev={best['lev']} ---")
    trades, pf, ff, total = backtest_fusion(
        C, O, H, L, V, OI, NS, ND, dates, syms,
        pair_indices, regime, signal,
        pair_params=best['pp'], factor_params=best['fp'])
    analyze_trades(trades, "Best Fusion")
    analyze_trades([t for t in trades if t['type'] == 'pair'], "  Pairs")
    analyze_trades([t for t in trades if t['type'] == 'factor'], "  Factor")

    # ================================================================
    # WALK-FORWARD for top 3 configs
    # ================================================================
    print("\n" + "=" * 60)
    print("  WALK-FORWARD VALIDATION")
    print("=" * 60)

    seen = set()
    wf_count = 0
    for r in fusion_results[:10]:
        if wf_count >= 3: break
        key = (r['zt'], r['hm'], r['lev'])
        if key in seen: continue
        seen.add(key)
        print(f"\n--- WF #{wf_count+1}: zt={r['zt']} hm={r['hm']} lev={r['lev']} ---")
        walk_forward_fusion(C, O, H, L, V, OI, NS, ND, dates, syms,
                            pair_indices,
                            pair_params=r['pp'], factor_params=r['fp'])
        wf_count += 1

    # ================================================================
    # PAIR-ONLY ANALYSIS (with corrected ferrous pairs)
    # ================================================================
    print("\n" + "=" * 60)
    print("  PAIR-ONLY ANALYSIS (Corrected Pairs)")
    print("=" * 60)

    z_scores, hyp_ret = precompute_pair_signals(C, NS, ND, pair_indices)

    for zt in [0.5, 0.8, 1.0]:
        for hm in [1, 2, 3]:
            for mp in [1, 2]:
                for rf in [True]:
                    trades, cash = backtest_pairs(
                        C, ND, dates, syms, pair_indices, z_scores, hyp_ret,
                        z_thresh=zt, hold_max=hm, max_pairs=mp,
                        regime=regime, regime_filter=rf, capital_fraction=1.0)
                    if len(trades) < 3: continue
                    nw = sum(1 for t in trades if t['pnl_abs'] > 0)
                    wr = nw / len(trades) * 100
                    cum = sum(t['pnl_abs'] for t in trades)
                    ann = ((cash / CASH0) ** (1 / max((dates[-1]-dates[0]).days/365.25, 0.1)) - 1) * 100
                    print(f"  zt={zt} hm={hm} mp={mp} rf={rf}: {len(trades)}t WR={wr:.1f}% "
                          f"ann={ann:+.1f}% cum={cum:+,.0f}")

    print(f"\n[V306] Done. Total: {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
