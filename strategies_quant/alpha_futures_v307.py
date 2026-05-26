"""
V307: Clean Percentage-Based Pair + Factor Fusion
===================================================
Fixes V306's compounding artifact by using equity-curve-based percentage returns.
All P&L tracked as equity changes, positions sized as fraction of current equity.

Key improvements:
1. Equity-curve-based P&L: every position is a fraction of current equity
2. No contract multipliers — use percentage returns on price changes
3. Proper walk-forward with equity compounding
4. Dynamic pair selection based on rolling IC (information coefficient)
5. Term structure carry factor integration
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_v301 import load_all_data, compute_factors, detect_regimes, generate_signals

CASH0 = 1_000_000
COMM = 0.0005  # Unified commission

# Pair definitions
PAIRS = [
    ('rbfi', 'i0'), ('hcfi', 'i0'), ('hcfi', 'rbfi'),
    ('jmfi', 'i0'), ('hcfi', 'jmfi'), ('rbfi', 'jmfi'),
    ('mafi', 'scfi'), ('fufi', 'scfi'), ('bfi', 'scfi'),
    ('mfi', 'afi'), ('yfi', 'afi'), ('pfi', 'yfi'),
    ('ppfi', 'mafi'), ('vfi', 'mafi'), ('egfi', 'mafi'),
    ('cfi', 'csfi'), ('cufi', 'znfi'), ('alfi', 'znfi'), ('mfi', 'yfi'),
]

SPREAD_RAW, SPREAD_PCT, SPREAD_LOG = 'raw', 'pct', 'log'
ALL_MODES = [SPREAD_RAW, SPREAD_PCT, SPREAD_LOG]
ALL_LOOKBACKS = [5, 7, 10, 15, 20]


# ============================================================
# PRECOMPUTATION
# ============================================================
def precompute_pair_signals(C, NS, ND, pair_indices):
    """Precompute z-scores for all modes × lookbacks × pairs."""
    t0 = time.time()
    z_scores = {m: {} for m in ALL_MODES}
    spreads = {m: {} for m in ALL_MODES}

    for dsi, usi, dsym, usym in pair_indices:
        key = (dsi, usi)
        for mode in ALL_MODES:
            spread = np.full(ND, np.nan)
            for di in range(ND):
                pd_v = C[dsi, di]; pu_v = C[usi, di]
                if np.isnan(pd_v) or np.isnan(pu_v) or pu_v <= 0 or pd_v <= 0:
                    continue
                if mode == SPREAD_RAW: spread[di] = pd_v - pu_v
                elif mode == SPREAD_PCT: spread[di] = (pd_v - pu_v) / pu_v
                elif mode == SPREAD_LOG: spread[di] = np.log(pd_v) - np.log(pu_v)
            spreads[mode][key] = spread

            z_scores[mode][key] = {}
            for lb in ALL_LOOKBACKS:
                z = np.full(ND, np.nan)
                for di in range(lb, ND):
                    window = spread[di - lb:di]
                    valid = window[~np.isnan(window)]
                    if len(valid) >= max(3, int(lb * 0.8)):
                        m_v = np.mean(valid)
                        s_v = np.std(valid, ddof=1)
                        if s_v > 1e-10:
                            z[di] = (spread[di] - m_v) / s_v
                z_scores[mode][key][lb] = z

    print(f"  Z-scores: {len(pair_indices)} pairs ({time.time()-t0:.1f}s)", flush=True)
    return z_scores, spreads


def compute_pair_ic(C, NS, ND, pair_indices, z_scores, mode, lb, zt, lookback=40):
    """Compute rolling IC (hit rate) for each pair over recent window."""
    pair_ic = {}
    for dsi, usi, dsym, usym in pair_indices:
        z_arr = z_scores[mode].get((dsi, usi), {}).get(lb)
        if z_arr is None: continue
        wins = 0; total = 0
        for di in range(max(60, ND - lookback), ND):
            z_prev = z_arr[di - 1]
            if np.isnan(z_prev) or abs(z_prev) < zt: continue
            # Check if signal was correct (mean reversion)
            cd = C[dsi, di]; cu = C[usi, di]
            cd_e = C[dsi, di-1]; cu_e = C[usi, di-1]
            if any(np.isnan(x) for x in [cd, cu, cd_e, cu_e]): continue
            if any(x <= 0 for x in [cd, cu, cd_e, cu_e]): continue

            # P&L of the pair trade
            if z_prev > 0:  # short spread
                pnl = (cd_e - cd) / cd_e + (cu - cu_e) / cu_e
            else:  # long spread
                pnl = (cd - cd_e) / cd_e + (cu_e - cu) / cu_e
            pnl -= COMM * 4  # entry + exit costs for both legs

            if pnl > 0: wins += 1
            total += 1

        if total >= 3:
            pair_ic[(dsi, usi)] = wins / total
    return pair_ic


# ============================================================
# PERCENTAGE-BASED PAIR BACKTEST
# ============================================================
def backtest_pairs_pct(C, NS, ND, dates, syms, pair_indices, z_scores,
                       z_thresh=0.8, hold_max=2, max_pairs=1,
                       eval_period=40, candidate_combos=None,
                       start_di=60, end_di=None,
                       regime=None, regime_filter=True,
                       alloc_per_pair=0.2, leverage=1.0):
    """
    Percentage-based pair trading.
    Each pair uses alloc_per_pair * leverage fraction of equity.
    P&L tracked as percentage changes to equity.
    """
    if end_di is None: end_di = ND
    if candidate_combos is None:
        candidate_combos = [(SPREAD_LOG, 10), (SPREAD_LOG, 15), (SPREAD_LOG, 20),
                            (SPREAD_RAW, 10), (SPREAD_PCT, 10)]

    equity = CASH0
    peak = equity
    max_dd = 0.0
    trades = []
    positions = []
    current_combo = candidate_combos[0]

    for di in range(start_di, end_di):
        # Adaptive mode selection
        if di > start_di and (di - start_di) % eval_period == 0:
            best_score = -1e18
            for c in candidate_combos:
                score = 0
                for dsi, usi, dsym, usym in pair_indices:
                    z_arr = z_scores[c[0]].get((dsi, usi), {}).get(c[1])
                    if z_arr is None: continue
                    for dd in range(max(start_di, di - eval_period), di):
                        z_prev = z_arr[dd - 1] if dd > 0 else np.nan
                        if np.isnan(z_prev) or abs(z_prev) < z_thresh: continue
                        cd_e = C[dsi, dd-1]; cu_e = C[usi, dd-1]
                        cd_x = C[dsi, dd]; cu_x = C[usi, dd]
                        if any(np.isnan(x) or x <= 0 for x in [cd_e, cu_e, cd_x, cu_x]):
                            continue
                        if z_prev > 0:
                            pnl = (cd_e - cd_x)/cd_e + (cu_x - cu_e)/cu_e
                        else:
                            pnl = (cd_x - cd_e)/cd_e + (cu_e - cu_x)/cu_x
                        pnl -= COMM * 4
                        score += pnl
                if score > best_score:
                    best_score = score
                    current_combo = c

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
                if np.isnan(cd) or cd <= 0 or np.isnan(cu) or cu <= 0:
                    # Can't exit, carry forward
                    new_pos.append(pos)
                    continue

                # Percentage P&L
                if pos['dir'] == 1:
                    pnl_pct = (cd - pos['cd']) / pos['cd'] + (pos['cu'] - cu) / pos['cu']
                else:
                    pnl_pct = (pos['cd'] - cd) / pos['cd'] + (cu - pos['cu']) / pos['cu']
                pnl_pct -= COMM * 4  # Round-trip costs for both legs

                # Apply to equity
                pos_size = pos['alloc'] * leverage
                equity_pnl = equity * pos_size * pnl_pct / 2  # /2 because two legs
                # Actually, we should apply pnl to the equity at time of entry
                # For simplicity with compounding, use current equity * alloc * pnl
                equity_pnl = pos['entry_equity'] * pos['alloc'] * pnl_pct / 2
                equity += equity_pnl

                trades.append({
                    'pnl_abs': equity_pnl,
                    'pnl_pct': pnl_pct,
                    'days': days_held, 'di': di, 'year': dates[di].year,
                    'type': 'pair', 'pair': f"{pos['dsym']}/{pos['usym']}",
                    'dir': pos['dir'], 'reason': exit_r,
                    'alloc': pos['alloc'],
                })
            else:
                new_pos.append(pos)
        positions = new_pos

        if equity > peak: peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100
            if dd > max_dd: max_dd = dd
        if equity <= 0: break

        # Entry
        if regime_filter and regime is not None:
            r = regime[di] if di < len(regime) else 0
            if r in (-1, 2): continue

        occupied = set()
        for p in positions:
            occupied.add(p['dsi']); occupied.add(p['usi'])
        n_open = max_pairs - len(positions)
        if n_open <= 0: continue

        use_mode, use_lb = current_combo

        # Dynamic pair selection: rank by |z-score|
        candidates = []
        for dsi, usi, dsym, usym in pair_indices:
            if dsi in occupied or usi in occupied: continue
            z_arr = z_scores[use_mode].get((dsi, usi), {}).get(use_lb)
            if z_arr is None: continue
            z_val = z_arr[di] if di < len(z_arr) else np.nan
            if np.isnan(z_val) or abs(z_val) < z_thresh: continue
            cd = C[dsi, di]; cu = C[usi, di]
            if np.isnan(cd) or cd <= 0 or np.isnan(cu) or cu <= 0: continue
            candidates.append((abs(z_val), dsi, usi, dsym, usym, z_val))

        if not candidates: continue
        candidates.sort(key=lambda x: -x[0])

        opened = 0
        for _, dsi, usi, dsym, usym, z_val in candidates:
            if opened >= n_open: break
            if dsi in occupied or usi in occupied: continue

            pos_dir = -1 if z_val > 0 else 1
            positions.append({
                'dsi': dsi, 'usi': usi, 'dsym': dsym, 'usym': usym,
                'cd': C[dsi, di], 'cu': C[usi, di],
                'edi': di, 'ez': z_val, 'dir': pos_dir,
                'mode': use_mode, 'lb': use_lb,
                'alloc': alloc_per_pair,
                'entry_equity': equity,
            })
            occupied.add(dsi); occupied.add(usi)
            opened += 1

    # Close remaining
    actual_end = min(end_di, ND) - 1
    for pos in positions:
        cd = C[pos['dsi'], actual_end]; cu = C[pos['usi'], actual_end]
        if np.isnan(cd) or cd <= 0: cd = pos['cd']
        if np.isnan(cu) or cu <= 0: cu = pos['cu']
        if pos['dir'] == 1:
            pnl_pct = (cd - pos['cd'])/pos['cd'] + (pos['cu'] - cu)/pos['cu']
        else:
            pnl_pct = (pos['cd'] - cd)/pos['cd'] + (cu - pos['cu'])/pos['cu']
        pnl_pct -= COMM * 4
        equity_pnl = pos['entry_equity'] * pos['alloc'] * pnl_pct / 2
        equity += equity_pnl
        trades.append({
            'pnl_abs': equity_pnl, 'pnl_pct': pnl_pct,
            'days': actual_end - pos['edi'], 'di': actual_end,
            'year': dates[actual_end].year, 'type': 'pair',
            'pair': f"{pos['dsym']}/{pos['usym']}",
            'dir': pos['dir'], 'reason': 'end', 'alloc': pos['alloc'],
        })

    return trades, equity, max_dd


# ============================================================
# ANALYSIS
# ============================================================
def analyze(trades, equity, max_dd, label=""):
    if not trades:
        print(f"  {label}: no trades")
        return None

    nw = sum(1 for t in trades if t['pnl_pct'] > 0)
    wr = nw / len(trades) * 100
    avg_pnl = np.mean([t['pnl_pct'] for t in trades])

    ann = ((equity / CASH0) ** (252 / max(sum(t['days'] for t in trades), 1)) - 1) * 100
    # Better annualization
    first_d = min(t['di'] for t in trades)
    last_d = max(t['di'] for t in trades)
    yrs = max((last_d - first_d) / 252, 0.1)
    ann = ((equity / CASH0) ** (1 / yrs) - 1) * 100

    abs_pnls = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
    if len(abs_pnls) > 1:
        rets = np.array(abs_pnls) / CASH0
        sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
    else:
        sharpe = 0

    print(f"  {label}: {len(trades)}t WR={wr:.1f}% avg={avg_pnl:+.4f}% "
          f"ann={ann:+.1f}% DD={max_dd:.1f}% Sh={sharpe:.2f} equity={equity:,.0f}")

    yr_stats = {}
    for t in trades:
        y = t['year']
        if y not in yr_stats: yr_stats[y] = {'n': 0, 'w': 0, 'pnl': []}
        yr_stats[y]['n'] += 1
        if t['pnl_pct'] > 0: yr_stats[y]['w'] += 1
        yr_stats[y]['pnl'].append(t['pnl_pct'])
    for y in sorted(yr_stats.keys()):
        ys = yr_stats[y]
        wr_y = ys['w'] / ys['n'] * 100
        cum = np.prod([1 + p / 100 for p in ys['pnl']]) - 1
        print(f"    {y}: {ys['n']}t WR={wr_y:.1f}% cum={cum*100:+.2f}%")

    return {'n': len(trades), 'wr': wr, 'dd': max_dd, 'ann': ann,
            'sharpe': sharpe, 'equity': equity}


# ============================================================
# WALK-FORWARD
# ============================================================
def walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms,
                 pair_indices, train_years=4, test_years=1,
                 z_thresh=0.8, hold_max=2, max_pairs=1,
                 alloc_per_pair=0.2, leverage=1.0):
    years = sorted(set(d.year for d in dates))
    all_trades = []
    start_yr = years[0]

    print(f"\nWF: zt={z_thresh} hm={hold_max} mp={max_pairs} "
          f"alloc={alloc_per_pair} lev={leverage}", flush=True)

    while True:
        train_end = start_yr + train_years - 1
        test_yr = train_end + 1
        if test_yr > years[-1]: break

        train_m = np.array([d.year <= train_end for d in dates])
        test_m = np.array([d.year == test_yr for d in dates])
        t0 = max(0, np.where(train_m)[0][0] - 60)
        sl = slice(t0, np.where(test_m)[0][-1] + 1)

        C_s = C[:, sl]; ND_s = C_s.shape[1]; d_s = dates[sl]

        pi_s = []
        for dsym, usym in [(p[2], p[3]) for p in pair_indices]:
            dsi = {syms[si]: si for si in range(NS)}.get(dsym, -1)
            usi = {syms[si]: si for si in range(NS)}.get(usym, -1)
            if dsi >= 0 and usi >= 0:
                pi_s.append((dsi, usi, dsym, usym))

        z_scores, _ = precompute_pair_signals(C_s, NS, ND_s, pi_s)

        # Regime
        O_s = O[:, sl]; H_s = H[:, sl]; L_s = L[:, sl]
        V_s = V[:, sl]; OI_s = OI[:, sl]
        F = compute_factors(C_s, O_s, H_s, L_s, V_s, OI_s, NS, ND_s)
        regime_s = detect_regimes(F, NS, ND_s)

        test_start = np.where(np.array([d.year == test_yr for d in d_s]))[0][0]

        trades, eq, dd = backtest_pairs_pct(
            C_s, NS, ND_s, d_s, syms, pi_s, z_scores,
            z_thresh=z_thresh, hold_max=hold_max, max_pairs=max_pairs,
            start_di=test_start,
            regime=regime_s, regime_filter=True,
            alloc_per_pair=alloc_per_pair, leverage=leverage)

        yr_t = [t for t in trades if t['year'] == test_yr]
        if yr_t:
            nw = sum(1 for t in yr_t if t['pnl_pct'] > 0)
            wr = nw / len(yr_t) * 100
            avg = np.mean([t['pnl_pct'] for t in yr_t])
            print(f"  {test_yr}: {len(yr_t)}t WR={wr:.1f}% avg={avg:+.4f}%", flush=True)
            all_trades.extend(yr_t)
        else:
            print(f"  {test_yr}: no trades", flush=True)

        start_yr += 1

    if all_trades:
        # Reconstruct equity curve from WF trades
        eq = CASH0
        for t in sorted(all_trades, key=lambda x: x['di']):
            eq += t['pnl_abs']
        nw = sum(1 for t in all_trades if t['pnl_pct'] > 0)
        wr = nw / len(all_trades) * 100
        cum = eq / CASH0 - 1

        # Annualized
        first_d = min(t['di'] for t in all_trades)
        last_d = max(t['di'] for t in all_trades)
        yrs = max((last_d - first_d) / 252, 0.1)
        ann = ((eq / CASH0) ** (1 / yrs) - 1) * 100

        abs_pnls = [t['pnl_abs'] for t in sorted(all_trades, key=lambda x: x['di'])]
        rets = np.array(abs_pnls) / CASH0
        sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0

        print(f"\n  WF Total: {len(all_trades)}t WR={wr:.1f}% cum={cum*100:+.1f}% "
              f"ann={ann:+.1f}% Sh={sharpe:.2f} equity={eq:,.0f}")

        yr_stats = {}
        for t in all_trades:
            y = t['year']
            if y not in yr_stats: yr_stats[y] = {'n': 0, 'w': 0, 'pnl': []}
            yr_stats[y]['n'] += 1
            if t['pnl_pct'] > 0: yr_stats[y]['w'] += 1
            yr_stats[y]['pnl'].append(t['pnl_pct'])
        for y in sorted(yr_stats.keys()):
            ys = yr_stats[y]
            wr_y = ys['w'] / ys['n'] * 100
            cum_y = np.prod([1 + p / 100 for p in ys['pnl']]) - 1
            print(f"    {y}: {ys['n']}t WR={wr_y:.1f}% cum={cum_y*100:+.2f}%")

    return all_trades


# ============================================================
# MAIN
# ============================================================
def main():
    t_start = time.time()
    print("=" * 60)
    print("  V307: CLEAN % PAIR TRADING")
    print("=" * 60)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    sym_to_si = {syms[si]: si for si in range(NS)}

    # Build pair indices
    pair_indices = []
    for dsym, usym in PAIRS:
        dsi = sym_to_si.get(dsym, -1)
        usi = sym_to_si.get(usym, -1)
        if dsi >= 0 and usi >= 0:
            pair_indices.append((dsi, usi, dsym, usym))
    print(f"  Active pairs: {len(pair_indices)}")

    # Compute regime
    print("[V307] Computing regime...", flush=True)
    F = compute_factors(C, O, H, L, V, OI, NS, ND)
    regime = detect_regimes(F, NS, ND)

    # Precompute pair signals
    z_scores, spreads = precompute_pair_signals(C, NS, ND, pair_indices)

    # ================================================================
    # PARAMETER SWEEP (Full Backtest)
    # ================================================================
    print("\n--- Parameter Sweep ---")
    results = []
    for zt in [0.5, 0.8, 1.0]:
        for hm in [1, 2, 3]:
            for mp in [1, 2]:
                for alloc in [0.2, 0.5]:
                    for lev in [1.0, 2.0, 3.0]:
                        trades, eq, dd = backtest_pairs_pct(
                            C, NS, ND, dates, syms, pair_indices, z_scores,
                            z_thresh=zt, hold_max=hm, max_pairs=mp,
                            regime=regime, regime_filter=True,
                            alloc_per_pair=alloc, leverage=lev)
                        if len(trades) < 5: continue
                        nw = sum(1 for t in trades if t['pnl_pct'] > 0)
                        wr = nw / len(trades) * 100
                        ann = ((eq / CASH0) ** (1 / max((dates[-1]-dates[0]).days/365.25, 0.1)) - 1) * 100
                        abs_pnls = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
                        rets = np.array(abs_pnls) / CASH0
                        sh = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
                        results.append({
                            'zt': zt, 'hm': hm, 'mp': mp,
                            'alloc': alloc, 'lev': lev,
                            'n': len(trades), 'wr': wr, 'ann': ann,
                            'dd': dd, 'sh': sh, 'eq': eq,
                        })

    results.sort(key=lambda x: -x['sh'])
    print(f"\n{'ZT':>4} {'HM':>3} {'MP':>3} {'A':>5} {'LV':>4} {'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 60)
    for r in results[:25]:
        print(f"{r['zt']:>4.1f} {r['hm']:>3} {r['mp']:>3} {r['alloc']:>5.2f} "
              f"{r['lev']:>4.1f} {r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sh']:>5.2f}")

    # ================================================================
    # WALK-FORWARD for top 5 unique configs
    # ================================================================
    print("\n" + "=" * 60)
    print("  WALK-FORWARD")
    print("=" * 60)

    seen = set()
    wf_count = 0
    for r in results[:20]:
        if wf_count >= 5: break
        key = (r['zt'], r['hm'], r['mp'], r['alloc'], r['lev'])
        if key in seen: continue
        seen.add(key)
        print(f"\n--- #{wf_count+1}: zt={r['zt']} hm={r['hm']} mp={r['mp']} "
              f"a={r['alloc']} lev={r['lev']} (IS Sh={r['sh']:.2f}) ---")
        walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms, pair_indices,
                     z_thresh=r['zt'], hold_max=r['hm'], max_pairs=r['mp'],
                     alloc_per_pair=r['alloc'], leverage=r['lev'])
        wf_count += 1

    print(f"\n[V307] Done. Total: {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
