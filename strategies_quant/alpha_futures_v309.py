"""
V309: Multi-Factor Carry + Momentum + Regime Fusion
=====================================================
Combines the three best alpha sources discovered in V300-V308:
1. Term structure carry (Sharpe ~1.03, from V308)
2. Cross-sectional momentum with regime awareness (Sharpe ~0.75, from V301/V303)
3. Adaptive position sizing based on market regime

Ensemble scoring approach:
- Score = w_carry * carry_z + w_mom * momentum_rank + w_regime * regime_mult
- Only long (no shorting — proven unprofitable in V302)
- Concentrated portfolio (top N by score)
- Walk-forward with proper equity curve tracking
"""
import sys, os, time, warnings, json, glob
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_v301 import load_all_data, compute_factors, detect_regimes

CASH0 = 1_000_000
COMM = 0.0005
TS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'data', 'futures_term_structure')


def load_term_structure(start='2021-01-01'):
    """Fast term structure loader."""
    ts_dir = os.path.abspath(TS_DIR)
    all_files = glob.glob(os.path.join(ts_dir, '*.json'))
    ts_data = {}
    for fp in all_files:
        try:
            with open(fp) as f:
                d = json.load(f)
            sym, date_str = d.get('symbol', ''), d.get('date', '')
            if sym and date_str:
                ts_data[(sym, date_str)] = d
        except: continue

    all_syms = sorted(set(k[0] for k in ts_data.keys()))
    all_dates = sorted(set(k[1] for k in ts_data.keys()))
    all_dates = [d for d in all_dates if pd.Timestamp(d) >= pd.Timestamp(start)]
    if not all_dates: return None

    NS, ND = len(all_syms), len(all_dates)
    sym_idx = {s: i for i, s in enumerate(all_syms)}
    date_idx = {d: i for i, d in enumerate(all_dates)}

    spread_pct = np.full((NS, ND), np.nan)
    structure = np.full((NS, ND), np.nan)
    curve_slope = np.full((NS, ND), np.nan)

    for (sym, date_str), d in ts_data.items():
        if sym not in sym_idx or date_str not in date_idx: continue
        si, di = sym_idx[sym], date_idx[date_str]
        sp = d.get('total_spread_pct')
        if sp is not None: spread_pct[si, di] = float(sp)
        struct = d.get('structure', '')
        if struct == 'backwardation': structure[si, di] = 1
        elif struct == 'contango': structure[si, di] = -1
        elif struct == 'flat': structure[si, di] = 0
        curve = d.get('curve', [])
        if curve and len(curve) >= 2:
            prices = [c.get('price', np.nan) for c in curve]
            months = list(range(len(prices)))
            valid = [(m, p) for m, p in zip(months, prices) if not np.isnan(p)]
            if len(valid) >= 2:
                ms, ps = np.array([v[0] for v in valid]), np.array([v[1] for v in valid])
                slope = np.polyfit(ms, ps, 1)[0]
                mean_p = np.mean(ps)
                if mean_p > 0: curve_slope[si, di] = slope / mean_p

    # Carry z-score
    carry_z = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(60, ND):
            w = spread_pct[si, di-60:di]
            v = w[~np.isnan(w)]
            if len(v) >= 20 and not np.isnan(spread_pct[si, di]):
                m, s = np.mean(v), np.std(v, ddof=1)
                if s > 1e-10: carry_z[si, di] = (spread_pct[si, di] - m) / s

    # Carry momentum 5-day
    carry_mom = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(spread_pct[si, di]) and not np.isnan(spread_pct[si, di-5]):
                carry_mom[si, di] = spread_pct[si, di] - spread_pct[si, di-5]

    dates = [pd.Timestamp(d) for d in all_dates]
    return {
        'spread_pct': spread_pct, 'structure': structure,
        'carry_z': carry_z, 'carry_mom': carry_mom,
        'curve_slope': curve_slope,
        'dates': dates, 'syms': all_syms, 'NS': NS, 'ND': ND,
    }


def compute_momentum_ranks(C, NS, ND):
    """Cross-sectional momentum ranks (V301-style)."""
    ranks = {}
    for period in [5, 10, 20, 60]:
        r = np.full((NS, ND), np.nan)
        for di in range(period, ND):
            rets = np.full(NS, np.nan)
            for si in range(NS):
                c0 = C[si, di - period]
                c1 = C[si, di]
                if not np.isnan(c0) and not np.isnan(c1) and c0 > 0:
                    rets[si] = (c1 - c0) / c0
            valid = ~np.isnan(rets)
            if valid.sum() >= 5:
                r[:, di] = pd.Series(rets).rank(pct=True, na_option='keep').values
        ranks[f'mom{period}'] = r

    # Combined momentum: average of ranks
    combined = np.full((NS, ND), np.nan)
    for di in range(60, ND):
        scores = []
        for name in ranks:
            vals = ranks[name][:, di]
            if not np.isnan(vals).all():
                scores.append(np.nan_to_num(vals, nan=0.5))
        if scores:
            combined[:, di] = np.mean(scores, axis=0)
    ranks['combined'] = combined
    return ranks


def backtest_multi_factor(C, O, H, L, NS, ND, dates, syms,
                          momentum_ranks, ts_data,
                          regime=None,
                          w_carry=0.4, w_mom=0.4, w_struct=0.2,
                          top_n=5, hold_days=5, atr_stop=2.5,
                          leverage=1.0, start_di=60, end_di=None):
    """Multi-factor backtest with carry + momentum + regime."""
    if end_di is None: end_di = ND

    # Map TS symbols to price symbols
    price_sym_set = set(syms[i] for i in range(NS))
    ts_sym_idx = {s: i for i, s in enumerate(ts_data['syms'])}
    ts_date_idx = {d: i for i, d in enumerate(ts_data['dates'])}

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions = []
    trades = []
    max_pos = top_n
    pos_alloc = leverage / max_pos

    for di in range(max(start_di, 1), end_di):
        d = dates[di]

        # Exit management
        daily_pnl = 0
        new_positions = []
        for si, edi, ep, sp, dr, a in positions:
            c = C[si, di]
            if np.isnan(c):
                new_positions.append((si, edi, ep, sp, dr, a)); continue
            exit_r = None
            if dr > 0 and c < sp: exit_r = 'stop'
            elif di - edi >= hold_days: exit_r = 'hold'
            if exit_r:
                pnl = dr * (c - ep) / ep - COMM
                profit = equity * a * pnl
                daily_pnl += profit
                trades.append({
                    'pnl_abs': profit, 'pnl_pct': pnl * 100,
                    'days': di - edi, 'di': di, 'year': d.year,
                    'sym': syms[si], 'reason': exit_r,
                })
            else:
                new_positions.append((si, edi, ep, sp, dr, a))
        positions = new_positions
        equity += daily_pnl
        if equity > peak: peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100
            if dd > max_dd: max_dd = dd
        if equity <= 0: break

        # Entry scoring
        held = {p[0] for p in positions}
        if len(positions) >= max_pos: continue

        # Regime sizing
        r = regime[di] if regime is not None and di < len(regime) else 0
        size_mult = {1: 1.0, 0: 0.8, -1: 0.5, 2: 0.2}.get(r, 0.5)

        # Get TS date index
        ts_di = ts_date_idx.get(d)

        candidates = []
        for si in range(NS):
            if si in held: continue
            if np.isnan(C[si, di]) or np.isnan(O[si, di]): continue

            score = 0
            weight_sum = 0

            # Momentum component
            mom_val = momentum_ranks['combined'][si, di]
            if not np.isnan(mom_val):
                score += w_mom * mom_val
                weight_sum += w_mom

            # Carry component
            if ts_di is not None:
                sym = syms[si]
                ts_si = ts_sym_idx.get(sym, -1)
                if ts_si >= 0 and ts_di < ts_data['carry_z'].shape[1]:
                    cz = ts_data['carry_z'][ts_si, ts_di]
                    if not np.isnan(cz):
                        # Normalize z-score to 0-1 range (sigmoid)
                        carry_signal = 1 / (1 + np.exp(-cz))
                        score += w_carry * carry_signal
                        weight_sum += w_carry

                    # Structure component (backwardation bonus)
                    struct = ts_data['structure'][ts_si, ts_di]
                    if not np.isnan(struct):
                        struct_signal = (struct + 1) / 2  # Map [-1,1] to [0,1]
                        score += w_struct * struct_signal
                        weight_sum += w_struct

            if weight_sum > 0:
                score /= weight_sum  # Normalize
                if score > 0.4:  # Minimum threshold
                    candidates.append((score, si))

        if not candidates: continue
        candidates.sort(key=lambda x: -x[0])

        alloc = pos_alloc * size_mult
        for score, si in candidates[:top_n]:
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
    for si, edi, ep, sp, dr, a in positions:
        c = C[si, ND - 1]
        if not np.isnan(c):
            pnl = dr * (c - ep) / ep - COMM
            profit = equity * a * pnl
            equity += profit
            trades.append({
                'pnl_abs': profit, 'pnl_pct': pnl * 100,
                'days': ND - 1 - edi, 'di': ND - 1,
                'year': dates[-1].year, 'sym': syms[si], 'reason': 'end',
            })

    return trades, equity, max_dd


def analyze(trades, equity, max_dd, label=""):
    if not trades:
        print(f"  {label}: no trades"); return None
    nw = sum(1 for t in trades if t['pnl_pct'] > 0)
    wr = nw / len(trades) * 100
    ann = ((equity / CASH0) ** (1 / max(1.0, (trades[-1]['di'] - trades[0]['di']) / 252)) - 1) * 100
    abs_pnls = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
    rets = np.array(abs_pnls) / CASH0
    sh = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0

    print(f"  {label}: {len(trades)}t WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
          f"Sh={sh:.2f} equity={equity:,.0f}")

    yr_stats = {}
    for t in trades:
        y = t['year']
        if y not in yr_stats: yr_stats[y] = {'n':0,'w':0,'pnl':[]}
        yr_stats[y]['n'] += 1
        if t['pnl_pct'] > 0: yr_stats[y]['w'] += 1
        yr_stats[y]['pnl'].append(t['pnl_pct'])
    for y in sorted(yr_stats.keys()):
        ys = yr_stats[y]
        wr_y = ys['w'] / ys['n'] * 100
        cum_y = np.prod([1+p/100 for p in ys['pnl']]) - 1
        print(f"    {y}: {ys['n']}t WR={wr_y:.1f}% cum={cum_y*100:+.1f}%")
    return {'n': len(trades), 'wr': wr, 'dd': max_dd, 'ann': ann, 'sh': sh}


def walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms,
                 momentum_ranks, ts_data, regime,
                 w_carry=0.4, w_mom=0.4, w_struct=0.2,
                 top_n=5, hold_days=5, atr_stop=2.5, leverage=2.0,
                 train_years=3, test_years=1):
    years = sorted(set(d.year for d in dates))
    all_trades = []
    start_yr = max(years[0], 2021)  # TS data starts 2021

    print(f"\nWF: wc={w_carry} wm={w_mom} ws={w_struct} tn={top_n} hd={hold_days} "
          f"lev={leverage} train={train_years}y", flush=True)

    while True:
        train_end = start_yr + train_years - 1
        test_yr = train_end + 1
        if test_yr > years[-1]: break

        train_m = np.array([d.year <= train_end for d in dates])
        test_m = np.array([d.year == test_yr for d in dates])
        if not train_m.any() or not test_m.any():
            start_yr += 1; continue

        t0 = max(0, np.where(train_m)[0][0] - 60)
        sl = slice(t0, np.where(test_m)[0][-1] + 1)

        C_s = C[:, sl]; O_s = O[:, sl]; H_s = H[:, sl]; L_s = L[:, sl]
        V_s = V[:, sl]; OI_s = OI[:, sl]
        d_s = dates[sl]; ND_s = len(d_s)

        # Recompute factors for this window
        F = compute_factors(C_s, O_s, H_s, L_s, V_s, OI_s, NS, ND_s)
        regime_s = detect_regimes(F, NS, ND_s)

        # Recompute momentum ranks
        mom_s = compute_momentum_ranks(C_s, NS, ND_s)

        test_start = np.where(np.array([d.year == test_yr for d in d_s]))[0][0]

        trades, eq, dd = backtest_multi_factor(
            C_s, O_s, H_s, L_s, NS, ND_s, d_s, syms,
            mom_s, ts_data, regime_s,
            w_carry=w_carry, w_mom=w_mom, w_struct=w_struct,
            top_n=top_n, hold_days=hold_days, atr_stop=atr_stop,
            leverage=leverage, start_di=test_start)

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
        eq = CASH0
        for t in sorted(all_trades, key=lambda x: x['di']):
            eq += t['pnl_abs']
        nw = sum(1 for t in all_trades if t['pnl_pct'] > 0)
        wr = nw / len(all_trades) * 100
        first_d = min(t['di'] for t in all_trades)
        last_d = max(t['di'] for t in all_trades)
        yrs = max((last_d - first_d) / 252, 0.1)
        ann = ((eq / CASH0) ** (1 / yrs) - 1) * 100
        abs_pnls = [t['pnl_abs'] for t in sorted(all_trades, key=lambda x: x['di'])]
        rets = np.array(abs_pnls) / CASH0
        sh = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0

        print(f"\n  WF Total: {len(all_trades)}t WR={wr:.1f}% ann={ann:+.1f}% "
              f"Sh={sh:.2f} equity={eq:,.0f}")

        yr_stats = {}
        for t in all_trades:
            y = t['year']
            if y not in yr_stats: yr_stats[y] = {'n':0,'w':0,'pnl':[]}
            yr_stats[y]['n'] += 1
            if t['pnl_pct'] > 0: yr_stats[y]['w'] += 1
            yr_stats[y]['pnl'].append(t['pnl_pct'])
        for y in sorted(yr_stats.keys()):
            ys = yr_stats[y]
            wr_y = ys['w'] / ys['n'] * 100
            cum_y = np.prod([1+p/100 for p in ys['pnl']]) - 1
            print(f"    {y}: {ys['n']}t WR={wr_y:.1f}% cum={cum_y*100:+.1f}%")

    return all_trades


def main():
    t_start = time.time()
    print("=" * 60)
    print("  V309: MULTI-FACTOR CARRY + MOMENTUM + REGIME")
    print("=" * 60)

    # Load data
    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    print(f"  Price: {NS} sym, {ND} days")

    # Load term structure
    ts_data = load_term_structure(start='2021-01-01')
    if ts_data is None:
        print("ERROR: No term structure data"); return
    print(f"  TS: {ts_data['NS']} sym, {ts_data['ND']} days")

    # Compute regime
    print("[V309] Computing regime...", flush=True)
    F = compute_factors(C, O, H, L, V, OI, NS, ND)
    regime = detect_regimes(F, NS, ND)

    # Compute momentum ranks
    print("[V309] Computing momentum...", flush=True)
    t0 = time.time()
    mom_ranks = compute_momentum_ranks(C, NS, ND)
    print(f"  Momentum ranks ({time.time()-t0:.1f}s)")

    # ================================================================
    # PARAMETER SWEEP
    # ================================================================
    print("\n--- Parameter Sweep ---")
    results = []
    for wc, wm, ws in [(0.5, 0.3, 0.2), (0.4, 0.4, 0.2), (0.3, 0.5, 0.2),
                         (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.5, 0.5, 0.0),
                         (0.3, 0.3, 0.4), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)]:
        for tn in [3, 5]:
            for hd in [3, 5]:
                for lev in [1.0, 2.0, 3.0, 5.0]:
                    trades, eq, dd = backtest_multi_factor(
                        C, O, H, L, NS, ND, dates, syms,
                        mom_ranks, ts_data, regime,
                        w_carry=wc, w_mom=wm, w_struct=ws,
                        top_n=tn, hold_days=hd, leverage=lev)
                    if len(trades) < 5: continue
                    nw = sum(1 for t in trades if t['pnl_pct'] > 0)
                    wr = nw / len(trades) * 100
                    ann = ((eq / CASH0) ** (1 / max(1.0, (trades[-1]['di'] - trades[0]['di']) / 252)) - 1) * 100
                    abs_pnls = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
                    rets = np.array(abs_pnls) / CASH0
                    sh = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
                    results.append({
                        'wc': wc, 'wm': wm, 'ws': ws,
                        'tn': tn, 'hd': hd, 'lev': lev,
                        'n': len(trades), 'wr': wr, 'ann': ann,
                        'dd': dd, 'sh': sh, 'eq': eq,
                    })

    results.sort(key=lambda x: -x['sh'])
    print(f"\n{'WC':>4} {'WM':>4} {'WS':>4} {'TN':>3} {'HD':>3} {'LV':>4} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 65)
    for r in results[:25]:
        print(f"{r['wc']:>4.1f} {r['wm']:>4.1f} {r['ws']:>4.1f} "
              f"{r['tn']:>3} {r['hd']:>3} {r['lev']:>4.1f} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sh']:>5.2f}")

    # ================================================================
    # WALK-FORWARD for top configs
    # ================================================================
    print("\n" + "=" * 60)
    print("  WALK-FORWARD VALIDATION")
    print("=" * 60)

    seen = set()
    wf_count = 0
    for r in results[:15]:
        if wf_count >= 5: break
        key = (r['wc'], r['wm'], r['ws'], r['tn'], r['hd'], r['lev'])
        if key in seen: continue
        seen.add(key)
        print(f"\n--- #{wf_count+1}: wc={r['wc']} wm={r['wm']} ws={r['ws']} "
              f"tn={r['tn']} hd={r['hd']} lev={r['lev']} (IS Sh={r['sh']:.2f}) ---")
        walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms,
                     mom_ranks, ts_data, regime,
                     w_carry=r['wc'], w_mom=r['wm'], w_struct=r['ws'],
                     top_n=r['tn'], hold_days=r['hd'], leverage=r['lev'])
        wf_count += 1

    print(f"\n[V309] Done. Total: {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
