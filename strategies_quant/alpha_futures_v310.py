"""
V310: Optimized Momentum — Maximize Returns
=============================================
Push the momentum signal to its limits:
1. ADX trend strength filter (only trade when trend is strong)
2. Volume confirmation (entry when volume > 1.5x average)
3. Ultra-concentrated portfolio (top 1-3)
4. Dynamic leverage: higher in trending, lower in choppy
5. Faster hold periods for more compounding
6. Carry overlay for position sizing boost
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


def load_term_structure_fast(start='2021-01-01'):
    ts_dir = os.path.abspath(TS_DIR)
    all_files = glob.glob(os.path.join(ts_dir, '*.json'))
    ts_data = {}
    for fp in all_files:
        try:
            with open(fp) as f: d = json.load(f)
            sym, date_str = d.get('symbol',''), d.get('date','')
            if sym and date_str: ts_data[(sym, date_str)] = d
        except: continue
    all_syms = sorted(set(k[0] for k in ts_data.keys()))
    all_dates = sorted(set(k[1] for k in ts_data.keys()))
    all_dates = [d for d in all_dates if pd.Timestamp(d) >= pd.Timestamp(start)]
    if not all_dates: return None
    NS, ND = len(all_syms), len(all_dates)
    sym_idx = {s: i for i, s in enumerate(all_syms)}
    date_idx = {d: i for i, d in enumerate(all_dates)}
    spread_pct = np.full((NS, ND), np.nan)
    for (sym, date_str), d in ts_data.items():
        if sym not in sym_idx or date_str not in date_idx: continue
        sp = d.get('total_spread_pct')
        if sp is not None: spread_pct[sym_idx[sym], date_idx[date_str]] = float(sp)
    carry_z = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(60, ND):
            w = spread_pct[si, di-60:di]
            v = w[~np.isnan(w)]
            if len(v) >= 20 and not np.isnan(spread_pct[si, di]):
                m, s = np.mean(v), np.std(v, ddof=1)
                if s > 1e-10: carry_z[si, di] = (spread_pct[si, di] - m) / s
    dates = [pd.Timestamp(d) for d in all_dates]
    return {'carry_z': carry_z, 'spread_pct': spread_pct,
            'dates': dates, 'syms': all_syms, 'NS': NS, 'ND': ND}


def compute_enhanced_factors(C, O, H, L, V, NS, ND):
    """Compute momentum + ADX + volume factors."""
    factors = {}

    # Momentum ranks (multi-period)
    mom_combined = np.full((NS, ND), np.nan)
    for di in range(60, ND):
        scores = []
        for period in [5, 10, 20]:
            rets = np.full(NS, np.nan)
            for si in range(NS):
                c0, c1 = C[si, di-period], C[si, di]
                if not np.isnan(c0) and not np.isnan(c1) and c0 > 0:
                    rets[si] = (c1 - c0) / c0
            valid = ~np.isnan(rets)
            if valid.sum() >= 5:
                r = pd.Series(rets).rank(pct=True, na_option='keep').values
                scores.append(np.nan_to_num(r, nan=0.5))
        if scores:
            mom_combined[:, di] = np.mean(scores, axis=0)
    factors['mom_rank'] = mom_combined

    # ADX (trend strength) — simplified
    adx = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(28, ND):
            # Simplified ADX: average directional movement
            tr_list = []
            plus_dm = []
            minus_dm = []
            for j in range(di-14, di):
                h, l, c_prev = H[si,j], L[si,j], C[si,j-1] if j > 0 else np.nan
                h_prev, l_prev = H[si,j-1] if j > 0 else np.nan, L[si,j-1] if j > 0 else np.nan
                if any(np.isnan([h, l])): continue
                tr = max(h - l, abs(h - c_prev) if not np.isnan(c_prev) else 0,
                         abs(l - c_prev) if not np.isnan(c_prev) else 0)
                tr_list.append(tr)
                if not np.isnan(h_prev) and not np.isnan(l_prev):
                    pdm = max(h - h_prev, 0)
                    mdm = max(l_prev - l, 0)
                    if pdm > mdm: plus_dm.append(pdm)
                    else: plus_dm.append(0)
                    if mdm > pdm: minus_dm.append(mdm)
                    else: minus_dm.append(0)
            if len(tr_list) >= 10:
                atr = np.mean(tr_list)
                if atr > 0:
                    pdi = np.mean(plus_dm) / atr * 100 if plus_dm else 0
                    mdi = np.mean(minus_dm) / atr * 100 if minus_dm else 0
                    dx = abs(pdi - mdi) / (pdi + mdi + 1e-10) * 100
                    adx[si, di] = dx
    factors['adx'] = adx

    # Volume anomaly (ratio of today's volume to 20-day average)
    vol_ratio = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            v_today = V[si, di]
            v_avg = np.nanmean(V[si, di-20:di])
            if not np.isnan(v_today) and not np.isnan(v_avg) and v_avg > 0:
                vol_ratio[si, di] = v_today / v_avg
    factors['vol_ratio'] = vol_ratio

    # RSI (14-day)
    rsi = np.full((NS, ND), np.nan)
    for si in range(NS):
        gains = []; losses = []
        for di in range(1, 15):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di-1]):
                chg = C[si, di] - C[si, di-1]
                gains.append(max(chg, 0))
                losses.append(max(-chg, 0))
        if gains:
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            for di in range(15, ND):
                if np.isnan(C[si, di]) or np.isnan(C[si, di-1]): continue
                chg = C[si, di] - C[si, di-1]
                avg_gain = (avg_gain * 13 + max(chg, 0)) / 14
                avg_loss = (avg_loss * 13 + max(-chg, 0)) / 14
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi[si, di] = 100 - 100 / (1 + rs)
                else:
                    rsi[si, di] = 100
    factors['rsi'] = rsi

    return factors


def backtest_optimized(C, O, H, L, NS, ND, dates, syms,
                       factors, ts_data, regime,
                       top_n=3, hold_days=3, atr_stop=2.5,
                       leverage=3.0, adx_thresh=20,
                       use_carry=True, use_vol=True,
                       start_di=60, end_di=None):
    if end_di is None: end_di = ND

    ts_sym_idx = {s: i for i, s in enumerate(ts_data['syms'])}
    ts_date_idx = {d: i for i, d in enumerate(ts_data['dates'])}

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions = []
    trades = []
    pos_alloc = leverage / max(top_n, 1)

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

        # Entry
        held = {p[0] for p in positions}
        if len(positions) >= top_n: continue

        # Dynamic leverage based on regime
        r = regime[di] if regime is not None and di < len(regime) else 0
        lev_mult = {1: 1.0, 0: 0.8, -1: 0.3, 2: 0.1}.get(r, 0.5)
        cur_alloc = pos_alloc * lev_mult

        candidates = []
        for si in range(NS):
            if si in held: continue
            if np.isnan(C[si, di]) or np.isnan(O[si, di]): continue

            # Momentum rank
            mom = factors['mom_rank'][si, di]
            if np.isnan(mom) or mom < 0.5: continue  # Only top half

            score = mom

            # ADX filter: prefer trending instruments
            adx_val = factors['adx'][si, di]
            if not np.isnan(adx_val):
                if adx_val < adx_thresh: continue  # Skip weak trends
                score += (adx_val - adx_thresh) / 100 * 0.3  # Bonus for strong trend

            # Volume confirmation
            if use_vol:
                vr = factors['vol_ratio'][si, di]
                if not np.isnan(vr):
                    if vr > 1.5: score += 0.1  # Volume breakout bonus

            # Carry bonus
            if use_carry:
                ts_di = ts_date_idx.get(d)
                if ts_di is not None:
                    sym = syms[si]
                    ts_si = ts_sym_idx.get(sym, -1)
                    if ts_si >= 0 and ts_di < ts_data['carry_z'].shape[1]:
                        cz = ts_data['carry_z'][ts_si, ts_di]
                        if not np.isnan(cz) and cz > 0:
                            score += 0.15 * min(cz / 2, 1)  # Carry bonus (capped)

            # RSI: avoid overbought
            rsi_val = factors['rsi'][si, di]
            if not np.isnan(rsi_val) and rsi_val > 75:
                score -= 0.2  # Penalty for overbought

            if score > 0.5:
                candidates.append((score, si))

        if not candidates: continue
        candidates.sort(key=lambda x: -x[0])

        for score, si in candidates[:top_n]:
            if len(positions) >= top_n or si in held: break
            op = O[si, di]
            if np.isnan(op) or op <= 0: continue
            atr_v = []
            for j in range(max(start_di, di - 14), di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
            if not atr_v: continue
            atr = np.mean(atr_v)
            positions.append((si, di, op, op - atr_stop * atr, 1, cur_alloc))
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
                 ts_data, regime,
                 top_n=3, hold_days=3, leverage=5.0,
                 adx_thresh=20, use_carry=True, use_vol=True,
                 train_years=3, test_years=1):
    years = sorted(set(d.year for d in dates))
    all_trades = []
    start_yr = max(years[0], 2021)

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

        C_s, O_s, H_s, L_s = C[:,sl], O[:,sl], H[:,sl], L[:,sl]
        V_s = V[:,sl]
        d_s = dates[sl]; ND_s = len(d_s)

        F = compute_factors(C_s, O_s, H_s, L_s, V_s, OI[:,sl], NS, ND_s)
        regime_s = detect_regimes(F, NS, ND_s)
        ef = compute_enhanced_factors(C_s, O_s, H_s, L_s, V_s, NS, ND_s)

        test_start = np.where(np.array([d.year == test_yr for d in d_s]))[0][0]

        trades, eq, dd = backtest_optimized(
            C_s, O_s, H_s, L_s, NS, ND_s, d_s, syms,
            ef, ts_data, regime_s,
            top_n=top_n, hold_days=hold_days, leverage=leverage,
            adx_thresh=adx_thresh, use_carry=use_carry, use_vol=use_vol,
            start_di=test_start)

        yr_t = [t for t in trades if t['year'] == test_yr]
        if yr_t:
            nw = sum(1 for t in yr_t if t['pnl_pct'] > 0)
            wr = nw / len(yr_t) * 100
            print(f"  {test_yr}: {len(yr_t)}t WR={wr:.1f}%", flush=True)
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
            if y not in yr_stats: yr_stats[y] = {'n':0,'w':0}
            yr_stats[y]['n'] += 1
            if t['pnl_pct'] > 0: yr_stats[y]['w'] += 1
        for y in sorted(yr_stats.keys()):
            ys = yr_stats[y]
            print(f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}%")

    return all_trades


def main():
    t_start = time.time()
    print("=" * 60)
    print("  V310: OPTIMIZED MOMENTUM — PUSH TO 600%")
    print("=" * 60)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    ts_data = load_term_structure_fast(start='2021-01-01')

    print("[V310] Computing factors...", flush=True)
    F = compute_factors(C, O, H, L, V, OI, NS, ND)
    regime = detect_regimes(F, NS, ND)
    ef = compute_enhanced_factors(C, O, H, L, V, NS, ND)

    # Full sweep
    print("\n--- Parameter Sweep ---")
    results = []
    for tn in [1, 2, 3, 5]:
        for hd in [2, 3, 5]:
            for lev in [3.0, 5.0, 8.0]:
                for adx_t in [15, 20, 25]:
                    for carry in [True, False]:
                        for vol in [True, False]:
                            trades, eq, dd = backtest_optimized(
                                C, O, H, L, NS, ND, dates, syms,
                                ef, ts_data, regime,
                                top_n=tn, hold_days=hd, leverage=lev,
                                adx_thresh=adx_t, use_carry=carry, use_vol=vol)
                            if len(trades) < 5: continue
                            nw = sum(1 for t in trades if t['pnl_pct'] > 0)
                            wr = nw / len(trades) * 100
                            ann = ((eq/CASH0)**(1/max(1.0,(trades[-1]['di']-trades[0]['di'])/252))-1)*100
                            abs_pnls = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
                            rets = np.array(abs_pnls) / CASH0
                            sh = np.mean(rets)/np.std(rets)*np.sqrt(252) if np.std(rets) > 0 else 0
                            results.append({
                                'tn':tn, 'hd':hd, 'lev':lev, 'adx':adx_t,
                                'carry':carry, 'vol':vol,
                                'n':len(trades), 'wr':wr, 'ann':ann,
                                'dd':dd, 'sh':sh, 'eq':eq,
                            })

    results.sort(key=lambda x: -x['sh'])
    print(f"\n{'TN':>3} {'HD':>3} {'LV':>4} {'ADX':>4} {'C':>2} {'V':>2} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 60)
    for r in results[:20]:
        print(f"{r['tn']:>3} {r['hd']:>3} {r['lev']:>4.1f} {r['adx']:>4} "
              f"{'Y' if r['carry'] else 'N':>2} {'Y' if r['vol'] else 'N':>2} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sh']:>5.2f}")

    # Also sort by annual return
    print("\n--- Sorted by Annual Return ---")
    results.sort(key=lambda x: -x['ann'])
    for r in results[:15]:
        print(f"  tn={r['tn']} hd={r['hd']} lev={r['lev']} adx={r['adx']} "
              f"c={'Y' if r['carry'] else 'N'} v={'Y' if r['vol'] else 'N'}: "
              f"{r['n']}t WR={r['wr']:.1f}% ann={r['ann']:+.1f}% DD={r['dd']:.1f}% Sh={r['sh']:.2f}")

    # Walk-forward for top configs
    print("\n" + "=" * 60)
    print("  WALK-FORWARD")
    print("=" * 60)

    # Sort back by Sharpe for WF
    results.sort(key=lambda x: -x['sh'])
    seen = set()
    wf_count = 0
    for r in results[:20]:
        if wf_count >= 5: break
        key = (r['tn'], r['hd'], r['lev'], r['adx'], r['carry'], r['vol'])
        if key in seen: continue
        seen.add(key)
        print(f"\n--- #{wf_count+1}: tn={r['tn']} hd={r['hd']} lev={r['lev']} "
              f"adx={r['adx']} c={'Y' if r['carry'] else 'N'} v={'Y' if r['vol'] else 'N'} "
              f"(IS Sh={r['sh']:.2f}, ann={r['ann']:+.1f}%) ---")
        walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms,
                     ts_data, regime,
                     top_n=r['tn'], hold_days=r['hd'], leverage=r['lev'],
                     adx_thresh=r['adx'], use_carry=r['carry'], use_vol=r['vol'])
        wf_count += 1

    print(f"\n[V310] Done. Total: {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
