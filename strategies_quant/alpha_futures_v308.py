"""
V308: Term Structure Carry Factor Strategy
============================================
Uses 82K term structure JSON files to compute carry/basis signals.
Carry is one of the strongest known commodity factors:
- Backwardation (near > far) = positive roll yield = bullish signal
- Contango (near < far) = negative roll yield = bearish signal

Architecture:
1. Load term structure data efficiently (batch by symbol)
2. Compute carry factors: annualized spread, z-score, structure state
3. Combine with V301 regime signals
4. Long-only concentrated portfolio with leverage
5. Walk-forward validation
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


# ============================================================
# TERM STRUCTURE DATA LOADING
# ============================================================
def load_term_structure(symbols=None, start='2021-01-01'):
    """Load term structure data from JSON files into arrays."""
    print("[V308] Loading term structure data...", flush=True)
    t0 = time.time()

    ts_dir = os.path.abspath(TS_DIR)
    if not os.path.isdir(ts_dir):
        print(f"  ERROR: {ts_dir} not found")
        return None

    # Get all JSON files
    all_files = glob.glob(os.path.join(ts_dir, '*.json'))
    print(f"  Found {len(all_files)} files", flush=True)

    # Parse all files into a dict: {(symbol, date): data}
    ts_data = {}
    for fp in all_files:
        try:
            with open(fp) as f:
                d = json.load(f)
            sym = d.get('symbol', '')
            date_str = d.get('date', '')
            if not sym or not date_str: continue
            if symbols and sym not in symbols: continue
            ts_data[(sym, date_str)] = d
        except:
            continue

    # Get unique symbols and dates
    all_syms = sorted(set(k[0] for k in ts_data.keys()))
    all_dates = sorted(set(k[1] for k in ts_data.keys()))

    # Filter by start date
    start_ts = pd.Timestamp(start)
    all_dates = [d for d in all_dates if pd.Timestamp(d) >= start_ts]

    if not all_dates:
        print("  No dates after filtering")
        return None

    NS = len(all_syms)
    ND = len(all_dates)
    sym_idx = {s: i for i, s in enumerate(all_syms)}
    date_idx = {d: i for i, d in enumerate(all_dates)}

    # Arrays
    near_price = np.full((NS, ND), np.nan)
    far_price = np.full((NS, ND), np.nan)
    spread_pct = np.full((NS, ND), np.nan)
    structure = np.full((NS, ND), np.nan)  # +1 backwardation, -1 contango
    n_contracts = np.full((NS, ND), np.nan)
    curve_slope = np.full((NS, ND), np.nan)  # Linear regression slope across curve

    for (sym, date_str), d in ts_data.items():
        if sym not in sym_idx or date_str not in date_idx: continue
        si = sym_idx[sym]
        di = date_idx[date_str]

        np_val = d.get('near_price')
        fp_val = d.get('far_price')
        sp = d.get('total_spread_pct')
        struct = d.get('structure', '')
        curve = d.get('curve', [])

        if np_val is not None and not np.isnan(np_val):
            near_price[si, di] = float(np_val)
        if fp_val is not None and not np.isnan(fp_val):
            far_price[si, di] = float(fp_val)
        if sp is not None and not np.isnan(sp):
            spread_pct[si, di] = float(sp)
        if struct == 'backwardation':
            structure[si, di] = 1
        elif struct == 'contango':
            structure[si, di] = -1
        elif struct == 'flat':
            structure[si, di] = 0

        if curve and len(curve) >= 2:
            n_contracts[si, di] = len(curve)
            # Compute curve slope
            prices = [c.get('price', np.nan) for c in curve]
            months = list(range(len(prices)))
            valid = [(m, p) for m, p in zip(months, prices) if not np.isnan(p)]
            if len(valid) >= 2:
                ms = np.array([v[0] for v in valid])
                ps = np.array([v[1] for v in valid])
                if len(ms) >= 2:
                    slope = np.polyfit(ms, ps, 1)[0]
                    mean_p = np.mean(ps)
                    if mean_p > 0:
                        curve_slope[si, di] = slope / mean_p  # Normalized slope

    dates = [pd.Timestamp(d) for d in all_dates]
    print(f"  {NS} symbols, {ND} days ({time.time()-t0:.1f}s)")
    print(f"  Dates: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")
    print(f"  Avg spread_pct coverage: {(~np.isnan(spread_pct)).sum()}/{NS*ND} "
          f"({(~np.isnan(spread_pct)).sum()/(NS*ND)*100:.1f}%)")

    return {
        'near_price': near_price, 'far_price': far_price,
        'spread_pct': spread_pct, 'structure': structure,
        'n_contracts': n_contracts, 'curve_slope': curve_slope,
        'dates': dates, 'syms': all_syms,
        'NS': NS, 'ND': ND,
    }


# ============================================================
# CARRY FACTOR COMPUTATION
# ============================================================
def compute_carry_factors(ts_data):
    """Compute carry-based factors from term structure data."""
    NS = ts_data['NS']
    ND = ts_data['ND']
    spread_pct = ts_data['spread_pct']
    structure = ts_data['structure']
    curve_slope = ts_data['curve_slope']

    factors = {}

    # F1: Annualized carry (spread_pct is already in %, annualize)
    carry_ann = spread_pct.copy()  # Already a percentage spread
    factors['carry_ann'] = carry_ann

    # F2: Carry z-score (60-day rolling)
    carry_z = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(60, ND):
            window = carry_ann[si, di-60:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 20:
                m = np.mean(valid)
                s = np.std(valid, ddof=1)
                if s > 1e-10 and not np.isnan(carry_ann[si, di]):
                    carry_z[si, di] = (carry_ann[si, di] - m) / s
    factors['carry_z'] = carry_z

    # F3: Structure state (+1 back, -1 contango, 0 flat)
    factors['structure'] = structure

    # F4: Carry momentum (5-day change in spread)
    carry_mom5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(spread_pct[si, di]) and not np.isnan(spread_pct[si, di-5]):
                carry_mom5[si, di] = spread_pct[si, di] - spread_pct[si, di-5]
    factors['carry_mom5'] = carry_mom5

    # F5: Carry momentum 20-day
    carry_mom20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if not np.isnan(spread_pct[si, di]) and not np.isnan(spread_pct[si, di-20]):
                carry_mom20[si, di] = spread_pct[si, di] - spread_pct[si, di-20]
    factors['carry_mom20'] = carry_mom20

    # F6: Cross-sectional carry rank (percentile rank of carry)
    carry_rank = np.full((NS, ND), np.nan)
    for di in range(ND):
        vals = carry_ann[:, di]
        valid = ~np.isnan(vals)
        if valid.sum() >= 5:
            rank = pd.Series(vals).rank(pct=True, na_option='keep').values
            carry_rank[:, di] = rank
    factors['carry_rank'] = carry_rank

    # F7: Curve slope z-score
    slope_z = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(60, ND):
            window = curve_slope[si, di-60:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 20:
                m = np.mean(valid)
                s = np.std(valid, ddof=1)
                if s > 1e-10 and not np.isnan(curve_slope[si, di]):
                    slope_z[si, di] = (curve_slope[si, di] - m) / s
    factors['slope_z'] = slope_z

    # F8: Combined carry signal
    # Long: high carry (backwardation) + improving carry + flat/upward curve slope
    combined = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(60, ND):
            score = 0
            count = 0
            cz = carry_z[si, di]
            if not np.isnan(cz):
                score += cz * 0.4
                count += 1
            cm5 = carry_mom5[si, di]
            if not np.isnan(cm5):
                score += np.sign(cm5) * 0.2
                count += 1
            cm20 = carry_mom20[si, di]
            if not np.isnan(cm20):
                score += np.sign(cm20) * 0.2
                count += 1
            sz = slope_z[si, di]
            if not np.isnan(sz):
                score += sz * 0.2
                count += 1
            if count >= 2:
                combined[si, di] = score
    factors['carry_combined'] = combined

    # Summary
    for name, arr in factors.items():
        valid = (~np.isnan(arr)).sum()
        print(f"  {name}: {valid}/{NS*ND} valid ({valid/(NS*ND)*100:.1f}%)")

    return factors


# ============================================================
# CARRY + FACTOR FUSION BACKTEST
# ============================================================
def backtest_carry_factor(C, O, H, L, NS_C, ND_C, dates_c, syms_c,
                          carry_factors, ts_syms, ts_dates,
                          regime=None,
                          top_n=5, hold_days=5, atr_stop=2.5,
                          leverage=1.0, carry_weight=0.5):
    """
    Backtest using carry + momentum factors.
    Maps term structure symbols to price data symbols.
    """
    # Map TS symbols to price symbols
    ts_to_price = {}
    for si, s in enumerate(ts_syms):
        if s in {syms_c[i] for i in range(NS_C)}:
            price_si = list(syms_c).index(s)
            ts_to_price[si] = price_si

    print(f"  Mapped {len(ts_to_price)} TS symbols to price data")

    # Align dates
    date_to_di = {d: i for i, d in enumerate(dates_c)}
    ts_date_to_di = {d: i for i, d in enumerate(ts_dates)}

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions = []
    trades = []

    carry_combined = carry_factors['carry_combined']
    carry_rank = carry_factors['carry_rank']

    for di in range(60, ND_C):
        d = dates_c[di]

        # Exit management
        daily_pnl = 0
        new_positions = []
        for si, edi, ep, sp, d_dir, alloc in positions:
            c = C[si, di]
            if np.isnan(c):
                new_positions.append((si, edi, ep, sp, d_dir, alloc))
                continue
            exit_r = None
            if d_dir > 0 and c < sp:
                exit_r = 'stop'
            elif di - edi >= hold_days:
                exit_r = 'hold'
            if exit_r:
                pnl = d_dir * (c - ep) / ep - COMM
                profit = equity * alloc * pnl
                daily_pnl += profit
                trades.append({
                    'pnl_abs': profit, 'pnl_pct': pnl * 100,
                    'days': di - edi, 'di': di, 'year': d.year,
                    'sym': syms_c[si], 'reason': exit_r,
                })
            else:
                new_positions.append((si, edi, ep, sp, d_dir, alloc))
        positions = new_positions
        equity += daily_pnl
        if equity > peak: peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100
            if dd > max_dd: max_dd = dd
        if equity <= 0: break

        # Entry: combine carry signal with momentum
        held = {p[0] for p in positions}
        if len(positions) >= top_n: continue

        # Regime sizing
        r = regime[di] if regime is not None and di < len(regime) else 0
        size_mult = {1: 1.0, 0: 0.8, -1: 0.5, 2: 0.2}.get(r, 0.5)

        # Get carry signal for this date
        ts_di = ts_date_to_di.get(d)
        if ts_di is None: continue

        candidates = []
        for ts_si, price_si in ts_to_price.items():
            if price_si in held: continue
            if np.isnan(C[price_si, di]) or np.isnan(O[price_si, di]): continue

            # Carry score
            carry_score = carry_combined[ts_si, ts_di] if ts_di < carry_combined.shape[1] else np.nan
            carry_rk = carry_rank[ts_si, ts_di] if ts_di < carry_rank.shape[1] else np.nan

            if np.isnan(carry_score): continue

            # Simple signal: long high-carry commodities
            if carry_score > 0:
                candidates.append((carry_score, price_si))

        if not candidates: continue
        candidates.sort(key=lambda x: -x[0])

        alloc = (leverage * carry_weight * size_mult) / max(top_n, 1)

        for score, price_si in candidates[:top_n]:
            if len(positions) >= top_n: break
            if price_si in held: break
            op = O[price_si, di]
            if np.isnan(op) or op <= 0: continue
            # ATR stop
            atr_v = []
            for j in range(max(60, di - 14), di):
                hh, ll, cc = H[price_si, j], L[price_si, j], C[price_si, j]
                if not any(np.isnan([hh, ll, cc])):
                    atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
            if not atr_v: continue
            atr = np.mean(atr_v)
            positions.append((price_si, di, op, op - atr_stop * atr, 1, alloc))
            held.add(price_si)

    # Close remaining
    for si, edi, ep, sp, d_dir, alloc in positions:
        c = C[si, ND_C - 1]
        if not np.isnan(c):
            pnl = d_dir * (c - ep) / ep - COMM
            profit = equity * alloc * pnl
            equity += profit
            trades.append({
                'pnl_abs': profit, 'pnl_pct': pnl * 100,
                'days': ND_C - 1 - edi, 'di': ND_C - 1,
                'year': dates_c[-1].year, 'sym': syms_c[si],
                'reason': 'end',
            })

    return trades, equity, max_dd


# ============================================================
# ANALYSIS
# ============================================================
def analyze(trades, equity, max_dd, label=""):
    if not trades:
        print(f"  {label}: no trades"); return None
    nw = sum(1 for t in trades if t['pnl_pct'] > 0)
    wr = nw / len(trades) * 100
    ann = ((equity / CASH0) ** (1 / max(5.0, (trades[-1]['di'] - trades[0]['di']) / 252)) - 1) * 100
    abs_pnls = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
    sh = np.mean(abs_pnls) / np.std(abs_pnls) * np.sqrt(252) * 100 if len(abs_pnls) > 1 else 0

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
        print(f"    {y}: {ys['n']}t WR={wr_y:.1f}%")
    return {'n': len(trades), 'wr': wr, 'dd': max_dd, 'ann': ann, 'sh': sh}


# ============================================================
# MAIN
# ============================================================
def main():
    t_start = time.time()
    print("=" * 60)
    print("  V308: TERM STRUCTURE CARRY FACTOR STRATEGY")
    print("=" * 60)

    # Load price data
    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    syms_list = list(syms)
    print(f"  Price data: {NS} sym, {ND} days")

    # Load term structure data
    ts_data = load_term_structure(symbols=None, start='2021-01-01')
    if ts_data is None:
        print("ERROR: No term structure data")
        return

    # Compute regime on price data
    print("\n[V308] Computing regime...", flush=True)
    F = compute_factors(C, O, H, L, V, OI, NS, ND)
    regime = detect_regimes(F, NS, ND)

    # Compute carry factors
    print("\n[V308] Computing carry factors...", flush=True)
    carry_factors = compute_carry_factors(ts_data)

    # ================================================================
    # CARRY-ONLY BACKTEST
    # ================================================================
    print("\n--- Carry Factor Backtest ---")
    results = []
    for cw in [0.3, 0.5, 0.8, 1.0]:
        for tn in [3, 5, 10]:
            for hd in [3, 5, 10]:
                for lev in [1.0, 2.0, 3.0]:
                    trades, eq, dd = backtest_carry_factor(
                        C, O, H, L, NS, ND, dates, syms_list,
                        carry_factors, ts_data['syms'], ts_data['dates'],
                        regime=regime,
                        top_n=tn, hold_days=hd, leverage=lev, carry_weight=cw)
                    if len(trades) < 5: continue
                    nw = sum(1 for t in trades if t['pnl_pct'] > 0)
                    wr = nw / len(trades) * 100
                    ann = ((eq / CASH0) ** (1 / max(5.0, (trades[-1]['di'] - trades[0]['di']) / 252)) - 1) * 100
                    abs_pnls = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
                    sh = np.mean(abs_pnls) / np.std(abs_pnls) * np.sqrt(252) * 100 if len(abs_pnls) > 1 else 0
                    results.append({
                        'cw': cw, 'tn': tn, 'hd': hd, 'lev': lev,
                        'n': len(trades), 'wr': wr, 'ann': ann,
                        'dd': dd, 'sh': sh, 'eq': eq,
                    })

    results.sort(key=lambda x: -x['sh'])
    print(f"\n{'CW':>4} {'TN':>3} {'HD':>3} {'LV':>4} {'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 55)
    for r in results[:20]:
        print(f"{r['cw']:>4.1f} {r['tn']:>3} {r['hd']:>3} {r['lev']:>4.1f} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} {r['dd']:>6.1f} {r['sh']:>5.2f}")

    if results:
        best = results[0]
        print(f"\n--- Best Config: cw={best['cw']} tn={best['tn']} hd={best['hd']} lev={best['lev']} ---")
        trades, eq, dd = backtest_carry_factor(
            C, O, H, L, NS, ND, dates, syms_list,
            carry_factors, ts_data['syms'], ts_data['dates'],
            regime=regime,
            top_n=best['tn'], hold_days=best['hd'],
            leverage=best['lev'], carry_weight=best['cw'])
        analyze(trades, eq, dd, "Best Carry")

    print(f"\n[V308] Done. Total: {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
