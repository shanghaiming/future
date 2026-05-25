"""
Alpha V7b — Enhanced Factor Interactions & Conditional Filters
===============================================================
V7发现: 单因子全负, Best5(R²+VOL_PCT+DD52W+TENSION+SHADOW)组合+14.4%
但DD=100%, 年度不一致(2017+499% vs 2023-172%)

改进方向:
  1. 因子交互项: DRAWDOWN_52W * VOL_ANOMALY (超跌+放量)
  2. 条件过滤: R_SQUARED > 阈值 时才计算MOM5
  3. ATR trailing stop 止损
  4. 更激进的集中: Top 3
  5. 多空对冲: 同时做多top N + 做空bottom N
  6. 因子衰减: 近期因子变化更重要
  7. 行业中性: 在同行业内排名
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.data_loader import list_available_symbols, load_stock_data
from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7 import compute_all_factors, backtest_v7, COMMISSION, STAMP_DUTY, CASH0


def compute_interaction_factors(factors, NS, ND, C, O, H, L, V):
    """Compute interaction and conditional factors from base factors."""
    print("[Factors] Computing interaction factors...", flush=True)
    t0 = time.time()

    new_factors = {}

    # === 1. DRAWDOWN * VOL_ANOMALY (oversold + volume surge = institutional buying) ===
    dd = factors.get('R_DRAWDOWN_52W')
    va = factors.get('R_VOL_ANOMALY')
    if dd is not None and va is not None:
        interact = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(253, ND):
                if not np.isnan(dd[si, di]) and not np.isnan(va[si, di]):
                    interact[si, di] = dd[si, di] * va[si, di]
        # Rank
        from alpha_v7 import compute_all_factors
        # Actually, let's inline the rank function
        def rank_pct(arr, start=60):
            res = np.full_like(arr, np.nan)
            for di in range(start, arr.shape[1]):
                vals = arr[:, di]
                mask = ~np.isnan(vals)
                if mask.sum() < 50:
                    continue
                ranked = np.argsort(np.argsort(vals[mask])).astype(float)
                n = len(ranked)
                pct = ranked / max(n - 1, 1) * 100
                for k, idx in enumerate(np.where(mask)[0]):
                    res[idx, di] = pct[k]
            return res
        new_factors['R_DD_VOL'] = rank_pct(interact)
    print(f"  DD*VOL done ({time.time()-t0:.1f}s)", flush=True)

    # === 2. Conditional MOM5: only when R_SQUARED > 50 (clean trend) ===
    r_sq = factors.get('R_R_SQUARED')
    mom5 = factors.get('R_MOM5')
    if r_sq is not None and mom5 is not None:
        cond_mom = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(25, ND):
                if not np.isnan(r_sq[si, di]) and r_sq[si, di] > 50:
                    if not np.isnan(mom5[si, di]):
                        cond_mom[si, di] = mom5[si, di]
        new_factors['R_COND_MOM5'] = cond_mom  # Already ranked (from mom5)
    print(f"  Conditional MOM5 done ({time.time()-t0:.1f}s)", flush=True)

    # === 3. BODY_RATIO * VOL_ANOMALY (strong candle + volume = conviction) ===
    br = factors.get('R_BODY_RATIO')
    if br is not None and va is not None:
        interact2 = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(25, ND):
                if not np.isnan(br[si, di]) and not np.isnan(va[si, di]):
                    interact2[si, di] = br[si, di] * va[si, di]
        def rank_pct(arr, start=60):
            res = np.full_like(arr, np.nan)
            for di in range(start, arr.shape[1]):
                vals = arr[:, di]
                mask = ~np.isnan(vals)
                if mask.sum() < 50:
                    continue
                ranked = np.argsort(np.argsort(vals[mask])).astype(float)
                n = len(ranked)
                pct = ranked / max(n - 1, 1) * 100
                for k, idx in enumerate(np.where(mask)[0]):
                    res[idx, di] = pct[k]
            return res
        new_factors['R_BODY_VOL'] = rank_pct(interact2)
    print(f"  BODY*VOL done ({time.time()-t0:.1f}s)", flush=True)

    # === 4. TENSION * SHADOW (structural displacement + candle pressure) ===
    tens = factors.get('R_TENSION')
    shad = factors.get('R_SHADOW_PRESSURE')
    if tens is not None and shad is not None:
        interact3 = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(25, ND):
                if not np.isnan(tens[si, di]) and not np.isnan(shad[si, di]):
                    interact3[si, di] = tens[si, di] * shad[si, di]
        def rank_pct(arr, start=60):
            res = np.full_like(arr, np.nan)
            for di in range(start, arr.shape[1]):
                vals = arr[:, di]
                mask = ~np.isnan(vals)
                if mask.sum() < 50:
                    continue
                ranked = np.argsort(np.argsort(vals[mask])).astype(float)
                n = len(ranked)
                pct = ranked / max(n - 1, 1) * 100
                for k, idx in enumerate(np.where(mask)[0]):
                    res[idx, di] = pct[k]
            return res
        new_factors['R_TENS_SHAD'] = rank_pct(interact3)
    print(f"  TENSION*SHADOW done ({time.time()-t0:.1f}s)", flush=True)

    # === 5. Composite score: (DRAWDOWN + TENSION + R_SQUARED) / 3 ===
    # This combines: oversold potential + structural position + trend quality
    if dd is not None and tens is not None and r_sq is not None:
        composite_raw = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(253, ND):
                d_val = dd[si, di]
                t_val = tens[si, di]
                r_val = r_sq[si, di]
                if not np.isnan(d_val) and not np.isnan(t_val) and not np.isnan(r_val):
                    composite_raw[si, di] = (d_val + t_val + r_val) / 3.0
        def rank_pct(arr, start=60):
            res = np.full_like(arr, np.nan)
            for di in range(start, arr.shape[1]):
                vals = arr[:, di]
                mask = ~np.isnan(vals)
                if mask.sum() < 50:
                    continue
                ranked = np.argsort(np.argsort(vals[mask])).astype(float)
                n = len(ranked)
                pct = ranked / max(n - 1, 1) * 100
                for k, idx in enumerate(np.where(mask)[0]):
                    res[idx, di] = pct[k]
            return res
        new_factors['R_COMPOSITE'] = rank_pct(composite_raw)
    print(f"  Composite done ({time.time()-t0:.1f}s)", flush=True)

    return new_factors


def backtest_v7b_with_stops(factor_weights, factors, NS, ND, dates, C, O, H, L, V,
                             top_n=5, rebalance_days=20, atr_stop_mult=2.5,
                             long_short=False):
    """Backtest with ATR trailing stop loss and optional long-short."""
    factor_names = list(factor_weights.keys())
    weights = np.array([factor_weights[f] for f in factor_names])

    cash = float(CASH0)
    holdings = []  # {'si', 'shares', 'entry', 'ed', 'high_water'}
    trades = []
    last_rebalance = -999
    year_stats = {}

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # Check ATR stop loss for all holdings
        for pos in list(holdings):
            p = C[pos['si'], di]  # Use close for stop check (intraday)
            if np.isnan(p):
                p = O[pos['si'], di]
            if not np.isnan(p) and p > 0:
                pos['high_water'] = max(pos.get('high_water', p), p)
                if atr_stop_mult > 0:
                    # Calculate ATR for this stock
                    atr = 0
                    atr_period = 14
                    for dd in range(max(di - atr_period, 1), di):
                        if not np.isnan(H[pos['si'], dd]) and not np.isnan(L[pos['si'], dd]):
                            tr = H[pos['si'], dd] - L[pos['si'], dd]
                            if not np.isnan(C[pos['si'], dd - 1]):
                                tr = max(tr, abs(H[pos['si'], dd] - C[pos['si'], dd - 1]),
                                         abs(L[pos['si'], dd] - C[pos['si'], dd - 1]))
                            atr += tr
                    atr /= atr_period
                    stop_price = pos['high_water'] - atr_stop_mult * atr
                    if p < stop_price:
                        # Stop loss hit
                        sell_p = O[pos['si'], di] if not np.isnan(O[pos['si'], di]) else p
                        pnl = (sell_p - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * sell_p * (1 - COMMISSION - STAMP_DUTY)
                        trades.append({
                            'pnl': pnl, 'days': (dates[di] - pos['ed']).days,
                            'di': di, 'reason': 'stop_loss', 'year': year
                        })
                        holdings.remove(pos)

        # Rebalance check
        if di - last_rebalance >= rebalance_days:
            # Compute composite score
            composite = np.zeros(NS)
            count = np.zeros(NS)
            for fname, w in zip(factor_names, weights):
                if fname not in factors:
                    continue
                arr = factors[fname]
                vals = arr[:, di]
                valid = ~np.isnan(vals)
                if valid.sum() < 50:
                    continue
                composite[valid] += w * vals[valid]
                count[valid] += abs(w)

            mask = count > 0
            if mask.sum() < top_n * 2:
                continue
            composite[mask] /= count[mask]
            composite[~mask] = -9999

            # Long: buy top N
            top_indices = set(np.argsort(-composite)[:top_n])
            current_indices = set(h['si'] for h in holdings)

            # Sell
            to_sell = current_indices - top_indices
            for pos in list(holdings):
                if pos['si'] in to_sell:
                    p = O[pos['si'], di]
                    if np.isnan(p) or p <= 0:
                        p = C[pos['si'], di]
                    if not np.isnan(p) and p > 0:
                        pnl = (p - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
                        trades.append({
                            'pnl': pnl, 'days': (dates[di] - pos['ed']).days,
                            'di': di, 'reason': 'rebalance', 'year': year
                        })
                        holdings.remove(pos)

            # Buy
            current_indices = set(h['si'] for h in holdings)
            to_buy = top_indices - current_indices
            n_to_buy = len(to_buy)
            if n_to_buy > 0 and cash > 10000:
                alloc = cash / n_to_buy
                for si in to_buy:
                    p = O[si, di]
                    if np.isnan(p) or p <= 0:
                        p = C[si, di]
                    if not np.isnan(p) and p > 0:
                        shares = int(alloc / (1 + COMMISSION) / p)
                        if shares > 0:
                            cost = shares * p * (1 + COMMISSION)
                            if cost <= cash:
                                cash -= cost
                                holdings.append({
                                    'si': si, 'shares': shares, 'entry': p,
                                    'ed': dates[di], 'high_water': p
                                })
            last_rebalance = di

    # Close remaining
    for pos in holdings:
        p = C[pos['si'], ND - 1]
        if not np.isnan(p) and p > 0:
            pnl = (p - pos['entry']) / pos['entry'] * 100
            cash += pos['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
            trades.append({'pnl': pnl, 'days': 999, 'di': ND - 1, 'reason': 'end',
                           'year': dates[ND - 1].year})

    if cash <= 0 or not trades:
        return None

    days_total = (dates[ND - 1] - dates[MIN_TRAIN]).days
    yr = max(days_total / 365.25, 0.01)
    ann = ((cash / CASH0) ** (1 / yr) - 1) * 100
    nw = sum(1 for t in trades if t['pnl'] > 0)
    wr = nw / max(len(trades), 1) * 100
    avg_w = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if nw > 0 else 0
    avg_l = np.mean([abs(t['pnl']) for t in trades if t['pnl'] <= 0]) if nw < len(trades) else 0

    for t in trades:
        y = t.get('year', 'unknown')
        if y not in year_stats:
            year_stats[y] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
        year_stats[y]['trades'] += 1
        if t['pnl'] > 0:
            year_stats[y]['wins'] += 1
        year_stats[y]['total_pnl'] += t['pnl']

    equity = float(CASH0)
    peak = float(CASH0)
    max_dd = 0
    for t in sorted(trades, key=lambda x: x['di']):
        equity *= (1 + t['pnl'] / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Count stop losses
    n_stop = sum(1 for t in trades if t.get('reason') == 'stop_loss')

    return {
        'ann': round(ann, 1), 'n': len(trades), 'wr': round(wr, 1),
        'avg_w': round(avg_w, 1), 'avg_l': round(avg_l, 1),
        'edge': round((nw / max(len(trades), 1)) * avg_w - (1 - nw / max(len(trades), 1)) * avg_l, 2),
        'max_dd': round(max_dd, 1), 'tpy': round(len(trades) / yr, 1),
        'final': round(cash, 0), 'year_stats': year_stats,
        'n_stop': n_stop,
    }


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V7b — Enhanced Interactions & Filters", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()
    base_factors = compute_all_factors(NS, ND, C, O, H, L, V)
    inter_factors = compute_interaction_factors(base_factors, NS, ND, C, O, H, L, V)

    # Merge all factors
    all_factors = {**base_factors, **inter_factors}
    print(f"\n  Total factors available: {len(all_factors)}", flush=True)

    # V7b test portfolios — focus on what worked in V7
    portfolios = {
        # V7 best combination
        'V7_Best5': {'R_R_SQUARED': 0.2, 'R_VOLATILITY_PCT': 0.2,
                     'R_DRAWDOWN_52W': 0.2, 'R_TENSION': 0.2,
                     'R_SHADOW_PRESSURE': 0.2},
        # With interaction terms
        'V7_Inter': {'R_R_SQUARED': 0.15, 'R_VOLATILITY_PCT': 0.15,
                     'R_DRAWDOWN_52W': 0.15, 'R_DD_VOL': 0.2,
                     'R_BODY_VOL': 0.2, 'R_TENS_SHAD': 0.15},
        # Composite only
        'Composite': {'R_COMPOSITE': 1.0},
        # Deep reversion + quality
        'DeepRev': {'R_DRAWDOWN_52W': 0.4, 'R_R_SQUARED': 0.3,
                    'R_DD_VOL': 0.3},
        # Structure + quality
        'StructQual': {'R_TENSION': 0.25, 'R_R_SQUARED': 0.25,
                       'R_TENS_SHAD': 0.25, 'R_BODY_VOL': 0.25},
        # Best of each dimension
        'BestDim': {'R_DRAWDOWN_52W': 0.2, 'R_R_SQUARED': 0.2,
                    'R_TENSION': 0.2, 'R_VOLATILITY_PCT': 0.2,
                    'R_DD_VOL': 0.2},
        # VDP heavy (volume delta pressure)
        'VDP_Heavy': {'R_VDP': 0.3, 'R_SHADOW_PRESSURE': 0.3,
                      'R_DD_VOL': 0.2, 'R_VOL_ANOMALY': 0.2},
        # Fisher + structure
        'FisherStruct': {'R_FISHER': 0.3, 'R_TENSION': 0.3,
                         'R_R_SQUARED': 0.2, 'R_VOLATILITY_PCT': 0.2},
    }

    results = []
    for pname, weights in portfolios.items():
        for top_n in [3, 5, 10]:
            for rebal in [10, 20, 30, 40]:
                for atr_mult in [0, 2.0, 3.0]:  # 0 = no stop
                    r = backtest_v7b_with_stops(weights, all_factors, NS, ND, dates,
                                                 C, O, H, L, V,
                                                 top_n=top_n, rebalance_days=rebal,
                                                 atr_stop_mult=atr_mult)
                    if r:
                        results.append({
                            'portfolio': pname,
                            'top_n': top_n,
                            'rebal': rebal,
                            'atr': atr_mult,
                            **r
                        })
        print(f"  {pname} done", flush=True)

    results.sort(key=lambda x: -x['ann'])

    print(f"\n{'='*120}", flush=True)
    print(f"  TOP 40 RESULTS", flush=True)
    print(f"  {'Portfolio':<15s} {'Top':>3s} {'Reb':>3s} {'ATR':>3s} | {'Ann':>7s} {'N':>5s} {'TPY':>4s} {'WR':>5s} "
          f"{'Edge':>6s} {'DD':>5s} {'Stop':>4s}", flush=True)
    print(f"  {'-'*110}", flush=True)
    for r in results[:40]:
        print(f"  {r['portfolio']:<15s} {r['top_n']:3d} {r['rebal']:3d} {r['atr']:3.0f} | {r['ann']:+7.1f}% {r['n']:5d} "
              f"{r['tpy']:4.0f} {r['wr']:5.1f}% {r['edge']:+6.2f}% {r['max_dd']:5.1f}% {r.get('n_stop', 0):4d}", flush=True)

    # Best per portfolio
    best_per = {}
    for r in results:
        p = r['portfolio']
        if p not in best_per or r['ann'] > best_per[p]['ann']:
            best_per[p] = r
    print(f"\n  Best per portfolio:", flush=True)
    for r in sorted(best_per.values(), key=lambda x: -x['ann']):
        print(f"    {r['portfolio']:<15s} → {r['ann']:+.1f}% (Top={r['top_n']}, Reb={r['rebal']}, "
              f"ATR={r['atr']:.0f}, WR={r['wr']:.0f}%, DD={r['max_dd']:.1f}%)", flush=True)

    # Year-by-year for best
    if best_per:
        best = sorted(best_per.values(), key=lambda x: -x['ann'])[0]
        print(f"\n  Year-by-year: {best['portfolio']} (Ann={best['ann']:+.1f}%)", flush=True)
        for y in sorted(best.get('year_stats', {}).keys()):
            s = best['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"  ALPHA V7b COMPLETE", flush=True)
    print(f"{'='*70}", flush=True)
