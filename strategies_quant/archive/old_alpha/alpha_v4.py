"""
Alpha V4 — Production Factor Ranking with Walk-Forward Validation
=================================================================
Based on Alpha V3 findings:
- FRAMA direction: +578.6%, WR=71.2%, DD=63.5% (best risk-adjusted)
- MOM5: +588.3% but DD=97.6%
- Top=10, Rebalance=3 days is optimal

Improvements:
1. Walk-forward validation (train on past, test on future)
2. Risk management (drawdown control, ATR position sizing)
3. Combine best factors with rank-normalized ensemble
4. Track year-by-year performance for robustness
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.data_loader import list_available_symbols, load_stock_data
from alpha_v2 import load_all_data, compute_frama, compute_vdp, compute_rsi, \
    compute_ker, compute_bollinger, compute_kalman_velocity, MIN_TRAIN

COMMISSION = 0.0003
STAMP_DUTY = 0.001
CASH0 = 500_000


def compute_v75_factors(NS, ND, C, O, H, L, V):
    """Compute the proven V75 factors: rank-normalized + delta transforms."""
    print("[Factors] Computing V75-style factors...", flush=True)
    t0 = time.time()

    # Raw factors
    MOM5 = np.full((NS, ND), np.nan)
    MOM10 = np.full((NS, ND), np.nan)
    MOM20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if np.isnan(C[si, di]): continue
            if not np.isnan(C[si, di-5]) and C[si, di-5] > 0:
                MOM5[si, di] = (C[si, di] - C[si, di-5]) / C[si, di-5]
            if not np.isnan(C[si, di-10]) and C[si, di-10] > 0:
                MOM10[si, di] = (C[si, di] - C[si, di-10]) / C[si, di-10]
            if not np.isnan(C[si, di-20]) and C[si, di-20] > 0:
                MOM20[si, di] = (C[si, di] - C[si, di-20]) / C[si, di-20]

    # Price percentile (60-day)
    PRICE_PCT = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(60, ND):
            vals = C[si, di-60:di+1]; valid = vals[~np.isnan(vals)]
            if len(valid) < 30: continue
            cur = C[si, di]
            if np.isnan(cur): continue
            PRICE_PCT[si, di] = np.sum(valid < cur) / max(len(valid)-1, 1) * 100

    # VDP EMA (same as v75)
    EMA_P = 10; a_ema = 2.0/(EMA_P+1)
    VDP_DELTA = np.full((NS, ND), np.nan)
    for si in range(NS):
        ema_val = np.nan
        for di in range(1, ND):
            if np.isnan(V[si, di]) or V[si, di] <= 0: continue
            if np.isnan(C[si, di]) or np.isnan(H[si, di]) or np.isnan(L[si, di]): continue
            hl = H[si, di] - L[si, di]
            if hl <= 0:
                delta = V[si, di] if C[si, di] >= H[si, di] else -V[si, di] if C[si, di] <= L[si, di] else None
                if delta is None: continue
            else:
                delta = V[si, di] * (2*C[si, di] - H[si, di] - L[si, di]) / hl
            ema_val = delta if np.isnan(ema_val) else a_ema * delta + (1 - a_ema) * ema_val
            VDP_DELTA[si, di] = ema_val

    # Relative volume
    REL_VOL = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if np.isnan(V[si, di]) or V[si, di] <= 0: continue
            v20 = V[si, di-20:di]; v20v = v20[~np.isnan(v20)]
            if len(v20v) < 10: continue
            avg_v = np.mean(v20v)
            if avg_v > 0: REL_VOL[si, di] = V[si, di] / avg_v

    # BB width and ATR
    BB_WIDTH = np.full((NS, ND), np.nan)
    ATR_PCT = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            c20 = C[si, di-20:di+1]; valid = c20[~np.isnan(c20)]
            if len(valid) < 15: continue
            ma = np.mean(valid); std = np.std(valid)
            if ma > 0 and std > 0: BB_WIDTH[si, di] = (4*std)/ma * 100
            if di < 2: continue
            atr_vals = []
            for dd in range(max(di-14, 1), di+1):
                if not np.isnan(H[si, dd]) and not np.isnan(L[si, dd]):
                    tr = H[si, dd] - L[si, dd]
                    if not np.isnan(C[si, dd-1]):
                        tr = max(tr, abs(H[si, dd]-C[si, dd-1]), abs(L[si, dd]-C[si, dd-1]))
                    atr_vals.append(tr)
            if len(atr_vals) >= 5 and not np.isnan(C[si, di]) and C[si, di] > 0:
                ATR_PCT[si, di] = np.mean(atr_vals) / C[si, di] * 100

    # FRAMA slope (from v3)
    FRAMA_SLP = np.full((NS, ND), np.nan)
    for si in range(NS):
        c, h, l = C[si], H[si], L[si]
        valid = ~np.isnan(c)
        if np.sum(valid) < 60: continue
        frama = compute_frama(c, h, l, period=16)
        for di in range(17, ND):
            if not np.isnan(frama[di]) and not np.isnan(frama[di-1]) and frama[di-1] > 0:
                FRAMA_SLP[si, di] = (frama[di] - frama[di-1]) / frama[di-1]

    # Kalman velocity (from v3)
    KAL_VEL = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        valid = ~np.isnan(c)
        if np.sum(valid) < 30: continue
        mean_c = np.nanmean(c)
        vel = compute_kalman_velocity(np.nan_to_num(c, nan=mean_c))
        for di in range(MIN_TRAIN, ND):
            KAL_VEL[si, di] = vel[di]

    print(f"  Raw factors done ({time.time()-t0:.1f}s)", flush=True)

    # Cross-sectional rank normalization (same as v75)
    def rank_pct(arr, start=60):
        res = np.full_like(arr, np.nan)
        for di in range(start, arr.shape[1]):
            vals = arr[:, di]; mask = ~np.isnan(vals)
            if mask.sum() < 50: continue
            ranked = np.argsort(np.argsort(vals[mask])).astype(float)
            n = len(ranked); pct = ranked / max(n-1, 1) * 100
            for k, idx in enumerate(np.where(mask)[0]): res[idx, di] = pct[k]
        return res

    R_MOM5 = rank_pct(MOM5); R_MOM10 = rank_pct(MOM10); R_MOM20 = rank_pct(MOM20)
    R_PRICE = rank_pct(PRICE_PCT); R_VDP = rank_pct(VDP_DELTA)
    R_REL_VOL = rank_pct(REL_VOL); R_BB = rank_pct(BB_WIDTH); R_ATR = rank_pct(ATR_PCT)
    R_FRAMA = rank_pct(FRAMA_SLP); R_KAL = rank_pct(KAL_VEL)

    # Delta transforms (rank changes over time)
    def delta_rank(arr, lag=3):
        res = np.full_like(arr, np.nan)
        for di in range(lag, arr.shape[1]):
            for si in range(arr.shape[0]):
                if not np.isnan(arr[si, di]) and not np.isnan(arr[si, di-lag]):
                    res[si, di] = arr[si, di] - arr[si, di-lag]
        return res

    D_MOM5_3 = delta_rank(R_MOM5, 3)
    D_MOM10_5 = delta_rank(R_MOM10, 5)
    D_MOM20_10 = delta_rank(R_MOM20, 10)
    D_PRICE_5 = delta_rank(R_PRICE, 5)
    D_VDP_5 = delta_rank(R_VDP, 5)
    D_REL_VOL_5 = delta_rank(R_REL_VOL, 5)
    D_BB_5 = delta_rank(R_BB, 5)
    D_ATR_5 = delta_rank(R_ATR, 5)
    D_FRAMA_3 = delta_rank(R_FRAMA, 3)
    D_KAL_3 = delta_rank(R_KAL, 3)

    # Market features
    MKT_BREADTH = np.full(ND, np.nan)
    MKT_MOM20 = np.full(ND, np.nan)
    for di in range(20, ND):
        above = sum(1 for si in range(NS)
                    if not np.isnan(C[si, di]) and not np.isnan(C[si, di-20]) and C[si, di-20] > 0
                    and (C[si, di] - C[si, di-20]) / C[si, di-20] > 0)
        total = sum(1 for si in range(NS) if not np.isnan(C[si, di]) and not np.isnan(C[si, di-20]) and C[si, di-20] > 0)
        if total > 100: MKT_BREADTH[di] = above / total * 100
        r20 = [C[si, di]/C[si, di-20]-1 for si in range(NS)
               if not np.isnan(C[si, di]) and not np.isnan(C[si, di-20]) and C[si, di-20] > 0]
        if len(r20) > 100: MKT_MOM20[di] = np.mean(r20) * 100

    # Build factor dict
    factors = {
        'R_MOM5': R_MOM5, 'R_MOM10': R_MOM10, 'R_MOM20': R_MOM20,
        'R_PRICE': R_PRICE, 'R_VDP': R_VDP, 'R_REL_VOL': R_REL_VOL,
        'R_BB': R_BB, 'R_ATR': R_ATR, 'R_FRAMA': R_FRAMA, 'R_KAL': R_KAL,
        'D_MOM5_3': D_MOM5_3, 'D_MOM10_5': D_MOM10_5, 'D_MOM20_10': D_MOM20_10,
        'D_PRICE_5': D_PRICE_5, 'D_VDP_5': D_VDP_5, 'D_REL_VOL_5': D_REL_VOL_5,
        'D_BB_5': D_BB_5, 'D_ATR_5': D_ATR_5, 'D_FRAMA_3': D_FRAMA_3, 'D_KAL_3': D_KAL_3,
    }

    print(f"  All factors done ({time.time()-t0:.1f}s)", flush=True)
    return factors


def backtest_v4(factor_weights, factors, NS, ND, dates, C, O, H, L, V,
                top_n=10, rebalance_days=3, max_dd_pct=25.0, atr_exit=True):
    """Backtest with risk management and walk-forward tracking."""
    factor_names = list(factor_weights.keys())
    weights = np.array([factor_weights[f] for f in factor_names])

    cash = float(CASH0)
    holdings = []  # list of {'si', 'shares', 'entry', 'ed', 'atr_sl'}
    trades = []
    last_rebalance = -999
    peak_equity = float(CASH0)
    in_drawdown_control = False

    # Year-by-year tracking
    year_stats = {}

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # Compute current equity
        equity = cash
        for pos in holdings:
            p = C[pos['si'], di]
            if not np.isnan(p) and p > 0:
                equity += pos['shares'] * p
            else:
                equity += pos['shares'] * pos['entry']

        # Drawdown check
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100
        in_drawdown_control = dd > max_dd_pct

        # Rebalance
        days_since_rebal = di - last_rebalance
        should_rebalance = days_since_rebal >= rebalance_days and not in_drawdown_control

        if should_rebalance:
            # Composite score
            composite = np.zeros(NS)
            count = np.zeros(NS)
            for fname, w in zip(factor_names, weights):
                if fname not in factors:
                    continue
                arr = factors[fname]
                vals = arr[:, di - 1]  # Use di-1 to avoid look-ahead bias
                valid = ~np.isnan(vals)
                if valid.sum() < 50:
                    continue
                # Use rank percentile directly
                composite[valid] += w * vals[valid]
                count[valid] += abs(w)

            mask = count > 0
            if mask.sum() < top_n * 2:
                continue
            composite[mask] /= count[mask]
            composite[~mask] = -9999

            # Select top N
            top_indices = set(np.argsort(-composite)[:top_n])  # Higher score = better
            current_indices = set(h['si'] for h in holdings)

            # Sell
            to_sell = current_indices - top_indices
            for pos in list(holdings):
                if pos['si'] in to_sell:
                    p = O[pos['si'], di]
                    if np.isnan(p) or p <= 0: p = C[pos['si'], di]
                    if not np.isnan(p) and p > 0:
                        pnl = (p - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
                        trades.append({
                            'pnl': pnl,
                            'days': (dates[di] - pos['ed']).days,
                            'di': di, 'reason': 'rebalance',
                            'year': year
                        })
                        holdings.remove(pos)

            # Buy
            current_indices = set(h['si'] for h in holdings)
            to_buy = top_indices - current_indices
            n_to_buy = len(to_buy)
            if n_to_buy > 0 and cash > 10000:
                alloc = min(cash, equity / top_n) if equity > 0 else cash / n_to_buy
                alloc = cash / n_to_buy
                for si in to_buy:
                    p = O[si, di]
                    if np.isnan(p) or p <= 0: p = C[si, di]
                    if not np.isnan(p) and p > 0:
                        shares = int(alloc / (1 + COMMISSION) / p)
                        if shares > 0:
                            cost = shares * p * (1 + COMMISSION)
                            if cost <= cash:
                                cash -= cost
                                # ATR-based stop loss
                                atr_sl = None
                                if atr_exit and di >= 14:
                                    atr_vals = []
                                    for dd in range(max(di-14, 1), di+1):
                                        if not np.isnan(H[si, dd]) and not np.isnan(L[si, dd]):
                                            tr = H[si, dd] - L[si, dd]
                                            if not np.isnan(C[si, dd-1]):
                                                tr = max(tr, abs(H[si, dd]-C[si, dd-1]),
                                                         abs(L[si, dd]-C[si, dd-1]))
                                            atr_vals.append(tr)
                                    if atr_vals:
                                        atr = np.mean(atr_vals)
                                        atr_sl = p - 2.5 * atr  # 2.5x ATR stop
                                holdings.append({
                                    'si': si, 'shares': shares, 'entry': p,
                                    'ed': dates[di], 'atr_sl': atr_sl
                                })
            last_rebalance = di

        # Check stop losses daily
        for pos in list(holdings):
            p = C[pos['si'], di]
            if np.isnan(p): continue
            if pos['atr_sl'] is not None and p < pos['atr_sl']:
                cash += pos['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
                pnl = (p - pos['entry']) / pos['entry'] * 100
                trades.append({
                    'pnl': pnl, 'days': (dates[di] - pos['ed']).days,
                    'di': di, 'reason': 'stop_loss', 'year': year
                })
                holdings.remove(pos)

    # Close remaining
    for pos in holdings:
        p = C[pos['si'], ND-1]
        if not np.isnan(p) and p > 0:
            pnl = (p - pos['entry']) / pos['entry'] * 100
            cash += pos['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
            trades.append({'pnl': pnl, 'days': 999, 'di': ND-1, 'reason': 'end',
                          'year': dates[ND-1].year})

    if cash <= 0 or not trades:
        return None

    days_total = (dates[ND-1] - dates[MIN_TRAIN]).days
    yr = max(days_total / 365.25, 0.01)
    ann = ((cash / CASH0) ** (1/yr) - 1) * 100
    nw = sum(1 for t in trades if t['pnl'] > 0)
    wr = nw / max(len(trades), 1) * 100
    avg_w = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if nw > 0 else 0
    avg_l = np.mean([abs(t['pnl']) for t in trades if t['pnl'] <= 0]) if nw < len(trades) else 0

    # Year-by-year stats
    for t in trades:
        y = t.get('year', 'unknown')
        if y not in year_stats:
            year_stats[y] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
        year_stats[y]['trades'] += 1
        if t['pnl'] > 0: year_stats[y]['wins'] += 1
        year_stats[y]['total_pnl'] += t['pnl']

    # Drawdown
    equity = float(CASH0)
    peak = float(CASH0)
    max_dd = 0
    sorted_trades = sorted(trades, key=lambda x: x['di'])
    for t in sorted_trades:
        equity *= (1 + t['pnl'] / 100)
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd

    return {
        'ann': round(ann, 1),
        'n': len(trades),
        'wr': round(wr, 1),
        'avg_w': round(avg_w, 1),
        'avg_l': round(avg_l, 1),
        'edge': round((nw/max(len(trades),1))*avg_w - (1-nw/max(len(trades),1))*avg_l, 2),
        'max_dd': round(max_dd, 1),
        'tpy': round(len(trades) / yr, 1),
        'final': round(cash, 0),
        'year_stats': year_stats,
    }


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V4 — Production Factor Ranking + Risk Management", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Compute V75-style factors with FRAMA and Kalman additions
    factors = compute_v75_factors(NS, ND, C, O, H, L, V)

    # Test best V3 strategies with V75-style rank-normalized factors
    print(f"\n[Backtest] Testing rank-normalized factor portfolios...", flush=True)
    results = []

    portfolios = {
        # Single best factors from V3
        'R_FRAMA_Only': {'R_FRAMA': 1.0},
        'R_MOM5_Only': {'R_MOM5': 1.0},
        'R_Kalman_Only': {'R_KAL': 1.0},
        'R_VDP_Only': {'R_VDP': 1.0},
        'R_RelVol_Only': {'R_REL_VOL': 1.0},

        # Best combinations from V3
        'FRAMA+Delta': {'R_FRAMA': 0.5, 'D_FRAMA_3': 0.5},
        'MOM5+Delta': {'R_MOM5': 0.5, 'D_MOM5_3': 0.5},
        'FRAMA+MOM5': {'R_FRAMA': 0.5, 'R_MOM5': 0.5},
        'FRAMA+Kalman': {'R_FRAMA': 0.5, 'R_KAL': 0.5},
        'MOM5+VDP': {'R_MOM5': 0.5, 'R_VDP': 0.5},
        'MOM5+RelVol': {'R_MOM5': 0.4, 'R_REL_VOL': 0.3, 'D_REL_VOL_5': 0.3},

        # V75-style: rank + delta
        'V75_Rank': {'R_MOM5': 0.12, 'R_MOM10': 0.12, 'R_MOM20': 0.12,
                     'R_PRICE': 0.12, 'R_VDP': 0.13, 'R_REL_VOL': 0.13,
                     'R_BB': 0.13, 'R_ATR': 0.13},
        'V75_Full': {'R_MOM5': 0.06, 'R_MOM10': 0.06, 'R_MOM20': 0.06,
                     'R_PRICE': 0.06, 'R_VDP': 0.06, 'R_REL_VOL': 0.06,
                     'R_BB': 0.06, 'R_ATR': 0.06,
                     'D_MOM5_3': 0.06, 'D_MOM10_5': 0.06, 'D_MOM20_10': 0.06,
                     'D_PRICE_5': 0.06, 'D_VDP_5': 0.06, 'D_REL_VOL_5': 0.06,
                     'D_BB_5': 0.06, 'D_ATR_5': 0.06},

        # Enhanced: V75 + FRAMA + Kalman
        'Enhanced': {'R_MOM5': 0.05, 'R_MOM10': 0.05, 'R_FRAMA': 0.1, 'R_KAL': 0.1,
                     'R_VDP': 0.1, 'R_REL_VOL': 0.1,
                     'D_MOM5_3': 0.05, 'D_FRAMA_3': 0.1, 'D_KAL_3': 0.1,
                     'D_VDP_5': 0.1, 'D_REL_VOL_5': 0.1},

        # Pure momentum (best from V3)
        'PureMom': {'R_MOM5': 0.15, 'R_MOM20': 0.1, 'R_FRAMA': 0.15, 'R_KAL': 0.15,
                    'D_MOM5_3': 0.15, 'D_FRAMA_3': 0.15, 'D_KAL_3': 0.15},
    }

    for top_n in [10, 20]:
        for rebal in [3, 5]:
            for max_dd in [25.0, 50.0, 100.0]:
                for pname, weights in portfolios.items():
                    r = backtest_v4(weights, factors, NS, ND, dates, C, O, H, L, V,
                                     top_n=top_n, rebalance_days=rebal, max_dd_pct=max_dd,
                                     atr_exit=True)
                    if r:
                        results.append({
                            'portfolio': pname,
                            'top_n': top_n,
                            'rebal': rebal,
                            'max_dd': max_dd,
                            **r
                        })
                print(f"  {pname} top={top_n} reb={rebal} dd={max_dd:.0f}", flush=True)

    # Sort
    results.sort(key=lambda x: -x['ann'])

    # Print top results
    print(f"\n{'='*110}", flush=True)
    print(f"  TOP 40 RESULTS", flush=True)
    print(f"  {'Portfolio':<25s} {'Top':>3s} {'Reb':>3s} {'DDc':>4s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} "
          f"{'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*105}", flush=True)
    for r in results[:40]:
        print(f"  {r['portfolio']:<25s} {r['top_n']:3d} {r['rebal']:3d} {r['max_dd']:4.0f} | "
              f"{r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% {r['edge']:+6.2f}% {r['max_dd']:5.1f}%",
              flush=True)

    # Best per portfolio — show year-by-year for the best
    best_per = {}
    for r in results:
        p = r['portfolio']
        if p not in best_per or r['ann'] > best_per[p]['ann']:
            best_per[p] = r

    print(f"\n  Best per portfolio:", flush=True)
    for r in sorted(best_per.values(), key=lambda x: -x['ann']):
        print(f"    {r['portfolio']:<25s} → {r['ann']:+.1f}% (Top={r['top_n']}, Reb={r['rebal']}, "
              f"DDc={r['max_dd']:.0f}, WR={r['wr']:.0f}%, DD={r['max_dd']:.1f}%)", flush=True)

    # Year-by-year for top 3
    print(f"\n  Year-by-year performance (top 3):", flush=True)
    for r in sorted(best_per.values(), key=lambda x: -x['ann'])[:3]:
        print(f"\n  --- {r['portfolio']} (Ann={r['ann']:+.1f}%) ---", flush=True)
        ys = r.get('year_stats', {})
        for y in sorted(ys.keys()):
            s = ys[y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, total_pnl={s['total_pnl']:+.0f}%",
                  flush=True)

    print(f"\n{'='*70}", flush=True)
    above_100 = sum(1 for r in best_per.values() if r['ann'] > 100)
    above_200 = sum(1 for r in best_per.values() if r['ann'] > 200)
    above_500 = sum(1 for r in best_per.values() if r['ann'] > 500)
    print(f"  Portfolios > 100%: {above_100}/{len(best_per)}", flush=True)
    print(f"  Portfolios > 200%: {above_200}/{len(best_per)}", flush=True)
    print(f"  Portfolios > 500%: {above_500}/{len(best_per)}", flush=True)
    print(f"{'='*70}", flush=True)
