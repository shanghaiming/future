"""
Alpha V6 — No Look-Ahead Bias Test
====================================
V4/V5 的致命问题：因子用当天(di)的收盘价，交易在当天(di)的开盘价。
这是未来数据泄漏！

修复：因子用 di-1 的数据，交易在 di 的开盘价。
如果修复后收益大幅下降，说明之前的收益主要来自 look-ahead。
如果修复后仍有正收益，说明因子排名有真正的 alpha。
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.data_loader import list_available_symbols, load_stock_data
from alpha_v2 import load_all_data, MIN_TRAIN

COMMISSION = 0.0003
STAMP_DUTY = 0.001
CASH0 = 500_000


def compute_v75_factors_nolook(NS, ND, C, O, H, L, V):
    """V75-style factors with NO look-ahead.
    All factors at day di use data only up to di-1 (yesterday's close)."""
    print("[Factors] Computing NO-LOOK-AHEAD factors...", flush=True)
    t0 = time.time()

    # Shift all raw data by 1 day — only use yesterday's info
    # We compute factors for di using data up to di-1
    # This means MOM5 at di = (C[di-1] - C[di-1-5]) / C[di-1-5]

    MOM5 = np.full((NS, ND), np.nan)
    MOM10 = np.full((NS, ND), np.nan)
    MOM20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):  # extra 1 for the shift
            # Use di-1 instead of di
            if np.isnan(C[si, di-1]): continue
            if not np.isnan(C[si, di-1-5]) and C[si, di-1-5] > 0:
                MOM5[si, di] = (C[si, di-1] - C[si, di-1-5]) / C[si, di-1-5]
            if not np.isnan(C[si, di-1-10]) and C[si, di-1-10] > 0:
                MOM10[si, di] = (C[si, di-1] - C[si, di-1-10]) / C[si, di-1-10]
            if not np.isnan(C[si, di-1-20]) and C[si, di-1-20] > 0:
                MOM20[si, di] = (C[si, di-1] - C[si, di-1-20]) / C[si, di-1-20]

    # Price percentile (60-day) — use data up to di-1
    PRICE_PCT = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(61, ND):
            vals = C[si, di-1-60:di]; valid = vals[~np.isnan(vals)]  # up to di-1
            if len(valid) < 30: continue
            cur = C[si, di-1]
            if np.isnan(cur): continue
            PRICE_PCT[si, di] = np.sum(valid < cur) / max(len(valid)-1, 1) * 100

    # VDP EMA — use data up to di-1
    EMA_P = 10; a_ema = 2.0/(EMA_P+1)
    VDP_DELTA = np.full((NS, ND), np.nan)
    for si in range(NS):
        ema_val = np.nan
        for di in range(1, ND):
            # Use di-1 data
            d = di - 1
            if np.isnan(V[si, d]) or V[si, d] <= 0: continue
            if np.isnan(C[si, d]) or np.isnan(H[si, d]) or np.isnan(L[si, d]): continue
            hl = H[si, d] - L[si, d]
            if hl <= 0:
                delta = V[si, d] if C[si, d] >= H[si, d] else -V[si, d] if C[si, d] <= L[si, d] else None
                if delta is None: continue
            else:
                delta = V[si, d] * (2*C[si, d] - H[si, d] - L[si, d]) / hl
            ema_val = delta if np.isnan(ema_val) else a_ema * delta + (1 - a_ema) * ema_val
            VDP_DELTA[si, di] = ema_val  # Store at di, but computed from di-1

    # Relative volume — use data up to di-1
    REL_VOL = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            if np.isnan(V[si, di-1]) or V[si, di-1] <= 0: continue
            v20 = V[si, di-1-20:di]; v20v = v20[~np.isnan(v20)]  # up to di-1
            if len(v20v) < 10: continue
            avg_v = np.mean(v20v)
            if avg_v > 0: REL_VOL[si, di] = V[si, di-1] / avg_v

    # BB width — use data up to di-1
    BB_WIDTH = np.full((NS, ND), np.nan)
    ATR_PCT = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            c20 = C[si, di-1-20:di]; valid = c20[~np.isnan(c20)]  # up to di-1
            if len(valid) < 15: continue
            ma = np.mean(valid); std = np.std(valid)
            if ma > 0 and std > 0: BB_WIDTH[si, di] = (4*std)/ma * 100
            # ATR — use data up to di-1
            if di < 3: continue
            atr_vals = []
            for dd in range(max(di-1-14, 1), di):
                if not np.isnan(H[si, dd]) and not np.isnan(L[si, dd]):
                    tr = H[si, dd] - L[si, dd]
                    if not np.isnan(C[si, dd-1]):
                        tr = max(tr, abs(H[si, dd]-C[si, dd-1]), abs(L[si, dd]-C[si, dd-1]))
                    atr_vals.append(tr)
            if len(atr_vals) >= 5 and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                ATR_PCT[si, di] = np.mean(atr_vals) / C[si, di-1] * 100

    print(f"  Raw factors done ({time.time()-t0:.1f}s)", flush=True)

    # Cross-sectional rank normalization
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

    factors = {
        'R_MOM5': R_MOM5, 'R_MOM10': R_MOM10, 'R_MOM20': R_MOM20,
        'R_PRICE': R_PRICE, 'R_VDP': R_VDP, 'R_REL_VOL': R_REL_VOL,
        'R_BB': R_BB, 'R_ATR': R_ATR,
        'D_MOM5_3': D_MOM5_3, 'D_MOM10_5': D_MOM10_5, 'D_MOM20_10': D_MOM20_10,
        'D_PRICE_5': D_PRICE_5, 'D_VDP_5': D_VDP_5, 'D_REL_VOL_5': D_REL_VOL_5,
        'D_BB_5': D_BB_5, 'D_ATR_5': D_ATR_5,
    }
    print(f"  All factors done ({time.time()-t0:.1f}s)", flush=True)
    return factors


def backtest_v6(factor_weights, factors, NS, ND, dates, C, O, H, L, V,
                top_n=10, rebalance_days=5, atr_exit=True):
    """No-look-ahead backtest. Factors use di-1 data, trade at di open."""
    factor_names = list(factor_weights.keys())
    weights = np.array([factor_weights[f] for f in factor_names])

    cash = float(CASH0)
    holdings = []
    trades = []
    last_rebalance = -999
    year_stats = {}

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        days_since_rebal = di - last_rebalance
        should_rebalance = days_since_rebal >= rebalance_days

        if should_rebalance:
            # Use factors at di (computed from di-1 data — no look-ahead)
            composite = np.zeros(NS)
            count = np.zeros(NS)
            for fname, w in zip(factor_names, weights):
                if fname not in factors: continue
                arr = factors[fname]
                vals = arr[:, di]
                valid = ~np.isnan(vals)
                if valid.sum() < 50: continue
                composite[valid] += w * vals[valid]
                count[valid] += abs(w)

            mask = count > 0
            if mask.sum() < top_n * 2: continue
            composite[mask] /= count[mask]
            composite[~mask] = -9999

            top_indices = set(np.argsort(-composite)[:top_n])
            current_indices = set(h['si'] for h in holdings)

            # Sell at OPEN price (no look-ahead — open is known at trade time)
            to_sell = current_indices - top_indices
            for pos in list(holdings):
                if pos['si'] in to_sell:
                    p = O[pos['si'], di]
                    if np.isnan(p) or p <= 0: p = C[pos['si'], di]
                    if not np.isnan(p) and p > 0:
                        pnl = (p - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
                        trades.append({
                            'pnl': pnl, 'days': (dates[di] - pos['ed']).days,
                            'di': di, 'reason': 'rebalance', 'year': year
                        })
                        holdings.remove(pos)

            # Buy at OPEN price
            current_indices = set(h['si'] for h in holdings)
            to_buy = top_indices - current_indices
            n_to_buy = len(to_buy)
            if n_to_buy > 0 and cash > 10000:
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
                                holdings.append({
                                    'si': si, 'shares': shares, 'entry': p,
                                    'ed': dates[di]
                                })
            last_rebalance = di

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

    for t in trades:
        y = t.get('year', 'unknown')
        if y not in year_stats: year_stats[y] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
        year_stats[y]['trades'] += 1
        if t['pnl'] > 0: year_stats[y]['wins'] += 1
        year_stats[y]['total_pnl'] += t['pnl']

    equity = float(CASH0); peak = float(CASH0); max_dd = 0
    for t in sorted(trades, key=lambda x: x['di']):
        equity *= (1 + t['pnl'] / 100)
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd

    return {
        'ann': round(ann, 1), 'n': len(trades), 'wr': round(wr, 1),
        'avg_w': round(avg_w, 1), 'avg_l': round(avg_l, 1),
        'edge': round((nw/max(len(trades),1))*avg_w - (1-nw/max(len(trades),1))*avg_l, 2),
        'max_dd': round(max_dd, 1), 'tpy': round(len(trades) / yr, 1),
        'final': round(cash, 0), 'year_stats': year_stats,
    }


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V6 — No Look-Ahead Bias Test", flush=True)
    print("  All factors use di-1 data, trade at di open", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()
    factors = compute_v75_factors_nolook(NS, ND, C, O, H, L, V)

    portfolios = {
        'MOM5_Only': {'R_MOM5': 0.5, 'D_MOM5_3': 0.5},
        'V75_Rank': {'R_MOM5': 0.12, 'R_MOM10': 0.12, 'R_MOM20': 0.12,
                     'R_PRICE': 0.13, 'R_VDP': 0.13, 'R_REL_VOL': 0.13,
                     'R_BB': 0.12, 'R_ATR': 0.13},
        'V75_Full': {'R_MOM5': 0.06, 'R_MOM10': 0.06, 'R_MOM20': 0.06,
                     'R_PRICE': 0.06, 'R_VDP': 0.06, 'R_REL_VOL': 0.06,
                     'R_BB': 0.06, 'R_ATR': 0.06,
                     'D_MOM5_3': 0.06, 'D_MOM10_5': 0.06, 'D_MOM20_10': 0.06,
                     'D_PRICE_5': 0.06, 'D_VDP_5': 0.06, 'D_REL_VOL_5': 0.06,
                     'D_BB_5': 0.06, 'D_ATR_5': 0.06},
        'MOM5_VDP': {'R_MOM5': 0.3, 'R_VDP': 0.3, 'D_VDP_5': 0.4},
    }

    results = []
    for pname, weights in portfolios.items():
        for top_n in [10, 20, 30]:
            for rebal in [3, 5, 10, 20]:
                r = backtest_v6(weights, factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, rebalance_days=rebal)
                if r:
                    results.append({
                        'portfolio': pname,
                        'top_n': top_n,
                        'rebal': rebal,
                        **r
                    })
        print(f"  {pname} done", flush=True)

    results.sort(key=lambda x: -x['ann'])

    print(f"\n{'='*100}", flush=True)
    print(f"  NO LOOK-AHEAD RESULTS", flush=True)
    print(f"  {'Portfolio':<15s} {'Top':>3s} {'Reb':>3s} | {'Ann':>7s} {'N':>5s} {'TPY':>4s} {'WR':>5s} "
          f"{'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*95}", flush=True)
    for r in results[:30]:
        print(f"  {r['portfolio']:<15s} {r['top_n']:3d} {r['rebal']:3d} | {r['ann']:+7.1f}% {r['n']:5d} "
              f"{r['tpy']:4.0f} {r['wr']:5.1f}% {r['edge']:+6.2f}% {r['max_dd']:5.1f}%", flush=True)

    # Best per portfolio
    best_per = {}
    for r in results:
        p = r['portfolio']
        if p not in best_per or r['ann'] > best_per[p]['ann']:
            best_per[p] = r
    print(f"\n  Best per portfolio:", flush=True)
    for r in sorted(best_per.values(), key=lambda x: -x['ann']):
        print(f"    {r['portfolio']:<15s} → {r['ann']:+.1f}% (Top={r['top_n']}, Reb={r['rebal']}, "
              f"WR={r['wr']:.0f}%, DD={r['max_dd']:.1f}%, TPY={r['tpy']:.0f})", flush=True)

    # Year-by-year for best
    if best_per:
        best = sorted(best_per.values(), key=lambda x: -x['ann'])[0]
        print(f"\n  Year-by-year: {best['portfolio']} (Ann={best['ann']:+.1f}%)", flush=True)
        for y in sorted(best.get('year_stats', {}).keys()):
            s = best['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    print(f"\n  COMPARISON WITH LOOK-AHEAD V4:", flush=True)
    print(f"  V4 Enhanced (look-ahead):  +1291.8%", flush=True)
    print(f"  V4 MOM5+VDP (look-ahead):  +976.2%", flush=True)
    print(f"  V4 V75_Full (look-ahead):   +755.4%", flush=True)
    if best_per:
        best = sorted(best_per.values(), key=lambda x: -x['ann'])[0]
        print(f"  V6 Best (no look-ahead):    {best['ann']:+.1f}%", flush=True)
    print(f"{'='*70}", flush=True)
