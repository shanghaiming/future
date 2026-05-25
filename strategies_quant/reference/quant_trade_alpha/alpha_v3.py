"""
Alpha V3 — Cross-Sectional Factor Ranking (No ML)
=================================================
Key insight: Don't predict absolute direction (buy/sell).
Predict RELATIVE performance (stock A > stock B).

Approach:
  1. Compute continuous factor scores for all 500 stocks each day
  2. Rank stocks by composite factor score
  3. Hold top N stocks, rebalance periodically
  4. No ML model, no training — just factor math + ranking

This is what LambdaRank does, but with simple rank combination instead of ML.
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


def compute_all_factors(NS, ND, C, O, H, L, V):
    """Compute factor scores for all stocks. Returns dict of factor_name -> score[NS, ND]."""
    print("[Factors] Computing...", flush=True)
    t0 = time.time()
    factors = {}

    # Factor 1: Cross-sectional momentum (5-day return)
    print("  F1: Momentum5...", flush=True)
    mom5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di-5]) and C[si, di-5] > 0:
                mom5[si, di] = (C[si, di] - C[si, di-5]) / C[si, di-5]
    factors['mom5'] = mom5

    # Factor 2: Cross-sectional momentum (20-day return)
    print("  F2: Momentum20...", flush=True)
    mom20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di-20]) and C[si, di-20] > 0:
                mom20[si, di] = (C[si, di] - C[si, di-20]) / C[si, di-20]
    factors['mom20'] = mom20

    # Factor 3: VDP strength (10-day cumulative)
    print("  F3: VDP cumulative...", flush=True)
    vdp_cum = np.full((NS, ND), np.nan)
    for si in range(NS):
        c, h, l, v = C[si], H[si], L[si], V[si]
        valid = ~np.isnan(c)
        if np.sum(valid) < 30:
            continue
        vdp = compute_vdp(c, h, l, v)
        for di in range(10, ND):
            window = vdp[di-10:di]
            valid_w = window[~np.isnan(window)]
            if len(valid_w) >= 5:
                vdp_cum[si, di] = np.sum(valid_w)
    factors['vdp_cum'] = vdp_cum

    # Factor 4: RSI (14-day, INVERTED for mean reversion — lower RSI = higher score)
    print("  F4: RSI reversal...", flush=True)
    rsi14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        rsi = compute_rsi(C[si], 14)
        for di in range(14, ND):
            if not np.isnan(rsi[di]):
                # Invert: RSI 30 → score 70, RSI 70 → score 30
                rsi14[si, di] = 100 - rsi[di]
    factors['rsi_reversal'] = rsi14

    # Factor 5: Volume surge (current volume / 20-day average)
    print("  F5: Volume surge...", flush=True)
    vol_surge = np.full((NS, ND), np.nan)
    for si in range(NS):
        v = V[si]
        for di in range(20, ND):
            window = v[di-20:di]
            valid_w = window[~np.isnan(window)]
            if len(valid_w) >= 10 and np.mean(valid_w) > 0:
                vol_surge[si, di] = v[di] / np.mean(valid_w) if not np.isnan(v[di]) else np.nan
    factors['vol_surge'] = vol_surge

    # Factor 6: FRAMA direction (FRAMA slope)
    print("  F6: FRAMA slope...", flush=True)
    frama_slope = np.full((NS, ND), np.nan)
    for si in range(NS):
        c, h, l = C[si], H[si], L[si]
        valid = ~np.isnan(c)
        if np.sum(valid) < 60:
            continue
        frama = compute_frama(c, h, l, period=16)
        for di in range(17, ND):
            if not np.isnan(frama[di]) and not np.isnan(frama[di-1]):
                frama_slope[si, di] = (frama[di] - frama[di-1]) / frama[di-1] if frama[di-1] > 0 else 0
    factors['frama_slope'] = frama_slope

    # Factor 7: Bollinger %B position (INVERTED — lower %B = higher score for mean reversion)
    print("  F7: BB position...", flush=True)
    bb_pos = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        valid = ~np.isnan(c)
        if np.sum(valid) < 40:
            continue
        bb_up, bb_lo, bb_mid, bb_pct = compute_bollinger(c, 20, 2.0)
        for di in range(20, ND):
            if not np.isnan(bb_pct[di]):
                bb_pos[si, di] = 1.0 - bb_pct[di]  # Invert: low %B = high score
    factors['bb_reversal'] = bb_pos

    # Factor 8: Kaufman Efficiency Ratio
    print("  F8: KER...", flush=True)
    ker_arr = np.full((NS, ND), np.nan)
    for si in range(NS):
        ker = compute_ker(C[si], 10)
        for di in range(10, ND):
            if not np.isnan(ker[di]):
                ker_arr[si, di] = ker[di]
    factors['ker'] = ker_arr

    # Factor 9: Kalman velocity
    print("  F9: Kalman velocity...", flush=True)
    kal_vel = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        valid = ~np.isnan(c)
        if np.sum(valid) < 30:
            continue
        mean_c = np.nanmean(c)
        vel = compute_kalman_velocity(np.nan_to_num(c, nan=mean_c))
        for di in range(MIN_TRAIN, ND):
            kal_vel[si, di] = vel[di]
    factors['kalman_vel'] = kal_vel

    # Factor 10: Price position in 20-day range (inverted — near low = higher score)
    print("  F10: Range position...", flush=True)
    range_pos = np.full((NS, ND), np.nan)
    for si in range(NS):
        c, h, l = C[si], H[si], L[si]
        for di in range(20, ND):
            h20 = np.nanmax(h[di-19:di+1])
            l20 = np.nanmin(l[di-19:di+1])
            if h20 > l20 and not np.isnan(c[di]):
                pos = (c[di] - l20) / (h20 - l20)
                range_pos[si, di] = 1.0 - pos  # Near low = high score
    factors['range_reversal'] = range_pos

    print(f"  All factors computed ({time.time()-t0:.1f}s)", flush=True)
    return factors


def cross_sectional_rank(factor_scores, di):
    """Rank stocks by factor score on day di. Returns rank array (1=best, NS=worst).
    Higher score = better rank."""
    scores = factor_scores[:, di]
    valid = ~np.isnan(scores)
    if np.sum(valid) < 20:
        return None
    ranks = np.full(len(scores), np.nan)
    valid_indices = np.where(valid)[0]
    valid_scores = scores[valid_indices]
    # Rank: higher score = lower rank number (1 = best)
    order = np.argsort(-valid_scores)  # descending
    for rank, idx in enumerate(order):
        ranks[valid_indices[idx]] = rank + 1
    return ranks


def backtest_factor_portfolio(factor_weights, factors, NS, ND, dates, C, O, V,
                               top_n=20, rebalance_days=5, hold_period=5):
    """Backtest cross-sectional factor ranking portfolio.
    factor_weights: dict of factor_name -> weight
    factors: dict of factor_name -> score[NS, ND]
    """
    factor_names = list(factor_weights.keys())
    weights = np.array([factor_weights[f] for f in factor_names])

    cash = float(CASH0)
    holdings = []  # list of {'si', 'shares', 'entry', 'ed'}
    trades = []
    last_rebalance = -999

    for di in range(MIN_TRAIN, ND):
        # Check if we need to rebalance
        days_since_rebal = di - last_rebalance
        should_rebalance = days_since_rebal >= rebalance_days

        if should_rebalance:
            # Compute composite rank
            composite_rank = np.zeros(NS)
            rank_count = np.zeros(NS)
            for fname, w in zip(factor_names, weights):
                if fname not in factors:
                    continue
                ranks = cross_sectional_rank(factors[fname], di)
                if ranks is not None:
                    valid = ~np.isnan(ranks)
                    composite_rank[valid] += w * ranks[valid]
                    rank_count[valid] += abs(w)

            if np.sum(rank_count > 0) < top_n * 2:
                continue

            # Normalize by number of factors
            composite_rank[rank_count > 0] /= rank_count[rank_count > 0]
            composite_rank[rank_count == 0] = 9999

            # Sell current holdings not in new top N
            best_n = min(top_n, NS)
            top_indices = set(np.argsort(composite_rank)[:best_n])
            current_indices = set(h['si'] for h in holdings)

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
                            'pnl': pnl,
                            'days': (dates[di] - pos['ed']).days,
                            'di': di,
                            'reason': 'rebalance_sell'
                        })
                        holdings.remove(pos)

            # Buy new top N not currently held
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
                                    'si': si, 'shares': shares, 'entry': p, 'ed': dates[di]
                                })

            last_rebalance = di

    # Close remaining
    for pos in holdings:
        p = C[pos['si'], ND-1]
        if not np.isnan(p) and p > 0:
            pnl = (p - pos['entry']) / pos['entry'] * 100
            cash += pos['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
            trades.append({'pnl': pnl, 'days': 999, 'di': ND-1, 'reason': 'end'})

    if cash <= 0 or not trades:
        return None

    days_total = (dates[ND-1] - dates[MIN_TRAIN]).days
    yr = max(days_total / 365.25, 0.01)
    ann = ((cash / CASH0) ** (1/yr) - 1) * 100
    nw = sum(1 for t in trades if t['pnl'] > 0)
    wr = nw / max(len(trades), 1) * 100
    avg_w = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if nw > 0 else 0
    avg_l = np.mean([abs(t['pnl']) for t in trades if t['pnl'] <= 0]) if nw < len(trades) else 0

    # Drawdown
    equity = CASH0
    peak = CASH0
    max_dd = 0
    # Reconstruct equity curve from trades
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
    }


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V3 — Cross-Sectional Factor Ranking (No ML)", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Compute all factors
    factors = compute_all_factors(NS, ND, C, O, H, L, V)

    # Test different factor combinations
    print(f"\n[Backtest] Testing factor portfolios...", flush=True)
    results = []

    portfolios = {
        'Mom5_Only': {'mom5': 1.0},
        'Mom20_Only': {'mom20': 1.0},
        'VDP_Only': {'vdp_cum': 1.0},
        'RSI_Reversal': {'rsi_reversal': 1.0},
        'Vol_Surge': {'vol_surge': 1.0},
        'FRAMA_Dir': {'frama_slope': 1.0},
        'BB_Reversal': {'bb_reversal': 1.0},
        'KER_Only': {'ker': 1.0},
        'Kalman_Vel': {'kalman_vel': 1.0},
        'Range_Rev': {'range_reversal': 1.0},
        'Mom5+VDP': {'mom5': 0.5, 'vdp_cum': 0.5},
        'Mom5+VolSurge': {'mom5': 0.5, 'vol_surge': 0.5},
        'RSI+BB+Range': {'rsi_reversal': 0.33, 'bb_reversal': 0.33, 'range_reversal': 0.34},
        'Mom5+VDP+VolSurge': {'mom5': 0.4, 'vdp_cum': 0.3, 'vol_surge': 0.3},
        'ReversalAll': {'rsi_reversal': 0.25, 'bb_reversal': 0.25, 'range_reversal': 0.25, 'vdp_cum': 0.25},
        'MomentumAll': {'mom5': 0.3, 'mom20': 0.2, 'frama_slope': 0.2, 'kalman_vel': 0.3},
        'Balanced': {'mom5': 0.2, 'vdp_cum': 0.2, 'rsi_reversal': 0.2, 'vol_surge': 0.2, 'frama_slope': 0.2},
        'Full10': {k: 0.1 for k in ['mom5', 'mom20', 'vdp_cum', 'rsi_reversal', 'vol_surge',
                                      'frama_slope', 'bb_reversal', 'ker', 'kalman_vel', 'range_reversal']},
    }

    for top_n in [10, 20, 30, 50]:
        for rebal in [3, 5, 10]:
            for pname, weights in portfolios.items():
                r = backtest_factor_portfolio(weights, factors, NS, ND, dates, C, O, V,
                                               top_n=top_n, rebalance_days=rebal)
                if r:
                    results.append({
                        'portfolio': pname,
                        'top_n': top_n,
                        'rebal': rebal,
                        **r
                    })

            print(f"  top_n={top_n}, rebal={rebal} done", flush=True)

    # Sort by annualized return
    results.sort(key=lambda x: -x['ann'])

    # Print results
    print(f"\n{'='*100}", flush=True)
    print(f"  TOP 40 RESULTS", flush=True)
    print(f"  {'Portfolio':<25s} {'Top':>3s} {'Reb':>3s} | {'Ann':>7s} {'N':>4s} {'WR':>5s} "
          f"{'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*95}", flush=True)
    for r in results[:40]:
        print(f"  {r['portfolio']:<25s} {r['top_n']:3d} {r['rebal']:3d} | {r['ann']:+7.1f}% {r['n']:4d} "
              f"{r['wr']:5.1f}% {r['edge']:+6.2f}% {r['max_dd']:5.1f}%", flush=True)

    # Best per portfolio
    print(f"\n  Best per portfolio:", flush=True)
    best_per = {}
    for r in results:
        p = r['portfolio']
        if p not in best_per or r['ann'] > best_per[p]['ann']:
            best_per[p] = r
    for r in sorted(best_per.values(), key=lambda x: -x['ann']):
        print(f"    {r['portfolio']:<25s} → {r['ann']:+.1f}% (Top={r['top_n']}, Reb={r['rebal']}, "
              f"WR={r['wr']:.0f}%, Edge={r['edge']:+.2f}%, DD={r['max_dd']:.1f}%)", flush=True)

    print(f"\n{'='*70}", flush=True)
    above_0 = sum(1 for r in best_per.values() if r['ann'] > 0)
    above_20 = sum(1 for r in best_per.values() if r['ann'] > 20)
    above_50 = sum(1 for r in best_per.values() if r['ann'] > 50)
    print(f"  Portfolios > 0%: {above_0}/{len(best_per)}", flush=True)
    print(f"  Portfolios > 20%: {above_20}/{len(best_per)}", flush=True)
    print(f"  Portfolios > 50%: {above_50}/{len(best_per)}", flush=True)
    print(f"{'='*70}", flush=True)
