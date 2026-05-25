"""
Alpha V5 — Low-Frequency Factor Ranking
========================================
V4 问题: 700+ trades/年, 太频繁, 不现实
V5 改进:
  1. 换仓门槛 — 只有新股票排名显著高于持仓才换
  2. 最小持仓天数 — 买入后至少持 N 天
  3. 更长再平衡周期 — 5/10/20 天
  4. 交易成本优化 — 换仓收益必须 > 交易成本
  5. 目标: 年交易 < 100 次, 同时保持正收益
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.data_loader import list_available_symbols, load_stock_data
from alpha_v2 import load_all_data, compute_frama, compute_vdp, compute_rsi, \
    compute_ker, compute_bollinger, compute_kalman_velocity, MIN_TRAIN
from alpha_v4 import compute_v75_factors

COMMISSION = 0.0003
STAMP_DUTY = 0.001
CASH0 = 500_000
COST_ROUND_TRIP = (1 - (1-COMMISSION)*(1-COMMISSION-STAMP_DUTY))  # ~0.13% per round trip


def backtest_v5(factor_weights, factors, NS, ND, dates, C, O, H, L, V,
                top_n=10, rebalance_days=5, min_hold_days=5,
                rank_improvement_threshold=20, atr_exit=True):
    """
    Low-frequency factor ranking.
    rank_improvement_threshold: new stock must rank at least this many positions
                                better than current holding to trigger a swap.
    min_hold_days: minimum days to hold before allowing swap.
    """
    factor_names = list(factor_weights.keys())
    weights = np.array([factor_weights[f] for f in factor_names])

    cash = float(CASH0)
    holdings = []  # {'si', 'shares', 'entry', 'ed', 'rank_at_entry'}
    trades = []
    last_rebalance = -999

    year_stats = {}

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # Rebalance check
        days_since_rebal = di - last_rebalance
        should_rebalance = days_since_rebal >= rebalance_days

        if should_rebalance:
            # Composite score
            composite = np.zeros(NS)
            count = np.zeros(NS)
            for fname, w in zip(factor_names, weights):
                if fname not in factors: continue
                arr = factors[fname]
                vals = arr[:, di - 1]  # Use di-1 to avoid look-ahead bias
                valid = ~np.isnan(vals)
                if valid.sum() < 50: continue
                composite[valid] += w * vals[valid]
                count[valid] += abs(w)

            mask = count > 0
            if mask.sum() < top_n * 2:
                continue
            composite[mask] /= count[mask]
            composite[~mask] = -9999

            # Rank all stocks (1=best)
            ranks = np.full(NS, 9999.0)
            order = np.argsort(-composite)  # higher score = better rank
            for rank_idx, si in enumerate(order):
                ranks[si] = rank_idx + 1

            # For each holding, check if there's a significantly better stock
            current_set = {h['si']: h for h in holdings}
            to_sell = []
            to_buy_candidates = []

            # Find all stocks in top N that we don't hold
            top_set = set(order[:top_n])
            not_held_in_top = top_set - set(current_set.keys())

            for si_not_held in not_held_in_top:
                to_buy_candidates.append(si_not_held)

            # For each held stock NOT in top N, check if swapping is worthwhile
            for si_held, pos in current_set.items():
                days_held = (dates[di] - pos['ed']).days
                if days_held < min_hold_days:
                    continue  # Don't sell if held less than minimum

                current_rank = ranks[si_held]
                if current_rank > top_n * 2:
                    # Current stock rank is far outside top N — sell
                    to_sell.append(si_held)
                elif current_rank > top_n:
                    # Current stock dropped out of top N
                    # Check if any candidate is significantly better
                    best_candidate_rank = min(ranks[c] for c in to_buy_candidates) if to_buy_candidates else 9999
                    if best_candidate_rank < current_rank - rank_improvement_threshold:
                        to_sell.append(si_held)

            # Execute sells
            for si in to_sell:
                pos = current_set[si]
                p = O[si, di]
                if np.isnan(p) or p <= 0: p = C[si, di]
                if not np.isnan(p) and p > 0:
                    pnl = (p - pos['entry']) / pos['entry'] * 100
                    cash += pos['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
                    trades.append({
                        'pnl': pnl, 'days': (dates[di] - pos['ed']).days,
                        'di': di, 'reason': 'swap_sell', 'year': year
                    })
                    holdings = [h for h in holdings if h['si'] != si]

            # Buy new stocks
            n_slots = top_n - len(holdings)
            if len(to_sell) > 0:
                n_to_buy = min(len(to_sell), n_slots)
            else:
                n_to_buy = n_slots  # Initial buy when no holdings
            if n_to_buy > 0 and cash > 10000:
                # Pick top candidates by rank
                candidates_sorted = sorted(to_buy_candidates, key=lambda s: ranks[s])[:n_to_buy]
                alloc = cash / max(n_to_buy, 1)
                for si in candidates_sorted:
                    p = O[si, di]
                    if np.isnan(p) or p <= 0: p = C[si, di]
                    if not np.isnan(p) and p > 0:
                        shares = int(alloc / (1 + COMMISSION) / p)
                        if shares > 0:
                            cost = shares * p * (1 + COMMISSION)
                            if cost <= cash:
                                cash -= cost
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
                                        atr_sl = p - 2.5 * atr
                                holdings.append({
                                    'si': si, 'shares': shares, 'entry': p,
                                    'ed': dates[di], 'atr_sl': atr_sl, 'rank': ranks[si]
                                })

            last_rebalance = di

        # ATR stop loss check
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

    # Year-by-year
    for t in trades:
        y = t.get('year', 'unknown')
        if y not in year_stats:
            year_stats[y] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
        year_stats[y]['trades'] += 1
        if t['pnl'] > 0: year_stats[y]['wins'] += 1
        year_stats[y]['total_pnl'] += t['pnl']

    # DD
    equity = float(CASH0); peak = float(CASH0); max_dd = 0
    for t in sorted(trades, key=lambda x: x['di']):
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
    print("  Alpha V5 — Low-Frequency Factor Ranking", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()
    factors = compute_v75_factors(NS, ND, C, O, H, L, V)

    # Best portfolios from V4
    portfolios = {
        'Enhanced': {'R_MOM5': 0.05, 'R_MOM10': 0.05, 'R_FRAMA': 0.1, 'R_KAL': 0.1,
                     'R_VDP': 0.1, 'R_REL_VOL': 0.1,
                     'D_MOM5_3': 0.05, 'D_FRAMA_3': 0.1, 'D_KAL_3': 0.1,
                     'D_VDP_5': 0.1, 'D_REL_VOL_5': 0.1},
        'FRAMA_Only': {'R_FRAMA': 0.5, 'D_FRAMA_3': 0.5},
        'MOM5_Only': {'R_MOM5': 0.5, 'D_MOM5_3': 0.5},
        'MOM5_VDP': {'R_MOM5': 0.3, 'R_VDP': 0.3, 'D_VDP_5': 0.4},
    }

    results = []
    for pname, weights in portfolios.items():
        for top_n in [10, 20]:
            for rebal in [5, 10, 20]:
                for min_hold in [5, 10]:
                    for rank_thresh in [10, 20, 30]:
                        r = backtest_v5(weights, factors, NS, ND, dates, C, O, H, L, V,
                                         top_n=top_n, rebalance_days=rebal,
                                         min_hold_days=min_hold,
                                         rank_improvement_threshold=rank_thresh)
                        if r:
                            results.append({
                                'portfolio': pname,
                                'top_n': top_n,
                                'rebal': rebal,
                                'min_hold': min_hold,
                                'rank_thresh': rank_thresh,
                                **r
                            })
        print(f"  {pname} done", flush=True)

    results.sort(key=lambda x: -x['ann'])

    print(f"\n{'='*120}", flush=True)
    print(f"  TOP 40 RESULTS (sorted by annualized return)", flush=True)
    print(f"  {'Portfolio':<15s} {'Top':>3s} {'Reb':>3s} {'Hold':>4s} {'RTh':>3s} | "
          f"{'Ann':>7s} {'N':>4s} {'TPY':>4s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*115}", flush=True)
    for r in results[:40]:
        print(f"  {r['portfolio']:<15s} {r['top_n']:3d} {r['rebal']:3d} {r['min_hold']:4d} {r['rank_thresh']:3d} | "
              f"{r['ann']:+7.1f}% {r['n']:4d} {r['tpy']:4.0f} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%", flush=True)

    # Filter: realistic trading frequency (< 150 trades/year)
    print(f"\n\n  --- REALISTIC RESULTS (TPY < 150) ---", flush=True)
    realistic = [r for r in results if r['tpy'] < 150]
    realistic.sort(key=lambda x: -x['ann'])
    print(f"  {'Portfolio':<15s} {'Top':>3s} {'Reb':>3s} {'Hold':>4s} {'RTh':>3s} | "
          f"{'Ann':>7s} {'N':>4s} {'TPY':>4s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*115}", flush=True)
    for r in realistic[:30]:
        print(f"  {r['portfolio']:<15s} {r['top_n']:3d} {r['rebal']:3d} {r['min_hold']:4d} {r['rank_thresh']:3d} | "
              f"{r['ann']:+7.1f}% {r['n']:4d} {r['tpy']:4.0f} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%", flush=True)

    # Year-by-year for best realistic
    if realistic:
        best = realistic[0]
        print(f"\n  Best realistic year-by-year: {best['portfolio']} (Ann={best['ann']:+.1f}%, TPY={best['tpy']:.0f})", flush=True)
        ys = best.get('year_stats', {})
        for y in sorted(ys.keys()):
            s = ys[y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            tpy = s['trades']
            print(f"    {y}: {tpy:3d} trades, WR={wr:.0f}%, total_pnl={s['total_pnl']:+.0f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
