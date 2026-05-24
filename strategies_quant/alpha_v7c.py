"""
Alpha V7c — Regime Filter + Optimization (BUG-FIXED)
=====================================================
V7b结果: StructQual +108.9%, 但2023年-39%, DD=67.5%

改进:
  1. 市场状态过滤: 大盘弱势时降低仓位/不交易
  2. 动态Top N: 波动率高时少持仓
  3. 更好的止损: 结合时间止损和ATR止损
  4. 尝试更多因子组合
  5. 加入行业分散: 不同行业的Top 1

BUG FIX (V7h审计):
  - ATR止损不再使用当日收盘价检查后以开盘价卖出 (look-ahead)
  - 使用当日最低价检查止损触发, 以止损价卖出 (realistic)
  - hw用当日最高价更新, 且只在止损未触发后更新
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.data_loader import list_available_symbols, load_stock_data
from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7 import compute_all_factors, COMMISSION, STAMP_DUTY, CASH0


def compute_market_regime(NS, ND, C, O, H, L, V, syms):
    """Compute market-wide regime signals using index proxy (top stocks average)."""
    # Use top 50 stocks by volume as market proxy
    avg_vol = np.zeros(ND)
    for di in range(ND):
        vols = V[:50, di]
        avg_vol[di] = np.nanmean(vols)

    # Market momentum: 20-day return of top-50 average
    market_ret20 = np.full(ND, np.nan)
    market_ret5 = np.full(ND, np.nan)
    # Market breadth: fraction of stocks above 20-day MA
    breadth = np.full(ND, np.nan)

    # Compute average close for top 50 stocks
    avg_close = np.zeros(ND)
    for di in range(ND):
        closes = C[:50, di]
        avg_close[di] = np.nanmean(closes)

    for di in range(21, ND):
        if not np.isnan(avg_close[di - 1]) and not np.isnan(avg_close[di - 1 - 20]) and avg_close[di - 1 - 20] > 0:
            market_ret20[di] = (avg_close[di - 1] - avg_close[di - 1 - 20]) / avg_close[di - 1 - 20]
        if not np.isnan(avg_close[di - 1]) and not np.isnan(avg_close[di - 1 - 5]) and avg_close[di - 1 - 5] > 0:
            market_ret5[di] = (avg_close[di - 1] - avg_close[di - 1 - 5]) / avg_close[di - 1 - 5]

        # Market breadth: fraction of stocks where C > MA20
        above_ma = 0
        total = 0
        for si in range(min(200, NS)):
            c = C[si, di - 1]
            if np.isnan(c):
                continue
            c20 = C[si, di - 1 - 20:di]
            valid = c20[~np.isnan(c20)]
            if len(valid) < 15:
                continue
            ma20 = np.mean(valid)
            if c > ma20:
                above_ma += 1
            total += 1
        if total > 50:
            breadth[di] = above_ma / total * 100

    return market_ret20, market_ret5, breadth


def backtest_v7c(factor_weights, factors, NS, ND, dates, C, O, H, L, V,
                 top_n=3, rebalance_days=10, atr_stop_mult=2.0,
                 market_ret20=None, breadth=None,
                 regime_filter=False, min_breadth=40, min_mkt_ret=-0.05,
                 position_scale=True):
    """Backtest with BUG-FIXED ATR stop loss — no look-ahead.

    FIX: ATR stop uses correct execution model:
    1. Compute stop from PREVIOUS hw (data up to di-1)
    2. Check if L[si,di] <= stop (realistic: intraday low hit the stop)
    3. If triggered: sell at stop price (realistic stop-loss execution)
    4. If NOT triggered: update hw = max(hw, H[si,di]) (use intraday high)
    """
    factor_names = list(factor_weights.keys())
    weights = np.array([factor_weights[f] for f in factor_names])

    cash = float(CASH0)
    holdings = []
    trades = []
    last_rebalance = -999
    year_stats = {}

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # ATR stop loss check — FIXED: no look-ahead
        for pos in list(holdings):
            si = pos['si']
            stopped_out = False

            if atr_stop_mult > 0:
                # Compute ATR from past 14 days (data up to di-1)
                atr = 0
                atr_count = 0
                for dd in range(max(di - 14, 1), di):
                    if not np.isnan(H[si, dd]) and not np.isnan(L[si, dd]):
                        tr = H[si, dd] - L[si, dd]
                        if not np.isnan(C[si, dd - 1]):
                            tr = max(tr, abs(H[si, dd] - C[si, dd - 1]),
                                     abs(L[si, dd] - C[si, dd - 1]))
                        atr += tr
                        atr_count += 1
                if atr_count > 0:
                    atr /= atr_count
                else:
                    atr = 0

                if atr > 0:
                    # Stop = previous hw - ATR*mult (hw from BEFORE today)
                    stop = pos['hw'] - atr_stop_mult * atr

                    # Check if today's LOW hit the stop (realistic)
                    today_low = L[si, di]
                    today_open = O[si, di]

                    if not np.isnan(today_low) and today_low <= stop:
                        # Stop triggered: sell at stop price (realistic)
                        # But if open gapped below stop, sell at open (worse)
                        if not np.isnan(today_open) and today_open < stop:
                            sp = today_open  # Gap down — sell at open
                        else:
                            sp = stop  # Normal stop execution
                        pnl = (sp - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * sp * (1 - COMMISSION - STAMP_DUTY)
                        trades.append({'pnl': pnl, 'days': (dates[di] - pos['ed']).days,
                                       'di': di, 'reason': 'stop', 'year': year})
                        holdings.remove(pos)
                        stopped_out = True

            if not stopped_out:
                # Update hw with today's HIGH (only if not stopped out)
                today_high = H[si, di]
                if not np.isnan(today_high) and today_high > 0:
                    pos['hw'] = max(pos['hw'], today_high)

            # Time-based stop: hold max 60 days
            if pos in holdings:
                days_held = (dates[di] - pos['ed']).days
                if days_held >= 60:
                    sp = O[si, di] if not np.isnan(O[si, di]) else C[si, di]
                    if not np.isnan(sp) and sp > 0:
                        pnl = (sp - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * sp * (1 - COMMISSION - STAMP_DUTY)
                        trades.append({'pnl': pnl, 'days': days_held,
                                       'di': di, 'reason': 'time_stop', 'year': year})
                        holdings.remove(pos)

        # Rebalance
        if di - last_rebalance >= rebalance_days:
            # Regime filter
            skip = False
            scale = 1.0
            if regime_filter and market_ret20 is not None and breadth is not None:
                mr = market_ret20[di]
                br = breadth[di]
                if not np.isnan(br) and br < min_breadth:
                    skip = True  # Too few stocks above MA — bear market
                if not np.isnan(mr) and mr < min_mkt_ret:
                    scale = 0.5  # Market declining — reduce exposure

            if skip:
                # Sell all holdings
                for pos in list(holdings):
                    p = O[pos['si'], di]
                    if np.isnan(p) or p <= 0:
                        p = C[pos['si'], di]
                    if not np.isnan(p) and p > 0:
                        pnl = (p - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
                        trades.append({'pnl': pnl, 'days': (dates[di] - pos['ed']).days,
                                       'di': di, 'reason': 'regime_exit', 'year': year})
                holdings = []
                last_rebalance = di
                continue

            # Composite score — factors at di use data up to di-1
            # (v7+ convention: FACTOR[:, di] computed from C[:, di-1] etc.)
            # (v48+ _rolling_mean/ema also fixed to use di-1 data)
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
                        trades.append({'pnl': pnl, 'days': (dates[di] - pos['ed']).days,
                                       'di': di, 'reason': 'rebalance', 'year': year})
                        holdings.remove(pos)

            # Buy
            current_indices = set(h['si'] for h in holdings)
            to_buy = top_indices - current_indices
            n_to_buy = len(to_buy)
            if n_to_buy > 0 and cash > 10000:
                alloc = cash / n_to_buy * scale  # Scale position by regime
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
                                    'ed': dates[di], 'hw': p
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

    if not trades:
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

    return {
        'ann': round(ann, 1), 'n': len(trades), 'wr': round(wr, 1),
        'avg_w': round(avg_w, 1), 'avg_l': round(avg_l, 1),
        'edge': round((nw / max(len(trades), 1)) * avg_w - (1 - nw / max(len(trades), 1)) * avg_l, 2),
        'max_dd': round(max_dd, 1), 'tpy': round(len(trades) / yr, 1),
        'final': round(cash, 0), 'year_stats': year_stats,
    }


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V7c — Regime Filter + Optimization (BUG-FIXED)", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()
    factors = compute_all_factors(NS, ND, C, O, H, L, V)

    # Compute interaction factors inline
    from alpha_v7b import compute_interaction_factors
    inter_factors = compute_interaction_factors(factors, NS, ND, C, O, H, L, V)
    all_factors = {**factors, **inter_factors}

    # Compute market regime
    print("[Regime] Computing market regime...", flush=True)
    mkt_ret20, mkt_ret5, breadth = compute_market_regime(NS, ND, C, O, H, L, V, syms)
    print("  Market regime done", flush=True)

    # StructQual — the winning combination from V7b
    structqual = {'R_TENSION': 0.25, 'R_R_SQUARED': 0.25,
                  'R_TENS_SHAD': 0.25, 'R_BODY_VOL': 0.25}
    fisherstruct = {'R_FISHER': 0.3, 'R_TENSION': 0.3,
                    'R_R_SQUARED': 0.2, 'R_VOLATILITY_PCT': 0.2}

    portfolios = {
        'StructQual': structqual,
        'FisherStruct': fisherstruct,
    }

    results = []
    for pname, weights in portfolios.items():
        for top_n in [3, 5]:
            for rebal in [10, 15, 20]:
                for atr in [1.5, 2.0, 2.5]:
                    # Without regime filter
                    r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                    top_n=top_n, rebalance_days=rebal, atr_stop_mult=atr,
                                    market_ret20=mkt_ret20, breadth=breadth,
                                    regime_filter=False)
                    if r:
                        r.update({'portfolio': pname, 'top_n': top_n, 'rebal': rebal,
                                  'atr': atr, 'regime': 'off'})
                        results.append(r)

                    # With regime filter
                    for min_br in [35, 40, 45]:
                        r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                        top_n=top_n, rebalance_days=rebal, atr_stop_mult=atr,
                                        market_ret20=mkt_ret20, breadth=breadth,
                                        regime_filter=True, min_breadth=min_br)
                        if r:
                            r.update({'portfolio': pname, 'top_n': top_n, 'rebal': rebal,
                                      'atr': atr, 'regime': f'br>{min_br}'})
                            results.append(r)
        print(f"  {pname} done", flush=True)

    results.sort(key=lambda x: -x['ann'])

    print(f"\n{'='*120}", flush=True)
    print(f"  TOP 30 RESULTS", flush=True)
    print(f"  {'Portfolio':<15s} {'Top':>3s} {'Reb':>3s} {'ATR':>3s} {'Regime':<8s} | "
          f"{'Ann':>7s} {'N':>5s} {'TPY':>4s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*110}", flush=True)
    for r in results[:30]:
        print(f"  {r['portfolio']:<15s} {r['top_n']:3d} {r['rebal']:3d} {r['atr']:3.1f} {r['regime']:<8s} | "
              f"{r['ann']:+7.1f}% {r['n']:5d} {r['tpy']:4.0f} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%", flush=True)

    # Best per portfolio
    best_per = {}
    for r in results:
        key = f"{r['portfolio']}_{r['regime']}"
        if key not in best_per or r['ann'] > best_per[key]['ann']:
            best_per[key] = r
    print(f"\n  Best per configuration:", flush=True)
    for r in sorted(best_per.values(), key=lambda x: -x['ann'])[:15]:
        print(f"    {r['portfolio']:<15s} regime={r['regime']:<8s} → {r['ann']:+.1f}% "
              f"(Top={r['top_n']}, Reb={r['rebal']}, ATR={r['atr']:.1f}, DD={r['max_dd']:.1f}%)", flush=True)

    # Year-by-year for best
    if results:
        best = results[0]
        print(f"\n  Year-by-year: {best['portfolio']} regime={best['regime']} "
              f"(Ann={best['ann']:+.1f}%, DD={best['max_dd']:.1f}%)", flush=True)
        for y in sorted(best.get('year_stats', {}).keys()):
            s = best['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

        # Compare with and without regime filter
        print(f"\n  === REGIME FILTER COMPARISON ===", flush=True)
        for pname in ['StructQual', 'FisherStruct']:
            no_regime = [r for r in results if r['portfolio'] == pname and r['regime'] == 'off']
            with_regime = [r for r in results if r['portfolio'] == pname and r['regime'] != 'off']
            if no_regime and with_regime:
                best_no = max(no_regime, key=lambda x: x['ann'])
                best_yes = max(with_regime, key=lambda x: x['ann'])
                print(f"  {pname}:", flush=True)
                print(f"    No filter:   {best_no['ann']:+.1f}% DD={best_no['max_dd']:.1f}%", flush=True)
                print(f"    With filter: {best_yes['ann']:+.1f}% DD={best_yes['max_dd']:.1f}% "
                      f"(regime={best_yes['regime']})", flush=True)

    print(f"\n{'='*70}", flush=True)
