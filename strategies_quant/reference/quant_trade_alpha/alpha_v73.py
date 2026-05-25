"""
Alpha V73 — Stock-Specific Adaptive Stop + Break-Even
======================================================
V62 baseline: +522% with fixed ATR=0.1 multiplier.
V62 with fixed ATR=2.0: +277%/48.2% WR (wider stops).

Insight: Different stocks have different volatility profiles.
A stock with 2% daily range needs different stops than one with 6% range.

Approach: Instead of a fixed ATR multiplier, use a stock-specific adaptive
stop that targets a consistent stop distance as a percentage of price:

    stop_distance_pct = target_stop_pct (e.g. 8%)
    actual_atr_mult = target_stop_pct / stock_ATR_pct
    stop = hw - adaptive_atr_mult * ATR

This means:
- Low-vol stock (ATR%=2%): gets ATR*4.0 (wider relative to ATR, but 8% from high)
- High-vol stock (ATR%=6%): gets ATR*1.33 (tighter relative to ATR, but 8% from high)

Also adds break-even stop: after price rises +5% from entry, raise stop to entry.
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.data_loader import list_available_symbols, load_stock_data
from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7 import compute_all_factors, COMMISSION, STAMP_DUTY, CASH0


def backtest_v73(factor_weights, factors, NS, ND, dates, C, O, H, L, V,
                 top_n=1, rebalance_days=10,
                 target_stop_pct=8.0,     # Target stop distance as % of price
                 be_threshold=0.05,       # Break-even at +5%
                 min_atr_mult=0.5,        # Minimum ATR multiplier
                 max_atr_mult=5.0,        # Maximum ATR multiplier
                 market_ret20=None, breadth=None,
                 regime_filter=False, min_breadth=40):
    """Backtest with stock-specific adaptive ATR stop + break-even protection.

    Stop-loss logic (no look-ahead, same as V7c bug-fixed):
    1. Compute stock-specific ATR over past 14 days
    2. Compute ATR as percentage of price: atr_pct = ATR / hw * 100
    3. Adaptive multiplier: adapt_mult = target_stop_pct / atr_pct
    4. Clamp to [min_atr_mult, max_atr_mult]
    5. Stop = hw - adapt_mult * ATR
    6. If price rose > be_threshold from entry, raise stop to entry (break-even)
    7. Check if L[si,di] <= stop (realistic intraday)
    8. If triggered: sell at stop price
    9. If NOT triggered: update hw = max(hw, H[si,di])
    """
    factor_names = list(factor_weights.keys())
    weights = np.array([factor_weights[f] for f in factor_names])

    cash = float(CASH0)
    holdings = []
    trades = []
    last_rebalance = -999
    year_stats = {}
    daily_nav = []

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # Adaptive ATR stop loss check
        for pos in list(holdings):
            si = pos['si']
            stopped_out = False

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
                hw = pos['hw']

                # Compute adaptive ATR multiplier
                # ATR as percentage of the high-water mark
                atr_pct = (atr / hw) * 100.0 if hw > 0 else 0

                if atr_pct > 0:
                    adaptive_mult = target_stop_pct / atr_pct
                else:
                    adaptive_mult = max_atr_mult

                # Clamp
                adaptive_mult = max(min_atr_mult, min(max_atr_mult, adaptive_mult))

                # Stop price from adaptive ATR
                stop = hw - adaptive_mult * atr

                # Break-even protection: if unrealized gain > be_threshold,
                # raise stop to at least entry price
                if be_threshold > 0:
                    current_price = C[si, di - 1] if di > 0 and not np.isnan(C[si, di - 1]) else hw
                    if current_price > 0:
                        unrealized = (current_price - pos['entry']) / pos['entry']
                        if unrealized >= be_threshold:
                            # Raise stop to entry (break-even)
                            stop = max(stop, pos['entry'])

                # Check if today's LOW hit the stop (realistic)
                today_low = L[si, di]
                today_open = O[si, di]

                if not np.isnan(today_low) and today_low <= stop:
                    # Stop triggered
                    if not np.isnan(today_open) and today_open < stop:
                        sp = today_open  # Gap down
                    else:
                        sp = stop
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
                    skip = True
                if not np.isnan(mr) and mr < -0.05:
                    scale = 0.5

            if skip:
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

            # Composite score
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
                alloc = cash / n_to_buy * scale
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

        # Track daily NAV
        nav = cash
        for pos in holdings:
            cp = C[pos['si'], di]
            if np.isnan(cp) or cp <= 0:
                cp = pos['entry']
            nav += pos['shares'] * cp
        daily_nav.append(nav)

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

    # Drawdown from daily NAV
    max_dd = 0
    if daily_nav:
        peak = daily_nav[0]
        for nav in daily_nav:
            if nav > peak:
                peak = nav
            if peak > 0:
                dd = (peak - nav) / peak * 100
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
    print("  Alpha V73 — Stock-Specific Adaptive Stop + Break-Even")
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Quick sanity test with V62 weights
    from alpha_v62 import compute_v62_factors, V62_WEIGHTS
    factors = compute_v62_factors(NS, ND, C, O, H, L, V)

    # Fixed ATR=2.0 baseline
    from alpha_v7c import backtest_v7c
    r_base = backtest_v7c(V62_WEIGHTS, factors, NS, ND, dates, C, O, H, L, V,
                          top_n=1, rebalance_days=5, atr_stop_mult=2.0)
    print(f"\n  Baseline (V62 + fixed ATR=2.0): {r_base['ann']:+.1f}% WR={r_base['wr']:.1f}% DD={r_base['max_dd']:.1f}%",
          flush=True)

    # Adaptive stop
    r_adapt = backtest_v73(V62_WEIGHTS, factors, NS, ND, dates, C, O, H, L, V,
                           top_n=1, rebalance_days=5,
                           target_stop_pct=8.0, be_threshold=0.05)
    print(f"  Adaptive (8% target, BE@5%): {r_adapt['ann']:+.1f}% WR={r_adapt['wr']:.1f}% DD={r_adapt['max_dd']:.1f}%",
          flush=True)

    print(f"\n{'='*70}", flush=True)
