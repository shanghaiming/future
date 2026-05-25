"""
Alpha V72 — Ultimate Combined Strategy
=======================================
Integrates ALL insights from 100+ configurations across 9 agents:

1. Stop widening is the #1 lever: ATR=2.0/R10 gives +277%/48.2% WR
2. V62's 6 factors are sufficient — new factors don't add alpha
3. Entry quality matters — composite score percentile filter
4. Market regime matters — bear defense via breadth filter

New features vs V7c:
  1. Composite score filter: skip entries below min_composite_pct percentile
  2. Trend filter: only enter if stock above its 20-day MA
  3. Break-even stop: after +be_threshold gain, raise stop to entry
  4. Bear defense: when breadth < min_breadth, sell all and wait
  5. Volatility scaling: position size inversely proportional to ATR
  6. Hard loss cap: exit if unrealized loss > max_loss_pct

Bug-fixed ATR stop (from V7c): no look-ahead, uses L[si,di] for trigger.
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


def backtest_v72(factor_weights, factors, NS, ND, dates, C, O, H, L, V,
                 top_n=1, rebalance_days=10, atr_stop_mult=2.0,
                 # NEW params
                 min_composite_pct=85,      # Only enter when score > 85th percentile
                 trend_filter=True,          # Only buy stocks above MA20
                 be_threshold=0.05,          # Break-even stop at +5%
                 bear_defense=True,          # Reduce position in bear markets
                 min_breadth=45,             # Minimum market breadth to trade
                 vol_scale=True,             # Size positions inversely to volatility
                 max_loss_pct=15,            # Maximum loss per position (hard stop)
                 # Pre-computed regime data
                 breadth=None,
                 ):
    """Backtest with ALL V72 features.

    Bug-fixed ATR stop from V7c (no look-ahead):
      1. Compute stop from PREVIOUS hw (data up to di-1)
      2. Check if L[si,di] <= stop (realistic: intraday low hit the stop)
      3. If triggered: sell at stop price (realistic stop-loss execution)
      4. If NOT triggered: update hw = max(hw, H[si,di])

    V72 additions:
      - Composite score filter: skip low-conviction entries
      - Trend filter: stock must be above 20-day MA
      - Break-even stop: lock in gains after be_threshold move
      - Bear defense: exit all when market breadth too low
      - Volatility scaling: size inversely to stock ATR
      - Hard loss cap: absolute max loss per position
    """
    factor_names = list(factor_weights.keys())
    weights = np.array([factor_weights[f] for f in factor_names])

    cash = float(CASH0)
    holdings = []
    trades = []
    last_rebalance = -999
    year_stats = {}
    daily_nav = []  # Track daily NAV for proper DD calculation

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # =====================================================================
        # STOP LOSS CHECK (ATR + break-even + hard cap)
        # =====================================================================
        for pos in list(holdings):
            si = pos['si']
            stopped_out = False
            sell_reason = ''
            sp = 0.0

            # --- Hard loss cap ---
            if max_loss_pct > 0:
                today_low = L[si, di]
                today_open = O[si, di]
                if not np.isnan(today_low) and not np.isnan(pos['entry']) and pos['entry'] > 0:
                    max_loss_price = pos['entry'] * (1 - max_loss_pct / 100.0)
                    if today_low <= max_loss_price:
                        if not np.isnan(today_open) and today_open < max_loss_price:
                            sp = today_open
                        else:
                            sp = max_loss_price
                        stopped_out = True
                        sell_reason = 'hard_cap'

            # --- ATR stop loss (bug-fixed, no look-ahead) ---
            if not stopped_out and atr_stop_mult > 0:
                # Compute ATR from past 14 days (data up to di-1)
                atr = 0.0
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
                    # Break-even stop logic: if unrealized gain > be_threshold,
                    # raise the stop to entry price
                    stop = pos['hw'] - atr_stop_mult * atr
                    if be_threshold > 0:
                        # Check if we've gained be_threshold from entry
                        hw_gain = (pos['hw'] - pos['entry']) / pos['entry']
                        if hw_gain >= be_threshold:
                            # Raise stop to at least entry price (break-even)
                            stop = max(stop, pos['entry'])

                    # Check if today's LOW hit the stop (realistic)
                    today_low = L[si, di]
                    today_open = O[si, di]

                    if not np.isnan(today_low) and today_low <= stop:
                        # Stop triggered: sell at stop price (realistic)
                        # But if open gapped below stop, sell at open (worse)
                        if not np.isnan(today_open) and today_open < stop:
                            sp = today_open  # Gap down -- sell at open
                        else:
                            sp = stop  # Normal stop execution
                        stopped_out = True
                        sell_reason = 'stop'

            # Execute stop
            if stopped_out and sp > 0:
                pnl = (sp - pos['entry']) / pos['entry'] * 100
                cash += pos['shares'] * sp * (1 - COMMISSION - STAMP_DUTY)
                trades.append({'pnl': pnl, 'days': (dates[di] - pos['ed']).days,
                               'di': di, 'reason': sell_reason, 'year': year})
                holdings.remove(pos)
                continue

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

        # =====================================================================
        # BEAR DEFENSE: if market breadth too low, sell all and wait
        # =====================================================================
        if bear_defense and breadth is not None:
            br = breadth[di]
            if not np.isnan(br) and br < min_breadth:
                # Sell all holdings at open
                for pos in list(holdings):
                    p = O[pos['si'], di]
                    if np.isnan(p) or p <= 0:
                        p = C[pos['si'], di]
                    if not np.isnan(p) and p > 0:
                        pnl = (p - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
                        trades.append({'pnl': pnl, 'days': (dates[di] - pos['ed']).days,
                                       'di': di, 'reason': 'bear_exit', 'year': year})
                holdings = []
                # Skip rebalance on this day
                last_rebalance = di
                # Still track NAV
                daily_nav.append(cash)
                continue

        # =====================================================================
        # REBALANCE
        # =====================================================================
        if di - last_rebalance >= rebalance_days:
            # Composite score -- factors at di use data up to di-1
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
                daily_nav.append(cash)
                continue
            composite[mask] /= count[mask]
            composite[~mask] = -9999

            # --- Composite score filter ---
            # Compute the percentile threshold: only consider stocks above min_composite_pct
            valid_composite = composite[mask]
            if len(valid_composite) > 10:
                pct_threshold = np.percentile(valid_composite, min_composite_pct)
            else:
                pct_threshold = -9999

            # --- Trend filter: stock must be above 20-day MA ---
            trend_ok = np.ones(NS, dtype=bool)
            if trend_filter:
                for si in range(NS):
                    c_prev = C[si, di - 1]
                    if np.isnan(c_prev):
                        trend_ok[si] = False
                        continue
                    c20 = C[si, max(di - 1 - 20, 0):di]
                    valid_c = c20[~np.isnan(c20)]
                    if len(valid_c) < 10:
                        trend_ok[si] = False
                        continue
                    ma20 = np.mean(valid_c)
                    if c_prev <= ma20:
                        trend_ok[si] = False

            # Combine filters: composite >= threshold AND trend ok
            eligible = mask & (composite >= pct_threshold) & trend_ok
            eligible_indices = np.where(eligible)[0]

            if len(eligible_indices) < top_n:
                # Not enough candidates -- relax or skip
                daily_nav.append(cash)
                last_rebalance = di
                continue

            # Rank eligible by composite, pick top_n
            eligible_scores = composite[eligible_indices]
            top_order = np.argsort(-eligible_scores)[:top_n]
            top_indices = set(eligible_indices[top_order])
            current_indices = set(h['si'] for h in holdings)

            # Sell positions not in top_indices
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

            # Buy new positions
            current_indices = set(h['si'] for h in holdings)
            to_buy = top_indices - current_indices
            n_to_buy = len(to_buy)

            if n_to_buy > 0 and cash > 10000:
                # Compute median ATR for volatility scaling
                median_atr = 0.0
                if vol_scale:
                    atrs = []
                    for si in to_buy:
                        atr_s = 0.0
                        atr_c = 0
                        for dd in range(max(di - 14, 1), di):
                            if not np.isnan(H[si, dd]) and not np.isnan(L[si, dd]):
                                tr = H[si, dd] - L[si, dd]
                                if not np.isnan(C[si, dd - 1]):
                                    tr = max(tr, abs(H[si, dd] - C[si, dd - 1]),
                                             abs(L[si, dd] - C[si, dd - 1]))
                                atr_s += tr
                                atr_c += 1
                        if atr_c > 0:
                            atrs.append(atr_s / atr_c)
                    if atrs:
                        median_atr = np.median(atrs)

                base_alloc = cash / n_to_buy
                for si in to_buy:
                    p = O[si, di]
                    if np.isnan(p) or p <= 0:
                        p = C[si, di]
                    if np.isnan(p) or p <= 0:
                        continue

                    # Volatility scaling
                    alloc = base_alloc
                    if vol_scale and median_atr > 0:
                        stock_atr = 0.0
                        atr_c = 0
                        for dd in range(max(di - 14, 1), di):
                            if not np.isnan(H[si, dd]) and not np.isnan(L[si, dd]):
                                tr = H[si, dd] - L[si, dd]
                                if not np.isnan(C[si, dd - 1]):
                                    tr = max(tr, abs(H[si, dd] - C[si, dd - 1]),
                                             abs(L[si, dd] - C[si, dd - 1]))
                                stock_atr += tr
                                atr_c += 1
                        if atr_c > 0:
                            stock_atr /= atr_c
                        if stock_atr > 0:
                            scale_factor = median_atr / stock_atr
                            scale_factor = max(0.5, min(2.0, scale_factor))
                            alloc = base_alloc * scale_factor

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

        # Track daily NAV for proper drawdown calculation
        nav = cash
        for pos in holdings:
            cp = C[pos['si'], di]
            if np.isnan(cp) or cp <= 0:
                cp = pos['entry']  # Fallback: use entry price for suspended stocks
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

    # Drawdown from daily NAV (corrected: uses actual portfolio value)
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
    print("  Alpha V72 -- Ultimate Combined Strategy", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()
    factors = compute_all_factors(NS, ND, C, O, H, L, V)

    # Compute market regime
    print("[Regime] Computing market regime...", flush=True)
    mkt_ret20, mkt_ret5, breadth = compute_market_regime(NS, ND, C, O, H, L, V, syms)
    print("  Market regime done", flush=True)

    # Quick sanity test with V62 weights
    v62_w = {
        'R_REL_STR_S3':       0.80,
        'R_TENSION':          1.00,
        'R_SMA_DEV':          0.30,
        'R_OIS':              0.15,
        'R_VOL_MOM':          0.10,
        'R_HAR_RV_RATIO_INV': 0.03,
    }

    r = backtest_v72(v62_w, factors, NS, ND, dates, C, O, H, L, V,
                     top_n=1, rebalance_days=10, atr_stop_mult=2.0,
                     min_composite_pct=85, trend_filter=True,
                     be_threshold=0.05, bear_defense=True, min_breadth=45,
                     vol_scale=True, max_loss_pct=15,
                     breadth=breadth)

    if r:
        print(f"\n  V72 (V62 weights, defaults): +{r['ann']}% Ann, {r['max_dd']}% DD")
        print(f"  Final: {r['final']:,.0f} | {r['n']} trades | WR={r['wr']}%")
        print(f"  Avg Win: {r['avg_w']}% | Avg Loss: {r['avg_l']}% | Edge: {r['edge']}%")
    else:
        print("  No trades generated!")
