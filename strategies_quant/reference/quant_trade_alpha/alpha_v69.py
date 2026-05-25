"""
Alpha V69 — Entry Quality Filters + Break-Even Stop
====================================================
Hypothesis: Wider stops (ATR>=1.5) improve WR but lower returns.
By adding entry quality filters, we skip low-conviction entries,
pushing WR above 50% while maintaining high returns.

New features on top of V7c backtest engine:
  1. min_composite_pct: Only enter when composite score is above Nth percentile
  2. trend_filter: Only buy when price > MA20 (no counter-trend entries)
  3. be_threshold: Break-even stop — raise stop to entry after gain > threshold%
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.data_loader import list_available_symbols, load_stock_data
from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7 import compute_all_factors, COMMISSION, STAMP_DUTY, CASH0


def backtest_v69(factor_weights, factors, NS, ND, dates, C, O, H, L, V,
                 top_n=1, rebalance_days=10, atr_stop_mult=2.0,
                 min_composite_pct=80, trend_filter=True, ma_period=20,
                 be_threshold=0.05, market_ret20=None, breadth=None,
                 regime_filter=False, min_breadth=50):
    """Backtest with entry quality filters + break-even stop.

    Extends V7c with:
      - min_composite_pct: Skip entries where composite < Nth percentile
      - trend_filter: Only buy when C[si, di-1] > MA20[si, di-1]
      - be_threshold: After gain > threshold%, raise stop to entry price
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

        # ----- ATR stop loss check (V7c logic, plus break-even stop) -----
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
                    # Stop = previous hw - ATR*mult
                    atr_stop = pos['hw'] - atr_stop_mult * atr

                    # --- NEW: Break-even stop ---
                    # If position is profitable beyond threshold, raise stop to entry
                    current_price = C[si, di - 1] if di > 0 and not np.isnan(C[si, di - 1]) else pos['entry']
                    if pos['entry'] > 0 and current_price > pos['entry']:
                        gain_pct = (current_price - pos['entry']) / pos['entry']
                        if gain_pct > be_threshold:
                            # Raise stop to max(entry_price, atr_stop)
                            atr_stop = max(atr_stop, pos['entry'])

                    # Check if today's LOW hit the stop
                    today_low = L[si, di]
                    today_open = O[si, di]

                    if not np.isnan(today_low) and today_low <= atr_stop:
                        # Stop triggered
                        if not np.isnan(today_open) and today_open < atr_stop:
                            sp = today_open  # Gap down — sell at open
                        else:
                            sp = atr_stop  # Normal stop execution
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

        # ----- Rebalance -----
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

            # ----- Composite score -----
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

            # --- NEW: Minimum composite score filter ---
            # Only consider stocks with composite >= Nth percentile
            valid_composite = composite[mask]
            if len(valid_composite) < top_n * 2:
                continue
            pct_cutoff = np.percentile(valid_composite, min_composite_pct)
            high_confidence = composite >= pct_cutoff

            # Get top_n from high-confidence pool
            # Sort all stocks by composite, but only pick those above cutoff
            sorted_indices = np.argsort(-composite)
            top_candidates = []
            for idx in sorted_indices:
                if len(top_candidates) >= top_n:
                    break
                if composite[idx] < -999:
                    break
                if high_confidence[idx]:
                    top_candidates.append(idx)

            # If not enough high-confidence candidates, skip this rebalance
            if len(top_candidates) < top_n:
                last_rebalance = di
                continue

            top_indices = set(top_candidates)
            current_indices = set(h['si'] for h in holdings)

            # --- NEW: Trend confirmation filter ---
            if trend_filter:
                filtered = set()
                for si in top_indices:
                    # Compute MA20 using data up to di-1
                    c_slice = C[si, max(0, di - 1 - ma_period):di]
                    valid_c = c_slice[~np.isnan(c_slice)]
                    if len(valid_c) >= ma_period // 2:
                        ma_val = np.mean(valid_c)
                        prev_close = C[si, di - 1]
                        if not np.isnan(prev_close) and prev_close > ma_val:
                            filtered.add(si)
                top_indices = filtered

                if len(top_indices) == 0:
                    last_rebalance = di
                    continue

            # Sell: current holdings not in top_indices
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

            # Buy: top_indices not currently held
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

    # Close remaining positions
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
