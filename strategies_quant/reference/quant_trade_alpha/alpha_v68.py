"""
Alpha V68 — Regime-Adaptive Stop Management
=============================================
Problem: V62 gets +466% but only 36.5% WR, 83.4% max DD.
  58% of trades stopped out in 1-3 days — stops are too tight.

Key insight: stops should be WIDER in volatile markets,
TIGHTER in calm markets. Dynamic lookback improves Sharpe by 66%.

Approach:
  1. Compute market-wide volatility regime (HIGH_VOL / NORMAL / LOW_VOL)
  2. Adapt ATR stop multiplier and rebalance interval per regime:
     - HIGH_VOL: atr_mult=4.0 (wider stops), rebalance=15 (longer holds)
     - LOW_VOL:  atr_mult=1.5 (tighter stops), rebalance=5 (faster rotation)
     - NORMAL:   atr_mult=2.5, rebalance=10
  3. MOM_REVERSAL factor: "buy the dip in uptrends"
     - 20-day momentum > 0 AND 5-day return < 0 => bullish pullback
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7 import COMMISSION, STAMP_DUTY, CASH0


# =====================================================================
# Market volatility regime computation
# =====================================================================
def compute_vol_regime(NS, ND, C, V):
    """Compute per-day market volatility regime.

    Returns:
        regime: array of str, length ND
            'HIGH_VOL' if market realized vol > 75th pctile
            'LOW_VOL'  if market realized vol < 25th pctile
            'NORMAL'   otherwise
        market_vol: array of float, length ND — daily market-wide 20d realized vol
    """
    PCTILE_LOOKBACK = 60  # rolling window for percentile thresholds
    VOL_PERIOD = 20       # realized vol lookback

    # Step 1: Compute cross-sectional average daily log-return (market return)
    market_ret = np.full(ND, np.nan)
    for di in range(1, ND):
        c0 = C[:, di - 1]
        c1 = C[:, di]
        valid = ~np.isnan(c0) & ~np.isnan(c1) & (c0 > 0)
        if valid.sum() > 50:
            log_ret = np.log(c1[valid] / c0[valid])
            market_ret[di] = np.nanmean(log_ret)

    # Step 2: Compute 20-day realized volatility of market return
    market_vol = np.full(ND, np.nan)
    for di in range(VOL_PERIOD + 1, ND):
        window = market_ret[di - VOL_PERIOD:di]
        valid = window[~np.isnan(window)]
        if len(valid) >= VOL_PERIOD - 5:
            market_vol[di] = np.std(valid)

    # Step 3: Assign regime based on rolling percentiles
    regime = np.array(['UNKNOWN'] * ND, dtype='U10')
    for di in range(VOL_PERIOD + PCTILE_LOOKBACK + 1, ND):
        vol_window = market_vol[di - PCTILE_LOOKBACK:di]
        valid = vol_window[~np.isnan(vol_window)]
        if len(valid) < 20:
            continue
        p25 = np.percentile(valid, 25)
        p75 = np.percentile(valid, 75)
        cur_vol = market_vol[di]
        if np.isnan(cur_vol):
            continue
        if cur_vol > p75:
            regime[di] = 'HIGH_VOL'
        elif cur_vol < p25:
            regime[di] = 'LOW_VOL'
        else:
            regime[di] = 'NORMAL'

    return regime, market_vol


# =====================================================================
# MOM_REVERSAL factor: "buy the dip in uptrends"
# =====================================================================
def compute_mom_reversal(NS, ND, C):
    """Compute MOM_REVERSAL factor: bullish pullback signal.

    For each stock on each day (using data up to di-1):
      1. Compute 20-day momentum: (C[di-1] - C[di-1-20]) / C[di-1-20]
      2. Compute 5-day return: (C[di-1] - C[di-1-5]) / C[di-1-5]
      3. If momentum > 0 (uptrend) AND 5d return < 0 (pullback):
           score = abs(5d_return) * momentum_strength
         Else: score = 0

    Returns: raw array (NS, ND), unranked. Caller should rank-normalize.
    """
    MOM_REVERSAL = np.zeros((NS, ND))

    for si in range(NS):
        for di in range(21, ND):
            c_now = C[si, di - 1]
            c_5 = C[si, di - 1 - 5]
            c_20 = C[si, di - 1 - 20]

            if np.isnan(c_now) or np.isnan(c_5) or np.isnan(c_20):
                MOM_REVERSAL[si, di] = 0
                continue
            if c_20 <= 0 or c_5 <= 0:
                MOM_REVERSAL[si, di] = 0
                continue

            mom20 = (c_now - c_20) / c_20
            ret5 = (c_now - c_5) / c_5

            if mom20 > 0 and ret5 < 0:
                # Uptrend with pullback — signal strength
                # Scale by momentum strength so stronger uptrends score higher
                MOM_REVERSAL[si, di] = abs(ret5) * (1 + mom20)
            else:
                MOM_REVERSAL[si, di] = 0

    # Set early days to NaN
    MOM_REVERSAL[:, :21] = np.nan

    return MOM_REVERSAL


# =====================================================================
# Rank normalization utility
# =====================================================================
def rank_pct(arr, start=60):
    """Rank normalize to 0-100 across stocks for each day."""
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


# =====================================================================
# Regime-adaptive backtest (based on backtest_v7c)
# =====================================================================
def backtest_v68(factor_weights, factors, NS, ND, dates, C, O, H, L, V,
                 regime=None, top_n=3, rebalance_days=10,
                 market_ret20=None, breadth=None,
                 regime_filter=False, min_breadth=40, min_mkt_ret=-0.05,
                 position_scale=True):
    """Backtest with regime-adaptive ATR stop loss.

    Identical to backtest_v7c except:
    - ATR stop multiplier varies by market volatility regime
    - Rebalance interval varies by regime (overrides rebalance_days)

    Regime parameters:
      HIGH_VOL: atr_mult = 4.0, rebalance = 15
      LOW_VOL:  atr_mult = 1.5, rebalance = 5
      NORMAL:   atr_mult = 2.5, rebalance = 10

    All data up to di-1, no look-ahead. ATR stop uses correct execution:
    1. Compute stop from PREVIOUS hw (data up to di-1)
    2. Check if L[si,di] <= stop (realistic: intraday low hit the stop)
    3. If triggered: sell at stop price
    4. If NOT triggered: update hw = max(hw, H[si,di])
    """
    # Regime-dependent parameters
    REGIME_ATR = {'HIGH_VOL': 4.0, 'NORMAL': 2.5, 'LOW_VOL': 1.5}
    REGIME_REBAL = {'HIGH_VOL': 15, 'NORMAL': 10, 'LOW_VOL': 5}

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

        # Determine today's ATR multiplier and rebalance interval from regime
        if regime is not None:
            cur_regime = regime[di]
            atr_stop_mult = REGIME_ATR.get(cur_regime, 2.5)
            effective_rebalance = REGIME_REBAL.get(cur_regime, rebalance_days)
        else:
            atr_stop_mult = 2.5
            effective_rebalance = rebalance_days

        # ---- ATR stop loss check (FIXED: no look-ahead) ----
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

        # ---- Rebalance (with effective_rebalance from regime) ----
        if di - last_rebalance >= effective_rebalance:
            # Regime filter
            skip = False
            scale = 1.0
            if regime_filter and market_ret20 is not None and breadth is not None:
                mr = market_ret20[di]
                br = breadth[di]
                if not np.isnan(br) and br < min_breadth:
                    skip = True
                if not np.isnan(mr) and mr < min_mkt_ret:
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


# =====================================================================
# Main: standalone test
# =====================================================================
if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V68 — Regime-Adaptive Stop Management", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Compute regime
    print("[Regime] Computing volatility regime...", flush=True)
    regime, market_vol = compute_vol_regime(NS, ND, C, V)

    # Count regime distribution
    for rname in ['HIGH_VOL', 'NORMAL', 'LOW_VOL', 'UNKNOWN']:
        cnt = np.sum(regime == rname)
        print(f"  {rname}: {cnt} days ({cnt / ND * 100:.1f}%)", flush=True)
