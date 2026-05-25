"""
Alpha V66 — Trend Following + Pullback Entry
=============================================
INSIGHT: Instead of buying the strongest stock at any price (which often
means chasing the top), wait for a pullback WITHIN an uptrend. This
dramatically improves entry price and win rate.

Strategy:
  1. IDENTIFY UPTRENDS: MA5 > MA20 > MA60 aligned, 20-day return > 5%
  2. WAIT FOR PULLBACK: Close < MA5 (dipped below short MA),
     but still above MA20 (trend intact), low volume (no panic)
  3. EXIT: Trailing stop at MA20 level, time stop at 30 days,
     partial take-profit at +10%
  4. POSITION SIZING: Max 3 stocks, equal weight

All MAs computed from di-1 data (no look-ahead).
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.data_loader import list_available_symbols, load_stock_data
from alpha_v2 import load_all_data, MIN_TRAIN, COMMISSION, STAMP_DUTY, CASH0


# ============================================================
# HELPER: Simple rolling mean (no library dependencies)
# ============================================================
def rolling_mean(arr_2d, window):
    """Compute rolling mean for a 2D array [stock, date].

    rolling_mean[:, di] uses data from arr[:, di-window] to arr[:, di-1].
    This means the MA at day di is computed from the PREVIOUS `window`
    closes, NOT including today — no look-ahead.
    """
    NS, ND = arr_2d.shape
    result = np.full((NS, ND), np.nan)
    for si in range(NS):
        row = arr_2d[si]
        # Use pandas rolling for speed but compute manually for correctness
        s = pd.Series(row)
        # Shift by 1 so that the window ending at di uses data up to di-1
        rolled = s.shift(1).rolling(window=window, min_periods=window).mean()
        result[si] = rolled.values
    return result


def rolling_mean_vol(arr_2d, window):
    """Same as rolling_mean but for volume arrays."""
    return rolling_mean(arr_2d, window)


# ============================================================
# BACKTEST: Trend Pullback Strategy
# ============================================================
def backtest_v66(NS, ND, dates, C, O, H, L, V,
                 ma5, ma20, ma60, ma20_vol,
                 max_hold=30, tp_pct=10.0, max_positions=3,
                 min_uptrend_ret=0.05, vol_pullback_ratio=1.0):
    """Backtest the trend-following pullback strategy.

    Parameters
    ----------
    max_hold : int
        Maximum holding days (time stop).
    tp_pct : float
        Take-profit threshold (%). Sell half when unrealized gain exceeds this.
    max_positions : int
        Maximum concurrent positions.
    min_uptrend_ret : float
        Minimum 20-day return to qualify as uptrend (e.g. 0.05 = 5%).
    vol_pullback_ratio : float
        Pullback day volume must be below this * avg_vol to qualify.
    """
    cash = float(CASH0)
    holdings = []       # list of dicts: si, shares, entry, ed, hw, half_sold
    trades = []
    year_stats = {}
    daily_nav = []

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # ---- EXIT CHECKS ----
        for pos in list(holdings):
            si = pos['si']
            cp = C[si, di]
            op = O[si, di]
            lp = L[si, di]

            if np.isnan(cp):
                continue

            days_held = (dates[di] - pos['ed']).days

            # 1. Trailing stop at MA20: sell if close drops below MA20
            stop_level = ma20[si, di]
            if not np.isnan(stop_level) and cp < stop_level:
                sp = op if (not np.isnan(op) and op < stop_level) else stop_level
                if np.isnan(sp) or sp <= 0:
                    sp = cp
                pnl = (sp - pos['entry']) / pos['entry'] * 100
                cash += pos['shares'] * sp * (1 - COMMISSION - STAMP_DUTY)
                trades.append({'pnl': pnl, 'days': days_held,
                               'di': di, 'reason': 'ma20_stop', 'year': year})
                holdings.remove(pos)
                continue

            # 2. Take-profit: sell half at +tp_pct%, raise stop to entry
            unrealized = (cp - pos['entry']) / pos['entry'] * 100
            if unrealized >= tp_pct and not pos['half_sold']:
                # Sell half at close price (realistic: we'd use a limit order)
                half_shares = pos['shares'] // 2
                if half_shares > 0:
                    sp = cp
                    cash += half_shares * sp * (1 - COMMISSION - STAMP_DUTY)
                    pos['shares'] -= half_shares
                    pos['half_sold'] = True
                    # Record the partial trade
                    pnl_partial = (sp - pos['entry']) / pos['entry'] * 100
                    trades.append({'pnl': pnl_partial, 'days': days_held,
                                   'di': di, 'reason': 'take_profit_half', 'year': year})
                # After half sold, raise stop to entry price (breakeven stop)
                pos['stop'] = pos['entry']
                continue

            # 3. Breakeven stop (after half sold): sell if close < entry
            if pos.get('half_sold', False) and cp < pos['entry']:
                sp = op if (not np.isnan(op) and op < pos['entry']) else pos['entry']
                if np.isnan(sp) or sp <= 0:
                    sp = cp
                pnl = (sp - pos['entry']) / pos['entry'] * 100
                cash += pos['shares'] * sp * (1 - COMMISSION - STAMP_DUTY)
                trades.append({'pnl': pnl, 'days': days_held,
                               'di': di, 'reason': 'breakeven_stop', 'year': year})
                holdings.remove(pos)
                continue

            # 4. Time stop: hold max `max_hold` days
            if days_held >= max_hold:
                sp = op if not np.isnan(op) else cp
                if np.isnan(sp) or sp <= 0:
                    sp = cp
                pnl = (sp - pos['entry']) / pos['entry'] * 100
                cash += pos['shares'] * sp * (1 - COMMISSION - STAMP_DUTY)
                trades.append({'pnl': pnl, 'days': days_held,
                               'di': di, 'reason': 'time_stop', 'year': year})
                holdings.remove(pos)
                continue

        # ---- ENTRY SCANNER ----
        n_current = len(holdings)
        if n_current < max_positions and cash > 10000:
            candidates = []

            for si in range(NS):
                # Skip if already holding this stock
                if any(h['si'] == si for h in holdings):
                    continue

                cp = C[si, di - 1]  # Yesterday's close (signal data)
                if np.isnan(cp) or cp <= 0:
                    continue

                m5 = ma5[si, di]    # MA5 at di uses data up to di-1
                m20 = ma20[si, di]
                m60 = ma60[si, di]

                if np.isnan(m5) or np.isnan(m20) or np.isnan(m60):
                    continue

                # ---- UPTREND CHECK ----
                # MA5 > MA20 > MA60 (aligned uptrend)
                if not (m5 > m20 > m60):
                    continue

                # 20-day return > min_uptrend_ret
                c20_ago = C[si, di - 1 - 20]
                if np.isnan(c20_ago) or c20_ago <= 0:
                    continue
                ret20 = (cp - c20_ago) / c20_ago
                if ret20 < min_uptrend_ret:
                    continue

                # ---- PULLBACK CHECK ----
                # Close below MA5 (pulled back from recent high)
                if cp >= m5:
                    continue

                # But still above MA20 (trend still intact)
                if cp < m20:
                    continue

                # Volume on pullback day is below average (no panic selling)
                vol_yesterday = V[si, di - 1]
                avg_vol = ma20_vol[si, di]
                if np.isnan(vol_yesterday) or np.isnan(avg_vol) or avg_vol <= 0:
                    continue
                if vol_yesterday > vol_pullback_ratio * avg_vol:
                    continue

                # Score: how deep is the pullback relative to MA5?
                # Deeper pullback (closer to MA20) = better entry
                # Range: 0 (at MA5) to 1 (at MA20)
                if m5 == m20:
                    continue
                pullback_depth = (m5 - cp) / (m5 - m20)
                pullback_depth = max(0, min(1, pullback_depth))

                # Strength: how strong is the uptrend (20-day return)
                score = pullback_depth * 0.4 + ret20 * 5.0  # weighted score

                candidates.append({
                    'si': si, 'score': score, 'ret20': ret20,
                    'pullback_depth': pullback_depth
                })

            # Sort by score, take best candidates
            candidates.sort(key=lambda x: -x['score'])
            n_slots = max_positions - n_current
            for cand in candidates[:n_slots]:
                si = cand['si']
                p = O[si, di]  # Buy at today's open
                if np.isnan(p) or p <= 0:
                    p = C[si, di]
                if np.isnan(p) or p <= 0:
                    continue

                alloc = cash / (n_slots - len([c for c in candidates[:n_slots]
                                                if any(h['si'] == c['si'] for h in holdings)]))
                alloc = cash / max(n_slots, 1)
                shares = int(alloc / (1 + COMMISSION) / p)
                if shares <= 0:
                    continue
                cost = shares * p * (1 + COMMISSION)
                if cost > cash:
                    shares = int(cash / (1 + COMMISSION) / p)
                    cost = shares * p * (1 + COMMISSION)
                if shares <= 0:
                    continue

                cash -= cost
                holdings.append({
                    'si': si, 'shares': shares, 'entry': p,
                    'ed': dates[di], 'hw': p, 'half_sold': False,
                    'stop': ma20[si, di] if not np.isnan(ma20[si, di]) else p * 0.95
                })

        # ---- DAILY NAV ----
        nav = cash
        for pos in holdings:
            cp = C[pos['si'], di]
            if np.isnan(cp) or cp <= 0:
                cp = pos['entry']
            nav += pos['shares'] * cp
        daily_nav.append(nav)

    # ---- CLOSE REMAINING ----
    for pos in holdings:
        p = C[pos['si'], ND - 1]
        if not np.isnan(p) and p > 0:
            pnl = (p - pos['entry']) / pos['entry'] * 100
            cash += pos['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
            trades.append({'pnl': pnl, 'days': 999, 'di': ND - 1,
                           'reason': 'end', 'year': dates[ND - 1].year})

    if not trades:
        return None

    # ---- STATS ----
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

    # Exit reason breakdown
    reason_counts = {}
    for t in trades:
        r = t['reason']
        if r not in reason_counts:
            reason_counts[r] = {'n': 0, 'wins': 0, 'pnl': 0}
        reason_counts[r]['n'] += 1
        if t['pnl'] > 0:
            reason_counts[r]['wins'] += 1
        reason_counts[r]['pnl'] += t['pnl']

    return {
        'ann': round(ann, 1), 'n': len(trades), 'wr': round(wr, 1),
        'avg_w': round(avg_w, 1), 'avg_l': round(avg_l, 1),
        'edge': round((nw / max(len(trades), 1)) * avg_w -
                       (1 - nw / max(len(trades), 1)) * avg_l, 2),
        'max_dd': round(max_dd, 1), 'tpy': round(len(trades) / yr, 1),
        'final': round(cash, 0), 'year_stats': year_stats,
        'reason_counts': reason_counts,
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V66 — Trend Following + Pullback Entry", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()
    print(f"  Data: {NS} stocks, {ND} trading days", flush=True)

    # ---- Precompute Moving Averages ----
    print("[MA] Computing moving averages...", flush=True)
    t0 = time.time()
    ma5 = rolling_mean(C, 5)
    ma20 = rolling_mean(C, 20)
    ma60 = rolling_mean(C, 60)
    ma20_vol = rolling_mean_vol(V, 20)
    print(f"  MAs done ({time.time()-t0:.1f}s)", flush=True)

    # ---- PARAMETER SCAN ----
    print("\n[SCAN] Running parameter scan...", flush=True)
    results = []

    param_grid = {
        'max_hold': [20, 25, 30],
        'tp_pct': [8.0, 10.0, 15.0],
        'min_uptrend_ret': [0.03, 0.05, 0.08],
        'vol_pullback_ratio': [0.8, 1.0, 1.2],
    }

    for max_hold in param_grid['max_hold']:
        for tp_pct in param_grid['tp_pct']:
            for min_ret in param_grid['min_uptrend_ret']:
                for vol_ratio in param_grid['vol_pullback_ratio']:
                    r = backtest_v66(
                        NS, ND, dates, C, O, H, L, V,
                        ma5, ma20, ma60, ma20_vol,
                        max_hold=max_hold, tp_pct=tp_pct,
                        max_positions=3, min_uptrend_ret=min_ret,
                        vol_pullback_ratio=vol_ratio
                    )
                    if r:
                        r['max_hold'] = max_hold
                        r['tp_pct'] = tp_pct
                        r['min_ret'] = min_ret
                        r['vol_ratio'] = vol_ratio
                        results.append(r)

    results.sort(key=lambda x: -x['ann'])

    # ---- PRINT RESULTS ----
    print(f"\n{'='*120}", flush=True)
    print(f"  TOP 30 RESULTS (sorted by annual return)", flush=True)
    print(f"  {'Hold':>4s} {'TP%':>4s} {'MinR':>5s} {'VolR':>4s} | "
          f"{'Ann':>7s} {'N':>5s} {'TPY':>4s} {'WR':>5s} {'Edge':>6s} {'DD':>5s} {'Final':>10s}",
          flush=True)
    print(f"  {'-'*100}", flush=True)
    for r in results[:30]:
        print(f"  {r['max_hold']:4d} {r['tp_pct']:4.0f} {r['min_ret']:5.2f} {r['vol_ratio']:4.1f} | "
              f"{r['ann']:+7.1f}% {r['n']:5d} {r['tpy']:4.0f} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}% {r['final']:10.0f}",
              flush=True)

    # ---- BEST RESULT DETAILS ----
    if results:
        best = results[0]
        print(f"\n{'='*70}", flush=True)
        print(f"  BEST: Hold={best['max_hold']}d, TP={best['tp_pct']}%, "
              f"MinRet={best['min_ret']:.0%}, VolR={best['vol_ratio']:.1f}", flush=True)
        print(f"  Annual: {best['ann']:+.1f}%  WR: {best['wr']:.1f}%  "
              f"DD: {best['max_dd']:.1f}%  Trades: {best['n']}", flush=True)

        # Exit reason breakdown
        print(f"\n  Exit reason breakdown:", flush=True)
        for reason, stats in sorted(best['reason_counts'].items(),
                                     key=lambda x: -x[1]['n']):
            wr_r = stats['wins'] / max(stats['n'], 1) * 100
            print(f"    {reason:<20s}: {stats['n']:4d} trades, "
                  f"WR={wr_r:.0f}%, total_pnl={stats['pnl']:+.0f}%", flush=True)

        # Year-by-year
        print(f"\n  Year-by-year performance:", flush=True)
        for y in sorted(best.get('year_stats', {}).keys()):
            s = best['year_stats'][y]
            wr_y = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr_y:.0f}%, "
                  f"pnl={s['total_pnl']:+.0f}%", flush=True)

    # ---- COMPARE: WR > 55% FILTER ----
    high_wr = [r for r in results if r['wr'] > 55]
    if high_wr:
        high_wr.sort(key=lambda x: -x['ann'])
        print(f"\n  Configurations with WR > 55%: {len(high_wr)}", flush=True)
        for r in high_wr[:10]:
            print(f"    Hold={r['max_hold']:2d}d TP={r['tp_pct']:4.0f}% "
                  f"MinR={r['min_ret']:.2f} VolR={r['vol_ratio']:.1f} | "
                  f"Ann={r['ann']:+.1f}% WR={r['wr']:.1f}% DD={r['max_dd']:.1f}%",
                  flush=True)
    else:
        print(f"\n  No configurations achieved WR > 55%", flush=True)
        # Show best WR
        by_wr = sorted(results, key=lambda x: -x['wr'])
        if by_wr:
            print(f"  Best WR: {by_wr[0]['wr']:.1f}% "
                  f"(Ann={by_wr[0]['ann']:+.1f}%, DD={by_wr[0]['max_dd']:.1f}%)",
                  flush=True)

    print(f"\n{'='*70}", flush=True)
