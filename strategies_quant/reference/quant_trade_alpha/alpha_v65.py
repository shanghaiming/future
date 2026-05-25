"""
Alpha V65 — Defensive Quality Factor Strategy (High Win Rate)
==============================================================
Problem: current strategies buy the TOP ranked stock, which is often
volatile and gets stopped out. V65 takes a defensive approach:

  1. QUALITY FILTER: Only consider stocks passing ALL criteria:
     - R_TENSION > 50       (not in downtrend)
     - R_VOLATILITY_PCT < 50 (not excessively volatile)
     - R_OIS > 30            (positive overnight sentiment)
     - 20-day return > 0     (positive momentum)

  2. COMPOSITE SCORING: Among quality stocks, rank by:
     - R_REL_STR_S3  (relative strength vs market)
     - R_VWCM        (volume-weighted close momentum)
     - R_OIS         (overnight-intraday spread)

  3. WIDE STOPS: ATR * 3.0 (give stocks room to breathe)
  4. LONG HOLD: rebalance every 15-20 days
  5. BREAK-EVEN PROTECTION: After +8% gain, raise stop to entry price

Target: WR > 55%, DD < 60%, annual return 100-150% is fine.
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7 import COMMISSION, STAMP_DUTY, CASH0


# =================================================================
# Factor computation — reuse existing pipelines
# =================================================================
def compute_v65_factors(NS, ND, C, O, H, L, V):
    """Compute all factors needed for V65.

    Sources:
      - V7 base factors  -> R_TENSION, R_VOLATILITY_PCT
      - V48 factors      -> R_OIS, R_VWCM (via V49)
      - V62 inline       -> R_REL_STR_S3
    """
    from alpha_v7 import compute_all_factors
    from alpha_v7b import compute_interaction_factors
    from alpha_v7d import compute_extra_factors
    from alpha_v7e import compute_v7e_factors
    from alpha_v7f import compute_advanced_interactions
    from alpha_v48 import compute_v48_factors
    from alpha_v49 import compute_v49_factors

    t0 = time.time()

    # V7 base + interactions (provides R_TENSION, R_VOLATILITY_PCT)
    base = compute_all_factors(NS, ND, C, O, H, L, V)
    print(f"  V7 base done ({time.time()-t0:.0f}s)", flush=True)

    inter = compute_interaction_factors(base, NS, ND, C, O, H, L, V)
    extra = compute_extra_factors(NS, ND, C, O, H, L, V)
    v7e = compute_v7e_factors(NS, ND, C, O, H, L, V)
    adv = compute_advanced_interactions({**base, **inter, **extra, **v7e}, NS, ND)
    v7_all = {**base, **inter, **extra, **v7e, **adv}
    print(f"  V7 full done ({time.time()-t0:.0f}s)", flush=True)

    # V48 factors (provides R_OIS)
    v48 = compute_v48_factors(NS, ND, C, O, H, L, V)
    print(f"  V48 done ({time.time()-t0:.0f}s)", flush=True)

    # V49 factors (provides R_VWCM)
    v49 = compute_v49_factors(NS, ND, C, O, H, L, V)
    print(f"  V49 done ({time.time()-t0:.0f}s)", flush=True)

    # R_REL_STR_S3 — computed inline (same as V62)
    ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(C[:, di]) & ~np.isnan(C[:, di - 1]) & (C[:, di - 1] > 0)
        ret[m, di] = (C[m, di] - C[m, di - 1]) / C[m, di - 1]

    mkt_ret = np.full(ND, np.nan)
    for di in range(1, ND):
        valid = ~np.isnan(ret[:, di])
        if valid.sum() > 50:
            mkt_ret[di] = np.mean(ret[valid, di])

    rel_ret = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        m = ~np.isnan(ret[:, di]) & ~np.isnan(mkt_ret[di])
        rel_ret[m, di] = ret[m, di] - mkt_ret[di]

    # EMA with span=3
    alpha_ema = 2.0 / 4
    ema_rel = np.full_like(rel_ret, np.nan)
    for di in range(2, ND):
        mp = ~np.isnan(ema_rel[:, di - 1])
        mc = ~np.isnan(rel_ret[:, di - 1])
        both = mp & mc
        ema_rel[both, di] = alpha_ema * rel_ret[both, di - 1] + (1 - alpha_ema) * ema_rel[both, di - 1]
        new = mc & ~mp
        ema_rel[new, di] = rel_ret[new, di - 1]

    # Rank normalize
    r_rel_s3 = np.full_like(ema_rel, np.nan)
    for di in range(ND):
        vals = ema_rel[:, di]
        valid = ~np.isnan(vals)
        n = valid.sum()
        if n < 50:
            continue
        order = np.argsort(vals[valid])
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1)
        r_rel_s3[valid, di] = ranks / n * 100

    # 20-day return for momentum filter
    mom_20 = np.full((NS, ND), np.nan)
    for di in range(21, ND):
        mask = ~np.isnan(C[:, di - 1]) & ~np.isnan(C[:, di - 21]) & (C[:, di - 21] > 0)
        mom_20[mask, di] = (C[mask, di - 1] - C[mask, di - 21]) / C[mask, di - 21]

    all_factors = {**v7_all, **v48, **v49, 'R_REL_STR_S3': r_rel_s3, 'MOM_20': mom_20}

    # Verify all needed factors exist
    needed = ['R_TENSION', 'R_VOLATILITY_PCT', 'R_OIS', 'R_REL_STR_S3', 'R_VWCM', 'MOM_20']
    for fname in needed:
        if fname not in all_factors:
            print(f"  WARNING: {fname} not found in computed factors!")
        else:
            valid_count = sum(1 for si in range(NS) if not np.isnan(all_factors[fname][si, ND - 1]))
            print(f"  {fname}: {valid_count} valid stocks on last day", flush=True)

    print(f"  All V65 factors ready ({time.time()-t0:.0f}s)", flush=True)
    return all_factors


# =================================================================
# Backtest engine — defensive quality with wide stops
# =================================================================
def backtest_v65(factors, NS, ND, dates, C, O, H, L, V,
                 score_weights=None,
                 quality_filters=None,
                 top_n=3,
                 rebalance_days=15,
                 atr_stop_mult=3.0,
                 breakeven_trigger=8.0,
                 max_hold_days=60):
    """Defensive quality backtest with:
    - Quality filter (pre-screen stocks)
    - Composite scoring (rank among quality stocks)
    - Wide ATR stops with break-even protection
    - Long hold periods

    No look-ahead: all factors use data up to di-1, trade at di open.
    """
    if score_weights is None:
        score_weights = {'R_REL_STR_S3': 0.40, 'R_VWCM': 0.35, 'R_OIS': 0.25}
    if quality_filters is None:
        quality_filters = {
            'R_TENSION': ('>', 50),
            'R_VOLATILITY_PCT': ('<', 50),
            'R_OIS': ('>', 30),
        }

    factor_names = list(score_weights.keys())
    weights = np.array([score_weights[f] for f in factor_names])

    cash = float(CASH0)
    holdings = []
    trades = []
    last_rebalance = -999
    year_stats = {}
    daily_nav = []

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # ---- Stop loss check ----
        for pos in list(holdings):
            si = pos['si']
            stopped_out = False
            days_held = (dates[di] - pos['ed']).days

            # Compute ATR(14) from data up to di-1
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

            if atr > 0 and atr_stop_mult > 0:
                # Break-even protection: if unrealized gain > trigger, stop = entry
                current_price = C[si, di - 1] if not np.isnan(C[si, di - 1]) else pos['entry']
                unrealized = (current_price - pos['entry']) / pos['entry'] * 100

                if unrealized >= breakeven_trigger and pos.get('be_active', False) is False:
                    pos['be_active'] = True

                if pos.get('be_active', False):
                    # Stop is the higher of break-even or trailing ATR stop
                    atr_stop = pos['hw'] - atr_stop_mult * atr
                    stop = max(pos['entry'], atr_stop)
                else:
                    stop = pos['hw'] - atr_stop_mult * atr

                today_low = L[si, di]
                today_open = O[si, di]

                if not np.isnan(today_low) and today_low <= stop:
                    if not np.isnan(today_open) and today_open < stop:
                        sp = today_open
                    else:
                        sp = stop
                    pnl = (sp - pos['entry']) / pos['entry'] * 100
                    cash += pos['shares'] * sp * (1 - COMMISSION - STAMP_DUTY)
                    trades.append({'pnl': pnl, 'days': days_held,
                                   'di': di, 'reason': 'stop', 'year': year})
                    holdings.remove(pos)
                    stopped_out = True

            if not stopped_out:
                today_high = H[si, di]
                if not np.isnan(today_high) and today_high > 0:
                    pos['hw'] = max(pos['hw'], today_high)

            # Time-based exit
            if pos in holdings:
                if days_held >= max_hold_days:
                    sp = O[si, di] if not np.isnan(O[si, di]) else C[si, di]
                    if not np.isnan(sp) and sp > 0:
                        pnl = (sp - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * sp * (1 - COMMISSION - STAMP_DUTY)
                        trades.append({'pnl': pnl, 'days': days_held,
                                       'di': di, 'reason': 'time_stop', 'year': year})
                        holdings.remove(pos)

        # ---- Rebalance ----
        if di - last_rebalance >= rebalance_days:
            # Step 1: Apply quality filters
            quality_mask = np.ones(NS, dtype=bool)
            filter_desc = []
            for fname, (op, threshold) in quality_filters.items():
                if fname not in factors:
                    continue
                vals = factors[fname][:, di]
                valid = ~np.isnan(vals)
                if op == '>':
                    quality_mask &= valid & (vals > threshold)
                elif op == '<':
                    quality_mask &= valid & (vals < threshold)
                elif op == '>=':
                    quality_mask &= valid & (vals >= threshold)
                elif op == '<=':
                    quality_mask &= valid & (vals <= threshold)
                filter_desc.append(f"{fname}{op}{threshold}")

            # Step 2: Momentum filter — 20-day return > 0
            if 'MOM_20' in factors:
                mom_vals = factors['MOM_20'][:, di]
                valid_mom = ~np.isnan(mom_vals)
                quality_mask &= valid_mom & (mom_vals > 0)

            n_quality = quality_mask.sum()
            if n_quality < top_n * 2:
                # Not enough quality stocks — sell all, stay in cash
                for pos in list(holdings):
                    p = O[pos['si'], di]
                    if np.isnan(p) or p <= 0:
                        p = C[pos['si'], di]
                    if not np.isnan(p) and p > 0:
                        pnl = (p - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
                        trades.append({'pnl': pnl, 'days': (dates[di] - pos['ed']).days,
                                       'di': di, 'reason': 'no_quality', 'year': year})
                holdings = []
                last_rebalance = di
                daily_nav.append(cash)
                continue

            # Step 3: Composite score among quality stocks
            composite = np.zeros(NS)
            count = np.zeros(NS)
            for fname, w in zip(factor_names, weights):
                if fname not in factors:
                    continue
                arr = factors[fname]
                vals = arr[:, di]
                valid = ~np.isnan(vals)
                composite[valid] += w * vals[valid]
                count[valid] += abs(w)

            mask = (count > 0) & quality_mask
            if mask.sum() < top_n:
                last_rebalance = di
                daily_nav.append(cash)
                continue

            composite[mask] /= count[mask]
            composite[~mask] = -9999

            top_indices = set(np.argsort(-composite)[:top_n])
            current_indices = set(h['si'] for h in holdings)

            # Sell positions not in top
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
                                    'si': si, 'shares': shares, 'entry': p,
                                    'ed': dates[di], 'hw': p, 'be_active': False,
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


# =================================================================
# Main — grid search over defensive parameters
# =================================================================
if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V65 — Defensive Quality Factor Strategy (High Win Rate)", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()
    factors = compute_v65_factors(NS, ND, C, O, H, L, V)

    # Score weight portfolios to test
    score_portfolios = {
        'RelStr+VWCM+OIS': {'R_REL_STR_S3': 0.40, 'R_VWCM': 0.35, 'R_OIS': 0.25},
        'RelStrHeavy':      {'R_REL_STR_S3': 0.60, 'R_VWCM': 0.25, 'R_OIS': 0.15},
        'VWCMHeavy':        {'R_REL_STR_S3': 0.25, 'R_VWCM': 0.55, 'R_OIS': 0.20},
        'OISHeavy':         {'R_REL_STR_S3': 0.20, 'R_VWCM': 0.25, 'R_OIS': 0.55},
        'Equal':            {'R_REL_STR_S3': 0.333, 'R_VWCM': 0.333, 'R_OIS': 0.334},
    }

    # Quality filter configurations
    filter_configs = {
        'Std':  {'R_TENSION': ('>', 50), 'R_VOLATILITY_PCT': ('<', 50), 'R_OIS': ('>', 30)},
        'Tight': {'R_TENSION': ('>', 60), 'R_VOLATILITY_PCT': ('<', 40), 'R_OIS': ('>', 40)},
        'Loose': {'R_TENSION': ('>', 40), 'R_VOLATILITY_PCT': ('<', 60), 'R_OIS': ('>', 20)},
    }

    results = []
    for p_name, score_w in score_portfolios.items():
        for f_name, qf in filter_configs.items():
            for top_n in [1, 3, 5]:
                for rebal in [15, 20]:
                    for atr_m in [2.5, 3.0, 3.5]:
                        for be_trig in [6.0, 8.0, 10.0]:
                            r = backtest_v65(
                                factors, NS, ND, dates, C, O, H, L, V,
                                score_weights=score_w,
                                quality_filters=qf,
                                top_n=top_n,
                                rebalance_days=rebal,
                                atr_stop_mult=atr_m,
                                breakeven_trigger=be_trig,
                            )
                            if r:
                                r.update({
                                    'portfolio': p_name,
                                    'filter': f_name,
                                    'top_n': top_n,
                                    'rebal': rebal,
                                    'atr': atr_m,
                                    'be': be_trig,
                                })
                                results.append(r)
        print(f"  {p_name} done", flush=True)

    # Sort by win rate first, then by annual return
    results.sort(key=lambda x: (-x['wr'], -x['ann']))

    print(f"\n{'='*140}", flush=True)
    print(f"  TOP 30 RESULTS (sorted by Win Rate, then Annual Return)", flush=True)
    print(f"  {'Portfolio':<18s} {'Filter':<6s} {'Top':>3s} {'Reb':>3s} {'ATR':>4s} {'BE':>4s} | "
          f"{'Ann':>7s} {'N':>5s} {'TPY':>4s} {'WR':>5s} {'Edge':>6s} {'DD':>5s} "
          f"{'AvgW':>5s} {'AvgL':>5s}", flush=True)
    print(f"  {'-'*130}", flush=True)
    for r in results[:30]:
        print(f"  {r['portfolio']:<18s} {r['filter']:<6s} {r['top_n']:3d} {r['rebal']:3d} "
              f"{r['atr']:4.1f} {r['be']:4.1f} | "
              f"{r['ann']:+7.1f}% {r['n']:5d} {r['tpy']:4.0f} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}% "
              f"{r['avg_w']:5.1f}% {r['avg_l']:5.1f}%", flush=True)

    # Also show best by annual return among WR > 50%
    high_wr = [r for r in results if r['wr'] >= 50]
    if high_wr:
        high_wr.sort(key=lambda x: -x['ann'])
        print(f"\n  === BEST ANNUAL RETURN (among WR >= 50%) ===", flush=True)
        for r in high_wr[:15]:
            print(f"    {r['portfolio']:<18s} filter={r['filter']:<6s} "
                  f"Ann={r['ann']:+.1f}% WR={r['wr']:.1f}% DD={r['max_dd']:.1f}% "
                  f"(Top={r['top_n']}, Reb={r['rebal']}, ATR={r['atr']:.1f}, BE={r['be']:.1f})", flush=True)
    else:
        print(f"\n  No configurations achieved WR >= 50%", flush=True)
        # Show best WR anyway
        results.sort(key=lambda x: -x['wr'])
        print(f"  Best WR configurations:", flush=True)
        for r in results[:15]:
            print(f"    {r['portfolio']:<18s} filter={r['filter']:<6s} "
                  f"Ann={r['ann']:+.1f}% WR={r['wr']:.1f}% DD={r['max_dd']:.1f}%", flush=True)

    # Year-by-year for best WR result
    if results:
        results.sort(key=lambda x: (-x['wr'], -x['ann']))
        best = results[0]
        print(f"\n  Year-by-year: {best['portfolio']} filter={best['filter']} "
              f"(Ann={best['ann']:+.1f}%, WR={best['wr']:.1f}%, DD={best['max_dd']:.1f}%)", flush=True)
        for y in sorted(best.get('year_stats', {}).keys()):
            s = best['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    # Compare filter strictness
    print(f"\n  === FILTER STRICTNESS COMPARISON ===", flush=True)
    for f_name in ['Loose', 'Std', 'Tight']:
        filt_results = [r for r in results if r['filter'] == f_name]
        if filt_results:
            filt_results.sort(key=lambda x: (-x['wr'], -x['ann']))
            best_f = filt_results[0]
            print(f"  {f_name:<6s}: Ann={best_f['ann']:+.1f}% WR={best_f['wr']:.1f}% "
                  f"DD={best_f['max_dd']:.1f}% ({best_f['portfolio']}, "
                  f"Top={best_f['top_n']}, Reb={best_f['rebal']}, ATR={best_f['atr']:.1f})", flush=True)

    print(f"\n{'='*70}", flush=True)
