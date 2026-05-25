"""
Alpha V43 — Trade Analysis + Market Timing
===========================================
Phase 1: Analyze actual V41 trades to find win/loss patterns
Phase 2: Market timing overlay using breadth/momentum signals

Uses V41 optimal: equal weights, ATR=0.8, rebalance=5d, top_n=1
"""
import sys, os, time, warnings
import numpy as np
import pickle
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7 import COMMISSION, STAMP_DUTY, CASH0


def backtest_v43_verbose(factor_weights, factors, NS, ND, dates, C, O, H, L, V,
                         top_n=1, rebalance_days=5, atr_stop_mult=0.8,
                         market_timing=False, breadth=None, market_ret20=None,
                         cash_frac=1.0):
    """Backtest with full trade logging + optional market timing."""
    factor_names = list(factor_weights.keys())
    weights = np.array([factor_weights[f] for f in factor_names])

    cash = float(CASH0)
    holdings = []
    trades = []
    last_rebalance = -999
    year_stats = {}
    equity_curve = []

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # ATR stop loss
        for pos in list(holdings):
            si = pos['si']
            stopped_out = False

            if atr_stop_mult > 0:
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
                        trades.append({
                            'si': si, 'pnl': pnl, 'days': (dates[di] - pos['ed']).days,
                            'di': di, 'reason': 'stop', 'year': year,
                            'entry': pos['entry'], 'exit': sp,
                            'composite': pos.get('score', 0),
                            'hold_days': (dates[di] - pos['ed']).days,
                        })
                        holdings.remove(pos)
                        stopped_out = True

            if not stopped_out:
                today_high = H[si, di]
                if not np.isnan(today_high) and today_high > 0:
                    pos['hw'] = max(pos['hw'], today_high)

            if pos in holdings:
                days_held = (dates[di] - pos['ed']).days
                if days_held >= 60:
                    sp = O[si, di] if not np.isnan(O[si, di]) else C[si, di]
                    if not np.isnan(sp) and sp > 0:
                        pnl = (sp - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * sp * (1 - COMMISSION - STAMP_DUTY)
                        trades.append({
                            'si': si, 'pnl': pnl, 'days': days_held,
                            'di': di, 'reason': 'time_stop', 'year': year,
                            'entry': pos['entry'], 'exit': sp,
                            'composite': pos.get('score', 0),
                            'hold_days': days_held,
                        })
                        holdings.remove(pos)

        # Record equity
        eq = cash
        for pos in holdings:
            p = C[pos['si'], di]
            if not np.isnan(p) and p > 0:
                eq += pos['shares'] * p
        equity_curve.append((di, eq))

        # Rebalance
        if di - last_rebalance >= rebalance_days:
            # Market timing check
            skip = False
            scale = cash_frac
            if market_timing and breadth is not None and market_ret20 is not None:
                br = breadth[di]
                mr = market_ret20[di]
                # If breadth < 40% (bear market), skip trading
                if not np.isnan(br) and br < 40:
                    skip = True
                # If market declining, reduce position
                elif not np.isnan(mr) and mr < -0.03:
                    scale = 0.5

            if skip:
                for pos in list(holdings):
                    p = O[pos['si'], di]
                    if np.isnan(p) or p <= 0:
                        p = C[pos['si'], di]
                    if not np.isnan(p) and p > 0:
                        pnl = (p - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
                        trades.append({
                            'si': pos['si'], 'pnl': pnl, 'days': (dates[di] - pos['ed']).days,
                            'di': di, 'reason': 'regime_exit', 'year': year,
                            'entry': pos['entry'], 'exit': p,
                            'composite': pos.get('score', 0),
                            'hold_days': (dates[di] - pos['ed']).days,
                        })
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
                            'si': pos['si'], 'pnl': pnl, 'days': (dates[di] - pos['ed']).days,
                            'di': di, 'reason': 'rebalance', 'year': year,
                            'entry': pos['entry'], 'exit': p,
                            'composite': pos.get('score', 0),
                            'hold_days': (dates[di] - pos['ed']).days,
                        })
                        holdings.remove(pos)

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
                                score = composite[si]
                                holdings.append({
                                    'si': si, 'shares': shares, 'entry': p,
                                    'ed': dates[di], 'hw': p, 'score': score
                                })
            last_rebalance = di

    # Close remaining
    for pos in holdings:
        p = C[pos['si'], ND - 1]
        if not np.isnan(p) and p > 0:
            pnl = (p - pos['entry']) / pos['entry'] * 100
            cash += pos['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
            trades.append({
                'si': pos['si'], 'pnl': pnl, 'days': 999, 'di': ND - 1,
                'reason': 'end', 'year': dates[ND - 1].year,
                'entry': pos['entry'], 'exit': p,
                'composite': pos.get('score', 0), 'hold_days': 999,
            })

    if cash <= 0 or not trades:
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
        'trades': trades, 'equity_curve': equity_curve,
    }


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V43 — Trade Analysis + Market Timing", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    from alpha_v7 import compute_all_factors
    from alpha_v7b import compute_interaction_factors
    from alpha_v7d import compute_extra_factors
    from alpha_v7e import compute_v7e_factors
    from alpha_v7f import compute_advanced_interactions
    from alpha_v8 import compute_v8_factors, compute_v8_interactions
    from alpha_v9 import compute_v9_factors, compute_v9_interactions
    from alpha_v10 import compute_v10_factors, compute_v10_interactions
    from alpha_v11 import compute_v11_factors, compute_v11_interactions
    from alpha_v14 import compute_v14_factors, compute_v14_interactions

    base = compute_all_factors(NS, ND, C, O, H, L, V)
    inter = compute_interaction_factors(base, NS, ND, C, O, H, L, V)
    extra = compute_extra_factors(NS, ND, C, O, H, L, V)
    v7e = compute_v7e_factors(NS, ND, C, O, H, L, V)
    adv = compute_advanced_interactions({**base, **inter, **extra, **v7e}, NS, ND)
    v8f = compute_v8_factors(NS, ND, C, O, H, L, V)
    v8_all = {**base, **inter, **extra, **v7e, **adv, **v8f}
    v8_inter = compute_v8_interactions(v8_all, NS, ND)
    v8_all.update(v8_inter)
    v9f = compute_v9_factors(NS, ND, C, O, H, L, V)
    v9_all = {**v8_all, **v9f}
    v9_inter = compute_v9_interactions(v9_all, NS, ND)
    v9_all.update(v9_inter)
    v10f = compute_v10_factors(NS, ND, C, O, H, L, V)
    v10_all = {**v9_all, **v10f}
    v10_inter = compute_v10_interactions(v10_all, NS, ND)
    v10_all.update(v10_inter)
    v11f = compute_v11_factors(NS, ND, C, O, H, L, V)
    v11_all = {**v10_all, **v11f}
    v11_inter = compute_v11_interactions(v11_all, NS, ND)
    v11_all.update(v11_inter)
    v14f = compute_v14_factors(NS, ND, C, O, H, L, V)
    all_factors = {**v11_all, **v14f}
    v14_inter = compute_v14_interactions(all_factors, NS, ND)
    all_factors.update(v14_inter)

    v41_weights = {'R_BWP_BNW': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2,
                   'R_SMA_DEV': 0.2, 'R_HAR_RV_RATIO_INV': 0.2}

    # =====================================================================
    # PHASE 1: Run V41 baseline with verbose logging
    # =====================================================================
    print("\n  Phase 1: Running V41 baseline with trade logging...", flush=True)
    r_base = backtest_v43_verbose(v41_weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                  top_n=1, rebalance_days=5, atr_stop_mult=0.8)
    if not r_base:
        print("  ERROR: Baseline failed!", flush=True)
        sys.exit(1)

    trades = r_base['trades']
    print(f"\n  V41 Baseline: {r_base['ann']:+.1f}% N={r_base['n']} WR={r_base['wr']:.1f}% DD={r_base['max_dd']:.1f}%", flush=True)

    # =====================================================================
    # TRADE ANALYSIS
    # =====================================================================
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]

    print(f"\n  === TRADE ANALYSIS ===", flush=True)
    print(f"  Total: {len(trades)} trades, {len(wins)} wins, {len(losses)} losses", flush=True)
    print(f"  Win rate: {len(wins)/len(trades)*100:.1f}%", flush=True)
    if wins:
        print(f"  Avg win: +{np.mean([t['pnl'] for t in wins]):.2f}%", flush=True)
        print(f"  Max win: +{max(t['pnl'] for t in wins):.2f}%", flush=True)
        print(f"  Avg win hold: {np.mean([t['hold_days'] for t in wins]):.1f}d", flush=True)
    if losses:
        print(f"  Avg loss: {np.mean([t['pnl'] for t in losses]):.2f}%", flush=True)
        print(f"  Max loss: {min(t['pnl'] for t in losses):.2f}%", flush=True)
        print(f"  Avg loss hold: {np.mean([t['hold_days'] for t in losses]):.1f}d", flush=True)

    # Exit reason analysis
    print(f"\n  === EXIT REASON ANALYSIS ===", flush=True)
    for reason in ['rebalance', 'stop', 'time_stop', 'regime_exit', 'end']:
        rt = [t for t in trades if t['reason'] == reason]
        if rt:
            wr = sum(1 for t in rt if t['pnl'] > 0) / len(rt) * 100
            avg = np.mean([t['pnl'] for t in rt])
            print(f"  {reason:15s}: {len(rt):4d} trades, WR={wr:.0f}%, avg={avg:+.2f}%", flush=True)

    # Win by holding period
    print(f"\n  === WIN RATE BY HOLDING PERIOD ===", flush=True)
    for lo, hi in [(1, 3), (4, 5), (6, 10), (11, 20), (21, 30), (31, 60)]:
        ht = [t for t in trades if lo <= t['hold_days'] <= hi]
        if ht:
            wr = sum(1 for t in ht if t['pnl'] > 0) / len(ht) * 100
            avg = np.mean([t['pnl'] for t in ht])
            print(f"  {lo:2d}-{hi:2d}d: {len(ht):4d} trades, WR={wr:.0f}%, avg={avg:+.2f}%", flush=True)

    # Win by composite score
    print(f"\n  === WIN RATE BY COMPOSITE SCORE ===", flush=True)
    scores = [t['composite'] for t in trades if not np.isnan(t.get('composite', np.nan))]
    if scores:
        for lo, hi in [(0, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]:
            ht = [t for t in trades if lo <= t.get('composite', 0) < hi]
            if ht:
                wr = sum(1 for t in ht if t['pnl'] > 0) / len(ht) * 100
                avg = np.mean([t['pnl'] for t in ht])
                print(f"  Score {lo:3d}-{hi:3d}: {len(ht):4d} trades, WR={wr:.0f}%, avg={avg:+.2f}%", flush=True)

    # Year-by-year trade patterns
    print(f"\n  === WINNING VS LOSING YEARS ===", flush=True)
    for y in sorted(r_base['year_stats'].keys()):
        s = r_base['year_stats'][y]
        yt = [t for t in trades if t['year'] == y]
        if not yt:
            continue
        w_yt = [t for t in yt if t['pnl'] > 0]
        l_yt = [t for t in yt if t['pnl'] <= 0]
        stop_yt = [t for t in yt if t['reason'] == 'stop']
        avg_win = np.mean([t['pnl'] for t in w_yt]) if w_yt else 0
        avg_loss = np.mean([t['pnl'] for t in l_yt]) if l_yt else 0
        print(f"  {y}: {s['trades']:3d}t WR={s['wins']/max(s['trades'],1)*100:.0f}% "
              f"avgW={avg_win:+.1f}% avgL={avg_loss:+.1f}% stops={len(stop_yt)} "
              f"total={s['total_pnl']:+.0f}%", flush=True)

    # =====================================================================
    # PHASE 2: Market Timing Experiments
    # =====================================================================
    print(f"\n  Phase 2: Market timing experiments...", flush=True)

    # Compute market regime signals
    print("  Computing market regime...", flush=True)
    avg_close = np.zeros(ND)
    for di in range(ND):
        avg_close[di] = np.nanmean(C[:50, di])

    market_ret20 = np.full(ND, np.nan)
    market_ret5 = np.full(ND, np.nan)
    breadth = np.full(ND, np.nan)

    for di in range(21, ND):
        if not np.isnan(avg_close[di - 1]) and not np.isnan(avg_close[di - 21]) and avg_close[di - 21] > 0:
            market_ret20[di] = (avg_close[di - 1] - avg_close[di - 21]) / avg_close[di - 21]
        if not np.isnan(avg_close[di - 1]) and not np.isnan(avg_close[di - 6]) and avg_close[di - 6] > 0:
            market_ret5[di] = (avg_close[di - 1] - avg_close[di - 6]) / avg_close[di - 6]

        above_ma = 0
        total = 0
        for si in range(min(200, NS)):
            c = C[si, di - 1]
            if np.isnan(c):
                continue
            c20 = C[si, di - 21:di]
            valid = c20[~np.isnan(c20)]
            if len(valid) < 15:
                continue
            ma20 = np.mean(valid)
            if c > ma20:
                above_ma += 1
            total += 1
        if total > 50:
            breadth[di] = above_ma / total * 100

    print("  Market regime computed", flush=True)

    timing_results = []

    # Test 1: No timing (baseline)
    timing_results.append({'name': 'NO_TIMING', 'ann': r_base['ann'],
                           'dd': r_base['max_dd'], 'n': r_base['n']})

    # Test 2: Various timing configs
    configs = [
        ('BR>40', True, 40, -0.05, 1.0),
        ('BR>45', True, 45, -0.05, 1.0),
        ('BR>35', True, 35, -0.05, 1.0),
        ('BR>40_MR>-3%', True, 40, -0.03, 1.0),
        ('BR>40_HALF', True, 40, -0.03, 0.5),
        ('BR>45_MR>-5%', True, 45, -0.05, 1.0),
        ('MR>0', True, 0, 0.0, 1.0),
    ]

    for name, mt, min_br, min_mr, cf in configs:
        r = backtest_v43_verbose(v41_weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=5, atr_stop_mult=0.8,
                                market_timing=mt, breadth=breadth, market_ret20=market_ret20,
                                cash_frac=cf)
        if r:
            timing_results.append({'name': name, 'ann': r['ann'],
                                   'dd': r['max_dd'], 'n': r['n']})
            # Year breakdown
            for y in sorted(r['year_stats'].keys()):
                s = r['year_stats'][y]
                wr = s['wins'] / max(s['trades'], 1) * 100
                print(f"    {name} {y}: {s['trades']:3d}t WR={wr:.0f}% pnl={s['total_pnl']:+.0f}%", flush=True)
        print(f"  {name}: {r['ann']:+.1f}% DD={r['max_dd']:.1f}%" if r else f"  {name}: FAILED", flush=True)

    # Results
    print(f"\n  === MARKET TIMING COMPARISON ===", flush=True)
    print(f"  {'Config':<20s} | {'Ann':>7s} {'N':>5s} {'DD':>5s} | vs base", flush=True)
    print(f"  {'-'*55}", flush=True)
    for r in sorted(timing_results, key=lambda x: -x['ann']):
        delta = r['ann'] - r_base['ann']
        print(f"  {r['name']:<20s} | {r['ann']:+7.1f}% {r['n']:5d} {r['dd']:5.1f}% | {delta:+.1f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
