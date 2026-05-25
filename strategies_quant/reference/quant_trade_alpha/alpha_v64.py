"""
Alpha V64 — Regime-Adaptive Strategy
=====================================
Key insight: different factors work in different market regimes.
  - BULL: momentum factors dominate (trend is your friend)
  - BEAR: defensive/quality factors protect (low vol, stability)
  - NEUTRAL: mean-reversion factors work (buy oversold, sell overbought)

Implementation:
  1. Regime detection using di-1 data only (no look-ahead)
     - Market breadth: fraction of stocks above MA20
     - Market momentum: avg 20-day return of top-50 stocks by volume
     - Classification: BULL / BEAR / NEUTRAL
  2. Regime-specific factor weights
  3. Regime-specific ATR stop multipliers
  4. Custom backtest loop with dynamic regime switching
  5. Track regime-specific performance in year_stats

Target: WR > 50% by avoiding bad trades in bear markets
        and using appropriate factors for each regime.
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7 import COMMISSION, STAMP_DUTY, CASH0


# =====================================================================
# REGIME DETECTION (no look-ahead: uses di-1 data only)
# =====================================================================
def compute_regime_signals(NS, ND, C, V):
    """Compute market breadth and momentum for regime classification.

    All signals use data up to di-1 only.
    Returns: breadth[ND], market_mom[ND], both np.nan before warmup.
    """
    breadth = np.full(ND, np.nan)
    market_mom = np.full(ND, np.nan)

    # Pre-compute MA20 for each stock (using di-1 data)
    for di in range(21, ND):
        # --- Market breadth: fraction of stocks where C[:,di-1] > MA20[:,di-1] ---
        above_ma = 0
        total = 0
        for si in range(min(200, NS)):
            c = C[si, di - 1]
            if np.isnan(c):
                continue
            # MA20 from C[si, di-1-20 : di] (last 20 closes ending at di-1)
            window = C[si, di - 1 - 20:di]
            valid = window[~np.isnan(window)]
            if len(valid) < 15:
                continue
            ma20 = np.mean(valid)
            if c > ma20:
                above_ma += 1
            total += 1
        if total > 50:
            breadth[di] = above_ma / total * 100  # 0-100 scale

        # --- Market momentum: average 20-day return of top-50 stocks by volume ---
        # Find top-50 stocks by average volume over last 20 days
        vol_sums = np.zeros(NS)
        vol_counts = np.zeros(NS, dtype=int)
        for dd in range(max(di - 20, 1), di):
            for si in range(NS):
                v = V[si, dd]
                if not np.isnan(v):
                    vol_sums[si] += v
                    vol_counts[si] += 1
        vol_counts[vol_counts == 0] = 1
        avg_vol = vol_sums / vol_counts
        top50 = np.argsort(-avg_vol)[:50]

        rets = []
        for si in top50:
            c_now = C[si, di - 1]
            c_20 = C[si, di - 1 - 20]
            if (not np.isnan(c_now) and not np.isnan(c_20)
                    and c_20 > 0):
                rets.append((c_now - c_20) / c_20 * 100)
        if len(rets) > 10:
            market_mom[di] = np.mean(rets)

    return breadth, market_mom


def classify_regime(breadth_val, market_mom_val):
    """Classify current market regime from breadth and momentum.

    BULL:    breadth > 60 AND market_mom > 0
    BEAR:    breadth < 40 OR market_mom < -5
    NEUTRAL: everything else
    """
    if np.isnan(breadth_val) or np.isnan(market_mom_val):
        return 'NEUTRAL'
    if breadth_val > 60 and market_mom_val > 0:
        return 'BULL'
    if breadth_val < 40 or market_mom_val < -5:
        return 'BEAR'
    return 'NEUTRAL'


# =====================================================================
# REGIME-SPECIFIC FACTOR WEIGHTS
# =====================================================================
REGIME_WEIGHTS = {
    'BULL': {
        # Momentum factors: trend is your friend
        'R_REL_STR_S3':  0.20,   # Short-span relative strength (fast momentum)
        'R_REL_STR_S10': 0.15,   # Medium-span relative strength
        'R_TREND_ACC':   0.15,   # Trend acceleration (from ACT decomposition)
        'R_TREND_MOM':   0.10,   # Trend component momentum
        'R_TENSION':      0.10,   # Structural tension (quality)
        'R_BUY_FRAC':     0.10,   # Buying pressure
        'R_VWCM':         0.10,   # Volume-conditional momentum
        'R_SHOCK_MOM':    0.10,   # Shock momentum (from ACT decomposition)
    },
    'BEAR': {
        # Defensive factors: quality, low volatility, stability
        'R_OIS':           0.20,  # Overnight-Intraday Spread (stability)
        'R_R_SQUARED':     0.15,  # Trend quality (high R2 = stable trend)
        'R_TENSION':       0.15,  # Structural quality
        'R_PCS':           0.10,  # Price curve stability (accumulation)
        'R_GK_RV':         0.10,  # Garman-Klass vol (inverted: low vol = good)
        'R_CS_SPREAD':     0.10,  # Corwin-Schultz spread (liquid = safe)
        'R_BWP_BNW':       0.10,  # Bandwidth × Body (quality signal)
        'R_SMA_DEV':       0.10,  # SMA deviation (inverted in bear = mean-revert)
    },
    'NEUTRAL': {
        # Mean-reversion factors: buy oversold, sell overbought
        'R_TENSION':       0.15,
        'R_SMA_DEV':       0.15,
        'R_BWP_BNW':       0.15,
        'R_R_SQUARED':     0.10,
        'R_VWCM':          0.10,
        'R_BUY_FRAC':      0.10,
        'R_VPIN':          0.10,
        'R_OIS':           0.05,
        'R_REL_STR_S10':   0.10,
    },
}

# Regime-specific ATR stop multipliers
REGIME_ATR_STOP = {
    'BULL':    1.5,   # Tighter stops: trend is your friend, quick cut if wrong
    'BEAR':    3.0,   # Wider stops: more noise, need room (or skip trading)
    'NEUTRAL': 2.0,   # Moderate stops
}

# Whether to skip trading entirely in this regime
REGIME_SKIP = {
    'BULL':    False,
    'BEAR':    False,  # Trade but with defensive factors
    'NEUTRAL': False,
}


# =====================================================================
# REGIME-ADAPTIVE BACKTEST (custom loop)
# =====================================================================
def backtest_regime_adaptive(factors, NS, ND, dates, C, O, H, L, V,
                              breadth, market_mom,
                              top_n=3, rebalance_days=5,
                              bear_scale=0.5):
    """Backtest with regime-adaptive factor weights and exit rules.

    On each rebalance day:
      1. Classify regime from breadth/momentum at di-1
      2. Select factor weights for that regime
      3. Apply regime-specific ATR stop multiplier
      4. Optionally reduce position size in BEAR

    Returns standard result dict with extra 'regime_stats'.
    """
    # Pre-compute regime at each day
    regimes = ['NEUTRAL'] * ND
    for di in range(ND):
        if not np.isnan(breadth[di]) and not np.isnan(market_mom[di]):
            regimes[di] = classify_regime(breadth[di], market_mom[di])

    cash = float(CASH0)
    holdings = []
    trades = []
    last_rebalance = -999
    year_stats = {}
    daily_nav = []

    # Track regime-specific trade counts
    regime_trade_counts = {'BULL': 0, 'BEAR': 0, 'NEUTRAL': 0}
    regime_trade_wins = {'BULL': 0, 'BEAR': 0, 'NEUTRAL': 0}
    regime_trade_pnl = {'BULL': 0.0, 'BEAR': 0.0, 'NEUTRAL': 0.0}

    # Track the current ATR stop multiplier (may change with regime)
    current_atr_mult = 2.0

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year
        regime = regimes[di]
        current_atr_mult = REGIME_ATR_STOP.get(regime, 2.0)

        # --- ATR stop-loss check (no look-ahead) ---
        for pos in list(holdings):
            si = pos['si']
            stopped_out = False

            # Use the ATR mult stored with the position (set at entry time)
            atr_mult = pos.get('atr_mult', 2.0)

            if atr_mult > 0:
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
                    stop = pos['hw'] - atr_mult * atr
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
                            'pnl': pnl,
                            'days': (dates[di] - pos['ed']).days,
                            'di': di, 'reason': 'stop', 'year': year,
                            'regime': pos.get('entry_regime', 'NEUTRAL'),
                        })
                        holdings.remove(pos)
                        stopped_out = True

            if not stopped_out:
                today_high = H[si, di]
                if not np.isnan(today_high) and today_high > 0:
                    pos['hw'] = max(pos['hw'], today_high)

            # Time-based stop: max 60 days
            if pos in holdings:
                days_held = (dates[di] - pos['ed']).days
                if days_held >= 60:
                    sp = O[si, di] if not np.isnan(O[si, di]) else C[si, di]
                    if not np.isnan(sp) and sp > 0:
                        pnl = (sp - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * sp * (1 - COMMISSION - STAMP_DUTY)
                        trades.append({
                            'pnl': pnl, 'days': days_held,
                            'di': di, 'reason': 'time_stop', 'year': year,
                            'regime': pos.get('entry_regime', 'NEUTRAL'),
                        })
                        holdings.remove(pos)

        # --- Rebalance ---
        if di - last_rebalance >= rebalance_days:
            # Regime-specific logic
            skip = REGIME_SKIP.get(regime, False)
            scale = bear_scale if regime == 'BEAR' else 1.0

            if skip:
                # Sell all holdings
                for pos in list(holdings):
                    p = O[pos['si'], di]
                    if np.isnan(p) or p <= 0:
                        p = C[pos['si'], di]
                    if not np.isnan(p) and p > 0:
                        pnl = (p - pos['entry']) / pos['entry'] * 100
                        cash += pos['shares'] * p * (1 - COMMISSION - STAMP_DUTY)
                        trades.append({
                            'pnl': pnl,
                            'days': (dates[di] - pos['ed']).days,
                            'di': di, 'reason': 'regime_exit', 'year': year,
                            'regime': pos.get('entry_regime', 'NEUTRAL'),
                        })
                holdings = []
                last_rebalance = di
                continue

            # Get regime-specific weights
            factor_weights = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS['NEUTRAL'])
            atr_mult_for_new = REGIME_ATR_STOP.get(regime, 2.0)

            factor_names = list(factor_weights.keys())
            weights = np.array([factor_weights[f] for f in factor_names])

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
                last_rebalance = di
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
                        trades.append({
                            'pnl': pnl,
                            'days': (dates[di] - pos['ed']).days,
                            'di': di, 'reason': 'rebalance', 'year': year,
                            'regime': pos.get('entry_regime', 'NEUTRAL'),
                        })
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
                                    'ed': dates[di], 'hw': p,
                                    'atr_mult': atr_mult_for_new,
                                    'entry_regime': regime,
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
            trades.append({
                'pnl': pnl, 'days': 999, 'di': ND - 1, 'reason': 'end',
                'year': dates[ND - 1].year,
                'regime': pos.get('entry_regime', 'NEUTRAL'),
            })

    if not trades:
        return None

    # Overall stats
    days_total = (dates[ND - 1] - dates[MIN_TRAIN]).days
    yr = max(days_total / 365.25, 0.01)
    ann = ((cash / CASH0) ** (1 / yr) - 1) * 100
    nw = sum(1 for t in trades if t['pnl'] > 0)
    wr = nw / max(len(trades), 1) * 100
    avg_w = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if nw > 0 else 0
    avg_l = np.mean([abs(t['pnl']) for t in trades if t['pnl'] <= 0]) if nw < len(trades) else 0

    # Year stats
    for t in trades:
        y = t.get('year', 'unknown')
        if y not in year_stats:
            year_stats[y] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
        year_stats[y]['trades'] += 1
        if t['pnl'] > 0:
            year_stats[y]['wins'] += 1
        year_stats[y]['total_pnl'] += t['pnl']

    # Regime-specific stats
    for t in trades:
        r = t.get('regime', 'NEUTRAL')
        if r in regime_trade_counts:
            regime_trade_counts[r] += 1
            if t['pnl'] > 0:
                regime_trade_wins[r] += 1
            regime_trade_pnl[r] += t['pnl']

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

    regime_stats = {}
    for r in ['BULL', 'BEAR', 'NEUTRAL']:
        n = regime_trade_counts[r]
        w = regime_trade_wins[r]
        pnl = regime_trade_pnl[r]
        regime_stats[r] = {
            'trades': n,
            'wins': w,
            'wr': round(w / max(n, 1) * 100, 1),
            'total_pnl': round(pnl, 1),
        }

    return {
        'ann': round(ann, 1), 'n': len(trades), 'wr': round(wr, 1),
        'avg_w': round(avg_w, 1), 'avg_l': round(avg_l, 1),
        'edge': round((nw / max(len(trades), 1)) * avg_w -
                       (1 - nw / max(len(trades), 1)) * avg_l, 2),
        'max_dd': round(max_dd, 1), 'tpy': round(len(trades) / yr, 1),
        'final': round(cash, 0), 'year_stats': year_stats,
        'regime_stats': regime_stats,
    }


# =====================================================================
# MAIN
# =====================================================================
if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V64 — Regime-Adaptive Strategy", flush=True)
    print("  Key insight: different factors work in different regimes", flush=True)
    print("=" * 70, flush=True)

    # Load data
    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Compute regime signals
    print("\n[Regime] Computing regime signals (breadth + momentum)...", flush=True)
    t0 = time.time()
    breadth, market_mom = compute_regime_signals(NS, ND, C, V)
    print(f"  Regime signals done ({time.time()-t0:.0f}s)", flush=True)

    # Print regime distribution
    regime_counts = {'BULL': 0, 'BEAR': 0, 'NEUTRAL': 0}
    for di in range(MIN_TRAIN, ND):
        r = classify_regime(breadth[di], market_mom[di])
        regime_counts[r] += 1
    total_days = ND - MIN_TRAIN
    print(f"  Regime distribution over {total_days} trading days:", flush=True)
    for r in ['BULL', 'NEUTRAL', 'BEAR']:
        pct = regime_counts[r] / max(total_days, 1) * 100
        print(f"    {r:8s}: {regime_counts[r]:5d} days ({pct:.1f}%)", flush=True)

    # Compute factors from all prior alpha versions
    print("\n[Factors] Computing factor library...", flush=True)
    t0 = time.time()
    from alpha_v44 import compute_v41_factors_only
    from alpha_v48 import compute_v48_factors
    from alpha_v49 import compute_v49_factors
    from alpha_v52 import compute_v52_factors
    from alpha_v55 import compute_decomposed_factors

    v41 = compute_v41_factors_only(NS, ND, C, O, H, L, V)
    v48 = compute_v48_factors(NS, ND, C, O, H, L, V)
    v49 = compute_v49_factors(NS, ND, C, O, H, L, V)
    v52 = compute_v52_factors(NS, ND, C, O, H, L, V)
    v55 = compute_decomposed_factors(NS, ND, C, O, H, L, V)

    # V61 REL_STRENGTH factors
    from alpha_v61 import compute_rel_strength_factors
    rel = compute_rel_strength_factors(NS, ND, C, O, H, L, V)

    all_factors = {**v41, **v48, **v49, **v52, **v55, **rel}
    print(f"  Factors computed: {len(all_factors)} factors ({time.time()-t0:.0f}s)", flush=True)

    # Check which regime weights are available
    for regime, wdict in REGIME_WEIGHTS.items():
        available = [f for f in wdict if f in all_factors]
        missing = [f for f in wdict if f not in all_factors]
        print(f"  {regime}: {len(available)}/{len(wdict)} factors available", flush=True)
        if missing:
            print(f"    Missing: {missing}", flush=True)

    # =====================================================================
    # Run backtests
    # =====================================================================
    results = []

    # --- Test 1: Regime-adaptive with default parameters ---
    print("\n[Backtest] Regime-adaptive parameter sweep...", flush=True)

    for top_n in [1, 2, 3]:
        for rebal in [3, 5, 7, 10]:
            for bear_scale in [0.3, 0.5, 0.7, 1.0]:
                r = backtest_regime_adaptive(
                    all_factors, NS, ND, dates, C, O, H, L, V,
                    breadth, market_mom,
                    top_n=top_n, rebalance_days=rebal,
                    bear_scale=bear_scale,
                )
                if r:
                    r['top_n'] = top_n
                    r['rebal'] = rebal
                    r['bear_scale'] = bear_scale
                    r['test'] = f'T{top_n}_R{rebal}_BS{bear_scale}'
                    results.append(r)

    # Print progress dots
    total_tests = len(results)
    print(f"  {total_tests} configurations tested", flush=True)

    # --- Baseline comparison: static weights (no regime) ---
    # Use NEUTRAL weights as the static baseline
    print("\n[Backtest] Static baseline (NEUTRAL weights, no regime)...", flush=True)
    from alpha_v7c import backtest_v7c
    static_weights = REGIME_WEIGHTS['NEUTRAL']
    # Filter to only available factors
    static_avail = {k: v for k, v in static_weights.items() if k in all_factors}
    if static_avail:
        total_w = sum(static_avail.values())
        static_norm = {k: v / total_w for k, v in static_avail.items()}
        for top_n in [1, 2, 3]:
            for rebal in [3, 5, 7, 10]:
                for atr in [1.5, 2.0, 3.0]:
                    r = backtest_v7c(
                        static_norm, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=top_n, rebalance_days=rebal, atr_stop_mult=atr,
                    )
                    if r:
                        r['top_n'] = top_n
                        r['rebal'] = rebal
                        r['atr'] = atr
                        r['test'] = f'STATIC_T{top_n}_R{rebal}_A{atr}'
                        results.append(r)
        print(f"  Static baselines added", flush=True)

    # =====================================================================
    # Results
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'=' * 120}", flush=True)
    print(f"  ALPHA V64 RESULTS — REGIME-ADAPTIVE STRATEGY", flush=True)
    print(f"  {'Test':<30s} | {'Ann':>7s} {'N':>5s} {'TPY':>4s} {'WR':>5s} "
          f"{'Edge':>6s} {'DD':>5s} | Notes", flush=True)
    print(f"  {'-' * 110}", flush=True)

    for r in results[:50]:
        pos_mark = " ALL+" if all_positive(r) else ""
        regime_info = ""
        rs = r.get('regime_stats', {})
        if rs:
            parts = []
            for rg in ['BULL', 'NEUTRAL', 'BEAR']:
                if rs[rg]['trades'] > 0:
                    parts.append(f"{rg[0]}={rs[rg]['wr']:.0f}%")
            regime_info = " ".join(parts)
        print(f"  {r['test']:<30s} | {r['ann']:+7.1f}% {r['n']:5d} {r['tpy']:4.0f} "
              f"{r['wr']:5.1f}% {r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark} | "
              f"{regime_info}", flush=True)

    # Regime-specific breakdown for top results
    print(f"\n  === REGIME-SPECIFIC BREAKDOWN (Top 5) ===", flush=True)
    regime_results = [r for r in results if 'regime_stats' in r]
    for i, r in enumerate(regime_results[:5]):
        rs = r['regime_stats']
        print(f"\n  #{i+1}: {r['test']}  Ann={r['ann']:+.1f}%  DD={r['max_dd']:.1f}%", flush=True)
        for rg in ['BULL', 'NEUTRAL', 'BEAR']:
            s = rs[rg]
            if s['trades'] > 0:
                print(f"    {rg:8s}: {s['trades']:4d} trades, "
                      f"WR={s['wr']:.1f}%, PnL={s['total_pnl']:+.0f}%", flush=True)

    # Year-by-year for best
    if results:
        best = results[0]
        print(f"\n  === YEAR-BY-YEAR: {best['test']} ===", flush=True)
        print(f"  Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  DD={best['max_dd']:.1f}%", flush=True)
        for y in sorted(best.get('year_stats', {}).keys()):
            s = best['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, "
                  f"pnl={s['total_pnl']:+.0f}%", flush=True)

    # Regime-adaptive vs static comparison
    adaptive = [r for r in results if not r['test'].startswith('STATIC')]
    static = [r for r in results if r['test'].startswith('STATIC')]
    if adaptive and static:
        best_adaptive = max(adaptive, key=lambda x: x['ann'])
        best_static = max(static, key=lambda x: x['ann'])
        print(f"\n  === REGIME-ADAPTIVE vs STATIC ===", flush=True)
        print(f"  Best adaptive: {best_adaptive['test']:<30s} "
              f"Ann={best_adaptive['ann']:+.1f}%  DD={best_adaptive['max_dd']:.1f}%", flush=True)
        print(f"  Best static:   {best_static['test']:<30s} "
              f"Ann={best_static['ann']:+.1f}%  DD={best_static['max_dd']:.1f}%", flush=True)
        delta = best_adaptive['ann'] - best_static['ann']
        print(f"  Delta: {delta:+.1f}%", flush=True)

    print(f"\n{'=' * 70}", flush=True)
