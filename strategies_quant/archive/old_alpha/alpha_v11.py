"""
Alpha V11 — Classic Technical + Market Relative Factors
========================================================
V10 best: BwpBNW +248.0% DD=32.0%, SdBNWx2 +222.6% DD=28.3%

New factors (from strategy study + philosophy):
  1. RSI(14): Relative Strength Index — overbought/oversold oscillator
  2. MACD_HIST: MACD histogram — EMA convergence/divergence momentum
  3. KER: Kaufman Efficiency Ratio — net displacement / total path length
  4. REL_STR: Relative strength — stock return vs market average

LOOK-AHEAD SELF-CHECK:
  [x] All factors use ONLY data up to di-1 (yesterday's close)
  [x] Results stored at index di, read by backtest at di
  [x] No same-day data used for any computation
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7 import compute_all_factors, COMMISSION, STAMP_DUTY, CASH0
from alpha_v7b import compute_interaction_factors
from alpha_v7d import compute_extra_factors
from alpha_v7e import compute_v7e_factors
from alpha_v7f import compute_advanced_interactions
from alpha_v8 import compute_v8_factors, compute_v8_interactions
from alpha_v9 import compute_v9_factors, compute_v9_interactions
from alpha_v10 import compute_v10_factors, compute_v10_interactions
from alpha_v7c import backtest_v7c


def _calc_ema(values, period):
    """EMA calculation on numpy array."""
    n = len(values)
    result = np.empty(n)
    result[0] = values[0]
    k = 2.0 / (period + 1)
    for i in range(1, n):
        result[i] = values[i] * k + result[i - 1] * (1 - k)
    return result


def compute_v11_factors(NS, ND, C, O, H, L, V):
    """V11 factors — STRICT no look-ahead.

    SELF-CHECK RULES:
    1. d = di - 1 (use yesterday as "current" data)
    2. Store result at index di (backtest reads at di)
    3. Never access C[si, di], O[si, di], H[si, di], L[si, di], V[si, di]
    """
    t0 = time.time()
    new = {}

    # === 1. RSI(14): Relative Strength Index ===
    RSI = np.full((NS, ND), np.nan)
    rsi_period = 14
    for si in range(NS):
        avg_gain = np.nan
        avg_loss = np.nan
        for di in range(rsi_period + 2, ND):
            d = di - 1  # SELF-CHECK: d = yesterday
            # Calculate gain/loss for day d
            if np.isnan(C[si, d]) or np.isnan(C[si, d - 1]) or C[si, d - 1] <= 0:
                continue
            change = C[si, d] - C[si, d - 1]  # SELF-CHECK: uses d and d-1
            gain = max(change, 0.0)
            loss = max(-change, 0.0)

            if np.isnan(avg_gain):
                # First RSI: SMA over rsi_period
                gains = []
                losses = []
                valid = True
                for dd in range(d - rsi_period + 1, d + 1):  # SELF-CHECK: up to d
                    if np.isnan(C[si, dd]) or np.isnan(C[si, dd - 1]) or C[si, dd - 1] <= 0:
                        valid = False
                        break
                    ch = C[si, dd] - C[si, dd - 1]
                    gains.append(max(ch, 0.0))
                    losses.append(max(-ch, 0.0))
                if not valid or len(gains) < rsi_period:
                    continue
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
            else:
                # EMA smoothing
                avg_gain = (avg_gain * (rsi_period - 1) + gain) / rsi_period
                avg_loss = (avg_loss * (rsi_period - 1) + loss) / rsi_period

            if avg_loss == 0:
                RSI[si, di] = 100.0
            else:
                rs = avg_gain / avg_loss
                RSI[si, di] = 100.0 - 100.0 / (1.0 + rs)
            # SELF-CHECK: stored at di, uses data up to d=di-1

    new['RSI'] = RSI
    print(f"  RSI done ({time.time()-t0:.1f}s)", flush=True)

    # === 2. MACD_HIST: MACD histogram ===
    MACD_HIST = np.full((NS, ND), np.nan)
    macd_fast, macd_slow, macd_signal = 12, 26, 9
    for si in range(NS):
        # Compute EMA series up to each di
        for di in range(macd_slow + macd_signal + 2, ND):
            d = di - 1  # SELF-CHECK: d = yesterday
            # Need at least macd_slow + macd_signal prices up to d
            start = max(0, d - macd_slow - macd_signal - 5)
            prices = C[si, start:d + 1]  # SELF-CHECK: up to d
            mask = ~np.isnan(prices)
            if mask.sum() < macd_slow + macd_signal:
                continue
            p = prices[mask]
            if len(p) < macd_slow + macd_signal:
                continue

            fast_ema = _calc_ema(p, macd_fast)
            slow_ema = _calc_ema(p, macd_slow)
            macd_line = fast_ema - slow_ema
            signal_line = _calc_ema(macd_line, macd_signal)
            hist = macd_line[-1] - signal_line[-1]

            if not np.isnan(hist):
                MACD_HIST[si, di] = hist  # SELF-CHECK: stored at di

    new['MACD_HIST'] = MACD_HIST
    print(f"  MACD histogram done ({time.time()-t0:.1f}s)", flush=True)

    # === 3. KER: Kaufman Efficiency Ratio ===
    # KER = |net displacement| / sum(|daily changes|) over period
    # High KER = efficient trending, Low KER = noisy/oscillating
    KER = np.full((NS, ND), np.nan)
    ker_period = 20
    for si in range(NS):
        for di in range(ker_period + 2, ND):
            d = di - 1  # SELF-CHECK: d = yesterday
            # Get prices over period
            prices = C[si, d - ker_period:d + 1]  # SELF-CHECK: up to d
            if np.any(np.isnan(prices)) or len(prices) < ker_period + 1:
                continue

            # Net displacement
            displacement = abs(prices[-1] - prices[0])
            # Total path length
            path = np.sum(np.abs(np.diff(prices)))

            if path > 0:
                KER[si, di] = displacement / path  # SELF-CHECK: stored at di
                # Range [0, 1], higher = more efficient trend

    new['KER'] = KER
    print(f"  KER done ({time.time()-t0:.1f}s)", flush=True)

    # === 4. REL_STR: Relative strength vs market ===
    # Stock's 20-day return minus market average 20-day return
    REL_STR = np.full((NS, ND), np.nan)
    rs_period = 20
    for di in range(rs_period + 2, ND):
        d = di - 1  # SELF-CHECK: d = yesterday
        # Compute each stock's return
        stock_rets = np.full(NS, np.nan)
        for si in range(NS):
            if np.isnan(C[si, d]) or np.isnan(C[si, d - rs_period]) or C[si, d - rs_period] <= 0:
                continue
            stock_rets[si] = (C[si, d] - C[si, d - rs_period]) / C[si, d - rs_period]
            # SELF-CHECK: uses data up to d

        # Market average return
        valid = ~np.isnan(stock_rets)
        if valid.sum() < 100:
            continue
        mkt_ret = np.mean(stock_rets[valid])

        # Relative strength
        for si in range(NS):
            if not np.isnan(stock_rets[si]):
                REL_STR[si, di] = stock_rets[si] - mkt_ret  # SELF-CHECK: stored at di

    new['REL_STR'] = REL_STR
    print(f"  Relative strength done ({time.time()-t0:.1f}s)", flush=True)

    # === Rank normalize ===
    def rank_pct(arr, start=60):
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

    for name in ['RSI', 'MACD_HIST', 'KER', 'REL_STR']:
        new[f'R_{name}'] = rank_pct(new[name])

    print(f"  Ranked done ({time.time()-t0:.1f}s)", flush=True)
    return new


def compute_v11_interactions(all_factors, NS, ND):
    """V11 interactions — classic tech × structural."""
    t0 = time.time()
    new = {}

    # MACD_HIST × BODY_NW — momentum confirmation + candle/NW signal
    macd = all_factors.get('R_MACD_HIST', np.full((NS, ND), np.nan))
    bnw = all_factors.get('R_BODY_NW', np.full((NS, ND), np.nan))
    MACD_BNW = np.full((NS, ND), np.nan)
    mask = ~np.isnan(macd) & ~np.isnan(bnw)
    MACD_BNW[mask] = macd[mask] * bnw[mask] / 100
    new['MACD_BNW'] = MACD_BNW

    # RSI × TENSION — RSI (not overbought) + structural displacement
    rsi = all_factors.get('R_RSI', np.full((NS, ND), np.nan))
    tens = all_factors.get('R_TENSION', np.full((NS, ND), np.nan))
    RSI_TENS = np.full((NS, ND), np.nan)
    mask = ~np.isnan(rsi) & ~np.isnan(tens)
    RSI_TENS[mask] = rsi[mask] * tens[mask] / 100
    new['RSI_TENS'] = RSI_TENS

    # KER × R_SQUARED — efficient + quality trend
    ker = all_factors.get('R_KER', np.full((NS, ND), np.nan))
    r2 = all_factors.get('R_R_SQUARED', np.full((NS, ND), np.nan))
    KER_R2 = np.full((NS, ND), np.nan)
    mask = ~np.isnan(ker) & ~np.isnan(r2)
    KER_R2[mask] = ker[mask] * r2[mask] / 100
    new['KER_R2'] = KER_R2

    # REL_STR × BODY_NW — market leader + conviction
    rels = all_factors.get('R_REL_STR', np.full((NS, ND), np.nan))
    bnw2 = all_factors.get('R_BODY_NW', np.full((NS, ND), np.nan))
    REL_BNW = np.full((NS, ND), np.nan)
    mask = ~np.isnan(rels) & ~np.isnan(bnw2)
    REL_BNW[mask] = rels[mask] * bnw2[mask] / 100
    new['REL_BNW'] = REL_BNW

    # MACD_HIST × BB_WIDTH_PCT_INV — momentum + extreme contraction
    bwp = all_factors.get('R_BB_WIDTH_PCT_INV', np.full((NS, ND), np.nan))
    MACD_BWP = np.full((NS, ND), np.nan)
    mask = ~np.isnan(macd) & ~np.isnan(bwp)
    MACD_BWP[mask] = macd[mask] * bwp[mask] / 100
    new['MACD_BWP'] = MACD_BWP

    # KER × BB_SQUEEZE_INV — efficient trend in squeeze
    sqz = all_factors.get('R_BB_SQUEEZE_INV', np.full((NS, ND), np.nan))
    KER_SQZ = np.full((NS, ND), np.nan)
    mask = ~np.isnan(ker) & ~np.isnan(sqz)
    KER_SQZ[mask] = ker[mask] * sqz[mask] / 100
    new['KER_SQZ'] = KER_SQZ

    # REL_STR × R_SQUARED — market leader + quality trend
    r2f = all_factors.get('R_R_SQUARED', np.full((NS, ND), np.nan))
    REL_R2 = np.full((NS, ND), np.nan)
    mask = ~np.isnan(rels) & ~np.isnan(r2f)
    REL_R2[mask] = rels[mask] * r2f[mask] / 100
    new['REL_R2'] = REL_R2

    # RSI × MACD_HIST — classic momentum agreement
    RSI_MACD = np.full((NS, ND), np.nan)
    mask = ~np.isnan(rsi) & ~np.isnan(macd)
    RSI_MACD[mask] = rsi[mask] * macd[mask] / 100
    new['RSI_MACD'] = RSI_MACD

    # Rank normalize
    def rank_pct(arr, start=60):
        res = np.full_like(arr, np.nan)
        for di in range(start, arr.shape[1]):
            vals = arr[:, di]
            m = ~np.isnan(vals)
            if m.sum() < 50:
                continue
            ranked = np.argsort(np.argsort(vals[m])).astype(float)
            n = len(ranked)
            pct = ranked / max(n - 1, 1) * 100
            for k, idx in enumerate(np.where(m)[0]):
                res[idx, di] = pct[k]
        return res

    for name in list(new.keys()):
        new[f'R_{name}'] = rank_pct(new[name])

    print(f"  V11 interactions done ({time.time()-t0:.1f}s)", flush=True)
    return new


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V11 — Classic Technical + Market Relative", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()
    base_factors = compute_all_factors(NS, ND, C, O, H, L, V)
    inter_factors = compute_interaction_factors(base_factors, NS, ND, C, O, H, L, V)
    extra_factors = compute_extra_factors(NS, ND, C, O, H, L, V)
    v7e_factors = compute_v7e_factors(NS, ND, C, O, H, L, V)
    adv_inter = compute_advanced_interactions(
        {**base_factors, **inter_factors, **extra_factors, **v7e_factors}, NS, ND)
    v8_factors = compute_v8_factors(NS, ND, C, O, H, L, V)
    v8_all = {**base_factors, **inter_factors, **extra_factors,
              **v7e_factors, **adv_inter, **v8_factors}
    v8_inter = compute_v8_interactions(v8_all, NS, ND)
    v8_all.update(v8_inter)
    v9_factors = compute_v9_factors(NS, ND, C, O, H, L, V)
    v9_all = {**v8_all, **v9_factors}
    v9_inter = compute_v9_interactions(v9_all, NS, ND)
    v9_all.update(v9_inter)
    v10_factors = compute_v10_factors(NS, ND, C, O, H, L, V)
    v10_all = {**v9_all, **v10_factors}
    v10_inter = compute_v10_interactions(v10_all, NS, ND)
    v10_all.update(v10_inter)

    v11_factors = compute_v11_factors(NS, ND, C, O, H, L, V)
    all_factors = {**v10_all, **v11_factors}
    v11_inter = compute_v11_interactions(all_factors, NS, ND)
    all_factors.update(v11_inter)

    print(f"\n  Total factors: {len(all_factors)}", flush=True)

    # SINGLE FACTOR TESTS
    print(f"\n  === SINGLE FACTOR TESTS (V11 new) ===", flush=True)
    for fname in ['R_RSI', 'R_MACD_HIST', 'R_KER', 'R_REL_STR']:
        r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=3, rebalance_days=10, atr_stop_mult=1.5)
        if r:
            print(f"  {fname}: Ann={r['ann']:+.1f}% WR={r['wr']:.0f}% DD={r['max_dd']:.1f}%", flush=True)

    # PORTFOLIO TESTS
    portfolios = {
        # V10 references
        'BodyNW': {'R_BODY_NW': 0.3, 'R_TENSION': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        'BwpBNW': {'R_BWP_BNW': 0.3, 'R_TENSION': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        'SdBNWx2': {'R_SD_BNW': 0.25, 'R_BB_SQUEEZE_INV': 0.15,
                    'R_TENSION': 0.25, 'R_R_SQUARED': 0.15, 'R_SMA_DEV': 0.2},
        # V11: MACD × BodyNW
        'MacdBNW': {'R_MACD_BNW': 0.3, 'R_TENSION': 0.3,
                    'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # V11: RSI × Tension
        'RsiTens': {'R_RSI_TENS': 0.3, 'R_BODY_NW': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # V11: KER × R²
        'KerR2': {'R_KER_R2': 0.3, 'R_BODY_NW': 0.3,
                  'R_TENSION': 0.2, 'R_SMA_DEV': 0.2},
        # V11: Relative strength × BodyNW
        'RelBNW': {'R_REL_BNW': 0.3, 'R_TENSION': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # V11: MACD × BB squeeze
        'MacdBwp': {'R_MACD_BWP': 0.3, 'R_BODY_NW': 0.3,
                    'R_TENSION': 0.2, 'R_SMA_DEV': 0.2},
        # V11: KER × squeeze
        'KerSqz': {'R_KER_SQZ': 0.3, 'R_BODY_NW': 0.3,
                   'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        # V11: Relative strength × R²
        'RelR2': {'R_REL_R2': 0.3, 'R_BODY_NW': 0.3,
                  'R_TENSION': 0.2, 'R_SMA_DEV': 0.2},
        # V11: RSI × MACD classic combo
        'RsiMacd': {'R_RSI_MACD': 0.3, 'R_BODY_NW': 0.3,
                    'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        # V11: BwpBNW + MACD confirmation
        'BwpMacd': {'R_BWP_BNW': 0.25, 'R_MACD_HIST': 0.15,
                    'R_TENSION': 0.25, 'R_R_SQUARED': 0.15, 'R_SMA_DEV': 0.2},
        # V11: Full classic tech + structural
        'TechStr': {'R_MACD_HIST': 0.1, 'R_RSI': 0.1, 'R_KER': 0.1,
                    'R_BODY_NW': 0.25, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.15,
                    'R_SMA_DEV': 0.1},
        # V11: Market leader + squeeze (best of V9+V10+V11)
        'LdrSqz': {'R_REL_STR': 0.15, 'R_BB_WIDTH_PCT_INV': 0.15,
                   'R_BODY_NW': 0.25, 'R_TENSION': 0.25, 'R_R_SQUARED': 0.2},
        # V11: KER + squeeze depth (efficiency × contraction)
        'KerSd': {'R_KER': 0.1, 'R_SQZ_DEPTH': 0.15,
                  'R_BODY_NW': 0.3, 'R_TENSION': 0.25, 'R_R_SQUARED': 0.2},
    }

    results = []
    for pname, weights in portfolios.items():
        for top_n in [1]:
            for rebal in [7, 10]:
                for atr in [1.0, 1.2, 1.5]:
                    r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                    top_n=top_n, rebalance_days=rebal, atr_stop_mult=atr)
                    if r:
                        r.update({'portfolio': pname, 'top_n': top_n,
                                  'rebal': rebal, 'atr': atr})
                        results.append(r)
        print(f"  {pname} done", flush=True)

    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  TOP 30 (V11 BUG-FIXED)", flush=True)
    print(f"  {'Portfolio':<15s} {'Top':>3s} {'Reb':>3s} {'ATR':>3s} | "
          f"{'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*95}", flush=True)
    for r in results[:30]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['portfolio']:<15s} {r['top_n']:3d} {r['rebal']:3d} {r['atr']:3.1f} | "
              f"{r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    # Best per portfolio
    best_per = {}
    for r in results:
        p = r['portfolio']
        if p not in best_per or r['ann'] > best_per[p]['ann']:
            best_per[p] = r
    print(f"\n  Best per portfolio:", flush=True)
    for r in sorted(best_per.values(), key=lambda x: -x['ann']):
        pos = " ALL+" if all_positive(r) else ""
        print(f"    {r['portfolio']:<15s} Top={r['top_n']} Reb={r['rebal']} ATR={r['atr']:.1f} → "
              f"{r['ann']:+.1f}%DD={r['max_dd']:.1f}%{pos}", flush=True)

    # Top 3 year-by-year
    for i, r in enumerate(results[:3]):
        print(f"\n  Year-by-year #{i+1}: {r['portfolio']} Top={r['top_n']} Reb={r['rebal']} "
              f"ATR={r['atr']:.1f} (Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
