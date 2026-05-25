"""
Alpha V9 — Fractal Dimension + Volatility Squeeze Factors
==========================================================
V8 best: BodyNW Top=1 = +185.9% DD=56.8%

New factors (from Hurst regime, Squeeze momentum, Supply/Demand strategies):
  1. HURST: R/S analysis Hurst exponent (trending vs mean-reverting)
  2. KFD: Katz Fractal Dimension (path complexity)
  3. BB_SQUEEZE: BB width / KC width ratio (volatility contraction energy)
  4. SUPPLY_DIST: Distance to nearest demand zone (support proximity)

LOOK-AHEAD SELF-CHECK:
  [x] All factors use ONLY data up to di-1 (yesterday's close)
  [x] Results stored at index di, read by backtest at di
  [x] No same-day data used for any computation
  [x] ATR stop uses L[si,di] check + stop price sell (bug-fixed)
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
from alpha_v7c import backtest_v7c  # BUG-FIXED version


def compute_v9_factors(NS, ND, C, O, H, L, V):
    """V9 factors — STRICT no look-ahead.

    SELF-CHECK RULES:
    1. d = di - 1 (use yesterday as "current" data)
    2. Store result at index di (backtest reads at di)
    3. Never access C[si, di], O[si, di], H[si, di], L[si, di], V[si, di]
    """
    t0 = time.time()
    new = {}

    # === 1. HURST: R/S analysis Hurst exponent ===
    # H > 0.5 = trending (persistent), H < 0.5 = mean-reverting
    # Use last 100 days of log returns, R/S over multiple sub-interval sizes
    HURST = np.full((NS, ND), np.nan)
    hurst_window = 100  # Use 100 days of returns
    for si in range(NS):
        for di in range(hurst_window + 2, ND):
            d = di - 1  # SELF-CHECK: d = yesterday
            # Get prices up to d (inclusive)
            prices = C[si, d - hurst_window:d + 1]  # SELF-CHECK: up to d only
            if np.any(np.isnan(prices)) or np.any(prices <= 0):
                continue
            returns = np.diff(np.log(prices))
            n_total = len(returns)

            # R/S analysis over multiple sub-interval sizes
            min_n, max_n = 10, n_total // 2
            if min_n >= max_n:
                continue

            num_splits = min(8, max_n - min_n + 1)
            split_sizes = np.unique(np.linspace(min_n, max_n, num_splits).astype(int))

            ns, rs_vals = [], []
            for n in split_sizes:
                num_sub = n_total // n
                if num_sub < 1:
                    continue
                rs_list = []
                for i in range(num_sub):
                    sub = returns[i * n:(i + 1) * n]
                    mean_sub = np.mean(sub)
                    cumdev = np.cumsum(sub - mean_sub)
                    r = np.max(cumdev) - np.min(cumdev)
                    s = np.std(sub, ddof=1)
                    if s > 0:
                        rs_list.append(r / s)
                if rs_list:
                    ns.append(np.log(n))
                    rs_vals.append(np.log(np.mean(rs_list)))

            if len(ns) >= 2:
                coeffs = np.polyfit(ns, rs_vals, 1)
                HURST[si, di] = float(np.clip(coeffs[0], 0.0, 1.0))
                # SELF-CHECK: stored at di, uses data up to d=di-1

    new['HURST'] = HURST
    print(f"  Hurst done ({time.time()-t0:.1f}s)", flush=True)

    # === 2. KFD: Katz Fractal Dimension ===
    # KFD = log(n) / (log(n) + log(d/L))
    # n = steps, d = straight-line distance (first to last), L = total path length
    # KFD near 1 = oscillating, near 0 = trending
    KFD = np.full((NS, ND), np.nan)
    kfd_window = 50
    for si in range(NS):
        for di in range(kfd_window + 1, ND):
            d = di - 1  # SELF-CHECK: d = yesterday
            prices = C[si, d - kfd_window:d + 1]  # SELF-CHECK: up to d only
            if np.any(np.isnan(prices)):
                continue

            n = len(prices) - 1  # steps
            if n < 2:
                continue

            # Path total length L
            diffs = np.abs(np.diff(prices))
            L_path = np.sum(diffs)

            # Straight-line distance d
            d_dist = np.abs(prices[-1] - prices[0])

            if L_path < 1e-10 or d_dist < 1e-10:
                KFD[si, di] = 1.0  # No change = extreme oscillation
                continue

            log_n = np.log(n)
            kfd = log_n / (log_n + np.log(d_dist / L_path))
            KFD[si, di] = kfd  # SELF-CHECK: stored at di

    new['KFD'] = KFD
    print(f"  KFD done ({time.time()-t0:.1f}s)", flush=True)

    # === 3. BB_SQUEEZE: Bollinger Band width / Keltner Channel width ===
    # BB width = 2 * bb_mult * std(close, period) / sma(close, period)
    # KC width = 2 * kc_mult * ATR(period) / sma(close, period)
    # Ratio < 1 = squeeze on (BB inside KC) = energy building
    # We store the ratio (lower = more squeezed = more potential)
    # For ranking: LOWER ratio = better (squeeze), so we invert for R_ ranking
    BB_SQUEEZE = np.full((NS, ND), np.nan)
    bb_period = 20
    bb_mult = 2.0
    kc_mult = 1.5
    atr_period = 14

    for si in range(NS):
        for di in range(bb_period + atr_period + 1, ND):
            d = di - 1  # SELF-CHECK: d = yesterday

            # Get close prices up to d for BB
            closes = C[si, d - bb_period + 1:d + 1]  # SELF-CHECK: up to d
            if np.any(np.isnan(closes)) or len(closes) < bb_period:
                continue

            bb_mid = np.mean(closes)
            bb_std = np.std(closes, ddof=1)
            if bb_mid <= 0 or bb_std <= 0:
                continue

            bb_width = 2.0 * bb_mult * bb_std  # Not normalized (both use same mid)

            # ATR for KC (using data up to d)
            atr = 0.0
            atr_count = 0
            for dd in range(max(d - atr_period + 1, 1), d + 1):  # SELF-CHECK: up to d
                if np.isnan(H[si, dd]) or np.isnan(L[si, dd]) or np.isnan(C[si, dd]):
                    continue
                tr1 = H[si, dd] - L[si, dd]
                tr2 = abs(H[si, dd] - C[si, dd - 1]) if dd > 0 and not np.isnan(C[si, dd - 1]) else 0
                tr3 = abs(L[si, dd] - C[si, dd - 1]) if dd > 0 and not np.isnan(C[si, dd - 1]) else 0
                atr += max(tr1, tr2, tr3)
                atr_count += 1

            if atr_count < atr_period // 2:
                continue
            atr /= atr_count

            kc_width = 2.0 * kc_mult * atr

            if kc_width > 0:
                BB_SQUEEZE[si, di] = bb_width / kc_width  # SELF-CHECK: stored at di
                # < 1.0 = squeeze on (BB inside KC)

    new['BB_SQUEEZE'] = BB_SQUEEZE
    print(f"  BB Squeeze done ({time.time()-t0:.1f}s)", flush=True)

    # === 4. SUPPLY_DIST: Distance to nearest demand zone ===
    # Simplified: find the highest volume node in recent price action (demand zone)
    # Distance from current price to that zone (normalized by ATR)
    # Closer to demand zone = stronger support = better entry
    SUPPLY_DIST = np.full((NS, ND), np.nan)
    lookback = 30  # Look back 30 days for demand zone

    for si in range(NS):
        for di in range(lookback + atr_period + 1, ND):
            d = di - 1  # SELF-CHECK: d = yesterday
            cur_price = C[si, d]  # SELF-CHECK: yesterday's close
            if np.isnan(cur_price) or cur_price <= 0:
                continue

            # Find demand zone: lowest close in recent window with high volume
            best_demand = np.nan
            best_vol_score = -1

            for dd in range(max(d - lookback + 1, 0), d + 1):  # SELF-CHECK: up to d
                if np.isnan(C[si, dd]) or np.isnan(V[si, dd]):
                    continue
                # Demand zone = low price with high volume
                # Score inversely proportional to price, proportional to volume
                if C[si, dd] < cur_price:  # Only consider levels below current price
                    vol_score = V[si, dd] * (cur_price - C[si, dd]) / cur_price
                    if vol_score > best_vol_score:
                        best_vol_score = vol_score
                        best_demand = C[si, dd]

            if np.isnan(best_demand):
                continue

            # Compute ATR for normalization
            atr = 0.0
            atr_count = 0
            for dd in range(max(d - atr_period + 1, 1), d + 1):
                if np.isnan(H[si, dd]) or np.isnan(L[si, dd]) or np.isnan(C[si, dd]):
                    continue
                tr1 = H[si, dd] - L[si, dd]
                tr2 = abs(H[si, dd] - C[si, dd - 1]) if dd > 0 and not np.isnan(C[si, dd - 1]) else 0
                tr3 = abs(L[si, dd] - C[si, dd - 1]) if dd > 0 and not np.isnan(C[si, dd - 1]) else 0
                atr += max(tr1, tr2, tr3)
                atr_count += 1

            if atr_count < atr_period // 2 or atr <= 0:
                continue
            atr /= atr_count

            # Distance to demand zone, normalized by ATR
            dist = (cur_price - best_demand) / atr
            SUPPLY_DIST[si, di] = dist  # SELF-CHECK: stored at di
            # Lower = closer to support = better

    new['SUPPLY_DIST'] = SUPPLY_DIST
    print(f"  Supply/Demand dist done ({time.time()-t0:.1f}s)", flush=True)

    # === Rank normalize all new factors ===
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

    for name in ['HURST', 'KFD', 'BB_SQUEEZE', 'SUPPLY_DIST']:
        new[f'R_{name}'] = rank_pct(new[name])

    # Invert BB_SQUEEZE ranking: lower ratio (more squeezed) should rank HIGHER
    # Since rank_pct ranks low→0, high→100, and we want squeeze (low) = good:
    # R_BB_SQUEEZE already ranks low values low, so stocks in squeeze rank low
    # We want the OPPOSITE: stocks in squeeze should rank HIGH
    # Solution: invert the ranking
    inv = new['R_BB_SQUEEZE'].copy()
    mask = ~np.isnan(inv)
    inv[mask] = 100.0 - inv[mask]
    new['R_BB_SQUEEZE_INV'] = inv  # Higher = more squeezed = better

    # Invert SUPPLY_DIST: lower distance = better support = rank higher
    inv_sd = new['R_SUPPLY_DIST'].copy()
    mask = ~np.isnan(inv_sd)
    inv_sd[mask] = 100.0 - inv_sd[mask]
    new['R_SUPPLY_DIST_INV'] = inv_sd  # Higher = closer to demand zone = better

    print(f"  Ranked done ({time.time()-t0:.1f}s)", flush=True)

    return new


def compute_v9_interactions(all_factors, NS, ND):
    """V9 interaction factors — fractal × structure, squeeze × momentum."""
    t0 = time.time()
    new = {}

    # HURST × BODY_NW — trending market + strong candle/NW signal
    hurst = all_factors.get('R_HURST', np.full((NS, ND), np.nan))
    bnw = all_factors.get('R_BODY_NW', np.full((NS, ND), np.nan))
    HURST_BNW = np.full((NS, ND), np.nan)
    mask = ~np.isnan(hurst) & ~np.isnan(bnw)
    HURST_BNW[mask] = hurst[mask] * bnw[mask] / 100
    new['HURST_BNW'] = HURST_BNW

    # KFD × TENSION — low fractal dim (trending) × high tension (displacement)
    # KFD near 0 = trending. Invert so high = trending.
    kfd = all_factors.get('R_KFD', np.full((NS, ND), np.nan))
    tens = all_factors.get('R_TENSION', np.full((NS, ND), np.nan))
    # Invert KFD ranking: low KFD = trending = rank high
    kfd_inv = kfd.copy()
    m = ~np.isnan(kfd_inv)
    kfd_inv[m] = 100.0 - kfd_inv[m]
    KFD_TENS = np.full((NS, ND), np.nan)
    mask = ~np.isnan(kfd_inv) & ~np.isnan(tens)
    KFD_TENS[mask] = kfd_inv[mask] * tens[mask] / 100
    new['KFD_TENS'] = KFD_TENS

    # BB_SQUEEZE_INV × R_SQUARED — squeeze + clean trend
    sqz = all_factors.get('R_BB_SQUEEZE_INV', np.full((NS, ND), np.nan))
    r2 = all_factors.get('R_R_SQUARED', np.full((NS, ND), np.nan))
    SQZ_R2 = np.full((NS, ND), np.nan)
    mask = ~np.isnan(sqz) & ~np.isnan(r2)
    SQZ_R2[mask] = sqz[mask] * r2[mask] / 100
    new['SQZ_R2'] = SQZ_R2

    # HURST × R_SQUARED — persistent + quality trend
    r2f = all_factors.get('R_R_SQUARED', np.full((NS, ND), np.nan))
    HURST_R2 = np.full((NS, ND), np.nan)
    mask = ~np.isnan(hurst) & ~np.isnan(r2f)
    HURST_R2[mask] = hurst[mask] * r2f[mask] / 100
    new['HURST_R2'] = HURST_R2

    # SUPPLY_DIST_INV × BODY_NW — near demand zone + conviction
    sdi = all_factors.get('R_SUPPLY_DIST_INV', np.full((NS, ND), np.nan))
    bnw2 = all_factors.get('R_BODY_NW', np.full((NS, ND), np.nan))
    SDI_BNW = np.full((NS, ND), np.nan)
    mask = ~np.isnan(sdi) & ~np.isnan(bnw2)
    SDI_BNW[mask] = sdi[mask] * bnw2[mask] / 100
    new['SDI_BNW'] = SDI_BNW

    # BB_SQUEEZE_INV × BODY_NW — squeeze energy + candle/NW
    sqz2 = all_factors.get('R_BB_SQUEEZE_INV', np.full((NS, ND), np.nan))
    bnw3 = all_factors.get('R_BODY_NW', np.full((NS, ND), np.nan))
    SQZ_BNW = np.full((NS, ND), np.nan)
    mask = ~np.isnan(sqz2) & ~np.isnan(bnw3)
    SQZ_BNW[mask] = sqz2[mask] * bnw3[mask] / 100
    new['SQZ_BNW'] = SQZ_BNW

    # Rank normalize all interactions
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

    print(f"  V9 interactions done ({time.time()-t0:.1f}s)", flush=True)
    return new


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V9 — Fractal Dimension + Volatility Squeeze", flush=True)
    print("=" * 70, flush=True)

    # Compute all factors
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
    all_factors = {**v8_all, **v9_factors}

    v9_inter = compute_v9_interactions(all_factors, NS, ND)
    all_factors.update(v9_inter)

    print(f"\n  Total factors: {len(all_factors)}", flush=True)

    # SINGLE FACTOR TESTS
    print(f"\n  === SINGLE FACTOR TESTS (V9 new) ===", flush=True)
    for fname in ['R_HURST', 'R_KFD', 'R_BB_SQUEEZE', 'R_BB_SQUEEZE_INV',
                  'R_SUPPLY_DIST', 'R_SUPPLY_DIST_INV']:
        r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=3, rebalance_days=10, atr_stop_mult=1.5)
        if r:
            print(f"  {fname}: Ann={r['ann']:+.1f}% WR={r['wr']:.0f}% DD={r['max_dd']:.1f}%", flush=True)

    # PORTFOLIO TESTS
    portfolios = {
        # V8 best (reference)
        'BodyNW': {'R_BODY_NW': 0.3, 'R_TENSION': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # Hurst + BodyNW
        'HurstBNW': {'R_HURST': 0.2, 'R_BODY_NW': 0.3,
                     'R_TENSION': 0.3, 'R_R_SQUARED': 0.2},
        # Hurst × BodyNW interaction
        'HurstBNWx': {'R_HURST_BNW': 0.3, 'R_TENSION': 0.3,
                      'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # BB Squeeze + BodyNW
        'SqzBNW': {'R_BB_SQUEEZE_INV': 0.2, 'R_BODY_NW': 0.3,
                   'R_TENSION': 0.3, 'R_R_SQUARED': 0.2},
        # Squeeze × BodyNW interaction
        'SqzBNWx': {'R_SQZ_BNW': 0.3, 'R_TENSION': 0.3,
                    'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # Squeeze × R²
        'SqzR2': {'R_SQZ_R2': 0.3, 'R_BODY_NW': 0.3,
                  'R_TENSION': 0.2, 'R_SMA_DEV': 0.2},
        # KFD inverted × Tension
        'KFDTens': {'R_KFD_TENS': 0.3, 'R_BODY_NW': 0.3,
                    'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # Supply dist + BodyNW
        'SupBNW': {'R_SUPPLY_DIST_INV': 0.2, 'R_BODY_NW': 0.3,
                   'R_TENSION': 0.3, 'R_R_SQUARED': 0.2},
        # Supply dist × BodyNW interaction
        'SupBNWx': {'R_SDI_BNW': 0.3, 'R_TENSION': 0.3,
                    'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # Hurst × R² interaction
        'HurstR2': {'R_HURST_R2': 0.3, 'R_BODY_NW': 0.3,
                    'R_TENSION': 0.2, 'R_SMA_DEV': 0.2},
        # Full fractal: Hurst + KFD_inv + Squeeze
        'Fractal': {'R_HURST': 0.15, 'R_KFD_TENS': 0.2, 'R_SQZ_BNW': 0.25,
                    'R_BODY_NW': 0.25, 'R_R_SQUARED': 0.15},
        # Kitchen sink V9
        'V9Full': {'R_HURST': 0.1, 'R_BB_SQUEEZE_INV': 0.1, 'R_SUPPLY_DIST_INV': 0.1,
                   'R_BODY_NW': 0.25, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.15,
                   'R_SMA_DEV': 0.1},
        # LowEnt + Hurst (V8 best + V9)
        'LowEntH': {'R_ENTROPY': 0.15, 'R_HURST': 0.15, 'R_BODY_NW': 0.25,
                    'R_TENSION': 0.25, 'R_R_SQUARED': 0.2},
        # Entropy + Squeeze (double filter)
        'EntSqz': {'R_ENTROPY': 0.1, 'R_BB_SQUEEZE_INV': 0.15,
                   'R_BODY_NW': 0.3, 'R_TENSION': 0.25, 'R_R_SQUARED': 0.2},
    }

    results = []
    for pname, weights in portfolios.items():
        for top_n in [1, 2]:
            for rebal in [10]:
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

    print(f"\n{'='*110}", flush=True)
    print(f"  TOP 30 (V9 BUG-FIXED)", flush=True)
    print(f"  {'Portfolio':<15s} {'Top':>3s} {'Reb':>3s} {'ATR':>3s} | "
          f"{'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*90}", flush=True)
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

    # V8 reference comparison
    v8_ref = [r for r in results if r['portfolio'] == 'BodyNW']
    if v8_ref:
        v8_best = max(v8_ref, key=lambda x: x['ann'])
        print(f"\n  V8 reference: BodyNW Top={v8_best['top_n']} ATR={v8_best['atr']:.1f} = "
              f"{v8_best['ann']:+.1f}%", flush=True)

    v9_bests = [r for r in results if r['portfolio'] != 'BodyNW']
    if v9_bests:
        v9_best = max(v9_bests, key=lambda x: x['ann'])
        delta = v9_best['ann'] - v8_best['ann'] if v8_ref else 0
        print(f"  V9 best: {v9_best['portfolio']} Top={v9_best['top_n']} ATR={v9_best['atr']:.1f} = "
              f"{v9_best['ann']:+.1f}% (Δ={delta:+.1f}%)", flush=True)

    print(f"\n{'='*70}", flush=True)
