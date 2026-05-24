"""
Alpha V7e — Deeper Factor Exploration
=====================================
V7d结果: SMA_DEV有alpha(+68.9%独因子), 但增加DD
StructQual仍是最佳: +161.9% / DD=51.5%

新方向:
  1. SMA_DEV轻量组合 — 只加少量SMA_DEV看是否能降DD
  2. 从更多策略提取新因子:
     - Nadaraya-Watson 趋势斜率 (epanechnikov_confluence)
     - 能量因子改进 (energt_structure)
     - Wyckoff累积/分配比率
  3. 交互因子: SMA_DEV × TENSION, SMA_DEV × R²
  4. Regime filter + SMA_DEV 组合
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7 import compute_all_factors, COMMISSION, STAMP_DUTY, CASH0
from alpha_v7d import compute_extra_factors


def compute_v7e_factors(NS, ND, C, O, H, L, V):
    """New factors for V7e."""
    t0 = time.time()
    new = {}

    # === NW_SLOPE: Nadaraya-Watson trend slope ===
    # Smooth price using Epanechnikov kernel, then compute slope
    # Captures smooth trend direction without lag
    NW_SLOPE = np.full((NS, ND), np.nan)
    bandwidth = 10  # kernel bandwidth

    for si in range(NS):
        prices = C[si, :].copy()
        for di in range(bandwidth + 5, ND):
            d = di - 1  # use yesterday's data
            # NW regression: weighted average of last 2*bandwidth points
            weights_sum = 0
            val_sum = 0
            count = 0
            for dd in range(max(0, d - 2 * bandwidth), d + 1):
                if np.isnan(prices[dd]):
                    continue
                u = (d - dd) / bandwidth
                if abs(u) >= 1:
                    continue
                w = 0.75 * (1 - u * u)  # Epanechnikov kernel
                val_sum += w * prices[dd]
                weights_sum += w
                count += 1

            if weights_sum < 0.1 or count < 5:
                continue

            nw_now = val_sum / weights_sum

            # NW regression at d-5 for slope
            d_prev = d - 5
            if d_prev < bandwidth:
                continue
            weights_sum2 = 0
            val_sum2 = 0
            for dd in range(max(0, d_prev - 2 * bandwidth), d_prev + 1):
                if np.isnan(prices[dd]):
                    continue
                u = (d_prev - dd) / bandwidth
                if abs(u) >= 1:
                    continue
                w = 0.75 * (1 - u * u)
                val_sum2 += w * prices[dd]
                weights_sum2 += w

            if weights_sum2 < 0.1:
                continue

            nw_prev = val_sum2 / weights_sum2

            if nw_prev > 0:
                NW_SLOPE[si, di] = (nw_now - nw_prev) / nw_prev * 100  # 5-day slope %

        if si % 100 == 0 and si > 0:
            print(f"    NW_SLOPE stock {si}/{NS} ({time.time()-t0:.1f}s)", flush=True)

    new['NW_SLOPE'] = NW_SLOPE
    print(f"  NW slope done ({time.time()-t0:.1f}s)", flush=True)

    # === KINETIC_EMA: Exponential-weighted kinetic energy ===
    # EMA of (log_return * volume) — more responsive than simple kinetic
    KINETIC_EMA = np.full((NS, ND), np.nan)
    for si in range(NS):
        ke = np.full(ND, np.nan)
        ema_val = 0
        ema_started = False
        alpha = 2 / 11  # 10-day EMA

        for di in range(1, ND):
            d = di - 1
            if np.isnan(C[si, d]) or np.isnan(C[si, d - 1]) or C[si, d - 1] <= 0:
                continue
            if np.isnan(V[si, d]) or V[si, d] <= 0:
                continue

            log_ret = np.log(C[si, d] / C[si, d - 1])
            ke_val = log_ret * V[si, d]

            if not ema_started:
                ema_val = ke_val
                ema_started = True
            else:
                ema_val = alpha * ke_val + (1 - alpha) * ema_val

            ke[di] = ema_val

        KINETIC_EMA[si, :] = ke

    new['KINETIC_EMA'] = KINETIC_EMA
    print(f"  Kinetic EMA done ({time.time()-t0:.1f}s)", flush=True)

    # === VOL_ACCEL: Volume acceleration (2nd derivative of volume) ===
    # Increasing volume acceleration = force building
    VOL_ACCEL = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(22, ND):
            d = di - 1
            vols = V[si, d - 19:d + 1]
            valid = vols[~np.isnan(vols)]
            if len(valid) < 15:
                continue
            # First half avg vs second half avg
            half = len(valid) // 2
            v1 = np.mean(valid[:half])
            v2 = np.mean(valid[half:])
            if v1 > 0:
                VOL_ACCEL[si, di] = (v2 - v1) / v1 * 100

    new['VOL_ACCEL'] = VOL_ACCEL
    print(f"  Vol accel done ({time.time()-t0:.1f}s)", flush=True)

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

    for name in ['NW_SLOPE', 'KINETIC_EMA', 'VOL_ACCEL']:
        new[f'R_{name}'] = rank_pct(new[name])
    print(f"  Ranked done ({time.time()-t0:.1f}s)", flush=True)

    return new


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V7e — Deeper Factor Exploration", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()
    base_factors = compute_all_factors(NS, ND, C, O, H, L, V)
    from alpha_v7b import compute_interaction_factors
    inter_factors = compute_interaction_factors(base_factors, NS, ND, C, O, H, L, V)
    extra_factors = compute_extra_factors(NS, ND, C, O, H, L, V)
    v7e_factors = compute_v7e_factors(NS, ND, C, O, H, L, V)

    all_factors = {**base_factors, **inter_factors, **extra_factors, **v7e_factors}
    print(f"\n  Total factors: {len(all_factors)}", flush=True)

    from alpha_v7c import backtest_v7c

    # Single factor tests
    print(f"\n  === SINGLE FACTOR TESTS ===", flush=True)
    for fname in ['R_SMA_DEV', 'R_NW_SLOPE', 'R_KINETIC_EMA', 'R_VOL_ACCEL']:
        r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=3, rebalance_days=10, atr_stop_mult=1.5)
        if r:
            print(f"  {fname}: Ann={r['ann']:+.1f}% WR={r['wr']:.0f}% DD={r['max_dd']:.1f}%", flush=True)

    # Combinations — focus on SMA_DEV lightweight + new factors
    portfolios = {
        # Baseline
        'StructQual': {'R_TENSION': 0.25, 'R_R_SQUARED': 0.25,
                       'R_TENS_SHAD': 0.25, 'R_BODY_VOL': 0.25},
        # SMA_DEV lightweight — 10% weight only
        'SQ_SMA10': {'R_TENSION': 0.225, 'R_R_SQUARED': 0.225,
                     'R_TENS_SHAD': 0.225, 'R_BODY_VOL': 0.225,
                     'R_SMA_DEV': 0.1},
        # NW slope added
        'SQ_NW': {'R_TENSION': 0.2, 'R_R_SQUARED': 0.2,
                  'R_TENS_SHAD': 0.2, 'R_BODY_VOL': 0.2,
                  'R_NW_SLOPE': 0.2},
        # Kinetic EMA added
        'SQ_KE': {'R_TENSION': 0.2, 'R_R_SQUARED': 0.2,
                  'R_TENS_SHAD': 0.2, 'R_BODY_VOL': 0.2,
                  'R_KINETIC_EMA': 0.2},
        # Vol acceleration
        'SQ_VA': {'R_TENSION': 0.2, 'R_R_SQUARED': 0.2,
                  'R_TENS_SHAD': 0.2, 'R_BODY_VOL': 0.2,
                  'R_VOL_ACCEL': 0.2},
        # All new factors combined
        'SQ_AllNew': {'R_TENSION': 0.15, 'R_R_SQUARED': 0.15,
                      'R_TENS_SHAD': 0.15, 'R_BODY_VOL': 0.15,
                      'R_SMA_DEV': 0.1, 'R_NW_SLOPE': 0.1,
                      'R_KINETIC_EMA': 0.1, 'R_VOL_ACCEL': 0.1},
        # SMA_DEV + TENSION interaction (heavy on structure + mean reversion)
        'StructMR': {'R_SMA_DEV': 0.3, 'R_TENSION': 0.3,
                     'R_R_SQUARED': 0.2, 'R_NW_SLOPE': 0.2},
        # Pure new factors
        'NewPure': {'R_SMA_DEV': 0.3, 'R_NW_SLOPE': 0.3,
                    'R_KINETIC_EMA': 0.2, 'R_VOL_ACCEL': 0.2},
        # NW + Kinetic + Tension (smooth trend + energy + structure)
        'TrendEnergy': {'R_NW_SLOPE': 0.3, 'R_KINETIC_EMA': 0.3,
                        'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        # Fisher + SMA_DEV (non-normal transform + mean reversion)
        'FisherMR': {'R_FISHER': 0.3, 'R_SMA_DEV': 0.3,
                     'R_TENSION': 0.2, 'R_VOL_ACCEL': 0.2},
    }

    results = []
    for pname, weights in portfolios.items():
        for top_n in [3, 5]:
            for rebal in [10, 15]:
                for atr in [1.5, 2.0]:
                    r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                    top_n=top_n, rebalance_days=rebal, atr_stop_mult=atr)
                    if r:
                        r.update({'portfolio': pname, 'top_n': top_n,
                                  'rebal': rebal, 'atr': atr})
                        results.append(r)
        print(f"  {pname} done", flush=True)

    results.sort(key=lambda x: -x['ann'])

    print(f"\n{'='*110}", flush=True)
    print(f"  TOP 30 RESULTS", flush=True)
    print(f"  {'Portfolio':<15s} {'Top':>3s} {'Reb':>3s} {'ATR':>3s} | "
          f"{'Ann':>7s} {'N':>5s} {'TPY':>4s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*100}", flush=True)
    for r in results[:30]:
        print(f"  {r['portfolio']:<15s} {r['top_n']:3d} {r['rebal']:3d} {r['atr']:3.1f} | "
              f"{r['ann']:+7.1f}% {r['n']:5d} {r['tpy']:4.0f} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%", flush=True)

    # Best per portfolio
    best_per = {}
    for r in results:
        p = r['portfolio']
        if p not in best_per or r['ann'] > best_per[p]['ann']:
            best_per[p] = r
    print(f"\n  Best per portfolio:", flush=True)
    for r in sorted(best_per.values(), key=lambda x: -x['ann']):
        print(f"    {r['portfolio']:<15s} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}% "
              f"(Top={r['top_n']}, Reb={r['rebal']}, ATR={r['atr']:.1f})", flush=True)

    # Year-by-year for top 3
    for i, r in enumerate(results[:3]):
        print(f"\n  Year-by-year #{i+1}: {r['portfolio']} (Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
