"""
Alpha V8 — New Factor Dimensions + Strict No-Look-Ahead
========================================================
V7 bug-fix后: BodyNW Top=1 = +165.4%, DD=57.7%

新因子 (从策略和概率论学习中提取):
  1. ENTROPY: Shannon entropy of returns (不确定性/市场效率)
  2. KALMAN_SLOPE: Kalman filter trend estimate slope
  3. OFI: Order Flow Imbalance = (C-L)/(H-L)*V (买卖压力)
  4. VOL_DELTA: Volume Delta Pressure = EMA of delta pressure

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
from alpha_v7c import backtest_v7c  # BUG-FIXED version


def compute_v8_factors(NS, ND, C, O, H, L, V):
    """New V8 factors — STRICT no look-ahead.

    SELF-CHECK RULES:
    1. d = di - 1 (use yesterday as "current" data)
    2. Store result at index di (backtest reads at di)
    3. Never access C[si, di], O[si, di], H[si, di], L[si, di], V[si, di]
    """
    t0 = time.time()
    new = {}

    # === 1. ENTROPY: Shannon entropy of returns ===
    # Higher entropy = more random = less predictable = avoid
    # Lower entropy = trending = more predictable = trade
    ENTROPY = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(22, ND):
            d = di - 1  # SELF-CHECK: d = yesterday
            # Get last 20 log returns up to d
            rets = []
            for dd in range(d - 19, d + 1):  # SELF-CHECK: up to d only
                if np.isnan(C[si, dd]) or np.isnan(C[si, dd - 1]) or C[si, dd - 1] <= 0:
                    continue
                rets.append(np.log(C[si, dd] / C[si, dd - 1]))
            if len(rets) < 15:
                continue
            rets = np.array(rets)
            # Bin returns into histogram and compute entropy
            n_bins = 10
            counts, _ = np.histogram(rets, bins=n_bins)
            probs = counts / len(rets)
            probs = probs[probs > 0]
            h = -np.sum(probs * np.log2(probs))
            ENTROPY[si, di] = h  # SELF-CHECK: stored at di, uses data up to d=di-1

    new['ENTROPY'] = ENTROPY
    print(f"  Entropy done ({time.time()-t0:.1f}s)", flush=True)

    # === 2. KALMAN_SLOPE: Simple Kalman filter trend slope ===
    # State: [price, velocity]. Update with C[si, d] as measurement.
    KALMAN_SLOPE = np.full((NS, ND), np.nan)
    for si in range(NS):
        # State initialization
        x = np.array([0.0, 0.0])  # [price, velocity]
        P = np.eye(2) * 1000  # covariance
        Q = np.array([[0.01, 0.001], [0.001, 0.001]])  # process noise
        R = np.array([[1.0]])  # measurement noise
        H_mat = np.array([[1.0, 0.0]])  # observation matrix
        initialized = False

        for di in range(2, ND):
            d = di - 1  # SELF-CHECK: d = yesterday
            z = C[si, d]  # measurement = yesterday's close
            if np.isnan(z):
                continue

            if not initialized:
                x = np.array([z, 0.0])
                P = np.eye(2) * 1.0
                initialized = True
                # Store initial velocity
                KALMAN_SLOPE[si, di] = 0.0
                continue

            # Predict
            F = np.array([[1.0, 1.0], [0.0, 1.0]])  # state transition
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

            # Update
            y_innov = z - H_mat @ x_pred
            S = H_mat @ P_pred @ H_mat.T + R
            K = P_pred @ H_mat.T / S[0, 0]
            x = x_pred + K @ y_innov
            P = (np.eye(2) - K @ H_mat) @ P_pred

            # Store velocity (trend slope) normalized by price
            if x[0] > 0:
                KALMAN_SLOPE[si, di] = x[1] / x[0] * 100  # SELF-CHECK: stored at di
    new['KALMAN_SLOPE'] = KALMAN_SLOPE
    print(f"  Kalman slope done ({time.time()-t0:.1f}s)", flush=True)

    # === 3. OFI: Order Flow Imbalance ===
    # (C-L)/(H-L) * V — how much volume is on the buy side
    # High OFI = buyers dominating
    OFI = np.full((NS, ND), np.nan)
    for si in range(NS):
        ema_ofi = np.nan
        alpha = 2.0 / 11  # 10-day EMA
        for di in range(2, ND):
            d = di - 1  # SELF-CHECK: d = yesterday
            c, h, l = C[si, d], H[si, d], L[si, d]  # SELF-CHECK: all at d=di-1
            v = V[si, d]
            if np.isnan(c) or np.isnan(h) or np.isnan(l) or np.isnan(v):
                continue
            hl = h - l
            if hl <= 0 or v <= 0:
                continue
            ofi_raw = (c - l) / hl * v  # buy pressure × volume
            if np.isnan(ema_ofi):
                ema_ofi = ofi_raw
            else:
                ema_ofi = alpha * ofi_raw + (1 - alpha) * ema_ofi
            OFI[si, di] = ema_ofi  # SELF-CHECK: stored at di

    new['OFI'] = OFI
    print(f"  OFI done ({time.time()-t0:.1f}s)", flush=True)

    # === 4. VOL_DELTA: Volume Delta Pressure ===
    # (2*C - H - L) / (H - L) * V — net buying/selling pressure
    # Positive = buying, Negative = selling
    VOL_DELTA = np.full((NS, ND), np.nan)
    for si in range(NS):
        ema_vd = np.nan
        alpha = 2.0 / 11
        for di in range(2, ND):
            d = di - 1  # SELF-CHECK: d = yesterday
            c, h, l = C[si, d], H[si, d], L[si, d]
            v = V[si, d]
            if np.isnan(c) or np.isnan(h) or np.isnan(l) or np.isnan(v):
                continue
            hl = h - l
            if hl <= 0 or v <= 0:
                continue
            delta = (2 * c - h - l) / hl * v
            if np.isnan(ema_vd):
                ema_vd = delta
            else:
                ema_vd = alpha * delta + (1 - alpha) * ema_vd
            VOL_DELTA[si, di] = ema_vd  # SELF-CHECK: stored at di

    new['VOL_DELTA'] = VOL_DELTA
    print(f"  Vol delta done ({time.time()-t0:.1f}s)", flush=True)

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

    for name in ['ENTROPY', 'KALMAN_SLOPE', 'OFI', 'VOL_DELTA']:
        new[f'R_{name}'] = rank_pct(new[name])

    # ENTROPY direction fix: high entropy = random = bad, so invert
    # R_ENTROPY: high rank = high entropy = SELECTED (wrong direction)
    # R_ENTROPY_INV: high rank = low entropy = SELECTED (correct direction)
    inv = new['R_ENTROPY'].copy()
    mask = ~np.isnan(inv)
    inv[mask] = 100.0 - inv[mask]
    new['R_ENTROPY_INV'] = inv
    print(f"  Ranked done ({time.time()-t0:.1f}s)", flush=True)

    # === New interactions with existing factors ===
    # KALMAN × TENSION — adaptive trend + structural displacement
    tens = new.get('R_KALMAN_SLOPE')  # Will be used from all_factors later

    return new


def compute_v8_interactions(all_factors, NS, ND):
    """V8 interaction factors."""
    t0 = time.time()
    new = {}

    # KALMAN × TENSION — adaptive trend confirmed by structure
    kal = all_factors.get('R_KALMAN_SLOPE', np.full((NS, ND), np.nan))
    tens = all_factors.get('R_TENSION', np.full((NS, ND), np.nan))
    KAL_TENS = np.full((NS, ND), np.nan)
    mask = ~np.isnan(kal) & ~np.isnan(tens)
    KAL_TENS[mask] = kal[mask] * tens[mask] / 100
    new['KAL_TENS'] = KAL_TENS

    # OFI × R² — strong order flow in quality trends
    ofi = all_factors.get('R_OFI', np.full((NS, ND), np.nan))
    r2 = all_factors.get('R_R_SQUARED', np.full((NS, ND), np.nan))
    OFI_R2 = np.full((NS, ND), np.nan)
    mask = ~np.isnan(ofi) & ~np.isnan(r2)
    OFI_R2[mask] = ofi[mask] * r2[mask] / 100
    new['OFI_R2'] = OFI_R2

    # VOL_DELTA × BODY_NW — pressure + conviction
    vd = all_factors.get('R_VOL_DELTA', np.full((NS, ND), np.nan))
    bnw = all_factors.get('R_BODY_NW', np.full((NS, ND), np.nan))
    VD_BNW = np.full((NS, ND), np.nan)
    mask = ~np.isnan(vd) & ~np.isnan(bnw)
    VD_BNW[mask] = vd[mask] * bnw[mask] / 100
    new['VD_BNW'] = VD_BNW

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

    print(f"  V8 interactions done ({time.time()-t0:.1f}s)", flush=True)
    return new


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V8 — New Factor Dimensions", flush=True)
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

    all_factors = {**base_factors, **inter_factors, **extra_factors,
                   **v7e_factors, **adv_inter, **v8_factors}

    v8_inter = compute_v8_interactions(all_factors, NS, ND)
    all_factors.update(v8_inter)

    print(f"\n  Total factors: {len(all_factors)}", flush=True)

    # SINGLE FACTOR TESTS (bug-fixed backtest)
    print(f"\n  === SINGLE FACTOR TESTS ===", flush=True)
    for fname in ['R_ENTROPY', 'R_KALMAN_SLOPE', 'R_OFI', 'R_VOL_DELTA']:
        r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=3, rebalance_days=10, atr_stop_mult=1.5)
        if r:
            print(f"  {fname}: Ann={r['ann']:+.1f}% WR={r['wr']:.0f}% DD={r['max_dd']:.1f}%", flush=True)

    # PORTFOLIO TESTS
    portfolios = {
        # V7 best (reference)
        'BodyNW': {'R_BODY_NW': 0.3, 'R_TENSION': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # Add Kalman slope
        'BNW_Kal': {'R_BODY_NW': 0.25, 'R_TENSION': 0.25,
                    'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.1, 'R_KALMAN_SLOPE': 0.2},
        # Add OFI
        'BNW_OFI': {'R_BODY_NW': 0.25, 'R_TENSION': 0.25,
                    'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.1, 'R_OFI': 0.2},
        # Add Volume Delta
        'BNW_VD': {'R_BODY_NW': 0.25, 'R_TENSION': 0.25,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.1, 'R_VOL_DELTA': 0.2},
        # All new factors
        'BNW_All': {'R_BODY_NW': 0.2, 'R_TENSION': 0.2, 'R_R_SQUARED': 0.15,
                    'R_KALMAN_SLOPE': 0.15, 'R_OFI': 0.15, 'R_VOL_DELTA': 0.15},
        # Kalman × Tension interaction
        'KalTens': {'R_KAL_TENS': 0.3, 'R_BODY_NW': 0.3,
                    'R_R_SQUARED': 0.2, 'R_OFI': 0.2},
        # OFI × R² interaction
        'OFI_R2': {'R_OFI_R2': 0.3, 'R_TENSION': 0.3,
                   'R_BODY_VOL': 0.2, 'R_SMA_DEV': 0.2},
        # Pure new factors
        'NewPure': {'R_KALMAN_SLOPE': 0.3, 'R_OFI': 0.3,
                    'R_VOL_DELTA': 0.2, 'R_BODY_NW': 0.2},
        # VD × BODY_NW interaction
        'VD_BNW': {'R_VD_BNW': 0.3, 'R_TENSION': 0.3,
                   'R_R_SQUARED': 0.2, 'R_KALMAN_SLOPE': 0.2},
        # Entropy filter: low entropy (trending) + strong structure
        'LowEnt': {'R_ENTROPY': 0.2, 'R_BODY_NW': 0.3,
                   'R_TENSION': 0.3, 'R_R_SQUARED': 0.2},
    }

    results = []
    for pname, weights in portfolios.items():
        for top_n in [1, 2, 3]:
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

    print(f"\n{'='*110}", flush=True)
    print(f"  TOP 30 (BUG-FIXED V8)", flush=True)
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

    print(f"\n{'='*70}", flush=True)
