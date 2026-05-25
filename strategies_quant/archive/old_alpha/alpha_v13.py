"""
Alpha V13 — Higher-Dimensional Factor Library
==============================================
V12 confirmed weight optimization can't break the +248% ceiling.
V13 takes a fundamentally different approach: extract genuinely NEW
mathematical dimensions from deep study of 269 strategies.

17 missing dimensions identified. This version implements the most
promising factors from 8 orthogonal dimensions:

  1. Volume-Price Microstructure: VDP, VW_CLOSE_POS, SMART_MONEY
  2. Information Theory: ENTROPY_INV, FISHER_INFO
  3. Higher-Order Statistics: SKEW, AC1
  4. Energy Physics: KINETIC_ENERGY, POTENTIAL_ENERGY
  5. Advanced Trend: IRLS_SLOPE
  6. Multi-Timeframe: MOM_ACCEL
  7. Structural Pattern: DRAWDOWN_DEPTH, VOL_SURGE
  8. Wavelet: WAVELET_ENERGY_RATIO

Each factor captures a dimension NOT covered by V7-V11 factors.

LOOK-AHEAD SELF-CHECK:
  [x] All factors use ONLY data up to d=di-1
  [x] Results stored at index di
  [x] No same-day data used
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
from alpha_v11 import compute_v11_factors, compute_v11_interactions
from alpha_v7c import backtest_v7c


def compute_v13_factors(NS, ND, C, O, H, L, V):
    """V13 factors — 8 new mathematical dimensions. STRICT no look-ahead.

    SELF-CHECK RULES:
    1. d = di - 1 (use yesterday as "current" data)
    2. Store result at index di
    3. Never access C[si, di], O[si, di], H[si, di], L[si, di], V[si, di]
    """
    t0 = time.time()
    new = {}

    # Precompute returns for reuse
    # returns[si, di] = return from d-1 to d, where d = di-1 (no look-ahead)
    returns = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(2, ND):
            d = di - 1  # SELF-CHECK: d = yesterday
            if not np.isnan(C[si, d]) and not np.isnan(C[si, d-1]) and C[si, d-1] > 0:
                returns[si, di] = (C[si, d] - C[si, d-1]) / C[si, d-1]

    # === DIMENSION 1: Volume-Price Microstructure ===

    # 1a. VDP — Volume Delta Pressure
    # Estimates buy/sell volume: V × (2C-H-L)/(H-L)
    # C near H → ~1 (buying), C near L → ~-1 (selling)
    VDP = np.full((NS, ND), np.nan)
    vdp_ema = np.full((NS, ND), np.nan)
    for si in range(NS):
        ema_val = np.nan
        ema_alpha = 2.0 / (10 + 1)  # 10-period EMA
        for di in range(2, ND):
            d = di - 1  # SELF-CHECK
            if np.isnan(V[si, d]) or np.isnan(H[si, d]) or np.isnan(L[si, d]) or np.isnan(C[si, d]):
                continue
            rng = H[si, d] - L[si, d]
            if rng < 1e-10:
                continue
            delta = V[si, d] * (2 * C[si, d] - H[si, d] - L[si, d]) / rng
            VDP[si, di] = delta  # SELF-CHECK: stored at di
            if np.isnan(ema_val):
                ema_val = delta
            else:
                ema_val = ema_val * (1 - ema_alpha) + delta * ema_alpha
            vdp_ema[si, di] = ema_val
    new['VDP'] = VDP
    new['VDP_EMA'] = vdp_ema
    print(f"  VDP done ({time.time()-t0:.1f}s)", flush=True)

    # 1b. VW_CLOSE_POS — Volume-Weighted Close Position
    # Where volume-weighted price falls in the daily range over 20 days
    # High value = volume at higher prices = accumulation
    VW_CP = np.full((NS, ND), np.nan)
    win = 20
    for si in range(NS):
        for di in range(win + 2, ND):
            d = di - 1  # SELF-CHECK
            cp_sum = 0.0
            v_sum = 0.0
            for dd in range(d - win + 1, d + 1):  # SELF-CHECK: up to d
                if np.isnan(C[si, dd]) or np.isnan(H[si, dd]) or np.isnan(L[si, dd]) or np.isnan(V[si, dd]):
                    continue
                rng = H[si, dd] - L[si, dd]
                if rng < 1e-10:
                    continue
                cp = (C[si, dd] - L[si, dd]) / rng
                cp_sum += cp * V[si, dd]
                v_sum += V[si, dd]
            if v_sum > 0:
                VW_CP[si, di] = cp_sum / v_sum  # SELF-CHECK: stored at di
    new['VW_CLOSE_POS'] = VW_CP
    print(f"  VW_CLOSE_POS done ({time.time()-t0:.1f}s)", flush=True)

    # 1c. SMART_MONEY — V/(H-L) z-score (institutional activity intensity)
    # Institutions: large volume, narrow spread. Retail: small volume, wide spread.
    SM_RAW = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(2, ND):
            d = di - 1  # SELF-CHECK
            if np.isnan(V[si, d]) or np.isnan(H[si, d]) or np.isnan(L[si, d]):
                continue
            rng = H[si, d] - L[si, d]
            if rng < 1e-10:
                continue
            SM_RAW[si, di] = V[si, d] / rng  # SELF-CHECK: stored at di

    # Z-score normalize over 60-day window
    SM_Z = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(62, ND):
            vals = SM_RAW[si, di-60:di]  # SELF-CHECK: up to di-1
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 30:
                mu = np.mean(valid)
                sd = np.std(valid)
                if sd > 1e-10:
                    SM_Z[si, di] = (SM_RAW[si, di] - mu) / sd  # SELF-CHECK: stored at di
    new['SMART_MONEY'] = SM_Z
    print(f"  SMART_MONEY done ({time.time()-t0:.1f}s)", flush=True)

    # === DIMENSION 2: Information Theory ===

    # 2a. ENTROPY_INV — Inverse Shannon entropy of returns
    # Low entropy = ordered/predictable market = good for trading
    # H = -Σ(p_i × log2(p_i)) over 10 bins, 50-day window
    ENTROPY = np.full((NS, ND), np.nan)
    n_bins = 10
    ent_win = 50
    for si in range(NS):
        for di in range(ent_win + 2, ND):
            d = di - 1  # SELF-CHECK
            rets = returns[si, di - ent_win:di]  # SELF-CHECK: up to di-1
            valid = rets[~np.isnan(rets)]
            if len(valid) < 20:
                continue
            # Bin returns
            counts, _ = np.histogram(valid, bins=n_bins)
            probs = counts[counts > 0] / len(valid)
            h = -np.sum(probs * np.log2(probs))
            ENTROPY[si, di] = h  # SELF-CHECK: stored at di
    # Invert: low entropy = high score (ordered market)
    ENT_INV = 100.0 - ENTROPY / np.log2(n_bins) * 100  # Normalize to 0-100
    new['ENTROPY_INV'] = ENT_INV
    print(f"  ENTROPY_INV done ({time.time()-t0:.1f}s)", flush=True)

    # 2b. FISHER_INFO — I(μ) = n/σ² over 20-day window
    # Measures how much information the price series carries about the regime
    # High = stable regime, Drop = regime change
    FISHER = np.full((NS, ND), np.nan)
    fi_win = 20
    for si in range(NS):
        for di in range(fi_win + 2, ND):
            d = di - 1  # SELF-CHECK
            rets = returns[si, di - fi_win:di]  # SELF-CHECK: up to di-1
            valid = rets[~np.isnan(rets)]
            if len(valid) < 10:
                continue
            var = np.var(valid)
            if var > 1e-15:
                FISHER[si, di] = len(valid) / var  # SELF-CHECK: stored at di
    new['FISHER_INFO'] = FISHER
    print(f"  FISHER_INFO done ({time.time()-t0:.1f}s)", flush=True)

    # === DIMENSION 3: Higher-Order Statistics ===

    # 3a. SKEW — Rolling skewness of returns (20-day)
    # Negative skew = more extreme down days. Positive skew = momentum bias
    SKEW = np.full((NS, ND), np.nan)
    sk_win = 20
    for si in range(NS):
        for di in range(sk_win + 2, ND):
            rets = returns[si, di - sk_win:di]  # SELF-CHECK: up to di-1
            valid = rets[~np.isnan(rets)]
            if len(valid) < 10:
                continue
            mu = np.mean(valid)
            sd = np.std(valid)
            if sd > 1e-10:
                SKEW[si, di] = np.mean(((valid - mu) / sd) ** 3)  # SELF-CHECK: stored at di
    new['SKEW'] = SKEW
    print(f"  SKEW done ({time.time()-t0:.1f}s)", flush=True)

    # 3b. AC1 — Autocorrelation at lag 1 over 20-day window
    # Positive = momentum regime, Negative = mean-reversion regime
    AC1 = np.full((NS, ND), np.nan)
    ac_win = 20
    for si in range(NS):
        for di in range(ac_win + 3, ND):
            rets = returns[si, di - ac_win:di]  # SELF-CHECK: up to di-1
            valid = rets[~np.isnan(rets)]
            if len(valid) < 10:
                continue
            mu = np.mean(valid)
            var = np.var(valid)
            if var < 1e-15:
                continue
            acov = np.mean((valid[:-1] - mu) * (valid[1:] - mu))
            AC1[si, di] = acov / var  # SELF-CHECK: stored at di
    new['AC1'] = AC1
    print(f"  AC1 done ({time.time()-t0:.1f}s)", flush=True)

    # === DIMENSION 4: Energy Physics ===

    # 4a. KINETIC_ENERGY — Rolling mean of log_return × volume
    # Captures directional movement energy weighted by participation
    KE = np.full((NS, ND), np.nan)
    ke_win = 20
    for si in range(NS):
        for di in range(ke_win + 2, ND):
            d = di - 1  # SELF-CHECK
            ke_vals = []
            for dd in range(d - ke_win + 1, d + 1):  # SELF-CHECK: up to d
                if np.isnan(returns[si, dd + 1]) or np.isnan(V[si, dd]):
                    continue
                ke_vals.append(returns[si, dd + 1] * V[si, dd])
            if len(ke_vals) >= 10:
                KE[si, di] = np.mean(ke_vals)  # SELF-CHECK: stored at di
    new['KINETIC_ENERGY'] = KE
    print(f"  KINETIC_ENERGY done ({time.time()-t0:.1f}s)", flush=True)

    # 4b. POTENTIAL_ENERGY — Rolling mean of (close_pos - 0.5) × V × 2
    # Volume at high prices = upside potential
    PE = np.full((NS, ND), np.nan)
    pe_win = 20
    for si in range(NS):
        for di in range(pe_win + 2, ND):
            d = di - 1  # SELF-CHECK
            pe_vals = []
            for dd in range(d - pe_win + 1, d + 1):  # SELF-CHECK: up to d
                if np.isnan(C[si, dd]) or np.isnan(H[si, dd]) or np.isnan(L[si, dd]) or np.isnan(V[si, dd]):
                    continue
                rng = H[si, dd] - L[si, dd]
                if rng < 1e-10:
                    continue
                cp = (C[si, dd] - L[si, dd]) / rng
                pe_vals.append((cp - 0.5) * V[si, dd] * 2)
            if len(pe_vals) >= 10:
                PE[si, di] = np.mean(pe_vals)  # SELF-CHECK: stored at di
    new['POTENTIAL_ENERGY'] = PE
    print(f"  POTENTIAL_ENERGY done ({time.time()-t0:.1f}s)", flush=True)

    # === DIMENSION 5: Advanced Trend ===

    # 5a. IRLS_SLOPE — Iteratively Reweighted Least Squares slope
    # L1-robust: immune to 涨跌停 outliers
    # Hardy weight: w(j) = 1/sqrt(dist² + eps²)
    IRLS = np.full((NS, ND), np.nan)
    irls_win = 15
    for si in range(NS):
        for di in range(irls_win + 2, ND):
            d = di - 1  # SELF-CHECK
            # Collect valid price points
            xs, ys = [], []
            for dd in range(d - irls_win + 1, d + 1):  # SELF-CHECK: up to d
                if not np.isnan(C[si, dd]):
                    xs.append(dd - (d - irls_win + 1))
                    ys.append(C[si, dd])
            if len(xs) < 8:
                continue
            xs = np.array(xs, dtype=float)
            ys = np.array(ys, dtype=float)

            # Adaptive epsilon from H-L range
            eps = np.mean(np.abs(np.diff(ys))) * 0.1 + 1e-10

            # IRLS: 3 iterations
            weights = np.ones(len(xs))
            for _ in range(3):
                wx = weights * xs
                wy = weights * ys
                wxx = np.sum(wx * xs)
                wxy = np.sum(wx * ys)
                ws = np.sum(weights)
                wsx = np.sum(wx)
                wsy = np.sum(wy)
                denom = ws * wxx - wsx * wsx
                if abs(denom) < 1e-15:
                    break
                slope = (ws * wxy - wsx * wsy) / denom
                # Update weights with Hardy kernel
                residuals = ys - (wsy / ws + slope * (xs - wsx / ws))
                dist_sq = residuals ** 2
                weights = 1.0 / np.sqrt(dist_sq + eps ** 2)

            IRLS[si, di] = slope  # SELF-CHECK: stored at di
    new['IRLS_SLOPE'] = IRLS
    print(f"  IRLS_SLOPE done ({time.time()-t0:.1f}s)", flush=True)

    # === DIMENSION 6: Multi-Timeframe ===

    # 6a. MOM_ACCEL — Momentum acceleration: MOM(5) - MOM(10)
    # Positive = short-term momentum accelerating ahead of longer-term
    MOM_ACCEL = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(12, ND):
            d = di - 1  # SELF-CHECK
            if d < 10:
                continue
            if np.isnan(C[si, d]) or np.isnan(C[si, d-4]) or np.isnan(C[si, d-9]):
                continue
            if C[si, d-4] <= 0 or C[si, d-9] <= 0:
                continue
            mom5 = (C[si, d] - C[si, d-4]) / C[si, d-4]
            mom10 = (C[si, d] - C[si, d-9]) / C[si, d-9]
            MOM_ACCEL[si, di] = mom5 - mom10  # SELF-CHECK: stored at di
    new['MOM_ACCEL'] = MOM_ACCEL
    print(f"  MOM_ACCEL done ({time.time()-t0:.1f}s)", flush=True)

    # === DIMENSION 7: Structural Patterns ===

    # 7a. DRAWDOWN_DEPTH — 52-week drawdown from high
    # Extreme drawdown = mean reversion opportunity
    DD_DEPTH = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(252, ND):
            d = di - 1  # SELF-CHECK
            prices = C[si, di-252:di]  # SELF-CHECK: up to di-1
            valid = prices[~np.isnan(prices)]
            if len(valid) < 100:
                continue
            high52 = np.max(valid)
            if high52 > 0:
                DD_DEPTH[si, di] = (C[si, d] - high52) / high52  # SELF-CHECK: stored at di
    new['DRAWDOWN_DEPTH'] = DD_DEPTH
    print(f"  DRAWDOWN_DEPTH done ({time.time()-t0:.1f}s)", flush=True)

    # 7b. VOL_SURGE — Volume surge relative to 20-day average
    # High surge during compression = institutional accumulation
    VOL_SURGE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(22, ND):
            d = di - 1  # SELF-CHECK
            vols = V[si, di-20:di]  # SELF-CHECK: up to di-1
            valid = vols[~np.isnan(vols)]
            if len(valid) < 10:
                continue
            avg = np.mean(valid)
            if avg > 0 and not np.isnan(V[si, d]):
                VOL_SURGE[si, di] = V[si, d] / avg  # SELF-CHECK: stored at di
    new['VOL_SURGE'] = VOL_SURGE
    print(f"  VOL_SURGE done ({time.time()-t0:.1f}s)", flush=True)

    # === DIMENSION 8: Wavelet Energy ===

    # 8a. WAVELET_ENERGY_RATIO — Haar wavelet noise/trend ratio
    # High = choppy (noise), Low = smooth (trending)
    WER = np.full((NS, ND), np.nan)
    wer_win = 32  # Must be power of 2
    for si in range(NS):
        for di in range(wer_win + 2, ND):
            d = di - 1  # SELF-CHECK
            prices = C[si, di - wer_win:di]  # SELF-CHECK: up to di-1
            valid = prices[~np.isnan(prices)]
            if len(valid) < 16:
                continue
            # Use last power-of-2 samples
            n_use = 1
            while n_use * 2 <= len(valid):
                n_use *= 2
            data = valid[-n_use:]

            # Haar wavelet 3 levels
            approx = data.copy()
            total_detail_energy = 0.0
            for level in range(min(3, int(np.log2(n_use)))):
                n_half = len(approx) // 2
                if n_half < 1:
                    break
                new_approx = (approx[0::2][:n_half] + approx[1::2][:n_half]) / 2.0
                detail = (approx[0::2][:n_half] - approx[1::2][:n_half]) / 2.0
                total_detail_energy += np.sum(detail ** 2)
                approx = new_approx

            approx_energy = np.sum(approx ** 2)
            total = total_detail_energy + approx_energy
            if total > 1e-15:
                WER[si, di] = total_detail_energy / total  # SELF-CHECK: stored at di
    # Invert: low noise ratio = trending = higher score
    WER_INV = 1.0 - WER
    new['WAVELET_TREND'] = WER_INV
    print(f"  WAVELET_TREND done ({time.time()-t0:.1f}s)", flush=True)

    # === Rank normalize all factors ===
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

    factor_names = ['VDP', 'VDP_EMA', 'VW_CLOSE_POS', 'SMART_MONEY',
                    'ENTROPY_INV', 'FISHER_INFO', 'SKEW', 'AC1',
                    'KINETIC_ENERGY', 'POTENTIAL_ENERGY', 'IRLS_SLOPE',
                    'MOM_ACCEL', 'DRAWDOWN_DEPTH', 'VOL_SURGE', 'WAVELET_TREND']

    for name in factor_names:
        new[f'R_{name}'] = rank_pct(new[name])

    # Invert factors where low values are better
    # DRAWDOWN_DEPTH: more negative = more oversold = better buy
    inv_names = ['DRAWDOWN_DEPTH']
    for name in inv_names:
        inv = new[f'R_{name}'].copy()
        mask = ~np.isnan(inv)
        inv[mask] = 100.0 - inv[mask]
        new[f'R_{name}_INV'] = inv

    print(f"  All ranked done ({time.time()-t0:.1f}s)", flush=True)
    print(f"  Total V13 factors: {len(new)}", flush=True)
    return new


def compute_v13_interactions(all_factors, NS, ND):
    """V13 interactions — volume×structure, information×squeeze, energy×body."""
    t0 = time.time()
    new = {}

    def interact(name_a, name_b, out_name):
        a = all_factors.get(name_a, np.full((NS, ND), np.nan))
        b = all_factors.get(name_b, np.full((NS, ND), np.nan))
        res = np.full((NS, ND), np.nan)
        mask = ~np.isnan(a) & ~np.isnan(b)
        res[mask] = a[mask] * b[mask] / 100
        new[out_name] = res

    # Volume × Structure interactions
    interact('R_VDP_EMA', 'R_BODY_NW', 'VDP_BNW')       # Volume flow + price quality
    interact('R_VDP_EMA', 'R_TENSION', 'VDP_TENS')       # Volume flow + displacement
    interact('R_VDP_EMA', 'R_BB_SQUEEZE_INV', 'VDP_SQZ') # Volume confirms squeeze
    interact('R_VW_CLOSE_POS', 'R_BODY_NW', 'VWCP_BNW')  # Accumulation + body quality
    interact('R_SMART_MONEY', 'R_BB_WIDTH_PCT_INV', 'SM_BWP') # Smart money in compression
    interact('R_SMART_MONEY', 'R_TENSION', 'SM_TENS')     # Smart money + displacement

    # Information × Volatility interactions
    interact('R_ENTROPY_INV', 'R_BB_WIDTH_PCT_INV', 'ENT_BWP')  # Ordered + compressed
    interact('R_ENTROPY_INV', 'R_R_SQUARED', 'ENT_R2')          # Ordered + coherent trend
    interact('R_FISHER_INFO', 'R_BB_SQUEEZE_INV', 'FI_SQZ')     # Stable regime + squeeze
    interact('R_FISHER_INFO', 'R_TENSION', 'FI_TENS')            # Stable + displacement

    # Energy × Trend interactions
    interact('R_KINETIC_ENERGY', 'R_TENSION', 'KE_TENS')         # Energy + displacement
    interact('R_KINETIC_ENERGY', 'R_BB_SQUEEZE_INV', 'KE_SQZ')   # Energy + squeeze
    interact('R_POTENTIAL_ENERGY', 'R_BODY_NW', 'PE_BNW')        # Latent energy + body

    # Higher moments × Structure
    interact('R_SKEW', 'R_BB_WIDTH_PCT_INV', 'SKEW_BWP')       # Skew + compression
    interact('R_AC1', 'R_TENSION', 'AC1_TENS')                   # Autocorr + displacement
    interact('R_AC1', 'R_BB_SQUEEZE_INV', 'AC1_SQZ')            # Momentum regime + squeeze

    # Advanced trend × Volume
    interact('R_IRLS_SLOPE', 'R_VDP_EMA', 'IRLS_VDP')           # Robust trend + volume
    interact('R_IRLS_SLOPE', 'R_BODY_NW', 'IRLS_BNW')           # Robust trend + body

    # Multi-dimension convergence
    interact('R_MOM_ACCEL', 'R_VDP_EMA', 'MA_VDP')              # Momentum accel + volume
    interact('R_VOL_SURGE', 'R_BB_SQUEEZE_INV', 'VS_SQZ')       # Volume surge + squeeze
    interact('R_WAVELET_TREND', 'R_TENSION', 'WT_TENS')         # Wavelet trend + displacement

    # Volume-confirmed best combos
    interact('R_VDP_EMA', 'R_BWP_BNW', 'VDP_BWPBNW')  # VDP + best combo
    interact('R_SMART_MONEY', 'R_BWP_BNW', 'SM_BWPBNW') # Smart money + best combo

    # Rank normalize interactions
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

    for name in list(new.keys()):
        new[f'R_{name}'] = rank_pct(new[name])

    print(f"  V13 interactions done ({time.time()-t0:.1f}s)", flush=True)
    return new


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V13 — Higher-Dimensional Factors", flush=True)
    print("  8 New Dimensions × 15 Factors × 22 Interactions", flush=True)
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
    v11_all = {**v10_all, **v11_factors}
    v11_inter = compute_v11_interactions(v11_all, NS, ND)
    v11_all.update(v11_inter)

    v13_factors = compute_v13_factors(NS, ND, C, O, H, L, V)
    all_factors = {**v11_all, **v13_factors}
    v13_inter = compute_v13_interactions(all_factors, NS, ND)
    all_factors.update(v13_inter)

    print(f"\n  Total factors: {len(all_factors)}", flush=True)

    # === SINGLE FACTOR TESTS ===
    print(f"\n  === SINGLE FACTOR TESTS (V13) ===", flush=True)
    single_factors = ['R_VDP_EMA', 'R_VW_CLOSE_POS', 'R_SMART_MONEY',
                      'R_ENTROPY_INV', 'R_FISHER_INFO', 'R_SKEW', 'R_AC1',
                      'R_KINETIC_ENERGY', 'R_IRLS_SLOPE', 'R_MOM_ACCEL',
                      'R_DRAWDOWN_DEPTH_INV', 'R_VOL_SURGE', 'R_WAVELET_TREND']
    for fname in single_factors:
        r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=3, rebalance_days=10, atr_stop_mult=1.5)
        if r:
            print(f"  {fname:<25s}: Ann={r['ann']:+7.1f}% WR={r['wr']:5.1f}% "
                  f"Edge={r['edge']:+5.2f}% DD={r['max_dd']:5.1f}%", flush=True)

    # === PORTFOLIO TESTS ===
    portfolios = {
        # V10 baseline
        'BwpBNW': {'R_BWP_BNW': 0.3, 'R_TENSION': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # VDP-confirmed
        'VdpBNW': {'R_VDP_BNW': 0.3, 'R_TENSION': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        'VdpTens': {'R_VDP_TENS': 0.3, 'R_BODY_NW': 0.3,
                    'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        'VdpSqz': {'R_VDP_SQZ': 0.3, 'R_BODY_NW': 0.3,
                   'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        # Information × Squeeze
        'EntBwp': {'R_ENT_BWP': 0.3, 'R_BODY_NW': 0.3,
                   'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        'FiSqz': {'R_FI_SQZ': 0.3, 'R_BODY_NW': 0.3,
                  'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        # Energy × Trend
        'KeTens': {'R_KE_TENS': 0.3, 'R_BODY_NW': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        'KeSqz': {'R_KE_SQZ': 0.3, 'R_TENSION': 0.3,
                  'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # Smart Money
        'SmBwp': {'R_SM_BWP': 0.3, 'R_TENSION': 0.3,
                  'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # Autocorrelation × Structure
        'Ac1Sqz': {'R_AC1_SQZ': 0.3, 'R_BODY_NW': 0.3,
                   'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        # Robust trend
        'IrlsVdp': {'R_IRLS_VDP': 0.3, 'R_TENSION': 0.3,
                    'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # Volume surge × Squeeze
        'VsSqz': {'R_VS_SQZ': 0.3, 'R_BODY_NW': 0.3,
                  'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        # Multi-dimension: VDP + Entropy + Squeeze + Body
        'MultiD': {'R_VDP_EMA': 0.15, 'R_ENTROPY_INV': 0.1,
                   'R_BB_WIDTH_PCT_INV': 0.15, 'R_BODY_NW': 0.25,
                   'R_TENSION': 0.2, 'R_R_SQUARED': 0.15},
        # Full 4-dimension
        'Full4D': {'R_VDP_BNW': 0.2, 'R_ENT_BWP': 0.2,
                   'R_KE_TENS': 0.2, 'R_FI_SQZ': 0.2,
                   'R_R_SQUARED': 0.2},
        # Best combo + VDP confirmation
        'BwpVdp': {'R_BWP_BNW': 0.25, 'R_VDP_EMA': 0.15,
                   'R_TENSION': 0.25, 'R_R_SQUARED': 0.15, 'R_SMA_DEV': 0.2},
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
    print(f"  TOP 30 (V13 HIGHER-DIMENSIONAL)", flush=True)
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
