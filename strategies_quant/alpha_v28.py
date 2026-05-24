"""
Alpha V28 — High-Dimensional Price Structure Analysis
======================================================
From user insight: "缺少结构性分析 价格行为的结构 空间结构 时间结构 要从高维看问题"

This strategy analyzes THREE structural dimensions:

1. SPATIAL STRUCTURE (空间结构):
   - Price position relative to key levels (52w high/low, VWAP, pivot points)
   - Support/resistance proximity and density
   - Price clustering: how concentrated are recent closes around certain levels
   - Spatial tension: distance compression between support and resistance

2. TEMPORAL STRUCTURE (时间结构):
   - Time since last major move (trend age)
   - Rhythm: regularity of swing highs/lows timing
   - Duration of current regime (bars since last direction change)
   - Temporal acceleration: are swings getting shorter (compression) or longer (expansion)

3. PRICE ACTION STRUCTURE (价格行为结构):
   - Bar sequence patterns: sequences of up/down bars
   - Body/shadow ratio consistency (trending vs chaotic)
   - Inside/outside bar detection (compression/expansion)
   - Gap analysis: overnight gaps and their fill rate
   - Consecutive pattern scoring

From probability_theory.md:
  - Section 24: Geometric unified formula — "一切技术指标都是几何"
  - Section 32: DTW pattern matching — "当前走势像历史上的哪段"
  - Section 28: Recurrence plots — nonlinear dynamics fingerprinting

NO LOOK-AHEAD: All computations use d = di - 1 only.
"""
import sys, os, time, warnings
import numpy as np
from scipy.stats import rankdata
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7 import compute_all_factors
from alpha_v7b import compute_interaction_factors
from alpha_v7d import compute_extra_factors
from alpha_v7e import compute_v7e_factors
from alpha_v7f import compute_advanced_interactions
from alpha_v8 import compute_v8_factors, compute_v8_interactions
from alpha_v9 import compute_v9_factors, compute_v9_interactions
from alpha_v10 import compute_v10_factors, compute_v10_interactions
from alpha_v11 import compute_v11_factors, compute_v11_interactions
from alpha_v7c import backtest_v7c


def compute_structure_factors(NS, ND, C, O, H, L, V):
    """Compute high-dimensional price structure factors.

    SELF-CHECK: d = di - 1. All data uses [0:d+1] only.
    """
    t0 = time.time()
    new = {}

    # =====================================================================
    # SPATIAL STRUCTURE
    # =====================================================================
    # 1. Position relative to 52-week range
    POS_52W = np.full((NS, ND), np.nan)
    # 2. Spatial tension: distance between recent support and resistance
    SPATIAL_TENSION = np.full((NS, ND), np.nan)
    # 3. Price clustering score (how concentrated recent closes are)
    PRICE_CLUSTERING = np.full((NS, ND), np.nan)
    # 4. VWAP deviation
    VWAP_DEV = np.full((NS, ND), np.nan)
    # 5. Pivot point proximity score
    PIVOT_PROX = np.full((NS, ND), np.nan)

    # =====================================================================
    # TEMPORAL STRUCTURE
    # =====================================================================
    # 6. Trend age: bars since last direction change
    TREND_AGE = np.full((NS, ND), np.nan)
    # 7. Swing rhythm: regularity of swing timing
    SWING_RHYTHM = np.full((NS, ND), np.nan)
    # 8. Temporal compression: are swings getting shorter?
    TEMPORAL_COMPRESS = np.full((NS, ND), np.nan)
    # 9. Duration-weighted momentum
    DURATION_MOM = np.full((NS, ND), np.nan)

    # =====================================================================
    # PRICE ACTION STRUCTURE
    # =====================================================================
    # 10. Consecutive direction score
    CONSEC_DIR = np.full((NS, ND), np.nan)
    # 11. Body consistency: how consistent are body directions
    BODY_CONSISTENCY = np.full((NS, ND), np.nan)
    # 12. Inside/outside bar pattern
    BAR_PATTERN = np.full((NS, ND), np.nan)
    # 13. Gap analysis
    GAP_SCORE = np.full((NS, ND), np.nan)
    # 14. Shadow balance: upper vs lower shadow over N bars
    SHADOW_BALANCE = np.full((NS, ND), np.nan)
    # 15. Range contraction/expansion sequence
    RANGE_SEQUENCE = np.full((NS, ND), np.nan)

    for si in range(NS):
        swing_times = []  # Track timing of swing points
        last_direction = 0
        trend_start = 0
        consecutive_up = 0
        consecutive_down = 0

        for di in range(20, ND):
            d = di - 1  # SELF-CHECK

            if np.isnan(C[si, d]):
                continue

            # =================================================================
            # SPATIAL STRUCTURE
            # =================================================================

            # 1. 52-week position (252 trading days)
            lookback = min(d, 252)
            high_52w = np.nanmax(H[si, d - lookback:d + 1])
            low_52w = np.nanmin(L[si, d - lookback:d + 1])
            if high_52w > low_52w and not np.isnan(high_52w):
                POS_52W[si, di] = (C[si, d] - low_52w) / (high_52w - low_52w) * 100

            # 2. Spatial tension: Bollinger bandwidth + ATR compression
            lookback_20 = min(d, 20)
            closes_20 = C[si, d - lookback_20 + 1:d + 1]
            valid_c = closes_20[~np.isnan(closes_20)]
            if len(valid_c) >= 10:
                ma = np.mean(valid_c)
                std = np.std(valid_c)
                if std > 0 and ma > 0:
                    bw = 2 * std / ma  # Bollinger bandwidth
                    # Also measure recent high-low range
                    recent_h = np.nanmax(H[si, d - lookback_20 + 1:d + 1])
                    recent_l = np.nanmin(L[si, d - lookback_20 + 1:d + 1])
                    if not np.isnan(recent_h) and not np.isnan(recent_l):
                        range_pct = (recent_h - recent_l) / ma
                        # Low bw + low range = high spatial tension (compressed)
                        SPATIAL_TENSION[si, di] = (1.0 / max(bw, 0.01)) + (1.0 / max(range_pct, 0.01))

            # 3. Price clustering: inverse of coefficient of variation of recent closes
            if len(valid_c) >= 10:
                cv = np.std(valid_c) / np.mean(valid_c) if np.mean(valid_c) > 0 else 999
                PRICE_CLUSTERING[si, di] = 1.0 / max(cv, 0.001)  # High = clustered

            # 4. VWAP deviation: (close - VWAP) / VWAP
            lookback_vwap = min(d, 20)
            vols = V[si, d - lookback_vwap + 1:d + 1]
            typical = (H[si, d - lookback_vwap + 1:d + 1] + L[si, d - lookback_vwap + 1:d + 1] + C[si, d - lookback_vwap + 1:d + 1]) / 3
            valid_mask_vwap = ~np.isnan(vols) & ~np.isnan(typical)
            if valid_mask_vwap.sum() >= 5:
                vp = np.sum(vols[valid_mask_vwap] * typical[valid_mask_vwap])
                tv = np.sum(vols[valid_mask_vwap])
                if tv > 0:
                    vwap = vp / tv
                    if vwap > 0 and not np.isnan(C[si, d]):
                        VWAP_DEV[si, di] = (C[si, d] - vwap) / vwap * 100

            # 5. Pivot points: classic S1/R1 proximity
            if d >= 1 and not np.isnan(H[si, d]) and not np.isnan(L[si, d]) and not np.isnan(C[si, d]):
                prev_h = H[si, d - 1] if not np.isnan(H[si, d - 1]) else H[si, d]
                prev_l = L[si, d - 1] if not np.isnan(L[si, d - 1]) else L[si, d]
                prev_c = C[si, d - 1] if not np.isnan(C[si, d - 1]) else C[si, d]
                if not np.isnan(prev_h) and not np.isnan(prev_l) and not np.isnan(prev_c):
                    pivot = (prev_h + prev_l + prev_c) / 3
                    r1 = 2 * pivot - prev_l
                    s1 = 2 * pivot - prev_h
                    dist_to_r1 = abs(C[si, d] - r1) / max(C[si, d], 0.01)
                    dist_to_s1 = abs(C[si, d] - s1) / max(C[si, d], 0.01)
                    dist_to_pivot = abs(C[si, d] - pivot) / max(C[si, d], 0.01)
                    # Low distance = near a key level
                    PIVOT_PROX[si, di] = 1.0 / max(min(dist_to_r1, dist_to_s1, dist_to_pivot), 0.001)

            # =================================================================
            # TEMPORAL STRUCTURE
            # =================================================================

            # 6. Trend age: bars since last direction change
            if d >= 2 and not np.isnan(C[si, d]) and not np.isnan(C[si, d - 1]) and not np.isnan(C[si, d - 2]):
                curr_dir = 1 if C[si, d] > C[si, d - 1] else -1
                if curr_dir != last_direction:
                    trend_start = di
                    last_direction = curr_dir
                TREND_AGE[si, di] = di - trend_start

            # 7. Swing rhythm: CV of swing durations
            if d >= 2 and not np.isnan(H[si, d]) and not np.isnan(L[si, d]):
                # Detect local extremes
                if d >= 3:
                    prev_h2 = H[si, d - 2] if not np.isnan(H[si, d - 2]) else None
                    prev_h1 = H[si, d - 1] if not np.isnan(H[si, d - 1]) else None
                    curr_h = H[si, d]

                    if prev_h2 is not None and prev_h1 is not None:
                        if prev_h1 > prev_h2 and prev_h1 > curr_h:
                            swing_times.append(di - 1)

                    if len(swing_times) >= 4:
                        durations = np.diff(swing_times[-6:])
                        if len(durations) >= 3:
                            mean_dur = np.mean(durations)
                            if mean_dur > 0:
                                cv = np.std(durations) / mean_dur
                                SWING_RHYTHM[si, di] = 1.0 / max(cv, 0.01)  # Low CV = regular rhythm

            # 8. Temporal compression: are swings getting shorter?
            if len(swing_times) >= 4:
                recent_durs = [swing_times[-1] - swing_times[-2],
                              swing_times[-2] - swing_times[-3]]
                older_durs = [swing_times[-3] - swing_times[-4]] if len(swing_times) >= 4 else [10]
                if len(older_durs) > 0 and np.mean(older_durs) > 0:
                    ratio = np.mean(recent_durs) / np.mean(older_durs)
                    TEMPORAL_COMPRESS[si, di] = ratio  # < 1 = compression

            # 9. Duration-weighted momentum: MOM5 * trend_age
            if not np.isnan(C[si, d]) and d >= 5:
                mom5 = (C[si, d] - C[si, d - 5]) / C[si, d - 5] if C[si, d - 5] > 0 else 0
                age = TREND_AGE[si, di]
                if not np.isnan(age) and age > 0:
                    DURATION_MOM[si, di] = np.sign(mom5) * abs(mom5) * min(age, 20) / 20

            # =================================================================
            # PRICE ACTION STRUCTURE
            # =================================================================

            # 10. Consecutive direction score
            if d >= 1 and not np.isnan(C[si, d]) and not np.isnan(C[si, d - 1]):
                if C[si, d] > C[si, d - 1]:
                    consecutive_up += 1
                    consecutive_down = 0
                else:
                    consecutive_down += 1
                    consecutive_up = 0
                # Signed: positive = consecutive up, negative = consecutive down
                CONSEC_DIR[si, di] = consecutive_up - consecutive_down

            # 11. Body consistency: fraction of same-direction bodies in last 10 bars
            if d >= 10:
                dirs = []
                for k in range(10):
                    dd = d - k
                    if dd >= 0 and not np.isnan(C[si, dd]) and not np.isnan(O[si, dd]):
                        dirs.append(1 if C[si, dd] > O[si, dd] else -1)
                if len(dirs) >= 5:
                    # Consistency = |sum| / count
                    BODY_CONSISTENCY[si, di] = abs(sum(dirs)) / len(dirs) * np.sign(sum(dirs))

            # 12. Inside/outside bar pattern
            if d >= 1 and not np.isnan(H[si, d]) and not np.isnan(L[si, d]):
                if not np.isnan(H[si, d - 1]) and not np.isnan(L[si, d - 1]):
                    if H[si, d] <= H[si, d - 1] and L[si, d] >= L[si, d - 1]:
                        BAR_PATTERN[si, di] = -1  # Inside bar (compression)
                    elif H[si, d] > H[si, d - 1] and L[si, d] < L[si, d - 1]:
                        BAR_PATTERN[si, di] = 1  # Outside bar (expansion)
                    else:
                        BAR_PATTERN[si, di] = 0

            # 13. Gap score
            if d >= 1 and not np.isnan(O[si, d]) and not np.isnan(C[si, d - 1]):
                gap = (O[si, d] - C[si, d - 1]) / C[si, d - 1] * 100
                # Gap fill: did price return to yesterday's close?
                if not np.isnan(L[si, d]) and not np.isnan(H[si, d]):
                    gap_filled = (L[si, d] <= C[si, d - 1]) if gap > 0 else (H[si, d] >= C[si, d - 1])
                    GAP_SCORE[si, di] = gap if not gap_filled else 0  # Only unfilled gaps matter

            # 14. Shadow balance
            if d >= 10 and not np.isnan(C[si, d]):
                upper_shadows = []
                lower_shadows = []
                for k in range(10):
                    dd = d - k
                    if dd >= 0 and not np.isnan(H[si, dd]) and not np.isnan(L[si, dd]):
                        body_top = max(C[si, dd], O[si, dd]) if not np.isnan(C[si, dd]) and not np.isnan(O[si, dd]) else H[si, dd]
                        body_bot = min(C[si, dd], O[si, dd]) if not np.isnan(C[si, dd]) and not np.isnan(O[si, dd]) else L[si, dd]
                        upper_shadows.append(H[si, dd] - body_top)
                        lower_shadows.append(body_bot - L[si, dd])
                if len(upper_shadows) >= 5:
                    SHADOW_BALANCE[si, di] = (np.mean(lower_shadows) - np.mean(upper_shadows)) / max(np.mean(upper_shadows) + np.mean(lower_shadows), 0.01)
                    # Positive = more lower shadows (support, bullish)

            # 15. Range sequence: is range expanding or contracting?
            if d >= 10:
                ranges = []
                for k in range(10):
                    dd = d - k
                    if dd >= 0 and not np.isnan(H[si, dd]) and not np.isnan(L[si, dd]):
                        ranges.append(H[si, dd] - L[si, dd])
                if len(ranges) >= 5:
                    # Compare last 3 vs first 3
                    recent = np.mean(ranges[:3])
                    older = np.mean(ranges[-3:])
                    if older > 0:
                        RANGE_SEQUENCE[si, di] = recent / older  # > 1 = expanding

    new['POS_52W'] = POS_52W
    new['SPATIAL_TENSION'] = SPATIAL_TENSION
    new['PRICE_CLUSTERING'] = PRICE_CLUSTERING
    new['VWAP_DEV'] = VWAP_DEV
    new['PIVOT_PROX'] = PIVOT_PROX
    new['TREND_AGE'] = TREND_AGE
    new['SWING_RHYTHM'] = SWING_RHYTHM
    new['TEMPORAL_COMPRESS'] = TEMPORAL_COMPRESS
    new['DURATION_MOM'] = DURATION_MOM
    new['CONSEC_DIR'] = CONSEC_DIR
    new['BODY_CONSISTENCY'] = BODY_CONSISTENCY
    new['BAR_PATTERN'] = BAR_PATTERN
    new['GAP_SCORE'] = GAP_SCORE
    new['SHADOW_BALANCE'] = SHADOW_BALANCE
    new['RANGE_SEQUENCE'] = RANGE_SEQUENCE

    print(f"  Structure factors done ({time.time()-t0:.1f}s)", flush=True)

    # =====================================================================
    # Rank normalize
    # =====================================================================
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

    # Spatial
    new['R_POS_52W'] = rank_pct(new['POS_52W'])
    new['R_SPATIAL_TENSION'] = rank_pct(new['SPATIAL_TENSION'])
    new['R_PRICE_CLUSTERING'] = rank_pct(new['PRICE_CLUSTERING'])
    new['R_VWAP_DEV'] = rank_pct(new['VWAP_DEV'])
    new['R_PIVOT_PROX'] = rank_pct(new['PIVOT_PROX'])

    # Temporal
    new['R_TREND_AGE'] = rank_pct(new['TREND_AGE'])
    new['R_SWING_RHYTHM'] = rank_pct(new['SWING_RHYTHM'])

    # Temporal compression: low = compression = potential breakout
    inv_compress = new['TEMPORAL_COMPRESS'].copy()
    mask = ~np.isnan(inv_compress)
    if mask.any():
        mn, mx = np.nanmin(inv_compress), np.nanmax(inv_compress)
        if mx > mn:
            inv_compress[mask] = mx - inv_compress[mask] + mn
    new['R_TEMP_COMPRESS_INV'] = rank_pct(inv_compress)

    new['R_DURATION_MOM'] = rank_pct(new['DURATION_MOM'])

    # Price action
    new['R_BODY_CONSISTENCY'] = rank_pct(new['BODY_CONSISTENCY'])
    new['R_SHADOW_BALANCE'] = rank_pct(new['SHADOW_BALANCE'])
    new['R_GAP_SCORE'] = rank_pct(new['GAP_SCORE'])

    # Range expansion (high = expanding = trending)
    new['R_RANGE_EXPAND'] = rank_pct(new['RANGE_SEQUENCE'])

    # Composite: Spatial compression + Temporal compression + PA consistency
    composite = np.full((NS, ND), np.nan)
    for di in range(MIN_TRAIN, ND):
        s = new['R_SPATIAL_TENSION'][:, di]
        t = new['R_TEMP_COMPRESS_INV'][:, di]
        p = new['R_BODY_CONSISTENCY'][:, di]
        mask = ~np.isnan(s) & ~np.isnan(t) & ~np.isnan(p)
        if mask.sum() >= 50:
            composite[mask, di] = (s[mask] + t[mask] + p[mask]) / 3.0
    new['R_STRUCT_COMPOSITE'] = rank_pct(composite)

    print(f"  Total structure factors: {len(new)}", flush=True)
    return new


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V28 — High-Dimensional Price Structure Analysis", flush=True)
    print("  Spatial + Temporal + Price Action = 3D structural view", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Load existing factors
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

    # V28 Structure factors
    struct_factors = compute_structure_factors(NS, ND, C, O, H, L, V)
    all_factors = {**v11_all, **struct_factors}

    print(f"\n  Total factors: {len(all_factors)}", flush=True)

    results = []

    # Baseline
    bwp = {'R_BWP_BNW': 0.3, 'R_TENSION': 0.3,
            'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2}
    for top_n in [1, 2]:
        for atr in [1.0, 1.2, 1.5]:
            r = backtest_v7c(bwp, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=top_n, rebalance_days=10, atr_stop_mult=atr)
            if r:
                r['test'] = f'BwpBNW_T{top_n}_A{atr}'
                results.append(r)
    print(f"  Baseline done", flush=True)

    # =====================================================================
    # SINGLE FACTOR TESTS
    # =====================================================================
    print(f"\n  === STRUCTURE SINGLE FACTOR TESTS ===", flush=True)
    for fname in ['R_POS_52W', 'R_SPATIAL_TENSION', 'R_PRICE_CLUSTERING',
                  'R_VWAP_DEV', 'R_TREND_AGE', 'R_SWING_RHYTHM',
                  'R_TEMP_COMPRESS_INV', 'R_DURATION_MOM', 'R_BODY_CONSISTENCY',
                  'R_SHADOW_BALANCE', 'R_GAP_SCORE', 'R_RANGE_EXPAND',
                  'R_STRUCT_COMPOSITE']:
        r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                        top_n=3, rebalance_days=10, atr_stop_mult=1.5)
        if r:
            print(f"  {fname:<25s}: Ann={r['ann']:+7.1f}% WR={r['wr']:5.1f}% "
                  f"Edge={r['edge']:+5.2f}% DD={r['max_dd']:5.1f}%", flush=True)

    # =====================================================================
    # COMBINATION TESTS
    # =====================================================================
    portfolios = {
        # Spatial: position + clustering + VWAP
        'Spc_pos': {'R_POS_52W': 0.34, 'R_PRICE_CLUSTERING': 0.33, 'R_VWAP_DEV': 0.33},
        # Temporal: trend age + compression + duration momentum
        'Tmp_age': {'R_TREND_AGE': 0.34, 'R_TEMP_COMPRESS_INV': 0.33, 'R_DURATION_MOM': 0.33},
        # PA: body consistency + shadow balance + range expand
        'PA_body': {'R_BODY_CONSISTENCY': 0.34, 'R_SHADOW_BALANCE': 0.33, 'R_RANGE_EXPAND': 0.33},
        # Cross-dimension: spatial + temporal + PA (one from each)
        '3D_stp': {'R_SPATIAL_TENSION': 0.34, 'R_DURATION_MOM': 0.33, 'R_BODY_CONSISTENCY': 0.33},
        # Structure composite + BwpBNW
        'SC_bwp': {'R_STRUCT_COMPOSITE': 0.3, 'R_BWP_BNW': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2},
        # Position + momentum + squeeze
        'SP_sqz': {'R_POS_52W': 0.3, 'R_BB_WIDTH_PCT_INV': 0.3,
                   'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        # Gap + VWAP + trend age
        'GV_age': {'R_GAP_SCORE': 0.25, 'R_VWAP_DEV': 0.25,
                   'R_TREND_AGE': 0.25, 'R_R_SQUARED': 0.25},
        # Full structure
        'Full': {'R_SPATIAL_TENSION': 0.15, 'R_DURATION_MOM': 0.15,
                 'R_BODY_CONSISTENCY': 0.15, 'R_SHADOW_BALANCE': 0.15,
                 'R_TENSION': 0.2, 'R_R_SQUARED': 0.2},
        # Structure + Kalman
        'SK_vel': {'R_STRUCT_COMPOSITE': 0.25, 'R_KALMAN_VEL_PCT': 0.25,
                   'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        # Temporal + HMM
        'TH_regime': {'R_DURATION_MOM': 0.25, 'R_HMM_REGIME_SCORE': 0.25,
                      'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
        # Spatial + DMD (spatial compression + spectral)
        'SD_bull': {'R_SPATIAL_TENSION': 0.25, 'R_DMD_BULL_RATIO': 0.25,
                    'R_TENSION': 0.25, 'R_R_SQUARED': 0.25},
    }

    for pname, weights in portfolios.items():
        missing = [f for f in weights if f not in all_factors]
        if missing:
            print(f"  SKIP {pname}: missing {missing}", flush=True)
            continue
        for top_n in [1, 2]:
            for atr in [1.0, 1.2, 1.5]:
                r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=top_n, rebalance_days=10, atr_stop_mult=atr)
                if r:
                    r['test'] = f'{pname}_T{top_n}_A{atr}'
                    results.append(r)
        print(f"  {pname} done", flush=True)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  TOP 40 RESULTS (V28 STRUCTURE)", flush=True)
    print(f"  {'Test':<30s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*80}", flush=True)
    for r in results[:40]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<30s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    for i, r in enumerate(results[:5]):
        print(f"\n  Year-by-year #{i+1}: {r['test']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
