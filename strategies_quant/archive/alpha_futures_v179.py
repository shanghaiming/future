"""
Alpha Futures V179 — Adaptive Signal Strength & Regime-Weighted Scoring
==============================================================================
V177 champion: Dual-side short_mirror, +187%/-16%, R/M=11.41.
V178 tested exit optimizations — ALL worse than hold=1.

V179 hypothesis: Instead of binary signals (in/out), use CONTINUOUS signal
strength score to weight position sizing. Stronger signals get bigger positions.

New ideas:
  1. Signal Strength Score: composite of ROC magnitude, Z-score magnitude,
     volume confirmation (vol > 20-day avg), ATR normalization (lower = better).
  2. Regime-Weighted Position Sizing: ADX-like trend strength indicator,
     scale up in trending regime, scale down in choppy regime,
     plus equity slope tracking.
  3. Volume-Weighted Signal Ranking: rank candidates by score * volume_ratio.

Test matrix:
  1. BASELINE: V177 champion config (short_mirror, atr<10, corr=0.5, top_n=3)
  2. SCORE_WEIGHTED: Use composite score for ranking
  3. REGIME_SIZING: Apply ADX-based regime sizing on top of DD sizing
  4. VOL_CONFIRM: Volume-weighted signal ranking
  5. COMBINED: Score + Regime + Vol together
"""
import sys, os, time, warnings
import numpy as np
import talib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

MULT = {'agfi': 15, 'alfi': 5, 'aufi': 1000, 'bufi': 10, 'cufi': 5, 'fufi': 10,
        'rbfi': 10, 'znfi': 5, 'nifi': 1, 'hcfi': 10, 'spfi': 10, 'ssfi': 5,
        'sffi': 5, 'smfi': 5, 'pbfi': 5, 'snfi': 1, 'rufi': 10, 'wrffi': 10,
        'afi': 10, 'bfi': 10, 'bbfi': 500, 'cffi': 5, 'cfi': 10, 'csfi': 10,
        'ebfi': 5, 'egfi': 10, 'fbfi': 500, 'ifi': 100, 'jfi': 100, 'jmfi': 60,
        'lfi': 5, 'mfi': 10, 'pgfi': 20, 'ppfi': 5, 'vfi': 5, 'yfi': 10,
        'pfi': 10, 'jdfi': 5, 'lhfi': 16, 'pkfi': 5, 'rrfi': 20, 'lrfi': 20,
        'jrfi': 20, 'pmfi': 20, 'whfi': 20, 'rsfi': 20, 'cjfi': 10, 'mafi': 10,
        'apfi': 10, 'cyfi': 5, 'fgfi': 20, 'oifi': 10, 'pfifi': 5, 'rmfi': 10,
        'srfi': 10, 'tafi': 5, 'safi': 20, 'urfi': 20, 'scfi': 1000, 'lufi': 10,
        'bcfi': 5, 'nrfi': 1, 'lgfi': 20, 'brfi': 5, 'lcfi': 1, 'sifi': 5,
        'ni': 1, 'tai': 5}
DEF_MULT = 10
COMM = 0.0003


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0: return -100.0
    return (final / initial) ** (1.0 / (n_days / 252)) * 100 - 100


def main():
    print("=" * 130)
    print("  V179 — Adaptive Signal Strength & Regime-Weighted Scoring")
    print("  Continuous signal strength for position sizing instead of binary in/out")
    print("=" * 130)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    print(f"  {NS} commodities, {ND} days")

    # ===================== PRECOMPUTE =====================
    print("\n[Precompute]...", flush=True)
    t0 = time.time()

    RET = np.full((NS, ND), np.nan)
    ROC5 = np.full((NS, ND), np.nan)
    ROC10 = np.full((NS, ND), np.nan)
    ROC20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100
        ROC5[si] = talib.ROC(c, timeperiod=5)
        ROC10[si] = talib.ROC(c, timeperiod=10)
        ROC20[si] = talib.ROC(c, timeperiod=20)

    ATR14 = np.full((NS, ND), np.nan)
    ADX14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        h = H[si].astype(np.float64)
        l = L[si].astype(np.float64)
        c = C[si].astype(np.float64)
        ATR14[si] = talib.ATR(h, l, c, timeperiod=14)
        ADX14[si] = talib.ADX(h, l, c, timeperiod=14)

    ATR_NORM = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            atr = ATR14[si, di]
            cp = C[si, di]
            if not np.isnan(atr) and not np.isnan(cp) and cp > 0:
                ATR_NORM[si, di] = atr / cp * 100

    ZSCORE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = RET[si, di-20:di]
            v = rets[~np.isnan(rets)]
            if len(v) < 10: continue
            s = np.std(v, ddof=1)
            if s > 0 and not np.isnan(RET[si, di]):
                ZSCORE[si, di] = (RET[si, di] - np.mean(v)) / s

    # Volume ratios: current volume / 20-day average volume
    VOL_RATIO20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        vol = V[si].astype(np.float64)
        for di in range(20, ND):
            v_now = vol[di]
            if np.isnan(v_now) or v_now <= 0: continue
            avg20 = np.nanmean(vol[di-20:di])
            if avg20 > 0 and not np.isnan(avg20):
                VOL_RATIO20[si, di] = v_now / avg20

    OV_GAP = np.full((NS, ND), np.nan)
    ID_RET = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            o, c = O[si, di], C[si, di]
            if not np.isnan(o) and not np.isnan(c):
                if di > 0 and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                    OV_GAP[si, di] = (o - C[si, di-1]) / C[si, di-1] * 100
                if o > 0: ID_RET[si, di] = (c - o) / o * 100

    # ===================== REGIME INDICATORS =====================
    print("  Computing regime indicators...", flush=True)

    BREADTH = np.full(ND, np.nan)
    for di in range(5, ND):
        pos_count = 0; total = 0
        for si in range(NS):
            r = ROC5[si, di]
            if not np.isnan(r):
                total += 1
                if r > 0: pos_count += 1
        if total > 0:
            BREADTH[di] = pos_count / total

    MKT_RET = np.full(ND, np.nan)
    for di in range(ND):
        rets_day = RET[:, di]
        valid = rets_day[~np.isnan(rets_day)]
        if len(valid) > 10:
            MKT_RET[di] = np.mean(valid)

    MKT_VOL = np.full(ND, np.nan)
    for di in range(20, ND):
        window = MKT_RET[di-20:di]
        valid = window[~np.isnan(window)]
        if len(valid) >= 10:
            MKT_VOL[di] = np.std(valid, ddof=1)

    valid_vols = MKT_VOL[~np.isnan(MKT_VOL)]
    VOL_MEDIAN = np.median(valid_vols) if len(valid_vols) > 0 else 1.0
    VOL_P50 = np.percentile(valid_vols, 50) if len(valid_vols) > 0 else 1.0
    VOL_P75 = np.percentile(valid_vols, 75) if len(valid_vols) > 0 else 1.5

    print(f"  Market vol: median={VOL_MEDIAN:.4f}%, P50={VOL_P50:.4f}%, P75={VOL_P75:.4f}%")
    print(f"  Precompute done ({time.time()-t0:.1f}s)")

    # ===================== SIGNAL DEFINITIONS =====================

    def sig_v121_long(di, edi):
        """V121 long signal with raw components for scoring."""
        c = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((roc * zs, s, ep, 'v121', roc, zs))
        return c

    def sig_v121_short(di, edi):
        """V121 short signal (mirror) with raw components."""
        c = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc >= -1.0 or zs >= -1.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc >= rp: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((abs(roc * zs), s, ep, 'v121_short', roc, zs))
        return c

    def sig_union_long(di, edi):
        """Union of v121 + ov_id + final_flag for long."""
        all_sigs = {}
        for item in sig_v121_long(di, edi):
            sc, s, ep, st, roc, zs = item
            if s not in all_sigs: all_sigs[s] = [0, ep, [], roc, zs]
            all_sigs[s][0] += sc * 3
            all_sigs[s][2].append('v121')
        for item in sig_ov_id(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs: all_sigs[s] = [0, ep, [], 0, 0]
            all_sigs[s][0] += sc * 2
            all_sigs[s][2].append('ov_id')
        for item in sig_final_flag(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs: all_sigs[s] = [0, ep, [], 0, 0]
            all_sigs[s][0] += sc
            all_sigs[s][2].append('ff')
        return [(sc, s, ep, '+'.join(sigs), all_sigs[s][3], all_sigs[s][4])
                for s, (sc, ep, sigs, _, _) in all_sigs.items()]

    def sig_ov_id(di, edi):
        c = []
        for s in range(NS):
            ov = OV_GAP[s, di]; idr = ID_RET[s, di]; roc = ROC5[s, di]
            if any(np.isnan(x) for x in [ov, idr, roc]): continue
            if ov <= 0.3 or idr <= 0.3 or roc <= 1.0: continue
            zs = ZSCORE[s, di]
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            z_bonus = zs if not np.isnan(zs) and zs > 1.0 else 1.0
            c.append(((ov + idr) * roc * z_bonus * 2, s, ep, 'ov_id'))
        return c

    def sig_final_flag(di, edi):
        c = []
        for s in range(NS):
            roc20 = ROC20[s, di]
            if np.isnan(roc20) or roc20 <= 5.0 or di < 6: continue
            h5 = H[s, di-4:di+1]; l5 = L[s, di-4:di+1]
            if any(np.isnan(x) for x in h5) or any(np.isnan(x) for x in l5): continue
            r5 = np.max(h5) - np.min(l5)
            atr = ATR14[s, di]
            if np.isnan(atr) or atr <= 0 or r5 > atr * 3.0: continue
            h4 = np.max(H[s, di-4:di])
            cp = C[s, di]
            if np.isnan(cp) or cp <= h4: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((roc20 * (cp - h4) / atr, s, ep, 'ff'))
        return c

    # ===================== COMPOSITE SCORE =====================

    def compute_signal_score(si, di, raw_score, roc_val, zs_val, mode='basic'):
        """
        Compute a composite signal strength score (0 to 1+).
        Components:
          - ROC magnitude: stronger momentum = higher score
          - Z-score magnitude: more unusual = higher score
          - Volume confirmation: vol > 20-day avg boosts score
          - ATR normalization: lower ATR% = more stable = higher score
        """
        components = []

        # 1. ROC magnitude score: ROC > 1% baseline, scale up to ~5%+ = 1.0
        if not np.isnan(roc_val):
            roc_mag = abs(roc_val)
            roc_score = np.clip((roc_mag - 1.0) / 4.0, 0.0, 1.0)  # 1%=0, 5%=1
            components.append(roc_score * 0.25)

        # 2. Z-score magnitude: Z > 1.5 baseline, scale up to ~3+ = 1.0
        if not np.isnan(zs_val):
            z_mag = abs(zs_val)
            z_score = np.clip((z_mag - 1.5) / 1.5, 0.0, 1.0)  # 1.5=0, 3.0=1
            components.append(z_score * 0.25)

        # 3. Volume confirmation
        vr = VOL_RATIO20[si, di]
        if not np.isnan(vr):
            # vol_ratio > 1 means above average, >2 means strong
            vol_score = np.clip((vr - 0.8) / 1.5, 0.0, 1.0)  # 0.8=0, 2.3=1
            components.append(vol_score * 0.25)

        # 4. ATR normalization: lower ATR% = more stable = better
        an = ATR_NORM[si, di]
        if not np.isnan(an):
            # ATR% < 3% = very stable = 1, > 8% = volatile = 0
            atr_score = np.clip((8.0 - an) / 5.0, 0.0, 1.0)
            components.append(atr_score * 0.25)

        if not components:
            return 1.0
        return max(0.3, sum(components) / sum(c > 0 for c in components) if components else 0.5)

    # ===================== REGIME INDICATOR =====================

    def compute_regime_score(di, daily_eq, high_water):
        """
        Market regime score: 0.5 = neutral, >0.5 = trending, <0.5 = choppy.
        Uses:
          - ADX-like breadth (fraction of commodities with strong ADX)
          - Equity slope
          - Market volatility
        """
        scores = []

        # 1. Breadth: how many commodities are moving in the same direction
        bth = BREADTH[di]
        if not np.isnan(bth):
            # Breadth near 0 or 1 = strong trend; near 0.5 = choppy
            # Score = max(breadth, 1-breadth) mapped to [0,1]
            trend_strength = max(bth, 1.0 - bth)
            # trend_strength: 0.5=no trend, 1.0=all same direction
            scores.append(np.clip((trend_strength - 0.5) / 0.3, 0.0, 1.0))

        # 2. Market vol: lower vol = more favorable for momentum
        vol = MKT_VOL[di]
        if not np.isnan(vol) and VOL_MEDIAN > 0:
            vol_ratio = vol / VOL_MEDIAN
            scores.append(np.clip((1.5 - vol_ratio) / (1.5 - 0.5), 0.0, 1.0))

        # 3. Equity slope: recent equity performance
        perf_window = 20
        if len(daily_eq) >= perf_window:
            eq_window = np.array(daily_eq[-perf_window:])
            x = np.arange(perf_window)
            try:
                slope = np.polyfit(x, eq_window, 1)[0]
                eq_mean = np.mean(eq_window)
                norm_slope = slope / eq_mean * 100 if eq_mean > 0 else 0
                eq_rets = np.diff(eq_window) / eq_window[:-1] * 100
                eq_rets = eq_rets[np.isfinite(eq_rets)]
                eq_std = np.std(eq_rets) if len(eq_rets) > 5 else 1.0
                z = norm_slope / eq_std if eq_std > 0 else 0
                scores.append(np.clip((z + 1.0) / 2.0, 0.0, 1.0))
            except Exception:
                pass

        # 4. Drawdown-based
        if high_water > 0:
            cur_dd = (daily_eq[-1] - high_water) / high_water
        else:
            cur_dd = 0
        scores.append(np.clip(1.0 + cur_dd / 0.3, 0.0, 1.0))

        return np.mean(scores) if scores else 0.5

    # ===================== HELPERS =====================
    def get_corr(si_a, si_b, di, window=20):
        start_idx = max(0, di - window)
        ret_a = RET[si_a, start_idx:di]
        ret_b = RET[si_b, start_idx:di]
        valid = ~(np.isnan(ret_a) | np.isnan(ret_b))
        n_valid = np.sum(valid)
        if n_valid < 8:
            return 0.5
        ra = ret_a[valid]; rb = ret_b[valid]
        if np.std(ra) == 0 or np.std(rb) == 0:
            return 0.5
        c = np.corrcoef(ra, rb)[0, 1]
        return c if not np.isnan(c) else 0.5

    def dd_size(pv, high_water, tiers):
        if high_water <= 0:
            return tiers[0][1]
        dd = (pv - high_water) / high_water
        for dd_thresh, size_frac in tiers:
            if dd >= -dd_thresh:
                return size_frac
        return tiers[-1][1]

    # ===================== BACKTEST ENGINE =====================
    # mode: 'baseline', 'score_weighted', 'regime_sizing', 'vol_confirm', 'combined'

    def backtest(start_di=MIN_TRAIN, end_di=None,
                 atr_norm_max=10.0, max_corr=0.5,
                 dd_tiers=None,
                 regime_lo=0.5, regime_hi=1.5,
                 hold=1, top_n=3,
                 short_mode='short_mirror',
                 mode='baseline'):
        if end_di is None: end_di = ND
        if dd_tiers is None:
            dd_tiers = [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)]

        cash = float(CASH0)
        positions = []
        trades = []
        daily_eq = []
        high_water = float(CASH0)

        for di in range(start_di, end_di - 1):
            # Mark-to-market
            pv = cash
            for p in positions:
                cp = C[p['si'], di]
                if not np.isnan(cp) and cp > 0:
                    m = MULT.get(p['sym'], DEF_MULT)
                    d = p.get('dir', 1)
                    unrealized = (cp - p['entry_price']) * m * p['lots'] * d
                    pv += p['entry_price'] * m * abs(p['lots']) + unrealized - cp * m * abs(p['lots']) * COMM
            daily_eq.append(pv)
            if pv > high_water:
                high_water = pv

            # Close positions past hold period
            cl = []
            for p in positions:
                if di - p['entry_di'] >= p['hold_days']:
                    ep = C[p['si'], di]
                    if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                    m = MULT.get(p['sym'], DEF_MULT)
                    d = p.get('dir', 1)
                    pnl = (ep - p['entry_price']) * m * p['lots'] * d
                    inv = p['entry_price'] * m * abs(p['lots'])
                    pp = pnl / inv * 100 if inv > 0 else 0
                    if d == 1:
                        cash += ep * m * abs(p['lots']) * (1 - COMM)
                    else:
                        margin = p['entry_price'] * m * abs(p['lots'])
                        cash += margin + pnl - ep * m * abs(p['lots']) * COMM
                    trades.append(pp)
                    cl.append(p)
            for p in cl: positions.remove(p)

            # --- Position sizing ---
            dd_sz = dd_size(pv, high_water, dd_tiers)

            # Regime multiplier (used in baseline mode and all others)
            composite = compute_regime_score(di, daily_eq, high_water)
            regime_mult = regime_lo + composite * (regime_hi - regime_lo)

            pos_size = dd_sz * regime_mult

            # --- Enter positions ---
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue

            held_si = set(p['si'] for p in positions)

            # Get long candidates
            cands_long = sig_v121_long(di, edi)
            # Add union candidates
            cands_union = sig_union_long(di, edi)

            # Filter by ATR
            cands_long_f = [c for c in cands_long
                            if not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < atr_norm_max]
            cands_union_f = [c for c in cands_union
                             if not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < atr_norm_max]

            # Get short candidates
            cands_short = sig_v121_short(di, edi)
            cands_short_f = [c for c in cands_short
                             if not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < atr_norm_max]

            # Apply mode-specific scoring and ranking
            def apply_mode_scoring(cands, is_short=False):
                """Apply mode-specific scoring/ranking to candidates."""
                if not cands:
                    return cands

                if mode == 'baseline':
                    # Original V177: just sort by raw_score (roc * |z|)
                    cands.sort(key=lambda x: -x[0])
                    return cands

                elif mode == 'score_weighted' or mode == 'combined':
                    # Composite signal strength score
                    scored = []
                    for item in cands:
                        raw_sc, s, ep, sig, roc_val, zs_val = item
                        ss = compute_signal_score(s, di, raw_sc, roc_val, zs_val)
                        # Position sizing multiplier: 0.5 to 1.5 based on score
                        size_mult = 0.5 + ss * 1.0  # score 0->0.5x, score 1->1.5x
                        # Score for ranking = raw * signal_strength
                        new_score = raw_sc * (0.5 + ss * 0.5)
                        scored.append((new_score, s, ep, sig, size_mult, is_short))
                    scored.sort(key=lambda x: -x[0])
                    return scored

                elif mode == 'regime_sizing':
                    # ADX-based regime sizing
                    scored = []
                    for item in cands:
                        raw_sc, s, ep, sig, roc_val, zs_val = item
                        # Check ADX for this specific commodity
                        adx = ADX14[s, di]
                        if not np.isnan(adx):
                            # ADX > 25 = trending, ADX < 20 = choppy
                            adx_mult = np.clip((adx - 15.0) / 20.0, 0.3, 1.5)
                        else:
                            adx_mult = 1.0
                        scored.append((raw_sc, s, ep, sig, adx_mult, is_short))
                    scored.sort(key=lambda x: -x[0])
                    return scored

                elif mode == 'vol_confirm':
                    # Volume-weighted ranking
                    scored = []
                    for item in cands:
                        raw_sc, s, ep, sig, roc_val, zs_val = item
                        vr = VOL_RATIO20[s, di]
                        vol_mult = 1.0
                        if not np.isnan(vr):
                            # Higher volume = more reliable signal
                            vol_mult = np.clip(vr, 0.5, 2.5)
                        scored.append((raw_sc * vol_mult, s, ep, sig, vol_mult, is_short))
                    scored.sort(key=lambda x: -x[0])
                    return scored

                return cands

            # Build entries list
            entries = []

            # For modes with extended tuple (score, si, ep, sig, size_mult, is_short)
            # For baseline, tuple is (raw_sc, si, ep, sig, roc, zs)
            if mode == 'baseline':
                # Long entries
                cands_long_f.sort(key=lambda x: -x[0])
                best_long = None
                for c in cands_long_f:
                    if c[1] not in held_si:
                        best_long = c
                        break

                cands_union_f.sort(key=lambda x: -x[0])
                best_union = None
                for c in cands_union_f:
                    if c[1] not in held_si:
                        best_union = c
                        break

                if best_long and best_union:
                    if best_long[1] == best_union[1]:
                        entries.append((best_long[0], best_long[1], best_long[2],
                                        'v121+union', pos_size * 1.5, 1))
                    else:
                        corr = get_corr(best_long[1], best_union[1], di)
                        if corr < max_corr:
                            entries.append((best_long[0], best_long[1], best_long[2],
                                            'v121', pos_size, 1))
                            entries.append((best_union[0], best_union[1], best_union[2],
                                            'union', pos_size, 1))
                        else:
                            best = best_long if best_long[0] >= best_union[0] else best_union
                            entries.append((best[0], best[1], best[2], 'best', pos_size, 1))
                elif best_long:
                    entries.append((best_long[0], best_long[1], best_long[2], 'v121', pos_size, 1))
                elif best_union:
                    entries.append((best_union[0], best_union[1], best_union[2], 'union', pos_size, 1))

                # Short entries
                if short_mode != 'long_only' and len(positions) < top_n:
                    cands_short_f.sort(key=lambda x: -x[0])
                    held_si = set(p['si'] for p in positions)
                    best_short = None
                    for c in cands_short_f:
                        if c[1] not in held_si:
                            best_short = c
                            break
                    if best_short:
                        entries.append((best_short[0], best_short[1], best_short[2],
                                       'v121_short', pos_size, -1))

            else:
                # Non-baseline modes: apply scoring to all candidates
                scored_long = apply_mode_scoring(cands_long_f, is_short=False)
                scored_union = apply_mode_scoring(cands_union_f, is_short=False)

                # Merge long + union, pick best
                all_scored = []
                for item in scored_long:
                    sc, s, ep, sig, size_m, is_s = item
                    all_scored.append((sc, s, ep, 'v121', size_m, 1))
                for item in scored_union:
                    sc, s, ep, sig, size_m, is_s = item
                    all_scored.append((sc * 1.5, s, ep, 'union', size_m, 1))  # union boost

                # Deduplicate: keep best score per symbol
                best_per_sym = {}
                for sc, s, ep, sig, sm, d in all_scored:
                    if s not in best_per_sym or sc > best_per_sym[s][0]:
                        best_per_sym[s] = (sc, s, ep, sig, sm, d)

                long_entries = sorted(best_per_sym.values(), key=lambda x: -x[0])

                # Apply correlation filter to top entries
                long_final = []
                for entry in long_entries:
                    sc, s, ep, sig, sm, d = entry
                    if s in set(p['si'] for p in positions): continue
                    if len(positions) + len(long_final) >= top_n: break
                    # Check correlation with already selected
                    corr_ok = True
                    for le in long_final:
                        corr = get_corr(le[1], s, di)
                        if corr >= max_corr:
                            corr_ok = False
                            break
                    if corr_ok:
                        long_final.append(entry)

                for sc, s, ep, sig, sm, d in long_final:
                    entries.append((sc, s, ep, sig, pos_size * sm, d))

                # Short entries
                if short_mode != 'long_only' and len(positions) + len(entries) < top_n:
                    scored_short = apply_mode_scoring(cands_short_f, is_short=True)
                    held_si = set(p['si'] for p in positions)
                    held_si |= set(e[1] for e in entries)
                    for sc, s, ep, sig, sm, is_s in scored_short:
                        if s in held_si: continue
                        if len(positions) + len(entries) >= top_n: break
                        entries.append((sc, s, ep, 'v121_short', pos_size * sm, -1))
                        break  # take top 1 short

            # Execute entries
            cash_snapshot = cash
            n_planned = len(entries)
            for sc, s, pr, sig_str, pct, d in entries:
                if s in set(p['si'] for p in positions): continue
                if len(positions) >= top_n: break
                cap = cash_snapshot * pct / max(n_planned, 1)
                sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                if d == 1:  # long
                    ct = max(1, int(cap / (pr * m * (1 + COMM))))
                    ci = pr * m * ct * (1 + COMM)
                    if ci > cash:
                        ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                        ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                    if ct <= 0 or ci <= 0 or ci > cash: continue
                    cash -= ci
                else:  # short
                    ct = max(1, int(cap / (pr * m * (1 + COMM))))
                    ci = pr * m * ct * (1 + COMM)
                    if ci > cash:
                        ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                        ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                    if ct <= 0 or ci <= 0 or ci > cash: continue
                    cash -= ci  # lock up margin
                positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                  'lots': ct, 'dir': d, 'sym': sym, 'hold_days': hold,
                                  'sig': sig_str, 'score': sc})

        # Close remaining
        for p in positions:
            ae = end_di - 1
            ep = C[p['si'], min(ae, ND-1)]
            if np.isnan(ep) or ep <= 0: ep = p['entry_price']
            m = MULT.get(p['sym'], DEF_MULT)
            d = p.get('dir', 1)
            if d == 1:
                cash += ep * m * abs(p['lots']) * (1 - COMM)
            else:
                pnl = (ep - p['entry_price']) * m * p['lots'] * d
                margin = p['entry_price'] * m * abs(p['lots'])
                cash += margin + pnl - ep * m * abs(p['lots']) * COMM

        nd = end_di - start_di
        ann = annual_return(cash, CASH0, nd)
        wr = np.mean([1 if t > 0 else 0 for t in trades]) * 100 if trades else 0
        nt = len(trades)
        if daily_eq:
            eq = np.array(daily_eq); pk = np.maximum.accumulate(eq)
            mdd = np.min((eq - pk) / pk * 100)
            r = np.diff(eq) / eq[:-1]
            r = np.where(np.isfinite(r), r, 0)
            sh = np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0
        else:
            mdd = 0; sh = 0
        return {'ann': ann, 'wr': wr, 'n': nt, 'mdd': mdd, 'sharpe': sh, 'final': cash}

    # ===================== PRINTING HELPERS =====================
    def pr(r, label=""):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  {label:95s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d}")

    def walk_forward(label="", **kwargs):
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue
            r = backtest(start_di=ys, end_di=ye, **kwargs)
            res[yr] = r
        return res

    def print_wf(wf_res, label=""):
        pos = sum(1 for r in wf_res.values() if r['ann'] > 0)
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"    {label}")
        print(f"      {pos}/6 pos | Avg={avg_ann:>+7.0f}% | WorstWfMDD={worst_mdd:>5.0f}%")
        print(f"      {ws}")

    # ===================== CONFIG =====================
    DD_AGGR100 = [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)]

    all_results = []

    # ===================== TEST 1: BASELINE (V177 champion) =====================
    print("\n" + "=" * 130)
    print("  TEST 1: BASELINE — V177 Champion (short_mirror, atr<10, corr=0.5, top_n=3)")
    print("=" * 130)

    r_base = backtest(atr_norm_max=10.0, max_corr=0.5,
                      dd_tiers=DD_AGGR100,
                      regime_lo=0.5, regime_hi=1.5,
                      hold=1, top_n=3,
                      short_mode='short_mirror',
                      mode='baseline')
    pr(r_base, "BASELINE: V177 short_mirror atr<10% corr=0.5")
    all_results.append({**r_base, 'label': 'BASELINE_V177', 'mode': 'baseline'})

    # ===================== TEST 2: SCORE_WEIGHTED =====================
    print("\n" + "=" * 130)
    print("  TEST 2: SCORE_WEIGHTED — Composite signal strength scoring")
    print("=" * 130)

    for anm in [10.0, 12.0]:
        for mc in [0.5, 0.7]:
            for tn in [3, 4]:
                r = backtest(atr_norm_max=anm, max_corr=mc,
                             dd_tiers=DD_AGGR100,
                             regime_lo=0.5, regime_hi=1.5,
                             hold=1, top_n=tn,
                             short_mode='short_mirror',
                             mode='score_weighted')
                lbl = f"SCORE_WEIGHTED atr<{anm:.0f}% corr={mc} top_n={tn}"
                pr(r, lbl)
                all_results.append({**r, 'label': lbl, 'mode': 'score_weighted',
                                    'atr_norm_max': anm, 'max_corr': mc, 'top_n': tn})

    # ===================== TEST 3: REGIME_SIZING =====================
    print("\n" + "=" * 130)
    print("  TEST 3: REGIME_SIZING — ADX-based regime position sizing")
    print("=" * 130)

    for anm in [10.0, 12.0]:
        for mc in [0.5, 0.7]:
            for tn in [3, 4]:
                r = backtest(atr_norm_max=anm, max_corr=mc,
                             dd_tiers=DD_AGGR100,
                             regime_lo=0.5, regime_hi=1.5,
                             hold=1, top_n=tn,
                             short_mode='short_mirror',
                             mode='regime_sizing')
                lbl = f"REGIME_SIZING atr<{anm:.0f}% corr={mc} top_n={tn}"
                pr(r, lbl)
                all_results.append({**r, 'label': lbl, 'mode': 'regime_sizing',
                                    'atr_norm_max': anm, 'max_corr': mc, 'top_n': tn})

    # ===================== TEST 4: VOL_CONFIRM =====================
    print("\n" + "=" * 130)
    print("  TEST 4: VOL_CONFIRM — Volume-weighted signal ranking")
    print("=" * 130)

    for anm in [10.0, 12.0]:
        for mc in [0.5, 0.7]:
            for tn in [3, 4]:
                r = backtest(atr_norm_max=anm, max_corr=mc,
                             dd_tiers=DD_AGGR100,
                             regime_lo=0.5, regime_hi=1.5,
                             hold=1, top_n=tn,
                             short_mode='short_mirror',
                             mode='vol_confirm')
                lbl = f"VOL_CONFIRM atr<{anm:.0f}% corr={mc} top_n={tn}"
                pr(r, lbl)
                all_results.append({**r, 'label': lbl, 'mode': 'vol_confirm',
                                    'atr_norm_max': anm, 'max_corr': mc, 'top_n': tn})

    # ===================== TEST 5: COMBINED =====================
    print("\n" + "=" * 130)
    print("  TEST 5: COMBINED — Score + Regime + Vol together")
    print("=" * 130)

    for anm in [10.0, 12.0]:
        for mc in [0.5, 0.7]:
            for tn in [3, 4]:
                r = backtest(atr_norm_max=anm, max_corr=mc,
                             dd_tiers=DD_AGGR100,
                             regime_lo=0.5, regime_hi=1.5,
                             hold=1, top_n=tn,
                             short_mode='short_mirror',
                             mode='combined')
                lbl = f"COMBINED atr<{anm:.0f}% corr={mc} top_n={tn}"
                pr(r, lbl)
                all_results.append({**r, 'label': lbl, 'mode': 'combined',
                                    'atr_norm_max': anm, 'max_corr': mc, 'top_n': tn})

    # ===================== WALK-FORWARD =====================
    print("\n" + "=" * 130)
    print("  WALK-FORWARD: Baseline + Top 3 configs by R/M")
    print("=" * 130)

    base_ann = r_base['ann']; base_mdd = r_base['mdd']
    base_rm = abs(base_ann / base_mdd) if base_mdd != 0 else 0
    print(f"\n  Baseline: Ann={base_ann:+.0f}% | MDD={base_mdd:.0f}% | R/M={base_rm:.2f}")

    # WF baseline
    wf_base = walk_forward(label="BASELINE WF",
                           atr_norm_max=10.0, max_corr=0.5,
                           dd_tiers=DD_AGGR100,
                           regime_lo=0.5, regime_hi=1.5,
                           hold=1, top_n=3,
                           short_mode='short_mirror',
                           mode='baseline')
    print_wf(wf_base, "BASELINE V177")

    # Top 5 by R/M (excluding baseline)
    ranked = sorted(all_results, key=lambda x: abs(x.get('ann', 0) / x.get('mdd', -1)), reverse=True)
    top5 = [r for r in ranked if r.get('mode') != 'baseline'][:5]
    for r in top5:
        lbl = r['label']; md = r.get('mode', 'baseline')
        an = r.get('atr_norm_max', 10.0); mc = r.get('max_corr', 0.5)
        tn = r.get('top_n', 3)
        print(f"\n  Walk-forward: {lbl}")
        wf = walk_forward(label=f"WF {lbl}",
                          atr_norm_max=an, max_corr=mc,
                          dd_tiers=DD_AGGR100,
                          regime_lo=0.5, regime_hi=1.5,
                          hold=1, top_n=tn,
                          short_mode='short_mirror',
                          mode=md)
        print_wf(wf, lbl)

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 130)
    print("  V179 FINAL SUMMARY: Adaptive Signal Strength & Regime-Weighted Scoring")
    print("=" * 130)

    print(f"\n  {'Config':60s} | {'Ann':>7s} | {'MDD':>5s} | {'R/M':>5s} | {'WR':>5s} | {'N':>4s} | Delta_R/M")
    print(f"  {'-'*60}-+-{'-'*7}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*4}-+-{'-'*8}")
    print(f"  {'BASELINE (V177 short_mirror atr<10%)':60s} | {base_ann:>+7.0f}% | {base_mdd:>5.0f}% | {base_rm:>5.2f} | {r_base['wr']:>5.1f}% | {r_base['n']:>4d} |    ---")

    for r in ranked[:20]:
        if r.get('mode') == 'baseline': continue
        ann = r['ann']; mdd = r['mdd']
        rm = abs(ann / mdd) if mdd != 0 else 0
        delta = rm - base_rm
        print(f"  {r['label']:60s} | {ann:>+7.0f}% | {mdd:>5.0f}% | {rm:>5.2f} | {r['wr']:>5.1f}% | {r['n']:>4d} | {delta:>+8.2f}")

    # Best per mode
    print(f"\n  --- BEST PER MODE ---")
    for mode_name in ['score_weighted', 'regime_sizing', 'vol_confirm', 'combined']:
        mode_results = [r for r in all_results if r.get('mode') == mode_name]
        if mode_results:
            best = max(mode_results, key=lambda x: abs(x.get('ann', 0) / x.get('mdd', -1)))
            ann = best['ann']; mdd = best['mdd']
            rm = abs(ann / mdd) if mdd != 0 else 0
            delta = rm - base_rm
            print(f"  {mode_name:20s} best: {best['label']}")
            print(f"  {'':20s}       Ann={ann:+.0f}% | MDD={mdd:.0f}% | R/M={rm:.2f} | Delta={delta:+.2f}")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
