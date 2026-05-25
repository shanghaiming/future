"""
Alpha Futures V176 — OI-Enhanced V121 Strategy
==============================================================================
V169 champion: +253%/-15% WF, R/M=16.91 (best risk-adjusted).

V176 adds OI (Open Interest) intelligence:
  A. OI_MOM: OI 5-day change rate — measures capital flow speed
  B. OI_PRICE_DIV: price↑+OI↑ = strong demand, price↑+OI↓ = short covering
  C. VOL_OI_RATIO: volume/OI — market activity vs depth (high = speculation)
  D. OI_SURGE: OI > 2x 20-day average — abnormal position building

Key insight: OI hard filters were tested (V118-V138) and ALL reduced returns
because they reduced trade count. V176 uses OI as a SIGNAL ENHANCER, not a filter:
  - OI rising + price rising → boost signal score (new capital entering)
  - OI falling + price rising → reduce score (short covering, weaker)
  - OI surge → boost score (abnormal institutional activity)

Test matrix:
  oi_mode: [none_baseline, oi_boost, oi_momentum, oi_combined, oi_ranking]
  atr_norm_max: [10, 12]
  max_corr: [0.5, 0.7]
  top_n: [3]

Base: V169 Kitchen Sink sizing, aggro100 DD, regime 0.5-1.5, hold=1.
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
    print("  V169 — Adaptive Vol Threshold + Vol Filter Optimization")
    print("  Pushing V167's vol filter approach further")
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
    for si in range(NS):
        ATR14[si] = talib.ATR(H[si].astype(np.float64), L[si].astype(np.float64),
                               C[si].astype(np.float64), timeperiod=14)

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
    VOL_P90 = np.percentile(valid_vols, 90) if len(valid_vols) > 0 else 2.0

    # ===================== OI INDICATORS =====================
    print("  Computing OI indicators...", flush=True)
    OI_MOM5 = np.full((NS, ND), np.nan)
    OI_MOM20 = np.full((NS, ND), np.nan)
    OI_RATIO20 = np.full((NS, ND), np.nan)  # OI / 20-day avg OI
    OI_PRICE_DIV = np.full((NS, ND), np.nan)  # +1 = price↑OI↑, -1 = price↑OI↓
    VOL_OI_RATIO = np.full((NS, ND), np.nan)
    OI_SURGE = np.full((NS, ND), 0.0)  # 1.0 if OI > 2x avg

    for si in range(NS):
        oi = OI[si].astype(np.float64)
        close = C[si].astype(np.float64)
        vol = V[si].astype(np.float64)
        for di in range(20, ND):
            oi_now = oi[di]; oi_5 = oi[di-5]; oi_20 = oi[di-20]
            if np.isnan(oi_now) or np.isnan(oi_5) or np.isnan(oi_20) or oi_20 <= 0 or oi_5 <= 0:
                continue
            OI_MOM5[si, di] = (oi_now / oi_5 - 1) * 100
            OI_MOM20[si, di] = (oi_now / oi_20 - 1) * 100
            avg_oi_20 = np.nanmean(oi[di-20:di])
            if avg_oi_20 > 0:
                OI_RATIO20[si, di] = oi_now / avg_oi_20
                if OI_RATIO20[si, di] > 2.0:
                    OI_SURGE[si, di] = 1.0
            # OI-Price divergence
            c_now = close[di]; c_5 = close[di-5]
            if not np.isnan(c_now) and not np.isnan(c_5) and c_5 > 0:
                price_up = c_now > c_5
                oi_up = oi_now > oi_5
                if price_up and oi_up:
                    OI_PRICE_DIV[si, di] = 1.0   # new longs entering
                elif price_up and not oi_up:
                    OI_PRICE_DIV[si, di] = -1.0  # short covering
                elif not price_up and oi_up:
                    OI_PRICE_DIV[si, di] = 0.5   # new shorts (bearish)
                # not price_up and not oi_up = long liquidation
            # Volume / OI ratio
            if oi_now > 0 and not np.isnan(vol[di]):
                VOL_OI_RATIO[si, di] = vol[di] / oi_now

    print(f"  Market vol: median={VOL_MEDIAN:.4f}%, P50={VOL_P50:.4f}%, P75={VOL_P75:.4f}%, P90={VOL_P90:.4f}%")
    print(f"  Done ({time.time()-t0:.1f}s)")
    VOL_P90 = np.percentile(valid_vols, 90) if len(valid_vols) > 0 else 2.0

    print(f"  Market vol: median={VOL_MEDIAN:.4f}%, P50={VOL_P50:.4f}%, P75={VOL_P75:.4f}%, P90={VOL_P90:.4f}%")
    print(f"  Done ({time.time()-t0:.1f}s)")

    # ===================== SIGNAL DEFINITIONS =====================
    def sig_v121(di, edi):
        c = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((roc * zs, s, ep, 'v121'))
        return c

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

    def sig_union(di, edi):
        all_sigs = {}
        for item in sig_v121(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc * 3
            all_sigs[s][2].append('v121')
        for item in sig_ov_id(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc * 2
            all_sigs[s][2].append('ov_id')
        for item in sig_final_flag(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc
            all_sigs[s][2].append('ff')
        return [(sc, s, ep, '+'.join(sigs)) for s, (sc, ep, sigs) in all_sigs.items()]

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

    def compute_composite(di, daily_eq, high_water, perf_window=20):
        scores = []
        bth = BREADTH[di]
        if not np.isnan(bth):
            scores.append(np.clip((bth - 0.4) / (0.7 - 0.4), 0, 1))
        vol = MKT_VOL[di]
        if not np.isnan(vol) and VOL_MEDIAN > 0:
            vol_ratio = vol / VOL_MEDIAN
            scores.append(np.clip((1.5 - vol_ratio) / (1.5 - 0.8), 0, 1))
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
                scores.append(np.clip((z + 1.0) / 2.0, 0, 1))
            except Exception:
                pass
        if high_water > 0:
            cur_dd = (daily_eq[-1] - high_water) / high_water
        else:
            cur_dd = 0
        scores.append(np.clip(1.0 + cur_dd / 0.3, 0, 1))
        return np.mean(scores) if scores else 0.5

    def dd_size(pv, high_water, tiers):
        if high_water <= 0:
            return tiers[0][1]
        dd = (pv - high_water) / high_water
        for dd_thresh, size_frac in tiers:
            if dd >= -dd_thresh:
                return size_frac
        return tiers[-1][1]

    def wr_size(trades, window=20):
        if len(trades) < window:
            return 1.0
        recent = trades[-window:]
        wr = np.mean([1 if t > 0 else 0 for t in recent])
        if wr > 0.65:
            return 1.3
        elif wr >= 0.50:
            return 1.0
        else:
            return 0.5

    def eq_streak(trades, streak_threshold=3, cooldown=5):
        if len(trades) < streak_threshold:
            return 1.0, 0
        consec_losses = 0
        for t in reversed(trades):
            if t <= 0:
                consec_losses += 1
            else:
                break
        if consec_losses >= streak_threshold:
            return 0.5, cooldown
        return 1.0, 0

    # ===================== ADAPTIVE VOL THRESHOLD =====================
    def adaptive_atr_max(di):
        """Dynamic atr_norm_max based on MKT_VOL regime."""
        vol = MKT_VOL[di]
        if np.isnan(vol):
            return 12.0  # default to V167 sweet spot if unknown
        if vol < VOL_P50:
            return 15.0  # low vol regime: be aggressive
        elif vol < VOL_P75:
            return 12.0  # normal regime: V167 sweet spot
        else:
            return 8.0   # high vol regime: defensive

    # ===================== BACKTEST ENGINE =====================
    # vol_mode controls how the vol filter is applied:
    #   'fixed'     : use atr_norm_max as a fixed threshold
    #   'adaptive'  : use dynamic threshold based on MKT_VOL regime
    #   'double'    : use BOTH atr_norm_max AND mkt_vol_pctile threshold
    #
    # eq_method: 'none' or 'wr' or 'streak'

    def backtest(start_di=MIN_TRAIN, end_di=None,
                 atr_norm_max=12.0, max_corr=0.7,
                 dd_tiers=None,
                 regime_lo=0.5, regime_hi=1.5,
                 sl_pct=0.0, hold=1, top_n=3,
                 vol_mode='fixed',
                 mkt_vol_pctile=None,
                 eq_method='none',
                 oi_mode='none'):
        if end_di is None: end_di = ND
        if dd_tiers is None:
            dd_tiers = [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)]

        cash = float(CASH0)
        positions = []
        trades = []
        daily_eq = []
        high_water = float(CASH0)

        streak_cooldown_remaining = 0

        for di in range(start_di, end_di - 1):
            # Mark-to-market
            pv = cash
            for p in positions:
                cp = C[p['si'], di]
                if not np.isnan(cp) and cp > 0:
                    m = MULT.get(p['sym'], DEF_MULT)
                    pv += cp * m * p['lots'] - cp * m * abs(p['lots']) * COMM
            daily_eq.append(pv)
            if pv > high_water:
                high_water = pv

            # --- Stop-loss check ---
            if sl_pct > 0:
                cl_early = []
                for p in positions:
                    cp = C[p['si'], di]
                    if np.isnan(cp) or cp <= 0: continue
                    m = MULT.get(p['sym'], DEF_MULT)
                    unrealized = (cp - p['entry_price']) * m * p['lots']
                    invested = p['entry_price'] * m * abs(p['lots'])
                    if invested > 0:
                        loss_pct = unrealized / invested
                        if loss_pct < -sl_pct:
                            cash += cp * m * abs(p['lots']) * (1 - COMM)
                            pnl_pct = unrealized / invested * 100
                            trades.append(pnl_pct)
                            cl_early.append(p)
                            if eq_method == 'streak':
                                _, streak_cooldown_remaining = eq_streak(trades)
                for p in cl_early: positions.remove(p)

            # Close positions past hold period
            cl = []
            for p in positions:
                if di - p['entry_di'] >= p['hold_days']:
                    ep = C[p['si'], di]
                    if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                    m = MULT.get(p['sym'], DEF_MULT)
                    pnl = (ep - p['entry_price']) * m * p['lots']
                    inv = p['entry_price'] * m * abs(p['lots'])
                    pp = pnl / inv * 100 if inv > 0 else 0
                    cash += ep * m * abs(p['lots']) * (1 - COMM)
                    trades.append(pp)
                    cl.append(p)
                    if eq_method == 'streak':
                        _, streak_cooldown_remaining = eq_streak(trades)
            for p in cl: positions.remove(p)

            # --- Kitchen Sink sizing ---
            dd_sz = dd_size(pv, high_water, dd_tiers)

            # WR adaptive sizing
            use_wr = eq_method == 'wr'
            wr_mult_val = wr_size(trades, window=20) if use_wr else 1.0

            # Streak sizing
            eq_mult_val = 1.0
            if eq_method == 'streak':
                if streak_cooldown_remaining > 0:
                    eq_mult_val = 0.5
                    streak_cooldown_remaining -= 1
                else:
                    eq_mult_val, streak_cooldown_remaining = eq_streak(trades)

            composite = compute_composite(di, daily_eq, high_water)
            regime_mult = regime_lo + composite * (regime_hi - regime_lo)

            if use_wr:
                pos_size = dd_sz * wr_mult_val * regime_mult
            elif eq_method == 'streak':
                pos_size = dd_sz * eq_mult_val * regime_mult
            else:
                pos_size = dd_sz * regime_mult

            pos_size = max(0.05, min(0.99, pos_size))

            # --- Enter positions ---
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue

            held_si = set(p['si'] for p in positions)

            # Get best V121 and best Union signal
            cands_v121 = sig_v121(di, edi)
            cands_union = sig_union(di, edi)

            # Apply vol filter based on vol_mode
            if vol_mode == 'adaptive':
                cur_atr_max = adaptive_atr_max(di)
                cands_v121_f = [c for c in cands_v121
                                if not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < cur_atr_max]
                cands_union_f = [c for c in cands_union
                                 if not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < cur_atr_max]
            elif vol_mode == 'double':
                # Double filter: atr_norm AND mkt_vol percentile
                mkt_vol_ok = True
                if mkt_vol_pctile is not None:
                    vol_thresh = np.percentile(valid_vols, mkt_vol_pctile)
                    cur_vol = MKT_VOL[di]
                    mkt_vol_ok = not np.isnan(cur_vol) and cur_vol < vol_thresh
                cands_v121_f = [c for c in cands_v121
                                if (not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < atr_norm_max)
                                and mkt_vol_ok]
                cands_union_f = [c for c in cands_union
                                 if (not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < atr_norm_max)
                                 and mkt_vol_ok]
            else:  # fixed
                cands_v121_f = [c for c in cands_v121
                                if not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < atr_norm_max]
                cands_union_f = [c for c in cands_union
                                 if not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < atr_norm_max]

            cands_v121_f.sort(key=lambda x: -x[0])
            cands_union_f.sort(key=lambda x: -x[0])

            # OI enhancement: modify scores based on OI data
            if oi_mode != 'none':
                oi_enhanced_v121 = []
                for sc, s, ep, sig in cands_v121_f:
                    oi_boost = 1.0
                    if oi_mode == 'oi_boost':
                        # Simple: OI-price divergence boosts or reduces score
                        div = OI_PRICE_DIV[s, di]
                        if not np.isnan(div):
                            if div > 0:   # price↑ + OI↑ = strong
                                oi_boost = 1.5
                            elif div < 0:  # price↑ + OI↓ = weak
                                oi_boost = 0.7
                        surge = OI_SURGE[s, di]
                        if surge > 0:
                            oi_boost *= 1.3
                    elif oi_mode == 'oi_momentum':
                        # Use OI momentum as signal amplifier
                        om5 = OI_MOM5[s, di]
                        if not np.isnan(om5):
                            if om5 > 5:     # OI surging
                                oi_boost = 1.0 + om5 / 50.0
                            elif om5 < -5:   # OI dropping
                                oi_boost = 0.6
                        surge = OI_SURGE[s, di]
                        if surge > 0:
                            oi_boost *= 1.4
                    elif oi_mode == 'oi_combined':
                        # Combine OI divergence + momentum + surge
                        div = OI_PRICE_DIV[s, di]
                        om5 = OI_MOM5[s, di]
                        surge = OI_SURGE[s, di]
                        voi = VOL_OI_RATIO[s, di]
                        bonus = 0
                        if not np.isnan(div):
                            bonus += div * 0.3  # +0.3 or -0.3
                        if not np.isnan(om5):
                            bonus += min(om5 / 20.0, 0.5)  # up to +0.5
                        if surge > 0:
                            bonus += 0.4
                        if not np.isnan(voi) and voi > 2.0:
                            bonus += 0.2  # high speculation activity
                        oi_boost = max(0.5, 1.0 + bonus)
                    elif oi_mode == 'oi_ranking':
                        # Use OI ratio (current vs 20-day avg) as rank multiplier
                        r20 = OI_RATIO20[s, di]
                        if not np.isnan(r20):
                            oi_boost = min(r20, 2.0)  # cap at 2x
                        surge = OI_SURGE[s, di]
                        if surge > 0:
                            oi_boost *= 1.3
                    oi_enhanced_v121.append((sc * oi_boost, s, ep, sig))
                cands_v121_f = oi_enhanced_v121

                oi_enhanced_union = []
                for sc, s, ep, sig in cands_union_f:
                    oi_boost = 1.0
                    if oi_mode == 'oi_boost':
                        div = OI_PRICE_DIV[s, di]
                        if not np.isnan(div):
                            if div > 0: oi_boost = 1.5
                            elif div < 0: oi_boost = 0.7
                        if OI_SURGE[s, di] > 0: oi_boost *= 1.3
                    elif oi_mode == 'oi_momentum':
                        om5 = OI_MOM5[s, di]
                        if not np.isnan(om5):
                            if om5 > 5: oi_boost = 1.0 + om5 / 50.0
                            elif om5 < -5: oi_boost = 0.6
                        if OI_SURGE[s, di] > 0: oi_boost *= 1.4
                    elif oi_mode == 'oi_combined':
                        div = OI_PRICE_DIV[s, di]
                        om5 = OI_MOM5[s, di]
                        surge = OI_SURGE[s, di]
                        voi = VOL_OI_RATIO[s, di]
                        bonus = 0
                        if not np.isnan(div): bonus += div * 0.3
                        if not np.isnan(om5): bonus += min(om5 / 20.0, 0.5)
                        if surge > 0: bonus += 0.4
                        if not np.isnan(voi) and voi > 2.0: bonus += 0.2
                        oi_boost = max(0.5, 1.0 + bonus)
                    elif oi_mode == 'oi_ranking':
                        r20 = OI_RATIO20[s, di]
                        if not np.isnan(r20): oi_boost = min(r20, 2.0)
                        if OI_SURGE[s, di] > 0: oi_boost *= 1.3
                    oi_enhanced_union.append((sc * oi_boost, s, ep, sig))
                cands_union_f = oi_enhanced_union

                cands_v121_f.sort(key=lambda x: -x[0])
                cands_union_f.sort(key=lambda x: -x[0])

            best_v121 = None
            for c in cands_v121_f:
                if c[1] not in held_si:
                    best_v121 = c
                    break

            best_union = None
            for c in cands_union_f:
                if c[1] not in held_si:
                    best_union = c
                    break

            entries = []
            if best_v121 and best_union:
                if best_v121[1] == best_union[1]:
                    entries.append((best_v121[0], best_v121[1], best_v121[2],
                                    'v121+union', pos_size * 1.5))
                else:
                    corr = get_corr(best_v121[1], best_union[1], di)
                    if corr < max_corr:
                        entries.append((best_v121[0], best_v121[1], best_v121[2],
                                        'v121', pos_size))
                        entries.append((best_union[0], best_union[1], best_union[2],
                                        'union', pos_size))
                    else:
                        best = best_v121 if best_v121[0] >= best_union[0] else best_union
                        entries.append((best[0], best[1], best[2], 'best', pos_size))
            elif best_v121:
                entries.append((best_v121[0], best_v121[1], best_v121[2], 'v121', pos_size))
            elif best_union:
                entries.append((best_union[0], best_union[1], best_union[2], 'union', pos_size))

            cash_snapshot = cash
            n_planned = len(entries)
            for sc, s, pr, sig_str, pct in entries:
                if s in set(p['si'] for p in positions): continue
                if len(positions) >= top_n: break
                cap = cash_snapshot * pct / n_planned
                sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                ct = max(1, int(cap / (pr * m * (1 + COMM))))
                ci = pr * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                    ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                if ct <= 0 or ci <= 0 or ci > cash: continue
                cash -= ci
                positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                  'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': hold,
                                  'sig': sig_str, 'score': sc})

        # Close remaining
        for p in positions:
            ae = end_di - 1
            ep = C[p['si'], min(ae, ND-1)]
            if np.isnan(ep) or ep <= 0: ep = p['entry_price']
            m = MULT.get(p['sym'], DEF_MULT)
            cash += ep * m * abs(p['lots']) * (1 - COMM)

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

    # Collect all results for final ranking
    all_results = []

    # ===================== V176: OI-ENHANCED STRATEGY TESTS =====================

    oi_modes = ['none', 'oi_boost', 'oi_momentum', 'oi_combined', 'oi_ranking']
    oi_names = {'none': 'NO OI (V169 baseline)', 'oi_boost': 'OI-Price Divergence',
                'oi_momentum': 'OI Momentum', 'oi_combined': 'OI Combined',
                'oi_ranking': 'OI Ranking'}

    all_results = []

    # --- BASELINE: V169 champion config without OI ---
    print("\n" + "=" * 130)
    print("  BASELINE: V169 champion (atr<12%, corr=0.5, no wr, top_n=3)")
    print("=" * 130)

    r_base = backtest(atr_norm_max=12.0, max_corr=0.5,
                      dd_tiers=DD_AGGR100, eq_method='none',
                      regime_lo=0.5, regime_hi=1.5, sl_pct=0.0,
                      hold=1, top_n=3, vol_mode='fixed', oi_mode='none')
    pr(r_base, "BASELINE: V169 atr<12%, corr=0.5, top_n=3")
    all_results.append({**r_base, 'label': 'baseline', 'oi_mode': 'none',
                        'atr_norm_max': 12.0, 'max_corr': 0.5, 'top_n': 3})

    # --- SECTION 1: OI modes with V169 best config ---
    print("\n" + "=" * 130)
    print("  SECTION 1: OI Enhancement Modes (atr<12%, corr=0.5, top_n=3)")
    print("=" * 130)

    for oi_m in oi_modes:
        if oi_m == 'none': continue
        lbl = f"OI={oi_m}"
        r = backtest(atr_norm_max=12.0, max_corr=0.5,
                     dd_tiers=DD_AGGR100, eq_method='none',
                     regime_lo=0.5, regime_hi=1.5, sl_pct=0.0,
                     hold=1, top_n=3, vol_mode='fixed', oi_mode=oi_m)
        pr(r, f"{lbl} (atr<12%, c=0.5)")
        all_results.append({**r, 'label': f'atr12_c05_{oi_m}', 'oi_mode': oi_m,
                            'atr_norm_max': 12.0, 'max_corr': 0.5, 'top_n': 3})

    # --- SECTION 2: OI modes with corr=0.7 ---
    print("\n" + "=" * 130)
    print("  SECTION 2: OI Enhancement (atr<12%, corr=0.7, top_n=3)")
    print("=" * 130)

    for oi_m in oi_modes:
        if oi_m == 'none': continue
        r = backtest(atr_norm_max=12.0, max_corr=0.7,
                     dd_tiers=DD_AGGR100, eq_method='none',
                     regime_lo=0.5, regime_hi=1.5, sl_pct=0.0,
                     hold=1, top_n=3, vol_mode='fixed', oi_mode=oi_m)
        pr(r, f"OI={oi_m} (atr<12%, c=0.7)")
        all_results.append({**r, 'label': f'atr12_c07_{oi_m}', 'oi_mode': oi_m,
                            'atr_norm_max': 12.0, 'max_corr': 0.7, 'top_n': 3})

    # --- SECTION 3: OI modes with atr<10% ---
    print("\n" + "=" * 130)
    print("  SECTION 3: OI Enhancement (atr<10%, corr=0.5, top_n=3)")
    print("=" * 130)

    for oi_m in oi_modes:
        if oi_m == 'none': continue
        r = backtest(atr_norm_max=10.0, max_corr=0.5,
                     dd_tiers=DD_AGGR100, eq_method='none',
                     regime_lo=0.5, regime_hi=1.5, sl_pct=0.0,
                     hold=1, top_n=3, vol_mode='fixed', oi_mode=oi_m)
        pr(r, f"OI={oi_m} (atr<10%, c=0.5)")
        all_results.append({**r, 'label': f'atr10_c05_{oi_m}', 'oi_mode': oi_m,
                            'atr_norm_max': 10.0, 'max_corr': 0.5, 'top_n': 3})

    # --- SECTION 4: OI modes with top_n=4 ---
    print("\n" + "=" * 130)
    print("  SECTION 4: OI Enhancement (atr<12%, corr=0.7, top_n=4)")
    print("=" * 130)

    for oi_m in oi_modes:
        if oi_m == 'none': continue
        r = backtest(atr_norm_max=12.0, max_corr=0.7,
                     dd_tiers=DD_AGGR100, eq_method='none',
                     regime_lo=0.5, regime_hi=1.5, sl_pct=0.0,
                     hold=1, top_n=4, vol_mode='fixed', oi_mode=oi_m)
        pr(r, f"OI={oi_m} (atr<12%, c=0.7, top4)")
        all_results.append({**r, 'label': f'atr12_c07_top4_{oi_m}', 'oi_mode': oi_m,
                            'atr_norm_max': 12.0, 'max_corr': 0.7, 'top_n': 4})

    # ===================== WALK-FORWARD VALIDATION =====================
    print("\n" + "=" * 130)
    print("  WALK-FORWARD: Top configs + baseline")
    print("=" * 130)

    # Sort by R/M ratio
    ranked = sorted(all_results, key=lambda x: abs(x.get('ann', 0) / x.get('mdd', -1)), reverse=True)

    base_ann = r_base['ann']
    base_mdd = r_base['mdd']
    base_rm = abs(base_ann / base_mdd) if base_mdd != 0 else 0

    print(f"\n  Baseline: Ann={base_ann:+.0f}% | MDD={base_mdd:.0f}% | R/M={base_rm:.2f}")

    # WF top 3 OI configs
    wf_years = [(2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025), (2025, 2026)]

    # Also do baseline WF
    print(f"\n  Walk-forward: BASELINE")
    b_wf = walk_forward(label="BASELINE WF", atr_norm_max=12.0, max_corr=0.5,
                         dd_tiers=DD_AGGR100, eq_method='none', regime_lo=0.5,
                         regime_hi=1.5, sl_pct=0.0, hold=1, top_n=3,
                         vol_mode='fixed', oi_mode='none')

    # Top 3 by full-period R/M
    top3 = [r for r in ranked if r.get('oi_mode') != 'none'][:3]
    for r in top3:
        lbl = r['label']
        oi_m = r['oi_mode']
        an = r.get('atr_norm_max', 12.0)
        mc = r.get('max_corr', 0.5)
        tn = r.get('top_n', 3)
        print(f"\n  Walk-forward: {lbl}")
        walk_forward(label=f"WF {lbl}", atr_norm_max=an, max_corr=mc,
                     dd_tiers=DD_AGGR100, eq_method='none', regime_lo=0.5,
                     regime_hi=1.5, sl_pct=0.0, hold=1, top_n=tn,
                     vol_mode='fixed', oi_mode=oi_m)

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 130)
    print("  V176 FINAL SUMMARY: OI-Enhanced V121")
    print("=" * 130)

    print(f"\n  {'Config':35s} | {'Ann':>7s} | {'MDD':>5s} | {'R/M':>5s} | {'WR':>5s} | {'N':>4s} | Delta_R/M")
    print(f"  {'-'*35}-+-{'-'*7}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*4}-+-{'-'*8}")

    print(f"  {'BASELINE (V169 no OI)':35s} | {base_ann:>+7.0f}% | {base_mdd:>5.0f}% | {base_rm:>5.2f} | {r_base['wr']:>5.1f}% | {r_base['n']:>4d} |    ---")

    for r in ranked[:15]:
        if r.get('oi_mode') == 'none': continue
        ann = r['ann']; mdd = r['mdd']
        rm = abs(ann / mdd) if mdd != 0 else 0
        delta = rm - base_rm
        print(f"  {r['label']:35s} | {ann:>+7.0f}% | {mdd:>5.0f}% | {rm:>5.2f} | {r['wr']:>5.1f}% | {r['n']:>4d} | {delta:>+8.2f}")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
