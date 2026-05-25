"""
Alpha Futures V175 — Multi-Day Signal Confirmation
===============================================================================
V169 champion uses single-day signal with atr_norm<12% vol filter giving
+253%/-15% WF (R/M=16.91).

V175 explores MULTI-DAY SIGNAL CONFIRMATION:
  Core idea: requiring a signal on 2+ consecutive days dramatically reduces
  false signals. The signal is much more reliable if ROC5>1% AND Z>1.5
  persists for multiple days.

Specific tests:
  A. Consecutive confirmation: signal on day di AND day di-1
  B. Window confirmation: signal on day di, AND >=1 of past 3 days also fired
  C. Score persistence: signal on day di, AND same commodity in top_n on di-1
  D. Trend+signal combo: V121 signal + ROC20>0 + price>SMA20
  E. Volume confirmation: V121 signal + volume > 1.2 * 20-day avg volume

Parameters swept:
  confirm_mode:  ['none_baseline', 'consecutive', 'window',
                  'persistence', 'trend_combo', 'volume']
  confirm_window: [2, 3]          (for window mode)
  atr_norm_max:  [10, 12]
  max_corr:      [0.5, 0.7]
  top_n:         [3]

Base: V169 vol filter + Kitchen Sink sizing, aggro100 DD, regime 0.5-1.5.
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
    print("  V175 — Multi-Day Signal Confirmation")
    print("  Testing consecutive / window / persistence / trend+signal / volume confirmation")
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

    # SMA20 for trend confirmation
    SMA20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        SMA20[si] = talib.SMA(c, timeperiod=20)

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

    # Volume 20-day average for volume confirmation
    VOL20_AVG = np.full((NS, ND), np.nan)
    for si in range(NS):
        v = V[si].astype(np.float64)
        for di in range(20, ND):
            window = v[di-20:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 10:
                VOL20_AVG[si, di] = np.mean(valid)

    # ===================== PRECOMPUTE SIGNAL QUALIFYING FLAGS =====================
    # For multi-day confirmation we need to know on which days each commodity
    # qualifies as a V121 signal (ROC5>1, Z>1.5, ROC improving).
    # Also precompute the full ranked list for persistence check.

    print("  Precomputing signal flags for multi-day confirmation...", flush=True)

    # V121 qualifying flag: si qualifies on day di?
    V121_QUAL = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(1, ND):
            roc = ROC5[si, di]
            zs = ZSCORE[si, di]
            if np.isnan(roc) or np.isnan(zs): continue
            if roc <= 1.0 or zs <= 1.5: continue
            rp = ROC5[si, di-1]
            if not np.isnan(rp) and roc <= rp: continue
            V121_QUAL[si, di] = True

    # OV+ID qualifying flag
    OVID_QUAL = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(ND):
            ov = OV_GAP[si, di]
            idr = ID_RET[si, di]
            roc = ROC5[si, di]
            if any(np.isnan(x) for x in [ov, idr, roc]): continue
            if ov <= 0.3 or idr <= 0.3 or roc <= 1.0: continue
            OVID_QUAL[si, di] = True

    # Final Flag qualifying
    FF_QUAL = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(6, ND):
            roc20 = ROC20[si, di]
            if np.isnan(roc20) or roc20 <= 5.0: continue
            h5 = H[si, di-4:di+1]; l5 = L[si, di-4:di+1]
            if any(np.isnan(x) for x in h5) or any(np.isnan(x) for x in l5): continue
            r5 = np.max(h5) - np.min(l5)
            atr = ATR14[si, di]
            if np.isnan(atr) or atr <= 0 or r5 > atr * 3.0: continue
            h4 = np.max(H[si, di-4:di])
            cp = C[si, di]
            if np.isnan(cp) or cp <= h4: continue
            FF_QUAL[si, di] = True

    # Pre-compute union score ranking per day for persistence check
    # We store top_n_candidates per day: set of si that were in top_n
    UNION_TOPN = [set() for _ in range(ND)]  # UNION_TOPN[di] = set of si that were top_n candidates
    TOP_N_PERSIST = 10  # use top 10 for persistence (generous window)
    for di in range(ND):
        cands = []
        for si in range(NS):
            score = 0.0
            if V121_QUAL[si, di]:
                roc = ROC5[si, di]; zs = ZSCORE[si, di]
                score += roc * zs * 3
            if OVID_QUAL[si, di]:
                ov = OV_GAP[si, di]; idr = ID_RET[si, di]; roc = ROC5[si, di]
                zs = ZSCORE[si, di]
                z_bonus = zs if not np.isnan(zs) and zs > 1.0 else 1.0
                score += (ov + idr) * roc * z_bonus * 2
            if FF_QUAL[si, di]:
                roc20 = ROC20[si, di]
                cp = C[si, di]; h4 = np.max(H[si, max(0,di-4):di])
                atr = ATR14[si, di]
                if not np.isnan(atr) and atr > 0:
                    score += roc20 * (cp - h4) / atr
            if score > 0:
                cands.append((score, si))
        cands.sort(key=lambda x: -x[0])
        for _, si in cands[:TOP_N_PERSIST]:
            UNION_TOPN[di].add(si)

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

    print(f"  Market vol median={VOL_MEDIAN:.4f}%")
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

    # ===================== MULTI-DAY CONFIRMATION FILTERS =====================
    def confirm_consecutive(cands, di):
        """A: Signal must fire on day di AND day di-1 (both V121-qualified)."""
        filtered = []
        for sc, s, ep, sig_str in cands:
            if di < 1: continue
            if V121_QUAL[s, di] and V121_QUAL[s, di-1]:
                filtered.append((sc, s, ep, sig_str))
        return filtered

    def confirm_window(cands, di, window=3):
        """B: Signal fires on day di, AND >=1 of past (window) days also fired."""
        filtered = []
        for sc, s, ep, sig_str in cands:
            count = 0
            for offset in range(1, window + 1):
                dd = di - offset
                if dd < 0: continue
                if V121_QUAL[s, dd]:
                    count += 1
            if count >= 1:
                filtered.append((sc, s, ep, sig_str))
        return filtered

    def confirm_persistence(cands, di):
        """C: Signal fires on day di, AND si was in top_n candidates on day di-1."""
        filtered = []
        if di < 1: return filtered
        yesterday_topn = UNION_TOPN[di - 1]
        for sc, s, ep, sig_str in cands:
            if s in yesterday_topn:
                filtered.append((sc, s, ep, sig_str))
        return filtered

    def confirm_trend_combo(cands, di):
        """D: V121 signal + ROC20 > 0 + price > SMA20."""
        filtered = []
        for sc, s, ep, sig_str in cands:
            roc20 = ROC20[s, di]
            if np.isnan(roc20) or roc20 <= 0: continue
            cp = C[s, di]
            sma = SMA20[s, di]
            if np.isnan(sma) or np.isnan(cp) or cp <= sma: continue
            filtered.append((sc, s, ep, sig_str))
        return filtered

    def confirm_volume(cands, di, vol_mult=1.2):
        """E: V121 signal + volume > vol_mult * 20-day avg volume."""
        filtered = []
        for sc, s, ep, sig_str in cands:
            v20 = VOL20_AVG[s, di]
            vol = V[s, di]
            if np.isnan(v20) or np.isnan(vol) or v20 <= 0: continue
            if vol > v20 * vol_mult:
                filtered.append((sc, s, ep, sig_str))
        return filtered

    def apply_confirm(cands, di, confirm_mode, confirm_window_val=3):
        """Apply the chosen confirmation filter to candidates."""
        if confirm_mode == 'none_baseline':
            return cands
        elif confirm_mode == 'consecutive':
            return confirm_consecutive(cands, di)
        elif confirm_mode == 'window':
            return confirm_window(cands, di, window=confirm_window_val)
        elif confirm_mode == 'persistence':
            return confirm_persistence(cands, di)
        elif confirm_mode == 'trend_combo':
            return confirm_trend_combo(cands, di)
        elif confirm_mode == 'volume':
            return confirm_volume(cands, di, vol_mult=1.2)
        else:
            return cands

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

    # ===================== BACKTEST ENGINE =====================
    def backtest(start_di=MIN_TRAIN, end_di=None,
                 atr_norm_max=12.0, max_corr=0.7,
                 dd_tiers=None,
                 regime_lo=0.5, regime_hi=1.5,
                 sl_pct=0.0, hold=1, top_n=3,
                 confirm_mode='none_baseline',
                 confirm_window_val=3):
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
            for p in cl: positions.remove(p)

            # --- Kitchen Sink sizing ---
            dd_sz = dd_size(pv, high_water, dd_tiers)
            composite = compute_composite(di, daily_eq, high_water)
            regime_mult = regime_lo + composite * (regime_hi - regime_lo)
            pos_size = dd_sz * regime_mult
            pos_size = max(0.05, min(0.99, pos_size))

            # --- Enter positions ---
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue

            held_si = set(p['si'] for p in positions)

            # Get raw candidates
            cands_v121 = sig_v121(di, edi)
            cands_union = sig_union(di, edi)

            # Apply vol filter
            cands_v121_f = [c for c in cands_v121
                            if not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < atr_norm_max]
            cands_union_f = [c for c in cands_union
                             if not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < atr_norm_max]

            # Apply confirmation filter
            cands_v121_f = apply_confirm(cands_v121_f, di, confirm_mode, confirm_window_val)
            cands_union_f = apply_confirm(cands_union_f, di, confirm_mode, confirm_window_val)

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
        print(f"  {label:100s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d}")

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

    # ======================================================================
    # PARAMETER GRID
    # ======================================================================
    confirm_modes = ['none_baseline', 'consecutive', 'window',
                     'persistence', 'trend_combo', 'volume']
    confirm_windows = [2, 3]
    atr_norm_maxes = [10.0, 12.0]
    max_corrs = [0.5, 0.7]
    top_ns = [3]

    # ===================== SECTION 0: BASELINE (no confirmation) =====================
    print("\n" + "=" * 130)
    print("  SECTION 0: BASELINE — V169 single-day signal, no confirmation")
    print("  atr_norm<12% or <10%, max_corr=0.5 or 0.7, top_n=3")
    print("=" * 130)

    for an_max in atr_norm_maxes:
        for mc in max_corrs:
            label = f"BASELINE atr<{an_max:.0f}%, corr={mc:.1f}"
            r = backtest(atr_norm_max=an_max, max_corr=mc,
                         dd_tiers=DD_AGGR100,
                         regime_lo=0.5, regime_hi=1.5, sl_pct=0.0,
                         hold=1, top_n=3,
                         confirm_mode='none_baseline', confirm_window_val=3)
            pr(r, label)
            all_results.append({**r, 'label': f'base_atr{an_max:.0f}_c{mc:.1f}',
                                'section': 0, 'confirm_mode': 'none_baseline',
                                'atr_norm_max': an_max, 'max_corr': mc, 'top_n': 3,
                                'confirm_window_val': 3})

    # ===================== SECTION 1: CONSECUTIVE CONFIRMATION =====================
    print("\n" + "=" * 130)
    print("  SECTION 1: CONSECUTIVE CONFIRMATION (A)")
    print("  Signal must fire on day di AND day di-1")
    print("  Both days must have ROC5 > 1.0 and Z > 1.5")
    print("=" * 130)

    for an_max in atr_norm_maxes:
        for mc in max_corrs:
            label = f"CONSECUTIVE atr<{an_max:.0f}%, corr={mc:.1f}"
            r = backtest(atr_norm_max=an_max, max_corr=mc,
                         dd_tiers=DD_AGGR100,
                         regime_lo=0.5, regime_hi=1.5, sl_pct=0.0,
                         hold=1, top_n=3,
                         confirm_mode='consecutive', confirm_window_val=2)
            pr(r, label)
            all_results.append({**r, 'label': f'consec_atr{an_max:.0f}_c{mc:.1f}',
                                'section': 1, 'confirm_mode': 'consecutive',
                                'atr_norm_max': an_max, 'max_corr': mc, 'top_n': 3,
                                'confirm_window_val': 2})

    # ===================== SECTION 2: WINDOW CONFIRMATION =====================
    print("\n" + "=" * 130)
    print("  SECTION 2: WINDOW CONFIRMATION (B)")
    print("  Signal fires on day di, AND >=1 of past N days also fired")
    print("  confirm_window: [2, 3]")
    print("=" * 130)

    for an_max in atr_norm_maxes:
        for mc in max_corrs:
            for cw in confirm_windows:
                label = f"WINDOW({cw}) atr<{an_max:.0f}%, corr={mc:.1f}"
                r = backtest(atr_norm_max=an_max, max_corr=mc,
                             dd_tiers=DD_AGGR100,
                             regime_lo=0.5, regime_hi=1.5, sl_pct=0.0,
                             hold=1, top_n=3,
                             confirm_mode='window', confirm_window_val=cw)
                pr(r, label)
                all_results.append({**r, 'label': f'win{cw}_atr{an_max:.0f}_c{mc:.1f}',
                                    'section': 2, 'confirm_mode': 'window',
                                    'atr_norm_max': an_max, 'max_corr': mc, 'top_n': 3,
                                    'confirm_window_val': cw})

    # ===================== SECTION 3: SCORE PERSISTENCE =====================
    print("\n" + "=" * 130)
    print("  SECTION 3: SCORE PERSISTENCE (C)")
    print("  Signal fires on day di, AND same commodity in top_n candidates on di-1")
    print("=" * 130)

    for an_max in atr_norm_maxes:
        for mc in max_corrs:
            label = f"PERSISTENCE atr<{an_max:.0f}%, corr={mc:.1f}"
            r = backtest(atr_norm_max=an_max, max_corr=mc,
                         dd_tiers=DD_AGGR100,
                         regime_lo=0.5, regime_hi=1.5, sl_pct=0.0,
                         hold=1, top_n=3,
                         confirm_mode='persistence', confirm_window_val=3)
            pr(r, label)
            all_results.append({**r, 'label': f'persist_atr{an_max:.0f}_c{mc:.1f}',
                                'section': 3, 'confirm_mode': 'persistence',
                                'atr_norm_max': an_max, 'max_corr': mc, 'top_n': 3,
                                'confirm_window_val': 3})

    # ===================== SECTION 4: TREND + SIGNAL COMBO =====================
    print("\n" + "=" * 130)
    print("  SECTION 4: TREND + SIGNAL COMBO (D)")
    print("  V121 signal + ROC20 > 0 + price > SMA20")
    print("=" * 130)

    for an_max in atr_norm_maxes:
        for mc in max_corrs:
            label = f"TREND_COMBO atr<{an_max:.0f}%, corr={mc:.1f}"
            r = backtest(atr_norm_max=an_max, max_corr=mc,
                         dd_tiers=DD_AGGR100,
                         regime_lo=0.5, regime_hi=1.5, sl_pct=0.0,
                         hold=1, top_n=3,
                         confirm_mode='trend_combo', confirm_window_val=3)
            pr(r, label)
            all_results.append({**r, 'label': f'trend_atr{an_max:.0f}_c{mc:.1f}',
                                'section': 4, 'confirm_mode': 'trend_combo',
                                'atr_norm_max': an_max, 'max_corr': mc, 'top_n': 3,
                                'confirm_window_val': 3})

    # ===================== SECTION 5: VOLUME CONFIRMATION =====================
    print("\n" + "=" * 130)
    print("  SECTION 5: VOLUME CONFIRMATION (E)")
    print("  V121 signal + volume > 1.2 * 20-day avg volume")
    print("=" * 130)

    for an_max in atr_norm_maxes:
        for mc in max_corrs:
            label = f"VOLUME atr<{an_max:.0f}%, corr={mc:.1f}"
            r = backtest(atr_norm_max=an_max, max_corr=mc,
                         dd_tiers=DD_AGGR100,
                         regime_lo=0.5, regime_hi=1.5, sl_pct=0.0,
                         hold=1, top_n=3,
                         confirm_mode='volume', confirm_window_val=3)
            pr(r, label)
            all_results.append({**r, 'label': f'vol_atr{an_max:.0f}_c{mc:.1f}',
                                'section': 5, 'confirm_mode': 'volume',
                                'atr_norm_max': an_max, 'max_corr': mc, 'top_n': 3,
                                'confirm_window_val': 3})

    # ===================== SECTION 6: RANKED RESULTS =====================
    print("\n" + "=" * 130)
    print("  SECTION 6: ALL CONFIGS RANKED BY ANNUAL RETURN (full period)")
    print("=" * 130)

    all_results.sort(key=lambda x: -x['ann'])
    for i, r in enumerate(all_results[:30]):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i+1:2d} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d} | {r['label']}")

    print("\n" + "=" * 130)
    print("  SECTION 6b: ALL CONFIGS RANKED BY R/M RATIO (risk-adjusted)")
    print("=" * 130)

    all_rm = sorted(all_results, key=lambda x: -abs(x['ann']/x['mdd']) if x['mdd'] != 0 else 0)
    for i, r in enumerate(all_rm[:30]):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i+1:2d} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d} | {r['label']}")

    # ===================== SECTION 7: CONFIRMATION MODE COMPARISON =====================
    print("\n" + "=" * 130)
    print("  SECTION 7: CONFIRMATION MODE COMPARISON (atr<12%, corr=0.7)")
    print("  Side-by-side: how does each mode affect signal quality?")
    print("=" * 130)

    comparison_base = None
    for mode in confirm_modes:
        cw = 2 if mode == 'window' else 3
        label = f"{mode:20s} atr<12%, corr=0.7"
        r = backtest(atr_norm_max=12.0, max_corr=0.7,
                     dd_tiers=DD_AGGR100,
                     regime_lo=0.5, regime_hi=1.5, sl_pct=0.0,
                     hold=1, top_n=3,
                     confirm_mode=mode, confirm_window_val=cw)
        pr(r, label)
        if mode == 'none_baseline':
            comparison_base = r

    if comparison_base:
        print(f"\n  Delta vs baseline (atr<12%, corr=0.7):")
        for mode in confirm_modes:
            if mode == 'none_baseline': continue
            cw = 2 if mode == 'window' else 3
            r = backtest(atr_norm_max=12.0, max_corr=0.7,
                         dd_tiers=DD_AGGR100,
                         regime_lo=0.5, regime_hi=1.5, sl_pct=0.0,
                         hold=1, top_n=3,
                         confirm_mode=mode, confirm_window_val=cw)
            d_ann = r['ann'] - comparison_base['ann']
            d_mdd = r['mdd'] - comparison_base['mdd']
            rm_b = abs(comparison_base['ann']/comparison_base['mdd']) if comparison_base['mdd'] != 0 else 0
            rm_r = abs(r['ann']/r['mdd']) if r['mdd'] != 0 else 0
            d_rm = rm_r - rm_b
            print(f"    {mode:20s}: Ann={d_ann:+.1f}%, MDD={d_mdd:+.1f}%, R/M={d_rm:+.2f}, N={r['n']:4d} (base N={comparison_base['n']})")

    # ===================== SECTION 8: WALK-FORWARD TOP 20 =====================
    print("\n" + "=" * 130)
    print("  SECTION 8: WALK-FORWARD VALIDATION — Top 20 by R/M")
    print("=" * 130)

    # Deduplicate by label
    seen = set()
    wf_candidates = []
    for r in all_rm:
        lbl = r['label']
        if lbl not in seen:
            seen.add(lbl)
            wf_candidates.append(r)

    wf_all = {}
    for r in wf_candidates[:20]:
        lbl = r['label']
        wf_kwargs = {
            'atr_norm_max': r['atr_norm_max'],
            'max_corr': r['max_corr'],
            'dd_tiers': DD_AGGR100,
            'regime_lo': 0.5, 'regime_hi': 1.5, 'sl_pct': 0.0,
            'hold': 1, 'top_n': r['top_n'],
            'confirm_mode': r['confirm_mode'],
            'confirm_window_val': r.get('confirm_window_val', 3),
        }
        wf_res = walk_forward(label=lbl, **wf_kwargs)
        wf_all[lbl] = (wf_res, r)
        print_wf(wf_res, lbl)

    # ===================== SECTION 9: WF COMPARISON TABLE =====================
    print("\n" + "=" * 130)
    print("  SECTION 9: WF COMPARISON TABLE — All configs ranked by WF avg")
    print("=" * 130)

    wf_ranked = []
    for lbl, (wf_res, r_info) in wf_all.items():
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        best_ann = max(r['ann'] for r in wf_res.values())
        pos = sum(1 for r in wf_res.values() if r['ann'] > 0)
        total_n = sum(r['n'] for r in wf_res.values())
        avg_wr = np.mean([r['wr'] for r in wf_res.values()])
        ratio = abs(avg_ann / worst_mdd) if worst_mdd != 0 else 0
        wf_ranked.append((avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr,
                          ratio, lbl, wf_res, r_info))

    # Sort by WF avg
    wf_ranked.sort(key=lambda x: -x[0])
    print(f"\n  Ranked by WF Average Annual Return:")
    print(f"  {'#':>3s}  {'WF AVG':>8s} {'WorstDD':>8s} {'R/M':>6s} {'Pos':>4s} {'N':>5s} {'WR':>5s}  {'Label'}")
    print(f"  {'-'*110}")
    for i, (avg_ann, wmdd, bann, pos, tn, awr, ratio, lbl, wf_res, ri) in enumerate(wf_ranked):
        print(f"  {i+1:3d}  {avg_ann:>+8.0f}% {wmdd:>7.0f}% {ratio:6.2f} {pos:4d}/6 {tn:5d} {awr:5.1f}%  {lbl}")

    # ===================== SECTION 10: BEST RISK-ADJUSTED (WF R/M) =====================
    print("\n" + "=" * 130)
    print("  SECTION 10: BEST RISK-ADJUSTED (WF R/M)")
    print("=" * 130)

    wf_ra = sorted(wf_ranked, key=lambda x: -x[6])
    for i, (avg_ann, wmdd, bann, pos, tn, awr, ratio, lbl, wf_res, ri) in enumerate(wf_ra[:15]):
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  #{i+1} WF AVG={avg_ann:+.0f}% | WorstMDD={wmdd:.0f}% | R/M={ratio:.2f} | {pos}/6 pos | {lbl}")
        print(f"     {ws}")

    # ===================== SECTION 11: BEST PER CONFIRMATION MODE =====================
    print("\n" + "=" * 130)
    print("  SECTION 11: BEST PER CONFIRMATION MODE")
    print("=" * 130)

    mode_names = {
        'none_baseline': 'No Confirmation (Baseline)',
        'consecutive':   'A: Consecutive Confirmation',
        'window':        'B: Window Confirmation',
        'persistence':   'C: Score Persistence',
        'trend_combo':   'D: Trend+Signal Combo',
        'volume':        'E: Volume Confirmation',
    }

    for cm, cm_name in mode_names.items():
        cat_items = [(avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri)
                     for avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri in wf_ranked
                     if ri['confirm_mode'] == cm]
        if not cat_items:
            print(f"\n  {cm_name:40s}: no WF results")
            continue
        cat_items.sort(key=lambda x: -x[6])
        best = cat_items[0]
        avg_ann, wmdd, bann, pos, tn, awr, ratio, lbl, wf_res, ri = best
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  {cm_name:40s} best: WF AVG={avg_ann:+.0f}% | WorstMDD={wmdd:.0f}% | R/M={ratio:.2f}")
        print(f"     {ws}")
        print(f"     Config: atr<{ri['atr_norm_max']:.0f}%, corr={ri['max_corr']:.1f}, N={tn}")

    # ===================== SECTION 12: DELTA vs BASELINE =====================
    print("\n" + "=" * 130)
    print("  SECTION 12: IMPROVEMENT vs BASELINE (atr<12%, corr=0.7, no confirm)")
    print("=" * 130)

    base_lbl = 'base_atr12_c0.7'
    if base_lbl in wf_all:
        b_wf, b_ri = wf_all[base_lbl]
        b_avg = np.mean([r['ann'] for r in b_wf.values()])
        b_wmdd = min(r['mdd'] for r in b_wf.values())
        b_rm = abs(b_avg / b_wmdd) if b_wmdd != 0 else 0
    else:
        # Compute baseline WF if not in top 20
        b_wf = walk_forward(label=base_lbl,
                            atr_norm_max=12.0, max_corr=0.7,
                            dd_tiers=DD_AGGR100,
                            regime_lo=0.5, regime_hi=1.5, sl_pct=0.0,
                            hold=1, top_n=3,
                            confirm_mode='none_baseline', confirm_window_val=3)
        b_avg = np.mean([r['ann'] for r in b_wf.values()])
        b_wmdd = min(r['mdd'] for r in b_wf.values())
        b_rm = abs(b_avg / b_wmdd) if b_wmdd != 0 else 0
        print_wf(b_wf, "BASELINE (computed)")

    print(f"\n  BASELINE: WF AVG={b_avg:+.0f}% | WorstMDD={b_wmdd:.0f}% | R/M={b_rm:.2f}")

    deltas = []
    for avg_ann, wmdd, bann, pos, tn, awr, ratio, lbl, wf_res, ri in wf_ranked:
        if lbl == base_lbl: continue
        delta_ann = avg_ann - b_avg
        delta_rm = ratio - b_rm
        deltas.append((delta_ann, delta_rm, ratio, avg_ann, wmdd, pos, lbl, wf_res))

    deltas.sort(key=lambda x: -x[1])
    print(f"\n  Configs by R/M improvement over baseline:")
    for i, (da, drm, ratio, avg, wmdd, pos, lbl, wfr) in enumerate(deltas):
        marker = "*** IMPROVED" if drm > 0 else "    worse"
        print(f"  {i+1:2d} | R/M={ratio:.2f} (delta={drm:+.2f}) | Ann delta={da:+.0f}% | {pos}/6 | {marker} | {lbl}")

    # ===================== SECTION 13: BEST COMBINATION DETAIL =====================
    print("\n" + "=" * 130)
    print("  SECTION 13: BEST COMBINATION — Full detail for top 5 by WF R/M")
    print("=" * 130)

    for i, (avg_ann, wmdd, bann, pos, tn, awr, ratio, lbl, wf_res, ri) in enumerate(wf_ra[:5]):
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  #{i+1}: {lbl}")
        print(f"       WF AVG={avg_ann:+.0f}% | WorstMDD={wmdd:.0f}% | R/M={ratio:.2f} | {pos}/6 pos | N={tn}")
        print(f"       Config: confirm={ri['confirm_mode']}, atr<{ri['atr_norm_max']:.0f}%, corr={ri['max_corr']:.1f}")
        print(f"       {ws}")
        for yr, r in sorted(wf_res.items()):
            print(f"         {yr}: Ann={r['ann']:+.1f}% | MDD={r['mdd']:.1f}% | WR={r['wr']:.1f}% | N={r['n']}")

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 130)
    print("  FINAL SUMMARY")
    print("=" * 130)

    print(f"\n  Baseline (atr<12%, corr=0.7, no confirmation):")
    print(f"    WF AVG={b_avg:+.0f}% | WorstMDD={b_wmdd:.0f}% | R/M={b_rm:.2f}")

    if wf_ra:
        best = wf_ra[0]
        print(f"\n  Best overall: {best[7]}")
        print(f"    WF AVG={best[0]:+.0f}% | WorstMDD={best[1]:.0f}% | R/M={best[6]:.2f}")
        delta_rm_best = best[6] - b_rm
        print(f"    R/M improvement over baseline: {delta_rm_best:+.2f}")

        # Best by confirmation mode
        print(f"\n  Best by confirmation mode (WF R/M):")
        for cm, cm_name in mode_names.items():
            cm_items = [(avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri)
                        for avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri in wf_ranked
                        if ri['confirm_mode'] == cm]
            if not cm_items: continue
            cm_items.sort(key=lambda x: -x[6])
            s = cm_items[0]
            delta_rm = s[6] - b_rm
            print(f"    {cm_name:40s}: {s[7]} | R/M={s[6]:.2f} (delta={delta_rm:+.2f}) | WF={s[0]:+.0f}%")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
