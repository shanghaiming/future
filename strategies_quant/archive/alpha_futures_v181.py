"""
Alpha Futures V181 — Overnight Gap + Intraday Momentum Strategy
==============================================================================
V177 dual-side champion: R/M=11.41, hold=1 day.

V181 exploits overnight gap + intraday patterns for short-term momentum:
  1. OVERNIGHT_GAP_FOLLOW: gap continuation (>0.5% gap + intraday confirms)
  2. OVERNIGHT_GAP_REVERSAL: gap fill (>1.5% gap + intraday reverses)
  3. GAP_MOMENTUM: gap > 0.3% + ROC(5) > 1% + Z > 1.5
  4. INTRADAY_BREAKOUT: large intraday range + close near high + volume surge

Key insight: Today's close-open gap and intraday pattern inform TOMORROW's
entry (we trade at next day's open). Gap signals capture overnight information
that pure close-to-close momentum misses.

Test matrix:
  1. BASELINE: V177 dual-side (short_mirror)
  2. GAP_FOLLOW: gap continuation signal
  3. GAP_REVERSAL: gap fill/reversal signal
  4. GAP_MOMENTUM: gap + ROC(5) + Z combined
  5. GAP_VOL: gap + volume confirmation
  6. COMBINED: best gap signal + V121 with 50/50 weighting

Config: same as V177 — short_mirror, atr_norm_max=10, max_corr=0.5, top_n=3, hold=1
DD_TIERS = [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)]
CASH0 = 500000, COMM = 0.0003
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
    print("  V181 — Overnight Gap + Intraday Momentum Strategy")
    print("  Exploiting overnight gap signals + intraday confirmation")
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

    # Overnight gap and intraday return
    OV_GAP = np.full((NS, ND), np.nan)
    ID_RET = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            o, c = O[si, di], C[si, di]
            if not np.isnan(o) and not np.isnan(c):
                if di > 0 and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                    OV_GAP[si, di] = (o - C[si, di-1]) / C[si, di-1] * 100
                if o > 0:
                    ID_RET[si, di] = (c - o) / o * 100

    # Intraday range as % of open
    ID_RANGE = np.full((NS, ND), np.nan)
    # Close position within range: (C - L) / (H - L), 0-1
    CLOSE_POS = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            o = O[si, di]; h = H[si, di]; l = L[si, di]; c = C[si, di]
            if not np.isnan(o) and o > 0 and not np.isnan(h) and not np.isnan(l):
                ID_RANGE[si, di] = (h - l) / o * 100
                rng = h - l
                if rng > 0 and not np.isnan(c):
                    CLOSE_POS[si, di] = (c - l) / rng

    # Volume ratio: today's volume vs 20-day avg
    VOL_RATIO = np.full((NS, ND), np.nan)
    for si in range(NS):
        v = V[si].astype(np.float64)
        for di in range(20, ND):
            vw = v[di-20:di]
            vv = vw[~np.isnan(vw)]
            if len(vv) >= 10 and not np.isnan(v[di]):
                avg_v = np.mean(vv)
                if avg_v > 0:
                    VOL_RATIO[si, di] = v[di] / avg_v

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

    print(f"  Market vol: median={VOL_MEDIAN:.4f}%, P50={VOL_P50:.4f}%, P75={VOL_P75:.4f}%, P90={VOL_P90:.4f}%")
    print(f"  Done ({time.time()-t0:.1f}s)")

    # ===================== SIGNAL DEFINITIONS =====================
    # All signals return: list of (score, si, entry_price, signal_name, direction)
    # direction: 1 = long, -1 = short

    def sig_v121_long(di, edi):
        """V121 long: ROC(5)>1% + Z>1.5 + ROC improving"""
        c = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((roc * zs, s, ep, 'v121_long', 1))
        return c

    def sig_v121_short(di, edi):
        """V121 short: ROC(5)<-1% + Z<-1.5 + ROC declining"""
        c = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc >= -1.0 or zs >= -1.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc >= rp: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((abs(roc * zs), s, ep, 'v121_short', -1))
        return c

    # --- NEW GAP SIGNALS ---

    def sig_gap_follow_long(di, edi):
        """GAP_FOLLOW long: gap > 0.5% AND intraday return > 0 → gap continuation
        Also short: gap < -0.5% AND intraday return < 0 → gap continuation (short side)
        Returns both longs and shorts in one call for the follow signal."""
        c = []
        for s in range(NS):
            ov = OV_GAP[s, di]; idr = ID_RET[s, di]
            if np.isnan(ov) or np.isnan(idr): continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue

            # Long: positive gap + positive intraday = bullish continuation
            if ov > 0.5 and idr > 0:
                score = (ov + idr) * (1 + abs(ov) / 5.0)
                c.append((score, s, ep, 'gap_follow_long', 1))
            # Short: negative gap + negative intraday = bearish continuation
            elif ov < -0.5 and idr < 0:
                score = (abs(ov) + abs(idr)) * (1 + abs(ov) / 5.0)
                c.append((score, s, ep, 'gap_follow_short', -1))
        return c

    def sig_gap_reversal_long(di, edi):
        """GAP_REVERSAL: gap > 1.5% AND intraday return < -0.3% → gap fill, go short
        gap < -1.5% AND intraday return > 0.3% → gap fill, go long"""
        c = []
        for s in range(NS):
            ov = OV_GAP[s, di]; idr = ID_RET[s, di]
            if np.isnan(ov) or np.isnan(idr): continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue

            # Large positive gap + intraday reversal → short (gap fill expected)
            if ov > 1.5 and idr < -0.3:
                score = ov * abs(idr) * 2
                c.append((score, s, ep, 'gap_reversal_short', -1))
            # Large negative gap + intraday reversal → long (gap fill expected)
            elif ov < -1.5 and idr > 0.3:
                score = abs(ov) * idr * 2
                c.append((score, s, ep, 'gap_reversal_long', 1))
        return c

    def sig_gap_momentum(di, edi):
        """GAP_MOMENTUM: gap > 0.3% + ROC(5) > 1% + Z > 1.5 → strong momentum + gap confirmation
        Also short side: gap < -0.3% + ROC(5) < -1% + Z < -1.5"""
        c = []
        for s in range(NS):
            ov = OV_GAP[s, di]; roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if any(np.isnan(x) for x in [ov, roc, zs]): continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue

            # Long: positive gap + strong momentum
            if ov > 0.3 and roc > 1.0 and zs > 1.5:
                score = ov * roc * zs * 3
                c.append((score, s, ep, 'gap_mom_long', 1))
            # Short: negative gap + strong negative momentum
            elif ov < -0.3 and roc < -1.0 and zs < -1.5:
                score = abs(ov) * abs(roc) * abs(zs) * 3
                c.append((score, s, ep, 'gap_mom_short', -1))
        return c

    def sig_gap_vol(di, edi):
        """GAP_VOL: gap > 0.3% + volume > 1.5x avg → gap confirmed by volume
        Also short side: gap < -0.3% + volume > 1.5x avg"""
        c = []
        for s in range(NS):
            ov = OV_GAP[s, di]; vr = VOL_RATIO[s, di]
            if np.isnan(ov) or np.isnan(vr): continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue

            # Long: positive gap + volume surge
            if ov > 0.3 and vr > 1.5:
                score = ov * vr * 2
                c.append((score, s, ep, 'gap_vol_long', 1))
            # Short: negative gap + volume surge
            elif ov < -0.3 and vr > 1.5:
                score = abs(ov) * vr * 2
                c.append((score, s, ep, 'gap_vol_short', -1))
        return c

    def sig_intraday_breakout(di, edi):
        """INTRADAY_BREAKOUT: intraday range > 2% + close near high (>90%) + volume > 1.5x avg
        Also short: intraday range > 2% + close near low (<10%) + volume > 1.5x avg"""
        c = []
        for s in range(NS):
            idr = ID_RANGE[s, di]; cp = CLOSE_POS[s, di]; vr = VOL_RATIO[s, di]
            if any(np.isnan(x) for x in [idr, cp, vr]): continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue

            # Bullish breakout: wide range + close near high + volume
            if idr > 2.0 and cp > 0.9 and vr > 1.5:
                score = idr * cp * vr
                c.append((score, s, ep, 'iday_breakout_long', 1))
            # Bearish breakout: wide range + close near low + volume
            elif idr > 2.0 and cp < 0.1 and vr > 1.5:
                score = idr * (1 - cp) * vr
                c.append((score, s, ep, 'iday_breakout_short', -1))
        return c

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
        corr = np.corrcoef(ra, rb)[0, 1]
        return corr if not np.isnan(corr) else 0.5

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
    # signal_mode controls which signals to use:
    #   'baseline'   : V121 long only + V121 short mirror (same as V177)
    #   'gap_follow' : gap continuation signals (both long & short)
    #   'gap_reversal': gap fill / reversal signals
    #   'gap_momentum': gap + ROC(5) + Z combined
    #   'gap_vol'    : gap + volume confirmation
    #   'iday_breakout': intraday breakout signals
    #   'combined'   : best gap signal + V121 signal with 50/50 weighting

    def backtest(start_di=MIN_TRAIN, end_di=None,
                 atr_norm_max=10.0, max_corr=0.5,
                 dd_tiers=None,
                 regime_lo=0.5, regime_hi=1.5,
                 sl_pct=0.0, hold=1, top_n=3,
                 signal_mode='baseline'):
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

            # --- Stop-loss check ---
            if sl_pct > 0:
                cl_early = []
                for p in positions:
                    cp = C[p['si'], di]
                    if np.isnan(cp) or cp <= 0: continue
                    m = MULT.get(p['sym'], DEF_MULT)
                    d = p.get('dir', 1)
                    unrealized = (cp - p['entry_price']) * m * p['lots'] * d
                    invested = p['entry_price'] * m * abs(p['lots'])
                    if invested > 0:
                        loss_pct = unrealized / invested
                        if loss_pct < -sl_pct:
                            if d == 1:
                                cash += cp * m * abs(p['lots']) * (1 - COMM)
                            else:
                                margin = p['entry_price'] * m * abs(p['lots'])
                                cash += margin + unrealized - cp * m * abs(p['lots']) * COMM
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
            composite = compute_composite(di, daily_eq, high_water)
            regime_mult = regime_lo + composite * (regime_hi - regime_lo)
            pos_size = max(0.05, min(0.99, dd_sz * regime_mult))

            # --- Enter positions ---
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue

            held_si = set(p['si'] for p in positions)

            # Collect candidates based on signal_mode
            all_cands = []

            if signal_mode == 'baseline':
                # V177 dual-side: V121 long + V121 short mirror
                all_cands = sig_v121_long(di, edi) + sig_v121_short(di, edi)

            elif signal_mode == 'gap_follow':
                all_cands = sig_gap_follow_long(di, edi)

            elif signal_mode == 'gap_reversal':
                all_cands = sig_gap_reversal_long(di, edi)

            elif signal_mode == 'gap_momentum':
                all_cands = sig_gap_momentum(di, edi)

            elif signal_mode == 'gap_vol':
                all_cands = sig_gap_vol(di, edi)

            elif signal_mode == 'iday_breakout':
                all_cands = sig_intraday_breakout(di, edi)

            elif signal_mode == 'combined':
                # Best gap signal + V121 with 50/50 weighting
                # Get V121 signals
                v121_cands = sig_v121_long(di, edi) + sig_v121_short(di, edi)
                # Get best gap signal (gap_momentum is the strongest)
                gap_cands = sig_gap_momentum(di, edi)

                # Weight: V121 gets 0.5, gap_momentum gets 0.5
                for sc, s, ep, sig, d in v121_cands:
                    all_cands.append((sc * 0.5, s, ep, sig, d))
                for sc, s, ep, sig, d in gap_cands:
                    # If already in from v121, boost score
                    existing = [i for i, x in enumerate(all_cands) if x[1] == s]
                    if existing:
                        idx = existing[0]
                        old_sc, old_s, old_ep, old_sig, old_d = all_cands[idx]
                        # If same direction, combine scores
                        if old_d == d:
                            all_cands[idx] = (old_sc + sc * 0.5, old_s, old_ep,
                                              old_sig + '+' + sig, old_d)
                        else:
                            all_cands.append((sc * 0.5, s, ep, sig, d))
                    else:
                        all_cands.append((sc * 0.5, s, ep, sig, d))

            elif signal_mode == 'union_all':
                # Union of ALL gap signals + V121
                v121_cands = sig_v121_long(di, edi) + sig_v121_short(di, edi)
                gap_follow = sig_gap_follow_long(di, edi)
                gap_mom = sig_gap_momentum(di, edi)
                gap_vol_c = sig_gap_vol(di, edi)

                # Aggregate by (symbol, direction)
                sig_map = {}
                def add_sigs(cands, weight):
                    for sc, s, ep, sig, d in cands:
                        key = (s, d)
                        if key not in sig_map:
                            sig_map[key] = [0, ep, [], d]
                        sig_map[key][0] += sc * weight
                        sig_map[key][2].append(sig)

                add_sigs(v121_cands, 1.0)
                add_sigs(gap_follow, 0.7)
                add_sigs(gap_mom, 0.8)
                add_sigs(gap_vol_c, 0.5)

                for (s, d), (sc, ep, sigs, dirn) in sig_map.items():
                    all_cands.append((sc, s, ep, '+'.join(sigs), dirn))

            # Apply ATR filter
            all_cands_f = [c for c in all_cands
                           if not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < atr_norm_max]

            # Sort by score descending
            all_cands_f.sort(key=lambda x: -x[0])

            # Apply correlation filter for top_n > 1
            selected = []
            for sc, s, ep, sig, d in all_cands_f:
                if s in held_si: continue
                if len(selected) >= top_n - len(positions): break
                # Check correlation with already-selected
                corr_ok = True
                for sel in selected:
                    if get_corr(sel[1], s, di) > max_corr:
                        corr_ok = False
                        break
                if corr_ok:
                    selected.append((sc, s, ep, sig, d))

            # Enter positions
            for sc, s, pr, sig_str, d in selected:
                if s in set(p['si'] for p in positions): continue
                if len(positions) >= top_n: break
                cap = cash * pos_size / max(len(selected), 1)
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
                    cash -= ci
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
    DD_TIERS = [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)]
    BASE_KWARGS = dict(
        atr_norm_max=10.0, max_corr=0.5,
        dd_tiers=DD_TIERS,
        regime_lo=0.5, regime_hi=1.5,
        sl_pct=0.0, hold=1, top_n=3,
    )

    all_results = []

    # ===================== TEST 1: BASELINE (V177 dual-side) =====================
    print("\n" + "=" * 130)
    print("  TEST 1: BASELINE — V177 Dual-Side (short_mirror)")
    print("=" * 130)

    r_base = backtest(signal_mode='baseline', **BASE_KWARGS)
    pr(r_base, "BASELINE: V121 long + short mirror, atr<10%, corr=0.5, top_n=3")
    all_results.append({**r_base, 'label': 'BASELINE_V177', 'signal_mode': 'baseline'})

    # ===================== TEST 2: GAP_FOLLOW =====================
    print("\n" + "=" * 130)
    print("  TEST 2: GAP_FOLLOW — Overnight Gap Continuation")
    print("  Long when gap>0.5% + intraday>0, Short when gap<-0.5% + intraday<0")
    print("=" * 130)

    r_gf = backtest(signal_mode='gap_follow', **BASE_KWARGS)
    pr(r_gf, "GAP_FOLLOW: gap continuation, atr<10%, corr=0.5, top_n=3")
    all_results.append({**r_gf, 'label': 'GAP_FOLLOW', 'signal_mode': 'gap_follow'})

    # ===================== TEST 3: GAP_REVERSAL =====================
    print("\n" + "=" * 130)
    print("  TEST 3: GAP_REVERSAL — Gap Fill / Reversal")
    print("  Short when gap>1.5% + intraday<-0.3%, Long when gap<-1.5% + intraday>0.3%")
    print("=" * 130)

    r_gr = backtest(signal_mode='gap_reversal', **BASE_KWARGS)
    pr(r_gr, "GAP_REVERSAL: gap fill, atr<10%, corr=0.5, top_n=3")
    all_results.append({**r_gr, 'label': 'GAP_REVERSAL', 'signal_mode': 'gap_reversal'})

    # ===================== TEST 4: GAP_MOMENTUM =====================
    print("\n" + "=" * 130)
    print("  TEST 4: GAP_MOMENTUM — Gap + ROC(5) + Z-Score")
    print("  Long: gap>0.3% + ROC(5)>1% + Z>1.5, Short: gap<-0.3% + ROC(5)<-1% + Z<-1.5")
    print("=" * 130)

    r_gm = backtest(signal_mode='gap_momentum', **BASE_KWARGS)
    pr(r_gm, "GAP_MOMENTUM: gap + ROC + Z, atr<10%, corr=0.5, top_n=3")
    all_results.append({**r_gm, 'label': 'GAP_MOMENTUM', 'signal_mode': 'gap_momentum'})

    # ===================== TEST 5: GAP_VOL =====================
    print("\n" + "=" * 130)
    print("  TEST 5: GAP_VOL — Gap + Volume Confirmation")
    print("  Long: gap>0.3% + vol>1.5x avg, Short: gap<-0.3% + vol>1.5x avg")
    print("=" * 130)

    r_gv = backtest(signal_mode='gap_vol', **BASE_KWARGS)
    pr(r_gv, "GAP_VOL: gap + volume surge, atr<10%, corr=0.5, top_n=3")
    all_results.append({**r_gv, 'label': 'GAP_VOL', 'signal_mode': 'gap_vol'})

    # ===================== TEST 5b: INTRADAY_BREAKOUT =====================
    print("\n" + "=" * 130)
    print("  TEST 5b: INTRADAY_BREAKOUT — Intraday Range + Close Position + Volume")
    print("  Long: range>2% + close>90% range + vol>1.5x, Short: range>2% + close<10% + vol>1.5x")
    print("=" * 130)

    r_ib = backtest(signal_mode='iday_breakout', **BASE_KWARGS)
    pr(r_ib, "INTRADAY_BREAKOUT: range + close_pos + vol, atr<10%, corr=0.5, top_n=3")
    all_results.append({**r_ib, 'label': 'INTRADAY_BREAKOUT', 'signal_mode': 'iday_breakout'})

    # ===================== TEST 6: COMBINED (V121 + GAP_MOMENTUM) =====================
    print("\n" + "=" * 130)
    print("  TEST 6: COMBINED — V121 + GAP_MOMENTUM (50/50 weighting)")
    print("=" * 130)

    r_comb = backtest(signal_mode='combined', **BASE_KWARGS)
    pr(r_comb, "COMBINED: V121 50% + GAP_MOMENTUM 50%, atr<10%, corr=0.5, top_n=3")
    all_results.append({**r_comb, 'label': 'COMBINED_V121_GAPMOM', 'signal_mode': 'combined'})

    # ===================== TEST 7: UNION ALL =====================
    print("\n" + "=" * 130)
    print("  TEST 7: UNION_ALL — All signals aggregated (V121 + all gap signals)")
    print("=" * 130)

    r_union = backtest(signal_mode='union_all', **BASE_KWARGS)
    pr(r_union, "UNION_ALL: V121 + all gap signals, atr<10%, corr=0.5, top_n=3")
    all_results.append({**r_union, 'label': 'UNION_ALL', 'signal_mode': 'union_all'})

    # ===================== SENSITIVITY: vary atr_norm_max =====================
    print("\n" + "=" * 130)
    print("  SENSITIVITY: Best gap signal with different ATR thresholds")
    print("=" * 130)

    # Determine best gap signal so far
    gap_modes = ['gap_follow', 'gap_reversal', 'gap_momentum', 'gap_vol', 'iday_breakout']
    gap_results = {r['signal_mode']: r for r in all_results if r['signal_mode'] in gap_modes}
    best_gap_mode = max(gap_results.items(), key=lambda x: abs(x[1]['ann'] / x[1]['mdd']) if x[1]['mdd'] != 0 else 0)[0]
    print(f"\n  Best gap signal: {best_gap_mode}")

    for atr_max in [8.0, 12.0, 15.0]:
        kw = {**BASE_KWARGS, 'atr_norm_max': atr_max}
        r = backtest(signal_mode=best_gap_mode, **kw)
        pr(r, f"  {best_gap_mode} atr<{atr_max:.0f}%")
        all_results.append({**r, 'label': f'{best_gap_mode}_atr{atr_max:.0f}', 'signal_mode': best_gap_mode})

    # ===================== SENSITIVITY: combined with different weights =====================
    print("\n" + "=" * 130)
    print("  SENSITIVITY: COMBINED with different ATR thresholds")
    print("=" * 130)

    for atr_max in [8.0, 12.0]:
        kw = {**BASE_KWARGS, 'atr_norm_max': atr_max}
        r = backtest(signal_mode='combined', **kw)
        pr(r, f"  COMBINED atr<{atr_max:.0f}%")
        all_results.append({**r, 'label': f'COMBINED_atr{atr_max:.0f}', 'signal_mode': 'combined'})

    for atr_max in [8.0, 12.0]:
        kw = {**BASE_KWARGS, 'atr_norm_max': atr_max}
        r = backtest(signal_mode='union_all', **kw)
        pr(r, f"  UNION_ALL atr<{atr_max:.0f}%")
        all_results.append({**r, 'label': f'UNION_ALL_atr{atr_max:.0f}', 'signal_mode': 'union_all'})

    # ===================== WALK-FORWARD =====================
    print("\n" + "=" * 130)
    print("  WALK-FORWARD: All signal modes")
    print("=" * 130)

    base_ann = r_base['ann']; base_mdd = r_base['mdd']
    base_rm = abs(base_ann / base_mdd) if base_mdd != 0 else 0
    print(f"\n  Baseline: Ann={base_ann:+.0f}% | MDD={base_mdd:.0f}% | R/M={base_rm:.2f}")

    print(f"\n  --- BASELINE WF ---")
    wf_base = walk_forward(label="BASELINE V177", signal_mode='baseline', **BASE_KWARGS)
    print_wf(wf_base, "BASELINE")

    for mode in ['gap_follow', 'gap_reversal', 'gap_momentum', 'gap_vol', 'iday_breakout', 'combined', 'union_all']:
        print(f"\n  --- {mode.upper()} WF ---")
        wf = walk_forward(label=mode, signal_mode=mode, **BASE_KWARGS)
        print_wf(wf, mode.upper())

    # Best gap mode at different ATR
    print(f"\n  --- {best_gap_mode.upper()} ATR=12 WF ---")
    kw12 = {**BASE_KWARGS, 'atr_norm_max': 12.0}
    wf = walk_forward(label=f"{best_gap_mode}_atr12", signal_mode=best_gap_mode, **kw12)
    print_wf(wf, f"{best_gap_mode.upper()} ATR=12")

    print(f"\n  --- COMBINED ATR=12 WF ---")
    wf = walk_forward(label="COMBINED_atr12", signal_mode='combined', **kw12)
    print_wf(wf, "COMBINED ATR=12")

    print(f"\n  --- UNION_ALL ATR=12 WF ---")
    wf = walk_forward(label="UNION_ALL_atr12", signal_mode='union_all', **kw12)
    print_wf(wf, "UNION_ALL ATR=12")

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 130)
    print("  V181 FINAL SUMMARY: Overnight Gap + Intraday Momentum")
    print("=" * 130)

    print(f"\n  {'Config':45s} | {'Ann':>7s} | {'MDD':>5s} | {'R/M':>5s} | {'WR':>5s} | {'N':>4s} | Delta_R/M")
    print(f"  {'-'*45}-+-{'-'*7}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*4}-+-{'-'*8}")
    print(f"  {'BASELINE (V177 dual-side)':45s} | {base_ann:>+7.0f}% | {base_mdd:>5.0f}% | {base_rm:>5.2f} | {r_base['wr']:>5.1f}% | {r_base['n']:>4d} |    ---")

    ranked = sorted(all_results, key=lambda x: abs(x.get('ann', 0) / x.get('mdd', -1)) if x.get('mdd', 0) != 0 else 0, reverse=True)
    for r in ranked[:20]:
        if r.get('signal_mode') == 'baseline': continue
        ann = r['ann']; mdd = r['mdd']
        rm = abs(ann / mdd) if mdd != 0 else 0
        delta = rm - base_rm
        print(f"  {r['label']:45s} | {ann:>+7.0f}% | {mdd:>5.0f}% | {rm:>5.2f} | {r['wr']:>5.1f}% | {r['n']:>4d} | {delta:>+8.2f}")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
