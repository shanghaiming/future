"""
Alpha Futures V156 — Trend Acceleration + Kitchen Sink Sizing
=============================================================================
Combines V146's best sizing (Kitchen Sink: DD tiers × WR-adaptive × regime)
with V153's trend acceleration filter (ROC5 > ROC10 > ROC20).

V146 champion: +185% WF avg, -24% worst WF MDD (no accel filter)
V153 trend accel: +86% WF avg, -16% worst WF MDD (simple DD sizing)

Can Kitchen Sink sizing amplify the acceleration filter's signal quality?
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
    print("  V146 — FINAL COMBINATION: Best of V140-V145")
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

    # 20-day returns for correlation calculation
    RET20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(20, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-20]) and c[di-20] > 0:
                RET20[si, di] = (c[di] / c[di-20] - 1) * 100

    # ===================== REGIME INDICATORS =====================
    print("  Computing regime indicators...", flush=True)

    # Market Breadth: fraction of commodities with positive 5-day ROC
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

    # Market Volatility: 20-day rolling std of equal-weighted market return
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
    print(f"  Market vol median: {VOL_MEDIAN:.4f}%")

    print(f"  Done ({time.time()-t0:.1f}s)")

    # ===================== SIGNAL DEFINITIONS =====================
    def sig_v121(di, edi, accel_filter=False):
        c = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue
            # Trend acceleration: ROC5 > ROC10 > ROC20
            if accel_filter:
                roc10 = ROC10[s, di]; roc20 = ROC20[s, di]
                if np.isnan(roc10) or np.isnan(roc20): continue
                if not (roc > roc10 > roc20): continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((roc * zs, s, ep, 'v121'))
        return c

    def sig_ov_id(di, edi, accel_filter=False):
        c = []
        for s in range(NS):
            ov = OV_GAP[s, di]; idr = ID_RET[s, di]; roc = ROC5[s, di]
            if any(np.isnan(x) for x in [ov, idr, roc]): continue
            if ov <= 0.3 or idr <= 0.3 or roc <= 1.0: continue
            # Trend acceleration for OV/ID too
            if accel_filter:
                roc10 = ROC10[s, di]; roc20 = ROC20[s, di]
                if np.isnan(roc10) or np.isnan(roc20): continue
                if not (roc > roc10 > roc20): continue
            zs = ZSCORE[s, di]
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            z_bonus = zs if not np.isnan(zs) and zs > 1.0 else 1.0
            c.append(((ov + idr) * roc * z_bonus * 2, s, ep, 'ov_id'))
        return c

    def sig_final_flag(di, edi, accel_filter=False):
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

    def sig_union(di, edi, accel_filter=False):
        all_sigs = {}
        for item in sig_v121(di, edi, accel_filter):
            sc, s, ep, st = item
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc * 3
            all_sigs[s][2].append('v121')
        for item in sig_ov_id(di, edi, accel_filter):
            sc, s, ep, st = item
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc * 2
            all_sigs[s][2].append('ov_id')
        for item in sig_final_flag(di, edi, accel_filter):
            sc, s, ep, st = item
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc
            all_sigs[s][2].append('ff')
        return [(sc, s, ep, '+'.join(sigs)) for s, (sc, ep, sigs) in all_sigs.items()]

    # ===================== HELPER: Correlation between two commodities =====================
    def get_corr(si_a, si_b, di, window=20):
        """Fix Bug #2: use daily returns instead of overlapping 20-day returns."""
        start_idx = max(0, di - window)
        ret_a = RET[si_a, start_idx:di]
        ret_b = RET[si_b, start_idx:di]
        valid = ~(np.isnan(ret_a) | np.isnan(ret_b))
        n_valid = np.sum(valid)
        if n_valid < 8:
            return 0.5  # default moderate
        ra = ret_a[valid]; rb = ret_b[valid]
        if np.std(ra) == 0 or np.std(rb) == 0:
            return 0.5
        c = np.corrcoef(ra, rb)[0, 1]
        return c if not np.isnan(c) else 0.5

    # ===================== HELPER: Compute composite regime score =====================
    def compute_composite(di, daily_eq, high_water, perf_window=20):
        scores = []

        # A. Breadth normalized to 0..1
        bth = BREADTH[di]
        if not np.isnan(bth):
            scores.append(np.clip((bth - 0.4) / (0.7 - 0.4), 0, 1))

        # B. Vol normalized (low vol=1, high vol=0)
        vol = MKT_VOL[di]
        if not np.isnan(vol) and VOL_MEDIAN > 0:
            vol_ratio = vol / VOL_MEDIAN
            scores.append(np.clip((1.5 - vol_ratio) / (1.5 - 0.8), 0, 1))

        # C. Equity curve slope normalized
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

        # D. DD normalized (new high=1, deep DD=0)
        if high_water > 0:
            cur_dd = (daily_eq[-1] - high_water) / high_water
        else:
            cur_dd = 0
        scores.append(np.clip(1.0 + cur_dd / 0.3, 0, 1))

        return np.mean(scores) if scores else 0.5

    # ===================== HELPER: DD-based sizing =====================
    def dd_size(pv, high_water, tiers):
        """
        tiers: list of (dd_threshold, size_frac) sorted from 0 DD upward
        e.g. [(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)]
        """
        if high_water <= 0:
            return tiers[0][1]
        dd = (pv - high_water) / high_water
        for dd_thresh, size_frac in tiers:
            if dd >= -dd_thresh:
                return size_frac
        return tiers[-1][1]

    # ===================== HELPER: WR-adaptive sizing =====================
    def wr_size(trades, window=20):
        if len(trades) < window:
            return 1.0  # default multiplier
        recent = trades[-window:]
        wr = np.mean([1 if t > 0 else 0 for t in recent])
        if wr > 0.65:
            return 1.3
        elif wr >= 0.50:
            return 1.0
        else:
            return 0.5

    # ===================== UNIFIED BACKTEST ENGINE =====================
    def backtest_v146(start_di=MIN_TRAIN, end_di=None,
                      # Signal selection
                      mode='cross_corr',  # 'cross_corr','portfolio','strength'
                      # Trend acceleration filter
                      accel_filter=False,
                      # Correlation filter
                      max_corr=0.5,
                      # Sizing mode
                      sizing='fixed',  # 'fixed','regime_combo','dd_tiers','wr_dd','kitchen_sink'
                      # Fixed sizing params
                      base_size=0.55,
                      # Regime combo sizing thresholds
                      regime_high=0.70, regime_mid=0.40,
                      size_regime_high=0.70, size_regime_mid=0.55, size_regime_low=0.30,
                      # DD tiers: list of (dd_threshold, size)
                      dd_tiers=None,
                      # WR-DD adaptive
                      wr_base=0.50,
                      # Stop-loss
                      sl_pct=0.0,  # intraday stop-loss fraction (0=off)
                      # General
                      hold=1, top_n=2):
        if end_di is None: end_di = ND
        if dd_tiers is None:
            dd_tiers = [(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)]

        cash = float(CASH0)
        positions = []
        trades = []  # list of pnl_pct
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

            # --- Compute position size based on sizing mode ---
            if sizing == 'fixed':
                pos_size = base_size

            elif sizing == 'regime_combo':
                composite = compute_composite(di, daily_eq, high_water)
                if composite > regime_high:
                    pos_size = size_regime_high
                elif composite > regime_mid:
                    pos_size = size_regime_mid
                else:
                    pos_size = size_regime_low

            elif sizing == 'dd_tiers':
                pos_size = dd_size(pv, high_water, dd_tiers)

            elif sizing == 'wr_dd':
                wr_mult = wr_size(trades, window=20)
                dd_mult_val = dd_size(pv, high_water, [
                    (0, 1.2), (0.10, 1.0), (0.20, 0.5)
                ])
                pos_size = wr_base * wr_mult * dd_mult_val

            elif sizing == 'kitchen_sink':
                # DD tier sizing
                dd_sz = dd_size(pv, high_water, dd_tiers)
                # WR multiplier
                wr_mult_val = wr_size(trades, window=20)
                # Regime combo multiplier (scale 0.5-1.5)
                composite = compute_composite(di, daily_eq, high_water)
                regime_mult = 0.5 + composite  # composite 0-1 -> 0.5-1.5
                pos_size = dd_sz * wr_mult_val * regime_mult

            else:
                pos_size = base_size

            # Clamp
            pos_size = max(0.05, min(0.95, pos_size))

            # --- Enter positions ---
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue

            held_si = set(p['si'] for p in positions)

            if mode == 'cross_corr':
                # Get best V121 and best Union signal
                cands_v121 = sig_v121(di, edi, accel_filter)
                cands_union = sig_union(di, edi, accel_filter)
                cands_v121.sort(key=lambda x: -x[0])
                cands_union.sort(key=lambda x: -x[0])

                best_v121 = None
                for c in cands_v121:
                    if c[1] not in held_si:
                        best_v121 = c
                        break

                best_union = None
                for c in cands_union:
                    if c[1] not in held_si:
                        best_union = c
                        break

                entries = []
                if best_v121 and best_union:
                    if best_v121[1] == best_union[1]:
                        # Same commodity: single position at boosted size
                        entries.append((best_v121[0], best_v121[1], best_v121[2],
                                        'v121+union', pos_size * 1.5))
                    else:
                        # Different commodities: check correlation
                        corr = get_corr(best_v121[1], best_union[1], di)
                        if corr < max_corr:
                            entries.append((best_v121[0], best_v121[1], best_v121[2],
                                            'v121', pos_size))
                            entries.append((best_union[0], best_union[1], best_union[2],
                                            'union', pos_size))
                        else:
                            # High correlation: take only the best
                            best = best_v121 if best_v121[0] >= best_union[0] else best_union
                            entries.append((best[0], best[1], best[2], 'best', pos_size))
                elif best_v121:
                    entries.append((best_v121[0], best_v121[1], best_v121[2], 'v121', pos_size))
                elif best_union:
                    entries.append((best_union[0], best_union[1], best_union[2], 'union', pos_size))

            elif mode == 'strength':
                # Signal Strength Scaling
                cands_v121 = sig_v121(di, edi, accel_filter)
                cands_union = sig_union(di, edi, accel_filter)
                cands_v121.sort(key=lambda x: -x[0])
                cands_union.sort(key=lambda x: -x[0])

                best_v121 = None
                for c in cands_v121:
                    if c[1] not in held_si:
                        best_v121 = c
                        break

                best_union = None
                for c in cands_union:
                    if c[1] not in held_si:
                        best_union = c
                        break

                entries = []
                if best_v121 and best_union:
                    if best_v121[1] == best_union[1]:
                        # Dual signal agreement: 80% size
                        entries.append((best_v121[0], best_v121[1], best_v121[2],
                                        'dual', pos_size * 1.6))
                    else:
                        # Different commodities with low corr -> 55% each
                        corr = get_corr(best_v121[1], best_union[1], di)
                        if corr < max_corr:
                            entries.append((best_v121[0], best_v121[1], best_v121[2],
                                            'v121', pos_size * 1.1))
                            entries.append((best_union[0], best_union[1], best_union[2],
                                            'union', pos_size * 1.1))
                        else:
                            # High correlation: skip weaker
                            best = best_v121 if best_v121[0] >= best_union[0] else best_union
                            entries.append((best[0], best[1], best[2], 'best', pos_size))
                elif best_v121:
                    entries.append((best_v121[0], best_v121[1], best_v121[2], 'v121', pos_size))
                elif best_union:
                    entries.append((best_union[0], best_union[1], best_union[2], 'union', pos_size))

            elif mode == 'portfolio':
                # Simple portfolio: run both signals, size from portfolio equity
                cands_v121 = sig_v121(di, edi, accel_filter)
                cands_union = sig_union(di, edi, accel_filter)
                cands_v121.sort(key=lambda x: -x[0])
                cands_union.sort(key=lambda x: -x[0])

                best_v121 = None
                for c in cands_v121:
                    if c[1] not in held_si:
                        best_v121 = c
                        break

                best_union = None
                for c in cands_union:
                    if c[1] not in held_si:
                        best_union = c
                        break

                entries = []
                if best_v121 and best_union:
                    if best_v121[1] == best_union[1]:
                        entries.append((best_v121[0], best_v121[1], best_v121[2],
                                        'v121+union', pos_size * 1.2))
                    else:
                        entries.append((best_v121[0], best_v121[1], best_v121[2],
                                        'v121', pos_size))
                        entries.append((best_union[0], best_union[1], best_union[2],
                                        'union', pos_size))
                elif best_v121:
                    entries.append((best_v121[0], best_v121[1], best_v121[2], 'v121', pos_size))
                elif best_union:
                    entries.append((best_union[0], best_union[1], best_union[2], 'union', pos_size))

            else:
                entries = []

            cash_snapshot = cash  # Fix Bug #1: snapshot before any allocation
            n_planned = len(entries)
            for sc, s, pr, sig_str, pct in entries:
                if s in set(p['si'] for p in positions): continue
                if len(positions) >= top_n: break
                cap = cash_snapshot * pct / n_planned  # Equal split among planned entries
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
        print(f"  {label:75s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d}")

    def walk_forward(mode='cross_corr', sizing='fixed', max_corr=0.5,
                     accel_filter=False,
                     base_size=0.55, hold=1, top_n=2, sl_pct=0.0,
                     regime_high=0.70, regime_mid=0.40,
                     size_regime_high=0.70, size_regime_mid=0.55, size_regime_low=0.30,
                     dd_tiers=None, wr_base=0.50, label=""):
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue
            r = backtest_v146(start_di=ys, end_di=ye, mode=mode, sizing=sizing,
                              max_corr=max_corr, accel_filter=accel_filter,
                              base_size=base_size, hold=hold,
                              top_n=top_n, sl_pct=sl_pct,
                              regime_high=regime_high, regime_mid=regime_mid,
                              size_regime_high=size_regime_high,
                              size_regime_mid=size_regime_mid,
                              size_regime_low=size_regime_low,
                              dd_tiers=dd_tiers, wr_base=wr_base)
            res[yr] = r
        return res

    def print_wf(wf_res, label=""):
        pos = sum(1 for r in wf_res.values() if r['ann'] > 0)
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"    {label:75s}")
        print(f"      {pos}/6 pos | Avg={avg_ann:>+7.0f}% | WorstWfMDD={worst_mdd:>5.0f}%")
        print(f"      {ws}")

    # ===================== SECTION 0: BASELINES =====================
    print("\n" + "=" * 120)
    print("  SECTION 0: BASELINES (V146 champion — no accel filter)")
    print("=" * 120)

    dd_champ = [(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)]
    dd_aggro = [(0, 0.80), (0.10, 0.65), (0.20, 0.45), (0.30, 0.25)]

    # V146 champion: Kitchen Sink, no accel
    r = backtest_v146(mode='cross_corr', sizing='kitchen_sink',
                      dd_tiers=dd_champ, max_corr=0.5, sl_pct=0.03,
                      accel_filter=False)
    pr(r, "V146 champ: Sink DD70/60/40/20 corr<0.5 SL3% NO accel")

    r = backtest_v146(mode='cross_corr', sizing='kitchen_sink',
                      dd_tiers=dd_aggro, max_corr=0.5, sl_pct=0.05,
                      accel_filter=False)
    pr(r, "V146 champ: Sink DD80/65/45/25 corr<0.5 SL5% NO accel")

    # ===================== SECTION 1: ACCELERATION + DD TIERS =====================
    print("\n" + "=" * 120)
    print("  SECTION 1: Trend Acceleration + DD-Based Sizing")
    print("  Filter: ROC5 > ROC10 > ROC20 (trend accelerating)")
    print("=" * 120)

    for accel in [False, True]:
        accel_str = "ACCEL" if accel else "NO accel"
        for dd_t, dd_name in [(dd_champ, "70/60/40/20"), (dd_aggro, "80/65/45/25")]:
            for sl in [0.0, 0.03, 0.05]:
                sl_str = "NO SL" if sl == 0 else f"SL{sl*100:.0f}%"
                r = backtest_v146(mode='cross_corr', sizing='dd_tiers',
                                  dd_tiers=dd_t, max_corr=0.5, sl_pct=sl,
                                  accel_filter=accel)
                pr(r, f"DD{dd_name} corr<0.5 {sl_str} {accel_str}")

    # ===================== SECTION 2: ACCELERATION + KITCHEN SINK =====================
    print("\n" + "=" * 120)
    print("  SECTION 2: Trend Acceleration + Kitchen Sink Sizing")
    print("  pos_size = dd_size * wr_mult * regime_mult")
    print("=" * 120)

    for accel in [False, True]:
        accel_str = "ACCEL" if accel else "NO accel"
        for dd_t, dd_name in [(dd_champ, "70/60/40/20"), (dd_aggro, "80/65/45/25")]:
            for sl in [0.0, 0.03, 0.05]:
                sl_str = "NO SL" if sl == 0 else f"SL{sl*100:.0f}%"
                r = backtest_v146(mode='cross_corr', sizing='kitchen_sink',
                                  dd_tiers=dd_t, max_corr=0.5, sl_pct=sl,
                                  accel_filter=accel)
                pr(r, f"Sink DD{dd_name} corr<0.5 {sl_str} {accel_str}")

    # ===================== SECTION 3: ACCELERATION + REGIME COMBO =====================
    print("\n" + "=" * 120)
    print("  SECTION 3: Trend Acceleration + Regime Combo Sizing")
    print("=" * 120)

    for accel in [False, True]:
        accel_str = "ACCEL" if accel else "NO accel"
        r = backtest_v146(mode='cross_corr', sizing='regime_combo',
                          regime_high=0.70, regime_mid=0.40,
                          size_regime_high=0.70, size_regime_mid=0.55,
                          size_regime_low=0.30, max_corr=0.5,
                          accel_filter=accel)
        pr(r, f"Regime 70/55/30 corr<0.5 {accel_str}")

    # ===================== SECTION 4: WF VALIDATION =====================
    print("\n" + "=" * 120)
    print("  WALK-FORWARD VALIDATION — BEST CONFIGS")
    print("=" * 120)

    # All Kitchen Sink configs with and without accel
    wf_configs = [
        # (accel, dd_tiers, sl, label)
        (False, dd_champ, 0.03, "Sink DD70/60/40/20 SL3% NO accel"),
        (False, dd_aggro, 0.05, "Sink DD80/65/45/25 SL5% NO accel"),
        (True,  dd_champ, 0.0,  "Sink DD70/60/40/20 NO SL ACCEL"),
        (True,  dd_champ, 0.03, "Sink DD70/60/40/20 SL3% ACCEL"),
        (True,  dd_champ, 0.05, "Sink DD70/60/40/20 SL5% ACCEL"),
        (True,  dd_aggro, 0.0,  "Sink DD80/65/45/25 NO SL ACCEL"),
        (True,  dd_aggro, 0.03, "Sink DD80/65/45/25 SL3% ACCEL"),
        (True,  dd_aggro, 0.05, "Sink DD80/65/45/25 SL5% ACCEL"),
        # DD tiers only
        (True,  dd_champ, 0.0,  "DD-only DD70/60/40/20 NO SL ACCEL"),
        (True,  dd_aggro, 0.0,  "DD-only DD80/65/45/25 NO SL ACCEL"),
    ]

    wf_all = {}
    for accel, dd_t, sl, label in wf_configs:
        sizing = 'kitchen_sink' if 'Sink' in label else 'dd_tiers'
        wf_res = walk_forward(mode='cross_corr', sizing=sizing,
                              max_corr=0.5, accel_filter=accel,
                              dd_tiers=dd_t, sl_pct=sl, label=label)
        wf_all[label] = wf_res
        print_wf(wf_res, label)

    # ===================== HIGHLIGHT =====================
    print("\n" + "=" * 120)
    print("  HIGHLIGHT: Best WF configs")
    print("=" * 120)

    # Compare ACCEL vs NO ACCEL
    print(f"\n  --- ACCEL vs NO ACCEL comparison (Kitchen Sink):")
    for accel_str, filter_val in [("NO ACCEL", False), ("ACCEL", True)]:
        best_ann = 0; best_label = ""; best_wf = None
        for label, wf_res in wf_all.items():
            if ('ACCEL' if filter_val else 'NO accel') not in label:
                continue
            avg = np.mean([r['ann'] for r in wf_res.values()])
            if avg > best_ann:
                best_ann = avg; best_label = label; best_wf = wf_res
        if best_wf:
            worst_mdd = min(r['mdd'] for r in best_wf.values())
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(best_wf.items())])
            print(f"  {accel_str} best: {best_label}")
            print(f"    AvgWF={best_ann:+.0f}% | WorstWfMDD={worst_mdd:.1f}%")
            print(f"    {ws}")

    # Show all configs above +150% WF avg
    print(f"\n  --- All configs with WF avg >= +150%:")
    for label, wf_res in sorted(wf_all.items()):
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        if avg_ann >= 150:
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(wf_res.items())])
            print(f"  *** {label}")
            print(f"      AvgWF={avg_ann:+.0f}% | WorstWfMDD={worst_mdd:.1f}%")
            print(f"      {ws}")

    # Show all configs with WF MDD > -20%
    print(f"\n  --- All configs with worst WF MDD > -20%:")
    for label, wf_res in sorted(wf_all.items()):
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        if worst_mdd > -20:
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(wf_res.items())])
            print(f"  *** {label}")
            print(f"      AvgWF={avg_ann:+.0f}% | WorstWfMDD={worst_mdd:.1f}%")
            print(f"      {ws}")

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 120)
    print("  FINAL SUMMARY")
    print("=" * 120)

    print(f"\n  V146 champion (no accel): +185% WF avg, -24% worst WF MDD")
    print(f"  V153 trend accel (simple DD): +86% WF avg, -16% worst WF MDD")
    print(f"\n  V156 tested Kitchen Sink + trend acceleration filter")

    # Find best combo
    best_combo = max(wf_all.items(),
                     key=lambda x: np.mean([r['ann'] for r in x[1].values()]))
    label, wf_res = best_combo
    avg_ann = np.mean([r['ann'] for r in wf_res.values()])
    worst_mdd = min(r['mdd'] for r in wf_res.values())
    ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                     for yr, r in sorted(wf_res.items())])
    print(f"\n  Best V156: {label}")
    print(f"    AvgWF={avg_ann:+.0f}% | WorstWfMDD={worst_mdd:.1f}%")
    print(f"    {ws}")

    # Best risk-adjusted
    best_ra = min(wf_all.items(),
                  key=lambda x: min(r['mdd'] for r in x[1].values()))
    label, wf_res = best_ra
    avg_ann = np.mean([r['ann'] for r in wf_res.values()])
    worst_mdd = min(r['mdd'] for r in wf_res.values())
    print(f"\n  Best risk-adjusted V156: {label}")
    print(f"    AvgWF={avg_ann:+.0f}% | WorstWfMDD={worst_mdd:.1f}%")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
