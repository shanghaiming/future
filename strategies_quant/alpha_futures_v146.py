"""
Alpha Futures V146 — FINAL COMBINATION: Best of V140-V145
=============================================================================
Goal: Maximize return/MDD frontier by combining TOP ideas:
  - V142 Cross+Corr: V121+Union with correlation filter for diversification
  - V145 Combo Regime: Composite regime score for dynamic sizing
  - V145 DD Portfolio: DD-based sizing tiers
  - WR-adaptive sizing from rolling trade history

Test 5 combinations:
  1. Cross+Corr + Regime Combo Sizing
  2. Cross+Corr + DD-Based Sizing + SL=5%
  3. Portfolio with WR-Adaptive + DD-Adaptive
  4. Cross+Corr with Signal Strength Scaling
  5. The "Kitchen Sink" (Cross+Corr + DD sizing + WR-adaptive + SL=5%)
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
    ROC20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100
        ROC5[si] = talib.ROC(c, timeperiod=5)
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
                cands_v121 = sig_v121(di, edi)
                cands_union = sig_union(di, edi)
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
                cands_v121 = sig_v121(di, edi)
                cands_union = sig_union(di, edi)
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
                cands_v121 = sig_v121(di, edi)
                cands_union = sig_union(di, edi)
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
                              max_corr=max_corr, base_size=base_size, hold=hold,
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
    print("\n" + "=" * 130)
    print("  SECTION 0: BASELINES")
    print("=" * 130)

    # Cross+Corr with fixed sizing at various levels
    for bs in [0.45, 0.50, 0.55]:
        r = backtest_v146(mode='cross_corr', sizing='fixed', base_size=bs, max_corr=0.5)
        pr(r, f"Cross+Corr fixed {bs*100:.0f}% corr<0.5")

    # Baseline with no corr filter
    r = backtest_v146(mode='cross_corr', sizing='fixed', base_size=0.55, max_corr=1.0)
    pr(r, "Cross+Corr fixed 55% corr<1.0 (no filter)")

    # ===================== TEST 1: Cross+Corr + Regime Combo Sizing =====================
    print("\n" + "=" * 130)
    print("  TEST 1: Cross+Corr + Regime Combo Sizing")
    print("  Composite score > 0.7: 70% | 0.4-0.7: 55% | < 0.4: 30%")
    print("=" * 130)

    test1_configs = [
        # (regime_high, regime_mid, size_high, size_mid, size_low, max_corr, label)
        (0.70, 0.40, 0.70, 0.55, 0.30, 0.5, "T1: Regime 0.7/0.4 -> 70/55/30% corr<0.5"),
        (0.70, 0.40, 0.70, 0.55, 0.30, 0.6, "T1: Regime 0.7/0.4 -> 70/55/30% corr<0.6"),
        (0.70, 0.40, 0.70, 0.55, 0.30, 0.7, "T1: Regime 0.7/0.4 -> 70/55/30% corr<0.7"),
        (0.65, 0.35, 0.75, 0.55, 0.30, 0.5, "T1: Regime 0.65/0.35 -> 75/55/30% corr<0.5"),
        (0.75, 0.45, 0.70, 0.55, 0.35, 0.5, "T1: Regime 0.75/0.45 -> 70/55/35% corr<0.5"),
        (0.70, 0.40, 0.80, 0.55, 0.25, 0.5, "T1: Regime 0.7/0.4 -> 80/55/25% corr<0.5"),
        (0.70, 0.40, 0.65, 0.55, 0.35, 0.5, "T1: Regime 0.7/0.4 -> 65/55/35% corr<0.5"),
    ]

    test1_results = []
    for rh, rm, sh, sm, sl, mc, label in test1_configs:
        r = backtest_v146(mode='cross_corr', sizing='regime_combo',
                          regime_high=rh, regime_mid=rm,
                          size_regime_high=sh, size_regime_mid=sm, size_regime_low=sl,
                          max_corr=mc)
        r['desc'] = label
        test1_results.append(r)
        pr(r, label)

    # ===================== TEST 2: Cross+Corr + DD-Based Sizing + SL=5% =====================
    print("\n" + "=" * 130)
    print("  TEST 2: Cross+Corr + DD-Based Sizing + SL=5%")
    print("  DD tiers: 0%->70%, 10%->60%, 20%->40%, 30%->20% + 5% intraday SL")
    print("=" * 130)

    test2_configs = [
        # (dd_tiers, max_corr, sl, label)
        ([(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)], 0.5, 0.05,
         "T2: DD 70/60/40/20% corr<0.5 SL=5%"),
        ([(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)], 0.6, 0.05,
         "T2: DD 70/60/40/20% corr<0.6 SL=5%"),
        ([(0, 0.65), (0.10, 0.55), (0.20, 0.40), (0.30, 0.25)], 0.5, 0.05,
         "T2: DD 65/55/40/25% corr<0.5 SL=5%"),
        ([(0, 0.75), (0.10, 0.65), (0.20, 0.50), (0.30, 0.35)], 0.5, 0.05,
         "T2: DD 75/65/50/35% corr<0.5 SL=5%"),
        ([(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)], 0.5, 0.03,
         "T2: DD 70/60/40/20% corr<0.5 SL=3%"),
        ([(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)], 0.5, 0.08,
         "T2: DD 70/60/40/20% corr<0.5 SL=8%"),
        ([(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)], 0.5, 0.0,
         "T2: DD 70/60/40/20% corr<0.5 NO SL"),
        # More aggressive DD tiers
        ([(0, 0.75), (0.10, 0.65), (0.20, 0.50), (0.30, 0.35), (0.40, 0.20)], 0.5, 0.05,
         "T2: DD 75/65/50/35/20% corr<0.5 SL=5%"),
    ]

    test2_results = []
    for dd_t, mc, sl, label in test2_configs:
        r = backtest_v146(mode='cross_corr', sizing='dd_tiers',
                          dd_tiers=dd_t, max_corr=mc, sl_pct=sl)
        r['desc'] = label
        test2_results.append(r)
        pr(r, label)

    # ===================== TEST 3: Portfolio with WR-Adaptive + DD-Adaptive =====================
    print("\n" + "=" * 130)
    print("  TEST 3: Portfolio with WR-Adaptive + DD-Adaptive")
    print("  size = base * WR_mult * DD_mult | base=50%")
    print("  WR: >65%->1.3, 50-65%->1.0, <50%->0.5")
    print("  DD: <10%->1.2, 10-20%->1.0, >20%->0.5")
    print("=" * 130)

    test3_configs = [
        # (base_size, max_corr, label)
        (0.50, 0.5, "T3: WR+DD base=50% corr<0.5"),
        (0.50, 1.0, "T3: WR+DD base=50% no corr filter"),
        (0.45, 0.5, "T3: WR+DD base=45% corr<0.5"),
        (0.55, 0.5, "T3: WR+DD base=55% corr<0.5"),
        (0.50, 0.6, "T3: WR+DD base=50% corr<0.6"),
    ]

    test3_results = []
    for bs, mc, label in test3_configs:
        r = backtest_v146(mode='cross_corr', sizing='wr_dd',
                          wr_base=bs, max_corr=mc)
        r['desc'] = label
        test3_results.append(r)
        pr(r, label)

    # ===================== TEST 4: Cross+Corr with Signal Strength Scaling =====================
    print("\n" + "=" * 130)
    print("  TEST 4: Cross+Corr with Signal Strength Scaling")
    print("  Dual signal (same commodity) -> boosted size")
    print("  Different low corr -> 55% each")
    print("  Different high corr -> skip weaker")
    print("=" * 130)

    test4_configs = [
        # (base_size, max_corr, label)
        (0.50, 0.5, "T4: Strength base=50% corr<0.5"),
        (0.55, 0.5, "T4: Strength base=55% corr<0.5"),
        (0.45, 0.5, "T4: Strength base=45% corr<0.5"),
        (0.50, 0.6, "T4: Strength base=50% corr<0.6"),
        (0.50, 0.4, "T4: Strength base=50% corr<0.4"),
    ]

    test4_results = []
    for bs, mc, label in test4_configs:
        r = backtest_v146(mode='strength', sizing='fixed',
                          base_size=bs, max_corr=mc)
        r['desc'] = label
        test4_results.append(r)
        pr(r, label)

    # ===================== TEST 5: The "Kitchen Sink" =====================
    print("\n" + "=" * 130)
    print("  TEST 5: THE KITCHEN SINK")
    print("  Cross+Corr + DD sizing * WR-adaptive * Regime-combo multiplier + SL=5%")
    print("=" * 130)

    test5_configs = [
        # (dd_tiers, max_corr, sl, label)
        ([(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)], 0.5, 0.05,
         "T5: Sink DD70/60/40/20 WR corr<0.5 SL=5%"),
        ([(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)], 0.6, 0.05,
         "T5: Sink DD70/60/40/20 WR corr<0.6 SL=5%"),
        ([(0, 0.65), (0.10, 0.55), (0.20, 0.40), (0.30, 0.25)], 0.5, 0.05,
         "T5: Sink DD65/55/40/25 WR corr<0.5 SL=5%"),
        ([(0, 0.75), (0.10, 0.65), (0.20, 0.50), (0.30, 0.35)], 0.5, 0.05,
         "T5: Sink DD75/65/50/35 WR corr<0.5 SL=5%"),
        ([(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)], 0.5, 0.03,
         "T5: Sink DD70/60/40/20 WR corr<0.5 SL=3%"),
        ([(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)], 0.5, 0.08,
         "T5: Sink DD70/60/40/20 WR corr<0.5 SL=8%"),
        ([(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)], 0.5, 0.0,
         "T5: Sink DD70/60/40/20 WR corr<0.5 NO SL"),
        ([(0, 0.65), (0.10, 0.55), (0.20, 0.35), (0.30, 0.15)], 0.5, 0.05,
         "T5: Sink DD65/55/35/15 WR corr<0.5 SL=5% (aggressive DD cut)"),
        ([(0, 0.80), (0.10, 0.65), (0.20, 0.45), (0.30, 0.25)], 0.5, 0.05,
         "T5: Sink DD80/65/45/25 WR corr<0.5 SL=5% (wide range)"),
    ]

    test5_results = []
    for dd_t, mc, sl, label in test5_configs:
        r = backtest_v146(mode='cross_corr', sizing='kitchen_sink',
                          dd_tiers=dd_t, max_corr=mc, sl_pct=sl)
        r['desc'] = label
        test5_results.append(r)
        pr(r, label)

    # ===================== SECTION: COMPREHENSIVE RANKING =====================
    print("\n" + "=" * 130)
    print("  COMPREHENSIVE RANKING")
    print("=" * 130)

    all_results = test1_results + test2_results + test3_results + test4_results + test5_results
    all_valid = [r for r in all_results if r.get('desc', '') and r['mdd'] > -80]

    # Sort by annual return
    all_valid.sort(key=lambda x: -x['ann'])
    print(f"\n  Top 15 by Annual Return:")
    for i, r in enumerate(all_valid[:15]):
        desc = r.get('desc', '')
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i+1:2d}: {desc:75s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    # Sort by return/MDD ratio
    all_with_ratio = [(r, abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0) for r in all_valid]
    all_with_ratio.sort(key=lambda x: -x[1])
    print(f"\n  Top 15 by Ann/MDD Ratio:")
    for i, (r, ratio) in enumerate(all_with_ratio[:15]):
        desc = r.get('desc', '')
        print(f"  #{i+1:2d}: {desc:75s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    # Sort by Sharpe
    all_valid_sh = list(all_valid)
    all_valid_sh.sort(key=lambda x: -x['sharpe'])
    print(f"\n  Top 10 by Sharpe:")
    for i, r in enumerate(all_valid_sh[:10]):
        desc = r.get('desc', '')
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i+1:2d}: {desc:75s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    # ===================== SECTION: WALK-FORWARD FOR TOP CONFIGS =====================
    print("\n" + "=" * 130)
    print("  WALK-FORWARD VALIDATION FOR TOP CONFIGS")
    print("=" * 130)

    # Select top configs for WF: best from each test, plus top overall
    # Get unique top 10 by ratio
    seen = set()
    wf_configs = []
    for r, ratio in all_with_ratio:
        desc = r.get('desc', '')
        if desc not in seen:
            seen.add(desc)
            wf_configs.append(r)
        if len(wf_configs) >= 10:
            break

    wf_all = {}
    for r in wf_configs:
        desc = r.get('desc', '')
        # Determine params from desc
        if desc.startswith('T1:'):
            mode = 'cross_corr'; sizing = 'regime_combo'
            rh, rm, sh, sm, sl = 0.70, 0.40, 0.70, 0.55, 0.30
            mc = 0.5; sl_pct = 0.0
            if '0.65/0.35' in desc: rh, rm = 0.65, 0.35
            elif '0.75/0.45' in desc: rh, rm = 0.75, 0.45
            if '-> 80' in desc: sh = 0.80
            elif '-> 65' in desc: sh = 0.65
            elif '-> 75' in desc: sh = 0.75
            if '35%' in desc and 'corr<0.5' in desc: sl_val = 0.35
            elif '25%' in desc: sl_val = 0.25
            else: sl_val = sl
            if 'corr<0.6' in desc: mc = 0.6
            elif 'corr<0.7' in desc: mc = 0.7
            wf_res = walk_forward(mode=mode, sizing=sizing, max_corr=mc,
                                  regime_high=rh, regime_mid=rm,
                                  size_regime_high=sh, size_regime_mid=sm,
                                  size_regime_low=sl_val, label=desc)
        elif desc.startswith('T2:'):
            mode = 'cross_corr'; sizing = 'dd_tiers'
            mc = 0.5; sl_pct = 0.05
            dd_t = [(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)]
            if '65/55/40/25' in desc: dd_t = [(0, 0.65), (0.10, 0.55), (0.20, 0.40), (0.30, 0.25)]
            elif '75/65/50/35' in desc: dd_t = [(0, 0.75), (0.10, 0.65), (0.20, 0.50), (0.30, 0.35)]
            elif '75/65/50/35/20' in desc: dd_t = [(0, 0.75), (0.10, 0.65), (0.20, 0.50), (0.30, 0.35), (0.40, 0.20)]
            if 'corr<0.6' in desc: mc = 0.6
            if 'SL=3%' in desc: sl_pct = 0.03
            elif 'SL=8%' in desc: sl_pct = 0.08
            elif 'NO SL' in desc: sl_pct = 0.0
            wf_res = walk_forward(mode=mode, sizing=sizing, max_corr=mc,
                                  dd_tiers=dd_t, sl_pct=sl_pct, label=desc)
        elif desc.startswith('T3:'):
            mode = 'cross_corr'; sizing = 'wr_dd'
            bs = 0.50; mc = 0.5
            if 'base=45' in desc: bs = 0.45
            elif 'base=55' in desc: bs = 0.55
            if 'no corr' in desc: mc = 1.0
            elif 'corr<0.6' in desc: mc = 0.6
            wf_res = walk_forward(mode=mode, sizing=sizing, max_corr=mc,
                                  wr_base=bs, label=desc)
        elif desc.startswith('T4:'):
            mode = 'strength'; sizing = 'fixed'
            bs = 0.50; mc = 0.5
            if 'base=55' in desc: bs = 0.55
            elif 'base=45' in desc: bs = 0.45
            if 'corr<0.6' in desc: mc = 0.6
            elif 'corr<0.4' in desc: mc = 0.4
            wf_res = walk_forward(mode=mode, sizing=sizing, max_corr=mc,
                                  base_size=bs, label=desc)
        elif desc.startswith('T5:'):
            mode = 'cross_corr'; sizing = 'kitchen_sink'
            mc = 0.5; sl_pct = 0.05
            dd_t = [(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)]
            if '65/55/40/25' in desc: dd_t = [(0, 0.65), (0.10, 0.55), (0.20, 0.40), (0.30, 0.25)]
            elif '75/65/50/35' in desc: dd_t = [(0, 0.75), (0.10, 0.65), (0.20, 0.50), (0.30, 0.35)]
            elif '65/55/35/15' in desc: dd_t = [(0, 0.65), (0.10, 0.55), (0.20, 0.35), (0.30, 0.15)]
            elif '80/65/45/25' in desc: dd_t = [(0, 0.80), (0.10, 0.65), (0.20, 0.45), (0.30, 0.25)]
            if 'corr<0.6' in desc: mc = 0.6
            if 'SL=3%' in desc: sl_pct = 0.03
            elif 'SL=8%' in desc: sl_pct = 0.08
            elif 'NO SL' in desc: sl_pct = 0.0
            wf_res = walk_forward(mode=mode, sizing=sizing, max_corr=mc,
                                  dd_tiers=dd_t, sl_pct=sl_pct, label=desc)
        else:
            continue

        wf_all[desc] = wf_res
        print_wf(wf_res, desc)

    # ===================== SECTION: HIGHLIGHT BEST CONFIGS =====================
    print("\n" + "=" * 130)
    print("  HIGHLIGHT: Configs achieving +180%+ annual with worst WF MDD < -28%")
    print("=" * 130)

    highlight_configs = []
    for desc, wf_res in wf_all.items():
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        pos_years = sum(1 for r in wf_res.values() if r['ann'] > 0)
        if avg_ann >= 180 and worst_mdd > -28:
            highlight_configs.append((desc, avg_ann, worst_mdd, pos_years, wf_res))

    if highlight_configs:
        highlight_configs.sort(key=lambda x: -x[1])
        for desc, avg_ann, worst_mdd, pos_years, wf_res in highlight_configs:
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(wf_res.items())])
            print(f"\n  *** {desc}")
            print(f"      AvgWF={avg_ann:+.0f}% | WorstWfMDD={worst_mdd:.1f}% | {pos_years}/6 positive")
            print(f"      {ws}")
    else:
        print("\n  No configs meet +180% avg WF annual with worst WF MDD < -28%")

        # Show what comes closest
        closest = []
        for desc, wf_res in wf_all.items():
            avg_ann = np.mean([r['ann'] for r in wf_res.values()])
            worst_mdd = min(r['mdd'] for r in wf_res.values())
            closest.append((desc, avg_ann, worst_mdd))
        closest.sort(key=lambda x: -x[1])
        print(f"\n  Closest configs by avg WF return:")
        for desc, avg_ann, worst_mdd in closest[:10]:
            print(f"  {desc:75s} | AvgWF={avg_ann:>+7.0f}% | WorstWfMDD={worst_mdd:>5.1f}%")

    # ===================== SECTION: ALSO SHOW RELAXED TARGETS =====================
    print("\n" + "=" * 130)
    print("  HIGHLIGHT: Configs achieving +150%+ annual with worst WF MDD < -30%")
    print("=" * 130)

    highlight2 = []
    for desc, wf_res in wf_all.items():
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        pos_years = sum(1 for r in wf_res.values() if r['ann'] > 0)
        if avg_ann >= 150 and worst_mdd > -30:
            highlight2.append((desc, avg_ann, worst_mdd, pos_years, wf_res))

    if highlight2:
        highlight2.sort(key=lambda x: -x[1])
        for desc, avg_ann, worst_mdd, pos_years, wf_res in highlight2[:10]:
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(wf_res.items())])
            print(f"\n  *** {desc}")
            print(f"      AvgWF={avg_ann:+.0f}% | WorstWfMDD={worst_mdd:.1f}% | {pos_years}/6 positive")
            print(f"      {ws}")
    else:
        print("\n  No configs meet this relaxed target either.")

    # ===================== SECTION: DETAILED WF TABLE =====================
    print("\n" + "=" * 130)
    print("  DETAILED WF TABLE: ALL TESTED CONFIGS")
    print("=" * 130)

    print(f"\n  {'Config':75s} | {'2020':>12s} | {'2021':>12s} | {'2022':>12s} | {'2023':>12s} | {'2024':>12s} | {'2025':>12s} | {'Avg':>7s} | {'WfMDD':>6s}")
    print(f"  {'-'*75}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*7}-+-{'-'*6}")

    for desc, wf_res in wf_all.items():
        vals = []
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            if yr in wf_res:
                vals.append(f"{wf_res[yr]['ann']:+.0f}/{wf_res[yr]['mdd']:.0f}")
            else:
                vals.append("N/A")
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        print(f"  {desc:75s} | {vals[0]:>12s} | {vals[1]:>12s} | {vals[2]:>12s} | {vals[3]:>12s} | {vals[4]:>12s} | {vals[5]:>12s} | {avg_ann:>+6.0f}% | {worst_mdd:>5.1f}%")

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 130)
    print("  FINAL SUMMARY")
    print("=" * 130)

    print(f"\n  Best configs by test category:")
    print(f"\n  --- Test 1 (Cross+Corr + Regime Combo):")
    test1_sorted = sorted(test1_results, key=lambda x: abs(x['ann']/x['mdd']) if x['mdd'] != 0 else 0, reverse=True)
    for r in test1_sorted[:3]:
        pr(r, r.get('desc', ''))

    print(f"\n  --- Test 2 (Cross+Corr + DD Sizing + SL):")
    test2_sorted = sorted(test2_results, key=lambda x: abs(x['ann']/x['mdd']) if x['mdd'] != 0 else 0, reverse=True)
    for r in test2_sorted[:3]:
        pr(r, r.get('desc', ''))

    print(f"\n  --- Test 3 (WR-Adaptive + DD-Adaptive):")
    test3_sorted = sorted(test3_results, key=lambda x: abs(x['ann']/x['mdd']) if x['mdd'] != 0 else 0, reverse=True)
    for r in test3_sorted[:3]:
        pr(r, r.get('desc', ''))

    print(f"\n  --- Test 4 (Signal Strength Scaling):")
    test4_sorted = sorted(test4_results, key=lambda x: abs(x['ann']/x['mdd']) if x['mdd'] != 0 else 0, reverse=True)
    for r in test4_sorted[:3]:
        pr(r, r.get('desc', ''))

    print(f"\n  --- Test 5 (Kitchen Sink):")
    test5_sorted = sorted(test5_results, key=lambda x: abs(x['ann']/x['mdd']) if x['mdd'] != 0 else 0, reverse=True)
    for r in test5_sorted[:3]:
        pr(r, r.get('desc', ''))

    print(f"\n  OVERALL TOP 5 (by full-period R/M ratio):")
    for i, (r, ratio) in enumerate(all_with_ratio[:5]):
        desc = r.get('desc', '')
        print(f"  #{i+1}: {desc:75s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
