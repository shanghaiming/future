"""
Alpha Futures V151 — OI-Enhanced Signal Selection
==============================================================================
Goal: Test OI-based filters and signal modifiers on top of V121/Union signals.

Sections:
  0: Baseline (V146 champion, no OI)
  1: OI Surge Filter (various thresholds)
  2: Price-OI Divergence as filter
  3: OI Momentum as score modifier
  4: Volume/OI Ratio filter
  5: Best OI combination
  6: WF validation for top configs
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
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    return (final / initial) ** (1.0 / (n_days / 252)) * 100 - 100


def main():
    print("=" * 130)
    print("  V151 — OI-Enhanced Signal Selection")
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
            if not np.isnan(c[di]) and not np.isnan(c[di - 1]) and c[di - 1] > 0:
                RET[si, di] = (c[di] / c[di - 1] - 1) * 100
        ROC5[si] = talib.ROC(c, timeperiod=5)
        ROC20[si] = talib.ROC(c, timeperiod=20)

    ATR14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        ATR14[si] = talib.ATR(H[si].astype(np.float64), L[si].astype(np.float64),
                               C[si].astype(np.float64), timeperiod=14)

    ZSCORE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = RET[si, di - 20:di]
            v = rets[~np.isnan(rets)]
            if len(v) < 10:
                continue
            s = np.std(v, ddof=1)
            if s > 0 and not np.isnan(RET[si, di]):
                ZSCORE[si, di] = (RET[si, di] - np.mean(v)) / s

    OV_GAP = np.full((NS, ND), np.nan)
    ID_RET = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            o, c = O[si, di], C[si, di]
            if not np.isnan(o) and not np.isnan(c):
                if di > 0 and not np.isnan(C[si, di - 1]) and C[si, di - 1] > 0:
                    OV_GAP[si, di] = (o - C[si, di - 1]) / C[si, di - 1] * 100
                if o > 0:
                    ID_RET[si, di] = (c - o) / o * 100

    # ===================== OI PRECOMPUTATION =====================
    print("  Computing OI indicators...", flush=True)

    # OI_MA20: 20-day moving average of OI
    OI_MA20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        oi_arr = OI[si].astype(np.float64)
        OI_MA20[si] = talib.MA(oi_arr, timeperiod=20)

    # OI_MOM5: 5-day OI change rate (%)
    OI_MOM5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        OI_MOM5[si] = talib.ROC(OI[si].astype(np.float64), timeperiod=5)

    # VOL_OI_RATIO: volume / OI (where OI > 0)
    VOL_OI_RATIO = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            v = V[si, di]
            oi = OI[si, di]
            if not np.isnan(v) and not np.isnan(oi) and oi > 0:
                VOL_OI_RATIO[si, di] = v / oi

    # OI_SURGE: OI / OI_MA20 ratio (how much above average)
    OI_SURGE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            oi = OI[si, di]
            oi_ma = OI_MA20[si, di]
            if not np.isnan(oi) and not np.isnan(oi_ma) and oi_ma > 0:
                OI_SURGE[si, di] = oi / oi_ma

    # Print OI stats
    surge_vals = OI_SURGE[~np.isnan(OI_SURGE)]
    mom_vals = OI_MOM5[~np.isnan(OI_MOM5)]
    voi_vals = VOL_OI_RATIO[~np.isnan(VOL_OI_RATIO)]
    if len(surge_vals) > 0:
        print(f"  OI_SURGE  mean={np.mean(surge_vals):.2f} median={np.median(surge_vals):.2f} "
              f"p75={np.percentile(surge_vals, 75):.2f} p90={np.percentile(surge_vals, 90):.2f}")
    if len(mom_vals) > 0:
        print(f"  OI_MOM5   mean={np.mean(mom_vals):.2f} median={np.median(mom_vals):.2f}")
    if len(voi_vals) > 0:
        print(f"  VOL/OI    mean={np.mean(voi_vals):.2f} median={np.median(voi_vals):.2f} "
              f"p75={np.percentile(voi_vals, 75):.2f}")

    print(f"  Done ({time.time() - t0:.1f}s)")

    # ===================== SIGNAL DEFINITIONS =====================
    def sig_v121(di, edi):
        c = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5:
                continue
            rp = ROC5[s, di - 1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp:
                continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0:
                continue
            c.append((roc * zs, s, ep, 'v121'))
        return c

    def sig_ov_id(di, edi):
        c = []
        for s in range(NS):
            ov = OV_GAP[s, di]; idr = ID_RET[s, di]; roc = ROC5[s, di]
            if any(np.isnan(x) for x in [ov, idr, roc]):
                continue
            if ov <= 0.3 or idr <= 0.3 or roc <= 1.0:
                continue
            zs = ZSCORE[s, di]
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0:
                continue
            z_bonus = zs if not np.isnan(zs) and zs > 1.0 else 1.0
            c.append(((ov + idr) * roc * z_bonus * 2, s, ep, 'ov_id'))
        return c

    def sig_final_flag(di, edi):
        c = []
        for s in range(NS):
            roc20 = ROC20[s, di]
            if np.isnan(roc20) or roc20 <= 5.0 or di < 6:
                continue
            h5 = H[s, di - 4:di + 1]; l5 = L[s, di - 4:di + 1]
            if any(np.isnan(x) for x in h5) or any(np.isnan(x) for x in l5):
                continue
            r5 = np.max(h5) - np.min(l5)
            atr = ATR14[s, di]
            if np.isnan(atr) or atr <= 0 or r5 > atr * 3.0:
                continue
            h4 = np.max(H[s, di - 4:di])
            cp = C[s, di]
            if np.isnan(cp) or cp <= h4:
                continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0:
                continue
            c.append((roc20 * (cp - h4) / atr, s, ep, 'ff'))
        return c

    def sig_union(di, edi):
        all_sigs = {}
        for item in sig_v121(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs:
                all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc * 3
            all_sigs[s][2].append('v121')
        for item in sig_ov_id(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs:
                all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc * 2
            all_sigs[s][2].append('ov_id')
        for item in sig_final_flag(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs:
                all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc
            all_sigs[s][2].append('ff')
        return [(sc, s, ep, '+'.join(sigs)) for s, (sc, ep, sigs) in all_sigs.items()]

    # ===================== HELPER: Correlation =====================
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

    # ===================== HELPER: DD-based sizing =====================
    def dd_size(pv, high_water, tiers):
        if high_water <= 0:
            return tiers[0][1]
        dd = (pv - high_water) / high_water
        for dd_thresh, size_frac in tiers:
            if dd >= -dd_thresh:
                return size_frac
        return tiers[-1][1]

    # ===================== OI FILTER / MODIFIER FUNCTIONS =====================

    def oi_surge_filter(candidates, di, threshold=1.5):
        """Filter: only keep candidates where OI > threshold * OI_MA20."""
        filtered = []
        for sc, s, ep, sig in candidates:
            surge = OI_SURGE[s, di]
            if not np.isnan(surge) and surge > threshold:
                filtered.append((sc, s, ep, sig))
        return filtered

    def oi_divergence_filter(candidates, di, mode='bullish'):
        """
        Price-OI divergence filter.
        mode='bullish': Price up + OI up = new longs (bullish confirmation). Keep only these.
        mode='bearish_reject': Reject if price up + OI down (short covering, potential reversal).
        """
        filtered = []
        for sc, s, ep, sig in candidates:
            ret_d = RET[s, di]
            oi_mom = OI_MOM5[s, di]
            if np.isnan(ret_d) or np.isnan(oi_mom):
                # No OI data: pass through (don't filter out)
                filtered.append((sc, s, ep, sig))
                continue
            if mode == 'bullish':
                # Price up AND OI up = bullish confirmation (new longs entering)
                if ret_d > 0 and oi_mom > 0:
                    filtered.append((sc, s, ep, sig))
                elif ret_d <= 0:
                    # Price down signal: keep (we're doing long-only, so unlikely)
                    filtered.append((sc, s, ep, sig))
                # else: price up + OI down = short covering, skip
            elif mode == 'bearish_reject':
                # Reject only when price up + OI down strongly
                if ret_d > 0 and oi_mom < -5:
                    continue
                filtered.append((sc, s, ep, sig))
        return filtered

    def oi_momentum_boost(candidates, di, weight=0.5):
        """Multiply signal score by (1 + weight * OI_MOM5/100). Positive OI momentum boosts."""
        boosted = []
        for sc, s, ep, sig in candidates:
            oi_mom = OI_MOM5[s, di]
            if not np.isnan(oi_mom):
                factor = 1.0 + weight * oi_mom / 100.0
                factor = max(0.2, min(3.0, factor))
                boosted.append((sc * factor, s, ep, sig))
            else:
                boosted.append((sc, s, ep, sig))
        return boosted

    def voi_filter(candidates, di, max_ratio=None, max_percentile=75):
        """Filter: reject candidates with Volume/OI ratio above threshold (exhaustion)."""
        if max_ratio is not None:
            threshold = max_ratio
        else:
            # Use a fixed default
            threshold = 5.0
        filtered = []
        for sc, s, ep, sig in candidates:
            voi = VOL_OI_RATIO[s, di]
            if np.isnan(voi):
                filtered.append((sc, s, ep, sig))
                continue
            if voi <= threshold:
                filtered.append((sc, s, ep, sig))
        return filtered

    # ===================== UNIFIED BACKTEST ENGINE =====================
    def backtest(start_di=MIN_TRAIN, end_di=None,
                 # OI filters/modifiers
                 oi_mode='none',        # 'none','surge','divergence_bull','divergence_reject',
                                         # 'oi_mom_boost','voi_filter','combo'
                 oi_surge_thresh=1.5,
                 oi_divergence_mode='bullish',
                 oi_mom_weight=0.5,
                 voi_max_ratio=5.0,
                 # Sizing
                 dd_tiers=None,
                 base_size=0.55,
                 # General
                 hold=1, top_n=2, max_corr=0.5):

        if end_di is None:
            end_di = ND
        if dd_tiers is None:
            dd_tiers = [(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)]

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

            # Close positions past hold period
            cl = []
            for p in positions:
                if di - p['entry_di'] >= p['hold_days']:
                    ep = C[p['si'], di]
                    if np.isnan(ep) or ep <= 0:
                        ep = p['entry_price']
                    m = MULT.get(p['sym'], DEF_MULT)
                    pnl = (ep - p['entry_price']) * m * p['lots']
                    inv = p['entry_price'] * m * abs(p['lots'])
                    pp = pnl / inv * 100 if inv > 0 else 0
                    cash += ep * m * abs(p['lots']) * (1 - COMM)
                    trades.append(pp)
                    cl.append(p)
            for p in cl:
                positions.remove(p)

            # Position sizing via DD tiers
            pos_size = dd_size(pv, high_water, dd_tiers)
            pos_size = max(0.05, min(0.95, pos_size))

            # --- Enter positions ---
            if len(positions) >= top_n:
                continue
            edi = di + 1
            if edi >= end_di:
                continue

            held_si = set(p['si'] for p in positions)

            # Get V121 and Union candidates
            cands_v121 = sig_v121(di, edi)
            cands_union = sig_union(di, edi)

            # Apply OI filters/modifiers to candidates
            if oi_mode == 'surge':
                cands_v121 = oi_surge_filter(cands_v121, di, oi_surge_thresh)
                cands_union = oi_surge_filter(cands_union, di, oi_surge_thresh)

            elif oi_mode == 'divergence_bull':
                cands_v121 = oi_divergence_filter(cands_v121, di, mode='bullish')
                cands_union = oi_divergence_filter(cands_union, di, mode='bullish')

            elif oi_mode == 'divergence_reject':
                cands_v121 = oi_divergence_filter(cands_v121, di, mode='bearish_reject')
                cands_union = oi_divergence_filter(cands_union, di, mode='bearish_reject')

            elif oi_mode == 'oi_mom_boost':
                cands_v121 = oi_momentum_boost(cands_v121, di, oi_mom_weight)
                cands_union = oi_momentum_boost(cands_union, di, oi_mom_weight)

            elif oi_mode == 'voi_filter':
                cands_v121 = voi_filter(cands_v121, di, voi_max_ratio)
                cands_union = voi_filter(cands_union, di, voi_max_ratio)

            elif oi_mode == 'combo':
                # Apply all OI filters + boost
                cands_v121 = oi_surge_filter(cands_v121, di, oi_surge_thresh)
                cands_union = oi_surge_filter(cands_union, di, oi_surge_thresh)
                cands_v121 = oi_divergence_filter(cands_v121, di, mode='bullish')
                cands_union = oi_divergence_filter(cands_union, di, mode='bullish')
                cands_v121 = voi_filter(cands_v121, di, voi_max_ratio)
                cands_union = voi_filter(cands_union, di, voi_max_ratio)
                cands_v121 = oi_momentum_boost(cands_v121, di, oi_mom_weight)
                cands_union = oi_momentum_boost(cands_union, di, oi_mom_weight)

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

            # CRITICAL: snapshot cash before entry loop, split equally
            cash_snapshot = cash
            n_planned = len(entries)
            for sc, s, pr, sig_str, pct in entries:
                if s in set(p['si'] for p in positions):
                    continue
                if len(positions) >= top_n:
                    break
                cap = cash_snapshot * pct / n_planned
                sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                ct = max(1, int(cap / (pr * m * (1 + COMM))))
                ci = pr * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                    ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                if ct <= 0 or ci <= 0 or ci > cash:
                    continue
                cash -= ci
                positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                  'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': hold,
                                  'sig': sig_str, 'score': sc})

        # Close remaining
        for p in positions:
            ae = end_di - 1
            ep = C[p['si'], min(ae, ND - 1)]
            if np.isnan(ep) or ep <= 0:
                ep = p['entry_price']
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
        print(f"  {label:75s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | "
              f"Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d}")

    def walk_forward(oi_mode='none', oi_surge_thresh=1.5, oi_divergence_mode='bullish',
                     oi_mom_weight=0.5, voi_max_ratio=5.0,
                     dd_tiers=None, base_size=0.55, hold=1, top_n=2, max_corr=0.5,
                     label=""):
        if dd_tiers is None:
            dd_tiers = [(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)]
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None:
                    ys = di
                if dates[di].year == yr:
                    ye = di + 1
            if ys is None:
                continue
            r = backtest(start_di=ys, end_di=ye, oi_mode=oi_mode,
                         oi_surge_thresh=oi_surge_thresh,
                         oi_divergence_mode=oi_divergence_mode,
                         oi_mom_weight=oi_mom_weight,
                         voi_max_ratio=voi_max_ratio,
                         dd_tiers=dd_tiers, base_size=base_size,
                         hold=hold, top_n=top_n, max_corr=max_corr)
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

    # ===================== SECTION 0: BASELINE (NO OI) =====================
    print("\n" + "=" * 130)
    print("  SECTION 0: BASELINE — V146 champion, no OI")
    print("=" * 130)

    dd_tiers_default = [(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)]

    r_base = backtest(oi_mode='none', dd_tiers=dd_tiers_default)
    r_base['desc'] = 'S0: Baseline DD70/60/40/20 no OI'
    pr(r_base, r_base['desc'])

    # Also test with different DD tiers
    dd_aggressive = [(0, 0.70), (0.10, 0.55), (0.20, 0.35), (0.30, 0.15)]
    r_base2 = backtest(oi_mode='none', dd_tiers=dd_aggressive)
    r_base2['desc'] = 'S0: Baseline DD70/55/35/15 no OI'
    pr(r_base2, r_base2['desc'])

    dd_conservative = [(0, 0.60), (0.10, 0.50), (0.20, 0.30), (0.30, 0.15)]
    r_base3 = backtest(oi_mode='none', dd_tiers=dd_conservative)
    r_base3['desc'] = 'S0: Baseline DD60/50/30/15 no OI'
    pr(r_base3, r_base3['desc'])

    # ===================== SECTION 1: OI SURGE FILTER =====================
    print("\n" + "=" * 130)
    print("  SECTION 1: OI SURGE FILTER")
    print("  Only take signals when OI > threshold * 20-day MA(OI)")
    print("=" * 130)

    s1_results = []
    for thresh in [1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]:
        for dd_t, dd_name in [(dd_tiers_default, 'DD70/60/40/20'),
                               (dd_aggressive, 'DD70/55/35/15')]:
            r = backtest(oi_mode='surge', oi_surge_thresh=thresh, dd_tiers=dd_t)
            label = f"S1: Surge>{thresh:.1f}x {dd_name}"
            r['desc'] = label
            s1_results.append(r)
            pr(r, label)

    # ===================== SECTION 2: PRICE-OI DIVERGENCE FILTER =====================
    print("\n" + "=" * 130)
    print("  SECTION 2: PRICE-OI DIVERGENCE FILTER")
    print("  Bullish: keep only price-up + OI-up (new longs)")
    print("  Reject: skip when price-up + OI-down strongly (short covering)")
    print("=" * 130)

    s2_results = []
    for div_mode in ['bullish', 'bearish_reject']:
        for dd_t, dd_name in [(dd_tiers_default, 'DD70/60/40/20'),
                               (dd_aggressive, 'DD70/55/35/15')]:
            r = backtest(oi_mode=f'divergence_{div_mode}', oi_divergence_mode=div_mode,
                         dd_tiers=dd_t)
            label = f"S2: Div-{div_mode} {dd_name}"
            r['desc'] = label
            s2_results.append(r)
            pr(r, label)

    # ===================== SECTION 3: OI MOMENTUM AS SCORE MODIFIER =====================
    print("\n" + "=" * 130)
    print("  SECTION 3: OI MOMENTUM AS SCORE MODIFIER")
    print("  score *= (1 + weight * OI_MOM5/100)")
    print("=" * 130)

    s3_results = []
    for weight in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
        for dd_t, dd_name in [(dd_tiers_default, 'DD70/60/40/20'),
                               (dd_aggressive, 'DD70/55/35/15')]:
            r = backtest(oi_mode='oi_mom_boost', oi_mom_weight=weight, dd_tiers=dd_t)
            label = f"S3: OI-Mom w={weight:.1f} {dd_name}"
            r['desc'] = label
            s3_results.append(r)
            pr(r, label)

    # ===================== SECTION 4: VOLUME/OI RATIO FILTER =====================
    print("\n" + "=" * 130)
    print("  SECTION 4: VOLUME/OI RATIO FILTER")
    print("  Reject when VOL/OI > threshold (exhaustion / turnover too high)")
    print("=" * 130)

    s4_results = []
    for max_ratio in [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        for dd_t, dd_name in [(dd_tiers_default, 'DD70/60/40/20'),
                               (dd_aggressive, 'DD70/55/35/15')]:
            r = backtest(oi_mode='voi_filter', voi_max_ratio=max_ratio, dd_tiers=dd_t)
            label = f"S4: VOI<{max_ratio:.0f} {dd_name}"
            r['desc'] = label
            s4_results.append(r)
            pr(r, label)

    # ===================== SECTION 5: BEST OI COMBINATIONS =====================
    print("\n" + "=" * 130)
    print("  SECTION 5: BEST OI COMBINATIONS")
    print("  Combine: OI surge + divergence + VOI + momentum boost")
    print("=" * 130)

    s5_results = []
    # Vary key combo parameters
    combo_configs = [
        # (surge_thresh, mom_weight, voi_max, dd_tiers, label)
        (1.3, 0.5, 5.0, dd_tiers_default, "S5: Combo surge>1.3 mom=0.5 voi<5 DD70/60/40/20"),
        (1.3, 0.5, 5.0, dd_aggressive, "S5: Combo surge>1.3 mom=0.5 voi<5 DD70/55/35/15"),
        (1.3, 0.8, 5.0, dd_tiers_default, "S5: Combo surge>1.3 mom=0.8 voi<5 DD70/60/40/20"),
        (1.3, 0.8, 5.0, dd_aggressive, "S5: Combo surge>1.3 mom=0.8 voi<5 DD70/55/35/15"),
        (1.5, 0.5, 5.0, dd_tiers_default, "S5: Combo surge>1.5 mom=0.5 voi<5 DD70/60/40/20"),
        (1.5, 0.5, 5.0, dd_aggressive, "S5: Combo surge>1.5 mom=0.5 voi<5 DD70/55/35/15"),
        (1.5, 0.8, 5.0, dd_tiers_default, "S5: Combo surge>1.5 mom=0.8 voi<5 DD70/60/40/20"),
        (1.5, 0.8, 5.0, dd_aggressive, "S5: Combo surge>1.5 mom=0.8 voi<5 DD70/55/35/15"),
        (1.3, 1.0, 5.0, dd_tiers_default, "S5: Combo surge>1.3 mom=1.0 voi<5 DD70/60/40/20"),
        (1.3, 1.0, 5.0, dd_aggressive, "S5: Combo surge>1.3 mom=1.0 voi<5 DD70/55/35/15"),
        (1.3, 0.5, 3.0, dd_tiers_default, "S5: Combo surge>1.3 mom=0.5 voi<3 DD70/60/40/20"),
        (1.3, 0.5, 3.0, dd_aggressive, "S5: Combo surge>1.3 mom=0.5 voi<3 DD70/55/35/15"),
        (1.5, 0.5, 3.0, dd_tiers_default, "S5: Combo surge>1.5 mom=0.5 voi<3 DD70/60/40/20"),
        (1.5, 0.5, 3.0, dd_aggressive, "S5: Combo surge>1.5 mom=0.5 voi<3 DD70/55/35/15"),
        (1.4, 0.8, 5.0, dd_tiers_default, "S5: Combo surge>1.4 mom=0.8 voi<5 DD70/60/40/20"),
        (1.4, 0.8, 5.0, dd_aggressive, "S5: Combo surge>1.4 mom=0.8 voi<5 DD70/55/35/15"),
        (1.2, 0.5, 5.0, dd_tiers_default, "S5: Combo surge>1.2 mom=0.5 voi<5 DD70/60/40/20"),
        (1.2, 0.5, 5.0, dd_aggressive, "S5: Combo surge>1.2 mom=0.5 voi<5 DD70/55/35/15"),
    ]

    for st, mw, vm, dd_t, label in combo_configs:
        r = backtest(oi_mode='combo', oi_surge_thresh=st, oi_mom_weight=mw,
                     voi_max_ratio=vm, dd_tiers=dd_t)
        r['desc'] = label
        s5_results.append(r)
        pr(r, label)

    # ===================== COMPREHENSIVE RANKING =====================
    print("\n" + "=" * 130)
    print("  COMPREHENSIVE RANKING")
    print("=" * 130)

    all_results = ([r_base, r_base2, r_base3]
                   + s1_results + s2_results + s3_results + s4_results + s5_results)
    all_valid = [r for r in all_results if r.get('desc', '') and r['mdd'] > -80]

    # Sort by annual return
    all_valid.sort(key=lambda x: -x['ann'])
    print(f"\n  Top 20 by Annual Return:")
    for i, r in enumerate(all_valid[:20]):
        desc = r.get('desc', '')
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i + 1:2d}: {desc:75s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | "
              f"Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    # Sort by R/M ratio
    all_with_ratio = [(r, abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0) for r in all_valid]
    all_with_ratio.sort(key=lambda x: -x[1])
    print(f"\n  Top 20 by Ann/MDD Ratio:")
    for i, (r, ratio) in enumerate(all_with_ratio[:20]):
        desc = r.get('desc', '')
        print(f"  #{i + 1:2d}: {desc:75s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | "
              f"Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    # ===================== SECTION 6: WALK-FORWARD FOR TOP CONFIGS =====================
    print("\n" + "=" * 130)
    print("  SECTION 6: WALK-FORWARD VALIDATION FOR TOP 10 CONFIGS")
    print("=" * 130)

    # Select top 10 unique configs by R/M ratio for WF
    seen = set()
    wf_configs = []
    for r, ratio in all_with_ratio:
        desc = r.get('desc', '')
        if desc not in seen:
            seen.add(desc)
            wf_configs.append(r)
        if len(wf_configs) >= 10:
            break

    # Mapping from desc to params
    def get_params_from_desc(desc):
        """Parse description to extract backtest parameters."""
        # Default DD tiers
        dd_t = dd_tiers_default
        if 'DD70/55/35/15' in desc:
            dd_t = dd_aggressive
        elif 'DD60/50/30/15' in desc:
            dd_t = dd_conservative

        # OI mode
        if desc.startswith('S0:'):
            return {'oi_mode': 'none', 'dd_tiers': dd_t}
        elif desc.startswith('S1:'):
            thresh = 1.5
            for t in [2.0, 1.8, 1.6, 1.5, 1.4, 1.3, 1.2]:
                if f'>{t:.1f}' in desc:
                    thresh = t
                    break
            return {'oi_mode': 'surge', 'oi_surge_thresh': thresh, 'dd_tiers': dd_t}
        elif desc.startswith('S2:'):
            if 'bullish' in desc:
                return {'oi_mode': 'divergence_bull', 'oi_divergence_mode': 'bullish', 'dd_tiers': dd_t}
            else:
                return {'oi_mode': 'divergence_reject', 'oi_divergence_mode': 'bearish_reject', 'dd_tiers': dd_t}
        elif desc.startswith('S3:'):
            weight = 0.5
            for w in [2.0, 1.5, 1.0, 0.8, 0.5, 0.3]:
                if f'w={w:.1f}' in desc:
                    weight = w
                    break
            return {'oi_mode': 'oi_mom_boost', 'oi_mom_weight': weight, 'dd_tiers': dd_t}
        elif desc.startswith('S4:'):
            max_r = 5.0
            for mr in [10, 8, 6, 5, 4, 3]:
                if f'voi<{mr}' in desc:
                    max_r = float(mr)
                    break
            return {'oi_mode': 'voi_filter', 'voi_max_ratio': max_r, 'dd_tiers': dd_t}
        elif desc.startswith('S5:'):
            thresh = 1.3
            for t in [1.5, 1.4, 1.3, 1.2]:
                if f'surge>{t:.1f}' in desc:
                    thresh = t
                    break
            weight = 0.5
            for w in [1.0, 0.8, 0.5]:
                if f'mom={w:.1f}' in desc:
                    weight = w
                    break
            max_r = 5.0
            for mr in [5, 3]:
                if f'voi<{mr}' in desc:
                    max_r = float(mr)
                    break
            return {'oi_mode': 'combo', 'oi_surge_thresh': thresh,
                    'oi_mom_weight': weight, 'voi_max_ratio': max_r, 'dd_tiers': dd_t}
        else:
            return {'oi_mode': 'none', 'dd_tiers': dd_t}

    wf_all = {}
    for r in wf_configs:
        desc = r.get('desc', '')
        params = get_params_from_desc(desc)
        wf_res = walk_forward(label=desc, **params)
        wf_all[desc] = wf_res
        print_wf(wf_res, desc)

    # ===================== FINAL SUMMARY: TOP 3 BY WF AVG, WF MDD < -30% =====================
    print("\n" + "=" * 130)
    print("  FINAL SUMMARY: TOP CONFIGS BY WF AVERAGE RETURN (WF MDD < -30%)")
    print("=" * 130)

    qualified = []
    for desc, wf_res in wf_all.items():
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        pos_years = sum(1 for r in wf_res.values() if r['ann'] > 0)
        if worst_mdd > -30:
            qualified.append((desc, avg_ann, worst_mdd, pos_years, wf_res))

    if qualified:
        qualified.sort(key=lambda x: -x[1])
        print(f"\n  {len(qualified)} configs with WorstWfMDD > -30%:")
        for i, (desc, avg_ann, worst_mdd, pos_years, wf_res) in enumerate(qualified[:10]):
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(wf_res.items())])
            print(f"\n  #{i + 1}: {desc}")
            print(f"       AvgWF={avg_ann:>+7.0f}% | WorstWfMDD={worst_mdd:>5.1f}% | {pos_years}/6 positive")
            print(f"       {ws}")
    else:
        print("\n  No configs with WorstWfMDD > -30%. Showing all by avg WF return:")
        all_wf = []
        for desc, wf_res in wf_all.items():
            avg_ann = np.mean([r['ann'] for r in wf_res.values()])
            worst_mdd = min(r['mdd'] for r in wf_res.values())
            pos_years = sum(1 for r in wf_res.values() if r['ann'] > 0)
            all_wf.append((desc, avg_ann, worst_mdd, pos_years, wf_res))
        all_wf.sort(key=lambda x: -x[1])
        for i, (desc, avg_ann, worst_mdd, pos_years, wf_res) in enumerate(all_wf[:10]):
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(wf_res.items())])
            print(f"\n  #{i + 1}: {desc}")
            print(f"       AvgWF={avg_ann:>+7.0f}% | WorstWfMDD={worst_mdd:>5.1f}% | {pos_years}/6 positive")
            print(f"       {ws}")

    # Also show a relaxed criterion
    print("\n" + "-" * 130)
    print("  RELAXED: TOP 3 by WF avg return with WF MDD < -35%")
    print("-" * 130)

    relaxed = []
    for desc, wf_res in wf_all.items():
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        pos_years = sum(1 for r in wf_res.values() if r['ann'] > 0)
        if worst_mdd > -35:
            relaxed.append((desc, avg_ann, worst_mdd, pos_years, wf_res))

    if relaxed:
        relaxed.sort(key=lambda x: -x[1])
        for i, (desc, avg_ann, worst_mdd, pos_years, wf_res) in enumerate(relaxed[:5]):
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(wf_res.items())])
            print(f"\n  #{i + 1}: {desc}")
            print(f"       AvgWF={avg_ann:>+7.0f}% | WorstWfMDD={worst_mdd:>5.1f}% | {pos_years}/6 positive")
            print(f"       {ws}")

    # ===================== DETAILED WF TABLE =====================
    print("\n" + "=" * 130)
    print("  DETAILED WF TABLE: ALL TESTED CONFIGS")
    print("=" * 130)

    hdr = (f"  {'Config':75s} | {'2020':>12s} | {'2021':>12s} | {'2022':>12s} | "
           f"{'2023':>12s} | {'2024':>12s} | {'2025':>12s} | {'Avg':>7s} | {'WfMDD':>6s}")
    print(f"\n{hdr}")
    print(f"  {'-' * 75}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+-"
          f"{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 7}-+-{'-' * 6}")

    for desc, wf_res in wf_all.items():
        vals = []
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            if yr in wf_res:
                vals.append(f"{wf_res[yr]['ann']:+.0f}/{wf_res[yr]['mdd']:.0f}")
            else:
                vals.append("N/A")
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        print(f"  {desc:75s} | {vals[0]:>12s} | {vals[1]:>12s} | {vals[2]:>12s} | "
              f"{vals[3]:>12s} | {vals[4]:>12s} | {vals[5]:>12s} | {avg_ann:>+6.0f}% | "
              f"{worst_mdd:>5.1f}%")

    # ===================== FINAL ELAPSED =====================
    print(f"\n  Elapsed: {time.time() - t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
