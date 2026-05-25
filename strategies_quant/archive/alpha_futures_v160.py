"""
Alpha Futures V160 — V157 Breakthrough + V158 Signal Optimization
=============================================================================
Combines:
  - V157's top_n=3 + DD100/90/70/50 + Kitchen Sink (+267%/-26% WF)
  - V158's OV/ID threshold optimization (OV>0.1 ID>0.5 gave +25pp improvement)

Goal: Push past +267% WF avg or improve MDD by tuning signal thresholds
within the aggressive top_n=3 framework.
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
    print("  V160 — V157 top_n=3 Breakthrough + V158 Signal Threshold Optimization")
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

    # Regime indicators
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
    print(f"  Market vol median: {VOL_MEDIAN:.4f}%")
    print(f"  Done ({time.time()-t0:.1f}s)")

    # ===================== PARAMETERIZED SIGNALS =====================
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

    def sig_ov_id(di, edi, ov_thresh=0.3, id_thresh=0.3, roc_thresh=1.0):
        c = []
        for s in range(NS):
            ov = OV_GAP[s, di]; idr = ID_RET[s, di]; roc = ROC5[s, di]
            if any(np.isnan(x) for x in [ov, idr, roc]): continue
            if ov <= ov_thresh or idr <= id_thresh or roc <= roc_thresh: continue
            zs = ZSCORE[s, di]
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            z_bonus = zs if not np.isnan(zs) and zs > 1.0 else 1.0
            c.append(((ov + idr) * roc * z_bonus * 2, s, ep, 'ov_id'))
        return c

    def sig_final_flag(di, edi, roc20_thresh=5.0, range_atr_mult=3.0):
        c = []
        for s in range(NS):
            roc20 = ROC20[s, di]
            if np.isnan(roc20) or roc20 <= roc20_thresh or di < 6: continue
            h5 = H[s, di-4:di+1]; l5 = L[s, di-4:di+1]
            if any(np.isnan(x) for x in h5) or any(np.isnan(x) for x in l5): continue
            r5 = np.max(h5) - np.min(l5)
            atr = ATR14[s, di]
            if np.isnan(atr) or atr <= 0 or r5 > atr * range_atr_mult: continue
            h4 = np.max(H[s, di-4:di])
            cp = C[s, di]
            if np.isnan(cp) or cp <= h4: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((roc20 * (cp - h4) / atr, s, ep, 'ff'))
        return c

    def sig_union(di, edi, w_v121=3, w_ov=2, w_ff=1,
                  ov_thresh=0.3, id_thresh=0.3, roc_thresh_ov=1.0,
                  roc20_thresh=5.0, range_atr_mult=3.0):
        all_sigs = {}
        if w_v121 > 0:
            for item in sig_v121(di, edi):
                sc, s, ep, st = item
                if s not in all_sigs: all_sigs[s] = [0, ep, []]
                all_sigs[s][0] += sc * w_v121
                all_sigs[s][2].append('v121')
        if w_ov > 0:
            for item in sig_ov_id(di, edi, ov_thresh, id_thresh, roc_thresh_ov):
                sc, s, ep, st = item
                if s not in all_sigs: all_sigs[s] = [0, ep, []]
                all_sigs[s][0] += sc * w_ov
                all_sigs[s][2].append('ov_id')
        if w_ff > 0:
            for item in sig_final_flag(di, edi, roc20_thresh, range_atr_mult):
                sc, s, ep, st = item
                if s not in all_sigs: all_sigs[s] = [0, ep, []]
                all_sigs[s][0] += sc * w_ff
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
        if wr > 0.65: return 1.3
        elif wr >= 0.50: return 1.0
        else: return 0.5

    # ===================== UNIFIED BACKTEST ENGINE =====================
    def backtest(start_di=MIN_TRAIN, end_di=None,
                 # Signal params (from V158)
                 w_v121=3, w_ov=2, w_ff=1,
                 ov_thresh=0.3, id_thresh=0.3, roc_thresh_ov=1.0,
                 roc20_thresh=5.0, range_atr_mult=3.0,
                 # Sizing params (from V157)
                 dd_tiers=None, max_corr=0.5, sl_pct=0.0,
                 regime_lo=0.5, regime_hi=1.5,
                 # General
                 hold=1, top_n=3):
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

            # Stop-loss check
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

            # Kitchen Sink sizing
            dd_sz = dd_size(pv, high_water, dd_tiers)
            wr_mult_val = wr_size(trades, window=20)
            composite = compute_composite(di, daily_eq, high_water)
            regime_mult = regime_lo + composite * (regime_hi - regime_lo)
            pos_size = dd_sz * wr_mult_val * regime_mult
            pos_size = max(0.05, min(0.99, pos_size))

            # Enter positions
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue

            held_si = set(p['si'] for p in positions)

            # Get best V121 and best Union signal
            cands_v121 = sig_v121(di, edi)
            cands_union = sig_union(di, edi, w_v121=w_v121, w_ov=w_ov, w_ff=w_ff,
                                    ov_thresh=ov_thresh, id_thresh=id_thresh,
                                    roc_thresh_ov=roc_thresh_ov,
                                    roc20_thresh=roc20_thresh,
                                    range_atr_mult=range_atr_mult)
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

    # ===================== PRINTING =====================
    def pr(r, label=""):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  {label:85s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d}")

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

    # ===================== DD TIERS =====================
    DD_TIERS = {
        'aggro100': [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)],
        'aggro90':  [(0, 0.90), (0.10, 0.80), (0.20, 0.60), (0.30, 0.40)],
        'champ':    [(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)],
    }

    # ===================== SECTION 0: BASELINES =====================
    print("\n" + "=" * 130)
    print("  SECTION 0: BASELINES")
    print("=" * 130)

    # V157 champion: top_n=3, DD100/90/70/50, ov>0.3 id>0.3, noSL, reg0.5-1.5
    r = backtest(dd_tiers=DD_TIERS['aggro100'], max_corr=0.5, sl_pct=0.0,
                 regime_lo=0.5, regime_hi=1.5, top_n=3,
                 ov_thresh=0.3, id_thresh=0.3, roc_thresh_ov=1.0)
    pr(r, "V157 CHAMPION: top3 DD100/90/70/50 ov>0.3 id>0.3 noSL reg0.5-1.5")

    # V158 best: top_n=2, ov>0.1 id>0.5
    r = backtest(dd_tiers=DD_TIERS['aggro100'], max_corr=0.5, sl_pct=0.03,
                 regime_lo=0.5, regime_hi=1.5, top_n=2,
                 ov_thresh=0.1, id_thresh=0.5, roc_thresh_ov=1.0)
    pr(r, "V158 BEST: top2 DD100/90/70/50 ov>0.1 id>0.5 SL3% reg0.5-1.5")

    # ===================== SECTION 1: OV/ID THRESHOLD SWEEP (top_n=3, aggro100) =====================
    print("\n" + "=" * 130)
    print("  SECTION 1: OV/ID THRESHOLD SWEEP — top_n=3, DD100/90/70/50, noSL")
    print("=" * 130)

    s1_configs = [
        # (ov, id, roc)
        (0.05, 0.1, 0.5),
        (0.1,  0.1, 0.5),
        (0.1,  0.1, 1.0),
        (0.1,  0.3, 1.0),
        (0.1,  0.5, 1.0),
        (0.15, 0.5, 1.0),
        (0.2,  0.2, 1.0),
        (0.2,  0.3, 1.0),
        (0.2,  0.5, 1.0),
        (0.3,  0.1, 1.0),
        (0.3,  0.3, 1.0),  # V157 baseline
        (0.3,  0.3, 0.5),
        (0.3,  0.5, 1.0),
        (0.3,  0.5, 1.5),
        (0.5,  0.3, 1.0),
        (0.5,  0.5, 1.0),
        (0.05, 0.5, 0.5),
        (0.0,  0.3, 1.0),
        (0.0,  0.5, 1.0),
    ]

    s1_results = []
    for ov, idr, roc in s1_configs:
        label = f"ov>{ov} id>{idr} roc>{roc}"
        r = backtest(dd_tiers=DD_TIERS['aggro100'], max_corr=0.5, sl_pct=0.0,
                     regime_lo=0.5, regime_hi=1.5, top_n=3,
                     ov_thresh=ov, id_thresh=idr, roc_thresh_ov=roc)
        r['desc'] = label
        r['params'] = {'ov': ov, 'id': idr, 'roc': roc}
        s1_results.append(r)
        pr(r, f"S1 top3 DD100 {label}")

    # ===================== SECTION 2: SIGNAL WEIGHT SWEEP =====================
    print("\n" + "=" * 130)
    print("  SECTION 2: SIGNAL WEIGHT SWEEP — top_n=3, DD100/90/70/50, noSL")
    print("  Using best OV/ID thresholds from V158 (ov>0.1 id>0.5)")
    print("=" * 130)

    s2_configs = [
        # (w_v121, w_ov, w_ff)
        (3, 2, 1),   # baseline
        (3, 2, 0),
        (3, 0, 1),
        (3, 1, 1),
        (3, 3, 1),
        (4, 2, 1),
        (5, 2, 1),
        (3, 2, 2),
        (3, 3, 2),
        (4, 3, 2),
        (5, 3, 2),
        (2, 2, 1),
        (3, 1, 2),
    ]

    s2_results = []
    for wv, wo, wf in s2_configs:
        label = f"V{wv}*O{wo}*F{wf}"
        # Test with both V157 baseline thresholds and V158 best thresholds
        for ov, idr, roc, thresh_label in [(0.3, 0.3, 1.0, "ov03id03"),
                                             (0.1, 0.5, 1.0, "ov01id05")]:
            full_label = f"S2 top3 DD100 {label} {thresh_label}"
            r = backtest(dd_tiers=DD_TIERS['aggro100'], max_corr=0.5, sl_pct=0.0,
                         regime_lo=0.5, regime_hi=1.5, top_n=3,
                         w_v121=wv, w_ov=wo, w_ff=wf,
                         ov_thresh=ov, id_thresh=idr, roc_thresh_ov=roc)
            r['desc'] = full_label
            s2_results.append(r)
            pr(r, full_label)

    # ===================== SECTION 3: DD TIER + SL COMBO =====================
    print("\n" + "=" * 130)
    print("  SECTION 3: DD TIER + SL COMBO — best signal thresholds, top_n=3")
    print("=" * 130)

    s3_configs = [
        # (dd_name, sl, ov, id, roc)
        ('aggro100', 0.0,  0.3, 0.3, 1.0),
        ('aggro100', 0.0,  0.1, 0.5, 1.0),
        ('aggro100', 0.0,  0.2, 0.5, 1.0),
        ('aggro100', 0.03, 0.3, 0.3, 1.0),
        ('aggro100', 0.03, 0.1, 0.5, 1.0),
        ('aggro100', 0.05, 0.3, 0.3, 1.0),
        ('aggro100', 0.05, 0.1, 0.5, 1.0),
        ('aggro90',  0.0,  0.3, 0.3, 1.0),
        ('aggro90',  0.0,  0.1, 0.5, 1.0),
        ('aggro90',  0.03, 0.1, 0.5, 1.0),
        ('champ',    0.0,  0.1, 0.5, 1.0),
        ('champ',    0.03, 0.1, 0.5, 1.0),
    ]

    s3_results = []
    for dd_name, sl, ov, idr, roc in s3_configs:
        sl_str = f"SL{sl*100:.0f}%" if sl > 0 else "noSL"
        label = f"S3 top3 {dd_name} ov>{ov} id>{idr} {sl_str}"
        r = backtest(dd_tiers=DD_TIERS[dd_name], max_corr=0.5, sl_pct=sl,
                     regime_lo=0.5, regime_hi=1.5, top_n=3,
                     ov_thresh=ov, id_thresh=idr, roc_thresh_ov=roc)
        r['desc'] = label
        s3_results.append(r)
        pr(r, label)

    # ===================== SECTION 4: TOP_N COMPARISON =====================
    print("\n" + "=" * 130)
    print("  SECTION 4: TOP_N COMPARISON — best thresholds at each top_n")
    print("=" * 130)

    s4_results = []
    for tn in [1, 2, 3, 4, 5]:
        for ov, idr, roc in [(0.3, 0.3, 1.0), (0.1, 0.5, 1.0), (0.2, 0.5, 1.0)]:
            label = f"S4 top{tn} DD100 ov>{ov} id>{idr} noSL"
            r = backtest(dd_tiers=DD_TIERS['aggro100'], max_corr=0.5, sl_pct=0.0,
                         regime_lo=0.5, regime_hi=1.5, top_n=tn,
                         ov_thresh=ov, id_thresh=idr, roc_thresh_ov=roc)
            r['desc'] = label
            s4_results.append(r)
            pr(r, label)

    # ===================== COMPREHENSIVE RANKING =====================
    print("\n" + "=" * 130)
    print("  COMPREHENSIVE RANKING — ALL CONFIGS")
    print("=" * 130)

    all_results = s1_results + s2_results + s3_results + s4_results
    all_results.sort(key=lambda x: -x['ann'])

    print(f"\n  Top 25 by Annual Return:")
    for i, r in enumerate(all_results[:25]):
        desc = r.get('desc', '')
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i+1:2d}: {desc:65s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    print(f"\n  Top 25 by R/M Ratio:")
    all_rm = sorted(all_results, key=lambda x: -abs(x['ann']/x['mdd']) if x['mdd'] != 0 else 0)
    for i, r in enumerate(all_rm[:25]):
        desc = r.get('desc', '')
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i+1:2d}: {desc:65s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    # ===================== WALK-FORWARD: TOP 15 CONFIGS =====================
    print("\n" + "=" * 130)
    print("  WALK-FORWARD VALIDATION — TOP 15 by full-period R/M")
    print("=" * 130)

    # Deduplicate
    seen = set()
    wf_candidates = []
    for r in all_rm:
        desc = r.get('desc', '')
        if desc not in seen:
            seen.add(desc)
            wf_candidates.append(r)

    wf_all = []
    for r in wf_candidates[:15]:
        desc = r.get('desc', '')
        # Parse params from desc for WF
        # Use backtest defaults (top_n=3, DD100) and extract signal params
        wf_res = walk_forward(label=desc,
                              dd_tiers=DD_TIERS['aggro100'],
                              max_corr=0.5, sl_pct=0.0,
                              regime_lo=0.5, regime_hi=1.5, top_n=3,
                              ov_thresh=r.get('params', {}).get('ov', 0.3),
                              id_thresh=r.get('params', {}).get('id', 0.3),
                              roc_thresh_ov=r.get('params', {}).get('roc', 1.0))
        avg_ann = np.mean([r2['ann'] for r2 in wf_res.values()])
        worst_mdd = min(r2['mdd'] for r2 in wf_res.values())
        pos = sum(1 for r2 in wf_res.values() if r2['ann'] > 0)
        wf_all.append({'desc': desc, 'avg_ann': avg_ann, 'worst_mdd': worst_mdd,
                       'pos': pos, 'wf_res': wf_res, 'full_ann': r['ann'],
                       'full_mdd': r['mdd']})
        print_wf(wf_res, desc)

    # Also run WF for specifically interesting configs
    interesting = [
        ("V157 CHAMPION ov>0.3 id>0.3", {'ov_thresh': 0.3, 'id_thresh': 0.3, 'roc_thresh_ov': 1.0}),
        ("V160 HYPOTHESIS ov>0.1 id>0.5", {'ov_thresh': 0.1, 'id_thresh': 0.5, 'roc_thresh_ov': 1.0}),
        ("V160 HYPOTHESIS ov>0.2 id>0.5", {'ov_thresh': 0.2, 'id_thresh': 0.5, 'roc_thresh_ov': 1.0}),
        ("V160 HYPOTHESIS ov>0.0 id>0.5", {'ov_thresh': 0.0, 'id_thresh': 0.5, 'roc_thresh_ov': 1.0}),
    ]
    for desc, params in interesting:
        if not any(w['desc'] == desc for w in wf_all):
            wf_res = walk_forward(label=desc,
                                  dd_tiers=DD_TIERS['aggro100'],
                                  max_corr=0.5, sl_pct=0.0,
                                  regime_lo=0.5, regime_hi=1.5, top_n=3,
                                  **params)
            avg_ann = np.mean([r2['ann'] for r2 in wf_res.values()])
            worst_mdd = min(r2['mdd'] for r2 in wf_res.values())
            pos = sum(1 for r2 in wf_res.values() if r2['ann'] > 0)
            wf_all.append({'desc': desc, 'avg_ann': avg_ann, 'worst_mdd': worst_mdd,
                           'pos': pos, 'wf_res': wf_res, 'full_ann': 0, 'full_mdd': 0})
            print_wf(wf_res, desc)

    # ===================== FINAL RANKING BY WF AVG =====================
    print("\n" + "=" * 130)
    print("  FINAL RANKING BY WF AVERAGE")
    print("=" * 130)

    wf_all.sort(key=lambda x: -x['avg_ann'])
    for i, w in enumerate(wf_all[:10]):
        desc = w['desc']
        ratio = abs(w['avg_ann'] / w['worst_mdd']) if w['worst_mdd'] != 0 else 0
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(w['wf_res'].items())])
        print(f"\n  #{i+1}: {desc}")
        print(f"       WF AVG={w['avg_ann']:+.0f}% | WorstWfMDD={w['worst_mdd']:.0f}% | R/M={ratio:.2f} | {w['pos']}/6 pos")
        print(f"       {ws}")

    # ===================== PARETO FRONTIER =====================
    print("\n" + "=" * 130)
    print("  PARETO FRONTIER UPDATE")
    print("=" * 130)

    print(f"\n  Previous champions:")
    print(f"    V157: top3 DD100/90/70/50 Kitchen Sink = +267%/-26%")
    print(f"    V146: Sink DD70/60/40/20 top2 = +185%/-24%")

    if wf_all:
        best = wf_all[0]
        print(f"\n  NEW V160 BEST: {best['desc']}")
        print(f"    WF AVG={best['avg_ann']:+.0f}% | WorstWfMDD={best['worst_mdd']:.0f}% | {best['pos']}/6 pos")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
