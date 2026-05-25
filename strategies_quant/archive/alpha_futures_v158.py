"""
Alpha Futures V158 — OV/ID and FinalFlag Signal Parameter Optimization
=============================================================================
V146 Kitchen Sink gives +185%/-24% WF. The signal is Union = V121(*3) + OV/ID(*2) + FinalFlag(*1).
V121 alone is strong but OV/ID and FinalFlag contribute.

This script systematically optimizes:
  1. OV/ID threshold sweep (ov_gap, id_ret, roc5)
  2. FinalFlag threshold sweep (roc20, range_atr)
  3. Signal weight sweep (V121 weight, OV/ID weight, FF weight)
  4. OV/ID score formula variants
  5. Signal ablation (disable each signal)
  6. Walk-forward for top configs

All configs use Kitchen Sink sizing (proven best in V146).
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
    print("  V158 — OV/ID and FinalFlag Signal Parameter Optimization")
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

    # Market indicators for Kitchen Sink sizing
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

    # ===================== PARAMETERIZED SIGNAL DEFINITIONS =====================
    def sig_v121(di, edi):
        """V121 signal: ROC(5)>1% + Z>1.5 + ROC improving"""
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

    def sig_ov_id(di, edi, ov_thresh=0.3, id_thresh=0.3, roc_thresh=1.0,
                  formula='sum'):
        """
        OV/ID signal with configurable thresholds.
        formula: 'sum' = (ov + idr) * roc * z_bonus * 2  (original)
                 'prod' = (ov * idr) * roc * z_bonus * 2
                 'sq'   = (ov + idr)^2 * roc * z_bonus * 2
        """
        c = []
        for s in range(NS):
            ov = OV_GAP[s, di]; idr = ID_RET[s, di]; roc = ROC5[s, di]
            if any(np.isnan(x) for x in [ov, idr, roc]): continue
            if ov <= ov_thresh or idr <= id_thresh or roc <= roc_thresh: continue
            zs = ZSCORE[s, di]
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            z_bonus = zs if not np.isnan(zs) and zs > 1.0 else 1.0
            if formula == 'sum':
                sc = (ov + idr) * roc * z_bonus * 2
            elif formula == 'prod':
                sc = (ov * idr) * roc * z_bonus * 2
            elif formula == 'sq':
                sc = (ov + idr) ** 2 * roc * z_bonus * 2
            else:
                sc = (ov + idr) * roc * z_bonus * 2
            c.append((sc, s, ep, 'ov_id'))
        return c

    def sig_final_flag(di, edi, roc20_thresh=5.0, range_atr_mult=3.0):
        """FinalFlag signal with configurable thresholds."""
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
                  ov_formula='sum', roc20_thresh=5.0, range_atr_mult=3.0):
        """Weighted union of signals."""
        all_sigs = {}
        if w_v121 > 0:
            for item in sig_v121(di, edi):
                sc, s, ep, st = item
                if s not in all_sigs: all_sigs[s] = [0, ep, []]
                all_sigs[s][0] += sc * w_v121
                all_sigs[s][2].append('v121')
        if w_ov > 0:
            for item in sig_ov_id(di, edi, ov_thresh, id_thresh, roc_thresh_ov, ov_formula):
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

    # ===================== HELPER: WR-adaptive sizing =====================
    def wr_size(trades, window=20):
        if len(trades) < window:
            return 1.0
        recent = trades[-window:]
        wr = np.mean([1 if t > 0 else 0 for t in recent])
        if wr > 0.65: return 1.3
        elif wr >= 0.50: return 1.0
        else: return 0.5

    # ===================== HELPER: Composite regime =====================
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

    # ===================== BACKTEST ENGINE =====================
    # Default Kitchen Sink params
    DEFAULT_DD_TIERS = [(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)]

    def backtest(start_di=MIN_TRAIN, end_di=None,
                 # Signal params
                 w_v121=3, w_ov=2, w_ff=1,
                 ov_thresh=0.3, id_thresh=0.3, roc_thresh_ov=1.0,
                 ov_formula='sum', roc20_thresh=5.0, range_atr_mult=3.0,
                 # Sizing params (Kitchen Sink)
                 dd_tiers=None, max_corr=0.5, sl_pct=0.03,
                 # General
                 hold=1, top_n=2):
        if end_di is None: end_di = ND
        if dd_tiers is None: dd_tiers = DEFAULT_DD_TIERS

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
            regime_mult = 0.5 + composite
            pos_size = dd_sz * wr_mult_val * regime_mult
            pos_size = max(0.05, min(0.95, pos_size))

            # Enter positions
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue

            held_si = set(p['si'] for p in positions)

            # Get union candidates
            cands_union = sig_union(di, edi,
                                    w_v121=w_v121, w_ov=w_ov, w_ff=w_ff,
                                    ov_thresh=ov_thresh, id_thresh=id_thresh,
                                    roc_thresh_ov=roc_thresh_ov,
                                    ov_formula=ov_formula,
                                    roc20_thresh=roc20_thresh,
                                    range_atr_mult=range_atr_mult)
            cands_union.sort(key=lambda x: -x[0])

            # Get V121 candidates (always separate for Cross+Corr mode)
            cands_v121 = sig_v121(di, edi)
            cands_v121.sort(key=lambda x: -x[0])

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
            for sc, s, pr_entry, sig_str, pct in entries:
                if s in set(p['si'] for p in positions): continue
                if len(positions) >= top_n: break
                cap = cash_snapshot * pct / n_planned
                sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                ct = max(1, int(cap / (pr_entry * m * (1 + COMM))))
                ci = pr_entry * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (pr_entry * m * (1 + COMM)))
                    ci = pr_entry * m * ct * (1 + COMM) if ct > 0 else 0
                if ct <= 0 or ci <= 0 or ci > cash: continue
                cash -= ci
                positions.append({'si': s, 'entry_price': pr_entry, 'entry_di': edi,
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

    # ===================== WALK-FORWARD =====================
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

    # ===================== PRINTING =====================
    def pr(r, label=""):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  {label:80s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d}")

    # ===================== SECTION 0: BASELINE (V146 default) =====================
    print("\n" + "=" * 130)
    print("  SECTION 0: BASELINE — V146 Kitchen Sink with SL=3%")
    print("=" * 130)
    r0 = backtest()
    pr(r0, "BASELINE: V121*3 OV*2 FF*1, ov>0.3 id>0.3 roc>1.0, sum, r20>5.0 range<3atr")
    # Also test with SL=0
    r0_nosl = backtest(sl_pct=0.0)
    pr(r0_nosl, "BASELINE NO SL: V121*3 OV*2 FF*1, ov>0.3 id>0.3 roc>1.0, sum")

    # ===================== SECTION 1: OV/ID THRESHOLD SWEEP =====================
    print("\n" + "=" * 130)
    print("  SECTION 1: OV/ID THRESHOLD SWEEP")
    print("  Fix V121*3, FF*1, sum formula. Sweep ov_thresh, id_thresh, roc_thresh_ov")
    print("=" * 130)

    s1_results = []
    s1_configs = [
        # (ov, id, roc, label)
        (0.1, 0.1, 0.5, "S1: ov>0.1 id>0.1 roc>0.5"),
        (0.1, 0.1, 1.0, "S1: ov>0.1 id>0.1 roc>1.0"),
        (0.1, 0.3, 1.0, "S1: ov>0.1 id>0.3 roc>1.0"),
        (0.2, 0.2, 1.0, "S1: ov>0.2 id>0.2 roc>1.0"),
        (0.2, 0.3, 1.0, "S1: ov>0.2 id>0.3 roc>1.0"),
        (0.3, 0.1, 1.0, "S1: ov>0.3 id>0.1 roc>1.0"),
        (0.3, 0.3, 0.5, "S1: ov>0.3 id>0.3 roc>0.5"),
        (0.3, 0.3, 1.0, "S1: ov>0.3 id>0.3 roc>1.0 (BASELINE)"),
        (0.3, 0.3, 1.5, "S1: ov>0.3 id>0.3 roc>1.5"),
        (0.5, 0.3, 1.0, "S1: ov>0.5 id>0.3 roc>1.0"),
        (0.3, 0.5, 1.0, "S1: ov>0.3 id>0.5 roc>1.0"),
        (0.5, 0.5, 1.0, "S1: ov>0.5 id>0.5 roc>1.0"),
        (0.5, 0.5, 1.5, "S1: ov>0.5 id>0.5 roc>1.5"),
        (0.2, 0.2, 0.5, "S1: ov>0.2 id>0.2 roc>0.5 (loose)"),
        (0.1, 0.5, 1.0, "S1: ov>0.1 id>0.5 roc>1.0"),
        (0.5, 0.1, 1.0, "S1: ov>0.5 id>0.1 roc>1.0"),
    ]
    for ov, idr, roc, label in s1_configs:
        r = backtest(ov_thresh=ov, id_thresh=idr, roc_thresh_ov=roc)
        r['desc'] = label
        s1_results.append(r)
        pr(r, label)

    # ===================== SECTION 2: SIGNAL WEIGHT SWEEP =====================
    print("\n" + "=" * 130)
    print("  SECTION 2: SIGNAL WEIGHT SWEEP")
    print("  Fix baseline OV/ID thresholds. Sweep weights for V121, OV/ID, FF")
    print("=" * 130)

    s2_results = []
    s2_configs = [
        # (w_v121, w_ov, w_ff, label)
        (3, 2, 1, "S2: V121*3 OV*2 FF*1 (BASELINE)"),
        (3, 0, 0, "S2: V121*3 only (no OV/ID, no FF)"),
        (3, 2, 0, "S2: V121*3 OV*2 (no FF)"),
        (3, 0, 1, "S2: V121*3 FF*1 (no OV/ID)"),
        (0, 2, 1, "S2: OV*2 FF*1 only (no V121)"),
        (2, 2, 1, "S2: V121*2 OV*2 FF*1"),
        (4, 2, 1, "S2: V121*4 OV*2 FF*1"),
        (5, 2, 1, "S2: V121*5 OV*2 FF*1"),
        (3, 1, 1, "S2: V121*3 OV*1 FF*1"),
        (3, 3, 1, "S2: V121*3 OV*3 FF*1"),
        (3, 2, 2, "S2: V121*3 OV*2 FF*2"),
        (3, 1, 2, "S2: V121*3 OV*1 FF*2"),
        (3, 3, 2, "S2: V121*3 OV*3 FF*2"),
        (4, 1, 1, "S2: V121*4 OV*1 FF*1"),
        (4, 3, 1, "S2: V121*4 OV*3 FF*1"),
        (5, 3, 2, "S2: V121*5 OV*3 FF*2"),
        (3, 2, 0, "S2: V121*3 OV*2 (no FF) [dup check]"),
        (5, 1, 0, "S2: V121*5 OV*1 (no FF)"),
    ]
    for wv, wo, wf, label in s2_configs:
        r = backtest(w_v121=wv, w_ov=wo, w_ff=wf)
        r['desc'] = label
        s2_results.append(r)
        pr(r, label)

    # ===================== SECTION 3: OV/ID SCORE FORMULA VARIANTS =====================
    print("\n" + "=" * 130)
    print("  SECTION 3: OV/ID SCORE FORMULA VARIANTS")
    print("  sum=(ov+idr)*roc*z*2, prod=(ov*idr)*roc*z*2, sq=(ov+idr)^2*roc*z*2")
    print("=" * 130)

    s3_results = []
    s3_configs = [
        # (formula, ov, id, roc, label)
        ('sum',  0.3, 0.3, 1.0, "S3: sum ov>0.3 id>0.3 roc>1.0 (BASELINE)"),
        ('prod', 0.3, 0.3, 1.0, "S3: prod ov>0.3 id>0.3 roc>1.0"),
        ('sq',   0.3, 0.3, 1.0, "S3: sq ov>0.3 id>0.3 roc>1.0"),
        ('sum',  0.2, 0.2, 1.0, "S3: sum ov>0.2 id>0.2 roc>1.0"),
        ('prod', 0.2, 0.2, 1.0, "S3: prod ov>0.2 id>0.2 roc>1.0"),
        ('sq',   0.2, 0.2, 1.0, "S3: sq ov>0.2 id>0.2 roc>1.0"),
        ('sum',  0.1, 0.3, 1.0, "S3: sum ov>0.1 id>0.3 roc>1.0"),
        ('prod', 0.1, 0.3, 1.0, "S3: prod ov>0.1 id>0.3 roc>1.0"),
        ('sq',   0.1, 0.3, 1.0, "S3: sq ov>0.1 id>0.3 roc>1.0"),
        ('sum',  0.3, 0.3, 0.5, "S3: sum ov>0.3 id>0.3 roc>0.5"),
        ('prod', 0.3, 0.3, 0.5, "S3: prod ov>0.3 id>0.3 roc>0.5"),
        ('sq',   0.3, 0.3, 0.5, "S3: sq ov>0.3 id>0.3 roc>0.5"),
    ]
    for formula, ov, idr, roc, label in s3_configs:
        r = backtest(ov_formula=formula, ov_thresh=ov, id_thresh=idr, roc_thresh_ov=roc)
        r['desc'] = label
        s3_results.append(r)
        pr(r, label)

    # ===================== SECTION 4: FINALFLAG THRESHOLD SWEEP =====================
    print("\n" + "=" * 130)
    print("  SECTION 4: FINALFLAG THRESHOLD SWEEP")
    print("  Fix V121*3, OV*2 baseline. Sweep roc20_thresh and range_atr_mult")
    print("=" * 130)

    s4_results = []
    s4_configs = [
        # (roc20_thresh, range_atr_mult, label)
        (3.0, 2.0, "S4: FF r20>3% range<2atr"),
        (3.0, 3.0, "S4: FF r20>3% range<3atr"),
        (3.0, 4.0, "S4: FF r20>3% range<4atr"),
        (5.0, 2.0, "S4: FF r20>5% range<2atr"),
        (5.0, 3.0, "S4: FF r20>5% range<3atr (BASELINE)"),
        (5.0, 4.0, "S4: FF r20>5% range<4atr"),
        (8.0, 2.0, "S4: FF r20>8% range<2atr"),
        (8.0, 3.0, "S4: FF r20>8% range<3atr"),
        (8.0, 4.0, "S4: FF r20>8% range<4atr"),
        (3.0, 3.0, "S4: FF r20>3% range<3atr (dup baseline r20)"),
        (4.0, 3.0, "S4: FF r20>4% range<3atr"),
        (6.0, 3.0, "S4: FF r20>6% range<3atr"),
        (7.0, 3.0, "S4: FF r20>7% range<3atr"),
        (10.0, 3.0, "S4: FF r20>10% range<3atr"),
    ]
    for r20t, ratr, label in s4_configs:
        r = backtest(roc20_thresh=r20t, range_atr_mult=ratr)
        r['desc'] = label
        s4_results.append(r)
        pr(r, label)

    # ===================== SECTION 5: SIGNAL ABLATION STUDY =====================
    print("\n" + "=" * 130)
    print("  SECTION 5: SIGNAL ABLATION STUDY")
    print("  Measure individual signal contribution")
    print("=" * 130)

    s5_results = []
    s5_configs = [
        # (w_v121, w_ov, w_ff, ov, id, roc, formula, r20, ratr, label)
        (3, 2, 1, 0.3, 0.3, 1.0, 'sum', 5.0, 3.0, "ABLATION: Full baseline V121*3 OV*2 FF*1"),
        (3, 0, 0, 0.3, 0.3, 1.0, 'sum', 5.0, 3.0, "ABLATION: V121 only (no OV/ID no FF)"),
        (3, 2, 0, 0.3, 0.3, 1.0, 'sum', 5.0, 3.0, "ABLATION: V121+OV/ID (no FF)"),
        (3, 0, 1, 0.3, 0.3, 1.0, 'sum', 5.0, 3.0, "ABLATION: V121+FF (no OV/ID)"),
        (0, 2, 1, 0.3, 0.3, 1.0, 'sum', 5.0, 3.0, "ABLATION: OV/ID+FF (no V121)"),
        (0, 2, 0, 0.3, 0.3, 1.0, 'sum', 5.0, 3.0, "ABLATION: OV/ID only"),
        (0, 0, 1, 0.3, 0.3, 1.0, 'sum', 5.0, 3.0, "ABLATION: FF only"),
    ]
    for wv, wo, wf, ov, idr, roc, formula, r20, ratr, label in s5_configs:
        r = backtest(w_v121=wv, w_ov=wo, w_ff=wf,
                     ov_thresh=ov, id_thresh=idr, roc_thresh_ov=roc,
                     ov_formula=formula, roc20_thresh=r20, range_atr_mult=ratr)
        r['desc'] = label
        s5_results.append(r)
        pr(r, label)

    # ===================== SECTION 6: COMBINED BEST PARAMETER SEARCH =====================
    print("\n" + "=" * 130)
    print("  SECTION 6: COMBINED PARAMETER SEARCH — best from Sections 1-4")
    print("  Try combining best OV/ID thresholds with best FF thresholds and weight combos")
    print("=" * 130)

    # Identify promising parameter regions from Sections 1-4
    # We'll test a grid of promising combos
    s6_results = []
    s6_configs = [
        # Combine relaxed OV/ID + tuned FF
        (3, 2, 1, 0.2, 0.2, 1.0, 'sum', 5.0, 3.0, "S6: V3 O2 F1, ov>0.2 id>0.2, r20>5 r<3"),
        (3, 2, 1, 0.2, 0.2, 1.0, 'sum', 3.0, 3.0, "S6: V3 O2 F1, ov>0.2 id>0.2, r20>3 r<3"),
        (3, 2, 1, 0.2, 0.2, 1.0, 'sum', 5.0, 4.0, "S6: V3 O2 F1, ov>0.2 id>0.2, r20>5 r<4"),
        (3, 2, 1, 0.1, 0.3, 1.0, 'sum', 5.0, 3.0, "S6: V3 O2 F1, ov>0.1 id>0.3, r20>5 r<3"),
        (3, 2, 1, 0.3, 0.1, 1.0, 'sum', 5.0, 3.0, "S6: V3 O2 F1, ov>0.3 id>0.1, r20>5 r<3"),
        (3, 2, 1, 0.3, 0.3, 0.5, 'sum', 5.0, 3.0, "S6: V3 O2 F1, ov>0.3 id>0.3 roc>0.5, r20>5 r<3"),
        (3, 2, 1, 0.2, 0.2, 0.5, 'sum', 5.0, 3.0, "S6: V3 O2 F1, ov>0.2 id>0.2 roc>0.5, r20>5 r<3"),
        (3, 2, 1, 0.1, 0.3, 1.0, 'sum', 3.0, 4.0, "S6: V3 O2 F1, ov>0.1 id>0.3, r20>3 r<4"),
        (3, 2, 1, 0.3, 0.3, 1.0, 'sum', 3.0, 4.0, "S6: V3 O2 F1, ov>0.3 id>0.3, r20>3 r<4"),
        # With prod formula
        (3, 2, 1, 0.3, 0.3, 1.0, 'prod', 5.0, 3.0, "S6: V3 O2 F1 prod, ov>0.3 id>0.3, r20>5 r<3"),
        (3, 2, 1, 0.2, 0.2, 1.0, 'prod', 5.0, 3.0, "S6: V3 O2 F1 prod, ov>0.2 id>0.2, r20>5 r<3"),
        # Weight variations with best thresholds
        (3, 3, 1, 0.2, 0.2, 1.0, 'sum', 5.0, 3.0, "S6: V3 O3 F1, ov>0.2 id>0.2, r20>5 r<3"),
        (4, 2, 1, 0.2, 0.2, 1.0, 'sum', 5.0, 3.0, "S6: V4 O2 F1, ov>0.2 id>0.2, r20>5 r<3"),
        (5, 2, 1, 0.2, 0.2, 1.0, 'sum', 5.0, 3.0, "S6: V5 O2 F1, ov>0.2 id>0.2, r20>5 r<3"),
        (3, 2, 2, 0.2, 0.2, 1.0, 'sum', 5.0, 3.0, "S6: V3 O2 F2, ov>0.2 id>0.2, r20>5 r<3"),
        (3, 1, 2, 0.2, 0.2, 1.0, 'sum', 5.0, 3.0, "S6: V3 O1 F2, ov>0.2 id>0.2, r20>5 r<3"),
        (3, 3, 2, 0.2, 0.2, 1.0, 'sum', 3.0, 4.0, "S6: V3 O3 F2, ov>0.2 id>0.2, r20>3 r<4"),
        (4, 3, 2, 0.2, 0.2, 1.0, 'sum', 5.0, 3.0, "S6: V4 O3 F2, ov>0.2 id>0.2, r20>5 r<3"),
        # More FF weight
        (3, 2, 2, 0.3, 0.3, 1.0, 'sum', 3.0, 4.0, "S6: V3 O2 F2, ov>0.3 id>0.3, r20>3 r<4"),
        (3, 2, 3, 0.3, 0.3, 1.0, 'sum', 3.0, 4.0, "S6: V3 O2 F3, ov>0.3 id>0.3, r20>3 r<4"),
    ]
    for wv, wo, wf, ov, idr, roc, formula, r20, ratr, label in s6_configs:
        r = backtest(w_v121=wv, w_ov=wo, w_ff=wf,
                     ov_thresh=ov, id_thresh=idr, roc_thresh_ov=roc,
                     ov_formula=formula, roc20_thresh=r20, range_atr_mult=ratr)
        r['desc'] = label
        s6_results.append(r)
        pr(r, label)

    # ===================== COMPREHENSIVE RANKING =====================
    print("\n" + "=" * 130)
    print("  COMPREHENSIVE RANKING — ALL CONFIGS")
    print("=" * 130)

    all_results = s1_results + s2_results + s3_results + s4_results + s5_results + s6_results
    all_valid = [r for r in all_results if r.get('desc', '') and r['mdd'] > -80]

    # Top by annual return
    all_valid.sort(key=lambda x: -x['ann'])
    print(f"\n  Top 20 by Annual Return:")
    for i, r in enumerate(all_valid[:20]):
        desc = r.get('desc', '')
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i+1:2d}: {desc:80s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    # Top by R/M ratio
    all_with_ratio = [(r, abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0) for r in all_valid]
    all_with_ratio.sort(key=lambda x: -x[1])
    print(f"\n  Top 20 by Ann/MDD Ratio:")
    for i, (r, ratio) in enumerate(all_with_ratio[:20]):
        desc = r.get('desc', '')
        print(f"  #{i+1:2d}: {desc:80s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    # Top by Sharpe
    all_valid_sh = list(all_valid)
    all_valid_sh.sort(key=lambda x: -x['sharpe'])
    print(f"\n  Top 10 by Sharpe:")
    for i, r in enumerate(all_valid_sh[:10]):
        desc = r.get('desc', '')
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i+1:2d}: {desc:80s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    # ===================== SELECT TOP CONFIGS FOR WALK-FORWARD =====================
    print("\n" + "=" * 130)
    print("  WALK-FORWARD VALIDATION")
    print("=" * 130)

    # Build WF configs from top results
    # We need to reconstruct params from desc
    # Strategy: store params alongside results for top configs

    # Collect all configs with their params
    all_configs_with_params = []
    for ov, idr, roc, label in s1_configs:
        all_configs_with_params.append({
            'label': label, 'w_v121': 3, 'w_ov': 2, 'w_ff': 1,
            'ov_thresh': ov, 'id_thresh': idr, 'roc_thresh_ov': roc,
            'ov_formula': 'sum', 'roc20_thresh': 5.0, 'range_atr_mult': 3.0
        })
    for wv, wo, wf, label in s2_configs:
        all_configs_with_params.append({
            'label': label, 'w_v121': wv, 'w_ov': wo, 'w_ff': wf,
            'ov_thresh': 0.3, 'id_thresh': 0.3, 'roc_thresh_ov': 1.0,
            'ov_formula': 'sum', 'roc20_thresh': 5.0, 'range_atr_mult': 3.0
        })
    for formula, ov, idr, roc, label in s3_configs:
        all_configs_with_params.append({
            'label': label, 'w_v121': 3, 'w_ov': 2, 'w_ff': 1,
            'ov_thresh': ov, 'id_thresh': idr, 'roc_thresh_ov': roc,
            'ov_formula': formula, 'roc20_thresh': 5.0, 'range_atr_mult': 3.0
        })
    for r20t, ratr, label in s4_configs:
        all_configs_with_params.append({
            'label': label, 'w_v121': 3, 'w_ov': 2, 'w_ff': 1,
            'ov_thresh': 0.3, 'id_thresh': 0.3, 'roc_thresh_ov': 1.0,
            'ov_formula': 'sum', 'roc20_thresh': r20t, 'range_atr_mult': ratr
        })
    for wv, wo, wf, ov, idr, roc, formula, r20, ratr, label in s5_configs:
        all_configs_with_params.append({
            'label': label, 'w_v121': wv, 'w_ov': wo, 'w_ff': wf,
            'ov_thresh': ov, 'id_thresh': idr, 'roc_thresh_ov': roc,
            'ov_formula': formula, 'roc20_thresh': r20, 'range_atr_mult': ratr
        })
    for wv, wo, wf, ov, idr, roc, formula, r20, ratr, label in s6_configs:
        all_configs_with_params.append({
            'label': label, 'w_v121': wv, 'w_ov': wo, 'w_ff': wf,
            'ov_thresh': ov, 'id_thresh': idr, 'roc_thresh_ov': roc,
            'ov_formula': formula, 'roc20_thresh': r20, 'range_atr_mult': ratr
        })

    # Create lookup from label to params
    param_lookup = {c['label']: c for c in all_configs_with_params}

    # Select top configs for WF from top by R/M ratio (deduplicated)
    seen = set()
    wf_candidates = []
    for r, ratio in all_with_ratio:
        desc = r.get('desc', '')
        if desc not in seen and desc in param_lookup:
            seen.add(desc)
            wf_candidates.append((r, ratio))

    # Run WF for top 20 by R/M ratio
    print(f"\n  Walk-forward for top 20 configs by R/M ratio...")
    wf_all = {}
    wf_results_list = []
    for r, ratio in wf_candidates[:20]:
        desc = r.get('desc', '')
        params = param_lookup[desc]
        wf_kwargs = {k: v for k, v in params.items() if k != 'label'}
        wf_res = walk_forward(label=desc, **wf_kwargs)
        wf_all[desc] = wf_res

        avg_ann = np.mean([r2['ann'] for r2 in wf_res.values()])
        worst_mdd = min(r2['mdd'] for r2 in wf_res.values())
        pos_yrs = sum(1 for r2 in wf_res.values() if r2['ann'] > 0)
        wf_results_list.append({
            'desc': desc, 'avg_ann': avg_ann, 'worst_mdd': worst_mdd,
            'pos_years': pos_yrs, 'params': params, 'wf_res': wf_res
        })
        print_wf(wf_res, desc)

    # ===================== TOP 5 WF CONFIGS WITH WF MDD > -30% =====================
    print("\n" + "=" * 130)
    print("  TOP 5 CONFIGS BY WF AVG (with WF MDD > -30%)")
    print("=" * 130)

    qualified = [w for w in wf_results_list if w['worst_mdd'] > -30]
    qualified.sort(key=lambda x: -x['avg_ann'])

    if not qualified:
        print("\n  No configs with WF MDD > -30%. Showing all sorted by avg WF:")
        wf_results_list.sort(key=lambda x: -x['avg_ann'])
        qualified = wf_results_list

    for i, w in enumerate(qualified[:5]):
        desc = w['desc']
        print(f"\n  #{i+1}: {desc}")
        print(f"        AvgWF={w['avg_ann']:+.0f}% | WorstWfMDD={w['worst_mdd']:.1f}% | {w['pos_years']}/6 pos")
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(w['wf_res'].items())])
        print(f"        {ws}")
        params = w['params']
        print(f"        Params: V121*{params['w_v121']} OV*{params['w_ov']} FF*{params['w_ff']} | "
              f"ov>{params['ov_thresh']} id>{params['id_thresh']} roc>{params['roc_thresh_ov']} | "
              f"{params['ov_formula']} | r20>{params['roc20_thresh']} r<{params['range_atr_mult']}atr")

    # ===================== DETAILED WF TABLE =====================
    print("\n" + "=" * 130)
    print("  DETAILED WF TABLE: ALL WALK-FORWARD RESULTS")
    print("=" * 130)

    print(f"\n  {'Config':80s} | {'2020':>12s} | {'2021':>12s} | {'2022':>12s} | {'2023':>12s} | {'2024':>12s} | {'2025':>12s} | {'Avg':>7s} | {'WfMDD':>6s}")
    print(f"  {'-'*80}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*7}-+-{'-'*6}")

    for desc, wf_res in wf_all.items():
        vals = []
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            if yr in wf_res:
                vals.append(f"{wf_res[yr]['ann']:+.0f}/{wf_res[yr]['mdd']:.0f}")
            else:
                vals.append("N/A")
        avg_ann = np.mean([r2['ann'] for r2 in wf_res.values()])
        worst_mdd = min(r2['mdd'] for r2 in wf_res.values())
        print(f"  {desc:80s} | {vals[0]:>12s} | {vals[1]:>12s} | {vals[2]:>12s} | {vals[3]:>12s} | {vals[4]:>12s} | {vals[5]:>12s} | {avg_ann:>+6.0f}% | {worst_mdd:>5.1f}%")

    # ===================== ABLATION SUMMARY =====================
    print("\n" + "=" * 130)
    print("  ABLATION SUMMARY — Signal contribution")
    print("=" * 130)

    ablation_lookup = {r['desc']: r for r in s5_results}
    ablation_order = [
        "ABLATION: Full baseline V121*3 OV*2 FF*1",
        "ABLATION: V121 only (no OV/ID no FF)",
        "ABLATION: V121+OV/ID (no FF)",
        "ABLATION: V121+FF (no OV/ID)",
        "ABLATION: OV/ID+FF (no V121)",
        "ABLATION: OV/ID only",
        "ABLATION: FF only",
    ]
    for label in ablation_order:
        if label in ablation_lookup:
            r = ablation_lookup[label]
            pr(r, label)

    # Contribution analysis
    if "ABLATION: Full baseline V121*3 OV*2 FF*1" in ablation_lookup:
        full = ablation_lookup["ABLATION: Full baseline V121*3 OV*2 FF*1"]
        v121_only = ablation_lookup.get("ABLATION: V121 only (no OV/ID no FF)", {})
        v121_ov = ablation_lookup.get("ABLATION: V121+OV/ID (no FF)", {})
        v121_ff = ablation_lookup.get("ABLATION: V121+FF (no OV/ID)", {})

        print(f"\n  Contribution Analysis (annual return):")
        print(f"    Full baseline: {full.get('ann', 0):+.1f}%")
        if v121_only:
            print(f"    V121 only:     {v121_only.get('ann', 0):+.1f}%")
            ov_contrib = v121_ov.get('ann', 0) - v121_only.get('ann', 0) if v121_ov else 0
            ff_contrib = v121_ff.get('ann', 0) - v121_only.get('ann', 0) if v121_ff else 0
            print(f"    OV/ID adds:    {ov_contrib:+.1f}% (V121+OV/ID vs V121 alone)")
            print(f"    FF adds:       {ff_contrib:+.1f}% (V121+FF vs V121 alone)")

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 130)
    print("  FINAL SUMMARY")
    print("=" * 130)

    print(f"\n  Best configs per section:")
    for sec_name, sec_results in [("S1: OV/ID Threshold", s1_results),
                                   ("S2: Signal Weights", s2_results),
                                   ("S3: OV/ID Formula", s3_results),
                                   ("S4: FinalFlag Threshold", s4_results),
                                   ("S5: Ablation", s5_results),
                                   ("S6: Combined", s6_results)]:
        sec_sorted = sorted(sec_results, key=lambda x: abs(x['ann']/x['mdd']) if x['mdd'] != 0 else 0, reverse=True)
        if sec_sorted:
            r = sec_sorted[0]
            desc = r.get('desc', '')
            ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
            print(f"  {sec_name:25s} best: {desc}")
            print(f"  {'':25s}       Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
