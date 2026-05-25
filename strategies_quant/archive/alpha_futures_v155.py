"""
Alpha Futures V155 — Regime-Adaptive Signal Parameters
=============================================================================
Goal: Instead of fixed signal thresholds, adapt them based on market regime.

Key ideas:
  1. Volatility regime: tight thresholds in low-vol, wider in high-vol
  2. Breadth regime: aggressive sizing when >60% trending up, reduce when <40%
  3. Trend vs mean-reversion: equity curve momentum to modulate exposure
  4. Correlation regime: reduce positions when cross-commodity corr is high

Signal base:
  V121: ROC5>1% AND Z>1.5 AND ROC improving. Score = ROC5*Z
  Union: V121(*3) + OV/ID(OV>0.3%+ID>0.3%+ROC>1%, *2) + FinalFlag(ROC20>5%+breakout, *1)

Sections:
  0: Baseline (fixed thresholds, V146-style)
  1: Vol-adaptive signal thresholds
  2: Breadth-adaptive sizing
  3: Cross-correlation regime filter
  4: Best combos with WF validation
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
CASH0 = 500000


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0: return -100.0
    return (final / initial) ** (1.0 / (n_days / 252)) * 100 - 100


def main():
    print("=" * 130)
    print("  V155 — Regime-Adaptive Signal Parameters")
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

    print(f"  Base indicators done ({time.time()-t0:.1f}s)")

    # ===================== REGIME INDICATORS (precompute) =====================
    print("  Computing regime indicators...", flush=True)
    t1 = time.time()

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
    VOL_P33 = np.percentile(valid_vols, 33) if len(valid_vols) > 0 else 0.5
    VOL_P66 = np.percentile(valid_vols, 66) if len(valid_vols) > 0 else 1.0
    VOL_MEDIAN = np.median(valid_vols) if len(valid_vols) > 0 else 1.0
    print(f"  Market vol: P33={VOL_P33:.4f}%  Median={VOL_MEDIAN:.4f}%  P66={VOL_P66:.4f}%")

    # Cross-sectional correlation: average pairwise corr of daily returns over rolling 20-day window
    # Use vectorized np.corrcoef on valid commodities for speed
    AVG_CORR = np.full(ND, np.nan)
    MAX_CORR_SAMPLE = 30  # limit number of commodities sampled
    for di in range(20, ND):
        rets_block = RET[:, di-20:di]  # (NS, 20)
        # Find commodities with enough valid returns
        valid_mask = np.array([np.sum(~np.isnan(rets_block[s])) >= 10 for s in range(NS)])
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) < 5:
            continue
        # Deterministic: take first MAX_CORR_SAMPLE
        if len(valid_idx) > MAX_CORR_SAMPLE:
            valid_idx = valid_idx[:MAX_CORR_SAMPLE]
        sub = rets_block[valid_idx].copy()
        # Replace NaN with column mean for correlation matrix computation
        for si in range(len(valid_idx)):
            row = sub[si]
            nans = np.isnan(row)
            if np.any(nans) and not np.all(nans):
                row[nans] = np.nanmean(row)
        # Compute full correlation matrix
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_mat = np.corrcoef(sub)
        # Extract upper triangle
        n = corr_mat.shape[0]
        upper = []
        for i in range(n):
            for j in range(i+1, n):
                v = corr_mat[i, j]
                if not np.isnan(v):
                    upper.append(v)
        if len(upper) >= 5:
            AVG_CORR[di] = np.mean(upper)

    valid_corrs = AVG_CORR[~np.isnan(AVG_CORR)]
    CORR_MEDIAN = np.median(valid_corrs) if len(valid_corrs) > 0 else 0.3
    CORR_P75 = np.percentile(valid_corrs, 75) if len(valid_corrs) > 0 else 0.5
    print(f"  Avg cross-corr: Median={CORR_MEDIAN:.3f}  P75={CORR_P75:.3f}")
    print(f"  Regime indicators done ({time.time()-t1:.1f}s)")
    print(f"  Total precompute ({time.time()-t0:.1f}s)")

    # ===================== SIGNAL DEFINITIONS =====================
    # Adaptive signal: roc_thresh, z_thresh can vary by regime
    def sig_v121(di, edi, roc_thresh=1.0, z_thresh=1.5):
        c = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= roc_thresh or zs <= z_thresh: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((roc * zs, s, ep, 'v121'))
        return c

    def sig_ov_id(di, edi, roc_thresh=1.0):
        c = []
        for s in range(NS):
            ov = OV_GAP[s, di]; idr = ID_RET[s, di]; roc = ROC5[s, di]
            if any(np.isnan(x) for x in [ov, idr, roc]): continue
            if ov <= 0.3 or idr <= 0.3 or roc <= roc_thresh: continue
            zs = ZSCORE[s, di]
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            z_bonus = zs if not np.isnan(zs) and zs > 1.0 else 1.0
            c.append(((ov + idr) * roc * z_bonus * 2, s, ep, 'ov_id'))
        return c

    def sig_final_flag(di, edi, roc20_thresh=5.0):
        c = []
        for s in range(NS):
            roc20 = ROC20[s, di]
            if np.isnan(roc20) or roc20 <= roc20_thresh or di < 6: continue
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

    def sig_union(di, edi, roc_thresh=1.0, z_thresh=1.5, roc20_thresh=5.0):
        all_sigs = {}
        for item in sig_v121(di, edi, roc_thresh=roc_thresh, z_thresh=z_thresh):
            sc, s, ep, st = item
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc * 3
            all_sigs[s][2].append('v121')
        for item in sig_ov_id(di, edi, roc_thresh=roc_thresh):
            sc, s, ep, st = item
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc * 2
            all_sigs[s][2].append('ov_id')
        for item in sig_final_flag(di, edi, roc20_thresh=roc20_thresh):
            sc, s, ep, st = item
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc
            all_sigs[s][2].append('ff')
        return [(sc, s, ep, '+'.join(sigs)) for s, (sc, ep, sigs) in all_sigs.items()]

    # ===================== HELPER: Correlation between two commodities =====================
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

    # ===================== UNIFIED BACKTEST ENGINE =====================
    def backtest_v155(start_di=MIN_TRAIN, end_di=None,
                      # Regime-adaptive signal thresholds
                      adapt_vol=False,       # vol-adaptive thresholds
                      adapt_breadth=False,   # breadth-adaptive sizing
                      adapt_corr=False,      # correlation regime filter
                      # Vol-adaptive thresholds: (low_roc, low_z), (med_roc, med_z), (high_roc, high_z)
                      vol_thresh_low=(0.5, 1.0),
                      vol_thresh_med=(1.0, 1.5),
                      vol_thresh_high=(2.0, 2.0),
                      # Breadth-adaptive sizing: breadth_high threshold, size_high; breadth_low, size_low; size_mid
                      breadth_high=0.60, size_at_high_breadth=0.70,
                      breadth_low=0.40, size_at_low_breadth=0.30,
                      size_at_mid_breadth=0.55,
                      # Correlation regime filter
                      corr_regime_thresh=0.45,  # avg pairwise corr threshold
                      corr_reduce_factor=0.5,   # multiply size by this when corr is high
                      # Base sizing / DD
                      base_size=0.55,
                      dd_tiers=None,
                      max_corr=0.5,
                      hold=1, top_n=2):
        if end_di is None: end_di = ND
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
                    if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                    m = MULT.get(p['sym'], DEF_MULT)
                    pnl = (ep - p['entry_price']) * m * p['lots']
                    inv = p['entry_price'] * m * abs(p['lots'])
                    pp = pnl / inv * 100 if inv > 0 else 0
                    cash += ep * m * abs(p['lots']) * (1 - COMM)
                    trades.append(pp)
                    cl.append(p)
            for p in cl: positions.remove(p)

            # --- Determine signal thresholds based on regime ---
            roc_thresh = 1.0
            z_thresh = 1.5
            roc20_thresh = 5.0

            if adapt_vol:
                vol = MKT_VOL[di]
                if not np.isnan(vol):
                    if vol < VOL_P33:
                        roc_thresh, z_thresh = vol_thresh_low
                    elif vol < VOL_P66:
                        roc_thresh, z_thresh = vol_thresh_med
                    else:
                        roc_thresh, z_thresh = vol_thresh_high

            # --- Determine position size based on regime ---
            pos_size = base_size

            if adapt_breadth:
                bth = BREADTH[di]
                if not np.isnan(bth):
                    if bth >= breadth_high:
                        pos_size = size_at_high_breadth
                    elif bth <= breadth_low:
                        pos_size = size_at_low_breadth
                    else:
                        pos_size = size_at_mid_breadth

            # Correlation regime: reduce when avg cross-corr is high
            if adapt_corr:
                ac = AVG_CORR[di]
                if not np.isnan(ac) and ac > corr_regime_thresh:
                    pos_size *= corr_reduce_factor

            # DD-based sizing on top of regime sizing
            dd_mult = dd_size(pv, high_water, dd_tiers)
            pos_size *= dd_mult

            # Clamp
            pos_size = max(0.05, min(0.95, pos_size))

            # --- Enter positions ---
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue

            held_si = set(p['si'] for p in positions)

            # Get V121 and Union signals with adaptive thresholds
            cands_v121 = sig_v121(di, edi, roc_thresh=roc_thresh, z_thresh=z_thresh)
            cands_union = sig_union(di, edi, roc_thresh=roc_thresh, z_thresh=z_thresh,
                                     roc20_thresh=roc20_thresh)
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

    # ===================== PRINTING HELPERS =====================
    def pr(r, label=""):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  {label:75s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d}")

    def walk_forward(label="", **kwargs):
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue
            r = backtest_v155(start_di=ys, end_di=ye, **kwargs)
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
    print("  SECTION 0: BASELINES (fixed thresholds, no regime adaptation)")
    print("=" * 130)

    baseline_configs = [
        {'base_size': 0.70, 'dd_tiers': [(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)],
         'label': "Baseline: DD70/60/40/20 base=70% corr<0.5"},
        {'base_size': 0.55, 'dd_tiers': [(0, 1.0)],
         'label': "Baseline: flat 55% no DD corr<0.5"},
        {'base_size': 0.55, 'dd_tiers': [(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)],
         'label': "Baseline: DD70/60/40/20 base=55% corr<0.5"},
    ]

    baseline_results = []
    for cfg in baseline_configs:
        r = backtest_v155(base_size=cfg['base_size'], dd_tiers=cfg['dd_tiers'])
        r['desc'] = cfg['label']
        baseline_results.append(r)
        pr(r, cfg['label'])

    # ===================== SECTION 1: VOL-ADAPTIVE SIGNAL THRESHOLDS =====================
    print("\n" + "=" * 130)
    print("  SECTION 1: VOL-ADAPTIVE SIGNAL THRESHOLDS")
    print("  Low vol (below P33): tighter thresholds -> more signals")
    print("  High vol (above P66): wider thresholds -> fewer but stronger signals")
    print(f"  Vol P33={VOL_P33:.4f}%  P66={VOL_P66:.4f}%")
    print("=" * 130)

    vol_configs = [
        # (vol_low_roc, vol_low_z, vol_med_roc, vol_med_z, vol_high_roc, vol_high_z, base_size, label)
        (0.5, 1.0, 1.0, 1.5, 2.0, 2.0, 0.55,
         "S1: Vol-adapt (0.5/1.0)-(1.0/1.5)-(2.0/2.0) DD70/60/40/20"),
        (0.3, 0.8, 1.0, 1.5, 1.5, 1.8, 0.55,
         "S1: Vol-adapt (0.3/0.8)-(1.0/1.5)-(1.5/1.8) DD70/60/40/20"),
        (0.5, 1.0, 1.0, 1.5, 2.5, 2.5, 0.55,
         "S1: Vol-adapt (0.5/1.0)-(1.0/1.5)-(2.5/2.5) DD70/60/40/20"),
        (0.5, 1.0, 1.0, 1.5, 2.0, 2.0, 0.70,
         "S1: Vol-adapt (0.5/1.0)-(1.0/1.5)-(2.0/2.0) base=70% DD70/60/40/20"),
        (0.5, 1.0, 1.0, 1.5, 2.0, 2.0, 0.50,
         "S1: Vol-adapt (0.5/1.0)-(1.0/1.5)-(2.0/2.0) base=50% DD70/60/40/20"),
        (0.8, 1.2, 1.0, 1.5, 1.5, 1.8, 0.55,
         "S1: Vol-adapt (0.8/1.2)-(1.0/1.5)-(1.5/1.8) DD70/60/40/20"),
        (0.5, 1.0, 1.0, 1.5, 3.0, 2.5, 0.55,
         "S1: Vol-adapt (0.5/1.0)-(1.0/1.5)-(3.0/2.5) DD70/60/40/20"),
    ]

    vol_results = []
    for vl_r, vl_z, vm_r, vm_z, vh_r, vh_z, bs, label in vol_configs:
        r = backtest_v155(adapt_vol=True,
                          vol_thresh_low=(vl_r, vl_z),
                          vol_thresh_med=(vm_r, vm_z),
                          vol_thresh_high=(vh_r, vh_z),
                          base_size=bs)
        r['desc'] = label
        vol_results.append(r)
        pr(r, label)

    # ===================== SECTION 2: BREADTH-ADAPTIVE SIZING =====================
    print("\n" + "=" * 130)
    print("  SECTION 2: BREADTH-ADAPTIVE SIZING")
    print("  When >60% of commodities trending up -> aggressive sizing")
    print("  When <40% -> reduce sizing")
    print("=" * 130)

    breadth_configs = [
        # (bth_high, sz_high, bth_low, sz_low, sz_mid, base, label)
        (0.60, 0.70, 0.40, 0.30, 0.55, 0.55,
         "S2: Bth>60%->70% <40%->30% mid=55% DD70/60/40/20"),
        (0.60, 0.75, 0.40, 0.25, 0.55, 0.55,
         "S2: Bth>60%->75% <40%->25% mid=55% DD70/60/40/20"),
        (0.55, 0.70, 0.45, 0.35, 0.55, 0.55,
         "S2: Bth>55%->70% <45%->35% mid=55% DD70/60/40/20"),
        (0.60, 0.70, 0.40, 0.30, 0.55, 0.70,
         "S2: Bth>60%->70% <40%->30% mid=55% base=70% DD70/60/40/20"),
        (0.65, 0.75, 0.35, 0.25, 0.55, 0.55,
         "S2: Bth>65%->75% <35%->25% mid=55% DD70/60/40/20"),
        (0.60, 0.65, 0.40, 0.35, 0.50, 0.55,
         "S2: Bth>60%->65% <40%->35% mid=50% DD70/60/40/20"),
        (0.60, 0.70, 0.40, 0.30, 0.55, 0.50,
         "S2: Bth>60%->70% <40%->30% mid=55% base=50% DD70/60/40/20"),
    ]

    breadth_results = []
    for bh, sh, bl, sl, sm, bs, label in breadth_configs:
        r = backtest_v155(adapt_breadth=True,
                          breadth_high=bh, size_at_high_breadth=sh,
                          breadth_low=bl, size_at_low_breadth=sl,
                          size_at_mid_breadth=sm,
                          base_size=bs)
        r['desc'] = label
        breadth_results.append(r)
        pr(r, label)

    # ===================== SECTION 3: CROSS-CORRELATION REGIME FILTER =====================
    print("\n" + "=" * 130)
    print("  SECTION 3: CROSS-CORRELATION REGIME FILTER")
    print(f"  Avg cross-corr median={CORR_MEDIAN:.3f}  P75={CORR_P75:.3f}")
    print("  When avg pairwise corr > threshold -> reduce position size")
    print("=" * 130)

    corr_configs = [
        # (corr_thresh, reduce_factor, base_size, label)
        (0.45, 0.5, 0.55, "S3: Corr>0.45 -> *0.5 DD70/60/40/20"),
        (0.45, 0.6, 0.55, "S3: Corr>0.45 -> *0.6 DD70/60/40/20"),
        (0.40, 0.5, 0.55, "S3: Corr>0.40 -> *0.5 DD70/60/40/20"),
        (0.50, 0.5, 0.55, "S3: Corr>0.50 -> *0.5 DD70/60/40/20"),
        (0.45, 0.3, 0.55, "S3: Corr>0.45 -> *0.3 DD70/60/40/20"),
        (0.45, 0.5, 0.70, "S3: Corr>0.45 -> *0.5 base=70% DD70/60/40/20"),
        (0.45, 0.7, 0.55, "S3: Corr>0.45 -> *0.7 DD70/60/40/20"),
        (0.35, 0.5, 0.55, "S3: Corr>0.35 -> *0.5 DD70/60/40/20"),
    ]

    corr_results = []
    for ct, rf, bs, label in corr_configs:
        r = backtest_v155(adapt_corr=True,
                          corr_regime_thresh=ct,
                          corr_reduce_factor=rf,
                          base_size=bs)
        r['desc'] = label
        corr_results.append(r)
        pr(r, label)

    # ===================== SECTION 4: BEST COMBOS WITH WF VALIDATION =====================
    print("\n" + "=" * 130)
    print("  SECTION 4: BEST COMBOS WITH WALK-FORWARD VALIDATION")
    print("=" * 130)

    # Combine the best ideas from Sections 1-3
    combo_configs = [
        # Vol-adaptive + Breadth-adaptive
        {'adapt_vol': True, 'adapt_breadth': True, 'adapt_corr': False,
         'vol_thresh_low': (0.5, 1.0), 'vol_thresh_med': (1.0, 1.5), 'vol_thresh_high': (2.0, 2.0),
         'breadth_high': 0.60, 'size_at_high_breadth': 0.70,
         'breadth_low': 0.40, 'size_at_low_breadth': 0.30, 'size_at_mid_breadth': 0.55,
         'base_size': 0.55,
         'label': "Combo: Vol+Breadth (0.5/1.0)-(1.0/1.5)-(2.0/2.0) Bth60/40->70/30%"},
        # Vol-adaptive + Corr regime
        {'adapt_vol': True, 'adapt_breadth': False, 'adapt_corr': True,
         'vol_thresh_low': (0.5, 1.0), 'vol_thresh_med': (1.0, 1.5), 'vol_thresh_high': (2.0, 2.0),
         'corr_regime_thresh': 0.45, 'corr_reduce_factor': 0.5,
         'base_size': 0.55,
         'label': "Combo: Vol+Corr (0.5/1.0)-(1.0/1.5)-(2.0/2.0) Corr>0.45->*0.5"},
        # Breadth + Corr
        {'adapt_vol': False, 'adapt_breadth': True, 'adapt_corr': True,
         'breadth_high': 0.60, 'size_at_high_breadth': 0.70,
         'breadth_low': 0.40, 'size_at_low_breadth': 0.30, 'size_at_mid_breadth': 0.55,
         'corr_regime_thresh': 0.45, 'corr_reduce_factor': 0.5,
         'base_size': 0.55,
         'label': "Combo: Breadth+Corr Bth60/40->70/30% Corr>0.45->*0.5"},
        # All three: Vol + Breadth + Corr
        {'adapt_vol': True, 'adapt_breadth': True, 'adapt_corr': True,
         'vol_thresh_low': (0.5, 1.0), 'vol_thresh_med': (1.0, 1.5), 'vol_thresh_high': (2.0, 2.0),
         'breadth_high': 0.60, 'size_at_high_breadth': 0.70,
         'breadth_low': 0.40, 'size_at_low_breadth': 0.30, 'size_at_mid_breadth': 0.55,
         'corr_regime_thresh': 0.45, 'corr_reduce_factor': 0.5,
         'base_size': 0.55,
         'label': "Combo: All Vol+Breadth+Corr (0.5/1.0)-(1.0/1.5)-(2.0/2.0) Bth60/40 Corr>0.45"},
        # All three, tighter vol
        {'adapt_vol': True, 'adapt_breadth': True, 'adapt_corr': True,
         'vol_thresh_low': (0.3, 0.8), 'vol_thresh_med': (1.0, 1.5), 'vol_thresh_high': (1.5, 1.8),
         'breadth_high': 0.60, 'size_at_high_breadth': 0.70,
         'breadth_low': 0.40, 'size_at_low_breadth': 0.30, 'size_at_mid_breadth': 0.55,
         'corr_regime_thresh': 0.45, 'corr_reduce_factor': 0.5,
         'base_size': 0.55,
         'label': "Combo: All tight (0.3/0.8)-(1.0/1.5)-(1.5/1.8) Bth60/40 Corr>0.45"},
        # All three, wider vol
        {'adapt_vol': True, 'adapt_breadth': True, 'adapt_corr': True,
         'vol_thresh_low': (0.5, 1.0), 'vol_thresh_med': (1.0, 1.5), 'vol_thresh_high': (3.0, 2.5),
         'breadth_high': 0.60, 'size_at_high_breadth': 0.70,
         'breadth_low': 0.40, 'size_at_low_breadth': 0.30, 'size_at_mid_breadth': 0.55,
         'corr_regime_thresh': 0.45, 'corr_reduce_factor': 0.5,
         'base_size': 0.55,
         'label': "Combo: All wide (0.5/1.0)-(1.0/1.5)-(3.0/2.5) Bth60/40 Corr>0.45"},
        # All three, aggressive base
        {'adapt_vol': True, 'adapt_breadth': True, 'adapt_corr': True,
         'vol_thresh_low': (0.5, 1.0), 'vol_thresh_med': (1.0, 1.5), 'vol_thresh_high': (2.0, 2.0),
         'breadth_high': 0.60, 'size_at_high_breadth': 0.75,
         'breadth_low': 0.40, 'size_at_low_breadth': 0.25, 'size_at_mid_breadth': 0.55,
         'corr_regime_thresh': 0.45, 'corr_reduce_factor': 0.5,
         'base_size': 0.55,
         'label': "Combo: All Vol+Bth+Corr aggr Bth75/25 Corr>0.45->*0.5"},
        # All three, mild corr reduction
        {'adapt_vol': True, 'adapt_breadth': True, 'adapt_corr': True,
         'vol_thresh_low': (0.5, 1.0), 'vol_thresh_med': (1.0, 1.5), 'vol_thresh_high': (2.0, 2.0),
         'breadth_high': 0.60, 'size_at_high_breadth': 0.70,
         'breadth_low': 0.40, 'size_at_low_breadth': 0.30, 'size_at_mid_breadth': 0.55,
         'corr_regime_thresh': 0.50, 'corr_reduce_factor': 0.6,
         'base_size': 0.55,
         'label': "Combo: All Vol+Bth+Corr mild Corr>0.50->*0.6"},
        # All three, moderate thresholds
        {'adapt_vol': True, 'adapt_breadth': True, 'adapt_corr': True,
         'vol_thresh_low': (0.5, 1.0), 'vol_thresh_med': (1.0, 1.5), 'vol_thresh_high': (2.0, 2.0),
         'breadth_high': 0.55, 'size_at_high_breadth': 0.65,
         'breadth_low': 0.45, 'size_at_low_breadth': 0.35, 'size_at_mid_breadth': 0.50,
         'corr_regime_thresh': 0.45, 'corr_reduce_factor': 0.5,
         'base_size': 0.55,
         'label': "Combo: All moderate Bth55/45->65/35% mid=50 Corr>0.45->*0.5"},
        # All three, tight corr
        {'adapt_vol': True, 'adapt_breadth': True, 'adapt_corr': True,
         'vol_thresh_low': (0.5, 1.0), 'vol_thresh_med': (1.0, 1.5), 'vol_thresh_high': (2.0, 2.0),
         'breadth_high': 0.60, 'size_at_high_breadth': 0.70,
         'breadth_low': 0.40, 'size_at_low_breadth': 0.30, 'size_at_mid_breadth': 0.55,
         'corr_regime_thresh': 0.40, 'corr_reduce_factor': 0.5,
         'base_size': 0.55,
         'label': "Combo: All Vol+Bth tight Corr>0.40->*0.5"},
    ]

    combo_results = []
    for cfg in combo_configs:
        label = cfg.pop('label')
        r = backtest_v155(**cfg)
        r['desc'] = label
        # Store the kwargs for WF
        r['wf_kwargs'] = cfg
        combo_results.append(r)
        pr(r, label)

    # ===================== COMPREHENSIVE RANKING =====================
    print("\n" + "=" * 130)
    print("  COMPREHENSIVE RANKING (all sections)")
    print("=" * 130)

    all_results = (baseline_results + vol_results + breadth_results +
                   corr_results + combo_results)
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

    # ===================== WALK-FORWARD FOR TOP CONFIGS =====================
    print("\n" + "=" * 130)
    print("  WALK-FORWARD VALIDATION FOR TOP CONFIGS")
    print("=" * 130)

    # Select top 15 unique configs by R/M ratio for WF
    seen = set()
    wf_configs = []
    for r, ratio in all_with_ratio:
        desc = r.get('desc', '')
        if desc not in seen:
            seen.add(desc)
            wf_configs.append(r)
        if len(wf_configs) >= 15:
            break

    # Also ensure at least 2 from each section
    for section_results in [vol_results, breadth_results, corr_results]:
        section_sorted = sorted(section_results,
                                key=lambda x: abs(x['ann']/x['mdd']) if x['mdd'] != 0 else 0,
                                reverse=True)
        for r in section_sorted[:2]:
            desc = r.get('desc', '')
            if desc not in seen:
                seen.add(desc)
                wf_configs.append(r)

    wf_all = {}
    for r in wf_configs:
        desc = r.get('desc', '')
        kwargs = r.get('wf_kwargs', None)

        # Build kwargs from result if not stored
        if kwargs is None:
            # Determine from section prefix
            if desc.startswith('S1:'):
                # Parse vol-adaptive config from the label
                kwargs = {
                    'adapt_vol': True,
                    'base_size': 0.55,
                }
                # Parse thresholds from label pattern like (0.5/1.0)-(1.0/1.5)-(2.0/2.0)
                import re
                m = re.findall(r'\(([^)]+)\)', desc)
                if len(m) == 3:
                    lr, lz = m[0].split('/'); mr, mz = m[1].split('/'); hr, hz = m[2].split('/')
                    kwargs['vol_thresh_low'] = (float(lr), float(lz))
                    kwargs['vol_thresh_med'] = (float(mr), float(mz))
                    kwargs['vol_thresh_high'] = (float(hr), float(hz))
                if 'base=70' in desc: kwargs['base_size'] = 0.70
                elif 'base=50' in desc: kwargs['base_size'] = 0.50
            elif desc.startswith('S2:'):
                kwargs = {'adapt_breadth': True, 'base_size': 0.55}
                import re
                # Parse Bth thresholds
                bth_match = re.findall(r'Bth>(\d+)%->(\d+)%\s+<(\d+)%->(\d+)%\s+mid=(\d+)%', desc)
                if bth_match:
                    bh, sh, bl, sl, sm = bth_match[0]
                    kwargs['breadth_high'] = int(bh) / 100
                    kwargs['size_at_high_breadth'] = int(sh) / 100
                    kwargs['breadth_low'] = int(bl) / 100
                    kwargs['size_at_low_breadth'] = int(sl) / 100
                    kwargs['size_at_mid_breadth'] = int(sm) / 100
                if 'base=70' in desc: kwargs['base_size'] = 0.70
                elif 'base=50' in desc: kwargs['base_size'] = 0.50
            elif desc.startswith('S3:'):
                kwargs = {'adapt_corr': True, 'base_size': 0.55}
                import re
                corr_match = re.search(r'Corr>([\d.]+)\s+->\s+\*([\d.]+)', desc)
                if corr_match:
                    kwargs['corr_regime_thresh'] = float(corr_match.group(1))
                    kwargs['corr_reduce_factor'] = float(corr_match.group(2))
                if 'base=70' in desc: kwargs['base_size'] = 0.70
            else:
                # Baseline
                if 'DD70/60/40/20' in desc:
                    kwargs = {'base_size': 0.70 if 'base=70' in desc else 0.55}
                else:
                    kwargs = {'base_size': 0.55}

        wf_res = walk_forward(label=desc, **kwargs)
        wf_all[desc] = wf_res
        print_wf(wf_res, desc)

    # ===================== TOP 3 BY WF AVG (MDD > -30%) =====================
    print("\n" + "=" * 130)
    print("  TOP 3 CONFIGS BY WF AVG (Worst WF MDD > -30%)")
    print("=" * 130)

    wf_summary = []
    for desc, wf_res in wf_all.items():
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        pos_years = sum(1 for r in wf_res.values() if r['ann'] > 0)
        avg_sharpe = np.mean([r['sharpe'] for r in wf_res.values()])
        wf_summary.append((desc, avg_ann, worst_mdd, pos_years, avg_sharpe, wf_res))

    # Filter by MDD > -30%
    filtered = [(d, a, m, p, s, w) for d, a, m, p, s, w in wf_summary if m > -30]
    filtered.sort(key=lambda x: -x[1])

    if filtered:
        for i, (desc, avg_ann, worst_mdd, pos_years, avg_sharpe, wf_res) in enumerate(filtered[:3]):
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(wf_res.items())])
            print(f"\n  #{i+1}: {desc}")
            print(f"       AvgWF={avg_ann:>+7.0f}% | WorstWfMDD={worst_mdd:>5.1f}% | AvgSh={avg_sharpe:.2f} | {pos_years}/6 positive")
            print(f"       {ws}")
    else:
        print("\n  No configs meet worst WF MDD > -30% threshold.")
        print("  Showing top 3 by avg WF regardless:")
        wf_summary.sort(key=lambda x: -x[1])
        for i, (desc, avg_ann, worst_mdd, pos_years, avg_sharpe, wf_res) in enumerate(wf_summary[:3]):
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(wf_res.items())])
            print(f"\n  #{i+1}: {desc}")
            print(f"       AvgWF={avg_ann:>+7.0f}% | WorstWfMDD={worst_mdd:>5.1f}% | AvgSh={avg_sharpe:.2f} | {pos_years}/6 positive")
            print(f"       {ws}")

    # ===================== DETAILED WF TABLE =====================
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

    print(f"\n  Best by section:")
    for name, sec_results in [("Baseline", baseline_results),
                               ("S1: Vol-adaptive", vol_results),
                               ("S2: Breadth-adaptive", breadth_results),
                               ("S3: Corr regime", corr_results),
                               ("Combo", combo_results)]:
        sec_sorted = sorted(sec_results,
                            key=lambda x: abs(x['ann']/x['mdd']) if x['mdd'] != 0 else 0,
                            reverse=True)
        if sec_sorted:
            best = sec_sorted[0]
            desc = best.get('desc', '')
            ratio = abs(best['ann'] / best['mdd']) if best['mdd'] != 0 else 0
            print(f"  {name:25s}: {desc:50s} | Ann={best['ann']:+8.1f}% | MDD={best['mdd']:6.1f}% | R/M={ratio:.2f}")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
