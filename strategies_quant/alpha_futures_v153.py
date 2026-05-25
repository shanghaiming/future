"""
Alpha Futures V153 — Multi-Timeframe Signal Confirmation
=============================================================================
Goal: Explore whether confirming V121/Union signals with longer timeframes
      improves signal quality.

Section 0: Baseline (V146 champion reproduction)
Section 1: Multi-TF consensus (ROC5+ROC10+ROC20 all positive)
Section 2: Trend acceleration (ROC5 > ROC10 > ROC20)
Section 3: LINREG R-squared filter (trend quality)
Section 4: Counter-trend filter (ROC20 > 0 required)
Section 5: Best combos with WF validation
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
CASH0_VAL = 500000

def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0: return -100.0
    return (final / initial) ** (1.0 / (n_days / 252)) * 100 - 100


def main():
    print("=" * 130)
    print("  V153 — Multi-Timeframe Signal Confirmation")
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

    # Daily returns for correlation (NOT overlapping 20-day)
    # RET already has daily returns computed above

    # ===================== LINREG R-squared for trend quality =====================
    # Precompute R^2 for 5, 10, 20 day linear regression slopes
    LR_SLOPE5 = np.full((NS, ND), np.nan)
    LR_SLOPE10 = np.full((NS, ND), np.nan)
    LR_SLOPE20 = np.full((NS, ND), np.nan)
    LR_RSQ5 = np.full((NS, ND), np.nan)
    LR_RSQ10 = np.full((NS, ND), np.nan)
    LR_RSQ20 = np.full((NS, ND), np.nan)

    for si in range(NS):
        c = C[si].astype(np.float64)
        for period, slope_arr, rsq_arr in [(5, LR_SLOPE5, LR_RSQ5),
                                            (10, LR_SLOPE10, LR_RSQ10),
                                            (20, LR_SLOPE20, LR_RSQ20)]:
            x = np.arange(period, dtype=np.float64)
            x_mean = np.mean(x)
            x_ss = np.sum((x - x_mean) ** 2)
            for di in range(period, ND):
                y = c[di-period:di]
                if any(np.isnan(v) for v in y): continue
                y_mean = np.mean(y)
                y_ss = np.sum((y - y_mean) ** 2)
                if y_ss == 0: continue
                xy = np.sum((x - x_mean) * (y - y_mean))
                slope = xy / x_ss if x_ss > 0 else 0
                # R^2 from regression
                y_hat = y_mean + slope * (x - x_mean)
                ss_res = np.sum((y - y_hat) ** 2)
                rsq = 1.0 - ss_res / y_ss if y_ss > 0 else 0.0
                # Normalize slope to % per day
                avg_price = y_mean
                slope_arr[si, di] = slope / avg_price * 100 if avg_price > 0 else 0
                rsq_arr[si, di] = rsq

    print(f"  LINREG R-squared computed for 5/10/20-day windows", flush=True)
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
        """Use daily returns instead of overlapping 20-day returns."""
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
    def backtest_v153(start_di=MIN_TRAIN, end_di=None,
                      # Multi-TF filter mode
                      mtf_filter='none',
                      # 'none': no filter (baseline)
                      # 'all_positive': require ROC5>0, ROC10>0, ROC20>0
                      # 'acceleration': require ROC5 > ROC10 > ROC20
                      # 'rsq': require average R^2 > threshold
                      # 'counter_trend': require ROC20 > 0
                      # 'all_positive_score': all positive + score boost
                      # 'accel_score': acceleration + score boost
                      # 'rsq_strict': higher R^2 threshold
                      # 'combo_1': all_positive + counter_trend (redundant but explicit)
                      # 'combo_2': acceleration + rsq filter
                      # 'combo_3': all_positive + rsq filter
                      # R-squared threshold
                      rsq_threshold=0.3,
                      # Score modifier: multiply score when filter passes
                      score_boost=1.0,
                      # Sizing: DD tiers
                      dd_tiers=None,
                      # General params
                      hold=1, top_n=2, max_corr=0.5):
        if end_di is None: end_di = ND
        if dd_tiers is None:
            dd_tiers = [(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)]

        cash = float(CASH0_VAL)
        positions = []
        trades = []
        daily_eq = []
        high_water = float(CASH0_VAL)

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

            # Position sizing from DD tiers
            pos_size = dd_size(pv, high_water, dd_tiers)
            pos_size = max(0.05, min(0.95, pos_size))

            # Enter positions
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue

            held_si = set(p['si'] for p in positions)

            # Get best V121 and best Union signal
            cands_v121 = sig_v121(di, edi)
            cands_union = sig_union(di, edi)

            # Apply multi-TF filters to candidates
            def apply_mtf_filter(candidates, di):
                """Filter and optionally boost scores based on multi-TF criteria."""
                filtered = []
                for sc, s, ep, sig_str in candidates:
                    r5 = ROC5[s, di]
                    r10 = ROC10[s, di]
                    r20 = ROC20[s, di]
                    rsq5 = LR_RSQ5[s, di]
                    rsq10 = LR_RSQ10[s, di]
                    rsq20 = LR_RSQ20[s, di]

                    if mtf_filter == 'none':
                        filtered.append((sc * score_boost, s, ep, sig_str))

                    elif mtf_filter == 'all_positive':
                        # Require all three ROCs positive
                        if (not np.isnan(r5) and r5 > 0 and
                            not np.isnan(r10) and r10 > 0 and
                            not np.isnan(r20) and r20 > 0):
                            filtered.append((sc * score_boost, s, ep, sig_str))

                    elif mtf_filter == 'acceleration':
                        # Require ROC5 > ROC10 > ROC20 (accelerating momentum)
                        if (not np.isnan(r5) and not np.isnan(r10) and not np.isnan(r20) and
                            r5 > r10 > r20):
                            filtered.append((sc * score_boost, s, ep, sig_str))

                    elif mtf_filter == 'rsq':
                        # Require average R^2 above threshold (clean trend)
                        avg_rsq = 0; n_rsq = 0
                        for rq in [rsq5, rsq10, rsq20]:
                            if not np.isnan(rq):
                                avg_rsq += rq; n_rsq += 1
                        if n_rsq >= 2 and avg_rsq / n_rsq >= rsq_threshold:
                            filtered.append((sc * score_boost, s, ep, sig_str))

                    elif mtf_filter == 'counter_trend':
                        # Require ROC20 > 0 (not buying into downtrend)
                        if not np.isnan(r20) and r20 > 0:
                            filtered.append((sc * score_boost, s, ep, sig_str))

                    elif mtf_filter == 'all_positive_score':
                        # All positive: boost score, don't filter
                        if (not np.isnan(r5) and r5 > 0 and
                            not np.isnan(r10) and r10 > 0 and
                            not np.isnan(r20) and r20 > 0):
                            filtered.append((sc * score_boost, s, ep, sig_str))
                        else:
                            filtered.append((sc * 0.5, s, ep, sig_str))  # penalty

                    elif mtf_filter == 'accel_score':
                        # Acceleration: boost score, don't filter
                        if (not np.isnan(r5) and not np.isnan(r10) and not np.isnan(r20) and
                            r5 > r10 > r20):
                            filtered.append((sc * score_boost, s, ep, sig_str))
                        else:
                            filtered.append((sc * 0.5, s, ep, sig_str))

                    elif mtf_filter == 'rsq_strict':
                        # Higher R^2 threshold
                        avg_rsq = 0; n_rsq = 0
                        for rq in [rsq5, rsq10, rsq20]:
                            if not np.isnan(rq):
                                avg_rsq += rq; n_rsq += 1
                        if n_rsq >= 2 and avg_rsq / n_rsq >= rsq_threshold:
                            filtered.append((sc * score_boost, s, ep, sig_str))

                    elif mtf_filter == 'combo_1':
                        # all_positive + counter_trend (redundant but explicit)
                        if (not np.isnan(r5) and r5 > 0 and
                            not np.isnan(r10) and r10 > 0 and
                            not np.isnan(r20) and r20 > 0):
                            filtered.append((sc * score_boost, s, ep, sig_str))

                    elif mtf_filter == 'combo_2':
                        # acceleration + rsq filter
                        if (not np.isnan(r5) and not np.isnan(r10) and not np.isnan(r20) and
                            r5 > r10 > r20):
                            avg_rsq = 0; n_rsq = 0
                            for rq in [rsq5, rsq10, rsq20]:
                                if not np.isnan(rq):
                                    avg_rsq += rq; n_rsq += 1
                            if n_rsq >= 2 and avg_rsq / n_rsq >= rsq_threshold:
                                filtered.append((sc * score_boost, s, ep, sig_str))

                    elif mtf_filter == 'combo_3':
                        # all_positive + rsq filter
                        if (not np.isnan(r5) and r5 > 0 and
                            not np.isnan(r10) and r10 > 0 and
                            not np.isnan(r20) and r20 > 0):
                            avg_rsq = 0; n_rsq = 0
                            for rq in [rsq5, rsq10, rsq20]:
                                if not np.isnan(rq):
                                    avg_rsq += rq; n_rsq += 1
                            if n_rsq >= 2 and avg_rsq / n_rsq >= rsq_threshold:
                                filtered.append((sc * score_boost, s, ep, sig_str))

                return filtered

            cands_v121_f = apply_mtf_filter(cands_v121, di)
            cands_union_f = apply_mtf_filter(cands_union, di)

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

            # CRITICAL: snapshot cash before entry loop
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
        ann = annual_return(cash, CASH0_VAL, nd)
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
        print(f"  {label:80s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d}")

    def walk_forward(mtf_filter='none', rsq_threshold=0.3, score_boost=1.0,
                     dd_tiers=None, hold=1, top_n=2, max_corr=0.5, label=""):
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue
            r = backtest_v153(start_di=ys, end_di=ye,
                              mtf_filter=mtf_filter,
                              rsq_threshold=rsq_threshold,
                              score_boost=score_boost,
                              dd_tiers=dd_tiers,
                              hold=hold, top_n=top_n,
                              max_corr=max_corr)
            res[yr] = r
        return res

    def print_wf(wf_res, label=""):
        pos = sum(1 for r in wf_res.values() if r['ann'] > 0)
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"    {label:80s}")
        print(f"      {pos}/6 pos | Avg={avg_ann:>+7.0f}% | WorstWfMDD={worst_mdd:>5.0f}%")
        print(f"      {ws}")

    # Default DD tiers
    DD_Tiers = [(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)]

    # ===================== SECTION 0: BASELINE =====================
    print("\n" + "=" * 130)
    print("  SECTION 0: BASELINE (V146 champion reproduction)")
    print("  top_n=2, DD70/60/40/20 sizing, corr<0.5, hold=1")
    print("=" * 130)

    baseline = backtest_v153(mtf_filter='none', dd_tiers=DD_Tiers, max_corr=0.5)
    pr(baseline, "Baseline: DD70/60/40/20 corr<0.5 no MTF filter")

    # Also test baseline with different DD tiers
    for dd_t, lbl in [
        ([(0, 0.55), (0.10, 0.45), (0.20, 0.35), (0.30, 0.20)],
         "Baseline: DD55/45/35/20 corr<0.5"),
    ]:
        r = backtest_v153(mtf_filter='none', dd_tiers=dd_t, max_corr=0.5)
        pr(r, lbl)

    # ===================== SECTION 1: Multi-TF Consensus =====================
    print("\n" + "=" * 130)
    print("  SECTION 1: MULTI-TF CONSENSUS")
    print("  Require ROC5>0, ROC10>0, ROC20>0 for entry")
    print("=" * 130)

    # All configs use 6-tuple: (mtf, score_boost, rsq_threshold, dd_tiers, max_corr, label)
    s1_configs = [
        ('all_positive', 1.0, 0.3, DD_Tiers, 0.5, "S1: AllPositive score=1.0 DD70/60/40/20"),
        ('all_positive', 1.5, 0.3, DD_Tiers, 0.5, "S1: AllPositive score=1.5 DD70/60/40/20"),
        ('all_positive', 2.0, 0.3, DD_Tiers, 0.5, "S1: AllPositive score=2.0 DD70/60/40/20"),
        ('all_positive', 1.0, 0.3, [(0, 0.55), (0.10, 0.45), (0.20, 0.35), (0.30, 0.20)], 0.5,
         "S1: AllPositive score=1.0 DD55/45/35/20"),
        ('all_positive_score', 1.5, 0.3, DD_Tiers, 0.5,
         "S1: AllPos+ScorePenalty boost=1.5 DD70/60/40/20"),
        ('all_positive_score', 2.0, 0.3, DD_Tiers, 0.5,
         "S1: AllPos+ScorePenalty boost=2.0 DD70/60/40/20"),
    ]

    s1_results = []
    for mtf, sb, rt, dt, mc, lbl in s1_configs:
        r = backtest_v153(mtf_filter=mtf, score_boost=sb, rsq_threshold=rt, dd_tiers=dt, max_corr=mc)
        r['desc'] = lbl
        s1_results.append(r)
        pr(r, lbl)

    # ===================== SECTION 2: Trend Acceleration =====================
    print("\n" + "=" * 130)
    print("  SECTION 2: TREND ACCELERATION")
    print("  Require ROC5 > ROC10 > ROC20 (accelerating momentum)")
    print("=" * 130)

    s2_configs = [
        ('acceleration', 1.0, 0.3, DD_Tiers, 0.5, "S2: Accel score=1.0 DD70/60/40/20"),
        ('acceleration', 1.5, 0.3, DD_Tiers, 0.5, "S2: Accel score=1.5 DD70/60/40/20"),
        ('acceleration', 2.0, 0.3, DD_Tiers, 0.5, "S2: Accel score=2.0 DD70/60/40/20"),
        ('acceleration', 1.0, 0.3, [(0, 0.55), (0.10, 0.45), (0.20, 0.35), (0.30, 0.20)], 0.5,
         "S2: Accel score=1.0 DD55/45/35/20"),
        ('accel_score', 1.5, 0.3, DD_Tiers, 0.5,
         "S2: Accel+ScorePenalty boost=1.5 DD70/60/40/20"),
        ('accel_score', 2.0, 0.3, DD_Tiers, 0.5,
         "S2: Accel+ScorePenalty boost=2.0 DD70/60/40/20"),
    ]

    s2_results = []
    for mtf, sb, rt, dt, mc, lbl in s2_configs:
        r = backtest_v153(mtf_filter=mtf, score_boost=sb, rsq_threshold=rt, dd_tiers=dt, max_corr=mc)
        r['desc'] = lbl
        s2_results.append(r)
        pr(r, lbl)

    # ===================== SECTION 3: LINREG R-squared Filter =====================
    print("\n" + "=" * 130)
    print("  SECTION 3: LINREG R-squared FILTER (Trend Quality)")
    print("  Require average R^2 > threshold (5/10/20 day windows)")
    print("=" * 130)

    s3_configs = [
        ('rsq', 0.3, 1.0, DD_Tiers, 0.5, "S3: R^2>0.3 score=1.0 DD70/60/40/20"),
        ('rsq', 0.4, 1.0, DD_Tiers, 0.5, "S3: R^2>0.4 score=1.0 DD70/60/40/20"),
        ('rsq', 0.5, 1.0, DD_Tiers, 0.5, "S3: R^2>0.5 score=1.0 DD70/60/40/20"),
        ('rsq', 0.2, 1.0, DD_Tiers, 0.5, "S3: R^2>0.2 score=1.0 DD70/60/40/20"),
        ('rsq', 0.3, 1.5, DD_Tiers, 0.5, "S3: R^2>0.3 score=1.5 DD70/60/40/20"),
        ('rsq', 0.3, 2.0, DD_Tiers, 0.5, "S3: R^2>0.3 score=2.0 DD70/60/40/20"),
        ('rsq_strict', 0.6, 1.0, DD_Tiers, 0.5, "S3: R^2strict>0.6 DD70/60/40/20"),
        ('rsq_strict', 0.7, 1.0, DD_Tiers, 0.5, "S3: R^2strict>0.7 DD70/60/40/20"),
    ]

    s3_results = []
    for mtf, rt, sb, dt, mc, lbl in s3_configs:
        r = backtest_v153(mtf_filter=mtf, rsq_threshold=rt, score_boost=sb,
                          dd_tiers=dt, max_corr=mc)
        r['desc'] = lbl
        s3_results.append(r)
        pr(r, lbl)

    # ===================== SECTION 4: Counter-Trend Filter =====================
    print("\n" + "=" * 130)
    print("  SECTION 4: COUNTER-TREND FILTER")
    print("  Require ROC20 > 0 (avoid buying into downtrend)")
    print("=" * 130)

    s4_configs = [
        ('counter_trend', 1.0, 0.3, DD_Tiers, 0.5, "S4: ROC20>0 score=1.0 DD70/60/40/20"),
        ('counter_trend', 1.5, 0.3, DD_Tiers, 0.5, "S4: ROC20>0 score=1.5 DD70/60/40/20"),
        ('counter_trend', 2.0, 0.3, DD_Tiers, 0.5, "S4: ROC20>0 score=2.0 DD70/60/40/20"),
        ('counter_trend', 1.0, 0.3, [(0, 0.55), (0.10, 0.45), (0.20, 0.35), (0.30, 0.20)], 0.5,
         "S4: ROC20>0 score=1.0 DD55/45/35/20"),
        ('counter_trend', 1.0, 0.3, DD_Tiers, 0.6, "S4: ROC20>0 score=1.0 corr<0.6"),
        ('counter_trend', 1.0, 0.3, DD_Tiers, 0.4, "S4: ROC20>0 score=1.0 corr<0.4"),
    ]

    s4_results = []
    for mtf, sb, rt, dt, mc, lbl in s4_configs:
        r = backtest_v153(mtf_filter=mtf, score_boost=sb, rsq_threshold=rt, dd_tiers=dt, max_corr=mc)
        r['desc'] = lbl
        s4_results.append(r)
        pr(r, lbl)

    # ===================== SECTION 5: BEST COMBOS =====================
    print("\n" + "=" * 130)
    print("  SECTION 5: BEST COMBOS WITH WF VALIDATION")
    print("=" * 130)

    # Test combinations of best ideas from Sections 1-4
    s5_configs = [
        ('combo_1', 1.0, 0.3, DD_Tiers, 0.5, "S5: AllPos+Counter DD70/60/40/20"),
        ('combo_1', 1.5, 0.3, DD_Tiers, 0.5, "S5: AllPos+Counter boost=1.5 DD70/60/40/20"),
        ('combo_2', 1.0, 0.3, DD_Tiers, 0.5, "S5: Accel+R^2>0.3 DD70/60/40/20"),
        ('combo_2', 1.0, 0.4, DD_Tiers, 0.5, "S5: Accel+R^2>0.4 DD70/60/40/20"),
        ('combo_2', 1.5, 0.3, DD_Tiers, 0.5, "S5: Accel+R^2>0.3 boost=1.5 DD70/60/40/20"),
        ('combo_3', 1.0, 0.3, DD_Tiers, 0.5, "S5: AllPos+R^2>0.3 DD70/60/40/20"),
        ('combo_3', 1.0, 0.4, DD_Tiers, 0.5, "S5: AllPos+R^2>0.4 DD70/60/40/20"),
        ('combo_3', 1.5, 0.3, DD_Tiers, 0.5, "S5: AllPos+R^2>0.3 boost=1.5 DD70/60/40/20"),
        # Conservative sizing combos
        ('all_positive', 1.0, 0.3, [(0, 0.55), (0.10, 0.45), (0.20, 0.35), (0.30, 0.20)], 0.5,
         "S5: AllPos DD55/45/35/20"),
        ('counter_trend', 1.0, 0.3, [(0, 0.55), (0.10, 0.45), (0.20, 0.35), (0.30, 0.20)], 0.5,
         "S5: CounterTrend DD55/45/35/20"),
        ('combo_3', 1.0, 0.3, [(0, 0.55), (0.10, 0.45), (0.20, 0.35), (0.30, 0.20)], 0.5,
         "S5: AllPos+R^2>0.3 DD55/45/35/20"),
        # With score boost on combos
        ('combo_3', 2.0, 0.3, DD_Tiers, 0.5, "S5: AllPos+R^2>0.3 boost=2.0 DD70/60/40/20"),
        ('combo_1', 2.0, 0.3, DD_Tiers, 0.5, "S5: AllPos+Counter boost=2.0 DD70/60/40/20"),
    ]

    s5_results = []
    for mtf, sb, rt, dt, mc, lbl in s5_configs:
        r = backtest_v153(mtf_filter=mtf, score_boost=sb, rsq_threshold=rt,
                          dd_tiers=dt, max_corr=mc)
        r['desc'] = lbl
        s5_results.append(r)
        pr(r, lbl)

    # ===================== COMPREHENSIVE RANKING =====================
    print("\n" + "=" * 130)
    print("  COMPREHENSIVE RANKING (Full Period)")
    print("=" * 130)

    all_results = s1_results + s2_results + s3_results + s4_results + s5_results
    all_valid = [r for r in all_results if r.get('desc', '') and r['mdd'] > -80]

    # Sort by annual return
    all_valid.sort(key=lambda x: -x['ann'])
    print(f"\n  Top 15 by Annual Return:")
    for i, r in enumerate(all_valid[:15]):
        desc = r.get('desc', '')
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i+1:2d}: {desc:80s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    # Sort by return/MDD ratio
    all_with_ratio = [(r, abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0) for r in all_valid]
    all_with_ratio.sort(key=lambda x: -x[1])
    print(f"\n  Top 15 by Ann/MDD Ratio:")
    for i, (r, ratio) in enumerate(all_with_ratio[:15]):
        desc = r.get('desc', '')
        print(f"  #{i+1:2d}: {desc:80s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    # ===================== WALK-FORWARD FOR TOP CONFIGS =====================
    print("\n" + "=" * 130)
    print("  WALK-FORWARD VALIDATION FOR TOP CONFIGS")
    print("=" * 130)

    # Build parameter lookup from all tested configs (all 6-tuples now)
    all_tested = s1_configs + s2_configs + s3_configs + s4_configs + s5_configs


    # Create param map: desc -> (mtf, sb, rt, dt, mc)
    param_map = {}
    for mtf, sb, rt, dt, mc, lbl in all_tested:
        param_map[lbl] = (mtf, sb, rt, dt, mc)

    # Select top 15 by R/M ratio for WF
    seen = set()
    wf_selection = []
    for r, ratio in all_with_ratio:
        desc = r.get('desc', '')
        if desc not in seen and desc in param_map:
            seen.add(desc)
            wf_selection.append(r)
        if len(wf_selection) >= 15:
            break

    wf_all = {}
    for r in wf_selection:
        desc = r.get('desc', '')
        if desc not in param_map:
            continue
        mtf, sb, rt, dt, mc = param_map[desc]
        wf_res = walk_forward(mtf_filter=mtf, score_boost=sb,
                               rsq_threshold=rt, dd_tiers=dt,
                               max_corr=mc, label=desc)
        wf_all[desc] = wf_res
        print_wf(wf_res, desc)

    # ===================== HIGHLIGHT: BEST WF CONFIGS =====================
    print("\n" + "=" * 130)
    print("  HIGHLIGHT: TOP 3 Configs by WF Avg with WF MDD > -30%")
    print("=" * 130)

    wf_highlight = []
    for desc, wf_res in wf_all.items():
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        pos_years = sum(1 for r in wf_res.values() if r['ann'] > 0)
        if worst_mdd > -30:
            wf_highlight.append((desc, avg_ann, worst_mdd, pos_years, wf_res))

    if wf_highlight:
        wf_highlight.sort(key=lambda x: -x[1])
        for rank, (desc, avg_ann, worst_mdd, pos_years, wf_res) in enumerate(wf_highlight[:10], 1):
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(wf_res.items())])
            print(f"\n  #{rank}: {desc}")
            print(f"      AvgWF={avg_ann:+.0f}% | WorstWfMDD={worst_mdd:.1f}% | {pos_years}/6 positive")
            print(f"      {ws}")
    else:
        print("\n  No configs meet WF MDD > -30% threshold.")
        # Show closest
        closest = []
        for desc, wf_res in wf_all.items():
            avg_ann = np.mean([r['ann'] for r in wf_res.values()])
            worst_mdd = min(r['mdd'] for r in wf_res.values())
            closest.append((desc, avg_ann, worst_mdd))
        closest.sort(key=lambda x: -x[1])
        print(f"\n  Closest configs by avg WF return:")
        for desc, avg_ann, worst_mdd in closest[:10]:
            print(f"  {desc:80s} | AvgWF={avg_ann:>+7.0f}% | WorstWfMDD={worst_mdd:>5.1f}%")

    # ===================== DETAILED WF TABLE =====================
    print("\n" + "=" * 130)
    print("  DETAILED WF TABLE: ALL TESTED CONFIGS")
    print("=" * 130)

    hdr = f"  {'Config':80s} | {'2020':>12s} | {'2021':>12s} | {'2022':>12s} | {'2023':>12s} | {'2024':>12s} | {'2025':>12s} | {'Avg':>7s} | {'WfMDD':>6s}"
    print(f"\n{hdr}")
    print(f"  {'-'*80}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*7}-+-{'-'*6}")

    for desc, wf_res in wf_all.items():
        vals = []
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            if yr in wf_res:
                vals.append(f"{wf_res[yr]['ann']:+.0f}/{wf_res[yr]['mdd']:.0f}")
            else:
                vals.append("N/A")
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        print(f"  {desc:80s} | {vals[0]:>12s} | {vals[1]:>12s} | {vals[2]:>12s} | {vals[3]:>12s} | {vals[4]:>12s} | {vals[5]:>12s} | {avg_ann:>+6.0f}% | {worst_mdd:>5.1f}%")

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 130)
    print("  FINAL SUMMARY")
    print("=" * 130)

    print(f"\n  --- Section 1 (Multi-TF All Positive):")
    for r in sorted(s1_results, key=lambda x: abs(x['ann']/x['mdd']) if x['mdd'] != 0 else 0, reverse=True)[:3]:
        pr(r, r.get('desc', ''))

    print(f"\n  --- Section 2 (Trend Acceleration):")
    for r in sorted(s2_results, key=lambda x: abs(x['ann']/x['mdd']) if x['mdd'] != 0 else 0, reverse=True)[:3]:
        pr(r, r.get('desc', ''))

    print(f"\n  --- Section 3 (LINREG R-squared):")
    for r in sorted(s3_results, key=lambda x: abs(x['ann']/x['mdd']) if x['mdd'] != 0 else 0, reverse=True)[:3]:
        pr(r, r.get('desc', ''))

    print(f"\n  --- Section 4 (Counter-Trend):")
    for r in sorted(s4_results, key=lambda x: abs(x['ann']/x['mdd']) if x['mdd'] != 0 else 0, reverse=True)[:3]:
        pr(r, r.get('desc', ''))

    print(f"\n  --- Section 5 (Combos):")
    for r in sorted(s5_results, key=lambda x: abs(x['ann']/x['mdd']) if x['mdd'] != 0 else 0, reverse=True)[:3]:
        pr(r, r.get('desc', ''))

    print(f"\n  OVERALL TOP 5 (by full-period R/M ratio):")
    for i, (r, ratio) in enumerate(all_with_ratio[:5]):
        desc = r.get('desc', '')
        print(f"  #{i+1}: {desc:80s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
