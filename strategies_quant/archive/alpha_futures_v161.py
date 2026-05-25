"""
Alpha Futures V161 — Dynamic Rebalance Frequency
==============================================================================
All prior versions rebalance every day (hold=1). V161 tests whether changing
WHEN we enter positions (not HOW LONG we hold them) improves risk-adjusted returns.

Four rebalance mechanisms:
1. Signal strength gate — only enter when top signal score > min_signal_score
2. Conditional rebalance — only enter on days where candidate count >= min_candidates
3. Skip-day trading — only look for new entries every N days (hold=1 each position)
4. Signal quality gate — only enter when score exceeds historical percentile threshold

Key parameters swept:
- min_signal_score: [0, 5, 10, 20, 50]
- rebalance_freq: [1, 2, 3, 5]
- min_candidates: [0, 1, 2, 3]

Base config: top_n=3, aggro100 DD tiers, no SL, regime 0.5-1.5, max_corr=0.5,
             Cross+Corr mechanism (V121 + Union), Kitchen Sink sizing.
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
    print("  V161 — Dynamic Rebalance Frequency Exploration")
    print("  Tests signal strength gates, skip-day trading, conditional rebalance")
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
    print(f"  Market vol median: {VOL_MEDIAN:.4f}%")

    # ===================== PRE-COMPUTE HISTORICAL SCORE DISTRIBUTION =====================
    # For signal quality percentile gate — compute a running percentile baseline
    # We collect all signal scores across all symbols/days into a single distribution
    # and compute the 25th, 50th, 75th percentiles for the quality gate.
    print("  Computing historical score distribution for quality gate...", flush=True)

    def collect_all_scores():
        """Quick pass to collect all V121 and Union signal scores for percentile calc."""
        scores_v121 = []
        scores_union = []
        for di in range(MIN_TRAIN, ND - 1):
            edi = di + 1
            # V121 signals
            for s in range(NS):
                roc = ROC5[s, di]; zs = ZSCORE[s, di]
                if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5: continue
                rp = ROC5[s, di-1] if di > 0 else np.nan
                if not np.isnan(rp) and roc <= rp: continue
                ep = O[s, edi]
                if np.isnan(ep) or ep <= 0: continue
                scores_v121.append(roc * zs)
            # Union signals (approximate — just count candidates)
            union_sigs = {}
            for s in range(NS):
                roc = ROC5[s, di]; zs = ZSCORE[s, di]
                if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5: continue
                rp = ROC5[s, di-1] if di > 0 else np.nan
                if not np.isnan(rp) and roc <= rp: continue
                ep = O[s, edi]
                if np.isnan(ep) or ep <= 0: continue
                if s not in union_sigs: union_sigs[s] = 0
                union_sigs[s] += roc * zs * 3
            for s in range(NS):
                ov = OV_GAP[s, di]; idr = ID_RET[s, di]; roc = ROC5[s, di]
                if any(np.isnan(x) for x in [ov, idr, roc]): continue
                if ov <= 0.3 or idr <= 0.3 or roc <= 1.0: continue
                ep = O[s, edi]
                if np.isnan(ep) or ep <= 0: continue
                z_bonus = ZSCORE[s, di] if not np.isnan(ZSCORE[s, di]) and ZSCORE[s, di] > 1.0 else 1.0
                if s not in union_sigs: union_sigs[s] = 0
                union_sigs[s] += (ov + idr) * roc * z_bonus * 2 * 2
            scores_union.extend(union_sigs.values())
        return scores_v121, scores_union

    all_v121_scores, all_union_scores = collect_all_scores()
    if all_v121_scores:
        pct25_v121 = np.percentile(all_v121_scores, 25)
        pct50_v121 = np.percentile(all_v121_scores, 50)
        pct75_v121 = np.percentile(all_v121_scores, 75)
    else:
        pct25_v121 = pct50_v121 = pct75_v121 = 0
    if all_union_scores:
        pct25_union = np.percentile(all_union_scores, 25)
        pct50_union = np.percentile(all_union_scores, 50)
        pct75_union = np.percentile(all_union_scores, 75)
    else:
        pct25_union = pct50_union = pct75_union = 0

    print(f"  V121 score percentiles: 25th={pct25_v121:.1f} 50th={pct50_v121:.1f} 75th={pct75_v121:.1f}")
    print(f"  Union score percentiles: 25th={pct25_union:.1f} 50th={pct50_union:.1f} 75th={pct75_union:.1f}")
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

    # ===================== HELPER: Compute composite regime score =====================
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
        if wr > 0.65:
            return 1.3
        elif wr >= 0.50:
            return 1.0
        else:
            return 0.5

    # ===================== UNIFIED BACKTEST ENGINE =====================
    def backtest(start_di=MIN_TRAIN, end_di=None,
                 max_corr=0.5,
                 dd_tiers=None,
                 regime_lo=0.5, regime_hi=1.5,
                 top_n=3,
                 min_signal_score=0,
                 rebalance_freq=1,
                 min_candidates=0,
                 score_pct_threshold=0.0):
        """
        Extended backtest with dynamic rebalance controls.

        Parameters:
        -----------
        min_signal_score : float
            Only enter when the best signal score >= this threshold.
            0 = no filter (baseline).
        rebalance_freq : int
            Only look for new entries every N trading days.
            1 = every day (baseline), 2 = every other day, etc.
            Positions still held for 1 day each.
        min_candidates : int
            Only enter new positions when the number of valid signal
            candidates on that day >= this threshold.
            0 = no filter (baseline).
        score_pct_threshold : float
            Only enter when best signal score exceeds this historical
            percentile of scores (0-100). 0 = no filter.
            25 = top 75th percentile, 50 = top 50th percentile.
        """
        if end_di is None: end_di = ND
        if dd_tiers is None:
            dd_tiers = [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)]

        cash = float(CASH0)
        positions = []
        trades = []
        daily_eq = []
        high_water = float(CASH0)
        days_since_rebalance = 0

        # Precompute score percentile thresholds for quality gate
        v121_threshold = 0.0
        union_threshold = 0.0
        if score_pct_threshold > 0:
            if all_v121_scores:
                v121_threshold = np.percentile(all_v121_scores, score_pct_threshold)
            if all_union_scores:
                union_threshold = np.percentile(all_union_scores, score_pct_threshold)

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

            # Close positions past hold period (always close — hold=1)
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
            wr_mult_val = wr_size(trades, window=20)
            composite = compute_composite(di, daily_eq, high_water)
            regime_mult = regime_lo + composite * (regime_hi - regime_lo)
            pos_size = dd_sz * wr_mult_val * regime_mult
            pos_size = max(0.05, min(0.99, pos_size))

            # --- REBALANCE FREQUENCY GATE ---
            days_since_rebalance += 1
            if rebalance_freq > 1 and days_since_rebalance < rebalance_freq:
                continue

            # --- Enter positions ---
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue

            held_si = set(p['si'] for p in positions)

            # Get V121 and Union signals
            cands_v121 = sig_v121(di, edi)
            cands_union = sig_union(di, edi)
            cands_v121.sort(key=lambda x: -x[0])
            cands_union.sort(key=lambda x: -x[0])

            # --- MIN CANDIDATES GATE ---
            total_candidates = len(cands_v121) + len(cands_union)
            if total_candidates < min_candidates:
                days_since_rebalance = 0  # reset so we check again tomorrow
                continue

            # --- Build entries via Cross+Corr mechanism ---
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

            if not entries:
                days_since_rebalance = 0
                continue

            # --- SIGNAL STRENGTH GATE ---
            if min_signal_score > 0:
                top_score = max(e[0] for e in entries)
                if top_score < min_signal_score:
                    days_since_rebalance = 0
                    continue

            # --- SIGNAL QUALITY PERCENTILE GATE ---
            if score_pct_threshold > 0:
                top_score = max(e[0] for e in entries)
                # Use the lower threshold (union is usually higher score)
                effective_threshold = max(v121_threshold, union_threshold)
                # Actually: use the appropriate threshold based on signal type
                passes = False
                for sc, s, pr, sig_str, pct in entries:
                    if 'v121' in sig_str and sc >= v121_threshold:
                        passes = True; break
                    elif 'union' in sig_str and sc >= union_threshold:
                        passes = True; break
                    elif sc >= effective_threshold:
                        passes = True; break
                if not passes:
                    days_since_rebalance = 0
                    continue

            # Reset rebalance counter — we are entering this rebalance window
            days_since_rebalance = 0

            # --- Execute entries ---
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
                                  'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': 1,
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
        print(f"  {label:85s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d}")

    def walk_forward(max_corr=0.5, dd_tiers=None,
                     regime_lo=0.5, regime_hi=1.5, top_n=3,
                     min_signal_score=0, rebalance_freq=1,
                     min_candidates=0, score_pct_threshold=0.0,
                     label=""):
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue
            r = backtest(start_di=ys, end_di=ye,
                         max_corr=max_corr, dd_tiers=dd_tiers,
                         regime_lo=regime_lo, regime_hi=regime_hi,
                         top_n=top_n,
                         min_signal_score=min_signal_score,
                         rebalance_freq=rebalance_freq,
                         min_candidates=min_candidates,
                         score_pct_threshold=score_pct_threshold)
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

    # ===================== CONFIG DEFINITIONS =====================
    DD_TIERS = {
        'aggro100': [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)],
    }
    dd_base = DD_TIERS['aggro100']

    MIN_SIGNAL_SCORES = [0, 5, 10, 20, 50]
    REBALANCE_FREQS = [1, 2, 3, 5]
    MIN_CANDIDATES = [0, 1, 2, 3]
    SCORE_PCT_THRESHOLDS = [0, 25, 50, 75]   # percentile thresholds

    # ===================== SECTION 0: BASELINE =====================
    print("\n" + "=" * 120)
    print("  SECTION 0: BASELINE (daily rebalance, no gates, aggro100, top_n=3)")
    print("=" * 120)

    r_base = backtest(dd_tiers=dd_base, top_n=3)
    pr(r_base, "BASELINE: freq=1 minScore=0 minCand=0 pctGate=0")

    # ===================== SECTION 1: SIGNAL STRENGTH GATE =====================
    print("\n" + "=" * 120)
    print("  SECTION 1: MIN SIGNAL SCORE GATE")
    print("  Only enter when best signal score >= min_signal_score")
    print("  Base: aggro100, top_n=3, no SL, regime 0.5-1.5, max_corr=0.5")
    print("=" * 120)

    score_gate_results = []
    for mss in MIN_SIGNAL_SCORES:
        r = backtest(dd_tiers=dd_base, top_n=3, min_signal_score=mss)
        label = f"minScore={mss:3d}"
        pr(r, label)
        score_gate_results.append((r, mss, label))

    # ===================== SECTION 2: REBALANCE FREQUENCY =====================
    print("\n" + "=" * 120)
    print("  SECTION 2: SKIP-DAY REBALANCE FREQUENCY")
    print("  Only look for new entries every N days (hold=1 per position)")
    print("=" * 120)

    freq_results = []
    for freq in REBALANCE_FREQS:
        r = backtest(dd_tiers=dd_base, top_n=3, rebalance_freq=freq)
        label = f"freq={freq}"
        pr(r, label)
        freq_results.append((r, freq, label))

    # ===================== SECTION 3: MIN CANDIDATES GATE =====================
    print("\n" + "=" * 120)
    print("  SECTION 3: MIN CANDIDATES GATE")
    print("  Only enter when total signal candidates >= min_candidates")
    print("=" * 120)

    cand_results = []
    for mc in MIN_CANDIDATES:
        r = backtest(dd_tiers=dd_base, top_n=3, min_candidates=mc)
        label = f"minCand={mc}"
        pr(r, label)
        cand_results.append((r, mc, label))

    # ===================== SECTION 4: SIGNAL QUALITY PERCENTILE GATE =====================
    print("\n" + "=" * 120)
    print("  SECTION 4: SIGNAL QUALITY PERCENTILE GATE")
    print("  Only enter when score exceeds historical percentile threshold")
    print("  0=no gate, 25=top 75th pctile, 50=top 50th, 75=top 25th")
    print("=" * 120)

    pct_results = []
    for pct in SCORE_PCT_THRESHOLDS:
        r = backtest(dd_tiers=dd_base, top_n=3, score_pct_threshold=pct)
        label = f"pctGate={pct}"
        pr(r, label)
        pct_results.append((r, pct, label))

    # ===================== SECTION 5: COMBINED SWEEP =====================
    print("\n" + "=" * 120)
    print("  SECTION 5: COMBINED PARAMETER SWEEP")
    print("  Sweep: min_signal_score x rebalance_freq x min_candidates")
    print("=" * 120)

    all_results = []   # (ann, mdd, sharpe, n, label, config_dict)

    for mss in MIN_SIGNAL_SCORES:
        for freq in REBALANCE_FREQS:
            for mc in MIN_CANDIDATES:
                r = backtest(dd_tiers=dd_base, top_n=3,
                             min_signal_score=mss, rebalance_freq=freq,
                             min_candidates=mc)
                label = f"mss={mss:3d} freq={freq} mc={mc}"
                pr(r, label)
                all_results.append((r['ann'], r['mdd'], r['sharpe'], r['n'],
                                    label,
                                    {'mss': mss, 'freq': freq, 'mc': mc}))

    # ===================== SECTION 6: COMBINED SWEEP WITH PCT GATE =====================
    print("\n" + "=" * 120)
    print("  SECTION 6: COMBINED SWEEP WITH PERCENTILE GATE")
    print("  Sweep: score_pct_threshold x rebalance_freq x min_candidates")
    print("  (min_signal_score=0 to avoid double-gating)")
    print("=" * 120)

    for pct in SCORE_PCT_THRESHOLDS:
        for freq in REBALANCE_FREQS:
            for mc in MIN_CANDIDATES:
                r = backtest(dd_tiers=dd_base, top_n=3,
                             score_pct_threshold=pct, rebalance_freq=freq,
                             min_candidates=mc)
                label = f"pct={pct:2d} freq={freq} mc={mc}"
                pr(r, label)
                all_results.append((r['ann'], r['mdd'], r['sharpe'], r['n'],
                                    label,
                                    {'pct': pct, 'freq': freq, 'mc': mc}))

    # ===================== SECTION 7: TOP 20 FULL-PERIOD =====================
    print("\n" + "=" * 120)
    print("  SECTION 7: TOP 20 FULL-PERIOD by annual return")
    print("=" * 120)

    all_results.sort(key=lambda x: -x[0])
    for i, (ann, mdd, sh, n, label, cfg) in enumerate(all_results[:20]):
        ratio = abs(ann / mdd) if mdd != 0 else 0
        print(f"  #{i+1:2d} | Ann={ann:+8.1f}% | MDD={mdd:6.1f}% | Sh={sh:4.2f} | R/M={ratio:.2f} | N={n:4d} | {label}")

    # ===================== SECTION 8: TOP 20 BY RISK-ADJUSTED (ANN/MDD) =====================
    print("\n" + "=" * 120)
    print("  SECTION 8: TOP 20 FULL-PERIOD by ANN/|MDD| ratio")
    print("=" * 120)

    all_results_ra = sorted(all_results, key=lambda x: -abs(x[0]/x[1]) if x[1] != 0 else 0)
    for i, (ann, mdd, sh, n, label, cfg) in enumerate(all_results_ra[:20]):
        ratio = abs(ann / mdd) if mdd != 0 else 0
        print(f"  #{i+1:2d} | Ann={ann:+8.1f}% | MDD={mdd:6.1f}% | Sh={sh:4.2f} | R/M={ratio:.2f} | N={n:4d} | {label}")

    # ===================== SECTION 9: WALK-FORWARD VALIDATION — TOP 20 CONFIGS =====================
    print("\n" + "=" * 120)
    print("  SECTION 9: WALK-FORWARD VALIDATION — TOP 20 by full-period ann")
    print("=" * 120)

    wf_all = {}
    for i, (ann, mdd, sh, n, label, cfg) in enumerate(all_results[:20]):
        # Reconstruct backtest params from config dict
        mss = cfg.get('mss', 0)
        freq = cfg.get('freq', 1)
        mc = cfg.get('mc', 0)
        pct = cfg.get('pct', 0)
        wf_res = walk_forward(dd_tiers=dd_base, top_n=3,
                               min_signal_score=mss, rebalance_freq=freq,
                               min_candidates=mc, score_pct_threshold=pct,
                               label=label)
        wf_all[label] = (wf_res, cfg)
        print_wf(wf_res, label)

    # Also WF validate the top 20 by risk-adjusted (to catch configs that may not
    # be in the top-20-by-ann list)
    for i, (ann, mdd, sh, n, label, cfg) in enumerate(all_results_ra[:20]):
        if label in wf_all: continue
        mss = cfg.get('mss', 0)
        freq = cfg.get('freq', 1)
        mc = cfg.get('mc', 0)
        pct = cfg.get('pct', 0)
        wf_res = walk_forward(dd_tiers=dd_base, top_n=3,
                               min_signal_score=mss, rebalance_freq=freq,
                               min_candidates=mc, score_pct_threshold=pct,
                               label=label)
        wf_all[label] = (wf_res, cfg)
        print_wf(wf_res, f"[RA #{i+1}] {label}")

    # ===================== SECTION 10: TOP 10 BY WF AVERAGE =====================
    print("\n" + "=" * 120)
    print("  SECTION 10: TOP 10 CONFIGS BY WF AVERAGE")
    print("=" * 120)

    wf_ranked = []
    for label, (wf_res, cfg) in wf_all.items():
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        best_ann = max(r['ann'] for r in wf_res.values())
        pos = sum(1 for r in wf_res.values() if r['ann'] > 0)
        total_n = sum(r['n'] for r in wf_res.values())
        avg_wr = np.mean([r['wr'] for r in wf_res.values()])
        wf_ranked.append((avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res))

    wf_ranked.sort(key=lambda x: -x[0])

    for i, (avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res) in enumerate(wf_ranked[:10]):
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        ratio = abs(avg_ann / worst_mdd) if worst_mdd != 0 else 0
        print(f"\n  #{i+1} WF AVG={avg_ann:+.0f}% | WorstMDD={worst_mdd:.0f}% | R/M={ratio:.2f} | {pos}/6 pos | TotalTrades={total_n} | AvgWR={avg_wr:.1f}%")
        print(f"     {label}")
        print(f"     {ws}")

    # ===================== SECTION 11: TOP 10 BY RISK-ADJUSTED WF =====================
    print("\n" + "=" * 120)
    print("  SECTION 11: BEST RISK-ADJUSTED (WF avg / |worst MDD|)")
    print("=" * 120)

    wf_ra = sorted(wf_ranked, key=lambda x: -abs(x[0] / x[1]) if x[1] != 0 else 0)
    for i, (avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res) in enumerate(wf_ra[:10]):
        ratio = abs(avg_ann / worst_mdd) if worst_mdd != 0 else 0
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  #{i+1} WF AVG={avg_ann:+.0f}% | WorstMDD={worst_mdd:.0f}% | R/M={ratio:.2f} | {pos}/6 pos")
        print(f"     {label}")
        print(f"     {ws}")

    # ===================== SECTION 12: COMPARISON TABLE =====================
    print("\n" + "=" * 120)
    print("  SECTION 12: COMPARISON — REBALANCE FREQ vs BASELINE")
    print("  Shows each rebalance_freq's best config vs baseline (freq=1)")
    print("=" * 120)

    for freq in REBALANCE_FREQS:
        freq_configs = [(avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr)
                        for avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr in wf_ranked
                        if cfg.get('freq', 1) == freq]
        if not freq_configs:
            print(f"\n  freq={freq}: no WF results")
            continue
        best = max(freq_configs, key=lambda x: x[0])
        avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res = best
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  freq={freq} BEST: {label}")
        print(f"    WF AVG={avg_ann:+.0f}% | WorstMDD={worst_mdd:.0f}% | {pos}/6 pos")
        print(f"    {ws}")

    # ===================== SECTION 13: SIGNAL SCORE GATE vs BASELINE =====================
    print("\n" + "=" * 120)
    print("  SECTION 13: MIN SIGNAL SCORE — best config per threshold")
    print("=" * 120)

    for mss in MIN_SIGNAL_SCORES:
        mss_configs = [(avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr)
                       for avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr in wf_ranked
                       if cfg.get('mss', 0) == mss]
        if not mss_configs:
            print(f"\n  minScore={mss}: no WF results")
            continue
        best = max(mss_configs, key=lambda x: x[0])
        avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res = best
        print(f"\n  minScore={mss} BEST: {label}")
        print(f"    WF AVG={avg_ann:+.0f}% | WorstMDD={worst_mdd:.0f}% | {pos}/6 pos | Trades={total_n}")

    # ===================== SECTION 14: MIN CANDIDATES — best config per threshold =====================
    print("\n" + "=" * 120)
    print("  SECTION 14: MIN CANDIDATES — best config per threshold")
    print("=" * 120)

    for mc in MIN_CANDIDATES:
        mc_configs = [(avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr)
                      for avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr in wf_ranked
                      if cfg.get('mc', 0) == mc]
        if not mc_configs:
            print(f"\n  minCand={mc}: no WF results")
            continue
        best = max(mc_configs, key=lambda x: x[0])
        avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res = best
        print(f"\n  minCand={mc} BEST: {label}")
        print(f"    WF AVG={avg_ann:+.0f}% | WorstMDD={worst_mdd:.0f}% | {pos}/6 pos | Trades={total_n}")

    # ===================== SECTION 15: PERCENTILE GATE — best config per threshold =====================
    print("\n" + "=" * 120)
    print("  SECTION 15: PERCENTILE GATE — best config per threshold")
    print("=" * 120)

    for pct in SCORE_PCT_THRESHOLDS:
        pct_configs = [(avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr)
                       for avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr in wf_ranked
                       if cfg.get('pct', 0) == pct]
        if not pct_configs:
            print(f"\n  pctGate={pct}: no WF results")
            continue
        best = max(pct_configs, key=lambda x: x[0])
        avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res = best
        print(f"\n  pctGate={pct} BEST: {label}")
        print(f"    WF AVG={avg_ann:+.0f}% | WorstMDD={worst_mdd:.0f}% | {pos}/6 pos | Trades={total_n}")

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 120)
    print("  FINAL SUMMARY")
    print("=" * 120)

    print(f"\n  BASELINE (daily rebalance, no gates, aggro100, top_n=3):")
    pr(r_base, "  Baseline")

    if wf_ranked:
        best = wf_ranked[0]
        avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res = best
        print(f"\n  BEST V161 (by WF avg): {label}")
        print(f"    WF AVG={avg_ann:+.0f}% | WorstMDD={worst_mdd:.0f}% | {pos}/6 pos | Trades={total_n} | AvgWR={avg_wr:.1f}%")
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"    {ws}")

        # Show improvement over baseline
        baseline_label = "mss=  0 freq=1 mc=0"
        if baseline_label in wf_all:
            base_wf = wf_all[baseline_label][0]
            base_avg = np.mean([r['ann'] for r in base_wf.values()])
            base_wmdd = min(r['mdd'] for r in base_wf.values())
            improvement = avg_ann - base_avg
            mdd_improvement = worst_mdd - base_wmdd  # less negative = better
            print(f"\n  vs WF baseline (freq=1, no gates):")
            print(f"    WF AVG improvement: {improvement:+.0f}% ({base_avg:+.0f}% -> {avg_ann:+.0f}%)")
            print(f"    Worst MDD change: {mdd_improvement:+.0f}% ({base_wmdd:.0f}% -> {worst_mdd:.0f}%)")

        # Best risk-adjusted
        if wf_ra:
            best_ra = wf_ra[0]
            avg_ann_ra, worst_mdd_ra, _, pos_ra, total_n_ra, avg_wr_ra, label_ra, cfg_ra, wf_res_ra = best_ra
            print(f"\n  BEST RISK-ADJUSTED V161: {label_ra}")
            print(f"    WF AVG={avg_ann_ra:+.0f}% | WorstMDD={worst_mdd_ra:.0f}% | {pos_ra}/6 pos")
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(wf_res_ra.items())])
            print(f"    {ws}")

        # Key findings: how many non-baseline configs beat baseline?
        non_baseline = [(avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr)
                        for avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr in wf_ranked
                        if not (cfg.get('mss', 0) == 0 and cfg.get('freq', 1) == 1 and
                                cfg.get('mc', 0) == 0 and cfg.get('pct', 0) == 0)]
        if non_baseline:
            baseline_label_chk = "mss=  0 freq=1 mc=0"
            if baseline_label_chk in wf_all:
                base_wf = wf_all[baseline_label_chk][0]
                base_avg = np.mean([r['ann'] for r in base_wf.values()])
                beat = sum(1 for avg, _, _, _, _, _, _, _, _ in non_baseline if avg > base_avg)
                print(f"\n  {beat}/{len(non_baseline)} non-baseline configs beat baseline WF avg ({base_avg:+.0f}%)")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
