"""
Alpha Futures V171 — Dynamic top_n Based on Daily Signal Quality
================================================================
V169 champion uses fixed top_n=3 with atr_norm<12% vol filter giving +253%/-15% WF.

V171 explores DYNAMIC top_n: instead of always holding 3 positions, adjust top_n
based on how many high-quality signals are available each day.

Four adaptive methods:
  A. signal_count  — top_n = min(max_n, count_of_signals_passing_vol_filter)
  B. quality       — if best signal score > 75th percentile of historical scores, top_n=4; else top_n=3
  C. confidence    — rank all signals, take top N until cumulative score > threshold
  D. breadth       — if BREADTH > 0.6 → top_n=4; < 0.4 → top_n=2; else top_n=3

Parameter sweep:
  - adaptive_method: ['signal_count', 'quality', 'confidence', 'breadth']
  - max_top_n: [3, 4, 5]
  - min_top_n: [1, 2]
  - atr_norm_max: [10, 12]
  - max_corr: [0.5, 0.7]

Base: V169's vol filter + Kitchen Sink (dd*regime), aggro100 DD, regime 0.5-1.5.
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
    print("  V171 — Dynamic top_n Based on Daily Signal Quality")
    print("  Instead of fixed top_n, adapt to signal count/quality/confidence/breadth")
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

    print(f"  Market vol median={VOL_MEDIAN:.4f}%")
    print(f"  Done ({time.time()-t0:.1f}s)")

    # ===================== HISTORICAL SCORE PERCENTILES (for quality method) =====================
    # Pre-compute the 75th percentile of signal scores over a rolling lookback
    # We collect all scores from signal generation during backtest, so we'll
    # compute a running history. For simplicity we pre-scan to get a stable
    # percentile baseline using the first pass approach.

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

    # ===================== ADAPTIVE top_n LOGIC =====================
    # Four methods to determine how many positions to hold each day:
    #
    # A. signal_count: top_n = clip(count_of_eligible_signals, min_top_n, max_top_n)
    #    Many signals -> more positions; few signals -> concentrate
    #
    # B. quality: compare best signal's score to historical 75th percentile
    #    Strong best signal -> top_n=max_top_n; else -> top_n=default
    #
    # C. confidence: accumulate sorted signal scores until sum > threshold fraction
    #    of total available signal score
    #
    # D. breadth: use market BREADTH indicator to decide
    #    BREADTH > 0.6 -> max_top_n; BREADTH < 0.4 -> min_top_n; else default

    def adaptive_top_n(method, min_tn, max_tn, filtered_cands, di,
                       score_history=None, confidence_threshold=0.7):
        """
        Compute dynamic top_n for a given day.

        Parameters
        ----------
        method : str - one of 'signal_count', 'quality', 'confidence', 'breadth'
        min_tn : int - minimum top_n (floor)
        max_tn : int - maximum top_n (ceiling)
        filtered_cands : list of (score, si, entry_price, sig_str) - vol-filtered candidates
        di : int - current day index
        score_history : list of float - rolling history of all observed signal scores
        confidence_threshold : float - for confidence method, fraction of total score to capture

        Returns
        -------
        int : the dynamic top_n for this day
        """
        default_tn = (min_tn + max_tn) // 2  # midpoint as default

        if method == 'signal_count':
            # A. Use count of signals passing vol filter directly
            n_signals = len(filtered_cands)
            tn = min(max_tn, max(min_tn, n_signals))
            return tn

        elif method == 'quality':
            # B. If best signal score > 75th percentile of historical scores, use max_top_n
            #    else use min_top_n or default
            if not filtered_cands:
                return min_tn
            best_score = filtered_cands[0][0]  # sorted descending
            if score_history is not None and len(score_history) >= 50:
                p75 = np.percentile(score_history, 75)
                if best_score > p75:
                    return max_tn
                else:
                    # Use min_top_n when quality is low
                    return min_tn
            else:
                # Not enough history yet, use default
                return default_tn

        elif method == 'confidence':
            # C. Take positions until cumulative score > threshold fraction of total
            if not filtered_cands:
                return min_tn
            scores = np.array([c[0] for c in filtered_cands])
            total_score = np.sum(np.abs(scores))
            if total_score <= 0:
                return min_tn
            cumsum = np.cumsum(np.abs(scores))
            # Find how many positions needed to reach confidence_threshold of total
            threshold_val = total_score * confidence_threshold
            n_needed = 1
            for k in range(len(cumsum)):
                if cumsum[k] >= threshold_val:
                    n_needed = k + 1
                    break
            tn = min(max_tn, max(min_tn, n_needed))
            return tn

        elif method == 'breadth':
            # D. Use market BREADTH indicator
            bth = BREADTH[di]
            if np.isnan(bth):
                return default_tn
            if bth > 0.6:
                return max_tn
            elif bth < 0.4:
                return min_tn
            else:
                return default_tn

        else:
            return default_tn

    # ===================== BACKTEST ENGINE =====================
    def backtest(start_di=MIN_TRAIN, end_di=None,
                 atr_norm_max=12.0, max_corr=0.7,
                 dd_tiers=None,
                 regime_lo=0.5, regime_hi=1.5,
                 sl_pct=0.0, hold=1,
                 adaptive_method='signal_count',
                 min_top_n=2, max_top_n=4,
                 confidence_threshold=0.7,
                 fixed_top_n=None):
        """
        Backtest with dynamic top_n based on daily signal quality.

        If fixed_top_n is set, overrides adaptive logic (used for baseline).
        """
        if end_di is None: end_di = ND
        if dd_tiers is None:
            dd_tiers = [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)]

        cash = float(CASH0)
        positions = []
        trades = []
        daily_eq = []
        high_water = float(CASH0)

        # Rolling history of all signal scores for quality method
        score_history = []
        # Track dynamic top_n usage for diagnostics
        topn_counts = {}

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

            # --- Determine dynamic top_n ---
            if fixed_top_n is not None:
                cur_top_n = fixed_top_n
            else:
                # Get signals first to determine top_n
                edi = di + 1
                if edi >= end_di: continue

                cands_v121 = sig_v121(di, edi)
                cands_union = sig_union(di, edi)

                # Apply vol filter
                cands_v121_f = [c for c in cands_v121
                                if not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < atr_norm_max]
                cands_union_f = [c for c in cands_union
                                 if not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < atr_norm_max]

                cands_v121_f.sort(key=lambda x: -x[0])
                cands_union_f.sort(key=lambda x: -x[0])

                # Merge into a single pool of all eligible candidates with dedup
                # (union subsumes v121, so we take union as primary, then add v121-only)
                union_sis = set(c[1] for c in cands_union_f)
                all_cands = list(cands_union_f)
                for c in cands_v121_f:
                    if c[1] not in union_sis:
                        all_cands.append(c)
                all_cands.sort(key=lambda x: -x[0])

                # Update score history for quality method
                for c in all_cands:
                    score_history.append(c[0])
                # Keep rolling window of 500 to avoid memory bloat
                if len(score_history) > 2000:
                    score_history = score_history[-1000:]

                cur_top_n = adaptive_top_n(
                    method=adaptive_method,
                    min_tn=min_top_n,
                    max_tn=max_top_n,
                    filtered_cands=all_cands,
                    di=di,
                    score_history=score_history,
                    confidence_threshold=confidence_threshold
                )

                # Track distribution
                topn_counts[cur_top_n] = topn_counts.get(cur_top_n, 0) + 1

                # Skip day entirely if no signals (cur_top_n effectively 0)
                if not all_cands:
                    continue

                # Now proceed with entry using cur_top_n
                held_si = set(p['si'] for p in positions)
                if len(positions) >= cur_top_n:
                    continue

                # Build entries from all candidates, respecting correlation and top_n
                entries = []
                used_si = set()

                for sc, s, pr, sig_str in all_cands:
                    if s in held_si or s in used_si: continue
                    if len(entries) >= cur_top_n: break

                    # Check correlation with already-selected entries
                    too_corr = False
                    for _, prev_s, _, _ in entries:
                        corr = get_corr(s, prev_s, di)
                        if corr >= max_corr:
                            too_corr = True
                            break
                    if too_corr: continue

                    # Also check correlation with existing positions
                    for p in positions:
                        corr = get_corr(s, p['si'], di)
                        if corr >= max_corr:
                            too_corr = True
                            break
                    if too_corr: continue

                    entries.append((sc, s, pr, sig_str))
                    used_si.add(s)

                if not entries:
                    continue

                # Determine per-position sizing
                # With dynamic top_n, we scale position size: more concentrated when fewer slots
                # When cur_top_n is small, each position gets larger allocation
                n_planned = len(entries)

                cash_snapshot = cash
                for sc, s, pr, sig_str in entries:
                    if s in set(p['si'] for p in positions): continue
                    if len(positions) >= cur_top_n: break

                    # Allocate capital: divide equally among planned positions
                    cap = cash_snapshot * pos_size / n_planned
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

                continue  # Skip the fixed-top_n entry block below

            # --- Fixed top_n entry (baseline path) ---
            if len(positions) >= cur_top_n: continue
            edi = di + 1
            if edi >= end_di: continue

            held_si = set(p['si'] for p in positions)

            cands_v121 = sig_v121(di, edi)
            cands_union = sig_union(di, edi)

            cands_v121_f = [c for c in cands_v121
                            if not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < atr_norm_max]
            cands_union_f = [c for c in cands_union
                             if not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < atr_norm_max]

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
                if len(positions) >= cur_top_n: break
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

        result = {'ann': ann, 'wr': wr, 'n': nt, 'mdd': mdd, 'sharpe': sh, 'final': cash}
        if topn_counts:
            result['topn_dist'] = topn_counts
        return result

    # ===================== PRINTING HELPERS =====================
    def pr(r, label=""):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        tn_info = ""
        if 'topn_dist' in r:
            tn_parts = [f"n{k}:{v}" for k, v in sorted(r['topn_dist'].items())]
            tn_info = f" [{', '.join(tn_parts)}]"
        print(f"  {label:95s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d}{tn_info}")

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

    # ===================== SECTION 0: V169 BASELINE REPRODUCTION =====================
    print("\n" + "=" * 130)
    print("  SECTION 0: V169 BASELINE REPRODUCTION")
    print("  Fixed top_n=3, atr_norm<12%, max_corr=0.7, aggro100 DD, Kitchen Sink")
    print("=" * 130)

    for atr_max in [10.0, 12.0]:
        for mc in [0.5, 0.7]:
            label = f"BASELINE: top_n=3, atr<{atr_max:.0f}%, corr={mc:.1f}"
            r = backtest(atr_norm_max=atr_max, max_corr=mc,
                         dd_tiers=DD_AGGR100, regime_lo=0.5, regime_hi=1.5,
                         sl_pct=0.0, hold=1, fixed_top_n=3)
            pr(r, label)
            all_results.append({**r, 'label': f'base_atr{atr_max:.0f}_c{mc:.1f}',
                                'section': 0, 'method': 'fixed', 'atr_norm_max': atr_max,
                                'max_corr': mc, 'min_top_n': 3, 'max_top_n': 3})

    # ===================== SECTION 1: METHOD A — SIGNAL COUNT ADAPTIVE =====================
    print("\n" + "=" * 130)
    print("  SECTION 1: SIGNAL COUNT ADAPTIVE")
    print("  top_n = clip(n_signals_passing_vol_filter, min_top_n, max_top_n)")
    print("  Many signals -> more positions; few signals -> concentrate")
    print("=" * 130)

    for atr_max in [10.0, 12.0]:
        for mc in [0.5, 0.7]:
            for min_tn in [1, 2]:
                for max_tn in [3, 4, 5]:
                    if max_tn <= min_tn: continue
                    label = f"SIG_COUNT: atr<{atr_max:.0f}%, c={mc:.1f}, tn=[{min_tn},{max_tn}]"
                    r = backtest(atr_norm_max=atr_max, max_corr=mc,
                                 dd_tiers=DD_AGGR100, regime_lo=0.5, regime_hi=1.5,
                                 sl_pct=0.0, hold=1,
                                 adaptive_method='signal_count',
                                 min_top_n=min_tn, max_top_n=max_tn)
                    pr(r, label)
                    all_results.append({**r, 'label': f'sigcnt_atr{atr_max:.0f}_c{mc:.1f}_tn{min_tn}-{max_tn}',
                                        'section': 1, 'method': 'signal_count',
                                        'atr_norm_max': atr_max, 'max_corr': mc,
                                        'min_top_n': min_tn, 'max_top_n': max_tn})

    # ===================== SECTION 2: METHOD B — QUALITY ADAPTIVE =====================
    print("\n" + "=" * 130)
    print("  SECTION 2: QUALITY ADAPTIVE")
    print("  If best score > P75 of historical scores -> max_top_n; else -> min_top_n")
    print("=" * 130)

    for atr_max in [10.0, 12.0]:
        for mc in [0.5, 0.7]:
            for min_tn in [1, 2]:
                for max_tn in [3, 4, 5]:
                    if max_tn <= min_tn: continue
                    label = f"QUALITY: atr<{atr_max:.0f}%, c={mc:.1f}, tn=[{min_tn},{max_tn}]"
                    r = backtest(atr_norm_max=atr_max, max_corr=mc,
                                 dd_tiers=DD_AGGR100, regime_lo=0.5, regime_hi=1.5,
                                 sl_pct=0.0, hold=1,
                                 adaptive_method='quality',
                                 min_top_n=min_tn, max_top_n=max_tn)
                    pr(r, label)
                    all_results.append({**r, 'label': f'qual_atr{atr_max:.0f}_c{mc:.1f}_tn{min_tn}-{max_tn}',
                                        'section': 2, 'method': 'quality',
                                        'atr_norm_max': atr_max, 'max_corr': mc,
                                        'min_top_n': min_tn, 'max_top_n': max_tn})

    # ===================== SECTION 3: METHOD C — CONFIDENCE WEIGHTED =====================
    print("\n" + "=" * 130)
    print("  SECTION 3: CONFIDENCE WEIGHTED")
    print("  Take top N until cumulative score > threshold fraction of total")
    print("=" * 130)

    for atr_max in [10.0, 12.0]:
        for mc in [0.5, 0.7]:
            for min_tn in [1, 2]:
                for max_tn in [3, 4, 5]:
                    if max_tn <= min_tn: continue
                    for conf_thresh in [0.5, 0.7, 0.9]:
                        label = (f"CONF: atr<{atr_max:.0f}%, c={mc:.1f}, "
                                 f"tn=[{min_tn},{max_tn}], thresh={conf_thresh}")
                        r = backtest(atr_norm_max=atr_max, max_corr=mc,
                                     dd_tiers=DD_AGGR100, regime_lo=0.5, regime_hi=1.5,
                                     sl_pct=0.0, hold=1,
                                     adaptive_method='confidence',
                                     min_top_n=min_tn, max_top_n=max_tn,
                                     confidence_threshold=conf_thresh)
                        pr(r, label)
                        all_results.append({
                            **r,
                            'label': (f'conf_atr{atr_max:.0f}_c{mc:.1f}_'
                                      f'tn{min_tn}-{max_tn}_t{conf_thresh}'),
                            'section': 3, 'method': 'confidence',
                            'atr_norm_max': atr_max, 'max_corr': mc,
                            'min_top_n': min_tn, 'max_top_n': max_tn,
                            'conf_thresh': conf_thresh})

    # ===================== SECTION 4: METHOD D — BREADTH ADAPTIVE =====================
    print("\n" + "=" * 130)
    print("  SECTION 4: BREADTH ADAPTIVE")
    print("  BREADTH > 0.6 -> max_top_n; < 0.4 -> min_top_n; else default")
    print("=" * 130)

    for atr_max in [10.0, 12.0]:
        for mc in [0.5, 0.7]:
            for min_tn in [1, 2]:
                for max_tn in [3, 4, 5]:
                    if max_tn <= min_tn: continue
                    label = f"BREADTH: atr<{atr_max:.0f}%, c={mc:.1f}, tn=[{min_tn},{max_tn}]"
                    r = backtest(atr_norm_max=atr_max, max_corr=mc,
                                 dd_tiers=DD_AGGR100, regime_lo=0.5, regime_hi=1.5,
                                 sl_pct=0.0, hold=1,
                                 adaptive_method='breadth',
                                 min_top_n=min_tn, max_top_n=max_tn)
                    pr(r, label)
                    all_results.append({**r, 'label': f'brdth_atr{atr_max:.0f}_c{mc:.1f}_tn{min_tn}-{max_tn}',
                                        'section': 4, 'method': 'breadth',
                                        'atr_norm_max': atr_max, 'max_corr': mc,
                                        'min_top_n': min_tn, 'max_top_n': max_tn})

    # ===================== SECTION 5: FULL PERIOD RANKING =====================
    print("\n" + "=" * 130)
    print("  SECTION 5: ALL CONFIGS RANKED BY ANNUAL RETURN (full period)")
    print("=" * 130)

    all_results.sort(key=lambda x: -x['ann'])
    for i, r in enumerate(all_results[:30]):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i+1:2d} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d} | {r['label']}")

    print("\n" + "=" * 130)
    print("  SECTION 5b: ALL CONFIGS RANKED BY R/M RATIO (risk-adjusted)")
    print("=" * 130)

    all_rm = sorted(all_results, key=lambda x: -abs(x['ann']/x['mdd']) if x['mdd'] != 0 else 0)
    for i, r in enumerate(all_rm[:30]):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i+1:2d} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d} | {r['label']}")

    # ===================== SECTION 6: WALK-FORWARD TOP CONFIGS =====================
    print("\n" + "=" * 130)
    print("  SECTION 6: WALK-FORWARD VALIDATION — Top 25 by R/M")
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
    for r in wf_candidates[:25]:
        lbl = r['label']
        wf_kwargs = {}

        if r['method'] == 'fixed':
            wf_kwargs = {
                'fixed_top_n': 3,
                'atr_norm_max': r['atr_norm_max'],
                'max_corr': r['max_corr'],
                'dd_tiers': DD_AGGR100,
                'regime_lo': 0.5, 'regime_hi': 1.5,
                'sl_pct': 0.0, 'hold': 1,
            }
        else:
            wf_kwargs = {
                'atr_norm_max': r['atr_norm_max'],
                'max_corr': r['max_corr'],
                'dd_tiers': DD_AGGR100,
                'regime_lo': 0.5, 'regime_hi': 1.5,
                'sl_pct': 0.0, 'hold': 1,
                'adaptive_method': r['method'],
                'min_top_n': r['min_top_n'],
                'max_top_n': r['max_top_n'],
            }
            if r['method'] == 'confidence' and 'conf_thresh' in r:
                wf_kwargs['confidence_threshold'] = r['conf_thresh']

        wf_res = walk_forward(label=lbl, **wf_kwargs)
        wf_all[lbl] = (wf_res, r)
        print_wf(wf_res, lbl)

    # ===================== SECTION 7: WF COMPARISON TABLE =====================
    print("\n" + "=" * 130)
    print("  SECTION 7: WF COMPARISON TABLE — All configs ranked by WF avg")
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
    print(f"  {'-'*100}")
    for i, (avg_ann, wmdd, bann, pos, tn, awr, ratio, lbl, wf_res, ri) in enumerate(wf_ranked):
        print(f"  {i+1:3d}  {avg_ann:>+8.0f}% {wmdd:>7.0f}% {ratio:6.2f} {pos:4d}/6 {tn:5d} {awr:5.1f}%  {lbl}")

    # ===================== SECTION 8: BEST RISK-ADJUSTED (WF R/M) =====================
    print("\n" + "=" * 130)
    print("  SECTION 8: BEST RISK-ADJUSTED (WF R/M)")
    print("=" * 130)

    wf_ra = sorted(wf_ranked, key=lambda x: -x[6])
    for i, (avg_ann, wmdd, bann, pos, tn, awr, ratio, lbl, wf_res, ri) in enumerate(wf_ra[:15]):
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  #{i+1} WF AVG={avg_ann:+.0f}% | WorstMDD={wmdd:.0f}% | R/M={ratio:.2f} | {pos}/6 pos | {lbl}")
        print(f"     {ws}")

    # ===================== SECTION 9: BEST PER METHOD =====================
    print("\n" + "=" * 130)
    print("  SECTION 9: BEST PER ADAPTIVE METHOD")
    print("=" * 130)

    methods = {
        'Fixed baseline': lambda x: x.get('method') == 'fixed',
        'Signal Count': lambda x: x.get('method') == 'signal_count',
        'Quality': lambda x: x.get('method') == 'quality',
        'Confidence': lambda x: x.get('method') == 'confidence',
        'Breadth': lambda x: x.get('method') == 'breadth',
    }

    for mname, mfilter in methods.items():
        m_items = [(avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri)
                   for avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri in wf_ranked
                   if mfilter(ri)]
        if not m_items:
            print(f"\n  {mname:20s}: no WF results")
            continue
        m_items.sort(key=lambda x: -x[6])
        best = m_items[0]
        avg_ann, wmdd, bann, pos, tn, awr, ratio, lbl, wf_res, ri = best
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  {mname:20s} best: WF AVG={avg_ann:+.0f}% | WorstMDD={wmdd:.0f}% | R/M={ratio:.2f} | {lbl}")
        print(f"     {ws}")

    # ===================== SECTION 10: DELTA vs BASELINE =====================
    print("\n" + "=" * 130)
    print("  SECTION 10: IMPROVEMENT vs V169 BASELINE (fixed top_n=3, atr<12%, corr=0.7)")
    print("=" * 130)

    base_lbl = 'base_atr12.0_c0.7'
    if base_lbl in wf_all:
        b_wf, b_ri = wf_all[base_lbl]
        b_avg = np.mean([r['ann'] for r in b_wf.values()])
        b_wmdd = min(r['mdd'] for r in b_wf.values())
        b_rm = abs(b_avg / b_wmdd) if b_wmdd != 0 else 0
    else:
        b_wf = walk_forward(label=base_lbl,
                            fixed_top_n=3, atr_norm_max=12.0, max_corr=0.7,
                            dd_tiers=DD_AGGR100, regime_lo=0.5, regime_hi=1.5,
                            sl_pct=0.0, hold=1)
        b_avg = np.mean([r['ann'] for r in b_wf.values()])
        b_wmdd = min(r['mdd'] for r in b_wf.values())
        b_rm = abs(b_avg / b_wmdd) if b_wmdd != 0 else 0
        print_wf(b_wf, "V169 BASELINE (computed)")

    print(f"\n  V169 BASELINE: WF AVG={b_avg:+.0f}% | WorstMDD={b_wmdd:.0f}% | R/M={b_rm:.2f}")

    deltas = []
    for avg_ann, wmdd, bann, pos, tn, awr, ratio, lbl, wf_res, ri in wf_ranked:
        if lbl == base_lbl: continue
        delta_ann = avg_ann - b_avg
        delta_rm = ratio - b_rm
        deltas.append((delta_ann, delta_rm, ratio, avg_ann, wmdd, pos, lbl, wf_res, ri))

    deltas.sort(key=lambda x: -x[1])
    print(f"\n  Configs by R/M improvement over V169 baseline:")
    for i, (da, drm, ratio, avg, wmdd, pos, lbl, wfr, ri) in enumerate(deltas[:30]):
        marker = "*** IMPROVED" if drm > 0 else "    worse"
        method = ri.get('method', '?')
        print(f"  {i+1:2d} | R/M={ratio:.2f} (d={drm:+.2f}) | Ann d={da:+.0f}% | {pos}/6 | {marker} | {method:13s} | {lbl}")

    # ===================== SECTION 11: BEST COMBINATION DETAIL =====================
    print("\n" + "=" * 130)
    print("  SECTION 11: BEST COMBINATION — Full detail for top 5 by WF R/M")
    print("=" * 130)

    for i, (avg_ann, wmdd, bann, pos, tn, awr, ratio, lbl, wf_res, ri) in enumerate(wf_ra[:5]):
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  #{i+1}: {lbl}")
        print(f"       WF AVG={avg_ann:+.0f}% | WorstMDD={wmdd:.0f}% | R/M={ratio:.2f} | {pos}/6 pos | N={tn}")
        print(f"       {ws}")
        for yr, r in sorted(wf_res.items()):
            tn_dist = r.get('topn_dist', {})
            tn_str = str(tn_dist) if tn_dist else ""
            print(f"         {yr}: Ann={r['ann']:+.1f}% | MDD={r['mdd']:.1f}% | WR={r['wr']:.1f}% | N={r['n']} {tn_str}")

    # ===================== SECTION 12: TOPN DISTRIBUTION ANALYSIS =====================
    print("\n" + "=" * 130)
    print("  SECTION 12: TOPN DISTRIBUTION ANALYSIS — How often each top_n was used")
    print("=" * 130)

    # For the best adaptive configs, show how often each top_n value was selected
    for i, (avg_ann, wmdd, bann, pos, tn, awr, ratio, lbl, wf_res, ri) in enumerate(wf_ra[:10]):
        if ri.get('method') == 'fixed':
            continue
        # Aggregate topn distribution across WF years
        total_topn = {}
        for yr, r in wf_res.items():
            for k, v in r.get('topn_dist', {}).items():
                total_topn[k] = total_topn.get(k, 0) + v
        if total_topn:
            total_days = sum(total_topn.values())
            dist_str = ", ".join([f"n{k}:{v} ({v/total_days*100:.0f}%)"
                                  for k, v in sorted(total_topn.items())])
            print(f"  #{i+1} {lbl}")
            print(f"       top_n usage: {dist_str} (total={total_days} days)")

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 130)
    print("  FINAL SUMMARY")
    print("=" * 130)

    print(f"\n  V169 Baseline (fixed top_n=3, atr<12%, corr=0.7):")
    print(f"    WF AVG={b_avg:+.0f}% | WorstMDD={b_wmdd:.0f}% | R/M={b_rm:.2f}")

    if wf_ra:
        best = wf_ra[0]
        print(f"\n  Best overall: {best[7]}")
        print(f"    WF AVG={best[0]:+.0f}% | WorstMDD={best[1]:.0f}% | R/M={best[6]:.2f}")
        delta_rm_best = best[6] - b_rm
        delta_ann_best = best[0] - b_avg
        print(f"    vs V169: R/M delta={delta_rm_best:+.2f}, Ann delta={delta_ann_best:+.0f}%")
        method_best = best[9].get('method', 'fixed')
        print(f"    Method: {method_best}")

        # Best by method
        print(f"\n  Best by adaptive method (WF R/M):")
        for mname, mfilter in methods.items():
            m_items = [(avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri)
                       for avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri in wf_ranked
                       if mfilter(ri)]
            if not m_items: continue
            m_items.sort(key=lambda x: -x[6])
            s = m_items[0]
            delta_rm = s[6] - b_rm
            print(f"    {mname:20s}: R/M={s[6]:.2f} (d={delta_rm:+.2f}) | WF={s[0]:+.0f}% | {s[7]}")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
