"""
Alpha Futures V173 — Hold Period Optimization
==============================================================================
V169 champion uses hold=1 day with atr_norm<12% vol filter giving +253%/-15% WF.

V173 explores whether different hold periods capture more of the trend:
  A. Fixed hold periods:      hold=1, 2, 3, 5 days
  B. Profit-taking hold:      exit early if unrealized profit > X%, else hold max N
  C. Trailing exit:           hold until trailing stop hit or max_hold reached
  D. Signal-strength hold:    stronger signals get longer hold periods

Base: V169 vol filter + Kitchen Sink (dd*regime), aggro100 DD, regime 0.5-1.5.
Uses cash_snapshot before entry loop, RET for correlation.
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
    print("  V173 — Hold Period Optimization")
    print("  Exploring fixed, profit-take, trailing, and signal-strength hold modes")
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
    VOL_P50 = np.percentile(valid_vols, 50) if len(valid_vols) > 0 else 1.0
    VOL_P75 = np.percentile(valid_vols, 75) if len(valid_vols) > 0 else 1.5

    # Precompute signal score percentiles for signal-strength hold mode
    # We'll compute on-the-fly per day instead of precomputing globally
    # since signal candidates change each day

    print(f"  Market vol: median={VOL_MEDIAN:.4f}%, P50={VOL_P50:.4f}%, P75={VOL_P75:.4f}%")
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

    # ===================== BACKTEST ENGINE =====================
    # hold_mode controls how hold period is determined:
    #   'fixed'           : hold exactly hold_days days
    #   'profit_take'     : hold up to hold_days, exit early if unrealized profit > profit_target%
    #   'trailing'        : hold up to hold_days, exit when price drops trail_pct% from best
    #   'signal_strength' : hold_days based on signal score percentile

    def backtest(start_di=MIN_TRAIN, end_di=None,
                 atr_norm_max=12.0, max_corr=0.7,
                 dd_tiers=None,
                 regime_lo=0.5, regime_hi=1.5,
                 top_n=3,
                 hold_mode='fixed',
                 hold_days=1,
                 profit_target=3.0,
                 trail_pct=1.5):
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

            # --- Close positions based on hold mode ---
            cl = []
            for p in positions:
                days_held = di - p['entry_di']

                if hold_mode == 'fixed':
                    # Simple fixed hold
                    if days_held >= p['hold_days']:
                        ep = C[p['si'], di]
                        if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                        m = MULT.get(p['sym'], DEF_MULT)
                        pnl = (ep - p['entry_price']) * m * p['lots']
                        inv = p['entry_price'] * m * abs(p['lots'])
                        pp = pnl / inv * 100 if inv > 0 else 0
                        cash += ep * m * abs(p['lots']) * (1 - COMM)
                        trades.append(pp)
                        cl.append(p)

                elif hold_mode == 'profit_take':
                    # Check profit target first
                    cp = C[p['si'], di]
                    if not np.isnan(cp) and cp > 0:
                        m = MULT.get(p['sym'], DEF_MULT)
                        unrealized_pct = (cp - p['entry_price']) / p['entry_price'] * 100
                        if unrealized_pct >= p['profit_target']:
                            # Profit target hit — exit early
                            pnl = (cp - p['entry_price']) * m * p['lots']
                            inv = p['entry_price'] * m * abs(p['lots'])
                            pp = pnl / inv * 100 if inv > 0 else 0
                            cash += cp * m * abs(p['lots']) * (1 - COMM)
                            trades.append(pp)
                            cl.append(p)
                            continue
                    # Max hold reached
                    if days_held >= p['hold_days']:
                        ep = C[p['si'], di]
                        if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                        m = MULT.get(p['sym'], DEF_MULT)
                        pnl = (ep - p['entry_price']) * m * p['lots']
                        inv = p['entry_price'] * m * abs(p['lots'])
                        pp = pnl / inv * 100 if inv > 0 else 0
                        cash += ep * m * abs(p['lots']) * (1 - COMM)
                        trades.append(pp)
                        cl.append(p)

                elif hold_mode == 'trailing':
                    # Update high-water mark for this position
                    cp = C[p['si'], di]
                    if not np.isnan(cp) and cp > 0:
                        if cp > p.get('trail_high', p['entry_price']):
                            p['trail_high'] = cp
                        trail_high = p['trail_high']
                        draw_from_high = (trail_high - cp) / trail_high * 100
                        if draw_from_high >= p['trail_pct']:
                            # Trailing stop hit — exit
                            m = MULT.get(p['sym'], DEF_MULT)
                            pnl = (cp - p['entry_price']) * m * p['lots']
                            inv = p['entry_price'] * m * abs(p['lots'])
                            pp = pnl / inv * 100 if inv > 0 else 0
                            cash += cp * m * abs(p['lots']) * (1 - COMM)
                            trades.append(pp)
                            cl.append(p)
                            continue
                    # Max hold reached
                    if days_held >= p['hold_days']:
                        ep = C[p['si'], di]
                        if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                        m = MULT.get(p['sym'], DEF_MULT)
                        pnl = (ep - p['entry_price']) * m * p['lots']
                        inv = p['entry_price'] * m * abs(p['lots'])
                        pp = pnl / inv * 100 if inv > 0 else 0
                        cash += ep * m * abs(p['lots']) * (1 - COMM)
                        trades.append(pp)
                        cl.append(p)

                elif hold_mode == 'signal_strength':
                    # Already determined hold_days at entry time
                    if days_held >= p['hold_days']:
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

            # --- Enter positions ---
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue

            held_si = set(p['si'] for p in positions)

            # Get best V121 and best Union signal
            cands_v121 = sig_v121(di, edi)
            cands_union = sig_union(di, edi)

            # Apply vol filter (fixed mode from V169)
            cands_v121_f = [c for c in cands_v121
                            if not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < atr_norm_max]
            cands_union_f = [c for c in cands_union
                             if not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < atr_norm_max]

            cands_v121_f.sort(key=lambda x: -x[0])
            cands_union_f.sort(key=lambda x: -x[0])

            # Collect all candidate scores for signal-strength percentile
            all_cand_scores = [c[0] for c in cands_v121_f] + [c[0] for c in cands_union_f]

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

                # Determine hold_days and extra params based on hold_mode
                pos_record = {
                    'si': s, 'entry_price': pr, 'entry_di': edi,
                    'lots': ct, 'dir': 1, 'sym': sym,
                    'hold_days': hold_days, 'sig': sig_str, 'score': sc,
                }

                if hold_mode == 'profit_take':
                    pos_record['profit_target'] = profit_target
                elif hold_mode == 'trailing':
                    pos_record['trail_pct'] = trail_pct
                    pos_record['trail_high'] = pr  # initialize at entry price
                elif hold_mode == 'signal_strength':
                    # Assign hold_days based on score percentile
                    if len(all_cand_scores) >= 4:
                        p75 = np.percentile(all_cand_scores, 75)
                        p50 = np.percentile(all_cand_scores, 50)
                        if sc >= p75:
                            pos_record['hold_days'] = min(hold_days, 3)
                        elif sc >= p50:
                            pos_record['hold_days'] = min(hold_days, 2)
                        else:
                            pos_record['hold_days'] = 1
                    else:
                        pos_record['hold_days'] = hold_days

                positions.append(pos_record)

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

        # Compute average hold duration from trades (approximate from positions data)
        avg_hold = 0  # not tracked per trade, hold_days is the nominal value

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
    DD_AGGR100 = [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)]

    # Collect all results
    all_results = []

    # ===================== SECTION 0: V169 BASELINE REPRODUCTION =====================
    print("\n" + "=" * 130)
    print("  SECTION 0: V169 BASELINE (hold=1, atr<12%, corr=0.7, no eq)")
    print("=" * 130)

    for an_max in [10.0, 12.0]:
        for mc in [0.5, 0.7]:
            label = f"BASELINE hold=1, atr<{an_max:.0f}%, corr={mc:.1f}"
            r = backtest(atr_norm_max=an_max, max_corr=mc,
                         dd_tiers=DD_AGGR100, regime_lo=0.5, regime_hi=1.5,
                         top_n=3, hold_mode='fixed', hold_days=1)
            pr(r, label)
            all_results.append({**r, 'label': f'base_atr{an_max:.0f}_c{mc:.1f}',
                                'section': 0, 'hold_mode': 'fixed', 'hold_days': 1,
                                'atr_norm_max': an_max, 'max_corr': mc, 'top_n': 3,
                                'profit_target': None, 'trail_pct': None})

    # ===================== SECTION 1: FIXED HOLD PERIODS =====================
    print("\n" + "=" * 130)
    print("  SECTION 1: FIXED HOLD PERIODS")
    print("  hold=1 (baseline) vs hold=2 vs hold=3 vs hold=5")
    print("=" * 130)

    for hd in [1, 2, 3, 5]:
        for an_max in [10.0, 12.0]:
            for mc in [0.5, 0.7]:
                label = f"FIXED hold={hd}, atr<{an_max:.0f}%, corr={mc:.1f}"
                r = backtest(atr_norm_max=an_max, max_corr=mc,
                             dd_tiers=DD_AGGR100, regime_lo=0.5, regime_hi=1.5,
                             top_n=3, hold_mode='fixed', hold_days=hd)
                pr(r, label)
                all_results.append({**r, 'label': f'fixed_h{hd}_atr{an_max:.0f}_c{mc:.1f}',
                                    'section': 1, 'hold_mode': 'fixed', 'hold_days': hd,
                                    'atr_norm_max': an_max, 'max_corr': mc, 'top_n': 3,
                                    'profit_target': None, 'trail_pct': None})

    # ===================== SECTION 2: PROFIT-TAKING HOLD =====================
    print("\n" + "=" * 130)
    print("  SECTION 2: PROFIT-TAKING HOLD")
    print("  Hold up to N days, exit early if unrealized profit > X%")
    print("  profit_target=[2%, 3%, 5%] x max_hold=[2, 3, 5]")
    print("=" * 130)

    for pt in [2.0, 3.0, 5.0]:
        for mh in [2, 3, 5]:
            for an_max in [10.0, 12.0]:
                for mc in [0.5, 0.7]:
                    label = f"PROFIT_TAKE pt={pt:.0f}%, max_hold={mh}, atr<{an_max:.0f}%, corr={mc:.1f}"
                    r = backtest(atr_norm_max=an_max, max_corr=mc,
                                 dd_tiers=DD_AGGR100, regime_lo=0.5, regime_hi=1.5,
                                 top_n=3, hold_mode='profit_take', hold_days=mh,
                                 profit_target=pt)
                    pr(r, label)
                    all_results.append({**r,
                                        'label': f'pt{pt:.0f}_mh{mh}_atr{an_max:.0f}_c{mc:.1f}',
                                        'section': 2, 'hold_mode': 'profit_take',
                                        'hold_days': mh, 'atr_norm_max': an_max,
                                        'max_corr': mc, 'top_n': 3,
                                        'profit_target': pt, 'trail_pct': None})

    # ===================== SECTION 3: TRAILING EXIT =====================
    print("\n" + "=" * 130)
    print("  SECTION 3: TRAILING EXIT")
    print("  Hold until trailing stop hit or max_hold reached")
    print("  trail_pct=[1%, 1.5%, 2%] x max_hold=[3, 5]")
    print("=" * 130)

    for tp in [1.0, 1.5, 2.0]:
        for mh in [3, 5]:
            for an_max in [10.0, 12.0]:
                for mc in [0.5, 0.7]:
                    label = f"TRAILING trail={tp:.1f}%, max_hold={mh}, atr<{an_max:.0f}%, corr={mc:.1f}"
                    r = backtest(atr_norm_max=an_max, max_corr=mc,
                                 dd_tiers=DD_AGGR100, regime_lo=0.5, regime_hi=1.5,
                                 top_n=3, hold_mode='trailing', hold_days=mh,
                                 trail_pct=tp)
                    pr(r, label)
                    all_results.append({**r,
                                        'label': f'tr{tp:.1f}_mh{mh}_atr{an_max:.0f}_c{mc:.1f}',
                                        'section': 3, 'hold_mode': 'trailing',
                                        'hold_days': mh, 'atr_norm_max': an_max,
                                        'max_corr': mc, 'top_n': 3,
                                        'profit_target': None, 'trail_pct': tp})

    # ===================== SECTION 4: SIGNAL-STRENGTH HOLD =====================
    print("\n" + "=" * 130)
    print("  SECTION 4: SIGNAL-STRENGTH HOLD")
    print("  Stronger signals get longer hold:")
    print("    score >= P75: hold=3")
    print("    score >= P50: hold=2")
    print("    score <  P50: hold=1")
    print("  max_hold caps the ceiling")
    print("=" * 130)

    for mh in [3, 5]:
        for an_max in [10.0, 12.0]:
            for mc in [0.5, 0.7]:
                label = f"SIG_STR max_hold={mh}, atr<{an_max:.0f}%, corr={mc:.1f}"
                r = backtest(atr_norm_max=an_max, max_corr=mc,
                             dd_tiers=DD_AGGR100, regime_lo=0.5, regime_hi=1.5,
                             top_n=3, hold_mode='signal_strength', hold_days=mh)
                pr(r, label)
                all_results.append({**r,
                                    'label': f'ss_mh{mh}_atr{an_max:.0f}_c{mc:.1f}',
                                    'section': 4, 'hold_mode': 'signal_strength',
                                    'hold_days': mh, 'atr_norm_max': an_max,
                                    'max_corr': mc, 'top_n': 3,
                                    'profit_target': None, 'trail_pct': None})

    # ===================== SECTION 5: RANKED RESULTS (FULL PERIOD) =====================
    print("\n" + "=" * 130)
    print("  SECTION 5: ALL CONFIGS RANKED BY ANNUAL RETURN (full period)")
    print("=" * 130)

    all_results.sort(key=lambda x: -x['ann'])
    for i, r in enumerate(all_results[:40]):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i+1:2d} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d} | {r['label']}")

    print("\n" + "=" * 130)
    print("  SECTION 5b: ALL CONFIGS RANKED BY R/M RATIO (risk-adjusted)")
    print("=" * 130)

    all_rm = sorted(all_results, key=lambda x: -abs(x['ann']/x['mdd']) if x['mdd'] != 0 else 0)
    for i, r in enumerate(all_rm[:40]):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i+1:2d} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d} | {r['label']}")

    # ===================== SECTION 6: HOLD MODE COMPARISON =====================
    print("\n" + "=" * 130)
    print("  SECTION 6: HOLD MODE COMPARISON (atr<12%, corr=0.7)")
    print("=" * 130)

    # Best in each mode at atr<12%, corr=0.7
    mode_best = {}
    for r in all_results:
        if r['atr_norm_max'] != 12.0 or r['max_corr'] != 0.7:
            continue
        hm = r['hold_mode']
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        if hm not in mode_best or ratio > mode_best[hm][1]:
            mode_best[hm] = (r, ratio)

    for mode in ['fixed', 'profit_take', 'trailing', 'signal_strength']:
        if mode in mode_best:
            r, ratio = mode_best[mode]
            print(f"  {mode:20s}: Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | R/M={ratio:.2f} | N={r['n']:4d} | {r['label']}")

    # ===================== SECTION 7: HOLD DAYS COMPARISON TABLE =====================
    print("\n" + "=" * 130)
    print("  SECTION 7: HOLD DAYS COMPARISON (fixed mode, atr<12%, corr=0.7)")
    print("=" * 130)

    hold_compare = {}
    for r in all_results:
        if r['hold_mode'] == 'fixed' and r['atr_norm_max'] == 12.0 and r['max_corr'] == 0.7:
            hd = r['hold_days']
            hold_compare[hd] = r

    print(f"  {'Hold Days':>10s} {'Ann':>9s} {'MDD':>7s} {'R/M':>6s} {'Sharpe':>7s} {'N':>5s} {'WR':>6s}")
    print(f"  {'-'*55}")
    for hd in [1, 2, 3, 5]:
        if hd in hold_compare:
            r = hold_compare[hd]
            ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
            print(f"  {hd:10d} {r['ann']:+8.1f}% {r['mdd']:6.1f}% {ratio:6.2f} {r['sharpe']:7.2f} {r['n']:5d} {r['wr']:5.1f}%")

    # ===================== SECTION 8: WALK-FORWARD TOP 20 CONFIGS =====================
    print("\n" + "=" * 130)
    print("  SECTION 8: WALK-FORWARD VALIDATION — Top 20 by R/M")
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
    for r in wf_candidates[:20]:
        lbl = r['label']
        wf_kwargs = {
            'atr_norm_max': r['atr_norm_max'],
            'max_corr': r['max_corr'],
            'dd_tiers': DD_AGGR100,
            'regime_lo': 0.5, 'regime_hi': 1.5,
            'top_n': r['top_n'],
            'hold_mode': r['hold_mode'],
            'hold_days': r['hold_days'],
            'profit_target': r.get('profit_target', 3.0),
            'trail_pct': r.get('trail_pct', 1.5),
        }
        wf_res = walk_forward(label=lbl, **wf_kwargs)
        wf_all[lbl] = (wf_res, r)
        print_wf(wf_res, lbl)

    # ===================== SECTION 9: WF COMPARISON TABLE =====================
    print("\n" + "=" * 130)
    print("  SECTION 9: WF COMPARISON TABLE — All configs ranked by WF avg")
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

    # ===================== SECTION 10: BEST RISK-ADJUSTED (WF R/M) =====================
    print("\n" + "=" * 130)
    print("  SECTION 10: BEST RISK-ADJUSTED (WF R/M)")
    print("=" * 130)

    wf_ra = sorted(wf_ranked, key=lambda x: -x[6])
    for i, (avg_ann, wmdd, bann, pos, tn, awr, ratio, lbl, wf_res, ri) in enumerate(wf_ra[:15]):
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  #{i+1} WF AVG={avg_ann:+.0f}% | WorstMDD={wmdd:.0f}% | R/M={ratio:.2f} | {pos}/6 pos | {lbl}")
        print(f"     {ws}")

    # ===================== SECTION 11: BEST PER HOLD MODE (WF) =====================
    print("\n" + "=" * 130)
    print("  SECTION 11: BEST PER HOLD MODE (WF)")
    print("=" * 130)

    categories = {
        'Fixed': lambda x: x['hold_mode'] == 'fixed',
        'Profit-Take': lambda x: x['hold_mode'] == 'profit_take',
        'Trailing': lambda x: x['hold_mode'] == 'trailing',
        'Signal-Str': lambda x: x['hold_mode'] == 'signal_strength',
        'hold=1': lambda x: x.get('hold_days') == 1,
        'hold=2': lambda x: x.get('hold_days') == 2,
        'hold=3': lambda x: x.get('hold_days') == 3,
        'hold=5': lambda x: x.get('hold_days') == 5,
        'atr<10%': lambda x: x.get('atr_norm_max') == 10.0,
        'atr<12%': lambda x: x.get('atr_norm_max') == 12.0,
        'corr=0.5': lambda x: x.get('max_corr') == 0.5,
        'corr=0.7': lambda x: x.get('max_corr') == 0.7,
    }

    for cat_name, cat_filter in categories.items():
        cat_items = [(avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri)
                     for avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri in wf_ranked
                     if cat_filter(ri)]
        if not cat_items:
            print(f"\n  {cat_name:20s}: no WF results")
            continue
        cat_items.sort(key=lambda x: -x[6])
        best = cat_items[0]
        avg_ann, wmdd, bann, pos, tn, awr, ratio, lbl, wf_res, ri = best
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  {cat_name:20s} best: WF AVG={avg_ann:+.0f}% | WorstMDD={wmdd:.0f}% | R/M={ratio:.2f} | {lbl}")
        print(f"     {ws}")

    # ===================== SECTION 12: DELTA vs V169 BASELINE =====================
    print("\n" + "=" * 130)
    print("  SECTION 12: IMPROVEMENT vs V169 BASELINE (hold=1, atr<12%, corr=0.7)")
    print("=" * 130)

    base_lbl = 'base_atr12_c0.7'
    if base_lbl in wf_all:
        b_wf, b_ri = wf_all[base_lbl]
    else:
        b_wf = walk_forward(label=base_lbl,
                            atr_norm_max=12.0, max_corr=0.7,
                            dd_tiers=DD_AGGR100, regime_lo=0.5, regime_hi=1.5,
                            top_n=3, hold_mode='fixed', hold_days=1,
                            profit_target=3.0, trail_pct=1.5)
        print_wf(b_wf, "V169 BASELINE (computed)")
    b_avg = np.mean([r['ann'] for r in b_wf.values()])
    b_wmdd = min(r['mdd'] for r in b_wf.values())
    b_rm = abs(b_avg / b_wmdd) if b_wmdd != 0 else 0

    print(f"\n  V169 BASELINE: WF AVG={b_avg:+.0f}% | WorstMDD={b_wmdd:.0f}% | R/M={b_rm:.2f}")

    deltas = []
    for avg_ann, wmdd, bann, pos, tn, awr, ratio, lbl, wf_res, ri in wf_ranked:
        if lbl == base_lbl: continue
        delta_ann = avg_ann - b_avg
        delta_rm = ratio - b_rm
        deltas.append((delta_ann, delta_rm, ratio, avg_ann, wmdd, pos, lbl, wf_res))

    deltas.sort(key=lambda x: -x[1])
    print(f"\n  Configs by R/M improvement over V169 baseline:")
    for i, (da, drm, ratio, avg, wmdd, pos, lbl, wfr) in enumerate(deltas):
        marker = "*** IMPROVED" if drm > 0 else "    worse"
        print(f"  {i+1:2d} | R/M={ratio:.2f} (delta={drm:+.2f}) | Ann delta={da:+.0f}% | {pos}/6 | {marker} | {lbl}")

    # ===================== SECTION 13: BEST COMBINATION DETAIL =====================
    print("\n" + "=" * 130)
    print("  SECTION 13: BEST COMBINATION — Full detail for top 5 by WF R/M")
    print("=" * 130)

    for i, (avg_ann, wmdd, bann, pos, tn, awr, ratio, lbl, wf_res, ri) in enumerate(wf_ra[:5]):
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  #{i+1}: {lbl}")
        print(f"       WF AVG={avg_ann:+.0f}% | WorstMDD={wmdd:.0f}% | R/M={ratio:.2f} | {pos}/6 pos | N={tn}")
        print(f"       {ws}")
        for yr, r in sorted(wf_res.items()):
            print(f"         {yr}: Ann={r['ann']:+.1f}% | MDD={r['mdd']:.1f}% | WR={r['wr']:.1f}% | N={r['n']}")

    # ===================== SECTION 14: HOLD DAYS IMPACT ANALYSIS =====================
    print("\n" + "=" * 130)
    print("  SECTION 14: HOLD DAYS IMPACT — How hold period affects performance")
    print("=" * 130)

    # Collect results by hold_days for fixed mode only
    impact = {}
    for r in all_results:
        if r['hold_mode'] != 'fixed':
            continue
        key = r['hold_days']
        if key not in impact:
            impact[key] = []
        impact[key].append(r)

    print(f"\n  Fixed hold mode — averages across all atr/corr combos:")
    print(f"  {'Hold':>5s} {'Avg Ann':>9s} {'Avg MDD':>9s} {'Avg R/M':>9s} {'Avg Sh':>8s} {'Avg N':>7s} {'Avg WR':>8s}")
    print(f"  {'-'*60}")
    for hd in [1, 2, 3, 5]:
        if hd in impact:
            items = impact[hd]
            avg_ann = np.mean([r['ann'] for r in items])
            avg_mdd = np.mean([r['mdd'] for r in items])
            avg_rm = np.mean([abs(r['ann']/r['mdd']) if r['mdd'] != 0 else 0 for r in items])
            avg_sh = np.mean([r['sharpe'] for r in items])
            avg_n = np.mean([r['n'] for r in items])
            avg_wr = np.mean([r['wr'] for r in items])
            print(f"  {hd:5d} {avg_ann:+8.1f}% {avg_mdd:8.1f}% {avg_rm:9.2f} {avg_sh:8.2f} {avg_n:7.0f} {avg_wr:7.1f}%")

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 130)
    print("  FINAL SUMMARY")
    print("=" * 130)

    print(f"\n  V169 Baseline (hold=1, atr<12%, corr=0.7, top_n=3):")
    print(f"    WF AVG={b_avg:+.0f}% | WorstMDD={b_wmdd:.0f}% | R/M={b_rm:.2f}")

    if wf_ra:
        best = wf_ra[0]
        print(f"\n  Best overall: {best[7]}")
        print(f"    WF AVG={best[0]:+.0f}% | WorstMDD={best[1]:.0f}% | R/M={best[6]:.2f}")
        delta_rm_best = best[6] - b_rm
        print(f"    R/M improvement over V169: {delta_rm_best:+.2f}")

        # Best by hold mode
        print(f"\n  Best by hold mode (WF R/M):")
        for mode in ['fixed', 'profit_take', 'trailing', 'signal_strength']:
            mode_items = [(avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri)
                          for avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri in wf_ranked
                          if ri['hold_mode'] == mode]
            if not mode_items: continue
            mode_items.sort(key=lambda x: -x[6])
            s = mode_items[0]
            mode_names = {'fixed': 'Fixed Hold', 'profit_take': 'Profit-Take',
                          'trailing': 'Trailing Stop', 'signal_strength': 'Signal Strength'}
            print(f"    {mode_names.get(mode, mode):20s}: {s[7]} | R/M={s[6]:.2f} | WF={s[0]:+.0f}%")

        # Best by section
        print(f"\n  Best by section (WF R/M):")
        sections_tested = set(ri.get('section') for _, _, _, _, _, _, _, _, _, ri in wf_ranked)
        for sec in sorted(sections_tested):
            sec_items = [(avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri)
                         for avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri in wf_ranked
                         if ri.get('section') == sec]
            if not sec_items: continue
            sec_items.sort(key=lambda x: -x[6])
            s = sec_items[0]
            sec_names = {0: "Baseline", 1: "Fixed Hold", 2: "Profit-Take",
                         3: "Trailing", 4: "Signal Strength"}
            sec_name = sec_names.get(sec, f"Section {sec}")
            print(f"    Sec {sec} ({sec_name:20s}): {s[7]} | R/M={s[6]:.2f} | WF={s[0]:+.0f}%")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
