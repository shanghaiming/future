"""
Alpha Futures V152 — Volatility-Targeted Position Sizing
==============================================================================
Goal: Use ATR to normalize position sizes so each position contributes equal
risk. Test four approaches:
  1. ATR-normalized lots: lots = capital / (ATR * MULT * target_risk)
  2. Portfolio vol targeting: target specific daily portfolio volatility
  3. Inverse-vol weighting: allocate capital inversely proportional to ATR%
  4. Dynamic vol target: reduce sizes when market vol is high

Signal portfolio: V121 + Union, balanced with corr filter.
Walk-forward validation for top configs.
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
    print("  V152 — Volatility-Targeted Position Sizing")
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

    # ATR% = ATR / Close — used for inverse-vol weighting
    ATR_PCT = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            atr = ATR14[si, di]
            cp = C[si, di]
            if not np.isnan(atr) and not np.isnan(cp) and cp > 0:
                ATR_PCT[si, di] = atr / cp

    # Market Vol: 20-day rolling std of equal-weighted market return
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
    VOL_P75 = np.percentile(valid_vols, 75) if len(valid_vols) > 0 else VOL_MEDIAN * 1.5
    VOL_P25 = np.percentile(valid_vols, 25) if len(valid_vols) > 0 else VOL_MEDIAN * 0.5
    print(f"  Market vol: median={VOL_MEDIAN:.4f}%, P25={VOL_P25:.4f}%, P75={VOL_P75:.4f}%")

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

    # ===================== HELPER: Correlation (daily returns) =====================
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

    # ===================== SIGNAL SELECTION (balanced portfolio) =====================
    def select_signals(di, edi, held_si, top_n, max_corr=0.5):
        """
        Balanced: top half from V121, rest from Union, with corr filter.
        Returns list of (score, si, entry_price, sig_label).
        """
        cands_v121 = sig_v121(di, edi)
        cands_union = sig_union(di, edi)
        cands_v121.sort(key=lambda x: -x[0])
        cands_union.sort(key=lambda x: -x[0])

        selected = []
        used_si = set(held_si)

        # Fill top half from V121
        v121_quota = max(1, top_n // 2)
        for sc, s, ep, st in cands_v121:
            if len(selected) >= v121_quota: break
            if s in used_si: continue
            # Check correlation with already selected
            ok = True
            for _, sel_s, _, _ in selected:
                if get_corr(s, sel_s, di) >= max_corr:
                    ok = False; break
            if ok:
                selected.append((sc, s, ep, 'v121'))
                used_si.add(s)

        # Fill rest from Union
        for sc, s, ep, st in cands_union:
            if len(selected) >= top_n: break
            if s in used_si: continue
            ok = True
            for _, sel_s, _, _ in selected:
                if get_corr(s, sel_s, di) >= max_corr:
                    ok = False; break
            if ok:
                selected.append((sc, s, ep, 'union'))
                used_si.add(s)

        return selected

    # ===================== BACKTEST ENGINE =====================
    def backtest(start_di=MIN_TRAIN, end_di=None,
                 # Sizing mode
                 sizing_mode='equal_dollar',
                 # Equal dollar params
                 base_size=0.55,
                 # ATR-normalized params
                 atr_target_risk=1.0,
                 # Portfolio vol targeting params
                 daily_vol_target=0.02,
                 # Inverse-vol params (uses base_size for total allocation)
                 # Dynamic vol target params
                 vol_reduce_threshold=1.5,  # reduce when MKT_VOL > VOL_MEDIAN * this
                 vol_reduce_factor=0.5,
                 # General
                 hold=1, top_n=2, max_corr=0.5,
                 # DD tiers
                 dd_tiers=None,
                 use_dd=False):
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

            # Skip if slots full
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue

            held_si = set(p['si'] for p in positions)

            # Select signals
            signals = select_signals(di, edi, held_si, top_n, max_corr)
            if not signals: continue

            # --- CRITICAL: snapshot cash before entry loop ---
            cash_snapshot = cash

            # --- Compute per-position sizing ---
            # Step 1: base allocation fraction
            if use_dd:
                size_frac = dd_size(pv, high_water, dd_tiers)
            else:
                size_frac = base_size

            # Dynamic vol reduction
            if sizing_mode == 'dynamic_vol':
                vol = MKT_VOL[di]
                if not np.isnan(vol) and VOL_MEDIAN > 0:
                    if vol / VOL_MEDIAN > vol_reduce_threshold:
                        size_frac *= vol_reduce_factor

            size_frac = max(0.05, min(0.95, size_frac))

            # Step 2: allocate capital per position
            n_planned = len(signals)
            total_cap = cash_snapshot * size_frac

            if sizing_mode == 'equal_dollar':
                # Equal dollar allocation across positions
                for sc, s, pr, sig_str in signals:
                    if s in set(p['si'] for p in positions): continue
                    if len(positions) >= top_n: break
                    cap = total_cap / n_planned
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

            elif sizing_mode == 'atr_normalized':
                # ATR-normalized lots: each position gets equal dollar volatility
                # lots_i = cap_i / (ATR_i * MULT_i * target_risk)
                # where cap_i = total_cap / n_planned (equal risk budget per position)
                for sc, s, pr, sig_str in signals:
                    if s in set(p['si'] for p in positions): continue
                    if len(positions) >= top_n: break
                    atr = ATR14[s, di]
                    if np.isnan(atr) or atr <= 0: continue
                    sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                    cap = total_cap / n_planned
                    # lots = cap / (ATR * MULT * target_risk)
                    atr_risk_unit = atr * m * atr_target_risk
                    if atr_risk_unit <= 0: continue
                    ct_atr = max(1, int(cap / atr_risk_unit))
                    # Cap by per-position capital budget (the intended allocation)
                    ct_cap = max(1, int(cap / (pr * m * (1 + COMM))))
                    # Use the SMALLER of the two (ATR gives fewer lots for high-vol,
                    # cap gives fewer for low-vol)
                    ct = min(ct_atr, ct_cap)
                    # Verify we can afford it
                    ci = pr * m * ct * (1 + COMM)
                    if ci > cash:
                        ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                        ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                    if ct <= 0 or ci <= 0 or ci > cash: continue
                    cash -= ci
                    positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                      'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': hold,
                                      'sig': sig_str, 'score': sc})

            elif sizing_mode == 'vol_target':
                # Portfolio vol targeting: adjust total position to target daily vol
                # For each candidate, estimate its daily vol = ATR% * price * MULT * lots
                # Target: total daily dollar vol = pv * daily_vol_target
                # Distribute equally among candidates
                target_dollar_vol = pv * daily_vol_target
                per_pos_vol = target_dollar_vol / n_planned if n_planned > 0 else 0

                for sc, s, pr, sig_str in signals:
                    if s in set(p['si'] for p in positions): continue
                    if len(positions) >= top_n: break
                    atr = ATR14[s, di]
                    if np.isnan(atr) or atr <= 0: continue
                    sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                    # lots = per_pos_vol / (ATR * MULT)
                    ct = max(1, int(per_pos_vol / (atr * m)))
                    # Clamp by total cap
                    cap = total_cap / n_planned
                    max_ct_by_cap = max(1, int(cap / (pr * m * (1 + COMM))))
                    ct = min(ct, max_ct_by_cap)
                    ci = pr * m * ct * (1 + COMM)
                    if ci > cash:
                        ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                        ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                    if ct <= 0 or ci <= 0 or ci > cash: continue
                    cash -= ci
                    positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                      'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': hold,
                                      'sig': sig_str, 'score': sc})

            elif sizing_mode == 'inv_vol':
                # Inverse-vol weighting: allocate capital inversely proportional to ATR%
                # Lower vol instruments get more capital
                atr_pcts = []
                valid_signals = []
                for sc, s, pr, sig_str in signals:
                    atr_pct = ATR_PCT[s, di]
                    if np.isnan(atr_pct) or atr_pct <= 0: continue
                    atr_pcts.append(atr_pct)
                    valid_signals.append((sc, s, pr, sig_str))

                if not valid_signals: continue

                # Inverse weights
                inv_weights = [1.0 / ap for ap in atr_pcts]
                total_inv = sum(inv_weights)
                if total_inv <= 0: continue
                norm_weights = [w / total_inv for w in inv_weights]

                for idx, (sc, s, pr, sig_str) in enumerate(valid_signals):
                    if s in set(p['si'] for p in positions): continue
                    if len(positions) >= top_n: break
                    cap = total_cap * norm_weights[idx]
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

            elif sizing_mode == 'dynamic_vol':
                # Dynamic vol target: same as equal_dollar but with vol-based reduction
                for sc, s, pr, sig_str in signals:
                    if s in set(p['si'] for p in positions): continue
                    if len(positions) >= top_n: break
                    cap = total_cap / n_planned
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

    def walk_forward(**kwargs):
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
        print(f"    {label:75s}")
        print(f"      {pos}/6 pos | Avg={avg_ann:>+7.0f}% | WorstWfMDD={worst_mdd:>5.0f}%")
        print(f"      {ws}")

    # Collect all results for ranking
    all_results = []

    # ===================== SECTION 0: BASELINES =====================
    print("\n" + "=" * 130)
    print("  SECTION 0: BASELINES (equal dollar allocation)")
    print("=" * 130)

    for bs in [0.45, 0.50, 0.55, 0.60, 0.70]:
        for tn in [2, 3]:
            r = backtest(sizing_mode='equal_dollar', base_size=bs, top_n=tn, hold=1, max_corr=0.5)
            label = f"S0: Equal$ size={bs*100:.0f}% top_n={tn} corr<0.5"
            r['desc'] = label; r['params'] = {'sizing_mode': 'equal_dollar', 'base_size': bs, 'top_n': tn}
            all_results.append(r)
            pr(r, label)

    # ===================== SECTION 1: ATR-NORMALIZED LOT SIZING =====================
    print("\n" + "=" * 130)
    print("  SECTION 1: ATR-NORMALIZED LOT SIZING")
    print("  lots = capital / (ATR * MULT * target_risk)")
    print("=" * 130)

    for tr in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        for tn in [2, 3, 4, 5]:
            r = backtest(sizing_mode='atr_normalized', atr_target_risk=tr,
                         top_n=tn, hold=1, max_corr=0.5)
            label = f"S1: ATR-norm risk={tr} top_n={tn}"
            r['desc'] = label; r['params'] = {'sizing_mode': 'atr_normalized', 'atr_target_risk': tr, 'top_n': tn}
            all_results.append(r)
            pr(r, label)

    # ===================== SECTION 2: PORTFOLIO VOL TARGETING =====================
    print("\n" + "=" * 130)
    print("  SECTION 2: PORTFOLIO VOL TARGETING")
    print("  Target specific daily portfolio volatility")
    print("=" * 130)

    for dvt in [0.01, 0.015, 0.02, 0.025, 0.03]:
        for tn in [2, 3, 4]:
            for bs in [0.50, 0.70]:
                r = backtest(sizing_mode='vol_target', daily_vol_target=dvt,
                             base_size=bs, top_n=tn, hold=1, max_corr=0.5)
                label = f"S2: VolTarget {dvt*100:.1f}% size={bs*100:.0f}% top_n={tn}"
                r['desc'] = label
                r['params'] = {'sizing_mode': 'vol_target', 'daily_vol_target': dvt, 'base_size': bs, 'top_n': tn}
                all_results.append(r)
                pr(r, label)

    # ===================== SECTION 3: INVERSE-VOL WEIGHTING =====================
    print("\n" + "=" * 130)
    print("  SECTION 3: INVERSE-VOL WEIGHTING")
    print("  Allocate capital inversely proportional to ATR%")
    print("=" * 130)

    for bs in [0.50, 0.55, 0.60, 0.70]:
        for tn in [2, 3, 4]:
            r = backtest(sizing_mode='inv_vol', base_size=bs,
                         top_n=tn, hold=1, max_corr=0.5)
            label = f"S3: InvVol size={bs*100:.0f}% top_n={tn}"
            r['desc'] = label; r['params'] = {'sizing_mode': 'inv_vol', 'base_size': bs, 'top_n': tn}
            all_results.append(r)
            pr(r, label)

    # ===================== SECTION 4: VOL TARGETING + DD TIERS =====================
    print("\n" + "=" * 130)
    print("  SECTION 4: VOL TARGETING + DD TIERS")
    print("  Dynamic vol reduction when market vol is high + DD-based sizing")
    print("=" * 130)

    dd_configs = [
        ([(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)], "DD70/60/40/20"),
        ([(0, 0.60), (0.10, 0.50), (0.20, 0.35), (0.30, 0.15)], "DD60/50/35/15"),
    ]

    for dd_t, dd_label in dd_configs:
        # ATR-normalized + DD
        for tr in [1.0, 2.0, 3.0]:
            for tn in [2, 3, 4]:
                r = backtest(sizing_mode='atr_normalized', atr_target_risk=tr,
                             top_n=tn, hold=1, max_corr=0.5,
                             use_dd=True, dd_tiers=dd_t)
                label = f"S4: ATR-norm risk={tr} {dd_label} top_n={tn}"
                r['desc'] = label
                r['params'] = {'sizing_mode': 'atr_normalized', 'atr_target_risk': tr,
                               'top_n': tn, 'use_dd': True, 'dd_tiers': dd_t}
                all_results.append(r)
                pr(r, label)

        # Dynamic vol + DD
        for vrt in [1.2, 1.5, 2.0]:
            for bs in [0.50, 0.55, 0.70]:
                for tn in [2, 3]:
                    r = backtest(sizing_mode='dynamic_vol', base_size=bs,
                                 top_n=tn, hold=1, max_corr=0.5,
                                 vol_reduce_threshold=vrt, vol_reduce_factor=0.5,
                                 use_dd=True, dd_tiers=dd_t)
                    label = f"S4: DynVol vrt={vrt} {dd_label} size={bs*100:.0f}% top_n={tn}"
                    r['desc'] = label
                    r['params'] = {'sizing_mode': 'dynamic_vol', 'base_size': bs, 'top_n': tn,
                                   'vol_reduce_threshold': vrt, 'vol_reduce_factor': 0.5,
                                   'use_dd': True, 'dd_tiers': dd_t}
                    all_results.append(r)
                    pr(r, label)

        # Inv vol + DD
        for bs in [0.50, 0.60, 0.70]:
            for tn in [2, 3]:
                r = backtest(sizing_mode='inv_vol', base_size=bs,
                             top_n=tn, hold=1, max_corr=0.5,
                             use_dd=True, dd_tiers=dd_t)
                label = f"S4: InvVol+{dd_label} size={bs*100:.0f}% top_n={tn}"
                r['desc'] = label
                r['params'] = {'sizing_mode': 'inv_vol', 'base_size': bs, 'top_n': tn,
                               'use_dd': True, 'dd_tiers': dd_t}
                all_results.append(r)
                pr(r, label)

    # ===================== COMPREHENSIVE RANKING =====================
    print("\n" + "=" * 130)
    print("  COMPREHENSIVE RANKING (full period)")
    print("=" * 130)

    all_valid = [r for r in all_results if r.get('desc', '') and r['mdd'] > -80]

    # By annual return
    all_valid.sort(key=lambda x: -x['ann'])
    print(f"\n  Top 20 by Annual Return:")
    for i, r in enumerate(all_valid[:20]):
        desc = r.get('desc', '')
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i+1:2d}: {desc:75s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    # By return/MDD ratio
    all_with_ratio = [(r, abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0) for r in all_valid]
    all_with_ratio.sort(key=lambda x: -x[1])
    print(f"\n  Top 20 by Ann/MDD Ratio:")
    for i, (r, ratio) in enumerate(all_with_ratio[:20]):
        desc = r.get('desc', '')
        print(f"  #{i+1:2d}: {desc:75s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    # By Sharpe
    all_valid_sh = list(all_valid)
    all_valid_sh.sort(key=lambda x: -x['sharpe'])
    print(f"\n  Top 15 by Sharpe:")
    for i, r in enumerate(all_valid_sh[:15]):
        desc = r.get('desc', '')
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i+1:2d}: {desc:75s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    # ===================== SECTION 5: WALK-FORWARD VALIDATION =====================
    print("\n" + "=" * 130)
    print("  SECTION 5: WALK-FORWARD VALIDATION FOR TOP 10 CONFIGS")
    print("=" * 130)

    # Select top 10 unique configs by ratio (mix of different approaches)
    seen = set()
    wf_configs = []
    # Prioritize diversity: pick top by ratio from each section
    sections = {'S0': [], 'S1': [], 'S2': [], 'S3': [], 'S4': []}
    for r, ratio in all_with_ratio:
        desc = r.get('desc', '')
        if desc.startswith('S0:'): sections['S0'].append((r, ratio))
        elif desc.startswith('S1:'): sections['S1'].append((r, ratio))
        elif desc.startswith('S2:'): sections['S2'].append((r, ratio))
        elif desc.startswith('S3:'): sections['S3'].append((r, ratio))
        elif desc.startswith('S4:'): sections['S4'].append((r, ratio))

    # Pick top 2 from each section
    for sec_key, sec_results in sections.items():
        for r, ratio in sec_results[:2]:
            desc = r.get('desc', '')
            if desc not in seen:
                seen.add(desc)
                wf_configs.append(r)
    # Fill remaining from overall ranking
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
        params = r.get('params', {})
        wf_res = walk_forward(**params)
        wf_all[desc] = wf_res
        print_wf(wf_res, desc)

    # ===================== FINAL RANKING BY WF PERFORMANCE =====================
    print("\n" + "=" * 130)
    print("  FINAL RANKING: WF Average Return (WF MDD < -30% filter)")
    print("=" * 130)

    wf_ranking = []
    for desc, wf_res in wf_all.items():
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        pos_years = sum(1 for r in wf_res.values() if r['ann'] > 0)
        avg_wr = np.mean([r['wr'] for r in wf_res.values()])
        avg_sh = np.mean([r['sharpe'] for r in wf_res.values()])
        wf_ranking.append({
            'desc': desc, 'avg_ann': avg_ann, 'worst_mdd': worst_mdd,
            'pos_years': pos_years, 'avg_wr': avg_wr, 'avg_sh': avg_sh,
            'wf': wf_res
        })

    # All configs (no filter)
    wf_ranking.sort(key=lambda x: -x['avg_ann'])
    print(f"\n  All WF configs sorted by avg WF return:")
    print(f"  {'#':>3s}  {'Config':75s} | {'AvgWF':>7s} | {'WfMDD':>6s} | {'WR':>5s} | {'Sh':>5s} | {'Pos':>3s}")
    print(f"  {'---':>3s}  {'-'*75}-+-{'-'*7}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*3}")
    for i, w in enumerate(wf_ranking):
        marker = " ***" if w['worst_mdd'] > -30 else ""
        print(f"  {i+1:3d}{marker:4s} {w['desc']:75s} | {w['avg_ann']:>+6.0f}% | {w['worst_mdd']:>5.1f}% | {w['avg_wr']:>4.0f}% | {w['avg_sh']:>4.2f} | {w['pos_years']}/6")
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(w['wf'].items())])
        print(f"       {ws}")

    # Filtered: only WF MDD < -30%
    filtered = [w for w in wf_ranking if w['worst_mdd'] > -30]
    filtered.sort(key=lambda x: -x['avg_ann'])

    print(f"\n  TOP 3 by WF average return (WF MDD < -30% filter):")
    print(f"  {'='*120}")
    if filtered:
        for i, w in enumerate(filtered[:3]):
            print(f"\n  #{i+1}: {w['desc']}")
            print(f"       AvgWF={w['avg_ann']:>+7.0f}% | WorstWfMDD={w['worst_mdd']:>5.1f}% | WR={w['avg_wr']:.0f}% | Sh={w['avg_sh']:.2f} | {w['pos_years']}/6 positive")
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(w['wf'].items())])
            print(f"       {ws}")
    else:
        print("  No configs pass WF MDD < -30% filter. Showing top 3 by avg WF return regardless:")
        for i, w in enumerate(wf_ranking[:3]):
            print(f"\n  #{i+1}: {w['desc']}")
            print(f"       AvgWF={w['avg_ann']:>+7.0f}% | WorstWfMDD={w['worst_mdd']:>5.1f}% | WR={w['avg_wr']:.0f}% | Sh={w['avg_sh']:.2f} | {w['pos_years']}/6 positive")
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(w['wf'].items())])
            print(f"       {ws}")

    # Also show full-period stats for the WF top 3
    print(f"\n  Full-period stats for WF top 3:")
    for i, w in enumerate(filtered[:3] if filtered else wf_ranking[:3]):
        desc = w['desc']
        # Find the matching full-period result
        for r in all_results:
            if r.get('desc', '') == desc:
                pr(r, f"  #{i+1} full-period: {desc}")
                break

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
