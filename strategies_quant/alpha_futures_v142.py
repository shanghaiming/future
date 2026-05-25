"""
Alpha Futures V142 — MULTI-POSITION & DIVERSIFICATION
=============================================================================
Goal: Improve return/MDD ratio via simultaneous position holding.

Approaches tested:
  A) Multi-position (top_n=2,3,4) with proportional sizing
  B) Signal agreement = larger position (V121 + Union agree -> 80%)
  C) Correlation-based pair selection (lowest 20-day correlation)
  D) Cross-signal diversification (1 V121 + 1 Union simultaneously)

Each approach:
  1. Full-period backtest: annual return, MDD, Sharpe
  2. Per-year walk-forward (2020-2025) with per-year MDD
  3. Sorted by return/MDD ratio
"""
import sys, os, time, warnings
import numpy as np
import talib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

MULT = {'agfi': 15, 'alfi': 5, 'aufi': 1000, 'bufi': 10, 'cufi': 5, 'fufi': 10,
        'rbfi': 10, 'znfi': 5, 'nifi': 1, 'hcfi': 10, 'spfi': 10, 'ssfi': 5,
        'sffi': 5, 'smfi': 5, 'pbfi': 5, 'snfi': 1, 'rufi': 10, 'wrfff': 10,
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
    print("=" * 120)
    print("  V142 — MULTI-POSITION & DIVERSIFICATION")
    print("=" * 120)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    print(f"  {NS} commodities, {ND} days")

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

    # Precompute 20-day returns for correlation calculation
    RET20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(20, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-20]) and c[di-20] > 0:
                RET20[si, di] = (c[di] / c[di-20] - 1) * 100

    print(f"  Done ({time.time()-t0:.1f}s)")

    # ===================== SIGNAL DEFINITIONS (from V140) =====================
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

    # ===================== GENERIC BACKTEST ENGINE =====================
    def backtest(signal_func, hold=1, top_n=1, start_di=MIN_TRAIN, end_di=None,
                 size_frac=0.95, per_pos_frac=None):
        """
        Multi-position backtest.
        per_pos_frac: if set, each position gets this fraction of capital (overrides top_n split).
        """
        if end_di is None: end_di = ND
        cash = float(CASH0)
        positions = []
        daily_eq = []

        for di in range(start_di, end_di - 1):
            # Mark-to-market
            pv = cash
            for p in positions:
                cp = C[p['si'], di]
                if not np.isnan(cp) and cp > 0:
                    m = MULT.get(p['sym'], DEF_MULT)
                    pv += cp * m * p['lots'] - cp * m * abs(p['lots']) * COMM
            daily_eq.append(pv)

            # Close positions past hold period
            cl = []
            for p in positions:
                if di - p['entry_di'] >= p['hold_days']:
                    ep = C[p['si'], di]
                    if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                    m = MULT.get(p['sym'], DEF_MULT)
                    cash += ep * m * abs(p['lots']) * (1 - COMM)
                    cl.append(p)
            for p in cl: positions.remove(p)

            # Open new positions
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue
            cands = signal_func(di, edi)
            if not cands: continue
            cands.sort(key=lambda x: -x[0])

            # Filter out commodities already held
            held_si = set(p['si'] for p in positions)
            cands = [c for c in cands if c[1] not in held_si]
            if not cands: continue

            ns = top_n - len(positions)
            for item in cands[:ns]:
                if len(item) == 3: sc, s, pr = item; sig = ''
                else: sc, s, pr, sig = item

                if per_pos_frac is not None:
                    cap = cash * per_pos_frac
                else:
                    cap = cash * size_frac / max(1, ns)

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
                                  'sig': sig, 'score': sc})

        # Close remaining
        for p in positions:
            ae = end_di - 1
            ep = C[p['si'], min(ae, ND-1)]
            if np.isnan(ep) or ep <= 0: ep = p['entry_price']
            m = MULT.get(p['sym'], DEF_MULT)
            cash += ep * m * abs(p['lots']) * (1 - COMM)

        nd = end_di - start_di
        ann = annual_return(cash, CASH0, nd)
        if daily_eq:
            eq = np.array(daily_eq); pk = np.maximum.accumulate(eq)
            mdd = np.min((eq - pk) / pk * 100)
            r = np.diff(eq) / eq[:-1]
            r = np.where(np.isfinite(r), r, 0)
            sh = np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0
        else:
            mdd = 0; sh = 0
        return {'ann': ann, 'mdd': mdd, 'sharpe': sh, 'final': cash, 'eq': np.array(daily_eq) if daily_eq else np.array([])}

    def pr(r, label=""):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  {label:70s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    def wf(signal_func, hold=1, top_n=1, size_frac=0.95, per_pos_frac=None, label=""):
        """Walk-forward per year, returns dict of year -> result dict."""
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue
            r = backtest(signal_func, hold=hold, top_n=top_n,
                         start_di=ys, end_di=ye,
                         size_frac=size_frac, per_pos_frac=per_pos_frac)
            res[yr] = r
        return res

    def print_wf(wf_res, label=""):
        pos = sum(1 for r in wf_res.values() if r['ann'] > 0)
        avg = np.mean([r['ann'] for r in wf_res.values()]) if wf_res else 0
        worst_mdd = min(r['mdd'] for r in wf_res.values()) if wf_res else 0
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"  {label}")
        print(f"    WF: {pos}/6 pos | Avg={avg:>+.0f}% | WorstYrMDD={worst_mdd:.1f}%")
        print(f"    {ws}")

    # ===================== SECTION 1: BASELINE (top_n=1, 50% sizing) =====================
    print("\n" + "=" * 120)
    print("  SECTION 1: BASELINES (top_n=1, 50% sizing)")
    print("=" * 120)

    r_v121_base = backtest(sig_v121, hold=1, top_n=1, size_frac=0.50)
    pr(r_v121_base, "V121 baseline top_n=1 @50%")

    r_union_base = backtest(sig_union, hold=1, top_n=1, size_frac=0.50)
    pr(r_union_base, "Union baseline top_n=1 @50%")

    # ===================== SECTION 2: MULTI-POSITION (Approach A) =====================
    print("\n" + "=" * 120)
    print("  SECTION 2: MULTI-POSITION (Approach A)")
    print("  top_n=2: 45% each | top_n=3: 30% each | top_n=4: 22% each")
    print("=" * 120)

    multi_configs = [
        # (signal, top_n, per_pos_frac, label)
        (sig_v121, 1, 0.50, "V121 top_n=1 @50% (ref)"),
        (sig_v121, 2, 0.45, "V121 top_n=2 @45%ea"),
        (sig_v121, 3, 0.30, "V121 top_n=3 @30%ea"),
        (sig_v121, 4, 0.22, "V121 top_n=4 @22%ea"),
        (sig_union, 1, 0.50, "Union top_n=1 @50% (ref)"),
        (sig_union, 2, 0.45, "Union top_n=2 @45%ea"),
        (sig_union, 3, 0.30, "Union top_n=3 @30%ea"),
        (sig_union, 4, 0.22, "Union top_n=4 @22%ea"),
    ]

    multi_results = []
    for sig_func, topn, ppf, label in multi_configs:
        r = backtest(sig_func, hold=1, top_n=topn, per_pos_frac=ppf)
        r['desc'] = label
        r['config'] = ('multi', sig_func, topn, ppf)
        multi_results.append(r)
        pr(r, label)

    # Walk-forward for multi-position
    print("\n  Walk-Forward for multi-position configs:")
    for sig_func, topn, ppf, label in multi_configs:
        wf_res = wf(sig_func, hold=1, top_n=topn, per_pos_frac=ppf, label=label)
        print_wf(wf_res, label)

    # ===================== SECTION 3: SIGNAL AGREEMENT (Approach B) =====================
    print("\n" + "=" * 120)
    print("  SECTION 3: SIGNAL AGREEMENT (Approach B)")
    print("  Both agree on same commodity -> 80% | One fires -> 50% | Disagree -> 25%")
    print("=" * 120)

    def backtest_agreement(start_di=MIN_TRAIN, end_di=None):
        """
        When V121 AND Union agree on same commodity -> 80% position.
        When only one fires -> 50% position.
        When they disagree (both fire, different commodities) -> 25% each (hold both).
        """
        if end_di is None: end_di = ND
        cash = float(CASH0)
        positions = []
        daily_eq = []

        for di in range(start_di, end_di - 1):
            pv = cash
            for p in positions:
                cp = C[p['si'], di]
                if not np.isnan(cp) and cp > 0:
                    m = MULT.get(p['sym'], DEF_MULT)
                    pv += cp * m * p['lots'] - cp * m * abs(p['lots']) * COMM
            daily_eq.append(pv)

            # Close positions past hold period
            cl = []
            for p in positions:
                if di - p['entry_di'] >= p['hold_days']:
                    ep = C[p['si'], di]
                    if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                    m = MULT.get(p['sym'], DEF_MULT)
                    cash += ep * m * abs(p['lots']) * (1 - COMM)
                    cl.append(p)
            for p in cl: positions.remove(p)

            edi = di + 1
            if edi >= end_di: continue

            # Get signals from both
            cands_v121 = sig_v121(di, edi)
            cands_union = sig_union(di, edi)

            # Extract commodity indices
            v121_syms = {item[1]: item for item in cands_v121}
            union_syms = {item[1]: item for item in cands_union}

            # Find agreement
            agreed = set(v121_syms.keys()) & set(union_syms.keys())
            v121_only = set(v121_syms.keys()) - set(union_syms.keys())
            union_only = set(union_syms.keys()) - set(v121_syms.keys())

            held_si = set(p['si'] for p in positions)

            # Priority 1: Agreement -> 80% position
            if agreed:
                best_agreed = max(agreed, key=lambda s: (v121_syms[s][0] + union_syms[s][0]))
                if best_agreed not in held_si and len(positions) < 2:
                    item = v121_syms[best_agreed]
                    sc, s, pr = item[0], item[1], item[2]
                    sig_str = 'agree'
                    cap = cash * 0.80
                    sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                    ct = max(1, int(cap / (pr * m * (1 + COMM))))
                    ci = pr * m * ct * (1 + COMM)
                    if ci > cash:
                        ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                        ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                    if ct > 0 and ci > 0 and ci <= cash:
                        cash -= ci
                        positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                          'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': 1,
                                          'sig': sig_str, 'score': sc})
                        held_si.add(s)
                continue  # agreement takes priority, skip rest

            # Priority 2: Only one signal fires -> 50%
            if v121_only and not union_only:
                if len(positions) < 1:
                    cands_sorted = sorted(cands_v121, key=lambda x: -x[0])
                    for item in cands_sorted:
                        sc, s, pr = item[0], item[1], item[2]
                        if s in held_si: continue
                        cap = cash * 0.50
                        sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                        ct = max(1, int(cap / (pr * m * (1 + COMM))))
                        ci = pr * m * ct * (1 + COMM)
                        if ci > cash:
                            ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                            ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                        if ct > 0 and ci > 0 and ci <= cash:
                            cash -= ci
                            positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                              'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': 1,
                                              'sig': 'v121_only', 'score': sc})
                        break
                continue

            if union_only and not v121_only:
                if len(positions) < 1:
                    cands_sorted = sorted(cands_union, key=lambda x: -x[0])
                    for item in cands_sorted:
                        sc, s, pr = item[0], item[1], item[2]
                        if s in held_si: continue
                        cap = cash * 0.50
                        sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                        ct = max(1, int(cap / (pr * m * (1 + COMM))))
                        ci = pr * m * ct * (1 + COMM)
                        if ci > cash:
                            ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                            ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                        if ct > 0 and ci > 0 and ci <= cash:
                            cash -= ci
                            positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                              'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': 1,
                                              'sig': 'union_only', 'score': sc})
                        break
                continue

            # Priority 3: Both fire, different commodities -> 25% each (hold both)
            if v121_only and union_only:
                n_entered = 0
                for item in sorted(cands_v121, key=lambda x: -x[0]):
                    if n_entered >= 2 or len(positions) + n_entered >= 2: break
                    sc, s, pr = item[0], item[1], item[2]
                    if s in held_si: continue
                    cap = cash * 0.25
                    sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                    ct = max(1, int(cap / (pr * m * (1 + COMM))))
                    ci = pr * m * ct * (1 + COMM)
                    if ci > cash:
                        ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                        ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                    if ct > 0 and ci > 0 and ci <= cash:
                        cash -= ci
                        positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                          'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': 1,
                                          'sig': 'v121_disagree', 'score': sc})
                        held_si.add(s)
                        n_entered += 1

                for item in sorted(cands_union, key=lambda x: -x[0]):
                    if n_entered >= 2 or len(positions) >= 2: break
                    sc, s, pr = item[0], item[1], item[2]
                    if s in held_si: continue
                    cap = cash * 0.25
                    sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                    ct = max(1, int(cap / (pr * m * (1 + COMM))))
                    ci = pr * m * ct * (1 + COMM)
                    if ci > cash:
                        ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                        ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                    if ct > 0 and ci > 0 and ci <= cash:
                        cash -= ci
                        positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                          'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': 1,
                                          'sig': 'union_disagree', 'score': sc})
                        held_si.add(s)
                        n_entered += 1

        for p in positions:
            ae = end_di - 1
            ep = C[p['si'], min(ae, ND-1)]
            if np.isnan(ep) or ep <= 0: ep = p['entry_price']
            m = MULT.get(p['sym'], DEF_MULT)
            cash += ep * m * abs(p['lots']) * (1 - COMM)

        nd = end_di - start_di
        ann = annual_return(cash, CASH0, nd)
        if daily_eq:
            eq = np.array(daily_eq); pk = np.maximum.accumulate(eq)
            mdd = np.min((eq - pk) / pk * 100)
            r = np.diff(eq) / eq[:-1]
            r = np.where(np.isfinite(r), r, 0)
            sh = np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0
        else:
            mdd = 0; sh = 0
        return {'ann': ann, 'mdd': mdd, 'sharpe': sh, 'final': cash, 'eq': np.array(daily_eq) if daily_eq else np.array([])}

    # Agreement configs: also test pure agree-only and scaled versions
    print("\n  --- Agreement variants ---")
    agree_results = []

    # Full agreement strategy
    r_agree = backtest_agreement()
    r_agree['desc'] = "Agreement: agree=80% solo=50% disagree=25%"
    agree_results.append(r_agree)
    pr(r_agree, r_agree['desc'])

    # Walk-forward for agreement
    print("\n  Walk-Forward for agreement strategy:")
    wf_agree = {}
    for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
        ys = ye = None
        for di in range(ND):
            if dates[di].year == yr and ys is None: ys = di
            if dates[di].year == yr: ye = di + 1
        if ys is None: continue
        wr = backtest_agreement(start_di=ys, end_di=ye)
        wf_agree[yr] = wr
    print_wf(wf_agree, "Agreement strategy")

    # Also test: agreement with 50% base (reduced sizing for lower MDD)
    def backtest_agreement_v2(start_di=MIN_TRAIN, end_di=None,
                               agree_pct=0.60, solo_pct=0.40, disagree_pct=0.20):
        """Configurable agreement sizes."""
        if end_di is None: end_di = ND
        cash = float(CASH0)
        positions = []
        daily_eq = []

        for di in range(start_di, end_di - 1):
            pv = cash
            for p in positions:
                cp = C[p['si'], di]
                if not np.isnan(cp) and cp > 0:
                    m = MULT.get(p['sym'], DEF_MULT)
                    pv += cp * m * p['lots'] - cp * m * abs(p['lots']) * COMM
            daily_eq.append(pv)

            cl = []
            for p in positions:
                if di - p['entry_di'] >= p['hold_days']:
                    ep = C[p['si'], di]
                    if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                    m = MULT.get(p['sym'], DEF_MULT)
                    cash += ep * m * abs(p['lots']) * (1 - COMM)
                    cl.append(p)
            for p in cl: positions.remove(p)

            edi = di + 1
            if edi >= end_di: continue

            cands_v121 = sig_v121(di, edi)
            cands_union = sig_union(di, edi)
            v121_syms = {item[1]: item for item in cands_v121}
            union_syms = {item[1]: item for item in cands_union}
            agreed = set(v121_syms.keys()) & set(union_syms.keys())
            v121_only = set(v121_syms.keys()) - set(union_syms.keys())
            union_only = set(union_syms.keys()) - set(v121_syms.keys())
            held_si = set(p['si'] for p in positions)

            if agreed:
                best_agreed = max(agreed, key=lambda s: (v121_syms[s][0] + union_syms[s][0]))
                if best_agreed not in held_si and len(positions) < 2:
                    item = v121_syms[best_agreed]
                    sc, s, pr = item[0], item[1], item[2]
                    cap = cash * agree_pct
                    sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                    ct = max(1, int(cap / (pr * m * (1 + COMM))))
                    ci = pr * m * ct * (1 + COMM)
                    if ci > cash:
                        ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                        ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                    if ct > 0 and ci > 0 and ci <= cash:
                        cash -= ci
                        positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                          'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': 1,
                                          'sig': 'agree', 'score': sc})
                continue

            # Solo signal
            all_solo = []
            for s in v121_only: all_solo.append((v121_syms[s][0], s, v121_syms[s][2], 'v121'))
            for s in union_only: all_solo.append((union_syms[s][0], s, union_syms[s][2], 'union'))
            all_solo.sort(key=lambda x: -x[0])

            if all_solo and len(positions) < 2:
                for sc, s, pr, src in all_solo[:2 - len(positions)]:
                    if s in held_si: continue
                    pct = disagree_pct if (v121_only and union_only) else solo_pct
                    cap = cash * pct
                    sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                    ct = max(1, int(cap / (pr * m * (1 + COMM))))
                    ci = pr * m * ct * (1 + COMM)
                    if ci > cash:
                        ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                        ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                    if ct > 0 and ci > 0 and ci <= cash:
                        cash -= ci
                        positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                          'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': 1,
                                          'sig': src, 'score': sc})
                        held_si.add(s)

        for p in positions:
            ae = end_di - 1
            ep = C[p['si'], min(ae, ND-1)]
            if np.isnan(ep) or ep <= 0: ep = p['entry_price']
            m = MULT.get(p['sym'], DEF_MULT)
            cash += ep * m * abs(p['lots']) * (1 - COMM)

        nd = end_di - start_di
        ann = annual_return(cash, CASH0, nd)
        if daily_eq:
            eq = np.array(daily_eq); pk = np.maximum.accumulate(eq)
            mdd = np.min((eq - pk) / pk * 100)
            r = np.diff(eq) / eq[:-1]
            r = np.where(np.isfinite(r), r, 0)
            sh = np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0
        else:
            mdd = 0; sh = 0
        return {'ann': ann, 'mdd': mdd, 'sharpe': sh, 'final': cash, 'eq': np.array(daily_eq) if daily_eq else np.array([])}

    for agree_pct, solo_pct, dis_pct, label in [
        (0.60, 0.40, 0.20, "Agreement v2: agree=60% solo=40% dis=20%"),
        (0.70, 0.35, 0.18, "Agreement v3: agree=70% solo=35% dis=18%"),
        (0.50, 0.30, 0.15, "Agreement v4: agree=50% solo=30% dis=15%"),
    ]:
        r = backtest_agreement_v2(agree_pct=agree_pct, solo_pct=solo_pct, disagree_pct=dis_pct)
        r['desc'] = label
        agree_results.append(r)
        pr(r, label)

    # ===================== SECTION 4: CORRELATION-BASED SELECTION (Approach C) =====================
    print("\n" + "=" * 120)
    print("  SECTION 4: CORRELATION-BASED SELECTION (Approach C)")
    print("  From signal candidates, select pair with LOWEST historical correlation")
    print("=" * 120)

    def backtest_corr_selection(signal_func, hold=1, start_di=MIN_TRAIN, end_di=None,
                                 per_pos_frac=0.40, corr_window=20):
        """
        Get all signal candidates, compute pairwise 20-day return correlations,
        select the 2 with lowest correlation to hold simultaneously.
        """
        if end_di is None: end_di = ND
        cash = float(CASH0)
        positions = []
        daily_eq = []

        for di in range(start_di, end_di - 1):
            pv = cash
            for p in positions:
                cp = C[p['si'], di]
                if not np.isnan(cp) and cp > 0:
                    m = MULT.get(p['sym'], DEF_MULT)
                    pv += cp * m * p['lots'] - cp * m * abs(p['lots']) * COMM
            daily_eq.append(pv)

            cl = []
            for p in positions:
                if di - p['entry_di'] >= p['hold_days']:
                    ep = C[p['si'], di]
                    if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                    m = MULT.get(p['sym'], DEF_MULT)
                    cash += ep * m * abs(p['lots']) * (1 - COMM)
                    cl.append(p)
            for p in cl: positions.remove(p)

            edi = di + 1
            if edi >= end_di: continue
            cands = signal_func(di, edi)
            if not cands: continue
            cands.sort(key=lambda x: -x[0])

            held_si = set(p['si'] for p in positions)
            cands = [c for c in cands if c[1] not in held_si]
            if not cands: continue

            # If only 1 candidate, just take it
            if len(cands) == 1:
                item = cands[0]
                sc, s, pr = item[0], item[1], item[2]
                cap = cash * per_pos_frac
                sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                ct = max(1, int(cap / (pr * m * (1 + COMM))))
                ci = pr * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                    ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                if ct > 0 and ci > 0 and ci <= cash:
                    cash -= ci
                    positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                      'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': hold,
                                      'sig': item[3] if len(item) > 3 else '', 'score': sc})
                continue

            # Multiple candidates: find the pair with lowest correlation
            # Use top 6 candidates max for computational efficiency
            top_cands = cands[:6]
            n_cands = len(top_cands)

            # Compute pairwise correlations using 20-day returns
            best_pair = None
            best_corr = 999.0

            for i in range(n_cands):
                for j in range(i+1, n_cands):
                    si_a = top_cands[i][1]
                    si_b = top_cands[j][1]
                    # Get 20-day returns for correlation window
                    start_idx = max(0, di - corr_window)
                    ret_a = RET20[si_a, start_idx:di]
                    ret_b = RET20[si_b, start_idx:di]
                    # Need at least 10 overlapping valid returns
                    valid = ~(np.isnan(ret_a) | np.isnan(ret_b))
                    n_valid = np.sum(valid)
                    if n_valid < 8:
                        corr = 0.5  # default moderate correlation
                    else:
                        ra = ret_a[valid]; rb = ret_b[valid]
                        if np.std(ra) == 0 or np.std(rb) == 0:
                            corr = 0.5
                        else:
                            corr = np.corrcoef(ra, rb)[0, 1]
                            if np.isnan(corr): corr = 0.5
                    if corr < best_corr:
                        best_corr = corr
                        best_pair = (i, j)

            if best_pair is not None:
                for idx in best_pair:
                    if len(positions) >= 2: break
                    item = top_cands[idx]
                    sc, s, pr = item[0], item[1], item[2]
                    if s in set(p['si'] for p in positions): continue
                    cap = cash * per_pos_frac
                    sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                    ct = max(1, int(cap / (pr * m * (1 + COMM))))
                    ci = pr * m * ct * (1 + COMM)
                    if ci > cash:
                        ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                        ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                    if ct > 0 and ci > 0 and ci <= cash:
                        cash -= ci
                        positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                          'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': hold,
                                          'sig': item[3] if len(item) > 3 else '', 'score': sc})

        for p in positions:
            ae = end_di - 1
            ep = C[p['si'], min(ae, ND-1)]
            if np.isnan(ep) or ep <= 0: ep = p['entry_price']
            m = MULT.get(p['sym'], DEF_MULT)
            cash += ep * m * abs(p['lots']) * (1 - COMM)

        nd = end_di - start_di
        ann = annual_return(cash, CASH0, nd)
        if daily_eq:
            eq = np.array(daily_eq); pk = np.maximum.accumulate(eq)
            mdd = np.min((eq - pk) / pk * 100)
            r = np.diff(eq) / eq[:-1]
            r = np.where(np.isfinite(r), r, 0)
            sh = np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0
        else:
            mdd = 0; sh = 0
        return {'ann': ann, 'mdd': mdd, 'sharpe': sh, 'final': cash, 'eq': np.array(daily_eq) if daily_eq else np.array([])}

    corr_results = []
    for sig_func, sig_label in [(sig_v121, "V121"), (sig_union, "Union")]:
        for ppf, ppf_label in [(0.40, "40%"), (0.45, "45%"), (0.50, "50%")]:
            label = f"{sig_label} corr-pair @{ppf_label}ea"
            r = backtest_corr_selection(sig_func, per_pos_frac=ppf)
            r['desc'] = label
            corr_results.append(r)
            pr(r, label)

    # Walk-forward for best corr selection
    print("\n  Walk-Forward for correlation-based selection (Union @40%):")
    wf_corr = {}
    for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
        ys = ye = None
        for di in range(ND):
            if dates[di].year == yr and ys is None: ys = di
            if dates[di].year == yr: ye = di + 1
        if ys is None: continue
        wr = backtest_corr_selection(sig_union, start_di=ys, end_di=ye, per_pos_frac=0.40)
        wf_corr[yr] = wr
    print_wf(wf_corr, "Union corr-pair @40%")

    # ===================== SECTION 5: CROSS-SIGNAL DIVERSIFICATION (Approach D) =====================
    print("\n" + "=" * 120)
    print("  SECTION 5: CROSS-SIGNAL DIVERSIFICATION (Approach D)")
    print("  Take best V121 signal AND best Union signal simultaneously")
    print("  If same commodity, take only 1 position")
    print("=" * 120)

    def backtest_cross_signal(start_di=MIN_TRAIN, end_di=None, per_pos_frac=0.45):
        """
        Each day, take the best V121 candidate AND the best Union candidate.
        If they pick the same commodity, take 1 position at per_pos_frac.
        If different, hold both at per_pos_frac each.
        """
        if end_di is None: end_di = ND
        cash = float(CASH0)
        positions = []
        daily_eq = []

        for di in range(start_di, end_di - 1):
            pv = cash
            for p in positions:
                cp = C[p['si'], di]
                if not np.isnan(cp) and cp > 0:
                    m = MULT.get(p['sym'], DEF_MULT)
                    pv += cp * m * p['lots'] - cp * m * abs(p['lots']) * COMM
            daily_eq.append(pv)

            cl = []
            for p in positions:
                if di - p['entry_di'] >= p['hold_days']:
                    ep = C[p['si'], di]
                    if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                    m = MULT.get(p['sym'], DEF_MULT)
                    cash += ep * m * abs(p['lots']) * (1 - COMM)
                    cl.append(p)
            for p in cl: positions.remove(p)

            edi = di + 1
            if edi >= end_di: continue

            cands_v121 = sig_v121(di, edi)
            cands_union = sig_union(di, edi)

            held_si = set(p['si'] for p in positions)

            # Get best from each signal
            best_v121 = None
            if cands_v121:
                cands_v121.sort(key=lambda x: -x[0])
                for c in cands_v121:
                    if c[1] not in held_si:
                        best_v121 = c
                        break

            best_union = None
            if cands_union:
                cands_union.sort(key=lambda x: -x[0])
                for c in cands_union:
                    if c[1] not in held_si:
                        best_union = c
                        break

            # Determine what to enter
            entries = []
            if best_v121 and best_union:
                if best_v121[1] == best_union[1]:
                    # Same commodity: take 1 position, boosted size
                    entries.append((best_v121[0], best_v121[1], best_v121[2],
                                    'v121+union', per_pos_frac * 1.5))
                else:
                    # Different commodities: hold both
                    entries.append((best_v121[0], best_v121[1], best_v121[2],
                                    'v121', per_pos_frac))
                    entries.append((best_union[0], best_union[1], best_union[2],
                                    'union', per_pos_frac))
            elif best_v121:
                entries.append((best_v121[0], best_v121[1], best_v121[2],
                                'v121', per_pos_frac))
            elif best_union:
                entries.append((best_union[0], best_union[1], best_union[2],
                                'union', per_pos_frac))

            for sc, s, pr, sig_str, pct in entries:
                if s in set(p['si'] for p in positions): continue
                if len(positions) >= 2: break
                cap = cash * pct
                sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                ct = max(1, int(cap / (pr * m * (1 + COMM))))
                ci = pr * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                    ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                if ct > 0 and ci > 0 and ci <= cash:
                    cash -= ci
                    positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                      'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': 1,
                                      'sig': sig_str, 'score': sc})

        for p in positions:
            ae = end_di - 1
            ep = C[p['si'], min(ae, ND-1)]
            if np.isnan(ep) or ep <= 0: ep = p['entry_price']
            m = MULT.get(p['sym'], DEF_MULT)
            cash += ep * m * abs(p['lots']) * (1 - COMM)

        nd = end_di - start_di
        ann = annual_return(cash, CASH0, nd)
        if daily_eq:
            eq = np.array(daily_eq); pk = np.maximum.accumulate(eq)
            mdd = np.min((eq - pk) / pk * 100)
            r = np.diff(eq) / eq[:-1]
            r = np.where(np.isfinite(r), r, 0)
            sh = np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0
        else:
            mdd = 0; sh = 0
        return {'ann': ann, 'mdd': mdd, 'sharpe': sh, 'final': cash, 'eq': np.array(daily_eq) if daily_eq else np.array([])}

    cross_results = []
    for ppf, ppf_label in [(0.30, "30%"), (0.40, "40%"), (0.45, "45%"), (0.50, "50%")]:
        label = f"Cross-signal V121+Union @{ppf_label}ea"
        r = backtest_cross_signal(per_pos_frac=ppf)
        r['desc'] = label
        cross_results.append(r)
        pr(r, label)

    # Walk-forward for cross-signal
    print("\n  Walk-Forward for cross-signal diversification:")
    for ppf, ppf_label in [(0.30, "30%"), (0.40, "40%"), (0.45, "45%")]:
        wf_cross = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue
            wr = backtest_cross_signal(start_di=ys, end_di=ye, per_pos_frac=ppf)
            wf_cross[yr] = wr
        print_wf(wf_cross, f"Cross-signal @{ppf_label}")

    # ===================== SECTION 6: COMBINED — CROSS-SIGNAL + CORRELATION =====================
    print("\n" + "=" * 120)
    print("  SECTION 6: CROSS-SIGNAL + CORRELATION FILTER")
    print("  Take V121+Union simultaneously, but only when their correlation is LOW")
    print("=" * 120)

    def backtest_cross_corr(start_di=MIN_TRAIN, end_di=None, per_pos_frac=0.40,
                             max_corr=0.6):
        """
        Take V121+Union simultaneously, but only if their 20-day correlation < max_corr.
        If correlation is too high, only take the best signal.
        """
        if end_di is None: end_di = ND
        cash = float(CASH0)
        positions = []
        daily_eq = []

        for di in range(start_di, end_di - 1):
            pv = cash
            for p in positions:
                cp = C[p['si'], di]
                if not np.isnan(cp) and cp > 0:
                    m = MULT.get(p['sym'], DEF_MULT)
                    pv += cp * m * p['lots'] - cp * m * abs(p['lots']) * COMM
            daily_eq.append(pv)

            cl = []
            for p in positions:
                if di - p['entry_di'] >= p['hold_days']:
                    ep = C[p['si'], di]
                    if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                    m = MULT.get(p['sym'], DEF_MULT)
                    cash += ep * m * abs(p['lots']) * (1 - COMM)
                    cl.append(p)
            for p in cl: positions.remove(p)

            edi = di + 1
            if edi >= end_di: continue

            cands_v121 = sig_v121(di, edi)
            cands_union = sig_union(di, edi)

            held_si = set(p['si'] for p in positions)

            best_v121 = None
            if cands_v121:
                cands_v121.sort(key=lambda x: -x[0])
                for c in cands_v121:
                    if c[1] not in held_si:
                        best_v121 = c
                        break

            best_union = None
            if cands_union:
                cands_union.sort(key=lambda x: -x[0])
                for c in cands_union:
                    if c[1] not in held_si:
                        best_union = c
                        break

            entries = []
            if best_v121 and best_union:
                if best_v121[1] == best_union[1]:
                    # Same commodity
                    entries.append((best_v121[0], best_v121[1], best_v121[2],
                                    'v121+union', per_pos_frac * 1.5))
                else:
                    # Check correlation
                    si_a = best_v121[1]; si_b = best_union[1]
                    start_idx = max(0, di - 20)
                    ret_a = RET20[si_a, start_idx:di]
                    ret_b = RET20[si_b, start_idx:di]
                    valid = ~(np.isnan(ret_a) | np.isnan(ret_b))
                    n_valid = np.sum(valid)
                    corr = 0.5  # default
                    if n_valid >= 8:
                        ra = ret_a[valid]; rb = ret_b[valid]
                        if np.std(ra) > 0 and np.std(rb) > 0:
                            c = np.corrcoef(ra, rb)[0, 1]
                            if not np.isnan(c): corr = c

                    if corr < max_corr:
                        entries.append((best_v121[0], best_v121[1], best_v121[2],
                                        'v121', per_pos_frac))
                        entries.append((best_union[0], best_union[1], best_union[2],
                                        'union', per_pos_frac))
                    else:
                        # Correlation too high: just take the best
                        best = best_v121 if best_v121[0] >= best_union[0] else best_union
                        entries.append((best[0], best[1], best[2], 'best', per_pos_frac))
            elif best_v121:
                entries.append((best_v121[0], best_v121[1], best_v121[2], 'v121', per_pos_frac))
            elif best_union:
                entries.append((best_union[0], best_union[1], best_union[2], 'union', per_pos_frac))

            for sc, s, pr, sig_str, pct in entries:
                if s in set(p['si'] for p in positions): continue
                if len(positions) >= 2: break
                cap = cash * pct
                sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                ct = max(1, int(cap / (pr * m * (1 + COMM))))
                ci = pr * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                    ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                if ct > 0 and ci > 0 and ci <= cash:
                    cash -= ci
                    positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                      'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': 1,
                                      'sig': sig_str, 'score': sc})

        for p in positions:
            ae = end_di - 1
            ep = C[p['si'], min(ae, ND-1)]
            if np.isnan(ep) or ep <= 0: ep = p['entry_price']
            m = MULT.get(p['sym'], DEF_MULT)
            cash += ep * m * abs(p['lots']) * (1 - COMM)

        nd = end_di - start_di
        ann = annual_return(cash, CASH0, nd)
        if daily_eq:
            eq = np.array(daily_eq); pk = np.maximum.accumulate(eq)
            mdd = np.min((eq - pk) / pk * 100)
            r = np.diff(eq) / eq[:-1]
            r = np.where(np.isfinite(r), r, 0)
            sh = np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0
        else:
            mdd = 0; sh = 0
        return {'ann': ann, 'mdd': mdd, 'sharpe': sh, 'final': cash, 'eq': np.array(daily_eq) if daily_eq else np.array([])}

    combined_results = []
    for ppf, ppf_label, mc in [(0.40, "40%", 0.5), (0.40, "40%", 0.7),
                                (0.45, "45%", 0.5), (0.45, "45%", 0.7),
                                (0.50, "50%", 0.5)]:
        label = f"Cross+Corr @{ppf_label} max_corr={mc}"
        r = backtest_cross_corr(per_pos_frac=ppf, max_corr=mc)
        r['desc'] = label
        combined_results.append(r)
        pr(r, label)

    # Walk-forward for best combined
    print("\n  Walk-Forward for Cross+Corr:")
    wf_comb = {}
    for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
        ys = ye = None
        for di in range(ND):
            if dates[di].year == yr and ys is None: ys = di
            if dates[di].year == yr: ye = di + 1
        if ys is None: continue
        wr = backtest_cross_corr(start_di=ys, end_di=ye, per_pos_frac=0.45, max_corr=0.5)
        wf_comb[yr] = wr
    print_wf(wf_comb, "Cross+Corr @45% max_corr=0.5")

    # ===================== SECTION 7: COMPREHENSIVE RANKING =====================
    print("\n" + "=" * 120)
    print("  SECTION 7: COMPREHENSIVE RANKING BY RETURN/MDD RATIO")
    print("=" * 120)

    # Collect all results
    all_results = []
    all_results.append({'desc': 'V121 baseline top_n=1 @50%', **r_v121_base})
    all_results.append({'desc': 'Union baseline top_n=1 @50%', **r_union_base})
    all_results.extend(multi_results)
    all_results.extend(agree_results)
    all_results.extend(corr_results)
    all_results.extend(cross_results)
    all_results.extend(combined_results)

    # Sort by return/MDD ratio
    all_with_ratio = []
    for r in all_results:
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        all_with_ratio.append((r, ratio))

    all_with_ratio.sort(key=lambda x: -x[1])

    print(f"\n  {'#':>3} {'Config':70s} | {'Ann':>8} | {'MDD':>6} | {'Sh':>4} | {'R/M':>5}")
    print(f"  {'---':>3} {'-'*70}-+-{'-'*8}-+-{'-'*6}-+-{'-'*4}-+-{'-'*5}")
    for i, (r, ratio) in enumerate(all_with_ratio[:25]):
        desc = r.get('desc', '')
        print(f"  {i+1:3d} {desc:70s} | {r['ann']:+8.1f}% | {r['mdd']:6.1f}% | {r['sharpe']:4.2f} | {ratio:.2f}")

    # Top configs sorted by Sharpe
    print(f"\n  --- Top 15 by Sharpe Ratio ---")
    all_results_copy = list(all_results)
    all_results_copy.sort(key=lambda x: -x['sharpe'])
    for i, r in enumerate(all_results_copy[:15]):
        desc = r.get('desc', '')
        print(f"  {i+1:3d} {desc:70s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    # Top configs with MDD > -40%
    print(f"\n  --- Top 10 by R/M ratio with MDD > -40% ---")
    safe = [(r, ratio) for r, ratio in all_with_ratio if r['mdd'] > -40]
    for i, (r, ratio) in enumerate(safe[:10]):
        desc = r.get('desc', '')
        print(f"  {i+1:3d} {desc:70s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    # ===================== SECTION 8: DETAILED WF FOR TOP 5 =====================
    print("\n" + "=" * 120)
    print("  SECTION 8: DETAILED WALK-FORWARD FOR TOP 5 BY R/M RATIO")
    print("=" * 120)

    # Identify top 5 unique configs
    seen = set()
    top5 = []
    for r, ratio in all_with_ratio:
        desc = r.get('desc', '')
        if desc in seen: continue
        seen.add(desc)
        top5.append((r, ratio))
        if len(top5) >= 5: break

    print(f"\n  Top 5 configs for detailed WF:")
    for i, (r, ratio) in enumerate(top5):
        desc = r.get('desc', '')
        print(f"\n  #{i+1}: {desc} (R/M={ratio:.2f}, Ann={r['ann']:+.1f}%, MDD={r['mdd']:.1f}%)")

        # Determine which backtest function to use based on desc
        if 'corr-pair' in desc:
            if 'V121' in desc:
                sig_func = sig_v121
            else:
                sig_func = sig_union
            ppf = float(desc.split('@')[1].replace('%ea', '').replace('%', '')) / 100
            wf_res = {}
            for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
                ys = ye = None
                for di in range(ND):
                    if dates[di].year == yr and ys is None: ys = di
                    if dates[di].year == yr: ye = di + 1
                if ys is None: continue
                wr = backtest_corr_selection(sig_func, start_di=ys, end_di=ye, per_pos_frac=ppf)
                wf_res[yr] = wr
            print_wf(wf_res, desc)

        elif 'Cross' in desc and 'Corr' in desc:
            ppf = float(desc.split('@')[1].split('%')[0]) / 100
            mc = float(desc.split('max_corr=')[1])
            wf_res = {}
            for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
                ys = ye = None
                for di in range(ND):
                    if dates[di].year == yr and ys is None: ys = di
                    if dates[di].year == yr: ye = di + 1
                if ys is None: continue
                wr = backtest_cross_corr(start_di=ys, end_di=ye, per_pos_frac=ppf, max_corr=mc)
                wf_res[yr] = wr
            print_wf(wf_res, desc)

        elif 'Cross-signal' in desc:
            ppf = float(desc.split('@')[1].split('%')[0]) / 100
            wf_res = {}
            for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
                ys = ye = None
                for di in range(ND):
                    if dates[di].year == yr and ys is None: ys = di
                    if dates[di].year == yr: ye = di + 1
                if ys is None: continue
                wr = backtest_cross_signal(start_di=ys, end_di=ye, per_pos_frac=ppf)
                wf_res[yr] = wr
            print_wf(wf_res, desc)

        elif 'Agreement' in desc:
            wf_res = {}
            for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
                ys = ye = None
                for di in range(ND):
                    if dates[di].year == yr and ys is None: ys = di
                    if dates[di].year == yr: ye = di + 1
                if ys is None: continue
                wr = backtest_agreement(start_di=ys, end_di=ye)
                wf_res[yr] = wr
            print_wf(wf_res, desc)

        elif 'top_n=' in desc:
            if 'V121' in desc:
                sig_func = sig_v121
            else:
                sig_func = sig_union
            topn = int(desc.split('top_n=')[1].split(' ')[0])
            ppf = float(desc.split('@')[1].replace('%ea', '').replace('%', '')) / 100
            wf_res = wf(sig_func, hold=1, top_n=topn, per_pos_frac=ppf)
            print_wf(wf_res, desc)

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 120)
    print("  FINAL SUMMARY")
    print("=" * 120)

    print(f"\n  Baselines:")
    pr(r_v121_base, "V121 top_n=1 @50%")
    pr(r_union_base, "Union top_n=1 @50%")

    print(f"\n  Best multi-position (Approach A):")
    multi_sorted = sorted(multi_results, key=lambda x: abs(x['ann']/x['mdd']) if x['mdd'] != 0 else 0, reverse=True)
    for r in multi_sorted[:3]:
        pr(r, r['desc'])

    print(f"\n  Best signal agreement (Approach B):")
    agree_sorted = sorted(agree_results, key=lambda x: abs(x['ann']/x['mdd']) if x['mdd'] != 0 else 0, reverse=True)
    for r in agree_sorted[:3]:
        pr(r, r['desc'])

    print(f"\n  Best correlation-based (Approach C):")
    corr_sorted = sorted(corr_results, key=lambda x: abs(x['ann']/x['mdd']) if x['mdd'] != 0 else 0, reverse=True)
    for r in corr_sorted[:3]:
        pr(r, r['desc'])

    print(f"\n  Best cross-signal (Approach D):")
    cross_sorted = sorted(cross_results, key=lambda x: abs(x['ann']/x['mdd']) if x['mdd'] != 0 else 0, reverse=True)
    for r in cross_sorted[:3]:
        pr(r, r['desc'])

    print(f"\n  Best combined (Cross + Correlation):")
    comb_sorted = sorted(combined_results, key=lambda x: abs(x['ann']/x['mdd']) if x['mdd'] != 0 else 0, reverse=True)
    for r in comb_sorted[:3]:
        pr(r, r['desc'])

    print(f"\n  OVERALL TOP 3:")
    for i, (r, ratio) in enumerate(all_with_ratio[:3]):
        desc = r.get('desc', '')
        print(f"  #{i+1}: {desc:70s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
