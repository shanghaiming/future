"""
Alpha Futures V133 — INTELLIGENT SIGNAL SWITCHING + MULTI-POSITION PORTFOLIO
=============================================================================
Key insight from V132: V121 peaks in 2021-2022 (+453%/+544%), OV/ID peaks in 2024 (+712%).
These are INDEPENDENT alpha sources. If we intelligently switch between them:
- V121 signal → take it (strong in trending years)
- No V121 → try OV/ID (strong in range/transition years)
- Neither → try Final Flag or Pullback
- Also: top_n=2 with BOTH V121 and OV/ID for diversification

Plus: dynamic allocation based on which strategy has been winning recently.
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
    print("=" * 120)
    print("  V133 — INTELLIGENT SIGNAL SWITCHING + MULTI-POSITION PORTFOLIO")
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
    BODY_RATIO = np.full((NS, ND), np.nan)
    BAR_DIR = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        for di in range(ND):
            o, c, h, l = O[si, di], C[si, di], H[si, di], L[si, di]
            if not np.isnan(o) and not np.isnan(c):
                if di > 0 and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                    OV_GAP[si, di] = (o - C[si, di-1]) / C[si, di-1] * 100
                if o > 0: ID_RET[si, di] = (c - o) / o * 100
            if not np.isnan(h) and not np.isnan(l) and not np.isnan(c) and not np.isnan(o) and h != l:
                BODY_RATIO[si, di] = abs(c - o) / (h - l)
                BAR_DIR[si, di] = 1 if c > o else (-1 if c < o else 0)

    print(f"  Done ({time.time()-t0:.1f}s)")

    def backtest(signal_func, hold=1, top_n=1, start_di=MIN_TRAIN, end_di=None, desc=""):
        if end_di is None: end_di = ND
        cash = float(CASH0); positions = []; trades = []; daily_eq = []
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
                    pnl = (ep - p['entry_price']) * m * p['lots']
                    inv = p['entry_price'] * m * abs(p['lots'])
                    pp = pnl / inv * 100 if inv > 0 else 0
                    cash += ep * m * abs(p['lots']) * (1 - COMM)
                    trades.append({'pnl_pct': pp, 'sig': p.get('sig', '')})
                    cl.append(p)
            for p in cl: positions.remove(p)
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue
            cands = signal_func(di, edi)
            if not cands: continue
            cands.sort(key=lambda x: -x[0])
            ns = top_n - len(positions)
            cap = cash / max(1, ns)
            for item in cands[:ns]:
                if len(item) == 3: sc, s, pr = item; sig = ''
                else: sc, s, pr, sig = item
                sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                ct = max(1, int(cap * 0.95 / (pr * m * (1 + COMM))))
                ci = pr * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                    ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                if ct <= 0 or ci <= 0 or ci > cash: continue
                cash -= ci
                positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                  'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': hold, 'sig': sig})
        for p in positions:
            ae = end_di - 1
            ep = C[p['si'], min(ae, ND-1)]
            if np.isnan(ep) or ep <= 0: ep = p['entry_price']
            m = MULT.get(p['sym'], DEF_MULT)
            cash += ep * m * abs(p['lots']) * (1 - COMM)
        nd = end_di - start_di
        ann = annual_return(cash, CASH0, nd)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        nt = len(trades)
        ap = np.mean([t['pnl_pct'] for t in trades]) if trades else 0
        if daily_eq:
            eq = np.array(daily_eq); pk = np.maximum.accumulate(eq)
            mdd = np.min((eq - pk) / pk * 100)
            r = np.diff(eq) / eq[:-1]
            sh = np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0
        else: mdd = 0; sh = 0
        # Signal breakdown
        sig_breakdown = {}
        for t in trades:
            s = t.get('sig', '?')
            if s not in sig_breakdown: sig_breakdown[s] = {'n': 0, 'w': 0, 'pnl': 0}
            sig_breakdown[s]['n'] += 1
            if t['pnl_pct'] > 0: sig_breakdown[s]['w'] += 1
            sig_breakdown[s]['pnl'] += t['pnl_pct']
        return {'ann': ann, 'wr': wr, 'n': nt, 'avg_pnl': ap, 'mdd': mdd, 'sharpe': sh,
                'desc': desc, 'sig_breakdown': sig_breakdown}

    def pr(r, label=""):
        print(f"  {label:60s} | Ann={r['ann']:+8.1f}% | WR={r['wr']:5.1f}% | "
              f"N={r['n']:4d} | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    def wf(func, hold=1, topn=1):
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys: res[yr] = backtest(func, hold=hold, top_n=topn, start_di=ys, end_di=ye)['ann']
        return res

    # ============ SIGNALS ============

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
            c.append(((ov + idr) * roc * z_bonus, s, ep, 'ov_id'))
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
            c.append((roc20 * (cp - h4) / atr, s, ep, 'final_flag'))
        return c

    # A) V121 primary, OV/ID secondary, Final Flag tertiary
    def sig_cascade(di, edi):
        v121 = sig_v121(di, edi)
        if v121: return v121
        ov = sig_ov_id(di, edi)
        if ov: return ov
        ff = sig_final_flag(di, edi)
        if ff: return ff
        return []

    # B) Scored combination: all signals compete, pick highest
    def sig_compete(di, edi):
        all_sigs = []
        for item in sig_v121(di, edi):
            sc, s, ep, st = item
            all_sigs.append((sc * 3, s, ep, st))  # V121 gets 3x weight
        for item in sig_ov_id(di, edi):
            sc, s, ep, st = item
            if not any(x[1] == s for x in all_sigs):
                all_sigs.append((sc * 2, s, ep, st))  # OV/ID gets 2x
        for item in sig_final_flag(di, edi):
            sc, s, ep, st = item
            if not any(x[1] == s for x in all_sigs):
                all_sigs.append((sc, s, ep, st))
        return all_sigs

    # C) Multi-slot: top_n=2, one V121 + one OV/ID (diversified)
    def sig_diversified(di, edi):
        v121 = sig_v121(di, edi)
        ov = sig_ov_id(di, edi)
        # Take best from each category (if different commodities)
        all_sigs = []
        used_si = set()
        if v121:
            v121.sort(key=lambda x: -x[0])
            best = v121[0]
            all_sigs.append((best[0] * 3, best[1], best[2], best[3]))
            used_si.add(best[1])
        if ov:
            ov.sort(key=lambda x: -x[0])
            for item in ov:
                if item[1] not in used_si:
                    all_sigs.append((item[0] * 2, item[1], item[2], item[3]))
                    used_si.add(item[1])
                    break
        return all_sigs

    # D) V121 + OV/ID both required (strict overlap)
    def sig_strict_overlap(di, edi):
        v121_sis = set()
        v121_map = {}
        for item in sig_v121(di, edi):
            v121_sis.add(item[1])
            v121_map[item[1]] = item
        ov_sis = set()
        ov_map = {}
        for item in sig_ov_id(di, edi):
            ov_sis.add(item[1])
            ov_map[item[1]] = item
        # Intersection: commodity appears in BOTH V121 and OV/ID
        overlap = v121_sis & ov_sis
        if not overlap: return []
        cands = []
        for si in overlap:
            v = v121_map[si]; o = ov_map[si]
            # Combined score
            score = v[0] + o[0]
            cands.append((score, si, v[2], 'v121+ov_id'))
        return cands

    # E) Recent-performance adaptive: boost the strategy that's been winning
    # This requires state, so we track via a global trade history approach
    # Simplified: alternate between V121 and OV/ID based on which had better last 20 trades
    recent_v121_wr = [0.63]  # rolling estimate
    recent_ov_wr = [0.62]

    def sig_adaptive(di, edi):
        # Get both signal sets
        v121 = sig_v121(di, edi)
        ov = sig_ov_id(di, edi)
        # Choose based on recent performance
        if recent_v121_wr[-1] >= recent_ov_wr[-1]:
            if v121: return v121
            if ov: return ov
        else:
            if ov: return ov
            if v121: return v121
        ff = sig_final_flag(di, edi)
        return ff if ff else []

    # F) V121 + OV/ID union: take ALL signals from both, rank by combined score
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

    # ============ SECTION 1: ALL CONFIGS ============
    print("\n" + "=" * 120)
    print("  SECTION 1: SIGNAL SWITCHING CONFIGURATIONS")
    print("=" * 120)

    configs = [
        ("V121 baseline", sig_v121, 1, 1),
        ("OV/ID baseline", sig_ov_id, 1, 1),
        ("Final Flag baseline", sig_final_flag, 1, 1),
        ("A) Cascade V121>OV>FF", sig_cascade, 1, 1),
        ("B) Compete (V121x3,OVx2,FFx1)", sig_compete, 1, 1),
        ("C) Diversified (V121+OV slot)", sig_diversified, 1, 2),
        ("D) Strict overlap (V121∩OV)", sig_strict_overlap, 1, 1),
        ("F) Union all ranked", sig_union, 1, 1),
    ]

    results = {}
    for name, func, hold, topn in configs:
        r = backtest(func, hold=hold, top_n=topn, desc=name)
        results[name] = r
        pr(r, label=name)

    # ============ SECTION 2: TOP_N x HOLD ============
    print("\n" + "=" * 120)
    print("  SECTION 2: TOP_N x HOLD for top configs")
    print("=" * 120)

    best3 = sorted(results.items(), key=lambda x: -x[1]['ann'])[:3]
    for name, r in best3:
        func_map = {n: f for n, f, _, _ in configs}
        func = func_map[name]
        print(f"\n  {name}:")
        for topn in [1, 2, 3]:
            for hold in [1, 2, 3]:
                r = backtest(func, hold=hold, top_n=topn, desc=f"{name} t={topn} h={hold}")
                print(f"    top_n={topn} hold={hold}: Ann={r['ann']:+8.1f}% | "
                      f"WR={r['wr']:5.1f}% | N={r['n']:4d} | MDD={r['mdd']:6.1f}%")

    # ============ SECTION 3: WALK-FORWARD ============
    print("\n" + "=" * 120)
    print("  SECTION 3: WALK-FORWARD")
    print("=" * 120)

    for name, func, hold, topn in configs:
        w = wf(func, hold=hold, topn=topn)
        ws = " | ".join([f"{yr}:{v:+.0f}%" for yr, v in sorted(w.items())])
        pos = sum(1 for v in w.values() if v > 0)
        avg = np.mean(list(w.values())) if w else 0
        print(f"  {name:60s} | {pos}/6 | Avg={avg:>+7.0f}% | {ws}")

    # ============ SECTION 4: SIGNAL BREAKDOWN ============
    print("\n" + "=" * 120)
    print("  SECTION 4: SIGNAL TYPE BREAKDOWN")
    print("=" * 120)

    for name in ["B) Compete (V121x3,OVx2,FFx1)", "F) Union all ranked"]:
        if name in results:
            r = results[name]
            bd = r.get('sig_breakdown', {})
            print(f"\n  {name}:")
            for sig, data in sorted(bd.items(), key=lambda x: -x[1]['n']):
                wr = data['w'] / data['n'] * 100 if data['n'] > 0 else 0
                ap = data['pnl'] / data['n'] if data['n'] > 0 else 0
                print(f"    {sig:20s}: N={data['n']:4d} | WR={wr:5.1f}% | AvgPnL={ap:+.2f}%")

    # ============ SECTION 5: DIVERSIFIED PORTFOLIO COMBINATIONS ============
    print("\n" + "=" * 120)
    print("  SECTION 5: DIVERSIFIED top_n=2-5 COMBINATIONS")
    print("=" * 120)

    div_configs = [
        ("Cascade t=2", sig_cascade, 1, 2),
        ("Cascade t=3", sig_cascade, 1, 3),
        ("Compete t=2", sig_compete, 1, 2),
        ("Compete t=3", sig_compete, 1, 3),
        ("Union t=2", sig_union, 1, 2),
        ("Union t=3", sig_union, 1, 3),
        ("Diversified t=2 h=1", sig_diversified, 1, 2),
        ("Union t=2 h=2", sig_union, 2, 2),
        ("Union t=3 h=2", sig_union, 2, 3),
    ]

    for name, func, hold, topn in div_configs:
        r = backtest(func, hold=hold, top_n=topn, desc=name)
        pr(r, label=name)

    # ============ SUMMARY ============
    print("\n" + "=" * 120)
    print("  SUMMARY: TOP 15 BY ANNUAL RETURN")
    print("=" * 120)

    all_r = {**results}
    for name, func, hold, topn in div_configs:
        r = backtest(func, hold=hold, top_n=topn, desc=name)
        all_r[name] = r

    sorted_r = sorted(all_r.items(), key=lambda x: -x[1]['ann'])
    for i, (name, r) in enumerate(sorted_r[:15]):
        print(f"  #{i+1}: {name:60s} | Ann={r['ann']:+8.1f}% | WR={r['wr']:5.1f}% | "
              f"N={r['n']:4d} | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    sorted_sh = sorted(all_r.items(), key=lambda x: -x[1]['sharpe'])
    print(f"\n  TOP 10 BY SHARPE:")
    for i, (name, r) in enumerate(sorted_sh[:10]):
        print(f"  #{i+1}: {name:60s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
