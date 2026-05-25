"""
Alpha Futures V139 — ROTATING PORTFOLIO WITH WALK-FORWARD VALIDATION
=============================================================================
Builds on V135 (rotating portfolio, never walk-forward validated) and V133
(union signal +384.5% annual, 6/6 WF).

V139 properly validates and optimizes rotating portfolio by:
  Part 1: V133 Union signal implementation (V121 + OV/ID + Final Flag)
  Part 2: Rotating portfolio framework with parameterized lookback & weight
  Part 3: Grid search over all strategy pairs x lookback x weight
  Part 4: Walk-forward validation for top configurations (per calendar year)
  Part 5: Union + rotating combinations
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
    print("  V139 — ROTATING PORTFOLIO WITH WALK-FORWARD VALIDATION")
    print("=" * 120)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    print(f"  {NS} commodities, {ND} days")

    print("\n[Precompute]...", flush=True)
    t0 = time.time()

    # Daily returns
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

    # ATR
    ATR14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        ATR14[si] = talib.ATR(H[si].astype(np.float64), L[si].astype(np.float64),
                               C[si].astype(np.float64), timeperiod=14)

    # Z-score (20-day rolling)
    ZSCORE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = RET[si, di-20:di]
            v = rets[~np.isnan(rets)]
            if len(v) < 10: continue
            s = np.std(v, ddof=1)
            if s > 0 and not np.isnan(RET[si, di]):
                ZSCORE[si, di] = (RET[si, di] - np.mean(v)) / s

    # OV/ID features
    OV_GAP = np.full((NS, ND), np.nan)
    ID_RET = np.full((NS, ND), np.nan)
    BODY_RATIO = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            o, c, h, l = O[si, di], C[si, di], H[si, di], L[si, di]
            if not np.isnan(o) and not np.isnan(c):
                if di > 0 and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                    OV_GAP[si, di] = (o - C[si, di-1]) / C[si, di-1] * 100
                if o > 0: ID_RET[si, di] = (c - o) / o * 100
            if not np.isnan(h) and not np.isnan(l) and not np.isnan(c) and not np.isnan(o) and h != l:
                BODY_RATIO[si, di] = abs(c - o) / (h - l)

    print(f"  Done ({time.time()-t0:.1f}s)")

    # ===================== BACKTEST FUNCTIONS =====================
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
        sig_breakdown = {}
        for t in trades:
            s = t.get('sig', '?')
            if s not in sig_breakdown: sig_breakdown[s] = {'n': 0, 'w': 0, 'pnl': 0}
            sig_breakdown[s]['n'] += 1
            if t['pnl_pct'] > 0: sig_breakdown[s]['w'] += 1
            sig_breakdown[s]['pnl'] += t['pnl_pct']
        return {'ann': ann, 'wr': wr, 'n': nt, 'avg_pnl': ap, 'mdd': mdd, 'sharpe': sh,
                'desc': desc, 'sig_breakdown': sig_breakdown, 'final': cash}

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

    # ===================== SIGNAL DEFINITIONS (from V133) =====================

    def sig_v121(di, edi):
        """V121: ROC(5)>1%, Z>1.5, ROC improving, score = ROC*Z"""
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
        """OV/ID: OV_GAP>0.3%, ID_RET>0.3%, ROC5>1%, score = (ov+id)*roc*z_bonus*2"""
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
        """Final Flag: ROC20>5%, 5-day range < 3*ATR, close > 4-day high, score = roc20*(cp-h4)/atr"""
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

    def sig_union(di, edi):
        """V133 Union: V121+OV/ID+Final Flag, all compete, ranked by combined score.
        V121 gets 3x weight, OV/ID gets 2x, FF gets 1x."""
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

    # ===================== EQUITY CURVE BACKTEST =====================

    def backtest_equity(signal_func, hold=1, top_n=1, start_di=MIN_TRAIN, end_di=None):
        """Backtest returning daily equity series for portfolio combination."""
        if end_di is None: end_di = ND
        cash = float(CASH0)
        positions = []; daily_eq = []
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
        daily_eq.append(cash)
        return np.array(daily_eq), cash

    # ===================== ROTATING PORTFOLIO =====================

    def backtest_rotating(sig_A, sig_B, lookback, weight_win, start_di=MIN_TRAIN, end_di=None):
        """
        Rotating portfolio: allocate between sig_A and sig_B based on rolling win rate.
        If win_A >= win_B over the lookback, give weight_win to A, else to B.
        Returns dict with ann, mdd, sharpe, final, eq.
        """
        if end_di is None: end_di = ND
        # Compute equity curves for each signal at 100% capital
        eq_A, _ = backtest_equity(sig_A, hold=1, top_n=1, start_di=start_di, end_di=end_di)
        eq_B, _ = backtest_equity(sig_B, hold=1, top_n=1, start_di=start_di, end_di=end_di)

        # Align lengths
        ml = min(len(eq_A), len(eq_B)) - 1
        if ml <= 0:
            return {'ann': -100.0, 'mdd': 0, 'sharpe': 0, 'final': CASH0}
        ret_A = np.diff(eq_A[:ml+1]) / eq_A[:ml]
        ret_B = np.diff(eq_B[:ml+1]) / eq_B[:ml]

        # Handle any NaN or inf in returns
        ret_A = np.where(np.isfinite(ret_A), ret_A, 0)
        ret_B = np.where(np.isfinite(ret_B), ret_B, 0)

        # Rolling win rate
        win_A = (ret_A > 0).astype(float)
        win_B = (ret_B > 0).astype(float)

        # Rotating allocation
        combined_ret = np.zeros(ml)
        for i in range(ml):
            if i < lookback:
                # Not enough history, use 50/50
                w_A = 0.5
            else:
                wr_A = np.mean(win_A[i-lookback:i])
                wr_B = np.mean(win_B[i-lookback:i])
                if wr_A >= wr_B:
                    w_A = weight_win
                else:
                    w_A = 1.0 - weight_win
            combined_ret[i] = w_A * ret_A[i] + (1.0 - w_A) * ret_B[i]

        # Build equity curve
        eq_rot = np.zeros(ml + 1)
        eq_rot[0] = float(CASH0)
        for i in range(ml):
            eq_rot[i+1] = eq_rot[i] * (1 + combined_ret[i])

        final = eq_rot[-1]
        nd = ml
        ann = annual_return(final, CASH0, nd)
        pk = np.maximum.accumulate(eq_rot)
        mdd = np.min((eq_rot - pk) / pk * 100)
        r = combined_ret
        sh = np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0
        return {'ann': ann, 'mdd': mdd, 'sharpe': sh, 'final': final, 'eq': eq_rot}

    def metrics_from_eq(eq):
        """Compute ann, mdd, sharpe from an equity series."""
        if len(eq) < 2:
            return {'ann': -100.0, 'mdd': 0, 'sharpe': 0, 'final': eq[-1] if len(eq) > 0 else CASH0}
        nd = len(eq)
        final = eq[-1]
        ann = annual_return(final, CASH0, nd)
        pk = np.maximum.accumulate(eq)
        mdd = np.min((eq - pk) / pk * 100)
        rets = np.diff(eq) / eq[:-1]
        rets = np.where(np.isfinite(rets), rets, 0)
        sh = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
        return {'ann': ann, 'mdd': mdd, 'sharpe': sh, 'final': final}

    # ============ SECTION 1: SIGNAL BASELINES ============
    print("\n" + "=" * 120)
    print("  SECTION 1: SIGNAL BASELINES (V121, OV/ID, Union, FF)")
    print("=" * 120)

    baseline_configs = [
        ("V121 baseline", sig_v121, 1, 1),
        ("OV/ID baseline", sig_ov_id, 1, 1),
        ("Final Flag baseline", sig_final_flag, 1, 1),
        ("V133 Union", sig_union, 1, 1),
    ]

    baseline_results = {}
    for name, func, hold, topn in baseline_configs:
        r = backtest(func, hold=hold, top_n=topn, desc=name)
        baseline_results[name] = r
        pr(r, label=name)

    # Walk-forward baselines
    print("\n  Walk-Forward for baselines:")
    for name, func, hold, topn in baseline_configs:
        w = wf(func, hold=hold, topn=topn)
        ws = " | ".join([f"{yr}:{v:+.0f}%" for yr, v in sorted(w.items())])
        pos = sum(1 for v in w.values() if v > 0)
        avg = np.mean(list(w.values())) if w else 0
        print(f"  {name:40s} | {pos}/6 | Avg={avg:>+7.0f}% | {ws}")

    # ============ SECTION 2: ROTATING PARAMETER SWEEP ============
    print("\n" + "=" * 120)
    print("  SECTION 2: ROTATING PARAMETER SWEEP")
    print("=" * 120)

    # Strategy pairs to test
    signal_pool = {
        'V121': sig_v121,
        'OV/ID': sig_ov_id,
        'Union': sig_union,
    }

    # Pre-compute equity curves for all signals (full period)
    print("\n  Pre-computing equity curves for all signals...", flush=True)
    eq_cache = {}
    ret_cache = {}
    for sname, sfunc in signal_pool.items():
        eq, final = backtest_equity(sfunc, hold=1, top_n=1, start_di=MIN_TRAIN, end_di=ND)
        ml = len(eq) - 1
        rets = np.diff(eq) / eq[:-1]
        rets = np.where(np.isfinite(rets), rets, 0)
        win = (rets > 0).astype(float)
        eq_cache[sname] = eq
        ret_cache[sname] = {'ret': rets, 'win': win, 'ml': ml}
        m = metrics_from_eq(eq)
        print(f"    {sname}: Ann={m['ann']:+.1f}% | Final={m['final']:,.0f} | {ml} days")

    lookbacks = [10, 15, 20, 30, 40, 60]
    weights = [0.6, 0.7, 0.8, 0.9, 1.0]

    # Strategy A candidates and B candidates
    pairs = [
        ('V121', 'OV/ID'),
        ('Union', 'OV/ID'),
        ('Union', 'V121'),
    ]

    all_rotating_results = []

    for sA, sB in pairs:
        print(f"\n  --- Rotating A={sA} vs B={sB} ---")
        ret_A = ret_cache[sA]['ret']
        ret_B = ret_cache[sB]['ret']
        win_A = ret_cache[sA]['win']
        win_B = ret_cache[sB]['win']
        ml = min(ret_cache[sA]['ml'], ret_cache[sB]['ml'])

        # Also test static 50/50
        combined_ret_5050 = 0.5 * ret_A[:ml] + 0.5 * ret_B[:ml]
        eq_5050 = np.zeros(ml + 1)
        eq_5050[0] = float(CASH0)
        for i in range(ml):
            eq_5050[i+1] = eq_5050[i] * (1 + combined_ret_5050[i])
        m5050 = metrics_from_eq(eq_5050)
        all_rotating_results.append({
            'pair': f"{sA}/{sB}", 'lookback': 0, 'weight': 0.5,
            'ann': m5050['ann'], 'mdd': m5050['mdd'], 'sharpe': m5050['sharpe'],
            'desc': f"{sA}/{sB} static 50/50"
        })
        print(f"    Static 50/50: Ann={m5050['ann']:+8.1f}% | MDD={m5050['mdd']:6.1f}% | Sh={m5050['sharpe']:4.2f}")

        for lookback in lookbacks:
            for weight in weights:
                combined_ret = np.zeros(ml)
                for i in range(ml):
                    if i < lookback:
                        w_A = 0.5
                    else:
                        wr_A = np.mean(win_A[i-lookback:i])
                        wr_B = np.mean(win_B[i-lookback:i])
                        if wr_A >= wr_B:
                            w_A = weight
                        else:
                            w_A = 1.0 - weight
                    combined_ret[i] = w_A * ret_A[i] + (1.0 - w_A) * ret_B[i]

                eq_rot = np.zeros(ml + 1)
                eq_rot[0] = float(CASH0)
                for i in range(ml):
                    eq_rot[i+1] = eq_rot[i] * (1 + combined_ret[i])

                final = eq_rot[-1]
                ann_val = annual_return(final, CASH0, ml)
                pk = np.maximum.accumulate(eq_rot)
                mdd_val = np.min((eq_rot - pk) / pk * 100)
                sh_val = np.mean(combined_ret) / np.std(combined_ret) * np.sqrt(252) if np.std(combined_ret) > 0 else 0

                desc = f"{sA}/{sB} LB={lookback} W={weight}"
                result = {
                    'pair': f"{sA}/{sB}", 'lookback': lookback, 'weight': weight,
                    'ann': ann_val, 'mdd': mdd_val, 'sharpe': sh_val,
                    'desc': desc, 'sA': sA, 'sB': sB
                }
                all_rotating_results.append(result)

        # Print summary for this pair
        pair_results = [r for r in all_rotating_results if r['pair'] == f"{sA}/{sB}" and r['lookback'] > 0]
        pair_results.sort(key=lambda x: -x['ann'])
        print(f"    Top 5 rotating configs for {sA}/{sB}:")
        for i, r in enumerate(pair_results[:5]):
            print(f"      LB={r['lookback']:2d} W={r['weight']:.1f}: "
                  f"Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    # ============ SECTION 3: TOP 10 CONFIGURATIONS ============
    print("\n" + "=" * 120)
    print("  SECTION 3: TOP 10 CONFIGURATIONS BY ANNUAL RETURN")
    print("=" * 120)

    all_rotating_results.sort(key=lambda x: -x['ann'])
    for i, r in enumerate(all_rotating_results[:10]):
        print(f"  #{i+1}: {r['desc']:45s} | Ann={r['ann']:+8.1f}% | "
              f"MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    # Also top by Sharpe
    print(f"\n  TOP 10 BY SHARPE:")
    sorted_sh = sorted(all_rotating_results, key=lambda x: -x['sharpe'])
    for i, r in enumerate(sorted_sh[:10]):
        print(f"  #{i+1}: {r['desc']:45s} | Ann={r['ann']:+8.1f}% | "
              f"MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    # ============ SECTION 4: WALK-FORWARD FOR TOP 5 ============
    print("\n" + "=" * 120)
    print("  SECTION 4: WALK-FORWARD FOR TOP 5 (per calendar year 2020-2025)")
    print("=" * 120)

    # Get top 5 by annual return (excluding static 50/50 for variety — include it if it's truly top)
    top5 = all_rotating_results[:5]

    # Also ensure we have at least one Union-based config in WF if not in top 5
    union_configs = [r for r in all_rotating_results if 'Union' in r['pair'] and r['lookback'] > 0]
    union_configs.sort(key=lambda x: -x['ann'])
    if union_configs and not any('Union' in r['pair'] for r in top5):
        top5.append(union_configs[0])

    # Also always test the V135 original config: V121/OV_ID LB=20 W=0.8
    v135_original = [r for r in all_rotating_results if r['pair'] == 'V121/OV/ID' and r['lookback'] == 20 and r['weight'] == 0.8]
    if not v135_original:
        v135_original = [r for r in all_rotating_results if r['desc'] == 'V121/OV/ID static 50/50']

    for idx, config in enumerate(top5):
        sA_name = config.get('sA', config['pair'].split('/')[0])
        sB_name = config.get('sB', config['pair'].split('/')[1])
        sA_func = signal_pool[sA_name]
        sB_func = signal_pool[sB_name]
        lookback = config['lookback']
        weight = config['weight']

        print(f"\n  --- WF #{idx+1}: {config['desc']} ---")
        print(f"      Full-period: Ann={config['ann']:+.1f}% | MDD={config['mdd']:.1f}% | Sh={config['sharpe']:.2f}")

        wf_results = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None:
                continue

            if lookback == 0:
                # Static 50/50
                eq_A_y, _ = backtest_equity(sA_func, hold=1, top_n=1, start_di=ys, end_di=ye)
                eq_B_y, _ = backtest_equity(sB_func, hold=1, top_n=1, start_di=ys, end_di=ye)
                ml_y = min(len(eq_A_y), len(eq_B_y)) - 1
                if ml_y <= 0:
                    wf_results[yr] = 0
                    continue
                ret_A_y = np.diff(eq_A_y[:ml_y+1]) / eq_A_y[:ml_y]
                ret_B_y = np.diff(eq_B_y[:ml_y+1]) / eq_B_y[:ml_y]
                ret_A_y = np.where(np.isfinite(ret_A_y), ret_A_y, 0)
                ret_B_y = np.where(np.isfinite(ret_B_y), ret_B_y, 0)
                combined_y = 0.5 * ret_A_y + 0.5 * ret_B_y
            else:
                # Rotating with expanding window for lookback calculation
                # Use data from MIN_TRAIN to ye for lookback context
                eq_A_y, _ = backtest_equity(sA_func, hold=1, top_n=1, start_di=MIN_TRAIN, end_di=ye)
                eq_B_y, _ = backtest_equity(sB_func, hold=1, top_n=1, start_di=MIN_TRAIN, end_di=ye)
                ml_y_full = min(len(eq_A_y), len(eq_B_y)) - 1
                if ml_y_full <= 0:
                    wf_results[yr] = 0
                    continue
                ret_A_full = np.diff(eq_A_y[:ml_y_full+1]) / eq_A_y[:ml_y_full]
                ret_B_full = np.diff(eq_B_y[:ml_y_full+1]) / eq_B_y[:ml_y_full]
                ret_A_full = np.where(np.isfinite(ret_A_full), ret_A_full, 0)
                ret_B_full = np.where(np.isfinite(ret_B_full), ret_B_full, 0)
                win_A_full = (ret_A_full > 0).astype(float)
                win_B_full = (ret_B_full > 0).astype(float)

                # Find the offset for start of year ys
                # The equity curve starts at MIN_TRAIN, so index 0 = MIN_TRAIN
                yr_offset = ys - MIN_TRAIN
                yr_end_offset = ye - MIN_TRAIN
                if yr_offset < 0: yr_offset = 0
                if yr_end_offset > ml_y_full: yr_end_offset = ml_y_full

                combined_y = np.zeros(yr_end_offset - yr_offset)
                for j_local in range(len(combined_y)):
                    j_global = yr_offset + j_local
                    if j_global < lookback:
                        w_A = 0.5
                    else:
                        wr_A_l = np.mean(win_A_full[j_global-lookback:j_global])
                        wr_B_l = np.mean(win_B_full[j_global-lookback:j_global])
                        if wr_A_l >= wr_B_l:
                            w_A = weight
                        else:
                            w_A = 1.0 - weight
                    combined_y[j_local] = w_A * ret_A_full[j_global] + (1.0 - w_A) * ret_B_full[j_global]

            # Build equity from combined returns for this year
            eq_yr = np.zeros(len(combined_y) + 1)
            eq_yr[0] = float(CASH0)
            for j in range(len(combined_y)):
                eq_yr[j+1] = eq_yr[j] * (1 + combined_y[j])
            nd_y = ye - ys
            ann_y = annual_return(eq_yr[-1], CASH0, nd_y)
            wf_results[yr] = ann_y

        pos = sum(1 for v in wf_results.values() if v > 0)
        avg = np.mean(list(wf_results.values())) if wf_results else 0
        ws = " | ".join([f"{yr}:{v:+.0f}%" for yr, v in sorted(wf_results.items())])
        print(f"      WF: {pos}/6 positive | Avg={avg:>+7.0f}% | {ws}")

    # ============ SECTION 5: UNION + ROTATING ============
    print("\n" + "=" * 120)
    print("  SECTION 5: UNION + ROTATING (MOST PROMISING COMBINATIONS)")
    print("=" * 120)

    # Test specific Union-based rotating combinations
    union_rotating_configs = [
        ("Union/OV_ID static 50/50", 'Union', 'OV/ID', 0, 0.5),
        ("Union/V121 static 50/50", 'Union', 'V121', 0, 0.5),
        ("Union/OV_ID LB=20 W=0.8", 'Union', 'OV/ID', 20, 0.8),
        ("Union/OV_ID LB=20 W=0.9", 'Union', 'OV/ID', 20, 0.9),
        ("Union/OV_ID LB=15 W=0.8", 'Union', 'OV/ID', 15, 0.8),
        ("Union/OV_ID LB=30 W=0.8", 'Union', 'OV/ID', 30, 0.8),
        ("Union/OV_ID LB=10 W=0.9", 'Union', 'OV/ID', 10, 0.9),
        ("Union/OV_ID LB=20 W=1.0", 'Union', 'OV/ID', 20, 1.0),
        ("Union/V121 LB=20 W=0.8", 'Union', 'V121', 20, 0.8),
        ("Union/V121 LB=20 W=0.9", 'Union', 'V121', 20, 0.9),
        ("Union/V121 LB=15 W=0.8", 'Union', 'V121', 15, 0.8),
        ("Union/V121 LB=20 W=1.0", 'Union', 'V121', 20, 1.0),
    ]

    print(f"\n  {'Config':45s} | {'Ann':>8s} | {'MDD':>6s} | {'Sh':>4s} | WF yrs | WF Avg")
    print(f"  {'-'*45}-+-{'-'*8}-+-{'-'*6}-+-{'-'*4}-+-{'-'*6}-+-{'-'*7}")

    union_rot_results = []
    for desc, sA_name, sB_name, lookback, weight in union_rotating_configs:
        sA_func = signal_pool[sA_name]
        sB_func = signal_pool[sB_name]
        r = backtest_rotating(sA_func, sB_func, lookback if lookback > 0 else 20, weight)
        # For static 50/50, force 50/50
        if lookback == 0:
            eq_A, _ = backtest_equity(sA_func, hold=1, top_n=1, start_di=MIN_TRAIN, end_di=ND)
            eq_B, _ = backtest_equity(sB_func, hold=1, top_n=1, start_di=MIN_TRAIN, end_di=ND)
            ml_s = min(len(eq_A), len(eq_B)) - 1
            ret_A_s = np.diff(eq_A[:ml_s+1]) / eq_A[:ml_s]
            ret_B_s = np.diff(eq_B[:ml_s+1]) / eq_B[:ml_s]
            ret_A_s = np.where(np.isfinite(ret_A_s), ret_A_s, 0)
            ret_B_s = np.where(np.isfinite(ret_B_s), ret_B_s, 0)
            combined_s = 0.5 * ret_A_s + 0.5 * ret_B_s
            eq_s = np.zeros(ml_s + 1)
            eq_s[0] = float(CASH0)
            for i in range(ml_s):
                eq_s[i+1] = eq_s[i] * (1 + combined_s[i])
            r = metrics_from_eq(eq_s)

        # Walk-forward for this config
        wf_res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue

            if lookback == 0:
                # Static 50/50
                eq_A_y, _ = backtest_equity(sA_func, hold=1, top_n=1, start_di=ys, end_di=ye)
                eq_B_y, _ = backtest_equity(sB_func, hold=1, top_n=1, start_di=ys, end_di=ye)
                ml_y = min(len(eq_A_y), len(eq_B_y)) - 1
                if ml_y <= 0: wf_res[yr] = 0; continue
                rA = np.diff(eq_A_y[:ml_y+1]) / eq_A_y[:ml_y]
                rB = np.diff(eq_B_y[:ml_y+1]) / eq_B_y[:ml_y]
                rA = np.where(np.isfinite(rA), rA, 0)
                rB = np.where(np.isfinite(rB), rB, 0)
                cy = 0.5 * rA + 0.5 * rB
            else:
                # Rotating with expanding window
                eq_A_y, _ = backtest_equity(sA_func, hold=1, top_n=1, start_di=MIN_TRAIN, end_di=ye)
                eq_B_y, _ = backtest_equity(sB_func, hold=1, top_n=1, start_di=MIN_TRAIN, end_di=ye)
                ml_yf = min(len(eq_A_y), len(eq_B_y)) - 1
                if ml_yf <= 0: wf_res[yr] = 0; continue
                rAf = np.diff(eq_A_y[:ml_yf+1]) / eq_A_y[:ml_yf]
                rBf = np.diff(eq_B_y[:ml_yf+1]) / eq_B_y[:ml_yf]
                rAf = np.where(np.isfinite(rAf), rAf, 0)
                rBf = np.where(np.isfinite(rBf), rBf, 0)
                wAf = (rAf > 0).astype(float)
                wBf = (rBf > 0).astype(float)
                yr_off = ys - MIN_TRAIN
                yr_end = ye - MIN_TRAIN
                if yr_off < 0: yr_off = 0
                if yr_end > ml_yf: yr_end = ml_yf
                cy = np.zeros(yr_end - yr_off)
                for jl in range(len(cy)):
                    jg = yr_off + jl
                    if jg < lookback:
                        wA = 0.5
                    else:
                        wrAl = np.mean(wAf[jg-lookback:jg])
                        wrBl = np.mean(wBf[jg-lookback:jg])
                        wA = weight if wrAl >= wrBl else 1.0 - weight
                    cy[jl] = wA * rAf[jg] + (1.0 - wA) * rBf[jg]

            eq_yr = np.zeros(len(cy) + 1)
            eq_yr[0] = float(CASH0)
            for j in range(len(cy)):
                eq_yr[j+1] = eq_yr[j] * (1 + cy[j])
            nd_y = ye - ys
            wf_res[yr] = annual_return(eq_yr[-1], CASH0, nd_y)

        pos = sum(1 for v in wf_res.values() if v > 0)
        avg = np.mean(list(wf_res.values())) if wf_res else 0
        print(f"  {desc:45s} | {r['ann']:+8.1f}% | {r['mdd']:6.1f}% | {r['sharpe']:4.2f} | {pos}/6   | {avg:>+7.0f}%")

        union_rot_results.append({
            'desc': desc, 'ann': r['ann'], 'mdd': r['mdd'], 'sharpe': r['sharpe'],
            'wf_pos': pos, 'wf_avg': avg, 'wf': wf_res,
            'sA': sA_name, 'sB': sB_name, 'lookback': lookback, 'weight': weight
        })

    # Detailed WF breakdown for best Union configs
    print(f"\n  Detailed Walk-Forward for Union+Rotating configs:")
    union_rot_results.sort(key=lambda x: (-x['wf_pos'], -x['wf_avg']))
    for i, cfg in enumerate(union_rot_results[:5]):
        ws = " | ".join([f"{yr}:{v:+.0f}%" for yr, v in sorted(cfg['wf'].items())])
        print(f"  #{i+1}: {cfg['desc']:45s} | {cfg['wf_pos']}/6 | Avg={cfg['wf_avg']:>+7.0f}% | {ws}")

    # Also test Union standalone (no rotation) as reference
    print(f"\n  Union standalone (no rotation) WF reference:")
    w_union = wf(sig_union, hold=1, topn=1)
    ws = " | ".join([f"{yr}:{v:+.0f}%" for yr, v in sorted(w_union.items())])
    pos_u = sum(1 for v in w_union.values() if v > 0)
    avg_u = np.mean(list(w_union.values())) if w_union else 0
    print(f"  Union standalone:                        | {pos_u}/6 | Avg={avg_u:>+7.0f}% | {ws}")

    # ============ SECTION 5 BEST CONFIGURATION DETAILS ============
    print("\n" + "=" * 120)
    print("  SECTION 5: BEST CONFIGURATION DETAILS")
    print("=" * 120)

    # Best by annual return (full period)
    best_ann = max(all_rotating_results, key=lambda x: x['ann'])
    print(f"\n  Best by Annual Return (full period):")
    print(f"    Config:  {best_ann['desc']}")
    print(f"    Annual:  {best_ann['ann']:+.1f}%")
    print(f"    MDD:     {best_ann['mdd']:.1f}%")
    print(f"    Sharpe:  {best_ann['sharpe']:.2f}")

    # Best by Sharpe
    best_sh = max(all_rotating_results, key=lambda x: x['sharpe'])
    print(f"\n  Best by Sharpe Ratio:")
    print(f"    Config:  {best_sh['desc']}")
    print(f"    Annual:  {best_sh['ann']:+.1f}%")
    print(f"    MDD:     {best_sh['mdd']:.1f}%")
    print(f"    Sharpe:  {best_sh['sharpe']:.2f}")

    # Best WF-validated (from union rotating)
    if union_rot_results:
        # Sort by WF score: most positive years, then highest avg
        best_wf = max(union_rot_results, key=lambda x: (x['wf_pos'], x['wf_avg']))
        print(f"\n  Best Walk-Forward Validated (Union+Rotating):")
        print(f"    Config:  {best_wf['desc']}")
        print(f"    Annual:  {best_wf['ann']:+.1f}%")
        print(f"    MDD:     {best_wf['mdd']:.1f}%")
        print(f"    Sharpe:  {best_wf['sharpe']:.2f}")
        print(f"    WF:      {best_wf['wf_pos']}/6 positive | Avg={best_wf['wf_avg']:>+7.0f}%")
        for yr, val in sorted(best_wf['wf'].items()):
            tag = "OK" if val > 0 else "LOSS"
            print(f"      {yr}: {val:+8.1f}%  [{tag}]")

    # ============ COMPREHENSIVE SUMMARY ============
    print("\n" + "=" * 120)
    print("  COMPREHENSIVE SUMMARY")
    print("=" * 120)

    print(f"\n  --- Baseline signals (standalone) ---")
    for name, r in baseline_results.items():
        print(f"    {name:40s}: Ann={r['ann']:+8.1f}% | WR={r['wr']:5.1f}% | N={r['n']:4d} | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    print(f"\n  --- Rotating Portfolio Top 10 (full period, NO WF) ---")
    for i, r in enumerate(all_rotating_results[:10]):
        print(f"    #{i+1}: {r['desc']:45s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    print(f"\n  --- Rotating Portfolio Top 10 (WF-validated from Union sweep) ---")
    union_rot_results.sort(key=lambda x: (-x['wf_pos'], -x['wf_avg']))
    for i, cfg in enumerate(union_rot_results[:10]):
        print(f"    #{i+1}: {cfg['desc']:45s} | {cfg['wf_pos']}/6 | Avg={cfg['wf_avg']:>+7.0f}% | Ann={cfg['ann']:+8.1f}%")

    print(f"\n  --- KEY FINDING ---")
    # Compare V135 original vs WF-validated
    v135_configs = [r for r in all_rotating_results if r['pair'] == 'V121/OV/ID']
    v135_best = max(v135_configs, key=lambda x: x['ann']) if v135_configs else None
    if v135_best:
        print(f"    V135 original approach (best V121/OV rotating): {v135_best['ann']:+.1f}% annual (NO WF validation)")

    # Find the best WF-validated config overall
    # Run WF on the top 3 from all_rotating_results
    print(f"\n  Running WF validation on overall top 3 rotating configs...")
    for idx in range(min(3, len(all_rotating_results))):
        cfg = all_rotating_results[idx]
        sA_n = cfg.get('sA', cfg['pair'].split('/')[0])
        sB_n = cfg.get('sB', cfg['pair'].split('/')[1])
        if sA_n not in signal_pool or sB_n not in signal_pool: continue
        lb = cfg['lookback']
        wt = cfg['weight']
        if lb == 0: continue  # skip static

        sA_f = signal_pool[sA_n]
        sB_f = signal_pool[sB_n]

        wf_r = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue
            eq_Af, _ = backtest_equity(sA_f, hold=1, top_n=1, start_di=MIN_TRAIN, end_di=ye)
            eq_Bf, _ = backtest_equity(sB_f, hold=1, top_n=1, start_di=MIN_TRAIN, end_di=ye)
            mlf = min(len(eq_Af), len(eq_Bf)) - 1
            if mlf <= 0: wf_r[yr] = 0; continue
            rAf = np.diff(eq_Af[:mlf+1]) / eq_Af[:mlf]
            rBf = np.diff(eq_Bf[:mlf+1]) / eq_Bf[:mlf]
            rAf = np.where(np.isfinite(rAf), rAf, 0)
            rBf = np.where(np.isfinite(rBf), rBf, 0)
            wAf = (rAf > 0).astype(float)
            wBf = (rBf > 0).astype(float)
            yr_off = ys - MIN_TRAIN
            yr_end = ye - MIN_TRAIN
            if yr_off < 0: yr_off = 0
            if yr_end > mlf: yr_end = mlf
            cy = np.zeros(yr_end - yr_off)
            for jl in range(len(cy)):
                jg = yr_off + jl
                if jg < lb:
                    wA = 0.5
                else:
                    wrA = np.mean(wAf[jg-lb:jg])
                    wrB = np.mean(wBf[jg-lb:jg])
                    wA = wt if wrA >= wrB else 1.0 - wt
                cy[jl] = wA * rAf[jg] + (1.0 - wA) * rBf[jg]
            eq_y = np.zeros(len(cy) + 1)
            eq_y[0] = float(CASH0)
            for j in range(len(cy)):
                eq_y[j+1] = eq_y[j] * (1 + cy[j])
            wf_r[yr] = annual_return(eq_y[-1], CASH0, ye - ys)

        pos = sum(1 for v in wf_r.values() if v > 0)
        avg = np.mean(list(wf_r.values())) if wf_r else 0
        ws = " | ".join([f"{yr}:{v:+.0f}%" for yr, v in sorted(wf_r.items())])
        print(f"    #{idx+1} {cfg['desc']:45s} | {pos}/6 | Avg={avg:>+7.0f}% | {ws}")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
