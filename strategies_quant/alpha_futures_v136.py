"""
Alpha Futures V136 — OI CONVICTION + VOLATILITY BREAKOUT
=============================================================================
Tests two families of signals using Open Interest and Volatility:

Part 1: OI-Based Conviction Signals
  A) OI Surge + Momentum: OI_CHG5 > 5% + ROC(5)>1% + Z>1.5
  B) OI Acceleration: OI_CHG5 accelerating 2+ days + ROC(5)>1% + Z>1.5
  C) Volume-OI Divergence: VOL/OI > 1.2x avg + ROC(5)>1%
  D) OI + Price Co-movement: rolling corr(OI_chg, price_chg) extremes

Part 2: Volatility Breakout
  E) ATR Compression + Breakout: ATR_pct<30 for 3+ days then breakout
  F) Narrow Range -> Expansion: NR day followed by expansion
  G) Combined Mega: OI + Volatility + Momentum all confirm

All signals use NEXT-OPEN execution: signal at close di, entry at O[si, di+1]
Walk-forward 2020-2025.
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
    print("  V136 — OI CONVICTION + VOLATILITY BREAKOUT")
    print("=" * 120)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    print(f"  {NS} commodities, {ND} days")

    print("\n[Precompute]...", flush=True)
    t0 = time.time()

    # --- Basic returns and momentum ---
    RET = np.full((NS, ND), np.nan)
    ROC5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100
        ROC5[si] = talib.ROC(c, timeperiod=5)

    # --- Z-score ---
    ZSCORE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = RET[si, di-20:di]
            v = rets[~np.isnan(rets)]
            if len(v) < 10: continue
            s = np.std(v, ddof=1)
            if s > 0 and not np.isnan(RET[si, di]):
                ZSCORE[si, di] = (RET[si, di] - np.mean(v)) / s

    # --- ATR(14) ---
    ATR14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        ATR14[si] = talib.ATR(H[si].astype(np.float64), L[si].astype(np.float64),
                               C[si].astype(np.float64), timeperiod=14)

    # --- OI change rate (5-day) ---
    OI_CHG5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        oi = OI[si].astype(np.float64)
        for di in range(5, ND):
            if not np.isnan(oi[di]) and not np.isnan(oi[di-5]) and oi[di-5] > 0:
                OI_CHG5[si, di] = (oi[di] - oi[di-5]) / oi[di-5] * 100

    # --- VOL/OI ratio and 20-day average ---
    VOL_OI = np.full((NS, ND), np.nan)
    AVG_VOL_OI20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        v = V[si].astype(np.float64)
        oi = OI[si].astype(np.float64)
        for di in range(ND):
            if not np.isnan(v[di]) and not np.isnan(oi[di]) and oi[di] > 0:
                VOL_OI[si, di] = v[di] / oi[di]
        # 20-day rolling average of VOL_OI
        for di in range(20, ND):
            vals = VOL_OI[si, di-20:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 10:
                AVG_VOL_OI20[si, di] = np.mean(valid)

    # --- ATR percentile vs 60-day range ---
    ATR_PCT = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(60, ND):
            window = ATR14[si, di-60:di+1]
            valid = window[~np.isnan(window)]
            if len(valid) < 20: continue
            cur = ATR14[si, di]
            if np.isnan(cur): continue
            lo = np.min(valid)
            hi = np.max(valid)
            if hi > lo:
                ATR_PCT[si, di] = (cur - lo) / (hi - lo) * 100

    # --- ATR compression streak (days with ATR_PCT < 30) ---
    ATR_COMP_STREAK = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        streak = 0
        for di in range(ND):
            if not np.isnan(ATR_PCT[si, di]) and ATR_PCT[si, di] < 30:
                streak += 1
            else:
                streak = 0
            ATR_COMP_STREAK[si, di] = streak

    # --- Daily range as % of close ---
    RANGE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            h, l, c = H[si, di], L[si, di], C[si, di]
            if not np.isnan(h) and not np.isnan(l) and not np.isnan(c) and c > 0:
                RANGE[si, di] = (h - l) / c * 100

    # --- 10-day average range ---
    AVG_RANGE10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            vals = RANGE[si, di-10:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 5:
                AVG_RANGE10[si, di] = np.mean(valid)

    # --- Rolling 10-day correlation between OI changes and price changes ---
    OI_PRICE_CORR = np.full((NS, ND), np.nan)
    for si in range(NS):
        oi = OI[si].astype(np.float64)
        c = C[si].astype(np.float64)
        for di in range(11, ND):
            oi_diffs = []
            pr_diffs = []
            for d in range(di-10, di+1):
                if d < 1: continue
                if np.isnan(oi[d]) or np.isnan(oi[d-1]) or oi[d-1] <= 0: continue
                if np.isnan(c[d]) or np.isnan(c[d-1]) or c[d-1] <= 0: continue
                oi_diffs.append((oi[d] - oi[d-1]) / oi[d-1])
                pr_diffs.append((c[d] - c[d-1]) / c[d-1])
            if len(oi_diffs) >= 8:
                oi_a = np.array(oi_diffs)
                pr_a = np.array(pr_diffs)
                if np.std(oi_a) > 0 and np.std(pr_a) > 0:
                    OI_PRICE_CORR[si, di] = np.corrcoef(oi_a, pr_a)[0, 1]

    print(f"  Done ({time.time()-t0:.1f}s)")

    # ============ BACKTEST ENGINE ============

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

    # ============ SIGNAL DEFINITIONS ============

    # A) OI Surge + Momentum
    # Signal: ROC(5) > 1% AND Z > 1.5 AND OI_CHG5 > 5%
    # Score: ROC(5) * Z * (1 + OI_CHG5/10)
    def sig_oi_surge(di, edi):
        c = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]; oi_chg = OI_CHG5[s, di]
            if any(np.isnan(x) for x in [roc, zs, oi_chg]): continue
            if roc <= 1.0 or zs <= 1.5 or oi_chg <= 5.0: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = roc * zs * (1 + oi_chg / 10)
            c.append((score, s, ep, 'oi_surge'))
        return c

    # B) OI Acceleration
    # OI_CHG5[today] > OI_CHG5[yesterday] > OI_CHG5[2 days ago] (2+ days accelerating)
    # Signal: ROC(5) > 1% AND Z > 1.5 AND OI accelerating
    # Score: ROC(5) * Z * OI_CHG5
    def sig_oi_accel(di, edi):
        c = []
        for s in range(NS):
            if di < 2: continue
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            oi0 = OI_CHG5[s, di]; oi1 = OI_CHG5[s, di-1]; oi2 = OI_CHG5[s, di-2]
            if any(np.isnan(x) for x in [roc, zs, oi0, oi1, oi2]): continue
            if roc <= 1.0 or zs <= 1.5: continue
            if not (oi0 > oi1 > oi2): continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = roc * zs * oi0
            c.append((score, s, ep, 'oi_accel'))
        return c

    # C) Volume-OI Divergence
    # VOL_OI > 1.2 * 20-day avg VOL_OI AND price up (ROC5 > 1%)
    # Score: ROC(5) * Z * (VOL_OI / avg_VOL_OI)
    def sig_vol_oi_div(di, edi):
        c = []
        for s in range(NS):
            roc = ROC5[s, di]; vo = VOL_OI[s, di]; avg_vo = AVG_VOL_OI20[s, di]
            if any(np.isnan(x) for x in [roc, vo, avg_vo]): continue
            if roc <= 1.0 or avg_vo <= 0 or vo <= 1.2 * avg_vo: continue
            zs = ZSCORE[s, di]
            if np.isnan(zs): continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = roc * zs * (vo / avg_vo) if zs > 0 else roc * (vo / avg_vo)
            c.append((score, s, ep, 'vol_oi_div'))
        return c

    # D) OI + Price Co-movement
    # D-A (long): correlation > 0.5 AND ROC(5) > 1% AND Z > 1.5
    # D-B (bottom): correlation < -0.5 AND ROC(5) > 1% AND oversold bounce (Z recovering)
    def sig_oi_comove(di, edi):
        c = []
        for s in range(NS):
            corr = OI_PRICE_CORR[s, di]
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if any(np.isnan(x) for x in [corr, roc, zs]): continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue

            # D-A: Trend confirmation — OI and price moving together
            if corr > 0.5 and roc > 1.0 and zs > 1.5:
                score = roc * zs * corr
                c.append((score, s, ep, 'oi_comove_up'))

            # D-B: Bottom fishing — OI rising while price was falling, now bouncing
            if corr < -0.5 and roc > 1.0 and di >= 1:
                # Check for oversold bounce: prior day Z was negative, today recovering
                prev_zs = ZSCORE[s, di-1] if di >= 1 else np.nan
                if not np.isnan(prev_zs) and prev_zs < 0:
                    score = roc * abs(corr) * (1 + abs(prev_zs))
                    c.append((score, s, ep, 'oi_comove_bottom'))
        return c

    # E) ATR Compression + Breakout
    # Compression: ATR_pct < 30 for 3+ days, then breakout: price > 5-day high AND ROC(5) > 0
    # Score: ROC(5) * Z * (100 - ATR_pct) / 50
    def sig_atr_breakout(di, edi):
        c = []
        for s in range(NS):
            streak = ATR_COMP_STREAK[s, di-1] if di >= 1 else 0  # was compressed yesterday
            if streak < 3: continue
            atr_pct = ATR_PCT[s, di]
            roc = ROC5[s, di]
            if np.isnan(atr_pct) or np.isnan(roc) or roc <= 0: continue
            if di < 5: continue
            h5 = H[s, di-4:di+1]
            if any(np.isnan(x) for x in h5): continue
            # Price breaks above 5-day high (close > max of last 5 days including today)
            cp = C[s, di]
            if np.isnan(cp): continue
            h5_prev = np.max(H[s, di-4:di])  # high of previous 4 days
            if np.isnan(h5_prev) or cp <= h5_prev: continue  # close must exceed recent high
            zs = ZSCORE[s, di]
            if np.isnan(zs): zs = 1.0
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = roc * max(zs, 0.5) * (100 - atr_pct) / 50
            c.append((score, s, ep, 'atr_breakout'))
        return c

    # F) Narrow Range -> Expansion
    # NR day: RANGE < 50% of 10-day avg range
    # Signal: NR day followed by close > NR day high AND ROC(5) > 0.5%
    def sig_nr_expansion(di, edi):
        c = []
        for s in range(NS):
            # Check if yesterday was a NR day
            if di < 1: continue
            rng_yd = RANGE[s, di-1]
            avg_rng = AVG_RANGE10[s, di-1]
            if np.isnan(rng_yd) or np.isnan(avg_rng) or avg_rng <= 0: continue
            if rng_yd >= 0.5 * avg_rng: continue  # not a NR day
            # Today: close > yesterday's high
            cp = C[s, di]; h_yd = H[s, di-1]
            roc = ROC5[s, di]
            if np.isnan(cp) or np.isnan(h_yd) or np.isnan(roc): continue
            if cp <= h_yd or roc <= 0.5: continue
            zs = ZSCORE[s, di]
            if np.isnan(zs): zs = 1.0
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = roc * max(zs, 0.5) * (avg_rng / rng_yd if rng_yd > 0 else 1)
            c.append((score, s, ep, 'nr_expand'))
        return c

    # G) Combined Mega: OI + Volatility + Momentum
    # ROC(5) > 1% AND Z > 1.5 AND OI_CHG5 > 3% AND ATR_pct < 50
    # Score: ROC(5) * Z * (1 + OI_CHG5/10) * (100 - ATR_pct)/50
    def sig_mega(di, edi):
        c = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]; oi_chg = OI_CHG5[s, di]
            atr_pct = ATR_PCT[s, di]
            if any(np.isnan(x) for x in [roc, zs, oi_chg, atr_pct]): continue
            if roc <= 1.0 or zs <= 1.5 or oi_chg <= 3.0 or atr_pct >= 50: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = roc * zs * (1 + oi_chg / 10) * (100 - atr_pct) / 50
            c.append((score, s, ep, 'mega'))
        return c

    # ============ SECTION 1: ALL INDIVIDUAL SIGNALS ============
    print("\n" + "=" * 120)
    print("  SECTION 1: OI CONVICTION + VOLATILITY BREAKOUT SIGNALS")
    print("=" * 120)

    configs = [
        ("A) OI Surge + Momentum", sig_oi_surge, 1, 1),
        ("B) OI Acceleration", sig_oi_accel, 1, 1),
        ("C) Volume-OI Divergence", sig_vol_oi_div, 1, 1),
        ("D) OI+Price Co-movement", sig_oi_comove, 1, 1),
        ("E) ATR Compression+Breakout", sig_atr_breakout, 1, 1),
        ("F) Narrow Range->Expansion", sig_nr_expansion, 1, 1),
        ("G) Mega: OI+Vol+Mom", sig_mega, 1, 1),
    ]

    results = {}
    for name, func, hold, topn in configs:
        r = backtest(func, hold=hold, top_n=topn, desc=name)
        results[name] = r
        pr(r, label=name)

    # ============ SECTION 2: TOP_N x HOLD for best ============
    print("\n" + "=" * 120)
    print("  SECTION 2: TOP_N x HOLD for top signals")
    print("=" * 120)

    best4 = sorted(results.items(), key=lambda x: -x[1]['ann'])[:4]
    for name, r in best4:
        func_map = {n: f for n, f, _, _ in configs}
        func = func_map[name]
        print(f"\n  {name}:")
        for topn in [1, 2, 3]:
            for hold in [1, 2, 3]:
                r = backtest(func, hold=hold, top_n=topn, desc=f"{name} t={topn} h={hold}")
                print(f"    top_n={topn} hold={hold}: Ann={r['ann']:+8.1f}% | "
                      f"WR={r['wr']:5.1f}% | N={r['n']:4d} | MDD={r['mdd']:6.1f}%")

    # ============ SECTION 3: COMBINED STRATEGIES ============
    print("\n" + "=" * 120)
    print("  SECTION 3: COMBINED STRATEGIES (union of multiple signals)")
    print("=" * 120)

    # Union of OI signals
    def sig_oi_union(di, edi):
        all_sigs = {}
        for item in sig_oi_surge(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc
            all_sigs[s][2].append(st)
        for item in sig_oi_accel(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc
            all_sigs[s][2].append(st)
        for item in sig_vol_oi_div(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc
            all_sigs[s][2].append(st)
        for item in sig_oi_comove(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc
            all_sigs[s][2].append(st)
        return [(sc, s, ep, '+'.join(sigs)) for s, (sc, ep, sigs) in all_sigs.items()]

    # Union of volatility signals
    def sig_vol_union(di, edi):
        all_sigs = {}
        for item in sig_atr_breakout(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc
            all_sigs[s][2].append(st)
        for item in sig_nr_expansion(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc
            all_sigs[s][2].append(st)
        return [(sc, s, ep, '+'.join(sigs)) for s, (sc, ep, sigs) in all_sigs.items()]

    # Union of ALL signals
    def sig_all_union(di, edi):
        all_sigs = {}
        for func in [sig_oi_surge, sig_oi_accel, sig_vol_oi_div, sig_oi_comove,
                      sig_atr_breakout, sig_nr_expansion, sig_mega]:
            for item in func(di, edi):
                sc, s, ep, st = item
                if s not in all_sigs: all_sigs[s] = [0, ep, []]
                all_sigs[s][0] += sc
                all_sigs[s][2].append(st)
        return [(sc, s, ep, '+'.join(sigs)) for s, (sc, ep, sigs) in all_sigs.items()]

    # OI + Vol combined (no mega to avoid double-counting)
    def sig_oi_vol(di, edi):
        all_sigs = {}
        for func in [sig_oi_surge, sig_oi_accel, sig_vol_oi_div, sig_oi_comove,
                      sig_atr_breakout, sig_nr_expansion]:
            for item in func(di, edi):
                sc, s, ep, st = item
                if s not in all_sigs: all_sigs[s] = [0, ep, []]
                all_sigs[s][0] += sc
                all_sigs[s][2].append(st)
        return [(sc, s, ep, '+'.join(sigs)) for s, (sc, ep, sigs) in all_sigs.items()]

    # Cascade: Mega first, then OI surge, then ATR breakout, then others
    def sig_cascade(di, edi):
        mega = sig_mega(di, edi)
        if mega: return mega
        surge = sig_oi_surge(di, edi)
        if surge: return surge
        atr = sig_atr_breakout(di, edi)
        if atr: return atr
        accel = sig_oi_accel(di, edi)
        if accel: return accel
        comove = sig_oi_comove(di, edi)
        if comove: return comove
        return []

    combo_configs = [
        ("OI Union (A+B+C+D)", sig_oi_union, 1, 1),
        ("Vol Union (E+F)", sig_vol_union, 1, 1),
        ("OI+Vol Union (A-F)", sig_oi_vol, 1, 1),
        ("ALL Union (A-G)", sig_all_union, 1, 1),
        ("Cascade (G>A>E>B>D)", sig_cascade, 1, 1),
        ("OI Union t=2", sig_oi_union, 1, 2),
        ("OI+Vol Union t=2", sig_oi_vol, 1, 2),
        ("ALL Union t=2", sig_all_union, 1, 2),
        ("OI+Vol Union t=2 h=2", sig_oi_vol, 2, 2),
        ("ALL Union t=2 h=2", sig_all_union, 2, 2),
        ("Cascade t=2", sig_cascade, 1, 2),
        ("Mega t=2 h=2", sig_mega, 2, 2),
    ]

    combo_results = {}
    for name, func, hold, topn in combo_configs:
        r = backtest(func, hold=hold, top_n=topn, desc=name)
        combo_results[name] = r
        pr(r, label=name)

    # ============ SECTION 4: WALK-FORWARD ============
    print("\n" + "=" * 120)
    print("  SECTION 4: WALK-FORWARD (2020-2025)")
    print("=" * 120)

    all_configs = configs + combo_configs
    for name, func, hold, topn in all_configs:
        w = wf(func, hold=hold, topn=topn)
        ws = " | ".join([f"{yr}:{v:+.0f}%" for yr, v in sorted(w.items())])
        pos = sum(1 for v in w.values() if v > 0)
        avg = np.mean(list(w.values())) if w else 0
        print(f"  {name:60s} | {pos}/6 | Avg={avg:>+7.0f}% | {ws}")

    # ============ SECTION 5: SIGNAL BREAKDOWN ============
    print("\n" + "=" * 120)
    print("  SECTION 5: SIGNAL TYPE BREAKDOWN")
    print("=" * 120)

    for name in ["ALL Union (A-G)", "OI+Vol Union (A-F)", "Cascade (G>A>E>B>D)"]:
        r = combo_results.get(name, results.get(name))
        if r is None: continue
        bd = r.get('sig_breakdown', {})
        print(f"\n  {name}:")
        for sig, data in sorted(bd.items(), key=lambda x: -x[1]['n']):
            wr = data['w'] / data['n'] * 100 if data['n'] > 0 else 0
            ap = data['pnl'] / data['n'] if data['n'] > 0 else 0
            print(f"    {sig:30s}: N={data['n']:4d} | WR={wr:5.1f}% | AvgPnL={ap:+.2f}%")

    # ============ SUMMARY ============
    print("\n" + "=" * 120)
    print("  SUMMARY: TOP 20 BY ANNUAL RETURN")
    print("=" * 120)

    all_r = {**results, **combo_results}

    sorted_r = sorted(all_r.items(), key=lambda x: -x[1]['ann'])
    for i, (name, r) in enumerate(sorted_r[:20]):
        print(f"  #{i+1}: {name:60s} | Ann={r['ann']:+8.1f}% | WR={r['wr']:5.1f}% | "
              f"N={r['n']:4d} | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    sorted_sh = sorted(all_r.items(), key=lambda x: -x[1]['sharpe'])
    print(f"\n  TOP 10 BY SHARPE:")
    for i, (name, r) in enumerate(sorted_sh[:10]):
        print(f"  #{i+1}: {name:60s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    sorted_wf = []
    for name, func, hold, topn in all_configs:
        w = wf(func, hold=hold, topn=topn)
        pos = sum(1 for v in w.values() if v > 0)
        avg = np.mean(list(w.values())) if w else 0
        sorted_wf.append((name, pos, avg))
    sorted_wf.sort(key=lambda x: (-x[1], -x[2]))
    print(f"\n  TOP 10 BY WALK-FORWARD CONSISTENCY:")
    for i, (name, pos, avg) in enumerate(sorted_wf[:10]):
        print(f"  #{i+1}: {name:60s} | WF_Pos={pos}/6 | WF_Avg={avg:>+7.0f}%")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
