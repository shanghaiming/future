"""
Alpha Futures V135 — MULTI-TIMEFRAME MOMENTUM + DUAL ALPHA PORTFOLIO
=============================================================================
Part 1: Multi-Timeframe Momentum Confirmation
  A) Weekly + Daily Alignment: 25-day ROC > 0 AND ROC(5) > 1% AND Z > 1.5 AND ROC improving
  B) Multi-ROC Confirmation: ROC(3,5,10,20) all > 0, rank by sum(ROCs)*Z
  C) ROC Acceleration: ROC(5) accelerating 3 days in a row
  D) Trend Quality Filter: 20-day linear regression R^2 > 0.6 + momentum

Part 2: Dual Alpha Portfolio (Independent Capital Allocation)
  E) 50/50 Split: V121 with 50% capital + OV/ID with 50% capital
  F) Rotating: 80/20 allocation based on rolling 20-day win rate
  G) Combined Equity: average of two 100%-capital equity curves
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
    print("  V135 — MULTI-TIMEFRAME MOMENTUM + DUAL ALPHA PORTFOLIO")
    print("=" * 120)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    print(f"  {NS} commodities, {ND} days")

    print("\n[Precompute]...", flush=True)
    t0 = time.time()

    # Daily returns
    RET = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100

    # ROC at multiple timeframes
    ROC3 = np.full((NS, ND), np.nan)
    ROC5 = np.full((NS, ND), np.nan)
    ROC10 = np.full((NS, ND), np.nan)
    ROC20 = np.full((NS, ND), np.nan)
    ROC25 = np.full((NS, ND), np.nan)  # ~5-week return
    for si in range(NS):
        c = C[si].astype(np.float64)
        ROC3[si] = talib.ROC(c, timeperiod=3)
        ROC5[si] = talib.ROC(c, timeperiod=5)
        ROC10[si] = talib.ROC(c, timeperiod=10)
        ROC20[si] = talib.ROC(c, timeperiod=20)
        ROC25[si] = talib.ROC(c, timeperiod=25)

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
    for si in range(NS):
        for di in range(ND):
            o, c = O[si, di], C[si, di]
            if not np.isnan(o) and not np.isnan(c):
                if di > 0 and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                    OV_GAP[si, di] = (o - C[si, di-1]) / C[si, di-1] * 100
                if o > 0:
                    ID_RET[si, di] = (c - o) / o * 100

    # 20-day linear regression slope and R^2 for Trend Quality
    LR_SLOPE = np.full((NS, ND), np.nan)
    LR_R2 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(20, ND):
            prices = c[di-19:di+1]
            if np.any(np.isnan(prices)): continue
            x = np.arange(20, dtype=np.float64)
            y = prices
            mx = np.mean(x); my = np.mean(y)
            ss_xx = np.sum((x - mx)**2)
            if ss_xx == 0: continue
            ss_xy = np.sum((x - mx) * (y - my))
            slope = ss_xy / ss_xx
            ss_yy = np.sum((y - my)**2)
            r2 = (ss_xy**2) / (ss_xx * ss_yy) if ss_yy > 0 else 0
            LR_SLOPE[si, di] = slope
            LR_R2[si, di] = r2

    print(f"  Done ({time.time()-t0:.1f}s)")

    # ===================== BACKTEST FUNCTION =====================
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

    # ===================== SIGNAL DEFINITIONS =====================

    # --- Part 1: Multi-Timeframe Momentum ---

    # A) Weekly + Daily Alignment
    def sig_weekly_daily(di, edi):
        c = []
        for s in range(NS):
            wroc = ROC25[s, di]
            roc = ROC5[s, di]
            zs = ZSCORE[s, di]
            if any(np.isnan(x) for x in [wroc, roc, zs]): continue
            if wroc <= 0 or roc <= 1.0 or zs <= 1.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((roc * zs * wroc, s, ep, 'wk+daily'))
        return c

    # B) Multi-ROC Confirmation
    def sig_multi_roc(di, edi):
        c = []
        for s in range(NS):
            r3 = ROC3[s, di]; r5 = ROC5[s, di]; r10 = ROC10[s, di]; r20 = ROC20[s, di]
            if any(np.isnan(x) for x in [r3, r5, r10, r20]): continue
            if r3 <= 0 or r5 <= 0 or r10 <= 0 or r20 <= 0: continue
            zs = ZSCORE[s, di]
            if np.isnan(zs): continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = (r3 + r5 + r10 + r20) * max(zs, 0.5)
            c.append((score, s, ep, 'multi_roc'))
        return c

    # C) ROC Acceleration
    def sig_roc_accel(di, edi):
        c = []
        for s in range(NS):
            if di < 2: continue
            r0 = ROC5[s, di]
            r1 = ROC5[s, di-1]
            r2 = ROC5[s, di-2]
            if any(np.isnan(x) for x in [r0, r1, r2]): continue
            if r0 <= 0 or r1 <= 0 or r2 <= 0: continue
            if not (r0 > r1 > r2): continue  # must be strictly accelerating
            zs = ZSCORE[s, di]
            if np.isnan(zs): zs = 1.0
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            accel = r0 - r2  # 2nd derivative proxy
            score = r0 * max(zs, 0.5) * accel
            c.append((score, s, ep, 'roc_accel'))
        return c

    # D) Trend Quality Filter
    def sig_trend_quality(di, edi):
        c = []
        for s in range(NS):
            roc = ROC5[s, di]
            r2 = LR_R2[s, di]
            slope = LR_SLOPE[s, di]
            zs = ZSCORE[s, di]
            if any(np.isnan(x) for x in [roc, r2, slope, zs]): continue
            if roc <= 1.0 or r2 < 0.6 or slope <= 0 or zs <= 1.0: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = roc * zs * r2 * (1 if slope > 0 else 0)
            c.append((score, s, ep, 'trend_q'))
        return c

    # --- V121 baseline (for Part 2 reference) ---
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

    # --- OV/ID signal (for Part 2 reference) ---
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

    # --- Combined: All multi-TF signals compete ---
    def sig_combined_mtf(di, edi):
        all_sigs = []
        for item in sig_weekly_daily(di, edi):
            all_sigs.append((item[0] * 2, item[1], item[2], item[3]))
        for item in sig_multi_roc(di, edi):
            all_sigs.append((item[0] * 1.5, item[1], item[2], item[3]))
        for item in sig_roc_accel(di, edi):
            all_sigs.append((item[0] * 1.5, item[1], item[2], item[3]))
        for item in sig_trend_quality(di, edi):
            all_sigs.append((item[0], item[1], item[2], item[3]))
        return all_sigs

    # --- Combined: Multi-TF + V121 + OV/ID all compete ---
    def sig_ultimate(di, edi):
        all_sigs = []
        for item in sig_v121(di, edi):
            all_sigs.append((item[0] * 3, item[1], item[2], item[3]))
        for item in sig_ov_id(di, edi):
            all_sigs.append((item[0] * 2, item[1], item[2], item[3]))
        for item in sig_weekly_daily(di, edi):
            all_sigs.append((item[0] * 2, item[1], item[2], item[3]))
        for item in sig_multi_roc(di, edi):
            all_sigs.append((item[0] * 1.5, item[1], item[2], item[3]))
        for item in sig_roc_accel(di, edi):
            all_sigs.append((item[0] * 1.5, item[1], item[2], item[3]))
        for item in sig_trend_quality(di, edi):
            all_sigs.append((item[0], item[1], item[2], item[3]))
        return all_sigs

    # ============ SECTION 1: MULTI-TIMEFRAME MOMENTUM SIGNALS ============
    print("\n" + "=" * 120)
    print("  SECTION 1: MULTI-TIMEFRAME MOMENTUM SIGNALS")
    print("=" * 120)

    mtf_configs = [
        ("V121 baseline (reference)", sig_v121, 1, 1),
        ("OV/ID baseline (reference)", sig_ov_id, 1, 1),
        ("A) Weekly+Daily Alignment", sig_weekly_daily, 1, 1),
        ("B) Multi-ROC Confirmation", sig_multi_roc, 1, 1),
        ("C) ROC Acceleration", sig_roc_accel, 1, 1),
        ("D) Trend Quality Filter", sig_trend_quality, 1, 1),
        ("Combined MTF (A+B+C+D)", sig_combined_mtf, 1, 1),
        ("Ultimate (MTF+V121+OV/ID)", sig_ultimate, 1, 1),
    ]

    results = {}
    for name, func, hold, topn in mtf_configs:
        r = backtest(func, hold=hold, top_n=topn, desc=name)
        results[name] = r
        pr(r, label=name)

    # ============ SECTION 2: TOP_N x HOLD FOR TOP CONFIGS ============
    print("\n" + "=" * 120)
    print("  SECTION 2: TOP_N x HOLD for top MTF configs")
    print("=" * 120)

    best3 = sorted(results.items(), key=lambda x: -x[1]['ann'])[:3]
    for name, r in best3:
        func_map = {n: f for n, f, _, _ in mtf_configs}
        func = func_map[name]
        print(f"\n  {name}:")
        for topn in [1, 2, 3]:
            for hold in [1, 2, 3]:
                r = backtest(func, hold=hold, top_n=topn, desc=f"{name} t={topn} h={hold}")
                print(f"    top_n={topn} hold={hold}: Ann={r['ann']:+8.1f}% | "
                      f"WR={r['wr']:5.1f}% | N={r['n']:4d} | MDD={r['mdd']:6.1f}%")

    # ============ SECTION 3: DUAL ALPHA PORTFOLIO ============
    print("\n" + "=" * 120)
    print("  SECTION 3: DUAL ALPHA PORTFOLIO")
    print("=" * 120)

    # --- E) 50/50 Split: Run V121 with 50% capital, OV/ID with 50% capital ---
    print("\n  --- E) 50/50 Capital Split ---")

    def backtest_equity(signal_func, hold=1, top_n=1, start_di=MIN_TRAIN, end_di=None,
                        capital_frac=1.0):
        """Backtest returning daily equity series for portfolio combination."""
        if end_di is None: end_di = ND
        cash = float(CASH0) * capital_frac
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

    # G) Combined Equity Simulation: Run each with 100% capital, average the curves
    print("\n  --- G) Combined Equity Simulation (V121@100% + OV/ID@100%, avg) ---")

    eq_v121, final_v121 = backtest_equity(sig_v121, hold=1, top_n=1, capital_frac=1.0)
    eq_ov, final_ov = backtest_equity(sig_ov_id, hold=1, top_n=1, capital_frac=1.0)

    min_len = min(len(eq_v121), len(eq_ov))
    eq_combined = (eq_v121[:min_len] + eq_ov[:min_len]) / 2.0
    final_combined = eq_combined[-1]
    n_days_combined = min_len
    ann_combined = annual_return(final_combined, CASH0, n_days_combined)
    # Compute MDD and Sharpe for combined
    pk = np.maximum.accumulate(eq_combined)
    mdd_combined = np.min((eq_combined - pk) / pk * 100)
    rets_combined = np.diff(eq_combined) / eq_combined[:-1]
    sh_combined = np.mean(rets_combined) / np.std(rets_combined) * np.sqrt(252) if np.std(rets_combined) > 0 else 0

    ann_v121 = annual_return(final_v121, CASH0, n_days_combined)
    ann_ov = annual_return(final_ov, CASH0, n_days_combined)

    print(f"  V121 standalone@100%:        Ann={ann_v121:+8.1f}% | Final={final_v121:,.0f}")
    print(f"  OV/ID standalone@100%:       Ann={ann_ov:+8.1f}% | Final={final_ov:,.0f}")
    print(f"  G) Combined (50/50 avg):     Ann={ann_combined:+8.1f}% | MDD={mdd_combined:6.1f}% | Sh={sh_combined:4.2f}")
    print(f"     Portfolio diversification benefit: combined annual = {ann_combined:.1f}% vs avg of standalone = {(ann_v121+ann_ov)/2:.1f}%")

    # E) 50/50 Split: Each strategy gets 50% capital, runs independently
    print("\n  --- E) 50/50 Independent Capital Split ---")

    eq_v121_50, final_v121_50 = backtest_equity(sig_v121, hold=1, top_n=1, capital_frac=0.5)
    eq_ov_50, final_ov_50 = backtest_equity(sig_ov_id, hold=1, top_n=1, capital_frac=0.5)
    final_split = final_v121_50 + final_ov_50
    ann_split = annual_return(final_split, CASH0, n_days_combined)
    eq_split = eq_v121_50[:min(len(eq_v121_50), len(eq_ov_50))] + eq_ov_50[:min(len(eq_v121_50), len(eq_ov_50))]
    pk_s = np.maximum.accumulate(eq_split)
    mdd_split = np.min((eq_split - pk_s) / pk_s * 100) if len(eq_split) > 1 else 0
    r_s = np.diff(eq_split) / eq_split[:-1] if len(eq_split) > 1 else np.array([0])
    sh_split = np.mean(r_s) / np.std(r_s) * np.sqrt(252) if np.std(r_s) > 0 else 0

    print(f"  V121@50%:       Final={final_v121_50:,.0f}")
    print(f"  OV/ID@50%:      Final={final_ov_50:,.0f}")
    print(f"  E) 50/50 Split: Ann={ann_split:+8.1f}% | MDD={mdd_split:6.1f}% | Sh={sh_split:4.2f} | Final={final_split:,.0f}")

    # F) Rotating Portfolio: 80/20 based on rolling 20-day win rate
    print("\n  --- F) Rotating Portfolio (80/20 by rolling 20-day WR) ---")

    def backtest_rotating(start_di=MIN_TRAIN, end_di=None):
        if end_di is None: end_di = ND
        # Pre-generate signals for both strategies
        v121_trades_by_day = {}  # di -> list of pnl_pct for trades closed that day
        ov_trades_by_day = {}
        # We need to track trade outcomes for rolling WR
        # Simpler: pre-run both and collect daily P&L

        # Run V121 and collect daily equity changes
        eq_v, _ = backtest_equity(sig_v121, hold=1, top_n=1, start_di=start_di, end_di=end_di, capital_frac=1.0)
        eq_o, _ = backtest_equity(sig_ov_id, hold=1, top_n=1, start_di=start_di, end_di=end_di, capital_frac=1.0)

        # Compute daily returns for each
        ml = min(len(eq_v), len(eq_o)) - 1
        ret_v = np.diff(eq_v[:ml+1]) / eq_v[:ml]
        ret_o = np.diff(eq_o[:ml+1]) / eq_o[:ml]

        # Rolling 20-day win rate for each
        win_v = (ret_v > 0).astype(float)
        win_o = (ret_o > 0).astype(float)

        # Rotating allocation
        combined_ret = np.zeros(ml)
        for i in range(ml):
            if i < 20:
                # Not enough history, use 50/50
                w_v = 0.5; w_o = 0.5
            else:
                wr_v = np.mean(win_v[i-20:i])
                wr_o = np.mean(win_o[i-20:i])
                if wr_v >= wr_o:
                    w_v = 0.8; w_o = 0.2
                else:
                    w_v = 0.2; w_o = 0.8
            combined_ret[i] = w_v * ret_v[i] + w_o * ret_o[i]

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

    rot = backtest_rotating()
    print(f"  F) Rotating 80/20: Ann={rot['ann']:+8.1f}% | MDD={rot['mdd']:6.1f}% | Sh={rot['sharpe']:4.2f} | Final={rot['final']:,.0f}")

    # Also do rotating with best MTF signal instead of V121
    print("\n  --- F2) Rotating: MTF vs OV/ID ---")

    def backtest_rotating_mtf(start_di=MIN_TRAIN, end_di=None):
        if end_di is None: end_di = ND
        eq_m, _ = backtest_equity(sig_combined_mtf, hold=1, top_n=1, start_di=start_di, end_di=end_di, capital_frac=1.0)
        eq_o, _ = backtest_equity(sig_ov_id, hold=1, top_n=1, start_di=start_di, end_di=end_di, capital_frac=1.0)
        ml = min(len(eq_m), len(eq_o)) - 1
        ret_m = np.diff(eq_m[:ml+1]) / eq_m[:ml]
        ret_o = np.diff(eq_o[:ml+1]) / eq_o[:ml]
        win_m = (ret_m > 0).astype(float)
        win_o = (ret_o > 0).astype(float)
        combined_ret = np.zeros(ml)
        for i in range(ml):
            if i < 20:
                w_m = 0.5; w_o = 0.5
            else:
                wr_m = np.mean(win_m[i-20:i])
                wr_o = np.mean(win_o[i-20:i])
                if wr_m >= wr_o:
                    w_m = 0.8; w_o = 0.2
                else:
                    w_m = 0.2; w_o = 0.8
            combined_ret[i] = w_m * ret_m[i] + w_o * ret_o[i]
        eq_rot = np.zeros(ml + 1)
        eq_rot[0] = float(CASH0)
        for i in range(ml):
            eq_rot[i+1] = eq_rot[i] * (1 + combined_ret[i])
        final = eq_rot[-1]
        nd = ml
        ann = annual_return(final, CASH0, nd)
        pk = np.maximum.accumulate(eq_rot)
        mdd = np.min((eq_rot - pk) / pk * 100)
        sh = np.mean(combined_ret) / np.std(combined_ret) * np.sqrt(252) if np.std(combined_ret) > 0 else 0
        return {'ann': ann, 'mdd': mdd, 'sharpe': sh, 'final': final}

    rot_mtf = backtest_rotating_mtf()
    print(f"  F2) Rotating MTF/OV: Ann={rot_mtf['ann']:+8.1f}% | MDD={rot_mtf['mdd']:6.1f}% | Sh={rot_mtf['sharpe']:4.2f}")

    # ============ SECTION 4: WALK-FORWARD ============
    print("\n" + "=" * 120)
    print("  SECTION 4: WALK-FORWARD")
    print("=" * 120)

    wf_configs = [
        ("V121 baseline", sig_v121, 1, 1),
        ("OV/ID baseline", sig_ov_id, 1, 1),
        ("A) Weekly+Daily", sig_weekly_daily, 1, 1),
        ("B) Multi-ROC", sig_multi_roc, 1, 1),
        ("C) ROC Acceleration", sig_roc_accel, 1, 1),
        ("D) Trend Quality", sig_trend_quality, 1, 1),
        ("Combined MTF", sig_combined_mtf, 1, 1),
        ("Ultimate", sig_ultimate, 1, 1),
    ]

    for name, func, hold, topn in wf_configs:
        w = wf(func, hold=hold, topn=topn)
        ws = " | ".join([f"{yr}:{v:+.0f}%" for yr, v in sorted(w.items())])
        pos = sum(1 for v in w.values() if v > 0)
        avg = np.mean(list(w.values())) if w else 0
        print(f"  {name:60s} | {pos}/6 | Avg={avg:>+7.0f}% | {ws}")

    # Walk-forward for portfolio strategies (special handling)
    print("\n  --- Walk-Forward for Portfolio Strategies ---")
    for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
        ys = ye = None
        for di in range(ND):
            if dates[di].year == yr and ys is None: ys = di
            if dates[di].year == yr: ye = di + 1
        if ys is None: continue
        # G) Combined equity
        eq_v_y, fv_y = backtest_equity(sig_v121, hold=1, top_n=1, start_di=ys, end_di=ye, capital_frac=1.0)
        eq_o_y, fo_y = backtest_equity(sig_ov_id, hold=1, top_n=1, start_di=ys, end_di=ye, capital_frac=1.0)
        ml_y = min(len(eq_v_y), len(eq_o_y))
        eq_c_y = (eq_v_y[:ml_y] + eq_o_y[:ml_y]) / 2.0
        nd_y = ye - ys
        ann_g = annual_return(eq_c_y[-1], CASH0, nd_y)
        # E) 50/50 split
        eq_v50, fv50 = backtest_equity(sig_v121, hold=1, top_n=1, start_di=ys, end_di=ye, capital_frac=0.5)
        eq_o50, fo50 = backtest_equity(sig_ov_id, hold=1, top_n=1, start_di=ys, end_di=ye, capital_frac=0.5)
        ann_e = annual_return(fv50 + fo50, CASH0, nd_y)
        # V121 and OV standalone
        ann_v_y = annual_return(fv_y, CASH0, nd_y)
        ann_o_y = annual_return(fo_y, CASH0, nd_y)
        print(f"  {yr}: V121={ann_v_y:+.0f}% | OV={ann_o_y:+.0f}% | E)50/50={ann_e:+.0f}% | G)avg={ann_g:+.0f}%")

    # ============ SECTION 5: DIVERSIFIED top_n x hold ============
    print("\n" + "=" * 120)
    print("  SECTION 5: DIVERSIFIED top_n=2-3 COMBINATIONS")
    print("=" * 120)

    div_configs = [
        ("Weekly+Daily t=2", sig_weekly_daily, 1, 2),
        ("Multi-ROC t=2", sig_multi_roc, 1, 2),
        ("ROC Accel t=2", sig_roc_accel, 1, 2),
        ("Trend Quality t=2", sig_trend_quality, 1, 2),
        ("Combined MTF t=2", sig_combined_mtf, 1, 2),
        ("Ultimate t=2", sig_ultimate, 1, 2),
        ("Ultimate t=3", sig_ultimate, 1, 3),
        ("Combined MTF t=2 h=2", sig_combined_mtf, 2, 2),
        ("Ultimate t=2 h=2", sig_ultimate, 2, 2),
    ]

    div_results = {}
    for name, func, hold, topn in div_configs:
        r = backtest(func, hold=hold, top_n=topn, desc=name)
        div_results[name] = r
        pr(r, label=name)

    # ============ SECTION 6: SIGNAL BREAKDOWN ============
    print("\n" + "=" * 120)
    print("  SECTION 6: SIGNAL TYPE BREAKDOWN")
    print("=" * 120)

    for name in ["Combined MTF (A+B+C+D)", "Ultimate (MTF+V121+OV/ID)"]:
        rname = name.split('(')[0].strip()
        for k in results:
            if k.startswith(rname):
                bd = results[k].get('sig_breakdown', {})
                print(f"\n  {k}:")
                for sig, data in sorted(bd.items(), key=lambda x: -x[1]['n']):
                    wr = data['w'] / data['n'] * 100 if data['n'] > 0 else 0
                    ap = data['pnl'] / data['n'] if data['n'] > 0 else 0
                    print(f"    {sig:20s}: N={data['n']:4d} | WR={wr:5.1f}% | AvgPnL={ap:+.2f}%")
                break

    # ============ SUMMARY ============
    print("\n" + "=" * 120)
    print("  SUMMARY: TOP 15 BY ANNUAL RETURN")
    print("=" * 120)

    all_r = {**results, **div_results}
    # Add portfolio results
    all_r['G) Combined Equity'] = {'ann': ann_combined, 'wr': 0, 'n': 0, 'avg_pnl': 0,
                                    'mdd': mdd_combined, 'sharpe': sh_combined, 'final': final_combined}
    all_r['E) 50/50 Split'] = {'ann': ann_split, 'wr': 0, 'n': 0, 'avg_pnl': 0,
                                'mdd': mdd_split, 'sharpe': sh_split, 'final': final_split}
    all_r['F) Rotating 80/20'] = {'ann': rot['ann'], 'wr': 0, 'n': 0, 'avg_pnl': 0,
                                   'mdd': rot['mdd'], 'sharpe': rot['sharpe'], 'final': rot['final']}
    all_r['F2) Rotating MTF/OV'] = {'ann': rot_mtf['ann'], 'wr': 0, 'n': 0, 'avg_pnl': 0,
                                     'mdd': rot_mtf['mdd'], 'sharpe': rot_mtf['sharpe'], 'final': rot_mtf['final']}

    sorted_r = sorted(all_r.items(), key=lambda x: -x[1]['ann'])
    for i, (name, r) in enumerate(sorted_r[:15]):
        n_str = f"N={r['n']:4d}" if r['n'] > 0 else "N=port"
        wr_str = f"WR={r['wr']:5.1f}%" if r['n'] > 0 else "WR=port "
        print(f"  #{i+1}: {name:60s} | Ann={r['ann']:+8.1f}% | {wr_str} | "
              f"{n_str} | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    sorted_sh = sorted(all_r.items(), key=lambda x: -x[1]['sharpe'])
    print(f"\n  TOP 10 BY SHARPE:")
    for i, (name, r) in enumerate(sorted_sh[:10]):
        print(f"  #{i+1}: {name:60s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
