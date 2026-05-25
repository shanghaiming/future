"""
Alpha Futures V137 — REGIME-ADAPTIVE CONCENTRATION + MOMENTUM BURST
=============================================================================
Part 1: Market Regime Detection
  - BREADTH = % of commodities with ROC(5) > 0
  - MARKET_MOM = average ROC(5) across all commodities
  - MARKET_VOL = average ATR(14)/C * 100 across all commodities
  A) Breadth-Gated V121: only trade when BREADTH > threshold
  B) Market Volatility Regime: adaptive thresholds based on vol percentile
  C) Concentration in Strong Markets: dynamic top_n based on breadth

Part 2: Momentum Burst Detection
  D) Consecutive Up Days + Volume burst
  E) Accelerating Returns
  F) Breakout from Consolidation
  G) Combined: Regime + Burst (triple confirmation)
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
    print("  V137 — REGIME-ADAPTIVE CONCENTRATION + MOMENTUM BURST")
    print("=" * 120)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    print(f"  {NS} commodities, {ND} days")

    print("\n[Precompute]...", flush=True)
    t0 = time.time()

    # --- Per-commodity indicators ---
    RET = np.full((NS, ND), np.nan)
    ROC5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100
        ROC5[si] = talib.ROC(c, timeperiod=5)

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

    # --- Market-wide indicators (computed daily across all commodities) ---
    BREADTH = np.full(ND, np.nan)       # % of commodities with ROC(5) > 0
    MARKET_MOM = np.full(ND, np.nan)    # average ROC(5) across all commodities
    MARKET_VOL = np.full(ND, np.nan)    # average ATR(14)/C * 100 across all commodities
    ADVANCING = np.full(ND, np.nan)     # count of commodities up today

    for di in range(ND):
        roc_vals = []
        vol_vals = []
        adv_count = 0
        for si in range(NS):
            roc = ROC5[si, di]
            if not np.isnan(roc):
                roc_vals.append(roc)
            ret = RET[si, di]
            if not np.isnan(ret) and ret > 0:
                adv_count += 1
            atr = ATR14[si, di]
            cp = C[si, di]
            if not np.isnan(atr) and not np.isnan(cp) and cp > 0:
                vol_vals.append(atr / cp * 100)
        if roc_vals:
            BREADTH[di] = sum(1 for r in roc_vals if r > 0) / len(roc_vals) * 100
            MARKET_MOM[di] = np.mean(roc_vals)
        if vol_vals:
            MARKET_VOL[di] = np.mean(vol_vals)
        ADVANCING[di] = adv_count

    # Percentile thresholds for MARKET_VOL (computed over full history)
    valid_vol = MARKET_VOL[~np.isnan(MARKET_VOL)]
    vol_p50 = np.percentile(valid_vol, 50) if len(valid_vol) > 0 else 0
    vol_p75 = np.percentile(valid_vol, 75) if len(valid_vol) > 0 else 0

    # --- Per-commodity: consecutive bullish days and avg volume ---
    CONSEC_UP = np.zeros((NS, ND), dtype=int)
    AVG_VOL10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            o = O[si, di]; c = C[si, di]
            if not np.isnan(o) and not np.isnan(c) and c > o:
                CONSEC_UP[si, di] = CONSEC_UP[si, di-1] + 1
            else:
                CONSEC_UP[si, di] = 0
        # 10-day average volume
        for di in range(10, ND):
            vols = V[si, di-9:di+1]
            valid = vols[~np.isnan(vols)]
            if len(valid) >= 5:
                AVG_VOL10[si, di] = np.mean(valid)

    print(f"  Done ({time.time()-t0:.1f}s)")
    print(f"  MARKET_VOL p50={vol_p50:.2f}, p75={vol_p75:.2f}")

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

    # ============ SIGNALS ============

    # --- Baseline V121 (for comparison) ---
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

    # --- A) Breadth-Gated V121: only trade when BREADTH > threshold ---
    def make_sig_breadth_gated(threshold=60):
        def sig(di, edi):
            br = BREADTH[di]
            if np.isnan(br) or br < threshold: return []
            return sig_v121(di, edi)
        return sig

    # --- B) Market Volatility Regime: adaptive thresholds ---
    def sig_vol_regime(di, edi):
        mv = MARKET_VOL[di]
        c = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs): continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(mv):
                if mv > vol_p75:
                    # High-vol regime: stricter thresholds
                    if roc <= 1.5 or zs <= 2.0: continue
                elif mv < vol_p50:
                    # Low-vol regime: standard thresholds
                    if roc <= 1.0 or zs <= 1.5: continue
                else:
                    # Medium-vol regime: standard thresholds
                    if roc <= 1.0 or zs <= 1.5: continue
            else:
                if roc <= 1.0 or zs <= 1.5: continue
            if not np.isnan(rp) and roc <= rp: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            regime = 'hivol' if (not np.isnan(mv) and mv > vol_p75) else ('lovol' if (not np.isnan(mv) and mv < vol_p50) else 'medvol')
            c.append((roc * zs, s, ep, f'volregime_{regime}'))
        return c

    # --- C) Concentration in Strong Markets ---
    def sig_concentration(di, edi):
        br = BREADTH[di]
        if np.isnan(br): return []
        if br > 70:
            tn = 1   # concentrated
        elif br >= 50:
            tn = 2   # diversified
        else:
            return []  # sit out
        # Return V121 signals, but caller should use dynamic top_n
        return sig_v121(di, edi)

    # Special backtest wrapper for concentration (dynamic top_n)
    def backtest_concentration(hold=1, start_di=MIN_TRAIN, end_di=None, desc=""):
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

            # Dynamic top_n based on breadth
            br = BREADTH[di]
            if np.isnan(br):
                continue
            if br > 70:
                top_n = 1
            elif br >= 50:
                top_n = 2
            else:
                continue  # sit out

            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue

            # V121 signals
            cands = sig_v121(di, edi)
            if not cands: continue
            cands.sort(key=lambda x: -x[0])
            ns = top_n - len(positions)
            cap = cash / max(1, ns)
            for item in cands[:ns]:
                sc, s, pr = item[:3]; sig = item[3] if len(item) > 3 else 'v121'
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

    # --- D) Consecutive Up Days + Volume Burst ---
    def sig_momentum_burst(di, edi):
        c = []
        for s in range(NS):
            consec = CONSEC_UP[s, di]
            if consec < 3: continue
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or roc <= 1.0: continue
            vol_today = V[s, di]
            avg_vol = AVG_VOL10[s, di]
            if np.isnan(vol_today) or np.isnan(avg_vol) or avg_vol <= 0: continue
            if vol_today < 1.5 * avg_vol: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            z_val = zs if not np.isnan(zs) else 1.0
            score = roc * z_val * consec
            c.append((score, s, ep, 'momentum_burst'))
        return c

    # --- E) Accelerating Returns ---
    def sig_accelerating(di, edi):
        c = []
        if di < 2: return c
        for s in range(NS):
            r0 = RET[s, di]; r1 = RET[s, di-1]; r2 = RET[s, di-2]
            if any(np.isnan(x) for x in [r0, r1, r2]): continue
            if not (r0 > r1 > r2 > 0): continue  # must be accelerating AND positive
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or roc <= 1.0: continue
            if np.isnan(zs) or zs <= 1.5: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            accel = r0 / max(r1, 0.1)
            score = roc * zs * accel
            c.append((score, s, ep, 'accelerating'))
        return c

    # --- F) Breakout from Consolidation ---
    def sig_breakout_consolidation(di, edi):
        c = []
        if di < 9: return c
        for s in range(NS):
            # 10-day range
            h10 = H[s, di-9:di+1]; l10 = L[s, di-9:di+1]
            if any(np.isnan(x) for x in h10) or any(np.isnan(x) for x in l10): continue
            range10 = np.max(h10) - np.min(l10)
            if range10 <= 0: continue
            # 3-day range
            h3 = H[s, di-2:di+1]; l3 = L[s, di-2:di+1]
            if any(np.isnan(x) for x in h3) or any(np.isnan(x) for x in l3): continue
            range3 = np.max(h3) - np.min(l3)
            # Consolidation check
            if range3 >= 0.4 * range10: continue
            # Breakout: new 10-day closing high
            cp = C[s, di]
            prev_closes = C[s, di-9:di]
            if np.isnan(cp) or any(np.isnan(x) for x in prev_closes): continue
            if cp <= np.max(prev_closes): continue
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or roc <= 0.5: continue
            if np.isnan(zs) or zs <= 1.0: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            squeeze = range10 / max(range3, 0.01)
            score = roc * zs * squeeze
            c.append((score, s, ep, 'breakout_consol'))
        return c

    # --- G) Combined: Regime + Burst (triple confirmation) ---
    def sig_regime_burst(di, edi):
        c = []
        br = BREADTH[di]
        if np.isnan(br) or br < 60: return c
        for s in range(NS):
            consec = CONSEC_UP[s, di]
            if consec < 3: continue
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue
            vol_today = V[s, di]
            avg_vol = AVG_VOL10[s, di]
            if np.isnan(vol_today) or np.isnan(avg_vol) or avg_vol <= 0: continue
            if vol_today < 1.5 * avg_vol: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = roc * zs * consec * (br / 100)
            c.append((score, s, ep, 'regime_burst'))
        return c

    # ============ SECTION 1: BASELINE + REGIME STRATEGIES ============
    print("\n" + "=" * 120)
    print("  SECTION 1: REGIME-ADAPTIVE STRATEGIES")
    print("=" * 120)

    configs = [
        ("V121 baseline (no regime)", sig_v121, 1, 1),
        ("A1) Breadth>50% gated V121", make_sig_breadth_gated(50), 1, 1),
        ("A2) Breadth>55% gated V121", make_sig_breadth_gated(55), 1, 1),
        ("A3) Breadth>60% gated V121", make_sig_breadth_gated(60), 1, 1),
        ("A4) Breadth>65% gated V121", make_sig_breadth_gated(65), 1, 1),
        ("A5) Breadth>70% gated V121", make_sig_breadth_gated(70), 1, 1),
        ("B) Vol-regime adaptive V121", sig_vol_regime, 1, 1),
        ("C) Concentration dynamic V121", None, 1, None),  # special handling
    ]

    results = {}
    for name, func, hold, topn in configs:
        if name == "C) Concentration dynamic V121":
            r = backtest_concentration(hold=hold, desc=name)
        else:
            r = backtest(func, hold=hold, top_n=topn, desc=name)
        results[name] = r
        pr(r, label=name)

    # ============ SECTION 2: MOMENTUM BURST STRATEGIES ============
    print("\n" + "=" * 120)
    print("  SECTION 2: MOMENTUM BURST STRATEGIES")
    print("=" * 120)

    burst_configs = [
        ("D) Momentum Burst (3+ up, vol>1.5x)", sig_momentum_burst, 1, 1),
        ("E) Accelerating Returns", sig_accelerating, 1, 1),
        ("F) Breakout from Consolidation", sig_breakout_consolidation, 1, 1),
        ("G) Combined Regime+Burst", sig_regime_burst, 1, 1),
    ]

    for name, func, hold, topn in burst_configs:
        r = backtest(func, hold=hold, top_n=topn, desc=name)
        results[name] = r
        pr(r, label=name)

    # ============ SECTION 3: WALK-FORWARD ============
    print("\n" + "=" * 120)
    print("  SECTION 3: WALK-FORWARD (2020-2025)")
    print("=" * 120)

    wf_configs = [
        ("V121 baseline", sig_v121, 1, 1),
        ("A1) Breadth>50%", make_sig_breadth_gated(50), 1, 1),
        ("A2) Breadth>55%", make_sig_breadth_gated(55), 1, 1),
        ("A3) Breadth>60%", make_sig_breadth_gated(60), 1, 1),
        ("A4) Breadth>65%", make_sig_breadth_gated(65), 1, 1),
        ("A5) Breadth>70%", make_sig_breadth_gated(70), 1, 1),
        ("B) Vol-regime", sig_vol_regime, 1, 1),
        ("D) Momentum Burst", sig_momentum_burst, 1, 1),
        ("E) Accelerating", sig_accelerating, 1, 1),
        ("F) Breakout Consol", sig_breakout_consolidation, 1, 1),
        ("G) Regime+Burst", sig_regime_burst, 1, 1),
    ]

    for name, func, hold, topn in wf_configs:
        w = wf(func, hold=hold, topn=topn)
        ws = " | ".join([f"{yr}:{v:+.0f}%" for yr, v in sorted(w.items())])
        pos = sum(1 for v in w.values() if v > 0)
        avg = np.mean(list(w.values())) if w else 0
        print(f"  {name:40s} | {pos}/6 | Avg={avg:>+7.0f}% | {ws}")

    # Walk-forward for concentration (special)
    print("\n  Walk-forward for Concentration (dynamic top_n):")
    conc_wf = {}
    for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
        ys = ye = None
        for di in range(ND):
            if dates[di].year == yr and ys is None: ys = di
            if dates[di].year == yr: ye = di + 1
        if ys:
            conc_wf[yr] = backtest_concentration(hold=1, start_di=ys, end_di=ye)['ann']
    ws = " | ".join([f"{yr}:{v:+.0f}%" for yr, v in sorted(conc_wf.items())])
    pos = sum(1 for v in conc_wf.values() if v > 0)
    avg = np.mean(list(conc_wf.values())) if conc_wf else 0
    print(f"  {'C) Concentration':40s} | {pos}/6 | Avg={avg:>+7.0f}% | {ws}")

    # ============ SECTION 4: TOP_N x HOLD for best configs ============
    print("\n" + "=" * 120)
    print("  SECTION 4: TOP_N x HOLD for top configs")
    print("=" * 120)

    best_funcs = [
        ("Momentum Burst", sig_momentum_burst),
        ("Breakout Consol", sig_breakout_consolidation),
        ("Regime+Burst", sig_regime_burst),
    ]
    for name, func in best_funcs:
        print(f"\n  {name}:")
        for topn in [1, 2, 3]:
            for hold in [1, 2, 3]:
                r = backtest(func, hold=hold, top_n=topn, desc=f"{name} t={topn} h={hold}")
                print(f"    top_n={topn} hold={hold}: Ann={r['ann']:+8.1f}% | "
                      f"WR={r['wr']:5.1f}% | N={r['n']:4d} | MDD={r['mdd']:6.1f}%")

    # ============ SECTION 5: BREADTH THRESHOLD SWEEP ============
    print("\n" + "=" * 120)
    print("  SECTION 5: BREADTH THRESHOLD SWEEP (detailed)")
    print("=" * 120)
    print(f"  {'Threshold':>10s} | {'Ann':>10s} | {'WR':>6s} | {'N':>5s} | {'MDD':>8s} | {'Sharpe':>7s}")
    print(f"  {'-'*10} | {'-'*10} | {'-'*6} | {'-'*5} | {'-'*8} | {'-'*7}")
    for thr in range(40, 85, 5):
        func = make_sig_breadth_gated(thr)
        r = backtest(func, hold=1, top_n=1, desc=f"breadth>{thr}")
        print(f"  {thr:>10d}% | {r['ann']:>+10.1f}% | {r['wr']:>5.1f}% | {r['n']:>5d} | {r['mdd']:>7.1f}% | {r['sharpe']:>6.2f}")

    # ============ SECTION 6: COMBINED REGIME + BURST VARIANTS ============
    print("\n" + "=" * 120)
    print("  SECTION 6: COMBINED REGIME + BURST VARIANTS")
    print("=" * 120)

    # G variant: different breadth thresholds for combined
    def make_sig_regime_burst_breadth(br_threshold=60):
        def sig(di, edi):
            c = []
            br = BREADTH[di]
            if np.isnan(br) or br < br_threshold: return c
            for s in range(NS):
                consec = CONSEC_UP[s, di]
                if consec < 3: continue
                roc = ROC5[s, di]; zs = ZSCORE[s, di]
                if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5: continue
                rp = ROC5[s, di-1] if di > 0 else np.nan
                if not np.isnan(rp) and roc <= rp: continue
                vol_today = V[s, di]
                avg_vol = AVG_VOL10[s, di]
                if np.isnan(vol_today) or np.isnan(avg_vol) or avg_vol <= 0: continue
                if vol_today < 1.5 * avg_vol: continue
                ep = O[s, edi]
                if np.isnan(ep) or ep <= 0: continue
                score = roc * zs * consec * (br / 100)
                c.append((score, s, ep, 'regime_burst'))
            return c
        return sig

    # Relax volume requirement
    def sig_regime_burst_vol1x(di, edi):
        c = []
        br = BREADTH[di]
        if np.isnan(br) or br < 60: return c
        for s in range(NS):
            consec = CONSEC_UP[s, di]
            if consec < 3: continue
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue
            vol_today = V[s, di]
            avg_vol = AVG_VOL10[s, di]
            if np.isnan(vol_today) or np.isnan(avg_vol) or avg_vol <= 0: continue
            if vol_today < 1.0 * avg_vol: continue  # relaxed from 1.5x to 1.0x
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = roc * zs * consec * (br / 100)
            c.append((score, s, ep, 'regime_burst_v1x'))
        return c

    # Relax consecutive days to 2
    def sig_regime_burst_2up(di, edi):
        c = []
        br = BREADTH[di]
        if np.isnan(br) or br < 60: return c
        for s in range(NS):
            consec = CONSEC_UP[s, di]
            if consec < 2: continue
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue
            vol_today = V[s, di]
            avg_vol = AVG_VOL10[s, di]
            if np.isnan(vol_today) or np.isnan(avg_vol) or avg_vol <= 0: continue
            if vol_today < 1.5 * avg_vol: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = roc * zs * consec * (br / 100)
            c.append((score, s, ep, 'regime_burst_2up'))
        return c

    # Accelerating + Breadth
    def sig_accelerating_breadth(di, edi):
        c = []
        if di < 2: return c
        br = BREADTH[di]
        if np.isnan(br) or br < 60: return c
        for s in range(NS):
            r0 = RET[s, di]; r1 = RET[s, di-1]; r2 = RET[s, di-2]
            if any(np.isnan(x) for x in [r0, r1, r2]): continue
            if not (r0 > r1 > r2 > 0): continue
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or roc <= 1.0: continue
            if np.isnan(zs) or zs <= 1.5: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            accel = r0 / max(r1, 0.1)
            score = roc * zs * accel * (br / 100)
            c.append((score, s, ep, 'accel_breadth'))
        return c

    # Breakout + Breadth
    def sig_breakout_breadth(di, edi):
        c = []
        if di < 9: return c
        br = BREADTH[di]
        if np.isnan(br) or br < 60: return c
        for s in range(NS):
            h10 = H[s, di-9:di+1]; l10 = L[s, di-9:di+1]
            if any(np.isnan(x) for x in h10) or any(np.isnan(x) for x in l10): continue
            range10 = np.max(h10) - np.min(l10)
            if range10 <= 0: continue
            h3 = H[s, di-2:di+1]; l3 = L[s, di-2:di+1]
            if any(np.isnan(x) for x in h3) or any(np.isnan(x) for x in l3): continue
            range3 = np.max(h3) - np.min(l3)
            if range3 >= 0.4 * range10: continue
            cp = C[s, di]
            prev_closes = C[s, di-9:di]
            if np.isnan(cp) or any(np.isnan(x) for x in prev_closes): continue
            if cp <= np.max(prev_closes): continue
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or roc <= 0.5: continue
            if np.isnan(zs) or zs <= 1.0: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            squeeze = range10 / max(range3, 0.01)
            score = roc * zs * squeeze * (br / 100)
            c.append((score, s, ep, 'breakout_breadth'))
        return c

    # Best burst signal union with breadth gate
    def sig_burst_union_breadth(di, edi):
        c = []
        br = BREADTH[di]
        if np.isnan(br) or br < 60: return c
        c.extend(sig_momentum_burst(di, edi))
        c.extend(sig_accelerating(di, edi))
        c.extend(sig_breakout_consolidation(di, edi))
        return c

    combined_configs = [
        ("G1) Regime+Burst breadth>55%", make_sig_regime_burst_breadth(55), 1, 1),
        ("G2) Regime+Burst breadth>60%", make_sig_regime_burst_breadth(60), 1, 1),
        ("G3) Regime+Burst breadth>65%", make_sig_regime_burst_breadth(65), 1, 1),
        ("G4) Regime+Burst breadth>70%", make_sig_regime_burst_breadth(70), 1, 1),
        ("G5) Regime+Burst vol>1x", sig_regime_burst_vol1x, 1, 1),
        ("G6) Regime+Burst 2+up days", sig_regime_burst_2up, 1, 1),
        ("G7) Accelerating+Breadth>60%", sig_accelerating_breadth, 1, 1),
        ("G8) Breakout+Breadth>60%", sig_breakout_breadth, 1, 1),
        ("G9) Burst Union+Breadth>60%", sig_burst_union_breadth, 1, 1),
    ]

    for name, func, hold, topn in combined_configs:
        r = backtest(func, hold=hold, top_n=topn, desc=name)
        results[name] = r
        pr(r, label=name)

    # ============ SUMMARY ============
    print("\n" + "=" * 120)
    print("  SUMMARY: ALL STRATEGIES RANKED BY ANNUAL RETURN")
    print("=" * 120)

    sorted_r = sorted(results.items(), key=lambda x: -x[1]['ann'])
    for i, (name, r) in enumerate(sorted_r):
        print(f"  #{i+1:2d}: {name:60s} | Ann={r['ann']:+8.1f}% | WR={r['wr']:5.1f}% | "
              f"N={r['n']:4d} | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    print(f"\n  TOP 10 BY SHARPE:")
    sorted_sh = sorted(results.items(), key=lambda x: -x[1]['sharpe'])
    for i, (name, r) in enumerate(sorted_sh[:10]):
        print(f"  #{i+1}: {name:60s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    print(f"\n  TOP 10 BY WIN RATE:")
    sorted_wr = sorted(results.items(), key=lambda x: -x[1]['wr'])
    for i, (name, r) in enumerate(sorted_wr[:10]):
        print(f"  #{i+1}: {name:60s} | Ann={r['ann']:+8.1f}% | WR={r['wr']:5.1f}% | N={r['n']:4d}")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
