"""
Alpha Futures V140 — DRAWDOWN-CONTROLLED PORTFOLIO (v2)
=============================================================================
Problem: V139 Union/V121 has -94.7% MDD. Previous V140 attempt showed that
scaling returns AFTER the fact doesn't help — need to control WITHIN the backtest.

New approach:
  A) Hard circuit breaker: STOP ALL TRADING when drawdown > threshold, resume when equity > MA
  B) Losing streak filter: skip signals after N consecutive losses
  C) Win rate gate: stop trading if recent 20-trade WR < threshold
  D) Half-Kelly position sizing based on rolling WR and avg win/loss
  E) Profit target: close early if position profit > X%
  F) Equity curve trading: only trade when equity > N-day MA

Combined with portfolio: Union/V121 50/50 where each sub-strategy gets independent control.
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
    print("  V140 — DRAWDOWN-CONTROLLED PORTFOLIO (v2: Hard Circuit Breakers)")
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

    # ===================== BACKTEST WITH HARD CONTROLS =====================
    def backtest_dd(signal_func, hold=1, top_n=1, start_di=MIN_TRAIN, end_di=None,
                     dd_stop=0.30,        # Stop ALL trading when drawdown > this (fraction)
                     dd_resume_ma=20,      # Resume trading when equity > MA(equity, N) after stop
                     lose_streak=0,        # Skip signals after N consecutive losses (0=off)
                     wr_gate=0.0,          # Stop trading if recent 20-trade WR < this (0=off)
                     kelly_half=False,     # Use half-Kelly position sizing
                     max_loss_pct=0.0,     # Close position intraday if loss > this % (0=off)
                     eq_ma_filter=0,       # Only trade when equity > MA(equity, N) (0=off)
                     size_frac=0.95):      # Position size as fraction of capital
        if end_di is None: end_di = ND
        cash = float(CASH0)
        positions = []; trades = []; daily_eq = []
        high_water = float(CASH0)
        trading_paused = False
        consecutive_losses = 0
        recent_trades = []  # Rolling window for WR calculation

        for di in range(start_di, end_di - 1):
            pv = cash
            for p in positions:
                cp = C[p['si'], di]
                if not np.isnan(cp) and cp > 0:
                    m = MULT.get(p['sym'], DEF_MULT)
                    pv += cp * m * p['lots'] - cp * m * abs(p['lots']) * COMM
            daily_eq.append(pv)

            if pv > high_water:
                high_water = pv

            # Check max_loss_pct for open positions (close early)
            if max_loss_pct > 0:
                cl_early = []
                for p in positions:
                    cp = C[p['si'], di]
                    if np.isnan(cp) or cp <= 0: continue
                    m = MULT.get(p['sym'], DEF_MULT)
                    unrealized = (cp - p['entry_price']) * m * p['lots']
                    invested = p['entry_price'] * m * abs(p['lots'])
                    loss_pct = unrealized / invested if invested > 0 else 0
                    if loss_pct < -max_loss_pct:
                        cash += cp * m * abs(p['lots']) * (1 - COMM)
                        pp = unrealized / invested * 100 if invested > 0 else 0
                        trades.append({'pnl_pct': pp, 'sig': p.get('sig', '')})
                        recent_trades.append(pp > 0)
                        consecutive_losses = 0 if pp > 0 else consecutive_losses + 1
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
                    trades.append({'pnl_pct': pp, 'sig': p.get('sig', '')})
                    recent_trades.append(pp > 0)
                    if len(recent_trades) > 50: recent_trades = recent_trades[-50:]
                    consecutive_losses = 0 if pp > 0 else consecutive_losses + 1
                    cl.append(p)
            for p in cl: positions.remove(p)

            # --- Drawdown circuit breaker ---
            cur_dd = (pv - high_water) / high_water if high_water > 0 else 0
            if dd_stop > 0 and cur_dd < -dd_stop:
                trading_paused = True

            # Resume condition: equity > MA of equity
            if trading_paused and dd_resume_ma > 0 and len(daily_eq) >= dd_resume_ma:
                eq_recent = daily_eq[-dd_resume_ma:]
                if pv >= np.mean(eq_recent):
                    trading_paused = False

            if trading_paused:
                continue

            # --- Equity curve MA filter ---
            if eq_ma_filter > 0 and len(daily_eq) >= eq_ma_filter:
                eq_ma = np.mean(daily_eq[-eq_ma_filter:])
                if pv < eq_ma:
                    continue

            # --- Losing streak filter ---
            if lose_streak > 0 and consecutive_losses >= lose_streak:
                continue

            # --- Win rate gate ---
            if wr_gate > 0 and len(recent_trades) >= 20:
                recent_wr = np.mean(recent_trades[-20:])
                if recent_wr < wr_gate:
                    continue

            # --- Position sizing ---
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue
            cands = signal_func(di, edi)
            if not cands: continue
            cands.sort(key=lambda x: -x[0])
            ns = top_n - len(positions)
            cap = cash * size_frac / max(1, ns)

            # Half-Kelly sizing
            if kelly_half and len(trades) >= 30:
                recent = trades[-100:]
                wins = [t['pnl_pct'] for t in recent if t['pnl_pct'] > 0]
                losses = [-t['pnl_pct'] for t in recent if t['pnl_pct'] <= 0]
                if wins and losses:
                    wr_k = len(wins) / len(recent)
                    avg_w = np.mean(wins)
                    avg_l = np.mean(losses)
                    if avg_l > 0:
                        kelly = wr_k - (1 - wr_k) / (avg_w / avg_l)
                        kelly = max(0.25, min(1.0, kelly))
                        cap *= kelly * 0.5  # Half Kelly

            for item in cands[:ns]:
                if len(item) == 3: sc, s, pr = item; sig = ''
                else: sc, s, pr, sig = item
                sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                ct = max(1, int(cap / (pr * m * (1 + COMM))))
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
            r = np.where(np.isfinite(r), r, 0)
            sh = np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0
        else: mdd = 0; sh = 0
        return {'ann': ann, 'wr': wr, 'n': nt, 'avg_pnl': ap, 'mdd': mdd, 'sharpe': sh,
                'final': cash}

    def pr(r, label=""):
        print(f"  {label:70s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | N={r['n']:4d}")

    def wf_params(signal_func, hold=1, topn=1, **kwargs):
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys:
                r = backtest_dd(signal_func, hold=hold, top_n=topn, start_di=ys, end_di=ye, **kwargs)
                res[yr] = r['ann']
        return res

    # ===================== SECTION 1: BASELINES =====================
    print("\n" + "=" * 120)
    print("  SECTION 1: BASELINES (NO DRAWDOWN CONTROL)")
    print("=" * 120)

    r_v121 = backtest_dd(sig_v121, hold=1, top_n=1, dd_stop=0)
    pr(r_v121, "V121 baseline (no control)")

    r_union = backtest_dd(sig_union, hold=1, top_n=1, dd_stop=0)
    pr(r_union, "Union baseline (no control)")

    # ===================== SECTION 2: DRAWDOWN CIRCUIT BREAKER =====================
    print("\n" + "=" * 120)
    print("  SECTION 2: DRAWDOWN CIRCUIT BREAKER (stop trading when DD > threshold)")
    print("=" * 120)

    dd_configs = [
        # (dd_stop, dd_resume_ma, signal, label)
        (0.15, 20, sig_v121, "V121 DD_stop=15% resume=MA20"),
        (0.20, 20, sig_v121, "V121 DD_stop=20% resume=MA20"),
        (0.25, 20, sig_v121, "V121 DD_stop=25% resume=MA20"),
        (0.30, 20, sig_v121, "V121 DD_stop=30% resume=MA20"),
        (0.15, 20, sig_union, "Union DD_stop=15% resume=MA20"),
        (0.20, 20, sig_union, "Union DD_stop=20% resume=MA20"),
        (0.25, 20, sig_union, "Union DD_stop=25% resume=MA20"),
        (0.30, 20, sig_union, "Union DD_stop=30% resume=MA20"),
        (0.20, 30, sig_union, "Union DD_stop=20% resume=MA30"),
        (0.25, 30, sig_union, "Union DD_stop=25% resume=MA30"),
        (0.30, 30, sig_union, "Union DD_stop=30% resume=MA30"),
    ]

    dd_results = []
    for dd_stop, dd_resume, sig, label in dd_configs:
        r = backtest_dd(sig, hold=1, top_n=1, dd_stop=dd_stop, dd_resume_ma=dd_resume)
        r['desc'] = label
        dd_results.append(r)
        pr(r, label)

    # ===================== SECTION 3: LOSING STREAK + WR GATE =====================
    print("\n" + "=" * 120)
    print("  SECTION 3: LOSING STREAK + WIN RATE GATE")
    print("=" * 120)

    streak_configs = [
        (3, 0.0, sig_v121, "V121 lose_streak=3"),
        (4, 0.0, sig_v121, "V121 lose_streak=4"),
        (5, 0.0, sig_v121, "V121 lose_streak=5"),
        (0, 0.40, sig_v121, "V121 wr_gate=40%"),
        (0, 0.45, sig_v121, "V121 wr_gate=45%"),
        (3, 0.0, sig_union, "Union lose_streak=3"),
        (4, 0.0, sig_union, "Union lose_streak=4"),
        (5, 0.0, sig_union, "Union lose_streak=5"),
        (0, 0.40, sig_union, "Union wr_gate=40%"),
        (0, 0.45, sig_union, "Union wr_gate=45%"),
    ]

    streak_results = []
    for lose_s, wr_g, sig, label in streak_configs:
        r = backtest_dd(sig, hold=1, top_n=1, dd_stop=0, lose_streak=lose_s, wr_gate=wr_g)
        r['desc'] = label
        streak_results.append(r)
        pr(r, label)

    # ===================== SECTION 4: EQUITY MA FILTER =====================
    print("\n" + "=" * 120)
    print("  SECTION 4: EQUITY CURVE MA FILTER (only trade when equity > N-day MA)")
    print("=" * 120)

    eqma_configs = [
        (20, sig_v121, "V121 EQMA=20"),
        (30, sig_v121, "V121 EQMA=30"),
        (40, sig_v121, "V121 EQMA=40"),
        (60, sig_v121, "V121 EQMA=60"),
        (20, sig_union, "Union EQMA=20"),
        (30, sig_union, "Union EQMA=30"),
        (40, sig_union, "Union EQMA=40"),
        (60, sig_union, "Union EQMA=60"),
    ]

    eqma_results = []
    for eq_ma, sig, label in eqma_configs:
        r = backtest_dd(sig, hold=1, top_n=1, dd_stop=0, eq_ma_filter=eq_ma)
        r['desc'] = label
        eqma_results.append(r)
        pr(r, label)

    # ===================== SECTION 5: KELLY + COMBINED =====================
    print("\n" + "=" * 120)
    print("  SECTION 5: COMBINED CONTROLS (best individual + layered)")
    print("=" * 120)

    combined_configs = [
        # Best from above combined
        (sig_union, 0.20, 20, 0, 0.0, False, 20, "Union DD20% + EQMA20"),
        (sig_union, 0.25, 20, 0, 0.0, False, 20, "Union DD25% + EQMA20"),
        (sig_union, 0.20, 20, 4, 0.0, False, 20, "Union DD20% + lose4 + EQMA20"),
        (sig_union, 0.25, 20, 4, 0.0, False, 20, "Union DD25% + lose4 + EQMA20"),
        (sig_union, 0.20, 20, 0, 0.0, False, 30, "Union DD20% + EQMA30"),
        (sig_union, 0.25, 20, 0, 0.0, False, 30, "Union DD25% + EQMA30"),
        (sig_union, 0.20, 20, 4, 0.0, False, 30, "Union DD20% + lose4 + EQMA30"),
        (sig_union, 0.25, 20, 4, 0.0, False, 30, "Union DD25% + lose4 + EQMA30"),
        (sig_union, 0.30, 20, 4, 0.0, False, 20, "Union DD30% + lose4 + EQMA20"),
        (sig_union, 0.30, 20, 4, 0.0, False, 30, "Union DD30% + lose4 + EQMA30"),
        (sig_union, 0.25, 20, 0, 0.0, True, 20, "Union DD25% + Kelly + EQMA20"),
        (sig_union, 0.25, 20, 4, 0.0, True, 20, "Union DD25% + lose4 + Kelly + EQMA20"),
        (sig_union, 0.25, 20, 0, 0.40, False, 20, "Union DD25% + wr40% + EQMA20"),
        (sig_union, 0.25, 20, 4, 0.40, False, 20, "Union DD25% + lose4 + wr40% + EQMA20"),
        # V121 combined
        (sig_v121, 0.20, 20, 0, 0.0, False, 20, "V121 DD20% + EQMA20"),
        (sig_v121, 0.25, 20, 4, 0.0, False, 20, "V121 DD25% + lose4 + EQMA20"),
        (sig_v121, 0.25, 20, 0, 0.0, True, 20, "V121 DD25% + Kelly + EQMA20"),
    ]

    combined_results = []
    for sig, dd_stop, dd_resume, lose_s, wr_g, kelly, eq_ma, label in combined_configs:
        r = backtest_dd(sig, hold=1, top_n=1, dd_stop=dd_stop, dd_resume_ma=dd_resume,
                         lose_streak=lose_s, wr_gate=wr_g, kelly_half=kelly, eq_ma_filter=eq_ma)
        r['desc'] = label
        r['params'] = (sig, dd_stop, dd_resume, lose_s, wr_g, kelly, eq_ma)
        combined_results.append(r)
        pr(r, label)

    # ===================== SECTION 6: PORTFOLIO (Union + V121 50/50) WITH DD CONTROL =====================
    print("\n" + "=" * 120)
    print("  SECTION 6: UNION/V121 50/50 PORTFOLIO WITH INDEPENDENT DD CONTROL")
    print("=" * 120)

    def backtest_portfolio_dd(sig_A, sig_B, start_di=MIN_TRAIN, end_di=None,
                                dd_stop=0.25, dd_resume_ma=20, eq_ma_filter=20,
                                lose_streak=4):
        """Each sub-strategy gets independent drawdown control, then 50/50 combined."""
        if end_di is None: end_di = ND

        def run_sub(sig_func):
            cash = float(CASH0); positions = []; daily_eq = []
            high_water = float(CASH0)
            trading_paused = False
            consecutive_losses = 0

            for di in range(start_di, end_di - 1):
                pv = cash
                for p in positions:
                    cp = C[p['si'], di]
                    if not np.isnan(cp) and cp > 0:
                        m = MULT.get(p['sym'], DEF_MULT)
                        pv += cp * m * p['lots'] - cp * m * abs(p['lots']) * COMM
                daily_eq.append(pv)
                if pv > high_water: high_water = pv

                # Close positions
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
                        consecutive_losses = 0 if pp > 0 else consecutive_losses + 1
                        cl.append(p)
                for p in cl: positions.remove(p)

                # DD circuit breaker
                cur_dd = (pv - high_water) / high_water if high_water > 0 else 0
                if dd_stop > 0 and cur_dd < -dd_stop:
                    trading_paused = True
                if trading_paused and dd_resume_ma > 0 and len(daily_eq) >= dd_resume_ma:
                    if pv >= np.mean(daily_eq[-dd_resume_ma:]):
                        trading_paused = False

                if trading_paused: continue
                if eq_ma_filter > 0 and len(daily_eq) >= eq_ma_filter:
                    if pv < np.mean(daily_eq[-eq_ma_filter:]): continue
                if lose_streak > 0 and consecutive_losses >= lose_streak: continue

                if len(positions) >= 1: continue
                edi = di + 1
                if edi >= end_di: continue
                cands = sig_func(di, edi)
                if not cands: continue
                cands.sort(key=lambda x: -x[0])
                item = cands[0]
                if len(item) == 3: sc, s, pr = item; sig = ''
                else: sc, s, pr, sig = item
                sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                cap = cash * 0.95
                ct = max(1, int(cap / (pr * m * (1 + COMM))))
                ci = pr * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                    ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                if ct <= 0 or ci <= 0 or ci > cash: continue
                cash -= ci
                positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                  'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': 1, 'sig': sig})

            for p in positions:
                ep = C[p['si'], min(end_di-1, ND-1)]
                if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                m = MULT.get(p['sym'], DEF_MULT)
                cash += ep * m * abs(p['lots']) * (1 - COMM)
            return np.array(daily_eq)

        eq_A = run_sub(sig_A)
        eq_B = run_sub(sig_B)

        ml = min(len(eq_A), len(eq_B))
        if ml <= 1:
            return {'ann': -100.0, 'mdd': 0, 'sharpe': 0, 'final': CASH0}

        # 50/50 combine
        ret_A = np.diff(eq_A[:ml]) / eq_A[:ml-1]
        ret_B = np.diff(eq_B[:ml]) / eq_B[:ml-1]
        ret_A = np.where(np.isfinite(ret_A), ret_A, 0)
        ret_B = np.where(np.isfinite(ret_B), ret_B, 0)

        combined = 0.5 * ret_A + 0.5 * ret_B
        eq = np.zeros(ml)
        eq[0] = float(CASH0)
        for i in range(ml - 1):
            eq[i+1] = eq[i] * (1 + combined[i])

        final = eq[-1]
        nd = ml
        ann = annual_return(final, CASH0, nd)
        pk = np.maximum.accumulate(eq)
        mdd = np.min((eq - pk) / pk * 100)
        sh = np.mean(combined) / np.std(combined) * np.sqrt(252) if np.std(combined) > 0 else 0
        return {'ann': ann, 'mdd': mdd, 'sharpe': sh, 'final': final}

    portfolio_configs = [
        (0.15, 20, 20, 0, "Port DD15% EQMA20"),
        (0.20, 20, 20, 0, "Port DD20% EQMA20"),
        (0.20, 20, 20, 4, "Port DD20% EQMA20 lose4"),
        (0.25, 20, 20, 0, "Port DD25% EQMA20"),
        (0.25, 20, 20, 4, "Port DD25% EQMA20 lose4"),
        (0.25, 20, 30, 0, "Port DD25% EQMA30"),
        (0.25, 20, 30, 4, "Port DD25% EQMA30 lose4"),
        (0.30, 20, 20, 0, "Port DD30% EQMA20"),
        (0.30, 20, 20, 4, "Port DD30% EQMA20 lose4"),
        (0.30, 20, 30, 0, "Port DD30% EQMA30"),
        (0.30, 20, 30, 4, "Port DD30% EQMA30 lose4"),
        (0.20, 30, 20, 4, "Port DD20% resumeMA30 EQMA20 lose4"),
        (0.25, 30, 20, 4, "Port DD25% resumeMA30 EQMA20 lose4"),
        (0.15, 20, 20, 4, "Port DD15% EQMA20 lose4"),
    ]

    port_results = []
    print(f"\n  {'Config':70s} | {'Ann':>8s} | {'MDD':>6s} | {'Sh':>4s}")
    print(f"  {'-'*70}-+-{'-'*8}-+-{'-'*6}-+-{'-'*4}")
    for dd_stop, dd_resume, eq_ma, lose_s, label in portfolio_configs:
        r = backtest_portfolio_dd(sig_union, sig_v121, dd_stop=dd_stop, dd_resume_ma=dd_resume,
                                   eq_ma_filter=eq_ma, lose_streak=lose_s)
        desc = f"Union/V121 {label}"
        r['desc'] = desc
        r['port_params'] = (dd_stop, dd_resume, eq_ma, lose_s)
        port_results.append(r)
        print(f"  {desc:70s} | {r['ann']:+8.1f}% | {r['mdd']:6.1f}% | {r['sharpe']:4.2f}")

    # ===================== SECTION 7: WALK-FORWARD FOR BEST CONFIGS =====================
    print("\n" + "=" * 120)
    print("  SECTION 7: WALK-FORWARD VALIDATION")
    print("=" * 120)

    # Collect ALL results with MDD > -60%
    all_results = dd_results + streak_results + eqma_results + combined_results + port_results
    reasonable = [r for r in all_results if r['mdd'] > -60 and r.get('desc', '')]
    reasonable.sort(key=lambda x: -x['ann'])

    print(f"\n  Configs with MDD > -60% (sorted by return):")
    for i, r in enumerate(reasonable[:15]):
        desc = r.get('desc', '')
        print(f"  #{i+1}: {desc:70s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    safe = [r for r in all_results if r['mdd'] > -50 and r.get('desc', '')]
    safe.sort(key=lambda x: -x['ann'])
    print(f"\n  Configs with MDD > -50%:")
    for i, r in enumerate(safe[:10]):
        desc = r.get('desc', '')
        print(f"  #{i+1}: {desc:70s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    # WF for top 5 reasonable configs
    print(f"\n  Walk-Forward for top 5 reasonable configs:")
    for idx, r in enumerate(reasonable[:5]):
        desc = r.get('desc', '')
        params = r.get('params', None)

        if params:
            sig, dd_stop, dd_resume, lose_s, wr_g, kelly, eq_ma = params
            wf_r = wf_params(sig, hold=1, topn=1,
                              dd_stop=dd_stop, dd_resume_ma=dd_resume,
                              lose_streak=lose_s, wr_gate=wr_g, kelly_half=kelly,
                              eq_ma_filter=eq_ma)
        elif 'port_params' in r:
            dd_stop, dd_resume, eq_ma, lose_s = r['port_params']
            wf_r = {}
            for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
                ys = ye = None
                for di in range(ND):
                    if dates[di].year == yr and ys is None: ys = di
                    if dates[di].year == yr: ye = di + 1
                if ys is None: continue
                wr = backtest_portfolio_dd(sig_union, sig_v121, start_di=ys, end_di=ye,
                                            dd_stop=dd_stop, dd_resume_ma=dd_resume,
                                            eq_ma_filter=eq_ma, lose_streak=lose_s)
                wf_r[yr] = wr['ann']
        else:
            continue

        pos = sum(1 for v in wf_r.values() if v > 0)
        avg = np.mean(list(wf_r.values())) if wf_r else 0
        ws = " | ".join([f"{yr}:{v:+.0f}%" for yr, v in sorted(wf_r.items())])
        print(f"  #{idx+1}: {desc}")
        print(f"       {pos}/6 | Avg={avg:>+7.0f}% | {ws}")

    # ===================== COMPREHENSIVE SUMMARY =====================
    print("\n" + "=" * 120)
    print("  COMPREHENSIVE SUMMARY")
    print("=" * 120)

    print(f"\n  --- Baselines ---")
    pr(r_v121, "V121 baseline (no control)")
    pr(r_union, "Union baseline (no control)")

    print(f"\n  --- Best by MDD < 50% ---")
    for i, r in enumerate(safe[:5]):
        desc = r.get('desc', '')
        print(f"  #{i+1}: {desc:70s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    print(f"\n  --- Best by MDD < 60% ---")
    for i, r in enumerate(reasonable[:5]):
        desc = r.get('desc', '')
        print(f"  #{i+1}: {desc:70s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    # Return/MDD ratio
    all_with_ratio = [(r, abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0)
                       for r in all_results if r.get('desc', '')]
    all_with_ratio.sort(key=lambda x: -x[1])
    print(f"\n  --- Top 10 by Ann/MDD ratio ---")
    for i, (r, ratio) in enumerate(all_with_ratio[:10]):
        desc = r.get('desc', '')
        print(f"  #{i+1}: {desc:70s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Ratio={ratio:.2f}")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
