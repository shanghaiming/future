"""
Alpha Futures V141 — ADAPTIVE/DYNAMIC POSITION SIZING
=============================================================================
Hypothesis: Adaptive sizing (bigger when winning, smaller when losing) should
give better return/MDD ratios than simple fixed 50% sizing.

Approaches tested:
  1) WR-Adaptive Sizing       — size based on rolling 20-trade win rate
  2) Signal Strength Sizing   — size based on signal score percentile
  3) Equity Curve Scaling     — continuous scaling based on equity vs MA
  4) Anti-Martingale Recovery  — size based on consecutive W/L streaks
  5) Combined Adaptive        — WR-adaptive x equity curve scaling
  6) Profit Lock-in           — reduce sizing after large monthly gains

Baselines:
  - V121 at 50% fixed sizing
  - Union at 50% fixed sizing
  - Union/V121 50/50 portfolio at 50% fixed sizing (same as V140 champion)
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
    print("  V141 — ADAPTIVE/DYNAMIC POSITION SIZING")
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

    # ===================== ADAPTIVE BACKTEST ENGINE =====================
    def backtest_adaptive(signal_func, hold=1, top_n=1, start_di=MIN_TRAIN, end_di=None,
                          sizing_mode='fixed', size_frac=0.50):
        """
        sizing_mode controls how position size is determined:
          'fixed'       — constant size_frac (baseline)
          'wr_adaptive' — size based on rolling 20-trade WR
          'signal_str'  — size based on signal score percentile
          'eq_curve'    — continuous scaling based on equity / MA(equity, 20)
          'anti_mart'   — size based on consecutive W/L streaks
          'combined'    — wr_adaptive x eq_curve
          'profit_lock' — reduce after large monthly gains
        """
        if end_di is None: end_di = ND
        cash = float(CASH0)
        positions = []
        daily_eq = []
        trades = []          # list of pnl_pct
        signal_scores = []   # rolling history for percentile
        high_water = float(CASH0)
        consecutive_wins = 0
        consecutive_losses = 0

        # For profit lock-in: track monthly start equity
        current_month = None
        month_start_eq = float(CASH0)

        for di in range(start_di, end_di - 1):
            # Mark-to-market
            pv = cash
            for p in positions:
                cp = C[p['si'], di]
                if not np.isnan(cp) and cp > 0:
                    m = MULT.get(p['sym'], DEF_MULT)
                    pv += cp * m * p['lots'] - cp * m * abs(p['lots']) * COMM
            daily_eq.append(pv)
            if pv > high_water: high_water = pv

            # Track month for profit lock-in
            cur_date = dates[di]
            cur_month_key = (cur_date.year, cur_date.month)
            if current_month is None or cur_month_key != current_month:
                current_month = cur_month_key
                month_start_eq = pv

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
                    # Update streaks
                    if pp > 0:
                        consecutive_wins += 1
                        consecutive_losses = 0
                    else:
                        consecutive_losses += 1
                        consecutive_wins = 0
                    cl.append(p)
            for p in cl: positions.remove(p)

            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue
            cands = signal_func(di, edi)
            if not cands: continue
            cands.sort(key=lambda x: -x[0])

            # ========== DETERMINE SIZING FRACTION ==========
            sf = size_frac  # default

            if sizing_mode == 'fixed':
                pass  # use size_frac as-is

            elif sizing_mode == 'wr_adaptive':
                if len(trades) >= 20:
                    recent_wr = np.mean([1 if t > 0 else 0 for t in trades[-20:]])
                    if recent_wr > 0.70:
                        sf = 0.90
                    elif recent_wr > 0.60:
                        sf = 0.70
                    elif recent_wr > 0.50:
                        sf = 0.40
                    else:
                        sf = 0.15
                else:
                    sf = 0.50  # not enough data yet

            elif sizing_mode == 'signal_str':
                if cands:
                    best_score = cands[0][0]
                    signal_scores.append(best_score)
                    if len(signal_scores) > 100:
                        signal_scores = signal_scores[-100:]
                    if len(signal_scores) >= 20:
                        pctl = np.percentile(signal_scores, [25, 50, 75])
                        if best_score >= pctl[2]:      # top 25%
                            sf = 0.90
                        elif best_score >= pctl[1]:     # 25-50%
                            sf = 0.70
                        elif best_score >= pctl[0]:     # 50-75%
                            sf = 0.50
                        else:                            # bottom 25%
                            sf = 0.30
                    else:
                        sf = 0.50

            elif sizing_mode == 'eq_curve':
                if len(daily_eq) >= 20:
                    eq_ma = np.mean(daily_eq[-20:])
                    if eq_ma > 0:
                        ratio = pv / eq_ma
                        # Linear interpolation between anchor points
                        # >1.05 -> 80%, 1.0 -> 60%, 0.95 -> 30%, <0.90 -> 10%
                        if ratio >= 1.05:
                            sf = 0.80
                        elif ratio >= 1.00:
                            # 1.00->0.60, 1.05->0.80, linear
                            sf = 0.60 + (ratio - 1.00) / 0.05 * 0.20
                        elif ratio >= 0.95:
                            # 0.95->0.30, 1.00->0.60, linear
                            sf = 0.30 + (ratio - 0.95) / 0.05 * 0.30
                        elif ratio >= 0.90:
                            # 0.90->0.10, 0.95->0.30, linear
                            sf = 0.10 + (ratio - 0.90) / 0.05 * 0.20
                        else:
                            sf = 0.10
                    else:
                        sf = 0.10
                else:
                    sf = 0.50

            elif sizing_mode == 'anti_mart':
                if consecutive_wins >= 3:
                    sf = 0.80
                elif consecutive_losses >= 4:
                    sf = 0.15
                elif consecutive_losses >= 2:
                    sf = 0.30
                else:
                    sf = 0.60

            elif sizing_mode == 'combined':
                # WR component
                if len(trades) >= 20:
                    recent_wr = np.mean([1 if t > 0 else 0 for t in trades[-20:]])
                    if recent_wr > 0.70:   wr_mult = 0.90
                    elif recent_wr > 0.60: wr_mult = 0.70
                    elif recent_wr > 0.50: wr_mult = 0.40
                    else:                  wr_mult = 0.15
                else:
                    wr_mult = 0.50

                # Equity curve component
                if len(daily_eq) >= 20:
                    eq_ma = np.mean(daily_eq[-20:])
                    if eq_ma > 0:
                        ratio = pv / eq_ma
                        if ratio >= 1.05:      eq_mult = 0.80
                        elif ratio >= 1.00:    eq_mult = 0.60 + (ratio - 1.00) / 0.05 * 0.20
                        elif ratio >= 0.95:    eq_mult = 0.30 + (ratio - 0.95) / 0.05 * 0.30
                        elif ratio >= 0.90:    eq_mult = 0.10 + (ratio - 0.90) / 0.05 * 0.20
                        else:                  eq_mult = 0.10
                    else:
                        eq_mult = 0.10
                else:
                    eq_mult = 0.50

                sf = wr_mult * eq_mult / 0.50  # normalize around 0.50 center
                sf = max(0.05, min(0.95, sf))

            elif sizing_mode == 'profit_lock':
                if month_start_eq > 0:
                    month_ret = (pv - month_start_eq) / month_start_eq
                    if month_ret > 0.25:
                        sf = 0.15
                    elif month_ret > 0.15:
                        sf = 0.30
                    else:
                        sf = 0.70  # default for profit_lock
                else:
                    sf = 0.70

            # ========== ENTER POSITION ==========
            ns = top_n - len(positions)
            cap = cash * sf / max(1, ns)
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
                                  'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': hold,
                                  'sig': sig})

        # Liquidate remaining
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
        ap = np.mean(trades) if trades else 0
        if daily_eq:
            eq = np.array(daily_eq); pk = np.maximum.accumulate(eq)
            mdd = np.min((eq - pk) / pk * 100)
            r = np.diff(eq) / eq[:-1]
            r = np.where(np.isfinite(r), r, 0)
            sh = np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0
        else:
            mdd = 0; sh = 0
        return {'ann': ann, 'wr': wr, 'n': nt, 'avg_pnl': ap, 'mdd': mdd, 'sharpe': sh,
                'final': cash}

    # ===================== PORTFOLIO BACKTEST (50/50 combined) =====================
    def backtest_portfolio(sig_A, sig_B, start_di=MIN_TRAIN, end_di=None,
                           sizing_mode='fixed', size_frac=0.50):
        """Run each signal independently, combine equity curves at 50/50."""
        if end_di is None: end_di = ND

        eq_A = backtest_adaptive(sig_A, hold=1, top_n=1,
                                  start_di=start_di, end_di=end_di,
                                  sizing_mode=sizing_mode, size_frac=size_frac)
        # Need to get daily_eq, not just summary. Refactor: run raw.
        def run_sub_raw(sig_func):
            cash = float(CASH0)
            positions = []
            daily_eq = []
            trades = []
            signal_scores = []
            high_water = float(CASH0)
            consecutive_wins = 0
            consecutive_losses = 0
            current_month = None
            month_start_eq = float(CASH0)

            for di in range(start_di, end_di - 1):
                pv = cash
                for p in positions:
                    cp = C[p['si'], di]
                    if not np.isnan(cp) and cp > 0:
                        m = MULT.get(p['sym'], DEF_MULT)
                        pv += cp * m * p['lots'] - cp * m * abs(p['lots']) * COMM
                daily_eq.append(pv)
                if pv > high_water: high_water = pv

                cur_date = dates[di]
                cur_month_key = (cur_date.year, cur_date.month)
                if current_month is None or cur_month_key != current_month:
                    current_month = cur_month_key
                    month_start_eq = pv

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
                        if pp > 0:
                            consecutive_wins += 1; consecutive_losses = 0
                        else:
                            consecutive_losses += 1; consecutive_wins = 0
                        cl.append(p)
                for p in cl: positions.remove(p)

                if len(positions) >= 1: continue
                edi = di + 1
                if edi >= end_di: continue
                cands = sig_func(di, edi)
                if not cands: continue
                cands.sort(key=lambda x: -x[0])

                sf = size_frac
                if sizing_mode == 'fixed':
                    pass
                elif sizing_mode == 'wr_adaptive':
                    if len(trades) >= 20:
                        recent_wr = np.mean([1 if t > 0 else 0 for t in trades[-20:]])
                        if recent_wr > 0.70:   sf = 0.90
                        elif recent_wr > 0.60: sf = 0.70
                        elif recent_wr > 0.50: sf = 0.40
                        else:                  sf = 0.15
                    else: sf = 0.50
                elif sizing_mode == 'signal_str':
                    if cands:
                        best_score = cands[0][0]
                        signal_scores.append(best_score)
                        if len(signal_scores) > 100: signal_scores = signal_scores[-100:]
                        if len(signal_scores) >= 20:
                            pctl = np.percentile(signal_scores, [25, 50, 75])
                            if best_score >= pctl[2]:      sf = 0.90
                            elif best_score >= pctl[1]:     sf = 0.70
                            elif best_score >= pctl[0]:     sf = 0.50
                            else:                           sf = 0.30
                        else: sf = 0.50
                elif sizing_mode == 'eq_curve':
                    if len(daily_eq) >= 20:
                        eq_ma = np.mean(daily_eq[-20:])
                        if eq_ma > 0:
                            ratio = pv / eq_ma
                            if ratio >= 1.05:      sf = 0.80
                            elif ratio >= 1.00:    sf = 0.60 + (ratio - 1.00) / 0.05 * 0.20
                            elif ratio >= 0.95:    sf = 0.30 + (ratio - 0.95) / 0.05 * 0.30
                            elif ratio >= 0.90:    sf = 0.10 + (ratio - 0.90) / 0.05 * 0.20
                            else:                  sf = 0.10
                        else: sf = 0.10
                    else: sf = 0.50
                elif sizing_mode == 'anti_mart':
                    if consecutive_wins >= 3:       sf = 0.80
                    elif consecutive_losses >= 4:   sf = 0.15
                    elif consecutive_losses >= 2:   sf = 0.30
                    else:                           sf = 0.60
                elif sizing_mode == 'combined':
                    if len(trades) >= 20:
                        recent_wr = np.mean([1 if t > 0 else 0 for t in trades[-20:]])
                        if recent_wr > 0.70:   wr_mult = 0.90
                        elif recent_wr > 0.60: wr_mult = 0.70
                        elif recent_wr > 0.50: wr_mult = 0.40
                        else:                  wr_mult = 0.15
                    else: wr_mult = 0.50
                    if len(daily_eq) >= 20:
                        eq_ma = np.mean(daily_eq[-20:])
                        if eq_ma > 0:
                            ratio = pv / eq_ma
                            if ratio >= 1.05:      eq_mult = 0.80
                            elif ratio >= 1.00:    eq_mult = 0.60 + (ratio - 1.00) / 0.05 * 0.20
                            elif ratio >= 0.95:    eq_mult = 0.30 + (ratio - 0.95) / 0.05 * 0.30
                            elif ratio >= 0.90:    eq_mult = 0.10 + (ratio - 0.90) / 0.05 * 0.20
                            else:                  eq_mult = 0.10
                        else: eq_mult = 0.10
                    else: eq_mult = 0.50
                    sf = wr_mult * eq_mult / 0.50
                    sf = max(0.05, min(0.95, sf))
                elif sizing_mode == 'profit_lock':
                    if month_start_eq > 0:
                        month_ret = (pv - month_start_eq) / month_start_eq
                        if month_ret > 0.25:    sf = 0.15
                        elif month_ret > 0.15:  sf = 0.30
                        else:                   sf = 0.70
                    else: sf = 0.70

                cap = cash * sf
                item = cands[0]
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
                                  'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': 1,
                                  'sig': sig})

            for p in positions:
                ae = end_di - 1
                ep = C[p['si'], min(ae, ND-1)]
                if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                m = MULT.get(p['sym'], DEF_MULT)
                cash += ep * m * abs(p['lots']) * (1 - COMM)
            return np.array(daily_eq)

        eqA = run_sub_raw(sig_A)
        eqB = run_sub_raw(sig_B)
        ml = min(len(eqA), len(eqB))
        if ml <= 1:
            return {'ann': -100.0, 'mdd': 0, 'sharpe': 0, 'final': CASH0, 'daily_eq': np.array([CASH0])}

        ret_A = np.diff(eqA[:ml]) / eqA[:ml-1]
        ret_B = np.diff(eqB[:ml]) / eqB[:ml-1]
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
        return {'ann': ann, 'mdd': mdd, 'sharpe': sh, 'final': final, 'daily_eq': eq}

    # ===================== WALK-FORWARD HELPERS =====================
    def pr(r, label=""):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  {label:65s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | "
              f"Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    def wf_portfolio(sig_A, sig_B, sizing_mode='fixed', size_frac=0.50):
        """Walk-forward for portfolio by year."""
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue
            r = backtest_portfolio(sig_A, sig_B, start_di=ys, end_di=ye,
                                    sizing_mode=sizing_mode, size_frac=size_frac)
            res[yr] = {'ann': r['ann'], 'mdd': r['mdd']}
        return res

    # ===================== BASELINES =====================
    print("\n" + "=" * 120)
    print("  BASELINES: Fixed 50% sizing")
    print("=" * 120)

    baseline_configs = [
        (sig_v121, 'fixed', 0.50, "V121 @50% fixed"),
        (sig_union, 'fixed', 0.50, "Union @50% fixed"),
    ]
    baseline_results = {}
    for sig, mode, sf, label in baseline_configs:
        r = backtest_adaptive(sig, hold=1, top_n=1, sizing_mode=mode, size_frac=sf)
        pr(r, label)
        baseline_results[label] = r

    # Portfolio baselines
    print()
    port_baseline = backtest_portfolio(sig_union, sig_v121, sizing_mode='fixed', size_frac=0.50)
    pr(port_baseline, "Union/V121 50/50 @50% fixed (PORTFOLIO)")
    baseline_results['port_5050_50'] = port_baseline

    # ===================== SECTION 1: WR-ADAPTIVE SIZING =====================
    print("\n" + "=" * 120)
    print("  APPROACH 1: WR-Adaptive Sizing (20-trade rolling WR)")
    print("  WR>70% -> 90% | WR>60% -> 70% | WR>50% -> 40% | WR<50% -> 15%")
    print("=" * 120)

    wr_results = {}
    for sig, label in [(sig_v121, "V121"), (sig_union, "Union")]:
        r = backtest_adaptive(sig, hold=1, top_n=1, sizing_mode='wr_adaptive')
        key = f"{label}_wr_adaptive"
        pr(r, f"{label} WR-adaptive")
        wr_results[key] = r

    print()
    r = backtest_portfolio(sig_union, sig_v121, sizing_mode='wr_adaptive')
    pr(r, "Union/V121 50/50 WR-adaptive (PORTFOLIO)")
    wr_results['port_wr_adaptive'] = r

    # ===================== SECTION 2: SIGNAL STRENGTH SIZING =====================
    print("\n" + "=" * 120)
    print("  APPROACH 2: Signal Strength Sizing (100-trade percentile)")
    print("  Top25% -> 90% | 25-50% -> 70% | 50-75% -> 50% | Bot25% -> 30%")
    print("=" * 120)

    sig_str_results = {}
    for sig, label in [(sig_v121, "V121"), (sig_union, "Union")]:
        r = backtest_adaptive(sig, hold=1, top_n=1, sizing_mode='signal_str')
        key = f"{label}_signal_str"
        pr(r, f"{label} Signal-strength")
        sig_str_results[key] = r

    print()
    r = backtest_portfolio(sig_union, sig_v121, sizing_mode='signal_str')
    pr(r, "Union/V121 50/50 Signal-strength (PORTFOLIO)")
    sig_str_results['port_signal_str'] = r

    # ===================== SECTION 3: EQUITY CURVE GENTLE SCALING =====================
    print("\n" + "=" * 120)
    print("  APPROACH 3: Equity Curve Gentle Scaling (equity / MA20)")
    print("  >1.05 -> 80% | 1.0 -> 60% | 0.95 -> 30% | <0.90 -> 10%")
    print("  (linear interpolation between anchor points)")
    print("=" * 120)

    eq_results = {}
    for sig, label in [(sig_v121, "V121"), (sig_union, "Union")]:
        r = backtest_adaptive(sig, hold=1, top_n=1, sizing_mode='eq_curve')
        key = f"{label}_eq_curve"
        pr(r, f"{label} Equity-curve scaling")
        eq_results[key] = r

    print()
    r = backtest_portfolio(sig_union, sig_v121, sizing_mode='eq_curve')
    pr(r, "Union/V121 50/50 Equity-curve (PORTFOLIO)")
    eq_results['port_eq_curve'] = r

    # ===================== SECTION 4: ANTI-MARTINGALE RECOVERY =====================
    print("\n" + "=" * 120)
    print("  APPROACH 4: Anti-Martingale Recovery")
    print("  3+ wins -> 80% | normal -> 60% | 2+ losses -> 30% | 4+ losses -> 15%")
    print("=" * 120)

    am_results = {}
    for sig, label in [(sig_v121, "V121"), (sig_union, "Union")]:
        r = backtest_adaptive(sig, hold=1, top_n=1, sizing_mode='anti_mart')
        key = f"{label}_anti_mart"
        pr(r, f"{label} Anti-martingale")
        am_results[key] = r

    print()
    r = backtest_portfolio(sig_union, sig_v121, sizing_mode='anti_mart')
    pr(r, "Union/V121 50/50 Anti-martingale (PORTFOLIO)")
    am_results['port_anti_mart'] = r

    # ===================== SECTION 5: COMBINED ADAPTIVE =====================
    print("\n" + "=" * 120)
    print("  APPROACH 5: Combined Adaptive (WR-adaptive x Equity-curve)")
    print("=" * 120)

    comb_results = {}
    for sig, label in [(sig_v121, "V121"), (sig_union, "Union")]:
        r = backtest_adaptive(sig, hold=1, top_n=1, sizing_mode='combined')
        key = f"{label}_combined"
        pr(r, f"{label} Combined WRxEQ")
        comb_results[key] = r

    print()
    r = backtest_portfolio(sig_union, sig_v121, sizing_mode='combined')
    pr(r, "Union/V121 50/50 Combined WRxEQ (PORTFOLIO)")
    comb_results['port_combined'] = r

    # ===================== SECTION 6: PROFIT LOCK-IN =====================
    print("\n" + "=" * 120)
    print("  APPROACH 6: Profit Lock-in")
    print("  Monthly >+15% -> 30% | Monthly >+25% -> 15% | otherwise -> 70%")
    print("  Reset at start of each month")
    print("=" * 120)

    pl_results = {}
    for sig, label in [(sig_v121, "V121"), (sig_union, "Union")]:
        r = backtest_adaptive(sig, hold=1, top_n=1, sizing_mode='profit_lock')
        key = f"{label}_profit_lock"
        pr(r, f"{label} Profit-lock")
        pl_results[key] = r

    print()
    r = backtest_portfolio(sig_union, sig_v121, sizing_mode='profit_lock')
    pr(r, "Union/V121 50/50 Profit-lock (PORTFOLIO)")
    pl_results['port_profit_lock'] = r

    # ===================== SECTION 7: WALK-FORWARD FOR ALL PORTFOLIO CONFIGS =====================
    print("\n" + "=" * 120)
    print("  WALK-FORWARD VALIDATION: Portfolio configs (2020-2025)")
    print("=" * 120)

    portfolio_configs = [
        ('fixed', 0.50, "Baseline 50% fixed"),
        ('wr_adaptive', 0.50, "WR-adaptive"),
        ('signal_str', 0.50, "Signal-strength"),
        ('eq_curve', 0.50, "Equity-curve scaling"),
        ('anti_mart', 0.50, "Anti-martingale"),
        ('combined', 0.50, "Combined WRxEQ"),
        ('profit_lock', 0.50, "Profit-lock"),
    ]

    wf_all = {}
    for mode, sf, label in portfolio_configs:
        print(f"\n  {label} ({mode}):")
        wf_r = wf_portfolio(sig_union, sig_v121, sizing_mode=mode, size_frac=sf)
        pos = sum(1 for v in wf_r.values() if v['ann'] > 0)
        avg_ann = np.mean([v['ann'] for v in wf_r.values()]) if wf_r else 0
        worst_mdd = min(v['mdd'] for v in wf_r.values()) if wf_r else 0
        ws = " | ".join([f"{yr}:{v['ann']:+.0f}%" for yr, v in sorted(wf_r.items())])
        print(f"    {pos}/6 | Avg={avg_ann:>+7.0f}% | WorstYrMDD={worst_mdd:>.1f}%")
        print(f"    {ws}")
        wf_all[mode] = {'wf': wf_r, 'pos': pos, 'avg': avg_ann, 'worst_mdd': worst_mdd, 'label': label}

    # ===================== COMPREHENSIVE SUMMARY =====================
    print("\n" + "=" * 120)
    print("  COMPREHENSIVE SUMMARY: All results sorted by Return/MDD ratio")
    print("=" * 120)

    all_results = {}
    all_results.update({f"base_{k}": v for k, v in baseline_results.items()})
    all_results.update({f"wr_{k}": v for k, v in wr_results.items()})
    all_results.update({f"sig_{k}": v for k, v in sig_str_results.items()})
    all_results.update({f"eq_{k}": v for k, v in eq_results.items()})
    all_results.update({f"am_{k}": v for k, v in am_results.items()})
    all_results.update({f"comb_{k}": v for k, v in comb_results.items()})
    all_results.update({f"pl_{k}": v for k, v in pl_results.items()})

    # Build sortable list
    ranked = []
    for key, r in all_results.items():
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        ranked.append((key, r, ratio))
    ranked.sort(key=lambda x: -x[2])

    print(f"\n  --- Top 15 by Return/MDD Ratio ---")
    print(f"  {'#':>3s}  {'Config':65s} | {'Ann':>8s} | {'MDD':>6s} | {'Sh':>4s} | {'R/M':>5s}")
    print(f"  {'---':>3s}  {'-'*65}-+-{'-'*8}-+-{'-'*6}-+-{'-'*4}-+-{'-'*5}")
    for i, (key, r, ratio) in enumerate(ranked[:15]):
        print(f"  {i+1:3d}  {key:65s} | {r['ann']:+8.1f}% | {r['mdd']:6.1f}% | {r['sharpe']:4.2f} | {ratio:.2f}")

    # Filter for portfolio results only
    print(f"\n  --- Portfolio configs only (Union/V121 50/50) ---")
    port_ranked = [(key, r, abs(r['ann']/r['mdd']) if r['mdd'] != 0 else 0)
                   for key, r in all_results.items() if 'port' in key]
    port_ranked.sort(key=lambda x: -x[2])
    print(f"  {'#':>3s}  {'Config':65s} | {'Ann':>8s} | {'MDD':>6s} | {'Sh':>4s} | {'R/M':>5s}")
    print(f"  {'---':>3s}  {'-'*65}-+-{'-'*8}-+-{'-'*6}-+-{'-'*4}-+-{'-'*5}")
    for i, (key, r, ratio) in enumerate(port_ranked):
        print(f"  {i+1:3d}  {key:65s} | {r['ann']:+8.1f}% | {r['mdd']:6.1f}% | {r['sharpe']:4.2f} | {ratio:.2f}")

    # Walk-forward summary table
    print(f"\n  --- Walk-Forward Summary (Portfolio configs) ---")
    print(f"  {'Approach':30s} | {'Pos':>3s} | {'AvgAnn':>7s} | {'WorstMDD':>8s} | {'Years'}")
    print(f"  {'-'*30}-+-{'-'*3}-+-{'-'*7}-+-{'-'*8}-+-{'-'*50}")
    for mode, info in wf_all.items():
        wf = info['wf']
        ws = " | ".join([f"{yr}:{v['ann']:+.0f}%/{v['mdd']:.0f}%"
                         for yr, v in sorted(wf.items())])
        print(f"  {info['label']:30s} | {info['pos']:3d}/6 | {info['avg']:>+7.0f}% | "
              f"{info['worst_mdd']:>8.1f}% | {ws}")

    # Beat baseline?
    base_ratio = abs(port_baseline['ann'] / port_baseline['mdd']) if port_baseline['mdd'] != 0 else 0
    print(f"\n  Baseline (50% fixed) R/M ratio: {base_ratio:.2f}")
    print(f"  Configs that BEAT baseline:")
    beat_count = 0
    for key, r, ratio in port_ranked:
        if ratio > base_ratio and 'base' not in key:
            beat_count += 1
            print(f"    {key:65s} | R/M={ratio:.2f} (baseline={base_ratio:.2f})")
    if beat_count == 0:
        print(f"    None beat the baseline.")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
