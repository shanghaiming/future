"""
Alpha Futures V178 — Exit Optimization + Multi-Timeframe Confirmation
==============================================================================
V177 champion: short_mirror, atr_norm<10%, corr=0.5, top_n=3
  → +187% annual, -16% MDD, R/M=11.41, 3771 trades

V178 explores:
  A. ADAPTIVE EXIT: Close when momentum fades instead of fixed hold=1
     - Long exit: ROC(5) turns negative OR ROC(5) declining for 2+ days
     - Short exit: ROC(5) turns positive OR ROC(5) improving for 2+ days
     - Max hold = 5 days (safety cap)
  B. ATR TRAILING STOP: Trail stop at entry ± N*ATR for risk control
  C. MULTI-TF CONFIRM: Require ROC(5) AND ROC(10) aligned for stronger signal
  D. PROFIT TARGET: Take profit at 2*ATR move (let winners run with trail)
  E. COMBINATIONS: Best signal + best exit combo

Key hypothesis: Fixed hold=1 leaves money on the table. Momentum signals should
persist — if momentum is still strong, keep the position. Exit only when it fades.
This should reduce whipsaw trades (enter + exit at loss next day) and capture
larger moves from strong trends.
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
    print("  V178 — Exit Optimization + Multi-TF Confirmation")
    print("  Base: V177 best config (short_mirror, atr<10%, c=0.5, top_n=3)")
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
    VOL_P90 = np.percentile(valid_vols, 90) if len(valid_vols) > 0 else 2.0

    print(f"  Market vol: median={VOL_MEDIAN:.4f}%, P50={VOL_P50:.4f}%, P75={VOL_P75:.4f}%")
    print(f"  Precompute done ({time.time()-t0:.1f}s)")

    # ===================== SIGNAL DEFINITIONS =====================
    def sig_v121(di, edi):
        """Original V121 long signal: ROC(5)>1% + Z>1.5 + ROC improving"""
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

    def sig_v121_mtf(di, edi):
        """Multi-timeframe V121: ROC(5)>1% AND ROC(10)>2% + Z>1.5"""
        c = []
        for s in range(NS):
            roc5 = ROC5[s, di]; roc10 = ROC10[s, di]; zs = ZSCORE[s, di]
            if any(np.isnan(x) for x in [roc5, roc10, zs]): continue
            if roc5 <= 1.0 or roc10 <= 2.0 or zs <= 1.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc5 <= rp: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((roc5 * roc10 * zs / 10.0, s, ep, 'v121_mtf'))
        return c

    def sig_v121_short(di, edi):
        """V121 short signal: ROC(5)<-1% + Z<-1.5 + ROC declining"""
        c = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc >= -1.0 or zs >= -1.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc >= rp: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((abs(roc * zs), s, ep, 'v121_short'))
        return c

    def sig_v121_short_mtf(di, edi):
        """Multi-timeframe short: ROC(5)<-1% AND ROC(10)<-2% + Z<-1.5"""
        c = []
        for s in range(NS):
            roc5 = ROC5[s, di]; roc10 = ROC10[s, di]; zs = ZSCORE[s, di]
            if any(np.isnan(x) for x in [roc5, roc10, zs]): continue
            if roc5 >= -1.0 or roc10 >= -2.0 or zs >= -1.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc5 >= rp: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((abs(roc5 * roc10 * zs) / 10.0, s, ep, 'v121_short_mtf'))
        return c

    def sig_union(di, edi):
        all_sigs = {}
        for item in sig_v121(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc * 3
            all_sigs[s][2].append('v121')
        return [(sc, s, ep, '+'.join(sigs)) for s, (sc, ep, sigs) in all_sigs.items()]

    # ===================== HELPERS =====================
    def get_corr(si_a, si_b, di, window=20):
        start_idx = max(0, di - window)
        ret_a = RET[si_a, start_idx:di]
        ret_b = RET[si_b, start_idx:di]
        valid = ~(np.isnan(ret_a) | np.isnan(ret_b))
        n_valid = np.sum(valid)
        if n_valid < 8: return 0.5
        ra = ret_a[valid]; rb = ret_b[valid]
        if np.std(ra) == 0 or np.std(rb) == 0: return 0.5
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
        if high_water <= 0: return tiers[0][1]
        dd = (pv - high_water) / high_water
        for dd_thresh, size_frac in tiers:
            if dd >= -dd_thresh: return size_frac
        return tiers[-1][1]

    # ===================== EXIT CONDITIONS =====================
    def should_exit_momentum(p, di):
        """Exit long when ROC turns negative, exit short when ROC turns positive.
        Also exit if momentum is fading (ROC declining for 2 days)."""
        si = p['si']; d = p.get('dir', 1)
        roc = ROC5[si, di]
        if np.isnan(roc): return False

        if d == 1:  # long: exit if momentum turns negative
            if roc < -0.5: return True  # clear reversal
            # Check 2-day declining ROC
            roc_prev = ROC5[si, di-1] if di > 0 else np.nan
            roc_prev2 = ROC5[si, di-2] if di > 1 else np.nan
            if (not np.isnan(roc_prev) and not np.isnan(roc_prev2)
                and roc < roc_prev < roc_prev2 and roc < 0.5):
                return True  # momentum fading
        else:  # short: exit if momentum turns positive
            if roc > 0.5: return True
            roc_prev = ROC5[si, di-1] if di > 0 else np.nan
            roc_prev2 = ROC5[si, di-2] if di > 1 else np.nan
            if (not np.isnan(roc_prev) and not np.isnan(roc_prev2)
                and roc > roc_prev > roc_prev2 and roc > -0.5):
                return True
        return False

    def should_exit_trail(p, di, trail_atr_mult=2.0):
        """ATR trailing stop: trail stop from best price seen."""
        si = p['si']; d = p.get('dir', 1)
        atr = ATR14[si, di]
        cp = C[si, di]
        if np.isnan(atr) or np.isnan(cp) or atr <= 0: return False

        best = p.get('best_price', p['entry_price'])
        if d == 1:
            if cp > best:
                p['best_price'] = cp
                best = cp
            trail = best - trail_atr_mult * atr
            return cp < trail
        else:
            if cp < best:
                p['best_price'] = cp
                best = cp
            trail = best + trail_atr_mult * atr
            return cp > trail

    # ===================== BACKTEST ENGINE =====================
    # exit_mode controls how positions are closed:
    #   'fixed'      : hold=N days, close on day N (V177 baseline)
    #   'momentum'   : exit when momentum fades (ROC reversal), max hold 5
    #   'trail'      : ATR trailing stop, no fixed hold
    #   'momentum_trail': momentum exit + ATR trail as safety
    #   'roc10_exit' : exit when ROC(10) reverses (slower, less whipsaw)
    #
    # signal_mode:
    #   'v121'       : original V121 signal (baseline)
    #   'v121_mtf'   : multi-timeframe (ROC5 + ROC10 confirmation)

    def backtest(start_di=MIN_TRAIN, end_di=None,
                 atr_norm_max=10.0, max_corr=0.5,
                 dd_tiers=None,
                 regime_lo=0.5, regime_hi=1.5,
                 top_n=3, short_mode='short_mirror',
                 exit_mode='fixed', hold=1,
                 signal_mode='v121',
                 trail_atr_mult=2.0,
                 max_hold=5):
        if end_di is None: end_di = ND
        if dd_tiers is None:
            dd_tiers = [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)]

        cash = float(CASH0)
        positions = []
        trades = []
        daily_eq = []
        high_water = float(CASH0)
        trade_pnls = []  # track PnL in yuan for each trade

        for di in range(start_di, end_di - 1):
            # Mark-to-market
            pv = cash
            for p in positions:
                cp = C[p['si'], di]
                if not np.isnan(cp) and cp > 0:
                    m = MULT.get(p['sym'], DEF_MULT)
                    d = p.get('dir', 1)
                    unrealized = (cp - p['entry_price']) * m * p['lots'] * d
                    pv += p['entry_price'] * m * abs(p['lots']) + unrealized - cp * m * abs(p['lots']) * COMM
            daily_eq.append(pv)
            if pv > high_water:
                high_water = pv

            # --- Exit logic ---
            cl = []
            for p in positions:
                si = p['si']; d = p.get('dir', 1)
                days_held = di - p['entry_di']
                cp = C[si, di]
                if np.isnan(cp) or cp <= 0: continue

                should_close = False

                if exit_mode == 'fixed':
                    if days_held >= hold:
                        should_close = True

                elif exit_mode == 'momentum':
                    # Exit if momentum fades OR max hold reached
                    if should_exit_momentum(p, di):
                        should_close = True
                    elif days_held >= max_hold:
                        should_close = True

                elif exit_mode == 'trail':
                    if should_exit_trail(p, di, trail_atr_mult):
                        should_close = True
                    elif days_held >= max_hold:
                        should_close = True

                elif exit_mode == 'momentum_trail':
                    if should_exit_momentum(p, di):
                        should_close = True
                    if should_exit_trail(p, di, trail_atr_mult):
                        should_close = True
                    if days_held >= max_hold:
                        should_close = True

                elif exit_mode == 'roc10_exit':
                    # Exit based on ROC(10) reversal — slower signal
                    roc10 = ROC10[si, di]
                    if d == 1 and not np.isnan(roc10) and roc10 < -1.0:
                        should_close = True
                    elif d == -1 and not np.isnan(roc10) and roc10 > 1.0:
                        should_close = True
                    elif days_held >= max_hold:
                        should_close = True

                if should_close:
                    m = MULT.get(p['sym'], DEF_MULT)
                    pnl = (cp - p['entry_price']) * m * p['lots'] * d
                    inv = p['entry_price'] * m * abs(p['lots'])
                    pp = pnl / inv * 100 if inv > 0 else 0
                    if d == 1:
                        cash += cp * m * abs(p['lots']) * (1 - COMM)
                    else:
                        margin = p['entry_price'] * m * abs(p['lots'])
                        cash += margin + pnl - cp * m * abs(p['lots']) * COMM
                    trades.append(pp)
                    trade_pnls.append(pnl)
                    cl.append(p)

            for p in cl: positions.remove(p)

            # --- Position sizing ---
            dd_sz = dd_size(pv, high_water, dd_tiers)
            composite = compute_composite(di, daily_eq, high_water)
            regime_mult = regime_lo + composite * (regime_hi - regime_lo)
            pos_size = max(0.05, min(0.99, dd_sz * regime_mult))

            # --- Enter positions ---
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue

            held_si = set(p['si'] for p in positions)

            # Choose signal function based on signal_mode
            if signal_mode == 'v121':
                sig_long = sig_v121
                sig_short_fn = sig_v121_short
            elif signal_mode == 'v121_mtf':
                sig_long = sig_v121_mtf
                sig_short_fn = sig_v121_short_mtf
            else:
                sig_long = sig_v121
                sig_short_fn = sig_v121_short

            # Long signals
            cands_long = sig_long(di, edi)
            cands_long_f = [c for c in cands_long
                           if not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < atr_norm_max]
            cands_long_f.sort(key=lambda x: -x[0])

            best_long = None
            for c in cands_long_f:
                if c[1] not in held_si:
                    best_long = c
                    break

            entries = []
            if best_long:
                entries.append((best_long[0], best_long[1], best_long[2], 'long', pos_size, 1))

            # Short signals
            if short_mode != 'long_only' and len(positions) + len(entries) < top_n:
                cands_short = sig_short_fn(di, edi)
                cands_short_f = [c for c in cands_short
                                if not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < atr_norm_max]
                cands_short_f.sort(key=lambda x: -x[0])

                held_si = set(p['si'] for p in positions) | set(e[1] for e in entries)
                best_short = None
                for c in cands_short_f:
                    if c[1] not in held_si:
                        best_short = c
                        break

                if best_short:
                    entries.append((best_short[0], best_short[1], best_short[2], 'short', pos_size, -1))

            cash_snapshot = cash
            n_planned = len(entries)
            for sc, s, pr, sig_str, pct, d in entries:
                if s in set(p['si'] for p in positions): continue
                if len(positions) >= top_n: break
                cap = cash_snapshot * pct / max(n_planned, 1)
                sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                ct = max(1, int(cap / (pr * m * (1 + COMM))))
                ci = pr * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                    ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                if ct <= 0 or ci <= 0 or ci > cash: continue
                cash -= ci
                pos = {'si': s, 'entry_price': pr, 'entry_di': edi,
                       'lots': ct, 'dir': d, 'sym': sym,
                       'hold_days': hold if exit_mode == 'fixed' else max_hold,
                       'sig': sig_str, 'score': sc}
                # Track best price for trailing stop
                pos['best_price'] = pr if d == -1 else pr  # for shorts, lower is better
                positions.append(pos)

        # Close remaining
        for p in positions:
            ae = end_di - 1
            ep = C[p['si'], min(ae, ND-1)]
            if np.isnan(ep) or ep <= 0: ep = p['entry_price']
            m = MULT.get(p['sym'], DEF_MULT)
            d = p.get('dir', 1)
            if d == 1:
                cash += ep * m * abs(p['lots']) * (1 - COMM)
            else:
                pnl = (ep - p['entry_price']) * m * p['lots'] * d
                margin = p['entry_price'] * m * abs(p['lots'])
                cash += margin + pnl - ep * m * abs(p['lots']) * COMM

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

        # Avg hold days
        if trade_pnls:
            avg_pnl = np.mean(trade_pnls)
        else:
            avg_pnl = 0

        return {'ann': ann, 'wr': wr, 'n': nt, 'mdd': mdd, 'sharpe': sh,
                'final': cash, 'avg_pnl': avg_pnl}

    # ===================== PRINTING HELPERS =====================
    def pr(r, label=""):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  {label:85s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:6.2f} | N={r['n']:4d} | AvgPnL={r['avg_pnl']:>8.0f}")

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
    all_results = []

    # ===================== SECTION 0: BASELINE (V177 best) =====================
    print("\n" + "=" * 130)
    print("  SECTION 0: BASELINE — V177 best (fixed hold=1)")
    print("=" * 130)

    r_base = backtest(exit_mode='fixed', hold=1, signal_mode='v121',
                      short_mode='short_mirror')
    pr(r_base, "BASELINE: fixed hold=1, v121, short_mirror")
    all_results.append({**r_base, 'label': 'baseline_v177', 'exit_mode': 'fixed',
                        'hold': 1, 'signal_mode': 'v121'})

    # ===================== SECTION 1: FIXED HOLD 2-5 DAYS =====================
    print("\n" + "=" * 130)
    print("  SECTION 1: Fixed hold = 2,3,5 days")
    print("=" * 130)

    for h in [2, 3, 5]:
        r = backtest(exit_mode='fixed', hold=h, signal_mode='v121',
                     short_mode='short_mirror')
        pr(r, f"FIXED hold={h}")
        all_results.append({**r, 'label': f'fixed_h{h}', 'exit_mode': 'fixed',
                            'hold': h, 'signal_mode': 'v121'})

    # ===================== SECTION 2: MOMENTUM EXIT =====================
    print("\n" + "=" * 130)
    print("  SECTION 2: Momentum exit (ROC reversal)")
    print("=" * 130)

    for mh in [3, 5, 8, 10]:
        r = backtest(exit_mode='momentum', hold=1, signal_mode='v121',
                     short_mode='short_mirror', max_hold=mh)
        pr(r, f"MOMENTUM exit, max_hold={mh}")
        all_results.append({**r, 'label': f'momentum_mh{mh}', 'exit_mode': 'momentum',
                            'max_hold': mh, 'signal_mode': 'v121'})

    # ===================== SECTION 3: ATR TRAILING STOP =====================
    print("\n" + "=" * 130)
    print("  SECTION 3: ATR Trailing Stop")
    print("=" * 130)

    for tam in [1.5, 2.0, 2.5, 3.0]:
        for mh in [5, 10]:
            r = backtest(exit_mode='trail', hold=1, signal_mode='v121',
                         short_mode='short_mirror', trail_atr_mult=tam, max_hold=mh)
            pr(r, f"TRAIL {tam}*ATR, max_hold={mh}")
            all_results.append({**r, 'label': f'trail_atr{tam}_mh{mh}',
                                'exit_mode': 'trail', 'trail_atr_mult': tam,
                                'max_hold': mh, 'signal_mode': 'v121'})

    # ===================== SECTION 4: MOMENTUM + TRAIL COMBINED =====================
    print("\n" + "=" * 130)
    print("  SECTION 4: Momentum + Trail combined")
    print("=" * 130)

    for tam in [2.0, 2.5, 3.0]:
        for mh in [5, 8]:
            r = backtest(exit_mode='momentum_trail', hold=1, signal_mode='v121',
                         short_mode='short_mirror', trail_atr_mult=tam, max_hold=mh)
            pr(r, f"MOMENTUM+TRAIL {tam}*ATR, max_hold={mh}")
            all_results.append({**r, 'label': f'mom_trail_atr{tam}_mh{mh}',
                                'exit_mode': 'momentum_trail', 'trail_atr_mult': tam,
                                'max_hold': mh, 'signal_mode': 'v121'})

    # ===================== SECTION 5: ROC(10) EXIT =====================
    print("\n" + "=" * 130)
    print("  SECTION 5: ROC(10) exit (slower signal)")
    print("=" * 130)

    for mh in [5, 8, 10]:
        r = backtest(exit_mode='roc10_exit', hold=1, signal_mode='v121',
                     short_mode='short_mirror', max_hold=mh)
        pr(r, f"ROC10_EXIT, max_hold={mh}")
        all_results.append({**r, 'label': f'roc10_exit_mh{mh}',
                            'exit_mode': 'roc10_exit', 'max_hold': mh,
                            'signal_mode': 'v121'})

    # ===================== SECTION 6: MULTI-TF SIGNAL =====================
    print("\n" + "=" * 130)
    print("  SECTION 6: Multi-TF Signal (ROC5 + ROC10 confirmation)")
    print("=" * 130)

    # MTF with fixed hold=1 (compare to baseline)
    r_mtf = backtest(exit_mode='fixed', hold=1, signal_mode='v121_mtf',
                     short_mode='short_mirror')
    pr(r_mtf, "MTF fixed hold=1")
    all_results.append({**r_mtf, 'label': 'mtf_fixed_h1',
                        'exit_mode': 'fixed', 'hold': 1, 'signal_mode': 'v121_mtf'})

    # MTF with momentum exit
    for mh in [5, 8]:
        r = backtest(exit_mode='momentum', hold=1, signal_mode='v121_mtf',
                     short_mode='short_mirror', max_hold=mh)
        pr(r, f"MTF momentum exit, max_hold={mh}")
        all_results.append({**r, 'label': f'mtf_momentum_mh{mh}',
                            'exit_mode': 'momentum', 'max_hold': mh,
                            'signal_mode': 'v121_mtf'})

    # MTF with trail
    for tam in [2.0, 2.5]:
        r = backtest(exit_mode='trail', hold=1, signal_mode='v121_mtf',
                     short_mode='short_mirror', trail_atr_mult=tam, max_hold=8)
        pr(r, f"MTF trail {tam}*ATR, max_hold=8")
        all_results.append({**r, 'label': f'mtf_trail_atr{tam}_mh8',
                            'exit_mode': 'trail', 'trail_atr_mult': tam,
                            'max_hold': 8, 'signal_mode': 'v121_mtf'})

    # MTF with momentum+trail
    for tam in [2.0, 2.5]:
        r = backtest(exit_mode='momentum_trail', hold=1, signal_mode='v121_mtf',
                     short_mode='short_mirror', trail_atr_mult=tam, max_hold=8)
        pr(r, f"MTF momentum+trail {tam}*ATR, max_hold=8")
        all_results.append({**r, 'label': f'mtf_mom_trail_atr{tam}_mh8',
                            'exit_mode': 'momentum_trail', 'trail_atr_mult': tam,
                            'max_hold': 8, 'signal_mode': 'v121_mtf'})

    # ===================== SECTION 7: LONG-ONLY BASELINES FOR EXIT COMPARISON =====================
    print("\n" + "=" * 130)
    print("  SECTION 7: Long-only with exit variants")
    print("=" * 130)

    r_lo_base = backtest(exit_mode='fixed', hold=1, signal_mode='v121',
                         short_mode='long_only')
    pr(r_lo_base, "LONG-ONLY fixed hold=1")
    all_results.append({**r_lo_base, 'label': 'longonly_fixed_h1',
                        'exit_mode': 'fixed', 'hold': 1, 'signal_mode': 'v121'})

    for em in ['momentum', 'trail', 'momentum_trail', 'roc10_exit']:
        kwargs = {'exit_mode': em, 'hold': 1, 'signal_mode': 'v121',
                  'short_mode': 'long_only'}
        if em in ('trail', 'momentum_trail'):
            kwargs['trail_atr_mult'] = 2.0
        if em != 'fixed':
            kwargs['max_hold'] = 5
        r = backtest(**kwargs)
        pr(r, f"LONG-ONLY {em}")
        all_results.append({**r, 'label': f'longonly_{em}',
                            'exit_mode': em, 'signal_mode': 'v121'})

    # ===================== WALK-FORWARD =====================
    print("\n" + "=" * 130)
    print("  WALK-FORWARD: Top configs")
    print("=" * 130)

    base_ann = r_base['ann']; base_mdd = r_base['mdd']
    base_rm = abs(base_ann / base_mdd) if base_mdd != 0 else 0
    print(f"\n  Baseline: Ann={base_ann:+.0f}% | MDD={base_mdd:.0f}% | R/M={base_rm:.2f}")

    # WF baseline
    print(f"\n  Walk-forward: BASELINE")
    wf_base = walk_forward(label="BASELINE WF", exit_mode='fixed', hold=1,
                           signal_mode='v121', short_mode='short_mirror')
    print_wf(wf_base, "BASELINE")

    # Top 5 by R/M
    ranked = sorted(all_results, key=lambda x: abs(x.get('ann', 0) / x.get('mdd', -1)), reverse=True)
    top5 = [r for r in ranked if r.get('label') != 'baseline_v177'][:5]

    for r in top5:
        lbl = r['label']
        em = r.get('exit_mode', 'fixed')
        sm = r.get('signal_mode', 'v121')
        kwargs = {'exit_mode': em, 'signal_mode': sm,
                  'short_mode': 'short_mirror'}
        if em == 'fixed':
            kwargs['hold'] = r.get('hold', 1)
        else:
            kwargs['hold'] = 1
            kwargs['max_hold'] = r.get('max_hold', 5)
        if em in ('trail', 'momentum_trail'):
            kwargs['trail_atr_mult'] = r.get('trail_atr_mult', 2.0)
        print(f"\n  Walk-forward: {lbl}")
        wf = walk_forward(label=f"WF {lbl}", **kwargs)
        print_wf(wf, lbl)

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 130)
    print("  V178 FINAL SUMMARY: Exit + Signal Optimization")
    print("=" * 130)

    print(f"\n  {'Config':40s} | {'Ann':>7s} | {'MDD':>5s} | {'R/M':>6s} | {'WR':>5s} | {'N':>4s} | {'Sh':>5s} | Delta_R/M")
    print(f"  {'-'*40}-+-{'-'*7}-+-{'-'*5}-+-{'-'*6}-+-{'-'*5}-+-{'-'*4}-+-{'-'*5}-+-{'-'*8}")
    print(f"  {'BASELINE V177 (hold=1)':40s} | {base_ann:>+7.0f}% | {base_mdd:>5.0f}% | {base_rm:>6.2f} | {r_base['wr']:>5.1f}% | {r_base['n']:>4d} | {r_base['sharpe']:>5.2f} |    ---")

    for r in ranked[:20]:
        ann = r['ann']; mdd = r['mdd']
        rm = abs(ann / mdd) if mdd != 0 else 0
        delta = rm - base_rm
        marker = " ***" if delta > 2.0 else (" **" if delta > 1.0 else "")
        print(f"  {r['label']:40s} | {ann:>+7.0f}% | {mdd:>5.0f}% | {rm:>6.2f} | {r['wr']:>5.1f}% | {r['n']:>4d} | {r['sharpe']:>5.2f} | {delta:>+8.2f}{marker}")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
