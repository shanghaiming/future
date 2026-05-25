"""
Alpha Futures V143 — EXIT STRATEGY ENHANCEMENTS
=============================================================================
Goal: Improve return/MDD ratio via better exit strategies.

Baseline: 50/50 portfolio of Union + V121, hold=1 day, 50% position sizing
          => ~+155% annual, ~-24% MDD

Tests:
  A. Profit Target (intraday): exit if up > X% intraday
  B. Stop Loss (intraday): exit if down > X% intraday
  C. Combined PT + SL
  D. Trailing Stop
  E. Conditional Hold Extension (winners extend)
  F. Winner Extension + Loser Quick Exit

All with 50% position sizing (size_frac=0.50).
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
SIZE_FRAC = 0.50  # 50% position sizing

def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0: return -100.0
    return (final / initial) ** (1.0 / (n_days / 252)) * 100 - 100


def main():
    print("=" * 130)
    print("  V143 — EXIT STRATEGY ENHANCEMENTS (50% position sizing)")
    print("=" * 130)
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

    # ===================== BACKTEST ENGINE WITH EXIT STRATEGIES =====================
    def backtest_exit(signal_func, hold=1, top_n=1, start_di=MIN_TRAIN, end_di=None,
                      profit_target_pct=0.0,    # A: exit if intraday high > entry * (1+X), 0=off
                      stop_loss_pct=0.0,        # B: exit if intraday low < entry * (1-X), 0=off
                      trailing_stop_pct=0.0,     # D: exit if close < highest_since_entry * (1-X), 0=off
                      max_hold_ext=0,            # E: extend hold if winning (max additional days)
                      winner_ext_threshold=0.0,  # F: if intraday high > entry*(1+X), extend hold
                      loser_exit_threshold=0.0,  # F: if intraday low < entry*(1-X), exit immediately
                      size_frac=SIZE_FRAC):      # position sizing
        """
        Exit logic priority per bar:
        1. Stop loss (intraday low breaches threshold) -> exit at stop price
        2. Profit target (intraday high breaches threshold) -> exit at target price
        3. Trailing stop (close breaches trailing) -> exit at close
        4. Normal hold expiry -> exit at close
        5. Winner extension check -> may extend hold
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

            # Process exits for each position
            cl = []
            for p in positions:
                si = p['si']
                entry = p['entry_price']
                m = MULT.get(p['sym'], DEF_MULT)
                invested = entry * m * abs(p['lots'])

                lo = L[si, di]
                hi = H[si, di]
                cp = C[si, di]

                if np.isnan(lo) or np.isnan(hi) or np.isnan(cp):
                    # No data, skip
                    continue

                exit_price = None
                exit_reason = 'hold'

                # 1. Stop loss check (intraday)
                if stop_loss_pct > 0 and lo < entry * (1 - stop_loss_pct / 100):
                    exit_price = entry * (1 - stop_loss_pct / 100)
                    exit_reason = 'sl'

                # 2. Profit target check (intraday) — only if not stopped
                if exit_price is None and profit_target_pct > 0 and hi > entry * (1 + profit_target_pct / 100):
                    exit_price = entry * (1 + profit_target_pct / 100)
                    exit_reason = 'pt'

                # 3. Trailing stop check — update high-water first
                if trailing_stop_pct > 0:
                    if not np.isnan(hi) and hi > p.get('trail_high', entry):
                        p['trail_high'] = hi
                    trail_high = p.get('trail_high', entry)
                    if exit_price is None and cp < trail_high * (1 - trailing_stop_pct / 100):
                        exit_price = cp  # exit at close for trailing stop
                        exit_reason = 'trail'

                # F-style loser quick exit (intraday)
                if exit_price is None and loser_exit_threshold > 0:
                    if lo < entry * (1 - loser_exit_threshold / 100):
                        exit_price = entry * (1 - loser_exit_threshold / 100)
                        exit_reason = 'loser_quick'

                # 4. Normal hold expiry
                if exit_price is None and di - p['entry_di'] >= p['hold_days']:
                    # Check winner extension (E-style)
                    if max_hold_ext > 0 and cp > entry:
                        days_held = di - p['entry_di']
                        if days_held < hold + max_hold_ext:
                            # Extend: don't exit yet
                            pass
                        else:
                            exit_price = cp
                            exit_reason = 'hold_max'
                    else:
                        exit_price = cp
                        exit_reason = 'hold'

                # F-style winner extension
                if exit_price is None and winner_ext_threshold > 0:
                    if hi > entry * (1 + winner_ext_threshold / 100):
                        # Mark for extension
                        if p.get('extended', False):
                            # Already extended once, hold expires normally
                            if di - p['entry_di'] >= p['hold_days']:
                                exit_price = cp
                                exit_reason = 'hold'
                        else:
                            p['extended'] = True
                            p['hold_days'] = max(p['hold_days'], di - p['entry_di'] + 2)
                    else:
                        # No extension, normal expiry
                        if di - p['entry_di'] >= p['hold_days']:
                            exit_price = cp
                            exit_reason = 'hold'

                if exit_price is not None:
                    if np.isnan(exit_price) or exit_price <= 0:
                        exit_price = entry  # fallback
                    pnl = (exit_price - entry) * m * p['lots']
                    pp = pnl / invested * 100 if invested > 0 else 0
                    cash += exit_price * m * abs(p['lots']) * (1 - COMM)
                    p['_exit_pnl_pct'] = pp
                    p['_exit_reason'] = exit_reason
                    cl.append(p)

            for p in cl:
                positions.remove(p)

            # Entry
            if len(positions) < top_n:
                edi = di + 1
                if edi < end_di:
                    cands = signal_func(di, edi)
                    if cands:
                        cands.sort(key=lambda x: -x[0])
                        ns = top_n - len(positions)
                        for item in cands[:ns]:
                            if len(item) == 3: sc, s, pr = item; sig = ''
                            else: sc, s, pr, sig = item
                            sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                            cap = cash * size_frac
                            ct = max(1, int(cap / (pr * m * (1 + COMM))))
                            ci = pr * m * ct * (1 + COMM)
                            if ci > cash:
                                ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                                ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                            if ct <= 0 or ci <= 0 or ci > cash: continue
                            cash -= ci
                            positions.append({
                                'si': s, 'entry_price': pr, 'entry_di': edi,
                                'lots': ct, 'dir': 1, 'sym': sym,
                                'hold_days': hold, 'sig': sig,
                                'trail_high': pr,
                                'extended': False,
                            })

        # Liquidate remaining
        for p in positions:
            ae = end_di - 1
            ep = C[p['si'], min(ae, ND-1)]
            if np.isnan(ep) or ep <= 0: ep = p['entry_price']
            m = MULT.get(p['sym'], DEF_MULT)
            cash += ep * m * abs(p['lots']) * (1 - COMM)

        # Compute metrics
        nd = end_di - start_di
        ann = annual_return(cash, CASH0, nd)
        if daily_eq:
            eq = np.array(daily_eq)
            pk = np.maximum.accumulate(eq)
            mdd = np.min((eq - pk) / pk * 100)
            r = np.diff(eq) / eq[:-1]
            r = np.where(np.isfinite(r), r, 0)
            sh = np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0
        else:
            mdd = 0; sh = 0

        ratio = abs(ann / mdd) if mdd != 0 else 0
        return {'ann': ann, 'mdd': mdd, 'sharpe': sh, 'final': cash,
                'ndays': nd, 'ratio': ratio}

    # ===================== PORTFOLIO BACKTEST =====================
    def backtest_portfolio(sig_A, sig_B, start_di=MIN_TRAIN, end_di=None, **kwargs):
        """Run two sub-strategies independently, combine 50/50."""
        if end_di is None: end_di = ND

        def run_sub(sig_func):
            r = backtest_exit(sig_func, start_di=start_di, end_di=end_di, **kwargs)
            # We need the daily equity curve, re-run to get it
            # Actually, let's modify to return equity curve
            pass

        # Simpler approach: run each, get daily equity
        def run_sub_eq(sig_func):
            if end_di is None: end_di_l = ND
            else: end_di_l = end_di
            cash = float(CASH0)
            positions = []
            daily_eq = []

            for di in range(start_di, end_di_l - 1):
                pv = cash
                for p in positions:
                    cp = C[p['si'], di]
                    if not np.isnan(cp) and cp > 0:
                        m = MULT.get(p['sym'], DEF_MULT)
                        pv += cp * m * p['lots'] - cp * m * abs(p['lots']) * COMM
                daily_eq.append(pv)

                cl = []
                for p in positions:
                    si = p['si']
                    entry = p['entry_price']
                    m = MULT.get(p['sym'], DEF_MULT)
                    invested = entry * m * abs(p['lots'])
                    lo = L[si, di]; hi = H[si, di]; cp = C[si, di]
                    if np.isnan(lo) or np.isnan(hi) or np.isnan(cp):
                        continue
                    exit_price = None
                    exit_reason = 'hold'

                    pt = kwargs.get('profit_target_pct', 0)
                    sl = kwargs.get('stop_loss_pct', 0)
                    ts = kwargs.get('trailing_stop_pct', 0)
                    mhe = kwargs.get('max_hold_ext', 0)
                    wet = kwargs.get('winner_ext_threshold', 0)
                    let = kwargs.get('loser_exit_threshold', 0)
                    hold_d = kwargs.get('hold', 1)
                    sf = kwargs.get('size_frac', SIZE_FRAC)

                    # Stop loss
                    if sl > 0 and lo < entry * (1 - sl / 100):
                        exit_price = entry * (1 - sl / 100); exit_reason = 'sl'
                    # Profit target
                    if exit_price is None and pt > 0 and hi > entry * (1 + pt / 100):
                        exit_price = entry * (1 + pt / 100); exit_reason = 'pt'
                    # Trailing stop
                    if ts > 0:
                        if hi > p.get('trail_high', entry):
                            p['trail_high'] = hi
                        th = p.get('trail_high', entry)
                        if exit_price is None and cp < th * (1 - ts / 100):
                            exit_price = cp; exit_reason = 'trail'
                    # Loser quick exit
                    if exit_price is None and let > 0 and lo < entry * (1 - let / 100):
                        exit_price = entry * (1 - let / 100); exit_reason = 'loser_quick'
                    # Hold expiry with extension
                    if exit_price is None and di - p['entry_di'] >= p['hold_days']:
                        if mhe > 0 and cp > entry:
                            days_held = di - p['entry_di']
                            if days_held < hold_d + mhe:
                                pass  # extend
                            else:
                                exit_price = cp; exit_reason = 'hold_max'
                        else:
                            exit_price = cp; exit_reason = 'hold'
                    # Winner extension (F-style)
                    if exit_price is None and wet > 0:
                        if hi > entry * (1 + wet / 100):
                            if not p.get('extended', False):
                                p['extended'] = True
                                p['hold_days'] = max(p['hold_days'], di - p['entry_di'] + 2)
                            else:
                                if di - p['entry_di'] >= p['hold_days']:
                                    exit_price = cp; exit_reason = 'hold'
                        else:
                            if di - p['entry_di'] >= p['hold_days']:
                                exit_price = cp; exit_reason = 'hold'

                    if exit_price is not None:
                        if np.isnan(exit_price) or exit_price <= 0:
                            exit_price = entry
                        pnl = (exit_price - entry) * m * p['lots']
                        cash += exit_price * m * abs(p['lots']) * (1 - COMM)
                        cl.append(p)

                for p in cl: positions.remove(p)

                # Entry
                sf = kwargs.get('size_frac', SIZE_FRAC)
                hold_d = kwargs.get('hold', 1)
                if len(positions) < 1:
                    edi = di + 1
                    if edi < end_di_l:
                        cands = sig_func(di, edi)
                        if cands:
                            cands.sort(key=lambda x: -x[0])
                            item = cands[0]
                            if len(item) == 3: sc, s, pr = item; sig = ''
                            else: sc, s, pr, sig = item
                            sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                            cap = cash * sf
                            ct = max(1, int(cap / (pr * m * (1 + COMM))))
                            ci = pr * m * ct * (1 + COMM)
                            if ci > cash:
                                ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                                ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                            if ct > 0 and ci > 0 and ci <= cash:
                                cash -= ci
                                positions.append({
                                    'si': s, 'entry_price': pr, 'entry_di': edi,
                                    'lots': ct, 'dir': 1, 'sym': sym,
                                    'hold_days': hold_d, 'sig': sig,
                                    'trail_high': pr, 'extended': False,
                                })

            for p in positions:
                ep = C[p['si'], min(end_di_l - 1, ND - 1)]
                if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                m = MULT.get(p['sym'], DEF_MULT)
                cash += ep * m * abs(p['lots']) * (1 - COMM)

            return np.array(daily_eq) if daily_eq else np.array([float(CASH0)])

        eq_A = run_sub_eq(sig_A)
        eq_B = run_sub_eq(sig_B)

        ml = min(len(eq_A), len(eq_B))
        if ml <= 1:
            return {'ann': -100.0, 'mdd': 0, 'sharpe': 0, 'final': CASH0, 'ratio': 0}

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
        ann = annual_return(final, CASH0, ml)
        pk = np.maximum.accumulate(eq)
        mdd = np.min((eq - pk) / pk * 100)
        sh = np.mean(combined) / np.std(combined) * np.sqrt(252) if np.std(combined) > 0 else 0
        ratio = abs(ann / mdd) if mdd != 0 else 0
        return {'ann': ann, 'mdd': mdd, 'sharpe': sh, 'final': final, 'ratio': ratio}

    # ===================== WALK-FORWARD HELPER =====================
    def walk_forward(sig_func, port=False, **kwargs):
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue
            if port:
                r = backtest_portfolio(sig_union, sig_v121, start_di=ys, end_di=ye, **kwargs)
            else:
                r = backtest_exit(sig_func, start_di=ys, end_di=ye, **kwargs)
            res[yr] = {'ann': r['ann'], 'mdd': r['mdd']}
        return res

    def pr(r, label=""):
        ratio = r.get('ratio', abs(r['ann'] / r['mdd']) if r.get('mdd', 0) != 0 else 0)
        print(f"  {label:75s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    def pr_wf(wf, label=""):
        yrs = sorted(wf.keys())
        anns = " | ".join([f"{yr}:{wf[yr]['ann']:+.0f}%" for yr in yrs])
        mdds = " | ".join([f"{yr}:{wf[yr]['mdd']:.0f}%" for yr in yrs])
        pos = sum(1 for yr in yrs if wf[yr]['ann'] > 0)
        avg = np.mean([wf[yr]['ann'] for yr in yrs])
        worst_mdd = min([wf[yr]['mdd'] for yr in yrs])
        print(f"  {label}")
        print(f"    WF Ann: {pos}/{len(yrs)} pos | Avg={avg:>+7.0f}% | {anns}")
        print(f"    WF MDD:                            Worst={worst_mdd:>.0f}% | {mdds}")

    # ===================== SECTION 0: BASELINE =====================
    print("\n" + "=" * 130)
    print("  SECTION 0: BASELINE (hold=1, 50% sizing, no exit enhancements)")
    print("=" * 130)

    r_v121_base = backtest_exit(sig_v121, hold=1, size_frac=SIZE_FRAC)
    pr(r_v121_base, "V121 baseline")
    r_union_base = backtest_exit(sig_union, hold=1, size_frac=SIZE_FRAC)
    pr(r_union_base, "Union baseline")
    r_port_base = backtest_portfolio(sig_union, sig_v121, hold=1, size_frac=SIZE_FRAC)
    pr(r_port_base, "Portfolio 50/50 baseline")

    base_ratio_port = r_port_base.get('ratio', 0)
    print(f"\n  Baseline portfolio Ann/MDD ratio = {base_ratio_port:.2f}")

    # ===================== SECTION A: PROFIT TARGET =====================
    print("\n" + "=" * 130)
    print("  SECTION A: PROFIT TARGET (intraday — exit if H > entry*(1+X%))")
    print("=" * 130)

    pt_levels = [1.0, 2.0, 3.0, 5.0]
    pt_results = {}

    for pt in pt_levels:
        print(f"\n  --- PT={pt}% ---")
        for sig_name, sig_func in [("V121", sig_v121), ("Union", sig_union)]:
            r = backtest_exit(sig_func, hold=1, profit_target_pct=pt, size_frac=SIZE_FRAC)
            key = f"PT={pt}%_{sig_name}"
            pt_results[key] = r
            pr(r, f"{sig_name} PT={pt}%")
        r = backtest_portfolio(sig_union, sig_v121, hold=1, profit_target_pct=pt, size_frac=SIZE_FRAC)
        key = f"PT={pt}%_Port"
        pt_results[key] = r
        pr(r, f"Port PT={pt}%")

    # ===================== SECTION B: STOP LOSS =====================
    print("\n" + "=" * 130)
    print("  SECTION B: STOP LOSS (intraday — exit if L < entry*(1-X%))")
    print("=" * 130)

    sl_levels = [2.0, 3.0, 5.0, 8.0, 10.0]
    sl_results = {}

    for sl in sl_levels:
        print(f"\n  --- SL={sl}% ---")
        for sig_name, sig_func in [("V121", sig_v121), ("Union", sig_union)]:
            r = backtest_exit(sig_func, hold=1, stop_loss_pct=sl, size_frac=SIZE_FRAC)
            key = f"SL={sl}%_{sig_name}"
            sl_results[key] = r
            pr(r, f"{sig_name} SL={sl}%")
        r = backtest_portfolio(sig_union, sig_v121, hold=1, stop_loss_pct=sl, size_frac=SIZE_FRAC)
        key = f"SL={sl}%_Port"
        sl_results[key] = r
        pr(r, f"Port SL={sl}%")

    # ===================== SECTION C: COMBINED PT + SL =====================
    print("\n" + "=" * 130)
    print("  SECTION C: COMBINED PROFIT TARGET + STOP LOSS")
    print("=" * 130)

    comb_configs = [
        (3.0, 5.0),
        (2.0, 3.0),
        (5.0, 8.0),
    ]
    comb_results = {}

    for pt, sl in comb_configs:
        print(f"\n  --- PT={pt}% / SL={sl}% ---")
        for sig_name, sig_func in [("V121", sig_v121), ("Union", sig_union)]:
            r = backtest_exit(sig_func, hold=1, profit_target_pct=pt, stop_loss_pct=sl, size_frac=SIZE_FRAC)
            key = f"PT={pt}%/SL={sl}%_{sig_name}"
            comb_results[key] = r
            pr(r, f"{sig_name} PT={pt}%/SL={sl}%")
        r = backtest_portfolio(sig_union, sig_v121, hold=1, profit_target_pct=pt, stop_loss_pct=sl, size_frac=SIZE_FRAC)
        key = f"PT={pt}%/SL={sl}%_Port"
        comb_results[key] = r
        pr(r, f"Port PT={pt}%/SL={sl}%")

    # ===================== SECTION D: TRAILING STOP =====================
    print("\n" + "=" * 130)
    print("  SECTION D: TRAILING STOP (daily bars — exit if C < high_since_entry*(1-X%))")
    print("=" * 130)

    ts_levels = [2.0, 3.0, 5.0]
    ts_results = {}

    for ts in ts_levels:
        print(f"\n  --- Trail={ts}% ---")
        for sig_name, sig_func in [("V121", sig_v121), ("Union", sig_union)]:
            r = backtest_exit(sig_func, hold=1, trailing_stop_pct=ts, size_frac=SIZE_FRAC)
            key = f"Trail={ts}%_{sig_name}"
            ts_results[key] = r
            pr(r, f"{sig_name} Trail={ts}%")
        r = backtest_portfolio(sig_union, sig_v121, hold=1, trailing_stop_pct=ts, size_frac=SIZE_FRAC)
        key = f"Trail={ts}%_Port"
        ts_results[key] = r
        pr(r, f"Port Trail={ts}%")

    # ===================== SECTION E: CONDITIONAL HOLD EXTENSION =====================
    print("\n" + "=" * 130)
    print("  SECTION E: CONDITIONAL HOLD EXTENSION (winners extend, max 3 days)")
    print("=" * 130)

    # max_hold_ext=2 means hold can go from 1 to 1+2=3 days max
    ext_results = {}

    for mhe in [1, 2]:  # extend by 1 or 2 days
        print(f"\n  --- Max extension=+{mhe} days ---")
        for sig_name, sig_func in [("V121", sig_v121), ("Union", sig_union)]:
            r = backtest_exit(sig_func, hold=1, max_hold_ext=mhe, size_frac=SIZE_FRAC)
            key = f"Ext+{mhe}_{sig_name}"
            ext_results[key] = r
            pr(r, f"{sig_name} Hold=1+ext{mhe}")
        r = backtest_portfolio(sig_union, sig_v121, hold=1, max_hold_ext=mhe, size_frac=SIZE_FRAC)
        key = f"Ext+{mhe}_Port"
        ext_results[key] = r
        pr(r, f"Port Hold=1+ext{mhe}")

    # ===================== SECTION F: WINNER EXT + LOSER QUICK EXIT =====================
    print("\n" + "=" * 130)
    print("  SECTION F: WINNER EXTENSION (up>1% intraday -> hold 2d) + LOSER QUICK EXIT (down>3% -> exit)")
    print("=" * 130)

    f_configs = [
        (1.0, 3.0),   # as specified
        (1.0, 5.0),
        (2.0, 3.0),
        (2.0, 5.0),
    ]
    f_results = {}

    for wet, let in f_configs:
        print(f"\n  --- WinExt>{wet}% / LoserExit<{let}% ---")
        for sig_name, sig_func in [("V121", sig_v121), ("Union", sig_union)]:
            r = backtest_exit(sig_func, hold=1, winner_ext_threshold=wet,
                              loser_exit_threshold=let, size_frac=SIZE_FRAC)
            key = f"WE={wet}/LE={let}_{sig_name}"
            f_results[key] = r
            pr(r, f"{sig_name} WE>{wet}%/LE<{let}%")
        r = backtest_portfolio(sig_union, sig_v121, hold=1, winner_ext_threshold=wet,
                               loser_exit_threshold=let, size_frac=SIZE_FRAC)
        key = f"WE={wet}/LE={let}_Port"
        f_results[key] = r
        pr(r, f"Port WE>{wet}%/LE<{let}%")

    # ===================== SECTION G: BEST COMBOS FROM A-F =====================
    print("\n" + "=" * 130)
    print("  SECTION G: BEST COMBOS (promising from above, combined)")
    print("=" * 130)

    combo_configs = [
        # (label, kwargs)
        ("PT=2% + SL=5% + Trail=3%", dict(profit_target_pct=2.0, stop_loss_pct=5.0, trailing_stop_pct=3.0)),
        ("PT=3% + SL=5% + Trail=5%", dict(profit_target_pct=3.0, stop_loss_pct=5.0, trailing_stop_pct=5.0)),
        ("PT=2% + SL=3%", dict(profit_target_pct=2.0, stop_loss_pct=3.0)),
        ("PT=3% + SL=5%", dict(profit_target_pct=3.0, stop_loss_pct=5.0)),
        ("PT=3% + Trail=3%", dict(profit_target_pct=3.0, trailing_stop_pct=3.0)),
        ("SL=5% + Trail=3%", dict(stop_loss_pct=5.0, trailing_stop_pct=3.0)),
        ("PT=2% + SL=3% + WE>1%/LE<3%", dict(profit_target_pct=2.0, stop_loss_pct=3.0,
                                              winner_ext_threshold=1.0, loser_exit_threshold=3.0)),
        ("PT=3% + SL=5% + WE>1%/LE<3%", dict(profit_target_pct=3.0, stop_loss_pct=5.0,
                                              winner_ext_threshold=1.0, loser_exit_threshold=3.0)),
    ]
    combo_results = {}

    for label, kwargs in combo_configs:
        print(f"\n  --- {label} ---")
        r = backtest_portfolio(sig_union, sig_v121, hold=1, size_frac=SIZE_FRAC, **kwargs)
        combo_results[label] = r
        pr(r, f"Port {label}")

    # ===================== WALK-FORWARD FOR TOP CONFIGS =====================
    print("\n" + "=" * 130)
    print("  WALK-FORWARD VALIDATION FOR TOP CONFIGS")
    print("=" * 130)

    # Collect all portfolio results with their labels
    all_port = {}
    all_port['BASELINE'] = r_port_base
    for k, v in pt_results.items():
        if k.endswith('_Port'): all_port[k] = v
    for k, v in sl_results.items():
        if k.endswith('_Port'): all_port[k] = v
    for k, v in comb_results.items():
        if k.endswith('_Port'): all_port[k] = v
    for k, v in ts_results.items():
        if k.endswith('_Port'): all_port[k] = v
    for k, v in ext_results.items():
        if k.endswith('_Port'): all_port[k] = v
    for k, v in f_results.items():
        if k.endswith('_Port'): all_port[k] = v
    for k, v in combo_results.items():
        all_port[k] = v

    # Sort by ratio
    ranked = sorted(all_port.items(), key=lambda x: -x[1].get('ratio', 0))

    print(f"\n  --- TOP 15 by Ann/MDD ratio (full period) ---")
    print(f"  {'Config':75s} | {'Ann':>8s} | {'MDD':>6s} | {'Sh':>4s} | {'R/M':>5s}")
    print(f"  {'-'*75}-+-{'-'*8}-+-{'-'*6}-+-{'-'*4}-+-{'-'*5}")
    for i, (label, r) in enumerate(ranked[:15]):
        print(f"  {label:75s} | {r['ann']:+8.1f}% | {r['mdd']:6.1f}% | {r['sharpe']:4.2f} | {r.get('ratio',0):.2f}")

    # Determine which configs had actual kwargs to pass
    def get_kwargs_for(label):
        """Map label back to kwargs for walk-forward."""
        if label == 'BASELINE':
            return {}
        for pt in pt_levels:
            if label == f'PT={pt}%_Port':
                return {'profit_target_pct': pt}
        for sl in sl_levels:
            if label == f'SL={sl}%_Port':
                return {'stop_loss_pct': sl}
        for pt, sl in comb_configs:
            if label == f'PT={pt}%/SL={sl}%_Port':
                return {'profit_target_pct': pt, 'stop_loss_pct': sl}
        for ts in ts_levels:
            if label == f'Trail={ts}%_Port':
                return {'trailing_stop_pct': ts}
        for mhe in [1, 2]:
            if label == f'Ext+{mhe}_Port':
                return {'max_hold_ext': mhe}
        for wet, let in f_configs:
            if label == f'WE={wet}/LE={let}_Port':
                return {'winner_ext_threshold': wet, 'loser_exit_threshold': let}
        # combo configs
        for cl, ck in combo_configs:
            if label == cl:
                return ck
        return None

    # WF for top 10
    print(f"\n  Walk-Forward for top 10 configs:")
    for i, (label, r) in enumerate(ranked[:10]):
        kwargs = get_kwargs_for(label)
        if kwargs is None:
            print(f"  #{i+1}: {label} — skip (no kwargs mapping)")
            continue
        wf = walk_forward(None, port=True, **kwargs)
        pos = sum(1 for v in wf.values() if v['ann'] > 0)
        avg = np.mean([v['ann'] for v in wf.values()])
        worst_mdd = min([v['mdd'] for v in wf.values()])
        anns = " | ".join([f"{yr}:{wf[yr]['ann']:+.0f}%" for yr in sorted(wf.keys())])
        mdds = " | ".join([f"{yr}:{wf[yr]['mdd']:.0f}%" for yr in sorted(wf.keys())])
        print(f"  #{i+1}: {label}")
        print(f"    Ann: {pos}/6 pos | Avg={avg:>+7.0f}% | {anns}")
        print(f"    MDD: Worst={worst_mdd:>.0f}% | {mdds}")

    # WF for baseline
    print(f"\n  Baseline WF:")
    wf_base = walk_forward(None, port=True)
    pos = sum(1 for v in wf_base.values() if v['ann'] > 0)
    avg = np.mean([v['ann'] for v in wf_base.values()])
    worst_mdd = min([v['mdd'] for v in wf_base.values()])
    anns = " | ".join([f"{yr}:{wf_base[yr]['ann']:+.0f}%" for yr in sorted(wf_base.keys())])
    mdds = " | ".join([f"{yr}:{wf_base[yr]['mdd']:.0f}%" for yr in sorted(wf_base.keys())])
    print(f"    Ann: {pos}/6 pos | Avg={avg:>+7.0f}% | {anns}")
    print(f"    MDD: Worst={worst_mdd:>.0f}% | {mdds}")

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 130)
    print("  FINAL SUMMARY")
    print("=" * 130)

    print(f"\n  BASELINE (50/50 Port, hold=1, 50% sizing):")
    pr(r_port_base, "Baseline")

    # Find best configs that BEAT baseline on ratio
    improvements = [(label, r) for label, r in ranked
                    if r.get('ratio', 0) > base_ratio_port and label != 'BASELINE']
    improvements.sort(key=lambda x: -x[1].get('ratio', 0))

    print(f"\n  CONFIGS THAT IMPROVE Ann/MDD RATIO vs baseline ({base_ratio_port:.2f}):")
    if improvements:
        for i, (label, r) in enumerate(improvements[:15]):
            delta = r.get('ratio', 0) - base_ratio_port
            print(f"  #{i+1}: {label:75s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | R/M={r.get('ratio',0):.2f} (+{delta:.2f})")
    else:
        print("  None found — baseline is hard to beat!")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
