"""
Alpha Futures V27 — v14b Engine with Configurable Stops
=======================================================
Testing: what if we reduce/eliminate stop losses?
v14b stops cost -1445.8% cumulative → if avoided, could boost from +73% to 200%+
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

MULT = {
    'agfi': 15, 'alfi': 5, 'aufi': 1000, 'bufi': 10, 'cufi': 5,
    'fufi': 10, 'rbfi': 10, 'znfi': 5, 'nifi': 1, 'hcfi': 10,
    'spfi': 10, 'ssfi': 5, 'sffi': 5, 'smfi': 5, 'pbfi': 5,
    'snfi': 1, 'rufi': 10, 'wrffi': 10,
    'afi': 10, 'bfi': 10, 'bbfi': 500, 'cffi': 5, 'cfi': 10,
    'csfi': 10, 'ebfi': 5, 'egfi': 10, 'fbfi': 500,
    'ifi': 100, 'jfi': 100, 'jmfi': 60, 'lfi': 5, 'mfi': 10,
    'pgfi': 20, 'ppfi': 5, 'vfi': 5, 'yfi': 10, 'pfi': 10,
    'jdfi': 5, 'lhfi': 16, 'pkfi': 5, 'rrfi': 20, 'lrfi': 20,
    'jrfi': 20, 'pmfi': 20, 'whfi': 20, 'rsfi': 20, 'cjfi': 10,
    'mafi': 10, 'apfi': 10, 'cyfi': 5, 'fgfi': 20, 'oifi': 10,
    'pfifi': 5, 'rmfi': 10, 'srfi': 10, 'tafi': 5, 'safi': 20,
    'urfi': 20, 'scfi': 1000, 'lufi': 10, 'bcfi': 5, 'nrfi': 1,
    'lgfi': 20, 'brfi': 5, 'lcfi': 1, 'sifi': 5,
    'ni': 1, 'tai': 5,
}
DEF_MULT = 10
COMM = 0.0003


def main():
    t_start = time.time()
    print("=" * 110)
    print("Alpha Futures V27 — v14b Engine + Stop Variants")
    print("=" * 110)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    # Precompute v14b-style signals
    print("[Signals]...", flush=True)
    t0 = time.time()

    mom5 = np.full((NS, ND), np.nan)
    vdp_ema = np.full((NS, ND), np.nan)
    oi_mom5 = np.full((NS, ND), np.nan)

    for si in range(NS):
        vdp_e = 0.0
        for di in range(20, ND):
            d = di - 1
            c_now = C[si, d]
            if np.isnan(c_now) or c_now <= 0: continue

            # Momentum 5
            c_prev = C[si, max(0, d - 5)]
            if not np.isnan(c_prev) and c_prev > 0:
                mom5[si, di] = (c_now - c_prev) / c_prev

            # OI momentum
            oi_now = OI[si, d]
            if not np.isnan(oi_now) and oi_now > 0:
                oi5 = OI[si, max(0, d-4)]
                if not np.isnan(oi5) and oi5 > 0:
                    oi_mom5[si, di] = (oi_now - oi5) / oi5

            # VDP EMA
            hl = H[si, d] - L[si, d]
            if not np.isnan(hl) and hl > 0:
                cd, hd, ld, vd = C[si, d], H[si, d], L[si, d], V[si, d]
                if not any(np.isnan([cd, hd, ld, vd])):
                    vdp_val = vd * (2*cd - hd - ld) / hl
                    alpha = 2.0 / 15
                    vdp_e = alpha * vdp_val + (1 - alpha) * vdp_e
                    vdp_ema[si, di] = vdp_e

    print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

    # v14b-style scoring
    def score_vdp_mom(si, di):
        m5 = mom5[si, di]
        vd = vdp_ema[si, di]
        if np.isnan(m5): return np.nan
        score = np.clip(m5 * 8, -1, 1)
        if not np.isnan(vd):
            if (m5 > 0 and vd > 0) or (m5 < 0 and vd < 0):
                score *= 1.3
            else:
                score *= 0.3
        return score

    # Backtest engine (v14b's run_swing logic)
    def run_swing(score_fn, name, hold_max=3, trail_atr=3.0, stop_loss=0.05,
                  allow_short=False, use_stop=True, use_trail=True):
        cash = float(CASH0)
        trades = []
        pos = None
        last_exit = {}

        for di in range(MIN_TRAIN, ND):
            year = dates[di].year

            if pos is not None:
                c = C[pos['si'], di]
                if np.isnan(c) or c <= 0: c = pos['entry']
                mult = MULT.get(pos['sym'], DEF_MULT)
                mkt_val = c * mult * pos['lots']
                pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
                pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] > 0 else 0
                days_held = di - pos['entry_di']

                exit_reason = None

                # 1. Stop loss (configurable)
                if use_stop and pnl_pct / 100 < -stop_loss:
                    exit_reason = 'stop'

                # 2. Trailing stop
                if exit_reason is None and use_trail and trail_atr > 0:
                    atr = pos.get('atr', 0)
                    if atr > 0:
                        trail_price = pos.get('trail_price', pos['entry'])
                        if pos['dir'] == 1:
                            new_trail = c - trail_atr * atr
                            if new_trail > trail_price: pos['trail_price'] = new_trail
                            if c < trail_price and days_held >= 2: exit_reason = 'trail'
                        else:
                            new_trail = c + trail_atr * atr
                            if new_trail < trail_price: pos['trail_price'] = new_trail
                            if c > trail_price and days_held >= 2: exit_reason = 'trail'

                # 3. Score exit
                if exit_reason is None and days_held >= 2:
                    cur_score = score_fn(pos['si'], di)
                    if not np.isnan(cur_score):
                        if pos['dir'] == 1 and cur_score < -0.02: exit_reason = 'signal_flip'
                        elif pos['dir'] == -1 and cur_score > 0.02: exit_reason = 'signal_flip'

                # 4. Time exit
                if exit_reason is None and days_held >= hold_max:
                    exit_reason = 'time'

                # 5. Rotation (50%+ better candidate)
                if exit_reason is None and days_held >= 2:
                    best_si, best_dir, best_sc = -1, 0, 0
                    for sj in range(NS):
                        sc = score_fn(sj, di)
                        if np.isnan(sc): continue
                        if sc > best_sc: best_sc = sc; best_si = sj; best_dir = 1
                        if allow_short and -sc > best_sc: best_sc = -sc; best_si = sj; best_dir = -1
                    cur_sc = abs(score_fn(pos['si'], di)) if not np.isnan(score_fn(pos['si'], di)) else 0
                    if best_sc > cur_sc * 1.5 + 0.05 and best_si != pos['si']:
                        exit_reason = 'rotate'

                if exit_reason:
                    cost_out = mkt_val * COMM
                    cash += mkt_val - cost_out
                    trades.append({
                        'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                        'days': days_held, 'di': di, 'year': year,
                        'sym': pos['sym'], 'dir': pos['dir'], 'reason': exit_reason
                    })
                    last_exit[pos['sym']] = di
                    pos = None

            if pos is None:
                best_si, best_dir, best_sc = -1, 0, 0
                for si in range(NS):
                    sc = score_fn(si, di)
                    if np.isnan(sc): continue
                    sym = syms[si]
                    if sym in last_exit and di - last_exit[sym] < 0: continue
                    if sc > best_sc: best_sc = sc; best_si = si; best_dir = 1
                    if allow_short and -sc > best_sc: best_sc = -sc; best_si = si; best_dir = -1

                if best_si >= 0 and best_sc > 0:
                    c = C[best_si, di]
                    if np.isnan(c) or c <= 0: continue
                    sym = syms[best_si]
                    mult = MULT.get(sym, DEF_MULT)
                    notional = c * mult
                    if notional <= 0: continue
                    lots = int(cash / notional)
                    if lots <= 0: continue
                    cost_in = notional * lots * (1 + COMM)
                    if cost_in > cash: continue

                    atr_val = 0; trs = []
                    for dd in range(max(1, di-10), di+1):
                        hi, lo, pc = H[best_si, dd], L[best_si, dd], C[best_si, dd-1]
                        if np.isnan(hi) or np.isnan(lo): continue
                        tr = hi - lo
                        if not np.isnan(pc): tr = max(tr, abs(hi-pc), abs(lo-pc))
                        trs.append(tr)
                    if trs: atr_val = np.mean(trs)

                    cash -= cost_in
                    trail_price = c - trail_atr * atr_val if best_dir == 1 else c + trail_atr * atr_val
                    pos = {
                        'si': best_si, 'entry': c, 'entry_di': di,
                        'lots': lots, 'dir': best_dir, 'sym': sym,
                        'atr': atr_val, 'trail_price': trail_price
                    }

        if pos is not None:
            c = C[pos['si'], ND-1]
            if np.isnan(c) or c <= 0: c = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
            cash += c * mult * pos['lots'] * (1 - COMM)
            trades.append({
                'pnl_pct': pnl / (pos['entry'] * mult * pos['lots']) * 100,
                'pnl_abs': pnl, 'days': ND-1 - pos['entry_di'],
                'di': ND-1, 'year': dates[ND-1].year,
                'sym': pos['sym'], 'dir': pos['dir'], 'reason': 'end'
            })

        if len(trades) < 20: return None

        equity = float(CASH0); peak = float(CASH0); max_dd = 0
        for t in sorted(trades, key=lambda x: x['di']):
            equity += t['pnl_abs']
            if equity > peak: peak = equity
            if peak > 0:
                dd = (peak - equity) / peak * 100
                if dd > max_dd: max_dd = dd

        days_total = (dates[ND-1] - dates[MIN_TRAIN]).days
        yr = max(days_total / 365.25, 0.01)
        ann = ((cash / CASH0) ** (1 / yr) - 1) * 100

        nw = sum(1 for t in trades if t['pnl_abs'] > 0)
        wr = nw / len(trades) * 100
        avg_pnl = np.mean([t['pnl_pct'] for t in trades])
        avg_days = np.mean([t['days'] for t in trades])
        avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_abs'] > 0]) if nw > 0 else 0
        avg_loss = np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_abs'] <= 0]) if nw < len(trades) else 0

        reasons = {}
        for t in trades:
            r = t['reason']
            if r not in reasons: reasons[r] = {'n': 0, 'w': 0, 'pnl': 0.0}
            reasons[r]['n'] += 1
            if t['pnl_abs'] > 0: reasons[r]['w'] += 1
            reasons[r]['pnl'] += t['pnl_pct']

        year_stats = {}
        for t in trades:
            y = t['year']
            if y not in year_stats: year_stats[y] = {'n': 0, 'w': 0, 'pnl': 0.0}
            year_stats[y]['n'] += 1
            if t['pnl_abs'] > 0: year_stats[y]['w'] += 1
            year_stats[y]['pnl'] += t['pnl_pct']

        return {
            'name': name, 'ann': round(ann, 1), 'n': len(trades),
            'wr': round(wr, 1), 'dd': round(max_dd, 1),
            'avg_pnl': round(avg_pnl, 3), 'avg_days': round(avg_days, 1),
            'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
            'cash': round(cash, 0), 'reasons': reasons, 'yearly': year_stats,
        }

    # Run configs
    results = []
    configs = [
        # (name, hold_max, trail_atr, stop_loss, use_stop, use_trail)
        ("H3_T3.0_S0.05_full", 3, 3.0, 0.05, True, True),      # v14b baseline
        ("H3_T3.0_NO_STOP", 3, 3.0, 0.00, False, True),        # no stop
        ("H3_T3.0_NO_TRAIL", 3, 0.0, 0.05, True, False),       # no trail
        ("H3_NO_STOP_NO_TRAIL", 3, 0.0, 0.00, False, False),   # neither
        ("H3_T2.0_S0.05", 3, 2.0, 0.05, True, True),
        ("H3_T2.0_NO_STOP", 3, 2.0, 0.00, False, True),
        ("H3_T1.5_S0.03", 3, 1.5, 0.03, True, True),
        ("H3_T1.5_NO_STOP", 3, 1.5, 0.00, False, True),
        ("H5_T2.0_S0.05", 5, 2.0, 0.05, True, True),
        ("H5_T2.0_NO_STOP", 5, 2.0, 0.00, False, True),
        ("H5_NO_STOP_NO_TRAIL", 5, 0.0, 0.00, False, False),
        ("H5_T3.0_S0.05", 5, 3.0, 0.05, True, True),
        ("H5_T3.0_NO_STOP", 5, 3.0, 0.00, False, True),
        ("H7_T2.0_S0.05", 7, 2.0, 0.05, True, True),
        ("H7_T2.0_NO_STOP", 7, 2.0, 0.00, False, True),
        ("H7_NO_STOP_NO_TRAIL", 7, 0.0, 0.00, False, False),
        ("H7_T3.0_S0.05", 7, 3.0, 0.05, True, True),
        ("H7_T3.0_NO_STOP", 7, 3.0, 0.00, False, True),
        # Wider stops
        ("H3_T3.0_S0.08", 3, 3.0, 0.08, True, True),
        ("H3_T3.0_S0.10", 3, 3.0, 0.10, True, True),
        ("H5_T3.0_S0.08", 5, 3.0, 0.08, True, True),
        ("H5_T3.0_S0.10", 5, 3.0, 0.10, True, True),
        # Squeeze variants - only trail after profit
        ("H3_T3.0_S0.05_TIGHT", 3, 3.0, 0.05, True, True),
        ("H3_NO_STOP_TIME_ONLY", 3, 0.0, 0.00, False, False),
    ]

    for name, hm, ta, sl, us, ut in configs:
        r = run_swing(score_vdp_mom, f"VDP_{name}",
                      hold_max=hm, trail_atr=ta, stop_loss=sl,
                      use_stop=us, use_trail=ut, allow_short=True)
        if r:
            results.append(r)
            print(f"  {r['name']:40s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"N {r['n']:4d} | DD {r['dd']:6.1f}% | AvgW {r['avg_win']:+.2f}% | "
                  f"AvgL {r['avg_loss']:.2f}% | AvgD {r['avg_days']:.1f}")
            # Exit breakdown
            parts = []
            for reason, stats in sorted(r['reasons'].items()):
                wr = stats['w'] / stats['n'] * 100 if stats['n'] > 0 else 0
                parts.append(f"{reason}:{stats['n']}({wr:.0f}%)pnl={stats['pnl']:+.0f}%")
            print(f"  {'':40s} | {' | '.join(parts)}")

    # Yearly for top configs
    results.sort(key=lambda x: -x['ann'])
    print(f"\n--- YEARLY BREAKDOWN (Top 5) ---")
    for r in results[:5]:
        print(f"\n  {r['name']}:")
        for y in sorted(r['yearly'].keys()):
            ys = r['yearly'][y]
            wr = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
            print(f"    {y}: {ys['n']:3d}t WR {wr:5.1f}% PnL {ys['pnl']:+.1f}%")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
