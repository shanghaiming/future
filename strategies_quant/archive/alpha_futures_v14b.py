"""
Alpha Futures V14b — Swing Rotation with Trailing Stop (无杠杆, 纯日线)
======================================================================
核心: 3-7天持仓, P1集中, trailing stop创造非对称收益
数学: 80次交易/年 × 2.5%平均收益 = 600%年化

策略:
  1. 每天/每N天扫描68个品种, 选最强信号品种
  2. 全仓买入, trailing stop锁定利润
  3. 信号确认: 动量 + OI + VDP + 量能共振
  4. 严格过滤: 只在高确信度时入场

约束: 不做gap, 不做日内, 无杠杆, 持仓2-7天
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


def run_swing(NS, ND, dates, C, O, H, L, V, OI, syms,
              score_fn, name,
              hold_max=5, trail_atr=2.0, stop_loss=0.03,
              allow_short=False, reentry_gap=0):
    """Swing rotation with trailing stop.
    hold_max: max holding days
    trail_atr: trailing stop = price - trail_atr * ATR (long)
    stop_loss: fixed % stop loss
    reentry_gap: min days before re-entering same symbol
    """
    cash = float(CASH0)
    trades = []
    pos = None
    last_exit = {}  # sym -> exit_di

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # === MANAGE POSITION ===
        if pos is not None:
            c = C[pos['si'], di]
            if np.isnan(c) or c <= 0:
                c = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = c * mult * pos['lots']
            pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
            pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] > 0 else 0
            days_held = di - pos['entry_di']

            exit_reason = None

            # 1. Fixed stop loss
            if pnl_pct / 100 < -stop_loss:
                exit_reason = 'stop'

            # 2. Trailing stop
            if exit_reason is None and trail_atr > 0:
                atr = pos.get('atr', 0)
                if atr > 0:
                    trail_price = pos.get('trail_price', pos['entry'])
                    if pos['dir'] == 1:
                        new_trail = c - trail_atr * atr
                        if new_trail > trail_price:
                            pos['trail_price'] = new_trail
                        if c < trail_price and days_held >= 2:
                            exit_reason = 'trail'
                    else:
                        new_trail = c + trail_atr * atr
                        if new_trail < trail_price:
                            pos['trail_price'] = new_trail
                        if c > trail_price and days_held >= 2:
                            exit_reason = 'trail'

            # 3. Score exit: if score drops below threshold, exit
            if exit_reason is None and days_held >= 2:
                cur_score = score_fn(pos['si'], di)
                if not np.isnan(cur_score):
                    # If direction flipped or score very weak
                    if pos['dir'] == 1 and cur_score < -0.02:
                        exit_reason = 'signal_flip'
                    elif pos['dir'] == -1 and cur_score > 0.02:
                        exit_reason = 'signal_flip'

            # 4. Time exit
            if exit_reason is None and days_held >= hold_max:
                exit_reason = 'time'

            # 5. Better candidate (rotate)
            if exit_reason is None and days_held >= 2:
                best_si, best_dir, best_sc = -1, 0, 0
                for sj in range(NS):
                    sc = score_fn(sj, di)
                    if np.isnan(sc): continue
                    if sc > best_sc:
                        best_sc = sc; best_si = sj; best_dir = 1
                    if allow_short and -sc > best_sc:
                        best_sc = -sc; best_si = sj; best_dir = -1

                cur_sc = abs(score_fn(pos['si'], di)) if not np.isnan(score_fn(pos['si'], di)) else 0
                # Rotate if new candidate is 50%+ better
                if best_sc > cur_sc * 1.5 + 0.05 and best_si != pos['si']:
                    exit_reason = 'rotate'

            if exit_reason:
                cost_out = mkt_val * COMM
                cash += mkt_val - cost_out
                trades.append({
                    'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                    'days': days_held, 'di': di, 'year': year,
                    'sym': pos['sym'], 'dir': pos['dir'],
                    'reason': exit_reason
                })
                last_exit[pos['sym']] = di
                pos = None

        # === ENTRY ===
        if pos is None:
            best_si, best_dir, best_sc = -1, 0, 0
            for si in range(NS):
                sc = score_fn(si, di)
                if np.isnan(sc): continue

                sym = syms[si]
                # Reentry gap
                if sym in last_exit and di - last_exit[sym] < reentry_gap:
                    continue

                if sc > best_sc:
                    best_sc = sc; best_si = si; best_dir = 1
                if allow_short and -sc > best_sc:
                    best_sc = -sc; best_si = si; best_dir = -1

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

                # Get ATR for trailing stop
                atr_val = 0
                trs = []
                for dd in range(max(1, di-10), di+1):
                    hi = H[best_si, dd]; lo = L[best_si, dd]; pc = C[best_si, dd-1]
                    if np.isnan(hi) or np.isnan(lo): continue
                    tr = hi - lo
                    if not np.isnan(pc):
                        tr = max(tr, abs(hi-pc), abs(lo-pc))
                    trs.append(tr)
                if trs:
                    atr_val = np.mean(trs)

                cash -= cost_in
                trail_price = c - trail_atr * atr_val if best_dir == 1 else c + trail_atr * atr_val
                pos = {
                    'si': best_si, 'entry': c, 'entry_di': di,
                    'lots': lots, 'dir': best_dir, 'sym': sym,
                    'atr': atr_val, 'trail_price': trail_price
                }

    # Close remaining
    if pos is not None:
        c = C[pos['si'], ND-1]
        if np.isnan(c) or c <= 0: c = pos['entry']
        mult = MULT.get(pos['sym'], DEF_MULT)
        pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
        cash += c * mult * pos['lots'] * (1 - COMM)
        trades.append({
            'pnl_pct': pnl / (pos['entry'] * mult * pos['lots']) * 100,
            'pnl_abs': pnl, 'days': di - pos['entry_di'],
            'di': ND-1, 'year': dates[ND-1].year,
            'sym': pos['sym'], 'dir': pos['dir'], 'reason': 'end'
        })

    if len(trades) < 20:
        return None

    # Stats
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

    year_stats = {}
    for t in trades:
        y = t['year']
        if y not in year_stats:
            year_stats[y] = {'n': 0, 'w': 0, 'pnl': 0.0}
        year_stats[y]['n'] += 1
        if t['pnl_abs'] > 0: year_stats[y]['w'] += 1
        year_stats[y]['pnl'] += t['pnl_pct']

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        r = t['reason']
        if r not in reasons:
            reasons[r] = {'n': 0, 'w': 0, 'pnl': 0.0}
        reasons[r]['n'] += 1
        if t['pnl_abs'] > 0: reasons[r]['w'] += 1
        reasons[r]['pnl'] += t['pnl_pct']

    return {
        'name': name, 'ann': round(ann, 1), 'n': len(trades),
        'wr': round(wr, 1), 'dd': round(max_dd, 1),
        'avg_pnl': round(avg_pnl, 3), 'avg_days': round(avg_days, 1),
        'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
        'final': round(cash, 0), 'years': year_stats, 'reasons': reasons
    }


if __name__ == '__main__':
    print("=" * 95, flush=True)
    print("  Alpha Futures V14b — Swing Rotation with Trailing Stop (纯日线, 无杠杆, 2-7天持仓)", flush=True)
    print("=" * 95, flush=True)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    print("\n  预计算因子...", flush=True)
    t0 = time.time()

    # Pre-compute factors
    mom3 = np.full((NS, ND), np.nan)
    mom5 = np.full((NS, ND), np.nan)
    mom10 = np.full((NS, ND), np.nan)
    oi_mom5 = np.full((NS, ND), np.nan)
    vdp_ema = np.full((NS, ND), np.nan)
    vol_ratio = np.full((NS, ND), np.nan)
    atr10 = np.full((NS, ND), np.nan)
    body_r = np.full((NS, ND), np.nan)
    # Donchian 20-day high/low position (0=at low, 1=at high)
    donch_pos = np.full((NS, ND), np.nan)
    # OI price divergence score
    oi_price_div = np.full((NS, ND), np.nan)

    for si in range(NS):
        vdp_e = 0.0
        for di in range(20, ND):
            d = di - 1
            c_now = C[si, d]
            if np.isnan(c_now) or c_now <= 0:
                continue

            # Momentum
            for lag, arr in [(3, mom3), (5, mom5), (10, mom10)]:
                c_prev = C[si, max(0, d - lag)]
                if not np.isnan(c_prev) and c_prev > 0:
                    arr[si, di] = (c_now - c_prev) / c_prev

            # OI momentum
            oi_now = OI[si, d]
            if not np.isnan(oi_now) and oi_now > 0:
                oi5 = OI[si, max(0, d-4)]
                if not np.isnan(oi5) and oi5 > 0:
                    oi_mom5[si, di] = (oi_now - oi5) / oi5

                # OI-Price divergence
                m3 = mom3[si, di]
                om = oi_mom5[si, di]
                if not np.isnan(m3) and not np.isnan(om):
                    # Both up = +1 (bullish), Both down = -1 (bearish), Diverge = 0
                    if m3 > 0.01 and om > 0.05:
                        oi_price_div[si, di] = min(m3 * 5, 1) * min(om * 2, 1)
                    elif m3 < -0.01 and om < -0.05:
                        oi_price_div[si, di] = -min(abs(m3) * 5, 1) * min(abs(om) * 2, 1)

            # VDP EMA
            hl = H[si, d] - L[si, d]
            if not np.isnan(hl) and hl > 0:
                cd = C[si, d]; hd = H[si, d]; ld = L[si, d]; vd = V[si, d]
                if not any(np.isnan([cd, hd, ld, vd])):
                    vdp_val = vd * (2*cd - hd - ld) / hl
                    alpha = 2.0 / 15
                    vdp_e = alpha * vdp_val + (1 - alpha) * vdp_e
                    vdp_ema[si, di] = vdp_e

            # Body ratio
            if not np.isnan(hl) and hl > 0:
                co = c_now - O[si, d]
                if not np.isnan(co):
                    body_r[si, di] = co / hl

            # Volume ratio
            v_now = V[si, d]
            if not np.isnan(v_now) and v_now > 0:
                v20 = V[si, max(0, d-19):d+1]
                v20v = v20[~np.isnan(v20)]
                if len(v20v) >= 10:
                    vol_ratio[si, di] = v_now / np.mean(v20v)

            # ATR
            trs = []
            for dd in range(max(1, d-9), d+1):
                hi = H[si, dd]; lo = L[si, dd]; pc = C[si, dd-1]
                if np.isnan(hi) or np.isnan(lo): continue
                tr = hi - lo
                if not np.isnan(pc):
                    tr = max(tr, abs(hi-pc), abs(lo-pc))
                trs.append(tr)
            if trs: atr10[si, di] = np.mean(trs)

            # Donchian position
            if di >= 20:
                h20 = H[si, max(0, d-19):d+1]
                l20 = L[si, max(0, d-19):d+1]
                h20v = h20[~np.isnan(h20)]
                l20v = l20[~np.isnan(l20)]
                if len(h20v) > 0 and len(l20v) > 0:
                    hh = np.max(h20v)
                    ll = np.min(l20v)
                    rng = hh - ll
                    if rng > 0:
                        donch_pos[si, di] = (c_now - ll) / rng  # 0=at low, 1=at high

    print(f"  因子完成 ({time.time()-t0:.0f}s)", flush=True)

    # ============================================================
    # Define scoring functions
    # ============================================================

    # A. Pure momentum (long only)
    def score_mom5_long(si, di):
        v = mom5[si, di]
        return v if not np.isnan(v) else np.nan

    # B. OI-price divergence (long + short)
    def score_oi_price(si, di):
        v = oi_price_div[si, di]
        return v if not np.isnan(v) else np.nan

    # C. Multi-factor composite with momentum emphasis
    def make_composite(w_mom=0.35, w_oi=0.25, w_vdp=0.20, w_vol=0.10, w_body=0.10):
        def score(si, di):
            vals = []; ws = []
            m5 = mom5[si, di]
            if not np.isnan(m5): vals.append(np.clip(m5*8, -1, 1)); ws.append(w_mom)
            od = oi_price_div[si, di]
            if not np.isnan(od): vals.append(np.clip(od, -1, 1)); ws.append(w_oi)
            vd = vdp_ema[si, di]
            if not np.isnan(vd): vals.append(np.sign(vd)*min(abs(vd)/5e6, 1)); ws.append(w_vdp)
            vr = vol_ratio[si, di]
            if not np.isnan(vr): vals.append(np.clip((vr-1)*2, -1, 1)); ws.append(w_vol)
            br = body_r[si, di]
            if not np.isnan(br): vals.append(br); ws.append(w_body)
            if len(vals) < 2: return np.nan
            return sum(v*w for v,w in zip(vals, ws)) / sum(ws)
        return score

    # D. Breakout + OI confirmation
    def score_breakout_oi(si, di):
        dp = donch_pos[si, di]
        om = oi_mom5[si, di]
        m5 = mom5[si, di]
        if np.isnan(dp) or np.isnan(m5): return np.nan
        score = m5 * 5  # Base momentum
        # Breakout bonus
        if dp > 0.9: score += 0.5   # Near 20-day high
        if dp < 0.1: score -= 0.5   # Near 20-day low
        # OI confirmation
        if not np.isnan(om):
            if om > 0.05 and score > 0: score += 0.3  # OI confirms uptrend
            if om < -0.05 and score < 0: score -= 0.3  # OI confirms downtrend
        return np.clip(score, -1, 1)

    # E. Strong momentum filter (only very strong signals)
    def make_filtered_mom(min_mom=0.02, require_vol=True):
        def score(si, di):
            m5 = mom5[si, di]
            if np.isnan(m5): return np.nan
            if abs(m5) < min_mom: return 0  # Too weak
            score = m5
            # Volume confirmation
            if require_vol:
                vr = vol_ratio[si, di]
                if np.isnan(vr) or vr < 0.8: return 0  # Low volume = no signal
            # OI confirmation bonus
            om = oi_mom5[si, di]
            if not np.isnan(om):
                if (m5 > 0 and om > 0) or (m5 < 0 and om < 0):
                    score *= 1.5  # OI confirms
                else:
                    score *= 0.5  # OI diverges
            return np.clip(score, -1, 1)
        return score

    # F. Multi-timeframe alignment (3d + 10d momentum)
    def score_mtf_align(si, di):
        m3 = mom3[si, di]
        m10 = mom10[si, di]
        if np.isnan(m3) or np.isnan(m10): return np.nan
        if (m3 > 0 and m10 > 0) or (m3 < 0 and m10 < 0):
            return np.clip((m3 + m10) * 5, -1, 1)
        return 0  # Timeframes disagree

    # G. VDP + momentum fusion
    def score_vdp_mom(si, di):
        m5 = mom5[si, di]
        vd = vdp_ema[si, di]
        if np.isnan(m5): return np.nan
        score = np.clip(m5 * 8, -1, 1)
        if not np.isnan(vd):
            if (m5 > 0 and vd > 0) or (m5 < 0 and vd < 0):
                score *= 1.3  # VDP confirms
            else:
                score *= 0.3  # VDP contradicts
        return score

    # ============================================================
    # Parameter configurations
    # ============================================================
    configs = []

    # Strategy × hold_max × trail_atr × stop_loss
    strats = [
        ("MOM5_L", score_mom5_long, False),
        ("COMBO_v1", make_composite(0.35, 0.25, 0.20, 0.10, 0.10), True),
        ("COMBO_v2", make_composite(0.40, 0.20, 0.25, 0.10, 0.05), True),
        ("COMBO_v3", make_composite(0.25, 0.35, 0.20, 0.10, 0.10), True),
        ("OI_PRICE", score_oi_price, True),
        ("BRK_OI", score_breakout_oi, True),
        ("FILT_MOM", make_filtered_mom(0.02, True), True),
        ("FILT_MOM2", make_filtered_mom(0.03, True), True),
        ("MTF_ALIGN", score_mtf_align, True),
        ("VDP_MOM", score_vdp_mom, True),
    ]

    for sname, sfn, short in strats:
        for hold in [3, 5, 7]:
            for trail in [1.5, 2.0, 3.0]:
                for sl in [0.03, 0.05]:
                    configs.append((sname, sfn, short, hold, trail, sl))

    print(f"  共 {len(configs)} 个配置", flush=True)

    results = []
    for ci, (sname, sfn, short, hold, trail, sl) in enumerate(configs):
        if ci % 100 == 0:
            print(f"  配置 {ci}/{len(configs)} ({len(results)} profitable)", flush=True)

        r = run_swing(NS, ND, dates, C, O, H, L, V, OI, syms,
                      sfn, f"{sname}_H{hold}_T{trail}_S{sl}",
                      hold_max=hold, trail_atr=trail, stop_loss=sl,
                      allow_short=short)
        if r and r['ann'] > 5:
            results.append(r)

    print(f"\n  完成 ({time.time()-t0:.0f}s, {len(results)} >5%)", flush=True)
    results.sort(key=lambda x: -x['ann'])

    print(f"\n{'='*95}", flush=True)
    print(f"  TOP 30", flush=True)
    print(f"  {'Strategy':35s} | {'Ann':>8s} {'WR':>5s} {'N':>4s} {'DD':>6s} {'AvgW':>6s} {'AvgL':>6s} {'AvgD':>5s}", flush=True)
    for r in results[:30]:
        print(f"  {r['name']:35s} | {r['ann']:+8.1f}% {r['wr']:5.1f}% {r['n']:4d} "
              f"{r['dd']:6.1f}% {r['avg_win']:+6.2f}% {r['avg_loss']:6.2f}% {r['avg_days']:5.1f}d", flush=True)

    for i, r in enumerate(results[:5]):
        print(f"\n  #{i+1}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.0f}%, DD={r['dd']:.1f}%)", flush=True)
        print(f"    AvgWin={r['avg_win']:+.2f}% AvgLoss={r['avg_loss']:.2f}% AvgDays={r['avg_days']:.1f}", flush=True)
        # Exit reasons
        for reason, s in sorted(r['reasons'].items(), key=lambda x: -x[1]['n']):
            rwr = s['w'] / max(s['n'], 1) * 100
            print(f"    {reason:12s}: {s['n']:4d}t WR={rwr:.0f}% pnl={s['pnl']:+.1f}%", flush=True)
        # Yearly
        for y in sorted(r['years'].keys()):
            s = r['years'][y]
            wr = s['w'] / max(s['n'], 1) * 100
            print(f"    {y}: {s['n']:3d}t WR={wr:.0f}% pnl={s['pnl']:+.1f}%", flush=True)

    print(f"\n  目标: 年化600%+ WR50%+ 无杠杆 纯日线 2-7天持仓", flush=True)
    if results and results[0]['ann'] >= 600:
        print(f"  >>> TARGET ACHIEVED: {results[0]['name']} = {results[0]['ann']:+.1f}% <<<", flush=True)
    elif results:
        print(f"  Best: {results[0]['ann']:+.1f}% — gap to 600%: {600-results[0]['ann']:.0f}%", flush=True)
    print(f"{'='*95}", flush=True)
