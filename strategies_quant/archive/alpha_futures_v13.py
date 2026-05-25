"""
Alpha Futures V13 — Multi-Factor Score Rotation (无杠杆, 纯日线)
=================================================================
核心思路: 每天对68个品种计算多因子评分, 持有最强品种, 动态换仓
频率目标: 50-100 trades/year, 3-7天持仓
因子:
  1. OI动量 (40%): OI 5日变化率 — 资金方向
  2. 价格动量 (25%): 5日/10日收益 — 趋势方向
  3. VDP量压 (20%): Volume Delta Pressure — 真实买卖力
  4. 量能 (15%): 成交量比率 — 参与度
约束: 纯日线, 不做日内, 不用gap, 无杠杆
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, COMMISSION, STAMP_DUTY, CASH0

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
COMM_RATE = 0.0003


def compute_scores(si, di, C, O, H, L, V, OI,
                    oi_mom5, mom5, mom10, vdp_ema, vol_ratio,
                    w_oi=0.40, w_mom=0.25, w_vdp=0.20, w_vol=0.15):
    """Compute long and short scores for symbol si on day di.
    All inputs use di-1 data. Returns (long_score, short_score)."""
    # OI component
    oi_val = oi_mom5[si, di]
    oi_score = np.clip(oi_val * 5, -1, 1) if not np.isnan(oi_val) else 0

    # Momentum component: blend 5d and 10d
    m5 = mom5[si, di]
    m10 = mom10[si, di]
    mom_val = 0
    mom_cnt = 0
    if not np.isnan(m5):
        mom_val += m5 * 10
        mom_cnt += 1
    if not np.isnan(m10):
        mom_val += m10 * 5
        mom_cnt += 1
    if mom_cnt > 0:
        mom_val /= mom_cnt
    mom_score = np.clip(mom_val, -1, 1)

    # VDP component
    vdp_val = vdp_ema[si, di]
    if not np.isnan(vdp_val):
        # Normalize VDP: sign * min(abs/scale, 1)
        vdp_score = np.sign(vdp_val) * min(abs(vdp_val) / 1e7, 1.0)
    else:
        vdp_score = 0

    # Volume component
    vr = vol_ratio[si, di]
    vol_score = np.clip((vr - 1) * 2, -1, 1) if not np.isnan(vr) else 0

    # Check data availability
    has_data = not np.isnan(oi_val) or mom_cnt > 0
    if not has_data:
        return 0, 0

    # Weighted composite
    composite = w_oi * oi_score + w_mom * mom_score + w_vdp * vdp_score + w_vol * vol_score

    # Long score: positive composite = bullish
    # Short score: negative composite = bearish
    long_score = composite if composite > 0 else 0
    short_score = -composite if composite < 0 else 0

    return long_score, short_score


if __name__ == '__main__':
    print("=" * 80, flush=True)
    print("  Alpha Futures V13 — Multi-Factor Score Rotation (纯日线, 无杠杆)", flush=True)
    print("=" * 80, flush=True)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    print("\n  预计算因子...", flush=True)
    t0 = time.time()

    oi_mom5 = np.full((NS, ND), np.nan)
    vol_ratio = np.full((NS, ND), np.nan)
    vdp_ema = np.full((NS, ND), np.nan)
    atr10 = np.full((NS, ND), np.nan)
    mom5 = np.full((NS, ND), np.nan)
    mom10 = np.full((NS, ND), np.nan)

    for si in range(NS):
        vdp_e = 0
        for di in range(20, ND):
            d = di - 1
            oi_now = OI[si, d]
            if not np.isnan(oi_now) and oi_now > 0:
                oi5 = OI[si, max(0, d-4):d+1]
                oi5v = oi5[~np.isnan(oi5)]
                if len(oi5v) >= 3:
                    oi_mom5[si, di] = (oi_now - oi5v[0]) / oi5v[0]
            v_now = V[si, d]
            if not np.isnan(v_now):
                v20 = V[si, max(0, d-19):d+1]
                v20v = v20[~np.isnan(v20)]
                if len(v20v) >= 10:
                    vol_ratio[si, di] = v_now / np.mean(v20v)
            hl = H[si, d] - L[si, d]
            if not np.isnan(hl) and hl > 0:
                cd = C[si, d]; hd = H[si, d]; ld = L[si, d]; vd = V[si, d]
                if not any(np.isnan([cd, hd, ld, vd])):
                    vdp_val = vd * (2*cd - hd - ld) / hl
                    vdp_e = 2.0/15 * vdp_val + (1 - 2.0/15) * vdp_e
                    vdp_ema[si, di] = vdp_e
            if di >= 11:
                trs = []
                for dd in range(max(1, d-9), d+1):
                    hi = H[si, dd]; lo = L[si, dd]; pc = C[si, dd-1]
                    if np.isnan(hi) or np.isnan(lo): continue
                    tr = hi - lo
                    if not np.isnan(pc): tr = max(tr, abs(hi-pc), abs(lo-pc))
                    trs.append(tr)
                if trs: atr10[si, di] = np.mean(trs)
            c_now = C[si, d]
            if not np.isnan(c_now) and c_now > 0:
                c5 = C[si, max(0, d-4)]
                if not np.isnan(c5) and c5 > 0: mom5[si, di] = (c_now - c5) / c5
                c10 = C[si, max(0, d-9)]
                if not np.isnan(c10) and c10 > 0: mom10[si, di] = (c_now - c10) / c10

    print(f"  因子完成 ({time.time()-t0:.0f}s)", flush=True)

    results = []

    # ==================================================================
    # Parameter sweep
    # ==================================================================
    configs = []
    for w_oi in [0.30, 0.40, 0.50]:
        for w_mom in [0.20, 0.30]:
            for w_vdp in [0.15, 0.25]:
                w_vol = round(1.0 - w_oi - w_mom - w_vdp, 2)
                if w_vol < 0.05: continue
                for min_score in [0.05, 0.10, 0.15, 0.20]:
                    for hold_min in [2, 3, 5]:
                        for sl in [0.03, 0.05, 0.08]:
                            for trail in [0, 2.0]:
                                configs.append((w_oi, w_mom, w_vdp, w_vol, min_score, hold_min, sl, trail))

    print(f"  共 {len(configs)} 个配置", flush=True)

    for ci, (w_oi, w_mom, w_vdp, w_vol, min_score, hold_min, sl, trail) in enumerate(configs):
        if ci % 200 == 0:
            print(f"  配置 {ci}/{len(configs)} ({len(results)} profitable)", flush=True)

        cash = float(CASH0)
        pos = None
        trades = []
        year_stats = {}

        for di in range(MIN_TRAIN, ND):
            year = dates[di].year

            # Manage position
            if pos is not None:
                c = C[pos['si'], di]
                if np.isnan(c):
                    pass
                else:
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
                    pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100

                    # Trail stop
                    if trail > 0 and pos.get('trail') and not np.isnan(pos['trail']):
                        if pos['dir'] == 1 and c < pos['trail']:
                            cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                            trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl, 'days': di-pos['entry_di'], 'di': di, 'reason': 'trail', 'year': year})
                            pos = None; continue
                        elif pos['dir'] == -1 and c > pos['trail']:
                            cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                            trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl, 'days': di-pos['entry_di'], 'di': di, 'reason': 'trail', 'year': year})
                            pos = None; continue
                        if pos is not None:
                            atr = atr10[pos['si'], di]
                            if not np.isnan(atr) and atr > 0:
                                if pos['dir'] == 1:
                                    ns = c - trail * atr
                                    if ns > pos['trail']: pos['trail'] = ns
                                else:
                                    ns = c + trail * atr
                                    if ns < pos['trail']: pos['trail'] = ns

                    # Fixed stop
                    if pos is not None and pnl_pct / 100 < -sl:
                        cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                        trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl, 'days': di-pos['entry_di'], 'di': di, 'reason': 'stop', 'year': year})
                        pos = None; continue

                    # Score-based exit: if hold_min passed, check if still best
                    if pos is not None and di - pos['entry_di'] >= hold_min:
                        # Check if current position's score is still high enough
                        l_score, s_score = compute_scores(pos['si'], di, C, O, H, L, V, OI,
                                                           oi_mom5, mom5, mom10, vdp_ema, vol_ratio,
                                                           w_oi, w_mom, w_vdp, w_vol)
                        pos_score = l_score if pos['dir'] == 1 else s_score

                        # Find best candidate
                        best_si, best_dir, best_sc = -1, 0, 0
                        for sj in range(NS):
                            ls, ss = compute_scores(sj, di, C, O, H, L, V, OI,
                                                     oi_mom5, mom5, mom10, vdp_ema, vol_ratio,
                                                     w_oi, w_mom, w_vdp, w_vol)
                            if ls > best_sc:
                                best_sc = ls; best_si = sj; best_dir = 1
                            if ss > best_sc:
                                best_sc = ss; best_si = sj; best_dir = -1

                        # Switch if new candidate is significantly better
                        if best_sc > pos_score * 1.5 + 0.05 and best_si != pos['si']:
                            cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                            trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl, 'days': di-pos['entry_di'], 'di': di, 'reason': 'rotate', 'year': year})
                            pos = None

                    # Max hold
                    if pos is not None and di - pos['entry_di'] >= 15:
                        cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                        trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl, 'days': di-pos['entry_di'], 'di': di, 'reason': 'time', 'year': year})
                        pos = None

            # Entry
            if pos is None:
                best_si, best_dir, best_sc = -1, 0, 0
                for si in range(NS):
                    ls, ss = compute_scores(si, di, C, O, H, L, V, OI,
                                             oi_mom5, mom5, mom10, vdp_ema, vol_ratio,
                                             w_oi, w_mom, w_vdp, w_vol)
                    if ls > min_score and ls > best_sc:
                        best_sc = ls; best_si = si; best_dir = 1
                    if ss > min_score and ss > best_sc:
                        best_sc = ss; best_si = si; best_dir = -1

                if best_si >= 0:
                    c = C[best_si, di]
                    if np.isnan(c) or c <= 0: continue
                    sym = syms[best_si]
                    mult = MULT.get(sym, DEF_MULT)
                    notional = c * mult
                    if notional > 0:
                        lots = int(cash / notional)
                        if lots > 0:
                            cost = notional * lots * (1 + COMM_RATE)
                            if cost <= cash:
                                cash -= cost
                                tr = 0
                                atr = atr10[best_si, di]
                                if trail > 0 and not np.isnan(atr) and atr > 0:
                                    tr = c - trail*atr if best_dir == 1 else c + trail*atr
                                pos = {'si': best_si, 'entry': c, 'entry_di': di,
                                       'lots': lots, 'dir': best_dir, 'sym': sym, 'trail': tr}

        # Close
        if pos is not None:
            c = C[pos['si'], ND-1]
            if np.isnan(c) or c <= 0: c = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
            pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100
            cash += c * mult * pos['lots'] * (1 - COMM_RATE)
            trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl, 'days': 999, 'di': ND-1, 'reason': 'end', 'year': dates[ND-1].year})

        if not trades or cash <= 0: continue

        equity = float(CASH0); peak = float(CASH0); max_dd = 0
        for t in sorted(trades, key=lambda x: x['di']):
            equity += t['pnl_abs']
            if equity > peak: peak = equity
            dd = (peak - equity) / peak * 100 if peak > 0 else 0
            if dd > max_dd: max_dd = dd

        days_total = (dates[ND-1] - dates[MIN_TRAIN]).days
        yr = max(days_total / 365.25, 0.01)
        ann = ((cash / CASH0) ** (1 / yr) - 1) * 100
        if ann < 10: continue

        nw = sum(1 for t in trades if t['pnl_abs'] > 0)
        wr = nw / max(len(trades), 1) * 100
        avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_abs'] > 0]) if nw > 0 else 0
        avg_loss = np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_abs'] <= 0]) if nw < len(trades) else 0

        for t in trades:
            y = t.get('year', 'unknown')
            if y not in year_stats: year_stats[y] = {'trades': 0, 'wins': 0, 'total_pnl': 0, 'pnl_abs_sum': 0}
            year_stats[y]['trades'] += 1
            if t['pnl_abs'] > 0: year_stats[y]['wins'] += 1
            year_stats[y]['total_pnl'] += t['pnl_pct']
            year_stats[y]['pnl_abs_sum'] += t['pnl_abs']

        results.append({
            'ann': round(ann, 1), 'n': len(trades), 'wr': round(wr, 1),
            'max_dd': round(max_dd, 1), 'final': round(cash, 0),
            'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
            'year_stats': year_stats,
            'w': f"O{w_oi:.0%}M{w_mom:.0%}V{w_vdp:.0%}L{w_vol:.0%}",
            'ms': min_score, 'hm': hold_min, 'sl': sl, 'tr': trail,
        })

    print(f"\n  完成 ({time.time()-t0:.0f}s, {len(results)} >10%)", flush=True)
    results.sort(key=lambda x: -x['ann'])

    print(f"\n{'='*80}", flush=True)
    print(f"  TOP 40", flush=True)
    print(f"  {'权重':>16s} {'MS':>4s} {'HM':>3s} {'SL':>4s} {'TR':>4s} | {'Ann':>8s} {'N':>5s} {'WR':>5s} {'AvgW':>6s} {'AvgL':>5s} {'DD':>6s}", flush=True)
    for r in results[:40]:
        print(f"  {r['w']:>16s} {r['ms']:.2f} {r['hm']:>3d} SL{r['sl']:.0%} T{r['tr']:.1f} | "
              f"{r['ann']:+8.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['avg_win']:+6.2f}% {r['avg_loss']:5.2f}% {r['max_dd']:6.1f}%", flush=True)

    for i, r in enumerate(results[:5]):
        print(f"\n  #{i+1}: {r['w']} ms={r['ms']} hm={r['hm']} sl={r['sl']} tr={r['tr']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%, WR={r['wr']:.0f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} t, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    if results:
        print(f"\n  Best: {results[0]['ann']:+.1f}% DD={results[0]['max_dd']:.1f}%", flush=True)
    print(f"  目标: 年化600%+ WR50%+ 无杠杆 纯日线", flush=True)
    print(f"{'='*80}", flush=True)
