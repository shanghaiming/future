"""
Alpha Futures V12b — VDP Swing (无杠杆, 无Gap)
===============================================
V12日内版本: +146% (无滑点) / +38% (0.1%滑点) → 不够
V12b改进: VDP信号+多日持仓 → 捕获更大波动
策略:
  A. VDP趋势跟踪: VDP_EMA翻转→入场, VDP反向→出场, 持仓1-10天
  B. VDP+OI强势突破: VDP>0 + OI增加 + 价格突破 → 持有到趋势结束
  C. 综合评分swing: 多因子评分排名, 持最强品种, 动态换仓
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


if __name__ == '__main__':
    print("=" * 80, flush=True)
    print("  Alpha Futures V12b — VDP Swing (无杠杆, 无Gap)", flush=True)
    print("=" * 80, flush=True)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    print("\n  预计算因子...", flush=True)
    t0 = time.time()

    oi_mom5 = np.full((NS, ND), np.nan)
    vol_ratio = np.full((NS, ND), np.nan)
    vdp_raw = np.full((NS, ND), np.nan)
    vdp_ema = np.full((NS, ND), np.nan)
    atr10 = np.full((NS, ND), np.nan)
    mom5 = np.full((NS, ND), np.nan)
    mom10 = np.full((NS, ND), np.nan)

    for si in range(NS):
        vdp_ema_val = 0
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
                    vdp_raw[si, di] = vdp_val
                    vdp_ema_val = 2.0/15 * vdp_val + (1 - 2.0/15) * vdp_ema_val
                    vdp_ema[si, di] = vdp_ema_val

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
                if not np.isnan(c5) and c5 > 0:
                    mom5[si, di] = (c_now - c5) / c5
                c10 = C[si, max(0, d-9)]
                if not np.isnan(c10) and c10 > 0:
                    mom10[si, di] = (c_now - c10) / c10

    print(f"  因子完成 ({time.time()-t0:.0f}s)", flush=True)
    results = []

    # ==================================================================
    # Strategy A: VDP趋势跟踪 (swing, dynamic exit)
    # VDP_EMA从正转负 → 平多; VDP_EMA从负转正 → 平空
    # 入场: VDP_EMA + OI + 动量 三重确认
    # ==================================================================
    print("\n  [A] VDP趋势跟踪", flush=True)

    for use_oi in [True, False]:
        for use_mom in [True, False]:
            for sl in [0.03, 0.05, 0.10]:
                for trail_atr in [0, 1.5, 2.0]:
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

                                # ATR trailing stop
                                if trail_atr > 0 and pos.get('trail'):
                                    if pos['dir'] == 1 and c < pos['trail']:
                                        cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                                        trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                                                       'days': di - pos['entry_di'], 'di': di,
                                                       'reason': 'trail', 'year': year})
                                        pos = None; continue
                                    elif pos['dir'] == -1 and c > pos['trail']:
                                        cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                                        trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                                                       'days': di - pos['entry_di'], 'di': di,
                                                       'reason': 'trail', 'year': year})
                                        pos = None; continue
                                    # Update trail
                                    atr = atr10[pos['si'], di]
                                    if not np.isnan(atr) and atr > 0:
                                        if pos['dir'] == 1:
                                            ns = c - trail_atr * atr
                                            if ns > pos['trail']: pos['trail'] = ns
                                        else:
                                            ns = c + trail_atr * atr
                                            if ns < pos['trail']: pos['trail'] = ns

                                # Fixed stop
                                if pos is not None and pnl_pct / 100 < -sl:
                                    cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                                    trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                                                   'days': di - pos['entry_di'], 'di': di,
                                                   'reason': 'stop', 'year': year})
                                    pos = None; continue

                                # VDP reversal exit
                                if pos is not None:
                                    vdp_now = vdp_ema[pos['si'], di]
                                    if not np.isnan(vdp_now):
                                        if pos['dir'] == 1 and vdp_now < 0:
                                            cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                                            trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                                                           'days': di - pos['entry_di'], 'di': di,
                                                           'reason': 'vdp_rev', 'year': year})
                                            pos = None; continue
                                        elif pos['dir'] == -1 and vdp_now > 0:
                                            cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                                            trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                                                           'days': di - pos['entry_di'], 'di': di,
                                                           'reason': 'vdp_rev', 'year': year})
                                            pos = None; continue

                                # Max hold
                                if pos is not None and di - pos['entry_di'] >= 15:
                                    cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                                    trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                                                   'days': di - pos['entry_di'], 'di': di,
                                                   'reason': 'time', 'year': year})
                                    pos = None

                        # Entry
                        if pos is None:
                            candidates = []
                            for si in range(NS):
                                c = C[si, di]
                                if np.isnan(c) or c <= 0: continue
                                v = vdp_ema[si, di]
                                if np.isnan(v): continue

                                # VDP direction
                                direction = 1 if v > 0 else -1
                                score = abs(v)

                                # OI filter
                                om = oi_mom5[si, di]
                                if use_oi:
                                    if np.isnan(om) or om <= 0: continue
                                    score *= (1 + om * 5)

                                # Momentum filter
                                m = mom5[si, di]
                                if use_mom:
                                    if np.isnan(m): continue
                                    if direction == 1 and m <= 0: continue
                                    if direction == -1 and m >= 0: continue
                                    score *= (1 + abs(m) * 5)

                                # Volume confirmation
                                vr = vol_ratio[si, di]
                                if not np.isnan(vr) and vr > 1.0:
                                    score *= vr

                                candidates.append((si, c, direction, score, syms[si]))

                            if candidates:
                                candidates.sort(key=lambda x: x[3], reverse=True)
                                si, price, direction, score, sym = candidates[0]
                                mult = MULT.get(sym, DEF_MULT)
                                notional = price * mult
                                if notional > 0:
                                    lots = int(cash / notional)
                                    if lots > 0:
                                        cost = notional * lots * (1 + COMM_RATE)
                                        if cost <= cash:
                                            cash -= cost
                                            trail = 0
                                            atr = atr10[si, di]
                                            if trail_atr > 0 and not np.isnan(atr) and atr > 0:
                                                trail = price - trail_atr * atr if direction == 1 else price + trail_atr * atr
                                            pos = {
                                                'si': si, 'entry': price, 'entry_di': di,
                                                'lots': lots, 'dir': direction, 'sym': sym,
                                                'trail': trail,
                                            }

                    # Close
                    if pos is not None:
                        c = C[pos['si'], ND-1]
                        if np.isnan(c) or c <= 0: c = pos['entry']
                        mult = MULT.get(pos['sym'], DEF_MULT)
                        pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
                        pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100
                        cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                        trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                                       'days': 999, 'di': ND-1, 'reason': 'end',
                                       'year': dates[ND-1].year})

                    if trades and cash > 0:
                        equity = float(CASH0); peak = float(CASH0); max_dd = 0
                        for t in sorted(trades, key=lambda x: x['di']):
                            equity += t['pnl_abs']
                            if equity > peak: peak = equity
                            dd = (peak - equity) / peak * 100 if peak > 0 else 0
                            if dd > max_dd: max_dd = dd

                        days_total = (dates[ND-1] - dates[MIN_TRAIN]).days
                        yr = max(days_total / 365.25, 0.01)
                        ann = ((cash / CASH0) ** (1 / yr) - 1) * 100
                        nw = sum(1 for t in trades if t['pnl_abs'] > 0)
                        wr = nw / max(len(trades), 1) * 100
                        avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_abs'] > 0]) if nw > 0 else 0
                        avg_loss = np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_abs'] <= 0]) if nw < len(trades) else 0

                        for t in trades:
                            y = t.get('year', 'unknown')
                            if y not in year_stats:
                                year_stats[y] = {'trades': 0, 'wins': 0, 'total_pnl': 0, 'pnl_abs_sum': 0}
                            year_stats[y]['trades'] += 1
                            if t['pnl_abs'] > 0: year_stats[y]['wins'] += 1
                            year_stats[y]['total_pnl'] += t['pnl_pct']
                            year_stats[y]['pnl_abs_sum'] += t['pnl_abs']

                        if ann > 10:
                            r = {
                                'ann': round(ann, 1), 'n': len(trades), 'wr': round(wr, 1),
                                'max_dd': round(max_dd, 1), 'final': round(cash, 0),
                                'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
                                'year_stats': year_stats,
                                'desc': f"VDP_SW_{'oi' if use_oi else 'x'}_{'mom' if use_mom else 'x'}",
                                'sl': sl, 'trail': trail_atr,
                            }
                            results.append(r)
                    # End of this config

        print(f"    {'oi' if use_oi else 'x'}_{'mom' if use_mom else 'x'} done ({len(results)})", flush=True)

    # ==================================================================
    # Strategy B: OI强势 + VDP确认 + 价格突破 (Donchian + OI + VDP)
    # ==================================================================
    print("  [B] OI强势突破", flush=True)

    for chan in [5, 7, 10]:
        for exit_vdp in [True, False]:
            for sl in [0.03, 0.05, 0.10]:
                for trail_atr in [0, 2.0]:
                    cash = float(CASH0)
                    pos = None
                    trades = []
                    year_stats = {}

                    for di in range(MIN_TRAIN, ND):
                        year = dates[di].year

                        if pos is not None:
                            c = C[pos['si'], di]
                            mult = MULT.get(pos['sym'], DEF_MULT)
                            if np.isnan(c):
                                pass
                            else:
                                pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
                                pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100

                                if trail_atr > 0 and pos.get('trail'):
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
                                                ns = c - trail_atr * atr
                                                if ns > pos['trail']: pos['trail'] = ns
                                            else:
                                                ns = c + trail_atr * atr
                                                if ns < pos['trail']: pos['trail'] = ns

                                if pos is not None and pnl_pct / 100 < -sl:
                                    cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                                    trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl, 'days': di-pos['entry_di'], 'di': di, 'reason': 'stop', 'year': year})
                                    pos = None; continue

                                if pos is not None and exit_vdp:
                                    v = vdp_ema[pos['si'], di]
                                    if not np.isnan(v):
                                        if pos['dir'] == 1 and v < 0:
                                            cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                                            trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl, 'days': di-pos['entry_di'], 'di': di, 'reason': 'vdp', 'year': year})
                                            pos = None; continue
                                        elif pos['dir'] == -1 and v > 0:
                                            cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                                            trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl, 'days': di-pos['entry_di'], 'di': di, 'reason': 'vdp', 'year': year})
                                            pos = None; continue

                                if pos is not None and di - pos['entry_di'] >= 20:
                                    cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                                    trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl, 'days': di-pos['entry_di'], 'di': di, 'reason': 'time', 'year': year})
                                    pos = None

                        if pos is None:
                            candidates = []
                            for si in range(NS):
                                c = C[si, di]
                                if np.isnan(c) or c <= 0: continue

                                # Donchian breakout
                                d = di - 1
                                ch = C[si, max(0, d-chan):d]
                                vc = ch[~np.isnan(ch)]
                                if len(vc) < max(1, chan // 2): continue
                                hch, lch = np.max(vc), np.min(vc)

                                is_breakout_up = c > hch
                                is_breakout_down = c < lch
                                if not is_breakout_up and not is_breakout_down: continue

                                direction = 1 if is_breakout_up else -1

                                # OI increasing
                                om = oi_mom5[si, di]
                                if np.isnan(om) or om <= 0: continue

                                # VDP confirms
                                v = vdp_ema[si, di]
                                if np.isnan(v): continue
                                if direction == 1 and v <= 0: continue
                                if direction == -1 and v >= 0: continue

                                # Volume above average
                                vr = vol_ratio[si, di]
                                vol_ok = not np.isnan(vr) and vr > 1.0

                                score = abs(om) * abs(v) * (vr if vol_ok else 1.0)
                                candidates.append((si, c, direction, score, syms[si]))

                            if candidates:
                                candidates.sort(key=lambda x: x[3], reverse=True)
                                si, price, direction, score, sym = candidates[0]
                                mult = MULT.get(sym, DEF_MULT)
                                notional = price * mult
                                if notional > 0:
                                    lots = int(cash / notional)
                                    if lots > 0:
                                        cost = notional * lots * (1 + COMM_RATE)
                                        if cost <= cash:
                                            cash -= cost
                                            trail = 0
                                            atr = atr10[si, di]
                                            if trail_atr > 0 and not np.isnan(atr) and atr > 0:
                                                trail = price - trail_atr * atr if direction == 1 else price + trail_atr * atr
                                            pos = {'si': si, 'entry': price, 'entry_di': di,
                                                   'lots': lots, 'dir': direction, 'sym': sym, 'trail': trail}

                    if pos is not None:
                        c = C[pos['si'], ND-1]
                        if np.isnan(c) or c <= 0: c = pos['entry']
                        mult = MULT.get(pos['sym'], DEF_MULT)
                        pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
                        pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100
                        cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                        trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl, 'days': 999, 'di': ND-1, 'reason': 'end', 'year': dates[ND-1].year})

                    if trades and cash > 0:
                        equity = float(CASH0); peak = float(CASH0); max_dd = 0
                        for t in sorted(trades, key=lambda x: x['di']):
                            equity += t['pnl_abs']
                            if equity > peak: peak = equity
                            dd = (peak - equity) / peak * 100 if peak > 0 else 0
                            if dd > max_dd: max_dd = dd

                        days_total = (dates[ND-1] - dates[MIN_TRAIN]).days
                        yr = max(days_total / 365.25, 0.01)
                        ann = ((cash / CASH0) ** (1 / yr) - 1) * 100
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

                        if ann > 10:
                            r = {
                                'ann': round(ann, 1), 'n': len(trades), 'wr': round(wr, 1),
                                'max_dd': round(max_dd, 1), 'final': round(cash, 0),
                                'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
                                'year_stats': year_stats,
                                'desc': f"OI_BRK_c{chan}_{'vdp' if exit_vdp else 'x'}",
                                'sl': sl, 'trail': trail_atr,
                            }
                            results.append(r)

        print(f"    c{chan} done ({len(results)})", flush=True)

    # ==================================================================
    # Results
    # ==================================================================
    print(f"\n  完成 ({time.time()-t0:.0f}s, {len(results)} >10%)", flush=True)
    results.sort(key=lambda x: -x['ann'])

    print(f"\n{'='*80}", flush=True)
    print(f"  TOP 40", flush=True)
    print(f"  {'策略':<22s} {'SL':>4s} {'Trail':>5s} | {'Ann':>8s} {'N':>5s} {'WR':>5s} {'AvgW':>6s} {'AvgL':>5s} {'DD':>6s}", flush=True)
    for r in results[:40]:
        print(f"  {r['desc']:<22s} SL{r['sl']:.0%} T{r['trail']:.1f} | "
              f"{r['ann']:+8.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['avg_win']:+6.2f}% {r['avg_loss']:5.2f}% {r['max_dd']:6.1f}%", flush=True)

    for i, r in enumerate(results[:5]):
        print(f"\n  #{i+1}: {r['desc']} SL{r['sl']:.0%} T{r['trail']:.1f} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%, WR={r['wr']:.0f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} t, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    seen = set()
    print(f"\n  --- 按策略类型 ---", flush=True)
    for r in results:
        prefix = r['desc'].split('_')[0]
        if prefix not in seen:
            seen.add(prefix)
            sub = [x for x in results if x['desc'].startswith(prefix)]
            best = sub[0]
            print(f"  {prefix:<12s}: Best={best['ann']:+.1f}% DD={best['max_dd']:.1f}% WR={best['wr']:.0f}%", flush=True)

    if results:
        print(f"\n  Best: {results[0]['ann']:+.1f}% DD={results[0]['max_dd']:.1f}%", flush=True)
    print(f"  目标: 年化600%+ WR50%+ 无杠杆 无Gap", flush=True)
    print(f"{'='*80}", flush=True)
