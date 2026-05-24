"""
Alpha Futures V10b — Overnight Gap Fade (无杠杆) — 修复版
=========================================================
修复:
  1. VDP look-ahead → 改用前一日数据
  2. P2/P3 allocation → 严格 total notional <= cash
策略:
  A. Gap Fade (反转) — gap up做空, gap down做多, 当天平仓
  B. Gap Momentum (延续) — gap up做多, gap down做空
  C. Gap Fade + Swing — fade方向持仓N天
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


def gap_backtest(NS, ND, dates, C, O, H, L, V, OI, syms,
                 gap_min, mode, max_pos, alloc_pct, filter_mode,
                 oi_mom5, vol_ratio):
    """
    日内Gap策略回测 (严格无杠杆)
    VDP使用前一日数据, 总名义价值不超过cash
    """
    cash = float(CASH0)
    trades = []
    year_stats = {}

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year
        candidates = []

        for si in range(NS):
            prev_close = C[si, di-1]
            open_p = O[si, di]
            close_p = C[si, di]

            if any(np.isnan(x) for x in [prev_close, open_p, close_p]):
                continue
            if prev_close <= 0 or open_p <= 0:
                continue

            gap = (open_p - prev_close) / prev_close
            if abs(gap) < gap_min:
                continue

            # 方向
            if mode == 'fade':
                direction = -1 if gap > 0 else 1
            else:
                direction = 1 if gap > 0 else -1

            score = abs(gap)

            # === 所有过滤器只用前一日数据 ===
            # VDP: 前一日
            hl_prev = H[si, di-1] - L[si, di-1]
            vdp_prev = 0
            if not np.isnan(hl_prev) and hl_prev > 0:
                v_prev = V[si, di-1]
                c_prev = C[si, di-1]
                h_prev = H[si, di-1]
                l_prev = L[si, di-1]
                if not any(np.isnan([v_prev, c_prev, h_prev, l_prev])):
                    vdp_prev = v_prev * (2 * c_prev - h_prev - l_prev) / hl_prev

            if filter_mode != 'none':
                om = oi_mom5[si, di]  # uses OI[di-1] → OK
                vr = vol_ratio[si, di]  # uses V[di-1] → OK

                if 'oi' in filter_mode:
                    if np.isnan(om) or om <= 0:
                        continue
                    score *= (1 + om)

                if 'vol' in filter_mode:
                    if np.isnan(vr) or vr <= 1.0:
                        continue

                if 'vdp' in filter_mode:
                    # VDP昨日买压+今日gap down → fade更可靠
                    # VDP昨日卖压+今日gap up → fade更可靠
                    if direction == 1 and vdp_prev <= 0:  # 做多需要昨日买压
                        continue
                    if direction == -1 and vdp_prev >= 0:  # 做空需要昨日卖压
                        continue

            candidates.append((si, open_p, close_p, direction, score, abs(gap), syms[si]))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[4], reverse=True)

        # 严格分配: total notional <= cash
        remaining = cash * alloc_pct
        day_trades = 0
        for si, open_p, close_p, direction, score, gap_abs, sym in candidates:
            if day_trades >= max_pos:
                break

            mult = MULT.get(sym, DEF_MULT)
            notional_per_lot = open_p * mult
            if notional_per_lot <= 0:
                continue

            # 分配剩余资金的 1/(remaining slots)
            slots_left = max_pos - day_trades
            alloc = remaining / slots_left
            lots = int(alloc / notional_per_lot)
            if lots <= 0:
                continue

            # 确保不超过剩余资金
            while lots * notional_per_lot > remaining:
                lots -= 1
            if lots <= 0:
                continue

            # P&L
            if direction == 1:
                pnl = (close_p - open_p) * mult * lots
            else:
                pnl = (open_p - close_p) * mult * lots

            comm = notional_per_lot * lots * COMM_RATE * 2
            pnl -= comm

            notional_used = notional_per_lot * lots
            remaining -= notional_used

            pnl_pct = pnl / notional_used * 100 if notional_used > 0 else 0
            cash += pnl
            day_trades += 1

            trades.append({
                'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                'di': di, 'year': year, 'si': si, 'dir': direction,
                'sym': sym, 'gap': gap_abs, 'days': 1,
            })

    if not trades:
        return None

    equity = float(CASH0); peak = float(CASH0); max_dd = 0
    for t in sorted(trades, key=lambda x: x['di']):
        equity += t['pnl_abs']
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        if dd > max_dd: max_dd = dd

    final_cash = cash
    if final_cash <= 0:
        return None

    days_total = (dates[ND-1] - dates[MIN_TRAIN]).days
    yr = max(days_total / 365.25, 0.01)
    ann = ((final_cash / CASH0) ** (1 / yr) - 1) * 100

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

    return {
        'ann': round(ann, 1), 'n': len(trades), 'wr': round(wr, 1),
        'max_dd': round(max_dd, 1), 'final': round(final_cash, 0),
        'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
        'year_stats': year_stats,
    }


def gap_swing_backtest(NS, ND, dates, C, O, H, L, V, OI, syms,
                        gap_min, hold_days, sl_pct, filter_mode,
                        oi_mom5, vol_ratio):
    """Gap Fade方向但持有N天的swing策略"""
    cash = float(CASH0)
    positions = []
    trades = []
    year_stats = {}

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # 平仓
        for pos in list(positions):
            si = pos['si']
            c = C[si, di]
            if np.isnan(c): continue
            mult = MULT.get(pos['sym'], DEF_MULT)
            pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
            pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100

            if pnl_pct / 100 < -sl_pct:
                cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                               'days': di - pos['entry_di'], 'di': di,
                               'reason': 'stop', 'year': year, 'si': si, 'dir': pos['dir']})
                positions.remove(pos)
                continue

            if di - pos['entry_di'] >= hold_days:
                cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                               'days': di - pos['entry_di'], 'di': di,
                               'reason': 'time', 'year': year, 'si': si, 'dir': pos['dir']})
                positions.remove(pos)

        # 开仓
        if len(positions) < 1:
            candidates = []
            for si in range(NS):
                prev_close = C[si, di-1]
                open_p = O[si, di]
                if np.isnan(prev_close) or np.isnan(open_p) or prev_close <= 0 or open_p <= 0:
                    continue
                gap = (open_p - prev_close) / prev_close
                if abs(gap) < gap_min:
                    continue

                direction = -1 if gap > 0 else 1  # fade
                score = abs(gap)

                if filter_mode != 'none':
                    om = oi_mom5[si, di]
                    if 'oi' in filter_mode:
                        if np.isnan(om) or om <= 0: continue
                        score *= (1 + om)
                    if 'vol' in filter_mode:
                        vr = vol_ratio[si, di]
                        if np.isnan(vr) or vr <= 1.0: continue

                candidates.append((si, open_p, direction, score, syms[si]))

            if candidates:
                candidates.sort(key=lambda x: x[3], reverse=True)
                si, price, direction, _, sym = candidates[0]
                mult = MULT.get(sym, DEF_MULT)
                notional = price * mult
                if notional > 0:
                    lots = int(cash / notional)
                    if lots > 0:
                        cost = notional * lots * (1 + COMM_RATE)
                        if cost <= cash:
                            cash -= cost
                            positions.append({
                                'si': si, 'entry': price, 'entry_di': di,
                                'lots': lots, 'dir': direction, 'sym': sym,
                            })

    # 清仓
    for pos in positions:
        c = C[pos['si'], ND-1]
        if np.isnan(c) or c <= 0: c = pos['entry']
        mult = MULT.get(pos['sym'], DEF_MULT)
        pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
        pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100
        cash += c * mult * pos['lots'] * (1 - COMM_RATE)
        trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                       'days': 999, 'di': ND-1, 'reason': 'end',
                       'year': dates[ND-1].year, 'si': pos['si'], 'dir': pos['dir']})

    if not trades: return None

    equity = float(CASH0); peak = float(CASH0); max_dd = 0
    for t in sorted(trades, key=lambda x: x['di']):
        equity += t['pnl_abs']
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        if dd > max_dd: max_dd = dd

    final_cash = cash
    if final_cash <= 0: return None

    days_total = (dates[ND-1] - dates[MIN_TRAIN]).days
    yr = max(days_total / 365.25, 0.01)
    ann = ((final_cash / CASH0) ** (1 / yr) - 1) * 100

    nw = sum(1 for t in trades if t['pnl_abs'] > 0)
    wr = nw / max(len(trades), 1) * 100

    for t in trades:
        y = t.get('year', 'unknown')
        if y not in year_stats:
            year_stats[y] = {'trades': 0, 'wins': 0, 'total_pnl': 0, 'pnl_abs_sum': 0}
        year_stats[y]['trades'] += 1
        if t['pnl_abs'] > 0: year_stats[y]['wins'] += 1
        year_stats[y]['total_pnl'] += t['pnl_pct']
        year_stats[y]['pnl_abs_sum'] += t['pnl_abs']

    return {
        'ann': round(ann, 1), 'n': len(trades), 'wr': round(wr, 1),
        'max_dd': round(max_dd, 1), 'final': round(final_cash, 0),
        'avg_win': 0, 'avg_loss': 0,
        'year_stats': year_stats,
    }


if __name__ == '__main__':
    print("=" * 80, flush=True)
    print("  Alpha Futures V10b — Gap Fade (无杠杆, 无look-ahead)", flush=True)
    print("=" * 80, flush=True)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    print("\n  预计算因子...", flush=True)
    t0 = time.time()

    oi_mom5 = np.full((NS, ND), np.nan)
    vol_ratio = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            oi_now = OI[si, di-1]
            if not np.isnan(oi_now) and oi_now > 0:
                oi5 = OI[si, max(0, di-6):di-1]
                oi5v = oi5[~np.isnan(oi5)]
                if len(oi5v) >= 3:
                    oi_mom5[si, di] = (oi_now - oi5v[0]) / oi5v[0]
            v_now = V[si, di-1]
            if not np.isnan(v_now):
                v20 = V[si, max(0, di-21):di-1]
                v20v = v20[~np.isnan(v20)]
                if len(v20v) >= 10:
                    vol_ratio[si, di] = v_now / np.mean(v20v)

    print(f"  因子完成 ({time.time()-t0:.0f}s)", flush=True)

    results = []

    # ===== A: Gap Fade (反转) =====
    print("  [A] Gap Fade (反转, 日内)", flush=True)
    for gap_min in [0.003, 0.005, 0.008, 0.01, 0.015, 0.02]:
        for filt in ['none', 'oi', 'vol', 'oi_vol', 'vdp', 'oi_vdp', 'oi_vol_vdp']:
            for mp in [1, 2, 3]:
                r = gap_backtest(NS, ND, dates, C, O, H, L, V, OI, syms,
                                 gap_min=gap_min, mode='fade', max_pos=mp,
                                 alloc_pct=1.0, filter_mode=filt,
                                 oi_mom5=oi_mom5, vol_ratio=vol_ratio)
                if r:
                    r['desc'] = f"FADE_g{gap_min:.1%}"
                    r['filt'] = filt
                    r['mp'] = mp
                    results.append(r)
        print(f"    g{gap_min:.1%} done ({len(results)})", flush=True)

    # ===== B: Gap Momentum =====
    print("  [B] Gap Momentum (延续, 日内)", flush=True)
    for gap_min in [0.003, 0.005, 0.008, 0.01, 0.015]:
        for filt in ['none', 'oi', 'vol', 'oi_vol']:
            for mp in [1, 2]:
                r = gap_backtest(NS, ND, dates, C, O, H, L, V, OI, syms,
                                 gap_min=gap_min, mode='momentum', max_pos=mp,
                                 alloc_pct=1.0, filter_mode=filt,
                                 oi_mom5=oi_mom5, vol_ratio=vol_ratio)
                if r:
                    r['desc'] = f"MOM_g{gap_min:.1%}"
                    r['filt'] = filt
                    r['mp'] = mp
                    results.append(r)
        print(f"    g{gap_min:.1%} done ({len(results)})", flush=True)

    # ===== C: Gap Fade + Swing =====
    print("  [C] Gap Fade + Swing", flush=True)
    for gap_min in [0.005, 0.008, 0.01, 0.015]:
        for hold in [3, 5, 7, 10]:
            for sl in [0.03, 0.05, 0.10]:
                for filt in ['none', 'oi_vol']:
                    r = gap_swing_backtest(NS, ND, dates, C, O, H, L, V, OI, syms,
                                            gap_min=gap_min, hold_days=hold,
                                            sl_pct=sl, filter_mode=filt,
                                            oi_mom5=oi_mom5, vol_ratio=vol_ratio)
                    if r:
                        r['desc'] = f"SWING_g{gap_min:.1%}"
                        r['filt'] = filt
                        r['mp'] = 1
                        r['hold'] = hold
                        r['sl'] = sl
                        results.append(r)
        print(f"    g{gap_min:.1%} done ({len(results)})", flush=True)

    print(f"\n  完成 ({time.time()-t0:.0f}s, {len(results)} configs)", flush=True)

    results.sort(key=lambda x: -x['ann'])
    print(f"\n{'='*80}", flush=True)
    print(f"  TOP 40", flush=True)
    hdr = f"  {'策略':<22s} {'Filt':<12s} {'MP':>2s} | {'Ann':>8s} {'N':>5s} {'WR':>5s} {'AvgW':>6s} {'AvgL':>5s} {'DD':>6s} {'Final':>12s}"
    print(hdr, flush=True)
    for r in results[:40]:
        extra = ''
        if 'hold' in r:
            extra = f" H{r['hold']}SL{r['sl']:.0%}"
        print(f"  {r['desc']:<22s} {r['filt']:<12s} P{r['mp']:>1d}{extra:>9s} | "
              f"{r['ann']:+8.1f}% {r['n']:5d} {r['wr']:5.1f}% {r['avg_win']:+6.2f}% {r['avg_loss']:5.2f}% {r['max_dd']:6.1f}% {r['final']:>12,.0f}", flush=True)

    for i, r in enumerate(results[:5]):
        print(f"\n  #{i+1}: {r['desc']} {r['filt']} P{r['mp']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%, WR={r['wr']:.0f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} t, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%, abs={s['pnl_abs_sum']:+,.0f}", flush=True)

    # 分策略类型统计
    print(f"\n  --- 按策略类型 ---", flush=True)
    for mode in ['FADE', 'MOM', 'SWING']:
        sub = [r for r in results if r['desc'].startswith(mode)]
        if sub:
            best = sub[0]
            print(f"  {mode}: Best={best['ann']:+.1f}% DD={best['max_dd']:.1f}% "
                  f"({best['desc']} {best['filt']} P{best['mp']})", flush=True)

    if results:
        print(f"\n  Best: {results[0]['ann']:+.1f}% DD={results[0]['max_dd']:.1f}%", flush=True)
    print(f"  目标: 年化600%+ | 基线: +69.4% (V8f Donchian)", flush=True)
    print(f"{'='*80}", flush=True)
