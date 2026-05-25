"""
Alpha Futures V9b — 高频OI突破 (无杠杆)
=========================================
策略: 短周期Donchian + OI确认 + 量确认
核心: P1集中持仓, 快速周转, 只选最强信号
迭代目标: 年化600%+
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


def backtest(buy_d, sell_d, short_d, cover_d,
             NS, ND, dates, C, O, H, L, V, OI, syms,
             max_positions=1, sl_pct=0.05, hold_max=30,
             ranking='mom5', vol_filter=False, atr_stop_mult=0):
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
            if np.isnan(c):
                continue
            mult = MULT.get(pos['sym'], DEF_MULT)
            if pos['dir'] == 1:
                pnl = (c - pos['entry']) * mult * pos['lots']
            else:
                pnl = (pos['entry'] - c) * mult * pos['lots']
            pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100

            # ATR追踪止损
            if atr_stop_mult > 0 and pos.get('trail_stop'):
                if pos['dir'] == 1 and c < pos['trail_stop']:
                    cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                    trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                                   'days': di - pos['entry_di'], 'di': di,
                                   'reason': 'trail', 'year': year, 'si': si, 'dir': pos['dir']})
                    positions.remove(pos)
                    continue
                elif pos['dir'] == -1 and c > pos['trail_stop']:
                    cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                    trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                                   'days': di - pos['entry_di'], 'di': di,
                                   'reason': 'trail', 'year': year, 'si': si, 'dir': pos['dir']})
                    positions.remove(pos)
                    continue
                # 更新追踪止损
                if pos['dir'] == 1:
                    atr = pos.get('atr', 0)
                    if atr > 0:
                        new_stop = c - atr_stop_mult * atr
                        if new_stop > pos['trail_stop']:
                            pos['trail_stop'] = new_stop
                else:
                    atr = pos.get('atr', 0)
                    if atr > 0:
                        new_stop = c + atr_stop_mult * atr
                        if new_stop < pos['trail_stop']:
                            pos['trail_stop'] = new_stop

            # 固定止损
            if pnl_pct / 100 < -sl_pct:
                cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                               'days': di - pos['entry_di'], 'di': di,
                               'reason': 'stop', 'year': year, 'si': si, 'dir': pos['dir']})
                positions.remove(pos)
                continue

            # 信号平仓
            if pos['dir'] == 1 and di in sell_d.get(si, set()):
                cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                               'days': di - pos['entry_di'], 'di': di,
                               'reason': 'signal', 'year': year, 'si': si, 'dir': pos['dir']})
                positions.remove(pos)
                continue
            if pos['dir'] == -1 and di in cover_d.get(si, set()):
                cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                               'days': di - pos['entry_di'], 'di': di,
                               'reason': 'signal', 'year': year, 'si': si, 'dir': pos['dir']})
                positions.remove(pos)
                continue

            # 超时
            if di - pos['entry_di'] >= hold_max:
                cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                               'days': di - pos['entry_di'], 'di': di,
                               'reason': 'time', 'year': year, 'si': si, 'dir': pos['dir']})
                positions.remove(pos)

        # 开仓
        if len(positions) < max_positions:
            candidates = []
            for si in range(NS):
                if di in buy_d.get(si, set()):
                    if any(p['si'] == si for p in positions):
                        continue
                    c = C[si, di]
                    if np.isnan(c) or c <= 0:
                        continue
                    candidates.append((si, c, 1, syms[si]))
            for si in range(NS):
                if di in short_d.get(si, set()):
                    if any(p['si'] == si for p in positions):
                        continue
                    c = C[si, di]
                    if np.isnan(c) or c <= 0:
                        continue
                    candidates.append((si, c, -1, syms[si]))

            if candidates:
                # 排序
                def _rank(x):
                    si, _, d, _ = x
                    dd = di - 1
                    if ranking == 'mom5':
                        c5 = C[si, max(0, dd-4):dd+1]
                        v = c5[~np.isnan(c5)]
                        return ((v[-1] - v[0]) / v[0] if len(v) >= 2 and v[0] > 0 else 0) * d
                    elif ranking == 'mom10':
                        c10 = C[si, max(0, dd-9):dd+1]
                        v = c10[~np.isnan(c10)]
                        return ((v[-1] - v[0]) / v[0] if len(v) >= 2 and v[0] > 0 else 0) * d
                    elif ranking == 'vol_ratio':
                        v20 = V[si, max(0, dd-19):dd+1]
                        vv = v20[~np.isnan(v20)]
                        return (V[si, dd] / np.mean(vv) if len(vv) >= 10 and not np.isnan(V[si, dd]) else 0) * d
                    elif ranking == 'oi_mom':
                        oi5 = OI[si, max(0, dd-4):dd+1]
                        ov = oi5[~np.isnan(oi5)]
                        return ((ov[-1] - ov[0]) / ov[0] if len(ov) >= 2 and ov[0] > 0 else 0) * d
                    return 0

                candidates.sort(key=_rank, reverse=True)

                slots = max_positions - len(positions)
                for si, price, direction, sym in candidates[:slots]:
                    mult = MULT.get(sym, DEF_MULT)
                    notional_per_lot = price * mult
                    if notional_per_lot <= 0:
                        continue
                    alloc = cash / max(1, max_positions - len(positions))
                    lots = int(alloc / notional_per_lot)
                    if lots > 0:
                        cost = notional_per_lot * lots * (1 + COMM_RATE)
                        if cost <= cash:
                            cash -= cost
                            # ATR
                            atr = 0
                            if di >= 11:
                                trs = []
                                for dd2 in range(max(1, di-10), di):
                                    hi = H[si, dd2]; lo = L[si, dd2]; pc = C[si, dd2-1]
                                    if np.isnan(hi) or np.isnan(lo): continue
                                    tr = hi - lo
                                    if not np.isnan(pc): tr = max(tr, abs(hi-pc), abs(lo-pc))
                                    trs.append(tr)
                                if trs: atr = np.mean(trs)
                            trail = 0
                            if atr > 0 and atr_stop_mult > 0:
                                if direction == 1:
                                    trail = price - atr_stop_mult * atr
                                else:
                                    trail = price + atr_stop_mult * atr
                            positions.append({
                                'si': si, 'entry': price, 'entry_di': di,
                                'lots': lots, 'dir': direction, 'sym': sym,
                                'atr': atr, 'trail_stop': trail,
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

    equity = float(CASH0); peak = float(CASH0); max_dd = 0; total_pnl = 0
    for t in sorted(trades, key=lambda x: x['di']):
        equity += t['pnl_abs']; total_pnl += t['pnl_abs']
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
    avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_abs'] > 0]) if nw > 0 else 0
    avg_loss = np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_abs'] <= 0]) if nw < len(trades) else 0

    for t in trades:
        y = t.get('year', 'unknown')
        if y not in year_stats: year_stats[y] = {'trades': 0, 'wins': 0, 'total_pnl': 0, 'pnl_abs_sum': 0}
        year_stats[y]['trades'] += 1
        if t['pnl_abs'] > 0: year_stats[y]['wins'] += 1
        year_stats[y]['total_pnl'] += t['pnl_pct']
        year_stats[y]['pnl_abs_sum'] += t['pnl_abs']

    return {
        'ann': round(ann, 1), 'n': len(trades), 'wr': round(wr, 1),
        'max_dd': round(max_dd, 1), 'final': round(final_cash, 0),
        'avg_win': round(avg_win, 1), 'avg_loss': round(avg_loss, 1),
        'year_stats': year_stats,
    }


if __name__ == '__main__':
    print("=" * 80, flush=True)
    print("  Alpha Futures V9b — 高频OI突破 (无杠杆)", flush=True)
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

    # 策略A: OI增强Donchian — 多种参数组合
    print("  [A] OI增强Donchian", flush=True)
    for chan in [3, 5, 7, 10, 15]:
        for exit_l in [3, 5, 7, 10, chan]:
            buy_d = {si: set() for si in range(NS)}
            sell_d = {si: set() for si in range(NS)}
            short_d = {si: set() for si in range(NS)}
            cover_d = {si: set() for si in range(NS)}
            for si in range(NS):
                for di in range(chan + 1, ND):
                    d = di - 1
                    c = C[si, d]
                    if np.isnan(c): continue
                    ch = C[si, max(0, d-chan):d]
                    vc = ch[~np.isnan(ch)]
                    if len(vc) < max(1, chan // 3): continue
                    hch, lch = np.max(vc), np.min(vc)
                    # OI+量过滤
                    om = oi_mom5[si, di]
                    vr = vol_ratio[si, di]
                    oi_ok = not np.isnan(om) and om > 0
                    vol_ok = not np.isnan(vr) and vr > 1.0
                    if c > hch and oi_ok and vol_ok:
                        buy_d[si].add(di)
                    if c < lch and oi_ok and vol_ok:
                        short_d[si].add(di)
                    ex = C[si, max(0, d-exit_l):d]
                    vex = ex[~np.isnan(ex)]
                    if len(vex) >= max(1, exit_l // 2):
                        if c < np.min(vex): sell_d[si].add(di)
                        if c > np.max(vex): cover_d[si].add(di)

            for mp in [1, 2]:
                for sl in [0.03, 0.05, 0.10]:
                    for hm in [10, 15, 20, 30, 45]:
                        for rank in ['mom5', 'mom10']:
                            r = backtest(buy_d, sell_d, short_d, cover_d,
                                        NS, ND, dates, C, O, H, L, V, OI, syms,
                                        max_positions=mp, sl_pct=sl, hold_max=hm,
                                        ranking=rank, atr_stop_mult=0)
                            if r and r['ann'] > 50:
                                r['desc'] = f"OI_c{chan}_e{exit_l}"
                                r['mp'] = mp; r['sl'] = sl; r['hm'] = hm; r['rank'] = rank
                                results.append(r)
            print(f"    c{chan} done", flush=True)

    # 策略B: 纯Donchian (无OI过滤) — 对比
    print("  [B] 纯Donchian", flush=True)
    for chan in [5, 10, 15, 20]:
        for exit_l in [5, 10, chan]:
            buy_d = {si: set() for si in range(NS)}
            sell_d = {si: set() for si in range(NS)}
            short_d = {si: set() for si in range(NS)}
            cover_d = {si: set() for si in range(NS)}
            for si in range(NS):
                for di in range(chan + 1, ND):
                    d = di - 1
                    c = C[si, d]
                    if np.isnan(c): continue
                    ch = C[si, max(0, d-chan):d]
                    vc = ch[~np.isnan(ch)]
                    if len(vc) < max(1, chan // 3): continue
                    hch, lch = np.max(vc), np.min(vc)
                    if c > hch: buy_d[si].add(di)
                    if c < lch: short_d[si].add(di)
                    ex = C[si, max(0, d-exit_l):d]
                    vex = ex[~np.isnan(ex)]
                    if len(vex) >= max(1, exit_l // 2):
                        if c < np.min(vex): sell_d[si].add(di)
                        if c > np.max(vex): cover_d[si].add(di)
            for mp in [1, 2]:
                for sl in [0.03, 0.05, 0.10]:
                    for hm in [15, 30, 45]:
                        r = backtest(buy_d, sell_d, short_d, cover_d,
                                    NS, ND, dates, C, O, H, L, V, OI, syms,
                                    max_positions=mp, sl_pct=sl, hold_max=hm, ranking='mom5')
                        if r and r['ann'] > 50:
                            r['desc'] = f"DONCH_c{chan}_e{exit_l}"
                            r['mp'] = mp; r['sl'] = sl; r['hm'] = hm; r['rank'] = 'mom5'
                            results.append(r)
        print(f"    c{chan} done", flush=True)

    # 策略C: 连涨/连跌反转 + OI确认
    print("  [C] 连涨连跌反转", flush=True)
    for streak in [3, 4, 5]:
        buy_d = {si: set() for si in range(NS)}
        sell_d = {si: set() for si in range(NS)}
        short_d = {si: set() for si in range(NS)}
        cover_d = {si: set() for si in range(NS)}
        for si in range(NS):
            for di in range(streak + 1, ND):
                # 检查连续下跌
                downs = 0
                for k in range(streak):
                    d = di - 1 - k
                    if C[si, d] < C[si, d-1]:
                        downs += 1
                if downs >= streak:  # 连续下跌streak天
                    om = oi_mom5[si, di]
                    vr = vol_ratio[si, di]
                    if not np.isnan(om) and om > 0 and not np.isnan(vr) and vr > 1.2:
                        buy_d[si].add(di)
                # 检查连续上涨
                ups = 0
                for k in range(streak):
                    d = di - 1 - k
                    if C[si, d] > C[si, d-1]:
                        ups += 1
                if ups >= streak:
                    om = oi_mom5[si, di]
                    vr = vol_ratio[si, di]
                    if not np.isnan(om) and om > 0 and not np.isnan(vr) and vr > 1.2:
                        short_d[si].add(di)
        for hold in [3, 5, 7]:
            for si in range(NS):
                for d in list(buy_d[si]):
                    sell_d[si].add(d + hold)
                for d in list(short_d[si]):
                    cover_d[si].add(d + hold)
            for mp in [1, 2, 3]:
                for sl in [0.03, 0.05]:
                    r = backtest(buy_d, sell_d, short_d, cover_d,
                                NS, ND, dates, C, O, H, L, V, OI, syms,
                                max_positions=mp, sl_pct=sl, hold_max=hold + 10, ranking='mom5')
                    if r and r['ann'] > 50:
                        r['desc'] = f"STREAK_{streak}_h{hold}"
                        r['mp'] = mp; r['sl'] = sl; r['hm'] = hold + 10; r['rank'] = 'mom5'
                        results.append(r)
        print(f"    streak={streak} done", flush=True)

    print(f"\n  完成 ({time.time()-t0:.0f}s, {len(results)} >50%)", flush=True)

    results.sort(key=lambda x: -x['ann'])
    print(f"\n{'='*80}", flush=True)
    print(f"  TOP 30 (只显示年化>50%)", flush=True)
    print(f"  {'策略':<22s} {'P':>2s} {'SL':>4s} {'H':>3s} {'Rank':>5s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'AvgW':>5s} {'AvgL':>5s} {'DD':>5s}", flush=True)
    for r in results[:30]:
        print(f"  {r['desc']:<22s} P{r['mp']} SL{r['sl']:.0%} H{r['hm']:>2d} {r['rank']:>5s} | "
              f"{r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% {r['avg_win']:+5.1f}% {r['avg_loss']:5.1f}% {r['max_dd']:5.1f}%", flush=True)

    for i, r in enumerate(results[:5]):
        print(f"\n  #{i+1}: {r['desc']} P{r['mp']} SL{r['sl']:.0%} H{r['hm']} {r['rank']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%, WR={r['wr']:.0f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} t, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    if results:
        print(f"\n  Best: {results[0]['ann']:+.1f}% DD={results[0]['max_dd']:.1f}%", flush=True)
    print(f"  目标: 年化600%+ | 当前基线: +69.4% (V8f OI_c5_e20 P1)", flush=True)
    print(f"{'='*80}", flush=True)
