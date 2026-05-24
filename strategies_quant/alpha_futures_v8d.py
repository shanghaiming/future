"""
Alpha Futures V8f — OI增强 Donchian 无杠杆回测
===============================================
lots = cash / (price × multiplier)  — 全款买入
P&L = (exit - entry) × multiplier × lots × direction
不加杠杆，不用保证金
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, COMMISSION, STAMP_DUTY, CASH0

# 合约乘数 (每手多少吨/克/桶) — 只是单位, 不是杠杆
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


def backtest(buy_days, sell_days, short_days, cover_days,
             NS, ND, dates, C, O, H, L, V, OI, syms,
             max_positions=1, sl_pct=0.10, hold_max=60):
    """
    无杠杆期货回测:
    lots = cash / (price × multiplier) — 全款买
    P&L = (exit - entry) × multiplier × lots
    """
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

            # 止损
            if pnl_pct / 100 < -sl_pct:
                cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                               'days': di - pos['entry_di'], 'di': di,
                               'reason': 'stop', 'year': year, 'si': si, 'dir': pos['dir']})
                positions.remove(pos)
                continue

            if pos['dir'] == 1 and di in sell_days.get(si, set()):
                cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                               'days': di - pos['entry_di'], 'di': di,
                               'reason': 'signal', 'year': year, 'si': si, 'dir': pos['dir']})
                positions.remove(pos)
                continue
            if pos['dir'] == -1 and di in cover_days.get(si, set()):
                cash += c * mult * pos['lots'] * (1 - COMM_RATE)
                trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                               'days': di - pos['entry_di'], 'di': di,
                               'reason': 'signal', 'year': year, 'si': si, 'dir': pos['dir']})
                positions.remove(pos)
                continue

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
                if di in buy_days.get(si, set()):
                    if any(p['si'] == si for p in positions):
                        continue
                    c = C[si, di]
                    if np.isnan(c) or c <= 0:
                        continue
                    candidates.append((si, c, 1, syms[si]))
            for si in range(NS):
                if di in short_days.get(si, set()):
                    if any(p['si'] == si for p in positions):
                        continue
                    c = C[si, di]
                    if np.isnan(c) or c <= 0:
                        continue
                    candidates.append((si, c, -1, syms[si]))

            def _mom(x):
                si, _, _, _ = x
                d = di - 1
                c5 = C[si, max(0, d-4):d+1]
                v = c5[~np.isnan(c5)]
                return ((v[-1] - v[0]) / v[0] if len(v) >= 2 and v[0] > 0 else 0) * x[2]
            candidates.sort(key=_mom, reverse=True)

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
                        positions.append({
                            'si': si, 'entry': price, 'entry_di': di,
                            'lots': lots, 'dir': direction, 'sym': sym,
                        })

    # 清仓
    for pos in positions:
        c = C[pos['si'], ND-1]
        if np.isnan(c) or c <= 0:
            c = pos['entry']
        mult = MULT.get(pos['sym'], DEF_MULT)
        if pos['dir'] == 1:
            pnl = (c - pos['entry']) * mult * pos['lots']
        else:
            pnl = (pos['entry'] - c) * mult * pos['lots']
        pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100
        cash += c * mult * pos['lots'] * (1 - COMM_RATE)
        trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                       'days': 999, 'di': ND-1, 'reason': 'end',
                       'year': dates[ND-1].year, 'si': pos['si'], 'dir': pos['dir']})

    if not trades:
        return None

    # equity curve
    equity = float(CASH0)
    peak = float(CASH0)
    max_dd = 0
    total_pnl_abs = 0
    for t in sorted(trades, key=lambda x: x['di']):
        equity += t['pnl_abs']
        total_pnl_abs += t['pnl_abs']
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    final_cash = cash
    if final_cash <= 0:
        return None

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
        if t['pnl_abs'] > 0:
            year_stats[y]['wins'] += 1
        year_stats[y]['total_pnl'] += t['pnl_pct']
        year_stats[y]['pnl_abs_sum'] += t['pnl_abs']

    return {
        'ann': round(ann, 1), 'n': len(trades), 'wr': round(wr, 1),
        'max_dd': round(max_dd, 1), 'final': round(final_cash, 0),
        'year_stats': year_stats,
    }


if __name__ == '__main__':
    print("=" * 80, flush=True)
    print("  Alpha Futures V8f — OI增强 Donchian 无杠杆", flush=True)
    print("  lots = cash / (price × mult), P&L = Δprice × mult × lots", flush=True)
    print("=" * 80, flush=True)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    # 预计算因子
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

    # OI增强Donchian
    for chan_len in [5, 7, 10, 15, 20, 30]:
        for exit_len in [5, 10, 15, 20]:
            buy_d = {si: set() for si in range(NS)}
            sell_d = {si: set() for si in range(NS)}
            short_d = {si: set() for si in range(NS)}
            cover_d = {si: set() for si in range(NS)}
            for si in range(NS):
                for di in range(chan_len + 1, ND):
                    d = di - 1
                    c = C[si, d]
                    if np.isnan(c):
                        continue
                    ch = C[si, max(0, d-chan_len):d]
                    vc = ch[~np.isnan(ch)]
                    if len(vc) < chan_len // 2:
                        continue
                    high_ch, low_ch = np.max(vc), np.min(vc)

                    # OI+量增强
                    if c > high_ch:
                        om = oi_mom5[si, di]
                        vr = vol_ratio[si, di]
                        if (not np.isnan(om) and om > 0 and
                            not np.isnan(vr) and vr > 1.0):
                            buy_d[si].add(di)
                    if c < low_ch:
                        om = oi_mom5[si, di]
                        vr = vol_ratio[si, di]
                        if (not np.isnan(om) and om > 0 and
                            not np.isnan(vr) and vr > 1.0):
                            short_d[si].add(di)

                    ex = C[si, max(0, d-exit_len):d]
                    vex = ex[~np.isnan(ex)]
                    if len(vex) >= max(1, exit_len // 2):
                        if c < np.min(vex): sell_d[si].add(di)
                        if c > np.max(vex): cover_d[si].add(di)

            for max_pos in [1, 2, 3, 5]:
                for sl in [0.05, 0.10, 0.20, 0.50]:
                    for hm in [30, 45, 60]:
                        r = backtest(buy_d, sell_d, short_d, cover_d,
                                    NS, ND, dates, C, O, H, L, V, OI, syms,
                                    max_positions=max_pos, sl_pct=sl, hold_max=hm)
                        if r and r['ann'] > 0:
                            r['desc'] = f"OI_c{chan_len}_e{exit_len}"
                            r['max_pos'] = max_pos
                            r['sl'] = sl
                            r['hm'] = hm
                            results.append(r)
            print(f"    c{chan_len}_e{exit_len} done ({len(results)})", flush=True)

    # 普通Donchian (无OI过滤) 作为对比
    for chan_len in [10, 15, 20]:
        for exit_len in [5, 10, 15]:
            buy_d = {si: set() for si in range(NS)}
            sell_d = {si: set() for si in range(NS)}
            short_d = {si: set() for si in range(NS)}
            cover_d = {si: set() for si in range(NS)}
            for si in range(NS):
                for di in range(chan_len + 1, ND):
                    d = di - 1
                    c = C[si, d]
                    if np.isnan(c):
                        continue
                    ch = C[si, max(0, d-chan_len):d]
                    vc = ch[~np.isnan(ch)]
                    if len(vc) < chan_len // 2:
                        continue
                    high_ch, low_ch = np.max(vc), np.min(vc)
                    if c > high_ch: buy_d[si].add(di)
                    if c < low_ch: short_d[si].add(di)
                    ex = C[si, max(0, d-exit_len):d]
                    vex = ex[~np.isnan(ex)]
                    if len(vex) >= max(1, exit_len // 2):
                        if c < np.min(vex): sell_d[si].add(di)
                        if c > np.max(vex): cover_d[si].add(di)
            for max_pos in [1, 2, 3]:
                for sl in [0.10, 0.20]:
                    r = backtest(buy_d, sell_d, short_d, cover_d,
                                NS, ND, dates, C, O, H, L, V, OI, syms,
                                max_positions=max_pos, sl_pct=sl, hold_max=60)
                    if r and r['ann'] > 0:
                        r['desc'] = f"DONCH_c{chan_len}_e{exit_len}"
                        r['max_pos'] = max_pos
                        r['sl'] = sl
                        r['hm'] = 60
                        results.append(r)
            print(f"    DONCH_c{chan_len} done ({len(results)})", flush=True)

    print(f"\n  完成 ({time.time()-t0:.0f}s, {len(results)} profitable)", flush=True)

    results.sort(key=lambda x: -x['ann'])
    print(f"\n{'='*80}", flush=True)
    print(f"  TOP 30", flush=True)
    print(f"  {'策略':<22s} {'P':>2s} {'SL':>4s} {'HM':>3s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'DD':>5s} {'Final':>12s}", flush=True)
    for r in results[:30]:
        print(f"  {r['desc']:<22s} P{r['max_pos']} SL{r['sl']:.0%} H{r['hm']:>2d} | "
              f"{r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% {r['max_dd']:5.1f}% {r['final']:>12,.0f}", flush=True)

    for i, r in enumerate(results[:5]):
        print(f"\n  #{i+1}: {r['desc']} P{r['max_pos']} SL{r['sl']:.0%} H{r['hm']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} t, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%, abs={s['pnl_abs_sum']:+,.0f}", flush=True)

    if results:
        print(f"\n  Best: {results[0]['ann']:+.1f}% DD={results[0]['max_dd']:.1f}%", flush=True)
    print(f"{'='*80}", flush=True)
