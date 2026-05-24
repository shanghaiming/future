"""
Alpha Futures V7 — 目标年化600% (无GAP策略)
============================================
不用缺口策略, 用趋势/动量/突破/OI:
  1. Donchian通道突破 (海龟交易)
  2. 均线交叉 + 放量确认
  3. 动量突破 (N日新高 + 放量)
  4. OI激增 + 趋势
  5. 连续上涨 + 放量加速
  6. 品种相对强弱轮动
  全仓进出, 复利, 目标600%年化
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, COMMISSION, STAMP_DUTY, CASH0

COMMODITY_GROUPS = {
    'ferrous':    ['rbfi', 'hci', 'ifi', 'jfi', 'jmfi'],
    'nonferrous': ['cufi', 'alfi', 'znfi', 'aufi', 'agfi', 'ni'],
    'energy':     ['scfi', 'mafi', 'ptafi', 'bufi', 'fufi', 'tai'],
    'agri':       ['afi', 'mfi', 'yfi', 'cfi', 'srfi', 'pfi', 'oi'],
    'chem':       ['ppfi', 'lfi', 'vfi', 'egfi', 'safi', 'fgfi'],
}


def backtest(buy_days, sell_days, NS, ND, dates, C, O, H, L, V, OI, syms,
             max_positions=1, sl_pct=0.10, hold_max=60):
    cash = float(CASH0)
    positions = []
    trades = []
    year_stats = {}

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        for pos in list(positions):
            si = pos['si']
            c = C[si, di]
            if np.isnan(c):
                continue
            pnl_pct = (c - pos['entry_price']) / pos['entry_price']

            if pnl_pct < -sl_pct:
                cash += pos['shares'] * c * (1 - COMMISSION)
                trades.append({'pnl': pnl_pct * 100, 'days': di - pos['entry_di'],
                               'di': di, 'reason': 'stop', 'year': year, 'si': si})
                positions.remove(pos)
                continue

            if di in sell_days.get(si, set()):
                cash += pos['shares'] * c * (1 - COMMISSION)
                trades.append({'pnl': pnl_pct * 100, 'days': di - pos['entry_di'],
                               'di': di, 'reason': 'signal', 'year': year, 'si': si})
                positions.remove(pos)
                continue

            if di - pos['entry_di'] >= hold_max:
                cash += pos['shares'] * c * (1 - COMMISSION)
                trades.append({'pnl': pnl_pct * 100, 'days': di - pos['entry_di'],
                               'di': di, 'reason': 'time', 'year': year, 'si': si})
                positions.remove(pos)

        if len(positions) < max_positions:
            candidates = []
            for si in range(NS):
                if di in buy_days.get(si, set()):
                    if any(p['si'] == si for p in positions):
                        continue
                    c = C[si, di]
                    if np.isnan(c) or c <= 0:
                        continue
                    d = di - 1
                    c5 = C[si, max(0, d-4):d+1]
                    valid_c5 = c5[~np.isnan(c5)]
                    mom5 = 0
                    if len(valid_c5) >= 3 and valid_c5[0] > 0:
                        mom5 = (valid_c5[-1] - valid_c5[0]) / valid_c5[0]
                    candidates.append((si, c, mom5))

            candidates.sort(key=lambda x: -x[2])
            slots = max_positions - len(positions)
            for si, price, _ in candidates[:slots]:
                alloc = cash / max(1, max_positions - len(positions))
                shares = int(alloc / (1 + COMMISSION) / price)
                if shares > 0 and shares * price * (1 + COMMISSION) <= cash:
                    cost = shares * price * (1 + COMMISSION)
                    cash -= cost
                    positions.append({
                        'si': si, 'entry_price': price, 'entry_di': di,
                        'shares': shares,
                    })

    for pos in positions:
        c = C[pos['si'], ND-1]
        if np.isnan(c) or c <= 0:
            c = pos['entry_price']
        pnl = (c - pos['entry_price']) / pos['entry_price'] * 100
        cash += pos['shares'] * c * (1 - COMMISSION)
        trades.append({'pnl': pnl, 'days': 999, 'di': ND-1, 'reason': 'end',
                       'year': dates[ND-1].year, 'si': pos['si']})

    if not trades:
        return None

    days_total = (dates[ND-1] - dates[MIN_TRAIN]).days
    yr = max(days_total / 365.25, 0.01)
    ann = ((cash / CASH0) ** (1 / yr) - 1) * 100
    nw = sum(1 for t in trades if t['pnl'] > 0)
    wr = nw / max(len(trades), 1) * 100

    for t in trades:
        y = t.get('year', 'unknown')
        if y not in year_stats:
            year_stats[y] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
        year_stats[y]['trades'] += 1
        if t['pnl'] > 0:
            year_stats[y]['wins'] += 1
        year_stats[y]['total_pnl'] += t['pnl']

    equity = float(CASH0)
    peak = float(CASH0)
    max_dd = 0
    for t in sorted(trades, key=lambda x: x['di']):
        equity *= (1 + t['pnl'] / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    return {
        'ann': round(ann, 1), 'n': len(trades), 'wr': round(wr, 1),
        'max_dd': round(max_dd, 1), 'final': round(cash, 0),
        'year_stats': year_stats,
    }


if __name__ == '__main__':
    print("=" * 80, flush=True)
    print("  Alpha Futures V7 — 目标600% (无GAP)", flush=True)
    print("=" * 80, flush=True)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    sym_idx = {s: i for i, s in enumerate(syms)}

    # MA预计算
    ma5 = np.full((NS, ND), np.nan)
    ma10 = np.full((NS, ND), np.nan)
    ma20 = np.full((NS, ND), np.nan)
    ma60 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for window, store in [(5, ma5), (10, ma10), (20, ma20), (60, ma60)]:
            for di in range(window, ND):
                vals = C[si, di-window:di]
                valid = vals[~np.isnan(vals)]
                if len(valid) >= window // 2:
                    store[si, di] = np.mean(valid)

    # =====================================================================
    # 信号生成 + 回测
    # =====================================================================
    results = []
    t0 = time.time()
    cfg_count = 0

    # ===== 1. Donchian通道突破 =====
    print("\n  [1] Donchian突破...", flush=True)
    for chan_len in [5, 10, 15, 20, 30, 40, 60]:
        for exit_len in [5, 10, 15, 20]:
            for hold in [10, 20, 30, 60]:
                buy_days = {si: set() for si in range(NS)}
                sell_days = {si: set() for si in range(NS)}
                for si in range(NS):
                    for di in range(chan_len + 1, ND):
                        d = di - 1
                        c = C[si, d]
                        if np.isnan(c):
                            continue
                        ch_slice = C[si, max(0, d-chan_len):d]
                        valid_ch = ch_slice[~np.isnan(ch_slice)]
                        if len(valid_ch) < chan_len // 2:
                            continue
                        high_ch = np.max(valid_ch)
                        # 突破新高
                        if c > high_ch:
                            buy_days[si].add(di)
                        # 跌破exit_len日低 → 平仓
                        ex_slice = C[si, max(0, d-exit_len):d]
                        valid_ex = ex_slice[~np.isnan(ex_slice)]
                        if len(valid_ex) >= exit_len // 2:
                            low_ex = np.min(valid_ex)
                            if c < low_ex:
                                sell_days[si].add(di)
                cfg_count += 1
                for max_pos in [1, 3, 5]:
                    for sl in [0.10, 0.20]:
                        r = backtest(buy_days, sell_days, NS, ND, dates, C, O, H, L, V, OI, syms,
                                    max_positions=max_pos, sl_pct=sl, hold_max=hold)
                        if r and r['ann'] > 0:
                            r['desc'] = f"DONCH_c{chan_len}_e{exit_len}_h{hold}"
                            r['max_pos'] = max_pos
                            r['sl'] = sl
                            results.append(r)

    # ===== 2. 动量突破 (N日新高 + 放量 + MA过滤) =====
    print("  [2] 动量突破...", flush=True)
    for new_high_w in [5, 10, 15, 20, 30]:
        for vol_thr in [1.0, 1.5, 2.0]:
            for ma_w in [20, 60]:
                for hold in [3, 5, 7, 10, 15]:
                    buy_days = {si: set() for si in range(NS)}
                    sell_days = {si: set() for si in range(NS)}
                    ma_store = ma20 if ma_w == 20 else ma60
                    for si in range(NS):
                        for di in range(max(new_high_w, ma_w) + 1, ND):
                            d = di - 1
                            c = C[si, d]
                            v = V[si, d]
                            if np.isnan(c) or np.isnan(v) or v <= 0:
                                continue
                            # N日新高
                            ch_slice = C[si, max(0, d-new_high_w):d]
                            valid_ch = ch_slice[~np.isnan(ch_slice)]
                            if len(valid_ch) < new_high_w // 2:
                                continue
                            if c <= np.max(valid_ch):
                                continue
                            # 放量
                            v_window = V[si, max(0, d-19):d+1]
                            valid_v = v_window[~np.isnan(v_window)]
                            if len(valid_v) < 10:
                                continue
                            v_avg = np.mean(valid_v)
                            if v < vol_thr * v_avg:
                                continue
                            # MA过滤
                            if np.isnan(ma_store[si, d]) or c < ma_store[si, d]:
                                continue
                            buy_days[si].add(di)
                            sell_di = di + hold
                            if sell_di < ND:
                                sell_days[si].add(sell_di)
                    cfg_count += 1
                    for max_pos in [1, 3, 5]:
                        for sl in [0.10, 0.20]:
                            r = backtest(buy_days, sell_days, NS, ND, dates, C, O, H, L, V, OI, syms,
                                        max_positions=max_pos, sl_pct=sl, hold_max=hold + 30)
                            if r and r['ann'] > 0:
                                r['desc'] = f"NEWH_h{new_high_w}_v{vol_thr*10:.0f}_MA{ma_w}_h{hold}"
                                r['max_pos'] = max_pos
                                r['sl'] = sl
                                results.append(r)

    # ===== 3. OI激增 + 趋势 =====
    print("  [3] OI激增...", flush=True)
    for oi_mult in [1.2, 1.5, 2.0, 2.5]:
        for hold in [5, 10, 15, 20, 30]:
            buy_days = {si: set() for si in range(NS)}
            sell_days = {si: set() for si in range(NS)}
            for si in range(NS):
                for di in range(22, ND):
                    d = di - 1
                    oi = OI[si, d]
                    if np.isnan(oi) or oi <= 0:
                        continue
                    oi_window = OI[si, d-19:d+1]
                    valid_oi = oi_window[~np.isnan(oi_window)]
                    if len(valid_oi) < 10:
                        continue
                    oi_avg = np.mean(valid_oi)
                    if oi < oi_mult * oi_avg:
                        continue
                    c = C[si, d]
                    if np.isnan(c):
                        continue
                    # 价格在MA20之上 + 上涨
                    if np.isnan(ma20[si, d]) or c < ma20[si, d]:
                        continue
                    c_prev = C[si, d-1] if d > 0 else np.nan
                    if not np.isnan(c_prev) and c_prev > 0 and c > c_prev:
                        buy_days[si].add(di)
                        sell_di = di + hold
                        if sell_di < ND:
                            sell_days[si].add(sell_di)
            cfg_count += 1
            for max_pos in [1, 3, 5]:
                for sl in [0.10, 0.20]:
                    r = backtest(buy_days, sell_days, NS, ND, dates, C, O, H, L, V, OI, syms,
                                max_positions=max_pos, sl_pct=sl, hold_max=hold + 30)
                    if r and r['ann'] > 0:
                        r['desc'] = f"OI_SURGE_m{oi_mult*10:.0f}_h{hold}"
                        r['max_pos'] = max_pos
                        r['sl'] = sl
                        results.append(r)

    # ===== 4. 连续上涨 + 放量 + OI =====
    print("  [4] 连涨加速...", flush=True)
    for up_days in [2, 3, 4, 5]:
        for hold in [3, 5, 7, 10]:
            for oi_filter in [False, True]:
                buy_days = {si: set() for si in range(NS)}
                sell_days = {si: set() for si in range(NS)}
                for si in range(NS):
                    for di in range(up_days + 1, ND):
                        d = di - 1
                        all_up = True
                        total_ret = 0
                        for k in range(up_days):
                            if d - k < 1:
                                all_up = False
                                break
                            c_now = C[si, d-k]
                            c_prev = C[si, d-k-1]
                            if np.isnan(c_now) or np.isnan(c_prev) or c_prev <= 0:
                                all_up = False
                                break
                            if c_now <= c_prev:
                                all_up = False
                                break
                            total_ret += (c_now - c_prev) / c_prev
                        if not all_up:
                            continue
                        # 放量
                        v = V[si, d]
                        v_window = V[si, max(0, d-19):d+1]
                        valid_v = v_window[~np.isnan(v_window)]
                        if len(valid_v) < 10:
                            continue
                        v_avg = np.mean(valid_v)
                        if v < 1.5 * v_avg:
                            continue
                        # OI
                        if oi_filter:
                            oi = OI[si, d]
                            oi_5ago = OI[si, d-5] if d >= 5 else np.nan
                            if np.isnan(oi) or np.isnan(oi_5ago) or oi <= oi_5ago:
                                continue
                        buy_days[si].add(di)
                        sell_di = di + hold
                        if sell_di < ND:
                            sell_days[si].add(sell_di)
                cfg_count += 1
                for max_pos in [1, 3, 5]:
                    for sl in [0.10, 0.20]:
                        r = backtest(buy_days, sell_days, NS, ND, dates, C, O, H, L, V, OI, syms,
                                    max_positions=max_pos, sl_pct=sl, hold_max=hold + 20)
                        if r and r['ann'] > 0:
                            oi_tag = "_OI" if oi_filter else ""
                            r['desc'] = f"STREAK_u{up_days}{oi_tag}_h{hold}"
                            r['max_pos'] = max_pos
                            r['sl'] = sl
                            results.append(r)

    # ===== 5. 均线交叉 + 放量 =====
    print("  [5] 均线交叉...", flush=True)
    for fast in [3, 5, 10]:
        for slow in [20, 30, 60]:
            if fast >= slow:
                continue
            for vol_thr in [1.0, 1.5]:
                for hold in [5, 10, 20]:
                    buy_days = {si: set() for si in range(NS)}
                    sell_days = {si: set() for si in range(NS)}
                    for si in range(NS):
                        for di in range(slow + 1, ND):
                            d = di - 1
                            # 快慢均线
                            fast_slice = C[si, max(0, d-fast):d+1]
                            slow_slice = C[si, max(0, d-slow):d+1]
                            valid_f = fast_slice[~np.isnan(fast_slice)]
                            valid_s = slow_slice[~np.isnan(slow_slice)]
                            if len(valid_f) < fast // 2 or len(valid_s) < slow // 2:
                                continue
                            ma_fast = np.mean(valid_f)
                            ma_slow = np.mean(valid_s)
                            # 前一天
                            fast_prev = C[si, max(0, d-1-fast):d]
                            slow_prev = C[si, max(0, d-1-slow):d]
                            valid_fp = fast_prev[~np.isnan(fast_prev)]
                            valid_sp = slow_prev[~np.isnan(slow_prev)]
                            if len(valid_fp) < fast // 2 or len(valid_sp) < slow // 2:
                                continue
                            ma_fast_prev = np.mean(valid_fp)
                            ma_slow_prev = np.mean(valid_sp)
                            # 金叉
                            if ma_fast > ma_slow and ma_fast_prev <= ma_slow_prev:
                                v = V[si, d]
                                v_window = V[si, max(0, d-19):d+1]
                                valid_v = v_window[~np.isnan(v_window)]
                                if len(valid_v) < 10:
                                    continue
                                v_avg = np.mean(valid_v)
                                if v < vol_thr * v_avg:
                                    continue
                                buy_days[si].add(di)
                                sell_di = di + hold
                                if sell_di < ND:
                                    sell_days[si].add(sell_di)
                    cfg_count += 1
                    for max_pos in [1, 3, 5]:
                        for sl in [0.10, 0.20]:
                            r = backtest(buy_days, sell_days, NS, ND, dates, C, O, H, L, V, OI, syms,
                                        max_positions=max_pos, sl_pct=sl, hold_max=hold + 30)
                            if r and r['ann'] > 0:
                                r['desc'] = f"MACROSS_{fast}/{slow}_v{vol_thr*10:.0f}_h{hold}"
                                r['max_pos'] = max_pos
                                r['sl'] = sl
                                results.append(r)

    print(f"\n  Done ({time.time()-t0:.0f}s, {cfg_count} configs, {len(results)} profitable)", flush=True)

    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    print(f"\n{'='*80}", flush=True)
    print(f"  TOP 50", flush=True)
    print(f"  {'策略':<40s} {'P':>2s} {'SL':>3s} | {'Ann':>8s} {'N':>5s} {'WR':>5s} {'DD':>5s}", flush=True)
    print(f"  {'-'*80}", flush=True)
    for r in results[:50]:
        print(f"  {r['desc']:<40s} P{r['max_pos']} SL{r['sl']:.0%} | "
              f"{r['ann']:+8.1f}% {r['n']:5d} {r['wr']:5.1f}% {r['max_dd']:5.1f}%", flush=True)

    # 按类型
    print(f"\n  === Best per type ===", flush=True)
    type_best = {}
    for r in results:
        t = r['desc'].split('_')[0]
        if t not in type_best or r['ann'] > type_best[t]['ann']:
            type_best[t] = r
    for t, r in sorted(type_best.items(), key=lambda x: -x[1]['ann']):
        print(f"    {t:<12s}: {r['desc']} P{r['max_pos']} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}%",
              flush=True)

    # Top 5 year-by-year
    print(f"\n  === TOP 5 YEAR-BY-YEAR ===", flush=True)
    for i, r in enumerate(results[:5]):
        print(f"\n  #{i+1}: {r['desc']} P{r['max_pos']} SL{r['sl']:.0%} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%",
                  flush=True)

    # >=600%
    high = [r for r in results if r['ann'] >= 600]
    if high:
        print(f"\n  *** >= 600%: {len(high)} strategies ***", flush=True)
        for r in high[:20]:
            print(f"    {r['desc']:<40s} P{r['max_pos']} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}%",
                  flush=True)
    else:
        top3 = results[:3]
        print(f"\n  Best: {top3[0]['ann']:+.1f}% (target 600%)", flush=True)

    print(f"\n{'='*80}", flush=True)
