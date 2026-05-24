"""
Alpha Futures V6 — 冲高收益
============================
V5教训: 过度过滤+dd暂停把收益砍到12.6%, 用户不满意

V6策略: 优化参数 + 新策略, 目标>40%年化
  1. GAP_FOLLOW参数精调 (缺口阈值/成交量阈值/持仓天数)
  2. VOL_BREAK_OI恢复V2原始退出逻辑
  3. 新策略: Donchian突破 (海龟交易法)
  4. 新策略: 双动量 (绝对+相对)
  5. 新策略: 缺口反转 (大缺口后的反转交易)
  6. 专注收益, 不过度约束
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


def generate_signals(NS, ND, C, O, H, L, V, OI, syms):
    sym_idx = {s: i for i, s in enumerate(syms)}
    sym_group = {}
    for gname, gsyms in COMMODITY_GROUPS.items():
        for s in gsyms:
            sym_group[s] = gname

    # 组动量
    group_mom = {}
    for di in range(21, ND):
        gm = {}
        for gname, gsyms in COMMODITY_GROUPS.items():
            rets = []
            for s in gsyms:
                si = sym_idx.get(s)
                if si is None:
                    continue
                d = di - 1
                c_now, c_prev = C[si, d], C[si, d-20]
                if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                    rets.append((c_now - c_prev) / c_prev)
            if rets:
                gm[gname] = np.mean(rets)
        group_mom[di] = gm

    # MA
    ma10 = np.full((NS, ND), np.nan)
    ma20 = np.full((NS, ND), np.nan)
    ma60 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for window, store in [(10, ma10), (20, ma20), (60, ma60)]:
            for di in range(window, ND):
                vals = C[si, di-window:di]
                valid = vals[~np.isnan(vals)]
                if len(valid) >= window // 2:
                    store[si, di] = np.mean(valid)

    signals = {}

    # ===== 1. GAP_FOLLOW 参数变体 =====
    for gap_thr in [0.005, 0.01, 0.015, 0.02]:
        for vol_thr in [1.0, 1.5, 2.0]:
            for hold in [3, 5, 7, 10]:
                buy_days = {si: set() for si in range(NS)}
                sell_days = {si: set() for si in range(NS)}
                for si in range(NS):
                    for di in range(2, ND):
                        d = di - 1
                        o = O[si, d]
                        c_prev = C[si, d-1]
                        if np.isnan(o) or np.isnan(c_prev) or c_prev <= 0:
                            continue
                        gap = (o - c_prev) / c_prev
                        v = V[si, d]
                        v_window = V[si, max(0, d-19):d+1]
                        valid_v = v_window[~np.isnan(v_window)]
                        if len(valid_v) < 10:
                            continue
                        v_avg = np.mean(valid_v)
                        if gap > gap_thr and not np.isnan(v) and v > vol_thr * v_avg:
                            buy_days[si].add(di)
                            sell_di = di + hold
                            if sell_di < ND:
                                sell_days[si].add(sell_di)
                name = f'GAP_G{gap_thr*100:.0f}V{vol_thr*10:.0f}H{hold}'
                signals[name] = (buy_days, sell_days)

    # ===== 2. GAP_FOLLOW + OI确认 =====
    for gap_thr in [0.01, 0.015, 0.02]:
        for hold in [3, 5, 7]:
            buy_days = {si: set() for si in range(NS)}
            sell_days = {si: set() for si in range(NS)}
            for si in range(NS):
                for di in range(2, ND):
                    d = di - 1
                    o = O[si, d]
                    c_prev = C[si, d-1]
                    if np.isnan(o) or np.isnan(c_prev) or c_prev <= 0:
                        continue
                    gap = (o - c_prev) / c_prev
                    v = V[si, d]
                    v_window = V[si, max(0, d-19):d+1]
                    valid_v = v_window[~np.isnan(v_window)]
                    if len(valid_v) < 10:
                        continue
                    v_avg = np.mean(valid_v)
                    oi_now = OI[si, d]
                    oi_5ago = OI[si, d-5] if d >= 5 else np.nan
                    oi_ok = (not np.isnan(oi_now) and not np.isnan(oi_5ago) and oi_now > oi_5ago)
                    if gap > gap_thr and not np.isnan(v) and v > 1.5 * v_avg and oi_ok:
                        buy_days[si].add(di)
                        sell_di = di + hold
                        if sell_di < ND:
                            sell_days[si].add(sell_di)
            name = f'GAP_OI_G{gap_thr*100:.0f}H{hold}'
            signals[name] = (buy_days, sell_days)

    # ===== 3. GAP_FOLLOW + MA趋势确认 =====
    for gap_thr in [0.01, 0.015, 0.02]:
        for ma_w in [20, 60]:
            for hold in [3, 5, 7]:
                buy_days = {si: set() for si in range(NS)}
                sell_days = {si: set() for si in range(NS)}
                ma_store = ma20 if ma_w == 20 else ma60
                for si in range(NS):
                    for di in range(ma_w, ND):
                        d = di - 1
                        o = O[si, d]
                        c_prev = C[si, d-1]
                        if np.isnan(o) or np.isnan(c_prev) or c_prev <= 0:
                            continue
                        gap = (o - c_prev) / c_prev
                        v = V[si, d]
                        v_window = V[si, max(0, d-19):d+1]
                        valid_v = v_window[~np.isnan(v_window)]
                        if len(valid_v) < 10:
                            continue
                        v_avg = np.mean(valid_v)
                        # 价格在MA之上
                        above_ma = not np.isnan(ma_store[si, d]) and C[si, d] > ma_store[si, d]
                        if gap > gap_thr and not np.isnan(v) and v > 1.5 * v_avg and above_ma:
                            buy_days[si].add(di)
                            sell_di = di + hold
                            if sell_di < ND:
                                sell_days[si].add(sell_di)
                name = f'GAP_MA{ma_w}_G{gap_thr*100:.0f}H{hold}'
                signals[name] = (buy_days, sell_days)

    # ===== 4. VOL_BREAK_OI: 修复high20 bug, V2原始退出 =====
    buy_days = {si: set() for si in range(NS)}
    sell_days = {si: set() for si in range(NS)}
    for si in range(NS):
        for di in range(22, ND):
            d = di - 1
            c = C[si, d]
            v = V[si, d]
            if np.isnan(c) or np.isnan(v) or v <= 0:
                continue
            v_window = V[si, d-19:d+1]
            valid_v = v_window[~np.isnan(v_window)]
            if len(valid_v) < 10:
                continue
            v_avg = np.mean(valid_v)
            high20 = np.nanmax(C[si, d-19:d])  # BUG FIX: 不含当天
            if np.isnan(high20):
                continue
            oi_now = OI[si, d]
            oi_5ago = OI[si, d-5] if d >= 5 else np.nan
            oi_ok = (not np.isnan(oi_now) and not np.isnan(oi_5ago) and oi_now > oi_5ago)
            if v > 2.0 * v_avg and c > high20 and oi_ok:
                buy_days[si].add(di)
                # V2原始退出: 5%止损 或 10日后3%止损
                entry_price = c
                for hold_di in range(di+1, min(di+30, ND)):
                    hc = C[si, hold_di]
                    if np.isnan(hc):
                        continue
                    if hc < entry_price * 0.95:
                        sell_days[si].add(hold_di)
                        break
                    if hc < entry_price * 0.97 and hold_di - di >= 10:
                        sell_days[si].add(hold_di)
                        break
    signals['VOL_BREAK_OI'] = (buy_days, sell_days)

    # ===== 5. Donchian突破 (海龟) =====
    for chan_len in [10, 20, 30, 40]:
        buy_days = {si: set() for si in range(NS)}
        sell_days = {si: set() for si in range(NS)}
        for si in range(NS):
            for di in range(chan_len + 1, ND):
                d = di - 1
                c = C[si, d]
                if np.isnan(c):
                    continue
                ch_slice = C[si, d-chan_len:d]
                valid_ch = ch_slice[~np.isnan(ch_slice)]
                if len(valid_ch) < chan_len // 2:
                    continue
                high_ch = np.max(valid_ch)
                low_ch = np.min(valid_ch)
                # 突破20日新高
                if c > high_ch:
                    buy_days[si].add(di)
                # 跌破10日新低 (平仓)
                low10 = np.nanmin(C[si, max(0, d-10):d])
                if not np.isnan(low10) and c < low10:
                    sell_days[si].add(di)
        signals[f'DONCH_{chan_len}'] = (buy_days, sell_days)

    # ===== 6. 双动量: 绝对动量 + 相对动量 =====
    for abs_w in [10, 20]:
        for hold in [5, 10]:
            buy_days = {si: set() for si in range(NS)}
            sell_days = {si: set() for si in range(NS)}
            for si in range(NS):
                for di in range(max(60, abs_w), ND):
                    d = di - 1
                    c = C[si, d]
                    if np.isnan(c):
                        continue
                    c_prev = C[si, d-abs_w]
                    if np.isnan(c_prev) or c_prev <= 0:
                        continue
                    # 绝对动量: 价格高于abs_w日前
                    mom = (c - c_prev) / c_prev
                    # 趋势过滤: 价格在MA60之上
                    above_ma60 = not np.isnan(ma60[si, d]) and c > ma60[si, d]
                    if mom > 0.02 and above_ma60:
                        buy_days[si].add(di)
                        sell_days[si].add(di + hold)
            signals[f'DUAL_M{abs_w}H{hold}'] = (buy_days, sell_days)

    # ===== 7. 大缺口 + 持仓激增 (最强确认) =====
    for gap_thr in [0.01, 0.015, 0.02]:
        for hold in [5, 7, 10]:
            buy_days = {si: set() for si in range(NS)}
            sell_days = {si: set() for si in range(NS)}
            for si in range(NS):
                for di in range(22, ND):
                    d = di - 1
                    o = O[si, d]
                    c_prev = C[si, d-1]
                    if np.isnan(o) or np.isnan(c_prev) or c_prev <= 0:
                        continue
                    gap = (o - c_prev) / c_prev
                    v = V[si, d]
                    v_window = V[si, max(0, d-19):d+1]
                    valid_v = v_window[~np.isnan(v_window)]
                    if len(valid_v) < 10:
                        continue
                    v_avg = np.mean(valid_v)
                    # OI激增
                    oi = OI[si, d]
                    if np.isnan(oi) or oi <= 0:
                        continue
                    oi_window = OI[si, d-19:d+1]
                    valid_oi = oi_window[~np.isnan(oi_window)]
                    if len(valid_oi) < 10:
                        continue
                    oi_avg = np.mean(valid_oi)
                    oi_surge = oi > 1.5 * oi_avg  # 1.5x (not 2x, more signals)
                    above_ma20 = not np.isnan(ma20[si, d]) and C[si, d] > ma20[si, d]

                    if gap > gap_thr and not np.isnan(v) and v > 1.5 * v_avg and oi_surge and above_ma20:
                        buy_days[si].add(di)
                        sell_di = di + hold
                        if sell_di < ND:
                            sell_days[si].add(sell_di)
            name = f'GAP_OI_SURGE_G{gap_thr*100:.0f}H{hold}'
            signals[name] = (buy_days, sell_days)

    # ===== 8. 产业链共振 + 放量 =====
    buy_days = {si: set() for si in range(NS)}
    sell_days = {si: set() for si in range(NS)}
    for si in range(NS):
        sym = syms[si]
        grp = sym_group.get(sym)
        if not grp:
            continue
        for di in range(21, ND):
            d = di - 1
            c = C[si, d]
            v = V[si, d]
            if np.isnan(c) or np.isnan(v) or v <= 0:
                continue
            v_window = V[si, d-19:d+1]
            valid_v = v_window[~np.isnan(v_window)]
            if len(valid_v) < 10:
                continue
            v_avg = np.mean(valid_v)
            c_prev = C[si, d-5]
            if np.isnan(c_prev) or c_prev <= 0:
                continue
            mom = (c - c_prev) / c_prev
            gm = group_mom.get(di, {})
            g_mom = gm.get(grp, 0)
            above_ma20 = not np.isnan(ma20[si, d]) and c > ma20[si, d]
            # 品种动量>3% + 组动量>0 + 放量 + MA20之上
            if mom > 0.03 and g_mom > 0 and v > 1.5 * v_avg and above_ma20:
                buy_days[si].add(di)
                sell_days[si].add(di + 5)
    signals['CHAIN_VOL'] = (buy_days, sell_days)

    return signals


def backtest(signals, NS, ND, dates, C, O, H, L, V, OI, syms,
             max_positions=2, sl_pct=0.05, hold_max=60):
    """简洁回测, 不过度约束"""
    buy_days, sell_days = signals

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
            pnl_pct = (c - pos['entry_price']) / pos['entry_price']

            # 止损
            if pnl_pct < -sl_pct:
                cash += pos['shares'] * c * (1 - COMMISSION)
                trades.append({'pnl': pnl_pct * 100, 'days': di - pos['entry_di'],
                               'di': di, 'reason': 'stop', 'year': year, 'si': si})
                positions.remove(pos)
                continue

            # 信号平仓
            if di in sell_days[si]:
                cash += pos['shares'] * c * (1 - COMMISSION)
                trades.append({'pnl': pnl_pct * 100, 'days': di - pos['entry_di'],
                               'di': di, 'reason': 'signal', 'year': year, 'si': si})
                positions.remove(pos)
                continue

            # 超时
            if di - pos['entry_di'] >= hold_max:
                cash += pos['shares'] * c * (1 - COMMISSION)
                trades.append({'pnl': pnl_pct * 100, 'days': di - pos['entry_di'],
                               'di': di, 'reason': 'time', 'year': year, 'si': si})
                positions.remove(pos)

        # 开仓
        if len(positions) < max_positions:
            candidates = []
            for si in range(NS):
                if di in buy_days[si]:
                    if any(p['si'] == si for p in positions):
                        continue
                    c = C[si, di]
                    if np.isnan(c) or c <= 0:
                        continue
                    candidates.append(si)
            candidates.sort(key=lambda si: -V[si, di] if not np.isnan(V[si, di]) else 0)
            for si in candidates[:max_positions - len(positions)]:
                c = C[si, di]
                alloc = cash / max(1, max_positions - len(positions))
                shares = int(alloc / (1 + COMMISSION) / c)
                if shares > 0 and shares * c * (1 + COMMISSION) <= cash:
                    cost = shares * c * (1 + COMMISSION)
                    cash -= cost
                    positions.append({
                        'si': si, 'entry_price': c, 'entry_di': di,
                        'shares': shares,
                    })

    # 平仓
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


def all_positive(r):
    ys = r.get('year_stats', {})
    return all(s['total_pnl'] > 0 for s in ys.values())


if __name__ == '__main__':
    print("=" * 80, flush=True)
    print("  Alpha Futures V6 — 冲高收益", flush=True)
    print("=" * 80, flush=True)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    print("\n[Signals] Generating...", flush=True)
    t0 = time.time()
    all_signals = generate_signals(NS, ND, C, O, H, L, V, OI, syms)
    print(f"  Done ({time.time()-t0:.1f}s) — {len(all_signals)} signal sets", flush=True)

    # =====================================================================
    # 全量回测
    # =====================================================================
    print(f"\n{'='*80}", flush=True)
    print(f"  回测中...", flush=True)

    results = []
    t0 = time.time()
    for name, sigs in all_signals.items():
        for max_pos in [2, 3]:
            for sl in [0.03, 0.05, 0.08]:
                r = backtest(sigs, NS, ND, dates, C, O, H, L, V, OI, syms,
                            max_positions=max_pos, sl_pct=sl, hold_max=60)
                if r:
                    r['name'] = name
                    r['max_pos'] = max_pos
                    r['sl'] = sl
                    results.append(r)

    print(f"  Done ({time.time()-t0:.1f}s, {len(results)} results)", flush=True)

    # 排序输出
    results.sort(key=lambda x: -x['ann'])

    # 统计类型
    gap_results = [r for r in results if r['name'].startswith('GAP_')]
    vol_results = [r for r in results if r['name'].startswith('VOL')]
    donch_results = [r for r in results if r['name'].startswith('DONCH')]
    dual_results = [r for r in results if r['name'].startswith('DUAL')]
    chain_results = [r for r in results if r['name'].startswith('CHAIN')]

    print(f"\n  === TOP 30 OVERALL ===", flush=True)
    print(f"  {'策略':<35s} {'P':>2s} {'SL':>3s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'DD':>5s}", flush=True)
    print(f"  {'-'*75}", flush=True)
    for r in results[:30]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['name']:<35s} P{r['max_pos']} SL{r['sl']:.0%} | "
              f"{r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% {r['max_dd']:5.1f}%{pos_mark}",
              flush=True)

    # 按类型top5
    for label, sub in [('GAP_FOLLOW', gap_results), ('VOL_BREAK', vol_results),
                       ('DONCHIAN', donch_results), ('DUAL_MOM', dual_results),
                       ('CHAIN', chain_results)]:
        sub.sort(key=lambda x: -x['ann'])
        if sub:
            print(f"\n  === Top 5 {label} ===", flush=True)
            for r in sub[:5]:
                pos_mark = " ALL+" if all_positive(r) else ""
                print(f"    {r['name']:<33s} P{r['max_pos']} SL{r['sl']:.0%} → "
                      f"{r['ann']:+.1f}% DD={r['max_dd']:.1f}%{pos_mark}", flush=True)

    # Year-by-year for top 3
    print(f"\n  === TOP 3 YEAR-BY-YEAR ===", flush=True)
    for i, r in enumerate(results[:3]):
        print(f"\n  #{i+1}: {r['name']} P{r['max_pos']} SL{r['sl']:.0%} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            mark = "+" if s['total_pnl'] > 0 else ""
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={mark}{s['total_pnl']:.0f}%",
                  flush=True)

    # ALL+
    all_pos = [r for r in results if all_positive(r)]
    all_pos.sort(key=lambda x: -x['ann'])
    print(f"\n  === ALL+ ({len(all_pos)} total) ===", flush=True)
    for r in all_pos[:15]:
        print(f"    {r['name']:<33s} P{r['max_pos']} SL{r['sl']:.0%} → "
              f"{r['ann']:+.1f}% DD={r['max_dd']:.1f}%", flush=True)

    # vs baseline
    print(f"\n  === BASELINES ===", flush=True)
    print(f"  V2 GAP_FOLLOW:     +35.9% DD=50.0%", flush=True)
    print(f"  V2 VOL_BREAK_OI:   +33.3% DD=47.6%", flush=True)
    if results:
        best = results[0]
        print(f"  V6 best: {best['name']} {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)

    print(f"\n{'='*80}", flush=True)
