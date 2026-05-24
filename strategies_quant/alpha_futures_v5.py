"""
Alpha Futures V5 — 组合优化
============================
V4教训: ATR止损把利润砍了, VOL_BREAK有bug(high20包含当天)

V5策略:
  1. GAP_FOLLOW: 保持V2原始逻辑(hold 5天), 只加追踪止盈
  2. VOL_BREAK_OI: 修复high20 bug, 恢复信号
  3. CHAIN_MOM: 保留多空
  4. OI_SURGE_TREND: 改进
  5. 多策略组合: 资金分配 + equity curve trading
  6. 回撤控制: 亏损暂停交易, 恢复后减仓
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


def generate_all_signals(NS, ND, C, O, H, L, V, OI, syms):
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

    # MA20
    ma20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            vals = C[si, di-20:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 10:
                ma20[si, di] = np.mean(valid)

    signals = {}

    # ===== 1. GAP_FOLLOW: V2原始逻辑, hold 5天 =====
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
            if gap > 0.01 and not np.isnan(v) and v > 1.5 * v_avg:
                buy_days[si].add(di)
                sell_di = di + 5
                if sell_di < ND:
                    sell_days[si].add(sell_di)
    signals['GAP_FOLLOW'] = (buy_days, sell_days)

    # ===== 1b. GAP_FOLLOW_3d: hold 3天 =====
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
            if gap > 0.01 and not np.isnan(v) and v > 1.5 * v_avg:
                buy_days[si].add(di)
                sell_di = di + 3
                if sell_di < ND:
                    sell_days[si].add(sell_di)
    signals['GAP_3D'] = (buy_days, sell_days)

    # ===== 1c. GAP_FOLLOW_7d: hold 7天 =====
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
            if gap > 0.01 and not np.isnan(v) and v > 1.5 * v_avg:
                buy_days[si].add(di)
                sell_di = di + 7
                if sell_di < ND:
                    sell_days[si].add(sell_di)
    signals['GAP_7D'] = (buy_days, sell_days)

    # ===== 2. VOL_BREAK_OI: 修复high20 bug =====
    buy_days = {si: set() for si in range(NS)}
    sell_days = {si: set() for si in range(NS)}
    for si in range(NS):
        for di in range(22, ND):
            d = di - 1
            c = C[si, d]
            v = V[si, d]
            if np.isnan(c) or np.isnan(v) or v <= 0:
                continue
            # 放量
            v_window = V[si, d-19:d+1]
            valid_v = v_window[~np.isnan(v_window)]
            if len(valid_v) < 10:
                continue
            v_avg = np.mean(valid_v)
            # 20日新高 (不含当天!) — BUG FIX
            high20 = np.nanmax(C[si, d-19:d])  # d is exclusive, so d-19 to d-1
            if np.isnan(high20):
                continue
            # OI确认
            oi_now = OI[si, d]
            oi_5ago = OI[si, d-5] if d >= 5 else np.nan
            oi_ok = (not np.isnan(oi_now) and not np.isnan(oi_5ago) and oi_now > oi_5ago)

            if v > 2.0 * v_avg and c > high20 and oi_ok:
                buy_days[si].add(di)
    signals['VOL_BREAK_OI'] = (buy_days, sell_days)

    # ===== 2b. VOL_BREAK_OI_10d: hold 10天 =====
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
            high20 = np.nanmax(C[si, d-19:d])
            if np.isnan(high20):
                continue
            oi_now = OI[si, d]
            oi_5ago = OI[si, d-5] if d >= 5 else np.nan
            oi_ok = (not np.isnan(oi_now) and not np.isnan(oi_5ago) and oi_now > oi_5ago)
            if v > 2.0 * v_avg and c > high20 and oi_ok:
                buy_days[si].add(di)
                sell_di = di + 10
                if sell_di < ND:
                    sell_days[si].add(sell_di)
    signals['VOL_BREAK_10D'] = (buy_days, sell_days)

    # ===== 3. OI_SURGE_TREND: V2逻辑 (long-only) =====
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
            if len(valid_oi) < 15:
                continue
            oi_avg = np.mean(valid_oi)
            oi_surge = oi > 2.0 * oi_avg
            c = C[si, d]
            c20 = C[si, d-19:d+1]
            valid_c = c20[~np.isnan(c20)]
            if len(valid_c) < 15:
                continue
            ma20_val = np.mean(valid_c)
            if oi_surge and c > ma20_val:
                buy_days[si].add(di)
            if not oi_surge and c < ma20_val:
                sell_days[si].add(di)
    signals['OI_SURGE'] = (buy_days, sell_days)

    # ===== 4. CHAIN_MOM_LS: 产业链共振多空 =====
    buy_days = {si: set() for si in range(NS)}
    sell_days = {si: set() for si in range(NS)}
    short_days = {si: set() for si in range(NS)}
    cover_days = {si: set() for si in range(NS)}
    for si in range(NS):
        sym = syms[si]
        grp = sym_group.get(sym)
        if not grp:
            continue
        for di in range(21, ND):
            d = di - 1
            c = C[si, d]
            if np.isnan(c):
                continue
            c_prev = C[si, d-10]
            if np.isnan(c_prev) or c_prev <= 0:
                continue
            mom = (c - c_prev) / c_prev
            gm = group_mom.get(di, {})
            g_mom = gm.get(grp, 0)
            above_ma20 = not np.isnan(ma20[si, d]) and c > ma20[si, d]
            below_ma20 = not np.isnan(ma20[si, d]) and c < ma20[si, d]

            if mom > 0.02 and g_mom > 0.01 and above_ma20:
                buy_days[si].add(di)
            if mom < -0.02 and g_mom < -0.01 and below_ma20:
                short_days[si].add(di)
            if g_mom < 0:
                sell_days[si].add(di)
            if g_mom > 0:
                cover_days[si].add(di)
    signals['CHAIN_MOM'] = (buy_days, sell_days, short_days, cover_days)

    return signals


def _is_ls(signals):
    """判断是否是多空策略"""
    return len(signals) == 4


def backtest_strategy(signals, NS, ND, dates, C, O, H, L, V, OI, syms,
                      max_positions=2, sl_pct=0.05, hold_max=30,
                      trail_pct=0.03, dd_pause=0.15):
    """
    通用回测引擎 (long-only或long+short)
    dd_pause: 回撤超过此值时暂停新开仓直到回撤恢复
    """
    if _is_ls(signals):
        buy_days, sell_days, short_days, cover_days = signals
    else:
        buy_days, sell_days = signals
        short_days = {si: set() for si in range(NS)}
        cover_days = {si: set() for si in range(NS)}

    cash = float(CASH0)
    positions = []
    trades = []
    year_stats = {}
    peak_equity = float(CASH0)
    in_pause = False
    pause_di = 0

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # 当前权益
        pos_value = 0
        for pos in positions:
            c = C[pos['si'], di]
            if np.isnan(c):
                c = pos['entry_price']
            if pos['direction'] == 'long':
                pos_value += pos['shares'] * c
            else:
                pos_value += pos['shares'] * (2 * pos['entry_price'] - c)
        current_eq = cash + pos_value
        if current_eq > peak_equity:
            peak_equity = current_eq
        current_dd = (peak_equity - current_eq) / peak_equity if peak_equity > 0 else 0

        # 回撤暂停：回撤>dd_pause时停止开仓，恢复到dd_pause/2以下恢复
        if current_dd > dd_pause:
            in_pause = True
            pause_di = di
        elif in_pause and current_dd < dd_pause / 2:
            in_pause = False

        # 平仓检查
        for pos in list(positions):
            si = pos['si']
            c = C[si, di]
            if np.isnan(c):
                continue
            if pos['direction'] == 'long':
                pnl_pct = (c - pos['entry_price']) / pos['entry_price']
            else:
                pnl_pct = (pos['entry_price'] - c) / pos['entry_price']

            # 固定止损
            if pnl_pct < -sl_pct:
                if pos['direction'] == 'long':
                    cash += pos['shares'] * c * (1 - COMMISSION)
                else:
                    cash += pos['shares'] * (2 * pos['entry_price'] - c) * (1 - COMMISSION)
                trades.append({'pnl': pnl_pct * 100, 'days': di - pos['entry_di'],
                               'di': di, 'reason': 'stop', 'year': year,
                               'si': si, 'dir': pos['direction']})
                positions.remove(pos)
                continue

            # 追踪止盈: 盈利超trail_pct后, 回撤2%平仓
            peak_pnl = pos.get('peak_pnl', 0)
            if pnl_pct > peak_pnl:
                pos['peak_pnl'] = pnl_pct
            if peak_pnl > trail_pct and pnl_pct < peak_pnl - 0.02:
                if pos['direction'] == 'long':
                    cash += pos['shares'] * c * (1 - COMMISSION)
                else:
                    cash += pos['shares'] * (2 * pos['entry_price'] - c) * (1 - COMMISSION)
                trades.append({'pnl': pnl_pct * 100, 'days': di - pos['entry_di'],
                               'di': di, 'reason': 'trail', 'year': year,
                               'si': si, 'dir': pos['direction']})
                positions.remove(pos)
                continue

            # 信号平仓
            if pos['direction'] == 'long' and di in sell_days[si]:
                cash += pos['shares'] * c * (1 - COMMISSION)
                trades.append({'pnl': pnl_pct * 100, 'days': di - pos['entry_di'],
                               'di': di, 'reason': 'signal', 'year': year,
                               'si': si, 'dir': pos['direction']})
                positions.remove(pos)
                continue
            if pos['direction'] == 'short' and di in cover_days[si]:
                cash += pos['shares'] * (2 * pos['entry_price'] - c) * (1 - COMMISSION)
                trades.append({'pnl': pnl_pct * 100, 'days': di - pos['entry_di'],
                               'di': di, 'reason': 'signal', 'year': year,
                               'si': si, 'dir': pos['direction']})
                positions.remove(pos)
                continue

            # 超时
            if di - pos['entry_di'] >= hold_max:
                if pos['direction'] == 'long':
                    cash += pos['shares'] * c * (1 - COMMISSION)
                else:
                    cash += pos['shares'] * (2 * pos['entry_price'] - c) * (1 - COMMISSION)
                trades.append({'pnl': pnl_pct * 100, 'days': di - pos['entry_di'],
                               'di': di, 'reason': 'time', 'year': year,
                               'si': si, 'dir': pos['direction']})
                positions.remove(pos)

        # 开仓 (暂停期间不开新仓)
        if not in_pause and len(positions) < max_positions:
            long_cands = []
            for si in range(NS):
                if di in buy_days[si]:
                    if any(p['si'] == si for p in positions):
                        continue
                    c = C[si, di]
                    if np.isnan(c) or c <= 0:
                        continue
                    long_cands.append(('long', si, c))

            short_cands = []
            for si in range(NS):
                if di in short_days[si]:
                    if any(p['si'] == si for p in positions):
                        continue
                    c = C[si, di]
                    if np.isnan(c) or c <= 0:
                        continue
                    short_cands.append(('short', si, c))

            all_cands = long_cands + short_cands
            all_cands.sort(key=lambda x: -V[x[1], di] if not np.isnan(V[x[1], di]) else 0)
            slots = max_positions - len(positions)
            for direction, si, price in all_cands[:slots]:
                alloc = cash / max(1, max_positions - len(positions))
                shares = int(alloc / (1 + COMMISSION) / price)
                if shares > 0 and shares * price * (1 + COMMISSION) <= cash:
                    cost = shares * price * (1 + COMMISSION)
                    cash -= cost
                    positions.append({
                        'si': si, 'entry_price': price, 'entry_di': di,
                        'shares': shares, 'direction': direction, 'peak_pnl': 0.0,
                    })

    # 平仓
    for pos in positions:
        c = C[pos['si'], ND-1]
        if np.isnan(c) or c <= 0:
            c = pos['entry_price']
        if pos['direction'] == 'long':
            pnl = (c - pos['entry_price']) / pos['entry_price'] * 100
            cash += pos['shares'] * c * (1 - COMMISSION)
        else:
            pnl = (pos['entry_price'] - c) / pos['entry_price'] * 100
            cash += pos['shares'] * (2 * pos['entry_price'] - c) * (1 - COMMISSION)
        trades.append({'pnl': pnl, 'days': 999, 'di': ND-1, 'reason': 'end',
                       'year': dates[ND-1].year, 'si': pos['si'], 'dir': pos['direction']})

    if not trades:
        return None

    days_total = (dates[ND-1] - dates[MIN_TRAIN]).days
    yr = max(days_total / 365.25, 0.01)
    ann = ((cash / CASH0) ** (1 / yr) - 1) * 100
    nw = sum(1 for t in trades if t['pnl'] > 0)
    wr = nw / max(len(trades), 1) * 100

    long_t = [t for t in trades if t.get('dir') == 'long']
    short_t = [t for t in trades if t.get('dir') == 'short']

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

    exit_reasons = {}
    for t in trades:
        r = t.get('reason', 'unknown')
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    result = {
        'ann': round(ann, 1), 'n': len(trades), 'wr': round(wr, 1),
        'max_dd': round(max_dd, 1), 'final': round(cash, 0),
        'year_stats': year_stats, 'exit_reasons': exit_reasons,
    }
    if short_t:
        result.update({
            'long_n': len(long_t), 'short_n': len(short_t),
            'long_wr': round(sum(1 for t in long_t if t['pnl'] > 0) / max(len(long_t), 1) * 100, 1),
            'short_wr': round(sum(1 for t in short_t if t['pnl'] > 0) / max(len(short_t), 1) * 100, 1),
        })
    return result


def all_positive(r):
    ys = r.get('year_stats', {})
    return all(s['total_pnl'] > 0 for s in ys.values())


# ============================================================
# 多策略组合回测
# ============================================================
def backtest_combo(all_signals, strategy_alloc, NS, ND, dates, C, O, H, L, V, OI, syms,
                   max_positions=3, sl_pct=0.05, hold_max=30, trail_pct=0.03, dd_pause=0.15):
    """
    多策略组合: 每个策略独立占用slot, 分配资金
    strategy_alloc: {name: allocation_fraction}
    """
    cash = float(CASH0)
    positions = []  # 每个pos额外记录strategy
    trades = []
    year_stats = {}
    peak_equity = float(CASH0)
    in_pause = False

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        pos_value = 0
        for pos in positions:
            c = C[pos['si'], di]
            if np.isnan(c):
                c = pos['entry_price']
            if pos['direction'] == 'long':
                pos_value += pos['shares'] * c
            else:
                pos_value += pos['shares'] * (2 * pos['entry_price'] - c)
        current_eq = cash + pos_value
        if current_eq > peak_equity:
            peak_equity = current_eq
        current_dd = (peak_equity - current_eq) / peak_equity if peak_equity > 0 else 0
        if current_dd > dd_pause:
            in_pause = True
        elif in_pause and current_dd < dd_pause / 2:
            in_pause = False

        # 平仓
        for pos in list(positions):
            si = pos['si']
            c = C[si, di]
            if np.isnan(c):
                continue
            if pos['direction'] == 'long':
                pnl_pct = (c - pos['entry_price']) / pos['entry_price']
            else:
                pnl_pct = (pos['entry_price'] - c) / pos['entry_price']

            if pnl_pct < -sl_pct:
                if pos['direction'] == 'long':
                    cash += pos['shares'] * c * (1 - COMMISSION)
                else:
                    cash += pos['shares'] * (2 * pos['entry_price'] - c) * (1 - COMMISSION)
                trades.append({'pnl': pnl_pct * 100, 'days': di - pos['entry_di'],
                               'di': di, 'reason': 'stop', 'year': year, 'si': si,
                               'dir': pos['direction'], 'strat': pos['strategy']})
                positions.remove(pos)
                continue

            peak_pnl = pos.get('peak_pnl', 0)
            if pnl_pct > peak_pnl:
                pos['peak_pnl'] = pnl_pct
            if peak_pnl > trail_pct and pnl_pct < peak_pnl - 0.02:
                if pos['direction'] == 'long':
                    cash += pos['shares'] * c * (1 - COMMISSION)
                else:
                    cash += pos['shares'] * (2 * pos['entry_price'] - c) * (1 - COMMISSION)
                trades.append({'pnl': pnl_pct * 100, 'days': di - pos['entry_di'],
                               'di': di, 'reason': 'trail', 'year': year, 'si': si,
                               'dir': pos['direction'], 'strat': pos['strategy']})
                positions.remove(pos)
                continue

            sigs = all_signals[pos['strategy']]
            sell_d = sigs[1] if len(sigs) >= 2 else {}
            cover_d = sigs[3] if len(sigs) == 4 else {}

            if pos['direction'] == 'long' and di in sell_d.get(si, set()):
                cash += pos['shares'] * c * (1 - COMMISSION)
                trades.append({'pnl': pnl_pct * 100, 'days': di - pos['entry_di'],
                               'di': di, 'reason': 'signal', 'year': year, 'si': si,
                               'dir': pos['direction'], 'strat': pos['strategy']})
                positions.remove(pos)
                continue
            if pos['direction'] == 'short' and di in cover_d.get(si, set()):
                cash += pos['shares'] * (2 * pos['entry_price'] - c) * (1 - COMMISSION)
                trades.append({'pnl': pnl_pct * 100, 'days': di - pos['entry_di'],
                               'di': di, 'reason': 'signal', 'year': year, 'si': si,
                               'dir': pos['direction'], 'strat': pos['strategy']})
                positions.remove(pos)
                continue

            if di - pos['entry_di'] >= hold_max:
                if pos['direction'] == 'long':
                    cash += pos['shares'] * c * (1 - COMMISSION)
                else:
                    cash += pos['shares'] * (2 * pos['entry_price'] - c) * (1 - COMMISSION)
                trades.append({'pnl': pnl_pct * 100, 'days': di - pos['entry_di'],
                               'di': di, 'reason': 'time', 'year': year, 'si': si,
                               'dir': pos['direction'], 'strat': pos['strategy']})
                positions.remove(pos)

        # 开仓
        if not in_pause and len(positions) < max_positions:
            # 每个策略独立的slot
            for sname, alloc in strategy_alloc.items():
                if sname not in all_signals:
                    continue
                sigs = all_signals[sname]
                buy_d = sigs[0]
                short_d = sigs[2] if len(sigs) == 4 else {}

                slots_for_strat = max(1, int(max_positions * alloc))
                strat_pos_count = sum(1 for p in positions if p['strategy'] == sname)
                if strat_pos_count >= slots_for_strat:
                    continue

                cands = []
                for si in range(NS):
                    if di in buy_d.get(si, set()):
                        if any(p['si'] == si for p in positions):
                            continue
                        c = C[si, di]
                        if np.isnan(c) or c <= 0:
                            continue
                        cands.append(('long', si, c))
                for si in range(NS):
                    if di in short_d.get(si, set()):
                        if any(p['si'] == si for p in positions):
                            continue
                        c = C[si, di]
                        if np.isnan(c) or c <= 0:
                            continue
                        cands.append(('short', si, c))

                cands.sort(key=lambda x: -V[x[1], di] if not np.isnan(V[x[1], di]) else 0)
                open_slots = min(slots_for_strat - strat_pos_count, max_positions - len(positions))
                for direction, si, price in cands[:open_slots]:
                    alloc_cash = cash * alloc / max(1, max_positions - len(positions))
                    shares = int(alloc_cash / (1 + COMMISSION) / price)
                    if shares > 0 and shares * price * (1 + COMMISSION) <= cash:
                        cost = shares * price * (1 + COMMISSION)
                        cash -= cost
                        positions.append({
                            'si': si, 'entry_price': price, 'entry_di': di,
                            'shares': shares, 'direction': direction, 'peak_pnl': 0.0,
                            'strategy': sname,
                        })

    for pos in positions:
        c = C[pos['si'], ND-1]
        if np.isnan(c) or c <= 0:
            c = pos['entry_price']
        if pos['direction'] == 'long':
            pnl = (c - pos['entry_price']) / pos['entry_price'] * 100
            cash += pos['shares'] * c * (1 - COMMISSION)
        else:
            pnl = (pos['entry_price'] - c) / pos['entry_price'] * 100
            cash += pos['shares'] * (2 * pos['entry_price'] - c) * (1 - COMMISSION)
        trades.append({'pnl': pnl, 'days': 999, 'di': ND-1, 'reason': 'end',
                       'year': dates[ND-1].year, 'si': pos['si'], 'dir': pos['direction'],
                       'strat': pos['strategy']})

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

    exit_reasons = {}
    for t in trades:
        r = t.get('reason', 'unknown')
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    # Per-strategy breakdown
    strat_breakdown = {}
    for t in trades:
        sn = t.get('strat', 'unknown')
        if sn not in strat_breakdown:
            strat_breakdown[sn] = {'n': 0, 'wins': 0, 'pnl': 0}
        strat_breakdown[sn]['n'] += 1
        if t['pnl'] > 0:
            strat_breakdown[sn]['wins'] += 1
        strat_breakdown[sn]['pnl'] += t['pnl']

    return {
        'ann': round(ann, 1), 'n': len(trades), 'wr': round(wr, 1),
        'max_dd': round(max_dd, 1), 'final': round(cash, 0),
        'year_stats': year_stats, 'exit_reasons': exit_reasons,
        'strat_breakdown': strat_breakdown,
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 80, flush=True)
    print("  Alpha Futures V5 — 组合优化", flush=True)
    print("=" * 80, flush=True)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    print("\n[Signals] Generating...", flush=True)
    t0 = time.time()
    all_signals = generate_all_signals(NS, ND, C, O, H, L, V, OI, syms)
    print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

    # 信号统计
    print(f"\n  Signal counts:", flush=True)
    for name, sigs in all_signals.items():
        if len(sigs) == 2:
            bd, sd = sigs
            nb = sum(len(v) for v in bd.values())
            ns = sum(len(v) for v in sd.values())
            print(f"    {name:<20s}: buy={nb:5d} sell={ns:5d}", flush=True)
        else:
            bd, sd, sh, cv = sigs
            nb = sum(len(v) for v in bd.values())
            ns = sum(len(v) for v in sd.values())
            nsh = sum(len(v) for v in sh.values())
            nc = sum(len(v) for v in cv.values())
            print(f"    {name:<20s}: buy={nb:5d} sell={ns:5d} short={nsh:5d} cover={nc:5d}", flush=True)

    # =====================================================================
    # 单策略
    # =====================================================================
    print(f"\n{'='*80}", flush=True)
    print(f"  单策略回测", flush=True)
    print(f"  {'策略':<20s} {'P':>2s} {'SL':>3s} {'DDp':>4s} |"
          f" {'Ann':>7s} {'N':>5s} {'WR':>5s} {'DD':>5s} {'终值':>10s}", flush=True)
    print(f"  {'-'*75}", flush=True)

    results = []
    for name, sigs in all_signals.items():
        for max_pos in [2, 3, 4]:
            for sl in [0.03, 0.05, 0.08]:
                for dd_p in [0.10, 0.15, 0.20, 0.99]:
                    r = backtest_strategy(sigs, NS, ND, dates, C, O, H, L, V, OI, syms,
                                         max_positions=max_pos, sl_pct=sl, hold_max=30,
                                         trail_pct=0.03, dd_pause=dd_p)
                    if r:
                        r['name'] = name
                        r['max_pos'] = max_pos
                        r['sl'] = sl
                        r['dd_pause'] = dd_p
                        results.append(r)

    results.sort(key=lambda x: -x['ann'])
    for r in results[:60]:
        pos_mark = " ALL+" if all_positive(r) else ""
        ls_info = ""
        if 'long_n' in r:
            ls_info = f" L:{r['long_n']}/{r['long_wr']}% S:{r['short_n']}/{r['short_wr']}%"
        print(f"  {r['name']:<20s} P{r['max_pos']} SL{r['sl']:.0%} DD{r['dd_pause']:.2f} |"
              f" {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% {r['max_dd']:5.1f}%"
              f" {r['final']:>10.0f}{pos_mark}{ls_info}", flush=True)

    # Best per strategy
    print(f"\n  === Best per strategy ===", flush=True)
    best_per = {}
    for r in results:
        key = f"{r['name']}_P{r['max_pos']}"
        if key not in best_per or r['ann'] > best_per[key]['ann']:
            best_per[key] = r
    for r in sorted(best_per.values(), key=lambda x: -x['ann']):
        ex = ', '.join(f"{k}={v}" for k, v in sorted(r['exit_reasons'].items()))
        ls_info = ""
        if 'long_n' in r:
            ls_info = f" | L:{r['long_n']}t/{r['long_wr']}% S:{r['short_n']}t/{r['short_wr']}%"
        print(f"    {r['name']:<20s} P{r['max_pos']} SL{r['sl']:.0%} → "
              f"{r['ann']:+.1f}% DD={r['max_dd']:.1f}%{ls_info}", flush=True)
        print(f"      Exits: {ex}", flush=True)

    # =====================================================================
    # 组合
    # =====================================================================
    print(f"\n{'='*80}", flush=True)
    print(f"  多策略组合", flush=True)
    print(f"  {'组合':<45s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'DD':>5s}", flush=True)
    print(f"  {'-'*75}", flush=True)

    combos = [
        ('GAP+CHAIN', {'GAP_FOLLOW': 0.6, 'CHAIN_MOM': 0.4}),
        ('GAP+VOL', {'GAP_FOLLOW': 0.5, 'VOL_BREAK_OI': 0.5}),
        ('GAP+OI', {'GAP_FOLLOW': 0.5, 'OI_SURGE': 0.5}),
        ('GAP+VOL+CHAIN', {'GAP_FOLLOW': 0.4, 'VOL_BREAK_OI': 0.3, 'CHAIN_MOM': 0.3}),
        ('GAP+CHAIN_60/40', {'GAP_FOLLOW': 0.6, 'CHAIN_MOM': 0.4}),
        ('GAP_3D+CHAIN', {'GAP_3D': 0.6, 'CHAIN_MOM': 0.4}),
        ('GAP_7D+CHAIN', {'GAP_7D': 0.6, 'CHAIN_MOM': 0.4}),
        ('GAP+VOL+CHAIN+OI', {'GAP_FOLLOW': 0.35, 'VOL_BREAK_OI': 0.2,
                               'CHAIN_MOM': 0.25, 'OI_SURGE': 0.2}),
        ('ALL_EQ', {'GAP_FOLLOW': 0.25, 'VOL_BREAK_OI': 0.25,
                    'CHAIN_MOM': 0.25, 'OI_SURGE': 0.25}),
    ]

    combo_results = []
    for cname, alloc in combos:
        for max_pos in [3, 4, 5]:
            for sl in [0.05, 0.08]:
                for dd_p in [0.15, 0.20, 0.99]:
                    r = backtest_combo(all_signals, alloc, NS, ND, dates, C, O, H, L, V, OI, syms,
                                       max_positions=max_pos, sl_pct=sl, hold_max=30,
                                       trail_pct=0.03, dd_pause=dd_p)
                    if r:
                        r['combo'] = cname
                        r['max_pos'] = max_pos
                        r['sl'] = sl
                        r['dd_pause'] = dd_p
                        combo_results.append(r)

    combo_results.sort(key=lambda x: -x['ann'])
    for r in combo_results[:40]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['combo']:<43s} P{r['max_pos']} SL{r['sl']:.0%} DD{r['dd_pause']:.2f} |"
              f" {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% {r['max_dd']:5.1f}%{pos_mark}",
              flush=True)

    # 组合策略分拆
    if combo_results:
        print(f"\n  === Best combo breakdown ===", flush=True)
        best_combo = combo_results[0]
        for sn, sb in sorted(best_combo.get('strat_breakdown', {}).items()):
            wr = sb['wins'] / max(sb['n'], 1) * 100
            print(f"    {sn}: {sb['n']} trades, WR={wr:.0f}%, pnl={sb['pnl']:+.0f}%", flush=True)

    # =====================================================================
    # TOP 5 year-by-year
    # =====================================================================
    all_r = results + combo_results
    all_r.sort(key=lambda x: -x['ann'])

    print(f"\n  === TOP 5 OVERALL ===", flush=True)
    for i, r in enumerate(all_r[:5]):
        label = r.get('combo', r.get('name', '?'))
        extra = f"P{r['max_pos']} SL{r['sl']:.0%}"
        if 'dd_pause' in r:
            extra += f" DD{r['dd_pause']:.2f}"
        print(f"\n  #{i+1}: {label} {extra} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            mark = "+" if s['total_pnl'] > 0 else ""
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={mark}{s['total_pnl']:.0f}%",
                  flush=True)

    # V2对比
    print(f"\n  === vs Baselines ===", flush=True)
    print(f"  V2 GAP_FOLLOW:   +35.9% DD=50.0%", flush=True)
    print(f"  V2 VOL_BREAK_OI: +33.3% DD=47.6%", flush=True)
    print(f"  V2 OI_SURGE:     +21.1% DD=79.9%", flush=True)
    if all_r:
        best = all_r[0]
        label = best.get('combo', best.get('name', '?'))
        print(f"  V5 best: {label} {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)

    # ALL+
    all_pos = [r for r in all_r if all_positive(r)]
    if all_pos:
        all_pos.sort(key=lambda x: -x['ann'])
        print(f"\n  ALL+ 策略 ({len(all_pos)} total):", flush=True)
        for r in all_pos[:10]:
            label = r.get('combo', r.get('name', '?'))
            print(f"    {label:<35s} P{r['max_pos']} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}%",
                  flush=True)
    else:
        print(f"\n  No ALL+ strategies", flush=True)

    print(f"\n{'='*80}", flush=True)
