"""
Alpha Futures V4 — 精准优化
============================
V3教训: 做空拖累GAP_FOLLOW, 过滤太严VOL_BREAK没信号

V4策略:
  1. GAP_FOLLOW: 保持long-only + 加ADX过滤减少假信号
  2. VOL_BREAK_OI: 保持long-only + 放宽过滤 + ATR止盈
  3. CHAIN_MOM: 保留多空(唯一多空都赚的策略)
  4. OI_SURGE_TREND: 改进出场逻辑
  5. 组合: 用做多策略+CHAIN_MOM对冲
  6. ATR追踪止损替代固定止损
  7. 专注降回撤: 连亏减仓 + 盈利加仓
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


def _atr(close, high, low, period=14):
    n = len(close)
    tr = np.full(n, np.nan)
    atr = np.full(n, np.nan)
    for i in range(1, n):
        if np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]):
            continue
        if np.isnan(close[i-1]):
            continue
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i-1]),
                     abs(low[i] - close[i-1]))
    for i in range(period, n):
        if np.isnan(tr[i]):
            continue
        if np.isnan(atr[i-1]):
            valid = tr[max(0, i-period):i]
            valid = valid[~np.isnan(valid)]
            if len(valid) > 0:
                atr[i] = np.mean(valid)
            continue
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


def _adx(close, high, low, period=14):
    n = len(close)
    adx = np.full(n, np.nan)
    if n < period * 2:
        return adx
    plus_dm = np.full(n, np.nan)
    minus_dm = np.full(n, np.nan)
    tr_arr = np.full(n, np.nan)
    for i in range(1, n):
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i]):
            continue
        if np.isnan(high[i-1]) or np.isnan(low[i-1]) or np.isnan(close[i-1]):
            continue
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        plus_dm[i] = up if up > down and up > 0 else 0
        minus_dm[i] = down if down > up and down > 0 else 0
        tr_arr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i-1]),
                        abs(low[i] - close[i-1]))

    def _smooth(arr, start, per):
        out = np.full(n, np.nan)
        for i in range(start, n):
            if np.isnan(arr[i]):
                continue
            if np.isnan(out[i-1]):
                valid = arr[max(0, i-per):i]
                valid = valid[~np.isnan(valid)]
                if len(valid) > 0:
                    out[i] = np.mean(valid)
                continue
            out[i] = out[i-1] - out[i-1] / per + arr[i]
        return out

    s_tr = _smooth(tr_arr, period, period)
    s_plus = _smooth(plus_dm, period, period)
    s_minus = _smooth(minus_dm, period, period)
    dx = np.full(n, np.nan)
    for i in range(period * 2, n):
        if np.isnan(s_tr[i]) or s_tr[i] <= 0:
            continue
        if np.isnan(s_plus[i]) or np.isnan(s_minus[i]):
            continue
        pdi = s_plus[i] / s_tr[i] * 100
        mdi = s_minus[i] / s_tr[i] * 100
        denom = pdi + mdi
        if denom > 0:
            dx[i] = abs(pdi - mdi) / denom * 100

    for i in range(period * 2, n):
        if np.isnan(dx[i]):
            continue
        if np.isnan(adx[i-1]):
            valid = dx[max(period*2, i-period):i]
            valid = valid[~np.isnan(valid)]
            if len(valid) > 0:
                adx[i] = np.mean(valid)
            continue
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    return adx


def generate_signals_v4(NS, ND, C, O, H, L, V, OI, syms):
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

    # ATR/ADX
    print("    ATR/ADX...", flush=True)
    atr_arr = np.full((NS, ND), np.nan)
    adx_arr = np.full((NS, ND), np.nan)
    ma20_arr = np.full((NS, ND), np.nan)
    ma60_arr = np.full((NS, ND), np.nan)
    for si in range(NS):
        atr_arr[si] = _atr(C[si], H[si], L[si], 14)
        adx_arr[si] = _adx(C[si], H[si], L[si], 14)
        for window, store in [(20, ma20_arr), (60, ma60_arr)]:
            for di in range(window, ND):
                vals = C[si, di-window:di]
                valid = vals[~np.isnan(vals)]
                if len(valid) >= window // 2:
                    store[si, di] = np.mean(valid)

    signals = {}

    # ===== 1. GAP_FOLLOW_LONG: v2 best, long-only, +ADX过滤 =====
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
                sell_days[si].add(di + 5)
    signals['GAP_LONG'] = (buy_days, sell_days)

    # ===== 1b. GAP_LONG_ADX: 同上+ADX>15 =====
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
            adx_val = adx_arr[si, d]
            if np.isnan(adx_val) or adx_val < 15:
                continue
            if gap > 0.01 and not np.isnan(v) and v > 1.5 * v_avg:
                buy_days[si].add(di)
                sell_days[si].add(di + 5)
    signals['GAP_LONG_ADX'] = (buy_days, sell_days)

    # ===== 1c. GAP_LONG_ADX_OI: 加OI确认 =====
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
            adx_val = adx_arr[si, d]
            if np.isnan(adx_val) or adx_val < 15:
                continue
            oi_now = OI[si, d]
            oi_5ago = OI[si, d-5] if d >= 5 else np.nan
            oi_ok = (not np.isnan(oi_now) and not np.isnan(oi_5ago) and oi_now > oi_5ago)
            if gap > 0.01 and not np.isnan(v) and v > 1.5 * v_avg and oi_ok:
                buy_days[si].add(di)
                sell_days[si].add(di + 5)
    signals['GAP_LONG_ADX_OI'] = (buy_days, sell_days)

    # ===== 2. VOL_BREAK_LONG: long-only 放量突破 =====
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
            c20 = C[si, d-19:d+1]
            valid_c = c20[~np.isnan(c20)]
            if len(valid_c) < 15:
                continue
            high20 = np.max(valid_c)
            oi_now = OI[si, d]
            oi_5ago = OI[si, d-5] if d >= 5 else np.nan
            oi_ok = (not np.isnan(oi_now) and not np.isnan(oi_5ago) and oi_now > oi_5ago)
            if v > 2.0 * v_avg and c > high20 and oi_ok:
                buy_days[si].add(di)
    signals['VOL_BREAK_LONG'] = (buy_days, sell_days)

    # ===== 2b. VOL_BREAK_LONG_ADX: +ADX =====
    buy_days = {si: set() for si in range(NS)}
    sell_days = {si: set() for si in range(NS)}
    for si in range(NS):
        for di in range(22, ND):
            d = di - 1
            c = C[si, d]
            v = V[si, d]
            if np.isnan(c) or np.isnan(v) or v <= 0:
                continue
            adx_val = adx_arr[si, d]
            if np.isnan(adx_val) or adx_val < 15:
                continue
            v_window = V[si, d-19:d+1]
            valid_v = v_window[~np.isnan(v_window)]
            if len(valid_v) < 10:
                continue
            v_avg = np.mean(valid_v)
            c20 = C[si, d-19:d+1]
            valid_c = c20[~np.isnan(c20)]
            if len(valid_c) < 15:
                continue
            high20 = np.max(valid_c)
            oi_now = OI[si, d]
            oi_5ago = OI[si, d-5] if d >= 5 else np.nan
            oi_ok = (not np.isnan(oi_now) and not np.isnan(oi_5ago) and oi_now > oi_5ago)
            if v > 2.0 * v_avg and c > high20 and oi_ok:
                buy_days[si].add(di)
    signals['VOL_BREAK_LONG_ADX'] = (buy_days, sell_days)

    # ===== 3. CHAIN_MOM: 多空产业链共振 (v3 best) =====
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
            adx_val = adx_arr[si, d]
            if np.isnan(adx_val) or adx_val < 15:
                continue
            above_ma20 = not np.isnan(ma20_arr[si, d]) and c > ma20_arr[si, d]
            below_ma20 = not np.isnan(ma20_arr[si, d]) and c < ma20_arr[si, d]
            if mom > 0.02 and g_mom > 0.01 and above_ma20:
                buy_days[si].add(di)
            if mom < -0.02 and g_mom < -0.01 and below_ma20:
                short_days[si].add(di)
            if g_mom < 0:
                sell_days[si].add(di)
            if g_mom > 0:
                cover_days[si].add(di)
    signals['CHAIN_MOM_LS'] = (buy_days, sell_days, short_days, cover_days)

    # ===== 4. OI_SURGE_LONG: OI激增做多(改进出场) =====
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
            # 做多：OI激增 + 价格在均线之上 + ADX>15
            adx_val = adx_arr[si, d]
            if np.isnan(adx_val) or adx_val < 15:
                continue
            if oi_surge and c > ma20_val:
                buy_days[si].add(di)
            # 出场：OI回落 OR 价格破MA20
            if not oi_surge and c < ma20_val:
                sell_days[si].add(di)
    signals['OI_SURGE_LONG'] = (buy_days, sell_days)

    return signals, atr_arr, adx_arr


# ============================================================
# Long-only 回测引擎 (简化, 更稳定)
# ============================================================
def backtest_long_only(signals, NS, ND, dates, C, O, H, L, V, OI, syms,
                       atr_arr, max_positions=2, atr_sl_mult=2.5, atr_tp_mult=5.0,
                       hold_max=30, dd_reduce=0.3):
    buy_days, sell_days = signals
    cash = float(CASH0)
    positions = []
    trades = []
    year_stats = {}
    peak_equity = float(CASH0)

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # 当前权益
        pos_value = sum(pos['shares'] * (C[pos['si'], di] if not np.isnan(C[pos['si'], di])
                                          else pos['entry_price']) for pos in positions)
        current_eq = cash + pos_value
        if current_eq > peak_equity:
            peak_equity = current_eq
        current_dd = (peak_equity - current_eq) / peak_equity if peak_equity > 0 else 0

        # 回撤控制
        pos_scale = 0.5 if current_dd > dd_reduce else 1.0

        # 平仓检查
        for pos in list(positions):
            si = pos['si']
            c = C[si, di]
            if np.isnan(c):
                continue
            pnl_pct = (c - pos['entry_price']) / pos['entry_price']

            # ATR止损
            atr_val = atr_arr[si, di]
            if not np.isnan(atr_val) and atr_val > 0 and pos['entry_price'] > 0:
                sl_dist = atr_sl_mult * atr_val / pos['entry_price']
                if pnl_pct < -sl_dist:
                    cash += pos['shares'] * c * (1 - COMMISSION)
                    trades.append({'pnl': pnl_pct * 100, 'days': di - pos['entry_di'],
                                   'di': di, 'reason': 'atr_sl', 'year': year, 'si': si})
                    positions.remove(pos)
                    continue
                # ATR止盈
                tp_dist = atr_tp_mult * atr_val / pos['entry_price']
                if pnl_pct > tp_dist:
                    cash += pos['shares'] * c * (1 - COMMISSION)
                    trades.append({'pnl': pnl_pct * 100, 'days': di - pos['entry_di'],
                                   'di': di, 'reason': 'atr_tp', 'year': year, 'si': si})
                    positions.remove(pos)
                    continue

            # 追踪止损：盈利超3%后回撤2%平仓
            if pos.get('peak_pnl', 0) > 0.03:
                if pnl_pct < pos['peak_pnl'] - 0.02:
                    cash += pos['shares'] * c * (1 - COMMISSION)
                    trades.append({'pnl': pnl_pct * 100, 'days': di - pos['entry_di'],
                                   'di': di, 'reason': 'trail', 'year': year, 'si': si})
                    positions.remove(pos)
                    continue

            # 更新峰值盈利
            if pnl_pct > pos.get('peak_pnl', 0):
                pos['peak_pnl'] = pnl_pct

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
            slots = max_positions - len(positions)
            for si in candidates[:slots]:
                c = C[si, di]
                alloc = cash * pos_scale / max(1, max_positions - len(positions))
                shares = int(alloc / (1 + COMMISSION) / c)
                if shares > 0 and shares * c * (1 + COMMISSION) <= cash:
                    cost = shares * c * (1 + COMMISSION)
                    cash -= cost
                    positions.append({
                        'si': si, 'entry_price': c, 'entry_di': di,
                        'shares': shares, 'peak_pnl': 0.0,
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

    exit_reasons = {}
    for t in trades:
        r = t.get('reason', 'unknown')
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    return {
        'ann': round(ann, 1), 'n': len(trades), 'wr': round(wr, 1),
        'max_dd': round(max_dd, 1), 'final': round(cash, 0),
        'year_stats': year_stats, 'exit_reasons': exit_reasons,
    }


def backtest_ls(strategy_signals, NS, ND, dates, C, O, H, L, V, OI, syms,
                atr_arr, max_positions=2, atr_sl_mult=2.5, atr_tp_mult=5.0,
                hold_max=30, dd_reduce=0.3):
    """多空回测 (for CHAIN_MOM etc)"""
    buy_days, sell_days, short_days, cover_days = strategy_signals
    cash = float(CASH0)
    positions = []
    trades = []
    year_stats = {}
    peak_equity = float(CASH0)

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
        pos_scale = 0.5 if current_dd > dd_reduce else 1.0

        for pos in list(positions):
            si = pos['si']
            c = C[si, di]
            if np.isnan(c):
                continue
            if pos['direction'] == 'long':
                pnl_pct = (c - pos['entry_price']) / pos['entry_price']
            else:
                pnl_pct = (pos['entry_price'] - c) / pos['entry_price']

            atr_val = atr_arr[si, di]
            if not np.isnan(atr_val) and atr_val > 0 and pos['entry_price'] > 0:
                sl_dist = atr_sl_mult * atr_val / pos['entry_price']
                if pnl_pct < -sl_dist:
                    if pos['direction'] == 'long':
                        cash += pos['shares'] * c * (1 - COMMISSION)
                    else:
                        cash += pos['shares'] * (2 * pos['entry_price'] - c) * (1 - COMMISSION)
                    trades.append({'pnl': pnl_pct * 100, 'days': di - pos['entry_di'],
                                   'di': di, 'reason': 'atr_sl', 'year': year,
                                   'si': si, 'dir': pos['direction']})
                    positions.remove(pos)
                    continue
                tp_dist = atr_tp_mult * atr_val / pos['entry_price']
                if pnl_pct > tp_dist:
                    if pos['direction'] == 'long':
                        cash += pos['shares'] * c * (1 - COMMISSION)
                    else:
                        cash += pos['shares'] * (2 * pos['entry_price'] - c) * (1 - COMMISSION)
                    trades.append({'pnl': pnl_pct * 100, 'days': di - pos['entry_di'],
                                   'di': di, 'reason': 'atr_tp', 'year': year,
                                   'si': si, 'dir': pos['direction']})
                    positions.remove(pos)
                    continue

            if pos.get('peak_pnl', 0) > 0.03 and pnl_pct < pos['peak_pnl'] - 0.02:
                if pos['direction'] == 'long':
                    cash += pos['shares'] * c * (1 - COMMISSION)
                else:
                    cash += pos['shares'] * (2 * pos['entry_price'] - c) * (1 - COMMISSION)
                trades.append({'pnl': pnl_pct * 100, 'days': di - pos['entry_di'],
                               'di': di, 'reason': 'trail', 'year': year,
                               'si': si, 'dir': pos['direction']})
                positions.remove(pos)
                continue

            if pnl_pct > pos.get('peak_pnl', 0):
                pos['peak_pnl'] = pnl_pct

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

            if di - pos['entry_di'] >= hold_max:
                if pos['direction'] == 'long':
                    cash += pos['shares'] * c * (1 - COMMISSION)
                else:
                    cash += pos['shares'] * (2 * pos['entry_price'] - c) * (1 - COMMISSION)
                trades.append({'pnl': pnl_pct * 100, 'days': di - pos['entry_di'],
                               'di': di, 'reason': 'time', 'year': year,
                               'si': si, 'dir': pos['direction']})
                positions.remove(pos)

        if len(positions) < max_positions:
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
                alloc = cash * pos_scale / max(1, max_positions - len(positions))
                shares = int(alloc / (1 + COMMISSION) / price)
                if shares > 0 and shares * price * (1 + COMMISSION) <= cash:
                    cost = shares * price * (1 + COMMISSION)
                    cash -= cost
                    positions.append({
                        'si': si, 'entry_price': price, 'entry_di': di,
                        'shares': shares, 'direction': direction, 'peak_pnl': 0.0,
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

    return {
        'ann': round(ann, 1), 'n': len(trades), 'wr': round(wr, 1),
        'max_dd': round(max_dd, 1), 'final': round(cash, 0),
        'year_stats': year_stats, 'exit_reasons': exit_reasons,
        'long_n': len(long_t), 'short_n': len(short_t),
        'long_wr': round(sum(1 for t in long_t if t['pnl'] > 0) / max(len(long_t), 1) * 100, 1),
        'short_wr': round(sum(1 for t in short_t if t['pnl'] > 0) / max(len(short_t), 1) * 100, 1),
    }


def all_positive(r):
    ys = r.get('year_stats', {})
    return all(s['total_pnl'] > 0 for s in ys.values())


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 80, flush=True)
    print("  Alpha Futures V4 — 精准优化", flush=True)
    print("=" * 80, flush=True)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    print("\n[Signals] Generating...", flush=True)
    t0 = time.time()
    all_signals, atr_arr, adx_arr = generate_signals_v4(NS, ND, C, O, H, L, V, OI, syms)
    print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

    # 信号统计
    print(f"\n  Signal counts:", flush=True)
    for name, sigs in all_signals.items():
        if len(sigs) == 2:
            bd, sd = sigs
            nb = sum(len(v) for v in bd.values())
            ns = sum(len(v) for v in sd.values())
            print(f"    {name:<25s}: buy={nb:5d} sell={ns:5d}", flush=True)
        else:
            bd, sd, sh, cv = sigs
            nb = sum(len(v) for v in bd.values())
            ns = sum(len(v) for v in sd.values())
            nsh = sum(len(v) for v in sh.values())
            nc = sum(len(v) for v in cv.values())
            print(f"    {name:<25s}: buy={nb:5d} sell={ns:5d} short={nsh:5d} cover={nc:5d}", flush=True)

    # =====================================================================
    # 策略回测
    # =====================================================================
    print(f"\n{'='*80}", flush=True)
    print(f"  单策略回测", flush=True)
    print(f"  {'策略':<25s} {'P':>2s} {'AS':>3s} {'AT':>3s} |"
          f" {'Ann':>7s} {'N':>5s} {'WR':>5s} {'DD':>5s} {'终值':>10s}", flush=True)
    print(f"  {'-'*80}", flush=True)

    results = []

    # Long-only 策略
    for name in ['GAP_LONG', 'GAP_LONG_ADX', 'GAP_LONG_ADX_OI',
                 'VOL_BREAK_LONG', 'VOL_BREAK_LONG_ADX', 'OI_SURGE_LONG']:
        if name not in all_signals:
            continue
        sigs = all_signals[name]
        for max_pos in [2, 3]:
            for atr_sl in [2.0, 2.5, 3.0, 4.0]:
                for atr_tp in [4.0, 6.0, 8.0]:
                    r = backtest_long_only(sigs, NS, ND, dates, C, O, H, L, V, OI, syms,
                                           atr_arr, max_positions=max_pos,
                                           atr_sl_mult=atr_sl, atr_tp_mult=atr_tp,
                                           hold_max=30)
                    if r:
                        r['name'] = name
                        r['max_pos'] = max_pos
                        r['atr_sl'] = atr_sl
                        r['atr_tp'] = atr_tp
                        results.append(r)

    # 多空策略
    for name in ['CHAIN_MOM_LS']:
        if name not in all_signals:
            continue
        sigs = all_signals[name]
        for max_pos in [2, 3]:
            for atr_sl in [2.0, 2.5, 3.0, 4.0]:
                for atr_tp in [4.0, 6.0, 8.0]:
                    r = backtest_ls(sigs, NS, ND, dates, C, O, H, L, V, OI, syms,
                                    atr_arr, max_positions=max_pos,
                                    atr_sl_mult=atr_sl, atr_tp_mult=atr_tp,
                                    hold_max=30)
                    if r:
                        r['name'] = name
                        r['max_pos'] = max_pos
                        r['atr_sl'] = atr_sl
                        r['atr_tp'] = atr_tp
                        results.append(r)

    results.sort(key=lambda x: -x['ann'])
    for r in results[:50]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['name']:<25s} P{r['max_pos']} {r['atr_sl']:3.0f} {r['atr_tp']:3.0f} |"
              f" {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% {r['max_dd']:5.1f}%"
              f" {r['final']:>10.0f}{pos_mark}", flush=True)

    # Best per strategy
    print(f"\n  === Best per strategy ===", flush=True)
    best_per = {}
    for r in results:
        n = r['name']
        if n not in best_per or r['ann'] > best_per[n]['ann']:
            best_per[n] = r
    for r in sorted(best_per.values(), key=lambda x: -x['ann']):
        ex = ', '.join(f"{k}={v}" for k, v in sorted(r['exit_reasons'].items()))
        ls_info = ""
        if 'long_n' in r:
            ls_info = f" | L:{r['long_n']}t/{r['long_wr']}% S:{r['short_n']}t/{r['short_wr']}%"
        print(f"    {r['name']:<25s} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}%{ls_info}", flush=True)
        print(f"      Exits: {ex}", flush=True)

    # =====================================================================
    # Top 5 year-by-year
    # =====================================================================
    print(f"\n  === TOP 5 YEAR-BY-YEAR ===", flush=True)
    for i, r in enumerate(results[:5]):
        print(f"\n  #{i+1}: {r['name']} P{r['max_pos']} AS{r['atr_sl']:.0f} AT{r['atr_tp']:.0f} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            mark = "+" if s['total_pnl'] > 0 else ""
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={mark}{s['total_pnl']:.0f}%",
                  flush=True)

    # =====================================================================
    # V2对比
    # =====================================================================
    print(f"\n  === vs V2 ===", flush=True)
    print(f"  V2: GAP_FOLLOW     +35.9% DD=50.0%", flush=True)
    print(f"  V2: VOL_BREAK_OI   +33.3% DD=47.6%", flush=True)
    print(f"  V2: OI_SURGE       +21.1% DD=79.9%", flush=True)
    print(f"  V3: CHAIN_MOM      +15.9% DD=44.3%", flush=True)
    if results:
        best = results[0]
        print(f"  V4 best: {best['name']} {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)

    # ALL+ 统计
    all_pos = [r for r in results if all_positive(r)]
    if all_pos:
        all_pos.sort(key=lambda x: -x['ann'])
        print(f"\n  ALL+ (所有年份盈利) 策略:", flush=True)
        for r in all_pos[:10]:
            print(f"    {r['name']:<25s} P{r['max_pos']} AS{r['atr_sl']:.0f} AT{r['atr_tp']:.0f} "
                  f"→ {r['ann']:+.1f}% DD={r['max_dd']:.1f}%", flush=True)
    else:
        print(f"\n  无ALL+策略", flush=True)

    print(f"\n{'='*80}", flush=True)
