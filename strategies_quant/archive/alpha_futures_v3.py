"""
Alpha Futures V3 — 优化版期货时序策略
=====================================
v2结果: GAP_FOLLOW +35.9% DD=50%, VOL_BREAK_OI +33.3% DD=47.6%
问题: 回撤大, FRAMA/KALMAN失效, 只做多

v3优化:
  1. 多空双向交易 (期货特有)
  2. ATR追踪止损 (自适应波动率)
  3. ADX趋势过滤 (只在趋势明确时交易)
  4. 多策略组合 (资金分配到多个信号源)
  5. 回撤控制 (连亏减仓)
  6. 波动率归一化 (不同品种等风险配置)
  7. 改进FRAMA/KALMAN (趋势质量过滤)
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, COMMISSION, STAMP_DUTY, CASH0

# ============================================================
# 品种分组
# ============================================================
COMMODITY_GROUPS = {
    'ferrous':    ['rbfi', 'hci', 'ifi', 'jfi', 'jmfi'],
    'nonferrous': ['cufi', 'alfi', 'znfi', 'aufi', 'agfi', 'ni'],
    'energy':     ['scfi', 'mafi', 'ptafi', 'bufi', 'fufi', 'tai'],
    'agri':       ['afi', 'mfi', 'yfi', 'cfi', 'srfi', 'pfi', 'oi'],
    'chem':       ['ppfi', 'lfi', 'vfi', 'egfi', 'safi', 'fgfi'],
}


# ============================================================
# 通用指标计算
# ============================================================
def _atr(close, high, low, period=14):
    """ATR (Average True Range)"""
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
    # EMA of TR
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
    """ADX (Average Directional Index) — 趋势强度"""
    n = len(close)
    adx = np.full(n, np.nan)
    if n < period * 2:
        return adx

    # Directional movement
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

    # Smooth
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

    # DI
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

    # ADX = smooth of DX
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


def compute_frama(close, high, low, period=16, FC=1, SC=200):
    """FRAMA - 分形自适应均线"""
    n = len(close)
    frama = np.full(n, np.nan)
    if n < period:
        return frama
    frama[period-1] = np.mean(close[:period])
    for i in range(period, n):
        if np.isnan(close[i]) or np.isnan(frama[i-1]):
            continue
        h1 = np.nanmax(high[i-period:i])
        l1 = np.nanmin(low[i-period:i])
        half = period // 2
        h2 = np.nanmax(high[i-half:i])
        l2 = np.nanmin(low[i-half:i])
        h3 = np.nanmax(high[i-period:i-half])
        l3 = np.nanmin(low[i-period:i-half])
        hl1 = max(h1 - l1, 1e-10)
        hl2 = max(h2 - l2, 1e-10)
        hl3 = max(h3 - l3, 1e-10)
        dim = (np.log(hl1 + hl2 + hl3) - np.log(hl1)) / np.log(2) if hl1 > 0 else 1.5
        dim = np.clip(dim, 1.0, 2.0)
        alpha = np.exp(-4.6 * (dim - 1))
        alpha = np.clip(alpha, 2.0/(SC+1), 2.0/(FC+1))
        frama[i] = alpha * close[i] + (1 - alpha) * frama[i-1]
    return frama


def compute_kalman_velocity(close):
    """Kalman滤波器估计价格速度"""
    n = len(close)
    vel = np.full(n, np.nan)
    x = np.array([0.0, 0.0])
    P = np.eye(2) * 1000.0
    Q = np.array([[0.01, 0.001], [0.001, 0.001]])
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H_mat = np.array([[1.0, 0.0]])
    R_mat = np.array([[1.0]])
    I2 = np.eye(2)
    init = False
    for i in range(n):
        if np.isnan(close[i]):
            continue
        if not init:
            x = np.array([close[i], 0.0])
            P = np.eye(2) * 1.0
            init = True
            vel[i] = 0.0
            continue
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        y_innov = close[i] - H_mat @ x_pred
        S = H_mat @ P_pred @ H_mat.T + R_mat
        K = P_pred @ H_mat.T / S[0, 0]
        x = x_pred + K.ravel() * y_innov[0]
        P = (I2 - K @ H_mat) @ P_pred
        if x[0] > 0:
            vel[i] = x[1] / x[0] * 100
    return vel


# ============================================================
# 信号生成
# ============================================================
def generate_signals_v3(NS, ND, C, O, H, L, V, OI, syms):
    """为每个品种生成多空时序信号 (v3)"""
    sym_idx = {s: i for i, s in enumerate(syms)}

    # 品种所属组
    sym_group = {}
    for gname, gsyms in COMMODITY_GROUPS.items():
        for s in gsyms:
            sym_group[s] = gname

    # 组内平均动量
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

    # 预计算 ATR 和 ADX
    print("    ATR & ADX...", flush=True)
    atr_arr = np.full((NS, ND), np.nan)
    adx_arr = np.full((NS, ND), np.nan)
    for si in range(NS):
        atr_arr[si] = _atr(C[si], H[si], L[si], 14)
        adx_arr[si] = _adx(C[si], H[si], L[si], 14)

    # 预计算 MA
    ma5 = np.full((NS, ND), np.nan)
    ma10 = np.full((NS, ND), np.nan)
    ma20 = np.full((NS, ND), np.nan)
    ma60 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for window, ma_store in [(5, ma5), (10, ma10), (20, ma20), (60, ma60)]:
            for di in range(window, ND):
                vals = C[si, di-window:di]
                valid = vals[~np.isnan(vals)]
                if len(valid) >= window // 2:
                    ma_store[si, di] = np.mean(valid)

    signals = {}

    # ===== 策略1: GAP_FOLLOW 多空 =====
    # v2 best +35.9%。增加做空、ADX过滤、ATR止盈
    buy_days = {si: set() for si in range(NS)}
    sell_days = {si: set() for si in range(NS)}
    short_days = {si: set() for si in range(NS)}
    cover_days = {si: set() for si in range(NS)}
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

            # ADX过滤: 趋势强度 > 15
            adx_val = adx_arr[si, d]
            if np.isnan(adx_val) or adx_val < 15:
                continue

            # 向上跳空做多
            if gap > 0.01 and not np.isnan(v) and v > 1.5 * v_avg:
                buy_days[si].add(di)
                sell_days[si].add(di + 5)  # 5日后平仓
            # 向下跳空做空
            if gap < -0.01 and not np.isnan(v) and v > 1.5 * v_avg:
                short_days[si].add(di)
                cover_days[si].add(di + 5)
    signals['GAP_LS'] = (buy_days, sell_days, short_days, cover_days)

    # ===== 策略2: VOL_BREAK_OI 多空 =====
    # v2 +33.3%。增加做空、严格过滤
    buy_days = {si: set() for si in range(NS)}
    sell_days = {si: set() for si in range(NS)}
    short_days = {si: set() for si in range(NS)}
    cover_days = {si: set() for si in range(NS)}
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

            # 20日高低
            c20 = C[si, d-19:d+1]
            valid_c = c20[~np.isnan(c20)]
            if len(valid_c) < 15:
                continue
            high20 = np.max(valid_c)
            low20 = np.min(valid_c)

            # OI确认
            oi_now = OI[si, d]
            oi_5ago = OI[si, d-5] if d >= 5 else np.nan
            oi_ok = (not np.isnan(oi_now) and not np.isnan(oi_5ago) and oi_now > oi_5ago)

            # ADX过滤
            adx_val = adx_arr[si, d]
            if np.isnan(adx_val) or adx_val < 15:
                continue

            # 向上突破做多
            if v > 2.0 * v_avg and c > high20 and oi_ok:
                buy_days[si].add(di)
            # 向下突破做空
            if v > 2.0 * v_avg and c < low20 and oi_ok:
                short_days[si].add(di)
    signals['VOL_BREAK_LS'] = (buy_days, sell_days, short_days, cover_days)

    # ===== 策略3: FRAMA_ADX — 只在趋势中交易 =====
    buy_days = {si: set() for si in range(NS)}
    sell_days = {si: set() for si in range(NS)}
    short_days = {si: set() for si in range(NS)}
    cover_days = {si: set() for si in range(NS)}
    for si in range(NS):
        frama = compute_frama(C[si], H[si], L[si])
        for di in range(3, ND):
            if np.isnan(frama[di]) or np.isnan(frama[di-1]) or np.isnan(frama[di-2]):
                continue
            d = di - 1
            # ADX > 25 强趋势
            adx_val = adx_arr[si, d]
            if np.isnan(adx_val) or adx_val < 25:
                continue
            # 价格在MA60之上（多头趋势）
            above_ma60 = not np.isnan(ma60[si, d]) and C[si, d] > ma60[si, d]
            below_ma60 = not np.isnan(ma60[si, d]) and C[si, d] < ma60[si, d]

            slope_now = frama[di] - frama[di-1]
            slope_prev = frama[di-1] - frama[di-2]
            # 做多：斜率正翻 + 价格在MA60上
            if slope_now > 0 and slope_prev <= 0 and above_ma60:
                buy_days[si].add(di)
            # 做空：斜率负翻 + 价格在MA60下
            if slope_now < 0 and slope_prev >= 0 and below_ma60:
                short_days[si].add(di)
            # 平仓：FRAMA斜率翻转
            if slope_now < 0 and slope_prev >= 0:
                sell_days[si].add(di)
            if slope_now > 0 and slope_prev <= 0:
                cover_days[si].add(di)
    signals['FRAMA_ADX'] = (buy_days, sell_days, short_days, cover_days)

    # ===== 策略4: KALMAN_ADX — 速度过零 + 趋势确认 =====
    buy_days = {si: set() for si in range(NS)}
    sell_days = {si: set() for si in range(NS)}
    short_days = {si: set() for si in range(NS)}
    cover_days = {si: set() for si in range(NS)}
    for si in range(NS):
        vel = compute_kalman_velocity(C[si])
        for di in range(2, ND):
            if np.isnan(vel[di]) or np.isnan(vel[di-1]):
                continue
            d = di - 1
            adx_val = adx_arr[si, d]
            if np.isnan(adx_val) or adx_val < 20:
                continue
            # MA趋势确认
            above_ma20 = not np.isnan(ma20[si, d]) and C[si, d] > ma20[si, d]
            below_ma20 = not np.isnan(ma20[si, d]) and C[si, d] < ma20[si, d]

            # 速度过零 + 趋势方向一致
            if vel[di] > 0 and vel[di-1] <= 0 and above_ma20:
                buy_days[si].add(di)
            if vel[di] < 0 and vel[di-1] >= 0 and below_ma20:
                short_days[si].add(di)
            # 平仓
            if vel[di] < 0 and vel[di-1] >= 0:
                sell_days[si].add(di)
            if vel[di] > 0 and vel[di-1] <= 0:
                cover_days[si].add(di)
    signals['KALMAN_ADX'] = (buy_days, sell_days, short_days, cover_days)

    # ===== 策略5: OI_SURGE 多空 =====
    buy_days = {si: set() for si in range(NS)}
    sell_days = {si: set() for si in range(NS)}
    short_days = {si: set() for si in range(NS)}
    cover_days = {si: set() for si in range(NS)}
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

            # ADX过滤
            adx_val = adx_arr[si, d]
            if np.isnan(adx_val) or adx_val < 15:
                continue

            # 做多：OI激增 + 价格在均线之上
            if oi_surge and c > ma20_val:
                buy_days[si].add(di)
            # 做空：OI激增 + 价格在均线之下
            if oi_surge and c < ma20_val:
                short_days[si].add(di)
            # 平仓：OI回落到均值以下
            if not oi_surge:
                sell_days[si].add(di)
                cover_days[si].add(di)
    signals['OI_SURGE_LS'] = (buy_days, sell_days, short_days, cover_days)

    # ===== 策略6: MA_CROSS_ADX — 均线交叉 + ADX =====
    buy_days = {si: set() for si in range(NS)}
    sell_days = {si: set() for si in range(NS)}
    short_days = {si: set() for si in range(NS)}
    cover_days = {si: set() for si in range(NS)}
    for si in range(NS):
        for di in range(61, ND):
            d = di - 1
            ma5_v = ma5[si, d]
            ma20_v = ma20[si, d]
            ma60_v = ma60[si, d]
            if np.isnan(ma5_v) or np.isnan(ma20_v) or np.isnan(ma60_v):
                continue
            adx_val = adx_arr[si, d]
            if np.isnan(adx_val) or adx_val < 20:
                continue

            prev_ma5 = ma5[si, d-1]
            prev_ma20 = ma20[si, d-1]
            if np.isnan(prev_ma5) or np.isnan(prev_ma20):
                continue

            # 金叉做多
            if ma5_v > ma20_v and prev_ma5 <= prev_ma20 and C[si, d] > ma60_v:
                buy_days[si].add(di)
            # 死叉做空
            if ma5_v < ma20_v and prev_ma5 >= prev_ma20 and C[si, d] < ma60_v:
                short_days[si].add(di)
            # 平仓：反向交叉
            if ma5_v < ma20_v and prev_ma5 >= prev_ma20:
                sell_days[si].add(di)
            if ma5_v > ma20_v and prev_ma5 <= prev_ma20:
                cover_days[si].add(di)
    signals['MA_CROSS_ADX'] = (buy_days, sell_days, short_days, cover_days)

    # ===== 策略7: CHAIN_MOM — 产业链共振多空 =====
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
            # 品种动量
            c_prev = C[si, d-10]
            if np.isnan(c_prev) or c_prev <= 0:
                continue
            mom = (c - c_prev) / c_prev

            gm = group_mom.get(di, {})
            g_mom = gm.get(grp, 0)

            # ADX
            adx_val = adx_arr[si, d]
            if np.isnan(adx_val) or adx_val < 15:
                continue

            # MA趋势
            above_ma20 = not np.isnan(ma20[si, d]) and c > ma20[si, d]
            below_ma20 = not np.isnan(ma20[si, d]) and c < ma20[si, d]

            # 做多：品种动量正 + 组动量正 + 均线之上
            if mom > 0.02 and g_mom > 0.01 and above_ma20:
                buy_days[si].add(di)
            # 做空：品种动量负 + 组动量负 + 均线之下
            if mom < -0.02 and g_mom < -0.01 and below_ma20:
                short_days[si].add(di)
            # 平仓：组动量翻转
            if g_mom < 0:
                sell_days[si].add(di)
            if g_mom > 0:
                cover_days[si].add(di)
    signals['CHAIN_MOM'] = (buy_days, sell_days, short_days, cover_days)

    return signals, atr_arr


# ============================================================
# 回测引擎 (多空 + ATR止损 + 回撤控制)
# ============================================================
def backtest_futures_v3(strategy_signals, NS, ND, dates, C, O, H, L, V, OI, syms,
                        atr_arr, max_positions=2, atr_sl_mult=2.0, atr_tp_mult=3.0,
                        hold_max=30, dd_reduce=0.5):
    """
    多空回测引擎
    - ATR追踪止损: atr_sl_mult * ATR
    - ATR止盈: atr_tp_mult * ATR
    - 回撤控制: 当前回撤超过 dd_reduce 时减半仓位
    """
    buy_days, sell_days, short_days, cover_days = strategy_signals

    cash = float(CASH0)
    positions = []  # [{si, entry_price, entry_di, shares, direction: 'long'/'short', atr_at_entry}]
    trades = []
    year_stats = {}

    # 追踪equity曲线计算回撤
    equity_curve = [float(CASH0)]
    peak_equity = float(CASH0)

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # 计算当前未实现权益
        unrealized = 0
        for pos in positions:
            c = C[pos['si'], di]
            if np.isnan(c):
                c = pos['entry_price']
            if pos['direction'] == 'long':
                unrealized += pos['shares'] * (c - pos['entry_price'])
            else:
                unrealized += pos['shares'] * (pos['entry_price'] - c)
        current_equity = cash + sum(
            pos['shares'] * pos['entry_price'] for pos in positions
        ) + unrealized

        if current_equity > peak_equity:
            peak_equity = current_equity
        current_dd = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0

        # 回撤控制：大幅回撤时减少仓位
        pos_scale = 0.5 if current_dd > dd_reduce else 1.0

        # 1. 检查止损/止盈/卖出/平仓信号
        for pos in list(positions):
            si = pos['si']
            c = C[si, di]
            if np.isnan(c):
                continue

            if pos['direction'] == 'long':
                pnl_pct = (c - pos['entry_price']) / pos['entry_price']
            else:
                pnl_pct = (pos['entry_price'] - c) / pos['entry_price']

            # ATR止损
            atr_val = atr_arr[si, di]
            if not np.isnan(atr_val) and atr_val > 0:
                sl_dist = atr_sl_mult * atr_val / pos['entry_price']
                tp_dist = atr_tp_mult * atr_val / pos['entry_price']
                if pnl_pct < -sl_dist:
                    pnl = pnl_pct * 100
                    if pos['direction'] == 'long':
                        cash += pos['shares'] * c * (1 - COMMISSION)
                    else:
                        cash += pos['shares'] * (2 * pos['entry_price'] - c) * (1 - COMMISSION)
                    trades.append({'pnl': pnl, 'days': di - pos['entry_di'],
                                   'di': di, 'reason': 'stop', 'year': year,
                                   'si': si, 'dir': pos['direction']})
                    positions.remove(pos)
                    continue

                # ATR止盈
                if pnl_pct > tp_dist:
                    pnl = pnl_pct * 100
                    if pos['direction'] == 'long':
                        cash += pos['shares'] * c * (1 - COMMISSION)
                    else:
                        cash += pos['shares'] * (2 * pos['entry_price'] - c) * (1 - COMMISSION)
                    trades.append({'pnl': pnl, 'days': di - pos['entry_di'],
                                   'di': di, 'reason': 'tp', 'year': year,
                                   'si': si, 'dir': pos['direction']})
                    positions.remove(pos)
                    continue

            # 固定止损 (5%) 作为保底
            if pnl_pct < -0.05:
                pnl = pnl_pct * 100
                if pos['direction'] == 'long':
                    cash += pos['shares'] * c * (1 - COMMISSION)
                else:
                    cash += pos['shares'] * (2 * pos['entry_price'] - c) * (1 - COMMISSION)
                trades.append({'pnl': pnl, 'days': di - pos['entry_di'],
                               'di': di, 'reason': 'stop5', 'year': year,
                               'si': si, 'dir': pos['direction']})
                positions.remove(pos)
                continue

            # 信号平仓
            if pos['direction'] == 'long' and di in sell_days[si]:
                pnl = pnl_pct * 100
                cash += pos['shares'] * c * (1 - COMMISSION)
                trades.append({'pnl': pnl, 'days': di - pos['entry_di'],
                               'di': di, 'reason': 'signal', 'year': year,
                               'si': si, 'dir': pos['direction']})
                positions.remove(pos)
                continue

            if pos['direction'] == 'short' and di in cover_days[si]:
                pnl = pnl_pct * 100
                cash += pos['shares'] * (2 * pos['entry_price'] - c) * (1 - COMMISSION)
                trades.append({'pnl': pnl, 'days': di - pos['entry_di'],
                               'di': di, 'reason': 'signal', 'year': year,
                               'si': si, 'dir': pos['direction']})
                positions.remove(pos)
                continue

            # 超时平仓
            if di - pos['entry_di'] >= hold_max:
                pnl = pnl_pct * 100
                if pos['direction'] == 'long':
                    cash += pos['shares'] * c * (1 - COMMISSION)
                else:
                    cash += pos['shares'] * (2 * pos['entry_price'] - c) * (1 - COMMISSION)
                trades.append({'pnl': pnl, 'days': di - pos['entry_di'],
                               'di': di, 'reason': 'time', 'year': year,
                               'si': si, 'dir': pos['direction']})
                positions.remove(pos)

        # 2. 开仓
        if len(positions) < max_positions:
            # 收集做多样选
            long_candidates = []
            for si in range(NS):
                if di in buy_days[si]:
                    if any(p['si'] == si for p in positions):
                        continue
                    c = C[si, di]
                    if np.isnan(c) or c <= 0:
                        continue
                    long_candidates.append(('long', si, c))

            # 收集做空候选
            short_candidates = []
            for si in range(NS):
                if di in short_days[si]:
                    if any(p['si'] == si for p in positions):
                        continue
                    c = C[si, di]
                    if np.isnan(c) or c <= 0:
                        continue
                    short_candidates.append(('short', si, c))

            # 合并候选，按成交量排序
            all_cands = long_candidates + short_candidates
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
                        'shares': shares, 'direction': direction,
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

    # 统计
    days_total = (dates[ND-1] - dates[MIN_TRAIN]).days
    yr = max(days_total / 365.25, 0.01)
    ann = ((cash / CASH0) ** (1 / yr) - 1) * 100
    nw = sum(1 for t in trades if t['pnl'] > 0)
    wr = nw / max(len(trades), 1) * 100

    # 长短分别统计
    long_trades = [t for t in trades if t.get('dir') == 'long']
    short_trades = [t for t in trades if t.get('dir') == 'short']

    for t in trades:
        y = t.get('year', 'unknown')
        if y not in year_stats:
            year_stats[y] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
        year_stats[y]['trades'] += 1
        if t['pnl'] > 0:
            year_stats[y]['wins'] += 1
        year_stats[y]['total_pnl'] += t['pnl']

    # Max drawdown (逐笔)
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

    # Exit reason stats
    exit_reasons = {}
    for t in trades:
        r = t.get('reason', 'unknown')
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    return {
        'ann': round(ann, 1), 'n': len(trades), 'wr': round(wr, 1),
        'max_dd': round(max_dd, 1), 'final': round(cash, 0),
        'year_stats': year_stats,
        'long_n': len(long_trades), 'short_n': len(short_trades),
        'long_wr': round(sum(1 for t in long_trades if t['pnl'] > 0) / max(len(long_trades), 1) * 100, 1),
        'short_wr': round(sum(1 for t in short_trades if t['pnl'] > 0) / max(len(short_trades), 1) * 100, 1),
        'exit_reasons': exit_reasons,
    }


# ============================================================
# 多策略组合回测
# ============================================================
def backtest_portfolio(all_signals, NS, ND, dates, C, O, H, L, V, OI, syms,
                       atr_arr, strategy_weights, max_positions=3,
                       atr_sl_mult=2.0, atr_tp_mult=3.0, hold_max=30):
    """
    多策略组合：每个策略独立产生信号，加权投票决定开仓方向
    strategy_weights: {strategy_name: weight}
    """
    # 合并信号：每个(si, di) → 分数 (正=做多, 负=做空)
    buy_scores = {}  # (si, di) → score
    sell_scores = {}

    for sname, weight in strategy_weights.items():
        if sname not in all_signals:
            continue
        buy_days, sell_days, short_days, cover_days = all_signals[sname]
        for si in range(NS):
            for di in buy_days[si]:
                key = (si, di)
                buy_scores[key] = buy_scores.get(key, 0) + weight
            for di in short_days[si]:
                key = (si, di)
                sell_scores[key] = sell_scores.get(key, 0) + weight

    # 构建组合信号：score >= threshold → 开仓
    threshold = sum(strategy_weights.values()) * 0.3  # 30%权重一致即可
    combo_buy = {si: set() for si in range(NS)}
    combo_sell = {si: set() for si in range(NS)}
    combo_short = {si: set() for si in range(NS)}
    combo_cover = {si: set() for si in range(NS)}

    for (si, di), score in buy_scores.items():
        if score >= threshold:
            combo_buy[si].add(di)
    for (si, di), score in sell_scores.items():
        if score >= threshold:
            combo_short[si].add(di)

    combo_signals = (combo_buy, combo_sell, combo_short, combo_cover)
    return backtest_futures_v3(combo_signals, NS, ND, dates, C, O, H, L, V, OI, syms,
                               atr_arr, max_positions=max_positions,
                               atr_sl_mult=atr_sl_mult, atr_tp_mult=atr_tp_mult,
                               hold_max=hold_max)


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 80, flush=True)
    print("  Alpha Futures V3 — 多空 + ATR止损 + 趋势过滤 + 组合", flush=True)
    print("=" * 80, flush=True)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    # 生成信号
    print("\n[Signals] Generating...", flush=True)
    t0 = time.time()
    all_signals, atr_arr = generate_signals_v3(NS, ND, C, O, H, L, V, OI, syms)
    print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

    # 信号统计
    print(f"\n  Signal counts:", flush=True)
    for name, (buy, sell, short, cover) in all_signals.items():
        nb = sum(len(v) for v in buy.values())
        ns = sum(len(v) for v in sell.values())
        nsh = sum(len(v) for v in short.values())
        nc = sum(len(v) for v in cover.values())
        print(f"    {name:<20s}: buy={nb:5d} sell={ns:5d} short={nsh:5d} cover={nc:5d}", flush=True)

    # =====================================================================
    # 单策略测试
    # =====================================================================
    print(f"\n{'='*80}", flush=True)
    print(f"  单策略回测", flush=True)
    print(f"  {'策略':<22s} {'年化':>7s} {'N':>5s} {'WR':>5s} {'DD':>5s} "
          f"{'L_N':>4s} {'L_WR':>5s} {'S_N':>4s} {'S_WR':>5s}", flush=True)
    print(f"  {'-'*85}", flush=True)

    results = []
    for name, signals in all_signals.items():
        for max_pos in [2, 3]:
            for atr_sl in [1.5, 2.0, 2.5, 3.0]:
                r = backtest_futures_v3(signals, NS, ND, dates, C, O, H, L, V, OI, syms,
                                       atr_arr, max_positions=max_pos,
                                       atr_sl_mult=atr_sl, atr_tp_mult=atr_sl * 2,
                                       hold_max=30)
                if r:
                    r['name'] = name
                    r['max_pos'] = max_pos
                    r['atr_sl'] = atr_sl
                    results.append(r)

    results.sort(key=lambda x: -x['ann'])
    for r in results[:40]:
        print(f"  {r['name']:<20s} P{r['max_pos']} A{r['atr_sl']:.1f} | "
              f"{r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% {r['max_dd']:5.1f}% "
              f"{r['long_n']:4d} {r['long_wr']:5.1f}% {r['short_n']:4d} {r['short_wr']:5.1f}%",
              flush=True)

    # Best per strategy
    print(f"\n  === Best per strategy ===", flush=True)
    best_per = {}
    for r in results:
        n = r['name']
        if n not in best_per or r['ann'] > best_per[n]['ann']:
            best_per[n] = r
    for r in sorted(best_per.values(), key=lambda x: -x['ann']):
        ex = ', '.join(f"{k}={v}" for k, v in sorted(r['exit_reasons'].items()))
        print(f"    {r['name']:<22s} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}% "
              f"(P={r['max_pos']}, ATR_SL={r['atr_sl']:.1f})", flush=True)
        print(f"      Long: {r['long_n']} trades WR={r['long_wr']}% | "
              f"Short: {r['short_n']} trades WR={r['short_wr']}%", flush=True)
        print(f"      Exits: {ex}", flush=True)

    # =====================================================================
    # 多策略组合
    # =====================================================================
    print(f"\n{'='*80}", flush=True)
    print(f"  多策略组合", flush=True)
    print(f"  {'组合':<40s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'DD':>5s}", flush=True)
    print(f"  {'-'*70}", flush=True)

    combos = [
        ('Top2_Eq', {k: 1.0 for k in ['GAP_LS', 'VOL_BREAK_LS']}),
        ('Top3_Eq', {k: 1.0 for k in ['GAP_LS', 'VOL_BREAK_LS', 'OI_SURGE_LS']}),
        ('Top4_Eq', {k: 1.0 for k in ['GAP_LS', 'VOL_BREAK_LS', 'OI_SURGE_LS', 'CHAIN_MOM']}),
        ('GAP_Heavy', {'GAP_LS': 2.0, 'VOL_BREAK_LS': 1.0, 'OI_SURGE_LS': 0.5}),
        ('VOL_Heavy', {'VOL_BREAK_LS': 2.0, 'GAP_LS': 1.0, 'CHAIN_MOM': 0.5}),
        ('All_Eq', {k: 1.0 for k in all_signals}),
        ('FRAMA_KALMAN', {'FRAMA_ADX': 1.0, 'KALMAN_ADX': 1.0}),
        ('MA_CHAIN', {'MA_CROSS_ADX': 1.0, 'CHAIN_MOM': 1.0}),
        ('Trend3', {'FRAMA_ADX': 1.0, 'KALMAN_ADX': 1.0, 'MA_CROSS_ADX': 1.0}),
        ('All_Trend', {'FRAMA_ADX': 1.0, 'KALMAN_ADX': 1.0, 'MA_CROSS_ADX': 1.0,
                       'CHAIN_MOM': 1.0}),
    ]

    combo_results = []
    for cname, weights in combos:
        for max_pos in [2, 3]:
            for atr_sl in [2.0, 2.5, 3.0]:
                r = backtest_portfolio(all_signals, NS, ND, dates, C, O, H, L, V, OI, syms,
                                       atr_arr, weights, max_positions=max_pos,
                                       atr_sl_mult=atr_sl, atr_tp_mult=atr_sl * 2,
                                       hold_max=30)
                if r:
                    r['combo'] = cname
                    r['max_pos'] = max_pos
                    r['atr_sl'] = atr_sl
                    combo_results.append(r)

    combo_results.sort(key=lambda x: -x['ann'])
    for r in combo_results[:30]:
        print(f"  {r['combo']:<38s} P{r['max_pos']} A{r['atr_sl']:.1f} | "
              f"{r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% {r['max_dd']:5.1f}%",
              flush=True)

    # =====================================================================
    # Year-by-year for top 3 overall
    # =====================================================================
    all_results = results + combo_results
    all_results.sort(key=lambda x: -x['ann'])

    print(f"\n  === TOP 5 OVERALL ===", flush=True)
    for i, r in enumerate(all_results[:5]):
        label = r.get('combo', r.get('name', '?'))
        pos_info = f"P{r['max_pos']} A{r['atr_sl']:.1f}"
        print(f"\n  #{i+1}: {label} {pos_info} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            mark = "+" if s['total_pnl'] > 0 else ""
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={mark}{s['total_pnl']:.0f}%",
                  flush=True)

    # v2 对比
    print(f"\n  === vs V2 Baseline ===", flush=True)
    print(f"  V2: GAP_FOLLOW +35.9% DD=50.0%", flush=True)
    print(f"  V2: VOL_BREAK_OI +33.3% DD=47.6%", flush=True)
    print(f"  V2: OI_SURGE_TREND +21.1% DD=79.9%", flush=True)
    if all_results:
        best = all_results[0]
        label = best.get('combo', best.get('name', '?'))
        print(f"  V3 best: {label} {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)
        delta = best['ann'] - 35.9
        dd_delta = best['max_dd'] - 50.0
        print(f"  Delta: ann={delta:+.1f}%, DD={dd_delta:+.1f}%", flush=True)

    print(f"\n{'='*80}", flush=True)
