"""
Alpha Futures — 期货市场专属时序信号策略
==========================================
v1 (横截面排名) 失败：68个品种不够，排名噪声大

v2 策略：时序信号 + OI/产业链确认
  - 基于v2已验证有效的信号 (FRAMA交叉 +40.2%, Kalman速度 +32.0%)
  - 增加 OI 确认过滤 (持仓量增加时信号更可靠)
  - 增加产业链共振过滤 (同组品种趋势一致时入场)
  - 增加隔夜缺口信号 (期货特有)
  - 使用单品种时序回测，不走横截面排名
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
# 信号计算函数
# ============================================================

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


def generate_signals(NS, ND, C, O, H, L, V, OI, syms):
    """为每个品种生成时序买卖信号"""
    sym_idx = {s: i for i, s in enumerate(syms)}

    # 计算组内平均动量 (用于共振过滤)
    group_mom = {}  # {date_idx: {group: avg_mom}}
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

    # 品种所属组
    sym_group = {}
    for gname, gsyms in COMMODITY_GROUPS.items():
        for s in gsyms:
            sym_group[s] = gname

    signals = {}  # {strategy_name: {si: set_of_buy_dis, si: set_of_sell_dis}}

    # ===== 策略1: FRAMA交叉 + OI确认 =====
    buy_days = {si: set() for si in range(NS)}
    sell_days = {si: set() for si in range(NS)}
    for si in range(NS):
        frama = compute_frama(C[si], H[si], L[si])
        for di in range(2, ND):
            if np.isnan(frama[di]) or np.isnan(frama[di-1]):
                continue
            slope_now = frama[di] - frama[di-1]
            slope_prev = frama[di-1] - frama[di-2] if di >= 2 and not np.isnan(frama[di-2]) else 0
            # 买入：斜率由负变正
            if slope_now > 0 and slope_prev <= 0:
                buy_days[si].add(di)
            # 卖出：斜率由正变负
            if slope_now < 0 and slope_prev >= 0:
                sell_days[si].add(di)
    signals['FRAMA'] = (buy_days, sell_days)

    # ===== 策略2: FRAMA + OI确认 =====
    buy_days = {si: set() for si in range(NS)}
    sell_days = {si: set() for si in range(NS)}
    for si in range(NS):
        frama = compute_frama(C[si], H[si], L[si])
        for di in range(2, ND):
            if np.isnan(frama[di]) or np.isnan(frama[di-1]):
                continue
            slope_now = frama[di] - frama[di-1]
            slope_prev = frama[di-1] - frama[di-2] if di >= 2 and not np.isnan(frama[di-2]) else 0
            d = di - 1
            oi_now = OI[si, d]
            oi_5ago = OI[si, d-5] if d >= 5 else np.nan
            oi_increasing = (not np.isnan(oi_now) and not np.isnan(oi_5ago) and oi_now > oi_5ago)
            # 买入：FRAMA正 + OI增加 (新资金入场)
            if slope_now > 0 and slope_prev <= 0 and oi_increasing:
                buy_days[si].add(di)
            # 卖出：FRAMA负 + OI减少 (资金撤离)
            oi_decreasing = (not np.isnan(oi_now) and not np.isnan(oi_5ago) and oi_now < oi_5ago)
            if slope_now < 0 and slope_prev >= 0 and oi_decreasing:
                sell_days[si].add(di)
            # 也可以在FRAMA翻负时就卖
            elif slope_now < 0 and slope_prev >= 0:
                sell_days[si].add(di)
    signals['FRAMA_OI'] = (buy_days, sell_days)

    # ===== 策略3: Kalman速度过零 =====
    buy_days = {si: set() for si in range(NS)}
    sell_days = {si: set() for si in range(NS)}
    for si in range(NS):
        vel = compute_kalman_velocity(C[si])
        for di in range(2, ND):
            if np.isnan(vel[di]) or np.isnan(vel[di-1]):
                continue
            if vel[di] > 0 and vel[di-1] <= 0:
                buy_days[si].add(di)
            if vel[di] < 0 and vel[di-1] >= 0:
                sell_days[si].add(di)
    signals['KALMAN'] = (buy_days, sell_days)

    # ===== 策略4: Kalman + OI确认 + 产业链共振 =====
    buy_days = {si: set() for si in range(NS)}
    sell_days = {si: set() for si in range(NS)}
    for si in range(NS):
        vel = compute_kalman_velocity(C[si])
        sym = syms[si]
        grp = sym_group.get(sym)
        for di in range(21, ND):
            if np.isnan(vel[di]) or np.isnan(vel[di-1]):
                continue
            d = di - 1
            # OI确认
            oi_now = OI[si, d]
            oi_5ago = OI[si, d-5] if d >= 5 else np.nan
            oi_ok = (not np.isnan(oi_now) and not np.isnan(oi_5ago) and oi_now > oi_5ago)
            # 产业链共振
            gm = group_mom.get(di, {})
            group_ok = True
            if grp and grp in gm:
                group_ok = gm[grp] > 0  # 组内平均动量为正

            # 买入：Kalman正 + OI增加 + 产业链正
            if vel[di] > 0 and vel[di-1] <= 0 and oi_ok and group_ok:
                buy_days[si].add(di)
            # 卖出：Kalman负
            if vel[di] < 0 and vel[di-1] >= 0:
                sell_days[si].add(di)
    signals['KALMAN_OI_CHAIN'] = (buy_days, sell_days)

    # ===== 策略5: 放量突破 + OI确认 =====
    buy_days = {si: set() for si in range(NS)}
    sell_days = {si: set() for si in range(NS)}
    for si in range(NS):
        for di in range(22, ND):
            d = di - 1
            c = C[si, d]
            v = V[si, d]
            oi = OI[si, d]
            if np.isnan(c) or np.isnan(v) or v <= 0:
                continue
            # 20日高点
            high20 = np.nanmax(C[si, d-19:d])
            if np.isnan(high20):
                continue
            # 放量
            v_window = V[si, d-19:d+1]
            valid_v = v_window[~np.isnan(v_window)]
            if len(valid_v) < 10:
                continue
            v_avg = np.mean(valid_v)
            vol_breakout = v > 2.0 * v_avg and c > high20
            # OI确认
            oi_now = OI[si, d]
            oi_5ago = OI[si, d-5] if d >= 5 else np.nan
            oi_ok = (not np.isnan(oi_now) and not np.isnan(oi_5ago) and oi_now > oi_5ago)

            if vol_breakout and oi_ok:
                buy_days[si].add(di)
        # 卖出：5%止损 或 10日最低
        for di in list(buy_days[si]):
            entry_price = C[si, di-1] if di > 0 and not np.isnan(C[si, di-1]) else None
            if entry_price is None:
                continue
            for hold_di in range(di+1, min(di+30, ND)):
                c = C[si, hold_di]
                if np.isnan(c):
                    continue
                if c < entry_price * 0.95:  # 5%止损
                    sell_days[si].add(hold_di)
                    break
                if c < entry_price * 0.97 and hold_di - di >= 10:  # 10日后3%止损
                    sell_days[si].add(hold_di)
                    break
    signals['VOL_BREAK_OI'] = (buy_days, sell_days)

    # ===== 策略6: 隔夜缺口跟踪 =====
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
            # 大幅向上跳空 + 成交量放大 = 跟踪
            v = V[si, d]
            v_window = V[si, d-19:d+1]
            valid_v = v_window[~np.isnan(v_window)]
            if len(valid_v) < 10:
                continue
            v_avg = np.mean(valid_v)
            if gap > 0.01 and not np.isnan(v) and v > 1.5 * v_avg:  # >1%跳空 + 放量
                buy_days[si].add(di)
            # 持有5天后卖出
            sell_di = di + 5
            if sell_di < ND:
                sell_days[si].add(sell_di)
    signals['GAP_FOLLOW'] = (buy_days, sell_days)

    # ===== 策略7: OI激增 + 趋势确认 =====
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
            # 趋势确认：价格在20日均线之上
            c = C[si, d]
            c20 = C[si, d-19:d+1]
            valid_c = c20[~np.isnan(c20)]
            if len(valid_c) < 15:
                continue
            ma20 = np.mean(valid_c)
            trend_up = c > ma20

            if oi_surge and trend_up:
                buy_days[si].add(di)
            # 卖出：OI回落到均值以下 + 价格破均线
            if not oi_surge and c < ma20:
                sell_days[si].add(di)
    signals['OI_SURGE_TREND'] = (buy_days, sell_days)

    return signals


# ============================================================
# 回测引擎 (时序信号，非横截面)
# ============================================================
def backtest_futures(signals, NS, ND, dates, C, O, H, L, V, OI, syms,
                    max_positions=2, sl_pct=0.05, hold_max=30):
    """回测时序信号策略，多品种轮动"""
    buy_days, sell_days = signals

    cash = float(CASH0)
    positions = []  # [{si, entry_price, entry_di, shares}]
    trades = []
    year_stats = {}

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # 1. 检查止损和卖出信号
        for pos in list(positions):
            si = pos['si']
            c = C[si, di]
            if np.isnan(c):
                continue

            # 止损
            pnl_pct = (c - pos['entry_price']) / pos['entry_price']
            if pnl_pct < -sl_pct:
                sp = c
                pnl = (sp - pos['entry_price']) / pos['entry_price'] * 100
                cash += pos['shares'] * sp * (1 - COMMISSION)
                trades.append({'pnl': pnl, 'days': di - pos['entry_di'],
                               'di': di, 'reason': 'stop', 'year': year, 'si': si})
                positions.remove(pos)
                continue

            # 信号卖出
            if di in sell_days[si]:
                sp = c
                pnl = (sp - pos['entry_price']) / pos['entry_price'] * 100
                cash += pos['shares'] * sp * (1 - COMMISSION)
                trades.append({'pnl': pnl, 'days': di - pos['entry_di'],
                               'di': di, 'reason': 'signal', 'year': year, 'si': si})
                positions.remove(pos)
                continue

            # 超时卖出
            if di - pos['entry_di'] >= hold_max:
                sp = c
                pnl = (sp - pos['entry_price']) / pos['entry_price'] * 100
                cash += pos['shares'] * sp * (1 - COMMISSION)
                trades.append({'pnl': pnl, 'days': di - pos['entry_di'],
                               'di': di, 'reason': 'time', 'year': year, 'si': si})
                positions.remove(pos)

        # 2. 买入信号
        if len(positions) < max_positions:
            candidates = []
            for si in range(NS):
                if di in buy_days[si]:
                    # 已持有的不再买
                    if any(p['si'] == si for p in positions):
                        continue
                    c = C[si, di]
                    if np.isnan(c) or c <= 0:
                        continue
                    candidates.append(si)

            # 按成交量排序（优先流动性好的）
            candidates.sort(key=lambda si: -V[si, di] if not np.isnan(V[si, di]) else 0)

            for si in candidates[:max_positions - len(positions)]:
                c = C[si, di]
                alloc = cash / (max_positions - len(positions))
                shares = int(alloc / (1 + COMMISSION) / c)
                if shares > 0 and shares * c * (1 + COMMISSION) <= cash:
                    cost = shares * c * (1 + COMMISSION)
                    cash -= cost
                    positions.append({
                        'si': si, 'entry_price': c, 'entry_di': di, 'shares': shares
                    })

    # 平仓
    for pos in positions:
        c = C[pos['si'], ND-1]
        if not np.isnan(c) and c > 0:
            pnl = (c - pos['entry_price']) / pos['entry_price'] * 100
            cash += pos['shares'] * c * (1 - COMMISSION)
            trades.append({'pnl': pnl, 'days': 999, 'di': ND-1, 'reason': 'end',
                           'year': dates[ND-1].year, 'si': pos['si']})

    if not trades:
        return None

    # 统计
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

    # Max drawdown
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


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 80, flush=True)
    print("  Alpha Futures v2 — 时序信号 + OI/产业链确认", flush=True)
    print("=" * 80, flush=True)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    # 生成信号
    print("\n[Signals] Generating...", flush=True)
    t0 = time.time()
    all_signals = generate_signals(NS, ND, C, O, H, L, V, OI, syms)
    print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

    # 回测每个策略
    print(f"\n{'='*80}", flush=True)
    print(f"  {'策略':<25s} {'年化':>7s} {'交易数':>5s} {'胜率':>5s} {'DD':>5s} {'终值':>10s}", flush=True)
    print(f"  {'-'*65}", flush=True)

    results = []
    for name, signals in all_signals.items():
        for max_pos in [2, 3]:
            for sl in [0.03, 0.05]:
                r = backtest_futures(signals, NS, ND, dates, C, O, H, L, V, OI, syms,
                                   max_positions=max_pos, sl_pct=sl, hold_max=30)
                if r:
                    r['name'] = name
                    r['max_pos'] = max_pos
                    r['sl'] = sl
                    results.append(r)

    results.sort(key=lambda x: -x['ann'])
    for r in results[:30]:
        print(f"  {r['name']:<20s} P{r['max_pos']} SL{r['sl']:.0%} | "
              f"{r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['max_dd']:5.1f}% {r['final']:>10.0f}", flush=True)

    # Best per strategy
    print(f"\n  === Best per strategy ===", flush=True)
    best_per = {}
    for r in results:
        n = r['name']
        if n not in best_per or r['ann'] > best_per[n]['ann']:
            best_per[n] = r
    for r in sorted(best_per.values(), key=lambda x: -x['ann']):
        print(f"    {r['name']:<25s} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}% "
              f"(P={r['max_pos']}, SL={r['sl']:.0%})", flush=True)

    # Year-by-year for top 3
    for i, r in enumerate(list(sorted(best_per.values(), key=lambda x: -x['ann']))[:3]):
        print(f"\n  Year-by-year #{i+1}: {r['name']} (Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    print(f"\n{'='*80}", flush=True)
