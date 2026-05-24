"""
Alpha Futures V8c — 期货合约回测 (保证金 + OI因子)
===================================================
保证金 = 价格 × 乘数 × 保证金率
P&L = (exit - entry) × 乘数 × 手数 × 方向
加入: OI持仓量因子, 跨品种动量, 日内波动因子
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, COMMISSION, STAMP_DUTY, CASH0

# 合约乘数 + 保证金率
SPECS = {
    'agfi': (15, 0.12), 'alfi': (5, 0.10), 'aufi': (1000, 0.10),
    'bufi': (10, 0.10), 'cufi': (5, 0.10), 'fufi': (10, 0.10),
    'rbfi': (10, 0.10), 'znfi': (5, 0.10), 'nifi': (1, 0.12),
    'hcfi': (10, 0.10), 'spfi': (10, 0.10), 'ssfi': (5, 0.10),
    'sffi': (5, 0.10), 'smfi': (5, 0.10), 'pbfi': (5, 0.10),
    'snfi': (1, 0.12), 'rufi': (10, 0.10), 'wrffi': (10, 0.08),
    'afi': (10, 0.08), 'bfi': (10, 0.08), 'bbfi': (500, 0.10),
    'cffi': (5, 0.07), 'cfi': (10, 0.08), 'csfi': (10, 0.08),
    'ebfi': (5, 0.08), 'egfi': (10, 0.08), 'fbfi': (500, 0.10),
    'ifi': (100, 0.12), 'jfi': (100, 0.12), 'jmfi': (60, 0.12),
    'lfi': (5, 0.08), 'mfi': (10, 0.08), 'pgfi': (20, 0.10),
    'ppfi': (5, 0.08), 'vfi': (5, 0.08), 'yfi': (10, 0.08),
    'pfi': (10, 0.08), 'jdfi': (5, 0.10), 'lhfi': (16, 0.12),
    'pkfi': (5, 0.08), 'rrfi': (20, 0.08), 'lrfi': (20, 0.08),
    'jrfi': (20, 0.08), 'pmfi': (20, 0.08), 'whfi': (20, 0.08),
    'rsfi': (20, 0.08), 'cjfi': (10, 0.08), 'mafi': (10, 0.08),
    'apfi': (10, 0.08), 'cyfi': (5, 0.08), 'fgfi': (20, 0.08),
    'oifi': (10, 0.08), 'pfifi': (5, 0.08), 'rmfi': (10, 0.08),
    'srfi': (10, 0.08), 'tafi': (5, 0.08), 'safi': (20, 0.08),
    'urfi': (20, 0.08), 'scfi': (1000, 0.12), 'lufi': (10, 0.10),
    'bcfi': (5, 0.10), 'nrfi': (1, 0.12), 'lgfi': (20, 0.10),
    'brfi': (5, 0.10), 'lcfi': (1, 0.12), 'sifi': (5, 0.12),
    'ni': (1, 0.12), 'tai': (5, 0.08),
}
DEF_MULT = 10
DEF_MARGIN = 0.10
COMM_RATE = 0.0003

# 品种分组 (产业链)
GROUPS = {
    'black': ['rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi'],       # 黑色
    'metal': ['cufi', 'alfi', 'znfi', 'nifi', 'pbfi', 'snfi'],  # 有色
    'precious': ['aufi', 'agfi'],                            # 贵金属
    'energy': ['scfi', 'mafi', 'fufi', 'pgfi', 'bufi', 'ebfi', 'egfi'],  # 能源化工
    'oil_chain': ['scfi', 'fufi', 'pgfi', 'bufi', 'ebfi', 'egfi', 'mafi', 'tafi', 'ppfi', 'vfi', 'lfi'],
    'farm': ['afi', 'mfi', 'yfi', 'pfi', 'cfi', 'srfi', 'oifi', 'rmfi'],  # 农产品
}

def get_mult(sym):
    return SPECS.get(sym, (DEF_MULT, DEF_MARGIN))[0]

def get_margin_rate(sym):
    return SPECS.get(sym, (DEF_MULT, DEF_MARGIN))[1]


def backtest(buy_days, sell_days, short_days, cover_days,
             NS, ND, dates, C, O, H, L, V, OI, syms, sym_set,
             max_positions=1, sl_pct=0.10, hold_max=60, margin_pct=0.35):
    cash = float(CASH0)
    positions = []
    trades = []
    year_stats = {}

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # 平仓检查
        for pos in list(positions):
            c = C[pos['si'], di]
            if np.isnan(c):
                continue
            mult = get_mult(pos['sym'])

            if pos['dir'] == 1:
                floating_pnl = (c - pos['entry']) * mult * pos['lots']
            else:
                floating_pnl = (pos['entry'] - c) * mult * pos['lots']

            entry_cost = pos['entry'] * mult * pos['lots']
            pnl_pct = floating_pnl / entry_cost * 100

            # 止损
            if pnl_pct / 100 < -sl_pct:
                cash += pos['margin_used'] + floating_pnl
                cash -= abs(c * mult * pos['lots']) * COMM_RATE
                trades.append({'pnl_pct': pnl_pct, 'pnl_abs': floating_pnl,
                               'days': di - pos['entry_di'], 'di': di,
                               'reason': 'stop', 'year': year,
                               'si': pos['si'], 'dir': pos['dir']})
                positions.remove(pos)
                continue

            # 信号平仓
            if pos['dir'] == 1 and di in sell_days.get(pos['si'], set()):
                cash += pos['margin_used'] + floating_pnl
                cash -= abs(c * mult * pos['lots']) * COMM_RATE
                trades.append({'pnl_pct': pnl_pct, 'pnl_abs': floating_pnl,
                               'days': di - pos['entry_di'], 'di': di,
                               'reason': 'signal', 'year': year,
                               'si': pos['si'], 'dir': pos['dir']})
                positions.remove(pos)
                continue
            if pos['dir'] == -1 and di in cover_days.get(pos['si'], set()):
                cash += pos['margin_used'] + floating_pnl
                cash -= abs(c * mult * pos['lots']) * COMM_RATE
                trades.append({'pnl_pct': pnl_pct, 'pnl_abs': floating_pnl,
                               'days': di - pos['entry_di'], 'di': di,
                               'reason': 'signal', 'year': year,
                               'si': pos['si'], 'dir': pos['dir']})
                positions.remove(pos)
                continue

            # 超时平仓
            if di - pos['entry_di'] >= hold_max:
                cash += pos['margin_used'] + floating_pnl
                cash -= abs(c * mult * pos['lots']) * COMM_RATE
                trades.append({'pnl_pct': pnl_pct, 'pnl_abs': floating_pnl,
                               'days': di - pos['entry_di'], 'di': di,
                               'reason': 'time', 'year': year,
                               'si': pos['si'], 'dir': pos['dir']})
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
                mult = get_mult(sym)
                mr = get_margin_rate(sym)
                margin_per_lot = price * mult * mr
                if margin_per_lot <= 0:
                    continue
                alloc = cash * margin_pct / max(1, max_positions - len(positions))
                lots = int(alloc / margin_per_lot)
                if lots > 0:
                    margin_needed = margin_per_lot * lots
                    if margin_needed <= cash:
                        cash -= margin_needed
                        positions.append({
                            'si': si, 'entry': price, 'entry_di': di,
                            'lots': lots, 'dir': direction, 'sym': sym,
                            'margin_used': margin_needed,
                        })

    # 清仓
    for pos in positions:
        c = C[pos['si'], ND-1]
        if np.isnan(c) or c <= 0:
            c = pos['entry']
        mult = get_mult(pos['sym'])
        if pos['dir'] == 1:
            pnl = (c - pos['entry']) * mult * pos['lots']
        else:
            pnl = (pos['entry'] - c) * mult * pos['lots']
        pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100
        cash += pos['margin_used'] + pnl
        trades.append({'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                       'days': 999, 'di': ND-1, 'reason': 'end',
                       'year': dates[ND-1].year, 'si': pos['si'], 'dir': pos['dir']})

    if not trades:
        return None

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

    final_cash = CASH0 + total_pnl_abs
    days_total = (dates[ND-1] - dates[MIN_TRAIN]).days
    yr = max(days_total / 365.25, 0.01)
    ann = ((final_cash / CASH0) ** (1 / yr) - 1) * 100 if final_cash > 0 else -100

    nw = sum(1 for t in trades if t['pnl_abs'] > 0)
    wr = nw / max(len(trades), 1) * 100

    for t in trades:
        y = t.get('year', 'unknown')
        if y not in year_stats:
            year_stats[y] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
        year_stats[y]['trades'] += 1
        if t['pnl_abs'] > 0:
            year_stats[y]['wins'] += 1
        year_stats[y]['total_pnl'] += t['pnl_pct']

    return {
        'ann': round(ann, 1), 'n': len(trades), 'wr': round(wr, 1),
        'max_dd': round(max_dd, 1), 'final': round(final_cash, 0),
        'year_stats': year_stats,
    }


if __name__ == '__main__':
    print("=" * 80, flush=True)
    print("  Alpha Futures V8c — 保证金 + OI因子 + 跨品种", flush=True)
    print("=" * 80, flush=True)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    # ---- 预计算因子 ----
    t0 = time.time()
    print("\n  预计算因子...", flush=True)

    # OI动量 (5日/10日/20日变化率)
    oi_mom5 = np.full((NS, ND), np.nan)
    oi_mom10 = np.full((NS, ND), np.nan)
    oi_mom20 = np.full((NS, ND), np.nan)
    # OI均值
    oi_ma20 = np.full((NS, ND), np.nan)
    # 价格×OI背离
    oi_price_div = np.full((NS, ND), np.nan)
    # VOL/OI比率
    vol_oi_ratio = np.full((NS, ND), np.nan)
    # 日内波幅
    intraday_range = np.full((NS, ND), np.nan)
    # K线实体比
    body_ratio = np.full((NS, ND), np.nan)
    # 动量
    mom5 = np.full((NS, ND), np.nan)
    mom10 = np.full((NS, ND), np.nan)
    mom20 = np.full((NS, ND), np.nan)
    # MA
    ma20 = np.full((NS, ND), np.nan)
    # ATR
    atr10 = np.full((NS, ND), np.nan)
    # 量比
    vol_ratio = np.full((NS, ND), np.nan)

    for si in range(NS):
        for di in range(20, ND):
            # OI动量
            oi_now = OI[si, di-1]
            if not np.isnan(oi_now) and oi_now > 0:
                oi5 = OI[si, max(0, di-6):di-1]
                oi5v = oi5[~np.isnan(oi5)]
                if len(oi5v) >= 3:
                    oi_mom5[si, di] = (oi_now - oi5v[0]) / oi5v[0]
                oi10 = OI[si, max(0, di-11):di-1]
                oi10v = oi10[~np.isnan(oi10)]
                if len(oi10v) >= 5:
                    oi_mom10[si, di] = (oi_now - oi10v[0]) / oi10v[0]
                oi20 = OI[si, max(0, di-21):di-1]
                oi20v = oi20[~np.isnan(oi20)]
                if len(oi20v) >= 10:
                    oi_mom20[si, di] = (oi_now - oi20v[0]) / oi20v[0]
                    oi_ma20[si, di] = np.mean(oi20v)

            # VOL/OI比率
            v_now = V[si, di-1]
            if not np.isnan(v_now) and not np.isnan(oi_now) and oi_now > 0:
                vol_oi_ratio[si, di] = v_now / oi_now

            # 价格×OI背离
            c_now = C[si, di-1]
            c5 = C[si, max(0, di-6):di-1]
            c5v = c5[~np.isnan(c5)]
            if not np.isnan(c_now) and len(c5v) >= 3:
                pmom = (c_now - c5v[0]) / c5v[0]
                omom = oi_mom5[si, di]
                if not np.isnan(omom):
                    # price↑ + OI↑ = +1 (新资金入场)
                    # price↑ + OI↓ = -1 (空头平仓)
                    # price↓ + OI↑ = +1 (新空头入场)
                    # price↓ + OI↓ = -1 (多头平仓)
                    if pmom > 0 and omom > 0:
                        oi_price_div[si, di] = 1
                    elif pmom > 0 and omom < 0:
                        oi_price_div[si, di] = -1
                    elif pmom < 0 and omom > 0:
                        oi_price_div[si, di] = 1
                    elif pmom < 0 and omom < 0:
                        oi_price_div[si, di] = -1

            # 日内波幅
            hi = H[si, di-1]
            lo = L[si, di-1]
            op = O[si, di-1]
            if not np.isnan(hi) and not np.isnan(lo) and not np.isnan(op) and op > 0:
                intraday_range[si, di] = (hi - lo) / op * 100
                cl = c_now if not np.isnan(c_now) else op
                if hi > lo:
                    body_ratio[si, di] = abs(cl - op) / (hi - lo)

            # 动量
            if not np.isnan(c_now) and c_now > 0:
                c5v2 = C[si, max(0, di-6):di-1]
                c5v2 = c5v2[~np.isnan(c5v2)]
                if len(c5v2) >= 3 and c5v2[0] > 0:
                    mom5[si, di] = (c_now - c5v2[0]) / c5v2[0]
                c10v = C[si, max(0, di-11):di-1]
                c10v = c10v[~np.isnan(c10v)]
                if len(c10v) >= 5 and c10v[0] > 0:
                    mom10[si, di] = (c_now - c10v[0]) / c10v[0]
                c20v = C[si, max(0, di-21):di-1]
                c20v = c20v[~np.isnan(c20v)]
                if len(c20v) >= 10 and c20v[0] > 0:
                    mom20[si, di] = (c_now - c20v[0]) / c20v[0]

            # MA20
            ma_vals = C[si, max(0, di-20):di]
            ma_valid = ma_vals[~np.isnan(ma_vals)]
            if len(ma_valid) >= 10:
                ma20[si, di] = np.mean(ma_valid)

            # ATR10
            if di >= 11:
                tr_sum = 0
                tr_cnt = 0
                for dd in range(max(1, di-10), di):
                    hi2 = H[si, dd]
                    lo2 = L[si, dd]
                    pc = C[si, dd-1]
                    if np.isnan(hi2) or np.isnan(lo2):
                        continue
                    tr = hi2 - lo2
                    if not np.isnan(pc):
                        tr = max(tr, abs(hi2 - pc), abs(lo2 - pc))
                    tr_sum += tr
                    tr_cnt += 1
                if tr_cnt >= 5:
                    atr10[si, di] = tr_sum / tr_cnt

            # 量比
            v_now2 = V[si, di-1]
            v20 = V[si, max(0, di-21):di-1]
            v20v = v20[~np.isnan(v20)]
            if not np.isnan(v_now2) and len(v20v) >= 10:
                vol_ratio[si, di] = v_now2 / np.mean(v20v)

    print(f"  因子计算完成 ({time.time()-t0:.0f}s)", flush=True)

    # ---- 品种→索引映射 ----
    sym_to_si = {}
    for si in range(NS):
        sym_to_si[syms[si]] = si

    # ---- 跨品种动量 ----
    # 对每个品种，计算同组品种的平均动量
    group_mom = np.full((NS, ND), np.nan)
    for gname, gsyms in GROUPS.items():
        gsis = [sym_to_si[s] for s in gsyms if s in sym_to_si]
        if len(gsis) < 2:
            continue
        for di in range(20, ND):
            gm = []
            for gsi in gsis:
                m = mom20[gsi, di]
                if not np.isnan(m):
                    gm.append(m)
            if len(gm) >= 2:
                for gsi in gsis:
                    group_mom[gsi, di] = np.mean(gm)

    results = []

    # ==== 策略1: Donchian突破 (基线) ====
    print("\n  [1] Donchian突破", flush=True)
    for chan_len in [10, 15, 20, 30]:
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
                    if len(vex) >= exit_len // 2:
                        if c < np.min(vex): sell_d[si].add(di)
                        if c > np.max(vex): cover_d[si].add(di)
            for max_pos in [1, 2, 3]:
                for sl in [0.10, 0.20, 0.50]:
                    for mp in [0.50, 0.80]:
                        r = backtest(buy_d, sell_d, short_d, cover_d,
                                    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set,
                                    max_positions=max_pos, sl_pct=sl,
                                    hold_max=60, margin_pct=mp)
                        if r and r['ann'] > 0:
                            r['desc'] = f"DONCH_c{chan_len}_e{exit_len}"
                            r['max_pos'] = max_pos
                            r['sl'] = sl
                            r['mp'] = mp
                            results.append(r)
        print(f"    chan={chan_len} done", flush=True)

    # ==== 策略2: OI增强突破 ====
    print("  [2] OI增强突破", flush=True)
    # 条件: Donchian突破 + OI增加 + 量比放大
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

                    # 多头: 价格突破 + OI增长 + 量比放大
                    if c > high_ch:
                        om = oi_mom5[si, di]
                        vr = vol_ratio[si, di]
                        if (not np.isnan(om) and om > 0 and
                            not np.isnan(vr) and vr > 1.0):
                            buy_d[si].add(di)

                    # 空头: 价格跌破 + OI增长 + 量比放大
                    if c < low_ch:
                        om = oi_mom5[si, di]
                        vr = vol_ratio[si, di]
                        if (not np.isnan(om) and om > 0 and
                            not np.isnan(vr) and vr > 1.0):
                            short_d[si].add(di)

                    ex = C[si, max(0, d-exit_len):d]
                    vex = ex[~np.isnan(ex)]
                    if len(vex) >= exit_len // 2:
                        if c < np.min(vex): sell_d[si].add(di)
                        if c > np.max(vex): cover_d[si].add(di)

            for max_pos in [1, 2, 3]:
                for sl in [0.10, 0.20, 0.50]:
                    for mp in [0.50, 0.80]:
                        r = backtest(buy_d, sell_d, short_d, cover_d,
                                    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set,
                                    max_positions=max_pos, sl_pct=sl,
                                    hold_max=60, margin_pct=mp)
                        if r and r['ann'] > 0:
                            r['desc'] = f"OI_DONCH_c{chan_len}_e{exit_len}"
                            r['max_pos'] = max_pos
                            r['sl'] = sl
                            r['mp'] = mp
                            results.append(r)
        print(f"    chan={chan_len} done", flush=True)

    # ==== 策略3: OI激增 + 趋势方向 ====
    print("  [3] OI激增趋势", flush=True)
    # 条件: OI > 2× 20日均值 + 价格在MA20上方(多)/下方(空)
    for oi_mult in [1.5, 2.0, 2.5]:
        for hold in [5, 10, 15, 20]:
            buy_d = {si: set() for si in range(NS)}
            sell_d = {si: set() for si in range(NS)}
            short_d = {si: set() for si in range(NS)}
            cover_d = {si: set() for si in range(NS)}
            for si in range(NS):
                for di in range(21, ND):
                    d = di - 1
                    c = C[si, d]
                    if np.isnan(c):
                        continue
                    oi_now = OI[si, d]
                    oi_ma = oi_ma20[si, di]
                    if np.isnan(oi_now) or np.isnan(oi_ma) or oi_ma <= 0:
                        continue
                    if oi_now <= oi_mult * oi_ma:
                        continue
                    m = ma20[si, di]
                    if np.isnan(m):
                        continue
                    # 放量确认
                    vr = vol_ratio[si, di]
                    if np.isnan(vr) or vr < 1.2:
                        continue
                    if c > m:
                        buy_d[si].add(di)
                    elif c < m:
                        short_d[si].add(di)
            for si in range(NS):
                for d in buy_d[si]:
                    sell_d[si].add(d + hold)
                for d in short_d[si]:
                    cover_d[si].add(d + hold)
            for max_pos in [1, 2, 3]:
                for sl in [0.10, 0.20]:
                    for mp in [0.50, 0.80]:
                        r = backtest(buy_d, sell_d, short_d, cover_d,
                                    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set,
                                    max_positions=max_pos, sl_pct=sl,
                                    hold_max=hold + 20, margin_pct=mp)
                        if r and r['ann'] > 0:
                            r['desc'] = f"OI_SURGE_{oi_mult}x_h{hold}"
                            r['max_pos'] = max_pos
                            r['sl'] = sl
                            r['mp'] = mp
                            results.append(r)
            print(f"    oi_mult={oi_mult} done", flush=True)

    # ==== 策略4: 跨品种共振 ====
    print("  [4] 跨品种共振", flush=True)
    # 条件: 同组品种平均动量 > 0 (多头) / < 0 (空头)
    # + 个品种突破20日高低
    for gm_thresh in [0.02, 0.05, 0.10]:
        for hold in [5, 10, 15, 20]:
            buy_d = {si: set() for si in range(NS)}
            sell_d = {si: set() for si in range(NS)}
            short_d = {si: set() for si in range(NS)}
            cover_d = {si: set() for si in range(NS)}
            for si in range(NS):
                for di in range(21, ND):
                    d = di - 1
                    c = C[si, d]
                    if np.isnan(c):
                        continue
                    gm = group_mom[si, di]
                    if np.isnan(gm):
                        continue
                    # 个品种也突破
                    ch20 = C[si, max(0, d-20):d]
                    vc20 = ch20[~np.isnan(ch20)]
                    if len(vc20) < 10:
                        continue
                    if gm > gm_thresh and c > np.max(vc20):
                        buy_d[si].add(di)
                    elif gm < -gm_thresh and c < np.min(vc20):
                        short_d[si].add(di)
            for si in range(NS):
                for d in buy_d[si]:
                    sell_d[si].add(d + hold)
                for d in short_d[si]:
                    cover_d[si].add(d + hold)
            for max_pos in [1, 2, 3]:
                for sl in [0.10, 0.20]:
                    for mp in [0.50, 0.80]:
                        r = backtest(buy_d, sell_d, short_d, cover_d,
                                    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set,
                                    max_positions=max_pos, sl_pct=sl,
                                    hold_max=hold + 20, margin_pct=mp)
                        if r and r['ann'] > 0:
                            r['desc'] = f"CROSS_gm{gm_thresh}_h{hold}"
                            r['max_pos'] = max_pos
                            r['sl'] = sl
                            r['mp'] = mp
                            results.append(r)
            print(f"    gm_thresh={gm_thresh} done", flush=True)

    # ==== 策略5: 价格+OI背离 ====
    print("  [5] 价格+OI背离", flush=True)
    # 价格↑ + OI↑ = 新多入场 → 做多
    # 价格↓ + OI↑ = 新空入场 → 做空
    # + 动量确认 + 量比确认
    for hold in [5, 10, 15, 20]:
        buy_d = {si: set() for si in range(NS)}
        sell_d = {si: set() for si in range(NS)}
        short_d = {si: set() for si in range(NS)}
        cover_d = {si: set() for si in range(NS)}
        for si in range(NS):
            for di in range(21, ND):
                d = di - 1
                div = oi_price_div[si, di]
                if np.isnan(div) or div <= 0:
                    continue
                c = C[si, d]
                if np.isnan(c):
                    continue
                m5 = mom5[si, di]
                vr = vol_ratio[si, di]
                if np.isnan(m5) or np.isnan(vr):
                    continue
                # 多头: 价格涨 + OI增 + 量放大
                if m5 > 0 and vr > 1.2:
                    buy_d[si].add(di)
                # 空头: 价格跌 + OI增 + 量放大
                elif m5 < 0 and vr > 1.2:
                    short_d[si].add(di)
        for si in range(NS):
            for d in buy_d[si]:
                sell_d[si].add(d + hold)
            for d in short_d[si]:
                cover_d[si].add(d + hold)
        for max_pos in [1, 2, 3]:
            for sl in [0.10, 0.20]:
                for mp in [0.50, 0.80]:
                    r = backtest(buy_d, sell_d, short_d, cover_d,
                                NS, ND, dates, C, O, H, L, V, OI, syms, sym_set,
                                max_positions=max_pos, sl_pct=sl,
                                hold_max=hold + 20, margin_pct=mp)
                    if r and r['ann'] > 0:
                        r['desc'] = f"DIV_PRICE_OI_h{hold}"
                        r['max_pos'] = max_pos
                        r['sl'] = sl
                        r['mp'] = mp
                        results.append(r)
        print(f"    hold={hold} done", flush=True)

    # ==== 策略6: 强趋势+大实体 ====
    print("  [6] 强趋势大实体", flush=True)
    # 条件: mom10 > 3% + body_ratio > 0.6 + vol_ratio > 1.5
    for mom_thresh in [0.02, 0.05, 0.08]:
        for hold in [5, 10, 15]:
            buy_d = {si: set() for si in range(NS)}
            sell_d = {si: set() for si in range(NS)}
            short_d = {si: set() for si in range(NS)}
            cover_d = {si: set() for si in range(NS)}
            for si in range(NS):
                for di in range(21, ND):
                    m = mom10[si, di]
                    if np.isnan(m):
                        continue
                    br = body_ratio[si, di]
                    vr = vol_ratio[si, di]
                    if np.isnan(br) or np.isnan(vr):
                        continue
                    if m > mom_thresh and br > 0.6 and vr > 1.5:
                        buy_d[si].add(di)
                    elif m < -mom_thresh and br > 0.6 and vr > 1.5:
                        short_d[si].add(di)
            for si in range(NS):
                for d in buy_d[si]:
                    sell_d[si].add(d + hold)
                for d in short_d[si]:
                    cover_d[si].add(d + hold)
            for max_pos in [1, 2, 3]:
                for sl in [0.10, 0.20]:
                    for mp in [0.50, 0.80]:
                        r = backtest(buy_d, sell_d, short_d, cover_d,
                                    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set,
                                    max_positions=max_pos, sl_pct=sl,
                                    hold_max=hold + 20, margin_pct=mp)
                        if r and r['ann'] > 0:
                            r['desc'] = f"TREND_BODY_m{mom_thresh}_h{hold}"
                            r['max_pos'] = max_pos
                            r['sl'] = sl
                            r['mp'] = mp
                            results.append(r)
            print(f"    mom_thresh={mom_thresh} done", flush=True)

    # ==== 策略7: 激进OI突破 — P1全仓 ====
    print("  [7] 激进OI突破 P1全仓", flush=True)
    for chan_len in [5, 10, 15]:
        for exit_len in [3, 5, 10]:
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
                    # 多头: 突破 + OI增 + 量增
                    if c > high_ch:
                        om = oi_mom5[si, di]
                        vr = vol_ratio[si, di]
                        if (not np.isnan(om) and om > 0 and
                            not np.isnan(vr) and vr > 1.2):
                            buy_d[si].add(di)
                    # 空头
                    if c < low_ch:
                        om = oi_mom5[si, di]
                        vr = vol_ratio[si, di]
                        if (not np.isnan(om) and om > 0 and
                            not np.isnan(vr) and vr > 1.2):
                            short_d[si].add(di)
                    ex = C[si, max(0, d-exit_len):d]
                    vex = ex[~np.isnan(ex)]
                    if len(vex) >= max(1, exit_len // 2):
                        if c < np.min(vex): sell_d[si].add(di)
                        if c > np.max(vex): cover_d[si].add(di)
            for hold_max_v in [15, 20, 30, 45]:
                for sl in [0.20, 0.50, 1.0]:
                    for mp in [0.80, 1.0]:
                        r = backtest(buy_d, sell_d, short_d, cover_d,
                                    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set,
                                    max_positions=1, sl_pct=sl,
                                    hold_max=hold_max_v, margin_pct=mp)
                        if r and r['ann'] > 0:
                            r['desc'] = f"AGGR_OI_c{chan_len}_e{exit_len}_hm{hold_max_v}"
                            r['max_pos'] = 1
                            r['sl'] = sl
                            r['mp'] = mp
                            results.append(r)
        print(f"    chan={chan_len} done", flush=True)

    # ==== 策略8: 动量轮动 — 每天换到最强品种 ====
    print("  [8] 动量轮动", flush=True)
    for lookback in [3, 5, 10]:
        for hold in [1, 3, 5]:
            buy_d = {si: set() for si in range(NS)}
            sell_d = {si: set() for si in range(NS)}
            short_d = {si: set() for si in range(NS)}
            cover_d = {si: set() for si in range(NS)}
            # 每天找动量最强的品种
            for di in range(lookback + 1, ND):
                d = di - 1
                best_long = -999
                best_short = 999
                best_long_si = -1
                best_short_si = -1
                for si in range(NS):
                    c = C[si, d]
                    if np.isnan(c) or c <= 0:
                        continue
                    cl = C[si, max(0, d-lookback):d]
                    clv = cl[~np.isnan(cl)]
                    if len(clv) < lookback // 2 or clv[0] <= 0:
                        continue
                    m = (c - clv[0]) / clv[0]
                    vr = vol_ratio[si, di]
                    om = oi_mom5[si, di]
                    # 只选有OI和量确认的
                    v_ok = not np.isnan(vr) and vr > 1.0
                    o_ok = not np.isnan(om) and om > 0
                    if v_ok and o_ok and m > best_long:
                        best_long = m
                        best_long_si = si
                    if v_ok and o_ok and m < best_short:
                        best_short = m
                        best_short_si = si
                if best_long_si >= 0:
                    buy_d[best_long_si].add(di)
                    sell_d[best_long_si].add(di + hold)
                if best_short_si >= 0:
                    short_d[best_short_si].add(di)
                    cover_d[best_short_si].add(di + hold)
            for sl in [0.20, 0.50]:
                for mp in [0.50, 0.80]:
                    r = backtest(buy_d, sell_d, short_d, cover_d,
                                NS, ND, dates, C, O, H, L, V, OI, syms, sym_set,
                                max_positions=2, sl_pct=sl,
                                hold_max=hold + 5, margin_pct=mp)
                    if r and r['ann'] > 0:
                        r['desc'] = f"MOM_ROT_lb{lookback}_h{hold}"
                        r['max_pos'] = 2
                        r['sl'] = sl
                        r['mp'] = mp
                        results.append(r)
            print(f"    lookback={lookback} done", flush=True)

    print(f"\n  全部完成 ({time.time()-t0:.0f}s, {len(results)} profitable)", flush=True)

    # 输出
    results.sort(key=lambda x: -x['ann'])
    print(f"\n{'='*80}", flush=True)
    print(f"  TOP 30", flush=True)
    print(f"  {'策略':<28s} {'P':>2s} {'SL':>4s} {'MP':>4s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'DD':>5s}", flush=True)
    for r in results[:30]:
        print(f"  {r['desc']:<28s} P{r['max_pos']} SL{r['sl']:.0%} MP{r['mp']:.0%} | "
              f"{r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% {r['max_dd']:5.1f}%", flush=True)

    # Top 5 year-by-year
    for i, r in enumerate(results[:5]):
        print(f"\n  #{i+1}: {r['desc']} P{r['max_pos']} SL{r['sl']:.0%} MP{r['mp']:.0%} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} t, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    if results:
        print(f"\n  Best: {results[0]['ann']:+.1f}% DD={results[0]['max_dd']:.1f}%", flush=True)
        # 按类型汇总
        type_stats = {}
        for r in results:
            t = r['desc'].split('_')[0]
            if t not in type_stats:
                type_stats[t] = {'best': r['ann'], 'avg_dd': 0, 'count': 0}
            type_stats[t]['best'] = max(type_stats[t]['best'], r['ann'])
            type_stats[t]['avg_dd'] += r['max_dd']
            type_stats[t]['count'] += 1
        print(f"\n  策略类型汇总:", flush=True)
        for t, s in sorted(type_stats.items(), key=lambda x: -x[1]['best']):
            print(f"    {t:<15s}: best={s['best']:+.1f}%, avg_dd={s['avg_dd']/s['count']:.1f}%, n={s['count']}", flush=True)
    print(f"{'='*80}", flush=True)
