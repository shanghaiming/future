#!/usr/bin/env python3
"""
策略 V66 — 全新信号体系: Carry动量 + OI + 成交量 + 多日形态

之前所有尝试都基于MR (RSI<25), edge太薄。V66尝试完全不同的信号:

  1. Carry动量: backwardated品种的短期动量 (carry提供结构性顺风)
  2. OI变化信号: OI增加 + 价格上涨 = 多头增仓 (趋势确认)
  3. 成交量突破: 放量突破近期高点的carry品种
  4. 多日连涨/跌: 连跌3天后反转的carry品种 (类似MR但用不同指标)
  5. 通道突破: 20日新高的backwardated品种 (趋势跟踪)
  6. 组合信号: 多个信号同时出现的品种

为什么可能更好:
  - MR: 57% WR, 0.8% avg — 边缘太薄
  - Carry动量: 学术文献显示carry+momentum组合可产生15-20%年化
  - OI增加+涨: 显示资金流入, 更强的趋势确认
  - 通道突破在carry品种上: 结构性顺风 + 趋势 = 更大的移动
"""

import os, sys, time, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from contract_specs import get_spec

COMM = 0.00015
INIT = 500000
TD = 252


def load_carry(ts_dir):
    carry = {}
    for f in os.listdir(ts_dir):
        if not f.endswith('.json'):
            continue
        with open(os.path.join(ts_dir, f)) as fh:
            d = json.load(fh)
        sym = d['symbol'].lower()
        near = d['near_price']
        far = d['far_price']
        if near > 0 and far > 0:
            c = (near - far) / near * 100
            carry[sym] = c
    return carry


def load(data_dir):
    raw = {}
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith('.csv'):
            continue
        sym = f.replace('.csv', '')
        df = pd.read_csv(os.path.join(data_dir, f))
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        df = df.sort_values('trade_date').reset_index(drop=True)
        if len(df) < 300:
            continue
        df['ret'] = df['close'].pct_change()

        # 动量指标
        for lag in [1, 2, 3, 5, 10, 20, 60]:
            df[f'm{lag}'] = df['close'].pct_change(lag)

        # 波动率
        for w in [5, 10, 20, 40, 60]:
            df[f'hv{w}'] = df['ret'].rolling(w).std() * np.sqrt(TD)
        df['hv_pct'] = df['hv20'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 10 else .5, raw=False)

        # 均线
        for w in [5, 10, 20, 40, 60]:
            df[f'ma{w}'] = df['close'].rolling(w).mean()

        # RSI
        d = df['close'].diff()
        g = d.where(d > 0, 0).rolling(14).mean()
        l = (-d.where(d < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + g / l))

        # 连续涨跌
        down = (df['ret'] < 0).astype(int).values
        up = (df['ret'] > 0).astype(int).values
        cons_d = []
        cons_u = []
        cd, cu = 0, 0
        for v_d, v_u in zip(down, up):
            cd = cd + 1 if v_d else 0
            cu = cu + 1 if v_u else 0
            cons_d.append(cd)
            cons_u.append(cu)
        df['cons_down'] = cons_d
        df['cons_up'] = cons_u

        # 通道
        ch_hi20 = df['close'].rolling(20).max()
        ch_lo20 = df['close'].rolling(20).min()
        df['ch_hi20'] = ch_hi20
        df['ch_lo20'] = ch_lo20
        df['at_new_high20'] = (df['close'] >= ch_hi20).astype(int)
        df['at_new_low20'] = (df['close'] <= ch_lo20).astype(int)

        # 成交量指标
        df['vol_ma5'] = df['vol'].rolling(5).mean()
        df['vol_ma20'] = df['vol'].rolling(20).mean()
        df['vol_ratio'] = df['vol'] / df['vol_ma20'].replace(0, np.nan)

        # OI指标
        df['oi_chg1'] = df['oi'].diff(1)
        df['oi_chg5'] = df['oi'].diff(5)
        df['oi_pct1'] = df['oi'].pct_change(1)
        df['oi_pct5'] = df['oi'].pct_change(5)
        df['oi_ma10'] = df['oi'].rolling(10).mean()
        df['oi_ratio'] = df['oi'] / df['oi_ma10'].replace(0, np.nan)

        # 价格位置
        df['pos_in_range20'] = (df['close'] - ch_lo20) / (ch_hi20 - ch_lo20).replace(0, np.nan)

        # 前向收益
        for hold in [3, 5, 10, 15, 20]:
            df[f'fwd{hold}'] = df['close'].shift(-hold) / df['close'] - 1

        df = df.dropna(subset=['ma20', 'hv20', 'rsi'])
        if len(df) < 100:
            continue
        try:
            spec = get_spec(sym)
        except:
            continue

        raw[sym] = {
            'spec': spec,
            'dates': df['trade_date'].values,
            'open': df['open'].values.astype(np.float64),
            'close': df['close'].values.astype(np.float64),
            'high': df['high'].values.astype(np.float64),
            'low': df['low'].values.astype(np.float64),
            'vol': df['vol'].values.astype(np.float64),
            'oi': df['oi'].values.astype(np.float64),
            'ret': df['ret'].values.astype(np.float64),
            'rsi': df['rsi'].values.astype(np.float64),
            'cons_down': np.array(cons_d, dtype=np.int32),
            'cons_up': np.array(cons_u, dtype=np.int32),
            'hv20': df['hv20'].values.astype(np.float64),
            'hv_pct': df['hv_pct'].values.astype(np.float64),
            'ma5': df['ma5'].values.astype(np.float64),
            'ma10': df['ma10'].values.astype(np.float64),
            'ma20': df['ma20'].values.astype(np.float64),
            'ma60': df['ma60'].values.astype(np.float64),
            'm1': df['m1'].values.astype(np.float64),
            'm3': df['m3'].values.astype(np.float64),
            'm5': df['m5'].values.astype(np.float64),
            'm10': df['m10'].values.astype(np.float64),
            'm20': df['m20'].values.astype(np.float64),
            'ch_hi20': df['ch_hi20'].values.astype(np.float64),
            'ch_lo20': df['ch_lo20'].values.astype(np.float64),
            'at_new_high20': df['at_new_high20'].values.astype(np.int32),
            'vol_ratio': df['vol_ratio'].values.astype(np.float64),
            'oi_pct1': df['oi_pct1'].values.astype(np.float64),
            'oi_pct5': df['oi_pct5'].values.astype(np.float64),
            'oi_ratio': df['oi_ratio'].values.astype(np.float64),
            'pos_in_range20': df['pos_in_range20'].values.astype(np.float64),
            'fwd5': df['fwd5'].values.astype(np.float64),
            'fwd10': df['fwd10'].values.astype(np.float64),
            'fwd20': df['fwd20'].values.astype(np.float64),
        }
    return raw


def build_idx(raw, s, e):
    ad = set()
    for d in raw.values():
        m = (d['dates'] >= s) & (d['dates'] <= e)
        for dt in d['dates'][m]:
            ad.add(dt)
    dates = np.array(sorted(ad))
    si = {}
    for sym, d in raw.items():
        im = {}
        m = (d['dates'] >= s) & (d['dates'] <= e)
        for dt, il in zip(d['dates'][m], np.where(m)[0]):
            im[dt] = int(il)
        si[sym] = im
    return dates, si


def analyze_signals(raw, carry, dates, si):
    """分析各种信号在测试期的表现"""
    results = {}
    signals = {
        'mr_rsi25': lambda d, i: d['rsi'][i] < 25,
        'mr_rsi30': lambda d, i: d['rsi'][i] < 30,
        'carry_mom5': lambda d, i: d['m5'][i] > 0,
        'carry_mom10': lambda d, i: d['m10'][i] > 0,
        'new_high20': lambda d, i: d['at_new_high20'][i] == 1,
        'oi_up': lambda d, i: d['oi_pct1'][i] > 0.02 and d['ret'][i] > 0,
        'vol_break': lambda d, i: d['vol_ratio'][i] > 2.0 and d['ret'][i] > 0,
        'cons_down3': lambda d, i: d['cons_down'][i] >= 3,
        'cons_up3': lambda d, i: d['cons_up'][i] >= 3,
        'above_ma5': lambda d, i: d['close'][i] > d['ma5'][i],
        'above_ma20': lambda d, i: d['close'][i] > d['ma20'][i],
        'ma5_gt_ma20': lambda d, i: d['ma5'][i] > d['ma20'][i],
        'oi_surge': lambda d, i: d['oi_ratio'][i] > 1.2,
        'low_vol': lambda d, i: d['hv_pct'][i] < 0.3,
    }

    for hold in [5, 10, 20]:
        fwd_key = f'fwd{hold}'
        for sig_name, sig_fn in signals.items():
            # 在backwardated品种中测试
            for carry_type, carry_syms in [('all', list(raw.keys())),
                                            ('backwd', [s for s in raw if carry.get(s, 0) > 0]),
                                            ('carry3', [s for s in raw if carry.get(s, 0) > 3])]:
                n_signals = 0
                n_valid = 0
                fwd_vals = []

                for sym in carry_syms:
                    d = raw[sym]
                    for i in range(60, len(d['dates']) - hold):
                        if sig_fn(d, i):
                            fwd = d[fwd_key][i]
                            if not np.isnan(fwd):
                                fwd_vals.append(fwd)
                                n_valid += 1
                            n_signals += 1

                if n_valid < 30:
                    continue

                fwd_arr = np.array(fwd_vals)
                wr = (fwd_arr > 0).mean()
                avg = fwd_arr.mean()
                key = f"{sig_name}_{carry_type}_h{hold}"
                results[key] = {
                    'signal': sig_name, 'carry': carry_type, 'hold': hold,
                    'wr': wr, 'avg': avg, 'n': n_valid,
                    'score': wr * max(avg, 0) * 100,
                }

    # 排序输出
    sorted_results = sorted(results.values(), key=lambda x: -x['score'])
    print(f"\n{'Signal':>15} {'Carry':>8} {'Hold':>5} {'WR':>6} {'Avg':>8} {'N':>6} {'Score':>8}")
    print("-" * 70)
    for r in sorted_results[:60]:
        print(f"{r['signal']:>15} {r['carry']:>8} {r['hold']:>5} "
              f"{r['wr']:>6.1%} {r['avg']:>8.3%} {r['n']:>6} {r['score']:>8.3f}")

    # 组合信号分析
    print("\n=== 组合信号 (2+信号同时触发) ===")
    for hold in [5, 10]:
        fwd_key = f'fwd{hold}'
        combo_results = []

        combos = [
            ('rsi25+oi_up', lambda d, i: d['rsi'][i] < 25 and d['oi_pct1'][i] > 0.02),
            ('rsi25+vol_break', lambda d, i: d['rsi'][i] < 25 and d['vol_ratio'][i] > 1.5),
            ('cons_down3+oi_up', lambda d, i: d['cons_down'][i] >= 3 and d['oi_pct1'][i] > 0.02),
            ('cons_down3+vol_break', lambda d, i: d['cons_down'][i] >= 3 and d['vol_ratio'][i] > 1.5),
            ('rsi25+above_ma20', lambda d, i: d['rsi'][i] < 25 and d['close'][i] > d['ma20'][i]),
            ('new_high+oi_surge', lambda d, i: d['at_new_high20'][i] == 1 and d['oi_ratio'][i] > 1.1),
            ('carry_mom5+oi_up', lambda d, i: d['m5'][i] > 0.02 and d['oi_pct1'][i] > 0.02),
            ('rsi25+low_vol', lambda d, i: d['rsi'][i] < 25 and d['hv_pct'][i] < 0.3),
            ('cons_down3+low_vol', lambda d, i: d['cons_down'][i] >= 3 and d['hv_pct'][i] < 0.3),
            ('above_ma5+oi_up+vol', lambda d, i: d['close'][i] > d['ma5'][i] and d['oi_pct1'][i] > 0.02 and d['vol_ratio'][i] > 1.5),
            ('ma5_gt_ma20+mom5', lambda d, i: d['ma5'][i] > d['ma20'][i] and d['m5'][i] > 0.02),
            ('rsi30+carry_mom5', lambda d, i: d['rsi'][i] < 30 and d['m5'][i] > 0),
            ('rsi25+cons_down3+carry3_only', lambda d, i: d['rsi'][i] < 30 and d['cons_down'][i] >= 2),
            ('new_high+carry_mom5', lambda d, i: d['at_new_high20'][i] == 1 and d['m5'][i] > 0.01),
        ]

        for combo_name, combo_fn in combos:
            for carry_type, carry_syms in [('backwd', [s for s in raw if carry.get(s, 0) > 0]),
                                            ('carry3', [s for s in raw if carry.get(s, 0) > 3]),
                                            ('all', list(raw.keys()))]:
                fwd_vals = []
                for sym in carry_syms:
                    d = raw[sym]
                    for i in range(60, len(d['dates']) - hold):
                        if combo_fn(d, i):
                            fwd = d[fwd_key][i]
                            if not np.isnan(fwd):
                                fwd_vals.append(fwd)

                if len(fwd_vals) < 15:
                    continue
                fwd_arr = np.array(fwd_vals)
                wr = (fwd_arr > 0).mean()
                avg = fwd_arr.mean()
                combo_results.append({
                    'combo': combo_name, 'carry': carry_type, 'hold': hold,
                    'wr': wr, 'avg': avg, 'n': len(fwd_vals),
                    'score': wr * max(avg, 0) * 100,
                })

        combo_results.sort(key=lambda x: -x['score'])
        print(f"\nHold={hold}天:")
        print(f"  {'Combo':>30} {'Carry':>8} {'WR':>6} {'Avg':>8} {'N':>6} {'Score':>8}")
        print("  " + "-" * 75)
        for r in combo_results[:40]:
            print(f"  {r['combo']:>30} {r['carry']:>8} {r['wr']:>6.1%} {r['avg']:>8.3%} "
                  f"{r['n']:>6} {r['score']:>8.3f}")

    return results


def bt_signal(raw, dates, si, good_syms, carry_map, p):
    """通用信号回测"""
    mp = p.get('max_pos', 3)
    hd = p.get('hold_days', 10)
    nm = p.get('notional_mult', 10)
    signal = p.get('signal', 'mr_rsi25')

    eq = float(INIT)
    hwm = float(INIT)
    pos = {}
    pnls = []
    eqh = [float(INIT)]

    for date in dates:
        # 退出
        for sym in list(pos):
            ps = pos[sym]
            im = si.get(sym)
            if not im or date not in im:
                continue
            il = im[date]
            S = raw[sym]['close'][il]
            h = int((date - ps['ed']) / np.timedelta64(1, 'D'))
            if h < hd:
                continue
            ml = ps['ml']
            trade_pnl = (S - ps['ep']) * ml * ps['ct']
            notional_exit = S * ml * ps['ct']
            comm = COMM * (ps['notional'] + notional_exit)
            pnl = trade_pnl - comm
            eq += pnl
            pnls.append(pnl)
            if eq > hwm:
                hwm = eq
            del pos[sym]

        # 入场
        if len(pos) < mp:
            sigs = []
            for sym, d in raw.items():
                if sym in pos or sym not in good_syms:
                    continue
                im = si.get(sym)
                if not im or date not in im:
                    continue
                il = im[date]
                if il < 60:
                    continue
                pi = il - 1  # 前一天信号

                # 信号判定
                triggered = False
                score = 0
                carry_val = carry_map.get(sym, 0)

                if signal == 'mr_rsi25':
                    if d['rsi'][pi] < 25:
                        triggered = True
                        score = (25 - d['rsi'][pi]) / 25

                elif signal == 'mr_rsi30':
                    if d['rsi'][pi] < 30:
                        triggered = True
                        score = (30 - d['rsi'][pi]) / 30

                elif signal == 'carry_mom5':
                    if d['m5'][pi] > 0.01:
                        triggered = True
                        score = d['m5'][pi] * 10

                elif signal == 'carry_mom10':
                    if d['m10'][pi] > 0.02:
                        triggered = True
                        score = d['m10'][pi] * 5

                elif signal == 'new_high20':
                    if d['at_new_high20'][pi] == 1:
                        triggered = True
                        score = carry_val / 25

                elif signal == 'oi_up_price':
                    if d['oi_pct1'][pi] > 0.02 and d['ret'][pi] > 0:
                        triggered = True
                        score = d['oi_pct1'][pi] * 10

                elif signal == 'vol_break':
                    if d['vol_ratio'][pi] > 2.0 and d['ret'][pi] > 0:
                        triggered = True
                        score = d['vol_ratio'][pi] / 5

                elif signal == 'cons_down3':
                    if d['cons_down'][pi] >= 3:
                        triggered = True
                        score = d['cons_down'][pi] / 5

                elif signal == 'mr_oi':
                    if d['rsi'][pi] < 30 and d['oi_pct1'][pi] > 0.01:
                        triggered = True
                        score = (30 - d['rsi'][pi]) / 30 + d['oi_pct1'][pi]

                elif signal == 'mr_vol':
                    if d['rsi'][pi] < 30 and d['vol_ratio'][pi] > 1.5:
                        triggered = True
                        score = (30 - d['rsi'][pi]) / 30 + d['vol_ratio'][pi] / 5

                elif signal == 'carry_trend':
                    # carry品种 + 价格在MA5上方 + MA5>MA20
                    if d['close'][pi] > d['ma5'][pi] and d['ma5'][pi] > d['ma20'][pi]:
                        triggered = True
                        score = carry_val / 25 + d['m5'][pi] * 5

                elif signal == 'mr_carry_strict':
                    if d['rsi'][pi] < 25 and carry_val > 3:
                        triggered = True
                        score = (25 - d['rsi'][pi]) / 25 + carry_val / 25

                elif signal == 'combo1':
                    # 连跌3天 + OI增加
                    if d['cons_down'][pi] >= 3 and d['oi_pct1'][pi] > 0.01:
                        triggered = True
                        score = d['cons_down'][pi] / 5 + d['oi_pct1'][pi] * 5

                elif signal == 'combo2':
                    # RSI<30 + 低波动率
                    if d['rsi'][pi] < 30 and d['hv_pct'][pi] < 0.4:
                        triggered = True
                        score = (30 - d['rsi'][pi]) / 30 + (0.4 - d['hv_pct'][pi])

                elif signal == 'combo3':
                    # 新高 + OI增加 + carry
                    if d['at_new_high20'][pi] == 1 and d['oi_ratio'][pi] > 1.1:
                        triggered = True
                        score = carry_val / 25 + d['oi_ratio'][pi]

                elif signal == 'combo4':
                    # MA5>MA20 + 5日动量>2% + carry
                    if d['ma5'][pi] > d['ma20'][pi] and d['m5'][pi] > 0.02 and carry_val > 0:
                        triggered = True
                        score = d['m5'][pi] * 10 + carry_val / 25

                elif signal == 'combo5':
                    # RSI<30 + 成交量放大 + OI增加
                    rsi_ok = d['rsi'][pi] < 30
                    vol_ok = d['vol_ratio'][pi] > 1.5
                    oi_ok = d['oi_pct1'][pi] > 0.01
                    if rsi_ok and vol_ok and oi_ok:
                        triggered = True
                        score = (30 - d['rsi'][pi]) / 30 + d['vol_ratio'][pi] / 5 + d['oi_pct1'][pi] * 5

                if not triggered:
                    continue
                if np.isnan(score):
                    continue

                # carry加分
                score += max(carry_val, 0) / 50

                sigs.append((sym, score, il))

            sigs.sort(key=lambda x: -x[1])

            for sym, score, il in sigs:
                if len(pos) >= mp:
                    break
                d = raw[sym]
                entry_price = d['open'][il]
                if np.isnan(entry_price) or entry_price <= 0:
                    continue
                ml, mr, _, _ = d['spec']
                notional_per = eq * nm / mp
                contracts = int(notional_per / (entry_price * ml))
                contracts = max(contracts, 1)
                notional = entry_price * ml * contracts
                margin = notional * mr
                if margin > eq * 0.9:
                    contracts = max(int(eq * 0.9 / (entry_price * ml * mr)), 1)
                    notional = entry_price * ml * contracts
                pos[sym] = {
                    'dir': 1, 'ed': date, 'ep': entry_price,
                    'ct': contracts, 'ml': ml, 'notional': notional,
                }

        # 权益
        ur = 0.
        for sym, ps in pos.items():
            im = si.get(sym)
            if im and date in im:
                S = raw[sym]['close'][im[date]]
                ur += (S - ps['ep']) * ps['ml'] * ps['ct']
        ceq = eq + ur
        eqh.append(ceq)
        if ceq < 1000:
            break

    if not pnls or eq <= 0:
        return None
    tr = (eq - INIT) / INIT
    if tr <= -1:
        return None
    dys = int((dates[-1] - dates[0]) / np.timedelta64(1, 'D'))
    yrs = max(dys / 365, .001)
    ann = float((1 + tr) ** (1 / yrs) - 1)
    pa = np.array(pnls)
    wr = float((pa > 0).mean())
    aw = float(pa[pa > 0].mean()) if (pa > 0).any() else 0
    al = float(abs(pa[pa <= 0].mean())) if (pa <= 0).any() else 1
    pf = aw * (pa > 0).sum() / (al * (pa <= 0).sum()) if (pa <= 0).sum() > 0 and al > 0 else 0
    ea = np.array(eqh[1:])
    if len(ea) > 1:
        cm = np.maximum.accumulate(ea)
        dd = (ea - cm) / cm
        mdd = float(dd.min())
        rets = np.diff(ea) / ea[:-1]
        sh = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    else:
        mdd = 0; sh = 0

    return {'annual': ann, 'wr': wr, 'mdd': mdd, 'pf': pf, 'trades': len(pa),
            'final': eq, 'sharpe': sh, **p}


def main():
    dd = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    ts_dir = os.path.expanduser("~/home/futures_platform/data/futures_term_structure")

    print("加载carry数据...")
    carry = load_carry(ts_dir)
    backwd = {s: c for s, c in carry.items() if c > 0}

    print("加载行情数据..."); t0 = time.time()
    raw = load(dd)
    print(f"  {len(raw)}品种, {time.time()-t0:.1f}s")

    sym_backwd = [s for s in raw if carry.get(s, 0) > 0]
    sym_carry3 = [s for s in raw if carry.get(s, 0) > 3]
    sym_all = list(raw.keys())

    print(f"Backwd: {len(sym_backwd)}, carry>3: {len(sym_carry3)}")

    test_s = pd.Timestamp('2022-01-01')
    test_e = pd.Timestamp('2026-05-08')
    dates, si = build_idx(raw, test_s, test_e)
    print(f"测试期: {len(dates)}交易日")

    # === 先做信号分析 ===
    print("\n" + "=" * 80)
    print("=== 信号分析 ===")
    analyze_signals(raw, carry, dates, si)

    # === 回测 ===
    print("\n" + "=" * 80)
    print("=== 回测 ===")

    res = []
    signals = ['mr_rsi25', 'mr_rsi30', 'carry_mom5', 'carry_mom10',
               'new_high20', 'oi_up_price', 'vol_break', 'cons_down3',
               'mr_oi', 'mr_vol', 'carry_trend', 'mr_carry_strict',
               'combo1', 'combo2', 'combo3', 'combo4', 'combo5']

    sym_sets = {
        'all': sym_all,
        'backwd': sym_backwd,
        'c3': sym_carry3,
    }

    for signal in signals:
        for sym_name, sym_set in sym_sets.items():
            for hd in [5, 10, 15]:
                for nm in [5, 10, 20]:
                    p = dict(signal=signal, hold_days=hd, notional_mult=nm,
                             max_pos=3, sym_set=sym_name)
                    r = bt_signal(raw, dates, si, sym_set, carry, p)
                    if r:
                        res.append(r)

    print(f"\n总结果: {len(res)}组有效")
    res.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n{'Signal':>18} {'Sym':>8} {'H':>3} {'NM':>4} "
          f"{'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5}")
    print("-" * 100)
    for r in res[:60]:
        print(f"{r['signal']:>18} {r['sym_set']:>8} {r['hold_days']:>3} {r['notional_mult']:>4.0f} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5}")

    # 目标
    print("\n" + "=" * 100)
    for ta, tw, lb in [(6., .50, "年化>=600%"),
                       (3., .50, "年化>=300%"),
                       (1., .50, "年化>=100%"),
                       (0.5, .50, "年化>=50%")]:
        g = [r for r in res if r['annual'] >= ta and r['wr'] >= tw]
        print(f"\n=== {lb}: {len(g)}组 ===")
        if g:
            for r in sorted(g, key=lambda x: x['annual'], reverse=True)[:10]:
                print(f"    年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  MDD={r['mdd']:>7.1%}  "
                      f"Sharpe={r['sharpe']:>5.2f}  "
                      f"{r['signal']} {r['sym_set']} H={r['hold_days']} NM={r['notional_mult']:.0f}  "
                      f"Trades={r['trades']}")

    od = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(od, exist_ok=True)
    sv = [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
           for k, v in r.items()} for r in res[:2000]]
    with open(os.path.join(od, 'backtest_v66.json'), 'w') as f:
        json.dump(sv, f, indent=2, default=str)
    print(f"\n→ backtest_results/backtest_v66.json")


if __name__ == '__main__':
    main()
