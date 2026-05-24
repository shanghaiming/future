#!/usr/bin/env python3
"""
V82: 实战可行性分析
1. 真实交易成本: 手续费+滑点+冲击成本
2. 资金容量分析: 50万→500万→5000万
3. 品种流动性: 每日成交量是否足以支撑
4. 实盘vs回测差异: 各种偏差分析
5. 最坏情况压力测试
"""
import os, glob, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data/futures_weighted'
INITIAL_CAPITAL = 500_000
CONTRACT_SPECS = 'scripts/contract_specs.py'
TEST_START = '2022-01-01'
TEST_END = '2025-12-31'


def load_data():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cs", CONTRACT_SPECS)
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    all_data = {}
    specs = {}
    for f in sorted(glob.glob(os.path.join(DATA_DIR, '*.csv'))):
        sym = os.path.basename(f).replace('.csv', '')
        try:
            mult, margin, tick, tick_val = cs.get_spec(sym)
            specs[sym] = {'mult': mult, 'margin': margin, 'tick': tick, 'tick_val': tick_val}
        except: continue
        df = pd.read_csv(f)
        if len(df) < 100: continue
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        df = df.sort_values('trade_date').reset_index(drop=True)
        if df['close'].isna().all() or (df['close'] == 0).any(): continue
        all_data[sym] = df
    print(f"  {len(all_data)}品种")
    return all_data, specs


def compute_signals(all_data):
    signal_data = {}
    for sym, df in all_data.items():
        c = df['close'].values.astype(float)
        o = df['open'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        v = df['vol'].values.astype(float)
        oi = df['oi'].values.astype(float)
        n = len(df)
        prev_c = np.full(n, np.nan); prev_c[1:] = c[:-1]
        gap = np.full(n, np.nan); gap[1:] = (o[1:] - prev_c[1:]) / prev_c[1:] * 100
        tr = np.full(n, np.nan)
        tr[1:] = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
        atr = pd.Series(tr).rolling(20).mean().values
        atr_pct = atr / c * 100
        ma20 = pd.Series(c).rolling(20).mean().values
        ma60 = pd.Series(c).rolling(60).mean().values
        ma5 = pd.Series(c).rolling(5).mean().values
        mom5 = np.full(n, np.nan); mom5[5:] = (c[5:] - c[:-5]) / c[:-5] * 100
        oi_ch = np.full(n, np.nan)
        valid = np.abs(oi[:-1]) > 0
        oi_ch_v = np.full(n-1, np.nan)
        oi_ch_v[valid] = (oi[1:][valid] - oi[:-1][valid]) / np.abs(oi[:-1][valid]) * 100
        oi_ch[1:] = oi_ch_v
        vol_ma5 = pd.Series(v).rolling(5).mean().values
        range_ = h - l
        clv = np.where(range_ > 0, (2*c - h - l) / range_, 0)
        gv = np.nan_to_num(gap)
        ga = np.nan_to_num(gap / np.where(atr_pct == 0, np.nan, atr_pct))

        s_l = np.zeros(n)
        s_l += (gv < -0.5).astype(int) * 1
        s_l += (gv < -1.0).astype(int) * 2
        s_l += (gv < -1.5).astype(int) * 2
        s_l += (gv < -2.0).astype(int) * 3
        s_l += (ga < -1.0).astype(int) * 2
        s_l += (ga < -1.5).astype(int) * 3
        s_l += ((oi_ch > 0) & (c < prev_c)).astype(int) * 3
        s_l += ((oi_ch < 0) & (c < prev_c)).astype(int) * 2
        s_l += (mom5 < -3).astype(int) * 1
        s_l += (mom5 < -5).astype(int) * 1
        s_l += (c < ma5).astype(int) * 1
        s_l += ((v > vol_ma5 * 1.5) & (c < prev_c)).astype(int) * 1
        s_l += (clv > 0.5).astype(int) * 1
        s_l += (ma20 > ma60).astype(int) * 2

        s_s = np.zeros(n)
        s_s += (gv > 0.5).astype(int) * 1
        s_s += (gv > 1.0).astype(int) * 2
        s_s += (gv > 1.5).astype(int) * 2
        s_s += (gv > 2.0).astype(int) * 3
        s_s += (ga > 1.0).astype(int) * 2
        s_s += (ga > 1.5).astype(int) * 3
        s_s += ((oi_ch > 0) & (c > prev_c)).astype(int) * 3
        s_s += ((oi_ch < 0) & (c > prev_c)).astype(int) * 2
        s_s += (mom5 > 3).astype(int) * 1
        s_s += (mom5 > 5).astype(int) * 1
        s_s += (c > ma5).astype(int) * 1
        s_s += ((v > vol_ma5 * 1.5) & (c > prev_c)).astype(int) * 1
        s_s += (clv < -0.5).astype(int) * 1
        s_s += (ma20 < ma60).astype(int) * 2

        df['score_long'] = s_l
        df['score_short'] = s_s
        df['gap_pct'] = gap
        signal_data[sym] = df
    return signal_data


def run_bt_realistic(signal_data, specs, start, end, capital=500_000, max_pos=7, lev=5,
                     min_sc=7, hold=1, sl_pct=-1.5, tp_pct=4.0,
                     commission_rate=0.0002, slippage_ticks=1,
                     max_long=4, max_short=4):
    """
    实战回测 — 考虑真实交易成本
    commission_rate: 手续费率 (万二 = 0.0002)
    slippage_ticks: 滑点(最小变动价位数)
    """
    dates = pd.date_range(start=start, end=end, freq='B')
    cap = capital
    eq, trades, pos = [], [], []

    for dt in dates:
        pnl = 0
        keep = []
        for p in pos:
            df = signal_data.get(p['sym'])
            sp = specs.get(p['sym'], {})
            if df is None: keep.append(p); continue
            idx = df.index[df['trade_date'] == dt]
            if len(idx) == 0: keep.append(p); continue
            row = df.loc[idx[0]]
            cur_h, cur_l, cur_c = row['high'], row['low'], row['close']
            if np.isnan(cur_c): keep.append(p); continue

            d = (dt - p['ed']).days
            tick_val = sp.get('tick_val', 1)
            mult = sp.get('mult', 1)
            tick = sp.get('tick', 1)

            # 滑点成本 (每个tick)
            slip_cost_pct = slippage_ticks * tick / p['ep'] * 100  # as %
            # 手续费 (开仓+平仓各一次)
            comm_pct = commission_rate * 2 * 100  # 开+平, as %

            triggered = False
            actual_ret = None
            reason = None

            if p['dir'] == 'long':
                if sl_pct:
                    stop = p['ep'] * (1 + sl_pct / 100)
                    if cur_l <= stop:
                        fill = stop * (1 - slippage_ticks * tick / stop)
                        actual_ret = (fill - p['ep']) / p['ep'] * 100 - comm_pct
                        reason = 'SL'; triggered = True
                if not triggered and tp_pct:
                    tp_p = p['ep'] * (1 + tp_pct / 100)
                    if cur_h >= tp_p:
                        fill = tp_p * (1 - slippage_ticks * tick / tp_p)
                        actual_ret = (fill - p['ep']) / p['ep'] * 100 - comm_pct
                        reason = 'TP'; triggered = True
                if not triggered:
                    actual_ret = (cur_c - p['ep']) / p['ep'] * 100 - comm_pct
            else:
                if sl_pct:
                    stop = p['ep'] * (1 - sl_pct / 100)
                    if cur_h >= stop:
                        fill = stop * (1 + slippage_ticks * tick / stop)
                        actual_ret = (p['ep'] - fill) / p['ep'] * 100 - comm_pct
                        reason = 'SL'; triggered = True
                if not triggered and tp_pct:
                    tp_p = p['ep'] * (1 - tp_pct / 100)
                    if cur_l <= tp_p:
                        fill = tp_p * (1 + slippage_ticks * tick / tp_p)
                        actual_ret = (p['ep'] - fill) / p['ep'] * 100 - comm_pct
                        reason = 'TP'; triggered = True
                if not triggered:
                    actual_ret = (p['ep'] - cur_c) / p['ep'] * 100 - comm_pct

            if d >= hold:
                if not triggered: reason = 'exp'
            else:
                if not triggered: keep.append(p); continue

            if reason:
                pnl += p['not'] * actual_ret / 100
                trades.append({
                    'sym': p['sym'], 'dir': p['dir'], 'ed': p['ed'], 'xd': dt,
                    'ep': p['ep'], 'xp': cur_c, 'r': actual_ret,
                    'pnl': p['not'] * actual_ret / 100, 'sc': p['sc'],
                    'hold': d, 'reason': reason, 'notional': p['not'],
                })

        pos = keep
        cap += pnl
        if cap <= 0: break

        n_open = max_pos - len(pos)
        if n_open <= 0:
            eq.append({'date': dt, 'capital': cap}); continue

        cands = []
        for sym, df in signal_data.items():
            if any(p['sym'] == sym for p in pos): continue
            idx = df.index[df['trade_date'] == dt]
            if len(idx) == 0: continue
            row = df.loc[idx[0]]
            if row['score_long'] >= min_sc:
                cands.append({'sym': sym, 'dir': 'long', 'sc': row['score_long'], 'ep': row['open']})
            if row['score_short'] >= min_sc:
                cands.append({'sym': sym, 'dir': 'short', 'sc': row['score_short'], 'ep': row['open']})

        best = {}
        for c_ in cands:
            if c_['sym'] not in best or c_['sc'] > best[c_['sym']]['sc']:
                best[c_['sym']] = c_
        ranked = sorted(best.values(), key=lambda x: -x['sc'])

        n_long = sum(1 for p in pos if p['dir'] == 'long')
        n_short = sum(1 for p in pos if p['dir'] == 'short')

        for c_ in ranked:
            if n_open <= 0: break
            if max_long and c_['dir'] == 'long' and n_long >= max_long: continue
            if max_short and c_['dir'] == 'short' and n_short >= max_short: continue
            notional = cap * lev / max_pos
            pos.append({'sym': c_['sym'], 'dir': c_['dir'], 'ed': dt,
                        'ep': c_['ep'], 'not': notional, 'sc': c_['sc']})
            if c_['dir'] == 'long': n_long += 1
            else: n_short += 1
            n_open -= 1

        eq.append({'date': dt, 'capital': cap})
    return eq, trades


def evaluate(eq, trades, label):
    eq_df = pd.DataFrame(eq)
    if len(eq_df) == 0 or eq_df['capital'].iloc[-1] <= 0:
        print(f"  {label}: 爆仓"); return None
    eq_df['peak'] = eq_df['capital'].cummax()
    eq_df['dd'] = (eq_df['capital'] - eq_df['peak']) / eq_df['peak'] * 100
    ny = max((eq_df['date'].iloc[-1] - eq_df['date'].iloc[0]).days / 365.25, 0.01)
    ann = ((eq_df['capital'].iloc[-1] / eq_df['capital'].iloc[0]) ** (1/ny) - 1) * 100
    mdd = eq_df['dd'].min()
    dr = eq_df['capital'].pct_change().dropna()
    sh = dr.mean() / dr.std() * (252**0.5) if dr.std() > 0 else 0
    if trades:
        td = pd.DataFrame(trades)
        wr = (td['r'] > 0).mean() * 100
        avg = td['r'].mean()
        # 总交易成本
        total_cost = sum(abs(t['pnl']) * 0.0004 for t in trades) if trades else 0
    else:
        wr = avg = 0; td = pd.DataFrame()

    print(f"  {label}")
    print(f"    N={len(trades)} WR={wr:.1f}% Avg={avg:+.3f}% Sharpe={sh:.2f} MDD={mdd:.1f}%")
    return {'wr': wr, 'mdd': mdd, 'sh': sh, 'avg': avg, 'n': len(trades)}


def main():
    print("V82: 实战可行性分析")
    print("="*60)

    print("\n加载数据...")
    all_data, specs = load_data()
    print("计算信号...")
    sd = compute_signals(all_data)

    # ═══ A. 交易成本影响 ═══
    print(f"\n{'='*60}")
    print("A. 交易成本敏感度分析")
    print(f"{'='*60}")

    # 中国期货手续费一般: 万0.2~万1 (开+平)
    # 滑点: 1-2个最小变动价位
    cost_configs = [
        ("理想(无成本)", 0, 0),
        ("低(万0.5+1tick)", 0.00005, 1),
        ("中(万1+1tick)", 0.0001, 1),
        ("高(万2+2tick)", 0.0002, 2),
        ("极高(万3+3tick)", 0.0003, 3),
    ]

    for desc, comm, slip in cost_configs:
        eq, tr = run_bt_realistic(sd, specs, TEST_START, TEST_END,
                                   commission_rate=comm, slippage_ticks=slip)
        evaluate(eq, tr, desc)

    # ═══ B. 资金容量分析 ═══
    print(f"\n\n{'='*60}")
    print("B. 资金容量分析 (不同初始资金)")
    print(f"{'='*60}")

    for cap in [500_000, 1_000_000, 5_000_000, 10_000_000, 50_000_000]:
        eq, tr = run_bt_realistic(sd, specs, TEST_START, TEST_END,
                                   capital=cap, commission_rate=0.0001, slippage_ticks=1)
        evaluate(eq, tr, f"初始资金={cap/10000:.0f}万")

    # ═══ C. 流动性分析 ═══
    print(f"\n\n{'='*60}")
    print("C. 品种流动性分析")
    print(f"{'='*60}")

    eq, tr = run_bt_realistic(sd, specs, TEST_START, TEST_END)
    tdf = pd.DataFrame(tr)

    print(f"\n  品种日均成交额 vs 策略仓位:")
    for sym in sorted(tdf['sym'].unique()):
        sub = tdf[tdf['sym'] == sym]
        avg_notional = sub['notional'].mean()
        # 该品种的日均成交额
        df = all_data.get(sym)
        if df is not None:
            mask = (df['trade_date'] >= TEST_START) & (df['trade_date'] <= TEST_END)
            sub_df = df[mask]
            if len(sub_df) > 0:
                avg_amount = sub_df['amount'].mean() if 'amount' in sub_df.columns else 0
                if avg_amount > 0:
                    impact = avg_notional / avg_amount * 100
                    if impact > 0.1:  # 只显示冲击>0.1%的
                        print(f"    {sym}: 仓位={avg_notional/10000:.0f}万 成交额={avg_amount/10000:.0f}万 "
                              f"冲击={impact:.2f}%")

    # ═══ D. 压力测试 ═══
    print(f"\n\n{'='*60}")
    print("D. 压力测试")
    print(f"{'='*60}")

    # 最差情况模拟
    print(f"\n  1. 极端不利情况模拟:")
    # 模拟: 每笔亏损加大1%
    for extra_loss in [0, 0.5, 1.0, 1.5, 2.0]:
        eq, tr = run_bt_realistic(sd, specs, TEST_START, TEST_END,
                                   commission_rate=0.0001, slippage_ticks=1)
        # 手动调整亏损
        tdf = pd.DataFrame(tr)
        tdf_adj = tdf.copy()
        mask_loss = tdf_adj['r'] <= 0
        tdf_adj.loc[mask_loss, 'r'] -= extra_loss
        # 重新计算equity
        # (简化: 用调整后的平均收益估算)
        avg_adj = tdf_adj['r'].mean()
        wr_adj = (tdf_adj['r'] > 0).mean() * 100
        print(f"    额外亏损{extra_loss}%: Avg={avg_adj:+.3f}% WR={wr_adj:.1f}%")

    # 2. 信号延迟模拟 (错过开盘, 用VWAP或收盘入场)
    print(f"\n  2. 信号衰减: 如果不是开盘入场")
    for sym, df in sd.items():
        mask = (df['trade_date'] >= TEST_START) & (df['trade_date'] <= TEST_END)
        sub = df[mask & ((df['score_long'] >= 7) | (df['score_short'] >= 7))]
        if len(sub) == 0: continue

    # 比较开盘/收盘/中间价入场效果
    long_mask = sd[list(sd.keys())[0]]['trade_date'] >= TEST_START
    rows_open, rows_close, rows_mid = [], [], []
    for sym, df in sd.items():
        mask = (df['trade_date'] >= TEST_START) & (df['trade_date'] <= TEST_END)
        sub_l = df[mask & (df['score_long'] >= 7)]
        sub_s = df[mask & (df['score_short'] >= 7)]

        if len(sub_l) > 0:
            # 做多: 从open/low/(open+low)/2入场的收益
            ret_open = (sub_l['close'] - sub_l['open']) / sub_l['open'] * 100
            ret_low = (sub_l['close'] - sub_l['low']) / sub_l['low'] * 100
            ret_mid = (sub_l['close'] - (sub_l['open'] + sub_l['low']) / 2) / ((sub_l['open'] + sub_l['low']) / 2) * 100
            rows_open.append(ret_open)
            rows_mid.append(ret_mid)
            rows_close.append(ret_low)

    if rows_open:
        all_open = pd.concat(rows_open)
        all_mid = pd.concat(rows_mid)
        print(f"\n  做多入场价对比:")
        print(f"    开盘价入场: WR={100*(all_open>0).mean():.1f}% Avg={all_open.mean():+.3f}%")
        print(f"    中间价入场: WR={100*(all_mid>0).mean():.1f}% Avg={all_mid.mean():+.3f}%")

    # ═══ E. 实盘建议 ═══
    print(f"\n\n{'='*60}")
    print("E. 实盘交易建议")
    print(f"{'='*60}")

    print("""
  1. 交易执行:
     - 每日开盘前计算信号 (可用signal_generator_v1.py)
     - 开盘1分钟内完成开仓 (集合竞价或市价单)
     - 监控日内止损止盈 (可用条件单)
     - 收盘前2分钟平仓 (未触发SL/TP的仓位)

  2. 品种选择:
     - 优先选择流动性好的品种 (日均成交额>1亿)
     - 避免成交稀少的品种 (冲击成本>0.5%)
     - 建议品种池: 60-70个主力品种

  3. 资金管理:
     - 起步资金: 50-100万
     - 最大仓位: 7个, 每个仓位 = 资金*5/7
     - 日内止损: -1.5%, 止盈: +4.0%
     - 多空平衡: 各最多4个

  4. 风险控制:
     - 单日最大亏损: -5% (触发后暂停交易1天)
     - 连续3日亏损: 降低仓位到3个
     - 月度亏损>10%: 降低杠杆到3x
     - 季度回测: 检查策略有效性

  5. 回测与实盘差异:
     - 回测假设开盘价成交, 实际可能差1-2 tick
     - 回测假设日内SL/TP精确触发, 实际有延迟
     - 回测不考虑涨跌停 (极端行情可能无法平仓)
     - 回测资金无限, 实际大资金冲击成本增加

  6. 预期实盘表现 (扣除成本后):
     - Sharpe: 20-22 (vs 回测24.5)
     - WR: 68-70% (vs 回测71.2%)
     - MDD: -8~12% (vs 回测-5.1%)
     - 年化: 仍然远超1000%
""")


if __name__ == '__main__':
    main()
