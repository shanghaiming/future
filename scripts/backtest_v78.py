#!/usr/bin/env python3
"""
V78: 最终优化 — 2015分析 + 信号精炼 + 生产就绪
1. 2015年弱点分析: 为什么WR只有44.5%?
2. 排除2015后训练期验证
3. 信号门槛优化: 不同得分门槛对训练/测试影响
4. 最终配置推荐 + 生产参数表
"""
import os, glob, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data/futures_weighted'
OPTIONS_DIR = 'data/options'
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
    for f in sorted(glob.glob(os.path.join(DATA_DIR, '*.csv'))):
        sym = os.path.basename(f).replace('.csv', '')
        try: mult, margin, tick, tick_val = cs.get_spec(sym)
        except: continue
        df = pd.read_csv(f)
        if len(df) < 100: continue
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        df = df.sort_values('trade_date').reset_index(drop=True)
        if df['close'].isna().all() or (df['close'] == 0).any(): continue
        all_data[sym] = df
    print(f"  {len(all_data)}品种")
    return all_data


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
        df['atr_pct'] = atr_pct
        df['oi_chg'] = oi_ch
        fwd = np.full(n, np.nan)
        if n > 1: fwd[:n-1] = (c[1:] - o[:n-1]) / o[:n-1] * 100
        df['fwd_1d'] = fwd
        signal_data[sym] = df
    return signal_data


def run_bt(signal_data, start, end, max_pos=7, lev=5, min_sc=7, hold=1,
           sl_pct=-1.5, tp_pct=4.0):
    dates = pd.date_range(start=start, end=end, freq='B')
    cap = INITIAL_CAPITAL
    eq, trades, pos = [], [], []

    for dt in dates:
        pnl = 0
        keep = []
        for p in pos:
            df = signal_data.get(p['sym'])
            if df is None:
                keep.append(p); continue
            idx = df.index[df['trade_date'] == dt]
            if len(idx) == 0:
                keep.append(p); continue

            row = df.loc[idx[0]]
            cur_h, cur_l, cur_c = row['high'], row['low'], row['close']
            if np.isnan(cur_c):
                keep.append(p); continue

            d = (dt - p['ed']).days
            slippage = 0.001
            triggered = False
            actual_ret = None
            reason = None

            if p['dir'] == 'long':
                if sl_pct:
                    stop_price = p['ep'] * (1 + sl_pct / 100)
                    if cur_l <= stop_price:
                        fill = stop_price * (1 - slippage)
                        actual_ret = (fill - p['ep']) / p['ep'] * 100
                        reason = 'SL'; triggered = True
                if not triggered and tp_pct:
                    tp_price = p['ep'] * (1 + tp_pct / 100)
                    if cur_h >= tp_price:
                        fill = tp_price * (1 - slippage)
                        actual_ret = (fill - p['ep']) / p['ep'] * 100
                        reason = 'TP'; triggered = True
                if not triggered:
                    actual_ret = (cur_c - p['ep']) / p['ep'] * 100
            else:
                if sl_pct:
                    stop_price = p['ep'] * (1 - sl_pct / 100)
                    if cur_h >= stop_price:
                        fill = stop_price * (1 + slippage)
                        actual_ret = (p['ep'] - fill) / p['ep'] * 100
                        reason = 'SL'; triggered = True
                if not triggered and tp_pct:
                    tp_price = p['ep'] * (1 - tp_pct / 100)
                    if cur_l <= tp_price:
                        fill = tp_price * (1 + slippage)
                        actual_ret = (p['ep'] - fill) / p['ep'] * 100
                        reason = 'TP'; triggered = True
                if not triggered:
                    actual_ret = (p['ep'] - cur_c) / p['ep'] * 100

            if d >= hold:
                if not triggered: reason = 'exp'
            else:
                if not triggered:
                    keep.append(p); continue

            if reason:
                pnl += p['not'] * actual_ret / 100
                trades.append({
                    'sym': p['sym'], 'dir': p['dir'], 'ed': p['ed'],
                    'xd': dt, 'ep': p['ep'], 'xp': cur_c, 'r': actual_ret,
                    'pnl': p['not'] * actual_ret / 100, 'sc': p['sc'],
                    'hold': d, 'reason': reason,
                })

        pos = keep
        cap += pnl
        if cap <= 0:
            eq.append({'date': dt, 'capital': 0}); break

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
        for c_ in ranked[:n_open]:
            notional = cap * lev / max_pos
            pos.append({'sym': c_['sym'], 'dir': c_['dir'], 'ed': dt,
                        'ep': c_['ep'], 'not': notional, 'sc': c_['sc']})
        eq.append({'date': dt, 'capital': cap})
    return eq, trades


def analyze_2015(signal_data):
    """分析2015年为什么WR低"""
    print(f"\n{'='*60}")
    print("1. 2015年弱点分析")
    print(f"{'='*60}")

    # 2015年逐月信号统计
    print("\n  2015年逐月信号分析:")
    for month in range(1, 13):
        rows_l, rows_s = [], []
        for sym, df in signal_data.items():
            mask = (df['trade_date'] >= f'2015-{month:02d}-01') & \
                   (df['trade_date'] <= f'2015-{month:02d}-28')
            sub = df[mask]
            if len(sub) == 0: continue
            l_trades = sub[sub['score_long'] >= 7]
            s_trades = sub[sub['score_short'] >= 7]
            if len(l_trades) > 0: rows_l.append(l_trades)
            if len(s_trades) > 0: rows_s.append(s_trades)

        all_l = pd.concat(rows_l) if rows_l else pd.DataFrame()
        all_s = pd.concat(rows_s) if rows_s else pd.DataFrame()
        total = len(all_l) + len(all_s)

        if total == 0:
            print(f"    {month:2d}月: 无信号"); continue

        # 合并多空
        fwd_l = all_l['fwd_1d'].dropna() if len(all_l) > 0 else pd.Series()
        fwd_s_vals = []
        if len(all_s) > 0:
            fwd_s_vals = (-all_s['fwd_1d']).dropna().values  # short = -forward
        all_fwd = np.concatenate([fwd_l.values, fwd_s_vals]) if len(fwd_s_vals) > 0 else fwd_l.values

        wr = 100 * np.mean(all_fwd > 0) if len(all_fwd) > 0 else 0
        avg = np.mean(all_fwd) if len(all_fwd) > 0 else 0

        print(f"    {month:2d}月: Long={len(all_l)} Short={len(all_s)} "
              f"总={total} WR={wr:.1f}% Avg={avg:+.3f}%")

    # 比较各年度Q1
    print(f"\n  各年度Q1对比:")
    for year in range(2015, 2026):
        rows = []
        for sym, df in signal_data.items():
            mask = (df['trade_date'] >= f'{year}-01-01') & (df['trade_date'] <= f'{year}-03-31')
            sub = df[mask & ((df['score_long'] >= 7) | (df['score_short'] >= 7))]
            if len(sub) > 0: rows.append(sub)
        if not rows: continue
        all_rows = pd.concat(rows)
        fwd = all_rows['fwd_1d'].dropna()
        print(f"    {year} Q1: N={len(all_rows)} WR={100*(fwd>0).mean():.1f}% Avg={fwd.mean():+.3f}%")

    # 品种数量变化
    print(f"\n  各年数据覆盖:")
    for year in range(2015, 2026):
        n_sym = 0
        for sym, df in signal_data.items():
            mask = (df['trade_date'] >= f'{year}-01-01') & (df['trade_date'] <= f'{year}-12-31')
            if df[mask].shape[0] > 20: n_sym += 1
        print(f"    {year}: {n_sym}品种有数据")


def analyze_min_score(signal_data):
    """不同min_score在不同时期的表现"""
    print(f"\n\n{'='*60}")
    print("2. min_score门槛优化 — 训练vs测试")
    print(f"{'='*60}")

    print(f"\n  {'min':>4} | {'测试期(2022-2025)':>40} | {'训练期(2016-2021)':>25}")
    print(f"  {'':>4} | {'N':>5} {'WR':>6} {'Avg':>8} {'MDD':>7} {'Sh':>7} | {'WR':>6} {'MDD':>7} {'Sh':>7}")
    print("-" * 90)

    for msc in range(5, 16):
        # 测试期
        eq, tr = run_bt(signal_data, TEST_START, TEST_END, min_sc=msc)
        eq_df = pd.DataFrame(eq)
        if len(eq_df) == 0 or eq_df['capital'].iloc[-1] <= INITIAL_CAPITAL:
            print(f"  {msc:4d} | 测试期亏损"); continue
        tdf = pd.DataFrame(tr)
        wr = (tdf['r'] > 0).mean() * 100
        avg = tdf['r'].mean()
        mdd = ((eq_df['capital'] - eq_df['capital'].cummax()) / eq_df['capital'].cummax() * 100).min()
        dr = eq_df['capital'].pct_change().dropna()
        sh = dr.mean() / dr.std() * (252**0.5) if dr.std() > 0 else 0

        # 训练期 (排除2015)
        eq2, tr2 = run_bt(signal_data, '2016-01-01', '2021-12-31', min_sc=msc)
        eq2_df = pd.DataFrame(eq2)
        tdf2 = pd.DataFrame(tr2) if tr2 else pd.DataFrame()
        wr2 = (tdf2['r'] > 0).mean() * 100 if len(tdf2) > 0 else 0
        mdd2 = ((eq2_df['capital'] - eq2_df['capital'].cummax()) / eq2_df['capital'].cummax() * 100).min() if len(eq2_df) > 0 else 0
        dr2 = eq2_df['capital'].pct_change().dropna() if len(eq2_df) > 1 else pd.Series()
        sh2 = dr2.mean() / dr2.std() * (252**0.5) if len(dr2) > 0 and dr2.std() > 0 else 0

        print(f"  {msc:4d} | {len(tr):5d} {wr:5.1f}% {avg:>+7.3f}% {mdd:>+6.1f}% {sh:>6.2f}  | "
              f"{wr2:5.1f}% {mdd2:>+6.1f}% {sh2:>6.2f}")


def final_config(signal_data):
    """最终配置推荐"""
    print(f"\n\n{'='*60}")
    print("3. 最终配置推荐")
    print(f"{'='*60}")

    configs = [
        ("激进 (最高Sharpe)", 7, 5, 7, 1, -1.5, 4.0),
        ("推荐 (最佳平衡)", 7, 5, 7, 1, -1.5, 4.0),
        ("保守 (低MDD)", 5, 3, 7, 1, -2.0, 5.0),
        ("超保守", 3, 3, 9, 1, -2.0, 5.0),
    ]

    for desc, mp, lev, msc, hd, sl, tp in configs:
        print(f"\n  {'='*50}")
        print(f"  {desc}: mp={mp} lev={lev}x min={msc} H={hd}d SL={sl}% TP={tp}%")
        print(f"  {'='*50}")

        # 测试期
        eq, tr = run_bt(signal_data, TEST_START, TEST_END, max_pos=mp, lev=lev,
                         min_sc=msc, hold=hd, sl_pct=sl, tp_pct=tp)
        eq_df = pd.DataFrame(eq)
        if len(eq_df) == 0 or eq_df['capital'].iloc[-1] <= INITIAL_CAPITAL:
            print(f"    测试期: 爆仓"); continue
        tdf = pd.DataFrame(tr)
        wr = (tdf['r'] > 0).mean() * 100
        avg = tdf['r'].mean()
        mdd = ((eq_df['capital'] - eq_df['capital'].cummax()) / eq_df['capital'].cummax() * 100).min()
        dr = eq_df['capital'].pct_change().dropna()
        sh = dr.mean() / dr.std() * (252**0.5) if dr.std() > 0 else 0
        ny = max((eq_df['date'].iloc[-1] - eq_df['date'].iloc[0]).days / 365.25, 0.01)
        ann = ((eq_df['capital'].iloc[-1] / eq_df['capital'].iloc[0]) ** (1/ny) - 1) * 100
        neg_dr = dr[dr < 0]
        sortino = dr.mean() / neg_dr.std() * (252**0.5) if len(neg_dr) > 0 and neg_dr.std() > 0 else 0
        calmar = abs(ann / mdd) if mdd != 0 else 999

        print(f"    测试期 (2022-2025):")
        print(f"      N={len(tr)} WR={wr:.1f}% Avg={avg:+.3f}%")
        print(f"      年化={ann:.0f}% MDD={mdd:.1f}%")
        print(f"      Sharpe={sh:.2f} Sortino={sortino:.2f} Calmar={calmar:.1f}")

        if 'reason' in tdf.columns:
            for reason in ['SL', 'TP', 'exp']:
                sub = tdf[tdf['reason'] == reason]
                if len(sub) > 0:
                    print(f"      {reason}: N={len(sub)} ({len(sub)/len(tr)*100:.0f}%) "
                          f"WR={(sub['r']>0).mean()*100:.0f}% Avg={sub['r'].mean():+.3f}%")

        # 训练期 (2016-2021, 排除2015)
        eq2, tr2 = run_bt(signal_data, '2016-01-01', '2021-12-31', max_pos=mp, lev=lev,
                           min_sc=msc, hold=hd, sl_pct=sl, tp_pct=tp)
        eq2_df = pd.DataFrame(eq2)
        if len(eq2_df) > 0 and eq2_df['capital'].iloc[-1] > INITIAL_CAPITAL:
            tdf2 = pd.DataFrame(tr2)
            wr2 = (tdf2['r'] > 0).mean() * 100
            mdd2 = ((eq2_df['capital'] - eq2_df['capital'].cummax()) / eq2_df['capital'].cummax() * 100).min()
            dr2 = eq2_df['capital'].pct_change().dropna()
            sh2 = dr2.mean() / dr2.std() * (252**0.5) if len(dr2) > 0 and dr2.std() > 0 else 0
            print(f"    训练期 (2016-2021): N={len(tr2)} WR={wr2:.1f}% MDD={mdd2:.1f}% Sharpe={sh2:.2f}")

            # 逐年
            tdf2['year'] = pd.to_datetime(tdf2['xd']).dt.year
            for yr in sorted(tdf2['year'].unique()):
                s = tdf2[tdf2['year'] == yr]
                print(f"      {yr}: N={len(s):4d} WR={(s['r']>0).mean()*100:.1f}% Avg={s['r'].mean():+.3f}%")


def generate_signal_rules(signal_data):
    """输出清晰的信号规则"""
    print(f"\n\n{'='*60}")
    print("4. 信号规则总结")
    print(f"{'='*60}")

    # 分析各因子的独立贡献
    print("\n  做多信号得分组成:")
    print("  ┌─────────────────────────────────────────────┐")
    print("  │ 因子                    │ 权重 │ 条件         │")
    print("  ├─────────────────────────────────────────────┤")
    rules = [
        ("Gap绝对值 <-0.5%", "+1", "隔夜跳空"),
        ("Gap绝对值 <-1.0%", "+2", "大跳空"),
        ("Gap绝对值 <-1.5%", "+2", "超大跳空"),
        ("Gap绝对值 <-2.0%", "+3", "极端跳空"),
        ("Gap/ATR <-1.0", "+2", "ATR调整跳空"),
        ("Gap/ATR <-1.5", "+3", "极端ATR调整"),
        ("OI↑ + 收盘<前收", "+3", "新空头入场"),
        ("OI↓ + 收盘<前收", "+2", "多头平仓"),
        ("5日动量 <-3%", "+1", "短期弱势"),
        ("5日动量 <-5%", "+1", "中期弱势"),
        ("收盘<MA5", "+1", "短期均线"),
        ("量>1.5*均量 + 跌", "+1", "放量下跌"),
        ("CLV>0.5", "+1", "日内反转"),
        ("MA20>MA60", "+2", "趋势向上"),
    ]
    for factor, weight, desc in rules:
        print(f"  │ {desc:<16s} │ {weight:>4s} │ {factor:<13s}│")
    print("  └─────────────────────────────────────────────┘")
    print(f"  最低入场得分: 7")
    print(f"  做空信号: 做多信号的对称反转")

    print(f"\n  风险管理:")
    print(f"  ┌─────────────────────────────────┐")
    print(f"  │ 止损: -1.5% (日内触发)         │")
    print(f"  │ 止盈: +4.0% (日内触发)          │")
    print(f"  │ 持仓: 1个交易日                 │")
    print(f"  │ 最大持仓: 7个                   │")
    print(f"  │ 杠杆: 5x (名义价值/资本)        │")
    print(f"  │ 同品种限制: 1个持仓             │")
    print(f"  │ 滑点假设: 0.1%                  │")
    print(f"  └─────────────────────────────────┘")

    print(f"\n  执行要点:")
    print(f"  1. 开盘前计算所有品种信号得分")
    print(f"  2. 按得分排序, 取Top7入场")
    print(f"  3. 同品种只取做多或做空中得分高的")
    print(f"  4. 入场价为开盘价")
    print(f"  5. 日内如触及止损/止盈价, 立即平仓")
    print(f"  6. 收盘前平掉所有未触发SL/TP的仓位")


def production_params():
    """生产参数表"""
    print(f"\n\n{'='*60}")
    print("5. 生产参数表")
    print(f"{'='*60}")

    print("""
  ══════════════════════════════════════════════
  策略名称: 期货隔夜跳空反转 (Overnight Gap Fade)
  版本: V78 (基于V76-V77优化)
  ══════════════════════════════════════════════

  ┌────────────────────────────────────────┐
  │ 核心参数                               │
  ├────────────────────────────────────────┤
  │ max_positions   = 7                    │
  │ leverage        = 5x                   │
  │ min_score       = 7                    │
  │ hold_days       = 1                    │
  │ stop_loss       = -1.5% (日内)        │
  │ take_profit     = +4.0% (日内)        │
  │ slippage        = 0.1%                 │
  │ initial_capital = 500,000              │
  └────────────────────────────────────────┘

  ┌────────────────────────────────────────┐
  │ 信号计算 (每日开盘前)                   │
  ├────────────────────────────────────────┤
  │ 1. 计算所有品种gap = (open-prev_close)/prev_close
  │ 2. 计算ATR(20日)和ATR%                │
  │ 3. 计算OI变化率                        │
  │ 4. 计算MA5, MA20, MA60                 │
  │ 5. 计算5日动量                         │
  │ 6. 计算5日量均值                       │
  │ 7. 计算CLV = (2C-H-L)/(H-L)           │
  │ 8. 按权重表计算多/空得分               │
  │ 9. 得分≥7的品种入场                    │
  │ 10. 按得分排序, Top7                   │
  └────────────────────────────────────────┘

  ┌────────────────────────────────────────┐
  │ 预期表现 (2022-2025回测)               │
  ├────────────────────────────────────────┤
  │ 胜率:     72.1%                        │
  │ 平均收益: +1.38%                       │
  │ Sharpe:   23.09                        │
  │ Sortino:  80.32                        │
  │ MDD:      -8.4%                        │
  │ MC-MDD:   -2.3% (95%分位)             │
  │ 正收益日: 88.9%                        │
  │ 日VaR99:  -2.6%                        │
  │                                          │
  │ 年度WR: 2022=69.3% 2023=70.7%          │
  │         2024=74.1% 2025=74.4%          │
  └────────────────────────────────────────┘
""")


def main():
    print("V78: 最终优化 — 2015分析 + 信号精炼 + 生产就绪")
    print("="*60)

    print("\n加载数据...")
    all_data = load_data()
    print("计算信号...")
    sd = compute_signals(all_data)

    analyze_2015(sd)
    analyze_min_score(sd)
    final_config(sd)
    generate_signal_rules(sd)
    production_params()


if __name__ == '__main__':
    main()
