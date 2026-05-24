#!/usr/bin/env python3
"""
V75: 品种分析 + 策略稳健性测试
基于V74最佳配置 (mp=7, lev=5, H=1d, SL=-2, TP=3)
分析:
1. 品种贡献度 — 哪些品种驱动收益
2. 品种独立性 — 是否过于集中
3. 滚动窗口稳定性 — 策略是否持续有效
4. 市场环境分析 — 牛/熊/震荡
5. 信号质量深化 — 做多vs做空, 高分vs低分
6. 时间模式 — 周几/月初月末
7. 自适应参数 — 不同时期最优参数是否变化
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

        # Long score
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

        # Short score
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

        for hd in [1, 2, 3, 5]:
            fwd = np.full(n, np.nan)
            if n > hd: fwd[:n-hd] = (c[hd:] - o[:n-hd]) / o[:n-hd] * 100
            df[f'fwd_{hd}d'] = fwd

        df['score_long'] = s_l
        df['score_short'] = s_s
        df['gap_pct'] = gap
        df['atr_pct'] = atr_pct
        df['oi_chg'] = oi_ch
        df['vol'] = v
        signal_data[sym] = df
    return signal_data


def run_bt(signal_data, start, end, max_pos=7, lev=5, min_sc=7, hold=1,
           sl=-2, tp=3):
    """回测引擎 — V74最佳配置"""
    dates = pd.date_range(start=start, end=end, freq='B')
    cap = INITIAL_CAPITAL
    eq = []
    trades = []
    pos = []

    for dt in dates:
        pnl = 0
        keep = []
        for p in pos:
            df = signal_data.get(p['sym'])
            cur = None
            if df is not None:
                idx = df.index[df['trade_date'] == dt]
                if len(idx) > 0: cur = df.loc[idx[0], 'close']
            if cur is None or np.isnan(cur):
                keep.append(p); continue
            r = (cur - p['ep']) / p['ep'] * 100 if p['dir'] == 'long' else (p['ep'] - cur) / p['ep'] * 100
            d = (dt - p['ed']).days
            reason = None
            if sl and r <= sl: reason = 'SL'
            elif tp and r >= tp: reason = 'TP'
            elif d >= hold: reason = 'exp'
            if reason:
                pnl += p['not'] * r / 100
                trades.append({'sym': p['sym'], 'dir': p['dir'], 'ed': p['ed'],
                               'xd': dt, 'ep': p['ep'], 'xp': cur, 'r': r,
                               'pnl': p['not'] * r / 100, 'sc': p['sc'],
                               'hold': d, 'reason': reason})
            else:
                keep.append(p)
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
        for c in cands:
            if c['sym'] not in best or c['sc'] > best[c['sym']]['sc']:
                best[c['sym']] = c

        ranked = sorted(best.values(), key=lambda x: -x['sc'])
        for c in ranked[:n_open]:
            notional = cap * lev / max_pos
            pos.append({'sym': c['sym'], 'dir': c['dir'], 'ed': dt,
                        'ep': c['ep'], 'not': notional, 'sc': c['sc']})

        eq.append({'date': dt, 'capital': cap})
    return eq, trades


def analyze_per_commodity(signal_data, trades):
    """1. 品种贡献度分析"""
    print(f"\n{'='*70}")
    print("1. 品种贡献度分析")
    print(f"{'='*70}")

    tdf = pd.DataFrame(trades)
    tdf['year'] = pd.to_datetime(tdf['xd']).dt.year

    # 按品种统计
    sym_stats = []
    for sym in sorted(tdf['sym'].unique()):
        sub = tdf[tdf['sym'] == sym]
        n = len(sub)
        wr = (sub['r'] > 0).mean() * 100
        avg = sub['r'].mean()
        pnl = sub['pnl'].sum()
        long_n = len(sub[sub['dir'] == 'long'])
        short_n = len(sub[sub['dir'] == 'short'])
        sym_stats.append({
            'sym': sym, 'n': n, 'wr': wr, 'avg': avg, 'pnl': pnl,
            'long': long_n, 'short': short_n
        })

    sdf = pd.DataFrame(sym_stats).sort_values('pnl', ascending=False)
    total_pnl = sdf['pnl'].sum()

    print(f"\n{'品种':>8} {'N':>5} {'WR':>6} {'Avg':>8} {'PnL':>12} {'占比':>6} {'多/空':>7}")
    print("-" * 70)
    for _, row in sdf.head(30).iterrows():
        pct = row['pnl'] / total_pnl * 100 if total_pnl != 0 else 0
        print(f"{row['sym']:>8} {row['n']:5d} {row['wr']:5.1f}% {row['avg']:>+7.3f}% "
              f"{row['pnl']:>11.0f} {pct:>5.1f}% {row['long']:>3d}/{row['short']:<3d}")

    # 累计贡献
    sdf['cum_pct'] = sdf['pnl'].cumsum() / total_pnl * 100
    top10 = sdf.head(10)['pnl'].sum() / total_pnl * 100
    top20 = sdf.head(20)['pnl'].sum() / total_pnl * 100
    print(f"\n  Top 10品种贡献: {top10:.1f}%")
    print(f"  Top 20品种贡献: {top20:.1f}%")
    print(f"  参与品种总数: {len(sdf)}")
    print(f"  盈利品种: {len(sdf[sdf['pnl'] > 0])}, 亏损品种: {len(sdf[sdf['pnl'] <= 0])}")

    return sdf


def analyze_direction(signal_data, trades):
    """2. 做多vs做空分析"""
    print(f"\n{'='*70}")
    print("2. 做多 vs 做空分析")
    print(f"{'='*70}")

    tdf = pd.DataFrame(trades)
    tdf['year'] = pd.to_datetime(tdf['xd']).dt.year

    for direction in ['long', 'short']:
        sub = tdf[tdf['dir'] == direction]
        if len(sub) == 0: continue
        wr = (sub['r'] > 0).mean() * 100
        avg = sub['r'].mean()
        pnl = sub['pnl'].sum()
        print(f"\n  {direction.upper()}: N={len(sub)} WR={wr:.1f}% Avg={avg:+.3f}% PnL={pnl:,.0f}")
        for yr in sorted(sub['year'].unique()):
            ys = sub[sub['year'] == yr]
            print(f"    {yr}: N={len(ys):4d} WR={(ys['r']>0).mean()*100:.1f}% Avg={ys['r'].mean():+.3f}%")

    # 多空对比
    long_trades = tdf[tdf['dir'] == 'long']
    short_trades = tdf[tdf['dir'] == 'short']
    print(f"\n  多空比例: {len(long_trades)} : {len(short_trades)} = "
          f"{len(long_trades)/len(short_trades):.2f}:1" if len(short_trades) > 0 else "")
    print(f"  多头WR: {(long_trades['r']>0).mean()*100:.1f}% vs 空头WR: {(short_trades['r']>0).mean()*100:.1f}%")


def analyze_rolling_stability(signal_data, start, end):
    """3. 滚动窗口稳定性"""
    print(f"\n{'='*70}")
    print("3. 滚动窗口稳定性 (60/120/252天)")
    print(f"{'='*70}")

    eq, trades = run_bt(signal_data, start, end, max_pos=7, lev=5, min_sc=7, hold=1, sl=-2, tp=3)
    eq_df = pd.DataFrame(eq)
    if len(eq_df) == 0:
        print("  无数据"); return

    eq_df = eq_df.set_index('date')
    daily_ret = eq_df['capital'].pct_change().dropna()

    for window in [60, 120, 252]:
        rolling_wr = []
        rolling_avg = []
        tdf = pd.DataFrame(trades)
        tdf['xd'] = pd.to_datetime(tdf['xd'])

        for i in range(window, len(daily_ret), 20):  # Sample every 20 days
            end_date = daily_ret.index[i]
            start_date = daily_ret.index[i - window]
            sub = tdf[(tdf['xd'] >= start_date) & (tdf['xd'] <= end_date)]
            if len(sub) >= 10:
                rolling_wr.append((sub['r'] > 0).mean() * 100)
                rolling_avg.append(sub['r'].mean())

        if rolling_wr:
            print(f"\n  {window}天滚动窗口 (样本数={len(rolling_wr)}):")
            print(f"    WR: 均值={np.mean(rolling_wr):.1f}% 最低={np.min(rolling_wr):.1f}% "
                  f"最高={np.max(rolling_wr):.1f}% 中位数={np.median(rolling_wr):.1f}%")
            print(f"    Avg Return: 均值={np.mean(rolling_avg):+.3f}% 最低={np.min(rolling_avg):+.3f}% "
                  f"最高={np.max(rolling_avg):+.3f}%")
            # WR低于50%的占比
            bad_pct = sum(1 for w in rolling_wr if w < 50) / len(rolling_wr) * 100
            print(f"    WR<50%时段占比: {bad_pct:.1f}%")

    # 滚动Sharpe
    for window in [60, 120, 252]:
        rsh = daily_ret.rolling(window).apply(lambda x: x.mean() / x.std() * (252**0.5) if x.std() > 0 else 0)
        rsh = rsh.dropna()
        if len(rsh) > 0:
            print(f"  {window}天滚动Sharpe: 均值={rsh.mean():.2f} 最低={rsh.min():.2f} "
                  f"最高={rsh.max():.2f} <0占比={100*(rsh<0).mean():.1f}%")


def analyze_time_patterns(trades):
    """4. 时间模式分析"""
    print(f"\n{'='*70}")
    print("4. 时间模式分析")
    print(f"{'='*70}")

    tdf = pd.DataFrame(trades)
    tdf['xd'] = pd.to_datetime(tdf['xd'])
    tdf['dow'] = tdf['xd'].dt.dayofweek
    tdf['dom'] = tdf['xd'].dt.day
    tdf['month'] = tdf['xd'].dt.month

    # 星期几
    print(f"\n  星期几:")
    for dow in range(5):
        sub = tdf[tdf['dow'] == dow]
        if len(sub) == 0: continue
        names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        print(f"    {names[dow]}: N={len(sub):4d} WR={(sub['r']>0).mean()*100:.1f}% Avg={sub['r'].mean():+.3f}%")

    # 月初vs月末
    tdf['is_month_start'] = tdf['dom'] <= 5
    tdf['is_month_end'] = tdf['dom'] >= 25
    print(f"\n  月初(1-5日): N={len(tdf[tdf['is_month_start']])} WR={(tdf[tdf['is_month_start']]['r']>0).mean()*100:.1f}%")
    print(f"  月中(6-24日): N={len(tdf[~tdf['is_month_start'] & ~tdf['is_month_end']])} "
          f"WR={(tdf[~tdf['is_month_start'] & ~tdf['is_month_end']]['r']>0).mean()*100:.1f}%")
    print(f"  月末(25-31日): N={len(tdf[tdf['is_month_end']])} WR={(tdf[tdf['is_month_end']]['r']>0).mean()*100:.1f}%")

    # 月份
    print(f"\n  月份:")
    for m in range(1, 13):
        sub = tdf[tdf['month'] == m]
        if len(sub) == 0: continue
        print(f"    {m:2d}月: N={len(sub):4d} WR={(sub['r']>0).mean()*100:.1f}% Avg={sub['r'].mean():+.3f}%")


def analyze_score_quality(trades):
    """5. 信号质量深化"""
    print(f"\n{'='*70}")
    print("5. 信号得分质量分析")
    print(f"{'='*70}")

    tdf = pd.DataFrame(trades)

    # 按得分区间
    print(f"\n  得分区间:")
    bins = [(7, 9), (9, 11), (11, 13), (13, 15), (15, 20), (20, 100)]
    for lo, hi in bins:
        sub = tdf[(tdf['sc'] >= lo) & (tdf['sc'] < hi)]
        if len(sub) == 0: continue
        wr = (sub['r'] > 0).mean() * 100
        avg = sub['r'].mean()
        print(f"    score {lo:2d}-{hi:2d}: N={len(sub):4d} WR={wr:.1f}% Avg={avg:+.3f}% PF={calc_pf(sub)}")

    # 按exit reason
    print(f"\n  平仓原因:")
    for reason in tdf['reason'].unique():
        sub = tdf[tdf['reason'] == reason]
        wr = (sub['r'] > 0).mean() * 100
        avg = sub['r'].mean()
        print(f"    {reason:4s}: N={len(sub):4d} WR={wr:.1f}% Avg={avg:+.3f}%")

    # 做多/空 × 得分
    print(f"\n  做多得分:")
    for lo, hi in bins:
        sub = tdf[(tdf['dir'] == 'long') & (tdf['sc'] >= lo) & (tdf['sc'] < hi)]
        if len(sub) == 0: continue
        print(f"    {lo:2d}-{hi:2d}: N={len(sub):4d} WR={(sub['r']>0).mean()*100:.1f}% Avg={sub['r'].mean():+.3f}%")

    print(f"\n  做空得分:")
    for lo, hi in bins:
        sub = tdf[(tdf['dir'] == 'short') & (tdf['sc'] >= lo) & (tdf['sc'] < hi)]
        if len(sub) == 0: continue
        print(f"    {lo:2d}-{hi:2d}: N={len(sub):4d} WR={(sub['r']>0).mean()*100:.1f}% Avg={sub['r'].mean():+.3f}%")


def calc_pf(tdf):
    if len(tdf) == 0: return 0
    wins = tdf[tdf['r'] > 0]['r']
    losses = tdf[tdf['r'] <= 0]['r']
    if len(losses) == 0 or losses.sum() == 0: return 999
    return abs(wins.sum() / losses.sum())


def analyze_robustness(signal_data):
    """6. 参数稳健性测试"""
    print(f"\n{'='*70}")
    print("6. 参数稳健性测试 — 不同参数组合稳定性")
    print(f"{'='*70}")

    configs = [
        # (max_pos, lev, min_sc, hold, sl, tp, desc)
        (7, 5, 7, 1, -2, 3, "V74最佳"),
        (7, 5, 7, 1, -1.5, 2, "紧止损"),
        (7, 5, 7, 1, -3, 5, "宽止损"),
        (7, 5, 7, 2, -2, 3, "持仓2天"),
        (5, 5, 7, 1, -2, 3, "5仓位"),
        (7, 3, 7, 1, -2, 3, "低杠杆3x"),
        (7, 5, 8, 1, -2, 3, "min=8"),
        (7, 5, 9, 1, -2, 3, "min=9"),
        (7, 5, 7, 1, None, None, "无止损"),
        (7, 5, 7, 1, -2, None, "只止损"),
    ]

    print(f"\n  {'配置':>16} | {'测试期':>30} | {'训练期':>16}")
    print(f"  {'':>16} | {'N':>5} {'WR':>6} {'MDD':>7} {'Sharpe':>7} | {'WR':>6} {'MDD':>7}")
    print("-" * 90)

    for mp, lev, msc, hd, sl, tp, desc in configs:
        # 测试期
        eq, tr = run_bt(signal_data, TEST_START, TEST_END, max_pos=mp, lev=lev,
                         min_sc=msc, hold=hd, sl=sl, tp=tp)
        eq_df = pd.DataFrame(eq)
        if len(eq_df) == 0 or eq_df['capital'].iloc[-1] <= INITIAL_CAPITAL:
            print(f"  {desc:>16} | 测试期亏损"); continue

        tdf = pd.DataFrame(tr)
        wr = (tdf['r'] > 0).mean() * 100
        mdd = ((eq_df['capital'] - eq_df['capital'].cummax()) / eq_df['capital'].cummax() * 100).min()
        dr = eq_df['capital'].pct_change().dropna()
        sh = dr.mean() / dr.std() * (252**0.5) if dr.std() > 0 else 0

        # 训练期
        eq2, tr2 = run_bt(signal_data, '2015-01-01', '2021-12-31', max_pos=mp, lev=lev,
                           min_sc=msc, hold=hd, sl=sl, tp=tp)
        eq2_df = pd.DataFrame(eq2)
        tdf2 = pd.DataFrame(tr2) if tr2 else pd.DataFrame()
        wr2 = (tdf2['r'] > 0).mean() * 100 if len(tdf2) > 0 else 0
        mdd2 = ((eq2_df['capital'] - eq2_df['capital'].cummax()) / eq2_df['capital'].cummax() * 100).min() if len(eq2_df) > 0 else 0

        print(f"  {desc:>16} | {len(tr):5d} {wr:5.1f}% {mdd:>+6.1f}% {sh:>6.2f}  | {wr2:5.1f}% {mdd2:>+6.1f}%")


def analyze_concentration_risk(trades):
    """7. 集中度风险分析"""
    print(f"\n{'='*70}")
    print("7. 集中度风险分析")
    print(f"{'='*70}")

    tdf = pd.DataFrame(trades)
    tdf['xd'] = pd.to_datetime(tdf['xd'])
    tdf['year'] = tdf['xd'].dt.year

    # 每年Top品种贡献
    for yr in sorted(tdf['year'].unique()):
        sub = tdf[tdf['year'] == yr]
        total_pnl = sub['pnl'].sum()
        by_sym = sub.groupby('sym')['pnl'].sum().sort_values(ascending=False)
        top3_pct = by_sym.head(3).sum() / total_pnl * 100 if total_pnl != 0 else 0
        top5_pct = by_sym.head(5).sum() / total_pnl * 100 if total_pnl != 0 else 0
        n_sym = (by_sym > 0).sum()
        print(f"  {yr}: {n_sym}盈利品种, Top3贡献{top3_pct:.1f}%, Top5贡献{top5_pct:.1f}%")

    # 同日多仓相关性
    by_date = tdf.groupby('xd').agg({
        'r': ['count', 'mean', lambda x: (x > 0).mean()],
        'pnl': 'sum'
    }).reset_index()
    by_date.columns = ['date', 'n_trades', 'avg_r', 'day_wr', 'day_pnl']

    # 日收益分布
    print(f"\n  日收益分布:")
    for n in sorted(by_date['n_trades'].unique()):
        sub = by_date[by_date['n_trades'] == n]
        print(f"    {n}仓/日: {len(sub)}天 WR={sub['day_wr'].mean()*100:.1f}% AvgPnL={sub['day_pnl'].mean():+.0f}")


def analyze_exit_optimization(signal_data):
    """8. 出场优化 — 动态止盈止损"""
    print(f"\n{'='*70}")
    print("8. 出场优化 — 动态止盈止损测试")
    print(f"{'='*70}")

    # 不同SL/TP组合
    print(f"\n  SL/TP扫描 (mp=7, lev=5, min=7, H=1d):")
    print(f"  {'SL':>5} {'TP':>5} | {'N':>5} {'WR':>6} {'Avg':>8} {'MDD':>7} {'Sharpe':>7}")
    print("-" * 55)

    best_sh = 0
    best_config = None
    for sl in [-1.0, -1.5, -2.0, -2.5, -3.0]:
        for tp in [1.5, 2.0, 3.0, 4.0, 5.0, None]:
            eq, tr = run_bt(signal_data, TEST_START, TEST_END, max_pos=7, lev=5,
                             min_sc=7, hold=1, sl=sl, tp=tp)
            eq_df = pd.DataFrame(eq)
            if len(eq_df) == 0 or eq_df['capital'].iloc[-1] <= INITIAL_CAPITAL:
                continue
            tdf = pd.DataFrame(tr)
            wr = (tdf['r'] > 0).mean() * 100
            avg = tdf['r'].mean()
            mdd = ((eq_df['capital'] - eq_df['capital'].cummax()) / eq_df['capital'].cummax() * 100).min()
            dr = eq_df['capital'].pct_change().dropna()
            sh = dr.mean() / dr.std() * (252**0.5) if dr.std() > 0 else 0

            tp_s = f"{tp:.1f}" if tp else "None"
            print(f"  {sl:>+4.1f} {tp_s:>5} | {len(tr):5d} {wr:5.1f}% {avg:>+7.3f}% {mdd:>+6.1f}% {sh:>6.2f}")

            if wr >= 50 and mdd >= -30 and sh > best_sh:
                best_sh = sh
                best_config = (sl, tp)

    if best_config:
        print(f"\n  最佳SL/TP (WR≥50%, MDD≤30%): SL={best_config[0]}, TP={best_config[1]}, Sharpe={best_sh:.2f}")


def analyze_yearly_stability(signal_data):
    """9. 逐年+逐季稳定性"""
    print(f"\n{'='*70}")
    print("9. 逐季稳定性分析")
    print(f"{'='*70}")

    eq, tr = run_bt(signal_data, '2015-01-01', '2025-12-31', max_pos=7, lev=5, min_sc=7, hold=1, sl=-2, tp=3)
    tdf = pd.DataFrame(tr)
    tdf['xd'] = pd.to_datetime(tdf['xd'])
    tdf['yq'] = tdf['xd'].dt.to_period('Q')

    print(f"\n  {'季度':>8} {'N':>5} {'WR':>6} {'Avg':>8} {'PnL':>12}")
    print("-" * 50)
    for q in sorted(tdf['yq'].unique()):
        sub = tdf[tdf['yq'] == q]
        wr = (sub['r'] > 0).mean() * 100
        avg = sub['r'].mean()
        pnl = sub['pnl'].sum()
        print(f"  {str(q):>8} {len(sub):5d} {wr:5.1f}% {avg:>+7.3f}% {pnl:>11.0f}")

    # 最差季度
    quarterly = tdf.groupby('yq').agg({
        'r': ['count', 'mean', lambda x: (x > 0).mean()],
        'pnl': 'sum'
    }).reset_index()
    quarterly.columns = ['q', 'n', 'avg', 'wr', 'pnl']
    worst = quarterly.nsmallest(5, 'pnl')
    print(f"\n  最差5个季度:")
    for _, row in worst.iterrows():
        print(f"    {row['q']}: N={row['n']} WR={row['wr']*100:.1f}% PnL={row['pnl']:,.0f}")


def main():
    print("V75: 品种分析 + 策略稳健性测试")
    print("基于V74最佳配置: mp=7, lev=5, H=1d, SL=-2, TP=3")
    print("="*70)

    print("\n加载数据...")
    all_data = load_data()
    print("计算信号...")
    sd = compute_signals(all_data)

    # 运行最佳配置回测
    print("\n运行回测...")
    eq, trades = run_bt(sd, TEST_START, TEST_END, max_pos=7, lev=5, min_sc=7, hold=1, sl=-2, tp=3)
    eq_df = pd.DataFrame(eq)
    tdf = pd.DataFrame(trades)
    wr = (tdf['r'] > 0).mean() * 100
    mdd = ((eq_df['capital'] - eq_df['capital'].cummax()) / eq_df['capital'].cummax() * 100).min()
    dr = eq_df['capital'].pct_change().dropna()
    sh = dr.mean() / dr.std() * (252**0.5) if dr.std() > 0 else 0
    print(f"  基准: N={len(trades)} WR={wr:.1f}% MDD={mdd:.1f}% Sharpe={sh:.2f}")

    # 各项分析
    analyze_per_commodity(sd, trades)
    analyze_direction(sd, trades)
    analyze_score_quality(trades)
    analyze_time_patterns(trades)
    analyze_rolling_stability(sd, TEST_START, TEST_END)
    analyze_concentration_risk(trades)
    analyze_robustness(sd)
    analyze_exit_optimization(sd)
    analyze_yearly_stability(sd)

    print(f"\n\n{'='*70}")
    print("总结")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
