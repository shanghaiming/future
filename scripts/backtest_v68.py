#!/usr/bin/env python3
"""
V68 — 全面信号实证分析: 用数据找出真正有预测力的信号

不预设结论, 直接测试所有可能的日内/隔夜/OI/成交量模式

信号分类:
  A. 隔夜跳空 (open vs prev close)
  B. 日内形态 (close在range中的位置)
  C. 波动率收缩/扩张
  D. OI变化+价格方向
  E. 成交量异常
  F. 连续涨跌
  G. 横截面排名 (cross-sectional)
  H. 组合信号

关键: 用WALK-FORWARD验证, 训练期找信号, 测试期验证
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from contract_specs import get_spec

TD = 252


def load(data_dir):
    raw = {}
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith('.csv'):
            continue
        sym = f.replace('.csv', '')
        df = pd.read_csv(os.path.join(data_dir, f))
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        df = df.sort_values('trade_date').reset_index(drop=True)
        if len(df) < 500:
            continue

        df['ret'] = df['close'].pct_change()
        df['ret_next1'] = df['close'].shift(-1) / df['close'] - 1
        df['ret_next3'] = df['close'].shift(-3) / df['close'] - 1
        df['ret_next5'] = df['close'].shift(-5) / df['close'] - 1
        df['ret_next10'] = df['close'].shift(-10) / df['close'] - 1

        # A: 隔夜跳空
        df['gap'] = df['open'] / df['close'].shift(1) - 1  # 当日open vs 昨日close

        # B: 日内形态
        df['range'] = df['high'] - df['low']
        df['range'] = df['range'].replace(0, np.nan)
        df['close_pos'] = (df['close'] - df['low']) / df['range']  # 0=最低 1=最高
        df['body'] = abs(df['close'] - df['open']) / df['range']
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['range']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['range']
        df['is_bull'] = (df['close'] > df['open']).astype(int)

        # 次日收益(从open入场)
        df['intraday_ret'] = df['close'] / df['open'] - 1  # 当日open到close
        df['next_intraday'] = df['ret_next1']  # 简化: 次日close/当日close-1

        # C: 波动率
        df['atr5'] = df['range'].rolling(5).mean()
        df['atr20'] = df['range'].rolling(20).mean()
        df['atr_ratio'] = df['range'] / df['atr20'].replace(0, np.nan)  # 当日range vs 20日均值
        df['hv20'] = df['ret'].rolling(20).std() * np.sqrt(TD)
        df['hv_pct'] = df['hv20'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 10 else .5, raw=False)

        # D: OI
        df['oi_chg1'] = df['oi'].pct_change(1)
        df['oi_chg5'] = df['oi'].pct_change(5)
        df['oi_ma10'] = df['oi'].rolling(10).mean()
        df['oi_ratio'] = df['oi'] / df['oi_ma10'].replace(0, np.nan)

        # E: 成交量
        df['vol_ma20'] = df['vol'].rolling(20).mean()
        df['vol_ratio'] = df['vol'] / df['vol_ma20'].replace(0, np.nan)
        df['vol_chg1'] = df['vol'].pct_change(1)

        # F: 连续涨跌
        down = (df['ret'] < 0).astype(int)
        up = (df['ret'] > 0).astype(int)
        cons_d, cons_u = [], []
        cd, cu = 0, 0
        for v in zip(down, up):
            cd = cd + 1 if v[0] else 0
            cu = cu + 1 if v[1] else 0
            cons_d.append(cd)
            cons_u.append(cu)
        df['cons_down'] = cons_d
        df['cons_up'] = cons_u

        # 动量
        for lag in [1, 3, 5, 10, 20, 60]:
            df[f'm{lag}'] = df['close'].pct_change(lag)

        # 均线
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()

        # RSI
        d = df['close'].diff()
        g = d.where(d > 0, 0).rolling(14).mean()
        l = (-d.where(d < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + g / l))

        # 20日高低
        df['high20'] = df['close'].rolling(20).max()
        df['low20'] = df['close'].rolling(20).min()
        df['pct_rank_20'] = (df['close'] - df['low20']) / (df['high20'] - df['low20']).replace(0, np.nan)

        df = df.dropna(subset=['hv20', 'rsi', 'ma20'])
        if len(df) < 200:
            continue

        raw[sym] = df

    return raw


def analyze_signal(df, signal_col, signal_fn, period_name, hold_days_list=[1, 3, 5, 10],
                   train_end='2021-12-31', test_start='2022-01-01'):
    """分析单个信号的训练期和测试期表现"""
    results = []

    for hold in hold_days_list:
        fwd_col = f'ret_next{hold}'
        mask_train = df['trade_date'] <= train_end
        mask_test = df['trade_date'] >= test_start

        for period_mask, period in [(mask_train, 'train'), (mask_test, 'test'),
                                     (pd.Series(True, index=df.index), 'all')]:
            sub = df[period_mask].copy()
            if signal_fn:
                sig_mask = signal_fn(sub)
            else:
                sig_mask = sub[signal_col] if isinstance(signal_col, str) else signal_col

            fwd = sub.loc[sig_mask, fwd_col].dropna()
            if len(fwd) < 30:
                continue

            wr = (fwd > 0).mean()
            avg = fwd.mean()
            std = fwd.std()
            t = avg / (std / np.sqrt(len(fwd))) if std > 0 else 0

            results.append({
                'signal': period_name,
                'hold': hold,
                'period': period,
                'n': len(fwd),
                'wr': wr,
                'avg': avg,
                'std': std,
                't': t,
            })

    return results


def main():
    dd = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    print("加载数据..."); t0 = time.time()
    raw = load(dd)
    print(f"  {len(raw)}品种, {time.time()-t0:.1f}s")

    all_results = []
    train_end = '2021-12-31'
    test_start = '2022-01-01'

    # 合并所有品种数据
    all_dfs = []
    for sym, df in raw.items():
        df = df.copy()
        df['sym'] = sym
        all_dfs.append(df)
    big = pd.concat(all_dfs, ignore_index=True)
    print(f"  总行数: {len(big):,}")

    print(f"\n训练期: ~{train_end}")
    print(f"测试期: {test_start}~")

    def print_top_results(results, title, top_n=25):
        if not results:
            return
        df_r = pd.DataFrame(results)
        # 只看测试期, hold=5
        test_r = df_r[(df_r['period'] == 'test') & (df_r['hold'] == 5)]
        if len(test_r) == 0:
            test_r = df_r[(df_r['period'] == 'test')]
        test_r = test_r.sort_values('t', ascending=False)

        print(f"\n=== {title} (测试期, hold=5d) ===")
        print(f"{'Signal':>45} {'N':>6} {'WR':>6} {'Avg':>8} {'t':>7}")
        print("-" * 80)
        for _, r in test_r.head(top_n).iterrows():
            print(f"{r['signal']:>45} {r['n']:>6} {r['wr']:>6.1%} "
                  f"{r['avg']:>8.3%} {r['t']:>7.2f}")

    # =================================================================
    # A: 隔夜跳空
    # =================================================================
    print("\n" + "=" * 80)
    print("A. 隔夜跳空信号")

    gap_signals = [
        ('gap_up_1%', lambda d: d['gap'] > 0.01),
        ('gap_up_2%', lambda d: d['gap'] > 0.02),
        ('gap_down_1%', lambda d: d['gap'] < -0.01),
        ('gap_down_2%', lambda d: d['gap'] < -0.02),
        ('gap_up_then_ret_neg', lambda d: (d['gap'] > 0.01) & (d['ret'] < 0)),  # 跳空高开但收阴
        ('gap_down_then_ret_pos', lambda d: (d['gap'] < -0.01) & (d['ret'] > 0)),  # 跳空低开但收阳
        ('gap_up_intraday_ret', lambda d: d['gap'] > 0.01),  # 跳空高开后日内收益
        ('gap_down_intraday_ret', lambda d: d['gap'] < -0.01),  # 跳空低开后日内收益
    ]

    # 分析跳空对次日(从open入场)收益的影响
    for name, fn in gap_signals:
        results = analyze_signal(big, None, fn, name, [1, 3, 5], train_end, test_start)
        all_results.extend(results)

    # 专门分析intraday收益
    for name, condition in [('gap_up>1%_intraday', big['gap'] > 0.01),
                             ('gap_down<-1%_intraday', big['gap'] < -0.01),
                             ('gap_up>2%_intraday', big['gap'] > 0.02),
                             ('gap_down<-2%_intraday', big['gap'] < -0.02)]:
        sub = big[condition].copy()
        for period_mask, period in [(sub['trade_date'] <= train_end, 'train'),
                                     (sub['trade_date'] >= test_start, 'test')]:
            s = sub[period_mask]['intraday_ret'].dropna()
            if len(s) > 30:
                all_results.append({
                    'signal': f'{name}',
                    'hold': 0,
                    'period': period,
                    'n': len(s),
                    'wr': (s > 0).mean(),
                    'avg': s.mean(),
                    'std': s.std(),
                    't': s.mean() / (s.std() / np.sqrt(len(s))) if s.std() > 0 else 0,
                })

    print_top_results(all_results, "A. 隔夜跳空")

    # =================================================================
    # B: 日内形态
    # =================================================================
    print("\n" + "=" * 80)
    print("B. 日内形态信号")

    intraday_signals = [
        ('close_at_high_top10%', lambda d: d['close_pos'] > 0.9),
        ('close_at_low_bot10%', lambda d: d['close_pos'] < 0.1),
        ('close_at_high_top5%', lambda d: d['close_pos'] > 0.95),
        ('close_at_low_bot5%', lambda d: d['close_pos'] < 0.05),
        ('big_bull_body>70%', lambda d: (d['is_bull'] == 1) & (d['body'] > 0.7)),
        ('big_bear_body>70%', lambda d: (d['is_bull'] == 0) & (d['body'] > 0.7)),
        ('long_lower_shadow', lambda d: d['lower_shadow'] > 0.4),  # 锤子线
        ('long_upper_shadow', lambda d: d['upper_shadow'] > 0.4),  # 射击之星
        ('doji_body<10%', lambda d: d['body'] < 0.1),
        ('bull_engulf', lambda d: (d['is_bull'] == 1) & (d['body'] > 0.6) &
         (d['ret'].shift(1) < 0) & (d['close'].shift(1).fillna(0) < d['open'])),
    ]

    for name, fn in intraday_signals:
        results = analyze_signal(big, None, fn, name, [1, 3, 5, 10], train_end, test_start)
        all_results.extend(results)

    print_top_results(all_results, "B. 日内形态")

    # =================================================================
    # C: 波动率
    # =================================================================
    print("\n" + "=" * 80)
    print("C. 波动率信号")

    vol_signals = [
        ('narrow_range_atr<0.5', lambda d: d['atr_ratio'] < 0.5),
        ('narrow_range_atr<0.3', lambda d: d['atr_ratio'] < 0.3),
        ('wide_range_atr>2.0', lambda d: d['atr_ratio'] > 2.0),
        ('wide_range_atr>3.0', lambda d: d['atr_ratio'] > 3.0),
        ('hv_low<20pct', lambda d: d['hv_pct'] < 0.2),
        ('hv_low<30pct', lambda d: d['hv_pct'] < 0.3),
        ('hv_high>80pct', lambda d: d['hv_pct'] > 0.8),
        ('narrow_then_bull', lambda d: (d['atr_ratio'] < 0.5) & (d['is_bull'] == 1)),
        ('narrow_then_bear', lambda d: (d['atr_ratio'] < 0.5) & (d['is_bull'] == 0)),
    ]

    for name, fn in vol_signals:
        results = analyze_signal(big, None, fn, name, [1, 3, 5, 10], train_end, test_start)
        all_results.extend(results)

    print_top_results(all_results, "C. 波动率")

    # =================================================================
    # D: OI信号
    # =================================================================
    print("\n" + "=" * 80)
    print("D. OI信号")

    oi_signals = [
        ('oi_up+price_up', lambda d: (d['oi_chg1'] > 0.02) & (d['ret'] > 0)),
        ('oi_up+price_down', lambda d: (d['oi_chg1'] > 0.02) & (d['ret'] < 0)),
        ('oi_down+price_up', lambda d: (d['oi_chg1'] < -0.02) & (d['ret'] > 0)),
        ('oi_down+price_down', lambda d: (d['oi_chg1'] < -0.02) & (d['ret'] < 0)),
        ('oi_surge>10%', lambda d: d['oi_chg1'] > 0.10),
        ('oi_surge>20%', lambda d: d['oi_chg1'] > 0.20),
        ('oi_5d_up>10%', lambda d: d['oi_chg5'] > 0.10),
        ('oi_above_ma10>1.2', lambda d: d['oi_ratio'] > 1.2),
        ('oi_above_ma10>1.5', lambda d: d['oi_ratio'] > 1.5),
        ('oi_up5d+price_up5d', lambda d: (d['oi_chg5'] > 0.05) & (d['m5'] > 0)),
        ('oi_up5d+price_down5d', lambda d: (d['oi_chg5'] > 0.05) & (d['m5'] < 0)),
    ]

    for name, fn in oi_signals:
        results = analyze_signal(big, None, fn, name, [1, 3, 5, 10], train_end, test_start)
        all_results.extend(results)

    print_top_results(all_results, "D. OI")

    # =================================================================
    # E: 成交量
    # =================================================================
    print("\n" + "=" * 80)
    print("E. 成交量信号")

    vol_price_signals = [
        ('vol_surge>2x+up', lambda d: (d['vol_ratio'] > 2.0) & (d['ret'] > 0)),
        ('vol_surge>2x+down', lambda d: (d['vol_ratio'] > 2.0) & (d['ret'] < 0)),
        ('vol_surge>3x+up', lambda d: (d['vol_ratio'] > 3.0) & (d['ret'] > 0)),
        ('vol_surge>3x+down', lambda d: (d['vol_ratio'] > 3.0) & (d['ret'] < 0)),
        ('vol_dry<0.5x', lambda d: d['vol_ratio'] < 0.5),
        ('vol_up+price_new_high20', lambda d: (d['vol_ratio'] > 1.5) & (d['close'] >= d['high20'])),
        ('vol_down+price_new_low20', lambda d: (d['vol_ratio'] < 0.5) & (d['close'] <= d['low20'])),
    ]

    for name, fn in vol_price_signals:
        results = analyze_signal(big, None, fn, name, [1, 3, 5, 10], train_end, test_start)
        all_results.extend(results)

    print_top_results(all_results, "E. 成交量")

    # =================================================================
    # F: 连续涨跌
    # =================================================================
    print("\n" + "=" * 80)
    print("F. 连续涨跌信号")

    cons_signals = [
        ('cons_down_3', lambda d: d['cons_down'] >= 3),
        ('cons_down_4', lambda d: d['cons_down'] >= 4),
        ('cons_down_5', lambda d: d['cons_down'] >= 5),
        ('cons_up_3', lambda d: d['cons_up'] >= 3),
        ('cons_up_4', lambda d: d['cons_up'] >= 4),
        ('cons_up_5', lambda d: d['cons_up'] >= 5),
        ('cons_down_3+gap_down', lambda d: (d['cons_down'] >= 3) & (d['gap'] < -0.005)),
        ('cons_down_3+vol_surge', lambda d: (d['cons_down'] >= 3) & (d['vol_ratio'] > 1.5)),
        ('cons_up_3+gap_up', lambda d: (d['cons_up'] >= 3) & (d['gap'] > 0.005)),
    ]

    for name, fn in cons_signals:
        results = analyze_signal(big, None, fn, name, [1, 3, 5, 10], train_end, test_start)
        all_results.extend(results)

    print_top_results(all_results, "F. 连续涨跌")

    # =================================================================
    # G: RSI/均线/动量
    # =================================================================
    print("\n" + "=" * 80)
    print("G. RSI/均线/动量信号")

    mom_signals = [
        ('RSI<20', lambda d: d['rsi'] < 20),
        ('RSI<25', lambda d: d['rsi'] < 25),
        ('RSI<30', lambda d: d['rsi'] < 30),
        ('RSI>80', lambda d: d['rsi'] > 80),
        ('RSI>75', lambda d: d['rsi'] > 75),
        ('above_ma5', lambda d: d['close'] > d['ma5']),
        ('below_ma5', lambda d: d['close'] < d['ma5']),
        ('ma5_cross_ma20_up', lambda d: (d['ma5'] > d['ma20']) & (d['ma5'].shift(1) <= d['ma20'].shift(1))),
        ('ma5_cross_ma20_dn', lambda d: (d['ma5'] < d['ma20']) & (d['ma5'].shift(1) >= d['ma20'].shift(1))),
        ('mom5>5%', lambda d: d['m5'] > 0.05),
        ('mom5<-5%', lambda d: d['m5'] < -0.05),
        ('mom20>10%', lambda d: d['m20'] > 0.10),
        ('mom20<-10%', lambda d: d['m20'] < -0.10),
        ('pct_rank<10%', lambda d: d['pct_rank_20'] < 0.1),
        ('pct_rank>90%', lambda d: d['pct_rank_20'] > 0.9),
    ]

    for name, fn in mom_signals:
        results = analyze_signal(big, None, fn, name, [1, 3, 5, 10], train_end, test_start)
        all_results.extend(results)

    print_top_results(all_results, "G. RSI/均线/动量")

    # =================================================================
    # H: 组合信号 (最有希望的)
    # =================================================================
    print("\n" + "=" * 80)
    print("H. 组合信号")

    combo_signals = [
        ('RSI<30+narrow_range', lambda d: (d['rsi'] < 30) & (d['atr_ratio'] < 0.7)),
        ('RSI<30+oi_up', lambda d: (d['rsi'] < 30) & (d['oi_chg1'] > 0.01)),
        ('RSI<30+vol_surge', lambda d: (d['rsi'] < 30) & (d['vol_ratio'] > 1.5)),
        ('RSI<30+gap_down', lambda d: (d['rsi'] < 30) & (d['gap'] < -0.005)),
        ('cons_down3+gap_down+narrow', lambda d: (d['cons_down'] >= 3) & (d['gap'] < -0.005) & (d['atr_ratio'] < 0.8)),
        ('gap_down2%+vol_surge', lambda d: (d['gap'] < -0.02) & (d['vol_ratio'] > 2.0)),
        ('gap_down2%+oi_up', lambda d: (d['gap'] < -0.02) & (d['oi_chg1'] > 0.02)),
        ('gap_down2%+RSI<30', lambda d: (d['gap'] < -0.02) & (d['rsi'] < 30)),
        ('gap_up2%+RSI>70', lambda d: (d['gap'] > 0.02) & (d['rsi'] > 70)),
        ('gap_up2%+vol_surge', lambda d: (d['gap'] > 0.02) & (d['vol_ratio'] > 2.0)),
        ('close_bot10%+vol_surge', lambda d: (d['close_pos'] < 0.1) & (d['vol_ratio'] > 1.5)),
        ('close_top10%+vol_surge', lambda d: (d['close_pos'] > 0.9) & (d['vol_ratio'] > 1.5)),
        ('long_lower_shadow+vol', lambda d: (d['lower_shadow'] > 0.4) & (d['vol_ratio'] > 1.5)),
        ('oi_up5d+mom_up5d+hv_low', lambda d: (d['oi_chg5'] > 0.05) & (d['m5'] > 0) & (d['hv_pct'] < 0.3)),
        ('gap_down+close_at_high', lambda d: (d['gap'] < -0.01) & (d['close_pos'] > 0.8)),  # 低开高走!
        ('gap_up+close_at_low', lambda d: (d['gap'] > 0.01) & (d['close_pos'] < 0.2)),  # 高开低走!
    ]

    for name, fn in combo_signals:
        results = analyze_signal(big, None, fn, name, [1, 3, 5, 10], train_end, test_start)
        all_results.extend(results)

    print_top_results(all_results, "H. 组合信号")

    # =================================================================
    # 汇总: 测试期+训练期一致性
    # =================================================================
    print("\n" + "=" * 80)
    print("\n=== 汇总: 训练期和测试期都有正t值且WR>50%的信号 ===")
    print("(hold=5天, 训练和测试都正)")

    df_all = pd.DataFrame(all_results)
    # 筛选hold=5
    df5 = df_all[df_all['hold'] == 5]

    # 找训练期和测试期都有的信号
    train_sig = df5[df5['period'] == 'train'][['signal', 'wr', 'avg', 't', 'n']].rename(
        columns={'wr': 'train_wr', 'avg': 'train_avg', 't': 'train_t', 'n': 'train_n'})
    test_sig = df5[df5['period'] == 'test'][['signal', 'wr', 'avg', 't', 'n']].rename(
        columns={'wr': 'test_wr', 'avg': 'test_avg', 't': 'test_t', 'n': 'test_n'})

    merged = train_sig.merge(test_sig, on='signal', how='inner')

    # 两个期间都正t, 测试期WR>50%
    good = merged[(merged['train_t'] > 1.5) & (merged['test_t'] > 1.5) & (merged['test_wr'] > 0.50)]
    good = good.sort_values('test_t', ascending=False)

    print(f"\n{'Signal':>45} {'TrN':>5} {'TrWR':>6} {'TrAvg':>8} {'TrT':>6} "
          f"{'TeN':>5} {'TeWR':>6} {'TeAvg':>8} {'TeT':>6}")
    print("-" * 115)
    for _, r in good.head(40).iterrows():
        print(f"{r['signal']:>45} {r['train_n']:>5} {r['train_wr']:>6.1%} {r['train_avg']:>8.3%} {r['train_t']:>6.2f} "
              f"{r['test_n']:>5} {r['test_wr']:>6.1%} {r['test_avg']:>8.3%} {r['test_t']:>6.2f}")

    # 也看看intraday的 (hold=0)
    print("\n=== 日内(intraday)信号 ===")
    df0 = df_all[df_all['hold'] == 0]
    if len(df0) > 0:
        for _, r in df0.sort_values('t', ascending=False).head(20).iterrows():
            print(f"  {r['signal']:>40}  {r['period']:>5}  N={r['n']:>5}  "
                  f"WR={r['wr']:.1%}  avg={r['avg']:.3%}  t={r['t']:.2f}")

    # 保存
    od = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(od, exist_ok=True)
    df_all.to_csv(os.path.join(od, 'signal_analysis_v68.csv'), index=False)
    print(f"\n→ backtest_results/signal_analysis_v68.csv")


if __name__ == '__main__':
    main()
