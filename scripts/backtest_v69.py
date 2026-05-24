#!/usr/bin/env python3
"""
V69: 基于V68实证信号的期货策略
核心发现:
  1. gap_down (隔夜跳空低开) = 最强信号, 62.5% WR, +1.14% avg
  2. oi_up+price_down (增仓下跌) = 次强信号, 60.2% WR, +0.75% avg
  3. gap_down2%+oi_up = 组合信号, 66.4% WR, +1.70% avg
  4. below_ma5 = 均线回归, 54.5% WR, +0.38% avg
  5. mom5<-5% = 动量极端, 57.7% WR, +1.47% avg
  6. vol_surge>2x+down = 放量下跌, 56.6% WR, +0.87% avg

策略逻辑:
  - 每个品种计算信号得分(0-6分)
  - 按得分排序, 取top 3品种建仓
  - 持有5天后平仓
  - 复利计算
  - Walk-forward: 训练<=2021, 测试>=2022
"""

import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ─── 配置 ───
DATA_DIR = 'data/futures_weighted'
INITIAL_CAPITAL = 500_000
MAX_POSITIONS = 3
HOLD_DAYS = 5
LEVERAGE = 5  # 名义杠杆倍数
TRAIN_END = '2021-12-31'
TEST_START = '2022-01-01'
CONTRACT_SPECS = 'scripts/contract_specs.py'

def load_data():
    """加载所有品种数据"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("cs", CONTRACT_SPECS)
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)

    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    all_data = {}
    skipped = 0
    for f in files:
        sym = os.path.basename(f).replace('.csv', '')
        try:
            mult, margin, tick, tick_val = cs.get_spec(sym)
        except:
            skipped += 1
            continue

        df = pd.read_csv(f)
        if len(df) < 100:
            skipped += 1
            continue

        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        df = df.sort_values('trade_date').reset_index(drop=True)

        if df['close'].isna().all() or (df['close'] == 0).any():
            skipped += 1
            continue

        all_data[sym] = {
            'df': df,
            'multiplier': mult,
            'margin_rate': margin,
        }

    print(f"  {len(all_data)}品种, 跳过{skipped}")
    return all_data


def compute_signals(df, hold_days=5):
    """计算所有V68验证过的信号"""
    d = df.copy()
    c = d['close'].values
    o = d['open'].values
    h = d['high'].values
    l = d['low'].values
    v = d['vol'].values
    oi = d['oi'].values
    n = len(d)

    # ─── 基础指标 ───
    # 前收
    prev_c = np.full(n, np.nan)
    prev_c[1:] = c[:-1]
    d['prev_close'] = prev_c

    # 隔夜gap
    gap = np.full(n, np.nan)
    gap[1:] = (o[1:] - prev_c[1:]) / prev_c[1:] * 100
    d['gap_pct'] = gap

    # MA5, MA20
    d['ma5'] = d['close'].rolling(5).mean()
    d['ma20'] = d['close'].rolling(20).mean()

    # 5日动量
    mom5 = np.full(n, np.nan)
    mom5[5:] = (c[5:] - c[:-5]) / c[:-5] * 100
    d['mom5'] = mom5

    # 20日动量
    mom20 = np.full(n, np.nan)
    mom20[20:] = (c[20:] - c[:-20]) / c[:-20] * 100
    d['mom20'] = mom20

    # 成交量MA
    d['vol_ma5'] = d['vol'].rolling(5).mean()

    # OI变化
    oi_ch = np.full(n, np.nan)
    oi_ch[1:] = (oi[1:] - oi[:-1]) / np.abs(oi[:-1]) * 100
    d['oi_chg'] = oi_ch

    # OI MA10
    d['oi_ma10'] = d['oi'].rolling(10).mean()

    # ATR
    tr = np.full(n, np.nan)
    tr[1:] = np.maximum(h[1:] - l[1:],
                         np.maximum(np.abs(h[1:] - c[:-1]),
                                    np.abs(l[1:] - c[:-1])))
    d['atr'] = pd.Series(tr).rolling(20).mean().values

    # 历史波动率
    ret = np.full(n, np.nan)
    ret[1:] = (c[1:] - c[:-1]) / c[:-1] * 100
    d['hv'] = pd.Series(ret).rolling(20).std().values

    # ─── 信号生成 ───
    # 每个信号是布尔列, 表示当天是否触发

    # 1. gap_down_1%: 隔夜低开>1%
    d['sig_gap_down1'] = d['gap_pct'] < -1.0

    # 2. gap_down_2%: 隔夜低开>2%
    d['sig_gap_down2'] = d['gap_pct'] < -2.0

    # 3. oi_up+price_down: OI增加且价格下跌(空头增仓)
    d['sig_oi_up_price_down'] = (d['oi_chg'] > 0) & (d['close'] < d['prev_close'])

    # 4. below_ma5: 收盘低于MA5
    d['sig_below_ma5'] = d['close'] < d['ma5']

    # 5. mom5<-5%: 5日动量小于-5%
    d['sig_mom5_extreme'] = d['mom5'] < -5.0

    # 6. vol_surge+down: 成交量>2x均量且价格下跌
    d['sig_vol_surge_down'] = (d['vol'] > d['vol_ma5'] * 2) & (d['close'] < d['prev_close'])

    # 7. gap_down+close_at_high: 低开后收在高点(反转信号)
    range_day = h - l
    range_day[range_day == 0] = 0.001
    close_position = (c - l) / range_day
    d['sig_gap_down_close_high'] = (d['gap_pct'] < -1.0) & (close_position > 0.8)

    # 8. oi_down+price_down: 减仓下跌(多头平仓接近尾声)
    d['sig_oi_down_price_down'] = (d['oi_chg'] < 0) & (d['close'] < d['prev_close'])

    # 9. pct_rank<10%: 收盘价处于20日低位
    roll_min = d['close'].rolling(20).min()
    roll_max = d['close'].rolling(20).max()
    roll_range = roll_max - roll_min
    roll_range[roll_range == 0] = 0.001
    d['pct_rank'] = (d['close'] - roll_min) / roll_range * 100
    d['sig_pct_rank_low'] = d['pct_rank'] < 10

    # 10. cons_down_3: 连跌3天
    price_down = c[1:] < c[:-1]
    cons_down = np.zeros(n, dtype=bool)
    for i in range(3, n):
        if price_down[i-1] and price_down[i-2] and price_down[i-3]:
            cons_down[i] = True
    d['sig_cons_down3'] = cons_down

    # ─── 信号得分 ───
    # 根据V68的t-stat加权
    # gap_down1: t=41.7 → weight 3
    # oi_up+price_down: t=41.1 → weight 3
    # gap_down2: t=38.1 → weight 3
    # below_ma5: t=32.9 → weight 2
    # mom5<-5%: t=16.2 → weight 2
    # vol_surge+down: t=7.7 → weight 1
    # gap_down+close_high: t=16.4 → weight 2
    # oi_down+price_down: t=19.5 → weight 2
    # pct_rank<10%: t=22.7 → weight 2
    # cons_down3: t=4.0 → weight 1

    score = np.zeros(n)
    score += d['sig_gap_down1'].astype(int) * 3
    score += d['sig_gap_down2'].astype(int) * 3  # 额外加分
    score += d['sig_oi_up_price_down'].astype(int) * 3
    score += d['sig_below_ma5'].astype(int) * 2
    score += d['sig_mom5_extreme'].astype(int) * 2
    score += d['sig_vol_surge_down'].astype(int) * 1
    score += d['sig_gap_down_close_high'].astype(int) * 2
    score += d['sig_oi_down_price_down'].astype(int) * 2
    score += d['sig_pct_rank_low'].astype(int) * 2
    score += d['sig_cons_down3'].astype(int) * 1
    d['signal_score'] = score

    # 入场价格: 当天开盘价(看到gap_down后立即在开盘买入)
    d['entry_price'] = d['open'].values

    # 5日后收盘价(出场价)
    exit_price = np.full(n, np.nan)
    exit_price[:-hold_days] = c[hold_days:]
    d['exit_price'] = exit_price

    # 入场日到出场日的收益率(close-to-close)
    fwd_ret = np.full(n, np.nan)
    valid = ~np.isnan(d['entry_price'].values) & ~np.isnan(d['exit_price'].values)
    fwd_ret[valid] = (d['exit_price'].values[valid] - d['entry_price'].values[valid]) / d['entry_price'].values[valid] * 100
    d['fwd_ret_5d'] = fwd_ret

    # 入场gap(用来验证edge来源)
    next_gap = np.full(n, np.nan)
    next_gap[:-1] = d['gap_pct'].values[1:]
    d['next_gap'] = next_gap

    return d


def run_backtest(all_data, start_date, end_date, max_pos, leverage, min_score=3, hold_days=5):
    """运行回测"""
    # 计算所有品种信号
    signal_data = {}
    for sym, info in all_data.items():
        df = compute_signals(info['df'], hold_days=hold_days)
        signal_data[sym] = df

    # 按日期收集所有信号
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    capital = INITIAL_CAPITAL
    equity_curve = []
    trades = []
    position_info = []  # 持仓中的交易

    for dt in date_range:
        # 1. 平仓: 检查到期持仓
        closed_pnl = 0
        still_open = []
        for trade in position_info:
            days_held = (dt - trade['entry_date']).days
            if days_held >= hold_days:
                # 平仓
                sym = trade['symbol']
                df = signal_data[sym]
                idx = df.index[df['trade_date'] == dt]
                if len(idx) > 0:
                    close_price = df.loc[idx[0], 'close']
                else:
                    # 如果当天没数据,用最后已知价
                    close_price = trade['exit_target']

                pnl = (close_price - trade['entry_price']) / trade['entry_price'] * 100
                trade_pnl = trade['notional'] * pnl / 100
                closed_pnl += trade_pnl
                trades.append({
                    'symbol': trade['symbol'],
                    'entry_date': trade['entry_date'],
                    'exit_date': dt,
                    'entry_price': trade['entry_price'],
                    'exit_price': close_price,
                    'pnl_pct': pnl,
                    'pnl_abs': trade_pnl,
                    'score': trade['score'],
                    'signals': trade['signals'],
                })
            else:
                still_open.append(trade)

        position_info = still_open
        capital += closed_pnl

        # 2. 开仓: 寻找信号
        n_open = max_pos - len(position_info)
        if n_open <= 0:
            equity_curve.append({'date': dt, 'capital': capital})
            continue

        # 收集当天所有品种的信号
        candidates = []
        for sym, df in signal_data.items():
            # 排除已持仓品种
            if any(p['symbol'] == sym for p in position_info):
                continue

            idx = df.index[df['trade_date'] == dt]
            if len(idx) == 0:
                continue
            row = df.loc[idx[0]]
            if row['signal_score'] < min_score:
                continue
            if np.isnan(row['entry_price']) or np.isnan(row['exit_price']):
                continue

            # 记录触发的信号
            sigs = []
            if row.get('sig_gap_down1', False): sigs.append('gap_d1')
            if row.get('sig_gap_down2', False): sigs.append('gap_d2')
            if row.get('sig_oi_up_price_down', False): sigs.append('oi_up+dn')
            if row.get('sig_below_ma5', False): sigs.append('blw_ma5')
            if row.get('sig_mom5_extreme', False): sigs.append('mom5<-5')
            if row.get('sig_vol_surge_down', False): sigs.append('vol_srg_dn')
            if row.get('sig_gap_down_close_high', False): sigs.append('gap+cl_hi')
            if row.get('sig_oi_down_price_down', False): sigs.append('oi_dn+dn')
            if row.get('sig_pct_rank_low', False): sigs.append('pct<10')
            if row.get('sig_cons_down3', False): sigs.append('cons_dn3')

            candidates.append({
                'symbol': sym,
                'score': row['signal_score'],
                'entry_price': row['entry_price'],
                'exit_target': row['exit_price'],
                'fwd_ret': row['fwd_ret_5d'],
                'signals': '+'.join(sigs),
            })

        # 按得分排序,取top N
        candidates.sort(key=lambda x: -x['score'])
        for cand in candidates[:n_open]:
            notional = capital * leverage / max_pos
            position_info.append({
                'symbol': cand['symbol'],
                'entry_date': dt,
                'entry_price': cand['entry_price'],
                'exit_target': cand['exit_target'],
                'notional': notional,
                'score': cand['score'],
                'signals': cand['signals'],
            })

        equity_curve.append({'date': dt, 'capital': capital})

    return equity_curve, trades


def analyze_results(equity_curve, trades, label=""):
    """分析回测结果"""
    eq = pd.DataFrame(equity_curve)
    if len(eq) == 0:
        print(f"  {label} 无数据")
        return {}

    eq['peak'] = eq['capital'].cummax()
    eq['dd'] = (eq['capital'] - eq['peak']) / eq['peak'] * 100

    total_ret = (eq['capital'].iloc[-1] / eq['capital'].iloc[0] - 1) * 100
    n_years = (eq['date'].iloc[-1] - eq['date'].iloc[0]).days / 365.25
    annual_ret = ((eq['capital'].iloc[-1] / eq['capital'].iloc[0]) ** (1/n_years) - 1) * 100 if n_years > 0 else 0
    max_dd = eq['dd'].min()

    # 交易统计
    if len(trades) > 0:
        trade_df = pd.DataFrame(trades)
        win_rate = (trade_df['pnl_pct'] > 0).mean() * 100
        avg_ret = trade_df['pnl_pct'].mean()
        avg_win = trade_df[trade_df['pnl_pct'] > 0]['pnl_pct'].mean() if (trade_df['pnl_pct'] > 0).any() else 0
        avg_loss = trade_df[trade_df['pnl_pct'] <= 0]['pnl_pct'].mean() if (trade_df['pnl_pct'] <= 0).any() else 0
        profit_factor = abs(avg_win * (trade_df['pnl_pct'] > 0).sum() / (avg_loss * (trade_df['pnl_pct'] <= 0).sum())) if avg_loss != 0 and (trade_df['pnl_pct'] <= 0).sum() > 0 else float('inf')
    else:
        win_rate = avg_ret = avg_win = avg_loss = profit_factor = 0

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  总交易: {len(trades)}")
    print(f"  胜率: {win_rate:.1f}%")
    print(f"  平均收益: {avg_ret:.3f}%")
    print(f"  平均盈利: {avg_win:.3f}%  平均亏损: {avg_loss:.3f}%")
    print(f"  盈亏比: {profit_factor:.2f}")
    print(f"  总收益: {total_ret:.1f}%")
    print(f"  年化收益: {annual_ret:.1f}%")
    print(f"  最大回撤: {max_dd:.1f}%")
    print(f"  期末资金: {eq['capital'].iloc[-1]:,.0f}")

    return {
        'trades': len(trades),
        'win_rate': win_rate,
        'avg_ret': avg_ret,
        'total_ret': total_ret,
        'annual_ret': annual_ret,
        'max_dd': max_dd,
        'profit_factor': profit_factor,
    }


def analyze_signals_detail(trades, label=""):
    """分析信号组合表现"""
    if len(trades) == 0:
        return
    df = pd.DataFrame(trades)

    print(f"\n--- {label} 信号组合分析 ---")

    # 按score分组
    print("\n按信号得分:")
    for score in sorted(df['score'].unique(), reverse=True):
        sub = df[df['score'] == score]
        wr = (sub['pnl_pct'] > 0).mean() * 100
        avg = sub['pnl_pct'].mean()
        print(f"  score={int(score):2d}: N={len(sub):4d}  WR={wr:.1f}%  Avg={avg:+.3f}%")

    # 触发频率最高的信号组合
    print("\nTop 10 信号组合:")
    combo_stats = df.groupby('signals').agg(
        N=('pnl_pct', 'count'),
        WR=('pnl_pct', lambda x: (x > 0).mean() * 100),
        Avg=('pnl_pct', 'mean'),
    ).sort_values('N', ascending=False)
    for sigs, row in combo_stats.head(10).iterrows():
        print(f"  {sigs:40s}  N={int(row['N']):4d}  WR={row['WR']:.1f}%  Avg={row['Avg']:+.3f}%")

    # 按年分析
    df['year'] = pd.to_datetime(df['exit_date']).dt.year
    print("\n按年表现:")
    for yr in sorted(df['year'].unique()):
        sub = df[df['year'] == yr]
        wr = (sub['pnl_pct'] > 0).mean() * 100
        avg = sub['pnl_pct'].mean()
        tot = sub['pnl_pct'].sum()
        print(f"  {yr}: N={len(sub):4d}  WR={wr:.1f}%  Avg={avg:+.3f}%  Sum={tot:+.1f}%")


def main():
    print("V69: 基于V68实证信号的期货策略")
    print("="*60)
    print(f"参数: 持仓={MAX_POSITIONS}, 杠杆={LEVERAGE}x, 持有={HOLD_DAYS}天")

    # 加载数据
    print("\n加载数据...")
    all_data = load_data()
    if not all_data:
        print("无数据!")
        return

    # ─── Walk-forward回测 ───
    print("\n" + "="*60)
    print("Walk-Forward 回测")
    print("="*60)

    # 测试不同最低得分门槛
    for min_score in [3, 5, 7, 9]:
        print(f"\n--- 最低得分门槛: {min_score} ---")
        eq, trades = run_backtest(all_data, TEST_START, '2025-12-31',
                                   MAX_POSITIONS, LEVERAGE, min_score)
        stats = analyze_results(eq, trades, f"测试期 (min_score={min_score})")
        if len(trades) > 0 and min_score <= 5:
            analyze_signals_detail(trades, f"min_score={min_score}")

    # ─── 不同杠杆测试 ───
    print("\n" + "="*60)
    print("杠杆敏感性分析")
    print("="*60)
    for lev in [1, 3, 5, 7, 10]:
        eq, trades = run_backtest(all_data, TEST_START, '2025-12-31',
                                   MAX_POSITIONS, lev, min_score=5)
        stats = analyze_results(eq, trades, f"杠杆={lev}x")

    # ─── 不同持仓天数 ───
    print("\n" + "="*60)
    print("持仓天数敏感性分析")
    print("="*60)
    for hd in [3, 5, 7, 10]:
        eq, trades = run_backtest(all_data, TEST_START, '2025-12-31',
                                   MAX_POSITIONS, LEVERAGE, min_score=5, hold_days=hd)
        stats = analyze_results(eq, trades, f"持有{hd}天")


if __name__ == '__main__':
    main()
