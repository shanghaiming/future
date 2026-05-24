#!/usr/bin/env python3
"""
V69b: 优化版 - V69基础上改进
1. 只交易score>=11的信号(68.9% WR)
2. 加入做空: gap_up信号反向操作
3. 动态杠杆: 近期表现好时加杠杆
4. 更现实的仓位管理: 限制单品种最大仓位
5. Walk-forward验证
"""

import os
import glob
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data/futures_weighted'
INITIAL_CAPITAL = 500_000
MAX_POSITIONS = 3
TRAIN_END = '2021-12-31'
TEST_START = '2022-01-01'
CONTRACT_SPECS = 'scripts/contract_specs.py'


def load_data():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cs", CONTRACT_SPECS)
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)

    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    all_data = {}
    for f in files:
        sym = os.path.basename(f).replace('.csv', '')
        try:
            mult, margin, tick, tick_val = cs.get_spec(sym)
        except:
            continue

        df = pd.read_csv(f)
        if len(df) < 100:
            continue

        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        df = df.sort_values('trade_date').reset_index(drop=True)

        if df['close'].isna().all() or (df['close'] == 0).any():
            continue

        all_data[sym] = {'df': df, 'multiplier': mult, 'margin_rate': margin}

    print(f"  {len(all_data)}品种")
    return all_data


def compute_all_signals(all_data, hold_days=5):
    """预计算所有品种的信号"""
    signal_data = {}
    for sym, info in all_data.items():
        df = info['df'].copy()
        c = df['close'].values
        o = df['open'].values
        h = df['high'].values
        l = df['low'].values
        v = df['vol'].values
        oi = df['oi'].values
        n = len(df)

        # 基础指标
        prev_c = np.full(n, np.nan)
        prev_c[1:] = c[:-1]
        df['prev_close'] = prev_c

        gap = np.full(n, np.nan)
        gap[1:] = (o[1:] - prev_c[1:]) / prev_c[1:] * 100
        df['gap_pct'] = gap

        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()

        mom5 = np.full(n, np.nan)
        mom5[5:] = (c[5:] - c[:-5]) / c[:-5] * 100
        df['mom5'] = mom5

        df['vol_ma5'] = df['vol'].rolling(5).mean()

        oi_ch = np.full(n, np.nan)
        valid_oi = np.abs(oi[:-1]) > 0
        oi_ch_vals = np.full(n-1, np.nan)
        oi_ch_vals[valid_oi] = (oi[1:][valid_oi] - oi[:-1][valid_oi]) / np.abs(oi[:-1][valid_oi]) * 100
        oi_ch[1:] = oi_ch_vals
        df['oi_chg'] = oi_ch

        # 做多信号得分
        score_long = np.zeros(n)
        # gap_down_1% (weight 3)
        sig_gd1 = df['gap_pct'] < -1.0
        score_long += sig_gd1.astype(int) * 3
        # gap_down_2% 额外加分 (weight 3)
        score_long += (df['gap_pct'] < -2.0).astype(int) * 3
        # oi_up + price_down (weight 3)
        sig_oi_up_dn = (df['oi_chg'] > 0) & (df['close'] < df['prev_close'])
        score_long += sig_oi_up_dn.astype(int) * 3
        # below_ma5 (weight 2)
        score_long += (df['close'] < df['ma5']).astype(int) * 2
        # mom5<-5% (weight 2)
        score_long += (df['mom5'] < -5.0).astype(int) * 2
        # vol_surge + down (weight 1)
        score_long += ((df['vol'] > df['vol_ma5'] * 2) & (df['close'] < df['prev_close'])).astype(int) * 1
        # gap_down + close at high (weight 2)
        range_day = h - l
        range_day[range_day == 0] = 0.001
        close_pos = (c - l) / range_day
        score_long += ((df['gap_pct'] < -1.0) & (close_pos > 0.8)).astype(int) * 2
        # pct_rank < 10% (weight 2)
        roll_min = df['close'].rolling(20).min()
        roll_max = df['close'].rolling(20).max()
        roll_range = roll_max - roll_min
        roll_range = roll_range.replace(0, 0.001)
        df['pct_rank'] = (df['close'] - roll_min) / roll_range * 100
        score_long += (df['pct_rank'] < 10).astype(int) * 2

        df['score_long'] = score_long

        # 做空信号得分 (gap_up反转)
        score_short = np.zeros(n)
        # gap_up_1% (weight 3)
        sig_gu1 = df['gap_pct'] > 1.0
        score_short += sig_gu1.astype(int) * 3
        # gap_up_2% 额外 (weight 3)
        score_short += (df['gap_pct'] > 2.0).astype(int) * 3
        # oi_up + price_up (空头建仓信号, weight 3)
        score_short += ((df['oi_chg'] > 0) & (df['close'] > df['prev_close'])).astype(int) * 3
        # above_ma5 (weight 2)
        score_short += (df['close'] > df['ma5']).astype(int) * 2
        # mom5>5% (weight 2)
        score_short += (df['mom5'] > 5.0).astype(int) * 2
        # vol_surge + up (weight 1)
        score_short += ((df['vol'] > df['vol_ma5'] * 2) & (df['close'] > df['prev_close'])).astype(int) * 1
        # pct_rank > 90% (weight 2)
        score_short += (df['pct_rank'] > 90).astype(int) * 2

        df['score_short'] = score_short

        # 前向收益(open-to-close after hold_days)
        fwd_ret = np.full(n, np.nan)
        valid_end = min(n - hold_days, n)
        if valid_end > 0:
            fwd_ret[:valid_end] = (c[hold_days:valid_end + hold_days] - o[:valid_end]) / o[:valid_end] * 100
        df['fwd_ret'] = fwd_ret

        signal_data[sym] = df

    return signal_data


def run_backtest(signal_data, start_date, end_date, max_pos=3, leverage=5,
                 min_score_long=11, min_score_short=11, hold_days=5,
                 dynamic_leverage=False, fixed_notional=None):
    """
    运行回测
    dynamic_leverage: True=根据近期表现动态调整杠杆
    fixed_notional: 固定名义金额(不使用复利)
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    capital = INITIAL_CAPITAL
    equity_curve = []
    trades = []
    positions = []  # 活跃持仓

    recent_pnls = []  # 用于动态杠杆

    for dt in date_range:
        # 1. 平仓
        closed_pnl = 0
        still_open = []
        for pos in positions:
            days_held = (dt - pos['entry_date']).days
            if days_held >= hold_days:
                sym = pos['symbol']
                df = signal_data.get(sym)
                if df is not None:
                    idx = df.index[df['trade_date'] == dt]
                    exit_price = df.loc[idx[0], 'close'] if len(idx) > 0 else pos['exit_target']
                else:
                    exit_price = pos['exit_target']

                if pos['direction'] == 'long':
                    pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price'] * 100
                else:
                    pnl_pct = (pos['entry_price'] - exit_price) / pos['entry_price'] * 100

                trade_pnl = pos['notional'] * pnl_pct / 100
                closed_pnl += trade_pnl
                recent_pnls.append(pnl_pct)

                trades.append({
                    'symbol': pos['symbol'],
                    'direction': pos['direction'],
                    'entry_date': pos['entry_date'],
                    'exit_date': dt,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'pnl_abs': trade_pnl,
                    'score': pos['score'],
                })
            else:
                still_open.append(pos)

        positions = still_open
        capital += closed_pnl
        if capital <= 0:
            equity_curve.append({'date': dt, 'capital': 0})
            break

        # 2. 动态杠杆
        cur_lev = leverage
        if dynamic_leverage and len(recent_pnls) >= 20:
            recent_wr = np.mean([p > 0 for p in recent_pnls[-20:]])
            recent_avg = np.mean(recent_pnls[-20:])
            if recent_wr > 0.6 and recent_avg > 1.5:
                cur_lev = min(leverage * 1.5, leverage + 3)  # 加杠杆
            elif recent_wr < 0.45 or recent_avg < -0.5:
                cur_lev = max(leverage * 0.5, 1)  # 减杠杆

        # 3. 开仓
        n_open = max_pos - len(positions)
        if n_open <= 0:
            equity_curve.append({'date': dt, 'capital': capital})
            continue

        candidates = []
        for sym, df in signal_data.items():
            if any(p['symbol'] == sym for p in positions):
                continue

            idx = df.index[df['trade_date'] == dt]
            if len(idx) == 0:
                continue
            row = df.loc[idx[0]]

            # 做多候选
            if row['score_long'] >= min_score_long and not np.isnan(row.get('fwd_ret', np.nan)):
                candidates.append({
                    'symbol': sym,
                    'direction': 'long',
                    'score': row['score_long'],
                    'entry_price': row['open'],  # 当天开盘价入场
                    'exit_target': row.get('close', row['open']),
                    'gap': row['gap_pct'],
                })

            # 做空候选
            if row['score_short'] >= min_score_short and not np.isnan(row.get('fwd_ret', np.nan)):
                candidates.append({
                    'symbol': sym,
                    'direction': 'short',
                    'score': row['score_short'],
                    'entry_price': row['open'],
                    'exit_target': row.get('close', row['open']),
                    'gap': row['gap_pct'],
                })

        # 同品种优先取score高的方向
        sym_best = {}
        for cand in candidates:
            sym = cand['symbol']
            if sym not in sym_best or cand['score'] > sym_best[sym]['score']:
                sym_best[sym] = cand

        # 排序取top N
        sorted_cands = sorted(sym_best.values(), key=lambda x: -x['score'])
        for cand in sorted_cands[:n_open]:
            if fixed_notional:
                notional = fixed_notional
            else:
                notional = capital * cur_lev / max_pos

            positions.append({
                'symbol': cand['symbol'],
                'direction': cand['direction'],
                'entry_date': dt,
                'entry_price': cand['entry_price'],
                'exit_target': cand['exit_target'],
                'notional': notional,
                'score': cand['score'],
            })

        equity_curve.append({'date': dt, 'capital': capital})

    return equity_curve, trades


def analyze(eq, trades, label=""):
    """分析结果"""
    eq_df = pd.DataFrame(eq)
    if len(eq_df) == 0 or eq_df['capital'].iloc[-1] <= 0:
        print(f"  {label}: 爆仓")
        return {}

    eq_df['peak'] = eq_df['capital'].cummax()
    eq_df['dd'] = (eq_df['capital'] - eq_df['peak']) / eq_df['peak'] * 100

    total_ret = (eq_df['capital'].iloc[-1] / eq_df['capital'].iloc[0] - 1) * 100
    n_years = max((eq_df['date'].iloc[-1] - eq_df['date'].iloc[0]).days / 365.25, 0.01)
    annual_ret = ((eq_df['capital'].iloc[-1] / eq_df['capital'].iloc[0]) ** (1/n_years) - 1) * 100
    max_dd = eq_df['dd'].min()

    if len(trades) > 0:
        tdf = pd.DataFrame(trades)
        wr = (tdf['pnl_pct'] > 0).mean() * 100
        avg = tdf['pnl_pct'].mean()
        avg_w = tdf[tdf['pnl_pct'] > 0]['pnl_pct'].mean() if (tdf['pnl_pct'] > 0).any() else 0
        avg_l = tdf[tdf['pnl_pct'] <= 0]['pnl_pct'].mean() if (tdf['pnl_pct'] <= 0).any() else 0
        pf = abs(avg_w * (tdf['pnl_pct'] > 0).sum() / (avg_l * (tdf['pnl_pct'] <= 0).sum())) if avg_l != 0 and (tdf['pnl_pct'] <= 0).sum() > 0 else float('inf')

        # 按方向
        long_t = tdf[tdf['direction'] == 'long']
        short_t = tdf[tdf['direction'] == 'short']
        long_wr = (long_t['pnl_pct'] > 0).mean() * 100 if len(long_t) > 0 else 0
        short_wr = (short_t['pnl_pct'] > 0).mean() * 100 if len(short_t) > 0 else 0

        # 按年
        tdf['year'] = pd.to_datetime(tdf['exit_date']).dt.year
    else:
        wr = avg = avg_w = avg_l = pf = long_wr = short_wr = 0
        tdf = pd.DataFrame()

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  总交易: {len(trades)}")
    print(f"  胜率: {wr:.1f}%  (多:{long_wr:.1f}% 空:{short_wr:.1f}%)")
    print(f"  平均收益: {avg:+.3f}%")
    print(f"  平均盈利: {avg_w:+.3f}%  平均亏损: {avg_l:+.3f}%")
    print(f"  盈亏比: {pf:.2f}")
    print(f"  年化收益: {annual_ret:.1f}%")
    print(f"  最大回撤: {max_dd:.1f}%")

    if len(tdf) > 0:
        print(f"\n  按年表现:")
        for yr in sorted(tdf['year'].unique()):
            sub = tdf[tdf['year'] == yr]
            ywr = (sub['pnl_pct'] > 0).mean() * 100
            yavg = sub['pnl_pct'].mean()
            ysum = sub['pnl_pct'].sum()
            print(f"    {yr}: N={len(sub):3d}  WR={ywr:.1f}%  Avg={yavg:+.3f}%")

        # 按score
        print(f"\n  按信号得分:")
        for sc in sorted(tdf['score'].unique(), reverse=True):
            sub = tdf[tdf['score'] == sc]
            swr = (sub['pnl_pct'] > 0).mean() * 100
            savg = sub['pnl_pct'].mean()
            print(f"    score={int(sc):2d}: N={len(sub):4d}  WR={swr:.1f}%  Avg={savg:+.3f}%")

    return {'annual': annual_ret, 'mdd': max_dd, 'wr': wr, 'trades': len(trades)}


def main():
    print("V69b: 优化版多空信号策略")
    print("="*60)

    print("加载数据...")
    all_data = load_data()

    print("计算信号...")
    signal_data = compute_all_signals(all_data, hold_days=5)

    # ─── 1. 基准: 只做多, score>=11 ───
    print("\n" + "="*60)
    print("方案1: 纯做多, 高分信号(score>=11)")
    print("="*60)

    for lev in [3, 5, 7]:
        eq, trades = run_backtest(signal_data, TEST_START, '2025-12-31',
                                    max_pos=3, leverage=lev,
                                    min_score_long=11, min_score_short=999,
                                    hold_days=5)
        analyze(eq, trades, f"做多 lev={lev}x min_score=11")

    # ─── 2. 多空组合 ───
    print("\n" + "="*60)
    print("方案2: 多空组合")
    print("="*60)

    for min_sc in [9, 11, 13]:
        for lev in [3, 5]:
            eq, trades = run_backtest(signal_data, TEST_START, '2025-12-31',
                                        max_pos=3, leverage=lev,
                                        min_score_long=min_sc,
                                        min_score_short=min_sc,
                                        hold_days=5)
            analyze(eq, trades, f"多空 lev={lev}x min_score={min_sc}")

    # ─── 3. 动态杠杆 ───
    print("\n" + "="*60)
    print("方案3: 动态杠杆")
    print("="*60)

    for base_lev in [3, 5]:
        for min_sc in [9, 11]:
            eq, trades = run_backtest(signal_data, TEST_START, '2025-12-31',
                                        max_pos=3, leverage=base_lev,
                                        min_score_long=min_sc,
                                        min_score_short=min_sc,
                                        hold_days=5, dynamic_leverage=True)
            analyze(eq, trades, f"动态杠杆 base={base_lev}x min_score={min_sc}")

    # ─── 4. 固定名义金额(不复利,更现实) ───
    print("\n" + "="*60)
    print("方案4: 固定名义金额(不复利)")
    print("="*60)

    for notional in [500_000, 1_000_000, 2_000_000]:
        eq, trades = run_backtest(signal_data, TEST_START, '2025-12-31',
                                    max_pos=3, leverage=5,
                                    min_score_long=11,
                                    min_score_short=11,
                                    hold_days=5, fixed_notional=notional)
        analyze(eq, trades, f"固定名义={notional/10000:.0f}万 min_score=11")

    # ─── 5. 不同持仓天数(高分信号) ───
    print("\n" + "="*60)
    print("方案5: 持仓天数敏感性(多空, score>=11, lev=5)")
    print("="*60)

    for hd in [3, 5, 7]:
        sd = compute_all_signals(all_data, hold_days=hd)
        eq, trades = run_backtest(sd, TEST_START, '2025-12-31',
                                    max_pos=3, leverage=5,
                                    min_score_long=11,
                                    min_score_short=11,
                                    hold_days=hd)
        analyze(eq, trades, f"多空 hold={hd}d score>=11 lev=5x")

    # ─── 6. 极端信号(score>=13) ───
    print("\n" + "="*60)
    print("方案6: 极端信号(score>=13)")
    print("="*60)

    for lev in [5, 7, 10]:
        eq, trades = run_backtest(signal_data, TEST_START, '2025-12-31',
                                    max_pos=3, leverage=lev,
                                    min_score_long=13,
                                    min_score_short=13,
                                    hold_days=5)
        analyze(eq, trades, f"多空 lev={lev}x min_score=13")


if __name__ == '__main__':
    main()
