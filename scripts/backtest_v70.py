#!/usr/bin/env python3
"""
V70: 基于学术研究优化的策略
改进点:
1. 日内gap fade (当天开盘买入,收盘平仓) — Sharpe 3.5
2. NR4/NR7窄幅突破过滤 — 65-77% WR
3. OI象限框架 (4种OI+价格组合)
4. 动态止损/止盈
5. 纯日内交易 vs 多日持仓对比
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
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    all_data = {}
    for f in files:
        sym = os.path.basename(f).replace('.csv', '')
        try: mult, margin, tick, tick_val = cs.get_spec(sym)
        except: continue
        df = pd.read_csv(f)
        if len(df) < 100: continue
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        df = df.sort_values('trade_date').reset_index(drop=True)
        if df['close'].isna().all() or (df['close'] == 0).any(): continue
        all_data[sym] = {'df': df, 'multiplier': mult, 'margin_rate': margin}
    print(f"  {len(all_data)}品种")
    return all_data


def compute_enhanced_signals(all_data):
    """计算增强信号"""
    signal_data = {}
    for sym, info in all_data.items():
        df = info['df'].copy()
        c = df['close'].values.astype(float)
        o = df['open'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        v = df['vol'].values.astype(float)
        oi = df['oi'].values.astype(float)
        n = len(df)

        prev_c = np.full(n, np.nan); prev_c[1:] = c[:-1]
        df['prev_close'] = prev_c

        # ─── Gap ───
        gap = np.full(n, np.nan)
        gap[1:] = (o[1:] - prev_c[1:]) / prev_c[1:] * 100
        df['gap_pct'] = gap

        # ─── ATR ───
        tr = np.full(n, np.nan)
        tr[1:] = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
        df['atr'] = pd.Series(tr).rolling(20).mean().values
        df['atr_pct'] = df['atr'] / df['close'] * 100

        # ─── 日内收益 (open to close) ───
        df['intraday_ret'] = (c - o) / o * 100

        # ─── 次日收益 ───
        next_open = np.full(n, np.nan); next_open[:-1] = o[1:]
        next_close = np.full(n, np.nan); next_close[:-1] = c[1:]
        df['next_open'] = next_open
        df['next_close_1d'] = next_close
        df['next_ret_1d'] = (next_close - next_open) / next_open * 100

        # 5日后收益 (从当天open)
        fwd5 = np.full(n, np.nan)
        if n > 5:
            fwd5[:n-5] = (c[5:] - o[:n-5]) / o[:n-5] * 100
        df['fwd_ret_5d'] = fwd5

        # 3日后收益
        fwd3 = np.full(n, np.nan)
        if n > 3:
            fwd3[:n-3] = (c[3:] - o[:n-3]) / o[:n-3] * 100
        df['fwd_ret_3d'] = fwd3

        # ─── MA ───
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()

        # ─── 动量 ───
        mom5 = np.full(n, np.nan)
        mom5[5:] = (c[5:] - c[:-5]) / c[:-5] * 100
        df['mom5'] = mom5

        # ─── OI变化 ───
        oi_ch = np.full(n, np.nan)
        valid = np.abs(oi[:-1]) > 0
        oi_ch_vals = np.full(n-1, np.nan)
        oi_ch_vals[valid] = (oi[1:][valid] - oi[:-1][valid]) / np.abs(oi[:-1][valid]) * 100
        oi_ch[1:] = oi_ch_vals
        df['oi_chg'] = oi_ch
        df['oi_ma5'] = df['oi'].rolling(5).mean()

        # ─── 成交量 ───
        df['vol_ma5'] = df['vol'].rolling(5).mean()
        df['vol_ma20'] = df['vol'].rolling(20).mean()

        # ─── NR4/NR7 (窄幅日) ───
        range_ = h - l
        df['range'] = range_
        df['range_ma4'] = pd.Series(range_).rolling(4).mean().values  # 不是MA,是min of last 4
        # NR4: 今天是近4天最窄的
        nr4 = np.zeros(n, dtype=bool)
        for i in range(4, n):
            if range_[i] == np.min(range_[i-3:i+1]) and range_[i] > 0:
                nr4[i] = True
        df['nr4'] = nr4

        nr7 = np.zeros(n, dtype=bool)
        for i in range(7, n):
            if range_[i] == np.min(range_[i-6:i+1]) and range_[i] > 0:
                nr7[i] = True
        df['nr7'] = nr7

        # ─── Close Location Value (CLV) ───
        clv = np.full(n, np.nan)
        valid_range = range_ > 0
        clv[valid_range] = (2*c[valid_range] - h[valid_range] - l[valid_range]) / range_[valid_range]
        df['clv'] = clv

        # ─── 20日百分位 ───
        roll_min = df['close'].rolling(20).min()
        roll_max = df['close'].rolling(20).max()
        roll_range = (roll_max - roll_min).replace(0, 0.001)
        df['pct_rank'] = (df['close'] - roll_min) / roll_range * 100

        # ─── 连续下跌 ───
        price_down = np.zeros(n, dtype=bool)
        price_down[1:] = c[1:] < c[:-1]
        cons_down = np.zeros(n, dtype=int)
        for i in range(1, n):
            if price_down[i]:
                cons_down[i] = cons_down[i-1] + 1
            else:
                cons_down[i] = 0
        df['cons_down'] = cons_down

        # ═══════════════════════════════════════
        # 信号定义
        # ═══════════════════════════════════════

        # A. Gap Down Fade (做多)
        # A1: gap_down > 1%
        df['sig_gap_dn1'] = gap < -1.0
        # A2: gap_down > 1.5 * ATR% (波动率调整)
        df['sig_gap_dn_atr'] = gap < -1.5 * df['atr_pct']
        # A3: gap_down > 2%
        df['sig_gap_dn2'] = gap < -2.0

        # B. OI象限
        # B1: OI增+价跌 = 新空头入场 → 做多反转
        df['sig_oi_up_price_dn'] = (oi_ch > 0) & (c < prev_c)
        # B2: OI减+价跌 = 多头平仓 → 做多抄底
        df['sig_oi_dn_price_dn'] = (oi_ch < 0) & (c < prev_c)
        # B3: OI增+价涨 = 新多头入场 → 做多追涨 (趋势)
        df['sig_oi_up_price_up'] = (oi_ch > 0) & (c > prev_c)

        # C. NR4/NR7 + 次日突破
        # 今天是NR4/NR7, 明天开盘>今天最高 = 突破做多
        next_gt_high = np.zeros(n, dtype=bool)
        next_lt_low = np.zeros(n, dtype=bool)
        next_gt_high[:-1] = o[1:] > h[:-1]
        next_lt_low[:-1] = o[1:] < l[:-1]
        df['sig_nr4_breakout_up'] = nr4 & next_gt_high  # 这个需要提前一天
        # 改为: 昨天NR4, 今天开盘>昨天最高
        prev_nr4 = np.zeros(n, dtype=bool)
        prev_nr4[1:] = nr4[:-1]
        prev_nr7 = np.zeros(n, dtype=bool)
        prev_nr7[1:] = nr7[:-1]
        prev_high = np.full(n, np.nan); prev_high[1:] = h[:-1]
        prev_low = np.full(n, np.nan); prev_low[1:] = l[:-1]
        df['sig_nr4_break'] = prev_nr4 & (o > prev_high)
        df['sig_nr7_break'] = prev_nr7 & (o > prev_high)

        # D. 放量+极端下跌
        df['sig_vol_surge_dn'] = (v > df['vol_ma5'] * 2) & (c < prev_c)
        df['sig_mom5_extreme'] = mom5 < -5.0
        df['sig_below_ma5'] = c < df['ma5'].values

        # ═══════════════════════════════════════
        # 信号得分 (基于V69b实证优化)
        # score=11 是最佳: gap_down(3) + oi_up+price_dn(3) + below_ma5(2) + ... = 73.8% WR
        # ═══════════════════════════════════════
        score = np.zeros(n)
        score += df['sig_gap_dn1'].astype(int) * 3    # 最强信号
        score += df['sig_gap_dn2'].astype(int) * 3    # 额外加分
        score += df['sig_oi_up_price_dn'].astype(int) * 3  # 次强信号
        score += df['sig_below_ma5'].astype(int) * 2
        score += df['sig_mom5_extreme'].astype(int) * 2
        score += df['sig_vol_surge_dn'].astype(int) * 1
        score += df['sig_oi_dn_price_dn'].astype(int) * 2
        score += (df['pct_rank'] < 10).astype(int) * 2
        df['score_long'] = score

        signal_data[sym] = df
    return signal_data


def analyze_signal_returns(signal_data, start_date, end_date):
    """分析各信号的收益率统计"""
    print(f"\n{'='*70}")
    print(f"信号收益率分析 ({start_date} ~ {end_date})")
    print(f"{'='*70}")

    signals = [
        ('sig_gap_dn1', 'Gap Down >1%'),
        ('sig_gap_dn2', 'Gap Down >2%'),
        ('sig_gap_dn_atr', 'Gap Down >1.5*ATR'),
        ('sig_oi_up_price_dn', 'OI↑+Price↓'),
        ('sig_oi_dn_price_dn', 'OI↓+Price↓'),
        ('sig_nr4_break', 'NR4 Breakout↑'),
        ('sig_nr7_break', 'NR7 Breakout↑'),
        ('sig_vol_surge_dn', 'Vol Surge+↓'),
        ('sig_mom5_extreme', 'Mom5<-5%'),
        ('sig_below_ma5', 'Below MA5'),
    ]

    for sig_col, sig_name in signals:
        rows = []
        for sym, df in signal_data.items():
            mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date) & df[sig_col]
            sub = df[mask]
            if len(sub) > 0:
                rows.append(sub)
        if not rows:
            continue
        all_rows = pd.concat(rows)

        # 日内收益 (open to close)
        intra = all_rows['intraday_ret'].dropna()
        # 次日收益 (next open to next close)
        next1d = all_rows['next_ret_1d'].dropna()
        # 3日后
        fwd3 = all_rows['fwd_ret_3d'].dropna()
        # 5日后
        fwd5 = all_rows['fwd_ret_5d'].dropna()

        print(f"\n  {sig_name} (N={len(all_rows)})")
        print(f"    Intraday:  WR={100*(intra>0).mean():.1f}% Avg={intra.mean():+.3f}% t={intra.mean()/(intra.std()/len(intra)**0.5):.1f}")
        if len(next1d) > 0:
            print(f"    Next 1d:   WR={100*(next1d>0).mean():.1f}% Avg={next1d.mean():+.3f}% t={next1d.mean()/(next1d.std()/len(next1d)**0.5):.1f}")
        if len(fwd3) > 0:
            print(f"    Fwd 3d:    WR={100*(fwd3>0).mean():.1f}% Avg={fwd3.mean():+.3f}% t={fwd3.mean()/(fwd3.std()/len(fwd3)**0.5):.1f}")
        if len(fwd5) > 0:
            print(f"    Fwd 5d:    WR={100*(fwd5>0).mean():.1f}% Avg={fwd5.mean():+.3f}% t={fwd5.mean()/(fwd5.std()/len(fwd5)**0.5):.1f}")


def run_intraday_backtest(signal_data, start_date, end_date, max_pos=3,
                           leverage=3, min_score=11, use_stop=False):
    """
    日内回测: 当天开盘买, 收盘卖
    如果intraday收益好, 可以隔夜持有到5天后
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    capital = INITIAL_CAPITAL
    eq_curve = []
    trades = []

    # 隔夜持仓
    overnight_positions = []

    for dt in date_range:
        # 1. 平隔夜仓
        closed_pnl = 0
        still_open = []
        for pos in overnight_positions:
            sym = pos['symbol']
            df = signal_data.get(sym)
            exit_price = None
            if df is not None:
                idx = df.index[df['trade_date'] == dt]
                if len(idx) > 0:
                    exit_price = df.loc[idx[0], 'open']  # 开盘平仓

            if exit_price is None or np.isnan(exit_price):
                still_open.append(pos)
                continue

            days_held = (dt - pos['entry_date']).days
            pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price'] * 100
            trade_pnl = pos['notional'] * pnl_pct / 100
            closed_pnl += trade_pnl
            trades.append({
                'symbol': sym, 'direction': 'long',
                'entry_date': pos['entry_date'], 'exit_date': dt,
                'entry_price': pos['entry_price'], 'exit_price': exit_price,
                'pnl_pct': pnl_pct, 'pnl_abs': trade_pnl,
                'score': pos['score'], 'hold_days': days_held,
                'type': 'overnight',
            })
        overnight_positions = still_open
        capital += closed_pnl
        if capital <= 0:
            eq_curve.append({'date': dt, 'capital': 0})
            break

        # 2. 日内交易 (当天open买, close卖)
        n_available = max_pos - len(overnight_positions)
        if n_available > 0:
            candidates = []
            for sym, df in signal_data.items():
                if any(p['symbol'] == sym for p in overnight_positions):
                    continue
                idx = df.index[df['trade_date'] == dt]
                if len(idx) == 0: continue
                row = df.loc[idx[0]]
                if row['score_long'] < min_score: continue
                if np.isnan(row.get('intraday_ret', np.nan)): continue

                candidates.append({
                    'symbol': sym,
                    'score': row['score_long'],
                    'entry_price': row['open'],
                    'intraday_ret': row['intraday_ret'],
                    'close_price': row['close'],
                    'fwd_ret_5d': row.get('fwd_ret_5d', np.nan),
                })

            candidates.sort(key=lambda x: -x['score'])

            for cand in candidates[:n_available]:
                notional = capital * leverage / max_pos

                # 日内收益
                intra_ret = cand['intraday_ret']
                if use_stop and intra_ret < -2.0:
                    # 止损: 如果日内亏损超过2%, 假设在-2%止损
                    intra_ret = -2.0
                    close_price = cand['entry_price'] * (1 - 0.02)
                else:
                    close_price = cand['close_price']

                intra_pnl = notional * intra_ret / 100
                capital += intra_pnl

                trades.append({
                    'symbol': cand['symbol'], 'direction': 'long',
                    'entry_date': dt, 'exit_date': dt,
                    'entry_price': cand['entry_price'], 'exit_price': close_price,
                    'pnl_pct': intra_ret, 'pnl_abs': intra_pnl,
                    'score': cand['score'], 'hold_days': 0,
                    'type': 'intraday',
                })

        eq_curve.append({'date': dt, 'capital': max(capital, 0)})

    return eq_curve, trades


def run_multi_hold_backtest(signal_data, start_date, end_date, max_pos=3,
                             leverage=3, min_score=11, hold_days=5,
                             stop_loss_pct=-3.0, take_profit_pct=None):
    """多日持仓回测, 带止损"""
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    capital = INITIAL_CAPITAL
    eq_curve = []
    trades = []
    positions = []

    for dt in date_range:
        # 1. 检查止损/止盈/到期平仓
        closed_pnl = 0
        still_open = []
        for pos in positions:
            sym = pos['symbol']
            df = signal_data.get(sym)
            if df is not None:
                idx = df.index[df['trade_date'] == dt]
                cur_price = df.loc[idx[0], 'close'] if len(idx) > 0 else None
            else:
                cur_price = None

            if cur_price is None or np.isnan(cur_price):
                still_open.append(pos)
                continue

            pnl_pct = (cur_price - pos['entry_price']) / pos['entry_price'] * 100
            days_held = (dt - pos['entry_date']).days

            # 止损
            if stop_loss_pct and pnl_pct <= stop_loss_pct:
                trade_pnl = pos['notional'] * pnl_pct / 100
                closed_pnl += trade_pnl
                trades.append({
                    'symbol': sym, 'direction': 'long',
                    'entry_date': pos['entry_date'], 'exit_date': dt,
                    'entry_price': pos['entry_price'], 'exit_price': cur_price,
                    'pnl_pct': pnl_pct, 'pnl_abs': trade_pnl,
                    'score': pos['score'], 'hold_days': days_held,
                    'exit_reason': 'stop_loss',
                })
                continue

            # 止盈
            if take_profit_pct and pnl_pct >= take_profit_pct:
                trade_pnl = pos['notional'] * pnl_pct / 100
                closed_pnl += trade_pnl
                trades.append({
                    'symbol': sym, 'direction': 'long',
                    'entry_date': pos['entry_date'], 'exit_date': dt,
                    'entry_price': pos['entry_price'], 'exit_price': cur_price,
                    'pnl_pct': pnl_pct, 'pnl_abs': trade_pnl,
                    'score': pos['score'], 'hold_days': days_held,
                    'exit_reason': 'take_profit',
                })
                continue

            # 到期
            if days_held >= hold_days:
                trade_pnl = pos['notional'] * pnl_pct / 100
                closed_pnl += trade_pnl
                trades.append({
                    'symbol': sym, 'direction': 'long',
                    'entry_date': pos['entry_date'], 'exit_date': dt,
                    'entry_price': pos['entry_price'], 'exit_price': cur_price,
                    'pnl_pct': pnl_pct, 'pnl_abs': trade_pnl,
                    'score': pos['score'], 'hold_days': days_held,
                    'exit_reason': 'expire',
                })
                continue

            still_open.append(pos)

        positions = still_open
        capital += closed_pnl
        if capital <= 0:
            eq_curve.append({'date': dt, 'capital': 0})
            break

        # 2. 开仓
        n_open = max_pos - len(positions)
        if n_open <= 0:
            eq_curve.append({'date': dt, 'capital': capital})
            continue

        candidates = []
        for sym, df in signal_data.items():
            if any(p['symbol'] == sym for p in positions): continue
            idx = df.index[df['trade_date'] == dt]
            if len(idx) == 0: continue
            row = df.loc[idx[0]]
            if row['score_long'] < min_score: continue
            if np.isnan(row['open']) or row['open'] <= 0: continue
            candidates.append({
                'symbol': sym,
                'score': row['score_long'],
                'entry_price': row['open'],
            })

        candidates.sort(key=lambda x: -x['score'])
        for cand in candidates[:n_open]:
            notional = capital * leverage / max_pos
            positions.append({
                'symbol': cand['symbol'],
                'entry_date': dt,
                'entry_price': cand['entry_price'],
                'notional': notional,
                'score': cand['score'],
            })

        eq_curve.append({'date': dt, 'capital': capital})

    return eq_curve, trades


def print_results(eq, trades, label):
    eq_df = pd.DataFrame(eq)
    if len(eq_df) == 0 or eq_df['capital'].iloc[-1] <= 0:
        print(f"  {label}: 爆仓")
        return

    eq_df['peak'] = eq_df['capital'].cummax()
    eq_df['dd'] = (eq_df['capital'] - eq_df['peak']) / eq_df['peak'] * 100

    total_ret = (eq_df['capital'].iloc[-1] / eq_df['capital'].iloc[0] - 1) * 100
    n_years = max((eq_df['date'].iloc[-1] - eq_df['date'].iloc[0]).days / 365.25, 0.01)
    annual_ret = ((eq_df['capital'].iloc[-1] / eq_df['capital'].iloc[0]) ** (1/n_years) - 1) * 100
    max_dd = eq_df['dd'].min()
    sharpe = 0

    if len(trades) > 0:
        tdf = pd.DataFrame(trades)
        wr = (tdf['pnl_pct'] > 0).mean() * 100
        avg = tdf['pnl_pct'].mean()
        avg_w = tdf[tdf['pnl_pct'] > 0]['pnl_pct'].mean() if (tdf['pnl_pct'] > 0).any() else 0
        avg_l = tdf[tdf['pnl_pct'] <= 0]['pnl_pct'].mean() if (tdf['pnl_pct'] <= 0).any() else 0
        pf = abs(avg_w * (tdf['pnl_pct'] > 0).sum() / (avg_l * (tdf['pnl_pct'] <= 0).sum())) if avg_l != 0 and (tdf['pnl_pct'] <= 0).sum() > 0 else 999

        # Sharpe (from daily equity changes)
        daily_ret = eq_df['capital'].pct_change().dropna()
        if len(daily_ret) > 0 and daily_ret.std() > 0:
            sharpe = daily_ret.mean() / daily_ret.std() * (252 ** 0.5)

        tdf['year'] = pd.to_datetime(tdf['exit_date']).dt.year
    else:
        wr = avg = avg_w = avg_l = pf = 0
        tdf = pd.DataFrame()

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  交易: {len(trades)}  胜率: {wr:.1f}%  盈亏比: {pf:.2f}  Sharpe: {sharpe:.2f}")
    print(f"  年化: {annual_ret:.1f}%  MDD: {max_dd:.1f}%  Avg: {avg:+.3f}%")

    if len(tdf) > 0 and 'year' in tdf.columns:
        print(f"  按年:")
        for yr in sorted(tdf['year'].unique()):
            sub = tdf[tdf['year'] == yr]
            ywr = (sub['pnl_pct'] > 0).mean() * 100
            yavg = sub['pnl_pct'].mean()
            print(f"    {yr}: N={len(sub):3d} WR={ywr:.1f}% Avg={yavg:+.3f}%")


def main():
    print("V70: 学术研究优化策略")
    print("="*60)

    print("加载数据...")
    all_data = load_data()

    print("计算信号...")
    signal_data = compute_enhanced_signals(all_data)

    # ─── 信号分析 ───
    analyze_signal_returns(signal_data, TEST_START, '2025-12-31')

    # ═══════════════════════════════════════════
    # 策略测试
    # ═══════════════════════════════════════════

    # ─── A. 日内Gap Fade ───
    print(f"\n\n{'='*60}")
    print("A. 日内Gap Fade (当天开盘买,收盘卖)")
    print(f"{'='*60}")

    for min_sc in [9, 11, 13]:
        for lev in [3, 5]:
            eq, trades = run_intraday_backtest(signal_data, TEST_START, '2025-12-31',
                                                 max_pos=3, leverage=lev, min_score=min_sc)
            print_results(eq, trades, f"日内 min={min_sc} lev={lev}x")

    # ─── B. 日内 + 2%止损 ───
    print(f"\n\n{'='*60}")
    print("B. 日内Gap Fade + 2%止损")
    print(f"{'='*60}")

    for lev in [3, 5, 7]:
        eq, trades = run_intraday_backtest(signal_data, TEST_START, '2025-12-31',
                                             max_pos=3, leverage=lev, min_score=11,
                                             use_stop=True)
        print_results(eq, trades, f"日内+止损 lev={lev}x min=11")

    # ─── C. 多日持仓 + 止损/止盈 ───
    print(f"\n\n{'='*60}")
    print("C. 多日持仓 + 止损/止盈")
    print(f"{'='*60}")

    for sl, tp in [(-2, None), (-3, None), (-2, 5), (-3, 8), (None, None)]:
        for lev in [3, 5]:
            eq, trades = run_multi_hold_backtest(signal_data, TEST_START, '2025-12-31',
                                                   max_pos=3, leverage=lev, min_score=11,
                                                   hold_days=5, stop_loss_pct=sl,
                                                   take_profit_pct=tp)
            sl_s = f"SL={sl}%" if sl else "NoSL"
            tp_s = f"TP={tp}%" if tp else "NoTP"
            print_results(eq, trades, f"{sl_s} {tp_s} lev={lev}x")

    # ─── D. 日内 + 隔夜延续 (日内盈利则持有到5天) ───
    print(f"\n\n{'='*60}")
    print("D. 混合: 日内盈利则持有5天, 亏损则当天平")
    print(f"{'='*60}")

    date_range = pd.date_range(start=TEST_START, end='2025-12-31', freq='B')
    for lev in [3, 5]:
        capital = INITIAL_CAPITAL
        eq_curve = []
        trades = []
        positions = []  # 隔夜持仓

        for dt in date_range:
            # 平到期仓
            closed_pnl = 0
            still_open = []
            for pos in positions:
                df = signal_data.get(pos['symbol'])
                exit_price = None
                if df is not None:
                    idx = df.index[df['trade_date'] == dt]
                    if len(idx) > 0: exit_price = df.loc[idx[0], 'open']
                if exit_price is None or np.isnan(exit_price):
                    still_open.append(pos); continue

                days_held = (dt - pos['entry_date']).days
                if days_held >= 5:
                    pnl = (exit_price - pos['entry_price']) / pos['entry_price'] * 100
                    closed_pnl += pos['notional'] * pnl / 100
                    trades.append({
                        'symbol': pos['symbol'], 'entry_date': pos['entry_date'],
                        'exit_date': dt, 'entry_price': pos['entry_price'],
                        'exit_price': exit_price, 'pnl_pct': pnl,
                        'pnl_abs': pos['notional'] * pnl / 100,
                        'score': pos['score'], 'hold_days': days_held,
                    })
                else:
                    still_open.append(pos)
            positions = still_open
            capital += closed_pnl
            if capital <= 0:
                eq_curve.append({'date': dt, 'capital': 0}); break

            # 新开仓
            n_open = 3 - len(positions)
            if n_open <= 0:
                eq_curve.append({'date': dt, 'capital': capital}); continue

            candidates = []
            for sym, df in signal_data.items():
                if any(p['symbol'] == sym for p in positions): continue
                idx = df.index[df['trade_date'] == dt]
                if len(idx) == 0: continue
                row = df.loc[idx[0]]
                if row['score_long'] < 11: continue
                candidates.append({
                    'symbol': sym, 'score': row['score_long'],
                    'open': row['open'], 'close': row['close'],
                    'intraday_ret': row.get('intraday_ret', np.nan),
                })

            candidates.sort(key=lambda x: -x['score'])
            for cand in candidates[:n_open]:
                notional = capital * lev / 3
                intra_ret = cand.get('intraday_ret', 0)
                if np.isnan(intra_ret): intra_ret = 0

                if intra_ret > 0:
                    # 日内盈利, 隔夜持有
                    positions.append({
                        'symbol': cand['symbol'], 'entry_date': dt,
                        'entry_price': cand['open'], 'notional': notional,
                        'score': cand['score'],
                    })
                else:
                    # 日内亏损, 当天平仓
                    pnl = notional * intra_ret / 100
                    capital += pnl
                    trades.append({
                        'symbol': cand['symbol'], 'entry_date': dt,
                        'exit_date': dt, 'entry_price': cand['open'],
                        'exit_price': cand['close'], 'pnl_pct': intra_ret,
                        'pnl_abs': pnl, 'score': cand['score'],
                        'hold_days': 0,
                    })

            eq_curve.append({'date': dt, 'capital': capital})

        print_results(eq_curve, trades, f"混合(盈利持有5d) lev={lev}x")

    # ─── E. 3日持仓 ───
    print(f"\n\n{'='*60}")
    print("E. 3日持仓 (score>=11, 无止损)")
    print(f"{'='*60}")

    for lev in [3, 5, 7]:
        eq, trades = run_multi_hold_backtest(signal_data, TEST_START, '2025-12-31',
                                               max_pos=3, leverage=lev, min_score=11,
                                               hold_days=3)
        print_results(eq, trades, f"3日持仓 lev={lev}x")


if __name__ == '__main__':
    main()
