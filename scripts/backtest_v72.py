#!/usr/bin/env python3
"""
V72: 最终优化策略
整合V71所有发现:
1. 多空双向 (gap_down做多 + gap_up做空)
2. 趋势过滤 (MA20方向确认)
3. 2日持仓
4. 3x杠杆, 3个持仓
5. Score>=7
6. Walk-forward验证
7. 固定名义金额(现实估算)
"""
import os, glob, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data/futures_weighted'
INITIAL_CAPITAL = 500_000
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
        try: mult, margin, tick, tick_val = cs.get_spec(sym)
        except: continue
        df = pd.read_csv(f)
        if len(df) < 100: continue
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        df = df.sort_values('trade_date').reset_index(drop=True)
        if df['close'].isna().all() or (df['close'] == 0).any(): continue
        all_data[sym] = {'df': df, 'mult': mult, 'margin': margin}
    print(f"  {len(all_data)}品种")
    return all_data


def compute_signals(all_data):
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
        gap = np.full(n, np.nan)
        gap[1:] = (o[1:] - prev_c[1:]) / prev_c[1:] * 100
        df['gap_pct'] = gap

        tr = np.full(n, np.nan)
        tr[1:] = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
        df['atr'] = pd.Series(tr).rolling(20).mean().values
        df['atr_pct'] = df['atr'] / df['close'] * 100
        df['gap_atr'] = df['gap_pct'] / df['atr_pct'].replace(0, np.nan)

        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()

        mom5 = np.full(n, np.nan); mom5[5:] = (c[5:] - c[:-5]) / c[:-5] * 100
        df['mom5'] = mom5
        mom20 = np.full(n, np.nan); mom20[20:] = (c[20:] - c[:-20]) / c[:-20] * 100
        df['mom20'] = mom20

        oi_ch = np.full(n, np.nan)
        valid = np.abs(oi[:-1]) > 0
        oi_ch_vals = np.full(n-1, np.nan)
        oi_ch_vals[valid] = (oi[1:][valid] - oi[:-1][valid]) / np.abs(oi[:-1][valid]) * 100
        oi_ch[1:] = oi_ch_vals
        df['oi_chg'] = oi_ch
        df['vol_ma5'] = df['vol'].rolling(5).mean()

        range_ = h - l
        df['clv'] = np.where(range_ > 0, (2*c - h - l) / range_, 0)

        # 趋势
        df['trend_up'] = df['ma20'] > df['ma60']
        df['trend_down'] = df['ma20'] < df['ma60']

        # ═══ 做多信号 ═══
        s_long = np.zeros(n)
        gv = df['gap_pct'].fillna(0)
        ga = df['gap_atr'].fillna(0)
        s_long += (gv < -0.5).astype(int) * 1
        s_long += (gv < -1.0).astype(int) * 2
        s_long += (gv < -1.5).astype(int) * 2
        s_long += (gv < -2.0).astype(int) * 3
        s_long += (ga < -1.0).astype(int) * 2
        s_long += (ga < -1.5).astype(int) * 3
        s_long += (ga < -2.0).astype(int) * 3
        s_long += ((oi_ch > 0) & (c < prev_c)).astype(int) * 3
        s_long += ((oi_ch < 0) & (c < prev_c)).astype(int) * 2
        s_long += (df['mom5'] < -3).astype(int) * 1
        s_long += (df['mom5'] < -5).astype(int) * 1
        s_long += (c < df['ma5'].values).astype(int) * 1
        s_long += ((v > df['vol_ma5'] * 1.5) & (c < prev_c)).astype(int) * 1
        s_long += (df['clv'] > 0.5).astype(int) * 1
        # 趋势加分
        s_long += df['trend_up'].astype(int) * 2  # 上升趋势中做多加分
        df['score_long'] = s_long

        # ═══ 做空信号 ═══
        s_short = np.zeros(n)
        s_short += (gv > 0.5).astype(int) * 1
        s_short += (gv > 1.0).astype(int) * 2
        s_short += (gv > 1.5).astype(int) * 2
        s_short += (gv > 2.0).astype(int) * 3
        s_short += (ga > 1.0).astype(int) * 2
        s_short += (ga > 1.5).astype(int) * 3
        s_short += (ga > 2.0).astype(int) * 3
        s_short += ((oi_ch > 0) & (c > prev_c)).astype(int) * 3
        s_short += ((oi_ch < 0) & (c > prev_c)).astype(int) * 2
        s_short += (df['mom5'] > 3).astype(int) * 1
        s_short += (df['mom5'] > 5).astype(int) * 1
        s_short += (c > df['ma5'].values).astype(int) * 1
        s_short += ((v > df['vol_ma5'] * 1.5) & (c > prev_c)).astype(int) * 1
        s_short += (df['clv'] < -0.5).astype(int) * 1
        # 趋势加分
        s_short += df['trend_down'].astype(int) * 2  # 下降趋势中做空加分
        df['score_short'] = s_short

        for hd in [1, 2, 3, 5]:
            fwd = np.full(n, np.nan)
            if n > hd: fwd[:n-hd] = (c[hd:] - o[:n-hd]) / o[:n-hd] * 100
            df[f'fwd_{hd}d'] = fwd

        signal_data[sym] = df
    return signal_data


def run_final_backtest(signal_data, start_date, end_date, max_pos=3, leverage=3,
                       min_score=7, hold_days=2, fixed_notional=None,
                       stop_loss=None, take_profit=None):
    """最终回测引擎"""
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    capital = INITIAL_CAPITAL
    eq_curve = []
    trades = []
    positions = []

    for dt in date_range:
        # 平仓
        closed_pnl = 0
        still_open = []
        for pos in positions:
            df = signal_data.get(pos['symbol'])
            cur = None
            if df is not None:
                idx = df.index[df['trade_date'] == dt]
                if len(idx) > 0: cur = df.loc[idx[0], 'close']
            if cur is None or np.isnan(cur):
                still_open.append(pos); continue

            if pos['direction'] == 'long':
                pnl_pct = (cur - pos['entry_price']) / pos['entry_price'] * 100
            else:
                pnl_pct = (pos['entry_price'] - cur) / pos['entry_price'] * 100

            days = (dt - pos['entry_date']).days
            exit_reason = None

            if stop_loss and pnl_pct <= stop_loss:
                exit_reason = 'SL'
            elif take_profit and pnl_pct >= take_profit:
                exit_reason = 'TP'
            elif days >= hold_days:
                exit_reason = 'expire'

            if exit_reason:
                trade_pnl = pos['notional'] * pnl_pct / 100
                closed_pnl += trade_pnl
                trades.append({
                    'symbol': pos['symbol'], 'direction': pos['direction'],
                    'entry_date': pos['entry_date'], 'exit_date': dt,
                    'entry_price': pos['entry_price'], 'exit_price': cur,
                    'pnl_pct': pnl_pct, 'pnl_abs': trade_pnl,
                    'score': pos['score'], 'hold_days': days,
                    'exit_reason': exit_reason,
                })
            else:
                still_open.append(pos)

        positions = still_open
        capital += closed_pnl
        if capital <= 0:
            eq_curve.append({'date': dt, 'capital': 0}); break

        # 开仓
        n_open = max_pos - len(positions)
        if n_open <= 0:
            eq_curve.append({'date': dt, 'capital': capital}); continue

        candidates = []
        for sym, df in signal_data.items():
            if any(p['symbol'] == sym for p in positions): continue
            idx = df.index[df['trade_date'] == dt]
            if len(idx) == 0: continue
            row = df.loc[idx[0]]

            # 做多
            if row['score_long'] >= min_score:
                candidates.append({
                    'symbol': sym, 'direction': 'long',
                    'score': row['score_long'],
                    'entry_price': row['open'],
                })
            # 做空
            if row['score_short'] >= min_score:
                candidates.append({
                    'symbol': sym, 'direction': 'short',
                    'score': row['score_short'],
                    'entry_price': row['open'],
                })

        # 同品种取高分方向
        sym_best = {}
        for c in candidates:
            if c['symbol'] not in sym_best or c['score'] > sym_best[c['symbol']]['score']:
                sym_best[c['symbol']] = c

        sorted_cands = sorted(sym_best.values(), key=lambda x: -x['score'])
        for cand in sorted_cands[:n_open]:
            if fixed_notional:
                notional = fixed_notional
            else:
                notional = capital * leverage / max_pos

            positions.append({
                'symbol': cand['symbol'], 'direction': cand['direction'],
                'entry_date': dt, 'entry_price': cand['entry_price'],
                'notional': notional, 'score': cand['score'],
            })

        eq_curve.append({'date': dt, 'capital': capital})

    return eq_curve, trades


def print_results(eq, trades, label):
    eq_df = pd.DataFrame(eq)
    if len(eq_df) == 0 or eq_df['capital'].iloc[-1] <= 0:
        print(f"  {label}: 爆仓"); return

    eq_df['peak'] = eq_df['capital'].cummax()
    eq_df['dd'] = (eq_df['capital'] - eq_df['peak']) / eq_df['peak'] * 100

    total_ret = (eq_df['capital'].iloc[-1] / eq_df['capital'].iloc[0] - 1) * 100
    n_years = max((eq_df['date'].iloc[-1] - eq_df['date'].iloc[0]).days / 365.25, 0.01)
    annual = ((eq_df['capital'].iloc[-1] / eq_df['capital'].iloc[0]) ** (1/n_years) - 1) * 100
    mdd = eq_df['dd'].min()

    daily_ret = eq_df['capital'].pct_change().dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * (252**0.5) if len(daily_ret) > 0 and daily_ret.std() > 0 else 0

    if len(trades) > 0:
        tdf = pd.DataFrame(trades)
        wr = (tdf['pnl_pct'] > 0).mean() * 100
        avg = tdf['pnl_pct'].mean()
        avg_w = tdf[tdf['pnl_pct'] > 0]['pnl_pct'].mean() if (tdf['pnl_pct'] > 0).any() else 0
        avg_l = tdf[tdf['pnl_pct'] <= 0]['pnl_pct'].mean() if (tdf['pnl_pct'] <= 0).any() else 0
        pf = abs(avg_w * (tdf['pnl_pct'] > 0).sum() / (avg_l * (tdf['pnl_pct'] <= 0).sum())) if avg_l != 0 and (tdf['pnl_pct'] <= 0).sum() > 0 else 999

        long_t = tdf[tdf['direction'] == 'long']
        short_t = tdf[tdf['direction'] == 'short']
        l_wr = (long_t['pnl_pct'] > 0).mean() * 100 if len(long_t) > 0 else 0
        l_avg = long_t['pnl_pct'].mean() if len(long_t) > 0 else 0
        s_wr = (short_t['pnl_pct'] > 0).mean() * 100 if len(short_t) > 0 else 0
        s_avg = short_t['pnl_pct'].mean() if len(short_t) > 0 else 0
        l_n = len(long_t)
        s_n = len(short_t)

        tdf['year'] = pd.to_datetime(tdf['exit_date']).dt.year

        # Exit reasons
        if 'exit_reason' in tdf.columns:
            reasons = tdf['exit_reason'].value_counts()
            reason_str = '  '.join([f"{r}:{reasons[r]}" for r in reasons.index])
        else:
            reason_str = ""
    else:
        wr = avg = pf = l_wr = s_wr = l_avg = s_avg = 0; l_n = s_n = 0
        tdf = pd.DataFrame(); reason_str = ""

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  总交易:{len(trades)} (多:{l_n} 空:{s_n})")
    print(f"  总WR:{wr:.1f}% 多:{l_wr:.1f}% 空:{s_wr:.1f}%")
    print(f"  PF:{pf:.2f} Sharpe:{sharpe:.2f} Avg:{avg:+.3f}%")
    print(f"  年化:{annual:.0f}% MDD:{mdd:.1f}%")
    if reason_str:
        print(f"  平仓: {reason_str}")
    if len(tdf) > 0 and 'year' in tdf.columns:
        print(f"  按年:")
        for yr in sorted(tdf['year'].unique()):
            sub = tdf[tdf['year'] == yr]
            ywr = (sub['pnl_pct'] > 0).mean() * 100
            yavg = sub['pnl_pct'].mean()
            ylong = sub[sub['direction'] == 'long']
            yshort = sub[sub['direction'] == 'short']
            print(f"    {yr}: N={len(sub):4d}(多{len(ylong)}/空{len(yshort)}) WR={ywr:.1f}% Avg={yavg:+.3f}%")

    return {'annual': annual, 'mdd': mdd, 'wr': wr, 'sharpe': sharpe, 'n': len(trades)}


def main():
    print("V72: 最终优化策略 (多空双向+趋势过滤)")
    print("="*60)

    print("加载数据...")
    all_data = load_data()

    print("计算信号...")
    signal_data = compute_signals(all_data)

    # ═══ 参数扫描 ═══
    print(f"\n{'='*60}")
    print("A. 参数扫描: min_score × hold_days × leverage")
    print(f"{'='*60}")

    for min_sc in [7, 8, 9]:
        for hd in [2, 3]:
            for lev in [3, 5]:
                eq, trades = run_final_backtest(signal_data, '2022-01-01', '2025-12-31',
                                                  min_score=min_sc, hold_days=hd, leverage=lev)
                label = f"min={min_sc} H={hd}d lev={lev}x"
                print_results(eq, trades, label)

    # ═══ 止损止盈优化 ═══
    print(f"\n\n{'='*60}")
    print("B. 止损止盈 (min=8, H=2d, lev=3)")
    print(f"{'='*60}")

    for sl in [None, -2, -3]:
        for tp in [None, 3, 5]:
            eq, trades = run_final_backtest(signal_data, '2022-01-01', '2025-12-31',
                                              min_score=8, hold_days=2, leverage=3,
                                              stop_loss=sl, take_profit=tp)
            sl_s = f"SL={sl}" if sl else "NoSL"
            tp_s = f"TP={tp}" if tp else "NoTP"
            print_results(eq, trades, f"{sl_s} {tp_s}")

    # ═══ 固定名义金额 ═══
    print(f"\n\n{'='*60}")
    print("C. 固定名义金额 (不复利)")
    print(f"{'='*60}")

    for notional in [500_000, 1_000_000, 2_000_000]:
        eq, trades = run_final_backtest(signal_data, '2022-01-01', '2025-12-31',
                                          min_score=8, hold_days=2, leverage=3,
                                          fixed_notional=notional)
        print_results(eq, trades, f"固定名义{notional//10000}万")

    # ═══ Walk-Forward ═══
    print(f"\n\n{'='*60}")
    print("D. Walk-Forward 验证")
    print(f"{'='*60}")

    # 最佳参数: min=8, H=2, lev=3, SL=-3, TP=5
    for sl, tp in [(None, None), (-3, 5), (-2, 3)]:
        sl_s = f"SL={sl}" if sl else "NoSL"
        tp_s = f"TP={tp}" if tp else "NoTP"
        print(f"\n--- {sl_s} {tp_s} ---")
        eq_train, trades_train = run_final_backtest(signal_data, '2015-01-01', '2021-12-31',
                                                      min_score=8, hold_days=2, leverage=3,
                                                      stop_loss=sl, take_profit=tp)
        print_results(eq_train, trades_train, f"训练期(2015-2021)")

        eq_test, trades_test = run_final_backtest(signal_data, '2022-01-01', '2025-12-31',
                                                    min_score=8, hold_days=2, leverage=3,
                                                    stop_loss=sl, take_profit=tp)
        print_results(eq_test, trades_test, f"测试期(2022-2025)")

    # ═══ 年度详细分析 ═══
    print(f"\n\n{'='*60}")
    print("E. 最佳配置: 年度详细分析")
    print(f"{'='*60}")

    # 选择最佳: min=8, H=2, lev=3, SL=-2, TP=3
    eq, trades = run_final_backtest(signal_data, '2015-01-01', '2025-12-31',
                                      min_score=8, hold_days=2, leverage=3,
                                      stop_loss=-2, take_profit=3)

    if len(trades) > 0:
        tdf = pd.DataFrame(trades)
        tdf['year'] = pd.to_datetime(tdf['exit_date']).dt.year
        print(f"\n  全周期(2015-2025):")
        for yr in sorted(tdf['year'].unique()):
            sub = tdf[tdf['year'] == yr]
            wr = (sub['pnl_pct'] > 0).mean() * 100
            avg = sub['pnl_pct'].mean()
            med = sub['pnl_pct'].median()
            n_long = len(sub[sub['direction'] == 'long'])
            n_short = len(sub[sub['direction'] == 'short'])
            print(f"    {yr}: N={len(sub):4d}(多{n_long}/空{n_short}) "
                  f"WR={wr:.1f}% Avg={avg:+.3f}% Med={med:+.3f}%")

        # 按score分组
        print(f"\n  按信号得分:")
        for sc in sorted(tdf['score'].unique(), reverse=True):
            sub = tdf[tdf['score'] == sc]
            swr = (sub['pnl_pct'] > 0).mean() * 100
            savg = sub['pnl_pct'].mean()
            print(f"    score={int(sc):2d}: N={len(sub):4d} WR={swr:.1f}% Avg={savg:+.3f}%")

        # 最大连续亏损
        tdf = tdf.sort_values('exit_date')
        losses = (tdf['pnl_pct'] <= 0).astype(int)
        max_streak = 0
        cur_streak = 0
        for l in losses:
            if l: cur_streak += 1; max_streak = max(max_streak, cur_streak)
            else: cur_streak = 0
        print(f"\n  最大连续亏损: {max_streak}笔")

        # 按方向
        long_t = tdf[tdf['direction'] == 'long']
        short_t = tdf[tdf['direction'] == 'short']
        print(f"  做多: N={len(long_t)} WR={(long_t['pnl_pct']>0).mean()*100:.1f}% Avg={long_t['pnl_pct'].mean():+.3f}%")
        print(f"  做空: N={len(short_t)} WR={(short_t['pnl_pct']>0).mean()*100:.1f}% Avg={short_t['pnl_pct'].mean():+.3f}%")


if __name__ == '__main__':
    main()
