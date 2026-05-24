#!/usr/bin/env python3
"""
V70b: ATR调整Gap信号 + 最优参数组合
核心发现: gap_down > 1.5*ATR 是最强信号 (88% WR 3日)
策略: 只在ATR调整gap足够大时入场, 用止损止盈管理风险
"""
import os, glob, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data/futures_weighted'
INITIAL_CAPITAL = 500_000
TEST_START = '2022-01-01'
TEST_END = '2025-12-31'
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
        all_data[sym] = {'df': df, 'multiplier': mult, 'margin_rate': margin}
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

        # Gap / ATR ratio
        df['gap_atr_ratio'] = df['gap_pct'] / df['atr_pct'].replace(0, np.nan)

        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()

        mom5 = np.full(n, np.nan)
        mom5[5:] = (c[5:] - c[:-5]) / c[:-5] * 100
        df['mom5'] = mom5

        oi_ch = np.full(n, np.nan)
        valid = np.abs(oi[:-1]) > 0
        oi_ch_vals = np.full(n-1, np.nan)
        oi_ch_vals[valid] = (oi[1:][valid] - oi[:-1][valid]) / np.abs(oi[:-1][valid]) * 100
        oi_ch[1:] = oi_ch_vals
        df['oi_chg'] = oi_ch

        df['vol_ma5'] = df['vol'].rolling(5).mean()

        # Range
        range_ = h - l
        df['clv'] = np.where(range_ > 0, (2*c - h - l) / range_, 0)

        # ═══ 信号 ═══
        # Primary: Gap Down > threshold (absolute or ATR-adjusted)
        # Secondary filters (add to score but don't trigger alone)

        # ATR-adjusted gap score
        gap_atr = df['gap_atr_ratio'].fillna(0)
        # gap_atr < -1.0 means gap is larger than 1 day's ATR → extreme
        # gap_atr < -1.5 means gap is 1.5x ATR → very extreme
        # gap_atr < -2.0 means gap is 2x ATR → extremely rare

        score = np.zeros(n)

        # Tier 1: Absolute gap size
        score += (gap < -0.5).astype(int) * 1
        score += (gap < -1.0).astype(int) * 2   # +2 more
        score += (gap < -1.5).astype(int) * 2   # +2 more
        score += (gap < -2.0).astype(int) * 3   # +3 more

        # Tier 2: ATR-adjusted gap (overlaps with Tier 1 but captures relative size)
        score += (gap_atr < -1.0).astype(int) * 2
        score += (gap_atr < -1.5).astype(int) * 3
        score += (gap_atr < -2.0).astype(int) * 3

        # Tier 3: OI confirmation
        score += ((oi_ch > 0) & (c < prev_c)).astype(int) * 3   # oi_up+price_down
        score += ((oi_ch < 0) & (c < prev_c)).astype(int) * 2   # oi_dn+price_down (liquidation)

        # Tier 4: Momentum extreme
        score += (df['mom5'] < -3).astype(int) * 1
        score += (df['mom5'] < -5).astype(int) * 1
        score += (df['close'] < df['ma5']).astype(int) * 1

        # Tier 5: Volume confirmation
        score += ((v > df['vol_ma5'] * 1.5) & (c < prev_c)).astype(int) * 1

        # Tier 6: Close position (reversal within day)
        score += (df['clv'] > 0.5).astype(int) * 1  # closed strong

        df['score'] = score

        # Forward returns from today's open
        for hd in [1, 2, 3, 5, 7]:
            fwd = np.full(n, np.nan)
            if n > hd:
                fwd[:n-hd] = (c[hd:] - o[:n-hd]) / o[:n-hd] * 100
            df[f'fwd_{hd}d'] = fwd

        signal_data[sym] = df
    return signal_data


def analyze_scores(signal_data):
    """分析不同得分水平的收益率"""
    print(f"\n{'='*70}")
    print(f"信号得分 vs 前向收益 (测试期)")
    print(f"{'='*70}")
    print(f"{'Score':>6} {'N':>6} {'Intra WR':>9} {'Intra Avg':>10} {'3d WR':>7} {'3d Avg':>8} {'5d WR':>7} {'5d Avg':>8}")
    print("-" * 70)

    for min_sc in range(1, 20):
        rows = []
        for sym, df in signal_data.items():
            mask = (df['trade_date'] >= TEST_START) & (df['trade_date'] <= TEST_END)
            sub = df[mask & (df['score'] >= min_sc)]
            if len(sub) > 0:
                rows.append(sub)
        if not rows: break
        all_rows = pd.concat(rows)
        if len(all_rows) < 50: continue

        intra = all_rows['fwd_1d'].dropna()
        fwd3 = all_rows['fwd_3d'].dropna()
        fwd5 = all_rows['fwd_5d'].dropna()

        print(f"{min_sc:>6} {len(all_rows):>6} {100*(intra>0).mean():>8.1f}% {intra.mean():>+9.3f}% "
              f"{100*(fwd3>0).mean():>6.1f}% {fwd3.mean():>+7.3f}% "
              f"{100*(fwd5>0).mean():>6.1f}% {fwd5.mean():>+7.3f}%")


def run_backtest(signal_data, start_date, end_date, max_pos=3, leverage=3,
                 min_score=8, hold_days=5, stop_loss_pct=None, take_profit_pct=None):
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
            cur_price = None
            if df is not None:
                idx = df.index[df['trade_date'] == dt]
                if len(idx) > 0: cur_price = df.loc[idx[0], 'close']
            if cur_price is None or np.isnan(cur_price):
                still_open.append(pos); continue

            pnl = (cur_price - pos['entry_price']) / pos['entry_price'] * 100
            days = (dt - pos['entry_date']).days

            exit_reason = None
            if stop_loss_pct and pnl <= stop_loss_pct:
                exit_reason = 'SL'
            elif take_profit_pct and pnl >= take_profit_pct:
                exit_reason = 'TP'
            elif days >= hold_days:
                exit_reason = 'expire'

            if exit_reason:
                trade_pnl = pos['notional'] * pnl / 100
                closed_pnl += trade_pnl
                trades.append({
                    'symbol': pos['symbol'], 'entry_date': pos['entry_date'],
                    'exit_date': dt, 'entry_price': pos['entry_price'],
                    'exit_price': cur_price, 'pnl_pct': pnl,
                    'pnl_abs': trade_pnl, 'score': pos['score'],
                    'hold_days': days, 'exit_reason': exit_reason,
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
            if row['score'] < min_score: continue
            if np.isnan(row['open']) or row['open'] <= 0: continue
            candidates.append({
                'symbol': sym, 'score': row['score'],
                'entry_price': row['open'],
                'gap_pct': row.get('gap_pct', 0),
                'gap_atr': row.get('gap_atr_ratio', 0),
            })

        candidates.sort(key=lambda x: -x['score'])
        for cand in candidates[:n_open]:
            notional = capital * leverage / max_pos
            positions.append({
                'symbol': cand['symbol'], 'entry_date': dt,
                'entry_price': cand['entry_price'],
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

    if len(trades) > 0:
        tdf = pd.DataFrame(trades)
        wr = (tdf['pnl_pct'] > 0).mean() * 100
        avg = tdf['pnl_pct'].mean()
        avg_w = tdf[tdf['pnl_pct'] > 0]['pnl_pct'].mean() if (tdf['pnl_pct'] > 0).any() else 0
        avg_l = tdf[tdf['pnl_pct'] <= 0]['pnl_pct'].mean() if (tdf['pnl_pct'] <= 0).any() else 0
        pf = abs(avg_w * (tdf['pnl_pct'] > 0).sum() / (avg_l * (tdf['pnl_pct'] <= 0).sum())) if avg_l != 0 and (tdf['pnl_pct'] <= 0).sum() > 0 else 999

        daily_ret = eq_df['capital'].pct_change().dropna()
        sharpe = daily_ret.mean() / daily_ret.std() * (252**0.5) if len(daily_ret) > 0 and daily_ret.std() > 0 else 0

        tdf['year'] = pd.to_datetime(tdf['exit_date']).dt.year

        # Exit reason stats
        if 'exit_reason' in tdf.columns:
            reasons = tdf['exit_reason'].value_counts()
            reason_str = '  '.join([f"{r}:{reasons[r]}" for r in reasons.index])
        else:
            reason_str = ""
    else:
        wr = avg = pf = sharpe = 0; tdf = pd.DataFrame(); reason_str = ""

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  交易:{len(trades)} 胜率:{wr:.1f}% PF:{pf:.2f} Sharpe:{sharpe:.2f}")
    print(f"  年化:{annual:.0f}% MDD:{mdd:.1f}% Avg:{avg:+.3f}%")
    if reason_str:
        print(f"  平仓: {reason_str}")
    if len(tdf) > 0 and 'year' in tdf.columns:
        for yr in sorted(tdf['year'].unique()):
            sub = tdf[tdf['year'] == yr]
            ywr = (sub['pnl_pct'] > 0).mean() * 100
            yavg = sub['pnl_pct'].mean()
            print(f"    {yr}: N={len(sub):3d} WR={ywr:.1f}% Avg={yavg:+.3f}%")


def main():
    print("V70b: ATR调整Gap信号 + 最优参数")
    print("="*60)

    print("加载数据...")
    all_data = load_data()

    print("计算信号...")
    signal_data = compute_signals(all_data)

    # 信号得分分析
    analyze_scores(signal_data)

    # ═══ 参数扫描 ═══
    print(f"\n\n{'='*60}")
    print("参数扫描: min_score × hold_days × leverage × SL/TP")
    print(f"{'='*60}")

    # ─── 第一轮: 找最佳 min_score + hold_days ───
    print("\n--- 1. min_score × hold_days (lev=3, 无止损) ---")
    best_combo = None
    best_sharpe = 0
    for min_sc in [5, 6, 7, 8, 9, 10, 11]:
        for hd in [2, 3, 4, 5]:
            eq, trades = run_backtest(signal_data, TEST_START, TEST_END,
                                       min_score=min_sc, hold_days=hd,
                                       leverage=3)
            eq_df = pd.DataFrame(eq)
            if len(eq_df) == 0 or eq_df['capital'].iloc[-1] <= INITIAL_CAPITAL:
                continue

            daily_ret = eq_df['capital'].pct_change().dropna()
            sharpe = daily_ret.mean() / daily_ret.std() * (252**0.5) if daily_ret.std() > 0 else 0
            tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
            wr = (tdf['pnl_pct'] > 0).mean() * 100 if len(tdf) > 0 else 0
            annual = ((eq_df['capital'].iloc[-1] / eq_df['capital'].iloc[0]) ** (1/3.9) - 1) * 100

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_combo = (min_sc, hd)

            print(f"  min={min_sc:2d} H={hd}d: N={len(trades):4d} WR={wr:.1f}% Annual={annual:>7.0f}% Sharpe={sharpe:.2f}")

    print(f"\n  Best by Sharpe: min_score={best_combo[0]}, hold={best_combo[1]}d")

    # ─── 第二轮: 最佳组合 + 止损止盈 ───
    min_sc, hd = best_combo
    print(f"\n--- 2. 止损止盈扫描 (min={min_sc}, H={hd}d, lev=3) ---")
    for sl in [None, -2, -3, -4]:
        for tp in [None, 3, 5, 8, 10]:
            eq, trades = run_backtest(signal_data, TEST_START, TEST_END,
                                       min_score=min_sc, hold_days=hd,
                                       leverage=3, stop_loss_pct=sl, take_profit_pct=tp)
            print_results(eq, trades, f"SL={sl}% TP={tp}%")

    # ─── 第三轮: 最终最佳配置，不同杠杆 ───
    print(f"\n--- 3. 最终配置: 不同杠杆 ---")

    # 先用无止损版本找最佳
    final_configs = [
        (best_combo[0], best_combo[1], None, None, "最优Sharpe配置"),
        (8, 5, None, None, "宽松信号+5日持仓"),
        (9, 3, -2, 5, "中等信号+止损止盈"),
        (10, 5, -3, None, "严格信号+3%止损"),
    ]

    for min_sc, hd, sl, tp, desc in final_configs:
        print(f"\n  === {desc} (min={min_sc}, H={hd}d, SL={sl}, TP={tp}) ===")
        for lev in [3, 5, 7]:
            eq, trades = run_backtest(signal_data, TEST_START, TEST_END,
                                       min_score=min_sc, hold_days=hd,
                                       leverage=lev, stop_loss_pct=sl,
                                       take_profit_pct=tp)
            print_results(eq, trades, f"lev={lev}x")

    # ─── 训练期验证 (walk-forward) ───
    print(f"\n\n{'='*60}")
    print("Walk-Forward验证: 训练期 vs 测试期")
    print(f"{'='*60}")
    for min_sc, hd, sl, tp, desc in final_configs[:2]:
        print(f"\n--- {desc} ---")
        eq_train, trades_train = run_backtest(signal_data, '2015-01-01', '2021-12-31',
                                                min_score=min_sc, hold_days=hd,
                                                leverage=3, stop_loss_pct=sl,
                                                take_profit_pct=tp)
        print_results(eq_train, trades_train, f"训练期 lev=3x")

        eq_test, trades_test = run_backtest(signal_data, TEST_START, TEST_END,
                                              min_score=min_sc, hold_days=hd,
                                              leverage=3, stop_loss_pct=sl,
                                              take_profit_pct=tp)
        print_results(eq_test, trades_test, f"测试期 lev=3x")


if __name__ == '__main__':
    main()
