#!/usr/bin/env python3
"""
V71: 深度探索
1. 品种分层: 哪些品种gap fade最强? 是否存在品种选择alpha?
2. 做空优化: gap_up fade的反向做空
3. 自适应得分: 动态调整min_score
4. 相关性过滤: 避免关联品种同向持仓
5. 趋势过滤: 只在趋势方向做gap fade
6. 月度效应: 某些月份是否更强
7. 流动性过滤: 排除低流动性品种
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
        df['vol_ma20'] = df['vol'].rolling(20).mean()

        # 流动性指标
        df['avg_amount'] = df['amount'].rolling(20).mean()  # 20日均成交额

        # 趋势方向
        df['trend_up'] = df['ma20'] > df['ma60']  # 中期上升趋势
        df['trend_down'] = df['ma20'] < df['ma60']

        range_ = h - l
        df['clv'] = np.where(range_ > 0, (2*c - h - l) / range_, 0)

        # ═══ 做多信号 ═══
        score_long = np.zeros(n)
        gap_atr = df['gap_atr'].fillna(0)
        gap_vals = df['gap_pct'].fillna(0)

        score_long += (gap_vals < -0.5).astype(int) * 1
        score_long += (gap_vals < -1.0).astype(int) * 2
        score_long += (gap_vals < -1.5).astype(int) * 2
        score_long += (gap_vals < -2.0).astype(int) * 3
        score_long += (gap_atr < -1.0).astype(int) * 2
        score_long += (gap_atr < -1.5).astype(int) * 3
        score_long += (gap_atr < -2.0).astype(int) * 3
        score_long += ((oi_ch > 0) & (c < prev_c)).astype(int) * 3
        score_long += ((oi_ch < 0) & (c < prev_c)).astype(int) * 2
        score_long += (df['mom5'] < -3).astype(int) * 1
        score_long += (df['mom5'] < -5).astype(int) * 1
        score_long += (c < df['ma5']).astype(int) * 1
        score_long += ((v > df['vol_ma5'] * 1.5) & (c < prev_c)).astype(int) * 1
        score_long += (df['clv'] > 0.5).astype(int) * 1
        df['score_long'] = score_long

        # ═══ 做空信号 (gap_up fade) ═══
        score_short = np.zeros(n)
        score_short += (gap_vals > 0.5).astype(int) * 1
        score_short += (gap_vals > 1.0).astype(int) * 2
        score_short += (gap_vals > 1.5).astype(int) * 2
        score_short += (gap_vals > 2.0).astype(int) * 3
        score_short += (gap_atr > 1.0).astype(int) * 2
        score_short += (gap_atr > 1.5).astype(int) * 3
        score_short += ((oi_ch > 0) & (c > prev_c)).astype(int) * 3  # 新空头入场
        score_short += ((oi_ch < 0) & (c > prev_c)).astype(int) * 2  # 多头平仓
        score_short += (df['mom5'] > 3).astype(int) * 1
        score_short += (df['mom5'] > 5).astype(int) * 1
        score_short += (c > df['ma5']).astype(int) * 1
        score_short += ((v > df['vol_ma5'] * 1.5) & (c > prev_c)).astype(int) * 1
        score_short += (df['clv'] < -0.5).astype(int) * 1
        df['score_short'] = score_short

        # Forward returns
        for hd in [1, 2, 3, 5]:
            fwd = np.full(n, np.nan)
            if n > hd: fwd[:n-hd] = (c[hd:] - o[:n-hd]) / o[:n-hd] * 100
            df[f'fwd_{hd}d'] = fwd

        signal_data[sym] = df
    return signal_data


def explore_per_commodity(signal_data):
    """1. 品种分层分析"""
    print(f"\n{'='*70}")
    print("1. 品种分层分析: Gap Fade效果最强的品种")
    print(f"{'='*70}")

    results = []
    for sym, df in signal_data.items():
        mask = (df['trade_date'] >= TEST_START) & (df['trade_date'] <= TEST_END)
        sub = df[mask & (df['score_long'] >= 7)]
        if len(sub) < 10: continue

        intra = sub['fwd_1d'].dropna()
        fwd2 = sub['fwd_2d'].dropna()
        fwd3 = sub['fwd_3d'].dropna()

        results.append({
            'symbol': sym,
            'trades': len(sub),
            'intra_wr': 100*(intra>0).mean() if len(intra)>0 else 0,
            'intra_avg': intra.mean() if len(intra)>0 else 0,
            'fwd2_wr': 100*(fwd2>0).mean() if len(fwd2)>0 else 0,
            'fwd2_avg': fwd2.mean() if len(fwd2)>0 else 0,
            'fwd3_wr': 100*(fwd3>0).mean() if len(fwd3)>0 else 0,
            'fwd3_avg': fwd3.mean() if len(fwd3)>0 else 0,
            'avg_amount': sub['avg_amount'].mean(),
        })

    rdf = pd.DataFrame(results)

    # 按平均收益排序
    print("\n--- Top 20 品种 (按2日前向平均收益) ---")
    top20 = rdf.nlargest(20, 'fwd2_avg')
    print(f"{'Symbol':>8} {'N':>4} {'2d_WR':>6} {'2d_Avg':>7} {'3d_WR':>6} {'3d_Avg':>7} {'AvgAmount':>10}")
    for _, r in top20.iterrows():
        print(f"{r['symbol']:>8} {int(r['trades']):>4} {r['fwd2_wr']:>5.1f}% {r['fwd2_avg']:>+6.3f}% "
              f"{r['fwd3_wr']:>5.1f}% {r['fwd3_avg']:>+6.3f}% {r['avg_amount']:>10.0f}")

    print("\n--- Bottom 10 品种 (最差) ---")
    bot10 = rdf.nsmallest(10, 'fwd2_avg')
    for _, r in bot10.iterrows():
        print(f"{r['symbol']:>8} {int(r['trades']):>4} {r['fwd2_wr']:>5.1f}% {r['fwd2_avg']:>+6.3f}%")

    # 按流动性分组
    print("\n--- 按流动性分组 ---")
    rdf['liq_group'] = pd.qcut(rdf['avg_amount'], 3, labels=['低', '中', '高'])
    for grp in ['低', '中', '高']:
        sub = rdf[rdf['liq_group'] == grp]
        print(f"  {grp}流动性: {len(sub)}品种, 2d_WR={sub['fwd2_wr'].mean():.1f}%, "
              f"2d_Avg={sub['fwd2_avg'].mean():+.3f}%, N_avg={sub['trades'].mean():.0f}")

    return top20['symbol'].tolist()


def explore_short_side(signal_data):
    """2. 做空gap_up fade"""
    print(f"\n{'='*70}")
    print("2. 做空: Gap Up Fade")
    print(f"{'='*70}")

    # 做空信号统计
    for min_sc in [7, 9, 11]:
        rows = []
        for sym, df in signal_data.items():
            mask = (df['trade_date'] >= TEST_START) & (df['trade_date'] <= TEST_END)
            sub = df[mask & (df['score_short'] >= min_sc)]
            if len(sub) > 0: rows.append(sub)
        if not rows: continue
        all_rows = pd.concat(rows)

        fwd1 = all_rows['fwd_1d'].dropna()
        fwd2 = all_rows['fwd_2d'].dropna()
        fwd3 = all_rows['fwd_3d'].dropna()

        # 做空收益 = 反转 (负的forward return)
        print(f"\n  Gap Up score>={min_sc} (N={len(all_rows)})")
        print(f"    1d: ShortWR={100*(fwd1<0).mean():.1f}% ShortAvg={-fwd1.mean():+.3f}%")
        print(f"    2d: ShortWR={100*(fwd2<0).mean():.1f}% ShortAvg={-fwd2.mean():+.3f}%")
        print(f"    3d: ShortWR={100*(fwd3<0).mean():.1f}% ShortAvg={-fwd3.mean():+.3f}%")

    # 多空组合回测
    print(f"\n  === 多空组合回测 ===")
    date_range = pd.date_range(start=TEST_START, end=TEST_END, freq='B')
    for min_sc in [7, 9]:
        for hold in [2, 3]:
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
                    days = (dt - pos['entry_date']).days
                    if days >= hold:
                        if pos['direction'] == 'long':
                            pnl = (cur - pos['entry_price']) / pos['entry_price'] * 100
                        else:
                            pnl = (pos['entry_price'] - cur) / pos['entry_price'] * 100
                        trade_pnl = pos['notional'] * pnl / 100
                        closed_pnl += trade_pnl
                        trades.append({
                            'symbol': pos['symbol'], 'direction': pos['direction'],
                            'entry_date': pos['entry_date'], 'exit_date': dt,
                            'pnl_pct': pnl, 'pnl_abs': trade_pnl, 'score': pos['score'],
                        })
                    else:
                        still_open.append(pos)
                positions = still_open
                capital += closed_pnl
                if capital <= 0:
                    eq_curve.append({'date': dt, 'capital': 0}); break

                # 开仓
                n_open = 3 - len(positions)
                if n_open <= 0:
                    eq_curve.append({'date': dt, 'capital': capital}); continue

                candidates = []
                for sym, df in signal_data.items():
                    if any(p['symbol'] == sym for p in positions): continue
                    idx = df.index[df['trade_date'] == dt]
                    if len(idx) == 0: continue
                    row = df.loc[idx[0]]

                    # 做多候选
                    if row['score_long'] >= min_sc:
                        candidates.append({
                            'symbol': sym, 'direction': 'long',
                            'score': row['score_long'],
                            'entry_price': row['open'],
                        })
                    # 做空候选
                    if row['score_short'] >= min_sc:
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
                    notional = capital * 3 / 3
                    positions.append({
                        'symbol': cand['symbol'], 'direction': cand['direction'],
                        'entry_date': dt, 'entry_price': cand['entry_price'],
                        'notional': notional, 'score': cand['score'],
                    })

                eq_curve.append({'date': dt, 'capital': capital})

            # 分析结果
            eq_df = pd.DataFrame(eq_curve)
            if len(eq_df) == 0 or eq_df['capital'].iloc[-1] <= 0:
                print(f"  多空 min={min_sc} H={hold}d: 爆仓"); continue

            tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
            wr = (tdf['pnl_pct'] > 0).mean() * 100 if len(tdf) > 0 else 0
            avg = tdf['pnl_pct'].mean() if len(tdf) > 0 else 0
            long_t = tdf[tdf['direction'] == 'long']
            short_t = tdf[tdf['direction'] == 'short']
            l_wr = (long_t['pnl_pct'] > 0).mean() * 100 if len(long_t) > 0 else 0
            s_wr = (short_t['pnl_pct'] > 0).mean() * 100 if len(short_t) > 0 else 0

            eq_df['peak'] = eq_df['capital'].cummax()
            mdd = ((eq_df['capital'] - eq_df['peak']) / eq_df['peak'] * 100).min()
            annual = ((eq_df['capital'].iloc[-1] / eq_df['capital'].iloc[0]) ** (1/3.9) - 1) * 100

            print(f"  多空 min={min_sc} H={hold}d: N={len(trades)} Total={len(tdf)} "
                  f"WR={wr:.1f}%(多{l_wr:.0f}%/空{s_wr:.0f}%) Annual={annual:.0f}% MDD={mdd:.1f}%")


def explore_trend_filter(signal_data):
    """3. 趋势过滤: 只在趋势方向做gap fade"""
    print(f"\n{'='*70}")
    print("3. 趋势过滤")
    print(f"{'='*70}")

    combos = [
        ('无过滤', lambda r: True),
        ('上升趋势gap_dn', lambda r: r['gap_pct'] < -1 and r.get('trend_up', False)),
        ('下降趋势gap_dn', lambda r: r['gap_pct'] < -1 and r.get('trend_down', False)),
        ('gap_dn+趋势向上', lambda r: r['score_long'] >= 7 and r.get('trend_up', False)),
        ('gap_dn+趋势向下', lambda r: r['score_long'] >= 7 and r.get('trend_down', False)),
        ('gap_dn+MA20向上', lambda r: r['score_long'] >= 7 and r.get('mom20', 0) > 0),
        ('gap_dn+MA20向下', lambda r: r['score_long'] >= 7 and r.get('mom20', 0) < 0),
    ]

    for name, filt in combos:
        rows = []
        for sym, df in signal_data.items():
            mask = (df['trade_date'] >= TEST_START) & (df['trade_date'] <= TEST_END)
            sub = df[mask].copy()
            # Apply filter
            filtered = sub[sub.apply(filt, axis=1)]
            if len(filtered) > 0: rows.append(filtered)
        if not rows: continue
        all_rows = pd.concat(rows)

        fwd1 = all_rows['fwd_1d'].dropna()
        fwd2 = all_rows['fwd_2d'].dropna()
        fwd3 = all_rows['fwd_3d'].dropna()

        print(f"  {name:25s} N={len(all_rows):5d}  "
              f"1d_WR={100*(fwd1>0).mean():5.1f}% Avg={fwd1.mean():+.3f}%  "
              f"2d_WR={100*(fwd2>0).mean():5.1f}% Avg={fwd2.mean():+.3f}%")


def explore_monthly_seasonality(signal_data):
    """4. 月度效应"""
    print(f"\n{'='*70}")
    print("4. 月度效应 (score>=7, gap down)")
    print(f"{'='*70}")

    for month_name, month_filter in [
        ('全部', None),
        ('1月', 1), ('2月', 2), ('3月', 3), ('4月', 4),
        ('5月', 5), ('6月', 6), ('7月', 7), ('8月', 8),
        ('9月', 9), ('10月', 10), ('11月', 11), ('12月', 12),
    ]:
        rows = []
        for sym, df in signal_data.items():
            mask = (df['trade_date'] >= TEST_START) & (df['trade_date'] <= TEST_END) & (df['score_long'] >= 7)
            if month_filter:
                mask = mask & (df['trade_date'].dt.month == month_filter)
            sub = df[mask]
            if len(sub) > 0: rows.append(sub)
        if not rows: continue
        all_rows = pd.concat(rows)

        fwd2 = all_rows['fwd_2d'].dropna()
        print(f"  {month_name:>4s}: N={len(all_rows):4d} WR={100*(fwd2>0).mean():5.1f}% Avg={fwd2.mean():+.3f}%")


def explore_liquidity_filter(signal_data):
    """5. 流动性过滤"""
    print(f"\n{'='*70}")
    print("5. 流动性过滤")
    print(f"{'='*70}")

    # 计算每个品种的平均成交额
    sym_amounts = {}
    for sym, df in signal_data.items():
        mask = (df['trade_date'] >= TEST_START) & (df['trade_date'] <= TEST_END)
        sym_amounts[sym] = df.loc[mask, 'avg_amount'].mean()

    amount_series = pd.Series(sym_amounts).dropna()
    thresholds = [
        ('全部', 0),
        ('Top50%', amount_series.median()),
        ('Top25%', amount_series.quantile(0.75)),
        ('Top10%', amount_series.quantile(0.9)),
    ]

    for name, threshold in thresholds:
        rows = []
        for sym, df in signal_data.items():
            if sym_amounts.get(sym, 0) < threshold: continue
            mask = (df['trade_date'] >= TEST_START) & (df['trade_date'] <= TEST_END) & (df['score_long'] >= 7)
            sub = df[mask]
            if len(sub) > 0: rows.append(sub)
        if not rows: continue
        all_rows = pd.concat(rows)

        fwd2 = all_rows['fwd_2d'].dropna()
        print(f"  {name:8s}: {len(all_rows):5d}交易 WR={100*(fwd2>0).mean():.1f}% Avg={fwd2.mean():+.3f}%")


def explore_correlation_filter(signal_data):
    """6. 相关性过滤回测"""
    print(f"\n{'='*70}")
    print("6. 相关性过滤: 避免同板块品种同时持仓")
    print(f"{'='*70}")

    # 定义板块
    SECTORS = {
        '黑色': ['rb', 'hc', 'i', 'j', 'jm', 'ZC', 'SF', 'SM', 'fg'],
        '有色': ['cu', 'al', 'zn', 'ni', 'sn', 'pb', 'ao', 'ss', 'SI'],
        '能化': ['sc', 'fu', 'bu', 'pg', 'pp', 'l', 'v', 'TA', 'MA', 'eg', 'eb', 'SA', 'UR', 'PF'],
        '农产品': ['m', 'y', 'a', 'p', 'c', 'cs', 'jd', 'CF', 'SR', 'AP', 'CJ', 'OI', 'RM', 'PK'],
        '贵金属': ['au', 'ag'],
    }

    def get_sector(sym):
        for sector, syms in SECTORS.items():
            if sym in syms or sym.upper() in [s.upper() for s in syms]:
                return sector
        return '其他'

    date_range = pd.date_range(start=TEST_START, end=TEST_END, freq='B')

    for use_corr_filter in [False, True]:
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
                days = (dt - pos['entry_date']).days
                if days >= 2:
                    pnl = (cur - pos['entry_price']) / pos['entry_price'] * 100
                    trade_pnl = pos['notional'] * pnl / 100
                    closed_pnl += trade_pnl
                    trades.append({
                        'symbol': pos['symbol'], 'pnl_pct': pnl,
                        'pnl_abs': trade_pnl, 'score': pos['score'],
                    })
                else:
                    still_open.append(pos)
            positions = still_open
            capital += closed_pnl
            if capital <= 0:
                eq_curve.append({'date': dt, 'capital': 0}); break

            # 开仓
            n_open = 3 - len(positions)
            if n_open <= 0:
                eq_curve.append({'date': dt, 'capital': capital}); continue

            candidates = []
            for sym, df in signal_data.items():
                if any(p['symbol'] == sym for p in positions): continue
                idx = df.index[df['trade_date'] == dt]
                if len(idx) == 0: continue
                row = df.loc[idx[0]]
                if row['score_long'] < 7: continue

                # 相关性过滤: 已持仓品种同板块不超过1个
                if use_corr_filter:
                    held_sectors = [get_sector(p['symbol']) for p in positions]
                    cand_sector = get_sector(sym)
                    if held_sectors.count(cand_sector) >= 1:
                        continue

                candidates.append({
                    'symbol': sym, 'score': row['score_long'],
                    'entry_price': row['open'],
                })

            candidates.sort(key=lambda x: -x['score'])
            for cand in candidates[:n_open]:
                notional = capital * 3 / 3
                positions.append({
                    'symbol': cand['symbol'], 'entry_date': dt,
                    'entry_price': cand['entry_price'],
                    'notional': notional, 'score': cand['score'],
                })

            eq_curve.append({'date': dt, 'capital': capital})

        eq_df = pd.DataFrame(eq_curve)
        if len(eq_df) == 0 or eq_df['capital'].iloc[-1] <= 0:
            print(f"  {'有' if use_corr_filter else '无'}相关性过滤: 爆仓"); continue

        tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
        wr = (tdf['pnl_pct'] > 0).mean() * 100 if len(tdf) > 0 else 0
        avg = tdf['pnl_pct'].mean() if len(tdf) > 0 else 0
        eq_df['peak'] = eq_df['capital'].cummax()
        mdd = ((eq_df['capital'] - eq_df['peak']) / eq_df['peak'] * 100).min()
        annual = ((eq_df['capital'].iloc[-1] / eq_df['capital'].iloc[0]) ** (1/3.9) - 1) * 100

        label = "有相关性过滤" if use_corr_filter else "无过滤"
        print(f"  {label:12s}: N={len(trades)} WR={wr:.1f}% Avg={avg:+.3f}% Annual={annual:.0f}% MDD={mdd:.1f}%")


def explore_top_commodities(signal_data, top_syms):
    """7. 只交易最强品种"""
    print(f"\n{'='*70}")
    print("7. 精选品种回测 (Top10最强品种)")
    print(f"{'='*70}")

    date_range = pd.date_range(start=TEST_START, end=TEST_END, freq='B')

    for n_top in [10, 15, 20]:
        selected = top_syms[:n_top]
        capital = INITIAL_CAPITAL
        eq_curve = []
        trades = []
        positions = []

        for dt in date_range:
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
                days = (dt - pos['entry_date']).days
                if days >= 2:
                    pnl = (cur - pos['entry_price']) / pos['entry_price'] * 100
                    trade_pnl = pos['notional'] * pnl / 100
                    closed_pnl += trade_pnl
                    trades.append({
                        'symbol': pos['symbol'], 'pnl_pct': pnl,
                        'pnl_abs': trade_pnl, 'score': pos['score'],
                    })
                else:
                    still_open.append(pos)
            positions = still_open
            capital += closed_pnl
            if capital <= 0:
                eq_curve.append({'date': dt, 'capital': 0}); break

            n_open = 3 - len(positions)
            if n_open <= 0:
                eq_curve.append({'date': dt, 'capital': capital}); continue

            candidates = []
            for sym in selected:
                if sym not in signal_data: continue
                if any(p['symbol'] == sym for p in positions): continue
                df = signal_data[sym]
                idx = df.index[df['trade_date'] == dt]
                if len(idx) == 0: continue
                row = df.loc[idx[0]]
                if row['score_long'] < 7: continue
                candidates.append({
                    'symbol': sym, 'score': row['score_long'],
                    'entry_price': row['open'],
                })

            candidates.sort(key=lambda x: -x['score'])
            for cand in candidates[:n_open]:
                notional = capital * 3 / 3
                positions.append({
                    'symbol': cand['symbol'], 'entry_date': dt,
                    'entry_price': cand['entry_price'],
                    'notional': notional, 'score': cand['score'],
                })

            eq_curve.append({'date': dt, 'capital': capital})

        eq_df = pd.DataFrame(eq_curve)
        if len(eq_df) == 0 or eq_df['capital'].iloc[-1] <= 0:
            print(f"  Top{n_top}: 爆仓"); continue

        tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
        wr = (tdf['pnl_pct'] > 0).mean() * 100 if len(tdf) > 0 else 0
        avg = tdf['pnl_pct'].mean() if len(tdf) > 0 else 0
        eq_df['peak'] = eq_df['capital'].cummax()
        mdd = ((eq_df['capital'] - eq_df['peak']) / eq_df['peak'] * 100).min()
        annual = ((eq_df['capital'].iloc[-1] / eq_df['capital'].iloc[0]) ** (1/3.9) - 1) * 100

        print(f"  Top{n_top:2d}品种: N={len(trades)} WR={wr:.1f}% Avg={avg:+.3f}% Annual={annual:.0f}% MDD={mdd:.1f}%")


def explore_adaptive_score(signal_data):
    """8. 自适应得分门槛"""
    print(f"\n{'='*70}")
    print("8. 自适应得分门槛 (基于近期胜率)")
    print(f"{'='*70}")

    date_range = pd.date_range(start=TEST_START, end=TEST_END, freq='B')
    capital = INITIAL_CAPITAL
    eq_curve = []
    trades = []
    positions = []
    recent_pnls = []
    current_min = 7  # 初始门槛

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
            days = (dt - pos['entry_date']).days
            if days >= 2:
                pnl = (cur - pos['entry_price']) / pos['entry_price'] * 100
                trade_pnl = pos['notional'] * pnl / 100
                closed_pnl += trade_pnl
                recent_pnls.append(pnl)
                trades.append({
                    'symbol': pos['symbol'], 'pnl_pct': pnl,
                    'pnl_abs': trade_pnl, 'score': pos['score'],
                    'min_score': current_min,
                })
            else:
                still_open.append(pos)
        positions = still_open
        capital += closed_pnl
        if capital <= 0:
            eq_curve.append({'date': dt, 'capital': 0}); break

        # 自适应门槛
        if len(recent_pnls) >= 30:
            recent_wr = np.mean([p > 0 for p in recent_pnls[-30:]])
            if recent_wr > 0.80:
                current_min = 5   # 胜率高, 降低门槛, 更多交易
            elif recent_wr > 0.70:
                current_min = 7   # 正常
            elif recent_wr > 0.60:
                current_min = 9   # 胜率降低, 提高门槛
            else:
                current_min = 12  # 胜率很低, 只交易极端信号

        # 开仓
        n_open = 3 - len(positions)
        if n_open <= 0:
            eq_curve.append({'date': dt, 'capital': capital}); continue

        candidates = []
        for sym, df in signal_data.items():
            if any(p['symbol'] == sym for p in positions): continue
            idx = df.index[df['trade_date'] == dt]
            if len(idx) == 0: continue
            row = df.loc[idx[0]]
            if row['score_long'] < current_min: continue
            candidates.append({
                'symbol': sym, 'score': row['score_long'],
                'entry_price': row['open'],
            })

        candidates.sort(key=lambda x: -x['score'])
        for cand in candidates[:n_open]:
            notional = capital * 3 / 3
            positions.append({
                'symbol': cand['symbol'], 'entry_date': dt,
                'entry_price': cand['entry_price'],
                'notional': notional, 'score': cand['score'],
            })

        eq_curve.append({'date': dt, 'capital': capital})

    eq_df = pd.DataFrame(eq_curve)
    if len(eq_df) > 0 and eq_df['capital'].iloc[-1] > 0:
        tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
        wr = (tdf['pnl_pct'] > 0).mean() * 100 if len(tdf) > 0 else 0
        avg = tdf['pnl_pct'].mean() if len(tdf) > 0 else 0
        eq_df['peak'] = eq_df['capital'].cummax()
        mdd = ((eq_df['capital'] - eq_df['peak']) / eq_df['peak'] * 100).min()
        annual = ((eq_df['capital'].iloc[-1] / eq_df['capital'].iloc[0]) ** (1/3.9) - 1) * 100

        print(f"  自适应门槛: N={len(trades)} WR={wr:.1f}% Avg={avg:+.3f}% Annual={annual:.0f}% MDD={mdd:.1f}%")

        # 按min_score分组
        if 'min_score' in tdf.columns:
            print(f"  各门槛下表现:")
            for ms in sorted(tdf['min_score'].unique()):
                sub = tdf[tdf['min_score'] == ms]
                mwr = (sub['pnl_pct'] > 0).mean() * 100
                print(f"    min={int(ms):2d}: N={len(sub):4d} WR={mwr:.1f}% Avg={sub['pnl_pct'].mean():+.3f}%")


def main():
    print("V71: 深度探索")
    print("="*60)

    print("加载数据...")
    all_data = load_data()

    print("计算信号...")
    signal_data = compute_signals(all_data)

    # 1. 品种分层
    top_syms = explore_per_commodity(signal_data)

    # 2. 做空
    explore_short_side(signal_data)

    # 3. 趋势过滤
    explore_trend_filter(signal_data)

    # 4. 月度效应
    explore_monthly_seasonality(signal_data)

    # 5. 流动性过滤
    explore_liquidity_filter(signal_data)

    # 6. 相关性过滤
    explore_correlation_filter(signal_data)

    # 7. 精选品种
    explore_top_commodities(signal_data, top_syms)

    # 8. 自适应门槛
    explore_adaptive_score(signal_data)


if __name__ == '__main__':
    main()
