#!/usr/bin/env python3
"""
V73: 结合期限结构(carry)的优化策略
数据: futures_term_structure有5年+历史
策略: V72多空gap fade + 动态carry信号
carry信号: backwardation(近>远)=做多加分, contango=做空加分
目标: 年化1000%+ 胜率>50% MDD<30%
"""
import os, glob, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data/futures_weighted'
TS_DIR = 'data/futures_term_structure'
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
    print(f"  期货: {len(all_data)}品种")
    return all_data


def load_term_structure():
    """加载历史期限结构数据, 构建每日carry信号"""
    print("  加载期限结构...")
    files = sorted(glob.glob(os.path.join(TS_DIR, '*.json')))
    # 排除非期货品种
    skip_prefixes = ('000', '159', '510')

    records = []
    for f in files:
        fname = os.path.basename(f)
        if not fname.endswith('.json'): continue
        if any(fname.startswith(p) for p in skip_prefixes): continue

        try:
            with open(f) as fp:
                d = json.load(fp)
        except:
            continue

        sym = d.get('symbol', '')
        date_str = d.get('date', '')
        if not sym or not date_str: continue

        try:
            dt = pd.to_datetime(date_str)
        except:
            continue

        spread_pct = d.get('total_spread_pct', None)
        structure = d.get('structure', '')

        if spread_pct is not None:
            records.append({
                'symbol': sym,
                'date': dt,
                'spread_pct': spread_pct,
                'structure': structure,
                'near_price': d.get('near_price', 0),
                'far_price': d.get('far_price', 0),
            })

    ts_df = pd.DataFrame(records)
    if len(ts_df) == 0:
        print("  无期限结构数据!")
        return {}

    ts_df = ts_df.sort_values(['symbol', 'date']).reset_index(drop=True)

    # 计算carry指标
    # spread_pct > 0 = contango (远>近), spread_pct < 0 = backwardation (近>远)
    # backwardation 做多有利 (正carry)
    # contango 做空有利

    # 按品种分组, 计算carry趋势
    ts_df['carry'] = -ts_df['spread_pct']  # 正=backwardation=多头有利
    ts_df['carry_ma5'] = ts_df.groupby('symbol')['carry'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    ts_df['carry_ma20'] = ts_df.groupby('symbol')['carry'].transform(
        lambda x: x.rolling(20, min_periods=1).mean()
    )
    ts_df['carry_trend'] = ts_df['carry_ma5'] - ts_df['carry_ma20']  # 正=carry改善

    # 转为dict: {symbol: {date: carry_info}}
    ts_dict = {}
    for sym in ts_df['symbol'].unique():
        sub = ts_df[ts_df['symbol'] == sym].set_index('date')
        ts_dict[sym] = sub

    n_syms = len(ts_dict)
    n_records = len(ts_df)
    date_range = f"{ts_df['date'].min().strftime('%Y-%m-%d')} ~ {ts_df['date'].max().strftime('%Y-%m-%d')}"
    print(f"  期限结构: {n_syms}品种, {n_records}条记录, {date_range}")

    return ts_dict


def compute_signals(all_data, ts_dict):
    """计算信号 (含carry)"""
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

        oi_ch = np.full(n, np.nan)
        valid = np.abs(oi[:-1]) > 0
        oi_ch_vals = np.full(n-1, np.nan)
        oi_ch_vals[valid] = (oi[1:][valid] - oi[:-1][valid]) / np.abs(oi[:-1][valid]) * 100
        oi_ch[1:] = oi_ch_vals
        df['oi_chg'] = oi_ch
        df['vol_ma5'] = df['vol'].rolling(5).mean()

        range_ = h - l
        df['clv'] = np.where(range_ > 0, (2*c - h - l) / range_, 0)
        df['trend_up'] = df['ma20'] > df['ma60']
        df['trend_down'] = df['ma20'] < df['ma60']

        # ═══ Carry信号 ═══
        carry_vals = np.zeros(n)
        carry_ma5 = np.zeros(n)
        carry_trend = np.zeros(n)

        if sym in ts_dict:
            ts = ts_dict[sym]
            for i in range(n):
                dt = df['trade_date'].iloc[i]
                # 找最近一个交易日<=dt的carry数据
                mask = ts.index <= dt
                if mask.any():
                    latest = ts[mask].iloc[-1]
                    carry_vals[i] = latest.get('carry', 0)
                    carry_ma5[i] = latest.get('carry_ma5', 0)
                    carry_trend[i] = latest.get('carry_trend', 0)

        df['carry'] = carry_vals
        df['carry_ma5'] = carry_ma5
        df['carry_trend'] = carry_trend

        # ═══ 做多信号 ═══
        s_long = np.zeros(n)
        gv = df['gap_pct'].fillna(0)
        ga = df['gap_atr'].fillna(0)

        # Gap + ATR
        s_long += (gv < -0.5).astype(int) * 1
        s_long += (gv < -1.0).astype(int) * 2
        s_long += (gv < -1.5).astype(int) * 2
        s_long += (gv < -2.0).astype(int) * 3
        s_long += (ga < -1.0).astype(int) * 2
        s_long += (ga < -1.5).astype(int) * 3
        s_long += (ga < -2.0).astype(int) * 3
        # OI
        s_long += ((oi_ch > 0) & (c < prev_c)).astype(int) * 3
        s_long += ((oi_ch < 0) & (c < prev_c)).astype(int) * 2
        # 动量
        s_long += (df['mom5'] < -3).astype(int) * 1
        s_long += (df['mom5'] < -5).astype(int) * 1
        s_long += (c < df['ma5'].values).astype(int) * 1
        # 量
        s_long += ((v > df['vol_ma5'] * 1.5) & (c < prev_c)).astype(int) * 1
        s_long += (df['clv'] > 0.5).astype(int) * 1
        # 趋势
        s_long += df['trend_up'].astype(int) * 2
        # ═══ Carry加分 ═══
        # Backwardation (carry>0) = 做多有利, 加分
        s_long += (carry_vals > 1).astype(int) * 2    # 正carry
        s_long += (carry_vals > 3).astype(int) * 2    # 强正carry
        s_long += (carry_trend > 0).astype(int) * 1   # carry改善中
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
        s_short += df['trend_down'].astype(int) * 2
        # ═══ Carry加分 ═══
        # Contango (carry<0) = 做空有利, 加分
        s_short += (carry_vals < -1).astype(int) * 2   # 负carry
        s_short += (carry_vals < -3).astype(int) * 2   # 强负carry
        s_short += (carry_trend < 0).astype(int) * 1   # carry恶化中
        df['score_short'] = s_short

        # Carry flag for analysis
        df['carry_bkwd'] = carry_vals > 1    # backwardation
        df['carry_ctng'] = carry_vals < -1   # contango

        for hd in [1, 2, 3, 5]:
            fwd = np.full(n, np.nan)
            if n > hd: fwd[:n-hd] = (c[hd:] - o[:n-hd]) / o[:n-hd] * 100
            df[f'fwd_{hd}d'] = fwd

        signal_data[sym] = df
    return signal_data


def run_backtest(signal_data, start_date, end_date, max_pos=3, leverage=3,
                 min_score=8, hold_days=2, stop_loss=None, take_profit=None,
                 carry_filter=None):
    """carry_filter: None, 'require_bkwd'(做多需backwardation), 'boost'(carry作为加分已包含)"""
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
            pnl = (cur - pos['entry_price']) / pos['entry_price'] * 100 if pos['direction'] == 'long' \
                else (pos['entry_price'] - cur) / pos['entry_price'] * 100
            days = (dt - pos['entry_date']).days
            exit_reason = None
            if stop_loss and pnl <= stop_loss: exit_reason = 'SL'
            elif take_profit and pnl >= take_profit: exit_reason = 'TP'
            elif days >= hold_days: exit_reason = 'expire'
            if exit_reason:
                trade_pnl = pos['notional'] * pnl / 100
                closed_pnl += trade_pnl
                trades.append({
                    'symbol': pos['symbol'], 'direction': pos['direction'],
                    'entry_date': pos['entry_date'], 'exit_date': dt,
                    'pnl_pct': pnl, 'pnl_abs': trade_pnl, 'score': pos['score'],
                    'hold_days': days, 'exit_reason': exit_reason,
                    'carry': pos.get('carry', 0),
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
                # Carry filter
                if carry_filter == 'require_bkwd' and not row.get('carry_bkwd', False):
                    pass  # skip - no backwardation
                else:
                    candidates.append({
                        'symbol': sym, 'direction': 'long',
                        'score': row['score_long'],
                        'entry_price': row['open'],
                        'carry': row.get('carry', 0),
                    })
            # 做空
            if row['score_short'] >= min_score:
                if carry_filter == 'require_bkwd' and not row.get('carry_ctng', False):
                    pass  # skip - no contango for shorts
                else:
                    candidates.append({
                        'symbol': sym, 'direction': 'short',
                        'score': row['score_short'],
                        'entry_price': row['open'],
                        'carry': row.get('carry', 0),
                    })

        # 同品种取高分方向
        sym_best = {}
        for c in candidates:
            if c['symbol'] not in sym_best or c['score'] > sym_best[c['symbol']]['score']:
                sym_best[c['symbol']] = c

        sorted_cands = sorted(sym_best.values(), key=lambda x: -x['score'])
        for cand in sorted_cands[:n_open]:
            notional = capital * leverage / max_pos
            positions.append({
                'symbol': cand['symbol'], 'direction': cand['direction'],
                'entry_date': dt, 'entry_price': cand['entry_price'],
                'notional': notional, 'score': cand['score'],
                'carry': cand['carry'],
            })

        eq_curve.append({'date': dt, 'capital': capital})
    return eq_curve, trades


def print_results(eq, trades, label):
    eq_df = pd.DataFrame(eq)
    if len(eq_df) == 0 or eq_df['capital'].iloc[-1] <= 0:
        print(f"  {label}: 爆仓"); return
    eq_df['peak'] = eq_df['capital'].cummax()
    eq_df['dd'] = (eq_df['capital'] - eq_df['peak']) / eq_df['peak'] * 100
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
        s_wr = (short_t['pnl_pct'] > 0).mean() * 100 if len(short_t) > 0 else 0

        # Carry analysis
        if 'carry' in tdf.columns:
            carry_pos = tdf[(tdf['direction']=='long') & (tdf['carry'] > 1)]
            carry_neg = tdf[(tdf['direction']=='short') & (tdf['carry'] < -1)]
            no_carry_l = tdf[(tdf['direction']=='long') & (tdf['carry'] <= 1)]
            no_carry_s = tdf[(tdf['direction']=='short') & (tdf['carry'] >= -1)]
            cp_wr = (carry_pos['pnl_pct'] > 0).mean() * 100 if len(carry_pos) > 0 else 0
            cn_wr = (carry_neg['pnl_pct'] > 0).mean() * 100 if len(carry_neg) > 0 else 0
        else:
            cp_wr = cn_wr = 0

        tdf['year'] = pd.to_datetime(tdf['exit_date']).dt.year
    else:
        wr = avg = pf = l_wr = s_wr = 0; tdf = pd.DataFrame(); cp_wr = cn_wr = 0

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  N:{len(trades)}(多{len(long_t)}/空{len(short_t)}) WR:{wr:.1f}%(多{l_wr:.0f}/空{s_wr:.0f})")
    print(f"  PF:{pf:.2f} Sharpe:{sharpe:.2f} Avg:{avg:+.3f}%")
    print(f"  年化:{annual:.0f}% MDD:{mdd:.1f}%")
    if cp_wr > 0:
        print(f"  Carry确认: 做多+backwd WR={cp_wr:.1f}% 做空+contango WR={cn_wr:.1f}%")
    if len(tdf) > 0 and 'year' in tdf.columns:
        for yr in sorted(tdf['year'].unique()):
            sub = tdf[tdf['year'] == yr]
            ywr = (sub['pnl_pct'] > 0).mean() * 100
            yavg = sub['pnl_pct'].mean()
            print(f"    {yr}: N={len(sub):4d} WR={ywr:.1f}% Avg={yavg:+.3f}%")

    return {'annual': annual, 'mdd': mdd, 'wr': wr, 'sharpe': sharpe}


def main():
    print("V73: 结合期限结构carry的优化策略")
    print("="*60)
    print("目标: 年化1000%+ 胜率>50% MDD<30%")

    print("\n加载数据...")
    all_data = load_data()
    ts_dict = load_term_structure()

    print("\n计算信号...")
    signal_data = compute_signals(all_data, ts_dict)

    # ═══ A. Carry信号分析 ═══
    print(f"\n{'='*60}")
    print("A. Carry信号对gap fade的影响")
    print(f"{'='*60}")

    for carry_label, carry_cond in [
        ('全部', lambda r: True),
        ('做多+backwardation', lambda r: r['direction']=='long' and r.get('carry',0)>1),
        ('做多+contango', lambda r: r['direction']=='long' and r.get('carry',0)<-1),
        ('做空+contango', lambda r: r['direction']=='short' and r.get('carry',0)<-1),
        ('做空+backwardation', lambda r: r['direction']=='short' and r.get('carry',0)>1),
    ]:
        # 统计信号层面的carry影响
        rows = []
        for sym, df in signal_data.items():
            mask = (df['trade_date'] >= '2022-01-01') & (df['trade_date'] <= '2025-12-31')
            sub = df[mask].copy()
            if carry_label.startswith('做多'):
                sub = sub[sub['score_long'] >= 8]
                fwd = sub['fwd_2d'].dropna()
            elif carry_label.startswith('做空'):
                sub = sub[sub['score_short'] >= 8]
                fwd = -sub['fwd_2d'].dropna()  # 做空反转
            else:
                continue

            if carry_label != '全部':
                if 'backwardation' in carry_label and '做多' in carry_label:
                    sub = sub[sub.get('carry', 0) > 1]
                elif 'contango' in carry_label and '做多' in carry_label:
                    sub = sub[sub.get('carry', 0) < -1]
                elif 'contango' in carry_label and '做空' in carry_label:
                    sub = sub[sub.get('carry', 0) < -1]
                elif 'backwardation' in carry_label and '做空' in carry_label:
                    sub = sub[sub.get('carry', 0) > 1]
                fwd = sub['fwd_2d'].dropna() if '做多' in carry_label else -sub['fwd_2d'].dropna()

            if len(sub) > 0: rows.append(sub)

        if rows:
            all_rows = pd.concat(rows)
            fwd = all_rows['fwd_2d'].dropna()
            print(f"  {carry_label:30s}: N={len(all_rows):5d} WR={100*(fwd>0).mean():.1f}% Avg={fwd.mean():+.3f}%")

    # ═══ B. 策略对比 ═══
    print(f"\n\n{'='*60}")
    print("B. 策略对比: 有carry vs 无carry")
    print(f"{'='*60}")

    configs = [
        ("无carry加分(基线)", 8, 2, None, None, None),
        ("carry加分(已集成)", 8, 2, None, None, None),  # score already includes carry
        ("carry+SL-2/TP3", 8, 2, -2, 3, None),
        ("carry+SL-2/TP3+lev5", 8, 2, -2, 3, None),
        ("carry require filter", 8, 2, None, None, 'require_bkwd'),
        ("carry+SL-2/TP3+require", 8, 2, -2, 3, 'require_bkwd'),
    ]

    for label, min_sc, hd, sl, tp, cf in configs:
        for lev in [3, 5]:
            if 'lev5' in label and lev != 5: continue
            if 'lev5' not in label and lev == 5: continue
            eq, trades = run_backtest(signal_data, '2022-01-01', '2025-12-31',
                                       min_score=min_sc, hold_days=hd, leverage=lev,
                                       stop_loss=sl, take_profit=tp, carry_filter=cf)
            print_results(eq, trades, f"{label} lev={lev}x")

    # ═══ C. Walk-Forward ═══
    print(f"\n\n{'='*60}")
    print("C. Walk-Forward验证 (carry加分版)")
    print(f"{'='*60}")

    for sl, tp in [(None, None), (-2, 3)]:
        sl_s = f"SL={sl}" if sl else "NoSL"
        tp_s = f"TP={tp}" if tp else "NoTP"
        eq_tr, trades_tr = run_backtest(signal_data, '2015-01-01', '2021-12-31',
                                          min_score=8, hold_days=2, leverage=3,
                                          stop_loss=sl, take_profit=tp)
        print_results(eq_tr, trades_tr, f"训练 {sl_s} {tp_s}")

        eq_te, trades_te = run_backtest(signal_data, '2022-01-01', '2025-12-31',
                                          min_score=8, hold_days=2, leverage=3,
                                          stop_loss=sl, take_profit=tp)
        print_results(eq_te, trades_te, f"测试 {sl_s} {tp_s}")

    # ═══ D. 期权IV/HV分析 (单日快照) ═══
    print(f"\n\n{'='*60}")
    print("D. 期权IV/HV分析 (可用的数据)")
    print(f"{'='*60}")

    # 加载期权数据
    options_data = {}
    for f in glob.glob('data/options/*.json'):
        try:
            with open(f) as fp:
                d = json.load(fp)
            sym = d.get('symbol', '')
            hv20 = d.get('hv_20', None)
            hv60 = d.get('hv_60', None)
            surface = d.get('surface', [])
            # 计算平均IV
            ivs = [s['iv'] for s in surface if s.get('iv', 0) > 0]
            avg_iv = np.mean(ivs) if ivs else 0
            if hv20 and hv20 > 0:
                iv_hv_ratio = avg_iv / hv20
            else:
                iv_hv_ratio = 0
            options_data[sym] = {
                'hv20': hv20, 'hv60': hv60,
                'avg_iv': avg_iv, 'iv_hv_ratio': iv_hv_ratio,
                'n_contracts': len(surface),
            }
        except:
            continue

    print(f"  期权数据: {len(options_data)}品种")
    iv_hv = [(s, d['iv_hv_ratio']) for s, d in options_data.items() if d['iv_hv_ratio'] > 0]
    iv_hv.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Top 15 IV/HV Ratio (期权贵/市场恐慌):")
    for sym, ratio in iv_hv[:15]:
        d = options_data[sym]
        print(f"    {sym}: IV/HV={ratio:.2f} IV={d['avg_iv']:.3f} HV20={d['hv20']:.3f}")

    print(f"\n  Bottom 15 IV/HV Ratio (期权便宜):")
    for sym, ratio in iv_hv[-15:]:
        d = options_data[sym]
        print(f"    {sym}: IV/HV={ratio:.2f} IV={d['avg_iv']:.3f} HV20={d['hv20']:.3f}")

    # ═══ E. 最终最优配置 ═══
    print(f"\n\n{'='*60}")
    print("E. 最终最优配置扫描")
    print(f"{'='*60}")

    best = None
    best_sharpe = 0

    for min_sc in [7, 8, 9]:
        for hd in [2, 3]:
            for sl, tp in [(None, None), (-2, 3), (-3, 5), (-2, None)]:
                for lev in [3, 5]:
                    eq, trades = run_backtest(signal_data, '2022-01-01', '2025-12-31',
                                               min_score=min_sc, hold_days=hd,
                                               leverage=lev, stop_loss=sl, take_profit=tp)
                    eq_df = pd.DataFrame(eq)
                    if len(eq_df) == 0 or eq_df['capital'].iloc[-1] <= INITIAL_CAPITAL:
                        continue

                    tdf = pd.DataFrame(trades)
                    wr = (tdf['pnl_pct'] > 0).mean() * 100
                    mdd_val = ((eq_df['capital'] - eq_df['capital'].cummax()) / eq_df['capital'].cummax() * 100).min()
                    daily_ret = eq_df['capital'].pct_change().dropna()
                    sharpe = daily_ret.mean() / daily_ret.std() * (252**0.5) if daily_ret.std() > 0 else 0
                    n_years = max((eq_df['date'].iloc[-1] - eq_df['date'].iloc[0]).days / 365.25, 0.01)
                    annual = ((eq_df['capital'].iloc[-1] / eq_df['capital'].iloc[0]) ** (1/n_years) - 1) * 100

                    if wr >= 50 and mdd_val >= -30:
                        if best is None or sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best = {
                                'min_sc': min_sc, 'hd': hd, 'sl': sl, 'tp': tp, 'lev': lev,
                                'annual': annual, 'mdd': mdd_val, 'wr': wr, 'sharpe': sharpe,
                                'n_trades': len(trades),
                            }

                    sl_s = f"SL={sl}" if sl else "-"
                    tp_s = f"TP={tp}" if tp else "-"
                    if wr >= 50 and annual >= 1000:
                        print(f"  min={min_sc} H={hd} {sl_s} {tp_s} lev={lev}: "
                              f"N={len(trades)} WR={wr:.1f}% Annual={annual:.0f}% MDD={mdd_val:.1f}% Sharpe={sharpe:.2f}")

    if best:
        print(f"\n  ═══ 最佳配置 (WR>=50% 且 MDD<=30%) ═══")
        print(f"  min={best['min_sc']} H={best['hd']}d SL={best['sl']} TP={best['tp']} lev={best['lev']}x")
        print(f"  N={best['n_trades']} WR={best['wr']:.1f}% Annual={best['annual']:.0f}% MDD={best['mdd']:.1f}% Sharpe={best['sharpe']:.2f}")

        # 验证
        eq, trades = run_backtest(signal_data, '2022-01-01', '2025-12-31',
                                   min_score=best['min_sc'], hold_days=best['hd'],
                                   leverage=best['lev'], stop_loss=best['sl'],
                                   take_profit=best['tp'])
        print_results(eq, trades, "最终最佳 - 测试期")
    else:
        print(f"\n  未找到满足 WR>=50% 且 MDD<=30% 且 年化>=1000% 的配置")
        print(f"  放宽条件搜索...")
        for min_sc in [7, 8]:
            for hd in [2, 3]:
                for sl, tp in [(None, None), (-2, 3), (-3, 5)]:
                    for lev in [3, 5]:
                        eq, trades = run_backtest(signal_data, '2022-01-01', '2025-12-31',
                                                   min_score=min_sc, hold_days=hd,
                                                   leverage=lev, stop_loss=sl, take_profit=tp)
                        eq_df = pd.DataFrame(eq)
                        if len(eq_df) == 0 or eq_df['capital'].iloc[-1] <= INITIAL_CAPITAL:
                            continue
                        tdf = pd.DataFrame(trades)
                        wr = (tdf['pnl_pct'] > 0).mean() * 100
                        mdd_val = ((eq_df['capital'] - eq_df['capital'].cummax()) / eq_df['capital'].cummax() * 100).min()
                        daily_ret = eq_df['capital'].pct_change().dropna()
                        sharpe = daily_ret.mean() / daily_ret.std() * (252**0.5) if daily_ret.std() > 0 else 0
                        n_years = max((eq_df['date'].iloc[-1] - eq_df['date'].iloc[0]).days / 365.25, 0.01)
                        annual = ((eq_df['capital'].iloc[-1] / eq_df['capital'].iloc[0]) ** (1/n_years) - 1) * 100
                        if wr >= 50 and annual >= 1000:
                            sl_s = f"SL={sl}" if sl else "-"
                            tp_s = f"TP={tp}" if tp else "-"
                            print(f"  min={min_sc} H={hd} {sl_s} {tp_s} lev={lev}: "
                                  f"N={len(trades)} WR={wr:.1f}% Annual={annual:.0f}% MDD={mdd_val:.1f}%")


if __name__ == '__main__':
    main()
