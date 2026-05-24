#!/usr/bin/env python3
"""
V81: 策略组合 — 多参数组合分散风险
思路: 同时运行多个不同参数的子策略, 分配资金
1. 激进组 (mp=7, lev=5, SL=-1.5, TP=4) + 保守组 (mp=5, lev=3, SL=-2, TP=5)
2. 不同min_score组合
3. 不同持仓天数组合
4. 等权重 vs 按Sharpe加权
5. 动态资金分配
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
        signal_data[sym] = df
    return signal_data


def run_single(signal_data, start, end, max_pos=7, lev=5, min_sc=7, hold=1,
               sl_pct=-1.5, tp_pct=4.0, max_long=None, max_short=None):
    """单策略回测, 返回日收益序列"""
    dates = pd.date_range(start=start, end=end, freq='B')
    cap = INITIAL_CAPITAL
    pos = []
    daily_pnl = {}

    for dt in dates:
        pnl = 0
        keep = []
        for p in pos:
            df = signal_data.get(p['sym'])
            if df is None: keep.append(p); continue
            idx = df.index[df['trade_date'] == dt]
            if len(idx) == 0: keep.append(p); continue
            row = df.loc[idx[0]]
            cur_h, cur_l, cur_c = row['high'], row['low'], row['close']
            if np.isnan(cur_c): keep.append(p); continue

            d = (dt - p['ed']).days
            sp = 0.001
            triggered = False
            actual_ret = None
            reason = None

            if p['dir'] == 'long':
                if sl_pct:
                    stop = p['ep'] * (1 + sl_pct / 100)
                    if cur_l <= stop:
                        actual_ret = (stop * (1 - sp) - p['ep']) / p['ep'] * 100
                        reason = 'SL'; triggered = True
                if not triggered and tp_pct:
                    tp_p = p['ep'] * (1 + tp_pct / 100)
                    if cur_h >= tp_p:
                        actual_ret = (tp_p * (1 - sp) - p['ep']) / p['ep'] * 100
                        reason = 'TP'; triggered = True
                if not triggered:
                    actual_ret = (cur_c - p['ep']) / p['ep'] * 100
            else:
                if sl_pct:
                    stop = p['ep'] * (1 - sl_pct / 100)
                    if cur_h >= stop:
                        actual_ret = (p['ep'] - stop * (1 + sp)) / p['ep'] * 100
                        reason = 'SL'; triggered = True
                if not triggered and tp_pct:
                    tp_p = p['ep'] * (1 - tp_pct / 100)
                    if cur_l <= tp_p:
                        actual_ret = (p['ep'] - tp_p * (1 + sp)) / p['ep'] * 100
                        reason = 'TP'; triggered = True
                if not triggered:
                    actual_ret = (p['ep'] - cur_c) / p['ep'] * 100

            if d >= hold:
                if not triggered: reason = 'exp'
            else:
                if not triggered: keep.append(p); continue

            if reason:
                pnl += p['not'] * actual_ret / 100

        pos = keep
        cap += pnl
        if cap <= 0: break

        n_open = max_pos - len(pos)
        if n_open <= 0:
            daily_pnl[dt] = 0; continue

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

        n_long = sum(1 for p in pos if p['dir'] == 'long')
        n_short = sum(1 for p in pos if p['dir'] == 'short')

        for c_ in ranked:
            if n_open <= 0: break
            if max_long and c_['dir'] == 'long' and n_long >= max_long: continue
            if max_short and c_['dir'] == 'short' and n_short >= max_short: continue
            notional = cap * lev / max_pos
            pos.append({'sym': c_['sym'], 'dir': c_['dir'], 'ed': dt,
                        'ep': c_['ep'], 'not': notional, 'sc': c_['sc']})
            if c_['dir'] == 'long': n_long += 1
            else: n_short += 1
            n_open -= 1

        daily_pnl[dt] = pnl

    return daily_pnl


def combine_strategies(signal_data, start, end, strategies, weights=None):
    """
    组合多个策略的日收益
    strategies: list of (name, kwargs)
    weights: list of weights (default equal)
    """
    if weights is None:
        weights = [1.0 / len(strategies)] * len(strategies)

    all_pnls = []
    for name, kwargs in strategies:
        pnl = run_single(signal_data, start, end, **kwargs)
        all_pnls.append(pnl)

    # 合并日收益 (加权)
    all_dates = sorted(set().union(*[p.keys() for p in all_pnls]))
    combined = {}
    for dt in all_dates:
        total_pnl = 0
        for i, pnl in enumerate(all_pnls):
            total_pnl += pnl.get(dt, 0) * weights[i]
        combined[dt] = total_pnl

    # 计算equity curve
    cap = INITIAL_CAPITAL
    eq = []
    for dt in all_dates:
        cap += combined[dt]
        if cap <= 0: break
        eq.append({'date': dt, 'capital': cap})

    return eq, combined


def evaluate(eq, label):
    eq_df = pd.DataFrame(eq)
    if len(eq_df) == 0 or eq_df['capital'].iloc[-1] <= 0:
        print(f"  {label}: 爆仓"); return
    eq_df['peak'] = eq_df['capital'].cummax()
    eq_df['dd'] = (eq_df['capital'] - eq_df['peak']) / eq_df['peak'] * 100
    ny = max((eq_df['date'].iloc[-1] - eq_df['date'].iloc[0]).days / 365.25, 0.01)
    ann = ((eq_df['capital'].iloc[-1] / eq_df['capital'].iloc[0]) ** (1/ny) - 1) * 100
    mdd = eq_df['dd'].min()
    dr = eq_df['capital'].pct_change().dropna()
    sh = dr.mean() / dr.std() * (252**0.5) if dr.std() > 0 else 0
    neg = dr[dr < 0]
    sortino = dr.mean() / neg.std() * (252**0.5) if len(neg) > 0 and neg.std() > 0 else 0
    pos_days = (dr > 0).mean() * 100

    print(f"  {label}")
    print(f"    年化:{ann:.0f}% MDD:{mdd:.1f}% Sharpe:{sh:.2f} Sortino:{sortino:.2f} 正收益日:{pos_days:.1f}%")


def main():
    print("V81: 策略组合 — 多参数分散风险")
    print("="*60)

    print("\n加载数据...")
    all_data = load_data()
    print("计算信号...")
    sd = compute_signals(all_data)

    # ═══ 子策略定义 ═══
    strategies = [
        ("激进7仓", {'max_pos': 7, 'lev': 5, 'min_sc': 7, 'hold': 1, 'sl_pct': -1.5, 'tp_pct': 4.0, 'max_long': 4, 'max_short': 4}),
        ("平衡5仓", {'max_pos': 5, 'lev': 5, 'min_sc': 7, 'hold': 1, 'sl_pct': -1.5, 'tp_pct': 4.0}),
        ("保守3仓", {'max_pos': 3, 'lev': 3, 'min_sc': 7, 'hold': 1, 'sl_pct': -2.0, 'tp_pct': 5.0}),
        ("严格9分", {'max_pos': 7, 'lev': 5, 'min_sc': 9, 'hold': 1, 'sl_pct': -1.5, 'tp_pct': 4.0}),
        ("宽松2天", {'max_pos': 5, 'lev': 3, 'min_sc': 7, 'hold': 2, 'sl_pct': -2.0, 'tp_pct': 3.0}),
        ("超严格11分", {'max_pos': 5, 'lev': 5, 'min_sc': 11, 'hold': 1, 'sl_pct': -1.5, 'tp_pct': 4.0}),
    ]

    # ═══ A. 各子策略单独表现 ═══
    print(f"\n{'='*60}")
    print("A. 子策略单独表现")
    print(f"{'='*60}")

    for name, kwargs in strategies:
        eq, _ = combine_strategies(sd, TEST_START, TEST_END, [(name, kwargs)], [1.0])
        evaluate(eq, name)

    # ═══ B. 等权组合 ═══
    print(f"\n\n{'='*60}")
    print("B. 等权组合")
    print(f"{'='*60}")

    # 2组合
    for combo in [
        ("激进+保守", [0, 2]),
        ("激进+严格", [0, 3]),
        ("平衡+保守", [1, 2]),
        ("激进+宽松2天", [0, 4]),
        ("保守+严格", [2, 3]),
    ]:
        name, indices = combo
        sub = [strategies[i] for i in indices]
        eq, _ = combine_strategies(sd, TEST_START, TEST_END, sub)
        evaluate(eq, name)

    # 3组合
    for combo in [
        ("激进+保守+严格", [0, 2, 3]),
        ("激进+平衡+保守", [0, 1, 2]),
        ("平衡+保守+严格", [1, 2, 3]),
    ]:
        name, indices = combo
        sub = [strategies[i] for i in indices]
        eq, _ = combine_strategies(sd, TEST_START, TEST_END, sub)
        evaluate(eq, name)

    # 全部组合
    eq_all, _ = combine_strategies(sd, TEST_START, TEST_END, strategies)
    evaluate(eq_all, "全部6策略等权")

    # ═══ C. 最优2组合详细 ═══
    print(f"\n\n{'='*60}")
    print("C. 最优组合详细分析")
    print(f"{'='*60}")

    # 基于上面的结果, 选最佳2组合
    best_combos = [
        ("激进+保守 (50/50)", [strategies[0], strategies[2]], [0.5, 0.5]),
        ("激进+保守 (60/40)", [strategies[0], strategies[2]], [0.6, 0.4]),
        ("激进+保守 (70/30)", [strategies[0], strategies[2]], [0.7, 0.3]),
        ("激进+严格 (50/50)", [strategies[0], strategies[3]], [0.5, 0.5]),
        ("激进+严格 (70/30)", [strategies[0], strategies[3]], [0.7, 0.3]),
    ]

    for name, sub, w in best_combos:
        eq, _ = combine_strategies(sd, TEST_START, TEST_END, sub, w)
        evaluate(eq, name)

    # ═══ D. Walk-forward ═══
    print(f"\n\n{'='*60}")
    print("D. Walk-Forward: 最优组合")
    print(f"{'='*60}")

    best_sub = [strategies[0], strategies[2]]  # 激进+保守
    best_w = [0.5, 0.5]

    eq_test, _ = combine_strategies(sd, TEST_START, TEST_END, best_sub, best_w)
    evaluate(eq_test, "测试期 激进+保守 50/50")

    eq_train, _ = combine_strategies(sd, '2016-01-01', '2021-12-31', best_sub, best_w)
    evaluate(eq_train, "训练期 激进+保守 50/50")

    # 单独激进 Walk-forward
    eq_aggr, _ = combine_strategies(sd, '2016-01-01', '2021-12-31',
                                     [strategies[0]], [1.0])
    evaluate(eq_aggr, "训练期 纯激进")

    eq_cons, _ = combine_strategies(sd, '2016-01-01', '2021-12-31',
                                     [strategies[2]], [1.0])
    evaluate(eq_cons, "训练期 纯保守")

    # ═══ E. 回撤对比 ═══
    print(f"\n\n{'='*60}")
    print("E. 回撤深度对比")
    print(f"{'='*60}")

    for name, sub, w in [
        ("纯激进", [strategies[0]], [1.0]),
        ("纯保守", [strategies[2]], [1.0]),
        ("50/50组合", [strategies[0], strategies[2]], [0.5, 0.5]),
        ("60/40组合", [strategies[0], strategies[2]], [0.6, 0.4]),
        ("70/30组合", [strategies[0], strategies[2]], [0.7, 0.3]),
    ]:
        eq, _ = combine_strategies(sd, TEST_START, TEST_END, sub, w)
        eq_df = pd.DataFrame(eq)
        if len(eq_df) == 0: continue
        eq_df['peak'] = eq_df['capital'].cummax()
        eq_df['dd'] = (eq_df['capital'] - eq_df['peak']) / eq_df['peak'] * 100
        dr = eq_df['capital'].pct_change().dropna()
        sh = dr.mean() / dr.std() * (252**0.5) if dr.std() > 0 else 0

        # 逐年MDD
        eq_df['year'] = eq_df['date'].dt.year
        print(f"\n  {name}: 总MDD={eq_df['dd'].min():.1f}% Sharpe={sh:.2f}")
        for yr in sorted(eq_df['year'].unique()):
            sub_df = eq_df[eq_df['year'] == yr]
            print(f"    {yr}: MDD={sub_df['dd'].min():.1f}%")


if __name__ == '__main__':
    main()
