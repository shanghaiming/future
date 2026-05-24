#!/usr/bin/env python3
"""
V85: 期限结构深度整合 + 期权IV快照分析
期限结构: 82055文件, 124品种, 2021-2026 (每日)
期权: 88文件, 85品种, 6个日期快照

方向:
1. 期限结构历史分析: carry, 结构变化, 价差趋势
2. 期限结构因子: backwardation/contango, carry强度, 近远月价差变化
3. 结合gap fade策略: 期限结构因子能否提升信号?
4. 期权IV分析: IV/HV比率, IV percentile, 波动率结构
5. IV高的品种 gap fade效果是否更好?
"""
import os, glob, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data/futures_weighted'
TS_DIR = 'data/futures_term_structure'
OPT_DIR = 'data/options'
INITIAL_CAPITAL = 500_000
CONTRACT_SPECS = 'scripts/contract_specs.py'
TEST_START = '2022-01-01'
TEST_END = '2025-12-31'


def load_futures_data():
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


def load_term_structure():
    """加载所有期限结构数据"""
    print("  加载期限结构...")
    records = []
    files = sorted(glob.glob(os.path.join(TS_DIR, '*.json')))
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
            records.append({
                'sym': d['symbol'],
                'date': pd.to_datetime(d['date']),
                'structure': d.get('structure', 'unknown'),
                'near_price': d.get('near_price', np.nan),
                'far_price': d.get('far_price', np.nan),
                'spread_pct': d.get('total_spread_pct', 0),
                'n_contracts': len(d.get('curve', [])),
                'near_vol': d['curve'][0]['volume'] if d.get('curve') else 0,
                'far_vol': d['curve'][-1]['volume'] if d.get('curve') else 0,
            })
        except:
            continue

    ts_df = pd.DataFrame(records)
    ts_df['is_backwardation'] = (ts_df['structure'] == 'backwardation').astype(int)
    ts_df['carry'] = -ts_df['spread_pct']  # 正carry = backwardation (近月>远月)
    print(f"    {len(ts_df)} records, {ts_df['sym'].nunique()} symbols, "
          f"{ts_df['date'].min().strftime('%Y-%m-%d')} ~ {ts_df['date'].max().strftime('%Y-%m-%d')}")
    return ts_df


def load_options_snapshot():
    """加载期权快照数据"""
    print("  加载期权数据...")
    opt_data = []
    for f in sorted(glob.glob(os.path.join(OPT_DIR, '*.json'))):
        bn = os.path.basename(f).replace('.json', '')
        parts = bn.rsplit('_', 1)
        if len(parts) != 2: continue
        sym, date_str = parts
        try:
            with open(f) as fh:
                d = json.load(fh)
        except:
            continue

        if isinstance(d, list):
            # 旧格式 (510050等)
            continue
        elif isinstance(d, dict) and 'surface' in d:
            # 新格式 (期货期权)
            hv20 = d.get('hv_20', np.nan)
            hv60 = d.get('hv_60', np.nan)
            underlying = d.get('underlying_price', np.nan)
            surface = d.get('surface', [])

            # 提取ATM IV (moneyness ≈ 1.0)
            atm_ivs = [s['iv'] for s in surface
                       if abs(s.get('moneyness', 0) - 1.0) < 0.05 and s.get('expiry_days', 0) <= 60]
            atm_iv = np.mean(atm_ivs) if atm_ivs else np.nan

            # 提取不同expiry的ATM IV
            iv_by_expiry = {}
            for s in surface:
                if abs(s.get('moneyness', 0) - 1.0) < 0.1:
                    exp = s.get('expiry_days', 0)
                    if exp not in iv_by_expiry:
                        iv_by_expiry[exp] = []
                    iv_by_expiry[exp].append(s['iv'])

            # IV term structure slope
            expiries = sorted(iv_by_expiry.keys())
            if len(expiries) >= 2:
                iv_short = np.mean(iv_by_expiry[expiries[0]])
                iv_long = np.mean(iv_by_expiry[expiries[-1]])
                iv_slope = (iv_long - iv_short) / iv_short * 100 if iv_short > 0 else 0
            else:
                iv_slope = 0

            # Skew (25-delta put vs call)
            put_25 = [s['iv'] for s in surface
                      if s.get('flag') == 'put' and abs(s.get('delta', 0) - (-0.25)) < 0.1]
            call_25 = [s['iv'] for s in surface
                       if s.get('flag') == 'call' and abs(s.get('delta', 0) - 0.25) < 0.1]
            skew = (np.mean(put_25) - np.mean(call_25)) / np.mean(call_25) * 100 if put_25 and call_25 else 0

            opt_data.append({
                'sym': sym,
                'date': date_str,
                'hv20': hv20,
                'hv60': hv60,
                'atm_iv': atm_iv,
                'iv_hv_ratio': atm_iv / hv20 if hv20 > 0 and not np.isnan(hv20) and atm_iv > 0 else np.nan,
                'iv_slope': iv_slope,
                'skew': skew,
            })

    opt_df = pd.DataFrame(opt_data)
    if len(opt_df) > 0:
        print(f"    {len(opt_df)} records, {opt_df['sym'].nunique()} symbols, dates: {sorted(opt_df['date'].unique())}")
    return opt_df


def compute_signals_with_ts(all_data, ts_df):
    """计算信号 + 期限结构因子"""
    signal_data = {}
    # 预处理期限结构
    ts_df['date'] = pd.to_datetime(ts_df['date'])

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

        # 合并期限结构
        sym_ts = ts_df[ts_df['sym'] == sym][['date', 'structure', 'carry', 'spread_pct',
                                               'is_backwardation', 'near_vol']].copy()
        sym_ts.columns = ['trade_date', 'ts_structure', 'ts_carry', 'ts_spread',
                           'ts_backwardation', 'ts_near_vol']
        df = df.merge(sym_ts, on='trade_date', how='left')
        df['ts_carry'] = df['ts_carry'].fillna(0)
        df['ts_spread'] = df['ts_spread'].fillna(0)
        df['ts_backwardation'] = df['ts_backwardation'].fillna(0)

        # 期限结构因子
        # carry > 0 意味着 backwardation (近月升水)
        df['ts_carry_ma5'] = df['ts_carry'].rolling(5, min_periods=1).mean()
        df['ts_carry_ma20'] = df['ts_carry'].rolling(20, min_periods=1).mean()
        df['ts_carry_chg'] = df['ts_carry'] - df['ts_carry'].shift(5)

        signal_data[sym] = df
    return signal_data


def analyze_ts_factors(signal_data):
    """分析期限结构因子对gap fade的影响"""
    print(f"\n{'='*60}")
    print("A. 期限结构因子分析")
    print(f"{'='*60}")

    # 收集所有得分>=7的信号
    rows = []
    for sym, df in signal_data.items():
        mask = (df['trade_date'] >= TEST_START) & (df['trade_date'] <= TEST_END)
        sub = df[mask & ((df['score_long'] >= 7) | (df['score_short'] >= 7))]
        if len(sub) > 0:
            rows.append(sub)
    if not rows: return
    all_sig = pd.concat(rows)

    # 前向收益
    all_sig['fwd_1d'] = np.nan
    for sym, df in signal_data.items():
        sym_mask = (df['trade_date'] >= TEST_START) & (df['trade_date'] <= TEST_END)
        sym_sub = df[sym_mask & ((df['score_long'] >= 7) | (df['score_short'] >= 7))]
        for idx in sym_sub.index:
            dt = df.loc[idx, 'trade_date']
            next_rows = df[df['trade_date'] > dt].head(1)
            if len(next_rows) > 0:
                o_price = df.loc[idx, 'open'] if not np.isnan(df.loc[idx, 'open']) else df.loc[idx, 'close']
                c_next = next_rows['close'].values[0]
                if o_price > 0:
                    all_sig.loc[idx, 'fwd_1d'] = (c_next - o_price) / o_price * 100

    # 按carry分组
    print(f"\n  按期限结构carry分组:")
    print(f"  {'carry区间':>16} {'N':>6} {'WR':>6} {'Avg':>8}")
    print("  " + "-"*40)
    for lo, hi in [(-10, -2), (-2, -0.5), (-0.5, 0.5), (0.5, 2), (2, 10)]:
        sub = all_sig[(all_sig['ts_carry'] >= lo) & (all_sig['ts_carry'] < hi)]
        if len(sub) < 50: continue
        fwd = sub['fwd_1d'].dropna()
        if len(fwd) == 0: continue
        wr = (fwd > 0).mean() * 100
        avg = fwd.mean()
        struct = "backwardation" if hi > 0 else "contango"
        print(f"  [{lo:+5.1f}, {hi:+5.1f}) {struct:>14s} {len(sub):>6} {wr:>5.1f}% {avg:>+7.3f}%")

    # 按carry变化分组
    print(f"\n  按carry变化(5日)分组:")
    print(f"  {'carry_chg':>16} {'N':>6} {'WR':>6} {'Avg':>8}")
    print("  " + "-"*40)
    for lo, hi in [(-5, -1), (-1, -0.2), (-0.2, 0.2), (0.2, 1), (1, 5)]:
        sub = all_sig[(all_sig['ts_carry_chg'] >= lo) & (all_sig['ts_carry_chg'] < hi)]
        if len(sub) < 50: continue
        fwd = sub['fwd_1d'].dropna()
        if len(fwd) == 0: continue
        wr = (fwd > 0).mean() * 100
        avg = fwd.mean()
        print(f"  [{lo:+5.1f}, {hi:+5.1f}) {len(sub):>6} {wr:>5.1f}% {avg:>+7.3f}%")


def analyze_options(opt_df):
    """分析期权快照"""
    print(f"\n\n{'='*60}")
    print("B. 期权IV快照分析")
    print(f"{'='*60}")

    if len(opt_df) == 0:
        print("  无期权数据")
        return

    print(f"\n  品种IV/HV统计 ({opt_df['date'].nunique()}个日期):")
    print(f"  {'品种':>6} {'HV20':>6} {'ATM_IV':>7} {'IV/HV':>6} {'Skew':>6} {'IV斜率':>6}")
    print("  " + "-"*40)

    for _, row in opt_df.sort_values('iv_hv_ratio', ascending=False).head(20).iterrows():
        print(f"  {row['sym']:>6} {row['hv20']:>6.2f} {row['atm_iv']:>7.3f} "
              f"{row['iv_hv_ratio']:>6.2f} {row['skew']:>+5.1f}% {row['iv_slope']:>+5.1f}%")

    # IV/HV分布
    valid = opt_df['iv_hv_ratio'].dropna()
    if len(valid) > 0:
        print(f"\n  IV/HV ratio统计:")
        print(f"    均值={valid.mean():.2f} 中位数={valid.median():.2f}")
        print(f"    IV>HV (波动率溢价): {(valid > 1).sum()}/{len(valid)} = {(valid > 1).mean()*100:.0f}%")
        print(f"    IV<HV (波动率折价): {(valid < 1).sum()}/{len(valid)} = {(valid < 1).mean()*100:.0f}%")

    # 按IV/HV分组看gap fade效果
    print(f"\n  IV/HV与信号关系 (快照日期):")
    # 加载期货数据获取gap信息
    for _, row in opt_df.iterrows():
        sym = row['sym']
        # 对应期货数据
        csv_file = os.path.join(DATA_DIR, f'{sym}.csv')
        if not os.path.exists(csv_file): continue
        df = pd.read_csv(csv_file)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        df = df.sort_values('trade_date').reset_index(drop=True)
        # 找最近的日期
        opt_date = pd.to_datetime(row['date'])
        idx = df.index[df['trade_date'] <= opt_date]
        if len(idx) < 5: continue
        last = df.loc[idx[-1]]
        prev = df.loc[idx[-2]] if len(idx) >= 2 else last
        gap = (last['open'] - prev['close']) / prev['close'] * 100
        # 这个暂时没法做回测, 只是展示


def run_bt_ts(signal_data, start, end, max_pos=7, lev=5, min_sc=7, hold=1,
              sl_pct=-1.5, tp_pct=4.0, ts_filter=None):
    """
    带期限结构过滤的回测
    ts_filter: 'backwardation', 'contango', 'carry_up', 'carry_down', None
    """
    dates = pd.date_range(start=start, end=end, freq='B')
    cap = INITIAL_CAPITAL
    eq, trades, pos = [], [], []

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
                trades.append({
                    'sym': p['sym'], 'dir': p['dir'], 'ed': p['ed'], 'xd': dt,
                    'ep': p['ep'], 'xp': cur_c, 'r': actual_ret,
                    'pnl': p['not'] * actual_ret / 100, 'sc': p['sc'],
                    'hold': d, 'reason': reason,
                })
        pos = keep
        cap += pnl
        if cap <= 0: break
        n_open = max_pos - len(pos)
        if n_open <= 0:
            eq.append({'date': dt, 'capital': cap}); continue

        cands = []
        for sym, df in signal_data.items():
            if any(p['sym'] == sym for p in pos): continue
            idx = df.index[df['trade_date'] == dt]
            if len(idx) == 0: continue
            row = df.loc[idx[0]]

            # 期限结构过滤
            if ts_filter == 'backwardation' and row.get('ts_backwardation', 0) != 1:
                continue
            elif ts_filter == 'contango' and row.get('ts_backwardation', 0) == 1:
                continue

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
            if n_long >= 4 and c_['dir'] == 'long': continue
            if n_short >= 4 and c_['dir'] == 'short': continue
            notional = cap * lev / max_pos
            pos.append({'sym': c_['sym'], 'dir': c_['dir'], 'ed': dt,
                        'ep': c_['ep'], 'not': notional, 'sc': c_['sc']})
            if c_['dir'] == 'long': n_long += 1
            else: n_short += 1
            n_open -= 1
        eq.append({'date': dt, 'capital': cap})
    return eq, trades


def pr(eq, trades, label):
    eq_df = pd.DataFrame(eq)
    if len(eq_df) == 0 or eq_df['capital'].iloc[-1] <= 0:
        print(f"  {label}: 爆仓"); return
    eq_df['peak'] = eq_df['capital'].cummax()
    eq_df['dd'] = (eq_df['capital'] - eq_df['peak']) / eq_df['peak'] * 100
    mdd = eq_df['dd'].min()
    dr = eq_df['capital'].pct_change().dropna()
    sh = dr.mean() / dr.std() * (252**0.5) if dr.std() > 0 else 0
    tdf = pd.DataFrame(trades)
    wr = (tdf['r'] > 0).mean() * 100 if len(tdf) > 0 else 0
    avg = tdf['r'].mean() if len(tdf) > 0 else 0
    print(f"  {label}")
    print(f"    N={len(trades)} WR={wr:.1f}% Sharpe={sh:.2f} Avg={avg:+.3f}% MDD={mdd:.1f}%")


def main():
    print("V85: 期限结构深度整合 + 期权IV分析")
    print("="*60)

    print("\n加载数据...")
    all_data = load_futures_data()
    ts_df = load_term_structure()
    opt_df = load_options_snapshot()

    print("\n计算信号(含期限结构)...")
    sd = compute_signals_with_ts(all_data, ts_df)

    # ═══ 期限结构因子分析 ═══
    analyze_ts_factors(sd)

    # ═══ 期权分析 ═══
    analyze_options(opt_df)

    # ═══ 带期限结构过滤的回测 ═══
    print(f"\n\n{'='*60}")
    print("C. 期限结构过滤回测")
    print(f"{'='*60}")

    pr(*run_bt_ts(sd, TEST_START, TEST_END), "无过滤 (基准)")
    pr(*run_bt_ts(sd, TEST_START, TEST_END, ts_filter='backwardation'), "只做backwardation品种")
    pr(*run_bt_ts(sd, TEST_START, TEST_END, ts_filter='contango'), "只做contango品种")

    # ═══ 期限结构加权得分 ═══
    print(f"\n\n{'='*60}")
    print("D. 期限结构加权得分")
    print(f"{'='*60}")

    # 给backwardation品种做多加分, contango品种做空加分
    for sym, df in sd.items():
        # 如果backwardation + 做多: +2分
        # 如果contango + 做空: +2分
        df['score_long_ts'] = df['score_long'] + df['ts_backwardation'] * 2
        df['score_short_ts'] = df['score_short'] + (1 - df['ts_backwardation']) * 2

    # 用TS加权得分回测
    # 修改min_sc为9因为加了2分
    dates = pd.date_range(start=TEST_START, end=TEST_END, freq='B')
    cap = INITIAL_CAPITAL
    eq, trades, pos = [], [], []
    for dt in dates:
        pnl = 0; keep = []
        for p in pos:
            df = sd.get(p['sym'])
            if df is None: keep.append(p); continue
            idx = df.index[df['trade_date'] == dt]
            if len(idx) == 0: keep.append(p); continue
            row = df.loc[idx[0]]
            cur_h, cur_l, cur_c = row['high'], row['low'], row['close']
            if np.isnan(cur_c): keep.append(p); continue
            d = (dt - p['ed']).days; sp = 0.001
            triggered = False; actual_ret = None; reason = None
            if p['dir'] == 'long':
                if -1.5:
                    stop = p['ep'] * (1 + (-1.5) / 100)
                    if cur_l <= stop:
                        actual_ret = (stop*(1-sp) - p['ep'])/p['ep']*100; reason='SL'; triggered=True
                if not triggered:
                    tp_p = p['ep'] * (1 + 4.0 / 100)
                    if cur_h >= tp_p:
                        actual_ret = (tp_p*(1-sp) - p['ep'])/p['ep']*100; reason='TP'; triggered=True
                if not triggered: actual_ret = (cur_c - p['ep'])/p['ep']*100
            else:
                if -1.5:
                    stop = p['ep'] * (1 - (-1.5) / 100)
                    if cur_h >= stop:
                        actual_ret = (p['ep'] - stop*(1+sp))/p['ep']*100; reason='SL'; triggered=True
                if not triggered:
                    tp_p = p['ep'] * (1 - 4.0 / 100)
                    if cur_l <= tp_p:
                        actual_ret = (p['ep'] - tp_p*(1+sp))/p['ep']*100; reason='TP'; triggered=True
                if not triggered: actual_ret = (p['ep'] - cur_c)/p['ep']*100
            if d >= 1:
                if not triggered: reason = 'exp'
            else:
                if not triggered: keep.append(p); continue
            if reason:
                pnl += p['not'] * actual_ret / 100
                trades.append({'sym':p['sym'],'dir':p['dir'],'ed':p['ed'],'xd':dt,'ep':p['ep'],'xp':cur_c,'r':actual_ret,'pnl':p['not']*actual_ret/100,'sc':p['sc'],'hold':d,'reason':reason})
        pos = keep; cap += pnl
        if cap <= 0: break
        n_open = 7 - len(pos)
        if n_open <= 0: eq.append({'date':dt,'capital':cap}); continue
        cands = []
        for sym, df in sd.items():
            if any(p['sym']==sym for p in pos): continue
            idx = df.index[df['trade_date']==dt]
            if len(idx)==0: continue
            row = df.loc[idx[0]]
            if row['score_long_ts'] >= 9:  # 7+2=9
                cands.append({'sym':sym,'dir':'long','sc':row['score_long_ts'],'ep':row['open']})
            if row['score_short_ts'] >= 9:
                cands.append({'sym':sym,'dir':'short','sc':row['score_short_ts'],'ep':row['open']})
        best = {}
        for c_ in cands:
            if c_['sym'] not in best or c_['sc'] > best[c_['sym']]['sc']:
                best[c_['sym']] = c_
        ranked = sorted(best.values(), key=lambda x: -x['sc'])
        n_long = sum(1 for p in pos if p['dir']=='long')
        n_short = sum(1 for p in pos if p['dir']=='short')
        for c_ in ranked:
            if n_open <= 0: break
            if n_long >= 4 and c_['dir']=='long': continue
            if n_short >= 4 and c_['dir']=='short': continue
            notional = cap * 5 / 7
            pos.append({'sym':c_['sym'],'dir':c_['dir'],'ed':dt,'ep':c_['ep'],'not':notional,'sc':c_['sc']})
            if c_['dir']=='long': n_long+=1
            else: n_short+=1
            n_open-=1
        eq.append({'date':dt,'capital':cap})

    pr(eq, trades, "TS加权得分 (backwardation+多, contango+空)")


if __name__ == '__main__':
    main()
