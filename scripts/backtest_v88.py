#!/usr/bin/env python3
"""
V88: 独立多策略并行组合
不是给gap fade加特征, 而是构建完全独立的策略并行运行:
1. Gap Fade (隔夜跳空反转) — 已验证核心
2. Momentum Breakout (动量突破) — 独立信号
3. Mean Reversion Extreme (极端均值回归) — 独立信号
4. Range Breakout (区间突破) — 独立信号
5. OI-Price Divergence (量价背离) — 独立信号

每个策略独立生成信号, 独立管理仓位, 独立平仓
组合层面: 总仓位限制 + 资金分配
"""
import os, glob, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data/futures_weighted'
CONTRACT_SPECS = 'scripts/contract_specs.py'
INITIAL_CAPITAL = 500_000


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
        if len(df) < 200: continue
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        df = df.sort_values('trade_date').reset_index(drop=True)
        if df['close'].isna().all() or (df['close'] == 0).any(): continue
        all_data[sym] = df
    print(f"  {len(all_data)}品种")
    return all_data


def precompute_indicators(all_data):
    """预计算所有技术指标"""
    for sym, df in all_data.items():
        c = df['close'].values.astype(float)
        o = df['open'].values.astype(float)
        h = df['high'].values.astype(float)
        l = df['low'].values.astype(float)
        v = df['vol'].values.astype(float)
        oi = df['oi'].values.astype(float)
        n = len(df)

        prev_c = np.full(n, np.nan); prev_c[1:] = c[:-1]
        df['prev_close'] = prev_c
        gap_arr = np.full(n, np.nan); gap_arr[1:] = (o[1:] - prev_c[1:]) / prev_c[1:] * 100
        df['gap_pct'] = gap_arr

        # ATR
        tr = np.full(n, np.nan)
        tr[1:] = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
        df['atr'] = pd.Series(tr).rolling(20).mean().values
        df['atr_pct'] = df['atr'].values / np.where(c > 0, c, np.nan) * 100

        # MA
        df['ma5'] = pd.Series(c).rolling(5).mean().values
        df['ma10'] = pd.Series(c).rolling(10).mean().values
        df['ma20'] = pd.Series(c).rolling(20).mean().values
        df['ma60'] = pd.Series(c).rolling(60).mean().values

        # 动量
        mom1 = np.full(n, np.nan); mom1[1:] = (c[1:] - c[:-1]) / c[:-1] * 100
        mom5 = np.full(n, np.nan); mom5[5:] = (c[5:] - c[:-5]) / c[:-5] * 100
        mom10 = np.full(n, np.nan); mom10[10:] = (c[10:] - c[:-10]) / c[:-10] * 100
        mom20 = np.full(n, np.nan); mom20[20:] = (c[20:] - c[:-20]) / c[:-20] * 100
        df['mom1'] = mom1; df['mom5'] = mom5; df['mom10'] = mom10; df['mom20'] = mom20

        # RSI
        delta = np.full(n, np.nan); delta[1:] = c[1:] - c[:-1]
        gain = np.where(delta > 0, delta, 0)
        loss_arr = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean().values
        avg_loss = pd.Series(loss_arr).rolling(14).mean().values
        df['rsi'] = np.where(avg_loss > 0, 100 - 100 / (1 + avg_gain / avg_loss), 50)

        # 布林带
        bb_std = pd.Series(c).rolling(20).std().values
        df['bb_upper'] = df['ma20'].values + 2 * bb_std
        df['bb_lower'] = df['ma20'].values - 2 * bb_std
        df['bb_width'] = np.where(df['ma20'].values > 0,
                                   (df['bb_upper'].values - df['bb_lower'].values) / df['ma20'].values * 100, np.nan)

        # Volume
        df['vol_ma5'] = pd.Series(v).rolling(5).mean().values
        df['vol_ma20'] = pd.Series(v).rolling(20).mean().values

        # OI变化
        oi_ch = np.full(n, np.nan)
        valid = np.abs(oi[:-1]) > 0
        oi_ch_v = np.full(n-1, np.nan)
        oi_ch_v[valid] = (oi[1:][valid] - oi[:-1][valid]) / np.abs(oi[:-1][valid]) * 100
        oi_ch[1:] = oi_ch_v
        df['oi_ch'] = oi_ch

        # CLV
        range_ = h - l
        df['clv'] = np.where(range_ > 0, (2*c - h - l) / range_, 0)

        # ATR channel (Donchian-like)
        df['high_20'] = pd.Series(h).rolling(20).max().values
        df['low_20'] = pd.Series(l).rolling(20).min().values
        df['high_10'] = pd.Series(h).rolling(10).max().values
        df['low_10'] = pd.Series(l).rolling(10).min().values

        # 日内振幅
        df['range_pct'] = np.where(c > 0, (h - l) / c * 100, 0)

        # ===== 信号计算 =====
        gv = df['gap_pct'].fillna(0).values
        atr_pct = df['atr_pct'].fillna(0).values
        ga = np.where(atr_pct != 0, gv / atr_pct, 0)

        # === 信号1: Gap Fade (核心) ===
        s_l_gap = np.zeros(n)
        s_l_gap += (gv < -0.5).astype(int) * 1
        s_l_gap += (gv < -1.0).astype(int) * 2
        s_l_gap += (gv < -1.5).astype(int) * 2
        s_l_gap += (gv < -2.0).astype(int) * 3
        s_l_gap += (ga < -1.0).astype(int) * 2
        s_l_gap += (ga < -1.5).astype(int) * 3
        s_l_gap += ((oi_ch > 0) & (c < prev_c)).astype(int) * 3
        s_l_gap += ((oi_ch < 0) & (c < prev_c)).astype(int) * 2
        mom5v = df['mom5'].fillna(0).values
        s_l_gap += (mom5v < -3).astype(int) * 1
        s_l_gap += (mom5v < -5).astype(int) * 1
        s_l_gap += (c < df['ma5'].values).astype(int) * 1
        s_l_gap += ((v > df['vol_ma5'].values * 1.5) & (c < prev_c)).astype(int) * 1
        s_l_gap += (df['clv'].values > 0.5).astype(int) * 1
        s_l_gap += (df['ma20'].values > df['ma60'].values).astype(int) * 2

        s_s_gap = np.zeros(n)
        s_s_gap += (gv > 0.5).astype(int) * 1
        s_s_gap += (gv > 1.0).astype(int) * 2
        s_s_gap += (gv > 1.5).astype(int) * 2
        s_s_gap += (gv > 2.0).astype(int) * 3
        s_s_gap += (ga > 1.0).astype(int) * 2
        s_s_gap += (ga > 1.5).astype(int) * 3
        s_s_gap += ((oi_ch > 0) & (c > prev_c)).astype(int) * 3
        s_s_gap += ((oi_ch < 0) & (c > prev_c)).astype(int) * 2
        s_s_gap += (mom5v > 3).astype(int) * 1
        s_s_gap += (mom5v > 5).astype(int) * 1
        s_s_gap += (c > df['ma5'].values).astype(int) * 1
        s_s_gap += ((v > df['vol_ma5'].values * 1.5) & (c > prev_c)).astype(int) * 1
        s_s_gap += (df['clv'].values < -0.5).astype(int) * 1
        s_s_gap += (df['ma20'].values < df['ma60'].values).astype(int) * 2

        # === 信号2: 动量突破 (新高新低) ===
        s_l_mom = np.zeros(n)
        s_l_mom += (c >= df['high_20'].values).astype(int) * 5  # 突破20日新高
        s_l_mom += (c >= df['high_10'].values).astype(int) * 3  # 突破10日新高
        s_l_mom += (df['ma5'].values > df['ma20'].values).astype(int) * 2
        s_l_mom += (df['ma20'].values > df['ma60'].values).astype(int) * 2
        s_l_mom += (mom5v > 3).astype(int) * 1
        s_l_mom += (v > df['vol_ma20'].values * 1.5).astype(int) * 2  # 放量突破

        s_s_mom = np.zeros(n)
        s_s_mom += (c <= df['low_20'].values).astype(int) * 5
        s_s_mom += (c <= df['low_10'].values).astype(int) * 3
        s_s_mom += (df['ma5'].values < df['ma20'].values).astype(int) * 2
        s_s_mom += (df['ma20'].values < df['ma60'].values).astype(int) * 2
        s_s_mom += (mom5v < -3).astype(int) * 1
        s_s_mom += (v > df['vol_ma20'].values * 1.5).astype(int) * 2

        # === 信号3: 极端均值回归 ===
        rsi_v = df['rsi'].fillna(50).values
        s_l_mr = np.zeros(n)
        s_l_mr += (rsi_v < 20).astype(int) * 5
        s_l_mr += (rsi_v < 25).astype(int) * 3
        s_l_mr += (rsi_v < 30).astype(int) * 2
        s_l_mr += (c < df['bb_lower'].values).astype(int) * 3
        s_l_mr += (mom5v < -8).astype(int) * 3  # 极端超跌
        s_l_mr += (mom5v < -5).astype(int) * 2
        s_l_mr += (c > df['ma60'].values).astype(int) * 1  # 长期趋势向上

        s_s_mr = np.zeros(n)
        s_s_mr += (rsi_v > 80).astype(int) * 5
        s_s_mr += (rsi_v > 75).astype(int) * 3
        s_s_mr += (rsi_v > 70).astype(int) * 2
        s_s_mr += (c > df['bb_upper'].values).astype(int) * 3
        s_s_mr += (mom5v > 8).astype(int) * 3
        s_s_mr += (mom5v > 5).astype(int) * 2
        s_s_mr += (c < df['ma60'].values).astype(int) * 1

        # === 信号4: 区间突破 (布林带收缩后突破) ===
        bw = df['bb_width'].fillna(0).values
        bw_med = np.nanmedian(bw[bw > 0]) if np.any(bw > 0) else 2
        s_l_range = np.zeros(n)
        s_l_range += ((bw < bw_med * 0.5) & (c > df['bb_upper'].values)).astype(int) * 5  # 低波动+上突破
        s_l_range += ((bw < bw_med * 0.7) & (c > df['ma20'].values) & (c > o)).astype(int) * 3
        s_l_range += (v > df['vol_ma20'].values * 2).astype(int) * 2

        s_s_range = np.zeros(n)
        s_s_range += ((bw < bw_med * 0.5) & (c < df['bb_lower'].values)).astype(int) * 5
        s_s_range += ((bw < bw_med * 0.7) & (c < df['ma20'].values) & (c < o)).astype(int) * 3
        s_s_range += (v > df['vol_ma20'].values * 2).astype(int) * 2

        # === 信号5: OI-Price背离 ===
        s_l_oi = np.zeros(n)
        # 价格跌+OI减 = 空头离场 → 做多
        s_l_oi += ((c < prev_c) & (oi_ch < -3)).astype(int) * 3
        s_l_oi += ((c < prev_c) & (oi_ch < -5)).astype(int) * 3
        s_l_oi += ((c < prev_c) & (oi_ch < -8)).astype(int) * 2
        # 加上gap确认
        s_l_oi += (gv < -0.5).astype(int) * 1
        s_l_oi += (mom5v < -2).astype(int) * 1

        s_s_oi = np.zeros(n)
        # 价格涨+OI减 = 多头离场 → 做空
        s_s_oi += ((c > prev_c) & (oi_ch < -3)).astype(int) * 3
        s_s_oi += ((c > prev_c) & (oi_ch < -5)).astype(int) * 3
        s_s_oi += ((c > prev_c) & (oi_ch < -8)).astype(int) * 2
        s_s_oi += (gv > 0.5).astype(int) * 1
        s_s_oi += (mom5v > 2).astype(int) * 1

        # 存储所有信号
        df['sig_gap_long'] = s_l_gap
        df['sig_gap_short'] = s_s_gap
        df['sig_mom_long'] = s_l_mom
        df['sig_mom_short'] = s_s_mom
        df['sig_mr_long'] = s_l_mr
        df['sig_mr_short'] = s_s_mr
        df['sig_range_long'] = s_l_range
        df['sig_range_short'] = s_s_range
        df['sig_oi_long'] = s_l_oi
        df['sig_oi_short'] = s_s_oi

    return all_data


def run_strategy(data, start, end, strategy_name, signal_col, min_score,
                 max_pos, hold, sl_pct, tp_pct, max_dir=None):
    """运行单个策略"""
    dates = pd.date_range(start=start, end=end, freq='B')
    cap = INITIAL_CAPITAL
    eq, trades, pos = [], [], []
    col_l = f'sig_{strategy_name}_long'
    col_s = f'sig_{strategy_name}_short'

    for dt in dates:
        pnl = 0
        keep = []
        for p in pos:
            df = data.get(p['sym'])
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
                    'hold': d, 'reason': reason, 'strategy': strategy_name,
                })
        pos = keep
        cap += pnl
        if cap <= 0: break

        n_open = max_pos - len(pos)
        if n_open <= 0:
            eq.append({'date': dt, 'capital': cap}); continue

        cands = []
        for sym, df in data.items():
            if any(p['sym'] == sym for p in pos): continue
            idx = df.index[df['trade_date'] == dt]
            if len(idx) == 0: continue
            row = df.loc[idx[0]]
            if np.isnan(row.get('close', np.nan)): continue
            sc_l = row.get(col_l, 0)
            sc_s = row.get(col_s, 0)
            if sc_l >= min_score:
                cands.append({'sym': sym, 'dir': 'long', 'sc': sc_l, 'ep': row['open']})
            if sc_s >= min_score:
                cands.append({'sym': sym, 'dir': 'short', 'sc': sc_s, 'ep': row['open']})

        best = {}
        for c_ in cands:
            if c_['sym'] not in best or c_['sc'] > best[c_['sym']]['sc']:
                best[c_['sym']] = c_
        ranked = sorted(best.values(), key=lambda x: -x['sc'])

        n_long = sum(1 for p in pos if p['dir'] == 'long')
        n_short = sum(1 for p in pos if p['dir'] == 'short')
        md = max_dir or max_pos

        for c_ in ranked:
            if n_open <= 0: break
            if c_['dir'] == 'long' and n_long >= md: continue
            if c_['dir'] == 'short' and n_short >= md: continue
            notional = cap * 5 / max_pos
            pos.append({'sym': c_['sym'], 'dir': c_['dir'], 'ed': dt,
                        'ep': c_['ep'], 'not': notional, 'sc': c_['sc']})
            if c_['dir'] == 'long': n_long += 1
            else: n_short += 1
            n_open -= 1
        eq.append({'date': dt, 'capital': cap})
    return eq, trades


def run_multi_strategy(data, start, end, strategy_configs):
    """运行多策略并行组合"""
    dates = pd.date_range(start=start, end=end, freq='B')
    cap = INITIAL_CAPITAL
    total_max_pos = 10  # 总最大持仓
    eq, trades, all_pos = [], [], []

    for dt in dates:
        pnl = 0
        keep = []
        for p in all_pos:
            df = data.get(p['sym'])
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
            cfg = p['cfg']
            if p['dir'] == 'long':
                if cfg['sl']:
                    stop = p['ep'] * (1 + cfg['sl'] / 100)
                    if cur_l <= stop:
                        actual_ret = (stop * (1 - sp) - p['ep']) / p['ep'] * 100
                        reason = 'SL'; triggered = True
                if not triggered and cfg['tp']:
                    tp_p = p['ep'] * (1 + cfg['tp'] / 100)
                    if cur_h >= tp_p:
                        actual_ret = (tp_p * (1 - sp) - p['ep']) / p['ep'] * 100
                        reason = 'TP'; triggered = True
                if not triggered:
                    actual_ret = (cur_c - p['ep']) / p['ep'] * 100
            else:
                if cfg['sl']:
                    stop = p['ep'] * (1 - cfg['sl'] / 100)
                    if cur_h >= stop:
                        actual_ret = (p['ep'] - stop * (1 + sp)) / p['ep'] * 100
                        reason = 'SL'; triggered = True
                if not triggered and cfg['tp']:
                    tp_p = p['ep'] * (1 - cfg['tp'] / 100)
                    if cur_l <= tp_p:
                        actual_ret = (p['ep'] - tp_p * (1 + sp)) / p['ep'] * 100
                        reason = 'TP'; triggered = True
                if not triggered:
                    actual_ret = (p['ep'] - cur_c) / p['ep'] * 100
            if d >= cfg['hold']:
                if not triggered: reason = 'exp'
            else:
                if not triggered: keep.append(p); continue
            if reason:
                pnl += p['not'] * actual_ret / 100
                trades.append({
                    'sym': p['sym'], 'dir': p['dir'], 'ed': p['ed'], 'xd': dt,
                    'ep': p['ep'], 'xp': cur_c, 'r': actual_ret,
                    'pnl': p['not'] * actual_ret / 100, 'sc': p['sc'],
                    'hold': d, 'reason': reason, 'strategy': p['strategy'],
                })
        all_pos = keep
        cap += pnl
        if cap <= 0: break

        # 开新仓: 按策略优先级分配
        n_open = total_max_pos - len(all_pos)
        if n_open <= 0:
            eq.append({'date': dt, 'capital': cap}); continue

        occupied_syms = set(p['sym'] for p in all_pos)
        n_long = sum(1 for p in all_pos if p['dir'] == 'long')
        n_short = sum(1 for p in all_pos if p['dir'] == 'short')

        # 每个策略收集候选
        all_cands = []
        for sname, cfg in strategy_configs:
            col_l = f'sig_{sname}_long'
            col_s = f'sig_{sname}_short'
            for sym, df in data.items():
                if sym in occupied_syms: continue
                idx = df.index[df['trade_date'] == dt]
                if len(idx) == 0: continue
                row = df.loc[idx[0]]
                if np.isnan(row.get('close', np.nan)): continue
                sc_l = row.get(col_l, 0)
                sc_s = row.get(col_s, 0)
                if sc_l >= cfg['min_score']:
                    all_cands.append({
                        'sym': sym, 'dir': 'long', 'sc': sc_l, 'ep': row['open'],
                        'strategy': sname, 'cfg': cfg, 'weight': cfg['weight'],
                    })
                if sc_s >= cfg['min_score']:
                    all_cands.append({
                        'sym': sym, 'dir': 'short', 'sc': sc_s, 'ep': row['open'],
                        'strategy': sname, 'cfg': cfg, 'weight': cfg['weight'],
                    })

        # 去重: 每品种取最高加权分
        best = {}
        for c_ in all_cands:
            key = (c_['sym'], c_['strategy'])  # 同品种可被不同策略选中
            weighted_sc = c_['sc'] * c_['weight']
            if key not in best or weighted_sc > best[key]['weighted_sc']:
                c_['weighted_sc'] = weighted_sc
                best[key] = c_

        # 全局排序
        ranked = sorted(best.values(), key=lambda x: -x['weighted_sc'])

        for c_ in ranked:
            if n_open <= 0: break
            if c_['sym'] in occupied_syms: continue  # 已被其他策略占用
            if c_['dir'] == 'long' and n_long >= 6: continue
            if c_['dir'] == 'short' and n_short >= 6: continue
            notional = cap * 5 / total_max_pos * c_['weight']
            all_pos.append({
                'sym': c_['sym'], 'dir': c_['dir'], 'ed': dt,
                'ep': c_['ep'], 'not': notional, 'sc': c_['sc'],
                'strategy': c_['strategy'], 'cfg': c_['cfg'],
            })
            occupied_syms.add(c_['sym'])
            if c_['dir'] == 'long': n_long += 1
            else: n_short += 1
            n_open -= 1
        eq.append({'date': dt, 'capital': cap})
    return eq, trades


def calc_metrics(eq_list, trades_list):
    eq_df = pd.DataFrame(eq_list)
    if len(eq_df) == 0:
        return {'N': 0, 'WR': 0, 'Sharpe': 0, 'MDD': 0, 'Avg': 0}
    tdf = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()
    dr = eq_df['capital'].pct_change().dropna()
    sh = dr.mean() / dr.std() * (252**0.5) if len(dr) > 0 and dr.std() > 0 else 0
    mdd = ((eq_df['capital'] - eq_df['capital'].cummax()) / eq_df['capital'].cummax() * 100).min()
    wr = (tdf['r'] > 0).mean() * 100 if len(tdf) > 0 else 0
    avg = tdf['r'].mean() if len(tdf) > 0 else 0
    return {'N': len(tdf), 'WR': wr, 'Sharpe': sh, 'MDD': mdd, 'Avg': avg}


def main():
    print("V88: 独立多策略并行组合")
    print("=" * 60)

    print("\n加载数据...")
    all_data = load_data()
    print("计算指标和信号...")
    all_data = precompute_indicators(all_data)

    START = '2016-01-01'
    END = '2025-12-31'

    # ═══════════════════════════════════════════
    # 1. 各策略独立表现
    # ═══════════════════════════════════════════
    print(f"\n{'='*60}")
    print("1. 各策略独立回测")
    print(f"{'='*60}")

    strategies = [
        ("gap",    "Gap Fade",      7, 7, 1, -1.5, 4.0, 4),
        ("mom",    "动量突破",       8, 7, 2, -2.0, 6.0, 4),
        ("mr",     "极端均值回归",    8, 5, 2, -2.0, 5.0, 3),
        ("range",  "区间突破",       7, 5, 2, -2.0, 6.0, 3),
        ("oi",     "OI-Price背离",   6, 5, 1, -1.5, 3.0, 3),
    ]

    print(f"\n  {'策略':>15} {'N':>5} {'WR':>6} {'Sharpe':>7} {'MDD':>7} {'Avg':>8}")
    print("  " + "-"*55)

    strat_results = {}
    for sname, slabel, min_sc, max_pos, hold, sl, tp, max_dir in strategies:
        eq, tr = run_strategy(all_data, START, END, sname, f'sig_{sname}', min_sc,
                               max_pos, hold, sl, tp, max_dir)
        m = calc_metrics(eq, tr)
        print(f"  {slabel:>15} {m['N']:>5} {m['WR']:>5.1f}% {m['Sharpe']:>7.2f} "
              f"{m['MDD']:>+6.1f}% {m['Avg']:>+7.3f}%")
        strat_results[sname] = m

    # ═══════════════════════════════════════════
    # 2. 策略相关性分析
    # ═══════════════════════════════════════════
    print(f"\n{'='*60}")
    print("2. 策略信号相关性")
    print(f"{'='*60}")

    # 收集各策略的日度信号方向
    for sym, df in all_data.items():
        df['gap_dir'] = np.where(df['sig_gap_long'] >= 7, 1, np.where(df['sig_gap_short'] >= 7, -1, 0))
        df['mom_dir'] = np.where(df['sig_mom_long'] >= 8, 1, np.where(df['sig_mom_short'] >= 8, -1, 0))
        df['mr_dir'] = np.where(df['sig_mr_long'] >= 8, 1, np.where(df['sig_mr_short'] >= 8, -1, 0))

    # 统计信号重叠
    total_gap = total_mom = total_mr = 0
    overlap_gap_mom = overlap_gap_mr = overlap_mom_mr = 0

    for sym, df in all_data.items():
        mask = (df['trade_date'] >= START) & (df['trade_date'] <= END)
        sub = df[mask]
        gap_active = sub['gap_dir'] != 0
        mom_active = sub['mom_dir'] != 0
        mr_active = sub['mr_dir'] != 0

        total_gap += gap_active.sum()
        total_mom += mom_active.sum()
        total_mr += mr_active.sum()
        overlap_gap_mom += (gap_active & mom_active).sum()
        overlap_gap_mr += (gap_active & mr_active).sum()
        overlap_mom_mr += (mom_active & mr_active).sum()

    print(f"\n  Gap信号总数: {total_gap}")
    print(f"  动量信号总数: {total_mom}")
    print(f"  MR信号总数: {total_mr}")
    if total_gap > 0:
        print(f"\n  Gap∩Mom重叠: {overlap_gap_mom} ({overlap_gap_mom/total_gap*100:.1f}% of Gap)")
        print(f"  Gap∩MR重叠:  {overlap_gap_mr} ({overlap_gap_mr/total_gap*100:.1f}% of Gap)")
    if total_mom > 0:
        print(f"  Mom∩MR重叠:  {overlap_mom_mr} ({overlap_mom_mr/total_mom*100:.1f}% of Mom)")

    # ═══════════════════════════════════════════
    # 3. 多策略并行组合
    # ═══════════════════════════════════════════
    print(f"\n{'='*60}")
    print("3. 多策略并行组合")
    print(f"{'='*60}")

    combo_configs = [
        ("Gap+Mom", [
            ("gap", {'min_score': 7, 'hold': 1, 'sl': -1.5, 'tp': 4.0, 'weight': 1.2}),
            ("mom", {'min_score': 8, 'hold': 2, 'sl': -2.0, 'tp': 6.0, 'weight': 0.8}),
        ]),
        ("Gap+MR", [
            ("gap", {'min_score': 7, 'hold': 1, 'sl': -1.5, 'tp': 4.0, 'weight': 1.2}),
            ("mr",  {'min_score': 8, 'hold': 2, 'sl': -2.0, 'tp': 5.0, 'weight': 0.8}),
        ]),
        ("Gap+OI", [
            ("gap", {'min_score': 7, 'hold': 1, 'sl': -1.5, 'tp': 4.0, 'weight': 1.2}),
            ("oi",  {'min_score': 6, 'hold': 1, 'sl': -1.5, 'tp': 3.0, 'weight': 0.8}),
        ]),
        ("Gap+Mom+MR", [
            ("gap", {'min_score': 7, 'hold': 1, 'sl': -1.5, 'tp': 4.0, 'weight': 1.0}),
            ("mom", {'min_score': 9, 'hold': 2, 'sl': -2.0, 'tp': 6.0, 'weight': 0.7}),
            ("mr",  {'min_score': 9, 'hold': 2, 'sl': -2.0, 'tp': 5.0, 'weight': 0.7}),
        ]),
        ("全策略", [
            ("gap",   {'min_score': 7, 'hold': 1, 'sl': -1.5, 'tp': 4.0, 'weight': 1.0}),
            ("mom",   {'min_score': 9, 'hold': 2, 'sl': -2.0, 'tp': 6.0, 'weight': 0.6}),
            ("mr",    {'min_score': 9, 'hold': 2, 'sl': -2.0, 'tp': 5.0, 'weight': 0.5}),
            ("range", {'min_score': 7, 'hold': 2, 'sl': -2.0, 'tp': 6.0, 'weight': 0.5}),
            ("oi",    {'min_score': 6, 'hold': 1, 'sl': -1.5, 'tp': 3.0, 'weight': 0.6}),
        ]),
    ]

    print(f"\n  {'组合':>15} {'N':>5} {'WR':>6} {'Sharpe':>7} {'MDD':>7} {'Avg':>8}")
    print("  " + "-"*55)

    best_combo = ""
    best_combo_sh = -999
    best_combo_config = None

    for cname, configs in combo_configs:
        eq, tr = run_multi_strategy(all_data, START, END, configs)
        m = calc_metrics(eq, tr)
        print(f"  {cname:>15} {m['N']:>5} {m['WR']:>5.1f}% {m['Sharpe']:>7.2f} "
              f"{m['MDD']:>+6.1f}% {m['Avg']:>+7.3f}%")
        if m['Sharpe'] > best_combo_sh:
            best_combo_sh = m['Sharpe']
            best_combo = cname
            best_combo_config = configs

    # ═══════════════════════════════════════════
    # 4. 最优组合策略分解
    # ═══════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"4. 最优组合 ({best_combo}) 策略分解")
    print(f"{'='*60}")

    eq_best, tr_best = run_multi_strategy(all_data, START, END, best_combo_config)
    m_best = calc_metrics(eq_best, tr_best)

    if tr_best:
        tdf = pd.DataFrame(tr_best)
        print(f"\n  按子策略:")
        for strat in tdf['strategy'].unique():
            sub = tdf[tdf['strategy'] == strat]
            print(f"    {strat:>10s}: N={len(sub):4d} WR={(sub['r']>0).mean()*100:.1f}% "
                  f"Avg={sub['r'].mean():+.3f}% PnL={sub['pnl'].sum():+.0f}")

        # 年度分解
        tdf['year'] = pd.to_datetime(tdf['xd']).dt.year
        print(f"\n  年度表现:")
        for yr in sorted(tdf['year'].unique()):
            s = tdf[tdf['year'] == yr]
            print(f"    {yr}: N={len(s):4d} WR={(s['r']>0).mean()*100:.1f}% Avg={s['r'].mean():+.3f}%")

    # ═══════════════════════════════════════════
    # 5. 最终对比
    # ═══════════════════════════════════════════
    print(f"\n{'='*60}")
    print("5. 最终对比")
    print(f"{'='*60}")
    # 纯Gap Fade
    eq_gap, tr_gap = run_strategy(all_data, START, END, 'gap', 'sig_gap', 7, 7, 1, -1.5, 4.0, 4)
    m_gap = calc_metrics(eq_gap, tr_gap)

    print(f"\n  {'策略':>20s} {'N':>5} {'WR':>6} {'Sharpe':>7} {'MDD':>7} {'Avg':>8}")
    print("  " + "-"*60)
    print(f"  {'Gap Fade (V80)':>20s} {m_gap['N']:>5} {m_gap['WR']:>5.1f}% {m_gap['Sharpe']:>7.2f} "
          f"{m_gap['MDD']:>+6.1f}% {m_gap['Avg']:>+7.3f}%")
    print(f"  {'最优组合 (V88)':>20s} {m_best['N']:>5} {m_best['WR']:>5.1f}% {m_best['Sharpe']:>7.2f} "
          f"{m_best['MDD']:>+6.1f}% {m_best['Avg']:>+7.3f}%")

    # 各独立策略
    print(f"\n  独立策略:")
    for sname, slabel, min_sc, max_pos, hold, sl, tp, max_dir in strategies:
        if sname in strat_results:
            m = strat_results[sname]
            print(f"    {slabel:>15s}: N={m['N']:5d} WR={m['WR']:.1f}% Sharpe={m['Sharpe']:.2f}")


if __name__ == '__main__':
    main()
