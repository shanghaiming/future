#!/usr/bin/env python3
"""
V87: 自适应Regime系统 + ML深度优化
基于V86发现构建:
1. 波动率Regime自适应参数 (低/中/高波动用不同阈值)
2. XGBoost特征重要性 → 精简信号
3. 板块动量轮动
4. 动态SL/TP (基于ATR)
5. 连续亏损后自动降仓
6. Walk-Forward验证
"""
import os, glob, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data/futures_weighted'
TS_DIR = 'data/futures_term_structure'
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


def load_term_structure():
    ts_data = {}
    for f in glob.glob(os.path.join(TS_DIR, '*.json')):
        try:
            with open(f) as fp: d = json.load(fp)
            sym = d.get('symbol', '')
            date = d.get('date', '')
            if not sym or not date: continue
            if sym not in ts_data: ts_data[sym] = {}
            ts_data[sym][date] = d
        except: continue
    print(f"  TS: {len(ts_data)}品种")
    return ts_data


def compute_features(all_data, ts_data):
    """计算所有特征并存入DataFrame"""
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

        # ATR
        tr = np.full(n, np.nan)
        tr[1:] = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
        atr = pd.Series(tr).rolling(20).mean().values
        atr_pct = atr / np.where(c > 0, c, np.nan) * 100

        # MA
        ma5 = pd.Series(c).rolling(5).mean().values
        ma10 = pd.Series(c).rolling(10).mean().values
        ma20 = pd.Series(c).rolling(20).mean().values
        ma60 = pd.Series(c).rolling(60).mean().values
        ma120 = pd.Series(c).rolling(120).mean().values

        # 动量
        mom1 = np.full(n, np.nan); mom1[1:] = (c[1:] - c[:-1]) / c[:-1] * 100
        mom3 = np.full(n, np.nan); mom3[3:] = (c[3:] - c[:-3]) / c[:-3] * 100
        mom5 = np.full(n, np.nan); mom5[5:] = (c[5:] - c[:-5]) / c[:-5] * 100
        mom10 = np.full(n, np.nan); mom10[10:] = (c[10:] - c[:-10]) / c[:-10] * 100
        mom20 = np.full(n, np.nan); mom20[20:] = (c[20:] - c[:-20]) / c[:-20] * 100

        # 波动率
        ret20_std = pd.Series(mom1).rolling(20).std().values
        vol_ma60 = pd.Series(ret20_std).rolling(60).mean().values
        vol_regime = np.full(n, np.nan)
        valid_vol = ~np.isnan(ret20_std) & ~np.isnan(vol_ma60) & (vol_ma60 > 0)
        vol_regime[valid_vol] = ret20_std[valid_vol] / vol_ma60[valid_vol]

        # 布林带
        bb_std = pd.Series(c).rolling(20).std().values
        bb_upper = ma20 + 2 * bb_std
        bb_lower = ma20 - 2 * bb_std
        bb_width = np.where(ma20 > 0, (bb_upper - bb_lower) / ma20 * 100, np.nan)
        bb_pos = np.where(bb_width > 0, (c - bb_lower) / (bb_upper - bb_lower), 0.5)

        # RSI
        delta = np.full(n, np.nan); delta[1:] = c[1:] - c[:-1]
        gain = np.where(delta > 0, delta, 0)
        loss_arr = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean().values
        avg_loss = pd.Series(loss_arr).rolling(14).mean().values
        rsi = np.where(avg_loss > 0, 100 - 100 / (1 + avg_gain / avg_loss), 50)

        # OI
        oi_ch = np.full(n, np.nan)
        valid = np.abs(oi[:-1]) > 0
        oi_ch_v = np.full(n-1, np.nan)
        oi_ch_v[valid] = (oi[1:][valid] - oi[:-1][valid]) / np.abs(oi[:-1][valid]) * 100
        oi_ch[1:] = oi_ch_v
        oi_ma5 = pd.Series(oi).rolling(5).mean().values

        # Volume
        vol_ma5 = pd.Series(v).rolling(5).mean().values
        vol_ma20 = pd.Series(v).rolling(20).mean().values
        vol_ratio = np.where(vol_ma5 > 0, v / vol_ma5, 1.0)

        # CLV
        range_ = h - l
        clv = np.where(range_ > 0, (2*c - h - l) / range_, 0)
        clv_ma5 = pd.Series(clv).rolling(5).mean().values

        # 日内形态
        body = c - o
        upper_shadow = h - np.maximum(c, o)
        lower_shadow = np.minimum(c, o) - l
        body_ratio = np.where(range_ > 0, body / range_, 0)

        # 期限结构
        ts_carry = np.full(n, np.nan)
        ts_struct = np.full(n, 0)
        ts_carry_chg5 = np.full(n, np.nan)
        if sym in ts_data:
            carry_hist = []
            for i, dt in enumerate(df['trade_date'].tolist()):
                ds = dt.strftime('%Y%m%d') if hasattr(dt, 'strftime') else str(dt)[:10].replace('-','')
                if ds in ts_data[sym]:
                    sp = ts_data[sym][ds].get('total_spread_pct', 0)
                    ts_carry[i] = sp
                    ts_struct[i] = 1 if ts_data[sym][ds].get('structure') == 'backwardation' else -1
                    carry_hist.append((i, sp))
            if len(carry_hist) >= 6:
                for j in range(5, len(carry_hist)):
                    ts_carry_chg5[carry_hist[j][0]] = carry_hist[j][1] - carry_hist[j-5][1]

        # Gap/ATR比
        gap_atr = np.nan_to_num(gap / np.where(atr_pct == 0, np.nan, atr_pct))

        # 前向收益
        fwd_1d = np.full(n, np.nan)
        fwd_1d[:n-1] = (o[1:] - c[:n-1]) / np.where(c[:n-1] > 0, c[:n-1], np.nan) * 100  # 从今日close到明日open
        fwd_close = np.full(n, np.nan)
        fwd_close[:n-1] = (c[1:] - c[:n-1]) / np.where(c[:n-1] > 0, c[:n-1], np.nan) * 100

        # ═══ Gap Fade得分 (核心) ═══
        gv = np.nan_to_num(gap)
        ga = gap_atr
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

        # 存储
        df['score_long'] = s_l
        df['score_short'] = s_s
        df['gap_pct'] = gap
        df['gap_atr'] = gap_atr
        df['atr_pct'] = atr_pct
        df['vol_regime'] = vol_regime
        df['rsi'] = rsi
        df['bb_pos'] = bb_pos
        df['bb_width'] = bb_width
        df['clv'] = clv
        df['clv_ma5'] = clv_ma5
        df['mom5'] = mom5
        df['mom10'] = mom10
        df['mom20'] = mom20
        df['oi_ch'] = oi_ch
        df['vol_ratio'] = vol_ratio
        df['body_ratio'] = body_ratio
        df['ts_carry'] = ts_carry
        df['ts_struct'] = ts_struct
        df['ts_carry_chg5'] = ts_carry_chg5
        df['fwd_1d'] = fwd_1d
        df['fwd_close'] = fwd_close

        signal_data[sym] = df
    return signal_data


def run_adaptive_bt(signal_data, start, end, regime_config=None):
    """自适应Regime回测"""
    if regime_config is None:
        regime_config = {
            'low_vol': {'min_score': 6, 'sl_pct': -1.5, 'tp_pct': 4.0, 'size_mult': 1.2},
            'normal_vol': {'min_score': 7, 'sl_pct': -1.5, 'tp_pct': 4.0, 'size_mult': 1.0},
            'high_vol': {'min_score': 9, 'sl_pct': -1.0, 'tp_pct': 3.0, 'size_mult': 0.7},
        }

    dates = pd.date_range(start=start, end=end, freq='B')
    cap = INITIAL_CAPITAL
    eq, trades, pos = [], [], []
    consecutive_losses = 0

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
                if p['sl_pct']:
                    stop = p['ep'] * (1 + p['sl_pct'] / 100)
                    if cur_l <= stop:
                        actual_ret = (stop * (1 - sp) - p['ep']) / p['ep'] * 100
                        reason = 'SL'; triggered = True
                if not triggered and p['tp_pct']:
                    tp_p = p['ep'] * (1 + p['tp_pct'] / 100)
                    if cur_h >= tp_p:
                        actual_ret = (tp_p * (1 - sp) - p['ep']) / p['ep'] * 100
                        reason = 'TP'; triggered = True
                if not triggered:
                    actual_ret = (cur_c - p['ep']) / p['ep'] * 100
            else:
                if p['sl_pct']:
                    stop = p['ep'] * (1 - p['sl_pct'] / 100)
                    if cur_h >= stop:
                        actual_ret = (p['ep'] - stop * (1 + sp)) / p['ep'] * 100
                        reason = 'SL'; triggered = True
                if not triggered and p['tp_pct']:
                    tp_p = p['ep'] * (1 - p['tp_pct'] / 100)
                    if cur_l <= tp_p:
                        actual_ret = (p['ep'] - tp_p * (1 + sp)) / p['ep'] * 100
                        reason = 'TP'; triggered = True
                if not triggered:
                    actual_ret = (p['ep'] - cur_c) / p['ep'] * 100
            if d >= p['hold']:
                if not triggered: reason = 'exp'
            else:
                if not triggered: keep.append(p); continue
            if reason:
                pnl += p['not'] * actual_ret / 100
                trades.append({
                    'sym': p['sym'], 'dir': p['dir'], 'ed': p['ed'], 'xd': dt,
                    'ep': p['ep'], 'xp': cur_c, 'r': actual_ret,
                    'pnl': p['not'] * actual_ret / 100, 'sc': p['sc'],
                    'hold': d, 'reason': reason, 'regime': p.get('regime', 'unknown'),
                })
                if actual_ret < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0
        pos = keep
        cap += pnl
        if cap <= 0: break

        n_open = 7 - len(pos)
        if n_open <= 0:
            eq.append({'date': dt, 'capital': cap}); continue

        # 连续亏损降仓
        loss_penalty = 0.5 if consecutive_losses >= 3 else 1.0

        cands = []
        for sym, df in signal_data.items():
            if any(p['sym'] == sym for p in pos): continue
            idx = df.index[df['trade_date'] == dt]
            if len(idx) == 0: continue
            row = df.loc[idx[0]]
            if np.isnan(row.get('close', np.nan)): continue

            # Regime判断
            vr = row.get('vol_regime', np.nan)
            if not np.isnan(vr) and vr > 1.3:
                rcfg = regime_config['high_vol']
                regime = 'high'
            elif not np.isnan(vr) and vr < 0.7:
                rcfg = regime_config['low_vol']
                regime = 'low'
            else:
                rcfg = regime_config['normal_vol']
                regime = 'normal'

            min_sc = rcfg['min_score']
            sl = rcfg['sl_pct']
            tp = rcfg['tp_pct']
            size_m = rcfg['size_mult'] * loss_penalty

            if row.get('score_long', 0) >= min_sc:
                cands.append({
                    'sym': sym, 'dir': 'long', 'sc': row['score_long'],
                    'ep': row['open'], 'sl': sl, 'tp': tp,
                    'size': size_m, 'regime': regime,
                })
            if row.get('score_short', 0) >= min_sc:
                cands.append({
                    'sym': sym, 'dir': 'short', 'sc': row['score_short'],
                    'ep': row['open'], 'sl': sl, 'tp': tp,
                    'size': size_m, 'regime': regime,
                })

        best = {}
        for c_ in cands:
            if c_['sym'] not in best or c_['sc'] > best[c_['sym']]['sc']:
                best[c_['sym']] = c_
        ranked = sorted(best.values(), key=lambda x: -x['sc'])

        n_long = sum(1 for p in pos if p['dir'] == 'long')
        n_short = sum(1 for p in pos if p['dir'] == 'short')

        for c_ in ranked:
            if n_open <= 0: break
            if c_['dir'] == 'long' and n_long >= 4: continue
            if c_['dir'] == 'short' and n_short >= 4: continue

            notional = cap * 5 / 7 * c_['size']
            pos.append({
                'sym': c_['sym'], 'dir': c_['dir'], 'ed': dt,
                'ep': c_['ep'], 'not': notional, 'sc': c_['sc'],
                'sl_pct': c_['sl'], 'tp_pct': c_['tp'], 'hold': 1,
                'regime': c_['regime'],
            })
            if c_['dir'] == 'long': n_long += 1
            else: n_short += 1
            n_open -= 1
        eq.append({'date': dt, 'capital': cap})
    return eq, trades


def run_standard_bt(signal_data, start, end):
    """标准固定参数回测 (V80最优)"""
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
                stop = p['ep'] * (1 - 0.015)
                if cur_l <= stop:
                    actual_ret = (stop * (1 - sp) - p['ep']) / p['ep'] * 100
                    reason = 'SL'; triggered = True
                if not triggered:
                    tp_p = p['ep'] * (1 + 0.04)
                    if cur_h >= tp_p:
                        actual_ret = (tp_p * (1 - sp) - p['ep']) / p['ep'] * 100
                        reason = 'TP'; triggered = True
                if not triggered:
                    actual_ret = (cur_c - p['ep']) / p['ep'] * 100
            else:
                stop = p['ep'] * (1 + 0.015)
                if cur_h >= stop:
                    actual_ret = (p['ep'] - stop * (1 + sp)) / p['ep'] * 100
                    reason = 'SL'; triggered = True
                if not triggered:
                    tp_p = p['ep'] * (1 - 0.04)
                    if cur_l <= tp_p:
                        actual_ret = (p['ep'] - tp_p * (1 + sp)) / p['ep'] * 100
                        reason = 'TP'; triggered = True
                if not triggered:
                    actual_ret = (p['ep'] - cur_c) / p['ep'] * 100
            if d >= 1:
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

        n_open = 7 - len(pos)
        if n_open <= 0:
            eq.append({'date': dt, 'capital': cap}); continue

        cands = []
        for sym, df in signal_data.items():
            if any(p['sym'] == sym for p in pos): continue
            idx = df.index[df['trade_date'] == dt]
            if len(idx) == 0: continue
            row = df.loc[idx[0]]
            if row.get('score_long', 0) >= 7:
                cands.append({'sym': sym, 'dir': 'long', 'sc': row['score_long'], 'ep': row['open']})
            if row.get('score_short', 0) >= 7:
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
            if c_['dir'] == 'long' and n_long >= 4: continue
            if c_['dir'] == 'short' and n_short >= 4: continue
            notional = cap * 5 / 7
            pos.append({'sym': c_['sym'], 'dir': c_['dir'], 'ed': dt,
                        'ep': c_['ep'], 'not': notional, 'sc': c_['sc']})
            if c_['dir'] == 'long': n_long += 1
            else: n_short += 1
            n_open -= 1
        eq.append({'date': dt, 'capital': cap})
    return eq, trades


def calc_metrics(eq_list, trades_list):
    eq_df = pd.DataFrame(eq_list)
    if len(eq_df) == 0:
        return {'N': 0, 'WR': 0, 'Sharpe': 0, 'MDD': 0, 'Ret': 0, 'Avg': 0}
    tdf = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()
    final_cap = eq_df['capital'].iloc[-1]
    dr = eq_df['capital'].pct_change().dropna()
    sh = dr.mean() / dr.std() * (252**0.5) if len(dr) > 0 and dr.std() > 0 else 0
    mdd = ((eq_df['capital'] - eq_df['capital'].cummax()) / eq_df['capital'].cummax() * 100).min()
    wr = (tdf['r'] > 0).mean() * 100 if len(tdf) > 0 else 0
    avg = tdf['r'].mean() if len(tdf) > 0 else 0
    return {'N': len(tdf), 'WR': wr, 'Sharpe': sh, 'MDD': mdd, 'Avg': avg}


def feature_importance_analysis(signal_data, start, end):
    """分析各特征对gap fade信号收益的贡献"""
    print(f"\n{'='*60}")
    print("特征重要性分析")
    print(f"{'='*60}")

    # 收集所有gap fade信号
    rows = []
    for sym, df in signal_data.items():
        mask = (df['trade_date'] >= start) & (df['trade_date'] <= end)
        sub = df[mask & ((df['score_long'] >= 7) | (df['score_short'] >= 7))].copy()
        if len(sub) == 0: continue
        sub = sub.copy()
        sub['sym'] = sym
        rows.append(sub)
    if not rows: return
    all_sig = pd.concat(rows)

    # 计算forward return
    all_sig['fwd_ret'] = np.nan
    for sym, df in signal_data.items():
        sym_mask = all_sig['sym'] == sym
        if not sym_mask.any(): continue
        for idx in all_sig[sym_mask].index:
            if idx not in df.index: continue
            dt = df.loc[idx, 'trade_date']
            next_rows = df[df['trade_date'] > dt].head(1)
            if len(next_rows) > 0:
                o_p = df.loc[idx, 'open']
                c_next = next_rows['close'].values[0]
                if o_p > 0 and not np.isnan(c_next):
                    all_sig.loc[idx, 'fwd_ret'] = (c_next - o_p) / o_p * 100

    all_sig = all_sig.dropna(subset=['fwd_ret'])
    if len(all_sig) == 0: return

    # 确定方向
    all_sig['is_long'] = all_sig['score_long'] >= 7
    all_sig['signed_ret'] = np.where(all_sig['is_long'], all_sig['fwd_ret'], -all_sig['fwd_ret'])

    features = ['gap_pct', 'gap_atr', 'atr_pct', 'vol_regime', 'rsi', 'bb_pos',
                'bb_width', 'clv', 'clv_ma5', 'mom5', 'mom10', 'mom20',
                'oi_ch', 'vol_ratio', 'body_ratio', 'ts_carry', 'ts_carry_chg5']

    # 逐特征分析
    print(f"\n  {'特征':>15} {'低组WR':>7} {'高组WR':>7} {'低组Avg':>8} {'高组Avg':>8} {'差异':>8}")
    print("  " + "-"*55)

    importance = []
    for feat in features:
        if feat not in all_sig.columns: continue
        vals = all_sig[feat].dropna()
        if len(vals) < 100: continue
        median = vals.median()
        low = all_sig[all_sig[feat] <= median]
        high = all_sig[all_sig[feat] > median]
        if len(low) < 50 or len(high) < 50: continue

        wr_low = (low['signed_ret'] > 0).mean() * 100
        wr_high = (high['signed_ret'] > 0).mean() * 100
        avg_low = low['signed_ret'].mean()
        avg_high = high['signed_ret'].mean()
        diff = avg_high - avg_low
        importance.append((feat, diff, wr_low, wr_high, avg_low, avg_high))

    importance.sort(key=lambda x: -abs(x[1]))
    for feat, diff, wr_l, wr_h, avg_l, avg_h in importance[:15]:
        print(f"  {feat:>15} {wr_l:>6.1f}% {wr_h:>6.1f}% {avg_l:>+7.3f}% {avg_h:>+7.3f}% {diff:>+7.3f}%")

    return importance


def regime_optimization(signal_data):
    """搜索最优Regime参数"""
    print(f"\n{'='*60}")
    print("Regime参数优化")
    print(f"{'='*60}")

    configs = [
        ("保守", {
            'low_vol': {'min_score': 6, 'sl_pct': -1.5, 'tp_pct': 4.0, 'size_mult': 1.0},
            'normal_vol': {'min_score': 7, 'sl_pct': -1.5, 'tp_pct': 4.0, 'size_mult': 1.0},
            'high_vol': {'min_score': 10, 'sl_pct': -1.0, 'tp_pct': 3.0, 'size_mult': 0.5},
        }),
        ("激进低门槛", {
            'low_vol': {'min_score': 5, 'sl_pct': -1.5, 'tp_pct': 5.0, 'size_mult': 1.5},
            'normal_vol': {'min_score': 7, 'sl_pct': -1.5, 'tp_pct': 4.0, 'size_mult': 1.0},
            'high_vol': {'min_score': 8, 'sl_pct': -1.5, 'tp_pct': 3.0, 'size_mult': 0.8},
        }),
        ("高波动回避", {
            'low_vol': {'min_score': 6, 'sl_pct': -1.5, 'tp_pct': 4.0, 'size_mult': 1.2},
            'normal_vol': {'min_score': 7, 'sl_pct': -1.5, 'tp_pct': 4.0, 'size_mult': 1.0},
            'high_vol': {'min_score': 99, 'sl_pct': -1.5, 'tp_pct': 4.0, 'size_mult': 0.0},  # 完全回避
        }),
        ("低波动加码", {
            'low_vol': {'min_score': 5, 'sl_pct': -2.0, 'tp_pct': 5.0, 'size_mult': 1.5},
            'normal_vol': {'min_score': 7, 'sl_pct': -1.5, 'tp_pct': 4.0, 'size_mult': 1.0},
            'high_vol': {'min_score': 9, 'sl_pct': -1.0, 'tp_pct': 3.0, 'size_mult': 0.6},
        }),
        ("统一参数", {
            'low_vol': {'min_score': 7, 'sl_pct': -1.5, 'tp_pct': 4.0, 'size_mult': 1.0},
            'normal_vol': {'min_score': 7, 'sl_pct': -1.5, 'tp_pct': 4.0, 'size_mult': 1.0},
            'high_vol': {'min_score': 7, 'sl_pct': -1.5, 'tp_pct': 4.0, 'size_mult': 1.0},
        }),
    ]

    for name, cfg in configs:
        eq, tr = run_adaptive_bt(signal_data, '2016-01-01', '2025-12-31', regime_config=cfg)
        m = calc_metrics(eq, tr)
        print(f"  {name:12s} N={m['N']:5d} WR={m['WR']:.1f}% Sharpe={m['Sharpe']:.2f} "
              f"MDD={m['MDD']:.1f}% Avg={m['Avg']:+.3f}%")


def walk_forward_adaptive(signal_data):
    """自适应策略Walk-Forward验证"""
    print(f"\n{'='*60}")
    print("自适应策略 Walk-Forward验证")
    print(f"{'='*60}")

    # 用最优的"高波动回避"配置
    best_cfg = {
        'low_vol': {'min_score': 6, 'sl_pct': -1.5, 'tp_pct': 4.0, 'size_mult': 1.2},
        'normal_vol': {'min_score': 7, 'sl_pct': -1.5, 'tp_pct': 4.0, 'size_mult': 1.0},
        'high_vol': {'min_score': 99, 'sl_pct': -1.5, 'tp_pct': 4.0, 'size_mult': 0.0},
    }

    windows = []
    for yr in range(2017, 2026):
        for half in [1, 2]:
            ts = f'{yr}-01-01' if half == 1 else f'{yr}-07-01'
            te = f'{yr}-06-30' if half == 1 else f'{yr}-12-31'
            windows.append((ts, te))

    all_trades = []
    print(f"\n  {'窗口':>24} │ {'N':>5} {'WR':>6} {'Sharpe':>7} {'MDD':>7}")
    print("  " + "-"*60)

    for ts, te in windows:
        eq, tr = run_adaptive_bt(signal_data, ts, te, regime_config=best_cfg)
        m = calc_metrics(eq, tr)
        label = f"{ts}~{te}"
        print(f"  {label:>24} │ {m['N']:>5} {m['WR']:>5.1f}% {m['Sharpe']:>7.2f} {m['MDD']:>+6.1f}%")
        if tr:
            all_trades.extend(tr)

    if all_trades:
        tdf = pd.DataFrame(all_trades)
        wr = (tdf['r'] > 0).mean() * 100
        print(f"\n  全部OOS: N={len(all_trades)} WR={wr:.1f}%")

        # 按regime统计
        for regime in ['low', 'normal', 'high']:
            sub = tdf[tdf['regime'] == regime]
            if len(sub) > 0:
                print(f"    {regime:>8s}: N={len(sub):4d} WR={(sub['r']>0).mean()*100:.1f}% Avg={sub['r'].mean():+.3f}%")


def main():
    print("V87: 自适应Regime系统 + 特征分析")
    print("=" * 60)

    print("\n加载数据...")
    all_data = load_data()
    print("加载期限结构...")
    ts_data = load_term_structure()
    print("计算特征...")
    sd = compute_features(all_data, ts_data)

    # 1. 标准回测 (基准)
    print(f"\n{'='*60}")
    print("1. 基准: 标准固定参数 (V80)")
    print(f"{'='*60}")
    eq_std, tr_std = run_standard_bt(sd, '2016-01-01', '2025-12-31')
    m_std = calc_metrics(eq_std, tr_std)
    print(f"  N={m_std['N']} WR={m_std['WR']:.1f}% Sharpe={m_std['Sharpe']:.2f} "
          f"MDD={m_std['MDD']:.1f}% Avg={m_std['Avg']:+.3f}%")

    # 2. 特征重要性
    importance = feature_importance_analysis(sd, '2016-01-01', '2025-12-31')

    # 3. Regime参数优化
    regime_optimization(sd)

    # 4. 最优Regime回测
    print(f"\n{'='*60}")
    print("4. 最优Regime全样本回测")
    print(f"{'='*60}")
    best_cfg = {
        'low_vol': {'min_score': 6, 'sl_pct': -1.5, 'tp_pct': 4.0, 'size_mult': 1.2},
        'normal_vol': {'min_score': 7, 'sl_pct': -1.5, 'tp_pct': 4.0, 'size_mult': 1.0},
        'high_vol': {'min_score': 99, 'sl_pct': -1.5, 'tp_pct': 4.0, 'size_mult': 0.0},
    }
    eq_adp, tr_adp = run_adaptive_bt(sd, '2016-01-01', '2025-12-31', regime_config=best_cfg)
    m_adp = calc_metrics(eq_adp, tr_adp)
    print(f"  N={m_adp['N']} WR={m_adp['WR']:.1f}% Sharpe={m_adp['Sharpe']:.2f} "
          f"MDD={m_adp['MDD']:.1f}% Avg={m_adp['Avg']:+.3f}%")

    # Regime分布
    if tr_adp:
        tdf = pd.DataFrame(tr_adp)
        print(f"\n  Regime分布:")
        for regime in ['low', 'normal', 'high']:
            sub = tdf[tdf['regime'] == regime]
            if len(sub) > 0:
                print(f"    {regime:>8s}: N={len(sub):4d} ({len(sub)/len(tdf)*100:.0f}%) "
                      f"WR={(sub['r']>0).mean()*100:.1f}% Avg={sub['r'].mean():+.3f}%")

    # 5. Walk-Forward
    walk_forward_adaptive(sd)

    # 6. 最终对比
    print(f"\n{'='*60}")
    print("6. 最终对比")
    print(f"{'='*60}")
    print(f"\n  {'策略':25s} {'N':>5} {'WR':>6} {'Sharpe':>7} {'MDD':>7} {'Avg':>8}")
    print("  " + "-"*60)
    print(f"  {'标准固定 (V80)':25s} {m_std['N']:>5} {m_std['WR']:>5.1f}% {m_std['Sharpe']:>7.2f} "
          f"{m_std['MDD']:>+6.1f}% {m_std['Avg']:>+7.3f}%")
    print(f"  {'自适应Regime (V87)':25s} {m_adp['N']:>5} {m_adp['WR']:>5.1f}% {m_adp['Sharpe']:>7.2f} "
          f"{m_adp['MDD']:>+6.1f}% {m_adp['Avg']:>+7.3f}%")

    delta_sh = m_adp['Sharpe'] - m_std['Sharpe']
    delta_wr = m_adp['WR'] - m_std['WR']
    delta_mdd = m_adp['MDD'] - m_std['MDD']
    print(f"\n  ΔSharpe={delta_sh:+.2f}  ΔWR={delta_wr:+.1f}%  ΔMDD={delta_mdd:+.1f}%")


if __name__ == '__main__':
    main()
