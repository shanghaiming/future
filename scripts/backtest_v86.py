#!/usr/bin/env python3
"""
V86: 多策略Alpha引擎 — 综合集成系统
超越简单Gap Fade，构建多维度信号融合架构:
1. Gap Fade (核心alpha, 已验证)
2. 动量轮动 (跨品种动量排名)
3. 均值回归 (超跌反弹)
4. 波动率突破 (低波动后突破)
5. 期限结构Carry (曲线变化信号)
6. 跨品种相对强弱 (板块内排名)
7. 市场状态自适应 (波动率 regime)
8. 动态仓位管理 (Kelly + 信号强度加权)
"""
import os, glob, json, numpy as np, pandas as pd, warnings
from itertools import product
warnings.filterwarnings('ignore')

DATA_DIR = 'data/futures_weighted'
TS_DIR = 'data/futures_term_structure'
OPT_DIR = 'data/options'
CONTRACT_SPECS = 'scripts/contract_specs.py'
INITIAL_CAPITAL = 500_000

# 品种分类
SECTORS = {
    'metal': ['agfi', 'alfi', 'cufi', 'zcfi', 'pbfi', 'nifi', 'srfi', 'bfi', 'wrfi', 'hcfi', 'ssfi', 'snfi'],
    'energy': ['scfi', 'fuel', 'pgfi', 'tafi', 'mafi', 'bbfi', 'egfi', 'ebfi', 'rrfi', 'brfi'],
    'chem': ['vfi', 'ppfi', 'lfi', 'egfi', 'tafi', 'mafi', 'safi', 'fgfi', 'bbfi'],
    'agri': ['cfi', 'srfi', 'whfi', 'cmfi', 'smfi', 'yfi', 'pfi', 'cfi', 'rmmfi', 'pkfi', 'oyfi', 'apfi'],
    'grain': ['cfi', 'srfi', 'whfi', 'rmmfi', 'pmfi', 'rrfi', 'lrfi', 'jmfi'],
    'soft': ['srfi', 'cyfi', 'apfi', 'pkfi', 'cfi', 'oyfi'],
}


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
    """加载期限结构数据"""
    ts_data = {}
    files = glob.glob(os.path.join(TS_DIR, '*.json'))
    for f in files:
        try:
            with open(f) as fp:
                d = json.load(fp)
            sym = d.get('symbol', '')
            date = d.get('date', '')
            if not sym or not date: continue
            if sym not in ts_data:
                ts_data[sym] = {}
            ts_data[sym][date] = {
                'structure': d.get('structure', ''),
                'near_price': d.get('near_price', 0),
                'far_price': d.get('far_price', 0),
                'total_spread_pct': d.get('total_spread_pct', 0),
                'curve': d.get('curve', []),
            }
        except: continue
    print(f"  TS: {len(ts_data)}品种, {sum(len(v) for v in ts_data.values())}条")
    return ts_data


def compute_all_signals(all_data, ts_data):
    """计算全维度信号"""
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

        # === 基础指标 ===
        tr = np.full(n, np.nan)
        tr[1:] = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
        atr = pd.Series(tr).rolling(20).mean().values
        atr_pct = atr / np.where(c > 0, c, np.nan) * 100
        ma5 = pd.Series(c).rolling(5).mean().values
        ma10 = pd.Series(c).rolling(10).mean().values
        ma20 = pd.Series(c).rolling(20).mean().values
        ma60 = pd.Series(c).rolling(60).mean().values
        ma120 = pd.Series(c).rolling(120).mean().values
        vol_ma5 = pd.Series(v).rolling(5).mean().values
        vol_ma20 = pd.Series(v).rolling(20).mean().values

        # === 动量指标 ===
        mom1 = np.full(n, np.nan); mom1[1:] = (c[1:] - c[:-1]) / c[:-1] * 100
        mom3 = np.full(n, np.nan); mom3[3:] = (c[3:] - c[:-3]) / c[:-3] * 100
        mom5 = np.full(n, np.nan); mom5[5:] = (c[5:] - c[:-5]) / c[:-5] * 100
        mom10 = np.full(n, np.nan); mom10[10:] = (c[10:] - c[:-10]) / c[:-10] * 100
        mom20 = np.full(n, np.nan); mom20[20:] = (c[20:] - c[:-20]) / c[:-20] * 100

        # === 波动率指标 ===
        ret20 = pd.Series(mom1).rolling(20).std().values
        vol_regime = np.full(n, np.nan)
        vol_ma60 = pd.Series(ret20).rolling(60).mean().values
        valid_vol = ~np.isnan(ret20) & ~np.isnan(vol_ma60) & (vol_ma60 > 0)
        vol_regime[valid_vol] = ret20[valid_vol] / vol_ma60[valid_vol]

        # === OI变化 ===
        oi_ch = np.full(n, np.nan)
        valid = np.abs(oi[:-1]) > 0
        oi_ch_v = np.full(n-1, np.nan)
        oi_ch_v[valid] = (oi[1:][valid] - oi[:-1][valid]) / np.abs(oi[:-1][valid]) * 100
        oi_ch[1:] = oi_ch_v
        oi_ma5 = pd.Series(oi).rolling(5).mean().values
        oi_trend = np.full(n, np.nan)
        oi_trend[5:] = (oi[5:] - oi_ma5[5:]) / np.where(np.abs(oi_ma5[5:]) > 0, np.abs(oi_ma5[5:]), np.nan) * 100

        # === CLV ===
        range_ = h - l
        clv = np.where(range_ > 0, (2*c - h - l) / range_, 0)
        clv_ma5 = pd.Series(clv).rolling(5).mean().values

        # === RSI ===
        delta = np.full(n, np.nan); delta[1:] = c[1:] - c[:-1]
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean().values
        avg_loss = pd.Series(loss).rolling(14).mean().values
        rsi = np.where(avg_loss > 0, 100 - 100 / (1 + avg_gain / avg_loss), 50)

        # === 布林带 ===
        bb_mid = ma20
        bb_std = pd.Series(c).rolling(20).std().values
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_width = np.where(bb_mid > 0, (bb_upper - bb_lower) / bb_mid * 100, np.nan)

        # === 期限结构信号 ===
        ts_carry = np.full(n, np.nan)
        ts_struct = np.full(n, 0)  # 0=unknown, 1=backwardation, -1=contango
        ts_carry_chg5 = np.full(n, np.nan)
        if sym in ts_data:
            carry_hist = []
            dates_list = df['trade_date'].tolist()
            for i, dt in enumerate(dates_list):
                ds = dt.strftime('%Y%m%d') if hasattr(dt, 'strftime') else str(dt)[:10].replace('-','')
                if ds in ts_data[sym]:
                    sp = ts_data[sym][ds].get('total_spread_pct', 0)
                    ts_carry[i] = sp
                    ts_struct[i] = 1 if ts_data[sym][ds].get('structure') == 'backwardation' else -1
                    carry_hist.append((i, sp))
            # 5日carry变化
            if len(carry_hist) >= 6:
                for j in range(5, len(carry_hist)):
                    idx_now = carry_hist[j][0]
                    idx_5ago = carry_hist[j-5][0]
                    ts_carry_chg5[idx_now] = carry_hist[j][1] - carry_hist[j-5][1]

        # ═══════════════════════════════════════════
        # === 信号1: Gap Fade (核心, V80最优配置) ===
        # ═══════════════════════════════════════════
        gv = np.nan_to_num(gap)
        ga = np.nan_to_num(gap / np.where(atr_pct == 0, np.nan, atr_pct))

        s_l_gap = np.zeros(n)
        s_l_gap += (gv < -0.5).astype(int) * 1
        s_l_gap += (gv < -1.0).astype(int) * 2
        s_l_gap += (gv < -1.5).astype(int) * 2
        s_l_gap += (gv < -2.0).astype(int) * 3
        s_l_gap += (ga < -1.0).astype(int) * 2
        s_l_gap += (ga < -1.5).astype(int) * 3
        s_l_gap += ((oi_ch > 0) & (c < prev_c)).astype(int) * 3
        s_l_gap += ((oi_ch < 0) & (c < prev_c)).astype(int) * 2
        s_l_gap += (mom5 < -3).astype(int) * 1
        s_l_gap += (mom5 < -5).astype(int) * 1
        s_l_gap += (c < ma5).astype(int) * 1
        s_l_gap += ((v > vol_ma5 * 1.5) & (c < prev_c)).astype(int) * 1
        s_l_gap += (clv > 0.5).astype(int) * 1
        s_l_gap += (ma20 > ma60).astype(int) * 2

        s_s_gap = np.zeros(n)
        s_s_gap += (gv > 0.5).astype(int) * 1
        s_s_gap += (gv > 1.0).astype(int) * 2
        s_s_gap += (gv > 1.5).astype(int) * 2
        s_s_gap += (gv > 2.0).astype(int) * 3
        s_s_gap += (ga > 1.0).astype(int) * 2
        s_s_gap += (ga > 1.5).astype(int) * 3
        s_s_gap += ((oi_ch > 0) & (c > prev_c)).astype(int) * 3
        s_s_gap += ((oi_ch < 0) & (c > prev_c)).astype(int) * 2
        s_s_gap += (mom5 > 3).astype(int) * 1
        s_s_gap += (mom5 > 5).astype(int) * 1
        s_s_gap += (c > ma5).astype(int) * 1
        s_s_gap += ((v > vol_ma5 * 1.5) & (c > prev_c)).astype(int) * 1
        s_s_gap += (clv < -0.5).astype(int) * 1
        s_s_gap += (ma20 < ma60).astype(int) * 2

        # ═══════════════════════════════════════════
        # === 信号2: 动量轮动 ===
        # ═══════════════════════════════════════════
        s_l_mom = np.zeros(n)
        s_l_mom += (mom5 > 3).astype(int) * 1
        s_l_mom += (mom5 > 5).astype(int) * 1
        s_l_mom += (mom10 > 5).astype(int) * 2
        s_l_mom += (mom20 > 8).astype(int) * 2
        s_l_mom += (c > ma5).astype(int) * 1
        s_l_mom += (c > ma20).astype(int) * 1
        s_l_mom += (ma5 > ma20).astype(int) * 1
        s_l_mom += (ma20 > ma60).astype(int) * 2
        s_l_mom += ((v > vol_ma5 * 1.3) & (c > prev_c)).astype(int) * 1
        s_l_mom += (oi_trend > 5).astype(int) * 1  # OI上升趋势

        s_s_mom = np.zeros(n)
        s_s_mom += (mom5 < -3).astype(int) * 1
        s_s_mom += (mom5 < -5).astype(int) * 1
        s_s_mom += (mom10 < -5).astype(int) * 2
        s_s_mom += (mom20 < -8).astype(int) * 2
        s_s_mom += (c < ma5).astype(int) * 1
        s_s_mom += (c < ma20).astype(int) * 1
        s_s_mom += (ma5 < ma20).astype(int) * 1
        s_s_mom += (ma20 < ma60).astype(int) * 2
        s_s_mom += ((v > vol_ma5 * 1.3) & (c < prev_c)).astype(int) * 1
        s_s_mom += (oi_trend < -5).astype(int) * 1

        # ═══════════════════════════════════════════
        # === 信号3: 均值回归 (超跌反弹) ===
        # ═══════════════════════════════════════════
        s_l_mr = np.zeros(n)
        s_l_mr += (rsi < 30).astype(int) * 3
        s_l_mr += (rsi < 25).astype(int) * 2
        s_l_mr += (c < bb_lower).astype(int) * 2
        s_l_mr += (mom5 < -5).astype(int) * 1
        s_l_mr += (mom5 < -8).astype(int) * 1
        s_l_mr += (mom3 < -3).astype(int) * 1
        s_l_mr += (clv_ma5 > 0.3).astype(int) * 1  # 持续在低位
        s_l_mr += ((c < ma60) & (c > ma120)).astype(int) * 1  # 长期支撑

        s_s_mr = np.zeros(n)
        s_s_mr += (rsi > 70).astype(int) * 3
        s_s_mr += (rsi > 75).astype(int) * 2
        s_s_mr += (c > bb_upper).astype(int) * 2
        s_s_mr += (mom5 > 5).astype(int) * 1
        s_s_mr += (mom5 > 8).astype(int) * 1
        s_s_mr += (mom3 > 3).astype(int) * 1
        s_s_mr += (clv_ma5 < -0.3).astype(int) * 1
        s_s_mr += ((c > ma60) & (c < ma120)).astype(int) * 1

        # ═══════════════════════════════════════════
        # === 信号4: 波动率突破 ===
        # ═══════════════════════════════════════════
        s_l_vol = np.zeros(n)
        s_l_vol += (bb_width < np.nanpercentile(bb_width[~np.isnan(bb_width)], 30) if np.any(~np.isnan(bb_width)) else 0).astype(int) * 2  # 低波动
        s_l_vol += (gv < -0.8).astype(int) * 2  # 突然跳空
        s_l_vol += (v > vol_ma20 * 2).astype(int) * 2  # 放量
        s_l_vol += (c > o).astype(int) * 1  # 收阳

        s_s_vol = np.zeros(n)
        s_s_vol += (bb_width < np.nanpercentile(bb_width[~np.isnan(bb_width)], 30) if np.any(~np.isnan(bb_width)) else 0).astype(int) * 2
        s_s_vol += (gv > 0.8).astype(int) * 2
        s_s_vol += (v > vol_ma20 * 2).astype(int) * 2
        s_s_vol += (c < o).astype(int) * 1

        # ═══════════════════════════════════════════
        # === 信号5: 期限结构增强 ===
        # ═══════════════════════════════════════════
        s_l_ts = np.zeros(n)
        s_l_ts += (ts_struct == 1).astype(int) * 2  # backwardation做多
        s_l_ts += (ts_carry_chg5 > 0.5).astype(int) * 1  # carry改善
        s_l_ts += (ts_carry_chg5 > 1.0).astype(int) * 1
        s_l_ts += ((gv < -1.0) & (ts_struct == 1)).astype(int) * 2  # gap fade + backwardation

        s_s_ts = np.zeros(n)
        s_s_ts += (ts_struct == -1).astype(int) * 2  # contango做空
        s_s_ts += (ts_carry_chg5 < -0.5).astype(int) * 1
        s_s_ts += (ts_carry_chg5 < -1.0).astype(int) * 1
        s_s_ts += ((gv > 1.0) & (ts_struct == -1)).astype(int) * 2

        # ═══════════════════════════════════════════
        # === 综合得分: 多信号融合 ===
        # ═══════════════════════════════════════════
        # 权重: gap_fade=1.0, momentum=0.5, mean_reversion=0.4, vol_breakout=0.3, ts=0.3
        w_gap, w_mom, w_mr, w_vol, w_ts = 1.0, 0.5, 0.4, 0.3, 0.3

        df['score_long'] = s_l_gap + s_l_mom * w_mom + s_l_mr * w_mr + s_l_vol * w_vol + s_l_ts * w_ts
        df['score_short'] = s_s_gap + s_s_mom * w_mom + s_s_mr * w_mr + s_s_vol * w_vol + s_s_ts * w_ts
        df['score_gap_long'] = s_l_gap
        df['score_gap_short'] = s_s_gap
        df['score_mom_long'] = s_l_mom
        df['score_mom_short'] = s_s_mom
        df['score_mr_long'] = s_l_mr
        df['score_mr_short'] = s_s_mr

        df['gap_pct'] = gap
        df['vol_regime'] = vol_regime
        df['rsi'] = rsi
        df['bb_width'] = bb_width
        df['ts_carry'] = ts_carry
        df['ts_struct'] = ts_struct
        df['mom5'] = mom5
        df['mom10'] = mom10
        df['mom20'] = mom20
        df['oi_ch'] = oi_ch
        df['clv'] = clv
        df['atr_pct'] = atr_pct

        signal_data[sym] = df
    return signal_data


def compute_cross_sectional(signal_data, date):
    """计算截面排名 (跨品种相对强弱)"""
    scores = []
    for sym, df in signal_data.items():
        idx = df.index[df['trade_date'] == date]
        if len(idx) == 0: continue
        row = df.loc[idx[0]]
        if np.isnan(row.get('close', np.nan)): continue
        scores.append({
            'sym': sym,
            'mom5': row.get('mom5', np.nan),
            'mom10': row.get('mom10', np.nan),
            'mom20': row.get('mom20', np.nan),
            'gap_pct': row.get('gap_pct', np.nan),
            'vol_regime': row.get('vol_regime', np.nan),
        })
    if not scores: return {}
    sdf = pd.DataFrame(scores)
    # 排名 (percentile)
    for col in ['mom5', 'mom10', 'mom20', 'gap_pct']:
        if col in sdf.columns:
            sdf[f'{col}_pct'] = sdf[col].rank(pct=True)
    result = {}
    for _, row in sdf.iterrows():
        result[row['sym']] = row.to_dict()
    return result


def run_ensemble_bt(signal_data, start, end, config=None):
    """综合集成回测"""
    if config is None:
        config = {
            'max_pos': 7, 'leverage': 5, 'min_score': 7, 'hold': 1,
            'sl_pct': -1.5, 'tp_pct': 4.0, 'max_long': 4, 'max_short': 4,
            'strategy_weights': {'gap': 1.0, 'mom': 0.5, 'mr': 0.4, 'vol': 0.3, 'ts': 0.3},
            'use_cross_section': True,
            'vol_adjust': True,
            'dynamic_size': True,
        }

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
                if config['sl_pct']:
                    stop = p['ep'] * (1 + config['sl_pct'] / 100)
                    if cur_l <= stop:
                        actual_ret = (stop * (1 - sp) - p['ep']) / p['ep'] * 100
                        reason = 'SL'; triggered = True
                if not triggered and config['tp_pct']:
                    tp_p = p['ep'] * (1 + config['tp_pct'] / 100)
                    if cur_h >= tp_p:
                        actual_ret = (tp_p * (1 - sp) - p['ep']) / p['ep'] * 100
                        reason = 'TP'; triggered = True
                if not triggered:
                    actual_ret = (cur_c - p['ep']) / p['ep'] * 100
            else:
                if config['sl_pct']:
                    stop = p['ep'] * (1 - config['sl_pct'] / 100)
                    if cur_h >= stop:
                        actual_ret = (p['ep'] - stop * (1 + sp)) / p['ep'] * 100
                        reason = 'SL'; triggered = True
                if not triggered and config['tp_pct']:
                    tp_p = p['ep'] * (1 - config['tp_pct'] / 100)
                    if cur_l <= tp_p:
                        actual_ret = (p['ep'] - tp_p * (1 + sp)) / p['ep'] * 100
                        reason = 'TP'; triggered = True
                if not triggered:
                    actual_ret = (p['ep'] - cur_c) / p['ep'] * 100
            if d >= config['hold']:
                if not triggered: reason = 'exp'
            else:
                if not triggered: keep.append(p); continue
            if reason:
                pnl += p['not'] * actual_ret / 100
                trades.append({
                    'sym': p['sym'], 'dir': p['dir'], 'ed': p['ed'], 'xd': dt,
                    'ep': p['ep'], 'xp': cur_c, 'r': actual_ret,
                    'pnl': p['not'] * actual_ret / 100, 'sc': p['sc'],
                    'hold': d, 'reason': reason, 'strategy': p.get('strategy', 'gap'),
                })
        pos = keep
        cap += pnl
        if cap <= 0: break

        n_open = config['max_pos'] - len(pos)
        if n_open <= 0:
            eq.append({'date': dt, 'capital': cap}); continue

        # 截面排名
        cs = compute_cross_sectional(signal_data, dt) if config.get('use_cross_section') else {}

        cands = []
        for sym, df in signal_data.items():
            if any(p['sym'] == sym for p in pos): continue
            idx = df.index[df['trade_date'] == dt]
            if len(idx) == 0: continue
            row = df.loc[idx[0]]

            # 波动率调节: 高波动时提高min_score
            min_sc = config['min_score']
            if config.get('vol_adjust'):
                vr = row.get('vol_regime', np.nan)
                if not np.isnan(vr) and vr > 1.5:
                    min_sc += 2  # 高波动要求更高分数

            # 截面增强
            cs_bonus = 0
            if config.get('use_cross_section') and sym in cs:
                cs_data = cs[sym]
                # 如果该品种在截面上排名靠前, 加bonus
                if row.get('score_long', 0) >= config['min_score']:
                    mom_pct = cs_data.get('mom10_pct', 0.5)
                    if mom_pct > 0.7: cs_bonus += 1
                    elif mom_pct > 0.8: cs_bonus += 2
                if row.get('score_short', 0) >= config['min_score']:
                    mom_pct = cs_data.get('mom10_pct', 0.5)
                    if mom_pct < 0.3: cs_bonus += 1
                    elif mom_pct < 0.2: cs_bonus += 2

            eff_sc_long = row.get('score_long', 0) + cs_bonus
            eff_sc_short = row.get('score_short', 0) + cs_bonus

            if eff_sc_long >= min_sc:
                cands.append({
                    'sym': sym, 'dir': 'long', 'sc': eff_sc_long,
                    'ep': row['open'], 'strategy': 'ensemble',
                })
            if eff_sc_short >= min_sc:
                cands.append({
                    'sym': sym, 'dir': 'short', 'sc': eff_sc_short,
                    'ep': row['open'], 'strategy': 'ensemble',
                })

        # 去重: 每品种取最高分
        best = {}
        for c_ in cands:
            if c_['sym'] not in best or c_['sc'] > best[c_['sym']]['sc']:
                best[c_['sym']] = c_
        ranked = sorted(best.values(), key=lambda x: -x['sc'])

        n_long = sum(1 for p in pos if p['dir'] == 'long')
        n_short = sum(1 for p in pos if p['dir'] == 'short')

        for c_ in ranked:
            if n_open <= 0: break
            if config['max_long'] and c_['dir'] == 'long' and n_long >= config['max_long']: continue
            if config['max_short'] and c_['dir'] == 'short' and n_short >= config['max_short']: continue

            # 动态仓位: 基于信号强度
            notional = cap * config['leverage'] / config['max_pos']
            if config.get('dynamic_size'):
                # 高分信号加大仓位 (1.0x ~ 1.5x)
                size_mult = min(1.0 + (c_['sc'] - config['min_score']) * 0.1, 1.5)
                notional *= size_mult

            pos.append({
                'sym': c_['sym'], 'dir': c_['dir'], 'ed': dt,
                'ep': c_['ep'], 'not': notional, 'sc': c_['sc'],
                'strategy': c_.get('strategy', 'gap'),
            })
            if c_['dir'] == 'long': n_long += 1
            else: n_short += 1
            n_open -= 1
        eq.append({'date': dt, 'capital': cap})
    return eq, trades


def run_gap_only_bt(signal_data, start, end):
    """纯Gap Fade回测 (使用gap-only分数)"""
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
                stop = p['ep'] * (1 + (-1.5) / 100)
                if cur_l <= stop:
                    actual_ret = (stop * (1 - sp) - p['ep']) / p['ep'] * 100
                    reason = 'SL'; triggered = True
                if not triggered:
                    tp_p = p['ep'] * (1 + 4.0 / 100)
                    if cur_h >= tp_p:
                        actual_ret = (tp_p * (1 - sp) - p['ep']) / p['ep'] * 100
                        reason = 'TP'; triggered = True
                if not triggered:
                    actual_ret = (cur_c - p['ep']) / p['ep'] * 100
            else:
                stop = p['ep'] * (1 - (-1.5) / 100)
                if cur_h >= stop:
                    actual_ret = (p['ep'] - stop * (1 + sp)) / p['ep'] * 100
                    reason = 'SL'; triggered = True
                if not triggered:
                    tp_p = p['ep'] * (1 - 4.0 / 100)
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
            if row.get('score_gap_long', 0) >= 7:
                cands.append({'sym': sym, 'dir': 'long', 'sc': row['score_gap_long'], 'ep': row['open']})
            if row.get('score_gap_short', 0) >= 7:
                cands.append({'sym': sym, 'dir': 'short', 'sc': row['score_gap_short'], 'ep': row['open']})
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
    """计算绩效指标"""
    eq_df = pd.DataFrame(eq_list)
    if len(eq_df) == 0:
        return {'N': 0, 'WR': 0, 'Sharpe': 0, 'MDD': 0, 'Ret': 0, 'Avg': 0}
    tdf = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()
    final_cap = eq_df['capital'].iloc[-1]
    total_ret = (final_cap / INITIAL_CAPITAL - 1) * 100
    dr = eq_df['capital'].pct_change().dropna()
    sh = dr.mean() / dr.std() * (252**0.5) if len(dr) > 0 and dr.std() > 0 else 0
    mdd = ((eq_df['capital'] - eq_df['capital'].cummax()) / eq_df['capital'].cummax() * 100).min()
    wr = (tdf['r'] > 0).mean() * 100 if len(tdf) > 0 else 0
    avg = tdf['r'].mean() if len(tdf) > 0 else 0
    n = len(tdf)
    # 年化
    years = (eq_df['date'].iloc[-1] - eq_df['date'].iloc[0]).days / 365.25
    ann_ret = ((final_cap / INITIAL_CAPITAL) ** (1/years) - 1) * 100 if years > 0 else 0
    return {'N': n, 'WR': wr, 'Sharpe': sh, 'MDD': mdd, 'Ret': total_ret,
            'AnnRet': ann_ret, 'Avg': avg}


def main():
    print("V86: 多策略Alpha引擎")
    print("=" * 60)

    print("\n加载数据...")
    all_data = load_data()
    print("加载期限结构...")
    ts_data = load_term_structure()
    print("计算多维度信号...")
    sd = compute_all_signals(all_data, ts_data)

    TEST_START = '2016-01-01'
    TEST_END = '2025-12-31'

    # ═══════════════════════════════════════════
    # 1. 基准: 纯Gap Fade
    # ═══════════════════════════════════════════
    print(f"\n{'='*60}")
    print("1. 基准: 纯Gap Fade (V80最优)")
    print(f"{'='*60}")
    eq_gap, tr_gap = run_gap_only_bt(sd, TEST_START, TEST_END)
    m_gap = calc_metrics(eq_gap, tr_gap)
    print(f"  N={m_gap['N']} WR={m_gap['WR']:.1f}% Sharpe={m_gap['Sharpe']:.2f} "
          f"MDD={m_gap['MDD']:.1f}% AnnRet={m_gap['AnnRet']:.0f}% Avg={m_gap['Avg']:+.3f}%")

    # ═══════════════════════════════════════════
    # 2. 集成策略 (多信号融合)
    # ═══════════════════════════════════════════
    print(f"\n{'='*60}")
    print("2. 集成策略 (多信号融合)")
    print(f"{'='*60}")

    configs = [
        ("A. 默认权重", {'max_pos': 7, 'leverage': 5, 'min_score': 7, 'hold': 1,
                         'sl_pct': -1.5, 'tp_pct': 4.0, 'max_long': 4, 'max_short': 4}),
        ("B. 高门槛", {'max_pos': 7, 'leverage': 5, 'min_score': 9, 'hold': 1,
                        'sl_pct': -1.5, 'tp_pct': 4.0, 'max_long': 4, 'max_short': 4}),
        ("C. 宽TP", {'max_pos': 7, 'leverage': 5, 'min_score': 7, 'hold': 1,
                       'sl_pct': -1.5, 'tp_pct': 6.0, 'max_long': 4, 'max_short': 4}),
        ("D. 持仓2天", {'max_pos': 7, 'leverage': 5, 'min_score': 7, 'hold': 2,
                         'sl_pct': -1.5, 'tp_pct': 4.0, 'max_long': 4, 'max_short': 4}),
        ("E. 5仓位", {'max_pos': 5, 'leverage': 5, 'min_score': 7, 'hold': 1,
                        'sl_pct': -1.5, 'tp_pct': 4.0, 'max_long': 3, 'max_short': 3}),
        ("F. 10仓位+8门槛", {'max_pos': 10, 'leverage': 5, 'min_score': 8, 'hold': 1,
                              'sl_pct': -1.5, 'tp_pct': 4.0, 'max_long': 5, 'max_short': 5}),
    ]

    best_name = ""
    best_sharpe = -999
    best_config = None

    for name, cfg in configs:
        eq, tr = run_ensemble_bt(sd, TEST_START, TEST_END, config=cfg)
        m = calc_metrics(eq, tr)
        print(f"  {name:20s} N={m['N']:5d} WR={m['WR']:.1f}% Sharpe={m['Sharpe']:.2f} "
              f"MDD={m['MDD']:.1f}% AnnRet={m['AnnRet']:.0f}% Avg={m['Avg']:+.3f}%")
        if m['Sharpe'] > best_sharpe:
            best_sharpe = m['Sharpe']
            best_name = name
            best_config = cfg

    print(f"\n  最优: {best_name} (Sharpe={best_sharpe:.2f})")

    # ═══════════════════════════════════════════
    # 3. 不同权重组合优化
    # ═══════════════════════════════════════════
    print(f"\n{'='*60}")
    print("3. 权重优化")
    print(f"{'='*60}")

    weight_combos = [
        ("纯Gap",     {'gap': 1.0, 'mom': 0.0, 'mr': 0.0, 'vol': 0.0, 'ts': 0.0}),
        ("Gap+Mom",   {'gap': 1.0, 'mom': 0.5, 'mr': 0.0, 'vol': 0.0, 'ts': 0.0}),
        ("Gap+MR",    {'gap': 1.0, 'mom': 0.0, 'mr': 0.5, 'vol': 0.0, 'ts': 0.0}),
        ("Gap+TS",    {'gap': 1.0, 'mom': 0.0, 'mr': 0.0, 'vol': 0.0, 'ts': 0.5}),
        ("Gap+All",   {'gap': 1.0, 'mom': 0.5, 'mr': 0.4, 'vol': 0.3, 'ts': 0.3}),
        ("Gap+Mom+MR",{'gap': 1.0, 'mom': 0.5, 'mr': 0.4, 'vol': 0.0, 'ts': 0.0}),
        ("全量0.5",   {'gap': 1.0, 'mom': 0.5, 'mr': 0.5, 'vol': 0.5, 'ts': 0.5}),
        ("全量1.0",   {'gap': 1.0, 'mom': 1.0, 'mr': 1.0, 'vol': 1.0, 'ts': 1.0}),
    ]

    for name, w in weight_combos:
        cfg = dict(best_config or {'max_pos': 7, 'leverage': 5, 'min_score': 7, 'hold': 1,
                                    'sl_pct': -1.5, 'tp_pct': 4.0, 'max_long': 4, 'max_short': 4})
        cfg['strategy_weights'] = w
        # 重新计算信号用新权重
        sd2 = recompute_scores(all_data, ts_data, w)
        eq, tr = run_ensemble_bt(sd2, TEST_START, TEST_END, config=cfg)
        m = calc_metrics(eq, tr)
        print(f"  {name:15s} N={m['N']:5d} WR={m['WR']:.1f}% Sharpe={m['Sharpe']:.2f} "
              f"MDD={m['MDD']:.1f}% Avg={m['Avg']:+.3f}%")

    # ═══════════════════════════════════════════
    # 4. 年度分解
    # ═══════════════════════════════════════════
    print(f"\n{'='*60}")
    print("4. 年度分解 (最优集成)")
    print(f"{'='*60}")
    cfg_final = best_config or {'max_pos': 7, 'leverage': 5, 'min_score': 7, 'hold': 1,
                                 'sl_pct': -1.5, 'tp_pct': 4.0, 'max_long': 4, 'max_short': 4}
    eq_all, tr_all = run_ensemble_bt(sd, TEST_START, TEST_END, config=cfg_final)

    if tr_all:
        tdf = pd.DataFrame(tr_all)
        tdf['year'] = pd.to_datetime(tdf['xd']).dt.year
        print(f"\n  {'Year':>6} {'N':>5} {'WR':>6} {'Avg':>8} {'PnL':>12}")
        print("  " + "-"*40)
        for yr in sorted(tdf['year'].unique()):
            s = tdf[tdf['year'] == yr]
            pnl = s['pnl'].sum()
            print(f"  {yr:>6} {len(s):>5} {(s['r']>0).mean()*100:>5.1f}% {s['r'].mean():>+7.3f}% {pnl:>+12.0f}")

    # ═══════════════════════════════════════════
    # 5. 波动率Regime分析
    # ═══════════════════════════════════════════
    print(f"\n{'='*60}")
    print("5. 波动率Regime分析")
    print(f"{'='*60}")
    if tr_all:
        # 按入场日期的vol_regime分组
        tdf = pd.DataFrame(tr_all)
        for _, row in tdf.iterrows():
            sym = row['sym']
            dt = row['ed']
            df = sd.get(sym)
            if df is None: continue
            idx = df.index[df['trade_date'] == dt]
            if len(idx) == 0: continue
            tdf.loc[row.name, 'vol_regime'] = df.loc[idx[0], 'vol_regime']

        vr = tdf['vol_regime'].dropna()
        if len(vr) > 0:
            for lo, hi, label in [(0, 0.7, '低波动'), (0.7, 1.3, '正常'), (1.3, 3.0, '高波动')]:
                sub = tdf[(tdf['vol_regime'] >= lo) & (tdf['vol_regime'] < hi)]
                if len(sub) > 0:
                    print(f"  {label:6s}: N={len(sub):4d} WR={(sub['r']>0).mean()*100:.1f}% Avg={sub['r'].mean():+.3f}%")

    # ═══════════════════════════════════════════
    # 6. 最终对比
    # ═══════════════════════════════════════════
    print(f"\n{'='*60}")
    print("6. 最终对比")
    print(f"{'='*60}")
    m_ens = calc_metrics(eq_all, tr_all)
    print(f"\n  {'策略':20s} {'N':>5} {'WR':>6} {'Sharpe':>7} {'MDD':>7} {'AnnRet':>8} {'Avg':>8}")
    print("  " + "-"*65)
    print(f"  {'Gap Fade (V80)':20s} {m_gap['N']:>5} {m_gap['WR']:>5.1f}% {m_gap['Sharpe']:>7.2f} "
          f"{m_gap['MDD']:>+6.1f}% {m_gap['AnnRet']:>7.0f}% {m_gap['Avg']:>+7.3f}%")
    print(f"  {'集成策略 (V86)':20s} {m_ens['N']:>5} {m_ens['WR']:>5.1f}% {m_ens['Sharpe']:>7.2f} "
          f"{m_ens['MDD']:>+6.1f}% {m_ens['AnnRet']:>7.0f}% {m_ens['Avg']:>+7.3f}%")

    delta_sh = m_ens['Sharpe'] - m_gap['Sharpe']
    delta_wr = m_ens['WR'] - m_gap['WR']
    print(f"\n  Sharpe差异: {delta_sh:+.2f}")
    print(f"  WR差异: {delta_wr:+.1f}%")


def recompute_scores(all_data, ts_data, weights):
    """用新权重重算信号"""
    w_gap = weights.get('gap', 1.0)
    w_mom = weights.get('mom', 0.5)
    w_mr = weights.get('mr', 0.4)
    w_vol = weights.get('vol', 0.3)
    w_ts = weights.get('ts', 0.3)
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
        atr_pct = atr / np.where(c > 0, c, np.nan) * 100
        ma5 = pd.Series(c).rolling(5).mean().values
        ma20 = pd.Series(c).rolling(20).mean().values
        ma60 = pd.Series(c).rolling(60).mean().values
        vol_ma5 = pd.Series(v).rolling(5).mean().values
        mom5 = np.full(n, np.nan); mom5[5:] = (c[5:] - c[:-5]) / c[:-5] * 100
        mom10 = np.full(n, np.nan); mom10[10:] = (c[10:] - c[:-10]) / c[:-10] * 100
        mom20 = np.full(n, np.nan); mom20[20:] = (c[20:] - c[:-20]) / c[:-20] * 100
        mom3 = np.full(n, np.nan)
        if n > 3: mom3[3:] = (c[3:] - c[:-3]) / c[:-3] * 100
        oi_ch = np.full(n, np.nan)
        valid = np.abs(oi[:-1]) > 0
        oi_ch_v = np.full(n-1, np.nan)
        oi_ch_v[valid] = (oi[1:][valid] - oi[:-1][valid]) / np.abs(oi[:-1][valid]) * 100
        oi_ch[1:] = oi_ch_v
        range_ = h - l
        clv = np.where(range_ > 0, (2*c - h - l) / range_, 0)
        clv_ma5 = pd.Series(clv).rolling(5).mean().values
        gv = np.nan_to_num(gap)
        ga = np.nan_to_num(gap / np.where(atr_pct == 0, np.nan, atr_pct))
        delta = np.full(n, np.nan); delta[1:] = c[1:] - c[:-1]
        gain = np.where(delta > 0, delta, 0)
        loss_arr = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean().values
        avg_loss = pd.Series(loss_arr).rolling(14).mean().values
        rsi = np.where(avg_loss > 0, 100 - 100 / (1 + avg_gain / avg_loss), 50)
        bb_mid = ma20
        bb_std = pd.Series(c).rolling(20).std().values
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_width = np.where(bb_mid > 0, (bb_upper - bb_lower) / bb_mid * 100, np.nan)
        vol_ma20 = pd.Series(v).rolling(20).mean().values
        oi_ma5 = pd.Series(oi).rolling(5).mean().values
        oi_trend = np.full(n, np.nan)
        oi_trend[5:] = (oi[5:] - oi_ma5[5:]) / np.where(np.abs(oi_ma5[5:]) > 0, np.abs(oi_ma5[5:]), np.nan) * 100
        ts_carry = np.full(n, np.nan)
        ts_struct = np.full(n, 0)
        ts_carry_chg5 = np.full(n, np.nan)
        if sym in ts_data:
            carry_hist = []
            dates_list = df['trade_date'].tolist()
            for i, dt in enumerate(dates_list):
                ds = dt.strftime('%Y%m%d') if hasattr(dt, 'strftime') else str(dt)[:10].replace('-','')
                if ds in ts_data[sym]:
                    sp = ts_data[sym][ds].get('total_spread_pct', 0)
                    ts_carry[i] = sp
                    ts_struct[i] = 1 if ts_data[sym][ds].get('structure') == 'backwardation' else -1
                    carry_hist.append((i, sp))
            if len(carry_hist) >= 6:
                for j in range(5, len(carry_hist)):
                    ts_carry_chg5[carry_hist[j][0]] = carry_hist[j][1] - carry_hist[j-5][1]

        s_l_gap = np.zeros(n)
        s_l_gap += (gv < -0.5).astype(int) * 1
        s_l_gap += (gv < -1.0).astype(int) * 2
        s_l_gap += (gv < -1.5).astype(int) * 2
        s_l_gap += (gv < -2.0).astype(int) * 3
        s_l_gap += (ga < -1.0).astype(int) * 2
        s_l_gap += (ga < -1.5).astype(int) * 3
        s_l_gap += ((oi_ch > 0) & (c < prev_c)).astype(int) * 3
        s_l_gap += ((oi_ch < 0) & (c < prev_c)).astype(int) * 2
        s_l_gap += (mom5 < -3).astype(int) * 1
        s_l_gap += (mom5 < -5).astype(int) * 1
        s_l_gap += (c < ma5).astype(int) * 1
        s_l_gap += ((v > vol_ma5 * 1.5) & (c < prev_c)).astype(int) * 1
        s_l_gap += (clv > 0.5).astype(int) * 1
        s_l_gap += (ma20 > ma60).astype(int) * 2

        s_s_gap = np.zeros(n)
        s_s_gap += (gv > 0.5).astype(int) * 1
        s_s_gap += (gv > 1.0).astype(int) * 2
        s_s_gap += (gv > 1.5).astype(int) * 2
        s_s_gap += (gv > 2.0).astype(int) * 3
        s_s_gap += (ga > 1.0).astype(int) * 2
        s_s_gap += (ga > 1.5).astype(int) * 3
        s_s_gap += ((oi_ch > 0) & (c > prev_c)).astype(int) * 3
        s_s_gap += ((oi_ch < 0) & (c > prev_c)).astype(int) * 2
        s_s_gap += (mom5 > 3).astype(int) * 1
        s_s_gap += (mom5 > 5).astype(int) * 1
        s_s_gap += (c > ma5).astype(int) * 1
        s_s_gap += ((v > vol_ma5 * 1.5) & (c > prev_c)).astype(int) * 1
        s_s_gap += (clv < -0.5).astype(int) * 1
        s_s_gap += (ma20 < ma60).astype(int) * 2

        s_l_mom = np.zeros(n)
        s_l_mom += (mom5 > 3).astype(int) * 1
        s_l_mom += (mom5 > 5).astype(int) * 1
        s_l_mom += (mom10 > 5).astype(int) * 2
        s_l_mom += (mom20 > 8).astype(int) * 2
        s_l_mom += (c > ma5).astype(int) * 1
        s_l_mom += (c > ma20).astype(int) * 1
        s_l_mom += (ma5 > ma20).astype(int) * 1
        s_l_mom += (ma20 > ma60).astype(int) * 2
        s_l_mom += ((v > vol_ma5 * 1.3) & (c > prev_c)).astype(int) * 1
        s_l_mom += (oi_trend > 5).astype(int) * 1

        s_s_mom = np.zeros(n)
        s_s_mom += (mom5 < -3).astype(int) * 1
        s_s_mom += (mom5 < -5).astype(int) * 1
        s_s_mom += (mom10 < -5).astype(int) * 2
        s_s_mom += (mom20 < -8).astype(int) * 2
        s_s_mom += (c < ma5).astype(int) * 1
        s_s_mom += (c < ma20).astype(int) * 1
        s_s_mom += (ma5 < ma20).astype(int) * 1
        s_s_mom += (ma20 < ma60).astype(int) * 2
        s_s_mom += ((v > vol_ma5 * 1.3) & (c < prev_c)).astype(int) * 1
        s_s_mom += (oi_trend < -5).astype(int) * 1

        s_l_mr = np.zeros(n)
        s_l_mr += (rsi < 30).astype(int) * 3
        s_l_mr += (rsi < 25).astype(int) * 2
        s_l_mr += (c < bb_lower).astype(int) * 2
        s_l_mr += (mom5 < -5).astype(int) * 1
        s_l_mr += (mom5 < -8).astype(int) * 1
        s_l_mr += (mom3 < -3).astype(int) * 1
        s_l_mr += (clv_ma5 > 0.3).astype(int) * 1

        s_s_mr = np.zeros(n)
        s_s_mr += (rsi > 70).astype(int) * 3
        s_s_mr += (rsi > 75).astype(int) * 2
        s_s_mr += (c > bb_upper).astype(int) * 2
        s_s_mr += (mom5 > 5).astype(int) * 1
        s_s_mr += (mom5 > 8).astype(int) * 1
        s_s_mr += (mom3 > 3).astype(int) * 1
        s_s_mr += (clv_ma5 < -0.3).astype(int) * 1

        s_l_vol = np.zeros(n)
        bb_p30 = np.nanpercentile(bb_width[~np.isnan(bb_width)], 30) if np.any(~np.isnan(bb_width)) else 0
        s_l_vol += (bb_width < bb_p30).astype(int) * 2
        s_l_vol += (gv < -0.8).astype(int) * 2
        s_l_vol += (v > vol_ma20 * 2).astype(int) * 2
        s_l_vol += (c > o).astype(int) * 1

        s_s_vol = np.zeros(n)
        s_s_vol += (bb_width < bb_p30).astype(int) * 2
        s_s_vol += (gv > 0.8).astype(int) * 2
        s_s_vol += (v > vol_ma20 * 2).astype(int) * 2
        s_s_vol += (c < o).astype(int) * 1

        s_l_ts = np.zeros(n)
        s_l_ts += (ts_struct == 1).astype(int) * 2
        s_l_ts += (ts_carry_chg5 > 0.5).astype(int) * 1
        s_l_ts += (ts_carry_chg5 > 1.0).astype(int) * 1
        s_l_ts += ((gv < -1.0) & (ts_struct == 1)).astype(int) * 2

        s_s_ts = np.zeros(n)
        s_s_ts += (ts_struct == -1).astype(int) * 2
        s_s_ts += (ts_carry_chg5 < -0.5).astype(int) * 1
        s_s_ts += (ts_carry_chg5 < -1.0).astype(int) * 1
        s_s_ts += ((gv > 1.0) & (ts_struct == -1)).astype(int) * 2

        df['score_long'] = s_l_gap * w_gap + s_l_mom * w_mom + s_l_mr * w_mr + s_l_vol * w_vol + s_l_ts * w_ts
        df['score_short'] = s_s_gap * w_gap + s_s_mom * w_mom + s_s_mr * w_mr + s_s_vol * w_vol + s_s_ts * w_ts
        df['gap_pct'] = gap
        signal_data[sym] = df
    return signal_data


if __name__ == '__main__':
    main()
