#!/usr/bin/env python3
"""
策略 v39 — 超精选 + 自适应Kelly + 趋势回调入场

核心创新:
1. 截面排名: 每天在72个品种中只选信号最强的3个 (超精选提高WR)
2. 趋势回调入场: 趋势方向明确后等待RSI回调再入场 (提高WR)
3. 自适应Kelly: 根据滚动胜率和赔率动态调整仓位
4. 波动率制度: 低HV时加仓, 高HV时减仓
5. 期权分析过滤: HV百分位 + 偏斜度 + 期限结构
6. 买入OTM期权增强: 仅在高确信度时买入期权提供凸性

目标: 年化>=600%, 胜率>=50%, 持仓<=3, 回测>=8年
"""

import os, sys, time, json, numpy as np, pandas as pd
from collections import defaultdict
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from contract_specs import get_spec

COMM = 0.00015
OPT_COMM = 0.0003
R_RATE = 0.02
INIT_CAPITAL = 500000


def bs_price(S, K, T, r, sigma, opt='call'):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0) if opt == 'call' else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def load_data(data_dir):
    data = {}
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith('.csv'):
            continue
        symbol = f.replace('.csv', '')
        df = pd.read_csv(os.path.join(data_dir, f))
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
        if len(df) < 300:
            continue

        df['return'] = df['close'].pct_change()
        df['log_ret'] = np.log(df['close'] / df['close'].shift())

        # === 趋势指标 ===
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['ma120'] = df['close'].rolling(120).mean()

        # 多时间框架趋势
        df['trend_short'] = np.where(df['ma5'] > df['ma20'], 1, -1)
        df['trend_mid'] = np.where(df['ma20'] > df['ma60'], 1, -1)
        df['trend_long'] = np.where(df['ma60'] > df['ma120'], 1, -1)

        # 趋势强度 (ADX近似)
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        atr14 = df['high'].rolling(14).max() - df['low'].rolling(14).min()
        plus_di = 100 * plus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
        minus_di = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        df['adx'] = dx.rolling(14).mean()

        # === 动量指标 ===
        for lag in [3, 5, 10, 20]:
            df[f'mom_{lag}'] = df['close'].pct_change(lag)

        # 标准化动量 (除以波动率)
        std20 = df['return'].rolling(20).std().replace(0, np.nan)
        df['norm_mom5'] = df['mom_5'] / (std20 * np.sqrt(5))
        df['norm_mom10'] = df['mom_10'] / (std20 * np.sqrt(10))

        # === EWMAC信号 ===
        for span in [4, 8, 16, 32]:
            ema_f = df['close'].ewm(span=span).mean()
            ema_s = df['close'].ewm(span=span*4).mean()
            df[f'ewmac_{span}'] = (ema_f - ema_s) / df['close'].rolling(20).std().replace(0, np.nan)

        # === 突破信号 ===
        for window in [10, 20, 40]:
            hh = df['close'].rolling(window).max()
            ll = df['close'].rolling(window).min()
            df[f'breakout_{window}'] = (df['close'] - 0.5 * (hh + ll)) / (hh - ll + 0.001) * 2

        # === RSI ===
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss_s = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss_s))

        # === 波动率 ===
        df['hv_5'] = df['return'].rolling(5).std() * np.sqrt(252)
        df['hv_10'] = df['return'].rolling(10).std() * np.sqrt(252)
        df['hv_20'] = df['return'].rolling(20).std() * np.sqrt(252)
        df['hv_60'] = df['return'].rolling(60).std() * np.sqrt(252)
        df['hv_120'] = df['return'].rolling(120).std() * np.sqrt(252)

        # HV百分位
        df['hv_pct'] = df['hv_20'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 10 else 0.5, raw=False
        )

        # HV期限结构斜率
        df['hv_slope'] = df['hv_5'] / df['hv_60'].replace(0, np.nan)

        # 偏斜度
        df['ret_skew_20'] = df['return'].rolling(20).skew()

        # === ATR ===
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close']

        # === OI变化 ===
        if 'oi' in df.columns and df['oi'].notna().sum() > 100:
            df['oi_chg'] = df['oi'].pct_change(5)
            df['oi_confirm'] = np.sign(df['oi_chg']) == np.sign(df['mom_5'])
        else:
            df['oi_confirm'] = False

        # === 成交量 ===
        df['vol_ma20'] = df['vol'].rolling(20).mean()
        df['vol_ratio'] = df['vol'] / df['vol_ma20'].replace(0, np.nan)

        df = df.dropna(subset=['ma20', 'ma60', 'hv_20', 'mom_10', 'rsi', 'hv_pct', 'adx'])
        if len(df) > 100:
            df.attrs['spec'] = get_spec(symbol)
            data[symbol] = df
    return data


def build_date_map(data, start_date, end_date):
    date_map = defaultdict(dict)
    specs = {}
    for symbol, df in data.items():
        specs[symbol] = df.attrs['spec']
        mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
        for _, row in df[mask].iterrows():
            date_map[row['trade_date']][symbol] = row
    return dict(date_map), specs


def compute_composite_score(row, mode='trend_pullback'):
    """计算综合信号评分 (多个维度)"""
    score = 0.0
    weight = 0.0

    if mode == 'trend_pullback':
        # 趋势方向 (权重最大)
        trend_mid = row.get('trend_mid', 0)
        trend_long = row.get('trend_long', 0)
        if trend_mid == trend_long and trend_mid != 0:
            score += trend_mid * 3.0
            weight += 3.0

        # EWMAC确认
        ew8 = row.get('ewmac_8', 0)
        ew16 = row.get('ewmac_16', 0)
        if not pd.isna(ew8):
            score += np.sign(ew8) * min(abs(ew8), 2) * 1.5
            weight += 1.5
        if not pd.isna(ew16):
            score += np.sign(ew16) * min(abs(ew16), 2) * 1.5
            weight += 1.5

        # 动量 (确认方向)
        mom10 = row.get('mom_10', 0)
        if not pd.isna(mom10):
            score += np.sign(mom10) * min(abs(mom10) * 10, 2) * 1.0
            weight += 1.0

        # RSI回调 (核心: 等待回调入场)
        rsi = row.get('rsi', 50)
        trend_dir = 1 if score > 0 else -1
        if trend_dir == 1 and 35 <= rsi <= 55:
            score += 2.0  # 多头回调加分
            weight += 2.0
        elif trend_dir == -1 and 45 <= rsi <= 65:
            score -= 2.0  # 空头回调加分
            weight += 2.0
        elif (trend_dir == 1 and rsi > 70) or (trend_dir == -1 and rsi < 30):
            score *= 0.3  # 超买超卖减分

        # ADX确认 (趋势强度)
        adx = row.get('adx', 20)
        if adx > 25:
            score *= min(adx / 25, 1.5)
        elif adx < 15:
            score *= 0.3

        # 突破确认
        bo20 = row.get('breakout_20', 0)
        if not pd.isna(bo20):
            if np.sign(bo20) == np.sign(score):
                score += bo20 * 1.0
                weight += 1.0

    elif mode == 'cross_mom':
        # 纯截面动量 (rank所有品种)
        nm10 = row.get('norm_mom10', 0)
        if not pd.isna(nm10):
            score += np.sign(nm10) * abs(nm10) * 3.0
            weight += 3.0
        ew8 = row.get('ewmac_8', 0)
        if not pd.isna(ew8):
            score += np.sign(ew8) * min(abs(ew8), 2)
            weight += 1.0
        bo20 = row.get('breakout_20', 0)
        if not pd.isna(bo20):
            score += bo20
            weight += 1.0

    elif mode == 'ewmac_breakout':
        ew8 = row.get('ewmac_8', 0)
        ew16 = row.get('ewmac_16', 0)
        bo20 = row.get('breakout_20', 0)
        bo40 = row.get('breakout_40', 0)
        if not pd.isna(ew8):
            score += np.sign(ew8) * min(abs(ew8), 2)
            weight += 1.0
        if not pd.isna(ew16):
            score += np.sign(ew16) * min(abs(ew16), 2)
            weight += 1.0
        if not pd.isna(bo20):
            score += bo20
            weight += 1.0
        if not pd.isna(bo40):
            score += bo40
            weight += 1.0

    elif mode == 'mean_revert':
        # 均值回归: 极端下跌后买入
        mom3 = row.get('mom_3', 0)
        mom5 = row.get('mom_5', 0)
        rsi = row.get('rsi', 50)
        # 极端超卖 → 买入
        if rsi < 25:
            score += 3.0
            weight += 3.0
        elif rsi < 30:
            score += 2.0
            weight += 2.0
        # 极端超买 → 卖出
        elif rsi > 75:
            score -= 3.0
            weight += 3.0
        elif rsi > 70:
            score -= 2.0
            weight += 2.0
        # 短期反转
        if not pd.isna(mom3):
            if mom3 < -0.05:
                score += 1.5  # 急跌反弹
                weight += 1.5
            elif mom3 > 0.05:
                score -= 1.5
                weight += 1.5

    elif mode == 'full_combo':
        # 全信号组合
        trend_mid = row.get('trend_mid', 0)
        ew8 = row.get('ewmac_8', 0)
        ew16 = row.get('ewmac_16', 0)
        bo20 = row.get('breakout_20', 0)
        nm10 = row.get('norm_mom10', 0)
        rsi = row.get('rsi', 50)
        adx = row.get('adx', 20)
        skew = row.get('ret_skew_20', 0)

        # 趋势 (权重最大)
        if trend_mid != 0:
            score += trend_mid * 2.0
            weight += 2.0

        # EWMAC
        if not pd.isna(ew8):
            score += np.sign(ew8) * min(abs(ew8), 2)
            weight += 1.0
        if not pd.isna(ew16):
            score += np.sign(ew16) * min(abs(ew16), 2)
            weight += 1.0

        # 突破
        if not pd.isna(bo20):
            score += bo20
            weight += 1.0

        # 标准化动量
        if not pd.isna(nm10):
            score += np.sign(nm10) * min(abs(nm10), 2)
            weight += 1.0

        # RSI回调加分
        trend_dir = 1 if score > 0 else -1
        if trend_dir == 1 and 35 <= rsi <= 55:
            score += 1.5
            weight += 1.5
        elif trend_dir == -1 and 45 <= rsi <= 65:
            score -= 1.5
            weight += 1.5

        # ADX
        if adx > 25:
            score *= 1.2

        # 偏斜度 (负偏斜 → 反转)
        if not pd.isna(skew):
            score += -np.sign(skew) * min(abs(skew), 0.5)
            weight += 0.5

    return score / max(weight, 1) if weight > 0 else 0


def fast_backtest(date_map, specs, dates, params):
    max_pos = params.get('max_pos', 3)
    hold_days = params.get('hold_days', 5)
    margin_usage = params.get('margin_usage', 0.90)
    min_score = params.get('min_score', 0.3)
    mode = params.get('mode', 'trend_pullback')

    # 入场过滤
    use_adx = params.get('use_adx', True)
    min_adx = params.get('min_adx', 20)
    use_hv_pct = params.get('use_hv_pct', True)
    hv_pct_hi = params.get('hv_pct_hi', 0.85)
    use_rsi = params.get('use_rsi', True)
    rsi_upper = params.get('rsi_upper', 75)
    rsi_lower = params.get('rsi_lower', 25)
    use_oi = params.get('use_oi', False)

    # 退出
    stop_loss_pct = params.get('stop_loss_pct', 0.0)  # 百分比止损
    take_profit_pct = params.get('take_profit_pct', 0.0)  # 百分比止盈
    trailing_stop_pct = params.get('trailing_stop_pct', 0.0)  # 追踪止损百分比

    # Kelly
    use_kelly = params.get('use_kelly', False)
    kelly_frac = params.get('kelly_frac', 0.5)  # 半Kelly

    # 波动率制度
    vol_regime = params.get('vol_regime', False)
    vol_lo_thresh = params.get('vol_lo_thresh', 0.30)  # HV百分位<30%=低波
    vol_hi_thresh = params.get('vol_hi_thresh', 0.70)  # HV百分位>70%=高波
    vol_lo_mult = params.get('vol_lo_mult', 1.5)  # 低波时乘数
    vol_hi_mult = params.get('vol_hi_mult', 0.5)  # 高波时乘数

    # 期权增强
    opt_enhance = params.get('opt_enhance', False)
    opt_risk_pct = params.get('opt_risk_pct', 0.01)
    opt_otm_pct = params.get('opt_otm_pct', 0.03)
    opt_min_score = params.get('opt_min_score', 0.8)

    # 回撤管理
    dd_reduce = params.get('dd_reduce', True)
    dd_threshold = params.get('dd_threshold', 0.15)
    dd_min_usage = params.get('dd_min_usage', 0.30)

    # 截面排名: 只取前N
    cross_rank = params.get('cross_rank', True)
    rank_pct = params.get('rank_pct', 0.05)  # 前5%

    equity = INIT_CAPITAL
    cash = INIT_CAPITAL
    positions = {}
    closed_pnls = []
    peak_equity = INIT_CAPITAL
    equity_curve = []

    # 滚动统计 (用于Kelly)
    rolling_wr = 0.5
    rolling_payoff = 1.5
    recent_trades = []  # 最近100笔

    for date in dates:
        day_data = date_map.get(date)
        if not day_data:
            continue

        # === 退出 ===
        for symbol in list(positions.keys()):
            pos = positions[symbol]
            if symbol not in day_data:
                continue
            row = day_data[symbol]
            price = row['close']
            hd = (date - pos['ed']).days

            should_close = False
            pnl_pct = (price / pos['ep'] - 1) * pos['d']

            # 时间退出
            if hd >= hold_days:
                should_close = True
            else:
                # 百分比止损
                if stop_loss_pct > 0 and pnl_pct < -stop_loss_pct:
                    should_close = True
                # 百分比止盈
                if take_profit_pct > 0 and pnl_pct > take_profit_pct:
                    should_close = True
                # 追踪止损
                if trailing_stop_pct > 0:
                    if not pos.get('max_pct'):
                        pos['max_pct'] = pnl_pct
                    else:
                        pos['max_pct'] = max(pos['max_pct'], pnl_pct)
                    if pos['max_pct'] > 0 and (pos['max_pct'] - pnl_pct) > trailing_stop_pct:
                        should_close = True

            if should_close:
                # 期货PnL
                pnl = (price - pos['fe']) * pos['d'] * pos['m'] * pos['fl']
                comm = price * pos['m'] * pos['fl'] * COMM
                cash += pos['fm'] + pnl - comm
                net = pnl - comm

                # 期权PnL
                if pos.get('oc', 0) > 0:
                    hv = row.get('hv_20', 0.25)
                    rem_T = max(0.001 / 365, 0.001)
                    val = bs_price(price, pos['ok'], rem_T, R_RATE, hv, pos['ot'])
                    intrinsic = max(price - pos['ok'], 0) if pos['d'] == 1 else max(pos['ok'] - price, 0)
                    val = max(val, intrinsic * 0.9)
                    tv = val * pos['m'] * pos['oc']
                    oc_comm = tv * OPT_COMM
                    opt_pnl = tv - pos['ox'] - oc_comm - pos.get('oxc', 0)
                    cash += tv - oc_comm
                    net += opt_pnl

                closed_pnls.append(net)
                recent_trades.append(net)
                if len(recent_trades) > 100:
                    recent_trades.pop(0)

                # 更新滚动统计
                if len(recent_trades) >= 20:
                    wins = sum(1 for t in recent_trades if t > 0)
                    rolling_wr = wins / len(recent_trades)
                    w_trades = [t for t in recent_trades if t > 0]
                    l_trades = [abs(t) for t in recent_trades if t <= 0]
                    if w_trades and l_trades:
                        rolling_payoff = np.mean(w_trades) / np.mean(l_trades)

                del positions[symbol]

        # === 入场 ===
        n_pos = len(positions)
        if n_pos < max_pos:
            cur_dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0

            # 确定有效保证金使用率
            eff_mu = margin_usage
            if dd_reduce and cur_dd < -dd_threshold:
                eff_mu = max(margin_usage * (1 + cur_dd / dd_threshold) * 0.5, dd_min_usage)

            # Kelly调整
            if use_kelly and len(recent_trades) >= 30:
                k = rolling_wr - (1 - rolling_wr) / max(rolling_payoff, 0.1)
                k = max(min(k, 0.5), 0.05)  # 限制Kelly在5-50%
                eff_mu = min(eff_mu * k / 0.25, 0.98)  # 归一化Kelly

            # 波动率制度调整
            # (per-instrument, applied below)

            # 计算所有品种的信号
            all_signals = []
            for symbol, row in day_data.items():
                if symbol in positions:
                    continue

                hv = row.get('hv_20', 0)
                if hv < 0.05 or hv > 0.70:
                    continue

                # ADX过滤
                adx = row.get('adx', 0)
                if use_adx and adx < min_adx:
                    continue

                # RSI极端过滤
                rsi = row.get('rsi', 50)
                if use_rsi and (rsi > rsi_upper or rsi < rsi_lower):
                    continue

                # HV百分位过滤
                hp = row.get('hv_pct', 0.5)
                if use_hv_pct and hp > hv_pct_hi:
                    continue

                # OI确认
                if use_oi and not row.get('oi_confirm', True):
                    continue

                # ATR%过滤 (太极端的跳过)
                atr_pct = row.get('atr_pct', 0)
                if atr_pct > 0.05:
                    continue

                # 计算综合评分
                score = compute_composite_score(row, mode)
                if abs(score) < min_score:
                    continue

                direction = 1 if score > 0 else -1
                all_signals.append((symbol, direction, abs(score), hv, hp, row))

            # 截面排名: 只取最强的几个
            if cross_rank and len(all_signals) > max_pos * 2:
                all_signals.sort(key=lambda x: x[2], reverse=True)
                n_take = max(max_pos - n_pos, 0)
                # 需要long和short方向都有机会
                longs = [(s, d, sc, hv, hp, r) for s, d, sc, hv, hp, r in all_signals if d == 1]
                shorts = [(s, d, sc, hv, hp, r) for s, d, sc, hv, hp, r in all_signals if d == -1]
                # 取每个方向最强的
                selected = longs[:n_take] + shorts[:n_take]
                selected.sort(key=lambda x: x[2], reverse=True)
                all_signals = selected

            # 按评分排序入场
            all_signals.sort(key=lambda x: x[2], reverse=True)

            for symbol, direction, score, hv, hp, row in all_signals:
                if len(positions) >= max_pos:
                    break

                S = row['close']
                mult, mr, _, _ = specs[symbol]
                mpl = S * mult * mr

                # 波动率制度调整
                vol_mult = 1.0
                if vol_regime:
                    if hp < vol_lo_thresh:
                        vol_mult = vol_lo_mult
                    elif hp > vol_hi_thresh:
                        vol_mult = vol_hi_mult

                # 波动率目标仓位管理
                vol_scalar = min(0.25 / hv, 3.0) if hv > 0 else 1.0

                # 目标仓位大小
                target_margin = equity * eff_mu / max_pos * vol_scalar * vol_mult
                fl = max(int(target_margin / mpl), 1)
                fm = mpl * fl
                fc = S * mult * fl * COMM

                # 保证金上限
                total_m = sum(p['fm'] for p in positions.values())
                if total_m + fm > equity * eff_mu:
                    fl = max(int((equity * eff_mu - total_m) / mpl), 0)
                    if fl <= 0:
                        continue
                    fm = mpl * fl
                    fc = S * mult * fl * COMM

                # 期权增强 (仅高分信号)
                oc, ox, oxc, ok, ot = 0, 0, 0, 0, None
                if opt_enhance and score >= opt_min_score:
                    ot = 'call' if direction == 1 else 'put'
                    ok = S * (1 + opt_otm_pct * direction)
                    T = hold_days / 365.0
                    premium = bs_price(S, ok, T, R_RATE, hv, ot)
                    if premium > 0:
                        cp = premium * mult
                        risk_amt = equity * opt_risk_pct
                        oc = max(int(risk_amt / cp), 1)
                        ox = cp * oc
                        oxc = ox * OPT_COMM

                total_needed = fm + fc + ox + oxc
                if total_needed > cash:
                    if oc > 0:
                        remain = cash - fm - fc
                        if remain > 0 and ox > 0:
                            oc = max(int(remain * 0.9 / (ox / max(oc, 1))), 0)
                            ox = (ox / max(oc, 1)) * oc if oc > 0 else 0
                            oxc = ox * OPT_COMM if oc > 0 else 0
                        else:
                            oc, ox, oxc = 0, 0, 0
                    total_needed = fm + fc + ox + oxc
                    if total_needed > cash:
                        fl = max(int((cash - ox - oxc) / (mpl + S * mult * COMM)), 0)
                        if fl <= 0 and oc <= 0:
                            continue
                        fm = mpl * fl
                        fc = S * mult * fl * COMM

                if fl == 0 and oc == 0:
                    continue

                cash -= fm + fc + ox + oxc
                positions[symbol] = {
                    'd': direction, 'ed': date, 'ep': S,
                    'fe': S * (1 + 0.0001 * direction),
                    'fl': fl, 'm': mult, 'fm': fm,
                    'oc': oc, 'ox': ox, 'oxc': oxc, 'ok': ok, 'ot': ot,
                    'max_pct': 0, 'score': score,
                }

        # === 权益 ===
        unrealized = 0
        for symbol, pos in positions.items():
            if symbol in day_data:
                price = day_data[symbol]['close']
                unrealized += (price - pos['fe']) * pos['d'] * pos['m'] * pos['fl']
                if pos.get('oc', 0) > 0:
                    hv = day_data[symbol].get('hv_20', 0.25)
                    hd = (date - pos['ed']).days
                    rem_T = max((hold_days - hd) / 365.0, 0.001)
                    val = bs_price(price, pos['ok'], rem_T, R_RATE, hv, pos['ot'])
                    intrinsic = max(price - pos['ok'], 0) if pos['d'] == 1 else max(pos['ok'] - price, 0)
                    val = max(val, intrinsic * 0.9)
                    unrealized += val * pos['m'] * pos['oc'] - pos['ox']

        equity = cash + unrealized
        peak_equity = max(peak_equity, equity)
        equity_curve.append((date, equity))
        if equity < 5000:
            break

    if not closed_pnls and equity <= 0:
        return None

    total_ret = (equity - INIT_CAPITAL) / INIT_CAPITAL
    if total_ret <= -1:
        return None

    days = (dates[-1] - dates[0]).days
    years = max(days / 365, 0.001)
    ann = float((1 + total_ret) ** (1 / years) - 1)

    eq = pd.DataFrame(equity_curve, columns=['date', 'equity'])
    eq['cummax'] = eq['equity'].cummax()
    eq['dd'] = (eq['equity'] - eq['cummax']) / eq['cummax']
    mdd = float(eq['dd'].min())

    pnls = np.array(closed_pnls) if closed_pnls else np.array([0])
    wr = float((pnls > 0).mean()) if len(pnls) > 0 else 0
    avg_w = float(pnls[pnls > 0].mean()) if (pnls > 0).any() else 0
    avg_l = float(abs(pnls[pnls <= 0].mean())) if (pnls <= 0).any() else 1
    pf = avg_w * (pnls > 0).sum() / (avg_l * (pnls <= 0).sum()) if (pnls <= 0).sum() > 0 and avg_l > 0 else 0

    eq['return'] = eq['equity'].pct_change()
    daily_ret = eq['return'].dropna()
    sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0

    return {
        'annual': ann, 'wr': wr, 'mdd': mdd, 'pf': pf,
        'trades': len(pnls), 'final': equity, 'sharpe': sharpe,
        **{k: v for k, v in params.items() if not isinstance(v, list)},
    }


def main():
    data_dir = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    print("加载数据...")
    t0 = time.time()
    data = load_data(data_dir)
    print(f"加载了 {len(data)} 个品种, {time.time()-t0:.1f}s")

    start_date = pd.Timestamp('2018-01-01')
    end_date = pd.Timestamp('2026-05-08')

    print("构建日期映射...")
    date_map, specs = build_date_map(data, start_date, end_date)
    dates = sorted(date_map.keys())
    print(f"交易日: {len(dates)}")

    results = []
    bt0 = time.time()

    # ============================================================
    # 扫描A: 趋势回调 + 不同杠杆和持有期
    # ============================================================
    print("\n=== 扫描A: 趋势回调入场 ===")
    prev = 0
    for mu in [0.70, 0.80, 0.90, 0.95]:
        for hd in [3, 5, 7, 10]:
            for ms in [0.2, 0.4, 0.6, 0.8]:
                params = dict(
                    margin_usage=mu, hold_days=hd, min_score=ms,
                    max_pos=3, mode='trend_pullback',
                    use_adx=True, min_adx=20,
                    use_hv_pct=True, hv_pct_hi=0.85,
                    use_rsi=True, rsi_upper=78, rsi_lower=22,
                    cross_rank=True,
                )
                r = fast_backtest(date_map, specs, dates, params)
                if r:
                    results.append(r)
    print(f"  A: {len(results)-prev}组, {time.time()-bt0:.0f}s")

    # ============================================================
    # 扫描B: Kelly动态仓位
    # ============================================================
    print("=== 扫描B: 自适应Kelly ===")
    prev = len(results)
    for mode in ['trend_pullback', 'full_combo', 'ewmac_breakout']:
        for mu in [0.80, 0.90, 0.95]:
            for hd in [3, 5, 7]:
                for kf in [0.3, 0.5, 0.7]:
                    params = dict(
                        margin_usage=mu, hold_days=hd, min_score=0.4,
                        max_pos=3, mode=mode,
                        use_adx=True, min_adx=18,
                        use_hv_pct=True, hv_pct_hi=0.85,
                        use_rsi=True, rsi_upper=78, rsi_lower=22,
                        use_kelly=True, kelly_frac=kf,
                        cross_rank=True,
                    )
                    r = fast_backtest(date_map, specs, dates, params)
                    if r:
                        results.append(r)
    print(f"  B: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # ============================================================
    # 扫描C: 波动率制度 + 止损止盈
    # ============================================================
    print("=== 扫描C: 波动率制度 + 退出策略 ===")
    prev = len(results)
    for mode in ['trend_pullback', 'full_combo']:
        for mu in [0.85, 0.90, 0.95]:
            for hd in [3, 5]:
                for sl in [0.03, 0.05, 0.08]:
                    # 止损 + 追踪止盈
                    for tp in [0.0, 0.10, 0.15]:
                        params = dict(
                            margin_usage=mu, hold_days=hd, min_score=0.4,
                            max_pos=3, mode=mode,
                            use_adx=True, min_adx=18,
                            use_hv_pct=True, hv_pct_hi=0.85,
                            use_rsi=True, rsi_upper=78, rsi_lower=22,
                            stop_loss_pct=sl, take_profit_pct=tp,
                            vol_regime=True,
                            vol_lo_thresh=0.30, vol_hi_thresh=0.70,
                            vol_lo_mult=1.5, vol_hi_mult=0.5,
                            cross_rank=True,
                        )
                        r = fast_backtest(date_map, specs, dates, params)
                        if r:
                            results.append(r)
                    # 追踪止损
                    for trail in [0.03, 0.05]:
                        params = dict(
                            margin_usage=mu, hold_days=hd, min_score=0.4,
                            max_pos=3, mode=mode,
                            use_adx=True, min_adx=18,
                            use_hv_pct=True, hv_pct_hi=0.85,
                            use_rsi=True, rsi_upper=78, rsi_lower=22,
                            trailing_stop_pct=trail,
                            vol_regime=True,
                            vol_lo_thresh=0.30, vol_hi_thresh=0.70,
                            vol_lo_mult=1.5, vol_hi_mult=0.5,
                            cross_rank=True,
                        )
                        r = fast_backtest(date_map, specs, dates, params)
                        if r:
                            results.append(r)
    print(f"  C: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # ============================================================
    # 扫描D: 均值回归模式
    # ============================================================
    print("=== 扫描D: 均值回归模式 ===")
    prev = len(results)
    for mu in [0.80, 0.90, 0.95]:
        for hd in [2, 3, 5]:
            for ms in [0.3, 0.5]:
                params = dict(
                    margin_usage=mu, hold_days=hd, min_score=ms,
                    max_pos=3, mode='mean_revert',
                    use_adx=False,
                    use_hv_pct=True, hv_pct_hi=0.90,
                    use_rsi=False,
                    cross_rank=True,
                )
                r = fast_backtest(date_map, specs, dates, params)
                if r:
                    results.append(r)
    print(f"  D: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # ============================================================
    # 扫描E: 截面动量 + Kelly + 期权增强
    # ============================================================
    print("=== 扫描E: 截面动量 + 期权增强 ===")
    prev = len(results)
    for mu in [0.60, 0.70, 0.80]:
        for hd in [3, 5, 7]:
            for opt_r in [0.01, 0.02]:
                params = dict(
                    margin_usage=mu, hold_days=hd, min_score=0.3,
                    max_pos=3, mode='cross_mom',
                    use_adx=True, min_adx=15,
                    use_hv_pct=True, hv_pct_hi=0.85,
                    use_rsi=True, rsi_upper=75, rsi_lower=25,
                    opt_enhance=True, opt_risk_pct=opt_r, opt_otm_pct=0.03,
                    opt_min_score=0.8,
                    use_kelly=True, kelly_frac=0.5,
                    cross_rank=True,
                )
                r = fast_backtest(date_map, specs, dates, params)
                if r:
                    results.append(r)
    print(f"  E: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # ============================================================
    # 扫描F: 全组合 + 极端杠杆 + 波动率制度
    # ============================================================
    print("=== 扫描F: 全组合极端杠杆 ===")
    prev = len(results)
    for mu in [0.95, 0.98]:
        for hd in [2, 3, 5]:
            for ms in [0.3, 0.5, 0.7]:
                for sl in [0.05, 0.08]:
                    params = dict(
                        margin_usage=mu, hold_days=hd, min_score=ms,
                        max_pos=3, mode='full_combo',
                        use_adx=True, min_adx=18,
                        use_hv_pct=True, hv_pct_hi=0.85,
                        use_rsi=True, rsi_upper=78, rsi_lower=22,
                        stop_loss_pct=sl,
                        use_kelly=True, kelly_frac=0.5,
                        vol_regime=True,
                        vol_lo_thresh=0.25, vol_hi_thresh=0.70,
                        vol_lo_mult=2.0, vol_hi_mult=0.4,
                        cross_rank=True,
                        dd_reduce=True, dd_threshold=0.15, dd_min_usage=0.30,
                    )
                    r = fast_backtest(date_map, specs, dates, params)
                    if r:
                        results.append(r)
    print(f"  F: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.0f}秒, {len(results)}组结果")

    # === 输出 ===
    results.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n\n{'模式':>16} {'保证金':>6} {'持有':>4} {'评分':>4} {'SL%':>4} {'TP%':>4} {'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5}")
    print("-" * 110)

    for r in results[:60]:
        sl_s = f"{r.get('stop_loss_pct',0)*100:.0f}" if r.get('stop_loss_pct',0) > 0 else ('T' if r.get('trailing_stop_pct',0) > 0 else '-')
        tp_s = f"{r.get('take_profit_pct',0)*100:.0f}" if r.get('take_profit_pct',0) > 0 else '-'
        print(f"{r.get('mode','')[:16]:>16} {r.get('margin_usage',0):>6.0%} {r.get('hold_days',0):>4} "
              f"{r.get('min_score',0):>4.1f} {sl_s:>4} {tp_s:>4} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5}")

    # 目标筛选
    print("\n\n" + "=" * 110)
    print("=== 目标: 年化>=600% & 胜率>=50% ===")
    good = [r for r in results if r['annual'] >= 6.0 and r['wr'] >= 0.50]
    if good:
        for r in sorted(good, key=lambda x: x['annual'], reverse=True)[:20]:
            print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                  f"Sharpe={r['sharpe']:>5.2f}  保证金={r.get('margin_usage',0):.0%}  "
                  f"持有={r.get('hold_days',0)}  模式={r.get('mode','')}  Kelly={'Y' if r.get('use_kelly') else 'N'}")
    else:
        print("  无")

    print("\n=== 目标: 年化>=300% & 胜率>=50% ===")
    good2 = [r for r in results if r['annual'] >= 3.0 and r['wr'] >= 0.50]
    if good2:
        for r in sorted(good2, key=lambda x: x['annual'], reverse=True)[:20]:
            print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                  f"Sharpe={r['sharpe']:>5.2f}  保证金={r.get('margin_usage',0):.0%}  "
                  f"持有={r.get('hold_days',0)}  模式={r.get('mode','')}")
    else:
        print("  无")

    print("\n=== 目标: 年化>=600% TOP 10 ===")
    t600 = [r for r in results if r['annual'] >= 6.0]
    if t600:
        for r in sorted(t600, key=lambda x: x['wr'], reverse=True)[:10]:
            print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                  f"模式={r.get('mode','')}")
    else:
        print("  无")

    print("\n=== 目标: 胜率>=50% TOP 15 ===")
    wr50 = [r for r in results if r['wr'] >= 0.50]
    if wr50:
        for r in sorted(wr50, key=lambda x: x['annual'], reverse=True)[:15]:
            print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                  f"Sharpe={r['sharpe']:>5.2f}  保证金={r.get('margin_usage',0):.0%}  "
                  f"持有={r.get('hold_days',0)}  模式={r.get('mode','')}")
    else:
        print("  无")

    # 保存
    output_dir = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(output_dir, exist_ok=True)
    save = []
    for r in results[:100]:
        s = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in r.items()}
        save.append(s)
    with open(os.path.join(output_dir, 'backtest_v39.json'), 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nTOP 100已保存到 backtest_results/backtest_v39.json")


if __name__ == '__main__':
    main()
