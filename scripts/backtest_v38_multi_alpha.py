#!/usr/bin/env python3
"""
策略 v38 — 多Alpha期货策略 (基于pysystemtrade/Carver框架)

参考:
- Rob Carver pysystemtrade: Sharpe 1.3, 30% annual, 54% WR, 146 instruments
- 学术TSMOM: vol-scaling是主要回报来源
- Carry是最强的独立信号 (Sharpe 0.9-0.95)
- 期权分析: HV百分位用于判断期权便宜/昂贵

创新:
1. 多信号融合: momentum + carry + skew + breakout
2. 极短持有期 (1-3天): 更快复利
3. 波动率目标仓位管理: vol-targeting
4. Kelly仓位管理
5. 买ATM期权保护 (selective: 仅当HV低时)
6. 期权分析过滤信号

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

        # === 趋势类信号 (Carver风格) ===
        # EWMAC (指数加权移动平均交叉)
        for span in [4, 8, 16, 32, 64]:
            ema_fast = df['close'].ewm(span=span).mean()
            ema_slow = df['close'].ewm(span=span*4).mean()
            df[f'ewmac_{span}'] = (ema_fast - ema_slow) / df['close'].rolling(20).std()

        # Breakout
        for window in [10, 20, 40, 80]:
            df[f'breakout_{window}'] = (
                (df['close'] - df['close'].rolling(window).min()) /
                (df['close'].rolling(window).max() - df['close'].rolling(window).min() + 0.001) - 0.5
            ) * 2

        # Momentum
        for lag in [5, 10, 20, 40, 60]:
            df[f'mom_{lag}'] = df['close'].pct_change(lag)

        # Normalized momentum (Carver)
        df['norm_mom'] = df['mom_10'] / df['return'].rolling(20).std().replace(0, np.nan) / np.sqrt(10)

        # === Carry信号 (近似) ===
        # 用OI变化+价格结构近似carry
        # 正carry: 近月 > 远月 (backwardation) → 做多获利
        # 负carry: 近月 < 远月 (contango) → 做空获利
        # 近似: 如果OI在增加且价格上涨 → 新资金进入 → 类似正carry
        if 'oi' in df.columns and df['oi'].notna().sum() > 100:
            df['oi_chg'] = df['oi'].pct_change(5)
            # Carry近似: OI增加+价格上涨 = 多头进入 = 正carry
            df['carry_proxy'] = np.where(
                df['oi_chg'] > 0,  # OI增加
                np.sign(df['mom_20']),  # 价格趋势方向
                0
            )
            df['carry_proxy'] = df['carry_proxy'].rolling(20).mean()
        else:
            df['carry_proxy'] = 0

        # === 均值回归/偏斜信号 ===
        # 收益率偏斜 (Carver: skewabs)
        df['ret_skew_20'] = df['return'].rolling(20).skew()
        df['ret_skew_60'] = df['return'].rolling(60).skew()

        # === 波动率信号 ===
        df['hv_5'] = df['return'].rolling(5).std() * np.sqrt(252)
        df['hv_10'] = df['return'].rolling(10).std() * np.sqrt(252)
        df['hv_20'] = df['return'].rolling(20).std() * np.sqrt(252)
        df['hv_60'] = df['return'].rolling(60).std() * np.sqrt(252)
        df['hv_120'] = df['return'].rolling(120).std() * np.sqrt(252)

        # HV百分位 (252天)
        df['hv_pct'] = df['hv_20'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 10 else 0.5, raw=False
        )

        # HV变化率
        df['hv_mom'] = df['hv_20'].pct_change(5)

        # 波动率期限结构
        df['hv_slope'] = df['hv_5'] / df['hv_60'].replace(0, np.nan)

        # === 基础指标 ===
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['trend'] = np.where(df['ma20'] > df['ma60'], 1, -1)

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss_s = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss_s))

        # ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close']

        # 成交量
        df['vol_ma20'] = df['vol'].rolling(20).mean()
        df['vol_ratio'] = df['vol'] / df['vol_ma20']

        df = df.dropna(subset=['ma20', 'ma60', 'hv_20', 'mom_10', 'rsi', 'hv_pct'])
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


def compute_signal_score(row, signal_set):
    """计算多因子信号评分"""
    score = 0.0
    count = 0

    if 'ewmac' in signal_set:
        # EWMAC 8和16 (Carver最有效的参数)
        ew8 = row.get('ewmac_8', 0)
        ew16 = row.get('ewmac_16', 0)
        if not pd.isna(ew8):
            score += np.sign(ew8) * min(abs(ew8), 2)
            count += 1
        if not pd.isna(ew16):
            score += np.sign(ew16) * min(abs(ew16), 2)
            count += 1

    if 'breakout' in signal_set:
        bo20 = row.get('breakout_20', 0)
        bo40 = row.get('breakout_40', 0)
        if not pd.isna(bo20):
            score += bo20
            count += 1
        if not pd.isna(bo40):
            score += bo40
            count += 1

    if 'momentum' in signal_set:
        m5 = row.get('mom_5', 0)
        m10 = row.get('mom_10', 0)
        m20 = row.get('mom_20', 0)
        if not pd.isna(m10):
            score += np.sign(m10) * min(abs(m10) * 10, 2)
            count += 1

    if 'norm_mom' in signal_set:
        nm = row.get('norm_mom', 0)
        if not pd.isna(nm):
            score += np.sign(nm) * min(abs(nm), 2)
            count += 1

    if 'carry' in signal_set:
        cp = row.get('carry_proxy', 0)
        if not pd.isna(cp):
            score += np.sign(cp) * min(abs(cp), 2)
            count += 1

    if 'skew' in signal_set:
        sk = row.get('ret_skew_20', 0)
        if not pd.isna(sk):
            # 负偏斜 → 反转向上 (做多信号)
            score += -np.sign(sk) * min(abs(sk), 1)
            count += 1

    if 'vol_mom' in signal_set:
        hv_mom = row.get('hv_mom', 0)
        if not pd.isna(hv_mom):
            # 波动率下降 → 趋势延续
            score += -np.sign(hv_mom) * min(abs(hv_mom) * 5, 1)
            count += 1

    return score / max(count, 1)


def fast_backtest(date_map, specs, dates, params):
    max_pos = params.get('max_pos', 3)
    hold_days = params.get('hold_days', 3)
    margin_usage = params.get('margin_usage', 0.90)
    min_score = params.get('min_score', 0.5)
    signal_set = params.get('signal_set', ['ewmac', 'momentum', 'breakout'])

    # 过滤
    use_rsi = params.get('use_rsi', True)
    rsi_upper = params.get('rsi_upper', 70)
    rsi_lower = params.get('rsi_lower', 30)
    use_hv_pct = params.get('use_hv_pct', True)
    hv_pct_hi = params.get('hv_pct_hi', 0.85)
    use_trend = params.get('use_trend', True)

    # 退出
    stop_loss_atr = params.get('stop_loss_atr', 0.0)  # ATR倍数止损
    take_profit_atr = params.get('take_profit_atr', 0.0)
    trailing_stop = params.get('trailing_stop', False)
    trail_atr_mult = params.get('trail_atr_mult', 3.0)

    # 期权
    opt_overlay = params.get('opt_overlay', False)
    opt_otm_pct = params.get('opt_otm_pct', 0.0)  # 0=ATM
    opt_risk_pct = params.get('opt_risk_pct', 0.02)
    opt_selective = params.get('opt_selective', False)  # 仅低HV时买期权
    opt_hv_threshold = params.get('opt_hv_threshold', 0.40)  # HV百分位阈值

    # 回撤管理
    dd_reduce = params.get('dd_reduce', True)
    dd_threshold = params.get('dd_threshold', 0.20)
    dd_min_usage = params.get('dd_min_usage', 0.30)

    equity = 500000.0
    cash = 500000.0
    positions = {}
    closed_pnls = []
    peak_equity = 500000.0
    equity_curve = []

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
            atr = row.get('atr', pos['ep'] * 0.02)

            should_close = False
            # 时间退出
            if hd >= hold_days:
                should_close = True
            # ATR止损
            elif stop_loss_atr > 0:
                adverse = (pos['ep'] - price) * pos['d']
                if adverse > stop_loss_atr * atr:
                    should_close = True
            # ATR止盈
            elif take_profit_atr > 0:
                favorable = (price - pos['ep']) * pos['d']
                if favorable > take_profit_atr * atr:
                    should_close = True
            # 追踪止损
            if trailing_stop and not should_close:
                cur_pnl = (price - pos['ep']) * pos['d']
                if pos.get('max_pnl', 0) == 0:
                    pos['max_pnl'] = max(cur_pnl, 0)
                else:
                    pos['max_pnl'] = max(pos['max_pnl'], cur_pnl)
                if pos['max_pnl'] > 0:
                    draw_from_max = pos['max_pnl'] - cur_pnl
                    if draw_from_max > trail_atr_mult * atr:
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
                del positions[symbol]

        # === 入场 ===
        n_pos = len(positions)
        if n_pos < max_pos:
            cur_dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0
            if dd_reduce and cur_dd < -dd_threshold:
                eff_mu = max(margin_usage * 0.5, dd_min_usage)
            else:
                eff_mu = margin_usage

            signals = []
            for symbol, row in day_data.items():
                if symbol in positions:
                    continue
                hv = row.get('hv_20', 0)
                if hv < 0.08 or hv > 0.60:
                    continue

                # 趋势过滤
                if use_trend:
                    trend = row.get('trend', 0)
                else:
                    trend = 0

                # RSI过滤
                rsi = row.get('rsi', 50)
                if use_rsi and ((rsi > rsi_upper) or (rsi < rsi_lower)):
                    continue

                # HV百分位过滤
                hp = row.get('hv_pct', 0.5)
                if use_hv_pct and hp > hv_pct_hi:
                    continue

                # 计算信号评分
                score = compute_signal_score(row, signal_set)

                # 方向: score > 0 → long, score < 0 → short
                if abs(score) < min_score:
                    continue
                direction = 1 if score > 0 else -1

                # 趋势一致性
                if use_trend and trend != 0 and direction != trend:
                    continue

                signals.append((symbol, direction, abs(score), hv, row))

            # 按评分排序
            signals.sort(key=lambda x: x[2], reverse=True)

            for symbol, direction, score, hv, row in signals:
                if len(positions) >= max_pos:
                    break

                S = row['close']
                mult, mr, _, _ = specs[symbol]
                mpl = S * mult * mr

                # 波动率目标仓位管理
                # Carver: position_size = target_vol / (instrument_vol * instrument_value)
                # 我们简化: 按波动率反比缩放
                if hv > 0:
                    vol_scalar = min(0.25 / hv, 3.0)  # 目标25%年化波动率, 最大3x
                else:
                    vol_scalar = 1.0

                target_n = equity * eff_mu / max_pos * vol_scalar
                fl = max(int(target_n / mpl), 1)
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

                # 期权叠加
                oc, ox, oxc, ok, ot = 0, 0, 0, 0, None
                if opt_overlay:
                    # 选择性期权: 仅当HV低时购买 (期权便宜)
                    should_buy_opt = True
                    if opt_selective:
                        hp = row.get('hv_pct', 0.5)
                        if hp > opt_hv_threshold:
                            should_buy_opt = False

                    if should_buy_opt:
                        ot = 'call' if direction == 1 else 'put'
                        ok = S * (1 + opt_otm_pct * direction)  # ATM或轻微OTM
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
                    'max_pnl': 0,
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

    total_ret = (equity - 500000) / 500000
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

    # 信号集定义
    signal_sets = {
        'ewmac_mom': ['ewmac', 'momentum'],
        'ewmac_bo': ['ewmac', 'breakout'],
        'ewmac_mom_bo': ['ewmac', 'momentum', 'breakout'],
        'ewmac_carry': ['ewmac', 'carry'],
        'ewmac_mom_skew': ['ewmac', 'momentum', 'skew'],
        'full': ['ewmac', 'momentum', 'breakout', 'carry', 'skew', 'vol_mom'],
        'full_nocarry': ['ewmac', 'momentum', 'breakout', 'skew', 'vol_mom'],
        'norm_mom': ['norm_mom'],
    }

    # === 扫描A: 多信号 + 短持有期 + 高杠杆 ===
    print("\n=== 扫描A: 多信号 + 短持有期 ===")
    prev = 0
    for sig_name, sig_set in signal_sets.items():
        for mu in [0.80, 0.90, 0.95]:
            for hd in [1, 2, 3, 5]:
                for ms in [0.3, 0.5, 0.8]:
                    params = dict(
                        margin_usage=mu, hold_days=hd, min_score=ms,
                        max_pos=3, signal_set=sig_set,
                        use_rsi=True, rsi_upper=70, rsi_lower=30,
                        use_hv_pct=True, hv_pct_hi=0.85,
                        use_trend=False,
                        opt_overlay=False,
                    )
                    r = fast_backtest(date_map, specs, dates, params)
                    if r:
                        r['sig'] = sig_name
                        results.append(r)
    print(f"  A: {len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === 扫描B: 止损/止盈/追踪止损 ===
    print("=== 扫描B: 退出策略 ===")
    prev = len(results)
    best_sigs = ['ewmac_mom_bo', 'full_nocarry', 'ewmac_mom']
    for sig_name in best_sigs:
        sig_set = signal_sets[sig_name]
        for mu in [0.85, 0.90, 0.95]:
            for hd in [2, 3, 5]:
                for sl_atr in [1.5, 2.0, 3.0]:
                    # 止损
                    params = dict(
                        margin_usage=mu, hold_days=hd, min_score=0.5,
                        max_pos=3, signal_set=sig_set,
                        use_rsi=True, rsi_upper=70, rsi_lower=30,
                        use_hv_pct=True, hv_pct_hi=0.85,
                        use_trend=False,
                        stop_loss_atr=sl_atr, take_profit_atr=0,
                        opt_overlay=False,
                    )
                    r = fast_backtest(date_map, specs, dates, params)
                    if r:
                        r['sig'] = sig_name
                        results.append(r)
                    # 追踪止损
                    params2 = {**params, 'stop_loss_atr': 0, 'trailing_stop': True, 'trail_atr_mult': sl_atr}
                    r = fast_backtest(date_map, specs, dates, params2)
                    if r:
                        r['sig'] = sig_name + '_trail'
                        results.append(r)
    print(f"  B: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === 扫描C: 买ATM期权保护 (选择性) ===
    print("=== 扫描C: ATM期权保护 ===")
    prev = len(results)
    for sig_name in ['ewmac_mom_bo', 'full_nocarry']:
        sig_set = signal_sets[sig_name]
        for mu in [0.60, 0.70, 0.80]:
            for hd in [3, 5]:
                for otm in [0.0, 0.01]:  # ATM 或 1% OTM
                    for opt_r in [0.01, 0.02, 0.03]:
                        # 选择性: 仅低HV时
                        params = dict(
                            margin_usage=mu, hold_days=hd, min_score=0.5,
                            max_pos=3, signal_set=sig_set,
                            use_rsi=True, rsi_upper=70, rsi_lower=30,
                            use_hv_pct=True, hv_pct_hi=0.85,
                            use_trend=False,
                            opt_overlay=True, opt_otm_pct=otm, opt_risk_pct=opt_r,
                            opt_selective=True, opt_hv_threshold=0.40,
                        )
                        r = fast_backtest(date_map, specs, dates, params)
                        if r:
                            r['sig'] = sig_name + '_selopt'
                            results.append(r)
    print(f"  C: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === 扫描D: 趋势过滤 + 极端杠杆 ===
    print("=== 扫描D: 趋势+极端杠杆 ===")
    prev = len(results)
    for sig_name in ['ewmac_mom_bo', 'full_nocarry']:
        sig_set = signal_sets[sig_name]
        for mu in [0.95, 0.98]:
            for hd in [1, 2, 3]:
                for ms in [0.3, 0.5]:
                    params = dict(
                        margin_usage=mu, hold_days=hd, min_score=ms,
                        max_pos=3, signal_set=sig_set,
                        use_rsi=True, rsi_upper=75, rsi_lower=25,
                        use_hv_pct=True, hv_pct_hi=0.90,
                        use_trend=True,
                        opt_overlay=False,
                        dd_reduce=True, dd_threshold=0.25, dd_min_usage=0.40,
                    )
                    r = fast_backtest(date_map, specs, dates, params)
                    if r:
                        r['sig'] = sig_name + '_trend'
                        results.append(r)
    print(f"  D: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.0f}秒, {len(results)}组结果")

    # === 输出 ===
    results.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n\n{'信号':>16} {'保证金':>6} {'持有':>4} {'评分':>4} {'SL':>5} {'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5}")
    print("-" * 100)

    for r in results[:60]:
        sl_str = f"{r.get('stop_loss_atr',0):.1f}" if r.get('stop_loss_atr',0) > 0 else ('T' if r.get('trailing_stop') else '-')
        print(f"{r.get('sig','')[:16]:>16} {r.get('margin_usage',0):>6.0%} {r.get('hold_days',0):>4} "
              f"{r.get('min_score',0):>4.1f} {sl_str:>5} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5}")

    # 目标筛选
    print("\n\n" + "=" * 100)
    print("=== 目标: 年化>=100% & 胜率>=50% ===")
    good = [r for r in results if r['annual'] >= 1.0 and r['wr'] >= 0.50]
    if good:
        for r in sorted(good, key=lambda x: x['annual'], reverse=True)[:20]:
            print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                  f"Sharpe={r['sharpe']:>5.2f}  保证金={r.get('margin_usage',0):.0%}  "
                  f"持有={r.get('hold_days',0)}  信号={r.get('sig','')}")
    else:
        print("  无")

    print("\n=== 目标: 年化>=600% ===")
    t600 = [r for r in results if r['annual'] >= 6.0]
    if t600:
        for r in sorted(t600, key=lambda x: x['wr'], reverse=True)[:15]:
            print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                  f"信号={r.get('sig','')}")
    else:
        print("  无")

    print("\n=== 目标: 胜率>=50% TOP 15 ===")
    wr50 = [r for r in results if r['wr'] >= 0.50]
    if wr50:
        for r in sorted(wr50, key=lambda x: x['annual'], reverse=True)[:15]:
            print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                  f"Sharpe={r['sharpe']:>5.2f}  保证金={r.get('margin_usage',0):.0%}  "
                  f"持有={r.get('hold_days',0)}  信号={r.get('sig','')}")
    else:
        print("  无")

    # 保存
    output_dir = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(output_dir, exist_ok=True)
    save = []
    for r in results[:100]:
        s = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in r.items()}
        if 'signal_set' in s:
            del s['signal_set']
        save.append(s)
    with open(os.path.join(output_dir, 'backtest_v38.json'), 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nTOP 100已保存到 backtest_results/backtest_v38.json")


if __name__ == '__main__':
    main()
