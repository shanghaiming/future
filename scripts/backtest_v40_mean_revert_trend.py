#!/usr/bin/env python3
"""
策略 v40 — 均值回归+趋势确认: 全新方向

核心洞察:
v38(动量): 3189%年化 但 WR=43%
v39(过度过滤): 24%年化 WR=50%
v34(卖期权): 698%年化 WR=74.8% (但有theta优势)

新思路: 均值回归 + 趋势确认 → 天然高WR
- 短期超跌/超涨后反转 (均值回归 → 高WR)
- 长期趋势方向一致 (提高胜率)
- 高杠杆 (放大收益)
- 截面排名 (只选最极端的3个)
- 买OTM期权增强凸性

入场:
- 多头: mom_3 < -X% (3日急跌) AND close > MA60 (长期上升趋势)
- 空头: mom_3 > +X% (3日急涨) AND close < MA60 (长期下降趋势)
- 截面排名: 选最极端的3个品种

退出:
- 固定持有期 (2-5天)
- RSI回归到中性退出
- 止损/止盈

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
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(d1)


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

        # 趋势
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['ma120'] = df['close'].rolling(120).mean()
        df['trend'] = np.where(df['ma20'] > df['ma60'], 1, -1)
        df['trend_strong'] = np.where(
            (df['ma10'] > df['ma20']) & (df['ma20'] > df['ma60']), 1,
            np.where(
                (df['ma10'] < df['ma20']) & (df['ma20'] < df['ma60']), -1, 0
            )
        )

        # 动量
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f'mom_{lag}'] = df['close'].pct_change(lag)

        # 波动率
        df['hv_5'] = df['return'].rolling(5).std() * np.sqrt(252)
        df['hv_10'] = df['return'].rolling(10).std() * np.sqrt(252)
        df['hv_20'] = df['return'].rolling(20).std() * np.sqrt(252)
        df['hv_60'] = df['return'].rolling(60).std() * np.sqrt(252)
        df['hv_pct'] = df['hv_20'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 10 else 0.5, raw=False
        )

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss_s = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss_s))

        # RSI极端度 (距离50的绝对值)
        df['rsi_extreme'] = abs(df['rsi'] - 50)

        # ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close']

        # EWMAC
        for span in [4, 8, 16]:
            ema_f = df['close'].ewm(span=span).mean()
            ema_s = df['close'].ewm(span=span*4).mean()
            df[f'ewmac_{span}'] = (ema_f - ema_s) / df['close'].rolling(20).std().replace(0, np.nan)

        # 突破
        hh20 = df['close'].rolling(20).max()
        ll20 = df['close'].rolling(20).min()
        df['breakout_20'] = (df['close'] - 0.5 * (hh20 + ll20)) / (hh20 - ll20 + 0.001) * 2

        # 偏斜
        df['ret_skew_20'] = df['return'].rolling(20).skew()

        # 成交量
        df['vol_ma20'] = df['vol'].rolling(20).mean()
        df['vol_ratio'] = df['vol'] / df['vol_ma20'].replace(0, np.nan)

        # OI
        if 'oi' in df.columns and df['oi'].notna().sum() > 100:
            df['oi_chg'] = df['oi'].pct_change(5)
        else:
            df['oi_chg'] = 0

        df = df.dropna(subset=['ma20', 'ma60', 'hv_20', 'mom_3', 'rsi'])
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


def fast_backtest(date_map, specs, dates, params):
    max_pos = params.get('max_pos', 3)
    hold_days = params.get('hold_days', 3)
    margin_usage = params.get('margin_usage', 0.90)

    # 入场信号参数
    mode = params.get('mode', 'mr_trend')  # mr_trend, mr_pure, mom_revert, ewmac_revert
    mom_lag = params.get('mom_lag', 3)
    min_extreme = params.get('min_extreme', 0.02)
    use_trend = params.get('use_trend', True)
    use_strong_trend = params.get('use_strong_trend', False)
    use_ewmac_confirm = params.get('use_ewmac_confirm', False)

    # 过滤
    use_hv_pct = params.get('use_hv_pct', False)
    hv_pct_hi = params.get('hv_pct_hi', 0.85)
    use_rsi = params.get('use_rsi', False)
    rsi_upper = params.get('rsi_upper', 80)
    rsi_lower = params.get('rsi_lower', 20)
    use_vol_confirm = params.get('use_vol_confirm', False)  # 成交量确认

    # 退出
    stop_loss_pct = params.get('stop_loss_pct', 0.0)
    take_profit_pct = params.get('take_profit_pct', 0.0)
    trailing_stop_pct = params.get('trailing_stop_pct', 0.0)
    rsi_exit = params.get('rsi_exit', False)  # RSI回归退出
    rsi_exit_long = params.get('rsi_exit_long', 60)
    rsi_exit_short = params.get('rsi_exit_short', 40)

    # 期权
    opt_enhance = params.get('opt_enhance', False)
    opt_risk_pct = params.get('opt_risk_pct', 0.01)
    opt_otm_pct = params.get('opt_otm_pct', 0.03)

    # 波动率仓位管理
    vol_target = params.get('vol_target', False)
    vol_target_pct = params.get('vol_target_pct', 0.25)

    equity = INIT_CAPITAL
    cash = INIT_CAPITAL
    positions = {}
    closed_pnls = []
    peak_equity = INIT_CAPITAL
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

            should_close = False
            pnl_pct = (price / pos['ep'] - 1) * pos['d']

            if hd >= hold_days:
                should_close = True
            else:
                if stop_loss_pct > 0 and pnl_pct < -stop_loss_pct:
                    should_close = True
                if take_profit_pct > 0 and pnl_pct > take_profit_pct:
                    should_close = True
                if trailing_stop_pct > 0:
                    if not pos.get('max_pct'):
                        pos['max_pct'] = pnl_pct
                    else:
                        pos['max_pct'] = max(pos['max_pct'], pnl_pct)
                    if pos['max_pct'] > 0 and (pos['max_pct'] - pnl_pct) > trailing_stop_pct:
                        should_close = True
                # RSI回归退出
                if rsi_exit and not should_close:
                    rsi = row.get('rsi', 50)
                    if pos['d'] == 1 and rsi > rsi_exit_long:
                        should_close = True
                    elif pos['d'] == -1 and rsi < rsi_exit_short:
                        should_close = True

            if should_close:
                pnl = (price - pos['fe']) * pos['d'] * pos['m'] * pos['fl']
                comm = price * pos['m'] * pos['fl'] * COMM
                cash += pos['fm'] + pnl - comm
                net = pnl - comm

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
        if len(positions) < max_pos:
            signals = []

            for symbol, row in day_data.items():
                if symbol in positions:
                    continue
                hv = row.get('hv_20', 0)
                if hv < 0.05 or hv > 0.70:
                    continue

                mom = row.get(f'mom_{mom_lag}', 0)
                if pd.isna(mom):
                    continue

                rsi = row.get('rsi', 50)
                trend = row.get('trend', 0)
                strong_trend = row.get('trend_strong', 0)

                # 信号计算
                direction = 0
                strength = 0

                if mode == 'mr_trend':
                    # 均值回归 + 趋势确认
                    # 多头: 急跌(mom<-min) + 上升趋势
                    if mom < -min_extreme and (not use_trend or trend == 1):
                        direction = 1
                        strength = abs(mom) + (0.1 if strong_trend == 1 else 0)
                    # 空头: 急涨(mom>+min) + 下降趋势
                    elif mom > min_extreme and (not use_trend or trend == -1):
                        direction = -1
                        strength = abs(mom) + (0.1 if strong_trend == -1 else 0)

                elif mode == 'mr_pure':
                    # 纯均值回归 (无趋势过滤)
                    if mom < -min_extreme:
                        direction = 1
                        strength = abs(mom)
                    elif mom > min_extreme:
                        direction = -1
                        strength = abs(mom)

                elif mode == 'mr_rsi':
                    # RSI极端反转
                    if rsi < 25:
                        direction = 1
                        strength = (50 - rsi) / 50
                    elif rsi > 75:
                        direction = -1
                        strength = (rsi - 50) / 50
                    # 结合动量
                    if direction == 1 and mom < 0:
                        strength += abs(mom)
                    elif direction == -1 and mom > 0:
                        strength += abs(mom)

                elif mode == 'mom_revert':
                    # 动量 + 回调入场
                    # 趋势向上 + 短期回调
                    ew8 = row.get('ewmac_8', 0)
                    ew16 = row.get('ewmac_16', 0)
                    bo20 = row.get('breakout_20', 0)

                    if not pd.isna(ew8) and not pd.isna(ew16):
                        trend_score = (np.sign(ew8) * min(abs(ew8), 2) +
                                      np.sign(ew16) * min(abs(ew16), 2)) / 2
                        # 趋势方向 + 短期回调
                        if trend_score > 0.3 and mom < -min_extreme * 0.5:
                            direction = 1
                            strength = trend_score + abs(mom) * 5
                        elif trend_score < -0.3 and mom > min_extreme * 0.5:
                            direction = -1
                            strength = abs(trend_score) + abs(mom) * 5

                elif mode == 'ewmac_revert':
                    # EWMAC信号 + 极端回调
                    ew8 = row.get('ewmac_8', 0)
                    ew16 = row.get('ewmac_16', 0)
                    if pd.isna(ew8) or pd.isna(ew16):
                        continue
                    trend_dir = 1 if (ew8 + ew16) > 0 else -1
                    # 趋势方向 + 回调
                    if trend_dir == 1 and mom < -min_extreme * 0.5:
                        direction = 1
                        strength = abs(ew8 + ew16) + abs(mom) * 5
                    elif trend_dir == -1 and mom > min_extreme * 0.5:
                        direction = -1
                        strength = abs(ew8 + ew16) + abs(mom) * 5

                if direction == 0:
                    continue

                # 额外过滤
                if use_hv_pct:
                    hp = row.get('hv_pct', 0.5)
                    if hp > hv_pct_hi:
                        continue

                if use_rsi:
                    if rsi > rsi_upper or rsi < rsi_lower:
                        continue

                if use_ewmac_confirm:
                    ew8 = row.get('ewmac_8', 0)
                    if not pd.isna(ew8):
                        if direction == 1 and ew8 < -0.5:
                            continue
                        elif direction == -1 and ew8 > 0.5:
                            continue

                if use_vol_confirm:
                    vr = row.get('vol_ratio', 1)
                    if vr < 0.8:
                        continue

                signals.append((symbol, direction, strength, hv, row))

            # 截面排名: 选最强的
            signals.sort(key=lambda x: x[2], reverse=True)

            for symbol, direction, strength, hv, row in signals:
                if len(positions) >= max_pos:
                    break

                S = row['close']
                mult, mr, _, _ = specs[symbol]
                mpl = S * mult * mr

                # 波动率目标仓位
                vol_scalar = 1.0
                if vol_target and hv > 0:
                    vol_scalar = min(vol_target_pct / hv, 3.0)

                target_margin = equity * margin_usage / max_pos * vol_scalar
                fl = max(int(target_margin / mpl), 1)
                fm = mpl * fl
                fc = S * mult * fl * COMM

                total_m = sum(p['fm'] for p in positions.values())
                if total_m + fm > equity * margin_usage:
                    fl = max(int((equity * margin_usage - total_m) / mpl), 0)
                    if fl <= 0:
                        continue
                    fm = mpl * fl
                    fc = S * mult * fl * COMM

                # 期权增强
                oc, ox, oxc, ok, ot = 0, 0, 0, 0, None
                if opt_enhance:
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
                    'max_pct': 0,
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
    # A: 均值回归+趋势确认 (核心)
    # ============================================================
    print("\n=== A: 均值回归+趋势 ===")
    prev = 0
    for mu in [0.80, 0.90, 0.95]:
        for hd in [2, 3, 5]:
            for me in [0.01, 0.02, 0.03, 0.04, 0.05]:
                for lag in [2, 3, 5]:
                    params = dict(
                        margin_usage=mu, hold_days=hd, mode='mr_trend',
                        mom_lag=lag, min_extreme=me,
                        use_trend=True, max_pos=3,
                    )
                    r = fast_backtest(date_map, specs, dates, params)
                    if r:
                        results.append(r)
    print(f"  A: {len(results)-prev}组, {time.time()-bt0:.0f}s")

    # ============================================================
    # B: 纯均值回归 (无趋势过滤)
    # ============================================================
    print("=== B: 纯均值回归 ===")
    prev = len(results)
    for mu in [0.80, 0.90, 0.95, 0.98]:
        for hd in [2, 3, 5]:
            for me in [0.02, 0.03, 0.05, 0.07]:
                params = dict(
                    margin_usage=mu, hold_days=hd, mode='mr_pure',
                    mom_lag=3, min_extreme=me,
                    use_trend=False, max_pos=3,
                )
                r = fast_backtest(date_map, specs, dates, params)
                if r:
                    results.append(r)
    print(f"  B: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # ============================================================
    # C: RSI极端反转
    # ============================================================
    print("=== C: RSI极端反转 ===")
    prev = len(results)
    for mu in [0.80, 0.90, 0.95]:
        for hd in [2, 3, 5]:
            params = dict(
                margin_usage=mu, hold_days=hd, mode='mr_rsi',
                mom_lag=3, min_extreme=0.01,
                use_trend=False, max_pos=3,
            )
            r = fast_backtest(date_map, specs, dates, params)
            if r:
                results.append(r)
    print(f"  C: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # ============================================================
    # D: EWMAC动量 + 回调入场
    # ============================================================
    print("=== D: EWMAC+回调 ===")
    prev = len(results)
    for mu in [0.80, 0.90, 0.95]:
        for hd in [2, 3, 5]:
            for me in [0.01, 0.02, 0.03]:
                params = dict(
                    margin_usage=mu, hold_days=hd, mode='mom_revert',
                    mom_lag=3, min_extreme=me,
                    max_pos=3,
                )
                r = fast_backtest(date_map, specs, dates, params)
                if r:
                    results.append(r)
    print(f"  D: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # ============================================================
    # E: 最佳模式 + 止损/止盈/追踪
    # ============================================================
    print("=== E: 退出策略优化 ===")
    prev = len(results)
    for mode in ['mr_trend', 'mr_pure']:
        for mu in [0.90, 0.95]:
            for hd in [2, 3, 5]:
                for sl in [0.03, 0.05]:
                    # 止损
                    params = dict(
                        margin_usage=mu, hold_days=hd, mode=mode,
                        mom_lag=3, min_extreme=0.02,
                        use_trend=(mode == 'mr_trend'), max_pos=3,
                        stop_loss_pct=sl,
                    )
                    r = fast_backtest(date_map, specs, dates, params)
                    if r:
                        results.append(r)
                    # 追踪止损
                    params2 = {**params, 'stop_loss_pct': 0, 'trailing_stop_pct': sl}
                    r = fast_backtest(date_map, specs, dates, params2)
                    if r:
                        results.append(r)
                # RSI退出
                params3 = dict(
                    margin_usage=mu, hold_days=hd, mode=mode,
                    mom_lag=3, min_extreme=0.02,
                    use_trend=(mode == 'mr_trend'), max_pos=3,
                    rsi_exit=True, rsi_exit_long=60, rsi_exit_short=40,
                )
                r = fast_backtest(date_map, specs, dates, params3)
                if r:
                    results.append(r)
    print(f"  E: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # ============================================================
    # F: 期权增强
    # ============================================================
    print("=== F: 期权增强 ===")
    prev = len(results)
    for mode in ['mr_trend', 'mr_pure']:
        for mu in [0.60, 0.70]:
            for hd in [3, 5]:
                for opt_r in [0.01, 0.02]:
                    params = dict(
                        margin_usage=mu, hold_days=hd, mode=mode,
                        mom_lag=3, min_extreme=0.02,
                        use_trend=(mode == 'mr_trend'), max_pos=3,
                        opt_enhance=True, opt_risk_pct=opt_r, opt_otm_pct=0.03,
                    )
                    r = fast_backtest(date_map, specs, dates, params)
                    if r:
                        results.append(r)
    print(f"  F: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # ============================================================
    # G: 波动率目标 + 极端杠杆
    # ============================================================
    print("=== G: 波动率目标 ===")
    prev = len(results)
    for mode in ['mr_trend', 'mr_pure', 'mom_revert']:
        for mu in [0.95, 0.98]:
            for hd in [2, 3]:
                for vt in [0.20, 0.30, 0.40]:
                    params = dict(
                        margin_usage=mu, hold_days=hd, mode=mode,
                        mom_lag=3, min_extreme=0.02,
                        use_trend=(mode == 'mr_trend'), max_pos=3,
                        vol_target=True, vol_target_pct=vt,
                    )
                    r = fast_backtest(date_map, specs, dates, params)
                    if r:
                        results.append(r)
    print(f"  G: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.0f}秒, {len(results)}组结果")

    # === 输出 ===
    results.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n\n{'模式':>14} {'保证金':>6} {'持有':>4} {'极值':>4} {'SL':>4} {'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5}")
    print("-" * 100)

    for r in results[:60]:
        sl_s = f"{r.get('stop_loss_pct',0)*100:.0f}" if r.get('stop_loss_pct',0) > 0 else ('T' if r.get('trailing_stop_pct',0) > 0 else 'R' if r.get('rsi_exit') else '-')
        print(f"{r.get('mode','')[:14]:>14} {r.get('margin_usage',0):>6.0%} {r.get('hold_days',0):>4} "
              f"{r.get('min_extreme',0):>4.0%} {sl_s:>4} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5}")

    # 目标筛选
    print("\n\n" + "=" * 100)
    print("=== 目标: 年化>=600% & 胜率>=50% ===")
    good = [r for r in results if r['annual'] >= 6.0 and r['wr'] >= 0.50]
    if good:
        for r in sorted(good, key=lambda x: x['annual'], reverse=True)[:20]:
            print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                  f"模式={r.get('mode','')}  保证金={r.get('margin_usage',0):.0%}  "
                  f"持有={r.get('hold_days',0)}  极值={r.get('min_extreme',0):.0%}")
    else:
        print("  无")

    print("\n=== 目标: 年化>=300% & 胜率>=50% ===")
    good2 = [r for r in results if r['annual'] >= 3.0 and r['wr'] >= 0.50]
    if good2:
        for r in sorted(good2, key=lambda x: x['annual'], reverse=True)[:20]:
            print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                  f"模式={r.get('mode','')}  保证金={r.get('margin_usage',0):.0%}  "
                  f"持有={r.get('hold_days',0)}  极值={r.get('min_extreme',0):.0%}")
    else:
        print("  无")

    print("\n=== 目标: 年化>=100% & 胜率>=50% ===")
    good3 = [r for r in results if r['annual'] >= 1.0 and r['wr'] >= 0.50]
    if good3:
        for r in sorted(good3, key=lambda x: x['annual'], reverse=True)[:20]:
            print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                  f"模式={r.get('mode','')}  保证金={r.get('margin_usage',0):.0%}  "
                  f"持有={r.get('hold_days',0)}  极值={r.get('min_extreme',0):.0%}")
    else:
        print("  无")

    print("\n=== 年化TOP 10 ===")
    for r in sorted(results, key=lambda x: x['annual'], reverse=True)[:10]:
        print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
              f"模式={r.get('mode','')}  保证金={r.get('margin_usage',0):.0%}  "
              f"持有={r.get('hold_days',0)}  极值={r.get('min_extreme',0):.0%}")

    print("\n=== 胜率TOP 10 ===")
    for r in sorted(results, key=lambda x: x['wr'], reverse=True)[:10]:
        print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
              f"模式={r.get('mode','')}  保证金={r.get('margin_usage',0):.0%}  "
              f"持有={r.get('hold_days',0)}  极值={r.get('min_extreme',0):.0%}")

    # 保存
    output_dir = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(output_dir, exist_ok=True)
    save = []
    for r in results[:100]:
        s = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in r.items()}
        save.append(s)
    with open(os.path.join(output_dir, 'backtest_v40.json'), 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nTOP 100已保存到 backtest_results/backtest_v40.json")


if __name__ == '__main__':
    main()
