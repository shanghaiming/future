#!/usr/bin/env python3
"""
策略 v41 — Risk-Adjusted TSMOM + 极端Vol-Targeting

基于研究发现:
1. RAMOM (风险调整动量): mom / vol → 信号质量显著提升
2. TSMOM在中国商品期货最佳lookback=1个月 (Cho et al 2019)
3. Vol-targeting: 低波时加仓,高波时减仓 → 自适应杠杆
4. Liu/Lu/Wang尾部风险调整: 上下偏矩不对称时减少敞口
5. 截面排名: 只选信号最强的3个
6. EWMAC (Carver): 最有效的趋势信号
7. 买期权凸性增强: 仅高分信号时买OTM期权

核心创新 vs v38:
- 标准化动量 (除以波动率)
- 更激进的vol-targeting (目标50-100%年化波动率)
- 尾部风险检测 (偏度不对称 → 减仓)
- 均线排列确认 (MA10>MA20>MA60 多头排列)
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

        # === 趋势 ===
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['ma120'] = df['close'].rolling(120).mean()
        df['trend'] = np.where(df['ma20'] > df['ma60'], 1, -1)
        df['trend_align'] = np.where(
            (df['ma10'] > df['ma20']) & (df['ma20'] > df['ma60']), 1,
            np.where(
                (df['ma10'] < df['ma20']) & (df['ma20'] < df['ma60']), -1, 0
            )
        )

        # === 动量 ===
        for lag in [5, 10, 20, 40, 60]:
            df[f'mom_{lag}'] = df['close'].pct_change(lag)

        # Risk-Adjusted Momentum (RAMOM): 标准化
        std20 = df['return'].rolling(20).std().replace(0, np.nan)
        std60 = df['return'].rolling(60).std().replace(0, np.nan)
        df['ramom_10'] = df['mom_10'] / (std20 * np.sqrt(10))
        df['ramom_20'] = df['mom_20'] / (std60 * np.sqrt(20))

        # === EWMAC ===
        for span in [4, 8, 16, 32]:
            ema_f = df['close'].ewm(span=span).mean()
            ema_s = df['close'].ewm(span=span*4).mean()
            df[f'ewmac_{span}'] = (ema_f - ema_s) / df['close'].rolling(20).std().replace(0, np.nan)

        # === 突破 ===
        for w in [10, 20, 40]:
            hh = df['close'].rolling(w).max()
            ll = df['close'].rolling(w).min()
            df[f'breakout_{w}'] = (df['close'] - 0.5*(hh+ll)) / (hh-ll+0.001) * 2

        # === 波动率 ===
        df['hv_5'] = df['return'].rolling(5).std() * np.sqrt(252)
        df['hv_10'] = df['return'].rolling(10).std() * np.sqrt(252)
        df['hv_20'] = df['return'].rolling(20).std() * np.sqrt(252)
        df['hv_60'] = df['return'].rolling(60).std() * np.sqrt(252)
        df['hv_pct'] = df['hv_20'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 10 else 0.5, raw=False
        )

        # 尾部风险: 偏度
        df['skew_20'] = df['return'].rolling(20).skew()
        df['skew_60'] = df['return'].rolling(60).skew()

        # 上行/下行分离矩 (Liu/Lu/Wang)
        ret = df['return']
        df['up_vol'] = ret.where(ret > 0, 0).rolling(20).std() * np.sqrt(252)
        df['dn_vol'] = ret.where(ret < 0, 0).rolling(20).std() * np.sqrt(252)
        df['tail_asym'] = (df['up_vol'] - df['dn_vol']) / (df['up_vol'] + df['dn_vol']).replace(0, np.nan)

        # === RSI ===
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss_s = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss_s))

        # === ATR ===
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close']

        df = df.dropna(subset=['ma20', 'ma60', 'hv_20', 'ramom_10', 'rsi'])
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


def compute_score(row, mode):
    """多模式信号评分"""
    score = 0.0
    w = 0.0

    if mode == 'ramom_ewmac':
        # RAMOM + EWMAC组合
        rm = row.get('ramom_10', 0)
        ew8 = row.get('ewmac_8', 0)
        ew16 = row.get('ewmac_16', 0)
        bo20 = row.get('breakout_20', 0)

        if not pd.isna(rm):
            score += np.sign(rm) * min(abs(rm), 3) * 2.0
            w += 2.0
        if not pd.isna(ew8):
            score += np.sign(ew8) * min(abs(ew8), 2)
            w += 1.0
        if not pd.isna(ew16):
            score += np.sign(ew16) * min(abs(ew16), 2)
            w += 1.0
        if not pd.isna(bo20):
            score += bo20 * 0.5
            w += 0.5

    elif mode == 'ramom_only':
        rm10 = row.get('ramom_10', 0)
        rm20 = row.get('ramom_20', 0)
        if not pd.isna(rm10):
            score += np.sign(rm10) * min(abs(rm10), 3) * 2.0
            w += 2.0
        if not pd.isna(rm20):
            score += np.sign(rm20) * min(abs(rm20), 3)
            w += 1.0

    elif mode == 'ewmac_bo':
        ew8 = row.get('ewmac_8', 0)
        ew16 = row.get('ewmac_16', 0)
        bo20 = row.get('breakout_20', 0)
        bo40 = row.get('breakout_40', 0)
        if not pd.isna(ew8):
            score += np.sign(ew8) * min(abs(ew8), 2)
            w += 1.0
        if not pd.isna(ew16):
            score += np.sign(ew16) * min(abs(ew16), 2)
            w += 1.0
        if not pd.isna(bo20):
            score += bo20
            w += 1.0
        if not pd.isna(bo40):
            score += bo40
            w += 1.0

    elif mode == 'full_ramom':
        # 全信号+RAMOM+尾部风险
        rm = row.get('ramom_10', 0)
        ew8 = row.get('ewmac_8', 0)
        ew16 = row.get('ewmac_16', 0)
        bo20 = row.get('breakout_20', 0)
        ta = row.get('tail_asym', 0)
        skew = row.get('skew_20', 0)

        if not pd.isna(rm):
            score += np.sign(rm) * min(abs(rm), 3) * 2.0
            w += 2.0
        if not pd.isna(ew8):
            score += np.sign(ew8) * min(abs(ew8), 2)
            w += 1.0
        if not pd.isna(ew16):
            score += np.sign(ew16) * min(abs(ew16), 2)
            w += 1.0
        if not pd.isna(bo20):
            score += bo20 * 0.5
            w += 0.5
        # 尾部风险调整: 偏度极端时减分
        if not pd.isna(ta):
            # 如果score>0但下行波动大(ta<0) → 减分
            if score > 0 and ta < -0.2:
                score *= 0.7
            elif score < 0 and ta > 0.2:
                score *= 0.7

    elif mode == 'trend_ramom':
        # 趋势确认 + RAMOM
        trend = row.get('trend', 0)
        align = row.get('trend_align', 0)
        rm = row.get('ramom_10', 0)
        ew8 = row.get('ewmac_8', 0)

        if not pd.isna(rm):
            score += np.sign(rm) * min(abs(rm), 3) * 2.0
            w += 2.0
        if not pd.isna(ew8):
            score += np.sign(ew8) * min(abs(ew8), 2)
            w += 1.0
        # 趋势确认加分
        if align != 0 and np.sign(align) == np.sign(score):
            score *= 1.5
        elif trend != 0 and np.sign(trend) != np.sign(score):
            score *= 0.5

    return score / max(w, 1) if w > 0 else 0


def fast_backtest(date_map, specs, dates, params):
    max_pos = params.get('max_pos', 3)
    hold_days = params.get('hold_days', 5)
    margin_usage = params.get('margin_usage', 0.90)
    min_score = params.get('min_score', 0.3)
    mode = params.get('mode', 'ramom_ewmac')

    # Vol-targeting
    vol_target = params.get('vol_target', 0.25)  # 目标年化波动率
    use_vol_target = params.get('use_vol_target', True)
    vol_max_mult = params.get('vol_max_mult', 4.0)  # 最大放大倍数

    # 过滤
    use_trend = params.get('use_trend', False)
    use_rsi = params.get('use_rsi', True)
    rsi_hi = params.get('rsi_hi', 75)
    rsi_lo = params.get('rsi_lo', 25)
    use_hv = params.get('use_hv', True)
    hv_hi = params.get('hv_hi', 0.85)

    # 尾部风险管理
    use_tail = params.get('use_tail', False)
    tail_threshold = params.get('tail_threshold', 0.3)

    # 退出
    stop_pct = params.get('stop_pct', 0.0)
    trail_pct = params.get('trail_pct', 0.0)

    # 期权
    opt_enhance = params.get('opt_enhance', False)
    opt_risk = params.get('opt_risk', 0.01)
    opt_otm = params.get('opt_otm', 0.03)

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

            should_close = hd >= hold_days
            pnl_pct = (price / pos['ep'] - 1) * pos['d']

            if not should_close:
                if stop_pct > 0 and pnl_pct < -stop_pct:
                    should_close = True
                if trail_pct > 0:
                    pos['mpct'] = max(pos.get('mpct', 0), pnl_pct)
                    if pos['mpct'] > 0 and (pos['mpct'] - pnl_pct) > trail_pct:
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
                    net += tv - pos['ox'] - oc_comm - pos.get('oxc', 0)
                    cash += tv - oc_comm

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

                rsi = row.get('rsi', 50)
                if use_rsi and (rsi > rsi_hi or rsi < rsi_lo):
                    continue

                hp = row.get('hv_pct', 0.5)
                if use_hv and hp > hv_hi:
                    continue

                score = compute_score(row, mode)
                if abs(score) < min_score:
                    continue

                direction = 1 if score > 0 else -1

                if use_trend:
                    trend = row.get('trend', 0)
                    if trend != 0 and direction != trend:
                        continue

                # 尾部风险: 偏度极端时跳过
                if use_tail:
                    ta = row.get('tail_asym', 0)
                    if not pd.isna(ta) and abs(ta) > tail_threshold:
                        continue

                signals.append((symbol, direction, abs(score), hv, row))

            signals.sort(key=lambda x: x[2], reverse=True)

            for symbol, direction, score, hv, row in signals:
                if len(positions) >= max_pos:
                    break

                S = row['close']
                mult, mr, _, _ = specs[symbol]
                mpl = S * mult * mr

                # Vol-targeting仓位
                vol_scalar = 1.0
                if use_vol_target and hv > 0:
                    vol_scalar = min(vol_target / hv, vol_max_mult)

                target = equity * margin_usage / max_pos * vol_scalar
                fl = max(int(target / mpl), 1)
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
                if opt_enhance and score >= 0.8:
                    ot = 'call' if direction == 1 else 'put'
                    ok = S * (1 + opt_otm * direction)
                    T = hold_days / 365.0
                    prem = bs_price(S, ok, T, R_RATE, hv, ot)
                    if prem > 0:
                        cp = prem * mult
                        oc = max(int(equity * opt_risk / cp), 1)
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
                    'mpct': 0,
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

    # === A: RAMOM+EWMAC 基础扫描 ===
    print("\n=== A: RAMOM+EWMAC ===")
    prev = 0
    for mu in [0.80, 0.90, 0.95]:
        for hd in [3, 5, 7]:
            for ms in [0.3, 0.5, 0.8]:
                for vt in [0.25, 0.40, 0.60]:
                    params = dict(
                        margin_usage=mu, hold_days=hd, min_score=ms,
                        max_pos=3, mode='ramom_ewmac',
                        vol_target=vt, vol_max_mult=4.0,
                    )
                    r = fast_backtest(date_map, specs, dates, params)
                    if r:
                        results.append(r)
    print(f"  A: {len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === B: 纯RAMOM ===
    print("=== B: 纯RAMOM ===")
    prev = len(results)
    for mu in [0.85, 0.90, 0.95]:
        for hd in [3, 5, 7]:
            for ms in [0.3, 0.5]:
                for vt in [0.30, 0.50, 0.80]:
                    params = dict(
                        margin_usage=mu, hold_days=hd, min_score=ms,
                        max_pos=3, mode='ramom_only',
                        vol_target=vt, vol_max_mult=5.0,
                    )
                    r = fast_backtest(date_map, specs, dates, params)
                    if r:
                        results.append(r)
    print(f"  B: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === C: 趋势确认+RAMOM ===
    print("=== C: 趋势确认+RAMOM ===")
    prev = len(results)
    for mu in [0.85, 0.90, 0.95]:
        for hd in [3, 5, 7]:
            for vt in [0.30, 0.50]:
                params = dict(
                    margin_usage=mu, hold_days=hd, min_score=0.4,
                    max_pos=3, mode='trend_ramom',
                    vol_target=vt, vol_max_mult=4.0,
                    use_trend=True,
                )
                r = fast_backtest(date_map, specs, dates, params)
                if r:
                    results.append(r)
    print(f"  C: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === D: 全信号+尾部风险 ===
    print("=== D: 全信号+尾部风险 ===")
    prev = len(results)
    for mu in [0.85, 0.90, 0.95]:
        for hd in [3, 5, 7]:
            for vt in [0.30, 0.50]:
                params = dict(
                    margin_usage=mu, hold_days=hd, min_score=0.4,
                    max_pos=3, mode='full_ramom',
                    vol_target=vt, vol_max_mult=4.0,
                    use_tail=True, tail_threshold=0.3,
                )
                r = fast_backtest(date_map, specs, dates, params)
                if r:
                    results.append(r)
    print(f"  D: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === E: EWMAC+Breakout (v38最佳) + 极端vol-target ===
    print("=== E: EWMAC+BO + 极端vol-target ===")
    prev = len(results)
    for mu in [0.90, 0.95, 0.98]:
        for hd in [2, 3, 5]:
            for vt in [0.50, 0.80, 1.20]:
                for vmm in [3.0, 5.0, 8.0]:
                    params = dict(
                        margin_usage=mu, hold_days=hd, min_score=0.5,
                        max_pos=3, mode='ewmac_bo',
                        vol_target=vt, vol_max_mult=vmm,
                    )
                    r = fast_backtest(date_map, specs, dates, params)
                    if r:
                        results.append(r)
    print(f"  E: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === F: 止损+追踪 ===
    print("=== F: 止损+追踪 ===")
    prev = len(results)
    for mode in ['ramom_ewmac', 'full_ramom']:
        for mu in [0.90, 0.95]:
            for hd in [3, 5]:
                for sl in [0.05, 0.08]:
                    params = dict(
                        margin_usage=mu, hold_days=hd, min_score=0.4,
                        max_pos=3, mode=mode,
                        vol_target=0.40, vol_max_mult=4.0,
                        stop_pct=sl,
                    )
                    r = fast_backtest(date_map, specs, dates, params)
                    if r:
                        results.append(r)
                    params2 = {**params, 'stop_pct': 0, 'trail_pct': sl}
                    r = fast_backtest(date_map, specs, dates, params2)
                    if r:
                        results.append(r)
    print(f"  F: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === G: 期权增强 ===
    print("=== G: 期权增强 ===")
    prev = len(results)
    for mode in ['ramom_ewmac', 'full_ramom']:
        for mu in [0.60, 0.70]:
            for hd in [3, 5]:
                params = dict(
                    margin_usage=mu, hold_days=hd, min_score=0.5,
                    max_pos=3, mode=mode,
                    vol_target=0.30, vol_max_mult=4.0,
                    opt_enhance=True, opt_risk=0.01, opt_otm=0.03,
                )
                r = fast_backtest(date_map, specs, dates, params)
                if r:
                    results.append(r)
    print(f"  G: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.0f}秒, {len(results)}组结果")

    # === 输出 ===
    results.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n\n{'模式':>14} {'保证金':>6} {'持有':>4} {'Vtgt':>4} {'Vmm':>3} {'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5}")
    print("-" * 100)

    for r in results[:60]:
        print(f"{r.get('mode','')[:14]:>14} {r.get('margin_usage',0):>6.0%} {r.get('hold_days',0):>4} "
              f"{r.get('vol_target',0):>4.0%} {r.get('vol_max_mult',0):>3.0f} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5}")

    # 目标筛选
    print("\n\n" + "=" * 100)
    for target_ann, target_wr in [(6.0, 0.50), (3.0, 0.50), (1.0, 0.50), (6.0, 0.45)]:
        label = f"年化>={target_ann*100:.0f}%"
        if target_wr >= 0.50:
            label += f" & 胜率>={target_wr*100:.0f}%"
        print(f"\n=== 目标: {label} ===")
        good = [r for r in results if r['annual'] >= target_ann and r['wr'] >= target_wr]
        if good:
            for r in sorted(good, key=lambda x: x['annual'], reverse=True)[:10]:
                print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                      f"Sharpe={r['sharpe']:>5.2f}  模式={r.get('mode','')}  "
                      f"保证金={r.get('margin_usage',0):.0%}  持有={r.get('hold_days',0)}  "
                      f"VT={r.get('vol_target',0):.0%}  VM={r.get('vol_max_mult',0):.0f}")
        else:
            print("  无")

    print("\n=== 年化TOP 10 ===")
    for r in sorted(results, key=lambda x: x['annual'], reverse=True)[:10]:
        print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
              f"模式={r.get('mode','')}  保证金={r.get('margin_usage',0):.0%}  "
              f"持有={r.get('hold_days',0)}  VT={r.get('vol_target',0):.0%}  VM={r.get('vol_max_mult',0):.0f}")

    print("\n=== 胜率TOP 10 ===")
    for r in sorted(results, key=lambda x: x['wr'], reverse=True)[:10]:
        print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
              f"模式={r.get('mode','')}  保证金={r.get('margin_usage',0):.0%}  "
              f"持有={r.get('hold_days',0)}  VT={r.get('vol_target',0):.0%}  VM={r.get('vol_max_mult',0):.0f}")

    # 保存
    output_dir = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(output_dir, exist_ok=True)
    save = []
    for r in results[:100]:
        s = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in r.items()}
        save.append(s)
    with open(os.path.join(output_dir, 'backtest_v41.json'), 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nTOP 100已保存到 backtest_results/backtest_v41.json")


if __name__ == '__main__':
    main()
