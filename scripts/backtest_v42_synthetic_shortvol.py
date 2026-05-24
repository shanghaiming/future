#!/usr/bin/env python3
"""
策略 v42 — 合成卖权 (Synthetic Short Vol via Futures)

核心思路: 用期货交易复制卖OTM期权的收益结构
- 卖期权为什么能高WR? 因为OTM期权大多数时间到期作废 → 收权利金
- 如何用期货复制?
  1. 入场: 趋势方向确认后入场
  2. 止盈: 极窄 (0.5-2%) → 像收权利金一样快速止盈 → 高WR
  3. 止损: 极宽 (3-8%) → 像卖期权的保证金,偶尔被触发
  4. 持有期: 短 (1-3天) → 更快的复利
  5. 高杠杆: 90%+保证金使用率
  6. 买OTM期权保护: 当确信度高时买入OTM期权对冲尾部风险

预期效果:
- WR 60-75% (因为止盈窄,容易触发)
- 年化取决于杠杆和频率
- PF取决于止盈/止损比率

数学分析:
- 如果TP=1%, SL=5%, WR=70%:
  E[trade] = 0.7 * 1% - 0.3 * 5% = 0.7% - 1.5% = -0.8% (负期望!)
- 如果TP=1%, SL=3%, WR=80%:
  E[trade] = 0.8 * 1% - 0.2 * 3% = 0.8% - 0.6% = +0.2% (正期望!)
- 需要TP/SL * (1-WR)/WR < 1
  即 WR > SL / (TP + SL)
  TP=1%, SL=3%: WR > 75%
  TP=1.5%, SL=4%: WR > 72.7%
  TP=2%, SL=5%: WR > 71.4%

关键: 止盈必须比止损窄很多,且WR必须足够高
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
        df['trend'] = np.where(df['ma20'] > df['ma60'], 1, -1)

        # 动量
        for lag in [3, 5, 10, 20]:
            df[f'mom_{lag}'] = df['close'].pct_change(lag)

        # EWMAC
        for span in [4, 8, 16]:
            ema_f = df['close'].ewm(span=span).mean()
            ema_s = df['close'].ewm(span=span*4).mean()
            df[f'ewmac_{span}'] = (ema_f - ema_s) / df['close'].rolling(20).std().replace(0, np.nan)

        # 突破
        hh20 = df['close'].rolling(20).max()
        ll20 = df['close'].rolling(20).min()
        df['breakout_20'] = (df['close'] - 0.5*(hh20+ll20)) / (hh20-ll20+0.001) * 2

        # RAMOM
        std20 = df['return'].rolling(20).std().replace(0, np.nan)
        df['ramom_10'] = df['mom_10'] / (std20 * np.sqrt(10))

        # 波动率
        df['hv_5'] = df['return'].rolling(5).std() * np.sqrt(252)
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

        # ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close']

        df = df.dropna(subset=['ma20', 'ma60', 'hv_20', 'mom_5', 'rsi'])
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
    margin_usage = params.get('margin_usage', 0.90)
    max_hold = params.get('max_hold', 5)  # 最大持有天数

    # 止盈止损 (核心参数)
    tp_pct = params.get('tp_pct', 0.01)   # 止盈百分比
    sl_pct = params.get('sl_pct', 0.03)   # 止损百分比

    # 信号
    mode = params.get('mode', 'trend_mom')
    min_score = params.get('min_score', 0.3)

    # 期权保护
    opt_protect = params.get('opt_protect', False)
    opt_risk = params.get('opt_risk', 0.005)

    # 波动率过滤
    use_hv = params.get('use_hv', True)
    hv_hi = params.get('hv_hi', 0.85)

    # Vol-targeting
    vol_target = params.get('vol_target', 0.40)
    vol_max = params.get('vol_max', 3.0)

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

            pnl_pct = (price / pos['ep'] - 1) * pos['d']

            should_close = False
            exit_price = price

            # 止盈: close到达TP
            if pnl_pct >= tp_pct:
                should_close = True
                exit_price = pos['ep'] * (1 + tp_pct * pos['d'])
            # 止损: close到达SL
            elif pnl_pct <= -sl_pct:
                should_close = True
                exit_price = pos['ep'] * (1 - sl_pct * pos['d'])
            # 最大持有期
            elif hd >= max_hold:
                should_close = True
                exit_price = price

            if should_close:
                pnl = (exit_price - pos['fe']) * pos['d'] * pos['m'] * pos['fl']
                comm = exit_price * pos['m'] * pos['fl'] * COMM
                cash += pos['fm'] + pnl - comm
                net = pnl - comm

                # 期权保护PnL
                if pos.get('oc', 0) > 0:
                    hv = row.get('hv_20', 0.25)
                    rem_T = max(0.001/365, 0.001)
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
                trend = row.get('trend', 0)

                direction = 0
                score = 0

                if mode == 'trend_mom':
                    # 趋势 + 短期动量
                    mom5 = row.get('mom_5', 0)
                    ew8 = row.get('ewmac_8', 0)
                    if pd.isna(mom5):
                        continue
                    if pd.isna(ew8):
                        continue

                    # 方向 = 趋势方向, 信号强度 = 动量 + EWMAC
                    if trend == 1 and ew8 > 0:
                        direction = 1
                        score = min(abs(ew8), 2) + min(abs(mom5) * 10, 1)
                    elif trend == -1 and ew8 < 0:
                        direction = -1
                        score = min(abs(ew8), 2) + min(abs(mom5) * 10, 1)

                elif mode == 'ewmac':
                    ew8 = row.get('ewmac_8', 0)
                    ew16 = row.get('ewmac_16', 0)
                    if pd.isna(ew8) or pd.isna(ew16):
                        continue
                    s = (np.sign(ew8) * min(abs(ew8), 2) + np.sign(ew16) * min(abs(ew16), 2)) / 2
                    if abs(s) >= min_score:
                        direction = 1 if s > 0 else -1
                        score = abs(s)

                elif mode == 'ramom_ewmac':
                    rm = row.get('ramom_10', 0)
                    ew8 = row.get('ewmac_8', 0)
                    if pd.isna(rm) or pd.isna(ew8):
                        continue
                    s = np.sign(rm) * min(abs(rm), 3) + np.sign(ew8) * min(abs(ew8), 2)
                    if abs(s) >= min_score * 2:
                        direction = 1 if s > 0 else -1
                        score = abs(s)

                elif mode == 'trend_ramom':
                    rm = row.get('ramom_10', 0)
                    if pd.isna(rm):
                        continue
                    if trend == 1 and rm > 0:
                        direction = 1
                        score = abs(rm)
                    elif trend == -1 and rm < 0:
                        direction = -1
                        score = abs(rm)

                if direction == 0:
                    continue

                # HV过滤
                hp = row.get('hv_pct', 0.5)
                if use_hv and hp > hv_hi:
                    continue

                # ATR%过滤 (极端波动跳过)
                atr_pct = row.get('atr_pct', 0)
                if atr_pct > 0.04:
                    continue

                signals.append((symbol, direction, score, hv, row))

            signals.sort(key=lambda x: x[2], reverse=True)

            for symbol, direction, score, hv, row in signals:
                if len(positions) >= max_pos:
                    break

                S = row['close']
                mult, mr, _, _ = specs[symbol]
                mpl = S * mult * mr

                # Vol-targeting
                vol_scalar = min(vol_target / hv, vol_max) if hv > 0 else 1.0

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

                # 期权保护 (买入反方向OTM期权)
                oc, ox, oxc, ok, ot = 0, 0, 0, 0, None
                if opt_protect:
                    # 买反方向期权作为保护
                    ot = 'put' if direction == 1 else 'call'  # 反方向
                    ok = S * (1 - 0.03 * direction)  # OTM
                    T = max_hold / 365.0
                    prem = bs_price(S, ok, T, R_RATE, hv, ot)
                    if prem > 0:
                        cp = prem * mult
                        oc = max(int(equity * opt_risk / cp), 1)
                        ox = cp * oc
                        oxc = ox * OPT_COMM

                total_needed = fm + fc + ox + oxc
                if total_needed > cash:
                    if oc > 0:
                        oc, ox, oxc = 0, 0, 0
                    total_needed = fm + fc
                    if total_needed > cash:
                        fl = max(int(cash / (mpl + S * mult * COMM)), 0)
                        if fl <= 0:
                            continue
                        fm = mpl * fl
                        fc = S * mult * fl * COMM

                if fl == 0:
                    continue

                cash -= fm + fc + ox + oxc
                positions[symbol] = {
                    'd': direction, 'ed': date, 'ep': S,
                    'fe': S * (1 + 0.0001 * direction),
                    'fl': fl, 'm': mult, 'fm': fm,
                    'oc': oc, 'ox': ox, 'oxc': oxc, 'ok': ok, 'ot': ot,
                }

        # === 权益 ===
        unrealized = 0
        for symbol, pos in positions.items():
            if symbol in day_data:
                price = day_data[symbol]['close']
                unrealized += (price - pos['fe']) * pos['d'] * pos['m'] * pos['fl']

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

    # === A: 合成卖权核心扫描 (窄TP + 宽SL) ===
    print("\n=== A: 合成卖权 (窄TP/宽SL) ===")
    prev = 0
    for tp in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]:
        for sl in [tp * 2, tp * 3, tp * 4, tp * 5]:
            for mu in [0.80, 0.90, 0.95]:
                for hd in [3, 5, 7]:
                    params = dict(
                        margin_usage=mu, tp_pct=tp, sl_pct=sl,
                        max_hold=hd, max_pos=3, mode='trend_mom',
                        min_score=0.3,
                    )
                    r = fast_backtest(date_map, specs, dates, params)
                    if r:
                        results.append(r)
    print(f"  A: {len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === B: EWMAC信号 ===
    print("=== B: EWMAC信号 ===")
    prev = len(results)
    for tp in [0.005, 0.01, 0.015, 0.02]:
        for sl in [tp * 2, tp * 3, tp * 4]:
            for mu in [0.85, 0.90, 0.95]:
                for hd in [3, 5]:
                    params = dict(
                        margin_usage=mu, tp_pct=tp, sl_pct=sl,
                        max_hold=hd, max_pos=3, mode='ewmac',
                        min_score=0.5,
                    )
                    r = fast_backtest(date_map, specs, dates, params)
                    if r:
                        results.append(r)
    print(f"  B: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === C: RAMOM+EWMAC ===
    print("=== C: RAMOM+EWMAC ===")
    prev = len(results)
    for tp in [0.005, 0.01, 0.015, 0.02]:
        for sl in [tp * 2, tp * 3, tp * 4]:
            for mu in [0.85, 0.90, 0.95]:
                params = dict(
                    margin_usage=mu, tp_pct=tp, sl_pct=sl,
                    max_hold=5, max_pos=3, mode='ramom_ewmac',
                    min_score=0.5,
                )
                r = fast_backtest(date_map, specs, dates, params)
                if r:
                    results.append(r)
    print(f"  C: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === D: 极端杠杆 + 窄TP ===
    print("=== D: 极端杠杆 ===")
    prev = len(results)
    for tp in [0.005, 0.01, 0.015]:
        for sl in [tp * 3, tp * 4, tp * 5]:
            for mu in [0.95, 0.98]:
                for vt in [0.50, 0.80, 1.20]:
                    params = dict(
                        margin_usage=mu, tp_pct=tp, sl_pct=sl,
                        max_hold=3, max_pos=3, mode='trend_mom',
                        min_score=0.3, vol_target=vt, vol_max=5.0,
                    )
                    r = fast_backtest(date_map, specs, dates, params)
                    if r:
                        results.append(r)
    print(f"  D: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === E: 期权保护 ===
    print("=== E: 期权保护 ===")
    prev = len(results)
    for tp in [0.01, 0.015, 0.02]:
        for sl in [tp * 3, tp * 4]:
            for mu in [0.70, 0.80]:
                params = dict(
                    margin_usage=mu, tp_pct=tp, sl_pct=sl,
                    max_hold=5, max_pos=3, mode='trend_mom',
                    min_score=0.3,
                    opt_protect=True, opt_risk=0.005,
                )
                r = fast_backtest(date_map, specs, dates, params)
                if r:
                    results.append(r)
    print(f"  E: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.0f}秒, {len(results)}组结果")

    # === 输出 ===
    results.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n\n{'TP':>5} {'SL':>5} {'保证金':>6} {'持有':>4} {'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5}")
    print("-" * 80)

    for r in results[:50]:
        print(f"{r.get('tp_pct',0)*100:>5.1f} {r.get('sl_pct',0)*100:>5.1f} "
              f"{r.get('margin_usage',0):>6.0%} {r.get('max_hold',0):>4} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5}")

    # 目标筛选
    print("\n\n" + "=" * 80)
    for target_ann, target_wr in [(6.0, 0.50), (3.0, 0.50), (1.0, 0.50), (6.0, 0.60), (6.0, 0.55)]:
        label = f"年化>={target_ann*100:.0f}%"
        if target_wr > 0:
            label += f" & 胜率>={target_wr*100:.0f}%"
        print(f"\n=== 目标: {label} ===")
        good = [r for r in results if r['annual'] >= target_ann and r['wr'] >= target_wr]
        if good:
            for r in sorted(good, key=lambda x: x['annual'], reverse=True)[:10]:
                tp = r.get('tp_pct', 0)
                sl = r.get('sl_pct', 0)
                print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                      f"TP={tp*100:.1f}%  SL={sl*100:.1f}%  保证金={r.get('margin_usage',0):.0%}  "
                      f"持有={r.get('max_hold',0)}  模式={r.get('mode','')}")
        else:
            print("  无")

    # 最小年化要求
    print(f"\n=== 年化>=600% TOP 5 ===")
    t600 = [r for r in results if r['annual'] >= 6.0]
    if t600:
        for r in sorted(t600, key=lambda x: x['wr'], reverse=True)[:5]:
            print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                  f"TP={r.get('tp_pct',0)*100:.1f}%  SL={r.get('sl_pct',0)*100:.1f}%")
    else:
        print("  无")

    print(f"\n=== 胜率>=60% TOP 10 ===")
    wr60 = [r for r in results if r['wr'] >= 0.60]
    if wr60:
        for r in sorted(wr60, key=lambda x: x['annual'], reverse=True)[:10]:
            print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                  f"TP={r.get('tp_pct',0)*100:.1f}%  SL={r.get('sl_pct',0)*100:.1f}%  保证金={r.get('margin_usage',0):.0%}")
    else:
        print("  无")

    print(f"\n=== 胜率>=70% TOP 10 ===")
    wr70 = [r for r in results if r['wr'] >= 0.70]
    if wr70:
        for r in sorted(wr70, key=lambda x: x['annual'], reverse=True)[:10]:
            print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                  f"TP={r.get('tp_pct',0)*100:.1f}%  SL={r.get('sl_pct',0)*100:.1f}%  保证金={r.get('margin_usage',0):.0%}")
    else:
        print("  无")

    # 理论分析
    print(f"\n\n=== 数学分析 ===")
    print(f"对于 TP/SL = X/Y 的策略:")
    print(f"  盈亏平衡WR = SL / (TP + SL)")
    for tp in [0.005, 0.01, 0.015, 0.02]:
        for sl_mult in [2, 3, 4, 5]:
            sl = tp * sl_mult
            be_wr = sl / (tp + sl)
            print(f"  TP={tp*100:.1f}%, SL={sl*100:.1f}% → 盈亏平衡WR = {be_wr:.1%}")

    # 保存
    output_dir = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(output_dir, exist_ok=True)
    save = []
    for r in results[:100]:
        s = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in r.items()}
        save.append(s)
    with open(os.path.join(output_dir, 'backtest_v42.json'), 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nTOP 100已保存到 backtest_results/backtest_v42.json")


if __name__ == '__main__':
    main()
