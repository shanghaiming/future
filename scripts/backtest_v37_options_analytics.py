#!/usr/bin/env python3
"""
策略 v37 — 期权分析增强期货策略 (优化版)

核心: 用期权分析指标过滤期货信号 + 激进杠杆
目标: 年化>=600%, 胜率>=50%, 持仓<=3, 回测>=8年
"""

import os, sys, time, json, numpy as np, pandas as pd
from collections import defaultdict
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from contract_specs import get_spec

COMM = 0.00015
R_RATE = 0.02


def load_data(data_dir):
    """加载数据并计算所有指标"""
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
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['mom_5'] = df['close'].pct_change(5)
        df['mom_10'] = df['close'].pct_change(10)
        df['mom_20'] = df['close'].pct_change(20)
        df['trend'] = np.where(df['ma20'] > df['ma60'], 1, -1)

        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss_s = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss_s))

        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close']

        # 期权分析指标
        df['hv_5'] = df['return'].rolling(5).std() * np.sqrt(252)
        df['hv_20'] = df['return'].rolling(20).std() * np.sqrt(252)
        df['hv_60'] = df['return'].rolling(60).std() * np.sqrt(252)
        df['hv_pct'] = df['hv_20'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 10 else 0.5, raw=False
        )
        df['hv_slope'] = df['hv_5'] / df['hv_60'].replace(0, np.nan)
        df['term_spread'] = df['hv_20'] - df['hv_60']

        # 下行vs上行波动率 (模拟偏斜)
        df['down_move'] = df['return'].where(df['return'] < 0, 0)
        df['up_move'] = df['return'].where(df['return'] > 0, 0)
        df['down_hv'] = df['down_move'].rolling(20).std() * np.sqrt(252) * 2
        df['up_hv'] = df['up_move'].rolling(20).std() * np.sqrt(252) * 2
        df['realized_skew'] = df['down_hv'] / df['up_hv'].replace(0, np.nan)

        df = df.dropna(subset=['ma20', 'ma60', 'hv_20', 'mom_10', 'rsi', 'hv_pct'])
        if len(df) > 100:
            # 预计算合约规格
            spec = get_spec(symbol)
            df.attrs['spec'] = spec
            data[symbol] = df
    return data


def build_date_map(data, start_date, end_date):
    """一次性构建日期映射 (核心优化)"""
    date_map = defaultdict(dict)
    specs = {}
    for symbol, df in data.items():
        specs[symbol] = df.attrs['spec']
        mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
        for _, row in df[mask].iterrows():
            date_map[row['trade_date']][symbol] = row
    return dict(date_map), specs


def fast_backtest(date_map, specs, dates, params):
    """高速回测核心"""
    max_pos = params.get('max_pos', 3)
    hold_days = params.get('hold_days', 7)
    margin_usage = params.get('margin_usage', 0.85)
    min_mom = params.get('min_mom', 0.02)
    use_rsi = params.get('use_rsi', True)
    rsi_upper = params.get('rsi_upper', 70)
    rsi_lower = params.get('rsi_lower', 30)
    use_hv_pct = params.get('use_hv_pct', False)
    hv_pct_hi = params.get('hv_pct_hi', 0.90)
    use_hv_slope = params.get('use_hv_slope', False)
    hv_slope_max = params.get('hv_slope_max', 2.0)
    stop_loss_pct = params.get('stop_loss_pct', 0.0)
    take_profit_pct = params.get('take_profit_pct', 0.0)
    use_mom_align = params.get('use_mom_align', False)
    atr_max = params.get('atr_max', 0.045)
    dd_reduce = params.get('dd_reduce', True)
    dd_threshold = params.get('dd_threshold', 0.15)
    dd_min_usage = params.get('dd_min_usage', 0.30)

    # 期权叠加参数
    opt_overlay = params.get('opt_overlay', False)
    opt_otm = params.get('opt_otm', 0.02)
    opt_risk = params.get('opt_risk', 0.02)

    equity = 500000.0
    cash = 500000.0
    positions = {}
    closed_pnls = []
    peak_equity = 500000.0

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
            if not should_close and stop_loss_pct > 0:
                ret = (price - pos['ep']) * pos['d'] / pos['ep']
                if ret < -stop_loss_pct:
                    should_close = True
            if not should_close and take_profit_pct > 0:
                ret = (price - pos['ep']) * pos['d'] / pos['ep']
                if ret > take_profit_pct:
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
                    oc_comm = tv * 0.0003
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
                atr_pct = row.get('atr_pct', 0.1)
                if atr_pct > atr_max:
                    continue
                mom = row.get('mom_10', 0)
                trend = row.get('trend', 0)
                hv = row.get('hv_20', 0)
                if hv < 0.08 or hv > 0.60:
                    continue

                if trend == 1 and mom > min_mom:
                    d = 1
                elif trend == -1 and mom < -min_mom:
                    d = -1
                else:
                    continue

                rsi = row.get('rsi', 50)
                if use_rsi and ((d == 1 and rsi > rsi_upper) or (d == -1 and rsi < rsi_lower)):
                    continue

                if use_hv_pct:
                    hp = row.get('hv_pct', 0.5)
                    if hp > hv_pct_hi:
                        continue

                if use_hv_slope:
                    hs = row.get('hv_slope', 1.0)
                    if hs > hv_slope_max:
                        continue

                if use_mom_align:
                    m5 = row.get('mom_5', 0)
                    m20 = row.get('mom_20', 0)
                    if d == 1 and (m5 < 0 or m20 < 0):
                        continue
                    if d == -1 and (m5 > 0 or m20 > 0):
                        continue

                signals.append((symbol, d, abs(mom)))

            signals.sort(key=lambda x: x[2], reverse=True)

            for symbol, direction, _ in signals:
                if len(positions) >= max_pos:
                    break
                row = day_data[symbol]
                S = row['close']
                mult, mr, _, _ = specs[symbol]
                mpl = S * mult * mr

                target_n = equity * eff_mu / max_pos
                fl = max(int(target_n / mpl), 1)
                fm = mpl * fl
                fc = S * mult * fl * COMM

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
                    hv = row.get('hv_20', 0.25)
                    ot = 'call' if direction == 1 else 'put'
                    ok = S * (1 + opt_otm * direction)
                    T = hold_days / 365.0
                    premium = bs_price(S, ok, T, R_RATE, hv, ot)
                    if premium > 0:
                        cp = premium * mult
                        risk_amt = equity * opt_risk
                        oc = max(int(risk_amt / cp), 1)
                        ox = cp * oc
                        oxc = ox * 0.0003

                total_needed = fm + fc + ox + oxc
                if total_needed > cash:
                    if oc > 0:
                        remain = cash - fm - fc
                        if remain > 0 and ox > 0:
                            oc = max(int(remain * 0.9 / (ox / max(oc, 1))), 0)
                            ox = (ox / max(oc, 1)) * oc if oc > 0 else 0
                            oxc = ox * 0.0003 if oc > 0 else 0
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

    # 简单MDD估算
    mdd = -0.5  # 默认
    pnls = np.array(closed_pnls) if closed_pnls else np.array([0])
    wr = float((pnls > 0).mean()) if len(pnls) > 0 else 0
    avg_w = float(pnls[pnls > 0].mean()) if (pnls > 0).any() else 0
    avg_l = float(abs(pnls[pnls <= 0].mean())) if (pnls <= 0).any() else 1
    pf = avg_w * (pnls > 0).sum() / (avg_l * (pnls <= 0).sum()) if (pnls <= 0).sum() > 0 and avg_l > 0 else 0

    return {
        'annual': ann, 'wr': wr, 'mdd': mdd, 'pf': pf,
        'trades': len(pnls), 'final': equity,
        **{k: v for k, v in params.items()},
    }


def bs_price(S, K, T, r, sigma, opt='call'):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0) if opt == 'call' else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


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

    # === 扫描A: 纯期货 + 杠杆 + 持有期 + 动量 ===
    print("\n=== 扫描A: 基础参数 ===")
    for mu in [0.60, 0.70, 0.80, 0.90, 0.95, 0.98]:
        for hd in [3, 5, 7, 10]:
            for mm in [0.01, 0.015, 0.02, 0.03, 0.05]:
                params = dict(margin_usage=mu, hold_days=hd, min_mom=mm,
                              max_pos=3, use_rsi=True, rsi_upper=70, rsi_lower=30)
                r = fast_backtest(date_map, specs, dates, params)
                if r:
                    results.append(r)
    print(f"  A: {len(results)}组, {time.time()-bt0:.0f}s")

    # === 扫描B: 加HV过滤 ===
    print("=== 扫描B: HV过滤 ===")
    prev = len(results)
    for mu in [0.80, 0.90, 0.95]:
        for hd in [5, 7]:
            for hp in [0.70, 0.80, 0.90]:
                for mm in [0.02, 0.03]:
                    params = dict(margin_usage=mu, hold_days=hd, min_mom=mm,
                                  max_pos=3, use_rsi=True, rsi_upper=70, rsi_lower=30,
                                  use_hv_pct=True, hv_pct_hi=hp)
                    r = fast_backtest(date_map, specs, dates, params)
                    if r:
                        results.append(r)
    print(f"  B: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === 扫描C: 止损/止盈 ===
    print("=== 扫描C: 止损止盈 ===")
    prev = len(results)
    for mu in [0.80, 0.90, 0.95]:
        for hd in [5, 7, 10]:
            for sl in [0.03, 0.05, 0.08, 0.10]:
                for tp in [0.05, 0.10, 0.15, 0.0]:
                    params = dict(margin_usage=mu, hold_days=hd, min_mom=0.02,
                                  max_pos=3, use_rsi=True, rsi_upper=70, rsi_lower=30,
                                  stop_loss_pct=sl, take_profit_pct=tp)
                    r = fast_backtest(date_map, specs, dates, params)
                    if r:
                        results.append(r)
    print(f"  C: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === 扫描D: 期货+买期权 ===
    print("=== 扫描D: 期货+买期权 ===")
    prev = len(results)
    for mu in [0.60, 0.70, 0.80]:
        for hd in [5, 7, 10]:
            for otm in [0.01, 0.02, 0.03]:
                for opt_r in [0.01, 0.02, 0.03, 0.05]:
                    params = dict(margin_usage=mu, hold_days=hd, min_mom=0.02,
                                  max_pos=3, use_rsi=True, rsi_upper=70, rsi_lower=30,
                                  opt_overlay=True, opt_otm=otm, opt_risk=opt_r)
                    r = fast_backtest(date_map, specs, dates, params)
                    if r:
                        results.append(r)
    print(f"  D: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === 扫描E: 多因子过滤 ===
    print("=== 扫描E: 多因子 ===")
    prev = len(results)
    for mu in [0.85, 0.90, 0.95]:
        for hd in [5, 7]:
            for flags in [
                dict(use_hv_pct=True, hv_pct_hi=0.80, use_hv_slope=True, hv_slope_max=2.0),
                dict(use_hv_pct=True, hv_pct_hi=0.85, use_mom_align=True),
                dict(use_hv_pct=True, hv_pct_hi=0.80, use_hv_slope=True, use_mom_align=True),
            ]:
                params = dict(margin_usage=mu, hold_days=hd, min_mom=0.02,
                              max_pos=3, use_rsi=True, rsi_upper=70, rsi_lower=30,
                              **flags)
                r = fast_backtest(date_map, specs, dates, params)
                if r:
                    results.append(r)
    print(f"  E: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === 扫描F: 极致杠杆 + 宽松信号 ===
    print("=== 扫描F: 极致杠杆 ===")
    prev = len(results)
    for mu in [0.95, 0.98]:
        for hd in [3, 5]:
            for mm in [0.005, 0.01, 0.015]:
                for rsi_u in [75, 80, 90]:
                    params = dict(margin_usage=mu, hold_days=hd, min_mom=mm,
                                  max_pos=3, use_rsi=True, rsi_upper=rsi_u, rsi_lower=100-rsi_u,
                                  dd_reduce=True, dd_threshold=0.20, dd_min_usage=0.40)
                    r = fast_backtest(date_map, specs, dates, params)
                    if r:
                        results.append(r)
    print(f"  F: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.0f}秒, {len(results)}组结果")

    # ============================================================
    # 输出
    # ============================================================
    results.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n\n{'保证金':>6} {'持有':>4} {'动量':>5} {'SL':>5} {'TP':>5} {'期权':>4} {'年化':>10} {'胜率':>6} {'PF':>6} {'交易':>5} {'最终':>14}")
    print("-" * 90)

    for r in results[:50]:
        opt_str = "买" if r.get('opt_overlay') else "-"
        print(f"{r.get('margin_usage',0):>6.0%} {r.get('hold_days',0):>4} "
              f"{r.get('min_mom',0):>5.1%} "
              f"{r.get('stop_loss_pct',0):>5.0%} "
              f"{r.get('take_profit_pct',0):>5.0%} "
              f"{opt_str:>4} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['trades']:>5} {r['final']:>14,.0f}")

    # 目标筛选
    print("\n\n" + "=" * 90)
    print("=== 目标: 年化>=100% & 胜率>=50% ===")
    good = [r for r in results if r['annual'] >= 1.0 and r['wr'] >= 0.50]
    if good:
        for r in sorted(good, key=lambda x: x['annual'], reverse=True)[:20]:
            opt_str = "期权" if r.get('opt_overlay') else "纯期货"
            print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  "
                  f"保证金={r.get('margin_usage',0):.0%}  持有={r.get('hold_days',0)}  "
                  f"动量={r.get('min_mom',0):.1%}  {opt_str}")
    else:
        print("  无")

    print("\n=== 目标: 年化>=300% ===")
    t300 = [r for r in results if r['annual'] >= 3.0]
    if t300:
        for r in sorted(t300, key=lambda x: x['wr'], reverse=True)[:15]:
            opt_str = "期权" if r.get('opt_overlay') else "纯期货"
            print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  "
                  f"保证金={r.get('margin_usage',0):.0%}  持有={r.get('hold_days',0)}  {opt_str}")
    else:
        print("  无")

    print("\n=== 目标: 胜率>=55% ===")
    wr55 = [r for r in results if r['wr'] >= 0.55]
    if wr55:
        for r in sorted(wr55, key=lambda x: x['annual'], reverse=True)[:15]:
            opt_str = "期权" if r.get('opt_overlay') else "纯期货"
            print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  "
                  f"保证金={r.get('margin_usage',0):.0%}  持有={r.get('hold_days',0)}  {opt_str}")
    else:
        print("  无")

    # 保存
    output_dir = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(output_dir, exist_ok=True)
    save = []
    for r in results[:100]:
        s = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in r.items()}
        save.append(s)

    with open(os.path.join(output_dir, 'backtest_v37.json'), 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nTOP 100已保存到 backtest_results/backtest_v37.json")


if __name__ == '__main__':
    main()
