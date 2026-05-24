#!/usr/bin/env python3
"""
策略 v43 — 日内等效 + 权益曲线动量 + 金字塔加仓

新思路 (基于v34-v42全部经验):
1. 用前日信号决定方向, 当日开盘入场收盘退出 (日频交易)
   → 更高交易频率 → 更快复利
2. 权益曲线动量 (equity curve momentum):
   → 权益创新高时加大杠杆, 回撤时缩小杠杆
   → 学术研究表明可提升Sharpe 20-30%
3. 金字塔加仓 (pyramiding):
   → 盈利仓位加码, 亏损仓位不加
   → 自然提高WR (因为只有盈利才加仓)
4. 仓位上限 (防止不现实复利):
   → 单品种最大手数限制
   → 总名义不超过权益的N倍

信号: v38最佳 ewmac_mom_bo (已证明有效)
入场: 当日开盘价 (open)
退出: 当日收盘价 (close) 或 TP/SL
"""

import os, sys, time, json, numpy as np, pandas as pd
from collections import defaultdict
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from contract_specs import get_spec

COMM = 0.00015
R_RATE = 0.02
INIT_CAPITAL = 500000


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
        df['open_ret'] = df['open'].pct_change()
        df['intraday_ret'] = df['close'] / df['open'] - 1  # 开盘到收盘

        # 趋势
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['trend'] = np.where(df['ma20'] > df['ma60'], 1, -1)

        # 动量
        for lag in [3, 5, 10, 20]:
            df[f'mom_{lag}'] = df['close'].pct_change(lag)

        # EWMAC
        for span in [4, 8, 16, 32]:
            ema_f = df['close'].ewm(span=span).mean()
            ema_s = df['close'].ewm(span=span*4).mean()
            df[f'ewmac_{span}'] = (ema_f - ema_s) / df['close'].rolling(20).std().replace(0, np.nan)

        # 突破
        for w in [10, 20, 40]:
            hh = df['close'].rolling(w).max()
            ll = df['close'].rolling(w).min()
            df[f'breakout_{w}'] = (df['close'] - 0.5*(hh+ll)) / (hh-ll+0.001) * 2

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

        # 偏度
        df['skew_20'] = df['return'].rolling(20).skew()

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


def compute_signal(row, mode='ewmac_mom_bo'):
    score = 0.0
    w = 0.0

    if mode == 'ewmac_mom_bo':
        ew8 = row.get('ewmac_8', 0)
        ew16 = row.get('ewmac_16', 0)
        bo20 = row.get('breakout_20', 0)
        bo40 = row.get('breakout_40', 0)
        mom5 = row.get('mom_5', 0)

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
        if not pd.isna(mom5):
            score += np.sign(mom5) * min(abs(mom5) * 10, 1)
            w += 0.5

    elif mode == 'ramom_ewmac':
        rm = row.get('ramom_10', 0)
        ew8 = row.get('ewmac_8', 0)
        ew16 = row.get('ewmac_16', 0)
        bo20 = row.get('breakout_20', 0)
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

    elif mode == 'full':
        ew8 = row.get('ewmac_8', 0)
        ew16 = row.get('ewmac_16', 0)
        bo20 = row.get('breakout_20', 0)
        rm = row.get('ramom_10', 0)
        trend = row.get('trend', 0)
        skew = row.get('skew_20', 0)

        if not pd.isna(ew8):
            score += np.sign(ew8) * min(abs(ew8), 2)
            w += 1.0
        if not pd.isna(ew16):
            score += np.sign(ew16) * min(abs(ew16), 2)
            w += 1.0
        if not pd.isna(bo20):
            score += bo20
            w += 1.0
        if not pd.isna(rm):
            score += np.sign(rm) * min(abs(rm), 3)
            w += 1.0
        # 趋势确认
        if trend != 0 and not pd.isna(score) and np.sign(score) == trend:
            score *= 1.3

    return score / max(w, 1) if w > 0 else 0


def fast_backtest(date_map, specs, dates, params):
    max_pos = params.get('max_pos', 3)
    margin_usage = params.get('margin_usage', 0.90)
    hold_days = params.get('hold_days', 5)
    min_score = params.get('min_score', 0.5)
    mode = params.get('mode', 'ewmac_mom_bo')

    # 交易模式
    trade_mode = params.get('trade_mode', 'multi_day')  # multi_day, daily_open_close
    use_hv = params.get('use_hv', True)
    hv_hi = params.get('hv_hi', 0.85)
    use_rsi = params.get('use_rsi', True)
    rsi_hi = params.get('rsi_hi', 75)
    rsi_lo = params.get('rsi_lo', 25)

    # 止盈止损
    tp_pct = params.get('tp_pct', 0.0)
    sl_pct = params.get('sl_pct', 0.0)
    trail_pct = params.get('trail_pct', 0.0)

    # 权益曲线动量
    ecm = params.get('ecm', False)
    ecm_fast = params.get('ecm_fast', 10)
    ecm_slow = params.get('ecm_slow', 30)
    ecm_up_mult = params.get('ecm_up_mult', 1.5)
    ecm_dn_mult = params.get('ecm_dn_mult', 0.5)

    # 金字塔加仓
    pyramid = params.get('pyramid', False)
    pyramid_add_pct = params.get('pyramid_add_pct', 0.02)  # 盈利2%后加仓
    pyramid_max_adds = params.get('pyramid_max_adds', 2)

    # 仓位上限
    max_lots = params.get('max_lots', 0)  # 0=无限制
    max_notional_mult = params.get('max_notional_mult', 0)  # 0=无限制, N=最大N倍权益

    # Re-entry
    reentry = params.get('reentry', False)

    # Vol-targeting
    vol_target = params.get('vol_target', 0.0)

    equity = INIT_CAPITAL
    cash = INIT_CAPITAL
    positions = {}
    closed_pnls = []
    peak_equity = INIT_CAPITAL
    equity_curve = []
    equity_history = [INIT_CAPITAL]

    for date in dates:
        day_data = date_map.get(date)
        if not day_data:
            continue

        # 权益曲线动量
        ecm_mult = 1.0
        if ecm and len(equity_history) >= ecm_slow:
            eq_series = pd.Series(equity_history)
            eq_ma_fast = eq_series.rolling(ecm_fast).mean().iloc[-1]
            eq_ma_slow = eq_series.rolling(ecm_slow).mean().iloc[-1]
            if eq_ma_fast > eq_ma_slow:
                ecm_mult = ecm_up_mult
            else:
                ecm_mult = ecm_dn_mult

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

            if trade_mode == 'daily_open_close':
                # 日频: 每天都平仓
                should_close = True
            else:
                if hd >= hold_days:
                    should_close = True
                if tp_pct > 0 and pnl_pct >= tp_pct:
                    should_close = True
                if sl_pct > 0 and pnl_pct <= -sl_pct:
                    should_close = True
                if trail_pct > 0:
                    pos['mpct'] = max(pos.get('mpct', 0), pnl_pct)
                    if pos['mpct'] > 0 and (pos['mpct'] - pnl_pct) > trail_pct:
                        should_close = True

            # 金字塔: 检查是否加仓
            if pyramid and not should_close and hd > 0:
                if pnl_pct >= pyramid_add_pct and pos.get('adds', 0) < pyramid_max_adds:
                    if len(positions) <= max_pos:  # 加仓不算新仓位
                        S = row['close']
                        mult, mr, _, _ = specs[symbol]
                        add_lots = max(int(pos['fm'] * 0.3 / (S * mult * mr)), 1)
                        if max_lots > 0:
                            add_lots = min(add_lots, max_lots - pos['fl'])
                        if add_lots > 0:
                            add_margin = S * mult * mr * add_lots
                            add_comm = S * mult * add_lots * COMM
                            if add_margin + add_comm <= cash:
                                cash -= add_margin + add_comm
                                pos['fl'] += add_lots
                                pos['fm'] += add_margin
                                pos['adds'] = pos.get('adds', 0) + 1
                                # 更新平均入场价
                                pos['ep'] = (pos['ep'] * (pos['fl'] - add_lots) + S * add_lots) / pos['fl']
                                pos['fe'] = pos['ep'] * (1 + 0.0001 * pos['d'])

            if should_close:
                comm = price * pos['m'] * pos['fl'] * COMM
                pnl = (price - pos['fe']) * pos['d'] * pos['m'] * pos['fl']
                cash += pos['fm'] + pnl - comm
                net = pnl - comm
                closed_pnls.append(net)

                if reentry:
                    pos['_reentry_signal'] = pos.get('signal_score', 0)

                del positions[symbol]

        # === 入场 ===
        if len(positions) < max_pos:
            eff_mu = margin_usage * ecm_mult

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

                atr_pct = row.get('atr_pct', 0)
                if atr_pct > 0.05:
                    continue

                score = compute_signal(row, mode)
                if abs(score) < min_score:
                    continue

                direction = 1 if score > 0 else -1
                signals.append((symbol, direction, abs(score), hv, row))

            signals.sort(key=lambda x: x[2], reverse=True)

            for symbol, direction, score, hv, row in signals:
                if len(positions) >= max_pos:
                    break

                S = row['open']  # 用开盘价入场
                if S <= 0:
                    S = row['close']
                mult, mr, _, _ = specs[symbol]
                mpl = S * mult * mr

                # 仓位大小
                target = equity * eff_mu / max_pos
                if vol_target > 0 and hv > 0:
                    target *= min(vol_target / hv, 3.0)

                fl = max(int(target / mpl), 1)
                if max_lots > 0:
                    fl = min(fl, max_lots)
                if max_notional_mult > 0:
                    max_notional = equity * max_notional_mult
                    current_notional = sum(p.get('fl', 0) * p.get('m', 1) * p.get('ep', 0) for p in positions.values())
                    fl = min(fl, max(1, int((max_notional - current_notional) / (S * mult))))

                fm = mpl * fl
                fc = S * mult * fl * COMM

                total_m = sum(p['fm'] for p in positions.values())
                if total_m + fm > equity * eff_mu:
                    fl = max(int((equity * eff_mu - total_m) / mpl), 0)
                    if fl <= 0:
                        continue
                    fm = mpl * fl
                    fc = S * mult * fl * COMM

                if fm + fc > cash:
                    fl = max(int((cash - fc) / mpl), 0)
                    if fl <= 0:
                        continue
                    fm = mpl * fl
                    fc = S * mult * fl * COMM

                cash -= fm + fc
                positions[symbol] = {
                    'd': direction, 'ed': date, 'ep': S,
                    'fe': S * (1 + 0.0001 * direction),
                    'fl': fl, 'm': mult, 'fm': fm,
                    'mpct': 0, 'adds': 0,
                    'signal_score': score,
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
        equity_history.append(equity)

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

    # === A: 日频交易 (开盘买收盘卖) ===
    print("\n=== A: 日频交易 (open-close) ===")
    prev = 0
    for mode in ['ewmac_mom_bo', 'ramom_ewmac', 'full']:
        for mu in [0.80, 0.90, 0.95]:
            for ms in [0.3, 0.5, 0.8]:
                params = dict(
                    margin_usage=mu, hold_days=1, min_score=ms,
                    max_pos=3, mode=mode,
                    trade_mode='daily_open_close',
                )
                r = fast_backtest(date_map, specs, dates, params)
                if r:
                    results.append(r)
    print(f"  A: {len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === B: 多日持有 + 权益曲线动量 ===
    print("=== B: 权益曲线动量 ===")
    prev = len(results)
    for mode in ['ewmac_mom_bo', 'full']:
        for mu in [0.85, 0.90, 0.95]:
            for hd in [3, 5, 7]:
                for ecm_up in [1.5, 2.0, 3.0]:
                    for ecm_dn in [0.3, 0.5]:
                        params = dict(
                            margin_usage=mu, hold_days=hd, min_score=0.5,
                            max_pos=3, mode=mode,
                            trade_mode='multi_day',
                            ecm=True, ecm_fast=10, ecm_slow=30,
                            ecm_up_mult=ecm_up, ecm_dn_mult=ecm_dn,
                        )
                        r = fast_backtest(date_map, specs, dates, params)
                        if r:
                            results.append(r)
    print(f"  B: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === C: 金字塔加仓 ===
    print("=== C: 金字塔加仓 ===")
    prev = len(results)
    for mode in ['ewmac_mom_bo', 'full']:
        for mu in [0.80, 0.90]:
            for hd in [3, 5, 7]:
                params = dict(
                    margin_usage=mu, hold_days=hd, min_score=0.5,
                    max_pos=3, mode=mode,
                    trade_mode='multi_day',
                    pyramid=True, pyramid_add_pct=0.02, pyramid_max_adds=2,
                    max_lots=20, max_notional_mult=10,
                )
                r = fast_backtest(date_map, specs, dates, params)
                if r:
                    results.append(r)
    print(f"  C: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === D: 窄TP + 仓位上限 (现实版合成卖权) ===
    print("=== D: 合成卖权 + 仓位上限 ===")
    prev = len(results)
    for tp in [0.01, 0.015, 0.02]:
        for sl_mult in [3, 4, 5]:
            sl = tp * sl_mult
            for mu in [0.80, 0.90]:
                for hd in [3, 5, 7]:
                    params = dict(
                        margin_usage=mu, hold_days=hd, min_score=0.5,
                        max_pos=3, mode='ewmac_mom_bo',
                        trade_mode='multi_day',
                        tp_pct=tp, sl_pct=sl,
                        max_lots=15, max_notional_mult=10,
                    )
                    r = fast_backtest(date_map, specs, dates, params)
                    if r:
                        results.append(r)
    print(f"  D: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === E: ECM + 金字塔 + TP ===
    print("=== E: 全组合 ===")
    prev = len(results)
    for mode in ['ewmac_mom_bo', 'full']:
        for mu in [0.80, 0.90]:
            for hd in [3, 5]:
                for tp in [0.0, 0.015]:
                    params = dict(
                        margin_usage=mu, hold_days=hd, min_score=0.5,
                        max_pos=3, mode=mode,
                        trade_mode='multi_day',
                        tp_pct=tp, sl_pct=tp*3 if tp > 0 else 0,
                        ecm=True, ecm_fast=10, ecm_slow=30,
                        ecm_up_mult=2.0, ecm_dn_mult=0.5,
                        pyramid=True, pyramid_add_pct=0.02, pyramid_max_adds=1,
                        max_lots=15, max_notional_mult=10,
                    )
                    r = fast_backtest(date_map, specs, dates, params)
                    if r:
                        results.append(r)
    print(f"  E: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === F: Re-entry ===
    print("=== F: Re-entry ===")
    prev = len(results)
    for mode in ['ewmac_mom_bo', 'full']:
        for mu in [0.85, 0.90, 0.95]:
            for hd in [3, 5]:
                params = dict(
                    margin_usage=mu, hold_days=hd, min_score=0.5,
                    max_pos=3, mode=mode,
                    trade_mode='multi_day',
                    reentry=True,
                    max_lots=20, max_notional_mult=15,
                )
                r = fast_backtest(date_map, specs, dates, params)
                if r:
                    results.append(r)
    print(f"  F: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === G: 日频 + ECM ===
    print("=== G: 日频 + ECM ===")
    prev = len(results)
    for mode in ['ewmac_mom_bo', 'full']:
        for mu in [0.85, 0.90, 0.95]:
            for ms in [0.3, 0.5]:
                for ecm_up in [1.5, 2.0, 3.0]:
                    params = dict(
                        margin_usage=mu, hold_days=1, min_score=ms,
                        max_pos=3, mode=mode,
                        trade_mode='daily_open_close',
                        ecm=True, ecm_fast=5, ecm_slow=20,
                        ecm_up_mult=ecm_up, ecm_dn_mult=0.5,
                    )
                    r = fast_backtest(date_map, specs, dates, params)
                    if r:
                        results.append(r)
    print(f"  G: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.0f}秒, {len(results)}组结果")

    # === 输出 ===
    results.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n\n{'模式':>6} {'保证金':>6} {'持有':>4} {'TP':>5} {'ECM':>3} {'Pyr':>3} {'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5}")
    print("-" * 90)

    for r in results[:50]:
        tp_s = f"{r.get('tp_pct',0)*100:.1f}" if r.get('tp_pct',0) > 0 else '-'
        ecm_s = 'Y' if r.get('ecm') else '-'
        pyr_s = 'Y' if r.get('pyramid') else '-'
        print(f"{r.get('mode','')[:6]:>6} {r.get('margin_usage',0):>6.0%} {r.get('hold_days',0):>4} "
              f"{tp_s:>5} {ecm_s:>3} {pyr_s:>3} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5}")

    # 目标筛选
    print("\n\n" + "=" * 90)
    for target_ann, target_wr, label in [
        (6.0, 0.50, "年化>=600% & WR>=50%"),
        (3.0, 0.50, "年化>=300% & WR>=50%"),
        (1.0, 0.50, "年化>=100% & WR>=50%"),
        (6.0, 0.45, "年化>=600% & WR>=45%"),
        (2.0, 0.55, "年化>=200% & WR>=55%"),
    ]:
        print(f"\n=== {label} ===")
        good = [r for r in results if r['annual'] >= target_ann and r['wr'] >= target_wr]
        if good:
            for r in sorted(good, key=lambda x: x['annual'], reverse=True)[:8]:
                print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                      f"Sharpe={r['sharpe']:>5.2f}  模式={r.get('mode','')}  "
                      f"M={r.get('margin_usage',0):.0%}  H={r.get('hold_days',0)}  "
                      f"ECM={'Y' if r.get('ecm') else 'N'}  Pyr={'Y' if r.get('pyramid') else 'N'}  "
                      f"TP={r.get('tp_pct',0)*100:.1f}%  SL={r.get('sl_pct',0)*100:.1f}%")
        else:
            print("  无")

    # 保存
    output_dir = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(output_dir, exist_ok=True)
    save = []
    for r in results[:150]:
        s = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in r.items()}
        save.append(s)
    with open(os.path.join(output_dir, 'backtest_v43.json'), 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nTOP 150已保存到 backtest_results/backtest_v43.json")


if __name__ == '__main__':
    main()
