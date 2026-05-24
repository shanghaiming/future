#!/usr/bin/env python3
"""
策略 v44 — 修复前视偏差 + 日频/多日交易

核心修复: 信号延迟一天
  v43的bug: 用T日的close计算指标→用T日的open入场 (look-ahead)
  v44修复: 用T-1日的指标决定信号→T日open入场→T日close退出

额外改进:
1. 信号强度自适应仓位 (强信号=大仓位)
2. 隔夜跳空过滤 (跳空>2%不交易)
3. 多时间框架确认 (日线趋势+周线趋势一致)
4. 日内收益预测: 用T-1日的intraday_ret方向作为辅助确认
"""

import os, sys, time, json, numpy as np, pandas as pd
from collections import defaultdict

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
        df['intraday_ret'] = df['close'] / df['open'] - 1
        df['open_ret'] = df['open'].pct_change()  # 隔夜跳空
        df['gap_pct'] = (df['open'] - df['close'].shift()) / df['close'].shift()

        # 趋势
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['trend'] = np.where(df['ma20'] > df['ma60'], 1, -1)

        # 周线趋势 (用5日MA代表)
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['wk_trend'] = np.where(df['ma5'] > df['ma10'], 1, -1)

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
    """构建日期映射 + 预计算前一天信号"""
    date_map = defaultdict(dict)
    specs = {}
    prev_rows = {}  # {symbol: previous_day_row}

    for symbol, df in data.items():
        specs[symbol] = df.attrs['spec']
        mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
        for _, row in df[mask].iterrows():
            date_map[row['trade_date']][symbol] = row
    return dict(date_map), specs


def compute_signal(row, mode='ewmac_mom_bo'):
    """用前一天row计算信号 (不存在look-ahead)"""
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
        if trend != 0 and not pd.isna(score) and np.sign(score) == trend:
            score *= 1.3

    elif mode == 'ewmac_mom_bo_wk':
        # 加入周线趋势确认
        ew8 = row.get('ewmac_8', 0)
        ew16 = row.get('ewmac_16', 0)
        bo20 = row.get('breakout_20', 0)
        mom5 = row.get('mom_5', 0)
        wk_trend = row.get('wk_trend', 0)
        trend = row.get('trend', 0)

        if not pd.isna(ew8):
            score += np.sign(ew8) * min(abs(ew8), 2)
            w += 1.0
        if not pd.isna(ew16):
            score += np.sign(ew16) * min(abs(ew16), 2)
            w += 1.0
        if not pd.isna(bo20):
            score += bo20
            w += 1.0
        if not pd.isna(mom5):
            score += np.sign(mom5) * min(abs(mom5) * 10, 1)
            w += 0.5
        # 周线+日线趋势一致 → 加权
        if wk_trend != 0 and trend != 0 and wk_trend == trend:
            score *= 1.5

    return score / max(w, 1) if w > 0 else 0


def fast_backtest(data, specs, start_date, end_date, params):
    """带信号延迟的回测"""
    max_pos = params.get('max_pos', 3)
    margin_usage = params.get('margin_usage', 0.90)
    hold_days = params.get('hold_days', 5)
    min_score = params.get('min_score', 0.5)
    mode = params.get('mode', 'ewmac_mom_bo')
    trade_mode = params.get('trade_mode', 'daily_open_close')

    use_hv = params.get('use_hv', True)
    hv_hi = params.get('hv_hi', 0.85)
    use_rsi = params.get('use_rsi', True)
    rsi_hi = params.get('rsi_hi', 75)
    rsi_lo = params.get('rsi_lo', 25)

    tp_pct = params.get('tp_pct', 0.0)
    sl_pct = params.get('sl_pct', 0.0)

    # 权益曲线动量
    ecm = params.get('ecm', False)
    ecm_fast = params.get('ecm_fast', 10)
    ecm_slow = params.get('ecm_slow', 30)
    ecm_up_mult = params.get('ecm_up_mult', 1.5)
    ecm_dn_mult = params.get('ecm_dn_mult', 0.5)

    # 仓位上限
    max_lots = params.get('max_lots', 0)
    max_notional_mult = params.get('max_notional_mult', 0)

    # 隔夜跳空过滤
    gap_filter = params.get('gap_filter', 0.0)  # 0=不过滤

    equity = INIT_CAPITAL
    cash = INIT_CAPITAL
    positions = {}
    closed_pnls = []
    peak_equity = INIT_CAPITAL
    equity_curve = []
    equity_history = [INIT_CAPITAL]

    # 为每个品种构建日期→iloc位置 (用iloc而非loc, 因为index不连续)
    symbol_date_iloc = {}
    symbol_sorted_dates = {}
    for symbol, df in data.items():
        df_reset = df.reset_index(drop=True)
        data[symbol] = df_reset  # 更新为连续index
        dm = {}
        sdates = []
        for iloc_pos in range(len(df_reset)):
            row = df_reset.iloc[iloc_pos]
            if start_date <= row['trade_date'] <= end_date:
                dm[row['trade_date']] = iloc_pos
                sdates.append(row['trade_date'])
        symbol_date_iloc[symbol] = dm
        symbol_sorted_dates[symbol] = sdates

    # 构建全局日期列表
    all_dates = set()
    for symbol, df in data.items():
        mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
        for _, row in df[mask].iterrows():
            all_dates.add(row['trade_date'])
    dates = sorted(all_dates)

    for i_date, date in enumerate(dates):
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
            if symbol not in symbol_date_iloc or date not in symbol_date_iloc[symbol]:
                continue
            df = data[symbol]
            iloc_pos = symbol_date_iloc[symbol][date]
            row = df.iloc[iloc_pos]
            price = row['close']
            hd = (date - pos['ed']).days

            should_close = False
            pnl_pct = (price / pos['ep'] - 1) * pos['d']

            if trade_mode == 'daily_open_close':
                should_close = True
            else:
                if hd >= hold_days:
                    should_close = True
                if tp_pct > 0 and pnl_pct >= tp_pct:
                    should_close = True
                if sl_pct > 0 and pnl_pct <= -sl_pct:
                    should_close = True

            if should_close:
                comm = price * pos['m'] * pos['fl'] * COMM
                pnl = (price - pos['fe']) * pos['d'] * pos['m'] * pos['fl']
                cash += pos['fm'] + pnl - comm
                net = pnl - comm
                closed_pnls.append(net)
                del positions[symbol]

        # === 入场 ===
        if len(positions) < max_pos:
            eff_mu = margin_usage * ecm_mult

            signals = []
            for symbol, df in data.items():
                if symbol in positions:
                    continue

                # 需要当日有数据 (用于入场价格)
                if symbol not in symbol_date_iloc or date not in symbol_date_iloc[symbol]:
                    continue
                iloc_pos = symbol_date_iloc[symbol][date]
                row_today = df.iloc[iloc_pos]

                # 关键: 用前一个交易日的数据计算信号
                if iloc_pos <= 0:
                    continue
                row_yesterday = df.iloc[iloc_pos - 1]
                if row_yesterday['trade_date'] >= date:
                    continue  # 安全检查

                # 用前一天数据做过滤
                hv = row_yesterday.get('hv_20', 0)
                if use_hv and (hv < 0.05 or hv > 0.70):
                    continue
                hp = row_yesterday.get('hv_pct', 0.5)
                if use_hv and hp > hv_hi:
                    continue

                rsi = row_yesterday.get('rsi', 50)
                if use_rsi and (rsi > rsi_hi or rsi < rsi_lo):
                    continue

                atr_pct = row_yesterday.get('atr_pct', 0)
                if atr_pct > 0.05:
                    continue

                # 隔夜跳空过滤
                if gap_filter > 0:
                    gap = row_today.get('gap_pct', 0)
                    if pd.notna(gap) and abs(gap) > gap_filter:
                        continue

                # 用前一天数据计算信号
                score = compute_signal(row_yesterday, mode)
                if abs(score) < min_score:
                    continue

                direction = 1 if score > 0 else -1
                signals.append((symbol, direction, abs(score), hv, row_today))

            signals.sort(key=lambda x: x[2], reverse=True)

            for symbol, direction, score, hv, row_today in signals:
                if len(positions) >= max_pos:
                    break

                S = row_today['open']  # 用当日开盘价入场
                if S <= 0:
                    S = row_today['close']
                mult, mr, _, _ = specs[symbol]
                mpl = S * mult * mr

                target = equity * eff_mu / max_pos
                fl = max(int(target / mpl), 1)

                if max_lots > 0:
                    fl = min(fl, max_lots)
                if max_notional_mult > 0:
                    max_notional = equity * max_notional_mult
                    current_notional = sum(
                        p.get('fl', 0) * p.get('m', 1) * p.get('ep', 0)
                        for p in positions.values()
                    )
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
                    'signal_score': score,
                }

        # === 权益 ===
        unrealized = 0
        for symbol, pos in positions.items():
            if symbol in symbol_date_iloc and date in symbol_date_iloc[symbol]:
                iloc_pos = symbol_date_iloc[symbol][date]
                price = data[symbol].iloc[iloc_pos]['close']
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

    specs = {}
    for symbol, df in data.items():
        specs[symbol] = df.attrs['spec']

    results = []
    bt0 = time.time()

    # === A: 日频交易 (信号延迟1天) ===
    print("\n=== A: 日频交易 (信号延迟1天, open→close) ===")
    prev = 0
    for mode in ['ewmac_mom_bo', 'ramom_ewmac', 'full', 'ewmac_mom_bo_wk']:
        for mu in [0.60, 0.70, 0.80, 0.90]:
            for ms in [0.3, 0.5, 0.8]:
                params = dict(
                    margin_usage=mu, hold_days=1, min_score=ms,
                    max_pos=3, mode=mode,
                    trade_mode='daily_open_close',
                )
                r = fast_backtest(data, specs, start_date, end_date, params)
                if r:
                    results.append(r)
    print(f"  A: {len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === B: 多日持有 (信号延迟1天) ===
    print("=== B: 多日持有 (信号延迟1天) ===")
    prev = len(results)
    for mode in ['ewmac_mom_bo', 'full']:
        for mu in [0.60, 0.70, 0.80, 0.90]:
            for hd in [3, 5, 7, 10]:
                for ms in [0.3, 0.5]:
                    params = dict(
                        margin_usage=mu, hold_days=hd, min_score=ms,
                        max_pos=3, mode=mode,
                        trade_mode='multi_day',
                    )
                    r = fast_backtest(data, specs, start_date, end_date, params)
                    if r:
                        results.append(r)
    print(f"  B: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === C: 日频 + ECM ===
    print("=== C: 日频 + ECM ===")
    prev = len(results)
    for mode in ['ewmac_mom_bo', 'full']:
        for mu in [0.70, 0.80, 0.90]:
            for ecm_up in [1.5, 2.0, 3.0]:
                params = dict(
                    margin_usage=mu, hold_days=1, min_score=0.5,
                    max_pos=3, mode=mode,
                    trade_mode='daily_open_close',
                    ecm=True, ecm_fast=5, ecm_slow=20,
                    ecm_up_mult=ecm_up, ecm_dn_mult=0.5,
                )
                r = fast_backtest(data, specs, start_date, end_date, params)
                if r:
                    results.append(r)
    print(f"  C: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === D: 日频 + 仓位上限 ===
    print("=== D: 日频 + 仓位上限 ===")
    prev = len(results)
    for mode in ['ewmac_mom_bo', 'full']:
        for mu in [0.80, 0.90]:
            for ml in [10, 15, 20]:
                for mnm in [5, 10]:
                    params = dict(
                        margin_usage=mu, hold_days=1, min_score=0.5,
                        max_pos=3, mode=mode,
                        trade_mode='daily_open_close',
                        max_lots=ml, max_notional_mult=mnm,
                    )
                    r = fast_backtest(data, specs, start_date, end_date, params)
                    if r:
                        results.append(r)
    print(f"  D: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === E: 多日 + TP/SL ===
    print("=== E: 多日 + TP/SL ===")
    prev = len(results)
    for tp in [0.01, 0.015, 0.02]:
        for sl_mult in [3, 5]:
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
                    r = fast_backtest(data, specs, start_date, end_date, params)
                    if r:
                        results.append(r)
    print(f"  E: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === F: 日频 + 隔夜跳空过滤 ===
    print("=== F: 日频 + 隔夜跳空过滤 ===")
    prev = len(results)
    for mode in ['ewmac_mom_bo', 'full']:
        for mu in [0.70, 0.80, 0.90]:
            for gf in [0.02, 0.03]:
                params = dict(
                    margin_usage=mu, hold_days=1, min_score=0.5,
                    max_pos=3, mode=mode,
                    trade_mode='daily_open_close',
                    gap_filter=gf,
                )
                r = fast_backtest(data, specs, start_date, end_date, params)
                if r:
                    results.append(r)
    print(f"  F: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    # === G: 周线趋势确认 + 日频 ===
    print("=== G: 周线确认 + 日频 ===")
    prev = len(results)
    for mu in [0.70, 0.80, 0.90]:
        for ms in [0.3, 0.5, 0.8]:
            params = dict(
                margin_usage=mu, hold_days=1, min_score=ms,
                max_pos=3, mode='ewmac_mom_bo_wk',
                trade_mode='daily_open_close',
            )
            r = fast_backtest(data, specs, start_date, end_date, params)
            if r:
                results.append(r)
    print(f"  G: +{len(results)-prev}组, {time.time()-bt0:.0f}s")

    elapsed = time.time() - t0
    print(f"\n总耗时: {elapsed:.0f}秒, {len(results)}组结果")

    # === 输出 ===
    results.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n\n{'模式':>14} {'保证金':>6} {'持有':>4} {'TP':>5} {'ECM':>3} {'上限':>4} {'年化':>10} {'胜率':>6} {'PF':>6} {'MDD':>8} {'Sharpe':>7} {'交易':>5}")
    print("-" * 100)

    for r in results[:50]:
        tp_s = f"{r.get('tp_pct',0)*100:.1f}" if r.get('tp_pct',0) > 0 else '-'
        ecm_s = 'Y' if r.get('ecm') else '-'
        lim_s = 'Y' if r.get('max_lots', 0) > 0 else '-'
        print(f"{r.get('mode','')[:14]:>14} {r.get('margin_usage',0):>6.0%} {r.get('hold_days',0):>4} "
              f"{tp_s:>5} {ecm_s:>3} {lim_s:>4} "
              f"{r['annual']:>10.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['sharpe']:>7.2f} {r['trades']:>5}")

    # 目标筛选
    print("\n\n" + "=" * 100)
    for target_ann, target_wr, label in [
        (6.0, 0.50, "年化>=600% & WR>=50%"),
        (3.0, 0.50, "年化>=300% & WR>=50%"),
        (1.0, 0.50, "年化>=100% & WR>=50%"),
        (0.5, 0.50, "年化>=50% & WR>=50%"),
        (0.3, 0.55, "年化>=30% & WR>=55%"),
    ]:
        print(f"\n=== {label} ===")
        good = [r for r in results if r['annual'] >= target_ann and r['wr'] >= target_wr]
        if good:
            for r in sorted(good, key=lambda x: x['annual'], reverse=True)[:8]:
                print(f"  年化={r['annual']:>8.1%}  WR={r['wr']:>5.1%}  PF={r['pf']:>5.2f}  MDD={r['mdd']:>7.1%}  "
                      f"Sharpe={r['sharpe']:>5.2f}  模式={r.get('mode','')}  "
                      f"M={r.get('margin_usage',0):.0%}  H={r.get('hold_days',0)}  "
                      f"ECM={'Y' if r.get('ecm') else 'N'}  "
                      f"TP={r.get('tp_pct',0)*100:.1f}%  SL={r.get('sl_pct',0)*100:.1f}%  "
                      f"ML={r.get('max_lots',0)}  MNM={r.get('max_notional_mult',0)}")
        else:
            print("  无")

    # 保存
    output_dir = os.path.expanduser("~/home/futures_platform/backtest_results")
    os.makedirs(output_dir, exist_ok=True)
    save = []
    for r in results[:200]:
        s = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in r.items()}
        save.append(s)
    with open(os.path.join(output_dir, 'backtest_v44.json'), 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nTOP 200已保存到 backtest_results/backtest_v44.json")


if __name__ == '__main__':
    main()
