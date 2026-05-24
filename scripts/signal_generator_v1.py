#!/usr/bin/env python3
"""
V79: 生产级每日信号生成器
功能:
1. 每日开盘前自动计算所有品种信号
2. 输出推荐开仓品种、方向、得分、止损止盈价
3. 显示当前持仓状态
4. 输出交易日志
5. 支持实时运行和历史回放两种模式
"""
import os, glob, json, numpy as np, pandas as pd, warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

DATA_DIR = 'data/futures_weighted'
CONTRACT_SPECS = 'scripts/contract_specs.py'

# ═══ 策略参数 ═══
STRATEGY_PARAMS = {
    'max_positions': 7,
    'leverage': 5,
    'min_score': 7,
    'hold_days': 1,
    'stop_loss_pct': -1.5,
    'take_profit_pct': 4.0,
    'slippage': 0.001,
    'initial_capital': 500_000,
}

# 信号权重
SCORE_WEIGHTS = {
    'long': [
        ('gap_abs_<-0.5%', 1), ('gap_abs_<-1.0%', 2), ('gap_abs_<-1.5%', 2), ('gap_abs_<-2.0%', 3),
        ('gap_atr_<-1.0', 2), ('gap_atr_<-1.5', 3),
        ('oi_up_price_down', 3), ('oi_down_price_down', 2),
        ('mom5_<-3%', 1), ('mom5_<-5%', 1),
        ('below_ma5', 1),
        ('vol_surge_down', 1),
        ('clv>0.5', 1),
        ('ma20>ma60', 2),
    ],
    'short': [
        ('gap_abs_>0.5%', 1), ('gap_abs_>1.0%', 2), ('gap_abs_>1.5%', 2), ('gap_abs_>2.0%', 3),
        ('gap_atr_>1.0', 2), ('gap_atr_>1.5', 3),
        ('oi_up_price_up', 3), ('oi_down_price_up', 2),
        ('mom5_>3%', 1), ('mom5_>5%', 1),
        ('above_ma5', 1),
        ('vol_surge_up', 1),
        ('clv<-0.5', 1),
        ('ma20<ma60', 2),
    ],
}


def load_data():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cs", CONTRACT_SPECS)
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    all_data = {}
    specs = {}
    for f in sorted(glob.glob(os.path.join(DATA_DIR, '*.csv'))):
        sym = os.path.basename(f).replace('.csv', '')
        try:
            mult, margin, tick, tick_val = cs.get_spec(sym)
            specs[sym] = {'mult': mult, 'margin': margin, 'tick': tick, 'tick_val': tick_val}
        except:
            continue
        df = pd.read_csv(f)
        if len(df) < 100:
            continue
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        df = df.sort_values('trade_date').reset_index(drop=True)
        if df['close'].isna().all() or (df['close'] == 0).any():
            continue
        all_data[sym] = df
    return all_data, specs


def compute_daily_signal(df, sym, specs):
    """计算单品种单日信号"""
    n = len(df)
    c = df['close'].values.astype(float)
    o = df['open'].values.astype(float)
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    v = df['vol'].values.astype(float)
    oi = df['oi'].values.astype(float)

    if n < 60:
        return None

    prev_c = c[-2] if n > 1 else np.nan
    today_o = o[-1]
    today_c = c[-1]
    today_h = h[-1]
    today_l = l[-1]
    today_v = v[-1]
    today_oi = oi[-1]

    if np.isnan(prev_c) or prev_c <= 0 or np.isnan(today_o):
        return None

    # Gap
    gap = (today_o - prev_c) / prev_c * 100

    # ATR
    tr = np.full(n, np.nan)
    tr[1:] = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    atr = np.nanmean(tr[-20:])
    atr_pct = atr / today_c * 100 if today_c > 0 else np.nan

    # MAs
    ma5 = np.mean(c[-5:])
    ma20 = np.mean(c[-20:])
    ma60 = np.mean(c[-60:])

    # Momentum
    mom5 = (today_c - c[-6]) / c[-6] * 100 if n > 5 else 0

    # OI change
    prev_oi = oi[-2] if n > 1 else 0
    oi_chg = (today_oi - prev_oi) / abs(prev_oi) * 100 if abs(prev_oi) > 0 else 0

    # Volume MA
    vol_ma5 = np.mean(v[-5:])

    # CLV
    range_ = today_h - today_l
    clv = (2*today_c - today_h - today_l) / range_ if range_ > 0 else 0

    # Gap/ATR ratio
    gap_atr = gap / atr_pct if atr_pct and atr_pct > 0 else 0

    # ═══ Long Score ═══
    s_l = 0
    factors_l = {}
    s_l += (gap < -0.5) * 1; factors_l['gap<-0.5%'] = (gap < -0.5) * 1
    s_l += (gap < -1.0) * 2; factors_l['gap<-1.0%'] = (gap < -1.0) * 2
    s_l += (gap < -1.5) * 2; factors_l['gap<-1.5%'] = (gap < -1.5) * 2
    s_l += (gap < -2.0) * 3; factors_l['gap<-2.0%'] = (gap < -2.0) * 3
    s_l += (gap_atr < -1.0) * 2; factors_l['gap/atr<-1.0'] = (gap_atr < -1.0) * 2
    s_l += (gap_atr < -1.5) * 3; factors_l['gap/atr<-1.5'] = (gap_atr < -1.5) * 3
    s_l += ((oi_chg > 0) & (today_c < prev_c)) * 3; factors_l['oi↑+price↓'] = ((oi_chg > 0) & (today_c < prev_c)) * 3
    s_l += ((oi_chg < 0) & (today_c < prev_c)) * 2; factors_l['oi↓+price↓'] = ((oi_chg < 0) & (today_c < prev_c)) * 2
    s_l += (mom5 < -3) * 1; factors_l['mom5<-3%'] = (mom5 < -3) * 1
    s_l += (mom5 < -5) * 1; factors_l['mom5<-5%'] = (mom5 < -5) * 1
    s_l += (today_c < ma5) * 1; factors_l['below_ma5'] = (today_c < ma5) * 1
    s_l += ((today_v > vol_ma5 * 1.5) & (today_c < prev_c)) * 1; factors_l['vol_surge+down'] = ((today_v > vol_ma5 * 1.5) & (today_c < prev_c)) * 1
    s_l += (clv > 0.5) * 1; factors_l['clv>0.5'] = (clv > 0.5) * 1
    s_l += (ma20 > ma60) * 2; factors_l['ma20>ma60'] = (ma20 > ma60) * 2

    # ═══ Short Score ═══
    s_s = 0
    factors_s = {}
    s_s += (gap > 0.5) * 1; factors_s['gap>0.5%'] = (gap > 0.5) * 1
    s_s += (gap > 1.0) * 2; factors_s['gap>1.0%'] = (gap > 1.0) * 2
    s_s += (gap > 1.5) * 2; factors_s['gap>1.5%'] = (gap > 1.5) * 2
    s_s += (gap > 2.0) * 3; factors_s['gap>2.0%'] = (gap > 2.0) * 3
    s_s += (gap_atr > 1.0) * 2; factors_s['gap/atr>1.0'] = (gap_atr > 1.0) * 2
    s_s += (gap_atr > 1.5) * 3; factors_s['gap/atr>1.5'] = (gap_atr > 1.5) * 3
    s_s += ((oi_chg > 0) & (today_c > prev_c)) * 3; factors_s['oi↑+price↑'] = ((oi_chg > 0) & (today_c > prev_c)) * 3
    s_s += ((oi_chg < 0) & (today_c > prev_c)) * 2; factors_s['oi↓+price↑'] = ((oi_chg < 0) & (today_c > prev_c)) * 2
    s_s += (mom5 > 3) * 1; factors_s['mom5>3%'] = (mom5 > 3) * 1
    s_s += (mom5 > 5) * 1; factors_s['mom5>5%'] = (mom5 > 5) * 1
    s_s += (today_c > ma5) * 1; factors_s['above_ma5'] = (today_c > ma5) * 1
    s_s += ((today_v > vol_ma5 * 1.5) & (today_c > prev_c)) * 1; factors_s['vol_surge+up'] = ((today_v > vol_ma5 * 1.5) & (today_c > prev_c)) * 1
    s_s += (clv < -0.5) * 1; factors_s['clv<-0.5'] = (clv < -0.5) * 1
    s_s += (ma20 < ma60) * 2; factors_s['ma20<ma60'] = (ma20 < ma60) * 2

    return {
        'sym': sym, 'gap': gap, 'gap_atr': gap_atr, 'atr_pct': atr_pct,
        'oi_chg': oi_chg, 'mom5': mom5, 'clv': clv,
        'open': today_o, 'prev_close': prev_c, 'close': today_c,
        'score_long': s_l, 'score_short': s_s,
        'factors_long': factors_l, 'factors_short': factors_s,
        'trend': 'up' if ma20 > ma60 else 'down',
        'spec': specs.get(sym, {}),
    }


def generate_signals_for_date(all_data, specs, target_date, capital=None, positions=None):
    """生成指定日期的信号"""
    params = STRATEGY_PARAMS
    target_dt = pd.to_datetime(target_date)
    min_sc = params['min_score']
    max_pos = params['max_positions']
    sl = params['stop_loss_pct']
    tp = params['take_profit_pct']

    signals = []
    for sym, df in all_data.items():
        # 取target_date及之前的数据
        mask = df['trade_date'] <= target_dt
        sub = df[mask].tail(61)
        if len(sub) < 61:
            continue
        sig = compute_daily_signal(sub, sym, specs)
        if sig is None:
            continue

        # 检查日期是否匹配
        if sub['trade_date'].iloc[-1] != target_dt:
            continue

        if sig['score_long'] >= min_sc or sig['score_short'] >= min_sc:
            signals.append(sig)

    # 按得分排序
    candidates = []
    for sig in signals:
        best_dir = 'long' if sig['score_long'] >= sig['score_short'] else 'short'
        best_sc = max(sig['score_long'], sig['score_short'])
        if best_sc >= min_sc:
            entry_price = sig['open']
            if best_dir == 'long':
                sl_price = entry_price * (1 + sl / 100)
                tp_price = entry_price * (1 + tp / 100)
            else:
                sl_price = entry_price * (1 - sl / 100)
                tp_price = entry_price * (1 - tp / 100)

            candidates.append({
                'sym': sig['sym'],
                'direction': best_dir,
                'score': best_sc,
                'entry_price': entry_price,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'gap': sig['gap'],
                'gap_atr': sig['gap_atr'],
                'trend': sig['trend'],
                'factors': sig['factors_long'] if best_dir == 'long' else sig['factors_short'],
            })

    # 排除已有持仓
    if positions:
        held = {p['sym'] for p in positions}
        candidates = [c for c in candidates if c['sym'] not in held]

    candidates.sort(key=lambda x: -x['score'])

    return candidates[:max_pos]


def print_daily_report(date, candidates, capital=None, positions=None):
    """打印每日信号报告"""
    params = STRATEGY_PARAMS

    print(f"\n{'═'*70}")
    print(f"  期货隔夜跳空反转策略 — 每日信号报告")
    print(f"  日期: {date}")
    print(f"{'═'*70}")

    if capital:
        print(f"  当前资金: {capital:>14,.0f}")
    if positions is not None:
        print(f"  当前持仓: {len(positions)}/{params['max_positions']}")

    if not candidates:
        print(f"\n  今日无信号 (无品种得分≥{params['min_score']})")
        return

    print(f"\n  {'品种':>6} {'方向':>4} {'得分':>4} {'入场价':>8} {'止损':>8} {'止盈':>8} "
          f"{'Gap%':>6} {'G/A':>5} {'趋势':>4} │ 激活因子")
    print(f"  {'─'*65}")

    for i, c in enumerate(candidates, 1):
        dir_str = '做多' if c['direction'] == 'long' else '做空'
        trend_str = '↑' if c['trend'] == 'up' else '↓'

        # 激活因子
        active = [k for k, v in c['factors'].items() if v > 0]
        factors_str = ', '.join(active[:5])

        print(f"  {i}. {c['sym']:>4} {dir_str:>4} {c['score']:>4} "
              f"{c['entry_price']:>8.1f} {c['sl_price']:>8.1f} {c['tp_price']:>8.1f} "
              f"{c['gap']:>+5.2f}% {c['gap_atr']:>+4.1f} {trend_str:>4} │ {factors_str}")

    # 操作指令
    print(f"\n  操作指令:")
    print(f"  ┌──────────────────────────────────────────────┐")
    for i, c in enumerate(candidates, 1):
        dir_cn = '买入开多' if c['direction'] == 'long' else '卖出开空'
        print(f"  │ {i}. {c['sym']:>4} {dir_cn} @ {c['entry_price']:.1f}    │")
        print(f"  │    止损: {c['sl_price']:.1f}  止盈: {c['tp_price']:.1f}        │")
    print(f"  └──────────────────────────────────────────────┘")

    # 风险提示
    print(f"\n  风险参数: SL={params['stop_loss_pct']}% TP={params['take_profit_pct']}% "
          f"杠杆={params['leverage']}x 仓位={len(candidates)}/{params['max_positions']}")


def run_backtest_with_logging(all_data, specs, start, end):
    """回测模式 — 每日输出信号"""
    params = STRATEGY_PARAMS
    dates = pd.date_range(start=start, end=end, freq='B')
    capital = params['initial_capital']
    positions = []
    trades = []
    eq = []

    print(f"\n回测模式: {start} → {end}")
    print(f"参数: mp={params['max_positions']} lev={params['leverage']}x "
          f"min={params['min_score']} H={params['hold_days']}d "
          f"SL={params['stop_loss_pct']}% TP={params['take_profit_pct']}%")

    for dt in dates:
        dt_str = dt.strftime('%Y-%m-%d')

        # 平仓
        pnl = 0
        keep = []
        for p in positions:
            # 找当日价格
            df = all_data.get(p['sym'])
            if df is None:
                keep.append(p); continue
            idx = df.index[df['trade_date'] == dt]
            if len(idx) == 0:
                keep.append(p); continue
            row = df.loc[idx[0]]
            cur_h, cur_l, cur_c = row['high'], row['low'], row['close']
            if np.isnan(cur_c):
                keep.append(p); continue

            d = (dt - p['ed']).days
            slippage = params['slippage']
            triggered = False
            actual_ret = None
            reason = None

            if p['dir'] == 'long':
                if params['stop_loss_pct']:
                    sp = p['ep'] * (1 + params['stop_loss_pct'] / 100)
                    if cur_l <= sp:
                        fill = sp * (1 - slippage)
                        actual_ret = (fill - p['ep']) / p['ep'] * 100
                        reason = 'SL'; triggered = True
                if not triggered and params['take_profit_pct']:
                    tp_p = p['ep'] * (1 + params['take_profit_pct'] / 100)
                    if cur_h >= tp_p:
                        fill = tp_p * (1 - slippage)
                        actual_ret = (fill - p['ep']) / p['ep'] * 100
                        reason = 'TP'; triggered = True
                if not triggered:
                    actual_ret = (cur_c - p['ep']) / p['ep'] * 100
            else:
                if params['stop_loss_pct']:
                    sp = p['ep'] * (1 - params['stop_loss_pct'] / 100)
                    if cur_h >= sp:
                        fill = sp * (1 + slippage)
                        actual_ret = (p['ep'] - fill) / p['ep'] * 100
                        reason = 'SL'; triggered = True
                if not triggered and params['take_profit_pct']:
                    tp_p = p['ep'] * (1 - params['take_profit_pct'] / 100)
                    if cur_l <= tp_p:
                        fill = tp_p * (1 + slippage)
                        actual_ret = (p['ep'] - fill) / p['ep'] * 100
                        reason = 'TP'; triggered = True
                if not triggered:
                    actual_ret = (p['ep'] - cur_c) / p['ep'] * 100

            if d >= params['hold_days']:
                if not triggered: reason = 'exp'
            else:
                if not triggered:
                    keep.append(p); continue

            if reason:
                pnl += p['not'] * actual_ret / 100
                trades.append({
                    'sym': p['sym'], 'dir': p['dir'], 'ed': p['ed'], 'xd': dt,
                    'ep': p['ep'], 'xp': cur_c, 'r': actual_ret,
                    'pnl': p['not'] * actual_ret / 100, 'sc': p['sc'],
                    'hold': d, 'reason': reason,
                })

        positions = keep
        capital += pnl
        if capital <= 0:
            break

        # 生成信号
        candidates = generate_signals_for_date(all_data, specs, dt_str, capital, positions)
        n_open = params['max_positions'] - len(positions)
        to_open = candidates[:n_open]

        # 打印最新日期的报告
        if dt == dates[-1] or (len(trades) > 0 and (dt - dates[0]).days < 10):
            print_daily_report(dt_str, to_open, capital, positions)

        for c in to_open:
            notional = capital * params['leverage'] / params['max_positions']
            positions.append({
                'sym': c['sym'], 'dir': c['direction'], 'ed': dt,
                'ep': c['entry_price'], 'not': notional, 'sc': c['score'],
            })

        eq.append({'date': dt, 'capital': capital})

    # 打印最新一天
    if len(dates) > 5:
        last_dt = dates[-1].strftime('%Y-%m-%d')
        candidates = generate_signals_for_date(all_data, specs, last_dt, capital, [])
        print_daily_report(last_dt, candidates, capital)

    # 汇总
    if trades:
        tdf = pd.DataFrame(trades)
        wr = (tdf['r'] > 0).mean() * 100
        avg = tdf['r'].mean()
        eq_df = pd.DataFrame(eq)
        mdd = ((eq_df['capital'] - eq_df['capital'].cummax()) / eq_df['capital'].cummax() * 100).min()
        dr = eq_df['capital'].pct_change().dropna()
        sh = dr.mean() / dr.std() * (252**0.5) if dr.std() > 0 else 0

        print(f"\n{'═'*70}")
        print(f"  回测汇总")
        print(f"{'═'*70}")
        print(f"  N={len(trades)} WR={wr:.1f}% Avg={avg:+.3f}% MDD={mdd:.1f}% Sharpe={sh:.2f}")
        print(f"  最终资金: {capital:,.0f}")


def main():
    import sys

    print("期货隔夜跳空反转策略 — 信号生成器 V79")
    print("="*60)

    print("\n加载数据...")
    all_data, specs = load_data()
    print(f"  {len(all_data)}品种")

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = 'latest'

    if mode == 'latest':
        # 最新日期信号
        print("\n查找最新交易日...")
        latest_date = None
        for sym, df in all_data.items():
            d = df['trade_date'].max()
            if latest_date is None or d > latest_date:
                latest_date = d

        if latest_date:
            date_str = latest_date.strftime('%Y-%m-%d')
            print(f"  最新交易日: {date_str}")
            candidates = generate_signals_for_date(all_data, specs, date_str)
            print_daily_report(date_str, candidates)

    elif mode == 'date' and len(sys.argv) > 2:
        # 指定日期信号
        date_str = sys.argv[2]
        candidates = generate_signals_for_date(all_data, specs, date_str)
        print_daily_report(date_str, candidates)

    elif mode == 'backtest':
        # 回测最近N天
        n_days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        # 找最新日期
        latest = max(df['trade_date'].max() for df in all_data.values())
        start = (latest - timedelta(days=n_days*2)).strftime('%Y-%m-%d')
        end = latest.strftime('%Y-%m-%d')
        run_backtest_with_logging(all_data, specs, start, end)

    else:
        print(f"\n用法:")
        print(f"  python {sys.argv[0]} latest        # 最新交易日信号")
        print(f"  python {sys.argv[0]} date 2025-05-20  # 指定日期信号")
        print(f"  python {sys.argv[0]} backtest [N]    # 回测最近N天")

        # 默认: 显示最新信号
        print(f"\n--- 默认: 最新交易日信号 ---")
        latest_date = max(df['trade_date'].max() for df in all_data.values())
        date_str = latest_date.strftime('%Y-%m-%d')
        candidates = generate_signals_for_date(all_data, specs, date_str)
        print_daily_report(date_str, candidates)


if __name__ == '__main__':
    main()
