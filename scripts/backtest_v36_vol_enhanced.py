#!/usr/bin/env python3
"""
策略 v36 — 波动率增强期货策略
核心: 交易期货方向，用期权分析思维辅助决策

期权衍生信号 (全部从期货价格推导):
1. HV期限结构: HV_5/HV_20/HV_60 → 短期vs长期波动率
   - 短期HV < 长期HV → 市场平静，趋势将延续
   - 短期HV > 长期HV → 近期压力，趋势可能反转
2. HV百分位: 当前HV在252天历史中的位置
   - 低百分位(<30%) → 波动率便宜，适合建仓
   - 高百分位(>70%) → 波动率贵，减仓或谨慎
3. 波动率锥: 判断HV是否超出正常范围
4. 理论BS Greeks: 计算理论Delta/Vega → 理解市场敞口
5. 隐含波动率溢价(IVP): 用HV变化率模拟IV vs HV差异

仓位管理:
- 波动率目标: 每个仓位贡献固定比例的波动率
- HV低时加仓 (市场低估风险)
- HV高时减仓 (市场高估风险)
"""

import os, sys, time, numpy as np, pandas as pd
from collections import defaultdict
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from contract_specs import get_spec


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

        # === 基础指标 ===
        df['return'] = df['close'].pct_change()
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['mom_5'] = df['close'].pct_change(5)
        df['mom_10'] = df['close'].pct_change(10)
        df['mom_20'] = df['close'].pct_change(20)
        df['trend'] = np.where(df['ma20'] > df['ma60'], 1, -1)

        # === 波动率分析 (期权视角) ===
        # HV期限结构
        df['hv_5'] = df['return'].rolling(5).std() * np.sqrt(252)
        df['hv_10'] = df['return'].rolling(10).std() * np.sqrt(252)
        df['hv_20'] = df['return'].rolling(20).std() * np.sqrt(252)
        df['hv_60'] = df['return'].rolling(60).std() * np.sqrt(252)
        df['hv_120'] = df['return'].rolling(120).std() * np.sqrt(252)

        # HV百分位 (252天窗口)
        df['hv_pct'] = df['hv_20'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 10 else 0.5, raw=False
        )

        # HV期限结构比率
        df['hv_ratio'] = df['hv_5'] / df['hv_20'].replace(0, np.nan)  # 短/长
        df['hv_slope'] = df['hv_5'] / df['hv_60'].replace(0, np.nan)  # 短/长

        # HV变化率 (模拟IV-HV差异)
        df['hv_change'] = df['hv_20'].pct_change(5)
        df['hv_accel'] = df['hv_change'].diff(5)

        # ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close']

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss_s = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss_s))

        # ADX
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff().abs()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        atr_s = df['atr']
        plus_di = 100 * plus_dm.rolling(14).mean() / (atr_s + 0.001)
        minus_di = 100 * minus_dm.rolling(14).mean() / (atr_s + 0.001)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.001)
        df['adx'] = dx.rolling(14).mean()

        # OI分析
        if 'oi' in df.columns:
            df['oi_ma5'] = df['oi'].rolling(5).mean()
            df['oi_rising'] = df['oi'] > df['oi'].shift(5)

        # 成交量
        df['vol_ma20'] = df['vol'].rolling(20).mean()
        df['vol_ratio'] = df['vol'] / df['vol_ma20']

        # 波动率regime
        df['vol_regime'] = 'normal'
        df.loc[df['hv_pct'] < 0.25, 'vol_regime'] = 'low'
        df.loc[df['hv_pct'] > 0.75, 'vol_regime'] = 'high'

        df = df.dropna(subset=['ma20', 'ma60', 'hv_20', 'mom_10', 'rsi', 'adx', 'hv_pct'])
        if len(df) > 100:
            data[symbol] = df
    return data


def run_backtest(data, start_date, end_date, params):
    max_pos = params.get('max_pos', 3)
    hold_days = params.get('hold_days', 5)
    margin_usage = params.get('margin_usage', 0.85)
    min_mom = params.get('min_mom', 0.02)
    # 波动率过滤参数
    hv_pct_low = params.get('hv_pct_low', 0.0)    # HV百分位下限
    hv_pct_high = params.get('hv_pct_high', 1.0)   # HV百分位上限
    hv_slope_max = params.get('hv_slope_max', 99)   # 期限结构斜率上限
    vol_target = params.get('vol_target', 0)         # 波动率目标(0=不用)
    use_oi = params.get('use_oi', False)
    require_align = params.get('require_align', False)
    dd_reduce = params.get('dd_reduce', False)       # 回撤时减仓
    r = 0.02
    comm_rate = 0.00015

    date_map = defaultdict(dict)
    for symbol, df in data.items():
        mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
        for _, row in df[mask].iterrows():
            date_map[row['trade_date']][symbol] = row
    dates = sorted(date_map.keys())

    equity = 500000.0
    cash = 500000.0
    positions = {}
    closed_pnls = []
    equity_curve = []
    peak_equity = 500000.0

    for date in dates:
        day_data = date_map[date]

        # === 退出 ===
        for symbol in list(positions.keys()):
            pos = positions[symbol]
            if symbol not in day_data:
                continue
            row = day_data[symbol]
            price = row['close']
            hd = (date - pos['entry_date']).days

            if hd >= hold_days:
                pnl = (price - pos['entry_price']) * pos['dir'] * pos['mult'] * pos['lots']
                comm = price * pos['mult'] * pos['lots'] * comm_rate
                cash += pos['margin'] + pnl - comm
                closed_pnls.append(pnl - comm)
                del positions[symbol]

        # === 入场 ===
        if len(positions) < max_pos:
            # 动态保证金 (回撤时减仓)
            mu = margin_usage
            if dd_reduce:
                dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
                if dd > 0.15:
                    mu = margin_usage * max(1 - dd, 0.3)

            signals = []
            for symbol, row in day_data.items():
                if symbol in positions:
                    continue
                if row.get('atr_pct', 0.1) > 0.05:
                    continue

                # 基础信号
                trend = row.get('trend', 0)
                mom_10 = row.get('mom_10', 0)
                mom_5 = row.get('mom_5', 0)
                mom_20 = row.get('mom_20', 0)
                adx = row.get('adx', 0)
                rsi = row.get('rsi', 50)
                hv = row.get('hv_20', 0)
                if hv < 0.08 or hv > 0.70:
                    continue

                direction = 0
                if trend == 1 and mom_10 > min_mom and rsi < 70:
                    direction = 1
                elif trend == -1 and mom_10 < -min_mom and rsi > 30:
                    direction = -1
                if direction == 0:
                    continue

                # 多时间框架对齐
                if require_align:
                    if direction == 1 and not (mom_5 > 0 and mom_10 > 0):
                        continue
                    elif direction == -1 and not (mom_5 < 0 and mom_10 < 0):
                        continue

                # === 期权衍生信号过滤 ===
                hv_pct = row.get('hv_pct', 0.5)
                if hv_pct < hv_pct_low or hv_pct > hv_pct_high:
                    continue

                hv_slope = row.get('hv_slope', 1.0)
                if hv_slope > hv_slope_max:
                    continue

                # OI确认
                if use_oi and not row.get('oi_rising', True):
                    continue

                # 综合评分 (动量 + 波动率regime + 趋势强度)
                score = abs(mom_10) * 100
                score += (adx - 20) * 0.3
                # HV百分位越低(波动率便宜)，越值得建仓
                if hv_pct < 0.3:
                    score += 1.0
                # 期限结构正常(短期<长期)→趋势延续
                if hv_slope < 1.0:
                    score += 0.5

                signals.append((symbol, direction, score))

            signals.sort(key=lambda x: x[2], reverse=True)

            for symbol, direction, score in signals:
                if len(positions) >= max_pos:
                    break

                row = day_data[symbol]
                S = row['close']
                mult, mr, _, _ = get_spec(symbol)
                mpl = S * mult * mr

                # 波动率目标仓位
                if vol_target > 0 and row.get('hv_20', 0) > 0:
                    # 仓位 = (目标波动率 * 权益) / (HV * 价格 * 乘数)
                    target_lots = (vol_target * equity) / (row['hv_20'] * S * mult * np.sqrt(252))
                    lots = max(int(target_lots), 1)
                else:
                    target_n = equity * (mu / max_pos)
                    lots = max(int(target_n / (S * mult)), 1)

                margin = mpl * lots
                comm = S * mult * lots * comm_rate

                total_m = sum(p['margin'] for p in positions.values()) + margin
                if total_m > equity * mu:
                    lots = max(int((equity * mu - sum(p['margin'] for p in positions.values())) / mpl), 0)
                    if lots <= 0:
                        continue
                    margin = mpl * lots
                    comm = S * mult * lots * comm_rate

                if margin + comm > cash:
                    lots = max(int((cash - comm) / mpl), 0)
                    if lots <= 0:
                        continue
                    margin = mpl * lots
                    comm = S * mult * lots * comm_rate

                cash -= margin + comm
                positions[symbol] = {
                    'dir': direction, 'mult': mult,
                    'entry_price': S * (1 + 0.0001 * direction),
                    'entry_date': date, 'lots': lots,
                    'margin': margin,
                }

        # === 权益 ===
        unrealized = 0
        for symbol, pos in positions.items():
            if symbol in day_data:
                price = day_data[symbol]['close']
                unrealized += (price - pos['entry_price']) * pos['dir'] * pos['mult'] * pos['lots']
        equity = cash + unrealized
        peak_equity = max(peak_equity, equity)
        equity_curve.append((date, equity))
        if equity < 5000:
            break

    if not equity_curve or equity_curve[-1][1] <= 0:
        return None

    final = equity_curve[-1][1]
    total_ret = (final - 500000) / 500000
    days = (equity_curve[-1][0] - equity_curve[0][0]).days
    years = max(days / 365, 0.001)
    ann = float((1 + total_ret) ** (1 / years) - 1)

    eq = pd.DataFrame(equity_curve, columns=['date', 'equity'])
    eq['cummax'] = eq['equity'].cummax()
    eq['dd'] = (eq['equity'] - eq['cummax']) / eq['cummax']
    mdd = float(eq['dd'].min())

    pnls = np.array(closed_pnls)
    wr = float((pnls > 0).mean()) if len(pnls) > 0 else 0
    avg_w = float(pnls[pnls > 0].mean()) if (pnls > 0).any() else 0
    avg_l = float(abs(pnls[pnls <= 0].mean())) if (pnls <= 0).any() else 1
    pf = avg_w * (pnls > 0).sum() / (avg_l * (pnls <= 0).sum()) if (pnls <= 0).sum() > 0 and avg_l > 0 else 0

    return {
        'annual': ann, 'wr': wr, 'mdd': mdd, 'pf': pf,
        'trades': len(pnls), 'final': final, 'total_ret': float(total_ret),
        **{k: v for k, v in params.items()},
    }


def main():
    data_dir = os.path.expanduser("~/home/futures_platform/data/futures_weighted")
    print("加载数据...")
    data = load_data(data_dir)
    print(f"加载了 {len(data)} 个品种\n")

    start_date = pd.Timestamp('2018-01-01')
    end_date = pd.Timestamp('2026-05-08')
    results = []

    # === 扫描1: 保证金 + 持有天数 + 动量 ===
    print("=== 扫描1: 基础参数 ===")
    for mu in [0.80, 0.90, 0.95]:
        for hd in [3, 5, 7]:
            for mom in [0.02, 0.03, 0.05]:
                params = dict(margin_usage=mu, hold_days=hd, min_mom=mom,
                             max_pos=3, vol_target=0, dd_reduce=False)
                r = run_backtest(data, start_date, end_date, params)
                if r: results.append(r)

    # === 扫描2: HV百分位过滤 ===
    print("=== 扫描2: HV百分位过滤 ===")
    for mu in [0.85, 0.90, 0.95]:
        for hd in [3, 5, 7]:
            for hv_lo, hv_hi in [(0.0, 0.6), (0.0, 0.7), (0.1, 0.5), (0.15, 0.5)]:
                for mom in [0.02, 0.03]:
                    params = dict(margin_usage=mu, hold_days=hd, min_mom=mom,
                                 max_pos=3, vol_target=0, dd_reduce=False,
                                 hv_pct_low=hv_lo, hv_pct_high=hv_hi)
                    r = run_backtest(data, start_date, end_date, params)
                    if r: results.append(r)

    # === 扫描3: HV期限结构过滤 ===
    print("=== 扫描3: HV期限结构过滤 ===")
    for mu in [0.85, 0.90, 0.95]:
        for hd in [3, 5]:
            for slope_max in [1.2, 1.5, 2.0, 99]:
                for mom in [0.02, 0.03]:
                    params = dict(margin_usage=mu, hold_days=hd, min_mom=mom,
                                 max_pos=3, vol_target=0, dd_reduce=False,
                                 hv_slope_max=slope_max)
                    r = run_backtest(data, start_date, end_date, params)
                    if r: results.append(r)

    # === 扫描4: 波动率目标 + 回撤减仓 ===
    print("=== 扫描4: 波动率目标 + 回撤减仓 ===")
    for mu in [0.85, 0.90, 0.95]:
        for hd in [3, 5, 7]:
            for vt in [0.01, 0.02, 0.03]:
                for dd in [True, False]:
                    for mom in [0.02, 0.03]:
                        params = dict(margin_usage=mu, hold_days=hd, min_mom=mom,
                                     max_pos=3, vol_target=vt, dd_reduce=dd)
                        r = run_backtest(data, start_date, end_date, params)
                        if r: results.append(r)

    # === 扫描5: OI + 对齐 ===
    print("=== 扫描5: OI + 对齐 ===")
    for mu in [0.90, 0.95]:
        for hd in [3, 5]:
            for oi in [True, False]:
                for align in [True, False]:
                    params = dict(margin_usage=mu, hold_days=hd, min_mom=0.02,
                                 max_pos=3, vol_target=0, dd_reduce=False,
                                 use_oi=oi, require_align=align)
                    r = run_backtest(data, start_date, end_date, params)
                    if r: results.append(r)

    # === 输出 ===
    results.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n\n{'保证金':>6} {'HD':>3} {'Mom':>5} {'HV%Lo':>5} {'HV%Hi':>5} {'斜率':>4} {'VolT':>5} {'DD':>3} {'OI':>3} {'对齐':>3} {'年化':>8} {'WR':>6} {'PF':>6} {'MDD':>8} {'交易':>5}")
    print("-" * 120)

    for r in results[:60]:
        dd_str = 'Y' if r.get('dd_reduce') else '-'
        oi_str = 'Y' if r.get('use_oi') else '-'
        al_str = 'Y' if r.get('require_align') else '-'
        vt_str = f"{r.get('vol_target',0):.0%}" if r.get('vol_target', 0) > 0 else '-'
        slope_str = f"{r.get('hv_slope_max',99):.1f}" if r.get('hv_slope_max', 99) < 99 else '-'
        print(f"{r.get('margin_usage',0):>6.0%} {r.get('hold_days',0):>3} {r.get('min_mom',0):>5.0%} "
              f"{r.get('hv_pct_low',0):>5.1f} {r.get('hv_pct_high',1.0):>5.1f} {slope_str:>4} "
              f"{vt_str:>5} {dd_str:>3} {oi_str:>3} {al_str:>3} "
              f"{r['annual']:>8.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['trades']:>5}")

    # 筛选
    print(f"\n\n=== 年化>=100% ===")
    good = [r for r in results if r['annual'] >= 1.0]
    if good:
        for r in sorted(good, key=lambda x: x['wr'], reverse=True)[:15]:
            print(f"  年化={r['annual']:.1%}  WR={r['wr']:.1%}  PF={r['pf']:.2f}  MDD={r['mdd']:.1%}  "
                  f"交易={r['trades']}  保证金={r.get('margin_usage',0):.0%}  HD={r.get('hold_days',0)}  "
                  f"Mom={r.get('min_mom',0):.0%}  HV=[{r.get('hv_pct_low',0):.1f},{r.get('hv_pct_high',1):.1f}]  "
                  f"VolT={r.get('vol_target',0):.0%}  DD={'Y' if r.get('dd_reduce') else '-'}")
    else:
        print("无")

    print(f"\n\n=== WR>=50% ===")
    wr50 = [r for r in results if r['wr'] >= 0.50]
    if wr50:
        for r in sorted(wr50, key=lambda x: x['annual'], reverse=True)[:15]:
            print(f"  年化={r['annual']:.1%}  WR={r['wr']:.1%}  PF={r['pf']:.2f}  MDD={r['mdd']:.1%}  "
                  f"交易={r['trades']}  保证金={r.get('margin_usage',0):.0%}  HD={r.get('hold_days',0)}  "
                  f"Mom={r.get('min_mom',0):.0%}  HV=[{r.get('hv_pct_low',0):.1f},{r.get('hv_pct_high',1):.1f}]  "
                  f"VolT={r.get('vol_target',0):.0%}  DD={'Y' if r.get('dd_reduce') else '-'}")
    else:
        print("无")

    print(f"\n\n=== WR>=45% AND 年化>=50% ===")
    balanced = [r for r in results if r['wr'] >= 0.45 and r['annual'] >= 0.5]
    if balanced:
        for r in sorted(balanced, key=lambda x: x['annual'], reverse=True)[:15]:
            print(f"  年化={r['annual']:.1%}  WR={r['wr']:.1%}  PF={r['pf']:.2f}  MDD={r['mdd']:.1%}  "
                  f"交易={r['trades']}  保证金={r.get('margin_usage',0):.0%}  HD={r.get('hold_days',0)}  "
                  f"Mom={r.get('min_mom',0):.0%}  HV=[{r.get('hv_pct_low',0):.1f},{r.get('hv_pct_high',1):.1f}]  "
                  f"VolT={r.get('vol_target',0):.0%}  DD={'Y' if r.get('dd_reduce') else '-'}")
    else:
        print("无")


if __name__ == '__main__':
    main()
