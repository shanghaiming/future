#!/usr/bin/env python3
"""
策略 v32 — Kelly仓位 + OTM期权 + 多因子对齐
核心发现:
  - 买ATM期权: 421%年化, 39%WR, PF=3.0 (v26)
  - 卖OTM期权: 81.6%年化, 77.5%WR, PF=0.99 (v27)
  - 混合score>70: 184.6%年化, 61.6%WR, PF=1.21 (v29)

突破口: 提高买期权的WR从39%到45-50% → Kelly=30%+ → 复利爆炸

方法:
1. OTM 1-3%: 更便宜, 盈亏比更高 → PF从3.0提到4.0+
2. 多时间框架对齐: mom_5/mom_10/mom_20同向 → 过滤假信号提WR
3. Kelly仓位: 根据信号强度动态调仓 → 15-25%权益/笔
4. 10天持有: 更多时间兑现 → 减少时间衰减的影响
5. 提前止盈: 期权价值翻倍时平仓 → 锁定利润
"""

import os, sys, time, numpy as np, pandas as pd
from collections import defaultdict
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from contract_specs import get_spec


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
        if len(df) < 100:
            continue
        df['return'] = df['close'].pct_change()
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['hv_20'] = df['return'].rolling(20).std() * np.sqrt(252)
        df['hv_60'] = df['return'].rolling(60).std() * np.sqrt(252)

        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close']

        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss_s = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss_s))

        df['mom_5'] = df['close'].pct_change(5)
        df['mom_10'] = df['close'].pct_change(10)
        df['mom_20'] = df['close'].pct_change(20)
        df['trend'] = np.where(df['ma20'] > df['ma60'], 1, -1)

        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff().abs()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        atr_s = df['atr']
        plus_di = 100 * plus_dm.rolling(14).mean() / (atr_s + 0.001)
        minus_di = 100 * minus_dm.rolling(14).mean() / (atr_s + 0.001)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.001)
        df['adx'] = dx.rolling(14).mean()

        df['vol_ma20'] = df['vol'].rolling(20).mean()
        df['vol_ratio'] = df['vol'] / df['vol_ma20']

        if 'oi' in df.columns:
            df['oi_change'] = df['oi'].pct_change(5)
        else:
            df['oi_change'] = 0

        df = df.dropna(subset=['ma20', 'ma60', 'hv_20', 'mom_10', 'rsi', 'adx'])
        if len(df) > 100:
            data[symbol] = df
    return data


def run_backtest(data, start_date, end_date, params):
    max_pos = params.get('max_pos', 3)
    hold_days = params.get('hold_days', 10)
    otm_pct = params.get('otm_pct', 0.02)  # OTM distance for buying
    risk_pct = params.get('risk_pct', 0.15)  # % of equity per trade (Kelly)
    min_mom = params.get('min_mom', 0.02)
    require_align = params.get('require_align', True)
    min_adx = params.get('min_adx', 0)
    take_profit_mult = params.get('take_profit', 0)  # 0=no early TP, else exit when val > cost * mult
    r = 0.02
    comm_rate = 0.0003

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

    for date in dates:
        day_data = date_map[date]

        # === 退出 ===
        for symbol in list(positions.keys()):
            pos = positions[symbol]
            if symbol not in day_data:
                continue
            row = day_data[symbol]
            price = row['close']
            hv = row.get('hv_20', 0.25)
            hd = (date - pos['entry_date']).days

            rem_T = max((hold_days - hd) / 365.0, 0.001)
            val = bs_price(price, pos['strike'], rem_T, r, hv, pos['otype'])
            intrinsic = max(price - pos['strike'], 0) if pos['dir'] == 1 else max(pos['strike'] - price, 0)
            val = max(val, intrinsic * 0.9)
            total_val = val * pos['mult'] * pos['contracts']

            should_close = False

            # 时间退出
            if hd >= hold_days:
                should_close = True

            # 提前止盈: 期权价值 > cost * multiplier
            elif take_profit_mult > 0 and total_val > pos['cost'] * take_profit_mult:
                should_close = True

            if should_close:
                comm = total_val * comm_rate
                net = total_val - pos['cost'] - comm - pos['comm']
                cash += total_val - comm
                closed_pnls.append(net)
                del positions[symbol]

        # === 入场 ===
        if len(positions) < max_pos:
            signals = []
            for symbol, row in day_data.items():
                if symbol in positions:
                    continue
                if row.get('atr_pct', 0.1) > 0.045:
                    continue

                trend = row.get('trend', 0)
                mom_5 = row.get('mom_5', 0)
                mom_10 = row.get('mom_10', 0)
                mom_20 = row.get('mom_20', 0)
                adx = row.get('adx', 0)
                rsi = row.get('rsi', 50)
                hv = row.get('hv_20', 0)
                vol_ratio = row.get('vol_ratio', 1.0)

                if hv < 0.10 or hv > 0.60:
                    continue

                direction = 0
                if trend == 1 and mom_10 > min_mom:
                    direction = 1
                elif trend == -1 and mom_10 < -min_mom:
                    direction = -1

                if direction == 0:
                    continue

                # 多时间框架对齐
                if require_align:
                    if direction == 1 and not (mom_5 > 0 and mom_10 > 0 and mom_20 > 0):
                        continue
                    elif direction == -1 and not (mom_5 < 0 and mom_10 < 0 and mom_20 < 0):
                        continue

                # ADX过滤
                if min_adx > 0 and adx < min_adx:
                    continue

                # 综合评分
                score = abs(mom_10) * 100
                if adx > 25:
                    score += 0.5
                if vol_ratio > 1.2:
                    score += 0.3

                signals.append((symbol, direction, score, hv))

            signals.sort(key=lambda x: x[2], reverse=True)

            for symbol, direction, score, hv in signals:
                if len(positions) >= max_pos:
                    break

                row = day_data[symbol]
                S = row['close']
                mult, _, _, _ = get_spec(symbol)

                # 买OTM期权
                if direction == 1:
                    K = S * (1 + otm_pct)  # 买OTM call (strike above current)
                    otype = 'call'
                else:
                    K = S * (1 - otm_pct)  # 买OTM put (strike below current)
                    otype = 'put'

                T = hold_days / 365.0
                premium = bs_price(S, K, T, r, hv, otype)
                if premium <= 0:
                    continue

                cost_per = premium * mult
                # Kelly仓位: risk_pct% of equity
                risk_amount = equity * risk_pct
                contracts = max(int(risk_amount / cost_per), 1)
                total_cost = cost_per * contracts
                comm = total_cost * comm_rate

                # 现金约束
                if total_cost + comm > cash * 0.6:
                    contracts = max(int((cash * 0.6 - comm) / cost_per), 0)
                    if contracts <= 0:
                        continue
                    total_cost = cost_per * contracts
                    comm = total_cost * comm_rate

                cash -= total_cost + comm
                positions[symbol] = {
                    'dir': direction, 'otype': otype,
                    'entry_price': S, 'strike': K,
                    'entry_date': date, 'contracts': contracts,
                    'mult': mult, 'cost': total_cost, 'comm': comm,
                }

        # === 权益 ===
        unrealized = 0
        for symbol, pos in positions.items():
            if symbol in day_data:
                row = day_data[symbol]
                price = row['close']
                hv = row.get('hv_20', 0.25)
                hd = (date - pos['entry_date']).days
                rem_T = max((hold_days - hd) / 365.0, 0.001)
                val = bs_price(price, pos['strike'], rem_T, r, hv, pos['otype'])
                intrinsic = max(price - pos['strike'], 0) if pos['dir'] == 1 else max(pos['strike'] - price, 0)
                val = max(val, intrinsic * 0.9)
                unrealized += val * pos['mult'] * pos['contracts']

        equity = cash + unrealized
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

    # === 1. 基础: 对齐 + OTM + Kelly ===
    print("=== 对齐 + OTM + Kelly仓位 ===")
    for otm in [0.0, 0.01, 0.02, 0.03, 0.05]:
        for risk in [0.05, 0.10, 0.15, 0.20]:
            for hd in [5, 7, 10]:
                params = dict(hold_days=hd, otm_pct=otm, risk_pct=risk,
                             require_align=True, min_mom=0.02, min_adx=0,
                             take_profit=0, max_pos=3)
                r = run_backtest(data, start_date, end_date, params)
                if r:
                    results.append(r)

    # === 2. 对齐 + ADX + 高Kelly ===
    print("\n=== 对齐 + ADX + 高Kelly ===")
    for adx_min in [20, 25, 30]:
        for otm in [0.01, 0.02, 0.03]:
            for risk in [0.10, 0.15, 0.20]:
                for hd in [7, 10]:
                    params = dict(hold_days=hd, otm_pct=otm, risk_pct=risk,
                                 require_align=True, min_mom=0.02, min_adx=adx_min,
                                 take_profit=0, max_pos=3)
                    r = run_backtest(data, start_date, end_date, params)
                    if r:
                        results.append(r)

    # === 3. 提前止盈 ===
    print("\n=== 提前止盈 ===")
    for tp in [1.5, 2.0, 3.0]:
        for otm in [0.01, 0.02, 0.03]:
            for risk in [0.10, 0.15, 0.20]:
                params = dict(hold_days=10, otm_pct=otm, risk_pct=risk,
                             require_align=True, min_mom=0.02, min_adx=0,
                             take_profit=tp, max_pos=3)
                r = run_backtest(data, start_date, end_date, params)
                if r:
                    results.append(r)

    # === 4. 不对齐 (对照) ===
    print("\n=== 不对齐 (对照) ===")
    for otm in [0.0, 0.02]:
        for risk in [0.05, 0.10, 0.15]:
            for hd in [5, 10]:
                params = dict(hold_days=hd, otm_pct=otm, risk_pct=risk,
                             require_align=False, min_mom=0.03, min_adx=0,
                             take_profit=0, max_pos=3)
                r = run_backtest(data, start_date, end_date, params)
                if r:
                    results.append(r)

    # === 输出 ===
    results.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n\n{'OTM':>5} {'持有':>4} {'风险':>5} {'对齐':>4} {'ADX':>4} {'止盈':>4} {'年化':>8} {'胜率':>6} {'盈亏比':>6} {'回撤':>8} {'交易':>4} {'最终':>14}")
    print("-" * 100)

    for r in results[:50]:
        align = 'Y' if r.get('require_align') else 'N'
        adx_str = f">{r.get('min_adx',0)}" if r.get('min_adx', 0) > 0 else '-'
        tp_str = f"{r.get('take_profit',0):.1f}x" if r.get('take_profit', 0) > 0 else '-'
        print(f"{r.get('otm_pct',0):>5.0%} {r.get('hold_days',0):>4} {r.get('risk_pct',0):>5.0%} "
              f"{align:>4} {adx_str:>4} {tp_str:>4} "
              f"{r['annual']:>8.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['trades']:>4} {r['final']:>14,.0f}")

    # 达标分析
    print("\n\n=== 达标分析 ===")
    for ann_target, label in [(6.0, "600%"), (3.0, "300%"), (1.0, "100%")]:
        good = [r for r in results if r['annual'] >= ann_target]
        print(f"\n年化>={label}: {len(good)}个组合")
        if good:
            best = max(good, key=lambda x: x['wr'])
            print(f"  最高WR: 年化={best['annual']:.1%}  WR={best['wr']:.1%}  PF={best['pf']:.2f}  "
                  f"MDD={best['mdd']:.1%}  交易={best['trades']}")
            best_wr50 = [r for r in good if r['wr'] >= 0.50]
            if best_wr50:
                b = max(best_wr50, key=lambda x: x['annual'])
                print(f"  WR>=50%最高年化: 年化={b['annual']:.1%}  WR={b['wr']:.1%}  PF={b['pf']:.2f}  MDD={b['mdd']:.1%}")
            else:
                print(f"  WR>=50%: 无")

    # WR分组
    print("\n\n=== 按WR分组 ===")
    for wr_min, wr_max in [(0.55, 1.0), (0.50, 0.55), (0.45, 0.50), (0.40, 0.45)]:
        group = [r for r in results if wr_min <= r['wr'] < wr_max]
        if group:
            best = max(group, key=lambda x: x['annual'])
            print(f"  WR {wr_min:.0%}-{wr_max:.0%}: {len(group)}个 | 最佳年化={best['annual']:.1%} "
                  f"WR={best['wr']:.1%} PF={best['pf']:.2f} MDD={best['mdd']:.1%} "
                  f"OTM={best.get('otm_pct',0):.0%} 持有={best.get('hold_days',0)} 风险={best.get('risk_pct',0):.0%}")

    # 对齐 vs 不对齐
    print("\n\n=== 对齐 vs 不对齐 ===")
    aligned = [r for r in results if r.get('require_align')]
    not_aligned = [r for r in results if not r.get('require_align')]
    for label, group in [("对齐", aligned), ("不对齐", not_aligned)]:
        if group:
            top5 = sorted(group, key=lambda x: x['annual'], reverse=True)[:5]
            avg_wr = np.mean([r['wr'] for r in top5])
            avg_ann = np.mean([r['annual'] for r in top5])
            avg_pf = np.mean([r['pf'] for r in top5])
            print(f"  {label}: TOP5 平均年化={avg_ann:.1%}  平均WR={avg_wr:.1%}  平均PF={avg_pf:.2f}")


if __name__ == '__main__':
    main()
