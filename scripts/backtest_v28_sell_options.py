#!/usr/bin/env python3
"""
策略 v28 — 卖OTM期权增强版
v27发现: 卖OTM 2%期权 7天持有 = 77.5% WR, 81.6%年化
问题: PF=0.99 (盈亏比差), 距600%年化还差7x

改进:
1. 多因子信号: 动量+ADX+RSI+量能+OI → 提高WR到80%+
2. 早期止损: 亏损>阈值时平仓 → 提高PF到1.5+
3. 激进仓位: 5-15%权益风险 → 放大收益3-7x
4. IV>HV过滤: 只在高VRP时卖 → 提高每笔收益

目标: 600%年化 + 70%+WR + MDD<40%
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

        # 历史波动率
        df['hv_10'] = df['return'].rolling(10).std() * np.sqrt(252)
        df['hv_20'] = df['return'].rolling(20).std() * np.sqrt(252)
        df['hv_60'] = df['return'].rolling(60).std() * np.sqrt(252)

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

        # 动量
        df['mom_5'] = df['close'].pct_change(5)
        df['mom_10'] = df['close'].pct_change(10)
        df['mom_20'] = df['close'].pct_change(20)

        # 趋势
        df['trend'] = np.where(df['ma20'] > df['ma60'], 1, -1)

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

        # 量能
        df['vol_ma20'] = df['vol'].rolling(20).mean()
        df['vol_ratio'] = df['vol'] / df['vol_ma20']

        # OI变化
        if 'oi' in df.columns:
            df['oi_change'] = df['oi'].pct_change(5)
        else:
            df['oi_change'] = 0

        df = df.dropna(subset=['ma20', 'ma60', 'hv_20', 'mom_10', 'rsi', 'adx'])
        if len(df) > 100:
            data[symbol] = df
    return data


def get_enhanced_signals(day_data, positions, params):
    """多因子评分: 动量+ADX+RSI+量能+OI"""
    signals = []
    min_mom = params.get('min_mom', 0.03)
    min_adx = params.get('min_adx', 0)
    min_score = params.get('min_score', 0)

    for symbol, row in day_data.items():
        if symbol in positions:
            continue
        if row.get('atr_pct', 0.1) > 0.045:
            continue

        trend = row.get('trend', 0)
        mom = row.get('mom_10', 0)
        adx = row.get('adx', 0)
        rsi = row.get('rsi', 50)
        vol_ratio = row.get('vol_ratio', 1.0)
        oi_change = row.get('oi_change', 0)
        hv = row.get('hv_20', 0)
        mom_5 = row.get('mom_5', 0)

        if hv < 0.10 or hv > 0.60:
            continue

        direction = 0
        score = 0

        if trend == 1 and mom > min_mom:
            direction = 1
            # 动量分 (0-30)
            score += min(abs(mom) * 300, 30)
            # ADX分 (0-20): 趋势越强越好
            if adx > min_adx:
                score += min((adx - 15) * 1.0, 20)
            # RSI分 (0-20): 回调区间更好(不追高)
            if 30 <= rsi <= 55:
                score += 20
            elif 55 < rsi <= 65:
                score += 10
            # 量能分 (0-15): 放量确认
            if vol_ratio > 1.2:
                score += 15
            elif vol_ratio > 1.0:
                score += 8
            # OI分 (0-15): OI增加=资金流入
            if oi_change > 0.02:
                score += 15
            elif oi_change > 0:
                score += 5
            # 短期加速分
            if mom_5 > 0.01:
                score += 10

        elif trend == -1 and mom < -min_mom:
            direction = -1
            score += min(abs(mom) * 300, 30)
            if adx > min_adx:
                score += min((adx - 15) * 1.0, 20)
            if 45 <= rsi <= 70:
                score += 20
            elif 35 <= rsi < 45:
                score += 10
            if vol_ratio > 1.2:
                score += 15
            elif vol_ratio > 1.0:
                score += 8
            if oi_change < -0.02:
                score += 15
            elif oi_change < 0:
                score += 5
            if mom_5 < -0.01:
                score += 10

        if direction != 0 and score >= min_score:
            signals.append((symbol, direction, score, hv))

    signals.sort(key=lambda x: x[2], reverse=True)
    return signals


def run_backtest(data, start_date, end_date, params):
    """卖OTM期权回测 with 早期止损"""
    max_pos = params.get('max_pos', 3)
    risk_pct = params.get('risk_pct', 0.05)
    otm_pct = params.get('otm_pct', 0.02)
    hold_days = params.get('hold_days', 7)
    stop_loss_ratio = params.get('stop_loss_ratio', 0)  # 0=不止损
    r = 0.02
    comm_rate = 0.0003

    # Build date index
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
    yearly_pnl = defaultdict(float)

    for date in dates:
        day_data = date_map[date]

        # === 退出检查 ===
        for symbol in list(positions.keys()):
            pos = positions[symbol]
            if symbol not in day_data:
                continue
            row = day_data[symbol]
            price = row['close']
            hv = row.get('hv_20', 0.25)
            hd = (date - pos['entry_date']).days

            # 计算当前买回价
            rem_T = max((pos['hold_days'] - hd) / 365.0, 0.001)
            # IV增加: 当价格不利时, IV通常上升 (波动率微笑)
            iv_mult = 1.0
            if pos['dir'] == 1:  # 卖put, 价格下跌不利
                price_move = (pos['entry_price'] - price) / pos['entry_price']
                if price_move > 0:
                    iv_mult = 1.0 + price_move * 2  # 不利时IV上升
            else:  # 卖call, 价格上涨不利
                price_move = (price - pos['entry_price']) / pos['entry_price']
                if price_move > 0:
                    iv_mult = 1.0 + price_move * 2

            buyback = bs_price(price, pos['strike'], rem_T, r, hv * iv_mult, pos['otype'])
            intrinsic = max(price - pos['strike'], 0) if pos['otype'] == 'call' else max(pos['strike'] - price, 0)
            buyback = max(buyback, intrinsic * 0.95)
            total_buyback = buyback * pos['mult'] * pos['contracts']

            unrealized_net = pos['credit'] - total_buyback
            comm = total_buyback * comm_rate

            should_close = False
            reason = ''

            # 时间退出
            if hd >= hold_days:
                should_close = True
                reason = 'time'

            # 止损: 亏损超过阈值
            elif stop_loss_ratio > 0 and unrealized_net < -stop_loss_ratio * pos['credit']:
                should_close = True
                reason = 'stop'

            # 价格穿过strike (OTM缓冲消失)
            if not should_close and params.get('stop_at_strike', False):
                if pos['dir'] == 1 and price < pos['strike']:
                    should_close = True
                    reason = 'strike_touch'
                elif pos['dir'] == -1 and price > pos['strike']:
                    should_close = True
                    reason = 'strike_touch'

            if should_close:
                net = unrealized_net - comm - pos['comm']
                cash += pos['margin'] + net
                closed_pnls.append(net)
                yearly_pnl[date.year] += net
                del positions[symbol]

        # === 入场 ===
        if len(positions) < max_pos:
            signals = get_enhanced_signals(day_data, positions, params)

            for symbol, direction, score, hv in signals:
                if len(positions) >= max_pos:
                    break

                row = day_data[symbol]
                S = row['close']
                mult, mr, _, _ = get_spec(symbol)

                # Strike: OTM
                if direction == 1:
                    K = S * (1 - otm_pct)  # 卖OTM put
                    otype = 'put'
                else:
                    K = S * (1 + otm_pct)  # 卖OTM call
                    otype = 'call'

                T = hold_days / 365.0
                premium = bs_price(S, K, T, r, hv, otype)
                if premium <= 0:
                    continue

                credit_per = premium * mult  # 每张权利金
                margin_per = S * mult * mr   # 保证金

                # 风险: OTM距离 * 乘数 = 最大亏损
                risk_per = abs(S - K) * mult
                if risk_per <= 0:
                    continue

                # 合约数量: 基于风险
                risk_amount = equity * risk_pct
                contracts = max(int(risk_amount / risk_per), 1)

                total_margin = margin_per * contracts
                total_credit = credit_per * contracts
                comm = total_credit * comm_rate

                # 现金约束: 保证金不超过现金60%
                if total_margin + comm > cash * 0.6:
                    contracts = max(int((cash * 0.6 - comm) / margin_per), 0)
                    if contracts <= 0:
                        continue
                    total_margin = margin_per * contracts
                    total_credit = credit_per * contracts
                    comm = total_credit * comm_rate

                # 总保证金不超过权益80%
                total_existing = sum(p['margin'] for p in positions.values())
                if total_existing + total_margin > equity * 0.8:
                    contracts = max(int((equity * 0.8 - total_existing) / margin_per), 0)
                    if contracts <= 0:
                        continue
                    total_margin = margin_per * contracts
                    total_credit = credit_per * contracts
                    comm = total_credit * comm_rate

                cash -= total_margin - total_credit + comm
                positions[symbol] = {
                    'type': 'sell_option',
                    'dir': direction,
                    'otype': otype,
                    'entry_price': S,
                    'strike': K,
                    'entry_date': date,
                    'contracts': contracts,
                    'mult': mult,
                    'margin': total_margin,
                    'credit': total_credit,
                    'comm': comm,
                    'hold_days': hold_days,
                    'score': score,
                }

        # === 权益 ===
        unrealized = 0
        for symbol, pos in positions.items():
            if symbol in day_data:
                row = day_data[symbol]
                price = row['close']
                hv = row.get('hv_20', 0.25)
                hd = (date - pos['entry_date']).days
                rem_T = max((pos['hold_days'] - hd) / 365.0, 0.001)
                buyback = bs_price(price, pos['strike'], rem_T, r, hv, pos['otype'])
                intrinsic = max(price - pos['strike'], 0) if pos['otype'] == 'call' else max(pos['strike'] - price, 0)
                buyback = max(buyback, intrinsic * 0.95)
                unrealized += pos['credit'] - buyback * pos['mult'] * pos['contracts']

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
        'yearly_pnl': dict(yearly_pnl),
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

    # === Phase 1: 激进仓位扫描 ===
    print("=== Phase 1: 风险仓位扫描 ===")
    for risk in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        for otm in [0.02, 0.025, 0.03]:
            for hd in [5, 7]:
                params = {'risk_pct': risk, 'otm_pct': otm, 'hold_days': hd,
                          'min_mom': 0.03, 'min_score': 0, 'max_pos': 3,
                          'stop_loss_ratio': 0, 'min_adx': 0}
                r = run_backtest(data, start_date, end_date, params)
                if r:
                    results.append(r)
                    print(f"  risk={risk:.0%} otm={otm:.1%} hd={hd} → "
                          f"ann={r['annual']:.1%} wr={r['wr']:.1%} mdd={r['mdd']:.1%} "
                          f"pf={r['pf']:.2f} trades={r['trades']}")

    # === Phase 2: 止损扫描 (基于Phase1最优) ===
    print("\n=== Phase 2: 止损扫描 ===")
    # 找出年化最高的3个配置
    top = sorted(results, key=lambda x: x['annual'], reverse=True)[:3]
    for base in top:
        for stop in [1.5, 2.0, 3.0, 5.0]:
            params = dict(base)
            params['stop_loss_ratio'] = stop
            r = run_backtest(data, start_date, end_date, params)
            if r:
                r['stop_loss'] = stop
                results.append(r)
                print(f"  stop={stop}x risk={params['risk_pct']:.0%} otm={params['otm_pct']:.1%} → "
                      f"ann={r['annual']:.1%} wr={r['wr']:.1%} mdd={r['mdd']:.1%} pf={r['pf']:.2f}")

    # === Phase 3: 信号增强 (基于Phase2最优) ===
    print("\n=== Phase 3: 信号增强 ===")
    # 找出年化最高且WR>=60%的配置
    good = sorted([r for r in results if r['wr'] >= 0.55], key=lambda x: x['annual'], reverse=True)[:3]
    for base in good:
        for min_score in [30, 50, 70]:
            for min_adx in [15, 20, 25]:
                params = dict(base)
                params['min_score'] = min_score
                params['min_adx'] = min_adx
                r = run_backtest(data, start_date, end_date, params)
                if r:
                    r['config'] = f'score>{min_score}_adx>{min_adx}'
                    results.append(r)

    # === 输出 ===
    print(f"\n\n{'风险':>5} {'OTM':>5} {'持有':>4} {'止损':>4} {'年化':>8} {'胜率':>6} {'盈亏比':>6} {'回撤':>8} {'交易':>4} {'最终':>14} {'配置':>20}")
    print("-" * 100)

    results.sort(key=lambda x: x['annual'], reverse=True)
    for r in results[:50]:
        stop_str = f"{r.get('stop_loss', 0):.1f}x" if r.get('stop_loss', 0) > 0 else "-"
        cfg = r.get('config', '')
        print(f"{r.get('risk_pct',0):>5.0%} {r.get('otm_pct',0):>5.1%} {r.get('hold_days',0):>4} "
              f"{stop_str:>4} {r['annual']:>8.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['trades']:>4} {r['final']:>14,.0f} {cfg:>20}")

    # === 筛选达标 ===
    print("\n\n=== 目标: 年化>=600% 且 WR>=50% ===")
    target = [r for r in results if r['annual'] >= 6.0 and r['wr'] >= 0.50]
    if target:
        for r in sorted(target, key=lambda x: x['annual'], reverse=True)[:10]:
            print(f"  年化={r['annual']:.1%}  胜率={r['wr']:.1%}  回撤={r['mdd']:.1%}  "
                  f"盈亏比={r['pf']:.2f}  交易={r['trades']}  "
                  f"风险={r.get('risk_pct',0):.0%}  OTM={r.get('otm_pct',0):.1%}  "
                  f"持有={r.get('hold_days',0)}  止损={r.get('stop_loss',0):.1f}x")
    else:
        print("无满足条件的组合")

    print("\n=== 年化>=300% ===")
    high = [r for r in results if r['annual'] >= 3.0]
    if high:
        for r in sorted(high, key=lambda x: x['annual'], reverse=True)[:10]:
            print(f"  年化={r['annual']:.1%}  胜率={r['wr']:.1%}  回撤={r['mdd']:.1%}  "
                  f"盈亏比={r['pf']:.2f}  交易={r['trades']}  "
                  f"风险={r.get('risk_pct',0):.0%}  OTM={r.get('otm_pct',0):.1%}  "
                  f"持有={r.get('hold_days',0)}  止损={r.get('stop_loss',0):.1f}x")
    else:
        print("无满足条件的组合")

    print("\n=== WR>=50% 按年化排序 ===")
    wr_good = [r for r in results if r['wr'] >= 0.50]
    if wr_good:
        for r in sorted(wr_good, key=lambda x: x['annual'], reverse=True)[:15]:
            print(f"  年化={r['annual']:.1%}  胜率={r['wr']:.1%}  回撤={r['mdd']:.1%}  "
                  f"盈亏比={r['pf']:.2f}  交易={r['trades']}  "
                  f"风险={r.get('risk_pct',0):.0%}  OTM={r.get('otm_pct',0):.1%}  "
                  f"持有={r.get('hold_days',0)}  止损={r.get('stop_loss',0):.1f}x")
    else:
        print("无满足条件的组合")

    # TOP3 年度明细
    print("\n\n=== TOP 3 年度明细 ===")
    for i, r in enumerate(sorted(results, key=lambda x: x['annual'], reverse=True)[:3]):
        print(f"\n--- #{i+1}: 年化={r['annual']:.1%}  胜率={r['wr']:.1%}  "
              f"风险={r.get('risk_pct',0):.0%}  OTM={r.get('otm_pct',0):.1%}  "
              f"持有={r.get('hold_days',0)} ---")
        if r.get('yearly_pnl'):
            for yr in sorted(r['yearly_pnl'].keys()):
                pnl = r['yearly_pnl'][yr]
                print(f"  {yr}: {pnl:>14,.0f}")


if __name__ == '__main__':
    main()
