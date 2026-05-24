#!/usr/bin/env python3
"""
策略 v29 — 期货+期权混合: 方向期货 + 卖期权收权利金
思路:
  - 3个持仓槽位
  - 当信号极强(score>70): 买期权 (凸性收益, 3-10x)
  - 当信号中等(score 40-70): 卖OTM期权 (高WR 70%+, 收权利金)
  - 当信号一般(score 0-40): 开期货方向仓 (线性收益, 50% WR)

高维信号:
  - 动量 (多周期: 5d/10d/20d)
  - ADX趋势强度
  - RSI超买超卖
  - 量能确认
  - OI资金流
  - 波动率状态 (HV分位数)

仓位管理:
  - Kelly比例: 根据信号强度动态调仓
  - 强信号 → 大仓位
  - 弱信号 → 小仓位
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
        df['hv_10'] = df['return'].rolling(10).std() * np.sqrt(252)
        df['hv_20'] = df['return'].rolling(20).std() * np.sqrt(252)
        df['hv_60'] = df['return'].rolling(60).std() * np.sqrt(252)

        # HV percentile (60-day lookback)
        df['hv_pct'] = df['hv_20'].rolling(60).rank(pct=True)

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

        # 多周期动量一致性
        df['mom_align'] = 0
        mask_all_pos = (df['mom_5'] > 0) & (df['mom_10'] > 0) & (df['mom_20'] > 0)
        mask_all_neg = (df['mom_5'] < 0) & (df['mom_10'] < 0) & (df['mom_20'] < 0)
        df.loc[mask_all_pos, 'mom_align'] = 1
        df.loc[mask_all_neg, 'mom_align'] = -1

        df = df.dropna(subset=['ma20', 'ma60', 'hv_20', 'mom_10', 'rsi', 'adx', 'hv_pct'])
        if len(df) > 100:
            data[symbol] = df
    return data


def score_signal(row, direction):
    """多因子评分 (0-100)"""
    score = 0
    mom = row.get('mom_10', 0)
    mom_5 = row.get('mom_5', 0)
    mom_20 = row.get('mom_20', 0)
    adx = row.get('adx', 0)
    rsi = row.get('rsi', 50)
    vol_ratio = row.get('vol_ratio', 1.0)
    oi_change = row.get('oi_change', 0)
    mom_align = row.get('mom_align', 0)
    hv_pct = row.get('hv_pct', 0.5)

    if direction == 1:
        # 动量 (0-25)
        score += min(abs(mom) * 250, 25)
        # 多周期一致性 (0-20)
        if mom_align == 1:
            score += 20
        elif mom_5 > 0 and mom > 0:
            score += 10
        # ADX趋势强度 (0-15)
        if adx > 30:
            score += 15
        elif adx > 20:
            score += 8
        # RSI: 回调区间好(不追高) (0-15)
        if 30 <= rsi <= 45:
            score += 15
        elif 45 < rsi <= 55:
            score += 10
        elif 55 < rsi <= 65:
            score += 5
        # 量能确认 (0-10)
        if vol_ratio > 1.5:
            score += 10
        elif vol_ratio > 1.2:
            score += 7
        elif vol_ratio > 1.0:
            score += 3
        # OI (0-10)
        if oi_change > 0.03:
            score += 10
        elif oi_change > 0:
            score += 5
        # HV位置: 中等波动率最好 (0-5)
        if 0.3 < hv_pct < 0.7:
            score += 5

    elif direction == -1:
        score += min(abs(mom) * 250, 25)
        if mom_align == -1:
            score += 20
        elif mom_5 < 0 and mom < 0:
            score += 10
        if adx > 30:
            score += 15
        elif adx > 20:
            score += 8
        if 55 <= rsi <= 70:
            score += 15
        elif 45 <= rsi < 55:
            score += 10
        elif 35 <= rsi < 45:
            score += 5
        if vol_ratio > 1.5:
            score += 10
        elif vol_ratio > 1.2:
            score += 7
        elif vol_ratio > 1.0:
            score += 3
        if oi_change < -0.03:
            score += 10
        elif oi_change < 0:
            score += 5
        if 0.3 < hv_pct < 0.7:
            score += 5

    return score


def run_backtest(data, start_date, end_date, params):
    """混合策略回测"""
    max_pos = params.get('max_pos', 3)
    r = 0.02
    lev = params.get('leverage', 5)
    hold_days = params.get('hold_days', 5)
    otm_pct = params.get('otm_pct', 0.02)
    risk_pct = params.get('risk_pct', 0.05)
    sell_risk = params.get('sell_risk', 0.08)

    # 信号阈值
    buy_opt_threshold = params.get('buy_opt_score', 70)
    sell_opt_threshold = params.get('sell_opt_score', 40)
    fut_threshold = params.get('fut_score', 20)
    min_mom = params.get('min_mom', 0.02)

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
    trade_stats = {'buy_opt': 0, 'sell_opt': 0, 'futures': 0}

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

            if hd < pos['hold_days']:
                continue

            if pos['type'] == 'futures':
                pnl = (price - pos['entry_price']) * pos['dir'] * pos['mult'] * pos['lots']
                comm = price * pos['mult'] * pos['lots'] * 0.00015 * 2
                net = pnl - comm
                cash += pos['margin'] + net
                closed_pnls.append(net)
                del positions[symbol]

            elif pos['type'] == 'buy_option':
                rem_T = max(0.001 / 365, 0.001)
                val = bs_price(price, pos['strike'], rem_T, r, hv, pos['otype'])
                intrinsic = max(price - pos['strike'], 0) if pos['dir'] == 1 else max(pos['strike'] - price, 0)
                val = max(val, intrinsic * 0.9)
                total_val = val * pos['mult'] * pos['contracts']
                comm = total_val * 0.0003
                net = total_val - pos['cost'] - comm - pos['comm']
                cash += total_val - comm
                closed_pnls.append(net)
                del positions[symbol]

            elif pos['type'] == 'sell_option':
                rem_T = max(0.001 / 365, 0.001)
                buyback = bs_price(price, pos['strike'], rem_T, r, hv, pos['otype'])
                intrinsic = max(price - pos['strike'], 0) if pos['otype'] == 'call' else max(pos['strike'] - price, 0)
                buyback = max(buyback, intrinsic * 0.95)
                total_buyback = buyback * pos['mult'] * pos['contracts']
                comm = total_buyback * 0.0003
                net = pos['credit'] - total_buyback - comm - pos['comm']
                cash += pos['margin'] + net
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
                mom = row.get('mom_10', 0)
                hv = row.get('hv_20', 0)
                if hv < 0.10 or hv > 0.60:
                    continue

                # 方向
                direction = 0
                if trend == 1 and mom > min_mom:
                    direction = 1
                elif trend == -1 and mom < -min_mom:
                    direction = -1

                if direction == 0:
                    continue

                score = score_signal(row, direction)
                if score >= fut_threshold:
                    signals.append((symbol, direction, score, hv))

            signals.sort(key=lambda x: x[2], reverse=True)

            for symbol, direction, score, hv in signals:
                if len(positions) >= max_pos:
                    break

                row = day_data[symbol]
                S = row['close']
                mult, mr, _, _ = get_spec(symbol)

                # 根据分数选择策略类型
                if score >= buy_opt_threshold:
                    # 买ATM期权 (凸性收益)
                    ptype = 'buy_option'
                    K = S
                    T = hold_days / 365.0
                    otype = 'call' if direction == 1 else 'put'
                    premium = bs_price(S, K, T, r, hv, otype)
                    if premium <= 0:
                        continue
                    cost_per = premium * mult
                    risk_amount = equity * risk_pct
                    contracts = max(int(risk_amount / cost_per), 1)
                    total_cost = cost_per * contracts
                    comm = total_cost * 0.0003
                    if total_cost + comm > cash * 0.4:
                        contracts = max(int((cash * 0.4 - comm) / cost_per), 0)
                        if contracts <= 0:
                            continue
                        total_cost = cost_per * contracts
                        comm = total_cost * 0.0003
                    cash -= total_cost + comm
                    positions[symbol] = {
                        'type': 'buy_option', 'dir': direction, 'otype': otype,
                        'entry_price': S, 'strike': K, 'entry_date': date,
                        'contracts': contracts, 'mult': mult, 'cost': total_cost,
                        'comm': comm, 'hold_days': hold_days,
                    }
                    trade_stats['buy_opt'] += 1

                elif score >= sell_opt_threshold:
                    # 卖OTM期权 (收权利金)
                    ptype = 'sell_option'
                    if direction == 1:
                        K = S * (1 - otm_pct)
                        otype = 'put'
                    else:
                        K = S * (1 + otm_pct)
                        otype = 'call'
                    T = hold_days / 365.0
                    premium = bs_price(S, K, T, r, hv, otype)
                    if premium <= 0:
                        continue

                    credit_per = premium * mult
                    margin_per = S * mult * mr
                    risk_per = abs(S - K) * mult
                    if risk_per <= 0:
                        continue

                    contracts = max(int(equity * sell_risk / risk_per), 1)
                    total_margin = margin_per * contracts
                    total_credit = credit_per * contracts
                    comm = total_credit * 0.0003

                    if total_margin + comm > cash * 0.5:
                        contracts = max(int((cash * 0.5 - comm) / margin_per), 0)
                        if contracts <= 0:
                            continue
                        total_margin = margin_per * contracts
                        total_credit = credit_per * contracts
                        comm = total_credit * 0.0003

                    total_existing = sum(p['margin'] for p in positions.values() if p.get('margin', 0) > 0)
                    if total_existing + total_margin > equity * 0.8:
                        contracts = max(int((equity * 0.8 - total_existing) / margin_per), 0)
                        if contracts <= 0:
                            continue
                        total_margin = margin_per * contracts
                        total_credit = credit_per * contracts
                        comm = total_credit * 0.0003

                    cash -= total_margin - total_credit + comm
                    positions[symbol] = {
                        'type': 'sell_option', 'dir': direction, 'otype': otype,
                        'entry_price': S, 'strike': K, 'entry_date': date,
                        'contracts': contracts, 'mult': mult, 'margin': total_margin,
                        'credit': total_credit, 'comm': comm, 'hold_days': hold_days,
                    }
                    trade_stats['sell_opt'] += 1

                else:
                    # 期货 (线性收益)
                    ptype = 'futures'
                    mpl = S * mult * mr
                    if mpl <= 0:
                        continue
                    target_n = equity * (lev / max_pos)
                    lots = max(int(target_n / (S * mult)), 1)

                    total_m = sum(p.get('margin', 0) for p in positions.values()) + mpl * lots
                    if total_m > equity * 0.85:
                        lots = max(int((equity * 0.85 - sum(p.get('margin', 0) for p in positions.values())) / mpl), 0)
                        if lots <= 0:
                            continue

                    am = mpl * lots
                    comm = S * mult * lots * 0.00015
                    if am + comm > cash:
                        lots = max(int((cash - comm) / mpl), 0)
                        if lots <= 0:
                            continue
                        am = mpl * lots
                        comm = S * mult * lots * 0.00015

                    cash -= am + comm
                    positions[symbol] = {
                        'type': 'futures', 'dir': direction,
                        'entry_price': S * (1 + 0.0001 * direction),
                        'entry_date': date, 'lots': lots, 'mult': mult,
                        'margin': am, 'hold_days': hold_days,
                    }
                    trade_stats['futures'] += 1

        # === 权益 ===
        unrealized = 0
        for symbol, pos in positions.items():
            if symbol not in day_data:
                continue
            row = day_data[symbol]
            price = row['close']
            hv = row.get('hv_20', 0.25)
            hd = (date - pos['entry_date']).days

            if pos['type'] == 'futures':
                unrealized += (price - pos['entry_price']) * pos['dir'] * pos['mult'] * pos['lots']
            elif pos['type'] == 'buy_option':
                rem_T = max((pos['hold_days'] - hd) / 365.0, 0.001)
                val = bs_price(price, pos['strike'], rem_T, r, hv, pos['otype'])
                intrinsic = max(price - pos['strike'], 0) if pos['dir'] == 1 else max(pos['strike'] - price, 0)
                val = max(val, intrinsic * 0.9)
                unrealized += val * pos['mult'] * pos['contracts']
            elif pos['type'] == 'sell_option':
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
        'trade_stats': dict(trade_stats),
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

    # === 扫描混合策略参数 ===
    print("=== 混合策略参数扫描 ===")
    for lev in [4, 5, 6]:
        for hd in [5, 7]:
            for buy_score in [60, 70, 80]:
                for sell_score in [30, 40, 50]:
                    params = {
                        'leverage': lev, 'hold_days': hd,
                        'buy_opt_score': buy_score, 'sell_opt_score': sell_score,
                        'fut_score': 10, 'min_mom': 0.02,
                        'otm_pct': 0.02, 'risk_pct': 0.03, 'sell_risk': 0.05,
                        'max_pos': 3,
                    }
                    r = run_backtest(data, start_date, end_date, params)
                    if r:
                        results.append(r)

    # === 更激进的期权策略 ===
    print("\n=== 激进期权参数 ===")
    for hd in [5, 7]:
        for risk in [0.05, 0.08, 0.10]:
            for sell_risk in [0.08, 0.12, 0.15]:
                params = {
                    'leverage': 5, 'hold_days': hd,
                    'buy_opt_score': 60, 'sell_opt_score': 30,
                    'fut_score': 10, 'min_mom': 0.02,
                    'otm_pct': 0.02, 'risk_pct': risk, 'sell_risk': sell_risk,
                    'max_pos': 3,
                }
                r = run_backtest(data, start_date, end_date, params)
                if r:
                    results.append(r)

    # === 纯买期权 (高信号) ===
    print("\n=== 纯买期权参数 ===")
    for hd in [5, 7]:
        for risk in [0.03, 0.05, 0.08]:
            params = {
                'leverage': 0, 'hold_days': hd,
                'buy_opt_score': 40, 'sell_opt_score': 100,  # 不卖期权
                'fut_score': 100,  # 不做期货
                'min_mom': 0.02, 'otm_pct': 0.0,
                'risk_pct': risk, 'sell_risk': 0, 'max_pos': 3,
            }
            r = run_backtest(data, start_date, end_date, params)
            if r:
                results.append(r)

    # === 输出 ===
    results.sort(key=lambda x: x['annual'], reverse=True)

    print(f"\n\n{'杠杆':>4} {'持有':>4} {'买期权':>4} {'卖期权':>4} {'风险买':>4} {'风险卖':>4} "
          f"{'年化':>8} {'胜率':>6} {'盈亏比':>6} {'回撤':>8} {'交易':>4} {'最终':>14}")
    print("-" * 110)

    for r in results[:40]:
        ts = r.get('trade_stats', {})
        print(f"{r.get('leverage',0):>4} {r.get('hold_days',0):>4} "
              f"{r.get('buy_opt_score',0):>4} {r.get('sell_opt_score',0):>4} "
              f"{r.get('risk_pct',0):>4.0%} {r.get('sell_risk',0):>4.0%} "
              f"{r['annual']:>8.1%} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['mdd']:>8.1%} {r['trades']:>4} {r['final']:>14,.0f}")

    # 策略类型统计
    print("\n\n=== 交易类型分布 ===")
    for r in sorted(results, key=lambda x: x['annual'], reverse=True)[:5]:
        ts = r.get('trade_stats', {})
        print(f"  年化={r['annual']:.1%}: 买期权={ts.get('buy_opt',0)} "
              f"卖期权={ts.get('sell_opt',0)} 期货={ts.get('futures',0)}")

    # 筛选
    print("\n\n=== 目标: 年化>=300% 且 WR>=50% ===")
    target = [r for r in results if r['annual'] >= 3.0 and r['wr'] >= 0.50]
    if target:
        for r in sorted(target, key=lambda x: x['annual'], reverse=True)[:10]:
            ts = r.get('trade_stats', {})
            print(f"  年化={r['annual']:.1%}  胜率={r['wr']:.1%}  回撤={r['mdd']:.1%}  "
                  f"交易: 买={ts.get('buy_opt',0)} 卖={ts.get('sell_opt',0)} 期={ts.get('futures',0)}")
    else:
        print("无满足条件的组合")

    print("\n=== WR>=50% 按年化排序 TOP10 ===")
    wr50 = [r for r in results if r['wr'] >= 0.50]
    if wr50:
        for r in sorted(wr50, key=lambda x: x['annual'], reverse=True)[:10]:
            ts = r.get('trade_stats', {})
            print(f"  年化={r['annual']:.1%}  胜率={r['wr']:.1%}  回撤={r['mdd']:.1%}  "
                  f"盈亏比={r['pf']:.2f}  交易={r['trades']}  "
                  f"买={ts.get('buy_opt',0)} 卖={ts.get('sell_opt',0)} 期={ts.get('futures',0)}")
    else:
        print("无")


if __name__ == '__main__':
    main()
