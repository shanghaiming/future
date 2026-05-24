#!/usr/bin/env python3
"""
策略合约乘数报告 - 用乘数计算真实盈亏
========================================
不加杠杆, 但计算盈亏时使用合约乘数:
  盈亏(元) = (平仓价 - 开仓价) × 方向 × 乘数 × 手数
  保证金(元) = 价格 × 乘数 × 手数 × 保证金率
  资金收益率 = 盈亏 / 保证金

用法: python3 scripts/strategy_leverage_report.py
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contract_specs import CONTRACT_SPECS, get_spec

BASE_DIR = os.path.expanduser("~/home/futures_platform")
INITIAL_CAPITAL = 1_000_000
EQUITY_USAGE_PCT = 0.30
COMMISSION_RATE = 0.0001  # 万1 双边

TYPICAL_SYMBOLS = list(CONTRACT_SPECS.keys())


def calc_average_margin_rate(symbols=None):
    if not symbols:
        symbols = TYPICAL_SYMBOLS
    rates = [get_spec(s)[1] for s in symbols if s in CONTRACT_SPECS]
    return np.mean(rates) if rates else 0.10


def load_sample_prices():
    """读取各品种最新价格"""
    prices = {}
    price_dir = os.path.join(BASE_DIR, "data/futures_weighted")
    for sym in TYPICAL_SYMBOLS:
        fp = os.path.join(price_dir, f"{sym}.csv")
        if os.path.exists(fp):
            try:
                df = pd.read_csv(fp)
                if len(df) > 0:
                    # 最新数据在文件头部 (第一行)
                    val = float(df.iloc[0]['close'])
                    if not (np.isnan(val) or val <= 0):
                        prices[sym] = val
            except Exception:
                pass
    return prices


def calc_pnl_example(symbol, direction, entry_price, exit_price, lots):
    """
    用合约乘数计算真实盈亏

    Args:
        symbol: 品种代码 (如 'safi')
        direction: 1=多, -1=空
        entry_price: 开仓价
        exit_price: 平仓价
        lots: 手数

    Returns:
        dict: 盈亏详情
    """
    multiplier, margin_rate, tick_size, name = get_spec(symbol)

    # 盈亏 = (平仓价 - 开仓价) × 方向 × 乘数 × 手数
    pnl = (exit_price - entry_price) * direction * multiplier * lots

    # 合约价值
    entry_value = entry_price * multiplier * lots
    exit_value = exit_price * multiplier * lots

    # 保证金
    margin = entry_value * margin_rate

    # 手续费 (双边)
    commission = (entry_value + exit_value) * COMMISSION_RATE

    # 滑点 (1个tick × 开平各一次)
    slippage = tick_size * multiplier * lots * 2

    # 净盈亏
    net_pnl = pnl - commission - slippage

    # 收益率 (相对保证金)
    ret_on_margin = net_pnl / margin if margin > 0 else 0

    # 名义收益率 (价格变动百分比)
    notional_ret = (exit_price - entry_price) / entry_price * direction

    return {
        'symbol': symbol,
        'name': name,
        'multiplier': multiplier,
        'margin_rate': margin_rate,
        'direction': '多' if direction == 1 else '空',
        'lots': lots,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'notional_ret': round(notional_ret * 100, 2),         # 价格变动%
        'pnl': round(pnl, 2),                                  # 盈亏金额(元)
        'entry_value': round(entry_value, 2),                  # 开仓合约价值
        'margin': round(margin, 2),                             # 保证金
        'commission': round(commission, 2),                     # 手续费
        'slippage': round(slippage, 2),                         # 滑点
        'net_pnl': round(net_pnl, 2),                           # 净盈亏
        'ret_on_margin': round(ret_on_margin * 100, 2),        # 保证金收益率%
    }


def calc_strategy_pnl(strategy_name, notional_ann_ret, symbols, n_positions,
                       holding_period, prices):
    """
    用合约乘数计算策略年化盈亏金额

    策略报告的是名义收益率 (价格变动百分比),
    这里转换为实际盈亏金额:
      每笔盈亏 = 名义收益率 × 合约价值
      年化盈亏 = 平均每笔盈亏 × 年交易次数 × 持仓品种数
    """
    if not symbols:
        symbols = TYPICAL_SYMBOLS

    # 每个品种分配的保证金
    margin_per_pos = INITIAL_CAPITAL * EQUITY_USAGE_PCT / n_positions

    # 按品种分别计算
    position_pnls = []
    for sym in symbols[:n_positions]:
        if sym not in CONTRACT_SPECS:
            continue
        mult, mr, ts, name = CONTRACT_SPECS[sym]
        price = prices.get(sym, 0)
        if not price or price <= 0 or (isinstance(price, float) and np.isnan(price)):
            continue

        # 1手保证金
        margin_1lot = price * mult * mr
        # 能开几手
        lots = max(int(margin_per_pos / margin_1lot), 1)
        # 实际保证金
        actual_margin = price * mult * lots * mr
        # 合约价值
        contract_value = price * mult * lots

        # 名义收益率对应的盈亏金额
        pnl_per_period = notional_ann_ret * contract_value * (holding_period / 252)

        # 年交易次数
        trades_per_year = 252 / holding_period

        # 年化盈亏
        annual_pnl = pnl_per_period * trades_per_year

        # 手续费
        commission_per_trade = contract_value * COMMISSION_RATE * 2  # 开+平
        annual_commission = commission_per_trade * trades_per_year

        position_pnls.append({
            'symbol': sym,
            'name': name,
            'multiplier': mult,
            'price': price,
            'lots': lots,
            'contract_value': contract_value,
            'margin': actual_margin,
            'margin_rate': mr,
            'annual_pnl': annual_pnl,
            'annual_commission': annual_commission,
            'ret_on_margin': annual_pnl / actual_margin if actual_margin > 0 else 0,
        })

    if not position_pnls:
        return None

    total_pnl = sum(p['annual_pnl'] for p in position_pnls)
    total_margin = sum(p['margin'] for p in position_pnls)
    total_commission = sum(p['annual_commission'] for p in position_pnls)

    return {
        'positions': position_pnls,
        'total_annual_pnl': total_pnl,
        'total_margin': total_margin,
        'total_commission': total_commission,
        'net_annual_pnl': total_pnl - total_commission,
        'capital_ret': (total_pnl - total_commission) / INITIAL_CAPITAL,
    }


# ---------------------------------------------------------------------------
# 已完成策略结果
# ---------------------------------------------------------------------------
COMPLETED_STRATEGIES = {
    'V95_POI_TS': {
        'desc': 'POI因子+期限结构组合策略',
        'best_config': 'POI_only / h20 / n5',
        'holding_period': 20,
        'test': {'ann_ret': 0.121, 'sharpe': 2.96, 'mdd': -0.005},
        'n_positions': 5,
        'direction': '多+空',
    },
    'V96_Cross_Momentum': {
        'desc': '截面均值回归策略 (5日反转)',
        'best_config': 'plain_reversed / LB5 / K5 / H10',
        'holding_period': 10,
        'test': {'ann_ret': 0.321, 'sharpe': 3.62, 'mdd': -0.041},
        'n_positions': 10,
        'direction': '多+空',
    },
    'V97_Low_Vol_Anomaly': {
        'desc': '低波动率异象策略',
        'best_config': 'HV20底部1/3买入',
        'holding_period': 20,
        'test': {'ann_ret': 0.35, 'sharpe': 6.99, 'mdd': -0.08},
        'n_positions': 10,
        'direction': '做多',
    },
    'V98_Spread_Reversion': {
        'desc': '价差均值回归+成交量过滤',
        'best_config': 'volume_filter / K=15',
        'holding_period': 10,
        'test': {'ann_ret': 0.045, 'sharpe': 0.64, 'mdd': -0.12},
        'n_positions': 15,
        'direction': '多+空',
    },
    'V101_Seasonality': {
        'desc': '月度季节性策略',
        'best_config': 'K=15 月份动量',
        'holding_period': 20,
        'test': {'ann_ret': 0.08, 'sharpe': 2.07, 'mdd': -0.05},
        'n_positions': 15,
        'direction': '多+空',
    },
}


def print_report():
    prices = load_sample_prices()

    print("=" * 80)
    print("期货策略合约乘数报告")
    print("=" * 80)
    print(f"\n初始资金: {INITIAL_CAPITAL:,.0f} 元")
    print(f"保证金占用上限: {EQUITY_USAGE_PCT*100:.0f}%")
    print(f"手续费率: {COMMISSION_RATE*100:.4f}% (双边)")
    print(f"\n不加杠杆, 但盈亏计算使用合约乘数:")

    # 合约规格表
    print(f"\n{'='*80}")
    print(f"合约规格表 (主要品种)")
    print(f"{'='*80}")
    print(f"\n{'代码':<8} {'名称':<6} {'乘数':<6} {'保证金':>6} {'最新价':>10} "
          f"{'1手价值':>12} {'1手保证金':>10} {'1个tick':>8}")
    print("-" * 80)

    sample_symbols = ['rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi', 'mfi', 'yfi',
                      'oifi', 'safi', 'aufi', 'agfi', 'cufi', 'cffi', 'srfi',
                      'tafi', 'egfi', 'pgfi', 'lfi', 'ppfi', 'vfi', 'scfi']

    for sym in sample_symbols:
        if sym in CONTRACT_SPECS:
            mult, mr, ts, name = CONTRACT_SPECS[sym]
            price = prices.get(sym, 0)
            if price > 0:
                value = price * mult
                margin = value * mr
                tick_val = ts * mult  # 1个tick值多少钱
                print(f"{sym:<8} {name:<6} {mult:<6} {mr*100:>5.0f}% {price:>10.1f} "
                      f"{value:>12,.0f} {margin:>10,.0f} {tick_val:>8,.0f}元")

    # 单品种盈亏示例
    print(f"\n{'='*80}")
    print(f"盈亏计算示例 (合约乘数 vs 简单百分比)")
    print(f"{'='*80}")

    examples = [
        ('safi', '纯碱', 1, 0.01),
        ('ifi', '铁矿', 1, 0.01),
        ('aufi', '沪金', 1, 0.01),
        ('rbfi', '螺纹', 1, 0.01),
        ('mfi', '豆粕', 1, 0.01),
    ]

    print(f"\n假设: 各品种均做多1手, 价格涨1%")
    print(f"\n{'品种':<8} {'价格':>8} {'乘数':>5} {'1手价值':>12} {'保证金':>10} "
          f"{'盈亏(元)':>10} {'保证金收益率':>12}")
    print("-" * 80)

    for sym, label, direction, pct_move in examples:
        price = prices.get(sym, 0)
        if price <= 0:
            continue
        r = calc_pnl_example(sym, direction, price, price * (1 + pct_move), lots=1)
        print(f"{sym:<8} {r['entry_price']:>8.0f} {r['multiplier']:>5} "
              f"{r['entry_value']:>12,.0f} {r['margin']:>10,.0f} "
              f"{r['net_pnl']:>+10,.0f} {r['ret_on_margin']:>+11.2f}%")

    # 每个策略的实际盈亏
    print(f"\n{'='*80}")
    print(f"各策略年化盈亏 (使用合约乘数, 1x杠杆)")
    print(f"{'='*80}")

    summary = []
    for name, info in COMPLETED_STRATEGIES.items():
        test = info['test']
        result = calc_strategy_pnl(
            name, test['ann_ret'], TYPICAL_SYMBOLS,
            info['n_positions'], info['holding_period'], prices
        )
        if result is None:
            continue

        print(f"\n--- {name}: {info['desc']} ---")
        print(f"  配置: {info['best_config']}")
        print(f"  方向: {info['direction']}, {info['n_positions']}个品种, {info['holding_period']}天换仓")
        print(f"  名义年化: {test['ann_ret']*100:.2f}% (价格变动百分比)")
        print(f"")
        print(f"  按合约乘数计算:")
        print(f"    总保证金占用: {result['total_margin']:>12,.0f} 元")
        print(f"    年化盈亏(扣费前): {result['total_annual_pnl']:>+12,.0f} 元")
        print(f"    年化手续费:     {-result['total_commission']:>12,.0f} 元")
        print(f"    年化盈亏(扣费后): {result['net_annual_pnl']:>+12,.0f} 元")
        print(f"    占总资金收益率: {result['capital_ret']*100:>+8.2f}%")
        print(f"")
        print(f"  各品种明细:")
        print(f"    {'品种':<6} {'乘数':>4} {'手数':>4} {'价值':>10} {'保证金':>10} {'年化盈亏':>12}")
        for p in result['positions'][:10]:
            print(f"    {p['symbol']:<6} {p['multiplier']:>4} {p['lots']:>4} "
                  f"{p['contract_value']:>10,.0f} {p['margin']:>10,.0f} {p['annual_pnl']:>+12,.0f}")
        if len(result['positions']) > 10:
            print(f"    ... (共{len(result['positions'])}个品种)")

        summary.append({
            'name': name,
            'desc': info['desc'],
            'notional_ret': test['ann_ret'] * 100,
            'sharpe': test['sharpe'],
            'mdd': test['mdd'] * 100,
            'net_pnl': result['net_annual_pnl'],
            'capital_ret': result['capital_ret'] * 100,
            'margin_used': result['total_margin'],
        })

    # 汇总表
    print(f"\n{'='*80}")
    print(f"汇总对比 (1x杠杆, 合约乘数)")
    print(f"{'='*80}")
    print(f"\n{'策略':<22} {'名义年化':>8} {'Sharpe':>7} {'年化盈亏(元)':>14} "
          f"{'占资金%':>8} {'保证金占用':>12} {'MDD%':>7}")
    print("-" * 85)

    for s in summary:
        print(f"{s['name']:<22} {s['notional_ret']:>+7.1f}% {s['sharpe']:>7.2f} "
              f"{s['net_pnl']:>+14,.0f} {s['capital_ret']:>+7.2f}% "
              f"{s['margin_used']:>12,.0f} {s['mdd']:>7.1f}")

    print(f"\n{'='*80}")
    print(f"关键说明")
    print(f"{'='*80}")
    print(f"""
  1. 不加杠杆: 每个品种按保证金约束开仓, 不额外放大
  2. 合约乘数: 盈亏金额 = 价格变动 × 乘数 × 手数
     - 铁矿(IF=100吨/手): 价格涨1元 → 1手赚100元
     - 纯碱(SA=20吨/手):  价格涨1元 → 1手赚20元
     - 螺纹(RB=10吨/手):  价格涨1元 → 1手赚10元
  3. 保证金: 决定开几手。100万资金, 30%保证金分配:
     - 铁矿(~976元): 保证金~11,713元/手 → 2手
     - 纯碱(~2500元): 保证金~4,000元/手 → 6手
  4. 名义收益率 vs 资金收益率:
     - 名义12%年化 → 实际赚多少钱取决于合约乘数
     - 上面表格同时列出名义百分比和实际金额
""")


if __name__ == '__main__':
    print_report()
