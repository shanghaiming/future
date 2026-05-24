#!/usr/bin/env python3
"""
期货回测引擎 - 考虑合约乘数、保证金、交易成本
================================================
用合约乘数计算真实盈亏金额, 不加杠杆放大。

核心公式:
  盈亏 = (exit_price - entry_price) × direction × 乘数 × 手数
  合约价值 = 价格 × 乘数 × 手数
  保证金 = 合约价值 × 保证金率
  资金收益率 = 盈亏 / 占用保证金 (不加额外杠杆)

用法:
  from backtest_engine import BacktestEngine
  engine = BacktestEngine(initial_capital=1_000_000)
  trade = engine.execute_trade('safi', 1, 2500, 2600, '2026-05-20', '2026-05-21', lots=10)
"""

import os
import sys
import math
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

# 合约规格
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contract_specs import CONTRACT_SPECS, DEFAULT_SPEC, get_spec

# ---------------------------------------------------------------------------
# 交易成本参数
# ---------------------------------------------------------------------------
COMMISSION_RATE = 0.00005   # 手续费率 (双边, 占合约价值比例)
SLIPPAGE_TICKS = 1          # 滑点 (tick数)


def get_commission(symbol: str) -> float:
    return COMMISSION_RATE


def get_slippage(symbol: str) -> float:
    _, _, tick_size, _ = get_spec(symbol)
    return tick_size * SLIPPAGE_TICKS


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------
@dataclass
class TradeRecord:
    symbol: str
    direction: int
    entry_price: float
    exit_price: float
    entry_date: str
    exit_date: str
    lots: int
    multiplier: float
    margin_rate: float
    notional_ret: float
    capital_ret: float
    pnl: float
    commission: float
    slippage: float
    net_pnl: float
    leverage: float
    holding_days: int


# ---------------------------------------------------------------------------
# 回测引擎
# ---------------------------------------------------------------------------
class BacktestEngine:
    """
    考虑合约乘数和保证金的回测引擎

    功能:
      1. 根据合约乘数和保证金率计算真实杠杆
      2. 跟踪资金曲线 (考虑保证金占用)
      3. 计算包含交易成本的净收益
      4. 输出详细的杠杆和资金使用报告
    """

    def __init__(self,
                 initial_capital: float = 1_000_000,
                 max_equity_pct: float = 0.30,
                 target_leverage: Optional[float] = None):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_equity_pct = max_equity_pct
        self.target_leverage = target_leverage
        self.trades: List[TradeRecord] = []

    def calc_lots(self, symbol: str, price: float, n_positions: int = 1) -> int:
        multiplier, margin_rate, _, _ = get_spec(symbol)
        if self.target_leverage is not None:
            target_notional = self.capital * self.target_leverage / n_positions
            lots = int(target_notional / (price * multiplier))
        else:
            capital_per_pos = self.capital * self.max_equity_pct / n_positions
            margin_per_lot = price * multiplier * margin_rate
            lots = int(capital_per_pos / margin_per_lot)
        return max(lots, 1)

    def get_leverage_info(self, symbol: str, price: float, lots: int = 1) -> Dict:
        multiplier, margin_rate, tick_size, name = get_spec(symbol)
        notional = price * multiplier * lots
        margin = notional * margin_rate
        return {
            'symbol': symbol, 'name': name,
            'multiplier': multiplier, 'margin_rate': margin_rate,
            'tick_size': tick_size, 'price': price, 'lots': lots,
            'notional_value': round(notional, 2),
            'margin': round(margin, 2),
            'leverage': round(1 / margin_rate, 1),
            'commission_per_side': round(notional * COMMISSION_RATE, 2),
            'slippage_cost': round(tick_size * SLIPPAGE_TICKS * multiplier * lots, 2),
        }

    def execute_trade(self,
                      symbol: str,
                      direction: int,
                      entry_price: float,
                      exit_price: float,
                      entry_date: str,
                      exit_date: str,
                      lots: Optional[int] = None,
                      n_positions: int = 1) -> TradeRecord:
        multiplier, margin_rate, tick_size, _ = get_spec(symbol)
        if lots is None:
            lots = self.calc_lots(symbol, entry_price, n_positions)

        notional_ret = (exit_price - entry_price) / entry_price * direction
        leverage = 1.0 / margin_rate
        capital_ret = notional_ret * leverage

        pnl = (exit_price - entry_price) * direction * multiplier * lots
        commission = (entry_price * multiplier * lots * COMMISSION_RATE +
                      exit_price * multiplier * lots * COMMISSION_RATE)
        slippage = tick_size * SLIPPAGE_TICKS * multiplier * lots * 2
        net_pnl = pnl - commission - slippage

        try:
            holding_days = (pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days
        except Exception:
            holding_days = 0

        record = TradeRecord(
            symbol=symbol, direction=direction,
            entry_price=entry_price, exit_price=exit_price,
            entry_date=entry_date, exit_date=exit_date,
            lots=lots, multiplier=multiplier, margin_rate=margin_rate,
            notional_ret=round(notional_ret, 6),
            capital_ret=round(capital_ret, 6),
            pnl=round(pnl, 2),
            commission=round(commission, 2),
            slippage=round(slippage, 2),
            net_pnl=round(net_pnl, 2),
            leverage=round(leverage, 1),
            holding_days=holding_days,
        )
        self.trades.append(record)
        self.capital += net_pnl
        return record

    def notional_to_capital_return(self, symbol: str,
                                    notional_ret: float,
                                    direction: int = 1) -> Tuple[float, float]:
        _, margin_rate, _, _ = get_spec(symbol)
        leverage = 1.0 / margin_rate
        capital_ret = notional_ret * direction * leverage
        return capital_ret, leverage

    def batch_convert(self, trade_list: List[Dict],
                      leverage_target: Optional[float] = None) -> pd.DataFrame:
        results = []
        for t in trade_list:
            sym = t['symbol']
            notional_ret = t['notional_ret']
            direction = t.get('direction', 1)
            _, margin_rate, tick_size, name = get_spec(sym)
            leverage = 1.0 / margin_rate
            if leverage_target is not None:
                leverage = leverage_target
            capital_ret = notional_ret * direction * leverage
            r = {
                'symbol': sym, 'name': name,
                'notional_ret': notional_ret,
                'direction': direction,
                'margin_rate': margin_rate,
                'leverage': leverage,
                'capital_ret': capital_ret,
            }
            if 'entry_price' in t:
                r['entry_price'] = t['entry_price']
            if 'exit_price' in t:
                r['exit_price'] = t['exit_price']
            results.append(r)
        return pd.DataFrame(results)

    def portfolio_stats(self) -> Dict:
        if not self.trades:
            return {}
        df = pd.DataFrame([vars(t) for t in self.trades])
        total_pnl = df['pnl'].sum()
        total_commission = df['commission'].sum()
        total_slippage = df['slippage'].sum()
        total_net_pnl = df['net_pnl'].sum()
        by_symbol = df.groupby('symbol').agg({
            'net_pnl': 'sum', 'pnl': 'sum', 'commission': 'sum',
            'leverage': 'first', 'lots': 'mean',
            'notional_ret': 'mean', 'capital_ret': 'mean',
        }).to_dict('index')
        return {
            'initial_capital': self.initial_capital,
            'final_capital': round(self.capital, 2),
            'total_return': round((self.capital / self.initial_capital - 1) * 100, 2),
            'total_trades': len(self.trades),
            'total_pnl': round(total_pnl, 2),
            'total_commission': round(total_commission, 2),
            'total_slippage': round(total_slippage, 2),
            'total_net_pnl': round(total_net_pnl, 2),
            'avg_leverage': round(df['leverage'].mean(), 1),
            'avg_holding_days': round(df['holding_days'].mean(), 1),
            'by_symbol': by_symbol,
        }


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def convert_strategy_returns(symbol_returns: Dict[str, float],
                              direction: int = 1) -> Dict:
    results = {}
    for sym, ret in symbol_returns.items():
        _, margin_rate, _, name = get_spec(sym)
        leverage = 1.0 / margin_rate
        capital_ret = ret * direction * leverage
        results[sym] = {
            'name': name,
            'margin_rate': margin_rate,
            'leverage': leverage,
            'notional_ret': round(ret * 100, 2),
            'capital_ret': round(capital_ret * 100, 2),
        }
    return results


def print_leverage_table():
    print(f"\n{'品种':<8} {'名称':<8} {'乘数':<8} {'保证金率':<10} {'杠杆倍数':<10}")
    print("-" * 50)

    exchanges = {
        '上期所 (SHFE)': ['agfi', 'alfi', 'aufi', 'bufi', 'cufi', 'fufi', 'hcfi',
                         'nifi', 'pbfi', 'rbfi', 'rufi', 'snfi', 'sffi', 'smfi',
                         'spfi', 'znfi', 'ssfi'],
        '大商所 (DCE)': ['afi', 'bfi', 'cffi', 'cfi', 'csfi', 'ebfi', 'egfi',
                         'ifi', 'jfi', 'jmfi', 'lfi', 'mfi', 'pgfi', 'ppfi',
                         'vfi', 'yfi', 'pfi', 'mafi'],
        '郑商所 (CZCE)': ['apfi', 'cyfi', 'fgfi', 'oifi', 'pfifi', 'rmfi', 'srfi',
                         'tafi', 'safi', 'urfi'],
        '能源中心 (INE)': ['scfi', 'lufi', 'bcfi'],
        '广期所 (GFEX)': ['lcfi', 'sifi'],
    }

    for exchange, symbols in exchanges.items():
        print(f"\n--- {exchange} ---")
        for sym in symbols:
            if sym in CONTRACT_SPECS:
                mult, mr, ts, name = CONTRACT_SPECS[sym]
                lev = 1.0 / mr
                print(f"{sym:<8} {name:<8} {mult:<8} {mr*100:>6.0f}%     {lev:>6.1f}x")

    print(f"\n默认 (未在表中): 乘数=10, 保证金=10%, 杠杆=10x")


def estimate_portfolio_leverage(symbols: List[str],
                                 capital: float,
                                 equity_pct: float = 0.30) -> Dict:
    total_notional = 0
    total_margin = 0
    details = []
    for sym in symbols:
        _, margin_rate, _, name = get_spec(sym)
        margin_allocated = capital * equity_pct
        leverage = 1.0 / margin_rate
        notional = margin_allocated / margin_rate
        total_notional += notional
        total_margin += margin_allocated
        details.append({
            'symbol': sym, 'name': name,
            'margin_rate': margin_rate, 'leverage': leverage,
            'margin_allocated': round(margin_allocated, 0),
            'notional': round(notional, 0),
        })
    portfolio_leverage = total_notional / capital if capital > 0 else 0
    return {
        'capital': capital,
        'n_positions': len(symbols),
        'total_margin': round(total_margin, 0),
        'total_notional': round(total_notional, 0),
        'portfolio_leverage': round(portfolio_leverage, 1),
        'margin_usage_pct': round(total_margin / capital * 100, 1),
        'details': details,
    }


def enhance_backtest_result(result: Dict,
                             symbols_traded: List[str],
                             holding_period: int = 10) -> Dict:
    if not symbols_traded:
        result['leverage_info'] = '无交易品种'
        return result

    portfolio_lev = estimate_portfolio_leverage(symbols_traded, capital=1_000_000)
    avg_leverage = np.mean([1.0 / get_spec(s)[1] for s in symbols_traded])

    ann_ret_notional = result.get('ann_return', result.get('CAGR', 0))
    if isinstance(ann_ret_notional, str):
        ann_ret_notional = float(str(ann_ret_notional).replace('%', '')) / 100

    enhanced = result.copy()
    enhanced['leverage_info'] = {
        'avg_single_leverage': round(avg_leverage, 1),
        'portfolio_leverage': portfolio_lev['portfolio_leverage'],
        'margin_usage_pct': portfolio_lev['margin_usage_pct'],
        'n_positions': len(symbols_traded),
        'leverage_adjusted_ann_return': round(ann_ret_notional * portfolio_lev['portfolio_leverage'] * 100, 2),
    }
    return enhanced


if __name__ == '__main__':
    print("=" * 60)
    print("期货回测引擎 - 合约乘数与杠杆计算")
    print("=" * 60)

    # 品种杠杆表
    print_leverage_table()

    # 单品种示例
    print("\n" + "=" * 60)
    print("示例: 纯碱(SA) 做多 10手")
    engine = BacktestEngine(initial_capital=1_000_000)
    info = engine.get_leverage_info('safi', 2500, lots=10)
    print(f"  合约乘数: {info['multiplier']} 吨/手")
    print(f"  单品种杠杆: {info['leverage']}x")
    print(f"  10手名义价值: {2500 * 20 * 10:,.0f} 元")
    print(f"  10手保证金: {2500 * 20 * 10 * 0.08:,.0f} 元")

    trade = engine.execute_trade('safi', 1, 2500, 2600, '2026-05-20', '2026-05-21', lots=10)
    print(f"\n  价格 2500→2600 (涨4%):")
    print(f"  名义收益率: {trade.notional_ret*100:.2f}%")
    print(f"  实际杠杆: {trade.leverage}x")
    print(f"  资金收益率: {trade.capital_ret*100:.2f}%")
    print(f"  盈亏: {trade.pnl:,.0f} 元 (手续费{trade.commission:.0f} 滑点{trade.slippage:.0f})")
    print(f"  净盈亏: {trade.net_pnl:,.0f} 元")

    # 组合杠杆
    print("\n" + "=" * 60)
    print("示例: 10品种组合杠杆估算")
    portfolio_symbols = ['safi', 'rbfi', 'ifi', 'aufi', 'mfi', 'oifi', 'hcfi', 'cufi', 'egfi', 'tafi']
    port_info = estimate_portfolio_leverage(portfolio_symbols, 1_000_000)
    print(f"  组合杠杆: {port_info['portfolio_leverage']}x")
    print(f"  保证金占用: {port_info['margin_usage_pct']}%")
    print(f"  总名义价值: {port_info['total_notional']:,.0f} 元")
