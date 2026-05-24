"""
回测引擎 - 执行策略回测并生成绩效报告

资金模型:
  - 做多: cash -= 买入成本, position = +N, equity = cash + N * price
  - 做空: cash += 卖出收入, position = -N, equity = cash - N * price
  - 平仓: cash += 卖出收入/减去买回成本, position = 0
  - equity = cash + position * current_price (统一公式)
"""
import pandas as pd
import numpy as np
from datetime import timedelta
from .performance import PerformanceAnalyzer
from pandas import Timestamp
import time


class BacktestEngine:
    def __init__(self, data: pd.DataFrame, strategy_class, initial_cash=1e6, commission=0.0003):
        self.data = data.copy()
        self.strategy_class = strategy_class
        self.initial_cash = float(initial_cash)
        self.commission = commission
        self.trades = []
        self.max_position_value = self.initial_cash * 10

        # 状态跟踪
        self.current_holding = None
        self.current_position = 0    # 正=多头股数, 负=空头股数, 0=空仓
        self.current_cash = self.initial_cash
        self.entry_price = 0.0
        self.entry_commission = 0.0
        self.cumulative_profit = 0.0

    def run_backtest(self, strategy_params: dict):
        """执行回测"""
        self.start_time = time.time()

        print(f"开始回测")
        print(f"初始资金: {self.initial_cash:,.2f}")
        print(f"回测时间范围: {self.data.index.min()} 到 {self.data.index.max()}")
        print(f"策略参数: {strategy_params}")
        print("-" * 80)

        # 重置状态
        self.current_holding = None
        self.current_position = 0
        self.current_cash = self.initial_cash
        self.entry_price = 0.0
        self.entry_commission = 0.0
        self.cumulative_profit = 0.0
        self.trades = []

        # 生成信号
        strategy = self.strategy_class(self.data, strategy_params)
        signals = strategy.generate_signals()

        if not signals:
            print("警告: 未生成任何交易信号")
            return self._generate_empty_results()

        print(f"生成 {len(signals)} 个交易信号")
        signals = sorted(signals, key=lambda x: x['timestamp'])

        print("开始执行交易...")

        # 执行所有交易
        for i, signal in enumerate(signals):
            if i % 10 == 0 or i == len(signals) - 1:
                progress = (i / len(signals)) * 100
                print(f"交易进度: {progress:.1f}% ({i}/{len(signals)}) | 累计盈亏: {self.cumulative_profit:+.2f}")

            timestamp = signal['timestamp']
            action = signal['action']
            symbol = signal['symbol']

            # 找到对应时间点的股票数据
            time_data = self.data.loc[timestamp]
            if isinstance(time_data, pd.Series):
                time_data = pd.DataFrame([time_data])

            stock_data = time_data[time_data['symbol'] == symbol]
            if len(stock_data) == 0:
                print(f"警告: 找不到股票 {symbol} 在时间 {timestamp} 的数据")
                continue

            stock_data = stock_data.iloc[0]
            price = stock_data['open']

            # 执行交易
            if action == 'buy':
                self._execute_buy(timestamp, symbol, price)
            elif action == 'sell':
                self._execute_sell(timestamp, symbol, price)

        # 构建资金曲线
        equity_curve = self._build_equity_curve()

        total_time = time.time() - self.start_time
        print(f"\n回测完成! 总耗时: {total_time:.2f}秒")
        print(f"总交易笔数: {len(self.trades)}")
        print(f"最终累计盈亏: {self.cumulative_profit:+.2f}")
        print(f"最终现金: {self.current_cash:,.2f}")

        final_equity = self._calculate_final_equity()
        print(f"最终净值: {final_equity:,.2f}")

        # 绩效分析
        analyzer = PerformanceAnalyzer(
            equity_curve=equity_curve,
            timestamps=equity_curve.index,
            trades=self.trades,
            initial_cash=self.initial_cash
        )
        report = analyzer.generate_report()

        return {
            'equity_curve': equity_curve,
            'trades_list': self.trades,
            **report
        }

    def _execute_buy(self, timestamp, symbol, price):
        """执行买入操作"""
        if self.current_holding is not None and self.current_holding != symbol:
            return  # 已有其他持仓

        commission_rate = self.commission

        if self.current_position < 0 and self.current_holding == symbol:
            # === 平空仓 ===
            shares_to_cover = abs(self.current_position)
            cover_cost = shares_to_cover * price * (1 + commission_rate)
            commission_total = price * commission_rate * shares_to_cover

            # 盈亏 = (卖出价 - 买回价) * 股数 - 手续费
            entry_comm_total = self.entry_commission * shares_to_cover
            exit_comm_total = commission_total
            profit = (self.entry_price - price) * shares_to_cover - entry_comm_total - exit_comm_total

            self.current_cash -= cover_cost
            self.cumulative_profit += profit
            self.current_position = 0
            self.current_holding = None

            self.trades.append({
                'timestamp': pd.Timestamp(timestamp),
                'action': 'cover_short',
                'symbol': symbol,
                'price': price,
                'shares': shares_to_cover,
                'commission': commission_total,
                'cash_before': self.current_cash + cover_cost,
                'cash_after': self.current_cash,
                'profit': profit
            })
            print(f"[{timestamp}] 平空仓 {symbol}: {shares_to_cover}股 @ {price:.2f}, "
                  f"盈亏: {profit:+.2f}")

        else:
            # === 开多仓 ===
            max_shares = int(self.current_cash / (price * (1 + commission_rate)))
            max_shares_limit = int(self.max_position_value / price)
            shares_to_trade = min(max_shares, max_shares_limit)
            if shares_to_trade == 0:
                return

            cost = shares_to_trade * price * (1 + commission_rate)
            commission_total = price * commission_rate * shares_to_trade

            self.current_cash -= cost
            self.current_position = shares_to_trade
            self.current_holding = symbol
            self.entry_price = price
            self.entry_commission = price * commission_rate

            self.trades.append({
                'timestamp': pd.Timestamp(timestamp),
                'action': 'buy',
                'symbol': symbol,
                'price': price,
                'shares': shares_to_trade,
                'commission': commission_total,
                'cash_before': self.current_cash + cost,
                'cash_after': self.current_cash,
                'profit': 0.0
            })
            print(f"[{timestamp}] 开多仓 {symbol}: {shares_to_trade}股 @ {price:.2f}")

    def _execute_sell(self, timestamp, symbol, price):
        """执行卖出操作"""
        commission_rate = self.commission

        if self.current_holding is None:
            # === 开空仓 ===
            max_shares = int(self.current_cash / (price * (1 + commission_rate)))
            max_shares_limit = int(self.max_position_value / price)
            shares_to_trade = min(max_shares, max_shares_limit)
            if shares_to_trade == 0:
                return

            # 做空: 收到卖出收入
            revenue = shares_to_trade * price * (1 - commission_rate)
            commission_total = price * commission_rate * shares_to_trade

            self.current_cash += revenue
            self.current_position = -shares_to_trade
            self.current_holding = symbol
            self.entry_price = price
            self.entry_commission = price * commission_rate

            self.trades.append({
                'timestamp': pd.Timestamp(timestamp),
                'action': 'short',
                'symbol': symbol,
                'price': price,
                'shares': shares_to_trade,
                'commission': commission_total,
                'cash_before': self.current_cash - revenue,
                'cash_after': self.current_cash,
                'profit': 0.0
            })
            print(f"[{timestamp}] 开空仓 {symbol}: {shares_to_trade}股 @ {price:.2f}")

        elif self.current_holding == symbol and self.current_position > 0:
            # === 平多仓 ===
            shares_to_trade = self.current_position
            revenue = shares_to_trade * price * (1 - commission_rate)
            commission_total = price * commission_rate * shares_to_trade

            entry_comm_total = self.entry_commission * shares_to_trade
            exit_comm_total = commission_total
            profit = (price - self.entry_price) * shares_to_trade - entry_comm_total - exit_comm_total

            self.current_cash += revenue
            self.cumulative_profit += profit
            self.current_position = 0
            self.current_holding = None

            self.trades.append({
                'timestamp': pd.Timestamp(timestamp),
                'action': 'sell',
                'symbol': symbol,
                'price': price,
                'shares': shares_to_trade,
                'commission': commission_total,
                'cash_before': self.current_cash - revenue,
                'cash_after': self.current_cash,
                'profit': profit
            })
            print(f"[{timestamp}] 平多仓 {symbol}: {shares_to_trade}股 @ {price:.2f}, "
                  f"盈亏: {profit:+.2f}")

    def _build_equity_curve(self):
        """构建资金曲线: equity = cash + position * close_price"""
        print("构建资金曲线...")

        unique_times = sorted(self.data.index.unique())
        equity_values = []

        # 按顺序回放交易，逐日计算净值
        trade_idx = 0
        sorted_trades = sorted(self.trades, key=lambda t: t['timestamp'])

        current_cash = self.initial_cash
        current_position = 0
        current_holding = None

        for timestamp in unique_times:
            # 处理该时间点的所有交易
            while trade_idx < len(sorted_trades) and sorted_trades[trade_idx]['timestamp'] <= timestamp:
                trade = sorted_trades[trade_idx]
                if trade['timestamp'] == timestamp:
                    current_cash = trade['cash_after']
                    action = trade['action']
                    if action == 'buy':
                        current_position = trade['shares']
                        current_holding = trade['symbol']
                    elif action == 'sell':
                        current_position = 0
                        current_holding = None
                    elif action == 'short':
                        current_position = -trade['shares']
                        current_holding = trade['symbol']
                    elif action == 'cover_short':
                        current_position = 0
                        current_holding = None
                trade_idx += 1

            # 计算当日净值: equity = cash + position * close
            if current_position != 0 and current_holding:
                time_data = self.data.loc[timestamp]
                if isinstance(time_data, pd.Series):
                    close = time_data['close']
                else:
                    holding_data = time_data[time_data['symbol'] == current_holding]
                    close = holding_data.iloc[0]['close'] if len(holding_data) > 0 else 0
                portfolio_value = current_cash + current_position * close
            else:
                portfolio_value = current_cash

            equity_values.append(portfolio_value)

        equity_curve = pd.Series(equity_values, index=unique_times)
        print(f"资金曲线构建完成，最终净值: {equity_curve.iloc[-1]:,.2f}")
        return equity_curve

    def _calculate_final_equity(self):
        """计算最终净值"""
        final_equity = self.current_cash
        if self.current_holding and self.current_position != 0:
            last_time = self.data.index.max()
            time_data = self.data.loc[last_time]
            if isinstance(time_data, pd.Series):
                price = time_data['close']
            else:
                holding_data = time_data[time_data['symbol'] == self.current_holding]
                price = holding_data.iloc[0]['close'] if len(holding_data) > 0 else 0
            final_equity = self.current_cash + self.current_position * price
        return final_equity

    def _generate_empty_results(self):
        """生成空结果"""
        empty_curve = pd.Series([self.initial_cash], index=[self.data.index[0]])
        return {
            'equity_curve': empty_curve,
            'trades_list': [],
            'summary': {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'trade_count': 0,
                'date_range': f"{self.data.index.min()} 到 {self.data.index.max()}"
            },
            'drawdown': {'max_drawdown': 0.0, 'drawdown_series': empty_curve},
            'risk_return': {'sharpe_ratio': 0.0},
            'trades': {'win_rate': 0.0, 'profit_factor': 0.0}
        }

    def save_trades(self, filename: str):
        if not self.trades:
            print("警告: 无交易记录可保存")
            return
        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv(filename, index=False)
        print(f"交易记录已保存到: {filename}, 共 {len(self.trades)} 笔交易")
