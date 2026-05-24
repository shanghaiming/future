"""
绩效分析模块 - 回测绩效指标计算
"""
import pandas as pd
import numpy as np
from typing import List, Dict

class PerformanceAnalyzer:
    """
    独立绩效计算模块
    职责: 从回测引擎接收原始数据，计算所有绩效指标
    """

    def __init__(self,
                 equity_curve: pd.Series,
                 timestamps: pd.DatetimeIndex,
                 trades: List[Dict],
                 initial_cash: float):
        self.equity = equity_curve
        self.timestamps = timestamps
        self.trades = trades
        self.initial_cash = initial_cash

        # 预处理数据
        self._returns = self._calculate_returns()

    def generate_report(self) -> Dict:
        """生成完整绩效报告"""
        return {
            'summary': self._summary_metrics(),
            'drawdown': self._drawdown_analysis(),
            'risk_return': self._risk_return_metrics(),
            'trades': self._trade_analytics()
        }

    def _calculate_returns(self) -> pd.Series:
        """计算日收益率序列"""
        return self.equity.pct_change().dropna()

    def _summary_metrics(self) -> Dict:
        """核心绩效指标"""
        total_return = self.equity.iloc[-1] / self.initial_cash - 1
        annualized_return = (1 + self._returns.mean()) ** 252 - 1

        start_date = self.timestamps[0].strftime("%Y-%m-%d")
        end_date = self.timestamps[-1].strftime("%Y-%m-%d")

        return {
            'initial_cash': self.initial_cash,
            'final_equity': self.equity.iloc[-1],
            'total_return': total_return,
            'date_range': f"{start_date} 至 {end_date}",
            'annualized_return': annualized_return,
            'trade_count': len(self.trades)
        }

    def _drawdown_analysis(self) -> Dict:
        """回撤分析"""
        peak = self.equity.cummax()
        drawdown_series = (self.equity - peak) / peak

        return {
            'max_drawdown': drawdown_series.min(),
            'drawdown_duration': self._max_drawdown_duration(),
            'drawdown_series': drawdown_series
        }

    def _max_drawdown_duration(self) -> pd.Timedelta:
        """计算最长回撤持续时间"""
        underwater = (self.equity == self.equity.cummax()).astype(int)
        durations = underwater.groupby((underwater.diff() != 0).cumsum()).cumcount() + 1
        return pd.to_timedelta(durations.max(), unit='days')

    def _risk_return_metrics(self) -> Dict:
        """风险收益指标"""
        sharpe = self._sharpe_ratio()
        sortino = self._sortino_ratio()

        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'volatility': self._returns.std() * np.sqrt(252)
        }

    def _sharpe_ratio(self, risk_free_rate=0) -> float:
        """夏普比率"""
        excess_returns = self._returns - risk_free_rate/252
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    def _sortino_ratio(self, risk_free_rate=0) -> float:
        """索提诺比率"""
        excess_returns = self._returns - risk_free_rate/252
        downside_returns = excess_returns[excess_returns < 0]
        return excess_returns.mean() / downside_returns.std() * np.sqrt(252)

    def _trade_analytics(self) -> Dict:
        """交易分析"""
        sell_trades = [
            t for t in self.trades
            if isinstance(t, dict) and
                t.get('action', '').lower() == 'sell' and
                isinstance(t.get('profit', 0), (int, float))
        ]

        if not sell_trades:
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                '_warning': '无有效卖出交易记录'
            }

        profits = [float(t['profit']) for t in sell_trades if isinstance(t.get('profit'), (int, float))]

        wins = [p for p in profits if p > 0]
        losses = [abs(p) for p in profits if p < 0]

        sum_wins = sum(wins) if wins else 0.0
        sum_losses = sum(losses) if losses else 0.0
        profit_factor = sum_wins / sum_losses if sum_losses != 0 else np.inf

        return {
            'win_rate': len(wins) / len(sell_trades) if len(sell_trades) > 0 else 0.0,
            'profit_factor': profit_factor,
            'avg_win': np.mean(wins) if wins else 0.0,
            'avg_loss': np.mean(losses) if losses else 0.0
        }

    def save_trades_to_csv(self, filename: str) -> None:
        """保存交易记录到CSV文件"""
        if not self.trades:
            raise ValueError("无交易记录可保存")

        df = pd.DataFrame(self.trades)
        df['profit_pct'] = df['profit'] / df['price'] / df['shares']
        df['hold_days'] = (df['timestamp'] - df['timestamp'].shift(1)).dt.days.where(df['action'] == 'sell')
        df = df.sort_values('timestamp').reset_index(drop=True)
        df.to_csv(filename, index=False,
                columns=['timestamp', 'action', 'price', 'shares', 'profit', 'profit_pct', 'hold_days'])
