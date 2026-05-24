"""
统一策略基类 - 所有策略必须继承此类
信号格式: List[Dict] 每个信号必须包含 {timestamp, action, symbol, price}
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Any


class BaseStrategy(ABC):
    """统一策略基类"""

    SIGNAL_ACTIONS = {'buy', 'sell', 'hold'}

    # 看板元数据（子类可覆盖）
    strategy_description: str = ""
    strategy_category: str = "general"  # ma, price_action, momentum, wave, ml, volume, general
    strategy_params_schema: Dict = {}   # 看板动态渲染参数表单用

    def __init__(self, data: pd.DataFrame, params: dict = None):
        self.data = data.copy()
        self.params = {**self.get_default_params(), **(params or {})}
        self.strategy_name = self.__class__.__name__
        self.signals: List[Dict] = []
        self._validate_data()
        self.validate_params()

    @abstractmethod
    def generate_signals(self) -> List[Dict]:
        """生成交易信号，返回信号列表，每个信号必须包含:
        - timestamp: 信号时间
        - action: 'buy'/'sell'/'hold'
        - symbol: 股票代码
        - price: 信号价格
        """
        pass

    def get_default_params(self) -> Dict[str, Any]:
        """子类重写以提供默认参数"""
        return {}

    def validate_params(self):
        """子类重写以添加参数验证，验证失败抛 ValueError"""
        pass

    def _validate_data(self):
        """验证输入数据"""
        required = ['open', 'high', 'low', 'close']
        missing = [c for c in required if c not in self.data.columns]
        if missing:
            raise ValueError(f"数据缺少必需列: {missing}")

        if self.data.empty:
            raise ValueError("数据为空")

        # 自动注入 symbol 列
        if 'symbol' not in self.data.columns:
            self.data['symbol'] = 'DEFAULT'

        if not pd.api.types.is_datetime64_any_dtype(self.data.index):
            try:
                self.data.index = pd.to_datetime(self.data.index)
            except Exception as e:
                raise ValueError(f"无法将索引转换为datetime: {e}")

        if not self.data.index.is_monotonic_increasing:
            self.data = self.data.sort_index()

    def _record_signal(self, timestamp, action, symbol='DEFAULT', price=0.0, **extra):
        """记录交易信号"""
        if action not in self.SIGNAL_ACTIONS:
            raise ValueError(f"无效信号动作: {action}, 允许: {self.SIGNAL_ACTIONS}")
        # 如果没传 symbol，尝试从数据中获取
        if symbol == 'DEFAULT' and 'symbol' in self.data.columns:
            symbol = self.data['symbol'].iloc[0]
        self.signals.append({
            'timestamp': timestamp,
            'action': action,
            'symbol': symbol,
            'price': float(price),
            **extra
        })

    def get_signals_summary(self) -> Dict:
        """返回信号统计摘要"""
        if not self.signals:
            return {"total": 0, "buys": 0, "sells": 0, "holds": 0}
        buys = len([s for s in self.signals if s['action'] == 'buy'])
        sells = len([s for s in self.signals if s['action'] == 'sell'])
        holds = len([s for s in self.signals if s['action'] == 'hold'])
        return {"total": len(self.signals), "buys": buys, "sells": sells, "holds": holds}

    def screen(self) -> Dict:
        """基于最新数据做实时选股判断

        默认实现：调用 generate_signals() 取最后一根K线的信号。
        子类可覆盖此方法实现自定义选股逻辑（更快或更精确）。

        Returns:
            Dict with keys: action ('buy'/'sell'/'hold'), reason (str), price (float)
        """
        if len(self.data) < 5:
            return {'action': 'hold', 'reason': '数据不足',
                    'price': float(self.data['close'].iloc[-1])}

        last_close = float(self.data['close'].iloc[-1])
        try:
            signals = self.generate_signals()
            if signals:
                last = signals[-1]
                return {
                    'action': last.get('action', 'hold'),
                    'reason': f'{self.strategy_name}',
                    'price': float(last.get('price', last_close)),
                }
        except Exception:
            pass
        return {'action': 'hold', 'reason': f'{self.strategy_name}:无信号',
                'price': last_close}
