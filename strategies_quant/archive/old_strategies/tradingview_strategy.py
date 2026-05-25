try:
    from core.base_strategy import BaseStrategy
except ImportError:
    from core.base_strategy import BaseStrategy

import pandas as pd
import numpy as np
try:
    import talib
    _HAS_TALIB = True
except ImportError:
    _HAS_TALIB = False
import sys
import os

# 添加TradingView指标路径
workspace_path = "/Users/chengming/.openclaw/workspace"
sys.path.append(workspace_path)
sys.path.append(os.path.join(workspace_path, "tradingview_indicators"))
sys.path.append(os.path.join(workspace_path, "tradingview_100_indicators"))

class TradingViewStrategy(BaseStrategy):
    def __init__(self, data, params):
        super().__init__(data, params)
        # 默认参数
        self.atr_period = params.get('atr_period', 10)
        self.multiplier = params.get('multiplier', 3)
        self.rsi_period = params.get('rsi_period', 14)
        self.rsi_overbought = params.get('rsi_overbought', 70)
        self.rsi_oversold = params.get('rsi_oversold', 30)
        
    def _manual_rsi(self, close, period=14):
        """手动计算RSI（talib不可用时的备选）"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _manual_atr(self, high, low, close, period=14):
        """手动计算ATR（talib不可用时的备选）"""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        return atr

    def generate_signals(self):
        """TradingView风格策略生成交易信号"""
        import numpy as np
        df = self.data
        # EMA交叉
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
        for i in range(26, len(df)):
            sym = df['symbol'].iloc[i] if 'symbol' in df.columns else 'DEFAULT'
            price = float(df['close'].iloc[i])
            if macd.iloc[i] > signal.iloc[i] and macd.iloc[i-1] <= signal.iloc[i-1] and rsi.iloc[i] < 70:
                self._record_signal(df.index[i], 'buy', sym, price)
            elif macd.iloc[i] < signal.iloc[i] and macd.iloc[i-1] >= signal.iloc[i-1] and rsi.iloc[i] > 30:
                self._record_signal(df.index[i], 'sell', sym, price)
        return self.signals
