"""
自适应均值回归策略 (Adaptive Mean Reversion Strategy)
==================================================
基于KER调整窗口的均值回归 + VDP量价确认 + ATR追踪止损

核心思想:
  "在市场效率高时（趋势市）使用长窗口，在市场效率低时（震荡市）使用短窗口，
   均值回归只在有足够的价格偏离和量价确认时进行"

核心机制:
  1. KER窗口自适应：
     - KER > 0.25（趋势市）：使用40周期长窗口（捕捉更大均值回归机会）
     - KER < 0.15（震荡市）：使用20周期短窗口（更快响应）
     - KER中间区域：使用30周期中窗口（平衡）

  2. Z-score信号：
     - Z = (价格 - 均值) / 标准差
     - 买入：Z < -2.0（显著低估）
     - 卖出：Z > +2.0（显著高估）

  3. VDP量价确认：
     - 买入需要：cum_delta > 0（有购买压力）
     - 卖出需要：cum_delta < 0（有卖出压力）

  4. ATR追踪止损：
     - 买入后，追踪止损 = max(入场价，最高价 - 2*ATR)
     - 卖出后，追踪止损 = min(入场价，最低价 + 2*ATR)

  5. 仓位管理：
     - 仓位大小与|Z-score|成反比（价格越偏离均值，仓位越大）
     - 最大仓位限制：总资金的50%

  6. 持仓限制：
     - 最多持有15天（均值回归是短期策略）
     - 时间止损：持仓达到15天自动平仓

优势：
  - 自适应窗口：根据市场效率调整，避免在趋势市中用短窗口造成频繁交易
  - 量价确认：避免在没有成交量的价格偏离中交易
  - 风险控制：双重退出机制（价格回归 + 时间止损 + ATR追踪）

技术指标：KER, Z-score, VDP cumulative delta, ATR, Volume Delta
"""

import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class AdaptiveMeanReversionStrategy(BaseStrategy):
    """自适应均值回归策略 — KER自适应窗口 + VDP确认 + ATR追踪止损"""

    strategy_description = "自适应均值回归：KER窗口自适应 + VDP量价确认 + ATR追踪止损"
    strategy_category = "mean_reversion"
    strategy_params_schema = {
        "base_window": {"type": "int", "default": 20, "label": "基础窗口大小"},
        "ker_period": {"type": "int", "default": 20, "label": "KER计算周期"},
        "ker_trend_threshold": {"type": "float", "default": 0.25, "label": "KER趋势阈值"},
        "ker_range_threshold": {"type": "float", "default": 0.15, "label": "KER震荡阈值"},
        "delta_ema_period": {"type": "int", "default": 10, "label": "VDP Delta EMA周期"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR计算周期"},
        "trail_atr_mult": {"type": "float", "default": 2.0, "label": "追踪止损ATR倍数"},
        "z_score_threshold": {"type": "float", "default": 2.0, "label": "Z-score阈值"},
        "max_hold_days": {"type": "int", "default": 15, "label": "最大持仓天数"},
        "position_size_base": {"type": "float", "default": 0.2, "label": "基础仓位比例"},
        "max_position": {"type": "float", "default": 0.5, "label": "最大仓位比例"},
    }

    def get_default_params(self):
        return {k: v["default"] for k, v in self.strategy_params_schema.items()}

    def validate_params(self):
        p = self.params
        if p.get('ker_period', 20) < 5:
            raise ValueError("ker_period must be >= 5")
        if p.get('base_window', 20) < 10:
            raise ValueError("base_window must be >= 10")
        if p.get('delta_ema_period', 10) < 2:
            raise ValueError("delta_ema_period must be >= 2")
        if p.get('atr_period', 14) < 1:
            raise ValueError("atr_period must be >= 1")
        if p.get('z_score_threshold', 2.0) <= 0:
            raise ValueError("z_score_threshold must be > 0")
        if p.get('max_hold_days', 15) <= 0:
            raise ValueError("max_hold_days must be > 0")
        if p.get('position_size_base', 0.2) <= 0:
            raise ValueError("position_size_base must be > 0")
        if p.get('max_position', 0.5) <= 0:
            raise ValueError("max_position must be > 0")

    def __init__(self, data, params=None):
        super().__init__(data, params)
        # Strategy parameters
        self.base_window = self.params['base_window']
        self.ker_period = self.params['ker_period']
        self.ker_trend_threshold = self.params['ker_trend_threshold']
        self.ker_range_threshold = self.params['ker_range_threshold']
        self.delta_ema_period = self.params['delta_ema_period']
        self.atr_period = self.params['atr_period']
        self.trail_atr_mult = self.params['trail_atr_mult']
        self.z_score_threshold = self.params['z_score_threshold']
        self.max_hold_days = self.params['max_hold_days']
        self.position_size_base = self.params['position_size_base']
        self.max_position = self.params['max_position']

        # Initialize tracking variables
        self.hold_days = 0
        self.entry_price = 0.0
        self.entry_type = None
        self.current_position = 0.0
        self.trail_stop = 0.0

    # ------------------------------------------------------------------
    # KER Calculation (Kaufman Efficiency Ratio)
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_ker(closes, period):
        """计算Kaufman效率比率 (KER)
        KER = |净变化| / Σ|日变化|
        值域：[0, 1]，越大表示趋势性越强
        """
        n = len(closes)
        ker = np.full(n, np.nan)
        for i in range(period, n):
            net_change = abs(closes[i] - closes[i - period])
            total_change = np.sum(np.abs(np.diff(closes[i - period:i + 1])))
            if total_change > 1e-10:
                ker[i] = net_change / total_change
            else:
                ker[i] = 0.0
        return ker

    def _get_adaptive_window(self, ker):
        """根据KER值获取自适应窗口大小"""
        n = len(ker)
        window = np.full(n, np.nan)

        for i in range(n):
            if np.isnan(ker[i]):
                window[i] = self.base_window
            elif ker[i] > self.ker_trend_threshold:
                # 趋势市：使用更长窗口（40）
                window[i] = 40
            elif ker[i] < self.ker_range_threshold:
                # 震荡市：使用更短窗口（20）
                window[i] = 20
            else:
                # 过渡区：使用中间窗口（30）
                window[i] = 30

        return window

    # ------------------------------------------------------------------
    # VDP (Volume Delta Pressure) Calculation
    # ------------------------------------------------------------------

    def _calc_bar_delta(self, close, high, low, volume):
        """计算单根K线的成交量Delta"""
        n = len(close)
        delta = np.zeros(n)

        for i in range(n):
            if high[i] == low[i]:
                delta[i] = 0.0
            else:
                # 估算买卖力量
                if close[i] > (high[i] + low[i]) / 2:
                    # 阳线或上涨，买方力量更强
                    delta[i] = volume[i] * (close[i] - low[i]) / (high[i] - low[i])
                else:
                    # 阴线或下跌，卖方力量更强
                    delta[i] = -volume[i] * (high[i] - close[i]) / (high[i] - low[i])

        return delta

    def _calc_cumulative_delta(self, delta, ema_period):
        """计算累积Delta（VDP）"""
        if len(delta) == 0:
            return np.array([])

        # 使用EMA平滑Delta
        ema_delta = np.zeros(len(delta))
        if len(delta) >= ema_period:
            seed = np.mean(delta[:ema_period])
            ema_delta[ema_period - 1] = seed
            k = 2.0 / (ema_period + 1)
            for i in range(ema_period, len(delta)):
                ema_delta[i] = delta[i] * k + ema_delta[i - 1] * (1 - k)

        # 计算累积Delta
        cum_delta = np.cumsum(delta)
        # 用EMA平滑后的Delta调整
        for i in range(1, len(cum_delta)):
            cum_delta[i] = cum_delta[i - 1] + 0.1 * (ema_delta[i] - cum_delta[i - 1])

        return cum_delta

    # ------------------------------------------------------------------
    # Technical Indicators
    # ------------------------------------------------------------------

    def _calc_atr(self, high, low, close):
        """计算平均真实范围 (ATR)"""
        n = len(high)
        if n < self.atr_period + 1:
            return np.full(n, np.nan)

        tr = np.empty(n)
        tr[0] = high[0] - low[0]

        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )

        atr = np.full(n, np.nan)
        for i in range(self.atr_period - 1, n):
            atr[i] = np.mean(tr[i - self.atr_period + 1:i + 1])

        return atr

    def _calc_zscore(self, price, window):
        """计算Z-score"""
        n = len(price)
        zscore = np.full(n, np.nan)

        for i in range(window, n):
            if np.isnan(price[i-window:i]).any():
                continue

            window_prices = price[i-window:i]
            mean = np.mean(window_prices)
            std = np.std(window_prices)

            if std > 1e-10:
                zscore[i] = (price[i] - mean) / std
            else:
                zscore[i] = 0.0

        return zscore

    def _generate_signals(self):
        """生成交易信号"""
        # Get necessary data columns
        closes = self.data['close'].values
        highs = self.data['high'].values
        lows = self.data['low'].values
        volumes = self.data['volume'].values if 'volume' in self.data.columns else self.data['vol'].values

        # Calculate indicators
        ker = self._calc_ker(closes, self.ker_period)
        adaptive_windows = self._get_adaptive_window(ker)

        # Get volume data
        if 'volume' not in self.data.columns:
            volumes = np.full(len(self.data), 1000000)  # Default volume if not provided

        # Calculate VDP indicators
        bar_deltas = self._calc_bar_delta(closes, highs, lows, volumes)
        cum_deltas = self._calc_cumulative_delta(bar_deltas, self.delta_ema_period)

        # Calculate technical indicators
        atr = self._calc_atr(highs, lows, closes)
        zscores = {}

        # Calculate z-scores for each window size
        for window_size in [20, 30, 40]:
            zscores[f'window_{window_size}'] = self._calc_zscore(closes, window_size)

        # Generate signals
        for i in range(len(self.data)):
            current_date = self.data.index[i]
            current_price = closes[i]

            # Get current window based on KER
            current_window = int(adaptive_windows[i])
            current_zscore = zscores[f'window_{current_window}'][i]

            # Skip if any indicator is NaN
            if (np.isnan(ker[i]) or np.isnan(current_zscore) or
                np.isnan(cum_deltas[i]) or np.isnan(atr[i])):
                continue

            # Reset hold days if not in position
            if self.current_position == 0.0:
                self.hold_days = 0

            # Time-based exit
            if self.current_position != 0.0:
                self.hold_days += 1

                # Time exit - max hold days reached
                if self.hold_days >= self.max_hold_days:
                    self._record_signal(
                        timestamp=current_date,
                        action='sell' if self.current_position > 0 else 'buy',
                        price=current_price,
                        reason=f'Time exit - reached max hold days ({self.hold_days})'
                    )
                    self.current_position = 0.0
                    self.entry_price = 0.0
                    self.entry_type = None
                    self.hold_days = 0
                    continue

                # Trailing stop exit
                if self.entry_type == 'long' and current_price < self.trail_stop:
                    self._record_signal(
                        timestamp=current_date,
                        action='sell',
                        price=current_price,
                        reason=f'Trailing stop exit - price hit {self.trail_stop:.2f}'
                    )
                    self.current_position = 0.0
                    self.entry_price = 0.0
                    self.entry_type = None
                    self.hold_days = 0
                    continue

                elif self.entry_type == 'short' and current_price > self.trail_stop:
                    self._record_signal(
                        timestamp=current_date,
                        action='buy',
                        price=current_price,
                        reason=f'Trailing stop exit - price hit {self.trail_stop:.2f}'
                    )
                    self.current_position = 0.0
                    self.entry_price = 0.0
                    self.entry_type = None
                    self.hold_days = 0
                    continue

            # Trading logic
            # Long entry: Z-score < -2.0 and cum_delta > 0
            if (current_zscore < -self.z_score_threshold and
                cum_deltas[i] > 0 and self.current_position == 0.0):

                # Calculate position size (inverse to z-score magnitude)
                z_magnitude = abs(current_zscore)
                position_size = self.position_size_base * min(3.0, z_magnitude / 2.0)
                position_size = min(position_size, self.max_position)

                self._record_signal(
                    timestamp=current_date,
                    action='buy',
                    price=current_price,
                    z_score=current_zscore,
                    ker_value=ker[i],
                    window_size=current_window,
                    cum_delta=cum_deltas[i],
                    position_size=position_size,
                    reason=f'Z-score mean reversion buy ({current_zscore:.2f}) with VDP confirmation'
                )

                # Set tracking variables
                self.current_position = position_size
                self.entry_price = current_price
                self.entry_type = 'long'
                self.hold_days = 1
                self.trail_stop = current_price  # Initial trail stop

            # Short entry: Z-score > +2.0 and cum_delta < 0
            elif (current_zscore > self.z_score_threshold and
                  cum_deltas[i] < 0 and self.current_position == 0.0):

                # Calculate position size (inverse to z-score magnitude)
                z_magnitude = abs(current_zscore)
                position_size = self.position_size_base * min(3.0, z_magnitude / 2.0)
                position_size = min(position_size, self.max_position)

                self._record_signal(
                    timestamp=current_date,
                    action='sell',
                    price=current_price,
                    z_score=current_zscore,
                    ker_value=ker[i],
                    window_size=current_window,
                    cum_delta=cum_deltas[i],
                    position_size=position_size,
                    reason=f'Z-score mean reversion sell ({current_zscore:.2f}) with VDP confirmation'
                )

                # Set tracking variables
                self.current_position = -position_size  # Negative for short
                self.entry_price = current_price
                self.entry_type = 'short'
                self.hold_days = 1
                self.trail_stop = current_price  # Initial trail stop

            # Update trailing stop
            elif self.current_position != 0.0:
                if not np.isnan(atr[i]):
                    if self.entry_type == 'long':
                        # Long trailing stop: max(entry_price, high - 2*ATR)
                        new_stop = max(self.entry_price, highs[i] - self.trail_atr_mult * atr[i])
                        self.trail_stop = max(self.trail_stop, new_stop)
                    elif self.entry_type == 'short':
                        # Short trailing stop: min(entry_price, low + 2*ATR)
                        new_stop = min(self.entry_price, lows[i] + self.trail_atr_mult * atr[i])
                        self.trail_stop = min(self.trail_stop, new_stop)

        return self.signals

    def generate_signals(self):
        """Generate trading signals"""
        # Reset tracking variables
        self.signals = []
        self.hold_days = 0
        self.entry_price = 0.0
        self.entry_type = None
        self.current_position = 0.0
        self.trail_stop = 0.0

        return self._generate_signals()

    def screen(self):
        """实时选股判断"""
        if len(self.data) < self.ker_period + 40:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(self.data['close'].iloc[-1])}

        try:
            signals = self.generate_signals()
            if signals:
                last = signals[-1]
                return {
                    'action': last.get('action', 'hold'),
                    'reason': f"Z-score: {last.get('z_score', 0):.2f}, KER: {last.get('ker_value', 0):.2f}",
                    'price': float(last.get('price', self.data['close'].iloc[-1])),
                }
        except Exception as e:
            pass

        return {'action': 'hold', 'reason': f'策略运行中: {self.strategy_name}', 'price': float(self.data['close'].iloc[-1])}