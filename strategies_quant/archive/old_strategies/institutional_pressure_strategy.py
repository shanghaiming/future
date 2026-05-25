"""
Institutional Pressure Oscillator
===================================
基于成交量压力的机构行为识别策略, 核心思路来自TradingView学习:
1. buy/sell pressure从K线范围推导机构意图
2. 对数归一化防止极端值扭曲
3. 自适应噪声门过滤假信号
4. 参与度上限(95th percentile cap)防止异常成交量干扰
5. EMA信号线交叉确认

知识来源:
- TradingView Institutional Pressure Oscillator
- Volume Spread Analysis (VSA) 概念
- 对数归一化信号处理
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class InstitutionalPressureStrategy(BaseStrategy):
    """Institutional Pressure Oscillator — 成交量压力 + 对数归一化 + 噪声门 + EMA确认"""

    strategy_description = (
        "机构压力振荡器: buy/sell压力推导 + 对数归一化 + "
        "自适应噪声门 + 参与度上限 + EMA信号线交叉"
    )
    strategy_category = "volume"
    strategy_params_schema = {
        "ema_period": {"type": "int", "default": 20, "label": "EMA信号线周期"},
        "sensitivity": {"type": "float", "default": 1.5, "label": "噪声门灵敏度"},
        "lookback": {"type": "int", "default": 50, "label": "滚动窗口"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "止损ATR倍数"},
        "hold_min": {"type": "int", "default": 2, "label": "最少持仓天数"},
        "max_hold": {"type": "int", "default": 60, "label": "最大持仓天数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.ema_period = params.get('ema_period', 20)
        self.sensitivity = params.get('sensitivity', 1.5)
        self.lookback = params.get('lookback', 50)
        self.atr_period = params.get('atr_period', 14)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)
        self.hold_min = params.get('hold_min', 2)
        self.max_hold = params.get('max_hold', 60)

    def get_default_params(self):
        return {
            'ema_period': 20, 'sensitivity': 1.5, 'lookback': 50,
            'atr_period': 14, 'trail_atr_mult': 2.5,
            'hold_min': 2, 'max_hold': 60,
        }

    # ------------------------------------------------------------------
    # Indicator helpers
    # ------------------------------------------------------------------

    def _calc_ema(self, values, period):
        """EMA calculation"""
        values = np.asarray(values, dtype=float)
        n = len(values)
        result = np.empty(n)
        result[0] = values[0]
        k = 2.0 / (period + 1)
        for i in range(1, n):
            if i < period:
                result[i] = np.mean(values[:i + 1])
            else:
                result[i] = values[i] * k + result[i - 1] * (1 - k)
        return result

    def _calc_atr(self, high, low, close, n):
        """Average True Range"""
        if n < self.atr_period + 1:
            return 0.0
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]),
                       np.abs(low[1:] - close[:-1]))
        )
        return float(np.mean(tr[-self.atr_period:]))

    def _calc_pressure_oscillator(self, data):
        """计算机构压力振荡器的完整序列
        返回: (log_norm, signal_line, noise_gate) arrays
        """
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        n = len(close)

        # Volume column
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        if vol_col is None:
            return None, None, None
        volume = data[vol_col].values.astype(float)

        # --- 参与度上限: cap at 95th percentile ---
        lookback = min(self.lookback, n)
        cap_val = np.percentile(volume[-lookback:], 95)
        volume_capped = np.minimum(volume, cap_val)

        # --- Buy/Sell Pressure ---
        bar_range = high - low + 1e-10
        buy_pressure = volume_capped * (close - low) / bar_range
        sell_pressure = volume_capped * (high - close) / bar_range
        delta = buy_pressure - sell_pressure

        # --- 对数归一化 ---
        lookback_len = min(self.lookback, n)
        rolling_max = np.zeros(n)
        log_norm = np.zeros(n)
        for i in range(n):
            start = max(0, i - lookback_len + 1)
            window_max = np.max(np.abs(delta[start:i + 1]))
            rolling_max[i] = window_max

            if rolling_max[i] > 0:
                sign = np.sign(delta[i])
                log_norm[i] = sign * np.log(1 + abs(delta[i])) / np.log(1 + rolling_max[i])
            else:
                log_norm[i] = 0.0

        # --- EMA信号线 ---
        signal_line = self._calc_ema(log_norm, self.ema_period)

        # --- 自适应噪声门 ---
        noise_gate = np.zeros(n)
        for i in range(n):
            start = max(0, i - lookback_len + 1)
            window_std = np.std(log_norm[start:i + 1])
            noise_gate[i] = window_std * self.sensitivity

        return log_norm, signal_line, noise_gate

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, data):
        """评估当前bar是否产生信号
        返回: (score, direction, atr_val) or None
        """
        if len(data) < max(self.lookback, self.ema_period) + 10:
            return None

        n = len(data)
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # ATR
        atr_val = self._calc_atr(high, low, close, n)
        if atr_val <= 0:
            return None

        # Pressure oscillator
        log_norm, signal_line, noise_gate = self._calc_pressure_oscillator(data)
        if log_norm is None:
            return None

        score = 0
        direction = 0

        # Current values
        cur_norm = log_norm[-1]
        prev_norm = log_norm[-2] if n >= 2 else 0.0
        cur_signal = signal_line[-1]
        prev_signal = signal_line[-2] if n >= 2 else 0.0
        cur_gate = noise_gate[-1]

        # --- 买入信号: log_norm从负转正且超过噪声门 ---
        buy_cross = (prev_norm <= 0 and cur_norm > 0) or (prev_norm < cur_signal and cur_norm > cur_signal)
        buy_strength = cur_norm > cur_gate and cur_norm > 0

        # --- 卖出信号: log_norm从正转负且超过噪声门 ---
        sell_cross = (prev_norm >= 0 and cur_norm < 0) or (prev_norm > cur_signal and cur_norm < cur_signal)
        sell_strength = abs(cur_norm) > cur_gate and cur_norm < 0

        # EMA信号线交叉确认
        ema_cross_up = n >= 2 and log_norm[-2] < signal_line[-2] and log_norm[-1] > signal_line[-1]
        ema_cross_down = n >= 2 and log_norm[-2] > signal_line[-2] and log_norm[-1] < signal_line[-1]

        # Scoring
        if buy_cross:
            score += 3
        if buy_strength:
            score += 2
        if ema_cross_up:
            score += 2
        if cur_norm > 0 and cur_signal > 0:
            score += 1  # Both positive: trend confirmation

        if sell_cross:
            score -= 3
        if sell_strength:
            score -= 2
        if ema_cross_down:
            score -= 2
        if cur_norm < 0 and cur_signal < 0:
            score -= 1

        direction = 1 if score > 0 else (-1 if score < 0 else 0)
        return abs(score), direction, atr_val

    # ------------------------------------------------------------------
    # Signal generation (backtest)
    # ------------------------------------------------------------------

    def generate_signals(self):
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'

        symbols = data['symbol'].unique()
        unique_times = sorted(data.index.unique())
        self.signals = []
        current_holding = None
        buy_time = None
        position_dir = 0
        high_water = 0.0
        score_threshold = 4  # Need at least score of 4

        for current_time in unique_times:
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            if current_holding is None:
                best_score = -1
                best_sym = None
                best_dir = 0

                for _, bar in current_bars.iterrows():
                    sym = bar['symbol']
                    hist = data[(data['symbol'] == sym) & (data.index < current_time)]
                    result = self._evaluate(hist)
                    if result is None:
                        continue
                    score, direction, atr_val = result
                    if score > best_score and direction != 0:
                        best_score = score
                        best_sym = sym
                        best_dir = direction

                if best_sym and best_score >= score_threshold:
                    price = float(current_bars[current_bars['symbol'] == best_sym].iloc[0]['close'])
                    if best_dir == 1:
                        self._record_signal(current_time, 'buy', best_sym, price)
                        position_dir = 1
                    else:
                        self._record_signal(current_time, 'sell', best_sym, price)
                        position_dir = -1
                    current_holding = best_sym
                    buy_time = current_time
                    high_water = 0.0

            else:
                days_held = len([t for t in unique_times if buy_time < t <= current_time])
                if days_held >= self.hold_min:
                    bar_data = current_bars[current_bars['symbol'] == current_holding]
                    if len(bar_data) == 0:
                        continue
                    current_price = float(bar_data.iloc[0]['close'])

                    # Track high/low water mark
                    if position_dir == 1:
                        high_water = max(high_water, current_price)
                    else:
                        high_water = min(high_water, current_price) if high_water > 0 else current_price

                    # ATR trailing stop
                    hist = data[(data['symbol'] == current_holding) & (data.index < current_time)]
                    atr_val = self._calc_atr(
                        hist['high'].values, hist['low'].values,
                        hist['close'].values, len(hist)
                    )
                    stop_hit = False

                    if atr_val > 0 and high_water > 0:
                        if position_dir == 1 and current_price < high_water - self.trail_atr_mult * atr_val:
                            stop_hit = True
                        elif position_dir == -1 and current_price > high_water + self.trail_atr_mult * atr_val:
                            stop_hit = True

                    # Max hold
                    if days_held >= self.max_hold:
                        stop_hit = True

                    # Signal-based exit
                    result = self._evaluate(hist)
                    signal_exit = False
                    if result is not None:
                        score, direction, _ = result
                        if position_dir == 1 and direction == -1 and score >= 3:
                            signal_exit = True
                        elif position_dir == -1 and direction == 1 and score >= 3:
                            signal_exit = True

                    if stop_hit or signal_exit:
                        if position_dir == 1:
                            self._record_signal(current_time, 'sell', current_holding, current_price)
                        else:
                            self._record_signal(current_time, 'buy', current_holding, current_price)
                        current_holding = None
                        buy_time = None
                        position_dir = 0
                        high_water = 0.0

        print(f"InstitutionalPressure: 生成 {len(self.signals)} 个信号")
        return self.signals

    # ------------------------------------------------------------------
    # Real-time screening
    # ------------------------------------------------------------------

    def screen(self):
        data = self.data.copy()
        if len(data) < 50:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '评估失败(缺少成交量数据或数据不足)', 'price': price}

        score, direction, atr_val = result
        if score >= 4 and direction != 0:
            action = 'buy' if direction == 1 else 'sell'
            return {
                'action': action,
                'reason': f"score={score}, dir={direction}, atr={atr_val:.2f} (institutional pressure)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}, dir={direction}', 'price': price}
