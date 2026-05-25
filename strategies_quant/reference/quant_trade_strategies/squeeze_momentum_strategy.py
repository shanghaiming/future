"""
Squeeze动量策略 (Squeeze Momentum Strategy)
===========================================
TTM Squeeze波动率收缩 + 动量方向 + MACD确认的三合一策略。

来源: TradingView "Squeeze Momentum + MACD Overlay"

核心逻辑:
  1. Squeeze检测: Bollinger Bands收窄到Keltner Channels内部
  2. Squeeze释放: BB重新扩张到KC外部 = 波动率爆发即将来临
  3. 动量方向: 线性回归斜率确认突破方向
  4. MACD确认: MACD histogram方向一致

技术指标: Bollinger Bands, Keltner Channels, 线性回归动量, MACD
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class SqueezeMomentumStrategy(BaseStrategy):
    """Squeeze动量策略 — 波动率收缩释放 + 动量方向 + MACD确认"""

    strategy_description = "TTM Squeeze: BB收窄KC + 动量方向 + MACD确认"
    strategy_category = "volatility"
    strategy_params_schema = {
        "bb_period": {"type": "int", "default": 20, "label": "BB周期"},
        "bb_mult": {"type": "float", "default": 2.0, "label": "BB倍数"},
        "kc_period": {"type": "int", "default": 20, "label": "KC周期"},
        "kc_mult": {"type": "float", "default": 1.5, "label": "KC倍数"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "momentum_period": {"type": "int", "default": 20, "label": "动量回归周期"},
        "macd_fast": {"type": "int", "default": 12, "label": "MACD快线"},
        "macd_slow": {"type": "int", "default": 26, "label": "MACD慢线"},
        "macd_signal": {"type": "int", "default": 9, "label": "MACD信号线"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.bb_period = params.get('bb_period', 20)
        self.bb_mult = params.get('bb_mult', 2.0)
        self.kc_period = params.get('kc_period', 20)
        self.kc_mult = params.get('kc_mult', 1.5)
        self.atr_period = params.get('atr_period', 14)
        self.momentum_period = params.get('momentum_period', 20)
        self.macd_fast = params.get('macd_fast', 12)
        self.macd_slow = params.get('macd_slow', 26)
        self.macd_signal = params.get('macd_signal', 9)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'bb_period': 20, 'bb_mult': 2.0, 'kc_period': 20, 'kc_mult': 1.5,
            'atr_period': 14, 'momentum_period': 20,
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'hold_min': 3, 'trail_atr_mult': 2.5,
        }

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
        low_water = float('inf')

        for current_time in unique_times:
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            if current_holding is None:
                best_score = 0
                best_sym = None
                best_dir = 0

                for _, bar in current_bars.iterrows():
                    sym = bar['symbol']
                    hist = data[(data['symbol'] == sym) & (data.index < current_time)]
                    result = self._evaluate(hist)
                    if result is None:
                        continue
                    score, direction, _ = result
                    if abs(score) > abs(best_score):
                        best_score = score
                        best_sym = sym
                        best_dir = direction

                if best_sym and abs(best_score) >= 3:
                    if best_dir == 1:
                        self._record_signal(current_time, 'buy', best_sym)
                        position_dir = 1
                    else:
                        self._record_signal(current_time, 'sell', best_sym)
                        position_dir = -1
                    current_holding = best_sym
                    buy_time = current_time
                    high_water = 0.0
                    low_water = float('inf')

            else:
                days_held = len([t for t in unique_times if buy_time < t <= current_time])
                bar_data = current_bars[current_bars['symbol'] == current_holding]
                if len(bar_data) == 0:
                    continue
                current_price = float(bar_data.iloc[0]['close'])

                if position_dir == 1:
                    high_water = max(high_water, current_price) if high_water > 0 else current_price
                else:
                    low_water = min(low_water, current_price) if low_water < float('inf') else current_price

                if days_held >= self.hold_min:
                    hist = data[(data['symbol'] == current_holding) & (data.index < current_time)]
                    atr_val = self._calc_atr(hist)
                    should_exit = False

                    # ATR trailing stop
                    if atr_val > 0:
                        if position_dir == 1 and high_water > 0:
                            if current_price < high_water - self.trail_atr_mult * atr_val:
                                should_exit = True
                        elif position_dir == -1 and low_water < float('inf'):
                            if current_price > low_water + self.trail_atr_mult * atr_val:
                                should_exit = True

                    # Max hold
                    if days_held >= 60:
                        should_exit = True

                    # Signal-based exit
                    result = self._evaluate(hist)
                    if result is not None and not should_exit:
                        score, direction, _ = result
                        if position_dir == 1 and direction == -1 and score < -3:
                            should_exit = True
                        elif position_dir == -1 and direction == 1 and score > 3:
                            should_exit = True

                    if should_exit:
                        if position_dir == 1:
                            self._record_signal(current_time, 'sell', current_holding)
                        else:
                            self._record_signal(current_time, 'buy', current_holding)
                        current_holding = None
                        buy_time = None
                        position_dir = 0
                        high_water = 0.0
                        low_water = float('inf')

        print(f"SqueezeMomentum: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        """综合评估: Squeeze状态 + 动量方向 + MACD确认"""
        min_len = max(self.bb_period, self.kc_period, self.macd_slow + self.macd_signal, self.momentum_period) + 10
        if len(data) < min_len:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)

        # ===== 1. Squeeze Detection (BB inside KC) =====
        squeeze_status = self._detect_squeeze(close, high, low, n)

        # ===== 2. Linear Regression Momentum =====
        momentum = self._calc_momentum(close, n)

        # ===== 3. MACD =====
        macd_hist = self._calc_macd_histogram(close)

        # ===== Scoring =====
        score = 0

        # Squeeze release + direction (weight 4)
        if squeeze_status == 'fired_long':
            score += 4
        elif squeeze_status == 'fired_short':
            score -= 4
        elif squeeze_status == 'squeeze':
            # In squeeze: prepare, don't enter yet unless strong momentum
            if abs(momentum) > 0.5:
                score += 2 * np.sign(momentum)

        # Momentum direction (weight 3)
        if momentum > 0.3:
            score += 3
        elif momentum > 0:
            score += 1
        elif momentum < -0.3:
            score -= 3
        elif momentum < 0:
            score -= 1

        # MACD confirmation (weight 2)
        if macd_hist > 0:
            score += 2
        elif macd_hist < 0:
            score -= 2

        # Momentum acceleration (weight 1)
        if n >= 2:
            prev_mom = self._calc_momentum(close[:-1], n - 1)
            if momentum > prev_mom and momentum > 0:
                score += 1  # Accelerating upward
            elif momentum < prev_mom and momentum < 0:
                score -= 1  # Accelerating downward

        direction = 1 if score > 0 else -1
        atr = self._calc_atr(data)
        return score, direction, atr

    def _detect_squeeze(self, close, high, low, n):
        """检测BB/KC squeeze状态"""
        if n < self.bb_period + 1:
            return 'unknown'

        # Bollinger Bands
        bb_mid = np.mean(close[-self.bb_period:])
        bb_std = np.std(close[-self.bb_period:], ddof=1)
        bb_upper = bb_mid + self.bb_mult * bb_std
        bb_lower = bb_mid - self.bb_mult * bb_std

        # Keltner Channels
        atr = self._calc_atr_arrays(high, low, close, n)
        kc_mid = np.mean(close[-self.kc_period:])
        kc_upper = kc_mid + self.kc_mult * atr
        kc_lower = kc_mid - self.kc_mult * atr

        # Current squeeze: BB inside KC
        currently_squeezed = (bb_lower > kc_lower) and (bb_upper < kc_upper)

        # Previous squeeze status
        if n >= self.bb_period + 5:
            prev_close = close[:-5]
            prev_high = high[:-5]
            prev_low = low[:-5]
            prev_bb_mid = np.mean(prev_close[-self.bb_period:])
            prev_bb_std = np.std(prev_close[-self.bb_period:], ddof=1)
            prev_bb_upper = prev_bb_mid + self.bb_mult * prev_bb_std
            prev_bb_lower = prev_bb_mid - self.bb_mult * prev_bb_std
            prev_atr = self._calc_atr_arrays(prev_high, prev_low, prev_close, len(prev_close))
            prev_kc_mid = np.mean(prev_close[-self.kc_period:])
            prev_kc_upper = prev_kc_mid + self.kc_mult * prev_atr
            prev_kc_lower = prev_kc_mid - self.kc_mult * prev_atr
            was_squeezed = (prev_bb_lower > prev_kc_lower) and (prev_bb_upper < prev_kc_upper)
        else:
            was_squeezed = False

        if currently_squeezed:
            return 'squeeze'
        elif was_squeezed and not currently_squeezed:
            # Squeeze fired! Check direction
            if close[-1] > np.mean(close[-self.bb_period:]):
                return 'fired_long'
            else:
                return 'fired_short'
        else:
            return 'normal'

    def _calc_momentum(self, close, n):
        """线性回归斜率作为动量"""
        period = self.momentum_period
        if n < period:
            return 0.0
        recent = close[-period:]
        x = np.arange(period, dtype=float)
        slope = np.polyfit(x, recent, 1)[0]
        # Normalize by price level
        mid_price = np.mean(recent)
        if mid_price > 0:
            return slope / mid_price * 100  # Percentage slope
        return 0.0

    def _calc_macd_histogram(self, close):
        """计算MACD histogram"""
        fast_ema = self._calc_ema_series(close, self.macd_fast)
        slow_ema = self._calc_ema_series(close, self.macd_slow)
        macd_line = fast_ema - slow_ema
        signal = self._calc_ema_series(macd_line, self.macd_signal)
        return float(macd_line[-1] - signal[-1])

    def _calc_ema_series(self, values, period):
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

    def _calc_atr(self, data):
        if len(data) < self.atr_period + 1:
            return 0
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        return self._calc_atr_arrays(high, low, close, len(close))

    def _calc_atr_arrays(self, high, low, close, n):
        if n < self.atr_period + 1:
            return 0
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        return np.mean(tr[-self.atr_period:])

    def screen(self):
        data = self.data.copy()
        min_len = max(self.bb_period, self.kc_period) + 10
        if len(data) < min_len:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '评估失败', 'price': price}

        score, direction, _ = result
        if abs(score) >= 3:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} (squeeze+momentum)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
