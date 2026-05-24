"""
Nadaraya-Watson Regression Strategy (纳达拉雅-沃森回归策略)
==========================================================
非参数核回归策略, 核心逻辑:
1. Nadaraya-Watson核回归估计器平滑收盘价
   NW(x) = sum(K(x-xi)*yi) / sum(K(x-xi))
   高斯核: K(u) = exp(-u^2 / (2*h^2))
2. 带宽h自适应: h = ATR * bandwidth_mult
3. NW线斜率从负转正=买入, 从正转负=卖出
4. Triple EMA过滤器(可选): EMA方向需一致
5. Residual bands: 价格远离NW线>2sigma时减仓
6. ATR trailing stop出场

知识来源:
- 非参数统计: Nadaraya-Watson核回归估计
- 自适应带宽: 基于ATR动态调整平滑度
- 残差分析: 2sigma带判断极端偏离
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class NadarayaWatsonStrategy(BaseStrategy):
    """Nadaraya-Watson核回归策略 — 非参数平滑趋势跟踪"""

    strategy_description = "NW核回归趋势跟踪: 高斯核平滑 + 斜率翻转信号 + EMA过滤 + 残差2sigma带 + ATR止损"
    strategy_category = "regression"
    strategy_params_schema = {
        "kernel_bandwidth": {"type": "int", "default": 10, "label": "核回归窗口"},
        "bandwidth_atr_mult": {"type": "float", "default": 1.0, "label": "ATR带宽倍数"},
        "ema_filter_period": {"type": "int", "default": 20, "label": "EMA过滤周期"},
        "use_ema_filter": {"type": "bool", "default": True, "label": "启用EMA过滤"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "止损ATR倍数"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "residual_sigma_mult": {"type": "float", "default": 2.0, "label": "残差sigma倍数"},
        "min_slope_change": {"type": "float", "default": 0.0, "label": "最小斜率变化阈值"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.kernel_bandwidth = params.get('kernel_bandwidth', 10)
        self.bandwidth_atr_mult = params.get('bandwidth_atr_mult', 1.0)
        self.ema_filter_period = params.get('ema_filter_period', 20)
        self.use_ema_filter = params.get('use_ema_filter', True)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)
        self.atr_period = params.get('atr_period', 14)
        self.residual_sigma_mult = params.get('residual_sigma_mult', 2.0)
        self.min_slope_change = params.get('min_slope_change', 0.0)

    def get_default_params(self):
        return {
            'kernel_bandwidth': 10,
            'bandwidth_atr_mult': 1.0,
            'ema_filter_period': 20,
            'use_ema_filter': True,
            'trail_atr_mult': 2.5,
            'atr_period': 14,
            'residual_sigma_mult': 2.0,
            'min_slope_change': 0.0,
        }

    def generate_signals(self):
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'

        symbols = data['symbol'].unique()
        unique_times = sorted(data.index.unique())
        self.signals = []

        # Track state per symbol
        state = {}
        for sym in symbols:
            state[sym] = {
                'holding': False,
                'position_dir': 0,   # 1=long, -1=short
                'buy_time': None,
                'high_water': 0.0,
                'entry_price': 0.0,
                'prev_slope': 0.0,
            }

        for current_time in unique_times:
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            for _, bar in current_bars.iterrows():
                sym = bar['symbol']
                s = state[sym]
                close_price = float(bar['close'])

                # Get historical data up to current time
                hist = data[(data['symbol'] == sym) & (data.index <= current_time)]
                min_len = max(self.kernel_bandwidth * 3, self.ema_filter_period, self.atr_period + 1, 50)
                if len(hist) < min_len:
                    continue

                close_arr = hist['close'].values
                high_arr = hist['high'].values
                low_arr = hist['low'].values
                n = len(close_arr)

                # Calculate ATR
                atr = self._calc_atr(high_arr, low_arr, close_arr)
                if atr <= 0:
                    continue

                # Calculate Nadaraya-Watson regression
                h = atr * self.bandwidth_atr_mult
                nw_line = self._nadaraya_watson(close_arr, h)

                # Calculate NW slope (derivative)
                if n >= 3:
                    slope = nw_line[-1] - nw_line[-2]
                else:
                    slope = 0.0

                # Normalized slope
                if close_arr[-1] > 0:
                    norm_slope = slope / close_arr[-1] * 100
                else:
                    norm_slope = 0.0

                # Calculate residuals and sigma bands
                residuals = close_arr - nw_line
                if len(residuals) > self.kernel_bandwidth:
                    sigma = np.std(residuals[-self.kernel_bandwidth:])
                else:
                    sigma = atr
                residual = close_price - nw_line[-1]

                # EMA filter
                ema_dir = 0
                if self.use_ema_filter:
                    ema = self._calc_ema(close_arr, self.ema_filter_period)
                    ema_dir = 1 if close_price > ema[-1] else -1
                    ema_slope = ema[-1] - ema[-3] if len(ema) >= 3 else 0
                    if ema_slope > 0:
                        ema_dir = max(ema_dir, 1)
                    else:
                        ema_dir = min(ema_dir, -1)

                # Slope transition detection
                prev_slope = s['prev_slope']
                slope_bullish_flip = prev_slope < -self.min_slope_change and norm_slope > self.min_slope_change
                slope_bearish_flip = prev_slope > self.min_slope_change and norm_slope < -self.min_slope_change
                s['prev_slope'] = norm_slope

                if not s['holding']:
                    # === ENTRY LOGIC ===
                    # Buy: NW slope flips from negative to positive
                    if slope_bullish_flip:
                        ema_ok = True
                        if self.use_ema_filter and ema_dir < 0:
                            ema_ok = False
                        if ema_ok:
                            self._record_signal(
                                current_time, 'buy', sym, close_price,
                                nw_slope=norm_slope,
                                residual_pct=residual / sigma if sigma > 0 else 0,
                            )
                            s['holding'] = True
                            s['position_dir'] = 1
                            s['buy_time'] = current_time
                            s['high_water'] = close_price
                            s['entry_price'] = close_price

                    # Sell (short): NW slope flips from positive to negative
                    elif slope_bearish_flip:
                        ema_ok = True
                        if self.use_ema_filter and ema_dir > 0:
                            ema_ok = False
                        if ema_ok:
                            self._record_signal(
                                current_time, 'sell', sym, close_price,
                                nw_slope=norm_slope,
                                residual_pct=residual / sigma if sigma > 0 else 0,
                            )
                            s['holding'] = True
                            s['position_dir'] = -1
                            s['buy_time'] = current_time
                            s['high_water'] = close_price
                            s['entry_price'] = close_price

                else:
                    # === EXIT LOGIC ===
                    days_held = len([t for t in unique_times if s['buy_time'] < t <= current_time])

                    # Update high/low water mark
                    if s['position_dir'] == 1:
                        s['high_water'] = max(s['high_water'], close_price)
                    else:
                        s['high_water'] = min(s['high_water'], close_price) if s['high_water'] > 0 else close_price

                    exit_reason = None

                    # Exit 1: ATR trailing stop
                    if atr > 0 and s['high_water'] > 0:
                        if s['position_dir'] == 1:
                            stop_price = s['high_water'] - self.trail_atr_mult * atr
                            if close_price < stop_price:
                                exit_reason = f"atr_stop@{stop_price:.2f}"
                        elif s['position_dir'] == -1:
                            stop_price = s['high_water'] + self.trail_atr_mult * atr
                            if close_price > stop_price:
                                exit_reason = f"atr_stop@{stop_price:.2f}"

                    # Exit 2: Slope reversal signal
                    if s['position_dir'] == 1 and slope_bearish_flip:
                        exit_reason = "slope_bearish_flip"
                    elif s['position_dir'] == -1 and slope_bullish_flip:
                        exit_reason = "slope_bullish_flip"

                    # Exit 3: Residual band breach (>2sigma)
                    if sigma > 0 and abs(residual) > self.residual_sigma_mult * sigma:
                        exit_reason = f"residual_{self.residual_sigma_mult}sigma"

                    # Exit 4: Max hold 60 days
                    if days_held >= 60:
                        exit_reason = f"max_hold_{days_held}d"

                    if exit_reason:
                        action = 'sell' if s['position_dir'] == 1 else 'buy'
                        self._record_signal(
                            current_time, action, sym, close_price,
                            reason=exit_reason,
                            days_held=days_held,
                            pnl_pct=(close_price - s['entry_price']) / s['entry_price'] if s['entry_price'] > 0 else 0,
                        )
                        s['holding'] = False
                        s['position_dir'] = 0
                        s['buy_time'] = None
                        s['high_water'] = 0.0
                        s['entry_price'] = 0.0

        print(f"NadarayaWatson: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _nadaraya_watson(self, y, h):
        """Compute Nadaraya-Watson kernel regression estimates.

        NW(x_i) = sum(K(x_i - x_j) * y_j) / sum(K(x_i - x_j))
        K(u) = exp(-u^2 / (2 * h^2))

        Uses a local window for efficiency.
        """
        n = len(y)
        nw = np.empty(n)
        bw = self.kernel_bandwidth
        x = np.arange(n, dtype=float)

        for i in range(n):
            # Use a local window around i for efficiency
            start = max(0, i - bw)
            end = min(n, i + bw + 1)
            x_local = x[start:end]
            y_local = y[start:end]

            # Gaussian kernel
            u = (x[i] - x_local) / (h if h > 0 else 1.0)
            kernel_weights = np.exp(-0.5 * u * u)

            weight_sum = np.sum(kernel_weights)
            if weight_sum > 0:
                nw[i] = np.sum(kernel_weights * y_local) / weight_sum
            else:
                nw[i] = y[i]

        return nw

    def _calc_atr(self, high, low, close):
        """Calculate ATR for the latest bar."""
        n = len(close)
        if n < self.atr_period + 1:
            return 0.0
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        return float(np.mean(tr[-self.atr_period:]))

    def _calc_ema(self, values, period):
        """Calculate EMA series."""
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

    def _calc_rsi(self, close):
        """Calculate RSI for the latest bar (used in screen)."""
        period = 14
        if len(close) < period + 1:
            return 50.0
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            return 100.0
        return 100.0 - 100.0 / (1 + avg_gain / avg_loss)

    def screen(self):
        """Real-time screening based on latest bar."""
        data = self.data.copy()
        min_len = max(self.kernel_bandwidth * 3, self.ema_filter_period, self.atr_period + 1, 50)
        if len(data) < min_len:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        close_arr = data['close'].values
        high_arr = data['high'].values
        low_arr = data['low'].values
        n = len(close_arr)
        price = float(close_arr[-1])

        # ATR
        atr = self._calc_atr(high_arr, low_arr, close_arr)
        if atr <= 0:
            return {'action': 'hold', 'reason': 'ATR无效', 'price': price}

        # NW regression
        h = atr * self.bandwidth_atr_mult
        nw_line = self._nadaraya_watson(close_arr, h)

        # Slope
        slope = nw_line[-1] - nw_line[-2]
        norm_slope = slope / price * 100 if price > 0 else 0

        # Residual bands
        residuals = close_arr - nw_line
        if len(residuals) > self.kernel_bandwidth:
            sigma = np.std(residuals[-self.kernel_bandwidth:])
        else:
            sigma = atr
        residual = price - nw_line[-1]
        residual_pct = residual / sigma if sigma > 0 else 0

        # EMA filter
        ema_dir = 0
        if self.use_ema_filter:
            ema = self._calc_ema(close_arr, self.ema_filter_period)
            ema_dir = 1 if price > ema[-1] else -1

        # Previous slope (from 2 bars ago)
        prev_slope = (nw_line[-2] - nw_line[-3]) / close_arr[-2] * 100 if n >= 3 else 0

        # Signal detection
        slope_bullish = prev_slope < -self.min_slope_change and norm_slope > self.min_slope_change
        slope_bearish = prev_slope > self.min_slope_change and norm_slope < -self.min_slope_change

        if slope_bullish and (not self.use_ema_filter or ema_dir > 0):
            return {
                'action': 'buy',
                'reason': f'NW_slope_flip_up {prev_slope:.3f}->{norm_slope:.3f} ema={ema_dir}',
                'price': price,
            }
        elif slope_bearish and (not self.use_ema_filter or ema_dir < 0):
            return {
                'action': 'sell',
                'reason': f'NW_slope_flip_down {prev_slope:.3f}->{norm_slope:.3f} ema={ema_dir}',
                'price': price,
            }

        # Check residual band
        if abs(residual_pct) > self.residual_sigma_mult:
            return {
                'action': 'hold',
                'reason': f'极端偏离 residual={residual_pct:.1f}sigma (谨慎)',
                'price': price,
            }

        return {
            'action': 'hold',
            'reason': f'slope={norm_slope:.3f} residual={residual_pct:.1f}sigma',
            'price': price,
        }
