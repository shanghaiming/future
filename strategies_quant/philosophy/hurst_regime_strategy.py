"""
Hurst Regime Strategy
======================

Philosophy: "Follow the regime, not the price. Markets have distinct behavioral states."

Core Logic:
- Compute rolling Hurst exponent using R/S analysis (100-bar window)
- H > 0.6: TRENDING → follow momentum (buy if up, sell if down)
- H < 0.4: MEAN-REVERTING → fade moves (buy dips, sell rallies)
- 0.4 <= H <= 0.6: RANDOM WALK → don't trade
- Use VDP for direction confirmation
- Use KER as fast proxy for Hurst (KER ≈ |H-0.5| at 1/100th computation cost)
- ATR trailing stop
- Position size proportional to |H - 0.5| (stronger regime = bigger position)

Mathematical Foundation:
- Hurst Exponent: R/S分析法，H=0.5随机游走，H>0.5趋势持续，H<0.5均值回归
- Volume Delta Pressure (VDP): delta = volume × (2×close - high - low) / (high - low)
- Kaufman Efficiency Ratio (KER): |net_change| / Σ|daily_changes|
- Position Sizing: |H - 0.5| determines position confidence
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from core.base_strategy import BaseStrategy


class HurstRegimeStrategy(BaseStrategy):
    """Hurst Regime Detection Strategy - Trading based on market efficiency states"""

    strategy_description = (
        "Hurst Regime: Trend-following when H>0.6, Mean-reverting when H<0.4, "
        "VDP confirmation, KER fast proxy, ATR trailing stop"
    )
    strategy_category = "market_state"

    strategy_params_schema = {
        "hurst_window": {"type": "int", "default": 100, "label": "Hurst计算窗口"},
        "ker_window": {"type": "int", "default": 20, "label": "KER计算窗口"},
        "vdp_ema_period": {"type": "int", "default": 10, "label": "VDP EMA平滑周期"},
        "ema_fast": {"type": "int", "default": 10, "label": "快速EMA周期"},
        "ema_slow": {"type": "int", "default": 30, "label": "慢速EMA周期"},
        "rsi_period": {"type": "int", "default": 14, "label": "RSI周期"},
        "rsi_ob": {"type": "int", "default": 70, "label": "RSI超买"},
        "rsi_os": {"type": "int", "default": 30, "label": "RSI超卖"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "max_hold": {"type": "int", "default": 60, "label": "最大持仓天数"},
        "hurst_trend_thresh": {"type": "float", "default": 0.6, "label": "Hurst趋势阈值"},
        "hurst_mr_thresh": {"type": "float", "default": 0.4, "label": "Hurst均值回归阈值"},
        "vdp_confirm_threshold": {"type": "float", "default": 0.01, "label": "VDP确认阈值"},
        "ker_fast_mode": {"type": "bool", "default": True, "label": "使用KER快速模式"},
    }

    def __init__(self, data, params=None):
        super().__init__(data, params)
        p = self.params
        self.hurst_window = p.get('hurst_window', 100)
        self.ker_window = p.get('ker_window', 20)
        self.vdp_ema_period = p.get('vdp_ema_period', 10)
        self.ema_fast = p.get('ema_fast', 10)
        self.ema_slow = p.get('ema_slow', 30)
        self.rsi_period = p.get('rsi_period', 14)
        self.rsi_ob = p.get('rsi_ob', 70)
        self.rsi_os = p.get('rsi_os', 30)
        self.atr_period = p.get('atr_period', 14)
        self.trail_atr_mult = p.get('trail_atr_mult', 2.5)
        self.hold_min = p.get('hold_min', 3)
        self.max_hold = p.get('max_hold', 60)
        self.hurst_trend_thresh = p.get('hurst_trend_thresh', 0.6)
        self.hurst_mr_thresh = p.get('hurst_mr_thresh', 0.4)
        self.vdp_confirm_threshold = p.get('vdp_confirm_threshold', 0.5)
        self.ker_fast_mode = p.get('ker_fast_mode', True)

        # Trading state tracking
        self.current_holding = None
        self.buy_time = None
        self.position_dir = 0
        self.high_water = 0.0
        self.low_water = float('inf')
        self.position_size = 1.0  # Position size based on regime strength

    def get_default_params(self):
        return {
            'hurst_window': 100,
            'ker_window': 20,
            'vdp_ema_period': 10,
            'ema_fast': 10,
            'ema_slow': 30,
            'rsi_period': 14,
            'rsi_ob': 70,
            'rsi_os': 30,
            'atr_period': 14,
            'trail_atr_mult': 2.5,
            'hold_min': 3,
            'max_hold': 60,
            'hurst_trend_thresh': 0.6,
            'hurst_mr_thresh': 0.4,
            'vdp_confirm_threshold': 0.5,
            'ker_fast_mode': True,
        }

    def generate_signals(self) -> List[Dict]:
        """Generate trading signals based on Hurst regime detection"""
        data = self.data.copy()
        unique_times = sorted(data.index.unique())
        self.signals = []

        for current_time in unique_times:
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            # No holding case: look for entry signals
            if self.current_holding is None:
                for _, bar in current_bars.iterrows():
                    sym = bar['symbol']
                    hist = data[(data['symbol'] == sym) & (data.index <= current_time)]
                    signal_info = self._evaluate_regime(hist, current_time)

                    if signal_info and signal_info['regime'] != 'random':
                        # Record buy signal
                        self._record_signal(
                            current_time,
                            'buy' if signal_info['direction'] > 0 else 'sell',
                            sym,
                            bar['close'],
                            regime=signal_info['regime'],
                            hurst=signal_info['hurst'],
                            vdp=signal_info['vdp'],
                            ker=signal_info['ker'],
                            position_size=signal_info['position_size']
                        )

                        # Update trading state
                        self.current_holding = sym
                        self.buy_time = current_time
                        self.position_dir = signal_info['direction']
                        self.high_water = bar['close'] if signal_info['direction'] > 0 else float('inf')
                        self.low_water = bar['close'] if signal_info['direction'] < 0 else 0.0
                        self.position_size = signal_info['position_size']
                        break

            # Holding case: check for exit conditions
            else:
                bar_data = current_bars[current_bars['symbol'] == self.current_holding]
                if len(bar_data) == 0:
                    continue

                bar = bar_data.iloc[0]
                current_price = bar['close']

                # Update high/low water marks
                if self.position_dir > 0:
                    self.high_water = max(self.high_water, current_price)
                else:
                    self.low_water = min(self.low_water, current_price)

                days_held = len([t for t in unique_times if self.buy_time < t <= current_time])

                # Check exit conditions
                if days_held >= self.hold_min:
                    hist = data[(data['symbol'] == self.current_holding) & (data.index <= current_time)]

                    # ATR trailing stop
                    atr_val = self._calc_atr(hist)
                    stop_triggered = False

                    if atr_val > 0 and days_held > self.hold_min:
                        if self.position_dir > 0:
                            # Long position: trailing stop below high water
                            stop_price = self.high_water - self.trail_atr_mult * atr_val
                            if current_price < stop_price:
                                stop_triggered = True
                        else:
                            # Short position: trailing stop above low water
                            stop_price = self.low_water + self.trail_atr_mult * atr_val
                            if current_price > stop_price:
                                stop_triggered = True

                    # Max hold days
                    if days_held >= self.max_hold:
                        stop_triggered = True

                    # Regime change exit
                    if not stop_triggered:
                        signal_info = self._evaluate_regime(hist, current_time)
                        if signal_info and signal_info['regime'] == 'random':
                            stop_triggered = True
                        elif signal_info and signal_info['direction'] * self.position_dir < 0:
                            stop_triggered = True

                    # Execute exit
                    if stop_triggered:
                        exit_action = 'sell' if self.position_dir > 0 else 'buy'
                        self._record_signal(
                            current_time,
                            exit_action,
                            self.current_holding,
                            current_price,
                            exit_reason='atr_stop' if stop_triggered and days_held < self.max_hold else 'regime_change'
                        )

                        # Reset trading state
                        self.current_holding = None
                        self.buy_time = None
                        self.position_dir = 0
                        self.high_water = 0.0
                        self.low_water = float('inf')
                        self.position_size = 1.0

        print(f"HurstRegimeStrategy: Generated {len(self.signals)} signals")
        return self.signals

    def _evaluate_regime(self, data, current_time):
        """Evaluate market regime and generate signal"""
        if len(data) < max(self.hurst_window, self.ema_slow) + 10:
            return None

        prices = data['close'].values.astype(float)

        # Calculate regime indicators
        hurst = self._calc_hurst(prices)
        vdp = self._calc_vdp(data)
        ker = self._calc_ker(prices)

        # Skip if calculations failed
        if np.isnan(hurst) or np.isnan(vdp) or np.isnan(ker):
            return None

        # Determine regime
        if hurst > self.hurst_trend_thresh:
            regime = 'trending'
            # Use price momentum as primary direction, VDP as confirmation
            price_change = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
            direction = 1 if price_change > 0 else -1
        elif hurst < self.hurst_mr_thresh:
            regime = 'mean_reverting'
            # Fade the price move
            price_change = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
            direction = -1 if price_change > 0 else 1  # Fade the move
        else:
            regime = 'random'
            return None

        # Position size based on regime strength
        regime_strength = abs(hurst - 0.5)
        position_size = min(2.0, regime_strength * 4)  # Scale 0-2 based on strength

        # VDP confirmation threshold (lower for more sensitivity)
        vdp_threshold = self.vdp_confirm_threshold * 0.1  # Make it more sensitive
        # For very strong regimes (hurst > 0.7), be more lenient with VDP
        if regime_strength > 0.2:  # hurst > 0.7 or hurst < 0.3
            vdp_threshold *= 2  # Double the tolerance for strong regimes
        elif regime_strength > 0.1:  # Moderate regime
            vdp_threshold *= 1.5  # Slightly more tolerant

        if abs(vdp) < vdp_threshold:
            return None

        # KER fast mode alternative
        if self.ker_fast_mode:
            # Use KER as proxy when computational cost matters
            ker_proxy = abs(ker - 0.5) * 2  # Scale to match Hurst range
            if ker_proxy < 0.05 and regime == 'trending':
                return None  # Very weak trend confirmation

        return {
            'regime': regime,
            'direction': direction,
            'hurst': hurst,
            'vdp': vdp,
            'ker': ker,
            'position_size': position_size
        }

    def _calc_hurst(self, prices):
        """Calculate Hurst exponent using R/S analysis"""
        if len(prices) < self.hurst_window:
            return np.nan

        series = prices[-self.hurst_window:]
        returns = np.diff(np.log(series))
        n_total = len(returns)

        # Use multiple time scales for regression
        ns = []
        rs_vals = []

        min_n, max_n = 10, n_total // 2
        if min_n >= max_n:
            return np.nan

        # Logarithmic spacing of time scales
        num_splits = min(8, max_n - min_n + 1)
        split_sizes = np.unique(np.linspace(min_n, max_n, num_splits).astype(int))

        for n in split_sizes:
            num_sub = n_total // n
            if num_sub < 1:
                continue

            rs_list = []
            for i in range(num_sub):
                sub = returns[i * n:(i + 1) * n]
                mean_sub = np.mean(sub)
                cumdev = np.cumsum(sub - mean_sub)
                r = np.max(cumdev) - np.min(cumdev)
                s = np.std(sub, ddof=1)
                if s > 0:
                    rs_list.append(r / s)

            if rs_list:
                ns.append(np.log(n))
                rs_vals.append(np.log(np.mean(rs_list)))

        if len(ns) < 2:
            return np.nan

        # Linear regression: log(R/S) = H * log(n) + log(c)
        coeffs = np.polyfit(ns, rs_vals, 1)
        hurst = coeffs[0]

        return float(np.clip(hurst, 0.0, 1.0))

    def _calc_vdp(self, data):
        """Calculate Volume Delta Pressure for direction confirmation"""
        if len(data) < self.vdp_ema_period:
            return np.nan

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values if 'volume' in data.columns else np.ones(len(data))

        # Calculate price position within range (0 = low, 1 = high)
        price_position = np.where(
            (high - low) > 1e-10,
            (close - low) / (high - low),
            0.5  # If no range, assume midpoint
        )

        # Volume-weighted position shift
        # Positive when close is near high (buy pressure)
        # Negative when close is near low (sell pressure)
        delta_shift = (price_position - 0.5) * 2  # Scale to [-1, 1]

        # Weight by volume and use recent values only
        recent_volume = volume[-self.vdp_ema_period:]
        recent_delta = delta_shift[-self.vdp_ema_period:]

        # Calculate weighted average of recent delta
        volume_weights = recent_volume / np.sum(recent_volume) if np.sum(recent_volume) > 0 else np.ones(len(recent_volume)) / len(recent_volume)
        weighted_delta = np.sum(recent_delta * volume_weights)

        return weighted_delta

    def _calc_ker(self, prices):
        """Kaufman Efficiency Ratio as fast Hurst proxy"""
        if len(prices) < self.ker_window:
            return np.nan

        n = len(prices)
        ker_values = np.full(n, np.nan)

        for i in range(self.ker_window, n):
            window_prices = prices[i - self.ker_window:i + 1]
            net_change = abs(window_prices[-1] - window_prices[0])
            total_change = np.sum(np.abs(np.diff(window_prices)))

            if total_change > 1e-10:
                ker_values[i] = net_change / total_change
            else:
                ker_values[i] = 0.0

        return float(ker_values[-1])

    def _calc_atr(self, data):
        """Calculate Average True Range"""
        if len(data) < self.atr_period + 1:
            return 0.0

        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        n = len(high)

        tr_list = []
        for i in range(max(1, n - self.atr_period), n):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )
            tr_list.append(tr)

        return float(np.mean(tr_list)) if tr_list else 0.0

    def _ema(self, values, period):
        """Exponential Moving Average"""
        if len(values) < period:
            return np.full(len(values), np.nan)

        ema = np.full(len(values), np.nan)
        seed = np.mean(values[:period])
        ema[period - 1] = seed
        k = 2.0 / (period + 1)

        for i in range(period, len(values)):
            ema[i] = values[i] * k + ema[i - 1] * (1 - k)

        return ema

    def _calc_rsi(self, data):
        """Calculate RSI"""
        if len(data) < self.rsi_period + 1:
            return 50.0

        close = data['close'].values
        delta = np.diff(close)
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)

        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])

        if avg_loss < 1e-10:
            return 100.0

        rs = avg_gain / avg_loss
        return float(100 - 100 / (1 + rs))

    def screen(self):
        """Real-time screening based on current regime"""
        data = self.data.copy()
        min_len = max(self.hurst_window, self.ema_slow) + 10

        if len(data) < min_len:
            return {
                'action': 'hold',
                'reason': 'Insufficient data',
                'price': float(data['close'].iloc[-1])
            }

        result = self._evaluate_regime(data, data.index[-1])
        price = float(data['close'].iloc[-1])

        if result is None:
            return {
                'action': 'hold',
                'reason': 'Random walk regime',
                'price': price
            }

        action = 'buy' if result['direction'] > 0 else 'sell'
        regime_desc = f"{result['regime'].replace('_', ' ').title()} (H={result['hurst']:.2f})"

        return {
            'action': action,
            'reason': f'{regime_desc} VDP={result["vdp"]:.2f} KER={result["ker"]:.2f}',
            'price': price
        }