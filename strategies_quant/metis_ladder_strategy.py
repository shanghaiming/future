"""
Metis Ladder Compression Engine (麦梯斯拉德压缩引擎)
=====================================================
52周回撤分层的均值回归策略, 核心逻辑:
1. 计算52周最高价, 衡量当前回撤深度
2. 4层累积区域按回撤深度分层建仓
3. First-touch信号: 价格首次进入某区域时触发买入
4. RSI oversold确认: 仅在RSI<阈值时入场
5. ATR trailing stop + 时间止损出场

知识来源:
- 均值回归理论: 极端回撤后的统计回归
- 分层建仓: 不同回撤深度分配不同仓位权重
- 达到亚(Wisdom of Metis): 分层而非一次性下注
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class MetisLadderStrategy(BaseStrategy):
    """Metis Ladder Compression Engine — 52周回撤分层均值回归策略"""

    strategy_description = "52周回撤分层均值回归: 4层累积区域 + First-touch买入 + RSI超卖确认 + ATR trailing stop"
    strategy_category = "mean_reversion"
    strategy_params_schema = {
        "lookback_52w": {"type": "int", "default": 252, "label": "52周回溯天数"},
        "zone1_upper": {"type": "float", "default": -0.15, "label": "Zone1上界(轻度调整)"},
        "zone2_upper": {"type": "float", "default": -0.25, "label": "Zone2上界(中度调整)"},
        "zone3_upper": {"type": "float", "default": -0.35, "label": "Zone3上界(深度调整)"},
        "zone4_upper": {"type": "float", "default": -0.50, "label": "Zone4上界(极端恐慌)"},
        "rsi_threshold": {"type": "float", "default": 40.0, "label": "RSI超卖阈值"},
        "hold_max": {"type": "int", "default": 60, "label": "最大持仓天数"},
        "rsi_period": {"type": "int", "default": 14, "label": "RSI周期"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "止损ATR倍数"},
        "exit_near_high_pct": {"type": "float", "default": -0.10, "label": "近52周高点出场百分比"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.lookback_52w = params.get('lookback_52w', 252)
        self.zone1_upper = params.get('zone1_upper', -0.15)
        self.zone2_upper = params.get('zone2_upper', -0.25)
        self.zone3_upper = params.get('zone3_upper', -0.35)
        self.zone4_upper = params.get('zone4_upper', -0.50)
        self.rsi_threshold = params.get('rsi_threshold', 40.0)
        self.hold_max = params.get('hold_max', 60)
        self.rsi_period = params.get('rsi_period', 14)
        self.atr_period = params.get('atr_period', 14)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)
        self.exit_near_high_pct = params.get('exit_near_high_pct', -0.10)

        # Zone weight mapping: position size per zone
        self.zone_weights = {
            1: 0.25,  # Zone 1: 25% position
            2: 0.35,  # Zone 2: 35% position
            3: 0.25,  # Zone 3: 25% position
            4: 0.15,  # Zone 4: 15% position
        }

    def get_default_params(self):
        return {
            'lookback_52w': 252,
            'zone1_upper': -0.15,
            'zone2_upper': -0.25,
            'zone3_upper': -0.35,
            'zone4_upper': -0.50,
            'rsi_threshold': 40.0,
            'hold_max': 60,
            'rsi_period': 14,
            'atr_period': 14,
            'trail_atr_mult': 2.5,
            'exit_near_high_pct': -0.10,
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
                'buy_time': None,
                'high_water': 0.0,
                'entered_zones': set(),  # Track which zones have been entered (first-touch)
                'entry_price': 0.0,
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
                if len(hist) < self.lookback_52w + 5:
                    continue

                close_arr = hist['close'].values
                high_arr = hist['high'].values
                low_arr = hist['low'].values

                # Calculate 52-week high
                lookback_start = max(0, len(close_arr) - self.lookback_52w)
                high_52w = np.max(high_arr[lookback_start:])

                # Current drawdown from 52-week high
                if high_52w > 0:
                    drawdown = (close_price - high_52w) / high_52w
                else:
                    drawdown = 0.0

                # Determine current zone (0 = no zone)
                current_zone = self._get_zone(drawdown)

                # Calculate RSI
                rsi = self._calc_rsi(close_arr)

                # Calculate ATR
                atr = self._calc_atr(high_arr, low_arr, close_arr)

                if not s['holding']:
                    # === ENTRY LOGIC ===
                    if current_zone > 0 and rsi < self.rsi_threshold:
                        # First-touch check: only trigger if this zone hasn't been entered before
                        if current_zone not in s['entered_zones']:
                            s['entered_zones'].add(current_zone)
                            self._record_signal(
                                current_time, 'buy', sym, close_price,
                                zone=current_zone,
                                drawdown=drawdown,
                                rsi=rsi,
                                weight=self.zone_weights[current_zone],
                            )
                            s['holding'] = True
                            s['buy_time'] = current_time
                            s['high_water'] = close_price
                            s['entry_price'] = close_price
                else:
                    # === EXIT LOGIC ===
                    days_held = len([t for t in unique_times if s['buy_time'] < t <= current_time])

                    # Update high water mark
                    s['high_water'] = max(s['high_water'], close_price)

                    exit_reason = None

                    # Exit 1: Price recovered near 52-week high
                    if drawdown > self.exit_near_high_pct:
                        exit_reason = f"recovered_to_{drawdown:.1%}"

                    # Exit 2: Max holding period
                    if days_held >= self.hold_max:
                        exit_reason = f"max_hold_{days_held}d"

                    # Exit 3: ATR trailing stop
                    if atr > 0 and s['high_water'] > 0:
                        stop_price = s['high_water'] - self.trail_atr_mult * atr
                        if close_price < stop_price:
                            exit_reason = f"atr_stop@{stop_price:.2f}"

                    if exit_reason:
                        self._record_signal(
                            current_time, 'sell', sym, close_price,
                            reason=exit_reason,
                            days_held=days_held,
                            pnl_pct=(close_price - s['entry_price']) / s['entry_price'] if s['entry_price'] > 0 else 0,
                        )
                        s['holding'] = False
                        s['buy_time'] = None
                        s['high_water'] = 0.0
                        s['entry_price'] = 0.0
                        # Reset entered zones after a complete trade cycle
                        s['entered_zones'] = set()

        print(f"MetisLadder: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _get_zone(self, drawdown):
        """Determine which accumulation zone the drawdown falls into.
        Returns zone number (1-4) or 0 if not in any zone."""
        if self.zone4_upper <= drawdown < self.zone3_upper:
            return 4  # Extreme panic: below -50%
        if self.zone3_upper <= drawdown < self.zone2_upper:
            return 3  # Deep correction: -35% to -50%
        if self.zone2_upper <= drawdown < self.zone1_upper:
            return 2  # Moderate correction: -25% to -35%
        if self.zone1_upper <= drawdown < 0:
            return 1  # Mild correction: -15% to -25%
        return 0  # Not in accumulation zone

    def _calc_rsi(self, close):
        """Calculate RSI for the latest bar."""
        period = self.rsi_period
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

    def screen(self):
        """Real-time screening based on latest bar."""
        data = self.data.copy()
        if len(data) < self.lookback_52w + 5:
            return {'action': 'hold', 'reason': '数据不足(需252+天)', 'price': float(data['close'].iloc[-1])}

        close_arr = data['close'].values
        high_arr = data['high'].values
        low_arr = data['low'].values
        price = float(close_arr[-1])

        # 52-week high
        lookback_start = max(0, len(close_arr) - self.lookback_52w)
        high_52w = np.max(high_arr[lookback_start:])

        if high_52w <= 0:
            return {'action': 'hold', 'reason': '52周高点无效', 'price': price}

        drawdown = (price - high_52w) / high_52w
        zone = self._get_zone(drawdown)
        rsi = self._calc_rsi(close_arr)
        atr = self._calc_atr(high_arr, low_arr, close_arr)

        if zone > 0 and rsi < self.rsi_threshold:
            return {
                'action': 'buy',
                'reason': f'Zone{zone} drawdown={drawdown:.1%} rsi={rsi:.1f} weight={self.zone_weights[zone]}',
                'price': price,
            }
        elif drawdown > self.exit_near_high_pct and zone == 0:
            return {
                'action': 'sell',
                'reason': f'已恢复至52周高点附近 drawdown={drawdown:.1%}',
                'price': price,
            }
        return {
            'action': 'hold',
            'reason': f'drawdown={drawdown:.1%} zone={zone} rsi={rsi:.1f}',
            'price': price,
        }
