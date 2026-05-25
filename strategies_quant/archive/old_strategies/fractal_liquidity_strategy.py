"""
JOAT Fractal Liquidity Map Strategy
=====================================
基于Williams Fractal检测流动性区域的策略:
1. Fractal检测: 5根K线中间最高/最低 = Swing High/Low
2. ATR Zone Merging: 相近fractal点(距离<0.5*ATR)合并为流动性区域
3. 区域强度评分: 被价格测试次数越多, 区域越强
4. Sweep检测: 价格穿越区域后反转(收盘回到区域内)
5. Ghost zone: 价格收盘穿过区域则区域失效
6. ATR trailing stop出场

买入: bullish sweep(价格扫过低点区域后反弹) + 成交量确认
卖出: bearish sweep(价格扫过高点区域后回落) + 成交量确认

知识来源:
- Williams Fractal (Bill Williams)
- Smart Money Concepts: Liquidity sweep
- ATR trailing stop for risk management
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class FractalLiquidityStrategy(BaseStrategy):
    """JOAT Fractal Liquidity Map — Williams分形流动性区域扫描策略"""

    strategy_description = "基于Williams分形检测流动性区域: ATR区域合并 + Sweep扫单检测 + Ghost失效 + ATR止损"
    strategy_category = "price_action"
    strategy_params_schema = {
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "fractal_length": {"type": "int", "default": 5, "label": "分形长度(K线数)"},
        "zone_merge_mult": {"type": "float", "default": 0.5, "label": "区域合并ATR倍数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.atr_period = params.get('atr_period', 14)
        self.fractal_length = params.get('fractal_length', 5)
        self.zone_merge_mult = params.get('zone_merge_mult', 0.5)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'atr_period': 14,
            'fractal_length': 5,
            'zone_merge_mult': 0.5,
            'trail_atr_mult': 2.5,
        }

    def generate_signals(self):
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'

        symbols = data['symbol'].unique()
        unique_times = sorted(data.index.unique())
        self.signals = []
        current_holding = None
        position_dir = 0
        high_water = 0.0
        entry_price = 0.0

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
                    score, direction = result
                    if score > best_score:
                        best_score = score
                        best_sym = sym
                        best_dir = direction

                if best_sym and best_score >= 3:
                    price = float(current_bars[current_bars['symbol'] == best_sym].iloc[0]['close'])
                    if best_dir == 1:
                        self._record_signal(current_time, 'buy', best_sym, price)
                        position_dir = 1
                    else:
                        self._record_signal(current_time, 'sell', best_sym, price)
                        position_dir = -1
                    current_holding = best_sym
                    entry_price = price
                    high_water = price

            else:
                bar_data = current_bars[current_bars['symbol'] == current_holding]
                if len(bar_data) == 0:
                    continue
                current_price = float(bar_data.iloc[0]['close'])

                # Update high/low water mark
                if position_dir == 1:
                    high_water = max(high_water, current_price)
                else:
                    high_water = min(high_water, current_price) if high_water > 0 else current_price

                # ATR trailing stop
                hist = data[(data['symbol'] == current_holding) & (data.index < current_time)]
                atr_val = self._calc_atr(hist)
                stop_hit = False

                if atr_val > 0 and high_water > 0:
                    if position_dir == 1 and current_price < high_water - self.trail_atr_mult * atr_val:
                        stop_hit = True
                    elif position_dir == -1 and current_price > high_water + self.trail_atr_mult * atr_val:
                        stop_hit = True

                # Signal-based exit
                result = self._evaluate(hist)
                signal_exit = False
                if result is not None:
                    score, direction = result
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
                    position_dir = 0
                    high_water = 0.0
                    entry_price = 0.0

        print(f"FractalLiquidity: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        """评估当前bar是否触发sweep信号"""
        half = self.fractal_length // 2
        min_len = max(self.atr_period + 1, self.fractal_length + 10)
        if len(data) < min_len:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)

        # ATR
        atr = self._calc_atr_from_arrays(high, low, close, n)
        if atr <= 0:
            return None

        # Detect fractals
        swing_highs = []  # (index, price)
        swing_lows = []   # (index, price)
        for i in range(half, n - half):
            # Swing High: middle bar has highest high
            is_high = True
            for j in range(i - half, i + half + 1):
                if j != i and high[j] >= high[i]:
                    is_high = False
                    break
            if is_high:
                swing_highs.append((i, high[i]))

            # Swing Low: middle bar has lowest low
            is_low = True
            for j in range(i - half, i + half + 1):
                if j != i and low[j] <= low[i]:
                    is_low = False
                    break
            if is_low:
                swing_lows.append((i, low[i]))

        if not swing_highs and not swing_lows:
            return None

        # Merge fractals into liquidity zones
        demand_zones = self._merge_zones(swing_lows, atr, 'demand')
        supply_zones = self._merge_zones(swing_highs, atr, 'supply')

        # Remove ghost zones (price closed through zone)
        demand_zones = self._filter_ghost_zones(demand_zones, close, high, low)
        supply_zones = self._filter_ghost_zones(supply_zones, close, high, low)

        # Volume confirmation
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        vol_confirm = False
        if vol_col and n >= 20:
            vol = data[vol_col].values
            vol_ma = np.mean(vol[-20:])
            if vol[-1] >= vol_ma * 1.2:
                vol_confirm = True

        # Score sweep signals
        bull_score = 0
        bear_score = 0

        # Bullish sweep: price sweeps below a demand zone then closes back above
        for zone in demand_zones:
            zl, zh, strength = zone
            if low[-1] < zl and close[-1] > zl:
                bull_score += strength
                if vol_confirm:
                    bull_score += 1

        # Bearish sweep: price sweeps above a supply zone then closes back below
        for zone in supply_zones:
            zl, zh, strength = zone
            if high[-1] > zh and close[-1] < zh:
                bear_score += strength
                if vol_confirm:
                    bear_score += 1

        if bull_score >= 3:
            return bull_score, 1
        elif bear_score >= 3:
            return bear_score, -1

        return None

    def _merge_zones(self, fractals, atr, zone_type):
        """Merge nearby fractal points into liquidity zones using ATR distance."""
        if not fractals:
            return []

        merge_dist = self.zone_merge_mult * atr
        # Sort by price
        sorted_fractals = sorted(fractals, key=lambda x: x[1])

        zones = []
        current_indices = [sorted_fractals[0][0]]
        current_prices = [sorted_fractals[0][1]]

        for i in range(1, len(sorted_fractals)):
            if abs(sorted_fractals[i][1] - current_prices[-1]) < merge_dist:
                current_indices.append(sorted_fractals[i][0])
                current_prices.append(sorted_fractals[i][1])
            else:
                # Finalize current zone
                zl = min(current_prices) - 0.1 * atr
                zh = max(current_prices) + 0.1 * atr
                strength = len(current_prices)  # More tests = stronger zone
                zones.append((zl, zh, strength))
                current_indices = [sorted_fractals[i][0]]
                current_prices = [sorted_fractals[i][1]]

        # Final zone
        if current_prices:
            zl = min(current_prices) - 0.1 * atr
            zh = max(current_prices) + 0.1 * atr
            strength = len(current_prices)
            zones.append((zl, zh, strength))

        return zones

    def _filter_ghost_zones(self, zones, close, high, low):
        """Remove zones where price has closed through (ghost zones)."""
        active_zones = []
        n = len(close)
        for zl, zh, strength in zones:
            is_ghost = False
            # Check if price closed through the zone in recent bars
            for j in range(max(0, n - 10), n):
                if close[j] < zl and zl < zh:  # Closed below demand zone
                    is_ghost = True
                    break
                if close[j] > zh and zl < zh:  # Closed above supply zone
                    is_ghost = True
                    break
            if not is_ghost:
                active_zones.append((zl, zh, strength))
        return active_zones

    def _calc_atr(self, data):
        """Calculate current ATR from DataFrame."""
        if len(data) < self.atr_period + 1:
            return 0
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        return self._calc_atr_from_arrays(high, low, close, len(close))

    def _calc_atr_from_arrays(self, high, low, close, n):
        """Calculate ATR from numpy arrays."""
        if n < self.atr_period + 1:
            return 0
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        )
        return float(np.mean(tr[-self.atr_period:]))

    def screen(self):
        data = self.data.copy()
        if len(data) < 50:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '无流动性区域信号', 'price': price}

        score, direction = result
        if score >= 3:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"sweep_score={score} (fractal_liquidity)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
