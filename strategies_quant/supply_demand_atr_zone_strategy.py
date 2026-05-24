"""
供需ATR区间策略 (Supply/Demand ATR Zones Strategy)
===================================================
基于价格行为识别供需区间 + ATR追踪止损。

来源: TradingView batch_1 — Supply/Demand ATR Zones

核心逻辑:
  1. 区间检测: 价格在N根K线内波动 <= 0.5*ATR → 盘整区间
  2. 突破确认: 价格从区间边界移动 > 1.5*ATR → 有效突破
  3. 区间分类:
     - 需求区间: 盘整位于波段低点，之后向上突破
     - 供给区间: 盘整位于波段高点，之后向下突破
  4. 信号触发: 价格回踩至区间(距区间顶部0.5*ATR内)时入场
  5. 区间失效: 价格收盘穿透区间超过1*ATR

WHY this works:
  - 供需区间本质是机构建仓/出货的区域，盘整代表多空平衡
  - ATR自适应波动率，不同股票/市场环境下阈值自动调整
  - 回踩入场提供更好的风险回报比，止损空间小
  - 区间失效机制防止在假区间中持仓

技术指标: ATR, Swing High/Low, Consolidation Range
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class SupplyDemandATRZoneStrategy(BaseStrategy):
    """供需ATR区间策略 — 盘整区间识别 + 回踩入场 + ATR追踪止损"""

    strategy_description = "供需ATR区间: 盘整区间识别+回踩入场+ATR追踪止损"
    strategy_category = "price_action"
    strategy_params_schema = {
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "consol_bars": {"type": "int", "default": 5, "label": "盘整K线数"},
        "consol_atr_mult": {"type": "float", "default": 0.5, "label": "盘整ATR倍数"},
        "breakout_atr_mult": {"type": "float", "default": 1.5, "label": "突破ATR倍数"},
        "reentry_atr_mult": {"type": "float", "default": 0.5, "label": "回踩ATR倍数"},
        "invalidation_atr_mult": {"type": "float", "default": 1.0, "label": "失效ATR倍数"},
        "trail_atr_mult": {"type": "float", "default": 2.0, "label": "追踪止损ATR倍数"},
        "swing_lookback": {"type": "int", "default": 10, "label": "波段回溯周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
    }

    def get_default_params(self):
        return {
            'atr_period': 14,
            'consol_bars': 5,
            'consol_atr_mult': 0.5,
            'breakout_atr_mult': 1.5,
            'reentry_atr_mult': 0.5,
            'invalidation_atr_mult': 1.0,
            'trail_atr_mult': 2.0,
            'swing_lookback': 10,
            'hold_min': 3,
        }

    def validate_params(self):
        p = self.params
        if p.get('atr_period', 14) < 2:
            raise ValueError("atr_period must be >= 2")
        if p.get('consol_bars', 5) < 2:
            raise ValueError("consol_bars must be >= 2")
        if p.get('breakout_atr_mult', 1.5) <= p.get('consol_atr_mult', 0.5):
            raise ValueError("breakout_atr_mult must be > consol_atr_mult")

    def __init__(self, data, params=None):
        super().__init__(data, params)
        self.atr_period = self.params['atr_period']
        self.consol_bars = self.params['consol_bars']
        self.consol_atr_mult = self.params['consol_atr_mult']
        self.breakout_atr_mult = self.params['breakout_atr_mult']
        self.reentry_atr_mult = self.params['reentry_atr_mult']
        self.invalidation_atr_mult = self.params['invalidation_atr_mult']
        self.trail_atr_mult = self.params['trail_atr_mult']
        self.swing_lookback = self.params['swing_lookback']
        self.hold_min = self.params['hold_min']

    # ------------------------------------------------------------------
    # Indicator helpers
    # ------------------------------------------------------------------

    def _calc_atr(self, high, low, close):
        """Average True Range over self.atr_period bars.
        WHY ATR: measures true volatility including gaps, adapts to each stock's
        own price range so thresholds are universal regardless of price level.
        """
        n = len(high)
        if n < self.atr_period + 1:
            return np.full(n, np.nan)
        tr = np.empty(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i - 1]),
                        abs(low[i] - close[i - 1]))
        # Rolling mean using convolution
        atr = np.full(n, np.nan)
        for i in range(self.atr_period - 1, n):
            atr[i] = np.mean(tr[i - self.atr_period + 1:i + 1])
        return atr

    def _find_swing_lows(self, low, lookback):
        """Boolean mask: True where low[i] is a local minimum within lookback.
        WHY: demand zones form at swing lows where sellers exhausted themselves.
        """
        n = len(low)
        is_swing_low = np.zeros(n, dtype=bool)
        half = lookback // 2
        for i in range(half, n - half):
            window = low[i - half:i + half + 1]
            if low[i] == window.min():
                is_swing_low[i] = True
        return is_swing_low

    def _find_swing_highs(self, high, lookback):
        """Boolean mask: True where high[i] is a local maximum within lookback.
        WHY: supply zones form at swing highs where buyers exhausted themselves.
        """
        n = len(high)
        is_swing_high = np.zeros(n, dtype=bool)
        half = lookback // 2
        for i in range(half, n - half):
            window = high[i - half:i + half + 1]
            if high[i] == window.max():
                is_swing_high[i] = True
        return is_swing_high

    def _detect_zones(self, high, low, close, atr):
        """
        Detect supply and demand zones via consolidation + breakout pattern.

        Returns two lists:
          demand_zones: [(zone_top, zone_bottom, start_idx, end_idx), ...]
          supply_zones: [(zone_top, zone_bottom, start_idx, end_idx), ...]

        WHY this detection works:
          - Consolidation within 0.5*ATR means the market is in equilibrium,
            absorbing orders without directional commitment.
          - A subsequent breakout > 1.5*ATR confirms that one side overwhelmed
            the other, leaving an order imbalance zone.
          - Price tends to revisit these zones because unfilled institutional
            orders rest there, acting as magnets on pullbacks.
        """
        n = len(close)
        demand_zones = []
        supply_zones = []

        i = 0
        max_lookahead = 5  # check up to N bars after consolidation for breakout
        while i < n - self.consol_bars:
            # Need ATR value at this position
            if np.isnan(atr[i + self.consol_bars - 1]):
                i += 1
                continue

            atr_val = atr[i + self.consol_bars - 1]
            if atr_val <= 0:
                i += 1
                continue

            # --- Check consolidation: N consecutive bars within consol_atr_mult * ATR range ---
            consol_range = self.consol_atr_mult * atr_val
            segment_high = high[i:i + self.consol_bars]
            segment_low = low[i:i + self.consol_bars]
            range_width = segment_high.max() - segment_low.min()

            if range_width <= consol_range:
                # Consolidation found; record its boundaries
                zone_top = segment_high.max()
                zone_bottom = segment_low.min()
                zone_end = i + self.consol_bars - 1

                # Check breakout in bars immediately after the consolidation.
                # WHY look ahead up to N bars: the breakout may not happen on the
                # very next bar — there can be 1-2 transition bars before momentum
                # explodes out of the equilibrium zone.
                zone_recorded = False
                for offset in range(1, max_lookahead + 1):
                    breakout_bar = zone_end + offset
                    if breakout_bar >= n:
                        break
                    if np.isnan(atr[breakout_bar]) or atr[breakout_bar] <= 0:
                        continue
                    breakout_atr = atr[breakout_bar]
                    # Breakout upward -> demand zone (buyers overwhelmed sellers)
                    # WHY: price left the equilibrium upward = demand exceeded supply
                    if high[breakout_bar] > zone_top + self.breakout_atr_mult * breakout_atr:
                        demand_zones.append((zone_top, zone_bottom, i, zone_end))
                        zone_recorded = True
                        break
                    # Breakout downward -> supply zone (sellers overwhelmed buyers)
                    # WHY: price left the equilibrium downward = supply exceeded demand
                    elif low[breakout_bar] < zone_bottom - self.breakout_atr_mult * breakout_atr:
                        supply_zones.append((zone_top, zone_bottom, i, zone_end))
                        zone_recorded = True
                        break

                # Only skip past the consolidation if a zone was recorded;
                # otherwise advance by 1 to allow overlapping windows to detect
                # a wider consolidation that includes the breakout bar.
                if zone_recorded:
                    i = zone_end + 1
                else:
                    i += 1
            else:
                i += 1

        return demand_zones, supply_zones

    def _is_zone_invalidated(self, close_val, zone_top, zone_bottom, atr_val, zone_type):
        """Check if a zone has been invalidated by price closing through it.
        WHY: once price fully penetrates a zone, the original order imbalance
        has been absorbed — the zone no longer offers support/resistance.
        """
        if atr_val <= 0:
            return False
        if zone_type == 'demand':
            # Invalidation: price closes below zone bottom by > 1*ATR
            return close_val < zone_bottom - self.invalidation_atr_mult * atr_val
        else:  # supply
            # Invalidation: price closes above zone top by > 1*ATR
            return close_val > zone_top + self.invalidation_atr_mult * atr_val

    def _is_in_zone(self, close_val, zone_top, zone_bottom, atr_val, zone_type):
        """Check if price is within re-entry range of a zone.
        WHY: entering near the zone top of a demand zone gives a tight stop;
        the reentry_atr_mult buffer allows for imprecise fills while keeping
        the risk/reward favourable.
        """
        if atr_val <= 0:
            return False
        if zone_type == 'demand':
            # Buy when price returns within reentry_atr_mult of zone top
            return (close_val >= zone_bottom - self.reentry_atr_mult * atr_val and
                    close_val <= zone_top + self.reentry_atr_mult * atr_val)
        else:  # supply
            # Sell when price returns within reentry_atr_mult of zone bottom
            return (close_val <= zone_top + self.reentry_atr_mult * atr_val and
                    close_val >= zone_bottom - self.reentry_atr_mult * atr_val)

    # ------------------------------------------------------------------
    # Signal generation (backtest)
    # ------------------------------------------------------------------

    def generate_signals(self):
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'

        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        n = len(close)

        if n < self.atr_period + self.consol_bars + 5:
            print(f"SupplyDemandATRZone: 数据不足({n} bars)")
            return []

        # Compute ATR series
        atr = self._calc_atr(high, low, close)

        # Detect all zones once
        demand_zones, supply_zones = self._detect_zones(high, low, close, atr)

        self.signals = []
        position_dir = 0       # 1=long, -1=short, 0=flat
        high_water = 0.0
        low_water = float('inf')
        entry_idx = 0

        for i in range(self.atr_period, n):
            if np.isnan(atr[i]):
                continue
            atr_val = atr[i]
            timestamp = data.index[i]
            symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else 'DEFAULT'
            price = close[i]

            # --- If in position, check exit conditions ---
            if position_dir != 0:
                bars_held = i - entry_idx

                # Update high/low water mark for trailing stop
                if position_dir == 1:
                    high_water = max(high_water, price)
                else:
                    low_water = min(low_water, price)

                should_exit = False

                # ATR trailing stop
                # WHY: adapts to volatility — wide in volatile markets, tight in calm
                if bars_held >= self.hold_min:
                    if position_dir == 1 and high_water > 0:
                        if price < high_water - self.trail_atr_mult * atr_val:
                            should_exit = True
                    elif position_dir == -1 and low_water < float('inf'):
                        if price > low_water + self.trail_atr_mult * atr_val:
                            should_exit = True

                # Time-based exit (avoid holding stale positions)
                if bars_held >= 60:
                    should_exit = True

                if should_exit:
                    action = 'sell' if position_dir == 1 else 'buy'
                    self._record_signal(timestamp, action, symbol, price)
                    position_dir = 0
                    high_water = 0.0
                    low_water = float('inf')

            # --- If flat, check entry conditions ---
            if position_dir == 0:
                # Check demand zones for buy entry
                # WHY: price returning to demand zone = retest of institutional buying area
                for zone_top, zone_bottom, z_start, z_end in demand_zones:
                    # Only consider zones that have already formed (z_end < i)
                    if z_end >= i:
                        continue
                    # Check zone hasn't been invalidated
                    if self._is_zone_invalidated(price, zone_top, zone_bottom, atr_val, 'demand'):
                        continue
                    # Check price is in re-entry range
                    if self._is_in_zone(price, zone_top, zone_bottom, atr_val, 'demand'):
                        self._record_signal(timestamp, 'buy', symbol, price)
                        position_dir = 1
                        high_water = price
                        entry_idx = i
                        break

                # Check supply zones for sell entry (short)
                # WHY: price returning to supply zone = retest of institutional selling area
                if position_dir == 0:
                    for zone_top, zone_bottom, z_start, z_end in supply_zones:
                        if z_end >= i:
                            continue
                        if self._is_zone_invalidated(price, zone_top, zone_bottom, atr_val, 'supply'):
                            continue
                        if self._is_in_zone(price, zone_top, zone_bottom, atr_val, 'supply'):
                            self._record_signal(timestamp, 'sell', symbol, price)
                            position_dir = -1
                            low_water = price
                            entry_idx = i
                            break

        print(f"SupplyDemandATRZone: 生成 {len(self.signals)} 个信号, "
              f"需求区间={len(demand_zones)}, 供给区间={len(supply_zones)}")
        return self.signals

    # ------------------------------------------------------------------
    # Real-time screening
    # ------------------------------------------------------------------

    def screen(self):
        """基于最新数据判断当前是否处于供需区间回踩位置。"""
        data = self.data
        if len(data) < self.atr_period + self.consol_bars + 5:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        atr = self._calc_atr(high, low, close)
        n = len(close)

        latest_price = close[-1]
        latest_atr = atr[-1]
        if np.isnan(latest_atr) or latest_atr <= 0:
            return {'action': 'hold', 'reason': 'ATR无效', 'price': float(latest_price)}

        # Detect zones
        demand_zones, supply_zones = self._detect_zones(high, low, close, atr)

        # Check if latest price is in any active demand zone
        for zone_top, zone_bottom, z_start, z_end in demand_zones:
            if self._is_zone_invalidated(latest_price, zone_top, zone_bottom, latest_atr, 'demand'):
                continue
            if self._is_in_zone(latest_price, zone_top, zone_bottom, latest_atr, 'demand'):
                return {
                    'action': 'buy',
                    'reason': f'回踩需求区间 [{zone_bottom:.2f}-{zone_top:.2f}]',
                    'price': float(latest_price),
                }

        # Check if latest price is in any active supply zone
        for zone_top, zone_bottom, z_start, z_end in supply_zones:
            if self._is_zone_invalidated(latest_price, zone_top, zone_bottom, latest_atr, 'supply'):
                continue
            if self._is_in_zone(latest_price, zone_top, zone_bottom, latest_atr, 'supply'):
                return {
                    'action': 'sell',
                    'reason': f'回踩供给区间 [{zone_bottom:.2f}-{zone_top:.2f}]',
                    'price': float(latest_price),
                }

        return {'action': 'hold', 'reason': '不在供需区间内', 'price': float(latest_price)}
