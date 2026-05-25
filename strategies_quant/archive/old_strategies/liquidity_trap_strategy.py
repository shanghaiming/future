"""
流动性陷阱111%策略 (Liquidity Trap 111% Strategy)
===================================================
基于价格行为识别流动性陷阱区域，检测止损猎取后的反转交易机会。

来源: TradingView batch_3 Innovation 4/5 — Liquidity Trap 111%

核心逻辑:
  1. 枢轴点检测: 找到近期摆动高低点作为关键价格水平
  2. 流动性区域计算:
     - 止损集群区(上方): swing_high × 111%~113%   — 散户止损密集区域
     - 止损集群区(下方): swing_low  × 87%~89%    — 散户止损密集区域
     - 深度扫荡区(上方): swing_high × 127.2%      — Fibonacci扩展区域
     - 深度扫荡区(下方): swing_low  × 72.8%       — Fibonacci扩展区域

  WHY 111-113%: 散户交易者通常在摆动高点上方1-3%设置止损，
  当价格扫过这些止损后会形成大量流动性，机构借此完成建仓后反转。

  WHY 127.2%/72.8%: 对应Fibonacci 1.272扩展位，是更深层的止损集群，
  扫到此处意味着更强的反转概率。

信号逻辑:
  - 价格冲入111-113%区域后回落至摆动高点以下 → 看跌陷阱反转(卖出)
  - 价格跌破87-89%区域后回升至摆动低点以上 → 看涨陷阱反转(买入)
  - 深度扫荡至127.2%/72.8% → 更强的反转信号
  - 成交量确认: 量能 > 1.2× 均量

风险管理: ATR追踪止损

技术指标: Swing Pivots, Liquidity Zones, Volume, ATR
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class LiquidityTrapStrategy(BaseStrategy):
    """流动性陷阱策略 — 111%止损猎取区域检测 + 反转交易"""

    strategy_description = "流动性陷阱111%: 止损猎取区域检测 + 枢轴反转 + 量能确认"
    strategy_category = "price_action"
    strategy_params_schema = {
        "pivot_lookback": {"type": "int", "default": 20, "label": "枢轴回望周期"},
        "pivot_strength": {"type": "int", "default": 3, "label": "枢轴强度(左右K线数)"},
        "trap_upper_low": {"type": "float", "default": 1.11, "label": "上方陷阱下界(%)"},
        "trap_upper_high": {"type": "float", "default": 1.13, "label": "上方陷阱上界(%)"},
        "trap_lower_low": {"type": "float", "default": 0.87, "label": "下方陷阱下界(%)"},
        "trap_lower_high": {"type": "float", "default": 0.89, "label": "下方陷阱上界(%)"},
        "deep_sweep_upper": {"type": "float", "default": 1.272, "label": "深度扫荡上界(%)"},
        "deep_sweep_lower": {"type": "float", "default": 0.728, "label": "深度扫荡下界(%)"},
        "vol_mult": {"type": "float", "default": 1.2, "label": "量能确认倍数"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params=None):
        super().__init__(data, params)
        self.pivot_lookback = self.params.get('pivot_lookback', 20)
        self.pivot_strength = self.params.get('pivot_strength', 3)
        self.trap_upper_low = self.params.get('trap_upper_low', 1.11)
        self.trap_upper_high = self.params.get('trap_upper_high', 1.13)
        self.trap_lower_low = self.params.get('trap_lower_low', 0.87)
        self.trap_lower_high = self.params.get('trap_lower_high', 0.89)
        self.deep_sweep_upper = self.params.get('deep_sweep_upper', 1.272)
        self.deep_sweep_lower = self.params.get('deep_sweep_lower', 0.728)
        self.vol_mult = self.params.get('vol_mult', 1.2)
        self.atr_period = self.params.get('atr_period', 14)
        self.hold_min = self.params.get('hold_min', 3)
        self.trail_atr_mult = self.params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'pivot_lookback': 20, 'pivot_strength': 3,
            'trap_upper_low': 1.11, 'trap_upper_high': 1.13,
            'trap_lower_low': 0.87, 'trap_lower_high': 0.89,
            'deep_sweep_upper': 1.272, 'deep_sweep_lower': 0.728,
            'vol_mult': 1.2, 'atr_period': 14,
            'hold_min': 3, 'trail_atr_mult': 2.5,
        }

    def generate_signals(self):
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'

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

                # Require score >= 3 for entry
                if best_sym and abs(best_score) >= 3:
                    entry_price = float(current_bars[current_bars['symbol'] == best_sym].iloc[0]['close'])
                    if best_dir == 1:
                        self._record_signal(current_time, 'buy', best_sym, price=entry_price)
                        position_dir = 1
                    else:
                        self._record_signal(current_time, 'sell', best_sym, price=entry_price)
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

                # Update high/low water mark for trailing stop
                if position_dir == 1:
                    high_water = max(high_water, current_price) if high_water > 0 else current_price
                else:
                    low_water = min(low_water, current_price) if low_water < float('inf') else current_price

                should_exit = False

                # ATR trailing stop check after minimum hold period
                if days_held >= self.hold_min:
                    hist = data[(data['symbol'] == current_holding) & (data.index < current_time)]
                    atr_val = self._calc_atr(hist)

                    if atr_val > 0:
                        # Trailing stop: long position drops below high_water - mult*ATR
                        if position_dir == 1 and high_water > 0:
                            if current_price < high_water - self.trail_atr_mult * atr_val:
                                should_exit = True
                        # Trailing stop: short position rises above low_water + mult*ATR
                        elif position_dir == -1 and low_water < float('inf'):
                            if current_price > low_water + self.trail_atr_mult * atr_val:
                                should_exit = True

                    # Max holding period cap
                    if days_held >= 60:
                        should_exit = True

                    # Check for opposing trap signal
                    if not should_exit:
                        result = self._evaluate(hist)
                        if result is not None:
                            score, direction, _ = result
                            # Opposing signal: exit if strong enough
                            if position_dir == 1 and direction == -1 and score < -3:
                                should_exit = True
                            elif position_dir == -1 and direction == 1 and score > 3:
                                should_exit = True

                if should_exit:
                    if position_dir == 1:
                        self._record_signal(current_time, 'sell', current_holding,
                                            price=current_price)
                    else:
                        self._record_signal(current_time, 'buy', current_holding,
                                            price=current_price)
                    current_holding = None
                    buy_time = None
                    position_dir = 0
                    high_water = 0.0
                    low_water = float('inf')

        print(f"LiquidityTrap: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _find_pivots(self, high, low, n):
        """Find swing high and swing low pivot points.

        A swing high at index i requires the pivot_strength bars on each side
        to all be <= high[i]. Similarly for swing lows.

        Returns lists of (index, price) for swing highs and swing lows.
        """
        strength = self.pivot_strength
        if n < strength * 2 + 1:
            return [], []

        swing_highs = []
        swing_lows = []

        for i in range(strength, n - strength):
            is_high = True
            is_low = True
            for j in range(-strength, strength + 1):
                if j == 0:
                    continue
                if high[i] < high[i + j]:
                    is_high = False
                if low[i] > low[i + j]:
                    is_low = False
                if not is_high and not is_low:
                    break

            if is_high:
                swing_highs.append((i, high[i]))
            if is_low:
                swing_lows.append((i, low[i]))

        return swing_highs, swing_lows

    def _detect_trap(self, data):
        """Detect liquidity trap reversal pattern.

        The core pattern:
        1. Identify the most recent swing high and swing low
        2. Calculate trap zones as percentage extensions of these pivots
        3. Check if recent price action swept into a trap zone then reversed

        WHY this works: Retail traders cluster stops just beyond swing points.
        When price reaches these zones, stop orders are triggered creating a
        burst of liquidity. Institutional traders use this liquidity to fill
        their opposing orders, causing a reversal.
        """
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)

        min_required = self.pivot_strength * 2 + 5
        if n < min_required:
            return 0

        swing_highs, swing_lows = self._find_pivots(high, low, n)
        score = 0

        # Need at least one swing high and one swing low
        if not swing_highs and not swing_lows:
            return 0

        # Most recent bar's price action
        recent_high = high[-1]
        recent_low = low[-1]
        recent_close = close[-1]
        prev_close = close[-2] if n >= 2 else close[-1]

        # --- Bearish trap detection (price sweeps above swing high, then reverses) ---
        # WHY: Price sweeps above a swing high to trigger buy-stops (breakout traders),
        # then reverses below the swing high. This is a "bull trap" — trapped longs.
        if swing_highs:
            # Use the most recent swing highs (last few for context)
            for idx, sh_price in swing_highs[-3:]:
                trap_zone_low = sh_price * self.trap_upper_low
                trap_zone_high = sh_price * self.trap_upper_high
                deep_zone = sh_price * self.deep_sweep_upper

                # Standard trap: high swept into 111-113% zone, closed back below swing high
                if recent_high >= trap_zone_low and recent_close < sh_price:
                    score -= 4  # Bearish trap reversal — sell signal

                    # Deep sweep: even stronger signal
                    if recent_high >= deep_zone:
                        score -= 3  # Additional strength from deep sweep

                # Narrower sweep: high just exceeded trap zone upper bound
                elif trap_zone_low <= recent_high <= trap_zone_high and recent_close < sh_price:
                    score -= 3

        # --- Bullish trap detection (price sweeps below swing low, then reverses) ---
        # WHY: Price sweeps below a swing low to trigger sell-stops (breakdown traders),
        # then reverses above the swing low. This is a "bear trap" — trapped shorts.
        if swing_lows:
            for idx, sl_price in swing_lows[-3:]:
                trap_zone_low = sl_price * self.trap_lower_low
                trap_zone_high = sl_price * self.trap_lower_high
                deep_zone = sl_price * self.deep_sweep_lower

                # Standard trap: low swept into 87-89% zone, closed back above swing low
                if recent_low <= trap_zone_high and recent_close > sl_price:
                    score += 4  # Bullish trap reversal — buy signal

                    # Deep sweep: even stronger signal
                    if recent_low <= deep_zone:
                        score += 3  # Additional strength from deep sweep

                # Narrower sweep
                elif trap_zone_low <= recent_low <= trap_zone_high and recent_close > sl_price:
                    score += 3

        # --- Volume confirmation ---
        # WHY: A trap reversal with high volume confirms institutional participation.
        # Low volume sweeps are less reliable — may just be noise.
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        if vol_col and n >= 20:
            vol = data[vol_col].values
            vol_ma = np.mean(vol[-20:])
            if vol_ma > 0 and vol[-1] > vol_ma * self.vol_mult:
                if score > 0:
                    score += 2  # Volume confirms bullish trap
                elif score < 0:
                    score -= 2  # Volume confirms bearish trap

        return score

    def _evaluate(self, data):
        """Evaluate liquidity trap signal.

        Returns (score, direction, atr) or None.
        """
        min_required = self.pivot_lookback + self.pivot_strength * 2 + 5
        if len(data) < min_required:
            return None

        score = self._detect_trap(data)
        direction = 1 if score > 0 else -1
        atr = self._calc_atr(data)
        return score, direction, atr

    def _calc_atr(self, data):
        """Calculate Average True Range."""
        if len(data) < self.atr_period + 1:
            return 0
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:] - close[:-1])))
        return np.mean(tr[-self.atr_period:])

    def screen(self):
        """Real-time screening based on latest bar liquidity trap signal."""
        data = self.data.copy()
        min_required = self.pivot_lookback + self.pivot_strength * 2 + 5
        if len(data) < min_required:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '无枢轴数据', 'price': price}

        score, direction, _ = result
        if abs(score) >= 3:
            action = 'buy' if direction == 1 else 'sell'
            trap_type = 'bull_trap' if direction == 1 else 'bear_trap'
            return {
                'action': action,
                'reason': f"score={score} {trap_type} (liquidity_trap_111)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
