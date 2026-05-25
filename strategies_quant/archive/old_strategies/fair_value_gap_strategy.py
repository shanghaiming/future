"""
公允价值缺口策略 (Fair Value Gap Strategy)
===========================================
检测三根K线之间的价格缺口，缺口回补时交易。

来源: TradingView Smart Money Concepts - FVG

核心逻辑:
  1. 看涨FVG: 前一根高点 < 后一根低点
  2. 看跌FVG: 前一根低点 > 后一根高点
  3. 等待价格回到FVG区域入场
  4. 成交量确认缺口有效性
  5. ATR追踪止损

技术指标: FVG Detection, Volume, ATR
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class FairValueGapStrategy(BaseStrategy):
    """公允价值缺口策略 — FVG检测 + 缺口回补交易"""

    strategy_description = "FVG: 公允价值缺口检测 + 缺口回补交易"
    strategy_category = "mean_reversion"
    strategy_params_schema = {
        "min_gap_atr": {"type": "float", "default": 0.3, "label": "最小缺口(ATR倍)"},
        "max_fvg_age": {"type": "int", "default": 20, "label": "FVG最大有效K线数"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.min_gap_atr = params.get('min_gap_atr', 0.3)
        self.max_fvg_age = params.get('max_fvg_age', 20)
        self.atr_period = params.get('atr_period', 14)
        self.hold_min = params.get('hold_min', 3)
        self.trail_atr_mult = params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'min_gap_atr': 0.3, 'max_fvg_age': 20,
            'atr_period': 14, 'hold_min': 3, 'trail_atr_mult': 2.5,
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

                    if atr_val > 0:
                        if position_dir == 1 and high_water > 0:
                            if current_price < high_water - self.trail_atr_mult * atr_val:
                                should_exit = True
                        elif position_dir == -1 and low_water < float('inf'):
                            if current_price > low_water + self.trail_atr_mult * atr_val:
                                should_exit = True

                    if days_held >= 60:
                        should_exit = True

                    if not should_exit:
                        result = self._evaluate(hist)
                        if result is not None:
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

        print(f"FairValueGap: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _find_fvgs(self, high, low, close, n, atr):
        """Find all active FVGs"""
        fvgs = []
        if n < 3 or atr <= 0:
            return fvgs

        for i in range(1, n - 1):
            # Bullish FVG: high[i-1] < low[i+1]
            gap = low[i + 1] - high[i - 1]
            if gap > atr * self.min_gap_atr:
                fvgs.append({
                    'type': 'bullish',
                    'top': low[i + 1],
                    'bottom': high[i - 1],
                    'mid': (low[i + 1] + high[i - 1]) / 2,
                    'age': n - 1 - i - 1,  # bars since formation
                })
            # Bearish FVG: low[i-1] > high[i+1]
            gap = low[i - 1] - high[i + 1]
            if gap > atr * self.min_gap_atr:
                fvgs.append({
                    'type': 'bearish',
                    'top': low[i - 1],
                    'bottom': high[i + 1],
                    'mid': (low[i - 1] + high[i + 1]) / 2,
                    'age': n - 1 - i - 1,
                })

        # Filter by age
        fvgs = [f for f in fvgs if f['age'] < self.max_fvg_age]
        return fvgs

    def _evaluate(self, data):
        if len(data) < 30:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        n = len(close)
        atr = self._calc_atr(data)
        if atr <= 0:
            return None

        score = 0
        fvgs = self._find_fvgs(high, low, close, n, atr)

        current_price = close[-1]

        for fvg in fvgs:
            if fvg['type'] == 'bullish':
                # Price in bullish FVG zone = buy
                if fvg['bottom'] <= current_price <= fvg['top']:
                    score += 4
                elif current_price < fvg['bottom'] and current_price > fvg['bottom'] - atr:
                    score += 3  # Near FVG
            elif fvg['type'] == 'bearish':
                if fvg['bottom'] <= current_price <= fvg['top']:
                    score -= 4
                elif current_price > fvg['top'] and current_price < fvg['top'] + atr:
                    score -= 3

        # Volume confirmation
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        if vol_col and n >= 20:
            vol = data[vol_col].values
            vol_ma = np.mean(vol[-20:])
            if vol[-1] > vol_ma * 1.2:
                if score > 0:
                    score += 2
                elif score < 0:
                    score -= 2

        # Trend context (EMA50)
        if n >= 50:
            ema50 = np.mean(close[-50:])
            if current_price > ema50 and score > 0:
                score += 1
            elif current_price < ema50 and score < 0:
                score += 1

        direction = 1 if score > 0 else -1
        return score, direction, atr

    def _calc_atr(self, data):
        if len(data) < self.atr_period + 1:
            return 0
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        return np.mean(tr[-self.atr_period:])

    def screen(self):
        data = self.data.copy()
        if len(data) < 40:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '无信号', 'price': price}

        score, direction, _ = result
        if abs(score) >= 3:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score} (fvg)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
