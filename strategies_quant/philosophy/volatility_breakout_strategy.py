"""
Volatility Breakout Strategy (波动突破策略)
=============================================
基于布林带带宽百分位的波动突破策略，结合VDP方向判断和ATR追踪止损。

核心逻辑:
1. 布林带带宽百分位检测:
   - 20日布林带，2倍标准差
   - 带宽百分位 < 20: 压缩状态 (低波动 → 即将突破)
   - 带宽百分位 > 80: 扩张状态 (高波动 → 动量延续)
2. 方向判断:
   - 收盘价在布林带中的位置
   - VDP (Volume Delta Pressure) 方向
3. 压缩状态:
   - 准备阶段，等待突破
   - 首根突破K线入场
4. 扩张状态:
   - 顺势而为，追踪止损
   - 20天最大持仓周期
5. 评分机制:
   - 带宽百分位 + VDP对齐 + 动量强度
6. 风险控制:
   - ATR追踪止损
   - 最大持仓20天

技术指标: Bollinger Bands, VDP, ATR, Volatility Percentile
"""

import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class VolatilityBreakoutStrategy(BaseStrategy):
    """波动突破策略 - BB带宽百分位 + VDP方向 + ATR追踪止损"""

    strategy_description = "波动突破: BB带宽百分位(<20压缩,>80扩张) + VDP方向 + ATR追踪止损"
    strategy_category = "volatility"
    strategy_params_schema = {
        "bb_period": {"type": "int", "default": 20, "label": "BB周期"},
        "bb_std": {"type": "float", "default": 2.0, "label": "BB标准差倍数"},
        "vol_window": {"type": "int", "default": 100, "label": "波动率百分位窗口"},
        "squeeze_threshold": {"type": "float", "default": 0.2, "label": "压缩阈值(20%)"},
        "expansion_threshold": {"type": "float", "default": 0.8, "label": "扩张阈值(80%)"},
        "vdp_period": {"type": "int", "default": 5, "label": "VDP周期"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
        "max_hold_days": {"type": "int", "default": 20, "label": "最大持仓天数"},
        "min_score": {"type": "float", "default": 0.5, "label": "最小入场评分"},
        "volume_min": {"type": "int", "default": 1000000, "label": "最小成交量要求"},
    }

    def get_default_params(self):
        return {
            'bb_period': 20, 'bb_std': 2.0, 'vol_window': 100,
            'squeeze_threshold': 0.2, 'expansion_threshold': 0.8,
            'vdp_period': 5, 'atr_period': 14,
            'trail_atr_mult': 2.5, 'max_hold_days': 20,
            'min_score': 0.5, 'volume_min': 1000000,
        }

    def validate_params(self):
        p = self.params
        if p['bb_period'] < 5:
            raise ValueError("bb_period must be >= 5")
        if p['bb_std'] < 1.0:
            raise ValueError("bb_std must be >= 1.0")
        if p['vol_window'] < 20:
            raise ValueError("vol_window must be >= 20")
        if p['squeeze_threshold'] >= p['expansion_threshold']:
            raise ValueError("squeeze_threshold must be < expansion_threshold")
        if p['vdp_period'] < 1:
            raise ValueError("vdp_period must be >= 1")
        if p['atr_period'] < 1:
            raise ValueError("atr_period must be >= 1")
        if p['trail_atr_mult'] < 1.0:
            raise ValueError("trail_atr_mult must be >= 1.0")
        if p['max_hold_days'] < 1:
            raise ValueError("max_hold_days must be >= 1")
        if p['min_score'] < 0 or p['min_score'] > 1:
            raise ValueError("min_score must be between 0 and 1")
        if p['volume_min'] < 0:
            raise ValueError("volume_min must be >= 0")

    def _calculate_bollinger_bands(self, data, period, std_mult):
        """计算布林带: mid, upper, lower, width"""
        s = pd.Series(data)
        mid = s.rolling(period).mean()
        std = s.rolling(period).std()
        std = std.fillna(0)

        upper = mid + std_mult * std
        lower = mid - std_mult * std
        width = (upper - lower) / np.where(np.abs(mid) < 1e-8, 1e-8, np.abs(mid))

        return mid, upper, lower, width

    def _calculate_volatility_percentile(self, width_series, window):
        """计算带宽百分位"""
        percentile = width_series.rolling(window).rank(pct=True)
        return percentile.fillna(0)

    def _calculate_vdp(self, df, period):
        """计算VDP (Volume Delta Pressure)"""
        if 'volume' not in df.columns:
            return pd.Series(0, index=df.index)

        # VDP = volume * (2*close - high - low) / (high - low)
        h, l, c, v = df['high'], df['low'], df['close'], df['volume']

        # 避免除零
        range_val = np.where((h - l) < 1e-8, 1e-8, h - l)
        vdp = v * (2 * c - h - l) / range_val

        # 平滑处理
        vdp_smooth = vdp.rolling(period).mean()
        return vdp_smooth.fillna(0)

    def _calculate_atr(self, df, period):
        """计算ATR (Average True Range)"""
        h, l, c = df['high'], df['low'], df['close']

        # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
        prev_close = c.shift(1)
        tr1 = h - l
        tr2 = np.abs(h - prev_close)
        tr3 = np.abs(l - prev_close)
        tr = np.maximum(np.maximum(tr1, tr2), tr3)

        # ATR
        atr = tr.rolling(period).mean()
        return atr.fillna(0)

    def _calculate_score(self, bb_percentile, vdp, close, bb_mid, position_in_bb, regime):
        """计算入场评分"""
        score = 0.0

        # 1. 带宽百分位得分 (0.3权重)
        if regime == 'squeeze':
            # 压缩状态下，带宽越小越好
            score += 0.3 * (1 - bb_percentile)
        elif regime == 'expansion':
            # 扩张状态下，带宽越大越好
            score += 0.3 * bb_percentile

        # 2. VDP方向得分 (0.3权重)
        vdp_strength = np.abs(vdp) / (np.abs(vdp).mean() + 1e-8)
        if vdp > 0:  # 正VDP
            score += 0.3 * vdp_strength
        else:  # 负VDP
            score -= 0.3 * vdp_strength

        # 3. 位置得分 (0.2权重)
        if position_in_bb > 0.6:  # 靠近上轨
            score += 0.2
        elif position_in_bb < 0.4:  # 靠近下轨
            score -= 0.2

        # 4. 动量得分 (0.2权重)
        momentum = (close - bb_mid) / np.abs(bb_mid + 1e-8)
        score += 0.2 * np.tanh(momentum * 2)  # tanh限制在[-1,1]

        return np.clip(score, -1, 1)

    def generate_signals(self):
        df = self.data.copy()
        p = self.params
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'DEFAULT'

        # 确保有成交量数据
        if 'volume' not in df.columns or df['volume'].isna().all():
            print(f"VolatilityBreakout: {symbol} 缺少成交量数据")
            return []

        # 检查最小成交量
        if (df['volume'] < p['volume_min']).all():
            print(f"VolatilityBreakout: {symbol} 成交量低于要求")
            return []

        # 计算技术指标
        mid, upper, lower, width = self._calculate_bollinger_bands(
            df['close'], p['bb_period'], p['bb_std']
        )
        bb_percentile = self._calculate_volatility_percentile(width, p['vol_window'])
        vdp = self._calculate_vdp(df, p['vdp_period'])
        atr = self._calculate_atr(df, p['atr_period'])

        # 添加到DataFrame
        df['bb_mid'] = mid
        df['bb_upper'] = upper
        df['bb_lower'] = lower
        df['bb_percentile'] = bb_percentile
        df['vdp'] = vdp
        df['atr'] = atr

        # 识别状态
        df['regime'] = 'normal'
        df.loc[df['bb_percentile'] < p['squeeze_threshold'], 'regime'] = 'squeeze'
        df.loc[df['bb_percentile'] > p['expansion_threshold'], 'regime'] = 'expansion'

        # 计算位置百分比
        df['position_in_bb'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        df['position_in_bb'] = df['position_in_bb'].clip(0, 1)

        # 计算评分
        df['score'] = df.apply(
            lambda row: self._calculate_score(
                row['bb_percentile'], row['vdp'], row['close'],
                row['bb_mid'], row['position_in_bb'], row['regime']
            ), axis=1
        )

        # 生成信号
        signals = []
        in_position = False
        position_dir = 0
        entry_time = None
        trail_stop = None
        entry_price = 0

        for i in range(1, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]

            if not in_position:
                # 寻找入场机会
                if current['regime'] == 'squeeze':
                    # 压缩状态：等待突破
                    if (prev['close'] <= prev['bb_upper'] and
                        current['close'] > current['bb_upper']):
                        # 向上突破
                        if current['score'] >= p['min_score']:
                            signals.append({
                                'timestamp': current.name,
                                'action': 'buy',
                                'symbol': symbol,
                                'price': float(current['close']),
                                'score': float(current['score']),
                                'regime': 'squeeze_breakout',
                                'atr': float(current['atr'])
                            })
                            in_position = True
                            position_dir = 1
                            entry_time = current.name
                            entry_price = current['close']
                            trail_stop = current['close'] - p['trail_atr_mult'] * current['atr']

                elif current['regime'] == 'expansion':
                    # 扩张状态：顺势而为
                    if current['vdp'] > 0 and current['score'] >= p['min_score']:
                        signals.append({
                            'timestamp': current.name,
                            'action': 'buy',
                            'symbol': symbol,
                            'price': float(current['close']),
                            'score': float(current['score']),
                            'regime': 'expansion_momentum',
                            'atr': float(current['atr'])
                        })
                        in_position = True
                        position_dir = 1
                        entry_time = current.name
                        entry_price = current['close']
                        trail_stop = current['close'] - p['trail_atr_mult'] * current['atr']

            else:
                # 持仓中，检查出场条件
                days_held = (current.name - entry_time).days if hasattr(entry_time, 'days') else 0

                # 1. ATR追踪止损
                new_trail_stop = current['high'] - p['trail_atr_mult'] * current['atr']
                if position_dir == 1:  # 多头
                    trail_stop = max(trail_stop, new_trail_stop)
                    if current['low'] <= trail_stop or days_held >= p['max_hold_days']:
                        signals.append({
                            'timestamp': current.name,
                            'action': 'sell',
                            'symbol': symbol,
                            'price': float(trail_stop),
                            'reason': 'atr_stop' if current['low'] <= trail_stop else 'max_hold',
                            'days_held': days_held,
                            'atr': float(current['atr'])
                        })
                        in_position = False
                        position_dir = 0
                        entry_time = None

                # 2. 布林带下轨止损（扩张状态）
                elif current['regime'] == 'expansion' and current['close'] <= current['bb_lower']:
                    signals.append({
                        'timestamp': current.name,
                        'action': 'sell',
                        'symbol': symbol,
                        'price': float(current['bb_lower']),
                        'reason': 'bb_lower_stop',
                        'days_held': days_held,
                        'atr': float(current['atr'])
                    })
                    in_position = False
                    position_dir = 0
                    entry_time = None

        self.signals = signals
        return signals