"""
迭代重加权最小二乘策略 (IRLS - Iteratively Reweighted Least Squares Strategy)
============================================================================
基于L1-robust价格平滑器 + Hardy权重函数 + 稀疏重加权 + 斜率翻转信号。

来源: TradingView Batch 2 Extended — IRLS [Jamallo] (Innovation 5/5)

核心逻辑:
  1. IRLS平滑器:
     - 对每个bar, 在window窗口内用Hardy权重函数计算加权回归
     - w(j) = 1 / sqrt(dist(j)^2 + epsilon^2)
     - epsilon自适应: 使用平均(H-L)范围
     - 稀疏重加权: 只取top s_ratio*N个最近邻样本投票
  2. 多次迭代(默认3次):
     - 每次迭代用上一次的残差更新权重
     - 残差大的点权重降低 → 对异常值(涨跌停)鲁棒
  3. 信号:
     - 买入: IRLS斜率翻正 + 价格在IRLS线上方
     - 卖出: IRLS斜率翻负 + ATR追踪止损

WHY this works:
  - L1-robust vs L2(Gaussian/NW): 对异常值(涨跌停)天然鲁棒
  - Hardy权重函数: 比Gaussian衰减更慢(保留更多远端信息), 比均匀权重更抗噪
  - 稀疏控制: 只让最近的N%样本投票, 减少远端噪音干扰
  - 与NW(+75%)同属非参数方法, 但IRLS更抗异常值 → A股涨跌停环境更合适

数学核心: w(j)=1/√(d²+ε²), IRLS迭代: β_new = (X'WX)⁻¹X'Wy, W=diag(w)
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class IRLSStrategy(BaseStrategy):
    """IRLS策略 — L1-robust Hardy权重平滑器+斜率翻转+ATR止损"""

    strategy_description = "IRLS: 迭代重加权最小二乘Hardy权重平滑器+斜率翻转+ATR止损"
    strategy_category = "regression"
    strategy_params_schema = {
        "window": {"type": "int", "default": 15, "label": "IRLS窗口"},
        "iterations": {"type": "int", "default": 3, "label": "IRLS迭代次数"},
        "sparsity_ratio": {"type": "float", "default": 0.3, "label": "稀疏比(0-1)"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
    }

    def get_default_params(self):
        return {k: v["default"] for k, v in self.strategy_params_schema.items()}

    def _irls_smooth(self, data):
        """IRLS平滑: Hardy权重函数+稀疏重加权+多次迭代"""
        n = len(data)
        result = np.copy(data)
        p = self.params
        window = p['window']
        iters = p['iterations']
        s_ratio = p['sparsity_ratio']

        # 自适应epsilon: 使用平均H-L范围
        if 'high' in self.data.columns and 'low' in self.data.columns:
            hl_range = (self.data['high'].values - self.data['low'].values)
            epsilon = np.nanmean(hl_range) * 0.5
        else:
            epsilon = np.nanstd(data) * 0.1

        epsilon = max(epsilon, 1e-8)

        for i in range(window, n):
            # 窗口数据
            y = data[i - window:i + 1]
            n_w = len(y)

            # 初始权重: 均匀
            weights = np.ones(n_w)

            for _ in range(iters):
                # 加权最小二乘: 简单加权平均作为局部估计
                w_sum = weights.sum()
                if w_sum < 1e-12:
                    break
                y_hat = np.sum(weights * y) / w_sum

                # 残差
                residuals = y - y_hat

                # Hardy权重: w(j) = 1/sqrt(r(j)^2 + epsilon^2)
                new_weights = 1.0 / np.sqrt(residuals ** 2 + epsilon ** 2)

                # 稀疏: 只保留top s_ratio的权重
                sorted_idx = np.argsort(new_weights)[::-1]
                k = max(1, int(s_ratio * n_w))
                sparse_mask = np.zeros(n_w, dtype=bool)
                sparse_mask[sorted_idx[:k]] = True

                weights = np.where(sparse_mask, new_weights, 0.0)

            # 最终估计
            w_sum = weights.sum()
            if w_sum > 1e-12:
                result[i] = np.sum(weights * y) / w_sum

        return result

    def generate_signals(self):
        df = self.data.copy()
        p = self.params
        sym = df['symbol'].iloc[0]

        H, L, C = df['high'].values, df['low'].values, df['close'].values
        n = len(C)

        # 1. IRLS平滑
        irls = self._irls_smooth(C)

        # 2. IRLS斜率
        slope = np.zeros(n)
        for i in range(1, n):
            slope[i] = irls[i] - irls[i - 1]

        # 3. ATR
        tr = np.maximum(H - L, np.maximum(np.abs(H - np.roll(C, 1)), np.abs(L - np.roll(C, 1))))
        tr[0] = H[0] - L[0]
        atr = pd.Series(tr).ewm(span=p['atr_period'], adjust=False).mean().values

        # 4. 信号生成
        signals = []
        in_pos = False
        entry_price = 0.0
        trail_stop = 0.0
        entry_idx = 0

        warmup = max(p['window'] + 5, p['atr_period'])

        for i in range(warmup, n):
            price = C[i]
            ts = df.index[i]

            if not in_pos:
                # 买入: IRLS斜率翻正 + 价格在IRLS线上方
                if slope[i] > 0 and slope[i - 1] <= 0 and price > irls[i]:
                    signals.append({
                        'timestamp': ts, 'action': 'buy', 'symbol': sym, 'price': price
                    })
                    in_pos = True
                    entry_price = price
                    trail_stop = price - p['trail_atr_mult'] * atr[i]
                    entry_idx = i
            else:
                hold_days = i - entry_idx
                new_stop = price - p['trail_atr_mult'] * atr[i]
                trail_stop = max(trail_stop, new_stop)

                # 卖出: 追踪止损 OR 斜率翻负
                sell_signal = False
                if price <= trail_stop and hold_days >= p['hold_min']:
                    sell_signal = True
                elif slope[i] < 0 and slope[i - 1] >= 0 and hold_days >= p['hold_min']:
                    sell_signal = True

                if sell_signal:
                    signals.append({
                        'timestamp': ts, 'action': 'sell', 'symbol': sym, 'price': price
                    })
                    in_pos = False

        # 强制平仓
        if in_pos:
            signals.append({
                'timestamp': df.index[-1], 'action': 'sell', 'symbol': sym,
                'price': C[-1]
            })

        self.signals = signals
        return signals
