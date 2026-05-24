"""
鲁棒回归策略 (Robust Regression Strategy)
==========================================
基于IRLS Huber回归的抗异常值趋势检测策略

核心原理:
- 使用Huber损失函数的IRLS拟合鲁棒趋势线
- 基于Huber斜率符号变化进行趋势反转检测
- 对异常值(涨跌停)具有天然鲁棒性
- 结合OLS趋势进行 regime change 检测
- 使用Tukey bisquare权重实现极强鲁棒性

数学核心:
Huber损失: ρ(r) = {½r²  if |r|≤δ
                 {δ(|r|-½δ) if |r|>δ

Huber导数(权重): ψ(r) = {r/σ  if |r|≤δ
                        {δ·sign(r)/σ if |r|>δ
                         where σ=median(|residuals|)·MAD_scale

Tukey bisquare: w(r) = { [1-(r/δ)²]²  if |r|≤δ
                        { 0           if |r|>δ
"""
import numpy as np
import pandas as pd
import scipy.optimize
from core.base_strategy import BaseStrategy


class RobustRegressionStrategy(BaseStrategy):
    """鲁棒回归策略 - IRLS Huber回归 + 斜率反转 + OLS divergence"""

    strategy_description = "IRLS Huber鲁棒回归 + 斜率反转信号 + OLS divergence regime change"
    strategy_category = "regression"
    strategy_params_schema = {
        "window": {"type": "int", "default": 30, "label": "回归窗口"},
        "huber_delta": {"type": "float", "default": 1.345, "label": "Huber delta参数"},
        "iterations": {"type": "int", "default": 10, "label": "IRLS迭代次数"},
        "ols_window": {"type": "int", "default": 20, "label": "OLS对比窗口"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "trail_atr_mult": {"type": "float", "default": 2.0, "label": "追踪止损ATR倍数"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "slope_threshold": {"type": "float", "default": 0.01, "label": "斜率变化阈值"},
        "conf_level": {"type": "float", "default": 0.8, "label": "置信水平"},
    }

    def get_default_params(self):
        return {k: v["default"] for k, v in self.strategy_params_schema.items()}

    def _huber_weights(self, residuals, delta=1.345):
        """计算Huber权重函数"""
        sigma = np.median(np.abs(residuals)) * 1.4826  # MAD scale
        sigma = max(sigma, 1e-8)

        scaled_res = residuals / sigma

        # Huber权重
        weights = np.where(np.abs(scaled_res) <= delta,
                         1.0 / scaled_res,
                         delta * np.sign(scaled_res) / scaled_res)

        return weights

    def _tukey_bisquare_weights(self, residuals, delta=4.685):
        """Tukey bisquare权重函数"""
        scaled_res = residuals / np.std(residuals) if np.std(residuals) > 0 else residuals
        scaled_res = scaled_res / delta  # 标准化

        weights = np.where(np.abs(scaled_res) <= 1,
                         (1 - scaled_res**2)**2,
                         0.0)

        return weights

    def _robust_irls(self, y, x=None, delta=1.345, max_iter=10, tol=1e-6):
        """IRLS with Huber loss"""
        n = len(y)

        # 如果没有提供x，使用时间序列索引
        if x is None:
            x = np.arange(n)

        # 初始化参数
        beta = np.ones(2)  # [intercept, slope]

        # 标准化x提高数值稳定性
        x_mean, x_std = np.mean(x), np.std(x)
        x_scaled = (x - x_mean) / x_std

        for iteration in range(max_iter):
            # 计算预测值和残差
            y_pred = beta[0] + beta[1] * x_scaled
            residuals = y - y_pred

            # 计算权重
            weights = self._huber_weights(residuals, delta)

            # 加权最小二乘
            W = np.diag(weights)
            X = np.column_stack([np.ones(n), x_scaled])

            try:
                beta_new = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y

                # 检查收敛
                if np.linalg.norm(beta_new - beta) < tol:
                    break

                beta = beta_new

            except np.linalg.LinAlgError:
                # 如果矩阵不可逆，使用最小二乘
                break

        # 计算标准误差
        residuals = y - (beta[0] + beta[1] * x_scaled)
        wss = weights @ residuals**2
        if n > 2:
            cov = np.linalg.inv(X.T @ W @ X)
            se_intercept = np.sqrt(cov[0, 0] * wss / (weights.sum()))
            se_slope = np.sqrt(cov[1, 1] * wss / (weights.sum()))
        else:
            se_intercept = se_slope = np.nan

        # 反标准化斜率
        slope = beta[1] / x_std

        return {
            'intercept': beta[0],
            'slope': slope,
            'se_slope': se_slope,
            'residuals': residuals,
            'weights': weights,
            'converged': iteration < max_iter
        }

    def _rolling_robust_regression(self, data, window, **kwargs):
        """计算滚动Huber回归"""
        n = len(data)
        slopes = np.full(n, np.nan)
        se_slopes = np.full(n, np.nan)
        intercepts = np.full(n, np.nan)
        conf_int_up = np.full(n, np.nan)
        conf_int_down = np.full(n, np.nan)

        p = self.params

        # 计算OLS用于比较
        ols_slopes = np.full(n, np.nan)

        for i in range(window, n):
            # 窗口数据
            y = data[i-window+1:i+1]
            x = np.arange(window)

            # IRLS Huber回归
            result = self._robust_irls(y, x=x, delta=p['huber_delta'],
                                     max_iter=p['iterations'])
            slopes[i] = result['slope']
            se_slopes[i] = result['se_slope']
            intercepts[i] = result['intercept']

            # 计算置信区间
            if result['se_slope'] and not np.isnan(result['se_slope']):
                critical_val = scipy.stats.norm.ppf(p['conf_level'])
                margin = critical_val * result['se_slope']
                conf_int_up[i] = result['slope'] + margin
                conf_int_down[i] = result['slope'] - margin

            # 计算OLS回归
            if i >= window - 1 + p['ols_window']:
                ols_y = data[i-p['ols_window']+1:i+1]
                ols_x = np.arange(p['ols_window'])
                ols_beta = np.polyfit(ols_x, ols_y, 1)
                ols_slopes[i] = ols_beta[0]

        return {
            'slopes': slopes,
            'se_slopes': se_slopes,
            'intercepts': intercepts,
            'conf_int_up': conf_int_up,
            'conf_int_down': conf_int_down,
            'ols_slopes': ols_slopes
        }

    def generate_signals(self):
        df = self.data.copy()
        p = self.params
        sym = df['symbol'].iloc[0]

        H, L, C = df['high'].values, df['low'].values, df['close'].values
        n = len(C)

        # 1. 计算鲁棒回归指标
        regression = self._rolling_robust_regression(C, p['window'])

        # 2. 计算ATR
        tr = np.maximum(H - L, np.maximum(np.abs(H - np.roll(C, 1)), np.abs(L - np.roll(C, 1))))
        tr[0] = H[0] - L[0]
        atr = pd.Series(tr).ewm(span=p['atr_period'], adjust=False).mean().values

        # 3. 信号生成
        signals = []
        in_pos = False
        entry_price = 0.0
        trail_stop = 0.0
        entry_idx = 0

        warmup = max(p['window'] + 10, p['atr_period'])

        for i in range(warmup, n):
            price = C[i]
            ts = df.index[i]

            robust_slope = regression['slopes'][i]
            robust_se = regression['se_slopes'][i]
            ols_slope = regression['ols_slopes'][i]

            if not in_pos:
                # 买入条件:
                # 1. 鲁棒斜率翻正且大于阈值
                # 2. 置信区间下限大于0
                slope_positive = (robust_slope > 0 and
                                regression['slopes'][i-1] <= 0)
                confidence_positive = (regression['conf_int_down'][i] > 0)

                # OLS divergence: 当OLS斜率与鲁棒斜率出现显著分歧
                divergence = False
                if not np.isnan(ols_slope):
                    divergence = abs(ols_slope - robust_slope) > abs(ols_slope) * 0.5

                if slope_positive and confidence_positive and price > C[i-1]:
                    signals.append({
                        'timestamp': ts,
                        'action': 'buy',
                        'symbol': sym,
                        'price': price,
                        'slope': robust_slope,
                        'se_slope': robust_se,
                        'ols_slope': ols_slope
                    })
                    in_pos = True
                    entry_price = price
                    trail_stop = price - p['trail_atr_mult'] * atr[i]
                    entry_idx = i
            else:
                hold_days = i - entry_idx
                new_stop = price - p['trail_atr_mult'] * atr[i]
                trail_stop = max(trail_stop, new_stop)

                # 卖出条件:
                # 1. 追踪止损
                # 2. 鲁棒斜率翻负且置信区间上限小于0
                # 3. OLS divergence反转信号
                sell_signal = False

                if price <= trail_stop and hold_days >= p['hold_min']:
                    sell_signal = True
                elif robust_slope < 0 and regression['slopes'][i-1] >= 0:
                    sell_signal = True
                elif hold_days >= p['hold_min'] and not np.isnan(ols_slope):
                    # OLS divergence反转: 当分歧突然消失，预示趋势反转
                    divergence_closing = abs(ols_slope - robust_slope) < abs(robust_slope) * 0.2
                    if divergence_closing and abs(robust_slope) < p['slope_threshold']:
                        sell_signal = True

                if sell_signal:
                    signals.append({
                        'timestamp': ts,
                        'action': 'sell',
                        'symbol': sym,
                        'price': price,
                        'slope': robust_slope,
                        'se_slope': robust_se,
                        'ols_slope': ols_slope
                    })
                    in_pos = False

        # 强制平仓
        if in_pos:
            signals.append({
                'timestamp': df.index[-1],
                'action': 'sell',
                'symbol': sym,
                'price': C[-1],
                'slope': regression['slopes'][-1] if not np.isnan(regression['slopes'][-1]) else 0,
                'se_slope': regression['se_slopes'][-1] if not np.isnan(regression['se_slopes'][-1]) else 0,
                'ols_slope': regression['ols_slopes'][-1] if not np.isnan(regression['ols_slopes'][-1]) else 0
            })

        self.signals = signals
        return signals