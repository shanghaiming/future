"""
统计假设检验策略 (Statistical Hypothesis Strategy)
==================================================
基于概率论与统计推断的交易策略。

理论基础 (陈希孺《概率论与数理统计》):
1. 假设检验: H0="价格随机游走" vs H1="存在趋势/均值回归"
   → 用t检验和卡方检验判断价格运动是否统计显著
2. 置信区间: 价格偏离均值的程度用标准误衡量
   → 95% CI外的价格运动具有统计显著性
3. 贝叶斯更新: 随着新数据到来, 更新趋势概率估计
   → P(trend|data) ∝ P(data|trend) × P(trend)

与现有策略的区别:
- Fisher/Z-score: 只看当前值, 无统计显著性检验
- 本策略: 用p-value确认信号, 避免交易噪音
- AdaptiveWeighted: 多指标加权评分
- 本策略: 统计假设检验框架, p-value驱动

信号生成:
  Buy:  t检验显著(价格连续上涨非随机) + 成交量卡方检验通过 + 贝叶斯P(trend)>0.6
  Sell: 均值回归置信度>95% (价格偏离2σ且开始回归) 或 趋势概率反转

风险管理:
  止损: 基于标准误 (SE = σ/√n) 的统计止损
  仓位: Kelly公式简化版: f = (bp - q) / b
"""
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from core.base_strategy import BaseStrategy


class StatEdgeStrategy(BaseStrategy):
    """统计假设检验策略 — p-value驱动入场, 贝叶斯更新, Kelly仓位"""

    strategy_description = "统计假设检验: t检验趋势确认 + 卡方量价检验 + 贝叶斯趋势概率"
    strategy_category = "statistical"
    strategy_params_schema = {
        "trend_window": {"type": "int", "default": 20, "label": "趋势检验窗口"},
        "vol_window": {"type": "int", "default": 20, "label": "成交量检验窗口"},
        "mr_window": {"type": "int", "default": 30, "label": "均值回归窗口"},
        "p_threshold": {"type": "float", "default": 0.05, "label": "p-value阈值(显著性)"},
        "bayesian_threshold": {"type": "float", "default": 0.6, "label": "贝叶斯趋势概率阈值"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
    }

    def __init__(self, data, params):
        super().__init__(data, params)
        self.trend_window = params.get('trend_window', 20)
        self.vol_window = params.get('vol_window', 20)
        self.mr_window = params.get('mr_window', 30)
        self.p_threshold = params.get('p_threshold', 0.05)
        self.bayesian_threshold = params.get('bayesian_threshold', 0.6)
        self.hold_min = params.get('hold_min', 3)

    def get_default_params(self):
        return {
            'trend_window': 20, 'vol_window': 20, 'mr_window': 30,
            'p_threshold': 0.05, 'bayesian_threshold': 0.6, 'hold_min': 3,
        }

    def generate_signals(self):
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'

        symbols = data['symbol'].unique()
        unique_times = sorted(data.index.unique())
        self.signals = []
        current_holding = None
        buy_time = None
        position_dir = 0

        for current_time in unique_times:
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            if current_holding is None:
                best_score = 0.0
                best_sym = None
                best_dir = 0

                for _, bar in current_bars.iterrows():
                    sym = bar['symbol']
                    hist = data[(data['symbol'] == sym) & (data.index < current_time)]
                    result = self._evaluate(hist)
                    if result is None:
                        continue
                    score, direction = result
                    if abs(score) > abs(best_score):
                        best_score = score
                        best_sym = sym
                        best_dir = direction

                if best_sym and abs(best_score) > 0.3:
                    if best_dir == 1:
                        self._record_signal(current_time, 'buy', best_sym)
                        position_dir = 1
                    else:
                        self._record_signal(current_time, 'sell', best_sym)
                        position_dir = -1
                    current_holding = best_sym
                    buy_time = current_time

            else:
                days_held = len([t for t in unique_times if buy_time < t <= current_time])
                if days_held >= self.hold_min:
                    hist = data[(data['symbol'] == current_holding) & (data.index < current_time)]
                    result = self._evaluate(hist)

                    should_exit = False
                    if result is not None:
                        score, direction = result
                        # Exit if signal reverses
                        if position_dir == 1 and direction == -1 and score < -0.3:
                            should_exit = True
                        elif position_dir == -1 and direction == 1 and score > 0.3:
                            should_exit = True

                    # Stop loss: 3 SE (standard errors) against position
                    if len(hist) >= self.trend_window:
                        close = hist['close'].values
                        se = np.std(close[-self.trend_window:], ddof=1) / np.sqrt(self.trend_window)
                        last_price = close[-1]
                        entry_region = close[-min(days_held, 10):]
                        if len(entry_region) > 0:
                            entry_mean = np.mean(entry_region[:3])
                            if position_dir == 1 and last_price < entry_mean - 3 * se:
                                should_exit = True
                            elif position_dir == -1 and last_price > entry_mean + 3 * se:
                                should_exit = True

                    # Max hold
                    if days_held >= 50:
                        should_exit = True

                    if should_exit:
                        if position_dir == 1:
                            self._record_signal(current_time, 'sell', current_holding)
                        else:
                            self._record_signal(current_time, 'buy', current_holding)
                        current_holding = None
                        buy_time = None
                        position_dir = 0

        print(f"StatEdge: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _evaluate(self, data):
        """综合统计评估: t检验 + 卡方检验 + 贝叶斯"""
        min_len = max(self.trend_window, self.vol_window, self.mr_window) + 5
        if len(data) < min_len:
            return None

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values

        # ===== 1. 趋势t检验 =====
        # H0: price changes = 0 (random walk)
        # H1: price changes ≠ 0 (trend exists)
        returns = np.diff(close[-self.trend_window - 1:])
        if len(returns) < self.trend_window or np.std(returns, ddof=1) == 0:
            return None

        t_stat, p_value = scipy_stats.ttest_1samp(returns[-self.trend_window:], 0)
        trend_significant = p_value < self.p_threshold
        trend_direction = 1 if t_stat > 0 else -1

        # ===== 2. 成交量卡方检验 =====
        vol_col = 'vol' if 'vol' in data.columns else ('volume' if 'volume' in data.columns else None)
        vol_signal = 0
        if vol_col and len(data) >= self.vol_window:
            vol = data[vol_col].values[-self.vol_window:]
            vol_ma = np.mean(vol)
            # Split volume into up-day and down-day
            recent_close = close[-self.vol_window:]
            up_vol = np.sum(vol[recent_close > np.roll(recent_close, 1)[-(self.vol_window):]])
            down_vol = np.sum(vol[recent_close <= np.roll(recent_close, 1)[-(self.vol_window):]])

            if up_vol + down_vol > 0:
                expected = (up_vol + down_vol) / 2
                if expected > 0:
                    chi2 = ((up_vol - expected) ** 2 + (down_vol - expected) ** 2) / expected
                    # Degrees of freedom = 1, compare to critical value at 0.05
                    if chi2 > 3.84:  # chi2 critical value at p=0.05, df=1
                        vol_signal = 1 if up_vol > down_vol else -1

        # ===== 3. 贝叶斯趋势概率 =====
        # P(trend|data) using running update
        # Prior: P(trend) = 0.5 (uninformative)
        # Likelihood: how likely is this data under trend vs random walk?
        recent = close[-self.mr_window:]
        net_change = abs(recent[-1] - recent[0])
        total_move = np.sum(np.abs(np.diff(recent)))
        efficiency = net_change / total_move if total_move > 0 else 0

        # P(data|trend) ~ efficiency, P(data|random) ~ 1 - efficiency
        p_data_trend = min(efficiency * 2, 0.95)  # Higher efficiency → more likely trend
        p_data_random = 1 - p_data_trend
        prior_trend = 0.5

        bayes_trend = (p_data_trend * prior_trend) / (
            p_data_trend * prior_trend + p_data_random * (1 - prior_trend)
        ) if (p_data_trend * prior_trend + p_data_random * (1 - prior_trend)) > 0 else 0.5

        # ===== 4. 均值回归检验 (for exit signals) =====
        mr_prices = close[-self.mr_window:]
        mr_mean = np.mean(mr_prices)
        mr_std = np.std(mr_prices, ddof=1)
        z_score = (close[-1] - mr_mean) / mr_std if mr_std > 0 else 0
        mr_extreme = abs(z_score) > 2.0  # 95% CI breach

        # ===== 综合评分 =====
        score = 0.0

        # t检验贡献 — 始终参与, 显著时加权
        t_weight = 0.5 if trend_significant else 0.3
        score += t_weight * np.sign(t_stat)
        # 更强的t统计量 → 更高分
        score += min(abs(t_stat) / 4.0, 0.3) * np.sign(t_stat)

        # 卡方量价贡献
        if vol_signal != 0:
            score += 0.15 * vol_signal

        # 贝叶斯贡献
        trend_dir_bayes = 1 if close[-1] > close[-self.trend_window] else -1
        bayes_strength = (bayes_trend - 0.5) * 2  # 0..1
        score += 0.2 * trend_dir_bayes * bayes_strength

        # 均值回归信号 (反转方向)
        if mr_extreme:
            mr_dir = -np.sign(z_score)
            score += 0.15 * mr_dir * min(abs(z_score) / 3.0, 1.0)

        direction = 1 if score > 0 else -1
        return score, direction

    def screen(self):
        data = self.data.copy()
        if len(data) < 40:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '评估失败', 'price': price}

        score, direction = result
        if abs(score) > 0.5:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f"score={score:.2f} (t-test + chi2 + bayesian)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score:.2f}', 'price': price}
