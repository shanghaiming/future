"""
Epanechnikov核汇聚策略 (Epanechnikov Kernel Confluence Strategy)
================================================================
基于Epanechnikov核函数的多因子汇聚评分策略。

来源: TradingView "Epanechnikov Kernel Confluence" 多因子核密度加权策略

核心逻辑:
  1. Epanechnikov核函数: K(u) = 0.75*(1-u^2) for |u|<=1, else 0
     对5个条件分别评分0~1, 用核函数平滑映射
  2. 5个因子:
     - 价格vs EMA200距离(归一化)
     - RSI水平(以50为中心)
     - 成交量异常(vs 20日均量)
     - ATR百分位(当前波动率水平)
     - 动量(变化率)
  3. 汇聚分数 = 5个核评分的均值
  4. 买入: 汇聚分数>=0.6 且 改善中
  5. 卖出: 汇聚分数<=0.4 且 恶化中
  6. ATR追踪止损

数学原理:
  Epanechnikov核: K(u) = 0.75*(1-u^2) for |u|<=1
  是非参数核密度估计中最优的核函数(MSE意义下)。
  相比高斯核, Epanechnikov核具有紧支撑(|u|>1时为0),
  这意味着远离中心的异常值被完全忽略, 不会干扰评分。

  5因子选择理由:
  - EMA200距离: 长期趋势方向的基本面
  - RSI: 动量极端程度的振荡器
  - 成交量: 资金参与度的确认信号
  - ATR百分位: 波动率环境适配
  - 动量ROC: 价格变化速度的直接度量

  汇聚思维: 5个独立因子同时看多/看空时, 信号可靠性远高于
  单一指标。核函数将每个因子平滑映射到[0,1], 避免阈值截断
  的信息损失。

技术指标: Epanechnikov核, EMA, RSI, ATR, 成交量, ROC
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class EpanechnikovConfluenceStrategy(BaseStrategy):
    """Epanechnikov核汇聚策略 — 5因子核密度加权 + 汇聚评分 + ATR止损"""

    strategy_description = "Epanechnikov核: 5因子核密度汇聚评分+动态阈值+ATR止损"
    strategy_category = "multi_factor"
    strategy_params_schema = {
        "ema_period": {"type": "int", "default": 200, "label": "长期EMA周期"},
        "rsi_period": {"type": "int", "default": 14, "label": "RSI周期"},
        "vol_ma_period": {"type": "int", "default": 20, "label": "成交量均线周期"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "atr_percentile_window": {"type": "int", "default": 50, "label": "ATR百分位窗口"},
        "roc_period": {"type": "int", "default": 10, "label": "动量ROC周期"},
        "buy_threshold": {"type": "float", "default": 0.6, "label": "买入汇聚阈值"},
        "sell_threshold": {"type": "float", "default": 0.4, "label": "卖出汇聚阈值"},
        "atr_period_trail": {"type": "int", "default": 14, "label": "止损ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
        "max_hold": {"type": "int", "default": 60, "label": "最大持仓天数"},
    }

    def __init__(self, data, params=None):
        super().__init__(data, params)
        p = self.params
        self.ema_period = p.get('ema_period', 200)
        self.rsi_period = p.get('rsi_period', 14)
        self.vol_ma_period = p.get('vol_ma_period', 20)
        self.atr_period = p.get('atr_period', 14)
        self.atr_percentile_window = p.get('atr_percentile_window', 50)
        self.roc_period = p.get('roc_period', 10)
        self.buy_threshold = p.get('buy_threshold', 0.6)
        self.sell_threshold = p.get('sell_threshold', 0.4)
        self.atr_period_trail = p.get('atr_period_trail', 14)
        self.hold_min = p.get('hold_min', 3)
        self.trail_atr_mult = p.get('trail_atr_mult', 2.5)
        self.max_hold = p.get('max_hold', 60)

    def get_default_params(self):
        return {
            'ema_period': 200, 'rsi_period': 14,
            'vol_ma_period': 20, 'atr_period': 14,
            'atr_percentile_window': 50, 'roc_period': 10,
            'buy_threshold': 0.6, 'sell_threshold': 0.4,
            'atr_period_trail': 14, 'hold_min': 3,
            'trail_atr_mult': 2.5, 'max_hold': 60,
        }

    # ================================================================
    # Epanechnikov核函数
    # ================================================================

    @staticmethod
    def _epanechnikov(u):
        """
        Epanechnikov核函数: K(u) = 0.75*(1-u^2) for |u|<=1, else 0

        这是非参数统计中MSE最优的核函数。
        紧支撑特性: |u|>1时严格为0, 天然过滤极端异常值。

        Args:
            u: 标准化距离(已除以带宽)
        Returns:
            核权重, 范围[0, 0.75], 最大值在u=0处
        """
        u_clipped = np.clip(u, -1.0, 1.0)
        return np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u_clipped ** 2), 0.0)

    # ================================================================
    # 5个因子评分
    # ================================================================

    def _score_ema_distance(self, close, ema_val):
        """
        因子1: 价格vs EMA200距离

        价格在EMA200之上且距离适中 → 看多
        用Epanechnikov核将距离映射为0~1评分

        为什么用EMA200: 200日均线是机构广泛使用的牛熊分界线,
        价格相对其位置反映长期趋势方向。
        """
        if np.isnan(ema_val) or ema_val < 1e-10:
            return 0.5
        # 标准化距离: (close-ema)/ema, 典型范围[-0.2, 0.2]
        dist = (close - ema_val) / ema_val
        # 映射: dist=0.05(价格在均线上方5%) → 最高分
        # 用u = 1 - dist/0.15, 使得dist=0.15时u=0(满核权重)
        u = 1.0 - dist / 0.15
        kernel_val = float(self._epanechnikov(u))
        # 归一化到0~1 (Epanechnikov最大值0.75)
        return min(1.0, kernel_val / 0.75) if kernel_val > 0 else 0.0

    def _score_rsi(self, rsi_val):
        """
        因子2: RSI水平

        RSI=70~80区间 → 强势看多(核评分高)
        RSI=20~30区间 → 强势看空(核评分低)
        RSI=50 → 中性

        为什么用RSI: 相对强弱指数衡量超买超卖,
        在趋势中RSI方向确认动量强度。
        """
        if np.isnan(rsi_val):
            return 0.5
        # 标准化: 将RSI映射到[-1, 1], 50为中心
        u = (rsi_val - 50.0) / 50.0  # RSI=0→u=-1, RSI=100→u=1
        # u>0 看多, 核值高; u<0 看空, 核值低
        # 映射核值到0~1: 最大值在u=1(RSI=100), 最小值在u=-1(RSI=0)
        return (u + 1.0) / 2.0

    def _score_volume(self, current_vol, vol_ma):
        """
        因子3: 成交量异常

        当前量 >> 均量 → 有资金推动, 信号可靠
        当前量 << 均量 → 低流动性, 信号不可靠

        为什么用成交量: 成交量是价格变动的燃料,
        放量确认价格方向, 缩量暗示信号不可靠。
        """
        if vol_ma < 1e-10:
            return 0.5
        vol_ratio = current_vol / vol_ma
        # vol_ratio=1.5~2.5 → 最佳区间
        # 用核函数: u = 1 - (ratio-1.5)/1.5
        # ratio=1.5 → u=1(满权重), ratio=3.0 → u=0, ratio<1.5 → 衰减
        u = 1.0 - (vol_ratio - 1.5) / 1.5
        kernel_val = float(self._epanechnikov(u))
        if kernel_val > 0:
            return min(1.0, kernel_val / 0.75)
        else:
            # 低量也不完全是坏事, 给一个基础分
            return max(0.0, vol_ratio / 2.0)

    def _score_atr_percentile(self, data):
        """
        因子4: ATR百分位

        当前ATR处于历史百分位:
        - 中等波动率(40~70百分位) → 最佳交易环境, 评分高
        - 极低波动率 → 市场沉寂, 评分低
        - 极高波动率 → 风险过大, 评分降低

        为什么用ATR百分位: 波动率过高时止损容易被触发,
        过低时利润空间不足, 中等波动率是最佳交易区间。
        """
        if len(data) < self.atr_period + 1:
            return 0.5

        atr_series = self._calc_atr_series(data)
        if len(atr_series) < self.atr_percentile_window:
            return 0.5

        current_atr = atr_series[-1]
        window = atr_series[-self.atr_percentile_window:]
        percentile = float(np.mean(window <= current_atr))

        # 最佳区间: 40~70百分位
        # 用核函数映射: u = 1 - (pct-0.55)/0.3
        u = 1.0 - (percentile - 0.55) / 0.3
        kernel_val = float(self._epanechnikov(u))
        if kernel_val > 0:
            return min(1.0, kernel_val / 0.75)
        # 极端情况给低分但不是0
        return max(0.1, 0.3 if percentile > 0.2 else 0.1)

    def _score_momentum(self, close, roc_val):
        """
        因子5: 动量(变化率ROC)

        正ROC且适中(3%~15%) → 健康上升趋势
        负ROC且适中 → 健康下降趋势
        ROC=0 → 无方向

        为什么用ROC: 价格变化率直接度量趋势强度,
        过高的ROC可能是衰竭前的加速, 过低则无动能。
        """
        if np.isnan(roc_val):
            return 0.5
        # 将ROC映射为看多/看空评分
        # ROC>0 → 看多, ROC<0 → 看空
        # 使用sigmoid-like映射
        raw = 1.0 / (1.0 + np.exp(-roc_val * 20))  # sigmoid of ROC
        return float(raw)

    # ================================================================
    # 辅助计算
    # ================================================================

    def _calc_ema_val(self, prices, period):
        """计算最新EMA值"""
        if len(prices) < period:
            return np.nan
        multiplier = 2.0 / (period + 1)
        ema = float(prices[-period])
        for i in range(-period + 1, 0):
            ema = (prices[i] - ema) * multiplier + ema
        return ema

    def _calc_rsi_val(self, data):
        """计算最新RSI"""
        if len(data) < self.rsi_period + 1:
            return np.nan
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

    def _calc_atr_series(self, data):
        """计算ATR序列"""
        if len(data) < self.atr_period + 1:
            return np.array([])
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        n = len(high)
        tr = np.zeros(n - 1)
        for i in range(1, n):
            tr[i - 1] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )
        # 简单移动平均ATR
        atr = np.zeros(len(tr) - self.atr_period + 1)
        for i in range(len(atr)):
            atr[i] = np.mean(tr[i:i + self.atr_period])
        return atr

    def _calc_atr(self, data):
        """计算最新ATR值"""
        if len(data) < self.atr_period_trail + 1:
            return 0.0
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        n = len(high)
        tr_list = []
        for i in range(max(1, n - self.atr_period_trail), n):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )
            tr_list.append(tr)
        return float(np.mean(tr_list)) if tr_list else 0.0

    # ================================================================
    # 核心评估
    # ================================================================

    def _evaluate(self, data, prev_data=None):
        """
        5因子Epanechnikov核汇聚评估。

        计算当前汇聚分数和(如果有前一期数据)变化趋势。

        Returns:
            (score, direction):
                score = 汇聚分数*5(放大到0~5便于统一阈值)
                direction: 1=做多, -1=做空
            None if 数据不足
        """
        min_len = max(self.ema_period, self.vol_ma_period, self.rsi_period,
                      self.atr_percentile_window, self.roc_period) + 10
        if len(data) < min_len:
            return None

        close = data['close'].values.astype(float)
        if 'volume' not in data.columns:
            return None

        volume = data['volume'].values.astype(float)

        # ===== 计算5个因子评分 =====

        # 因子1: 价格 vs EMA200
        ema_val = self._calc_ema_val(close, self.ema_period)
        score_ema = self._score_ema_distance(close[-1], ema_val)

        # 因子2: RSI水平
        rsi_val = self._calc_rsi_val(data)
        score_rsi = self._score_rsi(rsi_val)

        # 因子3: 成交量异常
        vol_ma = np.mean(volume[-self.vol_ma_period:]) if len(volume) >= self.vol_ma_period else volume[-1]
        score_vol = self._score_volume(volume[-1], vol_ma)

        # 因子4: ATR百分位
        score_atr = self._score_atr_percentile(data)

        # 因子5: 动量ROC
        roc_val = (close[-1] - close[-self.roc_period]) / (close[-self.roc_period] + 1e-10) if len(close) > self.roc_period else 0.0
        score_mom = self._score_momentum(close[-1], roc_val)

        # ===== 汇聚分数 =====
        confluence = (score_ema + score_rsi + score_vol + score_atr + score_mom) / 5.0

        # ===== 改善/恶化判断 =====
        improving = True
        if prev_data is not None and len(prev_data) >= min_len:
            prev_result = self._evaluate(prev_data)
            if prev_result is not None:
                prev_score_raw = abs(prev_result[0]) / 5.0
                # 当前分数与前一期分数比较
                current_score_for_direction = confluence if confluence > 0.5 else (1.0 - confluence)
                improving = current_score_for_direction > prev_score_raw * 0.95

        # ===== 决策 =====
        # 买入: 汇聚分数 >= buy_threshold 且 改善中
        if confluence >= self.buy_threshold and improving:
            scaled_score = confluence * 5.0
            direction = 1
            return (min(scaled_score, 5.0), direction)

        # 卖出: 汇聚分数 <= sell_threshold
        if confluence <= self.sell_threshold:
            # 对于卖出, 反转评分: 越低越强
            scaled_score = (1.0 - confluence) * 5.0
            direction = -1
            return (min(scaled_score, 5.0), direction)

        return (0, 0)

    # ================================================================
    # 信号生成
    # ================================================================

    def generate_signals(self) -> list:
        """
        生成交易信号。

        遍历时间线, 在无持仓时计算5因子汇聚分数寻找入场;
        在有持仓时用ATR追踪止损或最大持仓天数退出。
        支持做多(汇聚分数>=0.6且改善)和做空(汇聚分数<=0.4且恶化)。
        """
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

            # --- 无持仓: 寻找入场信号 ---
            if current_holding is None:
                best_score = 0
                best_sym = None
                best_dir = 0

                for _, bar in current_bars.iterrows():
                    sym = bar['symbol']
                    hist = data[(data['symbol'] == sym) & (data.index <= current_time)]
                    result = self._evaluate(hist)
                    if result is None:
                        continue
                    score, direction = result
                    if abs(score) > abs(best_score):
                        best_score = score
                        best_sym = sym
                        best_dir = direction

                # 汇聚分数*5 >= 3 → 原始汇聚分数 >= 0.6 (买入) 或 <= 0.4 (卖出)
                if best_sym and abs(best_score) >= 3.0:
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

            # --- 有持仓: 检查退出 ---
            else:
                days_held = len([t for t in unique_times if buy_time < t <= current_time])
                bar_data = current_bars[current_bars['symbol'] == current_holding]
                if len(bar_data) == 0:
                    continue
                current_price = float(bar_data.iloc[0]['close'])

                # 更新最高/最低水位
                if position_dir == 1:
                    high_water = max(high_water, current_price) if high_water > 0 else current_price
                else:
                    low_water = min(low_water, current_price) if low_water < float('inf') else current_price

                if days_held >= self.hold_min:
                    hist = data[(data['symbol'] == current_holding) & (data.index <= current_time)]
                    atr_val = self._calc_atr(hist)
                    should_exit = False

                    # ATR trailing stop
                    if atr_val > 0:
                        if position_dir == 1 and high_water > 0:
                            if current_price < high_water - self.trail_atr_mult * atr_val:
                                should_exit = True
                        elif position_dir == -1 and low_water < float('inf'):
                            if current_price > low_water + self.trail_atr_mult * atr_val:
                                should_exit = True

                    # Max hold days
                    if days_held >= self.max_hold:
                        should_exit = True

                    # Signal-based exit: 反向信号
                    if not should_exit:
                        result = self._evaluate(hist)
                        if result is not None:
                            score, direction = result
                            if position_dir == 1 and direction == -1 and score >= 3.0:
                                should_exit = True
                            elif position_dir == -1 and direction == 1 and score >= 3.0:
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

        print(f"EpanechnikovConfluence: 生成 {len(self.signals)} 个信号")
        return self.signals

    def screen(self):
        """
        实时选股: 计算当前5因子Epanechnikov核汇聚分数。

        Returns:
            Dict with keys: action, reason, price
        """
        data = self.data.copy()
        min_len = max(self.ema_period, self.vol_ma_period, self.rsi_period,
                      self.atr_percentile_window, self.roc_period) + 10
        if len(data) < min_len:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': '评估失败(缺volume?)', 'price': price}

        score, direction = result
        confluence = score / 5.0  # 还原为0~1

        if abs(score) >= 3.0:
            action = 'buy' if direction == 1 else 'sell'
            return {
                'action': action,
                'reason': f'汇聚={confluence:.2f} score={score:.1f}',
                'price': price,
            }
        return {'action': 'hold', 'reason': f'汇聚={confluence:.2f} score={score:.1f}', 'price': price}
