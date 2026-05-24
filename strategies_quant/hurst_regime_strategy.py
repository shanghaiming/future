"""
Hurst/KFD/Entropy 市场状态策略 (Hurst Exponent Regime Strategy)
==============================================================
基于Hurst指数、Katz分形维数、Shannon熵三维度市场状态分类策略。

来源: TradingView "Hurst Exponent + Fractal Dimension + Entropy" 组合策略

核心逻辑:
  1. Hurst指数(H): R/S分析法, H>0.5=趋势持续, H<0.5=均值回归
  2. Katz分形维数(KFD): 价格路径复杂度, KFD高=震荡, KFD低=趋势
  3. Shannon熵: 收益率分布的信息不确定性, 高熵=混乱, 低熵=有序
  4. 三维状态分类:
     - H>0.6 AND 低熵 → 趋势状态 → EMA交叉跟随趋势
     - H<0.4 AND 高KFD → 震荡状态 → RSI极端均值回归
     - 其他 → 中性(不交易)
  5. ATR追踪止损

数学原理:
  Hurst指数:
    R/S分析法将累计离差序列的极差(R)与标准差(S)建立幂律关系:
    R/S = c * n^H
    H=0.5 随机游走; H>0.5 趋势持续(正自相关); H<0.5 均值回归(负自相关)

  Katz分形维数:
    KFD = log(n) / (log(n) + log(d/L))
    n=步数, d=起点到终点直线距离, L=路径总长度
    KFD接近1=复杂曲折路径(震荡); KFD接近0=平滑路径(趋势)

  Shannon熵:
    H(X) = -sum(p_i * log2(p_i))
    将收益率分箱统计频率, 熵越高说明分布越均匀=无序混乱市场
    熵低说明收益集中在少数区间=有方向性的有序市场

  三维组合:
    趋势状态需要 Hurst>0.6(持续) AND 低熵(有序) 双重确认,
    避免在高噪声环境中追趋势;
    震荡状态需要 Hurst<0.4(回归) AND 高KFD(曲折) 双重确认,
    避免在真正趋势中做均值回归。

技术指标: Hurst Exponent, Katz FD, Shannon Entropy, EMA, RSI, ATR
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class HurstRegimeStrategy(BaseStrategy):
    """Hurst/KFD/Entropy市场状态策略 — 三维状态分类 + 自适应交易"""

    strategy_description = "Hurst/KFD/Entropy: 三维市场状态分类+趋势跟随/均值回归+ATR止损"
    strategy_category = "market_state"
    strategy_params_schema = {
        "hurst_window": {"type": "int", "default": 100, "label": "Hurst计算窗口"},
        "kfd_window": {"type": "int", "default": 50, "label": "KFD计算窗口"},
        "entropy_bins": {"type": "int", "default": 10, "label": "熵分箱数"},
        "entropy_window": {"type": "int", "default": 50, "label": "熵计算窗口"},
        "hurst_trend_thresh": {"type": "float", "default": 0.6, "label": "Hurst趋势阈值"},
        "hurst_mr_thresh": {"type": "float", "default": 0.4, "label": "Hurst均值回归阈值"},
        "ema_fast": {"type": "int", "default": 10, "label": "快速EMA周期"},
        "ema_slow": {"type": "int", "default": 30, "label": "慢速EMA周期"},
        "rsi_period": {"type": "int", "default": 14, "label": "RSI周期"},
        "rsi_ob": {"type": "int", "default": 70, "label": "RSI超买"},
        "rsi_os": {"type": "int", "default": 30, "label": "RSI超卖"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
        "max_hold": {"type": "int", "default": 60, "label": "最大持仓天数"},
    }

    def __init__(self, data, params=None):
        super().__init__(data, params)
        p = self.params
        self.hurst_window = p.get('hurst_window', 100)
        self.kfd_window = p.get('kfd_window', 50)
        self.entropy_bins = p.get('entropy_bins', 10)
        self.entropy_window = p.get('entropy_window', 50)
        self.hurst_trend_thresh = p.get('hurst_trend_thresh', 0.6)
        self.hurst_mr_thresh = p.get('hurst_mr_thresh', 0.4)
        self.ema_fast = p.get('ema_fast', 10)
        self.ema_slow = p.get('ema_slow', 30)
        self.rsi_period = p.get('rsi_period', 14)
        self.rsi_ob = p.get('rsi_ob', 70)
        self.rsi_os = p.get('rsi_os', 30)
        self.atr_period = p.get('atr_period', 14)
        self.hold_min = p.get('hold_min', 3)
        self.trail_atr_mult = p.get('trail_atr_mult', 2.5)
        self.max_hold = p.get('max_hold', 60)

    def get_default_params(self):
        return {
            'hurst_window': 100, 'kfd_window': 50,
            'entropy_bins': 10, 'entropy_window': 50,
            'hurst_trend_thresh': 0.6, 'hurst_mr_thresh': 0.4,
            'ema_fast': 10, 'ema_slow': 30,
            'rsi_period': 14, 'rsi_ob': 70, 'rsi_os': 30,
            'atr_period': 14, 'hold_min': 3,
            'trail_atr_mult': 2.5, 'max_hold': 60,
        }

    def generate_signals(self) -> list:
        """
        生成交易信号。

        对每个时间点:
        1. 计算Hurst/KFD/Entropy确定市场状态
        2. 趋势状态 → EMA交叉入场
        3. 震荡状态 → RSI极端入场
        4. 中性 → 不交易
        5. ATR追踪止损退出
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

        print(f"HurstRegime: 生成 {len(self.signals)} 个信号")
        return self.signals

    # ================================================================
    # 数学计算核心: Hurst / KFD / Entropy
    # ================================================================

    def _calc_hurst(self, prices):
        """
        R/S分析法计算Hurst指数。

        对不同子区间长度n, 计算R/S统计量:
          1. 将序列分为大小为n的子区间
          2. 对每个子区间: 计算累计离差的极差R 和 标准差S
          3. 取平均R/S
          4. log(R/S) vs log(n) 的斜率 = Hurst指数

        Returns:
            float: Hurst exponent (0~1), NaN if insufficient data
        """
        if len(prices) < self.hurst_window:
            return np.nan

        series = prices[-self.hurst_window:]
        returns = np.diff(np.log(series))
        n_total = len(returns)

        # 使用多个子区间大小做回归
        ns = []
        rs_vals = []
        # 子区间大小: 从10到n_total/2, 取对数均匀分布的若干点
        min_n, max_n = 10, n_total // 2
        if min_n >= max_n:
            return np.nan

        num_splits = min(8, max_n - min_n + 1)
        split_sizes = np.unique(np.linspace(min_n, max_n, num_splits).astype(int))

        for n in split_sizes:
            num_sub = n_total // n
            if num_sub < 1:
                continue
            rs_list = []
            for i in range(num_sub):
                sub = returns[i * n:(i + 1) * n]
                mean_sub = np.mean(sub)
                # 累计离差序列
                cumdev = np.cumsum(sub - mean_sub)
                # 极差R
                r = np.max(cumdev) - np.min(cumdev)
                # 标准差S
                s = np.std(sub, ddof=1)
                if s > 0:
                    rs_list.append(r / s)
            if rs_list:
                ns.append(np.log(n))
                rs_vals.append(np.log(np.mean(rs_list)))

        if len(ns) < 2:
            return np.nan

        # 线性回归: log(R/S) = H * log(n) + log(c)
        coeffs = np.polyfit(ns, rs_vals, 1)
        hurst = coeffs[0]
        # 限制在合理范围
        return float(np.clip(hurst, 0.0, 1.0))

    def _calc_kfd(self, prices):
        """
        Katz分形维数。

        KFD = log(n) / (log(n) + log(d/L))
        n = 数据点数(步数)
        d = 起点到终点的直线距离
        L = 路径总长度(相邻点距离之和)

        KFD接近1: 路径高度曲折 = 震荡市场
        KFD接近0: 路径接近直线 = 趋势市场

        Returns:
            float: Katz Fractal Dimension
        """
        if len(prices) < self.kfd_window:
            return np.nan

        series = prices[-self.kfd_window:]
        n = len(series) - 1  # 步数

        # 路径总长度 L
        diffs = np.abs(np.diff(series))
        L = np.sum(diffs)

        # 起点到终点直线距离 d
        d = np.abs(series[-1] - series[0])

        if L < 1e-10 or d < 1e-10:
            return 1.0  # 无变化视为极端震荡

        log_n = np.log(n)
        kfd = log_n / (log_n + np.log(d / L))
        return float(kfd)

    def _calc_entropy(self, prices):
        """
        Shannon熵: 收益率分布的信息不确定性。

        将收益率分成bins个等宽区间, 统计每个区间的频率p_i,
        则 H = -sum(p_i * log2(p_i))

        高熵: 收益率均匀分布 = 混乱无序市场
        低熵: 收益率集中在少数区间 = 有方向性的有序市场

        Returns:
            float: Shannon entropy (bits)
        """
        if len(prices) < self.entropy_window:
            return np.nan

        series = prices[-self.entropy_window:]
        returns = np.diff(np.log(series))
        returns = returns[np.isfinite(returns)]

        if len(returns) < 10:
            return np.nan

        # 分箱统计频率
        counts, _ = np.histogram(returns, bins=self.entropy_bins, density=False)
        total = np.sum(counts)
        if total == 0:
            return np.nan

        probs = counts[counts > 0] / total
        entropy = -np.sum(probs * np.log2(probs))

        return float(entropy)

    def _calc_rsi(self, data):
        """计算RSI"""
        if len(data) < self.rsi_period + 1:
            return 50.0
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

    def _calc_ema(self, prices, period):
        """计算EMA"""
        if len(prices) < period:
            return np.nan
        multiplier = 2.0 / (period + 1)
        ema = float(prices[-period])
        for i in range(-period + 1, 0):
            ema = (prices[i] - ema) * multiplier + ema
        return ema

    def _calc_atr(self, data):
        """计算最新ATR值"""
        if len(data) < self.atr_period + 1:
            return 0.0
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        n = len(high)
        tr_list = []
        for i in range(max(1, n - self.atr_period), n):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )
            tr_list.append(tr)
        return float(np.mean(tr_list)) if tr_list else 0.0

    def _evaluate(self, data):
        """
        三维市场状态评估: Hurst + KFD + Entropy → 状态分类 → 交易决策。

        Returns:
            (score, direction): score>=3触发信号, 1=做多, -1=做空
            None if 数据不足或中性状态
        """
        min_len = max(self.hurst_window, self.kfd_window, self.entropy_window, self.ema_slow) + 10
        if len(data) < min_len:
            return None

        close = data['close'].values
        prices = close.astype(float)

        # ===== 1. 计算三个状态指标 =====
        hurst = self._calc_hurst(prices)
        kfd = self._calc_kfd(prices)
        entropy = self._calc_entropy(prices)

        # 如果任一指标计算失败, 视为中性
        if np.isnan(hurst) or np.isnan(kfd) or np.isnan(entropy):
            return None

        # ===== 2. 状态分类 =====
        # 中位数熵作为高低阈值 (熵的典型范围约 1.5~3.0 bits for 10 bins)
        entropy_median = np.log2(self.entropy_bins) * 0.7  # 经验值

        score = 0
        direction = 0

        # --- 趋势状态: H > 0.6 AND 低熵(有序) ---
        if hurst > self.hurst_trend_thresh and entropy < entropy_median:
            # 使用EMA交叉判断趋势方向
            ema_fast = self._calc_ema(prices, self.ema_fast)
            ema_slow = self._calc_ema(prices, self.ema_slow)

            if np.isnan(ema_fast) or np.isnan(ema_slow):
                return None

            # EMA快线上穿慢线 → 做多
            if ema_fast > ema_slow:
                score = 3
                # 额外加分: Hurst越强, 趋势越可靠
                if hurst > 0.7:
                    score += 1
                direction = 1
            # EMA快线下穿慢线 → 做空
            elif ema_fast < ema_slow:
                score = 3
                if hurst > 0.7:
                    score += 1
                direction = -1

        # --- 震荡状态: H < 0.4 AND 高KFD(曲折) ---
        elif hurst < self.hurst_mr_thresh and kfd > 0.5:
            # 使用RSI极端做均值回归
            rsi = self._calc_rsi(data)

            # RSI超卖 → 买入(预期回归均值)
            if rsi < self.rsi_os:
                score = 3
                if rsi < 20:
                    score += 1
                direction = 1
            # RSI超买 → 卖出(预期回归均值)
            elif rsi > self.rsi_ob:
                score = 3
                if rsi > 80:
                    score += 1
                direction = -1

        # --- 其他: 中性, 不交易 ---
        else:
            return (0, 0)

        return (score, direction)

    def screen(self):
        """
        实时选股: 计算Hurst/KFD/Entropy判断当前市场状态。

        Returns:
            Dict with keys: action, reason, price
        """
        data = self.data.copy()
        min_len = max(self.hurst_window, self.kfd_window, self.entropy_window, self.ema_slow) + 10
        if len(data) < min_len:
            return {'action': 'hold', 'reason': '数据不足', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        # 额外获取状态描述
        close = data['close'].values.astype(float)
        hurst = self._calc_hurst(close)
        kfd = self._calc_kfd(close)
        entropy = self._calc_entropy(close)

        state_desc = f"H={hurst:.2f} KFD={kfd:.2f} E={entropy:.2f}"

        if result is None:
            return {'action': 'hold', 'reason': f'计算失败 {state_desc}', 'price': price}

        score, direction = result
        if abs(score) >= 3:
            action = 'buy' if direction == 1 else 'sell'
            regime = "趋势" if hurst > self.hurst_trend_thresh else "震荡"
            return {
                'action': action,
                'reason': f'{regime}状态 {state_desc} score={score}',
                'price': price,
            }
        return {'action': 'hold', 'reason': f'中性 {state_desc} score={score}', 'price': price}
