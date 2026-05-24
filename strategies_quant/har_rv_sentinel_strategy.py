"""
HAR-RV哨兵策略 (Heterogeneous Autoregressive Realized Volatility Sentinel)
==========================================================================
基于HAR-RV模型的波动率预测系统，用于判断波动率扩张/收缩，
结合EMA趋势方向进行趋势跟踪交易。

来源: TradingView batch_3 Innovation 5/5 — HAR-RV Sentinel

核心数学模型:
  HAR-RV (Corsi 2009): 将已实现波动率分解为日/周/月三个时间尺度
    RV_t+1 = β₀ + β₁ × RV_daily + β₂ × RV_weekly + β₃ × RV_monthly

  其中:
    RV_daily   = Σ(ln(C_t / C_{t-1}))²               — 1日已实现方差
    RV_weekly  = (1/5) Σ_{i=0}^{4} RV_daily_{t-i}    — 周均已实现方差
    RV_monthly = (1/22) Σ_{i=0}^{21} RV_daily_{t-i}  — 月均已实现方差

  通过滚动60天窗口的OLS回归估计β系数，预测下一期RV。

信号逻辑:
  predicted_RV > current_RV × 1.5 → 波动率扩张预期 → 减仓(退出多头)
  predicted_RV < current_RV × 0.5 → 波动率收缩 → 进入趋势跟踪
  趋势方向由EMA(20)确定

风险管理:
  分数Kelly仓位: f* = 0.25 × (p×b - q) / b
  ATR追踪止损

技术指标: HAR-RV, OLS, EMA, ATR, Fractional Kelly
"""
import numpy as np
import pandas as pd
from core.base_strategy import BaseStrategy


class HARRVSentinelStrategy(BaseStrategy):
    """HAR-RV哨兵策略 — 异质自回归已实现波动率预测 + 趋势跟踪"""

    strategy_description = "HAR-RV哨兵: 已实现波动率分解预测 + EMA趋势跟踪 + 分数Kelly仓位"
    strategy_category = "volatility"
    strategy_params_schema = {
        "rv_daily_len": {"type": "int", "default": 1, "label": "日RV周期"},
        "rv_weekly_len": {"type": "int", "default": 5, "label": "周RV周期"},
        "rv_monthly_len": {"type": "int", "default": 22, "label": "月RV周期"},
        "ols_window": {"type": "int", "default": 60, "label": "OLS回归窗口"},
        "ema_period": {"type": "int", "default": 20, "label": "趋势EMA周期"},
        "expansion_mult": {"type": "float", "default": 1.5, "label": "波动率扩张倍数"},
        "contraction_mult": {"type": "float", "default": 0.5, "label": "波动率收缩倍数"},
        "kelly_fraction": {"type": "float", "default": 0.25, "label": "Kelly分数系数"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR周期"},
        "hold_min": {"type": "int", "default": 3, "label": "最少持仓天数"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "追踪止损ATR倍数"},
    }

    def __init__(self, data, params=None):
        super().__init__(data, params)
        self.rv_daily_len = self.params.get('rv_daily_len', 1)
        self.rv_weekly_len = self.params.get('rv_weekly_len', 5)
        self.rv_monthly_len = self.params.get('rv_monthly_len', 22)
        self.ols_window = self.params.get('ols_window', 60)
        self.ema_period = self.params.get('ema_period', 20)
        self.expansion_mult = self.params.get('expansion_mult', 1.5)
        self.contraction_mult = self.params.get('contraction_mult', 0.5)
        self.kelly_fraction = self.params.get('kelly_fraction', 0.25)
        self.atr_period = self.params.get('atr_period', 14)
        self.hold_min = self.params.get('hold_min', 3)
        self.trail_atr_mult = self.params.get('trail_atr_mult', 2.5)

    def get_default_params(self):
        return {
            'rv_daily_len': 1, 'rv_weekly_len': 5, 'rv_monthly_len': 22,
            'ols_window': 60, 'ema_period': 20,
            'expansion_mult': 1.5, 'contraction_mult': 0.5,
            'kelly_fraction': 0.25, 'atr_period': 14,
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
        # Track recent trade outcomes for Kelly sizing
        trade_results = []

        for current_time in unique_times:
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            if current_holding is None:
                best_score = 0
                best_sym = None
                best_dir = 0
                best_kelly = 0.0

                for _, bar in current_bars.iterrows():
                    sym = bar['symbol']
                    hist = data[(data['symbol'] == sym) & (data.index < current_time)]
                    result = self._evaluate(hist, trade_results)
                    if result is None:
                        continue
                    score, direction, _, kelly_size = result
                    if abs(score) > abs(best_score):
                        best_score = score
                        best_sym = sym
                        best_dir = direction
                        best_kelly = kelly_size

                # Require score >= 3 for entry
                if best_sym and abs(best_score) >= 3:
                    entry_price = float(current_bars[current_bars['symbol'] == best_sym].iloc[0]['close'])
                    if best_dir == 1:
                        self._record_signal(current_time, 'buy', best_sym,
                                            price=entry_price, kelly_size=best_kelly)
                        position_dir = 1
                    else:
                        self._record_signal(current_time, 'sell', best_sym,
                                            price=entry_price, kelly_size=best_kelly)
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

                    # Max holding period cap to avoid stale positions
                    if days_held >= 60:
                        should_exit = True

                    # Check for opposing signal from HAR-RV model
                    if not should_exit:
                        result = self._evaluate(hist, trade_results)
                        if result is not None:
                            score, direction, _, _ = result
                            # Opposing signal: exit if strong enough
                            if position_dir == 1 and direction == -1 and score < -3:
                                should_exit = True
                            elif position_dir == -1 and direction == 1 and score > 3:
                                should_exit = True

                if should_exit:
                    # Record exit and track trade result for Kelly
                    entry_price = None
                    for s in reversed(self.signals):
                        if s['symbol'] == current_holding:
                            entry_price = s['price']
                            break
                    if entry_price is not None and entry_price > 0:
                        if position_dir == 1:
                            pnl = (current_price - entry_price) / entry_price
                        else:
                            pnl = (entry_price - current_price) / entry_price
                        trade_results.append(pnl)
                        # Keep only last 50 trades for Kelly estimation
                        if len(trade_results) > 50:
                            trade_results.pop(0)

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

        print(f"HARRVSentinel: 生成 {len(self.signals)} 个信号")
        return self.signals

    def _compute_rv_series(self, close):
        """Compute daily realized volatility series.

        RV_daily(t) = (ln(C_t / C_{t-1}))^2
        This is the squared log-return, a standard proxy for daily variance.
        """
        n = len(close)
        if n < 2:
            return np.array([])
        log_returns = np.log(close[1:] / close[:-1])
        rv_daily = log_returns ** 2
        return rv_daily

    def _compute_har_features(self, rv_daily, t):
        """Compute HAR-RV regressors at index t.

        RV_daily(t)   = rv_daily[t]                           — 1-day RV
        RV_weekly(t)  = mean(rv_daily[t-4 : t+1])            — 5-day RV
        RV_monthly(t) = mean(rv_daily[t-21 : t+1])           — 22-day RV

        WHY: Corsi (2009) showed that volatility has heterogeneous autocorrelation
        across time scales. Daily, weekly, and monthly RV capture short/medium/long
        memory components of the volatility process.
        """
        if t < self.rv_monthly_len - 1:
            return None

        rv_d = rv_daily[t]
        start_w = t - self.rv_weekly_len + 1
        start_m = t - self.rv_monthly_len + 1

        if start_w < 0 or start_m < 0:
            return None

        rv_w = np.mean(rv_daily[start_w:t + 1])
        rv_m = np.mean(rv_daily[start_m:t + 1])

        return rv_d, rv_w, rv_m

    def _ols_har(self, rv_daily, end_idx):
        """Estimate HAR-RV model via OLS on a rolling window.

        Model: RV_{t+1} = β₀ + β₁ × RV_daily(t) + β₂ × RV_weekly(t) + β₃ × RV_monthly(t)

        Uses ordinary least squares on the last `ols_window` observations.
        WHY: OLS provides unbiased estimates of the beta coefficients under the
        assumption that the HAR-RV model is correctly specified. The rolling window
        adapts to changing market regimes.
        """
        window = self.ols_window
        if end_idx < window + self.rv_monthly_len:
            return None

        # Build feature matrix over rolling window
        Y = []
        X = []
        for t in range(end_idx - window, end_idx):
            features = self._compute_har_features(rv_daily, t)
            if features is None:
                continue
            rv_d, rv_w, rv_m = features
            # Target: next day's RV
            if t + 1 >= len(rv_daily):
                continue
            rv_next = rv_daily[t + 1]
            Y.append(rv_next)
            X.append([1.0, rv_d, rv_w, rv_m])  # [intercept, daily, weekly, monthly]

        if len(Y) < 20:  # Need sufficient data for regression
            return None

        Y = np.array(Y)
        X = np.array(X)

        # OLS: β = (X'X)^(-1) X'Y
        try:
            XtX = X.T @ X
            XtY = X.T @ Y
            beta = np.linalg.solve(XtX, XtY)
        except np.linalg.LinAlgError:
            return None

        return beta

    def _predict_rv(self, rv_daily, current_idx):
        """Predict next-period RV using HAR-RV OLS coefficients.

        Returns (predicted_rv, current_rv) or None if insufficient data.
        """
        beta = self._ols_har(rv_daily, current_idx)
        if beta is None:
            return None

        features = self._compute_har_features(rv_daily, current_idx)
        if features is None:
            return None

        rv_d, rv_w, rv_m = features
        # Predicted RV: β₀ + β₁×RV_daily + β₂×RV_weekly + β₃×RV_monthly
        predicted_rv = beta[0] + beta[1] * rv_d + beta[2] * rv_w + beta[3] * rv_m
        current_rv = rv_d  # Current realized volatility = today's RV

        return predicted_rv, current_rv

    def _calc_kelly_size(self, trade_results):
        """Fractional Kelly position sizing.

        f* = kelly_fraction × (p×b - q) / b
        where:
          p = estimated win rate from recent trades
          q = 1 - p
          b = average win / average loss (reward-risk ratio)

        WHY: Kelly criterion maximizes long-term geometric growth rate.
        Using a fraction (0.25) reduces variance and avoids overbetting.
        """
        if len(trade_results) < 5:
            return 0.5  # Default half position when insufficient history

        wins = [r for r in trade_results if r > 0]
        losses = [r for r in trade_results if r <= 0]

        p = len(wins) / len(trade_results) if trade_results else 0.5
        q = 1.0 - p

        avg_win = np.mean(wins) if wins else 0.01
        avg_loss = abs(np.mean(losses)) if losses else 0.01
        b = avg_win / avg_loss if avg_loss > 0 else 1.0

        # f* = fraction × (p×b - q) / b
        kelly = self.kelly_fraction * (p * b - q) / b

        # Clamp to [0.1, 1.0] to ensure reasonable sizing
        return max(0.1, min(1.0, kelly))

    def _evaluate(self, data, trade_results=None):
        """Evaluate HAR-RV signal for the given data.

        Returns (score, direction, atr, kelly_size) or None.
        """
        min_data = max(self.ols_window + self.rv_monthly_len + 10, 100)
        if len(data) < min_data:
            return None

        close = data['close'].values
        n = len(close)

        # Compute RV series
        rv_daily = self._compute_rv_series(close)
        if len(rv_daily) < self.ols_window + self.rv_monthly_len:
            return None

        # Predict next-period RV
        result = self._predict_rv(rv_daily, len(rv_daily) - 1)
        if result is None:
            return None
        predicted_rv, current_rv = result

        # EMA for trend direction
        ema = self._calc_ema(close, self.ema_period)
        if ema is None:
            return None

        atr = self._calc_atr(data)
        if atr <= 0:
            return None

        score = 0

        # --- Volatility regime signals ---
        # WHY: When volatility is predicted to expand, trend-following strategies
        # tend to underperform due to whipsaw. When volatility contracts, trends
        # are cleaner and more profitable.
        if current_rv > 0:
            rv_ratio = predicted_rv / current_rv

            if rv_ratio > self.expansion_mult:
                # Predicted RV >> current → volatility expansion expected
                # Reduce exposure: sell if long, consider short
                score -= 5
            elif rv_ratio < self.contraction_mult:
                # Predicted RV << current → volatility contraction
                # Good environment for trend following
                score += 3
            elif rv_ratio < 0.8:
                # Moderate contraction bias
                score += 1
            elif rv_ratio > 1.2:
                # Moderate expansion bias
                score -= 1

        # --- EMA trend direction ---
        # WHY: We only take trend-following trades in the direction of the EMA.
        # This filters out counter-trend entries in trending markets.
        current_price = close[-1]
        if current_price > ema:
            score += 2  # Bullish trend
        else:
            score -= 2  # Bearish trend

        # --- Volatility level context ---
        # WHY: Low absolute volatility environments are better for breakout/trend entries.
        rv_percentile = self._rv_percentile(rv_daily, len(rv_daily) - 1)
        if rv_percentile < 0.3:
            score += 1  # Low vol regime → potential breakout coming
        elif rv_percentile > 0.7:
            score -= 1  # High vol regime → caution

        # Kelly position sizing
        kelly_size = 0.5
        if trade_results is not None:
            kelly_size = self._calc_kelly_size(trade_results)

        direction = 1 if score > 0 else -1
        return score, direction, atr, kelly_size

    def _rv_percentile(self, rv_daily, idx, lookback=60):
        """Compute percentile rank of current RV within recent history."""
        if idx < lookback:
            return 0.5
        recent = rv_daily[idx - lookback:idx + 1]
        current = rv_daily[idx]
        return np.mean(recent < current)

    def _calc_ema(self, close, period):
        """Calculate Exponential Moving Average."""
        if len(close) < period:
            return None
        multiplier = 2.0 / (period + 1)
        ema = close[0]
        for i in range(1, len(close)):
            ema = (close[i] - ema) * multiplier + ema
        return ema

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
        """Real-time screening based on latest bar HAR-RV signal."""
        data = self.data.copy()
        min_data = max(self.ols_window + self.rv_monthly_len + 10, 100)
        if len(data) < min_data:
            return {'action': 'hold', 'reason': '数据不足(需100+)', 'price': float(data['close'].iloc[-1])}

        result = self._evaluate(data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': 'HAR-RV评估失败', 'price': price}

        score, direction, _, kelly_size = result
        if abs(score) >= 3:
            action = 'buy' if direction == 1 else 'sell'
            return {
                'action': action,
                'reason': f"score={score} kelly={kelly_size:.2f} (har_rv_sentinel)",
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
