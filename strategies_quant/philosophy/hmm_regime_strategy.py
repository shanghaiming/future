"""
HMM Regime Detection Strategy (Hidden Markov Model Regime Strategy)
===================================================================
3-state Gaussian HMM regime detection (bull/bear/range) using daily returns,
ATR/close ratio, and volume percentile rank as features.

Core Logic:
  1. Gaussian HMM with 3 states and Dirichlet prior encouraging self-transition
  2. Bull state -> route to trend strategy (EMA cross + structural tension direction)
  3. Bear state -> suppress all long signals, allow short
  4. Range state -> route to mean reversion (RSI extremes + Z-score)
  5. Fallback to KER-based regime classification if hmmlearn not available
  6. ATR trailing stop for exits
  7. Score = regime confidence x direction signal strength

Technical Indicators: Gaussian HMM, EMA, Structural Tension, RSI, Z-score, ATR
Category: regime
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from core.base_strategy import BaseStrategy

# Attempt hmmlearn import; fall back to KER threshold classification
_HMM_AVAILABLE = False
try:
    from hmmlearn.hmm import GaussianHMM
    _HMM_AVAILABLE = True
except ImportError:
    pass


class HMMRegimeStrategy(BaseStrategy):
    """HMM Regime Detection Strategy -- 3-state Gaussian HMM + adaptive routing"""

    strategy_description = (
        "HMM Regime: 3-state Gaussian HMM (bull/bear/range) + "
        "trend/mean-reversion adaptive routing + ATR trailing stop"
    )
    strategy_category = "regime"
    strategy_params_schema = {
        "n_states": {"type": "int", "default": 3, "label": "HMM states"},
        "hmm_train_window": {"type": "int", "default": 200, "label": "HMM training window"},
        "hmm_refit_freq": {"type": "int", "default": 20, "label": "HMM refit frequency (bars)"},
        "regime_conf_thresh": {"type": "float", "default": 0.6, "label": "Regime confidence threshold"},
        "ema_fast": {"type": "int", "default": 10, "label": "Fast EMA period"},
        "ema_slow": {"type": "int", "default": 30, "label": "Slow EMA period"},
        "rsi_period": {"type": "int", "default": 14, "label": "RSI period"},
        "rsi_ob": {"type": "int", "default": 70, "label": "RSI overbought"},
        "rsi_os": {"type": "int", "default": 30, "label": "RSI oversold"},
        "zscore_period": {"type": "int", "default": 20, "label": "Z-score period"},
        "zscore_thresh": {"type": "float", "default": 1.5, "label": "Z-score threshold"},
        "pivot_len": {"type": "int", "default": 5, "label": "Pivot lookback"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR period"},
        "hold_min": {"type": "int", "default": 3, "label": "Min holding days"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "Trail stop ATR mult"},
        "max_hold": {"type": "int", "default": 60, "label": "Max holding days"},
        "ker_trend_thresh": {"type": "float", "default": 0.3, "label": "KER trend threshold (fallback)"},
        "ker_range_thresh": {"type": "float", "default": 0.15, "label": "KER range threshold (fallback)"},
        "score_threshold": {"type": "int", "default": 4, "label": "Signal score threshold"},
    }

    def __init__(self, data: pd.DataFrame, params: dict = None):
        super().__init__(data, params)
        p = self.params
        self.n_states = p.get('n_states', 3)
        self.hmm_train_window = p.get('hmm_train_window', 200)
        self.hmm_refit_freq = p.get('hmm_refit_freq', 20)
        self.regime_conf_thresh = p.get('regime_conf_thresh', 0.6)
        self.ema_fast = p.get('ema_fast', 10)
        self.ema_slow = p.get('ema_slow', 30)
        self.rsi_period = p.get('rsi_period', 14)
        self.rsi_ob = p.get('rsi_ob', 70)
        self.rsi_os = p.get('rsi_os', 30)
        self.zscore_period = p.get('zscore_period', 20)
        self.zscore_thresh = p.get('zscore_thresh', 1.5)
        self.pivot_len = p.get('pivot_len', 5)
        self.atr_period = p.get('atr_period', 14)
        self.hold_min = p.get('hold_min', 3)
        self.trail_atr_mult = p.get('trail_atr_mult', 2.5)
        self.max_hold = p.get('max_hold', 60)
        self.ker_trend_thresh = p.get('ker_trend_thresh', 0.3)
        self.ker_range_thresh = p.get('ker_range_thresh', 0.15)
        self.score_threshold = p.get('score_threshold', 4)
        # Cached HMM model per symbol
        self._hmm_models: Dict[str, Any] = {}
        self._hmm_labels: Dict[str, Dict] = {}  # state_id -> {label, mean_ret}
        self._last_fit_idx: Dict[str, int] = {}

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'n_states': 3, 'hmm_train_window': 200, 'hmm_refit_freq': 20,
            'regime_conf_thresh': 0.6,
            'ema_fast': 10, 'ema_slow': 30,
            'rsi_period': 14, 'rsi_ob': 70, 'rsi_os': 30,
            'zscore_period': 20, 'zscore_thresh': 1.5,
            'pivot_len': 5, 'atr_period': 14,
            'hold_min': 3, 'trail_atr_mult': 2.5, 'max_hold': 60,
            'ker_trend_thresh': 0.3, 'ker_range_thresh': 0.15,
            'score_threshold': 4,
        }

    # ================================================================
    # Signal Generation
    # ================================================================

    def generate_signals(self) -> List[Dict]:
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'

        symbols = data['symbol'].unique()
        unique_times = sorted(data.index.unique())
        self.signals = []

        # Per-symbol precompute
        self._hmm_models = {}
        self._hmm_labels = {}
        self._last_fit_idx = {}

        current_holding = None
        buy_time = None
        position_dir = 0
        high_water = 0.0
        low_water = float('inf')

        for current_time in unique_times:
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            # --- No position: seek entry ---
            if current_holding is None:
                best_score = 0
                best_sym = None
                best_dir = 0

                for _, bar in current_bars.iterrows():
                    sym = bar['symbol']
                    hist = data[(data['symbol'] == sym) & (data.index <= current_time)]
                    result = self._evaluate(sym, hist)
                    if result is None:
                        continue
                    score, direction, _ = result
                    if abs(score) > abs(best_score):
                        best_score = score
                        best_sym = sym
                        best_dir = direction

                if best_sym and abs(best_score) >= self.score_threshold:
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

            # --- In position: check exits ---
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

                    # Max hold
                    if days_held >= self.max_hold:
                        should_exit = True

                    # Signal-based exit
                    if not should_exit:
                        result = self._evaluate(current_holding, hist)
                        if result is not None:
                            score, direction, _ = result
                            if position_dir == 1 and direction == -1 and score < -self.score_threshold:
                                should_exit = True
                            elif position_dir == -1 and direction == 1 and score > self.score_threshold:
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

        print(f"HMMRegime: generated {len(self.signals)} signals "
              f"(hmmlearn={'available' if _HMM_AVAILABLE else 'fallback_KER'})")
        return self.signals

    # ================================================================
    # Core Evaluation
    # ================================================================

    def _evaluate(self, symbol: str, data: pd.DataFrame):
        """
        Evaluate regime and generate scored signal.

        Returns:
            (score, direction, atr) or None if insufficient data
        """
        min_len = max(self.hmm_train_window, self.ema_slow, self.rsi_period) + 10
        if len(data) < min_len:
            return None

        close = data['close'].values.astype(float)
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        n = len(close)
        atr = self._calc_atr(data)
        if atr <= 0:
            return None

        # Step 1: Determine regime
        if _HMM_AVAILABLE:
            regime, confidence = self._classify_hmm(symbol, close, high, low, data)
        else:
            regime, confidence = self._classify_ker(close)

        # Step 2: Route to appropriate sub-strategy
        score = 0
        direction = 0

        if regime == 'bull':
            score, direction = self._bull_strategy(close, atr)
        elif regime == 'bear':
            score, direction = self._bear_strategy(close, atr)
        elif regime == 'range':
            score, direction = self._range_strategy(close, data, atr)
        else:
            return None

        # Step 3: Modulate by regime confidence
        final_score = int(score * confidence)

        return final_score, direction, atr

    # ================================================================
    # HMM Regime Classification
    # ================================================================

    def _build_features(self, close, high, low, data):
        """
        Build feature matrix: [daily_returns, ATR(14)/close, vol_percentile_rank]
        Shape: (n_samples, 3)
        """
        n = len(close)
        returns = np.diff(close) / close[:-1]
        returns = np.nan_to_num(returns, nan=0.0)

        # ATR(14) / close ratio
        atr_ratio = np.full(n - 1, 0.0)
        atr_arr = self._calc_atr_series(data)
        if len(atr_arr) >= n:
            atr_ratio = atr_arr[1:] / close[1:]
            atr_ratio = np.nan_to_num(atr_ratio, nan=0.0, posinf=0.0, neginf=0.0)

        # Volume percentile rank
        vol_rank = np.full(n - 1, 0.5)
        vol_col = None
        for vc in ['vol', 'volume', 'Volume']:
            if vc in data.columns:
                vol_col = vc
                break
        if vol_col is not None:
            vol = data[vol_col].values.astype(float)
            vol = np.nan_to_num(vol, nan=0.0)
            for i in range(1, n):
                window = vol[max(0, i - 50):i + 1]
                if len(window) > 0:
                    rank = np.sum(window <= vol[i]) / len(window)
                    vol_rank[i - 1] = rank

        features = np.column_stack([returns, atr_ratio, vol_rank])
        return features

    def _classify_hmm(self, symbol, close, high, low, data):
        """
        Classify regime using Gaussian HMM.

        Returns:
            (regime: 'bull'/'bear'/'range', confidence: float)
        """
        features = self._build_features(close, high, low, data)
        n = len(features)
        train_n = min(self.hmm_train_window, n)
        train_data = features[-train_n:]

        # Fit or reuse model
        should_fit = False
        if symbol not in self._hmm_models:
            should_fit = True
        else:
            last_idx = self._last_fit_idx.get(symbol, 0)
            if n - last_idx >= self.hmm_refit_freq:
                should_fit = True

        if should_fit:
            try:
                model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type='full',
                    n_iter=100,
                    random_state=42,
                    tol=1e-3,
                )
                # Dirichlet prior encouraging self-transition
                # Start with high self-transition probabilities
                model.startprob_prior = np.full(self.n_states, 1.0)
                model.transmat_prior = np.full(
                    (self.n_states, self.n_states), 0.1
                )
                np.fill_diagonal(
                    model.transmat_prior,
                    10.0  # Strong self-transition prior
                )
                model.fit(train_data)
                self._hmm_models[symbol] = model
                self._last_fit_idx[symbol] = n

                # Map states: identify which is bull/bear/range
                self._label_states(symbol, model, train_data)
            except Exception:
                # Fallback to KER
                return self._classify_ker(close)

        model = self._hmm_models.get(symbol)
        if model is None:
            return self._classify_ker(close)

        try:
            # Predict on the full feature set using the trained model
            # Only use the portion that matches training alignment
            recent_features = features[-train_n:]
            if len(recent_features) < 5:
                return self._classify_ker(close)

            posterior = model.predict_proba(recent_features)
            state_seq = model.predict(recent_features)
            current_state = state_seq[-1]
            confidence = float(posterior[-1, current_state])
            labels = self._hmm_labels.get(symbol, {})
            regime = labels.get(current_state, {}).get('label', 'range')
            return regime, confidence
        except Exception:
            return self._classify_ker(close)

    def _label_states(self, symbol, model, train_data):
        """
        Map HMM state indices to bull/bear/range labels based on
        the mean return of each state.
        """
        state_seq = model.predict(train_data)
        means = model.means_  # (n_states, n_features) -- first feature is returns

        labels = {}
        state_returns = []
        for s in range(self.n_states):
            mask = state_seq == s
            if np.sum(mask) > 0:
                mean_ret = float(np.mean(train_data[mask, 0]))
            else:
                mean_ret = float(means[s, 0])
            state_returns.append((s, mean_ret))

        # Sort by mean return: highest = bull, lowest = bear, middle = range
        state_returns.sort(key=lambda x: x[1], reverse=True)

        if len(state_returns) >= 3:
            labels[state_returns[0][0]] = {'label': 'bull', 'mean_ret': state_returns[0][1]}
            labels[state_returns[1][0]] = {'label': 'range', 'mean_ret': state_returns[1][1]}
            labels[state_returns[2][0]] = {'label': 'bear', 'mean_ret': state_returns[2][1]}
        elif len(state_returns) == 2:
            labels[state_returns[0][0]] = {'label': 'bull', 'mean_ret': state_returns[0][1]}
            labels[state_returns[1][0]] = {'label': 'bear', 'mean_ret': state_returns[1][1]}
        else:
            labels[state_returns[0][0]] = {'label': 'range', 'mean_ret': state_returns[0][1]}

        self._hmm_labels[symbol] = labels

    # ================================================================
    # KER Fallback Classification
    # ================================================================

    def _classify_ker(self, close):
        """
        Fallback regime classification using Kaufman Efficiency Ratio.

        Returns:
            (regime: 'bull'/'bear'/'range', confidence: float)
        """
        er = self._calc_ker(close)
        if er > self.ker_trend_thresh:
            # Determine direction from recent price change
            lookback = min(20, len(close) - 1)
            if lookback > 0:
                price_change = (close[-1] - close[-lookback]) / close[-lookback]
            else:
                price_change = 0.0
            if price_change > 0:
                regime = 'bull'
            else:
                regime = 'bear'
            confidence = min(1.0, er / 0.5)
            return regime, confidence
        elif er < self.ker_range_thresh:
            return 'range', 1.0 - er / self.ker_range_thresh
        else:
            return 'range', 0.5

    def _calc_ker(self, close):
        """Kaufman Efficiency Ratio"""
        period = min(20, len(close) - 1)
        if period < 2:
            return 0.0
        recent = close[-period:]
        net = abs(float(recent[-1] - recent[0]))
        total = float(np.sum(np.abs(np.diff(recent))))
        return net / total if total > 0 else 0.0

    # ================================================================
    # Regime-Specific Strategies
    # ================================================================

    def _bull_strategy(self, close, atr):
        """
        Bull regime: EMA cross + structural tension direction.
        Returns (score, direction).
        """
        n = len(close)
        score = 0

        # EMA cross
        ema_f = self._calc_ema_series(close, self.ema_fast)
        ema_s = self._calc_ema_series(close, self.ema_slow)

        ema_cross_up = n >= 2 and ema_f[-2] < ema_s[-2] and ema_f[-1] > ema_s[-1]
        ema_cross_down = n >= 2 and ema_f[-2] > ema_s[-2] and ema_f[-1] < ema_s[-1]
        above_ema = close[-1] > ema_s[-1]

        if ema_cross_up:
            score += 4
        elif above_ema and ema_f[-1] > ema_s[-1]:
            score += 2

        if ema_cross_down:
            score -= 2  # In bull regime, bearish cross is weakened

        # EMA slope momentum
        lookback_slope = min(5, n - 1)
        if lookback_slope > 0 and ema_s[-lookback_slope] > 0:
            slope = (ema_s[-1] - ema_s[-lookback_slope]) / ema_s[-lookback_slope]
            if slope > 0:
                score += 2
            elif slope < 0:
                score -= 1

        # Structural tension (7-point displacement) for direction confirmation
        tension = self._calc_structural_tension(close, atr)
        if tension > 0.5:
            score += 3
        elif tension > 0.2:
            score += 1
        elif tension < -0.5:
            score -= 3
        elif tension < -0.2:
            score -= 1

        direction = 1 if score > 0 else -1
        return score, direction

    def _bear_strategy(self, close, atr):
        """
        Bear regime: suppress all long signals, allow short.
        Returns (score, direction).
        """
        n = len(close)
        score = 0

        # EMA direction
        ema_f = self._calc_ema_series(close, self.ema_fast)
        ema_s = self._calc_ema_series(close, self.ema_slow)

        below_ema = close[-1] < ema_s[-1]
        ema_bearish = ema_f[-1] < ema_s[-1]

        # Only allow short signals in bear regime
        if ema_bearish and below_ema:
            score -= 5  # Strong short
        elif ema_bearish:
            score -= 3
        elif below_ema:
            score -= 2

        # Structural tension: bearish confirmation
        tension = self._calc_structural_tension(close, atr)
        if tension < -0.3:
            score -= 2
        elif tension > 0.3:
            score += 1  # Weak bullish dissent, reduce short conviction

        # Price momentum confirmation
        lookback = min(10, n - 1)
        if lookback > 0:
            momentum = (close[-1] - close[-lookback]) / close[-lookback]
            if momentum < -0.03:
                score -= 2
            elif momentum > 0.03:
                score += 1  # Counter-trend, reduce short

        direction = -1 if score < 0 else 1
        return score, direction

    def _range_strategy(self, close, data, atr):
        """
        Range regime: mean reversion using RSI extremes + Z-score.
        Returns (score, direction).
        """
        score = 0

        # RSI
        rsi = self._calc_rsi(close)
        if rsi < self.rsi_os:
            score += 4
            if rsi < 20:
                score += 2
        elif rsi < 40:
            score += 1
        elif rsi > self.rsi_ob:
            score -= 4
            if rsi > 80:
                score -= 2
        elif rsi > 60:
            score -= 1

        # Z-score
        zscore = self._calc_zscore(close)
        if zscore < -self.zscore_thresh:
            score += 3
        elif zscore > self.zscore_thresh:
            score -= 3

        # Mean reversion bonus: price at Bollinger-like extremes
        if len(close) >= 20:
            sma20 = np.mean(close[-20:])
            std20 = np.std(close[-20:], ddof=1)
            if std20 > 0:
                bb_pos = (close[-1] - sma20) / (2 * std20)
                if bb_pos < -1.0:
                    score += 2
                elif bb_pos > 1.0:
                    score -= 2

        direction = 1 if score > 0 else -1
        return score, direction

    # ================================================================
    # Helper: Structural Tension (7-point displacement)
    # ================================================================

    def _calc_structural_tension(self, close, atr):
        """
        7-point structural tension oscillator.
        Positive = bullish displacement, negative = bearish.
        """
        n = len(close)
        if n < self.pivot_len * 2 + 5:
            return 0.0

        high_arr = self.data['high'].values.astype(float) if 'high' in self.data.columns else close
        low_arr = self.data['low'].values.astype(float) if 'low' in self.data.columns else close

        # Find recent swing points
        swing_highs = []
        swing_lows = []
        for i in range(self.pivot_len, n - self.pivot_len):
            is_high = all(high_arr[i] >= high_arr[i + j]
                         for j in range(-self.pivot_len, self.pivot_len + 1) if j != 0)
            is_low = all(low_arr[i] <= low_arr[i + j]
                        for j in range(-self.pivot_len, self.pivot_len + 1) if j != 0)
            if is_high:
                swing_highs.append((i, high_arr[i]))
            if is_low:
                swing_lows.append((i, low_arr[i]))

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 0.0

        # 7 reference points
        prev_sh = swing_highs[-2][1]
        curr_sh = swing_highs[-1][1]
        prev_sl = swing_lows[-2][1]
        curr_sl = swing_lows[-1][1]
        inter_high = max(curr_sh, prev_sh)
        inter_low = min(curr_sl, prev_sl)
        inter_mid = (inter_high + inter_low) / 2.0

        ref_points = [prev_sh, curr_sh, prev_sl, curr_sl, inter_high, inter_low, inter_mid]
        current_price = close[-1]

        if atr <= 0:
            return 0.0

        displacements = [(current_price - rp) / atr for rp in ref_points]
        tension = float(np.mean(displacements))
        return tension

    # ================================================================
    # Technical Indicators
    # ================================================================

    def _calc_atr(self, data):
        """ATR calculation"""
        if len(data) < self.atr_period + 1:
            return 0.0
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
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

    def _calc_atr_series(self, data):
        """ATR as a full series aligned with data length."""
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        n = len(high)
        atr_arr = np.full(n, 0.0)
        for i in range(1, n):
            tr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
            start = max(1, i - self.atr_period + 1)
            atr_arr[i] = np.mean([
                max(high[j] - low[j], abs(high[j] - close[j - 1]), abs(low[j] - close[j - 1]))
                for j in range(start, i + 1)
            ])
        return atr_arr

    def _calc_ema_series(self, values, period):
        """EMA as numpy array."""
        values = np.asarray(values, dtype=float)
        n = len(values)
        result = np.empty(n)
        result[0] = values[0]
        k = 2.0 / (period + 1)
        for i in range(1, n):
            if i < period:
                result[i] = np.mean(values[:i + 1])
            else:
                result[i] = values[i] * k + result[i - 1] * (1 - k)
        return result

    def _calc_rsi(self, close):
        """RSI calculation."""
        period = self.rsi_period
        if len(close) < period + 1:
            return 50.0
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = float(np.mean(gains[-period:]))
        avg_loss = float(np.mean(losses[-period:]))
        if avg_loss < 1e-10:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - 100 / (1 + rs))

    def _calc_zscore(self, close):
        """Z-score of current price vs recent window."""
        if len(close) < self.zscore_period:
            return 0.0
        recent = close[-self.zscore_period:]
        std = float(np.std(recent, ddof=1))
        if std < 1e-10:
            return 0.0
        return float((close[-1] - np.mean(recent)) / std)

    def screen(self):
        """Quick screen based on latest data."""
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'

        sym = data['symbol'].iloc[0]
        min_len = max(self.hmm_train_window, self.ema_slow, self.rsi_period) + 10
        if len(data) < min_len:
            return {'action': 'hold', 'reason': 'insufficient data',
                    'price': float(data['close'].iloc[-1])}

        result = self._evaluate(sym, data)
        price = float(data['close'].iloc[-1])

        if result is None:
            return {'action': 'hold', 'reason': 'evaluation failed', 'price': price}

        score, direction, _ = result
        if abs(score) >= self.score_threshold:
            regime = 'bull/bear/range'
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f'score={score} (HMM regime)',
                'price': price,
            }
        return {'action': 'hold', 'reason': f'score={score}', 'price': price}
