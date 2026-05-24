"""
HMM Regime Detection Strategy v2 (Hidden Markov Model Regime Strategy v2)
========================================================================
3-state Gaussian HMM regime detection (bull/bear/sideways) using continuous emissions
of returns, volatility, and volume rank with enhanced filtering.

Core Improvements over v1:
  1. Gaussian HMM with continuous emissions (no rank transformation)
  2. Regime persistence probability filter - only trade when regime stays stable
  3. Viterbi decoding for optimal state sequence
  4. Emission probabilities as confidence scores
  5. Enhanced feature engineering: returns + ATR_ratio + volume_percentile + volatility
  6. Integration with structural tension for direction confirmation
  7. State transition matrix analysis for regime stability

Technical Indicators: GaussianHMM, Viterbi, Emission probabilities, ATR, Structural tension
Category: regime
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from core.base_strategy import BaseStrategy

# Attempt hmmlearn import
_HMM_AVAILABLE = False
try:
    from hmmlearn.hmm import GaussianHMM
    _HMM_AVAILABLE = True
except ImportError:
    print("Warning: hmmlearn not available, using simplified KER fallback")


class HMMRegimeV2Strategy(BaseStrategy):
    """HMM Regime Detection Strategy v2 - Enhanced with continuous emissions and persistence filtering"""

    strategy_description = (
        "HMM Regime v2: 3-state Gaussian HMM (bull/bear/sideways) with continuous "
        "emissions + Viterbi decoding + regime persistence filter + ATR trailing stop"
    )
    strategy_category = "regime"
    strategy_params_schema = {
        "n_states": {"type": "int", "default": 3, "label": "HMM states"},
        "hmm_train_window": {"type": "int", "default": 150, "label": "HMM training window"},
        "hmm_refit_freq": {"type": "int", "default": 15, "label": "HMM refit frequency"},
        "regime_conf_thresh": {"type": "float", "default": 0.7, "label": "Regime confidence threshold"},
        "persistence_window": {"type": "int", "default": 5, "label": "Regime persistence days"},
        "persistence_min_prob": {"type": "float", "default": 0.8, "label": "Min persistence probability"},
        "ema_fast": {"type": "int", "default": 12, "label": "Fast EMA period"},
        "ema_slow": {"type": "int", "default": 26, "label": "Slow EMA period"},
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
        "score_threshold": {"type": "int", "default": 5, "label": "Signal score threshold"},
        "ker_trend_thresh": {"type": "float", "default": 0.3, "label": "KER trend threshold (fallback)"},
        "ker_range_thresh": {"type": "float", "default": 0.15, "label": "KER range threshold (fallback)"},
    }

    def __init__(self, data: pd.DataFrame, params: dict = None):
        super().__init__(data, params)
        p = self.params
        self.n_states = p.get('n_states', 3)
        self.hmm_train_window = p.get('hmm_train_window', 150)
        self.hmm_refit_freq = p.get('hmm_refit_freq', 15)
        self.regime_conf_thresh = p.get('regime_conf_thresh', 0.7)
        self.persistence_window = p.get('persistence_window', 5)
        self.persistence_min_prob = p.get('persistence_min_prob', 0.8)
        self.ema_fast = p.get('ema_fast', 12)
        self.ema_slow = p.get('ema_slow', 26)
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
        self.score_threshold = p.get('score_threshold', 5)
        self.ker_trend_thresh = p.get('ker_trend_thresh', 0.3)
        self.ker_range_thresh = p.get('ker_range_thresh', 0.15)

        # Cached HMM models and metadata
        self._hmm_models: Dict[str, Any] = {}
        self._hmm_labels: Dict[str, Dict] = {}
        self._last_fit_idx: Dict[str, int] = {}
        self._last_regime: Dict[str, str] = {}
        self._regime_persistence: Dict[str, List] = {}

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'n_states': 3, 'hmm_train_window': 150, 'hmm_refit_freq': 15,
            'regime_conf_thresh': 0.7,
            'persistence_window': 5, 'persistence_min_prob': 0.8,
            'ema_fast': 12, 'ema_slow': 26,
            'rsi_period': 14, 'rsi_ob': 70, 'rsi_os': 30,
            'zscore_period': 20, 'zscore_thresh': 1.5,
            'pivot_len': 5, 'atr_period': 14,
            'hold_min': 3, 'trail_atr_mult': 2.5, 'max_hold': 60,
            'score_threshold': 5,
            'ker_trend_thresh': 0.3, 'ker_range_thresh': 0.15,
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

        # Initialize per-symbol data
        self._hmm_models = {}
        self._hmm_labels = {}
        self._last_fit_idx = {}
        self._last_regime = {}
        self._regime_persistence = {sym: [] for sym in symbols}

        current_holding = None
        buy_time = None
        position_dir = 0
        high_water = 0.0
        low_water = float('inf')

        for current_time in unique_times:
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            # No position: seek entry with persistence filter
            if current_holding is None:
                best_score = 0
                best_sym = None
                best_dir = 0

                for _, bar in current_bars.iterrows():
                    sym = bar['symbol']
                    hist = data[(data['symbol'] == sym) & (data.index <= current_time)]
                    result = self._evaluate_with_persistence(sym, hist)
                    if result is None:
                        continue
                    score, direction, _, persistence_prob = result
                    # Apply persistence filter
                    if persistence_prob < self.persistence_min_prob:
                        continue
                    if abs(score) > abs(best_score):
                        best_score = score
                        best_sym = sym
                        best_dir = direction

                if best_sym and abs(best_score) >= self.score_threshold:
                    if best_dir == 1:
                        self._record_signal(current_time, 'buy', best_sym, score=best_score)
                    else:
                        self._record_signal(current_time, 'sell', best_sym, score=best_score)
                    current_holding = best_sym
                    buy_time = current_time
                    high_water = 0.0
                    low_water = float('inf')

            # In position: check exits
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

                    # Signal-based exit with persistence check
                    if not should_exit:
                        result = self._evaluate_with_persistence(current_holding, hist)
                        if result is not None:
                            score, direction, _, persistence_prob = result
                            # Exit if regime changes and new regime has high confidence
                            if (position_dir == 1 and direction == -1 and
                                score < -self.score_threshold and persistence_prob > 0.8):
                                should_exit = True
                            elif (position_dir == -1 and direction == 1 and
                                  score > self.score_threshold and persistence_prob > 0.8):
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

        print(f"HMMRegimeV2: generated {len(self.signals)} signals "
              f"(hmmlearn={'available' if _HMM_AVAILABLE else 'fallback_KER'})")
        return self.signals

    # ================================================================
    # Core Evaluation with Persistence
    # ================================================================

    def _evaluate_with_persistence(self, symbol: str, data: pd.DataFrame):
        """
        Evaluate regime with persistence tracking.

        Returns:
            (score, direction, confidence, persistence_prob) or None
        """
        min_len = max(self.hmm_train_window, self.ema_slow, self.rsi_period) + 10
        if len(data) < min_len:
            return None

        close = data['close'].values.astype(float)
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        atr = self._calc_atr(data)
        if atr <= 0:
            return None

        # Get regime and confidence
        if _HMM_AVAILABLE:
            regime, confidence, viterbi_seq, emission_probs = self._classify_hmm_v2(symbol, close, high, low, data)
        else:
            regime, confidence = self._classify_ker(close)
            viterbi_seq = [0] * len(close)
            emission_probs = np.array([1.0])

        # Track persistence
        self._update_persistence(symbol, regime, confidence)

        # Apply persistence filter
        persistence_prob = self._get_persistence_probability(symbol, regime)

        # Route to strategy
        score = 0
        direction = 0
        if regime == 'bull':
            score, direction = self._bull_strategy(close, atr)
        elif regime == 'bear':
            score, direction = self._bear_strategy(close, atr)
        elif regime == 'sideways':
            score, direction = self._sideways_strategy(close, data, atr)

        # Combine with emission confidence
        final_score = int(score * confidence * persistence_prob)

        return final_score, direction, confidence, persistence_prob

    def _update_persistence(self, symbol: str, regime: str, confidence: float):
        """Update regime persistence tracking."""
        if symbol not in self._regime_persistence:
            self._regime_persistence[symbol] = []

        # Add current regime with confidence
        self._regime_persistence[symbol].append({
            'regime': regime,
            'confidence': confidence,
            'timestamp': pd.Timestamp.now()
        })

        # Keep only recent history
        window_days = self.persistence_window
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=window_days)
        self._regime_persistence[symbol] = [
            r for r in self._regime_persistence[symbol]
            if r['timestamp'] > cutoff
        ]

    def _get_persistence_probability(self, symbol: str, current_regime: str) -> float:
        """Calculate probability that current regime persists."""
        if symbol not in self._regime_persistence:
            return 0.5

        history = self._regime_persistence[symbol]
        if len(history) < 2:
            return 0.5

        # Count how many times same regime persisted
        same_regime_count = 0
        for i in range(1, len(history)):
            if history[i]['regime'] == history[i-1]['regime']:
                same_regime_count += 1

        # Calculate persistence probability
        persistence = same_regime_count / (len(history) - 1) if len(history) > 1 else 0.5

        # Weight by recent confidence
        recent_confidences = [h['confidence'] for h in history[-3:]]
        if recent_confidences:
            confidence_weight = np.mean(recent_confidences)
            persistence = persistence * 0.7 + confidence_weight * 0.3

        return float(persistence)

    # ================================================================
    # HMM Regime Classification (Enhanced)
    # ================================================================

    def _build_enhanced_features(self, close, high, low, data):
        """
        Build enhanced feature matrix: [daily_returns, ATR_ratio, volume_percentile, volatility]
        Shape: (n_samples, 4)
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

        # Volume percentile rank (with smoothing)
        vol_rank = np.full(n - 1, 0.5)
        vol_col = None
        for vc in ['vol', 'volume', 'Volume']:
            if vc in data.columns:
                vol_col = vc
                break
        if vol_col is not None:
            vol = data[vol_col].values.astype(float)
            vol = np.nan_to_num(vol, nan=0.0)
            # Use exponential moving average for smoother percentile
            alpha = 2.0 / (20 + 1)
            vol_rank_ema = np.full(n, 0.5)
            for i in range(1, n):
                window = vol[max(0, i - 50):i + 1]
                if len(window) > 0:
                    rank = np.sum(window <= vol[i]) / len(window)
                    vol_rank_ema[i] = alpha * rank + (1 - alpha) * vol_rank_ema[i-1]
            vol_rank = vol_rank_ema[1:]

        # Volatility (rolling standard deviation of returns)
        volatility = np.full(n - 1, 0.1)
        for i in range(20, n):
            window_returns = returns[max(0, i-20):i]
            if len(window_returns) > 0:
                volatility[i-1] = np.std(window_returns, ddof=1)

        features = np.column_stack([returns, atr_ratio, vol_rank, volatility])
        return features

    def _classify_hmm_v2(self, symbol, close, high, low, data):
        """
        Enhanced HMM classification with Viterbi decoding.

        Returns:
            (regime, confidence, viterbi_seq, emission_probs)
        """
        features = self._build_enhanced_features(close, high, low, data)
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
                    n_iter=200,
                    random_state=42,
                    tol=1e-4,
                    verbose=False
                )
                # Dirichlet prior encouraging self-transition
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

                # Map states
                self._label_states_v2(symbol, model, train_data)
            except Exception as e:
                print(f"HMM fit failed for {symbol}: {e}")
                # Fallback
                return self._classify_ker_with_emission(close)

        model = self._hmm_models.get(symbol)
        if model is None:
            return self._classify_ker_with_emission(close)

        try:
            # Predict with Viterbi decoding for optimal sequence
            recent_features = features[-train_n:]
            if len(recent_features) < 5:
                return self._classify_ker_with_emission(close)

            # Viterbi decoding for optimal state sequence
            viterbi_seq = model.predict(recent_features)

            # Posterior probabilities (emission probabilities)
            emission_probs = model.predict_proba(recent_features)
            current_state = viterbi_seq[-1]
            confidence = float(emission_probs[-1, current_state])

            labels = self._hmm_labels.get(symbol, {})
            regime = labels.get(current_state, {}).get('label', 'sideways')

            return regime, confidence, viterbi_seq, emission_probs
        except Exception as e:
            print(f"HMM prediction failed for {symbol}: {e}")
            return self._classify_ker_with_emission(close)

    def _classify_ker_with_emission(self, close):
        """Fallback with emission probability simulation."""
        er = self._calc_ker(close)
        if er > self.ker_trend_thresh:
            # Determine direction
            lookback = min(20, len(close) - 1)
            if lookback > 0:
                price_change = (close[-1] - close[-lookback]) / close[-lookback]
            else:
                price_change = 0.0
            regime = 'bull' if price_change > 0 else 'bear'
            confidence = min(1.0, er / 0.5)
            viterbi_seq = [0] * len(close)
            emission_probs = np.array([[confidence, 1-confidence]])
            return regime, confidence, viterbi_seq, emission_probs
        elif er < self.ker_range_thresh:
            confidence = 1.0 - er / self.ker_range_thresh
            return 'sideways', confidence, [0] * len(close), np.array([[confidence]])
        else:
            return 'sideways', 0.5, [0] * len(close), np.array([[0.5]])

    def _label_states_v2(self, symbol, model, train_data):
        """
        Enhanced state labeling with more sophisticated analysis.
        """
        state_seq = model.predict(train_data)
        means = model.means_
        covars = model.covars_

        labels = {}
        state_metrics = []

        for s in range(self.n_states):
            mask = state_seq == s
            if np.sum(mask) > 0:
                state_returns = train_data[mask, 0]
                mean_ret = float(np.mean(state_returns))
                std_ret = float(np.std(state_returns, ddof=1))
                vol_level = float(np.mean(train_data[mask, 3]))  # volatility feature
            else:
                mean_ret = float(means[s, 0])
                std_ret = np.sqrt(float(covars[s, 0, 0]))
                vol_level = float(means[s, 3])

            state_metrics.append({
                'state': s,
                'mean_ret': mean_ret,
                'std_ret': std_ret,
                'vol_level': vol_level,
                'score': mean_ret / (std_ret + 1e-8)  # risk-adjusted return
            })

        # Sort by score (risk-adjusted returns)
        state_metrics.sort(key=lambda x: x['score'], reverse=True)

        # Assign labels based on characteristics
        for i, metric in enumerate(state_metrics):
            s = metric['state']
            if metric['mean_ret'] > 0.001 and metric['vol_level'] < 0.02:
                labels[s] = {'label': 'bull', 'score': metric['score']}
            elif metric['mean_ret'] < -0.001 and metric['vol_level'] < 0.02:
                labels[s] = {'label': 'bear', 'score': metric['score']}
            else:
                labels[s] = {'label': 'sideways', 'score': metric['score']}

        self._hmm_labels[symbol] = labels

    # ================================================================
    # Regime-Specific Strategies
    # ================================================================

    def _bull_strategy(self, close, atr):
        """Bull regime strategy with structural tension."""
        n = len(close)
        score = 0

        # EMA cross
        ema_f = self._calc_ema_series(close, self.ema_fast)
        ema_s = self._calc_ema_series(close, self.ema_slow)

        ema_cross_up = n >= 2 and ema_f[-2] < ema_s[-2] and ema_f[-1] > ema_s[-1]
        above_ema = close[-1] > ema_s[-1]

        if ema_cross_up:
            score += 5
        elif above_ema and ema_f[-1] > ema_s[-1]:
            score += 3

        # EMA momentum
        if n > 5:
            ema_slope = (ema_s[-1] - ema_s[-5]) / ema_s[-5]
            if ema_slope > 0.02:
                score += 2

        # Structural tension (enhanced)
        tension = self._calc_structural_tension(close, atr)
        if tension > 0.5:
            score += 4
        elif tension > 0.2:
            score += 2
        elif tension < -0.3:
            score -= 2

        # Price momentum confirmation
        lookback = min(10, n - 1)
        if lookback > 0:
            momentum = (close[-1] - close[-lookback]) / close[-lookback]
            if momentum > 0.05:
                score += 3

        direction = 1 if score > 0 else -1
        return score, direction

    def _bear_strategy(self, close, atr):
        """Bear regime strategy - primarily short focused."""
        n = len(close)
        score = 0

        # EMA direction
        ema_f = self._calc_ema_series(close, self.ema_fast)
        ema_s = self._calc_ema_series(close, self.ema_slow)

        below_ema = close[-1] < ema_s[-1]
        ema_bearish = ema_f[-1] < ema_s[-1]

        # Short signals
        if ema_bearish and below_ema:
            score -= 6
        elif ema_bearish:
            score -= 4
        elif below_ema:
            score -= 2

        # Structural tension
        tension = self._calc_structural_tension(close, atr)
        if tension < -0.4:
            score -= 3
        elif tension > 0.4:
            score += 1  # Weak bullish dissent

        # Momentum confirmation
        lookback = min(10, n - 1)
        if lookback > 0:
            momentum = (close[-1] - close[-lookback]) / close[-lookback]
            if momentum < -0.05:
                score -= 3

        direction = -1 if score < 0 else 1
        return score, direction

    def _sideways_strategy(self, close, data, atr):
        """Sideways regime with mean reversion signals."""
        score = 0

        # RSI
        rsi = self._calc_rsi(close)
        if rsi < self.rsi_os:
            score += 5
            if rsi < 20:
                score += 2
        elif rsi > self.rsi_ob:
            score -= 5
            if rsi > 80:
                score -= 2

        # Z-score
        zscore = self._calc_zscore(close)
        if zscore < -self.zscore_thresh:
            score += 4
        elif zscore > self.zscore_thresh:
            score -= 4

        # Bollinger band style
        if len(close) >= 20:
            sma20 = np.mean(close[-20:])
            std20 = np.std(close[-20:], ddof=1)
            if std20 > 0:
                bb_pos = (close[-1] - sma20) / (2 * std20)
                if bb_pos < -1.0:
                    score += 3
                elif bb_pos > 1.0:
                    score -= 3

        direction = 1 if score > 0 else -1
        return score, direction

    # ================================================================
    # Helper Methods
    # ================================================================

    def _calc_structural_tension(self, close, atr):
        """Enhanced structural tension calculation."""
        n = len(close)
        if n < self.pivot_len * 2 + 5:
            return 0.0

        high_arr = self.data['high'].values.astype(float) if 'high' in self.data.columns else close
        low_arr = self.data['low'].values.astype(float) if 'low' in self.data.columns else close

        # Find swing points
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

        # Enhanced tension calculation
        prev_sh = swing_highs[-2][1]
        curr_sh = swing_highs[-1][1]
        prev_sl = swing_lows[-2][1]
        curr_sl = swing_lows[-1][1]

        # Calculate displacement from reference points
        current_price = close[-1]
        reference_points = [prev_sh, curr_sh, prev_sl, curr_sl]

        if atr <= 0:
            return 0.0

        displacements = [(current_price - rp) / atr for rp in reference_points]
        tension = float(np.mean(displacements))

        return tension

    def _calc_atr(self, data):
        """ATR calculation."""
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
        """ATR as a full series."""
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
        """Z-score calculation."""
        if len(close) < self.zscore_period:
            return 0.0
        recent = close[-self.zscore_period:]
        std = float(np.std(recent, ddof=1))
        if std < 1e-10:
            return 0.0
        return float((close[-1] - np.mean(recent)) / std)

    def _calc_ker(self, close):
        """Kaufman Efficiency Ratio."""
        period = min(20, len(close) - 1)
        if period < 2:
            return 0.0
        recent = close[-period:]
        net = abs(float(recent[-1] - recent[0]))
        total = float(np.sum(np.abs(np.diff(recent))))
        return net / total if total > 0 else 0.0

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

        close = data['close'].values.astype(float)
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        atr = self._calc_atr(data)

        if _HMM_AVAILABLE:
            regime, confidence, _, _ = self._classify_hmm_v2(sym, close, high, low, data)
            persistence_prob = self._get_persistence_probability(sym, regime)
        else:
            regime, confidence = self._classify_ker(close)
            persistence_prob = 0.5

        # Generate strategy signal
        score = 0
        direction = 0
        if regime == 'bull':
            score, direction = self._bull_strategy(close, atr)
        elif regime == 'bear':
            score, direction = self._bear_strategy(close, atr)
        elif regime == 'sideways':
            score, direction = self._sideways_strategy(close, data, atr)

        # Apply persistence filter
        final_score = score * confidence * persistence_prob

        price = float(data['close'].iloc[-1])
        if abs(final_score) >= self.score_threshold:
            return {
                'action': 'buy' if direction == 1 else 'sell',
                'reason': f'HMM {regime} regime, score={final_score:.1f}',
                'price': price,
            }
        return {'action': 'hold', 'reason': f'insufficient score={final_score:.1f}',
                'price': price}