"""
Information Geometry Strategy (InfoGeometryStrategy)
===================================================
"道可道非常道" — the market distribution constantly changes.

Core Concept:
- Use Fisher Information distance to detect when the return distribution has shifted significantly
- Large Fisher distance = regime change = trading opportunity
- Fisher information matrix measures the curvature of the information geometry
- KL divergence as secondary signal for distribution change detection
- Direction from structural tension (price vs swing anchors)

Mathematical Foundation:
Fisher Information:
  I(θ) = E[∂log f(X;θ)/∂θ]² — measures the amount of information about θ contained in X
  For discrete returns: I(μ,σ²) = E[∂log f/∂μ]² + E[∂log f/∂σ²]²

Fisher Information Distance:
  d_Fisher(P,Q) = √[tr(I^{-1}(P)(I(P)-I(Q))²)] — distance between distributions in information space

KL Divergence:
  D_KL(P||Q) = Σ p_i log(p_i/q_i) — asymmetric measure of distribution difference

Strategy Logic:
1. Compute rolling return distribution (last 60 bars)
2. Calculate Fisher Information distance between current and previous distribution
3. Large Fisher distance = regime change = trading opportunity
4. Use KL divergence as secondary signal
5. Direction from structural tension (price vs swing anchors)
6. Position size proportional to Fisher distance (bigger shift = bigger position)
7. ATR trailing stop for exits

Technical Indicators: Fisher Distance, KL Divergence, Structural Tension, ATR
Category: adaptive
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from core.base_strategy import BaseStrategy


class InfoGeometryStrategy(BaseStrategy):
    """Information Geometry Strategy — trade regime changes detected by Fisher distance"""

    strategy_description = (
        "InfoGeometry: Fisher Information distance + KL divergence + "
        "structural tension direction + ATR trailing stop"
    )
    strategy_category = "adaptive"
    strategy_params_schema = {
        "return_window": {"type": "int", "default": 60, "label": "Return distribution window"},
        "fisher_threshold": {"type": "float", "default": 0.1, "label": "Fisher distance threshold"},
        "kl_threshold": {"type": "float", "default": 0.05, "label": "KL divergence threshold"},
        "num_bins": {"type": "int", "default": 20, "label": "Distribution bins for entropy"},
        "pivot_len": {"type": "int", "default": 7, "label": "Pivot lookback for structural tension"},
        "ema_period": {"type": "int", "default": 20, "label": "Trend EMA for structural tension"},
        "atr_period": {"type": "int", "default": 14, "label": "ATR period"},
        "trail_atr_mult": {"type": "float", "default": 2.5, "label": "Trail stop ATR multiplier"},
        "position_size_scale": {"type": "float", "default": 0.3, "label": "Position size scale per Fisher distance"},
        "max_position": {"type": "float", "default": 1.0, "label": "Maximum position size"},
        "hold_min": {"type": "int", "default": 3, "label": "Min holding days"},
        "max_hold": {"type": "int", "default": 60, "label": "Max holding days"},
        "fisher_history": {"type": "int", "default": 30, "label": "Lookback for Fisher baseline"},
    }

    def __init__(self, data: pd.DataFrame, params: dict = None):
        super().__init__(data, params)
        p = self.params
        self.return_window = p.get('return_window', 60)
        self.fisher_threshold = p.get('fisher_threshold', 0.1)
        self.kl_threshold = p.get('kl_threshold', 0.05)
        self.num_bins = p.get('num_bins', 20)
        self.pivot_len = p.get('pivot_len', 7)
        self.ema_period = p.get('ema_period', 20)
        self.atr_period = p.get('atr_period', 14)
        self.trail_atr_mult = p.get('trail_atr_mult', 2.5)
        self.position_size_scale = p.get('position_size_scale', 0.3)
        self.max_position = p.get('max_position', 1.0)
        self.hold_min = p.get('hold_min', 3)
        self.max_hold = p.get('max_hold', 60)
        self.fisher_history = p.get('fisher_history', 30)

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'return_window': 60, 'fisher_threshold': 0.1,
            'kl_threshold': 0.05, 'num_bins': 20,
            'pivot_len': 7, 'ema_period': 20,
            'atr_period': 14, 'trail_atr_mult': 2.5,
            'position_size_scale': 0.3, 'max_position': 1.0,
            'hold_min': 3, 'max_hold': 60,
            'fisher_history': 30,
        }

    def _compute_returns(self, data: pd.DataFrame) -> pd.Series:
        """Compute logarithmic returns"""
        return np.log(data['close']).diff().dropna()

    def _compute_histogram(self, returns: pd.Series, bins: int = 20) -> np.ndarray:
        """Compute normalized histogram of returns"""
        hist, _ = np.histogram(returns, bins=bins, density=True)
        # Normalize to ensure it sums to 1 (probability distribution)
        return hist / hist.sum()

    def _compute_fisher_information(self, distribution: np.ndarray) -> np.ndarray:
        """Compute Fisher information for a discrete distribution"""
        # For discrete distribution with uniform grid spacing
        # I(μ) = Σ p_i * (∂log p_i / ∂μ)²
        # Simplified for equally spaced bins
        n = len(distribution)
        if n < 2:
            return np.array([0.0])

        # Derivative approximation
        log_p = np.log(distribution + 1e-10)  # Avoid log(0)
        derivative = np.gradient(log_p)

        # Fisher information
        fisher = np.sum(distribution * derivative**2)
        return np.array([fisher])

    def _compute_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute KL divergence D_KL(p||q)"""
        p = np.maximum(p, 1e-10)  # Avoid log(0)
        q = np.maximum(q, 1e-10)  # Avoid log(0)
        return np.sum(p * np.log(p / q))

    def _compute_fisher_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute Fisher information distance between two distributions"""
        # Fisher information for each distribution
        I_p = self._compute_fisher_information(p)
        I_q = self._compute_fisher_information(q)

        # Fisher distance (simplified for 1D)
        if len(I_p) > 0 and len(I_q) > 0:
            return np.sqrt((I_p[0] - I_q[0])**2)
        return 0.0

    def _compute_structural_tension(self, data: pd.DataFrame) -> float:
        """
        Compute structural tension based on price vs swing anchors
        Returns: -1 to 1 (negative = bearish, positive = bullish)
        """
        close = data['close']

        # Find swing highs and lows
        highs = data['high'].rolling(window=self.pivot_len, center=True).max()
        lows = data['low'].rolling(window=self.pivot_len, center=True).min()

        # EMA trend
        ema = close.ewm(span=self.ema_period).mean()

        # Price position relative to anchors
        tension = (close - ema) / (highs - lows).replace(0, 1)

        # Normalize to [-1, 1]
        tension = np.tanh(tension * 2)

        return float(tension.iloc[-1]) if len(tension) > 0 else 0.0

    def _compute_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def generate_signals(self) -> List[Dict]:
        data = self.data.copy()
        if 'symbol' not in data.columns:
            data['symbol'] = 'DEFAULT'

        # Sort data by date
        data = data.sort_index()

        # Ensure we have enough data
        if len(data) < self.return_window + self.fisher_history + 50:
            return []

        self.signals = []

        # Precompute ATR
        data['atr'] = self._compute_atr(data, self.atr_period)

        # Generate signals for each unique time
        unique_times = sorted(data.index.unique())

        # Track holding period
        current_holding = None
        buy_time = None
        buy_price = None

        # Fisher history for baseline
        fisher_history = []

        for i, current_time in enumerate(unique_times):
            current_bars = data.loc[current_time]
            if isinstance(current_bars, pd.Series):
                current_bars = pd.DataFrame([current_bars])

            # Update fisher history
            if i >= self.return_window:
                # Compute returns for the lookback window
                lookback_data = data.loc[current_time - pd.Timedelta(days=self.return_window):current_time]
                returns = self._compute_returns(lookback_data)

                # Skip if not enough returns
                if len(returns) < 20:
                    continue

                # Compute current distribution
                current_dist = self._compute_histogram(returns, self.num_bins)

                # Compute Fisher information for current distribution
                current_fisher = self._compute_fisher_information(current_dist)[0]

                # Compare with previous distribution
                if i >= self.return_window + 1:
                    prev_lookback_data = data.loc[current_time - pd.Timedelta(days=self.return_window+1):current_time - pd.Timedelta(days=1)]
                    prev_returns = self._compute_returns(prev_lookback_data)
                    prev_dist = self._compute_histogram(prev_returns, self.num_bins)

                    # Compute Fisher distance
                    fisher_dist = self._compute_fisher_distance(current_dist, prev_dist)

                    # Compute KL divergence
                    kl_div = self._compute_kl_divergence(current_dist, prev_dist)

                    # Add to history
                    fisher_history.append(fisher_dist)
                    if len(fisher_history) > self.fisher_history:
                        fisher_history.pop(0)

                    # Detect regime change
                    fisher_baseline = np.mean(fisher_history) if fisher_history else 0.01
                    is_regime_change = fisher_dist > self.fisher_threshold and fisher_dist > fisher_baseline

                    # Get structural tension for direction
                    tension = self._compute_structural_tension(data.loc[:current_time])

                    # Signal generation
                    if current_holding is None and is_regime_change:
                        # Generate signal
                        signal_strength = min(fisher_dist / self.fisher_threshold, 3.0)
                        position_size = min(
                            self.position_size_scale * signal_strength,
                            self.max_position
                        )

                        # Only trade when both Fisher and KL suggest change
                        if kl_div > self.kl_threshold * 0.5:
                            action = 'buy' if tension > 0.2 else 'sell'

                            self._record_signal(
                                timestamp=current_time,
                                action=action,
                                symbol=current_bars['symbol'].iloc[0],
                                price=float(current_bars['close'].iloc[0]),
                                fisher_distance=fisher_dist,
                                kl_divergence=kl_div,
                                position_size=position_size,
                                structural_tension=tension,
                                regime_change='true'
                            )

                            current_holding = action
                            buy_time = current_time
                            buy_price = float(current_bars['close'].iloc[0])

                    elif current_holding is not None:
                        # Check exit conditions
                        days_held = (current_time - buy_time).days
                        current_price = float(current_bars['close'].iloc[0])
                        current_atr = float(current_bars['atr'].iloc[0])

                        # ATR trailing stop
                        if current_holding == 'buy':
                            stop_price = buy_price + current_atr * self.trail_atr_mult
                            if current_price < stop_price or days_held > self.max_hold:
                                action = 'sell'
                            else:
                                action = 'hold'
                        else:  # current_holding == 'sell'
                            stop_price = buy_price - current_atr * self.trail_atr_mult
                            if current_price > stop_price or days_held > self.max_hold:
                                action = 'buy'
                            else:
                                action = 'hold'

                        # Record exit signal
                        if action == 'hold' and days_held < self.hold_min:
                            continue

                        if action != 'hold':
                            self._record_signal(
                                timestamp=current_time,
                                action=action,
                                symbol=current_bars['symbol'].iloc[0],
                                price=float(current_bars['close'].iloc[0]),
                                exit_reason='atr_stop' if current_price < stop_price else 'max_hold',
                                days_held=days_held
                            )
                            current_holding = None
                            buy_time = None
                            buy_price = None

        return self.signals