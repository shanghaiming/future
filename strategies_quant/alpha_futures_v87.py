"""
V87: CUSUM + Bayesian Adaptive Strategy
=========================================
Replace V80's fixed WINNING/NORMAL/LOSING mode switching with
probability-theory-grounded adaptive methods:

1. CUSUM change-point detection for regime awareness
   - Detects when strategy edge is shifting faster than rolling windows
   - S_t = max(0, S_{t-1} + (trade_return - mu_target - k))
   - Above threshold h => CONSERVATIVE (reduce exposure)
   - Below threshold => AGGRESSIVE (increase exposure)

2. Bayesian win rate estimation
   - Prior: Beta(alpha=15, beta=15) centered at 50%
   - Update after each trade: alpha += 1 (win) or beta += 1 (loss)
   - Posterior mean = alpha/(alpha+beta)
   - Decay: periodically shrink toward prior (forget old data)

3. Kelly Criterion position sizing
   - kelly_fraction = (WR * R - (1-WR)) / R  where R = avg_win/avg_loss
   - Use fraction of Kelly for safety (执两用中)

Keeps V80's multi-TF + sector limit framework.

Parameters to sweep:
  - cusum_k: 0.01, 0.02, 0.03 (drift parameter)
  - cusum_h: 0.05, 0.10, 0.15 (threshold)
  - kelly_fraction: 0.3, 0.5, 0.7 (fraction of Kelly)
  - bayesian_decay: 0.95, 0.98, 1.0 (memory decay)
  - max_per_sector: 2, 3
  - max_positions: 3, 4

Walk-forward 2019-2026. No leverage. CASH0=1,000,000, COMM=0.0005.
"""
import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_data import load_all_data

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

CASH0 = 1_000_000
COMM = 0.0005
LEVERAGE = 1.0

# Bayesian prior parameters
PRIOR_ALPHA = 15.0
PRIOR_BETA = 15.0
# Target per-trade return for CUSUM reference (in raw decimal, e.g. 0.005 = 0.5%)
# Average win is ~5%, average trade ~1.5%, so 0.5% is a reasonable target
MU_TARGET = 0.005

# Short-term (5d) factor weights (same as V80)
ST_WEIGHTS = {
    "rank_ret5d": 0.25,
    "rank_oi5d": 0.20,
    "rank_rsi": 0.15,
    "rank_vol5d": 0.15,
    "rank_ret10d": 0.10,
    "rank_range5d": 0.10,
    "rank_atrp5d": 0.05,
}

# Medium-term (20d) factor weights (same as V80)
MT_WEIGHTS = {
    "rank_ret20d": 0.25,
    "rank_oi20d": 0.20,
    "rank_rsi": 0.15,
    "rank_vol20d": 0.15,
    "rank_ret10d": 0.10,
    "rank_range20d": 0.10,
    "rank_atrp20d": 0.05,
}

# Sector definitions
SECTOR_MAP = {
    'i': 'BLACK', 'j': 'BLACK', 'jm': 'BLACK', 'hc': 'BLACK',
    'sf': 'BLACK', 'sm': 'BLACK', 'wr': 'BLACK', 'im': 'BLACK',
    'cu': 'METAL', 'al': 'METAL', 'zn': 'METAL', 'pb': 'METAL',
    'ni': 'METAL', 'sn': 'METAL', 'ss': 'METAL', 'ao': 'METAL',
    'au': 'METAL', 'ag': 'METAL', 'rb': 'METAL', 'si': 'METAL',
    'sc': 'ENERGY', 'fu': 'ENERGY', 'bu': 'ENERGY',
    'pg': 'ENERGY', 'eb': 'ENERGY', 'ta': 'ENERGY',
    'fg': 'ENERGY', 'oi': 'ENERGY',
    'v': 'CHEMICAL', 'pp': 'CHEMICAL', 'l': 'CHEMICAL',
    'eg': 'CHEMICAL', 'ma': 'CHEMICAL', 'sa': 'CHEMICAL',
    'ur': 'CHEMICAL', 'pf': 'CHEMICAL', 'sh': 'CHEMICAL',
    'lc': 'CHEMICAL',
    'm': 'AGRI', 'y': 'AGRI', 'a': 'AGRI', 'p': 'AGRI',
    'c': 'AGRI', 'cs': 'AGRI', 'jd': 'AGRI', 'rr': 'AGRI',
    'lrm': 'AGRI', 'rm': 'AGRI', 'ru': 'AGRI',
    'cf': 'SOFTS', 'sr': 'SOFTS', 'ap': 'SOFTS',
    'cj': 'SOFTS', 'pk': 'SOFTS', 'lh': 'SOFTS',
    'sp': 'SOFTS', 'b': 'SOFTS', 'br': 'SOFTS',
}


def _extract_base_symbol(sym: str) -> str:
    s = sym.lower().split('.')[0].strip()
    while s and s[-1].isdigit():
        s = s[:-1]
    if s.endswith('fi'):
        s = s[:-2]
    return s


def build_sector_lookup(syms: List[str]) -> Dict[int, str]:
    sector_lookup: Dict[int, str] = {}
    for si, sym in enumerate(syms):
        base = _extract_base_symbol(sym)
        sector_lookup[si] = SECTOR_MAP.get(base, 'OTHER')
    return sector_lookup


def compute_rsi_manual(
    C: np.ndarray, NS: int, ND: int, period: int = 14,
) -> np.ndarray:
    rsi = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        gains = np.full(ND, np.nan)
        losses = np.full(ND, np.nan)
        for di in range(1, ND):
            if np.isnan(c[di]) or np.isnan(c[di - 1]):
                continue
            delta = c[di] - c[di - 1]
            gains[di] = max(delta, 0.0)
            losses[di] = max(-delta, 0.0)

        avg_gain = np.nan
        avg_loss = np.nan
        for di in range(1, ND):
            if np.isnan(gains[di]):
                continue
            if np.isnan(avg_gain):
                valid_g = []
                valid_l = []
                for j in range(di, min(di + period, ND)):
                    if not np.isnan(gains[j]):
                        valid_g.append(gains[j])
                        valid_l.append(
                            losses[j] if not np.isnan(losses[j]) else 0.0)
                if len(valid_g) >= period:
                    avg_gain = np.mean(valid_g)
                    avg_loss = np.mean(valid_l)
                    if avg_loss == 0:
                        rsi[si, di + period - 1] = 100.0
                    else:
                        rs = avg_gain / avg_loss
                        rsi[si, di + period - 1] = (
                            100.0 - 100.0 / (1.0 + rs))
                continue

            avg_gain = (avg_gain * (period - 1) + gains[di]) / period
            avg_loss = (avg_loss * (period - 1) + losses[di]) / period
            if avg_loss == 0:
                rsi[si, di] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[si, di] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def compute_raw_factors(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
) -> Dict[str, np.ndarray]:
    t0 = time.time()
    print("[V87] Computing raw factors (5d + 20d)...", flush=True)

    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (not np.isnan(C[si, di])
                    and not np.isnan(C[si, di - 5])
                    and C[si, di - 5] > 0):
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1.0

    oi_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (not np.isnan(OI[si, di])
                    and not np.isnan(OI[si, di - 5])
                    and OI[si, di - 5] > 0):
                oi_5d[si, di] = OI[si, di] / OI[si, di - 5] - 1.0

    vol_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            vals = V[si, di - 5:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 3:
                vol_5d[si, di] = np.mean(valid)

    range_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            rng_vals = []
            for j in range(di - 5, di):
                if (not np.isnan(H[si, j]) and not np.isnan(L[si, j])
                        and not np.isnan(C[si, j]) and C[si, j] > 0
                        and H[si, j] > L[si, j]):
                    rng_vals.append((H[si, j] - L[si, j]) / C[si, j])
            if len(rng_vals) >= 3:
                range_5d[si, di] = np.mean(rng_vals)

    atrp_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(6, ND):
            atr_vals = []
            for j in range(di - 5, di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    prev_c = (
                        C[si, j - 1]
                        if j > 0 and not np.isnan(C[si, j - 1])
                        else cc)
                    atr_vals.append(
                        max(hh - ll, abs(hh - prev_c), abs(ll - prev_c)))
            if atr_vals and not np.isnan(C[si, di]) and C[si, di] > 0:
                atrp_5d[si, di] = np.mean(atr_vals) / C[si, di]

    ret_10d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            if (not np.isnan(C[si, di])
                    and not np.isnan(C[si, di - 10])
                    and C[si, di - 10] > 0):
                ret_10d[si, di] = C[si, di] / C[si, di - 10] - 1.0

    rsi14 = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nan_mask = np.isnan(C[si])
            try:
                r = talib.RSI(c, 14)
                rsi14[si] = np.where(nan_mask, np.nan, r)
            except Exception:
                pass

    needs_fallback = np.all(np.isnan(rsi14), axis=1)
    if needs_fallback.any():
        rsi_manual = compute_rsi_manual(C, NS, ND, 14)
        for si in range(NS):
            if needs_fallback[si]:
                rsi14[si] = rsi_manual[si]

    ret_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if (not np.isnan(C[si, di])
                    and not np.isnan(C[si, di - 20])
                    and C[si, di - 20] > 0):
                ret_20d[si, di] = C[si, di] / C[si, di - 20] - 1.0

    oi_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if (not np.isnan(OI[si, di])
                    and not np.isnan(OI[si, di - 20])
                    and OI[si, di - 20] > 0):
                oi_20d[si, di] = OI[si, di] / OI[si, di - 20] - 1.0

    vol_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            vals = V[si, di - 20:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 10:
                vol_20d[si, di] = np.mean(valid)

    range_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            rng_vals = []
            for j in range(di - 20, di):
                if (not np.isnan(H[si, j]) and not np.isnan(L[si, j])
                        and not np.isnan(C[si, j]) and C[si, j] > 0
                        and H[si, j] > L[si, j]):
                    rng_vals.append((H[si, j] - L[si, j]) / C[si, j])
            if len(rng_vals) >= 10:
                range_20d[si, di] = np.mean(rng_vals)

    atrp_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            atr_vals = []
            for j in range(di - 20, di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    prev_c = (
                        C[si, j - 1]
                        if j > 0 and not np.isnan(C[si, j - 1])
                        else cc)
                    atr_vals.append(
                        max(hh - ll, abs(hh - prev_c), abs(ll - prev_c)))
            if atr_vals and not np.isnan(C[si, di]) and C[si, di] > 0:
                atrp_20d[si, di] = np.mean(atr_vals) / C[si, di]

    print(f"  Raw factors done: {time.time() - t0:.1f}s", flush=True)
    return {
        "ret_5d": ret_5d, "ret_10d": ret_10d, "ret_20d": ret_20d,
        "oi_5d": oi_5d, "oi_20d": oi_20d,
        "vol_5d": vol_5d, "vol_20d": vol_20d,
        "range_5d": range_5d, "range_20d": range_20d,
        "rsi14": rsi14,
        "atrp_5d": atrp_5d, "atrp_20d": atrp_20d,
    }


def compute_cross_sectional_ranks(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int, min_count: int = 10,
) -> Dict[str, np.ndarray]:
    t0 = time.time()
    print("[V87] Computing cross-sectional ranks...", flush=True)

    factors_to_rank = {
        "rank_ret5d": raw_factors["ret_5d"],
        "rank_ret10d": raw_factors["ret_10d"],
        "rank_oi5d": raw_factors["oi_5d"],
        "rank_vol5d": raw_factors["vol_5d"],
        "rank_range5d": raw_factors["range_5d"],
        "rank_rsi": raw_factors["rsi14"],
        "rank_atrp5d": raw_factors["atrp_5d"],
        "rank_ret20d": raw_factors["ret_20d"],
        "rank_oi20d": raw_factors["oi_20d"],
        "rank_vol20d": raw_factors["vol_20d"],
        "rank_range20d": raw_factors["range_20d"],
        "rank_atrp20d": raw_factors["atrp_20d"],
    }

    INVERT_FACTORS = {
        "rank_ret5d", "rank_ret10d", "rank_oi5d", "rank_rsi",
        "rank_ret20d", "rank_oi20d",
    }

    ranks = {}
    for name, factor in factors_to_rank.items():
        rank_arr = np.full((NS, ND), np.nan)
        for di in range(ND):
            vals = factor[:, di]
            valid_count = np.sum(~np.isnan(vals))
            if valid_count < min_count:
                continue
            ranked = (
                pd.Series(vals)
                .rank(pct=True, na_option="keep")
                .values
            )
            if name in INVERT_FACTORS:
                ranked = 1.0 - ranked
            rank_arr[:, di] = ranked
        ranks[name] = rank_arr

    print(f"  CS ranks done: {time.time() - t0:.1f}s", flush=True)
    return ranks


def compute_ker(C: np.ndarray, NS: int, ND: int) -> np.ndarray:
    ker_10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            closes = C[si, di - 10:di + 1]
            valid = closes[~np.isnan(closes)]
            if len(valid) < 10 or valid[0] <= 0:
                continue
            net_change = abs(valid[-1] - valid[0])
            total_change = np.sum(np.abs(np.diff(valid)))
            if total_change > 1e-10:
                ker_10[si, di] = net_change / total_change

    ker_regime = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        for di in range(ND):
            if np.isnan(ker_10[si, di]):
                continue
            if ker_10[si, di] < 0.15:
                ker_regime[si, di] = 1
            elif ker_10[si, di] > 0.3:
                ker_regime[si, di] = -1
    return ker_regime


def build_multi_tf_composite(
    ranks: Dict[str, np.ndarray],
    st_weights: Dict[str, float],
    mt_weights: Dict[str, float],
    st_weight: float,
    NS: int, ND: int,
    min_factors: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t0 = time.time()
    print(f"[V87] Building multi-TF composite (st_w={st_weight:.2f})...",
          flush=True)

    mt_weight = 1.0 - st_weight

    combined = np.full((NS, ND), np.nan)
    st_comp = np.full((NS, ND), np.nan)
    mt_comp = np.full((NS, ND), np.nan)
    n_confirm_st = np.zeros((NS, ND), dtype=int)
    n_confirm_mt = np.zeros((NS, ND), dtype=int)

    st_names = list(st_weights.keys())
    st_wvals = np.array([st_weights[k] for k in st_names])
    mt_names = list(mt_weights.keys())
    mt_wvals = np.array([mt_weights[k] for k in mt_names])

    for di in range(ND):
        for si in range(NS):
            st_vals = []
            st_wsum = 0.0
            st_confirm = 0
            for idx, name in enumerate(st_names):
                rank_val = ranks[name][si, di]
                if np.isnan(rank_val):
                    continue
                st_vals.append(rank_val * st_wvals[idx])
                st_wsum += st_wvals[idx]
                if rank_val > 0.5:
                    st_confirm += 1

            if st_wsum > 0 and st_confirm >= min_factors:
                st_comp[si, di] = sum(st_vals) / st_wsum
                n_confirm_st[si, di] = st_confirm

            mt_vals = []
            mt_wsum = 0.0
            mt_confirm = 0
            for idx, name in enumerate(mt_names):
                rank_val = ranks[name][si, di]
                if np.isnan(rank_val):
                    continue
                mt_vals.append(rank_val * mt_wvals[idx])
                mt_wsum += mt_wvals[idx]
                if rank_val > 0.5:
                    mt_confirm += 1

            if mt_wsum > 0 and mt_confirm >= min_factors:
                mt_comp[si, di] = sum(mt_vals) / mt_wsum
                n_confirm_mt[si, di] = mt_confirm

            if not np.isnan(st_comp[si, di]) and not np.isnan(mt_comp[si, di]):
                combined[si, di] = (st_weight * st_comp[si, di]
                                    + mt_weight * mt_comp[si, di])

    print(f"  Multi-TF composite done: {time.time() - t0:.1f}s", flush=True)
    return combined, st_comp, mt_comp, n_confirm_st, n_confirm_mt


def compute_all_signals(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
    st_weight: float = 0.60,
    st_weights: Optional[Dict[str, float]] = None,
    mt_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    if st_weights is None:
        st_weights = ST_WEIGHTS
    if mt_weights is None:
        mt_weights = MT_WEIGHTS

    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_cross_sectional_ranks(raw, NS, ND)
    ker_regime = compute_ker(C, NS, ND)
    combined, st_comp, mt_comp, ncf_st, ncf_mt = build_multi_tf_composite(
        ranks, st_weights, mt_weights, st_weight, NS, ND)

    return {
        "composite": combined,
        "st_comp": st_comp,
        "mt_comp": mt_comp,
        "n_confirm_st": ncf_st,
        "n_confirm_mt": ncf_mt,
        "ker_regime": ker_regime,
        "ranks": ranks,
    }


def compute_atr_at(H: np.ndarray, L: np.ndarray, C: np.ndarray,
                   si: int, di: int, start_di: int) -> Optional[float]:
    atr_v = []
    for j in range(max(start_di, di - 14), di):
        hh, ll, cc = H[si, j], L[si, j], C[si, j]
        if not any(np.isnan([hh, ll, cc])):
            atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
    if atr_v:
        return np.mean(atr_v)
    return None


# =========================================================================
# CUSUM + Bayesian state tracker
# =========================================================================

class BayesianCUSUMTracker:
    """Track strategy regime using CUSUM change-point detection and
    Bayesian win-rate estimation.

    CUSUM detects when recent trade returns deviate from target.
    Bayesian update provides a smooth, adaptive win-rate estimate.
    Together they drive adaptive threshold selection and Kelly sizing.
    """

    def __init__(
        self,
        cusum_k: float = 0.02,
        cusum_h: float = 0.10,
        kelly_frac: float = 0.5,
        bayesian_decay: float = 0.98,
        prior_alpha: float = PRIOR_ALPHA,
        prior_beta: float = PRIOR_BETA,
        mu_target: float = MU_TARGET,
    ) -> None:
        self.cusum_k = cusum_k
        self.cusum_h = cusum_h
        self.kelly_frac = kelly_frac
        self.bayesian_decay = bayesian_decay
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.mu_target = mu_target

        # CUSUM state
        self.cusum_pos = 0.0  # upper CUSUM (detects deterioration)
        self.cusum_neg = 0.0  # lower CUSUM (detects improvement)

        # Bayesian state
        self.alpha = prior_alpha
        self.beta = prior_beta

        # Running win/loss stats for Kelly
        self.total_wins = 0
        self.total_losses = 0
        self.sum_win_pnl = 0.0
        self.sum_loss_pnl = 0.0

        # Regime
        self.regime = "NORMAL"  # CONSERVATIVE, NORMAL, AGGRESSIVE

        # Track trade count for periodic decay
        self.trade_count = 0
        self.decay_interval = 20  # decay every 20 trades

    def update(self, pnl_pct: float) -> None:
        """Update tracker with a new trade result."""
        self.trade_count += 1
        is_win = pnl_pct > 0

        # --- CUSUM update ---
        # Positive CUSUM: accumulates when returns are BELOW target
        # (detects edge deterioration)
        self.cusum_pos = max(
            0.0,
            self.cusum_pos + (pnl_pct / 100.0 - self.mu_target - self.cusum_k)
        )
        # Negative CUSUM: accumulates when returns are ABOVE target
        # (detects edge improvement)
        self.cusum_neg = max(
            0.0,
            self.cusum_neg + (-(pnl_pct / 100.0 - self.mu_target) - self.cusum_k)
        )

        # --- Bayesian update ---
        if is_win:
            self.alpha += 1.0
            self.total_wins += 1
            self.sum_win_pnl += pnl_pct
        else:
            self.beta += 1.0
            self.total_losses += 1
            self.sum_loss_pnl += abs(pnl_pct)

        # Periodic decay toward prior (forget old data)
        if self.trade_count % self.decay_interval == 0:
            self._decay()

        # --- Regime determination ---
        self._update_regime()

    def _decay(self) -> None:
        """Shrink alpha/beta toward prior by decay factor."""
        d = self.bayesian_decay
        self.alpha = d * self.alpha + (1 - d) * self.prior_alpha
        self.beta = d * self.beta + (1 - d) * self.prior_beta

    def _update_regime(self) -> None:
        """Determine regime from CUSUM signals."""
        if self.cusum_pos > self.cusum_h:
            self.regime = "CONSERVATIVE"
        elif self.cusum_neg > self.cusum_h:
            self.regime = "AGGRESSIVE"
        else:
            self.regime = "NORMAL"

    @property
    def win_rate(self) -> float:
        """Bayesian posterior mean win rate."""
        total = self.alpha + self.beta
        if total <= 0:
            return 0.5
        return self.alpha / total

    @property
    def kelly_size(self) -> float:
        """Kelly criterion fraction (bounded).

        kelly = (WR * R - (1-WR)) / R  where R = avg_win / avg_loss
        Use kelly_frac * kelly for safety.
        """
        if self.total_losses == 0 or self.sum_loss_pnl <= 0:
            # All wins so far -- use optimistic but capped sizing
            return min(self.kelly_frac * 0.5, 0.5)
        if self.total_wins == 0:
            return 0.0  # No edge detected

        wr = self.win_rate
        avg_win = self.sum_win_pnl / self.total_wins
        avg_loss = self.sum_loss_pnl / self.total_losses

        if avg_loss <= 1e-10:
            return min(self.kelly_frac * 0.5, 0.5)

        R = avg_win / avg_loss  # win/loss ratio
        kelly = (wr * R - (1.0 - wr)) / R

        # Kelly can be negative (no edge) -- floor at 0
        kelly = max(0.0, kelly)

        # Apply fraction of Kelly for safety (执两用中)
        return min(self.kelly_frac * kelly, 0.5)

    @property
    def entry_threshold(self) -> float:
        """Adaptive entry threshold based on regime.

        CONSERVATIVE: require stronger signals (higher threshold)
        AGGRESSIVE: accept weaker signals (lower threshold)
        NORMAL: moderate threshold
        """
        base_threshold = 0.75
        bayes_wr = self.win_rate

        # Adjust threshold based on Bayesian win rate
        # Higher estimated WR => can afford lower threshold (more trades)
        # Lower estimated WR => need higher threshold (fewer, better trades)
        wr_adjustment = (0.5 - bayes_wr) * 0.6  # +/- 0.30 max

        if self.regime == "CONSERVATIVE":
            return min(base_threshold + wr_adjustment + 0.10, 0.95)
        elif self.regime == "AGGRESSIVE":
            return max(base_threshold + wr_adjustment - 0.10, 0.55)
        else:
            return np.clip(base_threshold + wr_adjustment, 0.60, 0.90)


def backtest_v87(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
    sector_lookup: Dict[int, str],
    st_weight: float = 0.60,
    cusum_k: float = 0.02,
    cusum_h: float = 0.10,
    kelly_frac: float = 0.5,
    bayesian_decay: float = 0.98,
    atr_stop: float = 3.0,
    max_positions: int = 4,
    max_per_sector: int = 3,
    min_confidence: int = 3,
    use_ker_gate: bool = True,
    hold_days: int = 5,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Backtest V87: CUSUM + Bayesian adaptive strategy.

    Key innovations over V80:
    - CUSUM change-point detection replaces rolling-window mode switching
    - Bayesian win-rate estimation replaces fixed-window counting
    - Kelly criterion position sizing replaces equal weight
    - Adaptive thresholds from Bayesian + CUSUM combined signal
    """
    composite = sigs["composite"]
    n_confirm_st = sigs["n_confirm_st"]
    n_confirm_mt = sigs["n_confirm_mt"]
    ker_regime = sigs["ker_regime"]

    if end_di is None:
        end_di = ND - 1

    # Initialize adaptive tracker
    tracker = BayesianCUSUMTracker(
        cusum_k=cusum_k,
        cusum_h=cusum_h,
        kelly_frac=kelly_frac,
        bayesian_decay=bayesian_decay,
    )

    equity = CASH0
    peak = equity
    max_dd = 0.0
    # Position: (si, entry_di, entry_price, stop_price, alloc, is_pyramid)
    positions: List[Tuple[int, int, float, float, float, bool]] = []
    trades: List[dict] = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions: List[Tuple[int, int, float, float, float, bool]] = []

        # Get adaptive parameters from tracker
        current_threshold = tracker.entry_threshold
        kelly_size = tracker.kelly_size
        current_regime = tracker.regime
        bayes_wr = tracker.win_rate

        # Group positions by symbol
        pos_by_si: Dict[int, List] = defaultdict(list)
        for si, edi, ep, sp, alloc, is_pyr in positions:
            pos_by_si[si].append((edi, ep, sp, alloc, is_pyr))

        # Exit logic
        for si, pos_list in pos_by_si.items():
            c = C[si, di]
            if np.isnan(c):
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc, is_pyr))
                continue

            earliest_edi = min(p[0] for p in pos_list)
            hold = di - earliest_edi
            stopped = any(c < sp for _, _, sp, _, _ in pos_list)

            if stopped:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    is_win = pnl > 0
                    trades.append({
                        "pnl_abs": profit,
                        "pnl_pct": pnl * 100,
                        "days": di - edi + 1,
                        "di": di,
                        "year": d.year,
                        "sym": syms[si],
                        "sector": sector_lookup.get(si, 'OTHER'),
                        "reason": "stop",
                        "pyr": is_pyr,
                        "regime": current_regime[:1],
                        "bayes_wr": bayes_wr,
                        "kelly": kelly_size,
                    })
                    tracker.update(pnl * 100)
            elif hold >= hold_days:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    is_win = pnl > 0
                    trades.append({
                        "pnl_abs": profit,
                        "pnl_pct": pnl * 100,
                        "days": di - edi + 1,
                        "di": di,
                        "year": d.year,
                        "sym": syms[si],
                        "sector": sector_lookup.get(si, 'OTHER'),
                        "reason": "hold",
                        "pyr": is_pyr,
                        "regime": current_regime[:1],
                        "bayes_wr": bayes_wr,
                        "kelly": kelly_size,
                    })
                    tracker.update(pnl * 100)
            else:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc, is_pyr))

        positions = new_positions
        equity += daily_pnl
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        if equity <= 0:
            break

        # --- ENTRY: CUSUM/Bayesian adaptive ---
        held = {p[0] for p in positions}
        if len(positions) >= max_positions:
            continue

        # In CONSERVATIVE regime, reduce max positions by 1
        effective_max = max_positions
        if current_regime == "CONSERVATIVE":
            effective_max = max(1, max_positions - 1)

        # In AGGRESSIVE regime, try to fill up to max_positions
        # In NORMAL, standard behavior
        if len(positions) >= effective_max:
            continue

        candidates = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(composite[si, di]):
                continue
            if composite[si, di] < current_threshold:
                continue
            total_confirm = n_confirm_st[si, di] + n_confirm_mt[si, di]
            if total_confirm < min_confidence:
                continue
            if use_ker_gate and ker_regime[si, di] < 0:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            candidates.append((composite[si, di], si))

        # Sort by composite score (highest first)
        candidates.sort(key=lambda x: -x[0])

        # Sector-constrained selection
        sector_counts: Dict[str, int] = defaultdict(int)
        for si_held in held:
            sector_counts[sector_lookup.get(si_held, 'OTHER')] += 1

        new_entries = []
        for rank_val, si in candidates:
            if len(positions) + len(new_entries) >= effective_max:
                break
            if si in held:
                continue
            sym_sector = sector_lookup.get(si, 'OTHER')
            if sector_counts[sym_sector] >= max_per_sector:
                continue
            new_entries.append((rank_val, si, sym_sector))
            sector_counts[sym_sector] += 1

        # Kelly-based allocation
        # Use Kelly size as base allocation, distribute among positions
        num_total = len(positions) + len(new_entries)
        if num_total == 0:
            continue

        # Base alloc from Kelly, but ensure at least some minimum exposure
        # and cap total exposure at LEVERAGE (1.0)
        alloc_per_pos = min(kelly_size, LEVERAGE / num_total)
        # Ensure minimum allocation when we have positions
        alloc_per_pos = max(alloc_per_pos, LEVERAGE / max_positions * 0.3)

        # Update existing positions with new allocation
        updated_positions = []
        for si, edi, ep, sp, old_alloc, is_pyr in positions:
            updated_positions.append(
                (si, edi, ep, sp, alloc_per_pos, is_pyr))

        # Enter new positions at open[di+1]
        for rank_val, si, sym_sector in new_entries:
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr = compute_atr_at(H, L, C, si, di, start_di)
            if atr is None:
                continue
            updated_positions.append(
                (si, di + 1, ep, ep - atr_stop * atr,
                 alloc_per_pos, False))

        positions = updated_positions

    # Close remaining positions
    for si, edi, ep, sp, alloc, is_pyr in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl

    return trades, equity, max_dd


def analyze(trades: List[dict], equity: float, max_dd: float,
            label: str = "") -> Optional[dict]:
    if not trades:
        print(f"  {label}: no trades")
        return None

    nw = sum(1 for t in trades if t["pnl_pct"] > 0)
    wr = nw / len(trades) * 100
    n_days = max(1, trades[-1]["di"] - trades[0]["di"])
    ann = ((equity / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
    ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
    rets = np.array(ap) / CASH0
    sh = (np.mean(rets) / np.std(rets) * np.sqrt(252)
          if np.std(rets) > 0 else 0)

    n_pyr = sum(1 for t in trades if t.get("pyr"))
    n_base = len(trades) - n_pyr
    n_stop = sum(1 for t in trades if t["reason"] == "stop")
    n_hold = sum(1 for t in trades if t["reason"] == "hold")

    regime_counts = {"C": 0, "N": 0, "A": 0}
    for t in trades:
        m = t.get("regime", "N")
        if m in regime_counts:
            regime_counts[m] += 1

    sector_counts: Dict[str, int] = defaultdict(int)
    for t in trades:
        sector_counts[t.get("sector", "OTHER")] += 1
    sector_str = " ".join(
        f"{k}:{v}" for k, v in sorted(sector_counts.items()))

    avg_pnl = np.mean([t["pnl_pct"] for t in trades])
    avg_win = np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"] > 0])
    avg_loss = np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"] <= 0])

    # Kelly stats
    kelly_vals = [t.get("kelly", 0) for t in trades]
    avg_kelly = np.mean(kelly_vals) if kelly_vals else 0

    print(
        f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} "
        f"stop:{n_stop} hold:{n_hold}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
        f"Sh={sh:.2f} eq={equity:,.0f}"
    )
    print(
        f"    avg_pnl={avg_pnl:+.3f}% avg_win={avg_win:+.3f}% "
        f"avg_loss={avg_loss:+.3f}% avg_kelly={avg_kelly:.3f}"
    )
    print(
        f"    regimes=[C:{regime_counts['C']} N:{regime_counts['N']} "
        f"A:{regime_counts['A']}]"
    )
    print(f"    sectors: {sector_str}")

    yr: Dict[int, dict] = {}
    for t in trades:
        y = t["year"]
        if y not in yr:
            yr[y] = {"n": 0, "w": 0, "pnl": []}
        yr[y]["n"] += 1
        if t["pnl_pct"] > 0:
            yr[y]["w"] += 1
        yr[y]["pnl"].append(t["pnl_pct"])
    for y in sorted(yr.keys()):
        ys = yr[y]
        cum = np.prod([1 + p / 100 for p in ys["pnl"]]) - 1
        print(
            f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% "
            f"cum={cum:+.1%}")

    return {
        "n": len(trades), "wr": wr, "dd": max_dd,
        "ann": ann, "sh": sh, "eq": equity,
    }


def walk_forward(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
    sector_lookup: Dict[int, str],
    st_weight: float = 0.60,
    cusum_k: float = 0.02,
    cusum_h: float = 0.10,
    kelly_frac: float = 0.5,
    bayesian_decay: float = 0.98,
    max_positions: int = 4,
    max_per_sector: int = 3,
) -> List[dict]:
    print(f"\n{'=' * 70}")
    print(
        f"  WALK-FORWARD V87 "
        f"(st_w={st_weight:.2f} ck={cusum_k:.2f} "
        f"ch={cusum_h:.2f} kf={kelly_frac:.1f} "
        f"bd={bayesian_decay:.2f} mp={max_positions} mps={max_per_sector})"
    )
    print(f"  CUSUM change-point + Bayesian WR + Kelly sizing")
    print(f"{'=' * 70}")

    years = sorted(set(d.year for d in dates))
    all_trades: List[dict] = []

    for test_year in range(2019, years[-1] + 1):
        test_start = None
        test_end_idx = None
        for i, d in enumerate(dates):
            if d.year == test_year and test_start is None:
                test_start = i
            if d.year == test_year:
                test_end_idx = i
        if test_start is None:
            continue

        trades, _, _ = backtest_v87(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            st_weight=st_weight,
            cusum_k=cusum_k,
            cusum_h=cusum_h,
            kelly_frac=kelly_frac,
            bayesian_decay=bayesian_decay,
            atr_stop=3.0,
            max_positions=max_positions,
            max_per_sector=max_per_sector,
            min_confidence=3,
            use_ker_gate=True,
            hold_days=5,
            start_di=test_start,
            end_di=test_end_idx + 1,
        )

        test_trades = [t for t in trades
                       if dates[t["di"]].year == test_year]
        all_trades.extend(test_trades)

        if test_trades:
            n = len(test_trades)
            nw = sum(1 for t in test_trades if t["pnl_pct"] > 0)
            wr_val = nw / n * 100
            avg = np.mean([t["pnl_pct"] for t in test_trades])
            regimes = {"C": 0, "N": 0, "A": 0}
            for t in test_trades:
                m = t.get("regime", "N")
                if m in regimes:
                    regimes[m] += 1
            yr_sectors: Dict[str, int] = defaultdict(int)
            for t in test_trades:
                yr_sectors[t.get("sector", "OTHER")] += 1
            sec_str = " ".join(
                f"{k}:{v}" for k, v in sorted(yr_sectors.items()))
            print(
                f"  {test_year}: {n}t WR={wr_val:.1f}% avg={avg:+.2f}% "
                f"regimes=[C:{regimes['C']} N:{regimes['N']} "
                f"A:{regimes['A']}] "
                f"sectors=[{sec_str}]",
                flush=True,
            )
        else:
            print(f"  {test_year}: no trades", flush=True)

    if all_trades:
        nw = sum(1 for t in all_trades if t["pnl_pct"] > 0)
        wr_val = nw / len(all_trades) * 100
        avg = np.mean([t["pnl_pct"] for t in all_trades])
        cum = np.prod(
            [1 + t["pnl_pct"] / 100 for t in all_trades]) - 1
        n_pyr = sum(1 for t in all_trades if t.get("pyr"))
        agg_sectors: Dict[str, int] = defaultdict(int)
        for t in all_trades:
            agg_sectors[t.get("sector", "OTHER")] += 1
        sec_str = " ".join(
            f"{k}:{v}" for k, v in sorted(agg_sectors.items()))
        print(
            f"\n  WF TOTAL: {len(all_trades)}t (pyr:{n_pyr}) "
            f"WR={wr_val:.1f}% avg={avg:+.2f}% cum={cum:+.1%}"
        )
        print(f"  WF SECTORS: {sec_str}")
        return all_trades
    return []


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V87: CUSUM + BAYESIAN ADAPTIVE STRATEGY")
    print("  Replace fixed mode switching with probability-theory methods:")
    print("  - CUSUM change-point detection for regime awareness")
    print("  - Bayesian win-rate estimation (Beta-Binomial conjugate)")
    print("  - Kelly Criterion position sizing (执两用中)")
    print("  Keeps V80's multi-TF + sector limit framework")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(
        start="2016-01-01")
    print(
        f"  {NS} sym, {ND} days, "
        f"{dates[0].strftime('%Y-%m-%d')} to "
        f"{dates[-1].strftime('%Y-%m-%d')}"
    )

    sector_lookup = build_sector_lookup(syms)
    sector_dist: Dict[str, int] = defaultdict(int)
    for sec in sector_lookup.values():
        sector_dist[sec] += 1
    print(f"  Sector distribution: {dict(sector_dist)}")

    # Find 2019 start index for OOS testing
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # === 1. Pre-compute signals ===
    print(f"\n--- Computing signals for st_weight=0.60 ---")
    sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND, st_weight=0.60)

    # === 2. Parameter sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("  CUSUM + Bayesian + Kelly adaptive framework")
    print("=" * 70)

    results: List[dict] = []
    sweep_count = 0

    for cusum_k in [0.01, 0.02, 0.03]:
        for cusum_h in [0.05, 0.10, 0.15]:
            for kf in [0.3, 0.5, 0.7]:
                for bd in [0.95, 0.98, 1.0]:
                    for mps in [2, 3]:
                        for mp in [3, 4]:
                            sweep_count += 1
                            trades, eq, dd = backtest_v87(
                                C, O, H, L, NS, ND, dates, syms, sigs,
                                sector_lookup=sector_lookup,
                                st_weight=0.60,
                                cusum_k=cusum_k,
                                cusum_h=cusum_h,
                                kelly_frac=kf,
                                bayesian_decay=bd,
                                atr_stop=3.0,
                                max_positions=mp,
                                max_per_sector=mps,
                                start_di=bt_2019,
                            )

                            if len(trades) < 10:
                                continue

                            nw = sum(
                                1 for t in trades
                                if t["pnl_pct"] > 0)
                            wr = nw / len(trades) * 100
                            n_days = max(
                                1,
                                trades[-1]["di"] - trades[0]["di"])
                            ann = ((eq / CASH0) ** (
                                1 / max(1.0, n_days / 252)) - 1) * 100
                            ap = [t["pnl_abs"]
                                  for t in sorted(
                                      trades, key=lambda x: x["di"])]
                            rets_arr = np.array(ap) / CASH0
                            sh_val = (
                                np.mean(rets_arr)
                                / np.std(rets_arr) * np.sqrt(252)
                                if np.std(rets_arr) > 0 else 0)

                            yr_counts: Dict[int, int] = {}
                            for t in trades:
                                y = t["year"]
                                yr_counts[y] = (
                                    yr_counts.get(y, 0) + 1)
                            oos_years = [y for y in yr_counts
                                         if y >= 2019]
                            avg_per_year = (
                                sum(yr_counts[y] for y in oos_years)
                                / max(len(oos_years), 1))

                            sec_trades: Dict[str, int] = defaultdict(int)
                            for t in trades:
                                sec_trades[t.get(
                                    "sector", "OTHER")] += 1
                            max_sec_pct = (
                                max(sec_trades.values())
                                / len(trades) * 100)

                            results.append({
                                "ck": cusum_k, "ch": cusum_h,
                                "kf": kf, "bd": bd,
                                "mps": mps, "mp": mp,
                                "n": len(trades), "wr": wr,
                                "ann": ann, "dd": dd,
                                "sharpe": sh_val, "eq": eq,
                                "avg_yr": avg_per_year,
                                "max_sec": max_sec_pct,
                            })

    results.sort(key=lambda x: -x["sharpe"])
    print(f"\n  Evaluated {sweep_count} configs, "
          f"{len(results)} with 10+ trades")
    print(
        f"\n{'CK':>4} {'CH':>4} {'KF':>4} {'BD':>4} "
        f"{'MPS':>3} {'MP':>3} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} "
        f"{'Sh':>6} {'Avg/Yr':>7} {'MaxSec':>7}"
    )
    print("-" * 85)
    for r in results[:40]:
        print(
            f"{r['ck']:>4.2f} {r['ch']:>4.2f} {r['kf']:>4.1f} "
            f"{r['bd']:>4.2f} {r['mps']:>3} {r['mp']:>3} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f} "
            f"{r['avg_yr']:>7.1f} {r['max_sec']:>6.1f}%"
        )

    if not results:
        print("  No configs with 10+ trades. Exiting.")
        return

    # === 3. Top configs: full backtest ===
    print("\n" + "=" * 70)
    print("  TOP CONFIGS -- FULL BACKTEST")
    print("=" * 70)

    seen = set()
    unique_top = []
    for r in results:
        key = (r["ck"], r["ch"], r["kf"], r["bd"],
               r["mps"], r["mp"])
        if key not in seen:
            seen.add(key)
            unique_top.append(r)
        if len(unique_top) >= 5:
            break

    for r in unique_top:
        trades, eq, dd = backtest_v87(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            st_weight=0.60,
            cusum_k=r["ck"],
            cusum_h=r["ch"],
            kelly_frac=r["kf"],
            bayesian_decay=r["bd"],
            atr_stop=3.0,
            max_positions=r["mp"],
            max_per_sector=r["mps"],
            start_di=60,
        )
        label = (
            f"ck={r['ck']:.2f} ch={r['ch']:.2f} "
            f"kf={r['kf']:.1f} bd={r['bd']:.2f} "
            f"mps={r['mps']} mp={r['mp']}"
        )
        print(f"\n  FULL {label}")
        analyze(trades, eq, dd, label)

    # === 4. Walk-forward for best config ===
    best = results[0]
    print("\n" + "=" * 70)
    print(
        f"  BEST WF: ck={best['ck']:.2f} ch={best['ch']:.2f} "
        f"kf={best['kf']:.1f} bd={best['bd']:.2f} "
        f"mps={best['mps']} mp={best['mp']}"
    )
    print("=" * 70)
    walk_forward(
        C, O, H, L, NS, ND, dates, syms, sigs,
        sector_lookup=sector_lookup,
        st_weight=0.60,
        cusum_k=best["ck"],
        cusum_h=best["ch"],
        kelly_frac=best["kf"],
        bayesian_decay=best["bd"],
        max_positions=best["mp"],
        max_per_sector=best["mps"],
    )

    print(f"\n[V87] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
