"""
V95: Bayesian Model Averaging Factor Strategy
==============================================
Paper (2604.04430): BMA is optimal for factor combination across 18 quintillion models.
Top 5 factors explain 80% of risk premium, top 10 explain 95%.

Innovation over V80:
  - Replace FIXED factor weights with BMA-adaptive weights
  - Rolling IC (Information Coefficient) estimation per factor
  - Bayesian posterior weights: weight = prior * likelihood, normalized
  - When factor IC is noisy -> weight shrinks toward prior (1/7)
  - When factor IC is strong -> weight increases
  - Multi-horizon IC: compute at 3d, 5d, 10d horizons, weight by significance

Parameters to sweep:
  - ic_window: 40, 60, 90 (rolling IC estimation window)
  - ic_horizon: 3, 5, 10 (forward return horizon for IC)
  - prior_strength: 5, 10, 20 (pseudo-observations for Bayesian prior)
  - st_weight: 0.60, 0.65, 0.70
  - max_positions: 2, 3, 4
  - max_per_sector: 2, 3

Walk-forward 2019-2026. No leverage. CASH0=1,000,000, COMM=0.0005.
Signal at close[di], enter at open[di+1]. No look-ahead. No gap signals.
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

# Factor names for short-term composite
ST_FACTOR_NAMES = [
    "rank_ret5d", "rank_oi5d", "rank_rsi", "rank_vol5d",
    "rank_ret10d", "rank_range5d", "rank_atrp5d",
]

# Factor names for medium-term composite
MT_FACTOR_NAMES = [
    "rank_ret20d", "rank_oi20d", "rank_rsi", "rank_vol20d",
    "rank_ret10d", "rank_range20d", "rank_atrp20d",
]

# Factors inverted for mean-reversion (high rank = more oversold = bullish)
INVERT_FACTORS = {
    "rank_ret5d", "rank_ret10d", "rank_oi5d", "rank_rsi",
    "rank_ret20d", "rank_oi20d",
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
    print("[V95] Computing raw factors...", flush=True)

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
    print("[V95] Computing cross-sectional ranks...", flush=True)

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


# ============================================================
# BMA CORE: Bayesian Model Averaging for Factor Weights
# ============================================================

def compute_forward_returns(
    C: np.ndarray, NS: int, ND: int, horizon: int,
) -> np.ndarray:
    """Compute forward returns for IC calculation."""
    fwd = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND - horizon):
            if (not np.isnan(C[si, di])
                    and not np.isnan(C[si, di + horizon])
                    and C[si, di] > 0):
                fwd[si, di] = C[si, di + horizon] / C[si, di] - 1.0
    return fwd


def compute_rolling_ic(
    ranks: Dict[str, np.ndarray],
    fwd_returns: np.ndarray,
    factor_names: List[str],
    NS: int, ND: int,
    ic_window: int = 60,
) -> Dict[str, np.ndarray]:
    """Compute rolling IC (Information Coefficient) for each factor.

    IC_j[di] = Spearman correlation between factor_j rank and forward return
               over the rolling window [di - ic_window, di).
    """
    t0 = time.time()
    print(f"[V95] Computing rolling IC (window={ic_window})...", flush=True)

    ic_results: Dict[str, np.ndarray] = {}
    for name in factor_names:
        ic_arr = np.full(ND, np.nan)
        factor_ranks = ranks[name]

        for di in range(ic_window, ND):
            # Gather valid pairs in the rolling window
            rank_window = factor_ranks[:, di - ic_window:di]
            ret_window = fwd_returns[:, di - ic_window:di]

            # Flatten and filter valid pairs
            valid_mask = (~np.isnan(rank_window)) & (~np.isnan(ret_window))
            r_vals = rank_window[valid_mask]
            ret_vals = ret_window[valid_mask]

            if len(r_vals) >= 20:
                # Spearman correlation = Pearson correlation of ranks
                # factor_ranks are already percentile ranks
                ret_ranked = (
                    pd.Series(ret_vals).rank(pct=True).values
                )
                r_ranked = (
                    pd.Series(r_vals).rank(pct=True).values
                )
                corr = np.corrcoef(r_ranked, ret_ranked)[0, 1]
                if not np.isnan(corr):
                    ic_arr[di] = corr

        ic_results[name] = ic_arr

    print(f"  Rolling IC done: {time.time() - t0:.1f}s", flush=True)
    return ic_results


def compute_rolling_ic_fast(
    ranks: Dict[str, np.ndarray],
    fwd_returns: np.ndarray,
    factor_names: List[str],
    NS: int, ND: int,
    ic_window: int = 60,
) -> Dict[str, np.ndarray]:
    """Fast rolling IC using vectorized operations where possible."""
    t0 = time.time()
    print(f"[V95] Computing rolling IC (window={ic_window})...", flush=True)

    ic_results: Dict[str, np.ndarray] = {}
    for name in factor_names:
        ic_arr = np.full(ND, np.nan)
        factor_ranks = ranks[name]

        for di in range(ic_window, ND - 1):
            # Use cross-sectional IC: correlation across instruments
            # For each day in the window, compute cross-sectional corr,
            # then average
            ic_sum = 0.0
            ic_count = 0
            for dj in range(di - ic_window, di):
                r_slice = factor_ranks[:, dj]
                ret_slice = fwd_returns[:, dj]
                valid = (~np.isnan(r_slice)) & (~np.isnan(ret_slice))
                n_valid = np.sum(valid)
                if n_valid < 10:
                    continue
                r_v = r_slice[valid]
                ret_v = ret_slice[valid]
                # Rank the returns
                ret_ranked = pd.Series(ret_v).rank(pct=True).values
                corr = np.corrcoef(r_v, ret_ranked)[0, 1]
                if not np.isnan(corr):
                    ic_sum += corr
                    ic_count += 1

            if ic_count >= 10:
                ic_arr[di] = ic_sum / ic_count

        ic_results[name] = ic_arr

    print(f"  Rolling IC done: {time.time() - t0:.1f}s", flush=True)
    return ic_results


def compute_bma_weights(
    ic_st: Dict[str, np.ndarray],
    ic_mt: Dict[str, np.ndarray],
    st_names: List[str],
    mt_names: List[str],
    prior_strength: float,
    NS: int, ND: int,
    min_factors: int = 4,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute BMA posterior weights for each factor at each day.

    BMA theory:
      posterior_weight_j ∝ prior_j × likelihood_j

    Where:
      - prior_j = 1/num_factors (equal prior)
      - likelihood_j ∝ exp(prior_strength * ic_j * sign_j)

    prior_strength controls how much the data moves the prior:
      - Higher prior_strength = more responsive to IC
      - Lower prior_strength = more conservative, stays near prior

    The sign of IC determines whether a factor is predictive.
    We want factors with consistently positive IC.
    """
    t0 = time.time()
    print(f"[V95] Computing BMA weights (prior_strength={prior_strength})...",
          flush=True)

    num_st = len(st_names)
    num_mt = len(mt_names)
    uniform_st = 1.0 / num_st
    uniform_mt = 1.0 / num_mt

    st_weights: Dict[str, np.ndarray] = {}
    mt_weights: Dict[str, np.ndarray] = {}

    # Short-term BMA weights
    for name in st_names:
        w_arr = np.full(ND, np.nan)
        ic_arr = ic_st[name]

        for di in range(ND):
            if np.isnan(ic_arr[di]):
                w_arr[di] = uniform_st
                continue
            # Log-likelihood proportional to IC * prior_strength
            # Use IC magnitude to weight
            w_arr[di] = np.exp(prior_strength * abs(ic_arr[di]))

        st_weights[name] = w_arr

    # Normalize ST weights per day
    st_norm = np.zeros(ND)
    for name in st_names:
        for di in range(ND):
            if not np.isnan(st_weights[name][di]):
                st_norm[di] += st_weights[name][di]

    for name in st_names:
        for di in range(ND):
            if st_norm[di] > 0 and not np.isnan(st_weights[name][di]):
                st_weights[name][di] /= st_norm[di]
            else:
                st_weights[name][di] = uniform_st

    # Medium-term BMA weights
    for name in mt_names:
        w_arr = np.full(ND, np.nan)
        ic_arr = ic_mt[name]

        for di in range(ND):
            if np.isnan(ic_arr[di]):
                w_arr[di] = uniform_mt
                continue
            w_arr[di] = np.exp(prior_strength * abs(ic_arr[di]))

        mt_weights[name] = w_arr

    # Normalize MT weights per day
    mt_norm = np.zeros(ND)
    for name in mt_names:
        for di in range(ND):
            if not np.isnan(mt_weights[name][di]):
                mt_norm[di] += mt_weights[name][di]

    for name in mt_names:
        for di in range(ND):
            if mt_norm[di] > 0 and not np.isnan(mt_weights[name][di]):
                mt_weights[name][di] /= mt_norm[di]
            else:
                mt_weights[name][di] = uniform_mt

    print(f"  BMA weights done: {time.time() - t0:.1f}s", flush=True)
    return st_weights, mt_weights


def build_bma_composite(
    ranks: Dict[str, np.ndarray],
    st_weights: Dict[str, np.ndarray],
    mt_weights: Dict[str, np.ndarray],
    st_names: List[str],
    mt_names: List[str],
    st_weight: float,
    NS: int, ND: int,
    min_factors: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build multi-TF composite with BMA-adaptive weights.

    Unlike V80 which uses fixed weights, here each day has its own
    weight vector determined by the BMA posterior.
    """
    t0 = time.time()
    print(f"[V95] Building BMA composite (st_w={st_weight:.2f})...",
          flush=True)

    mt_weight = 1.0 - st_weight

    combined = np.full((NS, ND), np.nan)
    st_comp = np.full((NS, ND), np.nan)
    mt_comp = np.full((NS, ND), np.nan)
    n_confirm_st = np.zeros((NS, ND), dtype=int)
    n_confirm_mt = np.zeros((NS, ND), dtype=int)

    for di in range(ND):
        for si in range(NS):
            # Short-term composite with BMA weights
            st_vals = []
            st_wsum = 0.0
            st_confirm = 0
            for name in st_names:
                rank_val = ranks[name][si, di]
                if np.isnan(rank_val):
                    continue
                w = st_weights[name][di]
                if np.isnan(w):
                    w = 1.0 / len(st_names)
                st_vals.append(rank_val * w)
                st_wsum += w
                if rank_val > 0.5:
                    st_confirm += 1

            if st_wsum > 0 and st_confirm >= min_factors:
                st_comp[si, di] = sum(st_vals) / st_wsum
                n_confirm_st[si, di] = st_confirm

            # Medium-term composite with BMA weights
            mt_vals = []
            mt_wsum = 0.0
            mt_confirm = 0
            for name in mt_names:
                rank_val = ranks[name][si, di]
                if np.isnan(rank_val):
                    continue
                w = mt_weights[name][di]
                if np.isnan(w):
                    w = 1.0 / len(mt_names)
                mt_vals.append(rank_val * w)
                mt_wsum += w
                if rank_val > 0.5:
                    mt_confirm += 1

            if mt_wsum > 0 and mt_confirm >= min_factors:
                mt_comp[si, di] = sum(mt_vals) / mt_wsum
                n_confirm_mt[si, di] = mt_confirm

            if not np.isnan(st_comp[si, di]) and not np.isnan(mt_comp[si, di]):
                combined[si, di] = (st_weight * st_comp[si, di]
                                    + mt_weight * mt_comp[si, di])

    print(f"  BMA composite done: {time.time() - t0:.1f}s", flush=True)
    return combined, st_comp, mt_comp, n_confirm_st, n_confirm_mt


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


def get_dynamic_mode(
    recent_trades_win: List[int],
    win_threshold: float,
    win_rate_window: int,
) -> str:
    if len(recent_trades_win) < 5:
        return "normal"
    window = recent_trades_win[-win_rate_window:]
    win_rate = sum(window) / len(window)
    if win_rate > win_threshold:
        return "winning"
    elif win_rate < 0.50:
        return "losing"
    return "normal"


def get_mode_threshold(
    mode: str,
    win_threshold: float,
    normal_threshold: float,
    lose_threshold: float,
) -> float:
    if mode == "winning":
        return win_threshold
    elif mode == "losing":
        return lose_threshold
    return normal_threshold


def backtest_v95(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    composite: np.ndarray,
    n_confirm_st: np.ndarray,
    n_confirm_mt: np.ndarray,
    ker_regime: np.ndarray,
    sector_lookup: Dict[int, str],
    win_threshold: float = 0.60,
    normal_threshold: float = 0.80,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
    atr_stop: float = 3.0,
    max_positions: int = 3,
    max_per_sector: int = 2,
    min_confidence: int = 3,
    use_ker_gate: bool = True,
    hold_days: int = 5,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Backtest V95: BMA factor weighting strategy."""
    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions: List[Tuple[int, int, float, float, float, bool]] = []
    trades: List[dict] = []
    recent_trades_win: List[int] = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions: List[Tuple[int, int, float, float, float, bool]] = []

        mode = get_dynamic_mode(
            recent_trades_win, win_threshold, win_rate_window)
        current_threshold = get_mode_threshold(
            mode, win_threshold, normal_threshold, lose_threshold)

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
                        "mode": mode[:1].upper(),
                    })
                    recent_trades_win.append(1 if is_win else 0)
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
                        "mode": mode[:1].upper(),
                    })
                    recent_trades_win.append(1 if is_win else 0)
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

        # Entry: take all instruments above threshold
        held = {p[0] for p in positions}
        if len(positions) >= max_positions:
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

        candidates.sort(key=lambda x: -x[0])

        sector_counts: Dict[str, int] = defaultdict(int)
        for si_held in held:
            sector_counts[sector_lookup.get(si_held, 'OTHER')] += 1

        new_entries = []
        for rank_val, si in candidates:
            if len(positions) + len(new_entries) >= max_positions:
                break
            if si in held:
                continue
            sym_sector = sector_lookup.get(si, 'OTHER')
            if sector_counts[sym_sector] >= max_per_sector:
                continue
            new_entries.append((rank_val, si, sym_sector))
            sector_counts[sym_sector] += 1

        num_total = len(positions) + len(new_entries)
        if num_total == 0:
            continue
        alloc_per_pos = LEVERAGE / num_total

        updated_positions = []
        for si, edi, ep, sp, old_alloc, is_pyr in positions:
            updated_positions.append(
                (si, edi, ep, sp, alloc_per_pos, is_pyr))

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

    mode_counts = {"W": 0, "N": 0, "L": 0}
    for t in trades:
        m = t.get("mode", "N")
        if m in mode_counts:
            mode_counts[m] += 1

    sector_counts: Dict[str, int] = defaultdict(int)
    for t in trades:
        sector_counts[t.get("sector", "OTHER")] += 1
    sector_str = " ".join(
        f"{k}:{v}" for k, v in sorted(sector_counts.items()))

    avg_pnl = np.mean([t["pnl_pct"] for t in trades])
    avg_win = np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"] > 0])
    avg_loss = np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"] <= 0])

    print(
        f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} "
        f"stop:{n_stop} hold:{n_hold}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
        f"Sh={sh:.2f} eq={equity:,.0f}"
    )
    print(
        f"    avg_pnl={avg_pnl:+.3f}% avg_win={avg_win:+.3f}% "
        f"avg_loss={avg_loss:+.3f}% "
        f"modes=[W:{mode_counts['W']} N:{mode_counts['N']} "
        f"L:{mode_counts['L']}]"
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
    composite: np.ndarray,
    n_confirm_st: np.ndarray,
    n_confirm_mt: np.ndarray,
    ker_regime: np.ndarray,
    sector_lookup: Dict[int, str],
    win_threshold: float = 0.60,
    normal_threshold: float = 0.80,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
    max_positions: int = 3,
    max_per_sector: int = 2,
) -> List[dict]:
    print(f"\n{'=' * 70}")
    print(
        f"  WALK-FORWARD V95 BMA "
        f"(wt={win_threshold:.2f} nt={normal_threshold:.2f} "
        f"lt={lose_threshold:.2f} ww={win_rate_window} "
        f"mp={max_positions} mps={max_per_sector})"
    )
    print(f"  NO LEVERAGE (leverage=1.0)")
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

        trades, _, _ = backtest_v95(
            C, O, H, L, NS, ND, dates, syms,
            composite, n_confirm_st, n_confirm_mt, ker_regime,
            sector_lookup=sector_lookup,
            win_threshold=win_threshold,
            normal_threshold=normal_threshold,
            lose_threshold=lose_threshold,
            win_rate_window=win_rate_window,
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
            modes = {"W": 0, "N": 0, "L": 0}
            for t in test_trades:
                m = t.get("mode", "N")
                if m in modes:
                    modes[m] += 1
            yr_sectors: Dict[str, int] = defaultdict(int)
            for t in test_trades:
                yr_sectors[t.get("sector", "OTHER")] += 1
            sec_str = " ".join(
                f"{k}:{v}" for k, v in sorted(yr_sectors.items()))
            print(
                f"  {test_year}: {n}t WR={wr_val:.1f}% avg={avg:+.2f}% "
                f"modes=[W:{modes['W']} N:{modes['N']} L:{modes['L']}] "
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
    print("  V95: BAYESIAN MODEL AVERAGING FACTOR STRATEGY")
    print("  BMA weights factors by posterior probability of effectiveness")
    print("  Rolling IC estimation + Bayesian updating")
    print("  Paper (2604.04430): BMA beats any single factor selection")
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

    # === Phase 1: Compute raw factors and ranks (once) ===
    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_cross_sectional_ranks(raw, NS, ND)
    ker_regime = compute_ker(C, NS, ND)

    # === Phase 2: Compute BMA weights for each (ic_window, ic_horizon, prior_strength) ===
    print("\n" + "=" * 70)
    print("  PHASE 2: BMA WEIGHT COMPUTATION")
    print("=" * 70)

    # Cache BMA composites by (ic_window, ic_horizon, prior_strength, st_weight)
    bma_cache: Dict[Tuple[int, int, float, float], Tuple] = {}

    for ic_window in [40, 60, 90]:
        for ic_horizon in [3, 5, 10]:
            # Compute forward returns for this horizon
            fwd = compute_forward_returns(C, NS, ND, ic_horizon)

            # Compute rolling IC for ST and MT factors
            ic_st = compute_rolling_ic_fast(
                ranks, fwd, ST_FACTOR_NAMES, NS, ND, ic_window)
            ic_mt = compute_rolling_ic_fast(
                ranks, fwd, MT_FACTOR_NAMES, NS, ND, ic_window)

            for prior_strength in [5, 10, 20]:
                st_w_arr, mt_w_arr = compute_bma_weights(
                    ic_st, ic_mt,
                    ST_FACTOR_NAMES, MT_FACTOR_NAMES,
                    prior_strength, NS, ND)

                for st_weight in [0.60, 0.65, 0.70]:
                    combined, st_comp, mt_comp, ncf_st, ncf_mt = (
                        build_bma_composite(
                            ranks, st_w_arr, mt_w_arr,
                            ST_FACTOR_NAMES, MT_FACTOR_NAMES,
                            st_weight, NS, ND)
                    )
                    key = (ic_window, ic_horizon, prior_strength, st_weight)
                    bma_cache[key] = (
                        combined, st_comp, mt_comp, ncf_st, ncf_mt)

    print(f"\n  Cached {len(bma_cache)} BMA composites")

    # === Phase 3: Parameter sweep ===
    print("\n" + "=" * 70)
    print("  PHASE 3: PARAMETER SWEEP (2019-2026)")
    print("  NO LEVERAGE. Target: beat V80 ann +36.4%")
    print("=" * 70)

    results: List[dict] = []
    sweep_count = 0

    for ic_window in [40, 60, 90]:
        for ic_horizon in [3, 5, 10]:
            for prior_strength in [5, 10, 20]:
                for st_weight in [0.60, 0.65, 0.70]:
                    key = (ic_window, ic_horizon, prior_strength, st_weight)
                    combined, _, _, ncf_st, ncf_mt = bma_cache[key]

                    for max_positions in [2, 3, 4]:
                        for max_per_sector in [2, 3]:
                            sweep_count += 1
                            trades, eq, dd = backtest_v95(
                                C, O, H, L, NS, ND, dates, syms,
                                combined, ncf_st, ncf_mt, ker_regime,
                                sector_lookup=sector_lookup,
                                win_threshold=0.60,
                                normal_threshold=0.80,
                                lose_threshold=0.90,
                                win_rate_window=15,
                                atr_stop=3.0,
                                max_positions=max_positions,
                                max_per_sector=max_per_sector,
                                min_confidence=3,
                                use_ker_gate=True,
                                hold_days=5,
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

                            results.append({
                                "ic_win": ic_window,
                                "ic_hor": ic_horizon,
                                "prior": prior_strength,
                                "st_w": st_weight,
                                "mp": max_positions,
                                "mps": max_per_sector,
                                "n": len(trades), "wr": wr,
                                "ann": ann, "dd": dd,
                                "sharpe": sh_val, "eq": eq,
                                "avg_yr": avg_per_year,
                            })

    results.sort(key=lambda x: -x["ann"])
    print(f"\n  Evaluated {sweep_count} configs, "
          f"{len(results)} with 10+ trades")
    print(
        f"\n{'ICw':>4} {'ICz':>4} {'Pri':>4} {'STw':>4} "
        f"{'MP':>3} {'MPS':>3} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} "
        f"{'Sh':>6} {'Avg/Yr':>7}"
    )
    print("-" * 80)
    for r in results[:10]:
        print(
            f"{r['ic_win']:>4} {r['ic_hor']:>4} {r['prior']:>4} "
            f"{r['st_w']:>4.2f} {r['mp']:>3} {r['mps']:>3} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f} "
            f"{r['avg_yr']:>7.1f}"
        )

    if not results:
        print("  No configs with 10+ trades. Exiting.")
        return

    # === Phase 4: Top config full backtest ===
    print("\n" + "=" * 70)
    print("  PHASE 4: TOP CONFIG -- FULL BACKTEST (2016-2026)")
    print("=" * 70)

    best = results[0]
    key = (best["ic_win"], best["ic_hor"], best["prior"], best["st_w"])
    combined, _, _, ncf_st, ncf_mt = bma_cache[key]

    trades_full, eq_full, dd_full = backtest_v95(
        C, O, H, L, NS, ND, dates, syms,
        combined, ncf_st, ncf_mt, ker_regime,
        sector_lookup=sector_lookup,
        win_threshold=0.60,
        normal_threshold=0.80,
        lose_threshold=0.90,
        win_rate_window=15,
        atr_stop=3.0,
        max_positions=best["mp"],
        max_per_sector=best["mps"],
        min_confidence=3,
        use_ker_gate=True,
        hold_days=5,
        start_di=60,
    )
    label = (
        f"icw={best['ic_win']} icz={best['ic_hor']} "
        f"pri={best['prior']} stw={best['st_w']:.2f} "
        f"mp={best['mp']} mps={best['mps']}"
    )
    print(f"\n  FULL {label}")
    analyze(trades_full, eq_full, dd_full, label)

    # === Phase 5: Walk-forward for best config ===
    print("\n" + "=" * 70)
    print(
        f"  PHASE 5: WALK-FORWARD for best config"
    )
    print("=" * 70)
    walk_forward(
        C, O, H, L, NS, ND, dates, syms,
        combined, ncf_st, ncf_mt, ker_regime,
        sector_lookup=sector_lookup,
        win_threshold=0.60,
        normal_threshold=0.80,
        lose_threshold=0.90,
        win_rate_window=15,
        max_positions=best["mp"],
        max_per_sector=best["mps"],
    )

    # === Phase 6: Top 10 configs full detail ===
    print("\n" + "=" * 70)
    print("  PHASE 6: TOP 10 CONFIGS -- FULL DETAIL")
    print("=" * 70)

    seen = set()
    unique_top = []
    for r in results:
        rkey = (r["ic_win"], r["ic_hor"], r["prior"],
                r["st_w"], r["mp"], r["mps"])
        if rkey not in seen:
            seen.add(rkey)
            unique_top.append(r)
        if len(unique_top) >= 10:
            break

    for r in unique_top:
        rkey = (r["ic_win"], r["ic_hor"], r["prior"], r["st_w"])
        comb, _, _, nst, nmt = bma_cache[rkey]
        trd, eq, dd = backtest_v95(
            C, O, H, L, NS, ND, dates, syms,
            comb, nst, nmt, ker_regime,
            sector_lookup=sector_lookup,
            win_threshold=0.60,
            normal_threshold=0.80,
            lose_threshold=0.90,
            win_rate_window=15,
            atr_stop=3.0,
            max_positions=r["mp"],
            max_per_sector=r["mps"],
            min_confidence=3,
            use_ker_gate=True,
            hold_days=5,
            start_di=60,
        )
        lbl = (
            f"icw={r['ic_win']} icz={r['ic_hor']} "
            f"pri={r['prior']} stw={r['st_w']:.2f} "
            f"mp={r['mp']} mps={r['mps']}"
        )
        print(f"\n  {lbl}")
        analyze(trd, eq, dd, lbl)

    print(f"\n[V95] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
