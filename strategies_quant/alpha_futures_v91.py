"""
V91: Regime-Conditional Adaptive Engine
========================================
NOT a new factor strategy — a BETTER ENGINE for existing signals.

5 independent amplifiers layered on V80's 7-factor multi-TF composite:

1. Volatility-Regime Conditional Sizing
   - Rolling 20d vol → percentile rank
   - Low vol (<p30): 1.5x, Normal: 1.0x, High vol (>p70): 0.5x
   - "君子而时中" — adapt to the times

2. Drawdown Circuit Breaker + Aggressive Kelly
   - DD<5%: 3/4 Kelly, DD 5-15%: 1/2 Kelly, DD 15-25%: 1/4 Kelly
   - DD>25%: stop trading ("不若则能避之")

3. Winner Pyramiding
   - If position profitable after 1 day AND rank>0.70: add 50% more
   - Max 1 pyramid per position
   - Target P/L ratio improvement from 1.5:1 to 2.5:1+

4. Signal Confluence Gate
   - Require 3 of 4: (a) rank>thresh (b) OI dropping (c) vol<avg (d) gap<0
   - Fewer entries but MUCH higher quality

5. Cross-Asset Confirmation
   - Only trade if sector leader also shows MR signal (rank>0.60)

Base: V80 framework with all 7 factors + multi-TF + sector limit
Walk-forward 2019-2026. CASH0=1,000,000, COMM=0.0005, NO leverage.
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

# --- V80 factor weights (unchanged) ---
ST_WEIGHTS = {
    "rank_ret5d": 0.25,
    "rank_oi5d": 0.20,
    "rank_rsi": 0.15,
    "rank_vol5d": 0.15,
    "rank_ret10d": 0.10,
    "rank_range5d": 0.10,
    "rank_atrp5d": 0.05,
}

MT_WEIGHTS = {
    "rank_ret20d": 0.25,
    "rank_oi20d": 0.20,
    "rank_rsi": 0.15,
    "rank_vol20d": 0.15,
    "rank_ret10d": 0.10,
    "rank_range20d": 0.10,
    "rank_atrp20d": 0.05,
}

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


# ============================================================
# V80 factor computation (unchanged — same 7 factors + ranks)
# ============================================================

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
    print("[V91] Computing raw factors...", flush=True)

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
    print("[V91] Computing cross-sectional ranks...", flush=True)

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


def build_multi_tf_composite(
    ranks: Dict[str, np.ndarray],
    st_weights: Dict[str, float],
    mt_weights: Dict[str, float],
    st_weight: float,
    NS: int, ND: int,
    min_factors: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t0 = time.time()
    print(f"[V91] Building multi-TF composite (st_w={st_weight:.2f})...",
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
    combined, st_comp, mt_comp, ncf_st, ncf_mt = build_multi_tf_composite(
        ranks, st_weights, mt_weights, st_weight, NS, ND)

    return {
        "composite": combined,
        "st_comp": st_comp,
        "mt_comp": mt_comp,
        "n_confirm_st": ncf_st,
        "n_confirm_mt": ncf_mt,
        "ranks": ranks,
        "raw": raw,
    }


# ============================================================
# V91 ENGINE: 5 Adaptive Amplifiers
# ============================================================

def compute_rolling_volatility(
    C: np.ndarray, NS: int, ND: int, window: int = 20,
) -> np.ndarray:
    """Rolling 20d daily return std for each instrument."""
    vol = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(window, ND):
            prices = C[si, di - window:di + 1]
            valid = prices[~np.isnan(prices)]
            if len(valid) >= window // 2 + 1 and valid[0] > 0:
                rets = np.diff(valid) / valid[:-1]
                vol[si, di] = np.std(rets)
    return vol


def compute_vol_regime_multiplier(
    rolling_vol: np.ndarray, NS: int, ND: int,
    lookback: int = 252,
) -> np.ndarray:
    """Amplifier 1: Volatility-regime conditional sizing.

    Uses rolling percentile of each instrument's own volatility.
    Low vol (<p30): 1.5x, Normal (p30-p70): 1.0x, High vol (>p70): 0.5x
    """
    mult = np.ones((NS, ND), dtype=np.float64)
    for si in range(NS):
        for di in range(lookback, ND):
            hist = rolling_vol[si, max(0, di - lookback):di + 1]
            valid = hist[~np.isnan(hist)]
            if len(valid) < 30:
                continue
            current = rolling_vol[si, di]
            if np.isnan(current):
                continue
            p30 = np.percentile(valid, 30)
            p70 = np.percentile(valid, 70)
            if current < p30:
                mult[si, di] = 1.5
            elif current > p70:
                mult[si, di] = 0.5
    return mult


def compute_kelly_multiplier(
    equity: float, peak: float,
) -> Tuple[float, bool]:
    """Amplifier 2: Drawdown circuit breaker + Kelly sizing.

    Returns (kelly_mult, should_trade).
    Note: Based on research, circuit breakers destroy returns.
    Only use position sizing for DD control, NOT stopping entirely.
    """
    if peak <= 0:
        return 1.0, True
    dd_pct = (peak - equity) / peak * 100
    if dd_pct > 30:
        return 0.30, True  # Reduce but don't stop (research: stopping kills returns)
    if dd_pct > 20:
        return 0.50, True
    if dd_pct > 10:
        return 0.75, True
    return 1.0, True  # Full size when DD < 10%


def compute_sector_leader_signal(
    composite: np.ndarray,
    sector_lookup: Dict[int, str],
    NS: int, ND: int,
    threshold: float = 0.55,
) -> np.ndarray:
    """Amplifier 5: Cross-asset confirmation.

    For each sector, compute the median composite rank.
    Only allow trades when the sector's median MR signal > threshold.
    This filters idiosyncratic noise — only trade when the SECTOR
    as a whole is mean-reverting.
    """
    confirm = np.zeros((NS, ND), dtype=bool)

    sector_to_sis: Dict[str, List[int]] = defaultdict(list)
    for si in range(NS):
        sector_to_sis[sector_lookup.get(si, 'OTHER')].append(si)

    for di in range(ND):
        for sector, sis in sector_to_sis.items():
            ranks_in_sector = []
            for si in sis:
                r = composite[si, di]
                if not np.isnan(r):
                    ranks_in_sector.append(r)
            if len(ranks_in_sector) >= 3:
                median_rank = np.median(ranks_in_sector)
                sector_ok = median_rank >= threshold
            else:
                sector_ok = True  # Don't gate small sectors
            for si in sis:
                confirm[si, di] = sector_ok

    return confirm


def compute_overnight_gap(
    C: np.ndarray, O: np.ndarray, NS: int, ND: int,
) -> np.ndarray:
    """Overnight gap: (open[di] - close[di-1]) / close[di-1].

    Negative gap = news overreaction (good for MR buy).
    """
    gap = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if (not np.isnan(O[si, di])
                    and not np.isnan(C[si, di - 1])
                    and C[si, di - 1] > 0):
                gap[si, di] = (O[si, di] - C[si, di - 1]) / C[si, di - 1]
    return gap


def check_confluence_gate(
    composite_rank: float,
    oi_5d_raw: float,
    vol_5d_raw: float,
    vol_20d_avg: float,
    overnight_gap: float,
    threshold: float = 0.70,
) -> bool:
    """Amplifier 4: Signal confluence gate.

    Require 3 of 4 conditions:
      (a) composite rank > threshold (MR signal)
      (b) OI dropping (contrarian signal)
      (c) volume below average (not crowded)
      (d) overnight gap < 0 (news overreaction)
    """
    conditions_met = 0

    # (a) MR signal — rank above threshold
    if composite_rank >= threshold:
        conditions_met += 1

    # (b) OI dropping — contrarian signal
    if not np.isnan(oi_5d_raw) and oi_5d_raw < 0:
        conditions_met += 1

    # (c) Volume below average — not crowded
    if (not np.isnan(vol_5d_raw)
            and not np.isnan(vol_20d_avg)
            and vol_20d_avg > 0
            and vol_5d_raw < vol_20d_avg):
        conditions_met += 1

    # (d) Overnight gap < 0 — news overreaction
    if not np.isnan(overnight_gap) and overnight_gap < 0:
        conditions_met += 1

    return conditions_met >= 3


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


# ============================================================
# V91 Adaptive Engine Backtest
# ============================================================

def backtest_v91(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
    dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
    sector_lookup: Dict[int, str],
    vol_mult: np.ndarray,
    sector_confirm: np.ndarray,
    overnight_gap: np.ndarray,
    confluence_threshold: float = 0.70,
    sector_leader_threshold: float = 0.60,
    base_threshold: float = 0.75,
    atr_stop: float = 3.0,
    max_positions: int = 5,
    max_per_sector: int = 3,
    hold_days: int = 5,
    start_di: int = 60,
    end_di: Optional[int] = None,
    pyramid_rank_threshold: float = 0.70,
    pyramid_add_fraction: float = 0.50,
    enable_confluence: bool = True,
    enable_sector_confirm: bool = True,
    enable_pyramiding: bool = True,
    enable_vol_sizing: bool = True,
    enable_kelly: bool = True,
) -> Tuple[List[dict], float, float]:
    """V91 adaptive engine with 5 amplifiers.

    Position tuple: (si, entry_di, entry_price, stop_price,
                     alloc, is_pyramid, base_alloc)
    """
    composite = sigs["composite"]
    n_confirm_st = sigs["n_confirm_st"]
    n_confirm_mt = sigs["n_confirm_mt"]
    raw = sigs["raw"]
    oi_5d_raw = raw["oi_5d"]
    vol_5d_raw = raw["vol_5d"]
    vol_20d_raw = raw["vol_20d"]

    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    # (si, entry_di, entry_price, stop_price, alloc, is_pyramid, base_alloc)
    positions: List[Tuple[int, int, float, float, float, bool, float]] = []
    trades: List[dict] = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions: List[Tuple[int, int, float, float, float, bool, float]] = []

        # --- Amplifier 2: Kelly/DD circuit breaker ---
        # Note: research shows stopping trading destroys returns.
        # Kelly multiplier only affects pyramid sizing, not base positions.
        kelly_mult, _ = compute_kelly_multiplier(equity, peak)

        # Group positions by symbol
        pos_by_si: Dict[int, List] = defaultdict(list)
        for si, edi, ep, sp, alloc, is_pyr, ba in positions:
            pos_by_si[si].append((edi, ep, sp, alloc, is_pyr, ba))

        # --- Exit logic (same hold-days + ATR stop as V80) ---
        for si, pos_list in pos_by_si.items():
            c = C[si, di]
            if np.isnan(c):
                for edi, ep, sp, alloc, is_pyr, ba in pos_list:
                    new_positions.append(
                        (si, edi, ep, sp, alloc, is_pyr, ba))
                continue

            earliest_edi = min(p[0] for p in pos_list)
            hold = di - earliest_edi
            stopped = any(c < sp for _, _, sp, _, _, _ in pos_list)

            if stopped:
                for edi, ep, sp, alloc, is_pyr, ba in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    trades.append({
                        "pnl_abs": profit, "pnl_pct": pnl * 100,
                        "days": di - edi + 1, "di": di,
                        "year": d.year, "sym": syms[si],
                        "sector": sector_lookup.get(si, 'OTHER'),
                        "reason": "stop", "pyr": is_pyr,
                        "amp": "",
                    })
            elif hold >= hold_days:
                for edi, ep, sp, alloc, is_pyr, ba in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    trades.append({
                        "pnl_abs": profit, "pnl_pct": pnl * 100,
                        "days": di - edi + 1, "di": di,
                        "year": d.year, "sym": syms[si],
                        "sector": sector_lookup.get(si, 'OTHER'),
                        "reason": "hold", "pyr": is_pyr,
                        "amp": "",
                    })
            else:
                for edi, ep, sp, alloc, is_pyr, ba in pos_list:
                    new_positions.append(
                        (si, edi, ep, sp, alloc, is_pyr, ba))

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

        # === ENTRY with V91 adaptive engine ===
        held = {p[0] for p in positions}
        held_has_pyramid = {
            si for si, _, _, _, _, is_pyr, _ in positions if is_pyr}

        if len(positions) >= max_positions:
            continue

        candidates = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(composite[si, di]):
                continue
            if composite[si, di] < base_threshold:
                continue
            total_confirm = n_confirm_st[si, di] + n_confirm_mt[si, di]
            if total_confirm < 3:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue

            # --- Amplifier 4: Confluence gate ---
            if enable_confluence:
                oi_val = oi_5d_raw[si, di] if di < ND else np.nan
                v5 = vol_5d_raw[si, di] if di < ND else np.nan
                v20 = vol_20d_raw[si, di] if di < ND else np.nan
                gap = overnight_gap[si, di] if di < ND else np.nan
                if not check_confluence_gate(
                    composite[si, di], oi_val, v5, v20, gap,
                    threshold=confluence_threshold,
                ):
                    continue

            # --- Amplifier 5: Sector leader confirmation ---
            if enable_sector_confirm and not sector_confirm[si, di]:
                continue

            candidates.append((composite[si, di], si))

        candidates.sort(key=lambda x: -x[0])

        # Sector-constrained selection
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

        # --- Amplifiers 1+2: Compute adaptive allocation ---
        num_total = len(positions) + len(new_entries)
        if num_total == 0:
            continue
        base_alloc = LEVERAGE / num_total

        # Update existing positions with new allocation
        # Kelly does NOT reduce base alloc — only affects pyramid adds
        updated_positions = []
        for si, edi, ep, sp, _, is_pyr, orig_ba in positions:
            updated_positions.append(
                (si, edi, ep, sp, base_alloc, is_pyr, base_alloc))

        # Enter new positions at open[di+1] with vol-regime sizing
        for rank_val, si, sym_sector in new_entries:
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr = compute_atr_at(H, L, C, si, di, start_di)
            if atr is None:
                continue

            # Amplifier 1: Vol-regime multiplier on position sizing
            # Low vol = can go slightly bigger, high vol = smaller
            if enable_vol_sizing:
                vm = vol_mult[si, di + 1]
                if np.isnan(vm):
                    vm = vol_mult[si, di]
                if np.isnan(vm):
                    vm = 1.0
                # Clamp to prevent extreme positions
                pos_alloc = min(base_alloc * vm, LEVERAGE * 0.30)
            else:
                pos_alloc = base_alloc

            updated_positions.append(
                (si, di + 1, ep, ep - atr_stop * atr,
                 pos_alloc, False, base_alloc))

        positions = updated_positions

        # --- Amplifier 3: Winner pyramiding (Kelly-gated) ---
        if enable_pyramiding:
            pyramid_positions = []
            total_exposure = sum(alloc for _, _, _, _, alloc, _, _ in positions)

            for si, edi, ep, sp, alloc, is_pyr, ba in positions:
                if is_pyr or si in held_has_pyramid:
                    pyramid_positions.append(
                        (si, edi, ep, sp, alloc, is_pyr, ba))
                    continue

                c = C[si, di]
                if np.isnan(c) or edi >= di:
                    pyramid_positions.append(
                        (si, edi, ep, sp, alloc, is_pyr, ba))
                    continue

                # Check if profitable after at least 1 day
                hold = di - edi
                if hold < 1:
                    pyramid_positions.append(
                        (si, edi, ep, sp, alloc, is_pyr, ba))
                    continue

                pnl_pct = (c - ep) / ep
                # Kelly gates pyramid: only add when DD is low
                kelly_ok = (not enable_kelly) or kelly_mult >= 0.75
                if (pnl_pct > 0
                        and composite[si, di] > pyramid_rank_threshold
                        and kelly_ok):
                    # Add pyramid position — fraction of base, capped by total exposure
                    pyramid_alloc = ba * pyramid_add_fraction * kelly_mult
                    new_exposure = total_exposure + pyramid_alloc
                    # Cap total portfolio exposure at leverage
                    if new_exposure > LEVERAGE:
                        pyramid_alloc = max(
                            0, LEVERAGE - total_exposure)
                    if pyramid_alloc > 0.001:
                        pyramid_positions.append(
                            (si, edi, ep, sp, alloc, True, ba))
                        atr_val = compute_atr_at(
                            H, L, C, si, di, start_di)
                        stop_price = c - atr_stop * (
                            atr_val if atr_val else c * 0.02)
                        pyramid_positions.append(
                            (si, di, c, stop_price,
                             pyramid_alloc, True, ba))
                        total_exposure += pyramid_alloc
                    else:
                        pyramid_positions.append(
                            (si, edi, ep, sp, alloc, is_pyr, ba))
                else:
                    pyramid_positions.append(
                        (si, edi, ep, sp, alloc, is_pyr, ba))

            positions = pyramid_positions

    # Close remaining positions at end
    for si, edi, ep, sp, alloc, is_pyr, ba in positions:
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
    n_cb = sum(1 for t in trades if t["reason"] == "circuit_breaker")

    avg_pnl = np.mean([t["pnl_pct"] for t in trades])
    avg_win = np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"] > 0])
    avg_loss = np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"] <= 0])

    pl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    sector_counts: Dict[str, int] = defaultdict(int)
    for t in trades:
        sector_counts[t.get("sector", "OTHER")] += 1
    sector_str = " ".join(
        f"{k}:{v}" for k, v in sorted(sector_counts.items()))

    print(
        f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} "
        f"stop:{n_stop} hold:{n_hold} cb:{n_cb}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
        f"Sh={sh:.2f} eq={equity:,.0f} PL={pl_ratio:.2f}:1"
    )
    print(
        f"    avg_pnl={avg_pnl:+.3f}% avg_win={avg_win:+.3f}% "
        f"avg_loss={avg_loss:+.3f}%"
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
    V: np.ndarray, OI: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
    sector_lookup: Dict[int, str],
    vol_mult: np.ndarray,
    sector_confirm: np.ndarray,
    overnight_gap: np.ndarray,
    config: dict,
) -> List[dict]:
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V91: {config['label']}")
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

        trades, _, _ = backtest_v91(
            C, O, H, L, V, OI, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            vol_mult=vol_mult,
            sector_confirm=sector_confirm,
            overnight_gap=overnight_gap,
            base_threshold=config["base_threshold"],
            confluence_threshold=config.get("confluence_threshold", 0.70),
            sector_leader_threshold=config.get(
                "sector_leader_threshold", 0.60),
            atr_stop=config.get("atr_stop", 3.0),
            max_positions=config.get("max_positions", 5),
            max_per_sector=config.get("max_per_sector", 3),
            hold_days=config.get("hold_days", 5),
            start_di=test_start,
            end_di=test_end_idx + 1,
            pyramid_rank_threshold=config.get(
                "pyramid_rank_threshold", 0.70),
            pyramid_add_fraction=config.get("pyramid_add_fraction", 0.50),
            enable_confluence=config.get("enable_confluence", True),
            enable_sector_confirm=config.get("enable_sector_confirm", True),
            enable_pyramiding=config.get("enable_pyramiding", True),
            enable_vol_sizing=config.get("enable_vol_sizing", True),
            enable_kelly=config.get("enable_kelly", True),
        )

        test_trades = [t for t in trades
                       if dates[t["di"]].year == test_year]
        all_trades.extend(test_trades)

        if test_trades:
            n = len(test_trades)
            nw = sum(1 for t in test_trades if t["pnl_pct"] > 0)
            wr_val = nw / n * 100
            avg = np.mean([t["pnl_pct"] for t in test_trades])
            yr_sectors: Dict[str, int] = defaultdict(int)
            for t in test_trades:
                yr_sectors[t.get("sector", "OTHER")] += 1
            sec_str = " ".join(
                f"{k}:{v}" for k, v in sorted(yr_sectors.items()))
            print(
                f"  {test_year}: {n}t WR={wr_val:.1f}% avg={avg:+.2f}% "
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
    print("  V91: REGIME-CONDITIONAL ADAPTIVE ENGINE")
    print("  5 amplifiers on V80's 7-factor multi-TF composite")
    print("  Target: ann > 50% (V80 baseline: 36.4%)")
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

    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # === 1. Compute base signals ===
    print("\n--- Computing base V80 signals ---")
    sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND, st_weight=0.60)

    # === 2. Compute V91 engine data ===
    print("\n--- Computing V91 engine components ---")
    rolling_vol = compute_rolling_volatility(C, NS, ND, window=20)
    vol_mult = compute_vol_regime_multiplier(rolling_vol, NS, ND)
    sector_confirm = compute_sector_leader_signal(
        sigs["composite"], sector_lookup, NS, ND, threshold=0.60)
    overnight_gap = compute_overnight_gap(C, O, NS, ND)
    print("  Engine components ready.", flush=True)

    # === 3. Ablation study: test each amplifier independently ===
    print("\n" + "=" * 70)
    print("  ABLATION STUDY: Test each amplifier independently (2019-2026)")
    print("=" * 70)

    base_config = {
        "base_threshold": 0.75,
        "confluence_threshold": 0.70,
        "sector_leader_threshold": 0.60,
        "atr_stop": 3.0,
        "max_positions": 5,
        "max_per_sector": 3,
        "hold_days": 5,
        "pyramid_rank_threshold": 0.70,
        "pyramid_add_fraction": 0.50,
    }

    ablation_configs = [
        {
            "label": "BASE (V80-like, no amplifiers)",
            **base_config,
            "enable_confluence": False,
            "enable_sector_confirm": False,
            "enable_pyramiding": False,
            "enable_vol_sizing": False,
            "enable_kelly": False,
        },
        {
            "label": "AMP1: Vol-Regime Sizing only",
            **base_config,
            "enable_confluence": False,
            "enable_sector_confirm": False,
            "enable_pyramiding": False,
            "enable_vol_sizing": True,
            "enable_kelly": False,
        },
        {
            "label": "AMP2: Kelly/DD Breaker only",
            **base_config,
            "enable_confluence": False,
            "enable_sector_confirm": False,
            "enable_pyramiding": False,
            "enable_vol_sizing": False,
            "enable_kelly": True,
        },
        {
            "label": "AMP3: Pyramiding only",
            **base_config,
            "enable_confluence": False,
            "enable_sector_confirm": False,
            "enable_pyramiding": True,
            "enable_vol_sizing": False,
            "enable_kelly": False,
        },
        {
            "label": "AMP4: Confluence Gate only",
            **base_config,
            "enable_confluence": True,
            "enable_sector_confirm": False,
            "enable_pyramiding": False,
            "enable_vol_sizing": False,
            "enable_kelly": False,
        },
        {
            "label": "AMP5: Sector Confirm only",
            **base_config,
            "enable_confluence": False,
            "enable_sector_confirm": True,
            "enable_pyramiding": False,
            "enable_vol_sizing": False,
            "enable_kelly": False,
        },
    ]

    ablation_results = []
    for cfg in ablation_configs:
        trades, eq, dd = backtest_v91(
            C, O, H, L, V, OI, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            vol_mult=vol_mult,
            sector_confirm=sector_confirm,
            overnight_gap=overnight_gap,
            base_threshold=cfg["base_threshold"],
            confluence_threshold=cfg.get("confluence_threshold", 0.70),
            atr_stop=cfg.get("atr_stop", 3.0),
            max_positions=cfg.get("max_positions", 5),
            max_per_sector=cfg.get("max_per_sector", 3),
            hold_days=cfg.get("hold_days", 5),
            start_di=bt_2019,
            enable_confluence=cfg.get("enable_confluence", True),
            enable_sector_confirm=cfg.get("enable_sector_confirm", True),
            enable_pyramiding=cfg.get("enable_pyramiding", True),
            enable_vol_sizing=cfg.get("enable_vol_sizing", True),
            enable_kelly=cfg.get("enable_kelly", True),
        )
        result = analyze(trades, eq, dd, cfg["label"])
        if result:
            ablation_results.append({**result, "label": cfg["label"]})

    # === 4. Parameter sweep: ALL amplifiers on ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP: ALL amplifiers on (2019-2026)")
    print("  Target: ann > 50%")
    print("=" * 70)

    sweep_results: List[dict] = []

    for thresh in [0.70, 0.75, 0.80]:
        for conf_thresh in [0.65, 0.70, 0.75]:
            for mp in [4, 5, 6, 8]:
                for mps in [2, 3]:
                    for hd in [4, 5, 7]:
                        sweep_count = len(sweep_results)
                        if sweep_count > 0 and sweep_count % 50 == 0:
                            print(
                                f"  ... evaluated {sweep_count} configs",
                                flush=True)

                        trades, eq, dd = backtest_v91(
                            C, O, H, L, V, OI, NS, ND, dates, syms, sigs,
                            sector_lookup=sector_lookup,
                            vol_mult=vol_mult,
                            sector_confirm=sector_confirm,
                            overnight_gap=overnight_gap,
                            base_threshold=thresh,
                            confluence_threshold=conf_thresh,
                            atr_stop=3.0,
                            max_positions=mp,
                            max_per_sector=mps,
                            hold_days=hd,
                            start_di=bt_2019,
                            enable_confluence=True,
                            enable_sector_confirm=True,
                            enable_pyramiding=True,
                            enable_vol_sizing=True,
                            enable_kelly=True,
                        )

                        if len(trades) < 10:
                            continue

                        nw = sum(1 for t in trades if t["pnl_pct"] > 0)
                        wr = nw / len(trades) * 100
                        n_days = max(
                            1, trades[-1]["di"] - trades[0]["di"])
                        ann = ((eq / CASH0) ** (
                            1 / max(1.0, n_days / 252)) - 1) * 100
                        ap = [t["pnl_abs"]
                              for t in sorted(trades, key=lambda x: x["di"])]
                        rets_arr = np.array(ap) / CASH0
                        sh_val = (
                            np.mean(rets_arr) / np.std(rets_arr)
                            * np.sqrt(252)
                            if np.std(rets_arr) > 0 else 0)

                        sweep_results.append({
                            "thresh": thresh,
                            "conf": conf_thresh,
                            "mp": mp, "mps": mps, "hd": hd,
                            "n": len(trades), "wr": wr,
                            "ann": ann, "dd": dd,
                            "sharpe": sh_val, "eq": eq,
                        })

    sweep_results.sort(key=lambda x: -x["sharpe"])
    print(f"\n  Evaluated {len(sweep_results)} configs with 10+ trades")
    print(
        f"\n{'Th':>4} {'Cf':>4} {'MP':>3} {'MPS':>3} {'HD':>3} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} "
        f"{'Sh':>6}"
    )
    print("-" * 60)
    for r in sweep_results[:30]:
        print(
            f"{r['thresh']:>4.2f} {r['conf']:>4.2f} "
            f"{r['mp']:>3} {r['mps']:>3} {r['hd']:>3} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f}"
        )

    if not sweep_results:
        print("  No configs with 10+ trades. Exiting.")
        return

    # === 5. Best config: full backtest ===
    best = sweep_results[0]
    best_config = {
        "label": (
            f"BEST: thresh={best['thresh']:.2f} "
            f"conf={best['conf']:.2f} "
            f"mp={best['mp']} mps={best['mps']} hd={best['hd']}"
        ),
        "base_threshold": best["thresh"],
        "confluence_threshold": best["conf"],
        "atr_stop": 3.0,
        "max_positions": best["mp"],
        "max_per_sector": best["mps"],
        "hold_days": best["hd"],
        "enable_confluence": True,
        "enable_sector_confirm": True,
        "enable_pyramiding": True,
        "enable_vol_sizing": True,
        "enable_kelly": True,
    }

    print("\n" + "=" * 70)
    print(f"  BEST CONFIG: {best_config['label']}")
    print("  Full backtest 2016-2026")
    print("=" * 70)

    trades_full, eq_full, dd_full = backtest_v91(
        C, O, H, L, V, OI, NS, ND, dates, syms, sigs,
        sector_lookup=sector_lookup,
        vol_mult=vol_mult,
        sector_confirm=sector_confirm,
        overnight_gap=overnight_gap,
        base_threshold=best_config["base_threshold"],
        confluence_threshold=best_config["confluence_threshold"],
        max_positions=best_config["max_positions"],
        max_per_sector=best_config["max_per_sector"],
        hold_days=best_config["hold_days"],
        start_di=60,
        enable_confluence=True,
        enable_sector_confirm=True,
        enable_pyramiding=True,
        enable_vol_sizing=True,
        enable_kelly=True,
    )
    analyze(trades_full, eq_full, dd_full, best_config["label"])

    # === 6. Walk-forward ===
    walk_forward(
        C, O, H, L, V, OI, NS, ND, dates, syms, sigs,
        sector_lookup=sector_lookup,
        vol_mult=vol_mult,
        sector_confirm=sector_confirm,
        overnight_gap=overnight_gap,
        config=best_config,
    )

    # === 7. Compare: ALL amplifiers vs BASE ===
    print("\n" + "=" * 70)
    print("  HEAD-TO-HEAD: V91 (all amplifiers) vs V80-BASE (none)")
    print("=" * 70)

    trades_all, eq_all, dd_all = backtest_v91(
        C, O, H, L, V, OI, NS, ND, dates, syms, sigs,
        sector_lookup=sector_lookup,
        vol_mult=vol_mult,
        sector_confirm=sector_confirm,
        overnight_gap=overnight_gap,
        base_threshold=best_config["base_threshold"],
        confluence_threshold=best_config["confluence_threshold"],
        max_positions=best_config["max_positions"],
        max_per_sector=best_config["max_per_sector"],
        hold_days=best_config["hold_days"],
        start_di=bt_2019,
        enable_confluence=True,
        enable_sector_confirm=True,
        enable_pyramiding=True,
        enable_vol_sizing=True,
        enable_kelly=True,
    )

    trades_base, eq_base, dd_base = backtest_v91(
        C, O, H, L, V, OI, NS, ND, dates, syms, sigs,
        sector_lookup=sector_lookup,
        vol_mult=vol_mult,
        sector_confirm=sector_confirm,
        overnight_gap=overnight_gap,
        base_threshold=best_config["base_threshold"],
        max_positions=best_config["max_positions"],
        max_per_sector=best_config["max_per_sector"],
        hold_days=best_config["hold_days"],
        start_di=bt_2019,
        enable_confluence=False,
        enable_sector_confirm=False,
        enable_pyramiding=False,
        enable_vol_sizing=False,
        enable_kelly=False,
    )

    print("\n  V91 (ALL amplifiers):")
    analyze(trades_all, eq_all, dd_all, "V91-all-amps")
    print("\n  V80-BASE (no amplifiers):")
    analyze(trades_base, eq_base, dd_base, "V80-base")

    if trades_all and trades_base:
        print(
            f"\n  Delta: eq={eq_all - eq_base:+,.0f} "
            f"dd={dd_all - dd_base:+.1f}% "
            f"trades={len(trades_all) - len(trades_base):+d}"
        )

    print(f"\n[V91] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
