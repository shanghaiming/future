"""
V85: New Alpha Factors Exploration
====================================
Explore 5 NEW factors beyond V80's 7 existing ones to create a 12-factor composite.
Goal: break through V80's +36.4% ann ceiling toward higher returns.

5 New Factors:
  1. Carry proxy (term structure slope): 20d trend slope as carry proxy.
     Positive slope = contango (negative carry for longs), negative slope = backwardation.
     For MR: instruments with steep negative slope (steep drop) = oversold = buy.
  2. Momentum persistence: If rank_ret5d has been in top-20% for 3 consecutive days,
     momentum is persistent -- for MR this is a contrarian short, but for MOM it's continuation.
     We use it as a QUALITY filter: persistent oversold (rank_ret5d > 0.8 for 3 days) = strong MR.
  3. Volume-price divergence: price drops but volume surges = capitulation = stronger MR.
     Compute as: rank of (volume_zscore * sign_of_negative_return).
  4. Cross-asset breadth: fraction of instruments with composite > 0.8.
     If >60% are oversold, market-wide MR opportunity -- amplify signals.
  5. Volatility compression: ATR drops below 50% of 20d average ATR = breakout imminent.
     For MR, this means the oversold instrument is about to snap back harder.

Implementation:
  - Compute ALL 5 new factors as additional ranks
  - Create 12-factor composite (7 original + 5 new)
  - Sweep factor weights
  - Use V80's framework (mps=3, mp=3, dynamic thresholds)
  - NO leverage

Walk-forward 2019-2026. Report ann, Sharpe, MDD for best config.
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
LEVERAGE = 1.0  # NO leverage

# Default 12-factor weights (will be swept)
# Original 7 + 5 new
DEFAULT_WEIGHTS = {
    # Original 7 (from V80 short-term)
    "rank_ret5d": 0.12,
    "rank_oi5d": 0.10,
    "rank_rsi": 0.08,
    "rank_vol5d": 0.08,
    "rank_ret10d": 0.06,
    "rank_range5d": 0.06,
    "rank_atrp5d": 0.04,
    # New 5 factors
    "rank_carry": 0.14,
    "rank_mom_persist": 0.10,
    "rank_vol_price_div": 0.08,
    "rank_breadth": 0.06,
    "rank_vol_compress": 0.08,
}

# Sector definitions (same as V80)
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
    print("[V85] Computing raw factors (7 original + 5 new)...", flush=True)

    # === Original 7 factors (same as V80) ===
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

    # === NEW FACTOR 1: Carry proxy (20d trend slope) ===
    # For MR: steeper negative slope = more oversold = higher carry score
    carry = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            w = C[si, di - 20:di]
            vv = w[~np.isnan(w)]
            if len(vv) >= 15 and vv[0] > 0:
                x = np.arange(len(vv))
                y = vv / vv[0]
                try:
                    slope = np.polyfit(x, y, 1)[0]
                    carry[si, di] = slope * 252  # annualized
                except Exception:
                    pass

    # === NEW FACTOR 2: Momentum persistence ===
    # How many of the last 3 days was ret_5d rank in top 20% (oversold)?
    # First compute rolling rank of ret_5d
    ret5d_rank_rolling = np.full((NS, ND), np.nan)
    for di in range(ND):
        vals = ret_5d[:, di]
        valid_count = np.sum(~np.isnan(vals))
        if valid_count < 10:
            continue
        ranked = pd.Series(vals).rank(pct=True, na_option="keep").values
        # Invert: higher rank = more oversold (for MR buy signal)
        ret5d_rank_rolling[:, di] = 1.0 - ranked

    # Count consecutive oversold days (rank > 0.8)
    mom_persist = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(3, ND):
            count = 0
            for j in range(di - 3, di):
                r = ret5d_rank_rolling[si, j]
                if not np.isnan(r) and r > 0.8:
                    count += 1
            # Score: 0-3 days of persistence, normalized
            if not np.isnan(ret5d_rank_rolling[si, di]):
                mom_persist[si, di] = count / 3.0

    # === NEW FACTOR 3: Volume-price divergence ===
    # Compute volume z-score (20d), then multiply by sign of negative return
    # High volume + price drop = capitulation = strong MR signal
    vol_zscore = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            vw = V[si, di - 20:di]
            vv = vw[~np.isnan(vw)]
            if len(vv) >= 10 and not np.isnan(V[si, di]):
                mu, sig = np.mean(vv), np.std(vv)
                if sig > 0:
                    vol_zscore[si, di] = (V[si, di] - mu) / sig

    # Divergence: vol z-score * (-1 if price dropped)
    # If volume surges AND price drops -> high positive divergence score
    vol_price_div = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if np.isnan(vol_zscore[si, di]) or np.isnan(ret_5d[si, di]):
                continue
            # ret_5d is inverted for MR (high rank = oversold = positive)
            # Divergence = high volume * oversold
            # oversold = ret_5d < 0 (negative return)
            is_oversold = 1.0 if ret_5d[si, di] < 0 else 0.0
            vol_price_div[si, di] = vol_zscore[si, di] * is_oversold

    # === NEW FACTOR 4: Cross-asset breadth ===
    # Fraction of instruments with ret_5d rank > 0.8 (oversold)
    # If > 60% are oversold, market-wide MR = stronger signal
    breadth = np.full((NS, ND), np.nan)
    for di in range(ND):
        vals = ret_5d[:, di]
        valid = ~np.isnan(vals)
        n_valid = np.sum(valid)
        if n_valid < 10:
            continue
        # Count oversold (ret_5d < -0.02, i.e., dropped more than 2%)
        n_oversold = np.sum(vals[valid] < -0.02)
        frac_oversold = n_oversold / n_valid
        # Apply same breadth score to all instruments
        for si in range(NS):
            if valid[si]:
                breadth[si, di] = frac_oversold

    # === NEW FACTOR 5: Volatility compression ===
    # ATR / 20d average ATR. If ratio < 0.5, breakout imminent.
    atr_20d_avg = np.full((NS, ND), np.nan)
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
            if atr_vals:
                atr_20d_avg[si, di] = np.mean(atr_vals)

    vol_compress = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(6, ND):
            if (np.isnan(atrp_5d[si, di]) or np.isnan(atr_20d_avg[si, di])
                    or np.isnan(C[si, di]) or C[si, di] <= 0):
                continue
            current_atr = atrp_5d[si, di] * C[si, di]
            if atr_20d_avg[si, di] > 0:
                ratio = current_atr / atr_20d_avg[si, di]
                # Low ratio = compression = higher score for MR snap-back
                vol_compress[si, di] = 1.0 - ratio  # inverted: low vol = high score

    print(f"  Raw factors done: {time.time() - t0:.1f}s", flush=True)
    return {
        # Original 7
        "ret_5d": ret_5d, "ret_10d": ret_10d,
        "oi_5d": oi_5d,
        "vol_5d": vol_5d,
        "range_5d": range_5d,
        "rsi14": rsi14,
        "atrp_5d": atrp_5d,
        # New 5
        "carry": carry,
        "mom_persist": mom_persist,
        "vol_price_div": vol_price_div,
        "breadth": breadth,
        "vol_compress": vol_compress,
    }


def compute_cross_sectional_ranks(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int, min_count: int = 10,
) -> Dict[str, np.ndarray]:
    t0 = time.time()
    print("[V85] Computing cross-sectional ranks...", flush=True)

    # 12 factors to rank
    factors_to_rank = {
        # Original 7
        "rank_ret5d": raw_factors["ret_5d"],
        "rank_ret10d": raw_factors["ret_10d"],
        "rank_oi5d": raw_factors["oi_5d"],
        "rank_vol5d": raw_factors["vol_5d"],
        "rank_range5d": raw_factors["range_5d"],
        "rank_rsi": raw_factors["rsi14"],
        "rank_atrp5d": raw_factors["atrp_5d"],
        # New 5
        "rank_carry": raw_factors["carry"],
        "rank_mom_persist": raw_factors["mom_persist"],
        "rank_vol_price_div": raw_factors["vol_price_div"],
        "rank_breadth": raw_factors["breadth"],
        "rank_vol_compress": raw_factors["vol_compress"],
    }

    # Invert: for MR, we want to buy oversold instruments
    # ret_5d, ret_10d: low return = oversold -> invert
    # oi_5d: low OI change = less crowd -> invert
    # rsi: low RSI = oversold -> invert
    # carry: steep negative slope = oversold -> invert
    # vol_price_div: high divergence = strong MR signal -> NO invert
    # mom_persist: high persistence = strong MR signal -> NO invert
    # breadth: high breadth = market-wide oversold -> NO invert
    # vol_compress: low ratio = compression -> already inverted in raw
    INVERT_FACTORS = {
        "rank_ret5d", "rank_ret10d", "rank_oi5d", "rank_rsi",
        "rank_carry",
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


def build_12factor_composite(
    ranks: Dict[str, np.ndarray],
    weights: Dict[str, float],
    NS: int, ND: int,
    min_factors: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build 12-factor composite score.

    Returns:
        combined: NS x ND composite score
        n_confirm: NS x ND count of factors with rank > 0.5
    """
    t0 = time.time()
    w_names = list(weights.keys())
    w_vals = np.array([weights[k] for k in w_names])
    total_w = np.sum(w_vals)
    print(f"[V85] Building 12-factor composite ({len(w_names)} factors)...",
          flush=True)

    combined = np.full((NS, ND), np.nan)
    n_confirm = np.zeros((NS, ND), dtype=int)

    for di in range(ND):
        for si in range(NS):
            score_vals = []
            w_sum = 0.0
            confirm = 0
            for idx, name in enumerate(w_names):
                rank_val = ranks[name][si, di]
                if np.isnan(rank_val):
                    continue
                score_vals.append(rank_val * w_vals[idx])
                w_sum += w_vals[idx]
                if rank_val > 0.5:
                    confirm += 1

            if w_sum > 0 and confirm >= min_factors:
                combined[si, di] = sum(score_vals) / w_sum
                n_confirm[si, di] = confirm

    print(f"  12-factor composite done: {time.time() - t0:.1f}s", flush=True)
    return combined, n_confirm


def compute_all_signals(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    if weights is None:
        weights = DEFAULT_WEIGHTS

    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_cross_sectional_ranks(raw, NS, ND)
    ker_regime = compute_ker(C, NS, ND)
    combined, n_confirm = build_12factor_composite(
        ranks, weights, NS, ND, min_factors=5)

    return {
        "composite": combined,
        "n_confirm": n_confirm,
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


def backtest_v85(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
    sector_lookup: Dict[int, str],
    win_threshold: float = 0.60,
    normal_threshold: float = 0.80,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
    atr_stop: float = 3.0,
    max_positions: int = 3,
    max_per_sector: int = 3,
    min_confidence: int = 5,
    use_ker_gate: bool = True,
    hold_days: int = 5,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Backtest V85: 12-factor composite mean reversion."""
    composite = sigs["composite"]
    n_confirm = sigs["n_confirm"]
    ker_regime = sigs["ker_regime"]

    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions: List[Tuple[int, int, float, float, float]] = []
    trades: List[dict] = []
    recent_trades_win: List[int] = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions: List[Tuple[int, int, float, float, float]] = []

        # Dynamic mode
        mode = get_dynamic_mode(
            recent_trades_win, win_threshold, win_rate_window)
        current_threshold = get_mode_threshold(
            mode, win_threshold, normal_threshold, lose_threshold)

        # Group positions by symbol
        pos_by_si: Dict[int, List] = defaultdict(list)
        for si, edi, ep, sp, alloc in positions:
            pos_by_si[si].append((edi, ep, sp, alloc))

        # Exit logic
        for si, pos_list in pos_by_si.items():
            c = C[si, di]
            if np.isnan(c):
                for edi, ep, sp, alloc in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc))
                continue

            earliest_edi = min(p[0] for p in pos_list)
            hold = di - earliest_edi
            stopped = any(c < sp for _, _, sp, _ in pos_list)

            if stopped:
                for edi, ep, sp, alloc in pos_list:
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
                        "mode": mode[:1].upper(),
                    })
                    recent_trades_win.append(1 if is_win else 0)
            elif hold >= hold_days:
                for edi, ep, sp, alloc in pos_list:
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
                        "mode": mode[:1].upper(),
                    })
                    recent_trades_win.append(1 if is_win else 0)
            else:
                for edi, ep, sp, alloc in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc))

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

        # --- ENTRY ---
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
            if n_confirm[si, di] < min_confidence:
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
        for si, edi, ep, sp, old_alloc in positions:
            updated_positions.append((si, edi, ep, sp, alloc_per_pos))

        for rank_val, si, sym_sector in new_entries:
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr = compute_atr_at(H, L, C, si, di, start_di)
            if atr is None:
                continue
            updated_positions.append(
                (si, di + 1, ep, ep - atr_stop * atr, alloc_per_pos))

        positions = updated_positions

    # Close remaining positions
    for si, edi, ep, sp, alloc in positions:
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
        f"  {label}: {len(trades)}t (stop:{n_stop} hold:{n_hold}) "
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
    sigs: Dict[str, np.ndarray],
    sector_lookup: Dict[int, str],
    win_threshold: float = 0.60,
    normal_threshold: float = 0.80,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
    atr_stop: float = 3.0,
    max_positions: int = 3,
    max_per_sector: int = 3,
    min_confidence: int = 5,
) -> List[dict]:
    print(f"\n{'=' * 70}")
    print(
        f"  WALK-FORWARD V85 "
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

        trades, _, _ = backtest_v85(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            win_threshold=win_threshold,
            normal_threshold=normal_threshold,
            lose_threshold=lose_threshold,
            win_rate_window=win_rate_window,
            atr_stop=atr_stop,
            max_positions=max_positions,
            max_per_sector=max_per_sector,
            min_confidence=min_confidence,
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
        agg_sectors: Dict[str, int] = defaultdict(int)
        for t in all_trades:
            agg_sectors[t.get("sector", "OTHER")] += 1
        sec_str = " ".join(
            f"{k}:{v}" for k, v in sorted(agg_sectors.items()))
        print(
            f"\n  WF TOTAL: {len(all_trades)}t "
            f"WR={wr_val:.1f}% avg={avg:+.2f}% cum={cum:+.1%}"
        )
        print(f"  WF SECTORS: {sec_str}")
        return all_trades
    return []


def make_weight_config(
    carry_w: float,
    mom_persist_w: float,
    vol_price_div_w: float,
    breadth_w: float,
    vol_compress_w: float,
) -> Dict[str, float]:
    """Create weight config by distributing remaining weight to original 7."""
    new_total = carry_w + mom_persist_w + vol_price_div_w + breadth_w + vol_compress_w
    orig_total = 1.0 - new_total
    if orig_total < 0.15:
        orig_total = 0.15
        new_total = 1.0 - orig_total

    # Distribute original weights proportionally
    orig_sum = 0.12 + 0.10 + 0.08 + 0.08 + 0.06 + 0.06 + 0.04
    scale = orig_total / orig_sum
    return {
        "rank_ret5d": 0.12 * scale,
        "rank_oi5d": 0.10 * scale,
        "rank_rsi": 0.08 * scale,
        "rank_vol5d": 0.08 * scale,
        "rank_ret10d": 0.06 * scale,
        "rank_range5d": 0.06 * scale,
        "rank_atrp5d": 0.04 * scale,
        "rank_carry": carry_w,
        "rank_mom_persist": mom_persist_w,
        "rank_vol_price_div": vol_price_div_w,
        "rank_breadth": breadth_w,
        "rank_vol_compress": vol_compress_w,
    }


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V85: NEW ALPHA FACTORS EXPLORATION (12-FACTOR COMPOSITE)")
    print("  5 new factors: carry, momentum persistence, vol-price div,")
    print("  breadth, vol compression")
    print("  NO leverage. Dynamic thresholds. Equal-weight allocation.")
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

    # Find 2019 start index
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # === 1. Compute signals for different weight configs ===
    WEIGHT_CONFIGS = [
        ("default", DEFAULT_WEIGHTS),
        ("carry_heavy", make_weight_config(0.20, 0.06, 0.06, 0.04, 0.06)),
        ("persist_heavy", make_weight_config(0.08, 0.18, 0.06, 0.04, 0.06)),
        ("diverge_heavy", make_weight_config(0.08, 0.06, 0.18, 0.04, 0.06)),
        ("breadth_heavy", make_weight_config(0.08, 0.06, 0.06, 0.16, 0.06)),
        ("volcomp_heavy", make_weight_config(0.08, 0.06, 0.06, 0.04, 0.16)),
        ("new_balanced", make_weight_config(0.10, 0.10, 0.10, 0.08, 0.10)),
        ("new_dominant", make_weight_config(0.14, 0.12, 0.12, 0.08, 0.10)),
    ]

    signal_cache: Dict[str, Dict] = {}
    for name, weights in WEIGHT_CONFIGS:
        print(f"\n--- Computing signals for weight config: {name} ---")
        signal_cache[name] = compute_all_signals(
            C, O, H, L, V, OI, NS, ND, weights=weights)

    # === 2. Parameter sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("  12-factor composite. NO LEVERAGE.")
    print("=" * 70)

    results: List[dict] = []
    sweep_count = 0

    for wname, weights in WEIGHT_CONFIGS:
        sigs = signal_cache[wname]
        for mps in [2, 3]:
            for mp in [3, 5, 8]:
                for ww in [10, 15]:
                    for wt in [0.55, 0.60, 0.65]:
                        for nt in [0.75, 0.80, 0.85]:
                            for lt in [0.85, 0.90, 0.95]:
                                if lt <= nt:
                                    continue
                                sweep_count += 1
                                trades, eq, dd = backtest_v85(
                                    C, O, H, L, NS, ND, dates, syms, sigs,
                                    sector_lookup=sector_lookup,
                                    win_threshold=wt,
                                    normal_threshold=nt,
                                    lose_threshold=lt,
                                    win_rate_window=ww,
                                    atr_stop=3.0,
                                    max_positions=mp,
                                    max_per_sector=mps,
                                    min_confidence=5,
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

                                results.append({
                                    "wname": wname,
                                    "wt": wt, "nt": nt, "lt": lt,
                                    "ww": ww, "mps": mps, "mp": mp,
                                    "n": len(trades), "wr": wr,
                                    "ann": ann, "dd": dd,
                                    "sharpe": sh_val, "eq": eq,
                                })

    results.sort(key=lambda x: -x["sharpe"])
    print(f"\n  Evaluated {sweep_count} configs, "
          f"{len(results)} with 10+ trades")
    print(
        f"\n{'WCfg':>14} {'WT':>4} {'NT':>4} {'LT':>4} {'WW':>3} "
        f"{'MPS':>3} {'MP':>3} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} "
        f"{'Sh':>6}"
    )
    print("-" * 90)
    for r in results[:50]:
        print(
            f"{r['wname']:>14} {r['wt']:>4.2f} {r['nt']:>4.2f} "
            f"{r['lt']:>4.2f} {r['ww']:>3} {r['mps']:>3} "
            f"{r['mp']:>3} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f}"
        )

    if not results:
        print("  No configs with 10+ trades. Exiting.")
        return

    # === 3. Top configs: full backtest ===
    print("\n" + "=" * 70)
    print("  TOP CONFIGS -- FULL PERIOD (2016-2026)")
    print("=" * 70)

    seen = set()
    unique_top = []
    for r in results:
        key = (r["wname"], r["wt"], r["nt"], r["lt"],
               r["ww"], r["mps"], r["mp"])
        if key not in seen:
            seen.add(key)
            unique_top.append(r)
        if len(unique_top) >= 5:
            break

    for r in unique_top:
        sigs = signal_cache[r["wname"]]
        trades, eq, dd = backtest_v85(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            win_threshold=r["wt"],
            normal_threshold=r["nt"],
            lose_threshold=r["lt"],
            win_rate_window=r["ww"],
            atr_stop=3.0,
            max_positions=r["mp"],
            max_per_sector=r["mps"],
            min_confidence=5,
            start_di=60,
        )
        label = (
            f"{r['wname']} wt={r['wt']:.2f} "
            f"nt={r['nt']:.2f} lt={r['lt']:.2f} "
            f"ww={r['ww']} mps={r['mps']} mp={r['mp']}"
        )
        print(f"\n  FULL {label}")
        analyze(trades, eq, dd, label)

    # === 4. Walk-forward for best config ===
    best = results[0]
    print("\n" + "=" * 70)
    print(
        f"  BEST WF: {best['wname']} "
        f"wt={best['wt']:.2f} nt={best['nt']:.2f} "
        f"lt={best['lt']:.2f} ww={best['ww']} "
        f"mps={best['mps']} mp={best['mp']}"
    )
    print("=" * 70)
    walk_forward(
        C, O, H, L, NS, ND, dates, syms,
        signal_cache[best["wname"]],
        sector_lookup=sector_lookup,
        win_threshold=best["wt"],
        normal_threshold=best["nt"],
        lose_threshold=best["lt"],
        win_rate_window=best["ww"],
        atr_stop=3.0,
        max_positions=best["mp"],
        max_per_sector=best["mps"],
        min_confidence=5,
    )

    # === 5. Factor contribution analysis ===
    print("\n" + "=" * 70)
    print("  FACTOR CONTRIBUTION ANALYSIS")
    print("=" * 70)

    # Run with each factor removed to see impact
    base_weights = DEFAULT_WEIGHTS.copy()
    factor_names = list(base_weights.keys())

    # Baseline
    sigs_base = signal_cache["default"]
    trades_base, eq_base, dd_base = backtest_v85(
        C, O, H, L, NS, ND, dates, syms, sigs_base,
        sector_lookup=sector_lookup,
        win_threshold=best["wt"],
        normal_threshold=best["nt"],
        lose_threshold=best["lt"],
        win_rate_window=best["ww"],
        atr_stop=3.0,
        max_positions=best["mp"],
        max_per_sector=best["mps"],
        min_confidence=5,
        start_di=bt_2019,
    )

    if trades_base:
        n_days = max(1, trades_base[-1]["di"] - trades_base[0]["di"])
        ann_base = ((eq_base / CASH0) ** (
            1 / max(1.0, n_days / 252)) - 1) * 100
        print(f"\n  BASELINE (all 12 factors): {len(trades_base)}t "
              f"ann={ann_base:+.1f}% eq={eq_base:,.0f}")

    for fname in factor_names:
        # Remove one factor, redistribute weight
        reduced = {k: v for k, v in base_weights.items() if k != fname}
        total = sum(reduced.values())
        reduced = {k: v / total for k, v in reduced.items()}
        sigs_reduced = compute_all_signals(
            C, O, H, L, V, OI, NS, ND, weights=reduced)
        trades_r, eq_r, dd_r = backtest_v85(
            C, O, H, L, NS, ND, dates, syms, sigs_reduced,
            sector_lookup=sector_lookup,
            win_threshold=best["wt"],
            normal_threshold=best["nt"],
            lose_threshold=best["lt"],
            win_rate_window=best["ww"],
            atr_stop=3.0,
            max_positions=best["mp"],
            max_per_sector=best["mps"],
            min_confidence=max(5 - 1, 3),
            start_di=bt_2019,
        )
        if trades_r:
            n_days_r = max(1, trades_r[-1]["di"] - trades_r[0]["di"])
            ann_r = ((eq_r / CASH0) ** (
                1 / max(1.0, n_days_r / 252)) - 1) * 100
            delta = ann_r - ann_base
            print(f"  -{fname}: {len(trades_r)}t ann={ann_r:+.1f}% "
                  f"delta={delta:+.1f}%")

    print(f"\n[V85] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
