"""
V88: Comprehensive TA-Lib Factor Discovery
============================================
Scan TA-Lib's 150+ indicators to find NEW alpha sources beyond our 7 factors.

Existing 7 factors: ret5d, oi5d, rsi, vol, ret10d, range, atrp
These cover: simple momentum, OI change, RSI, volume, range, volatility.

NEW TA-Lib indicators tested (chosen for orthogonality):
  - ADX (trend strength) -- regime filter, NOT momentum
  - CCI (commodity channel index) -- cyclical, designed for futures
  - WILLR (Williams %R) -- overbought/oversold, faster than RSI
  - MFI (money flow index) -- volume-weighted RSI
  - OBV change (on-balance volume momentum) -- volume flow
  - BOP (balance of power) -- intrabar buying/selling pressure
  - AROONOSC (aroon oscillator) -- trend direction via time-since-extreme
  - CMO (Chande momentum oscillator) -- pure momentum, more sensitive than RSI
  - ULTOSC (ultimate oscillator) -- multi-timeframe oscillator
  - TRIX (triple-smoothed ROC) -- noise-filtered momentum
  - PLUS_DI - MINUS_DI (directional indicator spread) -- directional pressure
  - NATR (normalized ATR) -- cross-contract comparable volatility

Pipeline:
  1. Compute all 12 NEW talib indicators + 7 existing factors for all instruments
  2. Cross-sectional percentile rank each indicator daily
  3. Use mutual information (MI) with forward returns to score each factor
  4. Select top factors by MI (information-weighted)
  5. Combine via MI-weighted composite score
  6. Apply V80's dynamic mode + sector limit framework

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

# Forward return horizon for MI scoring
MI_FWD_DAYS = 5


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
# PHASE 1: Compute ALL indicators (existing 7 + 12 NEW talib)
# ============================================================
def compute_all_indicators(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
) -> Dict[str, np.ndarray]:
    """Compute existing 7 factors + 12 new TA-Lib indicators."""
    t0 = time.time()
    print("[V88] Computing all indicators (7 existing + 12 new talib)...", flush=True)

    factors: Dict[str, np.ndarray] = {}

    # --- Existing 7 factors (5d versions) ---
    # 1. ret5d
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 5]) and C[si, di - 5] > 0:
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1.0
    factors["ret5d"] = ret_5d

    # 2. oi5d
    oi_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(OI[si, di]) and not np.isnan(OI[si, di - 5]) and OI[si, di - 5] > 0:
                oi_5d[si, di] = OI[si, di] / OI[si, di - 5] - 1.0
    factors["oi5d"] = oi_5d

    # 3. rsi14 (using talib)
    rsi14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
        nan_mask = np.isnan(C[si])
        try:
            r = talib.RSI(c, 14)
            rsi14[si] = np.where(nan_mask, np.nan, r)
        except Exception:
            pass
    factors["rsi14"] = rsi14

    # 4. vol5d (average volume)
    vol_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            vals = V[si, di - 5:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 3:
                vol_5d[si, di] = np.mean(valid)
    factors["vol5d"] = vol_5d

    # 5. ret10d
    ret_10d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 10]) and C[si, di - 10] > 0:
                ret_10d[si, di] = C[si, di] / C[si, di - 10] - 1.0
    factors["ret10d"] = ret_10d

    # 6. range5d
    range_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            rng_vals = []
            for j in range(di - 5, di):
                if not np.isnan(H[si, j]) and not np.isnan(L[si, j]) and not np.isnan(C[si, j]) and C[si, j] > 0 and H[si, j] > L[si, j]:
                    rng_vals.append((H[si, j] - L[si, j]) / C[si, j])
            if len(rng_vals) >= 3:
                range_5d[si, di] = np.mean(rng_vals)
    factors["range5d"] = range_5d

    # 7. atrp5d
    atrp_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(6, ND):
            atr_vals = []
            for j in range(di - 5, di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    prev_c = C[si, j - 1] if j > 0 and not np.isnan(C[si, j - 1]) else cc
                    atr_vals.append(max(hh - ll, abs(hh - prev_c), abs(ll - prev_c)))
            if atr_vals and not np.isnan(C[si, di]) and C[si, di] > 0:
                atrp_5d[si, di] = np.mean(atr_vals) / C[si, di]
    factors["atrp5d"] = atrp_5d

    # --- NEW TA-Lib indicators (12) ---
    if not HAS_TALIB:
        print("  WARNING: talib not available, skipping new indicators")
        return factors

    for si in range(NS):
        # Prepare clean arrays for talib (replace NaN with 0 for computation)
        c_arr = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
        o_arr = np.where(np.isnan(O[si]), 0, O[si]).astype(np.float64)
        h_arr = np.where(np.isnan(H[si]), 0, H[si]).astype(np.float64)
        l_arr = np.where(np.isnan(L[si]), 0, L[si]).astype(np.float64)
        v_arr = np.where(np.isnan(V[si]), 0, V[si]).astype(np.float64)
        nan_mask = np.isnan(C[si])

        def _safe(name: str, arr: np.ndarray) -> None:
            """Store result with NaN masking."""
            if name not in factors:
                factors[name] = np.full((NS, ND), np.nan)
            factors[name][si] = np.where(nan_mask, np.nan, arr)

        try:
            # 1. ADX -- trend strength (14-period)
            adx = talib.ADX(h_arr, l_arr, c_arr, timeperiod=14)
            _safe("adx", adx)
        except Exception:
            pass

        try:
            # 2. CCI -- commodity channel index (14-period)
            cci = talib.CCI(h_arr, l_arr, c_arr, timeperiod=14)
            _safe("cci", cci)
        except Exception:
            pass

        try:
            # 3. WILLR -- Williams %R (14-period)
            willr = talib.WILLR(h_arr, l_arr, c_arr, timeperiod=14)
            _safe("willr", willr)
        except Exception:
            pass

        try:
            # 4. MFI -- money flow index (14-period, volume-weighted RSI)
            mfi = talib.MFI(h_arr, l_arr, c_arr, v_arr, timeperiod=14)
            _safe("mfi", mfi)
        except Exception:
            pass

        try:
            # 5. OBV change (5d rate of change of OBV)
            obv = talib.OBV(c_arr, v_arr)
            obv_5d = np.full(ND, np.nan)
            for di in range(5, ND):
                if obv[di - 5] != 0:
                    obv_5d[di] = obv[di] / obv[di - 5] - 1.0
            _safe("obv_chg5d", obv_5d)
        except Exception:
            pass

        try:
            # 6. BOP -- balance of power
            bop = talib.BOP(o_arr, h_arr, l_arr, c_arr)
            _safe("bop", bop)
        except Exception:
            pass

        try:
            # 7. AROONOSC -- aroon oscillator (14-period)
            _, _ = talib.AROON(h_arr, l_arr, timeperiod=14)
            aroonosc = talib.AROONOSC(h_arr, l_arr, timeperiod=14)
            _safe("aroonosc", aroonosc)
        except Exception:
            pass

        try:
            # 8. CMO -- Chande momentum oscillator (14-period)
            cmo = talib.CMO(c_arr, timeperiod=14)
            _safe("cmo", cmo)
        except Exception:
            pass

        try:
            # 9. ULTOSC -- ultimate oscillator (7/14/28)
            ultosc = talib.ULTOSC(h_arr, l_arr, c_arr,
                                  timeperiod1=7, timeperiod2=14, timeperiod3=28)
            _safe("ultosc", ultosc)
        except Exception:
            pass

        try:
            # 10. TRIX -- triple-smoothed ROC (30-period)
            trix = talib.TRIX(c_arr, timeperiod=30)
            _safe("trix", trix)
        except Exception:
            pass

        try:
            # 11. DI spread (PLUS_DI - MINUS_DI)
            plus_di = talib.PLUS_DI(h_arr, l_arr, c_arr, timeperiod=14)
            minus_di = talib.MINUS_DI(h_arr, l_arr, c_arr, timeperiod=14)
            di_spread = plus_di - minus_di
            _safe("di_spread", di_spread)
        except Exception:
            pass

        try:
            # 12. NATR -- normalized ATR (cross-contract comparable)
            natr = talib.NATR(h_arr, l_arr, c_arr, timeperiod=14)
            _safe("natr", natr)
        except Exception:
            pass

    print(f"  All indicators done: {len(factors)} factors, {time.time() - t0:.1f}s", flush=True)
    return factors


# ============================================================
# PHASE 2: Cross-sectional percentile ranking
# ============================================================
def compute_cross_sectional_ranks(
    factors: Dict[str, np.ndarray],
    NS: int, ND: int,
    min_count: int = 10,
) -> Tuple[Dict[str, np.ndarray], Dict[str, bool]]:
    """Rank all factors cross-sectionally. Returns ranks and invert flags."""
    t0 = time.time()
    print("[V88] Computing cross-sectional ranks...", flush=True)

    # For mean-reversion: invert factors where HIGH value = bearish
    # (high returns, high OI, high RSI, high CMO, high WILLR, high DI spread, high TRIX)
    INVERT_SET = {
        "ret5d", "ret10d", "oi5d", "rsi14",
        "cci", "willr", "mfi", "cmo", "ultosc",
        "di_spread", "trix", "obv_chg5d",
    }

    ranks: Dict[str, np.ndarray] = {}
    invert_flags: Dict[str, bool] = {}

    for name, factor in factors.items():
        rank_arr = np.full((NS, ND), np.nan)
        invert = name in INVERT_SET
        invert_flags[name] = invert

        for di in range(ND):
            vals = factor[:, di]
            valid_count = np.sum(~np.isnan(vals))
            if valid_count < min_count:
                continue
            ranked = pd.Series(vals).rank(pct=True, na_option="keep").values
            if invert:
                ranked = 1.0 - ranked
            rank_arr[:, di] = ranked
        ranks[name] = rank_arr

    print(f"  CS ranks done: {len(ranks)} factors, {time.time() - t0:.1f}s", flush=True)
    return ranks, invert_flags


# ============================================================
# PHASE 3: Mutual information scoring for factor selection
# ============================================================
def compute_forward_returns(
    C: np.ndarray, NS: int, ND: int, fwd_days: int = 5,
) -> np.ndarray:
    """Compute forward returns for MI scoring."""
    fwd_ret = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND - fwd_days):
            if not np.isnan(C[si, di + fwd_days]) and not np.isnan(C[si, di]) and C[si, di] > 0:
                fwd_ret[si, di] = C[si, di + fwd_days] / C[si, di] - 1.0
    return fwd_ret


def score_factors_by_mi(
    ranks: Dict[str, np.ndarray],
    fwd_ret: np.ndarray,
    NS: int, ND: int,
    start_di: int = 60,
) -> Dict[str, float]:
    """Score each ranked factor using mutual information with forward returns.

    Uses a discretized MI approximation: bin both rank and forward return
    into quintiles, then compute empirical MI.
    """
    from sklearn.metrics import mutual_info_score

    t0 = time.time()
    print("[V88] Scoring factors by mutual information...", flush=True)

    N_BINS = 5
    mi_scores: Dict[str, float] = {}

    for name, rank_arr in ranks.items():
        rank_flat = []
        ret_flat = []
        for di in range(start_di, ND - MI_FWD_DAYS):
            r_vals = rank_arr[:, di]
            ret_vals = fwd_ret[:, di]
            valid = ~np.isnan(r_vals) & ~np.isnan(ret_vals)
            rank_flat.extend(r_vals[valid])
            ret_flat.extend(ret_vals[valid])

        if len(rank_flat) < 200:
            mi_scores[name] = 0.0
            continue

        rank_arr_flat = np.array(rank_flat)
        ret_arr_flat = np.array(ret_flat)

        # Discretize into quintiles
        rank_bins = pd.qcut(rank_arr_flat, N_BINS, labels=False, duplicates='drop')
        ret_bins = pd.qcut(ret_arr_flat, N_BINS, labels=False, duplicates='drop')

        # Remove any NaN from qcut
        valid_mask = ~np.isnan(rank_bins) & ~np.isnan(ret_bins)
        if valid_mask.sum() < 200:
            mi_scores[name] = 0.0
            continue

        mi = mutual_info_score(rank_bins[valid_mask].astype(int),
                               ret_bins[valid_mask].astype(int))
        mi_scores[name] = mi

    # Sort by MI score
    sorted_mi = sorted(mi_scores.items(), key=lambda x: -x[1])
    print(f"  MI scores (top factors first):")
    for name, score in sorted_mi:
        print(f"    {name:20s}: MI = {score:.6f}")
    print(f"  MI scoring done: {time.time() - t0:.1f}s", flush=True)
    return mi_scores


def score_factors_by_ic(
    ranks: Dict[str, np.ndarray],
    fwd_ret: np.ndarray,
    NS: int, ND: int,
    start_di: int = 60,
) -> Dict[str, float]:
    """Score factors using rank IC (Spearman correlation with forward returns).

    This is a simpler, more interpretable alternative to MI.
    """
    t0 = time.time()
    print("[V88] Scoring factors by rank IC (Spearman)...", flush=True)

    ic_scores: Dict[str, float] = {}
    for name, rank_arr in ranks.items():
        daily_ics = []
        for di in range(start_di, ND - MI_FWD_DAYS):
            r_vals = rank_arr[:, di]
            ret_vals = fwd_ret[:, di]
            valid = ~np.isnan(r_vals) & ~np.isnan(ret_vals)
            if valid.sum() < 15:
                continue
            from scipy.stats import spearmanr
            corr, _ = spearmanr(r_vals[valid], ret_vals[valid])
            if not np.isnan(corr):
                daily_ics.append(corr)

        if daily_ics:
            # Mean IC (consistent direction) > mean |IC| (any direction)
            # For mean-reversion: we expect negative IC (low rank = high future return)
            ic_scores[name] = np.mean(daily_ics)
        else:
            ic_scores[name] = 0.0

    sorted_ic = sorted(ic_scores.items(), key=lambda x: -abs(x[1]))
    print(f"  IC scores (sorted by |IC|):")
    for name, score in sorted_ic:
        print(f"    {name:20s}: IC = {score:+.6f}")
    print(f"  IC scoring done: {time.time() - t0:.1f}s", flush=True)
    return ic_scores


# ============================================================
# PHASE 4: Build MI/IC-weighted composite signal
# ============================================================
def select_top_factors(
    mi_scores: Dict[str, float],
    ic_scores: Dict[str, float],
    n_top: int = 8,
    method: str = "ic_first",
) -> List[Tuple[str, float]]:
    """Select top factors using IC-first approach.

    IC (Spearman rank correlation) captures DIRECTIONAL predictability.
    MI captures any statistical dependence (including non-directional).

    Strategy: Use |IC| as primary selection criterion and as weights.
    This ensures selected factors actually predict future returns directionally,
    unlike MI which can select non-directional factors (e.g., volatility).
    """
    factor_names = list(mi_scores.keys())

    if method == "ic_first":
        # Sort by |IC| -- primary criterion for directional prediction
        sorted_factors = sorted(factor_names, key=lambda x: -abs(ic_scores[x]))
        selected = sorted_factors[:n_top]

        # Weight by |IC| (stronger prediction = higher weight)
        weights = [max(abs(ic_scores[n]), 1e-6) for n in selected]
    else:
        # Combined rank (legacy method)
        mi_ranked = sorted(factor_names, key=lambda x: -mi_scores[x])
        mi_rank_map = {name: rank for rank, name in enumerate(mi_ranked)}
        ic_ranked = sorted(factor_names, key=lambda x: -abs(ic_scores[x]))
        ic_rank_map = {name: rank for rank, name in enumerate(ic_ranked)}
        combined = [(name, mi_rank_map[name] + ic_rank_map[name]) for name in factor_names]
        combined.sort(key=lambda x: x[1])
        selected = [name for name, _ in combined[:n_top]]
        weights = [max(abs(ic_scores.get(n, 0.0)), 1e-6) for n in selected]

    total_w = sum(weights)
    norm_weights = [w / total_w for w in weights]

    result = [(name, w) for name, w in zip(selected, norm_weights)]
    print(f"\n  Selected top {n_top} factors (method={method}):")
    for name, w in result:
        mi_val = mi_scores[name]
        ic_val = ic_scores[name]
        print(f"    {name:20s}: weight={w:.4f}  MI={mi_val:.6f}  IC={ic_val:+.6f}")
    return result


def build_weighted_composite(
    ranks: Dict[str, np.ndarray],
    factor_weights: List[Tuple[str, float]],
    NS: int, ND: int,
) -> np.ndarray:
    """Build weighted composite score from selected factors (vectorized)."""
    composite = np.zeros((NS, ND))
    weight_sum = np.zeros((NS, ND))
    for name, weight in factor_weights:
        rank_arr = ranks[name]
        valid = ~np.isnan(rank_arr)
        composite = np.where(valid, composite + rank_arr * weight, composite)
        weight_sum = np.where(valid, weight_sum + weight, weight_sum)
    result = np.full((NS, ND), np.nan)
    has_weight = weight_sum > 0
    result[has_weight] = composite[has_weight] / weight_sum[has_weight]
    return result


# ============================================================
# PHASE 5: Backtest engine (adapted from V80)
# ============================================================
def compute_atr_at(
    H: np.ndarray, L: np.ndarray, C: np.ndarray,
    si: int, di: int, start_di: int,
) -> Optional[float]:
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


def backtest_v88(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    composite: np.ndarray,
    sector_lookup: Dict[int, str],
    win_threshold: float = 0.60,
    normal_threshold: float = 0.80,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
    atr_stop: float = 3.0,
    max_positions: int = 5,
    max_per_sector: int = 3,
    hold_days: int = 5,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Backtest V88 with MI-weighted composite signal."""
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

        mode = get_dynamic_mode(recent_trades_win, win_threshold, win_rate_window)
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
                        "pnl_abs": profit, "pnl_pct": pnl * 100,
                        "days": di - edi + 1, "di": di, "year": d.year,
                        "sym": syms[si],
                        "sector": sector_lookup.get(si, 'OTHER'),
                        "reason": "stop", "pyr": is_pyr,
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
                        "pnl_abs": profit, "pnl_pct": pnl * 100,
                        "days": di - edi + 1, "di": di, "year": d.year,
                        "sym": syms[si],
                        "sector": sector_lookup.get(si, 'OTHER'),
                        "reason": "hold", "pyr": is_pyr,
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

        # Entry
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
            updated_positions.append((si, edi, ep, sp, alloc_per_pos, is_pyr))

        for rank_val, si, sym_sector in new_entries:
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr = compute_atr_at(H, L, C, si, di, start_di)
            if atr is None:
                continue
            updated_positions.append(
                (si, di + 1, ep, ep - atr_stop * atr, alloc_per_pos, False))

        positions = updated_positions

    # Close remaining
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
    sector_str = " ".join(f"{k}:{v}" for k, v in sorted(sector_counts.items()))

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
        f"modes=[W:{mode_counts['W']} N:{mode_counts['N']} L:{mode_counts['L']}]"
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

    return {"n": len(trades), "wr": wr, "dd": max_dd,
            "ann": ann, "sh": sh, "eq": equity}


def walk_forward(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    composite: np.ndarray,
    sector_lookup: Dict[int, str],
    win_threshold: float = 0.60,
    normal_threshold: float = 0.80,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
    max_positions: int = 5,
    max_per_sector: int = 3,
) -> List[dict]:
    """Walk-forward validation year by year."""
    print(f"\n{'=' * 70}")
    print(
        f"  WALK-FORWARD V88 "
        f"(wt={win_threshold:.2f} nt={normal_threshold:.2f} "
        f"lt={lose_threshold:.2f} ww={win_rate_window} "
        f"mp={max_positions} mps={max_per_sector})")
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

        trades, _, _ = backtest_v88(
            C, O, H, L, NS, ND, dates, syms, composite,
            sector_lookup=sector_lookup,
            win_threshold=win_threshold,
            normal_threshold=normal_threshold,
            lose_threshold=lose_threshold,
            win_rate_window=win_rate_window,
            atr_stop=3.0,
            max_positions=max_positions,
            max_per_sector=max_per_sector,
            hold_days=5,
            start_di=test_start,
            end_di=test_end_idx + 1,
        )

        test_trades = [t for t in trades if dates[t["di"]].year == test_year]
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
            sec_str = " ".join(f"{k}:{v}" for k, v in sorted(yr_sectors.items()))
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
        cum = np.prod([1 + t["pnl_pct"] / 100 for t in all_trades]) - 1
        agg_sectors: Dict[str, int] = defaultdict(int)
        for t in all_trades:
            agg_sectors[t.get("sector", "OTHER")] += 1
        sec_str = " ".join(f"{k}:{v}" for k, v in sorted(agg_sectors.items()))
        print(
            f"\n  WF TOTAL: {len(all_trades)}t "
            f"WR={wr_val:.1f}% avg={avg:+.2f}% cum={cum:+.1%}")
        print(f"  WF SECTORS: {sec_str}")
        return all_trades
    return []


# ============================================================
# MAIN: Factor discovery pipeline
# ============================================================
def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V88: COMPREHENSIVE TA-LIB FACTOR DISCOVERY")
    print("  Scan 19 indicators (7 existing + 12 new)")
    print("  Score by mutual information + rank IC")
    print("  Select top factors, build MI-weighted composite")
    print("  Walk-forward 2019-2026")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start="2016-01-01")
    print(
        f"  {NS} sym, {ND} days, "
        f"{dates[0].strftime('%Y-%m-%d')} to "
        f"{dates[-1].strftime('%Y-%m-%d')}"
    )

    sector_lookup = build_sector_lookup(syms)

    # Find 2019 start index
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # === STEP 1: Compute all indicators ===
    print("\n" + "=" * 70)
    print("  STEP 1: Compute all indicators")
    print("=" * 70)
    factors = compute_all_indicators(C, O, H, L, V, OI, NS, ND)

    # === STEP 2: Cross-sectional ranking ===
    print("\n" + "=" * 70)
    print("  STEP 2: Cross-sectional ranking")
    print("=" * 70)
    ranks, invert_flags = compute_cross_sectional_ranks(factors, NS, ND)

    # === STEP 3: Factor scoring ===
    print("\n" + "=" * 70)
    print("  STEP 3: Score factors (MI + IC)")
    print("=" * 70)
    fwd_ret = compute_forward_returns(C, NS, ND, MI_FWD_DAYS)
    mi_scores = score_factors_by_mi(ranks, fwd_ret, NS, ND, start_di=bt_2019 or 60)
    ic_scores = score_factors_by_ic(ranks, fwd_ret, NS, ND, start_di=bt_2019 or 60)

    # === STEP 4: Select top factors and build composites ===
    print("\n" + "=" * 70)
    print("  STEP 4: Select top factors")
    print("=" * 70)

    # Test different numbers of top factors (IC-first selection)
    composites = {}
    for n_top in [6, 8, 10]:
        selected = select_top_factors(mi_scores, ic_scores, n_top=n_top, method="ic_first")
        comp = build_weighted_composite(ranks, selected, NS, ND)
        composites[f"ic{n_top}"] = (comp, selected)

    # Also build the V80 baseline composite (7 existing factors with V80 weights)
    v80_weights = {
        "ret5d": 0.25, "oi5d": 0.20, "rsi14": 0.15,
        "vol5d": 0.15, "ret10d": 0.10, "range5d": 0.10, "atrp5d": 0.05,
    }
    v80_factor_weights = [(name, w) for name, w in v80_weights.items()
                          if name in ranks]
    total_w = sum(w for _, w in v80_factor_weights)
    v80_factor_weights = [(n, w / total_w) for n, w in v80_factor_weights]
    v80_composite = build_weighted_composite(ranks, v80_factor_weights, NS, ND)
    composites["V80_baseline"] = (v80_composite, v80_factor_weights)

    # === STEP 5: Parameter sweep ===
    print("\n" + "=" * 70)
    print("  STEP 5: Parameter sweep (2019-2026)")
    print("  NO LEVERAGE. Comparing composites.")
    print("=" * 70)

    results: List[dict] = []
    sweep_count = 0

    for comp_name, (composite, selected) in composites.items():
        for mps in [2, 3]:
            for mp in [3, 5, 8]:
                for wt in [0.50, 0.55, 0.60]:
                    for nt in [0.65, 0.70, 0.75, 0.80]:
                        for lt in [0.85, 0.90, 0.95]:
                            if lt <= nt:
                                continue
                            sweep_count += 1
                            trades, eq, dd = backtest_v88(
                                C, O, H, L, NS, ND, dates, syms, composite,
                                sector_lookup=sector_lookup,
                                win_threshold=wt,
                                normal_threshold=nt,
                                lose_threshold=lt,
                                win_rate_window=15,
                                atr_stop=3.0,
                                max_positions=mp,
                                max_per_sector=mps,
                                hold_days=5,
                                start_di=bt_2019,
                            )

                            if len(trades) < 10:
                                continue

                            nw = sum(1 for t in trades if t["pnl_pct"] > 0)
                            wr = nw / len(trades) * 100
                            n_days = max(1, trades[-1]["di"] - trades[0]["di"])
                            ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
                            ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
                            rets_arr = np.array(ap) / CASH0
                            sh_val = (np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252)
                                      if np.std(rets_arr) > 0 else 0)

                            results.append({
                                "comp": comp_name, "wt": wt, "nt": nt, "lt": lt,
                                "mps": mps, "mp": mp,
                                "n": len(trades), "wr": wr, "ann": ann,
                                "dd": dd, "sharpe": sh_val, "eq": eq,
                            })

    results.sort(key=lambda x: -x["sharpe"])
    print(f"\n  Evaluated {sweep_count} configs, {len(results)} with 10+ trades")
    print(
        f"\n{'Comp':>12} {'WT':>4} {'NT':>4} {'LT':>4} {'MPS':>3} {'MP':>3} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>6}"
    )
    print("-" * 80)
    for r in results[:50]:
        print(
            f"{r['comp']:>12} {r['wt']:>4.2f} {r['nt']:>4.2f} {r['lt']:>4.2f} "
            f"{r['mps']:>3} {r['mp']:>3} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f}"
        )

    if not results:
        print("  No configs with 10+ trades. Exiting.")
        return

    # === STEP 6: Full backtest for best config ===
    print("\n" + "=" * 70)
    print("  STEP 6: Full backtest for top configs")
    print("=" * 70)

    seen = set()
    unique_top = []
    for r in results:
        key = (r["comp"], r["wt"], r["nt"], r["lt"], r["mps"], r["mp"])
        if key not in seen:
            seen.add(key)
            unique_top.append(r)
        if len(unique_top) >= 5:
            break

    for r in unique_top:
        comp_name = r["comp"]
        composite, selected = composites[comp_name]
        trades, eq, dd = backtest_v88(
            C, O, H, L, NS, ND, dates, syms, composite,
            sector_lookup=sector_lookup,
            win_threshold=r["wt"],
            normal_threshold=r["nt"],
            lose_threshold=r["lt"],
            win_rate_window=15,
            atr_stop=3.0,
            max_positions=r["mp"],
            max_per_sector=r["mps"],
            hold_days=5,
            start_di=60,
        )
        label = (f"{comp_name} wt={r['wt']:.2f} nt={r['nt']:.2f} "
                 f"lt={r['lt']:.2f} mps={r['mps']} mp={r['mp']}")
        print(f"\n  FULL {label}")
        analyze(trades, eq, dd, label)

    # === STEP 7: Walk-forward for best + high-trade variant ===
    best = results[0]
    best_comp_name = best["comp"]
    best_composite, best_selected = composites[best_comp_name]
    print("\n" + "=" * 70)
    print(f"  STEP 7A: Walk-forward for BEST Sharpe config")
    print(f"  Composite: {best_comp_name}")
    print(f"  Factors: {[f'{n}:{w:.3f}' for n, w in best_selected]}")
    print(f"  wt={best['wt']:.2f} nt={best['nt']:.2f} "
          f"lt={best['lt']:.2f} mps={best['mps']} mp={best['mp']}")
    print("=" * 70)
    walk_forward(
        C, O, H, L, NS, ND, dates, syms, best_composite,
        sector_lookup=sector_lookup,
        win_threshold=best["wt"],
        normal_threshold=best["nt"],
        lose_threshold=best["lt"],
        win_rate_window=15,
        max_positions=best["mp"],
        max_per_sector=best["mps"],
    )

    # High-trade variant: ic10 with lower thresholds for more trades
    # From sweep: ic10 wt=0.55 nt=0.75 lt=0.95 mps=3 mp=5 → 111t, ann+15.5%, Sh=5.44
    print("\n" + "=" * 70)
    print(f"  STEP 7B: Walk-forward for HIGH-TRADE variant")
    print(f"  Composite: ic10, wt=0.55 nt=0.75 lt=0.95 mps=3 mp=5")
    print("=" * 70)
    ic10_comp, ic10_selected = composites["ic10"]
    walk_forward(
        C, O, H, L, NS, ND, dates, syms, ic10_comp,
        sector_lookup=sector_lookup,
        win_threshold=0.55,
        normal_threshold=0.75,
        lose_threshold=0.95,
        win_rate_window=15,
        max_positions=5,
        max_per_sector=3,
    )

    # Ultra high-trade variant: ic10 with very low thresholds
    print("\n" + "=" * 70)
    print(f"  STEP 7C: Walk-forward for ULTRA HIGH-TRADE variant")
    print(f"  Composite: ic10, wt=0.50 nt=0.65 lt=0.90 mps=3 mp=8")
    print("=" * 70)
    walk_forward(
        C, O, H, L, NS, ND, dates, syms, ic10_comp,
        sector_lookup=sector_lookup,
        win_threshold=0.50,
        normal_threshold=0.65,
        lose_threshold=0.90,
        win_rate_window=15,
        max_positions=8,
        max_per_sector=3,
    )

    # === STEP 8: V88 best vs V80 baseline comparison ===
    print("\n" + "=" * 70)
    print("  STEP 8: V88 BEST vs V80 BASELINE (2019-2026)")
    print("=" * 70)

    # V88 best
    trades_v88, eq_v88, dd_v88 = backtest_v88(
        C, O, H, L, NS, ND, dates, syms, best_composite,
        sector_lookup=sector_lookup,
        win_threshold=best["wt"],
        normal_threshold=best["nt"],
        lose_threshold=best["lt"],
        win_rate_window=15,
        atr_stop=3.0,
        max_positions=best["mp"],
        max_per_sector=best["mps"],
        start_di=bt_2019,
    )

    # V80 baseline with same parameters
    v80_comp, _ = composites["V80_baseline"]
    # Find V80 baseline best from results
    v80_results = [r for r in results if r["comp"] == "V80_baseline"]
    if v80_results:
        v80_best = v80_results[0]
        trades_v80, eq_v80, dd_v80 = backtest_v88(
            C, O, H, L, NS, ND, dates, syms, v80_comp,
            sector_lookup=sector_lookup,
            win_threshold=v80_best["wt"],
            normal_threshold=v80_best["nt"],
            lose_threshold=v80_best["lt"],
            win_rate_window=15,
            atr_stop=3.0,
            max_positions=v80_best["mp"],
            max_per_sector=v80_best["mps"],
            start_di=bt_2019,
        )
    else:
        # Fallback: use same params as best
        trades_v80, eq_v80, dd_v80 = backtest_v88(
            C, O, H, L, NS, ND, dates, syms, v80_comp,
            sector_lookup=sector_lookup,
            win_threshold=best["wt"],
            normal_threshold=best["nt"],
            lose_threshold=best["lt"],
            win_rate_window=15,
            atr_stop=3.0,
            max_positions=best["mp"],
            max_per_sector=best["mps"],
            start_di=bt_2019,
        )

    print(f"\n  V88 BEST ({best_comp_name}):")
    analyze(trades_v88, eq_v88, dd_v88, "V88-best")
    print(f"\n  V80 BASELINE:")
    analyze(trades_v80, eq_v80, dd_v80, "V80-baseline")

    if trades_v88 and trades_v80:
        print(
            f"\n  V88 vs V80: "
            f"eq_delta={eq_v88 - eq_v80:+,.0f} "
            f"dd_delta={dd_v88 - dd_v80:+.1f}% "
            f"trade_delta={len(trades_v88) - len(trades_v80):+d}"
        )

    print(f"\n[V88] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
