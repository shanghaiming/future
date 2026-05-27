"""
V92: Volatility-Adaptive Dynamic Lookback Strategy
====================================================
Research (2106.08420) proves dynamic lookback windows improve momentum
Sharpe by +66%. Fixed lookbacks (5d, 10d, 20d) are suboptimal.

Core Innovation: Make lookback windows ADAPTIVE based on volatility regime:
  - HIGH vol instruments (vol rank > high_vol_pct): SHORT lookback
    Rationale: In high vol, recent info more relevant, old signals decay fast
  - NORMAL vol (low_vol_pct to high_vol_pct): MEDIUM lookback
    Standard momentum window
  - LOW vol instruments (vol rank < low_vol_pct): LONG lookback
    Rationale: In low vol, need more data to detect signal, momentum persists

Implementation on top of V80 framework:
  - Replace fixed ret5d/ret10d with dynamic lookback returns
  - For each instrument, compute return using its vol-adaptive lookback
  - Then cross-sectionally rank as before
  - Apply V80's multi-TF + sector limit + dynamic mode

Data: from alpha_futures_data import load_all_data
CASH0=1,000,000, COMM=0.0005, NO leverage
Walk-forward 2019-2026.
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


def compute_rolling_vol(
    C: np.ndarray, NS: int, ND: int, vol_window: int = 20,
) -> np.ndarray:
    """Compute rolling realized volatility (std of daily returns)."""
    vol = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(vol_window, ND):
            closes = C[si, di - vol_window:di]
            valid = closes[~np.isnan(closes)]
            if len(valid) < vol_window // 2 or valid[0] <= 0:
                continue
            rets = np.diff(valid) / valid[:-1]
            if len(rets) >= vol_window // 2:
                vol[si, di] = np.std(rets)
    return vol


def compute_vol_rank(
    vol: np.ndarray, NS: int, ND: int, min_count: int = 10,
) -> np.ndarray:
    """Cross-sectional percentile rank of volatility."""
    vol_rank = np.full((NS, ND), np.nan)
    for di in range(ND):
        vals = vol[:, di]
        valid_count = np.sum(~np.isnan(vals))
        if valid_count < min_count:
            continue
        ranked = pd.Series(vals).rank(pct=True, na_option="keep").values
        vol_rank[:, di] = ranked
    return vol_rank


def compute_dynamic_lookback_returns(
    C: np.ndarray, OI: np.ndarray, V: np.ndarray, H: np.ndarray,
    L: np.ndarray, NS: int, ND: int,
    vol_rank: np.ndarray,
    high_vol_pct: float, low_vol_pct: float,
    short_lb: int, medium_lb: int, long_lb: int,
    vol_window: int = 20,
) -> Dict[str, np.ndarray]:
    """Compute returns using volatility-adaptive dynamic lookback windows.

    For each instrument on each day:
    - HIGH vol (rank > high_vol_pct): use short_lb
    - NORMAL vol (rank between low and high): use medium_lb
    - LOW vol (rank < low_vol_pct): use long_lb
    """
    t0 = time.time()
    print(
        f"[V92] Computing dynamic lookback returns "
        f"(short={short_lb} medium={medium_lb} long={long_lb})...",
        flush=True)

    # Price return with dynamic lookback
    ret_dyn = np.full((NS, ND), np.nan)
    # OI change with dynamic lookback
    oi_dyn = np.full((NS, ND), np.nan)
    # Volume with dynamic lookback
    vol_dyn = np.full((NS, ND), np.nan)
    # Range with dynamic lookback
    range_dyn = np.full((NS, ND), np.nan)
    # ATR% with dynamic lookback
    atrp_dyn = np.full((NS, ND), np.nan)

    for si in range(NS):
        for di in range(long_lb, ND):
            vr = vol_rank[si, di]
            if np.isnan(vr):
                continue

            # Determine lookback based on vol regime
            if vr > high_vol_pct:
                lb = short_lb
            elif vr < low_vol_pct:
                lb = long_lb
            else:
                lb = medium_lb

            if di < lb:
                continue

            # Dynamic return
            if (not np.isnan(C[si, di])
                    and not np.isnan(C[si, di - lb])
                    and C[si, di - lb] > 0):
                ret_dyn[si, di] = C[si, di] / C[si, di - lb] - 1.0

            # Dynamic OI change
            if (not np.isnan(OI[si, di])
                    and not np.isnan(OI[si, di - lb])
                    and OI[si, di - lb] > 0):
                oi_dyn[si, di] = OI[si, di] / OI[si, di - lb] - 1.0

            # Dynamic volume (avg over lookback)
            vals = V[si, di - lb:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= max(3, lb // 2):
                vol_dyn[si, di] = np.mean(valid)

            # Dynamic range (avg over lookback)
            rng_vals = []
            for j in range(di - lb, di):
                if (not np.isnan(H[si, j]) and not np.isnan(L[si, j])
                        and not np.isnan(C[si, j]) and C[si, j] > 0
                        and H[si, j] > L[si, j]):
                    rng_vals.append((H[si, j] - L[si, j]) / C[si, j])
            if len(rng_vals) >= max(3, lb // 2):
                range_dyn[si, di] = np.mean(rng_vals)

            # Dynamic ATR%
            atr_vals = []
            for j in range(max(1, di - lb), di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    prev_c = (
                        C[si, j - 1]
                        if j > 0 and not np.isnan(C[si, j - 1])
                        else cc)
                    atr_vals.append(
                        max(hh - ll, abs(hh - prev_c), abs(ll - prev_c)))
            if (atr_vals and not np.isnan(C[si, di])
                    and C[si, di] > 0 and len(atr_vals) >= max(3, lb // 2)):
                atrp_dyn[si, di] = np.mean(atr_vals) / C[si, di]

    print(f"  Dynamic lookback returns done: {time.time() - t0:.1f}s",
          flush=True)
    return {
        "ret_dyn": ret_dyn,
        "oi_dyn": oi_dyn,
        "vol_dyn": vol_dyn,
        "range_dyn": range_dyn,
        "atrp_dyn": atrp_dyn,
    }


def compute_raw_factors_static(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
) -> Dict[str, np.ndarray]:
    """Compute static 20d factors for medium-term component."""
    t0 = time.time()
    print("[V92] Computing static 20d factors...", flush=True)

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

    print(f"  Static factors done: {time.time() - t0:.1f}s", flush=True)
    return {
        "ret_20d": ret_20d,
        "oi_20d": oi_20d,
        "vol_20d": vol_20d,
        "range_20d": range_20d,
        "atrp_20d": atrp_20d,
        "rsi14": rsi14,
    }


def compute_cross_sectional_ranks(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int, min_count: int = 10,
) -> Dict[str, np.ndarray]:
    t0 = time.time()
    print("[V92] Computing cross-sectional ranks...", flush=True)

    factors_to_rank = {
        "rank_ret_dyn": raw_factors["ret_dyn"],
        "rank_oi_dyn": raw_factors["oi_dyn"],
        "rank_vol_dyn": raw_factors["vol_dyn"],
        "rank_range_dyn": raw_factors["range_dyn"],
        "rank_rsi": raw_factors["rsi14"],
        "rank_atrp_dyn": raw_factors["atrp_dyn"],
        "rank_ret20d": raw_factors["ret_20d"],
        "rank_oi20d": raw_factors["oi_20d"],
        "rank_vol20d": raw_factors["vol_20d"],
        "rank_range20d": raw_factors["range_20d"],
        "rank_atrp20d": raw_factors["atrp_20d"],
    }

    INVERT_FACTORS = {
        "rank_ret_dyn", "rank_oi_dyn", "rank_rsi",
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
    st_weight: float,
    NS: int, ND: int,
    min_factors: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build multi-TF composite with dynamic-lookback short-term factors.

    Short-term (dynamic lookback) weights:
      ret_dyn: 0.30, oi_dyn: 0.20, rsi: 0.15, vol_dyn: 0.10,
      range_dyn: 0.10, atrp_dyn: 0.05, ret_20d: 0.10

    Medium-term (static 20d) weights:
      ret_20d: 0.25, oi_20d: 0.20, rsi: 0.15, vol_20d: 0.15,
      range_20d: 0.10, atrp_20d: 0.05, ret_dyn: 0.10
    """
    t0 = time.time()
    print(
        f"[V92] Building multi-TF composite (st_w={st_weight:.2f})...",
        flush=True)

    mt_weight = 1.0 - st_weight

    # Short-term: dynamic lookback factors (higher weight on adaptive return)
    st_weights = {
        "rank_ret_dyn": 0.30,
        "rank_oi_dyn": 0.20,
        "rank_rsi": 0.15,
        "rank_vol_dyn": 0.10,
        "rank_range_dyn": 0.10,
        "rank_atrp_dyn": 0.05,
        "rank_ret20d": 0.10,
    }

    # Medium-term: static 20d factors
    mt_weights = {
        "rank_ret20d": 0.25,
        "rank_oi20d": 0.20,
        "rank_rsi": 0.15,
        "rank_vol20d": 0.15,
        "rank_range20d": 0.10,
        "rank_atrp20d": 0.05,
        "rank_ret_dyn": 0.10,
    }

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

    print(
        f"  Multi-TF composite done: {time.time() - t0:.1f}s",
        flush=True)
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


def backtest_v92(
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
    max_per_sector: int = 2,
    min_confidence: int = 3,
    use_ker_gate: bool = True,
    hold_days: int = 5,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Backtest V92: vol-adaptive dynamic lookback strategy."""
    composite = sigs["composite"]
    n_confirm_st = sigs["n_confirm_st"]
    n_confirm_mt = sigs["n_confirm_mt"]
    ker_regime = sigs["ker_regime"]

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

        # Entry logic
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
    sigs: Dict[str, np.ndarray],
    sector_lookup: Dict[int, str],
    win_threshold: float = 0.60,
    normal_threshold: float = 0.80,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
    atr_stop: float = 3.0,
    max_positions: int = 3,
    max_per_sector: int = 2,
) -> List[dict]:
    print(f"\n{'=' * 70}")
    print(
        f"  WALK-FORWARD V92 "
        f"(wt={win_threshold:.2f} nt={normal_threshold:.2f} "
        f"lt={lose_threshold:.2f} ww={win_rate_window} "
        f"mp={max_positions} mps={max_per_sector})"
    )
    print(f"  VOL-ADAPTIVE DYNAMIC LOOKBACK (NO LEVERAGE)")
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

        trades, _, _ = backtest_v92(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            win_threshold=win_threshold,
            normal_threshold=normal_threshold,
            lose_threshold=lose_threshold,
            win_rate_window=win_rate_window,
            atr_stop=atr_stop,
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
    print("  V92: VOLATILITY-ADAPTIVE DYNAMIC LOOKBACK STRATEGY")
    print("  Dynamic lookback: short in high-vol, long in low-vol")
    print("  Based on research (2106.08420): +66% Sharpe improvement")
    print("  NO LEVERAGE. CASH0=1,000,000. COMM=0.0005.")
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

    # === 1. Compute rolling volatility and vol ranks ===
    print("\n--- Computing rolling 20d volatility ---")
    vol_20d = compute_rolling_vol(C, NS, ND, vol_window=20)
    vol_rank = compute_vol_rank(vol_20d, NS, ND)

    # === 2. Pre-compute static 20d factors (shared across configs) ===
    print("\n--- Computing static 20d factors ---")
    static_factors = compute_raw_factors_static(C, O, H, L, V, OI, NS, ND)
    ker_regime = compute_ker(C, NS, ND)

    # === 3. Parameter sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("  NO LEVERAGE. Vol-adaptive dynamic lookback.")
    print("=" * 70)

    results: List[dict] = []
    sweep_count = 0

    # Parameters to sweep
    high_vol_pcts = [0.65, 0.70, 0.75]
    low_vol_pcts = [0.25, 0.30, 0.35]
    short_lbs = [2, 3, 5]
    medium_lbs = [8, 10, 12]
    long_lbs = [15, 20, 25]
    st_weights = [0.60, 0.65, 0.70]
    max_positions_list = [2, 3, 4]
    max_per_sector_list = [2, 3]

    # Cache dynamic factors by (high_vol_pct, low_vol_pct, short_lb,
    # medium_lb, long_lb)
    dyn_cache: Dict[tuple, Dict] = {}

    for hvp in high_vol_pcts:
        for lvp in low_vol_pcts:
            if lvp >= hvp:
                continue
            for slb in short_lbs:
                for mlb in medium_lbs:
                    for llb in long_lbs:
                        if slb >= mlb or mlb >= llb:
                            continue
                        cache_key = (hvp, lvp, slb, mlb, llb)
                        if cache_key not in dyn_cache:
                            dyn = compute_dynamic_lookback_returns(
                                C, OI, V, H, L, NS, ND,
                                vol_rank, hvp, lvp, slb, mlb, llb)
                            dyn_cache[cache_key] = dyn
                        dyn = dyn_cache[cache_key]

                        # Merge dynamic + static factors
                        merged = {}
                        merged.update(dyn)
                        for k in ["ret_20d", "oi_20d", "vol_20d",
                                   "range_20d", "atrp_20d", "rsi14"]:
                            merged[k] = static_factors[k]

                        ranks = compute_cross_sectional_ranks(
                            merged, NS, ND)

                        for st_w in st_weights:
                            combined, st_comp, mt_comp, ncf_st, ncf_mt = \
                                build_multi_tf_composite(
                                    ranks, st_w, NS, ND)

                            sigs = {
                                "composite": combined,
                                "st_comp": st_comp,
                                "mt_comp": mt_comp,
                                "n_confirm_st": ncf_st,
                                "n_confirm_mt": ncf_mt,
                                "ker_regime": ker_regime,
                            }

                            for mp in max_positions_list:
                                for mps in max_per_sector_list:
                                    sweep_count += 1
                                    trades, eq, dd = backtest_v92(
                                        C, O, H, L, NS, ND,
                                        dates, syms, sigs,
                                        sector_lookup=sector_lookup,
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
                                        trades[-1]["di"]
                                        - trades[0]["di"])
                                    ann = ((eq / CASH0) ** (
                                        1 / max(
                                            1.0, n_days / 252)) - 1) * 100
                                    ap = [t["pnl_abs"]
                                          for t in sorted(
                                              trades,
                                              key=lambda x: x["di"])]
                                    rets_arr = np.array(ap) / CASH0
                                    sh_val = (
                                        np.mean(rets_arr)
                                        / np.std(rets_arr)
                                        * np.sqrt(252)
                                        if np.std(rets_arr) > 0 else 0)

                                    results.append({
                                        "hvp": hvp, "lvp": lvp,
                                        "slb": slb, "mlb": mlb,
                                        "llb": llb,
                                        "st_w": st_w,
                                        "mp": mp, "mps": mps,
                                        "n": len(trades), "wr": wr,
                                        "ann": ann, "dd": dd,
                                        "sharpe": sh_val, "eq": eq,
                                    })

                                    if sweep_count % 50 == 0:
                                        print(
                                            f"  ... {sweep_count} configs "
                                            f"evaluated, "
                                            f"{len(results)} with 10+ trades",
                                            flush=True)

    results.sort(key=lambda x: -x["ann"])
    print(
        f"\n  Evaluated {sweep_count} configs, "
        f"{len(results)} with 10+ trades")

    # Print top 10 by annualized return
    print(
        f"\n{'HVP':>5} {'LVP':>5} {'SLB':>4} {'MLB':>4} {'LLB':>4} "
        f"{'STw':>4} {'MP':>3} {'MPS':>3} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} "
        f"{'Sh':>6} {'EQ':>12}"
    )
    print("-" * 110)
    for r in results[:10]:
        print(
            f"{r['hvp']:>5.2f} {r['lvp']:>5.2f} {r['slb']:>4} "
            f"{r['mlb']:>4} {r['llb']:>4} "
            f"{r['st_w']:>4.2f} {r['mp']:>3} {r['mps']:>3} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f} "
            f"{r['eq']:>12,.0f}"
        )

    if not results:
        print("  No configs with 10+ trades. Exiting.")
        return

    # === 4. Walk-forward for best config ===
    best = results[0]
    print("\n" + "=" * 70)
    print(
        f"  BEST CONFIG (by ann return): "
        f"hvp={best['hvp']:.2f} lvp={best['lvp']:.2f} "
        f"slb={best['slb']} mlb={best['mlb']} llb={best['llb']} "
        f"st_w={best['st_w']:.2f} mp={best['mp']} mps={best['mps']}"
    )
    print("=" * 70)

    # Rebuild signals for best config
    dyn_best = compute_dynamic_lookback_returns(
        C, OI, V, H, L, NS, ND,
        vol_rank, best["hvp"], best["lvp"],
        best["slb"], best["mlb"], best["llb"])
    merged_best = {}
    merged_best.update(dyn_best)
    for k in ["ret_20d", "oi_20d", "vol_20d",
              "range_20d", "atrp_20d", "rsi14"]:
        merged_best[k] = static_factors[k]
    ranks_best = compute_cross_sectional_ranks(merged_best, NS, ND)
    combined_best, _, _, ncf_st_best, ncf_mt_best = \
        build_multi_tf_composite(
            ranks_best, best["st_w"], NS, ND)
    sigs_best = {
        "composite": combined_best,
        "n_confirm_st": ncf_st_best,
        "n_confirm_mt": ncf_mt_best,
        "ker_regime": ker_regime,
    }

    # Full backtest
    print("\n--- Full backtest (2016-2026) ---")
    trades_full, eq_full, dd_full = backtest_v92(
        C, O, H, L, NS, ND, dates, syms, sigs_best,
        sector_lookup=sector_lookup,
        max_positions=best["mp"],
        max_per_sector=best["mps"],
        start_di=60,
    )
    label = (
        f"hvp={best['hvp']:.2f} lvp={best['lvp']:.2f} "
        f"slb={best['slb']} mlb={best['mlb']} llb={best['llb']} "
        f"st_w={best['st_w']:.2f}"
    )
    analyze(trades_full, eq_full, dd_full, label)

    # Walk-forward
    walk_forward(
        C, O, H, L, NS, ND, dates, syms,
        sigs_best,
        sector_lookup=sector_lookup,
        max_positions=best["mp"],
        max_per_sector=best["mps"],
    )

    # === 5. Top 3 configs: walk-forward ===
    print("\n" + "=" * 70)
    print("  TOP 3 CONFIGS WALK-FORWARD")
    print("=" * 70)

    seen_keys = set()
    unique_top = []
    for r in results:
        key = (r["hvp"], r["lvp"], r["slb"], r["mlb"],
               r["llb"], r["st_w"], r["mp"], r["mps"])
        if key not in seen_keys:
            seen_keys.add(key)
            unique_top.append(r)
        if len(unique_top) >= 3:
            break

    for r in unique_top:
        dyn_r = compute_dynamic_lookback_returns(
            C, OI, V, H, L, NS, ND,
            vol_rank, r["hvp"], r["lvp"],
            r["slb"], r["mlb"], r["llb"])
        merged_r = {}
        merged_r.update(dyn_r)
        for k in ["ret_20d", "oi_20d", "vol_20d",
                   "range_20d", "atrp_20d", "rsi14"]:
            merged_r[k] = static_factors[k]
        ranks_r = compute_cross_sectional_ranks(merged_r, NS, ND)
        combined_r, _, _, ncf_st_r, ncf_mt_r = \
            build_multi_tf_composite(
                ranks_r, r["st_w"], NS, ND)
        sigs_r = {
            "composite": combined_r,
            "n_confirm_st": ncf_st_r,
            "n_confirm_mt": ncf_mt_r,
            "ker_regime": ker_regime,
        }

        print(
            f"\n  #{unique_top.index(r) + 1}: "
            f"hvp={r['hvp']:.2f} lvp={r['lvp']:.2f} "
            f"slb={r['slb']} mlb={r['mlb']} llb={r['llb']} "
            f"st_w={r['st_w']:.2f} mp={r['mp']} mps={r['mps']}")
        walk_forward(
            C, O, H, L, NS, ND, dates, syms,
            sigs_r,
            sector_lookup=sector_lookup,
            max_positions=r["mp"],
            max_per_sector=r["mps"],
        )

    print(f"\n[V92] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
