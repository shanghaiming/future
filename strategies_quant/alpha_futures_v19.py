"""
V19: Sector Momentum + Mean Reversion Hybrid
==============================================
Use sector-level momentum (which sector is trending) + individual mean reversion
(oversold within trending sector). Combines momentum on sector level with MR on
individual level.

Architecture:
1. Sector momentum ranking (20d average return)
   - Top 2 sectors = "hot" (trend continuation)
   - Bottom 2 sectors = "cold" (mean reversion)
2. Hot sectors: buy the strongest individual (trend following)
   Cold sectors: buy the weakest individual (mean reversion / oversold bounce)
3. Signal scoring:
   - Hot: trend_score = EMA10 > EMA30 + ADX > 25
   - Cold: oversold_score = consec_dn + ret5d + OI decline (V1 signals)
4. KER gate per instrument
5. Pyramid on day-1 winners (ratio 0.5)
6. Walk-forward 2019-2026
7. Parameter sweep

Signal at close[di], enter at open[di+1]. No look-ahead. No gap signals.
No leverage.
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

SECTOR_DEFS: Dict[str, List[str]] = {
    "BLACK": ["rbfi", "hcfi", "ifi", "jfi", "jmfi"],
    "METAL": ["cufi", "alfi", "znfi", "nifi", "snfi"],
    "PRECIOUS": ["aufi", "agfi"],
    "ENERGY": ["scfi", "bufi", "fufi", "tafi", "mafi"],
    "CHEM": ["ppfi", "lfi", "vfi", "egfi", "ebfi", "safi"],
    "OILCHAIN": ["mfi", "yfi", "ofi", "pfi", "rmfi"],
    "GRAIN": ["cfi", "csfi", "srfi", "cffi"],
}


# ============================================================
# SECTOR MAPPING
# ============================================================
def build_sector_map(
    syms: List[str],
) -> Tuple[Dict[str, List[int]], Dict[int, str]]:
    """Build symbol-to-sector index mappings."""
    sym_set = set(syms)
    sym_to_idx = {s: i for i, s in enumerate(syms)}
    sector_to_si: Dict[str, List[int]] = {}
    si_to_sector: Dict[int, str] = {}
    for sector, members in SECTOR_DEFS.items():
        present = [m for m in members if m in sym_set]
        if len(present) >= 2:
            indices = [sym_to_idx[m] for m in present]
            sector_to_si[sector] = indices
            for idx in indices:
                si_to_sector[idx] = sector
    return sector_to_si, si_to_sector


# ============================================================
# SECTOR MOMENTUM: rank sectors by 20d average return
# ============================================================
def compute_sector_momentum(
    C: np.ndarray, NS: int, ND: int, syms: List[str],
    mom_period: int = 20,
) -> Tuple[np.ndarray, Dict[str, List[int]], Dict[int, str]]:
    """Compute sector-level momentum ranking.

    Returns:
        sector_rank: (NS, ND) - per-symbol sector rank percentile each day
                     high rank = hot sector (strong momentum)
                     low rank = cold sector (weak momentum)
        sector_to_si: sector -> list of symbol indices
        si_to_sector: symbol index -> sector name
    """
    t0 = time.time()
    print("[V19] Computing sector momentum...", flush=True)

    sector_to_si, si_to_sector = build_sector_map(syms)
    sector_names = sorted(sector_to_si.keys())

    # Per-symbol 20d return
    sym_ret = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(mom_period, ND):
            if (
                not np.isnan(C[si, di])
                and not np.isnan(C[si, di - mom_period])
                and C[si, di - mom_period] > 0
            ):
                sym_ret[si, di] = C[si, di] / C[si, di - mom_period] - 1

    # Sector average return per day
    sector_rank = np.full((NS, ND), np.nan)
    for di in range(mom_period, ND):
        sector_avg_ret: Dict[str, float] = {}
        for sector, indices in sector_to_si.items():
            rets = sym_ret[indices, di]
            valid = rets[~np.isnan(rets)]
            if len(valid) >= 2:
                sector_avg_ret[sector] = float(np.mean(valid))

        if len(sector_avg_ret) < 3:
            continue

        # Rank sectors: high rank = high return (hot)
        s = pd.Series(sector_avg_ret)
        ranked = s.rank(pct=True)
        for sector, rank_val in ranked.items():
            for si in sector_to_si[sector]:
                sector_rank[si, di] = rank_val

    print(
        f"  {len(sector_names)} sectors, done: {time.time() - t0:.1f}s", flush=True
    )
    return sector_rank, sector_to_si, si_to_sector


# ============================================================
# HOT SIGNAL: Trend following within hot sectors
# ============================================================
def compute_hot_signals(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, NS: int, ND: int,
) -> Dict[str, np.ndarray]:
    """Trend-following signals for hot sectors.

    trend_score:
    - EMA10 > EMA30 (short-term above long-term)
    - ADX > 25 (trending)
    - 5d return positive (confirming direction)
    """
    t0 = time.time()
    print("[V19] Computing hot (trend) signals...", flush=True)

    # EMA10 and EMA30
    ema10 = np.full((NS, ND), np.nan)
    ema30 = np.full((NS, ND), np.nan)
    for si in range(NS):
        alpha10 = 2.0 / (10 + 1)
        alpha30 = 2.0 / (30 + 1)
        e10 = np.nan
        e30 = np.nan
        for di in range(ND):
            if np.isnan(C[si, di]):
                continue
            if np.isnan(e10):
                e10 = C[si, di]
                e30 = C[si, di]
            else:
                e10 = alpha10 * C[si, di] + (1 - alpha10) * e10
                e30 = alpha30 * C[si, di] + (1 - alpha30) * e30
            ema10[si, di] = e10
            ema30[si, di] = e30

    # ADX
    adx = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            h = np.where(np.isnan(H[si]), 0, H[si]).astype(np.float64)
            l = np.where(np.isnan(L[si]), 0, L[si]).astype(np.float64)
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nan_mask = np.isnan(C[si])
            try:
                adx_vals = talib.ADX(h, l, c, 14)
                adx[si] = np.where(nan_mask, np.nan, adx_vals)
            except Exception:
                pass

    # 5d return
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (
                not np.isnan(C[si, di])
                and not np.isnan(C[si, di - 5])
                and C[si, di - 5] > 0
            ):
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1

    # 20d return (momentum)
    ret_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if (
                not np.isnan(C[si, di])
                and not np.isnan(C[si, di - 20])
                and C[si, di - 20] > 0
            ):
                ret_20d[si, di] = C[si, di] / C[si, di - 20] - 1

    # Composite trend score (0-1, higher = stronger trend buy)
    trend_score = np.full((NS, ND), np.nan)
    for di in range(30, ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue

            s_val = 0.0
            w_total = 0.0

            # EMA crossover: EMA10 > EMA30 (40% weight)
            e10_val = ema10[si, di]
            e30_val = ema30[si, di]
            if not np.isnan(e10_val) and not np.isnan(e30_val) and e30_val > 0:
                ema_ratio = e10_val / e30_val
                if ema_ratio > 1.0:
                    s_val += min((ema_ratio - 1.0) / 0.05, 1.0) * 0.40
                w_total += 0.40

            # ADX > 25 (30% weight)
            adx_val = adx[si, di]
            if not np.isnan(adx_val):
                if adx_val > 25:
                    s_val += min((adx_val - 25) / 25.0, 1.0) * 0.30
                w_total += 0.30

            # Positive 5d return (15% weight)
            r5 = ret_5d[si, di]
            if not np.isnan(r5):
                if r5 > 0:
                    s_val += min(r5 / 0.05, 1.0) * 0.15
                w_total += 0.15

            # Positive 20d return (15% weight)
            r20 = ret_20d[si, di]
            if not np.isnan(r20):
                if r20 > 0:
                    s_val += min(r20 / 0.10, 1.0) * 0.15
                w_total += 0.15

            if w_total > 0:
                scores[si] = s_val / w_total

        valid = ~np.isnan(scores)
        if valid.sum() >= 5:
            trend_score[:, di] = (
                pd.Series(scores).rank(pct=True, na_option="keep").values
            )

    print(f"  Hot signals done: {time.time() - t0:.1f}s", flush=True)
    return {"trend_score": trend_score, "ret_5d": ret_5d, "adx": adx}


# ============================================================
# COLD SIGNAL: Mean reversion within cold sectors (V1 signals)
# ============================================================
def compute_cold_signals(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
) -> Dict[str, np.ndarray]:
    """Mean-reversion / oversold signals for cold sectors.

    oversold_score:
    - Consecutive down days
    - 5d return deeply negative
    - OI decline (capitulation)
    - RSI < 30
    - BB position < 0.2
    """
    t0 = time.time()
    print("[V19] Computing cold (MR) signals...", flush=True)

    # Consecutive down days
    consec_dn = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        consec = 0
        for di in range(1, ND):
            if (
                not np.isnan(C[si, di])
                and not np.isnan(C[si, di - 1])
                and C[si, di - 1] > 0
            ):
                consec = consec + 1 if C[si, di] < C[si, di - 1] else 0
            else:
                consec = 0
            consec_dn[si, di] = consec

    # 5-day return
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (
                not np.isnan(C[si, di])
                and not np.isnan(C[si, di - 5])
                and C[si, di - 5] > 0
            ):
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1

    # OI decline signal
    oi_decline = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if np.isnan(OI[si, di]) or np.isnan(OI[si, di - 5]):
                continue
            if np.isnan(C[si, di]) or np.isnan(C[si, di - 5]) or C[si, di - 5] <= 0:
                continue
            oi_chg = OI[si, di] / OI[si, di - 5] - 1
            price_chg = C[si, di] / C[si, di - 5] - 1
            if oi_chg < -0.02 and price_chg < -0.02:
                oi_decline[si, di] = (
                    min(abs(oi_chg), 0.2) / 0.2 * min(abs(price_chg), 0.1) / 0.1
                )
            else:
                oi_decline[si, di] = 0.0

    # VDP exhaustion
    vdp = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if any(np.isnan([H[si, di], L[si, di], C[si, di], V[si, di]])):
                continue
            bar_range = H[si, di] - L[si, di]
            if bar_range > 0 and V[si, di] > 0:
                vdp[si, di] = (
                    V[si, di] * (2 * C[si, di] - H[si, di] - L[si, di]) / bar_range
                )

    vdp_10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            vals = vdp[si, di - 10 : di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 5:
                vdp_10[si, di] = np.mean(valid)

    vdp_exhaust = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if np.isnan(vdp_10[si, di]):
                continue
            window = vdp_10[si, max(0, di - 20) : di]
            wv = window[~np.isnan(window)]
            if len(wv) >= 10:
                mu, sig = np.mean(wv), np.std(wv)
                if sig > 0:
                    z = (vdp_10[si, di] - mu) / sig
                    vdp_exhaust[si, di] = min(-z, 3.0) / 3.0 if z < 0 else 0.0

    # RSI, BB via TA-Lib
    rsi14 = np.full((NS, ND), np.nan)
    bb_pos = np.full((NS, ND), np.nan)

    if HAS_TALIB:
        for si in range(NS):
            h = np.where(np.isnan(H[si]), 0, H[si]).astype(np.float64)
            l = np.where(np.isnan(L[si]), 0, L[si]).astype(np.float64)
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nan_mask = np.isnan(C[si])
            try:
                rsi = talib.RSI(c, 14)
                rsi14[si] = np.where(nan_mask, np.nan, rsi)
            except Exception:
                pass
            try:
                upper, mid, lower = talib.BBANDS(c, 20, 2.0, 2.0)
                bb_range = upper - lower
                valid_bb = (~nan_mask) & (bb_range > 1e-10)
                bb_pos[si] = np.where(valid_bb, (c - lower) / bb_range, np.nan)
            except Exception:
                pass

    # Composite oversold score (0-1, higher = more oversold)
    raw_score = np.full((NS, ND), np.nan)
    for di in range(ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            s = 0.0
            w_total = 0.0

            s += min(consec_dn[si, di] / 5.0, 1.0) * 0.20
            w_total += 0.20

            if not np.isnan(ret_5d[si, di]):
                s += min(max(-ret_5d[si, di] / 0.1, 0), 1.0) * 0.20
                w_total += 0.20

            if not np.isnan(oi_decline[si, di]):
                s += oi_decline[si, di] * 0.20
                w_total += 0.20

            if not np.isnan(vdp_exhaust[si, di]):
                s += vdp_exhaust[si, di] * 0.15
                w_total += 0.15

            if not np.isnan(rsi14[si, di]):
                if rsi14[si, di] < 30:
                    s += (30 - rsi14[si, di]) / 30.0 * 0.15
                w_total += 0.15

            if not np.isnan(bb_pos[si, di]):
                if bb_pos[si, di] < 0.2:
                    s += (0.2 - bb_pos[si, di]) / 0.2 * 0.10
                w_total += 0.10

            if w_total > 0:
                scores[si] = s / w_total

        valid = ~np.isnan(scores)
        if valid.sum() >= 5:
            raw_score[:, di] = (
                pd.Series(scores).rank(pct=True, na_option="keep").values
            )

    # Signal count
    n_signals = np.zeros((NS, ND), dtype=int)
    for di in range(ND):
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            n = 0
            if consec_dn[si, di] >= 3:
                n += 1
            if not np.isnan(ret_5d[si, di]) and ret_5d[si, di] < -0.03:
                n += 1
            if not np.isnan(oi_decline[si, di]) and oi_decline[si, di] > 0.1:
                n += 1
            if not np.isnan(vdp_exhaust[si, di]) and vdp_exhaust[si, di] > 0.3:
                n += 1
            if not np.isnan(rsi14[si, di]) and rsi14[si, di] < 35:
                n += 1
            if not np.isnan(bb_pos[si, di]) and bb_pos[si, di] < 0.15:
                n += 1
            n_signals[si, di] = n

    print(f"  Cold signals done: {time.time() - t0:.1f}s", flush=True)
    return {
        "oversold_score": raw_score,
        "n_signals": n_signals,
        "ret_5d": ret_5d,
        "rsi14": rsi14,
    }


# ============================================================
# KER GATE (Kaufman Efficiency Ratio)
# ============================================================
def compute_ker(C: np.ndarray, NS: int, ND: int) -> np.ndarray:
    """Kaufman Efficiency Ratio for regime gating."""
    ker_10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            closes = C[si, di - 10 : di + 1]
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
                ker_regime[si, di] = 1  # sideways -> good for MR
            elif ker_10[si, di] > 0.3:
                ker_regime[si, di] = -1  # trending -> avoid counter-trend MR
    return ker_regime


# ============================================================
# HYBRID SIGNAL COMBINATION
# ============================================================
def combine_signals(
    C: np.ndarray,
    sector_rank: np.ndarray,
    hot_sigs: Dict[str, np.ndarray],
    cold_sigs: Dict[str, np.ndarray],
    ker_regime: np.ndarray,
    sector_to_si: Dict[str, List[int]],
    NS: int,
    ND: int,
    hot_threshold: float = 0.70,
    cold_threshold: float = 0.30,
) -> Dict[str, np.ndarray]:
    """Combine hot (trend) and cold (MR) signals based on sector momentum rank.

    sector_rank > hot_threshold -> hot sector -> use trend_score
    sector_rank < cold_threshold -> cold sector -> use oversold_score
    """
    t0 = time.time()
    print("[V19] Combining signals...", flush=True)

    trend_score = hot_sigs["trend_score"]
    oversold_score = cold_sigs["oversold_score"]
    n_signals = cold_sigs["n_signals"]

    # Final hybrid score: high = buy candidate
    hybrid_score = np.full((NS, ND), np.nan)
    signal_type = np.zeros((NS, ND), dtype=int)  # 1=hot, 2=cold, 0=neutral

    for di in range(30, ND):
        for si in range(NS):
            if np.isnan(C[si, di]) or np.isnan(sector_rank[si, di]):
                continue

            sr = sector_rank[si, di]

            if sr > hot_threshold:
                # Hot sector -> use trend score
                ts = trend_score[si, di]
                if not np.isnan(ts) and ts > 0.5:
                    hybrid_score[si, di] = ts
                    signal_type[si, di] = 1

            elif sr < cold_threshold:
                # Cold sector -> use oversold score (MR)
                os_score = oversold_score[si, di]
                if not np.isnan(os_score) and os_score > 0.6:
                    # KER gate: avoid MR in strong trend
                    if ker_regime[si, di] >= 0:
                        hybrid_score[si, di] = os_score
                        signal_type[si, di] = 2

    # Count how many hot/cold signals per day
    n_hot = np.zeros(ND, dtype=int)
    n_cold = np.zeros(ND, dtype=int)
    for di in range(ND):
        n_hot[di] = np.sum(signal_type[:, di] == 1)
        n_cold[di] = np.sum(signal_type[:, di] == 2)

    print(
        f"  Avg hot/day: {np.mean(n_hot[30:]):.1f}, "
        f"avg cold/day: {np.mean(n_cold[30:]):.1f}, "
        f"done: {time.time() - t0:.1f}s",
        flush=True,
    )
    return {
        "hybrid_score": hybrid_score,
        "signal_type": signal_type,
        "n_signals": n_signals,
        "n_hot": n_hot,
        "n_cold": n_cold,
    }


def compute_all_signals(
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    V: np.ndarray,
    OI: np.ndarray,
    NS: int,
    ND: int,
    syms: List[str],
    hot_threshold: float = 0.70,
    cold_threshold: float = 0.30,
) -> Dict[str, np.ndarray]:
    """Compute all signals for V19."""
    t0 = time.time()
    print("[V19] Computing all signals...", flush=True)

    sector_rank, sector_to_si, si_to_sector = compute_sector_momentum(
        C, NS, ND, syms
    )
    hot_sigs = compute_hot_signals(C, O, H, L, V, NS, ND)
    cold_sigs = compute_cold_signals(C, O, H, L, V, OI, NS, ND)
    ker_regime = compute_ker(C, NS, ND)

    hybrid = combine_signals(
        C,
        sector_rank,
        hot_sigs,
        cold_sigs,
        ker_regime,
        sector_to_si,
        NS,
        ND,
        hot_threshold=hot_threshold,
        cold_threshold=cold_threshold,
    )

    print(f"  All signals done: {time.time() - t0:.1f}s", flush=True)
    return {
        "sector_rank": sector_rank,
        "hybrid_score": hybrid["hybrid_score"],
        "signal_type": hybrid["signal_type"],
        "n_signals": hybrid["n_signals"],
        "n_hot": hybrid["n_hot"],
        "n_cold": hybrid["n_cold"],
        "ker_regime": ker_regime,
        "sector_to_si": sector_to_si,
        "si_to_sector": si_to_sector,
    }


# ============================================================
# BACKTEST ENGINE
# ============================================================
def backtest_v19(
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    NS: int,
    ND: int,
    dates: np.ndarray,
    syms: List[str],
    sigs: Dict[str, np.ndarray],
    top_n: int = 1,
    min_rank: float = 0.65,
    atr_stop: float = 3.0,
    min_confidence: int = 2,
    use_ker_gate: bool = True,
    hold_days: int = 5,
    pyramid_ratio: float = 0.5,
    pyramid_day: int = 1,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Day-by-day backtest with hybrid signals + pyramid."""
    hybrid_score = sigs["hybrid_score"]
    signal_type = sigs["signal_type"]
    ker_regime = sigs["ker_regime"]
    n_signals = sigs["n_signals"]

    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions: List[Tuple[int, int, float, float, float, bool]] = []
    trades: List[dict] = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions: List[Tuple[int, int, float, float, float, bool]] = []

        pos_by_si: Dict[int, List[Tuple[int, float, float, float, bool]]] = (
            defaultdict(list)
        )
        for si, edi, ep, sp, alloc, is_pyr in positions:
            pos_by_si[si].append((edi, ep, sp, alloc, is_pyr))

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
                    trades.append(
                        {
                            "pnl_abs": profit,
                            "pnl_pct": pnl * 100,
                            "days": di - edi + 1,
                            "di": di,
                            "year": d.year,
                            "sym": syms[si],
                            "reason": "stop",
                            "stype": signal_type[si, edi],
                            "pyr": is_pyr,
                        }
                    )
            elif hold >= hold_days:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    trades.append(
                        {
                            "pnl_abs": profit,
                            "pnl_pct": pnl * 100,
                            "days": di - edi + 1,
                            "di": di,
                            "year": d.year,
                            "sym": syms[si],
                            "reason": "hold",
                            "stype": signal_type[si, edi],
                            "pyr": is_pyr,
                        }
                    )
            else:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc, is_pyr))

        # Pyramid on day-1 winners
        if pyramid_ratio > 0:
            held_with_pos: Dict[int, List[Tuple[int, float, float, float, bool]]] = (
                defaultdict(list)
            )
            for si, edi, ep, sp, alloc, is_pyr in new_positions:
                held_with_pos[si].append((edi, ep, sp, alloc, is_pyr))

            additions = []
            for si, pos_list in held_with_pos.items():
                has_pyr = any(is_pyr for _, _, _, _, is_pyr in pos_list)
                if has_pyr:
                    continue
                earliest_edi = min(p[0] for p in pos_list)
                hold = di - earliest_edi
                if hold == pyramid_day and not np.isnan(C[si, di]):
                    avg_ep = np.mean([ep for _, ep, _, _, _ in pos_list])
                    if C[si, di] > avg_ep:
                        base_alloc = sum(a for _, _, _, a, _ in pos_list)
                        pyr_alloc = base_alloc * pyramid_ratio
                        c_now = C[si, di]
                        atr_v = []
                        for j in range(max(start_di, di - 14), di):
                            hh, ll, cc = H[si, j], L[si, j], C[si, j]
                            if not any(np.isnan([hh, ll, cc])):
                                atr_v.append(
                                    max(hh - ll, abs(hh - cc), abs(ll - cc))
                                )
                        if atr_v:
                            atr = np.mean(atr_v)
                            additions.append(
                                (
                                    si,
                                    di,
                                    c_now,
                                    c_now - atr_stop * atr,
                                    pyr_alloc,
                                    True,
                                )
                            )
            new_positions.extend(additions)

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

        held = {p[0] for p in positions}
        if len(positions) >= top_n:
            continue

        # Entry: signal at close[di], enter at open[di+1]
        candidates = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(hybrid_score[si, di]):
                continue
            if hybrid_score[si, di] < min_rank:
                continue

            # For cold (MR) signals, require minimum confidence
            if signal_type[si, di] == 2:
                if n_signals[si, di] < min_confidence:
                    continue
                if use_ker_gate and ker_regime[si, di] < 0:
                    continue

            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue

            alloc = 1.0 / max(top_n, 1)
            candidates.append((hybrid_score[si, di], si, alloc))

        candidates.sort(key=lambda x: -x[0])
        for rank_val, si, alloc in candidates[:top_n]:
            if len(positions) >= top_n or si in held:
                break
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr_v = []
            for j in range(max(start_di, di - 14), di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
            if not atr_v:
                continue
            atr = np.mean(atr_v)
            positions.append((si, di + 1, ep, ep - atr_stop * atr, alloc, False))
            held.add(si)

    # Close remaining
    for si, edi, ep, sp, alloc, is_pyr in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl

    return trades, equity, max_dd


# ============================================================
# ANALYSIS
# ============================================================
def analyze(
    trades: List[dict], equity: float, max_dd: float, label: str = ""
) -> Optional[dict]:
    """Print analysis and return summary dict."""
    if not trades:
        print(f"  {label}: no trades")
        return None

    nw = sum(1 for t in trades if t["pnl_pct"] > 0)
    wr = nw / len(trades) * 100
    n_days = max(1, trades[-1]["di"] - trades[0]["di"])
    ann = ((equity / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
    ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
    rets = np.array(ap) / CASH0
    sh = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0

    n_hot = sum(1 for t in trades if t.get("stype") == 1)
    n_cold = sum(1 for t in trades if t.get("stype") == 2)
    n_pyr = sum(1 for t in trades if t.get("pyr"))
    n_base = len(trades) - n_pyr
    n_stop = sum(1 for t in trades if t["reason"] == "stop")
    n_hold = sum(1 for t in trades if t["reason"] == "hold")

    # Per-type WR
    hot_trades = [t for t in trades if t.get("stype") == 1]
    cold_trades = [t for t in trades if t.get("stype") == 2]
    hot_wr = (
        sum(1 for t in hot_trades if t["pnl_pct"] > 0) / len(hot_trades) * 100
        if hot_trades
        else 0
    )
    cold_wr = (
        sum(1 for t in cold_trades if t["pnl_pct"] > 0) / len(cold_trades) * 100
        if cold_trades
        else 0
    )

    print(
        f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} stop:{n_stop} hold:{n_hold}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% Sh={sh:.2f} eq={equity:,.0f}"
    )
    print(
        f"    Hot(trend): {n_hot}t WR={hot_wr:.1f}% | Cold(MR): {n_cold}t WR={cold_wr:.1f}%"
    )

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
            f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% cum={cum:+.1%}"
        )

    return {"n": len(trades), "wr": wr, "dd": max_dd, "ann": ann, "sh": sh, "eq": equity}


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================
def walk_forward(
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    V: np.ndarray,
    OI: np.ndarray,
    NS: int,
    ND: int,
    dates: np.ndarray,
    syms: List[str],
    top_n: int = 1,
    min_confidence: int = 2,
    hold_days: int = 5,
    atr_stop: float = 3.0,
    pyramid_ratio: float = 0.5,
    hot_threshold: float = 0.70,
    cold_threshold: float = 0.30,
) -> List[dict]:
    """Walk-forward: compute signals once, test year-by-year OOS."""
    print(f"\n{'=' * 70}")
    print(
        f"  WALK-FORWARD V19 (tn={top_n}, pyr={pyramid_ratio}, "
        f"hot>={hot_threshold}, cold<={cold_threshold})"
    )
    print(f"{'=' * 70}")

    # Compute signals once on all data (no look-ahead in signal design)
    sigs = compute_all_signals(
        C, O, H, L, V, OI, NS, ND, syms,
        hot_threshold=hot_threshold,
        cold_threshold=cold_threshold,
    )

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

        trades, _, _ = backtest_v19(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=top_n, hold_days=hold_days, atr_stop=atr_stop,
            min_confidence=min_confidence, use_ker_gate=True,
            pyramid_ratio=pyramid_ratio, pyramid_day=1,
            start_di=test_start, end_di=test_end_idx + 1,
        )

        test_trades = [t for t in trades if dates[t["di"]].year == test_year]
        all_trades.extend(test_trades)

        if test_trades:
            n = len(test_trades)
            nw = sum(1 for t in test_trades if t["pnl_pct"] > 0)
            wr_val = nw / n * 100
            avg = np.mean([t["pnl_pct"] for t in test_trades])
            n_hot = sum(1 for t in test_trades if t.get("stype") == 1)
            n_cold = sum(1 for t in test_trades if t.get("stype") == 2)
            print(
                f"  {test_year}: {n}t (H:{n_hot} C:{n_cold}) "
                f"WR={wr_val:.1f}% avg={avg:+.2f}%",
                flush=True,
            )
        else:
            print(f"  {test_year}: no trades", flush=True)

    if all_trades:
        nw = sum(1 for t in all_trades if t["pnl_pct"] > 0)
        wr_val = nw / len(all_trades) * 100
        avg = np.mean([t["pnl_pct"] for t in all_trades])
        cum = np.prod([1 + t["pnl_pct"] / 100 for t in all_trades]) - 1
        n_hot = sum(1 for t in all_trades if t.get("stype") == 1)
        n_cold = sum(1 for t in all_trades if t.get("stype") == 2)
        print(
            f"\n  WF TOTAL: {len(all_trades)}t (H:{n_hot} C:{n_cold}) "
            f"WR={wr_val:.1f}% avg={avg:+.2f}% cum={cum:+.1%}"
        )
        return all_trades
    return []


# ============================================================
# PARAMETER SWEEP
# ============================================================
def sweep(
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    V: np.ndarray,
    OI: np.ndarray,
    NS: int,
    ND: int,
    dates: np.ndarray,
    syms: List[str],
    start_di: int = 60,
) -> List[dict]:
    """Sweep over parameters to find best configuration.

    Optimized: compute raw signals once, only re-combine per threshold pair.
    """
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (V19)")
    print("=" * 70)

    # Compute raw signals once (expensive part)
    print("[Sweep] Computing raw signals once...", flush=True)
    sector_rank, sector_to_si, si_to_sector = compute_sector_momentum(
        C, NS, ND, syms
    )
    hot_sigs = compute_hot_signals(C, O, H, L, V, NS, ND)
    cold_sigs = compute_cold_signals(C, O, H, L, V, OI, NS, ND)
    ker_regime = compute_ker(C, NS, ND)
    print("[Sweep] Raw signals cached. Running sweep...", flush=True)

    results: List[dict] = []

    for hot_thr in [0.65, 0.70, 0.75]:
        for cold_thr in [0.25, 0.30, 0.35]:
            if hot_thr <= cold_thr:
                continue
            # Only re-combine (cheap ~0.5s)
            hybrid = combine_signals(
                C, sector_rank, hot_sigs, cold_sigs, ker_regime,
                sector_to_si, NS, ND,
                hot_threshold=hot_thr, cold_threshold=cold_thr,
            )
            sigs = {
                "sector_rank": sector_rank,
                "hybrid_score": hybrid["hybrid_score"],
                "signal_type": hybrid["signal_type"],
                "n_signals": hybrid["n_signals"],
                "n_hot": hybrid["n_hot"],
                "n_cold": hybrid["n_cold"],
                "ker_regime": ker_regime,
                "sector_to_si": sector_to_si,
                "si_to_sector": si_to_sector,
            }
            for tn in [1, 2, 3]:
                for mc in [2, 3]:
                    for atr in [2.5, 3.0, 3.5]:
                        for pyr in [0.0, 0.5]:
                            trades, eq, dd = backtest_v19(
                                C, O, H, L, NS, ND, dates, syms, sigs,
                                top_n=tn, min_rank=0.65,
                                atr_stop=atr, min_confidence=mc,
                                use_ker_gate=True, hold_days=5,
                                pyramid_ratio=pyr, pyramid_day=1,
                                start_di=start_di,
                            )
                            if len(trades) < 10:
                                continue
                            nw = sum(1 for t in trades if t["pnl_pct"] > 0)
                            wr = nw / len(trades) * 100
                            n_days = max(
                                1, trades[-1]["di"] - trades[0]["di"]
                            )
                            ann = (
                                (eq / CASH0)
                                ** (1 / max(1.0, n_days / 252))
                                - 1
                            ) * 100
                            ap = [
                                t["pnl_abs"]
                                for t in sorted(trades, key=lambda x: x["di"])
                            ]
                            rets_arr = np.array(ap) / CASH0
                            sh_val = (
                                np.mean(rets_arr)
                                / np.std(rets_arr)
                                * np.sqrt(252)
                                if np.std(rets_arr) > 0
                                else 0
                            )
                            results.append(
                                {
                                    "hot": hot_thr,
                                    "cold": cold_thr,
                                    "tn": tn,
                                    "mc": mc,
                                    "atr": atr,
                                    "pyr": pyr,
                                    "n": len(trades),
                                    "wr": wr,
                                    "ann": ann,
                                    "dd": dd,
                                    "sharpe": sh_val,
                                }
                            )

    results.sort(key=lambda x: -x["sharpe"])
    print(
        f"\n{'Hot':>4} {'Cold':>4} {'TN':>3} {'MC':>3} "
        f"{'ATR':>4} {'Pyr':>4} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}"
    )
    print("-" * 70)
    for r in results[:25]:
        print(
            f"{r['hot']:>4.2f} {r['cold']:>4.2f} {r['tn']:>3} {r['mc']:>3} "
            f"{r['atr']:>4.1f} {r['pyr']:>4.1f} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>5.2f}"
        )

    return results


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V19: SECTOR MOMENTUM + MEAN REVERSION HYBRID")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start="2016-01-01")
    print(
        f"  {NS} sym, {ND} days, "
        f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}"
    )

    # Print sector coverage
    sector_to_si, si_to_sector = build_sector_map(syms)
    for sector in sorted(SECTOR_DEFS.keys()):
        members = SECTOR_DEFS[sector]
        covered = [m for m in members if m in set(syms)]
        print(f"  {sector:>10}: {len(covered)} symbols - {', '.join(covered)}")

    # === 1. Default config full backtest ===
    print("\n" + "=" * 70)
    print("  FULL 10-YEAR (default config)")
    print("=" * 70)

    sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND, syms)

    for tn in [1, 2, 3]:
        for pyr in [0.0, 0.5]:
            trades, eq, dd = backtest_v19(
                C, O, H, L, NS, ND, dates, syms, sigs,
                top_n=tn, hold_days=5, atr_stop=3.0,
                min_confidence=2, use_ker_gate=True,
                pyramid_ratio=pyr, pyramid_day=1,
                start_di=60,
            )
            label = f"tn={tn}-pyr={pyr:.1f}"
            print(f"\n  {label}")
            analyze(trades, eq, dd, label)

    # === 2. 2019+ OOS ===
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    print("\n" + "=" * 70)
    print("  2019-2026 OOS")
    print("=" * 70)

    for tn in [1, 2, 3]:
        for pyr in [0.0, 0.5]:
            trades, eq, dd = backtest_v19(
                C, O, H, L, NS, ND, dates, syms, sigs,
                top_n=tn, hold_days=5, atr_stop=3.0,
                min_confidence=2, use_ker_gate=True,
                pyramid_ratio=pyr, pyramid_day=1,
                start_di=bt_2019,
            )
            label = f"OOS-tn={tn}-pyr={pyr:.1f}"
            print(f"\n  {label}")
            analyze(trades, eq, dd, label)

    # === 3. Parameter sweep ===
    results = sweep(
        C, O, H, L, V, OI, NS, ND, dates, syms, start_di=bt_2019
    )

    # === 4. Best config full 10-year ===
    if results:
        print("\n" + "=" * 70)
        print("  BEST CONFIG — FULL 10-YEAR")
        print("=" * 70)

        # Compute raw signals once for best configs
        sector_rank, sector_to_si, si_to_sector = compute_sector_momentum(
            C, NS, ND, syms
        )
        hot_sigs_best = compute_hot_signals(C, O, H, L, V, NS, ND)
        cold_sigs_best = compute_cold_signals(C, O, H, L, V, OI, NS, ND)
        ker_regime_best = compute_ker(C, NS, ND)

        for r in results[:5]:
            hybrid_best = combine_signals(
                C, sector_rank, hot_sigs_best, cold_sigs_best, ker_regime_best,
                sector_to_si, NS, ND,
                hot_threshold=r["hot"], cold_threshold=r["cold"],
            )
            sigs_best = {
                "sector_rank": sector_rank,
                "hybrid_score": hybrid_best["hybrid_score"],
                "signal_type": hybrid_best["signal_type"],
                "n_signals": hybrid_best["n_signals"],
                "n_hot": hybrid_best["n_hot"],
                "n_cold": hybrid_best["n_cold"],
                "ker_regime": ker_regime_best,
                "sector_to_si": sector_to_si,
                "si_to_sector": si_to_sector,
            }
            trades, eq, dd = backtest_v19(
                C, O, H, L, NS, ND, dates, syms, sigs_best,
                top_n=r["tn"], min_rank=0.65,
                atr_stop=r["atr"], min_confidence=r["mc"],
                use_ker_gate=True, hold_days=5,
                pyramid_ratio=r["pyr"], pyramid_day=1,
                start_di=60,
            )
            label = (
                f"hot={r['hot']:.2f} cold={r['cold']:.2f} "
                f"tn={r['tn']} mc={r['mc']} atr={r['atr']} pyr={r['pyr']:.1f}"
            )
            print(f"\n  FULL {label}")
            analyze(trades, eq, dd, label)

    # === 5. Walk-forward for best config ===
    if results:
        best = results[0]
        print("\n" + "=" * 70)
        print(
            f"  WALK-FORWARD BEST: hot={best['hot']:.2f} cold={best['cold']:.2f} "
            f"tn={best['tn']} mc={best['mc']} atr={best['atr']} pyr={best['pyr']:.1f}"
        )
        print("=" * 70)

        walk_forward(
            C, O, H, L, V, OI, NS, ND, dates, syms,
            top_n=best["tn"],
            min_confidence=best["mc"],
            hold_days=5,
            atr_stop=best["atr"],
            pyramid_ratio=best["pyr"],
            hot_threshold=best["hot"],
            cold_threshold=best["cold"],
        )

    print(f"\n[V19] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
