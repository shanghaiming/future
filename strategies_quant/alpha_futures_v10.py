"""
V10: Sector Rotation + Pairs Trading
=====================================
Instead of trading individual commodities, exploit SECTOR-level patterns.

Key ideas:
1. Commodity sectors with hierarchical selection
2. Sector-level signals: average oversold score, momentum rank, OI trend
3. Two-layer selection: right sector + right commodity
4. Pairs dimension: sector-relative score for mean reversion
5. Three modes: sector rotation, pure individual, combined
6. Walk-forward validation

Signal at close[di], enter at open[di+1]. No look-ahead. No gap signals.
No leverage.
"""
import sys
import os
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from collections import defaultdict

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
# INDIVIDUAL OVERSOLD SIGNALS (from V1/V5, battle-tested)
# ============================================================
def compute_individual_signals(
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    V: np.ndarray,
    OI: np.ndarray,
    NS: int,
    ND: int,
) -> Dict[str, np.ndarray]:
    """Compute per-symbol oversold signals. Returns dict of arrays (NS, ND)."""
    t0 = time.time()
    print("[V10] Computing individual signals...", flush=True)

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

    # RSI, BB, CCI via TA-Lib
    rsi14 = np.full((NS, ND), np.nan)
    bb_pos = np.full((NS, ND), np.nan)
    cci14 = np.full((NS, ND), np.nan)

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
            try:
                cci = talib.CCI(h, l, c, 14)
                cci14[si] = np.where(nan_mask, np.nan, cci)
            except Exception:
                pass

    # Composite oversold score per symbol per day (0=not oversold, 1=max oversold)
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
                    s += (30 - rsi14[si, di]) / 30.0 * 0.10
                w_total += 0.10
            if not np.isnan(bb_pos[si, di]):
                if bb_pos[si, di] < 0.2:
                    s += (0.2 - bb_pos[si, di]) / 0.2 * 0.10
                w_total += 0.10
            if not np.isnan(cci14[si, di]):
                if cci14[si, di] < -100:
                    s += min((-100 - cci14[si, di]) / 200.0, 1.0) * 0.05
                w_total += 0.05
            if w_total > 0:
                scores[si] = s / w_total
        valid = ~np.isnan(scores)
        if valid.sum() >= 5:
            raw_score[:, di] = (
                pd.Series(scores).rank(pct=True, na_option="keep").values
            )

    # Signal count per symbol per day
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
            if not np.isnan(cci14[si, di]) and cci14[si, di] < -100:
                n += 1
            n_signals[si, di] = n

    print(f"  Done: {time.time() - t0:.1f}s", flush=True)
    return {
        "combo_rank": raw_score,
        "n_signals": n_signals,
        "ret_5d": ret_5d,
        "rsi14": rsi14,
        "bb_pos": bb_pos,
    }


# ============================================================
# SECTOR-LEVEL SIGNALS
# ============================================================
def build_sector_map(
    syms: List[str],
) -> Tuple[Dict[str, List[int]], Dict[int, str], Dict[str, List[str]]]:
    """Build symbol-to-sector mappings.

    Returns:
        sector_to_si: sector_name -> list of symbol indices
        si_to_sector: symbol_index -> sector_name
        valid_sectors: sector_name -> list of symbol names (only those present)
    """
    sym_set = set(syms)
    sym_to_idx = {s: i for i, s in enumerate(syms)}

    sector_to_si: Dict[str, List[int]] = {}
    si_to_sector: Dict[int, str] = {}
    valid_sectors: Dict[str, List[str]] = {}

    for sector, members in SECTOR_DEFS.items():
        present = [m for m in members if m in sym_set]
        if len(present) >= 2:
            indices = [sym_to_idx[m] for m in present]
            sector_to_si[sector] = indices
            valid_sectors[sector] = present
            for idx in indices:
                si_to_sector[idx] = sector

    return sector_to_si, si_to_sector, valid_sectors


def compute_sector_signals(
    combo_rank: np.ndarray,
    n_signals: np.ndarray,
    OI: np.ndarray,
    C: np.ndarray,
    NS: int,
    ND: int,
    syms: List[str],
) -> Dict[str, np.ndarray]:
    """Compute sector-level aggregate signals.

    Returns dict with:
        sector_avg_rank: (NS, ND) - sector average oversold rank for each symbol
        sector_momentum_rank: (NS, ND) - sector momentum rank
        sector_oi_trend: (NS, ND) - sector OI trend score
        sector_rel_score: (NS, ND) - individual rank minus sector average
    """
    t0 = time.time()
    print("[V10] Computing sector signals...", flush=True)

    sector_to_si, si_to_sector, valid_sectors = build_sector_map(syms)
    sector_names = sorted(sector_to_si.keys())

    # Per-sector average oversold rank per day
    sector_avg_rank_arr = np.full((NS, ND), np.nan)
    # Per-sector momentum rank (which sector is strongest/weakest)
    sector_momentum_rank_arr = np.full((NS, ND), np.nan)
    # Sector OI trend
    sector_oi_trend_arr = np.full((NS, ND), np.nan)
    # Sector-relative score: individual - sector average
    sector_rel_score_arr = np.full((NS, ND), np.nan)

    for di in range(ND):
        # Step 1: Compute sector average oversold rank
        sector_avgs: Dict[str, float] = {}
        for sector, indices in sector_to_si.items():
            ranks = combo_rank[indices, di]
            valid_ranks = ranks[~np.isnan(ranks)]
            if len(valid_ranks) >= 2:
                sector_avgs[sector] = float(np.mean(valid_ranks))

        # Step 2: Rank sectors by their average oversold score
        if len(sector_avgs) >= 3:
            sector_series = pd.Series(sector_avgs)
            sector_ranked = sector_series.rank(pct=True)
            # Assign sector momentum rank to each member
            for sector, s_rank in sector_ranked.items():
                for si in sector_to_si[sector]:
                    sector_momentum_rank_arr[si, di] = s_rank

        # Step 3: Compute sector OI trend (5d average OI change)
        sector_oi_avgs: Dict[str, float] = {}
        for sector, indices in sector_to_si.items():
            oi_changes = []
            for si in indices:
                if di >= 5 and not np.isnan(OI[si, di]) and not np.isnan(OI[si, di - 5]):
                    if OI[si, di - 5] > 0:
                        oi_changes.append(OI[si, di] / OI[si, di - 5] - 1)
            if len(oi_changes) >= 2:
                sector_oi_avgs[sector] = float(np.mean(oi_changes))

        for sector, oi_chg in sector_oi_avgs.items():
            for si in sector_to_si[sector]:
                sector_oi_trend_arr[si, di] = oi_chg

        # Step 4: Assign sector average and relative score to each symbol
        for sector, avg_val in sector_avgs.items():
            for si in sector_to_si[sector]:
                sector_avg_rank_arr[si, di] = avg_val
                if not np.isnan(combo_rank[si, di]):
                    sector_rel_score_arr[si, di] = combo_rank[si, di] - avg_val

    print(f"  {len(sector_names)} sectors, Done: {time.time() - t0:.1f}s", flush=True)
    return {
        "sector_avg_rank": sector_avg_rank_arr,
        "sector_momentum_rank": sector_momentum_rank_arr,
        "sector_oi_trend": sector_oi_trend_arr,
        "sector_rel_score": sector_rel_score_arr,
        "sector_to_si": sector_to_si,
        "si_to_sector": si_to_sector,
    }


# ============================================================
# SELECTION STRATEGIES
# ============================================================
def select_pure_individual(
    combo_rank: np.ndarray,
    n_signals: np.ndarray,
    C: np.ndarray,
    O: np.ndarray,
    NS: int,
    ND: int,
    di: int,
    held: set,
    min_rank: float = 0.7,
    min_confidence: int = 3,
    top_n: int = 1,
) -> List[Tuple[float, int]]:
    """Pure individual selection (V5 baseline)."""
    candidates = []
    for si in range(NS):
        if si in held:
            continue
        if np.isnan(combo_rank[si, di]) or combo_rank[si, di] < min_rank:
            continue
        if n_signals[si, di] < min_confidence:
            continue
        if di + 1 >= ND or np.isnan(O[si, di + 1]):
            continue
        candidates.append((combo_rank[si, di], si))
    candidates.sort(key=lambda x: -x[0])
    return candidates[:top_n]


def select_sector_rotation(
    combo_rank: np.ndarray,
    n_signals: np.ndarray,
    sector_avg_rank: np.ndarray,
    sector_oi_trend: np.ndarray,
    C: np.ndarray,
    O: np.ndarray,
    NS: int,
    ND: int,
    di: int,
    held: set,
    sector_to_si: Dict[str, List[int]],
    min_rank: float = 0.7,
    min_confidence: int = 3,
    top_n: int = 1,
    top_sectors: int = 2,
) -> List[Tuple[float, int]]:
    """Sector rotation: pick the most oversold sector(s), then the most
    oversold commodity within those sectors."""
    if di + 1 >= ND:
        return []

    # Rank sectors by average oversold score (higher = more oversold)
    sector_scores: Dict[str, float] = {}
    for sector, indices in sector_to_si.items():
        ranks = sector_avg_rank[indices, di]
        valid_ranks = ranks[~np.isnan(ranks)]
        if len(valid_ranks) >= 2:
            sector_scores[sector] = float(np.mean(valid_ranks))

    if len(sector_scores) < 2:
        return []

    # Sort sectors: highest average rank = most oversold
    sorted_sectors = sorted(sector_scores.items(), key=lambda x: -x[1])
    best_sectors = [s for s, _ in sorted_sectors[:top_sectors]]
    best_sector_set = set(best_sectors)

    # Within best sectors, pick the most oversold individual
    candidates = []
    for si in range(NS):
        if si in held:
            continue
        if np.isnan(combo_rank[si, di]) or combo_rank[si, di] < min_rank:
            continue
        if n_signals[si, di] < min_confidence:
            continue
        if np.isnan(sector_avg_rank[si, di]):
            continue
        if np.isnan(O[si, di + 1]):
            continue
        # Must be in a top oversold sector
        sector_of_si = None
        for sector, indices in sector_to_si.items():
            if si in indices:
                sector_of_si = sector
                break
        if sector_of_si not in best_sector_set:
            continue
        # Score = combo_rank boosted by sector average rank
        boosted_score = combo_rank[si, di] * 0.6 + sector_avg_rank[si, di] * 0.4
        candidates.append((boosted_score, si))

    candidates.sort(key=lambda x: -x[0])
    return candidates[:top_n]


def select_combined(
    combo_rank: np.ndarray,
    n_signals: np.ndarray,
    sector_avg_rank: np.ndarray,
    sector_rel_score: np.ndarray,
    sector_oi_trend: np.ndarray,
    C: np.ndarray,
    O: np.ndarray,
    NS: int,
    ND: int,
    di: int,
    held: set,
    sector_to_si: Dict[str, List[int]],
    min_rank: float = 0.7,
    min_confidence: int = 3,
    top_n: int = 1,
) -> List[Tuple[float, int]]:
    """Combined: sector rotation + pairs mean reversion boost.

    Score = individual oversold rank
          + sector oversold bonus
          + pairs mean-reversion bonus (positive rel_score = oversold vs sector)
    """
    if di + 1 >= ND:
        return []

    candidates = []
    for si in range(NS):
        if si in held:
            continue
        if np.isnan(combo_rank[si, di]) or combo_rank[si, di] < min_rank:
            continue
        if n_signals[si, di] < min_confidence:
            continue
        if np.isnan(O[si, di + 1]):
            continue

        score = combo_rank[si, di]  # base: individual oversold rank

        # Sector oversold bonus: if sector is also oversold, boost
        if not np.isnan(sector_avg_rank[si, di]):
            if sector_avg_rank[si, di] > 0.7:
                # Sector is oversold -> stronger signal
                score += 0.15 * (sector_avg_rank[si, di] - 0.5)
            elif sector_avg_rank[si, di] < 0.3:
                # Sector is overbought -> weaker signal
                score -= 0.10

        # Pairs/relative bonus: oversold relative to sector = mean reversion
        if not np.isnan(sector_rel_score[si, di]):
            if sector_rel_score[si, di] > 0.1:
                # More oversold than sector average -> stronger reversion
                score += 0.10 * sector_rel_score[si, di]
            elif sector_rel_score[si, di] < -0.1:
                # Less oversold than sector -> weaker
                score -= 0.05

        # OI trend bonus: sector losing interest + price falling = capitulation
        if not np.isnan(sector_oi_trend[si, di]):
            if sector_oi_trend[si, di] < -0.03:
                score += 0.05
            elif sector_oi_trend[si, di] > 0.05:
                score -= 0.03

        candidates.append((score, si))

    candidates.sort(key=lambda x: -x[0])
    return candidates[:top_n]


# ============================================================
# GENERIC BACKTEST ENGINE
# ============================================================
def backtest(
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    NS: int,
    ND: int,
    dates: np.ndarray,
    syms: List[str],
    ind_sigs: Dict[str, np.ndarray],
    sec_sigs: Dict[str, np.ndarray],
    mode: str = "combined",
    top_n: int = 1,
    min_rank: float = 0.7,
    atr_stop: float = 3.0,
    min_confidence: int = 3,
    hold_days: int = 5,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Day-by-day backtest.

    mode: 'individual', 'sector', or 'combined'
    Returns (trades, final_equity, max_drawdown)
    """
    if end_di is None:
        end_di = ND - 1

    combo_rank = ind_sigs["combo_rank"]
    n_signals = ind_sigs["n_signals"]
    sector_avg_rank = sec_sigs["sector_avg_rank"]
    sector_rel_score = sec_sigs["sector_rel_score"]
    sector_oi_trend = sec_sigs["sector_oi_trend"]
    sector_to_si = sec_sigs["sector_to_si"]

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions: List[Tuple[int, int, float, float, float, bool]] = []
    trades: List[dict] = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions: List[Tuple[int, int, float, float, float, bool]] = []

        # Group positions by symbol
        pos_by_si: Dict[int, List[Tuple[int, float, float, float, bool]]] = (
            defaultdict(list)
        )
        for si, edi, ep, sp, alloc, _ in positions:
            pos_by_si[si].append((edi, ep, sp, alloc, False))

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
                            "mode": mode,
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
                            "mode": mode,
                        }
                    )
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

        held = {p[0] for p in positions}
        if len(positions) >= top_n:
            continue

        # --- Selection based on mode ---
        if mode == "individual":
            candidates = select_pure_individual(
                combo_rank, n_signals, C, O, NS, ND, di, held,
                min_rank=min_rank, min_confidence=min_confidence, top_n=top_n,
            )
        elif mode == "sector":
            candidates = select_sector_rotation(
                combo_rank, n_signals, sector_avg_rank, sector_oi_trend,
                C, O, NS, ND, di, held, sector_to_si,
                min_rank=min_rank, min_confidence=min_confidence, top_n=top_n,
            )
        else:  # combined
            candidates = select_combined(
                combo_rank, n_signals, sector_avg_rank, sector_rel_score,
                sector_oi_trend, C, O, NS, ND, di, held, sector_to_si,
                min_rank=min_rank, min_confidence=min_confidence, top_n=top_n,
            )

        for _, si in candidates:
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
            alloc = 1.0 / max(top_n, 1)
            positions.append(
                (si, di + 1, ep, ep - atr_stop * atr, alloc, False)
            )
            held.add(si)

    # Close remaining positions
    for si, edi, ep, sp, alloc, is_pyr in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl

    return trades, equity, max_dd


# ============================================================
# ANALYSIS
# ============================================================
def analyze(trades: List[dict], equity: float, max_dd: float, label: str = "") -> Optional[dict]:
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

    n_stop = sum(1 for t in trades if t["reason"] == "stop")
    n_hold = sum(1 for t in trades if t["reason"] == "hold")

    print(
        f"  {label}: {len(trades)}t (stop:{n_stop} hold:{n_hold}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% Sh={sh:.2f} eq={equity:,.0f}"
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
        print(f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% cum={cum:+.1%}")

    return {"n": len(trades), "wr": wr, "dd": max_dd, "ann": ann, "sh": sh, "eq": equity}


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================
def walk_forward(
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    NS: int,
    ND: int,
    dates: np.ndarray,
    syms: List[str],
    ind_sigs: Dict[str, np.ndarray],
    sec_sigs: Dict[str, np.ndarray],
    mode: str = "combined",
    top_n: int = 1,
    min_rank: float = 0.7,
    min_confidence: int = 3,
    hold_days: int = 5,
    atr_stop: float = 3.0,
) -> List[dict]:
    """Walk-forward: train on expanding window, test 1 year at a time."""
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD: mode={mode}, top_n={top_n}")
    print(f"{'='*70}")

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

        trades, _, _ = backtest(
            C, O, H, L, NS, ND, dates, syms, ind_sigs, sec_sigs,
            mode=mode, top_n=top_n, min_rank=min_rank,
            min_confidence=min_confidence, hold_days=hold_days,
            atr_stop=atr_stop,
            start_di=test_start, end_di=test_end_idx + 1,
        )

        test_trades = [t for t in trades if dates[t["di"]].year == test_year]
        all_trades.extend(test_trades)

        if test_trades:
            n = len(test_trades)
            nw = sum(1 for t in test_trades if t["pnl_pct"] > 0)
            wr = nw / n * 100
            avg = np.mean([t["pnl_pct"] for t in test_trades])
            print(f"  {test_year}: {n}t WR={wr:.1f}% avg={avg:+.2f}%", flush=True)
        else:
            print(f"  {test_year}: no trades", flush=True)

    if all_trades:
        nw = sum(1 for t in all_trades if t["pnl_pct"] > 0)
        wr = nw / len(all_trades) * 100
        avg = np.mean([t["pnl_pct"] for t in all_trades])
        cum = np.prod([1 + t["pnl_pct"] / 100 for t in all_trades]) - 1
        print(
            f"\n  WF TOTAL: {len(all_trades)}t WR={wr:.1f}% "
            f"avg={avg:+.2f}% cum={cum:+.1%}"
        )
    return all_trades


# ============================================================
# PARAMETER SWEEP
# ============================================================
def sweep(
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    NS: int,
    ND: int,
    dates: np.ndarray,
    syms: List[str],
    ind_sigs: Dict[str, np.ndarray],
    sec_sigs: Dict[str, np.ndarray],
    start_di: int = 60,
) -> List[dict]:
    """Sweep over modes and parameters to find best configuration."""
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP")
    print("=" * 70)

    results: List[dict] = []

    for mode in ["individual", "sector", "combined"]:
        for top_n in [1, 2, 3]:
            for mc in [2, 3]:
                for atr in [2.5, 3.0, 3.5]:
                    trades, eq, dd = backtest(
                        C, O, H, L, NS, ND, dates, syms,
                        ind_sigs, sec_sigs,
                        mode=mode, top_n=top_n, min_rank=0.7,
                        min_confidence=mc, hold_days=5, atr_stop=atr,
                        start_di=start_di,
                    )
                    if len(trades) < 10:
                        continue
                    nw = sum(1 for t in trades if t["pnl_pct"] > 0)
                    wr = nw / len(trades) * 100
                    n_days = max(1, trades[-1]["di"] - trades[0]["di"])
                    ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
                    ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
                    rets_arr = np.array(ap) / CASH0
                    sh_val = (
                        np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252)
                        if np.std(rets_arr) > 0
                        else 0
                    )
                    results.append(
                        {
                            "mode": mode,
                            "tn": top_n,
                            "mc": mc,
                            "atr": atr,
                            "n": len(trades),
                            "wr": wr,
                            "ann": ann,
                            "dd": dd,
                            "sharpe": sh_val,
                        }
                    )

    results.sort(key=lambda x: -x["sharpe"])
    print(
        f"\n{'Mode':>10} {'TN':>3} {'MC':>3} {'ATR':>4} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}"
    )
    print("-" * 60)
    for r in results[:25]:
        print(
            f"{r['mode']:>10} {r['tn']:>3} {r['mc']:>3} {r['atr']:>4.1f} "
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
    print("  V10: SECTOR ROTATION + PAIRS TRADING")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start="2016-01-01")
    print(
        f"  {NS} sym, {ND} days, "
        f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}"
    )

    # Print sector coverage
    sector_to_si, si_to_sector, valid_sectors = build_sector_map(syms)
    for sector, members in sorted(valid_sectors.items()):
        covered = [m for m in members if m in set(syms)]
        print(f"  {sector:>10}: {len(covered)} symbols - {', '.join(covered)}")

    # Compute signals
    ind_sigs = compute_individual_signals(C, O, H, L, V, OI, NS, ND)
    sec_sigs = compute_sector_signals(
        ind_sigs["combo_rank"], ind_sigs["n_signals"], OI, C, NS, ND, syms
    )

    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # === 1. Compare all three modes (full 10-year) ===
    print("\n" + "=" * 70)
    print("  FULL 10-YEAR: MODE COMPARISON")
    print("=" * 70)

    for mode in ["individual", "sector", "combined"]:
        trades, eq, dd = backtest(
            C, O, H, L, NS, ND, dates, syms, ind_sigs, sec_sigs,
            mode=mode, top_n=1, min_rank=0.7, min_confidence=3,
            hold_days=5, atr_stop=3.0, start_di=60,
        )
        analyze(trades, eq, dd, f"{mode}-1pos")

    # === 2. Multi-position comparison ===
    print("\n" + "=" * 70)
    print("  MULTI-POSITION (2019-2026)")
    print("=" * 70)

    for mode in ["individual", "sector", "combined"]:
        for tn in [1, 2, 3]:
            trades, eq, dd = backtest(
                C, O, H, L, NS, ND, dates, syms, ind_sigs, sec_sigs,
                mode=mode, top_n=tn, min_rank=0.7, min_confidence=3,
                hold_days=5, atr_stop=3.0, start_di=bt_2019,
            )
            analyze(trades, eq, dd, f"{mode}-tn{tn}")

    # === 3. Parameter sweep ===
    results = sweep(
        C, O, H, L, NS, ND, dates, syms, ind_sigs, sec_sigs, start_di=bt_2019
    )

    # === 4. Best full 10-year ===
    if results:
        print("\n" + "=" * 70)
        print("  BEST 10-YEAR (top-5 from sweep)")
        print("=" * 70)
        for r in results[:5]:
            trades, eq, dd = backtest(
                C, O, H, L, NS, ND, dates, syms, ind_sigs, sec_sigs,
                mode=r["mode"], top_n=r["tn"], min_rank=0.7,
                min_confidence=r["mc"], hold_days=5, atr_stop=r["atr"],
                start_di=60,
            )
            label = f"{r['mode']}-tn{r['tn']}-mc{r['mc']}-atr{r['atr']}"
            print(f"\n  FULL {label}")
            analyze(trades, eq, dd, label)

    # === 5. Walk-forward for best ===
    if results:
        best = results[0]
        print("\n" + "=" * 70)
        print(
            f"  WALK-FORWARD BEST: {best['mode']}, tn={best['tn']}, "
            f"mc={best['mc']}, atr={best['atr']}"
        )
        print("=" * 70)
        walk_forward(
            C, O, H, L, NS, ND, dates, syms, ind_sigs, sec_sigs,
            mode=best["mode"], top_n=best["tn"], min_rank=0.7,
            min_confidence=best["mc"], hold_days=5, atr_stop=best["atr"],
        )

        # Also walk-forward the other two modes with same params for comparison
        for other_mode in ["individual", "sector", "combined"]:
            if other_mode != best["mode"]:
                walk_forward(
                    C, O, H, L, NS, ND, dates, syms, ind_sigs, sec_sigs,
                    mode=other_mode, top_n=best["tn"], min_rank=0.7,
                    min_confidence=best["mc"], hold_days=5,
                    atr_stop=best["atr"],
                )

    print(f"\n[V10] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
