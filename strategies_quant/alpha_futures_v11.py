"""
V11: ULTIMATE COMBO STRATEGY
============================
Combines the BEST ideas from V1/V5 (pyramid), V6 (correlation filtering),
V10 (sector boost), into one unified strategy with walk-forward validation.

Architecture:
  1. Entry Signal: V1 multi-alpha (consec_dn, ret5d, OI capitulation, VDP,
     RSI, BB, CCI) with KER gate + confidence >= 3
  2. Sector Boost: V10's sector oversold ranking -- boost candidates from
     oversold sectors
  3. Correlation Filter: V6's correlation-based selection -- when running
     top_n > 1, prefer low-correlation candidates
  4. Pyramid on Winners: V5's pyramid -- add to positions in profit on day 1
  5. Walk-Forward Validation

Signal at close[di], enter at open[di+1]. No look-ahead. No gap signals.
No leverage. Import from alpha_futures_data. Use TA-Lib if available.
"""
import sys
import os
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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

CORR_WINDOW = 60
MIN_CORR_OVERLAP = 30

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
# PHASE 1: INDIVIDUAL MULTI-ALPHA SIGNALS
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
    """Core multi-alpha oversold signal computation (V1/V5 battle-tested)."""
    t0 = time.time()
    print("[V11] Computing individual signals...", flush=True)

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

    # OI capitulation signal
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

    # KER (Kaufman Efficiency Ratio) for regime gate
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
                ker_regime[si, di] = 1
            elif ker_10[si, di] > 0.3:
                ker_regime[si, di] = -1

    # TA-Lib indicators: RSI, BB, CCI
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

    # Composite oversold score + cross-sectional rank
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

    # Signal count (confidence)
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
        "ker_regime": ker_regime,
        "n_signals": n_signals,
    }


# ============================================================
# PHASE 2: SECTOR SIGNALS (from V10)
# ============================================================
def build_sector_map(
    syms: List[str],
) -> Tuple[Dict[str, List[int]], Dict[int, str]]:
    """Build symbol-to-sector mappings."""
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


def compute_sector_signals(
    combo_rank: np.ndarray,
    OI: np.ndarray,
    NS: int,
    ND: int,
    syms: List[str],
) -> Dict[str, np.ndarray]:
    """Compute sector-level aggregate signals (V10 sector oversold ranking)."""
    t0 = time.time()
    print("[V11] Computing sector signals...", flush=True)

    sector_to_si, si_to_sector = build_sector_map(syms)
    sector_names = sorted(sector_to_si.keys())

    # Per-sector average oversold rank per day
    sector_avg_rank_arr = np.full((NS, ND), np.nan)
    # Sector-relative score: individual - sector average
    sector_rel_score_arr = np.full((NS, ND), np.nan)
    # Sector OI trend (5d)
    sector_oi_trend_arr = np.full((NS, ND), np.nan)

    for di in range(ND):
        # Sector average oversold rank
        sector_avgs: Dict[str, float] = {}
        for sector, indices in sector_to_si.items():
            ranks = combo_rank[indices, di]
            valid_ranks = ranks[~np.isnan(ranks)]
            if len(valid_ranks) >= 2:
                sector_avgs[sector] = float(np.mean(valid_ranks))

        # Rank sectors by oversold score
        if len(sector_avgs) >= 3:
            sector_series = pd.Series(sector_avgs)
            sector_ranked = sector_series.rank(pct=True)

        # Sector OI trend
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

        # Assign per-symbol values
        for sector, avg_val in sector_avgs.items():
            for si in sector_to_si[sector]:
                sector_avg_rank_arr[si, di] = avg_val
                if not np.isnan(combo_rank[si, di]):
                    sector_rel_score_arr[si, di] = combo_rank[si, di] - avg_val

    print(
        f"  {len(sector_names)} sectors, Done: {time.time() - t0:.1f}s",
        flush=True,
    )
    return {
        "sector_avg_rank": sector_avg_rank_arr,
        "sector_rel_score": sector_rel_score_arr,
        "sector_oi_trend": sector_oi_trend_arr,
        "sector_to_si": sector_to_si,
        "si_to_sector": si_to_sector,
    }


# ============================================================
# PHASE 3: ROLLING CORRELATION (from V6)
# ============================================================
def compute_rolling_correlations(
    C: np.ndarray, NS: int, ND: int, window: int = CORR_WINDOW
) -> np.ndarray:
    """Compute rolling pairwise correlations (NS, NS, ND)."""
    t0 = time.time()
    print(f"[V11] Computing rolling {window}-day correlations...", flush=True)

    # Daily returns
    returns = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if (
                not np.isnan(C[si, di])
                and not np.isnan(C[si, di - 1])
                and C[si, di - 1] > 0
            ):
                returns[si, di] = C[si, di] / C[si, di - 1] - 1

    corr_matrix = np.full((NS, NS, ND), np.nan)
    for di in range(window, ND):
        ret_window = returns[:, di - window : di]
        for si in range(NS):
            ri = ret_window[si]
            valid_i = ~np.isnan(ri)
            if valid_i.sum() < MIN_CORR_OVERLAP:
                continue
            for sj in range(si, NS):
                rj = ret_window[sj]
                valid_j = ~np.isnan(rj)
                overlap = valid_i & valid_j
                n_overlap = overlap.sum()
                if n_overlap < MIN_CORR_OVERLAP:
                    continue
                ri_clean = ri[overlap]
                rj_clean = rj[overlap]
                std_i = np.std(ri_clean)
                std_j = np.std(rj_clean)
                if std_i > 1e-10 and std_j > 1e-10:
                    corr_val = np.corrcoef(ri_clean, rj_clean)[0, 1]
                    if not np.isnan(corr_val):
                        corr_matrix[si, sj, di] = corr_val
                        corr_matrix[sj, si, di] = corr_val

    n_pairs = NS * (NS - 1) // 2
    filled = 0
    sample_di = min(window + 100, ND - 1)
    for si in range(NS):
        for sj in range(si + 1, NS):
            if not np.isnan(corr_matrix[si, sj, sample_di]):
                filled += 1
    print(
        f"  Corr matrix sample (di={sample_di}): "
        f"{filled}/{n_pairs} pairs filled, {time.time() - t0:.1f}s",
        flush=True,
    )
    return corr_matrix


def get_avg_corr_to_positions(
    corr_matrix: np.ndarray,
    candidate_si: int,
    held_sis: List[int],
    di: int,
) -> float:
    """Average absolute correlation between candidate and held positions."""
    if not held_sis:
        return 0.0
    corr_vals = []
    for held_si in held_sis:
        c = corr_matrix[candidate_si, held_si, di]
        if not np.isnan(c):
            corr_vals.append(abs(c))
    return float(np.mean(corr_vals)) if corr_vals else 0.0


# ============================================================
# PHASE 4: UNIFIED BACKTEST ENGINE
# ============================================================
def backtest_v11(
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    NS: int,
    ND: int,
    dates: list,
    syms: List[str],
    ind_sigs: Dict[str, np.ndarray],
    sec_sigs: Dict[str, np.ndarray],
    corr_matrix: np.ndarray,
    # Feature toggles
    use_ker_gate: bool = True,
    use_sector_boost: bool = True,
    use_correlation_filter: bool = True,
    use_pyramid: bool = True,
    # Parameters
    top_n: int = 1,
    min_rank: float = 0.7,
    atr_stop: float = 3.0,
    min_confidence: int = 3,
    hold_days: int = 5,
    sector_boost_weight: float = 0.3,
    pyramid_ratio: float = 0.5,
    pyramid_day: int = 1,
    max_corr: float = 0.6,
    corr_penalty: float = 0.3,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """
    Unified backtest combining all V11 features.

    Scoring: combo_rank + sector_boost_weight * sector_bonus
             - corr_penalty * avg_abs_corr_to_held
    """
    combo_rank = ind_sigs["combo_rank"]
    ker_regime = ind_sigs["ker_regime"]
    n_signals = ind_sigs["n_signals"]

    sector_avg_rank = sec_sigs["sector_avg_rank"]
    sector_rel_score = sec_sigs["sector_rel_score"]
    sector_oi_trend = sec_sigs["sector_oi_trend"]
    sector_to_si = sec_sigs["sector_to_si"]

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

        # ---- EXIT LOGIC ----
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
                    trades.append({
                        "pnl_abs": profit,
                        "pnl_pct": pnl * 100,
                        "days": di - edi + 1,
                        "di": di,
                        "year": d.year,
                        "sym": syms[si],
                        "reason": "stop",
                        "pyr": is_pyr,
                    })
            elif hold >= hold_days:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    trades.append({
                        "pnl_abs": profit,
                        "pnl_pct": pnl * 100,
                        "days": di - edi + 1,
                        "di": di,
                        "year": d.year,
                        "sym": syms[si],
                        "reason": "hold",
                        "pyr": is_pyr,
                    })
            else:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc, is_pyr))

        # ---- PYRAMID CHECK ----
        if use_pyramid and pyramid_ratio > 0:
            held_with_pos: Dict[int, List[Tuple[int, float, float, float, bool]]] = (
                defaultdict(list)
            )
            for si, edi, ep, sp, alloc, is_pyr in new_positions:
                held_with_pos[si].append((edi, ep, sp, alloc, is_pyr))

            additions: List[Tuple[int, int, float, float, float, bool]] = []
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
                                (si, di, c_now, c_now - atr_stop * atr, pyr_alloc, True)
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

        # ---- ENTRY SELECTION ----
        held = {p[0] for p in positions}
        if len(positions) >= top_n:
            continue

        candidates = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(combo_rank[si, di]) or combo_rank[si, di] < min_rank:
                continue
            if n_signals[si, di] < min_confidence:
                continue
            if use_ker_gate and ker_regime[si, di] < 0:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue

            score = combo_rank[si, di]

            # Sector boost
            if use_sector_boost and sector_boost_weight > 0:
                sector_bonus = 0.0
                if not np.isnan(sector_avg_rank[si, di]):
                    if sector_avg_rank[si, di] > 0.7:
                        sector_bonus += 0.15 * (sector_avg_rank[si, di] - 0.5)
                    elif sector_avg_rank[si, di] < 0.3:
                        sector_bonus -= 0.10
                if not np.isnan(sector_rel_score[si, di]):
                    if sector_rel_score[si, di] > 0.1:
                        sector_bonus += 0.10 * sector_rel_score[si, di]
                if not np.isnan(sector_oi_trend[si, di]):
                    if sector_oi_trend[si, di] < -0.03:
                        sector_bonus += 0.05
                score += sector_boost_weight * sector_bonus

            # Correlation penalty
            avg_corr = 0.0
            if use_correlation_filter and len(held) > 0 and top_n > 1:
                held_sis = list(held)
                avg_corr = get_avg_corr_to_positions(corr_matrix, si, held_sis, di)
                if avg_corr > max_corr:
                    continue
                score -= corr_penalty * avg_corr

            candidates.append((score, si))

        candidates.sort(key=lambda x: -x[0])

        for score, si in candidates:
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

    n_pyr = sum(1 for t in trades if t.get("pyr"))
    n_base = len(trades) - n_pyr
    n_stop = sum(1 for t in trades if t["reason"] == "stop")
    n_hold = sum(1 for t in trades if t["reason"] == "hold")

    print(
        f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} "
        f"stop:{n_stop} hold:{n_hold}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
        f"Sh={sh:.2f} eq={equity:,.0f}"
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
    dates: list,
    syms: List[str],
    ind_sigs: Dict[str, np.ndarray],
    sec_sigs: Dict[str, np.ndarray],
    corr_matrix: np.ndarray,
    label: str = "",
    **kwargs,
) -> List[dict]:
    """Walk-forward: test 1 year at a time, no look-ahead."""
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD: {label}")
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

        trades, _, _ = backtest_v11(
            C, O, H, L, NS, ND, dates, syms,
            ind_sigs, sec_sigs, corr_matrix,
            start_di=test_start, end_di=test_end_idx + 1,
            **kwargs,
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
        n_pyr = sum(1 for t in all_trades if t.get("pyr"))
        print(
            f"\n  WF TOTAL: {len(all_trades)}t (pyr:{n_pyr}) "
            f"WR={wr:.1f}% avg={avg:+.2f}% cum={cum:+.1%}"
        )
    return all_trades


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V11: ULTIMATE COMBO STRATEGY")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start="2016-01-01")
    print(
        f"  {NS} sym, {ND} days, "
        f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}"
    )

    # Print sector coverage
    sector_to_si, si_to_sector = build_sector_map(syms)
    for sector, indices in sorted(sector_to_si.items()):
        names = [syms[i] for i in indices]
        print(f"  {sector:>10}: {len(indices)} symbols - {', '.join(names)}")

    # Compute all signals
    ind_sigs = compute_individual_signals(C, O, H, L, V, OI, NS, ND)
    sec_sigs = compute_sector_signals(
        ind_sigs["combo_rank"], OI, NS, ND, syms
    )
    corr_matrix = compute_rolling_correlations(C, NS, ND, window=CORR_WINDOW)

    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # ============================================================
    # SECTION 1: ABLATION STUDY
    # ============================================================
    print("\n" + "=" * 70)
    print("  SECTION 1: ABLATION STUDY (2019-2026)")
    print("=" * 70)

    ablation_configs = [
        {
            "label": "BASELINE (all off)",
            "use_ker_gate": False,
            "use_sector_boost": False,
            "use_correlation_filter": False,
            "use_pyramid": False,
        },
        {
            "label": "+KER gate only",
            "use_ker_gate": True,
            "use_sector_boost": False,
            "use_correlation_filter": False,
            "use_pyramid": False,
        },
        {
            "label": "+Sector boost only",
            "use_ker_gate": True,
            "use_sector_boost": True,
            "use_correlation_filter": False,
            "use_pyramid": False,
        },
        {
            "label": "+Correlation filter only",
            "use_ker_gate": True,
            "use_sector_boost": False,
            "use_correlation_filter": True,
            "use_pyramid": False,
            "top_n": 2,
        },
        {
            "label": "+Pyramid only",
            "use_ker_gate": True,
            "use_sector_boost": False,
            "use_correlation_filter": False,
            "use_pyramid": True,
        },
        {
            "label": "FULL (all on)",
            "use_ker_gate": True,
            "use_sector_boost": True,
            "use_correlation_filter": True,
            "use_pyramid": True,
        },
    ]

    for cfg in ablation_configs:
        label = cfg.pop("label")
        tn = cfg.pop("top_n", 1)
        trades, eq, dd = backtest_v11(
            C, O, H, L, NS, ND, dates, syms,
            ind_sigs, sec_sigs, corr_matrix,
            top_n=tn, start_di=bt_2019,
            **cfg,
        )
        analyze(trades, eq, dd, label)

    # ============================================================
    # SECTION 2: SECTOR BOOST WEIGHT SWEEP
    # ============================================================
    print("\n" + "=" * 70)
    print("  SECTION 2: SECTOR BOOST WEIGHT SWEEP (2019-2026)")
    print("=" * 70)

    for sbw in [0, 0.2, 0.3, 0.5]:
        for tn in [1, 2]:
            trades, eq, dd = backtest_v11(
                C, O, H, L, NS, ND, dates, syms,
                ind_sigs, sec_sigs, corr_matrix,
                top_n=tn, sector_boost_weight=sbw,
                start_di=bt_2019,
            )
            label = f"sbw={sbw:.1f} tn={tn}"
            analyze(trades, eq, dd, label)

    # ============================================================
    # SECTION 3: PYRAMID RATIO SWEEP
    # ============================================================
    print("\n" + "=" * 70)
    print("  SECTION 3: PYRAMID RATIO SWEEP (2019-2026)")
    print("=" * 70)

    for pr in [0, 0.3, 0.5, 0.7]:
        for tn in [1, 2]:
            trades, eq, dd = backtest_v11(
                C, O, H, L, NS, ND, dates, syms,
                ind_sigs, sec_sigs, corr_matrix,
                top_n=tn, pyramid_ratio=pr,
                start_di=bt_2019,
            )
            label = f"pyr={pr:.1f} tn={tn}"
            analyze(trades, eq, dd, label)

    # ============================================================
    # SECTION 4: TOP-N SWEEP
    # ============================================================
    print("\n" + "=" * 70)
    print("  SECTION 4: TOP-N SWEEP (2019-2026)")
    print("=" * 70)

    for tn in [1, 2, 3]:
        trades, eq, dd = backtest_v11(
            C, O, H, L, NS, ND, dates, syms,
            ind_sigs, sec_sigs, corr_matrix,
            top_n=tn, start_di=bt_2019,
        )
        analyze(trades, eq, dd, f"tn={tn}")

    # ============================================================
    # SECTION 5: FULL PARAMETER SWEEP
    # ============================================================
    print("\n" + "=" * 70)
    print("  SECTION 5: FULL PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    results: List[dict] = []
    for tn in [1, 2, 3]:
        for sbw in [0, 0.2, 0.3]:
            for pr in [0, 0.3, 0.5]:
                for mc in [2, 3]:
                    trades, eq, dd = backtest_v11(
                        C, O, H, L, NS, ND, dates, syms,
                        ind_sigs, sec_sigs, corr_matrix,
                        top_n=tn, sector_boost_weight=sbw,
                        pyramid_ratio=pr, min_confidence=mc,
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
                    sh_val = (
                        np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252)
                        if np.std(rets_arr) > 0
                        else 0
                    )
                    results.append({
                        "tn": tn, "sbw": sbw, "pr": pr, "mc": mc,
                        "n": len(trades), "wr": wr, "ann": ann,
                        "dd": dd, "sharpe": sh_val,
                    })

    results.sort(key=lambda x: -x["sharpe"])
    print(
        f"\n{'TN':>3} {'SBW':>4} {'Pyr':>4} {'MC':>3} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>6}"
    )
    print("-" * 60)
    for r in results[:30]:
        print(
            f"{r['tn']:>3} {r['sbw']:>4.1f} {r['pr']:>4.1f} {r['mc']:>3} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f}"
        )

    # ============================================================
    # SECTION 6: BEST 10-YEAR FULL RUN
    # ============================================================
    print("\n" + "=" * 70)
    print("  SECTION 6: BEST CONFIGS — FULL 10-YEAR (2016-2026)")
    print("=" * 70)

    best_configs = results[:5]
    for r in best_configs:
        trades, eq, dd = backtest_v11(
            C, O, H, L, NS, ND, dates, syms,
            ind_sigs, sec_sigs, corr_matrix,
            top_n=r["tn"], sector_boost_weight=r["sbw"],
            pyramid_ratio=r["pr"], min_confidence=r["mc"],
            start_di=60,
        )
        label = f"tn={r['tn']} sbw={r['sbw']:.1f} pyr={r['pr']:.1f} mc={r['mc']}"
        print(f"\n  10Y {label}")
        analyze(trades, eq, dd, label)

    # ============================================================
    # SECTION 7: WALK-FORWARD VALIDATION
    # ============================================================
    print("\n" + "=" * 70)
    print("  SECTION 7: WALK-FORWARD VALIDATION")
    print("=" * 70)

    for r in best_configs[:3]:
        label = f"tn={r['tn']} sbw={r['sbw']:.1f} pyr={r['pr']:.1f} mc={r['mc']}"
        walk_forward(
            C, O, H, L, NS, ND, dates, syms,
            ind_sigs, sec_sigs, corr_matrix,
            label=label,
            top_n=r["tn"],
            sector_boost_weight=r["sbw"],
            pyramid_ratio=r["pr"],
            min_confidence=r["mc"],
        )

    print(f"\n[V11] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
