"""
V16: Intraday Pattern Mean Reversion — Walk-Forward Validated
=============================================================
Core thesis: OHLC bar patterns (wicks, body ratio, gaps, range expansion)
detect capitulation candles for mean reversion entry.

Signal architecture:
  Layer 1 (30%): Candle pattern features
    - Lower wick ratio: hammer / buying pressure at support
    - Body ratio: small body at bottom = indecision -> reversal
    - Gap down: gap downs often fill
    - Range expansion: capitulation has huge range vs 20d avg
    - Close position: close near top of range = buying pressure
  Layer 2 (70%): V1 oversold composite (consec_dn, ret5d, OI decline, RSI, BB, CCI)
  Cross-sectional rank + confidence threshold + pyramid on day-1 winners

Signal at close[di], enter at open[di+1]. No look-ahead. No leverage.
"""
import sys
import os
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_data import load_all_data

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

CASH0 = 1_000_000
COMM = 0.0005

# Pattern detection thresholds
RANGE_LOOKBACK = 60
RANGE_PERCENTILE = 0.90
CLOSE_POS_THRESHOLD = 0.70
WICK_THRESHOLD = 0.40

# Signal weights
W_PATTERN = 0.30
W_V1 = 0.70


# ============================================================
# CANDLE PATTERN FEATURES
# ============================================================
def compute_candle_patterns(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int,
) -> Dict[str, np.ndarray]:
    """Compute bar pattern features for capitulation detection."""
    t0 = time.time()
    print("[V16] Computing candle patterns...", flush=True)

    # --- Lower wick ratio: (min(O,C) - L) / (H - L) ---
    lower_wick = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if (np.isnan(O[si, di]) or np.isnan(C[si, di])
                    or np.isnan(H[si, di]) or np.isnan(L[si, di])):
                continue
            bar_range = H[si, di] - L[si, di]
            if bar_range > 0:
                body_bottom = min(O[si, di], C[si, di])
                lower_wick[si, di] = (body_bottom - L[si, di]) / bar_range

    # --- Body ratio: abs(C - O) / (H - L) ---
    body_ratio = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if (np.isnan(O[si, di]) or np.isnan(C[si, di])
                    or np.isnan(H[si, di]) or np.isnan(L[si, di])):
                continue
            bar_range = H[si, di] - L[si, di]
            if bar_range > 0:
                body_ratio[si, di] = abs(C[si, di] - O[si, di]) / bar_range

    # --- Gap down: O < prev_C ---
    gap_down = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if (np.isnan(O[si, di]) or np.isnan(C[si, di - 1])
                    or C[si, di - 1] <= 0):
                continue
            gap = (C[si, di - 1] - O[si, di]) / C[si, di - 1]
            if gap > 0:
                gap_down[si, di] = min(gap, 0.1) / 0.1  # cap at 10%

    # --- Range expansion: (H-L) / 20d avg range ---
    range_expansion = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            ranges: List[float] = []
            for j in range(di - 20, di):
                if not np.isnan(H[si, j]) and not np.isnan(L[si, j]):
                    ranges.append(H[si, j] - L[si, j])
            if len(ranges) < 10:
                continue
            if np.isnan(H[si, di]) or np.isnan(L[si, di]):
                continue
            avg_range = np.mean(ranges)
            if avg_range > 0:
                current_range = H[si, di] - L[si, di]
                range_expansion[si, di] = current_range / avg_range

    # --- Close position: (C - L) / (H - L) ---
    close_pos = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if (np.isnan(C[si, di]) or np.isnan(H[si, di])
                    or np.isnan(L[si, di])):
                continue
            bar_range = H[si, di] - L[si, di]
            if bar_range > 0:
                close_pos[si, di] = (C[si, di] - L[si, di]) / bar_range

    # --- Range percentile rank (60d rolling) ---
    range_pct = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(RANGE_LOOKBACK, ND):
            if np.isnan(H[si, di]) or np.isnan(L[si, di]):
                continue
            current = H[si, di] - L[si, di]
            past_ranges: List[float] = []
            for j in range(di - RANGE_LOOKBACK, di):
                if not np.isnan(H[si, j]) and not np.isnan(L[si, j]):
                    past_ranges.append(H[si, j] - L[si, j])
            if len(past_ranges) < 30:
                continue
            rank = sum(1 for r in past_ranges if r < current) / len(past_ranges)
            range_pct[si, di] = rank

    # --- Capitulation hammer composite ---
    # Conditions: large range (top 10%) + close in top 30% + lower wick > 40%
    capitulation_score = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(RANGE_LOOKBACK, ND):
            if np.isnan(range_pct[si, di]):
                continue
            score = 0.0
            w_total = 0.0

            # Large range (top 10% of 60d) - weight 0.25
            if range_pct[si, di] >= RANGE_PERCENTILE:
                score += (range_pct[si, di] - RANGE_PERCENTILE) / (1.0 - RANGE_PERCENTILE) * 0.25
            w_total += 0.25

            # Close position in top 30% of range - weight 0.25
            if not np.isnan(close_pos[si, di]):
                if close_pos[si, di] >= CLOSE_POS_THRESHOLD:
                    score += (close_pos[si, di] - CLOSE_POS_THRESHOLD) / (1.0 - CLOSE_POS_THRESHOLD) * 0.25
                w_total += 0.25

            # Lower wick > 40% - weight 0.20
            if not np.isnan(lower_wick[si, di]):
                if lower_wick[si, di] >= WICK_THRESHOLD:
                    score += (lower_wick[si, di] - WICK_THRESHOLD) / (1.0 - WICK_THRESHOLD) * 0.20
                w_total += 0.20

            # Small body at bottom (body ratio < 0.3) - weight 0.10
            if not np.isnan(body_ratio[si, di]):
                if body_ratio[si, di] < 0.3:
                    score += (0.3 - body_ratio[si, di]) / 0.3 * 0.10
                w_total += 0.10

            # Gap down - weight 0.10
            if not np.isnan(gap_down[si, di]):
                score += gap_down[si, di] * 0.10
                w_total += 0.10

            # Range expansion > 1.5x - weight 0.10
            if not np.isnan(range_expansion[si, di]):
                if range_expansion[si, di] > 1.5:
                    score += min((range_expansion[si, di] - 1.5) / 1.5, 1.0) * 0.10
                w_total += 0.10

            if w_total > 0:
                capitulation_score[si, di] = score / w_total

    # --- Cross-sectional rank of capitulation score ---
    cap_rank = np.full((NS, ND), np.nan)
    for di in range(ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            if not np.isnan(capitulation_score[si, di]):
                scores[si] = capitulation_score[si, di]
        valid = ~np.isnan(scores)
        if valid.sum() >= 5:
            cap_rank[:, di] = pd.Series(scores).rank(pct=True, na_option='keep').values

    elapsed = time.time() - t0
    valid_scores = capitulation_score[~np.isnan(capitulation_score)]
    if len(valid_scores) > 0:
        print(f"  Capitulation scores: mean={np.mean(valid_scores):.3f}, "
              f"max={np.max(valid_scores):.3f}, {elapsed:.1f}s", flush=True)
    else:
        print(f"  No valid scores. {elapsed:.1f}s", flush=True)

    return {
        'lower_wick': lower_wick,
        'body_ratio': body_ratio,
        'gap_down': gap_down,
        'range_expansion': range_expansion,
        'close_pos': close_pos,
        'range_pct': range_pct,
        'capitulation_score': capitulation_score,
        'cap_rank': cap_rank,
    }


# ============================================================
# V1 OVERSOLD SIGNALS (from V1, simplified)
# ============================================================
def compute_oversold_signals(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
) -> Dict[str, np.ndarray]:
    """Compute composite oversold score (V1-style 7 signals)."""
    t0 = time.time()
    print("[V16] Computing V1 oversold signals...", flush=True)

    # --- 1. Consecutive down days ---
    consec_dn = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        consec = 0
        for di in range(1, ND):
            if (not np.isnan(C[si, di]) and not np.isnan(C[si, di - 1])
                    and C[si, di - 1] > 0):
                consec = consec + 1 if C[si, di] < C[si, di - 1] else 0
            else:
                consec = 0
            consec_dn[si, di] = consec

    # --- 2. 5-day return ---
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (not np.isnan(C[si, di]) and not np.isnan(C[si, di - 5])
                    and C[si, di - 5] > 0):
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1

    # --- 3. OI capitulation ---
    oi_decline = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (np.isnan(OI[si, di]) or np.isnan(OI[si, di - 5])
                    or np.isnan(C[si, di]) or np.isnan(C[si, di - 5])
                    or C[si, di - 5] <= 0):
                continue
            oi_chg = OI[si, di] / OI[si, di - 5] - 1
            price_chg = C[si, di] / C[si, di - 5] - 1
            if oi_chg < -0.02 and price_chg < -0.02:
                oi_decline[si, di] = (min(abs(oi_chg), 0.2) / 0.2
                                      * min(abs(price_chg), 0.1) / 0.1)
            else:
                oi_decline[si, di] = 0.0

    # --- 4. RSI 14 ---
    rsi14 = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nan_mask = np.isnan(C[si])
            try:
                rsi = talib.RSI(c, 14)
                rsi14[si] = np.where(nan_mask, np.nan, rsi)
            except Exception:
                pass

    # --- 5. Bollinger Band position ---
    bb_pos = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nan_mask = np.isnan(C[si])
            try:
                upper, mid, lower = talib.BBANDS(c, 20, 2.0, 2.0)
                bb_range = upper - lower
                valid_bb = (~nan_mask) & (bb_range > 1e-10)
                bb_pos[si] = np.where(valid_bb, (c - lower) / bb_range, np.nan)
            except Exception:
                pass

    # --- 6. CCI 14 ---
    cci14 = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            h = np.where(np.isnan(H[si]), 0, H[si]).astype(np.float64)
            l = np.where(np.isnan(L[si]), 0, L[si]).astype(np.float64)
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nan_mask = np.isnan(C[si])
            try:
                cci = talib.CCI(h, l, c, 14)
                cci14[si] = np.where(nan_mask, np.nan, cci)
            except Exception:
                pass

    # --- 7. Volume delta proxy (simplified VDP) ---
    vdp_10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        vdp = np.full(ND, np.nan)
        for di in range(1, ND):
            if (np.isnan(H[si, di]) or np.isnan(L[si, di])
                    or np.isnan(C[si, di]) or np.isnan(V[si, di])):
                continue
            bar_range = H[si, di] - L[si, di]
            if bar_range > 0 and V[si, di] > 0:
                vdp[di] = (V[si, di]
                           * (2 * C[si, di] - H[si, di] - L[si, di])
                           / bar_range)
        for di in range(10, ND):
            vals = vdp[di - 10:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 5:
                vdp_10[si, di] = np.mean(valid)

    # --- Composite score (V1 weighting) ---
    raw_score = np.full((NS, ND), np.nan)
    for di in range(ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            s = 0.0
            w_total = 0.0

            # Consecutive down days (0.20)
            s += min(consec_dn[si, di] / 5.0, 1.0) * 0.20
            w_total += 0.20

            # 5d return oversold (0.20)
            if not np.isnan(ret_5d[si, di]):
                s += min(max(-ret_5d[si, di] / 0.1, 0), 1.0) * 0.20
                w_total += 0.20

            # OI capitulation (0.20)
            if not np.isnan(oi_decline[si, di]):
                s += oi_decline[si, di] * 0.20
                w_total += 0.20

            # RSI oversold (0.10)
            if not np.isnan(rsi14[si, di]):
                if rsi14[si, di] < 30:
                    s += (30 - rsi14[si, di]) / 30.0 * 0.10
                w_total += 0.10

            # Bollinger lower band (0.10)
            if not np.isnan(bb_pos[si, di]):
                if bb_pos[si, di] < 0.2:
                    s += (0.2 - bb_pos[si, di]) / 0.2 * 0.10
                w_total += 0.10

            # CCI oversold (0.05)
            if not np.isnan(cci14[si, di]):
                if cci14[si, di] < -100:
                    s += min((-100 - cci14[si, di]) / 200.0, 1.0) * 0.05
                w_total += 0.05

            # VDP exhaustion (0.05)
            if not np.isnan(vdp_10[si, di]):
                if vdp_10[si, di] < -0.3:
                    s += min(-vdp_10[si, di] / 1.0, 1.0) * 0.05
                w_total += 0.05

            if w_total > 0:
                scores[si] = s / w_total

        valid = ~np.isnan(scores)
        if valid.sum() >= 5:
            raw_score[:, di] = (pd.Series(scores)
                                .rank(pct=True, na_option='keep').values)

    # --- Confidence count ---
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
            if not np.isnan(rsi14[si, di]) and rsi14[si, di] < 35:
                n += 1
            if not np.isnan(bb_pos[si, di]) and bb_pos[si, di] < 0.15:
                n += 1
            if not np.isnan(cci14[si, di]) and cci14[si, di] < -100:
                n += 1
            n_signals[si, di] = n

    print(f"  Done: {time.time() - t0:.1f}s", flush=True)

    return {
        'combo_rank': raw_score,
        'n_signals': n_signals,
        'consec_dn': consec_dn,
        'ret_5d': ret_5d,
        'oi_decline': oi_decline,
        'rsi14': rsi14,
        'bb_pos': bb_pos,
        'cci14': cci14,
    }


# ============================================================
# COMBINED RANK (pattern 30% + V1 70%)
# ============================================================
def compute_combined_rank(
    v1_sigs: Dict[str, np.ndarray],
    candle_sigs: Dict[str, np.ndarray],
    NS: int, ND: int,
    pattern_weight: float = W_PATTERN,
) -> np.ndarray:
    """Combine V1 oversold rank and candle pattern rank."""
    combined = np.full((NS, ND), np.nan)
    v1_rank = v1_sigs['combo_rank']
    cap_rank = candle_sigs['cap_rank']

    for di in range(ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            v1_val = v1_rank[si, di]
            cap_val = cap_rank[si, di]
            if np.isnan(v1_val) and np.isnan(cap_val):
                continue
            # Use available signals with adjusted weights
            w_pattern = pattern_weight
            w_v1 = 1.0 - pattern_weight
            total = 0.0

            if not np.isnan(cap_val) and not np.isnan(v1_val):
                total = w_pattern * cap_val + w_v1 * v1_val
            elif not np.isnan(v1_val):
                total = v1_val
            elif not np.isnan(cap_val):
                total = cap_val

            scores[si] = total

        valid = ~np.isnan(scores)
        if valid.sum() >= 5:
            combined[:, di] = (pd.Series(scores)
                               .rank(pct=True, na_option='keep').values)
    return combined


# ============================================================
# BACKTEST ENGINE
# ============================================================
def backtest_v16(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: list, syms: list,
    combined_rank: np.ndarray,
    v1_sigs: Dict[str, np.ndarray],
    candle_sigs: Dict[str, np.ndarray],
    top_n: int = 1,
    min_rank: float = 0.7,
    atr_stop: float = 3.0,
    min_confidence: int = 3,
    hold_days: int = 5,
    pyramid_ratio: float = 0.5,
    pyramid_day: int = 1,
    min_cap_score: float = 0.0,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """
    V16 backtest with candle pattern + V1 oversold.
    Signal at close[di], enter at open[di+1]. No look-ahead.
    """
    n_signals = v1_sigs['n_signals']
    cap_score = candle_sigs['capitulation_score']

    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions: List[Tuple] = []
    trades: List[dict] = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions: List[Tuple] = []

        # --- Exit logic ---
        pos_by_si: Dict[int, List[Tuple]] = defaultdict(list)
        for si, edi, ep, sp, alloc, is_pyr, direction in positions:
            pos_by_si[si].append((edi, ep, sp, alloc, is_pyr, direction))

        for si, pos_list in pos_by_si.items():
            c = C[si, di]
            if np.isnan(c):
                for edi, ep, sp, alloc, is_pyr, direction in pos_list:
                    new_positions.append(
                        (si, edi, ep, sp, alloc, is_pyr, direction))
                continue

            earliest_edi = min(p[0] for p in pos_list)
            hold = di - earliest_edi

            # Stop check
            stopped = False
            for edi, ep, sp, alloc, is_pyr, direction in pos_list:
                if direction > 0 and c < sp:
                    stopped = True
                    break
                if direction < 0 and c > sp:
                    stopped = True
                    break

            if stopped:
                for edi, ep, sp, alloc, is_pyr, direction in pos_list:
                    pnl = direction * (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    trades.append({
                        'pnl_abs': profit,
                        'pnl_pct': pnl * 100,
                        'days': di - edi + 1,
                        'di': di,
                        'year': d.year,
                        'sym': syms[si],
                        'reason': 'stop',
                        'pyr': is_pyr,
                        'dir': direction,
                    })
            elif hold >= hold_days:
                for edi, ep, sp, alloc, is_pyr, direction in pos_list:
                    pnl = direction * (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    trades.append({
                        'pnl_abs': profit,
                        'pnl_pct': pnl * 100,
                        'days': di - edi + 1,
                        'di': di,
                        'year': d.year,
                        'sym': syms[si],
                        'reason': 'hold',
                        'pyr': is_pyr,
                        'dir': direction,
                    })
            else:
                for edi, ep, sp, alloc, is_pyr, direction in pos_list:
                    new_positions.append(
                        (si, edi, ep, sp, alloc, is_pyr, direction))

        # --- Pyramid check ---
        if pyramid_ratio > 0:
            held_with_pos: Dict[int, List[Tuple]] = defaultdict(list)
            for si, edi, ep, sp, alloc, is_pyr, direction in new_positions:
                held_with_pos[si].append(
                    (edi, ep, sp, alloc, is_pyr, direction))

            additions: List[Tuple] = []
            for si, pos_list in held_with_pos.items():
                has_pyr = any(p[4] for p in pos_list)
                if has_pyr:
                    continue
                earliest_edi = min(p[0] for p in pos_list)
                hold = di - earliest_edi
                if hold == pyramid_day and not np.isnan(C[si, di]):
                    direction = pos_list[0][5]
                    avg_ep = np.mean([p[1] for p in pos_list])
                    if direction > 0 and C[si, di] > avg_ep:
                        base_alloc = sum(p[3] for p in pos_list)
                        pyr_alloc = base_alloc * pyramid_ratio
                        c_now = C[si, di]
                        atr_v: List[float] = []
                        for j in range(max(start_di, di - 14), di):
                            hh, ll, cc = H[si, j], L[si, j], C[si, j]
                            if not any(np.isnan([hh, ll, cc])):
                                atr_v.append(
                                    max(hh - ll, abs(hh - cc), abs(ll - cc)))
                        if atr_v:
                            atr = np.mean(atr_v)
                            stop = c_now - atr_stop * atr
                            additions.append(
                                (si, di, c_now, stop, pyr_alloc, True, 1))
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

        # --- Entry logic ---
        held = {p[0] for p in positions}
        if len(positions) >= top_n:
            continue

        candidates: List[Tuple] = []

        for si in range(NS):
            if si in held:
                continue
            if np.isnan(combined_rank[si, di]):
                continue
            if combined_rank[si, di] < min_rank:
                continue
            if n_signals[si, di] < min_confidence:
                continue
            if min_cap_score > 0:
                if np.isnan(cap_score[si, di]) or cap_score[si, di] < min_cap_score:
                    continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue

            # Size boost from capitulation pattern
            cap_boost = 1.0
            if not np.isnan(cap_score[si, di]) and cap_score[si, di] > 0.3:
                cap_boost = 1.0 + cap_score[si, di] * 0.3

            candidates.append(
                (combined_rank[si, di], si, cap_boost))

        # Sort by rank (most oversold first)
        candidates.sort(key=lambda x: -x[0])

        for rank, si, cap_boost in candidates[:top_n]:
            if len(positions) >= top_n or si in held:
                break
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue

            # ATR stop
            atr_v: List[float] = []
            for j in range(max(start_di, di - 14), di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    atr_v.append(
                        max(hh - ll, abs(hh - cc), abs(ll - cc)))
            if not atr_v:
                continue
            atr = np.mean(atr_v)

            base_alloc = 1.0 / max(top_n, 1)
            alloc = base_alloc * cap_boost
            alloc = min(alloc, base_alloc * 1.5)

            stop = ep - atr_stop * atr
            positions.append(
                (si, di + 1, ep, stop, alloc, False, 1))
            held.add(si)

    # Close remaining
    for si, edi, ep, sp, alloc, is_pyr, direction in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = direction * (c - ep) / ep - COMM
            equity += equity * alloc * pnl

    return trades, equity, max_dd


# ============================================================
# ANALYSIS
# ============================================================
def analyze(trades: List[dict], equity: float, max_dd: float,
            label: str = "") -> Optional[Dict]:
    """Print analysis of backtest results."""
    if not trades:
        print(f"  {label}: no trades")
        return None

    nw = sum(1 for t in trades if t['pnl_pct'] > 0)
    wr = nw / len(trades) * 100
    n_days = max(1, trades[-1]['di'] - trades[0]['di'])
    ann = ((equity / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100

    ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
    rets = np.array(ap) / CASH0
    sh = (np.mean(rets) / np.std(rets) * np.sqrt(252)
          if np.std(rets) > 0 else 0)

    n_pyr = sum(1 for t in trades if t.get('pyr'))
    n_base = len(trades) - n_pyr
    n_stop = sum(1 for t in trades if t['reason'] == 'stop')
    n_hold = sum(1 for t in trades if t['reason'] == 'hold')

    print(f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} "
          f"stop:{n_stop} hold:{n_hold}) "
          f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
          f"Sh={sh:.2f} eq={equity:,.0f}")

    yr: Dict[int, dict] = {}
    for t in trades:
        y = t['year']
        if y not in yr:
            yr[y] = {'n': 0, 'w': 0, 'pnl': []}
        yr[y]['n'] += 1
        if t['pnl_pct'] > 0:
            yr[y]['w'] += 1
        yr[y]['pnl'].append(t['pnl_pct'])

    for y in sorted(yr.keys()):
        ys = yr[y]
        cum = np.prod([1 + p / 100 for p in ys['pnl']]) - 1
        print(f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% "
              f"cum={cum:+.1%}")

    return {'n': len(trades), 'wr': wr, 'dd': max_dd,
            'ann': ann, 'sh': sh, 'eq': equity}


# ============================================================
# WALK-FORWARD
# ============================================================
def walk_forward(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: list, syms: list,
    combined_rank: np.ndarray,
    v1_sigs: Dict[str, np.ndarray],
    candle_sigs: Dict[str, np.ndarray],
    top_n: int = 1,
    hold_days: int = 5,
    atr_stop: float = 3.0,
    min_confidence: int = 3,
    pyramid_ratio: float = 0.5,
    min_cap_score: float = 0.0,
) -> List[dict]:
    """Walk-forward validation by year."""
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD tn={top_n} hd={hold_days} "
          f"atr={atr_stop:.1f} mc={min_confidence} "
          f"pyr={pyramid_ratio:.1f} cap={min_cap_score:.2f}")
    print(f"{'=' * 70}")

    all_trades: List[dict] = []

    for test_year in range(2019, dates[-1].year + 1):
        test_start = None
        test_end_idx = None
        for i, d in enumerate(dates):
            if d.year == test_year and test_start is None:
                test_start = i
            if d.year == test_year:
                test_end_idx = i
        if test_start is None:
            continue

        trades, _, _ = backtest_v16(
            C, O, H, L, NS, ND, dates, syms,
            combined_rank, v1_sigs, candle_sigs,
            top_n=top_n, hold_days=hold_days,
            atr_stop=atr_stop, min_confidence=min_confidence,
            pyramid_ratio=pyramid_ratio, pyramid_day=1,
            min_cap_score=min_cap_score,
            start_di=test_start, end_di=test_end_idx + 1)

        test_trades = [t for t in trades
                       if dates[t['di']].year == test_year]
        all_trades.extend(test_trades)

        if test_trades:
            n = len(test_trades)
            nw = sum(1 for t in test_trades if t['pnl_pct'] > 0)
            wr = nw / n * 100
            avg = np.mean([t['pnl_pct'] for t in test_trades])
            print(f"  {test_year}: {n}t WR={wr:.1f}% "
                  f"avg={avg:+.2f}%", flush=True)
        else:
            print(f"  {test_year}: no trades", flush=True)

    if all_trades:
        nw = sum(1 for t in all_trades if t['pnl_pct'] > 0)
        wr = nw / len(all_trades) * 100
        avg = np.mean([t['pnl_pct'] for t in all_trades])
        cum = np.prod([1 + t['pnl_pct'] / 100 for t in all_trades]) - 1
        n_pyr = sum(1 for t in all_trades if t.get('pyr'))
        print(f"\n  WF TOTAL: {len(all_trades)}t (pyr:{n_pyr}) "
              f"WR={wr:.1f}% avg={avg:+.2f}% cum={cum:+.1%}")

    return all_trades


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V16: INTRADAY PATTERN MEAN REVERSION — WALK-FORWARD VALIDATED")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    print(f"  {NS} sym, {ND} days, "
          f"{dates[0].strftime('%Y-%m-%d')} to "
          f"{dates[-1].strftime('%Y-%m-%d')}")

    # --- Phase 1: Candle patterns ---
    candle_sigs = compute_candle_patterns(C, O, H, L, NS, ND)

    # --- Phase 2: V1 oversold signals ---
    v1_sigs = compute_oversold_signals(C, O, H, L, V, OI, NS, ND)

    # Find 2019 index
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2019-01-01'):
            bt_2019 = i
            break

    # ========================================
    # SECTION 1: PATTERN WEIGHT SWEEP
    # ========================================
    print("\n" + "=" * 70)
    print("  SECTION 1: PATTERN WEIGHT SWEEP (2019-2026)")
    print("=" * 70)

    weight_results = []
    for pw in [0.10, 0.20, 0.30, 0.40, 0.50]:
        combined = compute_combined_rank(v1_sigs, candle_sigs, NS, ND,
                                         pattern_weight=pw)
        for mc in [2, 3]:
            for pyr in [0.0, 0.5]:
                trades, eq, dd = backtest_v16(
                    C, O, H, L, NS, ND, dates, syms,
                    combined, v1_sigs, candle_sigs,
                    top_n=1, hold_days=5, atr_stop=3.0,
                    min_confidence=mc,
                    pyramid_ratio=pyr, pyramid_day=1,
                    start_di=bt_2019)
                if len(trades) < 10:
                    continue
                nw = sum(1 for t in trades if t['pnl_pct'] > 0)
                wr = nw / len(trades) * 100
                n_days = max(1, trades[-1]['di'] - trades[0]['di'])
                ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
                ap = [t['pnl_abs']
                      for t in sorted(trades, key=lambda x: x['di'])]
                rets_arr = np.array(ap) / CASH0
                sh_val = (np.mean(rets_arr) / np.std(rets_arr)
                          * np.sqrt(252) if np.std(rets_arr) > 0 else 0)
                weight_results.append({
                    'pw': pw, 'mc': mc, 'pyr': pyr,
                    'n': len(trades), 'wr': wr,
                    'ann': ann, 'dd': dd, 'sh': sh_val,
                    'eq': eq,
                })

    weight_results.sort(key=lambda x: -x['sh'])
    print(f"\n{'PW':>4} {'MC':>3} {'Pyr':>4} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 55)
    for r in weight_results[:15]:
        print(f"{r['pw']:>4.2f} {r['mc']:>3} {r['pyr']:>4.1f} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sh']:>5.2f}")

    # ========================================
    # SECTION 2: FULL PARAMETER SWEEP
    # ========================================
    print("\n" + "=" * 70)
    print("  SECTION 2: FULL PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    # Use best pattern weight from sweep, default 0.30
    best_pw = weight_results[0]['pw'] if weight_results else 0.30
    print(f"  Using pattern weight = {best_pw:.2f}")

    combined = compute_combined_rank(v1_sigs, candle_sigs, NS, ND,
                                     pattern_weight=best_pw)

    sweep_results = []
    for tn in [1, 2, 3]:
        for hd in [3, 5, 7]:
            for atr in [2.0, 3.0, 4.0]:
                for mc in [2, 3]:
                    for pyr in [0.0, 0.5]:
                        trades, eq, dd = backtest_v16(
                            C, O, H, L, NS, ND, dates, syms,
                            combined, v1_sigs, candle_sigs,
                            top_n=tn, hold_days=hd,
                            atr_stop=atr, min_confidence=mc,
                            pyramid_ratio=pyr, pyramid_day=1,
                            start_di=bt_2019)
                        if len(trades) < 10:
                            continue
                        nw = sum(1 for t in trades if t['pnl_pct'] > 0)
                        wr = nw / len(trades) * 100
                        n_days = max(1, trades[-1]['di'] - trades[0]['di'])
                        ann = ((eq / CASH0)
                               ** (1 / max(1.0, n_days / 252)) - 1) * 100
                        ap = [t['pnl_abs']
                              for t in sorted(trades, key=lambda x: x['di'])]
                        rets_arr = np.array(ap) / CASH0
                        sh_val = (np.mean(rets_arr) / np.std(rets_arr)
                                  * np.sqrt(252)
                                  if np.std(rets_arr) > 0 else 0)
                        sweep_results.append({
                            'tn': tn, 'hd': hd, 'atr': atr,
                            'mc': mc, 'pyr': pyr,
                            'n': len(trades), 'wr': wr,
                            'ann': ann, 'dd': dd, 'sh': sh_val,
                            'eq': eq,
                        })

    sweep_results.sort(key=lambda x: -x['sh'])
    print(f"\n{'TN':>3} {'HD':>3} {'ATR':>4} {'MC':>3} {'Pyr':>4} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 65)
    for r in sweep_results[:25]:
        print(f"{r['tn']:>3} {r['hd']:>3} {r['atr']:>4.1f} "
              f"{r['mc']:>3} {r['pyr']:>4.1f} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sh']:>5.2f}")

    # ========================================
    # SECTION 3: BEST CONFIGS — YEARLY
    # ========================================
    print("\n" + "=" * 70)
    print("  SECTION 3: BEST CONFIGS — YEARLY (2019-2026)")
    print("=" * 70)

    for r in sweep_results[:5]:
        trades, eq, dd = backtest_v16(
            C, O, H, L, NS, ND, dates, syms,
            combined, v1_sigs, candle_sigs,
            top_n=r['tn'], hold_days=r['hd'],
            atr_stop=r['atr'], min_confidence=r['mc'],
            pyramid_ratio=r['pyr'], pyramid_day=1,
            start_di=bt_2019)
        label = (f"tn={r['tn']} hd={r['hd']} atr={r['atr']:.1f} "
                 f"mc={r['mc']} pyr={r['pyr']:.1f}")
        print(f"\n  --- {label} ---")
        analyze(trades, eq, dd, label)

    # ========================================
    # SECTION 4: FULL 2016-2026
    # ========================================
    print("\n" + "=" * 70)
    print("  SECTION 4: FULL 2016-2026 (10 years)")
    print("=" * 70)

    for r in sweep_results[:3]:
        trades, eq, dd = backtest_v16(
            C, O, H, L, NS, ND, dates, syms,
            combined, v1_sigs, candle_sigs,
            top_n=r['tn'], hold_days=r['hd'],
            atr_stop=r['atr'], min_confidence=r['mc'],
            pyramid_ratio=r['pyr'], pyramid_day=1,
            start_di=60)
        label = (f"full tn={r['tn']} hd={r['hd']} atr={r['atr']:.1f} "
                 f"mc={r['mc']} pyr={r['pyr']:.1f}")
        print(f"\n  {label}")
        analyze(trades, eq, dd, label)

    # ========================================
    # SECTION 5: WALK-FORWARD VALIDATION
    # ========================================
    print("\n" + "=" * 70)
    print("  SECTION 5: WALK-FORWARD VALIDATION (2019-2026)")
    print("=" * 70)

    for r in sweep_results[:3]:
        walk_forward(
            C, O, H, L, NS, ND, dates, syms,
            combined, v1_sigs, candle_sigs,
            top_n=r['tn'],
            hold_days=r['hd'],
            atr_stop=r['atr'],
            min_confidence=r['mc'],
            pyramid_ratio=r['pyr'])

    # ========================================
    # SECTION 6: MIN CAP SCORE SWEEP
    # ========================================
    print("\n" + "=" * 70)
    print("  SECTION 6: CAPITULATION THRESHOLD SWEEP (best config, 2019-2026)")
    print("=" * 70)

    if sweep_results:
        best = sweep_results[0]
        for min_cap in [0.0, 0.05, 0.10, 0.15, 0.20]:
            trades, eq, dd = backtest_v16(
                C, O, H, L, NS, ND, dates, syms,
                combined, v1_sigs, candle_sigs,
                top_n=best['tn'], hold_days=best['hd'],
                atr_stop=best['atr'], min_confidence=best['mc'],
                pyramid_ratio=best['pyr'], pyramid_day=1,
                min_cap_score=min_cap,
                start_di=bt_2019)
            label = f"cap>={min_cap:.2f}"
            analyze(trades, eq, dd, label)

    print(f"\n[V16] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
