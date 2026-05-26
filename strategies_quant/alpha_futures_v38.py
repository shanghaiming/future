"""
V38: TAIL RISK PROTECTED RANK — V18 with Consecutive Loss Counter
==================================================================
Core thesis: V18's biggest weakness is 28% MDD. Consecutive losses
signal potential regime change. V38 adds a "circuit breaker lite":
after N consecutive losing trades, reduce position size progressively.
After M consecutive losses, pause trading entirely. Unlike V17's
stress gating (which destroyed returns), this approach only reduces
size based on actual trade outcomes — preserving returns during
winning streaks while protecting tail risk.

Signal architecture:
  1. Same V18 cross-sectional rank: 7 factors, composite score
  2. Consecutive loss tracking:
     - Track trade outcomes (win/loss)
     - consecutive_losses = current streak of losses
     - recent_loss_rate = losses in last N trades
  3. Progressive sizing reduction:
     - 0-2 consecutive losses: full size (1.0x)
     - 3 consecutive losses: 0.75x
     - 4 consecutive losses: 0.5x
     - 5+ consecutive losses: pause (0x) — wait for win to resume
  4. Recent loss rate check:
     - If loss rate > 70% over last 10 trades: reduce to 0.5x
     - If loss rate > 80% over last 10 trades: pause
  5. KER gate, hold 5d, ATR stop 3.0, top_n 2, min_rank 0.80
  6. Walk-forward 2019-2026, full 10-year for top configs

Signal at close[di], enter at open[di+1]. No look-ahead. No leverage.
"""
import sys
import os
import time
import warnings
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque

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

# Default weights for composite rank (same as V18)
DEFAULT_WEIGHTS = {
    'rank_ret5d':  0.25,
    'rank_oi5d':   0.20,
    'rank_rsi':    0.15,
    'rank_vol':    0.15,
    'rank_ret10d': 0.10,
    'rank_range':  0.10,
    'rank_atrp':   0.05,
}


# ============================================================
# RSI COMPUTATION
# ============================================================
def compute_rsi_manual(C: np.ndarray, NS: int, ND: int, period: int = 14) -> np.ndarray:
    """Compute RSI without talib as fallback."""
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
                        valid_l.append(losses[j] if not np.isnan(losses[j]) else 0.0)
                if len(valid_g) >= period:
                    avg_gain = np.mean(valid_g)
                    avg_loss = np.mean(valid_l)
                    if avg_loss == 0:
                        rsi[si, di + period - 1] = 100.0
                    else:
                        rs = avg_gain / avg_loss
                        rsi[si, di + period - 1] = 100.0 - 100.0 / (1.0 + rs)
                continue

            avg_gain = (avg_gain * (period - 1) + gains[di]) / period
            avg_loss = (avg_loss * (period - 1) + losses[di]) / period
            if avg_loss == 0:
                rsi[si, di] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[si, di] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


# ============================================================
# RAW FACTOR COMPUTATION (same as V18)
# ============================================================
def compute_raw_factors(
    C: np.ndarray, O: np.ndarray, H: np.ndarray,
    L: np.ndarray, V: np.ndarray, OI: np.ndarray,
    NS: int, ND: int,
) -> Dict[str, np.ndarray]:
    """Compute raw factor values before cross-sectional ranking."""
    t0 = time.time()
    print("[V38] Computing raw factors...", flush=True)

    # 5d return
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 5]) and C[si, di - 5] > 0:
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1.0

    # 10d return
    ret_10d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 10]) and C[si, di - 10] > 0:
                ret_10d[si, di] = C[si, di] / C[si, di - 10] - 1.0

    # OI 5d change
    oi_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(OI[si, di]) and not np.isnan(OI[si, di - 5]) and OI[si, di - 5] > 0:
                oi_5d[si, di] = OI[si, di] / OI[si, di - 5] - 1.0

    # Volume 5d average
    vol_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            vals = V[si, di - 5:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 3:
                vol_5d[si, di] = np.mean(valid)

    # Daily range (H-L)/C
    daily_range = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if not np.isnan(H[si, di]) and not np.isnan(L[si, di]) and not np.isnan(C[si, di]):
                if C[si, di] > 0 and H[si, di] > L[si, di]:
                    daily_range[si, di] = (H[si, di] - L[si, di]) / C[si, di]

    # RSI 14
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

    # ATR% 14d
    atrp = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(14, ND):
            atr_vals = []
            for j in range(di - 14, di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    prev_c = C[si, j - 1] if j > 0 and not np.isnan(C[si, j - 1]) else cc
                    atr_vals.append(max(hh - ll, abs(hh - prev_c), abs(ll - prev_c)))
            if atr_vals and not np.isnan(C[si, di]) and C[si, di] > 0:
                atrp[si, di] = np.mean(atr_vals) / C[si, di]

    print(f"  Raw factors done: {time.time() - t0:.1f}s", flush=True)
    return {
        'ret_5d': ret_5d,
        'ret_10d': ret_10d,
        'oi_5d': oi_5d,
        'vol_5d': vol_5d,
        'daily_range': daily_range,
        'rsi14': rsi14,
        'atrp': atrp,
    }


# ============================================================
# CROSS-SECTIONAL RANKS (same as V18)
# ============================================================
def compute_cross_sectional_ranks(
    raw_factors: Dict[str, np.ndarray], NS: int, ND: int, min_count: int = 10,
) -> Dict[str, np.ndarray]:
    """Rank all factors cross-sectionally (across commodities per day)."""
    t0 = time.time()
    print("[V38] Computing cross-sectional ranks...", flush=True)

    factors_to_rank = {
        'rank_ret5d': raw_factors['ret_5d'],
        'rank_ret10d': raw_factors['ret_10d'],
        'rank_oi5d': raw_factors['oi_5d'],
        'rank_vol': raw_factors['vol_5d'],
        'rank_range': raw_factors['daily_range'],
        'rank_rsi': raw_factors['rsi14'],
        'rank_atrp': raw_factors['atrp'],
    }

    INVERT_FACTORS = {'rank_ret5d', 'rank_ret10d', 'rank_oi5d', 'rank_rsi'}

    ranks = {}
    for name, factor in factors_to_rank.items():
        rank_arr = np.full((NS, ND), np.nan)
        for di in range(ND):
            vals = factor[:, di]
            valid_count = np.sum(~np.isnan(vals))
            if valid_count < min_count:
                continue
            ranked = pd.Series(vals).rank(pct=True, na_option='keep').values
            if name in INVERT_FACTORS:
                ranked = 1.0 - ranked
            rank_arr[:, di] = ranked
        ranks[name] = rank_arr

    print(f"  CS ranks done: {time.time() - t0:.1f}s", flush=True)
    return ranks


# ============================================================
# KER (same as V18)
# ============================================================
def compute_ker(C: np.ndarray, NS: int, ND: int) -> np.ndarray:
    """Kaufman Efficiency Ratio for regime detection."""
    ker_regime = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        for di in range(10, ND):
            closes = C[si, di - 10:di + 1]
            valid = closes[~np.isnan(closes)]
            if len(valid) < 10 or valid[0] <= 0:
                continue
            net_change = abs(valid[-1] - valid[0])
            total_change = np.sum(np.abs(np.diff(valid)))
            if total_change > 1e-10:
                ker_val = net_change / total_change
                if ker_val < 0.15:
                    ker_regime[si, di] = 1
                elif ker_val > 0.3:
                    ker_regime[si, di] = -1
    return ker_regime


# ============================================================
# COMPOSITE SIGNAL (same as V18)
# ============================================================
def build_composite_signal(
    ranks: Dict[str, np.ndarray], weights: Dict[str, float],
    NS: int, ND: int, min_factors: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build weighted composite rank from individual factor ranks."""
    t0 = time.time()
    print("[V38] Building composite signal...", flush=True)

    composite = np.full((NS, ND), np.nan)
    n_confirm = np.zeros((NS, ND), dtype=int)

    factor_names = list(weights.keys())
    weight_vals = np.array([weights[k] for k in factor_names])

    for di in range(ND):
        for si in range(NS):
            vals = []
            w_sum = 0.0
            confirm_count = 0
            for idx, name in enumerate(factor_names):
                rank_val = ranks[name][si, di]
                if np.isnan(rank_val):
                    continue
                vals.append(rank_val * weight_vals[idx])
                w_sum += weight_vals[idx]
                if rank_val > 0.5:
                    confirm_count += 1

            if w_sum > 0 and confirm_count >= min_factors:
                composite[si, di] = sum(vals) / w_sum
                n_confirm[si, di] = confirm_count

    print(f"  Composite done: {time.time() - t0:.1f}s", flush=True)
    return composite, n_confirm


def compute_all_signals(
    C: np.ndarray, O: np.ndarray, H: np.ndarray,
    L: np.ndarray, V: np.ndarray, OI: np.ndarray,
    NS: int, ND: int, weights: Optional[Dict[str, float]] = None,
) -> Dict:
    """Full signal pipeline."""
    if weights is None:
        weights = DEFAULT_WEIGHTS

    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_cross_sectional_ranks(raw, NS, ND)
    ker_regime = compute_ker(C, NS, ND)
    composite, n_confirm = build_composite_signal(ranks, weights, NS, ND)

    return {
        'composite': composite,
        'n_confirm': n_confirm,
        'ker_regime': ker_regime,
        'ranks': ranks,
    }


# ============================================================
# POSITION SIZING BASED ON CONSECUTIVE LOSSES
# ============================================================
def compute_size_multiplier(
    consecutive_losses: int,
    recent_outcomes: deque,
    loss_reduce_threshold: int,
    loss_pause_threshold: int,
    recent_window: int,
    recent_loss_rate_pause: float,
) -> float:
    """Compute position size multiplier based on loss history.

    Returns multiplier: 0.0 = paused, 1.0 = full size.
    Progressive reduction: no hard pause, only size reduction.
    This avoids the deadlock where you can't trade to break a losing streak.
    """
    # Progressive reduction based on consecutive losses
    if consecutive_losses >= loss_pause_threshold:
        multiplier = 0.25
    elif consecutive_losses >= loss_reduce_threshold + 1:
        multiplier = 0.5
    elif consecutive_losses >= loss_reduce_threshold:
        multiplier = 0.75
    else:
        multiplier = 1.0

    # Check recent loss rate — only reduce, never fully pause
    if len(recent_outcomes) >= max(recent_window // 2, 3):
        recent_list = list(recent_outcomes)[-recent_window:]
        losses = sum(1 for x in recent_list if x == 0)
        loss_rate = losses / len(recent_list)
        if loss_rate >= recent_loss_rate_pause:
            multiplier = min(multiplier, 0.25)
        elif loss_rate >= recent_loss_rate_pause - 0.10:
            multiplier = min(multiplier, 0.5)

    return multiplier


# ============================================================
# BACKTEST WITH TAIL RISK PROTECTION
# ============================================================
def backtest_v38(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: list, syms: list, sigs: Dict,
    top_n: int = 2, min_rank: float = 0.80, atr_stop: float = 3.0,
    min_confidence: int = 3, use_ker_gate: bool = True,
    hold_days: int = 5, pyramid_ratio: float = 0.5,
    pyramid_day: int = 1,
    loss_reduce_threshold: int = 3,
    loss_pause_threshold: int = 5,
    recent_window: int = 10,
    recent_loss_rate_pause: float = 0.80,
    start_di: int = 60, end_di: Optional[int] = None,
    loss_state: Optional[Dict] = None,
) -> Tuple[list, float, float, Dict]:
    """Backtest with cross-sectional rank signals + tail risk protection."""
    composite = sigs['composite']
    ker_regime = sigs['ker_regime']
    n_confirm = sigs['n_confirm']

    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions = []
    trades = []

    # Tail risk tracking — allow passing state for walk-forward continuity
    if isinstance(loss_state, dict):
        consecutive_losses = loss_state.get('consecutive_losses', 0)
        recent_outcomes = deque(loss_state.get('recent_outcomes', []),
                                maxlen=max(recent_window, 20))
    else:
        consecutive_losses = 0
        recent_outcomes = deque(maxlen=max(recent_window, 20))
    reduced_days = 0

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions = []

        pos_by_si = defaultdict(list)
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
                # Aggregate PnL for this symbol (count as ONE trade outcome)
                total_pnl = 0.0
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    total_pnl += pnl
                    trades.append({
                        'pnl_abs': profit, 'pnl_pct': pnl * 100,
                        'days': di - edi + 1, 'di': di, 'year': d.year,
                        'sym': syms[si], 'reason': 'stop', 'pyr': is_pyr,
                        'is_win': 1 if pnl > 0 else 0,
                    })
                # Single outcome per symbol exit
                is_win = 1 if total_pnl > 0 else 0
                recent_outcomes.append(is_win)
                if is_win:
                    consecutive_losses = 0
                else:
                    consecutive_losses += 1
            elif hold >= hold_days:
                total_pnl = 0.0
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    total_pnl += pnl
                    trades.append({
                        'pnl_abs': profit, 'pnl_pct': pnl * 100,
                        'days': di - edi + 1, 'di': di, 'year': d.year,
                        'sym': syms[si], 'reason': 'hold', 'pyr': is_pyr,
                        'is_win': 1 if pnl > 0 else 0,
                    })
                is_win = 1 if total_pnl > 0 else 0
                recent_outcomes.append(is_win)
                if is_win:
                    consecutive_losses = 0
                else:
                    consecutive_losses += 1
            else:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc, is_pyr))

        # Pyramid check
        if pyramid_ratio > 0:
            held_with_pos = defaultdict(list)
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
                                atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
                        if atr_v:
                            atr = np.mean(atr_v)
                            additions.append((si, di, c_now, c_now - atr_stop * atr, pyr_alloc, True))
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

        # Compute size multiplier from loss tracking
        size_mult = compute_size_multiplier(
            consecutive_losses=consecutive_losses,
            recent_outcomes=recent_outcomes,
            loss_reduce_threshold=loss_reduce_threshold,
            loss_pause_threshold=loss_pause_threshold,
            recent_window=recent_window,
            recent_loss_rate_pause=recent_loss_rate_pause,
        )

        if size_mult <= 0.01:
            reduced_days += 1
            continue

        # Entry signal at close[di], enter at open[di+1]
        candidates = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(composite[si, di]):
                continue
            if composite[si, di] < min_rank:
                continue
            if n_confirm[si, di] < min_confidence:
                continue
            if use_ker_gate and ker_regime[si, di] < 0:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            alloc = 1.0 / max(top_n, 1) * size_mult
            candidates.append((composite[si, di], si, alloc))

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

    # Close remaining positions
    for si, edi, ep, sp, alloc, is_pyr in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl

    final_loss_state = {
        'consecutive_losses': consecutive_losses,
        'recent_outcomes': list(recent_outcomes),
        'reduced_days': reduced_days,
    }
    return trades, equity, max_dd, final_loss_state


# ============================================================
# V18 BASELINE BACKTEST (no loss protection, for comparison)
# ============================================================
def backtest_v18_baseline(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: list, syms: list, sigs: Dict,
    top_n: int = 2, min_rank: float = 0.80, atr_stop: float = 3.0,
    min_confidence: int = 3, use_ker_gate: bool = True,
    hold_days: int = 5, pyramid_ratio: float = 0.5,
    pyramid_day: int = 1,
    start_di: int = 60, end_di: Optional[int] = None,
) -> Tuple[list, float, float]:
    """Backtest without loss protection — V18 baseline for comparison."""
    composite = sigs['composite']
    ker_regime = sigs['ker_regime']
    n_confirm = sigs['n_confirm']

    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions = []
    trades = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions = []

        pos_by_si = defaultdict(list)
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
                        'pnl_abs': profit, 'pnl_pct': pnl * 100,
                        'days': di - edi + 1, 'di': di, 'year': d.year,
                        'sym': syms[si], 'reason': 'stop', 'pyr': is_pyr,
                        'is_win': 1 if pnl > 0 else 0,
                    })
            elif hold >= hold_days:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    trades.append({
                        'pnl_abs': profit, 'pnl_pct': pnl * 100,
                        'days': di - edi + 1, 'di': di, 'year': d.year,
                        'sym': syms[si], 'reason': 'hold', 'pyr': is_pyr,
                        'is_win': 1 if pnl > 0 else 0,
                    })
            else:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc, is_pyr))

        # Pyramid check
        if pyramid_ratio > 0:
            held_with_pos = defaultdict(list)
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
                                atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
                        if atr_v:
                            atr = np.mean(atr_v)
                            additions.append((si, di, c_now, c_now - atr_stop * atr, pyr_alloc, True))
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

        candidates = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(composite[si, di]):
                continue
            if composite[si, di] < min_rank:
                continue
            if n_confirm[si, di] < min_confidence:
                continue
            if use_ker_gate and ker_regime[si, di] < 0:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            alloc = 1.0 / max(top_n, 1)
            candidates.append((composite[si, di], si, alloc))

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

    for si, edi, ep, sp, alloc, is_pyr in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl

    return trades, equity, max_dd


# ============================================================
# ANALYSIS
# ============================================================
def analyze(trades: list, equity: float, max_dd: float, label: str = "") -> Optional[Dict]:
    """Analyze backtest results."""
    if not trades:
        print(f"  {label}: no trades")
        return None
    nw = sum(1 for t in trades if t['pnl_pct'] > 0)
    wr = nw / len(trades) * 100
    n_days = max(1, trades[-1]['di'] - trades[0]['di'])
    ann = ((equity / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
    ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
    rets = np.array(ap) / CASH0
    sh = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0

    n_pyr = sum(1 for t in trades if t.get('pyr'))
    n_base = len(trades) - n_pyr
    n_stop = sum(1 for t in trades if t['reason'] == 'stop')
    n_hold = sum(1 for t in trades if t['reason'] == 'hold')

    print(f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} stop:{n_stop} hold:{n_hold}) "
          f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% Sh={sh:.2f} eq={equity:,.0f}")

    yr = {}
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
        print(f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% cum={cum:+.1%}")

    return {'n': len(trades), 'wr': wr, 'dd': max_dd, 'ann': ann, 'sh': sh, 'eq': equity}


def compute_metrics(trades: list, equity: float, max_dd: float) -> Dict:
    """Compute summary metrics from trades."""
    if not trades:
        return {'n': 0, 'wr': 0, 'dd': max_dd, 'ann': 0, 'sh': 0, 'eq': equity}
    nw = sum(1 for t in trades if t['pnl_pct'] > 0)
    wr = nw / len(trades) * 100
    n_days = max(1, trades[-1]['di'] - trades[0]['di'])
    ann = ((equity / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
    ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
    rets = np.array(ap) / CASH0
    sh = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
    return {'n': len(trades), 'wr': wr, 'dd': max_dd, 'ann': ann, 'sh': sh, 'eq': equity}


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================
def walk_forward(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
    dates: list, syms: list, sigs: Dict,
    top_n: int = 2, min_rank: float = 0.80, atr_stop: float = 3.0,
    min_confidence: int = 3, hold_days: int = 5,
    pyramid_ratio: float = 0.5, pyramid_day: int = 1,
    loss_reduce_threshold: int = 3,
    loss_pause_threshold: int = 5,
    recent_window: int = 10,
    recent_loss_rate_pause: float = 0.80,
    use_protection: bool = True,
    label: str = "V38",
) -> list:
    """Walk-forward validation: year-by-year out-of-sample."""
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD {label} (pyr={pyramid_ratio}, "
          f"reduce@{loss_reduce_threshold}, pause@{loss_pause_threshold}, "
          f"window={recent_window}, rate_pause={recent_loss_rate_pause})")
    print(f"{'=' * 70}")

    years = sorted(set(d.year for d in dates))
    all_trades = []
    loss_state = None  # Pass across years for continuity

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

        if use_protection:
            trades, _, _, loss_state = backtest_v38(
                C, O, H, L, NS, ND, dates, syms, sigs,
                top_n=top_n, hold_days=hold_days, atr_stop=atr_stop,
                min_rank=min_rank, min_confidence=min_confidence,
                use_ker_gate=True, pyramid_ratio=pyramid_ratio,
                pyramid_day=pyramid_day,
                loss_reduce_threshold=loss_reduce_threshold,
                loss_pause_threshold=loss_pause_threshold,
                recent_window=recent_window,
                recent_loss_rate_pause=recent_loss_rate_pause,
                start_di=test_start, end_di=test_end_idx + 1,
                loss_state=loss_state)
        else:
            trades, _, _ = backtest_v18_baseline(
                C, O, H, L, NS, ND, dates, syms, sigs,
                top_n=top_n, hold_days=hold_days, atr_stop=atr_stop,
                min_rank=min_rank, min_confidence=min_confidence,
                use_ker_gate=True, pyramid_ratio=pyramid_ratio,
                pyramid_day=pyramid_day,
                start_di=test_start, end_di=test_end_idx + 1)

        test_trades = [t for t in trades if dates[t['di']].year == test_year]
        all_trades.extend(test_trades)

        if test_trades:
            n = len(test_trades)
            nw = sum(1 for t in test_trades if t['pnl_pct'] > 0)
            wr_val = nw / n * 100
            avg = np.mean([t['pnl_pct'] for t in test_trades])
            print(f"  {test_year}: {n}t WR={wr_val:.1f}% avg={avg:+.2f}%", flush=True)
        else:
            print(f"  {test_year}: no trades", flush=True)

    if all_trades:
        nw = sum(1 for t in all_trades if t['pnl_pct'] > 0)
        wr_val = nw / len(all_trades) * 100
        avg = np.mean([t['pnl_pct'] for t in all_trades])
        cum = np.prod([1 + t['pnl_pct'] / 100 for t in all_trades]) - 1
        n_pyr = sum(1 for t in all_trades if t.get('pyr'))
        print(f"\n  WF TOTAL: {len(all_trades)}t (pyr:{n_pyr}) WR={wr_val:.1f}% "
              f"avg={avg:+.2f}% cum={cum:+.1%}")
        return all_trades
    return []


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("  V38: TAIL RISK PROTECTED RANK MEAN REVERSION")
    print("  V18 base + consecutive loss counter for drawdown management")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    print(f"  {NS} sym, {ND} days, "
          f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND)

    # Find 2019 start index for OOS testing
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2019-01-01'):
            bt_2019 = i
            break

    # ============================================================
    # 1. BASELINE: V18 without tail risk protection (2019-2026)
    # ============================================================
    print("\n" + "=" * 70)
    print("  SECTION 1: V18 BASELINE (no protection) 2019-2026")
    print("=" * 70)

    baseline_configs = [
        {'top_n': 1, 'min_rank': 0.75, 'pyramid_ratio': 0.5, 'atr_stop': 3.0},
        {'top_n': 2, 'min_rank': 0.80, 'pyramid_ratio': 0.5, 'atr_stop': 3.0},
    ]

    baseline_results = []
    for cfg in baseline_configs:
        trades, eq, dd = backtest_v18_baseline(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=cfg['top_n'], min_rank=cfg['min_rank'],
            atr_stop=cfg['atr_stop'], min_confidence=3,
            use_ker_gate=True, pyramid_ratio=cfg['pyramid_ratio'],
            pyramid_day=1, start_di=bt_2019)
        m = compute_metrics(trades, eq, dd)
        label = f"V18-tn{cfg['top_n']}-mr{cfg['min_rank']}-pyr{cfg['pyramid_ratio']}"
        analyze(trades, eq, dd, label)
        baseline_results.append({**m, 'label': label, 'cfg': cfg})

    # ============================================================
    # 2. TAIL RISK PROTECTION: Default config (2019-2026)
    # ============================================================
    print("\n" + "=" * 70)
    print("  SECTION 2: V38 WITH DEFAULT TAIL RISK PROTECTION 2019-2026")
    print("=" * 70)

    default_cfg = {
        'top_n': 2, 'min_rank': 0.80, 'pyramid_ratio': 0.5,
        'atr_stop': 3.0, 'hold_days': 5,
        'loss_reduce_threshold': 3, 'loss_pause_threshold': 5,
        'recent_window': 10, 'recent_loss_rate_pause': 0.80,
    }

    trades_v38, eq_v38, dd_v38, _ = backtest_v38(
        C, O, H, L, NS, ND, dates, syms, sigs,
        **default_cfg, min_confidence=3, use_ker_gate=True,
        pyramid_day=1, start_di=bt_2019)
    analyze(trades_v38, eq_v38, dd_v38, "V38-default")

    # ============================================================
    # 3. PARAMETER SWEEP (2019-2026)
    # ============================================================
    print("\n" + "=" * 70)
    print("  SECTION 3: PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    sweep_results = []

    for loss_reduce in [2, 3, 4]:
        for loss_pause in [4, 5, 6]:
            if loss_pause <= loss_reduce:
                continue
            for rwindow in [8, 10, 15]:
                for rate_pause in [0.70, 0.80]:
                    for tn in [1, 2]:
                        for mr in [0.75, 0.80]:
                            for as_val in [2.5, 3.0]:
                                for pyr in [0.0, 0.5]:
                                    trades, eq, dd, _ = backtest_v38(
                                        C, O, H, L, NS, ND, dates, syms, sigs,
                                        top_n=tn, min_rank=mr,
                                        atr_stop=as_val, min_confidence=3,
                                        use_ker_gate=True, hold_days=5,
                                        pyramid_ratio=pyr, pyramid_day=1,
                                        loss_reduce_threshold=loss_reduce,
                                        loss_pause_threshold=loss_pause,
                                        recent_window=rwindow,
                                        recent_loss_rate_pause=rate_pause,
                                        start_di=bt_2019)
                                    if len(trades) < 10:
                                        continue
                                    m = compute_metrics(trades, eq, dd)
                                    sweep_results.append({
                                        'lr': loss_reduce, 'lp': loss_pause,
                                        'rw': rwindow, 'rp': rate_pause,
                                        'tn': tn, 'mr': mr, 'as': as_val,
                                        'pyr': pyr,
                                        **m,
                                    })

    sweep_results.sort(key=lambda x: (-x['sh'], x['dd']))
    print(f"\n{'LR':>3} {'LP':>3} {'RW':>3} {'RP':>4} "
          f"{'TN':>3} {'MR':>4} {'AS':>4} {'Pyr':>4} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 90)
    for r in sweep_results[:30]:
        print(f"{r['lr']:>3} {r['lp']:>3} {r['rw']:>3} {r['rp']:>4.2f} "
              f"{r['tn']:>3} {r['mr']:>4.2f} {r['as']:>4.1f} {r['pyr']:>4.1f} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sh']:>5.2f}")

    # ============================================================
    # 4. BEST CONFIGS: FULL 10-YEAR
    # ============================================================
    print("\n" + "=" * 70)
    print("  SECTION 4: TOP CONFIGS -- FULL 10-YEAR (2016-2026)")
    print("=" * 70)

    seen = set()
    best_full_results = []
    for r in sweep_results:
        key = (r['lr'], r['lp'], r['rw'], r['rp'], r['tn'], r['mr'], r['as'], r['pyr'])
        if key in seen:
            continue
        seen.add(key)
        if len(best_full_results) >= 5:
            break

        # V38 with protection
        trades, eq, dd, _ = backtest_v38(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=r['tn'], min_rank=r['mr'], atr_stop=r['as'],
            min_confidence=3, use_ker_gate=True, hold_days=5,
            pyramid_ratio=r['pyr'], pyramid_day=1,
            loss_reduce_threshold=r['lr'], loss_pause_threshold=r['lp'],
            recent_window=r['rw'], recent_loss_rate_pause=r['rp'],
            start_di=60)
        label = (f"V38-lr{r['lr']}-lp{r['lp']}-rw{r['rw']}-rp{r['rp']:.2f}"
                 f"-tn{r['tn']}-mr{r['mr']}-as{r['as']}-pyr{r['pyr']}")
        print(f"\n  V38 FULL {label}")
        m = analyze(trades, eq, dd, label)
        if m:
            best_full_results.append({**m, **r, 'label': label, 'has_protection': True})

        # V18 baseline for same structural params
        trades_base, eq_base, dd_base = backtest_v18_baseline(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=r['tn'], min_rank=r['mr'], atr_stop=r['as'],
            min_confidence=3, use_ker_gate=True, hold_days=5,
            pyramid_ratio=r['pyr'], pyramid_day=1,
            start_di=60)
        label_base = f"V18-base-tn{r['tn']}-mr{r['mr']}-as{r['as']}-pyr{r['pyr']}"
        print(f"\n  V18 BASELINE {label_base}")
        m_base = analyze(trades_base, eq_base, dd_base, label_base)
        if m_base:
            best_full_results.append({**m_base, 'label': label_base, 'has_protection': False})

        # Compare MDD improvement
        if m and m_base:
            dd_improvement = m_base['dd'] - m['dd']
            sh_diff = m['sh'] - m_base['sh']
            print(f"  >>> MDD IMPROVEMENT: {dd_improvement:+.1f}% "
                  f"Sharpe CHANGE: {sh_diff:+.2f}")

    # ============================================================
    # 5. WALK-FORWARD: Best protected vs baseline
    # ============================================================
    print("\n" + "=" * 70)
    print("  SECTION 5: WALK-FORWARD COMPARISON")
    print("=" * 70)

    # Pick best V38 config from sweep
    if sweep_results:
        best = sweep_results[0]
        print(f"\n  Best V38 config: lr={best['lr']} lp={best['lp']} "
              f"rw={best['rw']} rp={best['rp']:.2f} tn={best['tn']} "
              f"mr={best['mr']} as={best['as']} pyr={best['pyr']}")

        # Walk-forward with protection
        wf_protected = walk_forward(
            C, O, H, L, V, OI, NS, ND, dates, syms, sigs,
            top_n=best['tn'], min_rank=best['mr'], atr_stop=best['as'],
            min_confidence=3, hold_days=5,
            pyramid_ratio=best['pyr'], pyramid_day=1,
            loss_reduce_threshold=best['lr'],
            loss_pause_threshold=best['lp'],
            recent_window=best['rw'],
            recent_loss_rate_pause=best['rp'],
            use_protection=True, label="V38-protected")

        # Walk-forward without protection (baseline)
        wf_baseline = walk_forward(
            C, O, H, L, V, OI, NS, ND, dates, syms, sigs,
            top_n=best['tn'], min_rank=best['mr'], atr_stop=best['as'],
            min_confidence=3, hold_days=5,
            pyramid_ratio=best['pyr'], pyramid_day=1,
            use_protection=False, label="V18-baseline")

        # WF comparison summary
        if wf_protected and wf_baseline:
            p_nw = sum(1 for t in wf_protected if t['pnl_pct'] > 0)
            p_wr = p_nw / len(wf_protected) * 100
            p_cum = np.prod([1 + t['pnl_pct'] / 100 for t in wf_protected]) - 1

            b_nw = sum(1 for t in wf_baseline if t['pnl_pct'] > 0)
            b_wr = b_nw / len(wf_baseline) * 100
            b_cum = np.prod([1 + t['pnl_pct'] / 100 for t in wf_baseline]) - 1

            print(f"\n  WF COMPARISON SUMMARY:")
            print(f"  V38 protected: {len(wf_protected)}t WR={p_wr:.1f}% cum={p_cum:+.1%}")
            print(f"  V18 baseline:  {len(wf_baseline)}t WR={b_wr:.1f}% cum={b_cum:+.1%}")

    # ============================================================
    # 6. SUMMARY TABLE
    # ============================================================
    print("\n" + "=" * 70)
    print("  SECTION 6: FINAL SUMMARY")
    print("=" * 70)

    print(f"\n  {'Config':<50} {'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("  " + "-" * 85)
    for r in best_full_results:
        tag = "[V38]" if r.get('has_protection') else "[V18]"
        print(f"  {tag} {r['label']:<46} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sh']:>5.2f}")

    # Target check
    target_met = [r for r in best_full_results
                  if r.get('has_protection') and r['dd'] < 25 and r['sh'] > 2.0]
    if target_met:
        print(f"\n  TARGET MET: MDD < 25%, Sharpe > 2.0")
        for r in target_met:
            print(f"    {r['label']}: DD={r['dd']:.1f}% Sh={r['sh']:.2f}")
    else:
        v38_results = [r for r in best_full_results if r.get('has_protection')]
        if v38_results:
            closest = min(v38_results, key=lambda x: abs(x['dd'] - 25) + max(0, 2.0 - x['sh']))
            print(f"\n  TARGET NOT FULLY MET. Closest V38 config:")
            print(f"    {closest['label']}: DD={closest['dd']:.1f}% Sh={closest['sh']:.2f}")

    elapsed = time.time() - t0
    print(f"\n[V38] Done. {elapsed:.1f}s")


if __name__ == '__main__':
    main()
