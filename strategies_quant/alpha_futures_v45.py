"""
V45: ULTIMATE MULTI-LAYER STRATEGY
====================================
FINAL synthesis combining ALL proven improvements:

Layer 1 (V27): Multi-timeframe rank confirmation
  - Short-term (5d) composite: 4 ST factors
  - Medium-term (20d) composite: 3 MT factors
  - Combined = st_weight * ST + (1-st_weight) * MT
  - Entry requires BOTH ST and MT above min_rank (lower than adaptive threshold)

Layer 2 (V39): Adaptive threshold
  - Rolling win rate over 20 trades
  - Dynamic threshold: base +/- adapt_amount
  - Applied to the COMBINED composite

Layer 3 (V40): Breadth filter
  - A/D ratio < max_ad (market broadly oversold)
  - Skip when market conditions don't favor MR

Layer 4 (V38): Tail risk protection
  - After loss_reduce consecutive losses: reduce size to 0.5x
  - After loss_pause consecutive losses: pause until next win

Layer 5: Standard parameters
  - KER gate < 0.15
  - Hold 5d, ATR stop 3.0
  - Pyramid on day-1 winners (0.5)

Each layer is independently toggleable for ablation study.

Signal at close[di], enter at open[di+1]. No look-ahead. No leverage.
Walk-forward validation required.
"""
import sys
import os
import time
import warnings
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
from itertools import product

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

# Multi-timeframe weights (from V27)
ST_WEIGHTS = {
    'rank_ret5d':  0.30,
    'rank_oi5d':   0.25,
    'rank_rsi5d':  0.25,
    'rank_vol5d':  0.20,
}

MT_WEIGHTS = {
    'rank_ret20d': 0.40,
    'rank_oi20d':  0.35,
    'rank_vol20d':  0.25,
}

# V18 weights for single-TF mode (ablation)
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
def compute_rsi_manual(C: np.ndarray, NS: int, ND: int,
                       period: int = 14) -> np.ndarray:
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
# RAW FACTOR COMPUTATION
# ============================================================
def compute_raw_factors(C: np.ndarray, O: np.ndarray, H: np.ndarray,
                        L: np.ndarray, V: np.ndarray, OI: np.ndarray,
                        NS: int, ND: int) -> Dict[str, np.ndarray]:
    t0 = time.time()
    print("[V45] Computing raw factors...", flush=True)

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

    # 20d return
    ret_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 20]) and C[si, di - 20] > 0:
                ret_20d[si, di] = C[si, di] / C[si, di - 20] - 1.0

    # OI 5d change
    oi_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(OI[si, di]) and not np.isnan(OI[si, di - 5]) and OI[si, di - 5] > 0:
                oi_5d[si, di] = OI[si, di] / OI[si, di - 5] - 1.0

    # OI 20d change
    oi_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if not np.isnan(OI[si, di]) and not np.isnan(OI[si, di - 20]) and OI[si, di - 20] > 0:
                oi_20d[si, di] = OI[si, di] / OI[si, di - 20] - 1.0

    # Volume 5d average
    vol_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            vals = V[si, di - 5:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 3:
                vol_5d[si, di] = np.mean(valid)

    # Volume 20d average
    vol_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            vals = V[si, di - 20:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 10:
                vol_20d[si, di] = np.mean(valid)

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

    # RSI 5 (for short-term)
    rsi5 = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nan_mask = np.isnan(C[si])
            try:
                r = talib.RSI(c, 5)
                rsi5[si] = np.where(nan_mask, np.nan, r)
            except Exception:
                pass
    needs_fallback5 = np.all(np.isnan(rsi5), axis=1)
    if needs_fallback5.any():
        rsi5_manual = compute_rsi_manual(C, NS, ND, 5)
        for si in range(NS):
            if needs_fallback5[si]:
                rsi5[si] = rsi5_manual[si]

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
        'ret_5d': ret_5d, 'ret_10d': ret_10d, 'ret_20d': ret_20d,
        'oi_5d': oi_5d, 'oi_20d': oi_20d,
        'vol_5d': vol_5d, 'vol_20d': vol_20d,
        'daily_range': daily_range,
        'rsi14': rsi14, 'rsi5': rsi5,
        'atrp': atrp,
    }


# ============================================================
# CROSS-SECTIONAL RANKS
# ============================================================
def compute_cross_sectional_ranks(
    raw_factors: Dict[str, np.ndarray], NS: int, ND: int,
    min_count: int = 10,
) -> Dict[str, np.ndarray]:
    t0 = time.time()
    print("[V45] Computing cross-sectional ranks...", flush=True)

    # All unique factors needed across MTF and single-TF modes
    all_factors = {
        'rank_ret5d': raw_factors['ret_5d'],
        'rank_ret10d': raw_factors['ret_10d'],
        'rank_ret20d': raw_factors['ret_20d'],
        'rank_oi5d': raw_factors['oi_5d'],
        'rank_oi20d': raw_factors['oi_20d'],
        'rank_vol': raw_factors['vol_5d'],
        'rank_vol5d': raw_factors['vol_5d'],
        'rank_vol20d': raw_factors['vol_20d'],
        'rank_range': raw_factors['daily_range'],
        'rank_rsi': raw_factors['rsi14'],
        'rank_rsi5d': raw_factors['rsi5'],
        'rank_atrp': raw_factors['atrp'],
    }

    # Factors to invert: low raw value -> high rank (most oversold)
    INVERT_FACTORS = {
        'rank_ret5d', 'rank_ret10d', 'rank_ret20d',
        'rank_oi5d', 'rank_oi20d',
        'rank_rsi', 'rank_rsi5d',
    }

    ranks = {}
    for name, factor in all_factors.items():
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
# KER
# ============================================================
def compute_ker(C: np.ndarray, NS: int, ND: int) -> np.ndarray:
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
# MULTI-TF COMPOSITE (V27-style)
# ============================================================
def build_multi_tf_signal(ranks: Dict[str, np.ndarray],
                          st_weights: Dict[str, float],
                          mt_weights: Dict[str, float],
                          st_weight: float,
                          NS: int, ND: int,
                          min_factors: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t0 = time.time()
    print(f"[V45] Building multi-TF signal (st_w={st_weight:.2f})...", flush=True)

    composite = np.full((NS, ND), np.nan)
    st_comp = np.full((NS, ND), np.nan)
    mt_comp = np.full((NS, ND), np.nan)

    st_names = list(st_weights.keys())
    st_wvals = np.array([st_weights[k] for k in st_names])
    mt_names = list(mt_weights.keys())
    mt_wvals = np.array([mt_weights[k] for k in mt_names])

    for di in range(ND):
        for si in range(NS):
            # Short-term composite
            st_vals = []
            st_wsum = 0.0
            st_confirm = 0
            for idx, name in enumerate(st_names):
                rv = ranks[name][si, di]
                if np.isnan(rv):
                    continue
                st_vals.append(rv * st_wvals[idx])
                st_wsum += st_wvals[idx]
                if rv > 0.5:
                    st_confirm += 1
            if st_wsum > 0 and st_confirm >= min_factors:
                st_comp[si, di] = sum(st_vals) / st_wsum

            # Medium-term composite
            mt_vals = []
            mt_wsum = 0.0
            mt_confirm = 0
            for idx, name in enumerate(mt_names):
                rv = ranks[name][si, di]
                if np.isnan(rv):
                    continue
                mt_vals.append(rv * mt_wvals[idx])
                mt_wsum += mt_wvals[idx]
                if rv > 0.5:
                    mt_confirm += 1
            if mt_wsum > 0 and mt_confirm >= min_factors:
                mt_comp[si, di] = sum(mt_vals) / mt_wsum

            # Combined composite: only when both timeframes available
            if not np.isnan(st_comp[si, di]) and not np.isnan(mt_comp[si, di]):
                composite[si, di] = (st_weight * st_comp[si, di] +
                                     (1.0 - st_weight) * mt_comp[si, di])

    print(f"  Multi-TF done: {time.time() - t0:.1f}s", flush=True)
    return composite, st_comp, mt_comp


# ============================================================
# SINGLE-TF COMPOSITE (V18-style, for ablation)
# ============================================================
def build_single_tf_composite(ranks: Dict[str, np.ndarray],
                               weights: Dict[str, float],
                               NS: int, ND: int,
                               min_factors: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    print("[V45] Building single-TF composite...", flush=True)

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

    print(f"  Single-TF done: {time.time() - t0:.1f}s", flush=True)
    return composite, n_confirm


# ============================================================
# MARKET BREADTH (V40-style)
# ============================================================
def compute_market_breadth(C: np.ndarray, NS: int, ND: int) -> np.ndarray:
    """Compute daily A/D ratio (fraction of commodities with positive 5d return)."""
    t0 = time.time()
    print("[V45] Computing market breadth...", flush=True)

    ad_ratio = np.full(ND, np.nan)
    for di in range(20, ND):
        rets = []
        for si in range(NS):
            c_now = C[si, di]
            c_5d = C[si, di - 5]
            if not np.isnan(c_now) and not np.isnan(c_5d) and c_5d > 0:
                rets.append(c_now / c_5d - 1.0)
        if len(rets) >= 10:
            ad_ratio[di] = sum(1 for r in rets if r > 0) / len(rets)

    print(f"  Breadth done: {time.time() - t0:.1f}s", flush=True)
    return ad_ratio


# ============================================================
# ADAPTIVE THRESHOLD (V39-style)
# ============================================================
def adaptive_threshold(
    recent_trades_win: List[int],
    base_threshold: float,
    adapt_amount: float,
    win_rate_window: int,
    min_cap: float = 0.70,
    max_cap: float = 0.95,
) -> float:
    if len(recent_trades_win) < 5:
        return base_threshold
    window = recent_trades_win[-win_rate_window:]
    win_rate = sum(window) / len(window)
    if win_rate > 0.60:
        return max(min_cap, base_threshold - adapt_amount)
    elif win_rate < 0.50:
        return min(max_cap, base_threshold + adapt_amount)
    return base_threshold


# ============================================================
# TAIL RISK PROTECTION (V38-style)
# ============================================================
def compute_size_multiplier(consecutive_losses: int,
                            loss_reduce: int,
                            loss_pause: int) -> float:
    if consecutive_losses >= loss_pause:
        return 0.0
    elif consecutive_losses >= loss_reduce:
        return 0.5
    return 1.0


# ============================================================
# SIGNAL COMPUTATION PIPELINE
# ============================================================
def compute_signals(
    C, O, H, L, V, OI, NS, ND,
    use_mtf=True, st_weight=0.60,
) -> Dict:
    """Compute signals once. Stores all layers' data."""
    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_cross_sectional_ranks(raw, NS, ND)
    ker_regime = compute_ker(C, NS, ND)
    ad_ratio = compute_market_breadth(C, NS, ND)

    if use_mtf:
        composite, st_comp, mt_comp = build_multi_tf_signal(
            ranks, ST_WEIGHTS, MT_WEIGHTS, st_weight, NS, ND)
        # n_confirm not used in MTF mode (MTF has its own confirmation)
        n_confirm = np.full((NS, ND), 4, dtype=int)
    else:
        composite, n_confirm = build_single_tf_composite(
            ranks, DEFAULT_WEIGHTS, NS, ND)
        st_comp = np.full((NS, ND), np.nan)
        mt_comp = np.full((NS, ND), np.nan)

    return {
        'composite': composite,
        'n_confirm': n_confirm,
        'ker_regime': ker_regime,
        'ranks': ranks,
        'ad_ratio': ad_ratio,
        'st_comp': st_comp,
        'mt_comp': mt_comp,
    }


# ============================================================
# ATR HELPER
# ============================================================
def compute_atr_at(H, L, C, si, di, start_di):
    atr_v = []
    for j in range(max(start_di, di - 14), di):
        hh, ll, cc = H[si, j], L[si, j], C[si, j]
        if not any(np.isnan([hh, ll, cc])):
            atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
    if atr_v:
        return np.mean(atr_v)
    return None


# ============================================================
# BACKTEST (ALL LAYERS)
# ============================================================
def backtest_v45(
    C, O, H, L, NS, ND, dates, syms, sigs,
    # Layer toggles
    use_mtf: bool = True,
    use_adaptive: bool = True,
    use_breadth: bool = True,
    use_tail_risk: bool = True,
    # Layer 2: Adaptive threshold
    base_threshold: float = 0.80,
    adapt_amount: float = 0.05,
    win_rate_window: int = 20,
    min_cap: float = 0.70,
    max_cap: float = 0.95,
    # Layer 3: Breadth
    max_ad: float = 0.45,
    # Layer 4: Tail risk
    loss_reduce: int = 3,
    loss_pause: int = 5,
    # Layer 5: Standard
    top_n: int = 1,
    atr_stop: float = 3.0,
    min_confidence: int = 3,
    use_ker_gate: bool = True,
    hold_days: int = 5,
    pyramid_ratio: float = 0.5,
    pyramid_day: int = 1,
    start_di: int = 60,
    end_di: Optional[int] = None,
    trade_state: Optional[Dict] = None,
):
    """Backtest V45 with all layers toggleable."""
    composite = sigs['composite']
    ker_regime = sigs['ker_regime']
    n_confirm = sigs['n_confirm']
    ad_ratio = sigs['ad_ratio']
    st_comp = sigs.get('st_comp')
    mt_comp = sigs.get('mt_comp')

    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions = []
    trades = []

    # State for adaptive + tail risk
    if isinstance(trade_state, dict):
        recent_trades_win = list(trade_state.get('recent_trades_win', []))
        consecutive_losses = trade_state.get('consecutive_losses', 0)
    else:
        recent_trades_win = []
        consecutive_losses = 0

    # MTF sub-threshold: always lower than the main threshold to ensure
    # the MTF layer acts as a confirmation filter, not a blocker
    mtf_sub_threshold = 0.60  # Fixed: both ST and MT must exceed this

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions = []

        # Compute current adaptive threshold
        if use_adaptive:
            current_threshold = adaptive_threshold(
                recent_trades_win, base_threshold, adapt_amount,
                win_rate_window, min_cap, max_cap)
        else:
            current_threshold = base_threshold

        # Compute tail risk multiplier
        if use_tail_risk:
            size_mult = compute_size_multiplier(
                consecutive_losses, loss_reduce, loss_pause)
        else:
            size_mult = 1.0

        # Process existing positions
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

            if stopped or hold >= hold_days:
                total_pnl = 0.0
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    total_pnl += pnl
                    trades.append({
                        'pnl_abs': profit, 'pnl_pct': pnl * 100,
                        'days': di - edi + 1, 'di': di, 'year': d.year,
                        'sym': syms[si],
                        'reason': 'stop' if stopped else 'hold',
                        'pyr': is_pyr,
                    })
                is_win = 1 if total_pnl > 0 else 0
                recent_trades_win.append(is_win)
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
                        pyr_alloc = base_alloc * pyramid_ratio * size_mult
                        c_now = C[si, di]
                        atr = compute_atr_at(H, L, C, si, di, start_di)
                        if atr is not None:
                            additions.append(
                                (si, di, c_now, c_now - atr_stop * atr, pyr_alloc, True))
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

        # LAYER 4: Tail risk pause
        if size_mult <= 0.01:
            continue

        # LAYER 3: Breadth filter -- only trade when market is oversold
        if use_breadth:
            if np.isnan(ad_ratio[di]) or ad_ratio[di] >= max_ad:
                continue

        # LAYER 1 + 2 + 5: Entry signal at close[di], enter at open[di+1]
        candidates = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(composite[si, di]):
                continue

            # Combined composite must exceed adaptive threshold
            if composite[si, di] < current_threshold:
                continue

            # LAYER 1: MTF confirmation (lower sub-threshold, not adaptive)
            if use_mtf and st_comp is not None and mt_comp is not None:
                if np.isnan(st_comp[si, di]) or st_comp[si, di] < mtf_sub_threshold:
                    continue
                if np.isnan(mt_comp[si, di]) or mt_comp[si, di] < mtf_sub_threshold:
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
            atr = compute_atr_at(H, L, C, si, di, start_di)
            if atr is None:
                continue
            positions.append((si, di + 1, ep, ep - atr_stop * atr, alloc, False))
            held.add(si)

    # Close remaining positions
    for si, edi, ep, sp, alloc, is_pyr in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl

    final_state = {
        'recent_trades_win': recent_trades_win,
        'consecutive_losses': consecutive_losses,
    }
    return trades, equity, max_dd, final_state


# ============================================================
# ANALYSIS
# ============================================================
def analyze(trades: list, equity: float, max_dd: float,
            label: str = "") -> Optional[Dict]:
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
    n_stop = sum(1 for t in trades if t['reason'] == 'stop')
    n_hold = sum(1 for t in trades if t['reason'] == 'hold')

    print(f"  {label}: {len(trades)}t (pyr:{n_pyr} stop:{n_stop} hold:{n_hold}) "
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
    C, O, H, L, NS, ND, dates, syms, sigs,
    use_mtf=True, use_adaptive=True, use_breadth=True, use_tail_risk=True,
    base_threshold=0.80, adapt_amount=0.05, win_rate_window=20,
    min_cap=0.70, max_cap=0.95,
    max_ad=0.45, loss_reduce=3, loss_pause=5,
    top_n=1, atr_stop=3.0, hold_days=5,
    pyramid_ratio=0.5, pyramid_day=1,
    label="V45",
):
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD {label}")
    print(f"  MTF={use_mtf} ADAPT={use_adaptive} BREADTH={use_breadth} TAIL={use_tail_risk}")
    print(f"  bt={base_threshold} aa={adapt_amount} max_ad={max_ad} "
          f"lr={loss_reduce} lp={loss_pause} tn={top_n} pyr={pyramid_ratio}")
    print(f"{'=' * 70}")

    years = sorted(set(d.year for d in dates))
    all_trades = []
    trade_state = None

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

        trades, _, _, trade_state = backtest_v45(
            C, O, H, L, NS, ND, dates, syms, sigs,
            use_mtf=use_mtf, use_adaptive=use_adaptive,
            use_breadth=use_breadth, use_tail_risk=use_tail_risk,
            base_threshold=base_threshold, adapt_amount=adapt_amount,
            win_rate_window=win_rate_window,
            min_cap=min_cap, max_cap=max_cap,
            max_ad=max_ad, loss_reduce=loss_reduce, loss_pause=loss_pause,
            top_n=top_n, atr_stop=atr_stop, hold_days=hold_days,
            pyramid_ratio=pyramid_ratio, pyramid_day=pyramid_day,
            start_di=test_start, end_di=test_end_idx + 1,
            trade_state=trade_state)

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
# ABLATION STUDY
# ============================================================
def ablation_study(
    C, O, H, L, NS, ND, dates, syms, sigs_mtf, sigs_single,
    base_threshold, adapt_amount, max_ad, loss_reduce, loss_pause,
    top_n, pyramid_ratio, atr_stop, start_di, label_prefix="",
):
    print(f"\n{'=' * 70}")
    print(f"  ABLATION STUDY {label_prefix}")
    print(f"{'=' * 70}")

    base_params = dict(
        base_threshold=base_threshold, adapt_amount=adapt_amount,
        max_ad=max_ad, loss_reduce=loss_reduce, loss_pause=loss_pause,
        top_n=top_n, pyramid_ratio=pyramid_ratio, atr_stop=atr_stop,
        hold_days=5, pyramid_day=1, start_di=start_di,
    )

    configs = [
        ("ALL LAYERS (MTF+ADAPT+BRD+TAIL)", sigs_mtf, True, True, True, True),
        ("  -MTF  (ADAPT+BRD+TAIL)",        sigs_mtf, True, True, True, False),
        ("  -ADAPT (MTF+BRD+TAIL)",          sigs_mtf, True, False, True, True),
        ("  -BREADTH (MTF+ADAPT+TAIL)",      sigs_mtf, True, True, False, True),
        ("  -TAIL (MTF+ADAPT+BRD)",          sigs_mtf, True, True, True, False),
        ("  -MTF-BREADTH (ADAPT+TAIL)",      sigs_mtf, True, False, False, True),
        ("  BASELINE (no layers)",           sigs_single, False, False, False, False),
    ]

    results = []
    for name, sigs, has_mtf, has_adapt, has_breadth, has_tail in configs:
        # For the -TAIL config with MTF, use_mtf=True, use_tail_risk=False
        use_mtf_flag = has_mtf if name != "  -TAIL (MTF+ADAPT+BRD)" else True
        use_tail_flag = has_tail if name != "  -MTF  (ADAPT+BRD+TAIL)" else True

        actual_use_mtf = name != "  -MTF  (ADAPT+BRD+TAIL)" and name != "  BASELINE (no layers)"
        actual_use_tail = name != "  -TAIL (MTF+ADAPT+BRD)" and name != "  -MTF-BREADTH (ADAPT+TAIL)" and name != "  BASELINE (no layers)"
        actual_use_adapt = name != "  -ADAPT (MTF+BRD+TAIL)" and name != "  -MTF-BREADTH (ADAPT+TAIL)" and name != "  BASELINE (no layers)"
        actual_use_breadth = name != "  -BREADTH (MTF+ADAPT+TAIL)" and name != "  -MTF-BREADTH (ADAPT+TAIL)" and name != "  BASELINE (no layers)"

        trades, eq, dd, _ = backtest_v45(
            C, O, H, L, NS, ND, dates, syms, sigs,
            use_mtf=actual_use_mtf,
            use_adaptive=actual_use_adapt,
            use_breadth=actual_use_breadth,
            use_tail_risk=actual_use_tail,
            **base_params,
        )
        m = compute_metrics(trades, eq, dd)
        m['name'] = name
        results.append(m)
        print(f"  {name:<40} {m['n']:>4}t WR={m['wr']:>5.1f}% "
              f"ann={m['ann']:>+7.1f}% DD={m['dd']:>5.1f}% Sh={m['sh']:>5.2f}")

    return results


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("  V45: ULTIMATE MULTI-LAYER STRATEGY")
    print("  Combining ALL proven improvements: MTF + Adaptive + Breadth + Tail Risk")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    print(f"  {NS} sym, {ND} days, "
          f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    # Find 2019 start index for OOS testing
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2019-01-01'):
            bt_2019 = i
            break

    # ============================================================
    # 1. PRE-COMPUTE ALL SIGNALS (both st_weight variants + single-TF)
    # ============================================================
    print("\n" + "=" * 70)
    print("  SECTION 1: PRE-COMPUTING SIGNALS")
    print("=" * 70)

    print("\n--- MTF st_weight=0.55 ---")
    sigs_055 = compute_signals(C, O, H, L, V, OI, NS, ND, use_mtf=True, st_weight=0.55)

    print("\n--- MTF st_weight=0.60 ---")
    sigs_060 = compute_signals(C, O, H, L, V, OI, NS, ND, use_mtf=True, st_weight=0.60)

    print("\n--- Single-TF (V18-style, for ablation) ---")
    sigs_single = compute_signals(C, O, H, L, V, OI, NS, ND, use_mtf=False)

    sigs_map = {0.55: sigs_055, 0.60: sigs_060}

    # ============================================================
    # 2. DEFAULT CONFIG WALK-FORWARD
    # ============================================================
    print("\n" + "=" * 70)
    print("  SECTION 2: WALK-FORWARD VALIDATION (2019-2026) -- DEFAULT")
    print("=" * 70)

    default_configs = [
        (0.80, 0.05, 0.45, 3, 5, 1, 0.5),
        (0.80, 0.07, 0.45, 3, 5, 1, 0.5),
        (0.75, 0.05, 0.45, 3, 5, 1, 0.5),
        (0.80, 0.05, 0.45, 3, 5, 1, 0.0),
        (0.85, 0.05, 0.45, 3, 5, 1, 0.5),
        (0.80, 0.05, 0.45, 4, 6, 2, 0.5),
    ]

    for bt, aa, mad, lr, lp, tn, pyr in default_configs:
        walk_forward(
            C, O, H, L, NS, ND, dates, syms, sigs_060,
            use_mtf=True, use_adaptive=True, use_breadth=True, use_tail_risk=True,
            base_threshold=bt, adapt_amount=aa,
            max_ad=mad, loss_reduce=lr, loss_pause=lp,
            top_n=tn, atr_stop=3.0, hold_days=5,
            pyramid_ratio=pyr, pyramid_day=1,
            label=f"bt={bt} aa={aa} mad={mad} lr={lr} lp={lp} tn={tn} pyr={pyr}")

    # ============================================================
    # 3. PARAMETER SWEEP (2019-2026)
    # ============================================================
    print("\n" + "=" * 70)
    print("  SECTION 3: PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    sweep_results = []

    for bt, aa, stw, mad, lr, lp, tn, pyr in product(
        [0.80, 0.85],        # base_threshold
        [0.05, 0.07],        # adapt_amount
        [0.55, 0.60],        # st_weight
        [0.40, 0.45],        # max_ad
        [3, 4],              # loss_reduce
        [5, 6],              # loss_pause
        [1, 2],              # top_n
        [0.0, 0.5],          # pyramid
    ):
        if lp <= lr:
            continue

        use_sigs = sigs_map[stw]

        trades, eq, dd, _ = backtest_v45(
            C, O, H, L, NS, ND, dates, syms, use_sigs,
            use_mtf=True, use_adaptive=True, use_breadth=True, use_tail_risk=True,
            base_threshold=bt, adapt_amount=aa,
            win_rate_window=20, min_cap=0.70, max_cap=0.95,
            max_ad=mad, loss_reduce=lr, loss_pause=lp,
            top_n=tn, atr_stop=3.0, hold_days=5,
            pyramid_ratio=pyr, pyramid_day=1,
            start_di=bt_2019)

        if len(trades) < 10:
            continue

        m = compute_metrics(trades, eq, dd)
        sweep_results.append({
            'bt': bt, 'aa': aa, 'stw': stw, 'mad': mad,
            'lr': lr, 'lp': lp, 'tn': tn, 'pyr': pyr,
            **m,
        })

    sweep_results.sort(key=lambda x: (-x['sh'], x['dd']))
    print(f"\n  Evaluated {len(sweep_results)} configs with 10+ trades")
    print(f"\n{'BT':>4} {'AA':>4} {'STw':>4} {'MAD':>4} {'LR':>3} {'LP':>3} "
          f"{'TN':>3} {'Pyr':>4} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>6}")
    print("-" * 90)
    for r in sweep_results[:30]:
        print(f"{r['bt']:>4.2f} {r['aa']:>4.2f} {r['stw']:>4.2f} {r['mad']:>4.2f} "
              f"{r['lr']:>3} {r['lp']:>3} {r['tn']:>3} {r['pyr']:>4.1f} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sh']:>6.2f}")

    # ============================================================
    # 4. TOP CONFIGS: FULL 10-YEAR
    # ============================================================
    print("\n" + "=" * 70)
    print("  SECTION 4: TOP CONFIGS -- FULL 10-YEAR (2016-2026)")
    print("=" * 70)

    seen = set()
    best_full_results = []
    for r in sweep_results:
        key = (r['bt'], r['aa'], r['stw'], r['mad'], r['lr'], r['lp'], r['tn'], r['pyr'])
        if key in seen:
            continue
        seen.add(key)
        if len(best_full_results) >= 5:
            break

        use_sigs = sigs_map[r['stw']]

        trades, eq, dd, _ = backtest_v45(
            C, O, H, L, NS, ND, dates, syms, use_sigs,
            use_mtf=True, use_adaptive=True, use_breadth=True, use_tail_risk=True,
            base_threshold=r['bt'], adapt_amount=r['aa'],
            win_rate_window=20, min_cap=0.70, max_cap=0.95,
            max_ad=r['mad'], loss_reduce=r['lr'], loss_pause=r['lp'],
            top_n=r['tn'], atr_stop=3.0, hold_days=5,
            pyramid_ratio=r['pyr'], pyramid_day=1,
            start_di=60)
        label = (f"bt={r['bt']:.2f} aa={r['aa']:.2f} stw={r['stw']:.2f} "
                 f"mad={r['mad']:.2f} lr={r['lr']} lp={r['lp']} "
                 f"tn={r['tn']} pyr={r['pyr']:.1f}")
        print(f"\n  FULL {label}")
        m = analyze(trades, eq, dd, label)
        if m:
            best_full_results.append({**m, **r, 'label': label})

    # ============================================================
    # 5. WALK-FORWARD FOR BEST CONFIG
    # ============================================================
    if sweep_results:
        best = sweep_results[0]
        best_sigs = sigs_map[best['stw']]

        print("\n" + "=" * 70)
        print(f"  SECTION 5: BEST CONFIG WALK-FORWARD")
        print(f"  bt={best['bt']:.2f} aa={best['aa']:.2f} stw={best['stw']:.2f} "
              f"mad={best['mad']:.2f} lr={best['lr']} lp={best['lp']} "
              f"tn={best['tn']} pyr={best['pyr']:.1f}")
        print("=" * 70)

        walk_forward(
            C, O, H, L, NS, ND, dates, syms, best_sigs,
            use_mtf=True, use_adaptive=True, use_breadth=True, use_tail_risk=True,
            base_threshold=best['bt'], adapt_amount=best['aa'],
            win_rate_window=20, min_cap=0.70, max_cap=0.95,
            max_ad=best['mad'], loss_reduce=best['lr'], loss_pause=best['lp'],
            top_n=best['tn'], atr_stop=3.0, hold_days=5,
            pyramid_ratio=best['pyr'], pyramid_day=1,
            label="BEST V45")

        # ============================================================
        # 6. ABLATION STUDY
        # ============================================================
        ablation_study(
            C, O, H, L, NS, ND, dates, syms,
            best_sigs, sigs_single,
            base_threshold=best['bt'],
            adapt_amount=best['aa'],
            max_ad=best['mad'],
            loss_reduce=best['lr'],
            loss_pause=best['lp'],
            top_n=best['tn'],
            pyramid_ratio=best['pyr'],
            atr_stop=3.0,
            start_di=bt_2019,
            label_prefix=f"({best['bt']:.2f}/{best['aa']:.2f}/{best['mad']:.2f}/"
                         f"{best['lr']}/{best['lp']}/{best['tn']}/{best['pyr']:.1f})")

    # ============================================================
    # 7. SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("  SECTION 7: FINAL SUMMARY")
    print("=" * 70)

    print(f"\n  {'Config':<60} {'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("  " + "-" * 95)
    for r in best_full_results:
        print(f"  {r.get('label', 'unknown'):<60} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sh']:>5.2f}")

    # Target check
    target_met = [r for r in best_full_results
                  if r['sh'] > 4.0 and r['dd'] < 15 and r['ann'] > 20]
    if target_met:
        print(f"\n  *** TARGET MET: Sharpe > 4.0, MDD < 15%, Ann > 20% ***")
        for r in target_met:
            print(f"    {r.get('label', '')}: Sh={r['sh']:.2f} DD={r['dd']:.1f}% Ann={r['ann']:+.1f}%")
    else:
        near_target = sorted(best_full_results, key=lambda x: -x['sh'])[:3]
        if near_target:
            print(f"\n  Best configs (sorted by Sharpe):")
            for r in near_target:
                print(f"    {r.get('label', '')}: Sh={r['sh']:.2f} DD={r['dd']:.1f}% Ann={r['ann']:+.1f}%")

    elapsed = time.time() - t0
    print(f"\n[V45] Done. {elapsed:.1f}s")


if __name__ == '__main__':
    main()
