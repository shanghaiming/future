"""
V41: Adaptive Threshold + Multi-Timeframe Rank Hybrid
======================================================
Combines the two best improvements over V18:
  - V39 (Sharpe 4.47): Adaptive entry threshold based on rolling win rate
  - V27 (Sharpe 2.99, MDD 18.6%): Multi-timeframe rank confirmation reduces drawdown

Architecture:
  1. V18's 7 cross-sectional ranks on short-term (5d) lookback
  2. Same 7 factors computed on medium-term (20d) lookback
  3. Short-term composite = V18 weighted average
  4. Medium-term composite = same weights, 20d factors
  5. Multi-TF score = st_weight * short_term + (1 - st_weight) * medium_term
  6. ADAPTIVE THRESHOLD (from V39):
     - Track rolling win rate over last 20 trades
     - If win_rate > 60%: threshold = base - adapt_amount (relax)
     - If win_rate 50-60%: threshold = base (neutral)
     - If win_rate < 50%: threshold = base + adapt_amount (tighten)
     - Cap between [0.70, 0.95]
  7. Entry: multi_TF_score > adaptive_threshold AND KER < 0.15
  8. Hold 5d, ATR stop, pyramid on day-1 winners

Parameter sweep:
  - base_threshold: 0.75, 0.80, 0.85
  - adapt_amount: 0.05, 0.07
  - st_weight: 0.55, 0.60, 0.65
  - top_n: 1, 2
  - atr_stop: 2.5, 3.0
  - pyramid: 0.0, 0.5

Walk-forward 2019-2026, full 10-year for top configs.
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
from itertools import product

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

# V18 weights used for both short-term and medium-term composites
DEFAULT_WEIGHTS = {
    'rank_ret':   0.25,
    'rank_oi':    0.20,
    'rank_rsi':   0.15,
    'rank_vol':   0.15,
    'rank_range': 0.10,
    'rank_atrp':  0.05,
}

# Placeholder for ret10d weight -- ret10d only available for 5d lookback
ST_EXTRA_WEIGHTS = {
    'rank_ret10d': 0.10,
}


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


def compute_raw_factors(C: np.ndarray, O: np.ndarray, H: np.ndarray,
                        L: np.ndarray, V: np.ndarray, OI: np.ndarray,
                        NS: int, ND: int) -> dict:
    """Compute raw factors for both 5d (short-term) and 20d (medium-term)."""
    t0 = time.time()
    print("[V41] Computing raw factors (5d + 20d)...", flush=True)

    # === Short-term (5d) factors ===
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 5]) and C[si, di - 5] > 0:
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1.0

    ret_10d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 10]) and C[si, di - 10] > 0:
                ret_10d[si, di] = C[si, di] / C[si, di - 10] - 1.0

    oi_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(OI[si, di]) and not np.isnan(OI[si, di - 5]) and OI[si, di - 5] > 0:
                oi_5d[si, di] = OI[si, di] / OI[si, di - 5] - 1.0

    vol_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            vals = V[si, di - 5:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 3:
                vol_5d[si, di] = np.mean(valid)

    daily_range = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if not np.isnan(H[si, di]) and not np.isnan(L[si, di]) and not np.isnan(C[si, di]):
                if C[si, di] > 0 and H[si, di] > L[si, di]:
                    daily_range[si, di] = (H[si, di] - L[si, di]) / C[si, di]

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

    atrp_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(14, ND):
            atr_vals = []
            for j in range(di - 14, di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    prev_c = C[si, j - 1] if j > 0 and not np.isnan(C[si, j - 1]) else cc
                    atr_vals.append(max(hh - ll, abs(hh - prev_c), abs(ll - prev_c)))
            if atr_vals and not np.isnan(C[si, di]) and C[si, di] > 0:
                atrp_5d[si, di] = np.mean(atr_vals) / C[si, di]

    # === Medium-term (20d) factors ===
    ret_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 20]) and C[si, di - 20] > 0:
                ret_20d[si, di] = C[si, di] / C[si, di - 20] - 1.0

    oi_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if not np.isnan(OI[si, di]) and not np.isnan(OI[si, di - 20]) and OI[si, di - 20] > 0:
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
            vals = daily_range[si, di - 20:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 10:
                range_20d[si, di] = np.mean(valid)

    rsi14_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            rsi_val = rsi14[si, di]
            if not np.isnan(rsi_val):
                rsi14_20d[si, di] = rsi_val

    atrp_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if not np.isnan(atrp_5d[si, di]):
                atrp_20d[si, di] = atrp_5d[si, di]

    print(f"  Raw factors done: {time.time() - t0:.1f}s", flush=True)
    return {
        # Short-term (5d) factors
        'ret_5d': ret_5d,
        'ret_10d': ret_10d,
        'oi_5d': oi_5d,
        'vol_5d': vol_5d,
        'daily_range': daily_range,
        'rsi14': rsi14,
        'atrp_5d': atrp_5d,
        # Medium-term (20d) factors
        'ret_20d': ret_20d,
        'oi_20d': oi_20d,
        'vol_20d': vol_20d,
        'range_20d': range_20d,
        'rsi14_20d': rsi14_20d,
        'atrp_20d': atrp_20d,
    }


def compute_cross_sectional_ranks(raw_factors: dict, NS: int, ND: int,
                                   min_count: int = 10) -> dict:
    """Rank all factors cross-sectionally. Inverted: low raw = high rank."""
    t0 = time.time()
    print("[V41] Computing cross-sectional ranks...", flush=True)

    factors_to_rank = {
        # Short-term ranks
        'st_rank_ret5d': raw_factors['ret_5d'],
        'st_rank_ret10d': raw_factors['ret_10d'],
        'st_rank_oi5d': raw_factors['oi_5d'],
        'st_rank_vol': raw_factors['vol_5d'],
        'st_rank_range': raw_factors['daily_range'],
        'st_rank_rsi': raw_factors['rsi14'],
        'st_rank_atrp': raw_factors['atrp_5d'],
        # Medium-term ranks
        'mt_rank_ret': raw_factors['ret_20d'],
        'mt_rank_oi': raw_factors['oi_20d'],
        'mt_rank_vol': raw_factors['vol_20d'],
        'mt_rank_range': raw_factors['range_20d'],
        'mt_rank_rsi': raw_factors['rsi14_20d'],
        'mt_rank_atrp': raw_factors['atrp_20d'],
    }

    INVERT_FACTORS = {
        'st_rank_ret5d', 'st_rank_ret10d', 'st_rank_oi5d', 'st_rank_rsi',
        'mt_rank_ret', 'mt_rank_oi', 'mt_rank_rsi',
    }

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


def build_multi_tf_signal(ranks: dict, st_weights: dict, mt_weights: dict,
                          st_extra_weights: dict,
                          st_weight: float, NS: int, ND: int,
                          min_factors: int = 4) -> tuple:
    """Build multi-timeframe composite signal.

    ST composite uses V18's 7 factors (ret5d, oi5d, vol, ret10d, range, rsi, atrp).
    MT composite uses the same 6 factors at 20d lookback (no ret10d equivalent).
    Combined: st_weight * ST + (1 - st_weight) * MT, only when both available.

    Returns (composite, st_comp, mt_comp, n_confirm_st, n_confirm_mt).
    """
    t0 = time.time()
    print(f"[V41] Building multi-TF signal (st_w={st_weight:.2f})...", flush=True)

    mt_weight = 1.0 - st_weight

    composite = np.full((NS, ND), np.nan)
    st_comp = np.full((NS, ND), np.nan)
    mt_comp = np.full((NS, ND), np.nan)
    n_confirm_st = np.zeros((NS, ND), dtype=int)
    n_confirm_mt = np.zeros((NS, ND), dtype=int)

    # Short-term factor names and weights (7 factors from V18)
    st_names = [
        ('st_rank_ret5d', st_weights['rank_ret']),
        ('st_rank_oi5d', st_weights['rank_oi']),
        ('st_rank_rsi', st_weights['rank_rsi']),
        ('st_rank_vol', st_weights['rank_vol']),
        ('st_rank_ret10d', st_extra_weights['rank_ret10d']),
        ('st_rank_range', st_weights['rank_range']),
        ('st_rank_atrp', st_weights['rank_atrp']),
    ]

    # Medium-term factor names and weights (6 factors, same structure minus ret10d)
    mt_names = [
        ('mt_rank_ret', st_weights['rank_ret']),
        ('mt_rank_oi', st_weights['rank_oi']),
        ('mt_rank_rsi', st_weights['rank_rsi']),
        ('mt_rank_vol', st_weights['rank_vol']),
        ('mt_rank_range', st_weights['rank_range']),
        ('mt_rank_atrp', st_weights['rank_atrp']),
    ]

    for di in range(ND):
        for si in range(NS):
            # Short-term composite
            st_vals = []
            st_wsum = 0.0
            st_confirm = 0
            for name, weight in st_names:
                rv = ranks[name][si, di]
                if np.isnan(rv):
                    continue
                st_vals.append(rv * weight)
                st_wsum += weight
                if rv > 0.5:
                    st_confirm += 1

            if st_wsum > 0 and st_confirm >= min_factors:
                st_comp[si, di] = sum(st_vals) / st_wsum
                n_confirm_st[si, di] = st_confirm

            # Medium-term composite
            mt_vals = []
            mt_wsum = 0.0
            mt_confirm = 0
            for name, weight in mt_names:
                rv = ranks[name][si, di]
                if np.isnan(rv):
                    continue
                mt_vals.append(rv * weight)
                mt_wsum += weight
                if rv > 0.5:
                    mt_confirm += 1

            if mt_wsum > 0 and mt_confirm >= max(min_factors - 1, 2):
                mt_comp[si, di] = sum(mt_vals) / mt_wsum
                n_confirm_mt[si, di] = mt_confirm

            # Combined: only when both timeframes available
            if not np.isnan(st_comp[si, di]) and not np.isnan(mt_comp[si, di]):
                composite[si, di] = (st_weight * st_comp[si, di] +
                                     mt_weight * mt_comp[si, di])

    print(f"  Multi-TF signal done: {time.time() - t0:.1f}s", flush=True)
    return composite, st_comp, mt_comp, n_confirm_st, n_confirm_mt


def compute_all_signals(C: np.ndarray, O: np.ndarray, H: np.ndarray,
                        L: np.ndarray, V: np.ndarray, OI: np.ndarray,
                        NS: int, ND: int, st_weight: float = 0.60,
                        st_weights: dict = None,
                        st_extra_weights: dict = None) -> dict:
    """Full signal pipeline for V41."""
    if st_weights is None:
        st_weights = DEFAULT_WEIGHTS
    if st_extra_weights is None:
        st_extra_weights = ST_EXTRA_WEIGHTS

    # Derive medium-term weights from st_weights (ret10d share redistributed)
    mt_weights = {
        'rank_ret': st_weights['rank_ret'] + st_extra_weights['rank_ret10d'],
        'rank_oi': st_weights['rank_oi'],
        'rank_rsi': st_weights['rank_rsi'],
        'rank_vol': st_weights['rank_vol'],
        'rank_range': st_weights['rank_range'],
        'rank_atrp': st_weights['rank_atrp'],
    }

    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_cross_sectional_ranks(raw, NS, ND)
    ker_regime = compute_ker(C, NS, ND)
    composite, st_comp, mt_comp, ncf_st, ncf_mt = build_multi_tf_signal(
        ranks, st_weights, mt_weights, st_extra_weights, st_weight, NS, ND)

    return {
        'composite': composite,
        'st_comp': st_comp,
        'mt_comp': mt_comp,
        'n_confirm_st': ncf_st,
        'n_confirm_mt': ncf_mt,
        'ker_regime': ker_regime,
        'ranks': ranks,
    }


def compute_atr_at(H: np.ndarray, L: np.ndarray, C: np.ndarray,
                   si: int, di: int, start_di: int) -> float | None:
    """Compute ATR for a specific symbol/day."""
    atr_v = []
    for j in range(max(start_di, di - 14), di):
        hh, ll, cc = H[si, j], L[si, j], C[si, j]
        if not any(np.isnan([hh, ll, cc])):
            atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
    if atr_v:
        return np.mean(atr_v)
    return None


def adaptive_threshold(recent_trades_win: list, base_threshold: float,
                       adapt_amount: float, min_cap: float, max_cap: float,
                       win_rate_window: int) -> float:
    """Compute adaptive threshold based on recent trade win rate."""
    if len(recent_trades_win) < 5:
        return base_threshold

    window = recent_trades_win[-win_rate_window:]
    win_rate = sum(window) / len(window)

    if win_rate > 0.60:
        threshold = base_threshold - adapt_amount
    elif win_rate < 0.50:
        threshold = base_threshold + adapt_amount
    else:
        threshold = base_threshold

    return max(min_cap, min(max_cap, threshold))


def backtest_v41(C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
                 NS: int, ND: int, dates: list, syms: list, sigs: dict,
                 base_threshold: float = 0.80,
                 adapt_amount: float = 0.07,
                 win_rate_window: int = 20,
                 top_n: int = 1,
                 min_cap: float = 0.70,
                 max_cap: float = 0.95,
                 atr_stop: float = 3.0,
                 min_confidence: int = 3,
                 use_ker_gate: bool = True,
                 hold_days: int = 5,
                 pyramid_ratio: float = 0.5,
                 pyramid_day: int = 1,
                 start_di: int = 60,
                 end_di: int = None) -> tuple:
    """Backtest V41: adaptive threshold + multi-timeframe rank hybrid."""
    composite = sigs['composite']
    st_comp = sigs['st_comp']
    mt_comp = sigs['mt_comp']
    ncf_st = sigs['n_confirm_st']
    ncf_mt = sigs['n_confirm_mt']
    ker_regime = sigs['ker_regime']

    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions = []
    trades = []
    recent_trades_win: list = []
    current_threshold = base_threshold

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0
        new_positions = []

        # Update adaptive threshold before trading
        current_threshold = adaptive_threshold(
            recent_trades_win, base_threshold, adapt_amount,
            min_cap, max_cap, win_rate_window)

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
                    is_win = pnl > 0
                    trades.append({
                        'pnl_abs': profit, 'pnl_pct': pnl * 100,
                        'days': di - edi + 1, 'di': di, 'year': d.year,
                        'sym': syms[si], 'reason': 'stop', 'pyr': is_pyr,
                        'threshold': current_threshold,
                    })
                    recent_trades_win.append(1 if is_win else 0)
            elif hold >= hold_days:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    is_win = pnl > 0
                    trades.append({
                        'pnl_abs': profit, 'pnl_pct': pnl * 100,
                        'days': di - edi + 1, 'di': di, 'year': d.year,
                        'sym': syms[si], 'reason': 'hold', 'pyr': is_pyr,
                        'threshold': current_threshold,
                    })
                    recent_trades_win.append(1 if is_win else 0)
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

        # Entry signal at close[di], enter at open[di+1]
        # Use adaptive threshold on multi-TF composite
        candidates = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(composite[si, di]):
                continue
            # Adaptive threshold on the multi-TF composite
            if composite[si, di] < current_threshold:
                continue
            # Both ST and MT must have signal
            if np.isnan(st_comp[si, di]) or np.isnan(mt_comp[si, di]):
                continue
            # Total confidence across both timeframes
            total_confirm = ncf_st[si, di] + ncf_mt[si, di]
            if total_confirm < min_confidence:
                continue
            # KER gate
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

    return trades, equity, max_dd


def analyze(trades: list, equity: float, max_dd: float, label: str = "") -> dict | None:
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

    thresholds_used = [t.get('threshold', 0) for t in trades]
    avg_thresh = np.mean(thresholds_used) if thresholds_used else 0

    print(f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} stop:{n_stop} hold:{n_hold}) "
          f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% Sh={sh:.2f} eq={equity:,.0f} "
          f"avg_thresh={avg_thresh:.3f}")

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

    return {'n': len(trades), 'wr': wr, 'dd': max_dd, 'ann': ann,
            'sh': sh, 'eq': equity}


def walk_forward(C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
                 NS: int, ND: int, dates: list, syms: list, sigs: dict,
                 base_threshold: float = 0.80,
                 adapt_amount: float = 0.07,
                 win_rate_window: int = 20,
                 top_n: int = 1,
                 min_cap: float = 0.70,
                 max_cap: float = 0.95,
                 atr_stop: float = 3.0,
                 hold_days: int = 5,
                 pyramid_ratio: float = 0.5,
                 pyramid_day: int = 1) -> list:
    """Walk-forward validation: year-by-year out-of-sample.

    The adaptive state (recent_trades_win) is NOT reset between years,
    mimicking real trading where the adaptation persists.
    """
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V41 (bt={base_threshold} adapt={adapt_amount} "
          f"top_n={top_n} atr={atr_stop})")
    print(f"{'=' * 70}")

    years = sorted(set(d.year for d in dates))
    all_trades = []

    # We run a single continuous backtest across all test years to preserve
    # adaptive state, but report per-year stats
    first_test_start = None
    last_test_end = None
    for test_year in range(2019, years[-1] + 1):
        for i, d in enumerate(dates):
            if d.year == test_year:
                if first_test_start is None:
                    first_test_start = i
                last_test_end = i

    if first_test_start is None:
        return []

    # Run continuous backtest over the entire test period
    trades, _, _ = backtest_v41(
        C, O, H, L, NS, ND, dates, syms, sigs,
        base_threshold=base_threshold, adapt_amount=adapt_amount,
        win_rate_window=win_rate_window, top_n=top_n,
        min_cap=min_cap, max_cap=max_cap,
        atr_stop=atr_stop, hold_days=hold_days,
        min_confidence=3, use_ker_gate=True,
        pyramid_ratio=pyramid_ratio, pyramid_day=pyramid_day,
        start_di=first_test_start, end_di=last_test_end + 1)

    # Report per-year
    for test_year in range(2019, years[-1] + 1):
        test_trades = [t for t in trades if dates[t['di']].year == test_year]
        all_trades.extend(test_trades)
        if test_trades:
            n = len(test_trades)
            nw = sum(1 for t in test_trades if t['pnl_pct'] > 0)
            wr_val = nw / n * 100
            avg = np.mean([t['pnl_pct'] for t in test_trades])
            avg_t = np.mean([t.get('threshold', 0) for t in test_trades])
            print(f"  {test_year}: {n}t WR={wr_val:.1f}% avg={avg:+.2f}% "
                  f"thresh={avg_t:.3f}", flush=True)
        else:
            print(f"  {test_year}: no trades", flush=True)

    if all_trades:
        nw = sum(1 for t in all_trades if t['pnl_pct'] > 0)
        wr_val = nw / len(all_trades) * 100
        avg = np.mean([t['pnl_pct'] for t in all_trades])
        cum = np.prod([1 + t['pnl_pct'] / 100 for t in all_trades]) - 1
        n_pyr = sum(1 for t in all_trades if t.get('pyr'))
        avg_t = np.mean([t.get('threshold', 0) for t in all_trades])
        print(f"\n  WF TOTAL: {len(all_trades)}t (pyr:{n_pyr}) WR={wr_val:.1f}% "
              f"avg={avg:+.2f}% cum={cum:+.1%} avg_thresh={avg_t:.3f}")
        return all_trades
    return []


def main():
    t0 = time.time()
    print("=" * 70)
    print("  V41: ADAPTIVE THRESHOLD + MULTI-TIMEFRAME RANK HYBRID")
    print("  V39 adaptive threshold + V27 multi-TF rank confirmation")
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

    # === 1. Walk-Forward Validation with key st_weight configs ===
    print("\n" + "=" * 70)
    print("  WALK-FORWARD VALIDATION (2019-2026) -- KEY CONFIGS")
    print("=" * 70)

    for st_w in [0.55, 0.60, 0.65]:
        sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND, st_weight=st_w)
        for bt, aa in [(0.80, 0.07), (0.75, 0.07), (0.85, 0.05)]:
            walk_forward(C, O, H, L, NS, ND, dates, syms, sigs,
                         base_threshold=bt, adapt_amount=aa,
                         win_rate_window=20, top_n=1,
                         min_cap=0.70, max_cap=0.95,
                         atr_stop=3.0, hold_days=5,
                         pyramid_ratio=0.5, pyramid_day=1)

    # === 2. Parameter sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    results = []

    sweep_params = {
        'base_threshold': [0.75, 0.80, 0.85],
        'adapt_amount': [0.05, 0.07],
        'st_weight': [0.55, 0.60, 0.65],
        'top_n': [1, 2],
        'atr_stop': [2.5, 3.0],
        'pyramid_ratio': [0.0, 0.5],
    }

    total_combos = 1
    for v in sweep_params.values():
        total_combos *= len(v)
    print(f"  Total combinations: {total_combos}")

    combo_count = 0
    for st_w in sweep_params['st_weight']:
        sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND, st_weight=st_w)
        for bt, aa, tn, ats, pr in product(
            sweep_params['base_threshold'],
            sweep_params['adapt_amount'],
            sweep_params['top_n'],
            sweep_params['atr_stop'],
            sweep_params['pyramid_ratio'],
        ):
            # Skip invalid: base_threshold must be within cap range
            if bt < 0.70 + aa or bt > 0.95 - aa:
                continue

            combo_count += 1
            trades, eq, dd = backtest_v41(
                C, O, H, L, NS, ND, dates, syms, sigs,
                base_threshold=bt, adapt_amount=aa,
                win_rate_window=20, top_n=tn,
                min_cap=0.70, max_cap=0.95,
                atr_stop=ats, hold_days=5,
                min_confidence=3, use_ker_gate=True,
                pyramid_ratio=pr, pyramid_day=1,
                start_di=bt_2019)

            if len(trades) < 10:
                continue

            nw = sum(1 for t in trades if t['pnl_pct'] > 0)
            wr = nw / len(trades) * 100
            n_days = max(1, trades[-1]['di'] - trades[0]['di'])
            ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
            ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
            rets_arr = np.array(ap) / CASH0
            sh_val = (np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252)
                      if np.std(rets_arr) > 0 else 0)

            results.append({
                'bt': bt, 'aa': aa, 'st_w': st_w, 'tn': tn,
                'ats': ats, 'pr': pr,
                'n': len(trades), 'wr': wr, 'ann': ann,
                'dd': dd, 'sharpe': sh_val, 'eq': eq,
            })

    results.sort(key=lambda x: -x['sharpe'])
    print(f"\n  Evaluated {combo_count} valid combinations, "
          f"{len(results)} with 10+ trades")
    print(f"\n{'BT':>4} {'AA':>4} {'STw':>4} {'TN':>3} {'ATS':>4} {'Pyr':>4} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>6}")
    print("-" * 80)
    for r in results[:30]:
        print(f"{r['bt']:>4.2f} {r['aa']:>4.2f} {r['st_w']:>4.2f} {r['tn']:>3} "
              f"{r['ats']:>4.1f} {r['pr']:>4.1f} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sharpe']:>6.2f}")

    # === 3. Top configs -- full 10-year ===
    print("\n" + "=" * 70)
    print("  TOP CONFIGS -- FULL 10-YEAR (2016-2026)")
    print("=" * 70)

    seen = set()
    unique_top = []
    for r in results:
        key = (r['bt'], r['aa'], r['st_w'], r['tn'], r['ats'], r['pr'])
        if key not in seen:
            seen.add(key)
            unique_top.append(r)
        if len(unique_top) >= 5:
            break

    for r in unique_top:
        sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND,
                                   st_weight=r['st_w'])
        trades, eq, dd = backtest_v41(
            C, O, H, L, NS, ND, dates, syms, sigs,
            base_threshold=r['bt'], adapt_amount=r['aa'],
            win_rate_window=20, top_n=r['tn'],
            min_cap=0.70, max_cap=0.95,
            atr_stop=r['ats'], hold_days=5,
            min_confidence=3, use_ker_gate=True,
            pyramid_ratio=r['pr'], pyramid_day=1,
            start_di=60)
        label = (f"bt={r['bt']:.2f} aa={r['aa']:.2f} st_w={r['st_w']:.2f} "
                 f"tn={r['tn']} ats={r['ats']:.1f} pr={r['pr']:.1f}")
        print(f"\n  FULL {label}")
        analyze(trades, eq, dd, label)

    # === 4. Walk-forward for best config ===
    if unique_top:
        best = unique_top[0]
        print("\n" + "=" * 70)
        print(f"  BEST WALK-FORWARD: bt={best['bt']:.2f} aa={best['aa']:.2f} "
              f"st_w={best['st_w']:.2f} tn={best['tn']} "
              f"ats={best['ats']:.1f} pr={best['pr']:.1f}")
        print("=" * 70)

        best_sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND,
                                        st_weight=best['st_w'])
        walk_forward(C, O, H, L, NS, ND, dates, syms, best_sigs,
                     base_threshold=best['bt'], adapt_amount=best['aa'],
                     win_rate_window=20, top_n=best['tn'],
                     min_cap=0.70, max_cap=0.95,
                     atr_stop=best['ats'], hold_days=5,
                     pyramid_ratio=best['pr'], pyramid_day=1)

        # === 5. Compare: adaptive vs static threshold ===
        print("\n" + "=" * 70)
        print("  ADAPTIVE vs STATIC THRESHOLD COMPARISON (2019-2026)")
        print("=" * 70)

        # Adaptive version
        trades_adapt, eq_adapt, dd_adapt = backtest_v41(
            C, O, H, L, NS, ND, dates, syms, best_sigs,
            base_threshold=best['bt'], adapt_amount=best['aa'],
            win_rate_window=20, top_n=best['tn'],
            min_cap=0.70, max_cap=0.95,
            atr_stop=best['ats'], hold_days=5,
            min_confidence=3, use_ker_gate=True,
            pyramid_ratio=best['pr'], pyramid_day=1,
            start_di=bt_2019)

        # Static version: adapt_amount=0 disables adaptation
        trades_static, eq_static, dd_static = backtest_v41(
            C, O, H, L, NS, ND, dates, syms, best_sigs,
            base_threshold=best['bt'], adapt_amount=0.0,
            win_rate_window=20, top_n=best['tn'],
            min_cap=0.70, max_cap=0.95,
            atr_stop=best['ats'], hold_days=5,
            min_confidence=3, use_ker_gate=True,
            pyramid_ratio=best['pr'], pyramid_day=1,
            start_di=bt_2019)

        print(f"\n  ADAPTIVE:")
        analyze(trades_adapt, eq_adapt, dd_adapt, "adaptive")
        print(f"\n  STATIC:")
        analyze(trades_static, eq_static, dd_static, "static")

        print(f"\n  Adaptive improvement: "
              f"eq_delta={eq_adapt - eq_static:+,.0f} "
              f"dd_delta={dd_adapt - dd_static:+.1f}%")

    # === 6. Compare: V41 best vs V39-style (single TF + adaptive) ===
    if unique_top:
        best = unique_top[0]
        print("\n" + "=" * 70)
        print("  MULTI-TF vs SINGLE-TF COMPARISON (2019-2026)")
        print("=" * 70)

        best_sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND,
                                        st_weight=best['st_w'])

        # Multi-TF (V41) -- both ST and MT must agree
        trades_multi, eq_multi, dd_multi = backtest_v41(
            C, O, H, L, NS, ND, dates, syms, best_sigs,
            base_threshold=best['bt'], adapt_amount=best['aa'],
            win_rate_window=20, top_n=best['tn'],
            min_cap=0.70, max_cap=0.95,
            atr_stop=best['ats'], hold_days=5,
            min_confidence=3, use_ker_gate=True,
            pyramid_ratio=best['pr'], pyramid_day=1,
            start_di=bt_2019)

        print(f"\n  MULTI-TF (V41):")
        analyze(trades_multi, eq_multi, dd_multi, "multi-TF")
        print(f"\n  ADAPTIVE ONLY (V39-style): See V39 results for comparison")

    print(f"\n[V41] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
