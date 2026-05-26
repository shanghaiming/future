"""
V31: Regime-Adaptive Rank Mean Reversion
=========================================
Combines V18 (cross-sectional rank, WF +2019.6%, Sharpe 2.39) with
V8 (adaptive market-state parameters, 17.6% ann Sharpe 1.36).

Core thesis: V18's rank methodology is the all-time best signal, but it
uses fixed position sizing regardless of market volatility regime. V31
adapts position sizing AND rank threshold based on cross-sectional
market volatility:

  - LOW_VOL regime (< 25th pct): normal sizing, min_rank=0.75
  - MID_VOL regime (25-75th pct): normal sizing, min_rank=0.80
  - HIGH_VOL regime (> 75th pct): reduced sizing (0.7x), min_rank=0.85
    (require more extreme oversold signals in turbulent markets)

Signal architecture (from V18):
  1. Compute 7 cross-sectional ranks across 50 commodities per day
  2. Composite rank = weighted average:
     - rank_ret5d:  0.25  (low return = oversold)
     - rank_oi5d:   0.20  (declining OI + price drop = capitulation)
     - rank_vol:    0.20  (high volume = attention)
     - rank_ret10d: 0.10
     - rank_range:  0.10  (range expansion = capitulation)
     - rank_rsi:    0.10  (low RSI = oversold)
     - rank_atrp:   0.05  (high ATR% = opportunity)
  3. KER gate (avoid trending regimes), confidence >= 2
  4. Pyramid on day-1 winners (ratio 0.5)
  5. Hold 5d, ATR stop 3.0
  6. Walk-forward 2019-2026

Signal at close[di], enter at open[di+1]. No look-ahead. No leverage.
"""
import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

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

# Default weights for composite rank (V18 style with vol bumped to 0.20)
DEFAULT_WEIGHTS = {
    'rank_ret5d':  0.25,
    'rank_oi5d':   0.20,
    'rank_vol':    0.20,
    'rank_ret10d': 0.10,
    'rank_range':  0.10,
    'rank_rsi':    0.10,
    'rank_atrp':   0.05,
}


# ============================================================
# PHASE 1: FACTOR COMPUTATION (from V18)
# ============================================================
def compute_rsi_manual(C: np.ndarray, NS: int, ND: int,
                       period: int = 14) -> np.ndarray:
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


def compute_raw_factors(C: np.ndarray, O: np.ndarray, H: np.ndarray,
                        L: np.ndarray, V: np.ndarray, OI: np.ndarray,
                        NS: int, ND: int) -> Dict[str, np.ndarray]:
    """Compute raw factor values before cross-sectional ranking."""
    t0 = time.time()
    print("[V31] Computing raw factors...", flush=True)

    # --- 5d return ---
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 5]) and C[si, di - 5] > 0:
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1.0

    # --- 10d return ---
    ret_10d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 10]) and C[si, di - 10] > 0:
                ret_10d[si, di] = C[si, di] / C[si, di - 10] - 1.0

    # --- OI 5d change ---
    oi_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(OI[si, di]) and not np.isnan(OI[si, di - 5]) and OI[si, di - 5] > 0:
                oi_5d[si, di] = OI[si, di] / OI[si, di - 5] - 1.0

    # --- Volume (5d average for stability) ---
    vol_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            vals = V[si, di - 5:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 3:
                vol_5d[si, di] = np.mean(valid)

    # --- Daily range (H-L) / C ---
    daily_range = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if not np.isnan(H[si, di]) and not np.isnan(L[si, di]) and not np.isnan(C[si, di]):
                if C[si, di] > 0 and H[si, di] > L[si, di]:
                    daily_range[si, di] = (H[si, di] - L[si, di]) / C[si, di]

    # --- RSI 14 ---
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

    # --- ATR% (14d) ---
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


def compute_cross_sectional_ranks(raw_factors: Dict[str, np.ndarray],
                                   NS: int, ND: int,
                                   min_count: int = 10) -> Dict[str, np.ndarray]:
    """Rank all factors cross-sectionally (across commodities per day)."""
    t0 = time.time()
    print("[V31] Computing cross-sectional ranks...", flush=True)

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
                    ker_regime[si, di] = 1   # sideways -> good for MR
                elif ker_val > 0.3:
                    ker_regime[si, di] = -1  # trending -> avoid
    return ker_regime


def build_composite_signal(ranks: Dict[str, np.ndarray],
                           weights: Dict[str, float],
                           NS: int, ND: int,
                           min_factors: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Build weighted composite rank from individual factor ranks."""
    t0 = time.time()
    print("[V31] Building composite signal...", flush=True)

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


# ============================================================
# PHASE 2: MARKET VOLATILITY REGIME (from V8, adapted)
# ============================================================
def compute_market_vol_regime(C: np.ndarray, H: np.ndarray, L: np.ndarray,
                              NS: int, ND: int,
                              vol_window: int = 20) -> np.ndarray:
    """
    Compute rolling market-wide ATR% across all instruments.
    Returns per-day market_vol array (average ATR% across all commodities).
    Uses a rolling window of `vol_window` days.
    """
    t0 = time.time()
    print(f"[V31] Computing market volatility regime (window={vol_window})...", flush=True)

    # Per-instrument ATR% using `vol_window` period
    inst_atrp = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(vol_window + 1, ND):
            atr_vals = []
            for j in range(di - vol_window, di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if any(np.isnan([hh, ll, cc])):
                    continue
                prev_c = C[si, j - 1] if j > 0 and not np.isnan(C[si, j - 1]) else cc
                atr_vals.append(max(hh - ll, abs(hh - prev_c), abs(ll - prev_c)))
            if atr_vals and not np.isnan(C[si, di]) and C[si, di] > 0:
                inst_atrp[si, di] = np.mean(atr_vals) / C[si, di]

    # Cross-sectional average ATR% per day = market volatility
    market_vol = np.full(ND, np.nan)
    for di in range(ND):
        vals = inst_atrp[:, di]
        valid = vals[~np.isnan(vals)]
        if len(valid) >= 5:
            market_vol[di] = np.mean(valid)

    # Compute percentiles for regime thresholds (from full history)
    valid_mv = market_vol[~np.isnan(market_vol)]
    p25 = float(np.percentile(valid_mv, 25))
    p75 = float(np.percentile(valid_mv, 75))
    print(f"  Market vol range: {np.nanmin(market_vol):.4f} - {np.nanmax(market_vol):.4f}")
    print(f"  Regime thresholds: LOW < {p25:.4f} (p25), HIGH > {p75:.4f} (p75)")
    print(f"  Done: {time.time() - t0:.1f}s", flush=True)

    return market_vol


# ============================================================
# PHASE 3: FULL SIGNAL PIPELINE
# ============================================================
def compute_all_signals(C: np.ndarray, O: np.ndarray, H: np.ndarray,
                        L: np.ndarray, V: np.ndarray, OI: np.ndarray,
                        NS: int, ND: int,
                        weights: Optional[Dict[str, float]] = None,
                        vol_window: int = 20) -> Dict:
    """Full signal pipeline: V18 ranks + market volatility regime."""
    if weights is None:
        weights = DEFAULT_WEIGHTS

    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_cross_sectional_ranks(raw, NS, ND)
    ker_regime = compute_ker(C, NS, ND)
    composite, n_confirm = build_composite_signal(ranks, weights, NS, ND)
    market_vol = compute_market_vol_regime(C, H, L, NS, ND, vol_window=vol_window)

    # Compute regime percentiles
    valid_mv = market_vol[~np.isnan(market_vol)]
    p25 = float(np.percentile(valid_mv, 25))
    p75 = float(np.percentile(valid_mv, 75))

    return {
        'composite': composite,
        'n_confirm': n_confirm,
        'ker_regime': ker_regime,
        'ranks': ranks,
        'market_vol': market_vol,
        'vol_p25': p25,
        'vol_p75': p75,
    }


# ============================================================
# PHASE 4: REGIME-ADAPTIVE BACKTEST
# ============================================================
def backtest_v31(C: np.ndarray, O: np.ndarray, H: np.ndarray,
                 L: np.ndarray, NS: int, ND: int,
                 dates: pd.DatetimeIndex, syms: List[str],
                 sigs: Dict,
                 top_n: int = 1,
                 atr_stop: float = 3.0,
                 min_confidence: int = 2,
                 hold_days: int = 5,
                 pyramid_ratio: float = 0.5,
                 pyramid_day: int = 1,
                 high_vol_mult: float = 0.7,
                 high_vol_rank: float = 0.85,
                 mid_vol_rank: float = 0.80,
                 low_vol_rank: float = 0.75,
                 start_di: int = 60,
                 end_di: Optional[int] = None) -> Tuple[List[dict], float, float]:
    """
    Backtest with regime-adaptive position sizing and rank threshold.

    Regime logic:
      - LOW_VOL (< p25): normal sizing (1.0x), min_rank = low_vol_rank
      - MID_VOL (p25-p75): normal sizing (1.0x), min_rank = mid_vol_rank
      - HIGH_VOL (> p75): reduced sizing (high_vol_mult), min_rank = high_vol_rank
    """
    composite = sigs['composite']
    ker_regime = sigs['ker_regime']
    n_confirm = sigs['n_confirm']
    market_vol = sigs['market_vol']
    vol_p25 = sigs['vol_p25']
    vol_p75 = sigs['vol_p75']

    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions = []
    trades = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0
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

        # --- Regime-adaptive entry ---
        mv = market_vol[di]
        if np.isnan(mv):
            # No market vol data: use mid regime defaults
            size_mult = 1.0
            min_rank = mid_vol_rank
        elif mv < vol_p25:
            size_mult = 1.0
            min_rank = low_vol_rank
        elif mv > vol_p75:
            size_mult = high_vol_mult
            min_rank = high_vol_rank
        else:
            size_mult = 1.0
            min_rank = mid_vol_rank

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
            if ker_regime[si, di] < 0:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            alloc = size_mult / max(top_n, 1)
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

    return trades, equity, max_dd


# ============================================================
# PHASE 5: ANALYSIS
# ============================================================
def analyze(trades: List[dict], equity: float, max_dd: float,
            label: str = "") -> Optional[Dict]:
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


# ============================================================
# PHASE 6: WALK-FORWARD VALIDATION
# ============================================================
def walk_forward(C: np.ndarray, O: np.ndarray, H: np.ndarray,
                 L: np.ndarray, NS: int, ND: int,
                 dates: pd.DatetimeIndex, syms: List[str],
                 sigs: Dict,
                 pyramid_ratio: float = 0.5, pyramid_day: int = 1,
                 top_n: int = 1, min_confidence: int = 2,
                 hold_days: int = 5, atr_stop: float = 3.0,
                 high_vol_mult: float = 0.7,
                 high_vol_rank: float = 0.85,
                 mid_vol_rank: float = 0.80,
                 low_vol_rank: float = 0.75) -> List[dict]:
    """Walk-forward validation: year-by-year out-of-sample."""
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V31 (pyr={pyramid_ratio}, hvm={high_vol_mult}, "
          f"hvr={high_vol_rank}, mvr={mid_vol_rank}, lvr={low_vol_rank})")
    print(f"{'=' * 70}")

    years = sorted(set(d.year for d in dates))
    all_trades = []

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

        trades, _, _ = backtest_v31(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=top_n, hold_days=hold_days, atr_stop=atr_stop,
            min_confidence=min_confidence,
            pyramid_ratio=pyramid_ratio, pyramid_day=pyramid_day,
            high_vol_mult=high_vol_mult,
            high_vol_rank=high_vol_rank,
            mid_vol_rank=mid_vol_rank,
            low_vol_rank=low_vol_rank,
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
def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V31: REGIME-ADAPTIVE RANK MEAN REVERSION")
    print("  V18 rank + V8 market state adaptation")
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
    # SECTION 1: BASELINE — Default params (V31 defaults)
    # ============================================================
    print("\n" + "=" * 70)
    print("  SIGNAL COMPUTATION (default weights, vol_window=20)")
    print("=" * 70)

    sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND, vol_window=20)

    # ============================================================
    # SECTION 2: FULL 10-YEAR BASELINE
    # ============================================================
    print("\n" + "=" * 70)
    print("  FULL 2016-2026 (10 years) -- BASELINE PROFILES")
    print("=" * 70)

    profiles = [
        (0.0, 0.7, 0.85, 0.80, 0.75, "No pyramid, regime-adaptive"),
        (0.5, 0.7, 0.85, 0.80, 0.75, "Pyramid 0.5, regime-adaptive"),
        (0.5, 1.0, 0.80, 0.80, 0.80, "Pyramid 0.5, NO regime (uniform 0.80)"),
        (0.5, 1.0, 0.75, 0.75, 0.75, "Pyramid 0.5, NO regime (uniform 0.75)"),
    ]

    for pyr, hvm, hvr, mvr, lvr, label in profiles:
        trades, eq, dd = backtest_v31(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=1, hold_days=5, atr_stop=3.0,
            min_confidence=2,
            pyramid_ratio=pyr, pyramid_day=1,
            high_vol_mult=hvm, high_vol_rank=hvr,
            mid_vol_rank=mvr, low_vol_rank=lvr,
            start_di=60)
        print(f"\n  {label}")
        analyze(trades, eq, dd, label)

    # ============================================================
    # SECTION 3: WALK-FORWARD VALIDATION
    # ============================================================
    print("\n" + "=" * 70)
    print("  WALK-FORWARD VALIDATION (2019-2026)")
    print("=" * 70)

    for pyr in [0.0, 0.5]:
        walk_forward(C, O, H, L, NS, ND, dates, syms, sigs,
                     pyramid_ratio=pyr, pyramid_day=1,
                     top_n=1, min_confidence=2, hold_days=5, atr_stop=3.0,
                     high_vol_mult=0.7, high_vol_rank=0.85,
                     mid_vol_rank=0.80, low_vol_rank=0.75)

    # ============================================================
    # SECTION 4: PARAMETER SWEEP (2019-2026)
    # ============================================================
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    # Pre-compute signals for different vol_windows
    sigs_cache = {20: sigs}
    for vw in [15, 30]:
        sigs_cache[vw] = compute_all_signals(C, O, H, L, V, OI, NS, ND, vol_window=vw)

    results = []

    for vw in [15, 20, 30]:
        for hvm in [0.5, 0.7, 0.8]:
            for hvr in [0.85, 0.90]:
                for tn in [1, 2, 3]:
                    for pyr in [0.0, 0.5]:
                        for ats in [2.5, 3.0]:
                            cur_sigs = sigs_cache[vw]
                            trades, eq, dd = backtest_v31(
                                C, O, H, L, NS, ND, dates, syms, cur_sigs,
                                top_n=tn, hold_days=5, atr_stop=ats,
                                min_confidence=2,
                                pyramid_ratio=pyr, pyramid_day=1,
                                high_vol_mult=hvm, high_vol_rank=hvr,
                                mid_vol_rank=0.80, low_vol_rank=0.75,
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
                                'vw': vw, 'hvm': hvm, 'hvr': hvr,
                                'tn': tn, 'pyr': pyr, 'ats': ats,
                                'n': len(trades), 'wr': wr, 'ann': ann,
                                'dd': dd, 'sharpe': sh_val, 'eq': eq,
                            })

    results.sort(key=lambda x: -x['sharpe'])
    print(f"\n{'VW':>3} {'HVM':>4} {'HVR':>4} {'TN':>3} {'Pyr':>4} {'ATS':>4} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 75)
    for r in results[:30]:
        print(f"{r['vw']:>3} {r['hvm']:>4.1f} {r['hvr']:>4.2f} {r['tn']:>3} "
              f"{r['pyr']:>4.1f} {r['ats']:>4.1f} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sharpe']:>5.2f}")

    # ============================================================
    # SECTION 5: BEST CONFIGS — FULL 10-YEAR
    # ============================================================
    if results:
        print("\n" + "=" * 70)
        print("  BEST CONFIGS -- FULL 10-YEAR (2016-2026)")
        print("=" * 70)

        for r in results[:5]:
            cur_sigs = sigs_cache[r['vw']]
            trades, eq, dd = backtest_v31(
                C, O, H, L, NS, ND, dates, syms, cur_sigs,
                top_n=r['tn'], hold_days=5, atr_stop=r['ats'],
                min_confidence=2,
                pyramid_ratio=r['pyr'], pyramid_day=1,
                high_vol_mult=r['hvm'], high_vol_rank=r['hvr'],
                mid_vol_rank=0.80, low_vol_rank=0.75,
                start_di=60)
            label = (f"vw={r['vw']} hvm={r['hvm']:.1f} hvr={r['hvr']:.2f} "
                     f"tn={r['tn']} pyr={r['pyr']:.1f} ats={r['ats']:.1f}")
            print(f"\n  FULL {label}")
            analyze(trades, eq, dd, label)

    # ============================================================
    # SECTION 6: BEST CONFIG — WALK-FORWARD
    # ============================================================
    if results:
        best = results[0]
        print("\n" + "=" * 70)
        print(f"  BEST WF: vw={best['vw']} hvm={best['hvm']:.1f} hvr={best['hvr']:.2f} "
              f"tn={best['tn']} pyr={best['pyr']:.1f} ats={best['ats']:.1f}")
        print("=" * 70)

        best_sigs = sigs_cache[best['vw']]
        walk_forward(C, O, H, L, NS, ND, dates, syms, best_sigs,
                     pyramid_ratio=best['pyr'], pyramid_day=1,
                     top_n=best['tn'], min_confidence=2, hold_days=5,
                     atr_stop=best['ats'],
                     high_vol_mult=best['hvm'], high_vol_rank=best['hvr'],
                     mid_vol_rank=0.80, low_vol_rank=0.75)

    print(f"\n[V31] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
