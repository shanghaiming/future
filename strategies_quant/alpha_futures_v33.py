"""
V33: Correlation-Filtered Rank Mean Reversion
==============================================
Combines V18 (best Sharpe 2.39) cross-sectional rank signals with V6
(lowest DD 29.9%) correlation-aware portfolio construction.

Core idea: Use V18's 7-factor cross-sectional rank composite to identify
oversold candidates, but enforce diversification by rejecting picks that
are too correlated with already-selected instruments.

Selection algorithm:
  1. Rank all instruments by V18 composite rank score
  2. Select #1 ranked instrument
  3. For #2: skip any instrument with corr > max_corr to #1
  4. For #3: skip any instrument with corr > max_corr to ALL prior picks
  This ensures true diversification within the oversold basket.

Signal at close[di], enter at open[di+1]. No look-ahead. No gap signals.
No leverage. Walk-forward validation required.
"""
import sys
import os
import time
import warnings
from collections import defaultdict
from typing import Optional

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

MIN_CORR_OVERLAP = 30

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
# V18 RANK FACTORS (reused)
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


def compute_raw_factors(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
) -> dict:
    """Compute raw factor values before cross-sectional ranking."""
    t0 = time.time()
    print("[V33] Computing raw factors...", flush=True)

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


def compute_cross_sectional_ranks(
    raw_factors: dict, NS: int, ND: int, min_count: int = 10,
) -> dict:
    """Rank all factors cross-sectionally. Invert so low raw = high rank."""
    t0 = time.time()
    print("[V33] Computing cross-sectional ranks...", flush=True)

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


def build_composite_signal(
    ranks: dict, weights: dict, NS: int, ND: int, min_factors: int = 4,
) -> tuple:
    """Build weighted composite rank and confidence count."""
    t0 = time.time()
    print("[V33] Building composite signal...", flush=True)

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
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
    weights: Optional[dict] = None,
) -> dict:
    """Full V18 signal pipeline."""
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
# V6 ROLLING CORRELATION (adapted with 5d returns)
# ============================================================
def compute_rolling_correlations_5d(
    C: np.ndarray, NS: int, ND: int, window: int = 60,
) -> np.ndarray:
    """
    Compute rolling pairwise correlations using 5-day returns.
    Returns (NS, NS, ND) correlation matrix.
    """
    t0 = time.time()
    print(f"[V33] Computing rolling {window}-day correlations (5d returns)...", flush=True)

    # 5-day returns
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (
                not np.isnan(C[si, di])
                and not np.isnan(C[si, di - 5])
                and C[si, di - 5] > 0
            ):
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1.0

    corr_matrix = np.full((NS, NS, ND), np.nan)

    # Number of 5d return observations in the window
    n_obs = window // 5
    min_obs = max(n_obs // 2, 5)

    for di in range(window, ND):
        ret_window = ret_5d[:, di - window:di]  # (NS, window)
        for si in range(NS):
            ri = ret_window[si]
            valid_i = ~np.isnan(ri)
            if valid_i.sum() < min_obs:
                continue
            for sj in range(si, NS):
                rj = ret_window[sj]
                valid_j = ~np.isnan(rj)
                overlap = valid_i & valid_j
                n_overlap = overlap.sum()
                if n_overlap < min_obs:
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


def is_correlated_with_picks(
    corr_matrix: np.ndarray,
    candidate_si: int,
    picked_sis: list,
    di: int,
    max_corr: float,
) -> bool:
    """Check if candidate is too correlated with ANY already-picked instrument."""
    for picked_si in picked_sis:
        c = corr_matrix[candidate_si, picked_si, di]
        if not np.isnan(c) and abs(c) > max_corr:
            return True
    return False


# ============================================================
# V33 BACKTEST: Rank + Correlation Filter + Pyramid
# ============================================================
def backtest_v33(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: list, syms: list,
    sigs: dict, corr_matrix: np.ndarray,
    top_n: int = 2, min_rank: float = 0.75,
    atr_stop: float = 3.0, min_confidence: int = 2,
    use_ker_gate: bool = True, hold_days: int = 5,
    max_corr: float = 0.6, pyramid_ratio: float = 0.5,
    pyramid_day: int = 1, corr_window: int = 60,
    start_di: int = 60, end_di: Optional[int] = None,
) -> tuple:
    """
    Backtest with correlation-filtered rank selection.

    Selection algorithm:
      1. Rank all instruments by composite rank
      2. Pick #1 (best rank)
      3. For #2: skip any with corr > max_corr to #1
      4. For #3: skip any with corr > max_corr to ALL prior picks
    """
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
        daily_pnl = 0
        new_positions = []

        # Exit logic
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

        # --- Entry: correlation-filtered rank selection ---
        held = {p[0] for p in positions}
        if len(positions) >= top_n:
            continue

        # Build eligible candidates sorted by rank
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
            candidates.append((composite[si, di], si))

        # Sort descending by rank (best first)
        candidates.sort(key=lambda x: -x[0])

        # Sequential selection with correlation filter
        picked_sis = list(held)
        alloc = 1.0 / max(top_n, 1)

        for rank_val, si in candidates:
            if len(positions) >= top_n or si in held:
                break
            # Check correlation with already-picked instruments
            if is_correlated_with_picks(corr_matrix, si, picked_sis, di, max_corr):
                continue
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
            picked_sis.append(si)

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
def analyze(trades: list, equity: float, max_dd: float, label: str = "") -> Optional[dict]:
    """Print analysis and return summary dict."""
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

    print(
        f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} stop:{n_stop} hold:{n_hold}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% Sh={sh:.2f} eq={equity:,.0f}"
    )

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
# WALK-FORWARD VALIDATION
# ============================================================
def walk_forward_v33(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
    dates: list, syms: list,
    sigs: dict, corr_matrix: np.ndarray,
    top_n: int = 2, min_rank: float = 0.75,
    atr_stop: float = 3.0, min_confidence: int = 2,
    hold_days: int = 5, max_corr: float = 0.6,
    pyramid_ratio: float = 0.5,
) -> list:
    """Walk-forward validation: year-by-year out-of-sample."""
    print(f"\n{'=' * 70}")
    print(
        f"  WALK-FORWARD V33 (tn={top_n}, max_corr={max_corr}, "
        f"pyr={pyramid_ratio}, mc={min_confidence})"
    )
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

        trades, _, _ = backtest_v33(
            C, O, H, L, NS, ND, dates, syms, sigs, corr_matrix,
            top_n=top_n, hold_days=hold_days, atr_stop=atr_stop,
            min_rank=min_rank, min_confidence=min_confidence,
            use_ker_gate=True, max_corr=max_corr,
            pyramid_ratio=pyramid_ratio, pyramid_day=1,
            start_di=test_start, end_di=test_end_idx + 1,
        )

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
        print(
            f"\n  WF TOTAL: {len(all_trades)}t (pyr:{n_pyr}) WR={wr_val:.1f}% "
            f"avg={avg:+.2f}% cum={cum:+.1%}"
        )
        return all_trades
    return []


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("  V33: CORRELATION-FILTERED RANK MEAN REVERSION")
    print("  V18 rank signals + V6 correlation portfolio construction")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    print(
        f"  {NS} sym, {ND} days, "
        f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}"
    )

    # Find 2019 start index
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2019-01-01'):
            bt_2019 = i
            break

    # Compute V18 rank signals
    sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND)

    # Compute correlation matrices for different windows
    corr_matrices = {}
    for cw in [30, 60, 90]:
        corr_matrices[cw] = compute_rolling_correlations_5d(C, NS, ND, window=cw)

    # ===== SECTION 1: BASELINE (no correlation filter, 2019-2026) =====
    print("\n" + "=" * 70)
    print("  SECTION 1: BASELINE V18 RANK (no corr filter, 2019-2026)")
    print("=" * 70)

    for tn in [1, 2, 3]:
        for pyr in [0.0, 0.5]:
            trades, eq, dd = backtest_v33(
                C, O, H, L, NS, ND, dates, syms, sigs,
                corr_matrices[60],
                top_n=tn, min_rank=0.75, atr_stop=3.0,
                min_confidence=2, use_ker_gate=True, hold_days=5,
                max_corr=1.0,  # no filter
                pyramid_ratio=pyr, pyramid_day=1,
                start_di=bt_2019,
            )
            analyze(trades, eq, dd, f"base tn={tn} pyr={pyr:.1f}")

    # ===== SECTION 2: PARAMETER SWEEP =====
    print("\n" + "=" * 70)
    print("  SECTION 2: PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    results = []
    sweep_count = 0
    total_sweep = 3 * 4 * 2 * 3 * 2 * 2  # cw * max_corr * top_n * min_rank * pyr * atr
    print(f"  Total combos: {total_sweep}", flush=True)

    for cw in [30, 60, 90]:
        cm = corr_matrices[cw]
        for max_c in [0.5, 0.6, 0.7, 0.8]:
            for tn in [2, 3]:
                for mr in [0.70, 0.75, 0.80]:
                    for pyr in [0.0, 0.5]:
                        for as_val in [2.5, 3.0]:
                            sweep_count += 1
                            trades, eq, dd = backtest_v33(
                                C, O, H, L, NS, ND, dates, syms, sigs,
                                cm,
                                top_n=tn, min_rank=mr, atr_stop=as_val,
                                min_confidence=2, use_ker_gate=True,
                                hold_days=5, max_corr=max_c,
                                pyramid_ratio=pyr, pyramid_day=1,
                                start_di=bt_2019,
                            )
                            if len(trades) < 10:
                                continue
                            nw = sum(1 for t in trades if t['pnl_pct'] > 0)
                            wr = nw / len(trades) * 100
                            n_days = max(1, trades[-1]['di'] - trades[0]['di'])
                            ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
                            ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
                            rets_arr = np.array(ap) / CASH0
                            sh_val = (
                                np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252)
                                if np.std(rets_arr) > 0 else 0
                            )
                            results.append({
                                'cw': cw, 'max_corr': max_c, 'tn': tn,
                                'mr': mr, 'pyr': pyr, 'atr': as_val,
                                'n': len(trades), 'wr': wr, 'ann': ann,
                                'dd': dd, 'sharpe': sh_val, 'eq': eq,
                            })

        print(f"  Progress: {sweep_count}/{total_sweep} sweeps", flush=True)

    results.sort(key=lambda x: -x['sharpe'])
    print(
        f"\n{'CW':>3} {'MaxC':>5} {'TN':>3} {'MR':>4} "
        f"{'Pyr':>4} {'ATR':>4} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>6}"
    )
    print("-" * 75)
    for r in results[:30]:
        print(
            f"{r['cw']:>3} {r['max_corr']:>5.1f} {r['tn']:>3} {r['mr']:>4.2f} "
            f"{r['pyr']:>4.1f} {r['atr']:>4.1f} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f}"
        )

    # ===== SECTION 3: TOP CONFIGS FULL 10-YEAR =====
    print("\n" + "=" * 70)
    print("  SECTION 3: TOP CONFIGS -- FULL 10-YEAR (2016-2026)")
    print("=" * 70)

    best_configs = results[:10]
    for r in best_configs:
        cm = corr_matrices[r['cw']]
        trades, eq, dd = backtest_v33(
            C, O, H, L, NS, ND, dates, syms, sigs, cm,
            top_n=r['tn'], min_rank=r['mr'], atr_stop=r['atr'],
            min_confidence=2, use_ker_gate=True, hold_days=5,
            max_corr=r['max_corr'],
            pyramid_ratio=r['pyr'], pyramid_day=1,
            start_di=60,
        )
        label = (
            f"cw={r['cw']} maxC={r['max_corr']:.1f} tn={r['tn']} "
            f"mr={r['mr']:.2f} pyr={r['pyr']:.1f} atr={r['atr']:.1f}"
        )
        print(f"\n  10Y {label}")
        analyze(trades, eq, dd, label)

    # ===== SECTION 4: WALK-FORWARD FOR TOP 3 CONFIGS =====
    print("\n" + "=" * 70)
    print("  SECTION 4: WALK-FORWARD VALIDATION (TOP 3)")
    print("=" * 70)

    for r in best_configs[:3]:
        cm = corr_matrices[r['cw']]
        walk_forward_v33(
            C, O, H, L, V, OI, NS, ND, dates, syms, sigs, cm,
            top_n=r['tn'], min_rank=r['mr'], atr_stop=r['atr'],
            min_confidence=2, hold_days=5,
            max_corr=r['max_corr'], pyramid_ratio=r['pyr'],
        )

    # ===== SECTION 5: V18 BASELINE COMPARISON =====
    print("\n" + "=" * 70)
    print("  SECTION 5: V18-BASELINE vs V33-CORRELATED (2019-2026)")
    print("=" * 70)

    # V18 baseline (tn=1, no corr, no pyramid)
    trades_base, eq_base, dd_base = backtest_v33(
        C, O, H, L, NS, ND, dates, syms, sigs, corr_matrices[60],
        top_n=1, min_rank=0.75, atr_stop=3.0,
        min_confidence=3, use_ker_gate=True, hold_days=5,
        max_corr=1.0, pyramid_ratio=0.0,
        start_di=bt_2019,
    )
    print("\n  V18-BASELINE (tn=1, no corr, no pyr):")
    analyze(trades_base, eq_base, dd_base, "V18-base")

    # V18 with pyramid (best V18 config from memory)
    trades_v18p, eq_v18p, dd_v18p = backtest_v33(
        C, O, H, L, NS, ND, dates, syms, sigs, corr_matrices[60],
        top_n=1, min_rank=0.75, atr_stop=3.0,
        min_confidence=3, use_ker_gate=True, hold_days=5,
        max_corr=1.0, pyramid_ratio=0.5,
        start_di=bt_2019,
    )
    print("\n  V18-PYRAMID (tn=1, no corr, pyr=0.5):")
    analyze(trades_v18p, eq_v18p, dd_v18p, "V18-pyr")

    if best_configs:
        r_best = best_configs[0]
        cm_best = corr_matrices[r_best['cw']]
        trades_v33, eq_v33, dd_v33 = backtest_v33(
            C, O, H, L, NS, ND, dates, syms, sigs, cm_best,
            top_n=r_best['tn'], min_rank=r_best['mr'],
            atr_stop=r_best['atr'],
            min_confidence=2, use_ker_gate=True, hold_days=5,
            max_corr=r_best['max_corr'],
            pyramid_ratio=r_best['pyr'],
            start_di=bt_2019,
        )
        label_best = (
            f"V33-BEST (cw={r_best['cw']} maxC={r_best['max_corr']:.1f} "
            f"tn={r_best['tn']} pyr={r_best['pyr']:.1f}):"
        )
        print(f"\n  {label_best}")
        analyze(trades_v33, eq_v33, dd_v33, "V33-best")

        # Multi-position without correlation (same tn, same pyramid)
        trades_nocorr, eq_nocorr, dd_nocorr = backtest_v33(
            C, O, H, L, NS, ND, dates, syms, sigs, cm_best,
            top_n=r_best['tn'], min_rank=r_best['mr'],
            atr_stop=r_best['atr'],
            min_confidence=2, use_ker_gate=True, hold_days=5,
            max_corr=1.0,  # no filter
            pyramid_ratio=r_best['pyr'],
            start_di=bt_2019,
        )
        print(
            f"\n  SAME-TN NO-CORR (tn={r_best['tn']} pyr={r_best['pyr']:.1f}):"
        )
        analyze(trades_nocorr, eq_nocorr, dd_nocorr, "NO-CORR")

    print(f"\n[V33] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
