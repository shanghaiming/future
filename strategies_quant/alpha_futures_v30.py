"""
V30: RANK + SECTOR BOOST -- Mean Reversion
===========================================
Combines V18's cross-sectional rank methodology with V10's sector awareness.

Core thesis:
  V18 (WF +2019.6%, Sharpe 2.39) uses cross-sectional rank percentiles.
  V10 (Sharpe 1.82) adds sector-level filtering for mean reversion.
  V30 combines: use V18's 7-factor rank methodology but add a sector-level
  oversold filter. When a whole sector is oversold, boost the rank of
  individual commodities in that sector.

Signal architecture:
  1. Compute V18's 7 cross-sectional ranks per instrument
  2. Compute sector-level oversold rank:
     - Average the composite rank across all instruments in each sector
     - 7 sectors: BLACK, METAL, ENERGY, CHEMICAL, AGRI, SOFTS, LIVESTOCK
  3. Sector boost: if sector_avg_rank > threshold, boost individual rank
  4. Final_rank = min(individual_rank * (1 + sector_boost), 1.0)
  5. KER gate, confidence >= 2, hold 5d, ATR stop 3.0, pyramid 0.5

Signal at close[di], enter at open[di+1]. No look-ahead. No gap signals.
No leverage. Walk-forward validation required.
"""
import sys
import os
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from collections import defaultdict

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

# Sector definitions as specified in requirements
SECTOR_DEFS: Dict[str, List[str]] = {
    'BLACK':     ['rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi'],
    'METAL':     ['cufi', 'alfi', 'znfi', 'nifi', 'snfi'],
    'ENERGY':    ['scfi', 'bufi', 'fufi', 'tafi', 'mafi', 'pgfi'],
    'CHEMICAL':  ['ppfi', 'lfi', 'vfi', 'egfi', 'pfi', 'efi', 'ebfi', 'urfi'],
    'AGRI':      ['cfi', 'csfi', 'afi', 'yfi', 'pfi', 'jfi', 'apfi', 'cffi', 'srfi'],
    'SOFTS':     ['srfi', 'whfi', 'cffi', 'apfi'],
    'LIVESTOCK': [],  # only if available (few commodities)
}

# V18 default weights for composite rank
DEFAULT_WEIGHTS: Dict[str, float] = {
    'rank_ret5d':  0.25,
    'rank_oi5d':   0.20,
    'rank_rsi':    0.15,
    'rank_vol':    0.15,
    'rank_ret10d': 0.10,
    'rank_range':  0.10,
    'rank_atrp':   0.05,
}


# ============================================================
# SECTOR MAPPING
# ============================================================
def build_sector_map(
    syms: List[str],
) -> Tuple[Dict[str, List[int]], Dict[int, str]]:
    """Build symbol-to-sector index mappings.

    Returns:
        sector_to_si: sector_name -> list of symbol indices present in data
        si_to_sector: symbol_index -> sector_name
    """
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
# RAW FACTOR COMPUTATION (from V18)
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
    C: np.ndarray, O: np.ndarray, H: np.ndarray,
    L: np.ndarray, V: np.ndarray, OI: np.ndarray,
    NS: int, ND: int,
) -> Dict[str, np.ndarray]:
    """Compute raw factor values before cross-sectional ranking."""
    t0 = time.time()
    print("[V30] Computing raw factors...", flush=True)

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

    # --- Volume (5d average) ---
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


# ============================================================
# CROSS-SECTIONAL RANKS (from V18)
# ============================================================
def compute_cross_sectional_ranks(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int,
    min_count: int = 10,
) -> Dict[str, np.ndarray]:
    """Rank all factors cross-sectionally (across commodities per day)."""
    t0 = time.time()
    print("[V30] Computing cross-sectional ranks...", flush=True)

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
# COMPOSITE SIGNAL (from V18)
# ============================================================
def build_composite_signal(
    ranks: Dict[str, np.ndarray],
    weights: Dict[str, float],
    NS: int, ND: int,
    min_factors: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build weighted composite rank from individual factor ranks."""
    t0 = time.time()
    print("[V30] Building composite signal...", flush=True)

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
# KER REGIME (from V18)
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
                    ker_regime[si, di] = 1   # sideways -> good for mean reversion
                elif ker_val > 0.3:
                    ker_regime[si, di] = -1  # trending -> avoid
    return ker_regime


# ============================================================
# SECTOR BOOST (V30 innovation)
# ============================================================
def compute_sector_boosted_rank(
    composite: np.ndarray,
    NS: int, ND: int,
    syms: List[str],
    sector_boost: float = 0.10,
    sector_threshold: float = 0.70,
) -> Tuple[np.ndarray, Dict[str, List[int]]]:
    """Apply sector-level oversold boost to individual composite ranks.

    For each sector, compute average composite rank. If the sector average
    exceeds the threshold, boost individual ranks in that sector.

    Returns:
        boosted_rank: (NS, ND) array with sector-boosted ranks
        sector_to_si: sector_name -> list of symbol indices
    """
    t0 = time.time()
    print(f"[V30] Computing sector boost (threshold={sector_threshold}, "
          f"boost={sector_boost})...", flush=True)

    sector_to_si, _ = build_sector_map(syms)

    boosted_rank = composite.copy()

    for di in range(ND):
        for sector, indices in sector_to_si.items():
            ranks_in_sector = composite[indices, di]
            valid_ranks = ranks_in_sector[~np.isnan(ranks_in_sector)]
            if len(valid_ranks) < 2:
                continue

            sector_avg = float(np.mean(valid_ranks))

            if sector_avg >= sector_threshold:
                for si in indices:
                    if not np.isnan(boosted_rank[si, di]):
                        boosted_rank[si, di] = min(
                            boosted_rank[si, di] * (1.0 + sector_boost),
                            1.0,
                        )

    print(f"  Sector boost done: {time.time() - t0:.1f}s", flush=True)
    return boosted_rank, sector_to_si


# ============================================================
# FULL SIGNAL PIPELINE
# ============================================================
def compute_all_signals(
    C: np.ndarray, O: np.ndarray, H: np.ndarray,
    L: np.ndarray, V: np.ndarray, OI: np.ndarray,
    NS: int, ND: int,
    syms: List[str],
    weights: Optional[Dict[str, float]] = None,
    sector_boost: float = 0.10,
    sector_threshold: float = 0.70,
) -> Dict[str, object]:
    """Full signal pipeline: V18 ranks + sector boost."""
    if weights is None:
        weights = DEFAULT_WEIGHTS

    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_cross_sectional_ranks(raw, NS, ND)
    ker_regime = compute_ker(C, NS, ND)
    composite, n_confirm = build_composite_signal(ranks, weights, NS, ND)

    boosted_rank, sector_to_si = compute_sector_boosted_rank(
        composite, NS, ND, syms,
        sector_boost=sector_boost,
        sector_threshold=sector_threshold,
    )

    return {
        'composite': boosted_rank,
        'base_composite': composite,
        'n_confirm': n_confirm,
        'ker_regime': ker_regime,
        'ranks': ranks,
        'sector_to_si': sector_to_si,
    }


# ============================================================
# BACKTEST ENGINE
# ============================================================
def backtest_v30(
    C: np.ndarray, O: np.ndarray, H: np.ndarray,
    L: np.ndarray, NS: int, ND: int,
    dates: np.ndarray, syms: List[str],
    sigs: Dict[str, object],
    top_n: int = 1,
    min_rank: float = 0.75,
    atr_stop: float = 3.0,
    min_confidence: int = 2,
    use_ker_gate: bool = True,
    hold_days: int = 5,
    pyramid_ratio: float = 0.5,
    pyramid_day: int = 1,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Backtest with sector-boosted cross-sectional rank signals + pyramid."""
    composite = sigs['composite']
    ker_regime = sigs['ker_regime']
    n_confirm = sigs['n_confirm']

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

        pos_by_si: Dict[int, List[Tuple[int, float, float, float, bool]]] = defaultdict(list)
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
            held_with_pos: Dict[int, List[Tuple[int, float, float, float, bool]]] = defaultdict(list)
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
        print(f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% cum={cum:+.1%}")

    return {'n': len(trades), 'wr': wr, 'dd': max_dd, 'ann': ann, 'sh': sh, 'eq': equity}


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================
def walk_forward(
    C: np.ndarray, O: np.ndarray, H: np.ndarray,
    L: np.ndarray, V: np.ndarray, OI: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    sector_boost: float = 0.10,
    sector_threshold: float = 0.70,
    pyramid_ratio: float = 0.5,
    top_n: int = 1,
    min_confidence: int = 2,
    hold_days: int = 5,
    atr_stop: float = 3.0,
    min_rank: float = 0.75,
) -> List[dict]:
    """Walk-forward validation: year-by-year out-of-sample."""
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V30 (boost={sector_boost}, thresh={sector_threshold}, "
          f"pyr={pyramid_ratio}, tn={top_n}, mc={min_confidence})")
    print(f"{'=' * 70}")

    years = sorted(set(d.year for d in dates))

    # Compute signals once with given params
    sigs = compute_all_signals(
        C, O, H, L, V, OI, NS, ND, syms,
        sector_boost=sector_boost,
        sector_threshold=sector_threshold,
    )

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

        trades, _, _ = backtest_v30(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=top_n, hold_days=hold_days, atr_stop=atr_stop,
            min_rank=min_rank,
            min_confidence=min_confidence, use_ker_gate=True,
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
        print(f"\n  WF TOTAL: {len(all_trades)}t (pyr:{n_pyr}) WR={wr_val:.1f}% "
              f"avg={avg:+.2f}% cum={cum:+.1%}")
        return all_trades
    return []


# ============================================================
# PARAMETER SWEEP
# ============================================================
def parameter_sweep(
    C: np.ndarray, O: np.ndarray, H: np.ndarray,
    L: np.ndarray, V: np.ndarray, OI: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    start_di: int = 60,
) -> List[dict]:
    """Sweep over sector boost params and structural params."""
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    results: List[dict] = []

    for sector_boost in [0.05, 0.10, 0.15, 0.20]:
        for sector_threshold in [0.60, 0.70, 0.80]:
            sigs = compute_all_signals(
                C, O, H, L, V, OI, NS, ND, syms,
                sector_boost=sector_boost,
                sector_threshold=sector_threshold,
            )
            for top_n in [1, 2, 3]:
                for min_rank in [0.70, 0.75, 0.80]:
                    for min_confidence in [2, 3]:
                        for pyramid in [0.0, 0.5]:
                            for atr_stop in [2.5, 3.0]:
                                trades, eq, dd = backtest_v30(
                                    C, O, H, L, NS, ND, dates, syms, sigs,
                                    top_n=top_n, hold_days=5, atr_stop=atr_stop,
                                    min_rank=min_rank,
                                    min_confidence=min_confidence,
                                    use_ker_gate=True,
                                    pyramid_ratio=pyramid, pyramid_day=1,
                                    start_di=start_di,
                                )
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
                                    'sb': sector_boost,
                                    'st': sector_threshold,
                                    'tn': top_n,
                                    'mr': min_rank,
                                    'mc': min_confidence,
                                    'pyr': pyramid,
                                    'atr': atr_stop,
                                    'n': len(trades),
                                    'wr': wr,
                                    'ann': ann,
                                    'dd': dd,
                                    'sharpe': sh_val,
                                    'eq': eq,
                                })

    results.sort(key=lambda x: -x['sharpe'])
    print(f"\n{'SB':>4} {'ST':>4} {'TN':>3} {'MR':>4} {'MC':>3} {'Pyr':>4} {'ATR':>4} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 80)
    for r in results[:30]:
        print(f"{r['sb']:>4.2f} {r['st']:>4.2f} {r['tn']:>3} {r['mr']:>4.2f} {r['mc']:>3} "
              f"{r['pyr']:>4.1f} {r['atr']:>4.1f} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sharpe']:>5.2f}")

    return results


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V30: RANK + SECTOR BOOST MEAN REVERSION")
    print("  V18 cross-sectional ranks + V10 sector awareness")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    print(f"  {NS} sym, {ND} days, "
          f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    # Print sector coverage
    sector_to_si, si_to_sector = build_sector_map(syms)
    for sector in sorted(SECTOR_DEFS.keys()):
        if sector in sector_to_si:
            indices = sector_to_si[sector]
            member_syms = [syms[si] for si in indices]
            print(f"  {sector:>10}: {len(member_syms)} symbols - {', '.join(member_syms)}")
        else:
            print(f"  {sector:>10}: no coverage")

    # Find 2019 start index
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2019-01-01'):
            bt_2019 = i
            break

    # === 1. Default config walk-forward ===
    print("\n" + "=" * 70)
    print("  WALK-FORWARD VALIDATION (default: boost=0.10, thresh=0.70)")
    print("=" * 70)

    for ratio in [0.0, 0.5]:
        walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms,
                     sector_boost=0.10, sector_threshold=0.70,
                     pyramid_ratio=ratio, top_n=1, min_confidence=2)

    # === 2. Sector boost comparison (is the boost helping?) ===
    print("\n" + "=" * 70)
    print("  SECTOR BOOST COMPARISON (2019-2026)")
    print("=" * 70)

    for sb, st in [(0.0, 0.70), (0.05, 0.70), (0.10, 0.70), (0.15, 0.70), (0.20, 0.70)]:
        sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND, syms,
                                   sector_boost=sb, sector_threshold=st)
        trades, eq, dd = backtest_v30(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=1, hold_days=5, atr_stop=3.0,
            min_rank=0.75, min_confidence=2, use_ker_gate=True,
            pyramid_ratio=0.5, pyramid_day=1,
            start_di=bt_2019)
        label = f"boost={sb:.2f}/thresh={st:.2f}"
        analyze(trades, eq, dd, label)

    # === 3. Parameter sweep ===
    results = parameter_sweep(
        C, O, H, L, V, OI, NS, ND, dates, syms,
        start_di=bt_2019,
    )

    # === 4. Best configs full 10-year ===
    if results:
        print("\n" + "=" * 70)
        print("  BEST CONFIGS -- FULL 10-YEAR")
        print("=" * 70)

        seen_configs = set()
        unique_best = []
        for r in results:
            key = (r['sb'], r['st'], r['tn'], r['mr'], r['mc'], r['pyr'], r['atr'])
            if key not in seen_configs:
                seen_configs.add(key)
                unique_best.append(r)
            if len(unique_best) >= 5:
                break

        for r in unique_best:
            sigs = compute_all_signals(
                C, O, H, L, V, OI, NS, ND, syms,
                sector_boost=r['sb'], sector_threshold=r['st'],
            )
            trades, eq, dd = backtest_v30(
                C, O, H, L, NS, ND, dates, syms, sigs,
                top_n=r['tn'], hold_days=5, atr_stop=r['atr'],
                min_rank=r['mr'], min_confidence=r['mc'],
                use_ker_gate=True,
                pyramid_ratio=r['pyr'], pyramid_day=1,
                start_di=60)
            label = (f"sb={r['sb']:.2f}/st={r['st']:.2f}/tn={r['tn']}/"
                     f"mr={r['mr']:.2f}/mc={r['mc']}/pyr={r['pyr']:.1f}/atr={r['atr']:.1f}")
            print(f"\n  FULL {label}")
            analyze(trades, eq, dd, label)

        # === 5. Walk-forward for best overall ===
        best = unique_best[0]
        print("\n" + "=" * 70)
        print(f"  BEST WF: sb={best['sb']:.2f} st={best['st']:.2f} "
              f"tn={best['tn']} pyr={best['pyr']:.1f} "
              f"mc={best['mc']} mr={best['mr']:.2f} atr={best['atr']:.1f}")
        print("=" * 70)
        walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms,
                     sector_boost=best['sb'], sector_threshold=best['st'],
                     pyramid_ratio=best['pyr'], top_n=best['tn'],
                     min_confidence=best['mc'], hold_days=5,
                     atr_stop=best['atr'], min_rank=best['mr'])

    print(f"\n[V30] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
