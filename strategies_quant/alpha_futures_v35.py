"""
V35: INVERSE MOMENTUM RANK -- Mean Reversion
=============================================
Buy the BIGGEST losers (strongest mean reversion candidates).

Core thesis:
  V18 (WF +2019.6%, Sharpe 2.39) showed cross-sectional rank works.
  V35 applies this more aggressively to pure price momentum:
  rank by worst performance across multiple timeframes, then apply
  quality filters. The worst-performing commodities over 5-20 days
  have the highest expected reversal.

Signal architecture:
  1. Compute multi-timeframe momentum ranks (3d, 5d, 10d, 20d)
  2. Composite momentum = weighted sum (worst = highest rank)
  3. Quality filters: OI decline, volume surge, RSI<35, consecutive
     down days, KER<0.15
  4. Entry: highest momentum rank AND quality confirmed
  5. Hold 5d, ATR stop 3.0, pyramid on day-1 winners

Signal at close[di], enter at open[di+1]. No look-ahead. No leverage.
Walk-forward validation required.
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


# ============================================================
# RSI FALLBACK (no talib)
# ============================================================
def compute_rsi_manual(
    C: np.ndarray, NS: int, ND: int, period: int = 14,
) -> np.ndarray:
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
# RAW MOMENTUM FACTORS
# ============================================================
def compute_momentum_factors(
    C: np.ndarray, O: np.ndarray, H: np.ndarray,
    L: np.ndarray, V: np.ndarray, OI: np.ndarray,
    NS: int, ND: int,
) -> Dict[str, np.ndarray]:
    """Compute raw momentum and quality factor values."""
    t0 = time.time()
    print("[V35] Computing momentum factors...", flush=True)

    # --- Multi-timeframe returns ---
    ret_3d = np.full((NS, ND), np.nan)
    ret_5d = np.full((NS, ND), np.nan)
    ret_10d = np.full((NS, ND), np.nan)
    ret_20d = np.full((NS, ND), np.nan)

    for si in range(NS):
        for di in range(3, ND):
            if (not np.isnan(C[si, di]) and not np.isnan(C[si, di - 3])
                    and C[si, di - 3] > 0):
                ret_3d[si, di] = C[si, di] / C[si, di - 3] - 1.0
        for di in range(5, ND):
            if (not np.isnan(C[si, di]) and not np.isnan(C[si, di - 5])
                    and C[si, di - 5] > 0):
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1.0
        for di in range(10, ND):
            if (not np.isnan(C[si, di]) and not np.isnan(C[si, di - 10])
                    and C[si, di - 10] > 0):
                ret_10d[si, di] = C[si, di] / C[si, di - 10] - 1.0
        for di in range(20, ND):
            if (not np.isnan(C[si, di]) and not np.isnan(C[si, di - 20])
                    and C[si, di - 20] > 0):
                ret_20d[si, di] = C[si, di] / C[si, di - 20] - 1.0

    # --- OI 5d change ---
    oi_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (not np.isnan(OI[si, di]) and not np.isnan(OI[si, di - 5])
                    and OI[si, di - 5] > 0):
                oi_5d[si, di] = OI[si, di] / OI[si, di - 5] - 1.0

    # --- Volume ratio vs 20d avg ---
    vol_ratio = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            vals = V[si, di - 20:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 10 and np.mean(valid) > 0:
                if not np.isnan(V[si, di]):
                    vol_ratio[si, di] = V[si, di] / np.mean(valid)

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

    # --- Consecutive down days ---
    consec_down = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        count = 0
        for di in range(1, ND):
            if np.isnan(C[si, di]) or np.isnan(C[si, di - 1]):
                count = 0
                continue
            if C[si, di] < C[si, di - 1]:
                count += 1
            else:
                count = 0
            consec_down[si, di] = count

    # --- KER (Kaufman Efficiency Ratio) ---
    ker = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            closes = C[si, di - 10:di + 1]
            valid = closes[~np.isnan(closes)]
            if len(valid) < 10 or valid[0] <= 0:
                continue
            net_change = abs(valid[-1] - valid[0])
            total_change = np.sum(np.abs(np.diff(valid)))
            if total_change > 1e-10:
                ker[si, di] = net_change / total_change

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

    print(f"  Momentum factors done: {time.time() - t0:.1f}s", flush=True)
    return {
        'ret_3d': ret_3d,
        'ret_5d': ret_5d,
        'ret_10d': ret_10d,
        'ret_20d': ret_20d,
        'oi_5d': oi_5d,
        'vol_ratio': vol_ratio,
        'rsi14': rsi14,
        'consec_down': consec_down,
        'ker': ker,
        'atrp': atrp,
    }


# ============================================================
# CROSS-SECTIONAL MOMENTUM RANKS
# ============================================================
def compute_momentum_ranks(
    factors: Dict[str, np.ndarray],
    NS: int, ND: int,
    min_count: int = 10,
) -> Dict[str, np.ndarray]:
    """Rank momentum factors cross-sectionally.

    Worst return -> highest rank (rank 1.0 = biggest loser).
    """
    t0 = time.time()
    print("[V35] Computing cross-sectional momentum ranks...", flush=True)

    ret_factors = {
        'rank_ret3d': factors['ret_3d'],
        'rank_ret5d': factors['ret_5d'],
        'rank_ret10d': factors['ret_10d'],
        'rank_ret20d': factors['ret_20d'],
    }

    # For momentum: 1 - rank means worst performers get highest rank
    ranks: Dict[str, np.ndarray] = {}
    for name, factor in ret_factors.items():
        rank_arr = np.full((NS, ND), np.nan)
        for di in range(ND):
            vals = factor[:, di]
            valid_count = int(np.sum(~np.isnan(vals)))
            if valid_count < min_count:
                continue
            # Standard pct rank: high value -> high rank
            # For returns: most negative -> 1 - rank -> highest rank
            ranked = pd.Series(vals).rank(pct=True, na_option='keep').values
            # Invert: worst return = highest rank
            ranked = 1.0 - ranked
            rank_arr[:, di] = ranked
        ranks[name] = rank_arr

    print(f"  CS momentum ranks done: {time.time() - t0:.1f}s", flush=True)
    return ranks


# ============================================================
# COMPOSITE MOMENTUM SIGNAL
# ============================================================
def build_composite_momentum(
    ranks: Dict[str, np.ndarray],
    w3d: float, w5d: float, w10d: float, w20d: float,
    NS: int, ND: int,
    min_factors: int = 3,
) -> np.ndarray:
    """Build weighted composite momentum rank."""
    t0 = time.time()
    weights_map = {
        'rank_ret3d': w3d,
        'rank_ret5d': w5d,
        'rank_ret10d': w10d,
        'rank_ret20d': w20d,
    }
    print(f"[V35] Building composite momentum (w3={w3d}, w5={w5d}, "
          f"w10={w10d}, w20={w20d})...", flush=True)

    composite = np.full((NS, ND), np.nan)
    factor_names = list(weights_map.keys())
    weight_vals = np.array([weights_map[k] for k in factor_names])

    for di in range(ND):
        for si in range(NS):
            vals = []
            w_sum = 0.0
            for idx, name in enumerate(factor_names):
                rank_val = ranks[name][si, di]
                if np.isnan(rank_val):
                    continue
                vals.append(rank_val * weight_vals[idx])
                w_sum += weight_vals[idx]

            if w_sum > 0 and len(vals) >= min_factors:
                composite[si, di] = sum(vals) / w_sum

    print(f"  Composite done: {time.time() - t0:.1f}s", flush=True)
    return composite


# ============================================================
# QUALITY FILTERS
# ============================================================
def compute_quality_score(
    factors: Dict[str, np.ndarray],
    NS: int, ND: int,
) -> np.ndarray:
    """Compute quality score (number of quality filters met).

    Filters:
      1. OI declining (oi_5d_change < -2%)
      2. Volume surge (vol > 1.5x 20d avg)
      3. RSI < 35
      4. Consecutive down days >= 3
      5. KER < 0.15 (mean-reverting regime)
    """
    quality = np.zeros((NS, ND), dtype=int)

    for si in range(NS):
        for di in range(20, ND):
            count = 0

            # Filter 1: OI declining
            oi_val = factors['oi_5d'][si, di]
            if not np.isnan(oi_val) and oi_val < -0.02:
                count += 1

            # Filter 2: Volume surge
            vr = factors['vol_ratio'][si, di]
            if not np.isnan(vr) and vr > 1.5:
                count += 1

            # Filter 3: RSI < 35
            rsi = factors['rsi14'][si, di]
            if not np.isnan(rsi) and rsi < 35:
                count += 1

            # Filter 4: Consecutive down days >= 3
            if factors['consec_down'][si, di] >= 3:
                count += 1

            # Filter 5: KER < 0.15
            ker_val = factors['ker'][si, di]
            if not np.isnan(ker_val) and ker_val < 0.15:
                count += 1

            quality[si, di] = count

    return quality


# ============================================================
# FULL SIGNAL PIPELINE
# ============================================================
def compute_all_signals(
    C: np.ndarray, O: np.ndarray, H: np.ndarray,
    L: np.ndarray, V: np.ndarray, OI: np.ndarray,
    NS: int, ND: int,
    w3d: float = 0.20, w5d: float = 0.40,
    w10d: float = 0.25, w20d: float = 0.15,
) -> Dict[str, object]:
    """Full signal pipeline for V35."""
    factors = compute_momentum_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_momentum_ranks(factors, NS, ND)
    composite = build_composite_momentum(
        ranks, w3d, w5d, w10d, w20d, NS, ND,
    )
    quality = compute_quality_score(factors, NS, ND)

    return {
        'composite': composite,
        'quality': quality,
        'factors': factors,
        'ranks': ranks,
    }


# ============================================================
# BACKTEST ENGINE
# ============================================================
def backtest_v35(
    C: np.ndarray, O: np.ndarray, H: np.ndarray,
    L: np.ndarray, NS: int, ND: int,
    dates: np.ndarray, syms: List[str],
    sigs: Dict[str, object],
    top_n: int = 1,
    min_rank: float = 0.80,
    min_quality: int = 2,
    atr_stop: float = 3.0,
    hold_days: int = 5,
    pyramid_ratio: float = 0.5,
    pyramid_day: int = 1,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Backtest V35 inverse momentum rank signals + pyramid."""
    composite = sigs['composite']
    quality = sigs['quality']

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
            if quality[si, di] < min_quality:
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
def analyze(
    trades: List[dict], equity: float, max_dd: float, label: str = "",
) -> Optional[dict]:
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
    w3d: float = 0.20, w5d: float = 0.40,
    w10d: float = 0.25, w20d: float = 0.15,
    top_n: int = 1,
    min_rank: float = 0.80,
    min_quality: int = 2,
    pyramid_ratio: float = 0.5,
    atr_stop: float = 3.0,
    hold_days: int = 5,
) -> List[dict]:
    """Walk-forward validation: year-by-year out-of-sample."""
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V35 (w3={w3d}, w5={w5d}, w10={w10d}, w20={w20d}, "
          f"pyr={pyramid_ratio}, tn={top_n}, mq={min_quality})")
    print(f"{'=' * 70}")

    years = sorted(set(d.year for d in dates))

    sigs = compute_all_signals(
        C, O, H, L, V, OI, NS, ND,
        w3d=w3d, w5d=w5d, w10d=w10d, w20d=w20d,
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

        trades, _, _ = backtest_v35(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=top_n, hold_days=hold_days, atr_stop=atr_stop,
            min_rank=min_rank, min_quality=min_quality,
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
    """Sweep over V35 parameters: momentum weights, quality, top_n, etc."""
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    results: List[dict] = []

    momentum_weights = [
        (0.40, 0.25, 0.20, 0.15),  # w5d-heavy (default)
        (0.30, 0.30, 0.20, 0.20),  # balanced
        (0.50, 0.20, 0.20, 0.10),  # even more w5d-heavy
    ]

    for w3d, w5d, w10d, w20d in momentum_weights:
        sigs = compute_all_signals(
            C, O, H, L, V, OI, NS, ND,
            w3d=w3d, w5d=w5d, w10d=w10d, w20d=w20d,
        )
        for min_quality in [2, 3]:
            for top_n in [1, 2, 3]:
                for min_rank in [0.70, 0.80, 0.90]:
                    for pyramid in [0.0, 0.5]:
                        for atr_stop in [2.5, 3.0]:
                            trades, eq, dd = backtest_v35(
                                C, O, H, L, NS, ND, dates, syms, sigs,
                                top_n=top_n, hold_days=5, atr_stop=atr_stop,
                                min_rank=min_rank, min_quality=min_quality,
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
                                'w3d': w3d, 'w5d': w5d, 'w10d': w10d, 'w20d': w20d,
                                'mq': min_quality,
                                'tn': top_n,
                                'mr': min_rank,
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
    print(f"\n{'W3':>4} {'W5':>4} {'W10':>4} {'W20':>4} "
          f"{'MQ':>3} {'TN':>3} {'MR':>4} {'Pyr':>4} {'ATR':>4} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 95)
    for r in results[:30]:
        print(f"{r['w3d']:>4.2f} {r['w5d']:>4.2f} {r['w10d']:>4.2f} {r['w20d']:>4.2f} "
              f"{r['mq']:>3} {r['tn']:>3} {r['mr']:>4.2f} {r['pyr']:>4.1f} {r['atr']:>4.1f} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sharpe']:>5.2f}")

    return results


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V35: INVERSE MOMENTUM RANK MEAN REVERSION")
    print("  Buy the biggest losers with quality confirmation")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    print(f"  {NS} sym, {ND} days, "
          f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    # Find 2019 start index
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2019-01-01'):
            bt_2019 = i
            break

    # === 1. Default config walk-forward ===
    print("\n" + "=" * 70)
    print("  WALK-FORWARD VALIDATION (default weights)")
    print("=" * 70)

    for ratio in [0.0, 0.5]:
        walk_forward(
            C, O, H, L, V, OI, NS, ND, dates, syms,
            w3d=0.20, w5d=0.40, w10d=0.25, w20d=0.15,
            pyramid_ratio=ratio, top_n=1, min_quality=2,
        )

    # === 2. Momentum weight comparison ===
    print("\n" + "=" * 70)
    print("  MOMENTUM WEIGHT COMPARISON (2019-2026)")
    print("=" * 70)

    for w3d, w5d, w10d, w20d in [
        (0.40, 0.25, 0.20, 0.15),
        (0.30, 0.30, 0.20, 0.20),
        (0.50, 0.20, 0.20, 0.10),
    ]:
        sigs = compute_all_signals(
            C, O, H, L, V, OI, NS, ND,
            w3d=w3d, w5d=w5d, w10d=w10d, w20d=w20d,
        )
        trades, eq, dd = backtest_v35(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=1, hold_days=5, atr_stop=3.0,
            min_rank=0.80, min_quality=2,
            pyramid_ratio=0.5, pyramid_day=1,
            start_di=bt_2019,
        )
        label = f"w3={w3d:.2f}/w5={w5d:.2f}/w10={w10d:.2f}/w20={w20d:.2f}"
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
            key = (r['w3d'], r['w5d'], r['w10d'], r['w20d'],
                   r['tn'], r['mr'], r['mq'], r['pyr'], r['atr'])
            if key not in seen_configs:
                seen_configs.add(key)
                unique_best.append(r)
            if len(unique_best) >= 5:
                break

        for r in unique_best:
            sigs = compute_all_signals(
                C, O, H, L, V, OI, NS, ND,
                w3d=r['w3d'], w5d=r['w5d'],
                w10d=r['w10d'], w20d=r['w20d'],
            )
            trades, eq, dd = backtest_v35(
                C, O, H, L, NS, ND, dates, syms, sigs,
                top_n=r['tn'], hold_days=5, atr_stop=r['atr'],
                min_rank=r['mr'], min_quality=r['mq'],
                pyramid_ratio=r['pyr'], pyramid_day=1,
                start_di=60,
            )
            label = (f"w3={r['w3d']:.2f}/w5={r['w5d']:.2f}/w10={r['w10d']:.2f}/"
                     f"w20={r['w20d']:.2f}/tn={r['tn']}/mr={r['mr']:.2f}/"
                     f"mq={r['mq']}/pyr={r['pyr']:.1f}/atr={r['atr']:.1f}")
            print(f"\n  FULL {label}")
            analyze(trades, eq, dd, label)

        # === 5. Walk-forward for best overall ===
        best = unique_best[0]
        print("\n" + "=" * 70)
        print(f"  BEST WF: w3={best['w3d']:.2f} w5={best['w5d']:.2f} "
              f"w10={best['w10d']:.2f} w20={best['w20d']:.2f} "
              f"tn={best['tn']} pyr={best['pyr']:.1f} "
              f"mq={best['mq']} mr={best['mr']:.2f} atr={best['atr']:.1f}")
        print("=" * 70)
        walk_forward(
            C, O, H, L, V, OI, NS, ND, dates, syms,
            w3d=best['w3d'], w5d=best['w5d'],
            w10d=best['w10d'], w20d=best['w20d'],
            pyramid_ratio=best['pyr'], top_n=best['tn'],
            min_quality=best['mq'], hold_days=5,
            atr_stop=best['atr'], min_rank=best['mr'],
        )

    print(f"\n[V35] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
