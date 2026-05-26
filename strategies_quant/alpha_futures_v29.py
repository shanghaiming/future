"""
V29: Extreme Rank Oversold Mean Reversion
==========================================
Core thesis: V18 showed cross-sectional rank works with min_rank=0.80.
V29 pushes this further: only trade when rank > 0.90 (top 10% most
oversold). This should produce fewer but higher-quality trades.

Signal architecture:
  1. Same V18 cross-sectional rank methodology with 7 factors
     - rank_ret5d:  0.25  (low rank = oversold)
     - rank_oi5d:   0.20  (declining OI + price drop = capitulation)
     - rank_rsi:    0.15  (low RSI rank = oversold)
     - rank_vol:    0.15  (high vol rank = attention)
     - rank_ret10d: 0.10
     - rank_range:  0.10  (expansion = capitulation)
     - rank_atrp:   0.05  (high ATR% = opportunity)
  2. But require HIGHER minimum rank thresholds:
     - Test min_rank: 0.80, 0.85, 0.90, 0.95
  3. When extreme oversold (>0.90), allow up to top_n=3 positions
  4. When moderate oversold (>0.80), only top_n=1
  5. KER gate, hold 5d, ATR stop 3.0
  6. Pyramid on day-1 winners at higher ratio (0.7 for extreme, 0.5 moderate)

Parameter sweep:
  - min_rank: 0.80, 0.85, 0.90, 0.95
  - top_n: 1, 2, 3
  - pyramid: 0.0, 0.5, 0.7
  - atr_stop: 2.5, 3.0
  - min_confidence: 2, 3

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
from typing import Dict, List, Optional, Tuple

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

# Default weights for composite rank (same as V18)
DEFAULT_WEIGHTS = {
    "rank_ret5d": 0.25,
    "rank_oi5d": 0.20,
    "rank_rsi": 0.15,
    "rank_vol": 0.15,
    "rank_ret10d": 0.10,
    "rank_range": 0.10,
    "rank_atrp": 0.05,
}

# Extreme oversold threshold
EXTREME_THRESHOLD = 0.90


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
                        valid_l.append(
                            losses[j] if not np.isnan(losses[j]) else 0.0
                        )
                if len(valid_g) >= period:
                    avg_gain = np.mean(valid_g)
                    avg_loss = np.mean(valid_l)
                    if avg_loss == 0:
                        rsi[si, di + period - 1] = 100.0
                    else:
                        rs = avg_gain / avg_loss
                        rsi[si, di + period - 1] = (
                            100.0 - 100.0 / (1.0 + rs)
                        )
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
) -> Dict[str, np.ndarray]:
    """Compute raw factor values before cross-sectional ranking."""
    t0 = time.time()
    print("[V29] Computing raw factors...", flush=True)

    # --- 5d return ---
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (not np.isnan(C[si, di])
                    and not np.isnan(C[si, di - 5])
                    and C[si, di - 5] > 0):
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1.0

    # --- 10d return ---
    ret_10d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            if (not np.isnan(C[si, di])
                    and not np.isnan(C[si, di - 10])
                    and C[si, di - 10] > 0):
                ret_10d[si, di] = C[si, di] / C[si, di - 10] - 1.0

    # --- OI 5d change ---
    oi_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (not np.isnan(OI[si, di])
                    and not np.isnan(OI[si, di - 5])
                    and OI[si, di - 5] > 0):
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
            if (not np.isnan(H[si, di])
                    and not np.isnan(L[si, di])
                    and not np.isnan(C[si, di])):
                if C[si, di] > 0 and H[si, di] > L[si, di]:
                    daily_range[si, di] = (
                        (H[si, di] - L[si, di]) / C[si, di]
                    )

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
                    prev_c = (
                        C[si, j - 1]
                        if j > 0 and not np.isnan(C[si, j - 1])
                        else cc
                    )
                    atr_vals.append(
                        max(hh - ll, abs(hh - prev_c), abs(ll - prev_c))
                    )
            if atr_vals and not np.isnan(C[si, di]) and C[si, di] > 0:
                atrp[si, di] = np.mean(atr_vals) / C[si, di]

    print(f"  Raw factors done: {time.time() - t0:.1f}s", flush=True)
    return {
        "ret_5d": ret_5d,
        "ret_10d": ret_10d,
        "oi_5d": oi_5d,
        "vol_5d": vol_5d,
        "daily_range": daily_range,
        "rsi14": rsi14,
        "atrp": atrp,
    }


def compute_cross_sectional_ranks(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int, min_count: int = 10,
) -> Dict[str, np.ndarray]:
    """Rank all factors cross-sectionally (across commodities per day).

    Low rank = oversold / extreme for mean reversion.
    """
    t0 = time.time()
    print("[V29] Computing cross-sectional ranks...", flush=True)

    factors_to_rank = {
        "rank_ret5d": raw_factors["ret_5d"],
        "rank_ret10d": raw_factors["ret_10d"],
        "rank_oi5d": raw_factors["oi_5d"],
        "rank_vol": raw_factors["vol_5d"],
        "rank_range": raw_factors["daily_range"],
        "rank_rsi": raw_factors["rsi14"],
        "rank_atrp": raw_factors["atrp"],
    }

    INVERT_FACTORS = {"rank_ret5d", "rank_ret10d", "rank_oi5d", "rank_rsi"}

    ranks = {}
    for name, factor in factors_to_rank.items():
        rank_arr = np.full((NS, ND), np.nan)
        for di in range(ND):
            vals = factor[:, di]
            valid_count = np.sum(~np.isnan(vals))
            if valid_count < min_count:
                continue
            ranked = (
                pd.Series(vals)
                .rank(pct=True, na_option="keep")
                .values
            )
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
                ker_regime[si, di] = 1  # sideways -> good for MR
            elif ker_10[si, di] > 0.3:
                ker_regime[si, di] = -1  # trending -> avoid counter-trend
    return ker_regime


def build_composite_signal(
    ranks: Dict[str, np.ndarray],
    weights: Dict[str, float],
    NS: int, ND: int,
    min_factors: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build weighted composite rank from individual factor ranks.

    Also count how many factors confirm (rank > 0.5 for each factor).
    """
    t0 = time.time()
    print("[V29] Building composite signal...", flush=True)

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
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """Full signal pipeline."""
    if weights is None:
        weights = DEFAULT_WEIGHTS

    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_cross_sectional_ranks(raw, NS, ND)
    ker_regime = compute_ker(C, NS, ND)
    composite, n_confirm = build_composite_signal(ranks, weights, NS, ND)

    return {
        "composite": composite,
        "n_confirm": n_confirm,
        "ker_regime": ker_regime,
        "ranks": ranks,
    }


# ============================================================
# BACKTEST ENGINE — Adaptive extreme rank
# ============================================================
def backtest_v29(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
    top_n: int = 1,
    min_rank: float = 0.80,
    extreme_rank: float = 0.90,
    extreme_top_n: int = 3,
    atr_stop: float = 3.0,
    min_confidence: int = 3,
    use_ker_gate: bool = True,
    hold_days: int = 5,
    pyramid_ratio: float = 0.5,
    extreme_pyramid_ratio: float = 0.7,
    pyramid_day: int = 1,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Backtest with adaptive extreme rank thresholds.

    - Moderate oversold (min_rank <= rank < extreme_rank): top_n positions
    - Extreme oversold (rank >= extreme_rank): extreme_top_n positions
    - Pyramid ratio adapts: higher for extreme oversold entries
    """
    composite = sigs["composite"]
    ker_regime = sigs["ker_regime"]
    n_confirm = sigs["n_confirm"]

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

        pos_by_si: Dict[int, List] = defaultdict(list)
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

        # Pyramid on day-1 winners
        if pyramid_ratio > 0 or extreme_pyramid_ratio > 0:
            held_with_pos: Dict[int, List] = defaultdict(list)
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
                        # Use composite rank to determine pyramid ratio
                        rank_val = composite[si, di]
                        if (not np.isnan(rank_val)
                                and rank_val >= extreme_rank):
                            pyr_ratio = extreme_pyramid_ratio
                        else:
                            pyr_ratio = pyramid_ratio
                        pyr_alloc = base_alloc * pyr_ratio
                        c_now = C[si, di]
                        atr_v = []
                        for j in range(max(start_di, di - 14), di):
                            hh, ll, cc = H[si, j], L[si, j], C[si, j]
                            if not any(np.isnan([hh, ll, cc])):
                                atr_v.append(
                                    max(hh - ll,
                                        abs(hh - cc),
                                        abs(ll - cc))
                                )
                        if atr_v:
                            atr = np.mean(atr_v)
                            additions.append(
                                (si, di, c_now,
                                 c_now - atr_stop * atr,
                                 pyr_alloc, True)
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

        # Adaptive top_n: allow more positions for extreme oversold
        current_max_n = top_n
        for si_check in range(NS):
            if (si_check not in held
                    and not np.isnan(composite[si_check, di])):
                if composite[si_check, di] >= extreme_rank:
                    current_max_n = extreme_top_n
                    break

        if len(positions) >= current_max_n:
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

            alloc = 1.0 / max(current_max_n, 1)
            candidates.append((composite[si, di], si, alloc))

        candidates.sort(key=lambda x: -x[0])
        for rank_val, si, alloc in candidates[:current_max_n]:
            if len(positions) >= current_max_n or si in held:
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
def analyze(trades: List[dict], equity: float, max_dd: float,
            label: str = "") -> Optional[dict]:
    """Analyze backtest results."""
    if not trades:
        print(f"  {label}: no trades")
        return None

    nw = sum(1 for t in trades if t["pnl_pct"] > 0)
    wr = nw / len(trades) * 100
    n_days = max(1, trades[-1]["di"] - trades[0]["di"])
    ann = ((equity / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
    ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
    rets = np.array(ap) / CASH0
    sh = (np.mean(rets) / np.std(rets) * np.sqrt(252)
          if np.std(rets) > 0 else 0)

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
        print(
            f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% "
            f"cum={cum:+.1%}"
        )

    return {
        "n": len(trades),
        "wr": wr,
        "dd": max_dd,
        "ann": ann,
        "sh": sh,
        "eq": equity,
    }


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================
def walk_forward(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
    dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
    top_n: int = 1,
    extreme_top_n: int = 3,
    min_rank: float = 0.80,
    extreme_rank: float = 0.90,
    min_confidence: int = 3,
    hold_days: int = 5,
    atr_stop: float = 3.0,
    pyramid_ratio: float = 0.5,
    extreme_pyramid_ratio: float = 0.7,
) -> List[dict]:
    """Walk-forward validation: year-by-year out-of-sample."""
    print(f"\n{'=' * 70}")
    print(
        f"  WALK-FORWARD V29 "
        f"(tn={top_n} ext_tn={extreme_top_n} "
        f"mr={min_rank:.2f} ext={extreme_rank:.2f} "
        f"pyr={pyramid_ratio:.1f} ext_pyr={extreme_pyramid_ratio:.1f})"
    )
    print(f"{'=' * 70}")

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

        trades, _, _ = backtest_v29(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=top_n,
            extreme_top_n=extreme_top_n,
            min_rank=min_rank,
            extreme_rank=extreme_rank,
            hold_days=hold_days,
            atr_stop=atr_stop,
            min_confidence=min_confidence,
            use_ker_gate=True,
            pyramid_ratio=pyramid_ratio,
            extreme_pyramid_ratio=extreme_pyramid_ratio,
            pyramid_day=1,
            start_di=test_start,
            end_di=test_end_idx + 1,
        )

        test_trades = [t for t in trades if dates[t["di"]].year == test_year]
        all_trades.extend(test_trades)

        if test_trades:
            n = len(test_trades)
            nw = sum(1 for t in test_trades if t["pnl_pct"] > 0)
            wr_val = nw / n * 100
            avg = np.mean([t["pnl_pct"] for t in test_trades])
            print(
                f"  {test_year}: {n}t WR={wr_val:.1f}% avg={avg:+.2f}%",
                flush=True,
            )
        else:
            print(f"  {test_year}: no trades", flush=True)

    if all_trades:
        nw = sum(1 for t in all_trades if t["pnl_pct"] > 0)
        wr_val = nw / len(all_trades) * 100
        avg = np.mean([t["pnl_pct"] for t in all_trades])
        cum = np.prod(
            [1 + t["pnl_pct"] / 100 for t in all_trades]
        ) - 1
        n_pyr = sum(1 for t in all_trades if t.get("pyr"))
        print(
            f"\n  WF TOTAL: {len(all_trades)}t (pyr:{n_pyr}) "
            f"WR={wr_val:.1f}% avg={avg:+.2f}% cum={cum:+.1%}"
        )
        return all_trades
    return []


# ============================================================
# PARAMETER SWEEP
# ============================================================
def sweep(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
    dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
    start_di: int = 60,
) -> List[dict]:
    """Sweep over parameters to find best configuration."""
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (V29)")
    print("=" * 70)

    results: List[dict] = []

    sweep_configs = [
        # (min_rank, extreme_rank, top_n, extreme_top_n,
        #  pyramid, extreme_pyramid, atr_stop, min_confidence)
        (0.80, 0.90, 1, 3, 0.0, 0.0, 2.5, 2),
        (0.80, 0.90, 1, 3, 0.0, 0.0, 3.0, 2),
        (0.80, 0.90, 1, 3, 0.5, 0.7, 2.5, 2),
        (0.80, 0.90, 1, 3, 0.5, 0.7, 3.0, 2),
        (0.80, 0.90, 1, 3, 0.5, 0.7, 3.0, 3),
        (0.80, 0.90, 1, 2, 0.5, 0.7, 3.0, 2),
        (0.80, 0.90, 1, 2, 0.5, 0.7, 3.0, 3),
        (0.85, 0.90, 1, 3, 0.0, 0.0, 2.5, 2),
        (0.85, 0.90, 1, 3, 0.0, 0.0, 3.0, 2),
        (0.85, 0.90, 1, 3, 0.5, 0.7, 2.5, 2),
        (0.85, 0.90, 1, 3, 0.5, 0.7, 3.0, 2),
        (0.85, 0.90, 1, 3, 0.5, 0.7, 3.0, 3),
        (0.85, 0.90, 1, 2, 0.5, 0.7, 3.0, 2),
        (0.85, 0.90, 1, 2, 0.5, 0.7, 3.0, 3),
        (0.90, 0.95, 1, 3, 0.0, 0.0, 3.0, 2),
        (0.90, 0.95, 1, 3, 0.5, 0.7, 3.0, 2),
        (0.90, 0.95, 1, 3, 0.5, 0.7, 3.0, 3),
        (0.90, 0.95, 1, 2, 0.5, 0.7, 3.0, 2),
        (0.90, 0.95, 1, 2, 0.5, 0.7, 3.0, 3),
        (0.90, 0.95, 1, 3, 0.7, 0.7, 3.0, 3),
        # Non-adaptive baselines: fixed min_rank only
        (0.80, 0.80, 1, 1, 0.0, 0.0, 3.0, 2),
        (0.80, 0.80, 1, 1, 0.5, 0.5, 3.0, 2),
        (0.85, 0.85, 1, 1, 0.0, 0.0, 3.0, 2),
        (0.85, 0.85, 1, 1, 0.5, 0.5, 3.0, 2),
        (0.90, 0.90, 1, 1, 0.0, 0.0, 3.0, 2),
        (0.90, 0.90, 1, 1, 0.5, 0.5, 3.0, 2),
        (0.95, 0.95, 1, 1, 0.0, 0.0, 3.0, 2),
        (0.95, 0.95, 1, 1, 0.5, 0.5, 3.0, 2),
        # Moderate pyramid
        (0.80, 0.90, 1, 3, 0.5, 0.5, 3.0, 2),
        (0.85, 0.90, 1, 3, 0.5, 0.5, 3.0, 2),
        (0.80, 0.90, 1, 3, 0.5, 0.5, 3.0, 3),
        (0.85, 0.90, 1, 3, 0.5, 0.5, 3.0, 3),
    ]

    for cfg in sweep_configs:
        (min_rank, extreme_rank, top_n, extreme_top_n,
         pyr, ext_pyr, atr_stop, mc) = cfg

        trades, eq, dd = backtest_v29(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=top_n,
            extreme_top_n=extreme_top_n,
            min_rank=min_rank,
            extreme_rank=extreme_rank,
            atr_stop=atr_stop,
            min_confidence=mc,
            use_ker_gate=True,
            hold_days=5,
            pyramid_ratio=pyr,
            extreme_pyramid_ratio=ext_pyr,
            pyramid_day=1,
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
        sh_val = (np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252)
                  if np.std(rets_arr) > 0 else 0)

        results.append({
            "mr": min_rank,
            "er": extreme_rank,
            "tn": top_n,
            "etn": extreme_top_n,
            "pyr": pyr,
            "epyr": ext_pyr,
            "atr": atr_stop,
            "mc": mc,
            "n": len(trades),
            "wr": wr,
            "ann": ann,
            "dd": dd,
            "sharpe": sh_val,
        })

    results.sort(key=lambda x: -x["sharpe"])
    print(
        f"\n{'MR':>4} {'ER':>4} {'TN':>3} {'ETN':>4} "
        f"{'Pyr':>4} {'EPyr':>5} {'ATR':>4} {'MC':>3} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}"
    )
    print("-" * 80)
    for r in results[:30]:
        print(
            f"{r['mr']:>4.2f} {r['er']:>4.2f} {r['tn']:>3} {r['etn']:>4} "
            f"{r['pyr']:>4.1f} {r['epyr']:>5.1f} {r['atr']:>4.1f} "
            f"{r['mc']:>3} "
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
    print("  V29: EXTREME RANK OVERSOLD MEAN REVERSION")
    print("  Only trade the most extreme oversold readings")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start="2016-01-01")
    print(
        f"  {NS} sym, {ND} days, "
        f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}"
    )

    sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND)

    # Find 2019 start index for OOS testing
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # === 1. Walk-Forward Validation with default adaptive config ===
    print("\n" + "=" * 70)
    print("  WALK-FORWARD VALIDATION (2019-2026)")
    print("=" * 70)

    wf_configs = [
        # (min_rank, extreme_rank, top_n, extreme_top_n,
        #  pyramid, extreme_pyramid)
        (0.80, 0.90, 1, 3, 0.5, 0.7),
        (0.80, 0.90, 1, 3, 0.0, 0.0),
        (0.85, 0.90, 1, 3, 0.5, 0.7),
        (0.90, 0.95, 1, 3, 0.5, 0.7),
        # Non-adaptive baselines
        (0.80, 0.80, 1, 1, 0.5, 0.5),
        (0.90, 0.90, 1, 1, 0.5, 0.5),
        (0.95, 0.95, 1, 1, 0.5, 0.5),
    ]

    for min_rank, ext_rank, tn, etn, pyr, epyr in wf_configs:
        walk_forward(
            C, O, H, L, V, OI, NS, ND, dates, syms, sigs,
            top_n=tn,
            extreme_top_n=etn,
            min_rank=min_rank,
            extreme_rank=ext_rank,
            min_confidence=3,
            hold_days=5,
            atr_stop=3.0,
            pyramid_ratio=pyr,
            extreme_pyramid_ratio=epyr,
        )

    # === 2. Full 10-year backtest with adaptive profiles ===
    print("\n" + "=" * 70)
    print("  FULL 2016-2026 (10 years) -- ADAPTIVE PROFILES")
    print("=" * 70)

    profiles = [
        (0.80, 0.90, 1, 3, 0.5, 0.7,
         "Moderate+Extreme (mr=0.80 ext=0.90)"),
        (0.80, 0.90, 1, 3, 0.0, 0.0,
         "Moderate+Extreme no pyr"),
        (0.85, 0.90, 1, 3, 0.5, 0.7,
         "High+Extreme (mr=0.85 ext=0.90)"),
        (0.90, 0.95, 1, 3, 0.5, 0.7,
         "VeryHigh+Ultra (mr=0.90 ext=0.95)"),
        (0.80, 0.80, 1, 1, 0.5, 0.5,
         "Baseline mr=0.80 non-adaptive"),
        (0.90, 0.90, 1, 1, 0.5, 0.5,
         "Baseline mr=0.90 non-adaptive"),
        (0.95, 0.95, 1, 1, 0.5, 0.5,
         "Baseline mr=0.95 non-adaptive"),
    ]

    for min_rank, ext_rank, tn, etn, pyr, epyr, label in profiles:
        trades, eq, dd = backtest_v29(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=tn,
            extreme_top_n=etn,
            min_rank=min_rank,
            extreme_rank=ext_rank,
            atr_stop=3.0,
            min_confidence=3,
            use_ker_gate=True,
            hold_days=5,
            pyramid_ratio=pyr,
            extreme_pyramid_ratio=epyr,
            pyramid_day=1,
            start_di=60,
        )
        print(f"\n  {label}")
        analyze(trades, eq, dd, label)

    # === 3. Parameter sweep (2019-2026) ===
    results = sweep(
        C, O, H, L, V, OI, NS, ND, dates, syms, sigs,
        start_di=bt_2019,
    )

    # === 4. Best config full 10-year ===
    if results:
        print("\n" + "=" * 70)
        print("  BEST CONFIG -- FULL 10-YEAR")
        print("=" * 70)

        for r in results[:5]:
            trades, eq, dd = backtest_v29(
                C, O, H, L, NS, ND, dates, syms, sigs,
                top_n=r["tn"],
                extreme_top_n=r["etn"],
                min_rank=r["mr"],
                extreme_rank=r["er"],
                atr_stop=r["atr"],
                min_confidence=r["mc"],
                use_ker_gate=True,
                hold_days=5,
                pyramid_ratio=r["pyr"],
                extreme_pyramid_ratio=r["epyr"],
                pyramid_day=1,
                start_di=60,
            )
            label = (
                f"mr={r['mr']:.2f} er={r['er']:.2f} "
                f"tn={r['tn']} etn={r['etn']} "
                f"pyr={r['pyr']:.1f} epyr={r['epyr']:.1f} "
                f"atr={r['atr']} mc={r['mc']}"
            )
            print(f"\n  FULL {label}")
            analyze(trades, eq, dd, label)

        # === 5. Walk-forward for best overall config ===
        best = results[0]
        print("\n" + "=" * 70)
        print(
            f"  BEST WF: mr={best['mr']:.2f} er={best['er']:.2f} "
            f"tn={best['tn']} etn={best['etn']} "
            f"pyr={best['pyr']:.1f} epyr={best['epyr']:.1f} "
            f"atr={best['atr']} mc={best['mc']}"
        )
        print("=" * 70)
        walk_forward(
            C, O, H, L, V, OI, NS, ND, dates, syms, sigs,
            top_n=best["tn"],
            extreme_top_n=best["etn"],
            min_rank=best["mr"],
            extreme_rank=best["er"],
            min_confidence=best["mc"],
            hold_days=5,
            atr_stop=best["atr"],
            pyramid_ratio=best["pyr"],
            extreme_pyramid_ratio=best["epyr"],
        )

    print(f"\n[V29] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
