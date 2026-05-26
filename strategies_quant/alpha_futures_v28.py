"""
V28: Volume-OI Divergence Mean Reversion
=========================================
Core thesis: When volume and OI disagree, the divergence reveals
the true nature of the price move.

Divergence signals:
  - STRONG capitulation: price down + OI down + volume down
    (all declining = full capitulation, positions being liquidated)
  - MODERATE: price down + OI down (liquidation, any volume)
    (positions closing, selling exhausting)
  - AVOID: price down + OI up (new shorts = conviction sell)
    (fresh positions being added against the drop)

Score: divergence_type (40%) + price_oversold_rank (25%) + vol_rank (15%)
       + consec_dn (10%) + rsi_rank (10%)
Cross-sectional rank, KER gate, confidence >= 2, hold 5d,
ATR stop 3.0, pyramid 0.5.

Signal at close[di], enter at open[di+1]. No look-ahead. No gap signals.
No leverage. Walk-forward validation.
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


# ============================================================
# DIVERGENCE SIGNALS: Volume-OI disagreement
# ============================================================
def compute_divergence_signals(
    OI: np.ndarray,
    C: np.ndarray,
    V: np.ndarray,
    NS: int,
    ND: int,
    oi_decline_threshold: float = -0.05,
    lookback: int = 5,
) -> Dict[str, np.ndarray]:
    """Compute volume-OI divergence signals.

    Key insight: When price falls + OI falls, positions are being
    liquidated (exhaustion). When price falls + OI rises, new shorts
    are being added (conviction sell, avoid).

    Returns dict with:
        divergence_score: (NS, ND) float, 0-3 scale
        oi_change: (NS, ND) 5d OI percent change
        vol_change: (NS, ND) 5d Volume percent change
        price_change: (NS, ND) 5d price return
    """
    t0 = time.time()
    print("[V28] Computing volume-OI divergence signals...", flush=True)

    oi_change = np.full((NS, ND), np.nan)
    vol_change = np.full((NS, ND), np.nan)
    price_change = np.full((NS, ND), np.nan)

    for si in range(NS):
        for di in range(lookback, ND):
            # OI 5d change
            oi_now = OI[si, di]
            oi_prev = OI[si, di - lookback]
            if (
                not np.isnan(oi_now)
                and not np.isnan(oi_prev)
                and oi_prev > 0
            ):
                oi_change[si, di] = oi_now / oi_prev - 1

            # Volume 5d change
            v_now = V[si, di]
            v_prev = V[si, di - lookback]
            if (
                not np.isnan(v_now)
                and not np.isnan(v_prev)
                and v_prev > 0
            ):
                vol_change[si, di] = v_now / v_prev - 1

            # Price 5d return
            c_now = C[si, di]
            c_prev = C[si, di - lookback]
            if (
                not np.isnan(c_now)
                and not np.isnan(c_prev)
                and c_prev > 0
            ):
                price_change[si, di] = c_now / c_prev - 1

    # Divergence score: based on the 3-way agreement pattern
    # STRONG = all three declining (price, OI, vol) -> score 3.0
    # MODERATE = price down + OI down (any volume) -> score 2.0
    # WEAK = price down + vol down (any OI) -> score 1.0
    # AVOID = price down + OI up -> score -1.0 (conviction sell)
    divergence_score = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(lookback, ND):
            oi_c = oi_change[si, di]
            vol_c = vol_change[si, di]
            px_c = price_change[si, di]

            if np.isnan(oi_c) or np.isnan(vol_c) or np.isnan(px_c):
                continue

            price_down = px_c < -0.03
            oi_down = oi_c < oi_decline_threshold
            vol_down = vol_c < -0.15

            if not price_down:
                # No downside to revert from
                divergence_score[si, di] = 0.0
                continue

            if price_down and oi_down and vol_down:
                # STRONG capitulation: all three declining
                oi_str = min(abs(oi_c), 0.3) / 0.3
                vol_str = min(abs(vol_c), 0.5) / 0.5
                px_str = min(abs(px_c), 0.15) / 0.15
                divergence_score[si, di] = 1.5 + oi_str * vol_str * px_str
            elif price_down and oi_down:
                # MODERATE: OI declining = liquidation
                oi_str = min(abs(oi_c), 0.3) / 0.3
                px_str = min(abs(px_c), 0.15) / 0.15
                divergence_score[si, di] = 0.8 + oi_str * px_str * 0.7
            elif price_down and vol_down:
                # WEAK: volume declining but OI not
                vol_str = min(abs(vol_c), 0.5) / 0.5
                px_str = min(abs(px_c), 0.15) / 0.15
                divergence_score[si, di] = 0.3 + vol_str * px_str * 0.3
            elif price_down and oi_c > 0:
                # AVOID: OI rising while price falls = new shorts
                divergence_score[si, di] = -0.5
            else:
                divergence_score[si, di] = 0.0

    # Volume rank: cross-sectional rank of volume decline
    vol_rank = np.full((NS, ND), np.nan)
    for di in range(lookback, ND):
        vc = vol_change[:, di]
        valid = ~np.isnan(vc)
        if valid.sum() >= 5:
            # Rank so that LOW volume change = high rank (more capitulation)
            vol_rank[:, di] = (
                pd.Series(-vc).rank(pct=True, na_option="keep").values
            )

    print(f"  Divergence signals done: {time.time() - t0:.1f}s", flush=True)
    return {
        "divergence_score": divergence_score,
        "oi_change": oi_change,
        "vol_change": vol_change,
        "price_change": price_change,
        "vol_rank": vol_rank,
    }


# ============================================================
# PRICE OVERSOLD SIGNALS
# ============================================================
def compute_price_signals(
    C: np.ndarray, NS: int, ND: int,
) -> Dict[str, np.ndarray]:
    """Price-based oversold signals."""
    t0 = time.time()
    print("[V28] Computing price signals...", flush=True)

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

    print(f"  Price signals done: {time.time() - t0:.1f}s", flush=True)
    return {
        "consec_dn": consec_dn,
        "ret_5d": ret_5d,
    }


# ============================================================
# RSI CONFIRMATION SIGNALS
# ============================================================
def compute_rsi_signals(
    C: np.ndarray, NS: int, ND: int,
) -> Dict[str, np.ndarray]:
    """RSI-based oversold confirmation."""
    t0 = time.time()
    print("[V28] Computing RSI signals...", flush=True)

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
    else:
        # Manual RSI fallback
        for si in range(NS):
            gains = np.zeros(ND)
            losses = np.zeros(ND)
            for di in range(1, ND):
                if (
                    not np.isnan(C[si, di])
                    and not np.isnan(C[si, di - 1])
                    and C[si, di - 1] > 0
                ):
                    delta = C[si, di] - C[si, di - 1]
                    gains[di] = max(delta, 0)
                    losses[di] = max(-delta, 0)

            avg_gain = np.full(ND, np.nan)
            avg_loss = np.full(ND, np.nan)
            for di in range(14, ND):
                if di == 14:
                    avg_gain[di] = np.mean(gains[1:15])
                    avg_loss[di] = np.mean(losses[1:15])
                else:
                    avg_gain[di] = (avg_gain[di - 1] * 13 + gains[di]) / 14
                    avg_loss[di] = (avg_loss[di - 1] * 13 + losses[di]) / 14
                if avg_loss[di] > 1e-10:
                    rs = avg_gain[di] / avg_loss[di]
                    rsi14[si, di] = 100 - 100 / (1 + rs)

    print(f"  RSI signals done: {time.time() - t0:.1f}s", flush=True)
    return {"rsi14": rsi14}


# ============================================================
# KER GATE (Kaufman Efficiency Ratio)
# ============================================================
def compute_ker(C: np.ndarray, NS: int, ND: int) -> np.ndarray:
    """Kaufman Efficiency Ratio for regime gating."""
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
                ker_regime[si, di] = 1  # sideways -> good for MR
            elif ker_10[si, di] > 0.3:
                ker_regime[si, di] = -1  # trending -> avoid counter-trend
    return ker_regime


# ============================================================
# COMPOSITE SIGNAL BUILDER
# ============================================================
def compute_all_signals(
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    V: np.ndarray,
    OI: np.ndarray,
    NS: int,
    ND: int,
    oi_decline_threshold: float = -0.05,
) -> Dict[str, np.ndarray]:
    """Compute all V28 signals and build composite divergence score.

    Score weights:
        divergence_type (40%) + price_oversold_rank (25%) + vol_rank (15%)
        + consec_dn (10%) + rsi_rank (10%)
    """
    t0 = time.time()
    print("[V28] Computing all signals...", flush=True)

    div_sigs = compute_divergence_signals(
        OI, C, V, NS, ND,
        oi_decline_threshold=oi_decline_threshold,
    )
    price_sigs = compute_price_signals(C, NS, ND)
    rsi_sigs = compute_rsi_signals(C, NS, ND)
    ker_regime = compute_ker(C, NS, ND)

    # Composite score
    raw_score = np.full((NS, ND), np.nan)

    for di in range(ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue

            total_score = 0.0
            total_weight = 0.0

            # --- Divergence type (40%) ---
            ds = div_sigs["divergence_score"][si, di]
            if not np.isnan(ds):
                # Normalize: positive divergence = capitulation signal
                # ds ranges roughly from -0.5 to 2.5
                div_norm = min(max(ds / 2.0, 0.0), 1.0)
                total_score += div_norm * 0.40
                total_weight += 0.40

            # --- Price oversold rank (25%) ---
            ret5 = price_sigs["ret_5d"][si, di]
            if not np.isnan(ret5):
                oversold = min(max(-ret5 / 0.10, 0), 1.0)
                total_score += oversold * 0.25
                total_weight += 0.25

            # --- Volume rank (15%) ---
            vr = div_sigs["vol_rank"][si, di]
            if not np.isnan(vr):
                # vol_rank already 0-1, high = low volume = capitulation
                total_score += vr * 0.15
                total_weight += 0.15

            # --- Consecutive down days (10%) ---
            consec = price_sigs["consec_dn"][si, di]
            consec_norm = min(consec / 5.0, 1.0)
            total_score += consec_norm * 0.10
            total_weight += 0.10

            # --- RSI oversold (10%) ---
            rsi = rsi_sigs["rsi14"][si, di]
            if not np.isnan(rsi):
                if rsi < 30:
                    total_score += (30 - rsi) / 30.0 * 0.10
                elif rsi < 40:
                    total_score += (40 - rsi) / 10.0 * 0.05
                total_weight += 0.10

            if total_weight > 0:
                scores[si] = total_score / total_weight

        # Cross-sectional rank
        valid = ~np.isnan(scores)
        if valid.sum() >= 5:
            raw_score[:, di] = (
                pd.Series(scores).rank(pct=True, na_option="keep").values
            )

    # Count confirmation signals per bar
    n_signals = np.zeros((NS, ND), dtype=int)
    for di in range(ND):
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            n = 0
            # Strong or moderate divergence
            ds = div_sigs["divergence_score"][si, di]
            if not np.isnan(ds) and ds > 0.8:
                n += 1
            # OI declining
            oi_c = div_sigs["oi_change"][si, di]
            if not np.isnan(oi_c) and oi_c < oi_decline_threshold:
                n += 1
            # Volume declining
            vol_c = div_sigs["vol_change"][si, di]
            if not np.isnan(vol_c) and vol_c < -0.15:
                n += 1
            # Price declining
            px_c = div_sigs["price_change"][si, di]
            if not np.isnan(px_c) and px_c < -0.03:
                n += 1
            # Consecutive down
            if price_sigs["consec_dn"][si, di] >= 3:
                n += 1
            # RSI oversold
            rsi = rsi_sigs["rsi14"][si, di]
            if not np.isnan(rsi) and rsi < 35:
                n += 1
            n_signals[si, di] = n

    print(f"  All signals done: {time.time() - t0:.1f}s", flush=True)
    return {
        "combo_rank": raw_score,
        "divergence_score": div_sigs["divergence_score"],
        "ker_regime": ker_regime,
        "n_signals": n_signals,
    }


# ============================================================
# BACKTEST ENGINE
# ============================================================
def backtest_v28(
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    NS: int,
    ND: int,
    dates: np.ndarray,
    syms: List[str],
    sigs: Dict[str, np.ndarray],
    top_n: int = 1,
    min_rank: float = 0.70,
    atr_stop: float = 3.0,
    min_confidence: int = 2,
    use_ker_gate: bool = True,
    hold_days: int = 5,
    pyramid_ratio: float = 0.5,
    pyramid_day: int = 1,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Day-by-day backtest with volume-OI divergence signals + pyramid."""
    combo_rank = sigs["combo_rank"]
    ker_regime = sigs["ker_regime"]
    n_signals = sigs["n_signals"]

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
                    trades.append(
                        {
                            "pnl_abs": profit,
                            "pnl_pct": pnl * 100,
                            "days": di - edi + 1,
                            "di": di,
                            "year": d.year,
                            "sym": syms[si],
                            "reason": "stop",
                            "pyr": is_pyr,
                        }
                    )
            elif hold >= hold_days:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    trades.append(
                        {
                            "pnl_abs": profit,
                            "pnl_pct": pnl * 100,
                            "days": di - edi + 1,
                            "di": di,
                            "year": d.year,
                            "sym": syms[si],
                            "reason": "hold",
                            "pyr": is_pyr,
                        }
                    )
            else:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc, is_pyr))

        # Pyramid on day-1 winners
        if pyramid_ratio > 0:
            held_with_pos: Dict[int, List[Tuple[int, float, float, float, bool]]] = (
                defaultdict(list)
            )
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
                                atr_v.append(
                                    max(hh - ll, abs(hh - cc), abs(ll - cc))
                                )
                        if atr_v:
                            atr = np.mean(atr_v)
                            additions.append(
                                (
                                    si,
                                    di,
                                    c_now,
                                    c_now - atr_stop * atr,
                                    pyr_alloc,
                                    True,
                                )
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

        # Entry: signal at close[di], enter at open[di+1]
        candidates = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(combo_rank[si, di]):
                continue
            if combo_rank[si, di] < min_rank:
                continue
            if n_signals[si, di] < min_confidence:
                continue
            if use_ker_gate and ker_regime[si, di] < 0:
                continue
            # Avoid divergence: if OI rising + price falling, skip
            if (
                "divergence_score" in sigs
                and not np.isnan(sigs["divergence_score"][si, di])
                and sigs["divergence_score"][si, di] < 0
            ):
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue

            alloc = 1.0 / max(top_n, 1)
            candidates.append((combo_rank[si, di], si, alloc))

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
        print(
            f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% cum={cum:+.1%}"
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
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    V: np.ndarray,
    OI: np.ndarray,
    NS: int,
    ND: int,
    dates: np.ndarray,
    syms: List[str],
    top_n: int = 1,
    min_rank: float = 0.70,
    min_confidence: int = 2,
    hold_days: int = 5,
    atr_stop: float = 3.0,
    pyramid_ratio: float = 0.5,
    oi_decline_threshold: float = -0.05,
) -> List[dict]:
    """Walk-forward: compute signals once, test year-by-year OOS."""
    print(f"\n{'=' * 70}")
    print(
        f"  WALK-FORWARD V28 (tn={top_n}, pyr={pyramid_ratio}, "
        f"oi_thr={oi_decline_threshold:.0%}, mr={min_rank:.2f})"
    )
    print(f"{'=' * 70}")

    sigs = compute_all_signals(
        C, O, H, L, V, OI, NS, ND,
        oi_decline_threshold=oi_decline_threshold,
    )

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

        trades, _, _ = backtest_v28(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=top_n, min_rank=min_rank, hold_days=hold_days,
            atr_stop=atr_stop, min_confidence=min_confidence,
            use_ker_gate=True, pyramid_ratio=pyramid_ratio,
            pyramid_day=1, start_di=test_start,
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
        cum = np.prod([1 + t["pnl_pct"] / 100 for t in all_trades]) - 1
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
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    V: np.ndarray,
    OI: np.ndarray,
    NS: int,
    ND: int,
    dates: np.ndarray,
    syms: List[str],
    start_di: int = 60,
) -> List[dict]:
    """Sweep over parameters to find best configuration."""
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (V28)")
    print("=" * 70)

    results: List[dict] = []

    for oi_thr in [-0.03, -0.05, -0.08]:
        sigs = compute_all_signals(
            C, O, H, L, V, OI, NS, ND,
            oi_decline_threshold=oi_thr,
        )
        for tn in [1, 2, 3]:
            for mr in [0.65, 0.70, 0.75]:
                for mc in [2, 3]:
                    for atr in [2.5, 3.0]:
                        for pyr in [0.0, 0.5]:
                            trades, eq, dd = backtest_v28(
                                C, O, H, L, NS, ND, dates, syms, sigs,
                                top_n=tn, min_rank=mr,
                                atr_stop=atr, min_confidence=mc,
                                use_ker_gate=True, hold_days=5,
                                pyramid_ratio=pyr, pyramid_day=1,
                                start_di=start_di,
                            )
                            if len(trades) < 10:
                                continue
                            nw = sum(1 for t in trades if t["pnl_pct"] > 0)
                            wr = nw / len(trades) * 100
                            n_days = max(
                                1, trades[-1]["di"] - trades[0]["di"]
                            )
                            ann = (
                                (eq / CASH0)
                                ** (1 / max(1.0, n_days / 252))
                                - 1
                            ) * 100
                            ap = [
                                t["pnl_abs"]
                                for t in sorted(trades, key=lambda x: x["di"])
                            ]
                            rets_arr = np.array(ap) / CASH0
                            sh_val = (
                                np.mean(rets_arr)
                                / np.std(rets_arr)
                                * np.sqrt(252)
                                if np.std(rets_arr) > 0
                                else 0
                            )
                            results.append(
                                {
                                    "oi": oi_thr,
                                    "tn": tn,
                                    "mr": mr,
                                    "mc": mc,
                                    "atr": atr,
                                    "pyr": pyr,
                                    "n": len(trades),
                                    "wr": wr,
                                    "ann": ann,
                                    "dd": dd,
                                    "sharpe": sh_val,
                                }
                            )

    results.sort(key=lambda x: -x["sharpe"])
    print(
        f"\n{'OI':>5} {'TN':>3} {'MR':>4} {'MC':>3} "
        f"{'ATR':>4} {'Pyr':>4} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}"
    )
    print("-" * 70)
    for r in results[:25]:
        print(
            f"{r['oi']:>5.0%} {r['tn']:>3} {r['mr']:>4.2f} {r['mc']:>3} "
            f"{r['atr']:>4.1f} {r['pyr']:>4.1f} "
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
    print("  V28: VOLUME-OI DIVERGENCE MEAN REVERSION")
    print("  Volume and OI disagreement = capitulation signal")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start="2016-01-01")
    print(
        f"  {NS} sym, {ND} days, "
        f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}"
    )

    # === 1. Default config full backtest ===
    print("\n" + "=" * 70)
    print("  FULL 10-YEAR (default config)")
    print("=" * 70)

    sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND)

    for tn in [1, 2, 3]:
        for pyr in [0.0, 0.5]:
            trades, eq, dd = backtest_v28(
                C, O, H, L, NS, ND, dates, syms, sigs,
                top_n=tn, min_rank=0.70, hold_days=5, atr_stop=3.0,
                min_confidence=2, use_ker_gate=True,
                pyramid_ratio=pyr, pyramid_day=1,
                start_di=60,
            )
            label = f"tn={tn}-pyr={pyr:.1f}"
            print(f"\n  {label}")
            analyze(trades, eq, dd, label)

    # === 2. 2019+ OOS ===
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    print("\n" + "=" * 70)
    print("  2019-2026 OOS")
    print("=" * 70)

    for tn in [1, 2, 3]:
        for pyr in [0.0, 0.5]:
            trades, eq, dd = backtest_v28(
                C, O, H, L, NS, ND, dates, syms, sigs,
                top_n=tn, min_rank=0.70, hold_days=5, atr_stop=3.0,
                min_confidence=2, use_ker_gate=True,
                pyramid_ratio=pyr, pyramid_day=1,
                start_di=bt_2019,
            )
            label = f"OOS-tn={tn}-pyr={pyr:.1f}"
            print(f"\n  {label}")
            analyze(trades, eq, dd, label)

    # === 3. Parameter sweep ===
    results = sweep(
        C, O, H, L, V, OI, NS, ND, dates, syms, start_di=bt_2019
    )

    # === 4. Best config full 10-year ===
    if results:
        print("\n" + "=" * 70)
        print("  BEST CONFIG -- FULL 10-YEAR")
        print("=" * 70)

        for r in results[:5]:
            sigs_best = compute_all_signals(
                C, O, H, L, V, OI, NS, ND,
                oi_decline_threshold=r["oi"],
            )
            trades, eq, dd = backtest_v28(
                C, O, H, L, NS, ND, dates, syms, sigs_best,
                top_n=r["tn"], min_rank=r["mr"],
                atr_stop=r["atr"], min_confidence=r["mc"],
                use_ker_gate=True, hold_days=5,
                pyramid_ratio=r["pyr"], pyramid_day=1,
                start_di=60,
            )
            label = (
                f"oi={r['oi']:.0%} tn={r['tn']} mr={r['mr']:.2f} "
                f"mc={r['mc']} atr={r['atr']} pyr={r['pyr']:.1f}"
            )
            print(f"\n  FULL {label}")
            analyze(trades, eq, dd, label)

    # === 5. Walk-forward for best config ===
    if results:
        best = results[0]
        print("\n" + "=" * 70)
        print(
            f"  WALK-FORWARD BEST: oi={best['oi']:.0%} "
            f"tn={best['tn']} mr={best['mr']:.2f} "
            f"mc={best['mc']} atr={best['atr']} pyr={best['pyr']:.1f}"
        )
        print("=" * 70)

        walk_forward(
            C, O, H, L, V, OI, NS, ND, dates, syms,
            top_n=best["tn"],
            min_rank=best["mr"],
            min_confidence=best["mc"],
            hold_days=5,
            atr_stop=best["atr"],
            pyramid_ratio=best["pyr"],
            oi_decline_threshold=best["oi"],
        )

    print(f"\n[V28] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
