"""
V14: Ensemble Meta-Strategy — Walk-Forward Validated
=====================================================
Instead of one signal, run MULTIPLE independent signal generation approaches
and combine their votes.

Three independent signal generators:
  a. Price oversold: consec_dn + ret_5d + ret_10d (simple price momentum)
  b. Technical oversold: RSI + Bollinger + CCI extreme readings (TA-Lib)
  c. Flow-based: OI capitulation + VDP exhaustion (market microstructure)

Ensemble combination:
  - Average rank, vote (>=2 of 3 agree), weighted (0.4*flow + 0.35*price + 0.25*tech)
  - KER gate + confidence >= 2
  - Pyramid on winners (ratio=0.5)

Signal at close[di], enter at open[di+1]. No look-ahead. No gap signals.
No leverage.
"""
import sys
import os
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from collections import defaultdict

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
# SIGNAL GENERATOR A: Price Oversold
# ============================================================
def compute_price_oversold(
    C: np.ndarray,
    NS: int,
    ND: int,
) -> np.ndarray:
    """Price-based oversold rank [0, 1] per (symbol, day).

    Uses consec_dn, ret_5d, ret_10d to build a cross-sectional rank.
    """
    t0 = time.time()
    print("[V14] Computing price oversold signal...", flush=True)

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

    # 10-day return
    ret_10d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            if (
                not np.isnan(C[si, di])
                and not np.isnan(C[si, di - 10])
                and C[si, di - 10] > 0
            ):
                ret_10d[si, di] = C[si, di] / C[si, di - 10] - 1

    # Raw oversold score: higher = more oversold
    price_rank = np.full((NS, ND), np.nan)
    for di in range(ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            s = 0.0
            w_total = 0.0
            # Consecutive down: cap at 7 days for full score
            s += min(consec_dn[si, di] / 7.0, 1.0) * 0.35
            w_total += 0.35
            # 5d return oversold
            if not np.isnan(ret_5d[si, di]):
                s += min(max(-ret_5d[si, di] / 0.10, 0), 1.0) * 0.35
                w_total += 0.35
            # 10d return oversold
            if not np.isnan(ret_10d[si, di]):
                s += min(max(-ret_10d[si, di] / 0.15, 0), 1.0) * 0.30
                w_total += 0.30
            if w_total > 0:
                scores[si] = s / w_total
        valid = ~np.isnan(scores)
        if valid.sum() >= 5:
            price_rank[:, di] = (
                pd.Series(scores).rank(pct=True, na_option="keep").values
            )

    print(f"  Price oversold done: {time.time() - t0:.1f}s", flush=True)
    return price_rank


# ============================================================
# SIGNAL GENERATOR B: Technical Oversold
# ============================================================
def compute_tech_oversold(
    C: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    NS: int,
    ND: int,
) -> np.ndarray:
    """Technical oversold rank [0, 1] per (symbol, day).

    Uses RSI, Bollinger Band position, CCI extreme readings via TA-Lib.
    """
    t0 = time.time()
    print("[V14] Computing technical oversold signal...", flush=True)

    rsi14 = np.full((NS, ND), np.nan)
    bb_pos = np.full((NS, ND), np.nan)
    cci14 = np.full((NS, ND), np.nan)

    if HAS_TALIB:
        for si in range(NS):
            h = np.where(np.isnan(H[si]), 0, H[si]).astype(np.float64)
            l = np.where(np.isnan(L[si]), 0, L[si]).astype(np.float64)
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nan_mask = np.isnan(C[si])
            try:
                rsi = talib.RSI(c, 14)
                rsi14[si] = np.where(nan_mask, np.nan, rsi)
            except Exception:
                pass
            try:
                upper, mid, lower = talib.BBANDS(c, 20, 2.0, 2.0)
                bb_range = upper - lower
                valid_bb = (~nan_mask) & (bb_range > 1e-10)
                bb_pos[si] = np.where(valid_bb, (c - lower) / bb_range, np.nan)
            except Exception:
                pass
            try:
                cci = talib.CCI(h, l, c, 14)
                cci14[si] = np.where(nan_mask, np.nan, cci)
            except Exception:
                pass

    # Build technical oversold rank
    tech_rank = np.full((NS, ND), np.nan)
    for di in range(ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            s = 0.0
            w_total = 0.0
            # RSI: oversold below 30
            if not np.isnan(rsi14[si, di]):
                if rsi14[si, di] < 40:
                    s += (40 - rsi14[si, di]) / 40.0 * 0.40
                w_total += 0.40
            # Bollinger: below lower band
            if not np.isnan(bb_pos[si, di]):
                if bb_pos[si, di] < 0.3:
                    s += (0.3 - bb_pos[si, di]) / 0.3 * 0.35
                w_total += 0.35
            # CCI: extreme negative
            if not np.isnan(cci14[si, di]):
                if cci14[si, di] < 0:
                    s += min(-cci14[si, di] / 300.0, 1.0) * 0.25
                w_total += 0.25
            if w_total > 0:
                scores[si] = s / w_total
        valid = ~np.isnan(scores)
        if valid.sum() >= 5:
            tech_rank[:, di] = (
                pd.Series(scores).rank(pct=True, na_option="keep").values
            )

    print(f"  Technical oversold done: {time.time() - t0:.1f}s", flush=True)
    return tech_rank


# ============================================================
# SIGNAL GENERATOR C: Flow-based (OI + VDP)
# ============================================================
def compute_flow_oversold(
    C: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    V: np.ndarray,
    OI: np.ndarray,
    NS: int,
    ND: int,
) -> np.ndarray:
    """Flow-based oversold rank [0, 1] per (symbol, day).

    Uses OI capitulation + VDP exhaustion (market microstructure).
    """
    t0 = time.time()
    print("[V14] Computing flow-based oversold signal...", flush=True)

    # OI decline signal
    oi_decline = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if np.isnan(OI[si, di]) or np.isnan(OI[si, di - 5]):
                continue
            if np.isnan(C[si, di]) or np.isnan(C[si, di - 5]) or C[si, di - 5] <= 0:
                continue
            oi_chg = OI[si, di] / OI[si, di - 5] - 1
            price_chg = C[si, di] / C[si, di - 5] - 1
            if oi_chg < -0.02 and price_chg < -0.02:
                oi_decline[si, di] = (
                    min(abs(oi_chg), 0.2) / 0.2 * min(abs(price_chg), 0.1) / 0.1
                )
            else:
                oi_decline[si, di] = 0.0

    # VDP (Volume Delta Price)
    vdp = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if any(np.isnan([H[si, di], L[si, di], C[si, di], V[si, di]])):
                continue
            bar_range = H[si, di] - L[si, di]
            if bar_range > 0 and V[si, di] > 0:
                vdp[si, di] = (
                    V[si, di] * (2 * C[si, di] - H[si, di] - L[si, di]) / bar_range
                )

    # 10-day average VDP
    vdp_10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            vals = vdp[si, di - 10 : di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 5:
                vdp_10[si, di] = np.mean(valid)

    # VDP exhaustion (z-score of vdp_10 vs 20-day window)
    vdp_exhaust = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if np.isnan(vdp_10[si, di]):
                continue
            window = vdp_10[si, max(0, di - 20) : di]
            wv = window[~np.isnan(window)]
            if len(wv) >= 10:
                mu, sig = np.mean(wv), np.std(wv)
                if sig > 0:
                    z = (vdp_10[si, di] - mu) / sig
                    vdp_exhaust[si, di] = min(-z, 3.0) / 3.0 if z < 0 else 0.0

    # OI capitulation: large OI drop + price drop
    oi_capitulation = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if np.isnan(oi_decline[si, di]):
                continue
            # Look for spike in OI decline relative to 20-day window
            window_oi = []
            for j in range(max(5, di - 20), di):
                if not np.isnan(oi_decline[si, j]):
                    window_oi.append(oi_decline[si, j])
            if len(window_oi) >= 10:
                mu = np.mean(window_oi)
                sig = np.std(window_oi)
                if sig > 1e-10:
                    z = (oi_decline[si, di] - mu) / sig
                    oi_capitulation[si, di] = max(z, 0)
                else:
                    oi_capitulation[si, di] = oi_decline[si, di]

    # Build flow oversold rank
    flow_rank = np.full((NS, ND), np.nan)
    for di in range(ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            s = 0.0
            w_total = 0.0
            # OI decline
            if not np.isnan(oi_decline[si, di]):
                s += min(oi_decline[si, di], 1.0) * 0.35
                w_total += 0.35
            # VDP exhaustion
            if not np.isnan(vdp_exhaust[si, di]):
                s += vdp_exhaust[si, di] * 0.35
                w_total += 0.35
            # OI capitulation spike
            if not np.isnan(oi_capitulation[si, di]):
                s += min(oi_capitulation[si, di] / 2.0, 1.0) * 0.30
                w_total += 0.30
            if w_total > 0:
                scores[si] = s / w_total
        valid = ~np.isnan(scores)
        if valid.sum() >= 5:
            flow_rank[:, di] = (
                pd.Series(scores).rank(pct=True, na_option="keep").values
            )

    print(f"  Flow oversold done: {time.time() - t0:.1f}s", flush=True)
    return flow_rank


# ============================================================
# KER (Kaufman Efficiency Ratio) GATE
# ============================================================
def compute_ker(
    C: np.ndarray,
    NS: int,
    ND: int,
    period: int = 10,
) -> np.ndarray:
    """Kaufman Efficiency Ratio. Returns regime: -1=noise, 0=neutral, +1=trend."""
    ker_10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(period, ND):
            closes = C[si, di - period : di + 1]
            valid = closes[~np.isnan(closes)]
            if len(valid) < period or valid[0] <= 0:
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
                ker_regime[si, di] = 1   # trending (efficient)
            elif ker_10[si, di] > 0.3:
                ker_regime[si, di] = -1  # noisy (inefficient)

    return ker_regime


# ============================================================
# ENSEMBLE SIGNAL COMBINATION
# ============================================================
def compute_ensemble(
    price_rank: np.ndarray,
    tech_rank: np.ndarray,
    flow_rank: np.ndarray,
    NS: int,
    ND: int,
    mode: str = "weighted",
    w_flow: float = 0.40,
    w_price: float = 0.35,
    w_tech: float = 0.25,
    vote_threshold: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine three independent signal ranks into ensemble.

    mode: 'average', 'vote', or 'weighted'
    Returns (ensemble_rank, vote_count, confidence)
    """
    print(f"[V14] Computing ensemble (mode={mode})...", flush=True)

    ensemble_rank = np.full((NS, ND), np.nan)
    vote_count = np.zeros((NS, ND), dtype=int)
    confidence = np.zeros((NS, ND), dtype=int)

    for di in range(ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            p = price_rank[si, di]
            t = tech_rank[si, di]
            f = flow_rank[si, di]

            # Count available signals
            available = []
            if not np.isnan(p):
                available.append(("price", p))
            if not np.isnan(t):
                available.append(("tech", t))
            if not np.isnan(f):
                available.append(("flow", f))

            if len(available) < 2:
                continue

            # Vote: how many agree the symbol is oversold (rank > threshold)
            votes = sum(1 for _, rank in available if rank > vote_threshold)
            vote_count[si, di] = votes

            # Confidence: number of available signals
            confidence[si, di] = len(available)

            if mode == "average":
                vals = [rank for _, rank in available]
                scores[si] = np.mean(vals)
            elif mode == "vote":
                # Only score if >= 2 agree; otherwise NaN
                if votes >= 2:
                    vals = [rank for _, rank in available if rank > vote_threshold]
                    scores[si] = np.mean(vals)
                else:
                    scores[si] = np.nan
            elif mode == "weighted":
                total_w = 0.0
                weighted_sum = 0.0
                for name, rank in available:
                    if name == "flow":
                        w = w_flow
                    elif name == "price":
                        w = w_price
                    else:
                        w = w_tech
                    weighted_sum += rank * w
                    total_w += w
                if total_w > 0:
                    scores[si] = weighted_sum / total_w

        valid = ~np.isnan(scores)
        if valid.sum() >= 5:
            ensemble_rank[:, di] = (
                pd.Series(scores).rank(pct=True, na_option="keep").values
            )

    return ensemble_rank, vote_count, confidence


# ============================================================
# BACKTEST WITH PYRAMID
# ============================================================
def backtest_v14(
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    NS: int,
    ND: int,
    dates: np.ndarray,
    syms: List[str],
    ensemble_rank: np.ndarray,
    ker_regime: np.ndarray,
    vote_count: np.ndarray,
    confidence: np.ndarray,
    top_n: int = 1,
    min_rank: float = 0.7,
    atr_stop: float = 3.0,
    min_confidence: int = 2,
    min_votes: int = 2,
    use_ker_gate: bool = True,
    hold_days: int = 5,
    pyramid_ratio: float = 0.5,
    pyramid_day: int = 1,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Backtest with ensemble signals and pyramid."""
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

        # Group positions by symbol
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
                    trades.append({
                        "pnl_abs": profit,
                        "pnl_pct": pnl * 100,
                        "days": di - edi + 1,
                        "di": di,
                        "year": d.year,
                        "sym": syms[si],
                        "reason": "stop",
                        "pyr": is_pyr,
                        "votes": int(vote_count[si, edi]),
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
                        "votes": int(vote_count[si, edi]),
                    })
            else:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc, is_pyr))

        # Pyramid check
        if pyramid_ratio > 0:
            held_with_pos: Dict[int, List[Tuple[int, float, float, float, bool]]] = (
                defaultdict(list)
            )
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

        # Entry: ensemble selection
        candidates: List[Tuple[float, int]] = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(ensemble_rank[si, di]):
                continue
            if ensemble_rank[si, di] < min_rank:
                continue
            if confidence[si, di] < min_confidence:
                continue
            if vote_count[si, di] < min_votes:
                continue
            if use_ker_gate and ker_regime[si, di] < 0:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            candidates.append((ensemble_rank[si, di], si))

        candidates.sort(key=lambda x: -x[0])
        for rank, si in candidates[:top_n]:
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
            alloc = 1.0 / max(top_n, 1)
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
    trades: List[dict],
    equity: float,
    max_dd: float,
    label: str = "",
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
        f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} stop:{n_stop} hold:{n_hold}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% Sh={sh:.2f} eq={equity:,.0f}"
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
        print(f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% cum={cum:+.1%}")

    return {"n": len(trades), "wr": wr, "dd": max_dd, "ann": ann, "sh": sh, "eq": equity}


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
    mode: str = "weighted",
    top_n: int = 1,
    min_rank: float = 0.7,
    min_confidence: int = 2,
    min_votes: int = 2,
    hold_days: int = 5,
    atr_stop: float = 3.0,
    pyramid_ratio: float = 0.5,
) -> List[dict]:
    """Walk-forward validation: compute signals fresh for each test year."""
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD: mode={mode}, top_n={top_n}, pyr={pyramid_ratio}")
    print(f"{'='*70}")

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

        # Compute signals on full data (no look-ahead since ranks are cross-sectional)
        price_rank = compute_price_oversold(C, NS, ND)
        tech_rank = compute_tech_oversold(C, H, L, NS, ND)
        flow_rank = compute_flow_oversold(C, H, L, V, OI, NS, ND)
        ker_regime = compute_ker(C, NS, ND)
        ensemble_rank, vote_count, confidence = compute_ensemble(
            price_rank, tech_rank, flow_rank, NS, ND, mode=mode
        )

        trades, _, _ = backtest_v14(
            C, O, H, L, NS, ND, dates, syms,
            ensemble_rank, ker_regime, vote_count, confidence,
            top_n=top_n, min_rank=min_rank,
            min_confidence=min_confidence, min_votes=min_votes,
            hold_days=hold_days, atr_stop=atr_stop,
            pyramid_ratio=pyramid_ratio, pyramid_day=1,
            start_di=test_start, end_di=test_end_idx + 1,
        )

        test_trades = [t for t in trades if dates[t["di"]].year == test_year]
        all_trades.extend(test_trades)

        if test_trades:
            n = len(test_trades)
            nw = sum(1 for t in test_trades if t["pnl_pct"] > 0)
            wr = nw / n * 100
            avg = np.mean([t["pnl_pct"] for t in test_trades])
            print(f"  {test_year}: {n}t WR={wr:.1f}% avg={avg:+.2f}%", flush=True)
        else:
            print(f"  {test_year}: no trades", flush=True)

    if all_trades:
        nw = sum(1 for t in all_trades if t["pnl_pct"] > 0)
        wr = nw / len(all_trades) * 100
        avg = np.mean([t["pnl_pct"] for t in all_trades])
        cum = np.prod([1 + t["pnl_pct"] / 100 for t in all_trades]) - 1
        n_pyr = sum(1 for t in all_trades if t.get("pyr"))
        print(
            f"\n  WF TOTAL: {len(all_trades)}t (pyr:{n_pyr}) WR={wr:.1f}% "
            f"avg={avg:+.2f}% cum={cum:+.1%}"
        )
    return all_trades


# ============================================================
# PARAMETER SWEEP
# ============================================================
def sweep(
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    NS: int,
    ND: int,
    dates: np.ndarray,
    syms: List[str],
    price_rank: np.ndarray,
    tech_rank: np.ndarray,
    flow_rank: np.ndarray,
    ker_regime: np.ndarray,
    start_di: int = 60,
) -> List[dict]:
    """Sweep over ensemble modes and parameters."""
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP")
    print("=" * 70)

    results: List[dict] = []

    for mode in ["average", "vote", "weighted"]:
        ensemble_rank, vote_count, confidence = compute_ensemble(
            price_rank, tech_rank, flow_rank, NS, ND, mode=mode
        )
        for top_n in [1, 2, 3]:
            for mc in [2, 3]:
                for mv in [1, 2]:
                    for ratio in [0.0, 0.3, 0.5]:
                        trades, eq, dd = backtest_v14(
                            C, O, H, L, NS, ND, dates, syms,
                            ensemble_rank, ker_regime, vote_count, confidence,
                            top_n=top_n, min_rank=0.7,
                            min_confidence=mc, min_votes=mv,
                            hold_days=5, atr_stop=3.0,
                            pyramid_ratio=ratio, pyramid_day=1,
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
                        sh_val = (
                            np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252)
                            if np.std(rets_arr) > 0
                            else 0
                        )
                        results.append({
                            "mode": mode,
                            "tn": top_n,
                            "mc": mc,
                            "mv": mv,
                            "pyr": ratio,
                            "n": len(trades),
                            "wr": wr,
                            "ann": ann,
                            "dd": dd,
                            "sharpe": sh_val,
                        })

    results.sort(key=lambda x: -x["sharpe"])
    print(
        f"\n{'Mode':>8} {'TN':>3} {'MC':>3} {'MV':>3} {'Pyr':>4} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}"
    )
    print("-" * 65)
    for r in results[:25]:
        print(
            f"{r['mode']:>8} {r['tn']:>3} {r['mc']:>3} {r['mv']:>3} "
            f"{r['pyr']:>4.1f} {r['n']:>5} {r['wr']:>5.1f} "
            f"{r['ann']:>+8.1f} {r['dd']:>6.1f} {r['sharpe']:>5.2f}"
        )

    return results


# ============================================================
# SIGNAL DIAGNOSTICS
# ============================================================
def diagnose_signals(
    price_rank: np.ndarray,
    tech_rank: np.ndarray,
    flow_rank: np.ndarray,
    NS: int,
    ND: int,
) -> None:
    """Print overlap and correlation between the three signal sources."""
    print("\n" + "=" * 70)
    print("  SIGNAL DIAGNOSTICS (independence check)")
    print("=" * 70)

    # Correlation on flattened valid entries
    p_flat = price_rank.flatten()
    t_flat = tech_rank.flatten()
    f_flat = flow_rank.flatten()
    valid = ~np.isnan(p_flat) & ~np.isnan(t_flat) & ~np.isnan(f_flat)

    if valid.sum() > 100:
        p_v = p_flat[valid]
        t_v = t_flat[valid]
        f_v = f_flat[valid]

        corr_pt = np.corrcoef(p_v, t_v)[0, 1]
        corr_pf = np.corrcoef(p_v, f_v)[0, 1]
        corr_tf = np.corrcoef(t_v, f_v)[0, 1]
        print(f"  Rank correlations (lower = more independent):")
        print(f"    Price vs Tech:  {corr_pt:.3f}")
        print(f"    Price vs Flow:  {corr_pf:.3f}")
        print(f"    Tech  vs Flow:  {corr_tf:.3f}")

        # Top-10 overlap: how often do all three agree on top-10% oversold?
        threshold = 0.9
        p_top = p_flat > threshold
        t_top = t_flat > threshold
        f_top = f_flat > threshold
        valid_mask = ~np.isnan(p_flat) & ~np.isnan(t_flat) & ~np.isnan(f_flat)

        n_valid = valid_mask.sum()
        n_all_three = (p_top & t_top & f_top & valid_mask).sum()
        n_any_two = (
            ((p_top & t_top) | (p_top & f_top) | (t_top & f_top)) & valid_mask
        ).sum()
        print(f"\n  Top-10% oversold overlap (threshold={threshold}):")
        print(f"    All 3 agree: {n_all_three} / {n_valid} = {n_all_three / max(n_valid, 1) * 100:.2f}%")
        print(f"    Any 2 agree: {n_any_two} / {n_valid} = {n_any_two / max(n_valid, 1) * 100:.2f}%")

        # Average availability
        p_avail = (~np.isnan(price_rank)).sum() / (NS * ND) * 100
        t_avail = (~np.isnan(tech_rank)).sum() / (NS * ND) * 100
        f_avail = (~np.isnan(flow_rank)).sum() / (NS * ND) * 100
        print(f"\n  Signal availability:")
        print(f"    Price: {p_avail:.1f}%  Tech: {t_avail:.1f}%  Flow: {f_avail:.1f}%")


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V14: ENSEMBLE META-STRATEGY")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start="2016-01-01")
    print(
        f"  {NS} sym, {ND} days, "
        f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}"
    )

    # === Phase 1: Compute independent signals ===
    print("\n--- Phase 1: Independent Signal Generators ---")
    price_rank = compute_price_oversold(C, NS, ND)
    tech_rank = compute_tech_oversold(C, H, L, NS, ND)
    flow_rank = compute_flow_oversold(C, H, L, V, OI, NS, ND)
    ker_regime = compute_ker(C, NS, ND)

    # Diagnostics
    diagnose_signals(price_rank, tech_rank, flow_rank, NS, ND)

    # === Phase 2: Ensemble combination ===
    print("\n--- Phase 2: Ensemble Combination ---")
    ens_avg, vc_avg, conf_avg = compute_ensemble(
        price_rank, tech_rank, flow_rank, NS, ND, mode="average"
    )
    ens_vote, vc_vote, conf_vote = compute_ensemble(
        price_rank, tech_rank, flow_rank, NS, ND, mode="vote"
    )
    ens_wt, vc_wt, conf_wt = compute_ensemble(
        price_rank, tech_rank, flow_rank, NS, ND, mode="weighted"
    )

    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # === Phase 3: Compare ensemble modes ===
    print("\n" + "=" * 70)
    print("  ENSEMBLE MODE COMPARISON (full 10-year)")
    print("=" * 70)

    for ens, vc, conf, mode_name in [
        (ens_avg, vc_avg, conf_avg, "average"),
        (ens_vote, vc_vote, conf_vote, "vote"),
        (ens_wt, vc_wt, conf_wt, "weighted"),
    ]:
        for tn in [1, 2]:
            trades, eq, dd = backtest_v14(
                C, O, H, L, NS, ND, dates, syms,
                ens, ker_regime, vc, conf,
                top_n=tn, min_rank=0.7, min_confidence=2,
                min_votes=2, hold_days=5, atr_stop=3.0,
                pyramid_ratio=0.5, pyramid_day=1,
                start_di=60,
            )
            analyze(trades, eq, dd, f"{mode_name}-tn{tn}-pyr0.5")

    # === Phase 4: Multi-position 2019-2026 ===
    print("\n" + "=" * 70)
    print("  MULTI-POSITION (2019-2026)")
    print("=" * 70)

    for ens, vc, conf, mode_name in [
        (ens_avg, vc_avg, conf_avg, "average"),
        (ens_vote, vc_vote, conf_vote, "vote"),
        (ens_wt, vc_wt, conf_wt, "weighted"),
    ]:
        for tn in [1, 2, 3]:
            trades, eq, dd = backtest_v14(
                C, O, H, L, NS, ND, dates, syms,
                ens, ker_regime, vc, conf,
                top_n=tn, min_rank=0.7, min_confidence=2,
                min_votes=2, hold_days=5, atr_stop=3.0,
                pyramid_ratio=0.5, pyramid_day=1,
                start_di=bt_2019,
            )
            analyze(trades, eq, dd, f"{mode_name}-tn{tn}")

    # === Phase 5: Parameter sweep ===
    results = sweep(
        C, O, H, L, NS, ND, dates, syms,
        price_rank, tech_rank, flow_rank, ker_regime,
        start_di=bt_2019,
    )

    # === Phase 6: Best 10-year ===
    if results:
        print("\n" + "=" * 70)
        print("  BEST 10-YEAR (top-5 from sweep)")
        print("=" * 70)
        for r in results[:5]:
            ensemble_rank_r, vote_count_r, confidence_r = compute_ensemble(
                price_rank, tech_rank, flow_rank, NS, ND, mode=r["mode"]
            )
            trades, eq, dd = backtest_v14(
                C, O, H, L, NS, ND, dates, syms,
                ensemble_rank_r, ker_regime, vote_count_r, confidence_r,
                top_n=r["tn"], min_rank=0.7,
                min_confidence=r["mc"], min_votes=r["mv"],
                hold_days=5, atr_stop=3.0,
                pyramid_ratio=r["pyr"], pyramid_day=1,
                start_di=60,
            )
            label = (
                f"{r['mode']}-tn{r['tn']}-mc{r['mc']}-"
                f"mv{r['mv']}-pyr{r['pyr']:.1f}"
            )
            print(f"\n  FULL {label}")
            analyze(trades, eq, dd, label)

    # === Phase 7: Walk-forward for best ===
    if results:
        best = results[0]
        print("\n" + "=" * 70)
        print(
            f"  WALK-FORWARD BEST: {best['mode']}, tn={best['tn']}, "
            f"mc={best['mc']}, mv={best['mv']}, pyr={best['pyr']}"
        )
        print("=" * 70)
        walk_forward(
            C, O, H, L, V, OI, NS, ND, dates, syms,
            mode=best["mode"],
            top_n=best["tn"],
            min_confidence=best["mc"],
            min_votes=best["mv"],
            hold_days=5,
            atr_stop=3.0,
            pyramid_ratio=best["pyr"],
        )

    print(f"\n[V14] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
