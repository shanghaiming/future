"""
V7: Intraday Candle Pattern + Daily Signal Fusion
===================================================
Core innovation: detect bullish candle reversal patterns from daily OHLC
and use them as CONFIRMATION for the V1 multi-alpha oversold signal.

Signal at close[di], enter at open[di+1]. No look-ahead.
No gap signals. No leverage.

Candle patterns (all from daily OHLC, no intraday data needed):
  1. Hammer: long lower shadow, closes in upper half
  2. Bullish engulfing: today's body engulfs yesterday's
  3. Morning star: 3-candle reversal pattern
  4. Doji at bottom: tiny body after downtrend

Entry: V1 oversold rank >= 0.7 + at least one bullish candle pattern
       + KER gate + confidence >= 3 + pyramid on winners
"""
import sys
import os
import time
import warnings
from typing import Optional

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
# CANDLE PATTERN DETECTION
# ============================================================
def compute_candle_patterns(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int,
) -> dict:
    """Compute bullish candle reversal patterns from daily OHLC.

    Returns dict with pattern name -> bool array (NS, ND).
    All patterns are bullish reversal signals suitable for
    confirming oversold conditions.
    """
    t0 = time.time()
    print("[V7] Computing candle patterns...", flush=True)

    hammer = np.zeros((NS, ND), dtype=bool)
    bullish_engulf = np.zeros((NS, ND), dtype=bool)
    morning_star = np.zeros((NS, ND), dtype=bool)
    doji_bottom = np.zeros((NS, ND), dtype=bool)
    pattern_count = np.zeros((NS, ND), dtype=int)

    for si in range(NS):
        for di in range(2, ND):
            c0 = C[si, di]
            o0 = O[si, di]
            h0 = H[si, di]
            l0 = L[si, di]

            if any(np.isnan([c0, o0, h0, l0])):
                continue

            bar_range = h0 - l0
            if bar_range <= 0:
                continue

            body = abs(c0 - o0)
            upper_shadow = h0 - max(c0, o0)
            lower_shadow = min(c0, o0) - l0

            # --- Hammer ---
            # Long lower shadow (>60% of range), closes in upper half,
            # small upper shadow. Classic reversal candle.
            is_hammer = (
                lower_shadow / bar_range > 0.6
                and c0 > (h0 + l0) / 2
                and upper_shadow < 0.1 * bar_range
            )
            hammer[si, di] = is_hammer

            # --- Bullish Engulfing ---
            # Today bullish (C > O) and yesterday bearish (O_prev > C_prev)
            # Today's body engulfs yesterday's body
            c_prev = C[si, di - 1]
            o_prev = O[si, di - 1]
            if not np.isnan(c_prev) and not np.isnan(o_prev):
                prev_body_top = max(o_prev, c_prev)
                prev_body_bot = min(o_prev, c_prev)
                is_engulf = (
                    c0 > o0                          # today bullish
                    and o_prev > c_prev              # yesterday bearish
                    and c0 > prev_body_top           # engulfs top
                    and o0 < prev_body_bot           # engulfs bottom
                )
                bullish_engulf[si, di] = is_engulf

            # --- Morning Star ---
            # 3-candle pattern:
            #   Day -2: large bearish body
            #   Day -1: small body (star), gap down optional
            #   Day  0: large bullish body closing into day-2 body
            if di >= 2:
                c_m2 = C[si, di - 2]
                o_m2 = O[si, di - 2]
                c_m1 = C[si, di - 1]
                o_m1 = O[si, di - 1]

                if not any(np.isnan([c_m2, o_m2, c_m1, o_m1])):
                    body_m2 = abs(c_m2 - o_m2)
                    body_m1 = abs(c_m1 - o_m1)
                    body_m0 = body

                    m2_bearish = o_m2 > c_m2
                    m0_bullish = c0 > o0
                    star_small = body_m1 < 0.3 * body_m2 if body_m2 > 0 else False
                    close_into = c0 > (o_m2 + c_m2) / 2

                    is_morning = (
                        m2_bearish
                        and star_small
                        and m0_bullish
                        and close_into
                    )
                    morning_star[si, di] = is_morning

            # --- Doji at Bottom ---
            # Body < 10% of range after a downtrend (3+ consecutive down days)
            is_doji = body < 0.1 * bar_range
            if is_doji and di >= 3:
                down_count = 0
                for k in range(1, 4):
                    ck = C[si, di - k]
                    ck1 = C[si, di - k - 1]
                    if not np.isnan(ck) and not np.isnan(ck1) and ck1 > 0:
                        if ck < ck1:
                            down_count += 1
                        else:
                            break
                    else:
                        break
                doji_bottom[si, di] = down_count >= 3

            # Count total patterns
            pattern_count[si, di] = (
                int(hammer[si, di])
                + int(bullish_engulf[si, di])
                + int(morning_star[si, di])
                + int(doji_bottom[si, di])
            )

    n_hammer = hammer.sum()
    n_engulf = bullish_engulf.sum()
    n_morning = morning_star.sum()
    n_doji = doji_bottom.sum()
    n_any = (pattern_count > 0).sum()
    print(f"  Hammer={n_hammer} Engulf={n_engulf} "
          f"MorningStar={n_morning} DojiBottom={n_doji} "
          f"AnyPattern={n_any} ({time.time() - t0:.1f}s)", flush=True)

    return {
        "hammer": hammer,
        "bullish_engulf": bullish_engulf,
        "morning_star": morning_star,
        "doji_bottom": doji_bottom,
        "pattern_count": pattern_count,
    }


# ============================================================
# V1-STYLE MULTI-ALPHA SIGNALS (reused from V5)
# ============================================================
def compute_v1_signals(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
) -> dict:
    """Compute the V1 multi-alpha oversold composite score.

    Returns combo_rank (percentile), ker_regime, n_signals.
    """
    t0 = time.time()
    print("[V7] Computing V1 multi-alpha signals...", flush=True)

    # Consecutive down days
    consec_dn = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        consec = 0
        for di in range(1, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 1]) and C[si, di - 1] > 0:
                if C[si, di] < C[si, di - 1]:
                    consec += 1
                else:
                    consec = 0
            else:
                consec = 0
            consec_dn[si, di] = consec

    # 5-day return
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 5]) and C[si, di - 5] > 0:
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1

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

    # Volume delta position (VDP) exhaustion
    vdp = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if np.isnan(H[si, di]) or np.isnan(L[si, di]) or np.isnan(C[si, di]) or np.isnan(V[si, di]):
                continue
            bar_range = H[si, di] - L[si, di]
            if bar_range > 0 and V[si, di] > 0:
                vdp[si, di] = V[si, di] * (2 * C[si, di] - H[si, di] - L[si, di]) / bar_range

    vdp_10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            vals = vdp[si, di - 10:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 5:
                vdp_10[si, di] = np.mean(valid)

    vdp_exhaust = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if np.isnan(vdp_10[si, di]):
                continue
            window = vdp_10[si, max(0, di - 20):di]
            wv = window[~np.isnan(window)]
            if len(wv) >= 10:
                mu, sig = np.mean(wv), np.std(wv)
                if sig > 0:
                    z = (vdp_10[si, di] - mu) / sig
                    vdp_exhaust[si, di] = min(-z, 3.0) / 3.0 if z < 0 else 0.0

    # KER (Kaufman Efficiency Ratio)
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

    # RSI, BB, CCI via TA-Lib
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

    # Composite oversold score + cross-sectional rank
    raw_score = np.full((NS, ND), np.nan)
    for di in range(ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            s = 0.0
            w_total = 0.0
            s += min(consec_dn[si, di] / 5.0, 1.0) * 0.20
            w_total += 0.20
            if not np.isnan(ret_5d[si, di]):
                s += min(max(-ret_5d[si, di] / 0.1, 0), 1.0) * 0.20
                w_total += 0.20
            if not np.isnan(oi_decline[si, di]):
                s += oi_decline[si, di] * 0.20
                w_total += 0.20
            if not np.isnan(vdp_exhaust[si, di]):
                s += vdp_exhaust[si, di] * 0.15
                w_total += 0.15
            if not np.isnan(rsi14[si, di]):
                if rsi14[si, di] < 30:
                    s += (30 - rsi14[si, di]) / 30.0 * 0.10
                w_total += 0.10
            if not np.isnan(bb_pos[si, di]):
                if bb_pos[si, di] < 0.2:
                    s += (0.2 - bb_pos[si, di]) / 0.2 * 0.10
                w_total += 0.10
            if not np.isnan(cci14[si, di]):
                if cci14[si, di] < -100:
                    s += min((-100 - cci14[si, di]) / 200.0, 1.0) * 0.05
                w_total += 0.05
            if w_total > 0:
                scores[si] = s / w_total
        valid = ~np.isnan(scores)
        if valid.sum() >= 5:
            raw_score[:, di] = pd.Series(scores).rank(pct=True, na_option="keep").values

    # Signal confidence count
    n_signals = np.zeros((NS, ND), dtype=int)
    for di in range(ND):
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            n = 0
            if consec_dn[si, di] >= 3:
                n += 1
            if not np.isnan(ret_5d[si, di]) and ret_5d[si, di] < -0.03:
                n += 1
            if not np.isnan(oi_decline[si, di]) and oi_decline[si, di] > 0.1:
                n += 1
            if not np.isnan(vdp_exhaust[si, di]) and vdp_exhaust[si, di] > 0.3:
                n += 1
            if not np.isnan(rsi14[si, di]) and rsi14[si, di] < 35:
                n += 1
            if not np.isnan(bb_pos[si, di]) and bb_pos[si, di] < 0.15:
                n += 1
            if not np.isnan(cci14[si, di]) and cci14[si, di] < -100:
                n += 1
            n_signals[si, di] = n

    print(f"  Done: {time.time() - t0:.1f}s", flush=True)

    return {
        "combo_rank": raw_score,
        "ker_regime": ker_regime,
        "n_signals": n_signals,
    }


# ============================================================
# FUSION SIGNAL: V1 + CANDLE CONFIRMATION
# ============================================================
def compute_fused_signals(
    v1_sigs: dict, candle_sigs: dict, NS: int, ND: int,
) -> dict:
    """Fuse V1 multi-alpha oversold signal with candle pattern confirmation.

    Returns fused_rank which is the V1 rank boosted by candle confirmation,
    and candle_confirmed which is True where at least one pattern fires.
    """
    combo_rank = v1_sigs["combo_rank"]
    pattern_count = candle_sigs["pattern_count"]

    candle_confirmed = pattern_count > 0

    # Boosted rank: where V1 is oversold AND candle confirms,
    # push rank to top percentile
    fused_rank = combo_rank.copy()
    for si in range(NS):
        for di in range(ND):
            if np.isnan(combo_rank[si, di]):
                continue
            if combo_rank[si, di] >= 0.5 and candle_confirmed[si, di]:
                boost = 0.2 * pattern_count[si, di]
                fused_rank[si, di] = min(combo_rank[si, di] + boost, 1.0)

    return {
        "fused_rank": fused_rank,
        "candle_confirmed": candle_confirmed,
        "pattern_count": pattern_count,
    }


# ============================================================
# BACKTEST ENGINE (based on V5)
# ============================================================
def backtest_v7(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: list, syms: list,
    v1_sigs: dict, fused_sigs: dict, candle_sigs: dict,
    top_n: int = 1,
    min_rank: float = 0.7,
    atr_stop: float = 3.0,
    min_confidence: int = 3,
    use_ker_gate: bool = True,
    hold_days: int = 5,
    pyramid_ratio: float = 0.5,
    pyramid_day: int = 1,
    require_candle: bool = False,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> tuple:
    """Backtest V7 strategy with optional candle confirmation gate.

    Parameters
    ----------
    require_candle : bool
        If True, entry requires at least one bullish candle pattern
        in addition to V1 oversold + confidence + KER gate.
        If False, candle confirmation is used as a rank boost only.
    """
    use_rank = fused_sigs["fused_rank"]
    ker_regime = v1_sigs["ker_regime"]
    n_signals = v1_sigs["n_signals"]
    candle_confirmed = fused_sigs["candle_confirmed"]
    pattern_count = fused_sigs["pattern_count"]

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

        # Group positions by symbol
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

            # Stop check
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
                        "candle": int(pattern_count[si, edi]) if edi < ND else 0,
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
                        "candle": int(pattern_count[si, edi]) if edi < ND else 0,
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

        held = {p[0] for p in positions}
        if len(positions) >= top_n:
            continue

        # Entry selection
        candidates = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(use_rank[si, di]):
                continue
            if use_rank[si, di] < min_rank:
                continue
            if n_signals[si, di] < min_confidence:
                continue
            if use_ker_gate and ker_regime[si, di] < 0:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            # Optional: require candle confirmation as hard gate
            if require_candle and not candle_confirmed[si, di]:
                continue
            alloc = 1.0 / max(top_n, 1)
            candidates.append((use_rank[si, di], si, alloc))

        candidates.sort(key=lambda x: -x[0])
        for rank, si, alloc in candidates[:top_n]:
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
def analyze(trades: list, equity: float, max_dd: float, label: str = "") -> Optional[dict]:
    """Print trade analysis and return summary stats."""
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

    # Candle-confirmed trade performance
    candle_trades = [t for t in trades if t.get("candle", 0) > 0]
    no_candle_trades = [t for t in trades if t.get("candle", 0) == 0]

    print(f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} "
          f"stop:{n_stop} hold:{n_hold}) "
          f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
          f"Sh={sh:.2f} eq={equity:,.0f}")

    if candle_trades:
        c_wr = sum(1 for t in candle_trades if t["pnl_pct"] > 0) / len(candle_trades) * 100
        print(f"    Candle-confirmed: {len(candle_trades)}t WR={c_wr:.1f}%")
    if no_candle_trades:
        nc_wr = sum(1 for t in no_candle_trades if t["pnl_pct"] > 0) / len(no_candle_trades) * 100
        print(f"    No-candle:        {len(no_candle_trades)}t WR={nc_wr:.1f}%")

    yr = {}
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
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: list, syms: list,
    v1_sigs: dict, fused_sigs: dict, candle_sigs: dict,
    pyramid_ratio: float = 0.5,
    pyramid_day: int = 1,
    require_candle: bool = False,
) -> list:
    """Walk-forward validation year by year."""
    mode = "CANDLE-GATED" if require_candle else "CANDLE-BOOSTED"
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD V7 ({mode}, pyr={pyramid_ratio})")
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

        trades, _, _ = backtest_v7(
            C, O, H, L, NS, ND, dates, syms,
            v1_sigs, fused_sigs, candle_sigs,
            top_n=1, hold_days=5, atr_stop=3.0,
            min_confidence=3, use_ker_gate=True,
            pyramid_ratio=pyramid_ratio, pyramid_day=pyramid_day,
            require_candle=require_candle,
            start_di=test_start, end_di=test_end_idx + 1,
        )

        test_trades = [t for t in trades if dates[t["di"]].year == test_year]
        all_trades.extend(test_trades)

        if test_trades:
            n = len(test_trades)
            nw = sum(1 for t in test_trades if t["pnl_pct"] > 0)
            wr = nw / n * 100
            avg = np.mean([t["pnl_pct"] for t in test_trades])
            n_c = sum(1 for t in test_trades if t.get("candle", 0) > 0)
            print(f"  {test_year}: {n}t WR={wr:.1f}% avg={avg:+.2f}% candle={n_c}",
                  flush=True)
        else:
            print(f"  {test_year}: no trades", flush=True)

    if all_trades:
        nw = sum(1 for t in all_trades if t["pnl_pct"] > 0)
        wr = nw / len(all_trades) * 100
        avg = np.mean([t["pnl_pct"] for t in all_trades])
        cum = np.prod([1 + t["pnl_pct"] / 100 for t in all_trades]) - 1
        n_pyr = sum(1 for t in all_trades if t.get("pyr"))
        n_candle = sum(1 for t in all_trades if t.get("candle", 0) > 0)
        print(f"\n  WF TOTAL: {len(all_trades)}t (pyr:{n_pyr} candle:{n_candle}) "
              f"WR={wr:.1f}% avg={avg:+.2f}% cum={cum:+.1%}")
        return all_trades
    return []


# ============================================================
# PARAMETER SWEEP
# ============================================================
def sweep(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: list, syms: list,
    v1_sigs: dict, fused_sigs: dict, candle_sigs: dict,
    start_di: int,
) -> list:
    """Sweep over parameter combinations and rank by Sharpe."""
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    results = []
    for tn in [1, 2]:
        for ratio in [0, 0.3, 0.5]:
            for mc in [2, 3]:
                for req_candle in [False, True]:
                    trades, eq, dd = backtest_v7(
                        C, O, H, L, NS, ND, dates, syms,
                        v1_sigs, fused_sigs, candle_sigs,
                        top_n=tn, hold_days=5, atr_stop=3.0,
                        min_confidence=mc, use_ker_gate=True,
                        pyramid_ratio=ratio, pyramid_day=1,
                        require_candle=req_candle,
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
                        "tn": tn,
                        "ratio": ratio,
                        "mc": mc,
                        "candle": req_candle,
                        "n": len(trades),
                        "wr": wr,
                        "ann": ann,
                        "dd": dd,
                        "sharpe": sh_val,
                    })

    results.sort(key=lambda x: -x["sharpe"])
    print(f"\n{'TN':>3} {'Pyr':>4} {'MC':>3} {'Cdl':>4} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 60)
    for r in results[:25]:
        print(f"{r['tn']:>3} {r['ratio']:>4.1f} {r['mc']:>3} "
              f"{'Y' if r['candle'] else 'N':>4} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sharpe']:>5.2f}")

    return results


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V7: CANDLE PATTERN + DAILY SIGNAL FUSION")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start="2016-01-01")
    print(f"  {NS} sym, {ND} days, "
          f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    # Step 1: Compute candle patterns
    candle_sigs = compute_candle_patterns(C, O, H, L, NS, ND)

    # Step 2: Compute V1 multi-alpha signals
    v1_sigs = compute_v1_signals(C, O, H, L, V, OI, NS, ND)

    # Step 3: Fuse signals
    fused_sigs = compute_fused_signals(v1_sigs, candle_sigs, NS, ND)

    # Find 2019 start index
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # ==========================================================
    # COMPARISON: V1-only vs V1+Candle-boost vs V1+Candle-gated
    # ==========================================================
    print("\n" + "=" * 70)
    print("  COMPARISON: 2019-2026 (no pyramid, top_n=1)")
    print("=" * 70)

    configs = [
        (False, False, "V1-only (no candle)"),
        (True, False, "V1+Candle-BOOST"),
        (True, True, "V1+Candle-GATED"),
    ]

    for use_fused, req_candle, label in configs:
        if not use_fused:
            use_fused_sigs = {
                "fused_rank": v1_sigs["combo_rank"],
                "candle_confirmed": np.zeros((NS, ND), dtype=bool),
                "pattern_count": candle_sigs["pattern_count"],
            }
        else:
            use_fused_sigs = fused_sigs
        trades, eq, dd = backtest_v7(
            C, O, H, L, NS, ND, dates, syms,
            v1_sigs, use_fused_sigs, candle_sigs,
            top_n=1, hold_days=5, atr_stop=3.0,
            min_confidence=3, use_ker_gate=True,
            pyramid_ratio=0, pyramid_day=1,
            require_candle=req_candle,
            start_di=bt_2019,
        )
        analyze(trades, eq, dd, label)

    # ==========================================================
    # PYRAMID COMPARISON (2019-2026)
    # ==========================================================
    print("\n" + "=" * 70)
    print("  PYRAMID COMPARISON: 2019-2026 (pyr=0.5, top_n=1)")
    print("=" * 70)

    for req_candle in [False, True]:
        mode = "GATED" if req_candle else "BOOST"
        trades, eq, dd = backtest_v7(
            C, O, H, L, NS, ND, dates, syms,
            v1_sigs, fused_sigs, candle_sigs,
            top_n=1, hold_days=5, atr_stop=3.0,
            min_confidence=3, use_ker_gate=True,
            pyramid_ratio=0.5, pyramid_day=1,
            require_candle=req_candle,
            start_di=bt_2019,
        )
        analyze(trades, eq, dd, f"V7-{mode}-pyr0.5")

    # ==========================================================
    # WALK-FORWARD COMPARISON
    # ==========================================================
    print("\n" + "=" * 70)
    print("  WALK-FORWARD COMPARISON (pyramid=0.5)")
    print("=" * 70)

    for req_candle in [False, True]:
        walk_forward(
            C, O, H, L, NS, ND, dates, syms,
            v1_sigs, fused_sigs, candle_sigs,
            pyramid_ratio=0.5, pyramid_day=1,
            require_candle=req_candle,
        )

    # ==========================================================
    # PARAMETER SWEEP (2019-2026)
    # ==========================================================
    results = sweep(
        C, O, H, L, NS, ND, dates, syms,
        v1_sigs, fused_sigs, candle_sigs,
        start_di=bt_2019,
    )

    # ==========================================================
    # BEST CONFIG ON FULL 10-YEAR (2016-2026)
    # ==========================================================
    if results:
        print("\n" + "=" * 70)
        print("  BEST CONFIGS -- FULL 10-YEAR (2016-2026)")
        print("=" * 70)

        for r in results[:5]:
            trades, eq, dd = backtest_v7(
                C, O, H, L, NS, ND, dates, syms,
                v1_sigs, fused_sigs, candle_sigs,
                top_n=r["tn"], hold_days=5, atr_stop=3.0,
                min_confidence=r["mc"], use_ker_gate=True,
                pyramid_ratio=r["ratio"], pyramid_day=1,
                require_candle=r["candle"],
                start_di=60,
            )
            candle_mode = "GATED" if r["candle"] else "BOOST"
            label = f"tn={r['tn']} pyr={r['ratio']:.1f} mc={r['mc']} cdl={candle_mode}"
            print(f"\n  FULL {label}")
            analyze(trades, eq, dd, label)

    print(f"\n[V7] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
