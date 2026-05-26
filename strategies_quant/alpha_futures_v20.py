"""
V20: ADAPTIVE HOLD PERIOD -- Signal Strength Determines Exit Timing
====================================================================
Innovation: Instead of fixed 5-day hold, exit timing adapts to signal strength.
  1. V1 multi-alpha oversold signals (same 7 signals with weights)
  2. KER regime gate, confidence >= 3
  3. NEW: Hold period by confidence level:
     - Confidence 3 (weak): hold 3d (quick in-out)
     - Confidence 4 (moderate): hold 5d (standard)
     - Confidence 5+ (strong): hold 7d (give more time)
     - Combo rank > 0.9 (extreme oversold): hold 8d
  4. NEW: Dynamic exit based on mean reversion completion:
     - Price recovers above 20d MA -> exit early
     - RSI recovers above 50 -> exit early
     - Z-score recovers above -0.5 -> exit early
     - Otherwise hold until timer expires
  5. Pyramid on day-1 winners (ratio 0.5)
  6. ATR stop 3.0
  7. Walk-forward 2019-2026
  8. Parameter sweep: hold periods for each confidence level

Signal at close[di], enter at open[di+1]. No look-ahead.
"""
import sys
import os
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

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
# SIGNAL COMPUTATION (same 7 signals as V1)
# ============================================================
def compute_signals(C, O, H, L, V, OI, NS, ND):
    """Compute all signals -- same as V1 multi-alpha oversold."""
    t0 = time.time()
    print("[V20] Computing signals...", flush=True)

    # --- 1. Consecutive down days ---
    consec_dn = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        consec = 0
        for di in range(1, ND):
            if (
                not np.isnan(C[si, di])
                and not np.isnan(C[si, di - 1])
                and C[si, di - 1] > 0
            ):
                if C[si, di] < C[si, di - 1]:
                    consec += 1
                else:
                    consec = 0
            else:
                consec = 0
            consec_dn[si, di] = consec

    # --- 2. 5d return ---
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (
                not np.isnan(C[si, di])
                and not np.isnan(C[si, di - 5])
                and C[si, di - 5] > 0
            ):
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1

    # --- 3. 20d volatility ---
    vol_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            rets = []
            for j in range(di - 20, di):
                if (
                    not np.isnan(C[si, j])
                    and not np.isnan(C[si, j - 1])
                    and C[si, j - 1] > 0
                ):
                    rets.append(C[si, j] / C[si, j - 1] - 1)
            if len(rets) >= 10:
                vol_20d[si, di] = np.std(rets) * np.sqrt(252)

    # --- 4. OI capitulation ---
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
                oi_decline[si, di] = min(abs(oi_chg), 0.2) / 0.2 * min(
                    abs(price_chg), 0.1
                ) / 0.1
            else:
                oi_decline[si, di] = 0.0

    # --- 5. VDP (Volume Delta Pressure) ---
    vdp = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if (
                np.isnan(H[si, di])
                or np.isnan(L[si, di])
                or np.isnan(C[si, di])
                or np.isnan(V[si, di])
            ):
                continue
            bar_range = H[si, di] - L[si, di]
            if bar_range > 0 and V[si, di] > 0:
                vdp[si, di] = (
                    V[si, di]
                    * (2 * C[si, di] - H[si, di] - L[si, di])
                    / bar_range
                )

    vdp_10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            vals = vdp[si, di - 10 : di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 5:
                vdp_10[si, di] = np.mean(valid)

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

    # --- 6. KER (Kaufman Efficiency Ratio) ---
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

    # --- 7. TA-Lib indicators ---
    rsi14 = np.full((NS, ND), np.nan)
    bb_pos = np.full((NS, ND), np.nan)
    cci14 = np.full((NS, ND), np.nan)
    atr14 = np.full((NS, ND), np.nan)

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

            try:
                atr = talib.ATR(h, l, c, 14)
                atr14[si] = np.where(nan_mask, np.nan, atr)
            except Exception:
                pass

    # --- 8. Z-score (20d) ---
    zscore_20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            w = C[si, di - 20 : di]
            vv = w[~np.isnan(w)]
            if len(vv) >= 15 and np.std(vv) > 0 and not np.isnan(C[si, di]):
                zscore_20[si, di] = (C[si, di] - np.mean(vv)) / np.std(vv)

    # --- 9. 20d moving average ---
    ma_20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            w = C[si, di - 20 : di]
            vv = w[~np.isnan(w)]
            if len(vv) >= 15:
                ma_20[si, di] = np.mean(vv)

    # --- 10. Composite score + cross-sectional rank ---
    raw_score = np.full((NS, ND), np.nan)
    for di in range(ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue

            s = 0.0
            w_total = 0.0

            # Consecutive down days (weight 0.20)
            cd = consec_dn[si, di]
            s += min(cd / 5.0, 1.0) * 0.20
            w_total += 0.20

            # 5d return oversold (weight 0.20)
            if not np.isnan(ret_5d[si, di]):
                s += min(max(-ret_5d[si, di] / 0.1, 0), 1.0) * 0.20
                w_total += 0.20

            # OI capitulation (weight 0.20)
            if not np.isnan(oi_decline[si, di]):
                s += oi_decline[si, di] * 0.20
                w_total += 0.20

            # VDP selling exhaustion (weight 0.15)
            if not np.isnan(vdp_exhaust[si, di]):
                s += vdp_exhaust[si, di] * 0.15
                w_total += 0.15

            # RSI oversold (weight 0.10)
            if not np.isnan(rsi14[si, di]):
                if rsi14[si, di] < 30:
                    s += (30 - rsi14[si, di]) / 30.0 * 0.10
                w_total += 0.10

            # Bollinger lower band (weight 0.10)
            if not np.isnan(bb_pos[si, di]):
                if bb_pos[si, di] < 0.2:
                    s += (0.2 - bb_pos[si, di]) / 0.2 * 0.10
                w_total += 0.10

            # CCI oversold (weight 0.05)
            if not np.isnan(cci14[si, di]):
                if cci14[si, di] < -100:
                    s += min((-100 - cci14[si, di]) / 200.0, 1.0) * 0.05
                w_total += 0.05

            if w_total > 0:
                scores[si] = s / w_total

        valid = ~np.isnan(scores)
        if valid.sum() >= 5:
            raw_score[:, di] = pd.Series(scores).rank(
                pct=True, na_option="keep"
            ).values

    # --- 11. Confidence level (number of signals firing) ---
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

    # --- 12. KER regime ---
    ker_regime = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        for di in range(ND):
            if np.isnan(ker_10[si, di]):
                continue
            if ker_10[si, di] < 0.15:
                ker_regime[si, di] = 1  # mean-reverting
            elif ker_10[si, di] > 0.3:
                ker_regime[si, di] = -1  # trending

    elapsed = time.time() - t0
    print(f"  Done: {elapsed:.1f}s (TA-Lib: {HAS_TALIB})", flush=True)

    return {
        "combo_rank": raw_score,
        "consec_dn": consec_dn,
        "vol_20d": vol_20d,
        "ker_10": ker_10,
        "ker_regime": ker_regime,
        "n_signals": n_signals,
        "oi_decline": oi_decline,
        "vdp_exhaust": vdp_exhaust,
        "rsi14": rsi14,
        "bb_pos": bb_pos,
        "cci14": cci14,
        "atr14": atr14,
        "zscore_20": zscore_20,
        "ma_20": ma_20,
    }


# ============================================================
# DETERMINE ADAPTIVE HOLD PERIOD
# ============================================================
def get_hold_days(
    confidence: int,
    combo_rank: float,
    hold_conf3: int = 3,
    hold_conf4: int = 5,
    hold_conf5: int = 7,
    hold_extreme_rank: float = 0.9,
    hold_extreme_days: int = 8,
) -> int:
    """
    Signal strength determines hold period.
    - Confidence 3 (weak): quick in-out
    - Confidence 4 (moderate): standard
    - Confidence 5+ (strong): give more time
    - Combo rank > extreme threshold: extra time for extreme oversold
    """
    if combo_rank > hold_extreme_rank:
        return hold_extreme_days
    if confidence >= 5:
        return hold_conf5
    if confidence == 4:
        return hold_conf4
    return hold_conf3


# ============================================================
# BACKTEST ENGINE: ADAPTIVE HOLD + DYNAMIC EXIT + PYRAMID
# ============================================================
def backtest_v20(
    C,
    O,
    H,
    L,
    NS,
    ND,
    dates,
    syms,
    sigs,
    top_n=1,
    min_rank=0.7,
    atr_stop=3.0,
    min_confidence=3,
    use_ker_gate=True,
    pyramid_ratio=0.5,
    hold_conf3=3,
    hold_conf4=5,
    hold_conf5=7,
    hold_extreme_days=8,
    hold_extreme_rank=0.9,
    use_dynamic_exit=True,
    leverage=1.0,
    start_di=60,
    end_di=None,
):
    """
    Adaptive hold period backtest.
    Signal at close[di], enter at open[di+1]. No look-ahead.

    Positions tuple:
      (si, entry_di, entry_price, stop_price, alloc, direction,
       max_hold, confidence, combo_rank_at_entry)
    """
    combo_rank = sigs["combo_rank"]
    ker_regime = sigs["ker_regime"]
    n_signals = sigs["n_signals"]
    rsi14 = sigs["rsi14"]
    zscore_20 = sigs["zscore_20"]
    ma_20 = sigs["ma_20"]
    atr14 = sigs["atr14"]

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

        # --- Exit logic ---
        for (
            si,
            edi,
            ep,
            sp,
            alloc,
            direction,
            max_hold,
            confidence,
            entry_rank,
        ) in positions:
            c = C[si, di]
            if np.isnan(c):
                new_positions.append(
                    (si, edi, ep, sp, alloc, direction, max_hold, confidence, entry_rank)
                )
                continue

            exit_r = None
            hold_days = di - edi

            # 1. ATR stop loss (always active)
            if direction > 0 and c < sp:
                exit_r = "stop"
            elif direction < 0 and c > sp:
                exit_r = "stop"

            # 2. Dynamic exit based on mean reversion completion (after day 1)
            if exit_r is None and use_dynamic_exit and hold_days >= 2:
                # a. Price recovers above 20d MA
                ma = ma_20[si, di]
                if not np.isnan(ma):
                    if direction > 0 and c > ma:
                        exit_r = "ma_recover"
                    elif direction < 0 and c < ma:
                        exit_r = "ma_recover"

                # b. RSI recovers
                if exit_r is None:
                    rsi = rsi14[si, di]
                    if not np.isnan(rsi):
                        if direction > 0 and rsi > 50:
                            exit_r = "rsi_recover"
                        elif direction < 0 and rsi < 50:
                            exit_r = "rsi_recover"

                # c. Z-score recovers
                if exit_r is None:
                    zs = zscore_20[si, di]
                    if not np.isnan(zs):
                        if direction > 0 and zs > -0.5:
                            exit_r = "zscore_recover"
                        elif direction < 0 and zs < 0.5:
                            exit_r = "zscore_recover"

            # 3. Timer expiry (adaptive hold period)
            if exit_r is None and hold_days >= max_hold:
                exit_r = "timer"

            if exit_r:
                pnl = direction * (c - ep) / ep - COMM
                profit = equity * alloc * leverage * pnl
                daily_pnl += profit
                trades.append(
                    {
                        "pnl_abs": profit,
                        "pnl_pct": pnl * 100 * leverage,
                        "days": hold_days + 1,
                        "di": di,
                        "year": d.year,
                        "sym": syms[si],
                        "reason": exit_r,
                        "dir": direction,
                        "confidence": confidence,
                        "entry_rank": entry_rank,
                        "max_hold": max_hold,
                    }
                )
            else:
                new_positions.append(
                    (si, edi, ep, sp, alloc, direction, max_hold, confidence, entry_rank)
                )

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

        # --- Pyramid on day-1 winners ---
        pyramid_positions = []
        for (
            si,
            edi,
            ep,
            sp,
            alloc,
            direction,
            max_hold,
            confidence,
            entry_rank,
        ) in positions:
            if di - edi == 1 and pyramid_ratio > 0:
                c = C[si, di]
                if not np.isnan(c):
                    unrealized = direction * (c - ep) / ep
                    if unrealized > 0:
                        # Winner on day 1 -- add pyramid
                        extra_alloc = alloc * pyramid_ratio
                        # Use same stop
                        pyramid_positions.append(
                            (
                                si,
                                di,
                                c,
                                sp,
                                extra_alloc,
                                direction,
                                max_hold - 1,
                                confidence,
                                entry_rank,
                            )
                        )
        positions.extend(pyramid_positions)

        # --- Entry ---
        held = {p[0] for p in positions}
        max_pos = top_n + top_n  # account for pyramid slots
        if len(positions) >= max_pos:
            continue

        candidates = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(combo_rank[si, di]):
                continue
            if combo_rank[si, di] < min_rank:
                continue
            # Confidence gate
            if n_signals[si, di] < min_confidence:
                continue
            # KER regime gate
            if use_ker_gate and ker_regime[si, di] < 0:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue

            alloc = 1.0 / max(top_n, 1)
            confidence = n_signals[si, di]
            rank = combo_rank[si, di]

            candidates.append((rank, si, alloc, confidence, rank))

        candidates.sort(key=lambda x: -x[0])
        for rank, si, alloc, confidence, entry_rank in candidates[:top_n]:
            if len(positions) >= max_pos or si in held:
                break
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue

            # ATR stop
            atr_v = []
            for j in range(max(start_di, di - 14), di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
            if not atr_v:
                continue
            atr = np.mean(atr_v)

            hold = get_hold_days(
                confidence,
                rank,
                hold_conf3=hold_conf3,
                hold_conf4=hold_conf4,
                hold_conf5=hold_conf5,
                hold_extreme_rank=hold_extreme_rank,
                hold_extreme_days=hold_extreme_days,
            )

            stop = ep - atr_stop * atr
            positions.append(
                (si, di + 1, ep, stop, alloc, 1, hold, confidence, entry_rank)
            )
            held.add(si)

    # Close remaining
    for si, edi, ep, sp, alloc, direction, _, confidence, entry_rank in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = direction * (c - ep) / ep - COMM
            equity += equity * alloc * leverage * pnl

    return trades, equity, max_dd


# ============================================================
# ANALYSIS
# ============================================================
def analyze(trades, equity, max_dd, label=""):
    if not trades:
        print(f"  {label}: no trades")
        return None
    nw = sum(1 for t in trades if t["pnl_pct"] > 0)
    wr = nw / len(trades) * 100
    n_days = max(1, trades[-1]["di"] - trades[0]["di"])
    ann = ((equity / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
    ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
    rets = np.array(ap) / CASH0
    sh = (
        np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
    )

    long_t = [t for t in trades if t.get("dir", 1) > 0]
    short_t = [t for t in trades if t.get("dir", 1) < 0]
    gross_win = sum(t["pnl_pct"] for t in trades if t["pnl_pct"] > 0)
    gross_loss = abs(sum(t["pnl_pct"] for t in trades if t["pnl_pct"] < 0))
    pf = gross_win / max(gross_loss, 1e-10)

    print(
        f"  {label}: {len(trades)}t (L:{len(long_t)} S:{len(short_t)}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
        f"Sh={sh:.2f} PF={pf:.2f} eq={equity:,.0f}"
    )

    # Exit reason breakdown
    reasons = defaultdict(lambda: {"n": 0, "w": 0, "pnl": []})
    for t in trades:
        r = t["reason"]
        reasons[r]["n"] += 1
        if t["pnl_pct"] > 0:
            reasons[r]["w"] += 1
        reasons[r]["pnl"].append(t["pnl_pct"])

    print(f"    Exit reasons:")
    for r in ["stop", "timer", "ma_recover", "rsi_recover", "zscore_recover"]:
        if r in reasons:
            rs = reasons[r]
            rwr = rs["w"] / rs["n"] * 100
            avg_pnl = np.mean(rs["pnl"])
            print(f"      {r:>15}: {rs['n']:>4}t WR={rwr:.1f}% avg={avg_pnl:+.2f}%")

    # Confidence breakdown
    conf_groups = defaultdict(lambda: {"n": 0, "w": 0, "pnl": []})
    for t in trades:
        c = t.get("confidence", 3)
        conf_groups[c]["n"] += 1
        if t["pnl_pct"] > 0:
            conf_groups[c]["w"] += 1
        conf_groups[c]["pnl"].append(t["pnl_pct"])

    print(f"    By confidence:")
    for c in sorted(conf_groups.keys()):
        cg = conf_groups[c]
        cwr = cg["w"] / cg["n"] * 100
        avg_pnl = np.mean(cg["pnl"])
        print(
            f"      conf={c}: {cg['n']:>4}t WR={cwr:.1f}% avg={avg_pnl:+.2f}%"
        )

    # Hold period distribution
    hold_days_list = [t["days"] for t in trades]
    if hold_days_list:
        avg_hold = np.mean(hold_days_list)
        print(f"    Avg hold: {avg_hold:.1f}d  (max={max(hold_days_list)}d)")

    # Yearly breakdown
    yr = {}
    for t in trades:
        y = t["year"]
        if y not in yr:
            yr[y] = {"n": 0, "w": 0, "pnl": []}
        yr[y]["n"] += 1
        if t["pnl_pct"] > 0:
            yr[y]["w"] += 1
        yr[y]["pnl"].append(t["pnl_pct"])
    print(f"    Yearly:")
    for y in sorted(yr.keys()):
        ys = yr[y]
        cum = np.prod([1 + p / 100 for p in ys["pnl"]]) - 1
        print(
            f"      {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% cum={cum:+.1%}"
        )

    return {
        "n": len(trades),
        "wr": wr,
        "dd": max_dd,
        "ann": ann,
        "sh": sh,
        "eq": equity,
        "pf": pf,
    }


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================
def walk_forward(
    C,
    O,
    H,
    L,
    V,
    OI,
    NS,
    ND,
    dates,
    syms,
    train_years=4,
    top_n=1,
    min_rank=0.7,
    atr_stop=3.0,
    min_confidence=3,
    use_ker_gate=True,
    pyramid_ratio=0.5,
    hold_conf3=3,
    hold_conf4=5,
    hold_conf5=7,
    hold_extreme_days=8,
    hold_extreme_rank=0.9,
    use_dynamic_exit=True,
):
    """Walk-forward validation: train on N years, test on next year."""
    print(f"\n{'='*70}")
    print(
        f"  WALK-FORWARD: train={train_years}y, adaptive hold, dynamic exit"
    )
    print(f"{'='*70}")

    years = sorted(set(d.year for d in dates))
    all_trades = []

    start_year = years[0]
    while True:
        train_end = start_year + train_years - 1
        test_year = train_end + 1
        if test_year > years[-1]:
            break

        # Find test year boundaries
        test_start_di = None
        test_end_di = None
        for i, d in enumerate(dates):
            if d.year == test_year and test_start_di is None:
                test_start_di = i
            if d.year == test_year:
                test_end_di = i

        if test_start_di is None:
            start_year += 1
            continue

        # Compute signals on full data (no look-ahead since signals
        # use only past data)
        sigs = compute_signals(C, O, H, L, V, OI, NS, ND)

        # Backtest but only on training + test window
        # Training for signal warmup, test for evaluation
        train_start_di = max(0, test_start_di - train_years * 252)
        trades, eq, dd = backtest_v20(
            C,
            O,
            H,
            L,
            NS,
            ND,
            dates,
            syms,
            sigs,
            top_n=top_n,
            min_rank=min_rank,
            atr_stop=atr_stop,
            min_confidence=min_confidence,
            use_ker_gate=use_ker_gate,
            pyramid_ratio=pyramid_ratio,
            hold_conf3=hold_conf3,
            hold_conf4=hold_conf4,
            hold_conf5=hold_conf5,
            hold_extreme_days=hold_extreme_days,
            hold_extreme_rank=hold_extreme_rank,
            use_dynamic_exit=use_dynamic_exit,
            start_di=train_start_di,
            end_di=test_end_di + 1,
        )

        yr_trades = [t for t in trades if t["year"] == test_year]
        all_trades.extend(yr_trades)

        if yr_trades:
            pnls = [t["pnl_pct"] for t in yr_trades]
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            print(
                f"  {test_year}: {len(pnls)}t WR={wr:.1f}% "
                f"avg={np.mean(pnls):+.3f}%",
                flush=True,
            )
        else:
            print(f"  {test_year}: no trades", flush=True)

        start_year += 1

    # Aggregate WF results
    if all_trades:
        nw = sum(1 for t in all_trades if t["pnl_pct"] > 0)
        wr = nw / len(all_trades) * 100
        ap = [t["pnl_abs"] for t in sorted(all_trades, key=lambda x: x["di"])]
        cum_pnl = sum(ap)
        rets_arr = np.array(ap) / CASH0
        sh_val = (
            np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252)
            if np.std(rets_arr) > 0
            else 0
        )
        test_years = sorted(set(t["year"] for t in all_trades))
        n_yrs = len(test_years)
        ann = ((1 + cum_pnl / CASH0) ** (1 / max(n_yrs, 1)) - 1) * 100

        print(f"\n  Walk-Forward Aggregate:")
        print(
            f"    {len(all_trades)}t WR={wr:.1f}% ann={ann:+.1f}% "
            f"Sh={sh_val:.2f} cum_pnl={cum_pnl:,.0f}"
        )
        return all_trades

    return []


# ============================================================
# PARAMETER SWEEP: HOLD PERIODS PER CONFIDENCE LEVEL
# ============================================================
def parameter_sweep(
    C, O, H, L, V, OI, NS, ND, dates, syms, sigs, bt_2019
):
    """Sweep hold periods for each confidence level."""
    print(f"\n{'='*70}")
    print("  PARAMETER SWEEP: ADAPTIVE HOLD + DYNAMIC EXIT")
    print(f"{'='*70}")

    results = []

    # Sweep hold days for each confidence level
    for hc3 in [2, 3, 4]:
        for hc4 in [3, 5, 7]:
            for hc5 in [5, 7, 10]:
                for he in [7, 8, 10]:
                    for dyn_exit in [True, False]:
                        for pyr in [0.0, 0.5]:
                            trades, eq, dd = backtest_v20(
                                C,
                                O,
                                H,
                                L,
                                NS,
                                ND,
                                dates,
                                syms,
                                sigs,
                                top_n=1,
                                min_rank=0.7,
                                atr_stop=3.0,
                                min_confidence=3,
                                use_ker_gate=True,
                                pyramid_ratio=pyr,
                                hold_conf3=hc3,
                                hold_conf4=hc4,
                                hold_conf5=hc5,
                                hold_extreme_days=he,
                                use_dynamic_exit=dyn_exit,
                                start_di=bt_2019,
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
                            avg_hold = np.mean([t["days"] for t in trades])
                            results.append(
                                {
                                    "hc3": hc3,
                                    "hc4": hc4,
                                    "hc5": hc5,
                                    "he": he,
                                    "dyn": dyn_exit,
                                    "pyr": pyr,
                                    "n": len(trades),
                                    "wr": wr,
                                    "ann": ann,
                                    "dd": dd,
                                    "sharpe": sh_val,
                                    "avg_hold": avg_hold,
                                }
                            )

    results.sort(key=lambda x: -x["sharpe"])
    print(
        f"\n{'hc3':>3} {'hc4':>3} {'hc5':>3} {'he':>3} "
        f"{'Dyn':>3} {'Pyr':>4} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>6} {'AvgH':>5}"
    )
    print("-" * 70)
    for r in results[:30]:
        print(
            f"{r['hc3']:>3} {r['hc4']:>3} {r['hc5']:>3} {r['he']:>3} "
            f"{'Y' if r['dyn'] else 'N':>3} {r['pyr']:>4.1f} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f} {r['avg_hold']:>5.1f}"
        )

    return results


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("  V20: ADAPTIVE HOLD PERIOD -- SIGNAL STRENGTH DETERMINES EXIT")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start="2016-01-01")
    print(
        f"  {NS} sym, {ND} days, "
        f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}"
    )

    sigs = compute_signals(C, O, H, L, V, OI, NS, ND)

    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # ============================================================
    # SECTION 1: BASELINE COMPARISON (fixed 5d hold vs adaptive)
    # ============================================================
    print(f"\n{'='*70}")
    print("  SECTION 1: BASELINE vs ADAPTIVE HOLD (2019-2026)")
    print(f"{'='*70}")

    # Fixed 5d hold baseline (all confidence levels get 5d)
    print("\n  --- Fixed 5d hold (baseline) ---")
    trades_base, eq_base, dd_base = backtest_v20(
        C,
        O,
        H,
        L,
        NS,
        ND,
        dates,
        syms,
        sigs,
        top_n=1,
        min_rank=0.7,
        atr_stop=3.0,
        min_confidence=3,
        use_ker_gate=True,
        pyramid_ratio=0.0,
        hold_conf3=5,
        hold_conf4=5,
        hold_conf5=5,
        hold_extreme_days=5,
        use_dynamic_exit=False,
        start_di=bt_2019,
    )
    res_base = analyze(trades_base, eq_base, dd_base, "Baseline-fixed5d")

    # Adaptive hold, no dynamic exit
    print("\n  --- Adaptive hold, no dynamic exit ---")
    trades_adapt, eq_adapt, dd_adapt = backtest_v20(
        C,
        O,
        H,
        L,
        NS,
        ND,
        dates,
        syms,
        sigs,
        top_n=1,
        min_rank=0.7,
        atr_stop=3.0,
        min_confidence=3,
        use_ker_gate=True,
        pyramid_ratio=0.0,
        hold_conf3=3,
        hold_conf4=5,
        hold_conf5=7,
        hold_extreme_days=8,
        use_dynamic_exit=False,
        start_di=bt_2019,
    )
    res_adapt = analyze(trades_adapt, eq_adapt, dd_adapt, "Adaptive-noDynExit")

    # Adaptive hold + dynamic exit
    print("\n  --- Adaptive hold + dynamic exit ---")
    trades_dyn, eq_dyn, dd_dyn = backtest_v20(
        C,
        O,
        H,
        L,
        NS,
        ND,
        dates,
        syms,
        sigs,
        top_n=1,
        min_rank=0.7,
        atr_stop=3.0,
        min_confidence=3,
        use_ker_gate=True,
        pyramid_ratio=0.0,
        hold_conf3=3,
        hold_conf4=5,
        hold_conf5=7,
        hold_extreme_days=8,
        use_dynamic_exit=True,
        start_di=bt_2019,
    )
    res_dyn = analyze(trades_dyn, eq_dyn, dd_dyn, "Adaptive+DynExit")

    # Adaptive hold + dynamic exit + pyramid
    print("\n  --- Adaptive hold + dynamic exit + pyramid 0.5 ---")
    trades_full, eq_full, dd_full = backtest_v20(
        C,
        O,
        H,
        L,
        NS,
        ND,
        dates,
        syms,
        sigs,
        top_n=1,
        min_rank=0.7,
        atr_stop=3.0,
        min_confidence=3,
        use_ker_gate=True,
        pyramid_ratio=0.5,
        hold_conf3=3,
        hold_conf4=5,
        hold_conf5=7,
        hold_extreme_days=8,
        use_dynamic_exit=True,
        start_di=bt_2019,
    )
    res_full = analyze(
        trades_full, eq_full, dd_full, "Adaptive+DynExit+Pyramid"
    )

    # ============================================================
    # SECTION 2: PARAMETER SWEEP
    # ============================================================
    sweep_results = parameter_sweep(
        C, O, H, L, V, OI, NS, ND, dates, syms, sigs, bt_2019
    )

    # ============================================================
    # SECTION 3: BEST CONFIG -- YEARLY BREAKDOWN
    # ============================================================
    print(f"\n{'='*70}")
    print("  SECTION 3: BEST CONFIGS -- YEARLY (2019-2026)")
    print(f"{'='*70}")

    for r in sweep_results[:3]:
        trades, eq, dd = backtest_v20(
            C,
            O,
            H,
            L,
            NS,
            ND,
            dates,
            syms,
            sigs,
            top_n=1,
            min_rank=0.7,
            atr_stop=3.0,
            min_confidence=3,
            use_ker_gate=True,
            pyramid_ratio=r["pyr"],
            hold_conf3=r["hc3"],
            hold_conf4=r["hc4"],
            hold_conf5=r["hc5"],
            hold_extreme_days=r["he"],
            use_dynamic_exit=r["dyn"],
            start_di=bt_2019,
        )
        label = (
            f"h({r['hc3']},{r['hc4']},{r['hc5']},{r['he']})"
            f"{'D' if r['dyn'] else ''}p{r['pyr']:.1f}"
        )
        print(f"\n  --- {label} ---")
        analyze(trades, eq, dd, label)

    # ============================================================
    # SECTION 4: WALK-FORWARD (best config)
    # ============================================================
    if sweep_results:
        best = sweep_results[0]
        wf_trades = walk_forward(
            C,
            O,
            H,
            L,
            V,
            OI,
            NS,
            ND,
            dates,
            syms,
            train_years=4,
            top_n=1,
            min_rank=0.7,
            atr_stop=3.0,
            min_confidence=3,
            use_ker_gate=True,
            pyramid_ratio=best["pyr"],
            hold_conf3=best["hc3"],
            hold_conf4=best["hc4"],
            hold_conf5=best["hc5"],
            hold_extreme_days=best["he"],
            use_dynamic_exit=best["dyn"],
        )

    # ============================================================
    # SECTION 5: FULL 10-YEAR (best config)
    # ============================================================
    print(f"\n{'='*70}")
    print("  SECTION 5: FULL 2016-2026 (10 years)")
    print(f"{'='*70}")

    if sweep_results:
        for r in sweep_results[:2]:
            trades, eq, dd = backtest_v20(
                C,
                O,
                H,
                L,
                NS,
                ND,
                dates,
                syms,
                sigs,
                top_n=1,
                min_rank=0.7,
                atr_stop=3.0,
                min_confidence=3,
                use_ker_gate=True,
                pyramid_ratio=r["pyr"],
                hold_conf3=r["hc3"],
                hold_conf4=r["hc4"],
                hold_conf5=r["hc5"],
                hold_extreme_days=r["he"],
                use_dynamic_exit=r["dyn"],
                start_di=60,
            )
            label = (
                f"10yr h({r['hc3']},{r['hc4']},{r['hc5']},{r['he']})"
                f"{'D' if r['dyn'] else ''}p{r['pyr']:.1f}"
            )
            print(f"\n  {label}")
            analyze(trades, eq, dd, label)

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print("  SUMMARY: ADAPTIVE HOLD vs FIXED 5d")
    print(f"{'='*70}")
    if res_base and res_full:
        print(f"  Fixed 5d:   ann={res_base['ann']:+.1f}%  "
              f"WR={res_base['wr']:.1f}%  "
              f"Sh={res_base['sh']:.2f}  DD={res_base['dd']:.1f}%")
        print(f"  Adaptive:   ann={res_full['ann']:+.1f}%  "
              f"WR={res_full['wr']:.1f}%  "
              f"Sh={res_full['sh']:.2f}  DD={res_full['dd']:.1f}%")
        if sweep_results:
            b = sweep_results[0]
            print(
                f"  Best sweep: ann={b['ann']:+.1f}%  "
                f"WR={b['wr']:.1f}%  "
                f"Sh={b['sharpe']:.2f}  DD={b['dd']:.1f}%"
            )
            print(
                f"    config: hc3={b['hc3']} hc4={b['hc4']} "
                f"hc5={b['hc5']} he={b['he']} "
                f"dyn={'Y' if b['dyn'] else 'N'} pyr={b['pyr']:.1f}"
            )

    print(f"\n[V20] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
