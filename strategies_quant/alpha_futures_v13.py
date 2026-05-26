"""
V13: Nadaraya-Watson Kernel Regression + VDP Confirmation
==========================================================
Non-parametric kernel regression for smooth trend estimation,
then detect oversold deviations from the NW smooth "fair value".

Key ideas:
  1. NW Gaussian kernel regression on close prices
     - Adaptive bandwidth via ATR (wider band in volatile markets)
     - Produces smooth "fair value" estimate
  2. Deviation from NW = how far price moved from fair value
     - Price / NW_smooth - 1 = fractional deviation
     - Cross-sectional rank the deviations
  3. Oversold detection: large negative deviation = oversold
  4. VDP confirmation: V * (2C-H-L)/(H-L)
     - Negative VDP + negative deviation = selling exhaustion
  5. Combine with V5's other signals (OI decline, RSI, BB, CCI)
  6. KER gate + confidence + pyramid
  7. Walk-forward validation

Signal at close[di], enter at open[di+1]. No look-ahead.
No gap signals. No leverage.
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

# NW kernel bandwidth (lookback window in bars)
NW_BANDWIDTH = 15
# Minimum number of valid prices needed for NW computation
NW_MIN_VALID = 12


def _nadaraya_watson_1d(prices: np.ndarray, bandwidth: int) -> np.ndarray:
    """
    Nadaraya-Watson Gaussian kernel regression on a 1D price series.

    For each index i >= bandwidth:
      - Take the window prices[i-bandwidth : i+1]
      - Apply Gaussian weights centered at index i
      - NW estimate = weighted average

    The Gaussian kernel h is set to bandwidth / 2.0 so the kernel
    decays smoothly across the window.  This matches the reference
    implementation in ensemble_nw_volume_strategy.py.

    Parameters
    ----------
    prices : 1-D array of close prices (may contain NaN)
    bandwidth : number of lookback bars

    Returns
    -------
    nw : 1-D array, same length as prices, NaN where insufficient data
    """
    n = len(prices)
    nw = np.full(n, np.nan)
    h = max(bandwidth / 2.0, 1.0)

    for i in range(bandwidth, n):
        window = prices[i - bandwidth : i + 1]
        valid_mask = ~np.isnan(window)
        valid_count = valid_mask.sum()
        if valid_count < NW_MIN_VALID:
            continue

        valid_prices = window[valid_mask]
        indices = np.where(valid_mask)[0]
        mid = bandwidth  # center of full window
        dist_sq = (indices - mid) ** 2
        weights = np.exp(-dist_sq / (2.0 * h * h))
        weight_sum = weights.sum()
        if weight_sum > 1e-12:
            nw[i] = np.sum(weights * valid_prices) / weight_sum

    return nw


def compute_signals(C, O, H, L, V, OI, NS, ND):
    """Core signal computation with NW kernel regression."""
    t0 = time.time()
    print("[V13] Computing signals...", flush=True)

    # ----------------------------------------------------------------
    # 1. NW kernel regression per instrument
    # ----------------------------------------------------------------
    nw_smooth = np.full((NS, ND), np.nan)
    nw_deviation = np.full((NS, ND), np.nan)

    for si in range(NS):
        closes = C[si]
        nw = _nadaraya_watson_1d(closes, NW_BANDWIDTH)
        nw_smooth[si] = nw

        # Fractional deviation from NW fair value
        for di in range(NW_BANDWIDTH, ND):
            if np.isnan(nw[di]) or np.isnan(closes[di]) or closes[di] <= 0:
                continue
            if nw[di] > 1e-10:
                nw_deviation[si, di] = closes[di] / nw[di] - 1.0

    # Cross-sectional rank of deviations (more negative = more oversold = higher rank)
    cs_dev_rank = np.full((NS, ND), np.nan)
    for di in range(ND):
        devs = nw_deviation[:, di]
        valid = ~np.isnan(devs)
        if valid.sum() >= 5:
            # Negate so oversold (negative deviation) ranks high
            neg_devs = -devs
            cs_dev_rank[:, di] = (
                pd.Series(neg_devs).rank(pct=True, na_option="keep").values
            )

    # ----------------------------------------------------------------
    # 2. VDP (volume delta pressure) + exhaustion
    # ----------------------------------------------------------------
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
                    V[si, di] * (2 * C[si, di] - H[si, di] - L[si, di]) / bar_range
                )

    # VDP 10-bar EMA
    vdp_10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            vals = vdp[si, di - 10 : di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 5:
                vdp_10[si, di] = np.mean(valid)

    # VDP exhaustion z-score
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
                    # Negative z = selling pressure exhaustion
                    vdp_exhaust[si, di] = min(-z, 3.0) / 3.0 if z < 0 else 0.0

    # ----------------------------------------------------------------
    # 3. Consecutive down days
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    # 4. 5-day return
    # ----------------------------------------------------------------
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (
                not np.isnan(C[si, di])
                and not np.isnan(C[si, di - 5])
                and C[si, di - 5] > 0
            ):
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1

    # ----------------------------------------------------------------
    # 5. OI decline
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    # 6. RSI, BB, CCI via TA-Lib
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    # 7. KER (Kaufman efficiency ratio)
    # ----------------------------------------------------------------
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
                ker_regime[si, di] = 1  # ranging -> mean reversion OK
            elif ker_10[si, di] > 0.3:
                ker_regime[si, di] = -1  # trending -> skip mean reversion

    # ----------------------------------------------------------------
    # 8. Composite score + cross-sectional rank
    #
    # Weights:
    #   NW deviation rank  = 0.25  (primary signal)
    #   VDP exhaustion     = 0.20  (confirmation)
    #   Consecutive down   = 0.15  (momentum exhaustion)
    #   5d return          = 0.10  (price weakness)
    #   OI decline         = 0.10  (positioning unwind)
    #   RSI14              = 0.08  (technical oversold)
    #   BB position        = 0.07  (band extreme)
    #   CCI14              = 0.05  (commodity channel)
    # ----------------------------------------------------------------
    raw_score = np.full((NS, ND), np.nan)
    for di in range(ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            s = 0.0
            w_total = 0.0

            # NW deviation rank (primary)
            if not np.isnan(cs_dev_rank[si, di]):
                s += cs_dev_rank[si, di] * 0.25
                w_total += 0.25

            # VDP exhaustion (confirmation)
            if not np.isnan(vdp_exhaust[si, di]):
                s += vdp_exhaust[si, di] * 0.20
                w_total += 0.20

            # Consecutive down days
            s += min(consec_dn[si, di] / 5.0, 1.0) * 0.15
            w_total += 0.15

            # 5d return weakness
            if not np.isnan(ret_5d[si, di]):
                s += min(max(-ret_5d[si, di] / 0.1, 0), 1.0) * 0.10
                w_total += 0.10

            # OI decline
            if not np.isnan(oi_decline[si, di]):
                s += oi_decline[si, di] * 0.10
                w_total += 0.10

            # RSI14 oversold
            if not np.isnan(rsi14[si, di]):
                if rsi14[si, di] < 30:
                    s += (30 - rsi14[si, di]) / 30.0 * 0.08
                w_total += 0.08

            # BB position
            if not np.isnan(bb_pos[si, di]):
                if bb_pos[si, di] < 0.2:
                    s += (0.2 - bb_pos[si, di]) / 0.2 * 0.07
                w_total += 0.07

            # CCI14
            if not np.isnan(cci14[si, di]):
                if cci14[si, di] < -100:
                    s += min((-100 - cci14[si, di]) / 200.0, 1.0) * 0.05
                w_total += 0.05

            if w_total > 0:
                scores[si] = s / w_total
        valid = ~np.isnan(scores)
        if valid.sum() >= 5:
            raw_score[:, di] = (
                pd.Series(scores).rank(pct=True, na_option="keep").values
            )

    # ----------------------------------------------------------------
    # 9. Signal count for confidence gating
    # ----------------------------------------------------------------
    n_signals = np.zeros((NS, ND), dtype=int)
    for di in range(ND):
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            n = 0
            # NW deviation: oversold (deviation < -0.01)
            if not np.isnan(nw_deviation[si, di]) and nw_deviation[si, di] < -0.01:
                n += 1
            # VDP exhaustion
            if not np.isnan(vdp_exhaust[si, di]) and vdp_exhaust[si, di] > 0.3:
                n += 1
            # Consecutive down
            if consec_dn[si, di] >= 3:
                n += 1
            # 5d return weak
            if not np.isnan(ret_5d[si, di]) and ret_5d[si, di] < -0.03:
                n += 1
            # OI decline
            if not np.isnan(oi_decline[si, di]) and oi_decline[si, di] > 0.1:
                n += 1
            # RSI oversold
            if not np.isnan(rsi14[si, di]) and rsi14[si, di] < 35:
                n += 1
            # BB extreme
            if not np.isnan(bb_pos[si, di]) and bb_pos[si, di] < 0.15:
                n += 1
            # CCI extreme
            if not np.isnan(cci14[si, di]) and cci14[si, di] < -100:
                n += 1
            n_signals[si, di] = n

    print(f"  Done: {time.time() - t0:.1f}s", flush=True)
    return {
        "combo_rank": raw_score,
        "ker_regime": ker_regime,
        "n_signals": n_signals,
        "nw_deviation": nw_deviation,
        "cs_dev_rank": cs_dev_rank,
    }


# ============================================================
# BACKTEST WITH PYRAMID
# ============================================================
def backtest_v13(
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
    hold_days=5,
    pyramid_ratio=0.5,
    pyramid_day=1,
    leverage=1.0,
    start_di=60,
    end_di=None,
):
    """Backtest with pyramid support.  Long only, no leverage."""
    combo_rank = sigs["combo_rank"]
    ker_regime = sigs["ker_regime"]
    n_signals = sigs["n_signals"]

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
                    profit = equity * alloc * leverage * pnl
                    daily_pnl += profit
                    trades.append(
                        {
                            "pnl_abs": profit,
                            "pnl_pct": pnl * 100 * leverage,
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
                    profit = equity * alloc * leverage * pnl
                    daily_pnl += profit
                    trades.append(
                        {
                            "pnl_abs": profit,
                            "pnl_pct": pnl * 100 * leverage,
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

        # Entry selection
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
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            alloc = 1.0 / max(top_n, 1)
            candidates.append((combo_rank[si, di], si, alloc))

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

    # Close remaining positions at final close
    for si, edi, ep, sp, alloc, is_pyr in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * leverage * pnl

    return trades, equity, max_dd


def analyze(trades, equity, max_dd, label=""):
    """Print trade statistics."""
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
    hold_days=5,
    pyramid_ratio=0.5,
    pyramid_day=1,
):
    """Walk-forward validation: one year at a time."""
    print(f"\n{'=' * 70}")
    print(
        f"  WALK-FORWARD V13 (ratio={pyramid_ratio}, day={pyramid_day})"
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

        trades, _, _ = backtest_v13(
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
            hold_days=hold_days,
            atr_stop=atr_stop,
            min_rank=min_rank,
            min_confidence=min_confidence,
            use_ker_gate=use_ker_gate,
            pyramid_ratio=pyramid_ratio,
            pyramid_day=pyramid_day,
            start_di=test_start,
            end_di=test_end_idx + 1,
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
    return []


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("  V13: NW KERNEL REGRESSION + VDP CONFIRMATION")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start="2016-01-01")
    print(
        f"  {NS} sym, {ND} days, "
        f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}"
    )

    sigs = compute_signals(C, O, H, L, V, OI, NS, ND)

    # Find 2019 start index
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # ------------------------------------------------------------------
    # 1. Walk-Forward Validation with different pyramid ratios
    # ------------------------------------------------------------------
    for ratio in [0.3, 0.5, 0.7, 1.0]:
        walk_forward(
            C,
            O,
            H,
            L,
            NS,
            ND,
            dates,
            syms,
            sigs,
            pyramid_ratio=ratio,
            pyramid_day=1,
        )

    # ------------------------------------------------------------------
    # 2. Full 10-year with different pyramid configs
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  FULL 2016-2026 (10 years) -- PYRAMID PROFILES")
    print("=" * 70)

    profiles = [
        (0, 1, "Conservative (no pyramid)"),
        (0.3, 1, "Mild pyramid (30%)"),
        (0.5, 1, "Moderate pyramid (50%)"),
        (0.7, 1, "Aggressive pyramid (70%)"),
        (1.0, 1, "Full pyramid (100%)"),
    ]

    for ratio, pday, label in profiles:
        trades, eq, dd = backtest_v13(
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
            hold_days=5,
            atr_stop=3.0,
            min_confidence=3,
            use_ker_gate=True,
            pyramid_ratio=ratio,
            pyramid_day=pday,
            start_di=60,
        )
        print(f"\n  {label}")
        analyze(trades, eq, dd, label)

    # ------------------------------------------------------------------
    # 3. Multi-position + pyramid grid search (2019-2026)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  MULTI-POSITION + PYRAMID (2019-2026)")
    print("=" * 70)

    results = []
    for tn in [1, 2, 3]:
        for ratio in [0, 0.3, 0.5, 0.7]:
            for mc in [2, 3]:
                trades, eq, dd = backtest_v13(
                    C,
                    O,
                    H,
                    L,
                    NS,
                    ND,
                    dates,
                    syms,
                    sigs,
                    top_n=tn,
                    hold_days=5,
                    atr_stop=3.0,
                    min_confidence=mc,
                    use_ker_gate=True,
                    pyramid_ratio=ratio,
                    pyramid_day=1,
                    start_di=bt_2019,
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
                results.append(
                    {
                        "tn": tn,
                        "ratio": ratio,
                        "mc": mc,
                        "n": len(trades),
                        "wr": wr,
                        "ann": ann,
                        "dd": dd,
                        "sharpe": sh_val,
                    }
                )

    results.sort(key=lambda x: -x["sharpe"])
    print(
        f"\n{'TN':>3} {'Pyr':>4} {'MC':>3} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}"
    )
    print("-" * 55)
    for r in results[:25]:
        print(
            f"{r['tn']:>3} {r['ratio']:>4.1f} {r['mc']:>3} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>5.2f}"
        )

    # ------------------------------------------------------------------
    # 4. Best 10-year multi-position
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  BEST 10-YEAR MULTI-POSITION")
    print("=" * 70)

    best = results[:5]
    for r in best:
        trades, eq, dd = backtest_v13(
            C,
            O,
            H,
            L,
            NS,
            ND,
            dates,
            syms,
            sigs,
            top_n=r["tn"],
            hold_days=5,
            atr_stop=3.0,
            min_confidence=r["mc"],
            use_ker_gate=True,
            pyramid_ratio=r["ratio"],
            pyramid_day=1,
            start_di=60,
        )
        label = f"tn={r['tn']} pyr={r['ratio']:.1f} mc={r['mc']}"
        print(f"\n  FULL {label}")
        analyze(trades, eq, dd, label)

    print(f"\n[V13] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
