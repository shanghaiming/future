"""
V24: VOLATILITY BREAKOUT MEAN REVERSION
========================================
Enter when vol spikes + price drops. Volatility regime changes often
precede reversals. When ATR suddenly spikes (2x+ normal) and price is
oversold, the panic is likely to reverse.

Architecture:
1. Rolling 20d ATR and its z-score (volatility regime)
2. Vol spike: ATR_zscore > threshold (abnormally high volatility)
3. Price oversold: ret_5d < -3% OR consec_dn >= 3
4. OI declining (capitulation confirmation)
5. Composite score:
   - vol_spike_magnitude (0.30): how extreme the ATR spike
   - price_oversold (0.30): how oversold the price
   - oi_capitulation (0.20): OI declining with price
   - volume_surge (0.20): volume z-score > 1.5
6. Cross-sectional rank
7. KER gate, pyramid on day-1 winners, hold 5d, ATR stop 3.0

Signal at close[di], enter at open[di+1]. No look-ahead.
No leverage. Walk-forward 2019-2026.
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
# SIGNAL COMPUTATION
# ============================================================
def compute_signals(
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    V: np.ndarray,
    OI: np.ndarray,
    NS: int,
    ND: int,
    atr_z_threshold: float = 1.5,
) -> dict:
    """Compute volatility breakout mean reversion signals.

    Returns dict of signal arrays keyed by name.
    """
    t0 = time.time()
    print("[V24] Computing signals...", flush=True)

    # --- 1. Rolling 20-day ATR ---
    atr_20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        # Compute True Range first
        tr = np.full(ND, np.nan)
        for di in range(1, ND):
            hh = H[si, di]
            ll = L[si, di]
            cc = C[si, di - 1]
            if np.isnan(hh) or np.isnan(ll) or np.isnan(cc):
                continue
            tr[di] = max(hh - ll, abs(hh - cc), abs(ll - cc))

        # Rolling 20-day average of TR
        for di in range(20, ND):
            window = tr[di - 20 : di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 10:
                atr_20[si, di] = np.mean(valid)

    # --- 2. ATR z-score (60-day rolling lookback) ---
    atr_zscore = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(60, ND):
            window = atr_20[si, di - 60 : di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 30 and not np.isnan(atr_20[si, di]):
                mu = np.mean(valid)
                sig = np.std(valid)
                if sig > 1e-10:
                    atr_zscore[si, di] = (atr_20[si, di] - mu) / sig

    # --- 3. Consecutive down days ---
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

    # --- 4. 5d return ---
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (
                not np.isnan(C[si, di])
                and not np.isnan(C[si, di - 5])
                and C[si, di - 5] > 0
            ):
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1

    # --- 5. OI decline (capitulation) ---
    oi_capitulation = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (
                np.isnan(OI[si, di])
                or np.isnan(OI[si, di - 5])
                or np.isnan(C[si, di])
                or np.isnan(C[si, di - 5])
                or C[si, di - 5] <= 0
                or OI[si, di - 5] <= 0
            ):
                continue
            oi_chg = OI[si, di] / OI[si, di - 5] - 1
            price_chg = C[si, di] / C[si, di - 5] - 1
            # OI declining while price drops = capitulation
            if oi_chg < -0.02 and price_chg < -0.02:
                oi_capitulation[si, di] = min(abs(oi_chg), 0.2) / 0.2 * min(
                    abs(price_chg), 0.1
                ) / 0.1
            else:
                oi_capitulation[si, di] = 0.0

    # --- 6. Volume z-score (20d lookback) ---
    vol_zscore = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            window = V[si, di - 20 : di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 10 and not np.isnan(V[si, di]):
                mu = np.mean(valid)
                sig = np.std(valid)
                if sig > 1e-10:
                    vol_zscore[si, di] = (V[si, di] - mu) / sig

    # --- 7. KER (Kaufman Efficiency Ratio) 10d ---
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

    # --- 8. RSI 14 ---
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

    # --- 9. ATR for stops (14d) ---
    atr14 = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            h = np.where(np.isnan(H[si]), 0, H[si]).astype(np.float64)
            l = np.where(np.isnan(L[si]), 0, L[si]).astype(np.float64)
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nan_mask = np.isnan(C[si])
            try:
                atr = talib.ATR(h, l, c, 14)
                atr14[si] = np.where(nan_mask, np.nan, atr)
            except Exception:
                pass
    else:
        # Manual ATR14 fallback
        for si in range(NS):
            tr = np.full(ND, np.nan)
            for di in range(1, ND):
                hh = H[si, di]
                ll = L[si, di]
                cc = C[si, di - 1]
                if np.isnan(hh) or np.isnan(ll) or np.isnan(cc):
                    continue
                tr[di] = max(hh - ll, abs(hh - cc), abs(ll - cc))
            for di in range(14, ND):
                window = tr[di - 14 : di]
                valid = window[~np.isnan(window)]
                if len(valid) >= 7:
                    atr14[si, di] = np.mean(valid)

    # --- 10. KER regime ---
    ker_regime = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        for di in range(ND):
            if np.isnan(ker_10[si, di]):
                continue
            if ker_10[si, di] < 0.15:
                ker_regime[si, di] = 1  # mean-reverting regime
            elif ker_10[si, di] > 0.3:
                ker_regime[si, di] = -1  # trending regime

    # --- 11. Composite score ---
    raw_score = np.full((NS, ND), np.nan)
    for di in range(ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            if np.isnan(atr_zscore[si, di]):
                continue

            # Must have vol spike above threshold
            if atr_zscore[si, di] < atr_z_threshold:
                continue

            # Must be oversold (at least one condition)
            is_oversold = False
            if not np.isnan(ret_5d[si, di]) and ret_5d[si, di] < -0.03:
                is_oversold = True
            if consec_dn[si, di] >= 3:
                is_oversold = True
            if not is_oversold:
                continue

            s = 0.0
            w_total = 0.0

            # Vol spike magnitude (0.30): how extreme the ATR spike
            vol_spike = atr_zscore[si, di]
            s += min(vol_spike / 4.0, 1.0) * 0.30
            w_total += 0.30

            # Price oversold (0.30): how oversold the price
            oversold_score = 0.0
            if not np.isnan(ret_5d[si, di]):
                oversold_score += min(max(-ret_5d[si, di] / 0.1, 0), 1.0) * 0.5
            if consec_dn[si, di] > 0:
                oversold_score += min(consec_dn[si, di] / 5.0, 1.0) * 0.5
            s += min(oversold_score, 1.0) * 0.30
            w_total += 0.30

            # OI capitulation (0.20)
            if not np.isnan(oi_capitulation[si, di]):
                s += oi_capitulation[si, di] * 0.20
                w_total += 0.20

            # Volume surge (0.20): high volume confirms the panic
            if not np.isnan(vol_zscore[si, di]):
                vol_surge = max(vol_zscore[si, di] - 1.0, 0.0) / 2.0
                s += min(vol_surge, 1.0) * 0.20
                w_total += 0.20

            if w_total > 0:
                scores[si] = s / w_total

        valid = ~np.isnan(scores)
        if valid.sum() >= 5:
            raw_score[:, di] = pd.Series(scores).rank(
                pct=True, na_option="keep"
            ).values

    # --- 12. Confidence level (number of sub-signals firing) ---
    n_signals = np.zeros((NS, ND), dtype=int)
    for di in range(ND):
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            n = 0
            # Vol spike
            if not np.isnan(atr_zscore[si, di]) and atr_zscore[si, di] >= atr_z_threshold:
                n += 1
            # Price oversold via 5d return
            if not np.isnan(ret_5d[si, di]) and ret_5d[si, di] < -0.03:
                n += 1
            # Consecutive down days
            if consec_dn[si, di] >= 3:
                n += 1
            # OI capitulation
            if not np.isnan(oi_capitulation[si, di]) and oi_capitulation[si, di] > 0.1:
                n += 1
            # Volume surge
            if not np.isnan(vol_zscore[si, di]) and vol_zscore[si, di] > 1.5:
                n += 1
            # RSI oversold
            if not np.isnan(rsi14[si, di]) and rsi14[si, di] < 35:
                n += 1
            n_signals[si, di] = n

    elapsed = time.time() - t0
    print(f"  Done: {elapsed:.1f}s (TA-Lib: {HAS_TALIB})", flush=True)

    return {
        "combo_rank": raw_score,
        "atr_zscore": atr_zscore,
        "atr_20": atr_20,
        "consec_dn": consec_dn,
        "ret_5d": ret_5d,
        "oi_capitulation": oi_capitulation,
        "vol_zscore": vol_zscore,
        "ker_10": ker_10,
        "ker_regime": ker_regime,
        "n_signals": n_signals,
        "rsi14": rsi14,
        "atr14": atr14,
    }


# ============================================================
# BACKTEST ENGINE
# ============================================================
def backtest_v24(
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    NS: int,
    ND: int,
    dates: list,
    syms: list,
    sigs: dict,
    top_n: int = 1,
    min_confidence: int = 3,
    atr_stop: float = 3.0,
    hold_days: int = 5,
    pyramid_ratio: float = 0.5,
    use_ker_gate: bool = True,
    start_di: int = 60,
    end_di: int | None = None,
) -> tuple:
    """Day-by-day backtest with KER gate and pyramid.

    Signal at close[di], enter at open[di+1]. No look-ahead.

    Positions tuple:
      (si, entry_di, entry_price, stop_price, alloc, direction,
       max_hold, confidence, combo_rank_at_entry)

    Returns (trades, equity, max_dd).
    """
    combo_rank = sigs["combo_rank"]
    ker_regime = sigs["ker_regime"]
    n_signals = sigs["n_signals"]
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
        daily_pnl = 0.0
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
            hold = di - edi

            # ATR stop loss
            if direction > 0 and c < sp:
                exit_r = "stop"
            elif direction < 0 and c > sp:
                exit_r = "stop"

            # Timer expiry
            if exit_r is None and hold >= max_hold:
                exit_r = "timer"

            if exit_r:
                pnl = direction * (c - ep) / ep - COMM
                profit = equity * alloc * pnl
                daily_pnl += profit
                trades.append(
                    {
                        "pnl_abs": profit,
                        "pnl_pct": pnl * 100,
                        "days": hold + 1,
                        "di": di,
                        "year": d.year,
                        "sym": syms[si],
                        "reason": exit_r,
                        "dir": direction,
                        "confidence": confidence,
                        "entry_rank": entry_rank,
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
                        extra_alloc = alloc * pyramid_ratio
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

        # Check next day's open exists (no gap signal)
        if di + 1 >= ND:
            continue

        candidates = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(combo_rank[si, di]):
                continue
            # Confidence gate
            if n_signals[si, di] < min_confidence:
                continue
            # KER regime gate: only enter in mean-reverting regime
            if use_ker_gate and ker_regime[si, di] < 0:
                continue
            if np.isnan(O[si, di + 1]):
                continue

            rank = combo_rank[si, di]
            confidence = n_signals[si, di]
            alloc = 1.0 / max(top_n, 1)

            candidates.append((rank, si, alloc, confidence, rank))

        candidates.sort(key=lambda x: -x[0])
        for rank, si, alloc, confidence, entry_rank in candidates[:top_n]:
            if len(positions) >= max_pos or si in held:
                break
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue

            # ATR stop using atr14 at signal day
            atr_v = atr14[si, di]
            if np.isnan(atr_v):
                # Fallback: manual compute
                atr_vals = []
                for j in range(max(start_di, di - 14), di):
                    hh, ll, cc = H[si, j], L[si, j], C[si, j]
                    if not any(np.isnan([hh, ll, cc])):
                        atr_vals.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
                if not atr_vals:
                    continue
                atr_v = np.mean(atr_vals)

            stop = ep - atr_stop * atr_v
            positions.append(
                (si, di + 1, ep, stop, alloc, 1, hold_days, confidence, entry_rank)
            )
            held.add(si)

    # Close remaining
    for si, edi, ep, sp, alloc, direction, _, confidence, entry_rank in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = direction * (c - ep) / ep - COMM
            equity += equity * alloc * pnl

    return trades, equity, max_dd


# ============================================================
# ANALYSIS
# ============================================================
def analyze(trades: list, equity: float, max_dd: float, label: str = "") -> dict | None:
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

    print("    Exit reasons:")
    for r in ["stop", "timer"]:
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

    print("    By confidence:")
    for c in sorted(conf_groups.keys()):
        cg = conf_groups[c]
        cwr = cg["w"] / cg["n"] * 100
        avg_pnl = np.mean(cg["pnl"])
        print(f"      conf={c}: {cg['n']:>4}t WR={cwr:.1f}% avg={avg_pnl:+.2f}%")

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
    print("    Yearly:")
    for y in sorted(yr.keys()):
        ys = yr[y]
        cum = np.prod([1 + p / 100 for p in ys["pnl"]]) - 1
        print(f"      {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% cum={cum:+.1%}")

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
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    V: np.ndarray,
    OI: np.ndarray,
    NS: int,
    ND: int,
    dates: list,
    syms: list,
    train_years: int = 4,
    top_n: int = 1,
    min_confidence: int = 3,
    atr_stop: float = 3.0,
    hold_days: int = 5,
    pyramid_ratio: float = 0.5,
    use_ker_gate: bool = True,
    atr_z_threshold: float = 1.5,
) -> list:
    """Walk-forward validation: train on N years, test on next year."""
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD: train={train_years}y, V24 vol breakout MR")
    print(f"{'='*70}")

    years = sorted(set(d.year for d in dates))
    all_trades = []

    start_year = years[0]
    while True:
        train_end = start_year + train_years - 1
        test_year = train_end + 1
        if test_year > years[-1]:
            break

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

        sigs = compute_signals(C, O, H, L, V, OI, NS, ND, atr_z_threshold=atr_z_threshold)

        train_start_di = max(0, test_start_di - train_years * 252)
        trades, eq, dd = backtest_v24(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=top_n,
            min_confidence=min_confidence,
            atr_stop=atr_stop,
            hold_days=hold_days,
            pyramid_ratio=pyramid_ratio,
            use_ker_gate=use_ker_gate,
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
# PARAMETER SWEEP
# ============================================================
def parameter_sweep(
    C: np.ndarray,
    O: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    V: np.ndarray,
    OI: np.ndarray,
    NS: int,
    ND: int,
    dates: list,
    syms: list,
    bt_2019: int,
) -> list:
    """Sweep key parameters and return sorted results."""
    print(f"\n{'='*70}")
    print("  PARAMETER SWEEP: V24 VOL BREAKOUT MR")
    print(f"{'='*70}")

    results = []

    for atr_z in [1.0, 1.5, 2.0]:
        sigs = compute_signals(C, O, H, L, V, OI, NS, ND, atr_z_threshold=atr_z)
        for top_n in [1, 2, 3]:
            for min_conf in [2, 3]:
                for pyramid in [0.0, 0.5]:
                    for atr_stop_val in [2.5, 3.0]:
                        trades, eq, dd = backtest_v24(
                            C, O, H, L, NS, ND, dates, syms, sigs,
                            top_n=top_n,
                            min_confidence=min_conf,
                            atr_stop=atr_stop_val,
                            hold_days=5,
                            pyramid_ratio=pyramid,
                            use_ker_gate=True,
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
                                "atr_z": atr_z,
                                "top_n": top_n,
                                "min_conf": min_conf,
                                "pyramid": pyramid,
                                "atr_stop": atr_stop_val,
                                "n": len(trades),
                                "wr": wr,
                                "ann": ann,
                                "dd": dd,
                                "sharpe": sh_val,
                                "eq": eq,
                            }
                        )

    results.sort(key=lambda x: -x["sharpe"])
    print(
        f"\n{'ATRz':>4} {'TopN':>4} {'MinC':>4} {'Pyr':>4} {'ATRs':>4} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>6}"
    )
    print("-" * 65)
    for r in results[:30]:
        print(
            f"{r['atr_z']:>4.1f} {r['top_n']:>4} {r['min_conf']:>4} "
            f"{r['pyramid']:>4.1f} {r['atr_stop']:>4.1f} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f}"
        )

    return results


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("  V24: VOLATILITY BREAKOUT MEAN REVERSION")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start="2016-01-01")
    print(
        f"  {NS} sym, {ND} days, "
        f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}"
    )

    sigs = compute_signals(C, O, H, L, V, OI, NS, ND, atr_z_threshold=1.5)

    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # ============================================================
    # SECTION 1: BASELINE RUNS (2019-2026)
    # ============================================================
    print(f"\n{'='*70}")
    print("  SECTION 1: BASELINE V24 (2019-2026)")
    print(f"{'='*70}")

    # Default params
    print("\n  --- Default: atr_z=1.5, top_n=1, conf>=3, pyramid=0.5, atr_stop=3.0 ---")
    trades_def, eq_def, dd_def = backtest_v24(
        C, O, H, L, NS, ND, dates, syms, sigs,
        top_n=1, min_confidence=3, atr_stop=3.0,
        hold_days=5, pyramid_ratio=0.5, use_ker_gate=True,
        start_di=bt_2019,
    )
    analyze(trades_def, eq_def, dd_def, "Default")

    # No pyramid
    print("\n  --- No pyramid ---")
    trades_nopyr, eq_nopyr, dd_nopyr = backtest_v24(
        C, O, H, L, NS, ND, dates, syms, sigs,
        top_n=1, min_confidence=3, atr_stop=3.0,
        hold_days=5, pyramid_ratio=0.0, use_ker_gate=True,
        start_di=bt_2019,
    )
    analyze(trades_nopyr, eq_nopyr, dd_nopyr, "NoPyramid")

    # Top 2
    print("\n  --- Top 2 ---")
    trades_top2, eq_top2, dd_top2 = backtest_v24(
        C, O, H, L, NS, ND, dates, syms, sigs,
        top_n=2, min_confidence=3, atr_stop=3.0,
        hold_days=5, pyramid_ratio=0.5, use_ker_gate=True,
        start_di=bt_2019,
    )
    analyze(trades_top2, eq_top2, dd_top2, "Top2")

    # No KER gate
    print("\n  --- No KER gate ---")
    trades_noker, eq_noker, dd_noker = backtest_v24(
        C, O, H, L, NS, ND, dates, syms, sigs,
        top_n=1, min_confidence=3, atr_stop=3.0,
        hold_days=5, pyramid_ratio=0.5, use_ker_gate=False,
        start_di=bt_2019,
    )
    analyze(trades_noker, eq_noker, dd_noker, "NoKER")

    # ============================================================
    # SECTION 2: PARAMETER SWEEP
    # ============================================================
    sweep_results = parameter_sweep(
        C, O, H, L, V, OI, NS, ND, dates, syms, bt_2019
    )

    # ============================================================
    # SECTION 3: BEST CONFIG YEARLY BREAKDOWN (2019-2026)
    # ============================================================
    print(f"\n{'='*70}")
    print("  SECTION 3: BEST CONFIGS -- YEARLY (2019-2026)")
    print(f"{'='*70}")

    for r in sweep_results[:3]:
        best_sigs = compute_signals(
            C, O, H, L, V, OI, NS, ND, atr_z_threshold=r["atr_z"]
        )
        trades, eq, dd = backtest_v24(
            C, O, H, L, NS, ND, dates, syms, best_sigs,
            top_n=r["top_n"],
            min_confidence=r["min_conf"],
            atr_stop=r["atr_stop"],
            hold_days=5,
            pyramid_ratio=r["pyramid"],
            use_ker_gate=True,
            start_di=bt_2019,
        )
        label = (
            f"az{r['atr_z']:.1f}_tn{r['top_n']}_mc{r['min_conf']}"
            f"_p{r['pyramid']:.1f}_as{r['atr_stop']:.1f}"
        )
        print(f"\n  --- {label} ---")
        analyze(trades, eq, dd, label)

    # ============================================================
    # SECTION 4: WALK-FORWARD (best config)
    # ============================================================
    if sweep_results:
        best = sweep_results[0]
        print(f"\n  Best config: atr_z={best['atr_z']}, top_n={best['top_n']}, "
              f"min_conf={best['min_conf']}, pyramid={best['pyramid']}, "
              f"atr_stop={best['atr_stop']}, sharpe={best['sharpe']:.2f}")
        wf_trades = walk_forward(
            C, O, H, L, V, OI, NS, ND, dates, syms,
            train_years=4,
            top_n=best["top_n"],
            min_confidence=best["min_conf"],
            atr_stop=best["atr_stop"],
            hold_days=5,
            pyramid_ratio=best["pyramid"],
            use_ker_gate=True,
            atr_z_threshold=best["atr_z"],
        )

    # ============================================================
    # SECTION 5: FULL 10-YEAR (2016-2026)
    # ============================================================
    print(f"\n{'='*70}")
    print("  SECTION 5: FULL 2016-2026 (10 years)")
    print(f"{'='*70}")

    # Default params full 10yr
    print("\n  --- Default 10yr ---")
    trades_10yr, eq_10yr, dd_10yr = backtest_v24(
        C, O, H, L, NS, ND, dates, syms, sigs,
        top_n=1, min_confidence=3, atr_stop=3.0,
        hold_days=5, pyramid_ratio=0.5, use_ker_gate=True,
        start_di=60,
    )
    analyze(trades_10yr, eq_10yr, dd_10yr, "10yr-Default")

    if sweep_results:
        for r in sweep_results[:2]:
            best_sigs = compute_signals(
                C, O, H, L, V, OI, NS, ND, atr_z_threshold=r["atr_z"]
            )
            trades, eq, dd = backtest_v24(
                C, O, H, L, NS, ND, dates, syms, best_sigs,
                top_n=r["top_n"],
                min_confidence=r["min_conf"],
                atr_stop=r["atr_stop"],
                hold_days=5,
                pyramid_ratio=r["pyramid"],
                use_ker_gate=True,
                start_di=60,
            )
            label = (
                f"10yr-az{r['atr_z']:.1f}_tn{r['top_n']}"
                f"_mc{r['min_conf']}_p{r['pyramid']:.1f}"
            )
            print(f"\n  {label}")
            analyze(trades, eq, dd, label)

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print("  SUMMARY: V24 VOLATILITY BREAKOUT MEAN REVERSION")
    print(f"{'='*70}")
    res_def = analyze(
        trades_def, eq_def, dd_def, "2019-2026 Default"
    ) if trades_def else None
    res_10yr = analyze(
        trades_10yr, eq_10yr, dd_10yr, "2016-2026 Default"
    ) if trades_10yr else None

    if res_def:
        print(f"\n  2019-2026: ann={res_def['ann']:+.1f}%  "
              f"WR={res_def['wr']:.1f}%  "
              f"Sh={res_def['sh']:.2f}  DD={res_def['dd']:.1f}%")
    if res_10yr:
        print(f"  2016-2026: ann={res_10yr['ann']:+.1f}%  "
              f"WR={res_10yr['wr']:.1f}%  "
              f"Sh={res_10yr['sh']:.2f}  DD={res_10yr['dd']:.1f}%")
    if sweep_results:
        b = sweep_results[0]
        print(f"  Best sweep: ann={b['ann']:+.1f}%  "
              f"WR={b['wr']:.1f}%  "
              f"Sh={b['sharpe']:.2f}  DD={b['dd']:.1f}%")
        print(f"    config: atr_z={b['atr_z']} top_n={b['top_n']} "
              f"min_conf={b['min_conf']} pyramid={b['pyramid']:.1f} "
              f"atr_stop={b['atr_stop']:.1f}")

    print(f"\n[V24] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
