"""
V23: CALENDAR ANOMALY MEAN REVERSION
=====================================
Exploit seasonal/calendar patterns in Chinese commodity futures:
  1. V1 multi-alpha composite (consec_dn, ret_5d, OI, VDP, RSI, BB, CCI)
  2. Per-instrument calendar MR score:
     - How well does THIS instrument mean-revert in THIS calendar context?
     - Uses rolling historical win rate when MR signals fire in same month/DOW/holiday
  3. Calendar boost: blend base_signal rank with per-instrument calendar score
     - Since calendar score varies across instruments (not just across days),
       the cross-sectional ranking actually changes
  4. Cross-sectional re-rank (base_signal * calendar_boost)
  5. KER regime gate
  6. Pyramid on day-1 winners
  7. Walk-forward 2019-2026, full 10-year

Signal at close[di], enter at open[di+1]. No look-ahead.
"""
import sys
import os
import time
import warnings
from collections import defaultdict
from datetime import date, timedelta

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
# CHINESE HOLIDAYS 2016-2026 (approximate trading holidays)
# ============================================================
def build_chinese_holidays() -> set:
    """Build a set of approximate Chinese futures market holiday dates."""
    holidays = set()

    spring_festivals = [
        (2016, 2, 8), (2017, 1, 28), (2018, 2, 16), (2019, 2, 5),
        (2020, 1, 25), (2021, 2, 12), (2022, 2, 1), (2023, 1, 22),
        (2024, 2, 10), (2025, 1, 29), (2026, 2, 17),
    ]
    for y, m, d in spring_festivals:
        base = date(y, m, d)
        for offset in range(-1, 8):
            dt = base + timedelta(days=offset)
            if dt.weekday() < 5:
                holidays.add(dt)

    for year in range(2016, 2027):
        for day in range(1, 8):
            dt = date(year, 10, day)
            if dt.weekday() < 5:
                holidays.add(dt)

    for year in range(2016, 2027):
        for day in range(1, 4):
            dt = date(year, 5, day)
            if dt.weekday() < 5:
                holidays.add(dt)

    mid_autumn = [
        (2016, 9, 15), (2017, 10, 4), (2018, 9, 24), (2019, 9, 13),
        (2020, 10, 1), (2021, 9, 21), (2022, 9, 10), (2023, 9, 29),
        (2024, 9, 17), (2025, 10, 6), (2026, 9, 25),
    ]
    for y, m, d in mid_autumn:
        holidays.add(date(y, m, d))

    qingming_dates = [
        (2016, 4, 4), (2017, 4, 4), (2018, 4, 5), (2019, 4, 5),
        (2020, 4, 4), (2021, 4, 5), (2022, 4, 5), (2023, 4, 5),
        (2024, 4, 4), (2025, 4, 4), (2026, 4, 5),
    ]
    for y, m, d in qingming_dates:
        holidays.add(date(y, m, d))

    dragon_boat = [
        (2016, 6, 9), (2017, 5, 30), (2018, 6, 18), (2019, 6, 7),
        (2020, 6, 25), (2021, 6, 14), (2022, 6, 3), (2023, 6, 22),
        (2024, 6, 10), (2025, 5, 31), (2026, 6, 19),
    ]
    for y, m, d in dragon_boat:
        holidays.add(date(y, m, d))

    return holidays


CHINESE_HOLIDAYS = build_chinese_holidays()


# ============================================================
# CALENDAR DIMENSION FEATURES
# ============================================================
def compute_calendar_features(dates, ND):
    """Compute calendar dimension arrays for each day."""
    t0 = time.time()
    print("[V23] Computing calendar features...", flush=True)

    month_of_year = np.full(ND, -1, dtype=int)
    day_of_week = np.full(ND, -1, dtype=int)
    days_to_month_end = np.full(ND, -1, dtype=int)
    is_pre_holiday = np.zeros(ND, dtype=bool)

    for di in range(ND):
        d = dates[di]
        if isinstance(d, pd.Timestamp):
            dt = d.date()
        elif isinstance(d, np.datetime64):
            dt = pd.Timestamp(d).date()
        else:
            dt = d

        month_of_year[di] = dt.month
        day_of_week[di] = dt.weekday()

        import calendar as cal_mod
        last_day = cal_mod.monthrange(dt.year, dt.month)[1]
        remaining = last_day - dt.day
        if remaining == 0:
            days_to_month_end[di] = 0
        elif remaining == 1:
            days_to_month_end[di] = 1
        elif remaining <= 3:
            days_to_month_end[di] = 2
        elif remaining <= 7:
            days_to_month_end[di] = 3
        else:
            days_to_month_end[di] = 4

        for offset in range(1, 4):
            future = dt + timedelta(days=offset)
            if future in CHINESE_HOLIDAYS:
                is_pre_holiday[di] = True
                break

    elapsed = time.time() - t0
    print(
        f"  Calendar features done: {elapsed:.1f}s  "
        f"pre_holiday_days={is_pre_holiday.sum()}",
        flush=True,
    )
    return month_of_year, day_of_week, days_to_month_end, is_pre_holiday


# ============================================================
# PER-INSTRUMENT CALENDAR MR SCORE
# ============================================================
def compute_instrument_calendar_score(
    C, NS, ND, dates, month_of_year, day_of_week, is_pre_holiday,
    lookback=504,
):
    """
    Compute per-instrument calendar MR score: how well does this instrument
    mean-revert in this calendar context historically?

    Returns (NS, ND) array of calendar scores in [0, 1]:
    - 1.0 = historically strong MR in this calendar context
    - 0.0 = historically poor MR in this calendar context
    - 0.5 = neutral (insufficient data)
    """
    t0 = time.time()
    print("[V23] Computing per-instrument calendar scores...", flush=True)

    cal_score = np.full((NS, ND), 0.5)

    # Precompute daily direction for all instruments
    daily_up = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(1, ND):
            if (
                not np.isnan(C[si, di])
                and not np.isnan(C[si, di - 1])
                and C[si, di - 1] > 0
            ):
                daily_up[si, di] = C[si, di] > C[si, di - 1]

    for si in range(NS):
        if si % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"  Instrument {si}/{NS}... ({elapsed:.1f}s)",
                flush=True,
            )
        for di in range(lookback, ND):
            cur_month = month_of_year[di]
            cur_dow = day_of_week[di]
            cur_hol = is_pre_holiday[di]

            lb = max(0, di - lookback)
            wins = 0
            total = 0

            for j in range(lb, di):
                matched = False
                if month_of_year[j] == cur_month:
                    matched = True
                elif day_of_week[j] == cur_dow:
                    matched = True
                elif cur_hol and is_pre_holiday[j]:
                    matched = True

                if matched and j >= 2:
                    prev_down = not daily_up[si, j - 1]
                    cur_up = daily_up[si, j]
                    total += 1
                    if prev_down and cur_up:
                        wins += 1

            if total >= 10:
                cal_score[si, di] = wins / total

    elapsed = time.time() - t0
    print(f"  Instrument calendar scores done: {elapsed:.1f}s", flush=True)
    return cal_score


# ============================================================
# CALENDAR-BOOSTED RANKING
# ============================================================
def compute_calendar_boosted_rank(
    base_rank, NS, ND, inst_cal_score, calendar_weight=0.2,
):
    """
    Blend per-instrument calendar score into base rank.
    Since cal_score varies by instrument (not just by day), this
    actually changes the cross-sectional ranking order.

    boosted = (1 - w) * base_rank + w * cal_score
    """
    boosted = np.full((NS, ND), np.nan)

    for di in range(ND):
        for si in range(NS):
            if np.isnan(base_rank[si, di]):
                continue
            boosted[si, di] = (
                (1 - calendar_weight) * base_rank[si, di]
                + calendar_weight * inst_cal_score[si, di]
            )

    final_rank = np.full((NS, ND), np.nan)
    for di in range(ND):
        vals = boosted[:, di]
        valid = ~np.isnan(vals)
        if valid.sum() >= 5:
            final_rank[:, di] = pd.Series(vals).rank(
                pct=True, na_option="keep"
            ).values

    return final_rank


# ============================================================
# SIGNAL COMPUTATION (V1 multi-alpha)
# ============================================================
def compute_signals(C, O, H, L, V, OI, NS, ND):
    """Compute V1 multi-alpha oversold signals."""
    t0 = time.time()
    print("[V23] Computing base signals...", flush=True)

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

    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (
                not np.isnan(C[si, di])
                and not np.isnan(C[si, di - 5])
                and C[si, di - 5] > 0
            ):
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1

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

    oi_decline = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if np.isnan(OI[si, di]) or np.isnan(OI[si, di - 5]):
                continue
            if (
                np.isnan(C[si, di])
                or np.isnan(C[si, di - 5])
                or C[si, di - 5] <= 0
            ):
                continue
            oi_chg = OI[si, di] / OI[si, di - 5] - 1
            price_chg = C[si, di] / C[si, di - 5] - 1
            if oi_chg < -0.02 and price_chg < -0.02:
                oi_decline[si, di] = min(abs(oi_chg), 0.2) / 0.2 * min(
                    abs(price_chg), 0.1
                ) / 0.1
            else:
                oi_decline[si, di] = 0.0

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

    raw_score = np.full((NS, ND), np.nan)
    for di in range(ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue

            s = 0.0
            w_total = 0.0

            cd = consec_dn[si, di]
            s += min(cd / 5.0, 1.0) * 0.20
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
            raw_score[:, di] = pd.Series(scores).rank(
                pct=True, na_option="keep"
            ).values

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

    ker_regime = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        for di in range(ND):
            if np.isnan(ker_10[si, di]):
                continue
            if ker_10[si, di] < 0.15:
                ker_regime[si, di] = 1
            elif ker_10[si, di] > 0.3:
                ker_regime[si, di] = -1

    elapsed = time.time() - t0
    print(f"  Base signals done: {elapsed:.1f}s (TA-Lib: {HAS_TALIB})", flush=True)

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
    }


# ============================================================
# BACKTEST ENGINE
# ============================================================
def backtest_v23(
    C, O, H, L, NS, ND, dates, syms, sigs,
    final_rank,
    top_n=1,
    hold_days=5,
    atr_stop=3.0,
    min_confidence=3,
    use_ker_gate=True,
    pyramid_ratio=0.5,
    leverage=1.0,
    start_di=60,
    end_di=None,
):
    """Calendar-boosted mean reversion backtest."""
    combo_rank = final_rank
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

        for si, edi, ep, sp, alloc, direction in positions:
            c = C[si, di]
            if np.isnan(c):
                new_positions.append((si, edi, ep, sp, alloc, direction))
                continue

            exit_r = None
            hold_elapsed = di - edi

            if direction > 0 and c < sp:
                exit_r = "stop"
            elif direction < 0 and c > sp:
                exit_r = "stop"
            elif hold_elapsed >= hold_days:
                exit_r = "timer"

            if exit_r:
                pnl = direction * (c - ep) / ep - COMM
                profit = equity * alloc * leverage * pnl
                daily_pnl += profit
                trades.append(
                    {
                        "pnl_abs": profit,
                        "pnl_pct": pnl * 100 * leverage,
                        "days": hold_elapsed + 1,
                        "di": di,
                        "year": d.year,
                        "sym": syms[si],
                        "reason": exit_r,
                        "dir": direction,
                        "confidence": n_signals[si, edi],
                    }
                )
            else:
                new_positions.append((si, edi, ep, sp, alloc, direction))

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
        for si, edi, ep, sp, alloc, direction in positions:
            if di - edi == 1 and pyramid_ratio > 0:
                c = C[si, di]
                if not np.isnan(c):
                    unrealized = direction * (c - ep) / ep
                    if unrealized > 0:
                        extra_alloc = alloc * pyramid_ratio
                        pyramid_positions.append(
                            (si, di, c, sp, extra_alloc, direction)
                        )
        positions.extend(pyramid_positions)

        # --- Entry ---
        held = {p[0] for p in positions}
        max_pos = top_n + top_n
        if len(positions) >= max_pos:
            continue

        candidates = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(combo_rank[si, di]):
                continue
            if combo_rank[si, di] < 0.7:
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
        for rank_val, si, alloc in candidates[:top_n]:
            if len(positions) >= max_pos or si in held:
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

            stop = ep - atr_stop * atr
            positions.append((si, di + 1, ep, stop, alloc, 1))
            held.add(si)

    for si, edi, ep, sp, alloc, direction in positions:
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
        np.mean(rets) / np.std(rets) * np.sqrt(252)
        if np.std(rets) > 0
        else 0
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

    reasons = defaultdict(lambda: {"n": 0, "w": 0, "pnl": []})
    for t in trades:
        r = t["reason"]
        reasons[r]["n"] += 1
        if t["pnl_pct"] > 0:
            reasons[r]["w"] += 1
        reasons[r]["pnl"].append(t["pnl_pct"])

    print(f"    Exit reasons:")
    for r in ["stop", "timer"]:
        if r in reasons:
            rs = reasons[r]
            rwr = rs["w"] / rs["n"] * 100
            avg_pnl = np.mean(rs["pnl"])
            print(f"      {r:>10}: {rs['n']:>4}t WR={rwr:.1f}% avg={avg_pnl:+.2f}%")

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
    C, O, H, L, V, OI, NS, ND, dates, syms,
    train_years=4,
    top_n=1,
    atr_stop=3.0,
    min_confidence=3,
    use_ker_gate=True,
    pyramid_ratio=0.5,
    calendar_weight=0.2,
    hold_days=5,
):
    """Walk-forward: train on N years, test on next year."""
    print(f"\n{'='*70}")
    print(
        f"  WALK-FORWARD: train={train_years}y, "
        f"cal_w={calendar_weight}, tn={top_n}, mc={min_confidence}"
    )
    print(f"{'='*70}")

    years = sorted(set(d.year for d in dates))
    all_trades = []

    month_of_year, day_of_week, days_to_month_end, is_pre_holiday = (
        compute_calendar_features(dates, ND)
    )

    sigs = compute_signals(C, O, H, L, V, OI, NS, ND)

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

        # Compute instrument calendar scores up to test year start
        # Use only training data (no look-ahead)
        train_end_di = test_start_di
        inst_cal_score = compute_instrument_calendar_score(
            C, NS, train_end_di, dates,
            month_of_year, day_of_week, is_pre_holiday,
        )

        # Extend to full ND with last training value
        inst_cal_score_full = np.full((NS, ND), 0.5)
        inst_cal_score_full[:, : train_end_di] = inst_cal_score[:, : train_end_di]
        # Carry forward last training values for test period
        for si in range(NS):
            inst_cal_score_full[si, train_end_di:] = inst_cal_score[si, train_end_di - 1]

        final_rank = compute_calendar_boosted_rank(
            sigs["combo_rank"], NS, ND,
            inst_cal_score_full, calendar_weight=calendar_weight,
        )

        train_start_di = max(0, test_start_di - train_years * 252)
        trades, eq, dd = backtest_v23(
            C, O, H, L, NS, ND, dates, syms, sigs,
            final_rank,
            top_n=top_n,
            hold_days=hold_days,
            atr_stop=atr_stop,
            min_confidence=min_confidence,
            use_ker_gate=use_ker_gate,
            pyramid_ratio=pyramid_ratio,
            start_di=train_start_di,
            end_di=test_end_di + 1,
        )

        yr_trades = [t for t in trades if t["year"] == test_year]
        all_trades.extend(yr_trades)

        if yr_trades:
            pnls = [t["pnl_pct"] for t in yr_trades]
            wr_val = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            print(
                f"  {test_year}: {len(pnls)}t WR={wr_val:.1f}% "
                f"avg={np.mean(pnls):+.3f}%",
                flush=True,
            )
        else:
            print(f"  {test_year}: no trades", flush=True)

        start_year += 1

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

        print(f"    Per-year:")
        for y in sorted(set(t["year"] for t in all_trades)):
            yt = [t for t in all_trades if t["year"] == y]
            pnls = [t["pnl_pct"] for t in yt]
            ywr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            cum = np.prod([1 + p / 100 for p in pnls]) - 1
            print(f"      {y}: {len(yt)}t WR={ywr:.1f}% cum={cum:+.1%}")

        return all_trades

    return []


# ============================================================
# PARAMETER SWEEP
# ============================================================
def parameter_sweep(
    C, O, H, L, V, OI, NS, ND, dates, syms, sigs, bt_2019,
    inst_cal_score_full,
):
    """Sweep calendar_weight, top_n, min_confidence, pyramid, atr_stop."""
    print(f"\n{'='*70}")
    print("  PARAMETER SWEEP: CALENDAR ANOMALY MR")
    print(f"{'='*70}")

    results = []

    for cal_w in [0.1, 0.2, 0.3]:
        final_rank = compute_calendar_boosted_rank(
            sigs["combo_rank"], NS, ND,
            inst_cal_score_full, calendar_weight=cal_w,
        )

        for top_n in [1, 2, 3]:
            for mc in [2, 3]:
                for pyr in [0.0, 0.5]:
                    for atr_s in [2.5, 3.0]:
                        trades, eq, dd = backtest_v23(
                            C, O, H, L, NS, ND, dates, syms, sigs,
                            final_rank,
                            top_n=top_n,
                            hold_days=5,
                            atr_stop=atr_s,
                            min_confidence=mc,
                            use_ker_gate=True,
                            pyramid_ratio=pyr,
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
                        results.append(
                            {
                                "cal_w": cal_w,
                                "tn": top_n,
                                "mc": mc,
                                "pyr": pyr,
                                "atr": atr_s,
                                "n": len(trades),
                                "wr": wr,
                                "ann": ann,
                                "dd": dd,
                                "sharpe": sh_val,
                            }
                        )

    results.sort(key=lambda x: -x["sharpe"])
    print(
        f"\n{'CalW':>4} {'TN':>3} {'MC':>3} {'Pyr':>4} {'ATR':>4} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>6}"
    )
    print("-" * 65)
    for r in results[:30]:
        print(
            f"{r['cal_w']:>4.1f} {r['tn']:>3} {r['mc']:>3} "
            f"{r['pyr']:>4.1f} {r['atr']:>4.1f} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f}"
        )

    return results


# ============================================================
# CALENDAR ANOMALY ANALYSIS
# ============================================================
def analyze_calendar_anomalies(C, NS, ND, dates):
    """Analyze raw calendar effects for reporting."""
    print(f"\n{'='*70}")
    print("  CALENDAR ANOMALY ANALYSIS")
    print(f"{'='*70}")

    daily_ret = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if (
                not np.isnan(C[si, di])
                and not np.isnan(C[si, di - 1])
                and C[si, di - 1] > 0
            ):
                daily_ret[si, di] = C[si, di] / C[si, di - 1] - 1

    print(f"\n  Month-of-Year avg return (all instruments):")
    for m in range(1, 13):
        rets = []
        for di in range(ND):
            d = dates[di]
            if isinstance(d, pd.Timestamp):
                dm = d.month
            else:
                dm = d.month
            if dm == m:
                for si in range(NS):
                    if not np.isnan(daily_ret[si, di]):
                        rets.append(daily_ret[si, di])
        if rets:
            avg = np.mean(rets) * 100
            wr = sum(1 for r in rets if r > 0) / len(rets) * 100
            print(f"    {m:>2}: avg={avg:+.4f}% WR={wr:.1f}% n={len(rets)}")

    print(f"\n  Day-of-Week avg return:")
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    for dow in range(5):
        rets = []
        for di in range(ND):
            d = dates[di]
            if isinstance(d, pd.Timestamp):
                wd = d.weekday()
            else:
                wd = d.weekday()
            if wd == dow:
                for si in range(NS):
                    if not np.isnan(daily_ret[si, di]):
                        rets.append(daily_ret[si, di])
        if rets:
            avg = np.mean(rets) * 100
            wr = sum(1 for r in rets if r > 0) / len(rets) * 100
            print(f"    {dow_names[dow]:>3}: avg={avg:+.4f}% WR={wr:.1f}%")

    month_of_year, day_of_week, days_to_month_end, is_pre_holiday = (
        compute_calendar_features(dates, ND)
    )
    pre_hol_rets = []
    normal_rets = []
    for di in range(ND):
        for si in range(NS):
            if not np.isnan(daily_ret[si, di]):
                if is_pre_holiday[di]:
                    pre_hol_rets.append(daily_ret[si, di])
                else:
                    normal_rets.append(daily_ret[si, di])

    if pre_hol_rets:
        avg_ph = np.mean(pre_hol_rets) * 100
        wr_ph = sum(1 for r in pre_hol_rets if r > 0) / len(pre_hol_rets) * 100
        avg_n = np.mean(normal_rets) * 100
        wr_n = sum(1 for r in normal_rets if r > 0) / len(normal_rets) * 100
        print(
            f"\n  Pre-holiday: avg={avg_ph:+.4f}% WR={wr_ph:.1f}% n={len(pre_hol_rets)}"
        )
        print(f"  Normal:      avg={avg_n:+.4f}% WR={wr_n:.1f}% n={len(normal_rets)}")


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("  V23: CALENDAR ANOMALY MEAN REVERSION")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start="2016-01-01")
    print(
        f"  {NS} sym, {ND} days, "
        f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}"
    )

    # --- Calendar analysis ---
    analyze_calendar_anomalies(C, NS, ND, dates)

    # --- Compute calendar features ---
    month_of_year, day_of_week, days_to_month_end, is_pre_holiday = (
        compute_calendar_features(dates, ND)
    )

    # --- Compute per-instrument calendar scores ---
    inst_cal_score = compute_instrument_calendar_score(
        C, NS, ND, dates,
        month_of_year, day_of_week, is_pre_holiday,
    )

    # Print some stats
    valid_scores = inst_cal_score[inst_cal_score != 0.5]
    if len(valid_scores) > 0:
        print(
            f"\n  Instrument calendar score stats: "
            f"min={np.min(valid_scores):.3f} "
            f"max={np.max(valid_scores):.3f} "
            f"mean={np.mean(valid_scores):.3f} "
            f"std={np.std(valid_scores):.3f}"
        )

    # --- Compute base signals ---
    sigs = compute_signals(C, O, H, L, V, OI, NS, ND)

    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # ============================================================
    # SECTION 1: BASELINE vs CALENDAR BOOST (2019-2026)
    # ============================================================
    print(f"\n{'='*70}")
    print("  SECTION 1: BASELINE vs CALENDAR BOOST (2019-2026)")
    print(f"{'='*70}")

    print("\n  --- Baseline (no calendar boost) ---")
    trades_base, eq_base, dd_base = backtest_v23(
        C, O, H, L, NS, ND, dates, syms, sigs,
        sigs["combo_rank"],
        top_n=1, hold_days=5, atr_stop=3.0,
        min_confidence=3, use_ker_gate=True,
        pyramid_ratio=0.5, start_di=bt_2019,
    )
    res_base = analyze(trades_base, eq_base, dd_base, "Baseline-noCal")

    for cal_w in [0.1, 0.2, 0.3]:
        final_rank = compute_calendar_boosted_rank(
            sigs["combo_rank"], NS, ND,
            inst_cal_score, calendar_weight=cal_w,
        )
        trades_cal, eq_cal, dd_cal = backtest_v23(
            C, O, H, L, NS, ND, dates, syms, sigs,
            final_rank,
            top_n=1, hold_days=5, atr_stop=3.0,
            min_confidence=3, use_ker_gate=True,
            pyramid_ratio=0.5, start_di=bt_2019,
        )
        analyze(trades_cal, eq_cal, dd_cal, f"CalBoost-w{cal_w}")

    # ============================================================
    # SECTION 2: PARAMETER SWEEP
    # ============================================================
    sweep_results = parameter_sweep(
        C, O, H, L, V, OI, NS, ND, dates, syms, sigs, bt_2019,
        inst_cal_score,
    )

    # ============================================================
    # SECTION 3: BEST CONFIG -- YEARLY BREAKDOWN
    # ============================================================
    print(f"\n{'='*70}")
    print("  SECTION 3: BEST CONFIGS -- YEARLY (2019-2026)")
    print(f"{'='*70}")

    for r in sweep_results[:3]:
        final_rank = compute_calendar_boosted_rank(
            sigs["combo_rank"], NS, ND,
            inst_cal_score, calendar_weight=r["cal_w"],
        )
        trades, eq, dd = backtest_v23(
            C, O, H, L, NS, ND, dates, syms, sigs,
            final_rank,
            top_n=r["tn"], hold_days=5,
            atr_stop=r["atr"], min_confidence=r["mc"],
            use_ker_gate=True, pyramid_ratio=r["pyr"],
            start_di=bt_2019,
        )
        label = (
            f"cw{r['cal_w']:.1f}_tn{r['tn']}_mc{r['mc']}"
            f"_p{r['pyr']:.1f}_a{r['atr']:.1f}"
        )
        print(f"\n  --- {label} ---")
        analyze(trades, eq, dd, label)

    # ============================================================
    # SECTION 4: WALK-FORWARD (best config)
    # ============================================================
    if sweep_results:
        best = sweep_results[0]
        wf_trades = walk_forward(
            C, O, H, L, V, OI, NS, ND, dates, syms,
            train_years=4,
            top_n=best["tn"],
            atr_stop=best["atr"],
            min_confidence=best["mc"],
            use_ker_gate=True,
            pyramid_ratio=best["pyr"],
            calendar_weight=best["cal_w"],
            hold_days=5,
        )

    # ============================================================
    # SECTION 5: FULL 10-YEAR (best configs)
    # ============================================================
    print(f"\n{'='*70}")
    print("  SECTION 5: FULL 2016-2026 (10 years)")
    print(f"{'='*70}")

    if sweep_results:
        for r in sweep_results[:2]:
            final_rank = compute_calendar_boosted_rank(
                sigs["combo_rank"], NS, ND,
                inst_cal_score, calendar_weight=r["cal_w"],
            )
            trades, eq, dd = backtest_v23(
                C, O, H, L, NS, ND, dates, syms, sigs,
                final_rank,
                top_n=r["tn"], hold_days=5,
                atr_stop=r["atr"], min_confidence=r["mc"],
                use_ker_gate=True, pyramid_ratio=r["pyr"],
                start_di=60,
            )
            label = (
                f"10yr cw{r['cal_w']:.1f}_tn{r['tn']}_mc{r['mc']}"
                f"_p{r['pyr']:.1f}_a{r['atr']:.1f}"
            )
            print(f"\n  {label}")
            analyze(trades, eq, dd, label)

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print("  SUMMARY: CALENDAR ANOMALY MEAN REVERSION")
    print(f"{'='*70}")
    if res_base:
        print(
            f"  Baseline (no cal):   ann={res_base['ann']:+.1f}%  "
            f"WR={res_base['wr']:.1f}%  "
            f"Sh={res_base['sh']:.2f}  DD={res_base['dd']:.1f}%"
        )
    if sweep_results:
        b = sweep_results[0]
        print(
            f"  Best sweep:          ann={b['ann']:+.1f}%  "
            f"WR={b['wr']:.1f}%  "
            f"Sh={b['sharpe']:.2f}  DD={b['dd']:.1f}%"
        )
        print(
            f"    config: cal_w={b['cal_w']:.1f} tn={b['tn']} "
            f"mc={b['mc']} pyr={b['pyr']:.1f} atr={b['atr']:.1f}"
        )

    print(f"\n[V23] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
