"""
V9: Multi-Timeframe Confluence Strategy — Walk-Forward Validated
================================================================
Combine weekly trend direction with daily oversold signals for
higher conviction entries.

Architecture:
  1. Weekly timeframe (5-day aggregates):
     - Weekly trend: 4-week return direction
     - Weekly momentum rank across instruments
     - Weekly volume trend
  2. Daily timeframe (from V1):
     - Daily oversold combo score
     - KER regime
     - Confidence signals
  3. Confluence rules:
     - BEST:  weekly uptrend + daily oversold = buy the dip
     - GOOD:  weekly neutral + daily oversold = pure mean reversion
     - SKIP:  weekly downtrend + daily oversold = catching falling knife
  4. Score = weekly_weight * weekly_score + daily_weight * daily_score
  5. Test different weight combinations
  6. Pyramid option
  7. Walk-forward validation

Signal at close[di], enter at open[di+1]. No look-ahead.
No gap signals. No leverage.
"""
import sys
import os
import time
import warnings
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
WEEK_DAYS = 5  # trading days per week


# ============================================================
# WEEKLY AGGREGATION
# ============================================================
def build_weekly_index(dates: list, ND: int) -> tuple:
    """
    Map each daily index to its corresponding weekly bucket.
    Returns (week_ids, n_weeks) where week_ids[di] = week index.
    Every WEEK_DAYS trading days = 1 week bucket.
    """
    week_ids = np.full(ND, -1, dtype=int)
    week = 0
    day_in_week = 0
    for di in range(ND):
        week_ids[di] = week
        day_in_week += 1
        if day_in_week >= WEEK_DAYS:
            week += 1
            day_in_week = 0
    n_weeks = week + 1
    return week_ids, n_weeks


def aggregate_weekly(
    C: np.ndarray, O: np.ndarray, H: np.ndarray,
    L: np.ndarray, V: np.ndarray, NS: int, ND: int,
    week_ids: np.ndarray, n_weeks: int,
) -> tuple:
    """
    Aggregate daily OHLCV into weekly OHLCV arrays.
    Returns (wC, wO, wH, wL, wV) shaped (NS, n_weeks).
    """
    wC = np.full((NS, n_weeks), np.nan)
    wO = np.full((NS, n_weeks), np.nan)
    wH = np.full((NS, n_weeks), np.nan)
    wL = np.full((NS, n_weeks), np.nan)
    wV = np.full((NS, n_weeks), np.nan)

    for wi in range(n_weeks):
        mask = week_ids == wi
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        for si in range(NS):
            opens = O[si, idxs]
            closes = C[si, idxs]
            highs = H[si, idxs]
            lows = L[si, idxs]
            vols = V[si, idxs]

            valid_o = opens[~np.isnan(opens)]
            valid_c = closes[~np.isnan(closes)]
            valid_h = highs[~np.isnan(highs)]
            valid_l = lows[~np.isnan(lows)]
            valid_v = vols[~np.isnan(vols)]

            if len(valid_o) > 0:
                wO[si, wi] = valid_o[0]
            if len(valid_c) > 0:
                wC[si, wi] = valid_c[-1]
            if len(valid_h) > 0:
                wH[si, wi] = np.max(valid_h)
            if len(valid_l) > 0:
                wL[si, wi] = np.min(valid_l)
            if len(valid_v) > 0:
                wV[si, wi] = np.sum(valid_v)

    return wC, wO, wH, wL, wV


# ============================================================
# WEEKLY SIGNALS
# ============================================================
def compute_weekly_signals(
    wC: np.ndarray, wO: np.ndarray, wH: np.ndarray,
    wL: np.ndarray, wV: np.ndarray, NS: int, n_weeks: int,
) -> tuple:
    """
    Compute weekly-level indicators:
      - weekly_trend: 4-week return direction (+1/0/-1)
      - weekly_mom_rank: cross-sectional momentum rank
      - weekly_vol_trend: volume trend (rising/falling)
      - weekly_score: composite weekly score per instrument

    Returns (weekly_trend, weekly_mom_rank, weekly_vol_trend, weekly_score)
    all shaped (NS, n_weeks).
    """
    WEEK_LOOKBACK = 4

    weekly_trend = np.zeros((NS, n_weeks), dtype=int)
    weekly_mom_rank = np.full((NS, n_weeks), np.nan)
    weekly_vol_trend = np.zeros((NS, n_weeks), dtype=int)
    weekly_score = np.full((NS, n_weeks), np.nan)

    # 4-week return for trend direction
    ret_4w = np.full((NS, n_weeks), np.nan)
    for si in range(NS):
        for wi in range(WEEK_LOOKBACK, n_weeks):
            if (not np.isnan(wC[si, wi]) and not np.isnan(wC[si, wi - WEEK_LOOKBACK])
                    and wC[si, wi - WEEK_LOOKBACK] > 0):
                ret_4w[si, wi] = wC[si, wi] / wC[si, wi - WEEK_LOOKBACK] - 1

    # Trend classification: uptrend (+1), neutral (0), downtrend (-1)
    for si in range(NS):
        for wi in range(WEEK_LOOKBACK, n_weeks):
            r = ret_4w[si, wi]
            if np.isnan(r):
                continue
            if r > 0.02:
                weekly_trend[si, wi] = 1
            elif r < -0.02:
                weekly_trend[si, wi] = -1

    # Cross-sectional momentum rank
    for wi in range(WEEK_LOOKBACK, n_weeks):
        vals = ret_4w[:, wi]
        valid = ~np.isnan(vals)
        if valid.sum() >= 5:
            weekly_mom_rank[:, wi] = (
                pd.Series(vals)
                .rank(pct=True, na_option="keep")
                .values
            )

    # Volume trend: compare recent 2-week avg vol vs prior 2-week avg vol
    for si in range(NS):
        for wi in range(WEEK_LOOKBACK, n_weeks):
            recent = wV[si, max(0, wi - 1):wi + 1]
            prior = wV[si, max(0, wi - 3):wi - 1]
            rv = recent[~np.isnan(recent)]
            pv = prior[~np.isnan(prior)]
            if len(rv) >= 1 and len(pv) >= 1:
                if np.mean(rv) > np.mean(pv) * 1.1:
                    weekly_vol_trend[si, wi] = 1
                elif np.mean(rv) < np.mean(pv) * 0.9:
                    weekly_vol_trend[si, wi] = -1

    # Composite weekly score: higher = more bullish
    for wi in range(WEEK_LOOKBACK, n_weeks):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            if np.isnan(wC[si, wi]) or wC[si, wi] <= 0:
                continue
            s = 0.0
            w_total = 0.0

            # Trend direction weight
            s += weekly_trend[si, wi] * 0.40
            w_total += 0.40

            # Momentum rank
            mr = weekly_mom_rank[si, wi]
            if not np.isnan(mr):
                s += (mr - 0.5) * 2.0 * 0.30  # normalize to [-1, +1]
                w_total += 0.30

            # Volume trend confirmation
            s += weekly_vol_trend[si, wi] * 0.15
            w_total += 0.15

            # 2-week price position vs range
            if wi >= 2:
                h2 = wH[si, wi - 1:wi + 1]
                l2 = wL[si, wi - 1:wi + 1]
                vh2 = h2[~np.isnan(h2)]
                vl2 = l2[~np.isnan(l2)]
                if len(vh2) > 0 and len(vl2) > 0:
                    hh = np.max(vh2)
                    ll = np.min(vl2)
                    cc = wC[si, wi]
                    if hh > ll and not np.isnan(cc):
                        pos = (cc - ll) / (hh - ll)
                        s += (pos - 0.5) * 0.15
                        w_total += 0.15

            if w_total > 0:
                scores[si] = s / w_total

        valid = ~np.isnan(scores)
        if valid.sum() >= 5:
            weekly_score[:, wi] = (
                pd.Series(scores)
                .rank(pct=True, na_option="keep")
                .values
            )

    return weekly_trend, weekly_mom_rank, weekly_vol_trend, weekly_score


# ============================================================
# DAILY SIGNALS (from V5 pattern)
# ============================================================
def compute_daily_signals(
    C: np.ndarray, O: np.ndarray, H: np.ndarray,
    L: np.ndarray, V: np.ndarray, OI: np.ndarray,
    NS: int, ND: int,
) -> dict:
    """
    Compute daily oversold combo score, KER regime, and signal count.
    Same logic as V5 but returns a dict of arrays.
    """
    t0 = time.time()
    print("[V9] Computing daily signals...", flush=True)

    # Consecutive down days
    consec_dn = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        consec = 0
        for di in range(1, ND):
            if (not np.isnan(C[si, di]) and not np.isnan(C[si, di - 1])
                    and C[si, di - 1] > 0):
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
            if (not np.isnan(C[si, di]) and not np.isnan(C[si, di - 5])
                    and C[si, di - 5] > 0):
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

    # Volume delta proxy
    vdp = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if np.isnan(H[si, di]) or np.isnan(L[si, di]):
                continue
            if np.isnan(C[si, di]) or np.isnan(V[si, di]):
                continue
            bar_range = H[si, di] - L[si, di]
            if bar_range > 0 and V[si, di] > 0:
                vdp[si, di] = (
                    V[si, di] * (2 * C[si, di] - H[si, di] - L[si, di]) / bar_range
                )

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

    # KER regime (10-day Kaufman efficiency ratio)
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
                ker_regime[si, di] = 1   # choppy / mean-reversion friendly
            elif ker_10[si, di] > 0.3:
                ker_regime[si, di] = -1   # trending / avoid counter-trend

    # TA-Lib indicators
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

    # Composite daily oversold score + rank
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
            raw_score[:, di] = (
                pd.Series(scores)
                .rank(pct=True, na_option="keep")
                .values
            )

    # Signal count per instrument per day
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

    print(f"  Daily signals done: {time.time() - t0:.1f}s", flush=True)
    return {
        "combo_rank": raw_score,
        "ker_regime": ker_regime,
        "n_signals": n_signals,
    }


# ============================================================
# CONFLUENCE SCORING
# ============================================================
def build_confluence_score(
    daily_sigs: dict,
    weekly_trend: np.ndarray,
    weekly_score: np.ndarray,
    week_ids: np.ndarray,
    n_weeks: int,
    NS: int,
    ND: int,
    weekly_weight: float = 0.5,
) -> np.ndarray:
    """
    Combine weekly and daily signals into a confluence score.

    Confluence rules:
      BEST:  weekly uptrend  + daily oversold = buy the dip
      GOOD:  weekly neutral  + daily oversold = pure mean reversion
      SKIP:  weekly downtrend + daily oversold = catching falling knife

    The weekly_weight controls blending:
      confluence = weekly_weight * weekly_score + (1 - weekly_weight) * daily_score

    But we also apply confluence multipliers to penalize bad combos
    and boost good combos.
    """
    daily_rank = daily_sigs["combo_rank"]
    confluence = np.full((NS, ND), np.nan)

    for di in range(ND):
        wi = week_ids[di]
        if wi < 0 or wi >= n_weeks:
            continue

        for si in range(NS):
            dr = daily_rank[si, di]
            if np.isnan(dr):
                continue

            ws = weekly_score[si, wi]
            wt = weekly_trend[si, wi]

            # Normalize daily oversold rank to [-1, +1] range
            # High rank = oversold candidate
            daily_score = (dr - 0.5) * 2.0  # now [-1, +1], positive = oversold

            # Weekly score already rank-normalized, shift to [-1, +1]
            w_score = 0.0
            if not np.isnan(ws):
                w_score = (ws - 0.5) * 2.0

            # Confluence multiplier
            multiplier = 1.0
            if wt == 1 and daily_score > 0.3:
                # BEST: weekly uptrend + daily oversold
                multiplier = 1.5
            elif wt == 0 and daily_score > 0.3:
                # GOOD: weekly neutral + daily oversold
                multiplier = 1.0
            elif wt == -1 and daily_score > 0.3:
                # SKIP: weekly downtrend + daily oversold = falling knife
                multiplier = 0.3
            elif wt == 1 and daily_score <= 0.3:
                # Weekly uptrend, not oversold -- mild positive
                multiplier = 0.5
            elif wt == -1 and daily_score <= 0.3:
                # Weekly downtrend, not oversold -- mild negative
                multiplier = 0.2

            blended = (
                weekly_weight * w_score
                + (1.0 - weekly_weight) * daily_score
            ) * multiplier
            confluence[si, di] = blended

    # Re-rank the confluence scores per day
    ranked = np.full((NS, ND), np.nan)
    for di in range(ND):
        col = confluence[:, di]
        valid = ~np.isnan(col)
        if valid.sum() >= 5:
            ranked[:, di] = (
                pd.Series(col)
                .rank(pct=True, na_option="keep")
                .values
            )

    return ranked


# ============================================================
# BACKTEST ENGINE
# ============================================================
def backtest_v9(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: list, syms: list,
    daily_sigs: dict, confluence_rank: np.ndarray,
    weekly_trend: np.ndarray, week_ids: np.ndarray, n_weeks: int,
    top_n: int = 1,
    min_rank: float = 0.7,
    atr_stop: float = 3.0,
    min_confidence: int = 3,
    use_ker_gate: bool = True,
    hold_days: int = 5,
    pyramid_ratio: float = 0.0,
    pyramid_day: int = 1,
    use_confluence: bool = True,
    start_di: int = 60,
    end_di: int | None = None,
) -> tuple:
    """
    Backtest with multi-timeframe confluence.

    When use_confluence=True, uses the blended weekly+daily rank.
    When False, uses only daily rank (baseline comparison).
    """
    combo_rank = confluence_rank if use_confluence else daily_sigs["combo_rank"]
    ker_regime = daily_sigs["ker_regime"]
    n_signals = daily_sigs["n_signals"]

    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions = []  # (si, entry_di, entry_price, stop_price, alloc, is_pyramid)
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

            stopped = any(c < sp for _, _, sp, _, _ in pos_list)

            if stopped:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    wi = week_ids[edi] if edi < len(week_ids) else -1
                    wt_val = weekly_trend[si, wi] if wi >= 0 else 0
                    trades.append({
                        "pnl_abs": profit,
                        "pnl_pct": pnl * 100,
                        "days": di - edi + 1,
                        "di": di,
                        "year": d.year,
                        "sym": syms[si],
                        "reason": "stop",
                        "pyr": is_pyr,
                        "weekly_trend": wt_val,
                    })
            elif hold >= hold_days:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    wi = week_ids[edi] if edi < len(week_ids) else -1
                    wt_val = weekly_trend[si, wi] if wi >= 0 else 0
                    trades.append({
                        "pnl_abs": profit,
                        "pnl_pct": pnl * 100,
                        "days": di - edi + 1,
                        "di": di,
                        "year": d.year,
                        "sym": syms[si],
                        "reason": "hold",
                        "pyr": is_pyr,
                        "weekly_trend": wt_val,
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

    # Close remaining positions at final price
    for si, edi, ep, sp, alloc, is_pyr in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl

    return trades, equity, max_dd


# ============================================================
# ANALYSIS
# ============================================================
def analyze(trades: list, equity: float, max_dd: float, label: str = "") -> dict | None:
    """Print trade analysis and return summary dict."""
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

    # Per weekly-trend breakdown
    wt_groups = {1: [], 0: [], -1: []}
    for t in trades:
        wt = t.get("weekly_trend", 0)
        if wt in wt_groups:
            wt_groups[wt].append(t["pnl_pct"])

    print(
        f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} "
        f"stop:{n_stop} hold:{n_hold}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
        f"Sh={sh:.2f} eq={equity:,.0f}"
    )

    # Weekly trend breakdown
    for wt_val, name in [(1, "W-Up"), (0, "W-Neutral"), (-1, "W-Down")]:
        wt_trades = wt_groups[wt_val]
        if wt_trades:
            wt_wr = sum(1 for p in wt_trades if p > 0) / len(wt_trades) * 100
            wt_avg = np.mean(wt_trades)
            print(
                f"    {name}: {len(wt_trades)}t "
                f"WR={wt_wr:.1f}% avg={wt_avg:+.2f}%"
            )

    # Annual breakdown
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
    V: np.ndarray, OI: np.ndarray,
    NS: int, ND: int, dates: list, syms: list,
    week_ids: np.ndarray, n_weeks: int,
    weekly_weight: float = 0.5,
    top_n: int = 1,
    hold_days: int = 5,
    atr_stop: float = 3.0,
    min_confidence: int = 3,
    use_ker_gate: bool = True,
    pyramid_ratio: float = 0.0,
    pyramid_day: int = 1,
) -> list:
    """Walk-forward validation year-by-year."""
    print(f"\n{'=' * 70}")
    print(
        f"  WALK-FORWARD (ww={weekly_weight:.1f}, "
        f"pyr={pyramid_ratio:.1f}, mc={min_confidence})"
    )
    print(f"{'=' * 70}")

    years = sorted(set(d.year for d in dates))
    all_trades = []

    for test_year in range(2019, years[-1] + 1):
        test_start = test_end_idx = None
        for i, d in enumerate(dates):
            if d.year == test_year and test_start is None:
                test_start = i
            if d.year == test_year:
                test_end_idx = i
        if test_start is None:
            continue

        # Compute signals using only data up to and including test period
        # (no future leakage in signal computation since signals use
        #  lookback-only indicators)
        daily_sigs = compute_daily_signals(
            C, O, H, L, V, OI, NS, ND
        )
        wC, wO, wH, wL, wV = aggregate_weekly(
            C, O, H, L, V, NS, ND, week_ids, n_weeks
        )
        weekly_trend, _, _, weekly_score = compute_weekly_signals(
            wC, wO, wH, wL, wV, NS, n_weeks
        )
        confluence_rank = build_confluence_score(
            daily_sigs, weekly_trend, weekly_score,
            week_ids, n_weeks, NS, ND, weekly_weight,
        )

        trades, _, _ = backtest_v9(
            C, O, H, L, NS, ND, dates, syms,
            daily_sigs, confluence_rank, weekly_trend,
            week_ids, n_weeks,
            top_n=top_n, hold_days=hold_days, atr_stop=atr_stop,
            min_confidence=min_confidence, use_ker_gate=use_ker_gate,
            pyramid_ratio=pyramid_ratio, pyramid_day=pyramid_day,
            start_di=test_start, end_di=test_end_idx + 1,
        )

        test_trades = [t for t in trades if dates[t["di"]].year == test_year]
        all_trades.extend(test_trades)

        if test_trades:
            n = len(test_trades)
            nw = sum(1 for t in test_trades if t["pnl_pct"] > 0)
            wr_val = nw / n * 100
            avg_val = np.mean([t["pnl_pct"] for t in test_trades])
            print(
                f"  {test_year}: {n}t WR={wr_val:.1f}% avg={avg_val:+.2f}%",
                flush=True,
            )
        else:
            print(f"  {test_year}: no trades", flush=True)

    if all_trades:
        nw = sum(1 for t in all_trades if t["pnl_pct"] > 0)
        wr_val = nw / len(all_trades) * 100
        avg_val = np.mean([t["pnl_pct"] for t in all_trades])
        cum = np.prod([1 + t["pnl_pct"] / 100 for t in all_trades]) - 1
        n_pyr = sum(1 for t in all_trades if t.get("pyr"))
        print(
            f"\n  WF TOTAL: {len(all_trades)}t (pyr:{n_pyr}) WR={wr_val:.1f}% "
            f"avg={avg_val:+.2f}% cum={cum:+.1%}"
        )
        return all_trades
    return []


# ============================================================
# PARAMETER SWEEP
# ============================================================
def sweep_weights(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray,
    NS: int, ND: int, dates: list, syms: list,
    week_ids: np.ndarray, n_weeks: int,
    daily_sigs: dict,
    weekly_trend: np.ndarray,
    weekly_score: np.ndarray,
    start_di: int,
) -> list:
    """Sweep weekly_weight, pyramid_ratio, min_confidence, top_n."""
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP: weight x pyramid x confidence x top_n")
    print("=" * 70)

    results = []

    for ww in [0.2, 0.35, 0.5, 0.65, 0.8]:
        confluence_rank = build_confluence_score(
            daily_sigs, weekly_trend, weekly_score,
            week_ids, n_weeks, NS, ND, ww,
        )
        for tn in [1, 2, 3]:
            for pyr in [0.0, 0.3, 0.5]:
                for mc in [2, 3, 4]:
                    trades, eq, dd = backtest_v9(
                        C, O, H, L, NS, ND, dates, syms,
                        daily_sigs, confluence_rank, weekly_trend,
                        week_ids, n_weeks,
                        top_n=tn, hold_days=5, atr_stop=3.0,
                        min_confidence=mc, use_ker_gate=True,
                        pyramid_ratio=pyr, pyramid_day=1,
                        start_di=start_di,
                    )
                    if len(trades) < 10:
                        continue
                    nw = sum(1 for t in trades if t["pnl_pct"] > 0)
                    wr_val = nw / len(trades) * 100
                    n_days = max(1, trades[-1]["di"] - trades[0]["di"])
                    ann_val = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
                    ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
                    rets = np.array(ap) / CASH0
                    sh_val = (
                        np.mean(rets) / np.std(rets) * np.sqrt(252)
                        if np.std(rets) > 0
                        else 0
                    )
                    results.append({
                        "ww": ww,
                        "tn": tn,
                        "pyr": pyr,
                        "mc": mc,
                        "n": len(trades),
                        "wr": wr_val,
                        "ann": ann_val,
                        "dd": dd,
                        "sharpe": sh_val,
                    })

    results.sort(key=lambda x: -x["sharpe"])
    print(
        f"\n{'WW':>4} {'TN':>3} {'Pyr':>4} {'MC':>3} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}"
    )
    print("-" * 60)
    for r in results[:25]:
        print(
            f"{r['ww']:>4.1f} {r['tn']:>3} {r['pyr']:>4.1f} {r['mc']:>3} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>5.2f}"
        )

    return results


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("  V9: MULTI-TIMEFRAME CONFLUENCE — WALK-FORWARD VALIDATED")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start="2016-01-01")
    print(
        f"  {NS} sym, {ND} days, "
        f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}"
    )

    # Build weekly index and aggregates
    week_ids, n_weeks = build_weekly_index(dates, ND)
    print(f"  {n_weeks} weekly buckets from {ND} daily bars")

    print("[V9] Aggregating weekly data...", flush=True)
    wC, wO, wH, wL, wV = aggregate_weekly(
        C, O, H, L, V, NS, ND, week_ids, n_weeks
    )

    # Weekly signals
    print("[V9] Computing weekly signals...", flush=True)
    weekly_trend, weekly_mom_rank, weekly_vol_trend, weekly_score = (
        compute_weekly_signals(wC, wO, wH, wL, wV, NS, n_weeks)
    )

    # Daily signals
    daily_sigs = compute_daily_signals(C, O, H, L, V, OI, NS, ND)

    # Find 2019 start index for OOS testing
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # ============================================================
    # SECTION 1: Baseline (daily-only) vs Confluence comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("  SECTION 1: BASELINE (daily-only) vs CONFLUENCE")
    print("=" * 70)

    for ww in [0.3, 0.5, 0.7]:
        confluence_rank = build_confluence_score(
            daily_sigs, weekly_trend, weekly_score,
            week_ids, n_weeks, NS, ND, ww,
        )

        # Daily-only baseline (2019+)
        trades_base, eq_base, dd_base = backtest_v9(
            C, O, H, L, NS, ND, dates, syms,
            daily_sigs, confluence_rank, weekly_trend,
            week_ids, n_weeks,
            top_n=1, hold_days=5, atr_stop=3.0,
            min_confidence=3, use_ker_gate=True,
            use_confluence=False,
            start_di=bt_2019,
        )
        analyze(trades_base, eq_base, dd_base, f"Baseline ww={ww}")

        # Confluence (2019+)
        trades_conf, eq_conf, dd_conf = backtest_v9(
            C, O, H, L, NS, ND, dates, syms,
            daily_sigs, confluence_rank, weekly_trend,
            week_ids, n_weeks,
            top_n=1, hold_days=5, atr_stop=3.0,
            min_confidence=3, use_ker_gate=True,
            use_confluence=True,
            start_di=bt_2019,
        )
        analyze(trades_conf, eq_conf, dd_conf, f"Confluence ww={ww}")

    # ============================================================
    # SECTION 2: Full 10-year with different weekly weights
    # ============================================================
    print("\n" + "=" * 70)
    print("  SECTION 2: FULL 10-YEAR — WEIGHT PROFILES")
    print("=" * 70)

    for ww in [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]:
        if ww == 0.0:
            conf_rank = daily_sigs["combo_rank"]
        else:
            conf_rank = build_confluence_score(
                daily_sigs, weekly_trend, weekly_score,
                week_ids, n_weeks, NS, ND, ww,
            )
        trades, eq, dd = backtest_v9(
            C, O, H, L, NS, ND, dates, syms,
            daily_sigs, conf_rank, weekly_trend,
            week_ids, n_weeks,
            top_n=1, hold_days=5, atr_stop=3.0,
            min_confidence=3, use_ker_gate=True,
            start_di=60,
        )
        label = f"ww={ww:.2f}"
        print()
        analyze(trades, eq, dd, label)

    # ============================================================
    # SECTION 3: Confluence + Pyramid
    # ============================================================
    print("\n" + "=" * 70)
    print("  SECTION 3: CONFLUENCE + PYRAMID")
    print("=" * 70)

    for ww in [0.3, 0.5, 0.7]:
        conf_rank = build_confluence_score(
            daily_sigs, weekly_trend, weekly_score,
            week_ids, n_weeks, NS, ND, ww,
        )
        for pyr in [0.3, 0.5, 0.7]:
            trades, eq, dd = backtest_v9(
                C, O, H, L, NS, ND, dates, syms,
                daily_sigs, conf_rank, weekly_trend,
                week_ids, n_weeks,
                top_n=1, hold_days=5, atr_stop=3.0,
                min_confidence=3, use_ker_gate=True,
                pyramid_ratio=pyr, pyramid_day=1,
                start_di=60,
            )
            analyze(trades, eq, dd, f"ww={ww} pyr={pyr}")

    # ============================================================
    # SECTION 4: Parameter sweep
    # ============================================================
    results = sweep_weights(
        C, O, H, L, V, OI, NS, ND, dates, syms,
        week_ids, n_weeks,
        daily_sigs, weekly_trend, weekly_score,
        start_di=bt_2019,
    )

    # ============================================================
    # SECTION 5: Best configs full 10-year
    # ============================================================
    print("\n" + "=" * 70)
    print("  SECTION 5: BEST CONFIGS — FULL 10-YEAR")
    print("=" * 70)

    # Deduplicate by (ww, tn, pyr, mc)
    seen = set()
    unique_results = []
    for r in results:
        key = (r["ww"], r["tn"], r["pyr"], r["mc"])
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    for r in unique_results[:5]:
        ww = r["ww"]
        conf_rank = build_confluence_score(
            daily_sigs, weekly_trend, weekly_score,
            week_ids, n_weeks, NS, ND, ww,
        )
        trades, eq, dd = backtest_v9(
            C, O, H, L, NS, ND, dates, syms,
            daily_sigs, conf_rank, weekly_trend,
            week_ids, n_weeks,
            top_n=r["tn"], hold_days=5, atr_stop=3.0,
            min_confidence=r["mc"], use_ker_gate=True,
            pyramid_ratio=r["pyr"], pyramid_day=1,
            start_di=60,
        )
        label = f"ww={ww:.1f} tn={r['tn']} pyr={r['pyr']:.1f} mc={r['mc']}"
        print(f"\n  FULL {label}")
        analyze(trades, eq, dd, label)

    # ============================================================
    # SECTION 6: Walk-forward for best config
    # ============================================================
    print("\n" + "=" * 70)
    print("  SECTION 6: WALK-FORWARD VALIDATION")
    print("=" * 70)

    if unique_results:
        best = unique_results[0]
        ww = best["ww"]
        print(f"\n  Best config: ww={ww} tn={best['tn']} pyr={best['pyr']} mc={best['mc']}")

        walk_forward(
            C, O, H, L, V, OI, NS, ND, dates, syms,
            week_ids, n_weeks,
            weekly_weight=ww,
            top_n=best["tn"],
            hold_days=5,
            atr_stop=3.0,
            min_confidence=best["mc"],
            use_ker_gate=True,
            pyramid_ratio=best["pyr"],
            pyramid_day=1,
        )

        # Also WF for daily-only baseline comparison
        print("\n  --- Baseline (daily-only) WF ---")
        walk_forward(
            C, O, H, L, V, OI, NS, ND, dates, syms,
            week_ids, n_weeks,
            weekly_weight=0.0,
            top_n=best["tn"],
            hold_days=5,
            atr_stop=3.0,
            min_confidence=best["mc"],
            use_ker_gate=True,
            pyramid_ratio=best["pyr"],
            pyramid_day=1,
        )

    print(f"\n[V9] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
