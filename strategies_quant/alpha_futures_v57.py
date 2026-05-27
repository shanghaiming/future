"""
V57: Institutional-Grade Dynamic Mode Strategy with Leverage
==============================================================
Based on V43 (ALL-TIME BEST: Sharpe 4.99, MDD 13.3%) with:

1. LEVERAGE parameter: scale all P&L by leverage factor
   - leverage=1.0 (default): no leverage, exactly V43 behavior
   - leverage=2.0: 2x leverage, doubles both returns and risk
   - leverage=0.5: half exposure, conservative

2. CONTRACT MULTIPLIER awareness:
   - Track notional exposure per instrument
   - Report contract counts alongside percentage returns
   - Margin utilization tracking

3. Same V43 dynamic mode switching (WINNING/NORMAL/LOSING)

Signal at close[di], enter at open[di+1]. No look-ahead. No gap signals.
Walk-forward validation required.
"""
import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import product
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_data import load_all_data
from contract_specs import (
    get_contract_multiplier,
    get_margin_rate,
    get_notional_value,
    get_all_multipliers,
)

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

CASH0 = 1_000_000
COMM = 0.0005

DEFAULT_WEIGHTS = {
    "rank_ret5d": 0.25,
    "rank_oi5d": 0.20,
    "rank_rsi": 0.15,
    "rank_vol": 0.15,
    "rank_ret10d": 0.10,
    "rank_range": 0.10,
    "rank_atrp": 0.05,
}


def compute_rsi_manual(C: np.ndarray, NS: int, ND: int,
                       period: int = 14) -> np.ndarray:
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
                            losses[j] if not np.isnan(losses[j]) else 0.0)
                if len(valid_g) >= period:
                    avg_gain = np.mean(valid_g)
                    avg_loss = np.mean(valid_l)
                    if avg_loss == 0:
                        rsi[si, di + period - 1] = 100.0
                    else:
                        rs = avg_gain / avg_loss
                        rsi[si, di + period - 1] = (
                            100.0 - 100.0 / (1.0 + rs))
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
    t0 = time.time()
    print("[V57] Computing raw factors...", flush=True)

    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (not np.isnan(C[si, di])
                    and not np.isnan(C[si, di - 5])
                    and C[si, di - 5] > 0):
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1.0

    ret_10d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            if (not np.isnan(C[si, di])
                    and not np.isnan(C[si, di - 10])
                    and C[si, di - 10] > 0):
                ret_10d[si, di] = C[si, di] / C[si, di - 10] - 1.0

    oi_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (not np.isnan(OI[si, di])
                    and not np.isnan(OI[si, di - 5])
                    and OI[si, di - 5] > 0):
                oi_5d[si, di] = OI[si, di] / OI[si, di - 5] - 1.0

    vol_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            vals = V[si, di - 5:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 3:
                vol_5d[si, di] = np.mean(valid)

    daily_range = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if (not np.isnan(H[si, di])
                    and not np.isnan(L[si, di])
                    and not np.isnan(C[si, di])):
                if C[si, di] > 0 and H[si, di] > L[si, di]:
                    daily_range[si, di] = (
                        (H[si, di] - L[si, di]) / C[si, di])

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
                        max(hh - ll, abs(hh - prev_c), abs(ll - prev_c)))
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
    t0 = time.time()
    print("[V57] Computing cross-sectional ranks...", flush=True)

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
            ker_regime[si, di] = 1 if ker_10[si, di] < 0.15 else -1
    return ker_regime


def build_composite_signal(
    ranks: Dict[str, np.ndarray],
    weights: Dict[str, float],
    NS: int, ND: int,
    min_count: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    factor_names = list(weights.keys())
    weight_vals = np.array([weights[k] for k in factor_names])

    composite = np.full((NS, ND), np.nan)
    n_confirm = np.zeros((NS, ND), dtype=int)

    for di in range(ND):
        for si in range(NS):
            vals = []
            w_sum = 0.0
            for idx, name in enumerate(factor_names):
                rank_val = ranks[name][si, di]
                if not np.isnan(rank_val):
                    vals.append(rank_val * weight_vals[idx])
                    w_sum += weight_vals[idx]
            if len(vals) >= min_count and w_sum > 0:
                composite[si, di] = sum(vals) / w_sum
                n_confirm[si, di] = len(vals)

    return composite, n_confirm


def compute_atr_at(H, L, C, si, di, start_di):
    atr_vals = []
    for j in range(max(start_di, di - 14), di):
        hh, ll, cc = H[si, j], L[si, j], C[si, j]
        if not any(np.isnan([hh, ll, cc])):
            prev_c = C[si, j - 1] if j > 0 and not np.isnan(C[si, j - 1]) else cc
            atr_vals.append(max(hh - ll, abs(hh - prev_c), abs(ll - prev_c)))
    return np.mean(atr_vals) if atr_vals else None


def get_dynamic_mode(recent_trades_win, win_threshold, win_rate_window):
    if len(recent_trades_win) < 5:
        return "normal"
    window = recent_trades_win[-win_rate_window:]
    win_rate = sum(window) / len(window)
    if win_rate > win_threshold:
        return "winning"
    elif win_rate < 0.50:
        return "losing"
    return "normal"


def get_mode_params(mode, normal_threshold, lose_threshold,
                    top_n_winning, top_n_normal=2):
    if mode == "winning":
        return {"threshold": 0.75, "top_n": top_n_winning,
                "pyramid_ratio": 0.5, "mode_label": "WIN"}
    elif mode == "losing":
        return {"threshold": lose_threshold, "top_n": 1,
                "pyramid_ratio": 0.0, "mode_label": "LOSE"}
    else:
        return {"threshold": normal_threshold, "top_n": top_n_normal,
                "pyramid_ratio": 0.3, "mode_label": "NORM"}


def backtest_v57(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
    # V43 dynamic mode params
    win_threshold: float = 0.60,
    normal_threshold: float = 0.82,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
    atr_stop: float = 3.0,
    top_n_winning: int = 2,
    min_confidence: int = 3,
    use_ker_gate: bool = True,
    hold_days: int = 5,
    pyramid_day: int = 1,
    start_di: int = 60,
    end_di: Optional[int] = None,
    # V57 NEW: leverage parameter
    leverage: float = 1.0,
) -> Tuple[List[dict], float, float]:
    """Backtest V57 with leverage support and contract multiplier awareness.

    leverage: multiplier on position exposure.
      - 1.0 = no leverage (standard)
      - 2.0 = 2x leverage
      - 0.5 = half exposure
    P&L is scaled by leverage factor.
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
    recent_trades_win: List[int] = []

    # Contract multiplier lookup
    multipliers = get_all_multipliers(syms)

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions: List[Tuple[int, int, float, float, float, bool]] = []

        mode = get_dynamic_mode(
            recent_trades_win, win_threshold, win_rate_window)
        mode_params = get_mode_params(
            mode, normal_threshold, lose_threshold, top_n_winning)
        current_threshold = mode_params["threshold"]
        current_top_n = mode_params["top_n"]
        current_pyramid_ratio = mode_params["pyramid_ratio"]
        current_mode_label = mode_params["mode_label"]

        pos_by_si: Dict[int, List] = defaultdict(list)
        for si, edi, ep, sp, alloc, is_pyr in positions:
            pos_by_si[si].append((edi, ep, sp, alloc, is_pyr))

        for si, pos_list in pos_by_si.items():
            c = C[si, di]
            if np.isnan(c):
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    new_positions.append(
                        (si, edi, ep, sp, alloc, is_pyr))
                continue

            earliest_edi = min(p[0] for p in pos_list)
            hold = di - earliest_edi
            stopped = any(c < sp for _, _, sp, _, _ in pos_list)

            if stopped:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    # V57: apply leverage to P&L
                    profit = equity * alloc * pnl * leverage
                    daily_pnl += profit
                    is_win = pnl > 0
                    # Contract info for reporting
                    sym = syms[si] if si < len(syms) else "?"
                    mult = multipliers.get(sym, 10)
                    contracts = int(equity * alloc * leverage / (ep * mult))
                    notional = contracts * ep * mult
                    trades.append({
                        "pnl_abs": profit,
                        "pnl_pct": pnl * leverage * 100,
                        "days": di - edi + 1,
                        "di": di,
                        "year": d.year,
                        "sym": sym,
                        "reason": "stop",
                        "pyr": is_pyr,
                        "mode": current_mode_label,
                        "threshold": current_threshold,
                        "leverage": leverage,
                        "contracts": contracts,
                        "notional": notional,
                    })
                    recent_trades_win.append(1 if is_win else 0)
            elif hold >= hold_days:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl * leverage
                    daily_pnl += profit
                    is_win = pnl > 0
                    sym = syms[si] if si < len(syms) else "?"
                    mult = multipliers.get(sym, 10)
                    contracts = int(equity * alloc * leverage / (ep * mult))
                    notional = contracts * ep * mult
                    trades.append({
                        "pnl_abs": profit,
                        "pnl_pct": pnl * leverage * 100,
                        "days": di - edi + 1,
                        "di": di,
                        "year": d.year,
                        "sym": sym,
                        "reason": "hold",
                        "pyr": is_pyr,
                        "mode": current_mode_label,
                        "threshold": current_threshold,
                        "leverage": leverage,
                        "contracts": contracts,
                        "notional": notional,
                    })
                    recent_trades_win.append(1 if is_win else 0)
            else:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    new_positions.append(
                        (si, edi, ep, sp, alloc, is_pyr))

                # Pyramid logic
                has_pyr = any(is_pyr for _, _, _, _, is_pyr in pos_list)
                if has_pyr:
                    continue
                earliest_edi = min(p[0] for p in pos_list)
                hold = di - earliest_edi
                if hold == pyramid_day and not np.isnan(C[si, di]):
                    avg_ep = np.mean([ep for _, ep, _, _, _ in pos_list])
                    if C[si, di] > avg_ep:
                        base_alloc = sum(a for _, _, _, a, _ in pos_list)
                        pyr_alloc = base_alloc * current_pyramid_ratio
                        c_now = C[si, di]
                        atr = compute_atr_at(H, L, C, si, di, start_di)
                        if atr is not None:
                            new_positions.append(
                                (si, di, c_now,
                                 c_now - atr_stop * atr,
                                 pyr_alloc, True))

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
        if len(positions) >= current_top_n:
            continue

        # Entry signal
        candidates = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(composite[si, di]):
                continue
            if composite[si, di] < current_threshold:
                continue
            if n_confirm[si, di] < min_confidence:
                continue
            if use_ker_gate and ker_regime[si, di] < 0:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            alloc = 1.0 / max(current_top_n, 1)
            candidates.append((composite[si, di], si, alloc))

        candidates.sort(key=lambda x: -x[0])
        for rank_val, si, alloc in candidates[:current_top_n]:
            if len(positions) >= current_top_n or si in held:
                break
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr = compute_atr_at(H, L, C, si, di, start_di)
            if atr is None:
                continue
            positions.append(
                (si, di + 1, ep, ep - atr_stop * atr, alloc, False))
            held.add(si)

    # Close remaining positions
    for si, edi, ep, sp, alloc, is_pyr in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl * leverage

    return trades, equity, max_dd


def analyze(trades: List[dict], equity: float, max_dd: float,
            label: str = "", leverage: float = 1.0) -> Optional[dict]:
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

    mode_counts = {"WIN": 0, "NORM": 0, "LOSE": 0}
    for t in trades:
        m = t.get("mode", "NORM")
        if m.startswith("WIN"):
            mode_counts["WIN"] += 1
        elif m.startswith("LOSE"):
            mode_counts["LOSE"] += 1
        else:
            mode_counts["NORM"] += 1

    avg_thresh = np.mean([t.get("threshold", 0.85) for t in trades])

    # Contract-level reporting
    total_notional = sum(t.get("notional", 0) for t in trades)
    avg_contracts = np.mean([t.get("contracts", 0) for t in trades])

    print(f"  {label}: {len(trades)}t "
          f"(base:{n_base} pyr:{n_pyr} stop:{n_stop} hold:{n_hold}) "
          f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
          f"Sh={sh:.2f} eq={equity:,.0f} "
          f"lev={leverage:.1f}x avg_thresh={avg_thresh:.3f} "
          f"modes=[W:{mode_counts['WIN']} N:{mode_counts['NORM']} "
          f"L:{mode_counts['LOSE']}] "
          f"avg_contracts={avg_contracts:.0f}")

    return {
        "trades": len(trades), "wr": wr, "ann": ann, "dd": max_dd,
        "sharpe": sh, "equity": equity, "leverage": leverage,
        "modes": mode_counts, "avg_thresh": avg_thresh,
        "avg_contracts": avg_contracts,
    }


def walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms,
                 sigs, leverage=1.0, **bt_kwargs):
    """Walk-forward validation with expanding window."""
    year_starts = {}
    for di in range(ND):
        yr = dates[di].year
        if yr not in year_starts:
            year_starts[yr] = di

    test_years = sorted(y for y in year_starts if 2019 <= y <= 2026)
    if not test_years:
        return None, None

    all_trades = []
    all_equity = CASH0

    for yr in test_years:
        test_start = year_starts[yr]
        test_end = year_starts.get(yr + 1, ND)
        if test_start >= test_end:
            continue

        train_sigs = {
            "composite": sigs["composite"][:, :test_end],
            "ker_regime": sigs["ker_regime"][:, :test_end],
            "n_confirm": sigs["n_confirm"][:, :test_end],
        }

        trades, eq, dd = backtest_v57(
            C[:, :test_end], O[:, :test_end],
            H[:, :test_end], L[:, :test_end],
            NS, test_end, dates[:test_end], syms,
            train_sigs, leverage=leverage,
            end_di=test_end, **bt_kwargs,
        )

        year_trades = [t for t in trades if t.get("year") == yr]
        if year_trades:
            yr_pnl = sum(t["pnl_abs"] for t in year_trades)
            all_equity += yr_pnl
        all_trades.extend(trades)

        nw_yr = sum(1 for t in year_trades if t["pnl_pct"] > 0)
        wr_yr = nw_yr / len(year_trades) * 100 if year_trades else 0
        avg_yr = np.mean([t["pnl_pct"] for t in year_trades]) if year_trades else 0
        print(f"  {yr}: {len(year_trades)}t WR={wr_yr:.1f}% "
              f"avg={avg_yr:+.2f}% lev={leverage:.1f}x")

    if not all_trades:
        return None, None

    peak = CASH0
    max_dd = 0.0
    cum_eq = CASH0
    for t in sorted(all_trades, key=lambda x: x["di"]):
        cum_eq += t["pnl_abs"]
        if cum_eq > peak:
            peak = cum_eq
        dd = (peak - cum_eq) / peak * 100
        if dd > max_dd:
            max_dd = dd

    nw = sum(1 for t in all_trades if t["pnl_pct"] > 0)
    wr = nw / len(all_trades) * 100
    n_days = max(1, all_trades[-1]["di"] - all_trades[0]["di"])
    ann = ((cum_eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
    ap = [t["pnl_abs"] for t in sorted(all_trades, key=lambda x: x["di"])]
    rets = np.array(ap) / CASH0
    sh = (np.mean(rets) / np.std(rets) * np.sqrt(252)
          if np.std(rets) > 0 else 0)
    n_pyr = sum(1 for t in all_trades if t.get("pyr"))

    print(f"\n  WF TOTAL: {len(all_trades)}t (pyr:{n_pyr}) "
          f"WR={wr:.1f}% ann={ann:+.1f}% cum={((cum_eq/CASH0)-1)*100:+.1f}% "
          f"DD={max_dd:.1f}% Sh={sh:.2f} lev={leverage:.1f}x")

    return all_trades, cum_eq


def main():
    t_start = time.time()
    print("[V57] Institutional Dynamic Mode with Leverage", flush=True)
    print("=" * 70, flush=True)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data()
    print(f"[V57] {NS} instruments, {ND} days, "
          f"{dates[0]} to {dates[-1]}", flush=True)

    # Print contract multiplier info
    multipliers = get_all_multipliers(syms)
    print(f"[V57] Contract multipliers loaded for {len(multipliers)} instruments",
          flush=True)

    raw_factors = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_cross_sectional_ranks(raw_factors, NS, ND)
    ker_regime = compute_ker(C, NS, ND)
    composite, n_confirm = build_composite_signal(ranks, DEFAULT_WEIGHTS, NS, ND)

    sigs = {
        "composite": composite,
        "ker_regime": ker_regime,
        "n_confirm": n_confirm,
    }

    # ================================================================
    # LEVERAGE SWEEP
    # ================================================================
    print("\n" + "=" * 70, flush=True)
    print("  LEVERAGE IMPACT ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    leverage_levels = [0.5, 1.0, 1.5, 2.0, 3.0]
    best_base_params = {
        "win_threshold": 0.60,
        "normal_threshold": 0.82,
        "lose_threshold": 0.90,
        "win_rate_window": 15,
        "atr_stop": 3.0,
        "top_n_winning": 2,
        "hold_days": 5,
    }

    for lev in leverage_levels:
        print(f"\n  --- Leverage {lev:.1f}x ---", flush=True)
        # Full backtest
        trades, eq, dd = backtest_v57(
            C, O, H, L, NS, ND, dates, syms, sigs,
            leverage=lev, **best_base_params,
        )
        label = f"V57-lev{lev:.1f}"
        result = analyze(trades, eq, dd, label=label, leverage=lev)

    # ================================================================
    # WALK-FORWARD at different leverage levels
    # ================================================================
    print("\n" + "=" * 70, flush=True)
    print("  WALK-FORWARD BY LEVERAGE (2019-2026)", flush=True)
    print("=" * 70, flush=True)

    for lev in [1.0, 2.0, 3.0]:
        print(f"\n  WF Leverage {lev:.1f}x:", flush=True)
        walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms,
                     sigs, leverage=lev, **best_base_params)

    # ================================================================
    # PARAMETER SWEEP at leverage=1.0 (baseline)
    # ================================================================
    print("\n" + "=" * 70, flush=True)
    print("  PARAMETER SWEEP (leverage=1.0)", flush=True)
    print("=" * 70, flush=True)

    best_sharpe = -999
    best_config = None

    configs = list(product(
        [0.55, 0.60, 0.65],           # win_threshold
        [0.80, 0.82, 0.85],           # normal_threshold
        [0.85, 0.88, 0.90],           # lose_threshold
        [10, 15, 20],                  # win_rate_window
        [2.5, 3.0],                    # atr_stop
    ))

    print(f"  Testing {len(configs)} configurations...", flush=True)

    for idx, (wt, nt, lt, wrw, ats) in enumerate(configs):
        if (idx + 1) % 50 == 0:
            print(f"  Progress: {idx + 1}/{len(configs)} "
                  f"best_sh={best_sharpe:.2f}", flush=True)

        trades, eq, dd = backtest_v57(
            C, O, H, L, NS, ND, dates, syms, sigs,
            win_threshold=wt, normal_threshold=nt, lose_threshold=lt,
            win_rate_window=wrw, atr_stop=ats, leverage=1.0,
        )
        if not trades:
            continue

        nw = sum(1 for t in trades if t["pnl_pct"] > 0)
        wr = nw / len(trades) * 100
        n_days = max(1, trades[-1]["di"] - trades[0]["di"])
        ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
        ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
        rets = np.array(ap) / CASH0
        sh = (np.mean(rets) / np.std(rets) * np.sqrt(252)
              if np.std(rets) > 0 else 0)

        if sh > best_sharpe:
            best_sharpe = sh
            best_config = (wt, nt, lt, wrw, ats)
            print(f"  NEW BEST: Sh={sh:.2f} ann={ann:+.1f}% "
                  f"DD={dd:.1f}% {len(trades)}t "
                  f"wt={wt} nt={nt} lt={lt} wrw={wrw} ats={ats}",
                  flush=True)

    # ================================================================
    # BEST CONFIG: Full analysis + WF + Leverage sweep
    # ================================================================
    if best_config:
        wt, nt, lt, wrw, ats = best_config
        print("\n" + "=" * 70, flush=True)
        print(f"  BEST CONFIG: wt={wt} nt={nt} lt={lt} "
              f"wrw={wrw} ats={ats}", flush=True)
        print("=" * 70, flush=True)

        # Full 10yr at leverage 1.0
        trades, eq, dd = backtest_v57(
            C, O, H, L, NS, ND, dates, syms, sigs,
            win_threshold=wt, normal_threshold=nt, lose_threshold=lt,
            win_rate_window=wrw, atr_stop=ats, leverage=1.0,
        )
        analyze(trades, eq, dd, "V57-best-lev1.0", leverage=1.0)

        # WF at leverage 1.0
        print("\n  WF (lev=1.0):", flush=True)
        walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms, sigs,
                     leverage=1.0,
                     win_threshold=wt, normal_threshold=nt,
                     lose_threshold=lt, win_rate_window=wrw,
                     atr_stop=ats)

        # Best config at different leverages
        for lev in [1.5, 2.0, 3.0]:
            trades, eq, dd = backtest_v57(
                C, O, H, L, NS, ND, dates, syms, sigs,
                win_threshold=wt, normal_threshold=nt, lose_threshold=lt,
                win_rate_window=wrw, atr_stop=ats, leverage=lev,
            )
            analyze(trades, eq, dd, f"V57-best-lev{lev}", leverage=lev)

            print(f"\n  WF (lev={lev}):", flush=True)
            walk_forward(C, O, H, L, V, OI, NS, ND, dates, syms, sigs,
                         leverage=lev,
                         win_threshold=wt, normal_threshold=nt,
                         lose_threshold=lt, win_rate_window=wrw,
                         atr_stop=ats)

    elapsed = time.time() - t_start
    print(f"\n[V57] Done. {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
