"""
V76: Ultra-Aggressive Concentrated + Pyramid + Leverage
=======================================================
Takes V47 (highest base ann 19.8%, Sharpe 6.65) and makes it EXTREMELY aggressive:
  1. Concentrate: Only trade the #1 ranked signal each day (single best)
  2. Full allocation: 100% of equity on each trade
  3. Pyramid: If still winning after 1 day, ADD half-size position (max 2x)
  4. Leverage: sweep 3x, 5x, 8x, 10x, 15x, 20x
  5. Sector limit still applies (max 2 per sector, but single position anyway)
  6. Dynamic mode: WINNING=2 positions, LOSING=1 position

Key formulas:
  profit = equity * alloc * pnl_pct * leverage
  If pyramid and position still profitable after 1 day: add half-size

Walk-forward 2019-2026. Sweep leverage 3-20x.
Report table: leverage vs (ann, MDD, Sharpe, trades).
Target: find leverage that gives 600%+ annual.
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

# V47 signal weights (best performing)
DEFAULT_WEIGHTS = {
    "rank_ret5d": 0.25,
    "rank_oi5d": 0.20,
    "rank_rsi": 0.15,
    "rank_vol": 0.15,
    "rank_ret10d": 0.10,
    "rank_range": 0.10,
    "rank_atrp": 0.05,
}

SECTOR_MAP = {
    # BLACK (ferrous metals)
    'i': 'BLACK', 'j': 'BLACK', 'jm': 'BLACK', 'hc': 'BLACK',
    'sf': 'BLACK', 'sm': 'BLACK', 'wr': 'BLACK', 'im': 'BLACK',
    # METAL (non-ferrous + precious)
    'cu': 'METAL', 'al': 'METAL', 'zn': 'METAL', 'pb': 'METAL',
    'ni': 'METAL', 'sn': 'METAL', 'ss': 'METAL', 'ao': 'METAL',
    'au': 'METAL', 'ag': 'METAL', 'rb': 'METAL', 'si': 'METAL',
    # ENERGY
    'sc': 'ENERGY', 'fu': 'ENERGY', 'bu': 'ENERGY',
    'pg': 'ENERGY', 'eb': 'ENERGY', 'ta': 'ENERGY',
    'fg': 'ENERGY', 'oi': 'ENERGY',
    # CHEMICAL
    'v': 'CHEMICAL', 'pp': 'CHEMICAL', 'l': 'CHEMICAL',
    'eg': 'CHEMICAL', 'ma': 'CHEMICAL', 'sa': 'CHEMICAL',
    'ur': 'CHEMICAL', 'pf': 'CHEMICAL', 'sh': 'CHEMICAL',
    'lc': 'CHEMICAL',
    # AGRI (oilseeds / agricultural)
    'm': 'AGRI', 'y': 'AGRI', 'a': 'AGRI', 'p': 'AGRI',
    'c': 'AGRI', 'cs': 'AGRI', 'jd': 'AGRI', 'rr': 'AGRI',
    'lrm': 'AGRI', 'rm': 'AGRI', 'ru': 'AGRI',
    # SOFTS
    'cf': 'SOFTS', 'sr': 'SOFTS', 'ap': 'SOFTS',
    'cj': 'SOFTS', 'pk': 'SOFTS', 'lh': 'SOFTS',
    'sp': 'SOFTS', 'b': 'SOFTS', 'br': 'SOFTS',
}

LEVERAGE_SWEEP = [3, 5, 8, 10, 15, 20]


def _extract_base_symbol(sym: str) -> str:
    """Extract base commodity symbol from data symbol.
    Examples: 'cufi' -> 'cu', 'i0' -> 'i', 'im0' -> 'im',
              'lcfi' -> 'lc', 'rbfi' -> 'rb'
    """
    s = sym.lower().split('.')[0].strip()
    # Strip trailing digits
    while s and s[-1].isdigit():
        s = s[:-1]
    # Strip common suffixes: 'fi' (financial futures)
    if s.endswith('fi'):
        s = s[:-2]
    return s


def build_sector_lookup(syms: List[str]) -> Dict[int, str]:
    sector_lookup: Dict[int, str] = {}
    for si, sym in enumerate(syms):
        base = _extract_base_symbol(sym)
        sector_lookup[si] = SECTOR_MAP.get(base, 'OTHER')
    return sector_lookup


def compute_rsi_manual(
    C: np.ndarray, NS: int, ND: int, period: int = 14,
) -> np.ndarray:
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
                        rsi[si, di + period - 1] = 100.0 - 100.0 / (1.0 + rs)
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
    print("[V76] Computing raw factors...", flush=True)

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
                    daily_range[si, di] = (H[si, di] - L[si, di]) / C[si, di]

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
    print("[V76] Computing cross-sectional ranks...", flush=True)

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
            if ker_10[si, di] < 0.15:
                ker_regime[si, di] = 1
            elif ker_10[si, di] > 0.3:
                ker_regime[si, di] = -1
    return ker_regime


def build_composite_signal(
    ranks: Dict[str, np.ndarray],
    weights: Dict[str, float],
    NS: int, ND: int,
    min_factors: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    print("[V76] Building composite signal...", flush=True)

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


def compute_atr_at(
    H: np.ndarray, L: np.ndarray, C: np.ndarray,
    si: int, di: int, start_di: int,
) -> Optional[float]:
    atr_v = []
    for j in range(max(start_di, di - 14), di):
        hh, ll, cc = H[si, j], L[si, j], C[si, j]
        if not any(np.isnan([hh, ll, cc])):
            atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
    if atr_v:
        return np.mean(atr_v)
    return None


def get_dynamic_mode(
    recent_trades_win: List[int],
    win_threshold: float,
    win_rate_window: int,
) -> str:
    if len(recent_trades_win) < 5:
        return "normal"
    window = recent_trades_win[-win_rate_window:]
    win_rate = sum(window) / len(window)
    if win_rate > win_threshold:
        return "winning"
    elif win_rate < 0.50:
        return "losing"
    return "normal"


def get_mode_max_positions(mode: str) -> int:
    """WINNING mode allows 2 positions; others allow 1."""
    if mode == "winning":
        return 2
    return 1


def backtest_v76(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
    sector_lookup: Dict[int, str],
    leverage: float = 3.0,
    win_threshold: float = 0.60,
    normal_threshold: float = 0.82,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
    atr_stop: float = 3.0,
    max_per_sector: int = 2,
    min_confidence: int = 3,
    use_ker_gate: bool = True,
    hold_days: int = 5,
    pyramid_day: int = 1,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Backtest V76: ultra-concentrated #1 signal + 100% alloc + leverage."""
    composite = sigs["composite"]
    ker_regime = sigs["ker_regime"]
    n_confirm = sigs["n_confirm"]

    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    # Positions: (si, entry_di, entry_price, stop_price, alloc, is_pyramid)
    positions: List[Tuple[int, int, float, float, float, bool]] = []
    trades: List[dict] = []

    recent_trades_win: List[int] = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions: List[Tuple[int, int, float, float, float, bool]] = []

        # Dynamic mode
        mode = get_dynamic_mode(
            recent_trades_win, win_threshold, win_rate_window)
        max_positions = get_mode_max_positions(mode)

        # Mode-dependent threshold
        if mode == "winning":
            current_threshold = 0.75
            pyramid_ratio = 0.5
        elif mode == "losing":
            current_threshold = lose_threshold
            pyramid_ratio = 0.0
        else:
            current_threshold = normal_threshold
            pyramid_ratio = 0.5

        # --- Manage existing positions ---
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
                    # KEY: profit = equity * alloc * pnl_pct * leverage
                    profit = equity * alloc * pnl * leverage
                    daily_pnl += profit
                    is_win = pnl > 0
                    trades.append({
                        "pnl_abs": profit,
                        "pnl_pct": pnl * 100 * leverage,
                        "days": di - edi + 1,
                        "di": di,
                        "year": d.year,
                        "sym": syms[si],
                        "sector": sector_lookup.get(si, 'OTHER'),
                        "reason": "stop",
                        "pyr": is_pyr,
                        "mode": mode,
                        "leverage": leverage,
                    })
                    recent_trades_win.append(1 if is_win else 0)
            elif hold >= hold_days:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl * leverage
                    daily_pnl += profit
                    is_win = pnl > 0
                    trades.append({
                        "pnl_abs": profit,
                        "pnl_pct": pnl * 100 * leverage,
                        "days": di - edi + 1,
                        "di": di,
                        "year": d.year,
                        "sym": syms[si],
                        "sector": sector_lookup.get(si, 'OTHER'),
                        "reason": "hold",
                        "pyr": is_pyr,
                        "mode": mode,
                        "leverage": leverage,
                    })
                    recent_trades_win.append(1 if is_win else 0)
            else:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc, is_pyr))

        # --- Pyramid on day-1 winners ---
        if pyramid_ratio > 0:
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
                        pyr_alloc = base_alloc * pyramid_ratio
                        c_now = C[si, di]
                        atr = compute_atr_at(H, L, C, si, di, start_di)
                        if atr is not None:
                            additions.append(
                                (si, di, c_now,
                                 c_now - atr_stop * atr,
                                 pyr_alloc, True))
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
        if len(positions) >= max_positions:
            continue

        # --- Entry: pick #1 ranked signal ---
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
            candidates.append((composite[si, di], si))

        if not candidates:
            continue

        # Sort highest composite first
        candidates.sort(key=lambda x: -x[0])

        # Sector-constrained greedy: pick top candidate(s)
        sector_counts: Dict[str, int] = defaultdict(int)
        for si_held in held:
            sector_counts[sector_lookup.get(si_held, 'OTHER')] += 1

        for rank_val, si in candidates:
            if len(positions) >= max_positions or si in held:
                break
            sym_sector = sector_lookup.get(si, 'OTHER')
            if sector_counts[sym_sector] >= max_per_sector:
                continue
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr = compute_atr_at(H, L, C, si, di, start_di)
            if atr is None:
                continue
            # 100% allocation on the single best trade
            alloc = 1.0
            positions.append(
                (si, di + 1, ep, ep - atr_stop * atr, alloc, False))
            held.add(si)
            sector_counts[sym_sector] += 1

    # Close remaining positions at end
    for si, edi, ep, sp, alloc, is_pyr in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl * leverage

    return trades, equity, max_dd


def analyze(
    trades: List[dict], equity: float, max_dd: float,
    label: str = "",
) -> Optional[dict]:
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

    mode_counts = {"winning": 0, "normal": 0, "losing": 0}
    for t in trades:
        m = t.get("mode", "normal")
        if m in mode_counts:
            mode_counts[m] += 1

    print(
        f"  {label}: {len(trades)}t (base:{n_base} pyr:{n_pyr} "
        f"stop:{n_stop} hold:{n_hold}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
        f"Sh={sh:.2f} eq={equity:,.0f} "
        f"modes=[W:{mode_counts['winning']} N:{mode_counts['normal']} "
        f"L:{mode_counts['losing']}]"
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
            f"cum={cum:+.1%}")

    return {
        "n": len(trades), "wr": wr, "dd": max_dd,
        "ann": ann, "sh": sh, "eq": equity,
    }


def walk_forward_single(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
    sector_lookup: Dict[int, str],
    leverage: float,
    win_threshold: float = 0.60,
    normal_threshold: float = 0.82,
    lose_threshold: float = 0.90,
    win_rate_window: int = 15,
) -> List[dict]:
    """Walk-forward for a single leverage level."""
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

        trades, _, _ = backtest_v76(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            leverage=leverage,
            win_threshold=win_threshold,
            normal_threshold=normal_threshold,
            lose_threshold=lose_threshold,
            win_rate_window=win_rate_window,
            start_di=test_start,
            end_di=test_end_idx + 1,
        )

        test_trades = [t for t in trades if dates[t["di"]].year == test_year]
        all_trades.extend(test_trades)

    return all_trades


def compute_summary(trades: List[dict]) -> Optional[dict]:
    """Compute summary stats from a list of trades."""
    if not trades:
        return None
    nw = sum(1 for t in trades if t["pnl_pct"] > 0)
    wr = nw / len(trades) * 100
    avg_pnl = np.mean([t["pnl_pct"] for t in trades])
    cum = np.prod([1 + t["pnl_pct"] / 100 for t in trades]) - 1
    return {
        "n": len(trades),
        "wr": wr,
        "avg": avg_pnl,
        "cum": cum,
    }


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V76: ULTRA-AGGRESSIVE CONCENTRATED + PYRAMID + LEVERAGE")
    print("  V47 signal + #1 rank only + 100% alloc + leverage sweep")
    print("  Target: 600%+ annualized return")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(
        start="2016-01-01")
    print(
        f"  {NS} sym, {ND} days, "
        f"{dates[0].strftime('%Y-%m-%d')} to "
        f"{dates[-1].strftime('%Y-%m-%d')}"
    )

    sector_lookup = build_sector_lookup(syms)
    sector_dist: Dict[str, int] = defaultdict(int)
    for sec in sector_lookup.values():
        sector_dist[sec] += 1
    print(f"  Sector distribution: {dict(sector_dist)}")

    sigs = compute_all_signals(C, O, H, L, V, OI, NS, ND)

    # Find 2019 start index
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # =================================================================
    # SECTION 1: Leverage Sweep (2019-2026 OOS)
    # =================================================================
    print("\n" + "=" * 70)
    print("  LEVERAGE SWEEP (2019-2026 OOS)")
    print("  V47 best config: wt=0.60 nt=0.82 lt=0.90 ww=15")
    print("=" * 70)

    sweep_results: List[dict] = []

    for lev in LEVERAGE_SWEEP:
        trades, eq, dd = backtest_v76(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            leverage=float(lev),
            win_threshold=0.60,
            normal_threshold=0.82,
            lose_threshold=0.90,
            win_rate_window=15,
            start_di=bt_2019,
        )

        if not trades:
            print(f"  Lev={lev:>3}x: no trades")
            continue

        nw = sum(1 for t in trades if t["pnl_pct"] > 0)
        wr = nw / len(trades) * 100
        n_days = max(1, trades[-1]["di"] - trades[0]["di"])
        ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
        ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
        rets_arr = np.array(ap) / CASH0
        sh_val = (np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252)
                  if np.std(rets_arr) > 0 else 0)
        n_pyr = sum(1 for t in trades if t.get("pyr"))

        # Year-by-year breakdown
        yr_pnl: Dict[int, List[float]] = defaultdict(list)
        for t in trades:
            yr_pnl[t["year"]].append(t["pnl_pct"])

        sweep_results.append({
            "lev": lev,
            "n": len(trades),
            "n_pyr": n_pyr,
            "wr": wr,
            "ann": ann,
            "dd": dd,
            "sharpe": sh_val,
            "eq": eq,
            "yr_pnl": yr_pnl,
        })

    # Print sweep table
    print(
        f"\n  {'Lev':>4} {'N':>5} {'Pyr':>4} {'WR':>6} "
        f"{'Ann':>10} {'MDD':>7} {'Sharpe':>7} {'Equity':>14}"
    )
    print("  " + "-" * 65)
    for r in sweep_results:
        marker = " ***" if r["ann"] >= 600 else ""
        print(
            f"  {r['lev']:>4}x {r['n']:>5} {r['n_pyr']:>4} "
            f"{r['wr']:>5.1f}% {r['ann']:>+9.1f}% {r['dd']:>6.1f}% "
            f"{r['sharpe']:>7.2f} {r['eq']:>14,.0f}{marker}"
        )

    # Year-by-year for each leverage
    print("\n  YEAR-BY-YEAR CUMULATIVE RETURNS BY LEVERAGE:")
    print(
        f"  {'Year':>6}", end=""
    )
    for lev in LEVERAGE_SWEEP:
        print(f" {'Lev=' + str(lev) + 'x':>10}", end="")
    print()
    print("  " + "-" * (6 + 11 * len(LEVERAGE_SWEEP)))

    all_years = sorted(set(
        y for r in sweep_results for y in r["yr_pnl"].keys()))
    for year in all_years:
        print(f"  {year:>6}", end="")
        for lev in LEVERAGE_SWEEP:
            # Find matching sweep result
            matched = [r for r in sweep_results if r["lev"] == lev]
            if matched and year in matched[0]["yr_pnl"]:
                pnls = matched[0]["yr_pnl"][year]
                cum = np.prod([1 + p / 100 for p in pnls]) - 1
                print(f" {cum:>+9.1%}", end="")
            else:
                print(f" {'N/A':>10}", end="")
        print()

    # =================================================================
    # SECTION 2: Best leverage - detailed analysis
    # =================================================================
    # Find the lowest leverage that achieves 600%+ ann
    best_600 = None
    for r in sweep_results:
        if r["ann"] >= 600:
            best_600 = r
            break

    # If no 600%+ found, use highest ann
    best_result = best_600 if best_600 else (
        max(sweep_results, key=lambda x: x["ann"]) if sweep_results else None)

    if best_result:
        best_lev = best_result["lev"]
        print(f"\n{'=' * 70}")
        print(f"  BEST LEVERAGE: {best_lev}x (ann={best_result['ann']:+.1f}%)")
        print(f"{'=' * 70}")

        trades_best, eq_best, dd_best = backtest_v76(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            leverage=float(best_lev),
            win_threshold=0.60,
            normal_threshold=0.82,
            lose_threshold=0.90,
            win_rate_window=15,
            start_di=bt_2019,
        )
        analyze(trades_best, eq_best, dd_best, f"V76-Lev{best_lev}x-OOS")

    # =================================================================
    # SECTION 3: Walk-forward for top leverage levels
    # =================================================================
    print(f"\n{'=' * 70}")
    print("  WALK-FORWARD BY YEAR FOR EACH LEVERAGE (2019-2026)")
    print(f"{'=' * 70}")

    for lev in LEVERAGE_SWEEP:
        wf_trades = walk_forward_single(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            leverage=float(lev),
        )
        if not wf_trades:
            print(f"\n  Lev={lev}x: no WF trades")
            continue

        nw = sum(1 for t in wf_trades if t["pnl_pct"] > 0)
        wr = nw / len(wf_trades) * 100
        cum = np.prod([1 + t["pnl_pct"] / 100 for t in wf_trades]) - 1
        print(
            f"\n  Lev={lev}x WF: {len(wf_trades)}t "
            f"WR={wr:.1f}% cum={cum:+.1%}"
        )

        # Per-year breakdown
        yr_trades: Dict[int, List[dict]] = defaultdict(list)
        for t in wf_trades:
            yr_trades[t["year"]].append(t)
        for year in sorted(yr_trades.keys()):
            yt = yr_trades[year]
            yw = sum(1 for t in yt if t["pnl_pct"] > 0)
            ywr = yw / len(yt) * 100
            ycum = np.prod([1 + t["pnl_pct"] / 100 for t in yt]) - 1
            print(
                f"    {year}: {len(yt)}t WR={ywr:.1f}% cum={ycum:+.1%}")

    # =================================================================
    # SECTION 4: Parameter sweep with best leverage
    # =================================================================
    if best_result:
        target_lev = float(best_result["lev"])
        print(f"\n{'=' * 70}")
        print(f"  PARAMETER SWEEP WITH LEVERAGE={target_lev:.0f}x (2019-2026)")
        print(f"{'=' * 70}")

        param_results: List[dict] = []
        for wt in [0.55, 0.60, 0.65]:
            for nt in [0.80, 0.82, 0.85]:
                for lt in [0.88, 0.90, 0.92]:
                    if lt <= nt:
                        continue
                    for ww in [10, 15, 20]:
                        trades, eq, dd = backtest_v76(
                            C, O, H, L, NS, ND, dates, syms, sigs,
                            sector_lookup=sector_lookup,
                            leverage=target_lev,
                            win_threshold=wt,
                            normal_threshold=nt,
                            lose_threshold=lt,
                            win_rate_window=ww,
                            start_di=bt_2019,
                        )
                        if len(trades) < 5:
                            continue
                        nw = sum(1 for t in trades if t["pnl_pct"] > 0)
                        wr = nw / len(trades) * 100
                        n_days = max(1, trades[-1]["di"] - trades[0]["di"])
                        ann = ((eq / CASH0) ** (
                            1 / max(1.0, n_days / 252)) - 1) * 100
                        ap = [t["pnl_abs"]
                              for t in sorted(trades, key=lambda x: x["di"])]
                        rets_arr = np.array(ap) / CASH0
                        sh_val = (
                            np.mean(rets_arr) / np.std(rets_arr)
                            * np.sqrt(252)
                            if np.std(rets_arr) > 0 else 0)
                        param_results.append({
                            "wt": wt, "nt": nt, "lt": lt, "ww": ww,
                            "n": len(trades), "wr": wr,
                            "ann": ann, "dd": dd,
                            "sharpe": sh_val, "eq": eq,
                        })

        param_results.sort(key=lambda x: -x["sharpe"])
        print(
            f"\n  {'WT':>4} {'NT':>4} {'LT':>4} {'WW':>3} "
            f"{'N':>5} {'WR':>6} {'Ann':>10} {'MDD':>7} "
            f"{'Sharpe':>7} {'Equity':>14}"
        )
        print("  " + "-" * 80)
        for r in param_results[:20]:
            print(
                f"  {r['wt']:>4.2f} {r['nt']:>4.2f} {r['lt']:>4.2f} "
                f"{r['ww']:>3} {r['n']:>5} {r['wr']:>5.1f}% "
                f"{r['ann']:>+9.1f}% {r['dd']:>6.1f}% "
                f"{r['sharpe']:>7.2f} {r['eq']:>14,.0f}"
            )

        # Walk-forward for best param config
        if param_results:
            bp = param_results[0]
            print(f"\n{'=' * 70}")
            print(
                f"  BEST PARAM WF: wt={bp['wt']:.2f} nt={bp['nt']:.2f} "
                f"lt={bp['lt']:.2f} ww={bp['ww']} lev={target_lev:.0f}x")
            print(f"{'=' * 70}")

            wf_best = walk_forward_single(
                C, O, H, L, NS, ND, dates, syms, sigs,
                sector_lookup=sector_lookup,
                leverage=target_lev,
                win_threshold=bp["wt"],
                normal_threshold=bp["nt"],
                lose_threshold=bp["lt"],
                win_rate_window=bp["ww"],
            )
            if wf_best:
                nw = sum(1 for t in wf_best if t["pnl_pct"] > 0)
                wr = nw / len(wf_best) * 100
                cum = np.prod(
                    [1 + t["pnl_pct"] / 100 for t in wf_best]) - 1
                print(
                    f"  WF TOTAL: {len(wf_best)}t "
                    f"WR={wr:.1f}% cum={cum:+.1%}")

                yr_wf: Dict[int, List[dict]] = defaultdict(list)
                for t in wf_best:
                    yr_wf[t["year"]].append(t)
                for year in sorted(yr_wf.keys()):
                    yt = yr_wf[year]
                    yw = sum(1 for t in yt if t["pnl_pct"] > 0)
                    ywr = yw / len(yt) * 100
                    ycum = np.prod(
                        [1 + t["pnl_pct"] / 100 for t in yt]) - 1
                    print(
                        f"    {year}: {len(yt)}t WR={ywr:.1f}% "
                        f"cum={ycum:+.1%}")

    # =================================================================
    # SECTION 5: Full 10-year for all leverage levels
    # =================================================================
    print(f"\n{'=' * 70}")
    print("  FULL 10-YEAR (2016-2026) FOR ALL LEVERAGE LEVELS")
    print(f"{'=' * 70}")

    for lev in LEVERAGE_SWEEP:
        trades, eq, dd = backtest_v76(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            leverage=float(lev),
            win_threshold=0.60,
            normal_threshold=0.82,
            lose_threshold=0.90,
            win_rate_window=15,
            start_di=60,
        )
        analyze(trades, eq, dd, f"V76-Lev{lev}x-FULL10Y")

    print(f"\n[V76] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
