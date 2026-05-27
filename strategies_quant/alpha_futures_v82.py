"""
V82: Top-N Concentrated Daily Compound
=======================================
Take the top-N ranked signals every qualifying day with equal capital allocation.
Daily rebalancing: exit all positions each day, re-select top-N.

Key differences from V61:
  - HOLD = 1 day (daily rebalance, not 5-day)
  - Fixed top-N selection (not dynamic mode switching)
  - No pyramid / no ATR stop-loss / no KER gate
  - Simpler = more robust for daily compounding

Signal: same multi-TF composite from V61 (st_weight * ST + (1-st_weight) * MT)
Entry at open[di+1], exit at close[di+1] (1-day hold)
Quality filter: composite rank > min_rank
Sector limit: max_per_sector per sector

Parameter sweep:
  - top_n: 2, 3, 4, 5
  - min_rank: 0.65, 0.70, 0.75, 0.80
  - max_per_sector: 2, 3
  - st_weight: 0.55, 0.65, 0.75

Walk-forward 2019-2026, full 10-year for top configs.
Signal at close[di], enter at open[di+1], exit at close[di+1]. No look-ahead.
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

# Short-term (5d) factor weights
ST_WEIGHTS = {
    "rank_ret5d": 0.25,
    "rank_oi5d": 0.20,
    "rank_rsi": 0.15,
    "rank_vol5d": 0.15,
    "rank_ret10d": 0.10,
    "rank_range5d": 0.10,
    "rank_atrp5d": 0.05,
}

# Medium-term (20d) factor weights
MT_WEIGHTS = {
    "rank_ret20d": 0.25,
    "rank_oi20d": 0.20,
    "rank_rsi": 0.15,
    "rank_vol20d": 0.15,
    "rank_ret10d": 0.10,
    "rank_range20d": 0.10,
    "rank_atrp20d": 0.05,
}

# Sector definitions
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


def _extract_base_symbol(sym: str) -> str:
    s = sym.lower().split('.')[0].strip()
    while s and s[-1].isdigit():
        s = s[:-1]
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
    print("[V82] Computing raw factors (5d + 20d)...", flush=True)

    # --- Short-term (5d) factors ---
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (not np.isnan(C[si, di])
                    and not np.isnan(C[si, di - 5])
                    and C[si, di - 5] > 0):
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1.0

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

    range_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            rng_vals = []
            for j in range(di - 5, di):
                if (not np.isnan(H[si, j]) and not np.isnan(L[si, j])
                        and not np.isnan(C[si, j]) and C[si, j] > 0
                        and H[si, j] > L[si, j]):
                    rng_vals.append((H[si, j] - L[si, j]) / C[si, j])
            if len(rng_vals) >= 3:
                range_5d[si, di] = np.mean(rng_vals)

    atrp_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(6, ND):
            atr_vals = []
            for j in range(di - 5, di):
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
                atrp_5d[si, di] = np.mean(atr_vals) / C[si, di]

    # --- Shared factors ---
    ret_10d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            if (not np.isnan(C[si, di])
                    and not np.isnan(C[si, di - 10])
                    and C[si, di - 10] > 0):
                ret_10d[si, di] = C[si, di] / C[si, di - 10] - 1.0

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

    # --- Medium-term (20d) factors ---
    ret_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if (not np.isnan(C[si, di])
                    and not np.isnan(C[si, di - 20])
                    and C[si, di - 20] > 0):
                ret_20d[si, di] = C[si, di] / C[si, di - 20] - 1.0

    oi_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if (not np.isnan(OI[si, di])
                    and not np.isnan(OI[si, di - 20])
                    and OI[si, di - 20] > 0):
                oi_20d[si, di] = OI[si, di] / OI[si, di - 20] - 1.0

    vol_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            vals = V[si, di - 20:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 10:
                vol_20d[si, di] = np.mean(valid)

    range_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            rng_vals = []
            for j in range(di - 20, di):
                if (not np.isnan(H[si, j]) and not np.isnan(L[si, j])
                        and not np.isnan(C[si, j]) and C[si, j] > 0
                        and H[si, j] > L[si, j]):
                    rng_vals.append((H[si, j] - L[si, j]) / C[si, j])
            if len(rng_vals) >= 10:
                range_20d[si, di] = np.mean(rng_vals)

    atrp_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            atr_vals = []
            for j in range(di - 20, di):
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
                atrp_20d[si, di] = np.mean(atr_vals) / C[si, di]

    print(f"  Raw factors done: {time.time() - t0:.1f}s", flush=True)
    return {
        "ret_5d": ret_5d,
        "ret_10d": ret_10d,
        "ret_20d": ret_20d,
        "oi_5d": oi_5d,
        "oi_20d": oi_20d,
        "vol_5d": vol_5d,
        "vol_20d": vol_20d,
        "range_5d": range_5d,
        "range_20d": range_20d,
        "rsi14": rsi14,
        "atrp_5d": atrp_5d,
        "atrp_20d": atrp_20d,
    }


def compute_cross_sectional_ranks(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int, min_count: int = 10,
) -> Dict[str, np.ndarray]:
    t0 = time.time()
    print("[V82] Computing cross-sectional ranks...", flush=True)

    factors_to_rank = {
        "rank_ret5d": raw_factors["ret_5d"],
        "rank_ret10d": raw_factors["ret_10d"],
        "rank_oi5d": raw_factors["oi_5d"],
        "rank_vol5d": raw_factors["vol_5d"],
        "rank_range5d": raw_factors["range_5d"],
        "rank_rsi": raw_factors["rsi14"],
        "rank_atrp5d": raw_factors["atrp_5d"],
        "rank_ret20d": raw_factors["ret_20d"],
        "rank_oi20d": raw_factors["oi_20d"],
        "rank_vol20d": raw_factors["vol_20d"],
        "rank_range20d": raw_factors["range_20d"],
        "rank_atrp20d": raw_factors["atrp_20d"],
    }

    INVERT_FACTORS = {
        "rank_ret5d", "rank_ret10d", "rank_oi5d", "rank_rsi",
        "rank_ret20d", "rank_oi20d",
    }

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


def build_multi_tf_composite(
    ranks: Dict[str, np.ndarray],
    st_weights: Dict[str, float],
    mt_weights: Dict[str, float],
    st_weight: float,
    NS: int, ND: int,
    min_factors: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t0 = time.time()
    print(f"[V82] Building multi-TF composite (st_w={st_weight:.2f})...",
          flush=True)

    mt_weight = 1.0 - st_weight

    combined = np.full((NS, ND), np.nan)
    st_comp = np.full((NS, ND), np.nan)
    mt_comp = np.full((NS, ND), np.nan)
    n_confirm_st = np.zeros((NS, ND), dtype=int)
    n_confirm_mt = np.zeros((NS, ND), dtype=int)

    st_names = list(st_weights.keys())
    st_wvals = np.array([st_weights[k] for k in st_names])
    mt_names = list(mt_weights.keys())
    mt_wvals = np.array([mt_weights[k] for k in mt_names])

    for di in range(ND):
        for si in range(NS):
            # Short-term composite
            st_vals = []
            st_wsum = 0.0
            st_confirm = 0
            for idx, name in enumerate(st_names):
                rank_val = ranks[name][si, di]
                if np.isnan(rank_val):
                    continue
                st_vals.append(rank_val * st_wvals[idx])
                st_wsum += st_wvals[idx]
                if rank_val > 0.5:
                    st_confirm += 1

            if st_wsum > 0 and st_confirm >= min_factors:
                st_comp[si, di] = sum(st_vals) / st_wsum
                n_confirm_st[si, di] = st_confirm

            # Medium-term composite
            mt_vals = []
            mt_wsum = 0.0
            mt_confirm = 0
            for idx, name in enumerate(mt_names):
                rank_val = ranks[name][si, di]
                if np.isnan(rank_val):
                    continue
                mt_vals.append(rank_val * mt_wvals[idx])
                mt_wsum += mt_wvals[idx]
                if rank_val > 0.5:
                    mt_confirm += 1

            if mt_wsum > 0 and mt_confirm >= min_factors:
                mt_comp[si, di] = sum(mt_vals) / mt_wsum
                n_confirm_mt[si, di] = mt_confirm

            # Combined: only when both timeframes available
            if not np.isnan(st_comp[si, di]) and not np.isnan(mt_comp[si, di]):
                combined[si, di] = (st_weight * st_comp[si, di]
                                    + mt_weight * mt_comp[si, di])

    print(f"  Multi-TF composite done: {time.time() - t0:.1f}s", flush=True)
    return combined, st_comp, mt_comp, n_confirm_st, n_confirm_mt


def compute_all_signals(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
    st_weight: float = 0.65,
    st_weights: Optional[Dict[str, float]] = None,
    mt_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    if st_weights is None:
        st_weights = ST_WEIGHTS
    if mt_weights is None:
        mt_weights = MT_WEIGHTS

    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_cross_sectional_ranks(raw, NS, ND)
    combined, st_comp, mt_comp, ncf_st, ncf_mt = build_multi_tf_composite(
        ranks, st_weights, mt_weights, st_weight, NS, ND)

    return {
        "composite": combined,
        "st_comp": st_comp,
        "mt_comp": mt_comp,
        "n_confirm_st": ncf_st,
        "n_confirm_mt": ncf_mt,
        "ranks": ranks,
    }


def backtest_v82(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
    sector_lookup: Dict[int, str],
    top_n: int = 3,
    min_rank: float = 0.75,
    max_per_sector: int = 2,
    st_weight: float = 0.65,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Backtest V82: daily rebalanced top-N concentrated portfolio.

    Each day:
      1. Rank all instruments by composite score
      2. Select top-N (subject to min_rank quality filter + sector limit)
      3. Allocate 1/top_n equity to each
      4. Enter at open[di+1], exit at close[di+1] (1-day hold)
    """
    composite = sigs["composite"]

    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    trades: List[dict] = []

    for di in range(max(start_di, 1), end_di - 1):
        d = dates[di]
        daily_pnl = 0.0

        # --- Select top-N candidates for today ---
        # Signal computed at close[di], enter at open[di+1]
        candidates = []
        for si in range(NS):
            rank_val = composite[si, di]
            if np.isnan(rank_val):
                continue
            if rank_val < min_rank:
                continue
            # Need next day open for entry price
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            if O[si, di + 1] <= 0:
                continue
            candidates.append((rank_val, si))

        # Sort descending by composite rank (best first)
        candidates.sort(key=lambda x: -x[0])

        # Sector-constrained greedy selection
        sector_counts: Dict[str, int] = defaultdict(int)
        selected: List[Tuple[float, int]] = []

        for rank_val, si in candidates:
            if len(selected) >= top_n:
                break
            sym_sector = sector_lookup.get(si, 'OTHER')
            if sector_counts[sym_sector] >= max_per_sector:
                continue
            selected.append((rank_val, si))
            sector_counts[sym_sector] += 1

        if not selected:
            continue

        # --- Execute trades: equal-weight allocation ---
        alloc_per_pos = 1.0 / top_n

        for rank_val, si in selected:
            entry_price = O[si, di + 1]  # enter at next day open
            # Exit at close[di+1]
            exit_price = C[si, di + 1]
            if np.isnan(exit_price) or exit_price <= 0:
                continue

            pnl_pct = (exit_price - entry_price) / entry_price - COMM
            profit = equity * alloc_per_pos * pnl_pct
            daily_pnl += profit

            is_win = pnl_pct > 0
            trades.append({
                "pnl_abs": profit,
                "pnl_pct": pnl_pct * 100,
                "days": 1,
                "di": di + 1,
                "year": dates[di + 1].year,
                "sym": syms[si],
                "sector": sector_lookup.get(si, 'OTHER'),
                "reason": "daily",
                "rank": rank_val,
                "entry": entry_price,
                "exit": exit_price,
            })

        equity += daily_pnl
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        if equity <= 0:
            break

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

    avg_rank = np.mean([t["rank"] for t in trades])

    sector_counts: Dict[str, int] = defaultdict(int)
    for t in trades:
        sector_counts[t.get("sector", "OTHER")] += 1
    sector_str = " ".join(
        f"{k}:{v}" for k, v in sorted(sector_counts.items()))

    print(
        f"  {label}: {len(trades)}t "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
        f"Sh={sh:.2f} eq={equity:,.0f} avg_rank={avg_rank:.3f}"
    )
    print(f"    sectors: {sector_str}")

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


def walk_forward(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
    sector_lookup: Dict[int, str],
    top_n: int = 3,
    min_rank: float = 0.75,
    max_per_sector: int = 2,
    st_weight: float = 0.65,
) -> List[dict]:
    """Walk-forward validation: year-by-year out-of-sample."""
    print(f"\n{'=' * 70}")
    print(
        f"  WALK-FORWARD V82 "
        f"(top_n={top_n} min_rank={min_rank:.2f} "
        f"mps={max_per_sector} st_w={st_weight:.2f})"
    )
    print(f"{'=' * 70}")

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

        trades, _, _ = backtest_v82(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            top_n=top_n,
            min_rank=min_rank,
            max_per_sector=max_per_sector,
            st_weight=st_weight,
            start_di=test_start,
            end_di=test_end_idx + 1,
        )

        test_trades = [t for t in trades
                       if dates[t["di"]].year == test_year]
        all_trades.extend(test_trades)

        if test_trades:
            n = len(test_trades)
            nw = sum(1 for t in test_trades if t["pnl_pct"] > 0)
            wr_val = nw / n * 100
            avg = np.mean([t["pnl_pct"] for t in test_trades])
            yr_sectors: Dict[str, int] = defaultdict(int)
            for t in test_trades:
                yr_sectors[t.get("sector", "OTHER")] += 1
            sec_str = " ".join(
                f"{k}:{v}" for k, v in sorted(yr_sectors.items()))
            print(
                f"  {test_year}: {n}t WR={wr_val:.1f}% avg={avg:+.2f}% "
                f"sectors=[{sec_str}]",
                flush=True,
            )
        else:
            print(f"  {test_year}: no trades", flush=True)

    if all_trades:
        nw = sum(1 for t in all_trades if t["pnl_pct"] > 0)
        wr_val = nw / len(all_trades) * 100
        avg = np.mean([t["pnl_pct"] for t in all_trades])
        cum = np.prod(
            [1 + t["pnl_pct"] / 100 for t in all_trades]) - 1
        agg_sectors: Dict[str, int] = defaultdict(int)
        for t in all_trades:
            agg_sectors[t.get("sector", "OTHER")] += 1
        sec_str = " ".join(
            f"{k}:{v}" for k, v in sorted(agg_sectors.items()))
        print(
            f"\n  WF TOTAL: {len(all_trades)}t "
            f"WR={wr_val:.1f}% avg={avg:+.2f}% cum={cum:+.1%}"
        )
        print(f"  WF SECTORS: {sec_str}")
        return all_trades
    return []


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V82: TOP-N CONCENTRATED DAILY COMPOUND")
    print("  Daily rebalanced: select top-N, 1-day hold, equal weight")
    print("  Multi-TF composite (same as V61)")
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

    # Find 2019 start index for OOS testing
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # === 1. Pre-compute signals for each st_weight ===
    signal_cache: Dict[float, Dict] = {}
    for st_w in [0.55, 0.65, 0.75]:
        print(f"\n--- Computing signals for st_weight={st_w:.2f} ---")
        signal_cache[st_w] = compute_all_signals(
            C, O, H, L, V, OI, NS, ND, st_weight=st_w)

    # === 2. Parameter sweep ===
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("=" * 70)

    results: List[dict] = []

    for st_w in [0.55, 0.65, 0.75]:
        sigs = signal_cache[st_w]
        for tn in [2, 3, 4, 5]:
            for mr in [0.65, 0.70, 0.75, 0.80]:
                for mps in [2, 3]:
                    trades, eq, dd = backtest_v82(
                        C, O, H, L, NS, ND, dates, syms, sigs,
                        sector_lookup=sector_lookup,
                        top_n=tn,
                        min_rank=mr,
                        max_per_sector=mps,
                        st_weight=st_w,
                        start_di=bt_2019,
                    )

                    if len(trades) < 10:
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
                        np.mean(rets_arr)
                        / np.std(rets_arr) * np.sqrt(252)
                        if np.std(rets_arr) > 0 else 0)

                    # Days traded (unique di values)
                    unique_days = len(set(t["di"] for t in trades))
                    avg_daily_trades = len(trades) / max(unique_days, 1)

                    results.append({
                        "st_w": st_w, "top_n": tn,
                        "min_rank": mr, "mps": mps,
                        "n": len(trades), "wr": wr,
                        "ann": ann, "dd": dd,
                        "sharpe": sh_val, "eq": eq,
                        "avg_daily": avg_daily_trades,
                        "unique_days": unique_days,
                    })

    # Sort by Sharpe first, then by ann
    results.sort(key=lambda x: (-x["sharpe"], -x["ann"]))
    print(f"\n  Evaluated {len(results)} configs with 10+ trades")
    print(
        f"\n{'STw':>4} {'TN':>3} {'MR':>4} {'MPS':>3} "
        f"{'N':>6} {'WR':>5} {'Ann':>8} {'DD':>6} "
        f"{'Sh':>6} {'AvgD':>5} {'Days':>5}"
    )
    print("-" * 70)
    for r in results[:40]:
        print(
            f"{r['st_w']:>4.2f} {r['top_n']:>3} {r['min_rank']:>4.2f} "
            f"{r['mps']:>3} "
            f"{r['n']:>6} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
            f"{r['dd']:>6.1f} {r['sharpe']:>6.2f} "
            f"{r['avg_daily']:>5.1f} {r['unique_days']:>5}"
        )

    # === 3. Top configs: full 10-year backtest ===
    print("\n" + "=" * 70)
    print("  TOP CONFIGS -- FULL 10-YEAR (2016-2026)")
    print("=" * 70)

    seen = set()
    unique_top = []
    for r in results:
        key = (r["st_w"], r["top_n"], r["min_rank"], r["mps"])
        if key not in seen:
            seen.add(key)
            unique_top.append(r)
        if len(unique_top) >= 5:
            break

    for r in unique_top:
        sigs = signal_cache[r["st_w"]]
        trades, eq, dd = backtest_v82(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            top_n=r["top_n"],
            min_rank=r["min_rank"],
            max_per_sector=r["mps"],
            st_weight=r["st_w"],
            start_di=60,
        )
        label = (
            f"st_w={r['st_w']:.2f} top_n={r['top_n']} "
            f"min_rank={r['min_rank']:.2f} mps={r['mps']}"
        )
        print(f"\n  FULL {label}")
        analyze(trades, eq, dd, label)

    # === 4. Walk-forward for best config ===
    if results:
        best = results[0]
        print("\n" + "=" * 70)
        print(
            f"  BEST WF: st_w={best['st_w']:.2f} "
            f"top_n={best['top_n']} min_rank={best['min_rank']:.2f} "
            f"mps={best['mps']}"
        )
        print("=" * 70)
        walk_forward(
            C, O, H, L, NS, ND, dates, syms,
            signal_cache[best["st_w"]],
            sector_lookup=sector_lookup,
            top_n=best["top_n"],
            min_rank=best["min_rank"],
            max_per_sector=best["mps"],
            st_weight=best["st_w"],
        )

        # === 5. Walk-forward for top-3 configs ===
        print("\n" + "=" * 70)
        print("  TOP-3 CONFIGS WALK-FORWARD COMPARISON")
        print("=" * 70)

        for rank, r in enumerate(unique_top[:3]):
            print(f"\n  --- Config #{rank + 1} ---")
            wf_trades = walk_forward(
                C, O, H, L, NS, ND, dates, syms,
                signal_cache[r["st_w"]],
                sector_lookup=sector_lookup,
                top_n=r["top_n"],
                min_rank=r["min_rank"],
                max_per_sector=r["mps"],
                st_weight=r["st_w"],
            )
            if wf_trades:
                nw = sum(1 for t in wf_trades if t["pnl_pct"] > 0)
                wr_val = nw / len(wf_trades) * 100
                cum = np.prod(
                    [1 + t["pnl_pct"] / 100 for t in wf_trades]) - 1
                print(
                    f"  WF Summary #{rank + 1}: "
                    f"{len(wf_trades)}t WR={wr_val:.1f}% cum={cum:+.1%}"
                )

    # === 6. Comparison with different top_n at best st_weight ===
    if results:
        best_stw = results[0]["st_w"]
        best_mr = results[0]["min_rank"]
        best_mps = results[0]["mps"]
        sigs = signal_cache[best_stw]

        print("\n" + "=" * 70)
        print(
            f"  TOP_N COMPARISON (st_w={best_stw:.2f} "
            f"min_rank={best_mr:.2f} mps={best_mps})"
        )
        print("  How does N=2,3,4,5 compare?")
        print("=" * 70)

        for tn in [2, 3, 4, 5]:
            trades, eq, dd = backtest_v82(
                C, O, H, L, NS, ND, dates, syms, sigs,
                sector_lookup=sector_lookup,
                top_n=tn,
                min_rank=best_mr,
                max_per_sector=best_mps,
                st_weight=best_stw,
                start_di=bt_2019,
            )
            label = f"N={tn}"
            print(f"\n  {label}:")
            analyze(trades, eq, dd, label)

    print(f"\n[V82] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
