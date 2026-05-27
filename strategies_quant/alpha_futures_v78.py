"""
V78: Daily-Rebalanced Mean Reversion + Aggressive Leverage
==========================================================
High-frequency MR strategy that trades EVERY qualifying day.

Key differences from V61 (conservative, 78 trades over 7 years):
  1. No mode switching -- trade every day with a qualifying signal
  2. Lower composite threshold: 0.70 (more signals)
  3. Take ALL instruments above threshold (not top N)
  4. Equal-weight across qualifying instruments
  5. Position size: 100% equity / N positions * leverage
  6. Sector limit: max 2 per sector
  7. Leverage sweep: 3x, 5x, 8x, 10x, 15x, 20x, 25x, 30x

7 factors (same as V47/V61 short-term):
  ret5d(0.25), oi5d(0.20), rsi(0.15), vol(0.15), ret10d(0.10), range(0.10), atrp(0.05)

Walk-forward 2019-2026. Leverage sweep table.
Target: 600%+ annualized.

Signal at close[di], enter at open[di+1], exit at close[di+1] (1-day hold).
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

# Factor weights (same as V47/V61 short-term)
FACTOR_WEIGHTS = {
    "rank_ret5d": 0.25,
    "rank_oi5d": 0.20,
    "rank_rsi": 0.15,
    "rank_vol5d": 0.15,
    "rank_ret10d": 0.10,
    "rank_range5d": 0.10,
    "rank_atrp5d": 0.05,
}

# Sector definitions
SECTOR_MAP = {
    'i': 'BLACK', 'j': 'BLACK', 'jm': 'BLACK', 'hc': 'BLACK',
    'sf': 'BLACK', 'sm': 'BLACK', 'wr': 'BLACK', 'im': 'BLACK',
    'cu': 'METAL', 'al': 'METAL', 'zn': 'METAL', 'pb': 'METAL',
    'ni': 'METAL', 'sn': 'METAL', 'ss': 'METAL', 'ao': 'METAL',
    'au': 'METAL', 'ag': 'METAL', 'rb': 'METAL', 'si': 'METAL',
    'sc': 'ENERGY', 'fu': 'ENERGY', 'bu': 'ENERGY',
    'pg': 'ENERGY', 'eb': 'ENERGY', 'ta': 'ENERGY',
    'fg': 'ENERGY', 'oi': 'ENERGY',
    'v': 'CHEMICAL', 'pp': 'CHEMICAL', 'l': 'CHEMICAL',
    'eg': 'CHEMICAL', 'ma': 'CHEMICAL', 'sa': 'CHEMICAL',
    'ur': 'CHEMICAL', 'pf': 'CHEMICAL', 'sh': 'CHEMICAL',
    'lc': 'CHEMICAL',
    'm': 'AGRI', 'y': 'AGRI', 'a': 'AGRI', 'p': 'AGRI',
    'c': 'AGRI', 'cs': 'AGRI', 'jd': 'AGRI', 'rr': 'AGRI',
    'lrm': 'AGRI', 'rm': 'AGRI', 'ru': 'AGRI',
    'cf': 'SOFTS', 'sr': 'SOFTS', 'ap': 'SOFTS',
    'cj': 'SOFTS', 'pk': 'SOFTS', 'lh': 'SOFTS',
    'sp': 'SOFTS', 'b': 'SOFTS', 'br': 'SOFTS',
}

LEVERAGE_LEVELS = [3, 5, 8, 10, 15, 20, 25, 30]


def _extract_base_symbol(sym: str) -> str:
    """Extract base commodity symbol from data symbol."""
    s = sym.lower().split('.')[0].strip()
    while s and s[-1].isdigit():
        s = s[:-1]
    if s.endswith('fi'):
        s = s[:-2]
    return s


def build_sector_lookup(syms: List[str]) -> Dict[int, str]:
    """Build a symbol-index to sector mapping."""
    lookup: Dict[int, str] = {}
    for si, sym in enumerate(syms):
        base = _extract_base_symbol(sym)
        lookup[si] = SECTOR_MAP.get(base, 'OTHER')
    return lookup


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
    """Compute raw factor values (short-term 5d only)."""
    t0 = time.time()
    print("[V78] Computing raw factors...", flush=True)

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
                        else cc)
                    atr_vals.append(
                        max(hh - ll, abs(hh - prev_c), abs(ll - prev_c)))
            if atr_vals and not np.isnan(C[si, di]) and C[si, di] > 0:
                atrp_5d[si, di] = np.mean(atr_vals) / C[si, di]

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

    print(f"  Raw factors done: {time.time() - t0:.1f}s", flush=True)
    return {
        "ret_5d": ret_5d,
        "ret_10d": ret_10d,
        "oi_5d": oi_5d,
        "vol_5d": vol_5d,
        "range_5d": range_5d,
        "rsi14": rsi14,
        "atrp_5d": atrp_5d,
    }


def compute_cross_sectional_ranks(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int, min_count: int = 10,
) -> Dict[str, np.ndarray]:
    """Rank all factors cross-sectionally. Inverted for mean-reversion."""
    t0 = time.time()
    print("[V78] Computing cross-sectional ranks...", flush=True)

    factors_to_rank = {
        "rank_ret5d": raw_factors["ret_5d"],
        "rank_ret10d": raw_factors["ret_10d"],
        "rank_oi5d": raw_factors["oi_5d"],
        "rank_vol5d": raw_factors["vol_5d"],
        "rank_range5d": raw_factors["range_5d"],
        "rank_rsi": raw_factors["rsi14"],
        "rank_atrp5d": raw_factors["atrp_5d"],
    }

    INVERT_FACTORS = {
        "rank_ret5d", "rank_ret10d", "rank_oi5d", "rank_rsi",
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
                .values)
            if name in INVERT_FACTORS:
                ranked = 1.0 - ranked
            rank_arr[:, di] = ranked
        ranks[name] = rank_arr

    print(f"  CS ranks done: {time.time() - t0:.1f}s", flush=True)
    return ranks


def compute_composite(
    ranks: Dict[str, np.ndarray],
    weights: Dict[str, float],
    NS: int, ND: int,
    min_factors: int = 4,
) -> np.ndarray:
    """Compute weighted composite signal from ranked factors."""
    t0 = time.time()
    print("[V78] Computing composite...", flush=True)

    composite = np.full((NS, ND), np.nan)
    w_names = list(weights.keys())
    w_vals = np.array([weights[k] for k in w_names])

    for di in range(ND):
        for si in range(NS):
            vals = []
            wsum = 0.0
            for idx, name in enumerate(w_names):
                rank_val = ranks[name][si, di]
                if np.isnan(rank_val):
                    continue
                vals.append(rank_val * w_vals[idx])
                wsum += w_vals[idx]
            if wsum > 0 and len(vals) >= min_factors:
                composite[si, di] = sum(vals) / wsum

    print(f"  Composite done: {time.time() - t0:.1f}s", flush=True)
    return composite


def backtest_daily(
    C: np.ndarray, O: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    composite: np.ndarray,
    sector_lookup: Dict[int, str],
    leverage: float = 1.0,
    threshold: float = 0.70,
    max_per_sector: int = 2,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Daily-rebalanced backtest: enter at open[di+1], exit at close[di+1].

    Each day:
      1. Score all instruments using composite at close[di]
      2. Filter: composite >= threshold
      3. Apply sector limit (max 2 per sector)
      4. Equal-weight: alloc = 1.0 / N * leverage per instrument
      5. Enter at open[di+1], exit at close[di+1]
      6. PnL = (close - open) / open - COMM  per instrument
    """
    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    trades: List[dict] = []

    for di in range(max(start_di, 1), end_di - 1):
        d_signal = dates[di]
        di_entry = di + 1
        d_entry = dates[di_entry]

        # Collect candidates: composite[si, di] >= threshold, valid open[di+1]
        candidates: List[Tuple[float, int]] = []
        for si in range(NS):
            if np.isnan(composite[si, di]):
                continue
            if composite[si, di] < threshold:
                continue
            if di_entry >= ND or np.isnan(O[si, di_entry]):
                continue
            if np.isnan(C[si, di_entry]):
                continue
            if O[si, di_entry] <= 0:
                continue
            candidates.append((composite[si, di], si))

        if not candidates:
            continue

        # Sort by composite descending
        candidates.sort(key=lambda x: -x[0])

        # Apply sector limit
        sector_counts: Dict[str, int] = defaultdict(int)
        selected: List[Tuple[float, int]] = []
        for score, si in candidates:
            sec = sector_lookup.get(si, 'OTHER')
            if sector_counts[sec] >= max_per_sector:
                continue
            selected.append((score, si))
            sector_counts[sec] += 1

        if not selected:
            continue

        n_pos = len(selected)
        alloc_per = (1.0 / n_pos) * leverage

        daily_pnl = 0.0
        for score, si in selected:
            entry_price = O[si, di_entry]
            exit_price = C[si, di_entry]
            if np.isnan(exit_price) or exit_price <= 0:
                continue
            ret = (exit_price - entry_price) / entry_price - COMM
            pnl_abs = equity * alloc_per * ret
            daily_pnl += pnl_abs
            trades.append({
                "pnl_abs": pnl_abs,
                "pnl_pct": ret * 100,
                "days": 1,
                "di": di_entry,
                "year": d_entry.year,
                "sym": syms[si],
                "sector": sector_lookup.get(si, 'OTHER'),
                "reason": "daily",
                "score": score,
                "leverage": leverage,
                "n_pos": n_pos,
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

    avg_npos = np.mean([t["n_pos"] for t in trades])

    sector_counts: Dict[str, int] = defaultdict(int)
    for t in trades:
        sector_counts[t.get("sector", "OTHER")] += 1
    sector_str = " ".join(
        f"{k}:{v}" for k, v in sorted(sector_counts.items()))

    print(
        f"  {label}: {len(trades)}t WR={wr:.1f}% "
        f"ann={ann:+.1f}% DD={max_dd:.1f}% "
        f"Sh={sh:.2f} eq={equity:,.0f} avgN={avg_npos:.1f}"
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
    C: np.ndarray, O: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    composite: np.ndarray,
    sector_lookup: Dict[int, str],
    leverage: float = 1.0,
    threshold: float = 0.70,
    max_per_sector: int = 2,
) -> List[dict]:
    """Walk-forward: year-by-year out-of-sample."""
    print(f"\n{'=' * 70}")
    print(
        f"  WALK-FORWARD V78 "
        f"(lev={leverage:.0f}x thr={threshold:.2f} mps={max_per_sector})"
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

        trades, _, _ = backtest_daily(
            C, O, NS, ND, dates, syms, composite,
            sector_lookup=sector_lookup,
            leverage=leverage,
            threshold=threshold,
            max_per_sector=max_per_sector,
            start_di=test_start,
            end_di=test_end_idx,
        )

        test_trades = [t for t in trades
                       if dates[t["di"]].year == test_year]
        all_trades.extend(test_trades)

        if test_trades:
            n = len(test_trades)
            nw = sum(1 for t in test_trades if t["pnl_pct"] > 0)
            wr_val = nw / n * 100
            avg = np.mean([t["pnl_pct"] for t in test_trades])
            cum = np.prod(
                [1 + t["pnl_pct"] / 100 for t in test_trades]) - 1
            avg_npos = np.mean([t["n_pos"] for t in test_trades])
            yr_sectors: Dict[str, int] = defaultdict(int)
            for t in test_trades:
                yr_sectors[t.get("sector", "OTHER")] += 1
            sec_str = " ".join(
                f"{k}:{v}" for k, v in sorted(yr_sectors.items()))
            print(
                f"  {test_year}: {n}t WR={wr_val:.1f}% "
                f"avg={avg:+.2f}% cum={cum:+.1%} "
                f"avgN={avg_npos:.1f} [{sec_str}]",
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
        avg_npos = np.mean([t["n_pos"] for t in all_trades])
        agg_sectors: Dict[str, int] = defaultdict(int)
        for t in all_trades:
            agg_sectors[t.get("sector", "OTHER")] += 1
        sec_str = " ".join(
            f"{k}:{v}" for k, v in sorted(agg_sectors.items()))
        print(
            f"\n  WF TOTAL: {len(all_trades)}t "
            f"WR={wr_val:.1f}% avg={avg:+.2f}% cum={cum:+.1%} "
            f"avgN={avg_npos:.1f}"
        )
        print(f"  WF SECTORS: {sec_str}")
        return all_trades
    return []


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V78: DAILY-REBALANCED MR + AGGRESSIVE LEVERAGE SWEEP")
    print("  Every qualifying day, all instruments, sector-limited, leveraged")
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

    # Find 2019 start index
    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2019-01-01"):
            bt_2019 = i
            break

    # Compute signals once (no mode switching, no MT -- just ST composite)
    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_cross_sectional_ranks(raw, NS, ND)
    composite = compute_composite(ranks, FACTOR_WEIGHTS, NS, ND)

    # === 1. Leverage sweep (2019-2026) ===
    print("\n" + "=" * 70)
    print("  LEVERAGE SWEEP (2019-2026 OOS)")
    print("=" * 70)

    sweep_results: List[dict] = []
    for lev in LEVERAGE_LEVELS:
        trades, eq, dd = backtest_daily(
            C, O, NS, ND, dates, syms, composite,
            sector_lookup=sector_lookup,
            leverage=lev,
            threshold=0.70,
            max_per_sector=2,
            start_di=bt_2019,
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
        avg_npos = np.mean([t["n_pos"] for t in trades])

        sweep_results.append({
            "lev": lev, "n": len(trades), "wr": wr,
            "ann": ann, "dd": dd, "sharpe": sh, "eq": eq,
            "avg_npos": avg_npos,
        })

    # Print sweep table
    print(
        f"\n{'Lev':>4} {'N':>6} {'WR':>6} {'Ann':>10} "
        f"{'DD':>7} {'Sh':>7} {'Eq':>16} {'AvgN':>6}"
    )
    print("-" * 80)
    for r in sweep_results:
        print(
            f"{r['lev']:>4}x {r['n']:>6} {r['wr']:>5.1f}% "
            f"{r['ann']:>+9.1f}% {r['dd']:>6.1f}% "
            f"{r['sharpe']:>7.2f} {r['eq']:>16,.0f} "
            f"{r['avg_npos']:>6.1f}"
        )

    # === 2. Threshold sensitivity at best leverage ===
    if sweep_results:
        best_lev = max(sweep_results, key=lambda x: x["ann"])["lev"]
        print(
            f"\n{'=' * 70}")
        print(
            f"  THRESHOLD SENSITIVITY (lev={best_lev}x)")
        print(f"{'=' * 70}")

        for thr in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
            trades, eq, dd = backtest_daily(
                C, O, NS, ND, dates, syms, composite,
                sector_lookup=sector_lookup,
                leverage=best_lev,
                threshold=thr,
                max_per_sector=2,
                start_di=bt_2019,
            )
            if not trades:
                print(f"  thr={thr:.2f}: no trades")
                continue
            nw = sum(1 for t in trades if t["pnl_pct"] > 0)
            wr = nw / len(trades) * 100
            n_days = max(1, trades[-1]["di"] - trades[0]["di"])
            ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
            ap = [t["pnl_abs"]
                  for t in sorted(trades, key=lambda x: x["di"])]
            rets = np.array(ap) / CASH0
            sh = (np.mean(rets) / np.std(rets) * np.sqrt(252)
                  if np.std(rets) > 0 else 0)
            avg_npos = np.mean([t["n_pos"] for t in trades])
            print(
                f"  thr={thr:.2f}: {len(trades)}t WR={wr:.1f}% "
                f"ann={ann:+.1f}% DD={dd:.1f}% Sh={sh:.2f} "
                f"eq={eq:,.0f} avgN={avg_npos:.1f}"
            )

    # === 3. Walk-forward for top 3 leverage levels ===
    top_by_ann = sorted(sweep_results, key=lambda x: -x["ann"])[:3]
    for r in top_by_ann:
        walk_forward(
            C, O, NS, ND, dates, syms, composite,
            sector_lookup=sector_lookup,
            leverage=r["lev"],
            threshold=0.70,
            max_per_sector=2,
        )

    # === 4. Full detailed analysis for best config ===
    if top_by_ann:
        best = top_by_ann[0]
        print("\n" + "=" * 70)
        print(
            f"  BEST CONFIG FULL ANALYSIS "
            f"(lev={best['lev']}x thr=0.70 mps=2)")
        print(f"{'=' * 70}")

        trades, eq, dd = backtest_daily(
            C, O, NS, ND, dates, syms, composite,
            sector_lookup=sector_lookup,
            leverage=best["lev"],
            threshold=0.70,
            max_per_sector=2,
            start_di=bt_2019,
        )
        analyze(trades, eq, dd, f"V78-best-lev{best['lev']}x")

    print(f"\n[V78] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
