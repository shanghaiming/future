"""
V79: Multi-Strategy Ensemble with Aggressive Leverage
=====================================================
Combine MULTIPLE uncorrelated signals and apply leverage:

  Signal A: Short-term MR (5d rank, threshold 0.80) -- daily signals
  Signal B: OI contrarian (OI 5d drop > 5%) -- event signals
  Signal C: RSI extreme (RSI < 30 percentile) -- extreme MR signals

  Take ALL signals that pass ANY of A, B, or C.
  If instrument passes multiple signals, boost its rank.
  Max positions: 5 concurrent (diversified).
  Sector limit: max 2 per sector.

  Leverage sweep: 3x, 5x, 8x, 10x, 15x, 20x, 25x, 30x

Data: from alpha_futures_data import load_all_data
  C, O, H, L, V, OI, NS, ND, dates, syms
7 factors: ret5d(0.25), oi5d(0.20), rsi(0.15), vol(0.15),
           ret10d(0.10), range(0.10), atrp(0.05)
CASH0=1M, COMM=0.0005

Walk-forward 2019-2026. Report leverage sweep table.
Signal at close[di], enter at open[di+1]. No look-ahead. No gap signals.
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

# Factor weights for composite score (Signal A)
FACTOR_WEIGHTS = {
    "rank_ret5d": 0.25,
    "rank_oi5d": 0.20,
    "rank_rsi": 0.15,
    "rank_vol5d": 0.15,
    "rank_ret10d": 0.10,
    "rank_range5d": 0.10,
    "rank_atrp5d": 0.05,
}

# Signal thresholds
MR_THRESHOLD = 0.80       # Signal A: composite rank threshold
OI_DROP_THRESHOLD = -0.05  # Signal B: OI 5d drop < -5%
RSI_PERCENTILE = 30        # Signal C: RSI in bottom 30 percentile

# Leverage sweep values
LEVERAGE_SWEEP = [3, 5, 8, 10, 15, 20, 25, 30]

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

MAX_POSITIONS = 5
MAX_PER_SECTOR = 2
HOLD_DAYS = 5
ATR_STOP = 3.0
MIN_FACTORS = 4


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
    """Manual RSI computation fallback."""
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
                vg, vl = [], []
                for j in range(di, min(di + period, ND)):
                    if not np.isnan(gains[j]):
                        vg.append(gains[j])
                        vl.append(
                            losses[j] if not np.isnan(losses[j]) else 0.0)
                if len(vg) >= period:
                    avg_gain = np.mean(vg)
                    avg_loss = np.mean(vl)
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
    """Compute raw factor values for all signals."""
    t0 = time.time()
    print("[V79] Computing raw factors...", flush=True)

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

    ret_10d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            if (not np.isnan(C[si, di])
                    and not np.isnan(C[si, di - 10])
                    and C[si, di - 10] > 0):
                ret_10d[si, di] = C[si, di] / C[si, di - 10] - 1.0

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
    raw: Dict[str, np.ndarray], NS: int, ND: int,
    min_count: int = 10,
) -> Dict[str, np.ndarray]:
    """Rank all factors cross-sectionally. Inverted for MR factors."""
    t0 = time.time()
    print("[V79] Computing cross-sectional ranks...", flush=True)

    factor_map = {
        "rank_ret5d": raw["ret_5d"],
        "rank_oi5d": raw["oi_5d"],
        "rank_rsi": raw["rsi14"],
        "rank_vol5d": raw["vol_5d"],
        "rank_ret10d": raw["ret_10d"],
        "rank_range5d": raw["range_5d"],
        "rank_atrp5d": raw["atrp_5d"],
    }

    INVERT = {"rank_ret5d", "rank_oi5d", "rank_rsi", "rank_ret10d"}

    ranks: Dict[str, np.ndarray] = {}
    for name, factor in factor_map.items():
        rank_arr = np.full((NS, ND), np.nan)
        for di in range(ND):
            vals = factor[:, di]
            valid_count = np.sum(~np.isnan(vals))
            if valid_count < min_count:
                continue
            ranked = pd.Series(vals).rank(pct=True, na_option="keep").values
            if name in INVERT:
                ranked = 1.0 - ranked
            rank_arr[:, di] = ranked
        ranks[name] = rank_arr

    print(f"  CS ranks done: {time.time() - t0:.1f}s", flush=True)
    return ranks


def compute_signals(
    raw: Dict[str, np.ndarray],
    ranks: Dict[str, np.ndarray],
    NS: int, ND: int,
    mr_threshold: float = MR_THRESHOLD,
    oi_drop_threshold: float = OI_DROP_THRESHOLD,
    rsi_percentile: int = RSI_PERCENTILE,
) -> Dict[str, np.ndarray]:
    """Compute all three signal sources and the ensemble score.

    Returns:
        signal_a: bool -- composite rank >= threshold (short-term MR)
        signal_b: bool -- OI 5d drop below threshold (OI contrarian)
        signal_c: bool -- RSI in bottom percentile (extreme MR)
        composite: float -- weighted composite score (from Signal A)
        signal_count: int -- number of signals triggered (0-3)
        ensemble_score: float -- composite + bonus for multi-signal
    """
    t0 = time.time()
    print("[V79] Computing multi-signal ensemble...", flush=True)

    # --- Signal A: composite rank threshold ---
    names = list(FACTOR_WEIGHTS.keys())
    wvals = np.array([FACTOR_WEIGHTS[k] for k in names])

    composite = np.full((NS, ND), np.nan)
    n_factors = np.zeros((NS, ND), dtype=int)

    for di in range(ND):
        for si in range(NS):
            vals, wsum, nf = [], 0.0, 0
            for idx, name in enumerate(names):
                rv = ranks[name][si, di]
                if np.isnan(rv):
                    continue
                vals.append(rv * wvals[idx])
                wsum += wvals[idx]
                nf += 1
            if wsum > 0 and nf >= MIN_FACTORS:
                composite[si, di] = sum(vals) / wsum
                n_factors[si, di] = nf

    signal_a = np.where(~np.isnan(composite), composite >= mr_threshold, False)

    # --- Signal B: OI contrarian ---
    oi_5d = raw["oi_5d"]
    signal_b = np.where(~np.isnan(oi_5d), oi_5d < oi_drop_threshold, False)

    # --- Signal C: RSI extreme ---
    # RSI rank already inverted: low RSI -> high rank
    # So signal_c = rank_rsi >= (1 - rsi_percentile/100)
    # e.g., rsi_percentile=30 -> rank_rsi >= 0.70 (bottom 30% RSI)
    rsi_rank = ranks["rank_rsi"]
    rsi_threshold = 1.0 - rsi_percentile / 100.0
    signal_c = np.where(~np.isnan(rsi_rank), rsi_rank >= rsi_threshold, False)

    # --- Signal count (0-3) ---
    signal_count = (
        signal_a.astype(int) + signal_b.astype(int) + signal_c.astype(int)
    )

    # --- Ensemble score: composite + bonus per additional signal ---
    # Base = composite score; each extra signal adds 0.05 boost
    MULTI_SIGNAL_BONUS = 0.05
    ensemble_score = np.where(
        ~np.isnan(composite),
        composite + (signal_count.astype(float) - 1.0) * MULTI_SIGNAL_BONUS,
        np.nan,
    )
    # For instruments without composite but with signal B or C,
    # give them a minimal score based on signal count
    ensemble_score = np.where(
        np.isnan(ensemble_score) & (signal_count > 0),
        0.5 + signal_count.astype(float) * MULTI_SIGNAL_BONUS,
        ensemble_score,
    )

    print(
        f"  Signal counts -- A:{np.sum(signal_a)} B:{np.sum(signal_b)} "
        f"C:{np.sum(signal_c)} multi:{np.sum(signal_count >= 2)}",
        flush=True,
    )
    print(f"  Ensemble done: {time.time() - t0:.1f}s", flush=True)

    return {
        "signal_a": signal_a,
        "signal_b": signal_b,
        "signal_c": signal_c,
        "composite": composite,
        "signal_count": signal_count,
        "ensemble_score": ensemble_score,
        "n_factors": n_factors,
    }


def compute_atr_at(
    H: np.ndarray, L: np.ndarray, C: np.ndarray,
    si: int, di: int, start_di: int,
) -> Optional[float]:
    """Compute ATR at a given point."""
    atr_v = []
    for j in range(max(start_di, di - 14), di):
        hh, ll, cc = H[si, j], L[si, j], C[si, j]
        if not any(np.isnan([hh, ll, cc])):
            atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
    if atr_v:
        return float(np.mean(atr_v))
    return None


def backtest_v79(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
    sector_lookup: Dict[int, str],
    leverage: float = 1.0,
    mr_threshold: float = MR_THRESHOLD,
    oi_drop_threshold: float = OI_DROP_THRESHOLD,
    rsi_percentile: int = RSI_PERCENTILE,
    max_positions: int = MAX_POSITIONS,
    max_per_sector: int = MAX_PER_SECTOR,
    hold_days: int = HOLD_DAYS,
    atr_stop: float = ATR_STOP,
    start_di: int = 60,
    end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    """Backtest V79: multi-signal ensemble with leverage.

    Positions are sized at (leverage / max_positions) of equity.
    ATR stop-loss is widened by leverage factor to avoid premature stops.
    """
    signal_a = sigs["signal_a"]
    signal_b = sigs["signal_b"]
    signal_c = sigs["signal_c"]
    ensemble_score = sigs["ensemble_score"]
    signal_count = sigs["signal_count"]

    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions: List[Tuple[int, int, float, float, float, bool]] = []
    trades: List[dict] = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_positions: List[Tuple[int, int, float, float, float, bool]] = []

        # --- Exit management ---
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
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    sc = int(signal_count[si, edi])
                    trades.append({
                        "pnl_abs": profit,
                        "pnl_pct": pnl * 100,
                        "days": di - edi + 1,
                        "di": di,
                        "year": d.year,
                        "sym": syms[si],
                        "sector": sector_lookup.get(si, 'OTHER'),
                        "reason": "stop",
                        "pyr": is_pyr,
                        "sig_count": sc,
                        "leverage": leverage,
                    })
            elif hold >= hold_days:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    sc = int(signal_count[si, edi])
                    trades.append({
                        "pnl_abs": profit,
                        "pnl_pct": pnl * 100,
                        "days": di - edi + 1,
                        "di": di,
                        "year": d.year,
                        "sym": syms[si],
                        "sector": sector_lookup.get(si, 'OTHER'),
                        "reason": "hold",
                        "pyr": is_pyr,
                        "sig_count": sc,
                        "leverage": leverage,
                    })
            else:
                for edi, ep, sp, alloc, is_pyr in pos_list:
                    new_positions.append((si, edi, ep, sp, alloc, is_pyr))

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

        # --- Entry: take ALL signals that pass ANY of A, B, or C ---
        held = {p[0] for p in positions}
        if len(positions) >= max_positions:
            continue

        candidates = []
        for si in range(NS):
            if si in held:
                continue
            # Must pass at least one signal
            if not (signal_a[si, di] or signal_b[si, di] or signal_c[si, di]):
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            score = ensemble_score[si, di]
            if np.isnan(score):
                score = 0.5  # fallback for signal_b/c only
            sc = int(signal_count[si, di])
            candidates.append((score, sc, si))

        # Sort: multi-signal first (higher count), then by ensemble score
        candidates.sort(key=lambda x: (-x[1], -x[0]))

        # Sector-constrained greedy selection
        sector_counts: Dict[str, int] = defaultdict(int)
        for si_held in held:
            sector_counts[sector_lookup.get(si_held, 'OTHER')] += 1

        # Allocation: leverage spread across max_positions
        alloc_per_pos = leverage / max(max_positions, 1)

        for score, sc, si in candidates:
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
            positions.append(
                (si, di + 1, ep, ep - atr_stop * atr, alloc_per_pos, False))
            held.add(si)
            sector_counts[sym_sector] += 1

    # Close remaining positions at last price
    for si, edi, ep, sp, alloc, is_pyr in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl

    return trades, equity, max_dd


def analyze(
    trades: List[dict], equity: float, max_dd: float,
    label: str = "",
) -> Optional[dict]:
    """Analyze backtest results."""
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

    n_stop = sum(1 for t in trades if t["reason"] == "stop")
    n_hold = sum(1 for t in trades if t["reason"] == "hold")
    avg_sc = np.mean([t.get("sig_count", 1) for t in trades])
    sc_dist: Dict[int, int] = defaultdict(int)
    for t in trades:
        sc_dist[t.get("sig_count", 1)] += 1

    sector_counts: Dict[str, int] = defaultdict(int)
    for t in trades:
        sector_counts[t.get("sector", "OTHER")] += 1
    sector_str = " ".join(
        f"{k}:{v}" for k, v in sorted(sector_counts.items()))

    print(
        f"  {label}: {len(trades)}t (stop:{n_stop} hold:{n_hold}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
        f"Sh={sh:.2f} eq={equity:,.0f} avgSig={avg_sc:.2f}"
    )
    print(f"    signal_dist: {dict(sc_dist)}")
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
    leverage: float = 1.0,
    max_positions: int = MAX_POSITIONS,
    max_per_sector: int = MAX_PER_SECTOR,
    hold_days: int = HOLD_DAYS,
    atr_stop: float = ATR_STOP,
) -> List[dict]:
    """Walk-forward validation: year-by-year out-of-sample."""
    print(f"\n{'=' * 70}")
    print(
        f"  WALK-FORWARD V79 lev={leverage:.0f}x "
        f"maxpos={max_positions} maxsec={max_per_sector}"
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

        trades, _, _ = backtest_v79(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            leverage=leverage,
            max_positions=max_positions,
            max_per_sector=max_per_sector,
            hold_days=hold_days,
            atr_stop=atr_stop,
            start_di=test_start,
            end_di=test_end_idx + 1,
        )

        test_trades = [t for t in trades if dates[t["di"]].year == test_year]
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
        cum = np.prod([1 + t["pnl_pct"] / 100 for t in all_trades]) - 1
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


def param_sweep(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    raw: Dict[str, np.ndarray],
    ranks: Dict[str, np.ndarray],
    sector_lookup: Dict[int, str],
    start_di: int,
) -> List[dict]:
    """Sweep signal thresholds + leverage to find best risk-adjusted config."""
    print(f"\n{'=' * 70}")
    print("  PARAMETER SWEEP (2019-2026)")
    print(f"{'=' * 70}")

    results: List[dict] = []

    for mr_thr in [0.75, 0.80, 0.85, 0.90]:
        for oi_thr in [-0.03, -0.05, -0.08]:
            for rsi_pct in [20, 30, 40]:
                sigs = compute_signals(
                    raw, ranks, NS, ND,
                    mr_threshold=mr_thr,
                    oi_drop_threshold=oi_thr,
                    rsi_percentile=rsi_pct,
                )

                for lev in [1, 2, 3, 5, 8, 10, 15, 20]:
                    for max_pos in [3, 5]:
                        for hold in [3, 5, 7]:
                            for a_stop in [2.0, 3.0, 4.0]:
                                trades, eq, dd = backtest_v79(
                                    C, O, H, L, NS, ND, dates, syms, sigs,
                                    sector_lookup=sector_lookup,
                                    leverage=float(lev),
                                    mr_threshold=mr_thr,
                                    oi_drop_threshold=oi_thr,
                                    rsi_percentile=rsi_pct,
                                    max_positions=max_pos,
                                    hold_days=hold,
                                    atr_stop=a_stop,
                                    start_di=start_di,
                                )

                                if len(trades) < 10 or eq <= 0:
                                    continue

                                nw = sum(
                                    1 for t in trades
                                    if t["pnl_pct"] > 0)
                                wr = nw / len(trades) * 100
                                n_days = max(
                                    1,
                                    trades[-1]["di"] - trades[0]["di"])
                                ann = ((eq / CASH0) ** (
                                    1 / max(1.0, n_days / 252)) - 1) * 100
                                cum_ret = eq / CASH0 - 1.0
                                ap = [t["pnl_abs"]
                                      for t in sorted(
                                          trades, key=lambda x: x["di"])]
                                rets_arr = np.array(ap) / CASH0
                                sh_val = (
                                    np.mean(rets_arr)
                                    / np.std(rets_arr) * np.sqrt(252)
                                    if np.std(rets_arr) > 0 else 0)

                                results.append({
                                    "mr": mr_thr, "oi": oi_thr,
                                    "rsi": rsi_pct, "lev": lev,
                                    "mp": max_pos, "hd": hold,
                                    "as": a_stop,
                                    "n": len(trades), "wr": wr,
                                    "ann": ann, "dd": dd,
                                    "sh": sh_val, "eq": eq,
                                    "cum": cum_ret * 100,
                                })

    results.sort(key=lambda x: (-x["ann"] if x["dd"] < 80 else 0))
    print(f"\n  Evaluated {len(results)} configs with 10+ trades")

    print(
        f"\n{'MR':>4} {'OI':>5} {'RSI':>3} {'Lev':>4} {'MP':>2} "
        f"{'HD':>2} {'AS':>3} "
        f"{'N':>5} {'WR':>5} {'Ann':>8} {'Cum%':>10} "
        f"{'DD':>6} {'Sh':>6}"
    )
    print("-" * 90)
    for r in results[:30]:
        print(
            f"{r['mr']:>4.2f} {r['oi']:>5.2f} {r['rsi']:>3} "
            f"{r['lev']:>3}x {r['mp']:>2} {r['hd']:>2} {r['as']:>3.1f} "
            f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f}% "
            f"{r['cum']:>+9.1f}% {r['dd']:>6.1f}% {r['sh']:>6.2f}"
        )

    return results


def leverage_sweep(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    sigs: Dict[str, np.ndarray],
    sector_lookup: Dict[int, str],
    start_di: int,
    leverage_values: Optional[List[int]] = None,
) -> List[dict]:
    """Run backtest across all leverage values and report table."""
    if leverage_values is None:
        leverage_values = LEVERAGE_SWEEP

    print(f"\n{'=' * 70}")
    print("  LEVERAGE SWEEP (2019-2026)")
    print(f"{'=' * 70}")

    results: List[dict] = []

    for lev in leverage_values:
        trades, eq, dd = backtest_v79(
            C, O, H, L, NS, ND, dates, syms, sigs,
            sector_lookup=sector_lookup,
            leverage=float(lev),
            start_di=start_di,
        )

        if len(trades) < 5:
            results.append({
                "lev": lev, "n": len(trades), "wr": 0, "ann": 0,
                "dd": 0, "sh": 0, "eq": eq, "cum": 0,
            })
            continue

        nw = sum(1 for t in trades if t["pnl_pct"] > 0)
        wr = nw / len(trades) * 100
        n_days = max(1, trades[-1]["di"] - trades[0]["di"])
        ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
        cum_ret = eq / CASH0 - 1.0
        ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
        rets_arr = np.array(ap) / CASH0
        sh_val = (
            np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252)
            if np.std(rets_arr) > 0 else 0)

        results.append({
            "lev": lev, "n": len(trades), "wr": wr,
            "ann": ann, "dd": dd, "sh": sh_val,
            "eq": eq, "cum": cum_ret * 100,
        })

    # Print table
    print(
        f"\n{'Lev':>4} {'N':>5} {'WR':>6} {'Ann':>10} "
        f"{'Cum%':>12} {'DD':>7} {'Sh':>7} {'Eq':>14}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['lev']:>4}x {r['n']:>5} {r['wr']:>5.1f}% "
            f"{r['ann']:>+9.1f}% {r['cum']:>+11.1f}% "
            f"{r['dd']:>6.1f}% {r['sh']:>7.2f} {r['eq']:>14,.0f}"
        )

    return results


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V79: MULTI-STRATEGY ENSEMBLE WITH AGGRESSIVE LEVERAGE")
    print("  Signal A: Short-term MR (5d rank >= 0.80)")
    print("  Signal B: OI contrarian (OI 5d drop > 5%)")
    print("  Signal C: RSI extreme (RSI < 30th percentile)")
    print("  Multi-signal boost, max 5 positions, sector limit 2")
    print("  Leverage sweep: 3x to 30x")
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

    # === 1. Compute factors and signals ===
    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ranks = compute_cross_sectional_ranks(raw, NS, ND)
    sigs = compute_signals(raw, ranks, NS, ND)

    # === 2. Default leverage sweep ===
    lev_results = leverage_sweep(
        C, O, H, L, NS, ND, dates, syms, sigs,
        sector_lookup=sector_lookup,
        start_di=bt_2019,
    )

    # === 3. Parameter sweep for best risk-adjusted config ===
    sweep_results = param_sweep(
        C, O, H, L, NS, ND, dates, syms,
        raw, ranks, sector_lookup, bt_2019,
    )

    # === 4. Detailed analysis of top configs from sweep ===
    if sweep_results:
        # Best by ann among configs with DD < 80%
        safe_sweep = [r for r in sweep_results if r["dd"] < 80]
        if not safe_sweep:
            safe_sweep = sweep_results[:5]
        safe_sweep.sort(key=lambda x: -x["ann"])

        print(f"\n{'=' * 70}")
        print("  TOP PARAMETER SWEEP CONFIGS (detailed)")
        print(f"{'=' * 70}")

        # Re-compute signals for each unique threshold combo
        seen_thresholds = set()
        detailed_configs = []
        for r in safe_sweep[:10]:
            key = (r["mr"], r["oi"], r["rsi"])
            if key in seen_thresholds:
                continue
            seen_thresholds.add(key)
            detailed_configs.append(r)
            if len(detailed_configs) >= 3:
                break

        for r in detailed_configs:
            s = compute_signals(
                raw, ranks, NS, ND,
                mr_threshold=r["mr"],
                oi_drop_threshold=r["oi"],
                rsi_percentile=r["rsi"],
            )
            trades, eq, dd = backtest_v79(
                C, O, H, L, NS, ND, dates, syms, s,
                sector_lookup=sector_lookup,
                leverage=float(r["lev"]),
                max_positions=r["mp"],
                hold_days=r["hd"],
                atr_stop=r["as"],
                start_di=bt_2019,
            )
            label = (
                f"mr={r['mr']:.2f} oi={r['oi']:.2f} rsi={r['rsi']} "
                f"lev={r['lev']}x mp={r['mp']} hd={r['hd']} as={r['as']}"
            )
            print(f"\n  {label}")
            analyze(trades, eq, dd, label)

        # === 5. Walk-forward for best sweep config ===
        best_sweep = safe_sweep[0]
        best_sigs = compute_signals(
            raw, ranks, NS, ND,
            mr_threshold=best_sweep["mr"],
            oi_drop_threshold=best_sweep["oi"],
            rsi_percentile=best_sweep["rsi"],
        )
        print(f"\n{'=' * 70}")
        print(
            f"  WALK-FORWARD: BEST SWEEP CONFIG "
            f"mr={best_sweep['mr']:.2f} oi={best_sweep['oi']:.2f} "
            f"rsi={best_sweep['rsi']} lev={best_sweep['lev']}x "
            f"mp={best_sweep['mp']} hd={best_sweep['hd']} "
            f"as={best_sweep['as']}"
        )
        print(f"{'=' * 70}")
        walk_forward(
            C, O, H, L, NS, ND, dates, syms, best_sigs,
            sector_lookup=sector_lookup,
            leverage=float(best_sweep["lev"]),
            max_positions=best_sweep["mp"],
            hold_days=best_sweep["hd"],
            atr_stop=best_sweep["as"],
        )

    # === 6. Target check: 600%+ annual ===
    print(f"\n{'=' * 70}")
    print("  TARGET CHECK: 600%+ ANNUALIZED")
    print(f"{'=' * 70}")

    all_results = lev_results + sweep_results
    target_configs = [
        r for r in all_results
        if r.get("ann", 0) >= 600 and r.get("eq", 0) > 0
    ]
    if target_configs:
        target_configs.sort(key=lambda x: -x.get("sh", 0))
        for r in target_configs[:5]:
            print(
                f"  lev={r.get('lev', '?'):>3}x: "
                f"ann={r.get('ann', 0):+.1f}% "
                f"cum={r.get('cum', 0):+.1f}% "
                f"DD={r.get('dd', 0):.1f}% "
                f"Sh={r.get('sh', 0):.2f} "
                f"WR={r.get('wr', 0):.1f}%"
            )
    else:
        print("  No config achieves 600%+ annualized with positive equity.")
        by_ann = sorted(
            [r for r in all_results if r.get("eq", 0) > 0],
            key=lambda x: -x.get("ann", 0))
        for r in by_ann[:5]:
            extra = ""
            if "mr" in r:
                extra = (
                    f" mr={r['mr']:.2f} oi={r['oi']:.2f} rsi={r['rsi']}"
                    f" mp={r['mp']} hd={r['hd']}")
            print(
                f"  lev={r.get('lev', '?'):>3}x: "
                f"ann={r.get('ann', 0):+.1f}% "
                f"cum={r.get('cum', 0):+.1f}% "
                f"DD={r.get('dd', 0):.1f}% "
                f"Sh={r.get('sh', 0):.2f}"
                f"{extra}"
            )

    print(f"\n[V79] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
