"""
Shared utilities for NW kernel strategy backtests (V96/V103/etc).
Extracted to keep strategy files under 800 lines.
"""
import time
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

CASH0 = 1_000_000
COMM = 0.0005
LEVERAGE = 1.0

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

FACTOR_NAMES = [
    "ret_5d", "oi_5d", "rsi14", "vol_5d",
    "ret_10d", "range_5d", "atrp_5d",
]
N_FACTORS = len(FACTOR_NAMES)


def extract_base_symbol(sym: str) -> str:
    s = sym.lower().split('.')[0].strip()
    while s and s[-1].isdigit():
        s = s[:-1]
    if s.endswith('fi'):
        s = s[:-2]
    return s


def build_sector_lookup(syms: List[str]) -> Dict[int, str]:
    sector_lookup: Dict[int, str] = {}
    for si, sym in enumerate(syms):
        base = extract_base_symbol(sym)
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
                valid_g, valid_l = [], []
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
    tag: str = "V",
) -> Dict[str, np.ndarray]:
    """Compute 7 raw factors for NW regression features."""
    t0 = time.time()
    print(f"[{tag}] Computing raw factors...", flush=True)

    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (not np.isnan(C[si, di]) and not np.isnan(C[si, di - 5])
                    and C[si, di - 5] > 0):
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1.0

    oi_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if (not np.isnan(OI[si, di]) and not np.isnan(OI[si, di - 5])
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
            if (not np.isnan(C[si, di]) and not np.isnan(C[si, di - 10])
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

    fwd_ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND - 5):
            if (not np.isnan(C[si, di + 5]) and not np.isnan(C[si, di])
                    and C[si, di] > 0):
                fwd_ret_5d[si, di] = C[si, di + 5] / C[si, di] - 1.0

    atr_mean = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            atr_vals = []
            for j in range(di - 14, di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    prev_c = (
                        C[si, j - 1]
                        if j > 0 and not np.isnan(C[si, j - 1])
                        else cc)
                    atr_vals.append(
                        max(hh - ll, abs(hh - prev_c), abs(ll - prev_c)))
            if atr_vals and not np.isnan(C[si, di]) and C[si, di] > 0:
                atr_mean[si, di] = np.mean(atr_vals) / C[si, di]

    print(f"  Raw factors done: {time.time() - t0:.1f}s", flush=True)
    return {
        "ret_5d": ret_5d, "oi_5d": oi_5d, "vol_5d": vol_5d,
        "range_5d": range_5d, "atrp_5d": atrp_5d,
        "ret_10d": ret_10d, "rsi14": rsi14,
        "fwd_ret_5d": fwd_ret_5d, "atr_mean": atr_mean,
    }


def normalize_factor(
    factor: np.ndarray, NS: int, ND: int, min_count: int = 10,
) -> np.ndarray:
    """Cross-sectional z-score normalization for a factor."""
    normed = np.full((NS, ND), np.nan)
    for di in range(ND):
        vals = factor[:, di]
        valid = vals[~np.isnan(vals)]
        if len(valid) < min_count:
            continue
        mu = np.mean(valid)
        sigma = np.std(valid)
        if sigma < 1e-12:
            continue
        for si in range(NS):
            if not np.isnan(vals[si]):
                normed[si, di] = (vals[si] - mu) / sigma
    return normed


def compute_rolling_ic(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int,
    ic_window: int = 60,
    min_pairs: int = 15,
    tag: str = "V",
) -> np.ndarray:
    """Rolling IC (Spearman rank correlation) for each factor."""
    t0 = time.time()
    print(f"[{tag}] Rolling IC (window={ic_window})...", flush=True)

    fwd_ret = raw_factors["fwd_ret_5d"]
    ic_array = np.full((N_FACTORS, ND), np.nan)

    for fi, fname in enumerate(FACTOR_NAMES):
        factor = raw_factors[fname]
        for di in range(ic_window + 5, ND):
            ic_vals = []
            for tdi in range(di - ic_window, di):
                f_day = factor[:, tdi]
                r_day = fwd_ret[:, tdi]
                valid_mask = (~np.isnan(f_day)) & (~np.isnan(r_day))
                f_valid = f_day[valid_mask]
                r_valid = r_day[valid_mask]
                if len(f_valid) >= min_pairs:
                    f_rank = pd.Series(f_valid).rank().values
                    r_rank = pd.Series(r_valid).rank().values
                    corr = np.corrcoef(f_rank, r_rank)[0, 1]
                    if not np.isnan(corr):
                        ic_vals.append(corr)
            if len(ic_vals) >= 5:
                ic_array[fi, di] = np.mean(ic_vals)
        if fi % 2 == 0:
            print(f"  IC for {fname}: {time.time() - t0:.1f}s", flush=True)

    print(f"  Rolling IC done: {time.time() - t0:.1f}s", flush=True)
    return ic_array


def compute_bma_weights(
    ic_array: np.ndarray,
    ND: int,
    prior_strength: float = 5.0,
    min_ic_history: int = 20,
    tag: str = "V",
) -> np.ndarray:
    """BMA weights from IC history using Beta posterior."""
    t0 = time.time()
    print(f"[{tag}] BMA weights (prior={prior_strength:.1f})...", flush=True)

    weights = np.full((N_FACTORS, ND), np.nan)
    for fi in range(N_FACTORS):
        for di in range(min_ic_history, ND):
            ic_hist = ic_array[fi, max(0, di - 120):di]
            valid_ic = ic_hist[~np.isnan(ic_hist)]
            if len(valid_ic) < 5:
                continue
            n_pos = np.sum(valid_ic > 0)
            n_neg = len(valid_ic) - n_pos
            alpha_p = prior_strength / 2.0 + n_pos
            beta_p = prior_strength / 2.0 + n_neg
            weights[fi, di] = alpha_p / (alpha_p + beta_p)

    for di in range(ND):
        w = weights[:, di]
        valid = w[~np.isnan(w)]
        if len(valid) == N_FACTORS:
            w_sum = np.nansum(w)
            if w_sum > 0:
                weights[:, di] = w / w_sum

    print(f"  BMA weights done: {time.time() - t0:.1f}s", flush=True)
    return weights


def apply_bma_to_features(
    normed_factors: Dict[str, np.ndarray],
    bma_weights: np.ndarray,
    NS: int, ND: int,
) -> Dict[str, np.ndarray]:
    """Scale each factor by its BMA weight."""
    t0 = time.time()
    weighted = {}
    for fi, fname in enumerate(FACTOR_NAMES):
        original = normed_factors[fname]
        result = np.full((NS, ND), np.nan)
        for di in range(ND):
            w = bma_weights[fi, di]
            if np.isnan(w):
                w = 1.0 / N_FACTORS
            for si in range(NS):
                if not np.isnan(original[si, di]):
                    result[si, di] = original[si, di] * (w * N_FACTORS)
        weighted[fname] = result
    print(f"  BMA weighting done: {time.time() - t0:.1f}s", flush=True)
    return weighted


def compute_ker(C: np.ndarray, NS: int, ND: int) -> np.ndarray:
    """Kaufman efficiency ratio for regime detection."""
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


def compute_portfolio_volatility(
    C: np.ndarray, NS: int, ND: int, vol_lookback: int = 20,
) -> np.ndarray:
    """Rolling portfolio volatility proxy."""
    port_vol = np.full(ND, np.nan)
    for di in range(vol_lookback + 1, ND):
        daily_rets = []
        for dd in range(di - vol_lookback, di):
            rets = []
            for si in range(NS):
                if (not np.isnan(C[si, dd]) and not np.isnan(C[si, dd - 1])
                        and C[si, dd - 1] > 0):
                    rets.append(C[si, dd] / C[si, dd - 1] - 1.0)
            if rets:
                daily_rets.append(np.mean(rets))
        if len(daily_rets) >= vol_lookback // 2:
            port_vol[di] = np.std(daily_rets)
    return port_vol


def get_vol_multiplier(
    port_vol: float, vol_median: float,
    vol_high_mult: float, vol_low_mult: float,
    size_reduce: float, size_boost: float,
) -> float:
    """Position size multiplier based on volatility regime."""
    if np.isnan(port_vol) or np.isnan(vol_median) or vol_median < 1e-12:
        return 1.0
    ratio = port_vol / vol_median
    if ratio > vol_high_mult:
        return size_reduce
    elif ratio < vol_low_mult:
        return size_boost
    return 1.0


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


def analyze(
    trades: List[dict], equity: float, max_dd: float,
    label: str = "",
) -> Optional[dict]:
    """Analyze and print trade results."""
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
    mode_counts = {"W": 0, "N": 0, "L": 0}
    for t in trades:
        m = t.get("mode", "N")
        if m in mode_counts:
            mode_counts[m] += 1

    sector_counts: Dict[str, int] = defaultdict(int)
    for t in trades:
        sector_counts[t.get("sector", "OTHER")] += 1
    sector_str = " ".join(
        f"{k}:{v}" for k, v in sorted(sector_counts.items()))

    avg_pnl = np.mean([t["pnl_pct"] for t in trades])
    avg_win = np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"] > 0])
    avg_loss = np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"] <= 0])

    print(
        f"  {label}: {len(trades)}t (stop:{n_stop} hold:{n_hold}) "
        f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
        f"Sh={sh:.2f} eq={equity:,.0f}")
    print(
        f"    avg_pnl={avg_pnl:+.3f}% avg_win={avg_win:+.3f}% "
        f"avg_loss={avg_loss:+.3f}% "
        f"modes=[W:{mode_counts['W']} N:{mode_counts['N']} "
        f"L:{mode_counts['L']}]")
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
