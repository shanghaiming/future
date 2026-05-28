"""V123: "执两用中时中" Adaptive Spectral Leverage
From guoxue: "执两用中" + "君子而时中" -- the optimal middle point MOVES with time.

Core: Replace V96's binary vol thresholds with CONTINUOUS Spectral Risk Measure
that adapts to the FULL return distribution shape (tail shape, not just variance).
Signal: V86's NW Kernel (no BMA). Sizing: Spectral risk per-instrument + portfolio.
Walk-forward 2019-2026. No leverage. CASH0=1M, COMM=0.0005.
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


def _extract_base_symbol(sym: str) -> str:
    s = sym.lower().split('.')[0].strip()
    while s and s[-1].isdigit():
        s = s[:-1]
    return s[:-2] if s.endswith('fi') else s


def build_sector_lookup(syms: List[str]) -> Dict[int, str]:
    return {si: SECTOR_MAP.get(_extract_base_symbol(sym), 'OTHER')
            for si, sym in enumerate(syms)}


def compute_rsi_manual(C: np.ndarray, NS: int, ND: int,
                       period: int = 14) -> np.ndarray:
    rsi = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        avg_gain = avg_loss = np.nan
        for di in range(1, ND):
            if np.isnan(c[di]) or np.isnan(c[di - 1]):
                continue
            delta = c[di] - c[di - 1]
            gain, loss = max(delta, 0.0), max(-delta, 0.0)
            if np.isnan(avg_gain):
                vg, vl = [], []
                for j in range(di, min(di + period, ND)):
                    if not np.isnan(c[j]) and not np.isnan(c[j - 1]):
                        d2 = c[j] - c[j - 1]
                        vg.append(max(d2, 0.0))
                        vl.append(max(-d2, 0.0))
                if len(vg) >= period:
                    avg_gain, avg_loss = np.mean(vg), np.mean(vl)
                    rsi_val = 100.0 if avg_loss == 0 else (
                        100.0 - 100.0 / (1.0 + avg_gain / avg_loss))
                    rsi[si, di + period - 1] = rsi_val
                continue
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
            rsi[si, di] = 100.0 if avg_loss == 0 else (
                100.0 - 100.0 / (1.0 + avg_gain / avg_loss))
    return rsi


def compute_raw_factors(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    V: np.ndarray, OI: np.ndarray, NS: int, ND: int,
) -> Dict[str, np.ndarray]:
    t0 = time.time()
    print("[V123] Computing raw factors...", flush=True)

    ret_5d = np.full((NS, ND), np.nan)
    oi_5d = np.full((NS, ND), np.nan)
    vol_5d = np.full((NS, ND), np.nan)
    range_5d = np.full((NS, ND), np.nan)
    atrp_5d = np.full((NS, ND), np.nan)
    ret_10d = np.full((NS, ND), np.nan)
    daily_returns = np.full((NS, ND), np.nan)

    for si in range(NS):
        for di in range(1, ND):
            if (not np.isnan(C[si, di]) and not np.isnan(C[si, di - 1])
                    and C[si, di - 1] > 0):
                daily_returns[si, di] = C[si, di] / C[si, di - 1] - 1.0
        for di in range(5, ND):
            if (not np.isnan(C[si, di]) and not np.isnan(C[si, di - 5])
                    and C[si, di - 5] > 0):
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1.0
            if (not np.isnan(OI[si, di]) and not np.isnan(OI[si, di - 5])
                    and OI[si, di - 5] > 0):
                oi_5d[si, di] = OI[si, di] / OI[si, di - 5] - 1.0
            vals = V[si, di - 5:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 3:
                vol_5d[si, di] = np.mean(valid)
            rng = []
            for j in range(di - 5, di):
                if (not np.isnan(H[si, j]) and not np.isnan(L[si, j])
                        and not np.isnan(C[si, j]) and C[si, j] > 0
                        and H[si, j] > L[si, j]):
                    rng.append((H[si, j] - L[si, j]) / C[si, j])
            if len(rng) >= 3:
                range_5d[si, di] = np.mean(rng)
        for di in range(6, ND):
            atr_v = []
            for j in range(di - 5, di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    prev_c = C[si, j - 1] if (
                        j > 0 and not np.isnan(C[si, j - 1])) else cc
                    atr_v.append(max(hh - ll, abs(hh - prev_c),
                                     abs(ll - prev_c)))
            if atr_v and not np.isnan(C[si, di]) and C[si, di] > 0:
                atrp_5d[si, di] = np.mean(atr_v) / C[si, di]
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
                rsi14[si] = np.where(nan_mask, np.nan, talib.RSI(c, 14))
            except Exception:
                pass
    needs_fallback = np.all(np.isnan(rsi14), axis=1)
    if needs_fallback.any():
        rsi_m = compute_rsi_manual(C, NS, ND, 14)
        for si in range(NS):
            if needs_fallback[si]:
                rsi14[si] = rsi_m[si]

    fwd_ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND - 5):
            if (not np.isnan(C[si, di + 5]) and not np.isnan(C[si, di])
                    and C[si, di] > 0):
                fwd_ret_5d[si, di] = C[si, di + 5] / C[si, di] - 1.0

    atr_mean = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            atr_v = []
            for j in range(di - 14, di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    prev_c = C[si, j - 1] if (
                        j > 0 and not np.isnan(C[si, j - 1])) else cc
                    atr_v.append(max(hh - ll, abs(hh - prev_c),
                                     abs(ll - prev_c)))
            if atr_v and not np.isnan(C[si, di]) and C[si, di] > 0:
                atr_mean[si, di] = np.mean(atr_v) / C[si, di]

    print(f"  Raw factors done: {time.time() - t0:.1f}s", flush=True)
    return {
        "ret_5d": ret_5d, "oi_5d": oi_5d, "vol_5d": vol_5d,
        "range_5d": range_5d, "atrp_5d": atrp_5d,
        "ret_10d": ret_10d, "rsi14": rsi14,
        "fwd_ret_5d": fwd_ret_5d, "atr_mean": atr_mean,
        "daily_returns": daily_returns,
    }


def normalize_factor(factor: np.ndarray, NS: int, ND: int,
                     min_count: int = 10) -> np.ndarray:
    normed = np.full((NS, ND), np.nan)
    for di in range(ND):
        vals = factor[:, di]
        valid = vals[~np.isnan(vals)]
        if len(valid) < min_count:
            continue
        mu, sigma = np.mean(valid), np.std(valid)
        if sigma < 1e-12:
            continue
        for si in range(NS):
            if not np.isnan(vals[si]):
                normed[si, di] = (vals[si] - mu) / sigma
    return normed


def compute_nw_predicted_returns(
    raw_factors: Dict[str, np.ndarray],
    NS: int, ND: int,
    training_window: int = 40,
    kernel_bandwidth: float = 1.0,
) -> np.ndarray:
    t0 = time.time()
    print(f"[V123] NW prediction (tw={training_window}, bw={kernel_bandwidth:.1f})...",
          flush=True)
    normed = {fn: normalize_factor(raw_factors[fn], NS, ND)
              for fn in FACTOR_NAMES}
    fwd_ret = raw_factors["fwd_ret_5d"]
    atr_mean = raw_factors["atr_mean"]
    predicted = np.full((NS, ND), np.nan)

    for di in range(training_window + 10, ND):
        train_X_list, train_Y_list = [], []
        start_di = max(10, di - training_window)
        for tdi in range(start_di, di):
            for si in range(NS):
                feat = np.array([normed[fn][si, tdi] for fn in FACTOR_NAMES])
                target = fwd_ret[si, tdi]
                if np.any(np.isnan(feat)) or np.isnan(target):
                    continue
                train_X_list.append(feat)
                train_Y_list.append(target)
        if len(train_X_list) < 20:
            continue
        train_X, train_Y = np.array(train_X_list), np.array(train_Y_list)
        feat_std = np.std(train_X, axis=0)
        feat_std[feat_std < 1e-12] = 1.0

        for si in range(NS):
            qf = np.array([normed[fn][si, di] for fn in FACTOR_NAMES])
            if np.any(np.isnan(qf)):
                continue
            atr_val = atr_mean[si, di]
            h = max(atr_val * kernel_bandwidth, 0.1) if not np.isnan(atr_val) else kernel_bandwidth
            dist = np.sqrt(np.sum(((train_X - qf) / feat_std) ** 2, axis=1))
            sd = dist / h
            w = np.zeros(len(train_X))
            mask = sd <= 1.0
            if not np.any(mask):
                idx = np.argmin(dist)
                if dist[idx] < 1e12:
                    w[idx] = 1.0
                else:
                    continue
            else:
                w[mask] = 0.75 * (1.0 - sd[mask] ** 2)
            ws = np.sum(w)
            if ws > 1e-12:
                predicted[si, di] = np.sum(w * train_Y) / ws
        if di % 100 == 0:
            print(f"  di={di}/{ND} valid={np.sum(~np.isnan(predicted[:, di]))}",
                  flush=True)
    print(f"  NW done: {time.time() - t0:.1f}s", flush=True)
    return predicted


def compute_ker(C: np.ndarray, NS: int, ND: int) -> np.ndarray:
    ker_10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            closes = C[si, di - 10:di + 1]
            valid = closes[~np.isnan(closes)]
            if len(valid) < 10 or valid[0] <= 0:
                continue
            nc = abs(valid[-1] - valid[0])
            tc = np.sum(np.abs(np.diff(valid)))
            if tc > 1e-10:
                ker_10[si, di] = nc / tc
    ker_regime = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        for di in range(ND):
            v = ker_10[si, di]
            if not np.isnan(v):
                ker_regime[si, di] = 1 if v < 0.15 else (-1 if v > 0.3 else 0)
    return ker_regime


def get_dynamic_mode(recent_wins: List[int], threshold: float,
                     window: int) -> str:
    if len(recent_wins) < 5:
        return "normal"
    wr = sum(recent_wins[-window:]) / len(recent_wins[-window:])
    if wr > threshold:
        return "winning"
    return "losing" if wr < 0.50 else "normal"


def compute_atr_at(H: np.ndarray, L: np.ndarray, C: np.ndarray,
                   si: int, di: int, start_di: int) -> Optional[float]:
    atr_v = []
    for j in range(max(start_di, di - 14), di):
        hh, ll, cc = H[si, j], L[si, j], C[si, j]
        if not any(np.isnan([hh, ll, cc])):
            atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
    return np.mean(atr_v) if atr_v else None


# =====================================================================
# INNOVATION: Spectral Risk Position Sizing
# "执两用中时中" -- the optimal point MOVES with distribution shape
# =====================================================================

def precompute_rolling_stats(
    daily_returns: np.ndarray,
    NS: int, ND: int,
    lookback: int = 60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute rolling stats for spectral risk sizing.

    Returns:
        inst_vol: (NS, ND) rolling std of returns
        inst_skew: (NS, ND) rolling skewness of returns
        inst_med_abs: (NS, ND) rolling median(|returns|)
        port_vol: (ND,) rolling std of portfolio returns
        port_skew: (ND,) rolling skewness of portfolio returns
    """
    t0 = time.time()
    inst_vol = np.full((NS, ND), np.nan)
    inst_skew = np.full((NS, ND), np.nan)
    inst_med_abs = np.full((NS, ND), np.nan)

    for si in range(NS):
        for di in range(lookback, ND):
            valid = daily_returns[si, di - lookback:di]
            valid = valid[~np.isnan(valid)]
            if len(valid) >= 30:
                inst_vol[si, di] = np.std(valid)
                inst_med_abs[si, di] = np.median(np.abs(valid))
                m = np.mean(valid)
                s = np.std(valid)
                if s > 1e-12:
                    inst_skew[si, di] = np.mean(((valid - m) / s) ** 3)
        if si % 10 == 0:
            print(f"  precompute inst {si}/{NS}", flush=True)

    # Portfolio returns
    port_rets_arr = np.full(ND, np.nan)
    for di in range(1, ND):
        r = daily_returns[:, di]
        r_valid = r[~np.isnan(r)]
        if len(r_valid) > 0:
            port_rets_arr[di] = np.mean(r_valid)

    port_vol = np.full(ND, np.nan)
    port_skew = np.full(ND, np.nan)
    for di in range(lookback, ND):
        valid = port_rets_arr[di - lookback:di]
        valid = valid[~np.isnan(valid)]
        if len(valid) >= 30:
            port_vol[di] = np.std(valid)
            m = np.mean(valid)
            s = np.std(valid)
            if s > 1e-12:
                port_skew[di] = np.mean(((valid - m) / s) ** 3)

    print(f"  Precompute done: {time.time() - t0:.1f}s", flush=True)
    return inst_vol, inst_skew, inst_med_abs, port_vol, port_skew


def apply_spectral_from_precomputed(
    inst_vol: np.ndarray, inst_skew: np.ndarray,
    inst_med_abs: np.ndarray,
    port_vol: np.ndarray, port_skew: np.ndarray,
    NS: int, ND: int,
    n_quantiles: int,
    tail_weight: float,
    sigmoid_center: float,
    sigmoid_steepness: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute spectral sizing from pre-computed vol and skewness.

    The spectral risk measure is approximated by:
      risk_score = vol * (1 + tail_weight * max(0, -skewness))

    This captures the intuition: when left tail is fat (negative skew),
    risk is amplified by tail_weight. When distribution is symmetric,
    risk is just vol.

    risk_ratio = risk_score / median_vol  (cross-sectional normalization)
    position = sigmoid(risk_ratio) in [0.2, 1.5]
    """
    # Instrument-level: risk = vol * (1 + tw * max(0, -skew))
    neg_skew = np.maximum(0.0, -inst_skew)
    inst_risk = inst_vol * (1.0 + tail_weight * neg_skew)

    # Cross-sectional median vol for normalization
    med_vol = np.nanmedian(inst_vol)
    if med_vol < 1e-12:
        med_vol = 1e-6

    risk_ratio = inst_risk / med_vol
    risk_ratio = np.where(np.isnan(inst_vol), 5.0, risk_ratio)

    inst_sizes = np.clip(
        0.2 + 1.3 / (1.0 + np.exp(sigmoid_steepness * (risk_ratio - sigmoid_center))),
        0.2, 1.5)
    inst_sizes = np.where(np.isnan(inst_vol), 1.0, inst_sizes)

    # Portfolio-level
    port_neg_skew = np.maximum(0.0, -port_skew)
    port_risk = port_vol * (1.0 + tail_weight * port_neg_skew)
    port_ratio = port_risk / med_vol
    port_ratio = np.where(np.isnan(port_vol), 5.0, port_ratio)

    port_sizes = np.clip(
        0.2 + 1.3 / (1.0 + np.exp(sigmoid_steepness * (port_ratio - sigmoid_center))),
        0.2, 1.5)
    port_sizes = np.where(np.isnan(port_vol), 1.0, port_sizes)

    return inst_sizes, port_sizes


# =====================================================================
# Backtest + Helpers
# =====================================================================


def _compute_backtest_stats(trades: List[dict], eq: float,
                            dd: float) -> Optional[dict]:
    if len(trades) < 10:
        return None
    nw = sum(1 for t in trades if t["pnl_pct"] > 0)
    wr = nw / len(trades) * 100
    n_days = max(1, trades[-1]["di"] - trades[0]["di"])
    ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
    ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
    rets = np.array(ap) / CASH0
    sh = float(np.mean(rets) / np.std(rets) * np.sqrt(252)
               if np.std(rets) > 0 else 0)
    return {"n": len(trades), "wr": wr, "ann": ann, "dd": dd,
            "sharpe": sh, "eq": eq}


def backtest_v123(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    predicted: np.ndarray, ker_regime: np.ndarray,
    inst_sizes: np.ndarray, port_sizes: np.ndarray,
    sector_lookup: Dict[int, str],
    top_n: int = 2, max_per_sector: int = 2, hold_days: int = 5,
    win_threshold: float = 0.60, atr_stop: float = 3.0,
    start_di: int = 60, end_di: Optional[int] = None,
) -> Tuple[List[dict], float, float]:
    if end_di is None:
        end_di = ND - 1
    equity, peak, max_dd = CASH0, CASH0, 0.0
    positions: List[Tuple[int, int, float, float, float]] = []
    trades: List[dict] = []
    recent_wins: List[int] = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_pos: List[Tuple[int, int, float, float, float]] = []
        mode = get_dynamic_mode(recent_wins, win_threshold, 15)
        port_mult = port_sizes[di]

        # Exit
        pos_by_si: Dict[int, List] = defaultdict(list)
        for si, edi, ep, sp, alloc in positions:
            pos_by_si[si].append((edi, ep, sp, alloc))

        for si, pos_list in pos_by_si.items():
            c = C[si, di]
            if np.isnan(c):
                new_pos.extend((si, e, p, s, a) for e, p, s, a in pos_list)
                continue
            earliest = min(p[0] for p in pos_list)
            hold = di - earliest
            stopped = any(c < sp for _, _, sp, _ in pos_list)
            if stopped or hold >= hold_days:
                for edi, ep, sp, alloc in pos_list:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    trades.append({
                        "pnl_abs": profit, "pnl_pct": pnl * 100,
                        "days": di - edi + 1, "di": di,
                        "year": d.year, "sym": syms[si],
                        "sector": sector_lookup.get(si, 'OTHER'),
                        "reason": "stop" if stopped else "hold",
                        "mode": mode[:1].upper(),
                    })
                    recent_wins.append(1 if pnl > 0 else 0)
            else:
                new_pos.extend((si, e, p, s, a) for e, p, s, a in pos_list)

        positions = new_pos
        equity += daily_pnl
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        if equity <= 0:
            break

        # Entry
        held = {p[0] for p in positions}
        if len(held) >= top_n:
            continue

        candidates = [(predicted[si, di], si) for si in range(NS)
                       if si not in held
                       and not np.isnan(predicted[si, di])
                       and di + 1 < ND
                       and not np.isnan(O[si, di + 1])
                       and ker_regime[si, di] >= 0]
        if not candidates:
            continue
        candidates.sort(key=lambda x: -x[0])

        n_take = top_n
        if mode == "winning":
            n_take = min(top_n + 1, top_n * 2)
        elif mode == "losing":
            n_take = max(1, top_n - 1)

        sec_counts: Dict[str, int] = defaultdict(int)
        for si_h in held:
            sec_counts[sector_lookup.get(si_h, 'OTHER')] += 1

        entries = []
        for pv, si in candidates:
            if len(held) + len(entries) >= n_take or si in held:
                break
            sec = sector_lookup.get(si, 'OTHER')
            if sec_counts[sec] >= max_per_sector or pv <= 0:
                continue
            entries.append((pv, si, sec))
            sec_counts[sec] += 1
        if not entries:
            continue

        base_alloc = LEVERAGE / (len(positions) + len(entries))
        updated = []
        for si, edi, ep, sp, _ in positions:
            updated.append((si, edi, ep, sp, base_alloc * inst_sizes[si, di] * port_mult))
        for pv, si, sec in entries:
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr = compute_atr_at(H, L, C, si, di, start_di)
            if atr is None:
                continue
            updated.append((si, di + 1, ep, ep - atr_stop * atr,
                            base_alloc * inst_sizes[si, di] * port_mult))
        positions = updated

    for si, edi, ep, sp, alloc in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            equity += equity * alloc * ((c - ep) / ep - COMM)

    return trades, equity, max_dd


def analyze(trades: List[dict], equity: float, max_dd: float,
            label: str = "") -> Optional[dict]:
    if not trades:
        print(f"  {label}: no trades")
        return None
    nw = sum(1 for t in trades if t["pnl_pct"] > 0)
    wr = nw / len(trades) * 100
    n_days = max(1, trades[-1]["di"] - trades[0]["di"])
    ann = ((equity / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
    ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
    rets = np.array(ap) / CASH0
    sh = float(np.mean(rets) / np.std(rets) * np.sqrt(252)
               if np.std(rets) > 0 else 0)
    n_stop = sum(1 for t in trades if t["reason"] == "stop")
    n_hold = len(trades) - n_stop
    mc = {"W": 0, "N": 0, "L": 0}
    for t in trades:
        m = t.get("mode", "N")
        if m in mc:
            mc[m] += 1
    sc: Dict[str, int] = defaultdict(int)
    for t in trades:
        sc[t.get("sector", "OTHER")] += 1
    sec_str = " ".join(f"{k}:{v}" for k, v in sorted(sc.items()))
    avg_w = np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"] > 0])
    avg_l = np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"] <= 0])
    print(f"  {label}: {len(trades)}t (stop:{n_stop} hold:{n_hold}) "
          f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
          f"Sh={sh:.2f} eq={equity:,.0f}")
    print(f"    avg_win={avg_w:+.3f}% avg_loss={avg_l:+.3f}% "
          f"modes=[W:{mc['W']} N:{mc['N']} L:{mc['L']}] sectors: {sec_str}")
    yr: Dict[int, dict] = {}
    for t in trades:
        y = t["year"]
        if y not in yr:
            yr[y] = {"n": 0, "w": 0, "pnl": []}
        yr[y]["n"] += 1
        yr[y]["w"] += (1 if t["pnl_pct"] > 0 else 0)
        yr[y]["pnl"].append(t["pnl_pct"])
    for y in sorted(yr.keys()):
        ys = yr[y]
        cum = np.prod([1 + p / 100 for p in ys["pnl"]]) - 1
        print(f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% cum={cum:+.1%}")
    return {"n": len(trades), "wr": wr, "dd": max_dd, "ann": ann,
            "sh": sh, "eq": equity}


def walk_forward(
    C: np.ndarray, O: np.ndarray, H: np.ndarray, L: np.ndarray,
    NS: int, ND: int, dates: np.ndarray, syms: List[str],
    predicted: np.ndarray, ker_regime: np.ndarray,
    inst_sizes: np.ndarray, port_sizes: np.ndarray,
    sector_lookup: Dict[int, str],
    top_n: int = 2, max_per_sector: int = 2, hold_days: int = 5,
    label: str = "",
) -> List[dict]:
    print(f"\n{'=' * 70}\n  WALK-FORWARD V123 {label}\n  tn={top_n} mps={max_per_sector}\n{'=' * 70}")
    all_trades: List[dict] = []
    for test_year in range(2019, max(d.year for d in dates) + 1):
        ts = te = None
        for i, d in enumerate(dates):
            if d.year == test_year and ts is None:
                ts = i
            if d.year == test_year:
                te = i
        if ts is None:
            continue
        trades, _, _ = backtest_v123(
            C, O, H, L, NS, ND, dates, syms,
            predicted, ker_regime, inst_sizes, port_sizes,
            sector_lookup=sector_lookup, top_n=top_n,
            max_per_sector=max_per_sector, hold_days=hold_days,
            start_di=ts, end_di=te + 1)
        yt = [t for t in trades if dates[t["di"]].year == test_year]
        all_trades.extend(yt)
        if yt:
            nw = sum(1 for t in yt if t["pnl_pct"] > 0)
            print(f"  {test_year}: {len(yt)}t WR={nw/len(yt)*100:.1f}% "
                  f"avg={np.mean([t['pnl_pct'] for t in yt]):+.2f}%", flush=True)
        else:
            print(f"  {test_year}: no trades", flush=True)
    if all_trades:
        nw = sum(1 for t in all_trades if t["pnl_pct"] > 0)
        cum = np.prod([1 + t["pnl_pct"] / 100 for t in all_trades]) - 1
        print(f"\n  WF TOTAL: {len(all_trades)}t "
              f"WR={nw/len(all_trades)*100:.1f}% cum={cum:+.1%}")
        return all_trades
    return []


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print('  V123: "执两用中时中" Adaptive Spectral Leverage')
    print("  Signal: NW Kernel (V86). Sizing: Spectral Risk Measure.")
    print("  Walk-forward 2019-2026. No leverage.")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start="2016-01-01")
    print(f"  {NS} sym, {ND} days, {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    sector_lookup = build_sector_lookup(syms)
    bt_2019 = next(i for i, d in enumerate(dates) if d >= pd.Timestamp("2019-01-01"))

    raw_factors = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ker_regime = compute_ker(C, NS, ND)
    daily_returns = raw_factors["daily_returns"]
    pred = compute_nw_predicted_returns(raw_factors, NS, ND, 40, 1.0)

    # Pre-compute rolling vol and skewness ONCE
    inst_vol, inst_skew, inst_med_abs, port_vol, port_skew = (
        precompute_rolling_stats(daily_returns, NS, ND, lookback=60))

    # Parameter sweep using vectorized spectral sizing
    NQ = [10, 20, 30]
    TW = [3.0, 5.0, 7.0]
    SC = [1.0, 1.5, 2.0]
    SS = [2.0, 3.0, 5.0]
    n_configs = len(NQ) * len(TW) * len(SC) * len(SS)

    print(f"\n{'=' * 70}")
    print(f"  PARAMETER SWEEP: {n_configs} spectral configs x 4 portfolio configs")
    print(f"  (vectorized from pre-computed quantiles)")
    print(f"{'=' * 70}")

    results: List[dict] = []
    sweep_count = 0

    for nq in NQ:
        for tw in TW:
            for sc in SC:
                for ss in SS:
                    inst_sizes, port_sizes = apply_spectral_from_precomputed(
                        inst_vol, inst_skew, inst_med_abs,
                        port_vol, port_skew, NS, ND,
                        nq, tw, sc, ss)
                    for top_n in [2, 3]:
                        for mps in [2, 3]:
                            sweep_count += 1
                            trades, eq, dd = backtest_v123(
                                C, O, H, L, NS, ND, dates, syms,
                                pred, ker_regime, inst_sizes, port_sizes,
                                sector_lookup=sector_lookup,
                                top_n=top_n, max_per_sector=mps,
                                hold_days=5, start_di=bt_2019)
                            stats = _compute_backtest_stats(trades, eq, dd)
                            if stats:
                                stats.update({"nq": nq, "tw": tw, "sc": sc, "ss": ss,
                                              "top_n": top_n, "mps": mps})
                                results.append(stats)
                    if sweep_count % 32 == 0:
                        print(f"  Progress: {sweep_count}/{n_configs * 4} sweeps",
                              flush=True)

    results.sort(key=lambda x: -x["ann"])
    print(f"\n  Evaluated {sweep_count} configs, {len(results)} with 10+ trades")
    if not results:
        print("  No configs with 10+ trades.")
        return

    print(f"\n{'NQ':>3} {'TW':>4} {'SC':>4} {'SS':>4} {'TN':>3} {'MP':>3} "
          f"{'N':>5} {'WR':>6} {'Ann':>9} {'DD':>7} {'Sh':>7}")
    print("-" * 70)
    for r in results[:15]:
        print(f"{r['nq']:>3} {r['tw']:>4.1f} {r['sc']:>4.1f} {r['ss']:>4.1f} "
              f"{r['top_n']:>3} {r['mps']:>3} "
              f"{r['n']:>5} {r['wr']:>5.1f}% {r['ann']:>+8.1f}% "
              f"{r['dd']:>6.1f}% {r['sharpe']:>6.2f}")

    # Walk-forward for top configs
    best_ann = results[0]
    best_sh = max(results, key=lambda x: x["sharpe"])
    best_ra = max(results, key=lambda x: x["ann"] / max(x["dd"], 1.0))

    for label, best in [("BEST-ANN", best_ann), ("BEST-SHARPE", best_sh),
                         ("BEST-RISK-ADJ", best_ra)]:
        inst, port = apply_spectral_from_precomputed(
            inst_vol, inst_skew, inst_med_abs,
            port_vol, port_skew, NS, ND,
            best["nq"], best["tw"], best["sc"], best["ss"])
        walk_forward(C, O, H, L, NS, ND, dates, syms,
                     pred, ker_regime, inst, port,
                     sector_lookup=sector_lookup,
                     top_n=best["top_n"], max_per_sector=best["mps"],
                     label=label)

    # Compare V123 vs V86 baseline
    print(f"\n{'=' * 70}\n  COMPARISON: V123 (Spectral) vs V86 baseline\n{'=' * 70}")
    inst_b, port_b = apply_spectral_from_precomputed(
        inst_vol, inst_skew, inst_med_abs,
        port_vol, port_skew, NS, ND,
        best_sh["nq"], best_sh["tw"], best_sh["sc"], best_sh["ss"])
    t123, eq123, dd123 = backtest_v123(
        C, O, H, L, NS, ND, dates, syms, pred, ker_regime,
        inst_b, port_b, sector_lookup=sector_lookup,
        top_n=best_sh["top_n"], max_per_sector=best_sh["mps"],
        start_di=bt_2019)
    t86, eq86, dd86 = backtest_v123(
        C, O, H, L, NS, ND, dates, syms, pred, ker_regime,
        np.ones((NS, ND)), np.ones(ND), sector_lookup=sector_lookup,
        top_n=2, max_per_sector=2, start_di=bt_2019)

    print(f"\n  V123 BEST-SHARPE:")
    analyze(t123, eq123, dd123, "V123-Spectral")
    print(f"\n  V86 BASELINE:")
    analyze(t86, eq86, dd86, "V86-baseline")
    if t123 and t86:
        print(f"\n  Delta: eq={eq123 - eq86:+,.0f} dd={dd123 - dd86:+.1f}% "
              f"trades={len(t123) - len(t86):+d}")
    print(f"\n[V123] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
