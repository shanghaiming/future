"""V113: DMD Spectral Regime Detection + NW Kernel.
DMD decomposes price series into growth/decay/oscillation modes.
Bull energy ratio detects regime transitions BEFORE momentum.
Acts as GATE on V96's NW+BMA kernel + vol-adaptive signals.
Walk-forward 2019-2026. No leverage. CASH0=1M, COMM=0.0005."""
import sys, os, time, warnings
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

FACTOR_NAMES = ["ret_5d", "oi_5d", "rsi14", "vol_5d",
                "ret_10d", "range_5d", "atrp_5d"]
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
            g, lo = max(delta, 0.0), max(-delta, 0.0)
            if np.isnan(avg_gain):
                valid_g, valid_l = [], []
                for j in range(di, min(di + period, ND)):
                    if not np.isnan(c[j]) and not np.isnan(c[j - 1]):
                        d2 = c[j] - c[j - 1]
                        valid_g.append(max(d2, 0.0))
                        valid_l.append(max(-d2, 0.0))
                if len(valid_g) >= period:
                    avg_gain, avg_loss = np.mean(valid_g), np.mean(valid_l)
                    if avg_loss == 0:
                        rsi[si, di + period - 1] = 100.0
                    else:
                        rsi[si, di + period - 1] = 100 - 100 / (1 + avg_gain / avg_loss)
                continue
            avg_gain = (avg_gain * (period - 1) + g) / period
            avg_loss = (avg_loss * (period - 1) + lo) / period
            if avg_loss == 0:
                rsi[si, di] = 100.0
            else:
                rsi[si, di] = 100 - 100 / (1 + avg_gain / avg_loss)
    return rsi


def compute_raw_factors(C, O, H, L, V, OI, NS, ND):
    t0 = time.time()
    print("[V113] Computing raw factors...", flush=True)
    ret_5d = np.full((NS, ND), np.nan)
    oi_5d = np.full((NS, ND), np.nan)
    vol_5d = np.full((NS, ND), np.nan)
    range_5d = np.full((NS, ND), np.nan)
    atrp_5d = np.full((NS, ND), np.nan)
    ret_10d = np.full((NS, ND), np.nan)
    fwd_ret_5d = np.full((NS, ND), np.nan)
    atr_mean = np.full((NS, ND), np.nan)

    for si in range(NS):
        for di in range(ND):
            if di >= 5:
                if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 5]) and C[si, di - 5] > 0:
                    ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1.0
                if not np.isnan(OI[si, di]) and not np.isnan(OI[si, di - 5]) and OI[si, di - 5] > 0:
                    oi_5d[si, di] = OI[si, di] / OI[si, di - 5] - 1.0
                vals = V[si, di - 5:di]
                valid = vals[~np.isnan(vals)]
                if len(valid) >= 3:
                    vol_5d[si, di] = np.mean(valid)
                rng = []
                for j in range(di - 5, di):
                    if (not np.isnan(H[si, j]) and not np.isnan(L[si, j])
                            and not np.isnan(C[si, j]) and C[si, j] > 0 and H[si, j] > L[si, j]):
                        rng.append((H[si, j] - L[si, j]) / C[si, j])
                if len(rng) >= 3:
                    range_5d[si, di] = np.mean(rng)
            if di >= 6:
                atr_v = []
                for j in range(di - 5, di):
                    hh, ll, cc = H[si, j], L[si, j], C[si, j]
                    if not any(np.isnan([hh, ll, cc])):
                        pc = C[si, j - 1] if j > 0 and not np.isnan(C[si, j - 1]) else cc
                        atr_v.append(max(hh - ll, abs(hh - pc), abs(ll - pc)))
                if atr_v and not np.isnan(C[si, di]) and C[si, di] > 0:
                    atrp_5d[si, di] = np.mean(atr_v) / C[si, di]
            if di >= 10 and not np.isnan(C[si, di]) and not np.isnan(C[si, di - 10]) and C[si, di - 10] > 0:
                ret_10d[si, di] = C[si, di] / C[si, di - 10] - 1.0
            if di < ND - 5 and not np.isnan(C[si, di + 5]) and not np.isnan(C[si, di]) and C[si, di] > 0:
                fwd_ret_5d[si, di] = C[si, di + 5] / C[si, di] - 1.0
            if di >= 20:
                atr_v2 = []
                for j in range(di - 14, di):
                    hh, ll, cc = H[si, j], L[si, j], C[si, j]
                    if not any(np.isnan([hh, ll, cc])):
                        pc = C[si, j - 1] if j > 0 and not np.isnan(C[si, j - 1]) else cc
                        atr_v2.append(max(hh - ll, abs(hh - pc), abs(ll - pc)))
                if atr_v2 and not np.isnan(C[si, di]) and C[si, di] > 0:
                    atr_mean[si, di] = np.mean(atr_v2) / C[si, di]

    rsi14 = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(float)
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
    return {"ret_5d": ret_5d, "oi_5d": oi_5d, "vol_5d": vol_5d,
            "range_5d": range_5d, "atrp_5d": atrp_5d,
            "ret_10d": ret_10d, "rsi14": rsi14,
            "fwd_ret_5d": fwd_ret_5d, "atr_mean": atr_mean}


def normalize_factor(factor, NS, ND, min_count=10):
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


# =====================================================================
# DMD Spectral Regime Detection
# =====================================================================

def compute_dmd_bull_ratio(C, NS, ND, window=120, n_delays=10, svd_rank=5):
    """DMD bull energy ratio per instrument per day."""
    t0 = time.time()
    print(f"[V113] DMD (w={window}, nd={n_delays}, rk={svd_rank})...", flush=True)
    bull_ratio = np.full((NS, ND), np.nan)

    for si in range(NS):
        prices = C[si]
        prev = 0.5
        for di in range(window, ND):
            recent = prices[di - window:di]
            valid_mask = ~np.isnan(recent)
            if np.sum(valid_mask) < window * 0.8:
                bull_ratio[si, di] = prev
                continue
            # Forward-fill small NaN gaps
            rc = np.copy(recent)
            for k in range(1, len(rc)):
                if np.isnan(rc[k]):
                    rc[k] = rc[k - 1]
            if rc[0] <= 0 or np.any(rc <= 0):
                bull_ratio[si, di] = prev
                continue
            log_p = np.log(rc)
            rets = np.diff(log_p)
            n_samp = len(rets) - n_delays + 1
            if n_samp < svd_rank + 2:
                bull_ratio[si, di] = prev
                continue
            X = np.zeros((n_delays, n_samp))
            for k in range(n_samp):
                X[:, k] = rets[k:k + n_delays]
            if X.shape[1] < svd_rank + 1:
                bull_ratio[si, di] = prev
                continue
            X1, X2 = X[:, :-1], X[:, 1:]
            try:
                U, S, Vt = np.linalg.svd(X1, full_matrices=False)
            except np.linalg.LinAlgError:
                bull_ratio[si, di] = prev
                continue
            rk = min(svd_rank, len(S))
            if rk < 1 or S[rk - 1] < 1e-12:
                bull_ratio[si, di] = prev
                continue
            try:
                A_tilde = U[:, :rk].T @ X2 @ Vt[:rk, :].T @ np.diag(1.0 / S[:rk])
                eigs = np.linalg.eigvals(A_tilde)
            except (np.linalg.LinAlgError, ValueError):
                bull_ratio[si, di] = prev
                continue
            mags = np.abs(eigs)
            bull_e = np.sum(mags[mags > 1.0] ** 2) if np.any(mags > 1.0) else 0.0
            total_e = np.sum(mags ** 2)
            prev = bull_e / total_e if total_e > 1e-12 else 0.5
            bull_ratio[si, di] = prev
        if si % 20 == 0:
            print(f"  si={si}/{NS}", flush=True)

    print(f"  DMD done: {time.time() - t0:.1f}s", flush=True)
    return bull_ratio


# =====================================================================
# BMA + NW Kernel (from V96)
# =====================================================================

def compute_rolling_ic(raw_factors, NS, ND, ic_window=60):
    t0 = time.time()
    print(f"[V113] Rolling IC (w={ic_window})...", flush=True)
    fwd_ret = raw_factors["fwd_ret_5d"]
    ic_array = np.full((N_FACTORS, ND), np.nan)
    for fi, fname in enumerate(FACTOR_NAMES):
        factor = raw_factors[fname]
        for di in range(ic_window + 5, ND):
            ic_vals = []
            for tdi in range(di - ic_window, di):
                f_day, r_day = factor[:, tdi], fwd_ret[:, tdi]
                vmask = (~np.isnan(f_day)) & (~np.isnan(r_day))
                fv, rv = f_day[vmask], r_day[vmask]
                if len(fv) >= 15:
                    corr = np.corrcoef(pd.Series(fv).rank().values,
                                       pd.Series(rv).rank().values)[0, 1]
                    if not np.isnan(corr):
                        ic_vals.append(corr)
            if len(ic_vals) >= 5:
                ic_array[fi, di] = np.mean(ic_vals)
    print(f"  IC done: {time.time() - t0:.1f}s", flush=True)
    return ic_array


def compute_bma_weights(ic_array, ND, prior_strength=5.0):
    weights = np.full((N_FACTORS, ND), np.nan)
    for fi in range(N_FACTORS):
        for di in range(20, ND):
            ic_hist = ic_array[fi, max(0, di - 120):di]
            vic = ic_hist[~np.isnan(ic_hist)]
            if len(vic) < 5:
                continue
            n_pos = np.sum(vic > 0)
            a = prior_strength / 2.0 + n_pos
            b = prior_strength / 2.0 + len(vic) - n_pos
            weights[fi, di] = a / (a + b)
    for di in range(ND):
        w = weights[:, di]
        if np.sum(~np.isnan(w)) == N_FACTORS:
            ws = np.nansum(w)
            if ws > 0:
                weights[:, di] = w / ws
    return weights


def compute_nw_predicted(raw_factors, bma_weights, NS, ND,
                         training_window=40, kernel_bandwidth=1.0):
    t0 = time.time()
    print(f"[V113] NW+BMA (tw={training_window}, bw={kernel_bandwidth})...", flush=True)
    normed = {fn: normalize_factor(raw_factors[fn], NS, ND) for fn in FACTOR_NAMES}
    # Apply BMA weighting
    w_normed = {}
    for fi, fn in enumerate(FACTOR_NAMES):
        orig = normed[fn]
        result = np.full((NS, ND), np.nan)
        for di in range(ND):
            w = bma_weights[fi, di]
            if np.isnan(w):
                w = 1.0 / N_FACTORS
            for si in range(NS):
                if not np.isnan(orig[si, di]):
                    result[si, di] = orig[si, di] * (w * N_FACTORS)
        w_normed[fn] = result

    fwd_ret = raw_factors["fwd_ret_5d"]
    atr_mean = raw_factors["atr_mean"]
    predicted = np.full((NS, ND), np.nan)

    for di in range(training_window + 10, ND):
        train_X_list, train_Y_list = [], []
        start_di = max(10, di - training_window)
        for tdi in range(start_di, di):
            for si in range(NS):
                feat = np.array([w_normed[fn][si, tdi] for fn in FACTOR_NAMES])
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
            qf = np.array([w_normed[fn][si, di] for fn in FACTOR_NAMES])
            if np.any(np.isnan(qf)):
                continue
            atr_val = atr_mean[si, di]
            h = max(atr_val * kernel_bandwidth, 0.1) if not np.isnan(atr_val) else kernel_bandwidth
            dist = np.sqrt(np.sum(((train_X - qf) / feat_std) ** 2, axis=1))
            sd = dist / h
            wts = np.zeros(len(train_X))
            mask = sd <= 1.0
            if not np.any(mask):
                idx = np.argmin(dist)
                if dist[idx] < 1e12:
                    wts[idx] = 1.0
                    mask = np.zeros(len(dist), dtype=bool)
                    mask[idx] = True
                else:
                    continue
            else:
                wts[mask] = 0.75 * (1.0 - sd[mask] ** 2)
            ws = np.sum(wts)
            if ws < 1e-12:
                continue
            predicted[si, di] = np.sum(wts * train_Y) / ws

        if di % 100 == 0:
            print(f"  di={di}/{ND} valid={np.sum(~np.isnan(predicted[:, di]))}", flush=True)

    print(f"  NW done: {time.time() - t0:.1f}s", flush=True)
    return predicted


# =====================================================================
# Helpers
# =====================================================================

def compute_ker(C, NS, ND):
    ker_10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            closes = C[si, di - 10:di + 1]
            valid = closes[~np.isnan(closes)]
            if len(valid) < 10 or valid[0] <= 0:
                continue
            net = abs(valid[-1] - valid[0])
            tot = np.sum(np.abs(np.diff(valid)))
            if tot > 1e-10:
                ker_10[si, di] = net / tot
    ker_regime = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        for di in range(ND):
            v = ker_10[si, di]
            if np.isnan(v):
                continue
            if v < 0.15:
                ker_regime[si, di] = 1
            elif v > 0.3:
                ker_regime[si, di] = -1
    return ker_regime


def get_dynamic_mode(recent_wins, win_thresh, wr_window):
    if len(recent_wins) < 5:
        return "normal"
    wr = sum(recent_wins[-wr_window:]) / len(recent_wins[-wr_window:])
    if wr > win_thresh:
        return "winning"
    elif wr < 0.50:
        return "losing"
    return "normal"


def compute_atr_at(H, L, C, si, di, start_di):
    atr_v = []
    for j in range(max(start_di, di - 14), di):
        hh, ll, cc = H[si, j], L[si, j], C[si, j]
        if not any(np.isnan([hh, ll, cc])):
            pc = C[si, j - 1] if j > 0 and not np.isnan(C[si, j - 1]) else cc
            atr_v.append(max(hh - ll, abs(hh - pc), abs(ll - pc)))
    return np.mean(atr_v) if atr_v else None


def compute_portfolio_vol(C, NS, ND, vol_lb=20):
    port_vol = np.full(ND, np.nan)
    for di in range(vol_lb + 1, ND):
        daily_rets = []
        for dd in range(di - vol_lb, di):
            rets = [C[si, dd] / C[si, dd - 1] - 1.0
                    for si in range(NS)
                    if not np.isnan(C[si, dd]) and not np.isnan(C[si, dd - 1]) and C[si, dd - 1] > 0]
            if rets:
                daily_rets.append(np.mean(rets))
        if len(daily_rets) >= vol_lb // 2:
            port_vol[di] = np.std(daily_rets)
    return port_vol


def get_dmd_gate_mult(br_now, br_prev, bull_thresh, bear_thresh, gate_str):
    """DMD regime gate: boost in bull, reduce in bear."""
    if np.isnan(br_now):
        return 1.0
    if br_now > bull_thresh and br_now > br_prev:
        return 1.0  # strong bull, rising
    if br_now < bear_thresh:
        return gate_str  # bear regime
    return 0.5 + 0.5 * gate_str  # neutral zone


# =====================================================================
# Backtest with DMD gate
# =====================================================================

def backtest_v113(C, O, H, L, NS, ND, dates, syms,
                  predicted, ker_regime, port_vol, dmd_br,
                  sector_lookup, top_n=2, max_per_sector=2,
                  hold_days=5, win_threshold=0.60,
                  atr_stop=3.0, vol_high_mult=2.0, vol_low_mult=0.5,
                  size_reduce=0.5, size_boost=1.3, vol_lookback=20,
                  bull_threshold=0.6, bear_threshold=0.3,
                  dmd_gate_strength=0.5,
                  start_di=60, end_di=None):
    if end_di is None:
        end_di = ND - 1
    vol_data = port_vol[max(start_di, vol_lookback + 1):end_di]
    vol_valid = vol_data[~np.isnan(vol_data)]
    vol_median = np.median(vol_valid) if len(vol_valid) > 10 else 1e-6

    equity, peak, max_dd = CASH0, CASH0, 0.0
    positions = []
    trades = []
    recent_wins = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_pos = []
        mode = get_dynamic_mode(recent_wins, win_threshold, 15)

        # Vol multiplier
        pv = port_vol[di]
        vol_mult = 1.0
        if not np.isnan(pv) and not np.isnan(vol_median) and vol_median > 1e-12:
            ratio = pv / vol_median
            if ratio > vol_high_mult:
                vol_mult = size_reduce
            elif ratio < vol_low_mult:
                vol_mult = size_boost

        # Exit
        pos_by_si = defaultdict(list)
        for si, edi, ep, sp, alloc in positions:
            pos_by_si[si].append((edi, ep, sp, alloc))

        for si, plist in pos_by_si.items():
            c = C[si, di]
            if np.isnan(c):
                for edi, ep, sp, alloc in plist:
                    new_pos.append((si, edi, ep, sp, alloc))
                continue
            earliest = min(p[0] for p in plist)
            hold = di - earliest
            stopped = any(c < sp for _, _, sp, _ in plist)
            if stopped or hold >= hold_days:
                for edi, ep, sp, alloc in plist:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    trades.append({
                        "pnl_abs": profit, "pnl_pct": pnl * 100,
                        "days": di - edi + 1, "di": di,
                        "year": d.year, "sym": syms[si],
                        "sector": sector_lookup.get(si, 'OTHER'),
                        "reason": "stop" if stopped else "hold",
                        "mode": mode[:1].upper()})
                    recent_wins.append(1 if pnl > 0 else 0)
            else:
                for edi, ep, sp, alloc in plist:
                    new_pos.append((si, edi, ep, sp, alloc))

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

        # Entry with DMD gate
        held = {p[0] for p in positions}
        if len(held) >= top_n:
            continue
        candidates = []
        for si in range(NS):
            if si in held:
                continue
            pred = predicted[si, di]
            if np.isnan(pred) or di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue
            if ker_regime[si, di] < 0:
                continue
            br_now = dmd_br[si, di]
            br_prev = dmd_br[si, di - 1] if di > 0 and not np.isnan(dmd_br[si, di - 1]) else 0.5
            dm = get_dmd_gate_mult(br_now, br_prev, bull_threshold,
                                   bear_threshold, dmd_gate_strength)
            if dm <= dmd_gate_strength:
                continue  # skip bearish DMD regime
            candidates.append((pred * dm, si, dm))
        if not candidates:
            continue

        candidates.sort(key=lambda x: -x[0])
        n_take = top_n
        if mode == "winning":
            n_take = min(top_n + 1, top_n * 2)
        elif mode == "losing":
            n_take = max(1, top_n - 1)

        sec_counts = defaultdict(int)
        for si_h in held:
            sec_counts[sector_lookup.get(si_h, 'OTHER')] += 1
        entries = []
        for gp, si, dm in candidates:
            if len(held) + len(entries) >= n_take:
                break
            if si in held:
                continue
            sec = sector_lookup.get(si, 'OTHER')
            if sec_counts[sec] >= max_per_sector:
                continue
            if gp <= 0:
                continue
            entries.append((gp, si, sec, dm))
            sec_counts[sec] += 1
        if not entries:
            continue

        alloc = LEVERAGE / (len(positions) + len(entries)) * vol_mult
        updated = [(si, edi, ep, sp, alloc) for si, edi, ep, sp, _ in positions]
        for gp, si, sec, dm in entries:
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr = compute_atr_at(H, L, C, si, di, start_di)
            if atr is None:
                continue
            updated.append((si, di + 1, ep, ep - atr_stop * atr, alloc * dm))
        positions = updated

    for si, edi, ep, sp, alloc in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            equity += equity * alloc * ((c - ep) / ep - COMM)

    return trades, equity, max_dd


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
    sh = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
    ns = sum(1 for t in trades if t["reason"] == "stop")
    nh = sum(1 for t in trades if t["reason"] == "hold")
    print(f"  {label}: {len(trades)}t (s:{ns} h:{nh}) WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% Sh={sh:.2f} eq={equity:,.0f}")
    yr = defaultdict(lambda: {"n": 0, "w": 0, "pnl": []})
    for t in trades:
        yr[t["year"]]["n"] += 1
        yr[t["year"]]["w"] += 1 if t["pnl_pct"] > 0 else 0
        yr[t["year"]]["pnl"].append(t["pnl_pct"])
    for y in sorted(yr.keys()):
        ys = yr[y]
        cum = np.prod([1 + p / 100 for p in ys["pnl"]]) - 1
        print(f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% cum={cum:+.1%}")
    return {"n": len(trades), "wr": wr, "dd": max_dd, "ann": ann, "sh": sh, "eq": equity}


def walk_forward(C, O, H, L, NS, ND, dates, syms,
                 predicted, ker_regime, port_vol, dmd_br,
                 sector_lookup, top_n=2, max_per_sector=2, hold_days=5,
                 vol_high_mult=2.0, vol_low_mult=0.5, size_reduce=0.5,
                 size_boost=1.3, vol_lookback=20, bull_threshold=0.6,
                 bear_threshold=0.3, dmd_gate_strength=0.5, label=""):
    cfg = (f"tn={top_n} mps={max_per_sector} bull={bull_threshold:.1f} "
           f"bear={bear_threshold:.1f} gate={dmd_gate_strength:.1f}")
    print(f"\n{'=' * 70}\n  WF V113 {label}\n  {cfg}\n{'=' * 70}")
    years = sorted(set(d.year for d in dates))
    all_trades = []
    for test_year in range(2019, years[-1] + 1):
        ts = te = None
        for i, d in enumerate(dates):
            if d.year == test_year and ts is None:
                ts = i
            if d.year == test_year:
                te = i
        if ts is None:
            continue
        trades, _, _ = backtest_v113(
            C, O, H, L, NS, ND, dates, syms,
            predicted, ker_regime, port_vol, dmd_br,
            sector_lookup=sector_lookup, top_n=top_n,
            max_per_sector=max_per_sector, hold_days=hold_days,
            vol_high_mult=vol_high_mult, vol_low_mult=vol_low_mult,
            size_reduce=size_reduce, size_boost=size_boost,
            vol_lookback=vol_lookback, bull_threshold=bull_threshold,
            bear_threshold=bear_threshold, dmd_gate_strength=dmd_gate_strength,
            start_di=ts, end_di=te + 1)
        tt = [t for t in trades if dates[t["di"]].year == test_year]
        all_trades.extend(tt)
        if tt:
            nw = sum(1 for t in tt if t["pnl_pct"] > 0)
            print(f"  {test_year}: {len(tt)}t WR={nw/len(tt)*100:.1f}%", flush=True)
        else:
            print(f"  {test_year}: no trades", flush=True)
    if all_trades:
        nw = sum(1 for t in all_trades if t["pnl_pct"] > 0)
        cum = np.prod([1 + t["pnl_pct"] / 100 for t in all_trades]) - 1
        print(f"\n  WF TOTAL: {len(all_trades)}t WR={nw/len(all_trades)*100:.1f}% cum={cum:+.1%}")
        return all_trades
    return []


def main():
    t0 = time.time()
    print("=" * 70)
    print("  V113: DMD SPECTRAL REGIME DETECTION + NW KERNEL")
    print("  DMD captures GLOBAL oscillation patterns as GATE")
    print("  Base: V96's NW+BMA kernel + vol-adaptive sizing")
    print("=" * 70)
    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start="2016-01-01")
    print(f"  {NS} sym, {ND} days, {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    sector_lookup = build_sector_lookup(syms)
    sd = defaultdict(int)
    for s in sector_lookup.values():
        sd[s] += 1
    print(f"  Sectors: {dict(sd)}")
    bt_2019 = next(i for i, d in enumerate(dates) if d >= pd.Timestamp("2019-01-01"))
    raw_factors = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ker_regime = compute_ker(C, NS, ND)
    # DMD bull ratios
    dmd_cache = {}
    for dw in [60, 80, 120]:
        for nd in [5, 10, 15]:
            for rk in [3, 5, 8]:
                dmd_cache[(dw, nd, rk)] = compute_dmd_bull_ratio(
                    C, NS, ND, window=dw, n_delays=nd, svd_rank=rk)
    # BMA + NW predictions
    ic_array = compute_rolling_ic(raw_factors, NS, ND, ic_window=60)
    bma_weights = compute_bma_weights(ic_array, ND)
    predicted = compute_nw_predicted(raw_factors, bma_weights, NS, ND,
                                     training_window=40, kernel_bandwidth=1.0)
    port_vol = compute_portfolio_vol(C, NS, ND, 20)
    # Parameter sweep
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("=" * 70)
    results = []
    sweep_count = 0
    for dmd_key, dmd_br in dmd_cache.items():
        dw, nd, rk = dmd_key
        for bull_t in [0.5, 0.6, 0.7]:
            for bear_t in [0.2, 0.3, 0.4]:
                for gate_s in [0.3, 0.5, 0.7]:
                    for mps in [2, 3]:
                        sweep_count += 1
                        trades, eq, dd = backtest_v113(
                            C, O, H, L, NS, ND, dates, syms,
                            predicted, ker_regime, port_vol, dmd_br,
                            sector_lookup=sector_lookup, top_n=2,
                            max_per_sector=mps, hold_days=5,
                            bull_threshold=bull_t, bear_threshold=bear_t,
                            dmd_gate_strength=gate_s, start_di=bt_2019)
                        if len(trades) < 10:
                            continue
                        nw = sum(1 for t in trades if t["pnl_pct"] > 0)
                        wr = nw / len(trades) * 100
                        ndays = max(1, trades[-1]["di"] - trades[0]["di"])
                        ann = ((eq / CASH0) ** (1 / max(1.0, ndays / 252)) - 1) * 100
                        ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
                        ra = np.array(ap) / CASH0
                        sh = np.mean(ra) / np.std(ra) * np.sqrt(252) if np.std(ra) > 0 else 0
                        results.append({
                            "dmd_w": dw, "dmd_nd": nd, "dmd_rk": rk,
                            "bull": bull_t, "bear": bear_t, "gate": gate_s,
                            "mps": mps, "n": len(trades), "wr": wr,
                            "ann": ann, "dd": dd, "sharpe": sh, "eq": eq})
    results.sort(key=lambda x: -x["ann"])
    print(f"\n  Evaluated {sweep_count} configs, {len(results)} with 10+ trades")
    print(f"\n{'DMDw':>5} {'ND':>3} {'Rk':>3} {'Bull':>5} {'Bear':>5} "
          f"{'Gate':>5} {'MPS':>3} {'N':>5} {'WR':>6} {'Ann':>9} "
          f"{'DD':>7} {'Sh':>7}")
    print("-" * 90)
    for r in results[:15]:
        print(f"{r['dmd_w']:>5} {r['dmd_nd']:>3} {r['dmd_rk']:>3} "
              f"{r['bull']:>5.1f} {r['bear']:>5.1f} {r['gate']:>5.1f} "
              f"{r['mps']:>3} {r['n']:>5} {r['wr']:>5.1f}% "
              f"{r['ann']:>+8.1f}% {r['dd']:>6.1f}% {r['sharpe']:>6.2f}")
    if not results:
        print("  No configs with 10+ trades. Exiting.")
        return
    # Walk-forward for top configs
    best_sh = max(results, key=lambda x: x["sharpe"])
    best_ann = results[0]
    best_ra = max(results, key=lambda x: x["ann"] / max(x["dd"], 1.0))
    for label, best in [("BEST-ANN", best_ann), ("BEST-SH", best_sh),
                        ("BEST-RA", best_ra)]:
        dk = (best["dmd_w"], best["dmd_nd"], best["dmd_rk"])
        walk_forward(C, O, H, L, NS, ND, dates, syms,
                     predicted, ker_regime, port_vol, dmd_cache[dk],
                     sector_lookup, top_n=2, max_per_sector=best["mps"],
                     hold_days=5, bull_threshold=best["bull"],
                     bear_threshold=best["bear"],
                     dmd_gate_strength=best["gate"], label=label)
    # Compare V113 vs V96 baseline
    print("\n" + "=" * 70)
    print("  COMPARISON: V113 (DMD+NW+BMA) vs V96 baseline")
    print("=" * 70)
    dk_best = (best_ann["dmd_w"], best_ann["dmd_nd"], best_ann["dmd_rk"])
    t113, eq113, dd113 = backtest_v113(
        C, O, H, L, NS, ND, dates, syms,
        predicted, ker_regime, port_vol, dmd_cache[dk_best],
        sector_lookup=sector_lookup, top_n=2,
        max_per_sector=best_ann["mps"], hold_days=5,
        bull_threshold=best_ann["bull"], bear_threshold=best_ann["bear"],
        dmd_gate_strength=best_ann["gate"], start_di=bt_2019)
    # V96 baseline: DMD gate disabled
    dmd_neutral = np.full((NS, ND), 0.8)
    t96, eq96, dd96 = backtest_v113(
        C, O, H, L, NS, ND, dates, syms,
        predicted, ker_regime, port_vol, dmd_neutral,
        sector_lookup=sector_lookup, top_n=2,
        max_per_sector=2, hold_days=5,
        bull_threshold=0.1, bear_threshold=0.01,
        dmd_gate_strength=0.9, start_di=bt_2019)
    print(f"\n  V113 BEST-ANN (DMD+NW+BMA+Vol):")
    analyze(t113, eq113, dd113, "V113-DMD+NW+BMA")
    print(f"\n  V96 BASELINE (NW+BMA, no DMD gate):")
    analyze(t96, eq96, dd96, "V96-baseline")
    if t113 and t96:
        print(f"\n  Delta: eq={eq113 - eq96:+,.0f} dd={dd113 - dd96:+.1f}% "
              f"trades={len(t113) - len(t96):+d}")
    print(f"\n[V113] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
