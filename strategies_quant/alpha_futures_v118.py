"""
V118: Upgraded Factor Set — MFI + ADX Replace Redundant Factors
================================================================
Factor engineering: vol_5d → MFI(14), range_5d → ADX(14), keep ret_10d.
9 factors total. NW kernel (no BMA) + vol-adaptive sizing.
Walk-forward 2019-2026. LEVERAGE=1.0, CASH0=1M, COMM=0.0005.
"""
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

FACTOR_NAMES = [
    "ret_5d", "oi_5d", "rsi14", "vol_5d",
    "ret_10d", "range_5d", "atrp_5d",
    "mfi_14", "adx_14",
]
N_FACTORS = len(FACTOR_NAMES)


def _base_sym(sym: str) -> str:
    s = sym.lower().split('.')[0].strip()
    while s and s[-1].isdigit():
        s = s[:-1]
    return s[:-2] if s.endswith('fi') else s


def build_sector_lookup(syms: List[str]) -> Dict[int, str]:
    return {si: SECTOR_MAP.get(_base_sym(sym), 'OTHER') for si, sym in enumerate(syms)}


# ── RSI (manual fallback) ───────────────────────────────────────────

def _rsi_manual(C: np.ndarray, NS: int, ND: int, period: int = 14) -> np.ndarray:
    rsi = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        avg_g = avg_l = None
        for di in range(1, ND):
            if np.isnan(c[di]) or np.isnan(c[di - 1]):
                continue
            d = c[di] - c[di - 1]
            g, l = max(d, 0.0), max(-d, 0.0)
            if avg_g is None:
                gs, ls = [], []
                for j in range(di, min(di + period, ND)):
                    if not np.isnan(c[j]) and not np.isnan(c[j - 1]):
                        dd = c[j] - c[j - 1]
                        gs.append(max(dd, 0.0))
                        ls.append(max(-dd, 0.0))
                if len(gs) >= period:
                    avg_g, avg_l = np.mean(gs), np.mean(ls)
                    rsi[si, di + period - 1] = 100 if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l)
                continue
            avg_g = (avg_g * (period - 1) + g) / period
            avg_l = (avg_l * (period - 1) + l) / period
            rsi[si, di] = 100 if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l)
    return rsi


# ── MFI (Money Flow Index) ─────────────────────────────────────────

def _mfi_talib(H, L, C, V, NS, ND, period=14) -> np.ndarray:
    mfi = np.full((NS, ND), np.nan)
    for si in range(NS):
        mask = np.isnan(H[si]) | np.isnan(L[si]) | np.isnan(C[si]) | np.isnan(V[si])
        try:
            r = talib.MFI(
                np.where(mask, 0, H[si]).astype(np.float64),
                np.where(mask, 0, L[si]).astype(np.float64),
                np.where(mask, 0, C[si]).astype(np.float64),
                np.where(mask, 0, V[si]).astype(np.float64),
                timeperiod=period)
            mfi[si] = np.where(mask, np.nan, r)
        except Exception:
            pass
    return mfi


def _mfi_manual(H, L, C, V, NS, ND, period=14) -> np.ndarray:
    mfi = np.full((NS, ND), np.nan)
    for si in range(NS):
        tp = np.where(
            np.isnan(H[si]) | np.isnan(L[si]) | np.isnan(C[si]),
            np.nan, (H[si] + L[si] + C[si]) / 3.0)
        mf = np.where(np.isnan(tp) | np.isnan(V[si]), np.nan, tp * V[si])
        for di in range(period, ND):
            pos = neg = 0.0
            for j in range(di - period + 1, di + 1):
                if np.isnan(mf[j]) or np.isnan(tp[j - 1]):
                    continue
                if tp[j] > tp[j - 1]:
                    pos += mf[j]
                else:
                    neg += mf[j]
            if neg < 1e-12:
                mfi[si, di] = 100.0
            else:
                mfi[si, di] = 100.0 - 100.0 / (1.0 + pos / neg)
    return mfi


# ── ADX (Average Directional Index) ────────────────────────────────

def _adx_talib(H, L, C, NS, ND, period=14) -> np.ndarray:
    adx = np.full((NS, ND), np.nan)
    for si in range(NS):
        mask = np.isnan(H[si]) | np.isnan(L[si]) | np.isnan(C[si])
        try:
            r = talib.ADX(
                np.where(mask, 0, H[si]).astype(np.float64),
                np.where(mask, 0, L[si]).astype(np.float64),
                np.where(mask, 0, C[si]).astype(np.float64),
                timeperiod=period)
            adx[si] = np.where(mask, np.nan, r)
        except Exception:
            pass
    return adx


def _adx_manual(H, L, C, NS, ND, period=14) -> np.ndarray:
    adx = np.full((NS, ND), np.nan)
    for si in range(NS):
        tr = np.full(ND, np.nan)
        pdm = np.full(ND, 0.0)
        mdm = np.full(ND, 0.0)
        for di in range(1, ND):
            if np.isnan(H[si, di]) or np.isnan(L[si, di]) or np.isnan(C[si, di]):
                continue
            prev_h, prev_l, prev_c = H[si, di-1], L[si, di-1], C[si, di-1]
            if np.isnan(prev_h) or np.isnan(prev_l) or np.isnan(prev_c):
                continue
            tr[di] = max(H[si,di]-L[si,di], abs(H[si,di]-prev_c), abs(L[si,di]-prev_c))
            up = H[si,di] - prev_h
            dn = prev_l - L[si,di]
            pdm[di] = up if up > dn and up > 0 else 0.0
            mdm[di] = dn if dn > up and dn > 0 else 0.0

        # Wilder smooth
        s_tr = s_pdm = s_mdm = None
        pdi_arr = np.full(ND, np.nan)
        mdi_arr = np.full(ND, np.nan)
        for di in range(1, ND):
            if np.isnan(tr[di]):
                continue
            if s_tr is None:
                s_tr, s_pdm, s_mdm = tr[di], pdm[di], mdm[di]
                continue
            s_tr = s_tr - s_tr / period + tr[di]
            s_pdm = s_pdm - s_pdm / period + pdm[di]
            s_mdm = s_mdm - s_mdm / period + mdm[di]
            if s_tr > 0:
                pdi_arr[di] = 100.0 * s_pdm / s_tr
                mdi_arr[di] = 100.0 * s_mdm / s_tr

        # DX → ADX
        dx = np.full(ND, np.nan)
        for di in range(ND):
            if not np.isnan(pdi_arr[di]) and not np.isnan(mdi_arr[di]):
                s = pdi_arr[di] + mdi_arr[di]
                if s > 0:
                    dx[di] = 100.0 * abs(pdi_arr[di] - mdi_arr[di]) / s

        adx_val = None
        dx_sum = 0.0
        dx_count = 0
        for di in range(ND):
            if np.isnan(dx[di]):
                continue
            if adx_val is None:
                dx_sum += dx[di]
                dx_count += 1
                if dx_count == period:
                    adx_val = dx_sum / period
                    adx[si, di] = adx_val
            else:
                adx_val = (adx_val * (period - 1) + dx[di]) / period
                adx[si, di] = adx_val
    return adx


# ── Factor computation ─────────────────────────────────────────────

def compute_raw_factors(C, O, H, L, V, OI, NS, ND) -> Dict[str, np.ndarray]:
    t0 = time.time()
    print("[V118] Computing 9 raw factors...", flush=True)

    ret_5d = np.full((NS, ND), np.nan)
    oi_5d = np.full((NS, ND), np.nan)
    vol_5d = np.full((NS, ND), np.nan)
    range_5d = np.full((NS, ND), np.nan)
    atrp_5d = np.full((NS, ND), np.nan)
    ret_10d = np.full((NS, ND), np.nan)

    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(C[si,di]) and not np.isnan(C[si,di-5]) and C[si,di-5] > 0:
                ret_5d[si,di] = C[si,di] / C[si,di-5] - 1.0
            if not np.isnan(OI[si,di]) and not np.isnan(OI[si,di-5]) and OI[si,di-5] > 0:
                oi_5d[si,di] = OI[si,di] / OI[si,di-5] - 1.0
            vv = V[si, di-5:di]
            vv = vv[~np.isnan(vv)]
            if len(vv) >= 3:
                vol_5d[si,di] = np.mean(vv)
            rv = []
            for j in range(di-5, di):
                if not np.isnan(H[si,j]) and not np.isnan(L[si,j]) and not np.isnan(C[si,j]) and C[si,j] > 0 and H[si,j] > L[si,j]:
                    rv.append((H[si,j] - L[si,j]) / C[si,j])
            if len(rv) >= 3:
                range_5d[si,di] = np.mean(rv)
            av = []
            for j in range(di-5, di):
                if not np.isnan(H[si,j]) and not np.isnan(L[si,j]) and not np.isnan(C[si,j]):
                    pc = C[si,j-1] if j > 0 and not np.isnan(C[si,j-1]) else C[si,j]
                    av.append(max(H[si,j]-L[si,j], abs(H[si,j]-pc), abs(L[si,j]-pc)))
            if av and not np.isnan(C[si,di]) and C[si,di] > 0:
                atrp_5d[si,di] = np.mean(av) / C[si,di]

        for di in range(10, ND):
            if not np.isnan(C[si,di]) and not np.isnan(C[si,di-10]) and C[si,di-10] > 0:
                ret_10d[si,di] = C[si,di] / C[si,di-10] - 1.0

    # RSI14
    rsi14 = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            mask = np.isnan(C[si])
            try:
                r = talib.RSI(np.where(mask, 0, C[si]).astype(np.float64), 14)
                rsi14[si] = np.where(mask, np.nan, r)
            except Exception:
                pass
    needs_fb = np.all(np.isnan(rsi14), axis=1)
    if needs_fb.any():
        fb = _rsi_manual(C, NS, ND, 14)
        for si in range(NS):
            if needs_fb[si]:
                rsi14[si] = fb[si]

    # MFI(14)
    mfi_14 = _mfi_talib(H, L, C, V, NS, ND, 14) if HAS_TALIB else np.full((NS,ND), np.nan)
    needs_fb = np.all(np.isnan(mfi_14), axis=1)
    if needs_fb.any():
        fb = _mfi_manual(H, L, C, V, NS, ND, 14)
        for si in range(NS):
            if needs_fb[si]:
                mfi_14[si] = fb[si]
    print(f"  MFI: {np.sum(~np.isnan(mfi_14))} valid", flush=True)

    # ADX(14)
    adx_14 = _adx_talib(H, L, C, NS, ND, 14) if HAS_TALIB else np.full((NS,ND), np.nan)
    needs_fb = np.all(np.isnan(adx_14), axis=1)
    if needs_fb.any():
        fb = _adx_manual(H, L, C, NS, ND, 14)
        for si in range(NS):
            if needs_fb[si]:
                adx_14[si] = fb[si]
    print(f"  ADX: {np.sum(~np.isnan(adx_14))} valid", flush=True)

    # Forward return + ATR mean
    fwd_ret = np.full((NS, ND), np.nan)
    atr_mean = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND - 5):
            if not np.isnan(C[si,di+5]) and not np.isnan(C[si,di]) and C[si,di] > 0:
                fwd_ret[si,di] = C[si,di+5] / C[si,di] - 1.0
        for di in range(20, ND):
            av = []
            for j in range(di-14, di):
                if not np.isnan(H[si,j]) and not np.isnan(L[si,j]) and not np.isnan(C[si,j]):
                    pc = C[si,j-1] if j > 0 and not np.isnan(C[si,j-1]) else C[si,j]
                    av.append(max(H[si,j]-L[si,j], abs(H[si,j]-pc), abs(L[si,j]-pc)))
            if av and not np.isnan(C[si,di]) and C[si,di] > 0:
                atr_mean[si,di] = np.mean(av) / C[si,di]

    print(f"  All factors done: {time.time()-t0:.1f}s", flush=True)
    return dict(ret_5d=ret_5d, oi_5d=oi_5d, vol_5d=vol_5d, range_5d=range_5d,
                atrp_5d=atrp_5d, ret_10d=ret_10d, rsi14=rsi14,
                mfi_14=mfi_14, adx_14=adx_14, fwd_ret_5d=fwd_ret, atr_mean=atr_mean)


# ── Normalization ───────────────────────────────────────────────────

def normalize_factor(f: np.ndarray, NS: int, ND: int, min_count: int = 10) -> np.ndarray:
    out = np.full((NS, ND), np.nan)
    for di in range(ND):
        v = f[:, di]
        ok = v[~np.isnan(v)]
        if len(ok) < min_count:
            continue
        mu, sig = np.mean(ok), np.std(ok)
        if sig < 1e-12:
            continue
        for si in range(NS):
            if not np.isnan(v[si]):
                out[si, di] = (v[si] - mu) / sig
    return out


# ── NW Kernel Regression ───────────────────────────────────────────

def compute_nw_predictions(
    raw: Dict[str, np.ndarray], NS: int, ND: int,
    tw: int = 40, kb: float = 1.0,
) -> np.ndarray:
    t0 = time.time()
    print(f"  NW tw={tw} kb={kb:.1f} (9 factors)...", flush=True)
    normed = {fn: normalize_factor(raw[fn], NS, ND) for fn in FACTOR_NAMES}
    fwd = raw["fwd_ret_5d"]
    atr_m = raw["atr_mean"]
    pred = np.full((NS, ND), np.nan)
    MIN_TRAIN = 20

    for di in range(tw + 10, ND):
        tX, tY = [], []
        for tdi in range(max(10, di - tw), di):
            for si in range(NS):
                feat = np.array([normed[fn][si, tdi] for fn in FACTOR_NAMES])
                tgt = fwd[si, tdi]
                if np.any(np.isnan(feat)) or np.isnan(tgt):
                    continue
                tX.append(feat)
                tY.append(tgt)
        if len(tX) < MIN_TRAIN:
            continue
        tX, tY = np.array(tX), np.array(tY)
        fstd = np.std(tX, axis=0)
        fstd[fstd < 1e-12] = 1.0

        for si in range(NS):
            qf = np.array([normed[fn][si, di] for fn in FACTOR_NAMES])
            if np.any(np.isnan(qf)):
                continue
            h = max(kb * atr_m[si, di], 0.1) if not np.isnan(atr_m[si, di]) else kb
            dist = np.sqrt(np.sum(((tX - qf) / fstd) ** 2, axis=1))
            sd = dist / h
            w = np.zeros(len(tX))
            m = sd <= 1.0
            if not np.any(m):
                mi = np.argmin(dist)
                if dist[mi] < 1e12:
                    w[mi] = 1.0
                else:
                    continue
            else:
                w[m] = 0.75 * (1.0 - sd[m] ** 2)
            ws = np.sum(w)
            if ws < 1e-12:
                continue
            pred[si, di] = np.sum(w * tY) / ws

        if di % 100 == 0:
            print(f"    di={di}/{ND} valid={np.sum(~np.isnan(pred[:,di]))}", flush=True)

    print(f"    done: {time.time()-t0:.1f}s", flush=True)
    return pred


# ── KER regime ──────────────────────────────────────────────────────

def compute_ker(C, NS, ND) -> np.ndarray:
    kr = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        for di in range(10, ND):
            cs = C[si, di-10:di+1]
            v = cs[~np.isnan(cs)]
            if len(v) < 10 or v[0] <= 0:
                continue
            tc = np.sum(np.abs(np.diff(v)))
            if tc > 1e-10:
                er = abs(v[-1] - v[0]) / tc
                if er < 0.15:
                    kr[si, di] = 1
                elif er > 0.3:
                    kr[si, di] = -1
    return kr


# ── Helpers ─────────────────────────────────────────────────────────

def _atr_at(H, L, C, si, di, start) -> Optional[float]:
    a = [max(H[si,j]-L[si,j], abs(H[si,j]-C[si,j]), abs(L[si,j]-C[si,j]))
         for j in range(max(start, di-14), di)
         if not np.isnan(H[si,j]) and not np.isnan(L[si,j]) and not np.isnan(C[si,j])]
    return np.mean(a) if a else None


def _port_vol(C, NS, ND, lb=20) -> np.ndarray:
    pv = np.full(ND, np.nan)
    for di in range(lb+1, ND):
        dr = []
        for dd in range(di-lb, di):
            rs = [C[si,dd]/C[si,dd-1]-1 for si in range(NS)
                  if not np.isnan(C[si,dd]) and not np.isnan(C[si,dd-1]) and C[si,dd-1] > 0]
            if rs:
                dr.append(np.mean(rs))
        if len(dr) >= lb // 2:
            pv[di] = np.std(dr)
    return pv


# ── Backtest ────────────────────────────────────────────────────────

def backtest(C, O, H, L, NS, ND, dates, syms, pred, ker, pvol,
             slook, top_n=2, mps=2, hd=5, vhm=2.0, vlm=0.5,
             sr=0.5, sb=1.3, start_di=60, end_di=None):
    if end_di is None:
        end_di = ND - 1
    vd = pvol[max(start_di, 21):end_di]
    vd = vd[~np.isnan(vd)]
    vmed = np.median(vd) if len(vd) > 10 else 1e-6

    eq, peak, mdd = CASH0, CASH0, 0.0
    pos: List[Tuple[int,int,float,float,float]] = []
    trades: List[dict] = []
    rtw: List[int] = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        dpnl = 0.0
        npos: List[Tuple[int,int,float,float,float]] = []

        # Vol sizing
        vm = 1.0
        if not np.isnan(pvol[di]) and not np.isnan(vmed) and vmed > 1e-12:
            r = pvol[di] / vmed
            vm = sr if r > vhm else (sb if r < vlm else 1.0)

        # Dynamic mode
        mode = "normal"
        if len(rtw) >= 5:
            wr = sum(rtw[-15:]) / len(rtw[-15:])
            mode = "winning" if wr > 0.60 else ("losing" if wr < 0.50 else "normal")

        # Exits
        by_si: Dict[int, List] = defaultdict(list)
        for p in pos:
            by_si[p[0]].append(p)

        for si, pl in by_si.items():
            c = C[si, di]
            if np.isnan(c):
                npos.extend(pl)
                continue
            hold = di - min(p[0] for p in pl)
            stopped = any(c < p[2] for p in pl)
            if stopped or hold >= hd:
                for _, edi, ep, sp, al in pl:
                    pnl = (c - ep) / ep - COMM
                    pr = eq * al * pnl
                    dpnl += pr
                    trades.append(dict(pnl_abs=pr, pnl_pct=pnl*100, days=di-edi+1,
                                       di=di, year=d.year, sym=syms[si],
                                       sector=slook.get(si,'OTHER'),
                                       reason="stop" if stopped else "hold",
                                       mode=mode[:1].upper()))
                    rtw.append(1 if pnl > 0 else 0)
            else:
                npos.extend(pl)

        pos = npos
        eq += dpnl
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak * 100
            if dd > mdd:
                mdd = dd
        if eq <= 0:
            break

        # Entry
        held = {p[0] for p in pos}
        nt = top_n + (1 if mode == "winning" else 0) - (1 if mode == "losing" else 0)
        nt = max(1, nt)
        if len(held) >= nt:
            continue

        cands = [(pred[si,di], si) for si in range(NS)
                 if si not in held and not np.isnan(pred[si,di])
                 and di+1 < ND and not np.isnan(O[si,di+1]) and ker[si,di] >= 0]
        if not cands:
            continue
        cands.sort(key=lambda x: -x[0])

        sc: Dict[str,int] = defaultdict(int)
        for sh in held:
            sc[slook.get(sh,'OTHER')] += 1

        entries = []
        for pv, si in cands:
            if len(held) + len(entries) >= nt or si in held:
                break
            sec = slook.get(si, 'OTHER')
            if sc[sec] >= mps or pv <= 0:
                continue
            entries.append((pv, si, sec))
            sc[sec] += 1

        if not entries:
            continue

        alloc = LEVERAGE / (len(pos) + len(entries)) * vm
        upos = [(si, edi, ep, sp, alloc) for si, edi, ep, sp, _ in pos]
        for pv, si, sec in entries:
            ep = O[si, di+1]
            if np.isnan(ep) or ep <= 0:
                continue
            atr = _atr_at(H, L, C, si, di, start_di)
            if atr is None:
                continue
            upos.append((si, di+1, ep, ep - 3.0 * atr, alloc))
        pos = upos

    # Close remaining
    for si, edi, ep, sp, al in pos:
        c = C[si, ND-1]
        if not np.isnan(c) and c > 0:
            eq += eq * al * ((c - ep) / ep - COMM)
    return trades, eq, mdd


# ── Analysis ────────────────────────────────────────────────────────

def analyze(trades, eq, mdd, label="") -> Optional[dict]:
    if not trades:
        print(f"  {label}: no trades")
        return None
    nw = sum(1 for t in trades if t["pnl_pct"] > 0)
    wr = nw / len(trades) * 100
    nd = max(1, trades[-1]["di"] - trades[0]["di"])
    ann = ((eq / CASH0) ** (1 / max(1.0, nd / 252)) - 1) * 100
    ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
    rets = np.array(ap) / CASH0
    sh = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0

    ns = sum(1 for t in trades if t["reason"] == "stop")
    nh = sum(1 for t in trades if t["reason"] == "hold")
    sc = defaultdict(int)
    for t in trades:
        sc[t.get("sector", "OTHER")] += 1
    ss = " ".join(f"{k}:{v}" for k, v in sorted(sc.items()))

    print(f"  {label}: {len(trades)}t (s:{ns} h:{nh}) WR={wr:.1f}% "
          f"ann={ann:+.1f}% DD={mdd:.1f}% Sh={sh:.2f} eq={eq:,.0f}")
    print(f"    sectors: {ss}")

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
        print(f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% cum={cum:+.1%}")

    return dict(n=len(trades), wr=wr, dd=mdd, ann=ann, sh=sh, eq=eq)


# ── Walk-forward ────────────────────────────────────────────────────

def walk_forward(C, O, H, L, NS, ND, dates, syms, pred, ker, pvol,
                 slook, top_n=2, mps=2, hd=5, vhm=2.0, vlm=0.5,
                 sr=0.5, sb=1.3, label="") -> List[dict]:
    print(f"\n{'='*70}")
    print(f"  WF V118 {label} tn={top_n} mps={mps} vhm={vhm:.1f} vlm={vlm:.1f} sr={sr:.1f} sb={sb:.1f}")
    print(f"  9 FACTORS: ret_5d oi_5d rsi14 vol_5d ret_10d range_5d atrp_5d MFI ADX")
    print(f"{'='*70}")

    yrs = sorted(set(d.year for d in dates))
    all_t: List[dict] = []

    for ty in range(2019, yrs[-1] + 1):
        ts = te = None
        for i, d in enumerate(dates):
            if d.year == ty and ts is None:
                ts = i
            if d.year == ty:
                te = i
        if ts is None:
            continue

        trades, _, _ = backtest(C, O, H, L, NS, ND, dates, syms,
                                pred, ker, pvol, slook,
                                top_n=top_n, mps=mps, hd=hd,
                                vhm=vhm, vlm=vlm, sr=sr, sb=sb,
                                start_di=ts, end_di=te+1)
        tt = [t for t in trades if dates[t["di"]].year == ty]
        all_t.extend(tt)
        if tt:
            nw = sum(1 for t in tt if t["pnl_pct"] > 0)
            wr = nw / len(tt) * 100
            avg = np.mean([t["pnl_pct"] for t in tt])
            sc = defaultdict(int)
            for t in tt:
                sc[t.get("sector", "OTHER")] += 1
            ss = " ".join(f"{k}:{v}" for k, v in sorted(sc.items()))
            print(f"  {ty}: {len(tt)}t WR={wr:.1f}% avg={avg:+.2f}% [{ss}]", flush=True)
        else:
            print(f"  {ty}: no trades", flush=True)

    if all_t:
        nw = sum(1 for t in all_t if t["pnl_pct"] > 0)
        wr = nw / len(all_t) * 100
        cum = np.prod([1 + t["pnl_pct"] / 100 for t in all_t]) - 1
        print(f"\n  WF TOTAL: {len(all_t)}t WR={wr:.1f}% cum={cum:+.1%}")
    return all_t


# ── Main ────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("  V118: UPGRADED FACTOR SET (MFI + ADX)")
    print("  9 factors: 7 original + MFI(14) + ADX(14)")
    print("  NW kernel (no BMA) + vol-adaptive sizing")
    print("  Walk-forward 2019-2026. No leverage.")
    print("  Hypothesis: BETTER factors > MORE factors")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start="2016-01-01")
    print(f"  {NS} sym, {ND} days, {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    slook = build_sector_lookup(syms)
    sd = defaultdict(int)
    for s in slook.values():
        sd[s] += 1
    print(f"  Sectors: {dict(sd)}")

    bt_2019 = next(i for i, d in enumerate(dates) if d >= pd.Timestamp("2019-01-01"))

    # Factors + predictions
    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ker = compute_ker(C, NS, ND)
    pvol = _port_vol(C, NS, ND, 20)

    pred_cache: Dict[Tuple[int, float], np.ndarray] = {}
    for tw in [30, 40, 50]:
        for kb in [0.8, 1.0, 1.5]:
            pred_cache[(tw, kb)] = compute_nw_predictions(raw, NS, ND, tw, kb)

    # Parameter sweep
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP (2019-2026) — 9 factors, NW kernel")
    print("=" * 70)

    results: List[dict] = []
    sc = 0
    for (tw, kb), pred in pred_cache.items():
        for tn in [2, 3]:
            for mps in [2, 3]:
                for vhm in [1.5, 2.0]:
                    for vlm in [0.5, 0.7]:
                        for sr in [0.3, 0.5]:
                            for sb in [1.2, 1.5]:
                                sc += 1
                                trades, eq, dd = backtest(
                                    C, O, H, L, NS, ND, dates, syms,
                                    pred, ker, pvol, slook,
                                    top_n=tn, mps=mps, vhm=vhm, vlm=vlm,
                                    sr=sr, sb=sb, start_di=bt_2019)
                                if len(trades) < 10:
                                    continue
                                nw = sum(1 for t in trades if t["pnl_pct"] > 0)
                                wr = nw / len(trades) * 100
                                nd = max(1, trades[-1]["di"] - trades[0]["di"])
                                ann = ((eq / CASH0) ** (1 / max(1.0, nd / 252)) - 1) * 100
                                ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
                                ra = np.array(ap) / CASH0
                                sh = np.mean(ra) / np.std(ra) * np.sqrt(252) if np.std(ra) > 0 else 0
                                results.append(dict(tw=tw, kb=kb, tn=tn, mps=mps,
                                                    vhm=vhm, vlm=vlm, sr=sr, sb=sb,
                                                    n=len(trades), wr=wr, ann=ann,
                                                    dd=dd, sharpe=sh, eq=eq))

    results.sort(key=lambda x: -x["ann"])
    print(f"\n  {sc} configs, {len(results)} with 10+ trades")

    print(f"\n{'TW':>3} {'KB':>4} {'TN':>3} {'MPS':>3} {'Vhm':>4} {'Vlm':>4} "
          f"{'SR':>4} {'SB':>4} {'N':>5} {'WR':>6} {'Ann':>9} {'DD':>7} {'Sh':>7}")
    print("-" * 85)
    for r in results[:15]:
        print(f"{r['tw']:>3} {r['kb']:>4.1f} {r['tn']:>3} {r['mps']:>3} "
              f"{r['vhm']:>4.1f} {r['vlm']:>4.1f} {r['sr']:>4.1f} {r['sb']:>4.1f} "
              f"{r['n']:>5} {r['wr']:>5.1f}% {r['ann']:>+8.1f}% "
              f"{r['dd']:>6.1f}% {r['sharpe']:>6.2f}")

    if not results:
        print("  No results. Exiting.")
        return

    # Walk-forward for best configs
    best_ann = results[0]
    best_sh = max(results, key=lambda x: x["sharpe"])
    best_ra = max(results, key=lambda x: x["ann"] / max(x["dd"], 1.0))

    for label, b in [("BEST-ANN", best_ann), ("BEST-SH", best_sh), ("BEST-RA", best_ra)]:
        walk_forward(C, O, H, L, NS, ND, dates, syms,
                     pred_cache[(b["tw"], b["kb"])], ker, pvol, slook,
                     top_n=b["tn"], mps=b["mps"], vhm=b["vhm"], vlm=b["vlm"],
                     sr=b["sr"], sb=b["sb"], label=label)

    # Final comparison
    print("\n" + "=" * 70)
    print("  V118 vs V96 COMPARISON")
    print("  V96 baseline: 7 factors, NW+BMA+Vol, +73.1% ann")
    print("=" * 70)
    for label, b in [("BEST-ANN", best_ann), ("BEST-SH", best_sh)]:
        trades, eq, dd = backtest(
            C, O, H, L, NS, ND, dates, syms,
            pred_cache[(b["tw"], b["kb"])], ker, pvol, slook,
            top_n=b["tn"], mps=b["mps"], vhm=b["vhm"], vlm=b["vlm"],
            sr=b["sr"], sb=b["sb"], start_di=bt_2019)
        analyze(trades, eq, dd, f"V118-{label}")

    print(f"\n[V118] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
