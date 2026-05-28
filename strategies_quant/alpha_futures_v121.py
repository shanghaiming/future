"""
V121: "无为" Market Timer — Trade when the Dao exists, hide when not
=====================================================================
"天下有道则见，无道则隐" — Confucius

Innovation: Composite "Dao" (Market Tradability) Indicator
  1. Factor IC quality: mean |IC| across factors (rolling 20d)
  2. Regime clarity: KER distance from random midpoint 0.225
  3. Liquidity: volume percentile rank
Position sizing via smooth sigmoid: dao_mult = 0.3 + 0.7 * sigmoid((dao-th)*k)

NW kernel regression (V86) for signal generation.
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

FACTOR_NAMES = ["ret_5d", "oi_5d", "rsi14", "vol_5d",
                "ret_10d", "range_5d", "atrp_5d"]
N_FACTORS = len(FACTOR_NAMES)


def _extract_base(sym: str) -> str:
    s = sym.lower().split('.')[0].strip()
    while s and s[-1].isdigit():
        s = s[:-1]
    return s[:-2] if s.endswith('fi') else s


def build_sector_lookup(syms: List[str]) -> Dict[int, str]:
    return {si: SECTOR_MAP.get(_extract_base(sym), 'OTHER')
            for si, sym in enumerate(syms)}


def compute_rsi_manual(C: np.ndarray, NS: int, ND: int,
                       period: int = 14) -> np.ndarray:
    rsi = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        gains = np.where(np.diff(c, prepend=np.nan) > 0,
                         np.diff(c, prepend=np.nan), 0.0)
        losses = np.where(np.diff(c, prepend=np.nan) < 0,
                         -np.diff(c, prepend=np.nan), 0.0)
        gains[0] = losses[0] = np.nan
        ag = al = np.nan
        for di in range(1, ND):
            if np.isnan(gains[di]):
                continue
            if np.isnan(ag):
                vg = gains[di:di + period]
                vl = losses[di:di + period]
                vg = vg[~np.isnan(vg)]
                vl = vl[~np.isnan(vl)]
                if len(vg) >= period:
                    ag, al = np.mean(vg), np.mean(np.where(vl, vl, 0))
                    rsi[si, min(di + period - 1, ND - 1)] = (
                        100 if al == 0 else 100 - 100 / (1 + ag / al))
                continue
            ag = (ag * (period - 1) + gains[di]) / period
            al = (al * (period - 1) + losses[di]) / period
            rsi[si, di] = 100 if al == 0 else 100 - 100 / (1 + ag / al)
    return rsi


def compute_raw_factors(C, O, H, L, V, OI, NS, ND):
    t0 = time.time()
    print("[V121] Computing raw factors...", flush=True)
    ret_5d = np.full((NS, ND), np.nan)
    oi_5d = np.full((NS, ND), np.nan)
    vol_5d = np.full((NS, ND), np.nan)
    range_5d = np.full((NS, ND), np.nan)
    atrp_5d = np.full((NS, ND), np.nan)
    ret_10d = np.full((NS, ND), np.nan)
    fwd_ret_5d = np.full((NS, ND), np.nan)
    atr_mean = np.full((NS, ND), np.nan)

    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 5]) and C[si, di - 5] > 0:
                ret_5d[si, di] = C[si, di] / C[si, di - 5] - 1.0
            if not np.isnan(OI[si, di]) and not np.isnan(OI[si, di - 5]) and OI[si, di - 5] > 0:
                oi_5d[si, di] = OI[si, di] / OI[si, di - 5] - 1.0
            vv = V[si, di - 5:di]
            vm = vv[~np.isnan(vv)]
            if len(vm) >= 3:
                vol_5d[si, di] = np.mean(vm)
            rv = [(H[si, j] - L[si, j]) / C[si, j]
                  for j in range(di - 5, di)
                  if not any(np.isnan([H[si, j], L[si, j], C[si, j]]))
                  and C[si, j] > 0 and H[si, j] > L[si, j]]
            if len(rv) >= 3:
                range_5d[si, di] = np.mean(rv)
        for di in range(6, ND):
            av = [max(H[si, j] - L[si, j],
                      abs(H[si, j] - (C[si, j - 1] if j > 0 and not np.isnan(C[si, j - 1]) else C[si, j])),
                      abs(L[si, j] - (C[si, j - 1] if j > 0 and not np.isnan(C[si, j - 1]) else C[si, j])))
                  for j in range(di - 5, di)
                  if not any(np.isnan([H[si, j], L[si, j], C[si, j]]))]
            if av and not np.isnan(C[si, di]) and C[si, di] > 0:
                atrp_5d[si, di] = np.mean(av) / C[si, di]
        for di in range(10, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di - 10]) and C[si, di - 10] > 0:
                ret_10d[si, di] = C[si, di] / C[si, di - 10] - 1.0
        for di in range(ND - 5):
            if not np.isnan(C[si, di + 5]) and not np.isnan(C[si, di]) and C[si, di] > 0:
                fwd_ret_5d[si, di] = C[si, di + 5] / C[si, di] - 1.0
        for di in range(20, ND):
            av = [max(H[si, j] - L[si, j],
                      abs(H[si, j] - (C[si, j - 1] if j > 0 and not np.isnan(C[si, j - 1]) else C[si, j])),
                      abs(L[si, j] - (C[si, j - 1] if j > 0 and not np.isnan(C[si, j - 1]) else C[si, j])))
                  for j in range(di - 14, di)
                  if not any(np.isnan([H[si, j], L[si, j], C[si, j]]))]
            if av and not np.isnan(C[si, di]) and C[si, di] > 0:
                atr_mean[si, di] = np.mean(av) / C[si, di]

    rsi14 = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nm = np.isnan(C[si])
            try:
                rsi14[si] = np.where(nm, np.nan, talib.RSI(c, 14))
            except Exception:
                pass
    needs_fb = np.all(np.isnan(rsi14), axis=1)
    if needs_fb.any():
        rm = compute_rsi_manual(C, NS, ND, 14)
        for si in range(NS):
            if needs_fb[si]:
                rsi14[si] = rm[si]

    print(f"  Raw factors done: {time.time() - t0:.1f}s", flush=True)
    return {"ret_5d": ret_5d, "oi_5d": oi_5d, "vol_5d": vol_5d,
            "range_5d": range_5d, "atrp_5d": atrp_5d,
            "ret_10d": ret_10d, "rsi14": rsi14,
            "fwd_ret_5d": fwd_ret_5d, "atr_mean": atr_mean}


def normalize_factor(f, NS, ND, mc=10):
    out = np.full((NS, ND), np.nan)
    for di in range(ND):
        v = f[:, di]
        ok = v[~np.isnan(v)]
        if len(ok) < mc:
            continue
        mu, sig = np.mean(ok), np.std(ok)
        if sig < 1e-12:
            continue
        for si in range(NS):
            if not np.isnan(v[si]):
                out[si, di] = (v[si] - mu) / sig
    return out


# === Dao Composite Indicator ===

def compute_dao_sub_indicators(raw, C, V, NS, ND, ic_window=20):
    """Compute 3 Dao sub-indicators once (expensive), return them for reuse."""
    t0 = time.time()
    print("[V121] Computing Dao sub-indicators (one-time)...", flush=True)
    fwd = raw["fwd_ret_5d"]

    # Sub-indicator 1: IC quality — per-day mean |IC| across factors
    ic_quality = np.full(ND, np.nan)
    for di in range(ic_window + 5, ND):
        fics = []
        for fn in FACTOR_NAMES:
            fac = raw[fn]
            dics = []
            for tdi in range(di - ic_window, di):
                fd, rd = fac[:, tdi], fwd[:, tdi]
                m = (~np.isnan(fd)) & (~np.isnan(rd))
                fv, rv = fd[m], rd[m]
                if len(fv) >= 10:
                    c = np.corrcoef(pd.Series(fv).rank().values,
                                    pd.Series(rv).rank().values)[0, 1]
                    if not np.isnan(c):
                        dics.append(abs(c))
            if dics:
                fics.append(np.mean(dics))
        if fics:
            ic_quality[di] = np.mean(fics)
    # Normalize to [0,1]
    ic_norm = np.full(ND, np.nan)
    for di in range(ic_window + 10, ND):
        h = ic_quality[ic_window + 5:di]
        hv = h[~np.isnan(h)]
        if len(hv) >= 10 and not np.isnan(ic_quality[di]):
            ic_norm[di] = np.mean(ic_quality[di] >= hv)
    print(f"  IC quality done: {time.time() - t0:.1f}s", flush=True)

    # Sub-indicator 2: Regime clarity from KER
    ker_10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            cs = C[si, di - 10:di + 1]
            vs = cs[~np.isnan(cs)]
            if len(vs) >= 10 and vs[0] > 0:
                tc = np.sum(np.abs(np.diff(vs)))
                if tc > 1e-10:
                    ker_10[si, di] = abs(vs[-1] - vs[0]) / tc
    regime_clarity = np.full(ND, np.nan)
    for di in range(ND):
        kv = ker_10[:, di]
        vk = kv[~np.isnan(kv)]
        if len(vk) >= 5:
            regime_clarity[di] = abs(np.mean(vk) - 0.225) / 0.225
    reg_norm = np.full(ND, np.nan)
    for di in range(20, ND):
        h = regime_clarity[20:di]
        hv = h[~np.isnan(h)]
        if len(hv) >= 10 and not np.isnan(regime_clarity[di]):
            reg_norm[di] = np.mean(regime_clarity[di] >= hv)
    print(f"  Regime clarity done: {time.time() - t0:.1f}s", flush=True)

    # Sub-indicator 3: Liquidity — volume percentile rank
    liquidity = np.full(ND, np.nan)
    for di in range(5, ND):
        vt = V[:, di]
        vm = vt[~np.isnan(vt)]
        if len(vm) < 10:
            continue
        med = np.median(vm)
        hm = [np.median(V[:, tdi][~np.isnan(V[:, tdi])])
              for tdi in range(max(5, di - 60), di)
              if np.sum(~np.isnan(V[:, tdi])) >= 5]
        if len(hm) >= 10:
            liquidity[di] = np.mean(med >= np.array(hm))
    print(f"  All sub-indicators done: {time.time() - t0:.1f}s", flush=True)
    return ic_norm, reg_norm, liquidity


def combine_dao(ic_norm, reg_norm, liquidity, ND,
                dao_ic_w=0.4, dao_reg_w=0.3, dao_liq_w=0.3):
    """Combine sub-indicators with weights into composite Dao. Fast."""
    dao = np.full(ND, 0.5)
    for di in range(ND):
        parts, wts = [], []
        if not np.isnan(ic_norm[di]):
            parts.append(ic_norm[di]); wts.append(dao_ic_w)
        if not np.isnan(reg_norm[di]):
            parts.append(reg_norm[di]); wts.append(dao_reg_w)
        if not np.isnan(liquidity[di]):
            parts.append(liquidity[di]); wts.append(dao_liq_w)
        if parts:
            ws = sum(wts)
            dao[di] = sum(p * w for p, w in zip(parts, wts)) / ws
    return dao


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x)) if x >= 0 else np.exp(x) / (1.0 + np.exp(x))


def dao_mult(dao_val, threshold, steepness=10.0):
    """Smooth sigmoid: 0.3 + 0.7 * sigmoid((dao - th) * k)"""
    return 0.3 + 0.7 * sigmoid((dao_val - threshold) * steepness)


# === NW Kernel (V86 core) ===

def compute_nw_returns(raw, NS, ND, tw=40, bw=1.0):
    t0 = time.time()
    print(f"[V121] NW kernel (tw={tw}, bw={bw:.1f})...", flush=True)
    normed = {fn: normalize_factor(raw[fn], NS, ND) for fn in FACTOR_NAMES}
    fwd = raw["fwd_ret_5d"]
    atr = raw["atr_mean"]
    pred = np.full((NS, ND), np.nan)
    MIN_T = 20

    for di in range(tw + 10, ND):
        tX, tY = [], []
        for tdi in range(max(10, di - tw), di):
            for si in range(NS):
                ft = np.array([normed[fn][si, tdi] for fn in FACTOR_NAMES])
                tg = fwd[si, tdi]
                if np.any(np.isnan(ft)) or np.isnan(tg):
                    continue
                tX.append(ft); tY.append(tg)
        if len(tX) < MIN_T:
            continue
        Xa, Ya = np.array(tX), np.array(tY)
        fs = np.std(Xa, axis=0)
        fs[fs < 1e-12] = 1.0

        for si in range(NS):
            qf = np.array([normed[fn][si, di] for fn in FACTOR_NAMES])
            if np.any(np.isnan(qf)):
                continue
            h = max(bw if np.isnan(atr[si, di]) else atr[si, di] * bw, 0.1)
            dist = np.sqrt(np.sum(((Xa - qf) / fs) ** 2, axis=1))
            sd = dist / h
            w = np.zeros(len(Xa))
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
            if ws > 1e-12:
                pred[si, di] = np.sum(w * Ya) / ws
        if di % 100 == 0:
            print(f"  di={di}/{ND} valid={np.sum(~np.isnan(pred[:, di]))}/{NS}",
                  flush=True)
    print(f"  NW done: {time.time() - t0:.1f}s", flush=True)
    return pred


def compute_ker(C, NS, ND):
    ker = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        for di in range(10, ND):
            cs = C[si, di - 10:di + 1]
            vs = cs[~np.isnan(cs)]
            if len(vs) < 10 or vs[0] <= 0:
                continue
            tc = np.sum(np.abs(np.diff(vs)))
            if tc > 1e-10:
                k = abs(vs[-1] - vs[0]) / tc
                if k < 0.15:
                    ker[si, di] = 1
                elif k > 0.3:
                    ker[si, di] = -1
    return ker


def get_mode(rtw, wt, wwr):
    if len(rtw) < 5:
        return "normal"
    wr = sum(rtw[-wwr:]) / len(rtw[-wwr:])
    return "winning" if wr > wt else ("losing" if wr < 0.50 else "normal")


def atr_at(H, L, C, si, di, sdi):
    av = [max(H[si, j] - L[si, j], abs(H[si, j] - C[si, j]),
              abs(L[si, j] - C[si, j]))
          for j in range(max(sdi, di - 14), di)
          if not any(np.isnan([H[si, j], L[si, j], C[si, j]]))]
    return np.mean(av) if av else None


# === Backtest with Dao sizing ===

def backtest(C, O, H, L, NS, ND, dates, syms, pred, ker, dao,
             slo, top_n=2, mps=2, hd=5, dao_th=0.5, dao_st=10.0,
             wt=0.60, wwr=15, atr_stop=3.0, sdi=60, edi=None):
    if edi is None:
        edi = ND - 1
    eq, pk, mdd = CASH0, CASH0, 0.0
    pos = []
    trades, rtw = [], []
    dlvl = {"h": 0, "m": 0, "l": 0}

    for di in range(max(sdi, 1), edi):
        d = dates[di]
        dpnl = 0.0
        mode = get_mode(rtw, wt, wwr)
        dv = dao[di] if di < len(dao) else 0.5
        dm = dao_mult(dv, dao_th, dao_st)
        dlvl["h" if dv > 0.7 else ("m" if dv > 0.5 else "l")] += 1

        pb = defaultdict(list)
        for s, e, ep, sp, a in pos:
            pb[s].append((e, ep, sp, a))

        npos = []
        for si, pl in pb.items():
            c = C[si, di]
            if np.isnan(c):
                for e, ep, sp, a in pl:
                    npos.append((si, e, ep, sp, a))
                continue
            ee = min(p[0] for p in pl)
            hd_cur = di - ee
            stop = any(c < p[2] for p in pl)
            if stop or hd_cur >= hd:
                reason = "stop" if stop else "hold"
                for e, ep, sp, a in pl:
                    pnl = (c - ep) / ep - COMM
                    pr = eq * a * pnl
                    dpnl += pr
                    trades.append({
                        "pnl_abs": pr, "pnl_pct": pnl * 100,
                        "days": di - e + 1, "di": di, "year": d.year,
                        "sym": syms[si], "sector": slo.get(si, 'O'),
                        "reason": reason, "mode": mode[0].upper(),
                        "dao": dv, "dm": dm})
                    rtw.append(1 if pnl > 0 else 0)
            else:
                for e, ep, sp, a in pl:
                    npos.append((si, e, ep, sp, a))
        pos = npos
        eq += dpnl
        if eq > pk:
            pk = eq
        if pk > 0:
            dd = (pk - eq) / pk * 100
            if dd > mdd:
                mdd = dd
        if eq <= 0:
            break

        held = {p[0] for p in pos}
        if len(held) >= top_n:
            continue
        cands = [(pred[si, di], si) for si in range(NS)
                 if si not in held and not np.isnan(pred[si, di])
                 and di + 1 < ND and not np.isnan(O[si, di + 1])
                 and ker[si, di] >= 0]
        if not cands:
            continue
        cands.sort(key=lambda x: -x[0])
        nt = top_n + (1 if mode == "winning" else 0)
        if mode == "losing":
            nt = max(1, top_n - 1)
        sc = defaultdict(int)
        for sh in held:
            sc[slo.get(sh, 'O')] += 1
        ne = []
        for pv, si in cands:
            if len(held) + len(ne) >= nt:
                break
            ss = slo.get(si, 'O')
            if sc[ss] >= mps or pv <= 0:
                continue
            ne.append((pv, si, ss))
            sc[ss] += 1
        if not ne:
            continue
        ap = LEVERAGE / (len(pos) + len(ne)) * dm
        upos = [(s, e, ep, sp, ap) for s, e, ep, sp, a in pos]
        for pv, si, ss in ne:
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0:
                continue
            a = atr_at(H, L, C, si, di, sdi)
            if a is None:
                continue
            upos.append((si, di + 1, ep, ep - atr_stop * a, ap))
        pos = upos

    for si, e, ep, sp, a in pos:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            eq += eq * a * ((c - ep) / ep - COMM)
    if trades:
        td = sum(dlvl.values())
        trades[0]["dao_info"] = (
            f"dao=[h:{dlvl['h']} m:{dlvl['m']} l:{dlvl['l']}]/{td}")
    return trades, eq, mdd


def analyze(trades, eq, mdd, label=""):
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
    ad = np.mean([t.get("dao", 0.5) for t in trades])
    adm = np.mean([t.get("dm", 1.0) for t in trades])
    ns = sum(1 for t in trades if t["reason"] == "stop")
    nh = sum(1 for t in trades if t["reason"] == "hold")
    sc = defaultdict(int)
    for t in trades:
        sc[t.get("sector", "O")] += 1
    ss = " ".join(f"{k}:{v}" for k, v in sorted(sc.items()))
    aw = np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"] > 0])
    al = np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"] <= 0])
    print(f"  {label}: {len(trades)}t (s:{ns} h:{nh}) "
          f"WR={wr:.1f}% ann={ann:+.1f}% DD={mdd:.1f}% "
          f"Sh={sh:.2f} eq={eq:,.0f}")
    print(f"    dao_avg={ad:.3f} dao_mult_avg={adm:.3f} "
          f"avg_win={aw:+.3f}% avg_loss={al:+.3f}%")
    print(f"    sectors: {ss}")
    yr = {}
    for t in trades:
        y = t["year"]
        if y not in yr:
            yr[y] = {"n": 0, "w": 0, "p": [], "d": []}
        yr[y]["n"] += 1
        if t["pnl_pct"] > 0:
            yr[y]["w"] += 1
        yr[y]["p"].append(t["pnl_pct"])
        yr[y]["d"].append(t.get("dao", 0.5))
    for y in sorted(yr):
        ys = yr[y]
        cum = np.prod([1 + p / 100 for p in ys["p"]]) - 1
        print(f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% "
              f"cum={cum:+.1%} dao={np.mean(ys['d']):.3f}")
    return {"n": len(trades), "wr": wr, "dd": mdd, "ann": ann,
            "sh": sh, "eq": eq}


def walk_forward(C, O, H, L, NS, ND, dates, syms, pred, ker, dao,
                 slo, top_n=2, mps=2, dao_th=0.5, dao_st=10.0, label=""):
    cfg = f"tn={top_n} mps={mps} dao_th={dao_th:.2f}"
    print(f"\n{'=' * 70}\n  WF V121 {label}\n  {cfg}\n{'=' * 70}")
    years = sorted(set(d.year for d in dates))
    all_t = []
    for ty in range(2019, years[-1] + 1):
        ts = te = None
        for i, d in enumerate(dates):
            if d.year == ty and ts is None:
                ts = i
            if d.year == ty:
                te = i
        if ts is None:
            continue
        tr, _, _ = backtest(C, O, H, L, NS, ND, dates, syms, pred, ker,
                            dao, slo, top_n=top_n, mps=mps, dao_th=dao_th,
                            dao_st=dao_st, sdi=ts, edi=te + 1)
        yt = [t for t in tr if dates[t["di"]].year == ty]
        all_t.extend(yt)
        if yt:
            n = len(yt)
            nw = sum(1 for t in yt if t["pnl_pct"] > 0)
            ad = np.mean([t.get("dao", 0.5) for t in yt])
            adm = np.mean([t.get("dm", 1.0) for t in yt])
            print(f"  {ty}: {n}t WR={nw/n*100:.1f}% "
                  f"dao={ad:.3f} dm={adm:.3f}", flush=True)
        else:
            print(f"  {ty}: no trades", flush=True)
    if all_t:
        nw = sum(1 for t in all_t if t["pnl_pct"] > 0)
        cum = np.prod([1 + t["pnl_pct"] / 100 for t in all_t]) - 1
        ad = np.mean([t.get("dao", 0.5) for t in all_t])
        print(f"\n  WF TOTAL: {len(all_t)}t WR={nw/len(all_t)*100:.1f}% "
              f"cum={cum:+.1%} dao={ad:.3f}")
        return all_t
    return []


def main():
    t0 = time.time()
    print("=" * 70)
    print('  V121: "无为" Market Timer')
    print('  "天下有道则见，无道则隐"')
    print("  Dao indicator: IC quality + regime clarity + liquidity")
    print("  NW kernel + Dao sigmoid sizing. WF 2019-2026. No leverage.")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start="2016-01-01")
    print(f"  {NS} sym, {ND} days, {dates[0].strftime('%Y-%m-%d')} "
          f"to {dates[-1].strftime('%Y-%m-%d')}")
    slo = build_sector_lookup(syms)
    sd = defaultdict(int)
    for s in slo.values():
        sd[s] += 1
    print(f"  Sectors: {dict(sd)}")

    bt19 = next(i for i, d in enumerate(dates) if d >= pd.Timestamp("2019-01-01"))

    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ker = compute_ker(C, NS, ND)
    pred = compute_nw_returns(raw, NS, ND, tw=40, bw=1.0)

    # Dao sub-indicators (expensive, computed once)
    print("\n--- Computing Dao sub-indicators ---")
    ic_norm, reg_norm, liquidity = compute_dao_sub_indicators(
        raw, C, V, NS, ND, ic_window=20)

    # Dao sweep configs — only vary combination weights (fast)
    dao_cfgs = [(iw, rw, round(1.0 - iw - rw, 1))
                for iw in [0.3, 0.4, 0.5]
                for rw in [0.2, 0.3, 0.4]
                if round(1.0 - iw - rw, 1) >= 0.1]
    dao_cache = {}
    for iw, rw, lw in dao_cfgs:
        dao_cache[(iw, rw, lw)] = combine_dao(
            ic_norm, reg_norm, liquidity, ND,
            dao_ic_w=iw, dao_reg_w=rw, dao_liq_w=lw)
    print(f"  {len(dao_cache)} Dao variants combined.", flush=True)

    # Parameter sweep
    print(f"\n{'=' * 70}\n  PARAMETER SWEEP (2019-2026)\n{'=' * 70}")
    results = []
    sc = 0
    for dk, da in dao_cache.items():
        iw, rw, lw = dk
        for tn in [2, 3]:
            for mps in [2, 3]:
                for dth in [0.4, 0.5, 0.6]:
                    sc += 1
                    tr, eq, dd = backtest(
                        C, O, H, L, NS, ND, dates, syms, pred, ker,
                        da, slo, top_n=tn, mps=mps, dao_th=dth, sdi=bt19)
                    if len(tr) < 10:
                        continue
                    nw = sum(1 for t in tr if t["pnl_pct"] > 0)
                    wr = nw / len(tr) * 100
                    nd = max(1, tr[-1]["di"] - tr[0]["di"])
                    ann = ((eq / CASH0) ** (1 / max(1.0, nd / 252)) - 1) * 100
                    ap = [t["pnl_abs"] for t in sorted(tr, key=lambda x: x["di"])]
                    ra = np.array(ap) / CASH0
                    sh = np.mean(ra) / np.std(ra) * np.sqrt(252) if np.std(ra) > 0 else 0
                    results.append({"iw": iw, "rw": rw, "lw": lw,
                                    "tn": tn, "mps": mps, "dth": dth,
                                    "n": len(tr), "wr": wr, "ann": ann,
                                    "dd": dd, "sh": sh, "eq": eq})

    # Baseline (no Dao)
    db = np.ones(ND)
    for tn in [2, 3]:
        for mps in [2, 3]:
            sc += 1
            tr, eq, dd = backtest(
                C, O, H, L, NS, ND, dates, syms, pred, ker,
                db, slo, top_n=tn, mps=mps, dao_th=0.0, sdi=bt19)
            if len(tr) < 10:
                continue
            nw = sum(1 for t in tr if t["pnl_pct"] > 0)
            wr = nw / len(tr) * 100
            nd = max(1, tr[-1]["di"] - tr[0]["di"])
            ann = ((eq / CASH0) ** (1 / max(1.0, nd / 252)) - 1) * 100
            ap = [t["pnl_abs"] for t in sorted(tr, key=lambda x: x["di"])]
            ra = np.array(ap) / CASH0
            sh = np.mean(ra) / np.std(ra) * np.sqrt(252) if np.std(ra) > 0 else 0
            results.append({"iw": 0, "rw": 0, "lw": 0,
                            "tn": tn, "mps": mps, "dth": 0.0,
                            "n": len(tr), "wr": wr, "ann": ann,
                            "dd": dd, "sh": sh, "eq": eq})

    results.sort(key=lambda x: -x["ann"])
    print(f"\n  {sc} configs, {len(results)} with 10+ trades")
    print(f"\n{'ICw':>4} {'Rw':>4} {'Lw':>4} {'TN':>3} {'MPS':>3} "
          f"{'DaoTh':>6} {'N':>5} {'WR':>6} {'Ann':>9} {'DD':>7} {'Sh':>7}")
    print("-" * 75)
    for r in results[:15]:
        tg = f"{r['iw']:.1f}" if r['iw'] > 0 else "base"
        rg = f"{r['rw']:.1f}" if r['rw'] > 0 else "---"
        lg = f"{r['lw']:.1f}" if r['lw'] > 0 else "---"
        dt = f"{r['dth']:.2f}" if r['dth'] > 0 else "OFF"
        print(f"{tg:>4} {rg:>4} {lg:>4} {r['tn']:>3} {r['mps']:>3} "
              f"{dt:>6} {r['n']:>5} {r['wr']:>5.1f}% "
              f"{r['ann']:>+8.1f}% {r['dd']:>6.1f}% {r['sh']:>6.2f}")

    if not results:
        print("  No results. Exiting.")
        return

    # Walk-forward top configs
    ba = results[0]
    bs = max(results, key=lambda x: x["sh"])
    br = max(results, key=lambda x: x["ann"] / max(x["dd"], 1.0))

    for lb, b in [("BEST-ANN", ba), ("BEST-SH", bs), ("BEST-RISK", br)]:
        da = np.ones(ND) if b["iw"] == 0 else dao_cache[(b["iw"], b["rw"], b["lw"])]
        walk_forward(C, O, H, L, NS, ND, dates, syms, pred, ker, da,
                     slo, top_n=b["tn"], mps=b["mps"],
                     dao_th=b["dth"], label=lb)

    # Compare V121 vs V96 baseline
    print(f"\n{'=' * 70}\n  COMPARISON: V121 vs V96 baseline\n{'=' * 70}")
    da_best = np.ones(ND) if ba["iw"] == 0 else dao_cache[(ba["iw"], ba["rw"], ba["lw"])]
    t121, e121, d121 = backtest(
        C, O, H, L, NS, ND, dates, syms, pred, ker, da_best, slo,
        top_n=ba["tn"], mps=ba["mps"], dao_th=ba["dth"], sdi=bt19)
    t96, e96, d96 = backtest(
        C, O, H, L, NS, ND, dates, syms, pred, ker, np.ones(ND), slo,
        top_n=2, mps=2, dao_th=0.0, sdi=bt19)

    print(f'\n  V121 BEST-ANN (NW + Dao "无为"):')
    analyze(t121, e121, d121, "V121-NW+Dao")
    print(f"\n  V96 BASELINE (NW only):")
    analyze(t96, e96, d96, "V96-base")
    if t121 and t96:
        print(f"\n  Delta: eq={e121 - e96:+,.0f} dd={d121 - d96:+.1f}% "
              f"trades={len(t121) - len(t96):+d}")
    print(f"\n[V121] Done. {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
