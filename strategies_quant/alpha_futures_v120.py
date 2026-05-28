"""
V120: "凡事豫则立" Multi-Period Readiness Index Strategy
=========================================================
Trade Quality = min(Research, Regime, Risk, Execution).

"凡事豫则立，不豫则废" — preparation brings success.
  P_r = fraction of factors with |IC_window| > 0.02
  P_g = 1.0 if KER regime supports signal
  P_k = 1.0 if portfolio heat < threshold
  P_e = min(1.0, volume_5d / median_volume)

Overall readiness = min(P_r, P_g, P_k, P_e).
Position size = base_alloc * readiness. Skip if readiness < threshold.

Walk-forward 2019-2026. LEVERAGE=1.0, CASH0=1M, COMM=0.0005.
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_futures_data import load_all_data

try:
    import talib; HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

CASH0 = 1_000_000; COMM = 0.0005; LEVERAGE = 1.0

SECTOR_MAP = {
    'i':'BLACK','j':'BLACK','jm':'BLACK','hc':'BLACK','sf':'BLACK',
    'sm':'BLACK','wr':'BLACK','im':'BLACK',
    'cu':'METAL','al':'METAL','zn':'METAL','pb':'METAL','ni':'METAL',
    'sn':'METAL','ss':'METAL','ao':'METAL','au':'METAL','ag':'METAL',
    'rb':'METAL','si':'METAL',
    'sc':'ENERGY','fu':'ENERGY','bu':'ENERGY','pg':'ENERGY',
    'eb':'ENERGY','ta':'ENERGY','fg':'ENERGY','oi':'ENERGY',
    'v':'CHEMICAL','pp':'CHEMICAL','l':'CHEMICAL','eg':'CHEMICAL',
    'ma':'CHEMICAL','sa':'CHEMICAL','ur':'CHEMICAL','pf':'CHEMICAL',
    'sh':'CHEMICAL','lc':'CHEMICAL',
    'm':'AGRI','y':'AGRI','a':'AGRI','p':'AGRI','c':'AGRI','cs':'AGRI',
    'jd':'AGRI','rr':'AGRI','lrm':'AGRI','rm':'AGRI','ru':'AGRI',
    'cf':'SOFTS','sr':'SOFTS','ap':'SOFTS','cj':'SOFTS','pk':'SOFTS',
    'lh':'SOFTS','sp':'SOFTS','b':'SOFTS','br':'SOFTS',
}

FACTOR_NAMES = ["ret_5d","oi_5d","rsi14","vol_5d","ret_10d","range_5d","atrp_5d"]
N_FACTORS = len(FACTOR_NAMES)


def _base_sym(sym: str) -> str:
    s = sym.lower().split('.')[0].strip()
    while s and s[-1].isdigit(): s = s[:-1]
    return s[:-2] if s.endswith('fi') else s


def build_sector_lookup(syms: List[str]) -> Dict[int, str]:
    return {si: SECTOR_MAP.get(_base_sym(s), 'OTHER') for si, s in enumerate(syms)}


def compute_rsi_manual(C: np.ndarray, NS: int, ND: int, period: int = 14) -> np.ndarray:
    rsi = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        gains = np.where(np.isnan(c[:-1]) | np.isnan(c[1:]), np.nan,
                         np.maximum(np.diff(c), 0.0))
        losses = np.where(np.isnan(c[:-1]) | np.isnan(c[1:]), np.nan,
                          np.maximum(-np.diff(c), 0.0))
        for di in range(period, ND):
            if np.isnan(gains[di - 1]):
                continue
            window_g = gains[di - period:di]
            window_l = losses[di - period:di]
            vg = window_g[~np.isnan(window_g)]
            vl = window_l[~np.isnan(window_l)]
            if len(vg) < period:
                continue
            avg_g = np.mean(vg)
            avg_l = np.mean(vl) if len(vl) > 0 else 0.0
            rsi[si, di] = 100.0 if avg_l == 0 else 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    return rsi


def compute_raw_factors(C, O, H, L, V, OI, NS, ND) -> Dict[str, np.ndarray]:
    t0 = time.time()
    print("[V120] Computing raw factors...", flush=True)

    ret_5d = np.full((NS, ND), np.nan)
    oi_5d = np.full((NS, ND), np.nan)
    vol_5d = np.full((NS, ND), np.nan)
    range_5d = np.full((NS, ND), np.nan)
    atrp_5d = np.full((NS, ND), np.nan)
    ret_10d = np.full((NS, ND), np.nan)
    vol_median = np.full((NS, ND), np.nan)

    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(C[si,di]) and not np.isnan(C[si,di-5]) and C[si,di-5] > 0:
                ret_5d[si,di] = C[si,di] / C[si,di-5] - 1.0
            if not np.isnan(OI[si,di]) and not np.isnan(OI[si,di-5]) and OI[si,di-5] > 0:
                oi_5d[si,di] = OI[si,di] / OI[si,di-5] - 1.0
            vals = V[si, di-5:di]; valid = vals[~np.isnan(vals)]
            if len(valid) >= 3: vol_5d[si,di] = np.mean(valid)
            rng = []
            for j in range(di-5, di):
                if (not np.isnan(H[si,j]) and not np.isnan(L[si,j])
                        and not np.isnan(C[si,j]) and C[si,j] > 0 and H[si,j] > L[si,j]):
                    rng.append((H[si,j] - L[si,j]) / C[si,j])
            if len(rng) >= 3: range_5d[si,di] = np.mean(rng)
            if di >= 6:
                atrs = []
                for j in range(di-5, di):
                    hh,ll,cc = H[si,j],L[si,j],C[si,j]
                    if not any(np.isnan([hh,ll,cc])):
                        prev = C[si,j-1] if j>0 and not np.isnan(C[si,j-1]) else cc
                        atrs.append(max(hh-ll, abs(hh-prev), abs(ll-prev)))
                if atrs and not np.isnan(C[si,di]) and C[si,di] > 0:
                    atrp_5d[si,di] = np.mean(atrs) / C[si,di]
        for di in range(10, ND):
            if not np.isnan(C[si,di]) and not np.isnan(C[si,di-10]) and C[si,di-10] > 0:
                ret_10d[si,di] = C[si,di] / C[si,di-10] - 1.0
        for di in range(60, ND):
            vals = V[si, di-60:di]; valid = vals[~np.isnan(vals)]
            if len(valid) >= 20: vol_median[si,di] = np.median(valid)

    rsi14 = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nan_mask = np.isnan(C[si])
            try: rsi14[si] = np.where(nan_mask, np.nan, talib.RSI(c, 14))
            except Exception: pass
    needs_fb = np.all(np.isnan(rsi14), axis=1)
    if needs_fb.any():
        rm = compute_rsi_manual(C, NS, ND, 14)
        for si in range(NS):
            if needs_fb[si]: rsi14[si] = rm[si]

    fwd_ret = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND-5):
            if not np.isnan(C[si,di+5]) and not np.isnan(C[si,di]) and C[si,di] > 0:
                fwd_ret[si,di] = C[si,di+5] / C[si,di] - 1.0

    atr_mean = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            atrs = []
            for j in range(di-14, di):
                hh,ll,cc = H[si,j],L[si,j],C[si,j]
                if not any(np.isnan([hh,ll,cc])):
                    prev = C[si,j-1] if j>0 and not np.isnan(C[si,j-1]) else cc
                    atrs.append(max(hh-ll, abs(hh-prev), abs(ll-prev)))
            if atrs and not np.isnan(C[si,di]) and C[si,di] > 0:
                atr_mean[si,di] = np.mean(atrs) / C[si,di]

    print(f"  Raw factors done: {time.time()-t0:.1f}s", flush=True)
    return {"ret_5d":ret_5d,"oi_5d":oi_5d,"vol_5d":vol_5d,"range_5d":range_5d,
            "atrp_5d":atrp_5d,"ret_10d":ret_10d,"rsi14":rsi14,
            "fwd_ret_5d":fwd_ret,"atr_mean":atr_mean,"vol_median":vol_median}


def normalize_factor(f: np.ndarray, NS: int, ND: int, mc: int = 10) -> np.ndarray:
    n = np.full((NS, ND), np.nan)
    for di in range(ND):
        v = f[:,di]; valid = v[~np.isnan(v)]
        if len(valid) < mc: continue
        mu, sig = np.mean(valid), np.std(valid)
        if sig < 1e-12: continue
        for si in range(NS):
            if not np.isnan(v[si]): n[si,di] = (v[si]-mu)/sig
    return n


# =====================================================================
# READINESS DIMENSIONS
# =====================================================================

def compute_research_readiness(raw, NS, ND, ic_window=20, ic_thresh=0.02):
    """P_r = fraction of factors with |rolling_IC| > ic_thresh."""
    t0 = time.time()
    print(f"[V120] Research readiness (ic_w={ic_window}, th={ic_thresh})...", flush=True)
    fwd = raw["fwd_ret_5d"]
    strong = np.zeros((N_FACTORS, ND), dtype=bool)
    for fi, fn in enumerate(FACTOR_NAMES):
        fac = raw[fn]
        for di in range(ic_window+5, ND):
            ics = []
            for tdi in range(di-ic_window, di):
                fd, rd = fac[:,tdi], fwd[:,tdi]
                m = (~np.isnan(fd)) & (~np.isnan(rd))
                fv, rv = fd[m], rd[m]
                if len(fv) >= 15:
                    c = np.corrcoef(pd.Series(fv).rank().values,
                                    pd.Series(rv).rank().values)[0,1]
                    if not np.isnan(c): ics.append(c)
            if len(ics) >= 5 and abs(np.mean(ics)) > ic_thresh:
                strong[fi,di] = True
        if fi % 2 == 0: print(f"  {fn}: {time.time()-t0:.1f}s", flush=True)
    pr = np.zeros((NS, ND))
    for di in range(ND):
        pr[:,di] = np.sum(strong[:,di]) / N_FACTORS
    print(f"  Research readiness done: {time.time()-t0:.1f}s", flush=True)
    return pr


def compute_risk_readiness(C, NS, ND, heat_threshold=0.08, vol_lb=20):
    """P_k = 1.0 if portfolio vol < threshold, else decay."""
    pv = np.full(ND, np.nan)
    for di in range(vol_lb+1, ND):
        dr = []
        for dd in range(di-vol_lb, di):
            rs = []
            for si in range(NS):
                if not np.isnan(C[si,dd]) and not np.isnan(C[si,dd-1]) and C[si,dd-1]>0:
                    rs.append(C[si,dd]/C[si,dd-1]-1.0)
            if rs: dr.append(np.mean(rs))
        if len(dr) >= vol_lb//2: pv[di] = np.std(dr)
    rk = np.ones(ND)
    for di in range(ND):
        v = pv[di]
        if not np.isnan(v) and v > heat_threshold:
            rk[di] = max(0.2, heat_threshold/v)
    return rk


def compute_execution_readiness(raw, NS, ND, vol_ratio_min=0.5):
    """P_e = min(1.0, vol_5d / vol_median_60d)."""
    v5, vm = raw["vol_5d"], raw["vol_median"]
    pe = np.full((NS, ND), 0.5)
    for si in range(NS):
        for di in range(60, ND):
            a, b = v5[si,di], vm[si,di]
            if np.isnan(a) or np.isnan(b) or b <= 0: continue
            r = a / b
            pe[si,di] = min(1.0, r / vol_ratio_min) if r < vol_ratio_min else min(1.0, r)
    return pe


# =====================================================================
# NW Kernel (V86-style, no BMA)
# =====================================================================

def compute_nw_predictions(raw, NS, ND, tw=40, bw=1.0):
    t0 = time.time()
    print(f"[V120] NW predictions (tw={tw}, bw={bw:.1f})...", flush=True)
    normed = {fn: normalize_factor(raw[fn], NS, ND) for fn in FACTOR_NAMES}
    fwd = raw["fwd_ret_5d"]; atr = raw["atr_mean"]
    pred = np.full((NS, ND), np.nan)
    for di in range(tw+10, ND):
        tf, tt = [], []
        for tdi in range(max(10, di-tw), di):
            for si in range(NS):
                f = np.array([normed[fn][si,tdi] for fn in FACTOR_NAMES])
                t = fwd[si,tdi]
                if not np.any(np.isnan(f)) and not np.isnan(t):
                    tf.append(f); tt.append(t)
        if len(tf) < 20: continue
        tX, tY = np.array(tf), np.array(tt)
        fs = np.std(tX, axis=0); fs[fs<1e-12] = 1.0
        for si in range(NS):
            qf = np.array([normed[fn][si,di] for fn in FACTOR_NAMES])
            if np.any(np.isnan(qf)): continue
            h = max(0.1, atr[si,di]*bw) if not np.isnan(atr[si,di]) else bw
            d = np.sqrt(np.sum(((tX - qf)/fs)**2, axis=1))
            sd = d / h
            w = np.zeros(len(tX))
            m = sd <= 1.0
            if not np.any(m):
                idx = np.argmin(d)
                if d[idx] < 1e12: w[idx] = 1.0; m = np.zeros(len(d),bool); m[idx]=True
                else: continue
            else: w[m] = 0.75*(1.0-sd[m]**2)
            ws = np.sum(w)
            if ws > 1e-12: pred[si,di] = np.sum(w*tY)/ws
        if di % 100 == 0:
            print(f"  di={di}/{ND} valid={np.sum(~np.isnan(pred[:,di]))}/{NS}", flush=True)
    print(f"  NW done: {time.time()-t0:.1f}s", flush=True)
    return pred


def compute_ker(C, NS, ND):
    ker = np.full((NS, ND), np.nan)
    regime = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        for di in range(10, ND):
            cl = C[si, di-10:di+1]; v = cl[~np.isnan(cl)]
            if len(v) < 10 or v[0] <= 0: continue
            nc = abs(v[-1]-v[0]); tc = np.sum(np.abs(np.diff(v)))
            if tc > 1e-10: ker[si,di] = nc/tc
            if not np.isnan(ker[si,di]):
                if ker[si,di] < 0.15: regime[si,di] = 1
                elif ker[si,di] > 0.3: regime[si,di] = -1
    return regime


def get_mode(rtw, wt, wrw):
    if len(rtw) < 5: return "normal"
    wr = sum(rtw[-wrw:])/len(rtw[-wrw:])
    if wr > wt: return "winning"
    return "losing" if wr < 0.50 else "normal"


def atr_at(H, L, C, si, di, start):
    a = []
    for j in range(max(start, di-14), di):
        hh,ll,cc = H[si,j],L[si,j],C[si,j]
        if not any(np.isnan([hh,ll,cc])):
            a.append(max(hh-ll, abs(hh-cc), abs(ll-cc)))
    return np.mean(a) if a else None


# =====================================================================
# Backtest with Readiness scoring
# =====================================================================

def backtest_v120(C, O, H, L, NS, ND, dates, syms, pred, ker,
                  p_r, risk_rdy, p_e, sector_lu,
                  rdy_min=0.3, top_n=2, mps=2, hd=5,
                  atr_stop=3.0, start_di=60, end_di=None):
    if end_di is None: end_di = ND - 1
    eq = CASH0; peak = eq; max_dd = 0.0
    positions = []; trades = []; rtw = []
    skipped = 0; total_cand = 0

    for di in range(max(start_di,1), end_di):
        d = dates[di]; dpnl = 0.0
        mode = get_mode(rtw, 0.60, 15)
        new_pos = []
        pos_by_si = defaultdict(list)
        for p in positions: pos_by_si[p[0]].append(p)

        for si, plist in pos_by_si.items():
            c = C[si,di]
            if np.isnan(c):
                for p in plist: new_pos.append(p)
                continue
            hold = di - min(p[0] for p in plist)
            stopped = any(c < p[2] for p in plist)
            if stopped or hold >= hd:
                for _, edi, ep, sp, alloc in plist:
                    pnl = (c-ep)/ep - COMM; profit = eq*alloc*pnl; dpnl += profit
                    trades.append({"pnl_abs":profit,"pnl_pct":pnl*100,"days":di-edi+1,
                                   "di":di,"year":d.year,"sym":syms[si],
                                   "sector":sector_lu.get(si,'OTHER'),
                                   "reason":"stop" if stopped else "hold","mode":mode[0].upper()})
                    rtw.append(1 if pnl > 0 else 0)
            else:
                for p in plist: new_pos.append(p)

        positions = new_pos; eq += dpnl
        if eq > peak: peak = eq
        if peak > 0:
            dd = (peak-eq)/peak*100
            if dd > max_dd: max_dd = dd
        if eq <= 0: break

        # ENTRY
        held = {p[0] for p in positions}
        if len(held) >= top_n: continue
        cands = []
        for si in range(NS):
            if si in held: continue
            p = pred[si,di]
            if np.isnan(p) or di+1 >= ND or np.isnan(O[si,di+1]): continue
            if ker[si,di] < 0: continue
            total_cand += 1
            pr_v = p_r[si,di] if di < ND else 0.0
            pg_v = 1.0 if ker[si,di] >= 0 else 0.5
            pk_v = risk_rdy[di] if di < ND else 0.5
            pe_v = p_e[si,di] if not np.isnan(p_e[si,di]) else 0.5
            rdy = min(pr_v, pg_v, pk_v, pe_v)
            if rdy < rdy_min: skipped += 1; continue
            cands.append((p, si, rdy))

        if not cands: continue
        cands.sort(key=lambda x: -x[0])
        n_take = top_n
        if mode == "winning": n_take = min(top_n+1, top_n*2)
        elif mode == "losing": n_take = max(1, top_n-1)

        sec_cnt = defaultdict(int)
        for sh in held: sec_cnt[sector_lu.get(sh,'OTHER')] += 1
        entries = []
        for pv, si, rdy in cands:
            if len(held)+len(entries) >= n_take or si in held: break
            ss = sector_lu.get(si,'OTHER')
            if sec_cnt[ss] >= mps or pv <= 0: continue
            entries.append((pv, si, ss, rdy)); sec_cnt[ss] += 1
        if not entries: continue

        ba = LEVERAGE / (len(positions)+len(entries))
        upd = [(si, edi, ep, sp, ba) for si, edi, ep, sp, _ in positions]
        for pv, si, ss, rdy in entries:
            ep = O[si,di+1]
            if np.isnan(ep) or ep <= 0: continue
            a = atr_at(H, L, C, si, di, start_di)
            if a is None: continue
            upd.append((si, di+1, ep, ep-atr_stop*a, ba*rdy))
        positions = upd

    for si, edi, ep, sp, alloc in positions:
        c = C[si,ND-1]
        if not np.isnan(c) and c > 0: eq += eq*alloc*((c-ep)/ep-COMM)

    if trades:
        trades[0]["rdy_info"] = f"rdy_min={rdy_min:.2f} skip={skipped}/{total_cand}"
    return trades, eq, max_dd


def analyze(trades, equity, max_dd, label=""):
    if not trades: print(f"  {label}: no trades"); return None
    nw = sum(1 for t in trades if t["pnl_pct"]>0)
    wr = nw/len(trades)*100
    nd = max(1, trades[-1]["di"]-trades[0]["di"])
    ann = ((equity/CASH0)**(1/max(1.0,nd/252))-1)*100
    ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
    rets = np.array(ap)/CASH0
    sh = np.mean(rets)/np.std(rets)*np.sqrt(252) if np.std(rets)>0 else 0
    ns = sum(1 for t in trades if t["reason"]=="stop")
    nh = sum(1 for t in trades if t["reason"]=="hold")
    mc = defaultdict(int)
    for t in trades: mc[t.get("mode","N")] += 1
    sc = defaultdict(int)
    for t in trades: sc[t.get("sector","OTHER")] += 1
    ss = " ".join(f"{k}:{v}" for k,v in sorted(sc.items()))
    ap_ = np.mean([t["pnl_pct"] for t in trades])
    aw = np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"]>0])
    al = np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"]<=0])
    ri = trades[0].get("rdy_info","")

    print(f"  {label}: {len(trades)}t (s:{ns} h:{nh}) WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% Sh={sh:.2f} eq={equity:,.0f}")
    print(f"    avg={ap_:+.3f}% win={aw:+.3f}% loss={al:+.3f}% modes=[W:{mc['W']} N:{mc['N']} L:{mc['L']}]")
    if ri: print(f"    readiness: {ri}")
    print(f"    sectors: {ss}")

    yr = {}
    for t in trades:
        y = t["year"]
        if y not in yr: yr[y] = {"n":0,"w":0,"pnl":[]}
        yr[y]["n"] += 1; yr[y]["w"] += (1 if t["pnl_pct"]>0 else 0); yr[y]["pnl"].append(t["pnl_pct"])
    for y in sorted(yr):
        ys = yr[y]; cum = np.prod([1+p/100 for p in ys["pnl"]])-1
        print(f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% cum={cum:+.1%}")
    return {"n":len(trades),"wr":wr,"dd":max_dd,"ann":ann,"sh":sh,"eq":equity}


def walk_forward(C, O, H, L, NS, ND, dates, syms, pred, ker,
                 p_r, rk, pe, sector_lu, rdy_min=0.3, top_n=2, mps=2, hd=5, label=""):
    print(f"\n{'='*70}\n  WALK-FORWARD V120 {label}\n  tn={top_n} mps={mps} rdy_min={rdy_min:.2f}\n{'='*70}")
    years = sorted(set(d.year for d in dates))
    all_t = []
    for ty in range(2019, years[-1]+1):
        ts = te = None
        for i, d in enumerate(dates):
            if d.year == ty and ts is None: ts = i
            if d.year == ty: te = i
        if ts is None: continue
        trades, _, _ = backtest_v120(C,O,H,L,NS,ND,dates,syms,pred,ker,
                                     p_r,rk,pe,sector_lu,rdy_min=rdy_min,
                                     top_n=top_n,mps=mps,hd=hd,
                                     start_di=ts,end_di=te+1)
        tt = [t for t in trades if dates[t["di"]].year==ty]
        all_t.extend(tt)
        if tt:
            n=len(tt); nw=sum(1 for t in tt if t["pnl_pct"]>0)
            print(f"  {ty}: {n}t WR={nw/n*100:.1f}% avg={np.mean([t['pnl_pct'] for t in tt]):+.2f}%", flush=True)
        else:
            print(f"  {ty}: no trades", flush=True)

    if all_t:
        nw=sum(1 for t in all_t if t["pnl_pct"]>0)
        cum=np.prod([1+t["pnl_pct"]/100 for t in all_t])-1
        sc=defaultdict(int)
        for t in all_t: sc[t.get("sector","OTHER")]+=1
        print(f"\n  WF TOTAL: {len(all_t)}t WR={nw/len(all_t)*100:.1f}% cum={cum:+.1%}")
        print(f"  SECTORS: {' '.join(f'{k}:{v}' for k,v in sorted(sc.items()))}")
    return all_t


def main():
    t0 = time.time()
    print("="*70)
    print("  V120: 凡事豫则立 - Multi-Period Readiness Index Strategy")
    print("  readiness = min(Research, Regime, Risk, Execution)")
    print("  Walk-forward 2019-2026. No leverage.")
    print("="*70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start="2016-01-01")
    print(f"  {NS} sym, {ND} days, {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    sector_lu = build_sector_lookup(syms)
    sd = defaultdict(int)
    for s in sector_lu.values(): sd[s] += 1
    print(f"  Sectors: {dict(sd)}")

    bt_2019 = next((i for i, d in enumerate(dates) if d >= pd.Timestamp("2019-01-01")), None)

    # 1. Raw factors + NW predictions
    raw = compute_raw_factors(C, O, H, L, V, OI, NS, ND)
    ker = compute_ker(C, NS, ND)
    pred = compute_nw_predictions(raw, NS, ND, tw=40, bw=1.0)

    # 2. Pre-compute readiness components
    ic_configs = [15, 20, 30]
    ht_configs = [0.05, 0.08]
    vr_configs = [0.5, 0.7]

    pr_cache = {icw: compute_research_readiness(raw, NS, ND, ic_window=icw) for icw in ic_configs}
    rk_cache = {ht: compute_risk_readiness(C, NS, ND, heat_threshold=ht) for ht in ht_configs}
    pe_cache = {vrm: compute_execution_readiness(raw, NS, ND, vol_ratio_min=vrm) for vrm in vr_configs}

    # 3. Parameter sweep
    print(f"\n{'='*70}\n  PARAMETER SWEEP (2019-2026) - 凡事豫则立\n{'='*70}")
    results = []; sc_count = 0
    rdy_configs = [0.2, 0.3, 0.4]

    for icw in ic_configs:
        pr = pr_cache[icw]
        for ht in ht_configs:
            rk = rk_cache[ht]
            for vrm in vr_configs:
                pe = pe_cache[vrm]
                for rm in rdy_configs:
                    for tn in [2, 3]:
                        for mps in [2, 3]:
                            sc_count += 1
                            trades, eq, dd = backtest_v120(
                                C,O,H,L,NS,ND,dates,syms,pred,ker,
                                pr,rk,pe,sector_lu,rdy_min=rm,top_n=tn,mps=mps,
                                start_di=bt_2019)
                            if len(trades) < 10: continue
                            nw=sum(1 for t in trades if t["pnl_pct"]>0)
                            wr=nw/len(trades)*100
                            nd=max(1,trades[-1]["di"]-trades[0]["di"])
                            ann=((eq/CASH0)**(1/max(1.0,nd/252))-1)*100
                            ap=[t["pnl_abs"] for t in sorted(trades, key=lambda x:x["di"])]
                            ra=np.array(ap)/CASH0
                            sh=np.mean(ra)/np.std(ra)*np.sqrt(252) if np.std(ra)>0 else 0
                            results.append({"icw":icw,"ht":ht,"vrm":vrm,"rm":rm,
                                            "tn":tn,"mps":mps,"n":len(trades),"wr":wr,
                                            "ann":ann,"dd":dd,"sh":sh,"eq":eq})

    # V96 baseline (no readiness)
    print("\n--- V96 baseline (no readiness) ---")
    pr1 = np.ones((NS,ND)); rk1 = np.ones(ND); pe1 = np.ones((NS,ND))
    for tn in [2,3]:
        for mps in [2,3]:
            sc_count += 1
            trades, eq, dd = backtest_v120(C,O,H,L,NS,ND,dates,syms,pred,ker,
                                           pr1,rk1,pe1,sector_lu,rdy_min=0.0,
                                           top_n=tn,mps=mps,start_di=bt_2019)
            if len(trades) < 10: continue
            nw=sum(1 for t in trades if t["pnl_pct"]>0)
            wr=nw/len(trades)*100
            nd=max(1,trades[-1]["di"]-trades[0]["di"])
            ann=((eq/CASH0)**(1/max(1.0,nd/252))-1)*100
            ap=[t["pnl_abs"] for t in sorted(trades, key=lambda x:x["di"])]
            ra=np.array(ap)/CASH0
            sh=np.mean(ra)/np.std(ra)*np.sqrt(252) if np.std(ra)>0 else 0
            results.append({"icw":0,"ht":0,"vrm":0,"rm":0.0,"tn":tn,"mps":mps,
                            "n":len(trades),"wr":wr,"ann":ann,"dd":dd,"sh":sh,"eq":eq})

    results.sort(key=lambda x: -x["ann"])
    print(f"\n  Evaluated {sc_count} configs, {len(results)} with 10+ trades")
    print(f"\n{'ICw':>4} {'HT':>5} {'VRM':>4} {'Rdy':>4} {'TN':>3} {'MPS':>3} {'N':>5} {'WR':>6} {'Ann':>9} {'DD':>7} {'Sh':>7}")
    print("-"*75)
    for r in results[:15]:
        tag = f"{r['icw']}" if r['icw']>0 else "base"
        print(f"{tag:>4} {r['ht']:>5.2f} {r['vrm']:>4.1f} {r['rm']:>4.1f} {r['tn']:>3} {r['mps']:>3} {r['n']:>5} {r['wr']:>5.1f}% {r['ann']:>+8.1f}% {r['dd']:>6.1f}% {r['sh']:>6.2f}")

    if not results: print("  No configs with 10+ trades."); return

    # 4. Walk-forward for best configs
    best_ann = results[0]
    best_sh = max(results, key=lambda x: x["sh"])
    best_ra = max(results, key=lambda x: x["ann"]/max(x["dd"],1.0))

    for label, best in [("BEST-ANN",best_ann),("BEST-SHARPE",best_sh),("BEST-RISK-ADJ",best_ra)]:
        if best["icw"] == 0:
            pr_w, rk_w, pe_w = pr1, rk1, pe1
        else:
            pr_w = pr_cache[best["icw"]]; rk_w = rk_cache[best["ht"]]; pe_w = pe_cache[best["vrm"]]
        walk_forward(C,O,H,L,NS,ND,dates,syms,pred,ker,pr_w,rk_w,pe_w,sector_lu,
                     rdy_min=best["rm"],top_n=best["tn"],mps=best["mps"],label=label)

    # 5. Compare V120 vs V96 baseline
    print(f"\n{'='*70}\n  COMPARISON: V120 (Readiness) vs V96 baseline\n{'='*70}")
    b = best_ann
    pr_b = pr1 if b["icw"]==0 else pr_cache[b["icw"]]
    rk_b = rk1 if b["icw"]==0 else rk_cache[b["ht"]]
    pe_b = pe1 if b["icw"]==0 else pe_cache[b["vrm"]]

    t120, e120, d120 = backtest_v120(C,O,H,L,NS,ND,dates,syms,pred,ker,
                                     pr_b,rk_b,pe_b,sector_lu,rdy_min=b["rm"],
                                     top_n=b["tn"],mps=b["mps"],start_di=bt_2019)
    t96, e96, d96 = backtest_v120(C,O,H,L,NS,ND,dates,syms,pred,ker,
                                  pr1,rk1,pe1,sector_lu,rdy_min=0.0,
                                  top_n=2,mps=2,start_di=bt_2019)

    print(f"\n  V120 BEST-ANN (Readiness):")
    analyze(t120, e120, d120, "V120-Readiness")
    print(f"\n  V96 BASELINE (no filter):")
    analyze(t96, e96, d96, "V96-baseline")
    if t120 and t96:
        print(f"\n  Delta: eq={e120-e96:+,.0f} dd={d120-d96:+.1f}% trades={len(t120)-len(t96):+d}")
    print(f"\n[V120] Done. {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
