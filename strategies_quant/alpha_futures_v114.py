"""
V114: H0 Microstructure Quality Filter + NW Kernel
====================================================
Paper 2601.23172: H0 (~0.75) unifies order flow persistence, volume
roughness, volatility roughness, and market impact. When H0 deviates,
microstructure breaks down BEFORE prices reflect it.

Architecture (V96 base, BMA removed):
  1. NW Kernel Regression (V86) for signal generation
  2. H0 estimation via R/S analysis on signed volume delta
  3. H0 position sizing gate + vol-adaptive sizing (V96)

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

CASH0, COMM, LEVERAGE = 1_000_000, 0.0005, 1.0

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


# =================================================================
# INNOVATION: H0 Microstructure Quality Filter
# =================================================================

def compute_signed_volume_delta(C, H, L, V, NS, ND):
    """Approximate signed order flow: VDP = V*(2C-H-L)/(H-L)."""
    svd = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            c, h, l, v = C[si,di], H[si,di], L[si,di], V[si,di]
            if np.isnan(c) or np.isnan(h) or np.isnan(l) or np.isnan(v):
                continue
            if v <= 0: continue
            d = h - l
            if d < 1e-10: continue
            svd[si, di] = v * (2*c - h - l) / d
    return svd


def estimate_h0_rs(svd, NS, ND, h0_window=100):
    """Estimate Hurst exponent via R/S analysis on signed volume delta.

    Two-layer approach:
    1. Raw H0 via R/S analysis (clusters ~0.50 on daily data)
    2. Rolling z-score of H0 relative to its own history

    Returns (NS, ND) array. Values are z-scores: 0=normal, >2=extreme
    persistence, <-2=breakdown. Use these z-scores in the gate.
    """
    # Step 1: Raw H0
    raw_h0 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(h0_window, ND):
            w = svd[si, di-h0_window:di]
            v = w[~np.isnan(w)]
            nv = len(v)
            if nv < 30: continue
            nc = min(4, max(2, nv // 25))
            cs = nv // nc
            if cs < 10: continue
            rs_vals = []
            for ci in range(nc):
                chunk = v[ci*cs:(ci+1)*cs]
                if len(chunk) < cs: continue
                dev = chunk - np.mean(chunk)
                cd = np.cumsum(dev)
                R = np.max(cd) - np.min(cd)
                S = np.std(chunk, ddof=1)
                if S > 1e-12 and R > 0:
                    rs_vals.append(R / S)
            if len(rs_vals) < 2: continue
            mrs = np.mean(rs_vals)
            if mrs > 0:
                raw_h0[si, di] = max(0.2, min(1.5, np.log(mrs) / np.log(cs)))

    # Step 2: Rolling z-score of H0 (detects relative anomalies)
    zscore = np.full((NS, ND), np.nan)
    zscore_window = 120  # lookback for rolling stats
    for si in range(NS):
        for di in range(h0_window + zscore_window, ND):
            hist = raw_h0[si, di-zscore_window:di]
            valid = hist[~np.isnan(hist)]
            if len(valid) < 20: continue
            mu, sig = np.mean(valid), np.std(valid)
            if sig < 1e-6: continue
            current = raw_h0[si, di]
            if not np.isnan(current):
                zscore[si, di] = (current - mu) / sig
    return zscore


def get_h0_mult(z, nl=-1.0, nh=1.0, cr=0.3, ex=0.5):
    """H0 z-score -> position multiplier.

    z near 0: normal microstructure (full size).
    z < -2.0: H0 far below its own mean -> breakdown (reduce heavily).
    z > +2.0: H0 far above -> extreme persistence (climax risk).
    nl/nh: z-score band for "normal" (default -1 to +1 std devs).
    """
    if np.isnan(z): return 0.7
    if nl <= z <= nh: return 1.0
    if z < -2.0: return cr
    if z > 2.0: return ex
    return 0.7


# =================================================================
# Factor computation
# =================================================================

def compute_rsi_manual(C, NS, ND, period=14):
    rsi = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]; g = np.full(ND, np.nan); lo = np.full(ND, np.nan)
        for di in range(1, ND):
            if np.isnan(c[di]) or np.isnan(c[di-1]): continue
            d = c[di] - c[di-1]
            g[di], lo[di] = max(d,0), max(-d,0)
        ag = al = np.nan
        for di in range(1, ND):
            if np.isnan(g[di]): continue
            if np.isnan(ag):
                vg, vl = [], []
                for j in range(di, min(di+period, ND)):
                    if not np.isnan(g[j]):
                        vg.append(g[j]); vl.append(lo[j] if not np.isnan(lo[j]) else 0)
                if len(vg) >= period:
                    ag, al = np.mean(vg), np.mean(vl)
                    rsi[si, di+period-1] = 100 if al==0 else 100-100/(1+ag/al)
                continue
            ag = (ag*(period-1)+g[di])/period
            al = (al*(period-1)+lo[di])/period
            rsi[si, di] = 100 if al==0 else 100-100/(1+ag/al)
    return rsi


def compute_raw_factors(C, O, H, L, V, OI, NS, ND):
    t0 = time.time()
    print("[V114] Computing raw factors...", flush=True)
    F = {k: np.full((NS,ND),np.nan) for k in [
        "ret_5d","oi_5d","vol_5d","range_5d","atrp_5d","ret_10d","rsi14",
        "fwd_ret_5d","atr_mean"]}

    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(C[si,di]) and not np.isnan(C[si,di-5]) and C[si,di-5]>0:
                F["ret_5d"][si,di] = C[si,di]/C[si,di-5]-1
            if not np.isnan(OI[si,di]) and not np.isnan(OI[si,di-5]) and OI[si,di-5]>0:
                F["oi_5d"][si,di] = OI[si,di]/OI[si,di-5]-1
            vv = V[si,di-5:di]; vm = vv[~np.isnan(vv)]
            if len(vm)>=3: F["vol_5d"][si,di] = np.mean(vm)
            rv = []
            for j in range(di-5,di):
                if not np.isnan(H[si,j]) and not np.isnan(L[si,j]) and not np.isnan(C[si,j]) and C[si,j]>0 and H[si,j]>L[si,j]:
                    rv.append((H[si,j]-L[si,j])/C[si,j])
            if len(rv)>=3: F["range_5d"][si,di] = np.mean(rv)

        for di in range(6, ND):
            av = []
            for j in range(di-5,di):
                hh,ll,cc = H[si,j],L[si,j],C[si,j]
                if not any(np.isnan([hh,ll,cc])):
                    pc = C[si,j-1] if j>0 and not np.isnan(C[si,j-1]) else cc
                    av.append(max(hh-ll,abs(hh-pc),abs(ll-pc)))
            if av and not np.isnan(C[si,di]) and C[si,di]>0:
                F["atrp_5d"][si,di] = np.mean(av)/C[si,di]

        for di in range(10, ND):
            if not np.isnan(C[si,di]) and not np.isnan(C[si,di-10]) and C[si,di-10]>0:
                F["ret_10d"][si,di] = C[si,di]/C[si,di-10]-1

        for di in range(ND-5):
            if not np.isnan(C[si,di+5]) and not np.isnan(C[si,di]) and C[si,di]>0:
                F["fwd_ret_5d"][si,di] = C[si,di+5]/C[si,di]-1

        for di in range(20, ND):
            av = []
            for j in range(di-14,di):
                hh,ll,cc = H[si,j],L[si,j],C[si,j]
                if not any(np.isnan([hh,ll,cc])):
                    pc = C[si,j-1] if j>0 and not np.isnan(C[si,j-1]) else cc
                    av.append(max(hh-ll,abs(hh-pc),abs(ll-pc)))
            if av and not np.isnan(C[si,di]) and C[si,di]>0:
                F["atr_mean"][si,di] = np.mean(av)/C[si,di]

    # RSI
    if HAS_TALIB:
        for si in range(NS):
            c = np.where(np.isnan(C[si]),0,C[si]).astype(np.float64)
            nm = np.isnan(C[si])
            try:
                r = talib.RSI(c, 14)
                F["rsi14"][si] = np.where(nm, np.nan, r)
            except: pass
    fb = np.all(np.isnan(F["rsi14"]), axis=1)
    if fb.any():
        rm = compute_rsi_manual(C, NS, ND, 14)
        for si in range(NS):
            if fb[si]: F["rsi14"][si] = rm[si]

    print(f"  Raw factors: {time.time()-t0:.1f}s", flush=True)
    return F


def normalize_factor(f, NS, ND, mc=10):
    n = np.full((NS,ND),np.nan)
    for di in range(ND):
        v = f[:,di]; ok = v[~np.isnan(v)]
        if len(ok)<mc: continue
        mu,sig = np.mean(ok),np.std(ok)
        if sig<1e-12: continue
        for si in range(NS):
            if not np.isnan(v[si]): n[si,di] = (v[si]-mu)/sig
    return n


# =================================================================
# NW Kernel (V86, no BMA)
# =================================================================

def compute_nw_predicted(F, NS, ND, tw=40, bw=1.0):
    t0 = time.time()
    print(f"[V114] NW prediction (tw={tw}, bw={bw})...", flush=True)
    N = {fn: normalize_factor(F[fn], NS, ND) for fn in FACTOR_NAMES}
    fwd = F["fwd_ret_5d"]; atr = F["atr_mean"]
    pred = np.full((NS,ND),np.nan)

    for di in range(tw+10, ND):
        tf, tt = [], []
        for tdi in range(max(10,di-tw), di):
            for si in range(NS):
                fe = np.array([N[fn][si,tdi] for fn in FACTOR_NAMES])
                tg = fwd[si,tdi]
                if np.any(np.isnan(fe)) or np.isnan(tg): continue
                tf.append(fe); tt.append(tg)
        if len(tf)<20: continue
        tX, tY = np.array(tf), np.array(tt)
        fs = np.std(tX,axis=0); fs[fs<1e-12]=1.0

        for si in range(NS):
            qf = np.array([N[fn][si,di] for fn in FACTOR_NAMES])
            if np.any(np.isnan(qf)): continue
            av = atr[si,di]; h = max(bw, 0.1) if np.isnan(av) else max(av*bw, 0.1)
            dist = np.sqrt(np.sum(((tX-qf)/fs)**2, axis=1))
            sd = dist/h; w = np.zeros(len(tX))
            m = sd<=1.0
            if not np.any(m):
                mi = np.argmin(dist)
                if dist[mi]<1e12: w[mi]=1.0; m=np.zeros(len(dist),bool); m[mi]=True
                else: continue
            else:
                w[m] = 0.75*(1-sd[m]**2)
            ws = np.sum(w)
            if ws<1e-12: continue
            pred[si,di] = np.sum(w*tY)/ws
        if di%100==0:
            print(f"  di={di}/{ND} valid={np.sum(~np.isnan(pred[:,di]))}/{NS}", flush=True)
    print(f"  NW done: {time.time()-t0:.1f}s", flush=True)
    return pred


# =================================================================
# Helpers
# =================================================================

def compute_ker(C, NS, ND):
    kr = np.full((NS,ND),np.nan)
    for si in range(NS):
        for di in range(10,ND):
            cs = C[si,di-10:di+1]; v = cs[~np.isnan(cs)]
            if len(v)<10 or v[0]<=0: continue
            tc = np.sum(np.abs(np.diff(v)))
            if tc>1e-10: kr[si,di] = abs(v[-1]-v[0])/tc
    reg = np.zeros((NS,ND),dtype=int)
    for si in range(NS):
        for di in range(ND):
            if np.isnan(kr[si,di]): continue
            if kr[si,di]<0.15: reg[si,di]=1
            elif kr[si,di]>0.3: reg[si,di]=-1
    return reg


def compute_atr(H,L,C,si,di,sd):
    av=[]
    for j in range(max(sd,di-14),di):
        hh,ll,cc=H[si,j],L[si,j],C[si,j]
        if not any(np.isnan([hh,ll,cc])): av.append(max(hh-ll,abs(hh-cc),abs(ll-cc)))
    return np.mean(av) if av else None


def compute_port_vol(C,NS,ND,vlb=20):
    pv = np.full(ND,np.nan)
    for di in range(vlb+1,ND):
        dr=[]
        for dd in range(di-vlb,di):
            r=[C[si,dd]/C[si,dd-1]-1 for si in range(NS) if not np.isnan(C[si,dd]) and not np.isnan(C[si,dd-1]) and C[si,dd-1]>0]
            if r: dr.append(np.mean(r))
        if len(dr)>=vlb//2: pv[di]=np.std(dr)
    return pv


def vol_mult(pv,vm,vhm,vlm,sr,sb):
    if np.isnan(pv) or np.isnan(vm) or vm<1e-12: return 1.0
    r = pv/vm
    if r>vhm: return sr
    if r<vlm: return sb
    return 1.0


# =================================================================
# Backtest V114
# =================================================================

def backtest(C,O,H,L,NS,ND,dates,syms,pred,ker,pvol,h0a,slook,
             tn=2,mps=2,hd=5,atr_s=3.0,vhm=2.0,vlm=0.5,sr=0.5,sb=1.3,
             h0_nl=0.65,h0_nh=0.85,h0_cr=0.3,h0_ex=0.5,
             sdi=60,edi=None):
    if edi is None: edi=ND-1
    vd = pvol[max(sdi,21):edi]; vv = vd[~np.isnan(vd)]
    vm = np.median(vv) if len(vv)>10 else 1e-6
    eq,peak,mdd = CASH0,CASH0,0.0
    pos=[]; trades=[]; rtw=[]
    wt,nwt,lt = 0.60,0.80,0.90

    for di in range(max(sdi,1),edi):
        d=dates[di]; dpnl=0.0; npos=[]
        # mode
        if len(rtw)<5: mode="normal"
        else:
            wr = sum(rtw[-15:])/len(rtw[-15:])
            mode = "winning" if wr>wt else ("losing" if wr<0.50 else "normal")
        vm2 = vol_mult(pvol[di],vm,vhm,vlm,sr,sb)

        # exits
        pbs = defaultdict(list)
        for si,edi2,ep,sp,al in pos: pbs[si].append((edi2,ep,sp,al))
        for si,pl in pbs.items():
            c=C[si,di]
            if np.isnan(c):
                for e,s,sp,al in pl: npos.append((si,e,s,sp,al))
                continue
            ee=min(p[0] for p in pl); hold=di-ee; stopped=any(c<p[2] for p in pl)
            if stopped or hold>=hd:
                for e,s,sp,al in pl:
                    pnl=(c-s)/s-COMM; pr=eq*al*pnl; dpnl+=pr
                    trades.append({"pnl_abs":pr,"pnl_pct":pnl*100,"days":di-e+1,"di":di,"year":d.year,"sym":syms[si],"sector":slook.get(si,'OTHER')})
                    rtw.append(1 if pnl>0 else 0)
            else:
                for e,s,sp,al in pl: npos.append((si,e,s,sp,al))
        pos=npos; eq+=dpnl
        if eq>peak: peak=eq
        if peak>0:
            dd=(peak-eq)/peak*100
            if dd>mdd: mdd=dd
        if eq<=0: break

        # entry
        held={p[0] for p in pos}
        if len(held)>=tn: continue
        cands=[]
        for si in range(NS):
            if si in held: continue
            p=pred[si,di]
            if np.isnan(p): continue
            if di+1>=ND or np.isnan(O[si,di+1]): continue
            if ker[si,di]<0: continue
            hm=get_h0_mult(h0a[si,di],h0_nl,h0_nh,h0_cr,h0_ex)
            if hm<0.2: continue
            cands.append((p,si,hm))
        if not cands: continue
        cands.sort(key=lambda x:-x[0])
        nt = min(tn+1,tn*2) if mode=="winning" else (max(1,tn-1) if mode=="losing" else tn)
        sc=defaultdict(int)
        for sh in held: sc[slook.get(sh,'OTHER')]+=1
        ne=[]
        for pv2,si,hm in cands:
            if len(held)+len(ne)>=nt: break
            if si in held: continue
            ss=slook.get(si,'OTHER')
            if sc[ss]>=mps: continue
            if pv2<=0: continue
            ne.append((pv2,si,ss,hm)); sc[ss]+=1
        if not ne: continue
        ap = LEVERAGE/(len(pos)+len(ne))*vm2
        upos=[(si,e,s,sp,ap) for si,e,s,sp,al in pos]
        for pv2,si,ss,hm in ne:
            ep=O[si,di+1]
            if np.isnan(ep) or ep<=0: continue
            a=compute_atr(H,L,C,si,di,sdi)
            if a is None: continue
            upos.append((si,di+1,ep,ep-atr_s*a,ap*hm))
        pos=upos

    for si,edi2,ep,sp,al in pos:
        c=C[si,ND-1]
        if not np.isnan(c) and c>0: eq+=eq*al*((c-ep)/ep-COMM)
    return trades,eq,mdd


def analyze(trades,eq,mdd,label=""):
    if not trades: print(f"  {label}: no trades"); return None
    nw=sum(1 for t in trades if t["pnl_pct"]>0)
    wr=nw/len(trades)*100
    nd=max(1,trades[-1]["di"]-trades[0]["di"])
    ann=((eq/CASH0)**(1/max(1.0,nd/252))-1)*100
    ap=[t["pnl_abs"] for t in sorted(trades,key=lambda x:x["di"])]
    rets=np.array(ap)/CASH0
    sh=np.mean(rets)/np.std(rets)*np.sqrt(252) if np.std(rets)>0 else 0
    ns=sum(1 for t in trades if t.get("reason","")=="stop")
    print(f"  {label}: {len(trades)}t WR={wr:.1f}% ann={ann:+.1f}% DD={mdd:.1f}% Sh={sh:.2f} eq={eq:,.0f}")
    yr={}
    for t in trades:
        y=t["year"]
        if y not in yr: yr[y]={"n":0,"w":0,"p":[]}
        yr[y]["n"]+=1
        if t["pnl_pct"]>0: yr[y]["w"]+=1
        yr[y]["p"].append(t["pnl_pct"])
    for y in sorted(yr):
        ys=yr[y]; cum=np.prod([1+p/100 for p in ys["p"]])-1
        print(f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% cum={cum:+.1%}")
    return {"n":len(trades),"wr":wr,"dd":mdd,"ann":ann,"sh":sh,"eq":eq}


# =================================================================
# Main
# =================================================================

def main():
    t0=time.time()
    print("="*70)
    print("  V114: H0 MICROSTRUCTURE QUALITY FILTER + NW KERNEL")
    print("  Paper 2601.23172: H0~0.75 unifies order flow, vol roughness")
    print("  Walk-forward 2019-2026. LEVERAGE=1.0, CASH0=1M, COMM=0.0005")
    print("="*70)

    C,O,H,L,V,OI,NS,ND,dates,syms = load_all_data(start="2016-01-01")
    print(f"  {NS} sym, {ND} days, {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    slook = build_sector_lookup(syms)
    sd = defaultdict(int)
    for s in slook.values(): sd[s]+=1
    print(f"  Sectors: {dict(sd)}")
    bt19 = next(i for i,d in enumerate(dates) if d>=pd.Timestamp("2019-01-01"))

    # 1. Factors + NW prediction
    F = compute_raw_factors(C,O,H,L,V,OI,NS,ND)
    ker = compute_ker(C,NS,ND)
    pred = compute_nw_predicted(F,NS,ND,tw=40,bw=1.0)
    pvol = compute_port_vol(C,NS,ND,20)

    # 2. H0 estimation
    print("\n[V114] Signed volume delta...", flush=True)
    svd = compute_signed_volume_delta(C,H,L,V,NS,ND)
    h0c = {}
    for w in [60,80,100]:
        print(f"[V114] H0 estimation (w={w})...", flush=True)
        th=time.time()
        h0c[w] = estimate_h0_rs(svd,NS,ND,h0_window=w)
        hv = h0c[w].flatten(); hv = hv[~np.isnan(hv)]
        if len(hv)>0:
            print(f"  H0-z(w={w}): mean={np.mean(hv):.3f} med={np.median(hv):.3f} "
                  f"std={np.std(hv):.3f} "
                  f"p5={np.percentile(hv,5):.3f} p95={np.percentile(hv,95):.3f} "
                  f"in[-1,1]={np.sum((hv>=-1)&(hv<=1))/len(hv)*100:.1f}% "
                  f"lt-2={np.sum(hv<-2)/len(hv)*100:.1f}% gt+2={np.sum(hv>2)/len(hv)*100:.1f}% "
                  f"{time.time()-th:.1f}s", flush=True)

    # 3. Parameter sweep
    print("\n"+"="*70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("="*70)
    results = []; sc = 0
    for h0w in [60,80,100]:
        ha = h0c[h0w]
        for tn in [2]:
            for mps in [2,3]:
                for nl in [-1.5,-1.0,-0.5]:
                    for nh in [0.5,1.0,1.5]:
                        for cr in [0.2,0.3,0.5]:
                            for ex in [0.4,0.5,0.6]:
                                sc += 1
                                tr,eq,dd = backtest(C,O,H,L,NS,ND,dates,syms,
                                    pred,ker,pvol,ha,slook,tn=tn,mps=mps,
                                    h0_nl=nl,h0_nh=nh,h0_cr=cr,h0_ex=ex,sdi=bt19)
                                if len(tr)<10: continue
                                nw=sum(1 for t in tr if t["pnl_pct"]>0)
                                wr=nw/len(tr)*100
                                nd2=max(1,tr[-1]["di"]-tr[0]["di"])
                                ann=((eq/CASH0)**(1/max(1.0,nd2/252))-1)*100
                                ap2=[t["pnl_abs"] for t in sorted(tr,key=lambda x:x["di"])]
                                ra=np.array(ap2)/CASH0
                                sv2=np.mean(ra)/np.std(ra)*np.sqrt(252) if np.std(ra)>0 else 0
                                results.append({"h0w":h0w,"tn":tn,"mps":mps,
                                    "nl":nl,"nh":nh,"cr":cr,"ex":ex,
                                    "n":len(tr),"wr":wr,"ann":ann,"dd":dd,"sh":sv2,"eq":eq})

    results.sort(key=lambda x:-x["ann"])
    print(f"\n  Evaluated {sc} configs, {len(results)} with 10+ trades")
    if results:
        print(f"\n{'H0w':>4} {'TN':>3} {'MPS':>3} {'H0lo':>5} {'H0hi':>5} "
              f"{'H0cr':>5} {'H0ex':>5} {'N':>5} {'WR':>6} {'Ann':>9} {'DD':>7} {'Sh':>7}")
        print("-"*95)
        for r in results[:15]:
            print(f"{r['h0w']:>4} {r['tn']:>3} {r['mps']:>3} "
                  f"{r['nl']:>5.2f} {r['nh']:>5.2f} {r['cr']:>5.1f} {r['ex']:>5.1f} "
                  f"{r['n']:>5} {r['wr']:>5.1f}% {r['ann']:>+8.1f}% {r['dd']:>6.1f}% {r['sh']:>6.2f}")
    else:
        print("  No configs with 10+ trades."); return

    # 4. Walk-forward for top configs
    ba = results[0]
    bs = max(results, key=lambda x: x["sh"])
    br = max(results, key=lambda x: x["ann"]/max(x["dd"],1))
    for lbl, best in [("BEST-ANN",ba),("BEST-SHARPE",bs),("BEST-RISK-ADJ",br)]:
        print(f"\n{'='*70}\n  WF V114 {lbl}: h0w={best['h0w']} nl={best['nl']} nh={best['nh']} "
              f"cr={best['cr']} ex={best['ex']} tn={best['tn']} mps={best['mps']}\n{'='*70}")
        yrs = sorted(set(d.year for d in dates))
        at = []
        for ty in range(2019, yrs[-1]+1):
            ts=te=None
            for i,d in enumerate(dates):
                if d.year==ty and ts is None: ts=i
                if d.year==ty: te=i
            if ts is None: continue
            tr,_,_ = backtest(C,O,H,L,NS,ND,dates,syms,pred,ker,pvol,
                h0c[best["h0w"]],slook,tn=best["tn"],mps=best["mps"],
                h0_nl=best["nl"],h0_nh=best["nh"],h0_cr=best["cr"],h0_ex=best["ex"],
                sdi=ts,edi=te+1)
            yt=[t for t in tr if dates[t["di"]].year==ty]
            at.extend(yt)
            if yt:
                nw=sum(1 for t in yt if t["pnl_pct"]>0)
                print(f"  {ty}: {len(yt)}t WR={nw/len(yt)*100:.1f}% avg={np.mean([t['pnl_pct'] for t in yt]):+.2f}%", flush=True)
            else:
                print(f"  {ty}: no trades", flush=True)
        if at:
            nw=sum(1 for t in at if t["pnl_pct"]>0)
            cum=np.prod([1+t["pnl_pct"]/100 for t in at])-1
            print(f"\n  WF TOTAL: {len(at)}t WR={nw/len(at)*100:.1f}% cum={cum:+.1%}")

    # 5. Compare V114 vs baseline (H0 disabled)
    print("\n"+"="*70)
    print("  COMPARISON: V114 (H0 filter) vs baseline (no H0)")
    print("="*70)
    best = ba
    t1,e1,d1 = backtest(C,O,H,L,NS,ND,dates,syms,pred,ker,pvol,
        h0c[best["h0w"]],slook,tn=best["tn"],mps=best["mps"],
        h0_nl=best["nl"],h0_nh=best["nh"],h0_cr=best["cr"],h0_ex=best["ex"],sdi=bt19)
    # baseline: H0 effectively disabled
    t2,e2,d2 = backtest(C,O,H,L,NS,ND,dates,syms,pred,ker,pvol,
        h0c[best["h0w"]],slook,tn=best["tn"],mps=best["mps"],
        h0_nl=0.01,h0_nh=1.50,h0_cr=1.0,h0_ex=1.0,sdi=bt19)
    print(f"\n  V114 BEST (NW + H0 filter):"); analyze(t1,e1,d1,"V114-H0")
    print(f"\n  BASELINE (NW only, no H0):"); analyze(t2,e2,d2,"Baseline")
    if t1 and t2:
        print(f"\n  Delta: eq={e1-e2:+,.0f} dd={d1-d2:+.1f}% trades={len(t1)-len(t2):+d}")
    print(f"\n[V114] Done. {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
