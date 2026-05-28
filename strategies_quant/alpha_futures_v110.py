"""
V110: Adaptive Multi-Signal Integration (NW Kernel + Linear Rank)
=================================================================
Combine NW kernel (stable regimes) + linear rank (regime transitions)
with adaptive rolling-Sharpe-based weighting and min weight floor.

Walk-forward 2019-2026. No leverage. CASH0=1M, COMM=0.0005.
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
    import talib; HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

CASH0 = 1_000_000; COMM = 0.0005; LEVERAGE = 1.0

SECTOR_MAP = {
    'i':'BLACK','j':'BLACK','jm':'BLACK','hc':'BLACK','sf':'BLACK',
    'sm':'BLACK','wr':'BLACK','im':'BLACK','cu':'METAL','al':'METAL',
    'zn':'METAL','pb':'METAL','ni':'METAL','sn':'METAL','ss':'METAL',
    'ao':'METAL','au':'METAL','ag':'METAL','rb':'METAL','si':'METAL',
    'sc':'ENERGY','fu':'ENERGY','bu':'ENERGY','pg':'ENERGY',
    'eb':'ENERGY','ta':'ENERGY','fg':'ENERGY','oi':'ENERGY',
    'v':'CHEMICAL','pp':'CHEMICAL','l':'CHEMICAL','eg':'CHEMICAL',
    'ma':'CHEMICAL','sa':'CHEMICAL','ur':'CHEMICAL','pf':'CHEMICAL',
    'sh':'CHEMICAL','lc':'CHEMICAL','m':'AGRI','y':'AGRI','a':'AGRI',
    'p':'AGRI','c':'AGRI','cs':'AGRI','jd':'AGRI','rr':'AGRI',
    'lrm':'AGRI','rm':'AGRI','ru':'AGRI','cf':'SOFTS','sr':'SOFTS',
    'ap':'SOFTS','cj':'SOFTS','pk':'SOFTS','lh':'SOFTS','sp':'SOFTS',
    'b':'SOFTS','br':'SOFTS',
}

FACTOR_NAMES = ["ret_5d","oi_5d","rsi14","vol_5d","ret_10d","range_5d","atrp_5d"]
N_FACTORS = len(FACTOR_NAMES)
LINEAR_RANK_WEIGHTS = [0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05]


def _base(sym: str) -> str:
    s = sym.lower().split('.')[0].strip()
    while s and s[-1].isdigit(): s = s[:-1]
    return s[:-2] if s.endswith('fi') else s


def build_sector_lookup(syms: List[str]) -> Dict[int, str]:
    return {i: SECTOR_MAP.get(_base(s), 'OTHER') for i, s in enumerate(syms)}


def compute_rsi_manual(C: np.ndarray, NS: int, ND: int, period: int = 14) -> np.ndarray:
    rsi = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]; gains = np.full(ND, np.nan); losses = np.full(ND, np.nan)
        for di in range(1, ND):
            if np.isnan(c[di]) or np.isnan(c[di-1]): continue
            d = c[di] - c[di-1]; gains[di] = max(d,0.0); losses[di] = max(-d,0.0)
        ag = al = np.nan
        for di in range(1, ND):
            if np.isnan(gains[di]): continue
            if np.isnan(ag):
                vg = [gains[j] for j in range(di, min(di+period,ND)) if not np.isnan(gains[j])]
                vl = [losses[j] if not np.isnan(losses[j]) else 0.0 for j in range(di, min(di+period,ND)) if not np.isnan(gains[j])]
                if len(vg) >= period:
                    ag, al = np.mean(vg), np.mean(vl)
                    rsi[si, di+period-1] = 100.0 if al == 0 else 100.0-100.0/(1.0+ag/al)
                continue
            ag = (ag*(period-1)+gains[di])/period; al = (al*(period-1)+losses[di])/period
            rsi[si, di] = 100.0 if al == 0 else 100.0-100.0/(1.0+ag/al)
    return rsi


def compute_raw_factors(C,O,H,L,V,OI, NS,ND):
    """Compute 7 raw factors + fwd_ret_5d + atr_mean."""
    t0 = time.time(); print("[V110] Computing raw factors...", flush=True)
    ret_5d = np.full((NS,ND),np.nan)
    oi_5d = np.full((NS,ND),np.nan)
    vol_5d = np.full((NS,ND),np.nan)
    range_5d = np.full((NS,ND),np.nan)
    atrp_5d = np.full((NS,ND),np.nan)
    ret_10d = np.full((NS,ND),np.nan)
    for si in range(NS):
        for di in range(5,ND):
            if not np.isnan(C[si,di]) and not np.isnan(C[si,di-5]) and C[si,di-5]>0:
                ret_5d[si,di] = C[si,di]/C[si,di-5]-1.0
            if not np.isnan(OI[si,di]) and not np.isnan(OI[si,di-5]) and OI[si,di-5]>0:
                oi_5d[si,di] = OI[si,di]/OI[si,di-5]-1.0
            vals = V[si,di-5:di]; valid = vals[~np.isnan(vals)]
            if len(valid)>=3: vol_5d[si,di] = np.mean(valid)
            rng = []
            for j in range(di-5,di):
                if not np.isnan(H[si,j]) and not np.isnan(L[si,j]) and not np.isnan(C[si,j]) and C[si,j]>0 and H[si,j]>L[si,j]:
                    rng.append((H[si,j]-L[si,j])/C[si,j])
            if len(rng)>=3: range_5d[si,di] = np.mean(rng)
            if di>=6:
                av=[]
                for j in range(di-5,di):
                    hh,ll,cc = H[si,j],L[si,j],C[si,j]
                    if not any(np.isnan([hh,ll,cc])):
                        pc = C[si,j-1] if j>0 and not np.isnan(C[si,j-1]) else cc
                        av.append(max(hh-ll,abs(hh-pc),abs(ll-pc)))
                if av and not np.isnan(C[si,di]) and C[si,di]>0:
                    atrp_5d[si,di] = np.mean(av)/C[si,di]
        for di in range(10,ND):
            if not np.isnan(C[si,di]) and not np.isnan(C[si,di-10]) and C[si,di-10]>0:
                ret_10d[si,di] = C[si,di]/C[si,di-10]-1.0

    rsi14 = np.full((NS,ND),np.nan)
    if HAS_TALIB:
        for si in range(NS):
            c = np.where(np.isnan(C[si]),0,C[si]).astype(np.float64)
            nm = np.isnan(C[si])
            try: rsi14[si] = np.where(nm,np.nan,talib.RSI(c,14))
            except: pass
    fallback = np.all(np.isnan(rsi14),axis=1)
    if fallback.any():
        rm = compute_rsi_manual(C,NS,ND,14)
        for si in range(NS):
            if fallback[si]: rsi14[si] = rm[si]

    fwd = np.full((NS,ND),np.nan)
    for si in range(NS):
        for di in range(ND-5):
            if not np.isnan(C[si,di+5]) and not np.isnan(C[si,di]) and C[si,di]>0:
                fwd[si,di] = C[si,di+5]/C[si,di]-1.0

    atr_m = np.full((NS,ND),np.nan)
    for si in range(NS):
        for di in range(20,ND):
            av=[]
            for j in range(di-14,di):
                hh,ll,cc = H[si,j],L[si,j],C[si,j]
                if not any(np.isnan([hh,ll,cc])):
                    pc = C[si,j-1] if j>0 and not np.isnan(C[si,j-1]) else cc
                    av.append(max(hh-ll,abs(hh-pc),abs(ll-pc)))
            if av and not np.isnan(C[si,di]) and C[si,di]>0:
                atr_m[si,di] = np.mean(av)/C[si,di]

    print(f"  Raw factors done: {time.time()-t0:.1f}s", flush=True)
    return {"ret_5d":ret_5d,"oi_5d":oi_5d,"vol_5d":vol_5d,"range_5d":range_5d,
            "atrp_5d":atrp_5d,"ret_10d":ret_10d,"rsi14":rsi14,"fwd_ret_5d":fwd,"atr_mean":atr_m}


def normalize_factor(f, NS, ND, min_count=10):
    normed = np.full((NS,ND),np.nan)
    for di in range(ND):
        vals = f[:,di]; valid = vals[~np.isnan(vals)]
        if len(valid)<min_count: continue
        mu,sigma = np.mean(valid),np.std(valid)
        if sigma<1e-12: continue
        for si in range(NS):
            if not np.isnan(vals[si]): normed[si,di] = (vals[si]-mu)/sigma
    return normed


# --- SIGNAL A: NW Kernel ---
def compute_nw_signal(raw, NS, ND, tw=40, bw=1.0):
    t0 = time.time()
    print(f"[V110] NW kernel (tw={tw}, bw={bw:.1f})...", flush=True)
    normed = {fn: normalize_factor(raw[fn], NS, ND) for fn in FACTOR_NAMES}
    fwd = raw["fwd_ret_5d"]; atr_m = raw["atr_mean"]
    pred = np.full((NS,ND),np.nan)
    MIN_TRAIN = 20
    for di in range(tw+10, ND):
        tX,tY = [],[]
        for tdi in range(max(10,di-tw), di):
            for si in range(NS):
                feat = np.array([normed[fn][si,tdi] for fn in FACTOR_NAMES])
                tgt = fwd[si,tdi]
                if np.any(np.isnan(feat)) or np.isnan(tgt): continue
                tX.append(feat); tY.append(tgt)
        if len(tX)<MIN_TRAIN: continue
        tXa = np.array(tX); tYa = np.array(tY)
        fstd = np.std(tXa,axis=0); fstd[fstd<1e-12] = 1.0
        for si in range(NS):
            qf = np.array([normed[fn][si,di] for fn in FACTOR_NAMES])
            if np.any(np.isnan(qf)): continue
            atr = atr_m[si,di]; h = max(bw if np.isnan(atr) else atr*bw, 0.1)
            dist = np.sqrt(np.sum(((tXa-qf)/fstd)**2, axis=1))
            sd = dist/h; w = np.zeros(len(tXa))
            m = sd<=1.0
            if not np.any(m):
                mi = np.argmin(dist)
                if dist[mi]<1e12: w[mi]=1.0; m=np.zeros(len(dist),bool); m[mi]=True
                else: continue
            else: w[m] = 0.75*(1.0-sd[m]**2)
            ws = np.sum(w)
            if ws<1e-12: continue
            pred[si,di] = np.sum(w*tYa)/ws
        if di%100==0: print(f"  NW di={di}/{ND}", flush=True)
    print(f"  NW done: {time.time()-t0:.1f}s", flush=True)
    return pred


# --- SIGNAL B: Linear Rank Composite ---
def compute_linear_rank_signal(raw, NS, ND):
    t0 = time.time(); print("[V110] Linear rank composite...", flush=True)
    signal = np.zeros((NS,ND))
    for fi,fn in enumerate(FACTOR_NAMES):
        f = raw[fn]
        for di in range(ND):
            col = f[:,di]; vm = ~np.isnan(col); vv = col[vm]
            if len(vv)<5: continue
            order = np.argsort(vv); ranks = np.empty_like(order,dtype=float)
            ranks[order] = np.arange(1,len(vv)+1)/len(vv)
            result_col = np.full(NS,np.nan); result_col[vm] = ranks
            for si in range(NS):
                if not np.isnan(result_col[si]): signal[si,di] += LINEAR_RANK_WEIGHTS[fi]*result_col[si]
    print(f"  Linear rank done: {time.time()-t0:.1f}s", flush=True)
    return signal


# --- ADAPTIVE COMBINATION ---
def compute_adaptive_signal(nw_sig, lin_sig, raw, C, NS, ND, adapt_w=60, min_w=0.3):
    t0 = time.time()
    print(f"[V110] Adaptive combine (aw={adapt_w}, min_w={min_w:.1f})...", flush=True)
    fwd = raw["fwd_ret_5d"]; TOP=5
    nw_dr = np.full(ND,np.nan); lin_dr = np.full(ND,np.nan)
    for di in range(20,ND-1):
        for sig, arr in [(nw_sig,nw_dr),(lin_sig,lin_dr)]:
            scores = sig[:,di]; valid = ~np.isnan(scores) & ~np.isnan(fwd[:,di])
            cands = [(scores[si],si) for si in range(NS) if valid[si]]
            cands.sort(key=lambda x:-x[0])
            r = [fwd[si,di] for _,si in cands[:TOP]]
            if r: arr[di] = np.mean(r)

    def rolling_sh(arr, w):
        rs = np.full(ND,np.nan)
        for di in range(w,ND):
            v = arr[di-w:di]; v = v[~np.isnan(v)]
            if len(v)>=10: s=np.std(v); rs[di]=np.mean(v)/s*np.sqrt(252) if s>1e-12 else 0
        return rs

    nw_sh = rolling_sh(nw_dr, adapt_w); lin_sh = rolling_sh(lin_dr, adapt_w)
    combined = np.full((NS,ND),np.nan); nw_wins=0; lin_wins=0
    for di in range(ND):
        ns,ls = nw_sh[di],lin_sh[di]
        if np.isnan(ns) and np.isnan(ls): nww=0.5
        elif np.isnan(ns): nww=min_w
        elif np.isnan(ls): nww=1.0-min_w
        else:
            tot = abs(ns)+abs(ls); nww = abs(ns)/tot if tot>1e-12 else 0.5
            nww = max(min_w, min(1.0-min_w, nww))
        lw = 1.0-nww
        if nww>lw: nw_wins+=1
        elif lw>nww: lin_wins+=1
        for si in range(NS):
            nv,lv = nw_sig[si,di], lin_sig[si,di]*0.05
            if np.isnan(nv) and np.isnan(lv): continue
            elif np.isnan(nv): combined[si,di]=lv
            elif np.isnan(lv): combined[si,di]=nv
            else: combined[si,di] = nww*nv + lw*lv
    print(f"  NW dom={nw_wins}d Lin dom={lin_wins}d of {ND} | {time.time()-t0:.1f}s", flush=True)
    return combined


# --- Helpers ---
def compute_ker(C, NS, ND):
    ker10 = np.full((NS,ND),np.nan)
    for si in range(NS):
        for di in range(10,ND):
            cs = C[si,di-10:di+1]; v = cs[~np.isnan(cs)]
            if len(v)<10 or v[0]<=0: continue
            tc = np.sum(np.abs(np.diff(v)))
            if tc>1e-10: ker10[si,di] = abs(v[-1]-v[0])/tc
    kr = np.zeros((NS,ND),dtype=int)
    for si in range(NS):
        for di in range(ND):
            if np.isnan(ker10[si,di]): continue
            if ker10[si,di]<0.15: kr[si,di]=1
            elif ker10[si,di]>0.3: kr[si,di]=-1
    return kr


def get_mode(wins, wt, wvw):
    if len(wins)<5: return "normal"
    wr = sum(wins[-wvw:])/len(wins[-wvw:])
    return "winning" if wr>wt else ("losing" if wr<0.50 else "normal")


def atr_at(H,L,C, si,di,start):
    av=[]
    for j in range(max(start,di-14),di):
        hh,ll,cc = H[si,j],L[si,j],C[si,j]
        if not any(np.isnan([hh,ll,cc])): av.append(max(hh-ll,abs(hh-cc),abs(ll-cc)))
    return np.mean(av) if av else None


# --- Backtest ---
def backtest(C,O,H,L, NS,ND,dates,syms, signal, ker, sec_lk, top_n=2, mps=2, hd=5, wt=0.60, wvw=15, as_mult=3.0, sdi=60, edi=None):
    if edi is None: edi = ND-1
    eq,peak,mdd = CASH0,CASH0,0.0
    pos=[]; trades=[]; rwins=[]
    for di in range(max(sdi,1), edi):
        d = dates[di]; dpnl = 0.0; npos = []
        mode = get_mode(rwins, wt, wvw)
        pbs = defaultdict(list)
        for p in pos: pbs[p[0]].append(p)
        for si,pl in pbs.items():
            c = C[si,di]
            if np.isnan(c): npos.extend(pl); continue
            ee = min(p[0] for p in pl); hold = di-ee
            stopped = any(c<p[2] for p in pl)
            if stopped or hold>=hd:
                for edi2,ep,sp,al in pl:
                    pnl = (c-ep)/ep-COMM; pr = eq*al*pnl; dpnl+=pr
                    trades.append({"pnl_abs":pr,"pnl_pct":pnl*100,"days":di-edi2+1,"di":di,"year":d.year,"sym":syms[si],"sector":sec_lk.get(si,'OTHER'),"reason":"stop" if stopped else "hold","mode":mode[0].upper()})
                    rwins.append(1 if pnl>0 else 0)
            else: npos.extend(pl)
        pos=npos; eq+=dpnl
        if eq>peak: peak=eq
        if peak>0: dd=(peak-eq)/peak*100; mdd=max(mdd,dd)
        if eq<=0: break
        held={p[0] for p in pos}
        if len(held)>=top_n: continue
        cands=[]
        for si in range(NS):
            if si in held: continue
            s = signal[si,di]
            if np.isnan(s) or di+1>=ND or np.isnan(O[si,di+1]) or ker[si,di]<0: continue
            cands.append((s,si))
        if not cands: continue
        cands.sort(key=lambda x:-x[0])
        nt = top_n
        if mode=="winning": nt=min(top_n+1,top_n*2)
        elif mode=="losing": nt=max(1,top_n-1)
        sc = defaultdict(int)
        for sh in held: sc[sec_lk.get(sh,'OTHER')]+=1
        ne=[]
        for sv,si in cands:
            if len(held)+len(ne)>=nt or si in held: break
            s = sec_lk.get(si,'OTHER')
            if sc[s]>=mps or sv<=0: continue
            ne.append((sv,si,s)); sc[s]+=1
        if not ne: continue
        ap = LEVERAGE/(len(pos)+len(ne))
        up = [(si,edi2,ep,sp,ap) for si,edi2,ep,sp,_ in pos]
        for sv,si,s in ne:
            ep = O[si,di+1]
            if np.isnan(ep) or ep<=0: continue
            a = atr_at(H,L,C,si,di,sdi)
            if a is None: continue
            up.append((si,di+1,ep,ep-as_mult*a,ap))
        pos = up
    for si,edi2,ep,sp,al in pos:
        c = C[si,ND-1]
        if not np.isnan(c) and c>0: eq += eq*al*((c-ep)/ep-COMM)
    return trades, eq, mdd


def analyze(trades, eq, mdd, label=""):
    if not trades: print(f"  {label}: no trades"); return None
    nw = sum(1 for t in trades if t["pnl_pct"]>0)
    wr = nw/len(trades)*100; nd = max(1,trades[-1]["di"]-trades[0]["di"])
    ann = ((eq/CASH0)**(1/max(1.0,nd/252))-1)*100
    ap = [t["pnl_abs"] for t in sorted(trades,key=lambda x:x["di"])]
    r = np.array(ap)/CASH0; sh = np.mean(r)/np.std(r)*np.sqrt(252) if np.std(r)>0 else 0
    ns = sum(1 for t in trades if t["reason"]=="stop"); nh = len(trades)-ns
    mc = {"W":0,"N":0,"L":0}
    for t in trades: m=t.get("mode","N"); mc[m]=mc.get(m,0)+1
    sc = defaultdict(int)
    for t in trades: sc[t.get("sector","OTHER")]+=1
    ss = " ".join(f"{k}:{v}" for k,v in sorted(sc.items()))
    print(f"  {label}: {len(trades)}t (stop:{ns} hold:{nh}) WR={wr:.1f}% ann={ann:+.1f}% DD={mdd:.1f}% Sh={sh:.2f} eq={eq:,.0f}")
    print(f"    modes=[W:{mc['W']} N:{mc['N']} L:{mc['L']}] sectors: {ss}")
    yr = {}
    for t in trades:
        y=t["year"]
        if y not in yr: yr[y]={"n":0,"w":0,"pnl":[]}
        yr[y]["n"]+=1; yr[y]["w"]+= t["pnl_pct"]>0; yr[y]["pnl"].append(t["pnl_pct"])
    for y in sorted(yr):
        ys=yr[y]; cum=np.prod([1+p/100 for p in ys["pnl"]])-1
        print(f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% cum={cum:+.1%}")
    return {"n":len(trades),"wr":wr,"dd":mdd,"ann":ann,"sh":sh,"eq":eq}


def walk_forward(C,O,H,L, NS,ND,dates,syms, signal, ker, sec_lk, top_n=2, mps=2, hd=5, label=""):
    print(f"\n{'='*70}\n  WF V110 {label} tn={top_n} mps={mps}\n{'='*70}")
    years = sorted(set(d.year for d in dates)); at = []
    for ty in range(2019, years[-1]+1):
        ts=te=None
        for i,d in enumerate(dates):
            if d.year==ty and ts is None: ts=i
            if d.year==ty: te=i
        if ts is None: continue
        tr,_,_ = backtest(C,O,H,L,NS,ND,dates,syms,signal,ker,sec_lk,top_n=top_n,mps=mps,hd=hd,sdi=ts,edi=te+1)
        tt = [t for t in tr if dates[t["di"]].year==ty]; at.extend(tt)
        if tt:
            n=len(tt); nw=sum(1 for t in tt if t["pnl_pct"]>0)
            sc=defaultdict(int)
            for t in tt: sc[t.get("sector","OTHER")]+=1
            ss=" ".join(f"{k}:{v}" for k,v in sorted(sc.items()))
            print(f"  {ty}: {n}t WR={nw/n*100:.1f}% avg={np.mean([t['pnl_pct'] for t in tt]):+.2f}% [{ss}]",flush=True)
        else: print(f"  {ty}: no trades",flush=True)
    if at:
        nw=sum(1 for t in at if t["pnl_pct"]>0)
        cum=np.prod([1+t["pnl_pct"]/100 for t in at])-1
        sc=defaultdict(int)
        for t in at: sc[t.get("sector","OTHER")]+=1
        ss=" ".join(f"{k}:{v}" for k,v in sorted(sc.items()))
        print(f"\n  WF TOTAL: {len(at)}t WR={nw/len(at)*100:.1f}% cum={cum:+.1%} sectors: {ss}")
        return at
    return []


def main():
    t0 = time.time()
    print("="*70)
    print("  V110: ADAPTIVE MULTI-SIGNAL INTEGRATION (NW + Linear Rank)")
    print("  Walk-forward 2019-2026. No leverage.")
    print("="*70)

    C,O,H,L,V,OI,NS,ND,dates,syms = load_all_data(start="2016-01-01")
    print(f"  {NS} sym, {ND} days, {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    sec_lk = build_sector_lookup(syms)
    sd = defaultdict(int)
    for s in sec_lk.values(): sd[s]+=1
    print(f"  Sectors: {dict(sd)}")

    bt19 = None
    for i,d in enumerate(dates):
        if d>=pd.Timestamp("2019-01-01"): bt19=i; break

    # 1. Raw factors + regime
    raw = compute_raw_factors(C,O,H,L,V,OI,NS,ND)
    ker = compute_ker(C,NS,ND)

    # 2. Signal A: NW kernel
    nw_sig = compute_nw_signal(raw,NS,ND,tw=40,bw=1.0)

    # 3. Signal B: Linear rank
    lin_sig = compute_linear_rank_signal(raw,NS,ND)

    # 4. Adaptive combinations
    adapt_ws = [30,60,90]; min_ws = [0.2,0.3,0.4]
    comb_cache = {}
    for aw in adapt_ws:
        for mw in min_ws:
            comb_cache[(aw,mw)] = compute_adaptive_signal(nw_sig,lin_sig,raw,C,NS,ND,adapt_w=aw,min_w=mw)

    # 5. Parameter sweep
    print(f"\n{'='*70}\n  PARAMETER SWEEP (2019-2026)\n{'='*70}")
    results = []; sc = 0
    for aw in adapt_ws:
        for mw in min_ws:
            sig = comb_cache[(aw,mw)]
            for tn in [2,3]:
                for mps in [2,3]:
                    sc+=1
                    tr,eq,dd = backtest(C,O,H,L,NS,ND,dates,syms,sig,ker,sec_lk,top_n=tn,mps=mps,sdi=bt19)
                    if len(tr)<10: continue
                    nw=sum(1 for t in tr if t["pnl_pct"]>0); wr=nw/len(tr)*100
                    nd=max(1,tr[-1]["di"]-tr[0]["di"])
                    ann=((eq/CASH0)**(1/max(1.0,nd/252))-1)*100
                    ap=[t["pnl_abs"] for t in sorted(tr,key=lambda x:x["di"])]
                    ra=np.array(ap)/CASH0
                    sh=np.mean(ra)/np.std(ra)*np.sqrt(252) if np.std(ra)>0 else 0
                    results.append({"aw":aw,"mw":mw,"tn":tn,"mps":mps,"n":len(tr),"wr":wr,"ann":ann,"dd":dd,"sh":sh,"eq":eq})

    # NW-only baseline
    for tn in [2,3]:
        for mps in [2,3]:
            sc+=1
            tr,eq,dd = backtest(C,O,H,L,NS,ND,dates,syms,nw_sig,ker,sec_lk,top_n=tn,mps=mps,sdi=bt19)
            if len(tr)<10: continue
            nw=sum(1 for t in tr if t["pnl_pct"]>0); wr=nw/len(tr)*100
            nd=max(1,tr[-1]["di"]-tr[0]["di"])
            ann=((eq/CASH0)**(1/max(1.0,nd/252))-1)*100
            ap=[t["pnl_abs"] for t in sorted(tr,key=lambda x:x["di"])]
            ra=np.array(ap)/CASH0
            sh=np.mean(ra)/np.std(ra)*np.sqrt(252) if np.std(ra)>0 else 0
            results.append({"aw":0,"mw":0,"tn":tn,"mps":mps,"n":len(tr),"wr":wr,"ann":ann,"dd":dd,"sh":sh,"eq":eq})

    # Linear-only baseline
    lin_scaled = lin_sig * 0.05
    for tn in [2,3]:
        for mps in [2,3]:
            sc+=1
            tr,eq,dd = backtest(C,O,H,L,NS,ND,dates,syms,lin_scaled,ker,sec_lk,top_n=tn,mps=mps,sdi=bt19)
            if len(tr)<10: continue
            nw=sum(1 for t in tr if t["pnl_pct"]>0); wr=nw/len(tr)*100
            nd=max(1,tr[-1]["di"]-tr[0]["di"])
            ann=((eq/CASH0)**(1/max(1.0,nd/252))-1)*100
            ap=[t["pnl_abs"] for t in sorted(tr,key=lambda x:x["di"])]
            ra=np.array(ap)/CASH0
            sh=np.mean(ra)/np.std(ra)*np.sqrt(252) if np.std(ra)>0 else 0
            results.append({"aw":-1,"mw":0,"tn":tn,"mps":mps,"n":len(tr),"wr":wr,"ann":ann,"dd":dd,"sh":sh,"eq":eq})

    results.sort(key=lambda x:-x["ann"])
    print(f"\n  {sc} configs evaluated, {len(results)} with 10+ trades")
    print(f"\n{'AW':>4} {'MW':>4} {'TN':>3} {'MPS':>3} {'N':>5} {'WR':>6} {'Ann':>9} {'DD':>7} {'Sh':>7}")
    print("-"*60)
    for r in results[:15]:
        tag = "NW" if r["aw"]==0 else ("LIN" if r["aw"]==-1 else str(r["aw"]))
        print(f"{tag:>4} {r['mw']:>4.1f} {r['tn']:>3} {r['mps']:>3} {r['n']:>5} {r['wr']:>5.1f}% {r['ann']:>+8.1f}% {r['dd']:>6.1f}% {r['sh']:>6.2f}")

    if not results: print("  No configs with 10+ trades."); return

    # 6. Walk-forward for best configs
    best_sh = max(results, key=lambda x:x["sh"])
    best_ann = results[0]
    best_ra = max(results, key=lambda x:x["ann"]/max(x["dd"],1.0))

    for label, best in [("BEST-ANN",best_ann),("BEST-SHARPE",best_sh),("BEST-RISK-ADJ",best_ra)]:
        if best["aw"]==0: sig=nw_sig
        elif best["aw"]==-1: sig=lin_scaled
        else: sig=comb_cache[(best["aw"],best["mw"])]
        walk_forward(C,O,H,L,NS,ND,dates,syms,sig,ker,sec_lk,top_n=best["tn"],mps=best["mps"],label=label)

    # 7. Final comparison
    print(f"\n{'='*70}\n  COMPARISON: V110 (adaptive) vs NW-only vs Linear-only\n{'='*70}")
    def get_sig(b):
        if b["aw"]==0: return nw_sig
        if b["aw"]==-1: return lin_scaled
        return comb_cache[(b["aw"],b["mw"])]

    tr110,eq110,dd110 = backtest(C,O,H,L,NS,ND,dates,syms,get_sig(best_ann),ker,sec_lk,top_n=best_ann["tn"],mps=best_ann["mps"],sdi=bt19)
    trnw,eqnw,ddnw = backtest(C,O,H,L,NS,ND,dates,syms,nw_sig,ker,sec_lk,top_n=2,mps=2,sdi=bt19)

    print(f"\n  V110 BEST-ANN:"); analyze(tr110,eq110,dd110,"V110-adaptive")
    print(f"\n  NW-ONLY (V96 baseline):"); analyze(trnw,eqnw,ddnw,"NW-only")
    if tr110 and trnw:
        print(f"\n  Delta: eq={eq110-eqnw:+,.0f} dd={dd110-ddnw:+.1f}% trades={len(tr110)-len(trnw):+d}")

    print(f"\n[V110] Done. {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
