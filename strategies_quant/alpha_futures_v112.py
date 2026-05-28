"""V112: Spectral Risk + Kalman Adaptive Sizing
Two innovations over V96's vol-adaptive sizing:
1. Spectral Risk: risk = sum(w_i * |quantile_i|) w/ w_i=-ln(p_i)
   Exponentially weights tail losses. Fat left tail => risk >> vol.
2. Kalman Filter Volatility: adapts faster than fixed ATR.
   State: vol estimate, observation: |daily_ret|.
Walk-forward 2019-2026. LEVERAGE=1.0, CASH0=1M, COMM=0.0005."""
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

CASH0, COMM, LEVERAGE = 1_000_000, 0.0005, 1.0
SECTOR_MAP = {
    'i':'BLACK','j':'BLACK','jm':'BLACK','hc':'BLACK','sf':'BLACK',
    'sm':'BLACK','wr':'BLACK','im':'BLACK','cu':'METAL','al':'METAL',
    'zn':'METAL','pb':'METAL','ni':'METAL','sn':'METAL','ss':'METAL',
    'ao':'METAL','au':'METAL','ag':'METAL','rb':'METAL','si':'METAL',
    'sc':'ENERGY','fu':'ENERGY','bu':'ENERGY','pg':'ENERGY','eb':'ENERGY',
    'ta':'ENERGY','fg':'ENERGY','oi':'ENERGY','v':'CHEMICAL','pp':'CHEMICAL',
    'l':'CHEMICAL','eg':'CHEMICAL','ma':'CHEMICAL','sa':'CHEMICAL',
    'ur':'CHEMICAL','pf':'CHEMICAL','sh':'CHEMICAL','lc':'CHEMICAL',
    'm':'AGRI','y':'AGRI','a':'AGRI','p':'AGRI','c':'AGRI','cs':'AGRI',
    'jd':'AGRI','rr':'AGRI','lrm':'AGRI','rm':'AGRI','ru':'AGRI',
    'cf':'SOFTS','sr':'SOFTS','ap':'SOFTS','cj':'SOFTS','pk':'SOFTS',
    'lh':'SOFTS','sp':'SOFTS','b':'SOFTS','br':'SOFTS',
}
FACTOR_NAMES = ["ret_5d","oi_5d","rsi14","vol_5d","ret_10d","range_5d","atrp_5d"]
N_FACTORS = len(FACTOR_NAMES)

def _base_sym(sym):
    s = sym.lower().split('.')[0].strip()
    while s and s[-1].isdigit(): s = s[:-1]
    return s[:-2] if s.endswith('fi') else s

def build_sector_lookup(syms):
    return {i: SECTOR_MAP.get(_base_sym(s), 'OTHER') for i, s in enumerate(syms)}

def _safe_pct(arr, lag, NS, ND):
    """Compute arr[i]/arr[i-lag]-1 safely."""
    out = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(lag, ND):
            if not np.isnan(arr[si,di]) and not np.isnan(arr[si,di-lag]) and arr[si,di-lag]>0:
                out[si,di] = arr[si,di]/arr[si,di-lag]-1.0
    return out

def compute_rsi_manual(C, NS, ND, period=14):
    rsi = np.full((NS,ND), np.nan)
    for si in range(NS):
        c = C[si]; gains = np.full(ND,np.nan); losses = np.full(ND,np.nan)
        for di in range(1,ND):
            if np.isnan(c[di]) or np.isnan(c[di-1]): continue
            d = c[di]-c[di-1]; gains[di]=max(d,0); losses[di]=max(-d,0)
        ag = al = np.nan
        for di in range(1,ND):
            if np.isnan(gains[di]): continue
            if np.isnan(ag):
                vg=[gains[j] for j in range(di,min(di+period,ND)) if not np.isnan(gains[j])]
                vl=[losses[j] if not np.isnan(losses[j]) else 0 for j in range(di,min(di+period,ND)) if not np.isnan(gains[j])]
                if len(vg)>=period:
                    ag,al=np.mean(vg),np.mean(vl)
                    rsi[si,di+period-1]=100 if al==0 else 100-100/(1+ag/al)
                continue
            ag=(ag*(period-1)+gains[di])/period; al=(al*(period-1)+losses[di])/period
            rsi[si,di]=100 if al==0 else 100-100/(1+ag/al)
    return rsi

def compute_atr_vec(H, L, C, NS, ND, window=14):
    """Compute ATR as fraction of close for all instruments."""
    atrp = np.full((NS,ND), np.nan)
    for si in range(NS):
        for di in range(window+1, ND):
            vals=[]
            for j in range(di-window, di):
                hh,ll,cc = H[si,j],L[si,j],C[si,j]
                if np.isnan(hh) or np.isnan(ll) or np.isnan(cc): continue
                pc = C[si,j-1] if j>0 and not np.isnan(C[si,j-1]) else cc
                vals.append(max(hh-ll, abs(hh-pc), abs(ll-pc)))
            if vals and not np.isnan(C[si,di]) and C[si,di]>0:
                atrp[si,di] = np.mean(vals)/C[si,di]
    return atrp

def compute_raw_factors(C, O, H, L, V, OI, NS, ND):
    t0=time.time(); print("[V112] Computing raw factors...", flush=True)
    ret_5d = _safe_pct(C, 5, NS, ND)
    oi_5d = _safe_pct(OI, 5, NS, ND)
    vol_5d = np.full((NS,ND),np.nan)
    range_5d = np.full((NS,ND),np.nan)
    for si in range(NS):
        for di in range(5,ND):
            vv=V[si,di-5:di]; vld=vv[~np.isnan(vv)]
            if len(vld)>=3: vol_5d[si,di]=np.mean(vld)
            rv=[(H[si,j]-L[si,j])/C[si,j] for j in range(di-5,di)
                if not np.isnan(H[si,j]) and not np.isnan(L[si,j])
                and not np.isnan(C[si,j]) and C[si,j]>0 and H[si,j]>L[si,j]]
            if len(rv)>=3: range_5d[si,di]=np.mean(rv)
    atrp_5d = compute_atr_vec(H, L, C, NS, ND, 5)
    ret_10d = _safe_pct(C, 10, NS, ND)
    rsi14 = np.full((NS,ND),np.nan)
    if HAS_TALIB:
        for si in range(NS):
            c=np.where(np.isnan(C[si]),0,C[si]).astype(np.float64)
            nm=np.isnan(C[si])
            try:
                r=talib.RSI(c,14); rsi14[si]=np.where(nm,np.nan,r)
            except: pass
    if np.any(np.all(np.isnan(rsi14),axis=1)):
        rm=compute_rsi_manual(C,NS,ND,14)
        for si in range(NS):
            if np.all(np.isnan(rsi14[si])): rsi14[si]=rm[si]
    fwd=np.full((NS,ND),np.nan)
    for si in range(NS):
        for di in range(ND-5):
            if not np.isnan(C[si,di+5]) and not np.isnan(C[si,di]) and C[si,di]>0:
                fwd[si,di]=C[si,di+5]/C[si,di]-1.0
    atr_mean = compute_atr_vec(H, L, C, NS, ND, 14)
    print(f"  Raw factors done: {time.time()-t0:.1f}s", flush=True)
    return {"ret_5d":ret_5d,"oi_5d":oi_5d,"vol_5d":vol_5d,"range_5d":range_5d,
            "atrp_5d":atrp_5d,"ret_10d":ret_10d,"rsi14":rsi14,"fwd_ret_5d":fwd,"atr_mean":atr_mean}

def normalize_factor(f, NS, ND, minc=10):
    out=np.full((NS,ND),np.nan)
    for di in range(ND):
        v=f[:,di]; vl=v[~np.isnan(v)]
        if len(vl)<minc: continue
        mu,sig=np.mean(vl),np.std(vl)
        if sig<1e-12: continue
        for si in range(NS):
            if not np.isnan(v[si]): out[si,di]=(v[si]-mu)/sig
    return out

def compute_rolling_ic(raw, NS, ND, icw=60):
    t0=time.time(); print(f"[V112] Rolling IC w={icw}...", flush=True)
    fwd=raw["fwd_ret_5d"]; ic=np.full((N_FACTORS,ND),np.nan)
    for fi,fname in enumerate(FACTOR_NAMES):
        f=raw[fname]
        for di in range(icw+5,ND):
            cv=[]
            for tdi in range(di-icw,di):
                fd,rd=f[:,tdi],fwd[:,tdi]; m=(~np.isnan(fd))&(~np.isnan(rd))
                fv,rv=fd[m],rd[m]
                if len(fv)>=15:
                    c=np.corrcoef(pd.Series(fv).rank().values,pd.Series(rv).rank().values)[0,1]
                    if not np.isnan(c): cv.append(c)
            if len(cv)>=5: ic[fi,di]=np.mean(cv)
    print(f"  IC done: {time.time()-t0:.1f}s", flush=True); return ic

def compute_bma_weights(ic, ND, ps=5.0):
    w=np.full((N_FACTORS,ND),np.nan)
    for fi in range(N_FACTORS):
        for di in range(20,ND):
            vh=ic[fi,max(0,di-120):di]; vv=vh[~np.isnan(vh)]
            if len(vv)<5: continue
            np_=np.sum(vv>0); nn_=len(vv)-np_
            w[fi,di]=(ps/2+np_)/(ps+np_+nn_)
    for di in range(ND):
        wd=w[:,di]; v=wd[~np.isnan(wd)]
        if len(v)==N_FACTORS:
            s=np.nansum(wd)
            if s>0: w[:,di]=wd/s
    return w

# ===== INNOVATION 1: Spectral Risk =====
def spectral_risk(returns, nq=20):
    """risk = sum(w_i*|quantile_i|), w_i=-ln(p_i). Exponential tail weight."""
    v=returns[~np.isnan(returns)]
    if len(v)<nq: return np.nan
    p=np.linspace(0.01,1.0,nq); wt=-np.log(p); wt/=wt.sum()
    return float(np.sum(wt*np.abs(np.quantile(v,p))))

def rolling_spectral_risk(C, NS, ND, window=60, nq=20):
    t0=time.time(); print(f"[V112] Spectral risk w={window} q={nq}...", flush=True)
    sr=np.full(ND,np.nan)
    for di in range(window+1,ND):
        dr=[]
        for dd in range(di-window,di):
            r=[C[si,dd]/C[si,dd-1]-1 for si in range(NS)
               if not np.isnan(C[si,dd]) and not np.isnan(C[si,dd-1]) and C[si,dd-1]>0]
            if r: dr.append(np.mean(r))
        if len(dr)>=window//2: sr[di]=spectral_risk(np.array(dr),nq)
    print(f"  Spectral risk done: {time.time()-t0:.1f}s ({np.sum(~np.isnan(sr))} valid)", flush=True)
    return sr

def spectral_mult(val, med, hi=2.0, lo=0.5, sred=0.5, sbo=1.3):
    """Continuous position size multiplier from spectral risk ratio."""
    if np.isnan(val) or np.isnan(med) or med<1e-12: return 1.0
    ratio=val/med
    if ratio>=hi: return sred
    if ratio<=lo: return sbo
    if ratio>1.0: return 1.0-(ratio-1.0)/(hi-1.0)*(1.0-sred)
    return sbo+(ratio-lo)/(1.0-lo)*(1.0-sbo)

# ===== INNOVATION 2: Kalman Filter Volatility =====
def kalman_vol(prices, Q=0.001, Rf=0.1, pers=0.95):
    """Kalman vol estimator. State: vol, obs: |daily_ret|."""
    n=len(prices); x=np.full(n,np.nan); P=np.full(n,np.nan)
    ret=np.diff(prices)/np.clip(np.abs(prices[:-1]),1e-10,None)
    ret=np.where(np.isnan(ret),0,ret)
    iw=min(20,len(ret))
    x[0]=np.std(ret[:iw]) if iw>2 else 0.02; P[0]=1.0
    R=max(np.var(ret[:iw])*Rf if iw>2 else 0.001, 1e-8)
    for i in range(1,n):
        xp=x[i-1]*pers; Pp=P[i-1]+Q
        if i-1<len(ret):
            K=Pp/(Pp+R); x[i]=xp+K*(abs(ret[i-1])-xp); P[i]=(1-K)*Pp
        else: x[i]=xp; P[i]=Pp
    return x, P

def compute_kalman_bw(C, NS, ND, kb=1.0, Q=0.001, Rf=0.1, pers=0.95):
    t0=time.time(); print(f"[V112] Kalman BW Q={Q} pers={pers:.2f}...", flush=True)
    bw=np.full((NS,ND),np.nan)
    for si in range(NS):
        p=C[si].copy()
        if np.sum(~np.isnan(p))<30: continue
        for di in range(1,ND):
            if np.isnan(p[di]) and not np.isnan(p[di-1]): p[di]=p[di-1]
        ve,_=kalman_vol(p,Q=Q,Rf=Rf,pers=pers)
        for di in range(ND):
            if not np.isnan(C[si,di]) and not np.isnan(ve[di]):
                bw[si,di]=max(ve[di]*kb, 0.1)
    print(f"  Kalman BW done: {time.time()-t0:.1f}s", flush=True); return bw

# ===== NW Kernel + BMA + Kalman =====
def compute_nw_pred(raw, bma, kbw, NS, ND, tw=40, kb=1.0):
    t0=time.time(); print(f"[V112] NW+BMA+Kalman pred tw={tw}...", flush=True)
    nm={f:normalize_factor(raw[f],NS,ND) for f in FACTOR_NAMES}
    # Apply BMA weighting
    wn={}
    for fi,fname in enumerate(FACTOR_NAMES):
        orig=nm[fname]; res=np.full((NS,ND),np.nan)
        for di in range(ND):
            w=bma[fi,di]; w=w if not np.isnan(w) else 1.0/N_FACTORS
            for si in range(NS):
                if not np.isnan(orig[si,di]): res[si,di]=orig[si,di]*(w*N_FACTORS)
        wn[fname]=res
    fwd=raw["fwd_ret_5d"]; pred=np.full((NS,ND),np.nan)
    for di in range(tw+10,ND):
        tf,tt=[],[]
        for tdi in range(max(10,di-tw),di):
            for si in range(NS):
                fe=np.array([wn[f][si,tdi] for f in FACTOR_NAMES])
                tgt=fwd[si,tdi]
                if np.any(np.isnan(fe)) or np.isnan(tgt): continue
                tf.append(fe); tt.append(tgt)
        if len(tf)<20: continue
        tX,tY=np.array(tf),np.array(tt)
        fs=np.std(tX,axis=0); fs[fs<1e-12]=1.0
        for si in range(NS):
            qf=np.array([wn[f][si,di] for f in FACTOR_NAMES])
            if np.any(np.isnan(qf)): continue
            h=kbw[si,di]
            if np.isnan(h):
                av=raw["atr_mean"][si,di]
                h=max(av*kb,0.1) if not np.isnan(av) else kb
            d=tX-qf[None,:]; dist=np.sqrt(np.sum((d/fs[None,:])**2,axis=1))
            sd=dist/h; w=np.zeros(len(tX)); m=sd<=1.0
            if not np.any(m):
                mi=np.argmin(dist)
                if dist[mi]<1e12: w[mi]=1.0; m=np.zeros(len(dist),bool); m[mi]=True
                else: continue
            else: w[m]=0.75*(1-sd[m]**2)
            ws=np.sum(w)
            if ws<1e-12: continue
            pred[si,di]=np.sum(w*tY)/ws
        if di%100==0: print(f"  di={di}/{ND} valid={np.sum(~np.isnan(pred[:,di]))}",flush=True)
    print(f"  Pred done: {time.time()-t0:.1f}s", flush=True); return pred

def compute_ker(C, NS, ND):
    k10=np.full((NS,ND),np.nan)
    for si in range(NS):
        for di in range(10,ND):
            cs=C[si,di-10:di+1]; v=cs[~np.isnan(cs)]
            if len(v)<10 or v[0]<=0: continue
            tc=np.sum(np.abs(np.diff(v)))
            if tc>1e-10: k10[si,di]=abs(v[-1]-v[0])/tc
    kr=np.zeros((NS,ND),dtype=int)
    for si in range(NS):
        for di in range(ND):
            if np.isnan(k10[si,di]): continue
            if k10[si,di]<0.15: kr[si,di]=1
            elif k10[si,di]>0.3: kr[si,di]=-1
    return kr

def compute_atr_at(H,L,C,si,di,start):
    av=[]
    for j in range(max(start,di-14),di):
        hh,ll,cc=H[si,j],L[si,j],C[si,j]
        if not any(np.isnan([hh,ll,cc])): av.append(max(hh-ll,abs(hh-cc),abs(ll-cc)))
    return np.mean(av) if av else None

# ===== Backtest =====
def backtest(C,O,H,L,NS,ND,dates,syms,pred,ker,spec_r,slu,
             tn=2,mps=2,hd=5,shr=2.0,slr=0.5,sred=0.5,sbo=1.3,sdi=60,edi=None):
    if edi is None: edi=ND-1
    srd=spec_r[max(sdi,60):edi]; srd=srd[~np.isnan(srd)]
    smed=np.median(srd) if len(srd)>10 else np.nan
    eq=CASH0; pk=eq; mdd=0.0
    pos=[]; trades=[]; rtw=[]
    srh=srl=srn=0
    for di in range(max(sdi,1),edi):
        d=dates[di]; dp=0.0; npos=[]
        mode="normal"
        if len(rtw)>=5:
            wr_=sum(rtw[-15:])/len(rtw[-15:])
            if wr_>0.60: mode="winning"
            elif wr_<0.50: mode="losing"
        rm=spectral_mult(spec_r[di],smed,shr,slr,sred,sbo)
        if rm<0.9: srh+=1
        elif rm>1.1: srl+=1
        else: srn+=1
        pbys=defaultdict(list)
        for si,edi_,ep,sp,al in pos: pbys[si].append((edi_,ep,sp,al))
        for si,pl in pbys.items():
            c=C[si,di]
            if np.isnan(c):
                for ei,ep,sp,al in pl: npos.append((si,ei,ep,sp,al))
                continue
            ee=min(p[0] for p in pl); h=di-ee
            stopped=any(c<sp for _,_,sp,_ in pl)
            if stopped or h>=hd:
                for ei,ep,sp,al in pl:
                    pnl=(c-ep)/ep-COMM; pr=eq*al*pnl; dp+=pr
                    iw=pnl>0
                    trades.append({"pnl_abs":pr,"pnl_pct":pnl*100,"days":di-ei+1,
                                   "di":di,"year":d.year,"sym":syms[si],
                                   "sector":slu.get(si,'OTHER'),
                                   "reason":"stop" if stopped else "hold",
                                   "mode":mode[0].upper()})
                    rtw.append(1 if iw else 0)
            else:
                for ei,ep,sp,al in pl: npos.append((si,ei,ep,sp,al))
        pos=npos; eq+=dp
        if eq>pk: pk=eq
        if pk>0:
            dd=(pk-eq)/pk*100
            if dd>mdd: mdd=dd
        if eq<=0: break
        held={p[0] for p in pos}
        if len(held)>=tn: continue
        cands=[(pred[si,di],si) for si in range(NS)
               if si not in held and not np.isnan(pred[si,di])
               and di+1<ND and not np.isnan(O[si,di+1]) and ker[si,di]>=0]
        if not cands: continue
        cands.sort(key=lambda x:-x[0])
        nt=tn+(1 if mode=="winning" else (-1 if mode=="losing" else 0))
        nt=max(1,min(nt,tn*2))
        sc=defaultdict(int)
        for sh in held: sc[slu.get(sh,'OTHER')]+=1
        ne=[]
        for pv,si in cands:
            if len(held)+len(ne)>=nt or si in held: break
            ss=slu.get(si,'OTHER')
            if sc[ss]>=mps or pv<=0: continue
            ne.append((pv,si,ss)); sc[ss]+=1
        if not ne: continue
        ap=LEVERAGE/(len(pos)+len(ne))*rm
        upos=[(si,ei,ep,sp,ap) for si,ei,ep,sp,al in pos]
        for pv,si,ss in ne:
            ep=O[si,di+1]
            if np.isnan(ep) or ep<=0: continue
            atr=compute_atr_at(H,L,C,si,di,sdi)
            if atr is None: continue
            upos.append((si,di+1,ep,ep-3.0*atr,ap))
        pos=upos
    for si,ei,ep,sp,al in pos:
        c=C[si,ND-1]
        if not np.isnan(c) and c>0: eq+=eq*al*((c-ep)/ep-COMM)
    if trades:
        trades[0]["risk_info"]=f"sr_regime=[hi:{srh} norm:{srn} lo:{srl}]"
    return trades, eq, mdd

def analyze(trades, eq, mdd, label=""):
    if not trades: print(f"  {label}: no trades"); return None
    nw=sum(1 for t in trades if t["pnl_pct"]>0); wr=nw/len(trades)*100
    nd=max(1,trades[-1]["di"]-trades[0]["di"])
    ann=((eq/CASH0)**(1/max(1.0,nd/252))-1)*100
    ap=[t["pnl_abs"] for t in sorted(trades,key=lambda x:x["di"])]
    r=np.array(ap)/CASH0; sh=np.mean(r)/np.std(r)*np.sqrt(252) if np.std(r)>0 else 0
    ns=sum(1 for t in trades if t["reason"]=="stop")
    nh=sum(1 for t in trades if t["reason"]=="hold")
    mc={"W":0,"N":0,"L":0}
    for t in trades:
        m=t.get("mode","N")
        if m in mc: mc[m]+=1
    sc=defaultdict(int)
    for t in trades: sc[t.get("sector","OTHER")]+=1
    ss=" ".join(f"{k}:{v}" for k,v in sorted(sc.items()))
    apnl=np.mean([t["pnl_pct"] for t in trades])
    aw=np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"]>0])
    al=np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"]<=0])
    print(f"  {label}: {len(trades)}t (s:{ns} h:{nh}) WR={wr:.1f}% ann={ann:+.1f}% DD={mdd:.1f}% Sh={sh:.2f} eq={eq:,.0f}")
    print(f"    avg={apnl:+.3f}% win={aw:+.3f}% loss={al:+.3f}% modes=[W:{mc['W']} N:{mc['N']} L:{mc['L']}]")
    print(f"    sectors: {ss}")
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

def walk_forward(C,O,H,L,NS,ND,dates,syms,pred,ker,spec_r,slu,
                 tn=2,mps=2,hd=5,shr=2.0,slr=0.5,sred=0.5,sbo=1.3,label=""):
    print(f"\n{'='*70}\n  WF V112 {label}\n  tn={tn} mps={mps} shr={shr} slr={slr} sr={sred} sb={sbo}\n{'='*70}")
    yrs=sorted(set(d.year for d in dates)); at=[]
    for ty in range(2019,yrs[-1]+1):
        ts=te=None
        for i,d in enumerate(dates):
            if d.year==ty and ts is None: ts=i
            if d.year==ty: te=i
        if ts is None: continue
        tr,_,_=backtest(C,O,H,L,NS,ND,dates,syms,pred,ker,spec_r,slu,
                        tn=tn,mps=mps,hd=hd,shr=shr,slr=slr,sred=sred,sbo=sbo,sdi=ts,edi=te+1)
        tt=[t for t in tr if dates[t["di"]].year==ty]; at.extend(tt)
        if tt:
            n=len(tt); nw=sum(1 for t in tt if t["pnl_pct"]>0)
            sc=defaultdict(int)
            for t in tt: sc[t.get("sector","OTHER")]+=1
            ss=" ".join(f"{k}:{v}" for k,v in sorted(sc.items()))
            print(f"  {ty}: {n}t WR={nw/n*100:.1f}% avg={np.mean([t['pnl_pct'] for t in tt]):+.2f}% [{ss}]",flush=True)
        else: print(f"  {ty}: no trades",flush=True)
    if at:
        nw=sum(1 for t in at if t["pnl_pct"]>0); cum=np.prod([1+t["pnl_pct"]/100 for t in at])-1
        print(f"\n  WF TOTAL: {len(at)}t WR={nw/len(at)*100:.1f}% cum={cum:+.1%}")
    return at

def main():
    t0=time.time()
    print("="*70+"\n  V112: SPECTRAL RISK + KALMAN ADAPTIVE SIZING\n"+"="*70)
    C,O,H,L,V,OI,NS,ND,dates,syms = load_all_data(start="2016-01-01")
    print(f"  {NS} sym, {ND} days, {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    slu=build_sector_lookup(syms)
    sd=defaultdict(int)
    for s in slu.values(): sd[s]+=1
    print(f"  Sectors: {dict(sd)}")
    bt19=None
    for i,d in enumerate(dates):
        if d>=pd.Timestamp("2019-01-01"): bt19=i; break
    raw=compute_raw_factors(C,O,H,L,V,OI,NS,ND)
    ker=compute_ker(C,NS,ND)
    ic=compute_rolling_ic(raw,NS,ND,icw=60)
    bma=compute_bma_weights(ic,ND,ps=5.0)
    # Kalman bandwidth sweep
    kcache={}
    for Q in [0.001,0.005,0.01]:
        for pers in [0.90,0.95,0.98]:
            kcache[(Q,pers)]=compute_kalman_bw(C,NS,ND,kb=1.0,Q=Q,pers=pers)
    # Spectral risk sweep
    scache={}
    for w in [40,60,80]:
        for nq in [10,20,30]:
            scache[(w,nq)]=rolling_spectral_risk(C,NS,ND,window=w,nq=nq)
    # NW predictions per Kalman config
    pcache={}
    for (Q,pers),kbw in kcache.items():
        print(f"\n--- NW+BMA+Kalman (Q={Q}, pers={pers:.2f}) ---")
        pcache[(Q,pers)]=compute_nw_pred(raw,bma,kbw,NS,ND,tw=40,kb=1.0)
    # Parameter sweep
    print(f"\n{'='*70}\n  PARAMETER SWEEP (2019-2026)\n{'='*70}")
    results=[]; sc=0
    for (Q,pers),pred in pcache.items():
        for (sw,snq),sr_arr in scache.items():
            for mps in [2,3]:
                for sred in [0.3,0.5]:
                    for sbo in [1.2,1.5]:
                        sc+=1
                        tr,eq,dd=backtest(C,O,H,L,NS,ND,dates,syms,pred,ker,sr_arr,slu,
                                          tn=2,mps=mps,shr=2.0,slr=0.5,sred=sred,sbo=sbo,sdi=bt19)
                        if len(tr)<10: continue
                        nw=sum(1 for t in tr if t["pnl_pct"]>0); wr=nw/len(tr)*100
                        nd_=max(1,tr[-1]["di"]-tr[0]["di"])
                        ann=((eq/CASH0)**(1/max(1,nd_/252))-1)*100
                        ap=[t["pnl_abs"] for t in sorted(tr,key=lambda x:x["di"])]
                        ra=np.array(ap)/CASH0
                        sh=np.mean(ra)/np.std(ra)*np.sqrt(252) if np.std(ra)>0 else 0
                        results.append({"Q":Q,"pers":pers,"sw":sw,"snq":snq,"mps":mps,
                                        "sred":sred,"sbo":sbo,"n":len(tr),"wr":wr,
                                        "ann":ann,"dd":dd,"sh":sh,"eq":eq})
    results.sort(key=lambda x:-x["ann"])
    print(f"\n  {sc} configs, {len(results)} w/ 10+ trades")
    print(f"\n{'Q':>6} {'Prs':>5} {'SpW':>4} {'SpQ':>4} {'M':>2} {'SR':>4} {'SB':>4} {'N':>4} {'WR':>6} {'Ann':>9} {'DD':>6} {'Sh':>6}")
    print("-"*80)
    for r in results[:20]:
        print(f"{r['Q']:>6.3f} {r['pers']:>5.2f} {r['sw']:>4} {r['snq']:>4} {r['mps']:>2} {r['sred']:>4.1f} {r['sbo']:>4.1f} {r['n']:>4} {r['wr']:>5.1f}% {r['ann']:>+8.1f}% {r['dd']:>5.1f}% {r['sh']:>5.2f}")
    if not results: print("  No results."); return
    # Walk-forward top configs
    ba=results[0]
    bs=max(results,key=lambda x:x["sh"])
    br=max(results,key=lambda x:x["ann"]/max(x["dd"],1))
    for lbl,b in [("BEST-ANN",ba),("BEST-SHARPE",bs),("BEST-RISK-ADJ",br)]:
        pr=pcache[(b["Q"],b["pers"])]; sr_=scache[(b["sw"],b["snq"])]
        walk_forward(C,O,H,L,NS,ND,dates,syms,pr,ker,sr_,slu,
                     tn=2,mps=b["mps"],shr=2.0,slr=0.5,sred=b["sred"],sbo=b["sbo"],label=lbl)
    # V112 vs V96 baseline comparison
    print(f"\n{'='*70}\n  COMPARISON: V112 vs V96-style\n{'='*70}")
    pb=pcache[(ba["Q"],ba["pers"])]; sb=scache[(ba["sw"],ba["snq"])]
    t112,e112,d112=backtest(C,O,H,L,NS,ND,dates,syms,pb,ker,sb,slu,
                            tn=2,mps=ba["mps"],shr=2.0,slr=0.5,sred=ba["sred"],sbo=ba["sbo"],sdi=bt19)
    pbl=pcache[(0.001,0.95)]; sbl=np.full(ND,1.0)
    t96,e96,d96=backtest(C,O,H,L,NS,ND,dates,syms,pbl,ker,sbl,slu,
                         tn=2,mps=2,shr=99.0,slr=0.01,sred=1.0,sbo=1.0,sdi=bt19)
    print("\n  V112 BEST-ANN:")
    analyze(t112,e112,d112,"V112-Spectral+Kalman")
    print("\n  V96-STYLE BASELINE:")
    analyze(t96,e96,d96,"V96-baseline")
    if t112 and t96:
        print(f"\n  Delta: eq={e112-e96:+,.0f} dd={d112-d96:+.1f}% trades={len(t112)-len(t96):+d}")
    # Spectral risk diagnostic
    sv=sb[~np.isnan(sb)]
    if len(sv)>0:
        med=np.median(sv)
        print(f"\n  Spectral risk: median={med:.6f} p10={np.percentile(sv,10):.6f} p90={np.percentile(sv,90):.6f}")
        print(f"  Ratio range: min/med={np.min(sv)/med:.2f} max/med={np.max(sv)/med:.2f}")
    print(f"\n[V112] Done. {time.time()-t0:.1f}s")

if __name__=="__main__":
    main()
