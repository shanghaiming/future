"""
V117: Cross-Commodity Lead-Lag Dynamics + NW Kernel + BMA
==========================================================
Anchor commodities lead their sectors with 1-5 day lag.
Lead-lag ratio (LLR) = corr(leader_ret[t-1], follower_ret[t]).
High LLR + leader moved + follower lagging = expected catch-up alpha.
9 factors: 7 V96 base + llr_signal + sector_rel.
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
    import talib; HAS_TALIB = True
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
    'm':'AGRI','y':'AGRI','a':'AGRI','p':'AGRI','c':'AGRI',
    'cs':'AGRI','jd':'AGRI','rr':'AGRI','lrm':'AGRI','rm':'AGRI','ru':'AGRI',
    'cf':'SOFTS','sr':'SOFTS','ap':'SOFTS','cj':'SOFTS','pk':'SOFTS',
    'lh':'SOFTS','sp':'SOFTS','b':'SOFTS','br':'SOFTS',
}
SECTOR_LEADERS = {'BLACK':'i','METAL':'cu','ENERGY':'sc',
                  'AGRI':'m','SOFTS':'cf','CHEMICAL':'ta'}

FACTOR_NAMES = ["ret_5d","oi_5d","rsi14","vol_5d","ret_10d",
                "range_5d","atrp_5d","llr_signal","sector_rel"]
N_FACTORS = len(FACTOR_NAMES)


def _base(sym: str) -> str:
    s = sym.lower().split('.')[0].strip()
    while s and s[-1].isdigit(): s = s[:-1]
    return s[:-2] if s.endswith('fi') else s


def build_lookups(syms: List[str]):
    sec_lk = {si: SECTOR_MAP.get(_base(s), 'OTHER') for si, s in enumerate(syms)}
    b2si = {_base(s): si for si, s in enumerate(syms)}
    ld_lk: Dict[int, Optional[int]] = {}
    for si, s in enumerate(syms):
        sec = sec_lk.get(si, 'OTHER')
        lb = SECTOR_LEADERS.get(sec)
        ld_lk[si] = b2si.get(lb) if lb else None
    return sec_lk, ld_lk


def compute_rsi_manual(C, NS, ND, period=14):
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
                vg,vl = [],[]
                for j in range(di, min(di+period, ND)):
                    if not np.isnan(gains[j]):
                        vg.append(gains[j])
                        vl.append(losses[j] if not np.isnan(losses[j]) else 0.0)
                if len(vg) >= period:
                    ag,al = np.mean(vg), np.mean(vl)
                    rsi[si,di+period-1] = 100.0 if al==0 else 100.0-100.0/(1.0+ag/al)
                continue
            ag = (ag*(period-1)+gains[di])/period
            al = (al*(period-1)+losses[di])/period
            rsi[si,di] = 100.0 if al==0 else 100.0-100.0/(1.0+ag/al)
    return rsi


def _atr_slice(H, L, C, si, j):
    hh, ll, cc = H[si,j], L[si,j], C[si,j]
    if any(np.isnan([hh,ll,cc])): return None
    prev = C[si,j-1] if j>0 and not np.isnan(C[si,j-1]) else cc
    return max(hh-ll, abs(hh-prev), abs(ll-prev))


def compute_raw_factors(C, O, H, L, V, OI, NS, ND, ld_lk, sec_lk, llr_w=20):
    t0 = time.time()
    print(f"[V117] Raw factors (llr_w={llr_w})...", flush=True)
    ret_5d = np.full((NS,ND), np.nan)
    oi_5d = np.full((NS,ND), np.nan)
    vol_5d = np.full((NS,ND), np.nan)
    rng_5d = np.full((NS,ND), np.nan)
    atrp_5d = np.full((NS,ND), np.nan)
    ret_10d = np.full((NS,ND), np.nan)
    rsi14 = np.full((NS,ND), np.nan)
    daily_ret = np.full((NS,ND), np.nan)
    llr_signal = np.full((NS,ND), np.nan)
    sector_rel = np.full((NS,ND), np.nan)
    fwd = np.full((NS,ND), np.nan)
    atr_m = np.full((NS,ND), np.nan)

    for si in range(NS):
        for di in range(1, ND):
            if not np.isnan(C[si,di]) and not np.isnan(C[si,di-1]) and C[si,di-1]>0:
                daily_ret[si,di] = C[si,di]/C[si,di-1]-1.0
        for di in range(5, ND):
            if not np.isnan(C[si,di]) and not np.isnan(C[si,di-5]) and C[si,di-5]>0:
                ret_5d[si,di] = C[si,di]/C[si,di-5]-1.0
            if not np.isnan(OI[si,di]) and not np.isnan(OI[si,di-5]) and OI[si,di-5]>0:
                oi_5d[si,di] = OI[si,di]/OI[si,di-5]-1.0
            vv = V[si,di-5:di]; valid = vv[~np.isnan(vv)]
            if len(valid)>=3: vol_5d[si,di] = np.mean(valid)
            rv = []
            for j in range(di-5,di):
                if not np.isnan(H[si,j]) and not np.isnan(L[si,j]) \
                   and not np.isnan(C[si,j]) and C[si,j]>0 and H[si,j]>L[si,j]:
                    rv.append((H[si,j]-L[si,j])/C[si,j])
            if len(rv)>=3: rng_5d[si,di] = np.mean(rv)
        for di in range(6, ND):
            av = [_atr_slice(H,L,C,si,j) for j in range(di-5,di)]
            av = [x for x in av if x]
            if av and not np.isnan(C[si,di]) and C[si,di]>0:
                atrp_5d[si,di] = np.mean(av)/C[si,di]
        for di in range(10, ND):
            if not np.isnan(C[si,di]) and not np.isnan(C[si,di-10]) and C[si,di-10]>0:
                ret_10d[si,di] = C[si,di]/C[si,di-10]-1.0
        for di in range(ND-5):
            if not np.isnan(C[si,di+5]) and not np.isnan(C[si,di]) and C[si,di]>0:
                fwd[si,di] = C[si,di+5]/C[si,di]-1.0
        for di in range(20, ND):
            av = [_atr_slice(H,L,C,si,j) for j in range(di-14,di)]
            av = [x for x in av if x]
            if av and not np.isnan(C[si,di]) and C[si,di]>0:
                atr_m[si,di] = np.mean(av)/C[si,di]

    # RSI
    if HAS_TALIB:
        for si in range(NS):
            c = np.where(np.isnan(C[si]),0,C[si]).astype(np.float64)
            nm = np.isnan(C[si])
            try: rsi14[si] = np.where(nm, np.nan, talib.RSI(c,14))
            except: pass
    fb = np.all(np.isnan(rsi14), axis=1)
    if fb.any():
        rm = compute_rsi_manual(C, NS, ND, 14)
        for si in range(NS):
            if fb[si]: rsi14[si] = rm[si]

    # Lead-lag signal
    for si in range(NS):
        lsi = ld_lk.get(si)
        if lsi is None: continue
        for di in range(llr_w+1, ND):
            lag, cur = [], []
            for t in range(di-llr_w, di):
                lv, fv = daily_ret[lsi,t], daily_ret[si,t+1]
                if not np.isnan(lv) and not np.isnan(fv):
                    lag.append(lv); cur.append(fv)
            if len(lag) < max(5, llr_w//2): continue
            corr = np.corrcoef(lag, cur)[0,1]
            if np.isnan(corr): continue
            lret = daily_ret[lsi, di]
            if np.isnan(lret): continue
            llr_signal[si,di] = lret * corr

    # Sector relative strength
    sec_members: Dict[str,List[int]] = defaultdict(list)
    for s in range(NS): sec_members[sec_lk.get(s,'OTHER')].append(s)
    for di in range(5, ND):
        for sec, members in sec_members.items():
            rv = [ret_5d[s,di] for s in members if not np.isnan(ret_5d[s,di])]
            if len(rv)<3: continue
            avg = np.mean(rv)
            for s in members:
                if not np.isnan(ret_5d[s,di]): sector_rel[s,di] = ret_5d[s,di]-avg

    print(f"  llr_valid={np.sum(~np.isnan(llr_signal)):,} "
          f"sr_valid={np.sum(~np.isnan(sector_rel)):,} "
          f"{time.time()-t0:.1f}s", flush=True)
    return {"ret_5d":ret_5d,"oi_5d":oi_5d,"vol_5d":vol_5d,"range_5d":rng_5d,
            "atrp_5d":atrp_5d,"ret_10d":ret_10d,"rsi14":rsi14,
            "llr_signal":llr_signal,"sector_rel":sector_rel,
            "fwd_ret_5d":fwd,"atr_mean":atr_m}


def norm_factor(f, NS, ND, mc=10):
    out = np.full((NS,ND), np.nan)
    for di in range(ND):
        v = f[:,di]; ok = v[~np.isnan(v)]
        if len(ok)<mc: continue
        mu,sd = np.mean(ok), np.std(ok)
        if sd<1e-12: continue
        for si in range(NS):
            if not np.isnan(v[si]): out[si,di] = (v[si]-mu)/sd
    return out


def compute_ic(raw, NS, ND, icw=60):
    t0=time.time(); print(f"[V117] IC (w={icw})...", flush=True)
    fwd=raw["fwd_ret_5d"]; ic=np.full((N_FACTORS,ND),np.nan)
    for fi,fn in enumerate(FACTOR_NAMES):
        fac=raw[fn]
        for di in range(icw+5,ND):
            vals=[]
            for tdi in range(di-icw,di):
                m=(~np.isnan(fac[:,tdi]))&(~np.isnan(fwd[:,tdi]))
                fv,rv = fac[:,tdi][m], fwd[:,tdi][m]
                if len(fv)>=15:
                    c=np.corrcoef(pd.Series(fv).rank().values,
                                  pd.Series(rv).rank().values)[0,1]
                    if not np.isnan(c): vals.append(c)
            if len(vals)>=5: ic[fi,di] = np.mean(vals)
    print(f"  IC done {time.time()-t0:.1f}s", flush=True)
    return ic


def compute_bma(ic, ND, ps=5.0):
    w = np.full((N_FACTORS,ND), np.nan)
    for fi in range(N_FACTORS):
        for di in range(20, ND):
            h = ic[fi,max(0,di-120):di]; v = h[~np.isnan(h)]
            if len(v)<5: continue
            np_ = np.sum(v>0); nn = len(v)-np_
            w[fi,di] = (ps/2+np_)/(ps+nn+np_)
    for di in range(ND):
        row = w[:,di]; ok = row[~np.isnan(row)]
        if len(ok)==N_FACTORS:
            s = np.nansum(row)
            if s>0: w[:,di] = row/s
    return w


def nw_predict(raw, bma, NS, ND, tw=40, bw=1.0):
    t0=time.time()
    print(f"[V117] NW+BMA (tw={tw}, bw={bw:.1f})...", flush=True)
    nm = {fn: norm_factor(raw[fn], NS, ND) for fn in FACTOR_NAMES}
    wt = {}
    for fi,fn in enumerate(FACTOR_NAMES):
        orig=nm[fn]; res=np.full((NS,ND),np.nan)
        for di in range(ND):
            w=bma[fi,di]
            if np.isnan(w): w=1.0/N_FACTORS
            for si in range(NS):
                if not np.isnan(orig[si,di]): res[si,di] = orig[si,di]*(w*N_FACTORS)
        wt[fn]=res
    fwd=raw["fwd_ret_5d"]; atr=raw["atr_mean"]
    pred=np.full((NS,ND),np.nan)
    for di in range(tw+10, ND):
        tf,tt = [],[]
        for tdi in range(max(10,di-tw), di):
            for si in range(NS):
                f=np.array([wt[fn][si,tdi] for fn in FACTOR_NAMES])
                t_=fwd[si,tdi]
                if np.any(np.isnan(f)) or np.isnan(t_): continue
                tf.append(f); tt.append(t_)
        if len(tf)<20: continue
        tX=np.array(tf); tY=np.array(tt)
        fs=np.std(tX,axis=0); fs[fs<1e-12]=1.0
        for si in range(NS):
            qf=np.array([wt[fn][si,di] for fn in FACTOR_NAMES])
            if np.any(np.isnan(qf)): continue
            av=atr[si,di]; h=max(av*bw,0.1) if not np.isnan(av) else bw
            d_=np.sqrt(np.sum(((tX-qf)/fs)**2, axis=1)); sd=d_/h
            kw=np.zeros(len(tX)); m=sd<=1.0
            if not np.any(m):
                idx=np.argmin(d_)
                if d_[idx]<1e12: kw[idx]=1.0; m=np.zeros(len(d_),bool); m[idx]=True
                else: continue
            else: kw[m]=0.75*(1.0-sd[m]**2)
            ws=np.sum(kw)
            if ws<1e-12: continue
            pred[si,di] = np.sum(kw*tY)/ws
        if di%100==0:
            print(f"  di={di}/{ND} valid={np.sum(~np.isnan(pred[:,di]))}", flush=True)
    print(f"  NW done {time.time()-t0:.1f}s", flush=True)
    return pred


def compute_ker(C, NS, ND):
    k10=np.full((NS,ND),np.nan)
    for si in range(NS):
        for di in range(10,ND):
            v=C[si,di-10:di+1]; ok=v[~np.isnan(v)]
            if len(ok)<10 or ok[0]<=0: continue
            tc=np.sum(np.abs(np.diff(ok)))
            if tc>1e-10: k10[si,di]=abs(ok[-1]-ok[0])/tc
    kr=np.zeros((NS,ND),dtype=int)
    for si in range(NS):
        for di in range(ND):
            v=k10[si,di]
            if np.isnan(v): continue
            if v<0.15: kr[si,di]=1
            elif v>0.3: kr[si,di]=-1
    return kr


def compute_pvol(C, NS, ND, lb=20):
    pv=np.full(ND,np.nan)
    for di in range(lb+1,ND):
        dr=[]
        for dd in range(di-lb,di):
            r=[C[si,dd]/C[si,dd-1]-1.0 for si in range(NS)
               if not np.isnan(C[si,dd]) and not np.isnan(C[si,dd-1]) and C[si,dd-1]>0]
            if r: dr.append(np.mean(r))
        if len(dr)>=lb//2: pv[di]=np.std(dr)
    return pv


def backtest(C, O, H, L, NS, ND, dates, syms, pred, ker, pvol, sec_lk,
             top_n=2, mps=2, hd=5, wt_=0.60, as_=3.0,
             vhm=2.0, vlm=0.5, sr=0.5, sb=1.3, sdi=60, edi=None):
    if edi is None: edi=ND-1
    vd=pvol[max(sdi,21):edi]; vv=vd[~np.isnan(vd)]
    vmed = np.median(vv) if len(vv)>10 else 1e-6
    eq,peak,mdd = float(CASH0), float(CASH0), 0.0
    pos=[]; trades=[]; rw=[]
    for di in range(max(sdi,1), edi):
        d=dates[di]; dpnl=0.0; npos=[]
        # mode
        mode="normal"
        if len(rw)>=5:
            wr=sum(rw[-15:])/len(rw[-15:])
            if wr>wt_: mode="winning"
            elif wr<0.50: mode="losing"
        # vol mult
        vm=1.0
        if not np.isnan(pvol[di]) and not np.isnan(vmed) and vmed>1e-12:
            r_=pvol[di]/vmed
            if r_>vhm: vm=sr
            elif r_<vlm: vm=sb
        # exit
        pb=defaultdict(list)
        for si,edi_,ep,sp,al in pos: pb[si].append((edi_,ep,sp,al))
        for si, pl in pb.items():
            c=C[si,di]
            if np.isnan(c):
                for edi_,ep,sp,al in pl: npos.append((si,edi_,ep,sp,al))
                continue
            early=min(p[0] for p in pl); hold=di-early
            stopped=any(c<sp for _,_,sp,_ in pl)
            if stopped or hold>=hd:
                reason="stop" if stopped else "hold"
                for edi_,ep,sp,al in pl:
                    pnl=(c-ep)/ep-COMM; profit=eq*al*pnl; dpnl+=profit
                    trades.append({"pnl_abs":profit,"pnl_pct":pnl*100,
                        "days":di-edi_+1,"di":di,"year":d.year,
                        "sym":syms[si],"sector":sec_lk.get(si,'OTHER'),
                        "reason":reason,"mode":mode[0].upper()})
                    rw.append(1 if pnl>0 else 0)
            else:
                for edi_,ep,sp,al in pl: npos.append((si,edi_,ep,sp,al))
        pos=npos; eq+=dpnl
        if eq>peak: peak=eq
        if peak>0:
            dd=(peak-eq)/peak*100
            if dd>mdd: mdd=dd
        if eq<=0: break
        # entry
        held={p[0] for p in pos}
        if len(held)>=top_n: continue
        cands=[]
        for si in range(NS):
            if si in held: continue
            p_=pred[si,di]
            if np.isnan(p_): continue
            if di+1>=ND or np.isnan(O[si,di+1]): continue
            if ker[si,di]<0: continue
            cands.append((p_,si))
        if not cands: continue
        cands.sort(key=lambda x:-x[0])
        nt=top_n
        if mode=="winning": nt=min(top_n+1,top_n*2)
        elif mode=="losing": nt=max(1,top_n-1)
        sc=defaultdict(int)
        for sh in held: sc[sec_lk.get(sh,'OTHER')]+=1
        ne=[]
        for pv_,si in cands:
            if len(held)+len(ne)>=nt: break
            s_=sec_lk.get(si,'OTHER')
            if sc[s_]>=mps: continue
            if pv_<=0: continue
            ne.append((pv_,si,s_)); sc[s_]+=1
        if not ne: continue
        al=LEVERAGE/(len(pos)+len(ne))*vm
        up=[(si,edi_,ep,sp,al) for si,edi_,ep,sp,_ in pos]
        for pv_,si,s_ in ne:
            ep=O[si,di+1]
            if np.isnan(ep) or ep<=0: continue
            av=[_atr_slice(H,L,C,si,j) for j in range(max(sdi,di-14),di)]
            av=[x for x in av if x]
            if not av: continue
            up.append((si,di+1,ep,ep-as_*np.mean(av),al))
        pos=up
    for si,edi_,ep,sp,al in pos:
        c=C[si,ND-1]
        if not np.isnan(c) and c>0: eq+=eq*al*((c-ep)/ep-COMM)
    return trades, eq, mdd


def analyze(trades, eq, mdd, label=""):
    if not trades: print(f"  {label}: no trades"); return None
    nw=sum(1 for t in trades if t["pnl_pct"]>0)
    wr=nw/len(trades)*100
    nd_=max(1,trades[-1]["di"]-trades[0]["di"])
    ann=((eq/CASH0)**(1/max(1.0,nd_/252))-1)*100
    ap=[t["pnl_abs"] for t in sorted(trades,key=lambda x:x["di"])]
    r=np.array(ap)/CASH0
    sh=np.mean(r)/np.std(r)*np.sqrt(252) if np.std(r)>0 else 0
    ns=sum(1 for t in trades if t["reason"]=="stop")
    nh=len(trades)-ns
    sc=defaultdict(int)
    for t in trades: sc[t.get("sector","OTHER")]+=1
    ss=" ".join(f"{k}:{v}" for k,v in sorted(sc.items()))
    print(f"  {label}: {len(trades)}t (s:{ns} h:{nh}) WR={wr:.1f}% "
          f"ann={ann:+.1f}% DD={mdd:.1f}% Sh={sh:.2f} eq={eq:,.0f}")
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


def walk_fwd(C,O,H,L,NS,ND,dates,syms,pred,ker,pvol,sec_lk,
             top_n=2,mps=2,hd=5,vhm=2.0,vlm=0.5,sr=0.5,sb=1.3,label=""):
    print(f"\n{'='*70}")
    print(f"  WF V117 {label} tn={top_n} mps={mps} vhm={vhm:.1f} vlm={vlm:.1f}")
    print(f"{'='*70}")
    yrs=sorted(set(d.year for d in dates)); at=[]
    for ty in range(2019, yrs[-1]+1):
        ts=te=None
        for i,d in enumerate(dates):
            if d.year==ty and ts is None: ts=i
            if d.year==ty: te=i
        if ts is None: continue
        tr,_,_ = backtest(C,O,H,L,NS,ND,dates,syms,pred,ker,pvol,sec_lk,
                          top_n=top_n,mps=mps,hd=hd,vhm=vhm,vlm=vlm,sr=sr,sb=sb,
                          sdi=ts,edi=te+1)
        yt=[t for t in tr if dates[t["di"]].year==ty]; at.extend(yt)
        if yt:
            nw=sum(1 for t in yt if t["pnl_pct"]>0)
            sc=defaultdict(int)
            for t in yt: sc[t.get("sector","OTHER")]+=1
            ss=" ".join(f"{k}:{v}" for k,v in sorted(sc.items()))
            print(f"  {ty}: {len(yt)}t WR={nw/len(yt)*100:.1f}% "
                  f"avg={np.mean([t['pnl_pct'] for t in yt]):+.2f}% [{ss}]",flush=True)
        else: print(f"  {ty}: no trades",flush=True)
    if at:
        nw=sum(1 for t in at if t["pnl_pct"]>0)
        cum=np.prod([1+t["pnl_pct"]/100 for t in at])-1
        print(f"\n  WF TOTAL: {len(at)}t WR={nw/len(at)*100:.1f}% "
              f"avg={np.mean([t['pnl_pct'] for t in at]):+.2f}% cum={cum:+.1%}")
    return at


def main():
    t0=time.time()
    print("="*70)
    print("  V117: CROSS-COMMODITY LEAD-LAG + NW KERNEL + BMA")
    print("  9 factors: 7 V96 base + llr_signal + sector_rel")
    print("  Walk-forward 2019-2026. LEVERAGE=1.0.")
    print("="*70)
    C,O,H,L,V,OI,NS,ND,dates,syms = load_all_data(start="2016-01-01")
    print(f"  {NS} sym, {ND} days, {dates[0].strftime('%Y-%m-%d')} "
          f"to {dates[-1].strftime('%Y-%m-%d')}")
    sec_lk, ld_lk = build_lookups(syms)
    # report leaders
    lm = {sec: SECTOR_LEADERS[sec] for sec in SECTOR_LEADERS}
    print(f"  Sector leaders: {lm}")
    sd = defaultdict(int)
    for s in sec_lk.values(): sd[s]+=1
    print(f"  Sector dist: {dict(sd)}")
    bt19 = None
    for i,d in enumerate(dates):
        if d>=pd.Timestamp("2019-01-01"): bt19=i; break

    # 1. Raw factors per llr_window
    raw_c = {w: compute_raw_factors(C,O,H,L,V,OI,NS,ND,ld_lk,sec_lk,llr_w=w)
             for w in [10,20,30]}
    ker = compute_ker(C,NS,ND)
    pvol = compute_pvol(C,NS,ND,20)

    # 2. IC + BMA per llr_window
    bma_c = {w: compute_bma(compute_ic(raw_c[w],NS,ND,60), ND) for w in [10,20,30]}

    # 3. NW predictions per llr_window
    pred_c = {w: nw_predict(raw_c[w], bma_c[w], NS, ND, tw=40, bw=1.0)
              for w in [10,20,30]}

    # 4. Sweep: llr_w x top_n x mps = 3 x 2 x 2 = 12 configs
    print(f"\n{'='*70}\n  PARAMETER SWEEP (2019-2026)\n{'='*70}")
    results = []
    for w in [10,20,30]:
        for tn in [2,3]:
            for mps in [2,3]:
                tr,eq,dd = backtest(C,O,H,L,NS,ND,dates,syms,
                    pred_c[w],ker,pvol,sec_lk,top_n=tn,mps=mps,sdi=bt19)
                if len(tr)<10: continue
                nw=sum(1 for t in tr if t["pnl_pct"]>0)
                wr=nw/len(tr)*100; nd_=max(1,tr[-1]["di"]-tr[0]["di"])
                ann=((eq/CASH0)**(1/max(1.0,nd_/252))-1)*100
                ap=[t["pnl_abs"] for t in sorted(tr,key=lambda x:x["di"])]
                ra=np.array(ap)/CASH0
                sh=np.mean(ra)/np.std(ra)*np.sqrt(252) if np.std(ra)>0 else 0
                results.append({"w":w,"tn":tn,"mps":mps,"n":len(tr),"wr":wr,
                                "ann":ann,"dd":dd,"sh":sh,"eq":eq})
    results.sort(key=lambda x:-x["ann"])
    print(f"\n  {len(results)} configs with 10+ trades")
    print(f"{'LLRw':>5} {'TN':>3} {'MPS':>3} {'N':>5} {'WR':>6} "
          f"{'Ann':>9} {'DD':>7} {'Sh':>7}")
    print("-"*60)
    for r in results[:10]:
        print(f"{r['w']:>5} {r['tn']:>3} {r['mps']:>3} {r['n']:>5} "
              f"{r['wr']:>5.1f}% {r['ann']:>+8.1f}% {r['dd']:>6.1f}% {r['sh']:>6.2f}")
    if not results: print("  No results."); return

    # 5. Walk-forward top configs
    ba = results[0]
    bs = max(results, key=lambda x: x["sh"])
    br = max(results, key=lambda x: x["ann"]/max(x["dd"],1.0))
    for lb, b in [("BEST-ANN",ba),("BEST-SHARPE",bs),("BEST-RISK",br)]:
        walk_fwd(C,O,H,L,NS,ND,dates,syms,pred_c[b["w"]],ker,pvol,sec_lk,
                 top_n=b["tn"],mps=b["mps"],label=lb)

    # 6. Compare V117 vs V96 baseline
    print(f"\n{'='*70}\n  COMPARISON: V117 vs V96 baseline\n{'='*70}")
    tr117,eq117,dd117 = backtest(C,O,H,L,NS,ND,dates,syms,
        pred_c[ba["w"]],ker,pvol,sec_lk,top_n=ba["tn"],mps=ba["mps"],sdi=bt19)
    # V96 baseline: uniform BMA on llr_w=20 factors
    uw = np.full((N_FACTORS,ND),1.0/N_FACTORS)
    pred96 = nw_predict(raw_c[20], uw, NS, ND, tw=40, bw=1.0)
    tr96,eq96,dd96 = backtest(C,O,H,L,NS,ND,dates,syms,
        pred96,ker,pvol,sec_lk,top_n=2,mps=2,sdi=bt19)
    print("\n  V117 (lead-lag):"); analyze(tr117,eq117,dd117,"V117")
    print("\n  V96 baseline:"); analyze(tr96,eq96,dd96,"V96")
    if tr117 and tr96:
        print(f"\n  Delta: eq={eq117-eq96:+,.0f} dd={dd117-dd96:+.1f}% "
              f"trades={len(tr117)-len(tr96):+d}")
    print(f"\n[V117] Done. {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
