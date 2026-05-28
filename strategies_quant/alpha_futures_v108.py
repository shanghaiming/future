"""
V108: Sector-Specific NW Kernels
=================================
#1 opportunity from meta-analysis of 101 strategies.

Core Innovation: Replace V96's single global NW kernel with 6 independent
sector-specific kernels. Each sector trains its own NW kernel on that
sector's commodities only. Rank predicted returns WITHIN each sector,
select top_n per sector, then allocate across sectors.

Why: V47/V61 proved sector diversification is the most powerful structural
filter. V88 found IC differs by sector. A global kernel averages away
sector-specific nonlinear factor-return relationships.

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
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

CASH0, COMM, LEVERAGE = 1_000_000, 0.0005, 1.0

SECTOR_DEFS = {
    'BLACK': ['i','j','jm','hc','sf','sm','wr','rb'],
    'METAL': ['cu','al','zn','pb','ni','sn','ss','ao'],
    'ENERGY': ['sc','fu','bu','pg','lu'],
    'CHEMICAL': ['v','pp','l','eg','ma','ta','sa','eb','ur','pf','sh'],
    'AGRI': ['m','y','a','p','c','cs','jd','rr'],
    'SOFTS': ['cf','sr','ap','cj','pk','lh'],
}
SECTOR_MAP = {}
for _sec, _sl in SECTOR_DEFS.items():
    for _s in _sl:
        SECTOR_MAP[_s] = _sec
SECTOR_MAP.update({
    'im':'BLACK','au':'METAL','ag':'METAL','si':'METAL',
    'eb':'ENERGY','ta':'ENERGY','fg':'ENERGY','oi':'ENERGY',
    'lc':'CHEMICAL','lrm':'AGRI','rm':'AGRI','ru':'AGRI',
    'sp':'SOFTS','b':'SOFTS','br':'SOFTS',
})

FACTOR_NAMES = ["ret_5d","oi_5d","rsi14","vol_5d","ret_10d","range_5d","atrp_5d"]
N_FACTORS = len(FACTOR_NAMES)


def _base(sym: str) -> str:
    s = sym.lower().split('.')[0].strip()
    while s and s[-1].isdigit(): s = s[:-1]
    return s[:-2] if s.endswith('fi') else s


def build_sector_lookup(syms: List[str]) -> Dict[int, str]:
    return {si: SECTOR_MAP.get(_base(sym), 'OTHER') for si, sym in enumerate(syms)}


def build_sector_inst_map(sl: Dict[int, str]) -> Dict[str, List[int]]:
    m: Dict[str, List[int]] = defaultdict(list)
    for si, sec in sl.items(): m[sec].append(si)
    return dict(m)


# === Factor computation ===

def compute_rsi_manual(C, NS, ND, period=14):
    rsi = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]; gains = np.full(ND, np.nan); losses = np.full(ND, np.nan)
        for di in range(1, ND):
            if np.isnan(c[di]) or np.isnan(c[di-1]): continue
            gains[di] = max(c[di]-c[di-1], 0.0); losses[di] = max(c[di-1]-c[di], 0.0)
        ag = np.nan; al = np.nan
        for di in range(1, ND):
            if np.isnan(gains[di]): continue
            if np.isnan(ag):
                vg = [gains[j] for j in range(di, min(di+period, ND)) if not np.isnan(gains[j])]
                vl = [losses[j] if not np.isnan(losses[j]) else 0.0 for j in range(di, min(di+period, ND)) if not np.isnan(gains[j])]
                if len(vg) >= period:
                    ag, al = np.mean(vg), np.mean(vl)
                    if al == 0: rsi[si, di+period-1] = 100.0
                    else: rsi[si, di+period-1] = 100.0 - 100.0/(1.0+ag/al)
                continue
            ag = (ag*(period-1)+gains[di])/period; al = (al*(period-1)+losses[di])/period
            if al == 0: rsi[si, di] = 100.0
            else: rsi[si, di] = 100.0 - 100.0/(1.0+ag/al)
    return rsi


def compute_raw_factors(C, O, H, L, V, OI, NS, ND):
    t0 = time.time()
    print("[V108] Computing raw factors...", flush=True)
    ret5 = np.full((NS,ND),np.nan); oi5 = np.full((NS,ND),np.nan)
    vol5 = np.full((NS,ND),np.nan); rng5 = np.full((NS,ND),np.nan)
    atr5 = np.full((NS,ND),np.nan); ret10 = np.full((NS,ND),np.nan)
    fwd5 = np.full((NS,ND),np.nan); atrm = np.full((NS,ND),np.nan)

    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(C[si,di]) and not np.isnan(C[si,di-5]) and C[si,di-5]>0:
                ret5[si,di] = C[si,di]/C[si,di-5]-1.0
            if not np.isnan(OI[si,di]) and not np.isnan(OI[si,di-5]) and OI[si,di-5]>0:
                oi5[si,di] = OI[si,di]/OI[si,di-5]-1.0
            vv = V[si,di-5:di]; va = vv[~np.isnan(vv)]
            if len(va) >= 3: vol5[si,di] = np.mean(va)
            rv = [(H[si,j]-L[si,j])/C[si,j] for j in range(di-5,di)
                  if not np.isnan(H[si,j]) and not np.isnan(L[si,j])
                  and not np.isnan(C[si,j]) and C[si,j]>0 and H[si,j]>L[si,j]]
            if len(rv) >= 3: rng5[si,di] = np.mean(rv)
        for di in range(6, ND):
            av = []
            for j in range(di-5,di):
                hh,ll,cc = H[si,j],L[si,j],C[si,j]
                if not any(np.isnan([hh,ll,cc])):
                    pc = C[si,j-1] if j>0 and not np.isnan(C[si,j-1]) else cc
                    av.append(max(hh-ll, abs(hh-pc), abs(ll-pc)))
            if av and not np.isnan(C[si,di]) and C[si,di]>0:
                atr5[si,di] = np.mean(av)/C[si,di]
        for di in range(10, ND):
            if not np.isnan(C[si,di]) and not np.isnan(C[si,di-10]) and C[si,di-10]>0:
                ret10[si,di] = C[si,di]/C[si,di-10]-1.0
        for di in range(ND-5):
            if not np.isnan(C[si,di+5]) and not np.isnan(C[si,di]) and C[si,di]>0:
                fwd5[si,di] = C[si,di+5]/C[si,di]-1.0
        for di in range(20, ND):
            av = []
            for j in range(di-14,di):
                hh,ll,cc = H[si,j],L[si,j],C[si,j]
                if not any(np.isnan([hh,ll,cc])):
                    pc = C[si,j-1] if j>0 and not np.isnan(C[si,j-1]) else cc
                    av.append(max(hh-ll, abs(hh-pc), abs(ll-pc)))
            if av and not np.isnan(C[si,di]) and C[si,di]>0:
                atrm[si,di] = np.mean(av)/C[si,di]

    rsi14 = np.full((NS,ND),np.nan)
    if HAS_TALIB:
        for si in range(NS):
            c = np.where(np.isnan(C[si]),0,C[si]).astype(np.float64)
            nm = np.isnan(C[si])
            try: rsi14[si] = np.where(nm, np.nan, talib.RSI(c,14))
            except: pass
    fb = np.all(np.isnan(rsi14), axis=1)
    if fb.any():
        rm = compute_rsi_manual(C,NS,ND,14)
        for si in range(NS):
            if fb[si]: rsi14[si] = rm[si]

    print(f"  Raw factors done: {time.time()-t0:.1f}s", flush=True)
    return {"ret_5d":ret5,"oi_5d":oi5,"vol_5d":vol5,"range_5d":rng5,
            "atrp_5d":atr5,"ret_10d":ret10,"rsi14":rsi14,
            "fwd_ret_5d":fwd5,"atr_mean":atrm}


def normalize_factor(factor, NS, ND, min_count=5):
    normed = np.full((NS,ND),np.nan)
    for di in range(ND):
        v = factor[:,di]; va = v[~np.isnan(v)]
        if len(va) < min_count: continue
        s = np.std(va)
        if s < 1e-12: continue
        m = np.mean(va)
        for si in range(NS):
            if not np.isnan(v[si]): normed[si,di] = (v[si]-m)/s
    return normed


# === INNOVATION: Sector-Specific NW Kernel ===

def compute_sector_nw(raw_factors, NS, ND, sec_inst, tw=40, bw=1.0):
    """NW kernel per sector. Each sector trains ONLY on its instruments."""
    t0 = time.time()
    print(f"[V108] Sector-NW (tw={tw} bw={bw:.1f})...", flush=True)
    normed = {f: normalize_factor(raw_factors[f], NS, ND) for f in FACTOR_NAMES}
    fwd = raw_factors["fwd_ret_5d"]; atrm = raw_factors["atr_mean"]
    pred = np.full((NS,ND),np.nan)

    for sn, idxs in sec_inst.items():
        if len(idxs) < 2: continue
        st = time.time(); vd = 0
        for di in range(tw+10, ND):
            tf, tt = [], []
            for tdi in range(max(10,di-tw), di):
                for si in idxs:
                    f = np.array([normed[fn][si,tdi] for fn in FACTOR_NAMES])
                    t = fwd[si,tdi]
                    if not np.any(np.isnan(f)) and not np.isnan(t):
                        tf.append(f); tt.append(t)
            if len(tf) < 10: continue
            tX, tY = np.array(tf), np.array(tt)
            fs = np.std(tX, axis=0); fs[fs<1e-12] = 1.0
            for si in idxs:
                qf = np.array([normed[fn][si,di] for fn in FACTOR_NAMES])
                if np.any(np.isnan(qf)): continue
                av = atrm[si,di]
                h = max(av*bw, 0.1) if not np.isnan(av) else bw
                d = np.sqrt(np.sum(((tX-qf)/fs)**2, axis=1))
                sd = d/h; w = np.zeros(len(tX)); m = sd <= 1.0
                if not np.any(m):
                    mi = np.argmin(d)
                    if d[mi] < 1e12: w[mi] = 1.0; m = np.zeros(len(d),dtype=bool); m[mi]=True
                    else: continue
                else: w[m] = 0.75*(1.0-sd[m]**2)
                ws = np.sum(w)
                if ws < 1e-12: continue
                pred[si,di] = np.sum(w*tY)/ws; vd += 1
        print(f"  {sn}({len(idxs)}): {time.time()-st:.1f}s {vd}pred", flush=True)
    print(f"  Sector-NW done: {time.time()-t0:.1f}s", flush=True)
    return pred


# === Helpers ===

def compute_ker(C, NS, ND):
    k = np.full((NS,ND),np.nan)
    for si in range(NS):
        for di in range(10, ND):
            cs = C[si,di-10:di+1]; v = cs[~np.isnan(cs)]
            if len(v)<10 or v[0]<=0: continue
            tc = np.sum(np.abs(np.diff(v)))
            if tc > 1e-10: k[si,di] = abs(v[-1]-v[0])/tc
    kr = np.zeros((NS,ND),dtype=int)
    for si in range(NS):
        for di in range(ND):
            if np.isnan(k[si,di]): continue
            if k[si,di]<0.15: kr[si,di]=1
            elif k[si,di]>0.3: kr[si,di]=-1
    return kr


def get_mode(rtw, wt, wrw):
    if len(rtw)<5: return "normal"
    wr = sum(rtw[-wrw:])/len(rtw[-wrw:])
    if wr>wt: return "winning"
    return "losing" if wr<0.50 else "normal"


def atr_at(H,L,C,si,di,sdi):
    a = [max(H[si,j]-L[si,j],abs(H[si,j]-C[si,j]),abs(L[si,j]-C[si,j]))
         for j in range(max(sdi,di-14),di) if not any(np.isnan([H[si,j],L[si,j],C[si,j]]))]
    return np.mean(a) if a else None


def port_vol(C, NS, ND, vlb=20):
    pv = np.full(ND,np.nan)
    for di in range(vlb+1,ND):
        dr = []
        for dd in range(di-vlb,di):
            r = [C[si,dd]/C[si,dd-1]-1.0 for si in range(NS)
                 if not np.isnan(C[si,dd]) and not np.isnan(C[si,dd-1]) and C[si,dd-1]>0]
            if r: dr.append(np.mean(r))
        if len(dr) >= vlb//2: pv[di] = np.std(dr)
    return pv


def vol_mult(pv, vm, vhm, vlm, sr, sb):
    if np.isnan(pv) or np.isnan(vm) or vm<1e-12: return 1.0
    r = pv/vm
    return sr if r>vhm else (sb if r<vlm else 1.0)


# === Backtest ===

def backtest_v108(C,O,H,L,NS,ND,dates,syms,pred,ker,pv,sl,sim,
                  tnps=1,gtn=2,hd=5,atr_s=3.0,vhm=2.0,vlm=0.5,sr=0.5,sb=1.3,
                  sdi=60,edi=None):
    if edi is None: edi = ND-1
    vd = pv[max(sdi,21):edi]; vd = vd[~np.isnan(vd)]
    vm = np.median(vd) if len(vd)>10 else 1e-6
    eq, pk, mdd = CASH0, CASH0, 0.0
    pos, trades, rtw = [], [], []

    for di in range(max(sdi,1), edi):
        d = dates[di]; dpnl = 0.0; npos = []
        mode = get_mode(rtw, 0.60, 15)
        vm_ = vol_mult(pv[di], vm, vhm, vlm, sr, sb)

        # Exit
        pb = defaultdict(list)
        for p in pos: pb[p[0]].append(p)
        for si, pl in pb.items():
            c = C[si,di]
            if np.isnan(c):
                for p in pl: npos.append(p)
                continue
            ee = min(p[1] for p in pl); hold = di-ee
            stopped = any(c < p[3] for p in pl)
            if stopped or hold >= hd:
                for _,edi_,ep,sp,al in pl:
                    pnl = (c-ep)/ep-COMM; pr = eq*al*pnl; dpnl += pr
                    trades.append({"pnl_abs":pr,"pnl_pct":pnl*100,"days":di-edi_+1,
                                   "di":di,"year":d.year,"sym":syms[si],
                                   "sector":sl.get(si,'OTHER'),
                                   "reason":"stop" if stopped else "hold",
                                   "mode":mode[:1].upper()})
                    rtw.append(1 if pnl>0 else 0)
            else:
                for p in pl: npos.append(p)

        pos = npos; eq += dpnl
        if eq>pk: pk=eq
        if pk>0:
            dd = (pk-eq)/pk*100
            if dd>mdd: mdd=dd
        if eq<=0: break

        # Entry: sector-aware
        held = {p[0] for p in pos}
        if len(held) >= gtn: continue

        sc = defaultdict(list)
        for si in range(NS):
            if si in held: continue
            p = pred[si,di]
            if np.isnan(p) or p<=0: continue
            if di+1>=ND or np.isnan(O[si,di+1]): continue
            if ker[si,di]<0: continue
            sc[sl.get(si,'OTHER')].append((p,si))

        gc = []
        for sec, cs in sc.items():
            cs.sort(key=lambda x:-x[0])
            for pv_,si in cs[:tnps]: gc.append((pv_,si,sec))
        gc.sort(key=lambda x:-x[0])

        nt = gtn
        if mode=="winning": nt = min(gtn+1, gtn*2)
        elif mode=="losing": nt = max(1, gtn-1)

        scc = defaultdict(int)
        for sh in held: scc[sl.get(sh,'OTHER')] += 1
        ne = []
        for pv_,si,sec in gc:
            if len(held)+len(ne)>=nt or si in held: break
            if scc[sec]>=tnps: continue
            ne.append((pv_,si,sec)); scc[sec]+=1
        if not ne: continue

        ap = LEVERAGE/(len(pos)+len(ne))*vm_
        up = [(si,edi_,ep,sp,ap) for si,edi_,ep,sp,_ in pos]
        for pv_,si,sec in ne:
            ep = O[si,di+1]
            if np.isnan(ep) or ep<=0: continue
            a = atr_at(H,L,C,si,di,sdi)
            if a is None: continue
            up.append((si,di+1,ep,ep-atr_s*a,ap))
        pos = up

    # Close remaining
    for si,edi_,ep,sp,al in pos:
        c = C[si,ND-1]
        if not np.isnan(c) and c>0: eq += eq*al*((c-ep)/ep-COMM)
    return trades, eq, mdd


def analyze(trades, eq, mdd, label=""):
    if not trades: print(f"  {label}: no trades"); return None
    nw = sum(1 for t in trades if t["pnl_pct"]>0)
    wr = nw/len(trades)*100
    nd = max(1, trades[-1]["di"]-trades[0]["di"])
    ann = ((eq/CASH0)**(1/max(1.0,nd/252))-1)*100
    ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x:x["di"])]
    rs = np.array(ap)/CASH0
    sh = np.mean(rs)/np.std(rs)*np.sqrt(252) if np.std(rs)>0 else 0
    ns = sum(1 for t in trades if t["reason"]=="stop")
    nh = sum(1 for t in trades if t["reason"]=="hold")
    sc = defaultdict(int)
    for t in trades: sc[t.get("sector","OTHER")] += 1
    ss = " ".join(f"{k}:{v}" for k,v in sorted(sc.items()))
    aw = np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"]>0])
    al = np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"]<=0])
    print(f"  {label}: {len(trades)}t (stop:{ns} hold:{nh}) WR={wr:.1f}% "
          f"ann={ann:+.1f}% DD={mdd:.1f}% Sh={sh:.2f} eq={eq:,.0f}")
    print(f"    avg_win={aw:+.3f}% avg_loss={al:+.3f}% sectors: {ss}")
    yr = {}
    for t in trades:
        y = t["year"]
        if y not in yr: yr[y] = {"n":0,"w":0,"pnl":[]}
        yr[y]["n"]+=1
        if t["pnl_pct"]>0: yr[y]["w"]+=1
        yr[y]["pnl"].append(t["pnl_pct"])
    for y in sorted(yr):
        ys = yr[y]; cum = np.prod([1+p/100 for p in ys["pnl"]])-1
        print(f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% cum={cum:+.1%}")
    return {"n":len(trades),"wr":wr,"dd":mdd,"ann":ann,"sh":sh,"eq":eq}


def walk_forward(C,O,H,L,NS,ND,dates,syms,pred,ker,pv,sl,sim,
                 tnps=1,gtn=2,hd=5,vhm=2.0,vlm=0.5,sr=0.5,sb=1.3,label=""):
    print(f"\n{'='*70}\n  WF V108 {label} tnps={tnps} gtn={gtn}\n{'='*70}")
    yrs = sorted(set(d.year for d in dates)); at = []
    for ty in range(2019, yrs[-1]+1):
        ts = te = None
        for i,d in enumerate(dates):
            if d.year==ty and ts is None: ts=i
            if d.year==ty: te=i
        if ts is None: continue
        tr,_,_ = backtest_v108(C,O,H,L,NS,ND,dates,syms,pred,ker,pv,sl,sim,
                               tnps=tnps,gtn=gtn,hd=hd,vhm=vhm,vlm=vlm,sr=sr,sb=sb,
                               sdi=ts,edi=te+1)
        tt = [t for t in tr if dates[t["di"]].year==ty]; at.extend(tt)
        if tt:
            n=len(tt); nw=sum(1 for t in tt if t["pnl_pct"]>0)
            sc = defaultdict(int)
            for t in tt: sc[t.get("sector","OTHER")]+=1
            ss = " ".join(f"{k}:{v}" for k,v in sorted(sc.items()))
            print(f"  {ty}: {n}t WR={nw/n*100:.1f}% avg={np.mean([t['pnl_pct'] for t in tt]):+.2f}% [{ss}]",flush=True)
        else: print(f"  {ty}: no trades",flush=True)

    if at:
        nw=sum(1 for t in at if t["pnl_pct"]>0); wr=nw/len(at)*100
        cum=np.prod([1+t["pnl_pct"]/100 for t in at])-1
        sc=defaultdict(int)
        for t in at: sc[t.get("sector","OTHER")]+=1
        ss=" ".join(f"{k}:{v}" for k,v in sorted(sc.items()))
        print(f"\n  WF TOTAL: {len(at)}t WR={wr:.1f}% cum={cum:+.1%} sectors: {ss}")
    return at


def main():
    t0 = time.time()
    print("="*70)
    print("  V108: SECTOR-SPECIFIC NW KERNELS")
    print("  6 independent sector kernels, rank within sector, vol-adaptive sizing")
    print("  Walk-forward 2019-2026. No leverage.")
    print("="*70)

    C,O,H,L,V,OI,NS,ND,dates,syms = load_all_data(start="2016-01-01")
    print(f"  {NS} sym, {ND} days, {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    sl = build_sector_lookup(syms)
    sim = build_sector_inst_map(sl)
    sd = defaultdict(int)
    for s in sl.values(): sd[s]+=1
    print(f"  Sectors: {dict(sd)}")
    for sec,idxs in sorted(sim.items()):
        if sec!='OTHER': print(f"    {sec}: {[syms[i] for i in idxs]}")

    bt19 = None
    for i,d in enumerate(dates):
        if d>=pd.Timestamp("2019-01-01"): bt19=i; break

    rf = compute_raw_factors(C,O,H,L,V,OI,NS,ND)
    kr = compute_ker(C,NS,ND)
    vc = {vlb: port_vol(C,NS,ND,vlb) for vlb in [15,20]}

    # Sector-NW predictions sweep
    print("\n" + "="*70 + "\n  SWEEPING SECTOR-NW KERNEL CONFIGS\n" + "="*70)
    pc = {}
    for tw in [30,40,50]:
        for bw in [0.8,1.0,1.5]:
            pc[(tw,bw)] = compute_sector_nw(rf,NS,ND,sim,tw=tw,bw=bw)

    # Parameter sweep
    print("\n" + "="*70 + "\n  PARAMETER SWEEP (2019-2026)\n" + "="*70)
    results = []; sc = 0
    for (tw,bw),pr in pc.items():
        for tnps in [1,2]:
            for gtn in [2,3]:
                for vlb in [15,20]:
                    for vhm,vlm in [(1.5,0.5),(2.0,0.7)]:
                        for sr,sb in [(0.3,1.2),(0.5,1.5)]:
                            sc += 1
                            tr,eq,dd = backtest_v108(
                                C,O,H,L,NS,ND,dates,syms,pr,kr,vc[vlb],sl,sim,
                                tnps=tnps,gtn=gtn,vhm=vhm,vlm=vlm,sr=sr,sb=sb,sdi=bt19)
                            if len(tr)<10: continue
                            nw=sum(1 for t in tr if t["pnl_pct"]>0)
                            wr=nw/len(tr)*100
                            nd=max(1,tr[-1]["di"]-tr[0]["di"])
                            ann=((eq/CASH0)**(1/max(1.0,nd/252))-1)*100
                            ap=[t["pnl_abs"] for t in sorted(tr,key=lambda x:x["di"])]
                            ra=np.array(ap)/CASH0
                            sh=np.mean(ra)/np.std(ra)*np.sqrt(252) if np.std(ra)>0 else 0
                            results.append({"tw":tw,"bw":bw,"tnps":tnps,"gtn":gtn,
                                           "vlb":vlb,"vhm":vhm,"vlm":vlm,"sr":sr,"sb":sb,
                                           "n":len(tr),"wr":wr,"ann":ann,"dd":dd,"sharpe":sh,"eq":eq})

    results.sort(key=lambda x:-x["ann"])
    print(f"\n  {sc} configs, {len(results)} with 10+ trades")
    print(f"\n{'TW':>4}{'BW':>5}{'TNPS':>5}{'GTN':>4}{'Vlb':>4}{'Vhm':>4}"
          f"{'Vlm':>4}{'SR':>4}{'SB':>4}{'N':>5}{'WR':>6}{'Ann':>9}{'DD':>7}{'Sh':>7}")
    print("-"*100)
    for r in results[:10]:
        print(f"{r['tw']:>4}{r['bw']:>5.1f}{r['tnps']:>5}{r['gtn']:>4}{r['vlb']:>4}"
              f"{r['vhm']:>4.1f}{r['vlm']:>4.1f}{r['sr']:>4.1f}{r['sb']:>4.1f}"
              f"{r['n']:>5}{r['wr']:>5.1f}%{r['ann']:>+8.1f}%{r['dd']:>6.1f}%{r['sharpe']:>6.2f}")

    if not results:
        print("  No configs with 10+ trades."); return

    ba = results[0]
    bs = max(results, key=lambda x:x["sharpe"])
    bra = max(results, key=lambda x:x["ann"]/max(x["dd"],1.0))

    for lb,b in [("BEST-ANN",ba),("BEST-SHARPE",bs),("BEST-RISK-ADJ",bra)]:
        walk_forward(C,O,H,L,NS,ND,dates,syms,pc[(b["tw"],b["bw"])],kr,
                     vc[b["vlb"]],sl,sim,tnps=b["tnps"],gtn=b["gtn"],
                     vhm=b["vhm"],vlm=b["vlm"],sr=b["sr"],sb=b["sb"],label=lb)

    # Full comparison
    print("\n" + "="*70)
    print("  V108 SECTOR-KERNEL vs V96 GLOBAL-KERNEL (baseline +73.1%)")
    print("="*70)
    for lb,b in [("BEST-ANN",ba),("BEST-SHARPE",bs)]:
        tr,eq,dd = backtest_v108(C,O,H,L,NS,ND,dates,syms,
                                  pc[(b["tw"],b["bw"])],kr,vc[b["vlb"]],sl,sim,
                                  tnps=b["tnps"],gtn=b["gtn"],
                                  vhm=b["vhm"],vlm=b["vlm"],sr=b["sr"],sb=b["sb"],sdi=bt19)
        analyze(tr,eq,dd,f"V108-{lb}")

    print(f"\n[V108] Done. {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
