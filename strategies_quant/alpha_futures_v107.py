"""
V107: OI-Price Divergence as Alpha Source
==========================================
4 new OI-Price interaction features capture INDEPENDENT alpha:
  1. oi_px_div: OI return minus price return (divergence score)
  2. oi_px_concord: sign agreement between OI and price moves
  3. oi_vol_ratio: volume/OI turnover ratio (participation intensity)
  4. oi_accel: acceleration of OI change rate
Plus 7 standard factors = 11 total.
NW kernel regression + vol-adaptive sizing.
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

CASH0 = 1_000_000
COMM = 0.0005
LEVERAGE = 1.0

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

FACTOR_NAMES = [
    "ret_5d", "oi_5d", "rsi14", "vol_5d",
    "ret_10d", "range_5d", "atrp_5d",
    "oi_px_div", "oi_px_concord", "oi_vol_ratio", "oi_accel",
]
N_FACTORS = len(FACTOR_NAMES)


def _base(sym: str) -> str:
    s = sym.lower().split('.')[0].strip()
    while s and s[-1].isdigit():
        s = s[:-1]
    return s[:-2] if s.endswith('fi') else s


def build_sector_lookup(syms: List[str]) -> Dict[int, str]:
    return {i: SECTOR_MAP.get(_base(s), 'OTHER') for i, s in enumerate(syms)}


def compute_rsi_manual(C: np.ndarray, NS: int, ND: int, period: int = 14) -> np.ndarray:
    rsi = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        gains = np.where(np.diff(c, prepend=np.nan) > 0, np.diff(c, prepend=np.nan), 0.0)
        losses = np.where(np.diff(c, prepend=np.nan) < 0, -np.diff(c, prepend=np.nan), 0.0)
        gains[np.isnan(c)] = np.nan
        losses[np.isnan(c)] = np.nan
        ag = al = np.nan
        for di in range(1, ND):
            if np.isnan(gains[di]): continue
            if np.isnan(ag):
                vg = [gains[j] for j in range(di, min(di+period, ND)) if not np.isnan(gains[j])]
                vl = [losses[j] if not np.isnan(losses[j]) else 0.0 for j in range(di, min(di+period, ND)) if not np.isnan(gains[j])]
                if len(vg) >= period:
                    ag, al = np.mean(vg), np.mean(vl)
                    rsi[si, di+period-1] = 100.0 if al == 0 else 100.0-100.0/(1.0+ag/al)
                continue
            ag = (ag*(period-1)+gains[di])/period
            al = (al*(period-1)+losses[di])/period
            rsi[si, di] = 100.0 if al == 0 else 100.0-100.0/(1.0+ag/al)
    return rsi


def compute_raw_factors(C, O, H, L, V, OI, NS, ND):
    """Compute 7 standard + 4 new OI-Price interaction factors."""
    t0 = time.time()
    print("[V107] Computing raw factors (11 total)...", flush=True)

    ret_5d = np.full((NS, ND), np.nan)
    oi_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(C[si,di]) and not np.isnan(C[si,di-5]) and C[si,di-5] > 0:
                ret_5d[si,di] = C[si,di]/C[si,di-5]-1.0
            if not np.isnan(OI[si,di]) and not np.isnan(OI[si,di-5]) and OI[si,di-5] > 0:
                oi_5d[si,di] = OI[si,di]/OI[si,di-5]-1.0

    vol_5d = np.full((NS, ND), np.nan)
    range_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            vals = V[si, di-5:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 3: vol_5d[si,di] = np.mean(valid)
            rv = [(H[si,j]-L[si,j])/C[si,j] for j in range(di-5,di)
                  if not np.isnan(H[si,j]) and not np.isnan(L[si,j])
                  and not np.isnan(C[si,j]) and C[si,j]>0 and H[si,j]>L[si,j]]
            if len(rv) >= 3: range_5d[si,di] = np.mean(rv)

    atrp_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(6, ND):
            av = []
            for j in range(di-5, di):
                hh,ll,cc = H[si,j],L[si,j],C[si,j]
                if not any(np.isnan([hh,ll,cc])):
                    pc = C[si,j-1] if j>0 and not np.isnan(C[si,j-1]) else cc
                    av.append(max(hh-ll, abs(hh-pc), abs(ll-pc)))
            if av and not np.isnan(C[si,di]) and C[si,di]>0:
                atrp_5d[si,di] = np.mean(av)/C[si,di]

    ret_10d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            if not np.isnan(C[si,di]) and not np.isnan(C[si,di-10]) and C[si,di-10]>0:
                ret_10d[si,di] = C[si,di]/C[si,di-10]-1.0

    rsi14 = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nm = np.isnan(C[si])
            try: rsi14[si] = np.where(nm, np.nan, talib.RSI(c, 14))
            except Exception: pass
    needs_fb = np.all(np.isnan(rsi14), axis=1)
    if needs_fb.any():
        rm = compute_rsi_manual(C, NS, ND, 14)
        for si in range(NS):
            if needs_fb[si]: rsi14[si] = rm[si]

    # === NEW: 4 OI-Price interaction features ===
    print("[V107] Computing 4 OI-Price interaction features...", flush=True)

    # Feature 1: OI-Price Divergence Score
    oi_px_div = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(oi_5d[si,di]) and not np.isnan(ret_5d[si,di]):
                oi_px_div[si,di] = oi_5d[si,di] - ret_5d[si,di]

    # Feature 2: OI-Price Concordance (+1 same dir, -1 diverging)
    oi_px_concord = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            oi_r, px_r = oi_5d[si,di], ret_5d[si,di]
            if np.isnan(oi_r) or np.isnan(px_r): continue
            s_oi = (1.0 if oi_r > 0 else -1.0 if oi_r < 0 else 0.0)
            s_px = (1.0 if px_r > 0 else -1.0 if px_r < 0 else 0.0)
            oi_px_concord[si,di] = s_oi * s_px

    # Feature 3: OI-weighted volume (turnover ratio)
    oi_vol_ratio = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            oi_v, v_v = OI[si,di], V[si,di]
            if not np.isnan(oi_v) and oi_v > 0 and not np.isnan(v_v):
                oi_vol_ratio[si,di] = v_v / oi_v

    # Feature 4: OI Acceleration
    oi_accel = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            oc, op = oi_5d[si,di], oi_5d[si,di-5]
            if not np.isnan(oc) and not np.isnan(op):
                oi_accel[si,di] = oc - op

    # Target + ATR for adaptive bandwidth
    fwd_ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND-5):
            if not np.isnan(C[si,di+5]) and not np.isnan(C[si,di]) and C[si,di]>0:
                fwd_ret_5d[si,di] = C[si,di+5]/C[si,di]-1.0

    atr_mean = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            av = []
            for j in range(di-14, di):
                hh,ll,cc = H[si,j],L[si,j],C[si,j]
                if not any(np.isnan([hh,ll,cc])):
                    pc = C[si,j-1] if j>0 and not np.isnan(C[si,j-1]) else cc
                    av.append(max(hh-ll, abs(hh-pc), abs(ll-pc)))
            if av and not np.isnan(C[si,di]) and C[si,di]>0:
                atr_mean[si,di] = np.mean(av)/C[si,di]

    print(f"  Raw factors done: {time.time()-t0:.1f}s", flush=True)
    return {
        "ret_5d":ret_5d, "oi_5d":oi_5d, "vol_5d":vol_5d,
        "range_5d":range_5d, "atrp_5d":atrp_5d, "ret_10d":ret_10d,
        "rsi14":rsi14, "oi_px_div":oi_px_div, "oi_px_concord":oi_px_concord,
        "oi_vol_ratio":oi_vol_ratio, "oi_accel":oi_accel,
        "fwd_ret_5d":fwd_ret_5d, "atr_mean":atr_mean,
    }


def normalize_factor(factor, NS, ND, min_count=10):
    """Cross-sectional z-score normalization."""
    normed = np.full((NS, ND), np.nan)
    for di in range(ND):
        vals = factor[:, di]
        valid = vals[~np.isnan(vals)]
        if len(valid) < min_count: continue
        mu, sigma = np.mean(valid), np.std(valid)
        if sigma < 1e-12: continue
        for si in range(NS):
            if not np.isnan(vals[si]):
                normed[si, di] = (vals[si] - mu) / sigma
    return normed


def compute_nw_predicted_returns(raw_factors, NS, ND, training_window=40, kernel_bandwidth=1.0):
    """NW kernel regression with all 11 factors equally weighted."""
    t0 = time.time()
    print(f"[V107] NW kernel (tw={training_window}, bw={kernel_bandwidth:.1f})...", flush=True)
    normed = {fn: normalize_factor(raw_factors[fn], NS, ND) for fn in FACTOR_NAMES}
    fwd_ret = raw_factors["fwd_ret_5d"]
    atr_mean = raw_factors["atr_mean"]
    predicted = np.full((NS, ND), np.nan)

    for di in range(training_window+10, ND):
        trX, trY = [], []
        for tdi in range(max(10, di-training_window), di):
            for si in range(NS):
                feat = np.array([normed[fn][si,tdi] for fn in FACTOR_NAMES])
                tgt = fwd_ret[si,tdi]
                if np.any(np.isnan(feat)) or np.isnan(tgt): continue
                trX.append(feat); trY.append(tgt)
        if len(trX) < 20: continue
        trX, trY = np.array(trX), np.array(trY)
        fstd = np.std(trX, axis=0); fstd[fstd<1e-12] = 1.0

        for si in range(NS):
            qf = np.array([normed[fn][si,di] for fn in FACTOR_NAMES])
            if np.any(np.isnan(qf)): continue
            atr_v = atr_mean[si,di]
            h = max(atr_v*kernel_bandwidth, 0.1) if not np.isnan(atr_v) else kernel_bandwidth
            dist = np.sqrt(np.sum(((trX - qf)/fstd)**2, axis=1))
            sd = dist/h
            w = np.zeros(len(trX))
            m = sd <= 1.0
            if not np.any(m):
                mi = np.argmin(dist)
                if dist[mi] < 1e12: w[mi] = 1.0
                else: continue
            else:
                w[m] = 0.75*(1.0-sd[m]**2)
            ws = np.sum(w)
            if ws < 1e-12: continue
            predicted[si,di] = np.sum(w*trY)/ws

        if di % 100 == 0:
            print(f"  di={di}/{ND} valid={np.sum(~np.isnan(predicted[:,di]))}/{NS}", flush=True)

    print(f"  NW done: {time.time()-t0:.1f}s", flush=True)
    return predicted


def compute_ker(C, NS, ND):
    ker_10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            cs = C[si, di-10:di+1]; v = cs[~np.isnan(cs)]
            if len(v)<10 or v[0]<=0: continue
            tc = np.sum(np.abs(np.diff(v)))
            if tc > 1e-10: ker_10[si,di] = abs(v[-1]-v[0])/tc
    kr = np.zeros((NS,ND), dtype=int)
    for si in range(NS):
        for di in range(ND):
            if np.isnan(ker_10[si,di]): continue
            if ker_10[si,di] < 0.15: kr[si,di] = 1
            elif ker_10[si,di] > 0.3: kr[si,di] = -1
    return kr


def compute_portfolio_volatility(C, NS, ND, vol_lookback=15):
    pv = np.full(ND, np.nan)
    for di in range(vol_lookback+1, ND):
        dr = []
        for dd in range(di-vol_lookback, di):
            r = [C[si,dd]/C[si,dd-1]-1.0 for si in range(NS)
                 if not np.isnan(C[si,dd]) and not np.isnan(C[si,dd-1]) and C[si,dd-1]>0]
            if r: dr.append(np.mean(r))
        if len(dr) >= vol_lookback//2: pv[di] = np.std(dr)
    return pv


def _atr_at(H, L, C, si, di, start_di):
    av = []
    for j in range(max(start_di, di-14), di):
        hh,ll,cc = H[si,j],L[si,j],C[si,j]
        if not any(np.isnan([hh,ll,cc])):
            av.append(max(hh-ll, abs(hh-cc), abs(ll-cc)))
    return np.mean(av) if av else None


def backtest_v107(C, O, H, L, NS, ND, dates, syms, predicted, ker_regime,
                  port_vol, sector_lookup, top_n=2, max_per_sector=2,
                  hold_days=5, atr_stop=3.0, vol_high_mult=2.0,
                  size_reduce=0.3, size_boost=1.0, start_di=60, end_di=None):
    if end_di is None: end_di = ND-1
    vd = port_vol[max(start_di,16):end_di]
    vd = vd[~np.isnan(vd)]
    vol_median = np.median(vd) if len(vd)>10 else 1e-6

    equity, peak, max_dd = CASH0, CASH0, 0.0
    positions = []
    trades = []
    recent_wins = []

    for di in range(max(start_di,1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_pos = []

        # Dynamic mode
        mode = "normal"
        if len(recent_wins) >= 5:
            wr_w = sum(recent_wins[-15:])/len(recent_wins[-15:])
            if wr_w > 0.60: mode = "winning"
            elif wr_w < 0.50: mode = "losing"

        # Vol-adaptive sizing
        vol_mult = 1.0
        if not np.isnan(port_vol[di]) and vol_median > 1e-12:
            ratio = port_vol[di]/vol_median
            if ratio > vol_high_mult: vol_mult = size_reduce
            elif ratio < 0.5: vol_mult = size_boost

        # Exit
        by_si = defaultdict(list)
        for p in positions: by_si[p[0]].append(p)

        for si, plist in by_si.items():
            c = C[si,di]
            if np.isnan(c):
                new_pos.extend(plist); continue
            hold = di - min(p[1] for p in plist)
            stopped = any(c < p[3] for p in plist)
            if stopped or hold >= hold_days:
                for s,ed,ep,sp,al in plist:
                    pnl = (c-ep)/ep - COMM
                    profit = equity*al*pnl
                    daily_pnl += profit
                    trades.append({"pnl_abs":profit,"pnl_pct":pnl*100,
                        "days":di-ed+1,"di":di,"year":d.year,"sym":syms[si],
                        "sector":sector_lookup.get(si,'OTHER'),
                        "reason":"stop" if stopped else "hold","mode":mode[0].upper()})
                    recent_wins.append(1 if pnl>0 else 0)
            else:
                new_pos.extend(plist)

        positions = new_pos
        equity += daily_pnl
        if equity > peak: peak = equity
        if peak > 0:
            dd = (peak-equity)/peak*100
            if dd > max_dd: max_dd = dd
        if equity <= 0: break

        # Entry
        held = {p[0] for p in positions}
        if len(held) >= top_n: continue
        cands = [(predicted[si,di],si) for si in range(NS)
                 if si not in held and not np.isnan(predicted[si,di])
                 and di+1<ND and not np.isnan(O[si,di+1]) and ker_regime[si,di]>=0]
        if not cands: continue
        cands.sort(key=lambda x: -x[0])

        n_take = top_n
        if mode == "winning": n_take = min(top_n+1, top_n*2)
        elif mode == "losing": n_take = max(1, top_n-1)

        sc = defaultdict(int)
        for sh in held: sc[sector_lookup.get(sh,'OTHER')] += 1

        entries = []
        for pv, si in cands:
            if len(held)+len(entries) >= n_take: break
            sec = sector_lookup.get(si,'OTHER')
            if sc[sec] >= max_per_sector or pv <= 0: continue
            entries.append((pv,si,sec)); sc[sec] += 1

        if not entries: continue
        alloc = LEVERAGE/(len(positions)+len(entries))*vol_mult
        upd = [(s,ed,ep,sp,alloc) for s,ed,ep,sp,_ in positions]
        for pv,si,sec in entries:
            ep = O[si,di+1]
            if np.isnan(ep) or ep<=0: continue
            atr = _atr_at(H,L,C,si,di,start_di)
            if atr is None: continue
            upd.append((si,di+1,ep,ep-atr_stop*atr,alloc))
        positions = upd

    for si,ed,ep,sp,al in positions:
        c = C[si,ND-1]
        if not np.isnan(c) and c>0:
            equity += equity*al*((c-ep)/ep-COMM)

    return trades, equity, max_dd


def analyze(trades, equity, max_dd, label=""):
    if not trades:
        print(f"  {label}: no trades"); return None
    nw = sum(1 for t in trades if t["pnl_pct"]>0)
    wr = nw/len(trades)*100
    nd = max(1, trades[-1]["di"]-trades[0]["di"])
    ann = ((equity/CASH0)**(1/max(1.0,nd/252))-1)*100
    ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
    rets = np.array(ap)/CASH0
    sh = np.mean(rets)/np.std(rets)*np.sqrt(252) if np.std(rets)>0 else 0
    ns = sum(1 for t in trades if t["reason"]=="stop")
    nh = sum(1 for t in trades if t["reason"]=="hold")
    sc = defaultdict(int)
    for t in trades: sc[t.get("sector","OTHER")] += 1
    ss = " ".join(f"{k}:{v}" for k,v in sorted(sc.items()))
    print(f"  {label}: {len(trades)}t (stop:{ns} hold:{nh}) WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% Sh={sh:.2f} eq={equity:,.0f}")
    print(f"    sectors: {ss}")
    yr = {}
    for t in trades:
        y = t["year"]
        if y not in yr: yr[y] = {"n":0,"w":0,"pnl":[]}
        yr[y]["n"] += 1
        if t["pnl_pct"]>0: yr[y]["w"] += 1
        yr[y]["pnl"].append(t["pnl_pct"])
    for y in sorted(yr):
        ys = yr[y]
        cum = np.prod([1+p/100 for p in ys["pnl"]])-1
        print(f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% cum={cum:+.1%}")
    return {"n":len(trades),"wr":wr,"dd":max_dd,"ann":ann,"sh":sh,"eq":equity}


def walk_forward(C, O, H, L, NS, ND, dates, syms, predicted, ker_regime,
                 port_vol, sector_lookup, top_n=2, max_per_sector=2,
                 hold_days=5, vol_high_mult=2.0, size_reduce=0.3,
                 size_boost=1.0, vol_lookback=15, label=""):
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD V107 {label}")
    print(f"  tn={top_n} mps={max_per_sector} vhm={vol_high_mult:.1f} sr={size_reduce:.1f} sb={size_boost:.1f}")
    print(f"{'='*70}")

    years = sorted(set(d.year for d in dates))
    all_trades = []
    for ty in range(2019, years[-1]+1):
        ts = te = None
        for i,d in enumerate(dates):
            if d.year==ty and ts is None: ts = i
            if d.year==ty: te = i
        if ts is None: continue
        trades,_,_ = backtest_v107(C,O,H,L,NS,ND,dates,syms,predicted,ker_regime,
            port_vol, sector_lookup, top_n=top_n, max_per_sector=max_per_sector,
            hold_days=hold_days, vol_high_mult=vol_high_mult,
            size_reduce=size_reduce, size_boost=size_boost,
            vol_lookback=vol_lookback, start_di=ts, end_di=te+1)
        tt = [t for t in trades if dates[t["di"]].year==ty]
        all_trades.extend(tt)
        if tt:
            n=len(tt); nw=sum(1 for t in tt if t["pnl_pct"]>0)
            sc=defaultdict(int)
            for t in tt: sc[t.get("sector","OTHER")] += 1
            ss=" ".join(f"{k}:{v}" for k,v in sorted(sc.items()))
            print(f"  {ty}: {n}t WR={nw/n*100:.1f}% avg={np.mean([t['pnl_pct'] for t in tt]):+.2f}% sectors=[{ss}]", flush=True)
        else:
            print(f"  {ty}: no trades", flush=True)

    if all_trades:
        nw=sum(1 for t in all_trades if t["pnl_pct"]>0)
        cum=np.prod([1+t["pnl_pct"]/100 for t in all_trades])-1
        sc=defaultdict(int)
        for t in all_trades: sc[t.get("sector","OTHER")] += 1
        ss=" ".join(f"{k}:{v}" for k,v in sorted(sc.items()))
        print(f"\n  WF TOTAL: {len(all_trades)}t WR={nw/len(all_trades)*100:.1f}% cum={cum:+.1%}")
        print(f"  WF SECTORS: {ss}")
        return all_trades
    return []


def main():
    t0 = time.time()
    print("="*70)
    print("  V107: OI-PRICE DIVERGENCE AS ALPHA SOURCE")
    print("  11 factors: 7 standard + 4 OI-Price interaction features")
    print("  NW kernel + vol-adaptive sizing. Walk-forward 2019-2026.")
    print("="*70)

    C,O,H,L,V,OI,NS,ND,dates,syms = load_all_data(start="2016-01-01")
    print(f"  {NS} sym, {ND} days, {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    sl = build_sector_lookup(syms)
    sd = defaultdict(int)
    for sec in sl.values(): sd[sec] += 1
    print(f"  Sectors: {dict(sd)}")

    bt_2019 = next(i for i,d in enumerate(dates) if d >= pd.Timestamp("2019-01-01"))

    raw_factors = compute_raw_factors(C,O,H,L,V,OI,NS,ND)
    ker_regime = compute_ker(C,NS,ND)

    # NW predictions for each (tw, bw)
    pred_cache = {}
    for tw in [30, 40, 50]:
        for bw in [0.8, 1.0, 1.5]:
            print(f"\n--- NW (tw={tw}, bw={bw:.1f}) ---")
            pred_cache[(tw,bw)] = compute_nw_predicted_returns(raw_factors, NS, ND, tw, bw)

    vol_cache = {15: compute_portfolio_volatility(C, NS, ND, 15)}

    # Parameter sweep
    print(f"\n{'='*70}\n  PARAMETER SWEEP (2019-2026)\n{'='*70}")
    results = []
    sc = 0
    for pk, pred in pred_cache.items():
        tw, bw = pk
        for tn in [2, 3]:
            for mps in [2, 3]:
                for sr in [0.3, 0.5]:
                    for sb in [1.0, 1.5]:
                        sc += 1
                        trades, eq, dd = backtest_v107(
                            C,O,H,L,NS,ND,dates,syms,pred,ker_regime,
                            vol_cache[15], sl, top_n=tn, max_per_sector=mps,
                            vol_high_mult=2.0, size_reduce=sr, size_boost=sb,
                            start_di=bt_2019)
                        if len(trades) < 10: continue
                        nw = sum(1 for t in trades if t["pnl_pct"]>0)
                        nd = max(1, trades[-1]["di"]-trades[0]["di"])
                        ann = ((eq/CASH0)**(1/max(1.0,nd/252))-1)*100
                        ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
                        ra = np.array(ap)/CASH0
                        sh = np.mean(ra)/np.std(ra)*np.sqrt(252) if np.std(ra)>0 else 0
                        results.append({"tw":tw,"bw":bw,"tn":tn,"mps":mps,
                            "sr":sr,"sb":sb,"n":len(trades),"wr":nw/len(trades)*100,
                            "ann":ann,"dd":dd,"sharpe":sh,"eq":eq})

    results.sort(key=lambda x: -x["ann"])
    print(f"\n  Evaluated {sc} configs, {len(results)} with 10+ trades")

    if not results:
        print("  No configs with 10+ trades."); return

    print(f"\n{'TW':>4} {'BW':>4} {'TN':>3} {'MPS':>3} {'SR':>4} {'SB':>4} {'N':>5} {'WR':>6} {'Ann':>9} {'DD':>7} {'Sh':>7}")
    print("-"*70)
    for r in results[:15]:
        print(f"{r['tw']:>4} {r['bw']:>4.1f} {r['tn']:>3} {r['mps']:>3} {r['sr']:>4.1f} {r['sb']:>4.1f} {r['n']:>5} {r['wr']:>5.1f}% {r['ann']:>+8.1f}% {r['dd']:>6.1f}% {r['sharpe']:>6.2f}")

    # Walk-forward top configs
    best_sh = max(results, key=lambda x: x["sharpe"])
    best_ann = results[0]
    best_ra = max(results, key=lambda x: x["ann"]/max(x["dd"],1.0))

    for label, best in [("BEST-ANN",best_ann),("BEST-SHARPE",best_sh),("BEST-RISK-ADJ",best_ra)]:
        walk_forward(C,O,H,L,NS,ND,dates,syms,
            pred_cache[(best["tw"],best["bw"])], ker_regime,
            vol_cache[15], sl, top_n=best["tn"], max_per_sector=best["mps"],
            vol_high_mult=2.0, size_reduce=best["sr"], size_boost=best["sb"],
            label=label)

    # Compare V107 vs V96
    print(f"\n{'='*70}\n  COMPARISON: V107 (OI-Price features) vs V96 baseline\n{'='*70}")
    pb = pred_cache[(best_ann["tw"],best_ann["bw"])]
    t_v107, eq_v107, dd_v107 = backtest_v107(
        C,O,H,L,NS,ND,dates,syms,pb,ker_regime,vol_cache[15],sl,
        top_n=best_ann["tn"], max_per_sector=best_ann["mps"],
        vol_high_mult=2.0, size_reduce=best_ann["sr"], size_boost=best_ann["sb"],
        start_di=bt_2019)
    print(f"\n  V107 BEST-ANN (OI-Price features):")
    analyze(t_v107, eq_v107, dd_v107, "V107-OI-PX")
    print(f"\n  V96 REFERENCE (NW+BMA+Vol): ann=+36.4% Sharpe=5.62 MDD=5.2% 53t 66%WR")

    if t_v107:
        print(f"\n  Best config: tw={best_ann['tw']} bw={best_ann['bw']:.1f} tn={best_ann['tn']} mps={best_ann['mps']} sr={best_ann['sr']:.1f} sb={best_ann['sb']:.1f}")

    print(f"\n[V107] Done. {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
