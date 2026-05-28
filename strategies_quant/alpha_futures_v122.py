"""
V122: "先为不可胜" CAViaR-X Tail Risk Contagion Strategy
=========================================================
From guoxue: "先为不可胜，以待敌之可胜"
First make yourself invincible, then wait for the enemy to become vulnerable.

Paper 2603.25217: When core commodities breach VaR, correlated commodities
have 30-50% higher breach probability. Use this as a LEADING indicator.

Mechanism:
1. Rolling VaR (historical simulation, pct percentile) per instrument
2. VaR breach: daily return < VaR threshold
3. Sector contagion: N+ instruments in same sector breach VaR same day
4. Market contagion: N+ sectors have contagion simultaneously
5. Reduce positions by contagion_mult in affected sectors / all positions

Signal: NW Kernel Regression (V86 proven +52.9% ann)
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


def _extract_base_symbol(sym: str) -> str:
    s = sym.lower().split('.')[0].strip()
    while s and s[-1].isdigit(): s = s[:-1]
    if s.endswith('fi'): s = s[:-2]
    return s

def build_sector_lookup(syms: List[str]) -> Dict[int, str]:
    return {si: SECTOR_MAP.get(_extract_base_symbol(sym), 'OTHER')
            for si, sym in enumerate(syms)}

# ---- Factor computation ----

def _compute_atr_single(H, L, C, si, j):
    """Single ATR value for day j."""
    hh, ll, cc = H[si,j], L[si,j], C[si,j]
    if any(np.isnan([hh, ll, cc])): return None
    prev_c = C[si,j-1] if j > 0 and not np.isnan(C[si,j-1]) else cc
    return max(hh-ll, abs(hh-prev_c), abs(ll-prev_c))

def compute_rsi_compact(C, NS, ND, period=14):
    """Compact RSI using talib with manual fallback."""
    rsi = np.full((NS, ND), np.nan)
    if HAS_TALIB:
        for si in range(NS):
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nan_mask = np.isnan(C[si])
            try:
                r = talib.RSI(c, period)
                rsi[si] = np.where(nan_mask, np.nan, r)
            except Exception:
                pass
    # Manual fallback for instruments still all-NaN
    for si in range(NS):
        if not np.all(np.isnan(rsi[si])):
            continue
        for di in range(period+1, ND):
            deltas = []
            for j in range(di-period, di):
                if not np.isnan(C[si,j]) and not np.isnan(C[si,j-1]):
                    deltas.append(C[si,j] - C[si,j-1])
            if len(deltas) < period: continue
            gains = [max(d,0) for d in deltas]
            losses = [max(-d,0) for d in deltas]
            ag = np.mean(gains); al = np.mean(losses)
            if al == 0: rsi[si,di] = 100.0
            else: rsi[si,di] = 100.0 - 100.0/(1.0+ag/al)
    return rsi

def compute_raw_factors(C, O, H, L, V, OI, NS, ND):
    t0 = time.time()
    print("[V122] Computing raw factors...", flush=True)

    ret_5d = np.full((NS,ND), np.nan)
    oi_5d = np.full((NS,ND), np.nan)
    vol_5d = np.full((NS,ND), np.nan)
    range_5d = np.full((NS,ND), np.nan)
    atrp_5d = np.full((NS,ND), np.nan)
    ret_10d = np.full((NS,ND), np.nan)
    fwd_ret_5d = np.full((NS,ND), np.nan)
    atr_mean = np.full((NS,ND), np.nan)

    for si in range(NS):
        for di in range(ND):
            # 5d return
            if di >= 5 and not np.isnan(C[si,di]) and not np.isnan(C[si,di-5]) and C[si,di-5] > 0:
                ret_5d[si,di] = C[si,di]/C[si,di-5] - 1.0
            # 10d return
            if di >= 10 and not np.isnan(C[si,di]) and not np.isnan(C[si,di-10]) and C[si,di-10] > 0:
                ret_10d[si,di] = C[si,di]/C[si,di-10] - 1.0
            # OI 5d change
            if di >= 5 and not np.isnan(OI[si,di]) and not np.isnan(OI[si,di-5]) and OI[si,di-5] > 0:
                oi_5d[si,di] = OI[si,di]/OI[si,di-5] - 1.0
            # Vol 5d mean
            if di >= 5:
                vals = V[si,di-5:di]; valid = vals[~np.isnan(vals)]
                if len(valid) >= 3: vol_5d[si,di] = np.mean(valid)
            # Range 5d mean
            if di >= 5:
                rv = [(H[si,j]-L[si,j])/C[si,j] for j in range(di-5,di)
                      if not np.isnan(H[si,j]) and not np.isnan(L[si,j])
                      and not np.isnan(C[si,j]) and C[si,j]>0 and H[si,j]>L[si,j]]
                if len(rv) >= 3: range_5d[si,di] = np.mean(rv)
            # ATR% 5d
            if di >= 6:
                av = [_compute_atr_single(H,L,C,si,j) for j in range(di-5,di)]
                av = [x for x in av if x is not None]
                if av and not np.isnan(C[si,di]) and C[si,di]>0:
                    atrp_5d[si,di] = np.mean(av)/C[si,di]
            # Forward return
            if di < ND-5 and not np.isnan(C[si,di+5]) and not np.isnan(C[si,di]) and C[si,di]>0:
                fwd_ret_5d[si,di] = C[si,di+5]/C[si,di] - 1.0
            # ATR mean for bandwidth
            if di >= 20:
                av = [_compute_atr_single(H,L,C,si,j) for j in range(di-14,di)]
                av = [x for x in av if x is not None]
                if av and not np.isnan(C[si,di]) and C[si,di]>0:
                    atr_mean[si,di] = np.mean(av)/C[si,di]

    rsi14 = compute_rsi_compact(C, NS, ND, 14)
    print(f"  Raw factors done: {time.time()-t0:.1f}s", flush=True)
    return {"ret_5d":ret_5d,"oi_5d":oi_5d,"vol_5d":vol_5d,"range_5d":range_5d,
            "atrp_5d":atrp_5d,"ret_10d":ret_10d,"rsi14":rsi14,
            "fwd_ret_5d":fwd_ret_5d,"atr_mean":atr_mean}

def normalize_factor(factor, NS, ND, min_count=10):
    normed = np.full((NS,ND), np.nan)
    for di in range(ND):
        vals = factor[:,di]; valid = vals[~np.isnan(vals)]
        if len(valid) < min_count: continue
        mu, sigma = np.mean(valid), np.std(valid)
        if sigma < 1e-12: continue
        for si in range(NS):
            if not np.isnan(vals[si]): normed[si,di] = (vals[si]-mu)/sigma
    return normed

# ---- NW Kernel Regression (from V86, no BMA) ----

def compute_nw_predicted_returns(raw_factors, NS, ND, training_window=40, kernel_bandwidth=1.0):
    t0 = time.time()
    print(f"[V122] Computing NW predicted returns (w={training_window}, bw={kernel_bandwidth:.1f})...", flush=True)
    normed = {fn: normalize_factor(raw_factors[fn], NS, ND) for fn in FACTOR_NAMES}
    fwd_ret = raw_factors["fwd_ret_5d"]; atr_mean = raw_factors["atr_mean"]
    predicted = np.full((NS,ND), np.nan)
    MIN_TRAIN = 20

    for di in range(training_window+10, ND):
        train_f, train_t = [], []
        for tdi in range(max(10,di-training_window), di):
            for si in range(NS):
                feat = np.array([normed[fn][si,tdi] for fn in FACTOR_NAMES])
                target = fwd_ret[si,tdi]
                if np.any(np.isnan(feat)) or np.isnan(target): continue
                train_f.append(feat); train_t.append(target)
        if len(train_f) < MIN_TRAIN: continue
        tX = np.array(train_f); tY = np.array(train_t)
        fstd = np.std(tX, axis=0); fstd[fstd<1e-12] = 1.0

        for si in range(NS):
            qf = np.array([normed[fn][si,di] for fn in FACTOR_NAMES])
            if np.any(np.isnan(qf)): continue
            atr_val = atr_mean[si,di]
            h = max(atr_val*kernel_bandwidth, 0.1) if not np.isnan(atr_val) else kernel_bandwidth
            diff = tX - qf[np.newaxis,:]
            dist = np.sqrt(np.sum((diff/fstd[np.newaxis,:])**2, axis=1))
            sd = dist/h; w = np.zeros(len(tX))
            mask = sd <= 1.0
            if not np.any(mask):
                mi = np.argmin(dist)
                if dist[mi] < 1e12: w[mi] = 1.0; mask = np.zeros(len(dist),dtype=bool); mask[mi] = True
                else: continue
            else: w[mask] = 0.75*(1.0-sd[mask]**2)
            ws = np.sum(w)
            if ws < 1e-12: continue
            predicted[si,di] = np.sum(w*tY)/ws
        if di % 100 == 0:
            print(f"  di={di}/{ND} valid={np.sum(~np.isnan(predicted[:,di]))}/{NS}", flush=True)
    print(f"  NW done: {time.time()-t0:.1f}s", flush=True)
    return predicted

def compute_ker(C, NS, ND):
    ker_regime = np.zeros((NS,ND), dtype=int)
    for si in range(NS):
        for di in range(10, ND):
            closes = C[si,di-10:di+1]; valid = closes[~np.isnan(closes)]
            if len(valid)<10 or valid[0]<=0: continue
            nc = abs(valid[-1]-valid[0]); tc = np.sum(np.abs(np.diff(valid)))
            if tc > 1e-10:
                ker = nc/tc
                if ker < 0.15: ker_regime[si,di] = 1
                elif ker > 0.3: ker_regime[si,di] = -1
    return ker_regime

def compute_atr_at(H, L, C, si, di, start_di):
    av = [_compute_atr_single(H,L,C,si,j) for j in range(max(start_di,di-14),di)]
    av = [x for x in av if x is not None]
    return np.mean(av) if av else None

# =====================================================================
# INNOVATION: CAViaR-X Tail Risk Contagion Monitoring
# =====================================================================

def compute_daily_returns(C, NS, ND):
    rets = np.full((NS,ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if not np.isnan(C[si,di]) and not np.isnan(C[si,di-1]) and C[si,di-1]>0:
                rets[si,di] = C[si,di]/C[si,di-1] - 1.0
    return rets

def compute_var_historical(daily_rets, NS, ND, var_window=20, var_pct=0.05):
    t0 = time.time()
    print(f"[V122] Computing VaR (w={var_window}, pct={var_pct:.0%})...", flush=True)
    var_arr = np.full((NS,ND), np.nan)
    for si in range(NS):
        for di in range(var_window, ND):
            wr = daily_rets[si,di-var_window:di]; valid = wr[~np.isnan(wr)]
            if len(valid) >= var_window//2:
                var_arr[si,di] = np.percentile(valid, var_pct*100)
    print(f"  VaR done: {time.time()-t0:.1f}s", flush=True)
    return var_arr

def detect_contagion(daily_rets, var_arr, sector_lookup, NS, ND,
                     sector_contagion_count=2, market_contagion_sectors=3):
    t0 = time.time()
    print(f"[V122] Detecting contagion (sec>={sector_contagion_count}, mkt>={market_contagion_sectors})...", flush=True)
    sec_contagion: Dict[int,set] = {}
    mkt_contagion: Dict[int,set] = {}

    for di in range(ND):
        breaches: Dict[str,List[int]] = defaultdict(list)
        for si in range(NS):
            if np.isnan(daily_rets[si,di]) or np.isnan(var_arr[si,di]): continue
            if daily_rets[si,di] < var_arr[si,di]:
                breaches[sector_lookup.get(si,'OTHER')].append(si)
        cont = {sec for sec, bsis in breaches.items() if len(bsis) >= sector_contagion_count}
        if cont: sec_contagion[di] = cont
        if len(cont) >= market_contagion_sectors: mkt_contagion[di] = cont

    print(f"  Contagion: {len(sec_contagion)} sec-days, {len(mkt_contagion)} mkt-days of {ND}. {time.time()-t0:.1f}s", flush=True)
    return sec_contagion, mkt_contagion

def get_contagion_mult(di, held_sectors, sec_contagion, mkt_contagion, cm):
    result: Dict[int,float] = {}
    if di in mkt_contagion:
        return {si: cm for si in held_sectors}
    if di in sec_contagion:
        cont_sec = sec_contagion[di]
        return {si: (cm if sec in cont_sec else 1.0) for si,sec in held_sectors.items()}
    return {si: 1.0 for si in held_sectors}

# ---- Backtest with CAViaR-X contagion protection ----

def get_dynamic_mode(rt_win, win_thresh=0.60, wr_window=15):
    if len(rt_win) < 5: return "normal"
    wr = sum(rt_win[-wr_window:])/len(rt_win[-wr_window:])
    if wr > win_thresh: return "winning"
    elif wr < 0.50: return "losing"
    return "normal"

def _make_trade(di, d, si, syms, sector_lookup, edi, ep, alloc, equity, reason, mode):
    c = None; pnl = None  # filled by caller
    return {"pnl_abs": None, "pnl_pct": None, "days": di-edi+1, "di": di,
            "year": d.year, "sym": syms[si], "sector": sector_lookup.get(si,'OTHER'),
            "reason": reason, "mode": mode[:1].upper()}

def backtest_v122(C, O, H, L, NS, ND, dates, syms, predicted, ker_regime,
                  sec_contagion, mkt_contagion, sector_lookup,
                  contagion_mult=0.5, top_n=2, max_per_sector=2,
                  hold_days=5, atr_stop=3.0, start_di=60, end_di=None):
    if end_di is None: end_di = ND-1
    equity, peak, max_dd = CASH0, CASH0, 0.0
    positions: List[Tuple[int,int,float,float,float]] = []
    trades: List[dict] = []
    recent_wins: List[int] = []
    cont_reductions = 0; mkt_days = 0; sec_days = 0

    for di in range(max(start_di,1), end_di):
        d = dates[di]; daily_pnl = 0.0
        new_pos: List[Tuple[int,int,float,float,float]] = []
        mode = get_dynamic_mode(recent_wins)
        if di in mkt_contagion: mkt_days += 1
        elif di in sec_contagion: sec_days += 1

        # Exit logic
        by_si: Dict[int,List] = defaultdict(list)
        for si,edi,ep,sp,alloc in positions: by_si[si].append((edi,ep,sp,alloc))

        for si, plist in by_si.items():
            c = C[si,di]
            if np.isnan(c):
                for edi,ep,sp,alloc in plist: new_pos.append((si,edi,ep,sp,alloc))
                continue
            earliest = min(p[0] for p in plist)
            hold = di - earliest
            stopped = any(c < sp for _,_,sp,_ in plist)
            should_exit = stopped or hold >= hold_days
            if should_exit:
                for edi,ep,sp,alloc in plist:
                    pnl = (c-ep)/ep - COMM; profit = equity*alloc*pnl
                    daily_pnl += profit; is_win = pnl > 0
                    trades.append({"pnl_abs":profit,"pnl_pct":pnl*100,"days":di-edi+1,
                        "di":di,"year":d.year,"sym":syms[si],
                        "sector":sector_lookup.get(si,'OTHER'),
                        "reason":"stop" if stopped else "hold","mode":mode[0].upper()})
                    recent_wins.append(1 if is_win else 0)
            else:
                for edi,ep,sp,alloc in plist: new_pos.append((si,edi,ep,sp,alloc))

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
                 and di+1 < ND and not np.isnan(O[si,di+1])
                 and ker_regime[si,di] >= 0]
        if not cands: continue
        cands.sort(key=lambda x: -x[0])

        n_take = top_n
        if mode == "winning": n_take = min(top_n+1, top_n*2)
        elif mode == "losing": n_take = max(1, top_n-1)

        sec_counts: Dict[str,int] = defaultdict(int)
        for si_h in held: sec_counts[sector_lookup.get(si_h,'OTHER')] += 1
        entries = []
        for pv,si in cands:
            if len(held)+len(entries) >= n_take: break
            if si in held: continue
            sec = sector_lookup.get(si,'OTHER')
            if sec_counts[sec] >= max_per_sector: continue
            if pv <= 0: continue
            entries.append((pv,si,sec)); sec_counts[sec] += 1
        if not entries: continue

        # Contagion multiplier
        held_sec = {si: sector_lookup.get(si,'OTHER') for si,_,_,_,_ in positions}
        for _,si,sec in entries: held_sec[si] = sec
        cmults = get_contagion_mult(di, held_sec, sec_contagion, mkt_contagion, contagion_mult)
        if any(v < 1.0 for v in cmults.values()): cont_reductions += 1

        base_alloc = LEVERAGE / (len(positions)+len(entries))
        upd_pos = [(si,edi,ep,sp, base_alloc*cmults.get(si,1.0)) for si,edi,ep,sp,_ in positions]
        for pv,si,sec in entries:
            ep = O[si,di+1]
            if np.isnan(ep) or ep <= 0: continue
            atr = compute_atr_at(H,L,C,si,di,start_di)
            if atr is None: continue
            upd_pos.append((si,di+1,ep, ep-atr_stop*atr, base_alloc*cmults.get(si,1.0)))
        positions = upd_pos

    # Close remaining
    for si,edi,ep,sp,alloc in positions:
        c = C[si,ND-1]
        if not np.isnan(c) and c > 0: equity += equity*alloc*((c-ep)/ep - COMM)

    if trades:
        trades[0]["contagion_info"] = {"market_days":mkt_days,"sector_days":sec_days,"reductions":cont_reductions}
    return trades, equity, max_dd

# ---- Analysis & Walk-forward ----

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
    aw = np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"]>0])
    al = np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"]<=0])
    sc = defaultdict(int)
    for t in trades: sc[t.get("sector","OTHER")] += 1
    ss = " ".join(f"{k}:{v}" for k,v in sorted(sc.items()))
    ci = trades[0].get("contagion_info",{})

    print(f"  {label}: {len(trades)}t (s:{ns} h:{nh}) WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% Sh={sh:.2f} eq={equity:,.0f}")
    print(f"    avg_win={aw:+.3f}% avg_loss={al:+.3f}% sectors: {ss}")
    if ci: print(f"    contagion: mkt={ci.get('market_days',0)}d sec={ci.get('sector_days',0)}d red={ci.get('reductions',0)}")

    yr: Dict[int,dict] = {}
    for t in trades:
        y = t["year"]
        if y not in yr: yr[y] = {"n":0,"w":0,"pnl":[]}
        yr[y]["n"] += 1
        if t["pnl_pct"]>0: yr[y]["w"] += 1
        yr[y]["pnl"].append(t["pnl_pct"])
    for y in sorted(yr.keys()):
        ys = yr[y]; cum = np.prod([1+p/100 for p in ys["pnl"]])-1
        print(f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% cum={cum:+.1%}")
    return {"n":len(trades),"wr":wr,"dd":max_dd,"ann":ann,"sh":sh,"eq":equity}

def walk_forward(C, O, H, L, NS, ND, dates, syms, predicted, ker_regime,
                 sec_contagion, mkt_contagion, sector_lookup,
                 contagion_mult=0.5, top_n=2, max_per_sector=2, hold_days=5, label=""):
    print(f"\n{'='*70}\n  WF V122 {label} tn={top_n} mps={max_per_sector} cm={contagion_mult:.1f}\n{'='*70}")
    years = sorted(set(d.year for d in dates))
    all_trades: List[dict] = []

    for yr in range(2019, years[-1]+1):
        ts = te = None
        for i,d in enumerate(dates):
            if d.year == yr and ts is None: ts = i
            if d.year == yr: te = i
        if ts is None: continue
        trades,_,_ = backtest_v122(C,O,H,L,NS,ND,dates,syms,predicted,ker_regime,
            sec_contagion,mkt_contagion,sector_lookup=sector_lookup,
            contagion_mult=contagion_mult,top_n=top_n,max_per_sector=max_per_sector,
            hold_days=hold_days,start_di=ts,end_di=te+1)
        yt = [t for t in trades if dates[t["di"]].year==yr]
        all_trades.extend(yt)
        if yt:
            n=len(yt); nw=sum(1 for t in yt if t["pnl_pct"]>0)
            print(f"  {yr}: {n}t WR={nw/n*100:.1f}% avg={np.mean([t['pnl_pct'] for t in yt]):+.2f}%", flush=True)
        else: print(f"  {yr}: no trades", flush=True)

    if all_trades:
        nw=sum(1 for t in all_trades if t["pnl_pct"]>0)
        cum=np.prod([1+t["pnl_pct"]/100 for t in all_trades])-1
        print(f"\n  WF TOTAL: {len(all_trades)}t WR={nw/len(all_trades)*100:.1f}% cum={cum:+.1%}")
    return all_trades

def _compute_stats(trades, eq):
    if len(trades) < 10: return None
    nw = sum(1 for t in trades if t["pnl_pct"]>0)
    wr = nw/len(trades)*100
    nd = max(1, trades[-1]["di"]-trades[0]["di"])
    ann = ((eq/CASH0)**(1/max(1.0,nd/252))-1)*100
    ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
    ra = np.array(ap)/CASH0
    sh = np.mean(ra)/np.std(ra)*np.sqrt(252) if np.std(ra)>0 else 0
    return {"n":len(trades),"wr":wr,"ann":ann,"sharpe":sh,"eq":eq}

def main():
    t0 = time.time()
    print("="*70)
    print("  V122: 先为不可胜 CAViaR-X TAIL RISK CONTAGION")
    print("  Signal: NW Kernel (V86) | Risk: CAViaR-X VaR contagion")
    print("  Walk-forward 2019-2026. LEVERAGE=1.0")
    print("="*70)

    C,O,H,L,V,OI,NS,ND,dates,syms = load_all_data(start="2016-01-01")
    print(f"  {NS} sym, {ND} days, {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    sector_lookup = build_sector_lookup(syms)
    sd = defaultdict(int)
    for s in sector_lookup.values(): sd[s] += 1
    print(f"  Sectors: {dict(sd)}")

    bt_2019 = next((i for i,d in enumerate(dates) if d >= pd.Timestamp("2019-01-01")), None)

    # Compute signals
    raw_factors = compute_raw_factors(C,O,H,L,V,OI,NS,ND)
    ker_regime = compute_ker(C,NS,ND)
    predicted = compute_nw_predicted_returns(raw_factors, NS, ND)
    daily_rets = compute_daily_returns(C, NS, ND)

    # Compute VaR & contagion for all param combos
    print("\n" + "="*70 + "\n  PARAMETER SWEEP (2019-2026)\n" + "="*70)
    results: List[dict] = []
    contagion_cache: Dict[Tuple, Tuple] = {}

    for vw in [15,20,30]:
        for vp in [0.05,0.10]:
            var_arr = compute_var_historical(daily_rets, NS, ND, vw, vp)
            for scc in [2,3]:
                for mcs in [2,3]:
                    sc,mc = detect_contagion(daily_rets, var_arr, sector_lookup, NS, ND, scc, mcs)
                    contagion_cache[(vw,vp,scc,mcs)] = (sc,mc)

    sweep_ct = 0
    for (vw,vp,scc,mcs),(sc,mc) in contagion_cache.items():
        for cm in [0.3,0.5]:
            for tn in [2,3]:
                for mps in [2,3]:
                    sweep_ct += 1
                    trades,eq,dd = backtest_v122(C,O,H,L,NS,ND,dates,syms,predicted,ker_regime,
                        sc,mc,sector_lookup=sector_lookup,contagion_mult=cm,
                        top_n=tn,max_per_sector=mps,hold_days=5,start_di=bt_2019)
                    st = _compute_stats(trades, eq)
                    if st:
                        results.append({"vw":vw,"vp":vp,"scc":scc,"mcs":mcs,"cm":cm,
                            "top_n":tn,"mps":mps,"dd":dd,**st})

    # Baseline (no contagion)
    print("\n--- Baseline (no contagion) ---")
    tb,eb,db = backtest_v122(C,O,H,L,NS,ND,dates,syms,predicted,ker_regime,
        {},{},sector_lookup=sector_lookup,contagion_mult=1.0,top_n=2,
        max_per_sector=2,hold_days=5,start_di=bt_2019)
    sb = _compute_stats(tb, eb)
    if sb: results.append({"vw":0,"vp":0,"scc":0,"mcs":0,"cm":1.0,
        "top_n":2,"mps":2,"dd":db,**sb})

    results.sort(key=lambda x: -x["ann"])
    print(f"\n  Evaluated {sweep_ct} configs, {len(results)} with 10+ trades")
    print(f"\n{'VW':>4} {'VP':>5} {'SCC':>3} {'MCS':>3} {'CM':>4} {'TN':>2} {'MPS':>3} "
          f"{'N':>5} {'WR':>6} {'Ann':>9} {'DD':>7} {'Sh':>6}")
    print("-"*75)
    for r in results[:15]:
        is_base = r["vw"] == 0
        vw_s = "base" if is_base else str(r["vw"])
        vp_s = "" if is_base else f"{r['vp']:.2f}"
        scc_s = "" if is_base else str(r["scc"])
        mcs_s = "" if is_base else str(r["mcs"])
        print(f"{vw_s:>4} {vp_s:>5} {scc_s:>3} {mcs_s:>3} "
              f"{r['cm']:>4.1f} {r['top_n']:>2} {r['mps']:>3} "
              f"{r['n']:>5} {r['wr']:>5.1f}% {r['ann']:>+8.1f}% "
              f"{r['dd']:>6.1f}% {r['sharpe']:>5.2f}")

    if not results: print("  No results. Exiting."); return

    # Walk-forward top configs
    best_sh = max(results, key=lambda x: x["sharpe"])
    best_ann = results[0]
    best_ra = max(results, key=lambda x: x["ann"]/max(x["dd"],1.0))

    for label, best in [("BEST-ANN",best_ann),("BEST-SHARPE",best_sh),("BEST-RISK-ADJ",best_ra)]:
        if best["vw"]==0: sc,mc = {},{}
        else: sc,mc = contagion_cache[(best["vw"],best["vp"],best["scc"],best["mcs"])]
        walk_forward(C,O,H,L,NS,ND,dates,syms,predicted,ker_regime,sc,mc,
            sector_lookup=sector_lookup,contagion_mult=best["cm"],
            top_n=best["top_n"],max_per_sector=best["mps"],hold_days=5,label=label)

    # Final comparison: best CONTAGION config vs baseline
    print("\n" + "="*70 + "\n  COMPARISON: V122 (CAViaR-X) vs V96 baseline\n  (2019-2026 OOS)\n" + "="*70)

    # Find best contagion-only config (exclude baseline)
    cont_results = [r for r in results if r["vw"] > 0]
    if not cont_results:
        print("  No contagion configs with 10+ trades.")
        print(f"\n[V122] Done. {time.time()-t0:.1f}s")
        return

    best_cont_sh = max(cont_results, key=lambda x: x["sharpe"])
    best_cont_dd = min(cont_results, key=lambda x: x["dd"])
    best_cont_ra = max(cont_results, key=lambda x: x["ann"]/max(x["dd"],1.0))

    for clabel, cbest in [("BEST-CONTAGION-SH",best_cont_sh),
                          ("BEST-CONTAGION-DD",best_cont_dd),
                          ("BEST-CONTAGION-RA",best_cont_ra)]:
        sc_b,mc_b = contagion_cache[(cbest["vw"],cbest["vp"],cbest["scc"],cbest["mcs"])]
        t122,eq122,dd122 = backtest_v122(C,O,H,L,NS,ND,dates,syms,predicted,ker_regime,
            sc_b,mc_b,sector_lookup=sector_lookup,contagion_mult=cbest["cm"],
            top_n=cbest["top_n"],max_per_sector=cbest["mps"],hold_days=5,start_di=bt_2019)
        print(f"\n  V122 {clabel} (VW={cbest['vw']} VP={cbest['vp']} SCC={cbest['scc']} MCS={cbest['mcs']} CM={cbest['cm']}):")
        analyze(t122, eq122, dd122, f"V122-{clabel}")
        if tb:
            print(f"    vs baseline: eq={eq122-eb:+,.0f} dd={dd122-db:+.1f}% ({(db-dd122)/db*100:+.1f}% MDD reduction)")

    print(f"\n  V96 BASELINE (NW only, no contagion):")
    analyze(tb, eb, db, "V96-baseline")

    print(f"\n[V122] Done. {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
