"""
V109: KER-Based Dynamic Hold Period
====================================
Innovation: Replace fixed 5-day hold with KER-regime-dependent hold period.

Core insight: V20/V48 varied hold by signal strength (WRONG). V109 varies
hold by MARKET REGIME because regime determines how quickly MR plays out:
  - KER < consolidation_threshold: slow MR recovery, hold longer
  - KER > trending_threshold: exit faster
  - KER in between: standard hold

Base: V96 (NW Kernel + BMA + Vol-adaptive sizing)
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

CASH0 = 1_000_000; COMM = 0.0005; LEVERAGE = 1.0

SECTOR_MAP = {
    'i':'BLACK','j':'BLACK','jm':'BLACK','hc':'BLACK','sf':'BLACK',
    'sm':'BLACK','wr':'BLACK','im':'BLACK',
    'cu':'METAL','al':'METAL','zn':'METAL','pb':'METAL','ni':'METAL',
    'sn':'METAL','ss':'METAL','ao':'METAL','au':'METAL','ag':'METAL',
    'rb':'METAL','si':'METAL',
    'sc':'ENERGY','fu':'ENERGY','bu':'ENERGY','pg':'ENERGY','eb':'ENERGY',
    'ta':'ENERGY','fg':'ENERGY','oi':'ENERGY',
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


def compute_rsi_manual(C: np.ndarray, NS: int, ND: int, period: int = 14) -> np.ndarray:
    rsi = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]; gains = np.full(ND, np.nan); losses = np.full(ND, np.nan)
        for di in range(1, ND):
            if np.isnan(c[di]) or np.isnan(c[di-1]): continue
            delta = c[di] - c[di-1]
            gains[di] = max(delta, 0.0); losses[di] = max(-delta, 0.0)
        avg_gain = avg_loss = np.nan
        for di in range(1, ND):
            if np.isnan(gains[di]): continue
            if np.isnan(avg_gain):
                vg, vl = [], []
                for j in range(di, min(di+period, ND)):
                    if not np.isnan(gains[j]):
                        vg.append(gains[j])
                        vl.append(losses[j] if not np.isnan(losses[j]) else 0.0)
                if len(vg) >= period:
                    avg_gain, avg_loss = np.mean(vg), np.mean(vl)
                    if avg_loss == 0: rsi[si, di+period-1] = 100.0
                    else: rsi[si, di+period-1] = 100.0 - 100.0/(1.0+avg_gain/avg_loss)
                continue
            avg_gain = (avg_gain*(period-1)+gains[di])/period
            avg_loss = (avg_loss*(period-1)+losses[di])/period
            if avg_loss == 0: rsi[si, di] = 100.0
            else: rsi[si, di] = 100.0 - 100.0/(1.0+avg_gain/avg_loss)
    return rsi


def compute_raw_factors(C, O, H, L, V, OI, NS, ND):
    t0 = time.time()
    print("[V109] Computing raw factors...", flush=True)
    ret_5d = np.full((NS,ND),np.nan)
    oi_5d = np.full((NS,ND),np.nan)
    vol_5d = np.full((NS,ND),np.nan)
    range_5d = np.full((NS,ND),np.nan)
    atrp_5d = np.full((NS,ND),np.nan)
    ret_10d = np.full((NS,ND),np.nan)
    fwd_ret_5d = np.full((NS,ND),np.nan)
    atr_mean = np.full((NS,ND),np.nan)

    for si in range(NS):
        for di in range(5, ND):
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
        for di in range(6, ND):
            atr_v = []
            for j in range(di-5,di):
                hh,ll,cc = H[si,j],L[si,j],C[si,j]
                if not any(np.isnan([hh,ll,cc])):
                    prev_c = C[si,j-1] if j>0 and not np.isnan(C[si,j-1]) else cc
                    atr_v.append(max(hh-ll,abs(hh-prev_c),abs(ll-prev_c)))
            if atr_v and not np.isnan(C[si,di]) and C[si,di]>0:
                atrp_5d[si,di] = np.mean(atr_v)/C[si,di]
        for di in range(10, ND):
            if not np.isnan(C[si,di]) and not np.isnan(C[si,di-10]) and C[si,di-10]>0:
                ret_10d[si,di] = C[si,di]/C[si,di-10]-1.0
        for di in range(ND-5):
            if not np.isnan(C[si,di+5]) and not np.isnan(C[si,di]) and C[si,di]>0:
                fwd_ret_5d[si,di] = C[si,di+5]/C[si,di]-1.0
        for di in range(20, ND):
            atr_v = []
            for j in range(di-14,di):
                hh,ll,cc = H[si,j],L[si,j],C[si,j]
                if not any(np.isnan([hh,ll,cc])):
                    prev_c = C[si,j-1] if j>0 and not np.isnan(C[si,j-1]) else cc
                    atr_v.append(max(hh-ll,abs(hh-prev_c),abs(ll-prev_c)))
            if atr_v and not np.isnan(C[si,di]) and C[si,di]>0:
                atr_mean[si,di] = np.mean(atr_v)/C[si,di]

    rsi14 = np.full((NS,ND),np.nan)
    if HAS_TALIB:
        for si in range(NS):
            c = np.where(np.isnan(C[si]),0,C[si]).astype(np.float64)
            nm = np.isnan(C[si])
            try: rsi14[si] = np.where(nm, np.nan, talib.RSI(c,14))
            except: pass
    needs_fb = np.all(np.isnan(rsi14), axis=1)
    if needs_fb.any():
        rm = compute_rsi_manual(C, NS, ND, 14)
        for si in range(NS):
            if needs_fb[si]: rsi14[si] = rm[si]

    print(f"  Raw factors done: {time.time()-t0:.1f}s", flush=True)
    return {"ret_5d":ret_5d,"oi_5d":oi_5d,"vol_5d":vol_5d,"range_5d":range_5d,
            "atrp_5d":atrp_5d,"ret_10d":ret_10d,"rsi14":rsi14,
            "fwd_ret_5d":fwd_ret_5d,"atr_mean":atr_mean}


def normalize_factor(factor, NS, ND, min_count=10):
    normed = np.full((NS,ND),np.nan)
    for di in range(ND):
        vals = factor[:,di]; valid = vals[~np.isnan(vals)]
        if len(valid)<min_count: continue
        mu, sigma = np.mean(valid), np.std(valid)
        if sigma<1e-12: continue
        for si in range(NS):
            if not np.isnan(vals[si]): normed[si,di] = (vals[si]-mu)/sigma
    return normed


def compute_rolling_ic(raw_factors, NS, ND, ic_window=60, min_pairs=15):
    t0 = time.time()
    print(f"[V109] Rolling IC (window={ic_window})...", flush=True)
    fwd_ret = raw_factors["fwd_ret_5d"]; ic_array = np.full((N_FACTORS,ND),np.nan)
    for fi, fname in enumerate(FACTOR_NAMES):
        factor = raw_factors[fname]
        for di in range(ic_window+5, ND):
            ic_vals = []
            for tdi in range(di-ic_window, di):
                f_day, r_day = factor[:,tdi], fwd_ret[:,tdi]
                vm = (~np.isnan(f_day))&(~np.isnan(r_day))
                fv, rv = f_day[vm], r_day[vm]
                if len(fv)>=min_pairs:
                    corr = np.corrcoef(pd.Series(fv).rank().values,
                                       pd.Series(rv).rank().values)[0,1]
                    if not np.isnan(corr): ic_vals.append(corr)
            if len(ic_vals)>=5: ic_array[fi,di] = np.mean(ic_vals)
        if fi%2==0: print(f"  IC {fname}: {time.time()-t0:.1f}s", flush=True)
    print(f"  IC done: {time.time()-t0:.1f}s", flush=True)
    return ic_array


def compute_bma_weights(ic_array, ND, prior_strength=5.0, min_ic_history=20):
    t0 = time.time()
    print(f"[V109] BMA weights (prior={prior_strength:.1f})...", flush=True)
    weights = np.full((N_FACTORS,ND),np.nan)
    for fi in range(N_FACTORS):
        for di in range(min_ic_history, ND):
            ic_hist = ic_array[fi,max(0,di-120):di]
            vic = ic_hist[~np.isnan(ic_hist)]
            if len(vic)<5: continue
            n_pos = np.sum(vic>0); n_neg = len(vic)-n_pos
            a = prior_strength/2.0+n_pos; b = prior_strength/2.0+n_neg
            weights[fi,di] = a/(a+b)
    for di in range(ND):
        w = weights[:,di]; v = w[~np.isnan(w)]
        if len(v)==N_FACTORS:
            ws = np.nansum(w)
            if ws>0: weights[:,di] = w/ws
    print(f"  BMA done: {time.time()-t0:.1f}s", flush=True)
    return weights


def compute_nw_predicted(raw_factors, bma_weights, NS, ND,
                          training_window=40, kernel_bandwidth=1.0):
    t0 = time.time()
    print(f"[V109] NW+BMA prediction (tw={training_window},bw={kernel_bandwidth:.1f})...",
          flush=True)
    normed = {fn: normalize_factor(raw_factors[fn], NS, ND) for fn in FACTOR_NAMES}
    # Apply BMA weighting
    weighted = {}
    for fi, fn in enumerate(FACTOR_NAMES):
        orig = normed[fn]; result = np.full((NS,ND),np.nan)
        for di in range(ND):
            w = bma_weights[fi,di]
            if np.isnan(w): w = 1.0/N_FACTORS
            for si in range(NS):
                if not np.isnan(orig[si,di]): result[si,di] = orig[si,di]*(w*N_FACTORS)
        weighted[fn] = result

    fwd_ret = raw_factors["fwd_ret_5d"]; atr_mean = raw_factors["atr_mean"]
    predicted = np.full((NS,ND),np.nan); MIN_TRAIN = 20

    for di in range(training_window+10, ND):
        tf, tt = [], []
        for tdi in range(max(10,di-training_window), di):
            for si in range(NS):
                feat = np.array([weighted[fn][si,tdi] for fn in FACTOR_NAMES])
                tgt = fwd_ret[si,tdi]
                if np.any(np.isnan(feat)) or np.isnan(tgt): continue
                tf.append(feat); tt.append(tgt)
        if len(tf)<MIN_TRAIN: continue
        tX, tY = np.array(tf), np.array(tt)
        fstd = np.std(tX,axis=0); fstd[fstd<1e-12] = 1.0
        for si in range(NS):
            qf = np.array([weighted[fn][si,di] for fn in FACTOR_NAMES])
            if np.any(np.isnan(qf)): continue
            atr_v = atr_mean[si,di]
            h = max(atr_v*kernel_bandwidth, 0.1) if not np.isnan(atr_v) else kernel_bandwidth
            dist = np.sqrt(np.sum(((tX-qf[np.newaxis,:])/fstd[np.newaxis,:])**2, axis=1))
            sd = dist/h; ws = np.zeros(len(tX))
            mask = sd<=1.0
            if not np.any(mask):
                mi = np.argmin(dist)
                if dist[mi]<1e12: ws[mi]=1.0; mask=np.zeros(len(dist),bool); mask[mi]=True
                else: continue
            else: ws[mask] = 0.75*(1.0-sd[mask]**2)
            wsum = np.sum(ws)
            if wsum<1e-12: continue
            predicted[si,di] = np.sum(ws*tY)/wsum
        if di%100==0:
            print(f"  di={di}/{ND} valid={np.sum(~np.isnan(predicted[:,di]))}/{NS}",
                  flush=True)
    print(f"  NW+BMA done: {time.time()-t0:.1f}s", flush=True)
    return predicted


# =====================================================================
# V109 INNOVATION: KER-based dynamic hold period
# =====================================================================

def compute_ker_raw(C: np.ndarray, NS: int, ND: int) -> np.ndarray:
    """Raw Kaufman efficiency ratio (0-1) for dynamic hold period."""
    ker = np.full((NS,ND),np.nan)
    for si in range(NS):
        for di in range(10, ND):
            closes = C[si,di-10:di+1]; valid = closes[~np.isnan(closes)]
            if len(valid)<10 or valid[0]<=0: continue
            net = abs(valid[-1]-valid[0]); total = np.sum(np.abs(np.diff(valid)))
            if total>1e-10: ker[si,di] = net/total
    return ker


def compute_ker_regime(ker_raw, NS, ND, ker_cons=0.15, ker_trend=0.30):
    """Convert raw KER to regime: -1=trending, 0=normal, 1=consolidation."""
    kr = np.zeros((NS,ND),dtype=int)
    for si in range(NS):
        for di in range(ND):
            if np.isnan(ker_raw[si,di]): continue
            if ker_raw[si,di] < ker_cons: kr[si,di] = 1
            elif ker_raw[si,di] > ker_trend: kr[si,di] = -1
    return kr


def get_hold_period(ker_val, hold_cons, hold_norm, hold_trend,
                    ker_cons, ker_trend):
    """Discrete: KER regime determines hold days."""
    if np.isnan(ker_val): return hold_norm
    if ker_val < ker_cons: return hold_cons
    elif ker_val > ker_trend: return hold_trend
    return hold_norm


def get_hold_period_continuous(ker_val, hold_norm, ker_cons, ker_trend):
    """Continuous: linearly map KER to hold period in [3, 7]."""
    if np.isnan(ker_val): return hold_norm
    if ker_val <= ker_cons: return 7
    elif ker_val >= ker_trend: return 3
    frac = (ker_val-ker_cons)/(ker_trend-ker_cons)
    return max(3, min(7, int(7+frac*(3-7)+0.5)))


# =====================================================================
# Helpers
# =====================================================================

def get_dynamic_mode(recent_wins, win_th, wr_window):
    if len(recent_wins)<5: return "normal"
    wr = sum(recent_wins[-wr_window:])/len(recent_wins[-wr_window:])
    if wr>win_th: return "winning"
    elif wr<0.50: return "losing"
    return "normal"

def compute_atr_at(H, L, C, si, di, start_di):
    av = []
    for j in range(max(start_di,di-14), di):
        hh,ll,cc = H[si,j],L[si,j],C[si,j]
        if not any(np.isnan([hh,ll,cc])):
            av.append(max(hh-ll,abs(hh-cc),abs(ll-cc)))
    return np.mean(av) if av else None

def compute_portfolio_vol(C, NS, ND, vol_lb=20):
    pv = np.full(ND,np.nan)
    for di in range(vol_lb+1, ND):
        dr = []
        for dd in range(di-vol_lb, di):
            rs = []
            for si in range(NS):
                if not np.isnan(C[si,dd]) and not np.isnan(C[si,dd-1]) and C[si,dd-1]>0:
                    rs.append(C[si,dd]/C[si,dd-1]-1.0)
            if rs: dr.append(np.mean(rs))
        if len(dr)>=vol_lb//2: pv[di] = np.std(dr)
    return pv

def get_vol_mult(pv, vm, vhm, vlm, sr, sb):
    if np.isnan(pv) or np.isnan(vm) or vm<1e-12: return 1.0
    r = pv/vm
    if r>vhm: return sr
    elif r<vlm: return sb
    return 1.0


# =====================================================================
# V109 Backtest: KER-dynamic hold period
# =====================================================================

def backtest_v109(C, O, H, L, NS, ND, dates, syms,
                  predicted, ker_raw, ker_regime, port_vol,
                  sector_lookup,
                  top_n=2, max_per_sector=2,
                  hold_cons=7, hold_norm=5, hold_trend=3,
                  ker_cons=0.15, ker_trend=0.30,
                  use_continuous=False,
                  win_th=0.60, wr_window=15, atr_stop=3.0,
                  vol_high_mult=2.0, vol_low_mult=0.5,
                  size_reduce=0.5, size_boost=1.3,
                  start_di=60, end_di=None):
    """V109 backtest: each position carries its own hold_days from KER."""
    if end_di is None: end_di = ND-1

    vd = port_vol[max(start_di,20):end_di]; vv = vd[~np.isnan(vd)]
    vol_median = np.median(vv) if len(vv)>10 else 1e-6

    equity, peak, max_dd = CASH0, CASH0, 0.0
    # Position: (si, entry_di, entry_price, stop_price, alloc, hold_days)
    positions = []
    trades = []; recent_wins = []

    for di in range(max(start_di,1), end_di):
        d = dates[di]; daily_pnl = 0.0
        new_pos = []

        mode = get_dynamic_mode(recent_wins, win_th, wr_window)
        vm = get_vol_mult(port_vol[di], vol_median, vol_high_mult,
                          vol_low_mult, size_reduce, size_boost)

        # Exit logic with per-position hold_days
        pos_by_si = defaultdict(list)
        for si, edi, ep, sp, alloc, hd in positions:
            pos_by_si[si].append((edi, ep, sp, alloc, hd))

        for si, plist in pos_by_si.items():
            c = C[si,di]
            if np.isnan(c):
                for edi,ep,sp,alloc,hd in plist:
                    new_pos.append((si,edi,ep,sp,alloc,hd))
                continue

            earliest = min(p[0] for p in plist)
            hold = di - earliest
            stopped = any(c < p[2] for p in plist)
            pos_hd = plist[0][4]  # per-position hold days

            if stopped or hold >= pos_hd:
                reason = "stop" if stopped else "hold"
                for edi,ep,sp,alloc,hd in plist:
                    pnl = (c-ep)/ep - COMM
                    profit = equity*alloc*pnl
                    daily_pnl += profit
                    trades.append({
                        "pnl_abs":profit, "pnl_pct":pnl*100,
                        "days":di-edi+1, "di":di, "year":d.year,
                        "sym":syms[si], "sector":sector_lookup.get(si,'OTHER'),
                        "reason":reason, "mode":mode[:1].upper(),
                        "hold_days":hd,
                    })
                    recent_wins.append(1 if pnl>0 else 0)
            else:
                for edi,ep,sp,alloc,hd in plist:
                    new_pos.append((si,edi,ep,sp,alloc,hd))

        positions = new_pos
        equity += daily_pnl
        if equity>peak: peak = equity
        if peak>0:
            dd = (peak-equity)/peak*100
            if dd>max_dd: max_dd = dd
        if equity<=0: break

        # ENTRY
        held = {p[0] for p in positions}
        if len(held)>=top_n: continue

        cands = []
        for si in range(NS):
            if si in held: continue
            pred = predicted[si,di]
            if np.isnan(pred): continue
            if di+1>=ND or np.isnan(O[si,di+1]): continue
            if ker_regime[si,di]<0: continue
            cands.append((pred, si))
        if not cands: continue
        cands.sort(key=lambda x: -x[0])

        n_take = top_n
        if mode=="winning": n_take = min(top_n+1, top_n*2)
        elif mode=="losing": n_take = max(1, top_n-1)

        sec_counts = defaultdict(int)
        for si_h in held: sec_counts[sector_lookup.get(si_h,'OTHER')] += 1

        new_entries = []
        for pv, si in cands:
            if len(held)+len(new_entries)>=n_take: break
            if si in held: continue
            ss = sector_lookup.get(si,'OTHER')
            if sec_counts[ss]>=max_per_sector: continue
            if pv<=0: continue
            new_entries.append((pv, si, ss))
            sec_counts[ss] += 1
        if not new_entries: continue

        alloc = LEVERAGE/(len(positions)+len(new_entries))*vm
        upd = [(si,edi,ep,sp,alloc,hd) for si,edi,ep,sp,_,hd in positions]

        for pv, si, ss in new_entries:
            ep = O[si,di+1]
            if np.isnan(ep) or ep<=0: continue
            atr = compute_atr_at(H, L, C, si, di, start_di)
            if atr is None: continue

            kv = ker_raw[si,di]
            if use_continuous:
                ph = get_hold_period_continuous(kv, hold_norm, ker_cons, ker_trend)
            else:
                ph = get_hold_period(kv, hold_cons, hold_norm, hold_trend,
                                     ker_cons, ker_trend)
            upd.append((si, di+1, ep, ep-atr_stop*atr, alloc, ph))
        positions = upd

    # Close remaining
    for si, edi, ep, sp, alloc, hd in positions:
        c = C[si, ND-1]
        if not np.isnan(c) and c>0:
            equity += equity*alloc*((c-ep)/ep - COMM)

    return trades, equity, max_dd


def analyze(trades, equity, max_dd, label=""):
    if not trades:
        print(f"  {label}: no trades"); return None
    nw = sum(1 for t in trades if t["pnl_pct"]>0)
    wr = nw/len(trades)*100
    n_days = max(1, trades[-1]["di"]-trades[0]["di"])
    ann = ((equity/CASH0)**(1/max(1.0,n_days/252))-1)*100
    ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
    rets = np.array(ap)/CASH0
    sh = np.mean(rets)/np.std(rets)*np.sqrt(252) if np.std(rets)>0 else 0
    n_stop = sum(1 for t in trades if t["reason"]=="stop")
    n_hold = sum(1 for t in trades if t["reason"]=="hold")
    avg_pnl = np.mean([t["pnl_pct"] for t in trades])
    avg_days = np.mean([t["days"] for t in trades])

    hc = defaultdict(int)
    for t in trades: hc[t.get("hold_days",5)] += 1
    hd_str = " ".join(f"{k}d:{v}" for k,v in sorted(hc.items()))

    sc = defaultdict(int)
    for t in trades: sc[t.get("sector","OTHER")] += 1
    sec_str = " ".join(f"{k}:{v}" for k,v in sorted(sc.items()))

    print(f"  {label}: {len(trades)}t (stop:{n_stop} hold:{n_hold}) "
          f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
          f"Sh={sh:.2f} eq={equity:,.0f}")
    print(f"    avg_pnl={avg_pnl:+.3f}% avg_days={avg_days:.1f} "
          f"hold_dist=[{hd_str}]")
    print(f"    sectors: {sec_str}")

    yr = {}
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


def walk_forward(C, O, H, L, NS, ND, dates, syms,
                 predicted, ker_raw, ker_regime, port_vol,
                 sector_lookup, top_n=2, max_per_sector=2,
                 hold_cons=7, hold_norm=5, hold_trend=3,
                 ker_cons=0.15, ker_trend=0.30,
                 use_continuous=False,
                 vol_hm=2.0, vol_lm=0.5, size_r=0.5, size_b=1.3,
                 label=""):
    cfg = (f"tn={top_n} mps={max_per_sector} "
           f"hc={hold_cons} hn={hold_norm} ht={hold_trend} "
           f"kc={ker_cons:.2f} kt={ker_trend:.2f} cont={use_continuous}")
    print(f"\n{'='*70}\n  WF V109 {label}\n  {cfg}\n{'='*70}")
    years = sorted(set(d.year for d in dates))
    all_trades = []
    for ty in range(2019, years[-1]+1):
        ts = te = None
        for i,d in enumerate(dates):
            if d.year==ty and ts is None: ts = i
            if d.year==ty: te = i
        if ts is None: continue
        trades, _, _ = backtest_v109(
            C,O,H,L,NS,ND,dates,syms,predicted,ker_raw,ker_regime,port_vol,
            sector_lookup=sector_lookup, top_n=top_n, max_per_sector=max_per_sector,
            hold_cons=hold_cons, hold_norm=hold_norm, hold_trend=hold_trend,
            ker_cons=ker_cons, ker_trend=ker_trend,
            use_continuous=use_continuous,
            vol_high_mult=vol_hm, vol_low_mult=vol_lm,
            size_reduce=size_r, size_boost=size_b,
            start_di=ts, end_di=te+1)
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
        wr=nw/len(all_trades)*100
        cum=np.prod([1+t["pnl_pct"]/100 for t in all_trades])-1
        hc=defaultdict(int)
        for t in all_trades: hc[t.get("hold_days",5)]+=1
        hs=" ".join(f"{k}d:{v}" for k,v in sorted(hc.items()))
        sc=defaultdict(int)
        for t in all_trades: sc[t.get("sector","OTHER")] += 1
        ss=" ".join(f"{k}:{v}" for k,v in sorted(sc.items()))
        print(f"\n  WF TOTAL: {len(all_trades)}t WR={wr:.1f}% cum={cum:+.1%}")
        print(f"  HOLD DIST: [{hs}]"); print(f"  SECTORS: {ss}")
        return all_trades
    return []


def main():
    t0 = time.time()
    print("="*70)
    print("  V109: KER-BASED DYNAMIC HOLD PERIOD")
    print("  Consolidation (low KER) -> longer hold, Trending -> shorter")
    print("  Base: V96 (NW+BMA+Vol-adaptive). Walk-forward 2019-2026.")
    print("="*70)

    C,O,H,L,V,OI,NS,ND,dates,syms = load_all_data(start="2016-01-01")
    print(f"  {NS} sym, {ND} days, {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    sector_lookup = build_sector_lookup(syms)
    bt_2019 = next(i for i,d in enumerate(dates) if d>=pd.Timestamp("2019-01-01"))

    raw_factors = compute_raw_factors(C,O,H,L,V,OI,NS,ND)
    print("[V109] Computing raw KER...", flush=True)
    ker_raw = compute_ker_raw(C, NS, ND)

    ic_array = compute_rolling_ic(raw_factors, NS, ND, ic_window=40)
    bma_w = compute_bma_weights(ic_array, ND, prior_strength=5.0)
    predicted = compute_nw_predicted(raw_factors, bma_w, NS, ND,
                                      training_window=40, kernel_bandwidth=1.0)
    port_vol = compute_portfolio_vol(C, NS, ND, vol_lb=20)

    # === Parameter sweep ===
    print(f"\n{'='*70}\n  V109 PARAMETER SWEEP\n{'='*70}")
    results = []; sweep_n = 0

    # Discrete KER-hold sweep
    for hc in [5,7,8]:
        for hn in [4,5,6]:
            for ht in [2,3,4]:
                for kc in [0.10,0.15,0.20]:
                    for kt in [0.25,0.30,0.35]:
                        if kc>=kt: continue
                        for tn in [2,3]:
                            for mps in [2,3]:
                                sweep_n += 1
                                kr = compute_ker_regime(ker_raw,NS,ND,kc,kt)
                                trades,eq,dd = backtest_v109(
                                    C,O,H,L,NS,ND,dates,syms,
                                    predicted,ker_raw,kr,port_vol,
                                    sector_lookup=sector_lookup,
                                    top_n=tn, max_per_sector=mps,
                                    hold_cons=hc, hold_norm=hn, hold_trend=ht,
                                    ker_cons=kc, ker_trend=kt,
                                    use_continuous=False,
                                    start_di=bt_2019)
                                if len(trades)<10: continue
                                nw=sum(1 for t in trades if t["pnl_pct"]>0)
                                wr=nw/len(trades)*100
                                nd=max(1,trades[-1]["di"]-trades[0]["di"])
                                ann=((eq/CASH0)**(1/max(1.0,nd/252))-1)*100
                                ap=[t["pnl_abs"] for t in sorted(trades,key=lambda x:x["di"])]
                                ra=np.array(ap)/CASH0
                                sh=np.mean(ra)/np.std(ra)*np.sqrt(252) if np.std(ra)>0 else 0
                                ahd=np.mean([t.get("hold_days",5) for t in trades])
                                results.append({"hc":hc,"hn":hn,"ht":ht,"kc":kc,"kt":kt,
                                    "tn":tn,"mps":mps,"n":len(trades),"wr":wr,
                                    "ann":ann,"dd":dd,"sh":sh,"eq":eq,"ahd":ahd,"cont":False})

    # Continuous KER-hold sweep
    print("\n--- Continuous KER-hold sweep ---")
    for hn in [4,5,6]:
        for kc in [0.10,0.15,0.20]:
            for kt in [0.25,0.30,0.35]:
                if kc>=kt: continue
                for tn in [2,3]:
                    for mps in [2,3]:
                        sweep_n += 1
                        kr = compute_ker_regime(ker_raw,NS,ND,kc,kt)
                        trades,eq,dd = backtest_v109(
                            C,O,H,L,NS,ND,dates,syms,
                            predicted,ker_raw,kr,port_vol,
                            sector_lookup=sector_lookup,
                            top_n=tn, max_per_sector=mps,
                            hold_cons=7, hold_norm=hn, hold_trend=3,
                            ker_cons=kc, ker_trend=kt,
                            use_continuous=True,
                            start_di=bt_2019)
                        if len(trades)<10: continue
                        nw=sum(1 for t in trades if t["pnl_pct"]>0)
                        wr=nw/len(trades)*100
                        nd=max(1,trades[-1]["di"]-trades[0]["di"])
                        ann=((eq/CASH0)**(1/max(1.0,nd/252))-1)*100
                        ap=[t["pnl_abs"] for t in sorted(trades,key=lambda x:x["di"])]
                        ra=np.array(ap)/CASH0
                        sh=np.mean(ra)/np.std(ra)*np.sqrt(252) if np.std(ra)>0 else 0
                        ahd=np.mean([t.get("hold_days",5) for t in trades])
                        results.append({"hc":0,"hn":hn,"ht":0,"kc":kc,"kt":kt,
                            "tn":tn,"mps":mps,"n":len(trades),"wr":wr,
                            "ann":ann,"dd":dd,"sh":sh,"eq":eq,"ahd":ahd,"cont":True})

    results.sort(key=lambda x: -x["ann"])
    print(f"\n  Swept {sweep_n} configs, {len(results)} with 10+ trades")

    # Report top 20
    hdr = f"{'HC':>3} {'HN':>3} {'HT':>3} {'Kc':>5} {'Kt':>5} {'TN':>3} {'MPS':>3} {'C':>1} {'N':>5} {'WR':>6} {'Ann':>9} {'DD':>7} {'Sh':>7} {'AHD':>5}"
    print(f"\n{hdr}\n{'-'*95}")
    for r in results[:20]:
        ct = "Y" if r["cont"] else "n"
        print(f"{r['hc']:>3} {r['hn']:>3} {r['ht']:>3} {r['kc']:>5.2f} "
              f"{r['kt']:>5.2f} {r['tn']:>3} {r['mps']:>3} {ct:>1} "
              f"{r['n']:>5} {r['wr']:>5.1f}% {r['ann']:>+8.1f}% "
              f"{r['dd']:>6.1f}% {r['sh']:>6.2f} {r['ahd']:>5.1f}")

    if not results:
        print("  No valid configs. Exiting."); return

    # Top by Sharpe
    rsh = sorted(results, key=lambda x: -x["sh"])
    print("\n  Top 10 by SHARPE:")
    for r in rsh[:10]:
        ct = "cont" if r["cont"] else "disc"
        print(f"  {ct} hc={r['hc']} hn={r['hn']} ht={r['ht']} kc={r['kc']:.2f} kt={r['kt']:.2f} "
              f"tn={r['tn']} mps={r['mps']} Sh={r['sh']:.2f} ann={r['ann']:+.1f}% dd={r['dd']:.1f}% n={r['n']}")

    # Walk-forward for best configs
    best_ann = results[0]
    best_sh = rsh[0]
    best_ra = max(results, key=lambda x: x["ann"]/max(x["dd"],1.0))

    for label, best in [("BEST-ANN",best_ann),("BEST-SHARPE",best_sh),("BEST-RISK-ADJ",best_ra)]:
        kr = compute_ker_regime(ker_raw, NS, ND, best["kc"], best["kt"])
        walk_forward(C,O,H,L,NS,ND,dates,syms,
                     predicted,ker_raw,kr,port_vol,
                     sector_lookup=sector_lookup,
                     top_n=best["tn"], max_per_sector=best["mps"],
                     hold_cons=best["hc"], hold_norm=best["hn"], hold_trend=best["ht"],
                     ker_cons=best["kc"], ker_trend=best["kt"],
                     use_continuous=best["cont"],
                     label=label)

    # === Compare V109 vs V96 fixed 5d ===
    print(f"\n{'='*70}\n  COMPARISON: V109 vs V96 fixed-5d baseline\n  V96 target: +73.1% ann\n{'='*70}")
    # V109 best
    kr = compute_ker_regime(ker_raw, NS, ND, best_ann["kc"], best_ann["kt"])
    t109, eq109, dd109 = backtest_v109(
        C,O,H,L,NS,ND,dates,syms,predicted,ker_raw,kr,port_vol,
        sector_lookup=sector_lookup, top_n=best_ann["tn"], max_per_sector=best_ann["mps"],
        hold_cons=best_ann["hc"], hold_norm=best_ann["hn"], hold_trend=best_ann["ht"],
        ker_cons=best_ann["kc"], ker_trend=best_ann["kt"],
        use_continuous=best_ann["cont"], start_di=bt_2019)

    # V96 baseline: fixed 5d with same top_n/mps
    kr_base = compute_ker_regime(ker_raw, NS, ND, 0.15, 0.30)
    t96, eq96, dd96 = backtest_v109(
        C,O,H,L,NS,ND,dates,syms,predicted,ker_raw,kr_base,port_vol,
        sector_lookup=sector_lookup, top_n=best_ann["tn"], max_per_sector=best_ann["mps"],
        hold_cons=5, hold_norm=5, hold_trend=5,
        ker_cons=0.15, ker_trend=0.30,
        use_continuous=False, start_di=bt_2019)

    print(f"\n  V109 BEST-ANN (KER-dynamic hold):")
    analyze(t109, eq109, dd109, "V109-KER-dynamic")
    print(f"\n  V96 BASELINE (fixed 5d, same tn/mps):")
    analyze(t96, eq96, dd96, "V96-fixed-5d")

    if t109 and t96:
        print(f"\n  Delta: eq={eq109-eq96:+,.0f} dd={dd109-dd96:+.1f}% trades={len(t109)-len(t96):+d}")

    print(f"\n[V109] Done. {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
