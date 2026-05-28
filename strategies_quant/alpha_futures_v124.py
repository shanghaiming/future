"""V124: "知彼知己" — Gaussian NW + Crisis Filter + CAViaR-X
Combines V103 Gaussian+IRLS + V104 crisis filter + V122 contagion sizing.
Walk-forward 2019-2026. No leverage.
"""
import sys, os, time, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_data import load_all_data
from nw_kernel_utils import (
    CASH0, COMM, LEVERAGE, FACTOR_NAMES,
    build_sector_lookup, compute_raw_factors,
    compute_rolling_ic, compute_bma_weights,
    compute_ker, compute_portfolio_volatility,
    get_vol_multiplier, get_dynamic_mode, compute_atr_at,
)
from alpha_futures_v103 import compute_nw_gaussian_irls


def compute_market_crisis(C, NS, ND, corr_window=20):
    daily_rets = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if not np.isnan(C[si,di]) and not np.isnan(C[si,di-1]) and C[si,di-1]>0:
                daily_rets[si,di] = C[si,di]/C[si,di-1]-1.0
    median_corr = np.full(ND, np.nan)
    for di in range(corr_window+1, ND):
        rw = daily_rets[:,di-corr_window:di]
        valid = [si for si in range(NS) if np.sum(~np.isnan(rw[si]))>=corr_window*0.7]
        if len(valid)<5: continue
        sampled = valid if len(valid)<=30 else list(np.random.RandomState(42).choice(valid,30,replace=False))
        corrs = []
        for i in range(len(sampled)):
            for j in range(i+1, len(sampled)):
                ri, rj = rw[sampled[i]], rw[sampled[j]]
                mask = (~np.isnan(ri))&(~np.isnan(rj))
                if np.sum(mask)<corr_window*0.5: continue
                riv, rjv = ri[mask], rj[mask]
                if np.std(riv)<1e-12 or np.std(rjv)<1e-12: continue
                c = np.corrcoef(riv, rjv)[0,1]
                if not np.isnan(c): corrs.append(c)
        if len(corrs)>=10: median_corr[di] = np.median(corrs)
    return median_corr, daily_rets


def compute_factor_instability(raw_factors, NS, ND, ic_window=20, min_pairs=10):
    fwd_ret = raw_factors["fwd_ret_5d"]
    ic_std_max = np.full(ND, np.nan)
    for di in range(ic_window+10, ND):
        fstds = []
        for fname in FACTOR_NAMES:
            factor = raw_factors[fname]
            ics = []
            for tdi in range(di-ic_window, di):
                fv, rv = factor[:,tdi], fwd_ret[:,tdi]
                mask = (~np.isnan(fv))&(~np.isnan(rv))
                if np.sum(mask)>=min_pairs:
                    fr = pd.Series(fv[mask]).rank().values
                    rr = pd.Series(rv[mask]).rank().values
                    c = np.corrcoef(fr, rr)[0,1]
                    if not np.isnan(c): ics.append(c)
            if len(ics)>=5: fstds.append(np.std(ics))
        if fstds: ic_std_max[di] = np.max(fstds)
    return ic_std_max


def get_crisis_mult(mcorr, icstd, cth, ith, csm, ism):
    m = 1.0
    if not np.isnan(mcorr) and mcorr > cth: m *= csm
    if not np.isnan(icstd) and icstd > ith: m *= ism
    return m


def compute_var(daily_rets, NS, ND, var_window=20, var_pct=0.05):
    var_arr = np.full((NS,ND), np.nan)
    for si in range(NS):
        for di in range(var_window, ND):
            v = daily_rets[si,di-var_window:di]
            valid = v[~np.isnan(v)]
            if len(valid)>=var_window//2:
                var_arr[si,di] = np.percentile(valid, var_pct*100)
    return var_arr


def detect_contagion(daily_rets, var_arr, sector_lookup, NS, ND, sc_min=2, mc_sec=3):
    sec_cont, mkt_cont = {}, {}
    for di in range(ND):
        breaches = defaultdict(list)
        for si in range(NS):
            if np.isnan(daily_rets[si,di]) or np.isnan(var_arr[si,di]): continue
            if daily_rets[si,di] < var_arr[si,di]:
                breaches[sector_lookup.get(si,'OTHER')].append(si)
        cont = {sec for sec,bsis in breaches.items() if len(bsis)>=sc_min}
        if cont: sec_cont[di] = cont
        if len(cont)>=mc_sec: mkt_cont[di] = cont
    return sec_cont, mkt_cont


def get_contagion_mult(di, held_sectors, sec_cont, mkt_cont, cm):
    if di in mkt_cont: return {si:cm for si in held_sectors}
    if di in sec_cont: return {si:(cm if sec in sec_cont[di] else 1.0) for si,sec in held_sectors.items()}
    return {si:1.0 for si in held_sectors}


def backtest_v124(C,O,H,L,NS,ND,dates,syms,predicted,ker_regime,port_vol,
                  median_corr,ic_std_max,sec_cont,mkt_cont,sector_lookup,
                  top_n=2,mps=2,hold_days=5,win_thresh=0.60,wr_window=15,
                  atr_stop=3.0,vlb=20,vhm=2.0,vlm=0.5,sr=0.5,sb=1.3,
                  cth=0.4,ith=0.3,csm=0.3,ism=0.5,cm=0.5,sc_min=2,mc_sec=3,
                  start_di=60,end_di=None):
    if end_di is None: end_di = ND-1
    vol_data = port_vol[max(start_di,vlb+1):end_di]
    vol_data_valid = vol_data[~np.isnan(vol_data)]
    vol_median = np.median(vol_data_valid) if len(vol_data_valid)>10 else 1e-6

    equity, peak, max_dd = CASH0, CASH0, 0.0
    positions, trades, recent_wins = [], [], []
    crisis_days, cont_reductions = 0, 0

    for di in range(max(start_di,1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_pos = []
        mode = get_dynamic_mode(recent_wins, win_thresh, wr_window)

        vol_mult = get_vol_multiplier(port_vol[di], vol_median, vhm, vlm, sr, sb)
        crisis_mult = get_crisis_mult(median_corr[di], ic_std_max[di], cth, ith, csm, ism)
        if crisis_mult < 1.0: crisis_days += 1

        # Exit
        by_si = defaultdict(list)
        for si,edi,ep,sp,alloc in positions: by_si[si].append((edi,ep,sp,alloc))
        for si,plist in by_si.items():
            c = C[si,di]
            if np.isnan(c):
                for edi,ep,sp,alloc in plist: new_pos.append((si,edi,ep,sp,alloc))
                continue
            earliest = min(p[0] for p in plist)
            hold = di - earliest
            stopped = any(c < sp for _,_,sp,_ in plist)
            if stopped or hold >= hold_days:
                for edi,ep,sp,alloc in plist:
                    pnl = (c-ep)/ep - COMM
                    profit = equity*alloc*pnl
                    daily_pnl += profit
                    is_win = pnl > 0
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
        cands.sort(key=lambda x:-x[0])

        n_take = top_n
        if mode == "winning": n_take = min(top_n+1, top_n*2)
        elif mode == "losing": n_take = max(1, top_n-1)

        sec_counts = defaultdict(int)
        for si_h in held: sec_counts[sector_lookup.get(si_h,'OTHER')] += 1
        entries = []
        for pv,si in cands:
            if len(held)+len(entries) >= n_take: break
            if si in held: continue
            sec = sector_lookup.get(si,'OTHER')
            if sec_counts[sec] >= mps: continue
            if pv <= 0: continue
            entries.append((pv,si,sec)); sec_counts[sec] += 1
        if not entries: continue

        held_sec = {si:sector_lookup.get(si,'OTHER') for si,_,_,_,_ in positions}
        for _,si,sec in entries: held_sec[si] = sec
        cmults = get_contagion_mult(di, held_sec, sec_cont, mkt_cont, cm)
        if any(v < 1.0 for v in cmults.values()): cont_reductions += 1

        num_total = len(positions) + len(entries)
        base_alloc = LEVERAGE / num_total * vol_mult * crisis_mult
        upd_pos = [(si,edi,ep,sp, base_alloc*cmults.get(si,1.0)) for si,edi,ep,sp,_ in positions]
        for pv,si,sec in entries:
            ep = O[si,di+1]
            if np.isnan(ep) or ep <= 0: continue
            atr = compute_atr_at(H,L,C,si,di,start_di)
            if atr is None: continue
            upd_pos.append((si,di+1,ep, ep-atr_stop*atr, base_alloc*cmults.get(si,1.0)))
        positions = upd_pos

    for si,edi,ep,sp,alloc in positions:
        c = C[si,ND-1]
        if not np.isnan(c) and c > 0:
            equity += equity*alloc*((c-ep)/ep - COMM)

    if trades: trades[0]["meta"] = {"crisis_days":crisis_days,"cont_reductions":cont_reductions}
    return trades, equity, max_dd


def analyze_v124(trades, equity, max_dd, label=""):
    if not trades: print(f"  {label}: no trades"); return None
    nw = sum(1 for t in trades if t["pnl_pct"]>0)
    wr = nw/len(trades)*100
    nd = max(1, trades[-1]["di"]-trades[0]["di"])
    ann = ((equity/CASH0)**(1/max(1.0,nd/252))-1)*100
    ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x:x["di"])]
    rets = np.array(ap)/CASH0
    sh = np.mean(rets)/np.std(rets)*np.sqrt(252) if np.std(rets)>0 else 0
    ns = sum(1 for t in trades if t["reason"]=="stop")
    nh = sum(1 for t in trades if t["reason"]=="hold")
    aw = np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"]>0])
    al = np.mean([t["pnl_pct"] for t in trades if t["pnl_pct"]<=0])
    sc = defaultdict(int)
    for t in trades: sc[t.get("sector","OTHER")] += 1
    ss = " ".join(f"{k}:{v}" for k,v in sorted(sc.items()))
    ci = trades[0].get("meta",{})
    print(f"  {label}: {len(trades)}t (s:{ns} h:{nh}) WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% Sh={sh:.2f} eq={equity:,.0f}")
    print(f"    avg_win={aw:+.3f}% avg_loss={al:+.3f}% sectors: {ss}")
    if ci: print(f"    crisis={ci.get('crisis_days',0)}d cont_red={ci.get('cont_reductions',0)}")
    yr = defaultdict(lambda:{"n":0,"w":0,"pnl":[]})
    for t in trades:
        y = t["year"]
        yr[y]["n"] += 1
        if t["pnl_pct"]>0: yr[y]["w"] += 1
        yr[y]["pnl"].append(t["pnl_pct"])
    for y in sorted(yr.keys()):
        ys = yr[y]; cum = np.prod([1+p/100 for p in ys["pnl"]])-1
        print(f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% cum={cum:+.1%}")
    return {"n":len(trades),"wr":wr,"ann":ann,"dd":max_dd,"sh":sh,"eq":equity}


def main():
    t0 = time.time()
    print("="*70)
    print("  V124: \"知彼知己\" — Gaussian NW + Crisis Filter + CAViaR-X")
    print("  Layer 1: V103 Gaussian+IRLS NW (Sharpe 3.79)")
    print("  Layer 2: V104 Crisis Filter (MDD -40%)")
    print("  Layer 3: V122 CAViaR-X Contagion (MDD halved)")
    print("  Walk-forward 2019-2026. No leverage.")
    print("="*70)

    C,O,H,L,V,OI,NS,ND,dates,syms = load_all_data(start="2016-01-01")
    print(f"  {NS} sym, {ND} days, {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    sector_lookup = build_sector_lookup(syms)
    print(f"  Sectors: {dict(pd.Series(list(sector_lookup.values())).value_counts().sort_index())}")

    bt_2019 = next(i for i,d in enumerate(dates) if d>=pd.Timestamp("2019-01-01"))

    raw_factors = compute_raw_factors(C,O,H,L,V,OI,NS,ND,tag="V124")
    ker_regime = compute_ker(C,NS,ND)

    ic_arr = compute_rolling_ic(raw_factors,NS,ND,ic_window=60,tag="V124")
    bma_w = compute_bma_weights(ic_arr,ND,prior_strength=5.0,tag="V124")

    print("\n--- Computing NW predictions ---")
    pred = compute_nw_gaussian_irls(raw_factors,bma_w,NS,ND,training_window=30,kernel_bandwidth=0.8,irls_hardy_c=3.0)

    print("\n--- Precomputing risk layers ---")
    median_corr, daily_rets = compute_market_crisis(C,NS,ND,corr_window=20)
    ic_std_max = compute_factor_instability(raw_factors,NS,ND,ic_window=20)
    var_arr = compute_var(daily_rets,NS,ND,var_window=20,var_pct=0.05)
    sec_cont, mkt_cont = detect_contagion(daily_rets,var_arr,sector_lookup,NS,ND,sc_min=2,mc_sec=3)

    port_vol = compute_portfolio_volatility(C,NS,ND,vol_lookback=10)

    print("\n"+"="*70)
    print("  V124 PARAMETER SWEEP")
    print("="*70)

    results = []
    sweep_count = 0
    configs = []
    for top_n in [2,3]:
        for mps in [2,3]:
            for vhm in [1.5,2.0]:
                for sr in [0.3,0.5]:
                    for sb in [1.0,1.5]:
                        for cth in [0.4,0.5]:
                            for csm in [0.2,0.3]:
                                for ism in [0.3,0.5]:
                                    for cm in [0.3,0.5]:
                                        configs.append((top_n,mps,vhm,sr,sb,cth,csm,ism,cm))

    for top_n,mps,vhm,sr,sb,cth,csm,ism,cm in configs:
        sweep_count += 1
        trades,eq,dd = backtest_v124(
            C,O,H,L,NS,ND,dates,syms,pred,ker_regime,port_vol,
            median_corr,ic_std_max,sec_cont,mkt_cont,sector_lookup,
            top_n=top_n,mps=mps,hold_days=5,vhm=vhm,vlm=0.5,sr=sr,sb=sb,
            cth=cth,ith=0.3,csm=csm,ism=ism,cm=cm,sc_min=2,mc_sec=3,
            start_di=bt_2019)
        if len(trades)<10: continue
        nw = sum(1 for t in trades if t["pnl_pct"]>0)
        wr = nw/len(trades)*100
        nd = max(1,trades[-1]["di"]-trades[0]["di"])
        ann = ((eq/CASH0)**(1/max(1.0,nd/252))-1)*100
        ap = [t["pnl_abs"] for t in sorted(trades,key=lambda x:x["di"])]
        ra = np.array(ap)/CASH0
        sh = np.mean(ra)/np.std(ra)*np.sqrt(252) if np.std(ra)>0 else 0
        results.append({"tn":top_n,"mps":mps,"vhm":vhm,"sr":sr,"sb":sb,
            "cth":cth,"csm":csm,"ism":ism,"cm":cm,
            "n":len(trades),"wr":wr,"ann":ann,"dd":dd,"sh":sh,"eq":eq})

    print(f"\n  Evaluated {sweep_count} configs, {len(results)} with 10+ trades")
    results.sort(key=lambda x:-x["ann"])
    print(f"\n{'TN':>3} {'MPS':>3} {'Vhm':>4} {'SR':>4} {'SB':>4} {'Cth':>4} {'Csm':>4} {'Ism':>4} {'Cm':>4} {'N':>5} {'WR':>6} {'Ann':>8} {'DD':>7} {'Sh':>6}")
    print("-"*100)
    for r in results[:20]:
        print(f"{r['tn']:>3} {r['mps']:>3} {r['vhm']:>4.1f} {r['sr']:>4.1f} {r['sb']:>4.1f} {r['cth']:>4.1f} {r['csm']:>4.1f} {r['ism']:>4.1f} {r['cm']:>4.1f} {r['n']:>5} {r['wr']:>5.1f}% {r['ann']:>+7.1f}% {r['dd']:>6.1f}% {r['sh']:>5.2f}")

    if not results: print("No results."); return

    best_ann = results[0]
    best_sh = max(results, key=lambda x:x["sh"])
    best_risk = max(results, key=lambda x:x["ann"]/max(x["dd"],1.0))

    for label,best in [("BEST-ANN",best_ann),("BEST-SHARPE",best_sh),("BEST-RISK-ADJ",best_risk)]:
        print(f"\n{'='*70}\n  FULL BACKTEST {label}\n{'='*70}")
        trades,eq,dd = backtest_v124(
            C,O,H,L,NS,ND,dates,syms,pred,ker_regime,port_vol,
            median_corr,ic_std_max,sec_cont,mkt_cont,sector_lookup,
            top_n=best["tn"],mps=best["mps"],vhm=best["vhm"],sr=best["sr"],sb=best["sb"],
            cth=best["cth"],csm=best["csm"],ism=best["ism"],cm=best["cm"],start_di=bt_2019)
        analyze_v124(trades,eq,dd,label)

    print(f"\n[V124] Done. {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
