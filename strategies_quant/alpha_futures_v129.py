"""V129: "快马加鞭" — Fast focused V103 sweep with multiprocessing
Only sweeps top_n, hold_days, tw, bw, hc. No leverage.
Walk-forward 2019-2026.
"""
import sys, os, time, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count

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

def backtest_v129(C,O,H,L,NS,ND,dates,syms,predicted,ker_regime,port_vol,
                  sector_lookup,top_n=2,max_per_sector=2,hold_days=5,
                  win_thresh=0.60,wr_window=15,atr_stop=3.0,vlb=20,
                  vhm=2.0,vlm=0.5,sr=0.5,sb=1.3,
                  leverage=1.0,start_di=60,end_di=None):
    if end_di is None: end_di = ND-1
    vol_data = port_vol[max(start_di,vlb+1):end_di]
    vol_data_valid = vol_data[~np.isnan(vol_data)]
    vol_median = np.median(vol_data_valid) if len(vol_data_valid)>10 else 1e-6
    equity, peak, max_dd = CASH0, CASH0, 0.0
    positions, trades, recent_wins = [], [], []
    for di in range(max(start_di,1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_pos = []
        mode = get_dynamic_mode(recent_wins, win_thresh, wr_window)
        vol_mult = get_vol_multiplier(port_vol[di], vol_median, vhm, vlm, sr, sb)
        pos_by_si = defaultdict(list)
        for si,edi,ep,sp,alloc in positions:
            pos_by_si[si].append((edi,ep,sp,alloc))
        for si,pos_list in pos_by_si.items():
            c = C[si,di]
            if np.isnan(c):
                for edi,ep,sp,alloc in pos_list: new_pos.append((si,edi,ep,sp,alloc))
                continue
            earliest_edi = min(p[0] for p in pos_list)
            hold = di - earliest_edi
            stopped = any(c < sp for _,_,sp,_ in pos_list)
            if stopped or hold >= hold_days:
                for edi,ep,sp,alloc in pos_list:
                    pnl = (c - ep)/ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    is_win = pnl > 0
                    trades.append({"pnl_abs":profit,"pnl_pct":pnl*100,"days":di-edi+1,"di":di,"year":d.year,"sym":syms[si],"sector":sector_lookup.get(si,'OTHER'),"reason":"stop" if stopped else "hold","mode":mode[:1].upper()})
                    recent_wins.append(1 if is_win else 0)
            else:
                for edi,ep,sp,alloc in pos_list: new_pos.append((si,edi,ep,sp,alloc))
        positions = new_pos
        equity += daily_pnl
        if equity > peak: peak = equity
        if peak > 0:
            dd = (peak - equity)/peak*100
            if dd > max_dd: max_dd = dd
        if equity <= 0: break
        held = {p[0] for p in positions}
        if len(held) >= top_n: continue
        candidates = []
        for si in range(NS):
            if si in held: continue
            pred = predicted[si,di]
            if np.isnan(pred): continue
            if di+1 >= ND or np.isnan(O[si,di+1]): continue
            if ker_regime[si,di] < 0: continue
            candidates.append((pred,si))
        if not candidates: continue
        candidates.sort(key=lambda x:-x[0])
        n_to_take = top_n
        if mode == "winning": n_to_take = min(top_n+1, top_n*2)
        elif mode == "losing": n_to_take = max(1, top_n-1)
        sec_counts = defaultdict(int)
        for si_h in held: sec_counts[sector_lookup.get(si_h,'OTHER')] += 1
        new_entries = []
        for pred_val,si in candidates:
            if len(held)+len(new_entries) >= n_to_take: break
            sym_sec = sector_lookup.get(si,'OTHER')
            if sec_counts[sym_sec] >= max_per_sector: continue
            if pred_val <= 0: continue
            new_entries.append((pred_val,si,sym_sec))
            sec_counts[sym_sec] += 1
        if not new_entries: continue
        num_total = len(positions) + len(new_entries)
        alloc_per_pos = leverage / num_total * vol_mult
        updated = []
        for si,edi,ep,sp,old_alloc in positions:
            updated.append((si,edi,ep,sp,alloc_per_pos))
        for pred_val,si,sym_sec in new_entries:
            ep = O[si,di+1]
            if np.isnan(ep) or ep <= 0: continue
            atr = compute_atr_at(H,L,C,si,di,start_di)
            if atr is None: continue
            updated.append((si,di+1,ep,ep-atr_stop*atr,alloc_per_pos))
        positions = updated
    for si,edi,ep,sp,alloc in positions:
        c = C[si,ND-1]
        if not np.isnan(c) and c > 0:
            pnl = (c-ep)/ep - COMM
            equity += equity * alloc * pnl
    return trades, equity, max_dd

def run_single(args):
    C,O,H,L,NS,ND,dates,syms,pred,ker_regime,port_vol,sector_lookup,bt_2019,tw,bw,hc,top_n,hd = args
    t,eq,dd = backtest_v129(C,O,H,L,NS,ND,dates,syms,pred,ker_regime,port_vol,
        sector_lookup,top_n=top_n,max_per_sector=2,hold_days=hd,
        vhm=2.0,vlm=0.5,sr=0.5,sb=1.3,leverage=1.0,vlb=20,start_di=bt_2019)
    if len(t) < 10: return None
    yrs = max(1, sum(x['days'] for x in t)/252)
    ann = ((eq/CASH0)**(1/max(1.0,yrs))-1)*100
    rets = [x['pnl_pct'] for x in t]
    sh = np.mean(rets)/np.std(rets)*np.sqrt(len(rets)/max(1.0,yrs)) if np.std(rets)>1e-12 else 0
    return {"tw":tw,"bw":bw,"hc":hc,"tn":top_n,"hd":hd,"ann":ann,"sh":sh,"dd":dd,"n":len(t),"ra":ann/max(dd,1.0)}

def main():
    t0 = time.time()
    print("="*70)
    print("  V129: Fast Focused V103 Sweep (multiprocessing)")
    print("  Sweep: tw x bw x hc x top_n x hold_days")
    print("="*70)
    C,O,H,L,V,OI,NS,ND,dates,syms = load_all_data(start="2016-01-01")
    print(f"  {NS} sym, {ND} days, {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    sector_lookup = build_sector_lookup(syms)
    bt_2019 = next(i for i,d in enumerate(dates) if d >= pd.Timestamp("2019-01-01"))
    raw_factors = compute_raw_factors(C,O,H,L,V,OI,NS,ND,tag="V129")
    ker_regime = compute_ker(C,NS,ND)
    ic_array = compute_rolling_ic(raw_factors,NS,ND,ic_window=60,tag="V129")
    bma_weights = compute_bma_weights(ic_array,ND,prior_strength=5.0,tag="V129")
    pred_cache = {}
    for tw in [20,30,40,50]:
        for bw in [0.5,0.8,1.0,1.5]:
            for hc in [2.0,3.0,4.0]:
                print(f"  NW tw={tw} bw={bw:.1f} hc={hc:.1f}...")
                pred_cache[(tw,bw,hc)] = compute_nw_gaussian_irls(raw_factors,bma_weights,NS,ND,training_window=tw,kernel_bandwidth=bw,irls_hardy_c=hc)
    port_vol = compute_portfolio_volatility(C,NS,ND,20)
    print(f"\n  Starting parallel sweep with {cpu_count()} cores...")
    tasks = []
    for pk,pred in pred_cache.items():
        tw,bw,hc = pk
        for top_n in [1,2,3]:
            for hd in [3,5,10]:
                tasks.append((C,O,H,L,NS,ND,dates,syms,pred,ker_regime,port_vol,sector_lookup,bt_2019,tw,bw,hc,top_n,hd))
    print(f"  Total combinations: {len(tasks)}")
    with Pool(processes=max(1,cpu_count()-1)) as pool:
        results = pool.map(run_single, tasks)
    results = [r for r in results if r is not None]
    print(f"  Valid results: {len(results)}")
    if not results:
        print("  No valid results.")
        return
    results.sort(key=lambda x:-x['ann'])
    best_ann = results[0]
    results.sort(key=lambda x:-x['sh'])
    best_sh = results[0]
    results.sort(key=lambda x:-x['ra'])
    best_ra = results[0]
    print("\n" + "="*70)
    print("  TOP 10 BY ANNUALIZED RETURN")
    print("="*70)
    for r in results[:10]:
        print(f"  tw={r['tw']} bw={r['bw']:.1f} hc={r['hc']:.1f} tn={r['tn']} hd={r['hd']} → ann={r['ann']:+.1f}% DD={r['dd']:.1f}% Sh={r['sh']:.2f} n={r['n']}")
    print("\n" + "="*70)
    print("  TOP 10 BY SHARPE")
    print("="*70)
    for r in results[:10]:
        print(f"  tw={r['tw']} bw={r['bw']:.1f} hc={r['hc']:.1f} tn={r['tn']} hd={r['hd']} → ann={r['ann']:+.1f}% DD={r['dd']:.1f}% Sh={r['sh']:.2f} n={r['n']}")
    print(f"\n[V129] Done. {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
