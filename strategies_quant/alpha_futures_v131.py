"""V131: "阴阳初试" — Focused long/short V103 test
Quick test: does short side add alpha?
Fixed: tw=30 bw=0.8 hc=3.0 vlb=20 vhm=2.0 sr=0.5 sb=1.3 mps=2
Sweep: top_n x lw/sw x hold_days
Walk-forward 2019-2026. No leverage.
"""
import sys, os, time, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

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

def backtest_v131(C,O,H,L,NS,ND,dates,syms,predicted,ker_regime,port_vol,
                  sector_lookup,top_n=2,max_per_sector=2,hold_days=5,
                  win_thresh=0.60,wr_window=15,atr_stop=3.0,vlb=20,
                  vhm=2.0,vlm=0.5,sr=0.5,sb=1.3,
                  long_weight=0.7,short_weight=0.3,
                  start_di=60,end_di=None):
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
        for si,edi,ep,sp,alloc,side in positions:
            pos_by_si[si].append((edi,ep,sp,alloc,side))
        for si,pos_list in pos_by_si.items():
            c = C[si,di]
            if np.isnan(c):
                for edi,ep,sp,alloc,side in pos_list: new_pos.append((si,edi,ep,sp,alloc,side))
                continue
            earliest_edi = min(p[0] for p in pos_list)
            hold = di - earliest_edi
            stopped = any((c < sp and side==1) or (c > sp and side==-1) for _,_,sp,_,side in pos_list)
            if stopped or hold >= hold_days:
                for edi,ep,sp,alloc,side in pos_list:
                    pnl = ((c - ep)/ep - COMM) if side==1 else ((ep - c)/ep - COMM)
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    is_win = pnl > 0
                    trades.append({"pnl_abs":profit,"pnl_pct":pnl*100,"days":di-edi+1,"di":di,"year":d.year,"sym":syms[si],"sector":sector_lookup.get(si,'OTHER'),"reason":"stop" if stopped else "hold","mode":mode[:1].upper(),"side":"L" if side==1 else "S"})
                    recent_wins.append(1 if is_win else 0)
            else:
                for edi,ep,sp,alloc,side in pos_list: new_pos.append((si,edi,ep,sp,alloc,side))
        positions = new_pos
        equity += daily_pnl
        if equity > peak: peak = equity
        if peak > 0:
            dd = (peak - equity)/peak*100
            if dd > max_dd: max_dd = dd
        if equity <= 0: break
        held_long = {p[0] for p in positions if p[5]==1}
        held_short = {p[0] for p in positions if p[5]==-1}
        # Long candidates
        long_cands = []
        for si in range(NS):
            if si in held_long: continue
            pred = predicted[si,di]
            if np.isnan(pred) or pred <= 0: continue
            if di+1 >= ND or np.isnan(O[si,di+1]): continue
            if ker_regime[si,di] < 0: continue
            long_cands.append((pred,si))
        # Short candidates
        short_cands = []
        for si in range(NS):
            if si in held_short: continue
            pred = predicted[si,di]
            if np.isnan(pred) or pred >= 0: continue
            if di+1 >= ND or np.isnan(O[si,di+1]): continue
            if ker_regime[si,di] < 0: continue
            short_cands.append((pred,si))
        n_long = top_n
        n_short = max(1, top_n//2)
        if mode == "winning":
            n_long = min(top_n+1, top_n*2)
            n_short = max(1, n_short+1)
        elif mode == "losing":
            n_long = max(1, top_n-1)
            n_short = max(1, n_short-1)
        long_entries = []
        long_cands.sort(key=lambda x:-x[0])
        sec_counts = defaultdict(int)
        for si_h in held_long: sec_counts[sector_lookup.get(si_h,'OTHER')] += 1
        for pred_val,si in long_cands:
            if len(held_long)+len(long_entries) >= n_long: break
            sym_sec = sector_lookup.get(si,'OTHER')
            if sec_counts[sym_sec] >= max_per_sector: continue
            long_entries.append((si,pred_val))
            sec_counts[sym_sec] += 1
        short_entries = []
        short_cands.sort(key=lambda x:x[0])
        sec_counts = defaultdict(int)
        for si_h in held_short: sec_counts[sector_lookup.get(si_h,'OTHER')] += 1
        for pred_val,si in short_cands:
            if len(held_short)+len(short_entries) >= n_short: break
            sym_sec = sector_lookup.get(si,'OTHER')
            if sec_counts[sym_sec] >= max_per_sector: continue
            short_entries.append((si,pred_val))
            sec_counts[sym_sec] += 1
        if not long_entries and not short_entries: continue
        n_long_e = len(long_entries) if long_entries else 0
        n_short_e = len(short_entries) if short_entries else 0
        long_alloc = (long_weight / max(n_long_e,1) * vol_mult) if long_entries else 0
        short_alloc = (short_weight / max(n_short_e,1) * vol_mult) if short_entries else 0
        updated = []
        for si,edi,ep,sp,alloc,side in positions:
            updated.append((si,edi,ep,sp,alloc,side))
        for si,pred_val in long_entries:
            ep = O[si,di+1]
            if np.isnan(ep) or ep <= 0: continue
            atr = compute_atr_at(H,L,C,si,di,start_di)
            if atr is None: continue
            updated.append((si,di+1,ep,ep-atr_stop*atr,long_alloc,1))
        for si,pred_val in short_entries:
            ep = O[si,di+1]
            if np.isnan(ep) or ep <= 0: continue
            atr = compute_atr_at(H,L,C,si,di,start_di)
            if atr is None: continue
            updated.append((si,di+1,ep,ep+atr_stop*atr,short_alloc,-1))
        positions = updated
    for si,edi,ep,sp,alloc,side in positions:
        c = C[si,ND-1]
        if not np.isnan(c) and c > 0:
            pnl = ((c-ep)/ep - COMM) if side==1 else ((ep-c)/ep - COMM)
            equity += equity * alloc * pnl
    return trades, equity, max_dd

def main():
    t0 = time.time()
    print("="*70)
    print("  V131: Focused Long/Short Test")
    print("  Fixed: tw=30 bw=0.8 hc=3.0")
    print("  Sweep: top_n x lw/sw x hold_days")
    print("="*70)
    C,O,H,L,V,OI,NS,ND,dates,syms = load_all_data(start="2016-01-01")
    print(f"  {NS} sym, {ND} days, {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    sector_lookup = build_sector_lookup(syms)
    bt_2019 = next(i for i,d in enumerate(dates) if d >= pd.Timestamp("2019-01-01"))
    raw_factors = compute_raw_factors(C,O,H,L,V,OI,NS,ND,tag="V131")
    ker_regime = compute_ker(C,NS,ND)
    ic_array = compute_rolling_ic(raw_factors,NS,ND,ic_window=60,tag="V131")
    bma_weights = compute_bma_weights(ic_array,ND,prior_strength=5.0,tag="V131")
    pred = compute_nw_gaussian_irls(raw_factors,bma_weights,NS,ND,training_window=30,kernel_bandwidth=0.8,irls_hardy_c=3.0)
    port_vol = compute_portfolio_volatility(C,NS,ND,20)
    print("\n" + "="*70)
    print("  SWEEP")
    print("="*70)
    results = []
    for top_n in [1,2]:
        for lw,sw in [(1.0,0.0),(0.7,0.3),(0.5,0.5)]:
            for hd in [3,5]:
                t,eq,dd = backtest_v131(C,O,H,L,NS,ND,dates,syms,pred,ker_regime,port_vol,
                    sector_lookup,top_n=top_n,max_per_sector=2,hold_days=hd,
                    vhm=2.0,vlm=0.5,sr=0.5,sb=1.3,
                    long_weight=lw,short_weight=sw,start_di=bt_2019)
                if len(t) < 10: continue
                yrs = max(1, sum(x['days'] for x in t)/252)
                ann = ((eq/CASH0)**(1/max(1.0,yrs))-1)*100
                rets = [x['pnl_pct'] for x in t]
                sh = np.mean(rets)/np.std(rets)*np.sqrt(len(rets)/max(1.0,yrs)) if np.std(rets)>1e-12 else 0
                n_long = sum(1 for x in t if x.get('side')=='L')
                n_short = sum(1 for x in t if x.get('side')=='S')
                wr_long = sum(1 for x in t if x.get('side')=='L' and x['pnl_pct']>0)/max(n_long,1)*100
                wr_short = sum(1 for x in t if x.get('side')=='S' and x['pnl_pct']>0)/max(n_short,1)*100
                print(f"  tn={top_n} lw={lw:.1f} sw={sw:.1f} hd={hd} → ann={ann:+.1f}% DD={dd:.1f}% Sh={sh:.2f} L={n_long}(WR{wr_long:.0f}%) S={n_short}(WR{wr_short:.0f}%)")
                results.append({"tn":top_n,"lw":lw,"sw":sw,"hd":hd,"ann":ann,"sh":sh,"dd":dd,"n":len(t)})
    print(f"\n[V131] Done. {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
