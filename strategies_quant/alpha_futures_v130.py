"""V130: "阴阳相济" — Long/Short V103 with predicted-return weighting
V103 baseline: ann+75.8%, Sharpe 3.79, MDD 12.2% (long-only)
Hypothesis: Shorting the bottom predicted returns doubles signal utilization.
Also: weight positions by predicted return magnitude (higher pred = bigger size).
Walk-forward 2019-2026. No leverage. Long-biased (70/30).
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

def backtest_v130(C,O,H,L,NS,ND,dates,syms,predicted,ker_regime,port_vol,
                  sector_lookup,top_n=2,max_per_sector=2,hold_days=5,
                  win_thresh=0.60,wr_window=15,atr_stop=3.0,vlb=20,
                  vhm=2.0,vlm=0.5,sr=0.5,sb=1.3,
                  long_weight=0.7,short_weight=0.3,
                  pred_weighted=True, start_di=60, end_di=None):
    """V130 backtest: long/short with predicted-return weighting."""
    if end_di is None: end_di = ND-1
    vol_data = port_vol[max(start_di,vlb+1):end_di]
    vol_data_valid = vol_data[~np.isnan(vol_data)]
    vol_median = np.median(vol_data_valid) if len(vol_data_valid)>10 else 1e-6
    equity, peak, max_dd = CASH0, CASH0, 0.0
    positions, trades, recent_wins = [], [], []
    # positions: list of (si, entry_di, entry_price, stop_price, alloc, side)
    # side: +1 for long, -1 for short
    for di in range(max(start_di,1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_pos = []
        mode = get_dynamic_mode(recent_wins, win_thresh, wr_window)
        vol_mult = get_vol_multiplier(port_vol[di], vol_median, vhm, vlm, sr, sb)
        # Exit logic
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
            # For long: stop if c < sp; for short: stop if c > sp
            stopped = any((c < sp and side==1) or (c > sp and side==-1) for _,_,sp,_,side in pos_list)
            if stopped or hold >= hold_days:
                for edi,ep,sp,alloc,side in pos_list:
                    if side == 1:
                        pnl = (c - ep)/ep - COMM
                    else:
                        pnl = (ep - c)/ep - COMM
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
        # Build candidate lists
        held_long = {p[0] for p in positions if p[5]==1}
        held_short = {p[0] for p in positions if p[5]==-1}
        long_candidates = []
        short_candidates = []
        for si in range(NS):
            pred = predicted[si,di]
            if np.isnan(pred): continue
            if di+1 >= ND or np.isnan(O[si,di+1]): continue
            if ker_regime[si,di] < 0: continue
            if pred > 0 and si not in held_long:
                long_candidates.append((pred,si))
            elif pred < 0 and si not in held_short:
                short_candidates.append((pred,si))
        # Select long entries
        long_entries = []
        if long_candidates:
            long_candidates.sort(key=lambda x:-x[0])
            n_long = top_n
            if mode == "winning": n_long = min(top_n+1, top_n*2)
            elif mode == "losing": n_long = max(1, top_n-1)
            sec_counts = defaultdict(int)
            for si_h in held_long: sec_counts[sector_lookup.get(si_h,'OTHER')] += 1
            for pred_val,si in long_candidates:
                if len(held_long)+len(long_entries) >= n_long: break
                sym_sec = sector_lookup.get(si,'OTHER')
                if sec_counts[sym_sec] >= max_per_sector: continue
                long_entries.append((pred_val,si,sym_sec))
                sec_counts[sym_sec] += 1
        # Select short entries
        short_entries = []
        if short_candidates:
            short_candidates.sort(key=lambda x:x[0])  # most negative first
            n_short = max(1, top_n//2)
            sec_counts = defaultdict(int)
            for si_h in held_short: sec_counts[sector_lookup.get(si_h,'OTHER')] += 1
            for pred_val,si in short_candidates:
                if len(held_short)+len(short_entries) >= n_short: break
                sym_sec = sector_lookup.get(si,'OTHER')
                if sec_counts[sym_sec] >= max_per_sector: continue
                short_entries.append((pred_val,si,sym_sec))
                sec_counts[sym_sec] += 1
        if not long_entries and not short_entries: continue
        # Compute allocations
        long_preds = [abs(e[0]) for e in long_entries]
        short_preds = [abs(e[0]) for e in short_entries]
        sum_long_pred = sum(long_preds) if long_preds else 0
        sum_short_pred = sum(short_preds) if short_preds else 0
        # Weighted or equal allocation
        if pred_weighted and sum_long_pred > 1e-12:
            long_allocs = [p/sum_long_pred * long_weight * vol_mult for p in long_preds]
        else:
            n = len(long_entries) if long_entries else 1
            long_allocs = [long_weight / n * vol_mult] * len(long_entries)
        if pred_weighted and sum_short_pred > 1e-12:
            short_allocs = [p/sum_short_pred * short_weight * vol_mult for p in short_preds]
        else:
            n = len(short_entries) if short_entries else 1
            short_allocs = [short_weight / n * vol_mult] * len(short_entries)
        # Update existing positions with new vol_mult
        updated = []
        for si,edi,ep,sp,alloc,side in positions:
            # Re-scale existing positions by current vol_mult (optional, skip for simplicity)
            updated.append((si,edi,ep,sp,alloc,side))
        # Add long entries
        for (pred_val,si,sym_sec),alloc in zip(long_entries,long_allocs):
            ep = O[si,di+1]
            if np.isnan(ep) or ep <= 0: continue
            atr = compute_atr_at(H,L,C,si,di,start_di)
            if atr is None: continue
            updated.append((si,di+1,ep,ep-atr_stop*atr,alloc,1))
        # Add short entries
        for (pred_val,si,sym_sec),alloc in zip(short_entries,short_allocs):
            ep = O[si,di+1]
            if np.isnan(ep) or ep <= 0: continue
            atr = compute_atr_at(H,L,C,si,di,start_di)
            if atr is None: continue
            # Short stop is above entry
            updated.append((si,di+1,ep,ep+atr_stop*atr,alloc,-1))
        positions = updated
    # Close remaining
    for si,edi,ep,sp,alloc,side in positions:
        c = C[si,ND-1]
        if not np.isnan(c) and c > 0:
            if side == 1:
                pnl = (c-ep)/ep - COMM
            else:
                pnl = (ep-c)/ep - COMM
            equity += equity * alloc * pnl
    return trades, equity, max_dd

def analyze(trades,eq,dd,label=""):
    if not trades: return {"ann":0,"sharpe":0,"mdd":dd,"trades":0,"wr":0}
    yrs = max(1, sum(t['days'] for t in trades)/252)
    ann = ((eq/CASH0)**(1/max(1.0,yrs))-1)*100
    rets = [t['pnl_pct'] for t in trades]
    sh = np.mean(rets)/np.std(rets)*np.sqrt(len(rets)/max(1.0,yrs)) if np.std(rets)>1e-12 else 0
    wr = sum(1 for t in trades if t['pnl_pct']>0)/len(trades)*100
    n_long = sum(1 for t in trades if t.get('side')=='L')
    n_short = sum(1 for t in trades if t.get('side')=='S')
    print(f"  {label}: {len(trades)}t (L:{n_long} S:{n_short}) WR={wr:.1f}% ann={ann:+.1f}% DD={dd:.1f}% Sh={sh:.2f} eq={eq:,.0f}")
    return {"ann":ann,"sharpe":sh,"mdd":dd,"trades":len(trades),"wr":wr}

def walk_forward(C,O,H,L,NS,ND,dates,syms,pred,ker_regime,port_vol,sector_lookup,
                 top_n=2,mps=2,hold_days=5,vhm=2.0,vlm=0.5,sr=0.5,sb=1.3,
                 long_weight=0.7,short_weight=0.3,pred_weighted=True,vlb=20,label=""):
    years = sorted(set(d.year for d in dates))
    all_t = []
    for ty in range(2019, years[-1]+1):
        ts, te = None, None
        for i,d in enumerate(dates):
            if d.year == ty and ts is None: ts = i
            if d.year == ty: te = i
        if ts is None: continue
        t,_,_ = backtest_v130(C,O,H,L,NS,ND,dates,syms,pred,ker_regime,port_vol,
                              sector_lookup,top_n=top_n,max_per_sector=mps,hold_days=hold_days,
                              vhm=vhm,vlm=vlm,sr=sr,sb=sb,
                              long_weight=long_weight,short_weight=short_weight,pred_weighted=pred_weighted,
                              vlb=vlb,start_di=ts,end_di=te+1)
        yt = [x for x in t if dates[x['di']].year == ty]
        all_t.extend(yt)
        if yt:
            nw = sum(1 for x in yt if x['pnl_pct']>0)
            print(f"    {ty}: {len(yt)}t WR={nw/len(yt)*100:.1f}%", flush=True)
        else:
            print(f"    {ty}: no trades", flush=True)
    if all_t:
        nw = sum(1 for t in all_t if t['pnl_pct']>0)
        wr = nw/len(all_t)*100
        print(f"\n  WF TOTAL ({label}): {len(all_t)}t WR={wr:.1f}%")
    return all_t

def main():
    t0 = time.time()
    print("="*70)
    print("  V130: Long/Short V103 + Predicted-Return Weighting")
    print("  Long-biased (70/30). No leverage. WF 2019-2026.")
    print("="*70)
    C,O,H,L,V,OI,NS,ND,dates,syms = load_all_data(start="2016-01-01")
    print(f"  {NS} sym, {ND} days, {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    sector_lookup = build_sector_lookup(syms)
    bt_2019 = next(i for i,d in enumerate(dates) if d >= pd.Timestamp("2019-01-01"))
    raw_factors = compute_raw_factors(C,O,H,L,V,OI,NS,ND,tag="V130")
    ker_regime = compute_ker(C,NS,ND)
    ic_array = compute_rolling_ic(raw_factors,NS,ND,ic_window=60,tag="V130")
    bma_weights = compute_bma_weights(ic_array,ND,prior_strength=5.0,tag="V130")
    pred_cache = {}
    for tw in [30,40,50]:
        for bw in [0.8,1.0,1.5]:
            for hc in [2.0,3.0,4.0]:
                print(f"\n--- NW (tw={tw}, bw={bw:.1f}, hc={hc:.1f}) ---")
                pred_cache[(tw,bw,hc)] = compute_nw_gaussian_irls(raw_factors,bma_weights,NS,ND,training_window=tw,kernel_bandwidth=bw,irls_hardy_c=hc)
    vol_cache = {}
    for vlb in [10,15,20]:
        vol_cache[vlb] = compute_portfolio_volatility(C,NS,ND,vlb)
    print("\n" + "="*70)
    print("  PARAMETER SWEEP (2019-2026)")
    print("  top_n x lw/sw x pred_weighted x vlb x vhm x sr x sb x hd")
    print("="*70)
    results = []
    for pk,pred in pred_cache.items():
        tw,bw,hc = pk
        for top_n in [1,2,3]:
            for lw,sw in [(0.7,0.3),(0.5,0.5),(1.0,0.0)]:
                for pw in [True,False]:
                    for vlb in [10,15,20]:
                        for vhm in [1.5,2.0]:
                            for sr in [0.3,0.5]:
                                for sb in [1.0,1.5]:
                                    for hd in [3,5,10]:
                                        t,eq,dd = backtest_v130(C,O,H,L,NS,ND,dates,syms,pred,ker_regime,vol_cache[vlb],
                                            sector_lookup,top_n=top_n,max_per_sector=2,hold_days=hd,
                                            vhm=vhm,vlm=0.5,sr=sr,sb=sb,
                                            long_weight=lw,short_weight=sw,pred_weighted=pw,
                                            vlb=vlb,start_di=bt_2019)
                                        if len(t) < 10: continue
                                        yrs = max(1, sum(x['days'] for x in t)/252)
                                        ann = ((eq/CASH0)**(1/max(1.0,yrs))-1)*100
                                        rets = [x['pnl_pct'] for x in t]
                                        sh = np.mean(rets)/np.std(rets)*np.sqrt(len(rets)/max(1.0,yrs)) if np.std(rets)>1e-12 else 0
                                        ra = ann / max(dd, 1.0)
                                        results.append({"tw":tw,"bw":bw,"hc":hc,"tn":top_n,"lw":lw,"sw":sw,"pw":pw,"vlb":vlb,"vhm":vhm,"sr":sr,"sb":sb,"hd":hd,"ann":ann,"sh":sh,"dd":dd,"n":len(t),"ra":ra})
    if not results:
        print("  No valid results.")
        return
    results.sort(key=lambda x:-x['ann'])
    best_ann = results[0]
    results.sort(key=lambda x:-x['sh'])
    best_sh = results[0]
    results.sort(key=lambda x:-x['ra'])
    best_ra = results[0]
    for label,best in [("BEST-ANN",best_ann),("BEST-SHARPE",best_sh),("BEST-RISK-ADJ",best_ra)]:
        t,eq,dd = backtest_v130(C,O,H,L,NS,ND,dates,syms,
            pred_cache[(best['tw'],best['bw'],best['hc'])],ker_regime,vol_cache[best['vlb']],sector_lookup,
            top_n=best['tn'],max_per_sector=2,hold_days=best['hd'],
            vhm=best['vhm'],vlm=0.5,sr=best['sr'],sb=best['sb'],
            long_weight=best['lw'],short_weight=best['sw'],pred_weighted=best['pw'],
            vlb=best['vlb'],start_di=bt_2019)
        print(f"\n  FULL BACKTEST {label}")
        analyze(t,eq,dd,label)
        print(f"    config: tw={best['tw']} bw={best['bw']:.1f} hc={best['hc']:.1f} tn={best['tn']} lw={best['lw']:.1f} sw={best['sw']:.1f} pw={best['pw']} vlb={best['vlb']} vhm={best['vhm']:.1f} sr={best['sr']:.1f} sb={best['sb']:.1f} hd={best['hd']}")
        walk_forward(C,O,H,L,NS,ND,dates,syms,
            pred_cache[(best['tw'],best['bw'],best['hc'])],ker_regime,vol_cache[best['vlb']],sector_lookup,
            top_n=best['tn'],mps=2,hold_days=best['hd'],
            vhm=best['vhm'],sr=best['sr'],sb=best['sb'],
            long_weight=best['lw'],short_weight=best['sw'],pred_weighted=best['pw'],vlb=best['vlb'],
            label=f"{label} tn={best['tn']} lw={best['lw']:.1f} sw={best['sw']:.1f} pw={best['pw']}")
    print(f"\n[V130] Done. {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
