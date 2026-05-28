"""V128: "三日为期" — V103 with 3-day prediction horizon
Hypothesis: Shorter prediction target (3d vs 5d) enables more frequent
compounding and better alignment with hold period.
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
    normalize_factor,
)
from alpha_futures_v103 import compute_nw_gaussian_irls

def compute_raw_factors_3d(C,O,H,L,V,OI,NS,ND,tag="V128"):
    """Compute raw factors with 3-day forward return target."""
    t0 = time.time()
    print(f"[{tag}] Computing raw factors (3d target)...", flush=True)
    raw = compute_raw_factors(C,O,H,L,V,OI,NS,ND,tag=tag)
    # Replace fwd_ret_5d with fwd_ret_3d
    fwd_ret_3d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND - 3):
            if (not np.isnan(C[si, di + 3]) and not np.isnan(C[si, di])
                    and C[si, di] > 0):
                fwd_ret_3d[si, di] = C[si, di + 3] / C[si, di] - 1.0
    raw["fwd_ret_3d"] = fwd_ret_3d
    # Keep fwd_ret_5d for IC computation
    return raw

def compute_nw_gaussian_irls_3d(raw_factors, bma_weights, NS, ND,
                                training_window=30, kernel_bandwidth=0.8,
                                irls_hardy_c=3.0, min_train=30):
    """Gaussian+IRLS NW predicting 3-day returns."""
    t0 = time.time()
    print(f"[V128] Gaussian+IRLS NW (3d target, tw={training_window}, bw={kernel_bandwidth:.1f}, hc={irls_hardy_c:.1f})...")
    predicted = np.full((NS, ND), np.nan)
    weighted_normed = {}
    for fi, fname in enumerate(FACTOR_NAMES):
        w = bma_weights[fi, :]
        w_mat = np.tile(w, (NS, 1))
        normed = normalize_factor(raw_factors[fname], NS, ND)
        weighted_normed[fname] = normed * w_mat
    fwd_ret = raw_factors["fwd_ret_3d"]
    atr_mean = raw_factors["atr_mean"]
    MIN_TRAIN = min_train
    SQRT_2PI = np.sqrt(2.0 * np.pi)
    for di in range(training_window + 3, ND):
        train_features = []
        train_targets = []
        for tdi in range(di - training_window, di - 2):
            for si in range(NS):
                feat = np.array([weighted_normed[fname][si, tdi] for fname in FACTOR_NAMES])
                target = fwd_ret[si, tdi]
                if np.any(np.isnan(feat)) or np.isnan(target):
                    continue
                train_features.append(feat)
                train_targets.append(target)
        if len(train_features) < MIN_TRAIN:
            continue
        train_X = np.array(train_features)
        train_Y = np.array(train_targets)
        feat_std = np.std(train_X, axis=0)
        feat_std[feat_std < 1e-12] = 1.0
        for si in range(NS):
            query_feat = np.array([weighted_normed[fname][si, di] for fname in FACTOR_NAMES])
            if np.any(np.isnan(query_feat)):
                continue
            atr_val = atr_mean[si, di]
            h = max(atr_val * kernel_bandwidth, 0.1) if not np.isnan(atr_val) else kernel_bandwidth
            diff = train_X - query_feat[np.newaxis, :]
            dist = np.sqrt(np.sum((diff / feat_std[np.newaxis, :]) ** 2, axis=1))
            scaled_dist = dist / h
            gauss_w = np.exp(-0.5 * scaled_dist ** 2) / SQRT_2PI
            gauss_sum = np.sum(gauss_w)
            if gauss_sum < 1e-12:
                continue
            y_hat_init = np.sum(gauss_w * train_Y) / gauss_sum
            residuals = train_Y - y_hat_init
            abs_res = np.abs(residuals)
            mad = np.median(abs_res)
            if mad > 1e-12:
                hardy_w = np.minimum(1.0, irls_hardy_c * mad / (abs_res + 1e-10))
                combined_w = hardy_w * gauss_w
            else:
                combined_w = gauss_w
            combined_sum = np.sum(combined_w)
            if combined_sum < 1e-12:
                predicted[si, di] = y_hat_init
            else:
                predicted[si, di] = np.sum(combined_w * train_Y) / combined_sum
        if di % 100 == 0:
            valid_count = np.sum(~np.isnan(predicted[:, di]))
            print(f"  di={di}/{ND} valid={valid_count}/{NS} train_size={len(train_features)}", flush=True)
    print(f"  Gaussian+IRLS (3d) done: {time.time() - t0:.1f}s", flush=True)
    return predicted

def backtest_v128(C,O,H,L,NS,ND,dates,syms,predicted,ker_regime,port_vol,
                  sector_lookup,top_n=2,max_per_sector=2,hold_days=3,
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

def analyze(trades,eq,dd,label=""):
    if not trades: return {"ann":0,"sharpe":0,"mdd":dd,"trades":0,"wr":0}
    yrs = max(1, sum(t['days'] for t in trades)/252)
    ann = ((eq/CASH0)**(1/max(1.0,yrs))-1)*100
    rets = [t['pnl_pct'] for t in trades]
    sh = np.mean(rets)/np.std(rets)*np.sqrt(len(rets)/max(1.0,yrs)) if np.std(rets)>1e-12 else 0
    wr = sum(1 for t in trades if t['pnl_pct']>0)/len(trades)*100
    print(f"  {label}: {len(trades)}t WR={wr:.1f}% ann={ann:+.1f}% DD={dd:.1f}% Sh={sh:.2f} eq={eq:,.0f}")
    return {"ann":ann,"sharpe":sh,"mdd":dd,"trades":len(trades),"wr":wr}

def walk_forward(C,O,H,L,NS,ND,dates,syms,pred,ker_regime,port_vol,sector_lookup,
                 top_n=2,mps=2,hold_days=3,vhm=2.0,vlm=0.5,sr=0.5,sb=1.3,
                 leverage=1.0,vlb=20,label=""):
    years = sorted(set(d.year for d in dates))
    all_t = []
    for ty in range(2019, years[-1]+1):
        ts, te = None, None
        for i,d in enumerate(dates):
            if d.year == ty and ts is None: ts = i
            if d.year == ty: te = i
        if ts is None: continue
        t,_,_ = backtest_v128(C,O,H,L,NS,ND,dates,syms,pred,ker_regime,port_vol,
                              sector_lookup,top_n=top_n,max_per_sector=mps,hold_days=hold_days,
                              vhm=vhm,vlm=vlm,sr=sr,sb=sb,leverage=leverage,vlb=vlb,
                              start_di=ts,end_di=te+1)
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
    print("  V128: 3-Day Prediction Horizon + V103 Gaussian+IRLS")
    print("  Hypothesis: Shorter target aligns better with shorter hold.")
    print("="*70)
    C,O,H,L,V,OI,NS,ND,dates,syms = load_all_data(start="2016-01-01")
    print(f"  {NS} sym, {ND} days, {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    sector_lookup = build_sector_lookup(syms)
    bt_2019 = next(i for i,d in enumerate(dates) if d >= pd.Timestamp("2019-01-01"))
    raw_factors = compute_raw_factors_3d(C,O,H,L,V,OI,NS,ND,tag="V128")
    ker_regime = compute_ker(C,NS,ND)
    ic_array = compute_rolling_ic(raw_factors,NS,ND,ic_window=60,tag="V128")
    bma_weights = compute_bma_weights(ic_array,ND,prior_strength=5.0,tag="V128")
    pred_cache = {}
    for tw in [20,30,40]:
        for bw in [0.5,0.8,1.0]:
            for hc in [2.0,3.0,4.0]:
                print(f"\n--- NW 3d (tw={tw}, bw={bw:.1f}, hc={hc:.1f}) ---")
                pred_cache[(tw,bw,hc)] = compute_nw_gaussian_irls_3d(raw_factors,bma_weights,NS,ND,training_window=tw,kernel_bandwidth=bw,irls_hardy_c=hc)
    vol_cache = {}
    for vlb in [10,15,20]:
        vol_cache[vlb] = compute_portfolio_volatility(C,NS,ND,vlb)
    print("\n" + "="*70)
    print("  PARAMETER SWEEP (2019-2026) — No Leverage")
    print("  top_n x hold_days x tw x bw x hc x vlb x vhm x sr x sb")
    print("="*70)
    results = []
    for pk,pred in pred_cache.items():
        tw,bw,hc = pk
        for top_n in [1,2]:
            for mps in [2,3]:
                for vlb in [10,15,20]:
                    for vhm in [1.5,2.0]:
                        for sr in [0.3,0.5]:
                            for sb in [1.0,1.5]:
                                for hd in [3,5]:
                                    t,eq,dd = backtest_v128(C,O,H,L,NS,ND,dates,syms,pred,ker_regime,vol_cache[vlb],
                                        sector_lookup,top_n=top_n,max_per_sector=mps,hold_days=hd,
                                        vhm=vhm,vlm=0.5,sr=sr,sb=sb,leverage=1.0,vlb=vlb,start_di=bt_2019)
                                    if len(t) < 10: continue
                                    yrs = max(1, sum(x['days'] for x in t)/252)
                                    ann = ((eq/CASH0)**(1/max(1.0,yrs))-1)*100
                                    rets = [x['pnl_pct'] for x in t]
                                    sh = np.mean(rets)/np.std(rets)*np.sqrt(len(rets)/max(1.0,yrs)) if np.std(rets)>1e-12 else 0
                                    ra = ann / max(dd, 1.0)
                                    results.append({"tw":tw,"bw":bw,"hc":hc,"tn":top_n,"mps":mps,"vlb":vlb,"vhm":vhm,"sr":sr,"sb":sb,"hd":hd,"ann":ann,"sh":sh,"dd":dd,"n":len(t),"ra":ra})
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
        t,eq,dd = backtest_v128(C,O,H,L,NS,ND,dates,syms,pred_cache[(best['tw'],best['bw'],best['hc'])],ker_regime,vol_cache[best['vlb']],sector_lookup,
            top_n=best['tn'],max_per_sector=best['mps'],hold_days=best['hd'],
            vhm=best['vhm'],vlm=0.5,sr=best['sr'],sb=best['sb'],leverage=1.0,vlb=best['vlb'],start_di=bt_2019)
        print(f"\n  FULL BACKTEST {label}")
        analyze(t,eq,dd,label)
        print(f"    config: tw={best['tw']} bw={best['bw']:.1f} hc={best['hc']:.1f} tn={best['tn']} mps={best['mps']} vlb={best['vlb']} vhm={best['vhm']:.1f} sr={best['sr']:.1f} sb={best['sb']:.1f} hd={best['hd']}")
        walk_forward(C,O,H,L,NS,ND,dates,syms,pred_cache[(best['tw'],best['bw'],best['hc'])],ker_regime,vol_cache[best['vlb']],sector_lookup,
            top_n=best['tn'],mps=best['mps'],hold_days=best['hd'],
            vhm=best['vhm'],sr=best['sr'],sb=best['sb'],leverage=1.0,vlb=best['vlb'],
            label=f"{label} tw={best['tw']} hd={best['hd']}")
    print(f"\n[V128] Done. {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
