"""V125: "顺势而为" — Term Structure Carry Factor Strategy
============================================================
Uses 82K term structure files as the 8th factor in NW regression.
Carry signal: total_spread_pct (backwardation = bullish, contango = bearish)
Combined with V103 Gaussian+IRLS NW kernel.
Walk-forward 2019-2026. No leverage.
"""
import sys, os, time, warnings, json, glob
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_data import load_all_data
from nw_kernel_utils import (
    CASH0, COMM, LEVERAGE, FACTOR_NAMES, N_FACTORS,
    build_sector_lookup, compute_raw_factors,
    compute_rolling_ic, compute_bma_weights, apply_bma_to_features,
    compute_ker, compute_portfolio_volatility,
    get_vol_multiplier, get_dynamic_mode, compute_atr_at,
)
from alpha_futures_v103 import compute_nw_gaussian_irls

TS_DIR = "/Users/chengming/home/futures_platform/data/futures_term_structure"


def load_carry_factor(syms, dates, NS, ND):
    """Load term structure carry (total_spread_pct) as cross-sectional factor."""
    t0 = time.time()
    print(f"[V125] Loading carry factor from {TS_DIR}...")
    
    # Map: date_str -> {symbol: total_spread_pct}
    date_carry: Dict[str, Dict[str, float]] = defaultdict(dict)
    files = glob.glob(os.path.join(TS_DIR, "*.json"))
    for f in files:
        try:
            with open(f) as fp:
                d = json.load(fp)
            sym = d.get("symbol", "")
            date = d.get("date", "")
            tsp = d.get("total_spread_pct", None)
            if sym and date and tsp is not None:
                date_carry[date][sym] = float(tsp)
        except Exception:
            continue
    
    # Create carry array aligned with main data
    carry = np.full((NS, ND), np.nan)
    sym_to_idx = {s.lower(): i for i, s in enumerate(syms)}
    
    mapped = 0
    for di, d in enumerate(dates):
        date_str = d.strftime("%Y-%m-%d")
        if date_str in date_carry:
            for sym, val in date_carry[date_str].items():
                sym_base = sym.lower().replace("fi", "").replace("2", "").replace("0", "")
                # Try exact match first
                if sym.lower() in sym_to_idx:
                    carry[sym_to_idx[sym.lower()], di] = val
                    mapped += 1
                else:
                    # Try fuzzy match
                    for s, idx in sym_to_idx.items():
                        if s.startswith(sym_base) or sym_base.startswith(s):
                            carry[idx, di] = val
                            mapped += 1
                            break
    
    print(f"  Carry loaded: {len(files)} files, {mapped} mapped values, {time.time()-t0:.1f}s")
    valid_days = sum(1 for di in range(ND) if np.sum(~np.isnan(carry[:,di])) > 0)
    print(f"  Valid carry days: {valid_days}/{ND} ({valid_days/ND*100:.1f}%)")
    return carry


def backtest_v125(C,O,H,L,NS,ND,dates,syms,predicted,ker_regime,port_vol,
                  sector_lookup, top_n=2,mps=2,hold_days=5,win_thresh=0.60,wr_window=15,
                  atr_stop=3.0,vlb=20,vhm=2.0,vlm=0.5,sr=0.5,sb=1.3,start_di=60,end_di=None):
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

        num_total = len(positions) + len(entries)
        alloc_per_pos = LEVERAGE / num_total * vol_mult
        upd_pos = [(si,edi,ep,sp, alloc_per_pos) for si,edi,ep,sp,_ in positions]
        for pv,si,sec in entries:
            ep = O[si,di+1]
            if np.isnan(ep) or ep <= 0: continue
            atr = compute_atr_at(H,L,C,si,di,start_di)
            if atr is None: continue
            upd_pos.append((si,di+1,ep, ep-atr_stop*atr, alloc_per_pos))
        positions = upd_pos

    for si,edi,ep,sp,alloc in positions:
        c = C[si,ND-1]
        if not np.isnan(c) and c > 0:
            equity += equity*alloc*((c-ep)/ep - COMM)
    return trades, equity, max_dd


def analyze_v125(trades, equity, max_dd, label=""):
    if not trades: print(f"  {label}: no trades"); return None
    nw = sum(1 for t in trades if t["pnl_pct"]>0)
    wr = nw/len(trades)*100
    nd = max(1, trades[-1]["di"]-trades[0]["di"])
    ann = ((equity/CASH0)**(1/max(1.0,nd/252))-1)*100
    ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x:x["di"])]
    rets = np.array(ap)/CASH0
    sh = np.mean(rets)/np.std(rets)*np.sqrt(252) if np.std(rets)>0 else 0
    print(f"  {label}: {len(trades)}t WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% Sh={sh:.2f} eq={equity:,.0f}")
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
    print("  V125: \"顺势而为\" — Term Structure Carry Factor")
    print("  8th factor: total_spread_pct from 82K term structure files")
    print("  Combined with V103 Gaussian+IRLS NW kernel")
    print("  Walk-forward 2019-2026. No leverage.")
    print("="*70)

    C,O,H,L,V,OI,NS,ND,dates,syms = load_all_data(start="2016-01-01")
    print(f"  {NS} sym, {ND} days, {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    sector_lookup = build_sector_lookup(syms)
    bt_2019 = next(i for i,d in enumerate(dates) if d>=pd.Timestamp("2019-01-01"))

    # 1. Load carry factor
    carry = load_carry_factor(syms, dates, NS, ND)

    # 2. Raw factors + carry
    raw_factors = compute_raw_factors(C,O,H,L,V,OI,NS,ND,tag="V125")
    raw_factors["carry_pct"] = carry
    ker_regime = compute_ker(C,NS,ND)

    # 3. BMA weights (8 factors now)
    ic_arr = compute_rolling_ic(raw_factors,NS,ND,ic_window=60,tag="V125")
    bma_w = compute_bma_weights(ic_arr,ND,prior_strength=5.0,tag="V125")

    # 4. Gaussian+IRLS predictions
    print("\n--- Computing NW predictions (8 factors) ---")
    pred = compute_nw_gaussian_irls(raw_factors,bma_w,NS,ND,training_window=30,kernel_bandwidth=0.8,irls_hardy_c=3.0)

    # 5. Portfolio vol
    port_vol = compute_portfolio_volatility(C,NS,ND,vol_lookback=10)

    # 6. Parameter sweep
    print("\n"+"="*70)
    print("  V125 PARAMETER SWEEP")
    print("="*70)

    results = []
    sweep_count = 0
    for top_n in [2,3]:
        for mps in [2,3]:
            for vhm in [1.5,2.0]:
                for sr in [0.3,0.5]:
                    for sb in [1.0,1.5]:
                        sweep_count += 1
                        trades,eq,dd = backtest_v125(
                            C,O,H,L,NS,ND,dates,syms,pred,ker_regime,port_vol,
                            sector_lookup, top_n=top_n,mps=mps,hold_days=5,
                            vhm=vhm,vlm=0.5,sr=sr,sb=sb,vlb=20,start_di=bt_2019)
                        if len(trades)<10: continue
                        nw = sum(1 for t in trades if t["pnl_pct"]>0)
                        wr = nw/len(trades)*100
                        nd = max(1,trades[-1]["di"]-trades[0]["di"])
                        ann = ((eq/CASH0)**(1/max(1.0,nd/252))-1)*100
                        ap = [t["pnl_abs"] for t in sorted(trades,key=lambda x:x["di"])]
                        ra = np.array(ap)/CASH0
                        sh = np.mean(ra)/np.std(ra)*np.sqrt(252) if np.std(ra)>0 else 0
                        results.append({"tn":top_n,"mps":mps,"vhm":vhm,"sr":sr,"sb":sb,
                            "n":len(trades),"wr":wr,"ann":ann,"dd":dd,"sh":sh,"eq":eq})

    print(f"\n  Evaluated {sweep_count} configs, {len(results)} with 10+ trades")
    results.sort(key=lambda x:-x["ann"])
    print(f"\n{'TN':>3} {'MPS':>3} {'Vhm':>4} {'SR':>4} {'SB':>4} {'N':>5} {'WR':>6} {'Ann':>8} {'DD':>7} {'Sh':>6}")
    print("-"*70)
    for r in results[:15]:
        print(f"{r['tn']:>3} {r['mps']:>3} {r['vhm']:>4.1f} {r['sr']:>4.1f} {r['sb']:>4.1f} {r['n']:>5} {r['wr']:>5.1f}% {r['ann']:>+7.1f}% {r['dd']:>6.1f}% {r['sh']:>5.2f}")

    if not results: print("No results."); return

    for label,best in [("BEST-ANN",results[0]),("BEST-SHARPE",max(results,key=lambda x:x["sh"])),("BEST-RISK-ADJ",max(results,key=lambda x:x["ann"]/max(x["dd"],1.0)))]:
        print(f"\n{'='*70}\n  FULL BACKTEST {label}\n{'='*70}")
        trades,eq,dd = backtest_v125(
            C,O,H,L,NS,ND,dates,syms,pred,ker_regime,port_vol,
            sector_lookup, top_n=best["tn"],mps=best["mps"],vhm=best["vhm"],
            sr=best["sr"],sb=best["sb"],start_di=bt_2019)
        analyze_v125(trades,eq,dd,label)

    print(f"\n[V125] Done. {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
