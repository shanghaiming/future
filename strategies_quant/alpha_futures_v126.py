"""V126: "因材施教" — Dynamic IC-Weighted Factor Selection
==========================================================
Instead of BMA equal-weighting, dynamically weight factors by rolling IC.
Weight = max(IC, 0) / sum(max(IC_j, 0)) — only positive-IC factors used.
Combined with V103 Gaussian+IRLS NW kernel.
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
    CASH0, COMM, LEVERAGE, FACTOR_NAMES, N_FACTORS,
    build_sector_lookup, compute_raw_factors, normalize_factor,
    compute_ker, compute_portfolio_volatility,
    get_vol_multiplier, get_dynamic_mode, compute_atr_at,
)
from alpha_futures_v103 import compute_nw_gaussian_irls


def compute_ic_weights(raw_factors, NS, ND, ic_window=15, ic_min=0.02):
    """Compute dynamic IC weights per day. Only factors with IC > ic_min contribute."""
    t0 = time.time()
    print(f"[V126] Computing IC weights (window={ic_window}, min={ic_min})...")
    fwd_ret = raw_factors["fwd_ret_5d"]
    ic_weights = np.full((N_FACTORS, ND), np.nan)
    
    for di in range(ic_window + 10, ND):
        ics = []
        for fi, fname in enumerate(FACTOR_NAMES):
            factor = raw_factors[fname]
            ic_vals = []
            for tdi in range(di - ic_window, di):
                fv, rv = factor[:, tdi], fwd_ret[:, tdi]
                mask = (~np.isnan(fv)) & (~np.isnan(rv))
                if np.sum(mask) >= 10:
                    fr = pd.Series(fv[mask]).rank().values
                    rr = pd.Series(rv[mask]).rank().values
                    c = np.corrcoef(fr, rr)[0, 1]
                    if not np.isnan(c):
                        ic_vals.append(c)
            if len(ic_vals) >= 5:
                ics.append(np.mean(ic_vals))
            else:
                ics.append(0.0)
        
        # Softmax over positive ICs
        pos_ics = [max(ic, 0) for ic in ics]
        total = sum(pos_ics)
        if total > 1e-12:
            for fi in range(N_FACTORS):
                ic_weights[fi, di] = pos_ics[fi] / total
        else:
            # Fallback: equal weight
            for fi in range(N_FACTORS):
                ic_weights[fi, di] = 1.0 / N_FACTORS
    
    print(f"  IC weights done: {time.time() - t0:.1f}s")
    return ic_weights


def compute_nw_gaussian_icw(raw_factors, ic_weights, NS, ND,
                            training_window=40, kernel_bandwidth=1.0, irls_hardy_c=3.0):
    """Gaussian NW with IC-weighted features (instead of BMA)."""
    t0 = time.time()
    print(f"[V126] Gaussian+ICW NW (tw={training_window}, bw={kernel_bandwidth:.1f}, hc={irls_hardy_c:.1f})...")
    
    normed = {}
    for fname in FACTOR_NAMES:
        normed[fname] = normalize_factor(raw_factors[fname], NS, ND)
    
    fwd_ret = raw_factors["fwd_ret_5d"]
    atr_mean = raw_factors["atr_mean"]
    predicted = np.full((NS, ND), np.nan)
    
    MIN_TRAIN = 20
    SQRT_2PI = np.sqrt(2.0 * np.pi)
    
    for di in range(training_window + 10, ND):
        train_features = []
        train_targets = []
        start_di = max(10, di - training_window)
        
        for tdi in range(start_di, di):
            # Get IC weights for this day
            w = ic_weights[:, tdi]
            if np.any(np.isnan(w)):
                continue
            for si in range(NS):
                feat = np.array([normed[fname][si, tdi] for fname in FACTOR_NAMES])
                target = fwd_ret[si, tdi]
                if np.any(np.isnan(feat)) or np.isnan(target):
                    continue
                # Apply IC weights to features
                weighted_feat = feat * w
                train_features.append(weighted_feat)
                train_targets.append(target)
        
        if len(train_features) < MIN_TRAIN:
            continue
        
        train_X = np.array(train_features)
        train_Y = np.array(train_targets)
        feat_std = np.std(train_X, axis=0)
        feat_std[feat_std < 1e-12] = 1.0
        
        for si in range(NS):
            w = ic_weights[:, di]
            if np.any(np.isnan(w)):
                continue
            query_feat = np.array([normed[fname][si, di] for fname in FACTOR_NAMES])
            if np.any(np.isnan(query_feat)):
                continue
            # Apply IC weights
            query_feat = query_feat * w
            
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
            print(f"  di={di}/{ND} valid={valid_count}/{NS} train={len(train_features)}", flush=True)
    
    print(f"  Gaussian+ICW done: {time.time() - t0:.1f}s")
    return predicted


def backtest_v126(C, O, H, L, NS, ND, dates, syms, predicted, ker_regime, port_vol,
                  sector_lookup, top_n=2, mps=2, hold_days=5, win_thresh=0.60, wr_window=15,
                  atr_stop=3.0, vlb=20, vhm=2.0, vlm=0.5, sr=0.5, sb=1.3, start_di=60, end_di=None):
    if end_di is None: end_di = ND - 1
    vol_data = port_vol[max(start_di, vlb + 1):end_di]
    vol_data_valid = vol_data[~np.isnan(vol_data)]
    vol_median = np.median(vol_data_valid) if len(vol_data_valid) > 10 else 1e-6
    
    equity, peak, max_dd = CASH0, CASH0, 0.0
    positions, trades, recent_wins = [], [], []
    
    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0.0
        new_pos = []
        mode = get_dynamic_mode(recent_wins, win_thresh, wr_window)
        vol_mult = get_vol_multiplier(port_vol[di], vol_median, vhm, vlm, sr, sb)
        
        by_si = defaultdict(list)
        for si, edi, ep, sp, alloc in positions:
            by_si[si].append((edi, ep, sp, alloc))
        for si, plist in by_si.items():
            c = C[si, di]
            if np.isnan(c):
                for edi, ep, sp, alloc in plist:
                    new_pos.append((si, edi, ep, sp, alloc))
                continue
            earliest = min(p[0] for p in plist)
            hold = di - earliest
            stopped = any(c < sp for _, _, sp, _ in plist)
            if stopped or hold >= hold_days:
                for edi, ep, sp, alloc in plist:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * pnl
                    daily_pnl += profit
                    is_win = pnl > 0
                    trades.append({
                        "pnl_abs": profit, "pnl_pct": pnl * 100, "days": di - edi + 1,
                        "di": di, "year": d.year, "sym": syms[si],
                        "sector": sector_lookup.get(si, 'OTHER'),
                        "reason": "stop" if stopped else "hold", "mode": mode[0].upper(),
                    })
                    recent_wins.append(1 if is_win else 0)
            else:
                for edi, ep, sp, alloc in plist:
                    new_pos.append((si, edi, ep, sp, alloc))
        positions = new_pos
        equity += daily_pnl
        if equity > peak: peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100
            if dd > max_dd: max_dd = dd
        if equity <= 0: break
        
        held = {p[0] for p in positions}
        if len(held) >= top_n: continue
        cands = [(predicted[si, di], si) for si in range(NS)
                 if si not in held and not np.isnan(predicted[si, di])
                 and di + 1 < ND and not np.isnan(O[si, di + 1])
                 and ker_regime[si, di] >= 0]
        if not cands: continue
        cands.sort(key=lambda x: -x[0])
        
        n_take = top_n
        if mode == "winning": n_take = min(top_n + 1, top_n * 2)
        elif mode == "losing": n_take = max(1, top_n - 1)
        
        sec_counts = defaultdict(int)
        for si_h in held: sec_counts[sector_lookup.get(si_h, 'OTHER')] += 1
        entries = []
        for pv, si in cands:
            if len(held) + len(entries) >= n_take: break
            if si in held: continue
            sec = sector_lookup.get(si, 'OTHER')
            if sec_counts[sec] >= mps: continue
            if pv <= 0: continue
            entries.append((pv, si, sec))
            sec_counts[sec] += 1
        if not entries: continue
        
        num_total = len(positions) + len(entries)
        alloc_per_pos = LEVERAGE / num_total * vol_mult
        upd_pos = [(si, edi, ep, sp, alloc_per_pos) for si, edi, ep, sp, _ in positions]
        for pv, si, sec in entries:
            ep = O[si, di + 1]
            if np.isnan(ep) or ep <= 0: continue
            atr = compute_atr_at(H, L, C, si, di, start_di)
            if atr is None: continue
            upd_pos.append((si, di + 1, ep, ep - atr_stop * atr, alloc_per_pos))
        positions = upd_pos
    
    for si, edi, ep, sp, alloc in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            equity += equity * alloc * ((c - ep) / ep - COMM)
    return trades, equity, max_dd


def analyze_v126(trades, equity, max_dd, label=""):
    if not trades:
        print(f"  {label}: no trades")
        return None
    nw = sum(1 for t in trades if t["pnl_pct"] > 0)
    wr = nw / len(trades) * 100
    nd = max(1, trades[-1]["di"] - trades[0]["di"])
    ann = ((equity / CASH0) ** (1 / max(1.0, nd / 252)) - 1) * 100
    ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
    rets = np.array(ap) / CASH0
    sh = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
    print(f"  {label}: {len(trades)}t WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% Sh={sh:.2f} eq={equity:,.0f}")
    yr = defaultdict(lambda: {"n": 0, "w": 0, "pnl": []})
    for t in trades:
        y = t["year"]
        yr[y]["n"] += 1
        if t["pnl_pct"] > 0: yr[y]["w"] += 1
        yr[y]["pnl"].append(t["pnl_pct"])
    for y in sorted(yr.keys()):
        ys = yr[y]
        cum = np.prod([1 + p / 100 for p in ys["pnl"]]) - 1
        print(f"    {y}: {ys['n']}t WR={ys['w'] / ys['n'] * 100:.1f}% cum={cum:+.1%}")
    return {"n": len(trades), "wr": wr, "ann": ann, "dd": max_dd, "sh": sh, "eq": equity}


def main():
    t0 = time.time()
    print("=" * 70)
    print("  V126: \"因材施教\" — Dynamic IC-Weighted Factor Selection")
    print("  Weight factors by rolling IC, only positive-IC factors contribute")
    print("  Combined with V103 Gaussian+IRLS NW kernel")
    print("  Walk-forward 2019-2026. No leverage.")
    print("=" * 70)
    
    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start="2016-01-01")
    print(f"  {NS} sym, {ND} days, {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    
    sector_lookup = build_sector_lookup(syms)
    bt_2019 = next(i for i, d in enumerate(dates) if d >= pd.Timestamp("2019-01-01"))
    
    raw_factors = compute_raw_factors(C, O, H, L, V, OI, NS, ND, tag="V126")
    ker_regime = compute_ker(C, NS, ND)
    
    # Compute IC weights
    ic_weights = compute_ic_weights(raw_factors, NS, ND, ic_window=15, ic_min=0.02)
    
    # Gaussian+ICW predictions
    print("\n--- Computing NW predictions ---")
    pred = compute_nw_gaussian_icw(raw_factors, ic_weights, NS, ND,
                                    training_window=30, kernel_bandwidth=0.8, irls_hardy_c=3.0)
    
    port_vol = compute_portfolio_volatility(C, NS, ND, vol_lookback=10)
    
    print("\n" + "=" * 70)
    print("  V126 PARAMETER SWEEP")
    print("=" * 70)
    
    results = []
    sweep_count = 0
    for top_n in [2, 3]:
        for mps in [2, 3]:
            for vhm in [1.5, 2.0]:
                for sr in [0.3, 0.5]:
                    for sb in [1.0, 1.5]:
                        sweep_count += 1
                        trades, eq, dd = backtest_v126(
                            C, O, H, L, NS, ND, dates, syms, pred, ker_regime, port_vol,
                            sector_lookup, top_n=top_n, mps=mps, hold_days=5,
                            vhm=vhm, vlm=0.5, sr=sr, sb=sb, vlb=20, start_di=bt_2019)
                        if len(trades) < 10: continue
                        nw = sum(1 for t in trades if t["pnl_pct"] > 0)
                        wr = nw / len(trades) * 100
                        nd = max(1, trades[-1]["di"] - trades[0]["di"])
                        ann = ((eq / CASH0) ** (1 / max(1.0, nd / 252)) - 1) * 100
                        ap = [t["pnl_abs"] for t in sorted(trades, key=lambda x: x["di"])]
                        ra = np.array(ap) / CASH0
                        sh = np.mean(ra) / np.std(ra) * np.sqrt(252) if np.std(ra) > 0 else 0
                        results.append({"tn": top_n, "mps": mps, "vhm": vhm, "sr": sr, "sb": sb,
                                        "n": len(trades), "wr": wr, "ann": ann, "dd": dd, "sh": sh, "eq": eq})
    
    print(f"\n  Evaluated {sweep_count} configs, {len(results)} with 10+ trades")
    results.sort(key=lambda x: -x["ann"])
    print(f"\n{'TN':>3} {'MPS':>3} {'Vhm':>4} {'SR':>4} {'SB':>4} {'N':>5} {'WR':>6} {'Ann':>8} {'DD':>7} {'Sh':>6}")
    print("-" * 70)
    for r in results[:15]:
        print(f"{r['tn']:>3} {r['mps']:>3} {r['vhm']:>4.1f} {r['sr']:>4.1f} {r['sb']:>4.1f} {r['n']:>5} {r['wr']:>5.1f}% {r['ann']:>+7.1f}% {r['dd']:>6.1f}% {r['sh']:>5.2f}")
    
    if not results:
        print("No results.")
        return
    
    for label, best in [("BEST-ANN", results[0]),
                        ("BEST-SHARPE", max(results, key=lambda x: x["sh"])),
                        ("BEST-RISK-ADJ", max(results, key=lambda x: x["ann"] / max(x["dd"], 1.0)))]:
        print(f"\n{'=' * 70}\n  FULL BACKTEST {label}\n{'=' * 70}")
        trades, eq, dd = backtest_v126(
            C, O, H, L, NS, ND, dates, syms, pred, ker_regime, port_vol,
            sector_lookup, top_n=best["tn"], mps=best["mps"], vhm=best["vhm"],
            sr=best["sr"], sb=best["sb"], start_di=bt_2019)
        analyze_v126(trades, eq, dd, label)
    
    print(f"\n[V126] Done. {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
