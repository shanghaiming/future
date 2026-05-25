"""
Alpha Futures V174 — Profit-Taking and Target-Based Exits
==============================================================================
V169 champion: hold=1 day, no stop-loss, atr_norm<12% giving +253%/-15% WF (R/M=16.91).
Previous tests showed stop-loss hurts returns, but TARGET-BASED exits haven't been tested.

This version tests PROFIT-TAKING (capturing gains early) vs always holding 1 day.

Exit modes tested:
  A. hold1_baseline  — standard hold=1 day, exit at C[di+1] (V169 baseline)
  B. extended_target — hold up to 3 days, exit early if profit_target hit or stop_loss hit
  C. intraday_high   — simulate intraday exit using H[di+1] as proxy for intraday high
  D. gap_exit        — exit at O[di+2] if gap-up target met, else C[di+2]

Parameter sweep:
  exit_mode:    ['hold1_baseline', 'extended_target', 'intraday_high', 'gap_exit']
  profit_target:[1.5, 2.0, 3.0, 5.0] (percent)
  stop_loss:    [0, -1.0, -2.0] (percent, 0=no stop) — only for extended_target
  atr_norm_max: [10, 12]
  max_corr:     [0.5, 0.7]

Base: V169 vol filter + Kitchen Sink sizing, aggro100 DD, regime 0.5-1.5.
Walk-forward validation for top 20 configs.
"""
import sys, os, time, warnings
import numpy as np
import talib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

MULT = {'agfi': 15, 'alfi': 5, 'aufi': 1000, 'bufi': 10, 'cufi': 5, 'fufi': 10,
        'rbfi': 10, 'znfi': 5, 'nifi': 1, 'hcfi': 10, 'spfi': 10, 'ssfi': 5,
        'sffi': 5, 'smfi': 5, 'pbfi': 5, 'snfi': 1, 'rufi': 10, 'wrffi': 10,
        'afi': 10, 'bfi': 10, 'bbfi': 500, 'cffi': 5, 'cfi': 10, 'csfi': 10,
        'ebfi': 5, 'egfi': 10, 'fbfi': 500, 'ifi': 100, 'jfi': 100, 'jmfi': 60,
        'lfi': 5, 'mfi': 10, 'pgfi': 20, 'ppfi': 5, 'vfi': 5, 'yfi': 10,
        'pfi': 10, 'jdfi': 5, 'lhfi': 16, 'pkfi': 5, 'rrfi': 20, 'lrfi': 20,
        'jrfi': 20, 'pmfi': 20, 'whfi': 20, 'rsfi': 20, 'cjfi': 10, 'mafi': 10,
        'apfi': 10, 'cyfi': 5, 'fgfi': 20, 'oifi': 10, 'pfifi': 5, 'rmfi': 10,
        'srfi': 10, 'tafi': 5, 'safi': 20, 'urfi': 20, 'scfi': 1000, 'lufi': 10,
        'bcfi': 5, 'nrfi': 1, 'lgfi': 20, 'brfi': 5, 'lcfi': 1, 'sifi': 5,
        'ni': 1, 'tai': 5}
DEF_MULT = 10
COMM = 0.0003


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0: return -100.0
    return (final / initial) ** (1.0 / (n_days / 252)) * 100 - 100


def main():
    print("=" * 130)
    print("  V174 — Profit-Taking and Target-Based Exits")
    print("  Testing if early profit capture beats hold-1-day baseline")
    print("=" * 130)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    print(f"  {NS} commodities, {ND} days")

    # ===================== PRECOMPUTE =====================
    print("\n[Precompute]...", flush=True)
    t0 = time.time()

    RET = np.full((NS, ND), np.nan)
    ROC5 = np.full((NS, ND), np.nan)
    ROC10 = np.full((NS, ND), np.nan)
    ROC20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100
        ROC5[si] = talib.ROC(c, timeperiod=5)
        ROC10[si] = talib.ROC(c, timeperiod=10)
        ROC20[si] = talib.ROC(c, timeperiod=20)

    ATR14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        ATR14[si] = talib.ATR(H[si].astype(np.float64), L[si].astype(np.float64),
                               C[si].astype(np.float64), timeperiod=14)

    ATR_NORM = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            atr = ATR14[si, di]
            cp = C[si, di]
            if not np.isnan(atr) and not np.isnan(cp) and cp > 0:
                ATR_NORM[si, di] = atr / cp * 100

    ZSCORE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = RET[si, di-20:di]
            v = rets[~np.isnan(rets)]
            if len(v) < 10: continue
            s = np.std(v, ddof=1)
            if s > 0 and not np.isnan(RET[si, di]):
                ZSCORE[si, di] = (RET[si, di] - np.mean(v)) / s

    OV_GAP = np.full((NS, ND), np.nan)
    ID_RET = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            o, c = O[si, di], C[si, di]
            if not np.isnan(o) and not np.isnan(c):
                if di > 0 and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                    OV_GAP[si, di] = (o - C[si, di-1]) / C[si, di-1] * 100
                if o > 0: ID_RET[si, di] = (c - o) / o * 100

    # ===================== REGIME INDICATORS =====================
    print("  Computing regime indicators...", flush=True)

    BREADTH = np.full(ND, np.nan)
    for di in range(5, ND):
        pos_count = 0; total = 0
        for si in range(NS):
            r = ROC5[si, di]
            if not np.isnan(r):
                total += 1
                if r > 0: pos_count += 1
        if total > 0:
            BREADTH[di] = pos_count / total

    MKT_RET = np.full(ND, np.nan)
    for di in range(ND):
        rets_day = RET[:, di]
        valid = rets_day[~np.isnan(rets_day)]
        if len(valid) > 10:
            MKT_RET[di] = np.mean(valid)

    MKT_VOL = np.full(ND, np.nan)
    for di in range(20, ND):
        window = MKT_RET[di-20:di]
        valid = window[~np.isnan(window)]
        if len(valid) >= 10:
            MKT_VOL[di] = np.std(valid, ddof=1)

    valid_vols = MKT_VOL[~np.isnan(MKT_VOL)]
    VOL_MEDIAN = np.median(valid_vols) if len(valid_vols) > 0 else 1.0

    print(f"  Market vol median={VOL_MEDIAN:.4f}%")
    print(f"  Done ({time.time()-t0:.1f}s)")

    # ===================== SIGNAL DEFINITIONS =====================
    def sig_v121(di, edi):
        c = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((roc * zs, s, ep, 'v121'))
        return c

    def sig_ov_id(di, edi):
        c = []
        for s in range(NS):
            ov = OV_GAP[s, di]; idr = ID_RET[s, di]; roc = ROC5[s, di]
            if any(np.isnan(x) for x in [ov, idr, roc]): continue
            if ov <= 0.3 or idr <= 0.3 or roc <= 1.0: continue
            zs = ZSCORE[s, di]
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            z_bonus = zs if not np.isnan(zs) and zs > 1.0 else 1.0
            c.append(((ov + idr) * roc * z_bonus * 2, s, ep, 'ov_id'))
        return c

    def sig_final_flag(di, edi):
        c = []
        for s in range(NS):
            roc20 = ROC20[s, di]
            if np.isnan(roc20) or roc20 <= 5.0 or di < 6: continue
            h5 = H[s, di-4:di+1]; l5 = L[s, di-4:di+1]
            if any(np.isnan(x) for x in h5) or any(np.isnan(x) for x in l5): continue
            r5 = np.max(h5) - np.min(l5)
            atr = ATR14[s, di]
            if np.isnan(atr) or atr <= 0 or r5 > atr * 3.0: continue
            h4 = np.max(H[s, di-4:di])
            cp = C[s, di]
            if np.isnan(cp) or cp <= h4: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((roc20 * (cp - h4) / atr, s, ep, 'ff'))
        return c

    def sig_union(di, edi):
        all_sigs = {}
        for item in sig_v121(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc * 3
            all_sigs[s][2].append('v121')
        for item in sig_ov_id(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc * 2
            all_sigs[s][2].append('ov_id')
        for item in sig_final_flag(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc
            all_sigs[s][2].append('ff')
        return [(sc, s, ep, '+'.join(sigs)) for s, (sc, ep, sigs) in all_sigs.items()]

    # ===================== HELPERS =====================
    def get_corr(si_a, si_b, di, window=20):
        start_idx = max(0, di - window)
        ret_a = RET[si_a, start_idx:di]
        ret_b = RET[si_b, start_idx:di]
        valid = ~(np.isnan(ret_a) | np.isnan(ret_b))
        n_valid = np.sum(valid)
        if n_valid < 8:
            return 0.5
        ra = ret_a[valid]; rb = ret_b[valid]
        if np.std(ra) == 0 or np.std(rb) == 0:
            return 0.5
        c = np.corrcoef(ra, rb)[0, 1]
        return c if not np.isnan(c) else 0.5

    def compute_composite(di, daily_eq, high_water, perf_window=20):
        scores = []
        bth = BREADTH[di]
        if not np.isnan(bth):
            scores.append(np.clip((bth - 0.4) / (0.7 - 0.4), 0, 1))
        vol = MKT_VOL[di]
        if not np.isnan(vol) and VOL_MEDIAN > 0:
            vol_ratio = vol / VOL_MEDIAN
            scores.append(np.clip((1.5 - vol_ratio) / (1.5 - 0.8), 0, 1))
        if len(daily_eq) >= perf_window:
            eq_window = np.array(daily_eq[-perf_window:])
            x = np.arange(perf_window)
            try:
                slope = np.polyfit(x, eq_window, 1)[0]
                eq_mean = np.mean(eq_window)
                norm_slope = slope / eq_mean * 100 if eq_mean > 0 else 0
                eq_rets = np.diff(eq_window) / eq_window[:-1] * 100
                eq_rets = eq_rets[np.isfinite(eq_rets)]
                eq_std = np.std(eq_rets) if len(eq_rets) > 5 else 1.0
                z = norm_slope / eq_std if eq_std > 0 else 0
                scores.append(np.clip((z + 1.0) / 2.0, 0, 1))
            except Exception:
                pass
        if high_water > 0:
            cur_dd = (daily_eq[-1] - high_water) / high_water
        else:
            cur_dd = 0
        scores.append(np.clip(1.0 + cur_dd / 0.3, 0, 1))
        return np.mean(scores) if scores else 0.5

    def dd_size(pv, high_water, tiers):
        if high_water <= 0:
            return tiers[0][1]
        dd = (pv - high_water) / high_water
        for dd_thresh, size_frac in tiers:
            if dd >= -dd_thresh:
                return size_frac
        return tiers[-1][1]

    # ===================== BACKTEST ENGINE =====================
    # exit_mode controls how positions are exited:
    #   'hold1_baseline'  — hold exactly 1 day, exit at C[entry_di+1] (V169 baseline)
    #   'extended_target'  — hold up to max_hold days, exit early if profit_target or stop_loss hit
    #   'intraday_high'    — on entry day, if H[entry_di] > entry*(1+target), assume exit at target
    #   'gap_exit'         — check O[entry_di+1] for gap-up exit, else exit at C[entry_di+1]
    #
    # profit_target: percent gain threshold for early exit (e.g., 2.0 = 2%)
    # stop_loss_pct: percent loss threshold for early exit (e.g., -1.0 = -1%, 0=no stop)
    # max_hold: maximum holding period in days (for extended_target)

    def backtest(start_di=MIN_TRAIN, end_di=None,
                 atr_norm_max=12.0, max_corr=0.7,
                 dd_tiers=None,
                 regime_lo=0.5, regime_hi=1.5,
                 exit_mode='hold1_baseline',
                 profit_target=2.0,
                 stop_loss_pct=0.0,
                 max_hold=3,
                 top_n=3):
        if end_di is None: end_di = ND
        if dd_tiers is None:
            dd_tiers = [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)]

        cash = float(CASH0)
        positions = []
        trades = []
        daily_eq = []
        high_water = float(CASH0)

        for di in range(start_di, end_di - 1):
            # Mark-to-market
            pv = cash
            for p in positions:
                cp = C[p['si'], di]
                if not np.isnan(cp) and cp > 0:
                    m = MULT.get(p['sym'], DEF_MULT)
                    pv += cp * m * p['lots'] - cp * m * abs(p['lots']) * COMM
            daily_eq.append(pv)
            if pv > high_water:
                high_water = pv

            # --- Exit logic ---
            to_close = []
            for p in positions:
                entry_di = p['entry_di']
                entry_price = p['entry_price']
                days_held = di - entry_di

                if exit_mode == 'hold1_baseline':
                    # Standard: hold 1 day, exit at C[di] on the day after entry
                    # entry happens at O[edi] where edi = signal_di + 1
                    # So entry_di = signal_di + 1, and we close at C[entry_di + 1] which is di when days_held=1
                    # Actually: entry_di stores the di of entry. We need to check if days_held >= 1
                    if days_held >= 1:
                        exit_price = C[p['si'], di]
                        if np.isnan(exit_price) or exit_price <= 0:
                            exit_price = entry_price
                        to_close.append((p, exit_price))

                elif exit_mode == 'extended_target':
                    cp_today = C[p['si'], di]
                    hp_today = H[p['si'], di]
                    lp_today = L[p['si'], di]
                    if np.isnan(cp_today) or cp_today <= 0:
                        # Can't evaluate, skip unless max hold reached
                        if days_held >= max_hold:
                            to_close.append((p, entry_price))
                        continue

                    unrealized_pct = (cp_today - entry_price) / entry_price * 100

                    # Check profit target using high of day (best intraday price)
                    target_hit = False
                    stop_hit = False

                    # Profit target: check if intraday high reached target
                    if profit_target > 0 and not np.isnan(hp_today) and hp_today > 0:
                        high_pct = (hp_today - entry_price) / entry_price * 100
                        if high_pct >= profit_target:
                            target_hit = True

                    # Stop-loss: check if intraday low breached stop
                    if stop_loss_pct < 0 and not np.isnan(lp_today) and lp_today > 0:
                        low_pct = (lp_today - entry_price) / entry_price * 100
                        if low_pct <= stop_loss_pct:
                            stop_hit = True

                    if target_hit or stop_hit:
                        # Exit at close (conservative: use actual close price)
                        exit_price = cp_today
                        to_close.append((p, exit_price))
                    elif days_held >= max_hold:
                        # Time-based exit
                        exit_price = cp_today
                        if np.isnan(exit_price) or exit_price <= 0:
                            exit_price = entry_price
                        to_close.append((p, exit_price))

                elif exit_mode == 'intraday_high':
                    if days_held >= 1:
                        # On the exit day, check if intraday high hit target
                        hp = H[p['si'], di]
                        cp_today = C[p['si'], di]

                        if not np.isnan(hp) and hp > entry_price * (1 + profit_target / 100):
                            # Intraday target was reached — assume exit at target price
                            exit_price = entry_price * (1 + profit_target / 100)
                        else:
                            # Target not reached, exit at close
                            exit_price = cp_today
                            if np.isnan(exit_price) or exit_price <= 0:
                                exit_price = entry_price
                        to_close.append((p, exit_price))

                elif exit_mode == 'gap_exit':
                    if days_held >= 1:
                        # Check if next-day open (which is today's di from position perspective)
                        # gap_exit: enter at O[di+1], check O[di+2] for gap-up
                        # entry_di = di+1 (signal day +1). On the NEXT day (entry_di+1 = di+2),
                        # we check O[entry_di+1] for gap-up
                        # days_held >=1 means we are at entry_di+1 or later
                        op_today = O[p['si'], di]
                        cp_today = C[p['si'], di]

                        if days_held == 1 and not np.isnan(op_today) and op_today > 0:
                            gap_pct = (op_today - entry_price) / entry_price * 100
                            if gap_pct >= profit_target:
                                # Gap-up exit at open
                                exit_price = op_today
                                to_close.append((p, exit_price))
                                continue

                        # No gap-up or past gap-up window, exit at close
                        exit_price = cp_today
                        if np.isnan(exit_price) or exit_price <= 0:
                            exit_price = entry_price
                        to_close.append((p, exit_price))

            # Execute closes
            for p, exit_price in to_close:
                m = MULT.get(p['sym'], DEF_MULT)
                pnl = (exit_price - p['entry_price']) * m * p['lots']
                inv = p['entry_price'] * m * abs(p['lots'])
                pp = pnl / inv * 100 if inv > 0 else 0
                cash += exit_price * m * abs(p['lots']) * (1 - COMM)
                trades.append(pp)
                positions.remove(p)

            # --- Kitchen Sink sizing ---
            dd_sz = dd_size(pv, high_water, dd_tiers)
            composite = compute_composite(di, daily_eq, high_water)
            regime_mult = regime_lo + composite * (regime_hi - regime_lo)
            pos_size = dd_sz * regime_mult
            pos_size = max(0.05, min(0.99, pos_size))

            # --- Enter positions ---
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue

            held_si = set(p['si'] for p in positions)

            cands_v121 = sig_v121(di, edi)
            cands_union = sig_union(di, edi)

            # Apply vol filter (fixed mode)
            cands_v121_f = [c for c in cands_v121
                            if not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < atr_norm_max]
            cands_union_f = [c for c in cands_union
                             if not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < atr_norm_max]

            cands_v121_f.sort(key=lambda x: -x[0])
            cands_union_f.sort(key=lambda x: -x[0])

            best_v121 = None
            for c in cands_v121_f:
                if c[1] not in held_si:
                    best_v121 = c
                    break

            best_union = None
            for c in cands_union_f:
                if c[1] not in held_si:
                    best_union = c
                    break

            entries = []
            if best_v121 and best_union:
                if best_v121[1] == best_union[1]:
                    entries.append((best_v121[0], best_v121[1], best_v121[2],
                                    'v121+union', pos_size * 1.5))
                else:
                    corr = get_corr(best_v121[1], best_union[1], di)
                    if corr < max_corr:
                        entries.append((best_v121[0], best_v121[1], best_v121[2],
                                        'v121', pos_size))
                        entries.append((best_union[0], best_union[1], best_union[2],
                                        'union', pos_size))
                    else:
                        best = best_v121 if best_v121[0] >= best_union[0] else best_union
                        entries.append((best[0], best[1], best[2], 'best', pos_size))
            elif best_v121:
                entries.append((best_v121[0], best_v121[1], best_v121[2], 'v121', pos_size))
            elif best_union:
                entries.append((best_union[0], best_union[1], best_union[2], 'union', pos_size))

            cash_snapshot = cash
            n_planned = len(entries)
            for sc, s, pr, sig_str, pct in entries:
                if s in set(p['si'] for p in positions): continue
                if len(positions) >= top_n: break
                cap = cash_snapshot * pct / n_planned
                sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                ct = max(1, int(cap / (pr * m * (1 + COMM))))
                ci = pr * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                    ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                if ct <= 0 or ci <= 0 or ci > cash: continue
                cash -= ci
                positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                  'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': 1,
                                  'sig': sig_str, 'score': sc})

        # Close remaining
        for p in positions:
            ae = end_di - 1
            ep = C[p['si'], min(ae, ND-1)]
            if np.isnan(ep) or ep <= 0: ep = p['entry_price']
            m = MULT.get(p['sym'], DEF_MULT)
            cash += ep * m * abs(p['lots']) * (1 - COMM)

        nd = end_di - start_di
        ann = annual_return(cash, CASH0, nd)
        wr = np.mean([1 if t > 0 else 0 for t in trades]) * 100 if trades else 0
        nt = len(trades)
        if daily_eq:
            eq = np.array(daily_eq); pk = np.maximum.accumulate(eq)
            mdd = np.min((eq - pk) / pk * 100)
            r = np.diff(eq) / eq[:-1]
            r = np.where(np.isfinite(r), r, 0)
            sh = np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0
        else:
            mdd = 0; sh = 0
        return {'ann': ann, 'wr': wr, 'n': nt, 'mdd': mdd, 'sharpe': sh, 'final': cash}

    # ===================== PRINTING HELPERS =====================
    def pr(r, label=""):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  {label:100s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d}")

    def walk_forward(label="", **kwargs):
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue
            r = backtest(start_di=ys, end_di=ye, **kwargs)
            res[yr] = r
        return res

    def print_wf(wf_res, label=""):
        pos = sum(1 for r in wf_res.values() if r['ann'] > 0)
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"    {label}")
        print(f"      {pos}/6 pos | Avg={avg_ann:>+7.0f}% | WorstWfMDD={worst_mdd:>5.0f}%")
        print(f"      {ws}")

    # ===================== CONFIG =====================
    DD_AGGR100 = [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)]

    all_results = []

    # ===================== SECTION 0: BASELINE (V169 hold=1, no exit logic) =====================
    print("\n" + "=" * 130)
    print("  SECTION 0: BASELINE — V169 hold=1 day, no target exits")
    print("=" * 130)

    for atr_max in [10.0, 12.0]:
        for mc in [0.5, 0.7]:
            label = f"BASELINE hold1 atr<{atr_max:.0f}% corr={mc:.1f}"
            r = backtest(atr_norm_max=atr_max, max_corr=mc,
                         dd_tiers=DD_AGGR100,
                         regime_lo=0.5, regime_hi=1.5,
                         exit_mode='hold1_baseline',
                         profit_target=0, stop_loss_pct=0, max_hold=1,
                         top_n=3)
            pr(r, label)
            all_results.append({**r, 'label': f'base_atr{atr_max:.0f}_c{mc:.1f}',
                                'section': 0, 'exit_mode': 'hold1_baseline',
                                'atr_norm_max': atr_max, 'max_corr': mc,
                                'profit_target': 0, 'stop_loss_pct': 0,
                                'max_hold': 1, 'top_n': 3})

    # ===================== SECTION 1: EXTENDED HOLD WITH PROFIT TARGET =====================
    print("\n" + "=" * 130)
    print("  SECTION 1: EXTENDED HOLD (up to 3 days) WITH PROFIT TARGET + STOP-LOSS")
    print("  Enter at O[di+1], hold up to 3 days, exit early if target/stop hit")
    print("=" * 130)

    for atr_max in [10.0, 12.0]:
        for mc in [0.5, 0.7]:
            for pt in [1.5, 2.0, 3.0, 5.0]:
                for sl in [0.0, -1.0, -2.0]:
                    label = f"EXTENDED atr<{atr_max:.0f}% c={mc:.1f} pt={pt:.1f}% sl={sl:.1f}% hold3"
                    r = backtest(atr_norm_max=atr_max, max_corr=mc,
                                 dd_tiers=DD_AGGR100,
                                 regime_lo=0.5, regime_hi=1.5,
                                 exit_mode='extended_target',
                                 profit_target=pt, stop_loss_pct=sl,
                                 max_hold=3, top_n=3)
                    pr(r, label)
                    all_results.append({**r,
                                        'label': f'ext_atr{atr_max:.0f}_c{mc:.1f}_pt{pt:.1f}_sl{sl:.1f}',
                                        'section': 1, 'exit_mode': 'extended_target',
                                        'atr_norm_max': atr_max, 'max_corr': mc,
                                        'profit_target': pt, 'stop_loss_pct': sl,
                                        'max_hold': 3, 'top_n': 3})

    # ===================== SECTION 2: INTRADAY HIGH EXIT SIMULATION =====================
    print("\n" + "=" * 130)
    print("  SECTION 2: INTRADAY HIGH EXIT — Use H[di+1] as proxy for intraday high")
    print("  If H > entry*(1+target), assume exit at target price. Else exit at C.")
    print("=" * 130)

    for atr_max in [10.0, 12.0]:
        for mc in [0.5, 0.7]:
            for pt in [1.0, 1.5, 2.0, 2.5]:
                label = f"INTRA_HIGH atr<{atr_max:.0f}% c={mc:.1f} pt={pt:.1f}%"
                r = backtest(atr_norm_max=atr_max, max_corr=mc,
                             dd_tiers=DD_AGGR100,
                             regime_lo=0.5, regime_hi=1.5,
                             exit_mode='intraday_high',
                             profit_target=pt, stop_loss_pct=0,
                             max_hold=1, top_n=3)
                pr(r, label)
                all_results.append({**r,
                                    'label': f'intra_atr{atr_max:.0f}_c{mc:.1f}_pt{pt:.1f}',
                                    'section': 2, 'exit_mode': 'intraday_high',
                                    'atr_norm_max': atr_max, 'max_corr': mc,
                                    'profit_target': pt, 'stop_loss_pct': 0,
                                    'max_hold': 1, 'top_n': 3})

    # ===================== SECTION 3: NEXT-DAY GAP EXIT =====================
    print("\n" + "=" * 130)
    print("  SECTION 3: GAP EXIT — Enter O[di+1], exit at O[di+2] if gap-up target met")
    print("  If O[di+2] > entry*(1+gap_target), exit at O[di+2]. Else exit at C[di+2].")
    print("=" * 130)

    for atr_max in [10.0, 12.0]:
        for mc in [0.5, 0.7]:
            for pt in [0.5, 1.0, 1.5]:
                label = f"GAP_EXIT atr<{atr_max:.0f}% c={mc:.1f} gap={pt:.1f}%"
                r = backtest(atr_norm_max=atr_max, max_corr=mc,
                             dd_tiers=DD_AGGR100,
                             regime_lo=0.5, regime_hi=1.5,
                             exit_mode='gap_exit',
                             profit_target=pt, stop_loss_pct=0,
                             max_hold=1, top_n=3)
                pr(r, label)
                all_results.append({**r,
                                    'label': f'gap_atr{atr_max:.0f}_c{mc:.1f}_pt{pt:.1f}',
                                    'section': 3, 'exit_mode': 'gap_exit',
                                    'atr_norm_max': atr_max, 'max_corr': mc,
                                    'profit_target': pt, 'stop_loss_pct': 0,
                                    'max_hold': 1, 'top_n': 3})

    # ===================== SECTION 4: RANKED RESULTS (Full Period) =====================
    print("\n" + "=" * 130)
    print("  SECTION 4: ALL CONFIGS RANKED BY ANNUAL RETURN (full period)")
    print("=" * 130)

    all_results.sort(key=lambda x: -x['ann'])
    for i, r in enumerate(all_results[:40]):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i+1:2d} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d} | {r['label']}")

    print("\n" + "=" * 130)
    print("  SECTION 4b: ALL CONFIGS RANKED BY R/M RATIO (risk-adjusted)")
    print("=" * 130)

    all_rm = sorted(all_results, key=lambda x: -abs(x['ann']/x['mdd']) if x['mdd'] != 0 else 0)
    for i, r in enumerate(all_rm[:40]):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i+1:2d} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d} | {r['label']}")

    # ===================== SECTION 5: DELTA vs BASELINE =====================
    print("\n" + "=" * 130)
    print("  SECTION 5: DELTA vs BASELINE — How does each exit mode compare?")
    print("=" * 130)

    # Get matching baselines
    base_lookup = {}
    for r in all_results:
        if r['exit_mode'] == 'hold1_baseline':
            key = (r['atr_norm_max'], r['max_corr'])
            base_lookup[key] = r

    for r in all_rm:
        if r['exit_mode'] == 'hold1_baseline':
            continue
        key = (r['atr_norm_max'], r['max_corr'])
        base = base_lookup.get(key)
        if base:
            r['delta_ann'] = r['ann'] - base['ann']
            r['delta_mdd'] = r['mdd'] - base['mdd']
            base_rm = abs(base['ann'] / base['mdd']) if base['mdd'] != 0 else 0
            cur_rm = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
            r['delta_rm'] = cur_rm - base_rm
        else:
            r['delta_ann'] = 0
            r['delta_mdd'] = 0
            r['delta_rm'] = 0

    # Show best improvements per exit mode
    for mode in ['extended_target', 'intraday_high', 'gap_exit']:
        mode_items = [r for r in all_rm if r['exit_mode'] == mode and 'delta_rm' in r]
        if not mode_items:
            continue
        mode_items.sort(key=lambda x: -x.get('delta_rm', -999))
        print(f"\n  {mode} — Top 5 by R/M improvement over baseline:")
        for i, r in enumerate(mode_items[:5]):
            marker = "IMPROVED" if r.get('delta_rm', 0) > 0 else "WORSE"
            print(f"    #{i+1} | dAnn={r['delta_ann']:+.1f}% | dMDD={r['delta_mdd']:+.1f}% | dR/M={r['delta_rm']:+.2f} | {r['label']} | {marker}")

    # ===================== SECTION 6: WALK-FORWARD TOP 20 =====================
    print("\n" + "=" * 130)
    print("  SECTION 6: WALK-FORWARD VALIDATION — Top 20 by R/M")
    print("=" * 130)

    # Deduplicate by label
    seen = set()
    wf_candidates = []
    for r in all_rm:
        lbl = r['label']
        if lbl not in seen:
            seen.add(lbl)
            wf_candidates.append(r)

    wf_all = {}
    for r in wf_candidates[:20]:
        lbl = r['label']
        wf_kwargs = {
            'atr_norm_max': r['atr_norm_max'],
            'max_corr': r['max_corr'],
            'dd_tiers': DD_AGGR100,
            'regime_lo': 0.5, 'regime_hi': 1.5,
            'exit_mode': r['exit_mode'],
            'profit_target': r['profit_target'],
            'stop_loss_pct': r['stop_loss_pct'],
            'max_hold': r['max_hold'],
            'top_n': r['top_n'],
        }
        wf_res = walk_forward(label=lbl, **wf_kwargs)
        wf_all[lbl] = (wf_res, r)
        print_wf(wf_res, lbl)

    # ===================== SECTION 7: WF COMPARISON TABLE =====================
    print("\n" + "=" * 130)
    print("  SECTION 7: WF COMPARISON TABLE — All configs ranked by WF avg")
    print("=" * 130)

    wf_ranked = []
    for lbl, (wf_res, r_info) in wf_all.items():
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        best_ann = max(r['ann'] for r in wf_res.values())
        pos = sum(1 for r in wf_res.values() if r['ann'] > 0)
        total_n = sum(r['n'] for r in wf_res.values())
        avg_wr = np.mean([r['wr'] for r in wf_res.values()])
        ratio = abs(avg_ann / worst_mdd) if worst_mdd != 0 else 0
        wf_ranked.append((avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr,
                          ratio, lbl, wf_res, r_info))

    wf_ranked.sort(key=lambda x: -x[0])
    print(f"\n  Ranked by WF Average Annual Return:")
    print(f"  {'#':>3s}  {'WF AVG':>8s} {'WorstDD':>8s} {'R/M':>6s} {'Pos':>4s} {'N':>5s} {'WR':>5s}  {'Label'}")
    print(f"  {'-'*105}")
    for i, (avg_ann, wmdd, bann, pos, tn, awr, ratio, lbl, wf_res, ri) in enumerate(wf_ranked):
        print(f"  {i+1:3d}  {avg_ann:>+8.0f}% {wmdd:>7.0f}% {ratio:6.2f} {pos:4d}/6 {tn:5d} {awr:5.1f}%  {lbl}")

    # ===================== SECTION 8: BEST RISK-ADJUSTED (WF R/M) =====================
    print("\n" + "=" * 130)
    print("  SECTION 8: BEST RISK-ADJUSTED (WF R/M)")
    print("=" * 130)

    wf_ra = sorted(wf_ranked, key=lambda x: -x[6])
    for i, (avg_ann, wmdd, bann, pos, tn, awr, ratio, lbl, wf_res, ri) in enumerate(wf_ra[:15]):
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  #{i+1} WF AVG={avg_ann:+.0f}% | WorstMDD={wmdd:.0f}% | R/M={ratio:.2f} | {pos}/6 pos | {lbl}")
        print(f"     {ws}")

    # ===================== SECTION 9: BEST PER EXIT MODE =====================
    print("\n" + "=" * 130)
    print("  SECTION 9: BEST PER EXIT MODE")
    print("=" * 130)

    exit_modes = {
        'hold1_baseline': 'Hold-1 Baseline',
        'extended_target': 'Extended Hold (3d) + Target',
        'intraday_high': 'Intraday High Exit',
        'gap_exit': 'Gap-Up Exit',
    }

    for mode_key, mode_name in exit_modes.items():
        mode_items = [(avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri)
                      for avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri in wf_ranked
                      if ri['exit_mode'] == mode_key]
        if not mode_items:
            print(f"\n  {mode_name:30s}: no WF results")
            continue
        mode_items.sort(key=lambda x: -x[6])
        best = mode_items[0]
        avg_ann, wmdd, bann, pos, tn, awr, ratio, lbl, wf_res, ri = best
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  {mode_name:30s} best: WF AVG={avg_ann:+.0f}% | WorstMDD={wmdd:.0f}% | R/M={ratio:.2f} | {lbl}")
        print(f"     {ws}")

    # ===================== SECTION 10: DELTA vs BASELINE (WF) =====================
    print("\n" + "=" * 130)
    print("  SECTION 10: IMPROVEMENT vs HOLD-1 BASELINE (WF)")
    print("=" * 130)

    # Find best baseline WF result
    base_wf_items = [(avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri)
                     for avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri in wf_ranked
                     if ri['exit_mode'] == 'hold1_baseline']
    if base_wf_items:
        base_wf_items.sort(key=lambda x: -x[6])
        b_avg = base_wf_items[0][0]
        b_wmdd = base_wf_items[0][1]
        b_rm = base_wf_items[0][6]
        b_lbl = base_wf_items[0][7]
        print(f"\n  Best baseline: {b_lbl} | WF AVG={b_avg:+.0f}% | WorstMDD={b_wmdd:.0f}% | R/M={b_rm:.2f}")
    else:
        # Fall back to computing baseline WF
        for atr_max in [12.0]:
            for mc in [0.7]:
                b_wf = walk_forward(label=f'base_atr{atr_max:.0f}_c{mc:.1f}',
                                    atr_norm_max=atr_max, max_corr=mc,
                                    dd_tiers=DD_AGGR100,
                                    regime_lo=0.5, regime_hi=1.5,
                                    exit_mode='hold1_baseline',
                                    profit_target=0, stop_loss_pct=0,
                                    max_hold=1, top_n=3)
                b_avg = np.mean([r['ann'] for r in b_wf.values()])
                b_wmdd = min(r['mdd'] for r in b_wf.values())
                b_rm = abs(b_avg / b_wmdd) if b_wmdd != 0 else 0
                b_lbl = f'base_atr{atr_max:.0f}_c{mc:.1f}'
                print(f"\n  Baseline (computed): {b_lbl} | WF AVG={b_avg:+.0f}% | WorstMDD={b_wmdd:.0f}% | R/M={b_rm:.2f}")
                print_wf(b_wf, b_lbl)

    deltas = []
    for avg_ann, wmdd, bann, pos, tn, awr, ratio, lbl, wf_res, ri in wf_ranked:
        if ri['exit_mode'] == 'hold1_baseline': continue
        delta_ann = avg_ann - b_avg
        delta_rm = ratio - b_rm
        deltas.append((delta_ann, delta_rm, ratio, avg_ann, wmdd, pos, lbl, wf_res, ri))

    deltas.sort(key=lambda x: -x[1])
    print(f"\n  Configs by R/M improvement over hold-1 baseline:")
    for i, (da, drm, ratio, avg, wmdd, pos, lbl, wfr, ri) in enumerate(deltas):
        marker = "*** IMPROVED" if drm > 0 else "    worse"
        mode = ri['exit_mode']
        pt = ri['profit_target']
        sl = ri['stop_loss_pct']
        print(f"  {i+1:2d} | R/M={ratio:.2f} (dR/M={drm:+.2f}) | dAnn={da:+.0f}% | {pos}/6 | {mode} pt={pt:.1f}% sl={sl:.1f}% | {marker} | {lbl}")

    # ===================== SECTION 11: TOP 5 DETAIL PER MODE =====================
    print("\n" + "=" * 130)
    print("  SECTION 11: TOP 5 DETAIL — Best from each exit mode (WF R/M)")
    print("=" * 130)

    for mode_key, mode_name in exit_modes.items():
        mode_items = [(avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri)
                      for avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri in wf_ranked
                      if ri['exit_mode'] == mode_key]
        if not mode_items:
            continue
        mode_items.sort(key=lambda x: -x[6])
        print(f"\n  --- {mode_name} ---")
        for i, (avg_ann, wmdd, bann, pos, tn, awr, ratio, lbl, wf_res, ri) in enumerate(mode_items[:5]):
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(wf_res.items())])
            pt = ri['profit_target']
            sl = ri['stop_loss_pct']
            mh = ri['max_hold']
            print(f"    #{i+1}: {lbl}")
            print(f"         WF AVG={avg_ann:+.0f}% | WorstMDD={wmdd:.0f}% | R/M={ratio:.2f} | {pos}/6 | pt={pt:.1f}% sl={sl:.1f}% hold={mh}")
            print(f"         {ws}")

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 130)
    print("  FINAL SUMMARY")
    print("=" * 130)

    print(f"\n  Hold-1 Baseline: WF AVG={b_avg:+.0f}% | WorstMDD={b_wmdd:.0f}% | R/M={b_rm:.2f}")

    if wf_ra:
        best = wf_ra[0]
        print(f"\n  Best overall: {best[7]}")
        print(f"    WF AVG={best[0]:+.0f}% | WorstMDD={best[1]:.0f}% | R/M={best[6]:.2f}")
        delta_rm_best = best[6] - b_rm
        print(f"    R/M improvement over baseline: {delta_rm_best:+.2f}")

        # Best per exit mode
        print(f"\n  Best per exit mode (WF R/M):")
        for mode_key, mode_name in exit_modes.items():
            mode_items = [(avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri)
                          for avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri in wf_ranked
                          if ri['exit_mode'] == mode_key]
            if not mode_items: continue
            mode_items.sort(key=lambda x: -x[6])
            s = mode_items[0]
            delta_rm = s[6] - b_rm
            marker = "BETTER" if delta_rm > 0 else "WORSE"
            print(f"    {mode_name:30s}: R/M={s[6]:.2f} (dR/M={delta_rm:+.2f}) | WF={s[0]:+.0f}% | {s[7]} | {marker}")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
