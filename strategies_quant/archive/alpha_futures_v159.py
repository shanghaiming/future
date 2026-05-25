"""
Alpha Futures V159 -- Anti-Fragile Loss Recovery Strategies
=============================================================================
Goal: Explore whether intelligent post-loss behavior can improve returns
WITHOUT the full Kitchen Sink complexity from V146.

Five strategies tested:
  1. Anti-Martingale scaling: size up after win, down after loss
  2. Consecutive loss gate: skip N signals after M consecutive losses (brief pause)
  3. Equity curve momentum sizing: base or half based on 10-day equity slope
  4. Drawdown recovery accelerator: boost size after equity new high
  5. Trade clustering: boost/reduce based on last 3 trade outcomes

Base: top_n=2, DD70/60/40/20, corr<0.5, hold=1, balanced
Each tested with Kitchen Sink sizing for fair comparison.
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
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    return (final / initial) ** (1.0 / (n_days / 252)) * 100 - 100


def main():
    print("=" * 130)
    print("  V159 -- Anti-Fragile Loss Recovery Strategies")
    print("=" * 130)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    print(f"  {NS} commodities, {ND} days")

    # ===================== PRECOMPUTE =====================
    print("\n[Precompute]...", flush=True)
    t0 = time.time()

    RET = np.full((NS, ND), np.nan)
    ROC5 = np.full((NS, ND), np.nan)
    ROC20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di - 1]) and c[di - 1] > 0:
                RET[si, di] = (c[di] / c[di - 1] - 1) * 100
        ROC5[si] = talib.ROC(c, timeperiod=5)
        ROC20[si] = talib.ROC(c, timeperiod=20)

    ATR14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        ATR14[si] = talib.ATR(H[si].astype(np.float64), L[si].astype(np.float64),
                               C[si].astype(np.float64), timeperiod=14)

    ZSCORE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = RET[si, di - 20:di]
            v = rets[~np.isnan(rets)]
            if len(v) < 10:
                continue
            s = np.std(v, ddof=1)
            if s > 0 and not np.isnan(RET[si, di]):
                ZSCORE[si, di] = (RET[si, di] - np.mean(v)) / s

    OV_GAP = np.full((NS, ND), np.nan)
    ID_RET = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            o, c = O[si, di], C[si, di]
            if not np.isnan(o) and not np.isnan(c):
                if di > 0 and not np.isnan(C[si, di - 1]) and C[si, di - 1] > 0:
                    OV_GAP[si, di] = (o - C[si, di - 1]) / C[si, di - 1] * 100
                if o > 0:
                    ID_RET[si, di] = (c - o) / o * 100

    print(f"  Done ({time.time() - t0:.1f}s)")

    # ===================== SIGNAL DEFINITIONS =====================
    def sig_v121(di, edi):
        c = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5:
                continue
            rp = ROC5[s, di - 1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp:
                continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0:
                continue
            c.append((roc * zs, s, ep, 'v121'))
        return c

    def sig_ov_id(di, edi):
        c = []
        for s in range(NS):
            ov = OV_GAP[s, di]; idr = ID_RET[s, di]; roc = ROC5[s, di]
            if any(np.isnan(x) for x in [ov, idr, roc]):
                continue
            if ov <= 0.3 or idr <= 0.3 or roc <= 1.0:
                continue
            zs = ZSCORE[s, di]
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0:
                continue
            z_bonus = zs if not np.isnan(zs) and zs > 1.0 else 1.0
            c.append(((ov + idr) * roc * z_bonus * 2, s, ep, 'ov_id'))
        return c

    def sig_final_flag(di, edi):
        c = []
        for s in range(NS):
            roc20 = ROC20[s, di]
            if np.isnan(roc20) or roc20 <= 5.0 or di < 6:
                continue
            h5 = H[s, di - 4:di + 1]; l5 = L[s, di - 4:di + 1]
            if any(np.isnan(x) for x in h5) or any(np.isnan(x) for x in l5):
                continue
            r5 = np.max(h5) - np.min(l5)
            atr = ATR14[s, di]
            if np.isnan(atr) or atr <= 0 or r5 > atr * 3.0:
                continue
            h4 = np.max(H[s, di - 4:di])
            cp = C[s, di]
            if np.isnan(cp) or cp <= h4:
                continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0:
                continue
            c.append((roc20 * (cp - h4) / atr, s, ep, 'ff'))
        return c

    def sig_union(di, edi):
        all_sigs = {}
        for item in sig_v121(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs:
                all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc * 3
            all_sigs[s][2].append('v121')
        for item in sig_ov_id(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs:
                all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc * 2
            all_sigs[s][2].append('ov_id')
        for item in sig_final_flag(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs:
                all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc
            all_sigs[s][2].append('ff')
        return [(sc, s, ep, '+'.join(sigs)) for s, (sc, ep, sigs) in all_sigs.items()]

    # ===================== HELPER: Correlation =====================
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

    # ===================== HELPER: DD-based sizing =====================
    def dd_size(pv, high_water, tiers):
        if high_water <= 0:
            return tiers[0][1]
        dd = (pv - high_water) / high_water
        for dd_thresh, size_frac in tiers:
            if dd >= -dd_thresh:
                return size_frac
        return tiers[-1][1]

    # ===================== HELPER: WR-adaptive sizing =====================
    def wr_size(trades, window=20):
        if len(trades) < window:
            return 1.0
        recent = trades[-window:]
        wr = np.mean([1 if t > 0 else 0 for t in recent])
        if wr > 0.65:
            return 1.3
        elif wr >= 0.50:
            return 1.0
        else:
            return 0.5

    # ===================== HELPER: Composite regime =====================
    BREADTH = np.full(ND, np.nan)
    for di in range(5, ND):
        pos_count = 0; total = 0
        for si in range(NS):
            r = ROC5[si, di]
            if not np.isnan(r):
                total += 1
                if r > 0:
                    pos_count += 1
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
        window = MKT_RET[di - 20:di]
        valid = window[~np.isnan(window)]
        if len(valid) >= 10:
            MKT_VOL[di] = np.std(valid, ddof=1)

    valid_vols = MKT_VOL[~np.isnan(MKT_VOL)]
    VOL_MEDIAN = np.median(valid_vols) if len(valid_vols) > 0 else 1.0

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

    # ===================== UNIFIED BACKTEST ENGINE =====================
    # loss_recovery: one of 'none','anti_martingale','consec_loss_gate',
    #                'eq_momentum','dd_recovery','trade_clustering'
    def backtest(start_di=MIN_TRAIN, end_di=None,
                 loss_recovery='none',
                 # Anti-Martingale params
                 am_win_mult=1.1, am_loss_mult=0.8,
                 # Consecutive loss gate params
                 clg_n_losses=3, clg_skip=1,
                 # Equity momentum params
                 em_window=10, em_down_frac=0.5,
                 # DD recovery params
                 dr_boost=1.3, dr_boost_trades=3,
                 # Trade clustering params
                 tc_window=3, tc_win_boost=1.3, tc_loss_cut=0.7,
                 # Kitchen Sink base params
                 dd_tiers=None, max_corr=0.5,
                 hold=1, top_n=2):

        if end_di is None:
            end_di = ND
        if dd_tiers is None:
            dd_tiers = [(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)]

        cash = float(CASH0)
        positions = []
        trades = []          # list of pnl_pct (signed)
        daily_eq = []
        high_water = float(CASH0)

        # Anti-Martingale state
        am_current_mult = 1.0

        # Consecutive loss gate state
        clg_consec_losses = 0
        clg_skip_remaining = 0

        # DD recovery state
        dr_boost_remaining = 0

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

            # Close positions past hold period
            cl = []
            for p in positions:
                if di - p['entry_di'] >= p['hold_days']:
                    ep = C[p['si'], di]
                    if np.isnan(ep) or ep <= 0:
                        ep = p['entry_price']
                    m = MULT.get(p['sym'], DEF_MULT)
                    pnl = (ep - p['entry_price']) * m * p['lots']
                    inv = p['entry_price'] * m * abs(p['lots'])
                    pp = pnl / inv * 100 if inv > 0 else 0
                    cash += ep * m * abs(p['lots']) * (1 - COMM)
                    trades.append(pp)
                    cl.append(p)

                    # Update loss recovery states based on this trade outcome
                    if loss_recovery == 'anti_martingale':
                        if pp > 0:
                            am_current_mult = min(am_current_mult * am_win_mult, 2.0)
                        else:
                            am_current_mult = max(am_current_mult * am_loss_mult, 0.3)

                    if loss_recovery == 'consec_loss_gate':
                        if pp > 0:
                            clg_consec_losses = 0
                        else:
                            clg_consec_losses += 1
                            if clg_consec_losses >= clg_n_losses:
                                clg_skip_remaining = clg_skip
                                clg_consec_losses = 0

                    if loss_recovery == 'dd_recovery':
                        if pp > 0 and pv >= high_water:
                            dr_boost_remaining = dr_boost_trades

                    if loss_recovery == 'trade_clustering':
                        pass  # evaluated live from trades list

            for p in cl:
                positions.remove(p)

            # --- Kitchen Sink sizing: DD * WR * Regime ---
            dd_sz = dd_size(pv, high_water, dd_tiers)
            wr_mult_val = wr_size(trades, window=20)
            composite = compute_composite(di, daily_eq, high_water)
            regime_mult = 0.5 + composite  # 0.5-1.5
            base_pos_size = dd_sz * wr_mult_val * regime_mult

            # --- Apply loss recovery modifier ---
            if loss_recovery == 'anti_martingale':
                pos_size = base_pos_size * am_current_mult

            elif loss_recovery == 'consec_loss_gate':
                if clg_skip_remaining > 0:
                    clg_skip_remaining -= 1
                    continue  # skip this signal day
                pos_size = base_pos_size

            elif loss_recovery == 'eq_momentum':
                if len(daily_eq) >= em_window:
                    eq_arr = np.array(daily_eq[-em_window:])
                    x = np.arange(em_window)
                    try:
                        slope = np.polyfit(x, eq_arr, 1)[0]
                        if slope < 0:
                            pos_size = base_pos_size * em_down_frac
                        else:
                            pos_size = base_pos_size
                    except Exception:
                        pos_size = base_pos_size
                else:
                    pos_size = base_pos_size

            elif loss_recovery == 'dd_recovery':
                if dr_boost_remaining > 0:
                    pos_size = base_pos_size * dr_boost
                    dr_boost_remaining -= 1
                else:
                    pos_size = base_pos_size

            elif loss_recovery == 'trade_clustering':
                if len(trades) >= tc_window:
                    last_n = trades[-tc_window:]
                    n_wins = sum(1 for t in last_n if t > 0)
                    if n_wins == tc_window:
                        pos_size = base_pos_size * tc_win_boost
                    elif n_wins == 0:
                        pos_size = base_pos_size * tc_loss_cut
                    else:
                        pos_size = base_pos_size
                else:
                    pos_size = base_pos_size

            else:  # 'none' = pure Kitchen Sink
                pos_size = base_pos_size

            # Clamp
            pos_size = max(0.05, min(0.95, pos_size))

            # --- Enter positions ---
            if len(positions) >= top_n:
                continue
            edi = di + 1
            if edi >= end_di:
                continue

            held_si = set(p['si'] for p in positions)

            # Cross+Corr: best V121 + best Union
            cands_v121 = sig_v121(di, edi)
            cands_union = sig_union(di, edi)
            cands_v121.sort(key=lambda x: -x[0])
            cands_union.sort(key=lambda x: -x[0])

            best_v121 = None
            for c in cands_v121:
                if c[1] not in held_si:
                    best_v121 = c
                    break

            best_union = None
            for c in cands_union:
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
                if s in set(p['si'] for p in positions):
                    continue
                if len(positions) >= top_n:
                    break
                cap = cash_snapshot * pct / n_planned
                sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                ct = max(1, int(cap / (pr * m * (1 + COMM))))
                ci = pr * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                    ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                if ct <= 0 or ci <= 0 or ci > cash:
                    continue
                cash -= ci
                positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                  'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': hold,
                                  'sig': sig_str, 'score': sc})

        # Close remaining
        for p in positions:
            ae = end_di - 1
            ep = C[p['si'], min(ae, ND - 1)]
            if np.isnan(ep) or ep <= 0:
                ep = p['entry_price']
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
        print(f"  {label:85s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d}")

    def walk_forward(loss_recovery='none', dd_tiers=None, max_corr=0.5,
                     hold=1, top_n=2,
                     am_win_mult=1.1, am_loss_mult=0.8,
                     clg_n_losses=3, clg_skip=1,
                     em_window=10, em_down_frac=0.5,
                     dr_boost=1.3, dr_boost_trades=3,
                     tc_window=3, tc_win_boost=1.3, tc_loss_cut=0.7,
                     label=""):
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None:
                    ys = di
                if dates[di].year == yr:
                    ye = di + 1
            if ys is None:
                continue
            r = backtest(start_di=ys, end_di=ye,
                         loss_recovery=loss_recovery,
                         dd_tiers=dd_tiers, max_corr=max_corr,
                         hold=hold, top_n=top_n,
                         am_win_mult=am_win_mult, am_loss_mult=am_loss_mult,
                         clg_n_losses=clg_n_losses, clg_skip=clg_skip,
                         em_window=em_window, em_down_frac=em_down_frac,
                         dr_boost=dr_boost, dr_boost_trades=dr_boost_trades,
                         tc_window=tc_window, tc_win_boost=tc_win_boost,
                         tc_loss_cut=tc_loss_cut)
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

    DD_TIERS = [(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)]

    # ===================== BASELINE: Kitchen Sink (no loss recovery) =====================
    print("\n" + "=" * 130)
    print("  BASELINE: Kitchen Sink (V146 base) -- no loss recovery modifier")
    print("=" * 130)

    r_base = backtest(loss_recovery='none', dd_tiers=DD_TIERS, max_corr=0.5)
    pr(r_base, "KS Baseline: DD70/60/40/20 WR Regime corr<0.5")

    # ===================== STRATEGY 1: Anti-Martingale Scaling =====================
    print("\n" + "=" * 130)
    print("  STRATEGY 1: Anti-Martingale Scaling")
    print("  After win: mult *= win_factor (cap 2.0). After loss: mult *= loss_factor (floor 0.3)")
    print("=" * 130)

    s1_configs = [
        (1.1, 0.8, "S1: AM win*1.1 loss*0.8"),
        (1.15, 0.75, "S1: AM win*1.15 loss*0.75"),
        (1.2, 0.7, "S1: AM win*1.2 loss*0.7"),
        (1.1, 0.9, "S1: AM win*1.1 loss*0.9 (mild)"),
        (1.2, 0.85, "S1: AM win*1.2 loss*0.85 (mild loss)"),
        (1.05, 0.85, "S1: AM win*1.05 loss*0.85 (very mild)"),
        (1.3, 0.6, "S1: AM win*1.3 loss*0.6 (aggressive)"),
    ]

    s1_results = []
    for w, l, label in s1_configs:
        r = backtest(loss_recovery='anti_martingale', dd_tiers=DD_TIERS,
                     max_corr=0.5, am_win_mult=w, am_loss_mult=l)
        r['desc'] = label
        s1_results.append(r)
        pr(r, label)

    # ===================== STRATEGY 2: Consecutive Loss Gate =====================
    print("\n" + "=" * 130)
    print("  STRATEGY 2: Consecutive Loss Gate")
    print("  After N consecutive losses, skip next M signals. Brief pause, not circuit breaker.")
    print("=" * 130)

    s2_configs = [
        (3, 1, "S2: CLG 3-loss skip 1"),
        (3, 2, "S2: CLG 3-loss skip 2"),
        (4, 1, "S2: CLG 4-loss skip 1"),
        (2, 1, "S2: CLG 2-loss skip 1"),
        (5, 1, "S2: CLG 5-loss skip 1"),
        (3, 3, "S2: CLG 3-loss skip 3"),
        (4, 2, "S2: CLG 4-loss skip 2"),
    ]

    s2_results = []
    for n_l, sk, label in s2_configs:
        r = backtest(loss_recovery='consec_loss_gate', dd_tiers=DD_TIERS,
                     max_corr=0.5, clg_n_losses=n_l, clg_skip=sk)
        r['desc'] = label
        s2_results.append(r)
        pr(r, label)

    # ===================== STRATEGY 3: Equity Curve Momentum Sizing =====================
    print("\n" + "=" * 130)
    print("  STRATEGY 3: Equity Curve Momentum Sizing")
    print("  If equity curve 10-day slope positive: base_size. Negative: base_size * down_frac")
    print("=" * 130)

    s3_configs = [
        (10, 0.5, "S3: EQ mom 10d slope, down=0.50"),
        (10, 0.6, "S3: EQ mom 10d slope, down=0.60"),
        (10, 0.7, "S3: EQ mom 10d slope, down=0.70"),
        (15, 0.5, "S3: EQ mom 15d slope, down=0.50"),
        (5, 0.5, "S3: EQ mom 5d slope, down=0.50"),
        (20, 0.5, "S3: EQ mom 20d slope, down=0.50"),
        (10, 0.4, "S3: EQ mom 10d slope, down=0.40"),
    ]

    s3_results = []
    for w, df, label in s3_configs:
        r = backtest(loss_recovery='eq_momentum', dd_tiers=DD_TIERS,
                     max_corr=0.5, em_window=w, em_down_frac=df)
        r['desc'] = label
        s3_results.append(r)
        pr(r, label)

    # ===================== STRATEGY 4: Drawdown Recovery Accelerator =====================
    print("\n" + "=" * 130)
    print("  STRATEGY 4: Drawdown Recovery Accelerator")
    print("  Keep sizing constant in DD. After equity new high, BOOST size for N trades.")
    print("=" * 130)

    s4_configs = [
        (1.3, 3, "S4: DR boost*1.3 for 3 trades after new high"),
        (1.3, 5, "S4: DR boost*1.3 for 5 trades after new high"),
        (1.2, 3, "S4: DR boost*1.2 for 3 trades after new high"),
        (1.5, 2, "S4: DR boost*1.5 for 2 trades after new high"),
        (1.4, 3, "S4: DR boost*1.4 for 3 trades after new high"),
        (1.2, 5, "S4: DR boost*1.2 for 5 trades after new high"),
        (1.6, 2, "S4: DR boost*1.6 for 2 trades after new high"),
    ]

    s4_results = []
    for bst, nt, label in s4_configs:
        r = backtest(loss_recovery='dd_recovery', dd_tiers=DD_TIERS,
                     max_corr=0.5, dr_boost=bst, dr_boost_trades=nt)
        r['desc'] = label
        s4_results.append(r)
        pr(r, label)

    # ===================== STRATEGY 5: Trade Clustering =====================
    print("\n" + "=" * 130)
    print("  STRATEGY 5: Trade Clustering")
    print("  If last 3 trades all winners: boost by 30%. All losers: cut by 30%.")
    print("=" * 130)

    s5_configs = [
        (3, 1.3, 0.7, "S5: TC 3-win*1.3 3-loss*0.7"),
        (3, 1.2, 0.8, "S5: TC 3-win*1.2 3-loss*0.8"),
        (3, 1.4, 0.6, "S5: TC 3-win*1.4 3-loss*0.6"),
        (3, 1.5, 0.5, "S5: TC 3-win*1.5 3-loss*0.5 (aggressive)"),
        (2, 1.3, 0.7, "S5: TC 2-win*1.3 2-loss*0.7"),
        (4, 1.3, 0.7, "S5: TC 4-win*1.3 4-loss*0.7"),
        (3, 1.25, 0.75, "S5: TC 3-win*1.25 3-loss*0.75"),
    ]

    s5_results = []
    for tw, wb, lc, label in s5_configs:
        r = backtest(loss_recovery='trade_clustering', dd_tiers=DD_TIERS,
                     max_corr=0.5, tc_window=tw, tc_win_boost=wb, tc_loss_cut=lc)
        r['desc'] = label
        s5_results.append(r)
        pr(r, label)

    # ===================== COMPREHENSIVE RANKING =====================
    print("\n" + "=" * 130)
    print("  COMPREHENSIVE RANKING (full period)")
    print("=" * 130)

    all_results = ([{'desc': 'KS Baseline (no LR)', **r_base}] +
                   s1_results + s2_results + s3_results + s4_results + s5_results)
    all_valid = [r for r in all_results if r.get('desc', '') and r['mdd'] > -80]

    # Sort by annual return
    all_valid.sort(key=lambda x: -x['ann'])
    print(f"\n  Top 20 by Annual Return:")
    for i, r in enumerate(all_valid[:20]):
        desc = r.get('desc', '')
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i + 1:2d}: {desc:85s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    # Sort by R/M ratio
    all_with_ratio = [(r, abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0) for r in all_valid]
    all_with_ratio.sort(key=lambda x: -x[1])
    print(f"\n  Top 20 by Ann/MDD Ratio:")
    for i, (r, ratio) in enumerate(all_with_ratio[:20]):
        desc = r.get('desc', '')
        print(f"  #{i + 1:2d}: {desc:85s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    # ===================== WALK-FORWARD FOR TOP CONFIGS =====================
    print("\n" + "=" * 130)
    print("  WALK-FORWARD VALIDATION FOR TOP CONFIGS")
    print("=" * 130)

    # Pick top configs by R/M ratio for WF -- need to reconstruct params
    seen = set()
    wf_top = []
    for r, ratio in all_with_ratio:
        desc = r.get('desc', '')
        if desc not in seen and not desc.startswith('KS Baseline'):
            seen.add(desc)
            wf_top.append(r)
        if len(wf_top) >= 15:
            break

    # Always include baseline
    wf_configs = [{'desc': 'KS Baseline (no LR)', 'loss_recovery': 'none'}]
    for r in wf_top:
        desc = r.get('desc', '')
        if desc.startswith('S1:'):
            # Parse AM params from the desc
            wf_configs.append({'desc': desc, 'loss_recovery': 'anti_martingale'})
        elif desc.startswith('S2:'):
            wf_configs.append({'desc': desc, 'loss_recovery': 'consec_loss_gate'})
        elif desc.startswith('S3:'):
            wf_configs.append({'desc': desc, 'loss_recovery': 'eq_momentum'})
        elif desc.startswith('S4:'):
            wf_configs.append({'desc': desc, 'loss_recovery': 'dd_recovery'})
        elif desc.startswith('S5:'):
            wf_configs.append({'desc': desc, 'loss_recovery': 'trade_clustering'})

    wf_all = {}
    for cfg in wf_configs:
        desc = cfg['desc']
        lr = cfg['loss_recovery']

        if lr == 'anti_martingale':
            # Parse win and loss multipliers from desc
            w, l = 1.1, 0.8
            if 'win*1.15' in desc:
                w = 1.15
            elif 'win*1.2' in desc:
                w = 1.2
            elif 'win*1.05' in desc:
                w = 1.05
            elif 'win*1.3' in desc:
                w = 1.3
            if 'loss*0.75' in desc:
                l = 0.75
            elif 'loss*0.7 ' in desc or 'loss*0.7(' in desc:
                l = 0.7
            elif 'loss*0.9' in desc:
                l = 0.9
            elif 'loss*0.85' in desc:
                l = 0.85
            elif 'loss*0.6' in desc:
                l = 0.6
            wf_res = walk_forward(loss_recovery='anti_martingale',
                                  am_win_mult=w, am_loss_mult=l, label=desc)

        elif lr == 'consec_loss_gate':
            n_l, sk = 3, 1
            if '2-loss' in desc:
                n_l = 2
            elif '4-loss' in desc:
                n_l = 4
            elif '5-loss' in desc:
                n_l = 5
            if 'skip 2' in desc:
                sk = 2
            elif 'skip 3' in desc:
                sk = 3
            wf_res = walk_forward(loss_recovery='consec_loss_gate',
                                  clg_n_losses=n_l, clg_skip=sk, label=desc)

        elif lr == 'eq_momentum':
            em_w, em_df = 10, 0.5
            if '15d' in desc:
                em_w = 15
            elif '5d' in desc:
                em_w = 5
            elif '20d' in desc:
                em_w = 20
            if 'down=0.60' in desc:
                em_df = 0.6
            elif 'down=0.70' in desc:
                em_df = 0.7
            elif 'down=0.40' in desc:
                em_df = 0.4
            wf_res = walk_forward(loss_recovery='eq_momentum',
                                  em_window=em_w, em_down_frac=em_df, label=desc)

        elif lr == 'dd_recovery':
            bst, nt = 1.3, 3
            if 'boost*1.2' in desc:
                bst = 1.2
            elif 'boost*1.4' in desc:
                bst = 1.4
            elif 'boost*1.5' in desc:
                bst = 1.5
            elif 'boost*1.6' in desc:
                bst = 1.6
            if 'for 2 ' in desc:
                nt = 2
            elif 'for 5 ' in desc:
                nt = 5
            wf_res = walk_forward(loss_recovery='dd_recovery',
                                  dr_boost=bst, dr_boost_trades=nt, label=desc)

        elif lr == 'trade_clustering':
            tw, wb, lc = 3, 1.3, 0.7
            if '2-win' in desc:
                tw = 2
            elif '4-win' in desc:
                tw = 4
            if '*1.2' in desc:
                wb = 1.2
            elif '*1.4' in desc:
                wb = 1.4
            elif '*1.5' in desc:
                wb = 1.5
            elif '*1.25' in desc:
                wb = 1.25
            if '*0.8' in desc:
                lc = 0.8
            elif '*0.6' in desc:
                lc = 0.6
            elif '*0.5' in desc:
                lc = 0.5
            elif '*0.75' in desc:
                lc = 0.75
            wf_res = walk_forward(loss_recovery='trade_clustering',
                                  tc_window=tw, tc_win_boost=wb, tc_loss_cut=lc,
                                  label=desc)

        else:  # none = baseline
            wf_res = walk_forward(loss_recovery='none', label=desc)

        wf_all[desc] = wf_res
        print_wf(wf_res, desc)

    # ===================== FINAL RANKING BY WF AVG with MDD FILTER =====================
    print("\n" + "=" * 130)
    print("  TOP 5 CONFIGS BY WF AVG ANNUAL RETURN (with WF MDD > -30%)")
    print("=" * 130)

    wf_scored = []
    for desc, wf_res in wf_all.items():
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        pos_years = sum(1 for r in wf_res.values() if r['ann'] > 0)
        avg_mdd = np.mean([r['mdd'] for r in wf_res.values()])
        avg_wr = np.mean([r['wr'] for r in wf_res.values()])
        wf_scored.append({
            'desc': desc, 'avg_ann': avg_ann, 'worst_mdd': worst_mdd,
            'pos_years': pos_years, 'avg_mdd': avg_mdd, 'avg_wr': avg_wr,
            'wf_res': wf_res
        })

    # Filter: worst WF MDD > -30%
    wf_filtered = [w for w in wf_scored if w['worst_mdd'] > -30]
    wf_filtered.sort(key=lambda x: -x['avg_ann'])

    print(f"\n  Configs with worst WF MDD > -30%: {len(wf_filtered)}/{len(wf_scored)}")

    for i, w in enumerate(wf_filtered[:5]):
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(w['wf_res'].items())])
        print(f"\n  #{i + 1}: {w['desc']}")
        print(f"       AvgWF={w['avg_ann']:+.0f}% | WorstWfMDD={w['worst_mdd']:.1f}% | "
              f"{w['pos_years']}/6 pos | AvgWR={w['avg_wr']:.1f}%")
        print(f"       {ws}")

    # ===================== ALSO SHOW TOP 5 WITHOUT MDD FILTER =====================
    print("\n" + "=" * 130)
    print("  TOP 5 CONFIGS BY WF AVG ANNUAL RETURN (no MDD filter)")
    print("=" * 130)

    wf_scored.sort(key=lambda x: -x['avg_ann'])
    for i, w in enumerate(wf_scored[:5]):
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(w['wf_res'].items())])
        print(f"\n  #{i + 1}: {w['desc']}")
        print(f"       AvgWF={w['avg_ann']:+.0f}% | WorstWfMDD={w['worst_mdd']:.1f}% | "
              f"{w['pos_years']}/6 pos | AvgWR={w['avg_wr']:.1f}%")
        print(f"       {ws}")

    # ===================== DETAILED WF TABLE =====================
    print("\n" + "=" * 130)
    print("  DETAILED WF TABLE: ALL TESTED CONFIGS")
    print("=" * 130)

    print(f"\n  {'Config':85s} | {'2020':>12s} | {'2021':>12s} | {'2022':>12s} | {'2023':>12s} | {'2024':>12s} | {'2025':>12s} | {'Avg':>7s} | {'WfMDD':>6s}")
    print(f"  {'-' * 85}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 7}-+-{'-' * 6}")

    for w in wf_scored:
        vals = []
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            if yr in w['wf_res']:
                vals.append(f"{w['wf_res'][yr]['ann']:+.0f}/{w['wf_res'][yr]['mdd']:.0f}")
            else:
                vals.append("N/A")
        print(f"  {w['desc']:85s} | {vals[0]:>12s} | {vals[1]:>12s} | {vals[2]:>12s} | {vals[3]:>12s} | {vals[4]:>12s} | {vals[5]:>12s} | {w['avg_ann']:>+6.0f}% | {w['worst_mdd']:>5.1f}%")

    # ===================== DELTA VS BASELINE =====================
    print("\n" + "=" * 130)
    print("  DELTA VS BASELINE (KS without loss recovery)")
    print("=" * 130)

    base_wf = wf_all.get('KS Baseline (no LR)', None)
    if base_wf:
        base_avg = np.mean([r['ann'] for r in base_wf.values()])
        print(f"\n  Baseline AvgWF: {base_avg:+.0f}%")
        print(f"\n  {'Config':85s} | {'AvgWF':>7s} | {'Delta':>7s} | {'WfMDD':>6s}")
        print(f"  {'-' * 85}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 6}")
        for w in wf_scored:
            delta = w['avg_ann'] - base_avg
            marker = " ***" if delta > 10 else ""
            print(f"  {w['desc']:85s} | {w['avg_ann']:>+6.0f}% | {delta:>+6.0f}% | {w['worst_mdd']:>5.1f}%{marker}")

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 130)
    print("  FINAL SUMMARY")
    print("=" * 130)

    print(f"\n  Full-period top 5 by R/M ratio:")
    for i, (r, ratio) in enumerate(all_with_ratio[:5]):
        desc = r.get('desc', '')
        print(f"  #{i + 1}: {desc:85s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    if wf_filtered:
        print(f"\n  WF top 5 (MDD>-30%):")
        for i, w in enumerate(wf_filtered[:5]):
            print(f"  #{i + 1}: {w['desc']:85s} | AvgWF={w['avg_ann']:+.0f}% | WorstWfMDD={w['worst_mdd']:.1f}% | {w['pos_years']}/6 pos")
    else:
        print(f"\n  No configs passed WF MDD>-30% filter.")

    print(f"\n  Elapsed: {time.time() - t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
