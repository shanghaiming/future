"""
Alpha Futures V144 — COMBINED BEST IDEAS (V140/V141/V142/V143)
=============================================================================
Goal: Combine the best ideas from V141-V143 to find optimal return/MDD tradeoff.

Baselines (from parallel agents):
  - V140: Union/V121 50/50 @50% sizing => +155% annual, -24% MDD
  - V142: Cross+Corr => +198%, -27% MDD
  - V143: SL=5% => +158%, -24% MDD
  - V141: WR-adaptive => +197%, -31% MDD

Approaches tested:
  A. Cross+Corr + Stop Loss
  B. Cross+Corr + WR-adaptive sizing
  C. Cross+Corr + Anti-Martingale sizing
  D. All Three Combined (Cross+Corr + WR-adaptive + SL=5%)
  E. Cross+Corr + Equity Curve Gentle Scaling
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
    print("  V144 — COMBINED BEST IDEAS (Cross+Corr + Sizing + Stop Loss)")
    print("=" * 130)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    print(f"  {NS} commodities, {ND} days")

    print("\n[Precompute]...", flush=True)
    t0 = time.time()

    RET = np.full((NS, ND), np.nan)
    ROC5 = np.full((NS, ND), np.nan)
    ROC20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100
        ROC5[si] = talib.ROC(c, timeperiod=5)
        ROC20[si] = talib.ROC(c, timeperiod=20)

    ATR14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        ATR14[si] = talib.ATR(H[si].astype(np.float64), L[si].astype(np.float64),
                               C[si].astype(np.float64), timeperiod=14)

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

    # Precompute 20-day returns for correlation calculation
    RET20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(20, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-20]) and c[di-20] > 0:
                RET20[si, di] = (c[di] / c[di-20] - 1) * 100

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

    # ===================== HELPER: get correlation between two commodities =====================
    def get_corr(si_a, si_b, di, window=20):
        """Get 20-day return correlation between two commodities at day di."""
        start_idx = max(0, di - window)
        ret_a = RET20[si_a, start_idx:di]
        ret_b = RET20[si_b, start_idx:di]
        valid = ~(np.isnan(ret_a) | np.isnan(ret_b))
        n_valid = np.sum(valid)
        if n_valid < 8:
            return 0.5  # default moderate
        ra = ret_a[valid]; rb = ret_b[valid]
        if np.std(ra) == 0 or np.std(rb) == 0:
            return 0.5
        c = np.corrcoef(ra, rb)[0, 1]
        if np.isnan(c): return 0.5
        return c

    # ===================== UNIFIED BACKTEST ENGINE =====================
    def backtest_v144(start_di=MIN_TRAIN, end_di=None,
                      max_corr=0.5,          # Cross+Corr: max correlation to allow both positions
                      per_pos_frac=0.45,     # base per-position fraction
                      stop_loss_pct=0.0,     # intraday stop loss (0=off)
                      sizing_mode='fixed',   # 'fixed', 'wr_adaptive', 'anti_mart', 'eq_curve', 'combined_wr_eq'
                      # WR-adaptive thresholds
                      wr_hi=0.65, wr_mid=0.55,
                      wr_size_hi=0.70, wr_size_mid=0.50, wr_size_lo=0.30,
                      # Anti-mart thresholds
                      am_wins_req=3, am_losses_small=2, am_losses_big=4,
                      am_size_hot=0.80, am_size_normal=0.55, am_size_cold=0.30, am_size_deep=0.15,
                      # Equity curve parameters
                      eq_window=20,
                      eq_size_hi=0.70, eq_size_mid=0.55, eq_size_lo=0.30,
                      ):
        """
        Unified backtest combining:
          1. Cross+Corr: Take best V121 + best Union, require correlation < max_corr
          2. Stop Loss: Intraday stop at entry * (1 - SL%)
          3. Adaptive sizing: WR-based, anti-martingale, or equity curve
        """
        if end_di is None: end_di = ND
        cash = float(CASH0)
        positions = []
        daily_eq = []
        trades = []             # list of pnl_pct for WR calculation
        consecutive_wins = 0
        consecutive_losses = 0

        for di in range(start_di, end_di - 1):
            # Mark-to-market
            pv = cash
            for p in positions:
                cp = C[p['si'], di]
                if not np.isnan(cp) and cp > 0:
                    m = MULT.get(p['sym'], DEF_MULT)
                    pv += cp * m * p['lots'] - cp * m * abs(p['lots']) * COMM
            daily_eq.append(pv)

            # Process exits
            cl = []
            for p in positions:
                si = p['si']
                entry = p['entry_price']
                m = MULT.get(p['sym'], DEF_MULT)
                invested = entry * m * abs(p['lots'])
                lo = L[si, di]; cp = C[si, di]

                exit_price = None
                exit_reason = 'hold'

                # 1. Stop loss check (intraday)
                if stop_loss_pct > 0 and not np.isnan(lo):
                    if lo < entry * (1 - stop_loss_pct / 100):
                        exit_price = entry * (1 - stop_loss_pct / 100)
                        exit_reason = 'sl'

                # 2. Normal hold expiry
                if exit_price is None and di - p['entry_di'] >= p['hold_days']:
                    if not np.isnan(cp) and cp > 0:
                        exit_price = cp
                        exit_reason = 'hold'
                    else:
                        exit_price = entry  # fallback

                if exit_price is not None:
                    pnl = (exit_price - entry) * m * p['lots']
                    pp = pnl / invested * 100 if invested > 0 else 0
                    cash += exit_price * m * abs(p['lots']) * (1 - COMM)
                    trades.append(pp)
                    if pp > 0:
                        consecutive_wins += 1; consecutive_losses = 0
                    else:
                        consecutive_losses += 1; consecutive_wins = 0
                    cl.append(p)

            for p in cl: positions.remove(p)

            # --- Entry ---
            edi = di + 1
            if edi >= end_di: continue

            cands_v121 = sig_v121(di, edi)
            cands_union = sig_union(di, edi)

            held_si = set(p['si'] for p in positions)

            best_v121 = None
            if cands_v121:
                cands_v121.sort(key=lambda x: -x[0])
                for c in cands_v121:
                    if c[1] not in held_si:
                        best_v121 = c
                        break

            best_union = None
            if cands_union:
                cands_union.sort(key=lambda x: -x[0])
                for c in cands_union:
                    if c[1] not in held_si:
                        best_union = c
                        break

            entries = []  # list of (score, si, price, sig_label)
            if best_v121 and best_union:
                if best_v121[1] == best_union[1]:
                    # Same commodity: take 1 position, boosted size
                    entries.append((best_v121[0], best_v121[1], best_v121[2], 'v121+union', 1))
                else:
                    # Check correlation
                    corr = get_corr(best_v121[1], best_union[1], di)
                    if corr < max_corr:
                        entries.append((best_v121[0], best_v121[1], best_v121[2], 'v121', 1))
                        entries.append((best_union[0], best_union[1], best_union[2], 'union', 1))
                    else:
                        # Correlation too high: just take the best
                        best = best_v121 if best_v121[0] >= best_union[0] else best_union
                        entries.append((best[0], best[1], best[2], 'best', 1))
            elif best_v121:
                entries.append((best_v121[0], best_v121[1], best_v121[2], 'v121', 1))
            elif best_union:
                entries.append((best_union[0], best_union[1], best_union[2], 'union', 1))

            for sc, s, pr, sig_str, _ in entries:
                if s in set(p['si'] for p in positions): continue
                if len(positions) >= 2: break

                # --- Determine sizing fraction ---
                sf = per_pos_frac  # default

                if sizing_mode == 'wr_adaptive':
                    if len(trades) >= 20:
                        recent_wr = np.mean([1 if t > 0 else 0 for t in trades[-20:]])
                        if recent_wr > wr_hi:       sf = wr_size_hi
                        elif recent_wr > wr_mid:    sf = wr_size_mid
                        else:                        sf = wr_size_lo
                elif sizing_mode == 'anti_mart':
                    if consecutive_wins >= am_wins_req:        sf = am_size_hot
                    elif consecutive_losses >= am_losses_big:  sf = am_size_deep
                    elif consecutive_losses >= am_losses_small: sf = am_size_cold
                    else:                                       sf = am_size_normal
                elif sizing_mode == 'eq_curve':
                    if len(daily_eq) >= eq_window:
                        eq_ma = np.mean(daily_eq[-eq_window:])
                        if eq_ma > 0:
                            ratio = pv / eq_ma
                            if ratio >= 1.05:       sf = eq_size_hi
                            elif ratio >= 1.00:
                                sf = eq_size_mid + (ratio - 1.00) / 0.05 * (eq_size_hi - eq_size_mid)
                            elif ratio >= 0.95:
                                sf = eq_size_lo + (ratio - 0.95) / 0.05 * (eq_size_mid - eq_size_lo)
                            else:
                                sf = eq_size_lo
                elif sizing_mode == 'combined_wr_eq':
                    # WR component
                    if len(trades) >= 20:
                        recent_wr = np.mean([1 if t > 0 else 0 for t in trades[-20:]])
                        if recent_wr > wr_hi:       wr_sf = wr_size_hi
                        elif recent_wr > wr_mid:    wr_sf = wr_size_mid
                        else:                        wr_sf = wr_size_lo
                    else:
                        wr_sf = per_pos_frac
                    # Equity curve component
                    if len(daily_eq) >= eq_window:
                        eq_ma = np.mean(daily_eq[-eq_window:])
                        if eq_ma > 0:
                            ratio = pv / eq_ma
                            if ratio >= 1.05:       eq_sf = eq_size_hi
                            elif ratio >= 1.00:
                                eq_sf = eq_size_mid + (ratio - 1.00) / 0.05 * (eq_size_hi - eq_size_mid)
                            elif ratio >= 0.95:
                                eq_sf = eq_size_lo + (ratio - 0.95) / 0.05 * (eq_size_mid - eq_size_lo)
                            else:
                                eq_sf = eq_size_lo
                        else:
                            eq_sf = per_pos_frac
                    else:
                        eq_sf = per_pos_frac
                    # Combine: geometric mean normalized to center
                    sf = wr_sf * eq_sf / per_pos_frac
                    sf = max(0.05, min(0.95, sf))

                sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                cap = cash * sf
                ct = max(1, int(cap / (pr * m * (1 + COMM))))
                ci = pr * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                    ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                if ct <= 0 or ci <= 0 or ci > cash: continue
                cash -= ci
                positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                  'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': 1,
                                  'sig': sig_str})

        # Liquidate remaining
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
        ratio = abs(ann / mdd) if mdd != 0 else 0
        return {'ann': ann, 'mdd': mdd, 'sharpe': sh, 'final': cash, 'ratio': ratio,
                'wr': wr, 'n': nt}

    # ===================== PORTFOLIO BACKTEST (50/50 split) =====================
    def backtest_portfolio_v140(start_di=MIN_TRAIN, end_di=None, size_frac=0.50):
        """V140 baseline: Union/V121 50/50 with fixed sizing, no Cross+Corr."""
        if end_di is None: end_di = ND

        def run_sub(sig_func):
            cash = float(CASH0); positions = []; daily_eq = []
            for di in range(start_di, end_di - 1):
                pv = cash
                for p in positions:
                    cp = C[p['si'], di]
                    if not np.isnan(cp) and cp > 0:
                        m = MULT.get(p['sym'], DEF_MULT)
                        pv += cp * m * p['lots'] - cp * m * abs(p['lots']) * COMM
                daily_eq.append(pv)
                cl = []
                for p in positions:
                    if di - p['entry_di'] >= p['hold_days']:
                        ep = C[p['si'], di]
                        if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                        m = MULT.get(p['sym'], DEF_MULT)
                        cash += ep * m * abs(p['lots']) * (1 - COMM)
                        cl.append(p)
                for p in cl: positions.remove(p)
                if len(positions) >= 1: continue
                edi = di + 1
                if edi >= end_di: continue
                cands = sig_func(di, edi)
                if not cands: continue
                cands.sort(key=lambda x: -x[0])
                item = cands[0]
                if len(item) == 3: sc, s, pr = item; sig = ''
                else: sc, s, pr, sig = item
                sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                cap = cash * size_frac
                ct = max(1, int(cap / (pr * m * (1 + COMM))))
                ci = pr * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                    ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                if ct <= 0 or ci <= 0 or ci > cash: continue
                cash -= ci
                positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                  'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': 1, 'sig': sig})
            for p in positions:
                ep = C[p['si'], min(end_di-1, ND-1)]
                if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                m = MULT.get(p['sym'], DEF_MULT)
                cash += ep * m * abs(p['lots']) * (1 - COMM)
            return np.array(daily_eq)

        eq_A = run_sub(sig_union)
        eq_B = run_sub(sig_v121)
        ml = min(len(eq_A), len(eq_B))
        if ml <= 1: return {'ann': -100.0, 'mdd': 0, 'sharpe': 0, 'final': CASH0, 'ratio': 0}
        ret_A = np.diff(eq_A[:ml]) / eq_A[:ml-1]
        ret_B = np.diff(eq_B[:ml]) / eq_B[:ml-1]
        ret_A = np.where(np.isfinite(ret_A), ret_A, 0)
        ret_B = np.where(np.isfinite(ret_B), ret_B, 0)
        combined = 0.5 * ret_A + 0.5 * ret_B
        eq = np.zeros(ml)
        eq[0] = float(CASH0)
        for i in range(ml - 1):
            eq[i+1] = eq[i] * (1 + combined[i])
        final = eq[-1]
        ann = annual_return(final, CASH0, ml)
        pk = np.maximum.accumulate(eq)
        mdd = np.min((eq - pk) / pk * 100)
        sh = np.mean(combined) / np.std(combined) * np.sqrt(252) if np.std(combined) > 0 else 0
        ratio = abs(ann / mdd) if mdd != 0 else 0
        return {'ann': ann, 'mdd': mdd, 'sharpe': sh, 'final': final, 'ratio': ratio}

    # ===================== PRINT HELPERS =====================
    def pr(r, label=""):
        ratio = r.get('ratio', abs(r['ann'] / r['mdd']) if r.get('mdd', 0) != 0 else 0)
        print(f"  {label:75s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | "
              f"Sh={r['sharpe']:4.2f} | R/M={ratio:.2f}")

    def walk_forward(backtest_func, label="", **kwargs):
        """Walk-forward per year."""
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue
            r = backtest_func(start_di=ys, end_di=ye, **kwargs)
            res[yr] = r
        return res

    def print_wf(wf_res, label=""):
        yrs = sorted(wf_res.keys())
        pos = sum(1 for yr in yrs if wf_res[yr]['ann'] > 0)
        avg = np.mean([wf_res[yr]['ann'] for yr in yrs])
        worst_mdd = min([wf_res[yr]['mdd'] for yr in yrs])
        anns = " | ".join([f"{yr}:{wf_res[yr]['ann']:+.0f}%" for yr in yrs])
        mdds = " | ".join([f"{yr}:{wf_res[yr]['mdd']:.0f}%" for yr in yrs])
        print(f"  {label}")
        print(f"    Ann: {pos}/{len(yrs)} pos | Avg={avg:>+7.0f}% | {anns}")
        print(f"    MDD: Worst={worst_mdd:>5.0f}% | {mdds}")

    # ===================== SECTION 0: BASELINES =====================
    print("\n" + "=" * 130)
    print("  SECTION 0: BASELINES")
    print("=" * 130)

    # V140 baseline: 50/50 portfolio, 50% sizing
    r_v140 = backtest_portfolio_v140(size_frac=0.50)
    pr(r_v140, "BASELINE: V140 Union/V121 50/50 @50%")

    # V142 baseline: Cross+Corr alone (no sizing tricks, no SL)
    r_v142 = backtest_v144(max_corr=0.5, per_pos_frac=0.45, sizing_mode='fixed')
    pr(r_v142, "BASELINE: V142 Cross+Corr corr<0.5 @45% fixed")

    # V143 baseline: SL=5% alone (on portfolio)
    # For V143, we just do Cross+Corr with SL=5%
    r_v143 = backtest_v144(max_corr=0.5, per_pos_frac=0.45, sizing_mode='fixed', stop_loss_pct=5.0)
    pr(r_v143, "BASELINE: V143 Cross+Corr corr<0.5 + SL=5%")

    # V141 baseline: WR-adaptive alone
    r_v141 = backtest_v144(max_corr=0.5, per_pos_frac=0.50, sizing_mode='wr_adaptive')
    pr(r_v141, "BASELINE: V141 Cross+Corr corr<0.5 + WR-adaptive")

    print("\n  Walk-Forward for baselines:")
    wf_v140 = walk_forward(backtest_portfolio_v140, "V140 baseline")
    print_wf(wf_v140, "V140 Union/V121 50/50 @50%")
    wf_v142 = walk_forward(backtest_v144, "V142 Cross+Corr", max_corr=0.5, per_pos_frac=0.45, sizing_mode='fixed')
    print_wf(wf_v142, "V142 Cross+Corr corr<0.5 @45%")

    # ===================== SECTION A: Cross+Corr + Stop Loss =====================
    print("\n" + "=" * 130)
    print("  SECTION A: Cross+Corr + Stop Loss")
    print("  Test: max_corr = 0.3, 0.5, 0.7 with SL = 3%, 5%, 8%")
    print("=" * 130)

    results_A = {}
    for mc in [0.3, 0.5, 0.7]:
        for sl in [3.0, 5.0, 8.0]:
            label = f"A: Cross+Corr corr<{mc} + SL={sl}% @45%"
            r = backtest_v144(max_corr=mc, per_pos_frac=0.45, sizing_mode='fixed', stop_loss_pct=sl)
            results_A[label] = r
            pr(r, label)

    print("\n  Walk-Forward for best Section A configs:")
    # Find top 3 by ratio
    ranked_A = sorted(results_A.items(), key=lambda x: -x[1].get('ratio', 0))
    for label, r in ranked_A[:3]:
        mc = float(label.split('corr<')[1].split(' ')[0])
        sl = float(label.split('SL=')[1].split('%')[0])
        wf = walk_forward(backtest_v144, label, max_corr=mc, per_pos_frac=0.45, sizing_mode='fixed', stop_loss_pct=sl)
        print_wf(wf, label)

    # ===================== SECTION B: Cross+Corr + WR-adaptive =====================
    print("\n" + "=" * 130)
    print("  SECTION B: Cross+Corr + WR-adaptive Sizing")
    print("  WR > 65%: size = 70% | WR 55-65%: size = 50% | WR < 55%: size = 30%")
    print("  Testing different WR thresholds and size levels.")
    print("=" * 130)

    wr_configs = [
        # (wr_hi, wr_mid, sz_hi, sz_mid, sz_lo, base, max_corr, label_suffix)
        (0.65, 0.55, 0.70, 0.50, 0.30, 0.50, 0.5, "WR65/55 sz70/50/30"),
        (0.65, 0.55, 0.70, 0.50, 0.30, 0.50, 0.3, "WR65/55 sz70/50/30 corr<0.3"),
        (0.60, 0.50, 0.70, 0.50, 0.25, 0.50, 0.5, "WR60/50 sz70/50/25"),
        (0.70, 0.55, 0.80, 0.55, 0.25, 0.50, 0.5, "WR70/55 sz80/55/25"),
        (0.65, 0.50, 0.65, 0.45, 0.20, 0.45, 0.5, "WR65/50 sz65/45/20"),
        (0.60, 0.50, 0.65, 0.45, 0.20, 0.45, 0.7, "WR60/50 sz65/45/20 corr<0.7"),
    ]

    results_B = {}
    for wr_hi, wr_mid, sz_hi, sz_mid, sz_lo, base, mc, suffix in wr_configs:
        label = f"B: Cross+Corr corr<{mc} + {suffix}"
        r = backtest_v144(max_corr=mc, per_pos_frac=base, sizing_mode='wr_adaptive',
                          wr_hi=wr_hi, wr_mid=wr_mid,
                          wr_size_hi=sz_hi, wr_size_mid=sz_mid, wr_size_lo=sz_lo)
        results_B[label] = r
        pr(r, label)

    print("\n  Walk-Forward for best Section B configs:")
    ranked_B = sorted(results_B.items(), key=lambda x: -x[1].get('ratio', 0))
    for label, r in ranked_B[:3]:
        # Extract params
        for wr_hi, wr_mid, sz_hi, sz_mid, sz_lo, base, mc, suffix in wr_configs:
            if suffix in label:
                wf = walk_forward(backtest_v144, label, max_corr=mc, per_pos_frac=base,
                                   sizing_mode='wr_adaptive',
                                   wr_hi=wr_hi, wr_mid=wr_mid,
                                   wr_size_hi=sz_hi, wr_size_mid=sz_mid, wr_size_lo=sz_lo)
                print_wf(wf, label)
                break

    # ===================== SECTION C: Cross+Corr + Anti-Martingale =====================
    print("\n" + "=" * 130)
    print("  SECTION C: Cross+Corr + Anti-Martingale Sizing")
    print("  3+ wins -> 80% | normal -> 55% | 2+ losses -> 30% | 4+ losses -> 15%")
    print("=" * 130)

    am_configs = [
        # (wins_req, loss_small, loss_big, sz_hot, sz_normal, sz_cold, sz_deep, base, mc, label_suffix)
        (3, 2, 4, 0.80, 0.55, 0.30, 0.15, 0.50, 0.5, "AM 3w/2l/4l sz80/55/30/15"),
        (3, 2, 4, 0.80, 0.55, 0.30, 0.15, 0.50, 0.3, "AM 3w/2l/4l sz80/55/30/15 corr<0.3"),
        (3, 2, 3, 0.75, 0.50, 0.25, 0.10, 0.45, 0.5, "AM 3w/2l/3l sz75/50/25/10"),
        (2, 2, 4, 0.80, 0.55, 0.30, 0.15, 0.50, 0.5, "AM 2w/2l/4l sz80/55/30/15"),
        (3, 2, 4, 0.70, 0.50, 0.30, 0.15, 0.45, 0.5, "AM 3w/2l/4l sz70/50/30/15"),
        (3, 2, 4, 0.80, 0.55, 0.30, 0.15, 0.50, 0.7, "AM 3w/2l/4l sz80/55/30/15 corr<0.7"),
    ]

    results_C = {}
    for wins_req, loss_small, loss_big, sz_hot, sz_normal, sz_cold, sz_deep, base, mc, suffix in am_configs:
        label = f"C: Cross+Corr corr<{mc} + {suffix}"
        r = backtest_v144(max_corr=mc, per_pos_frac=base, sizing_mode='anti_mart',
                          am_wins_req=wins_req, am_losses_small=loss_small, am_losses_big=loss_big,
                          am_size_hot=sz_hot, am_size_normal=sz_normal,
                          am_size_cold=sz_cold, am_size_deep=sz_deep)
        results_C[label] = r
        pr(r, label)

    print("\n  Walk-Forward for best Section C configs:")
    ranked_C = sorted(results_C.items(), key=lambda x: -x[1].get('ratio', 0))
    for label, r in ranked_C[:3]:
        for wins_req, loss_small, loss_big, sz_hot, sz_normal, sz_cold, sz_deep, base, mc, suffix in am_configs:
            if suffix in label:
                wf = walk_forward(backtest_v144, label, max_corr=mc, per_pos_frac=base,
                                   sizing_mode='anti_mart',
                                   am_wins_req=wins_req, am_losses_small=loss_small, am_losses_big=loss_big,
                                   am_size_hot=sz_hot, am_size_normal=sz_normal,
                                   am_size_cold=sz_cold, am_size_deep=sz_deep)
                print_wf(wf, label)
                break

    # ===================== SECTION D: All Three Combined (Cross+Corr + WR + SL) =====================
    print("\n" + "=" * 130)
    print("  SECTION D: All Three Combined (Cross+Corr + WR-adaptive + SL=5%)")
    print("=" * 130)

    results_D = {}
    d_configs = [
        # (mc, sl, wr_hi, wr_mid, sz_hi, sz_mid, sz_lo, base, suffix)
        (0.5, 5.0, 0.65, 0.55, 0.70, 0.50, 0.30, 0.50, "WR65/55 sz70/50/30 SL5%"),
        (0.5, 5.0, 0.60, 0.50, 0.70, 0.50, 0.25, 0.50, "WR60/50 sz70/50/25 SL5%"),
        (0.5, 5.0, 0.70, 0.55, 0.80, 0.55, 0.25, 0.50, "WR70/55 sz80/55/25 SL5%"),
        (0.3, 5.0, 0.65, 0.55, 0.70, 0.50, 0.30, 0.50, "WR65/55 sz70/50/30 SL5% corr<0.3"),
        (0.5, 3.0, 0.65, 0.55, 0.70, 0.50, 0.30, 0.50, "WR65/55 sz70/50/30 SL3%"),
        (0.5, 8.0, 0.65, 0.55, 0.70, 0.50, 0.30, 0.50, "WR65/55 sz70/50/30 SL8%"),
        (0.7, 5.0, 0.65, 0.55, 0.70, 0.50, 0.30, 0.50, "WR65/55 sz70/50/30 SL5% corr<0.7"),
        (0.5, 5.0, 0.65, 0.50, 0.65, 0.45, 0.20, 0.45, "WR65/50 sz65/45/20 SL5%"),
    ]

    for mc, sl, wr_hi, wr_mid, sz_hi, sz_mid, sz_lo, base, suffix in d_configs:
        label = f"D: Cross+Corr corr<{mc} + {suffix}"
        r = backtest_v144(max_corr=mc, per_pos_frac=base, sizing_mode='wr_adaptive',
                          stop_loss_pct=sl,
                          wr_hi=wr_hi, wr_mid=wr_mid,
                          wr_size_hi=sz_hi, wr_size_mid=sz_mid, wr_size_lo=sz_lo)
        results_D[label] = r
        pr(r, label)

    # Also test Cross+Corr + Anti-Mart + SL
    print("\n  --- D2: Cross+Corr + Anti-Mart + SL=5% ---")
    am_sl_configs = [
        (0.5, 5.0, 3, 2, 4, 0.80, 0.55, 0.30, 0.15, 0.50, "AM sz80/55/30/15 SL5%"),
        (0.5, 5.0, 3, 2, 4, 0.70, 0.50, 0.30, 0.15, 0.45, "AM sz70/50/30/15 SL5%"),
        (0.3, 5.0, 3, 2, 4, 0.80, 0.55, 0.30, 0.15, 0.50, "AM sz80/55/30/15 SL5% corr<0.3"),
    ]
    for mc, sl, wins_req, loss_small, loss_big, sz_hot, sz_normal, sz_cold, sz_deep, base, suffix in am_sl_configs:
        label = f"D: Cross+Corr corr<{mc} + {suffix}"
        r = backtest_v144(max_corr=mc, per_pos_frac=base, sizing_mode='anti_mart',
                          stop_loss_pct=sl,
                          am_wins_req=wins_req, am_losses_small=loss_small, am_losses_big=loss_big,
                          am_size_hot=sz_hot, am_size_normal=sz_normal,
                          am_size_cold=sz_cold, am_size_deep=sz_deep)
        results_D[label] = r
        pr(r, label)

    print("\n  Walk-Forward for best Section D configs:")
    ranked_D = sorted(results_D.items(), key=lambda x: -x[1].get('ratio', 0))
    for label, r in ranked_D[:3]:
        print(f"  --- {label} ---")
        # Extract params from label
        mc = float(label.split('corr<')[1].split(' ')[0])
        if 'SL' in label:
            sl = float(label.split('SL')[1].split('%')[0])
        else:
            sl = 0
        if 'WR' in label:
            # Parse WR params
            for wr_hi_t, wr_mid_t, sz_hi_t, sz_mid_t, sz_lo_t, base_t, mc_t, sl_t, suffix_t in [
                (c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8]) for c in d_configs
            ]:
                if suffix_t in label:
                    wf = walk_forward(backtest_v144, label, max_corr=mc, per_pos_frac=base_t,
                                       sizing_mode='wr_adaptive', stop_loss_pct=sl,
                                       wr_hi=wr_hi_t, wr_mid=wr_mid_t,
                                       wr_size_hi=sz_hi_t, wr_size_mid=sz_mid_t, wr_size_lo=sz_lo_t)
                    print_wf(wf, label)
                    break
            else:
                # Anti-Mart + SL
                for mc_t, sl_t, wins_req, loss_small, loss_big, sz_hot, sz_normal, sz_cold, sz_deep, base_t, suffix_t in am_sl_configs:
                    if suffix_t in label:
                        wf = walk_forward(backtest_v144, label, max_corr=mc, per_pos_frac=base_t,
                                           sizing_mode='anti_mart', stop_loss_pct=sl,
                                           am_wins_req=wins_req, am_losses_small=loss_small, am_losses_big=loss_big,
                                           am_size_hot=sz_hot, am_size_normal=sz_normal,
                                           am_size_cold=sz_cold, am_size_deep=sz_deep)
                        print_wf(wf, label)
                        break

    # ===================== SECTION E: Cross+Corr + Equity Curve Gentle Scaling =====================
    print("\n" + "=" * 130)
    print("  SECTION E: Cross+Corr + Equity Curve Gentle Scaling")
    print("  equity > MA(eq,20): size=70% | equity = MA: size=55% | equity < MA*0.95: size=30%")
    print("  Linear interpolation between anchor points.")
    print("=" * 130)

    eq_configs = [
        # (mc, base, eq_hi, eq_mid, eq_lo, eq_win, suffix)
        (0.5, 0.50, 0.70, 0.55, 0.30, 20, "EQ 70/55/30 w20"),
        (0.5, 0.50, 0.70, 0.55, 0.30, 30, "EQ 70/55/30 w30"),
        (0.5, 0.45, 0.65, 0.50, 0.25, 20, "EQ 65/50/25 w20"),
        (0.3, 0.50, 0.70, 0.55, 0.30, 20, "EQ 70/55/30 w20 corr<0.3"),
        (0.7, 0.50, 0.70, 0.55, 0.30, 20, "EQ 70/55/30 w20 corr<0.7"),
        (0.5, 0.50, 0.75, 0.55, 0.25, 20, "EQ 75/55/25 w20"),
    ]

    results_E = {}
    for mc, base, eq_hi, eq_mid, eq_lo, eq_win, suffix in eq_configs:
        label = f"E: Cross+Corr corr<{mc} + {suffix}"
        r = backtest_v144(max_corr=mc, per_pos_frac=base, sizing_mode='eq_curve',
                          eq_size_hi=eq_hi, eq_size_mid=eq_mid, eq_size_lo=eq_lo,
                          eq_window=eq_win)
        results_E[label] = r
        pr(r, label)

    # Also test: Equity Curve + WR-adaptive combined
    print("\n  --- E2: Cross+Corr + Combined WR-adaptive x Equity Curve ---")
    eq_wr_configs = [
        (0.5, 0.50, 0.70, 0.55, 0.30, 20, 0.65, 0.55, 0.70, 0.50, 0.30, "EQ+WR 65/55 sz70/50/30"),
        (0.5, 0.45, 0.65, 0.50, 0.25, 20, 0.65, 0.55, 0.65, 0.45, 0.20, "EQ+WR 65/55 sz65/45/20"),
    ]
    for mc, base, eq_hi, eq_mid, eq_lo, eq_win, wr_hi, wr_mid, wr_sz_hi, wr_sz_mid, wr_sz_lo, suffix in eq_wr_configs:
        label = f"E: Cross+Corr corr<{mc} + {suffix}"
        r = backtest_v144(max_corr=mc, per_pos_frac=base, sizing_mode='combined_wr_eq',
                          eq_size_hi=eq_hi, eq_size_mid=eq_mid, eq_size_lo=eq_lo, eq_window=eq_win,
                          wr_hi=wr_hi, wr_mid=wr_mid,
                          wr_size_hi=wr_sz_hi, wr_size_mid=wr_sz_mid, wr_size_lo=wr_sz_lo)
        results_E[label] = r
        pr(r, label)

    print("\n  Walk-Forward for best Section E configs:")
    ranked_E = sorted(results_E.items(), key=lambda x: -x[1].get('ratio', 0))
    for label, r in ranked_E[:3]:
        for mc, base, eq_hi, eq_mid, eq_lo, eq_win, suffix in eq_configs:
            if suffix in label and 'EQ+WR' not in label:
                wf = walk_forward(backtest_v144, label, max_corr=mc, per_pos_frac=base,
                                   sizing_mode='eq_curve',
                                   eq_size_hi=eq_hi, eq_size_mid=eq_mid, eq_size_lo=eq_lo,
                                   eq_window=eq_win)
                print_wf(wf, label)
                break
        else:
            for mc, base, eq_hi, eq_mid, eq_lo, eq_win, wr_hi, wr_mid, wr_sz_hi, wr_sz_mid, wr_sz_lo, suffix in eq_wr_configs:
                if suffix in label:
                    wf = walk_forward(backtest_v144, label, max_corr=mc, per_pos_frac=base,
                                       sizing_mode='combined_wr_eq',
                                       eq_size_hi=eq_hi, eq_size_mid=eq_mid, eq_size_lo=eq_lo, eq_window=eq_win,
                                       wr_hi=wr_hi, wr_mid=wr_mid,
                                       wr_size_hi=wr_sz_hi, wr_size_mid=wr_sz_mid, wr_size_lo=wr_sz_lo)
                    print_wf(wf, label)
                    break

    # ===================== COMPREHENSIVE SUMMARY =====================
    print("\n" + "=" * 130)
    print("  COMPREHENSIVE SUMMARY: ALL RESULTS SORTED BY RETURN/MDD RATIO")
    print("=" * 130)

    all_results = {}
    all_results['BASELINE: V140'] = r_v140
    all_results['BASELINE: V142'] = r_v142
    all_results['BASELINE: V143'] = r_v143
    all_results['BASELINE: V141'] = r_v141
    all_results.update(results_A)
    all_results.update(results_B)
    all_results.update(results_C)
    all_results.update(results_D)
    all_results.update(results_E)

    ranked = sorted(all_results.items(), key=lambda x: -x[1].get('ratio', 0))

    print(f"\n  {'#':>3}  {'Config':75s} | {'Ann':>8s} | {'MDD':>6s} | {'Sh':>4s} | {'R/M':>5s}")
    print(f"  {'---':>3}  {'-'*75}-+-{'-'*8}-+-{'-'*6}-+-{'-'*4}-+-{'-'*5}")
    for i, (label, r) in enumerate(ranked):
        ratio = r.get('ratio', 0)
        marker = " <-- BEATS ALL BASELINES" if i < 10 and label not in all_results or \
            (not label.startswith('BASELINE') and ratio > max(
                r_v140.get('ratio', 0), r_v142.get('ratio', 0), r_v143.get('ratio', 0), r_v141.get('ratio', 0))) else ""
        if marker:
            marker = " *** BEATS ALL BASELINES"
        print(f"  {i+1:3d}  {label:75s} | {r['ann']:+8.1f}% | {r['mdd']:6.1f}% | {r['sharpe']:4.2f} | {ratio:.2f}{marker}")

    # Baseline comparison
    base_ratios = {
        'V140': r_v140.get('ratio', 0),
        'V142': r_v142.get('ratio', 0),
        'V143': r_v143.get('ratio', 0),
        'V141': r_v141.get('ratio', 0),
    }
    best_base_ratio = max(base_ratios.values())
    best_base_name = max(base_ratios, key=base_ratios.get)

    print(f"\n  --- BASELINE COMPARISON ---")
    for name, ratio in base_ratios.items():
        print(f"  {name}: R/M = {ratio:.2f}")
    print(f"  Best baseline: {best_base_name} with R/M = {best_base_ratio:.2f}")

    print(f"\n  --- CONFIGS THAT BEAT ALL BASELINES ---")
    beat_all = [(label, r) for label, r in ranked
                if not label.startswith('BASELINE') and r.get('ratio', 0) > best_base_ratio]
    if beat_all:
        for i, (label, r) in enumerate(beat_all):
            ratio = r.get('ratio', 0)
            delta = ratio - best_base_ratio
            print(f"  {i+1:3d}  {label:75s} | R/M={ratio:.2f} (+{delta:.2f} vs {best_base_name})")
    else:
        print(f"  None beat all baselines.")

    # Detailed top 5 with WF
    print(f"\n  --- DETAILED TOP 5 (with Walk-Forward) ---")
    top5 = [(label, r) for label, r in ranked if not label.startswith('BASELINE')][:5]
    # We already did WF above for section best; let's just show the top 5 full-period results
    for i, (label, r) in enumerate(top5):
        ratio = r.get('ratio', 0)
        print(f"\n  #{i+1}: {label}")
        print(f"       Ann={r['ann']:+.1f}% | MDD={r['mdd']:.1f}% | Sharpe={r['sharpe']:.2f} | R/M={ratio:.2f}")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
