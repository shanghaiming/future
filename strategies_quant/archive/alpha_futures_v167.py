"""
Alpha Futures V167 — Vol Filter + Equity Curve Momentum Sizing
==============================================================================
V164 showed atr_norm<10% + max_corr=0.7 (no wr_mult) gives +222%/-17% WF,
R/M=12.71 (best risk-adjusted ever).

V167 combines the vol filter (controls WHICH trades to take) with equity curve
momentum sizing (controls HOW MUCH to size based on recent performance).
These are orthogonal dimensions of risk management.

Tests:
1. V164 baseline reproduction: atr_norm<10%, max_corr=0.7, no wr_mult
2. Add wr_mult on top of vol filter
3. Add equity curve SMA cross: SMA(20) > SMA(50) -> mult=1.0, else 0.5
4. Add loss streak protection: after 3 consecutive losses -> 0.5x for 5 trades
5. Add drawdown recovery sizing: in first 50% of DD recovery -> 1.3x
6. Best combination

Kitchen Sink formula variants:
- Base:  pos_size = dd_sz * regime_mult
- +WR:   pos_size = dd_sz * wr_mult * regime_mult
- +EQ:   pos_size = dd_sz * eq_mult * regime_mult
- +All:  pos_size = dd_sz * wr_mult * eq_mult * regime_mult

Parameters: atr_norm_max in [8,10,12], max_corr=0.7, top_n=3, aggro100 DD.
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
    print("  V167 — Vol Filter + Equity Curve Momentum Sizing")
    print("  Core idea: vol filter controls WHICH trades, eq curve controls HOW MUCH")
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

    # ATR normalized by close price (individual commodity vol measure)
    ATR_NORM = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            atr = ATR14[si, di]
            cp = C[si, di]
            if not np.isnan(atr) and not np.isnan(cp) and cp > 0:
                ATR_NORM[si, di] = atr / cp * 100  # as percentage

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

    print(f"  Market vol median: {VOL_MEDIAN:.4f}%")
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

    # ===================== EQUITY CURVE MOMENTUM METHODS =====================
    def eq_sma_cross(daily_eq, sma_fast=20, sma_slow=50):
        """Equity curve SMA cross: if SMA(fast) > SMA(slow) -> 1.0, else 0.5."""
        if len(daily_eq) < sma_slow:
            return 1.0
        recent = daily_eq[-sma_slow:]
        fast_ma = np.mean(recent[-sma_fast:])
        slow_ma = np.mean(recent)
        return 1.0 if fast_ma > slow_ma else 0.5

    def eq_streak(trades, streak_threshold=3, cooldown=5):
        """Loss streak protection: after N consecutive losses, reduce size for next M trades.
        Returns (multiplier, remaining_cooldown_trades)."""
        if len(trades) < streak_threshold:
            return 1.0, 0
        # Check if we need to be in cooldown
        # Count consecutive losses at end of trade history
        consec_losses = 0
        for t in reversed(trades):
            if t <= 0:
                consec_losses += 1
            else:
                break
        if consec_losses >= streak_threshold:
            return 0.5, cooldown
        return 1.0, 0

    def eq_recovery(daily_eq, high_water, recovery_boost=1.3):
        """Drawdown recovery sizing: in first 50% of DD recovery, use boosted size.
        Returns multiplier."""
        if high_water <= 0 or len(daily_eq) == 0:
            return 1.0
        cur_eq = daily_eq[-1]
        # Find the DD bottom (lowest point since HWM)
        if len(daily_eq) >= 2:
            # Look back from HWM achievement to find the bottom
            cur_dd = (cur_eq - high_water) / high_water
            if cur_dd >= 0:
                # At or above HWM, no recovery boost
                return 1.0
            # Find the lowest point in recent history
            lookback = min(len(daily_eq), 60)
            recent_eq = daily_eq[-lookback:]
            eq_min = min(recent_eq)
            if eq_min >= high_water:
                return 1.0
            # Recovery progress: how far from bottom to HWM
            total_range = high_water - eq_min
            if total_range <= 0:
                return 1.0
            progress = (cur_eq - eq_min) / total_range
            # In first 50% of recovery -> boost
            if progress < 0.5:
                return recovery_boost
        return 1.0

    # ===================== BACKTEST ENGINE =====================
    # eq_method controls how equity curve momentum sizing is applied:
    #   'none'     : no eq curve sizing, no wr_mult -> pos_size = dd_sz * regime_mult
    #   'wr'       : WR-adaptive sizing -> pos_size = dd_sz * wr_mult * regime_mult
    #   'sma_cross': equity SMA cross -> pos_size = dd_sz * eq_mult * regime_mult
    #   'streak'   : loss streak protection -> pos_size = dd_sz * eq_mult * regime_mult
    #   'recovery' : DD recovery boost -> pos_size = dd_sz * eq_mult * regime_mult
    #   'wr+sma'   : WR + SMA cross -> pos_size = dd_sz * wr_mult * eq_mult * regime_mult
    #   'wr+streak': WR + streak -> pos_size = dd_sz * wr_mult * eq_mult * regime_mult
    #                streak also tracks cooldown across trades

    def backtest(start_di=MIN_TRAIN, end_di=None,
                 atr_norm_max=10.0, max_corr=0.7,
                 dd_tiers=None,
                 regime_lo=0.5, regime_hi=1.5,
                 sl_pct=0.0, hold=1, top_n=3,
                 eq_method='none'):
        if end_di is None: end_di = ND
        if dd_tiers is None:
            dd_tiers = [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)]

        cash = float(CASH0)
        positions = []
        trades = []
        daily_eq = []
        high_water = float(CASH0)

        # Streak cooldown state
        streak_cooldown_remaining = 0

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

            # --- Stop-loss check ---
            if sl_pct > 0:
                cl_early = []
                for p in positions:
                    cp = C[p['si'], di]
                    if np.isnan(cp) or cp <= 0: continue
                    m = MULT.get(p['sym'], DEF_MULT)
                    unrealized = (cp - p['entry_price']) * m * p['lots']
                    invested = p['entry_price'] * m * abs(p['lots'])
                    if invested > 0:
                        loss_pct = unrealized / invested
                        if loss_pct < -sl_pct:
                            cash += cp * m * abs(p['lots']) * (1 - COMM)
                            pnl_pct = unrealized / invested * 100
                            trades.append(pnl_pct)
                            cl_early.append(p)
                            # Update streak cooldown
                            if eq_method in ('streak', 'wr+streak'):
                                _, streak_cooldown_remaining = eq_streak(trades)
                for p in cl_early: positions.remove(p)

            # Close positions past hold period
            cl = []
            for p in positions:
                if di - p['entry_di'] >= p['hold_days']:
                    ep = C[p['si'], di]
                    if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                    m = MULT.get(p['sym'], DEF_MULT)
                    pnl = (ep - p['entry_price']) * m * p['lots']
                    inv = p['entry_price'] * m * abs(p['lots'])
                    pp = pnl / inv * 100 if inv > 0 else 0
                    cash += ep * m * abs(p['lots']) * (1 - COMM)
                    trades.append(pp)
                    cl.append(p)
                    # Update streak cooldown
                    if eq_method in ('streak', 'wr+streak'):
                        _, streak_cooldown_remaining = eq_streak(trades)
            for p in cl: positions.remove(p)

            # --- Kitchen Sink sizing ---
            dd_sz = dd_size(pv, high_water, dd_tiers)

            # WR adaptive sizing
            use_wr = eq_method in ('wr', 'wr+sma', 'wr+streak')
            wr_mult_val = wr_size(trades, window=20) if use_wr else 1.0

            # Equity curve momentum sizing
            eq_mult_val = 1.0
            if eq_method == 'sma_cross':
                eq_mult_val = eq_sma_cross(daily_eq, sma_fast=20, sma_slow=50)
            elif eq_method == 'streak':
                if streak_cooldown_remaining > 0:
                    eq_mult_val = 0.5
                    streak_cooldown_remaining -= 1
                else:
                    eq_mult_val, streak_cooldown_remaining = eq_streak(trades)
            elif eq_method == 'recovery':
                eq_mult_val = eq_recovery(daily_eq, high_water, recovery_boost=1.3)
            elif eq_method == 'wr+sma':
                eq_mult_val = eq_sma_cross(daily_eq, sma_fast=20, sma_slow=50)
            elif eq_method == 'wr+streak':
                if streak_cooldown_remaining > 0:
                    eq_mult_val = 0.5
                    streak_cooldown_remaining -= 1
                else:
                    eq_mult_val, streak_cooldown_remaining = eq_streak(trades)

            composite = compute_composite(di, daily_eq, high_water)
            regime_mult = regime_lo + composite * (regime_hi - regime_lo)

            # Final pos_size formula
            # Base:  dd_sz * regime_mult
            # +WR:   dd_sz * wr_mult * regime_mult
            # +EQ:   dd_sz * eq_mult * regime_mult
            # +All:  dd_sz * wr_mult * eq_mult * regime_mult
            if use_wr and eq_method not in ('none',):
                # Both WR and EQ curve active
                pos_size = dd_sz * wr_mult_val * eq_mult_val * regime_mult
            elif use_wr:
                pos_size = dd_sz * wr_mult_val * regime_mult
            elif eq_method != 'none':
                pos_size = dd_sz * eq_mult_val * regime_mult
            else:
                pos_size = dd_sz * regime_mult

            pos_size = max(0.05, min(0.99, pos_size))

            # --- Enter positions ---
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue

            held_si = set(p['si'] for p in positions)

            # Get best V121 and best Union signal
            cands_v121 = sig_v121(di, edi)
            cands_union = sig_union(di, edi)

            # Apply ATR norm vol filter BEFORE selecting best
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
                                  'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': hold,
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
        print(f"  {label:95s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d}")

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

    EQ_METHODS = ['none', 'wr', 'sma_cross', 'streak', 'recovery', 'wr+sma', 'wr+streak']
    ATR_NORMS = [8.0, 10.0, 12.0]

    # ===================== SECTION 0: V164 BASELINE REPRODUCTION =====================
    print("\n" + "=" * 130)
    print("  SECTION 0: V164 BASELINE REPRODUCTION")
    print("  atr_norm<10%, max_corr=0.7, NO wr_mult, aggro100 DD, top_n=3, noSL")
    print("=" * 130)

    # V164 baseline: vol filter but no equity curve sizing, no wr_mult
    r_baseline = backtest(atr_norm_max=10.0, max_corr=0.7,
                          dd_tiers=DD_AGGR100, eq_method='none',
                          regime_lo=0.5, regime_hi=1.5, sl_pct=0.0,
                          hold=1, top_n=3)
    pr(r_baseline, "V164 BASELINE: atr<10%, corr=0.7, no wr_mult, aggro100")

    # Also test with no vol filter for reference
    r_novol = backtest(atr_norm_max=999.0, max_corr=0.7,
                       dd_tiers=DD_AGGR100, eq_method='none',
                       regime_lo=0.5, regime_hi=1.5, sl_pct=0.0,
                       hold=1, top_n=3)
    pr(r_novol, "NO VOL: atr=off, corr=0.7, no wr_mult, aggro100")

    # ===================== SECTION 1: INDIVIDUAL EQ CURVE METHODS =====================
    print("\n" + "=" * 130)
    print("  SECTION 1: EQUITY CURVE METHODS — Full period, atr_norm_max=10%, max_corr=0.7")
    print("  Testing each eq_method independently (no wr_mult unless specified)")
    print("=" * 130)

    s1_results = []
    for eq_m in EQ_METHODS:
        for an_max in ATR_NORMS:
            label = f"eq={eq_m:10s} atr<{an_max:.0f}%"
            r = backtest(atr_norm_max=an_max, max_corr=0.7,
                         dd_tiers=DD_AGGR100, eq_method=eq_m,
                         regime_lo=0.5, regime_hi=1.5, sl_pct=0.0,
                         hold=1, top_n=3)
            r['label'] = label
            r['eq_method'] = eq_m
            r['atr_norm_max'] = an_max
            s1_results.append(r)
            pr(r, label)

    # ===================== SECTION 2: RANKED BY ANNUAL RETURN =====================
    print("\n" + "=" * 130)
    print("  SECTION 2: RANKED BY ANNUAL RETURN (full period)")
    print("=" * 130)

    s1_results.sort(key=lambda x: -x['ann'])
    for i, r in enumerate(s1_results[:20]):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i+1:2d} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d} | {r['label']}")

    # ===================== SECTION 3: RANKED BY R/M RATIO =====================
    print("\n" + "=" * 130)
    print("  SECTION 3: RANKED BY R/M RATIO (risk-adjusted, full period)")
    print("=" * 130)

    s1_rm = sorted(s1_results, key=lambda x: -abs(x['ann']/x['mdd']) if x['mdd'] != 0 else 0)
    for i, r in enumerate(s1_rm[:20]):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i+1:2d} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d} | {r['label']}")

    # ===================== SECTION 4: WALK-FORWARD — TOP 15 BY R/M =====================
    print("\n" + "=" * 130)
    print("  SECTION 4: WALK-FORWARD VALIDATION — Top 15 by R/M")
    print("=" * 130)

    seen = set()
    wf_candidates = []
    for r in s1_rm:
        lbl = r['label']
        if lbl not in seen:
            seen.add(lbl)
            wf_candidates.append(r)

    wf_all = {}
    for r in wf_candidates[:15]:
        lbl = r['label']
        wf_res = walk_forward(label=lbl,
                              atr_norm_max=r['atr_norm_max'], max_corr=0.7,
                              dd_tiers=DD_AGGR100, eq_method=r['eq_method'],
                              regime_lo=0.5, regime_hi=1.5, sl_pct=0.0,
                              hold=1, top_n=3)
        wf_all[lbl] = (wf_res, r)
        print_wf(wf_res, lbl)

    # Also WF for baseline (no vol filter, no eq method)
    baseline_lbl = "V164 BASELINE (reproduced)"
    wf_baseline = walk_forward(label=baseline_lbl,
                               atr_norm_max=10.0, max_corr=0.7,
                               dd_tiers=DD_AGGR100, eq_method='none',
                               regime_lo=0.5, regime_hi=1.5, sl_pct=0.0,
                               hold=1, top_n=3)
    wf_all[baseline_lbl] = (wf_baseline, {'label': baseline_lbl, 'eq_method': 'none', 'atr_norm_max': 10.0})
    print_wf(wf_baseline, baseline_lbl + " [BASELINE]")

    # ===================== SECTION 5: WF COMPARISON TABLE =====================
    print("\n" + "=" * 130)
    print("  SECTION 5: WF COMPARISON TABLE — All configs ranked by WF avg")
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

    # Sort by WF avg
    wf_ranked.sort(key=lambda x: -x[0])
    print(f"\n  Ranked by WF Average Annual Return:")
    print(f"  {'#':>3s}  {'WF AVG':>8s} {'WorstDD':>8s} {'R/M':>6s} {'Pos':>4s} {'N':>5s} {'WR':>5s}  {'Label'}")
    print(f"  {'-'*100}")
    for i, (avg_ann, wmdd, bann, pos, tn, awr, ratio, lbl, wf_res, ri) in enumerate(wf_ranked):
        print(f"  {i+1:3d}  {avg_ann:>+8.0f}% {wmdd:>7.0f}% {ratio:6.2f} {pos:4d}/6 {tn:5d} {awr:5.1f}%  {lbl}")

    ws_line = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                          for yr, r in sorted(wf_baseline.items())])
    print(f"\n  BASELINE WF detail: {ws_line}")

    # ===================== SECTION 6: BEST RISK-ADJUSTED (WF) =====================
    print("\n" + "=" * 130)
    print("  SECTION 6: BEST RISK-ADJUSTED (WF R/M)")
    print("=" * 130)

    wf_ra = sorted(wf_ranked, key=lambda x: -x[6])  # sort by R/M
    for i, (avg_ann, wmdd, bann, pos, tn, awr, ratio, lbl, wf_res, ri) in enumerate(wf_ra[:10]):
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  #{i+1} WF AVG={avg_ann:+.0f}% | WorstMDD={wmdd:.0f}% | R/M={ratio:.2f} | {pos}/6 pos | {lbl}")
        print(f"     {ws}")

    # ===================== SECTION 7: EQ METHOD HEAD-TO-HEAD =====================
    print("\n" + "=" * 130)
    print("  SECTION 7: EQ METHOD HEAD-TO-HEAD (best atr_norm per method)")
    print("=" * 130)

    for method in EQ_METHODS:
        method_results = [(avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri)
                          for avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri in wf_ranked
                          if ri.get('eq_method') == method]
        if not method_results:
            print(f"\n  {method:12s}: no WF results")
            continue
        method_results.sort(key=lambda x: -x[6])  # best R/M
        best = method_results[0]
        avg_ann, wmdd, bann, pos, tn, awr, ratio, lbl, wf_res, ri = best
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  {method:12s} best: WF AVG={avg_ann:+.0f}% | WorstMDD={wmdd:.0f}% | R/M={ratio:.2f} | atr<{ri.get('atr_norm_max', '?')}")
        print(f"     {ws}")

    # ===================== SECTION 8: DELTA vs BASELINE =====================
    print("\n" + "=" * 130)
    print("  SECTION 8: IMPROVEMENT vs V164 BASELINE")
    print("=" * 130)

    b_avg = np.mean([r['ann'] for r in wf_baseline.values()])
    b_wmdd = min(r['mdd'] for r in wf_baseline.values())
    b_rm = abs(b_avg / b_wmdd) if b_wmdd != 0 else 0

    print(f"\n  V164 BASELINE: WF AVG={b_avg:+.0f}% | WorstMDD={b_wmdd:.0f}% | R/M={b_rm:.2f}")

    deltas = []
    for avg_ann, wmdd, bann, pos, tn, awr, ratio, lbl, wf_res, ri in wf_ranked:
        if ri.get('eq_method') == 'none' and ri.get('atr_norm_max') == 10.0:
            continue  # skip baseline itself
        delta_ann = avg_ann - b_avg
        delta_rm = ratio - b_rm
        deltas.append((delta_ann, delta_rm, ratio, avg_ann, wmdd, pos, lbl, wf_res))

    # Sort by R/M improvement
    deltas.sort(key=lambda x: -x[1])
    print(f"\n  Configs by R/M improvement over baseline:")
    for i, (da, drm, ratio, avg, wmdd, pos, lbl, wfr) in enumerate(deltas[:15]):
        marker = "*** IMPROVED" if drm > 0 else "    worse"
        print(f"  {i+1:2d} | R/M={ratio:.2f} (delta={drm:+.2f}) | Ann delta={da:+.0f}% | {pos}/6 | {marker} | {lbl}")

    # ===================== SECTION 9: BEST COMBINATION DETAIL =====================
    print("\n" + "=" * 130)
    print("  SECTION 9: BEST COMBINATION — Full detail for top 3 by R/M")
    print("=" * 130)

    for i, (avg_ann, wmdd, bann, pos, tn, awr, ratio, lbl, wf_res, ri) in enumerate(wf_ra[:3]):
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  #{i+1}: {lbl}")
        print(f"       WF AVG={avg_ann:+.0f}% | WorstMDD={wmdd:.0f}% | R/M={ratio:.2f} | {pos}/6 pos | N={tn}")
        print(f"       {ws}")
        # Per-year detail
        for yr, r in sorted(wf_res.items()):
            print(f"         {yr}: Ann={r['ann']:+.1f}% | MDD={r['mdd']:.1f}% | WR={r['wr']:.1f}% | N={r['n']}")

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 130)
    print("  FINAL SUMMARY")
    print("=" * 130)

    print(f"\n  V164 Baseline (atr<10%, corr=0.7, no wr_mult):")
    print(f"    WF AVG={b_avg:+.0f}% | WorstMDD={b_wmdd:.0f}% | R/M={b_rm:.2f}")

    # Best vol-filter only (no eq curve)
    vol_only = [(avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri)
                for avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri in wf_ranked
                if ri.get('eq_method') == 'none']
    if vol_only:
        vol_only.sort(key=lambda x: -x[6])
        best_vol = vol_only[0]
        print(f"\n  Best vol-only (no eq curve):")
        print(f"    {best_vol[7]}: WF AVG={best_vol[0]:+.0f}% | R/M={best_vol[6]:.2f}")

    # Best with eq curve
    eq_only = [(avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri)
               for avg, wmdd, bann, pos, tn, awr, ratio, lbl, wfr, ri in wf_ranked
               if ri.get('eq_method') != 'none']
    if eq_only:
        eq_only.sort(key=lambda x: -x[6])
        best_eq = eq_only[0]
        print(f"\n  Best vol+eq curve:")
        print(f"    {best_eq[7]}: WF AVG={best_eq[0]:+.0f}% | R/M={best_eq[6]:.2f}")
        eq_delta = best_eq[6] - b_rm
        print(f"    R/M improvement over baseline: {eq_delta:+.2f}")

        # Show improvement direction
        if eq_delta > 0:
            print(f"    CONCLUSION: Equity curve sizing HELPS with vol filter")
        else:
            print(f"    CONCLUSION: Equity curve sizing does NOT improve on vol filter alone")

    # Best overall
    overall_best = wf_ra[0]
    print(f"\n  Best overall: {overall_best[7]}")
    print(f"    WF AVG={overall_best[0]:+.0f}% | WorstMDD={overall_best[1]:.0f}% | R/M={overall_best[6]:.2f}")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
