"""
Alpha Futures V162 — Equity Curve Momentum & Anti-Martingale Approaches
==============================================================================
V159 showed mild anti-Martingale (win*1.1/loss*0.9) gives +186%/-29%.
V162 tests more sophisticated equity-curve-based sizing to REPLACE wr_size().

Kitchen Sink becomes: pos_size = dd_sz * equity_curve_mult * regime_mult

Five equity curve methods (replaces wr_size):
  1. none      — equity_curve_mult = 1.0 (pure DD * regime)
  2. sma_cross — equity curve SMA(fast) vs SMA(slow): above=1.0, below=0.5
  3. slope     — rolling equity curve slope over window, mapped to [0.3, 1.5]
  4. recovery  — after DD > 10%, gradually increase size as equity recovers
  5. streak    — consecutive wins increase, consecutive losses decrease
  6. combined  — geometric mean of sma_cross + slope + recovery + streak

Base: top_n=3, aggro100 DD, no SL, regime 0.5-1.5, hold=1
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
    print("  V162 — Equity Curve Momentum & Anti-Martingale Approaches")
    print("  Kitchen Sink: dd_sz * equity_curve_mult * regime_mult")
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
            if not np.isnan(c[di]) and not np.isnan(c[di - 1]) and c[di - 1] > 0:
                RET[si, di] = (c[di] / c[di - 1] - 1) * 100
        ROC5[si] = talib.ROC(c, timeperiod=5)
        ROC10[si] = talib.ROC(c, timeperiod=10)
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

    # ===================== REGIME INDICATORS =====================
    print("  Computing regime indicators...", flush=True)

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
    print(f"  Market vol median: {VOL_MEDIAN:.4f}%")

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

    # ===================== HELPER: Composite regime =====================
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

    # ===================== HELPER: DD-based sizing =====================
    def dd_size(pv, high_water, tiers):
        if high_water <= 0:
            return tiers[0][1]
        dd = (pv - high_water) / high_water
        for dd_thresh, size_frac in tiers:
            if dd >= -dd_thresh:
                return size_frac
        return tiers[-1][1]

    # ===================== EQUITY CURVE SIZING METHODS =====================
    # These REPLACE wr_size(). They return equity_curve_mult in [0.3, 1.5].

    def eq_mult_none(daily_eq, high_water, trades, streak_state, **kwargs):
        """No equity curve adjustment."""
        return 1.0

    def eq_mult_sma_cross(daily_eq, high_water, trades, streak_state,
                          fast=20, slow=50, **kwargs):
        """SMA cross on equity curve. Above=1.0, Below=0.5."""
        if len(daily_eq) < slow:
            return 1.0
        eq_arr = np.array(daily_eq)
        fast_ma = np.mean(eq_arr[-fast:])
        slow_ma = np.mean(eq_arr[-slow:])
        if fast_ma >= slow_ma:
            return 1.0
        else:
            return 0.5

    def eq_mult_slope(daily_eq, high_water, trades, streak_state,
                      window=20, **kwargs):
        """Size proportional to rolling equity curve slope. Maps to [0.3, 1.5]."""
        if len(daily_eq) < window:
            return 1.0
        eq_arr = np.array(daily_eq[-window:])
        x = np.arange(window)
        try:
            slope = np.polyfit(x, eq_arr, 1)[0]
            eq_mean = np.mean(eq_arr)
            if eq_mean <= 0:
                return 0.3
            # Normalise slope as daily % change
            norm_slope = slope / eq_mean * 100  # approx daily pct
            # Map: slope=0 -> 1.0, slope>0.5%/day -> 1.5, slope<-0.5%/day -> 0.3
            mult = np.clip(1.0 + norm_slope * 1.0, 0.3, 1.5)
            return mult
        except Exception:
            return 1.0

    def eq_mult_recovery(daily_eq, high_water, trades, streak_state,
                         recovery_rate=1.0, **kwargs):
        """After drawdown > 10%, gradually increase size as equity recovers.

        When at high water: mult = 1.0 + recovery_rate * 0.5 (small bonus)
        When in DD: mult = recovery fraction * recovery_rate, floored at 0.5
        """
        if high_water <= 0:
            return 1.0
        dd_pct = (daily_eq[-1] - high_water) / high_water  # e.g. -0.15 = 15% DD

        if dd_pct >= 0:
            # At or above high water — small momentum bonus
            return 1.0 + recovery_rate * 0.3
        elif dd_pct > -0.10:
            # Mild DD (< 10%) — normal sizing
            return 1.0
        else:
            # In deeper DD (> 10%) — scale by recovery progress
            # dd_pct is between -1.0 and -0.10
            # Recovery fraction: 0 at worst (equity=0), 1 at high water
            # We use a simpler approach: mult proportional to dd_pct
            # dd_pct = -0.10 -> mult = 1.0, dd_pct = -0.30 -> mult depends on rate
            # Higher recovery_rate = more aggressive re-entry
            recovery_frac = 1.0 + dd_pct  # 0.9 at -10%, 0.7 at -30%
            mult = 0.5 + recovery_frac * recovery_rate
            return np.clip(mult, 0.3, 1.5)

    def eq_mult_streak(daily_eq, high_water, trades, streak_state,
                       win_bonus=0.1, loss_penalty=0.1, **kwargs):
        """Consecutive wins increase size (cap 1.3), consecutive losses decrease (floor 0.5).

        streak_state: dict with 'consec_wins' and 'consec_losses' updated externally.
        """
        cw = streak_state.get('consec_wins', 0)
        cl = streak_state.get('consec_losses', 0)

        mult = 1.0
        # Win streak bonus: 1.0 -> 1.1 -> 1.2 -> 1.3 (cap)
        if cw > 0:
            mult += min(cw, 3) * win_bonus
        # Loss streak penalty: 1.0 -> 0.9 -> 0.8 -> 0.5 (floor)
        if cl > 0:
            mult -= min(cl, 3) * loss_penalty

        return np.clip(mult, 0.3, 1.5)

    def eq_mult_combined(daily_eq, high_water, trades, streak_state,
                         fast=20, slow=50, slope_window=20,
                         recovery_rate=1.0, win_bonus=0.1, loss_penalty=0.1,
                         **kwargs):
        """Geometric mean of sma_cross + slope + recovery + streak.
        Each sub-method contributes a multiplier; final = geometric mean."""
        m_sma = eq_mult_sma_cross(daily_eq, high_water, trades, streak_state,
                                   fast=fast, slow=slow)
        m_slope = eq_mult_slope(daily_eq, high_water, trades, streak_state,
                                 window=slope_window)
        m_recov = eq_mult_recovery(daily_eq, high_water, trades, streak_state,
                                    recovery_rate=recovery_rate)
        m_streak = eq_mult_streak(daily_eq, high_water, trades, streak_state,
                                   win_bonus=win_bonus, loss_penalty=loss_penalty)
        # Geometric mean
        gmean = (m_sma * m_slope * m_recov * m_streak) ** 0.25
        return np.clip(gmean, 0.3, 1.5)

    EQ_METHODS = {
        'none': eq_mult_none,
        'sma_cross': eq_mult_sma_cross,
        'slope': eq_mult_slope,
        'recovery': eq_mult_recovery,
        'streak': eq_mult_streak,
        'combined': eq_mult_combined,
    }

    # ===================== UNIFIED BACKTEST ENGINE =====================
    def backtest(start_di=MIN_TRAIN, end_di=None,
                 eq_method='none',
                 # SMA cross params
                 sma_fast=20, sma_slow=50,
                 # Slope params
                 slope_window=20,
                 # Recovery params
                 recovery_rate=1.0,
                 # Streak params
                 win_bonus=0.1, loss_penalty=0.1,
                 # Kitchen Sink base
                 dd_tiers=None, max_corr=0.5,
                 regime_lo=0.5, regime_hi=1.5,
                 hold=1, top_n=3):

        if end_di is None:
            end_di = ND
        if dd_tiers is None:
            dd_tiers = [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)]

        cash = float(CASH0)
        positions = []
        trades = []          # list of pnl_pct (signed)
        daily_eq = []
        high_water = float(CASH0)

        # Streak state (updated on trade close)
        streak_state = {'consec_wins': 0, 'consec_losses': 0}

        eq_fn = EQ_METHODS[eq_method]

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

                    # Update streak state
                    if pp > 0:
                        streak_state['consec_wins'] += 1
                        streak_state['consec_losses'] = 0
                    else:
                        streak_state['consec_losses'] += 1
                        streak_state['consec_wins'] = 0

            for p in cl:
                positions.remove(p)

            # --- Kitchen Sink sizing: DD * equity_curve_mult * regime ---
            dd_sz = dd_size(pv, high_water, dd_tiers)

            # Equity curve multiplier (replaces wr_size)
            equity_curve_mult = eq_fn(
                daily_eq, high_water, trades, streak_state,
                fast=sma_fast, slow=sma_slow,
                window=slope_window, slope_window=slope_window,
                recovery_rate=recovery_rate,
                win_bonus=win_bonus, loss_penalty=loss_penalty
            )

            composite = compute_composite(di, daily_eq, high_water)
            regime_mult = regime_lo + composite * (regime_hi - regime_lo)

            pos_size = dd_sz * equity_curve_mult * regime_mult
            pos_size = max(0.05, min(0.99, pos_size))

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
        print(f"  {label:95s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d}")

    def walk_forward(eq_method='none',
                     sma_fast=20, sma_slow=50,
                     slope_window=20,
                     recovery_rate=1.0,
                     win_bonus=0.1, loss_penalty=0.1,
                     dd_tiers=None, max_corr=0.5,
                     regime_lo=0.5, regime_hi=1.5,
                     hold=1, top_n=3, label=""):
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
                         eq_method=eq_method,
                         sma_fast=sma_fast, sma_slow=sma_slow,
                         slope_window=slope_window,
                         recovery_rate=recovery_rate,
                         win_bonus=win_bonus, loss_penalty=loss_penalty,
                         dd_tiers=dd_tiers, max_corr=max_corr,
                         regime_lo=regime_lo, regime_hi=regime_hi,
                         hold=hold, top_n=top_n)
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

    # ===================== CONFIG DEFINITIONS =====================
    DD_TIERS = {
        'aggro100': [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)],
    }

    # ===================== SECTION 0: BASELINE =====================
    print("\n" + "=" * 130)
    print("  SECTION 0: BASELINES (no equity curve sizing)")
    print("  Kitchen Sink = dd_sz * 1.0 * regime_mult")
    print("=" * 130)

    dd_t = DD_TIERS['aggro100']
    r_base = backtest(eq_method='none', dd_tiers=dd_t, max_corr=0.5,
                      regime_lo=0.5, regime_hi=1.5, top_n=3)
    pr(r_base, "Baseline: none DD100/90/70/50 corr<0.5 reg0.5-1.5 top3")

    # ===================== SECTION 1: SMA CROSS SWEEP =====================
    print("\n" + "=" * 130)
    print("  SECTION 1: EQUITY CURVE SMA CROSS")
    print("  If equity SMA(fast) > SMA(slow): mult=1.0. Below: mult=0.5")
    print("=" * 130)

    sma_results = []
    for fast in [10, 20]:
        for slow in [40, 50, 60]:
            label = f"sma_cross F{fast}/S{slow}"
            r = backtest(eq_method='sma_cross', sma_fast=fast, sma_slow=slow,
                         dd_tiers=dd_t, max_corr=0.5, regime_lo=0.5, regime_hi=1.5, top_n=3)
            pr(r, label)
            sma_results.append((r, label, {'eq_method': 'sma_cross', 'sma_fast': fast, 'sma_slow': slow}))

    # ===================== SECTION 2: SLOPE SWEEP =====================
    print("\n" + "=" * 130)
    print("  SECTION 2: ROLLING EQUITY CURVE SLOPE")
    print("  Size proportional to rolling slope, mapped to [0.3, 1.5]")
    print("=" * 130)

    slope_results = []
    for window in [10, 20, 30]:
        label = f"slope W{window}"
        r = backtest(eq_method='slope', slope_window=window,
                     dd_tiers=dd_t, max_corr=0.5, regime_lo=0.5, regime_hi=1.5, top_n=3)
        pr(r, label)
        slope_results.append((r, label, {'eq_method': 'slope', 'slope_window': window}))

    # ===================== SECTION 3: RECOVERY SWEEP =====================
    print("\n" + "=" * 130)
    print("  SECTION 3: DRAWDOWN RECOVERY ACCELERATION")
    print("  After DD>10%, increase size as equity recovers toward high water")
    print("=" * 130)

    recovery_results = []
    for rate in [0.5, 1.0, 1.5]:
        label = f"recovery rate={rate}"
        r = backtest(eq_method='recovery', recovery_rate=rate,
                     dd_tiers=dd_t, max_corr=0.5, regime_lo=0.5, regime_hi=1.5, top_n=3)
        pr(r, label)
        recovery_results.append((r, label, {'eq_method': 'recovery', 'recovery_rate': rate}))

    # ===================== SECTION 4: STREAK SWEEP =====================
    print("\n" + "=" * 130)
    print("  SECTION 4: WIN/LOSS STREAK SIZING")
    print("  Consecutive wins: +bonus per win (cap 1.3). Consecutive losses: -penalty per loss (floor 0.5)")
    print("=" * 130)

    streak_results = []
    for wb in [0.1, 0.15, 0.2]:
        for lp in [0.1, 0.15, 0.2]:
            label = f"streak W+{wb}/L-{lp}"
            r = backtest(eq_method='streak', win_bonus=wb, loss_penalty=lp,
                         dd_tiers=dd_t, max_corr=0.5, regime_lo=0.5, regime_hi=1.5, top_n=3)
            pr(r, label)
            streak_results.append((r, label, {'eq_method': 'streak', 'win_bonus': wb, 'loss_penalty': lp}))

    # ===================== SECTION 5: COMBINED SWEEP =====================
    print("\n" + "=" * 130)
    print("  SECTION 5: COMBINED (geometric mean of sma_cross + slope + recovery + streak)")
    print("=" * 130)

    combined_results = []
    # Sweep key sub-params for combined
    for fast in [10, 20]:
        for slow in [40, 50]:
            for slope_w in [10, 20]:
                for rate in [1.0, 1.5]:
                    for wb in [0.1, 0.15]:
                        for lp in [0.1, 0.15]:
                            label = f"combined F{fast}/S{slow} slW{slope_w} rec{rate} W+{wb}/L-{lp}"
                            r = backtest(eq_method='combined',
                                         sma_fast=fast, sma_slow=slow,
                                         slope_window=slope_w,
                                         recovery_rate=rate,
                                         win_bonus=wb, loss_penalty=lp,
                                         dd_tiers=dd_t, max_corr=0.5,
                                         regime_lo=0.5, regime_hi=1.5, top_n=3)
                            pr(r, label)
                            combined_results.append((r, label, {
                                'eq_method': 'combined', 'sma_fast': fast, 'sma_slow': slow,
                                'slope_window': slope_w, 'recovery_rate': rate,
                                'win_bonus': wb, 'loss_penalty': lp
                            }))

    # ===================== SECTION 6: FULL-PERIOD RANKING =====================
    print("\n" + "=" * 130)
    print("  SECTION 6: FULL-PERIOD TOP 20 BY ANNUAL RETURN")
    print("=" * 130)

    all_results = ([(r_base, 'Baseline: none', {'eq_method': 'none'})] +
                   sma_results + slope_results + recovery_results +
                   streak_results + combined_results)

    all_valid = [(r, label, cfg) for r, label, cfg in all_results if r['mdd'] > -80]
    all_valid.sort(key=lambda x: -x[0]['ann'])

    for i, (r, label, cfg) in enumerate(all_valid[:20]):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i + 1:2d} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | "
              f"Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d} | {label}")

    # ===================== SECTION 7: TOP 20 BY R/M RATIO =====================
    print("\n" + "=" * 130)
    print("  SECTION 7: FULL-PERIOD TOP 20 BY R/M RATIO")
    print("=" * 130)

    all_rm = sorted(all_valid, key=lambda x: -abs(x[0]['ann'] / x[0]['mdd']) if x[0]['mdd'] != 0 else 0)
    for i, (r, label, cfg) in enumerate(all_rm[:20]):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  #{i + 1:2d} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | "
              f"Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d} | {label}")

    # ===================== SECTION 8: WALK-FORWARD VALIDATION =====================
    print("\n" + "=" * 130)
    print("  SECTION 8: WALK-FORWARD VALIDATION — TOP 25 by full-period ann")
    print("=" * 130)

    wf_all = {}
    # Deduplicate by label
    seen_labels = set()
    wf_candidates = []
    for r, label, cfg in all_valid:
        if label not in seen_labels:
            seen_labels.add(label)
            wf_candidates.append((r, label, cfg))
    # Take top 25 by annual return
    wf_candidates.sort(key=lambda x: -x[0]['ann'])
    wf_candidates = wf_candidates[:25]

    for r, label, cfg in wf_candidates:
        wf_res = walk_forward(
            eq_method=cfg.get('eq_method', 'none'),
            sma_fast=cfg.get('sma_fast', 20), sma_slow=cfg.get('sma_slow', 50),
            slope_window=cfg.get('slope_window', 20),
            recovery_rate=cfg.get('recovery_rate', 1.0),
            win_bonus=cfg.get('win_bonus', 0.1), loss_penalty=cfg.get('loss_penalty', 0.1),
            dd_tiers=dd_t, max_corr=0.5, regime_lo=0.5, regime_hi=1.5,
            top_n=3, label=label
        )
        wf_all[label] = (wf_res, cfg)
        print_wf(wf_res, label)

    # Also WF for a few targeted combined configs
    targeted_combined = [
        {'eq_method': 'combined', 'sma_fast': 10, 'sma_slow': 40, 'slope_window': 10,
         'recovery_rate': 1.0, 'win_bonus': 0.1, 'loss_penalty': 0.1},
        {'eq_method': 'combined', 'sma_fast': 20, 'sma_slow': 50, 'slope_window': 20,
         'recovery_rate': 1.5, 'win_bonus': 0.15, 'loss_penalty': 0.15},
        {'eq_method': 'combined', 'sma_fast': 10, 'sma_slow': 50, 'slope_window': 20,
         'recovery_rate': 1.0, 'win_bonus': 0.1, 'loss_penalty': 0.15},
    ]
    for cfg in targeted_combined:
        label = (f"combined F{cfg['sma_fast']}/S{cfg['sma_slow']} "
                 f"slW{cfg['slope_window']} rec{cfg['recovery_rate']} "
                 f"W+{cfg['win_bonus']}/L-{cfg['loss_penalty']}")
        if label not in wf_all:
            wf_res = walk_forward(**cfg, dd_tiers=dd_t, max_corr=0.5,
                                  regime_lo=0.5, regime_hi=1.5, top_n=3, label=label)
            wf_all[label] = (wf_res, cfg)
            print_wf(wf_res, label)

    # Also include baseline WF
    if 'Baseline: none' not in wf_all:
        wf_res = walk_forward(eq_method='none', dd_tiers=dd_t, max_corr=0.5,
                              regime_lo=0.5, regime_hi=1.5, top_n=3, label='Baseline: none')
        wf_all['Baseline: none'] = (wf_res, {'eq_method': 'none'})

    # ===================== SECTION 9: TOP 10 BY WF AVG =====================
    print("\n" + "=" * 130)
    print("  SECTION 9: TOP 10 CONFIGS BY WF AVERAGE ANNUAL RETURN")
    print("=" * 130)

    wf_ranked = []
    for label, (wf_res, cfg) in wf_all.items():
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        best_ann = max(r['ann'] for r in wf_res.values())
        pos = sum(1 for r in wf_res.values() if r['ann'] > 0)
        total_n = sum(r['n'] for r in wf_res.values())
        avg_wr = np.mean([r['wr'] for r in wf_res.values()])
        wf_ranked.append((avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res))

    wf_ranked.sort(key=lambda x: -x[0])

    for i, (avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res) in enumerate(wf_ranked[:10]):
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  #{i + 1} WF AVG={avg_ann:+.0f}% | WorstMDD={worst_mdd:.0f}% | {pos}/6 pos | TotalTrades={total_n} | AvgWR={avg_wr:.1f}%")
        print(f"     {label}")
        print(f"     {ws}")

    # ===================== SECTION 10: BEST RISK-ADJUSTED =====================
    print("\n" + "=" * 130)
    print("  SECTION 10: BEST RISK-ADJUSTED (WF avg / |worst MDD|)")
    print("=" * 130)

    wf_ra = sorted(wf_ranked, key=lambda x: -abs(x[0] / x[1]) if x[1] != 0 else 0)
    for i, (avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res) in enumerate(wf_ra[:10]):
        ratio = abs(avg_ann / worst_mdd) if worst_mdd != 0 else 0
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  #{i + 1} WF AVG={avg_ann:+.0f}% | WorstMDD={worst_mdd:.0f}% | R/M={ratio:.2f} | {pos}/6 pos")
        print(f"     {label}")
        print(f"     {ws}")

    # ===================== SECTION 11: BEST PER METHOD =====================
    print("\n" + "=" * 130)
    print("  SECTION 11: BEST PER EQUITY CURVE METHOD (WF avg)")
    print("=" * 130)

    for method in ['none', 'sma_cross', 'slope', 'recovery', 'streak', 'combined']:
        method_results = [(avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr)
                          for avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr in wf_ranked
                          if cfg.get('eq_method', 'none') == method]
        if not method_results:
            print(f"\n  {method}: no WF results")
            continue
        method_results.sort(key=lambda x: -x[0])
        best = method_results[0]
        avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res = best
        ratio = abs(avg_ann / worst_mdd) if worst_mdd != 0 else 0
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  {method:12s} best: WF AVG={avg_ann:+.0f}% | WorstMDD={worst_mdd:.0f}% | R/M={ratio:.2f} | {pos}/6 pos")
        print(f"     {label}")
        print(f"     {ws}")

    # ===================== SECTION 12: DELTA VS BASELINE =====================
    print("\n" + "=" * 130)
    print("  SECTION 12: DELTA VS BASELINE (equity curve sizing OFF)")
    print("=" * 130)

    base_wf = wf_all.get('Baseline: none', None)
    if base_wf:
        base_avg = np.mean([r['ann'] for r in base_wf[0].values()])
        print(f"\n  Baseline AvgWF: {base_avg:+.0f}%")
        print(f"\n  {'Config':95s} | {'AvgWF':>7s} | {'Delta':>7s} | {'WfMDD':>6s}")
        print(f"  {'-' * 95}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 6}")
        for avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res in wf_ranked:
            delta = avg_ann - base_avg
            marker = " ***" if delta > 10 else ""
            print(f"  {label:95s} | {avg_ann:>+6.0f}% | {delta:>+6.0f}% | {worst_mdd:>5.1f}%{marker}")

    # ===================== SECTION 13: CONFIGS MEETING TARGET =====================
    print("\n" + "=" * 130)
    print("  SECTION 13: CONFIGS MEETING TARGET (>160% WF avg AND >-25% worst WF MDD)")
    print("=" * 130)

    targets_met = [(avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr)
                   for avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr in wf_ranked
                   if avg > 160 and wmdd > -25]
    if targets_met:
        for i, (avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res) in enumerate(targets_met[:10]):
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(wf_res.items())])
            print(f"\n  *** TARGET MET #{i + 1}: WF AVG={avg_ann:+.0f}% | WorstMDD={worst_mdd:.0f}% | {pos}/6 pos")
            print(f"      {label}")
            print(f"      {ws}")
    else:
        print("\n  No configs met both targets. Showing closest:")
        for avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr in wf_ranked[:5]:
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(wfr.items())])
            print(f"  WF AVG={avg:+.0f}% | WorstMDD={wmdd:.0f}% | {lbl}")
            print(f"    {ws}")

    # Also show configs with WF avg > 150 (relaxed threshold)
    print(f"\n  --- All configs with WF avg > +150%:")
    for avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr in wf_ranked:
        if avg > 150:
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(wfr.items())])
            print(f"  *** {lbl}")
            print(f"      WF AVG={avg:+.0f}% | WorstMDD={wmdd:.0f}%")
            print(f"      {ws}")

    # And configs with worst MDD > -25%
    print(f"\n  --- All configs with worst WF MDD > -25%:")
    for avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr in sorted(wf_ranked, key=lambda x: -x[0]):
        if wmdd > -25:
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(wfr.items())])
            print(f"  *** {lbl}")
            print(f"      WF AVG={avg:+.0f}% | WorstMDD={wmdd:.0f}%")
            print(f"      {ws}")

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 130)
    print("  FINAL SUMMARY")
    print("=" * 130)

    print(f"\n  V162: Equity Curve Momentum & Anti-Martingale Approaches")
    print(f"  Kitchen Sink = dd_sz * equity_curve_mult * regime_mult")
    print(f"  Methods tested: none, sma_cross, slope, recovery, streak, combined")

    if wf_ranked:
        best = wf_ranked[0]
        avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res = best
        print(f"\n  Best V162: {label}")
        print(f"    WF AVG={avg_ann:+.0f}% | WorstMDD={worst_mdd:.0f}% | {pos}/6 pos")
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"    {ws}")

        # Best non-none method
        non_none = [(avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr)
                    for avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr in wf_ranked
                    if cfg.get('eq_method', 'none') != 'none']
        if non_none:
            best_method = max(non_none, key=lambda x: x[0])
            avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res = best_method
            print(f"\n  Best NON-NONE V162: {label}")
            print(f"    WF AVG={avg_ann:+.0f}% | WorstMDD={worst_mdd:.0f}% | {pos}/6 pos")
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(wf_res.items())])
            print(f"    {ws}")

        if base_wf:
            delta = best[0] - base_avg
            print(f"\n  Delta vs baseline: {delta:+.0f}% WF avg")

    print(f"\n  Elapsed: {time.time() - t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
