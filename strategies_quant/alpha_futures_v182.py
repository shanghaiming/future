"""
Alpha Futures V182 — OI Accumulation Signal Strategy
==============================================================================
V177 champion: short_mirror, atr_norm<10%, corr=0.5, top_n=3
  -> +187% annual, -16% MDD, R/M=11.41, 3771 trades

V176 previously tested OI as a signal enhancer on V169 and found that
all OI modes hurt R/M. V182 takes a fundamentally different approach:

Instead of boosting V121 with OI, V182 builds PURE OI-based signals from
scratch, testing each factor independently then combining the best:

  1. OI Surge:    OI > 2x 20-day avg -> abnormal position building
  2. OI-Price Div: price_up + OI_up = new longs (bullish),
                   price_up + OI_down = short covering (bearish for next day)
  3. OI Momentum:  OI change rate over 5/10/20 days
  4. Volume-OI:    vol/OI ratio spike = high turnover vs holding depth

Signal generation:
  - For each factor, compute z-scores across all symbols on each day
  - Test each factor individually, then combine the best ones
  - Long signals only (positive signals)

Walk-forward validation:
  - Training: 2019-2023
  - Test: 2024-2026
  - Report R/M ratio, only show configs beating V177 baseline R/M=11.41
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
    print("  V182 — OI Accumulation Signal Strategy")
    print("  Pure OI-based factors: Surge, Price-Div, Momentum, Vol-OI Ratio")
    print("  Walk-forward: Train 2019-2023, Test 2024-2026")
    print("=" * 130)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    print(f"  {NS} commodities, {ND} days")

    # Check if OI data is actually available
    oi_valid = ~np.isnan(OI)
    oi_coverage = np.sum(oi_valid) / OI.size * 100
    print(f"  OI data coverage: {oi_coverage:.1f}% of cells have valid OI")
    if oi_coverage < 1.0:
        print("  WARNING: Very low OI coverage. Results may be unreliable.")

    # ===================== PRECOMPUTE =====================
    print("\n[Precompute]...", flush=True)
    t0 = time.time()

    # Basic price indicators
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

    # ===================== OI FACTOR COMPUTATION =====================
    print("  Computing OI factors...", flush=True)

    # Factor 1: OI Surge — OI relative to 20-day average
    OI_RATIO20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        oi = OI[si].astype(np.float64)
        for di in range(20, ND):
            oi_now = oi[di]
            if np.isnan(oi_now) or oi_now <= 0: continue
            window_oi = oi[di-20:di]
            avg_oi = np.nanmean(window_oi)
            if avg_oi > 0:
                OI_RATIO20[si, di] = oi_now / avg_oi

    # Factor 2: OI Momentum — change rate over 5, 10, 20 days
    OI_MOM5 = np.full((NS, ND), np.nan)
    OI_MOM10 = np.full((NS, ND), np.nan)
    OI_MOM20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        oi = OI[si].astype(np.float64)
        for di in range(20, ND):
            oi_now = oi[di]
            if np.isnan(oi_now) or oi_now <= 0: continue
            oi_5 = oi[di-5]; oi_10 = oi[di-10]; oi_20 = oi[di-20]
            if not np.isnan(oi_5) and oi_5 > 0:
                OI_MOM5[si, di] = (oi_now / oi_5 - 1) * 100
            if not np.isnan(oi_10) and oi_10 > 0:
                OI_MOM10[si, di] = (oi_now / oi_10 - 1) * 100
            if not np.isnan(oi_20) and oi_20 > 0:
                OI_MOM20[si, di] = (oi_now / oi_20 - 1) * 100

    # Factor 3: OI-Price Divergence
    #   price_up + OI_up   = +2 (new longs entering, bullish)
    #   price_up + OI_down = -1 (short covering, bearish next day)
    #   price_down + OI_up = -1 (new shorts entering, bearish)
    #   price_down + OI_down = 0 (long liquidation, neutral)
    OI_PRICE_DIV = np.full((NS, ND), np.nan)
    for si in range(NS):
        oi = OI[si].astype(np.float64)
        close = C[si].astype(np.float64)
        for di in range(5, ND):
            oi_now = oi[di]; oi_5 = oi[di-5]
            c_now = close[di]; c_5 = close[di-5]
            if any(np.isnan(x) for x in [oi_now, oi_5, c_now, c_5]): continue
            if oi_5 <= 0 or c_5 <= 0: continue
            price_up = c_now > c_5
            oi_up = oi_now > oi_5
            if price_up and oi_up:
                OI_PRICE_DIV[si, di] = 2.0
            elif price_up and not oi_up:
                OI_PRICE_DIV[si, di] = -1.0
            elif not price_up and oi_up:
                OI_PRICE_DIV[si, di] = -1.0
            else:
                OI_PRICE_DIV[si, di] = 0.0

    # Factor 4: Volume-OI Ratio — vol/OI, then z-score within symbol history
    VOL_OI_RATIO = np.full((NS, ND), np.nan)
    for si in range(NS):
        oi = OI[si].astype(np.float64)
        vol = V[si].astype(np.float64)
        for di in range(ND):
            oi_now = oi[di]; vol_now = vol[di]
            if np.isnan(oi_now) or np.isnan(vol_now) or oi_now <= 0: continue
            VOL_OI_RATIO[si, di] = vol_now / oi_now

    # ===================== CROSS-SECTIONAL Z-SCORES =====================
    # For each day, compute z-score of each OI factor across all symbols.
    # This normalizes factors so we can compare across instruments.

    def cross_sectional_zscore(factor_array, min_valid=5):
        """Compute z-scores across symbols for each day."""
        Z = np.full_like(factor_array, np.nan)
        for di in range(ND):
            vals = factor_array[:, di]
            valid = vals[~np.isnan(vals)]
            if len(valid) < min_valid: continue
            mu = np.mean(valid)
            sigma = np.std(valid, ddof=1)
            if sigma <= 0: continue
            for si in range(NS):
                if not np.isnan(vals[si]):
                    Z[si, di] = (vals[si] - mu) / sigma
        return Z

    print("  Computing cross-sectional z-scores...", flush=True)
    Z_OI_SURGE = cross_sectional_zscore(OI_RATIO20)
    Z_OI_MOM5 = cross_sectional_zscore(OI_MOM5)
    Z_OI_MOM10 = cross_sectional_zscore(OI_MOM10)
    Z_OI_MOM20 = cross_sectional_zscore(OI_MOM20)
    Z_OI_PRICE_DIV = cross_sectional_zscore(OI_PRICE_DIV)
    Z_VOL_OI = cross_sectional_zscore(VOL_OI_RATIO)

    # Summary stats for OI factors
    surge_count = np.sum(OI_RATIO20[~np.isnan(OI_RATIO20)] > 2.0)
    surge_total = np.sum(~np.isnan(OI_RATIO20))
    bull_div_count = np.sum(OI_PRICE_DIV[~np.isnan(OI_PRICE_DIV)] == 2.0)
    div_total = np.sum(~np.isnan(OI_PRICE_DIV))
    print(f"  OI Surge > 2x: {surge_count}/{surge_total} ({surge_count/max(surge_total,1)*100:.1f}%)")
    print(f"  Bullish OI-Price Div: {bull_div_count}/{div_total} ({bull_div_count/max(div_total,1)*100:.1f}%)")
    print(f"  Precompute done ({time.time()-t0:.1f}s)")

    # ===================== WALK-FORWARD DATE BOUNDARIES =====================
    train_start = train_end = test_start = test_end = None
    for di in range(ND):
        yr = dates[di].year
        if yr == 2019 and train_start is None: train_start = di
        if yr == 2023: train_end = di + 1
        if yr == 2024 and test_start is None: test_start = di
        if yr <= 2026: test_end = di + 1
    if train_start is None: train_start = MIN_TRAIN
    if train_end is None: train_end = ND // 2
    if test_start is None: test_start = train_end
    if test_end is None: test_end = ND

    print(f"\n  Train: {dates[train_start].date()} -> {dates[min(train_end-1, ND-1)].date()} "
          f"({train_end - train_start} days)")
    print(f"  Test:  {dates[test_start].date()} -> {dates[min(test_end-1, ND-1)].date()} "
          f"({test_end - test_start} days)")

    # ===================== BACKTEST ENGINE =====================
    def backtest(start_di=None, end_di=None, signal_fn=None,
                 atr_norm_max=10.0, max_corr=0.5,
                 dd_tiers=None, regime_lo=0.5, regime_hi=1.5,
                 top_n=3, hold=1):
        """Standard backtest engine for OI factor signals.
        signal_fn(di, edi) -> list of (score, si, entry_price, label)
        """
        if start_di is None: start_di = MIN_TRAIN
        if end_di is None: end_di = ND
        if signal_fn is None: signal_fn = lambda di, edi: []
        if dd_tiers is None:
            dd_tiers = [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)]

        cash = float(CASH0)
        positions = []
        trades = []
        daily_eq = []
        high_water = float(CASH0)
        trade_pnls = []

        for di in range(start_di, end_di - 1):
            # Mark-to-market
            pv = cash
            for p in positions:
                cp = C[p['si'], di]
                if not np.isnan(cp) and cp > 0:
                    m = MULT.get(p['sym'], DEF_MULT)
                    unrealized = (cp - p['entry_price']) * m * p['lots']
                    pv += p['entry_price'] * m * abs(p['lots']) + unrealized - cp * m * abs(p['lots']) * COMM
            daily_eq.append(pv)
            if pv > high_water:
                high_water = pv

            # --- Exit logic (fixed hold) ---
            cl = []
            for p in positions:
                days_held = di - p['entry_di']
                if days_held >= hold:
                    cp = C[p['si'], di]
                    if np.isnan(cp) or cp <= 0: continue
                    m = MULT.get(p['sym'], DEF_MULT)
                    pnl = (cp - p['entry_price']) * m * p['lots']
                    inv = p['entry_price'] * m * abs(p['lots'])
                    pp = pnl / inv * 100 if inv > 0 else 0
                    cash += cp * m * abs(p['lots']) * (1 - COMM)
                    trades.append(pp)
                    trade_pnls.append(pnl)
                    cl.append(p)
            for p in cl: positions.remove(p)

            # --- Position sizing ---
            dd_sz = dd_size(pv, high_water, dd_tiers)
            composite = compute_composite(di, daily_eq, high_water)
            regime_mult = regime_lo + composite * (regime_hi - regime_lo)
            pos_size = max(0.05, min(0.99, dd_sz * regime_mult))

            # --- Enter positions ---
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue

            held_si = set(p['si'] for p in positions)

            # Get signals from factor function
            candidates = signal_fn(di, edi)
            # Filter by ATR norm
            candidates_f = [c for c in candidates
                           if not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < atr_norm_max]
            candidates_f.sort(key=lambda x: -x[0])

            cash_snapshot = cash
            for sc, s, pr, lbl in candidates_f:
                if s in held_si: continue
                if len(positions) >= top_n: break
                cap = cash_snapshot * pos_size
                sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                ct = max(1, int(cap / (pr * m * (1 + COMM))))
                ci = pr * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                    ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                if ct <= 0 or ci <= 0 or ci > cash: continue
                cash -= ci
                positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                  'lots': ct, 'sym': sym, 'sig': lbl, 'score': sc})
                held_si.add(s)

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

        avg_pnl = np.mean(trade_pnls) if trade_pnls else 0

        return {'ann': ann, 'wr': wr, 'n': nt, 'mdd': mdd, 'sharpe': sh,
                'final': cash, 'avg_pnl': avg_pnl}

    # ===================== HELPERS =====================
    def dd_size(pv, high_water, tiers):
        if high_water <= 0: return tiers[0][1]
        dd = (pv - high_water) / high_water
        for dd_thresh, size_frac in tiers:
            if dd >= -dd_thresh: return size_frac
        return tiers[-1][1]

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

    # ===================== SIGNAL FUNCTIONS =====================
    # Each signal function takes (di, edi) and returns [(score, si, entry_price, label), ...]

    def sig_v121_baseline(di, edi):
        """V121 baseline signal for comparison."""
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

    def sig_oi_surge(di, edi, z_thresh=1.0):
        """OI Surge: z-score of OI/20d_avg > threshold."""
        c = []
        for s in range(NS):
            z = Z_OI_SURGE[s, di]
            if np.isnan(z) or z <= z_thresh: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((z, s, ep, f'oi_surge_z{z_thresh}'))
        return c

    def sig_oi_surge_price(di, edi, z_thresh=1.0):
        """OI Surge + price confirmation (price up on the day)."""
        c = []
        for s in range(NS):
            z = Z_OI_SURGE[s, di]
            if np.isnan(z) or z <= z_thresh: continue
            ret = RET[s, di]
            if np.isnan(ret) or ret <= 0: continue  # require price up
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((z * max(ret, 0.1), s, ep, f'oi_surge_price_z{z_thresh}'))
        return c

    def sig_oi_price_bull(di, edi, z_thresh=0.5):
        """OI-Price Divergence bullish: price_up + OI_up (new longs entering)."""
        c = []
        for s in range(NS):
            z = Z_OI_PRICE_DIV[s, di]
            if np.isnan(z) or z <= z_thresh: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((z, s, ep, f'oi_price_bull_z{z_thresh}'))
        return c

    def sig_oi_mom5(di, edi, z_thresh=1.0):
        """OI Momentum 5-day: z-score of OI 5-day change rate."""
        c = []
        for s in range(NS):
            z = Z_OI_MOM5[s, di]
            if np.isnan(z) or z <= z_thresh: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((z, s, ep, f'oi_mom5_z{z_thresh}'))
        return c

    def sig_oi_mom10(di, edi, z_thresh=1.0):
        """OI Momentum 10-day: z-score of OI 10-day change rate."""
        c = []
        for s in range(NS):
            z = Z_OI_MOM10[s, di]
            if np.isnan(z) or z <= z_thresh: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((z, s, ep, f'oi_mom10_z{z_thresh}'))
        return c

    def sig_oi_mom20(di, edi, z_thresh=0.5):
        """OI Momentum 20-day: z-score of OI 20-day change rate."""
        c = []
        for s in range(NS):
            z = Z_OI_MOM20[s, di]
            if np.isnan(z) or z <= z_thresh: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((z, s, ep, f'oi_mom20_z{z_thresh}'))
        return c

    def sig_vol_oi_high(di, edi, z_thresh=1.0):
        """High Volume-OI ratio: indicates speculative activity / high turnover."""
        c = []
        for s in range(NS):
            z = Z_VOL_OI[s, di]
            if np.isnan(z) or z <= z_thresh: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((z, s, ep, f'vol_oi_z{z_thresh}'))
        return c

    def sig_vol_oi_price(di, edi, z_thresh=0.5):
        """High Vol-OI + price up: speculative buying with high turnover."""
        c = []
        for s in range(NS):
            z = Z_VOL_OI[s, di]
            if np.isnan(z) or z <= z_thresh: continue
            ret = RET[s, di]
            if np.isnan(ret) or ret <= 0: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((z * max(ret, 0.1), s, ep, f'vol_oi_price_z{z_thresh}'))
        return c

    def sig_oi_combined_best(di, edi, surge_w=1.0, mom5_w=0.5, bull_w=1.0):
        """Combined OI signal: surge + momentum + bullish divergence.
        Score = weighted sum of z-scores, require at least 2 factors present."""
        c = []
        for s in range(NS):
            scores = []
            z_surge = Z_OI_SURGE[s, di]
            z_mom5 = Z_OI_MOM5[s, di]
            z_bull = Z_OI_PRICE_DIV[s, di]
            if not np.isnan(z_surge) and z_surge > 0: scores.append(z_surge * surge_w)
            if not np.isnan(z_mom5) and z_mom5 > 0: scores.append(z_mom5 * mom5_w)
            if not np.isnan(z_bull) and z_bull > 0: scores.append(z_bull * bull_w)
            if len(scores) < 2: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((sum(scores), s, ep, f'oi_combo_s{surge_w}m{mom5_w}b{bull_w}'))
        return c

    def sig_oi_v121_blend(di, edi, oi_weight=0.5):
        """Blend OI composite score with V121 signal.
        Score = oi_weight * OI_z + (1-oi_weight) * V121_score."""
        c = []
        for s in range(NS):
            # V121 component
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            v121_ok = (not np.isnan(roc) and not np.isnan(zs)
                       and roc > 1.0 and zs > 1.5)
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if v121_ok and not np.isnan(rp) and roc <= rp:
                v121_ok = False
            v121_score = roc * zs if v121_ok else 0.0

            # OI component
            oi_parts = []
            z_surge = Z_OI_SURGE[s, di]
            z_mom5 = Z_OI_MOM5[s, di]
            z_bull = Z_OI_PRICE_DIV[s, di]
            if not np.isnan(z_surge) and z_surge > 0: oi_parts.append(z_surge)
            if not np.isnan(z_mom5) and z_mom5 > 0: oi_parts.append(z_mom5 * 0.5)
            if not np.isnan(z_bull) and z_bull > 0: oi_parts.append(z_bull)
            oi_score = sum(oi_parts) if oi_parts else 0.0

            # Must have at least one positive component
            if v121_score <= 0 and oi_score <= 0: continue
            # Must have some OI signal
            if oi_score <= 0: continue

            total_score = oi_weight * oi_score + (1 - oi_weight) * v121_score
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((total_score, s, ep, f'blend_oi{oi_weight}'))
        return c

    # ===================== PRINTING HELPERS =====================
    BASELINE_RM = 11.41  # V177 baseline R/M to beat

    def pr(r, label=""):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        marker = " ** BEATS V177 **" if ratio > BASELINE_RM else ""
        print(f"  {label:75s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | "
              f"Sh={r['sharpe']:4.2f} | R/M={ratio:6.2f} | N={r['n']:4d} | "
              f"AvgPnL={r['avg_pnl']:>8.0f}{marker}")
        return ratio

    def run_wf_train_test(signal_fn, label="", **bt_kwargs):
        """Run walk-forward: train on 2019-2023, test on 2024-2026."""
        # Train period
        r_train = backtest(start_di=train_start, end_di=train_end,
                           signal_fn=signal_fn, **bt_kwargs)
        # Test period
        r_test = backtest(start_di=test_start, end_di=test_end,
                          signal_fn=signal_fn, **bt_kwargs)
        # Full period
        r_full = backtest(start_di=MIN_TRAIN, signal_fn=signal_fn, **bt_kwargs)

        train_rm = abs(r_train['ann'] / r_train['mdd']) if r_train['mdd'] != 0 else 0
        test_rm = abs(r_test['ann'] / r_test['mdd']) if r_test['mdd'] != 0 else 0
        full_rm = abs(r_full['ann'] / r_full['mdd']) if r_full['mdd'] != 0 else 0

        beats_train = "Y" if train_rm > BASELINE_RM else "N"
        beats_test = "Y" if test_rm > BASELINE_RM else "N"
        beats_full = "Y" if full_rm > BASELINE_RM else "N"

        print(f"  {label}")
        print(f"    TRAIN (2019-2023): Ann={r_train['ann']:+7.1f}% | MDD={r_train['mdd']:5.1f}% | "
              f"R/M={train_rm:6.2f} | N={r_train['n']:4d} | Beats={beats_train}")
        print(f"    TEST  (2024-2026): Ann={r_test['ann']:+7.1f}% | MDD={r_test['mdd']:5.1f}% | "
              f"R/M={test_rm:6.2f} | N={r_test['n']:4d} | Beats={beats_test}")
        print(f"    FULL:             Ann={r_full['ann']:+7.1f}% | MDD={r_full['mdd']:5.1f}% | "
              f"R/M={full_rm:6.2f} | N={r_full['n']:4d} | Beats={beats_full}")

        return {'train': r_train, 'test': r_test, 'full': r_full,
                'train_rm': train_rm, 'test_rm': test_rm, 'full_rm': full_rm,
                'label': label}

    all_results = []

    # ===================== SECTION 0: BASELINE =====================
    print("\n" + "=" * 130)
    print("  SECTION 0: V121 BASELINE (V177 champion config)")
    print("=" * 130)

    r_base = backtest(signal_fn=sig_v121_baseline)
    base_rm = pr(r_base, "V121 BASELINE")
    all_results.append({**r_base, 'label': 'v121_baseline', 'factor': 'baseline',
                        'rm': base_rm})

    print(f"\n  V177 reference R/M = {BASELINE_RM:.2f}")

    # ===================== SECTION 1: OI SURGE FACTOR =====================
    print("\n" + "=" * 130)
    print("  SECTION 1: OI Surge Factor (OI > 2x 20-day average)")
    print("  Hypothesis: Abnormal position building predicts continuation")
    print("=" * 130)

    for zt in [0.5, 1.0, 1.5, 2.0]:
        r = backtest(signal_fn=lambda di, edi, z=zt: sig_oi_surge(di, edi, z_thresh=z))
        rm = pr(r, f"OI SURGE z>{zt}")
        all_results.append({**r, 'label': f'oi_surge_z{zt}', 'factor': 'oi_surge', 'rm': rm})

    # OI Surge + price confirmation
    for zt in [0.5, 1.0, 1.5]:
        r = backtest(signal_fn=lambda di, edi, z=zt: sig_oi_surge_price(di, edi, z_thresh=z))
        rm = pr(r, f"OI SURGE + PRICE z>{zt}")
        all_results.append({**r, 'label': f'oi_surge_price_z{zt}', 'factor': 'oi_surge_price', 'rm': rm})

    # ===================== SECTION 2: OI-PRICE DIVERGENCE =====================
    print("\n" + "=" * 130)
    print("  SECTION 2: OI-Price Divergence (new longs entering)")
    print("  Hypothesis: price_up + OI_up = institutional demand, bullish next day")
    print("=" * 130)

    for zt in [0.0, 0.5, 1.0, 1.5]:
        r = backtest(signal_fn=lambda di, edi, z=zt: sig_oi_price_bull(di, edi, z_thresh=z))
        rm = pr(r, f"OI-PRICE BULL z>{zt}")
        all_results.append({**r, 'label': f'oi_price_bull_z{zt}', 'factor': 'oi_price_div', 'rm': rm})

    # ===================== SECTION 3: OI MOMENTUM =====================
    print("\n" + "=" * 130)
    print("  SECTION 3: OI Momentum (change rate)")
    print("  Hypothesis: Rapidly increasing OI = capital inflow = continuation")
    print("=" * 130)

    for zt in [0.5, 1.0, 1.5]:
        r = backtest(signal_fn=lambda di, edi, z=zt: sig_oi_mom5(di, edi, z_thresh=z))
        rm = pr(r, f"OI MOM5 z>{zt}")
        all_results.append({**r, 'label': f'oi_mom5_z{zt}', 'factor': 'oi_mom5', 'rm': rm})

    for zt in [0.5, 1.0, 1.5]:
        r = backtest(signal_fn=lambda di, edi, z=zt: sig_oi_mom10(di, edi, z_thresh=z))
        rm = pr(r, f"OI MOM10 z>{zt}")
        all_results.append({**r, 'label': f'oi_mom10_z{zt}', 'factor': 'oi_mom10', 'rm': rm})

    for zt in [0.0, 0.5, 1.0]:
        r = backtest(signal_fn=lambda di, edi, z=zt: sig_oi_mom20(di, edi, z_thresh=z))
        rm = pr(r, f"OI MOM20 z>{zt}")
        all_results.append({**r, 'label': f'oi_mom20_z{zt}', 'factor': 'oi_mom20', 'rm': rm})

    # ===================== SECTION 4: VOLUME-OI RATIO =====================
    print("\n" + "=" * 130)
    print("  SECTION 4: Volume-OI Ratio (turnover vs depth)")
    print("  Hypothesis: High vol/OI = speculative spike, captures attention")
    print("=" * 130)

    for zt in [0.5, 1.0, 1.5, 2.0]:
        r = backtest(signal_fn=lambda di, edi, z=zt: sig_vol_oi_high(di, edi, z_thresh=z))
        rm = pr(r, f"VOL-OI HIGH z>{zt}")
        all_results.append({**r, 'label': f'vol_oi_z{zt}', 'factor': 'vol_oi', 'rm': rm})

    for zt in [0.0, 0.5, 1.0]:
        r = backtest(signal_fn=lambda di, edi, z=zt: sig_vol_oi_price(di, edi, z_thresh=z))
        rm = pr(r, f"VOL-OI + PRICE z>{zt}")
        all_results.append({**r, 'label': f'vol_oi_price_z{zt}', 'factor': 'vol_oi_price', 'rm': rm})

    # ===================== SECTION 5: OI HOLD PERIOD VARIANTS =====================
    print("\n" + "=" * 130)
    print("  SECTION 5: Best single factors with hold=2,3")
    print("  Hypothesis: OI signals may need longer to play out")
    print("=" * 130)

    # Identify top 3 single factors by R/M so far
    single_factors = [r for r in all_results if r.get('factor', '') != 'baseline']
    single_factors.sort(key=lambda x: x.get('rm', 0), reverse=True)
    top_singles = single_factors[:3]

    for ts in top_singles:
        lbl = ts['label']
        # Reconstruct signal function from label
        if 'oi_surge_price' in lbl:
            zt = float(lbl.split('z')[-1])
            sig_fn = lambda di, edi, z=zt: sig_oi_surge_price(di, edi, z_thresh=z)
        elif 'oi_surge' in lbl:
            zt = float(lbl.split('z')[-1])
            sig_fn = lambda di, edi, z=zt: sig_oi_surge(di, edi, z_thresh=z)
        elif 'oi_price_bull' in lbl:
            zt = float(lbl.split('z')[-1])
            sig_fn = lambda di, edi, z=zt: sig_oi_price_bull(di, edi, z_thresh=z)
        elif 'oi_mom5' in lbl:
            zt = float(lbl.split('z')[-1])
            sig_fn = lambda di, edi, z=zt: sig_oi_mom5(di, edi, z_thresh=z)
        elif 'oi_mom10' in lbl:
            zt = float(lbl.split('z')[-1])
            sig_fn = lambda di, edi, z=zt: sig_oi_mom10(di, edi, z_thresh=z)
        elif 'oi_mom20' in lbl:
            zt = float(lbl.split('z')[-1])
            sig_fn = lambda di, edi, z=zt: sig_oi_mom20(di, edi, z_thresh=z)
        elif 'vol_oi_price' in lbl:
            zt = float(lbl.split('z')[-1])
            sig_fn = lambda di, edi, z=zt: sig_vol_oi_price(di, edi, z_thresh=z)
        elif 'vol_oi' in lbl:
            zt = float(lbl.split('z')[-1])
            sig_fn = lambda di, edi, z=zt: sig_vol_oi_high(di, edi, z_thresh=z)
        else:
            continue

        for h in [2, 3]:
            r = backtest(signal_fn=sig_fn, hold=h)
            rm = pr(r, f"{lbl} hold={h}")
            all_results.append({**r, 'label': f'{lbl}_h{h}', 'factor': ts['factor'],
                                'rm': rm, 'hold': h})

    # ===================== SECTION 6: COMBINED OI SIGNALS =====================
    print("\n" + "=" * 130)
    print("  SECTION 6: Combined OI Signals (multi-factor)")
    print("  Hypothesis: Combining multiple OI factors improves robustness")
    print("=" * 130)

    # Equal weight
    for combo in [
        (1.0, 0.5, 1.0, 'equal'),
        (2.0, 0.5, 1.0, 'surge_heavy'),
        (1.0, 0.5, 2.0, 'div_heavy'),
        (1.5, 1.0, 1.5, 'all_moderate'),
    ]:
        sw, mw, bw, name = combo
        sig_fn = lambda di, edi, s=sw, m=mw, b=bw: sig_oi_combined_best(
            di, edi, surge_w=s, mom5_w=m, bull_w=b)
        r = backtest(signal_fn=sig_fn)
        rm = pr(r, f"COMBO {name} (s={sw} m={mw} b={bw})")
        all_results.append({**r, 'label': f'combo_{name}', 'factor': 'combined', 'rm': rm})

    # Combined with hold=2
    for combo in [
        (1.0, 0.5, 1.0, 'equal_h2'),
        (2.0, 0.5, 1.0, 'surge_heavy_h2'),
    ]:
        sw, mw, bw, name = combo
        sig_fn = lambda di, edi, s=sw, m=mw, b=bw: sig_oi_combined_best(
            di, edi, surge_w=s, mom5_w=m, bull_w=b)
        r = backtest(signal_fn=sig_fn, hold=2)
        rm = pr(r, f"COMBO {name} (s={sw} m={mw} b={bw})")
        all_results.append({**r, 'label': f'combo_{name}', 'factor': 'combined', 'rm': rm})

    # ===================== SECTION 7: OI + V121 BLEND =====================
    print("\n" + "=" * 130)
    print("  SECTION 7: OI + V121 Blend")
    print("  Hypothesis: OI adds value as a confirmation/booster to V121")
    print("=" * 130)

    for ow in [0.3, 0.5, 0.7, 1.0]:
        sig_fn = lambda di, edi, w=ow: sig_oi_v121_blend(di, edi, oi_weight=w)
        r = backtest(signal_fn=sig_fn)
        rm = pr(r, f"BLEND oi_weight={ow}")
        all_results.append({**r, 'label': f'blend_oi{ow}', 'factor': 'blend', 'rm': rm})

    # Blend with hold=2
    for ow in [0.3, 0.5]:
        sig_fn = lambda di, edi, w=ow: sig_oi_v121_blend(di, edi, oi_weight=w)
        r = backtest(signal_fn=sig_fn, hold=2)
        rm = pr(r, f"BLEND oi_weight={ow} hold=2")
        all_results.append({**r, 'label': f'blend_oi{ow}_h2', 'factor': 'blend', 'rm': rm})

    # ===================== SECTION 8: TOP_N AND ATR VARIANTS =====================
    print("\n" + "=" * 130)
    print("  SECTION 8: Best OI signals with top_n=2,4 and atr_norm=7,12")
    print("=" * 130)

    # Re-run top 3 configs with different top_n and atr_norm
    ranked = sorted(all_results, key=lambda x: x.get('rm', 0), reverse=True)
    top3 = [r for r in ranked if r.get('factor', '') not in ('baseline', 'combined', 'blend')][:3]

    for ts in top3:
        lbl = ts['label']
        base_lbl = lbl.replace('_h2', '').replace('_h3', '')
        # Parse signal function
        if 'oi_surge_price' in base_lbl:
            zt = float(base_lbl.split('z')[-1])
            sig_fn = lambda di, edi, z=zt: sig_oi_surge_price(di, edi, z_thresh=z)
        elif 'oi_surge' in base_lbl:
            zt = float(base_lbl.split('z')[-1])
            sig_fn = lambda di, edi, z=zt: sig_oi_surge(di, edi, z_thresh=z)
        elif 'oi_price_bull' in base_lbl:
            zt = float(base_lbl.split('z')[-1])
            sig_fn = lambda di, edi, z=zt: sig_oi_price_bull(di, edi, z_thresh=z)
        elif 'oi_mom5' in base_lbl:
            zt = float(base_lbl.split('z')[-1])
            sig_fn = lambda di, edi, z=zt: sig_oi_mom5(di, edi, z_thresh=z)
        elif 'oi_mom10' in base_lbl:
            zt = float(base_lbl.split('z')[-1])
            sig_fn = lambda di, edi, z=zt: sig_oi_mom10(di, edi, z_thresh=z)
        elif 'vol_oi_price' in base_lbl:
            zt = float(base_lbl.split('z')[-1])
            sig_fn = lambda di, edi, z=zt: sig_vol_oi_price(di, edi, z_thresh=z)
        elif 'vol_oi' in base_lbl:
            zt = float(base_lbl.split('z')[-1])
            sig_fn = lambda di, edi, z=zt: sig_vol_oi_high(di, edi, z_thresh=z)
        else:
            continue

        hold_val = ts.get('hold', 1)
        for tn in [2, 4]:
            r = backtest(signal_fn=sig_fn, top_n=tn, hold=hold_val)
            rm = pr(r, f"{lbl} top_n={tn}")
            all_results.append({**r, 'label': f'{lbl}_tn{tn}', 'factor': ts['factor'], 'rm': rm})

        for atr_max in [7.0, 12.0]:
            r = backtest(signal_fn=sig_fn, atr_norm_max=atr_max, hold=hold_val)
            rm = pr(r, f"{lbl} atr<{atr_max}%")
            all_results.append({**r, 'label': f'{lbl}_atr{atr_max}', 'factor': ts['factor'], 'rm': rm})

    # ===================== SECTION 9: WALK-FORWARD VALIDATION =====================
    print("\n" + "=" * 130)
    print("  SECTION 9: Walk-Forward Validation (Train 2019-2023, Test 2024-2026)")
    print(f"  V177 Baseline R/M = {BASELINE_RM:.2f}")
    print("=" * 130)

    # Baseline WF
    print(f"\n  --- V121 BASELINE WF ---")
    wf_base = run_wf_train_test(sig_v121_baseline, "V121 BASELINE")
    all_results.append({**wf_base['full'], 'label': 'v121_wf', 'rm': wf_base['full_rm']})

    # Run WF for all configs that beat baseline
    beating = [r for r in all_results if r.get('rm', 0) > BASELINE_RM]
    beating.sort(key=lambda x: x.get('rm', 0), reverse=True)
    # Take top 10
    top_wf = beating[:10]

    for ts in top_wf:
        lbl = ts['label']
        hold_val = ts.get('hold', 1)
        # Reconstruct signal function
        if 'combo_' in lbl:
            name = lbl.replace('combo_', '').replace('_h2', '').replace('_h3', '')
            combos = {'equal': (1.0, 0.5, 1.0), 'surge_heavy': (2.0, 0.5, 1.0),
                      'div_heavy': (1.0, 0.5, 2.0), 'all_moderate': (1.5, 1.0, 1.5),
                      'equal_h2': (1.0, 0.5, 1.0), 'surge_heavy_h2': (2.0, 0.5, 1.0)}
            if name in combos:
                sw, mw, bw = combos[name]
                sig_fn = lambda di, edi, s=sw, m=mw, b=bw: sig_oi_combined_best(
                    di, edi, surge_w=s, mom5_w=m, bull_w=b)
            else:
                continue
        elif 'blend_' in lbl:
            ow = float(lbl.split('oi')[-1].split('_')[0])
            sig_fn = lambda di, edi, w=ow: sig_oi_v121_blend(di, edi, oi_weight=w)
        elif 'oi_surge_price' in lbl:
            zt = float(lbl.split('z')[-1])
            sig_fn = lambda di, edi, z=zt: sig_oi_surge_price(di, edi, z_thresh=z)
        elif 'oi_surge' in lbl:
            zt = float(lbl.split('z')[-1])
            sig_fn = lambda di, edi, z=zt: sig_oi_surge(di, edi, z_thresh=z)
        elif 'oi_price_bull' in lbl:
            zt = float(lbl.split('z')[-1])
            sig_fn = lambda di, edi, z=zt: sig_oi_price_bull(di, edi, z_thresh=z)
        elif 'oi_mom5' in lbl:
            zt = float(lbl.split('z')[-1])
            sig_fn = lambda di, edi, z=zt: sig_oi_mom5(di, edi, z_thresh=z)
        elif 'oi_mom10' in lbl:
            zt = float(lbl.split('z')[-1])
            sig_fn = lambda di, edi, z=zt: sig_oi_mom10(di, edi, z_thresh=z)
        elif 'vol_oi_price' in lbl:
            zt = float(lbl.split('z')[-1])
            sig_fn = lambda di, edi, z=zt: sig_vol_oi_price(di, edi, z_thresh=z)
        elif 'vol_oi' in lbl:
            zt = float(lbl.split('z')[-1])
            sig_fn = lambda di, edi, z=zt: sig_vol_oi_high(di, edi, z_thresh=z)
        else:
            continue

        print(f"\n  --- {lbl} WF ---")
        wf = run_wf_train_test(sig_fn, lbl, hold=hold_val)
        all_results.append({**wf['full'], 'label': f'{lbl}_wf',
                            'rm': wf['full_rm'],
                            'train_rm': wf['train_rm'], 'test_rm': wf['test_rm']})

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 130)
    print("  V182 FINAL SUMMARY: OI Accumulation Signals")
    print(f"  V177 Baseline R/M = {BASELINE_RM:.2f}")
    print("=" * 130)

    # All results ranked by R/M
    ranked_final = sorted(all_results, key=lambda x: x.get('rm', 0), reverse=True)

    print(f"\n  {'Config':45s} | {'Ann':>7s} | {'MDD':>5s} | {'R/M':>6s} | {'WR':>5s} | "
          f"{'N':>4s} | {'Sh':>5s} | vs V177")
    print(f"  {'-'*45}-+-{'-'*7}-+-{'-'*5}-+-{'-'*6}-+-{'-'*5}-+-{'-'*4}-+-{'-'*5}-+-{'-'*8}")

    for r in ranked_final[:30]:
        ann = r['ann']; mdd = r['mdd']
        rm = r.get('rm', abs(ann / mdd) if mdd != 0 else 0)
        delta = rm - BASELINE_RM
        marker = " *** BEATS ***" if delta > 0 else ""
        print(f"  {r['label']:45s} | {ann:>+7.0f}% | {mdd:>5.0f}% | {rm:>6.2f} | "
              f"{r.get('wr',0):>5.1f}% | {r.get('n',0):>4d} | {r.get('sharpe',0):>5.2f} | "
              f"{delta:>+8.2f}{marker}")

    # Walk-forward summary for configs that beat baseline
    wf_configs = [r for r in ranked_final if 'train_rm' in r]
    if wf_configs:
        print(f"\n  {'--- Walk-Forward Summary (configs that beat V177 on full period) ---':^130}")
        print(f"  {'Config':45s} | {'TrainRM':>7s} | {'TestRM':>7s} | {'FullRM':>7s} | "
              f"{'Test Ann':>8s} | {'Test MDD':>8s}")
        print(f"  {'-'*45}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}")
        for r in wf_configs[:15]:
            print(f"  {r['label']:45s} | {r['train_rm']:>7.2f} | {r['test_rm']:>7.2f} | "
                  f"{r['rm']:>7.2f} | {r.get('ann',0):>+7.0f}% | {r.get('mdd',0):>7.0f}%")

    # Summary stats
    beating_final = [r for r in all_results if r.get('rm', 0) > BASELINE_RM]
    print(f"\n  Total configs tested: {len(all_results)}")
    print(f"  Configs beating V177 R/M={BASELINE_RM:.2f}: {len(beating_final)}")
    if beating_final:
        best = max(beating_final, key=lambda x: x.get('rm', 0))
        print(f"  Best config: {best['label']} with R/M={best.get('rm', 0):.2f}")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
