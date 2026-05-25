"""
Alpha V39 — TA-Lib Factor Exploration
======================================
TA-Lib has 158 technical indicators. Most haven't been tested as factors.
V39 systematically tests promising TA-Lib indicators as cross-sectional factors.

Priority factors (based on orthogonality potential):
  1. HT_TRENDMODE — Hilbert trend/cycle mode detection (regime signal)
  2. ULTOSC — Ultimate Oscillator (multi-timeframe momentum)
  3. NATR — Normalized ATR (volatility as % of price)
  4. CCI — Commodity Channel Index (statistical deviation)
  5. CMO — Chande Momentum Oscillator (pure momentum)
  6. MFI — Money Flow Index (volume-weighted RSI)
  7. PPO — Percentage Price Oscillator (normalized MACD)
  8. WILLR — Williams %R (normalized high-low position)
  9. ADX — Average Directional Index (trend strength)
  10. STOCHRSI — StochRSI (sensitive overbought/oversold)

Each factor is tested:
  A. Alone as single factor
  B. Combined with V15 baseline (BWP_BNW + HAR_RV + R_SQUARED + SMA_DEV)
  C. Best combos with ATR sweep

NO LOOK-AHEAD: All TA-Lib functions use data up to di-1 only.
  - For each stock, compute TA-Lib on full price history
  - Store the value at index di (computed from data up to di-1)
  - Cross-sectional rank normalize to [0, 100]
"""
import sys, os, time, warnings
import numpy as np
import talib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN
from alpha_v7 import compute_all_factors
from alpha_v7b import compute_interaction_factors
from alpha_v7d import compute_extra_factors
from alpha_v7e import compute_v7e_factors
from alpha_v7f import compute_advanced_interactions
from alpha_v8 import compute_v8_factors, compute_v8_interactions
from alpha_v9 import compute_v9_factors, compute_v9_interactions
from alpha_v10 import compute_v10_factors, compute_v10_interactions
from alpha_v11 import compute_v11_factors, compute_v11_interactions
from alpha_v14 import compute_v14_factors, compute_v14_interactions
from alpha_v7c import backtest_v7c


def compute_talib_factors(NS, ND, C, O, H, L, V):
    """Compute TA-Lib factors for all stocks.

    NO LOOK-AHEAD: Each indicator at index di uses only OHLCV data
    from indices 0..di-1. TA-Lib functions naturally handle this since
    they compute from the beginning of the array.
    """
    factors = {}
    t0 = time.time()

    # LOOK-AHEAD FIX: All TA-Lib factors use vals[di-1] stored at index di.
    # This means at rebalance day di, we use the indicator value computed from
    # data up to di-1 (yesterday), NOT including today's OHLCV.
    # Without this shift, BOP would use today's close to decide today's trade = look-ahead.

    # === HT_TRENDMODE: Hilbert Trend/Cycle Mode ===
    # Returns: 1 = trend mode, 0 = cycle mode
    print("  Computing HT_TRENDMODE...", flush=True)
    ht_trendmode = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        valid = ~np.isnan(c)
        if valid.sum() < 50:
            continue
        try:
            vals = talib.HT_TRENDMODE(c)
            for di in range(1, len(vals)):
                if di < ND and not np.isnan(vals[di - 1]):
                    ht_trendmode[si, di] = float(vals[di - 1])
        except:
            pass
    factors['HT_TRENDMODE'] = ht_trendmode
    print(f"    HT_TRENDMODE done ({time.time()-t0:.0f}s)", flush=True)

    # === ULTOSC: Ultimate Oscillator (7/14/28 period) ===
    # Range: 0-100. <30 oversold, >70 overbought
    print("  Computing ULTOSC...", flush=True)
    ultosc = np.full((NS, ND), np.nan)
    for si in range(NS):
        h, l, c = H[si], L[si], C[si]
        valid = ~(np.isnan(h) | np.isnan(l) | np.isnan(c))
        if valid.sum() < 50:
            continue
        try:
            vals = talib.ULTOSC(h, l, c, timeperiod1=7, timeperiod2=14, timeperiod3=28)
            for di in range(1, len(vals)):
                if di < ND and not np.isnan(vals[di - 1]):
                    ultosc[si, di] = float(vals[di - 1])
        except:
            pass
    factors['ULTOSC'] = ultosc
    print(f"    ULTOSC done ({time.time()-t0:.0f}s)", flush=True)

    # === NATR: Normalized ATR ===
    # ATR as percentage of close price
    print("  Computing NATR...", flush=True)
    natr = np.full((NS, ND), np.nan)
    for si in range(NS):
        h, l, c = H[si], L[si], C[si]
        valid = ~(np.isnan(h) | np.isnan(l) | np.isnan(c))
        if valid.sum() < 30:
            continue
        try:
            vals = talib.NATR(h, l, c, timeperiod=14)
            for di in range(1, len(vals)):
                if di < ND and not np.isnan(vals[di - 1]):
                    natr[si, di] = float(vals[di - 1])
        except:
            pass
    factors['NATR'] = natr
    print(f"    NATR done ({time.time()-t0:.0f}s)", flush=True)

    # === CCI: Commodity Channel Index ===
    # Measures deviation from statistical average
    print("  Computing CCI...", flush=True)
    cci = np.full((NS, ND), np.nan)
    for si in range(NS):
        h, l, c = H[si], L[si], C[si]
        valid = ~(np.isnan(h) | np.isnan(l) | np.isnan(c))
        if valid.sum() < 30:
            continue
        try:
            vals = talib.CCI(h, l, c, timeperiod=14)
            for di in range(1, len(vals)):
                if di < ND and not np.isnan(vals[di - 1]):
                    cci[si, di] = float(vals[di - 1])
        except:
            pass
    factors['CCI'] = cci
    print(f"    CCI done ({time.time()-t0:.0f}s)", flush=True)

    # === CMO: Chande Momentum Oscillator ===
    # Pure momentum: sum of up days - sum of down days
    print("  Computing CMO...", flush=True)
    cmo = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        valid = ~np.isnan(c)
        if valid.sum() < 30:
            continue
        try:
            vals = talib.CMO(c, timeperiod=14)
            for di in range(1, len(vals)):
                if di < ND and not np.isnan(vals[di - 1]):
                    cmo[si, di] = float(vals[di - 1])
        except:
            pass
    factors['CMO'] = cmo
    print(f"    CMO done ({time.time()-t0:.0f}s)", flush=True)

    # === MFI: Money Flow Index ===
    # Volume-weighted RSI equivalent
    print("  Computing MFI...", flush=True)
    mfi = np.full((NS, ND), np.nan)
    for si in range(NS):
        h, l, c, v = H[si], L[si], C[si], V[si]
        valid = ~(np.isnan(h) | np.isnan(l) | np.isnan(c) | np.isnan(v))
        if valid.sum() < 30:
            continue
        try:
            vals = talib.MFI(h, l, c, v, timeperiod=14)
            for di in range(1, len(vals)):
                if di < ND and not np.isnan(vals[di - 1]):
                    mfi[si, di] = float(vals[di - 1])
        except:
            pass
    factors['MFI'] = mfi
    print(f"    MFI done ({time.time()-t0:.0f}s)", flush=True)

    # === PPO: Percentage Price Oscillator ===
    # Normalized MACD (fast=12, slow=26)
    print("  Computing PPO...", flush=True)
    ppo = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        valid = ~np.isnan(c)
        if valid.sum() < 30:
            continue
        try:
            vals = talib.PPO(c, fastperiod=12, slowperiod=26, matype=0)
            for di in range(1, len(vals)):
                if di < ND and not np.isnan(vals[di - 1]):
                    ppo[si, di] = float(vals[di - 1])
        except:
            pass
    factors['PPO'] = ppo
    print(f"    PPO done ({time.time()-t0:.0f}s)", flush=True)

    # === WILLR: Williams %R ===
    # Normalized position within N-day high-low range
    print("  Computing WILLR...", flush=True)
    willr = np.full((NS, ND), np.nan)
    for si in range(NS):
        h, l, c = H[si], L[si], C[si]
        valid = ~(np.isnan(h) | np.isnan(l) | np.isnan(c))
        if valid.sum() < 30:
            continue
        try:
            vals = talib.WILLR(h, l, c, timeperiod=14)
            for di in range(1, len(vals)):
                if di < ND and not np.isnan(vals[di - 1]):
                    willr[si, di] = float(vals[di - 1])
        except:
            pass
    factors['WILLR'] = willr
    print(f"    WILLR done ({time.time()-t0:.0f}s)", flush=True)

    # === ADX: Average Directional Index ===
    # Trend strength (regardless of direction)
    print("  Computing ADX...", flush=True)
    adx = np.full((NS, ND), np.nan)
    for si in range(NS):
        h, l, c = H[si], L[si], C[si]
        valid = ~(np.isnan(h) | np.isnan(l) | np.isnan(c))
        if valid.sum() < 30:
            continue
        try:
            vals = talib.ADX(h, l, c, timeperiod=14)
            for di in range(1, len(vals)):
                if di < ND and not np.isnan(vals[di - 1]):
                    adx[si, di] = float(vals[di - 1])
        except:
            pass
    factors['ADX'] = adx
    print(f"    ADX done ({time.time()-t0:.0f}s)", flush=True)

    # === STOCHRSI: Stochastic RSI ===
    # RSI of RSI, very sensitive
    print("  Computing STOCHRSI...", flush=True)
    stochrsi_k = np.full((NS, ND), np.nan)
    stochrsi_d = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        valid = ~np.isnan(c)
        if valid.sum() < 30:
            continue
        try:
            k, d = talib.STOCHRSI(c, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
            for di in range(1, len(k)):
                if di < ND:
                    if not np.isnan(k[di - 1]):
                        stochrsi_k[si, di] = float(k[di - 1])
                    if not np.isnan(d[di - 1]):
                        stochrsi_d[si, di] = float(d[di - 1])
        except:
            pass
    factors['STOCHRSI_K'] = stochrsi_k
    factors['STOCHRSI_D'] = stochrsi_d
    print(f"    STOCHRSI done ({time.time()-t0:.0f}s)", flush=True)

    # === AROONOSC: Aroon Oscillator ===
    # Time since high vs time since low
    print("  Computing AROONOSC...", flush=True)
    aroonosc = np.full((NS, ND), np.nan)
    for si in range(NS):
        h, l = H[si], L[si]
        valid = ~(np.isnan(h) | np.isnan(l))
        if valid.sum() < 30:
            continue
        try:
            vals = talib.AROONOSC(h, l, timeperiod=14)
            for di in range(1, len(vals)):
                if di < ND and not np.isnan(vals[di - 1]):
                    aroonosc[si, di] = float(vals[di - 1])
        except:
            pass
    factors['AROONOSC'] = aroonosc
    print(f"    AROONOSC done ({time.time()-t0:.0f}s)", flush=True)

    # === BOP: Balance of Power ===
    # (Close - Open) / (High - Low)
    print("  Computing BOP...", flush=True)
    bop = np.full((NS, ND), np.nan)
    for si in range(NS):
        o, h, l, c = O[si], H[si], L[si], C[si]
        valid = ~(np.isnan(o) | np.isnan(h) | np.isnan(l) | np.isnan(c))
        if valid.sum() < 10:
            continue
        try:
            vals = talib.BOP(o, h, l, c)
            for di in range(1, len(vals)):
                if di < ND and not np.isnan(vals[di - 1]):
                    bop[si, di] = float(vals[di - 1])
        except:
            pass
    factors['BOP'] = bop
    print(f"    BOP done ({time.time()-t0:.0f}s)", flush=True)

    # === TRIX: Triple EMA Rate of Change ===
    # Filters out noise, shows underlying trend
    print("  Computing TRIX...", flush=True)
    trix = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        valid = ~np.isnan(c)
        if valid.sum() < 30:
            continue
        try:
            vals = talib.TRIX(c, timeperiod=14)
            for di in range(1, len(vals)):
                if di < ND and not np.isnan(vals[di - 1]):
                    trix[si, di] = float(vals[di - 1])
        except:
            pass
    factors['TRIX'] = trix
    print(f"    TRIX done ({time.time()-t0:.0f}s)", flush=True)

    # === HT_DCPERIOD: Dominant Cycle Period ===
    # Market microstructure: current cycle length
    print("  Computing HT_DCPERIOD...", flush=True)
    ht_dcperiod = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si]
        valid = ~np.isnan(c)
        if valid.sum() < 50:
            continue
        try:
            vals = talib.HT_DCPERIOD(c)
            for di in range(1, len(vals)):
                if di < ND and not np.isnan(vals[di - 1]):
                    ht_dcperiod[si, di] = float(vals[di - 1])
        except:
            pass
    factors['HT_DCPERIOD'] = ht_dcperiod
    print(f"    HT_DCPERIOD done ({time.time()-t0:.0f}s)", flush=True)

    # Now rank normalize each factor cross-sectionally
    print("  Rank normalizing TA-Lib factors...", flush=True)
    ranked = {}
    for fname, arr in factors.items():
        r = np.full_like(arr, np.nan)
        for di in range(ND):
            vals = arr[:, di]
            valid = ~np.isnan(vals)
            n = valid.sum()
            if n < 50:
                continue
            order = np.argsort(vals[valid])
            ranks = np.empty(n)
            ranks[order] = np.arange(1, n + 1)
            r[valid, di] = ranks / n * 100
        ranked[f'R_{fname}'] = r
    # For NATR, inverse (low vol = better)
    if 'R_NATR' in ranked:
        inv = ranked['R_NATR'].copy()
        mask = ~np.isnan(inv)
        inv[mask] = 100.0 - inv[mask]
        ranked['R_NATR_INV'] = inv
    # For WILLR, inverse (WILLR is -100 to 0, high = overbought)
    # Actually rank normalized is fine, high WILLR rank = overbought
    # We want oversold stocks, so invert
    if 'R_WILLR' in ranked:
        inv = ranked['R_WILLR'].copy()
        mask = ~np.isnan(inv)
        inv[mask] = 100.0 - inv[mask]
        ranked['R_WILLR_INV'] = inv

    print(f"  TA-Lib factors done: {len(ranked)} ranked factors ({time.time()-t0:.0f}s)", flush=True)
    return ranked


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("  Alpha V39 — TA-Lib Factor Exploration", flush=True)
    print("  14 new TA-Lib indicators as cross-sectional factors", flush=True)
    print("=" * 70, flush=True)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data()

    # Load existing factors
    base_factors = compute_all_factors(NS, ND, C, O, H, L, V)
    inter_factors = compute_interaction_factors(base_factors, NS, ND, C, O, H, L, V)
    extra_factors = compute_extra_factors(NS, ND, C, O, H, L, V)
    v7e_factors = compute_v7e_factors(NS, ND, C, O, H, L, V)
    adv_inter = compute_advanced_interactions(
        {**base_factors, **inter_factors, **extra_factors, **v7e_factors}, NS, ND)
    v8_factors = compute_v8_factors(NS, ND, C, O, H, L, V)
    v8_all = {**base_factors, **inter_factors, **extra_factors,
              **v7e_factors, **adv_inter, **v8_factors}
    v8_inter = compute_v8_interactions(v8_all, NS, ND)
    v8_all.update(v8_inter)
    v9_factors = compute_v9_factors(NS, ND, C, O, H, L, V)
    v9_all = {**v8_all, **v9_factors}
    v9_inter = compute_v9_interactions(v9_all, NS, ND)
    v9_all.update(v9_inter)
    v10_factors = compute_v10_factors(NS, ND, C, O, H, L, V)
    v10_all = {**v9_all, **v10_factors}
    v10_inter = compute_v10_interactions(v10_all, NS, ND)
    v10_all.update(v10_inter)
    v11_factors = compute_v11_factors(NS, ND, C, O, H, L, V)
    v11_all = {**v10_all, **v11_factors}
    v11_inter = compute_v11_interactions(v11_all, NS, ND)
    v11_all.update(v11_inter)
    v14_factors = compute_v14_factors(NS, ND, C, O, H, L, V)
    all_factors = {**v11_all, **v14_factors}
    v14_inter = compute_v14_interactions(all_factors, NS, ND)
    all_factors.update(v14_inter)

    # Compute TA-Lib factors
    talib_factors = compute_talib_factors(NS, ND, C, O, H, L, V)
    all_factors.update(talib_factors)

    print(f"\n  Total factors (incl TA-Lib): {len(all_factors)}", flush=True)

    # V15 baseline weights
    v15_weights = {'R_BWP_BNW': 0.3, 'R_HAR_RV_RATIO_INV': 0.3,
                   'R_R_SQUARED': 0.2, 'R_SMA_DEV': 0.2}

    results = []

    # =====================================================================
    # TEST 1: Each TA-Lib factor ALONE as single factor
    # =====================================================================
    print("\n  Test 1: Single TA-Lib factors...", flush=True)
    talib_names = sorted(talib_factors.keys())
    for fname in talib_names:
        for atr in [1.0, 1.2]:
            r = backtest_v7c({fname: 1.0}, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'{fname}_A{atr}'
                results.append(r)
    print(f"  Single factors done: {len(results)} results", flush=True)

    # =====================================================================
    # TEST 2: Each TA-Lib factor + V15 baseline
    # =====================================================================
    print("\n  Test 2: TA-Lib + V15 baseline...", flush=True)
    for fname in talib_names:
        for atr in [1.0, 1.2]:
            weights = {**v15_weights, fname: 0.15}
            # Renormalize
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
            r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                            top_n=1, rebalance_days=5, atr_stop_mult=atr)
            if r:
                r['test'] = f'V15+{fname}_A{atr}'
                results.append(r)
    print(f"  V15 combos done: {len(results)} total", flush=True)

    # =====================================================================
    # TEST 3: Best TA-Lib pairs (multi-TA-Lib factor combos)
    # =====================================================================
    print("\n  Test 3: TA-Lib pairs...", flush=True)
    # Test promising pairs based on theory
    pairs = [
        ('R_HT_TRENDMODE', 'R_NATR_INV'),     # Trend + low vol
        ('R_ULTOSC', 'R_CMO'),                   # Multi-timeframe momentum
        ('R_CCI', 'R_MFI'),                      # Statistical + volume
        ('R_PPO', 'R_ADX'),                      # Trend strength + momentum
        ('R_STOCHRSI_K', 'R_AROONOSC'),          # Sensitive momentum
        ('R_HT_TRENDMODE', 'R_ADX'),             # Regime + trend strength
        ('R_BOP', 'R_MFI'),                      # Volume + price balance
        ('R_NATR_INV', 'R_ADX'),                  # Low vol + trend
    ]
    for f1, f2 in pairs:
        if f1 in all_factors and f2 in all_factors:
            for atr in [1.0, 1.2]:
                weights = {'R_BWP_BNW': 0.25, 'R_HAR_RV_RATIO_INV': 0.25,
                           f1: 0.25, f2: 0.25}
                r = backtest_v7c(weights, all_factors, NS, ND, dates, C, O, H, L, V,
                                top_n=1, rebalance_days=5, atr_stop_mult=atr)
                if r:
                    tag = f'{f1[2:5]}{f2[2:5]}'
                    r['test'] = f'P_{tag}_A{atr}'
                    results.append(r)
    print(f"  Pairs done: {len(results)} total", flush=True)

    # =====================================================================
    # RESULTS
    # =====================================================================
    results.sort(key=lambda x: -x['ann'])

    def all_positive(r):
        ys = r.get('year_stats', {})
        return all(s['total_pnl'] > 0 for s in ys.values())

    print(f"\n{'='*120}", flush=True)
    print(f"  TOP 40 RESULTS (V39 TA-LIB FACTORS)", flush=True)
    print(f"  {'Test':<35s} | {'Ann':>7s} {'N':>5s} {'WR':>5s} {'Edge':>6s} {'DD':>5s}", flush=True)
    print(f"  {'-'*80}", flush=True)
    for r in results[:40]:
        pos_mark = " ALL+" if all_positive(r) else ""
        print(f"  {r['test']:<35s} | {r['ann']:+7.1f}% {r['n']:5d} {r['wr']:5.1f}% "
              f"{r['edge']:+6.2f}% {r['max_dd']:5.1f}%{pos_mark}", flush=True)

    # Best single TA-Lib factor
    singles = [r for r in results if not r['test'].startswith('V15+') and not r['test'].startswith('P_')]
    if singles:
        print(f"\n  Best single TA-Lib factors:", flush=True)
        for r in sorted(singles, key=lambda x: -x['ann'])[:15]:
            pos = " ALL+" if all_positive(r) else ""
            print(f"    {r['test']:<35s} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}%{pos}", flush=True)

    # Best V15 + TA-Lib combo
    combos = [r for r in results if r['test'].startswith('V15+')]
    if combos:
        print(f"\n  Best V15 + TA-Lib combos:", flush=True)
        for r in sorted(combos, key=lambda x: -x['ann'])[:10]:
            pos = " ALL+" if all_positive(r) else ""
            delta = r['ann'] - 235.6
            print(f"    {r['test']:<35s} → {r['ann']:+.1f}% DD={r['max_dd']:.1f}% Δ={delta:+.1f}%{pos}", flush=True)

    # Top 5 year-by-year
    for i, r in enumerate(results[:5]):
        print(f"\n  Year-by-year #{i+1}: {r['test']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['max_dd']:.1f}%)", flush=True)
        for y in sorted(r.get('year_stats', {}).keys()):
            s = r['year_stats'][y]
            wr = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr:.0f}%, pnl={s['total_pnl']:+.0f}%", flush=True)

    # V15 baseline comparison
    if results:
        best = results[0]
        print(f"\n  === V39 BEST vs V15 BASELINE ===", flush=True)
        print(f"  V39: {best['test']} = {best['ann']:+.1f}% DD={best['max_dd']:.1f}%", flush=True)
        print(f"  V15: HAR_RV_T1_A1.0 = +235.6% DD=32.4%", flush=True)
        print(f"  Delta: {best['ann'] - 235.6:+.1f}%", flush=True)

    print(f"\n{'='*70}", flush=True)
