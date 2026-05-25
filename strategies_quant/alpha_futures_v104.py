"""
Alpha Futures V104 -- TA-Lib Trend + Momentum Indicators
=========================================================
Tests 11 standalone TA-Lib indicator signals with next-open execution.

Indicators:
  A) ADX (trend strength + direction)
  B) CCI (momentum crossover)
  C) WILLR (oversold reversal)
  D) STOCHASTIC (%K/%D crossover in oversold zone)
  E) CMO (momentum zero-cross)
  F) ULTOSC (oversold reversal)
  G) ROC (rate-of-change zero-cross, multiple periods)
  H) TRIX (triple-EMA ROC zero-cross)
  I) COMBINED (ADX + ROC + PLUS_DI triple confirmation)
  J) PARABOLIC SAR (bullish flip)
  K) HT_TRENDMODE (Hilbert trend + SMA filter)

ALL signals: computed at close of day di, entry at O[si, di+1], exit at C[si, di+hold].
Walk-forward: top configs across 2020-2025.
"""
import sys, os, time, warnings
import numpy as np
import talib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

# ============================================================
# CONSTANTS
# ============================================================
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
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def nan_to_num(arr, fill=0.0):
    """Replace NaN with fill for talib input."""
    out = arr.copy()
    mask = np.isnan(out)
    if np.any(mask):
        # forward-fill then backfill, then fill remaining with 0
        for i in range(1, len(out)):
            if mask[i] and not mask[i-1]:
                out[i] = out[i-1]
                mask[i] = False
        if mask[0] and len(out) > 1:
            first_valid = np.where(~mask)[0]
            if len(first_valid) > 0:
                out[0] = out[first_valid[0]]
                mask[0] = False
        out[mask] = fill
    return out


def main():
    print("=" * 150)
    print("Alpha Futures V104 -- TA-Lib Trend + Momentum Indicators (Next-Open Execution)")
    print("=" * 150)
    print("\n  Testing 11 TA-Lib indicators: ADX, CCI, WILLR, STOCH, CMO, ULTOSC, ROC, TRIX,")
    print("  COMBINED, SAR, HT_TRENDMODE -- each as standalone signal with multiple holds")
    print("  ALL signals computed at close di, entry at O[si, di+1] (NEXT DAY OPEN)")

    # -- Load data --
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ================================================================
    # PRECOMPUTE ALL TA-LIB INDICATORS PER COMMODITY
    # ================================================================
    print("\n[Indicators] Computing TA-Lib indicators for all commodities...", flush=True)
    t0 = time.time()

    # Storage: [NS, ND] arrays for each indicator
    adx_val = np.full((NS, ND), np.nan)
    plus_di = np.full((NS, ND), np.nan)
    minus_di = np.full((NS, ND), np.nan)

    cci_val = np.full((NS, ND), np.nan)

    willr_val = np.full((NS, ND), np.nan)

    slowk_val = np.full((NS, ND), np.nan)
    slowd_val = np.full((NS, ND), np.nan)

    cmo_val = np.full((NS, ND), np.nan)

    ultosc_val = np.full((NS, ND), np.nan)

    roc_5 = np.full((NS, ND), np.nan)
    roc_10 = np.full((NS, ND), np.nan)
    roc_20 = np.full((NS, ND), np.nan)

    trix_val = np.full((NS, ND), np.nan)

    sar_val = np.full((NS, ND), np.nan)

    ht_trendmode = np.full((NS, ND), np.nan)

    sma_20 = np.full((NS, ND), np.nan)

    for si in range(NS):
        c = nan_to_num(C[si])
        h = nan_to_num(H[si])
        l = nan_to_num(L[si])
        o = nan_to_num(O[si])
        v = nan_to_num(V[si], fill=1.0)

        valid_mask = ~np.isnan(C[si])

        # A) ADX system
        try:
            _adx = talib.ADX(h, l, c, timeperiod=14)
            _pdi = talib.PLUS_DI(h, l, c, timeperiod=14)
            _mdi = talib.MINUS_DI(h, l, c, timeperiod=14)
            adx_val[si] = np.where(valid_mask, _adx, np.nan)
            plus_di[si] = np.where(valid_mask, _pdi, np.nan)
            minus_di[si] = np.where(valid_mask, _mdi, np.nan)
        except Exception:
            pass

        # B) CCI
        try:
            _cci = talib.CCI(h, l, c, timeperiod=14)
            cci_val[si] = np.where(valid_mask, _cci, np.nan)
        except Exception:
            pass

        # C) WILLR
        try:
            _wr = talib.WILLR(h, l, c, timeperiod=14)
            willr_val[si] = np.where(valid_mask, _wr, np.nan)
        except Exception:
            pass

        # D) STOCHASTIC
        try:
            _sk, _sd = talib.STOCH(h, l, c,
                                    fastk_period=14,
                                    slowk_period=3,
                                    slowk_matype=0,
                                    slowd_period=3,
                                    slowd_matype=0)
            slowk_val[si] = np.where(valid_mask, _sk, np.nan)
            slowd_val[si] = np.where(valid_mask, _sd, np.nan)
        except Exception:
            pass

        # E) CMO
        try:
            _cmo = talib.CMO(c, timeperiod=14)
            cmo_val[si] = np.where(valid_mask, _cmo, np.nan)
        except Exception:
            pass

        # F) ULTOSC
        try:
            _ult = talib.ULTOSC(h, l, c,
                                timeperiod1=7, timeperiod2=14, timeperiod3=28)
            ultosc_val[si] = np.where(valid_mask, _ult, np.nan)
        except Exception:
            pass

        # G) ROC
        try:
            roc_5[si] = np.where(valid_mask, talib.ROC(c, timeperiod=5), np.nan)
            roc_10[si] = np.where(valid_mask, talib.ROC(c, timeperiod=10), np.nan)
            roc_20[si] = np.where(valid_mask, talib.ROC(c, timeperiod=20), np.nan)
        except Exception:
            pass

        # H) TRIX
        try:
            _trix = talib.TRIX(c, timeperiod=30)
            trix_val[si] = np.where(valid_mask, _trix, np.nan)
        except Exception:
            pass

        # J) SAR
        try:
            _sar = talib.SAR(h, l, acceleration=0.02, maximum=0.2)
            sar_val[si] = np.where(valid_mask, _sar, np.nan)
        except Exception:
            pass

        # K) HT_TRENDMODE
        try:
            _htm = talib.HT_TRENDMODE(c)
            ht_trendmode[si] = np.where(valid_mask, _htm, np.nan)
        except Exception:
            pass

        # SMA_20 for HT filter
        try:
            _sma = talib.SMA(c, timeperiod=20)
            sma_20[si] = np.where(valid_mask, _sma, np.nan)
        except Exception:
            pass

    print(f"  All indicators computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # BACKTEST ENGINE (reuse from V99 pattern)
    # ================================================================
    def run_backtest(config, wf_test_year=None):
        sig_type = config['signal']
        hold_days = config['hold_days']
        threshold = config.get('threshold', 0)
        top_n = config['top_n']
        comm = config.get('comm', COMM)

        # Date boundaries
        if wf_test_year is not None:
            test_start_di = None
            test_end_di = None
            for di in range(ND):
                if dates[di].year == wf_test_year and test_start_di is None:
                    test_start_di = di
                if dates[di].year == wf_test_year + 1 and test_end_di is None:
                    test_end_di = di
            if test_start_di is None:
                return None
            if test_end_di is None:
                test_end_di = ND
            start_di = MIN_TRAIN
            end_di = test_end_di
        else:
            test_start_di = MIN_TRAIN
            start_di = MIN_TRAIN
            end_di = ND
            test_end_di = ND

        if end_di < start_di + hold_days + 2:
            return None

        cash = float(CASH0)
        positions = []
        trades = []

        for di in range(start_di, end_di - 1):
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                positions = []

            # -- Close positions held long enough --
            closed = []
            for pos in positions:
                days_held = di - pos['entry_di']
                if days_held >= pos['hold_days']:
                    exit_price = C[pos['si'], di]
                    if np.isnan(exit_price) or exit_price <= 0:
                        exit_price = pos['entry_price']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    mkt_val = exit_price * mult * abs(pos['lots'])
                    cash += mkt_val - mkt_val * comm
                    pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
                    invested = pos['entry_price'] * mult * abs(pos['lots'])
                    pnl_pct = pnl / invested * 100 if invested > 0 else 0
                    trades.append({
                        'pnl_pct': pnl_pct,
                        'entry_di': pos['entry_di'],
                        'exit_di': di,
                        'year': dates[di].year if di < ND else dates[-1].year,
                    })
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            # -- Generate signals at day di --
            entry_di = di + 1
            if entry_di >= end_di:
                continue

            candidates = []  # (score, info_dict)

            if sig_type == 'adx':
                # ADX > threshold AND PLUS_DI > MINUS_DI
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    a = adx_val[si, di]
                    p_di = plus_di[si, di]
                    m_di = minus_di[si, di]
                    if np.isnan(a) or np.isnan(p_di) or np.isnan(m_di):
                        continue
                    if a > threshold and p_di > m_di:
                        ep = O[si, entry_di]
                        if np.isnan(ep) or ep <= 0:
                            continue
                        candidates.append((a, {'si': si, 'sym': syms[si], 'entry_price': ep}))

            elif sig_type == 'cci_cross':
                # CCI crosses above +100
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    cc = cci_val[si, di]
                    cc_prev = cci_val[si, di - 1] if di > 0 else np.nan
                    if np.isnan(cc) or np.isnan(cc_prev):
                        continue
                    if cc > 100 and cc_prev < 100:
                        ep = O[si, entry_di]
                        if np.isnan(ep) or ep <= 0:
                            continue
                        candidates.append((cc - 100, {'si': si, 'sym': syms[si], 'entry_price': ep}))

            elif sig_type == 'willr_cross':
                # WILLR crosses above -80 from below
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    wr = willr_val[si, di]
                    wr_prev = willr_val[si, di - 1] if di > 0 else np.nan
                    if np.isnan(wr) or np.isnan(wr_prev):
                        continue
                    if wr > -80 and wr_prev < -80:
                        ep = O[si, entry_di]
                        if np.isnan(ep) or ep <= 0:
                            continue
                        candidates.append((wr + 80, {'si': si, 'sym': syms[si], 'entry_price': ep}))

            elif sig_type == 'stoch_cross':
                # %K crosses above %D, both in oversold zone (%K < 25)
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    sk = slowk_val[si, di]
                    sd = slowd_val[si, di]
                    sk_prev = slowk_val[si, di - 1] if di > 0 else np.nan
                    sd_prev = slowd_val[si, di - 1] if di > 0 else np.nan
                    if np.isnan(sk) or np.isnan(sd) or np.isnan(sk_prev) or np.isnan(sd_prev):
                        continue
                    if sk > sd and sk_prev <= sd_prev and sk < 25:
                        ep = O[si, entry_di]
                        if np.isnan(ep) or ep <= 0:
                            continue
                        candidates.append((25 - sk, {'si': si, 'sym': syms[si], 'entry_price': ep}))

            elif sig_type == 'cmo_cross':
                # CMO crosses above 0
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    cv = cmo_val[si, di]
                    cv_prev = cmo_val[si, di - 1] if di > 0 else np.nan
                    if np.isnan(cv) or np.isnan(cv_prev):
                        continue
                    if cv > 0 and cv_prev < 0:
                        ep = O[si, entry_di]
                        if np.isnan(ep) or ep <= 0:
                            continue
                        candidates.append((cv, {'si': si, 'sym': syms[si], 'entry_price': ep}))

            elif sig_type == 'ultosc_cross':
                # ULTOSC crosses above 30 from below
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    uv = ultosc_val[si, di]
                    uv_prev = ultosc_val[si, di - 1] if di > 0 else np.nan
                    if np.isnan(uv) or np.isnan(uv_prev):
                        continue
                    if uv > 30 and uv_prev <= 30:
                        ep = O[si, entry_di]
                        if np.isnan(ep) or ep <= 0:
                            continue
                        candidates.append((uv - 30, {'si': si, 'sym': syms[si], 'entry_price': ep}))

            elif sig_type == 'roc_cross':
                # ROC crosses above 0 (period from threshold)
                roc_period = config.get('roc_period', 10)
                roc_arr = {5: roc_5, 10: roc_10, 20: roc_20}[roc_period]
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    rv = roc_arr[si, di]
                    rv_prev = roc_arr[si, di - 1] if di > 0 else np.nan
                    if np.isnan(rv) or np.isnan(rv_prev):
                        continue
                    if rv > 0 and rv_prev < 0:
                        ep = O[si, entry_di]
                        if np.isnan(ep) or ep <= 0:
                            continue
                        candidates.append((rv, {'si': si, 'sym': syms[si], 'entry_price': ep}))

            elif sig_type == 'trix_cross':
                # TRIX crosses above 0
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    tv = trix_val[si, di]
                    tv_prev = trix_val[si, di - 1] if di > 0 else np.nan
                    if np.isnan(tv) or np.isnan(tv_prev):
                        continue
                    if tv > 0 and tv_prev <= 0:
                        ep = O[si, entry_di]
                        if np.isnan(ep) or ep <= 0:
                            continue
                        candidates.append((tv, {'si': si, 'sym': syms[si], 'entry_price': ep}))

            elif sig_type == 'combined':
                # ADX > 25 + ROC_10 > 0 + PLUS_DI > MINUS_DI
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    a = adx_val[si, di]
                    p_di = plus_di[si, di]
                    m_di = minus_di[si, di]
                    rv = roc_10[si, di]
                    if np.isnan(a) or np.isnan(p_di) or np.isnan(m_di) or np.isnan(rv):
                        continue
                    if a > 25 and p_di > m_di and rv > 0:
                        ep = O[si, entry_di]
                        if np.isnan(ep) or ep <= 0:
                            continue
                        score = a + rv  # higher ADX + ROC = stronger
                        candidates.append((score, {'si': si, 'sym': syms[si], 'entry_price': ep}))

            elif sig_type == 'sar_flip':
                # SAR flips below price (bullish): SAR[di] < C[di] AND SAR[di-1] >= C[di-1]
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    s = sar_val[si, di]
                    s_prev = sar_val[si, di - 1] if di > 0 else np.nan
                    c_cur = C[si, di]
                    c_prev = C[si, di - 1] if di > 0 else np.nan
                    if np.isnan(s) or np.isnan(s_prev) or np.isnan(c_cur) or np.isnan(c_prev):
                        continue
                    if s < c_cur and s_prev >= c_prev:
                        ep = O[si, entry_di]
                        if np.isnan(ep) or ep <= 0:
                            continue
                        candidates.append((c_cur - s, {'si': si, 'sym': syms[si], 'entry_price': ep}))

            elif sig_type == 'ht_trend':
                # HT_TRENDMODE = 1 (trending) AND price > SMA_20
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    ht = ht_trendmode[si, di]
                    sm = sma_20[si, di]
                    c_cur = C[si, di]
                    if np.isnan(ht) or np.isnan(sm) or np.isnan(c_cur):
                        continue
                    if ht >= 1 and c_cur > sm:
                        ep = O[si, entry_di]
                        if np.isnan(ep) or ep <= 0:
                            continue
                        pct_above = (c_cur - sm) / sm * 100 if sm > 0 else 0
                        candidates.append((pct_above, {'si': si, 'sym': syms[si], 'entry_price': ep}))

            if not candidates:
                continue

            # Sort by score descending
            candidates.sort(key=lambda x: -x[0])

            # Open positions (long only)
            n_slots = top_n - len(positions)
            for score, info in candidates[:max(0, n_slots)]:
                si = info['si']
                sym = info['sym']
                price = info['entry_price']
                mult = MULT.get(sym, DEF_MULT)
                notional = price * mult
                lots = int(cash / (notional * (1 + comm) * top_n))
                if lots <= 0:
                    lots = int(cash * 0.9 / (notional * (1 + comm)))
                if lots <= 0:
                    continue
                cost_in = notional * lots * (1 + comm)
                if cost_in > cash:
                    lots = int(cash * 0.85 / (notional * (1 + comm)))
                    cost_in = notional * lots * (1 + comm) if lots > 0 else 0
                if lots <= 0 or cost_in <= 0 or cost_in > cash:
                    continue

                cash -= cost_in
                positions.append({
                    'si': si, 'entry_price': price, 'entry_di': entry_di,
                    'lots': lots, 'sym': sym, 'hold_days': hold_days,
                })

        # Close remaining positions at end
        for pos in positions:
            ae = end_di - 1 if end_di < ND else ND - 1
            exit_price = C[pos['si'], ae]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos['entry_price']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = exit_price * mult * abs(pos['lots'])
            cash += mkt_val - mkt_val * comm

        # Results
        n_days_test = (test_end_di - test_start_di) if wf_test_year is not None else (end_di - start_di)
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0

        # Max drawdown
        eq = float(CASH0)
        peak = eq
        mdd = 0.0
        for t in trades:
            eq *= (1 + t['pnl_pct'] / 100)
            if eq > peak:
                peak = eq
            dd = (eq - peak) / peak * 100
            if dd < mdd:
                mdd = dd

        return {
            'ann': ann, 'wr': wr, 'n': n_trades, 'avg_pnl': avg_pnl,
            'final_cash': cash, 'n_days': n_days_test, 'mdd': mdd,
        }

    # ================================================================
    # BUILD CONFIGURATIONS
    # ================================================================
    print("\n[Sweep] Building configurations...", flush=True)
    configs = []
    cid = 0

    # --- A: ADX with thresholds 20, 25, 30, 35, hold 5/10/20, top_n 3 ---
    for thresh in [20, 25, 30, 35]:
        for hd in [5, 10, 20]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'adx',
                'hold_days': hd, 'threshold': thresh,
                'top_n': 3, 'comm': COMM,
                'label': f"ADX>{thresh}_H{hd}_TN3",
            })

    # --- B: CCI crossover, hold 5/10/20, top_n 3 ---
    for hd in [5, 10, 20]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'cci_cross',
                'hold_days': hd, 'threshold': 100,
                'top_n': tn, 'comm': COMM,
                'label': f"CCI_Cross_H{hd}_TN{tn}",
            })

    # --- C: WILLR crossover, hold 5/10/20, top_n 3 ---
    for hd in [5, 10, 20]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'willr_cross',
                'hold_days': hd, 'threshold': -80,
                'top_n': tn, 'comm': COMM,
                'label': f"WILLR_Cross_H{hd}_TN{tn}",
            })

    # --- D: STOCH crossover, hold 5/10/20, top_n 3 ---
    for hd in [5, 10, 20]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'stoch_cross',
                'hold_days': hd, 'threshold': 25,
                'top_n': tn, 'comm': COMM,
                'label': f"STOCH_Cross_H{hd}_TN{tn}",
            })

    # --- E: CMO crossover, hold 5/10/20, top_n 3 ---
    for hd in [5, 10, 20]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'cmo_cross',
                'hold_days': hd, 'threshold': 0,
                'top_n': tn, 'comm': COMM,
                'label': f"CMO_Cross_H{hd}_TN{tn}",
            })

    # --- F: ULTOSC crossover, hold 5/10/20, top_n 3 ---
    for hd in [5, 10, 20]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'ultosc_cross',
                'hold_days': hd, 'threshold': 30,
                'top_n': tn, 'comm': COMM,
                'label': f"ULTOSC_Cross_H{hd}_TN{tn}",
            })

    # --- G: ROC crossover, periods 5/10/20, hold 5/10/20, top_n 3 ---
    for roc_p in [5, 10, 20]:
        for hd in [5, 10, 20]:
            for tn in [1, 3, 5]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': 'roc_cross',
                    'hold_days': hd, 'threshold': 0,
                    'top_n': tn, 'comm': COMM,
                    'roc_period': roc_p,
                    'label': f"ROC{roc_p}_Cross_H{hd}_TN{tn}",
                })

    # --- H: TRIX crossover, hold 5/10/20, top_n 3 ---
    for hd in [5, 10, 20]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'trix_cross',
                'hold_days': hd, 'threshold': 0,
                'top_n': tn, 'comm': COMM,
                'label': f"TRIX_Cross_H{hd}_TN{tn}",
            })

    # --- I: COMBINED ADX+ROC+PLUS_DI, hold 5/10/20, top_n 3 ---
    for hd in [5, 10, 20]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'combined',
                'hold_days': hd, 'threshold': 25,
                'top_n': tn, 'comm': COMM,
                'label': f"COMBINED_ADX25_H{hd}_TN{tn}",
            })

    # --- J: SAR flip, hold 5/10/20, top_n 3 ---
    for hd in [5, 10, 20]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'sar_flip',
                'hold_days': hd, 'threshold': 0,
                'top_n': tn, 'comm': COMM,
                'label': f"SAR_Flip_H{hd}_TN{tn}",
            })

    # --- K: HT_TRENDMODE + SMA20, hold 5/10/20, top_n 3 ---
    for hd in [5, 10, 20]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'ht_trend',
                'hold_days': hd, 'threshold': 1,
                'top_n': tn, 'comm': COMM,
                'label': f"HT_Trend_H{hd}_TN{tn}",
            })

    print(f"  Total configs: {len(configs)}")

    # ================================================================
    # RUN FULL-PERIOD BACKTEST
    # ================================================================
    print("\n[Backtest] Running full-period sweep...", flush=True)
    results = []
    for i, cfg in enumerate(configs):
        r = run_backtest(cfg)
        if r:
            r['config'] = cfg
            r['label'] = cfg['label']
            results.append(r)
        if (i + 1) % 50 == 0 or i == len(configs) - 1:
            print(f"  ... {i+1}/{len(configs)} done ({time.time()-t_start:.0f}s)", flush=True)

    results.sort(key=lambda x: -x['ann'])

    # ================================================================
    # FULL-PERIOD RESULTS (Top 30)
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  FULL-PERIOD RESULTS (Top 30) -- ALL NEXT-OPEN EXECUTION")
    print(f"{'=' * 150}")
    print(f"  {'#':>3} | {'Label':<35} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'Final':>14}")
    print("-" * 130)
    for i, r in enumerate(results[:30]):
        print(f"  {i+1:>3} | {r['label']:<35} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}% | {r['final_cash']:>13,.0f}")

    # ================================================================
    # BEST PER SIGNAL TYPE (full period)
    # ================================================================
    sig_order = ['adx', 'cci_cross', 'willr_cross', 'stoch_cross',
                 'cmo_cross', 'ultosc_cross', 'roc_cross', 'trix_cross',
                 'combined', 'sar_flip', 'ht_trend']
    sig_names = {
        'adx':         'A) ADX Trend Strength',
        'cci_cross':   'B) CCI Momentum Cross',
        'willr_cross': 'C) Williams %R Cross',
        'stoch_cross': 'D) Stochastic %K/%D Cross',
        'cmo_cross':   'E) CMO Zero-Cross',
        'ultosc_cross':'F) Ultimate Oscillator Cross',
        'roc_cross':   'G) ROC Zero-Cross',
        'trix_cross':  'H) TRIX Zero-Cross',
        'combined':    'I) Combined ADX+ROC+DI',
        'sar_flip':    'J) Parabolic SAR Flip',
        'ht_trend':    'K) HT TrendMode + SMA20',
    }

    print(f"\n{'=' * 150}")
    print("  BEST PER SIGNAL TYPE (Full Period) -- ALL NEXT-OPEN EXECUTION")
    print(f"{'=' * 150}")
    print(f"  {'Signal':<40} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | Best Config")
    print("-" * 150)

    best_per_sig = {}
    for r in results:
        key = r['config']['signal']
        if key not in best_per_sig:
            best_per_sig[key] = r

    for sig in sig_order:
        if sig in best_per_sig:
            b = best_per_sig[sig]
            print(f"  {sig_names.get(sig, sig):<40} | {b['ann']:>+8.1f}% | {b['wr']:>5.1f}% | {b['n']:>5} | {b['avg_pnl']:>+6.3f}% | {b['mdd']:>6.1f}% | {b['label']}")

    # ================================================================
    # SIGNAL TYPE SUMMARY
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  SIGNAL TYPE SUMMARY (Average of Top 5 configs per type)")
    print(f"{'=' * 150}")
    print(f"  {'Signal':<40} | {'Avg Ann':>9} | {'Avg WR':>7} | {'Avg N':>7} | {'Avg PnL':>8} | {'Avg MDD':>8} | {'#Positive':>9}")
    print("-" * 150)

    for sig in sig_order:
        sub = [r for r in results if r['config']['signal'] == sig]
        if not sub:
            continue
        top5 = sub[:5]
        avg_ann = np.mean([r['ann'] for r in top5])
        avg_wr = np.mean([r['wr'] for r in top5])
        avg_n = np.mean([r['n'] for r in top5])
        avg_pnl = np.mean([r['avg_pnl'] for r in top5])
        avg_mdd = np.mean([r['mdd'] for r in top5])
        n_pos = sum(1 for r in sub if r['ann'] > 0)
        print(f"  {sig_names.get(sig, sig):<40} | {avg_ann:>+8.1f}% | {avg_wr:>6.1f}% | {avg_n:>7.0f} | {avg_pnl:>+7.3f}% | {avg_mdd:>7.1f}% | {n_pos:>5}/{len(sub)}")

    # ================================================================
    # WALK-FORWARD (Top 15 configs + best per signal type)
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    wf_configs = list(results[:15])
    for sig in sig_order:
        if sig in best_per_sig:
            r = best_per_sig[sig]
            if r['config'] not in [w['config'] for w in wf_configs]:
                wf_configs.append(r)

    print(f"\n{'=' * 170}")
    print(f"  WALK-FORWARD ({len(wf_configs)} configs)")
    print(f"{'=' * 170}")

    header = f"  {'#':>3} | {'Config':<35} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>7} |"
    header += f" {'Pos':>4} | {'MDD':>7}"
    print(header)
    print("-" * 170)

    wf_rows = []
    for i, r in enumerate(wf_configs):
        cfg = r['config']
        wf_row = {'label': cfg['label'], 'signal': cfg['signal'],
                  'windows': {}, 'mdd': {}}
        for yr in wf_years:
            wr = run_backtest(cfg, wf_test_year=yr)
            if wr:
                wf_row['windows'][yr] = wr['ann']
                wf_row['mdd'][yr] = wr['mdd']
        wf_rows.append(wf_row)

        vals = [wf_row['windows'].get(yr, 0) for yr in wf_years]
        avg = np.mean(vals) if vals else 0
        pos = sum(1 for v in vals if v > 0)
        avg_mdd = np.mean(list(wf_row['mdd'].values())) if wf_row['mdd'] else 0

        row_str = f"  {i+1:>3} | {wf_row['label']:<35} | {avg:>+7.1f}% |"
        for v in vals:
            row_str += f" {v:>+7.1f}% |"
        row_str += f" {pos}/6 | {avg_mdd:>6.1f}%"
        print(row_str)

    # ================================================================
    # WF COMPARISON PER SIGNAL
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  WALK-FORWARD COMPARISON (Best per signal type)")
    print(f"{'=' * 150}")
    header2 = f"  {'Signal':<40} | {'WF Avg':>8} |"
    for yr in wf_years:
        header2 += f" {yr:>7} |"
    header2 += f" {'Pos':>4} | Avg MDD"
    print(header2)
    print("-" * 150)

    for sig in sig_order:
        wf_match = [w for w in wf_rows if w['signal'] == sig]
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            avg = np.mean(vals) if vals else 0
            pos = sum(1 for v in vals if v > 0)
            avg_mdd = np.mean(list(wf['mdd'].values())) if wf['mdd'] else 0
            row_str = f"  {sig_names.get(sig, sig):<40} | {avg:>+7.1f}% |"
            for v in vals:
                row_str += f" {v:>+7.1f}% |"
            row_str += f" {pos}/6 | {avg_mdd:>6.1f}%"
            print(row_str)

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  FINAL VERDICT: TA-Lib TREND + MOMENTUM INDICATORS WITH NEXT-OPEN EXECUTION")
    print(f"{'=' * 150}")
    print()

    for sig in sig_order:
        sub = [r for r in results if r['config']['signal'] == sig]
        if not sub:
            continue
        best = sub[0]
        n_pos = sum(1 for r in sub if r['ann'] > 0)
        avg_top5 = np.mean([r['ann'] for r in sub[:5]])

        # WF stats
        wf_match = [w for w in wf_rows if w['signal'] == sig]
        wf_pos = 0
        wf_avg = 0
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            wf_pos = sum(1 for v in vals if v > 0)
            wf_avg = np.mean(vals)

        verdict = "POSITIVE" if best['ann'] > 0 else "NEGATIVE"
        genuine = ("GENUINE ALPHA" if wf_pos >= 4 and best['ann'] > 0
                   else ("MARGINAL" if wf_pos >= 3 and best['ann'] > 0 else "NO ALPHA"))

        print(f"  {sig_names.get(sig, sig)}")
        print(f"    Best annual: {best['ann']:>+8.1f}%  |  Avg top-5: {avg_top5:>+8.1f}%  |  {n_pos}/{len(sub)} positive configs")
        print(f"    Walk-forward: {wf_pos}/6 positive  |  WF avg: {wf_avg:>+8.1f}%")
        print(f"    VERDICT: {verdict}  -->  {genuine}")
        print()

    # Overall best
    all_prac = [r for r in results]
    if all_prac:
        best_overall = all_prac[0]
        print(f"  BEST OVERALL STRATEGY (next-open execution):")
        print(f"    {best_overall['label']}")
        print(f"    Annual: {best_overall['ann']:>+8.1f}%")
        print(f"    WR:     {best_overall['wr']:>5.1f}%")
        print(f"    N:      {best_overall['n']:>5}")
        print(f"    MDD:    {best_overall['mdd']:>6.1f}%")
        print(f"    Final:  {best_overall['final_cash']:>13,.0f}")

        # Find best WF
        if wf_rows:
            best_wf = max(wf_rows[:15], key=lambda w: np.mean([w['windows'].get(yr, 0) for yr in wf_years]))
            wf_vals = [best_wf['windows'].get(yr, 0) for yr in wf_years]
            wf_avg = np.mean(wf_vals)
            wf_pos = sum(1 for v in wf_vals if v > 0)
            print(f"\n  BEST WALK-FORWARD STRATEGY:")
            print(f"    {best_wf['label']}")
            print(f"    WF Avg: {wf_avg:>+8.1f}%  |  {wf_pos}/6 positive windows")

        # Compare with benchmark
        print(f"\n  BENCHMARK COMPARISON:")
        print(f"    Current best practical: +49.6% (50-day Breakout), +37.2% (Vol Breakout), +31.6% (Bullish Engulfing)")
        beating = [r for r in results if r['ann'] > 49.6]
        print(f"    Configs beating +49.6%: {len(beating)}")
        if beating:
            print(f"    Best beating benchmark: {beating[0]['label']} at {beating[0]['ann']:>+8.1f}%")

    print(f"\n  Total time: {time.time()-t_start:.1f}s")
    print("=" * 150)


if __name__ == '__main__':
    main()
