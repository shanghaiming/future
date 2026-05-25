"""
Alpha Futures V126 -- NOVEL SIGNAL ENGINEERING (Next-Open Execution)
====================================================================
The current champion (+333.5%) uses ROC(5) + Z-score + ROC improving.
Try completely NEW signal constructions that might capture different alpha.

ALL signals use NEXT-OPEN execution: signal at close di, entry at O[si, di+1].

Tests A through L:
  A) Momentum Acceleration Index (MAI = ROC3/ROC10)
  B) Volatility-Adjusted Momentum (VMOM = ROC5/ATR14*100)
  C) Price Velocity + Acceleration (physics-inspired)
  D) Trend Purity (fraction of positive days in window)
  E) Asymmetric Momentum (avg_up / |avg_down|)
  F) Consecutive New Highs (5-day highs streak)
  G) Volume Momentum Concordance (dual rank)
  H) Overnight-Intraday Decomposition (both components positive)
  I) Entropy-Based Regime (Shannon entropy filter)
  J) Multi-Factor Composite v2 (novel factor blend)
  K) Champion + Trend Purity Filter
  L) Champion + Asymmetry Filter

Walk-forward by year (2020-2025).
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


def cross_sectional_percentile_arr(arr):
    """Precompute cross-sectional percentile ranks for all days.
    arr: (NS, ND) -> returns (NS, ND) of percentile ranks (0-100).
    """
    NS, ND = arr.shape
    ranks = np.full((NS, ND), np.nan)
    for di in range(ND):
        vals = arr[:, di]
        valid_mask = ~np.isnan(vals)
        valid = vals[valid_mask]
        n = len(valid)
        if n < 10:
            continue
        # Use argsort for efficient ranking
        order = np.argsort(valid)
        rank_vals = np.zeros(n)
        rank_vals[order] = np.arange(1, n + 1)
        pct = rank_vals / n * 100
        # Write back
        idx = 0
        for si in range(NS):
            if valid_mask[si]:
                ranks[si, di] = pct[idx]
                idx += 1
    return ranks


def main():
    print("=" * 150, flush=True)
    print("  Alpha Futures V126 -- NOVEL SIGNAL ENGINEERING (Next-Open Execution)", flush=True)
    print("=" * 150, flush=True)
    print(f"  Champion: +333.5% (ROC5>1% + Z>1.5 + ROC improving)")
    print(f"  Goal: Find novel signals that produce positive alpha or beat champion")

    # -- Load data -------------------------------------------------
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")
    print(f"  MIN_TRAIN={MIN_TRAIN}, CASH0={CASH0:,}")

    # ================================================================
    # PRECOMPUTE ALL INDICATORS
    # ================================================================
    print("\n[Precompute] Building all novel indicators...", flush=True)
    t0 = time.time()

    # --- Basic indicators ---
    # Daily returns (%)
    RET = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100

    # ROC(3), ROC(5), ROC(10), ATR(14), ADX(14) using TA-Lib
    ROC3 = np.full((NS, ND), np.nan)
    ROC5 = np.full((NS, ND), np.nan)
    ROC10 = np.full((NS, ND), np.nan)
    ATR14 = np.full((NS, ND), np.nan)
    ADX14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        h = H[si].astype(np.float64)
        l = L[si].astype(np.float64)
        ROC3[si] = talib.ROC(c, timeperiod=3)
        ROC5[si] = talib.ROC(c, timeperiod=5)
        ROC10[si] = talib.ROC(c, timeperiod=10)
        ATR14[si] = talib.ATR(h, l, c, timeperiod=14)
        ADX14[si] = talib.ADX(h, l, c, timeperiod=14)
    print(f"  ROC/ATR/ADX done ({time.time()-t0:.1f}s)", flush=True)

    # Z-score of daily returns (20-day rolling) -- champion uses this
    ZSCORE_20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = RET[si, di-20:di]
            valid = rets[~np.isnan(rets)]
            if len(valid) < 10:
                continue
            mean_r = np.mean(valid)
            std_r = np.std(valid, ddof=1)
            if std_r > 0 and not np.isnan(RET[si, di]):
                ZSCORE_20[si, di] = (RET[si, di] - mean_r) / std_r
    print(f"  ZSCORE_20 done ({time.time()-t0:.1f}s)", flush=True)

    # --- A) Momentum Acceleration Index ---
    # MAI = ROC(3) / ROC(10) (short-term vs medium-term momentum ratio)
    MAI = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            r3 = ROC3[si, di]
            r10 = ROC10[si, di]
            if not np.isnan(r3) and not np.isnan(r10) and r10 > 0:
                MAI[si, di] = r3 / r10
    print(f"  MAI done ({time.time()-t0:.1f}s)", flush=True)

    # --- B) Volatility-Adjusted Momentum ---
    # VMOM = ROC(5) / (ATR(14)/Close * 100) = momentum per unit of volatility
    VMOM = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            r5 = ROC5[si, di]
            atr = ATR14[si, di]
            c_val = C[si, di]
            if not np.isnan(r5) and not np.isnan(atr) and atr > 0 and not np.isnan(c_val) and c_val > 0:
                VMOM[si, di] = r5 / (atr / c_val * 100)
    print(f"  VMOM done ({time.time()-t0:.1f}s)", flush=True)

    # --- C) Price Velocity + Acceleration ---
    # Velocity = ROC(5), Acceleration = ROC(3) - ROC(3)[di-3]
    VELOCITY = ROC5.copy()
    ACCELERATION = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(3, ND):
            r3_now = ROC3[si, di]
            r3_prev = ROC3[si, di-3]
            if not np.isnan(r3_now) and not np.isnan(r3_prev):
                ACCELERATION[si, di] = r3_now - r3_prev
    print(f"  Acceleration done ({time.time()-t0:.1f}s)", flush=True)

    # --- D) Trend Purity ---
    # Fraction of positive return days in last 10 days
    PURITY = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            window = RET[si, di-10:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 5:
                PURITY[si, di] = np.sum(valid > 0) / len(valid)
    print(f"  Purity done ({time.time()-t0:.1f}s)", flush=True)

    # --- E) Asymmetric Momentum ---
    # avg_positive_return / |avg_negative_return| over 20 days
    ASYMMETRY = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            window = RET[si, di-20:di]
            valid = window[~np.isnan(window)]
            pos = valid[valid > 0]
            neg = valid[valid < 0]
            if len(pos) >= 3 and len(neg) >= 2:
                avg_up = np.mean(pos)
                avg_down = np.mean(np.abs(neg))
                if avg_down > 0:
                    ASYMMETRY[si, di] = avg_up / avg_down
    print(f"  Asymmetry done ({time.time()-t0:.1f}s)", flush=True)

    # --- F) Consecutive New 5-day Highs ---
    CONSEC_HIGHS = np.full((NS, ND), 0)
    for si in range(NS):
        c = C[si].astype(np.float64)
        streak = 0
        for di in range(5, ND):
            if np.isnan(c[di]):
                streak = 0
                continue
            window = c[di-5:di]
            valid_w = window[~np.isnan(window)]
            if len(valid_w) >= 3 and not np.isnan(c[di]):
                if c[di] > np.max(valid_w):
                    streak += 1
                else:
                    streak = 0
            else:
                streak = 0
            CONSEC_HIGHS[si, di] = streak
    print(f"  ConsecHighs done ({time.time()-t0:.1f}s)", flush=True)

    # --- G) Volume Momentum Concordance (cross-sectional) ---
    # Precompute cross-sectional percentile ranks for ROC5 and VOL_CHANGE
    VOL_CHANGE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            v_now = V[si, di]
            v_prev = V[si, di-5]
            if not np.isnan(v_now) and not np.isnan(v_prev) and v_prev > 0:
                VOL_CHANGE[si, di] = v_now / v_prev
    print(f"  VolChange done ({time.time()-t0:.1f}s)", flush=True)

    ROC5_RANK = cross_sectional_percentile_arr(ROC5)
    VOL_RANK = cross_sectional_percentile_arr(VOL_CHANGE)
    print(f"  Cross-sectional ranks done ({time.time()-t0:.1f}s)", flush=True)

    # --- H) Overnight-Intraday Decomposition ---
    OVERNIGHT_RET = np.full((NS, ND), np.nan)
    INTRADAY_RET = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            c_prev = C[si, di-1]
            o_now = O[si, di]
            c_now = C[si, di]
            if not np.isnan(c_prev) and c_prev > 0 and not np.isnan(o_now) and o_now > 0:
                OVERNIGHT_RET[si, di] = (o_now / c_prev - 1) * 100
            if not np.isnan(o_now) and o_now > 0 and not np.isnan(c_now) and c_now > 0:
                INTRADAY_RET[si, di] = (c_now / o_now - 1) * 100
    print(f"  Overnight/Intraday done ({time.time()-t0:.1f}s)", flush=True)

    # --- I) Entropy-Based Regime ---
    ENTROPY = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            window = RET[si, di-10:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 5:
                n_pos = np.sum(valid > 0)
                n_neg = np.sum(valid < 0)
                n_total = n_pos + n_neg
                if n_total > 0:
                    p_pos = n_pos / n_total
                    p_neg = n_neg / n_total
                    ent = 0.0
                    if p_pos > 0:
                        ent -= p_pos * np.log2(p_pos)
                    if p_neg > 0:
                        ent -= p_neg * np.log2(p_neg)
                    ENTROPY[si, di] = ent
    print(f"  Entropy done ({time.time()-t0:.1f}s)", flush=True)

    # --- J) Multi-Factor Composite v2 (PRECOMPUTE) ---
    # Score = 0.3*norm(ROC5) + 0.2*norm(Z20) + 0.15*norm(ADX) +
    #         0.15*norm(momentum_accel) + 0.1*norm(purity) + 0.1*norm(asymmetry)
    print(f"  Precomputing J composite...", flush=True)
    ROC5_PCT = cross_sectional_percentile_arr(ROC5)
    Z20_PCT = cross_sectional_percentile_arr(ZSCORE_20)
    ADX_PCT = cross_sectional_percentile_arr(ADX14)
    MAI_PCT = cross_sectional_percentile_arr(MAI)
    PUR_PCT = cross_sectional_percentile_arr(PURITY)
    ASY_PCT = cross_sectional_percentile_arr(ASYMMETRY)

    J_COMPOSITE = np.full((NS, ND), np.nan)
    weights = [0.3, 0.2, 0.15, 0.15, 0.1, 0.1]
    for si in range(NS):
        for di in range(ND):
            vals = [ROC5_PCT[si, di], Z20_PCT[si, di], ADX_PCT[si, di],
                    MAI_PCT[si, di], PUR_PCT[si, di], ASY_PCT[si, di]]
            if all(not np.isnan(v) for v in vals):
                J_COMPOSITE[si, di] = sum(w * v for w, v in zip(weights, vals))
    print(f"  J Composite done ({time.time()-t0:.1f}s)", flush=True)

    print(f"  All indicators computed ({time.time()-t0:.1f}s)", flush=True)

    # ================================================================
    # GENERIC BACKTEST ENGINE
    # ================================================================
    def backtest(cfg, start_di=MIN_TRAIN, end_di=None):
        if end_di is None:
            end_di = ND

        signal_type = cfg.get('signal_type', 'A')
        threshold = cfg.get('threshold', 1.5)
        hold = cfg.get('hold', 1)
        top_n = cfg.get('top_n', 1)
        label = cfg.get('label', '')

        cash = float(CASH0)
        positions = []
        trades = []
        daily_equity = []

        for di in range(start_di, end_di - 1):
            # Track daily equity
            port_val = cash
            for pos in positions:
                cp = C[pos['si'], di]
                if not np.isnan(cp) and cp > 0:
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    port_val += cp * mult * pos['lots'] - cp * mult * abs(pos['lots']) * COMM
            daily_equity.append(port_val)

            # Close positions whose hold period is up
            closed = []
            for pos in positions:
                days_held = di - pos['entry_di']
                if days_held >= pos['hold_days']:
                    exit_price = C[pos['si'], di]
                    if np.isnan(exit_price) or exit_price <= 0:
                        exit_price = pos['entry_price']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    mkt_val = exit_price * mult * abs(pos['lots'])
                    cash += mkt_val - mkt_val * COMM
                    pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
                    invested = pos['entry_price'] * mult * abs(pos['lots'])
                    pnl_pct = pnl / invested * 100 if invested > 0 else 0
                    trades.append({
                        'pnl': pnl, 'pnl_pct': pnl_pct,
                        'entry_di': pos['entry_di'], 'exit_di': di,
                        'sym': pos['sym'],
                    })
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            if len(positions) >= top_n:
                continue

            entry_di = di + 1
            if entry_di >= end_di:
                continue

            candidates = []

            for s in range(NS):
                ep = O[s, entry_di]
                if np.isnan(ep) or ep <= 0:
                    continue
                if any(p['si'] == s for p in positions):
                    continue

                score = 0.0
                valid_signal = False

                if signal_type == 'A':
                    # Momentum Acceleration: MAI > threshold AND ROC10 > 0
                    mai = MAI[s, di]
                    r10 = ROC10[s, di]
                    if not np.isnan(mai) and mai > threshold and not np.isnan(r10) and r10 > 0:
                        valid_signal = True
                        score = mai

                elif signal_type == 'B':
                    # Volatility-Adjusted Momentum: VMOM > threshold
                    vmom = VMOM[s, di]
                    if not np.isnan(vmom) and vmom > threshold:
                        valid_signal = True
                        score = vmom

                elif signal_type == 'C':
                    # Velocity > 1% AND Acceleration > 0
                    vel = VELOCITY[s, di]
                    acc = ACCELERATION[s, di]
                    if not np.isnan(vel) and vel > 1.0 and not np.isnan(acc) and acc > 0:
                        valid_signal = True
                        score = vel + acc

                elif signal_type == 'D':
                    # Trend Purity > threshold AND ROC5 > 1%
                    purity = PURITY[s, di]
                    r5 = ROC5[s, di]
                    if not np.isnan(purity) and purity > threshold and not np.isnan(r5) and r5 > 1.0:
                        valid_signal = True
                        score = purity * r5

                elif signal_type == 'E':
                    # Asymmetry > threshold AND ROC5 > 1%
                    asym = ASYMMETRY[s, di]
                    r5 = ROC5[s, di]
                    if not np.isnan(asym) and asym > threshold and not np.isnan(r5) and r5 > 1.0:
                        valid_signal = True
                        score = asym * r5

                elif signal_type == 'F':
                    # Consecutive 5-day Highs >= threshold AND ROC5 > 1%
                    consec = CONSEC_HIGHS[s, di]
                    r5 = ROC5[s, di]
                    if consec >= threshold and not np.isnan(r5) and r5 > 1.0:
                        valid_signal = True
                        score = consec * r5

                elif signal_type == 'G':
                    # Top threshold for BOTH ROC5 and VOL rank
                    roc_rank = ROC5_RANK[s, di]
                    vol_rank = VOL_RANK[s, di]
                    thresh_pct = (1.0 - threshold) * 100
                    if not np.isnan(roc_rank) and roc_rank >= thresh_pct:
                        if not np.isnan(vol_rank) and vol_rank >= thresh_pct:
                            valid_signal = True
                            score = (roc_rank + vol_rank) / 2

                elif signal_type == 'H':
                    # Overnight and Intraday both > threshold, ROC5 > 1%
                    ov = OVERNIGHT_RET[s, di]
                    ir = INTRADAY_RET[s, di]
                    r5 = ROC5[s, di]
                    if (not np.isnan(ov) and ov > threshold and
                        not np.isnan(ir) and ir > threshold and
                        not np.isnan(r5) and r5 > 1.0):
                        valid_signal = True
                        score = ov + ir + r5

                elif signal_type == 'I':
                    # Entropy < threshold AND ROC5 > 1% AND Z > 1.5
                    ent = ENTROPY[s, di]
                    r5 = ROC5[s, di]
                    z20 = ZSCORE_20[s, di]
                    if (not np.isnan(ent) and ent < threshold and
                        not np.isnan(r5) and r5 > 1.0 and
                        not np.isnan(z20) and z20 > 1.5):
                        valid_signal = True
                        score = r5 * z20

                elif signal_type == 'J':
                    # Precomputed composite > threshold percentile
                    comp = J_COMPOSITE[s, di]
                    if not np.isnan(comp) and comp > threshold:
                        valid_signal = True
                        score = comp

                elif signal_type == 'K':
                    # Champion + Trend Purity filter
                    r5 = ROC5[s, di]
                    r5_prev = ROC5[s, di-1]
                    z20 = ZSCORE_20[s, di]
                    purity = PURITY[s, di]
                    if (not np.isnan(r5) and r5 > 1.0 and
                        not np.isnan(z20) and z20 > 1.5 and
                        not np.isnan(r5_prev) and r5 > r5_prev and
                        not np.isnan(purity) and purity > threshold):
                        valid_signal = True
                        score = r5 * z20

                elif signal_type == 'L':
                    # Champion + Asymmetry filter
                    r5 = ROC5[s, di]
                    r5_prev = ROC5[s, di-1]
                    z20 = ZSCORE_20[s, di]
                    asym = ASYMMETRY[s, di]
                    if (not np.isnan(r5) and r5 > 1.0 and
                        not np.isnan(z20) and z20 > 1.5 and
                        not np.isnan(r5_prev) and r5 > r5_prev and
                        not np.isnan(asym) and asym > threshold):
                        valid_signal = True
                        score = r5 * z20 * asym

                if valid_signal:
                    candidates.append((score, s, ep))

            if not candidates:
                continue

            candidates.sort(key=lambda x: -x[0])
            n_slots = top_n - len(positions)
            cap_per_slot = cash / max(1, n_slots)

            for sc_val, s, price in candidates[:max(0, n_slots)]:
                sym = syms[s]
                mult = MULT.get(sym, DEF_MULT)
                contracts = max(1, int(cap_per_slot * 0.95 / (price * mult * (1 + COMM))))
                cost_in = price * mult * contracts * (1 + COMM)
                if cost_in > cash:
                    contracts = int(cash * 0.9 / (price * mult * (1 + COMM)))
                    cost_in = price * mult * contracts * (1 + COMM) if contracts > 0 else 0
                if contracts <= 0 or cost_in <= 0 or cost_in > cash:
                    continue
                cash -= cost_in
                positions.append({
                    'si': s, 'entry_price': price, 'entry_di': entry_di,
                    'lots': contracts, 'dir': 1, 'sym': sym,
                    'hold_days': hold,
                })

        # Close remaining positions
        for pos in positions:
            ae = end_di - 1
            exit_price = C[pos['si'], min(ae, ND-1)]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos['entry_price']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = exit_price * mult * abs(pos['lots'])
            cash += mkt_val - mkt_val * COMM
            pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
            invested = pos['entry_price'] * mult * abs(pos['lots'])
            pnl_pct = pnl / invested * 100 if invested > 0 else 0
            trades.append({
                'pnl': pnl, 'pnl_pct': pnl_pct,
                'entry_di': pos['entry_di'], 'exit_di': ae,
                'sym': pos['sym'],
            })

        n_days_test = end_di - start_di
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0

        if daily_equity:
            eq_arr = np.array(daily_equity)
            peak_arr = np.maximum.accumulate(eq_arr)
            dd_arr = (eq_arr - peak_arr) / peak_arr * 100
            mdd = float(np.min(dd_arr))
        else:
            mdd = 0.0

        return {
            'ann': ann, 'wr': wr, 'n': n_trades, 'avg_pnl': avg_pnl,
            'final_cash': cash, 'n_days': n_days_test, 'mdd': mdd,
            'label': label,
        }

    # ================================================================
    # HELPER: Walk-forward by year
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    def walk_forward(cfg):
        wf = {}
        for yr in wf_years:
            ts = te = None
            for di in range(ND):
                if dates[di].year == yr and ts is None:
                    ts = di
                if dates[di].year == yr + 1 and te is None:
                    te = di
            if ts is None:
                wf[yr] = None
                continue
            if te is None:
                te = ND
            r = backtest(cfg, start_di=ts, end_di=te)
            wf[yr] = r
        return wf

    def print_wf(label, wf):
        vals = {yr: wf[yr]['ann'] if wf[yr] else 0 for yr in wf_years}
        avg = np.mean(list(vals.values()))
        pos = sum(1 for v in vals.values() if v > 0)
        mdds = [wf[yr]['mdd'] for yr in wf_years if wf[yr]]
        avg_mdd = np.mean(mdds) if mdds else 0
        row = f"  {label:<60} | {avg:>+8.1f}% |"
        for yr in wf_years:
            v = vals[yr]
            row += f" {v:>+8.1f}% |"
        row += f" {pos}/6 | {avg_mdd:>6.1f}%"
        print(row, flush=True)
        return avg, pos

    # ================================================================
    # BUILD ALL CONFIGS
    # ================================================================
    print(f"\n[Config] Building test configurations A-L...", flush=True)

    all_configs = []

    # A) Momentum Acceleration Index: threshold sweep
    for t_val in [1.2, 1.5, 2.0]:
        all_configs.append({
            'label': f'A) MAI>{t_val} (ROC3/ROC10, ROC10>0)',
            'signal_type': 'A', 'threshold': t_val, 'hold': 1, 'top_n': 1,
        })

    # B) Volatility-Adjusted Momentum: threshold sweep
    for t_val in [1.5, 2.0, 2.5, 3.0]:
        all_configs.append({
            'label': f'B) VMOM>{t_val} (ROC5/ATR14%)',
            'signal_type': 'B', 'threshold': t_val, 'hold': 1, 'top_n': 1,
        })

    # C) Price Velocity + Acceleration
    all_configs.append({
        'label': 'C) Vel>1% AND Acc>0',
        'signal_type': 'C', 'threshold': 0, 'hold': 1, 'top_n': 1,
    })

    # D) Trend Purity: threshold sweep
    for t_val in [0.6, 0.7, 0.8, 0.9]:
        all_configs.append({
            'label': f'D) Purity>{t_val} AND ROC5>1%',
            'signal_type': 'D', 'threshold': t_val, 'hold': 1, 'top_n': 1,
        })

    # E) Asymmetric Momentum: threshold sweep
    for t_val in [1.5, 2.0, 2.5]:
        all_configs.append({
            'label': f'E) Asymmetry>{t_val} AND ROC5>1%',
            'signal_type': 'E', 'threshold': t_val, 'hold': 1, 'top_n': 1,
        })

    # F) Consecutive New 5-day Highs
    for t_val in [2, 3, 4]:
        all_configs.append({
            'label': f'F) Consec5dHighs>={t_val} AND ROC5>1%',
            'signal_type': 'F', 'threshold': t_val, 'hold': 1, 'top_n': 1,
        })

    # G) Volume Momentum Concordance
    for t_val in [0.2, 0.3]:
        all_configs.append({
            'label': f'G) Top {int(t_val*100)}% ROC5 AND Vol',
            'signal_type': 'G', 'threshold': t_val, 'hold': 1, 'top_n': 1,
        })

    # H) Overnight-Intraday Decomposition
    for t_val in [0.3, 0.5, 0.7]:
        all_configs.append({
            'label': f'H) OV>{t_val}% AND ID>{t_val}% AND ROC5>1%',
            'signal_type': 'H', 'threshold': t_val, 'hold': 1, 'top_n': 1,
        })

    # I) Entropy-Based Regime
    for t_val in [0.6, 0.7, 0.8, 0.9]:
        all_configs.append({
            'label': f'I) Entropy<{t_val} AND ROC5>1% AND Z>1.5',
            'signal_type': 'I', 'threshold': t_val, 'hold': 1, 'top_n': 1,
        })

    # J) Multi-Factor Composite v2
    for t_val in [60, 70, 75, 80, 85]:
        all_configs.append({
            'label': f'J) Composite>{t_val}th percentile',
            'signal_type': 'J', 'threshold': t_val, 'hold': 1, 'top_n': 1,
        })

    # K) Champion + Trend Purity Filter
    for t_val in [0.6, 0.7, 0.8]:
        all_configs.append({
            'label': f'K) Champion + Purity>{t_val}',
            'signal_type': 'K', 'threshold': t_val, 'hold': 1, 'top_n': 1,
        })

    # L) Champion + Asymmetry Filter
    for t_val in [1.0, 1.5, 2.0]:
        all_configs.append({
            'label': f'L) Champion + Asymmetry>{t_val}',
            'signal_type': 'L', 'threshold': t_val, 'hold': 1, 'top_n': 1,
        })

    print(f"  Total configs: {len(all_configs)}", flush=True)

    # ================================================================
    # SECTION 1: FULL-PERIOD BACKTEST ALL CONFIGS
    # ================================================================
    print(f"\n{'=' * 150}", flush=True)
    print("  SECTION 1: FULL-PERIOD BACKTEST (ALL NOVEL SIGNALS)", flush=True)
    print(f"{'=' * 150}", flush=True)
    print(f"  {'#':>3} | {'Config':<60} | {'Ann':>10} | {'WR':>6} | {'N':>5} | {'AvgPnL':>8} | {'MDD':>8} | {'Final':>12}", flush=True)
    print("-" * 150, flush=True)

    full_results = []
    for i, cfg in enumerate(all_configs):
        r = backtest(cfg)
        full_results.append(r)
        print(f"  {i+1:>3} | {cfg['label']:<60} | {r['ann']:>+9.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+7.3f}% | {r['mdd']:>+7.1f}% | {r['final_cash']:>11,.0f}", flush=True)

    # ================================================================
    # SECTION 2: WALK-FORWARD ALL CONFIGS
    # ================================================================
    print(f"\n{'=' * 180}", flush=True)
    print("  SECTION 2: WALK-FORWARD BY YEAR (2020-2025)", flush=True)
    print(f"{'=' * 180}", flush=True)
    header = f"  {'#':>3} | {'Config':<60} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>8} |"
    header += f" {'Pos':>4} | {'AvgMDD':>7}"
    print(header, flush=True)
    print("-" * 180, flush=True)

    wf_summary = []
    for i, (cfg, r) in enumerate(zip(all_configs, full_results)):
        wf = walk_forward(cfg)
        avg, pos = print_wf(f"{i+1}. {cfg['label']}", wf)
        wf_summary.append({'avg': avg, 'pos': pos, 'full_ann': r['ann'],
                           'label': cfg['label'], 'cfg': cfg})

    # ================================================================
    # SECTION 3: SUMMARY & ANALYSIS
    # ================================================================
    print(f"\n{'=' * 150}", flush=True)
    print("  SECTION 3: ANALYSIS & RANKING", flush=True)
    print(f"{'=' * 150}", flush=True)

    # Sort by full-period annual return
    ranked = sorted(zip(all_configs, full_results), key=lambda x: -x[1]['ann'])

    print(f"\n  TOP 10 by Annual Return:", flush=True)
    print(f"  {'#':>3} | {'Config':<60} | {'Ann':>10} | {'WR':>6} | {'N':>5} | {'MDD':>8}", flush=True)
    print("-" * 120, flush=True)
    for i, (cfg, r) in enumerate(ranked[:10]):
        print(f"  {i+1:>3} | {cfg['label']:<60} | {r['ann']:>+9.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['mdd']:>+7.1f}%", flush=True)

    # Sort by walk-forward average
    wf_ranked = sorted(wf_summary, key=lambda x: -x['avg'])
    print(f"\n  TOP 10 by Walk-Forward Average:", flush=True)
    print(f"  {'#':>3} | {'Config':<60} | {'WF Avg':>8} | {'WF+':>4} | {'Full Ann':>10} | {'Full MDD':>8}", flush=True)
    print("-" * 120, flush=True)
    for i, ws in enumerate(wf_ranked[:10]):
        r = full_results[all_configs.index(ws['cfg'])]
        print(f"  {i+1:>3} | {ws['label']:<60} | {ws['avg']:>+7.1f}% | {ws['pos']}/6 | {r['ann']:>+9.1f}% | {r['mdd']:>+7.1f}%", flush=True)

    # ================================================================
    # ANSWER THE 4 KEY QUESTIONS
    # ================================================================
    print(f"\n{'=' * 150}", flush=True)
    print("  KEY QUESTIONS & ANSWERS", flush=True)
    print(f"{'=' * 150}", flush=True)

    # Q1: Which novel signals produce positive alpha?
    print(f"\n  Q1: Which novel signals produce positive alpha?", flush=True)
    positive_alpha = [(cfg, r) for cfg, r in zip(all_configs, full_results) if r['ann'] > 0]
    print(f"      {len(positive_alpha)}/{len(all_configs)} configs have positive alpha", flush=True)
    if positive_alpha:
        by_type = {}
        for cfg, r in positive_alpha:
            sig_type = cfg['signal_type']
            if sig_type not in by_type:
                by_type[sig_type] = []
            by_type[sig_type].append((cfg, r))
        for sig_type in sorted(by_type.keys()):
            configs_type = by_type[sig_type]
            best_in_type = max(configs_type, key=lambda x: x[1]['ann'])
            print(f"      Signal {sig_type}: {len(configs_type)} positive configs, best={best_in_type[1]['ann']:>+.1f}% ({best_in_type[0]['label']})", flush=True)

    # Q2: Any novel signal beating +333.5%?
    print(f"\n  Q2: Any novel signal beating +333.5%?", flush=True)
    beat_champ = [(cfg, r) for cfg, r in ranked if r['ann'] > 333.5]
    if beat_champ:
        for cfg, r in beat_champ:
            print(f"      YES! {cfg['label']}: {r['ann']:>+.1f}% (WR={r['wr']:.1f}%, MDD={r['mdd']:>+.1f}%)", flush=True)
    else:
        best_novel = ranked[0]
        print(f"      NO. Best novel signal: {best_novel[1]['ann']:>+.1f}% ({best_novel[0]['label']})", flush=True)
        print(f"      Gap to champion: {333.5 - best_novel[1]['ann']:.1f}pp", flush=True)

    # Q3: Does adding novel filters (K, L) improve the champion?
    print(f"\n  Q3: Does adding novel filters (K, L) improve the champion?", flush=True)
    k_results = [(cfg, r) for cfg, r in zip(all_configs, full_results) if cfg['signal_type'] == 'K']
    print(f"      K) Champion + Trend Purity filter:", flush=True)
    for cfg, r in k_results:
        diff = r['ann'] - 333.5
        print(f"         {cfg['label']}: {r['ann']:>+.1f}% (diff: {diff:>+.1f}pp, WR={r['wr']:.1f}%, N={r['n']}, MDD={r['mdd']:>+.1f}%)", flush=True)
        ws = [w for w in wf_summary if w['cfg'] == cfg]
        if ws:
            print(f"           WF: avg={ws[0]['avg']:>+.1f}%, positive={ws[0]['pos']}/6", flush=True)

    l_results = [(cfg, r) for cfg, r in zip(all_configs, full_results) if cfg['signal_type'] == 'L']
    print(f"      L) Champion + Asymmetry filter:", flush=True)
    for cfg, r in l_results:
        diff = r['ann'] - 333.5
        print(f"         {cfg['label']}: {r['ann']:>+.1f}% (diff: {diff:>+.1f}pp, WR={r['wr']:.1f}%, N={r['n']}, MDD={r['mdd']:>+.1f}%)", flush=True)
        ws = [w for w in wf_summary if w['cfg'] == cfg]
        if ws:
            print(f"           WF: avg={ws[0]['avg']:>+.1f}%, positive={ws[0]['pos']}/6", flush=True)

    # Q4: Which novel factors have the most predictive power?
    print(f"\n  Q4: Which novel factors have the most predictive power?", flush=True)
    sig_types = {}
    for ws in wf_summary:
        st = ws['cfg']['signal_type']
        if st not in sig_types:
            sig_types[st] = []
        sig_types[st].append(ws)

    print(f"      {'Signal':<10} | {'Best Full Ann':>13} | {'Best WF Avg':>11} | {'Best WF+':>8} | {'# Positive':>10}", flush=True)
    print(f"      {'-'*70}", flush=True)
    for st in sorted(sig_types.keys()):
        entries = sig_types[st]
        best_full = max(entries, key=lambda x: x['full_ann'])
        best_wf = max(entries, key=lambda x: x['avg'])
        best_pos = max(entries, key=lambda x: x['pos'])
        n_positive = sum(1 for e in entries if e['full_ann'] > 0)
        print(f"      {st:<10} | {best_full['full_ann']:>+12.1f}% | {best_wf['avg']:>+10.1f}% | {best_pos['pos']:>6}/6 | {n_positive:>5}/{len(entries)}", flush=True)

    # Detailed walk-forward for the best overall novel signals (excluding K/L)
    print(f"\n  DETAILED WALK-FORWARD FOR TOP 5 NOVEL SIGNALS (excl. K/L):", flush=True)
    novel_ranked = [(cfg, r) for cfg, r in ranked if cfg['signal_type'] not in ('K', 'L')]
    for i, (cfg, r) in enumerate(novel_ranked[:5]):
        wf = walk_forward(cfg)
        print(f"\n  #{i+1}: {cfg['label']}", flush=True)
        print(f"       Full: {r['ann']:>+.1f}%, WR={r['wr']:.1f}%, N={r['n']}, MDD={r['mdd']:>+.1f}%", flush=True)
        print(f"       Walk-Forward:", flush=True)
        for yr in wf_years:
            if wf[yr]:
                print(f"         {yr}: {wf[yr]['ann']:>+8.1f}%  WR={wf[yr]['wr']:.1f}%  N={wf[yr]['n']}  MDD={wf[yr]['mdd']:>+6.1f}%", flush=True)
            else:
                print(f"         {yr}: N/A", flush=True)

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed: {elapsed:.1f}s", flush=True)
    print("=" * 150, flush=True)


if __name__ == '__main__':
    main()
