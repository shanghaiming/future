"""
Alpha Futures V103 -- Maximum Concentration: High-Conviction Single-Trade Compounding
=====================================================================================
Current best next-open: +49.6% (50-day breakout).

V103 IDEA: Put ALL capital into 1 trade when MULTIPLE high-conviction signals agree.
Maximize compounding by being extremely selective -- only trade when stars align.

SIGNALS (all computed at close of day di, entry at O[si, di+1]):

A) TRIPLE_CONFIRM:  50-day breakout + high vol day + bullish close + uptrend
B) QUAD_CONFIRM:    Triple + volume surge + OI increasing
C) BREAKOUT_STRENGTH_RANKED: 50-day highs ranked by breakout strength %, top 1
D) TREND_ACCELERATION: 5d return > 2x 20d return + 20-day high
E) NEW_HIGH_WITH_PULLBACK_ENTRY: recent 50d high + pullback 2-3% + recovery
F) BREAKOUT_WITH_TRAILING_STOP: 50-day breakout + ATR trailing stop

Long-only. COMM=0.0003. Walk-forward: top 15 configs across 2020-2025.
"""
import sys, os, time, warnings
import numpy as np
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


def main():
    print("=" * 170)
    print("Alpha Futures V103 -- Maximum Concentration: High-Conviction Single-Trade Compounding")
    print("=" * 170)
    print("\n  Only trade when MULTIPLE signals agree. ALL capital into 1 position.")
    print("  ALL signals computed at close di, entry at O[si, di+1] (NEXT DAY OPEN)")

    # -- Load data -------------------------------------------------
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ================================================================
    # PRECOMPUTE INDICATORS
    # ================================================================
    print("\n[Indicators] Computing...", flush=True)
    t0 = time.time()

    # ---- ATR_20 ----
    ATR20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            trs = []
            for k in range(di - 20, di + 1):
                h = H[si, k]
                l = L[si, k]
                cp = C[si, k - 1] if k > 0 else np.nan
                if np.isnan(h) or np.isnan(l):
                    continue
                tr = h - l
                if not np.isnan(cp) and cp > 0:
                    tr = max(tr, abs(h - cp), abs(l - cp))
                trs.append(tr)
            if len(trs) >= 15:
                ATR20[si, di] = np.mean(trs)
    print(f"  ATR_20 computed ({time.time()-t0:.1f}s)")

    # ---- 50-day high (lookback window, not including today) ----
    high50 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(50, ND):
            h50 = np.nanmax(C[si, di - 50:di])
            if not np.isnan(h50):
                high50[si, di] = h50
    print(f"  50-day high computed ({time.time()-t0:.1f}s)")

    # ---- 20-day high ----
    high20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            h20 = np.nanmax(C[si, di - 20:di])
            if not np.isnan(h20):
                high20[si, di] = h20
    print(f"  20-day high computed ({time.time()-t0:.1f}s)")

    # ---- 20-day return ----
    ret20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            c_cur = C[si, di]
            c_prev = C[si, di - 20]
            if not np.isnan(c_cur) and not np.isnan(c_prev) and c_prev > 0:
                ret20[si, di] = (c_cur - c_prev) / c_prev
    print(f"  20-day return computed ({time.time()-t0:.1f}s)")

    # ---- 5-day return ----
    ret5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            c_cur = C[si, di]
            c_prev = C[si, di - 5]
            if not np.isnan(c_cur) and not np.isnan(c_prev) and c_prev > 0:
                ret5[si, di] = (c_cur - c_prev) / c_prev
    print(f"  5-day return computed ({time.time()-t0:.1f}s)")

    # ---- 20-day volume MA ----
    vol_ma20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            vw = V[si, di - 20:di]
            valid = vw[~np.isnan(vw)]
            if len(valid) >= 10:
                vol_ma20[si, di] = np.mean(valid)
    print(f"  20-day volume MA computed ({time.time()-t0:.1f}s)")

    # ---- Day range (H - L) ----
    day_range = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            h = H[si, di]
            l = L[si, di]
            if not np.isnan(h) and not np.isnan(l):
                day_range[si, di] = h - l
    print(f"  Day range computed ({time.time()-t0:.1f}s)")

    # ---- 50-day high within last 5 days (for signal E) ----
    recent_high50 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(55, ND):
            # max of C[di-5:di+1] across the 50-day high lookback
            for k in range(di, max(di - 5, 50) - 1, -1):
                if k >= 50:
                    h50_val = high50[si, k]
                    c_val = C[si, k]
                    if not np.isnan(h50_val) and not np.isnan(c_val) and c_val > h50_val:
                        recent_high50[si, di] = c_val
                        break
    print(f"  Recent 50-day highs computed ({time.time()-t0:.1f}s)")

    print(f"\n  All indicators computed ({time.time()-t_start:.1f}s total)")

    # ================================================================
    # SIGNAL GENERATION (at close of day di)
    # ================================================================
    print("\n[Signals] Computing all signals...", flush=True)

    # Signal A: TRIPLE_CONFIRM
    sig_triple = np.zeros((NS, ND), dtype=bool)
    sig_triple_score = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(50, ND):
            c = C[si, di]
            o = O[si, di]
            h50_val = high50[si, di]
            atr = ATR20[si, di]
            r20 = ret20[si, di]
            rng = day_range[si, di]
            if np.isnan(c) or np.isnan(o) or np.isnan(h50_val) or np.isnan(atr) or np.isnan(r20) or np.isnan(rng):
                continue
            # Condition 1: 50-day high breakout
            if c <= h50_val:
                continue
            # Condition 2: high vol day (range > 2 * ATR_20)
            if atr <= 0 or rng <= 2 * atr:
                continue
            # Condition 3: bullish close
            if c <= o:
                continue
            # Condition 4: 20-day return > 0
            if r20 <= 0:
                continue
            sig_triple[si, di] = True
            sig_triple_score[si, di] = (c - h50_val) / h50_val * 100
    print(f"  A) TRIPLE_CONFIRM: {np.sum(sig_triple)} signals")

    # Signal B: QUAD_CONFIRM (Triple + volume + OI)
    sig_quad = np.zeros((NS, ND), dtype=bool)
    sig_quad_score = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(50, ND):
            if not sig_triple[si, di]:
                continue
            v_cur = V[si, di]
            v_ma = vol_ma20[si, di]
            oi_cur = OI[si, di]
            oi_5ago = OI[si, di - 5] if di >= 5 else np.nan
            if np.isnan(v_cur) or np.isnan(v_ma) or v_ma <= 0:
                continue
            if v_cur <= 1.5 * v_ma:
                continue
            if np.isnan(oi_cur) or np.isnan(oi_5ago):
                continue
            if oi_cur <= oi_5ago:
                continue
            sig_quad[si, di] = True
            sig_quad_score[si, di] = sig_triple_score[si, di] if not np.isnan(sig_triple_score[si, di]) else 0
    print(f"  B) QUAD_CONFIRM: {np.sum(sig_quad)} signals")

    # Signal C: BREAKOUT_STRENGTH_RANKED
    sig_brk_str = np.zeros((NS, ND), dtype=bool)
    sig_brk_str_score = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(50, ND):
            c = C[si, di]
            h50_val = high50[si, di]
            if np.isnan(c) or np.isnan(h50_val) or h50_val <= 0:
                continue
            if c > h50_val:
                strength = (c - h50_val) / h50_val * 100
                if strength > 1.0:  # breaking out by at least 1%
                    sig_brk_str[si, di] = True
                    sig_brk_str_score[si, di] = strength
    print(f"  C) BREAKOUT_STRENGTH_RANKED: {np.sum(sig_brk_str)} signals")

    # Signal D: TREND_ACCELERATION
    sig_accel = np.zeros((NS, ND), dtype=bool)
    sig_accel_score = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            r5 = ret5[si, di]
            r20 = ret20[si, di]
            h20_val = high20[si, di]
            c = C[si, di]
            if np.isnan(r5) or np.isnan(r20) or np.isnan(h20_val) or np.isnan(c):
                continue
            # Both positive
            if r5 <= 0 or r20 <= 0:
                continue
            # 5d return > 2x 20d return (trend accelerating)
            if r5 <= 2.0 * r20:
                continue
            # Making 20-day high
            if c <= h20_val:
                continue
            sig_accel[si, di] = True
            sig_accel_score[si, di] = r5 / r20  # acceleration ratio
    print(f"  D) TREND_ACCELERATION: {np.sum(sig_accel)} signals")

    # Signal E: NEW_HIGH_WITH_PULLBACK_ENTRY
    sig_pullback = np.zeros((NS, ND), dtype=bool)
    sig_pullback_score = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(55, ND):
            c = C[si, di]
            c_prev = C[si, di - 1]
            if np.isnan(c) or np.isnan(c_prev):
                continue
            # Must be recovering (today's close > yesterday's close)
            if c <= c_prev:
                continue
            # Look back 5 days for a 50-day high
            found_high = False
            recent_peak = np.nan
            for k in range(di - 5, di):
                if k < 50:
                    continue
                h50_val = high50[si, k]
                c_k = C[si, k]
                if not np.isnan(h50_val) and not np.isnan(c_k) and c_k > h50_val:
                    found_high = True
                    if np.isnan(recent_peak) or c_k > recent_peak:
                        recent_peak = c_k
            if not found_high or np.isnan(recent_peak):
                continue
            # Pulled back 2-3% from that high
            pct_drop = (recent_peak - c) / recent_peak * 100
            if pct_drop < 1.5 or pct_drop > 4.0:
                continue
            sig_pullback[si, di] = True
            sig_pullback_score[si, di] = pct_drop
    print(f"  E) PULLBACK_ENTRY: {np.sum(sig_pullback)} signals")

    # Signal F: BREAKOUT_WITH_TRAILING_STOP (signal is 50-day breakout, exit is trailing)
    sig_trail = np.zeros((NS, ND), dtype=bool)
    sig_trail_score = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(50, ND):
            c = C[si, di]
            h50_val = high50[si, di]
            if np.isnan(c) or np.isnan(h50_val) or h50_val <= 0:
                continue
            if c > h50_val:
                sig_trail[si, di] = True
                sig_trail_score[si, di] = (c - h50_val) / h50_val * 100
    print(f"  F) BREAKOUT_TRAILING_STOP: {np.sum(sig_trail)} signals")

    print(f"\n  All signals computed ({time.time()-t_start:.1f}s total)")

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(config, wf_test_year=None):
        """
        Config:
            signal: 'triple' | 'quad' | 'brk_strength' | 'accel' | 'pullback' | 'trail'
            hold_days: int (ignored for 'trail' signal)
            top_n: int (1 or 3, max concurrent positions)
            comm: float
            use_oi: bool (whether OI is required -- for quad signal)
        """
        sig_type = config['signal']
        hold_days = config['hold_days']
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
        positions = []  # list of position dicts
        trades = []

        for di in range(start_di, end_di - 1):
            # Reset cash at test window start (WF mode)
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                positions = []

            # -- Close positions -----------------------------------------
            closed = []
            for pos in positions:
                if sig_type == 'trail':
                    # Trailing stop exit
                    c_now = C[pos['si'], di]
                    if not np.isnan(c_now):
                        # Update highest close since entry
                        if c_now > pos.get('max_c', 0):
                            pos['max_c'] = c_now
                        # Check trailing stop
                        atr_val = ATR20[pos['si'], di] if not np.isnan(ATR20[pos['si'], di]) else 0
                        if atr_val > 0:
                            trail = pos['max_c'] - 2 * atr_val
                            if trail > pos.get('trail_stop', 0):
                                pos['trail_stop'] = trail
                            if c_now < pos['trail_stop']:
                                # Exit via trailing stop
                                exit_price = c_now
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
                                    'sym': pos.get('sym', ''),
                                    'days_held': di - pos['entry_di'],
                                })
                                closed.append(pos)
                                continue

                    # Max hold 30 days
                    days_held = di - pos['entry_di']
                    if days_held >= 30:
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
                            'sym': pos.get('sym', ''),
                            'days_held': days_held,
                        })
                        if pos not in closed:
                            closed.append(pos)
                else:
                    # Fixed hold exit
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
                            'sym': pos.get('sym', ''),
                            'days_held': days_held,
                        })
                        closed.append(pos)

            for pos in closed:
                positions.remove(pos)

            # If we still have positions, don't open new ones (sequential for top_n=1)
            if len(positions) >= top_n:
                continue

            # -- Generate signals at day di --------------------------------
            entry_di = di + 1
            if entry_di >= end_di:
                continue

            candidates = []

            if sig_type == 'triple':
                for si in range(NS):
                    if not sig_triple[si, di]:
                        continue
                    if any(p['si'] == si for p in positions):
                        continue
                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    sc = sig_triple_score[si, di] if not np.isnan(sig_triple_score[si, di]) else 0
                    candidates.append((sc, {
                        'si': si, 'sym': syms[si], 'entry_price': ep,
                    }))

            elif sig_type == 'quad':
                for si in range(NS):
                    if not sig_quad[si, di]:
                        continue
                    if any(p['si'] == si for p in positions):
                        continue
                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    sc = sig_quad_score[si, di] if not np.isnan(sig_quad_score[si, di]) else 0
                    candidates.append((sc, {
                        'si': si, 'sym': syms[si], 'entry_price': ep,
                    }))

            elif sig_type == 'brk_strength':
                for si in range(NS):
                    if not sig_brk_str[si, di]:
                        continue
                    if any(p['si'] == si for p in positions):
                        continue
                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    sc = sig_brk_str_score[si, di] if not np.isnan(sig_brk_str_score[si, di]) else 0
                    candidates.append((sc, {
                        'si': si, 'sym': syms[si], 'entry_price': ep,
                    }))

            elif sig_type == 'accel':
                for si in range(NS):
                    if not sig_accel[si, di]:
                        continue
                    if any(p['si'] == si for p in positions):
                        continue
                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    sc = sig_accel_score[si, di] if not np.isnan(sig_accel_score[si, di]) else 0
                    candidates.append((sc, {
                        'si': si, 'sym': syms[si], 'entry_price': ep,
                    }))

            elif sig_type == 'pullback':
                for si in range(NS):
                    if not sig_pullback[si, di]:
                        continue
                    if any(p['si'] == si for p in positions):
                        continue
                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    sc = sig_pullback_score[si, di] if not np.isnan(sig_pullback_score[si, di]) else 0
                    candidates.append((sc, {
                        'si': si, 'sym': syms[si], 'entry_price': ep,
                    }))

            elif sig_type == 'trail':
                for si in range(NS):
                    if not sig_trail[si, di]:
                        continue
                    if any(p['si'] == si for p in positions):
                        continue
                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    sc = sig_trail_score[si, di] if not np.isnan(sig_trail_score[si, di]) else 0
                    candidates.append((sc, {
                        'si': si, 'sym': syms[si], 'entry_price': ep,
                    }))

            if not candidates:
                continue

            # Sort by score descending
            candidates.sort(key=lambda x: -x[0])

            # Open positions -- concentrated allocation
            n_slots = top_n - len(positions)
            for sc, info in candidates[:max(0, n_slots)]:
                si = info['si']
                sym = info['sym']
                price = info['entry_price']
                mult = MULT.get(sym, DEF_MULT)
                notional = price * mult
                # Allocate all available cash to this single position
                lots = int(cash / (notional * (1 + comm)))
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

                pos_dict = {
                    'si': si, 'entry_price': price, 'entry_di': entry_di,
                    'lots': lots, 'dir': 1, 'sym': sym,
                    'hold_days': hold_days,
                }
                if sig_type == 'trail':
                    atr_val = ATR20[si, di] if not np.isnan(ATR20[si, di]) else 0
                    pos_dict['trail_stop'] = price - 2 * atr_val if atr_val > 0 else price * 0.95
                    pos_dict['max_c'] = price
                    pos_dict['hold_days'] = 30

                positions.append(pos_dict)

        # Close remaining positions at end
        for pos in positions:
            ae = end_di - 1 if end_di < ND else ND - 1
            exit_price = C[pos['si'], ae]
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
                'exit_di': ae,
                'year': dates[ae].year if ae < ND else dates[-1].year,
                'sym': pos.get('sym', ''),
                'days_held': ae - pos['entry_di'],
            })

        # Results
        wf_mode = wf_test_year is not None
        n_days_test = (test_end_di - test_start_di) if wf_mode else (end_di - start_di)
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0
        avg_hold = np.mean([t['days_held'] for t in trades]) if trades else 0

        # Max drawdown from equity curve
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

        # Trade frequency
        freq_per_yr = n_trades / (n_days_test / 252) if n_days_test > 0 else 0

        return {
            'ann': ann, 'wr': wr, 'n': n_trades, 'avg_pnl': avg_pnl,
            'final_cash': cash, 'n_days': n_days_test, 'mdd': mdd,
            'avg_hold': avg_hold, 'freq': freq_per_yr,
        }

    # ================================================================
    # BUILD CONFIGURATIONS
    # ================================================================
    print("\n[Sweep] Building configurations...", flush=True)
    configs = []
    cid = 0

    # --- A: TRIPLE_CONFIRM: hold 5/10/20, top_n 1/3 ---
    for hd in [5, 10, 20]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'triple',
                'hold_days': hd, 'top_n': tn, 'comm': COMM,
                'label': f"Triple_H{hd}_TN{tn}",
            })

    # --- B: QUAD_CONFIRM: hold 5/10/20, top_n 1/3 ---
    for hd in [5, 10, 20]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'quad',
                'hold_days': hd, 'top_n': tn, 'comm': COMM,
                'label': f"Quad_H{hd}_TN{tn}",
            })

    # --- C: BREAKOUT_STRENGTH_RANKED: hold 5/10, top_n 1/3 ---
    for hd in [5, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'brk_strength',
                'hold_days': hd, 'top_n': tn, 'comm': COMM,
                'label': f"BrkStr_H{hd}_TN{tn}",
            })

    # --- D: TREND_ACCELERATION: hold 5/10, top_n 1/3 ---
    for hd in [5, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'accel',
                'hold_days': hd, 'top_n': tn, 'comm': COMM,
                'label': f"Accel_H{hd}_TN{tn}",
            })

    # --- E: PULLBACK_ENTRY: hold 5/10, top_n 1/3 ---
    for hd in [5, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'pullback',
                'hold_days': hd, 'top_n': tn, 'comm': COMM,
                'label': f"Pullback_H{hd}_TN{tn}",
            })

    # --- F: BREAKOUT_WITH_TRAILING_STOP: top_n 1/3 ---
    for tn in [1, 3]:
        cid += 1
        configs.append({
            'id': cid, 'signal': 'trail',
            'hold_days': 30, 'top_n': tn, 'comm': COMM,
            'label': f"Trail_TN{tn}",
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
        if (i + 1) % 10 == 0 or i == len(configs) - 1:
            print(f"  ... {i+1}/{len(configs)} done ({time.time()-t_start:.0f}s)", flush=True)

    results.sort(key=lambda x: -x['ann'])

    # ================================================================
    # FULL-PERIOD RESULTS
    # ================================================================
    print(f"\n{'=' * 170}")
    print("  FULL-PERIOD RESULTS (All configs) -- ALL NEXT-OPEN EXECUTION, CONCENTRATED")
    print(f"{'=' * 170}")
    print(f"  {'#':>3} | {'Label':<25} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'AvgHold':>7} | {'Freq/Yr':>7} | {'Final':>14}")
    print("-" * 150)
    for i, r in enumerate(results):
        print(f"  {i+1:>3} | {r['label']:<25} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}% | {r['avg_hold']:>6.1f}d | {r['freq']:>6.1f}/yr | {r['final_cash']:>13,.0f}")

    # ================================================================
    # BEST PER SIGNAL TYPE
    # ================================================================
    sig_order = ['triple', 'quad', 'brk_strength', 'accel', 'pullback', 'trail']
    sig_names = {
        'triple':       'A) TRIPLE_CONFIRM (brk+vol+bull+up)',
        'quad':         'B) QUAD_CONFIRM (+OI+Vol)',
        'brk_strength': 'C) BREAKOUT_STRENGTH_RANKED',
        'accel':        'D) TREND_ACCELERATION',
        'pullback':     'E) PULLBACK_ENTRY',
        'trail':        'F) BREAKOUT_TRAILING_STOP',
    }

    print(f"\n{'=' * 170}")
    print("  BEST PER SIGNAL TYPE (Full Period) -- CONCENTRATED ALLOCATION")
    print(f"{'=' * 170}")
    print(f"  {'Signal':<42} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'AvgHold':>7} | {'Freq/Yr':>7} | Best Config")
    print("-" * 170)

    best_per_sig = {}
    for r in results:
        key = r['config']['signal']
        if key not in best_per_sig:
            best_per_sig[key] = r

    for sig in sig_order:
        if sig in best_per_sig:
            b = best_per_sig[sig]
            print(f"  {sig_names.get(sig, sig):<42} | {b['ann']:>+8.1f}% | {b['wr']:>5.1f}% | {b['n']:>5} | {b['avg_pnl']:>+6.3f}% | {b['mdd']:>6.1f}% | {b['avg_hold']:>6.1f}d | {b['freq']:>6.1f}/yr | {b['label']}")

    # ================================================================
    # SIGNAL TYPE SUMMARY
    # ================================================================
    print(f"\n{'=' * 170}")
    print("  SIGNAL TYPE SUMMARY (Average of all configs per type)")
    print(f"{'=' * 170}")
    print(f"  {'Signal':<42} | {'Avg Ann':>9} | {'Avg WR':>7} | {'Avg N':>7} | {'Avg PnL':>8} | {'Avg MDD':>8} | {'#Positive':>9}")
    print("-" * 150)

    for sig in sig_order:
        sub = [r for r in results if r['config']['signal'] == sig]
        if not sub:
            continue
        avg_ann = np.mean([r['ann'] for r in sub])
        avg_wr = np.mean([r['wr'] for r in sub])
        avg_n = np.mean([r['n'] for r in sub])
        avg_pnl = np.mean([r['avg_pnl'] for r in sub])
        avg_mdd = np.mean([r['mdd'] for r in sub])
        n_pos = sum(1 for r in sub if r['ann'] > 0)
        print(f"  {sig_names.get(sig, sig):<42} | {avg_ann:>+8.1f}% | {avg_wr:>6.1f}% | {avg_n:>7.0f} | {avg_pnl:>+7.3f}% | {avg_mdd:>7.1f}% | {n_pos:>5}/{len(sub)}")

    # ================================================================
    # WALK-FORWARD (Top 15 configs)
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Collect top 15 overall + best per signal type
    wf_configs = list(results[:15])
    for sig in sig_order:
        if sig in best_per_sig:
            r = best_per_sig[sig]
            if r['config'] not in [w['config'] for w in wf_configs]:
                wf_configs.append(r)

    print(f"\n{'=' * 200}")
    print(f"  WALK-FORWARD ({len(wf_configs)} configs)")
    print(f"{'=' * 200}")

    header = f"  {'#':>3} | {'Config':<25} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>7} |"
    header += f" {'Pos':>4} | {'MDD':>7} | {'WR':>6}"
    print(header)
    print("-" * 200)

    wf_rows = []
    for i, r in enumerate(wf_configs):
        cfg = r['config']
        wf_row = {'label': cfg['label'], 'signal': cfg['signal'],
                  'entry': 'next_open', 'windows': {}, 'mdd': {}, 'wr': {}}
        for yr in wf_years:
            wr = run_backtest(cfg, wf_test_year=yr)
            if wr:
                wf_row['windows'][yr] = wr['ann']
                wf_row['mdd'][yr] = wr['mdd']
                wf_row['wr'][yr] = wr['wr']
        wf_rows.append(wf_row)

        vals = [wf_row['windows'].get(yr, 0) for yr in wf_years]
        avg = np.mean(vals) if vals else 0
        pos = sum(1 for v in vals if v > 0)
        avg_mdd = np.mean(list(wf_row['mdd'].values())) if wf_row['mdd'] else 0
        avg_wr = np.mean(list(wf_row['wr'].values())) if wf_row['wr'] else 0

        row_str = f"  {i+1:>3} | {wf_row['label']:<25} | {avg:>+7.1f}% |"
        for v in vals:
            row_str += f" {v:>+7.1f}% |"
        row_str += f" {pos}/6 | {avg_mdd:>6.1f}% | {avg_wr:>5.1f}%"
        print(row_str)

    # ================================================================
    # WF COMPARISON PER SIGNAL
    # ================================================================
    print(f"\n{'=' * 170}")
    print("  WALK-FORWARD COMPARISON (Best per signal type)")
    print(f"{'=' * 170}")
    header2 = f"  {'Signal':<42} | {'WF Avg':>8} |"
    for yr in wf_years:
        header2 += f" {yr:>7} |"
    header2 += f" {'Pos':>4} | Avg MDD | Avg WR"
    print(header2)
    print("-" * 170)

    for sig in sig_order:
        wf_match = [w for w in wf_rows if w['signal'] == sig]
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            avg = np.mean(vals) if vals else 0
            pos = sum(1 for v in vals if v > 0)
            avg_mdd = np.mean(list(wf['mdd'].values())) if wf['mdd'] else 0
            avg_wr = np.mean(list(wf['wr'].values())) if wf['wr'] else 0
            row_str = f"  {sig_names.get(sig, sig):<42} | {avg:>+7.1f}% |"
            for v in vals:
                row_str += f" {v:>+7.1f}% |"
            row_str += f" {pos}/6 | {avg_mdd:>6.1f}% | {avg_wr:>5.1f}%"
            print(row_str)

    # ================================================================
    # CONCENTRATION ANALYSIS: top_n=1 vs top_n=3
    # ================================================================
    print(f"\n{'=' * 170}")
    print("  CONCENTRATION ANALYSIS: top_n=1 (ALL IN) vs top_n=3 (diversified)")
    print(f"{'=' * 170}")

    for sig in sig_order:
        sub1 = [r for r in results if r['config']['signal'] == sig and r['config']['top_n'] == 1]
        sub3 = [r for r in results if r['config']['signal'] == sig and r['config']['top_n'] == 3]
        best1 = sub1[0] if sub1 else None
        best3 = sub3[0] if sub3 else None
        if best1 and best3:
            print(f"  {sig_names.get(sig, sig):<42}")
            print(f"    top_n=1: Ann={best1['ann']:>+8.1f}%  WR={best1['wr']:>5.1f}%  N={best1['n']:>4}  MDD={best1['mdd']:>6.1f}%  Freq={best1['freq']:>5.1f}/yr")
            print(f"    top_n=3: Ann={best3['ann']:>+8.1f}%  WR={best3['wr']:>5.1f}%  N={best3['n']:>4}  MDD={best3['mdd']:>6.1f}%  Freq={best3['freq']:>5.1f}/yr")
            print()

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print(f"\n{'=' * 170}")
    print("  FINAL VERDICT: MAXIMUM CONCENTRATION -- HIGH-CONVICTION TRADING")
    print(f"{'=' * 170}")
    print()
    print("  KEY QUESTION: Can concentrated high-conviction trading push returns to 100%+")
    print("  with practical (next-open) execution?")
    print()

    for sig in sig_order:
        sub = [r for r in results if r['config']['signal'] == sig]
        if not sub:
            continue
        best = sub[0]
        n_pos = sum(1 for r in sub if r['ann'] > 0)

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
        genuine = "GENUINE ALPHA" if wf_pos >= 4 and best['ann'] > 0 else ("MARGINAL" if wf_pos >= 3 and best['ann'] > 0 else "NO ALPHA")
        concentrated = "CONCENTRATION WORKS" if best['ann'] > 50 else ("MODERATE" if best['ann'] > 20 else "INSUFFICIENT")

        print(f"  {sig_names.get(sig, sig)}")
        print(f"    Best annual: {best['ann']:>+8.1f}%  |  {n_pos}/{len(sub)} positive configs")
        print(f"    Walk-forward: {wf_pos}/6 positive  |  WF avg: {wf_avg:>+8.1f}%")
        print(f"    Trade freq: {best['freq']:>5.1f}/yr  |  Avg hold: {best['avg_hold']:>5.1f}d  |  Avg PnL: {best['avg_pnl']:>+6.3f}%")
        print(f"    VERDICT: {verdict}  -->  {genuine}  -->  {concentrated}")
        print()

    # Absolute best
    if results:
        champ = results[0]
        print(f"  {'='*60}")
        print(f"  CHAMPION: {champ['label']}")
        print(f"    Annual: {champ['ann']:>+8.1f}%  |  WR: {champ['wr']:>5.1f}%  |  N: {champ['n']:>4}  |  MDD: {champ['mdd']:>6.1f}%")
        print(f"    Avg PnL/trade: {champ['avg_pnl']:>+6.3f}%  |  Avg Hold: {champ['avg_hold']:>5.1f}d  |  Freq: {champ['freq']:>5.1f}/yr")
        # WF stats for champion
        champ_wf = [w for w in wf_rows if w['label'] == champ['label']]
        if champ_wf:
            cw = champ_wf[0]
            vals = [cw['windows'].get(yr, 0) for yr in wf_years]
            print(f"    WF: {['{:>+7.1f}%'.format(v) for v in vals]}  |  {sum(1 for v in vals if v > 0)}/6 positive")
        print(f"  {'='*60}")

    print(f"\n  Total runtime: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
