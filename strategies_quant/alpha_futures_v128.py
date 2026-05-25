"""
Alpha Futures V128 — AL BROOKS PRICE ACTION QUANTIFIED
======================================================
Quantify key concepts from Brooks' 3 books:

A) Strong Trend Bar Continuation: Large body + trend direction = continuation
B) Two-legged Pullback: In uptrend, buy after 2 pullback legs complete
C) Final Flag Breakout: After consolidation in trend, breakout = continuation
D) Buying Climax Exhaustion: Extreme bars → fade (SHORT signal)
E) Channel Walker: Price bouncing in broad channel, buy at lower boundary

ALL signals use NEXT-OPEN execution: signal at close di, entry at O[si, di+1].
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


def main():
    print("=" * 120)
    print("  Alpha Futures V128 — AL BROOKS PRICE ACTION QUANTIFIED")
    print("=" * 120)
    print(f"\n  Quantifying Brooks concepts: Trend Bar, Pullback Legs, Final Flag, Climax, Channel")
    print(f"  ALL signals at close di, entry at O[si, di+1] (NEXT DAY OPEN)")

    # -- Load data -------------------------------------------------
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")
    print(f"  MIN_TRAIN={MIN_TRAIN}, CASH0={CASH0:,}")

    # ================================================================
    # PRECOMPUTE INDICATORS
    # ================================================================
    print("\n[Precompute] All indicators...", flush=True)
    t0 = time.time()

    # Daily returns in percent
    RET = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100

    # ROC(5), ROC(10), ROC(20)
    ROC5 = np.full((NS, ND), np.nan)
    ROC10 = np.full((NS, ND), np.nan)
    ROC20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        ROC5[si] = talib.ROC(c, timeperiod=5)
        ROC10[si] = talib.ROC(c, timeperiod=10)
        ROC20[si] = talib.ROC(c, timeperiod=20)

    # ATR(14)
    ATR14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        h = H[si].astype(np.float64)
        l = L[si].astype(np.float64)
        c = C[si].astype(np.float64)
        ATR14[si] = talib.ATR(h, l, c, timeperiod=14)

    # ADX(14)
    ADX14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        h = H[si].astype(np.float64)
        l = L[si].astype(np.float64)
        c = C[si].astype(np.float64)
        ADX14[si] = talib.ADX(h, l, c, timeperiod=14)

    # RSI(14)
    RSI14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        RSI14[si] = talib.RSI(c, timeperiod=14)

    # Bollinger Bands (20, 2)
    BB_UP = np.full((NS, ND), np.nan)
    BB_MID = np.full((NS, ND), np.nan)
    BB_LOW = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        bb_up, bb_mid, bb_low = talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2)
        BB_UP[si] = bb_up
        BB_MID[si] = bb_mid
        BB_LOW[si] = bb_low

    # Body ratio: |C-O| / (H-L) — measures trend bar strength
    BODY_RATIO = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            h = H[si, di]; l = L[si, di]; c = C[si, di]; o = O[si, di]
            if any(np.isnan(x) for x in [h, l, c, o]) or h == l:
                continue
            BODY_RATIO[si, di] = abs(c - o) / (h - l)

    # Direction: 1=bull bar (C>O), -1=bear bar (C<O), 0=doji
    BAR_DIR = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        for di in range(ND):
            c = C[si, di]; o = O[si, di]
            if np.isnan(c) or np.isnan(o):
                continue
            if c > o: BAR_DIR[si, di] = 1
            elif c < o: BAR_DIR[si, di] = -1

    # Upper shadow ratio: (H - max(C,O)) / (H-L) — selling pressure
    UPPER_SHADOW = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            h = H[si, di]; l = L[si, di]; c = C[si, di]; o = O[si, di]
            if any(np.isnan(x) for x in [h, l, c, o]) or h == l:
                continue
            UPPER_SHADOW[si, di] = (h - max(c, o)) / (h - l)

    # Z-score of daily returns (20-day rolling)
    ZSCORE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = RET[si, di-20:di]
            valid = rets[~np.isnan(rets)]
            if len(valid) < 10:
                continue
            mean_r = np.mean(valid)
            std_r = np.std(valid, ddof=1)
            if std_r > 0 and not np.isnan(RET[si, di]):
                ZSCORE[si, di] = (RET[si, di] - mean_r) / std_r

    # Intraday range as % of close
    RANGE_PCT = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            h = H[si, di]; l = L[si, di]; c = C[si, di]
            if any(np.isnan(x) for x in [h, l, c]) or c <= 0:
                continue
            RANGE_PCT[si, di] = (h - l) / c * 100

    # Volume ratio (today / 20-day avg)
    VOL_RATIO = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            vols = V[si, di-20:di]
            valid = vols[~np.isnan(vols)]
            if len(valid) < 10 or np.mean(valid) == 0:
                continue
            if not np.isnan(V[si, di]):
                VOL_RATIO[si, di] = V[si, di] / np.mean(valid)

    # OI change ratio (today / 5-day avg)
    OI_CHANGE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(6, ND):
            ois = OI[si, di-5:di]
            valid = ois[~np.isnan(ois)]
            if len(valid) < 3 or np.mean(valid) == 0:
                continue
            if not np.isnan(OI[si, di]):
                OI_CHANGE[si, di] = OI[si, di] / np.mean(valid) - 1

    # Pullback leg counter: count consecutive down days (for pullback detection)
    PULLBACK_LEGS = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        in_pullback = 0
        for di in range(1, ND):
            if BAR_DIR[si, di] == -1:  # bear bar
                in_pullback += 1
            else:
                if in_pullback >= 2:
                    PULLBACK_LEGS[si, di] = in_pullback  # today starts recovery
                in_pullback = 0

    print(f"  All indicators computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # GENERIC BACKTEST ENGINE (supports LONG and SHORT)
    # ================================================================
    def backtest_signal(signal_func, hold_days=1, top_n=1,
                        start_di=MIN_TRAIN, end_di=None,
                        return_trades=False, desc=""):
        """
        signal_func(si, di) -> (score, direction) or None
        direction: 1=long, -1=short
        """
        if end_di is None:
            end_di = ND

        cash = float(CASH0)
        positions = []
        trades = []
        daily_equity = []

        for di in range(start_di, end_di - 1):
            port_val = cash
            for pos in positions:
                cp = C[pos['si'], di]
                if not np.isnan(cp) and cp > 0:
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    if pos['dir'] == 1:
                        port_val += cp * mult * pos['lots'] - cp * mult * abs(pos['lots']) * COMM
                    else:  # short
                        entry_val = pos['entry_price'] * mult * abs(pos['lots'])
                        cur_val = (2 * pos['entry_price'] - cp) * mult * abs(pos['lots'])
                        port_val += cur_val - cp * mult * abs(pos['lots']) * COMM
            daily_equity.append(port_val)

            # Close positions
            closed = []
            for pos in positions:
                days_held = di - pos['entry_di']
                if days_held >= pos['hold_days']:
                    exit_price = C[pos['si'], di]
                    if np.isnan(exit_price) or exit_price <= 0:
                        exit_price = pos['entry_price']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    if pos['dir'] == 1:
                        pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
                    else:  # short
                        pnl = (pos['entry_price'] - exit_price) * mult * abs(pos['lots'])
                    invested = pos['entry_price'] * mult * abs(pos['lots'])
                    pnl_pct = pnl / invested * 100 if invested > 0 else 0
                    mkt_val = exit_price * mult * abs(pos['lots'])
                    if pos['dir'] == 1:
                        cash += mkt_val - mkt_val * COMM
                    else:
                        cash += 2 * pos['entry_price'] * mult * abs(pos['lots']) - mkt_val - mkt_val * COMM
                    trades.append({
                        'pnl': pnl, 'pnl_pct': pnl_pct,
                        'entry_di': pos['entry_di'], 'exit_di': di,
                        'sym': pos['sym'], 'dir': pos['dir'],
                        'entry_price': pos['entry_price'], 'exit_price': exit_price,
                    })
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            if len(positions) >= top_n:
                continue

            # Generate signals
            entry_di = di + 1
            if entry_di >= end_di:
                continue

            candidates = []
            for s in range(NS):
                sig = signal_func(s, di)
                if sig is None:
                    continue
                score, direction = sig
                ep = O[s, entry_di]
                if np.isnan(ep) or ep <= 0:
                    continue
                if any(p['si'] == s for p in positions):
                    continue
                candidates.append((score, s, ep, direction))

            if not candidates:
                continue

            candidates.sort(key=lambda x: -x[0])
            n_slots = top_n - len(positions)
            cap_per_slot = cash / max(1, n_slots)

            for score_val, s, price, direction in candidates[:max(0, n_slots)]:
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
                    'lots': contracts, 'dir': direction, 'sym': sym,
                    'hold_days': hold_days,
                })

        # Close remaining
        for pos in positions:
            ae = end_di - 1
            exit_price = C[pos['si'], min(ae, ND-1)]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos['entry_price']
            mult = MULT.get(pos['sym'], DEF_MULT)
            if pos['dir'] == 1:
                pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
            else:
                pnl = (pos['entry_price'] - exit_price) * mult * abs(pos['lots'])
            invested = pos['entry_price'] * mult * abs(pos['lots'])
            pnl_pct = pnl / invested * 100 if invested > 0 else 0
            mkt_val = exit_price * mult * abs(pos['lots'])
            if pos['dir'] == 1:
                cash += mkt_val - mkt_val * COMM
            else:
                cash += 2 * pos['entry_price'] * mult * abs(pos['lots']) - mkt_val - mkt_val * COMM
            trades.append({
                'pnl': pnl, 'pnl_pct': pnl_pct,
                'entry_di': pos['entry_di'], 'exit_di': ae,
                'sym': pos['sym'], 'dir': pos['dir'],
                'entry_price': pos['entry_price'], 'exit_price': exit_price,
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
            mdd = np.min(dd_arr)
        else:
            mdd = 0.0

        result = {
            'ann': ann, 'wr': wr, 'n': n_trades, 'avg_pnl': avg_pnl,
            'final_cash': cash, 'n_days': n_days_test, 'mdd': mdd,
            'trades': trades, 'daily_equity': daily_equity, 'desc': desc,
        }
        return result

    # ================================================================
    # WALK-FORWARD HELPER
    # ================================================================
    def walk_forward(signal_func, hold_days=1, top_n=1, desc=""):
        wf_results = {}
        test_years = [2020, 2021, 2022, 2023, 2024, 2025]
        for yr in test_years:
            yr_start = None
            yr_end = None
            for di in range(ND):
                if dates[di].year == yr and yr_start is None:
                    yr_start = di
                if dates[di].year == yr:
                    yr_end = di + 1
            if yr_start is None:
                continue
            r = backtest_signal(signal_func, hold_days=hold_days, top_n=top_n,
                                start_di=yr_start, end_di=yr_end, desc=f"{desc} {yr}")
            wf_results[yr] = r['ann']
        return wf_results

    def print_result(r, label=""):
        desc = r.get('desc', '')
        print(f"  {label:40s} | Ann={r['ann']:+8.1f}% | WR={r['wr']:5.1f}% | "
              f"N={r['n']:4d} | Avg={r['avg_pnl']:+5.2f}% | MDD={r['mdd']:6.1f}% | "
              f"Cash={r['final_cash']:>14,.0f}")

    # ================================================================
    # SIGNAL DEFINITIONS
    # ================================================================

    # --- A) Strong Trend Bar Continuation ---
    # Brooks: "A strong bull bar closing on its high in a bull trend = buy"
    # Large body (>60% of range) + up close + momentum + z-score extreme
    def signal_trend_bar(si, di):
        roc5 = ROC5[si, di]
        zs = ZSCORE[si, di]
        br = BODY_RATIO[si, di]
        bd = BAR_DIR[si, di]
        if np.isnan(roc5) or np.isnan(zs) or np.isnan(br):
            return None
        # Strong bull bar with momentum
        if bd != 1:  # must be up bar
            return None
        if roc5 <= 1.0:  # must have momentum
            return None
        if br <= 0.5:  # body must be > 50% of range (strong bar)
            return None
        score = roc5 * br  # momentum * bar strength
        return (score, 1)

    # --- A2) Enhanced Trend Bar: add Z-score requirement ---
    def signal_trend_bar_z(si, di):
        sig = signal_trend_bar(si, di)
        if sig is None:
            return None
        score, direction = sig
        zs = ZSCORE[si, di]
        if np.isnan(zs) or zs <= 1.0:
            return None
        score = score * zs  # trend bar strength * z-score
        return (score, 1)

    # --- B) Two-legged Pullback Entry ---
    # Brooks: "In a bull trend, after 2 legs down, buy the first bull reversal bar"
    # ROC(20) > 0 (uptrend) + 2+ consecutive down days just ended + today is up
    def signal_two_leg_pullback(si, di):
        roc20 = ROC20[si, di]
        roc5 = ROC5[si, di]
        if np.isnan(roc20) or roc20 <= 2.0:  # must be in clear uptrend
            return None
        if di < 3:
            return None
        # Check: 2+ down days recently, today is up (reversal)
        pullback = 0
        lookback = min(di, 6)
        for k in range(1, lookback + 1):
            if BAR_DIR[si, di - k] == -1:
                pullback += 1
            else:
                break
        if pullback < 2:  # need at least 2 pullback legs
            return None
        if BAR_DIR[si, di] != 1:  # today must be bull bar (reversal)
            return None
        # Body ratio of reversal bar
        br = BODY_RATIO[si, di]
        if np.isnan(br) or br < 0.3:
            return None
        score = roc20 * br  # trend strength * reversal bar quality
        return (score, 1)

    # --- C) Final Flag Breakout ---
    # Brooks: "After a strong move, a trading range forms. Breakout in trend direction = continuation."
    # Strong trend (ROC20>5%) + recent 5-day range < ATR(14)*0.7 (consolidation)
    # + today breaks above 5-day high
    def signal_final_flag(si, di):
        roc20 = ROC20[si, di]
        if np.isnan(roc20) or roc20 <= 3.0:  # must be strong trend
            return None
        if di < 6:
            return None
        # Check consolidation: 5-day range vs ATR
        highs_5 = H[si, di-4:di+1]
        lows_5 = L[si, di-4:di+1]
        if any(np.isnan(x) for x in highs_5) or any(np.isnan(x) for x in lows_5):
            return None
        range_5 = np.max(highs_5) - np.min(lows_5)
        atr = ATR14[si, di]
        if np.isnan(atr) or atr <= 0:
            return None
        if range_5 > atr * 3.5:  # not consolidated enough
            return None
        # Today must break above 4-day high (breakout)
        high_4 = np.max(H[si, di-4:di])
        c = C[si, di]
        if np.isnan(c) or c <= high_4:  # not a breakout
            return None
        score = roc20 * (c - high_4) / atr  # trend * breakout strength
        return (score, 1)

    # --- D) Buying Climax Exhaustion (SHORT) ---
    # Brooks: "Buying climax: huge bull bar, massive volume, close on high → exhaustion → short"
    # Z-score > 2.5 + body > 80% + large range + volume spike
    def signal_climax_short(si, di):
        zs = ZSCORE[si, di]
        br = BODY_RATIO[si, di]
        bd = BAR_DIR[si, di]
        rp = RANGE_PCT[si, di]
        vr = VOL_RATIO[si, di]
        if any(np.isnan(x) for x in [zs, br, rp]):
            return None
        if bd != 1:  # must be up bar
            return None
        if zs <= 2.5:  # extreme move
            return None
        if br <= 0.7:  # very large body
            return None
        # Check if RSI overbought
        rsi = RSI14[si, di]
        if not np.isnan(rsi) and rsi < 70:
            return None
        # Volume spike bonus
        vol_mult = vr if not np.isnan(vr) else 1.0
        score = zs * br * vol_mult  # extreme * body * volume
        return (score, -1)  # SHORT

    # --- E) Channel Walker (Bollinger Bounce in Uptrend) ---
    # Brooks: "In a broad bull channel, buy when price touches the bottom of the channel."
    # ROC(20) > 0 + close near lower BB + RSI not yet oversold
    def signal_channel_walker(si, di):
        roc20 = ROC20[si, di]
        roc5 = ROC5[si, di]
        c = C[si, di]
        bb_low = BB_LOW[si, di]
        bb_mid = BB_MID[si, di]
        rsi = RSI14[si, di]
        if any(np.isnan(x) for x in [roc20, c, bb_low, bb_mid]):
            return None
        if roc20 <= 0:  # must be in uptrend
            return None
        if roc5 > 0:  # must be pulling back (negative short-term momentum)
            return None
        # Price must be near or below lower BB
        bb_width = BB_UP[si, di] - BB_LOW[si, di] if not np.isnan(BB_UP[si, di]) else 0
        if bb_width <= 0:
            return None
        dist_to_low = (c - bb_low) / bb_width
        if dist_to_low > 0.15:  # must be within bottom 15% of BB
            return None
        # RSI should show some weakness but not extreme oversold
        if not np.isnan(rsi) and rsi < 25:
            return None
        score = roc20 * (1 - dist_to_low)  # stronger trend + closer to bottom
        return (score, 1)

    # --- F) BROOKS COMPOSITE ---
    # Best elements from above combined:
    # Primary: Trend Bar + Z-score (strongest signal from V121)
    # Secondary: Two-leg pullback in strong trend
    # Exit: Climax detection for short signals
    def signal_composite(si, di):
        # Try trend bar with z-score first (strongest)
        sig_a2 = signal_trend_bar_z(si, di)
        if sig_a2 is not None:
            # Boost if also OI increasing (new money flowing in)
            oi_ch = OI_CHANGE[si, di]
            if not np.isnan(oi_ch) and oi_ch > 0:
                score = sig_a2[0] * (1 + oi_ch)
            else:
                score = sig_a2[0]
            return (score, 1)

        # Try two-leg pullback as secondary
        sig_b = signal_two_leg_pullback(si, di)
        if sig_b is not None:
            # Boost with ADX (stronger trend = better pullback entry)
            adx = ADX14[si, di]
            if not np.isnan(adx) and adx > 25:
                return (sig_b[0] * 1.5, 1)
            return sig_b

        # Try final flag breakout
        sig_c = signal_final_flag(si, di)
        if sig_c is not None:
            return sig_c

        return None

    # --- G) V121 CHAMPION (baseline for comparison) ---
    def signal_v121(si, di):
        roc5 = ROC5[si, di]
        zs = ZSCORE[si, di]
        if np.isnan(roc5) or np.isnan(zs):
            return None
        if roc5 <= 1.0 or zs <= 1.5:
            return None
        # ROC improving filter
        roc_prev = ROC5[si, di-1] if di > 0 else np.nan
        if not np.isnan(roc_prev) and roc5 <= roc_prev:
            return None
        score = roc5 * zs
        return (score, 1)

    # ================================================================
    # RUN ALL STRATEGIES
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 1: FULL SAMPLE BACKTEST (ALL YEARS)")
    print("=" * 120)

    strategies = [
        ("A) Trend Bar Cont.", signal_trend_bar, 1, 1),
        ("A2) Trend Bar + Z", signal_trend_bar_z, 1, 1),
        ("B) Two-leg Pullback", signal_two_leg_pullback, 1, 1),
        ("C) Final Flag Break", signal_final_flag, 1, 1),
        ("D) Climax Fade SHORT", signal_climax_short, 1, 1),
        ("E) Channel Walker", signal_channel_walker, 1, 1),
        ("F) Brooks Composite", signal_composite, 1, 1),
        ("G) V121 Champion", signal_v121, 1, 1),
    ]

    results = {}
    for name, func, hold, topn in strategies:
        r = backtest_signal(func, hold_days=hold, top_n=topn, desc=name)
        results[name] = r
        print_result(r, label=name)

    # Find best
    best_name = max(results, key=lambda k: results[k]['ann'])
    best_r = results[best_name]
    print(f"\n  >>> BEST: {best_name} with {best_r['ann']:+.1f}% annual")

    # ================================================================
    # SECTION 2: HOLD PERIOD SWEEP FOR BEST
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 2: HOLD PERIOD SWEEP (best strategy)")
    print("=" * 120)

    best_func = dict((n, f) for n, f, _, _ in strategies)[best_name]
    for hold in [1, 2, 3, 5, 10]:
        r = backtest_signal(best_func, hold_days=hold, top_n=1, desc=f"{best_name} hold={hold}")
        print_result(r, label=f"hold={hold}")

    # ================================================================
    # SECTION 3: TOP_N SWEEP FOR BEST
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 3: TOP_N SWEEP (best strategy)")
    print("=" * 120)

    for topn in [1, 2, 3, 5]:
        r = backtest_signal(best_func, hold_days=1, top_n=topn, desc=f"{best_name} top_n={topn}")
        print_result(r, label=f"top_n={topn}")

    # ================================================================
    # SECTION 4: WALK-FORWARD VALIDATION
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 4: WALK-FORWARD VALIDATION (by year)")
    print("=" * 120)

    for name, func, hold, topn in strategies:
        wf = walk_forward(func, hold_days=hold, top_n=topn, desc=name)
        wf_str = " | ".join([f"{yr}:{ann:+.0f}%" for yr, ann in sorted(wf.items())])
        positive = sum(1 for v in wf.values() if v > 0)
        print(f"  {name:40s} | {positive}/6 WF | {wf_str}")

    # ================================================================
    # SECTION 5: PARAMETER SENSITIVITY FOR BEST SIGNAL
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 5: PARAMETER SENSITIVITY")
    print("=" * 120)

    # Sensitivity for Trend Bar body ratio threshold
    print("\n  A2) Trend Bar + Z: body_ratio threshold sweep")
    for br_thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        def make_signal(br_t):
            def sig(si, di):
                roc5 = ROC5[si, di]
                zs = ZSCORE[si, di]
                br = BODY_RATIO[si, di]
                bd = BAR_DIR[si, di]
                if any(np.isnan(x) for x in [roc5, zs, br]):
                    return None
                if bd != 1 or roc5 <= 1.0 or br <= br_t or zs <= 1.0:
                    return None
                return (roc5 * br * zs, 1)
            return sig
        r = backtest_signal(make_signal(br_thresh), hold_days=1, top_n=1, desc=f"BR>{br_thresh}")
        print_result(r, label=f"BR>{br_thresh:.1f}")

    # Sensitivity for Two-leg Pullback ROC(20) threshold
    print("\n  B) Two-leg Pullback: ROC(20) threshold sweep")
    for roc20_t in [0, 2, 5, 8, 10, 15]:
        def make_signal(r20):
            def sig(si, di):
                roc20 = ROC20[si, di]
                if np.isnan(roc20) or roc20 <= r20:
                    return None
                if di < 3:
                    return None
                pb = 0
                for k in range(1, min(di, 6) + 1):
                    if BAR_DIR[si, di - k] == -1:
                        pb += 1
                    else:
                        break
                if pb < 2 or BAR_DIR[si, di] != 1:
                    return None
                br = BODY_RATIO[si, di]
                if np.isnan(br) or br < 0.3:
                    return None
                return (roc20 * br, 1)
            return sig
        r = backtest_signal(make_signal(roc20_t), hold_days=1, top_n=1, desc=f"ROC20>{roc20_t}")
        print_result(r, label=f"ROC20>{roc20_t}")

    # Sensitivity for Final Flag consolidation tightness
    print("\n  C) Final Flag: consolidation tightness sweep")
    for tight in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        def make_signal(t):
            def sig(si, di):
                roc20 = ROC20[si, di]
                if np.isnan(roc20) or roc20 <= 3.0:
                    return None
                if di < 6:
                    return None
                highs_5 = H[si, di-4:di+1]
                lows_5 = L[si, di-4:di+1]
                if any(np.isnan(x) for x in highs_5) or any(np.isnan(x) for x in lows_5):
                    return None
                range_5 = np.max(highs_5) - np.min(lows_5)
                atr = ATR14[si, di]
                if np.isnan(atr) or atr <= 0:
                    return None
                if range_5 > atr * t:
                    return None
                high_4 = np.max(H[si, di-4:di])
                c = C[si, di]
                if np.isnan(c) or c <= high_4:
                    return None
                return (roc20 * (c - high_4) / atr, 1)
            return sig
        r = backtest_signal(make_signal(tight), hold_days=1, top_n=1, desc=f"tight={tight}")
        print_result(r, label=f"tight={tight}")

    # ================================================================
    # SECTION 6: ENHANCED COMPOSITE WITH BEST PARAMS
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 6: ENHANCED COMPOSITE VARIANTS")
    print("=" * 120)

    # Composite v2: Best of each category
    def signal_composite_v2(si, di):
        candidates = []

        # Trend Bar (relaxed params)
        roc5 = ROC5[si, di]
        zs = ZSCORE[si, di]
        br = BODY_RATIO[si, di]
        bd = BAR_DIR[si, di]
        if not any(np.isnan(x) for x in [roc5, zs, br]) and bd == 1 and roc5 > 1.0 and zs > 1.5 and br > 0.5:
            # ROC improving check
            roc_prev = ROC5[si, di-1] if di > 0 else np.nan
            if np.isnan(roc_prev) or roc5 > roc_prev:
                oi_ch = OI_CHANGE[si, di]
                oi_boost = 1 + max(0, oi_ch) if not np.isnan(oi_ch) else 1
                candidates.append((roc5 * zs * br * oi_boost * 3.0, 1, "TB+Z"))

        # Two-leg pullback
        roc20 = ROC20[si, di]
        if not np.isnan(roc20) and roc20 > 3.0 and di >= 3:
            pb = 0
            for k in range(1, min(di, 6) + 1):
                if BAR_DIR[si, di - k] == -1:
                    pb += 1
                else:
                    break
            if pb >= 2 and bd == 1 and not np.isnan(br) and br >= 0.3:
                adx = ADX14[si, di]
                adx_boost = 1.5 if not np.isnan(adx) and adx > 25 else 1.0
                candidates.append((roc20 * br * adx_boost, 1, "Pullback"))

        # Final Flag
        if not np.isnan(roc20) and roc20 > 5.0 and di >= 6:
            highs_5 = H[si, di-4:di+1]
            lows_5 = L[si, di-4:di+1]
            if not any(np.isnan(x) for x in highs_5) and not any(np.isnan(x) for x in lows_5):
                range_5 = np.max(highs_5) - np.min(lows_5)
                atr = ATR14[si, di]
                if not np.isnan(atr) and atr > 0:
                    if range_5 <= atr * 3.0:
                        high_4 = np.max(H[si, di-4:di])
                        c = C[si, di]
                        if not np.isnan(c) and c > high_4:
                            candidates.append((roc20 * (c - high_4) / atr, 1, "FinalFlag"))

        if not candidates:
            return None
        candidates.sort(key=lambda x: -x[0])
        return (candidates[0][0], candidates[0][1])

    # Composite v3: Trend Bar + Pullback only (most promising from Brooks)
    def signal_composite_v3(si, di):
        roc5 = ROC5[si, di]
        zs = ZSCORE[si, di]
        br = BODY_RATIO[si, di]
        bd = BAR_DIR[si, di]

        # Primary: Trend bar with z-score + ROC improving
        if not any(np.isnan(x) for x in [roc5, zs, br]) and bd == 1:
            if roc5 > 1.0 and zs > 1.5 and br > 0.5:
                roc_prev = ROC5[si, di-1] if di > 0 else np.nan
                if np.isnan(roc_prev) or roc5 > roc_prev:
                    # Volume confirmation
                    vr = VOL_RATIO[si, di]
                    vol_bonus = 1 + max(0, (vr - 1) * 0.5) if not np.isnan(vr) else 1.0
                    return (roc5 * zs * br * vol_bonus, 1)

        # Secondary: Two-leg pullback in strong trend + body > 0.4
        roc20 = ROC20[si, di]
        if not np.isnan(roc20) and roc20 > 5.0 and di >= 3:
            pb = 0
            for k in range(1, min(di, 5) + 1):
                if BAR_DIR[si, di - k] == -1:
                    pb += 1
                else:
                    break
            if pb >= 2 and bd == 1 and not np.isnan(br) and br >= 0.4:
                return (roc20 * br * 0.5, 1)  # Scale down to not dominate

        return None

    for name, func in [("Composite v2", signal_composite_v2),
                       ("Composite v3", signal_composite_v3)]:
        r = backtest_signal(func, hold_days=1, top_n=1, desc=name)
        results[name] = r
        print_result(r, label=name)

        # Walk-forward for composites
        wf = walk_forward(func, hold_days=1, top_n=1, desc=name)
        wf_str = " | ".join([f"{yr}:{ann:+.0f}%" for yr, ann in sorted(wf.items())])
        positive = sum(1 for v in wf.values() if v > 0)
        print(f"  {'WF':40s} | {positive}/6 WF | {wf_str}")

    # ================================================================
    # SECTION 7: COMPOSITE WITH HOLD SWEEP
    # ================================================================
    print("\n" + "=" * 120)
    print("  SECTION 7: BEST COMPOSITE HOLD + TOP_N SWEEP")
    print("=" * 120)

    for func_name in ["Composite v2", "Composite v3"]:
        func = signal_composite_v2 if "v2" in func_name else signal_composite_v3
        print(f"\n  {func_name}:")
        for hold in [1, 2, 3, 5]:
            for topn in [1, 2, 3]:
                r = backtest_signal(func, hold_days=hold, top_n=topn, desc=f"{func_name} h={hold} t={topn}")
                print(f"    hold={hold} top_n={topn}: Ann={r['ann']:+8.1f}% | WR={r['wr']:5.1f}% | N={r['n']:4d} | MDD={r['mdd']:6.1f}%")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 120)
    print("  SUMMARY: ALL STRATEGIES RANKED")
    print("=" * 120)

    all_results = sorted(results.items(), key=lambda x: -x[1]['ann'])
    for i, (name, r) in enumerate(all_results):
        print(f"  #{i+1}: {name:40s} | Ann={r['ann']:+8.1f}% | WR={r['wr']:5.1f}% | N={r['n']:4d} | MDD={r['mdd']:6.1f}%")

    print(f"\n  Total elapsed: {time.time()-t_start:.0f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
