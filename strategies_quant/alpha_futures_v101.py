"""
Alpha Futures V101 -- Classic Technical Analysis with Next-Open Execution
==========================================================================
V99 best next-open: +6.9% (trend quality).

V101 tests CLASSIC TA indicators that should survive the 1-day delay
because they capture multi-day structural conditions:

  A) RSI Oversold Bounce: RSI < 30, hold 5/10/20
  B) RSI Divergence: price new low but RSI higher low, hold 5/10
  C) Bollinger Band: lower band touch + squeeze breakout, hold 5/10
  D) MACD Crossover: MACD crosses above signal, hold 10/20
  E) Support/Resistance Breakout: 50-day high breakout, hold 10/20
  F) Volume Breakout: price > MA + volume > 2x MA, hold 5/10
  G) Multi-Timeframe RSI: weekly uptrend + daily oversold, hold 5/10
  H) Candlestick Patterns: hammer + bullish engulfing, hold 3/5
  I) ATR Trailing Stop: turtle-style trailing exit

ALL signals: computed at close of day di, entry at O[si, di+1] (NEXT DAY OPEN).
Walk-forward: top 15 configs across 2020-2025.
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
    print("=" * 150)
    print("Alpha Futures V101 -- Classic Technical Analysis (Next-Open Execution)")
    print("=" * 150)
    print("\n  Testing classic TA indicators: RSI, Bollinger, MACD, Breakouts, Candlesticks, ATR")
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

    # ---- A) RSI (14-day Wilder's smoothing) ----
    RSI = np.full((NS, ND), np.nan)
    for si in range(NS):
        gains = []
        losses = []
        # First pass: compute initial average gain/loss over 14 periods
        for di in range(1, 15):
            cn = C[si, di]
            cp = C[si, di - 1]
            if np.isnan(cn) or np.isnan(cp):
                continue
            chg = cn - cp
            if chg > 0:
                gains.append(chg)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(-chg)
        if len(gains) < 14:
            continue
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            RSI[si, 14] = 100.0
        else:
            rs = avg_gain / avg_loss
            RSI[si, 14] = 100.0 - (100.0 / (1.0 + rs))

        # Wilder's smoothing for subsequent periods
        for di in range(15, ND):
            cn = C[si, di]
            cp = C[si, di - 1]
            if np.isnan(cn) or np.isnan(cp):
                continue
            chg = cn - cp
            gain = chg if chg > 0 else 0.0
            loss = -chg if chg < 0 else 0.0
            avg_gain = (avg_gain * 13 + gain) / 14.0
            avg_loss = (avg_loss * 13 + loss) / 14.0
            if avg_loss == 0:
                RSI[si, di] = 100.0
            else:
                rs = avg_gain / avg_loss
                RSI[si, di] = 100.0 - (100.0 / (1.0 + rs))
    print(f"  RSI computed ({time.time()-t0:.1f}s)")

    # ---- B) RSI Divergence: price new 20-day low BUT RSI makes higher low ----
    rsi_div_signal = np.zeros((NS, ND), dtype=bool)
    rsi_div_score = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(25, ND):
            c_cur = C[si, di]
            c_low20 = np.nanmin(C[si, di-20:di+1])
            if np.isnan(c_cur) or np.isnan(c_low20):
                continue
            if c_cur != c_low20:
                continue  # today must be at 20-day low
            # Find previous 20-day low (look further back)
            prev_low_region = C[si, max(0,di-50):di-5]
            if len(prev_low_region) == 0:
                continue
            prev_low = np.nanmin(prev_low_region)
            if np.isnan(prev_low):
                continue
            # Price made lower low (current 20d low < previous low region)
            if c_low20 >= prev_low:
                continue  # need lower low
            # Find RSI at both points
            rsi_cur = RSI[si, di]
            if np.isnan(rsi_cur):
                continue
            # Find di of previous low
            prev_low_dis = [k for k in range(max(0,di-50), di-5)
                           if not np.isnan(C[si,k]) and C[si,k] == prev_low]
            rsi_at_prev = None
            for pdi in prev_low_dis:
                if not np.isnan(RSI[si, pdi]):
                    rsi_at_prev = RSI[si, pdi]
                    break
            if rsi_at_prev is None:
                continue
            # Divergence: price lower but RSI higher
            if rsi_cur > rsi_at_prev:
                rsi_div_signal[si, di] = True
                rsi_div_score[si, di] = rsi_cur - rsi_at_prev
    print(f"  RSI Divergence computed ({time.time()-t0:.1f}s)")

    # ---- C) Bollinger Bands (20-day SMA +/- 2*std) ----
    bb_lower = np.full((NS, ND), np.nan)
    bb_upper = np.full((NS, ND), np.nan)
    bb_ma = np.full((NS, ND), np.nan)
    bb_width = np.full((NS, ND), np.nan)
    bb_lower_touch = np.zeros((NS, ND), dtype=bool)
    bb_squeeze = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(20, ND):
            window = C[si, di-20:di]
            if np.sum(~np.isnan(window)) < 18:
                continue
            ma = np.nanmean(window)
            std = np.nanstd(window, ddof=1)
            if np.isnan(ma) or np.isnan(std) or ma <= 0:
                continue
            bb_ma[si, di] = ma
            bb_lower[si, di] = ma - 2 * std
            bb_upper[si, di] = ma + 2 * std
            bb_width[si, di] = (bb_upper[si, di] - bb_lower[si, di]) / ma

            c = C[si, di]
            if not np.isnan(c) and c < bb_lower[si, di]:
                bb_lower_touch[si, di] = True

            # Squeeze: bandwidth at 20-day minimum
            if di >= 40:
                bw_window = bb_width[si, di-20:di]
                if not np.any(np.isnan(bw_window)):
                    if bb_width[si, di] <= np.min(bw_window):
                        bb_squeeze[si, di] = True
    print(f"  Bollinger Bands computed ({time.time()-t0:.1f}s)")

    # ---- D) MACD (12, 26, 9) ----
    MACD_line = np.full((NS, ND), np.nan)
    MACD_signal = np.full((NS, ND), np.nan)
    MACD_hist = np.full((NS, ND), np.nan)
    macd_cross = np.zeros((NS, ND), dtype=bool)  # bullish cross above signal

    for si in range(NS):
        # Compute EMA12 and EMA26
        ema12 = np.full(ND, np.nan)
        ema26 = np.full(ND, np.nan)
        # Find first valid close to seed EMA
        first_valid = None
        for di in range(ND):
            if not np.isnan(C[si, di]):
                first_valid = di
                break
        if first_valid is None:
            continue

        # Seed EMAs with first price
        ema12[first_valid] = C[si, first_valid]
        ema26[first_valid] = C[si, first_valid]

        k12 = 2.0 / 13.0  # smoothing factor for 12-period EMA
        k26 = 2.0 / 27.0
        for di in range(first_valid + 1, ND):
            c = C[si, di]
            if np.isnan(c):
                continue
            if not np.isnan(ema12[di-1]):
                ema12[di] = c * k12 + ema12[di-1] * (1 - k12)
            else:
                ema12[di] = c
            if not np.isnan(ema26[di-1]):
                ema26[di] = c * k26 + ema26[di-1] * (1 - k26)
            else:
                ema26[di] = c

        # MACD line = EMA12 - EMA26
        macd_vals = np.full(ND, np.nan)
        for di in range(ND):
            if not np.isnan(ema12[di]) and not np.isnan(ema26[di]):
                macd_vals[di] = ema12[di] - ema26[di]

        # Signal line = 9-period EMA of MACD
        sig_vals = np.full(ND, np.nan)
        first_macd = None
        for di in range(ND):
            if not np.isnan(macd_vals[di]):
                first_macd = di
                break
        if first_macd is None:
            continue
        sig_vals[first_macd] = macd_vals[first_macd]
        k9 = 2.0 / 10.0
        for di in range(first_macd + 1, ND):
            if np.isnan(macd_vals[di]):
                continue
            if not np.isnan(sig_vals[di-1]):
                sig_vals[di] = macd_vals[di] * k9 + sig_vals[di-1] * (1 - k9)
            else:
                sig_vals[di] = macd_vals[di]

        for di in range(ND):
            MACD_line[si, di] = macd_vals[di]
            MACD_signal[si, di] = sig_vals[di]
            if not np.isnan(macd_vals[di]) and not np.isnan(sig_vals[di]):
                MACD_hist[si, di] = macd_vals[di] - sig_vals[di]

        # Detect bullish crossover
        for di in range(1, ND):
            h_cur = MACD_hist[si, di]
            h_prev = MACD_hist[si, di-1]
            if np.isnan(h_cur) or np.isnan(h_prev):
                continue
            if h_prev <= 0 and h_cur > 0:  # cross from negative to positive
                macd_cross[si, di] = True
    print(f"  MACD computed ({time.time()-t0:.1f}s)")

    # ---- E) 50-day high/low breakout (Donchian) ----
    high50 = np.full((NS, ND), np.nan)
    low50 = np.full((NS, ND), np.nan)
    breakout_high = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(50, ND):
            h50 = np.nanmax(H[si, di-50:di])
            if np.isnan(h50):
                continue
            high50[si, di] = h50
            c = C[si, di]
            if not np.isnan(c) and c > h50:
                breakout_high[si, di] = True
    print(f"  50-day Breakout computed ({time.time()-t0:.1f}s)")

    # ---- F) Volume Breakout ----
    vol_breakout = np.zeros((NS, ND), dtype=bool)
    vol_breakout_score = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            # 20-day MA of close and volume
            c_win = C[si, di-20:di]
            v_win = V[si, di-20:di]
            c_ma = np.nanmean(c_win)
            v_ma = np.nanmean(v_win)
            c_cur = C[si, di]
            v_cur = V[si, di]
            if np.isnan(c_ma) or np.isnan(v_ma) or v_ma <= 0:
                continue
            if np.isnan(c_cur) or np.isnan(v_cur):
                continue
            if c_cur > c_ma and v_cur > 2.0 * v_ma:
                vol_breakout[si, di] = True
                vol_breakout_score[si, di] = v_cur / v_ma
    print(f"  Volume Breakout computed ({time.time()-t0:.1f}s)")

    # ---- G) Multi-Timeframe RSI (weekly + daily) ----
    # Build weekly bars
    from datetime import datetime
    week_map = {}
    for di in range(ND):
        d = dates[di]
        iso = d.isocalendar()
        wk_key = (iso[0], iso[1])
        week_map.setdefault(wk_key, []).append(di)

    # Weekly closes per symbol
    weekly_close_arr = {}  # si -> list of (last_di, close_price)
    for si in range(NS):
        wc = []
        for wk_key in sorted(week_map.keys()):
            dis = week_map[wk_key]
            for d in reversed(dis):
                c = C[si, d]
                if not np.isnan(c) and c > 0:
                    wc.append((d, c))
                    break
        weekly_close_arr[si] = wc

    # Compute weekly RSI (14-week)
    weekly_rsi = np.full((NS, ND), np.nan)
    for si in range(NS):
        wc = weekly_close_arr[si]
        if len(wc) < 15:
            continue
        # Weekly returns
        w_rets = []
        for i in range(1, len(wc)):
            d_cur, c_cur = wc[i]
            d_prev, c_prev = wc[i-1]
            if c_prev > 0:
                w_rets.append((d_cur, c_cur - c_prev))

        if len(w_rets) < 14:
            continue

        # Wilder's RSI on weekly changes
        gains = []
        losses = []
        for i in range(14):
            chg = w_rets[i][1]
            if chg > 0:
                gains.append(chg)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(-chg)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            weekly_rsi[si, w_rets[13][0]] = 100.0
        else:
            rs = avg_gain / avg_loss
            weekly_rsi[si, w_rets[13][0]] = 100.0 - (100.0 / (1.0 + rs))

        for i in range(14, len(w_rets)):
            chg = w_rets[i][1]
            gain = chg if chg > 0 else 0.0
            loss = -chg if chg < 0 else 0.0
            avg_gain = (avg_gain * 13 + gain) / 14.0
            avg_loss = (avg_loss * 13 + loss) / 14.0
            if avg_loss == 0:
                rsi_val = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_val = 100.0 - (100.0 / (1.0 + rs))
            # Map to all days in that week
            cur_di = w_rets[i][0]
            # Forward-fill weekly RSI to all subsequent days until next week
            weekly_rsi[si, cur_di] = rsi_val

        # Forward-fill weekly RSI across all days
        last_val = np.nan
        for di in range(ND):
            if not np.isnan(weekly_rsi[si, di]):
                last_val = weekly_rsi[si, di]
            elif not np.isnan(last_val):
                weekly_rsi[si, di] = last_val
    print(f"  Weekly RSI computed ({time.time()-t0:.1f}s)")

    # ---- H) Candlestick Patterns ----
    hammer_signal = np.zeros((NS, ND), dtype=bool)
    engulfing_signal = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(6, ND):
            h = H[si, di]
            l = L[si, di]
            o = O[si, di]
            c = C[si, di]
            if np.isnan(h) or np.isnan(l) or np.isnan(o) or np.isnan(c):
                continue

            body_top = max(o, c)
            body_bot = min(o, c)
            body = body_top - body_bot
            upper_shadow = h - body_top
            lower_shadow = body_bot - l

            # Hammer: lower shadow > 2x body, upper shadow small
            # In downtrend: C[di-5] > C[di]
            if body > 0 and lower_shadow > 2 * body and upper_shadow < body:
                c5ago = C[si, di-5]
                if not np.isnan(c5ago) and c5ago > c:
                    hammer_signal[si, di] = True

            # Bullish engulfing
            o_prev = O[si, di-1]
            c_prev = C[si, di-1]
            if np.isnan(o_prev) or np.isnan(c_prev):
                continue
            if c_prev < o_prev and c > o and c > o_prev and o < c_prev:
                engulfing_signal[si, di] = True
    print(f"  Candlestick patterns computed ({time.time()-t0:.1f}s)")

    # ---- I) ATR (14-day) ----
    ATR14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(15, ND):
            trs = []
            for k in range(di - 14, di + 1):
                h = H[si, k]
                l = L[si, k]
                cp = C[si, k-1] if k > 0 else np.nan
                if np.isnan(h) or np.isnan(l):
                    continue
                tr = h - l
                if not np.isnan(cp) and cp > 0:
                    tr = max(tr, abs(h - cp), abs(l - cp))
                trs.append(tr)
            if len(trs) >= 10:
                ATR14[si, di] = np.mean(trs)
    print(f"  ATR computed ({time.time()-t0:.1f}s)")

    print(f"\n  All indicators computed ({time.time()-t_start:.1f}s total)")

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(config, wf_test_year=None):
        """
        Config:
            signal: 'rsi_oversold' | 'rsi_div' | 'bb_touch' | 'bb_squeeze' |
                    'macd_cross' | 'breakout50' | 'vol_breakout' |
                    'mtf_rsi' | 'hammer' | 'engulfing' | 'atr_trail'
            hold_days: int (ignored for atr_trail)
            threshold: float (signal-specific)
            top_n: int (max concurrent positions)
            comm: float
        """
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
        positions = []  # {si, entry_price, entry_di, lots, dir, sym, hold_days, trail_stop}
        trades = []

        for di in range(start_di, end_di - 1):
            # Reset cash at test window start (WF mode)
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                positions = []

            # -- Close positions -----------------------------------------
            closed = []
            for pos in positions:
                if sig_type == 'atr_trail':
                    # Trailing stop exit: if C[di] < trail_stop
                    c_now = C[pos['si'], di]
                    if not np.isnan(c_now) and c_now <= pos.get('trail_stop', 0):
                        exit_price = c_now
                        mult = MULT.get(pos['sym'], DEF_MULT)
                        mkt_val = exit_price * mult * abs(pos['lots'])
                        cash += mkt_val - mkt_val * comm
                        pnl = (exit_price - pos['entry_price']) * mult * pos['lots'] * pos['dir']
                        invested = pos['entry_price'] * mult * abs(pos['lots'])
                        pnl_pct = pnl / invested * 100 if invested > 0 else 0
                        trades.append({
                            'pnl_pct': pnl_pct,
                            'entry_di': pos['entry_di'],
                            'exit_di': di,
                            'year': dates[di].year if di < ND else dates[-1].year,
                            'dir': pos['dir'],
                            'sym': pos.get('sym', ''),
                        })
                        closed.append(pos)
                    else:
                        # Update trailing stop
                        if not np.isnan(c_now):
                            atr = ATR14[pos['si'], di] if not np.isnan(ATR14[pos['si'], di]) else 0
                            h_since = np.nanmax(H[pos['si'], pos['entry_di']:di+1])
                            if atr > 0 and not np.isnan(h_since):
                                new_stop = h_since - 2 * atr
                                old_stop = pos.get('trail_stop', 0)
                                if new_stop > old_stop:
                                    pos['trail_stop'] = new_stop
                    # Also force close if held too long (50 days)
                    days_held = di - pos['entry_di']
                    if days_held >= 50:
                        exit_price = C[pos['si'], di]
                        if np.isnan(exit_price) or exit_price <= 0:
                            exit_price = pos['entry_price']
                        mult = MULT.get(pos['sym'], DEF_MULT)
                        mkt_val = exit_price * mult * abs(pos['lots'])
                        cash += mkt_val - mkt_val * comm
                        pnl = (exit_price - pos['entry_price']) * mult * pos['lots'] * pos['dir']
                        invested = pos['entry_price'] * mult * abs(pos['lots'])
                        pnl_pct = pnl / invested * 100 if invested > 0 else 0
                        trades.append({
                            'pnl_pct': pnl_pct,
                            'entry_di': pos['entry_di'],
                            'exit_di': di,
                            'year': dates[di].year if di < ND else dates[-1].year,
                            'dir': pos['dir'],
                            'sym': pos.get('sym', ''),
                        })
                        if pos not in closed:
                            closed.append(pos)
                else:
                    days_held = di - pos['entry_di']
                    if days_held >= pos['hold_days']:
                        exit_price = C[pos['si'], di]
                        if np.isnan(exit_price) or exit_price <= 0:
                            exit_price = pos['entry_price']
                        mult = MULT.get(pos['sym'], DEF_MULT)
                        mkt_val = exit_price * mult * abs(pos['lots'])
                        cash += mkt_val - mkt_val * comm
                        pnl = (exit_price - pos['entry_price']) * mult * pos['lots'] * pos['dir']
                        invested = pos['entry_price'] * mult * abs(pos['lots'])
                        pnl_pct = pnl / invested * 100 if invested > 0 else 0
                        trades.append({
                            'pnl_pct': pnl_pct,
                            'entry_di': pos['entry_di'],
                            'exit_di': di,
                            'year': dates[di].year if di < ND else dates[-1].year,
                            'dir': pos['dir'],
                            'sym': pos.get('sym', ''),
                        })
                        closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            # -- Generate signals at day di --------------------------------
            entry_di = di + 1
            if entry_di >= end_di:
                continue

            candidates = []

            if sig_type == 'rsi_oversold':
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    rsi_val = RSI[si, di]
                    if np.isnan(rsi_val):
                        continue
                    if rsi_val < threshold:
                        ep = O[si, entry_di]
                        if np.isnan(ep) or ep <= 0:
                            continue
                        score = threshold - rsi_val  # more oversold = higher score
                        candidates.append((score, 1, {
                            'si': si, 'sym': syms[si], 'entry_price': ep,
                        }))

            elif sig_type == 'rsi_div':
                for si in range(NS):
                    if not rsi_div_signal[si, di]:
                        continue
                    if any(p['si'] == si for p in positions):
                        continue
                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    sc = rsi_div_score[si, di] if not np.isnan(rsi_div_score[si, di]) else 1.0
                    candidates.append((sc, 1, {
                        'si': si, 'sym': syms[si], 'entry_price': ep,
                    }))

            elif sig_type == 'bb_touch':
                for si in range(NS):
                    if not bb_lower_touch[si, di]:
                        continue
                    if any(p['si'] == si for p in positions):
                        continue
                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    # Score: how far below lower band
                    bw = bb_width[si, di] if not np.isnan(bb_width[si, di]) else 0.1
                    candidates.append((1.0/bw, 1, {
                        'si': si, 'sym': syms[si], 'entry_price': ep,
                    }))

            elif sig_type == 'bb_squeeze':
                for si in range(NS):
                    if not bb_squeeze[si, di]:
                        continue
                    if any(p['si'] == si for p in positions):
                        continue
                    c = C[si, di]
                    ma = bb_ma[si, di]
                    if np.isnan(c) or np.isnan(ma):
                        continue
                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    # Buy when price breaks above MA after squeeze
                    if c > ma:
                        bw = bb_width[si, di] if not np.isnan(bb_width[si, di]) else 0.1
                        candidates.append((1.0/bw, 1, {
                            'si': si, 'sym': syms[si], 'entry_price': ep,
                        }))

            elif sig_type == 'macd_cross':
                for si in range(NS):
                    if not macd_cross[si, di]:
                        continue
                    if any(p['si'] == si for p in positions):
                        continue
                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    hist = MACD_hist[si, di] if not np.isnan(MACD_hist[si, di]) else 0
                    candidates.append((hist, 1, {
                        'si': si, 'sym': syms[si], 'entry_price': ep,
                    }))

            elif sig_type == 'breakout50':
                for si in range(NS):
                    if not breakout_high[si, di]:
                        continue
                    if any(p['si'] == si for p in positions):
                        continue
                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    h50 = high50[si, di] if not np.isnan(high50[si, di]) else 0
                    c = C[si, di] if not np.isnan(C[si, di]) else h50
                    score = (c - h50) / h50 * 100 if h50 > 0 else 0
                    candidates.append((score, 1, {
                        'si': si, 'sym': syms[si], 'entry_price': ep,
                    }))

            elif sig_type == 'vol_breakout':
                for si in range(NS):
                    if not vol_breakout[si, di]:
                        continue
                    if any(p['si'] == si for p in positions):
                        continue
                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    sc = vol_breakout_score[si, di] if not np.isnan(vol_breakout_score[si, di]) else 2.0
                    candidates.append((sc, 1, {
                        'si': si, 'sym': syms[si], 'entry_price': ep,
                    }))

            elif sig_type == 'mtf_rsi':
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    wrsi = weekly_rsi[si, di]
                    drsi = RSI[si, di]
                    if np.isnan(wrsi) or np.isnan(drsi):
                        continue
                    if wrsi > threshold and drsi < 30:
                        ep = O[si, entry_di]
                        if np.isnan(ep) or ep <= 0:
                            continue
                        score = wrsi - drsi  # bigger spread = stronger signal
                        candidates.append((score, 1, {
                            'si': si, 'sym': syms[si], 'entry_price': ep,
                        }))

            elif sig_type == 'hammer':
                for si in range(NS):
                    if not hammer_signal[si, di]:
                        continue
                    if any(p['si'] == si for p in positions):
                        continue
                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    candidates.append((1.0, 1, {
                        'si': si, 'sym': syms[si], 'entry_price': ep,
                    }))

            elif sig_type == 'engulfing':
                for si in range(NS):
                    if not engulfing_signal[si, di]:
                        continue
                    if any(p['si'] == si for p in positions):
                        continue
                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    c = C[si, di] if not np.isnan(C[si, di]) else 0
                    o = O[si, di] if not np.isnan(O[si, di]) else 0
                    score = c - o  # bigger body = stronger signal
                    candidates.append((score, 1, {
                        'si': si, 'sym': syms[si], 'entry_price': ep,
                    }))

            elif sig_type == 'atr_trail':
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    atr = ATR14[si, di]
                    if np.isnan(atr) or atr <= 0:
                        continue
                    # 20-day high minus 2*ATR
                    h20 = np.nanmax(C[si, max(0,di-20):di])
                    if np.isnan(h20):
                        continue
                    c = C[si, di]
                    if np.isnan(c):
                        continue
                    # Price must be above the threshold line
                    threshold_line = h20 - 2 * atr
                    if c > threshold_line:
                        ep = O[si, entry_di]
                        if np.isnan(ep) or ep <= 0:
                            continue
                        score = (c - threshold_line) / atr
                        candidates.append((score, 1, {
                            'si': si, 'sym': syms[si], 'entry_price': ep,
                        }))

            if not candidates:
                continue

            # Sort by score descending
            candidates.sort(key=lambda x: -x[0])

            # Open positions (long only)
            n_slots = top_n - len(positions)
            for score, direction, info in candidates[:max(0, n_slots)]:
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

                pos_dict = {
                    'si': si, 'entry_price': price, 'entry_di': entry_di,
                    'lots': lots, 'dir': direction, 'sym': sym,
                    'hold_days': hold_days,
                }
                # For ATR trail, set initial trailing stop
                if sig_type == 'atr_trail':
                    atr = ATR14[si, di] if not np.isnan(ATR14[si, di]) else 0
                    h_entry = H[si, di] if not np.isnan(H[si, di]) else price
                    pos_dict['trail_stop'] = h_entry - 2 * atr if atr > 0 else price * 0.95
                    pos_dict['hold_days'] = 50  # max hold

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

        # Results
        wf_mode = wf_test_year is not None
        n_days_test = (test_end_di - test_start_di) if wf_mode else (end_di - start_di)
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0

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

    # --- A: RSI Oversold Bounce: RSI < 30/25/20, hold 5/10/20, top_n 1/3/5 ---
    for rsi_thresh in [20, 25, 30, 35]:
        for hd in [5, 10, 20]:
            for tn in [1, 3, 5]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': 'rsi_oversold',
                    'hold_days': hd, 'threshold': rsi_thresh,
                    'top_n': tn, 'comm': COMM,
                    'label': f"RSI_OS_T{rsi_thresh}_H{hd}_TN{tn}",
                })

    # --- B: RSI Divergence: hold 5/10, top_n 1/3/5 ---
    for hd in [5, 10]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'rsi_div',
                'hold_days': hd, 'threshold': 0,
                'top_n': tn, 'comm': COMM,
                'label': f"RSI_Div_H{hd}_TN{tn}",
            })

    # --- C: Bollinger Band Lower Touch: hold 5/10, top_n 1/3/5 ---
    for hd in [5, 10]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'bb_touch',
                'hold_days': hd, 'threshold': 0,
                'top_n': tn, 'comm': COMM,
                'label': f"BB_Touch_H{hd}_TN{tn}",
            })

    # --- C2: Bollinger Squeeze Breakout: hold 5/10, top_n 1/3/5 ---
    for hd in [5, 10]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'bb_squeeze',
                'hold_days': hd, 'threshold': 0,
                'top_n': tn, 'comm': COMM,
                'label': f"BB_Squeeze_H{hd}_TN{tn}",
            })

    # --- D: MACD Crossover: hold 10/20, top_n 1/3/5 ---
    for hd in [10, 20]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'macd_cross',
                'hold_days': hd, 'threshold': 0,
                'top_n': tn, 'comm': COMM,
                'label': f"MACD_H{hd}_TN{tn}",
            })

    # --- E: 50-day Breakout: hold 10/20, top_n 1/3/5 ---
    for hd in [10, 20]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'breakout50',
                'hold_days': hd, 'threshold': 0,
                'top_n': tn, 'comm': COMM,
                'label': f"BRK50_H{hd}_TN{tn}",
            })

    # --- F: Volume Breakout: hold 5/10, top_n 1/3/5 ---
    for hd in [5, 10]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'vol_breakout',
                'hold_days': hd, 'threshold': 0,
                'top_n': tn, 'comm': COMM,
                'label': f"VolBrk_H{hd}_TN{tn}",
            })

    # --- G: Multi-Timeframe RSI: weekly > 50/60, daily < 30, hold 5/10 ---
    for wrsi_thresh in [50, 60]:
        for hd in [5, 10]:
            for tn in [1, 3, 5]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': 'mtf_rsi',
                    'hold_days': hd, 'threshold': wrsi_thresh,
                    'top_n': tn, 'comm': COMM,
                    'label': f"MTF_RSI_W{wrsi_thresh}_H{hd}_TN{tn}",
                })

    # --- H: Hammer: hold 3/5, top_n 1/3/5 ---
    for hd in [3, 5]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'hammer',
                'hold_days': hd, 'threshold': 0,
                'top_n': tn, 'comm': COMM,
                'label': f"Hammer_H{hd}_TN{tn}",
            })

    # --- H2: Bullish Engulfing: hold 3/5, top_n 1/3/5 ---
    for hd in [3, 5]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'engulfing',
                'hold_days': hd, 'threshold': 0,
                'top_n': tn, 'comm': COMM,
                'label': f"Engulf_H{hd}_TN{tn}",
            })

    # --- I: ATR Trailing Stop: top_n 1/3/5 ---
    for tn in [1, 3, 5]:
        cid += 1
        configs.append({
            'id': cid, 'signal': 'atr_trail',
            'hold_days': 50, 'threshold': 0,
            'top_n': tn, 'comm': COMM,
            'label': f"ATR_Trail_TN{tn}",
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
    sig_order = ['rsi_oversold', 'rsi_div', 'bb_touch', 'bb_squeeze',
                 'macd_cross', 'breakout50', 'vol_breakout',
                 'mtf_rsi', 'hammer', 'engulfing', 'atr_trail']
    sig_names = {
        'rsi_oversold': 'A) RSI Oversold Bounce',
        'rsi_div':      'B) RSI Divergence',
        'bb_touch':     'C1) Bollinger Lower Touch',
        'bb_squeeze':   'C2) Bollinger Squeeze Breakout',
        'macd_cross':   'D) MACD Crossover',
        'breakout50':   'E) 50-Day Breakout',
        'vol_breakout': 'F) Volume Breakout',
        'mtf_rsi':      'G) Multi-Timeframe RSI',
        'hammer':       'H1) Hammer Candle',
        'engulfing':    'H2) Bullish Engulfing',
        'atr_trail':    'I) ATR Trailing Stop',
    }

    print(f"\n{'=' * 150}")
    print("  BEST PER SIGNAL TYPE (Full Period) -- ALL NEXT-OPEN EXECUTION")
    print(f"{'=' * 150}")
    print(f"  {'Signal':<37} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | Best Config")
    print("-" * 150)

    best_per_sig = {}
    for r in results:
        key = r['config']['signal']
        if key not in best_per_sig:
            best_per_sig[key] = r

    for sig in sig_order:
        if sig in best_per_sig:
            b = best_per_sig[sig]
            print(f"  {sig_names.get(sig, sig):<37} | {b['ann']:>+8.1f}% | {b['wr']:>5.1f}% | {b['n']:>5} | {b['avg_pnl']:>+6.3f}% | {b['mdd']:>6.1f}% | {b['label']}")

    # ================================================================
    # SIGNAL TYPE SUMMARY
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  SIGNAL TYPE SUMMARY (Average of Top 5 configs per type)")
    print(f"{'=' * 150}")
    print(f"  {'Signal':<37} | {'Avg Ann':>9} | {'Avg WR':>7} | {'Avg N':>7} | {'Avg PnL':>8} | {'Avg MDD':>8} | {'#Positive':>9}")
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
        print(f"  {sig_names.get(sig, sig):<37} | {avg_ann:>+8.1f}% | {avg_wr:>6.1f}% | {avg_n:>7.0f} | {avg_pnl:>+7.3f}% | {avg_mdd:>7.1f}% | {n_pos:>5}/{len(sub)}")

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

    print(f"\n{'=' * 180}")
    print(f"  WALK-FORWARD ({len(wf_configs)} configs)")
    print(f"{'=' * 180}")

    header = f"  {'#':>3} | {'Config':<35} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>7} |"
    header += f" {'Pos':>4} | {'MDD':>7}"
    print(header)
    print("-" * 180)

    wf_rows = []
    for i, r in enumerate(wf_configs):
        cfg = r['config']
        wf_row = {'label': cfg['label'], 'signal': cfg['signal'],
                  'entry': 'next_open', 'windows': {}, 'mdd': {}}
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
    header2 = f"  {'Signal':<37} | {'WF Avg':>8} |"
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
            row_str = f"  {sig_names.get(sig, sig):<37} | {avg:>+7.1f}% |"
            for v in vals:
                row_str += f" {v:>+7.1f}% |"
            row_str += f" {pos}/6 | {avg_mdd:>6.1f}%"
            print(row_str)

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  FINAL VERDICT: CLASSIC TA WITH NEXT-OPEN EXECUTION")
    print(f"{'=' * 150}")
    print()
    print("  KEY QUESTION: Which classic TA patterns provide genuine forward-looking")
    print("  alpha with practical (next-open) execution?")
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
        genuine = "GENUINE ALPHA" if wf_pos >= 4 and best['ann'] > 0 else ("MARGINAL" if wf_pos >= 3 and best['ann'] > 0 else "NO ALPHA")

        print(f"  {sig_names.get(sig, sig)}")
        print(f"    Best annual: {best['ann']:>+8.1f}%  |  Avg top-5: {avg_top5:>+8.1f}%  |  {n_pos}/{len(sub)} positive configs")
        print(f"    Walk-forward: {wf_pos}/6 positive  |  WF avg: {wf_avg:>+8.1f}%")
        print(f"    VERDICT: {verdict}  -->  {genuine}")
        print()

    print(f"  Total runtime: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
