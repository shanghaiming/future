"""
Alpha Futures V102 -- Combined Signal Strategy (Next-Open Execution)
=====================================================================
V101 best signals were: 50-day Breakout (+49.6% ann), Vol Breakout (+37.2%),
Bullish Engulfing (+31.6%), OI Capitulation (+21.3%).

V102 tests COMBINATIONS: when multiple signals agree, conviction is stronger.

Signals tested (ALL next-open execution):
  A) Breakout + Volatility Confirmation
  B) Breakout + OI Confirmation
  C) Bullish Engulfing + Vol Breakout
  D) Multi-Signal Scoring (score >= 2)
  E) Breakout with Trailing Stop
  F) Portfolio of Strategies (equal-weight)
  G) Breakout + Term Structure Confirmation
  H) Breakout Momentum (strongest breakouts only)

Walk-forward: top 15 configs across 2020-2025.
"""
import sys, os, time, warnings, json
import numpy as np
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

TS_DIR = '/Users/chengming/home/futures_platform/data/futures_term_structure/'


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def main():
    t_start = time.time()
    print("=" * 150)
    print("Alpha Futures V102 -- Combined Signal Strategy (Next-Open Execution)")
    print("=" * 150)
    print("\n  Testing signal combinations: Breakout+Vol, Breakout+OI, Engulfing+Vol,")
    print("  Multi-Signal Scoring, Trailing Stop, Portfolio, Term Structure, Breakout Momentum")
    print("  ALL signals computed at close di, entry at O[si, di+1] (NEXT DAY OPEN)")

    # -- Load data -------------------------------------------------
    print("\n[Data] Loading...", flush=True)
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ================================================================
    # PRECOMPUTE BASE INDICATORS
    # ================================================================
    print("\n[Indicators] Computing base signals...", flush=True)
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
    print(f"  ATR20 computed ({time.time()-t0:.1f}s)")

    # ---- 50-day high breakout ----
    high50 = np.full((NS, ND), np.nan)
    breakout_high = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(50, ND):
            h50 = np.nanmax(C[si, di-50:di])  # use close for Donchian
            if np.isnan(h50):
                continue
            high50[si, di] = h50
            c = C[si, di]
            if not np.isnan(c) and c > h50:
                breakout_high[si, di] = True
    print(f"  50-day Breakout computed ({time.time()-t0:.1f}s)")

    # ---- Bullish Engulfing ----
    engulfing_signal = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(1, ND):
            o = O[si, di]
            c = C[si, di]
            o_prev = O[si, di - 1]
            c_prev = C[si, di - 1]
            if np.isnan(o) or np.isnan(c) or np.isnan(o_prev) or np.isnan(c_prev):
                continue
            # Bullish engulfing: today's body engulfs yesterday's body
            # yesterday bearish (c_prev < o_prev), today bullish (c > o)
            # and c > o_prev and o < c_prev
            if c_prev < o_prev and c > o and c > o_prev and o < c_prev:
                engulfing_signal[si, di] = True
    print(f"  Bullish Engulfing computed ({time.time()-t0:.1f}s)")

    # ---- Volatility Breakout: range > 2x ATR20, bullish close ----
    vol_breakout = np.zeros((NS, ND), dtype=bool)
    vol_breakout_ratio = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            h = H[si, di]
            l = L[si, di]
            c = C[si, di]
            o = O[si, di]
            atr = ATR20[si, di]
            if np.isnan(h) or np.isnan(l) or np.isnan(c) or np.isnan(o) or np.isnan(atr):
                continue
            if atr <= 0:
                continue
            rng = h - l
            ratio = rng / atr
            vol_breakout_ratio[si, di] = ratio
            if ratio > 2.0 and c > o:  # high vol + bullish close
                vol_breakout[si, di] = True
    print(f"  Volatility Breakout computed ({time.time()-t0:.1f}s)")

    # ---- OI Capitulation: OI declining + price declining over 5 days ----
    oi_capitulation = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(5, ND):
            oi_now = OI[si, di]
            oi_5ago = OI[si, di - 5]
            c_now = C[si, di]
            c_5ago = C[si, di - 5]
            if np.isnan(oi_now) or np.isnan(oi_5ago) or np.isnan(c_now) or np.isnan(c_5ago):
                continue
            if oi_5ago <= 0 or c_5ago <= 0:
                continue
            if oi_now < oi_5ago and c_now < c_5ago:
                oi_capitulation[si, di] = True
    print(f"  OI Capitulation computed ({time.time()-t0:.1f}s)")

    # ---- 20-day momentum ----
    mom20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            c_now = C[si, di]
            c_prev = C[si, di - 20]
            if np.isnan(c_now) or np.isnan(c_prev) or c_prev <= 0:
                continue
            mom20[si, di] = (c_now - c_prev) / c_prev
    print(f"  20-day Momentum computed ({time.time()-t0:.1f}s)")

    # ---- OI increase over 5 days ----
    oi_increasing = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(5, ND):
            oi_now = OI[si, di]
            oi_5ago = OI[si, di - 5]
            if np.isnan(oi_now) or np.isnan(oi_5ago):
                continue
            if oi_5ago > 0 and oi_now > oi_5ago:
                oi_increasing[si, di] = True
    print(f"  OI Increase computed ({time.time()-t0:.1f}s)")

    # ---- Load Term Structure (for strategy G) ----
    print("\n[TS] Loading term structure data...", flush=True)
    structure_ts = np.full((NS, ND), np.nan)
    sym_to_si = {syms[si]: si for si in range(NS)}
    date_str_map = {dates[di].strftime('%Y%m%d'): di for di in range(ND)}
    ts_loaded = 0
    if os.path.isdir(TS_DIR):
        for fname in sorted(os.listdir(TS_DIR)):
            if not fname.endswith('.json'):
                continue
            parts = fname.rsplit('_', 1)
            if len(parts) != 2:
                continue
            sym, date_part = parts
            date_part = date_part.replace('.json', '')
            si = sym_to_si.get(sym)
            if si is None:
                continue
            di = date_str_map.get(date_part)
            if di is None:
                continue
            try:
                with open(os.path.join(TS_DIR, fname), 'r') as f:
                    data = json.load(f)
                struct = data.get('structure', '')
                if struct == 'backwardation':
                    structure_ts[si, di] = 1.0
                elif struct == 'contango':
                    structure_ts[si, di] = -1.0
                ts_loaded += 1
            except:
                pass
    print(f"  Loaded {ts_loaded} TS files ({time.time()-t0:.1f}s)")

    print(f"\n  All indicators computed ({time.time()-t_start:.1f}s total)")

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(config, wf_test_year=None):
        """
        Config:
            signal: signal type string
            hold_days: int (or 'trail' for trailing stop)
            top_n: int (max concurrent positions)
            comm: float
            score_thresh: minimum signal score to trade (for D)
            trail_atr_mult: ATR multiplier for trailing stop (for E)
            max_hold: maximum hold days (for E)
        """
        sig_type = config['signal']
        hold_days = config['hold_days']
        top_n = config['top_n']
        comm = config.get('comm', COMM)
        score_thresh = config.get('score_thresh', 0)
        trail_atr_mult = config.get('trail_atr_mult', 2.0)
        max_hold = config.get('max_hold', 20)

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

        if end_di < start_di + 2:
            return None

        cash = float(CASH0)
        positions = []
        trades = []

        for di in range(start_di, end_di - 1):
            # Reset cash at test window start (WF mode)
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                positions = []

            # -- Close positions -----------------------------------------
            closed = []
            for pos in positions:
                si_pos = pos['si']
                should_close = False

                if sig_type in ('breakout_trail',):
                    # Trailing stop exit
                    c_now = C[si_pos, di]
                    if not np.isnan(c_now):
                        atr = ATR20[si_pos, di] if not np.isnan(ATR20[si_pos, di]) else 0
                        h_since = np.nanmax(H[si_pos, pos['entry_di']:di + 1])
                        if atr > 0 and not np.isnan(h_since):
                            new_stop = h_since - trail_atr_mult * atr
                            if new_stop > pos.get('trail_stop', 0):
                                pos['trail_stop'] = new_stop
                        if c_now <= pos.get('trail_stop', 0):
                            should_close = True
                    # Max hold
                    if di - pos['entry_di'] >= max_hold:
                        should_close = True
                else:
                    if di - pos['entry_di'] >= pos['hold_days']:
                        should_close = True

                if should_close:
                    exit_price = C[si_pos, di]
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

            if sig_type == 'breakout_vol':
                # A) 50-day breakout + range > 1.5*ATR20 + bullish close
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    if not breakout_high[si, di]:
                        continue
                    c = C[si, di]
                    o = O[si, di]
                    atr = ATR20[si, di]
                    h = H[si, di]
                    l = L[si, di]
                    if np.isnan(c) or np.isnan(o) or np.isnan(atr) or np.isnan(h) or np.isnan(l):
                        continue
                    if atr <= 0:
                        continue
                    if c <= o:  # need bullish close
                        continue
                    rng = h - l
                    if rng < 1.5 * atr:
                        continue
                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    score = rng / atr  # higher vol = stronger
                    candidates.append((score, 1, {
                        'si': si, 'sym': syms[si], 'entry_price': ep,
                    }))

            elif sig_type == 'breakout_oi':
                # B) 50-day breakout + OI increasing over 5 days
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    if not breakout_high[si, di]:
                        continue
                    if not oi_increasing[si, di]:
                        continue
                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    # Score: OI growth rate
                    oi_now = OI[si, di]
                    oi_5ago = OI[si, di - 5]
                    if np.isnan(oi_now) or np.isnan(oi_5ago) or oi_5ago <= 0:
                        score = 1.0
                    else:
                        score = (oi_now - oi_5ago) / oi_5ago
                    candidates.append((score, 1, {
                        'si': si, 'sym': syms[si], 'entry_price': ep,
                    }))

            elif sig_type == 'engulfing_vol':
                # C) Bullish engulfing + range > 1.5*ATR20
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    if not engulfing_signal[si, di]:
                        continue
                    h = H[si, di]
                    l = L[si, di]
                    atr = ATR20[si, di]
                    if np.isnan(h) or np.isnan(l) or np.isnan(atr):
                        continue
                    if atr <= 0:
                        continue
                    rng = h - l
                    if rng < 1.5 * atr:
                        continue
                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    score = rng / atr
                    candidates.append((score, 1, {
                        'si': si, 'sym': syms[si], 'entry_price': ep,
                    }))

            elif sig_type == 'multi_score':
                # D) Multi-signal scoring
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    c = C[si, di]
                    o = O[si, di]
                    if np.isnan(c) or np.isnan(o):
                        continue
                    score = 0

                    # +1: 50-day breakout
                    if breakout_high[si, di]:
                        score += 1

                    # +1: volatility breakout (range > 2x ATR, bullish close)
                    if vol_breakout[si, di]:
                        score += 1

                    # +1: bullish engulfing
                    if engulfing_signal[si, di]:
                        score += 1

                    # +1: OI declining + price declining (capitulation reversal setup)
                    if oi_capitulation[si, di]:
                        score += 1

                    # +1: 20-day momentum > 0
                    if not np.isnan(mom20[si, di]) and mom20[si, di] > 0:
                        score += 1

                    if score < score_thresh:
                        continue

                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    candidates.append((score, 1, {
                        'si': si, 'sym': syms[si], 'entry_price': ep,
                    }))

            elif sig_type == 'breakout_trail':
                # E) 50-day breakout with trailing stop
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    if not breakout_high[si, di]:
                        continue
                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    atr = ATR20[si, di] if not np.isnan(ATR20[si, di]) else 0
                    h_today = H[si, di] if not np.isnan(H[si, di]) else ep
                    # Score: breakout strength
                    h50 = high50[si, di] if not np.isnan(high50[si, di]) else 0
                    c = C[si, di] if not np.isnan(C[si, di]) else 0
                    score = (c - h50) / h50 * 100 if h50 > 0 else 0
                    candidates.append((score, 1, {
                        'si': si, 'sym': syms[si], 'entry_price': ep,
                        'trail_stop': h_today - trail_atr_mult * atr if atr > 0 else ep * 0.95,
                    }))

            elif sig_type == 'breakout_ts':
                # G) 50-day breakout + term structure backwardation
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    if not breakout_high[si, di]:
                        continue
                    ts = structure_ts[si, di]
                    if np.isnan(ts) or ts <= 0:  # not backwardation
                        continue
                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    h50 = high50[si, di] if not np.isnan(high50[si, di]) else 0
                    c = C[si, di] if not np.isnan(C[si, di]) else 0
                    score = (c - h50) / h50 * 100 if h50 > 0 else 0
                    candidates.append((score, 1, {
                        'si': si, 'sym': syms[si], 'entry_price': ep,
                    }))

            elif sig_type == 'breakout_momentum':
                # H) 50-day breakout + breakout strength > threshold%
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    if not breakout_high[si, di]:
                        continue
                    h50 = high50[si, di]
                    c = C[si, di]
                    if np.isnan(h50) or np.isnan(c) or h50 <= 0:
                        continue
                    strength = (c - h50) / h50  # fractional breakout
                    if strength < score_thresh / 100.0:
                        continue
                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    candidates.append((strength * 100, 1, {
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
                if sig_type == 'breakout_trail':
                    pos_dict['trail_stop'] = info.get('trail_stop', price * 0.95)
                    pos_dict['hold_days'] = max_hold

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

    # --- A: Breakout + Vol Confirmation: hold 5/10/20, top_n 1/3/5 ---
    for hd in [5, 10, 20]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'breakout_vol',
                'hold_days': hd, 'top_n': tn, 'comm': COMM,
                'label': f"A_BrkVol_H{hd}_TN{tn}",
            })

    # --- B: Breakout + OI Confirmation: hold 10/20, top_n 1/3/5 ---
    for hd in [10, 20]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'breakout_oi',
                'hold_days': hd, 'top_n': tn, 'comm': COMM,
                'label': f"B_BrkOI_H{hd}_TN{tn}",
            })

    # --- C: Engulfing + Vol Breakout: hold 5/10, top_n 1/3/5 ---
    for hd in [5, 10]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'engulfing_vol',
                'hold_days': hd, 'top_n': tn, 'comm': COMM,
                'label': f"C_EngVol_H{hd}_TN{tn}",
            })

    # --- D: Multi-Signal Scoring: thresh 2/3/4, hold 5/10, top_n 1/3/5 ---
    for thresh in [2, 3, 4]:
        for hd in [5, 10]:
            for tn in [1, 3, 5]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': 'multi_score',
                    'hold_days': hd, 'top_n': tn, 'comm': COMM,
                    'score_thresh': thresh,
                    'label': f"D_Multi_T{thresh}_H{hd}_TN{tn}",
                })

    # --- E: Breakout with Trailing Stop: ATR_mult 1.5/2/3, max_hold 20/30, top_n 1/3/5 ---
    for atr_m in [1.5, 2.0, 3.0]:
        for mh in [20, 30]:
            for tn in [1, 3, 5]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': 'breakout_trail',
                    'hold_days': mh, 'top_n': tn, 'comm': COMM,
                    'trail_atr_mult': atr_m, 'max_hold': mh,
                    'label': f"E_Trail_ATR{atr_m}_MH{mh}_TN{tn}",
                })

    # --- G: Breakout + Term Structure: hold 10/20, top_n 1/3/5 ---
    for hd in [10, 20]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'breakout_ts',
                'hold_days': hd, 'top_n': tn, 'comm': COMM,
                'label': f"G_BrkTS_H{hd}_TN{tn}",
            })

    # --- H: Breakout Momentum: thresh 1/2/3%, hold 10/20, top_n 1/3/5 ---
    for thresh in [1, 2, 3]:
        for hd in [10, 20]:
            for tn in [1, 3, 5]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': 'breakout_momentum',
                    'hold_days': hd, 'top_n': tn, 'comm': COMM,
                    'score_thresh': thresh,
                    'label': f"H_BrkMom_T{thresh}_H{hd}_TN{tn}",
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
    sig_order = ['breakout_vol', 'breakout_oi', 'engulfing_vol', 'multi_score',
                 'breakout_trail', 'breakout_ts', 'breakout_momentum']
    sig_names = {
        'breakout_vol':     'A) Breakout + Vol Confirmation',
        'breakout_oi':      'B) Breakout + OI Confirmation',
        'engulfing_vol':    'C) Engulfing + Vol Breakout',
        'multi_score':      'D) Multi-Signal Scoring',
        'breakout_trail':   'E) Breakout + Trailing Stop',
        'breakout_ts':      'G) Breakout + Term Structure',
        'breakout_momentum':'H) Breakout Momentum (Strong)',
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
    # F) PORTFOLIO OF STRATEGIES
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  F) PORTFOLIO OF STRATEGIES (Non-overlapping signal combination)")
    print(f"{'=' * 150}")

    # For each base signal, run individually with 25% capital, then combine returns
    # Use the best config per signal from full-period results
    portfolio_signals = {
        'breakout50': {
            'signal_fn': lambda si, di: breakout_high[si, di],
            'hold': 10,
        },
        'vol_breakout': {
            'signal_fn': lambda si, di: vol_breakout[si, di],
            'hold': 10,
        },
        'engulfing': {
            'signal_fn': lambda si, di: engulfing_signal[si, di],
            'hold': 5,
        },
        'oi_capitulation': {
            'signal_fn': lambda si, di: oi_capitulation[si, di],
            'hold': 10,
        },
    }

    # Run each strategy independently with 25% capital
    n_strats = len(portfolio_signals)
    alloc_per_strat = CASH0 / n_strats

    for tn_pf in [1, 3, 5]:
        portfolio_cash = 0.0
        portfolio_details = []

        for sname, sdef in portfolio_signals.items():
            # Run simple backtest for this sub-strategy
            sub_cash = float(alloc_per_strat)
            sub_positions = []
            sub_trades = []
            sig_fn = sdef['signal_fn']
            sub_hold = sdef['hold']

            for di in range(MIN_TRAIN, ND - 1):
                entry_di = di + 1

                # Close positions
                closed_sub = []
                for pos in sub_positions:
                    if di - pos['entry_di'] >= pos['hold_days']:
                        exit_price = C[pos['si'], di]
                        if np.isnan(exit_price) or exit_price <= 0:
                            exit_price = pos['entry_price']
                        mult = MULT.get(pos['sym'], DEF_MULT)
                        mkt_val = exit_price * mult * abs(pos['lots'])
                        sub_cash += mkt_val - mkt_val * COMM
                        pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
                        invested = pos['entry_price'] * mult * abs(pos['lots'])
                        pnl_pct = pnl / invested * 100 if invested > 0 else 0
                        sub_trades.append({'pnl_pct': pnl_pct, 'year': dates[di].year if di < ND else dates[-1].year})
                        closed_sub.append(pos)
                for pos in closed_sub:
                    sub_positions.remove(pos)

                # Generate signals
                cands = []
                for si in range(NS):
                    if any(p['si'] == si for p in sub_positions):
                        continue
                    if sig_fn(si, di):
                        ep = O[si, entry_di]
                        if np.isnan(ep) or ep <= 0:
                            continue
                        cands.append({'si': si, 'sym': syms[si], 'entry_price': ep})
                if not cands:
                    continue

                n_slots = tn_pf - len(sub_positions)
                for info in cands[:max(0, n_slots)]:
                    si = info['si']
                    sym = info['sym']
                    price = info['entry_price']
                    mult = MULT.get(sym, DEF_MULT)
                    notional = price * mult
                    lots = int(sub_cash / (notional * (1 + COMM) * max(tn_pf, 1)))
                    if lots <= 0:
                        lots = int(sub_cash * 0.9 / (notional * (1 + COMM)))
                    if lots <= 0:
                        continue
                    cost_in = notional * lots * (1 + COMM)
                    if cost_in > sub_cash:
                        lots = int(sub_cash * 0.85 / (notional * (1 + COMM)))
                        cost_in = notional * lots * (1 + COMM) if lots > 0 else 0
                    if lots <= 0 or cost_in <= 0 or cost_in > sub_cash:
                        continue
                    sub_cash -= cost_in
                    sub_positions.append({
                        'si': si, 'entry_price': price, 'entry_di': entry_di,
                        'lots': lots, 'sym': sym, 'hold_days': sub_hold,
                    })

            # Close remaining
            for pos in sub_positions:
                ae = ND - 1
                exit_price = C[pos['si'], ae]
                if np.isnan(exit_price) or exit_price <= 0:
                    exit_price = pos['entry_price']
                mult = MULT.get(pos['sym'], DEF_MULT)
                sub_cash += exit_price * mult * abs(pos['lots']) * (1 - COMM)

            portfolio_cash += sub_cash
            n_sub_trades = len(sub_trades)
            sub_wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in sub_trades]) * 100 if sub_trades else 0
            sub_ann = annual_return(sub_cash, alloc_per_strat, ND - MIN_TRAIN)
            portfolio_details.append({
                'name': sname, 'final': sub_cash, 'ann': sub_ann,
                'n': n_sub_trades, 'wr': sub_wr,
            })

        pf_total_ret = (portfolio_cash - CASH0) / CASH0 * 100
        pf_ann = annual_return(portfolio_cash, CASH0, ND - MIN_TRAIN)

        print(f"\n  Portfolio TN={tn_pf}:  Total={pf_total_ret:+.1f}%  Ann={pf_ann:+.1f}%  Final={portfolio_cash:,.0f}")
        print(f"    {'Strategy':<20} | {'Alloc':>10} | {'Final':>14} | {'Ann':>9} | {'WR':>6} | {'N':>5}")
        for pd in portfolio_details:
            print(f"    {pd['name']:<20} | {alloc_per_strat:>10,.0f} | {pd['final']:>14,.0f} | {pd['ann']:>+8.1f}% | {pd['wr']:>5.1f}% | {pd['n']:>5}")

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
    print("  FINAL VERDICT: COMBINED SIGNAL STRATEGIES WITH NEXT-OPEN EXECUTION")
    print(f"{'=' * 150}")
    print()
    print("  KEY QUESTION: Can combining signals push returns above +50% with practical execution?")
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

        genuine = "GENUINE ALPHA" if wf_pos >= 4 and best['ann'] > 0 else ("MARGINAL" if wf_pos >= 3 and best['ann'] > 0 else "WEAK")

        print(f"  {sig_names.get(sig, sig)}")
        print(f"    Best annual: {best['ann']:>+8.1f}%  |  Avg top-5: {avg_top5:>+8.1f}%  |  {n_pos}/{len(sub)} positive configs")
        print(f"    Walk-forward: {wf_pos}/6 positive  |  WF avg: {wf_avg:>+8.1f}%")
        print(f"    VERDICT: {genuine}")
        print()

    # Print the absolute best
    if results:
        best_overall = results[0]
        wf_best = [w for w in wf_rows if w['label'] == best_overall['label']]
        if wf_best:
            wf = wf_best[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            wf_pos = sum(1 for v in vals if v > 0)
            wf_avg = np.mean(vals)
            print(f"  *** BEST OVERALL: {best_overall['label']} ***")
            print(f"      Full-period: {best_overall['ann']:+.1f}% annual  |  WF avg: {wf_avg:+.1f}%  |  WF positive: {wf_pos}/6")
            print(f"      WR: {best_overall['wr']:.1f}%  |  MDD: {best_overall['mdd']:.1f}%  |  Trades: {best_overall['n']}")

    print(f"\n  Total runtime: {time.time()-t_start:.0f}s")
    print("=" * 150)


if __name__ == '__main__':
    main()
