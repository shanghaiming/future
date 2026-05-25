"""
Alpha Futures V99 -- OI Dynamics, Seasonality & Lead-Lag
=========================================================
V96 proved same-close-entry strategies are contaminated.
V97 best next-open: +6.9% (trend quality).

V99 tests PERSISTENT STRUCTURAL STATES that survive the 1-day delay:

  A) OI Surge Reversal: OI surged + price dropped -> short covering rally
  B) OI Decline + Capitulation: OI declining + price dropping -> exhaustion
  C) Seasonal Pattern: Monthly average returns (rolling lookback)
  D) OI Momentum: Rank by OI change, buy top-K
  E) Volatility Breakout: Range > 2*ATR_20 bullish breakout
  F) Mean-Reversion on Weekly Bars: Weekly z-score extreme
  G) Cross-Commodity Lead-Lag: Leader returns predict follower

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

# Lead-lag pairs: (leader, follower)
LEAD_LAG_PAIRS = [
    ('ifi', 'rbfi'),   # iron ore -> rebar
    ('scfi', 'tafi'),  # crude -> PTA
    ('afi', 'mfi'),    # soybean -> meal
    ('ifi', 'jfi'),    # iron ore -> coke
    ('scfi', 'mafi'),  # crude -> methanol
]


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def main():
    print("=" * 150)
    print("Alpha Futures V99 -- OI Dynamics, Seasonality & Lead-Lag (Next-Open Execution)")
    print("=" * 150)
    print("\n  Testing persistent structural states: OI dynamics, seasonality, volatility, lead-lag")
    print("  ALL signals computed at close di, entry at O[si, di+1] (NEXT DAY OPEN)")

    # ── Load data ────────────────────────────────────────────────────
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    sym_to_si = {syms[si]: si for si in range(NS)}

    # ================================================================
    # PRECOMPUTE BASELINE SIGNALS
    # ================================================================
    print("\n[Signals] Computing returns and indicators...", flush=True)
    t0 = time.time()

    # 1-day close-to-close return
    ret1 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            cn = C[si, di]
            cp = C[si, di - 1]
            if not np.isnan(cn) and not np.isnan(cp) and cp > 0:
                ret1[si, di] = (cn - cp) / cp

    # 5-day cumulative return
    ret5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            c0 = C[si, di - 5]
            c5 = C[si, di]
            if not np.isnan(c0) and not np.isnan(c5) and c0 > 0:
                ret5[si, di] = (c5 - c0) / c0

    # 3-day cumulative return
    ret3 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(3, ND):
            c0 = C[si, di - 3]
            c3 = C[si, di]
            if not np.isnan(c0) and not np.isnan(c3) and c0 > 0:
                ret3[si, di] = (c3 - c0) / c0

    # 20-day cumulative return
    ret20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            c0 = C[si, di - 20]
            c20 = C[si, di]
            if not np.isnan(c0) and not np.isnan(c20) and c0 > 0:
                ret20[si, di] = (c20 - c0) / c0

    # OI change over 5 days (percentage)
    oi_chg5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            oi_now = OI[si, di]
            oi_prev = OI[si, di - 5]
            if (not np.isnan(oi_now) and not np.isnan(oi_prev)
                    and oi_prev > 0 and oi_now > 0):
                oi_chg5[si, di] = (oi_now - oi_prev) / oi_prev

    # OI change over 20 days (percentage)
    oi_chg20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            oi_now = OI[si, di]
            oi_prev = OI[si, di - 20]
            if (not np.isnan(oi_now) and not np.isnan(oi_prev)
                    and oi_prev > 0 and oi_now > 0):
                oi_chg20[si, di] = (oi_now - oi_prev) / oi_prev

    print(f"  Returns and OI changes computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # A) OI SURGE REVERSAL
    # ================================================================
    print("\n[Signals] A) OI Surge Reversal: precomputing...", flush=True)
    t0 = time.time()

    # Signal: oi_chg5 > threshold AND ret5 < -price_thresh
    # OI surge + price drop -> shorts accumulating, potential short covering rally
    oi_surge_signal = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(5, ND):
            oc = oi_chg5[si, di]
            r5 = ret5[si, di]
            if not np.isnan(oc) and not np.isnan(r5):
                if oc > 0.20 and r5 < -0.03:
                    oi_surge_signal[si, di] = True

    print(f"  OI surge signal computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # B) OI DECLINE + CAPITULATION
    # ================================================================
    print("\n[Signals] B) OI Decline + Capitulation: precomputing...", flush=True)
    t0 = time.time()

    # Signal: oi_chg5 < -threshold AND ret5 < -price_thresh
    # Shorts covering + price still dropping -> selling exhaustion
    oi_capit_signal = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(5, ND):
            oc = oi_chg5[si, di]
            r5 = ret5[si, di]
            if not np.isnan(oc) and not np.isnan(r5):
                if oc < -0.10 and r5 < -0.03:
                    oi_capit_signal[si, di] = True

    print(f"  OI capitulation signal computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # C) SEASONAL PATTERN
    # ================================================================
    print("\n[Signals] C) Seasonal Pattern: precomputing...", flush=True)
    t0 = time.time()

    # For each commodity and calendar month, compute average daily return
    # using rolling lookback (exclude current year)
    # Store: seasonal_score[si, di] = average return of this month in past years
    seasonal_score = np.full((NS, ND), np.nan)
    for si in range(NS):
        # Gather monthly returns by (year, month)
        monthly_rets = {}  # month -> list of avg daily returns in that month
        for di in range(1, ND):
            r = ret1[si, di]
            if np.isnan(r):
                continue
            m = dates[di].month
            y = dates[di].year
            monthly_rets.setdefault((m, y), []).append(r)

        # For each day, compute average return of this month across prior years
        for di in range(1, ND):
            cur_month = dates[di].month
            cur_year = dates[di].year
            past_rets = []
            for (m, y), rets in monthly_rets.items():
                if m == cur_month and y < cur_year:
                    if len(rets) >= 3:  # at least 3 days of data
                        past_rets.append(np.mean(rets))
            if len(past_rets) >= 2:  # at least 2 prior years
                seasonal_score[si, di] = np.mean(past_rets)

    print(f"  Seasonal scores computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # D) OI MOMENTUM (FOLLOW THE MONEY)
    # ================================================================
    print("\n[Signals] D) OI Momentum: precomputing...", flush=True)
    t0 = time.time()

    # Rank by OI change over 20 days
    oi_mom_rank = np.full((NS, ND), np.nan)
    for di in range(20, ND):
        vals = []
        for si in range(NS):
            oc = oi_chg20[si, di]
            if not np.isnan(oc):
                vals.append((si, oc))
        if len(vals) < 5:
            continue
        vals.sort(key=lambda x: x[1])
        n_vals = len(vals)
        for rank, (si, oc) in enumerate(vals):
            oi_mom_rank[si, di] = rank / n_vals  # 0 = lowest OI growth, 1 = highest

    print(f"  OI momentum ranks computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # E) VOLATILITY BREAKOUT
    # ================================================================
    print("\n[Signals] E) Volatility Breakout: precomputing...", flush=True)
    t0 = time.time()

    # 20-day ATR
    atr20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            trs = []
            for k in range(di - 20, di):
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
                atr20[si, di] = np.mean(trs)

    # Breakout signal: today's range > 2 * ATR20 AND close > open (bullish)
    breakout_signal = np.zeros((NS, ND), dtype=bool)
    breakout_score = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            a = atr20[si, di]
            h = H[si, di]
            l = L[si, di]
            c = C[si, di]
            o = O[si, di]
            if np.isnan(a) or np.isnan(h) or np.isnan(l) or np.isnan(c) or np.isnan(o):
                continue
            if a <= 0:
                continue
            today_range = h - l
            if today_range > 2.0 * a and c > o:
                breakout_signal[si, di] = True
                breakout_score[si, di] = today_range / a

    print(f"  Volatility breakout computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # F) MEAN-REVERSION ON WEEKLY BARS
    # ================================================================
    print("\n[Signals] F) Weekly Bar Mean-Reversion: precomputing...", flush=True)
    t0 = time.time()

    # Build weekly bar index: find Friday (or last trading day of each week)
    # Group days by (year, week)
    from datetime import datetime
    week_map = {}  # (year, week) -> [list of di]
    for di in range(ND):
        d = dates[di]
        iso = d.isocalendar()
        wk_key = (iso[0], iso[1])  # (iso_year, iso_week)
        week_map.setdefault(wk_key, []).append(di)

    # For each commodity, build weekly closes and weekly returns
    weekly_close = {}  # si -> list of (last_di_of_week, close_price)
    weekly_open = {}   # si -> list of (first_di_of_week, open_price)
    for si in range(NS):
        wc = []
        wo = []
        for wk_key in sorted(week_map.keys()):
            dis = week_map[wk_key]
            # Last day close
            for d in reversed(dis):
                c = C[si, d]
                if not np.isnan(c) and c > 0:
                    wc.append((d, c))
                    break
            # First day open
            for d in dis:
                o = O[si, d]
                if not np.isnan(o) and o > 0:
                    wo.append((d, o))
                    break
        weekly_close[si] = wc
        weekly_open[si] = wo

    # Compute 4-week z-score of weekly returns, mapped back to daily index
    weekly_zscore = np.full((NS, ND), np.nan)
    for si in range(NS):
        wc = weekly_close[si]
        if len(wc) < 8:
            continue
        # Weekly returns
        w_rets = []
        for i in range(1, len(wc)):
            d_cur, c_cur = wc[i]
            d_prev, c_prev = wc[i - 1]
            if c_prev > 0:
                w_rets.append((d_cur, (c_cur - c_prev) / c_prev))

        # Rolling 4-week z-score
        for i in range(4, len(w_rets)):
            window = [w_rets[j][1] for j in range(i - 4, i)]
            cur_ret = w_rets[i][1]
            cur_di = w_rets[i][0]
            m_val = np.mean(window)
            s_val = np.std(window, ddof=1)
            if s_val > 1e-10:
                z = (cur_ret - m_val) / s_val
                weekly_zscore[si, cur_di] = z

    print(f"  Weekly z-scores computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # G) CROSS-COMMODITY LEAD-LAG
    # ================================================================
    print("\n[Signals] G) Cross-Commodity Lead-Lag: precomputing...", flush=True)
    t0 = time.time()

    lead_lag_indices = []
    for leader_sym, follower_sym in LEAD_LAG_PAIRS:
        si_leader = sym_to_si.get(leader_sym, -1)
        si_follower = sym_to_si.get(follower_sym, -1)
        if si_leader >= 0 and si_follower >= 0:
            lead_lag_indices.append((si_leader, si_follower, leader_sym, follower_sym))
        else:
            print(f"  WARNING: lead-lag pair ({leader_sym}, {follower_sym}) not found")
    print(f"  Active lead-lag pairs: {len(lead_lag_indices)}")

    print(f"  Lead-lag setup done ({time.time()-t0:.1f}s)")

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(config, wf_test_year=None):
        """
        Config:
            signal: 'oi_surge' | 'oi_capit' | 'seasonal' | 'oi_mom' |
                    'vol_breakout' | 'weekly_mr' | 'lead_lag'
            hold_days: int
            threshold: float (signal-specific)
            top_n: int (max concurrent positions)
            comm: float
        """
        sig_type = config['signal']
        hold_days = config['hold_days']
        threshold = config['threshold']
        top_n = config['top_n']
        comm = config.get('comm', COMM)
        rebalance_days = config.get('rebalance_days', 0)

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
        positions = []  # {si, entry_price, entry_di, lots, dir, sym, hold_days}
        trades = []
        last_rebalance_di = -999

        for di in range(start_di, end_di - 1):  # need di+1 for entry
            # Reset cash at test window start (WF mode)
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                positions = []
                last_rebalance_di = -999

            # ── Close positions that have been held long enough ───────
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

            # ── Generate signals at day di ───────────────────────────
            entry_di = di + 1
            if entry_di >= end_di:
                continue

            # For rebalanced strategies, check if it's time to rebalance
            if rebalance_days > 0:
                if (di - last_rebalance_di) < rebalance_days:
                    continue  # skip signal generation until rebalance time
                # Close all positions at rebalance
                for pos in list(positions):
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
                positions = []
                last_rebalance_di = di

            candidates = []  # (score, direction, info_dict)

            if sig_type == 'oi_surge':
                # A) OI Surge Reversal: OI up > 20% + price down > 3% in 5 days
                for si in range(NS):
                    if not oi_surge_signal[si, di]:
                        continue
                    if any(p['si'] == si for p in positions):
                        continue
                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    # Score: bigger OI surge + bigger price drop
                    oc = oi_chg5[si, di] if not np.isnan(oi_chg5[si, di]) else 0
                    r5 = ret5[si, di] if not np.isnan(ret5[si, di]) else 0
                    score = oc - r5  # high OI growth + big drop = strong signal
                    candidates.append((score, 1, {
                        'si': si, 'sym': syms[si], 'entry_price': ep,
                    }))

            elif sig_type == 'oi_capit':
                # B) OI Decline + Capitulation: OI down > 10% + price down > 3%
                for si in range(NS):
                    if not oi_capit_signal[si, di]:
                        continue
                    if any(p['si'] == si for p in positions):
                        continue
                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    oc = oi_chg5[si, di] if not np.isnan(oi_chg5[si, di]) else 0
                    r5 = ret5[si, di] if not np.isnan(ret5[si, di]) else 0
                    score = -oc - r5  # bigger OI decline + bigger drop
                    candidates.append((score, 1, {
                        'si': si, 'sym': syms[si], 'entry_price': ep,
                    }))

            elif sig_type == 'seasonal':
                # C) Seasonal: buy if current month historically strong
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    ss = seasonal_score[si, di]
                    if np.isnan(ss):
                        continue
                    if ss > threshold:  # threshold = min avg daily return for the month
                        ep = O[si, entry_di]
                        if np.isnan(ep) or ep <= 0:
                            continue
                        candidates.append((ss, 1, {
                            'si': si, 'sym': syms[si], 'entry_price': ep,
                        }))

            elif sig_type == 'oi_mom':
                # D) OI Momentum: rank by OI change, buy top K
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    rk = oi_mom_rank[si, di]
                    if np.isnan(rk):
                        continue
                    if rk > threshold:  # threshold = min rank (e.g., 0.7 = top 30%)
                        ep = O[si, entry_di]
                        if np.isnan(ep) or ep <= 0:
                            continue
                        oc = oi_chg20[si, di] if not np.isnan(oi_chg20[si, di]) else 0
                        candidates.append((oc, 1, {
                            'si': si, 'sym': syms[si], 'entry_price': ep,
                        }))

            elif sig_type == 'vol_breakout':
                # E) Volatility Breakout: range > 2*ATR20 + bullish candle
                for si in range(NS):
                    if not breakout_signal[si, di]:
                        continue
                    if any(p['si'] == si for p in positions):
                        continue
                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    bs = breakout_score[si, di] if not np.isnan(breakout_score[si, di]) else 2.0
                    candidates.append((bs, 1, {
                        'si': si, 'sym': syms[si], 'entry_price': ep,
                    }))

            elif sig_type == 'weekly_mr':
                # F) Weekly Mean-Reversion: weekly z < -threshold
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    wz = weekly_zscore[si, di]
                    if np.isnan(wz):
                        continue
                    if wz < -threshold:
                        ep = O[si, entry_di]
                        if np.isnan(ep) or ep <= 0:
                            continue
                        candidates.append((-wz, 1, {
                            'si': si, 'sym': syms[si], 'entry_price': ep,
                        }))

            elif sig_type == 'lead_lag':
                # G) Cross-Commodity Lead-Lag
                for si_leader, si_follower, leader_sym, follower_sym in lead_lag_indices:
                    lr = ret3[si_leader, di]
                    if np.isnan(lr):
                        continue
                    if any(p['si'] == si_follower for p in positions):
                        continue
                    if lr > threshold:  # leader's 3d return > threshold
                        ep = O[si_follower, entry_di]
                        if np.isnan(ep) or ep <= 0:
                            continue
                        candidates.append((lr, 1, {
                            'si': si_follower, 'sym': follower_sym,
                            'entry_price': ep,
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
                lots = int(cash / (notional * (1 + comm) * top_n))  # equal weight
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
                actual_hold = rebalance_days if rebalance_days > 0 else hold_days
                positions.append({
                    'si': si, 'entry_price': price, 'entry_di': entry_di,
                    'lots': lots, 'dir': direction, 'sym': sym,
                    'hold_days': actual_hold if actual_hold > 0 else hold_days,
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

    # --- A: OI Surge Reversal: hold 3/5/10, top_n 1/3/5 ---
    for hd in [3, 5, 10]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'oi_surge',
                'hold_days': hd, 'threshold': 0.20,
                'top_n': tn, 'comm': COMM,
                'label': f"OI_Surge_H{hd}_TN{tn}",
            })

    # --- B: OI Decline + Capitulation: hold 3/5/10, top_n 1/3/5 ---
    for hd in [3, 5, 10]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'oi_capit',
                'hold_days': hd, 'threshold': 0.10,
                'top_n': tn, 'comm': COMM,
                'label': f"OI_Capit_H{hd}_TN{tn}",
            })

    # --- C: Seasonal: threshold for monthly avg daily return, hold 20, top_n 3/5/10 ---
    for thresh in [0.0002, 0.0005, 0.001, 0.002]:
        for tn in [3, 5, 10]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'seasonal',
                'hold_days': 20, 'threshold': thresh,
                'top_n': tn, 'comm': COMM,
                'rebalance_days': 20,
                'label': f"Season_T{thresh}_TN{tn}_R20",
            })

    # --- D: OI Momentum: rank threshold, top 3/5/10, rebalance 10d ---
    for thresh in [0.7, 0.8, 0.9]:
        for tn in [3, 5, 10]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'oi_mom',
                'hold_days': 10, 'threshold': thresh,
                'top_n': tn, 'comm': COMM,
                'rebalance_days': 10,
                'label': f"OI_Mom_RK{thresh}_TN{tn}_R10",
            })

    # --- E: Volatility Breakout: hold 5/10, top_n 1/3/5 ---
    for hd in [5, 10]:
        for tn in [1, 3, 5]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'vol_breakout',
                'hold_days': hd, 'threshold': 2.0,
                'top_n': tn, 'comm': COMM,
                'label': f"VolBreak_H{hd}_TN{tn}",
            })

    # --- F: Weekly Mean-Reversion: z threshold, hold 10/20 (2w/4w), top_n 1/3/5 ---
    for thresh in [1.0, 1.5, 2.0]:
        for hd in [10, 20]:
            for tn in [1, 3, 5]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': 'weekly_mr',
                    'hold_days': hd, 'threshold': thresh,
                    'top_n': tn, 'comm': COMM,
                    'label': f"WkMR_Z{thresh}_H{hd}_TN{tn}",
                })

    # --- G: Lead-Lag: threshold for leader return, hold 3/5 ---
    for thresh in [0.02, 0.03, 0.05]:
        for hd in [3, 5]:
            for tn in [1, 3, 5]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': 'lead_lag',
                    'hold_days': hd, 'threshold': thresh,
                    'top_n': tn, 'comm': COMM,
                    'label': f"LeadLag_T{thresh}_H{hd}_TN{tn}",
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
    print(f"  {'#':>3} | {'Label':<40} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'Final':>14}")
    print("-" * 130)
    for i, r in enumerate(results[:30]):
        print(f"  {i+1:>3} | {r['label']:<40} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}% | {r['final_cash']:>13,.0f}")

    # ================================================================
    # BEST PER SIGNAL TYPE (full period)
    # ================================================================
    sig_order = ['oi_surge', 'oi_capit', 'seasonal', 'oi_mom',
                 'vol_breakout', 'weekly_mr', 'lead_lag']
    sig_names = {
        'oi_surge': 'A) OI Surge Reversal',
        'oi_capit': 'B) OI Decline + Capitulation',
        'seasonal': 'C) Seasonal Pattern',
        'oi_mom': 'D) OI Momentum',
        'vol_breakout': 'E) Volatility Breakout',
        'weekly_mr': 'F) Weekly Mean-Reversion',
        'lead_lag': 'G) Cross-Commodity Lead-Lag',
    }

    print(f"\n{'=' * 150}")
    print("  BEST PER SIGNAL TYPE (Full Period) -- ALL NEXT-OPEN EXECUTION")
    print(f"{'=' * 150}")
    print(f"  {'Signal':<42} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | Best Config")
    print("-" * 150)

    best_per_sig = {}
    for r in results:
        key = r['config']['signal']
        if key not in best_per_sig:
            best_per_sig[key] = r

    for sig in sig_order:
        if sig in best_per_sig:
            b = best_per_sig[sig]
            print(f"  {sig_names.get(sig, sig):<42} | {b['ann']:>+8.1f}% | {b['wr']:>5.1f}% | {b['n']:>5} | {b['avg_pnl']:>+6.3f}% | {b['mdd']:>6.1f}% | {b['label']}")

    # ================================================================
    # SIGNAL TYPE SUMMARY
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  SIGNAL TYPE SUMMARY (Average of Top 5 configs per type)")
    print(f"{'=' * 150}")
    print(f"  {'Signal':<42} | {'Avg Ann':>9} | {'Avg WR':>7} | {'Avg N':>7} | {'Avg PnL':>8} | {'Avg MDD':>8} | {'#Positive':>9}")
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

    print(f"\n{'=' * 170}")
    print(f"  WALK-FORWARD ({len(wf_configs)} configs)")
    print(f"{'=' * 170}")

    header = f"  {'#':>3} | {'Config':<40} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>7} |"
    header += f" {'Pos':>4} | {'MDD':>7}"
    print(header)
    print("-" * 170)

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

        row_str = f"  {i+1:>3} | {wf_row['label']:<40} | {avg:>+7.1f}% |"
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
    header2 = f"  {'Signal':<42} | {'WF Avg':>8} |"
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
            row_str = f"  {sig_names.get(sig, sig):<42} | {avg:>+7.1f}% |"
            for v in vals:
                row_str += f" {v:>+7.1f}% |"
            row_str += f" {pos}/6 | {avg_mdd:>6.1f}%"
            print(row_str)

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  FINAL VERDICT: OI DYNAMICS, SEASONALITY & LEAD-LAG WITH NEXT-OPEN EXECUTION")
    print(f"{'=' * 150}")
    print()
    print("  KEY QUESTION: Can OI dynamics, seasonality, or lead-lag achieve meaningful")
    print("  returns with practical (next-open) execution?")
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

    print(f"\n  Total time: {time.time()-t_start:.1f}s")
    print("=" * 150)


if __name__ == '__main__':
    main()
