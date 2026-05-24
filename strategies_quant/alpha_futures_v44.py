"""
Alpha Futures V44 — Seasonal Pattern Strategy for Chinese Commodity Futures
===========================================================================
Core idea: Agricultural commodities (soybeans, corn, sugar) and energy
(crude oil, natural gas) have strong seasonal patterns driven by
planting/harvesting cycles and weather. Detect and trade these patterns
as an alpha source orthogonal to momentum and pair trading.

Signals:
  S1: Seasonal Return Prediction — historically bullish DOY → long
  S2: Seasonal + Momentum Confirmation — seasonal up + mom up
  S3: Seasonal + OI Confirmation — seasonal up + OI rising
  S4: Anti-seasonal Reversal — bullish season but price dipped → buy dip
  S5: Cross-commodity Seasonal Alignment — group consensus

Config sweep (~200 configs):
  signal_type: [1, 2, 3, 4, 5]
  seasonal_threshold: [0.001, 0.003, 0.005]
  hit_min: [0.55, 0.60, 0.65]
  window: [5, 10, 15]
  hold: [5, 10, 15]
  top_n: [1, 3, 5]
  Walk-forward: 2023, 2024
"""
import sys, os, time, warnings
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

GROUP_MAP = {
    'rbfi': 'ferrous', 'hcfi': 'ferrous', 'ifi': 'ferrous', 'jfi': 'ferrous', 'jmfi': 'ferrous',
    'cufi': 'nonferrous', 'alfi': 'nonferrous', 'znfi': 'nonferrous', 'nifi': 'nonferrous',
    'afi': 'oils', 'mfi': 'oils', 'yfi': 'oils', 'pfi': 'oils', 'cfi': 'oils',
    'scfi': 'energy', 'mafi': 'energy', 'bfi': 'energy', 'fufi': 'energy',
    'ppfi': 'chemical', 'vfi': 'chemical', 'egfi': 'chemical', 'pgfi': 'chemical',
}


def main():
    t_start = time.time()
    print("=" * 120)
    print("Alpha Futures V44 — Seasonal Pattern Strategy for Chinese Commodity Futures")
    print("Core: trade agricultural/energy seasonal patterns (planting/harvest/weather cycles)")
    print("=" * 120)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    sym_to_si = {syms[si]: si for si in range(NS)}
    group_members = {}
    for si in range(NS):
        grp = GROUP_MAP.get(syms[si])
        if grp is None:
            continue
        if grp not in group_members:
            group_members[grp] = []
        group_members[grp].append(si)

    print(f"  {NS} commodities, {ND} days, Groups: {len(group_members)}")

    # ========================================
    # PRECOMPUTE RETURNS AND DOY
    # ========================================
    print("\n[Signals] Computing returns, DOY, momentum, OI...", flush=True)
    t0 = time.time()

    # Daily returns
    ret = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            c_now = C[si, di]
            c_prev = C[si, di - 1]
            if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                ret[si, di] = (c_now - c_prev) / c_prev

    # Day-of-year array
    doy_arr = np.zeros(ND, dtype=np.int32)
    year_arr = np.zeros(ND, dtype=np.int32)
    for di in range(ND):
        doy_arr[di] = dates[di].timetuple().tm_yday
        year_arr[di] = dates[di].year

    # 5-day momentum
    mom5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            c_now = C[si, di]
            c_prev = C[si, di - 5]
            if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                mom5[si, di] = (c_now - c_prev) / c_prev

    # OI rising: 5-day EMA trend
    oi_ema = np.full((NS, ND), np.nan)
    oi_rising = np.full((NS, ND), False)
    for si in range(NS):
        oe = 0.0
        alpha_oi = 2.0 / 6
        for di in range(1, ND):
            oi_val = OI[si, di]
            if np.isnan(oi_val):
                continue
            oe = alpha_oi * oi_val + (1 - alpha_oi) * oe
            oi_ema[si, di] = oe
        for di in range(6, ND):
            cur = oi_ema[si, di]
            prev = oi_ema[si, di - 5]
            if not np.isnan(cur) and not np.isnan(prev) and prev > 0:
                oi_rising[si, di] = (cur - prev) / prev > 0.01

    # ATR for trailing stops
    atr10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(11, ND):
            trs = []
            for dd in range(di - 10, di):
                hi, lo, pc = H[si, dd], L[si, dd], C[si, dd - 1]
                if np.isnan(hi) or np.isnan(lo):
                    continue
                tr = hi - lo
                if not np.isnan(pc):
                    tr = max(tr, abs(hi - pc), abs(lo - pc))
                trs.append(tr)
            if trs:
                atr10[si, di] = np.mean(trs)

    print(f"  Basic signals done ({time.time() - t0:.1f}s)", flush=True)

    # ========================================
    # PRECOMPUTE SEASONAL STATISTICS
    # ========================================
    # For each (si, di), we need seasonal stats computed from ONLY prior years.
    # We precompute a lookup: for each si, a dict keyed by (year, doy) returning
    # the accumulated seasonal stats up to but NOT including that year.
    #
    # To speed this up, we first collect all returns by (si, doy) across years,
    # then for each (si, year) compute expanding-window stats.

    print("\n[Signals] Computing seasonal statistics...", flush=True)
    t1 = time.time()

    # Collect returns by (si, doy) across all years
    # ret_by_si_doy[si][doy] = list of (year, return)
    ret_by_si_doy = [[[] for _ in range(367)] for _ in range(NS)]
    for si in range(NS):
        for di in range(1, ND):
            r = ret[si, di]
            if not np.isnan(r):
                ret_by_si_doy[si][doy_arr[di]].append((year_arr[di], r))

    # For a given window_days, for each DOY we look at DOYs in [doy - window, doy + window]
    # and aggregate returns from prior years only.
    # We precompute for multiple window sizes.

    def compute_seasonal_stats(window_days):
        """
        Returns arrays:
          seasonal_ret[si, di] = mean return on this DOY (window) from prior years
          seasonal_hit[si, di] = fraction of positive returns
          seasonal_mag[si, di] = mean absolute return
          seasonal_n[si, di] = number of prior-year observations
        """
        seasonal_ret = np.full((NS, ND), np.nan)
        seasonal_hit = np.full((NS, ND), np.nan)
        seasonal_mag = np.full((NS, ND), np.nan)
        seasonal_n = np.full((NS, ND), 0, dtype=np.int32)

        for si in range(NS):
            # For efficiency, pre-sort returns by year for each DOY
            # For each target DOY d, gather returns from DOYs [d-w, d+w] in prior years
            cur_year = -1
            # Accumulator per year: for each year, all returns in the window
            year_returns = {}  # year -> list of returns (across window DOYs)

            # Pre-collect: for each DOY in the window range, collect returns
            # We iterate through dates and build expanding-window stats per (year, doy)
            # Actually, we compute per-di directly for clarity
            for di in range(1, ND):
                y = year_arr[di]
                d = doy_arr[di]

                # Gather returns from DOYs [d - window_days, d + window_days]
                # but ONLY from years strictly before current year
                prior_rets = []
                for wd in range(d - window_days, d + window_days + 1):
                    # Wrap around year boundary
                    wd_adj = wd
                    if wd_adj < 1:
                        wd_adj += 365
                    elif wd_adj > 365:
                        wd_adj -= 365
                    if wd_adj < 1 or wd_adj > 366:
                        continue
                    for yr, r in ret_by_si_doy[si][wd_adj]:
                        if yr < y:
                            prior_rets.append(r)

                if len(prior_rets) >= 5:  # need at least some observations
                    seasonal_ret[si, di] = np.mean(prior_rets)
                    seasonal_hit[si, di] = sum(1 for r in prior_rets if r > 0) / len(prior_rets)
                    seasonal_mag[si, di] = np.mean([abs(r) for r in prior_rets])
                    seasonal_n[si, di] = len(prior_rets)

        return seasonal_ret, seasonal_hit, seasonal_mag, seasonal_n

    seasonal_cache = {}
    for wd in [5, 10, 15]:
        print(f"  Computing seasonal stats for window={wd}...", flush=True)
        sr, sh, sm, sn = compute_seasonal_stats(wd)
        seasonal_cache[wd] = (sr, sh, sm, sn)
        print(f"    window={wd} done ({time.time() - t1:.1f}s)", flush=True)

    print(f"  All seasonal stats computed ({time.time() - t0:.1f}s)", flush=True)

    # ========================================
    # SIGNAL SCORING FUNCTIONS
    # ========================================

    def make_score(signal_type, threshold, hit_min, window):
        """
        Return a scoring function score(si, di) -> float or nan.
        signal_type: 1-5
        threshold: minimum absolute seasonal return to trigger
        hit_min: minimum hit rate for seasonal bullish
        window: DOY window size for seasonal stats
        """
        sr, sh, sm, sn = seasonal_cache[window]

        def score(si, di):
            s_ret = sr[si, di]
            s_hit = sh[si, di]
            s_n = sn[si, di]

            # Need at least some prior-year observations
            # With 2 prior years and window of 10, expect ~40 obs min
            if s_n < 20:
                return np.nan
            if np.isnan(s_ret) or np.isnan(s_hit):
                return np.nan

            if signal_type == 1:
                # S1: Pure seasonal return prediction
                if s_ret > threshold and s_hit > hit_min:
                    return s_ret * s_hit * 100  # scale for ranking
                return np.nan

            elif signal_type == 2:
                # S2: Seasonal + Momentum confirmation
                if s_ret > threshold and s_hit > hit_min:
                    m = mom5[si, di]
                    if np.isnan(m):
                        return np.nan
                    if m > 0:
                        return s_ret * s_hit * 100 * (1 + m * 10)
                    return np.nan
                return np.nan

            elif signal_type == 3:
                # S3: Seasonal + OI confirmation
                if s_ret > threshold and s_hit > hit_min:
                    if oi_rising[si, di]:
                        return s_ret * s_hit * 100 * 1.5
                    return np.nan
                return np.nan

            elif signal_type == 4:
                # S4: Anti-seasonal reversal
                # Bullish season but price dropped recently -> buy the dip
                if s_ret > threshold and s_hit > hit_min:
                    m = mom5[si, di]
                    if np.isnan(m):
                        return np.nan
                    if m < -0.03:
                        # Stronger dip = stronger signal (capped)
                        dip_strength = min(abs(m) / 0.1, 2.0)
                        return s_ret * s_hit * 100 * (1 + dip_strength)
                    return np.nan
                return np.nan

            elif signal_type == 5:
                # S5: Cross-commodity seasonal alignment
                sym = syms[si]
                grp = GROUP_MAP.get(sym)
                if grp is None:
                    return np.nan
                members = group_members.get(grp, [])
                if len(members) < 2:
                    return np.nan

                # Count how many group members are seasonally bullish today
                bullish_count = 0
                for sj in members:
                    s_ret_j = sr[sj, di]
                    s_hit_j = sh[sj, di]
                    s_n_j = sn[sj, di]
                    if (not np.isnan(s_ret_j) and not np.isnan(s_hit_j)
                            and s_n_j >= 20 and s_ret_j > threshold and s_hit_j > hit_min):
                        bullish_count += 1

                pct_bullish = bullish_count / len(members)
                if pct_bullish >= 0.6:
                    # This commodity itself must also be seasonally bullish
                    if s_ret > threshold and s_hit > hit_min:
                        return s_ret * s_hit * 100 * (1 + pct_bullish)
                return np.nan

            return np.nan

        return score

    # ========================================
    # BACKTEST ENGINE
    # ========================================
    def run_backtest(score_fn, name, top_n=1, hold_max=5,
                     trail_atr_mult=3.0, wf_split_year=None):
        """
        Single position per symbol, long only.
        Exit: time exit or trailing stop.
        """
        cash = float(CASH0)
        trades = []
        positions = []
        last_exit = {}

        for di in range(MIN_TRAIN, ND):
            year = dates[di].year

            # Walk-forward: skip training period
            if wf_split_year is not None and year < wf_split_year:
                continue

            # Manage existing positions
            new_positions = []
            for pos in positions:
                c = C[pos['si'], di]
                if np.isnan(c) or c <= 0:
                    c = pos['entry']
                mult = MULT.get(pos['sym'], DEF_MULT)
                mkt_val = c * mult * pos['lots']
                pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
                pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] > 0 else 0
                days_held = di - pos['entry_di']

                exit_reason = None

                # Trailing stop
                if trail_atr_mult > 0 and days_held >= 2:
                    atr = pos.get('atr', 0)
                    if atr > 0 and pos['dir'] == 1:
                        new_trail = c - trail_atr_mult * atr
                        if new_trail > pos.get('trail_price', pos['entry']):
                            pos['trail_price'] = new_trail
                        if c < pos['trail_price']:
                            exit_reason = 'trail'

                # Time exit
                if exit_reason is None and days_held >= hold_max:
                    exit_reason = 'time'

                if exit_reason:
                    cost_out = mkt_val * COMM
                    cash += mkt_val - cost_out
                    trades.append({
                        'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                        'days': days_held, 'di': di, 'year': year,
                        'sym': pos['sym'], 'dir': pos['dir'], 'reason': exit_reason,
                        'doy': doy_arr[di],
                    })
                    last_exit[pos['sym']] = di
                else:
                    new_positions.append(pos)

            positions = new_positions

            # Open new positions
            n_open = len(positions)
            if n_open < top_n:
                slots = top_n - n_open
                scored = []
                for si in range(NS):
                    sc = score_fn(si, di)
                    if np.isnan(sc) or sc <= 0.01:
                        continue
                    sym = syms[si]
                    if any(p['sym'] == sym for p in positions):
                        continue
                    scored.append((si, sc, sym))

                if scored:
                    scored.sort(key=lambda x: -x[1])
                    cash_per_slot = cash / slots if slots > 0 else cash

                    for best_si, best_sc, best_sym in scored[:slots]:
                        c = C[best_si, di]
                        if np.isnan(c) or c <= 0:
                            continue
                        mult = MULT.get(best_sym, DEF_MULT)
                        notional = c * mult
                        if notional <= 0:
                            continue

                        lots = int(cash_per_slot / (notional * (1 + COMM)))
                        if lots <= 0:
                            continue
                        cost_in = notional * lots * (1 + COMM)
                        if cost_in > cash:
                            lots = int(cash / (notional * (1 + COMM)))
                            if lots <= 0:
                                continue
                            cost_in = notional * lots * (1 + COMM)

                        atr_val = atr10[best_si, di] if not np.isnan(atr10[best_si, di]) else 0
                        cash -= cost_in
                        trail_price = c - trail_atr_mult * atr_val if atr_val > 0 else c * 0.97
                        positions.append({
                            'si': best_si, 'entry': c, 'entry_di': di,
                            'lots': lots, 'dir': 1, 'sym': best_sym,
                            'atr': atr_val, 'trail_price': trail_price,
                        })

        # Close remaining
        for pos in positions:
            c = C[pos['si'], ND - 1]
            if np.isnan(c) or c <= 0:
                c = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
            cash += c * mult * pos['lots'] * (1 - COMM)
            trades.append({
                'pnl_pct': pnl / (pos['entry'] * mult * pos['lots']) * 100,
                'pnl_abs': pnl, 'days': ND - 1 - pos['entry_di'],
                'di': ND - 1, 'year': dates[ND - 1].year,
                'sym': pos['sym'], 'dir': pos['dir'], 'reason': 'end',
                'doy': doy_arr[ND - 1],
            })

        if len(trades) < 10:
            return None

        # Stats
        equity = float(CASH0)
        peak = float(CASH0)
        max_dd = 0
        for t in sorted(trades, key=lambda x: x['di']):
            equity += t['pnl_abs']
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (peak - equity) / peak * 100
                if dd > max_dd:
                    max_dd = dd

        days_total = (dates[ND - 1] - dates[MIN_TRAIN]).days
        yr = max(days_total / 365.25, 0.01)
        if wf_split_year:
            first_test_di = None
            for di in range(MIN_TRAIN, ND):
                if dates[di].year >= wf_split_year:
                    first_test_di = di
                    break
            if first_test_di:
                days_total = (dates[ND - 1] - dates[first_test_di]).days
                yr = max(days_total / 365.25, 0.01)
        else:
            pass

        ann = ((cash / CASH0) ** (1 / yr) - 1) * 100

        nw = sum(1 for t in trades if t['pnl_abs'] > 0)
        wr = nw / len(trades) * 100
        avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_abs'] > 0]) if nw > 0 else 0
        avg_loss = np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_abs'] <= 0]) if nw < len(trades) else 0
        avg_days = np.mean([t['days'] for t in trades])
        pf = (sum(t['pnl_abs'] for t in trades if t['pnl_abs'] > 0) /
              max(abs(sum(t['pnl_abs'] for t in trades if t['pnl_abs'] < 0)), 1))

        reasons = {}
        for t in trades:
            r = t['reason']
            if r not in reasons:
                reasons[r] = {'n': 0, 'w': 0, 'pnl': 0.0}
            reasons[r]['n'] += 1
            if t['pnl_abs'] > 0:
                reasons[r]['w'] += 1
            reasons[r]['pnl'] += t['pnl_pct']

        year_stats = {}
        for t in trades:
            y = t['year']
            if y not in year_stats:
                year_stats[y] = {'n': 0, 'w': 0, 'pnl': 0.0}
            year_stats[y]['n'] += 1
            if t['pnl_abs'] > 0:
                year_stats[y]['w'] += 1
            year_stats[y]['pnl'] += t['pnl_pct']

        grp_counts = {}
        for t in trades:
            g = GROUP_MAP.get(t['sym'], 'other')
            if g not in grp_counts:
                grp_counts[g] = {'n': 0, 'w': 0, 'pnl': 0.0}
            grp_counts[g]['n'] += 1
            if t['pnl_abs'] > 0:
                grp_counts[g]['w'] += 1
            grp_counts[g]['pnl'] += t['pnl_abs']

        # DOY/month breakdown
        month_stats = {}
        for t in trades:
            # Reconstruct month from DOY
            t_di = t['di']
            if 0 <= t_di < ND:
                m = dates[t_di].month
            else:
                m = 0
            if m not in month_stats:
                month_stats[m] = {'n': 0, 'w': 0, 'pnl': 0.0}
            month_stats[m]['n'] += 1
            if t['pnl_abs'] > 0:
                month_stats[m]['w'] += 1
            month_stats[m]['pnl'] += t['pnl_pct']

        # DOY range breakdown (split year into quarters by ~91 days)
        doy_range_stats = {}
        for t in trades:
            d = t.get('doy', 0)
            if d <= 91:
                rng = 'DOY_001-091'
            elif d <= 182:
                rng = 'DOY_092-182'
            elif d <= 273:
                rng = 'DOY_183-273'
            else:
                rng = 'DOY_274-365'
            if rng not in doy_range_stats:
                doy_range_stats[rng] = {'n': 0, 'w': 0, 'pnl': 0.0}
            doy_range_stats[rng]['n'] += 1
            if t['pnl_abs'] > 0:
                doy_range_stats[rng]['w'] += 1
            doy_range_stats[rng]['pnl'] += t['pnl_pct']

        return {
            'name': name, 'ann': round(ann, 1), 'n': len(trades),
            'wr': round(wr, 1), 'dd': round(max_dd, 1),
            'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
            'avg_days': round(avg_days, 1), 'pf': round(pf, 2),
            'cash': round(cash, 0),
            'reasons': reasons, 'yearly': year_stats, 'grp_counts': grp_counts,
            'month_stats': month_stats, 'doy_range_stats': doy_range_stats,
        }

    # ========================================
    # PARAMETER SWEEP
    # ========================================
    print("\n[Backtest] Building configurations...", flush=True)
    results = []
    configs = []

    signal_types = [1, 2, 3, 4, 5]
    thresholds = [0.001, 0.003, 0.005]
    hit_mins = [0.55, 0.60, 0.65]
    windows = [5, 10, 15]
    holds = [5, 10, 15]
    top_ns = [1, 3, 5]

    sig_labels = {1: 'S1_Seasonal', 2: 'S2_SeasMom', 3: 'S3_SeasOI',
                  4: 'S4_AntiSeas', 5: 'S5_GrpAlign'}

    for st in signal_types:
        for thr in thresholds:
            for hm in hit_mins:
                for w in windows:
                    for h in holds:
                        for tn in top_ns:
                            name = f"{sig_labels[st]}_T{thr*1000:.0f}_H{hm*100:.0f}_W{w}_D{h}_N{tn}"
                            configs.append((
                                make_score(st, thr, hm, w),
                                name, tn, h, 3.0, None
                            ))

    # Walk-forward configs for best parameter ranges
    for st in signal_types:
        for thr in [0.001, 0.003, 0.005]:
            for hm in [0.55, 0.60]:
                for w in [5, 10, 15]:
                    for h in [5, 10, 15]:
                        for tn in [1, 3]:
                            for wf_year in [2023, 2024]:
                                name = (f"{sig_labels[st]}_T{thr*1000:.0f}_H{hm*100:.0f}"
                                        f"_W{w}_D{h}_N{tn}_WF{wf_year}")
                                configs.append((
                                    make_score(st, thr, hm, w),
                                    name, tn, h, 3.0, wf_year
                                ))

    print(f"  {len(configs)} configurations to test", flush=True)

    print("\n[Backtest] Running sweep...", flush=True)
    for ci, (fn, name, tn, h, trail, wf) in enumerate(configs):
        r = run_backtest(fn, name, top_n=tn, hold_max=h,
                         trail_atr_mult=trail, wf_split_year=wf)
        if r and r['ann'] > 0:
            results.append(r)
            if r['ann'] > 30:
                print(f"  {r['name']:55s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                      f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                      f"AvgW {r['avg_win']:+.2f}% | AvgL {r['avg_loss']:.2f}% | AvgD {r['avg_days']:.1f}")

        if (ci + 1) % 100 == 0:
            print(f"  [{ci+1}/{len(configs)}] {len(results)} profitable", flush=True)

    # ========================================
    # RESULTS
    # ========================================
    results.sort(key=lambda x: -x['ann'])

    wf_results = [r for r in results if '_WF' in r['name']]
    full_results = [r for r in results if '_WF' not in r['name']]

    print(f"\n{'=' * 130}")
    print(f"  TOP 20 FULL-PERIOD RESULTS")
    print(f"{'=' * 130}")
    print(f"  {'Config':55s} | {'Ann':>7s} | {'WR':>5s} | {'N':>4s} | {'DD':>6s} | "
          f"{'PF':>4s} | {'AvgW':>7s} | {'AvgL':>6s} | {'AvgD':>4s}")
    print(f"  {'-' * 130}")
    for r in full_results[:20]:
        print(f"  {r['name']:55s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:4.2f} | "
              f"{r['avg_win']:+6.2f}% | {r['avg_loss']:5.2f}% | "
              f"{r['avg_days']:4.1f}")

    if wf_results:
        wf_results.sort(key=lambda x: -x['ann'])
        print(f"\n  TOP 10 WALK-FORWARD RESULTS (out-of-sample)")
        print(f"  {'-' * 130}")
        for r in wf_results[:10]:
            print(f"  {r['name']:55s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f}")

    # Best config details
    if full_results:
        best = full_results[0]
        print(f"\n{'=' * 130}")
        print(f"  BEST CONFIG DETAIL: {best['name']}")
        print(f"  Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  N={best['n']}  "
              f"DD={best['dd']:.1f}%  PF={best['pf']:.2f}  Final={best['cash']:.0f}")
        print(f"{'=' * 130}")

        print(f"\n  EXIT REASON BREAKDOWN:")
        for reason, s in sorted(best['reasons'].items(), key=lambda x: -x[1]['n']):
            rwr = s['w'] / max(s['n'], 1) * 100
            print(f"    {reason:12s}: {s['n']:4d} trades  WR={rwr:5.1f}%  PnL={s['pnl']:+.1f}%")

        print(f"\n  YEARLY BREAKDOWN:")
        for y in sorted(best['yearly'].keys()):
            s = best['yearly'][y]
            wr = s['w'] / max(s['n'], 1) * 100
            print(f"    {y}: {s['n']:3d} trades  WR={wr:5.1f}%  PnL={s['pnl']:+.1f}%")

        print(f"\n  GROUP BREAKDOWN:")
        for g in sorted(best['grp_counts'].keys(), key=lambda x: -best['grp_counts'][x]['n']):
            gs = best['grp_counts'][g]
            wr_g = gs['w'] / max(gs['n'], 1) * 100
            print(f"    {g:15s}: {gs['n']:3d}t  WR={wr_g:5.1f}%  Abs={gs['pnl']:+.0f}")

        print(f"\n  MONTHLY BREAKDOWN (which months are most profitable):")
        for m in sorted(best['month_stats'].keys()):
            ms = best['month_stats'][m]
            if ms['n'] == 0:
                continue
            wr_m = ms['w'] / ms['n'] * 100
            print(f"    Month {m:2d}: {ms['n']:3d}t  WR={wr_m:5.1f}%  PnL={ms['pnl']:+.1f}%")

        print(f"\n  DOY RANGE BREAKDOWN:")
        for rng in sorted(best['doy_range_stats'].keys()):
            ds = best['doy_range_stats'][rng]
            if ds['n'] == 0:
                continue
            wr_d = ds['w'] / ds['n'] * 100
            print(f"    {rng}: {ds['n']:3d}t  WR={wr_d:5.1f}%  PnL={ds['pnl']:+.1f}%")

    # Yearly for top 5
    if len(full_results) >= 2:
        print(f"\n  YEARLY BREAKDOWN FOR TOP 5:")
        for idx, r in enumerate(full_results[:5]):
            print(f"\n  #{idx+1}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.1f}%, DD={r['dd']:.1f}%)")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:3d}t  WR={wr_y:5.1f}%  PnL={ys['pnl']:+.1f}%")

    # ========================================
    # SEASONAL ANALYSIS ACROSS TOP RESULTS
    # ========================================
    if full_results:
        print(f"\n{'=' * 130}")
        print(f"  SEASONAL ANALYSIS ACROSS TOP 20 CONFIGS")
        print(f"{'=' * 130}")

        # Aggregate monthly stats across top 20
        agg_month = {}
        for r in full_results[:20]:
            for m, ms in r['month_stats'].items():
                if m not in agg_month:
                    agg_month[m] = {'n': 0, 'w': 0, 'pnl': 0.0}
                agg_month[m]['n'] += ms['n']
                agg_month[m]['w'] += ms['w']
                agg_month[m]['pnl'] += ms['pnl']

        print(f"\n  AGGREGATE MONTHLY PROFITABILITY (top 20 configs combined):")
        month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                       7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        for m in sorted(agg_month.keys()):
            ms = agg_month[m]
            if ms['n'] == 0:
                continue
            wr_m = ms['w'] / ms['n'] * 100
            print(f"    {month_names.get(m, str(m)):3s} (Month {m:2d}): "
                  f"{ms['n']:5d}t  WR={wr_m:5.1f}%  PnL={ms['pnl']:+.1f}%")

        # Aggregate group stats across top 20
        agg_grp = {}
        for r in full_results[:20]:
            for g, gs in r['grp_counts'].items():
                if g not in agg_grp:
                    agg_grp[g] = {'n': 0, 'w': 0, 'pnl': 0.0}
                agg_grp[g]['n'] += gs['n']
                agg_grp[g]['w'] += gs['w']
                agg_grp[g]['pnl'] += gs['pnl']

        print(f"\n  COMMODITY GROUP SEASONAL STRENGTH (top 20 configs combined):")
        for g in sorted(agg_grp.keys(), key=lambda x: -agg_grp[x]['pnl']):
            gs = agg_grp[g]
            if gs['n'] == 0:
                continue
            wr_g = gs['w'] / gs['n'] * 100
            print(f"    {g:15s}: {gs['n']:5d}t  WR={wr_g:5.1f}%  Total Abs={gs['pnl']:+12.0f}")

        # Signal type breakdown across top 20
        sig_stats = {}
        for r in full_results[:20]:
            for sig_label in sig_labels.values():
                if r['name'].startswith(sig_label):
                    if sig_label not in sig_stats:
                        sig_stats[sig_label] = {'count': 0, 'ann_sum': 0.0, 'best_ann': -999}
                    sig_stats[sig_label]['count'] += 1
                    sig_stats[sig_label]['ann_sum'] += r['ann']
                    sig_stats[sig_label]['best_ann'] = max(sig_stats[sig_label]['best_ann'], r['ann'])

        if sig_stats:
            print(f"\n  SIGNAL TYPE EFFECTIVENESS (in top 20):")
            for sig, ss in sorted(sig_stats.items(), key=lambda x: -x[1]['best_ann']):
                avg = ss['ann_sum'] / ss['count'] if ss['count'] > 0 else 0
                print(f"    {sig:20s}: appears {ss['count']:2d}x  "
                      f"best_ann={ss['best_ann']:+.1f}%  avg_ann={avg:+.1f}%")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
