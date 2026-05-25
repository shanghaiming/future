"""
Alpha Futures V97 -- Genuine Forward-Looking Predictive Power Test
==================================================================
V96 proved that ALL cross-sectional z-score strategies (V74/V82/V92) have
ZERO genuine predictive power. Their +3000-4000% returns came entirely from
same-bar contamination (signal and entry at same close). With next-open
execution, ALL go negative.

Now test strategies with GENUINE forward-looking predictive power -- signals
based on PERSISTENT states, not daily events.

STRATEGIES TESTED (all next-open execution):
  A) V62 Pair Trading: LOG spread mean-reversion, z > 1.0, hold 1/2/3/5 days
  B) V82 Cross-Group 5D Z-Score: 5-day cumulative return z-score, not 1-day
  C) Trend Quality + Group Divergence: 20d regression slope + R-squared
  D) Supply Chain Spread Momentum: 5d spread change, buy lagging downstream
  E) Multi-Day Momentum Reversal: bottom 20% over 5 days + return < -5%

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

# Group map
GROUP_MAP = {}
for _s in ['rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi']:
    GROUP_MAP[_s] = 'ferrous'
for _s in ['cufi', 'alfi', 'znfi', 'nifi', 'pbfi', 'snfi', 'ssfi', 'sffi']:
    GROUP_MAP[_s] = 'nonferrous'
for _s in ['aufi', 'agfi']:
    GROUP_MAP[_s] = 'precious'
for _s in ['afi', 'mfi', 'yfi', 'pfi', 'cfi', 'csfi', 'rrfi', 'lrfi']:
    GROUP_MAP[_s] = 'oils'
for _s in ['scfi', 'mafi', 'bfi', 'fufi', 'pgfi', 'ebfi', 'fbfi']:
    GROUP_MAP[_s] = 'energy'
for _s in ['ppfi', 'vfi', 'egfi', 'tafi', 'fgfi', 'lfi']:
    GROUP_MAP[_s] = 'chemical'
for _s in ['whfi', 'apfi', 'cjfi', 'oifi', 'rmfi', 'srfi', 'cffi']:
    GROUP_MAP[_s] = 'soft'
for _s in ['jdfi', 'lhfi', 'pkfi']:
    GROUP_MAP[_s] = 'livestock'

# Supply chain pairs for pair trading (from V62 context)
PAIRS = [
    # steel chain
    ('rbfi', 'ifi'), ('rbfi', 'jfi'), ('hcfi', 'ifi'),
    # oil chain
    ('scfi', 'mafi'), ('scfi', 'tafi'),
    # soy chain
    ('afi', 'mfi'), ('afi', 'yfi'), ('mfi', 'yfi'), ('pfi', 'yfi'),
    # chemical chain
    ('ppfi', 'vfi'), ('ppfi', 'lfi'), ('tafi', 'egfi'), ('ppfi', 'egfi'),
]

# Supply chain pairs for spread momentum (upstream, downstream)
SPREAD_MOM_PAIRS = [
    ('scfi', 'mafi'),   # crude -> methanol
    ('scfi', 'tafi'),   # crude -> PTA
    ('scfi', 'ppfi'),   # crude -> PP
    ('scfi', 'egfi'),   # crude -> EG
    ('ifi', 'rbfi'),    # iron ore -> rebar
    ('ifi', 'hcfi'),    # iron ore -> hot coil
    ('jfi', 'jmfi'),    # coke -> coking coal
    ('afi', 'mfi'),     # soybean -> meal
    ('afi', 'yfi'),     # soybean -> soyoil
    ('afi', 'pfi'),     # soybean -> palm (proxy)
]


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def main():
    print("=" * 140)
    print("Alpha Futures V97 -- Genuine Forward-Looking Predictive Power Test")
    print("=" * 140)
    print("\n  CRITICAL: After V96 proved all cross-sectional z-scores have zero")
    print("  predictive power with next-open execution, we now test strategies")
    print("  based on PERSISTENT states (spread MR, multi-day momentum, trends).")

    # ── Load data ────────────────────────────────────────────────────
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    sym_to_si = {syms[si]: si for si in range(NS)}

    # ── Build group membership ───────────────────────────────────────
    gm_map = {}
    si_group = {}
    for si in range(NS):
        g = GROUP_MAP.get(syms[si])
        if g:
            gm_map.setdefault(g, []).append(si)
            si_group[si] = g

    trade_sis = [si for si in range(NS) if si in si_group]
    group_names = sorted(gm_map.keys())
    print(f"  Tradeable: {len(trade_sis)} commodities in {len(group_names)} groups")

    # ================================================================
    # PRECOMPUTE RETURNS
    # ================================================================
    print("\n[Signals] Computing returns...", flush=True)
    t0 = time.time()

    # Close-to-close 1-day return
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

    print(f"  Returns computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # A) PAIR TRADING: LOG SPREAD + Z-SCORE
    # ================================================================
    print("\n[Signals] A) Computing pair trading spreads...", flush=True)
    t0 = time.time()

    pair_indices = []
    for sym_a, sym_b in PAIRS:
        si_a = sym_to_si.get(sym_a, -1)
        si_b = sym_to_si.get(sym_b, -1)
        if si_a >= 0 and si_b >= 0:
            pair_indices.append((si_a, si_b, sym_a, sym_b))
        else:
            print(f"  WARNING: pair ({sym_a}, {sym_b}) not found")
    print(f"  Active pairs: {len(pair_indices)}")

    # Compute log spreads and z-scores for multiple lookbacks
    pair_zscores = {}  # (si_a, si_b, lb) -> z_array
    pair_spreads = {}  # (si_a, si_b) -> spread_array

    for si_a, si_b, sym_a, sym_b in pair_indices:
        # LOG spread
        spread = np.full(ND, np.nan)
        for di in range(ND):
            ca = C[si_a, di]
            cb = C[si_b, di]
            if not np.isnan(ca) and not np.isnan(cb) and ca > 0 and cb > 0:
                spread[di] = np.log(ca) - np.log(cb)
        pair_spreads[(si_a, si_b)] = spread

        # Z-scores for multiple lookbacks
        for lb in [10, 15, 20, 30]:
            z = np.full(ND, np.nan)
            for di in range(lb, ND):
                window = spread[di - lb:di]
                valid = window[~np.isnan(window)]
                if len(valid) >= max(3, int(lb * 0.8)):
                    m_val = np.mean(valid)
                    s_val = np.std(valid, ddof=1)
                    if s_val > 1e-10:
                        z[di] = (spread[di] - m_val) / s_val
            pair_zscores[(si_a, si_b, lb)] = z

    print(f"  Pair z-scores computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # B) CROSS-GROUP 5-DAY Z-SCORE
    # ================================================================
    print("\n[Signals] B) Computing 5-day cross-group z-scores...", flush=True)
    t0 = time.time()

    # Group-level 5d return average
    grp_ret5 = {}
    for grp in group_names:
        arr = np.full(ND, np.nan)
        members = gm_map[grp]
        for di in range(5, ND):
            vals = [ret5[sk, di] for sk in members if not np.isnan(ret5[sk, di])]
            if vals:
                arr[di] = np.mean(vals)
        grp_ret5[grp] = arr

    # All-groups average and std of 5d returns
    all_grp_avg5 = np.full(ND, np.nan)
    all_grp_std5 = np.full(ND, np.nan)
    for di in range(5, ND):
        vals = [grp_ret5[g][di] for g in group_names if not np.isnan(grp_ret5[g][di])]
        if len(vals) >= 2:
            all_grp_avg5[di] = np.mean(vals)
            all_grp_std5[di] = np.std(vals)

    # 5-day cross-group z-score for each commodity
    z_5d = np.full((NS, ND), np.nan)
    for si in trade_sis:
        for di in range(5, ND):
            own = ret5[si, di]
            if np.isnan(own) or np.isnan(all_grp_avg5[di]) or np.isnan(all_grp_std5[di]):
                continue
            if all_grp_std5[di] < 1e-8:
                continue
            z_5d[si, di] = (own - all_grp_avg5[di]) / all_grp_std5[di]

    print(f"  5-day z-scores computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # C) TREND QUALITY + GROUP DIVERGENCE
    # ================================================================
    print("\n[Signals] C) Computing trend quality (20d regression)...", flush=True)
    t0 = time.time()

    slope_20 = np.full((NS, ND), np.nan)
    rsq_20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            prices = C[si, di - 20:di]
            valid_mask = ~np.isnan(prices)
            n_valid = np.sum(valid_mask)
            if n_valid < 15:
                continue
            y = prices[valid_mask]
            n = len(y)
            x = np.arange(n, dtype=float)
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            ss_xx = np.sum((x - x_mean) ** 2)
            ss_xy = np.sum((x - x_mean) * (y - y_mean))
            ss_yy = np.sum((y - y_mean) ** 2)
            if ss_xx < 1e-12 or ss_yy < 1e-12:
                continue
            beta = ss_xy / ss_xx
            # Normalize slope to percentage terms
            if y_mean > 0:
                slope_20[si, di] = beta / y_mean * 100  # daily % slope
            rsq_20[si, di] = (ss_xy ** 2) / (ss_xx * ss_yy)

    # Group average slope
    grp_slope = {}
    for grp in group_names:
        arr = np.full(ND, np.nan)
        members = gm_map[grp]
        for di in range(20, ND):
            vals = [slope_20[sk, di] for sk in members if not np.isnan(slope_20[sk, di])]
            if vals:
                arr[di] = np.mean(vals)
        grp_slope[grp] = arr

    print(f"  Trend quality computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # D) SUPPLY CHAIN SPREAD MOMENTUM
    # ================================================================
    print("\n[Signals] D) Computing supply chain spread momentum...", flush=True)
    t0 = time.time()

    spread_mom_indices = []
    for sym_up, sym_dn in SPREAD_MOM_PAIRS:
        si_up = sym_to_si.get(sym_up, -1)
        si_dn = sym_to_si.get(sym_dn, -1)
        if si_up >= 0 and si_dn >= 0:
            spread_mom_indices.append((si_up, si_dn, sym_up, sym_dn))
    print(f"  Spread momentum pairs: {len(spread_mom_indices)}")

    # Compute 5-day change in log spread
    spread_mom_5d = {}  # (si_up, si_dn) -> 5d_spread_change array
    for si_up, si_dn, sym_up, sym_dn in spread_mom_indices:
        log_spread = np.full(ND, np.nan)
        for di in range(ND):
            cu = C[si_up, di]
            cd = C[si_dn, di]
            if not np.isnan(cu) and not np.isnan(cd) and cu > 0 and cd > 0:
                log_spread[di] = np.log(cu) - np.log(cd)

        chg5 = np.full(ND, np.nan)
        for di in range(5, ND):
            s0 = log_spread[di - 5]
            s5 = log_spread[di]
            if not np.isnan(s0) and not np.isnan(s5):
                chg5[di] = s5 - s0
        spread_mom_5d[(si_up, si_dn)] = chg5

    print(f"  Spread momentum computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # E) MULTI-DAY MOMENTUM REVERSAL
    # ================================================================
    print("\n[Signals] E) Computing multi-day momentum reversal...", flush=True)
    t0 = time.time()

    # For each day, rank commodities by 5-day return
    # Bottom 20% AND ret < -5% -> buy signal
    pct_rank5 = np.full((NS, ND), np.nan)
    for di in range(5, ND):
        vals = []
        for si in trade_sis:
            r = ret5[si, di]
            if not np.isnan(r):
                vals.append((si, r))
        if len(vals) < 5:
            continue
        vals.sort(key=lambda x: x[1])
        n_vals = len(vals)
        for rank, (si, r) in enumerate(vals):
            pct_rank5[si, di] = rank / n_vals  # 0 = worst, 1 = best

    print(f"  Momentum reversal computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(config, wf_test_year=None):
        """
        Config:
            signal: 'pair_trading' | 'cross_group_5d' | 'trend_div' |
                    'spread_momentum' | 'momentum_reversal'
            hold_days: int
            threshold: float (signal-specific)
            lb: int (lookback, for pairs)
            top_n: int (max concurrent positions)
            comm: float
        """
        sig_type = config['signal']
        hold_days = config['hold_days']
        threshold = config['threshold']
        lb = config.get('lb', 15)
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
        positions = []  # {si, si2(for pairs), entry_price, entry_price2, entry_di, lots, dir, sym, hold_days}
        trades = []

        for di in range(start_di, end_di - 1):  # need di+1 for entry
            # Reset cash at test window start (WF mode)
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                positions = []

            # ── Close positions that have been held long enough ───────
            closed = []
            for pos in positions:
                days_held = di - pos['entry_di']
                should_exit = False

                if sig_type == 'pair_trading':
                    # For pairs, exit when z crosses 0 OR after hold_days
                    if days_held >= hold_days:
                        should_exit = True
                    elif days_held >= 1:
                        # Check if z-score crossed zero
                        zkey = (pos['si'], pos['si2'], lb)
                        z_arr = pair_zscores.get(zkey)
                        if z_arr is not None and di < ND:
                            z_now = z_arr[di]
                            if not np.isnan(z_now) and pos['dir'] == 1 and z_now > 0:
                                should_exit = True
                            elif not np.isnan(z_now) and pos['dir'] == -1 and z_now < 0:
                                should_exit = True
                else:
                    if days_held >= hold_days:
                        should_exit = True

                if not should_exit:
                    continue

                if sig_type == 'pair_trading':
                    # Close both legs
                    exit_a = C[pos['si'], di]
                    exit_b = C[pos['si2'], di]
                    if np.isnan(exit_a) or exit_a <= 0:
                        exit_a = pos['entry_price']
                    if np.isnan(exit_b) or exit_b <= 0:
                        exit_b = pos['entry_price2']
                    mult_a = MULT.get(pos['sym'], DEF_MULT)
                    mult_b = MULT.get(pos['sym2'], DEF_MULT)

                    # Leg A: dir * (exit - entry)
                    mkt_a = exit_a * mult_a * abs(pos['lots'])
                    mkt_b = exit_b * mult_b * abs(pos['lots'])

                    pnl_a = (exit_a - pos['entry_price']) * mult_a * pos['lots'] * pos['dir']
                    pnl_b = (exit_b - pos['entry_price2']) * mult_b * pos['lots'] * (-pos['dir'])
                    # Return cash from both legs
                    cash += mkt_a - mkt_a * comm
                    cash += mkt_b - mkt_b * comm
                    total_invested = pos['entry_price'] * mult_a * abs(pos['lots']) + pos['entry_price2'] * mult_b * abs(pos['lots'])
                    pnl = pnl_a + pnl_b
                    pnl_pct = pnl / total_invested * 100 if total_invested > 0 else 0
                else:
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

            candidates = []  # (score, direction, info_dict)

            if sig_type == 'pair_trading':
                # --- A: Pair Trading LOG spread mean-reversion ---
                for si_a, si_b, sym_a, sym_b in pair_indices:
                    zkey = (si_a, si_b, lb)
                    z_arr = pair_zscores.get(zkey)
                    if z_arr is None:
                        continue
                    z = z_arr[di]
                    if np.isnan(z):
                        continue
                    # Check entry prices available at next open
                    ea = O[si_a, entry_di]
                    eb = O[si_b, entry_di]
                    if np.isnan(ea) or ea <= 0 or np.isnan(eb) or eb <= 0:
                        continue

                    if z > threshold:
                        # Short spread: sell A, buy B (expect spread to revert down)
                        candidates.append((-z, -1, {
                            'si': si_a, 'si2': si_b, 'sym': sym_a, 'sym2': sym_b,
                            'entry_price': ea, 'entry_price2': eb,
                        }))
                    elif z < -threshold:
                        # Long spread: buy A, sell B (expect spread to revert up)
                        candidates.append((z, 1, {
                            'si': si_a, 'si2': si_b, 'sym': sym_a, 'sym2': sym_b,
                            'entry_price': ea, 'entry_price2': eb,
                        }))

            elif sig_type == 'cross_group_5d':
                # --- B: 5-day cross-group z-score ---
                for si in trade_sis:
                    if any(p['si'] == si for p in positions):
                        continue
                    z = z_5d[si, di]
                    if np.isnan(z):
                        continue
                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    # z < -threshold -> buy (oversold relative to groups)
                    if z < -threshold:
                        score = -z
                        candidates.append((score, 1, {
                            'si': si, 'sym': syms[si], 'entry_price': ep,
                        }))

            elif sig_type == 'trend_div':
                # --- C: Trend Quality + Group Divergence ---
                for si in trade_sis:
                    if any(p['si'] == si for p in positions):
                        continue
                    sl = slope_20[si, di]
                    rq = rsq_20[si, di]
                    if np.isnan(sl) or np.isnan(rq):
                        continue
                    grp = si_group.get(si)
                    if grp is None:
                        continue
                    g_slope = grp_slope[grp][di]
                    if np.isnan(g_slope):
                        continue
                    # Strong downtrend (slope < 0, R2 > 0.5) but group uptrending
                    if sl < 0 and rq > 0.5 and g_slope > 0:
                        ep = O[si, entry_di]
                        if np.isnan(ep) or ep <= 0:
                            continue
                        score = abs(sl) * rq  # stronger trend divergence = higher score
                        candidates.append((score, 1, {
                            'si': si, 'sym': syms[si], 'entry_price': ep,
                        }))

            elif sig_type == 'spread_momentum':
                # --- D: Supply Chain Spread Momentum ---
                for si_up, si_dn, sym_up, sym_dn in spread_mom_indices:
                    chg5 = spread_mom_5d.get((si_up, si_dn))
                    if chg5 is None:
                        continue
                    val = chg5[di]
                    if np.isnan(val):
                        continue
                    # If spread widened (upstream outperformed) -> buy downstream
                    if val > threshold:
                        ep = O[si_dn, entry_di]
                        if np.isnan(ep) or ep <= 0:
                            continue
                        score = val
                        candidates.append((score, 1, {
                            'si': si_dn, 'sym': sym_dn, 'entry_price': ep,
                        }))
                    # If spread narrowed (downstream outperformed) -> buy upstream
                    elif val < -threshold:
                        ep = O[si_up, entry_di]
                        if np.isnan(ep) or ep <= 0:
                            continue
                        score = -val
                        candidates.append((score, 1, {
                            'si': si_up, 'sym': sym_up, 'entry_price': ep,
                        }))

            elif sig_type == 'momentum_reversal':
                # --- E: Multi-Day Momentum Reversal ---
                for si in trade_sis:
                    if any(p['si'] == si for p in positions):
                        continue
                    rk = pct_rank5[si, di]
                    r5 = ret5[si, di]
                    if np.isnan(rk) or np.isnan(r5):
                        continue
                    # Bottom 20% AND return < -5%
                    if rk < 0.2 and r5 < -threshold:
                        ep = O[si, entry_di]
                        if np.isnan(ep) or ep <= 0:
                            continue
                        score = -r5  # bigger drop = higher score
                        candidates.append((score, 1, {
                            'si': si, 'sym': syms[si], 'entry_price': ep,
                        }))

            if not candidates:
                continue

            # Sort by score descending
            candidates.sort(key=lambda x: -x[0])

            # Open positions
            if sig_type == 'pair_trading':
                # Each pair trade takes 2 legs but counts as 1 position
                n_current = len(positions)
                for score, direction, info in candidates:
                    if len(positions) - n_current >= top_n:
                        break
                    si_a = info['si']
                    si_b = info['si2']
                    ea = info['entry_price']
                    eb = info['entry_price2']
                    sym_a = info['sym']
                    sym_b = info['sym2']
                    mult_a = MULT.get(sym_a, DEF_MULT)
                    mult_b = MULT.get(sym_b, DEF_MULT)

                    # Use smaller of the two legs for sizing
                    notional = ea * mult_a + eb * mult_b
                    lots = int(cash * 0.4 / (notional * (1 + comm)))  # 40% per pair
                    if lots <= 0:
                        continue
                    cost_in = notional * lots * (1 + comm)
                    if cost_in > cash:
                        lots = int(cash * 0.38 / (notional * (1 + comm)))
                        cost_in = notional * lots * (1 + comm) if lots > 0 else 0
                    if lots <= 0 or cost_in <= 0 or cost_in > cash:
                        continue

                    cash -= cost_in

                    # For pair trading:
                    # dir=1: buy A, sell B (long spread)
                    # dir=-1: sell A, buy B (short spread)
                    # We pay for leg A entry; leg B we receive cash (short)
                    # Net cost = |leg_a| - |leg_b|, but for simplicity we track full notional
                    positions.append({
                        'si': si_a, 'si2': si_b,
                        'entry_price': ea, 'entry_price2': eb,
                        'entry_di': entry_di,
                        'lots': lots, 'dir': direction,
                        'sym': sym_a, 'sym2': sym_b,
                        'hold_days': hold_days,
                    })
            else:
                # Directional trades (long only)
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
                    positions.append({
                        'si': si, 'entry_price': price, 'entry_di': entry_di,
                        'lots': lots, 'dir': direction, 'sym': sym,
                        'hold_days': hold_days,
                    })

        # Close remaining positions at end
        for pos in positions:
            if sig_type == 'pair_trading':
                ae = end_di - 1 if end_di < ND else ND - 1
                exit_a = C[pos['si'], ae]
                exit_b = C[pos['si2'], ae]
                if np.isnan(exit_a) or exit_a <= 0:
                    exit_a = pos['entry_price']
                if np.isnan(exit_b) or exit_b <= 0:
                    exit_b = pos['entry_price2']
                mult_a = MULT.get(pos['sym'], DEF_MULT)
                mult_b = MULT.get(pos['sym2'], DEF_MULT)
                mkt_a = exit_a * mult_a * abs(pos['lots'])
                mkt_b = exit_b * mult_b * abs(pos['lots'])
                cash += mkt_a - mkt_a * comm
                cash += mkt_b - mkt_b * comm
            else:
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

    # --- A: Pair Trading: LOG spread, z threshold, multiple lookbacks, hold 1/2/3/5 ---
    for lb in [10, 15, 20, 30]:
        for thresh in [0.8, 1.0, 1.5, 2.0]:
            for tn in [2, 4, 6]:
                for hd in [1, 2, 3, 5]:
                    cid += 1
                    configs.append({
                        'id': cid, 'signal': 'pair_trading',
                        'hold_days': hd, 'threshold': thresh,
                        'lb': lb, 'top_n': tn, 'comm': COMM,
                        'label': f"Pair_LB{lb}_Z{thresh}_TN{tn}_H{hd}",
                    })

    # --- B: Cross-Group 5D Z-Score: threshold, hold 1/3/5 ---
    for thresh in [0.3, 0.5, 0.7, 1.0]:
        for tn in [1, 3, 5]:
            for hd in [1, 3, 5]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': 'cross_group_5d',
                    'hold_days': hd, 'threshold': thresh,
                    'top_n': tn, 'comm': COMM,
                    'label': f"CG5d_Z{thresh}_TN{tn}_H{hd}",
                })

    # --- C: Trend Quality + Group Divergence: hold 3/5/10 ---
    # (threshold is R-squared threshold, fixed at 0.5)
    for tn in [1, 3, 5]:
        for hd in [3, 5, 10]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'trend_div',
                'hold_days': hd, 'threshold': 0.5,
                'top_n': tn, 'comm': COMM,
                'label': f"TrendDiv_RQ50_TN{tn}_H{hd}",
            })

    # --- D: Supply Chain Spread Momentum: threshold, hold 1/3/5 ---
    for thresh in [0.02, 0.03, 0.05, 0.08]:
        for tn in [1, 3, 5]:
            for hd in [1, 3, 5]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': 'spread_momentum',
                    'hold_days': hd, 'threshold': thresh,
                    'top_n': tn, 'comm': COMM,
                    'label': f"SpreadMom_T{thresh}_TN{tn}_H{hd}",
                })

    # --- E: Multi-Day Momentum Reversal: threshold = min 5d drop, hold 3/5 ---
    for thresh in [0.03, 0.05, 0.08, 0.10]:
        for tn in [1, 3, 5]:
            for hd in [3, 5]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': 'momentum_reversal',
                    'hold_days': hd, 'threshold': thresh,
                    'top_n': tn, 'comm': COMM,
                    'label': f"MomRev_T{thresh}_TN{tn}_H{hd}",
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
    print(f"\n{'=' * 140}")
    print("  FULL-PERIOD RESULTS (Top 30)")
    print(f"{'=' * 140}")
    print(f"  {'#':>3} | {'Label':<40} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'Final':>14}")
    print("-" * 120)
    for i, r in enumerate(results[:30]):
        print(f"  {i+1:>3} | {r['label']:<40} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}% | {r['final_cash']:>13,.0f}")

    # ================================================================
    # BEST PER SIGNAL TYPE (full period)
    # ================================================================
    sig_order = ['pair_trading', 'cross_group_5d', 'trend_div',
                 'spread_momentum', 'momentum_reversal']
    sig_names = {
        'pair_trading': 'A) Pair Trading (LOG spread MR)',
        'cross_group_5d': 'B) Cross-Group 5D Z-Score',
        'trend_div': 'C) Trend Quality + Group Div',
        'spread_momentum': 'D) Supply Chain Spread Mom',
        'momentum_reversal': 'E) Multi-Day Momentum Reversal',
    }

    print(f"\n{'=' * 140}")
    print("  BEST PER SIGNAL TYPE (Full Period) -- ALL NEXT-OPEN EXECUTION")
    print(f"{'=' * 140}")
    print(f"  {'Signal':<42} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | Best Config")
    print("-" * 140)

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
    # SIGNAL TYPE SUMMARY: avg of top 5 per type
    # ================================================================
    print(f"\n{'=' * 140}")
    print("  SIGNAL TYPE SUMMARY (Average of Top 5 configs per type)")
    print(f"{'=' * 140}")
    print(f"  {'Signal':<42} | {'Avg Ann':>9} | {'Avg WR':>7} | {'Avg N':>7} | {'Avg PnL':>8} | {'Avg MDD':>8} | {'#Positive':>9}")
    print("-" * 140)

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
    # PAIR TRADING DETAIL: BY LOOKBACK AND HOLD
    # ================================================================
    print(f"\n{'=' * 140}")
    print("  PAIR TRADING DETAIL (Best per lookback x hold)")
    print(f"{'=' * 140}")
    print(f"  {'Lookback':>8} | {'Hold':>4} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | Best Config")
    print("-" * 130)

    for lb in [10, 15, 20, 30]:
        for hd in [1, 2, 3, 5]:
            sub = [r for r in results
                   if r['config']['signal'] == 'pair_trading'
                   and r['config']['lb'] == lb
                   and r['config']['hold_days'] == hd]
            if sub:
                best = sub[0]
                print(f"  {lb:>8} | {hd:>4} | {best['ann']:>+8.1f}% | {best['wr']:>5.1f}% | {best['n']:>5} | {best['avg_pnl']:>+6.3f}% | {best['mdd']:>6.1f}% | {best['label']}")

    # ================================================================
    # CROSS-GROUP 5D DETAIL: BY HOLD
    # ================================================================
    print(f"\n{'=' * 140}")
    print("  CROSS-GROUP 5D Z-SCORE DETAIL (Best per hold)")
    print(f"{'=' * 140}")
    print(f"  {'Hold':>4} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | Best Config")
    print("-" * 130)

    for hd in [1, 3, 5]:
        sub = [r for r in results
               if r['config']['signal'] == 'cross_group_5d'
               and r['config']['hold_days'] == hd]
        if sub:
            best = sub[0]
            print(f"  {hd:>4} | {best['ann']:>+8.1f}% | {best['wr']:>5.1f}% | {best['n']:>5} | {best['avg_pnl']:>+6.3f}% | {best['mdd']:>6.1f}% | {best['label']}")

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

    print(f"\n{'=' * 160}")
    print(f"  WALK-FORWARD ({len(wf_configs)} configs)")
    print(f"{'=' * 160}")

    header = f"  {'#':>3} | {'Config':<40} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>7} |"
    header += f" {'Pos':>4} | {'MDD':>7}"
    print(header)
    print("-" * 160)

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
    print(f"\n{'=' * 140}")
    print("  WALK-FORWARD COMPARISON (Best per signal type)")
    print(f"{'=' * 140}")
    header2 = f"  {'Signal':<42} | {'WF Avg':>8} |"
    for yr in wf_years:
        header2 += f" {yr:>7} |"
    header2 += f" {'Pos':>4} | Avg MDD"
    print(header2)
    print("-" * 140)

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
    print(f"\n{'=' * 140}")
    print("  FINAL VERDICT: WHICH STRATEGIES HAVE GENUINE FORWARD-LOOKING PREDICTIVE POWER?")
    print(f"{'=' * 140}")
    print()
    print("  KEY QUESTION: Which strategies remain positive with next-open execution?")
    print("  (V96 showed all cross-sectional 1-day z-scores go NEGATIVE)")
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
    print("=" * 140)


if __name__ == '__main__':
    main()
