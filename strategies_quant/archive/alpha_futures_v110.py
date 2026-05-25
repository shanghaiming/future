"""
Alpha Futures V110 -- CROSS-COMMODITY SIGNALS with Next-Open Execution
=======================================================================
Current best: ROC(5) cross +81.9%, 6/6 WF.

V110 IDEA: Use information from correlated commodities (same group or supply chain)
to predict individual commodity moves. All signals computed at close di, entry at O[di+1].

8 signal types (A-H) testing cross-commodity information transfer:
A) Group Momentum Lag -- group avg > own return -> expect catch-up
B) Supply Chain Lead-Lag -- upstream leads, downstream follows
C) Cross-Commodity ROC Rank -- rank jump detection
D) Group Relative Strength -- cross above group avg
E) Pair Divergence -- mean-reversion on supply chain spreads
F) Group OI Flow -- new money entering group + commodity
G) Cross-Group Momentum Transfer -- ferrous leads -> nonferrous/chemical lag
H) Multi-Commodity Vote -- group confirmation of positive momentum
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

GROUP_MAP = {}
for _s in ['rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi']: GROUP_MAP[_s] = 'ferrous'
for _s in ['cufi', 'alfi', 'znfi', 'nifi', 'pbfi', 'snfi', 'ssfi', 'sffi']: GROUP_MAP[_s] = 'nonferrous'
for _s in ['aufi', 'agfi']: GROUP_MAP[_s] = 'precious'
for _s in ['afi', 'mfi', 'yfi', 'pfi', 'cfi', 'csfi', 'rrfi', 'lrfi']: GROUP_MAP[_s] = 'oils'
for _s in ['scfi', 'mafi', 'bfi', 'fufi', 'pgfi', 'ebfi', 'fbfi']: GROUP_MAP[_s] = 'energy'
for _s in ['ppfi', 'vfi', 'egfi', 'tafi', 'fgfi', 'lfi']: GROUP_MAP[_s] = 'chemical'
for _s in ['whfi', 'apfi', 'cjfi', 'oifi', 'rmfi', 'cffi', 'srfi']: GROUP_MAP[_s] = 'soft'
for _s in ['jdfi', 'lhfi', 'pkfi']: GROUP_MAP[_s] = 'livestock'

PAIRS = [
    ('ifi', 'rbfi'), ('ifi', 'hcfi'), ('rbfi', 'hcfi'),  # iron->steel
    ('jmfi', 'jfi'),  # coal->coke
    ('scfi', 'mafi'), ('scfi', 'bfi'), ('scfi', 'fufi'),  # crude->downstream
    ('afi', 'mfi'), ('afi', 'yfi'), ('mfi', 'yfi'),  # soybean chain
    ('yfi', 'pfi'),  # soy oil->palm oil
    ('mafi', 'ppfi'), ('mafi', 'vfi'), ('mafi', 'egfi'),  # methanol->chemicals
]


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def main():
    print("=" * 200)
    print("Alpha Futures V110 -- CROSS-COMMODITY SIGNALS with Next-Open Execution")
    print("=" * 200)
    print("\n  Using information from correlated/supply-chain commodities to predict moves.")
    print("  ALL signals at close di, entry at O[si, di+1] (NEXT DAY OPEN)")
    print("  8 cross-commodity signal types (A-H)")

    # -- Load data -------------------------------------------------
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # Build symbol index map
    sym_idx = {s: i for i, s in enumerate(syms)}

    # Build group membership arrays
    group_members = {}  # group_name -> [si, ...]
    si_group = {}       # si -> group_name
    for si in range(NS):
        g = GROUP_MAP.get(syms[si])
        if g:
            group_members.setdefault(g, []).append(si)
            si_group[si] = g

    print(f"  Groups: {list(group_members.keys())}")
    for g, members in sorted(group_members.items()):
        print(f"    {g}: {len(members)} commodities")

    # ================================================================
    # PRECOMPUTE ROC(5) AND OI CHANGES
    # ================================================================
    print("\n[Precompute] ROC(5) and OI changes...", flush=True)
    t0 = time.time()

    ROC5 = np.full((NS, ND), np.nan)
    OI_chg5 = np.full((NS, ND), np.nan)
    SMA20_price = np.full((NS, ND), np.nan)

    for si in range(NS):
        c = C[si].astype(np.float64)
        oi = OI[si].astype(np.float64)
        ROC5[si] = talib.ROC(c, timeperiod=5)
        # OI 5-day change (percentage)
        oi_sma5 = talib.SMA(oi, timeperiod=5)
        oi_sma5_prev = np.roll(oi_sma5, 5)
        with np.errstate(divide='ignore', invalid='ignore'):
            OI_chg5[si] = (oi_sma5 - oi_sma5_prev) / np.abs(oi_sma5_prev) * 100
        SMA20_price[si] = talib.SMA(c, timeperiod=20)

    # Previous day ROC5 for cross detection
    ROC5_prev = np.roll(ROC5, 1, axis=1)
    ROC5_prev[:, 0] = np.nan

    print(f"  ROC(5) and OI computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # A) GROUP MOMENTUM LAG
    # ================================================================
    print("\n[Signals] A) Group Momentum Lag...", flush=True)
    t0 = time.time()

    # Precompute group average ROC5 (excluding self)
    group_avg_roc5 = np.full((NS, ND), np.nan)
    for di in range(ND):
        for g, members in group_members.items():
            vals = [ROC5[m, di] for m in members if not np.isnan(ROC5[m, di])]
            if not vals:
                continue
            g_avg = np.mean(vals)
            for m in members:
                group_avg_roc5[m, di] = g_avg

    # Signal: group_avg_roc5 - own_roc5 > threshold (group ahead, expect catch-up)
    sig_A = {}  # (threshold, hold, top_n) -> signal array
    for threshold in [0, 0.005, 0.01, 0.02]:
        sig_arr = np.zeros((NS, ND), dtype=bool)
        for si in range(NS):
            if si not in si_group:
                continue
            for di in range(10, ND):
                own = ROC5[si, di]
                g_avg = group_avg_roc5[si, di]
                if np.isnan(own) or np.isnan(g_avg):
                    continue
                # Group is ahead -> buy expecting catch-up
                if (g_avg - own) > threshold:
                    sig_arr[si, di] = True
        sig_A[threshold] = sig_arr
        n_sig = np.sum(sig_arr)
        print(f"    threshold={threshold:.3f}: {n_sig} signals")

    print(f"    ({time.time()-t0:.1f}s)")

    # ================================================================
    # B) SUPPLY CHAIN LEAD-LAG
    # ================================================================
    print("\n[Signals] B) Supply Chain Lead-Lag...", flush=True)
    t0 = time.time()

    # For each pair, if upstream ROC5 > 0 AND downstream ROC5 < upstream ROC5 -> buy downstream
    sig_B = np.zeros((NS, ND), dtype=bool)
    pair_signals_count = {}

    for upstream_sym, downstream_sym in PAIRS:
        usi = sym_idx.get(upstream_sym)
        dsi = sym_idx.get(downstream_sym)
        if usi is None or dsi is None:
            continue

        pair_count = 0
        for di in range(10, ND):
            u_roc = ROC5[usi, di]
            d_roc = ROC5[dsi, di]
            if np.isnan(u_roc) or np.isnan(d_roc):
                continue
            # Upstream is positive AND upstream outperforming downstream -> buy downstream (catch-up)
            if u_roc > 0 and d_roc < u_roc:
                sig_B[dsi, di] = True
                pair_count += 1

        pair_signals_count[(upstream_sym, downstream_sym)] = pair_count

    print(f"    Total signals: {np.sum(sig_B)}")
    for pair, cnt in sorted(pair_signals_count.items(), key=lambda x: -x[1]):
        print(f"      {pair[0]}->{pair[1]}: {cnt} signals")

    print(f"    ({time.time()-t0:.1f}s)")

    # ================================================================
    # C) CROSS-COMMODITY ROC RANK (rank jump detection)
    # ================================================================
    print("\n[Signals] C) Cross-Commodity ROC Rank...", flush=True)
    t0 = time.time()

    # Rank all commodities by ROC5 each day
    roc_rank = np.full((NS, ND), np.nan)  # 1=best, NS=worst
    for di in range(10, ND):
        vals = []
        for si in range(NS):
            if not np.isnan(ROC5[si, di]):
                vals.append((ROC5[si, di], si))
        if len(vals) < 10:
            continue
        vals.sort(key=lambda x: -x[0])  # descending
        for rank, (v, si) in enumerate(vals, 1):
            roc_rank[si, di] = rank
        # Store total count
        n_valid = len(vals)
        for _, si in vals:
            # Normalize rank to 0-1 range (percentile)
            pass  # we'll use raw rank

    roc_rank_prev = np.roll(roc_rank, 1, axis=1)
    roc_rank_prev[:, 0] = np.nan

    # Signal: rank jumped from bottom half to top half
    sig_C = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(11, ND):
            r_now = roc_rank[si, di]
            r_prev = roc_rank_prev[si, di]
            if np.isnan(r_now) or np.isnan(r_prev):
                continue
            # Count valid commodities on this day for normalization
            n_valid = np.sum(~np.isnan(ROC5[:, di]))
            if n_valid < 10:
                continue
            half = n_valid / 2
            # Was in bottom half, now in top half
            if r_prev > half and r_now <= half:
                sig_C[si, di] = True

    print(f"    Rank jump signals: {np.sum(sig_C)}")
    print(f"    ({time.time()-t0:.1f}s)")

    # ================================================================
    # D) GROUP RELATIVE STRENGTH (cross above group avg)
    # ================================================================
    print("\n[Signals] D) Group Relative Strength...", flush=True)
    t0 = time.time()

    sig_D = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        if si not in si_group:
            continue
        for di in range(11, ND):
            own = ROC5[si, di]
            own_prev = ROC5_prev[si, di]
            g_avg = group_avg_roc5[si, di]
            g_avg_prev = np.roll(group_avg_roc5[si], 1)
            if di == 0:
                continue
            g_avg_p = g_avg_prev[di]
            if np.isnan(own) or np.isnan(own_prev) or np.isnan(g_avg) or np.isnan(g_avg_p):
                continue
            # Crossed from below group avg to above
            if own_prev < g_avg_p and own > g_avg:
                sig_D[si, di] = True

    print(f"    Cross above group avg signals: {np.sum(sig_D)}")
    print(f"    ({time.time()-t0:.1f}s)")

    # ================================================================
    # E) PAIR DIVERGENCE (mean-reversion on supply chain spreads)
    # ================================================================
    print("\n[Signals] E) Pair Divergence...", flush=True)
    t0 = time.time()

    # For each pair, compute z-score of spread over 20-day lookback
    sig_E = np.zeros((NS, ND), dtype=bool)
    pair_e_counts = {}

    for upstream_sym, downstream_sym in PAIRS:
        usi = sym_idx.get(upstream_sym)
        dsi = sym_idx.get(downstream_sym)
        if usi is None or dsi is None:
            continue

        pair_count = 0
        for di in range(25, ND):
            # Compute 20-day spread (price ratio)
            spreads = []
            for lookback in range(20):
                dd = di - lookback
                cp_u = C[usi, dd]
                cp_d = C[dsi, dd]
                if np.isnan(cp_u) or np.isnan(cp_d) or cp_d <= 0 or cp_u <= 0:
                    continue
                spreads.append(cp_u / cp_d)

            if len(spreads) < 10:
                continue

            sp_mean = np.mean(spreads)
            sp_std = np.std(spreads)
            if sp_std < 1e-10:
                continue

            cp_u_now = C[usi, di]
            cp_d_now = C[dsi, di]
            if np.isnan(cp_u_now) or np.isnan(cp_d_now) or cp_d_now <= 0 or cp_u_now <= 0:
                continue

            current_spread = cp_u_now / cp_d_now
            z = (current_spread - sp_mean) / sp_std

            # z > 1.5: spread too wide -> upstream overvalued relative to downstream
            #   -> buy downstream (laggard)
            # z < -1.5: spread too narrow -> downstream overvalued relative to upstream
            #   -> buy upstream (leader in this context)
            if z > 1.5:
                sig_E[dsi, di] = True  # buy downstream
                pair_count += 1
            elif z < -1.5:
                sig_E[usi, di] = True  # buy upstream
                pair_count += 1

        pair_e_counts[(upstream_sym, downstream_sym)] = pair_count

    print(f"    Total pair divergence signals: {np.sum(sig_E)}")
    for pair, cnt in sorted(pair_e_counts.items(), key=lambda x: -x[1]):
        print(f"      {pair[0]}->{pair[1]}: {cnt} signals")
    print(f"    ({time.time()-t0:.1f}s)")

    # ================================================================
    # F) GROUP OI FLOW
    # ================================================================
    print("\n[Signals] F) Group OI Flow...", flush=True)
    t0 = time.time()

    # Precompute group total OI change
    group_oi_chg = np.full((NS, ND), np.nan)
    for di in range(10, ND):
        for g, members in group_members.items():
            vals = [OI_chg5[m, di] for m in members if not np.isnan(OI_chg5[m, di])]
            if not vals:
                continue
            g_oi_chg = np.mean(vals)
            for m in members:
                group_oi_chg[m, di] = g_oi_chg

    sig_F = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        if si not in si_group:
            continue
        for di in range(10, ND):
            g_oi = group_oi_chg[si, di]
            own_oi = OI_chg5[si, di]
            own_roc = ROC5[si, di]
            if np.isnan(g_oi) or np.isnan(own_oi) or np.isnan(own_roc):
                continue
            # Group OI increasing + own OI increasing + positive momentum
            if g_oi > 0 and own_oi > 0 and own_roc > 0:
                sig_F[si, di] = True

    print(f"    Group OI flow signals: {np.sum(sig_F)}")
    print(f"    ({time.time()-t0:.1f}s)")

    # ================================================================
    # G) CROSS-GROUP MOMENTUM TRANSFER
    # ================================================================
    print("\n[Signals] G) Cross-Group Momentum Transfer...", flush=True)
    t0 = time.time()

    # Compute group average ROC5
    group_roc5_avg = {}
    for g, members in group_members.items():
        group_roc5_avg[g] = np.zeros(ND)
        for di in range(ND):
            vals = [ROC5[m, di] for m in members if not np.isnan(ROC5[m, di])]
            group_roc5_avg[g][di] = np.mean(vals) if vals else np.nan

    # If ferrous group ROC5 > 0, buy lagging commodities in nonferrous/chemical
    # Generalize: if any group has strong positive ROC5, buy lagging in other groups
    sig_G = np.zeros((NS, ND), dtype=bool)
    lead_groups = ['ferrous', 'energy', 'oils']  # leading sectors
    lag_groups = ['nonferrous', 'chemical', 'soft', 'livestock']  # lagging sectors

    for di in range(10, ND):
        # Check if any lead group is positive
        lead_positive = False
        for lg in lead_groups:
            if lg in group_roc5_avg and not np.isnan(group_roc5_avg[lg][di]):
                if group_roc5_avg[lg][di] > 0:
                    lead_positive = True
                    break

        if not lead_positive:
            continue

        # Buy lagging commodities in lag groups with negative ROC5
        for si in range(NS):
            g = si_group.get(si)
            if g not in lag_groups:
                continue
            own_roc = ROC5[si, di]
            if np.isnan(own_roc):
                continue
            if own_roc < 0:  # lagging behind leading groups
                sig_G[si, di] = True

    print(f"    Cross-group momentum transfer signals: {np.sum(sig_G)}")
    print(f"    ({time.time()-t0:.1f}s)")

    # ================================================================
    # H) MULTI-COMMODITY VOTE
    # ================================================================
    print("\n[Signals] H) Multi-Commodity Vote...", flush=True)
    t0 = time.time()

    sig_H = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        if si not in si_group:
            continue
        g = si_group[si]
        members = group_members[g]
        n_members = len(members)

        for di in range(11, ND):
            own_roc = ROC5[si, di]
            own_roc_prev = ROC5_prev[si, di]
            if np.isnan(own_roc) or np.isnan(own_roc_prev):
                continue

            # Count how many OTHER group members have positive ROC5
            n_positive = 0
            for m in members:
                if m == si:
                    continue
                m_roc = ROC5[m, di]
                if not np.isnan(m_roc) and m_roc > 0:
                    n_positive += 1

            pct_positive = n_positive / max(1, n_members - 1)

            # >= 60% of group positive AND this commodity just crossed to positive
            if pct_positive >= 0.6 and own_roc > 0 and own_roc_prev <= 0:
                sig_H[si, di] = True

    print(f"    Multi-commodity vote signals: {np.sum(sig_H)}")
    print(f"    ({time.time()-t0:.1f}s)")

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(sig_arr, hold_days, top_n, wf_test_year=None):
        """Generic backtest for a signal array."""
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
            # Reset cash at test window start (WF mode)
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                positions = []

            # -- Close positions -----------------------------------------
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

            if len(positions) >= top_n:
                continue

            # -- Generate signals at day di --------------------------------
            entry_di = di + 1
            if entry_di >= end_di:
                continue

            candidates = []
            for si in range(NS):
                if not sig_arr[si, di]:
                    continue
                if any(p['si'] == si for p in positions):
                    continue
                ep = O[si, entry_di]
                if np.isnan(ep) or ep <= 0:
                    continue
                # Score by ROC5 magnitude for ranking
                sc = ROC5[si, di] if not np.isnan(ROC5[si, di]) else 0
                candidates.append((sc, {
                    'si': si, 'sym': syms[si], 'entry_price': ep,
                }))

            if not candidates:
                continue

            # Sort by score descending (highest ROC5 first)
            candidates.sort(key=lambda x: -x[0])

            # Open positions
            n_slots = top_n - len(positions)
            for sc_val, info in candidates[:max(0, n_slots)]:
                si = info['si']
                sym = info['sym']
                price = info['entry_price']
                mult = MULT.get(sym, DEF_MULT)
                contracts = max(1, int(cash / (price * mult)))
                cost_in = price * mult * contracts * (1 + COMM)
                if cost_in > cash:
                    contracts = int(cash * 0.9 / (price * mult * (1 + COMM)))
                    cost_in = price * mult * contracts * (1 + COMM) if contracts > 0 else 0
                if contracts <= 0 or cost_in <= 0 or cost_in > cash:
                    continue

                cash -= cost_in
                positions.append({
                    'si': si, 'entry_price': price, 'entry_di': entry_di,
                    'lots': contracts, 'dir': 1, 'sym': sym,
                    'hold_days': hold_days,
                })

        # Close remaining positions at end
        for pos in positions:
            ae = end_di - 1 if end_di < ND else ND - 1
            exit_price = C[pos['si'], ae]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos['entry_price']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = exit_price * mult * abs(pos['lots'])
            cash += mkt_val - mkt_val * COMM
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

    # A) Group Momentum Lag: threshold x hold x top_n
    for threshold in [0, 0.005, 0.01, 0.02]:
        for hd in [5, 10, 20]:
            for tn in [1, 3]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': 'A', 'threshold': threshold,
                    'hold_days': hd, 'top_n': tn, 'comm': COMM,
                    'label': f"A_GrpMomLag_T{threshold:.3f}_H{hd}_TN{tn}",
                    'sig_arr': sig_A[threshold],
                })

    # B) Supply Chain Lead-Lag: hold x top_n
    for hd in [5, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'B', 'hold_days': hd, 'top_n': tn, 'comm': COMM,
                'label': f"B_SupplyLeadLag_H{hd}_TN{tn}",
                'sig_arr': sig_B,
            })

    # C) Cross-Commodity ROC Rank: hold x top_n
    for hd in [5, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'C', 'hold_days': hd, 'top_n': tn, 'comm': COMM,
                'label': f"C_RankJump_H{hd}_TN{tn}",
                'sig_arr': sig_C,
            })

    # D) Group Relative Strength: hold x top_n
    for hd in [5, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'D', 'hold_days': hd, 'top_n': tn, 'comm': COMM,
                'label': f"D_GrpRelStr_H{hd}_TN{tn}",
                'sig_arr': sig_D,
            })

    # E) Pair Divergence: hold x top_n
    for hd in [3, 5, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'E', 'hold_days': hd, 'top_n': tn, 'comm': COMM,
                'label': f"E_PairDiv_H{hd}_TN{tn}",
                'sig_arr': sig_E,
            })

    # F) Group OI Flow: hold x top_n
    for hd in [5]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'F', 'hold_days': hd, 'top_n': tn, 'comm': COMM,
                'label': f"F_GrpOIFlow_H{hd}_TN{tn}",
                'sig_arr': sig_F,
            })

    # G) Cross-Group Momentum Transfer: hold x top_n
    for hd in [5, 10]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'G', 'hold_days': hd, 'top_n': tn, 'comm': COMM,
                'label': f"G_CrossGrpMom_H{hd}_TN{tn}",
                'sig_arr': sig_G,
            })

    # H) Multi-Commodity Vote: hold x top_n
    for hd in [5]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'H', 'hold_days': hd, 'top_n': tn, 'comm': COMM,
                'label': f"H_MultiVote_H{hd}_TN{tn}",
                'sig_arr': sig_H,
            })

    print(f"  Total configs: {len(configs)}")

    # ================================================================
    # RUN FULL-PERIOD BACKTEST
    # ================================================================
    print("\n[Backtest] Running full-period sweep...", flush=True)
    results = []
    for i, cfg in enumerate(configs):
        r = run_backtest(cfg['sig_arr'], cfg['hold_days'], cfg['top_n'])
        if r:
            r['config'] = cfg
            r['label'] = cfg['label']
            results.append(r)
        if (i + 1) % 5 == 0 or i == len(configs) - 1:
            print(f"  ... {i+1}/{len(configs)} done ({time.time()-t_start:.0f}s)", flush=True)

    results.sort(key=lambda x: -x['ann'])

    # ================================================================
    # FULL-PERIOD RESULTS
    # ================================================================
    print(f"\n{'=' * 200}")
    print("  FULL-PERIOD RESULTS -- CROSS-COMMODITY SIGNALS, NEXT-OPEN EXECUTION")
    print(f"{'=' * 200}")
    print(f"  {'#':>3} | {'Label':<38} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'AvgHold':>7} | {'Freq/Yr':>7} | {'Final':>14}")
    print("-" * 190)
    for i, r in enumerate(results):
        print(f"  {i+1:>3} | {r['label']:<38} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}% | {r['avg_hold']:>6.1f}d | {r['freq']:>6.1f}/yr | {r['final_cash']:>13,.0f}")

    # ================================================================
    # BEST PER SIGNAL TYPE
    # ================================================================
    sig_names = {
        'A': 'A) GROUP MOMENTUM LAG',
        'B': 'B) SUPPLY CHAIN LEAD-LAG',
        'C': 'C) ROC RANK JUMP',
        'D': 'D) GROUP RELATIVE STRENGTH',
        'E': 'E) PAIR DIVERGENCE (mean-rev)',
        'F': 'F) GROUP OI FLOW',
        'G': 'G) CROSS-GROUP MOM TRANSFER',
        'H': 'H) MULTI-COMMODITY VOTE',
    }

    print(f"\n{'=' * 200}")
    print("  BEST PER SIGNAL TYPE (Full Period)")
    print(f"{'=' * 200}")
    print(f"  {'Signal':<40} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'AvgHold':>7} | {'Freq/Yr':>7} | Best Config")
    print("-" * 200)

    best_per_sig = {}
    for r in results:
        key = r['config']['signal']
        if key not in best_per_sig:
            best_per_sig[key] = r

    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
        if sig_key in best_per_sig:
            b = best_per_sig[sig_key]
            print(f"  {sig_names.get(sig_key, sig_key):<40} | {b['ann']:>+8.1f}% | {b['wr']:>5.1f}% | {b['n']:>5} | {b['avg_pnl']:>+6.3f}% | {b['mdd']:>6.1f}% | {b['avg_hold']:>6.1f}d | {b['freq']:>6.1f}/yr | {b['label']}")

    # ================================================================
    # SIGNAL TYPE SUMMARY
    # ================================================================
    print(f"\n{'=' * 200}")
    print("  SIGNAL TYPE SUMMARY (Average of all configs per type)")
    print(f"{'=' * 200}")
    print(f"  {'Signal':<40} | {'Avg Ann':>9} | {'Avg WR':>7} | {'Avg N':>7} | {'Avg PnL':>8} | {'Avg MDD':>8} | {'#Positive':>9}")
    print("-" * 160)

    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
        sub = [r for r in results if r['config']['signal'] == sig_key]
        if not sub:
            continue
        avg_ann = np.mean([r['ann'] for r in sub])
        avg_wr = np.mean([r['wr'] for r in sub])
        avg_n = np.mean([r['n'] for r in sub])
        avg_pnl = np.mean([r['avg_pnl'] for r in sub])
        avg_mdd = np.mean([r['mdd'] for r in sub])
        n_pos = sum(1 for r in sub if r['ann'] > 0)
        print(f"  {sig_names.get(sig_key, sig_key):<40} | {avg_ann:>+8.1f}% | {avg_wr:>6.1f}% | {avg_n:>7.0f} | {avg_pnl:>+7.3f}% | {avg_mdd:>7.1f}% | {n_pos:>5}/{len(sub)}")

    # ================================================================
    # WALK-FORWARD (Top configs + best per signal type)
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Collect top 15 overall + best per signal type
    wf_configs = list(results[:15])
    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
        if sig_key in best_per_sig:
            r = best_per_sig[sig_key]
            if r['config'] not in [w['config'] for w in wf_configs]:
                wf_configs.append(r)

    print(f"\n{'=' * 220}")
    print(f"  WALK-FORWARD ({len(wf_configs)} configs)")
    print(f"{'=' * 220}")

    header = f"  {'#':>3} | {'Config':<38} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>7} |"
    header += f" {'Pos':>4} | {'MDD':>7} | {'WR':>6}"
    print(header)
    print("-" * 220)

    wf_rows = []
    for i, r in enumerate(wf_configs):
        cfg = r['config']
        wf_row = {'label': cfg['label'], 'signal': cfg['signal'],
                  'entry': 'next_open', 'windows': {}, 'mdd': {}, 'wr': {}}
        for yr in wf_years:
            wr = run_backtest(cfg['sig_arr'], cfg['hold_days'], cfg['top_n'], wf_test_year=yr)
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

        row_str = f"  {i+1:>3} | {wf_row['label']:<38} | {avg:>+7.1f}% |"
        for v in vals:
            row_str += f" {v:>+7.1f}% |"
        row_str += f" {pos}/6 | {avg_mdd:>6.1f}% | {avg_wr:>5.1f}%"
        print(row_str)

    # ================================================================
    # WF COMPARISON PER SIGNAL
    # ================================================================
    print(f"\n{'=' * 200}")
    print("  WALK-FORWARD COMPARISON (Best per signal type)")
    print(f"{'=' * 200}")
    header2 = f"  {'Signal':<40} | {'WF Avg':>8} |"
    for yr in wf_years:
        header2 += f" {yr:>7} |"
    header2 += f" {'Pos':>4} | Avg MDD | Avg WR"
    print(header2)
    print("-" * 200)

    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
        wf_match = [w for w in wf_rows if w['signal'] == sig_key]
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            avg = np.mean(vals) if vals else 0
            pos = sum(1 for v in vals if v > 0)
            avg_mdd = np.mean(list(wf['mdd'].values())) if wf['mdd'] else 0
            avg_wr = np.mean(list(wf['wr'].values())) if wf['wr'] else 0
            row_str = f"  {sig_names.get(sig_key, sig_key):<40} | {avg:>+7.1f}% |"
            for v in vals:
                row_str += f" {v:>+7.1f}% |"
            row_str += f" {pos}/6 | {avg_mdd:>6.1f}% | {avg_wr:>5.1f}%"
            print(row_str)

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print(f"\n{'=' * 200}")
    print("  FINAL VERDICT: CROSS-COMMODITY SIGNALS with Next-Open Execution")
    print(f"{'=' * 200}")
    print()
    print("  KEY QUESTIONS:")
    print("  1. Which cross-commodity signals work with next-open execution?")
    print("  2. Any config beating +81.9% (ROC(5) standalone)?")
    print("  3. Supply chain pair trading viability with next-open execution?")
    print()

    beats_best = []
    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
        sub = [r for r in results if r['config']['signal'] == sig_key]
        if not sub:
            continue
        best = sub[0]
        n_pos = sum(1 for r in sub if r['ann'] > 0)

        wf_match = [w for w in wf_rows if w['signal'] == sig_key]
        wf_pos = 0
        wf_avg = 0
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            wf_pos = sum(1 for v in vals if v > 0)
            wf_avg = np.mean(vals) if vals else 0

        verdict = "POSITIVE" if best['ann'] > 0 else "NEGATIVE"
        genuine = "GENUINE ALPHA" if wf_pos >= 4 and best['ann'] > 0 else ("MARGINAL" if wf_pos >= 3 and best['ann'] > 0 else "NO ALPHA")
        beats = "BEATS +81.9%" if best['ann'] > 81.9 else ("CLOSE" if best['ann'] > 50 else "INSUFFICIENT")

        if best['ann'] > 81.9:
            beats_best.append((sig_key, best))

        print(f"  {sig_names.get(sig_key, sig_key)}")
        print(f"    Best annual: {best['ann']:>+8.1f}%  |  {n_pos}/{len(sub)} positive configs")
        print(f"    Walk-forward: {wf_pos}/6 positive  |  WF avg: {wf_avg:>+8.1f}%")
        print(f"    Trade freq: {best['freq']:>5.1f}/yr  |  Avg hold: {best['avg_hold']:>5.1f}d  |  Avg PnL: {best['avg_pnl']:>+6.3f}%")
        print(f"    VERDICT: {verdict}  -->  {genuine}  -->  {beats}")
        print()

    # Absolute best
    if results:
        champ = results[0]
        print(f"  {'='*70}")
        print(f"  CHAMPION: {champ['label']}")
        print(f"    Annual: {champ['ann']:>+8.1f}%  |  WR: {champ['wr']:>5.1f}%  |  N: {champ['n']:>4}  |  MDD: {champ['mdd']:>6.1f}%")
        print(f"    Avg PnL/trade: {champ['avg_pnl']:>+6.3f}%  |  Avg Hold: {champ['avg_hold']:>5.1f}d  |  Freq: {champ['freq']:>5.1f}/yr")
        champ_wf = [w for w in wf_rows if w['label'] == champ['label']]
        if champ_wf:
            cw = champ_wf[0]
            vals = [cw['windows'].get(yr, 0) for yr in wf_years]
            print(f"    WF: {[f'{v:>+7.1f}%' for v in vals]}  |  {sum(1 for v in vals if v > 0)}/6 positive")
        print(f"  {'='*70}")

    # Top 10 summary
    print(f"\n  TOP 10 CONFIGS:")
    print(f"  {'#':>3} | {'Label':<38} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'MDD':>7} | WF_Avg | WF_Pos")
    print("-" * 140)
    for i, r in enumerate(results[:10]):
        wf_match = [w for w in wf_rows if w['label'] == r['label']]
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            wf_avg = np.mean(vals)
            wf_pos = sum(1 for v in vals if v > 0)
        else:
            wf_avg = 0
            wf_pos = 0
        print(f"  {i+1:>3} | {r['label']:<38} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['mdd']:>6.1f}% | {wf_avg:>+7.1f}% | {wf_pos}/6")

    # Signal comparison table
    if beats_best:
        print(f"\n  CONFIGS BEATING +81.9% (ROC(5) standalone):")
        for sig_key, best in beats_best:
            wf_match = [w for w in wf_rows if w['signal'] == sig_key]
            wf_pos = 0
            if wf_match:
                vals = [wf_match[0]['windows'].get(yr, 0) for yr in wf_years]
                wf_pos = sum(1 for v in vals if v > 0)
            print(f"    {sig_names.get(sig_key)}: {best['ann']:>+8.1f}%  |  WF: {wf_pos}/6  |  {best['label']}")
    else:
        print(f"\n  NO config beats +81.9% (ROC(5) standalone)")

    print(f"\n  Total runtime: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
