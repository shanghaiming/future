"""
Alpha Futures V84 — Portfolio Combination: V74 (Group Momentum) + V62 (Pair Trading)
=====================================================================================
V74 champion: +2185% — group momentum lag (44 commodities, 8 groups, LB=1, 1-day hold)
V62 champion: +334%  — LOG-biased adaptive pair trading (14 supply chain pairs, z-score)

These strategies are ORTHOGONAL:
  V74: directional (long only), different instruments, momentum-based signals
  V62: market-neutral (long+short pairs), supply chain pairs, mean-reversion signals

V84 runs both strategies SIMULTANEOUSLY in the same account with different capital
allocations. Tests allocation grid to find optimal risk-adjusted returns.

Allocation grid:
  V74%: [100, 80, 60, 50, 40, 20, 0]
  V62%: [0, 20, 40, 50, 60, 80, 100]

V74 threshold sweep: [0.003, 0.005, 0.01]
Walk-forward: 6 windows (2020-2025)
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

# ── Multipliers ──────────────────────────────────────────────────────
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

# ── V74: Group Map (extended 8 groups, 44 commodities) ──────────────
GROUP_MAP = {}
for s in ['rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi']:
    GROUP_MAP[s] = 'ferrous'
for s in ['cufi', 'alfi', 'znfi', 'nifi', 'pbfi', 'snfi', 'ssfi', 'sffi']:
    GROUP_MAP[s] = 'nonferrous'
for s in ['aufi', 'agfi']:
    GROUP_MAP[s] = 'precious'
for s in ['afi', 'mfi', 'yfi', 'pfi', 'cfi', 'csfi', 'rrfi', 'lrfi']:
    GROUP_MAP[s] = 'oils'
for s in ['scfi', 'mafi', 'bfi', 'fufi', 'pgfi', 'ebfi', 'fbfi']:
    GROUP_MAP[s] = 'energy'
for s in ['ppfi', 'vfi', 'egfi', 'srfi', 'tafi', 'fgfi', 'lfi']:
    GROUP_MAP[s] = 'chemical'
for s in ['whfi', 'apfi', 'cjfi', 'oifi', 'rmfi', 'srfi', 'cffi']:
    GROUP_MAP[s] = 'soft'
for s in ['jdfi', 'lhfi', 'pkfi']:
    GROUP_MAP[s] = 'livestock'

# ── V62: Supply chain pairs (14) ────────────────────────────────────
PAIRS_14 = [
    ('ifi', 'rbfi'),   # iron ore -> rebar
    ('ifi', 'hcfi'),   # iron ore -> hot coil
    ('jfi', 'jmfi'),   # coke -> coking coal
    ('scfi', 'mafi'),  # crude -> methanol
    ('scfi', 'bfi'),   # crude -> asphalt
    ('scfi', 'fufi'),  # crude -> fuel oil
    ('scfi', 'ppfi'),  # crude -> PP
    ('scfi', 'egfi'),  # crude -> EG
    ('scfi', 'vfi'),   # crude -> PVC
    ('scfi', 'pgfi'),  # crude -> LPG
    ('scfi', 'tafi'),  # crude -> PTA
    ('mfi', 'yfi'),    # soybean meal -> soybean oil
    ('mfi', 'afi'),    # soybean meal -> soybean
    ('cfi', 'csfi'),   # corn -> cornstarch
]

SPREAD_LOG = 'log'
ALL_LOOKBACKS_V62 = [15]


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def main():
    t_start = time.time()
    print("=" * 120)
    print("Alpha Futures V84 -- Portfolio: V74 (Group Momentum) + V62 (Pair Trading)")
    print("V74: +2185% directional  |  V62: +334% market-neutral")
    print("Testing allocation grid to find optimal risk-adjusted combination")
    print("=" * 120)

    # ── Load data ────────────────────────────────────────────────────
    print("\n[Data] Loading...", flush=True)
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    sym_to_si = {syms[si]: si for si in range(NS)}
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # Year boundaries for WF
    year_start_di = {}
    year_end_di = {}
    for di in range(ND):
        y = dates[di].year
        if y not in year_start_di:
            year_start_di[y] = di
        year_end_di[y] = di

    # ══════════════════════════════════════════════════════════════════
    # PRECOMPUTE V74 SIGNALS (Group Momentum Lag)
    # ══════════════════════════════════════════════════════════════════
    print("\n[V74] Computing group momentum signals...", flush=True)
    t0 = time.time()

    # LB=1 momentum
    mom1 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            cn = C[si, di]
            cp = C[si, di - 1]
            if not np.isnan(cn) and not np.isnan(cp) and cp > 0:
                mom1[si, di] = (cn - cp) / cp

    # Group membership
    grp_members = {}
    for si in range(NS):
        g = GROUP_MAP.get(syms[si])
        if g:
            grp_members.setdefault(g, []).append(si)
    n_tradeable_v74 = sum(1 for si in range(NS) if GROUP_MAP.get(syms[si]))
    print(f"  Groups: {len(grp_members)}, tradeable commodities: {n_tradeable_v74}")

    # Group momentum (leave-one-out mean) for LB=1
    grp_mom1 = np.full((NS, ND), np.nan)
    for grp, members in grp_members.items():
        for di in range(1, ND):
            for sj in members:
                ms = [mom1[sk, di] for sk in members
                      if sk != sj and not np.isnan(mom1[sk, di])]
                if ms:
                    grp_mom1[sj, di] = np.mean(ms)

    # Divergence = group_mom - own_mom
    div1 = grp_mom1 - mom1
    print(f"  V74 signals ready ({time.time()-t0:.1f}s)")

    # ══════════════════════════════════════════════════════════════════
    # PRECOMPUTE V62 SIGNALS (Pair Z-scores)
    # ══════════════════════════════════════════════════════════════════
    print("\n[V62] Computing pair z-scores...", flush=True)
    t0 = time.time()

    pair_indices = []
    for down_sym, up_sym in PAIRS_14:
        down_si = sym_to_si.get(down_sym, -1)
        up_si = sym_to_si.get(up_sym, -1)
        if down_si >= 0 and up_si >= 0:
            pair_indices.append((down_si, up_si, down_sym, up_sym))
        else:
            print(f"  WARNING: pair ({down_sym}, {up_sym}) not found")
    print(f"  Valid pairs: {len(pair_indices)}")

    # Compute LOG spread z-scores with LB=15
    z_lb = 15
    z_scores = {}
    for down_si, up_si, down_sym, up_sym in pair_indices:
        spread = np.full(ND, np.nan)
        for di in range(ND):
            pd_val = C[down_si, di]
            pu = C[up_si, di]
            if not np.isnan(pd_val) and not np.isnan(pu) and pu > 0 and pd_val > 0:
                spread[di] = np.log(pd_val) - np.log(pu)

        z = np.full(ND, np.nan)
        for di in range(z_lb, ND):
            window = spread[di - z_lb:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= max(3, z_lb * 0.8):
                m_val = np.mean(valid)
                s_val = np.std(valid, ddof=1)
                if s_val > 1e-10:
                    z[di] = (spread[di] - m_val) / s_val
        z_scores[(down_si, up_si)] = z

    print(f"  V62 signals ready ({time.time()-t0:.1f}s)")

    # ══════════════════════════════════════════════════════════════════
    # COMBINED BACKTEST ENGINE
    # ══════════════════════════════════════════════════════════════════
    def run_combined(v74_pct, v62_pct, v74_threshold=0.005,
                     v74_top_n=3, v62_z_thresh=1.0,
                     wf_test_year=None):
        """
        Run V74 and V62 simultaneously in the same account.
        v74_pct / v62_pct: percentage of capital allocated to each strategy.
        Each strategy sees its own allocated capital and trades independently.
        """
        # Date range
        start_di = MIN_TRAIN
        if wf_test_year is not None:
            test_start_di = year_start_di.get(wf_test_year)
            test_end_di = year_end_di.get(wf_test_year)
            if test_start_di is None:
                return None
            end_di = (test_end_di or ND - 1) + 1
        else:
            test_start_di = start_di
            end_di = ND

        cash = float(CASH0)
        v74_positions = []   # V74 momentum positions
        v62_positions = []   # V62 pair positions
        trades = []

        # Track equity for drawdown
        equity_history = []
        peak_equity = float(CASH0)
        max_dd = 0.0

        for di in range(start_di, end_di):
            # Reset cash for WF
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                v74_positions = []
                v62_positions = []
                peak_equity = float(CASH0)
                max_dd = 0.0

            # ── Close V74 positions held >= 1 day ────────────────────
            closed_v74 = []
            for pos in v74_positions:
                if di - pos['entry_di'] >= 1:
                    cn = C[pos['si'], di]
                    if np.isnan(cn) or cn <= 0:
                        cn = pos['entry']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    mkt_val = cn * mult * pos['lots']
                    cash += mkt_val - mkt_val * COMM
                    pnl = (cn - pos['entry']) * mult * pos['lots'] * pos['dir']
                    invested = pos['entry'] * mult * pos['lots']
                    pnl_pct = pnl / invested * 100 if invested > 0 else 0
                    trades.append({
                        'pnl_pct': pnl_pct,
                        'pnl_abs': pnl,
                        'di': pos['entry_di'],
                        'exit_di': di,
                        'year': dates[di].year if di < ND else dates[-1].year,
                        'dir': pos['dir'],
                        'strategy': 'V74',
                        'sym': pos['sym'],
                    })
                    closed_v74.append(pos)
            for pos in closed_v74:
                v74_positions.remove(pos)

            # ── Close V62 positions (1-day hold or z crosses 0) ──────
            closed_v62 = []
            for pos in v62_positions:
                p_down_si = pos['down_si']
                p_up_si = pos['up_si']
                z_arr = z_scores.get((p_down_si, p_up_si))
                z_now = z_arr[di] if z_arr is not None and di < len(z_arr) else np.nan
                days_held = di - pos['entry_di']
                exit_reason = None

                # Mean reversion exit
                if not np.isnan(z_now):
                    if pos['dir'] == 1 and z_now >= 0:
                        exit_reason = 'mean_rev'
                    elif pos['dir'] == -1 and z_now <= 0:
                        exit_reason = 'mean_rev'

                # Time exit
                if exit_reason is None and days_held >= 1:
                    exit_reason = 'time'

                if exit_reason:
                    c_down = C[p_down_si, di]
                    c_up = C[p_up_si, di]
                    if np.isnan(c_down) or c_down <= 0:
                        c_down = pos['entry_down']
                    if np.isnan(c_up) or c_up <= 0:
                        c_up = pos['entry_up']

                    mult_down = MULT.get(pos['down_sym'], DEF_MULT)
                    mult_up = MULT.get(pos['up_sym'], DEF_MULT)
                    lots_down = pos['lots_down']
                    lots_up = pos['lots_up']

                    if pos['dir'] == 1:
                        pnl_down = (c_down - pos['entry_down']) * mult_down * lots_down
                        pnl_up = (pos['entry_up'] - c_up) * mult_up * lots_up
                        cash_return = c_down * mult_down * lots_down - c_up * mult_up * lots_up
                    else:
                        pnl_down = (pos['entry_down'] - c_down) * mult_down * lots_down
                        pnl_up = (c_up - pos['entry_up']) * mult_up * lots_up
                        cash_return = -c_down * mult_down * lots_down + c_up * mult_up * lots_up

                    exit_val_down = c_down * mult_down * lots_down
                    exit_val_up = c_up * mult_up * lots_up
                    cost = (exit_val_down + exit_val_up) * COMM
                    total_pnl = pnl_down + pnl_up - cost
                    invested = pos['entry_down'] * mult_down * lots_down + \
                               pos['entry_up'] * mult_up * lots_up
                    pnl_pct = total_pnl / invested * 100 if invested > 0 else 0

                    cash += pos['cash_invested'] + cash_return - cost
                    trades.append({
                        'pnl_pct': pnl_pct,
                        'pnl_abs': total_pnl,
                        'di': pos['entry_di'],
                        'exit_di': di,
                        'year': dates[di].year if di < ND else dates[-1].year,
                        'dir': pos['dir'],
                        'strategy': 'V62',
                        'pair': (pos['down_sym'], pos['up_sym']),
                        'reason': exit_reason,
                    })
                    closed_v62.append(pos)
            for pos in closed_v62:
                v62_positions.remove(pos)

            # ── Calculate current equity ─────────────────────────────
            equity = cash
            for pos in v74_positions:
                cn = C[pos['si'], di]
                if np.isnan(cn) or cn <= 0:
                    cn = pos['entry']
                mult = MULT.get(pos['sym'], DEF_MULT)
                equity += cn * mult * pos['lots']
            for pos in v62_positions:
                c_down = C[pos['down_si'], di]
                c_up = C[pos['up_si'], di]
                if np.isnan(c_down) or c_down <= 0:
                    c_down = pos['entry_down']
                if np.isnan(c_up) or c_up <= 0:
                    c_up = pos['entry_up']
                mult_down = MULT.get(pos['down_sym'], DEF_MULT)
                mult_up = MULT.get(pos['up_sym'], DEF_MULT)
                if pos['dir'] == 1:
                    equity += c_down * mult_down * pos['lots_down'] - c_up * mult_up * pos['lots_up']
                else:
                    equity += -c_down * mult_down * pos['lots_down'] + c_up * mult_up * pos['lots_up']

            if equity > peak_equity:
                peak_equity = equity
            if peak_equity > 0:
                dd = (peak_equity - equity) / peak_equity * 100
                if dd > max_dd:
                    max_dd = dd
            equity_history.append(equity)

            # ── V74: Open new momentum positions ─────────────────────
            v74_cash = cash * v74_pct / 100.0
            if v74_pct > 0:
                candidates_v74 = []
                for si in range(NS):
                    sym = syms[si]
                    if GROUP_MAP.get(sym) is None:
                        continue
                    if np.isnan(C[si, di]) or C[si, di] <= 0:
                        continue
                    if any(p['si'] == si for p in v74_positions):
                        continue
                    d = div1[si, di]
                    if np.isnan(d):
                        continue
                    if d > v74_threshold:
                        candidates_v74.append((si, d))

                candidates_v74.sort(key=lambda x: -x[1])
                n_slots = v74_top_n - len(v74_positions)
                for si, score in candidates_v74[:n_slots]:
                    c = C[si, di]
                    if np.isnan(c) or c <= 0:
                        continue
                    mult = MULT.get(syms[si], DEF_MULT)
                    notional = c * mult
                    lots = int(v74_cash / (notional * (1 + COMM)))
                    if lots <= 0:
                        continue
                    cost_in = notional * lots * (1 + COMM)
                    if cost_in > v74_cash:
                        lots = int(v74_cash * 0.95 / (notional * (1 + COMM)))
                        cost_in = notional * lots * (1 + COMM) if lots > 0 else 0
                    if lots <= 0 or cost_in <= 0 or cost_in > v74_cash:
                        continue

                    cash -= cost_in
                    v74_cash -= cost_in
                    v74_positions.append({
                        'si': si, 'entry': c, 'entry_di': di,
                        'lots': lots, 'dir': 1, 'sym': syms[si],
                    })

            # ── V62: Open new pair positions ─────────────────────────
            v62_cash = cash * v62_pct / 100.0
            if v62_pct > 0:
                # Check occupied commodities
                occupied = set()
                for pos in v62_positions:
                    occupied.add(pos['down_si'])
                    occupied.add(pos['up_si'])

                # Only 1 pair at a time for V62
                if len(v62_positions) < 1:
                    pair_cands = []
                    for down_si, up_si, down_sym, up_sym in pair_indices:
                        if down_si in occupied or up_si in occupied:
                            continue
                        z_arr = z_scores.get((down_si, up_si))
                        if z_arr is None:
                            continue
                        z_val = z_arr[di] if di < len(z_arr) else np.nan
                        if np.isnan(z_val):
                            continue
                        if abs(z_val) < v62_z_thresh:
                            continue
                        pair_cands.append((abs(z_val), down_si, up_si,
                                          down_sym, up_sym, z_val))

                    if pair_cands:
                        pair_cands.sort(key=lambda x: -x[0])
                        _, down_si, up_si, down_sym, up_sym, z_val = pair_cands[0]

                        c_down = C[down_si, di]
                        c_up = C[up_si, di]
                        if (not np.isnan(c_down) and c_down > 0 and
                            not np.isnan(c_up) and c_up > 0):

                            mult_down = MULT.get(down_sym, DEF_MULT)
                            mult_up = MULT.get(up_sym, DEF_MULT)

                            cash_per_leg = v62_cash / 2
                            lots_down = int(cash_per_leg / (c_down * mult_down * (1 + COMM)))
                            lots_up = int(cash_per_leg / (c_up * mult_up * (1 + COMM)))
                            if lots_down > 0 and lots_up > 0:
                                cost_down = c_down * mult_down * lots_down * (1 + COMM)
                                cost_up = c_up * mult_up * lots_up * (1 + COMM)
                                total_cost = cost_down + cost_up
                                if total_cost > v62_cash:
                                    scale = v62_cash * 0.95 / total_cost
                                    lots_down = max(1, int(lots_down * scale))
                                    lots_up = max(1, int(lots_up * scale))
                                    cost_down = c_down * mult_down * lots_down * (1 + COMM)
                                    cost_up = c_up * mult_up * lots_up * (1 + COMM)
                                    total_cost = cost_down + cost_up
                                    if total_cost > v62_cash:
                                        continue

                                if z_val > 0:
                                    pos_dir = -1  # short downstream, long upstream
                                else:
                                    pos_dir = 1   # long downstream, short upstream

                                cash -= total_cost
                                v62_cash -= total_cost
                                v62_positions.append({
                                    'down_si': down_si, 'up_si': up_si,
                                    'down_sym': down_sym, 'up_sym': up_sym,
                                    'entry_down': c_down, 'entry_up': c_up,
                                    'lots_down': lots_down, 'lots_up': lots_up,
                                    'entry_di': di, 'dir': pos_dir,
                                    'cash_invested': total_cost,
                                })

        # ── Close remaining positions at end ─────────────────────────
        actual_end = min(end_di, ND) - 1
        for pos in v74_positions:
            cn = C[pos['si'], actual_end]
            if np.isnan(cn) or cn <= 0:
                cn = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = cn * mult * pos['lots']
            cash += mkt_val - mkt_val * COMM

        for pos in v62_positions:
            c_down = C[pos['down_si'], actual_end]
            c_up = C[pos['up_si'], actual_end]
            if np.isnan(c_down) or c_down <= 0:
                c_down = pos['entry_down']
            if np.isnan(c_up) or c_up <= 0:
                c_up = pos['entry_up']

            mult_down = MULT.get(pos['down_sym'], DEF_MULT)
            mult_up = MULT.get(pos['up_sym'], DEF_MULT)
            if pos['dir'] == 1:
                cash_return = c_down * mult_down * pos['lots_down'] - c_up * mult_up * pos['lots_up']
            else:
                cash_return = -c_down * mult_down * pos['lots_down'] + c_up * mult_up * pos['lots_up']
            exit_val_down = c_down * mult_down * pos['lots_down']
            exit_val_up = c_up * mult_up * pos['lots_up']
            cost = (exit_val_down + exit_val_up) * COMM
            cash += pos['cash_invested'] + cash_return - cost

        # ── Calculate statistics ─────────────────────────────────────
        if wf_test_year is not None:
            n_days_test = (test_end_di or ND-1) - test_start_di + 1
        else:
            n_days_test = ND - MIN_TRAIN

        ann = annual_return(cash, CASH0, n_days_test)

        if not trades:
            return None

        wins = [t for t in trades if t['pnl_abs'] > 0]
        losses = [t for t in trades if t['pnl_abs'] <= 0]
        wr = len(wins) / len(trades) * 100 if trades else 0

        total_win = sum(t['pnl_abs'] for t in wins) if wins else 0
        total_loss = abs(sum(t['pnl_abs'] for t in losses)) if losses else 1
        pf = total_win / total_loss if total_loss > 0 else 0

        # Sharpe approximation from equity curve
        if len(equity_history) > 2:
            eq_arr = np.array(equity_history)
            rets = np.diff(eq_arr) / eq_arr[:-1]
            rets = rets[~np.isnan(rets) & np.isfinite(rets)]
            if len(rets) > 1 and np.std(rets) > 0:
                sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252)
            else:
                sharpe = 0
        else:
            sharpe = 0

        # Per-strategy stats
        v74_trades = [t for t in trades if t['strategy'] == 'V74']
        v62_trades = [t for t in trades if t['strategy'] == 'V62']
        v74_wr = (sum(1 for t in v74_trades if t['pnl_abs'] > 0) / max(len(v74_trades), 1)) * 100
        v62_wr = (sum(1 for t in v62_trades if t['pnl_abs'] > 0) / max(len(v62_trades), 1)) * 100
        v74_pnl = sum(t['pnl_abs'] for t in v74_trades)
        v62_pnl = sum(t['pnl_abs'] for t in v62_trades)

        # Yearly breakdown
        year_stats = {}
        for t in trades:
            y = t['year']
            if y not in year_stats:
                year_stats[y] = {'n': 0, 'w': 0, 'pnl_abs': 0.0, 'pnl_pct': []}
            year_stats[y]['n'] += 1
            if t['pnl_abs'] > 0:
                year_stats[y]['w'] += 1
            year_stats[y]['pnl_abs'] += t['pnl_abs']
            year_stats[y]['pnl_pct'].append(t['pnl_pct'])

        return {
            'ann': ann,
            'wr': wr,
            'n': len(trades),
            'dd': max_dd,
            'pf': pf,
            'sharpe': sharpe,
            'final_cash': cash,
            'n_days': n_days_test,
            'v74_trades': len(v74_trades),
            'v62_trades': len(v62_trades),
            'v74_wr': v74_wr,
            'v62_wr': v62_wr,
            'v74_pnl': v74_pnl,
            'v62_pnl': v62_pnl,
            'yearly': year_stats,
        }

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: ALLOCATION SWEEP (full period)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("  PHASE 1: FULL-PERIOD ALLOCATION SWEEP")
    print("=" * 120)

    allocations = [
        (100, 0), (80, 20), (60, 40), (50, 50), (40, 60), (20, 80), (0, 100),
    ]
    thresholds = [0.003, 0.005, 0.01]

    full_results = []
    for v74_pct, v62_pct in allocations:
        for thresh in thresholds:
            label = f"V74={v74_pct:3d}%_V62={v62_pct:3d}%_T{thresh}"
            r = run_combined(v74_pct, v62_pct, v74_threshold=thresh)
            if r:
                r['label'] = label
                r['v74_pct'] = v74_pct
                r['v62_pct'] = v62_pct
                r['threshold'] = thresh
                full_results.append(r)
                print(f"  {label:30s}  Ann={r['ann']:+8.1f}%  WR={r['wr']:5.1f}%  "
                      f"DD={r['dd']:5.1f}%  PF={r['pf']:5.2f}  Sharpe={r['sharpe']:6.2f}  "
                      f"N={r['n']:5d}  Cash={r['final_cash']:12.0f}")

    # Sort by annual return
    full_results.sort(key=lambda x: -x['ann'])

    # Print top results
    print("\n" + "=" * 120)
    print("  FULL-PERIOD TOP RESULTS (sorted by annual return)")
    print("=" * 120)
    print(f"  {'#':>3} | {'Allocation':30s} | {'Ann':>9} | {'WR':>6} | {'N':>5} | "
          f"{'DD':>6} | {'PF':>5} | {'Sharpe':>7} | {'V74 N':>6} | {'V62 N':>6} | "
          f"{'V74 PnL':>12} | {'V62 PnL':>12} | {'Cash':>14}")
    print("-" * 145)
    for i, r in enumerate(full_results):
        print(f"  {i+1:>3} | {r['label']:30s} | {r['ann']:+8.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:5d} | {r['dd']:5.1f}% | {r['pf']:5.2f} | {r['sharpe']:6.2f} | "
              f"{r['v74_trades']:6d} | {r['v62_trades']:6d} | "
              f"{r['v74_pnl']:+11.0f} | {r['v62_pnl']:+11.0f} | {r['final_cash']:13.0f}")

    # ── Best by Sharpe ───────────────────────────────────────────────
    full_by_sharpe = sorted(full_results, key=lambda x: -x['sharpe'])
    print("\n" + "=" * 120)
    print("  FULL-PERIOD TOP RESULTS (sorted by Sharpe ratio)")
    print("=" * 120)
    print(f"  {'#':>3} | {'Allocation':30s} | {'Sharpe':>7} | {'Ann':>9} | {'WR':>6} | "
          f"{'DD':>6} | {'PF':>5} | {'N':>5} | {'Cash':>14}")
    print("-" * 105)
    for i, r in enumerate(full_by_sharpe[:15]):
        print(f"  {i+1:>3} | {r['label']:30s} | {r['sharpe']:6.2f} | {r['ann']:+8.1f}% | "
              f"{r['wr']:5.1f}% | {r['dd']:5.1f}% | {r['pf']:5.2f} | {r['n']:5d} | "
              f"{r['final_cash']:13.0f}")

    # ── Best by PF ───────────────────────────────────────────────────
    full_by_pf = sorted(full_results, key=lambda x: -x['pf'])
    print("\n" + "=" * 120)
    print("  FULL-PERIOD TOP RESULTS (sorted by Profit Factor)")
    print("=" * 120)
    print(f"  {'#':>3} | {'Allocation':30s} | {'PF':>5} | {'Ann':>9} | {'WR':>6} | "
          f"{'DD':>6} | {'Sharpe':>7} | {'N':>5} | {'Cash':>14}")
    print("-" * 100)
    for i, r in enumerate(full_by_pf[:15]):
        print(f"  {i+1:>3} | {r['label']:30s} | {r['pf']:5.2f} | {r['ann']:+8.1f}% | "
              f"{r['wr']:5.1f}% | {r['dd']:5.1f}% | {r['sharpe']:6.2f} | {r['n']:5d} | "
              f"{r['final_cash']:13.0f}")

    # ── Best by lowest DD ────────────────────────────────────────────
    full_by_dd = sorted(full_results, key=lambda x: x['dd'])
    print("\n" + "=" * 120)
    print("  FULL-PERIOD TOP RESULTS (sorted by lowest Drawdown)")
    print("=" * 120)
    print(f"  {'#':>3} | {'Allocation':30s} | {'DD':>6} | {'Ann':>9} | {'WR':>6} | "
          f"{'PF':>5} | {'Sharpe':>7} | {'N':>5} | {'Cash':>14}")
    print("-" * 100)
    for i, r in enumerate(full_by_dd[:15]):
        print(f"  {i+1:>3} | {r['label']:30s} | {r['dd']:5.1f}% | {r['ann']:+8.1f}% | "
              f"{r['wr']:5.1f}% | {r['pf']:5.2f} | {r['sharpe']:6.2f} | {r['n']:5d} | "
              f"{r['final_cash']:13.0f}")

    # ── Allocation comparison at best threshold ──────────────────────
    print("\n" + "=" * 120)
    print("  ALLOCATION COMPARISON (threshold=0.005)")
    print("=" * 120)
    t005 = [r for r in full_results if r['threshold'] == 0.005]
    print(f"  {'V74%':>5} | {'V62%':>5} | {'Ann':>9} | {'WR':>6} | {'N':>5} | "
          f"{'DD':>6} | {'PF':>5} | {'Sharpe':>7} | {'V74 N':>6} | {'V62 N':>6} | "
          f"{'V74 PnL':>12} | {'V62 PnL':>12}")
    print("-" * 120)
    for r in sorted(t005, key=lambda x: -x['ann']):
        print(f"  {r['v74_pct']:5d} | {r['v62_pct']:5d} | {r['ann']:+8.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:5d} | {r['dd']:5.1f}% | {r['pf']:5.2f} | {r['sharpe']:6.2f} | "
              f"{r['v74_trades']:6d} | {r['v62_trades']:6d} | "
              f"{r['v74_pnl']:+11.0f} | {r['v62_pnl']:+11.0f}")

    # ── Year-by-year for best config ─────────────────────────────────
    if full_results:
        best = full_results[0]
        print("\n" + "=" * 120)
        print(f"  YEAR-BY-YEAR for #1 Config: {best['label']}")
        print(f"  Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  DD={best['dd']:.1f}%  "
              f"PF={best['pf']:.2f}  Sharpe={best['sharpe']:.2f}")
        print("=" * 120)
        print(f"  {'Year':>6} | {'N':>5} | {'WR':>5} | {'PnL Abs':>12} | {'Avg PnL%':>10}")
        print("-" * 60)
        for y in sorted(best['yearly'].keys()):
            ys = best['yearly'][y]
            wr_y = ys['w'] / max(ys['n'], 1) * 100
            avg_pct = np.mean(ys['pnl_pct']) if ys['pnl_pct'] else 0
            print(f"  {y:6d} | {ys['n']:5d} | {wr_y:4.1f}% | {ys['pnl_abs']:+11.0f} | "
                  f"{avg_pct:+9.2f}%")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: WALK-FORWARD FOR TOP CONFIGS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("  PHASE 2: WALK-FORWARD VALIDATION (Top configs)")
    print("=" * 120)

    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Take top configs from different categories for WF
    wf_configs = []
    # Top 3 by annual return
    for r in full_results[:3]:
        wf_configs.append(r)
    # Best Sharpe not already in list
    for r in full_by_sharpe:
        if r['label'] not in [x['label'] for x in wf_configs]:
            wf_configs.append(r)
            break
    # Best PF not already in list
    for r in full_by_pf:
        if r['label'] not in [x['label'] for x in wf_configs]:
            wf_configs.append(r)
            break
    # Lowest DD not already in list
    for r in full_by_dd:
        if r['label'] not in [x['label'] for x in wf_configs]:
            wf_configs.append(r)
            break
    # 80/20 and 50/50 at T=0.005 if not already in
    for alloc in [(80, 20), (50, 50)]:
        match = [r for r in t005 if r['v74_pct'] == alloc[0] and r['v62_pct'] == alloc[1]]
        if match and match[0]['label'] not in [x['label'] for x in wf_configs]:
            wf_configs.append(match[0])

    print(f"  WF configs to test: {len(wf_configs)}")
    for r in wf_configs:
        print(f"    {r['label']}")

    wf_results = []
    for cfg in wf_configs:
        label = cfg['label']
        wf_row = {
            'label': label,
            'v74_pct': cfg['v74_pct'],
            'v62_pct': cfg['v62_pct'],
            'threshold': cfg['threshold'],
            'full_ann': cfg['ann'],
            'full_sharpe': cfg['sharpe'],
            'windows': {},
        }
        for yr in wf_years:
            r = run_combined(cfg['v74_pct'], cfg['v62_pct'],
                             v74_threshold=cfg['threshold'],
                             wf_test_year=yr)
            if r:
                wf_row['windows'][yr] = r
        wf_results.append(wf_row)

    # Print WF table
    print(f"\n  {'#':>2} | {'Config':30s} | {'Full':>9} | {'WF Avg':>9} | {'WF Med':>9} | "
          f"{'WF Min':>9} | {'WF Max':>9} | {'Pos':>4} |", end="")
    for yr in wf_years:
        print(f"  {yr:>7} |", end="")
    print()
    print("-" * 150)

    for i, wf in enumerate(wf_results):
        anns = [wf['windows'][yr]['ann'] for yr in wf_years if yr in wf['windows']]
        if not anns:
            continue
        avg_ann = np.mean(anns)
        med_ann = np.median(anns)
        min_ann = min(anns)
        max_ann = max(anns)
        n_pos = sum(1 for a in anns if a > 0)

        print(f"  {i+1:>2} | {wf['label']:30s} | {wf['full_ann']:+8.1f}% | {avg_ann:+8.1f}% | "
              f"{med_ann:+8.1f}% | {min_ann:+8.1f}% | {max_ann:+8.1f}% | {n_pos:>2}/6 |", end="")
        for yr in wf_years:
            if yr in wf['windows']:
                print(f"  {wf['windows'][yr]['ann']:+7.1f}% |", end="")
            else:
                print(f"  {'N/A':>7} |", end="")
        print()

    # ── WF Detail table with DD and Sharpe ───────────────────────────
    print("\n" + "=" * 120)
    print("  WF DETAIL: Annual Return / Max DD / Sharpe per window")
    print("=" * 120)
    for i, wf in enumerate(wf_results):
        print(f"\n  [{i+1}] {wf['label']}  (full-period Ann={wf['full_ann']:+.1f}%)")
        print(f"  {'Year':>6} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'DD':>6} | "
              f"{'PF':>5} | {'Sharpe':>7} | {'V74 N':>6} | {'V62 N':>6} | "
              f"{'V74 PnL':>12} | {'V62 PnL':>12}")
        print(f"  {'-' * 120}")
        for yr in wf_years:
            if yr in wf['windows']:
                r = wf['windows'][yr]
                print(f"  {yr:6d} | {r['ann']:+8.1f}% | {r['wr']:5.1f}% | "
                      f"{r['n']:5d} | {r['dd']:5.1f}% | {r['pf']:5.2f} | "
                      f"{r['sharpe']:6.2f} | {r['v74_trades']:6d} | {r['v62_trades']:6d} | "
                      f"{r['v74_pnl']:+11.0f} | {r['v62_pnl']:+11.0f}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("  ANALYSIS: DOES ADDING V62 IMPROVE RISK-ADJUSTED RETURNS?")
    print("=" * 120)

    # Compare pure V74 vs mixed allocations
    v74_pure = [r for r in full_results if r['v74_pct'] == 100]
    v62_pure = [r for r in full_results if r['v62_pct'] == 100]
    mixed = [r for r in full_results if r['v74_pct'] > 0 and r['v62_pct'] > 0]

    if v74_pure:
        best_v74 = max(v74_pure, key=lambda x: x['ann'])
        print(f"\n  Best pure V74: {best_v74['label']}")
        print(f"    Ann={best_v74['ann']:+.1f}%  Sharpe={best_v74['sharpe']:.2f}  "
              f"DD={best_v74['dd']:.1f}%  PF={best_v74['pf']:.2f}  WR={best_v74['wr']:.1f}%")

    if v62_pure:
        best_v62 = max(v62_pure, key=lambda x: x['ann'])
        print(f"\n  Best pure V62: {best_v62['label']}")
        print(f"    Ann={best_v62['ann']:+.1f}%  Sharpe={best_v62['sharpe']:.2f}  "
              f"DD={best_v62['dd']:.1f}%  PF={best_v62['pf']:.2f}  WR={best_v62['wr']:.1f}%")

    if mixed:
        best_mixed_ann = max(mixed, key=lambda x: x['ann'])
        best_mixed_sharpe = max(mixed, key=lambda x: x['sharpe'])
        best_mixed_dd = min(mixed, key=lambda x: x['dd'])
        print(f"\n  Best mixed (by Ann): {best_mixed_ann['label']}")
        print(f"    Ann={best_mixed_ann['ann']:+.1f}%  Sharpe={best_mixed_ann['sharpe']:.2f}  "
              f"DD={best_mixed_ann['dd']:.1f}%  PF={best_mixed_ann['pf']:.2f}  WR={best_mixed_ann['wr']:.1f}%")
        print(f"\n  Best mixed (by Sharpe): {best_mixed_sharpe['label']}")
        print(f"    Ann={best_mixed_sharpe['ann']:+.1f}%  Sharpe={best_mixed_sharpe['sharpe']:.2f}  "
              f"DD={best_mixed_sharpe['dd']:.1f}%  PF={best_mixed_sharpe['pf']:.2f}  "
              f"WR={best_mixed_sharpe['wr']:.1f}%")
        print(f"\n  Best mixed (by lowest DD): {best_mixed_dd['label']}")
        print(f"    Ann={best_mixed_dd['ann']:+.1f}%  Sharpe={best_mixed_dd['sharpe']:.2f}  "
              f"DD={best_mixed_dd['dd']:.1f}%  PF={best_mixed_dd['pf']:.2f}  "
              f"WR={best_mixed_dd['wr']:.1f}%")

    # Sharpe vs Return tradeoff
    print("\n" + "=" * 120)
    print("  SHARPE vs RETURN TRADEOFF (all configs)")
    print("=" * 120)
    print(f"  {'Allocation':30s} | {'Ann':>9} | {'Sharpe':>7} | {'DD':>6} | {'PF':>5} | "
          f"{'Ann/DD':>8}")
    print("-" * 80)
    for r in sorted(full_results, key=lambda x: -x['sharpe']):
        ann_dd = r['ann'] / max(r['dd'], 0.1)
        print(f"  {r['label']:30s} | {r['ann']:+8.1f}% | {r['sharpe']:6.2f} | "
              f"{r['dd']:5.1f}% | {r['pf']:5.2f} | {ann_dd:+7.1f}")

    # WF comparison
    print("\n" + "=" * 120)
    print("  WALK-FORWARD COMPARISON: V74 Pure vs Best Mixed vs V62 Pure")
    print("=" * 120)

    wf_compare = {}
    for wf in wf_results:
        anns = [wf['windows'][yr]['ann'] for yr in wf_years if yr in wf['windows']]
        dds = [wf['windows'][yr]['dd'] for yr in wf_years if yr in wf['windows']]
        sharpe_vals = [wf['windows'][yr]['sharpe'] for yr in wf_years if yr in wf['windows']]
        pfs = [wf['windows'][yr]['pf'] for yr in wf_years if yr in wf['windows']]
        if anns:
            wf_compare[wf['label']] = {
                'label': wf['label'],
                'v74_pct': wf['v74_pct'],
                'v62_pct': wf['v62_pct'],
                'avg_ann': np.mean(anns),
                'min_ann': min(anns),
                'max_ann': max(anns),
                'avg_dd': np.mean(dds),
                'max_dd': max(dds),
                'avg_sharpe': np.mean(sharpe_vals),
                'avg_pf': np.mean(pfs),
                'n_pos': sum(1 for a in anns if a > 0),
            }

    for label, wc in sorted(wf_compare.items(), key=lambda x: -x[1]['avg_ann']):
        print(f"  {label:30s}  WF_Avg={wc['avg_ann']:+7.1f}%  "
              f"WF_Min={wc['min_ann']:+7.1f}%  WF_Max={wc['max_ann']:+7.1f}%  "
              f"AvgDD={wc['avg_dd']:5.1f}%  AvgSh={wc['avg_sharpe']:5.2f}  "
              f"AvgPF={wc['avg_pf']:4.2f}  Pos={wc['n_pos']}/6")

    # ══════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("  FINAL SUMMARY")
    print("=" * 120)

    if v74_pure and mixed:
        best_v74 = max(v74_pure, key=lambda x: x['ann'])
        print(f"\n  V74 alone (best):  Ann={best_v74['ann']:+.1f}%  "
              f"Sharpe={best_v74['sharpe']:.2f}  DD={best_v74['dd']:.1f}%  PF={best_v74['pf']:.2f}")

    if v62_pure:
        best_v62 = max(v62_pure, key=lambda x: x['ann'])
        print(f"  V62 alone (best):  Ann={best_v62['ann']:+.1f}%  "
              f"Sharpe={best_v62['sharpe']:.2f}  DD={best_v62['dd']:.1f}%  PF={best_v62['pf']:.2f}")

    if mixed:
        best_mix_s = max(mixed, key=lambda x: x['sharpe'])
        best_mix_a = max(mixed, key=lambda x: x['ann'])
        print(f"\n  Best mixed (Ann):    {best_mix_a['label']}")
        print(f"    Ann={best_mix_a['ann']:+.1f}%  Sharpe={best_mix_a['sharpe']:.2f}  "
              f"DD={best_mix_a['dd']:.1f}%  PF={best_mix_a['pf']:.2f}")
        print(f"\n  Best mixed (Sharpe): {best_mix_s['label']}")
        print(f"    Ann={best_mix_s['ann']:+.1f}%  Sharpe={best_mix_s['sharpe']:.2f}  "
              f"DD={best_mix_s['dd']:.1f}%  PF={best_mix_s['pf']:.2f}")

    # Key conclusions
    print(f"\n  KEY QUESTIONS ANSWERED:")
    if v74_pure and mixed:
        best_v74_s = max(v74_pure, key=lambda x: x['sharpe'])
        best_mix_s = max(mixed, key=lambda x: x['sharpe'])
        if best_mix_s['sharpe'] > best_v74_s['sharpe']:
            print(f"  1. Adding V62 IMPROVES risk-adjusted returns:")
            print(f"     V74 Sharpe={best_v74_s['sharpe']:.2f} -> Mixed Sharpe={best_mix_s['sharpe']:.2f} "
                  f"(+{best_mix_s['sharpe']-best_v74_s['sharpe']:.2f})")
        else:
            print(f"  1. Adding V62 DOES NOT improve risk-adjusted returns:")
            print(f"     V74 Sharpe={best_v74_s['sharpe']:.2f} > Mixed Sharpe={best_mix_s['sharpe']:.2f}")

        best_mix_dd = min(mixed, key=lambda x: x['dd'])
        best_v74_dd = min(v74_pure, key=lambda x: x['dd'])
        if best_mix_dd['dd'] < best_v74_dd['dd']:
            print(f"  2. Adding V62 REDUCES drawdown:")
            print(f"     V74 DD={best_v74_dd['dd']:.1f}% -> Mixed DD={best_mix_dd['dd']:.1f}% "
                  f"({best_mix_dd['dd']-best_v74_dd['dd']:+.1f}%)")
        else:
            print(f"  2. Adding V62 DOES NOT reduce drawdown:")
            print(f"     V74 DD={best_v74_dd['dd']:.1f}% <= Mixed DD={best_mix_dd['dd']:.1f}%")

    # WF champion
    if wf_compare:
        wf_champ = max(wf_compare.values(), key=lambda x: x['avg_ann'])
        print(f"\n  WALK-FORWARD CHAMPION: {wf_champ['label']}")
        print(f"    WF Avg Ann={wf_champ['avg_ann']:+.1f}%  WF Min={wf_champ['min_ann']:+.1f}%  "
              f"Avg DD={wf_champ['avg_dd']:.1f}%  Avg Sharpe={wf_champ['avg_sharpe']:.2f}  "
              f"Positive={wf_champ['n_pos']}/6")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
