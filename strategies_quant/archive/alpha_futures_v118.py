"""
Alpha Futures V118 — EXTREME MOMENTUM: Push Returns to Maximum
==============================================================
V118 FOCUS: Try every extreme angle to break past +89.8%.

10 test configurations (A-J):
A) Triple momentum confirmation (ROC3>3% + ROC5>2% + ROC10>0)
B) Extreme ROC (ROC5 > 5%)
C) Triple extreme: ROC5>2% + Z-score>2.5 + ADX>30
D) All-in with trailing stop + pyramiding
E) Daily re-evaluation (rolling positions)
F) Commodity rotation (rank jump)
G) Aggressive compounding
H) Leverage simulation (informational)
I) Top 1 per group (diversified concentration)
J) Signal aggregation with weighting

ALL signals use NEXT-OPEN execution: signal at close di, entry at O[si, di+1].
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

GROUP_MAP = {}
for _s in ['rbfi', 'hcfi', 'ifi', 'jfi', 'jmfi', 'wrffi']: GROUP_MAP[_s] = 'ferrous'
for _s in ['cufi', 'alfi', 'znfi', 'nifi', 'pbfi', 'snfi', 'ssfi', 'sffi']: GROUP_MAP[_s] = 'nonferrous'
for _s in ['aufi', 'agfi']: GROUP_MAP[_s] = 'precious'
for _s in ['afi', 'mfi', 'yfi', 'pfi', 'cfi', 'csfi', 'rrfi', 'lrfi', 'rsfi']: GROUP_MAP[_s] = 'oils'
for _s in ['scfi', 'mafi', 'bfi', 'fufi', 'pgfi', 'ebfi', 'fbfi', 'lufi', 'urfi', 'safi', 'lgfi']: GROUP_MAP[_s] = 'energy'
for _s in ['ppfi', 'vfi', 'egfi', 'tafi', 'fgfi', 'lfi', 'ebfi', 'bbfi', 'pfifi', 'brfi', 'sifi', 'bcfi', 'cyfi']: GROUP_MAP[_s] = 'chemical'
for _s in ['whfi', 'apfi', 'cjfi', 'oifi', 'rmfi', 'cffi', 'srfi', 'jrfi', 'pmfi']: GROUP_MAP[_s] = 'soft'
for _s in ['jdfi', 'lhfi', 'pkfi', 'nrfi', 'lcfi']: GROUP_MAP[_s] = 'livestock'
for _s in ['bufi', 'cufi', 'spfi', 'smfi', 'rufi', 'ni', 'tai']: GROUP_MAP[_s] = 'other'


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def main():
    print("=" * 220)
    print("  Alpha Futures V118 — EXTREME MOMENTUM: Push Returns to Maximum")
    print("=" * 220)
    print("\n  10 extreme approaches (A-J), walk-forward 2020-2025")
    print("  ALL signals at close di, entry at O[si, di+1] (NEXT DAY OPEN)")

    # -- Load data -------------------------------------------------
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # Build symbol index map
    sym_idx = {s: i for i, s in enumerate(syms)}

    # Build group membership arrays
    group_members = {}
    si_group = {}
    for si in range(NS):
        g = GROUP_MAP.get(syms[si], 'other')
        group_members.setdefault(g, []).append(si)
        si_group[si] = g

    print(f"  Groups: {list(group_members.keys())}")
    for g, members in sorted(group_members.items()):
        syms_in_g = [syms[si] for si in members]
        print(f"    {g}: {len(members)} commodities -> {syms_in_g}")

    # ================================================================
    # PRECOMPUTE ALL INDICATORS
    # ================================================================
    print("\n[Precompute] ROC, Z-score, ADX, ATR, OI change...", flush=True)
    t0 = time.time()

    ROC3  = np.full((NS, ND), np.nan)
    ROC5  = np.full((NS, ND), np.nan)
    ROC10 = np.full((NS, ND), np.nan)
    ROC20 = np.full((NS, ND), np.nan)
    ADX14 = np.full((NS, ND), np.nan)
    ATR14 = np.full((NS, ND), np.nan)
    ZSCORE20 = np.full((NS, ND), np.nan)
    OI_CHG5  = np.full((NS, ND), np.nan)

    for si in range(NS):
        c = C[si].astype(np.float64)
        h = H[si].astype(np.float64)
        l = L[si].astype(np.float64)
        oi = OI[si].astype(np.float64)

        ROC3[si]  = talib.ROC(c, timeperiod=3)
        ROC5[si]  = talib.ROC(c, timeperiod=5)
        ROC10[si] = talib.ROC(c, timeperiod=10)
        ROC20[si] = talib.ROC(c, timeperiod=20)
        ADX14[si] = talib.ADX(h, l, c, timeperiod=14)
        ATR14[si] = talib.ATR(h, l, c, timeperiod=14)

        # 20-day rolling z-score of close
        for di in range(20, ND):
            window = c[di-20:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 10:
                ZSCORE20[si, di] = (c[di] - np.mean(valid)) / (np.std(valid) + 1e-10)

        # OI 5-day change
        for di in range(6, ND):
            if not np.isnan(oi[di]) and not np.isnan(oi[di-5]) and oi[di-5] != 0:
                OI_CHG5[si, di] = (oi[di] - oi[di-5]) / (abs(oi[di-5]) + 1e-10)

    print(f"  All indicators computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # SIGNAL GENERATION — 10 systems
    # ================================================================
    print("\n[Signals] Computing 10 extreme systems...", flush=True)
    t0 = time.time()

    # ------------------------------------------------------------------
    # A) TRIPLE MOMENTUM: ROC(3)>3% + ROC(5)>2% + ROC(10)>0
    # ------------------------------------------------------------------
    sig_A = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(11, ND):
            r3 = ROC3[si, di]
            r5 = ROC5[si, di]
            r10 = ROC10[si, di]
            if np.isnan(r3) or np.isnan(r5) or np.isnan(r10):
                continue
            if r3 > 3.0 and r5 > 2.0 and r10 > 0:
                sig_A[si, di] = True
    print(f"  A) Triple momentum: {np.sum(sig_A)} signals")

    # ------------------------------------------------------------------
    # B) EXTREME ROC: ROC(5) > 5%
    # ------------------------------------------------------------------
    sig_B = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(6, ND):
            r5 = ROC5[si, di]
            if np.isnan(r5):
                continue
            if r5 > 5.0:
                sig_B[si, di] = True
    print(f"  B) Extreme ROC(5)>5%: {np.sum(sig_B)} signals")

    # ------------------------------------------------------------------
    # C) TRIPLE EXTREME: ROC(5)>2% + Z-score>2.5 + ADX>30
    # ------------------------------------------------------------------
    sig_C = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(20, ND):
            r5 = ROC5[si, di]
            zs = ZSCORE20[si, di]
            adx = ADX14[si, di]
            if np.isnan(r5) or np.isnan(zs) or np.isnan(adx):
                continue
            if r5 > 2.0 and zs > 2.5 and adx > 30:
                sig_C[si, di] = True
    print(f"  C) Triple extreme (ROC+Z+ADX): {np.sum(sig_C)} signals")

    # ------------------------------------------------------------------
    # D) ALL-IN with trailing stop + pyramiding
    # (handled by special backtest engine below)
    # Uses ROC(5)>2% as base signal
    # ------------------------------------------------------------------
    sig_D = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(6, ND):
            r5 = ROC5[si, di]
            if np.isnan(r5):
                continue
            if r5 > 2.0:
                sig_D[si, di] = True
    print(f"  D) ROC(5)>2% base for trailing stop: {np.sum(sig_D)} signals")

    # ------------------------------------------------------------------
    # E) DAILY RE-EVALUATION: ROC(5)>2% entry, ROC(5)<0 exit
    # (handled by special backtest engine below)
    # ------------------------------------------------------------------
    sig_E = sig_D.copy()
    print(f"  E) ROC(5)>2% base for daily re-eval: {np.sum(sig_E)} signals")

    # ------------------------------------------------------------------
    # F) COMMODITY ROTATION: rank jump from >30 to <5
    # Pre-compute daily ROC(5) ranks
    # ------------------------------------------------------------------
    ROC5_RANK = np.full((NS, ND), np.nan)
    for di in range(6, ND):
        r5_vals = ROC5[:, di]
        valid_mask = ~np.isnan(r5_vals)
        if valid_mask.sum() < 5:
            continue
        valid_indices = np.where(valid_mask)[0]
        valid_vals = r5_vals[valid_indices]
        # Rank: 1 = highest ROC5
        order = np.argsort(-valid_vals)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        for idx_in_valid, si in enumerate(valid_indices):
            ROC5_RANK[si, di] = ranks[idx_in_valid]

    # Rank yesterday
    ROC5_RANK_PREV = np.full((NS, ND), np.nan)
    for si in range(NS):
        ROC5_RANK_PREV[si] = np.roll(ROC5_RANK[si], 1)
        ROC5_RANK_PREV[si, 0] = np.nan

    sig_F = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(7, ND):
            rank_now = ROC5_RANK[si, di]
            rank_prev = ROC5_RANK_PREV[si, di]
            if np.isnan(rank_now) or np.isnan(rank_prev):
                continue
            if rank_now <= 5 and rank_prev >= 30:
                sig_F[si, di] = True
    print(f"  F) Commodity rotation (rank jump): {np.sum(sig_F)} signals")

    # ------------------------------------------------------------------
    # G) AGGRESSIVE COMPOUNDING
    # Uses ROC(5)>2% as base signal, special backtest engine
    # ------------------------------------------------------------------
    sig_G = sig_D.copy()
    print(f"  G) ROC(5)>2% base for aggressive compounding: {np.sum(sig_G)} signals")

    # ------------------------------------------------------------------
    # I) TOP 1 PER GROUP (diversified concentration)
    # For each group, rank by ROC5 and take top 1 if ROC5>2%
    # ------------------------------------------------------------------
    sig_I = np.zeros((NS, ND), dtype=bool)
    for di in range(6, ND):
        for g, members in group_members.items():
            best_si = -1
            best_roc = 2.0  # minimum threshold
            for si in members:
                r5 = ROC5[si, di]
                if np.isnan(r5):
                    continue
                if r5 > best_roc:
                    best_roc = r5
                    best_si = si
            if best_si >= 0:
                sig_I[best_si, di] = True
    print(f"  I) Top 1 per group ROC(5)>2%: {np.sum(sig_I)} signals")

    # ------------------------------------------------------------------
    # J) SIGNAL AGGREGATION WITH WEIGHTING
    # Weight = ROC(5) * ADX/50 * (1 + sign(OI_change))
    # ------------------------------------------------------------------
    WEIGHT = np.zeros((NS, ND))
    for si in range(NS):
        for di in range(14, ND):
            r5 = ROC5[si, di]
            adx = ADX14[si, di]
            oi_chg = OI_CHG5[si, di]
            if np.isnan(r5) or np.isnan(adx):
                continue
            oi_sign = 0 if np.isnan(oi_chg) else (1 if oi_chg > 0 else -1)
            WEIGHT[si, di] = r5 * (adx / 50.0) * (1 + oi_sign)

    sig_J = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(14, ND):
            r5 = ROC5[si, di]
            if np.isnan(r5):
                continue
            if r5 > 2.0 and WEIGHT[si, di] > 0:
                sig_J[si, di] = True
    print(f"  J) Weighted aggregation (ROC5>2%): {np.sum(sig_J)} signals")

    print(f"  All signals computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # STANDARD BACKTEST ENGINE (for A, B, C, F, I, J)
    # ================================================================
    def run_backtest(sig_arr, hold_days, top_n, wf_test_year=None,
                     score_arr=None, leverage=1.0):
        """Generic backtest for a signal array with next-open execution."""
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
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                positions = []

            # -- Close positions at end of hold period ---------------------
            closed = []
            for pos in positions:
                days_held = di - pos['entry_di']
                if days_held >= pos['hold_days']:
                    exit_price = C[pos['si'], di]
                    if np.isnan(exit_price) or exit_price <= 0:
                        exit_price = pos['entry_price']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    mkt_val = exit_price * mult * abs(pos['lots'])
                    pnl = (exit_price - pos['entry_price']) * mult * pos['lots'] * leverage
                    cash += mkt_val + pnl - mkt_val * COMM
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
                if score_arr is not None:
                    sc = score_arr[si, di]
                    if np.isnan(sc):
                        sc = 0
                else:
                    sc = ROC5[si, di] if not np.isnan(ROC5[si, di]) else 0
                candidates.append((sc, si, ep))

            if not candidates:
                continue

            candidates.sort(key=lambda x: -x[0])

            n_slots = top_n - len(positions)
            for sc_val, si, price in candidates[:max(0, n_slots)]:
                sym = syms[si]
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
            pnl = (exit_price - pos['entry_price']) * mult * pos['lots'] * leverage
            cash += mkt_val + pnl - mkt_val * COMM
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

        wf_mode = wf_test_year is not None
        n_days_test = (test_end_di - test_start_di) if wf_mode else (end_di - start_di)
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0
        avg_hold = np.mean([t['days_held'] for t in trades]) if trades else 0

        eq = float(CASH0)
        peak = eq
        mdd = 0.0
        for t in sorted(trades, key=lambda x: x['entry_di']):
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
    # D) TRAILING STOP BACKTEST ENGINE
    # ROC(5)>2% -> ALL IN, trailing stop at 2*ATR below highest close
    # ================================================================
    def run_trailing_stop_backtest(wf_test_year=None):
        """All-in with trailing stop: entry on ROC(5)>2%, max hold 20 days."""
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

        cash = float(CASH0)
        position = None  # only one position at a time (ALL IN)
        trades = []

        for di in range(start_di, end_di - 1):
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                position = None

            # -- If in position, check trailing stop -----------------------
            if position is not None:
                c_now = C[position['si'], di]
                h_now = H[position['si'], di] if di < ND else c_now
                if np.isnan(c_now):
                    c_now = position['highest_close']
                if np.isnan(h_now):
                    h_now = c_now

                # Update highest close
                if c_now > position['highest_close']:
                    position['highest_close'] = c_now

                # Compute profit %
                profit_pct = (c_now - position['entry_price']) / position['entry_price'] * 100

                # Determine stop level
                if profit_pct > 10:
                    stop_price = position['entry_price'] * 1.05
                elif profit_pct > 5:
                    stop_price = position['entry_price'] * 1.0  # breakeven
                else:
                    atr = ATR14[position['si'], di] if not np.isnan(ATR14[position['si'], di]) else position['entry_price'] * 0.02
                    stop_price = position['highest_close'] - 2 * atr

                days_held = di - position['entry_di']

                # Check exit conditions
                should_exit = False
                if days_held >= 20:
                    should_exit = True
                elif c_now < stop_price and days_held >= 1:
                    should_exit = True

                if should_exit:
                    exit_price = c_now
                    if np.isnan(exit_price) or exit_price <= 0:
                        exit_price = position['entry_price']
                    mult = MULT.get(position['sym'], DEF_MULT)
                    mkt_val = exit_price * mult * abs(position['lots'])
                    pnl = (exit_price - position['entry_price']) * mult * position['lots']
                    cash += mkt_val - mkt_val * COMM
                    invested = position['entry_price'] * mult * abs(position['lots'])
                    pnl_pct = pnl / invested * 100 if invested > 0 else 0
                    trades.append({
                        'pnl_pct': pnl_pct,
                        'entry_di': position['entry_di'],
                        'exit_di': di,
                        'year': dates[di].year if di < ND else dates[-1].year,
                        'sym': position.get('sym', ''),
                        'days_held': days_held,
                        'exit_reason': 'stop' if c_now < stop_price else 'max_hold',
                    })
                    position = None

            # -- If no position, look for entry signal --------------------
            if position is None:
                entry_di = di + 1
                if entry_di >= end_di:
                    continue

                candidates = []
                for si in range(NS):
                    if not sig_D[si, di]:
                        continue
                    ep = O[si, entry_di]
                    if np.isnan(ep) or ep <= 0:
                        continue
                    r5 = ROC5[si, di] if not np.isnan(ROC5[si, di]) else 0
                    candidates.append((r5, si, ep))

                if not candidates:
                    continue

                candidates.sort(key=lambda x: -x[0])
                _, si, price = candidates[0]

                sym = syms[si]
                mult = MULT.get(sym, DEF_MULT)
                contracts = max(1, int(cash / (price * mult)))
                cost_in = price * mult * contracts * (1 + COMM)
                if cost_in > cash:
                    contracts = int(cash * 0.9 / (price * mult * (1 + COMM)))
                    cost_in = price * mult * contracts * (1 + COMM) if contracts > 0 else 0
                if contracts <= 0 or cost_in <= 0 or cost_in > cash:
                    continue

                cash -= cost_in
                position = {
                    'si': si, 'entry_price': price, 'entry_di': entry_di,
                    'lots': contracts, 'dir': 1, 'sym': sym,
                    'highest_close': price,
                }

        # Close remaining
        if position is not None:
            ae = end_di - 1 if end_di < ND else ND - 1
            exit_price = C[position['si'], ae]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = position['entry_price']
            mult = MULT.get(position['sym'], DEF_MULT)
            mkt_val = exit_price * mult * abs(position['lots'])
            pnl = (exit_price - position['entry_price']) * mult * position['lots']
            cash += mkt_val - mkt_val * COMM
            invested = position['entry_price'] * mult * abs(position['lots'])
            pnl_pct = pnl / invested * 100 if invested > 0 else 0
            trades.append({
                'pnl_pct': pnl_pct,
                'entry_di': position['entry_di'],
                'exit_di': ae,
                'year': dates[ae].year if ae < ND else dates[-1].year,
                'sym': position.get('sym', ''),
                'days_held': ae - position['entry_di'],
                'exit_reason': 'end',
            })

        wf_mode = wf_test_year is not None
        n_days_test = (test_end_di - test_start_di) if wf_mode else (end_di - start_di)
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0
        avg_hold = np.mean([t['days_held'] for t in trades]) if trades else 0

        eq = float(CASH0)
        peak = eq
        mdd = 0.0
        for t in sorted(trades, key=lambda x: x['entry_di']):
            eq *= (1 + t['pnl_pct'] / 100)
            if eq > peak:
                peak = eq
            dd = (eq - peak) / peak * 100
            if dd < mdd:
                mdd = dd

        n_stop = sum(1 for t in trades if t.get('exit_reason') == 'stop')
        freq_per_yr = n_trades / (n_days_test / 252) if n_days_test > 0 else 0

        return {
            'ann': ann, 'wr': wr, 'n': n_trades, 'avg_pnl': avg_pnl,
            'final_cash': cash, 'n_days': n_days_test, 'mdd': mdd,
            'avg_hold': avg_hold, 'freq': freq_per_yr,
            'n_stop_exits': n_stop,
        }

    # ================================================================
    # E) DAILY RE-EVALUATION BACKTEST ENGINE
    # Enter on ROC(5)>2%, hold while ROC(5)>0, exit when ROC(5)<0
    # ================================================================
    def run_daily_reeval_backtest(wf_test_year=None):
        """Variable-length positions based on ROC(5) re-evaluation."""
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

        cash = float(CASH0)
        positions = []
        trades = []

        for di in range(start_di, end_di - 1):
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                positions = []

            # -- Check existing positions: exit if ROC(5) < 0 --------------
            closed = []
            for pos in positions:
                r5_now = ROC5[pos['si'], di]
                days_held = di - pos['entry_di']

                should_exit = False
                if days_held >= 1 and (np.isnan(r5_now) or r5_now < 0):
                    should_exit = True
                if days_held >= 30:  # max hold as safety
                    should_exit = True

                if should_exit:
                    exit_price = C[pos['si'], di]
                    if np.isnan(exit_price) or exit_price <= 0:
                        exit_price = pos['entry_price']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    mkt_val = exit_price * mult * abs(pos['lots'])
                    pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
                    cash += mkt_val - mkt_val * COMM
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

            # -- Enter new positions on ROC(5)>2% --------------------------
            entry_di = di + 1
            if entry_di >= end_di:
                continue

            candidates = []
            for si in range(NS):
                if not sig_E[si, di]:
                    continue
                if any(p['si'] == si for p in positions):
                    continue
                ep = O[si, entry_di]
                if np.isnan(ep) or ep <= 0:
                    continue
                r5 = ROC5[si, di] if not np.isnan(ROC5[si, di]) else 0
                candidates.append((r5, si, ep))

            if not candidates:
                continue

            candidates.sort(key=lambda x: -x[0])

            # All-in on top 1
            for sc_val, si, price in candidates[:1]:
                sym = syms[si]
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
                })

        # Close remaining
        for pos in positions:
            ae = end_di - 1 if end_di < ND else ND - 1
            exit_price = C[pos['si'], ae]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos['entry_price']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = exit_price * mult * abs(pos['lots'])
            pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
            cash += mkt_val - mkt_val * COMM
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

        wf_mode = wf_test_year is not None
        n_days_test = (test_end_di - test_start_di) if wf_mode else (end_di - start_di)
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0
        avg_hold = np.mean([t['days_held'] for t in trades]) if trades else 0

        eq = float(CASH0)
        peak = eq
        mdd = 0.0
        for t in sorted(trades, key=lambda x: x['entry_di']):
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
    # G) AGGRESSIVE COMPOUNDING BACKTEST ENGINE
    # After WIN: add 50% of profit to next trade's capital
    # After LOSS: reduce by 20% of loss
    # ================================================================
    def run_aggressive_compound_backtest(wf_test_year=None):
        """Aggressive compounding: ROC(5)>2%, top 1, 5-day hold."""
        hold_days = 5
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

        cash = float(CASH0)
        trade_capital = float(CASH0)  # adjusted after each trade
        positions = []
        trades = []

        for di in range(start_di, end_di - 1):
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                trade_capital = float(CASH0)
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
                    pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
                    cash += mkt_val - mkt_val * COMM
                    invested = pos['entry_price'] * mult * abs(pos['lots'])
                    pnl_pct = pnl / invested * 100 if invested > 0 else 0
                    trades.append({
                        'pnl_pct': pnl_pct,
                        'entry_di': pos['entry_di'],
                        'exit_di': di,
                        'year': dates[di].year if di < ND else dates[-1].year,
                        'sym': pos.get('sym', ''),
                        'days_held': days_held,
                        'invested': invested,
                        'pnl_abs': pnl,
                    })
                    closed.append(pos)

                    # Adjust trade capital
                    if pnl > 0:
                        trade_capital += pnl * 0.5
                    else:
                        trade_capital += pnl * 0.2  # pnl is negative, so this reduces
                    trade_capital = max(trade_capital, 10000)  # minimum

            for pos in closed:
                positions.remove(pos)

            if len(positions) >= 1:
                continue

            # -- Generate signals -----------------------------------------
            entry_di = di + 1
            if entry_di >= end_di:
                continue

            candidates = []
            for si in range(NS):
                if not sig_G[si, di]:
                    continue
                if any(p['si'] == si for p in positions):
                    continue
                ep = O[si, entry_di]
                if np.isnan(ep) or ep <= 0:
                    continue
                r5 = ROC5[si, di] if not np.isnan(ROC5[si, di]) else 0
                candidates.append((r5, si, ep))

            if not candidates:
                continue

            candidates.sort(key=lambda x: -x[0])

            # All-in on top 1, using trade_capital
            _, si, price = candidates[0]
            sym = syms[si]
            mult = MULT.get(sym, DEF_MULT)
            use_cap = min(trade_capital, cash)
            contracts = max(1, int(use_cap / (price * mult)))
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

        # Close remaining
        for pos in positions:
            ae = end_di - 1 if end_di < ND else ND - 1
            exit_price = C[pos['si'], ae]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos['entry_price']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = exit_price * mult * abs(pos['lots'])
            pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
            cash += mkt_val - mkt_val * COMM
            invested = pos['entry_price'] * mult * abs(pos['lots'])
            pnl_pct = pnl / invested * 100 if invested > 0 else 0
            trades.append({
                'pnl_pct': pnl_pct,
                'entry_di': pos['entry_di'],
                'exit_di': ae,
                'year': dates[ae].year if ae < ND else dates[-1].year,
                'sym': pos.get('sym', ''),
                'days_held': ae - pos['entry_di'],
                'invested': invested,
                'pnl_abs': pnl,
            })

        wf_mode = wf_test_year is not None
        n_days_test = (test_end_di - test_start_di) if wf_mode else (end_di - start_di)
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0
        avg_hold = np.mean([t['days_held'] for t in trades]) if trades else 0

        eq = float(CASH0)
        peak = eq
        mdd = 0.0
        for t in sorted(trades, key=lambda x: x['entry_di']):
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
    # F) COMMODITY ROTATION BACKTEST ENGINE
    # Enter on rank jump, exit when rank drops back >20
    # ================================================================
    def run_rotation_backtest(wf_test_year=None):
        """Rotation: enter on rank jump, exit when rank drops >20."""
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

        cash = float(CASH0)
        positions = []
        trades = []

        for di in range(start_di, end_di - 1):
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                positions = []

            # -- Check existing positions: exit if rank > 20 ---------------
            closed = []
            for pos in positions:
                rank_now = ROC5_RANK[pos['si'], di]
                days_held = di - pos['entry_di']

                should_exit = False
                if days_held >= 1 and (np.isnan(rank_now) or rank_now > 20):
                    should_exit = True
                if days_held >= 20:
                    should_exit = True

                if should_exit:
                    exit_price = C[pos['si'], di]
                    if np.isnan(exit_price) or exit_price <= 0:
                        exit_price = pos['entry_price']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    mkt_val = exit_price * mult * abs(pos['lots'])
                    pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
                    cash += mkt_val - mkt_val * COMM
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

            if len(positions) >= 1:
                continue

            # -- Enter on rank jump signal --------------------------------
            entry_di = di + 1
            if entry_di >= end_di:
                continue

            candidates = []
            for si in range(NS):
                if not sig_F[si, di]:
                    continue
                if any(p['si'] == si for p in positions):
                    continue
                ep = O[si, entry_di]
                if np.isnan(ep) or ep <= 0:
                    continue
                r5 = ROC5[si, di] if not np.isnan(ROC5[si, di]) else 0
                candidates.append((r5, si, ep))

            if not candidates:
                continue

            candidates.sort(key=lambda x: -x[0])
            _, si, price = candidates[0]

            sym = syms[si]
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
            })

        # Close remaining
        for pos in positions:
            ae = end_di - 1 if end_di < ND else ND - 1
            exit_price = C[pos['si'], ae]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos['entry_price']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = exit_price * mult * abs(pos['lots'])
            pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
            cash += mkt_val - mkt_val * COMM
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

        wf_mode = wf_test_year is not None
        n_days_test = (test_end_di - test_start_di) if wf_mode else (end_di - start_di)
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0
        avg_hold = np.mean([t['days_held'] for t in trades]) if trades else 0

        eq = float(CASH0)
        peak = eq
        mdd = 0.0
        for t in sorted(trades, key=lambda x: x['entry_di']):
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

    # A) Triple momentum: ROC(3)>3% + ROC(5)>2% + ROC(10)>0
    for hd in [3, 5]:
        for tn in [1]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'A', 'engine': 'standard',
                'hold_days': hd, 'top_n': tn,
                'label': f"A_TripleMom_H{hd}_TN{tn}",
                'sig_arr': sig_A, 'score_arr': None,
            })

    # B) Extreme ROC: ROC(5) > 5%
    for hd in [3, 5]:
        for tn in [1]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'B', 'engine': 'standard',
                'hold_days': hd, 'top_n': tn,
                'label': f"B_ExtremeROC5_H{hd}_TN{tn}",
                'sig_arr': sig_B, 'score_arr': None,
            })

    # C) Triple extreme: ROC(5)>2% + Z-score>2.5 + ADX>30
    for hd in [5, 10]:
        for tn in [1]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'C', 'engine': 'standard',
                'hold_days': hd, 'top_n': tn,
                'label': f"C_TripleExt_H{hd}_TN{tn}",
                'sig_arr': sig_C, 'score_arr': None,
            })

    # D) All-in with trailing stop (special engine)
    configs.append({
        'id': len(configs) + 1, 'signal': 'D', 'engine': 'trailing_stop',
        'label': "D_TrailStop_AllIn",
        'sig_arr': sig_D, 'score_arr': None,
        'hold_days': 20, 'top_n': 1,
    })

    # E) Daily re-evaluation (special engine)
    configs.append({
        'id': len(configs) + 1, 'signal': 'E', 'engine': 'daily_reeval',
        'label': "E_DailyReeval",
        'sig_arr': sig_E, 'score_arr': None,
        'hold_days': 30, 'top_n': 1,
    })

    # F) Commodity rotation (special engine)
    configs.append({
        'id': len(configs) + 1, 'signal': 'F', 'engine': 'rotation',
        'label': "F_RankRotation",
        'sig_arr': sig_F, 'score_arr': None,
        'hold_days': 20, 'top_n': 1,
    })

    # G) Aggressive compounding (special engine)
    configs.append({
        'id': len(configs) + 1, 'signal': 'G', 'engine': 'compound',
        'label': "G_AggCompound",
        'sig_arr': sig_G, 'score_arr': None,
        'hold_days': 5, 'top_n': 1,
    })

    # H) Leverage simulation (informational)
    for lev in [1.0, 1.5, 2.0, 3.0]:
        cid += 1
        configs.append({
            'id': cid, 'signal': 'H', 'engine': 'standard',
            'hold_days': 5, 'top_n': 1,
            'label': f"H_Leverage{lev:.1f}x",
            'sig_arr': sig_D, 'score_arr': None,
            'leverage': lev,
        })

    # I) Top 1 per group ROC(5)>2%
    for hd in [5]:
        for tn in [8]:  # up to 8 groups
            cid += 1
            configs.append({
                'id': cid, 'signal': 'I', 'engine': 'standard',
                'hold_days': hd, 'top_n': tn,
                'label': f"I_TopPerGrp_H{hd}_TN{tn}",
                'sig_arr': sig_I, 'score_arr': None,
            })

    # J) Signal aggregation with weighting
    for hd in [5]:
        for tn in [1, 3]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'J', 'engine': 'standard',
                'hold_days': hd, 'top_n': tn,
                'label': f"J_Weighted_H{hd}_TN{tn}",
                'sig_arr': sig_J, 'score_arr': WEIGHT,
            })

    total = len(configs)
    print(f"  Total configs: {total}")

    # ================================================================
    # RUN ALL CONFIGS (full backtest)
    # ================================================================
    print("\n[Backtest] Running all configs...", flush=True)
    t1 = time.time()
    results = []

    for ci, cfg in enumerate(configs):
        if ci % 5 == 0:
            print(f"  Config {ci}/{total} ({len(results)} done, {time.time()-t1:.0f}s)", flush=True)

        engine = cfg.get('engine', 'standard')

        if engine == 'standard':
            lev = cfg.get('leverage', 1.0)
            r = run_backtest(cfg['sig_arr'], cfg['hold_days'], cfg['top_n'],
                             score_arr=cfg.get('score_arr'), leverage=lev)
        elif engine == 'trailing_stop':
            r = run_trailing_stop_backtest()
        elif engine == 'daily_reeval':
            r = run_daily_reeval_backtest()
        elif engine == 'rotation':
            r = run_rotation_backtest()
        elif engine == 'compound':
            r = run_aggressive_compound_backtest()
        else:
            continue

        if r and r['n'] >= 3:
            r['config'] = cfg
            r['label'] = cfg['label']
            r['signal'] = cfg['signal']
            results.append(r)

    print(f"\n  Done ({time.time()-t1:.0f}s, {len(results)} configs with >= 3 trades)")

    # Separate leverage results for special reporting
    lev_results = [r for r in results if r['signal'] == 'H']
    non_lev_results = [r for r in results if r['signal'] != 'H']
    non_lev_results.sort(key=lambda x: -x['ann'])

    # ================================================================
    # PRINT TOP RESULTS (non-leverage)
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  TOP RESULTS — NON-LEVERAGE (sorted by annual return)")
    print(f"{'=' * 160}")
    print(f"  {'#':>3} | {'Label':<30} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'MDD':>7} | {'AvgPnL':>8} | {'AvgHold':>7} | {'Freq':>6}")
    print("-" * 160)
    for i, r in enumerate(non_lev_results[:30]):
        print(f"  {i+1:>3} | {r['label']:<30} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['mdd']:>6.1f}% | {r['avg_pnl']:>+7.3f}% | {r['avg_hold']:>6.1f}d | {r['freq']:>5.1f}/yr")

    # ================================================================
    # LEVERAGE SIMULATION (INFORMATIONAL)
    # ================================================================
    print(f"\n{'=' * 160}")
    print(f"  H) LEVERAGE SIMULATION (INFORMATIONAL ONLY)")
    print(f"{'=' * 160}")
    print(f"  {'Label':<30} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'MDD':>7} | {'AvgPnL':>8} | {'Final$':>12}")
    print("-" * 160)
    lev_results.sort(key=lambda x: -x['ann'])
    for r in lev_results:
        print(f"  {r['label']:<30} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['mdd']:>6.1f}% | {r['avg_pnl']:>+7.3f}% | {r['final_cash']:>11.0f}")

    # ================================================================
    # BEST PER SIGNAL TYPE
    # ================================================================
    sig_names = {
        'A': 'A) Triple momentum confirm',
        'B': 'B) Extreme ROC(5)>5%',
        'C': 'C) Triple extreme ROC+Z+ADX',
        'D': 'D) All-in trailing stop',
        'E': 'E) Daily re-evaluation',
        'F': 'F) Commodity rotation',
        'G': 'G) Aggressive compounding',
        'I': 'I) Top 1 per group',
        'J': 'J) Weighted aggregation',
    }

    print(f"\n  BEST PER SIGNAL TYPE (non-leverage):")
    print(f"  {'Signal':<35} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'MDD':>7} | {'AvgPnL':>8} | AvgHold")
    print("-" * 120)

    best_per_sig = {}
    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J']:
        sub = [r for r in non_lev_results if r['signal'] == sig_key]
        if not sub:
            print(f"  {sig_names.get(sig_key, sig_key):<35} | NO RESULTS")
            continue
        best = sub[0]
        best_per_sig[sig_key] = best
        print(f"  {sig_names.get(sig_key, sig_key):<35} | {best['ann']:>+8.1f}% | {best['wr']:>5.1f}% | {best['n']:>5} | {best['mdd']:>6.1f}% | {best['avg_pnl']:>+7.3f}% | {best['avg_hold']:>5.1f}d")

    # ================================================================
    # WALK-FORWARD
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Collect WF configs: top 10 + best per signal
    wf_configs = list(non_lev_results[:10])
    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J']:
        if sig_key in best_per_sig:
            r = best_per_sig[sig_key]
            if r not in wf_configs:
                wf_configs.append(r)

    print(f"\n{'=' * 220}")
    print(f"  WALK-FORWARD ({len(wf_configs)} configs, years 2020-2025)")
    print(f"{'=' * 220}")

    header = f"  {'#':>3} | {'Config':<32} | {'Avg':>8} |"
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

        engine = cfg.get('engine', 'standard')
        for yr in wf_years:
            if engine == 'standard':
                wr_result = run_backtest(cfg['sig_arr'], cfg['hold_days'], cfg['top_n'],
                                         wf_test_year=yr, score_arr=cfg.get('score_arr'))
            elif engine == 'trailing_stop':
                wr_result = run_trailing_stop_backtest(wf_test_year=yr)
            elif engine == 'daily_reeval':
                wr_result = run_daily_reeval_backtest(wf_test_year=yr)
            elif engine == 'rotation':
                wr_result = run_rotation_backtest(wf_test_year=yr)
            elif engine == 'compound':
                wr_result = run_aggressive_compound_backtest(wf_test_year=yr)
            else:
                wr_result = None

            if wr_result:
                wf_row['windows'][yr] = wr_result['ann']
                wf_row['mdd'][yr] = wr_result['mdd']
                wf_row['wr'][yr] = wr_result['wr']
        wf_rows.append(wf_row)

        vals = [wf_row['windows'].get(yr, 0) for yr in wf_years]
        avg = np.mean(vals) if vals else 0
        pos = sum(1 for v in vals if v > 0)
        avg_mdd = np.mean(list(wf_row['mdd'].values())) if wf_row['mdd'] else 0
        avg_wr = np.mean(list(wf_row['wr'].values())) if wf_row['wr'] else 0

        row_str = f"  {i+1:>3} | {wf_row['label']:<32} | {avg:>+7.1f}% |"
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
    header2 = f"  {'Signal':<35} | {'WF Avg':>8} |"
    for yr in wf_years:
        header2 += f" {yr:>7} |"
    header2 += f" {'Pos':>4} | Avg MDD | Avg WR"
    print(header2)
    print("-" * 200)

    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J']:
        wf_match = [w for w in wf_rows if w['signal'] == sig_key]
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            avg = np.mean(vals) if vals else 0
            pos = sum(1 for v in vals if v > 0)
            avg_mdd = np.mean(list(wf['mdd'].values())) if wf['mdd'] else 0
            avg_wr = np.mean(list(wf['wr'].values())) if wf['wr'] else 0
            row_str = f"  {sig_names.get(sig_key, sig_key):<35} | {avg:>+7.1f}% |"
            for v in vals:
                row_str += f" {v:>+7.1f}% |"
            row_str += f" {pos}/6 | {avg_mdd:>6.1f}% | {avg_wr:>5.1f}%"
            print(row_str)
        else:
            print(f"  {sig_names.get(sig_key, sig_key):<35} | NO DATA")

    # ================================================================
    # KEY COMPARISONS
    # ================================================================
    print(f"\n{'=' * 100}")
    print("  KEY COMPARISONS")
    print(f"{'=' * 100}")

    # D trailing stop details
    d_results = [r for r in non_lev_results if r['signal'] == 'D']
    if d_results:
        d = d_results[0]
        print(f"\n  D) Trailing Stop Details:")
        print(f"    Stop exits: {d.get('n_stop_exits', 'N/A')}  |  Avg hold: {d['avg_hold']:.1f}d")

    # E daily re-eval vs fixed hold
    print(f"\n  E) Daily Re-evaluation vs Fixed Hold:")
    e_results = [r for r in non_lev_results if r['signal'] == 'E']
    e_wf = [w for w in wf_rows if w['signal'] == 'E']
    base_results = [r for r in non_lev_results if r['signal'] == 'A' and r['config']['hold_days'] == 5]
    if e_results and base_results:
        e = e_results[0]
        b = base_results[0]
        e_wf_avg = np.mean([e_wf[0]['windows'].get(yr, 0) for yr in wf_years]) if e_wf else 0
        print(f"    Daily re-eval: Ann={e['ann']:>+8.1f}% | Avg hold={e['avg_hold']:.1f}d | N={e['n']}")
        print(f"    Fixed hold 5d: Ann={b['ann']:>+8.1f}% | Avg hold={b['avg_hold']:.1f}d | N={b['n']}")
        if e_wf:
            print(f"    Daily re-eval WF avg: {e_wf_avg:>+8.1f}%")

    # G compounding
    g_results = [r for r in non_lev_results if r['signal'] == 'G']
    if g_results:
        g = g_results[0]
        print(f"\n  G) Aggressive Compounding:")
        print(f"    Ann={g['ann']:>+8.1f}% | WR={g['wr']:>5.1f}% | Final cash={g['final_cash']:,.0f}")

    # ================================================================
    # FINAL REPORT
    # ================================================================
    print(f"\n{'=' * 200}")
    print("  FINAL REPORT: EXTREME MOMENTUM V118")
    print(f"{'=' * 200}")
    print()

    # 1. Any config breaking 100%? 200%?
    over_100 = [r for r in non_lev_results if r['ann'] > 100]
    over_200 = [r for r in non_lev_results if r['ann'] > 200]
    print(f"  1. CONFIGS BREAKING THRESHOLDS:")
    print(f"     >100% annual: {len(over_100)} configs")
    for r in over_100[:10]:
        print(f"       {r['label']:<32} | {r['ann']:>+8.1f}% | WR={r['wr']:>5.1f}% | N={r['n']}")
    print(f"     >200% annual: {len(over_200)} configs")
    for r in over_200[:5]:
        print(f"       {r['label']:<32} | {r['ann']:>+8.1f}% | WR={r['wr']:>5.1f}% | N={r['n']}")

    # 2. Which extreme approach works best
    print(f"\n  2. BEST EXTREME APPROACH:")
    if non_lev_results:
        champ = non_lev_results[0]
        print(f"     Champion: {champ['label']}")
        print(f"     Ann={champ['ann']:>+8.1f}% | WR={champ['wr']:>5.1f}% | N={champ['n']} | MDD={champ['mdd']:>6.1f}%")
        print(f"     Avg PnL/trade: {champ['avg_pnl']:>+6.3f}% | Avg Hold: {champ['avg_hold']:>5.1f}d")

    # 3. Leverage simulation
    print(f"\n  3. LEVERAGE SIMULATION (INFORMATIONAL ONLY - NO LEVERAGE IN REAL TRADING):")
    for r in lev_results:
        print(f"     {r['label']:<30} | Ann={r['ann']:>+8.1f}% | Final={r['final_cash']:>12,.0f} | MDD={r['mdd']:>6.1f}%")

    # 4. Daily re-eval vs fixed hold
    print(f"\n  4. DAILY RE-EVALUATION vs FIXED HOLD:")
    if e_results:
        e = e_results[0]
        e_vals = [e_wf[0]['windows'].get(yr, 0) for yr in wf_years] if e_wf else [0]*6
        e_avg = np.mean(e_vals)
        e_pos = sum(1 for v in e_vals if v > 0)
        print(f"     Daily re-eval: Ann={e['ann']:>+8.1f}% | WF avg={e_avg:>+8.1f}% | WF pos={e_pos}/6")

        # Compare with best standard 5-day hold
        best5 = [r for r in non_lev_results if r['signal'] in ('A','B','C','I','J') and r['config']['hold_days'] == 5]
        if best5:
            b5 = best5[0]
            b5_wf = [w for w in wf_rows if w['label'] == b5['label']]
            if b5_wf:
                b5_vals = [b5_wf[0]['windows'].get(yr, 0) for yr in wf_years]
                b5_avg = np.mean(b5_vals)
                print(f"     Best fixed 5d: {b5['label']:<26} | Ann={b5['ann']:>+8.1f}% | WF avg={b5_avg:>+8.1f}%")
            else:
                print(f"     Best fixed 5d: {b5['label']:<26} | Ann={b5['ann']:>+8.1f}%")

    # 5. Aggressive compounding
    print(f"\n  5. AGGRESSIVE COMPOUNDING:")
    if g_results:
        g = g_results[0]
        g_wf = [w for w in wf_rows if w['signal'] == 'G']
        if g_wf:
            g_vals = [g_wf[0]['windows'].get(yr, 0) for yr in wf_years]
            g_avg = np.mean(g_vals)
            g_pos = sum(1 for v in g_vals if v > 0)
            print(f"     Ann={g['ann']:>+8.1f}% | WR={g['wr']:>5.1f}% | Final={g['final_cash']:>12,.0f}")
            print(f"     WF avg={g_avg:>+8.1f}% | WF pos={g_pos}/6")
        else:
            print(f"     Ann={g['ann']:>+8.1f}% | WR={g['wr']:>5.1f}% | Final={g['final_cash']:>12,.0f}")

    # Champion with WF
    if non_lev_results:
        champ = non_lev_results[0]
        champ_wf = [w for w in wf_rows if w['label'] == champ['label']]
        print(f"\n  {'='*70}")
        print(f"  CHAMPION: {champ['label']}")
        print(f"    Annual: {champ['ann']:>+8.1f}%  |  WR: {champ['wr']:>5.1f}%  |  N: {champ['n']:>4}  |  MDD: {champ['mdd']:>6.1f}%")
        if champ_wf:
            cw = champ_wf[0]
            vals = [cw['windows'].get(yr, 0) for yr in wf_years]
            print(f"    WF: {[f'{v:>+7.1f}%' for v in vals]}  |  {sum(1 for v in vals if v > 0)}/6 positive")
            print(f"    WF avg: {np.mean(vals):>+8.1f}%")
        print(f"  {'='*70}")

    # Summary of all signal types
    print(f"\n  ALL SIGNAL TYPES RANKED:")
    print(f"  {'#':>3} | {'Signal':<35} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'MDD':>7} | WF_Avg | WF_Pos")
    print("-" * 130)
    ranked_sigs = []
    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J']:
        sub = [r for r in non_lev_results if r['signal'] == sig_key]
        if sub:
            best = sub[0]
            wf_match = [w for w in wf_rows if w['signal'] == sig_key]
            if wf_match:
                vals = [wf_match[0]['windows'].get(yr, 0) for yr in wf_years]
                wf_avg = np.mean(vals)
                wf_pos = sum(1 for v in vals if v > 0)
            else:
                wf_avg = 0
                wf_pos = 0
            ranked_sigs.append((sig_key, best, wf_avg, wf_pos))

    ranked_sigs.sort(key=lambda x: -x[1]['ann'])
    for i, (sig_key, best, wf_avg, wf_pos) in enumerate(ranked_sigs):
        print(f"  {i+1:>3} | {sig_names.get(sig_key, sig_key):<35} | {best['ann']:>+8.1f}% | {best['wr']:>5.1f}% | {best['n']:>5} | {best['mdd']:>6.1f}% | {wf_avg:>+7.1f}% | {wf_pos}/6")

    print(f"\n  Total runtime: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
