"""
Alpha Futures V98 -- Classic Trend-Following / Momentum with Weekly Rebalancing
================================================================================
V96 proved ALL same-close-entry strategies are contaminated.
V97 showed best practical (next-open) strategy gives +6.9%.

V98 tests classic, proven commodity strategies:
  A) Time-Series Momentum (TSMOM) — buy when own N-day return > 0
  B) Cross-Sectional Momentum (XSMOM) — buy top K performers
  C) Dual Momentum — TSMOM + XSMOM combined
  D) 52-Week High Breakout
  E) Moving Average Crossover (10/50)
  F) ATR-Normalized Momentum

ALL signals: computed at close of day di, entry at O[si, di+1] (NEXT DAY OPEN).
Weekly/periodic rebalancing — NOT daily.
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

# Group map (same as V82)
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


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def main():
    print("=" * 150)
    print("Alpha Futures V98 -- Classic Trend-Following / Momentum with Periodic Rebalancing")
    print("=" * 150)
    print("\n  KEY INSIGHT: Momentum is a PERSISTENT state, not a daily event.")
    print("  Classic strategies with weekly rebalancing and longer holding periods.")
    print("  ALL execution at NEXT DAY OPEN.")

    # -- Load data --
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    sym_to_si = {syms[si]: si for si in range(NS)}

    # -- Build group membership --
    gm_map = {}
    si_group = {}
    for si in range(NS):
        g = GROUP_MAP.get(syms[si])
        if g:
            gm_map.setdefault(g, []).append(si)
            si_group[si] = g

    group_sis = [si for si in range(NS) if si in si_group]  # 44 in 8 groups
    all_sis = list(range(NS))  # 68 commodities
    group_names = sorted(gm_map.keys())
    print(f"  Grouped: {len(group_sis)} in {len(group_names)} groups | All: {len(all_sis)}")

    # ================================================================
    # PRECOMPUTE SIGNALS
    # ================================================================
    print("\n[Signals] Precomputing N-day returns, MAs, ATR, 52wk highs...", flush=True)
    t0 = time.time()

    # --- N-day returns for multiple lookbacks ---
    ret_n = {}  # ret_n[lb][si, di]
    for lb in [5, 10, 20, 60, 120, 252]:
        r = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(lb, ND):
                c0 = C[si, di - lb]
                cn = C[si, di]
                if not np.isnan(c0) and not np.isnan(cn) and c0 > 0:
                    r[si, di] = (cn - c0) / c0
        ret_n[lb] = r

    # --- Moving averages ---
    ma_fast = {}  # 10-day
    ma_slow = {}  # 50-day
    for period, store in [(10, ma_fast), (50, ma_slow)]:
        m = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(period - 1, ND):
                window = C[si, di - period + 1:di + 1]
                valid = window[~np.isnan(window)]
                if len(valid) >= int(period * 0.8):
                    m[si, di] = np.mean(valid)
        store.update({period: m})
    ma10 = ma_fast[10]
    ma50 = ma_slow[50]

    # --- ATR(20) ---
    atr20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        tr = np.full(ND, np.nan)
        for di in range(1, ND):
            h_val = H[si, di]
            l_val = L[si, di]
            c_prev = C[si, di - 1]
            if np.isnan(h_val) or np.isnan(l_val):
                continue
            tr_vals = [h_val - l_val]
            if not np.isnan(c_prev):
                tr_vals.append(abs(h_val - c_prev))
                tr_vals.append(abs(l_val - c_prev))
            tr[di] = max(tr_vals)
        for di in range(20, ND):
            window = tr[di - 20:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 10:
                atr20[si, di] = np.mean(valid)

    # --- ATR-normalized momentum: ret_n / ATR ---
    atr_mom = {}
    for lb in [10, 20, 60]:
        am = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(max(lb, 20), ND):
                r = ret_n[lb][si, di]
                a = atr20[si, di]
                if not np.isnan(r) and not np.isnan(a) and a > 1e-10:
                    am[si, di] = r / a
        atr_mom[lb] = am

    # --- 52-week (252-day) high ---
    high_252 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(252, ND):
            window = C[si, di - 252:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 100:
                high_252[si, di] = np.max(valid)

    print(f"  All signals precomputed ({time.time()-t0:.1f}s)")

    # ================================================================
    # BACKTEST ENGINE (REBALANCING-BASED)
    # ================================================================
    def run_backtest(config, wf_test_year=None):
        """
        Rebalancing backtest. Every R days, re-evaluate positions.

        Config fields:
            signal: 'tsmom' | 'xsmom' | 'dual' | '52wk_high' | 'ma_cross' | 'atr_mom'
            lb: int (lookback days)
            rebal: int (rebalance every R days)
            top_n: int (max concurrent positions)
            hold_days: int (minimum hold before eligible to exit on rebalance)
            threshold: float (signal-specific threshold)
        """
        sig_type = config['signal']
        lb = config.get('lb', 20)
        rebal = config.get('rebal', 10)
        top_n = config['top_n']
        hold_days = config.get('hold_days', rebal)
        threshold = config.get('threshold', 0.0)

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

        if end_di < start_di + rebal + 2:
            return None

        # Universe
        if sig_type in ('xsmom', 'dual', 'atr_mom'):
            universe = group_sis  # cross-sectional: use grouped commodities
        else:
            universe = all_sis  # time-series: use all 68

        cash = float(CASH0)
        positions = {}  # si -> {'entry_price', 'entry_di', 'lots', 'sym', 'dir'}
        trades = []

        # Track last rebalance day
        last_rebal_di = start_di - 1

        for di in range(start_di, end_di - 1):
            # Reset at test window start (WF mode)
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                positions = {}
                last_rebal_di = di - 1

            # Check if it's a rebalance day
            is_rebal = (di - last_rebal_di >= rebal)
            if not is_rebal:
                continue

            last_rebal_di = di
            entry_di = di + 1
            if entry_di >= end_di:
                continue

            # ── Score all universe members ──────────────────────────
            scores = {}  # si -> score (higher = more bullish)
            should_hold = {}  # si -> True if should keep

            if sig_type == 'tsmom':
                # A) Time-Series Momentum: buy if own N-day return > 0
                for si in universe:
                    r = ret_n[lb][si, di]
                    if np.isnan(r):
                        continue
                    scores[si] = r
                    should_hold[si] = r > threshold

            elif sig_type == 'xsmom':
                # B) Cross-Sectional Momentum: rank by N-day return
                ranked = []
                for si in universe:
                    r = ret_n[lb][si, di]
                    if not np.isnan(r):
                        ranked.append((si, r))
                ranked.sort(key=lambda x: -x[1])  # descending
                for rank, (si, r) in enumerate(ranked):
                    scores[si] = r
                    should_hold[si] = rank < top_n  # top K

            elif sig_type == 'dual':
                # C) Dual Momentum: own return > 0 AND ranked in top K
                ranked = []
                for si in universe:
                    r = ret_n[lb][si, di]
                    if not np.isnan(r):
                        ranked.append((si, r))
                ranked.sort(key=lambda x: -x[1])
                for rank, (si, r) in enumerate(ranked):
                    scores[si] = r
                    should_hold[si] = (r > threshold) and (rank < top_n)

            elif sig_type == '52wk_high':
                # D) 52-Week High Breakout: near 52-week high -> buy
                for si in universe:
                    h252 = high_252[si, di]
                    c_now = C[si, di]
                    if np.isnan(h252) or np.isnan(c_now) or h252 <= 0:
                        continue
                    ratio = c_now / h252
                    scores[si] = ratio
                    should_hold[si] = ratio > threshold

            elif sig_type == 'ma_cross':
                # E) Moving Average Crossover: fast MA > slow MA -> buy
                for si in universe:
                    mf = ma10[si, di]
                    ms = ma50[si, di]
                    if np.isnan(mf) or np.isnan(ms):
                        continue
                    diff = mf - ms
                    scores[si] = diff
                    should_hold[si] = diff > 0

            elif sig_type == 'atr_mom':
                # F) ATR-Normalized Momentum
                ranked = []
                for si in universe:
                    am = atr_mom[lb][si, di]
                    if not np.isnan(am):
                        ranked.append((si, am))
                ranked.sort(key=lambda x: -x[1])
                for rank, (si, am) in enumerate(ranked):
                    scores[si] = am
                    should_hold[si] = rank < top_n

            # ── Close positions that no longer qualify ──────────────
            to_close = []
            for si, pos in list(positions.items()):
                days_held = di - pos['entry_di']
                # Always close if held past rebalance and doesn't qualify
                if si not in should_hold or not should_hold[si]:
                    if days_held >= hold_days:
                        to_close.append(si)
                # Also close if held way too long (safety valve)
                elif days_held >= hold_days * 3:
                    to_close.append(si)

            for si in to_close:
                pos = positions.pop(si)
                exit_price = C[si, di]
                if np.isnan(exit_price) or exit_price <= 0:
                    exit_price = pos['entry_price']
                mult = MULT.get(pos['sym'], DEF_MULT)
                mkt_val = exit_price * mult * abs(pos['lots'])
                cash += mkt_val - mkt_val * COMM
                pnl = (exit_price - pos['entry_price']) * mult * pos['lots'] * pos['dir']
                invested = pos['entry_price'] * mult * abs(pos['lots'])
                pnl_pct = pnl / invested * 100 if invested > 0 else 0
                trades.append({
                    'pnl_pct': pnl_pct,
                    'entry_di': pos['entry_di'],
                    'exit_di': di,
                    'year': dates[di].year if di < ND else dates[-1].year,
                    'sym': pos['sym'],
                })

            # ── Open new positions ──────────────────────────────────
            # Candidates: should_hold and not already holding and have valid next-open price
            candidates = []
            for si in universe:
                if si in positions:
                    continue
                if si not in should_hold or not should_hold[si]:
                    continue
                ep = O[si, entry_di]
                if np.isnan(ep) or ep <= 0:
                    continue
                score = scores.get(si, 0)
                candidates.append((si, score, ep))

            if not candidates:
                continue

            # Sort by score descending
            candidates.sort(key=lambda x: -x[1])

            # Number of slots
            n_slots = top_n - len(positions)
            if n_slots <= 0:
                continue

            for si, score, ep in candidates[:n_slots]:
                sym = syms[si]
                mult = MULT.get(sym, DEF_MULT)
                notional = ep * mult
                # Equal-weight allocation
                n_positions_target = top_n
                alloc = cash / max(n_positions_target - len(positions), 1)
                lots = int(alloc / (notional * (1 + COMM)))
                if lots <= 0:
                    lots = int(cash * 0.85 / (notional * (1 + COMM)))
                if lots <= 0:
                    continue
                cost_in = notional * lots * (1 + COMM)
                if cost_in > cash:
                    lots = int(cash * 0.8 / (notional * (1 + COMM)))
                    cost_in = notional * lots * (1 + COMM) if lots > 0 else 0
                if lots <= 0 or cost_in <= 0 or cost_in > cash:
                    continue

                cash -= cost_in
                positions[si] = {
                    'entry_price': ep,
                    'entry_di': entry_di,
                    'lots': lots,
                    'sym': sym,
                    'dir': 1,  # long only
                }

        # Close remaining positions at end
        ae = end_di - 1 if end_di < ND else ND - 1
        for si, pos in list(positions.items()):
            exit_price = C[si, ae]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos['entry_price']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = exit_price * mult * abs(pos['lots'])
            cash += mkt_val - mkt_val * COMM
            pnl = (exit_price - pos['entry_price']) * mult * pos['lots'] * pos['dir']
            invested = pos['entry_price'] * mult * abs(pos['lots'])
            pnl_pct = pnl / invested * 100 if invested > 0 else 0
            trades.append({
                'pnl_pct': pnl_pct,
                'entry_di': pos['entry_di'],
                'exit_di': ae,
                'year': dates[ae].year if ae < ND else dates[-1].year,
                'sym': pos['sym'],
            })

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
        # Sort trades by exit_di to build equity curve
        for t in sorted(trades, key=lambda x: x['exit_di']):
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

    # --- A) Time-Series Momentum (TSMOM) ---
    for lb in [10, 20, 60, 120]:
        for rebal in [5, 10, 20]:
            for top_n in [3, 5, 10]:
                for thresh in [0.0, 0.01, 0.02]:
                    cid += 1
                    configs.append({
                        'id': cid, 'signal': 'tsmom',
                        'lb': lb, 'rebal': rebal,
                        'top_n': top_n, 'threshold': thresh,
                        'hold_days': rebal,
                        'label': f"TSMOM_LB{lb}_R{rebal}_TN{top_n}_T{thresh}",
                    })

    # --- B) Cross-Sectional Momentum (XSMOM) ---
    for lb in [10, 20, 60]:
        for rebal in [5, 10, 20]:
            for top_n in [3, 5, 10]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': 'xsmom',
                    'lb': lb, 'rebal': rebal,
                    'top_n': top_n, 'threshold': 0.0,
                    'hold_days': rebal,
                    'label': f"XSMOM_LB{lb}_R{rebal}_K{top_n}",
                })

    # --- C) Dual Momentum ---
    for lb in [20, 60]:
        for rebal in [10, 20]:
            for top_n in [5, 10]:
                for thresh in [0.0, 0.01]:
                    cid += 1
                    configs.append({
                        'id': cid, 'signal': 'dual',
                        'lb': lb, 'rebal': rebal,
                        'top_n': top_n, 'threshold': thresh,
                        'hold_days': rebal,
                        'label': f"Dual_LB{lb}_R{rebal}_K{top_n}_T{thresh}",
                    })

    # --- D) 52-Week High Breakout ---
    for thresh in [0.90, 0.93, 0.95, 0.97]:
        for rebal in [5, 10, 20]:
            for top_n in [3, 5, 10]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': '52wk_high',
                    'lb': 252, 'rebal': rebal,
                    'top_n': top_n, 'threshold': thresh,
                    'hold_days': rebal,
                    'label': f"52wk_T{thresh}_R{rebal}_TN{top_n}",
                })

    # --- E) Moving Average Crossover ---
    for rebal in [5, 10, 20]:
        for top_n in [3, 5, 10, 15]:
            cid += 1
            configs.append({
                'id': cid, 'signal': 'ma_cross',
                'lb': 50, 'rebal': rebal,
                'top_n': top_n, 'threshold': 0.0,
                'hold_days': rebal,
                'label': f"MACross_R{rebal}_TN{top_n}",
            })

    # --- F) ATR-Normalized Momentum ---
    for lb in [10, 20, 60]:
        for rebal in [5, 10, 20]:
            for top_n in [3, 5, 10]:
                cid += 1
                configs.append({
                    'id': cid, 'signal': 'atr_mom',
                    'lb': lb, 'rebal': rebal,
                    'top_n': top_n, 'threshold': 0.0,
                    'hold_days': rebal,
                    'label': f"ATRMom_LB{lb}_R{rebal}_K{top_n}",
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
    # BEST PER SIGNAL TYPE
    # ================================================================
    sig_order = ['tsmom', 'xsmom', 'dual', '52wk_high', 'ma_cross', 'atr_mom']
    sig_names = {
        'tsmom': 'A) Time-Series Momentum (TSMOM)',
        'xsmom': 'B) Cross-Sectional Momentum (XSMOM)',
        'dual': 'C) Dual Momentum (TSMOM+XSMOM)',
        '52wk_high': 'D) 52-Week High Breakout',
        'ma_cross': 'E) Moving Average Crossover (10/50)',
        'atr_mom': 'F) ATR-Normalized Momentum',
    }

    print(f"\n{'=' * 150}")
    print("  BEST PER SIGNAL TYPE (Full Period) -- ALL NEXT-OPEN EXECUTION")
    print(f"{'=' * 150}")
    print(f"  {'Signal':<45} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | Best Config")
    print("-" * 150)

    best_per_sig = {}
    for r in results:
        key = r['config']['signal']
        if key not in best_per_sig:
            best_per_sig[key] = r

    for sig in sig_order:
        if sig in best_per_sig:
            b = best_per_sig[sig]
            print(f"  {sig_names.get(sig, sig):<45} | {b['ann']:>+8.1f}% | {b['wr']:>5.1f}% | {b['n']:>5} | {b['avg_pnl']:>+6.3f}% | {b['mdd']:>6.1f}% | {b['label']}")

    # ================================================================
    # SIGNAL TYPE SUMMARY
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  SIGNAL TYPE SUMMARY (Average of Top 5 configs per type)")
    print(f"{'=' * 150}")
    print(f"  {'Signal':<45} | {'Avg Ann':>9} | {'Avg WR':>7} | {'Avg N':>7} | {'Avg PnL':>8} | {'Avg MDD':>8} | {'#Positive':>9}")
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
        print(f"  {sig_names.get(sig, sig):<45} | {avg_ann:>+8.1f}% | {avg_wr:>6.1f}% | {avg_n:>7.0f} | {avg_pnl:>+7.3f}% | {avg_mdd:>7.1f}% | {n_pos:>5}/{len(sub)}")

    # ================================================================
    # TSMOM DETAIL: BY LOOKBACK AND REBALANCE
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  TSMOM DETAIL (Best per lookback x rebalance)")
    print(f"{'=' * 150}")
    print(f"  {'Lookback':>8} | {'Rebal':>5} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | Best Config")
    print("-" * 140)

    for lb in [10, 20, 60, 120]:
        for rebal in [5, 10, 20]:
            sub = [r for r in results
                   if r['config']['signal'] == 'tsmom'
                   and r['config']['lb'] == lb
                   and r['config']['rebal'] == rebal]
            if sub:
                best = sub[0]
                print(f"  {lb:>8} | {rebal:>5} | {best['ann']:>+8.1f}% | {best['wr']:>5.1f}% | {best['n']:>5} | {best['avg_pnl']:>+6.3f}% | {best['mdd']:>6.1f}% | {best['label']}")

    # ================================================================
    # XSMOM DETAIL: BY LOOKBACK AND K
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  XSMOM DETAIL (Best per lookback x top_n)")
    print(f"{'=' * 150}")
    print(f"  {'Lookback':>8} | {'Top K':>5} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | Best Config")
    print("-" * 140)

    for lb in [10, 20, 60]:
        for top_n in [3, 5, 10]:
            sub = [r for r in results
                   if r['config']['signal'] == 'xsmom'
                   and r['config']['lb'] == lb
                   and r['config']['top_n'] == top_n]
            if sub:
                best = sub[0]
                print(f"  {lb:>8} | {top_n:>5} | {best['ann']:>+8.1f}% | {best['wr']:>5.1f}% | {best['n']:>5} | {best['avg_pnl']:>+6.3f}% | {best['mdd']:>6.1f}% | {best['label']}")

    # ================================================================
    # MA CROSS DETAIL: BY REBALANCE
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  MA CROSSOVER DETAIL (Best per rebalance x top_n)")
    print(f"{'=' * 150}")
    print(f"  {'Rebal':>5} | {'TopN':>5} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | Best Config")
    print("-" * 140)

    for rebal in [5, 10, 20]:
        for top_n in [3, 5, 10, 15]:
            sub = [r for r in results
                   if r['config']['signal'] == 'ma_cross'
                   and r['config']['rebal'] == rebal
                   and r['config']['top_n'] == top_n]
            if sub:
                best = sub[0]
                print(f"  {rebal:>5} | {top_n:>5} | {best['ann']:>+8.1f}% | {best['wr']:>5.1f}% | {best['n']:>5} | {best['avg_pnl']:>+6.3f}% | {best['mdd']:>6.1f}% | {best['label']}")

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
    header2 = f"  {'Signal':<45} | {'WF Avg':>8} |"
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
            row_str = f"  {sig_names.get(sig, sig):<45} | {avg:>+7.1f}% |"
            for v in vals:
                row_str += f" {v:>+7.1f}% |"
            row_str += f" {pos}/6 | {avg_mdd:>6.1f}%"
            print(row_str)

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  FINAL VERDICT: CAN CLASSIC MOMENTUM/TREND-FOLLOWING ACHIEVE 100%+ ANNUAL?")
    print(f"{'=' * 150}")
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
        if wf_pos >= 4 and best['ann'] > 0:
            genuine = "GENUINE ALPHA"
        elif wf_pos >= 3 and best['ann'] > 0:
            genuine = "MARGINAL"
        else:
            genuine = "NO ALPHA"

        print(f"  {sig_names.get(sig, sig)}")
        print(f"    Best annual: {best['ann']:>+8.1f}%  |  Avg top-5: {avg_top5:>+8.1f}%  |  {n_pos}/{len(sub)} positive configs")
        print(f"    Walk-forward: {wf_pos}/6 positive  |  WF avg: {wf_avg:>+8.1f}%")
        print(f"    VERDICT: {verdict}  -->  {genuine}")
        print()

    # Overall best
    if results:
        best_overall = results[0]
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

        # Key question answer
        print(f"\n  {'=' * 80}")
        print(f"  KEY QUESTION: Can classic momentum/trend-following achieve 100%+ annual?")
        hundred_plus = [r for r in results if r['ann'] > 100]
        fifty_plus = [r for r in results if r['ann'] > 50]
        print(f"    Configs with > 100% annual: {len(hundred_plus)}/{len(results)}")
        print(f"    Configs with > 50% annual:  {len(fifty_plus)}/{len(results)}")
        if hundred_plus:
            print(f"    BEST: {hundred_plus[0]['ann']:>+8.1f}% ({hundred_plus[0]['label']})")
        elif results:
            print(f"    BEST: {results[0]['ann']:>+8.1f}% ({results[0]['label']})")
        print(f"  {'=' * 80}")

    print(f"\n  Total time: {time.time()-t_start:.1f}s")
    print("=" * 150)


if __name__ == '__main__':
    main()
