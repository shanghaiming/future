"""
Alpha Futures V85 — Overnight Gap Decomposition + Group Momentum
================================================================
V74 champion: +2185% with close-to-close group momentum LB=1, extended groups.

V85 IDEA: Decompose daily returns into overnight (close-to-open) and intraday
(open-to-close) components. Overnight gaps reflect new information arrival.
Intraday returns reflect microstructure.

KEY INSIGHT: overnight_ret[si, di] = (O[di] - C[di-1]) / C[di-1] is KNOWN at
the open of day di. So for open-to-close trades (A, D), we can use today's
overnight return to trade at today's open.

For close-to-close signals (B, C, E, V74_baseline), same as V74: signal uses
today's data, entry at today's close, exit next close.

ALL SIGNALS ARE LONG-ONLY (V74 proved shorting destroys returns).

SIGNALS:
  A: Pure overnight gap divergence — group gaps up but commodity lags -> buy at open, sell at close
  B: V74 baseline with overnight filter — only trade when overnight gap agrees with V74 direction
  C: Intraday continuation — positive gap AND positive V74 -> stronger signal
  D: Gap reversal — negative gap but positive V74 -> expect intraday recovery
  E: Decomposed momentum — alpha*overnight_div + beta*intraday_div (sweep alpha/beta)
  V74_baseline: reproduced V74 signal for comparison

Walk-forward: 6 windows (2020-2025)
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
        'bcfi': 5, 'nrfi': 1, 'lgfi': 20, 'brfi': 5, 'lcfi': 1, 'sisi': 5,
        'ni': 1, 'tai': 5}
DEF_MULT = 10
COMM = 0.0003

# Extended group map (same as V74)
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


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def main():
    print("=" * 110)
    print("Alpha Futures V85 -- Overnight Gap Decomposition + Group Momentum")
    print("=" * 110)

    # Load data
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ================================================================
    # PRECOMPUTE: Overnight, Intraday, Close-to-Close returns
    # ================================================================
    print("\n[Signals] Computing return decomposition...", flush=True)
    t0 = time.time()

    # overnight_ret[si, di] = (O[di] - C[di-1]) / C[di-1]  -- known at open of day di
    # intraday_ret[si, di]  = (C[di] - O[di]) / O[di]      -- known at close of day di
    # ctc_ret[si, di]       = (C[di] - C[di-1]) / C[di-1]  -- known at close of day di

    overnight_ret = np.full((NS, ND), np.nan)
    intraday_ret = np.full((NS, ND), np.nan)
    ctc_ret = np.full((NS, ND), np.nan)

    for si in range(NS):
        for di in range(1, ND):
            c_prev = C[si, di - 1]
            o_now = O[si, di]
            c_now = C[si, di]
            if not np.isnan(c_prev) and c_prev > 0 and not np.isnan(o_now) and o_now > 0:
                overnight_ret[si, di] = (o_now - c_prev) / c_prev
            if not np.isnan(o_now) and o_now > 0 and not np.isnan(c_now) and c_now > 0:
                intraday_ret[si, di] = (c_now - o_now) / o_now
            if not np.isnan(c_prev) and c_prev > 0 and not np.isnan(c_now) and c_now > 0:
                ctc_ret[si, di] = (c_now - c_prev) / c_prev

    # Build group membership
    gm_map = {}
    for si in range(NS):
        g = GROUP_MAP.get(syms[si])
        if g:
            gm_map.setdefault(g, []).append(si)

    print(f"  Groups: {len(gm_map)} groups, {sum(len(v) for v in gm_map.values())} members")

    # Group means for each return type (exclude self)
    grp_overnight = np.full((NS, ND), np.nan)
    grp_intraday = np.full((NS, ND), np.nan)
    grp_ctc = np.full((NS, ND), np.nan)

    for grp, members in gm_map.items():
        for di in range(ND):
            for sj in members:
                vals_on = [overnight_ret[sk, di] for sk in members
                           if sk != sj and not np.isnan(overnight_ret[sk, di])]
                vals_id = [intraday_ret[sk, di] for sk in members
                           if sk != sj and not np.isnan(intraday_ret[sk, di])]
                vals_ctc = [ctc_ret[sk, di] for sk in members
                            if sk != sj and not np.isnan(ctc_ret[sk, di])]
                if vals_on:
                    grp_overnight[sj, di] = np.mean(vals_on)
                if vals_id:
                    grp_intraday[sj, di] = np.mean(vals_id)
                if vals_ctc:
                    grp_ctc[sj, di] = np.mean(vals_ctc)

    # Divergences: positive = group ahead, commodity lagging -> buy signal
    overnight_div = grp_overnight - overnight_ret   # group_overnight - own_overnight
    intraday_div = grp_intraday - intraday_ret       # group_intraday - own_intraday
    ctc_div = grp_ctc - ctc_ret                      # group_ctc - own_ctc (V74 baseline)

    print(f"  Signals computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # CORRELATION ANALYSIS
    # ================================================================
    print("\n[Analysis] Overnight vs Intraday group correlation...")

    all_on_corrs = []
    all_id_corrs = []
    for grp, members in gm_map.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                si, sj = members[i], members[j]
                valid = ~np.isnan(overnight_ret[si]) & ~np.isnan(overnight_ret[sj])
                if valid.sum() > 100:
                    c_on = np.corrcoef(overnight_ret[si, valid], overnight_ret[sj, valid])[0, 1]
                    if not np.isnan(c_on):
                        all_on_corrs.append(c_on)
                valid2 = ~np.isnan(intraday_ret[si]) & ~np.isnan(intraday_ret[sj])
                if valid2.sum() > 100:
                    c_id = np.corrcoef(intraday_ret[si, valid2], intraday_ret[sj, valid2])[0, 1]
                    if not np.isnan(c_id):
                        all_id_corrs.append(c_id)

    print(f"  Pairwise overnight correlation (mean): {np.mean(all_on_corrs):.4f}")
    print(f"  Pairwise intraday  correlation (mean): {np.mean(all_id_corrs):.4f}")
    if all_on_corrs and all_id_corrs:
        diff = np.mean(all_on_corrs) - np.mean(all_id_corrs)
        winner = "overnight" if diff > 0 else "intraday"
        print(f"  --> {winner} returns are MORE correlated within groups (diff={diff:+.4f})")

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(config, wf_test_year=None):
        """
        All signals are LONG-ONLY (V74 proved shorting destroys returns).

        Signal types:
          A_overnight_div:  open-to-close trade, signal = overnight_div[di] (known at open)
          D_gap_reversal:   open-to-close trade, signal uses overnight_ret + ctc_div[di-1]
          B,C,E,V74:        close-to-close trade, same timing as V74

        For open-to-close (A, D):
          - Signal uses data available at today's open (overnight_ret[di] and/or ctc_div[di-1])
          - Entry at today's open O[si, di]
          - Exit at today's close C[si, di]

        For close-to-close (B, C, E, V74_baseline):
          - Signal uses today's close-to-close data ctc_div[si, di]
          - Entry at today's close C[si, di]
          - Exit at next day's close C[si, di+1]
        """
        signal_type = config['signal']
        threshold = config['threshold']
        top_n = config['top_n']
        comm = config.get('comm', COMM)
        alpha = config.get('alpha', 0.5)
        beta = config.get('beta', 0.5)

        is_open_entry = signal_type in ('A_overnight_div', 'D_gap_reversal')

        # Get tradeable si indices (those with group)
        trade_sis = [si for si in range(NS) if GROUP_MAP.get(syms[si])]

        # Date range setup for walk-forward
        wf_mode = wf_test_year is not None
        if wf_mode:
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
            end_di = test_end_di
            start_di = MIN_TRAIN
        else:
            test_start_di = MIN_TRAIN
            test_end_di = ND
            start_di = MIN_TRAIN
            end_di = ND

        cash = float(CASH0)
        positions = []
        trades = []

        for di in range(start_di, end_di):
            # Reset cash at start of test window (WF only)
            if wf_mode and di == test_start_di:
                cash = float(CASH0)
                positions = []

            # --- Close positions ---
            closed = []
            for pos in positions:
                should_close = False
                if is_open_entry:
                    # Open-to-close: entered at today's open, exit at today's close
                    if di == pos['entry_di']:
                        should_close = True
                else:
                    # Close-to-close: entered yesterday, exit today
                    if di - pos['entry_di'] >= 1:
                        should_close = True

                if should_close:
                    cn = C[pos['si'], di]
                    if np.isnan(cn) or cn <= 0:
                        cn = pos['entry']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    mkt_val = cn * mult * pos['lots']
                    cash += mkt_val - mkt_val * comm
                    pnl = (cn - pos['entry']) * mult * pos['lots'] * pos['dir']
                    invested = pos['entry'] * mult * pos['lots']
                    pnl_pct = pnl / invested * 100 if invested > 0 else 0
                    trades.append({
                        'pnl_pct': pnl_pct,
                        'di': pos['entry_di'],
                        'year': dates[di].year if di < ND else dates[-1].year,
                        'dir': pos['dir'],
                    })
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            # --- Score candidates ---
            candidates = []
            for si in trade_sis:
                sym = syms[si]

                if any(p['si'] == si for p in positions):
                    continue

                # Compute signal based on type
                if signal_type == 'A_overnight_div':
                    # Pure overnight gap divergence, known at today's open
                    # overnight_ret[si, di] is known at open of day di
                    # grp_overnight[si, di] is also known at open of day di
                    if di < 1:
                        continue
                    o_now = O[si, di]
                    if np.isnan(o_now) or o_now <= 0:
                        continue
                    div = overnight_div[si, di]  # group_overnight - own_overnight
                    if np.isnan(div):
                        continue
                    if div > threshold:
                        candidates.append((si, div, 1))  # LONG only

                elif signal_type == 'B_overnight_filter':
                    # V74 signal (ctc_div at di) + overnight filter
                    # ctc_div[si, di] uses today's close price, so signal known at close
                    c_now = C[si, di]
                    if np.isnan(c_now) or c_now <= 0:
                        continue
                    v74_div = ctc_div[si, di]
                    own_on = overnight_ret[si, di]
                    grp_on = grp_overnight[si, di]
                    if np.isnan(v74_div) or np.isnan(own_on) or np.isnan(grp_on):
                        continue
                    # V74: buy if group > self, AND overnight gap agrees
                    if v74_div > threshold and grp_on > own_on:
                        candidates.append((si, v74_div, 1))  # LONG only

                elif signal_type == 'C_continuation':
                    # Positive overnight gap AND positive V74 divergence -> stronger
                    c_now = C[si, di]
                    if np.isnan(c_now) or c_now <= 0:
                        continue
                    v74_div = ctc_div[si, di]
                    on_ret = overnight_ret[si, di]
                    if np.isnan(v74_div) or np.isnan(on_ret):
                        continue
                    # Buy: V74 says group > self AND overnight gap is positive -> continuation
                    if v74_div > threshold and on_ret > 0:
                        score = v74_div * (1 + abs(on_ret) * 10)
                        candidates.append((si, score, 1))

                elif signal_type == 'D_gap_reversal':
                    # Negative overnight gap but positive V74 (using yesterday's ctc_div)
                    # -> expect intraday recovery, buy at open
                    # Use ctc_div[si, di-1] (known at yesterday's close) + overnight_ret[si, di] (known at open)
                    if di < 1:
                        continue
                    o_now = O[si, di]
                    if np.isnan(o_now) or o_now <= 0:
                        continue
                    v74_div = ctc_div[si, di - 1]  # yesterday's V74 signal
                    on_ret = overnight_ret[si, di]   # today's overnight gap
                    if np.isnan(v74_div) or np.isnan(on_ret):
                        continue
                    # Buy: yesterday V74 said group > self, BUT today gapped down -> recovery
                    if v74_div > threshold and on_ret < 0:
                        candidates.append((si, v74_div, 1))

                elif signal_type == 'E_decomposed':
                    # alpha*overnight_div + beta*intraday_div
                    # Both use yesterday's data (di-1) to avoid look-ahead
                    c_now = C[si, di]
                    if np.isnan(c_now) or c_now <= 0:
                        continue
                    ov_div = overnight_div[si, di]
                    id_div = intraday_div[si, di]
                    if np.isnan(ov_div) or np.isnan(id_div):
                        continue
                    score = alpha * ov_div + beta * id_div
                    if score > threshold:
                        candidates.append((si, score, 1))

                elif signal_type == 'V74_baseline':
                    # Pure V74: ctc_div with LB=1, LONG ONLY
                    c_now = C[si, di]
                    if np.isnan(c_now) or c_now <= 0:
                        continue
                    div = ctc_div[si, di]
                    if np.isnan(div):
                        continue
                    if div > threshold:
                        candidates.append((si, div, 1))

            if not candidates:
                continue

            # Sort by score (highest first)
            candidates.sort(key=lambda x: -x[1])

            # Open positions
            n_slots = top_n - len(positions)
            for si, score, direction in candidates[:n_slots]:
                if is_open_entry:
                    price = O[si, di]
                else:
                    price = C[si, di]
                if np.isnan(price) or price <= 0:
                    continue
                mult = MULT.get(syms[si], DEF_MULT)
                notional = price * mult
                lots = int(cash / (notional * (1 + comm)))
                if lots <= 0:
                    continue
                cost_in = notional * lots * (1 + comm)
                if cost_in > cash:
                    lots = int(cash * 0.95 / (notional * (1 + comm)))
                    cost_in = notional * lots * (1 + comm) if lots > 0 else 0
                if lots <= 0 or cost_in <= 0 or cost_in > cash:
                    continue

                cash -= cost_in
                positions.append({
                    'si': si, 'entry': price, 'entry_di': di,
                    'lots': lots, 'dir': direction, 'sym': syms[si],
                })

        # Close remaining positions
        for pos in positions:
            ae = end_di - 1 if end_di < ND else ND - 1
            cn = C[pos['si'], ae]
            if np.isnan(cn) or cn <= 0:
                cn = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = cn * mult * pos['lots']
            cash += mkt_val - mkt_val * comm

        # Calculate results
        if wf_mode:
            test_trades = trades
            n_days_test = test_end_di - test_start_di
        else:
            test_trades = trades
            n_days_test = ND - start_di

        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in test_trades]) * 100 if test_trades else 0
        n_trades = len(test_trades)

        # Max drawdown from trade PnLs
        if test_trades:
            cum_pnl = np.cumsum([t['pnl_pct'] for t in test_trades])
            running_max = np.maximum.accumulate(cum_pnl)
            dd = cum_pnl - running_max
            mdd = np.min(dd) if len(dd) > 0 else 0
        else:
            mdd = 0

        return {
            'ann': ann, 'wr': wr, 'n': n_trades, 'mdd': mdd,
            'final_cash': cash, 'n_days': n_days_test,
        }

    # ================================================================
    # SWEEP CONFIGURATIONS
    # ================================================================
    print("\n[Sweep] Testing configurations...", flush=True)

    configs = []
    config_id = 0

    thresholds = [0.001, 0.003, 0.005, 0.01]
    top_ns = [1, 3]

    # V74 baseline (long-only, close-to-close)
    for thresh in thresholds:
        for tn in top_ns:
            config_id += 1
            configs.append({
                'id': config_id, 'signal': 'V74_baseline',
                'threshold': thresh, 'top_n': tn,
                'comm': COMM,
                'label': f"V74_T{thresh}_TN{tn}",
            })

    # Signal A: Pure overnight gap divergence (open-to-close)
    for thresh in thresholds:
        for tn in top_ns:
            config_id += 1
            configs.append({
                'id': config_id, 'signal': 'A_overnight_div',
                'threshold': thresh, 'top_n': tn,
                'comm': COMM,
                'label': f"A_ONdiv_T{thresh}_TN{tn}",
            })

    # Signal B: V74 with overnight filter (close-to-close)
    for thresh in thresholds:
        for tn in top_ns:
            config_id += 1
            configs.append({
                'id': config_id, 'signal': 'B_overnight_filter',
                'threshold': thresh, 'top_n': tn,
                'comm': COMM,
                'label': f"B_ONfilt_T{thresh}_TN{tn}",
            })

    # Signal C: Intraday continuation (close-to-close)
    for thresh in thresholds:
        for tn in top_ns:
            config_id += 1
            configs.append({
                'id': config_id, 'signal': 'C_continuation',
                'threshold': thresh, 'top_n': tn,
                'comm': COMM,
                'label': f"C_Cont_T{thresh}_TN{tn}",
            })

    # Signal D: Gap reversal (open-to-close)
    for thresh in thresholds:
        for tn in top_ns:
            config_id += 1
            configs.append({
                'id': config_id, 'signal': 'D_gap_reversal',
                'threshold': thresh, 'top_n': tn,
                'comm': COMM,
                'label': f"D_Rev_T{thresh}_TN{tn}",
            })

    # Signal E: Decomposed momentum (close-to-close)
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        beta = round(1.0 - alpha, 1)
        for thresh in [0.003, 0.005]:
            for tn in [1, 3]:
                config_id += 1
                configs.append({
                    'id': config_id, 'signal': 'E_decomposed',
                    'threshold': thresh, 'top_n': tn,
                    'alpha': alpha, 'beta': beta,
                    'comm': COMM,
                    'label': f"E_a{alpha:.1f}b{beta:.1f}_T{thresh}_TN{tn}",
                })

    print(f"  Total configs: {len(configs)}")

    # Run all configs (full period)
    results = []
    for i, cfg in enumerate(configs):
        r = run_backtest(cfg)
        if r:
            r['config'] = cfg
            r['label'] = cfg['label']
            results.append(r)
        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(configs)} done", flush=True)

    # Sort by annual return
    results.sort(key=lambda x: -x['ann'])

    # ================================================================
    # FULL-PERIOD RESULTS
    # ================================================================
    print("\n" + "=" * 110)
    print("  FULL-PERIOD RESULTS (Top 30)")
    print("=" * 110)
    print(f"  {'#':>3} | {'Label':<35} | {'Ann':>10} | {'WR':>6} | {'MDD':>7} | {'N':>5}")
    print("-" * 80)
    for i, r in enumerate(results[:30]):
        print(f"  {i+1:>3} | {r['label']:<35} | {r['ann']:>+9.1f}% | {r['wr']:>5.1f}% | {r['mdd']:>+6.1f}% | {r['n']:>5}")

    # ================================================================
    # RESULTS BY SIGNAL TYPE
    # ================================================================
    print("\n" + "=" * 110)
    print("  BEST PER SIGNAL TYPE")
    print("=" * 110)
    for sig_type in ['V74_baseline', 'A_overnight_div', 'B_overnight_filter',
                     'C_continuation', 'D_gap_reversal', 'E_decomposed']:
        sig_results = [r for r in results if r['config']['signal'] == sig_type]
        if sig_results:
            best = sig_results[0]
            print(f"  {sig_type:<20} | Best Ann: {best['ann']:>+9.1f}% | WR: {best['wr']:>5.1f}% | MDD: {best['mdd']:>+6.1f}% | {best['label']}")

    # ================================================================
    # WALK-FORWARD (Top 15)
    # ================================================================
    print("\n" + "=" * 110)
    print("  WALK-FORWARD (Top 15 configs)")
    print("=" * 110)

    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]
    wf_results = []
    for i, r in enumerate(results[:15]):
        cfg = r['config']
        wf_row = {'label': cfg['label'], 'signal': cfg['signal'], 'windows': {}}
        for yr in wf_years:
            wr = run_backtest(cfg, wf_test_year=yr)
            if wr:
                wf_row['windows'][yr] = wr['ann']
        wf_results.append(wf_row)

    # Print WF table
    print(f"  {'#':>3} | {'Signal':<20} {'Config':<20} | {'Avg':>8} | ", end="")
    for yr in wf_years:
        print(f" {yr:>7} |", end="")
    print(f"  {'Pos':>4}")
    print("-" * 140)
    for i, wf in enumerate(wf_results):
        vals = [wf['windows'].get(yr, 0) for yr in wf_years]
        avg = np.mean(vals) if vals else 0
        pos = sum(1 for v in vals if v > 0)
        short_label = wf['label'][:20]
        print(f"  {i+1:>3} | {wf['signal']:<20} {short_label:<20} | {avg:>+7.1f}% |", end="")
        for v in vals:
            print(f" {v:>+7.1f}% |", end="")
        print(f"  {pos}/6")

    # ================================================================
    # WF BY SIGNAL TYPE (best config per type)
    # ================================================================
    print("\n" + "=" * 110)
    print("  WALK-FORWARD BY SIGNAL TYPE (best config per type)")
    print("=" * 110)

    wf_by_type = []
    for sig_type in ['V74_baseline', 'A_overnight_div', 'B_overnight_filter',
                     'C_continuation', 'D_gap_reversal', 'E_decomposed']:
        sig_results = [r for r in results if r['config']['signal'] == sig_type]
        if sig_results:
            cfg = sig_results[0]['config']
            wf_row = {'label': cfg['label'], 'signal': sig_type, 'windows': {}}
            for yr in wf_years:
                wr = run_backtest(cfg, wf_test_year=yr)
                if wr:
                    wf_row['windows'][yr] = wr['ann']
            wf_by_type.append(wf_row)

    print(f"  {'Signal':<20} | {'Config':<25} | {'Avg':>8} | ", end="")
    for yr in wf_years:
        print(f" {yr:>7} |", end="")
    print(f"  {'Pos':>4}")
    print("-" * 140)
    for wf in wf_by_type:
        vals = [wf['windows'].get(yr, 0) for yr in wf_years]
        avg = np.mean(vals) if vals else 0
        pos = sum(1 for v in vals if v > 0)
        print(f"  {wf['signal']:<20} | {wf['label']:<25} | {avg:>+7.1f}% |", end="")
        for v in vals:
            print(f" {v:>+7.1f}% |", end="")
        print(f"  {pos}/6")

    # ================================================================
    # SIGNAL E: ALPHA SENSITIVITY
    # ================================================================
    print("\n" + "=" * 110)
    print("  SIGNAL E: ALPHA SENSITIVITY (best threshold/tn per alpha)")
    print("=" * 110)
    print(f"  {'Alpha':>5} | {'Beta':>5} | {'Best Ann':>10} | {'WR':>6} | {'Config':<30}")
    print("-" * 70)
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        beta = round(1.0 - alpha, 1)
        e_results = [r for r in results
                     if r['config']['signal'] == 'E_decomposed'
                     and r['config'].get('alpha', -1) == alpha]
        if e_results:
            best = e_results[0]
            print(f"  {alpha:>5.1f} | {beta:>5.1f} | {best['ann']:>+9.1f}% | {best['wr']:>5.1f}% | {best['label']}")
        else:
            print(f"  {alpha:>5.1f} | {beta:>5.1f} | {'N/A':>10} |")

    # ================================================================
    # KEY COMPARISONS
    # ================================================================
    print("\n" + "=" * 110)
    print("  KEY COMPARISONS")
    print("=" * 110)

    v74_best = [r for r in results if r['config']['signal'] == 'V74_baseline']
    a_best = [r for r in results if r['config']['signal'] == 'A_overnight_div']
    b_best = [r for r in results if r['config']['signal'] == 'B_overnight_filter']
    c_best = [r for r in results if r['config']['signal'] == 'C_continuation']
    d_best = [r for r in results if r['config']['signal'] == 'D_gap_reversal']
    e_best = [r for r in results if r['config']['signal'] == 'E_decomposed']

    if v74_best:
        print(f"  V74 baseline:       {v74_best[0]['ann']:>+9.1f}%  {v74_best[0]['label']}")
    if a_best:
        print(f"  A overnight div:    {a_best[0]['ann']:>+9.1f}%  {a_best[0]['label']}")
    if b_best:
        print(f"  B overnight filter: {b_best[0]['ann']:>+9.1f}%  {b_best[0]['label']}")
    if c_best:
        print(f"  C continuation:     {c_best[0]['ann']:>+9.1f}%  {c_best[0]['label']}")
    if d_best:
        print(f"  D gap reversal:     {d_best[0]['ann']:>+9.1f}%  {d_best[0]['label']}")
    if e_best:
        print(f"  E decomposed:       {e_best[0]['ann']:>+9.1f}%  {e_best[0]['label']}")

    # Does overnight decomposition improve V74?
    print()
    if v74_best:
        v74_ann = v74_best[0]['ann']
        if b_best:
            print(f"  B (overnight filter) vs V74: {b_best[0]['ann'] - v74_ann:>+8.1f}% improvement")
        if e_best:
            print(f"  E (decomposed) vs V74:       {e_best[0]['ann'] - v74_ann:>+8.1f}% improvement")
        if a_best:
            print(f"  A (overnight div) vs V74:    {a_best[0]['ann'] - v74_ann:>+8.1f}% improvement")
        if c_best:
            print(f"  C (continuation) vs V74:     {c_best[0]['ann'] - v74_ann:>+8.1f}% improvement")
        if d_best:
            print(f"  D (gap reversal) vs V74:     {d_best[0]['ann'] - v74_ann:>+8.1f}% improvement")

    print(f"\n  Total time: {time.time()-t_start:.1f}s")
    print("=" * 110)


if __name__ == '__main__':
    main()
