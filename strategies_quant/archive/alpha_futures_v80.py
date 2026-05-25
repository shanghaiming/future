"""
Alpha Futures V80 — Supply Chain Cascade: Upstream -> Downstream Lag Exploitation
==================================================================================
V74 champion uses within-group mean reversion (LB=1) for +2185%.
V80 hypothesis: Upstream commodity moves first, downstream follows with a lag.
With 1-day hold, we capture this supply chain cascade.

Signals tested:
  A_cascade:    Upstream 1-day return > threshold -> buy downstream
  B_divergence: Upstream_return - downstream_return > threshold -> buy downstream
  C_hybrid:     V74 group signal + supply chain signal, trade when either fires
  D_spread_mom: Spread momentum (log-spread change) -> pair trade
  V74_baseline: Reproduce V74 as control

Walk-forward: 6 windows (2020-2025), reset cash at test year start.
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

# ── Supply chain pairs (upstream -> downstream) ─────────────────────
SUPPLY_CHAIN = [
    ('ifi',  'rbfi'),   # iron ore -> rebar
    ('ifi',  'hcfi'),   # iron ore -> hot coil
    ('jfi',  'jmfi'),   # coke -> coking coal
    ('scfi', 'mafi'),   # crude oil -> methanol
    ('scfi', 'bfi'),    # crude oil -> asphalt
    ('scfi', 'fufi'),   # crude oil -> fuel oil
    ('scfi', 'ppfi'),   # crude oil -> polypropylene
    ('scfi', 'egfi'),   # crude oil -> ethylene glycol
    ('scfi', 'vfi'),    # crude oil -> PVC
    ('scfi', 'pgfi'),   # crude oil -> LPG
    ('scfi', 'tafi'),   # crude oil -> PTA
    ('mfi',  'yfi'),    # soybean meal -> soybean oil
    ('mfi',  'afi'),    # soybean meal -> soybean
    ('cfi',  'csfi'),   # corn -> corn starch
]

# ── V74 group map (original 25 commodities) ─────────────────────────
GROUP_MAP = {
    'rbfi': 'ferrous', 'hcfi': 'ferrous', 'ifi': 'ferrous', 'jfi': 'ferrous', 'jmfi': 'ferrous',
    'cufi': 'nonferrous', 'alfi': 'nonferrous', 'znfi': 'nonferrous', 'nifi': 'nonferrous',
    'aufi': 'precious', 'agfi': 'precious',
    'afi': 'oils', 'mfi': 'oils', 'yfi': 'oils', 'pfi': 'oils', 'cfi': 'oils',
    'scfi': 'energy', 'mafi': 'energy', 'bfi': 'energy', 'fufi': 'energy', 'pgfi': 'energy',
    'ppfi': 'chemical', 'vfi': 'chemical', 'egfi': 'chemical', 'srfi': 'chemical',
}


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def main():
    print("=" * 100)
    print("Alpha Futures V80 -- Supply Chain Cascade: Upstream -> Downstream Lag")
    print("=" * 100)

    # ── Load data ────────────────────────────────────────────────────
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    sym_idx = {s: i for i, s in enumerate(syms)}

    # ── Precompute 1-day returns ─────────────────────────────────────
    print("\n[Signals] Computing 1-day returns...", flush=True)
    t0 = time.time()
    ret1 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            cn = C[si, di]
            cp = C[si, di - 1]
            if not np.isnan(cn) and not np.isnan(cp) and cp > 0:
                ret1[si, di] = (cn - cp) / cp

    # ── Precompute V74 group mean reversion signal ───────────────────
    print("[Signals] Computing V74 group mean reversion (LB=1)...", flush=True)
    # Build group members
    gm_map = {}
    for si in range(NS):
        g = GROUP_MAP.get(syms[si])
        if g:
            gm_map.setdefault(g, []).append(si)

    grp_avg = np.full((NS, ND), np.nan)
    for grp, members in gm_map.items():
        for di in range(1, ND):
            for sj in members:
                vals = [ret1[sk, di] for sk in members
                        if sk != sj and not np.isnan(ret1[sk, di])]
                if vals:
                    grp_avg[sj, di] = np.mean(vals)
    print(f"  Signals computed ({time.time()-t0:.1f}s)")

    # ── Validate supply chain pairs against available symbols ────────
    valid_pairs = []
    for up_sym, dn_sym in SUPPLY_CHAIN:
        if up_sym in sym_idx and dn_sym in sym_idx:
            valid_pairs.append((sym_idx[up_sym], sym_idx[dn_sym], up_sym, dn_sym))
        else:
            print(f"  WARNING: pair ({up_sym}, {dn_sym}) not in sym_set, skipping")
    print(f"\n  Valid supply chain pairs: {len(valid_pairs)}")
    for up_i, dn_i, up_s, dn_s in valid_pairs:
        print(f"    {up_s:>4s} -> {dn_s}")

    # ══════════════════════════════════════════════════════════════════
    # BACKTEST ENGINE (unified for all signals)
    # ══════════════════════════════════════════════════════════════════
    def run_backtest(config, wf_test_year=None):
        """
        Config:
            signal: 'A_cascade' | 'B_divergence' | 'C_hybrid' | 'D_spread_mom' | 'V74_baseline'
            threshold: float
            top_n: 1 | 3
            legs: 'single' | 'pair'  (for pair signals A,B,D: trade 1 leg or both)
            comm: float
        """
        sig_type = config['signal']
        threshold = config['threshold']
        top_n = config['top_n']
        legs = config.get('legs', 'single')
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

        cash = float(CASH0)
        positions = []
        trades = []

        for di in range(start_di, end_di):
            # Reset cash at test window start (WF mode)
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                positions = []

            # ── Close positions held 1 day ───────────────────────────
            closed = []
            for pos in positions:
                if di - pos['entry_di'] >= 1:
                    cn = C[pos['si'], di]
                    if np.isnan(cn) or cn <= 0:
                        cn = pos['entry']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    mkt_val = cn * mult * abs(pos['lots'])
                    cash += mkt_val - mkt_val * comm
                    pnl = (cn - pos['entry']) * mult * pos['lots'] * pos['dir']
                    invested = pos['entry'] * mult * abs(pos['lots'])
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

            # ── Generate signals ─────────────────────────────────────
            candidates = []  # list of (si, score, direction, sym)

            # Signal A: Upstream momentum cascade
            #   If upstream 1-day return > threshold -> buy downstream
            #   If upstream 1-day return < -threshold -> short downstream
            if sig_type == 'A_cascade':
                for up_i, dn_i, up_s, dn_s in valid_pairs:
                    up_ret = ret1[up_i, di]
                    if np.isnan(up_ret):
                        continue
                    # Check downstream not already held
                    if any(p['si'] == dn_i for p in positions):
                        continue
                    cdn = C[dn_i, di]
                    if np.isnan(cdn) or cdn <= 0:
                        continue
                    if up_ret > threshold:
                        candidates.append((dn_i, up_ret, 1, dn_s))
                    elif up_ret < -threshold:
                        candidates.append((dn_i, -up_ret, -1, dn_s))

            # Signal B: Upstream-downstream divergence
            #   divergence = upstream_return - downstream_return
            #   If div > threshold -> buy downstream (lagging, will catch up)
            #   If div < -threshold -> buy upstream (lagging, will catch up)
            elif sig_type == 'B_divergence':
                for up_i, dn_i, up_s, dn_s in valid_pairs:
                    up_ret = ret1[up_i, di]
                    dn_ret = ret1[dn_i, di]
                    if np.isnan(up_ret) or np.isnan(dn_ret):
                        continue
                    div = up_ret - dn_ret

                    # Downstream lagging -> buy downstream
                    if div > threshold:
                        if not any(p['si'] == dn_i for p in positions):
                            cdn = C[dn_i, di]
                            if not np.isnan(cdn) and cdn > 0:
                                candidates.append((dn_i, div, 1, dn_s))

                    # Upstream lagging -> buy upstream
                    if div < -threshold:
                        if not any(p['si'] == up_i for p in positions):
                            cup = C[up_i, di]
                            if not np.isnan(cup) and cup > 0:
                                candidates.append((up_i, -div, 1, up_s))

                    # Pair mode: both legs
                    if legs == 'pair':
                        if abs(div) > threshold:
                            # Always trade downstream in direction of divergence
                            if div > threshold:
                                # buy downstream, short upstream
                                if not any(p['si'] == dn_i for p in positions):
                                    cdn = C[dn_i, di]
                                    if not np.isnan(cdn) and cdn > 0:
                                        candidates.append((dn_i, div, 1, dn_s))
                                if not any(p['si'] == up_i for p in positions):
                                    cup = C[up_i, di]
                                    if not np.isnan(cup) and cup > 0:
                                        candidates.append((up_i, div, -1, up_s))
                            else:
                                # short downstream, buy upstream
                                if not any(p['si'] == dn_i for p in positions):
                                    cdn = C[dn_i, di]
                                    if not np.isnan(cdn) and cdn > 0:
                                        candidates.append((dn_i, -div, -1, dn_s))
                                if not any(p['si'] == up_i for p in positions):
                                    cup = C[up_i, di]
                                    if not np.isnan(cup) and cup > 0:
                                        candidates.append((up_i, -div, 1, up_s))

            # Signal C: Hybrid (V74 group + supply chain)
            #   V74 group mean reversion fires OR supply chain cascade fires -> trade
            elif sig_type == 'C_hybrid':
                # V74 signal: group_avg - own_return > threshold -> buy
                for si in range(NS):
                    g = GROUP_MAP.get(syms[si])
                    if not g:
                        continue
                    own = ret1[si, di]
                    ga = grp_avg[si, di]
                    if np.isnan(own) or np.isnan(ga):
                        continue
                    div = ga - own
                    cc = C[si, di]
                    if np.isnan(cc) or cc <= 0:
                        continue
                    if any(p['si'] == si for p in positions):
                        continue
                    if abs(div) > threshold:
                        direction = 1 if div > 0 else -1
                        candidates.append((si, abs(div), direction, syms[si]))

                # Supply chain cascade (signal A logic)
                for up_i, dn_i, up_s, dn_s in valid_pairs:
                    up_ret = ret1[up_i, di]
                    if np.isnan(up_ret):
                        continue
                    if any(p['si'] == dn_i for p in positions):
                        continue
                    cdn = C[dn_i, di]
                    if np.isnan(cdn) or cdn <= 0:
                        continue
                    if up_ret > threshold:
                        candidates.append((dn_i, up_ret, 1, dn_s))
                    elif up_ret < -threshold:
                        candidates.append((dn_i, -up_ret, -1, dn_s))

            # Signal D: Spread momentum (pair trade)
            #   spread = log(C_downstream) - log(C_upstream)
            #   spread_mom = spread[today] - spread[yesterday]
            #   If spread narrowing -> buy downstream, short upstream
            #   If spread widening -> short downstream, buy upstream
            elif sig_type == 'D_spread_mom':
                for up_i, dn_i, up_s, dn_s in valid_pairs:
                    cdn0 = C[dn_i, di]
                    cup0 = C[up_i, di]
                    if di < 1:
                        continue
                    cdn1 = C[dn_i, di - 1]
                    cup1 = C[up_i, di - 1]
                    if any(np.isnan(x) or x <= 0 for x in [cdn0, cup0, cdn1, cup1]):
                        continue

                    spread0 = np.log(cdn0) - np.log(cup0)
                    spread1 = np.log(cdn1) - np.log(cup1)
                    spread_mom = spread0 - spread1

                    # Spread narrowing: downstream catching up -> buy downstream
                    if spread_mom < -threshold:
                        if not any(p['si'] == dn_i for p in positions):
                            candidates.append((dn_i, -spread_mom, 1, dn_s))
                        if legs == 'pair' and not any(p['si'] == up_i for p in positions):
                            candidates.append((up_i, -spread_mom, -1, up_s))

                    # Spread widening: upstream pulling away -> buy upstream
                    elif spread_mom > threshold:
                        if not any(p['si'] == up_i for p in positions):
                            candidates.append((up_i, spread_mom, 1, up_s))
                        if legs == 'pair' and not any(p['si'] == dn_i for p in positions):
                            candidates.append((dn_i, spread_mom, -1, dn_s))

            # V74 baseline: group mean reversion, LB=1
            elif sig_type == 'V74_baseline':
                for si in range(NS):
                    g = GROUP_MAP.get(syms[si])
                    if not g:
                        continue
                    own = ret1[si, di]
                    ga = grp_avg[si, di]
                    if np.isnan(own) or np.isnan(ga):
                        continue
                    div = ga - own
                    cc = C[si, di]
                    if np.isnan(cc) or cc <= 0:
                        continue
                    if any(p['si'] == si for p in positions):
                        continue
                    if abs(div) > threshold:
                        direction = 1 if div > 0 else -1
                        candidates.append((si, abs(div), direction, syms[si]))

            if not candidates:
                continue

            # Sort by score descending
            candidates.sort(key=lambda x: -x[1])

            # Open positions (up to top_n slots)
            n_slots = top_n - len(positions)
            for si, score, direction, sym in candidates[:max(0, n_slots)]:
                c = C[si, di]
                if np.isnan(c) or c <= 0:
                    continue
                mult = MULT.get(sym, DEF_MULT)
                notional = c * mult
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
                    'si': si, 'entry': c, 'entry_di': di,
                    'lots': lots, 'dir': direction, 'sym': sym,
                })

        # Close remaining
        for pos in positions:
            ae = ND - 1
            cn = C[pos['si'], ae]
            if np.isnan(cn) or cn <= 0:
                cn = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = cn * mult * abs(pos['lots'])
            cash += mkt_val - mkt_val * comm

        # Results
        wf_mode = wf_test_year is not None
        n_days_test = (test_end_di - test_start_di) if wf_mode else (ND - start_di)
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0

        # Max drawdown from equity curve (approximate from trades)
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

    # ══════════════════════════════════════════════════════════════════
    # BUILD CONFIGURATIONS
    # ══════════════════════════════════════════════════════════════════
    print("\n[Sweep] Building configurations...", flush=True)
    configs = []
    cid = 0

    for sig in ['A_cascade', 'B_divergence', 'C_hybrid', 'D_spread_mom', 'V74_baseline']:
        for thresh in [0.001, 0.003, 0.005, 0.01]:
            for tn in [1, 3]:
                for leg in ['single', 'pair']:
                    # V74_baseline doesn't use legs
                    if sig == 'V74_baseline' and leg == 'pair':
                        continue
                    # A_cascade only trades downstream, pair mode not applicable in same way
                    # But we can still skip pair for A_cascade to reduce grid
                    if sig == 'A_cascade' and leg == 'pair':
                        continue
                    # C_hybrid uses both signals, legs not directly applicable
                    if sig == 'C_hybrid' and leg == 'pair':
                        continue
                    cid += 1
                    cfg = {
                        'id': cid,
                        'signal': sig,
                        'threshold': thresh,
                        'top_n': tn,
                        'legs': leg,
                        'comm': COMM,
                        'label': f"{sig}_T{thresh}_TN{tn}_{leg}",
                    }
                    configs.append(cfg)

    print(f"  Total configs: {len(configs)}")

    # ══════════════════════════════════════════════════════════════════
    # RUN FULL-PERIOD BACKTEST
    # ══════════════════════════════════════════════════════════════════
    print("\n[Backtest] Running full-period sweep...", flush=True)
    results = []
    for i, cfg in enumerate(configs):
        r = run_backtest(cfg)
        if r:
            r['config'] = cfg
            r['label'] = cfg['label']
            results.append(r)
        if (i + 1) % 10 == 0 or i == len(configs) - 1:
            print(f"  ... {i+1}/{len(configs)} done", flush=True)

    results.sort(key=lambda x: -x['ann'])

    # Print top 20
    print("\n" + "=" * 120)
    print("  FULL-PERIOD RESULTS (Top 20)")
    print("=" * 120)
    print(f"  {'#':>3} | {'Label':<50} | {'Ann':>8} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7}")
    print("-" * 110)
    for i, r in enumerate(results[:20]):
        print(f"  {i+1:>3} | {r['label']:<50} | {r['ann']:>+7.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}%")

    # ══════════════════════════════════════════════════════════════════
    # WALK-FORWARD (Top 10)
    # ══════════════════════════════════════════════════════════════════
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Also ensure we have at least 1 config per signal type in WF
    best_per_sig = {}
    for r in results:
        s = r['config']['signal']
        if s not in best_per_sig:
            best_per_sig[s] = r

    # Take top 10 + best of each signal type (dedup)
    wf_configs = results[:10]
    for s, r in best_per_sig.items():
        if r['config'] not in [w['config'] for w in wf_configs]:
            wf_configs.append(r)

    print(f"\n{'=' * 140}")
    print(f"  WALK-FORWARD ({len(wf_configs)} configs)")
    print(f"{'=' * 140}")

    header = f"  {'#':>3} | {'Config':<50} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>7} |"
    header += f" {'Pos':>4} | {'MDD':>7}"
    print(header)
    print("-" * 140)

    wf_rows = []
    for i, r in enumerate(wf_configs):
        cfg = r['config']
        wf_row = {'label': cfg['label'], 'signal': cfg['signal'], 'windows': {}, 'mdd': {}}
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

        row_str = f"  {i+1:>3} | {wf_row['label']:<50} | {avg:>+7.1f}% |"
        for v in vals:
            row_str += f" {v:>+7.1f}% |"
        row_str += f" {pos}/6 | {avg_mdd:>6.1f}%"
        print(row_str)

    # ══════════════════════════════════════════════════════════════════
    # SIGNAL COMPARISON
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 120}")
    print("  SIGNAL COMPARISON (Best per signal type, full period)")
    print(f"{'=' * 120}")
    print(f"  {'Signal':<20} | {'Ann':>8} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | Best Config")
    print("-" * 120)

    for sig in ['A_cascade', 'B_divergence', 'C_hybrid', 'D_spread_mom', 'V74_baseline']:
        sig_results = [r for r in results if r['config']['signal'] == sig]
        if sig_results:
            best = sig_results[0]
            print(f"  {sig:<20} | {best['ann']:>+7.1f}% | {best['wr']:>5.1f}% | {best['n']:>5} | {best['avg_pnl']:>+6.3f}% | {best['mdd']:>6.1f}% | {best['label']}")

    # WF comparison per signal
    print(f"\n{'=' * 120}")
    print("  WALK-FORWARD COMPARISON (Best per signal type)")
    print(f"{'=' * 120}")
    header2 = f"  {'Signal':<20} | {'WF Avg':>8} |"
    for yr in wf_years:
        header2 += f" {yr:>7} |"
    header2 += f" {'Pos':>4}"
    print(header2)
    print("-" * 120)

    for sig in ['A_cascade', 'B_divergence', 'C_hybrid', 'D_spread_mom', 'V74_baseline']:
        # Find WF result for this signal
        wf_match = [w for w in wf_rows if w['signal'] == sig]
        if wf_match:
            wf = wf_match[0]
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            avg = np.mean(vals) if vals else 0
            pos = sum(1 for v in vals if v > 0)
            row_str = f"  {sig:<20} | {avg:>+7.1f}% |"
            for v in vals:
                row_str += f" {v:>+7.1f}% |"
            row_str += f" {pos}/6"
            print(row_str)

    # ── Detailed best config analysis ────────────────────────────────
    print(f"\n{'=' * 120}")
    print("  CHAMPION ANALYSIS")
    print(f"{'=' * 120}")
    if results:
        champ = results[0]
        print(f"  Best full-period: {champ['label']}")
        print(f"    Annual return: {champ['ann']:>+.1f}%")
        print(f"    Win rate:      {champ['wr']:>.1f}%")
        print(f"    Trades:        {champ['n']}")
        print(f"    Avg trade PnL: {champ['avg_pnl']:>+.3f}%")
        print(f"    Max DD:        {champ['mdd']:>.1f}%")

        # Compare V74 baseline vs best supply chain signal
        v74_results = [r for r in results if r['config']['signal'] == 'V74_baseline']
        sc_results = [r for r in results if r['config']['signal'] != 'V74_baseline']
        if v74_results and sc_results:
            v74_best = v74_results[0]
            sc_best = sc_results[0]
            print(f"\n  V74 baseline best:  {v74_best['ann']:>+8.1f}%  ({v74_best['label']})")
            print(f"  Supply chain best:  {sc_best['ann']:>+8.1f}%  ({sc_best['label']})")
            diff = sc_best['ann'] - v74_best['ann']
            if diff > 0:
                print(f"  >>> Supply chain ADDS +{diff:.1f}% alpha over V74 baseline <<<")
            else:
                print(f"  >>> Supply chain DOES NOT add alpha ({diff:+.1f}%) <<<")

    print(f"\n  Total time: {time.time()-t_start:.1f}s")
    print("=" * 100)


if __name__ == '__main__':
    main()
