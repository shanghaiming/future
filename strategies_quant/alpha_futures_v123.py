"""
Alpha Futures V123 -- COMBINE BEST OF V121 + V122 (Next-Open Execution)
=======================================================================
V121 found: ROC*Z ranking + ROC improving filter gives +333.5%
V122 found: 60-day Z-score window gives +331.6%
Both use ROC(5)>1% as base. COMBINE them.

ALL signals use NEXT-OPEN execution: signal at close di, entry at O[si, di+1].
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


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def rolling_zscore(arr, window):
    """Compute rolling Z-score: (value - rolling_mean) / rolling_std."""
    n = len(arr)
    z = np.full(n, np.nan)
    for i in range(window, n):
        w = arr[i - window:i]
        valid = w[~np.isnan(w)]
        if len(valid) >= max(window // 2, 5):
            m = np.mean(valid)
            s = np.std(valid, ddof=0)
            if s > 1e-10 and not np.isnan(arr[i]):
                z[i] = (arr[i] - m) / s
    return z


def main():
    print("=" * 150)
    print("  Alpha Futures V123 -- COMBINE BEST OF V121 + V122 (Next-Open Execution)")
    print("=" * 150)
    print(f"  V121: ROC*Z ranking + ROC improving = +333.5%")
    print(f"  V122: 60-day Z-score window = +331.6%")
    print(f"  Goal: Combine to find 400%+ config")

    # -- Load data -------------------------------------------------
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")
    print(f"  MIN_TRAIN={MIN_TRAIN}, CASH0={CASH0:,}")

    # ================================================================
    # PRECOMPUTE INDICATORS
    # ================================================================
    print("\n[Precompute] Daily returns, ROC(5), Z-scores...", flush=True)
    t0 = time.time()

    # Daily returns
    RET = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100

    # ROC(5) in percent
    ROC5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        ROC5[si] = talib.ROC(c, timeperiod=5)

    # Z-score of daily returns (20-day rolling) -- V120/V121 style
    ZSCORE_20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = RET[si, di-20:di]
            valid = rets[~np.isnan(rets)]
            if len(valid) < 10:
                continue
            mean_r = np.mean(valid)
            std_r = np.std(valid, ddof=1)
            if std_r > 0 and not np.isnan(RET[si, di]):
                ZSCORE_20[si, di] = (RET[si, di] - mean_r) / std_r

    # Z-score of daily returns (60-day rolling) -- V122 champion style
    ZSCORE_60 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(61, ND):
            rets = RET[si, di-60:di]
            valid = rets[~np.isnan(rets)]
            if len(valid) < 20:
                continue
            mean_r = np.mean(valid)
            std_r = np.std(valid, ddof=1)
            if std_r > 0 and not np.isnan(RET[si, di]):
                ZSCORE_60[si, di] = (RET[si, di] - mean_r) / std_r

    # Cross-sectional Z-score of ROC5
    Z_CROSS = np.full((NS, ND), np.nan)
    for di in range(ND):
        vals = ROC5[:, di]
        valid = vals[~np.isnan(vals)]
        if len(valid) >= 10:
            m = np.mean(valid)
            s = np.std(valid, ddof=0)
            if s > 1e-10:
                for si in range(NS):
                    if not np.isnan(vals[si]):
                        Z_CROSS[si, di] = (vals[si] - m) / s

    print(f"  All indicators computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # GENERIC BACKTEST ENGINE
    # ================================================================
    def backtest(cfg, start_di=MIN_TRAIN, end_di=None, return_trades=False,
                 equity_mgmt=None):
        """
        cfg dict keys:
          roc_thresh, z60_thresh, z20_thresh, z_cross_thresh,
          rank_mode: 'roc', 'roc_z60', 'roc_z20'
          filter_roc_improving: bool
          hold: int
          top_n: int
        equity_mgmt: None or dict with keys 'levels' like [(1.5, 0.5), (2.0, 0.33)]
        """
        if end_di is None:
            end_di = ND

        roc_thresh = cfg.get('roc_thresh', 1.0)
        z60_thresh = cfg.get('z60_thresh', 1.0)
        z20_thresh = cfg.get('z20_thresh', 0.0)
        z_cross_thresh = cfg.get('z_cross_thresh', 0.0)
        rank_mode = cfg.get('rank_mode', 'roc')
        filter_roc_imp = cfg.get('filter_roc_improving', False)
        hold = cfg.get('hold', 1)
        top_n = cfg.get('top_n', 1)
        label = cfg.get('label', '')

        cash = float(CASH0)
        initial_cash = float(CASH0)
        positions = []
        trades = []
        daily_equity = []

        for di in range(start_di, end_di - 1):
            # Track daily equity
            port_val = cash
            for pos in positions:
                cp = C[pos['si'], di]
                if not np.isnan(cp) and cp > 0:
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    port_val += cp * mult * pos['lots'] - cp * mult * abs(pos['lots']) * COMM
            daily_equity.append(port_val)

            # Close positions whose hold period is up
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
                        'pnl': pnl, 'pnl_pct': pnl_pct,
                        'entry_di': pos['entry_di'], 'exit_di': di,
                        'sym': pos['sym'],
                    })
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            if len(positions) >= top_n:
                continue

            # Determine capital fraction based on equity management
            cap_fraction = 1.0
            if equity_mgmt:
                eq_ratio = port_val / initial_cash
                for threshold, frac in sorted(equity_mgmt.get('levels', []), reverse=True):
                    if eq_ratio >= threshold:
                        cap_fraction = frac
                        break

            # Generate entry signals
            entry_di = di + 1
            if entry_di >= end_di:
                continue

            candidates = []
            for s in range(NS):
                roc = ROC5[s, di]
                z60 = ZSCORE_60[s, di]
                z20 = ZSCORE_20[s, di]
                z_cr = Z_CROSS[s, di]

                if np.isnan(roc):
                    continue
                if roc <= roc_thresh:
                    continue
                if not np.isnan(z60) and z60_thresh > 0 and z60 <= z60_thresh:
                    continue
                if not np.isnan(z20) and z20_thresh > 0 and z20 <= z20_thresh:
                    continue
                if not np.isnan(z_cr) and z_cross_thresh > 0 and z_cr <= z_cross_thresh:
                    continue
                # If threshold > 0 but z is NaN, skip
                if z60_thresh > 0 and np.isnan(z60):
                    continue
                if z20_thresh > 0 and np.isnan(z20):
                    continue
                if z_cross_thresh > 0 and np.isnan(z_cr):
                    continue

                # ROC improving filter
                if filter_roc_imp:
                    roc_prev = ROC5[s, di-1]
                    if np.isnan(roc_prev) or roc <= roc_prev:
                        continue

                ep = O[s, entry_di]
                if np.isnan(ep) or ep <= 0:
                    continue
                if any(p['si'] == s for p in positions):
                    continue

                # Ranking score
                if rank_mode == 'roc_z60':
                    sc = roc * z60 if not np.isnan(roc) and not np.isnan(z60) else 0
                elif rank_mode == 'roc_z20':
                    sc = roc * z20 if not np.isnan(roc) and not np.isnan(z20) else 0
                else:
                    sc = roc
                candidates.append((sc, s, ep, roc, z60))

            if not candidates:
                continue

            candidates.sort(key=lambda x: -x[0])
            n_slots = top_n - len(positions)
            usable_cash = cash * cap_fraction
            cap_per_slot = usable_cash / max(1, n_slots)

            for sc_val, s, price, roc_val, zs_val in candidates[:max(0, n_slots)]:
                sym = syms[s]
                mult = MULT.get(sym, DEF_MULT)
                contracts = max(1, int(cap_per_slot * 0.95 / (price * mult * (1 + COMM))))
                cost_in = price * mult * contracts * (1 + COMM)
                if cost_in > cash:
                    contracts = int(cash * 0.9 / (price * mult * (1 + COMM)))
                    cost_in = price * mult * contracts * (1 + COMM) if contracts > 0 else 0
                if contracts <= 0 or cost_in <= 0 or cost_in > cash:
                    continue
                cash -= cost_in
                positions.append({
                    'si': s, 'entry_price': price, 'entry_di': entry_di,
                    'lots': contracts, 'dir': 1, 'sym': sym,
                    'hold_days': hold,
                })

        # Close remaining positions
        for pos in positions:
            ae = end_di - 1
            exit_price = C[pos['si'], min(ae, ND-1)]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos['entry_price']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = exit_price * mult * abs(pos['lots'])
            cash += mkt_val - mkt_val * COMM
            pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
            invested = pos['entry_price'] * mult * abs(pos['lots'])
            pnl_pct = pnl / invested * 100 if invested > 0 else 0
            trades.append({
                'pnl': pnl, 'pnl_pct': pnl_pct,
                'entry_di': pos['entry_di'], 'exit_di': ae,
                'sym': pos['sym'],
            })

        n_days_test = end_di - start_di
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0

        # Max drawdown from daily equity
        if daily_equity:
            eq_arr = np.array(daily_equity)
            peak_arr = np.maximum.accumulate(eq_arr)
            dd_arr = (eq_arr - peak_arr) / peak_arr * 100
            mdd = np.min(dd_arr)
        else:
            mdd = 0.0

        return {
            'ann': ann, 'wr': wr, 'n': n_trades, 'avg_pnl': avg_pnl,
            'final_cash': cash, 'n_days': n_days_test, 'mdd': mdd,
            'label': label,
        }

    # ================================================================
    # HELPER: Walk-forward by year
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    def walk_forward(cfg, equity_mgmt=None):
        """Run walk-forward by year, return dict of {year: result}."""
        wf = {}
        for yr in wf_years:
            ts = te = None
            for di in range(ND):
                if dates[di].year == yr and ts is None:
                    ts = di
                if dates[di].year == yr + 1 and te is None:
                    te = di
            if ts is None:
                wf[yr] = None
                continue
            if te is None:
                te = ND
            r = backtest(cfg, start_di=ts, end_di=te, equity_mgmt=equity_mgmt)
            wf[yr] = r
        return wf

    def print_wf(label, wf):
        vals = {yr: wf[yr]['ann'] if wf[yr] else 0 for yr in wf_years}
        avg = np.mean(list(vals.values()))
        pos = sum(1 for v in vals.values() if v > 0)
        mdds = [wf[yr]['mdd'] for yr in wf_years if wf[yr]]
        avg_mdd = np.mean(mdds) if mdds else 0
        row = f"  {label:<50} | {avg:>+8.1f}% |"
        for yr in wf_years:
            v = vals[yr]
            row += f" {v:>+8.1f}% |"
        row += f" {pos}/6 | {avg_mdd:>6.1f}%"
        print(row)
        return avg, pos

    # ================================================================
    # BUILD ALL CONFIGS (A through J)
    # ================================================================
    print(f"\n[Config] Building test configurations A-J...", flush=True)
    configs = []

    # A) 60d Z + ROC*Z ranking + ROC improving
    configs.append({
        'label': 'A) 60d Z + ROC*Z rank + ROC improving',
        'roc_thresh': 1.0, 'z60_thresh': 1.0, 'z20_thresh': 0, 'z_cross_thresh': 0,
        'rank_mode': 'roc_z60', 'filter_roc_improving': True, 'hold': 1, 'top_n': 1,
    })

    # B) 60d Z + ROC*Z ranking (no improving filter)
    configs.append({
        'label': 'B) 60d Z + ROC*Z rank (no filter)',
        'roc_thresh': 1.0, 'z60_thresh': 1.0, 'z20_thresh': 0, 'z_cross_thresh': 0,
        'rank_mode': 'roc_z60', 'filter_roc_improving': False, 'hold': 1, 'top_n': 1,
    })

    # C) 60d Z + ROC improving only (rank by ROC)
    configs.append({
        'label': 'C) 60d Z + ROC improving (rank ROC)',
        'roc_thresh': 1.0, 'z60_thresh': 1.0, 'z20_thresh': 0, 'z_cross_thresh': 0,
        'rank_mode': 'roc', 'filter_roc_improving': True, 'hold': 1, 'top_n': 1,
    })

    # E) Combined time-series + cross-sectional Z
    configs.append({
        'label': 'E) 60d Z + Z_cross > 0.5 + ROC*Z rank',
        'roc_thresh': 1.0, 'z60_thresh': 1.0, 'z20_thresh': 0, 'z_cross_thresh': 0.5,
        'rank_mode': 'roc_z60', 'filter_roc_improving': False, 'hold': 1, 'top_n': 1,
    })

    # F) Double Z (60d + 20d)
    configs.append({
        'label': 'F) 60d Z + 20d Z (both > 1.0) + ROC*Z60',
        'roc_thresh': 1.0, 'z60_thresh': 1.0, 'z20_thresh': 1.0, 'z_cross_thresh': 0,
        'rank_mode': 'roc_z60', 'filter_roc_improving': False, 'hold': 1, 'top_n': 1,
    })

    # G) ALL enhancements combined
    configs.append({
        'label': 'G) 60d Z + ROC improving + Z_cross>0 + ROC*Z',
        'roc_thresh': 1.0, 'z60_thresh': 1.0, 'z20_thresh': 0, 'z_cross_thresh': 0.0,
        'rank_mode': 'roc_z60', 'filter_roc_improving': True, 'hold': 1, 'top_n': 1,
    })

    # Also baseline configs for reference
    configs.append({
        'label': 'REF: V121 champ (20d Z>1.5 + ROC*Z + improving)',
        'roc_thresh': 1.0, 'z60_thresh': 0, 'z20_thresh': 1.5, 'z_cross_thresh': 0,
        'rank_mode': 'roc_z20', 'filter_roc_improving': True, 'hold': 1, 'top_n': 1,
    })

    configs.append({
        'label': 'REF: V122 champ (60d Z>1.0 + rank Z)',
        'roc_thresh': 1.0, 'z60_thresh': 1.0, 'z20_thresh': 0, 'z_cross_thresh': 0,
        'rank_mode': 'roc_z60', 'filter_roc_improving': False, 'hold': 1, 'top_n': 1,
    })

    print(f"  Core configs: {len(configs)}")

    # ================================================================
    # D) 60d Z + threshold sweep (20 combinations)
    # ================================================================
    roc_sweep_d = [0.5, 0.75, 1.0, 1.25, 1.5]
    z_sweep_d = [0.75, 1.0, 1.25, 1.5]
    d_configs = []
    for rv in roc_sweep_d:
        for zv in z_sweep_d:
            d_configs.append({
                'label': f'D) ROC>{rv}% Z60>{zv} + ROC*Z + improving',
                'roc_thresh': rv, 'z60_thresh': zv, 'z20_thresh': 0, 'z_cross_thresh': 0,
                'rank_mode': 'roc_z60', 'filter_roc_improving': True, 'hold': 1, 'top_n': 1,
            })
    print(f"  D) Threshold sweep configs: {len(d_configs)}")

    # ================================================================
    # H) FINE-TUNE ROC threshold with 60d Z
    # ============================================================
    h_roc_vals = [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]
    h_configs = []
    for rv in h_roc_vals:
        h_configs.append({
            'label': f'H) ROC>{rv}% Z60>1.0 + ROC*Z + improving',
            'roc_thresh': rv, 'z60_thresh': 1.0, 'z20_thresh': 0, 'z_cross_thresh': 0,
            'rank_mode': 'roc_z60', 'filter_roc_improving': True, 'hold': 1, 'top_n': 1,
        })
    print(f"  H) ROC fine-tune configs: {len(h_configs)}")

    # ================================================================
    # I) FINE-TUNE Z_60 threshold
    # ================================================================
    i_z_vals = [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5]
    i_configs = []
    for zv in i_z_vals:
        i_configs.append({
            'label': f'I) ROC>1% Z60>{zv} + ROC*Z + improving',
            'roc_thresh': 1.0, 'z60_thresh': zv, 'z20_thresh': 0, 'z_cross_thresh': 0,
            'rank_mode': 'roc_z60', 'filter_roc_improving': True, 'hold': 1, 'top_n': 1,
        })
    print(f"  I) Z60 fine-tune configs: {len(i_configs)}")

    # ================================================================
    # RUN FULL-PERIOD BACKTEST: Core configs (A-G + REF)
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  SECTION 1: CORE CONFIGS (A-G + References)")
    print(f"{'=' * 150}")
    print(f"  {'#':>3} | {'Config':<50} | {'Ann':>10} | {'WR':>6} | {'N':>5} | {'AvgPnL':>8} | {'MDD':>8} | {'Final':>12}")
    print("-" * 150)

    core_results = []
    for i, cfg in enumerate(configs):
        r = backtest(cfg)
        core_results.append(r)
        print(f"  {i+1:>3} | {cfg['label']:<50} | {r['ann']:>+9.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+7.3f}% | {r['mdd']:>+7.1f}% | {r['final_cash']:>11,.0f}")

    # ================================================================
    # WALK-FORWARD: Core configs
    # ================================================================
    print(f"\n{'=' * 170}")
    print("  WALK-FORWARD: CORE CONFIGS")
    print(f"{'=' * 170}")
    header = f"  {'#':>3} | {'Config':<50} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>8} |"
    header += f" {'Pos':>4} | {'AvgMDD':>7}"
    print(header)
    print("-" * 170)

    for i, (cfg, r) in enumerate(zip(configs, core_results)):
        wf = walk_forward(cfg)
        avg, pos = print_wf(f"{i+1}. {cfg['label']}", wf)

    # ================================================================
    # SECTION 2: D) THRESHOLD SWEEP (ROC x Z60)
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  SECTION 2: D) THRESHOLD SWEEP (ROC x Z60, with ROC*Z rank + ROC improving)")
    print(f"{'=' * 150}")

    # Print as grid
    print(f"\n  {'ROC\\Z60':>10}", end="")
    for zv in z_sweep_d:
        print(f" | {'Z>'+str(zv):>18}", end="")
    print()
    print(f"  {'-'*110}")

    d_results = []
    for rv in roc_sweep_d:
        print(f"  {'ROC>'+str(rv)+'%':>10}", end="")
        for zv in z_sweep_d:
            cfg = {
                'label': f'D) ROC>{rv}% Z60>{zv}',
                'roc_thresh': rv, 'z60_thresh': zv, 'z20_thresh': 0, 'z_cross_thresh': 0,
                'rank_mode': 'roc_z60', 'filter_roc_improving': True, 'hold': 1, 'top_n': 1,
            }
            r = backtest(cfg)
            d_results.append({**r, 'roc': rv, 'z60': zv})
            print(f" | {r['ann']:>+9.1f}%/{r['n']:>4} ", end="")
        print()

    best_d = max(d_results, key=lambda x: x['ann'])
    print(f"\n  Best sweep: ROC>{best_d['roc']}% Z60>{best_d['z60']}")
    print(f"    Annual: {best_d['ann']:>+.1f}%, WR: {best_d['wr']:.1f}%, Trades: {best_d['n']}, MDD: {best_d['mdd']:.1f}%")

    # Walk-forward for top 5 sweep results
    d_sorted = sorted(d_results, key=lambda x: -x['ann'])[:5]
    print(f"\n  Top 5 D) configs — Walk-Forward:")
    header = f"  {'Config':>30} | {'Ann':>8} |"
    for yr in wf_years:
        header += f" {yr:>8} |"
    header += f" {'WF+':>4}"
    print(header)
    print("-" * 110)

    for dr in d_sorted:
        cfg = {
            'label': dr['label'],
            'roc_thresh': dr['roc'], 'z60_thresh': dr['z60'], 'z20_thresh': 0,
            'z_cross_thresh': 0,
            'rank_mode': 'roc_z60', 'filter_roc_improving': True, 'hold': 1, 'top_n': 1,
        }
        wf = walk_forward(cfg)
        vals = {yr: wf[yr]['ann'] if wf[yr] else 0 for yr in wf_years}
        avg = np.mean(list(vals.values()))
        pos = sum(1 for v in vals.values() if v > 0)
        row = f"  ROC>{dr['roc']}% Z60>{dr['z60']:>20} | {dr['ann']:>+7.1f}% |"
        for yr in wf_years:
            row += f" {vals[yr]:>+7.1f}% |"
        row += f" {pos}/6"
        print(row)

    # ================================================================
    # SECTION 3: H) FINE-TUNE ROC THRESHOLD
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  SECTION 3: H) FINE-TUNE ROC THRESHOLD (Z60>1.0, ROC*Z rank, ROC improving)")
    print(f"{'=' * 150}")
    print(f"  {'ROC%':>8} | {'Ann':>10} | {'WR':>6} | {'N':>5} | {'AvgPnL':>8} | {'MDD':>8}")
    print("-" * 60)

    h_results = []
    for cfg in h_configs:
        r = backtest(cfg)
        h_results.append({**r, 'roc': cfg['roc_thresh']})
        print(f"  {cfg['roc_thresh']:>7.1f}% | {r['ann']:>+9.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+7.3f}% | {r['mdd']:>+7.1f}%")

    best_h = max(h_results, key=lambda x: x['ann'])
    print(f"\n  Best ROC threshold: {best_h['roc']}%  => {best_h['ann']:>+.1f}%")

    # Walk-forward for top 3
    h_sorted = sorted(h_results, key=lambda x: -x['ann'])[:3]
    print(f"\n  Top 3 ROC thresholds — Walk-Forward:")
    header = f"  {'ROC%':>8} | {'Ann':>8} |"
    for yr in wf_years:
        header += f" {yr:>8} |"
    header += f" {'WF+':>4}"
    print(header)
    print("-" * 100)
    for hr in h_sorted:
        cfg = {
            'label': hr['label'],
            'roc_thresh': hr['roc'], 'z60_thresh': 1.0, 'z20_thresh': 0,
            'z_cross_thresh': 0,
            'rank_mode': 'roc_z60', 'filter_roc_improving': True, 'hold': 1, 'top_n': 1,
        }
        wf = walk_forward(cfg)
        vals = {yr: wf[yr]['ann'] if wf[yr] else 0 for yr in wf_years}
        avg = np.mean(list(vals.values()))
        pos = sum(1 for v in vals.values() if v > 0)
        row = f"  {hr['roc']:>7.1f}% | {hr['ann']:>+7.1f}% |"
        for yr in wf_years:
            row += f" {vals[yr]:>+7.1f}% |"
        row += f" {pos}/6"
        print(row)

    # ================================================================
    # SECTION 4: I) FINE-TUNE Z60 THRESHOLD
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  SECTION 4: I) FINE-TUNE Z60 THRESHOLD (ROC>1%, ROC*Z rank, ROC improving)")
    print(f"{'=' * 150}")
    print(f"  {'Z60':>8} | {'Ann':>10} | {'WR':>6} | {'N':>5} | {'AvgPnL':>8} | {'MDD':>8}")
    print("-" * 60)

    i_results = []
    for cfg in i_configs:
        r = backtest(cfg)
        i_results.append({**r, 'z60': cfg['z60_thresh']})
        print(f"  {cfg['z60_thresh']:>7.2f} | {r['ann']:>+9.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+7.3f}% | {r['mdd']:>+7.1f}%")

    best_i = max(i_results, key=lambda x: x['ann'])
    print(f"\n  Best Z60 threshold: {best_i['z60']}  => {best_i['ann']:>+.1f}%")

    # Walk-forward for top 3
    i_sorted = sorted(i_results, key=lambda x: -x['ann'])[:3]
    print(f"\n  Top 3 Z60 thresholds — Walk-Forward:")
    header = f"  {'Z60':>8} | {'Ann':>8} |"
    for yr in wf_years:
        header += f" {yr:>8} |"
    header += f" {'WF+':>4}"
    print(header)
    print("-" * 100)
    for ir in i_sorted:
        cfg = {
            'label': ir['label'],
            'roc_thresh': 1.0, 'z60_thresh': ir['z60'], 'z20_thresh': 0,
            'z_cross_thresh': 0,
            'rank_mode': 'roc_z60', 'filter_roc_improving': True, 'hold': 1, 'top_n': 1,
        }
        wf = walk_forward(cfg)
        vals = {yr: wf[yr]['ann'] if wf[yr] else 0 for yr in wf_years}
        avg = np.mean(list(vals.values()))
        pos = sum(1 for v in vals.values() if v > 0)
        row = f"  {ir['z60']:>7.2f} | {ir['ann']:>+7.1f}% |"
        for yr in wf_years:
            row += f" {vals[yr]:>+7.1f}% |"
        row += f" {pos}/6"
        print(row)

    # ================================================================
    # SECTION 5: J) MDD MANAGEMENT
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  SECTION 5: J) MDD MANAGEMENT — Equity-based position sizing")
    print(f"{'=' * 150}")

    # Find the overall champion so far
    all_main = core_results + d_results + h_results + i_results
    champion_result = max(all_main, key=lambda x: x['ann'])
    # Find the matching config
    champion_cfg = None
    for cfg in configs + d_configs + h_configs + i_configs:
        if cfg['label'] == champion_result.get('label', ''):
            champion_cfg = cfg
            break
    # If we can't find by label, build from best sweep params
    if champion_cfg is None:
        # Use best from D sweep
        champion_cfg = {
            'roc_thresh': best_d['roc'], 'z60_thresh': best_d['z60'],
            'z20_thresh': 0, 'z_cross_thresh': 0,
            'rank_mode': 'roc_z60', 'filter_roc_improving': True,
            'hold': 1, 'top_n': 1,
            'label': f"Champion ROC>{best_d['roc']}% Z60>{best_d['z60']}",
        }
        print(f"  Using sweep champion: ROC>{best_d['roc']}% Z60>{best_d['z60']}")

    print(f"  Champion config: {champion_cfg['label']}")
    print(f"  Champion annual: {champion_result['ann']:>+.1f}%, MDD: {champion_result['mdd']:>+.1f}%")

    equity_mgmt_configs = [
        None,
        {'levels': [(1.5, 0.5)]},            # >1.5x: risk 50%
        {'levels': [(2.0, 0.33)]},           # >2x: risk 33%
        {'levels': [(1.5, 0.5), (2.0, 0.33)]},  # Both
        {'levels': [(1.5, 0.5), (2.0, 0.33), (3.0, 0.25)]},  # Progressive
        {'levels': [(2.0, 0.5)]},            # Only >2x: risk 50%
        {'levels': [(3.0, 0.5)]},            # Only >3x: risk 50%
    ]
    mgmt_labels = [
        'No management (baseline)',
        '>1.5x: risk 50%',
        '>2x: risk 33%',
        '>1.5x: 50%, >2x: 33%',
        '>1.5x: 50%, >2x: 33%, >3x: 25%',
        '>2x: risk 50%',
        '>3x: risk 50%',
    ]

    print(f"\n  {'Strategy':<45} | {'Ann':>10} | {'WR':>6} | {'N':>5} | {'MDD':>8} | {'Final':>12}")
    print("-" * 110)

    mgmt_results = []
    for mgmt, mgmt_label in zip(equity_mgmt_configs, mgmt_labels):
        r = backtest(champion_cfg, equity_mgmt=mgmt)
        mgmt_results.append(r)
        print(f"  {mgmt_label:<45} | {r['ann']:>+9.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['mdd']:>+7.1f}% | {r['final_cash']:>11,.0f}")

    # Walk-forward for best MDD management
    # Find the one that best balances return and MDD
    best_mgmt = max(mgmt_results, key=lambda x: x['ann'] if x['mdd'] > -95 else -999)
    best_mgmt_idx = mgmt_results.index(best_mgmt)
    print(f"\n  Best MDD management: {mgmt_labels[best_mgmt_idx]}")
    print(f"    Annual: {best_mgmt['ann']:>+.1f}%, MDD: {best_mgmt['mdd']:>+.1f}%")

    print(f"\n  Walk-Forward for MDD management configs:")
    header = f"  {'Strategy':<45} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>8} |"
    header += f" {'Pos':>4} | {'AvgMDD':>7}"
    print(header)
    print("-" * 150)

    for mgmt, mgmt_label in zip(equity_mgmt_configs, mgmt_labels):
        wf = walk_forward(champion_cfg, equity_mgmt=mgmt)
        print_wf(mgmt_label, wf)

    # ================================================================
    # OVERALL CHAMPION DETERMINATION
    # ================================================================
    print(f"\n{'=' * 150}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 150}")

    # Gather all unique results (excluding MDD management variants for simplicity)
    all_results = []
    for r in core_results:
        all_results.append(r)
    for r in d_results:
        all_results.append(r)
    for r in h_results:
        all_results.append(r)
    for r in i_results:
        all_results.append(r)

    all_sorted = sorted(all_results, key=lambda x: -x['ann'])

    print(f"\n  TOP 10 CONFIGURATIONS (by annual return):")
    print(f"  {'#':>3} | {'Config':<55} | {'Ann':>10} | {'WR':>6} | {'N':>5} | {'MDD':>8}")
    print("-" * 110)
    for i, r in enumerate(all_sorted[:10]):
        print(f"  {i+1:>3} | {r.get('label',''):<55} | {r['ann']:>+9.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['mdd']:>+7.1f}%")

    # Question 1: Does 60d Z + V121 enhancements beat +333.5%?
    print(f"\n  Q1: Does 60d Z + V121 enhancements beat +333.5%?")
    beat_333 = [r for r in all_results if r['ann'] > 333.5]
    if beat_333:
        best_beat = max(beat_333, key=lambda x: x['ann'])
        print(f"      YES! Best = {best_beat['ann']:>+.1f}% ({best_beat.get('label','')})")
    else:
        best_all = all_sorted[0]
        print(f"      NO. Best = {best_all['ann']:>+.1f}% (gap: {333.5 - best_all['ann']:.1f}pp)")

    # Question 2: Optimal thresholds
    print(f"\n  Q2: Optimal thresholds:")
    print(f"      Best ROC threshold: {best_h['roc']}% (ann={best_h['ann']:>+.1f}%)")
    print(f"      Best Z60 threshold: {best_i['z60']} (ann={best_i['ann']:>+.1f}%)")
    print(f"      Best combined: ROC>{best_d['roc']}% Z60>{best_d['z60']} (ann={best_d['ann']:>+.1f}%)")

    # Question 3: MDD management
    print(f"\n  Q3: MDD management results:")
    baseline_mgmt = mgmt_results[0]
    for mgmt_label, mgmt_r in zip(mgmt_labels[1:], mgmt_results[1:]):
        ret_diff = mgmt_r['ann'] - baseline_mgmt['ann']
        mdd_diff = mgmt_r['mdd'] - baseline_mgmt['mdd']
        print(f"      {mgmt_label:<40}: ann={mgmt_r['ann']:>+.1f}% ({ret_diff:>+.1f}pp), MDD={mgmt_r['mdd']:>+.1f}% ({mdd_diff:>+.1f}pp)")

    # Question 4: Best overall
    print(f"\n  Q4: Best overall config:")
    champion = all_sorted[0]
    print(f"      {champion.get('label', '')}")
    print(f"      Annual: {champion['ann']:>+.1f}%")
    print(f"      WR: {champion['wr']:.1f}%")
    print(f"      Trades: {champion['n']}")
    print(f"      MDD: {champion['mdd']:>+.1f}%")

    # Walk-forward for the absolute champion
    champ_cfg = None
    for cfg_list in [configs, d_configs, h_configs, i_configs]:
        for c in cfg_list:
            if c['label'] == champion.get('label', ''):
                champ_cfg = c
                break
        if champ_cfg:
            break

    if champ_cfg:
        print(f"\n  Champion Walk-Forward:")
        wf = walk_forward(champ_cfg)
        vals = {yr: wf[yr]['ann'] if wf[yr] else 0 for yr in wf_years}
        for yr in wf_years:
            v = vals[yr]
            wr_yr = wf[yr]['wr'] if wf[yr] else 0
            mdd_yr = wf[yr]['mdd'] if wf[yr] else 0
            n_yr = wf[yr]['n'] if wf[yr] else 0
            print(f"    {yr}: {v:>+8.1f}%  WR={wr_yr:.1f}%  N={n_yr}  MDD={mdd_yr:>+6.1f}%")
        wf_avg = np.mean(list(vals.values()))
        wf_pos = sum(1 for v in vals.values() if v > 0)
        print(f"    WF Average: {wf_avg:>+.1f}%, Positive: {wf_pos}/6")

    # Question 5: Any config approaching 400%+?
    print(f"\n  Q5: Any config approaching 400%+?")
    over_400 = [r for r in all_results if r['ann'] >= 400]
    over_350 = [r for r in all_results if r['ann'] >= 350]
    if over_400:
        print(f"      YES! {len(over_400)} configs above 400%:")
        for r in sorted(over_400, key=lambda x: -x['ann'])[:5]:
            print(f"        {r.get('label','')}: {r['ann']:>+.1f}% (WR={r['wr']:.1f}%, MDD={r['mdd']:>+.1f}%)")
    elif over_350:
        print(f"      No config above 400%. {len(over_350)} configs above 350%:")
        for r in sorted(over_350, key=lambda x: -x['ann'])[:5]:
            print(f"        {r.get('label','')}: {r['ann']:>+.1f}% (WR={r['wr']:.1f}%, MDD={r['mdd']:>+.1f}%)")
    else:
        best = all_sorted[0]
        print(f"      No. Best = {best['ann']:>+.1f}% ({best.get('label','')})")

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed: {elapsed:.1f}s")
    print("=" * 150)


if __name__ == '__main__':
    main()
