"""
Alpha Futures V91 -- Volume & Volatility Confirmation of Z-Score Signal
========================================================================
V82 champion: cross-group z-score gives +3305% annual.
Signal: z = (own_return - all_groups_avg) / all_groups_std; z < -0.5 -> buy.

V91 tests whether volume and volatility CONFIRM the z-score signal.
When a commodity is weak vs cross-group AND has abnormal volume or volatility,
the mean-reversion signal may be stronger.

Signals:
  A) z_baseline:           V82 exact copy (z < threshold -> buy, long-only, 1-day hold)
  B) z_and_vol_surge:      z < threshold AND volume > 1.5x 20-day average
  C) z_and_vol_rank:       z < threshold, rank by vol_surge * (-z)
  D) z_and_range_expand:   z < threshold AND (H-L)/C > 1.5x 20-day avg range
  E) z_and_body_ratio:     z < threshold AND body_ratio < 0.3 (doji at bottom)
  F) z_and_lower_shadow:   z < threshold AND lower_shadow > 0.4 (rejection of lows)
  G) z_vol_combined_score: Score = -z * (1 + vol_surge_ratio)
  H) z_and_atr_pct_low:    z < threshold AND ATR percentile < 30 (low vol context)
  I) z_and_atr_pct_high:   z < threshold AND ATR percentile > 70 (high vol context)

Walk-forward: top 15 configs across 2020-2025.
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

# ── Group map (same as V82 champion) ────────────────────────────────
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
for _s in ['ppfi', 'vfi', 'egfi', 'srfi', 'tafi', 'fgfi', 'lfi']:
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


def rolling_mean(arr_1d, window):
    """Fast rolling mean over a 1D array, ignoring NaN."""
    n = len(arr_1d)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = arr_1d[i - window + 1: i + 1]
        valid = w[~np.isnan(w)]
        if len(valid) >= window // 2:
            out[i] = np.mean(valid)
    return out


def rolling_rank_percentile(arr_1d, lookback):
    """Percentile rank of arr_1d[i] vs past `lookback` values."""
    n = len(arr_1d)
    out = np.full(n, np.nan)
    for i in range(lookback, n):
        w = arr_1d[i - lookback: i]
        valid = w[~np.isnan(w)]
        if len(valid) < lookback // 2:
            continue
        val = arr_1d[i]
        if np.isnan(val):
            continue
        out[i] = np.mean(valid < val) * 100  # 0-100 percentile
    return out


def main():
    print("=" * 130)
    print("Alpha Futures V91 -- Volume & Volatility Confirmation of Z-Score Signal")
    print("=" * 130)

    # ── Load data ────────────────────────────────────────────────────
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

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
    for gn in group_names:
        print(f"    {gn}: {len(gm_map[gn])} commodities")

    # ── Precompute 1-day returns ─────────────────────────────────────
    print("\n[Precompute] 1-day returns...", flush=True)
    t0 = time.time()
    ret1 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            cn = C[si, di]
            cp = C[si, di - 1]
            if not np.isnan(cn) and not np.isnan(cp) and cp > 0:
                ret1[si, di] = (cn - cp) / cp

    # ── Precompute group-level signals ───────────────────────────────
    print("[Precompute] Group-level aggregates...", flush=True)

    # group_total_avg[group_name] -> array[ND]
    grp_total = {}
    for grp in group_names:
        arr = np.full(ND, np.nan)
        members = gm_map[grp]
        for di in range(1, ND):
            vals = [ret1[sk, di] for sk in members if not np.isnan(ret1[sk, di])]
            if vals:
                arr[di] = np.mean(vals)
        grp_total[grp] = arr

    # all_groups_avg[di], all_groups_std[di]
    all_groups_avg = np.full(ND, np.nan)
    all_groups_std = np.full(ND, np.nan)
    for di in range(1, ND):
        vals = [grp_total[g][di] for g in group_names if not np.isnan(grp_total[g][di])]
        if len(vals) >= 2:
            all_groups_avg[di] = np.mean(vals)
            all_groups_std[di] = np.std(vals)

    # z_score[si, di] = (own_return - all_groups_avg) / all_groups_std
    z_score = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        aga = all_groups_avg[di]
        ags = all_groups_std[di]
        if np.isnan(aga) or np.isnan(ags) or ags < 1e-8:
            continue
        for si in trade_sis:
            own = ret1[si, di]
            if not np.isnan(own):
                z_score[si, di] = (own - aga) / ags

    print(f"  Group signals done ({time.time()-t0:.1f}s)")

    # ── Precompute volume & volatility features ──────────────────────
    print("[Precompute] Volume & volatility features...", flush=True)
    t1 = time.time()

    # Volume 20-day rolling average
    vol_avg_20 = np.full((NS, ND), np.nan)
    # Range ratio = (H-L)/C, 20-day rolling average
    range_avg_20 = np.full((NS, ND), np.nan)
    # ATR = (H-L), 20-day rolling average
    atr_20 = np.full((NS, ND), np.nan)
    # ATR percentile (252-day lookback)
    atr_pct = np.full((NS, ND), np.nan)
    # Volume surge ratio = V[di] / vol_avg_20[di]
    vol_surge = np.full((NS, ND), np.nan)
    # Range expand ratio = ((H-L)/C)[di] / range_avg_20[di]
    range_expand = np.full((NS, ND), np.nan)
    # Body ratio = |C-O| / (H-L)
    body_ratio = np.full((NS, ND), np.nan)
    # Lower shadow = (min(C,O) - L) / (H - L)
    lower_shadow = np.full((NS, ND), np.nan)

    for si in trade_sis:
        # Volume rolling mean
        vol_avg_20[si] = rolling_mean(V[si], 20)
        # Daily range ratio
        daily_range = np.full(ND, np.nan)
        for di in range(ND):
            c = C[si, di]
            h = H[si, di]
            l = L[si, di]
            if not np.isnan(c) and c > 0 and not np.isnan(h) and not np.isnan(l):
                daily_range[di] = (h - l) / c
        range_avg_20[si] = rolling_mean(daily_range, 20)

        # ATR rolling mean (H-L absolute)
        daily_hl = np.full(ND, np.nan)
        for di in range(ND):
            h = H[si, di]
            l = L[si, di]
            if not np.isnan(h) and not np.isnan(l):
                daily_hl[di] = h - l
        atr_20[si] = rolling_mean(daily_hl, 20)

        # ATR percentile (rank vs past 252 days)
        atr_pct[si] = rolling_rank_percentile(atr_20[si], 252)

        # Volume surge ratio
        for di in range(ND):
            va = vol_avg_20[si, di]
            v = V[si, di]
            if not np.isnan(va) and va > 0 and not np.isnan(v):
                vol_surge[si, di] = v / va

        # Range expand ratio
        for di in range(ND):
            ra = range_avg_20[si, di]
            dr = daily_range[di]
            if not np.isnan(ra) and ra > 0 and not np.isnan(dr):
                range_expand[si, di] = dr / ra

        # Body ratio & lower shadow
        for di in range(ND):
            c = C[si, di]
            o = O[si, di]
            h = H[si, di]
            l = L[si, di]
            if np.isnan(c) or np.isnan(o) or np.isnan(h) or np.isnan(l):
                continue
            rng = h - l
            if rng > 0:
                body_ratio[si, di] = abs(c - o) / rng
                lower_shadow[si, di] = (min(c, o) - l) / rng

    print(f"  Volume & volatility features done ({time.time()-t1:.1f}s)")

    # ── Signal summary ───────────────────────────────────────────────
    # Count how often each filter triggers
    print("\n[Feature Stats] Filter frequency (across all tradeable si, days >= MIN_TRAIN):")
    for label, arr, cond_desc in [
        ("vol_surge > 1.5x", vol_surge, "v > 1.5"),
        ("range_expand > 1.5x", range_expand, "r > 1.5"),
        ("body_ratio < 0.3", body_ratio, "br < 0.3"),
        ("lower_shadow > 0.4", lower_shadow, "ls > 0.4"),
        ("atr_pct < 30", atr_pct, "pct < 30"),
        ("atr_pct > 70", atr_pct, "pct > 70"),
    ]:
        cnt = 0
        total = 0
        for si in trade_sis:
            for di in range(MIN_TRAIN, ND):
                v = arr[si, di]
                if np.isnan(v):
                    continue
                total += 1
                if "body_ratio" in label and v < 0.3:
                    cnt += 1
                elif "lower_shadow" in label and v > 0.4:
                    cnt += 1
                elif "atr_pct <" in label and v < 30:
                    cnt += 1
                elif "atr_pct >" in label and v > 70:
                    cnt += 1
                elif "vol_surge" in label and v > 1.5:
                    cnt += 1
                elif "range_expand" in label and v > 1.5:
                    cnt += 1
        pct = cnt / max(total, 1) * 100
        print(f"    {label:<25s} triggers {cnt:>8d} / {total:>8d}  ({pct:.1f}%)")

    # ══════════════════════════════════════════════════════════════════
    # BACKTEST ENGINE
    # ══════════════════════════════════════════════════════════════════
    def run_backtest(config, wf_test_year=None):
        """
        Config:
            signal: 'A'..'I'
            threshold: float (z-score cutoff)
            top_n: 1 | 3
            comm: float
        """
        sig_type = config['signal']
        threshold = config['threshold']
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
            candidates = []  # (si, score, direction, sym)

            aga = all_groups_avg[di]
            ags = all_groups_std[di]
            if np.isnan(aga) or np.isnan(ags) or ags < 1e-8:
                continue

            for si in trade_sis:
                z = z_score[si, di]
                if np.isnan(z):
                    continue
                cc = C[si, di]
                if np.isnan(cc) or cc <= 0:
                    continue
                if any(p['si'] == si for p in positions):
                    continue

                # ── Signal A: z_baseline (V82 exact) ────────────────
                if sig_type == 'A':
                    if z < -threshold:
                        candidates.append((si, -z, 1, syms[si]))

                # ── Signal B: z AND vol_surge ────────────────────────
                elif sig_type == 'B':
                    vs = vol_surge[si, di]
                    if np.isnan(vs):
                        continue
                    if z < -threshold and vs > 1.5:
                        candidates.append((si, -z, 1, syms[si]))

                # ── Signal C: z filter, rank by vol_surge * (-z) ────
                elif sig_type == 'C':
                    vs = vol_surge[si, di]
                    if np.isnan(vs):
                        continue
                    if z < -threshold:
                        score = vs * (-z)
                        candidates.append((si, score, 1, syms[si]))

                # ── Signal D: z AND range_expand ─────────────────────
                elif sig_type == 'D':
                    re = range_expand[si, di]
                    if np.isnan(re):
                        continue
                    if z < -threshold and re > 1.5:
                        candidates.append((si, -z, 1, syms[si]))

                # ── Signal E: z AND body_ratio < 0.3 ────────────────
                elif sig_type == 'E':
                    br = body_ratio[si, di]
                    if np.isnan(br):
                        continue
                    if z < -threshold and br < 0.3:
                        candidates.append((si, -z, 1, syms[si]))

                # ── Signal F: z AND lower_shadow > 0.4 ──────────────
                elif sig_type == 'F':
                    ls = lower_shadow[si, di]
                    if np.isnan(ls):
                        continue
                    if z < -threshold and ls > 0.4:
                        candidates.append((si, -z, 1, syms[si]))

                # ── Signal G: combined score = -z * (1 + vol_surge) ─
                elif sig_type == 'G':
                    vs = vol_surge[si, di]
                    if np.isnan(vs):
                        continue
                    if z < -threshold:
                        score = (-z) * (1 + vs)
                        candidates.append((si, score, 1, syms[si]))

                # ── Signal H: z AND ATR percentile < 30 ─────────────
                elif sig_type == 'H':
                    ap = atr_pct[si, di]
                    if np.isnan(ap):
                        continue
                    if z < -threshold and ap < 30:
                        candidates.append((si, -z, 1, syms[si]))

                # ── Signal I: z AND ATR percentile > 70 ─────────────
                elif sig_type == 'I':
                    ap = atr_pct[si, di]
                    if np.isnan(ap):
                        continue
                    if z < -threshold and ap > 70:
                        candidates.append((si, -z, 1, syms[si]))

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
        n_days_test = (test_end_di - test_start_di) if wf_mode else (end_di - start_di)
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0

        # Max drawdown
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

    signal_names = {
        'A': 'z_baseline',
        'B': 'z_and_vol_surge',
        'C': 'z_and_vol_rank',
        'D': 'z_and_range_expand',
        'E': 'z_and_body_ratio',
        'F': 'z_and_lower_shadow',
        'G': 'z_vol_combined_score',
        'H': 'z_and_atr_pct_low',
        'I': 'z_and_atr_pct_high',
    }

    for sig_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
        for thresh in [0.3, 0.5, 0.7]:
            for tn in [1, 3]:
                cid += 1
                sig_name = signal_names[sig_key]
                label = f"{sig_key}_{sig_name}_T{thresh}_TN{tn}"
                configs.append({
                    'id': cid, 'signal': sig_key,
                    'threshold': thresh, 'top_n': tn, 'comm': COMM,
                    'label': label, 'sig_name': sig_name,
                })

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

    # Print top 25
    print(f"\n{'=' * 140}")
    print("  FULL-PERIOD RESULTS (Top 25)")
    print(f"{'=' * 140}")
    print(f"  {'#':>3} | {'Label':<50} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7}")
    print("-" * 130)
    for i, r in enumerate(results[:25]):
        print(f"  {i+1:>3} | {r['label']:<50} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}%")

    # ══════════════════════════════════════════════════════════════════
    # SIGNAL COMPARISON (full period)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 140}")
    print("  SIGNAL COMPARISON (Best per signal type, full period)")
    print(f"{'=' * 140}")
    print(f"  {'Signal':<15} | {'Description':<25} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | Best Config")
    print("-" * 140)

    best_per_sig = {}
    for r in results:
        s = r['config']['signal']
        if s not in best_per_sig:
            best_per_sig[s] = r

    sig_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    for sig in sig_order:
        if sig in best_per_sig:
            b = best_per_sig[sig]
            desc = signal_names[sig]
            print(f"  {sig:<15} | {desc:<25} | {b['ann']:>+8.1f}% | {b['wr']:>5.1f}% | {b['n']:>5} | {b['avg_pnl']:>+6.3f}% | {b['mdd']:>6.1f}% | {b['label']}")

    # Alpha vs baseline (A)
    a_base = best_per_sig.get('A')
    if a_base:
        print(f"\n  --- Alpha vs Baseline (Signal A) ---")
        print(f"  Baseline A: {a_base['ann']:>+8.1f}%  (N={a_base['n']}, WR={a_base['wr']:.1f}%)")
        for sig in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
            if sig in best_per_sig:
                b = best_per_sig[sig]
                diff = b['ann'] - a_base['ann']
                tag = "BEATS" if diff > 0 else "LOSES"
                print(f"  Signal {sig} ({signal_names[sig]:<25}): {b['ann']:>+8.1f}%  ({tag} by {diff:>+.1f}%, N={b['n']}, WR={b['wr']:.1f}%)")

    # ══════════════════════════════════════════════════════════════════
    # WALK-FORWARD (Top 15 configs)
    # ══════════════════════════════════════════════════════════════════
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Take top 15 unique configs + best per signal type
    wf_configs = list(results[:15])
    seen_cfgs = set(id(w['config']) for w in wf_configs)
    for sig in sig_order:
        if sig in best_per_sig:
            r = best_per_sig[sig]
            if id(r['config']) not in seen_cfgs:
                wf_configs.append(r)
                seen_cfgs.add(id(r['config']))

    print(f"\n{'=' * 160}")
    print(f"  WALK-FORWARD ({len(wf_configs)} configs)")
    print(f"{'=' * 160}")

    header = f"  {'#':>3} | {'Config':<50} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>7} |"
    header += f" {'Pos':>4} | {'MDD':>7}"
    print(header)
    print("-" * 160)

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
    # WF COMPARISON PER SIGNAL
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 140}")
    print("  WALK-FORWARD COMPARISON (Best per signal type)")
    print(f"{'=' * 140}")
    header2 = f"  {'Signal':<15} | {'Desc':<25} | {'WF Avg':>8} |"
    for yr in wf_years:
        header2 += f" {yr:>7} |"
    header2 += f" {'Pos':>4}"
    print(header2)
    print("-" * 140)

    wf_best = {}
    for w in wf_rows:
        s = w['signal']
        vals = [w['windows'].get(yr, 0) for yr in wf_years]
        avg = np.mean(vals)
        if s not in wf_best or avg > wf_best[s]['avg']:
            wf_best[s] = {'row': w, 'avg': avg}

    for sig in sig_order:
        if sig in wf_best:
            wf = wf_best[sig]['row']
            vals = [wf['windows'].get(yr, 0) for yr in wf_years]
            avg = np.mean(vals) if vals else 0
            pos = sum(1 for v in vals if v > 0)
            row_str = f"  {sig:<15} | {signal_names[sig]:<25} | {avg:>+7.1f}% |"
            for v in vals:
                row_str += f" {v:>+7.1f}% |"
            row_str += f" {pos}/6"
            print(row_str)

    # ══════════════════════════════════════════════════════════════════
    # DETAILED BREAKDOWN: Volume confirmation signals (B, C, G)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 140}")
    print("  VOLUME CONFIRMATION BREAKDOWN")
    print(f"{'=' * 140}")
    for sig_key in ['B', 'C', 'G']:
        sig_results = [r for r in results if r['config']['signal'] == sig_key]
        if sig_results:
            best = sig_results[0]
            print(f"  {sig_key} ({signal_names[sig_key]:<25}): Best = {best['ann']:>+8.1f}%  N={best['n']}  WR={best['wr']:>5.1f}%  MDD={best['mdd']:>6.1f}%")
            # Show all thresholds
            for thresh in [0.3, 0.5, 0.7]:
                for tn in [1, 3]:
                    match = [r for r in sig_results
                             if r['config']['threshold'] == thresh and r['config']['top_n'] == tn]
                    if match:
                        m = match[0]
                        print(f"      T={thresh} TN={tn}: {m['ann']:>+8.1f}%  N={m['n']}  WR={m['wr']:>5.1f}%  MDD={m['mdd']:>6.1f}%")

    # ══════════════════════════════════════════════════════════════════
    # DETAILED BREAKDOWN: Volatility confirmation signals (D, E, F)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 140}")
    print("  VOLATILITY CONFIRMATION BREAKDOWN")
    print(f"{'=' * 140}")
    for sig_key in ['D', 'E', 'F']:
        sig_results = [r for r in results if r['config']['signal'] == sig_key]
        if sig_results:
            best = sig_results[0]
            print(f"  {sig_key} ({signal_names[sig_key]:<25}): Best = {best['ann']:>+8.1f}%  N={best['n']}  WR={best['wr']:>5.1f}%  MDD={best['mdd']:>6.1f}%")
            for thresh in [0.3, 0.5, 0.7]:
                for tn in [1, 3]:
                    match = [r for r in sig_results
                             if r['config']['threshold'] == thresh and r['config']['top_n'] == tn]
                    if match:
                        m = match[0]
                        print(f"      T={thresh} TN={tn}: {m['ann']:>+8.1f}%  N={m['n']}  WR={m['wr']:>5.1f}%  MDD={m['mdd']:>6.1f}%")

    # ══════════════════════════════════════════════════════════════════
    # DETAILED BREAKDOWN: ATR context signals (H, I)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 140}")
    print("  ATR CONTEXT BREAKDOWN")
    print(f"{'=' * 140}")
    for sig_key in ['H', 'I']:
        sig_results = [r for r in results if r['config']['signal'] == sig_key]
        if sig_results:
            best = sig_results[0]
            print(f"  {sig_key} ({signal_names[sig_key]:<25}): Best = {best['ann']:>+8.1f}%  N={best['n']}  WR={best['wr']:>5.1f}%  MDD={best['mdd']:>6.1f}%")
            for thresh in [0.3, 0.5, 0.7]:
                for tn in [1, 3]:
                    match = [r for r in sig_results
                             if r['config']['threshold'] == thresh and r['config']['top_n'] == tn]
                    if match:
                        m = match[0]
                        print(f"      T={thresh} TN={tn}: {m['ann']:>+8.1f}%  N={m['n']}  WR={m['wr']:>5.1f}%  MDD={m['mdd']:>6.1f}%")

    # ══════════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 140}")
    print("  FINAL VERDICT")
    print(f"{'=' * 140}")

    if a_base:
        base_ann = a_base['ann']
        print(f"  Baseline A (z_baseline, V82 copy): {base_ann:>+8.1f}%  (N={a_base['n']}, WR={a_base['wr']:.1f}%, MDD={a_base['mdd']:.1f}%)")

        # Find best overall
        best_overall = results[0]
        best_sig = best_overall['config']['signal']
        best_desc = signal_names[best_sig]
        print(f"  Best overall: {best_overall['ann']:>+8.1f}%  Signal {best_sig} ({best_desc}), N={best_overall['n']}, WR={best_overall['wr']:.1f}%, MDD={best_overall['mdd']:.1f}%")

        # Best volume confirmation
        vol_sigs = ['B', 'C', 'G']
        vol_best = None
        for sig in vol_sigs:
            if sig in best_per_sig:
                if vol_best is None or best_per_sig[sig]['ann'] > vol_best['ann']:
                    vol_best = best_per_sig[sig]
        if vol_best:
            diff = vol_best['ann'] - base_ann
            tag = "YES" if diff > 0 else "NO"
            print(f"\n  Volume confirmation: {tag}")
            print(f"    Best vol signal: {vol_best['ann']:>+8.1f}% ({vol_best['config']['signal']} - {vol_best['config']['sig_name']})")
            print(f"    Alpha vs baseline: {diff:>+.1f}%")
            if diff > 0:
                print(f"    >>> VOLUME CONFIRMATION ADDS ALPHA <<<")
            else:
                print(f"    >>> VOLUME CONFIRMATION DOES NOT ADD ALPHA <<<")

        # Best volatility confirmation
        volat_sigs = ['D', 'E', 'F']
        volat_best = None
        for sig in volat_sigs:
            if sig in best_per_sig:
                if volat_best is None or best_per_sig[sig]['ann'] > volat_best['ann']:
                    volat_best = best_per_sig[sig]
        if volat_best:
            diff = volat_best['ann'] - base_ann
            tag = "YES" if diff > 0 else "NO"
            print(f"\n  Volatility/candlestick confirmation: {tag}")
            print(f"    Best volat signal: {volat_best['ann']:>+8.1f}% ({volat_best['config']['signal']} - {volat_best['config']['sig_name']})")
            print(f"    Alpha vs baseline: {diff:>+.1f}%")
            if diff > 0:
                print(f"    >>> VOLATILITY CONFIRMATION ADDS ALPHA <<<")
            else:
                print(f"    >>> VOLATILITY CONFIRMATION DOES NOT ADD ALPHA <<<")

        # ATR context comparison
        h_best = best_per_sig.get('H')
        i_best = best_per_sig.get('I')
        if h_best and i_best:
            print(f"\n  ATR Context comparison:")
            print(f"    Low vol context  (H): {h_best['ann']:>+8.1f}%  N={h_best['n']}")
            print(f"    High vol context (I): {i_best['ann']:>+8.1f}%  N={i_best['n']}")
            if i_best['ann'] > h_best['ann']:
                print(f"    >>> Z-Score works better in HIGH VOLATILITY (capitulation) <<<")
            else:
                print(f"    >>> Z-Score works better in LOW VOLATILITY (quiet drift) <<<")

        # WF verdict
        print(f"\n  Walk-Forward verdict:")
        wf_a = wf_best.get('A')
        if wf_a:
            a_vals = [wf_a['row']['windows'].get(yr, 0) for yr in wf_years]
            a_avg = np.mean(a_vals)
            a_pos = sum(1 for v in a_vals if v > 0)
            print(f"    Baseline A WF avg: {a_avg:>+7.1f}% ({a_pos}/6 positive)")

            for sig in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
                if sig in wf_best:
                    wf_s = wf_best[sig]['row']
                    s_vals = [wf_s['windows'].get(yr, 0) for yr in wf_years]
                    s_avg = np.mean(s_vals)
                    s_pos = sum(1 for v in s_vals if v > 0)
                    diff = s_avg - a_avg
                    tag = "+" if diff > 0 else ""
                    print(f"    Signal {sig} WF avg: {s_avg:>+7.1f}% ({s_pos}/6 positive)  [{tag}{diff:.1f}% vs A]")

    print(f"\n  Total time: {time.time()-t_start:.1f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
