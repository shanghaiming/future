"""
Alpha Futures V20 — Multi-Strategy Ensemble (Confluence Scoring)
=================================================================
CORE IDEA: Run multiple independent signal strategies simultaneously.
When multiple signals agree on the same commodity -> concentrated bet.

5 Independent Signals:
  1. mom5       = 5-day price momentum (trend following)
  2. oi_mom5    = 5-day OI change (money flow)
  3. vdp_ema    = VDP EMA direction (buying pressure)
  4. donchian   = (price - 20d low) / (20d high - 20d low) (breakout level)
  5. body_ratio = (C-O)/(H-L) (candle conviction)

CONFLUENCE SCORE = sum of votes (+1 bullish, -1 bearish, 0 neutral)
Only trade when |confluence| >= min_confluence (default 3)

EXIT RULES:
  - Trail stop at trail_atr * ATR from entry
  - Time stop at hold_max days
  - Fixed stop at stop_loss %
  - Signal reversal: exit if confluence drops below 1

CONCENTRATED variant:
  - confluence >= 4: use 100% capital
  - confluence == 3: use 50% capital
  - confluence < 3: no trade

Constraint: no gap, no intraday, no leverage, 2-7 day hold
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

MULT = {
    'agfi': 15, 'alfi': 5, 'aufi': 1000, 'bufi': 10, 'cufi': 5,
    'fufi': 10, 'rbfi': 10, 'znfi': 5, 'nifi': 1, 'hcfi': 10,
    'spfi': 10, 'ssfi': 5, 'sffi': 5, 'smfi': 5, 'pbfi': 5,
    'snfi': 1, 'rufi': 10, 'wrffi': 10,
    'afi': 10, 'bfi': 10, 'bbfi': 500, 'cffi': 5, 'cfi': 10,
    'csfi': 10, 'ebfi': 5, 'egfi': 10, 'fbfi': 500,
    'ifi': 100, 'jfi': 100, 'jmfi': 60, 'lfi': 5, 'mfi': 10,
    'pgfi': 20, 'ppfi': 5, 'vfi': 5, 'yfi': 10, 'pfi': 10,
    'jdfi': 5, 'lhfi': 16, 'pkfi': 5, 'rrfi': 20, 'lrfi': 20,
    'jrfi': 20, 'pmfi': 20, 'whfi': 20, 'rsfi': 20, 'cjfi': 10,
    'mafi': 10, 'apfi': 10, 'cyfi': 5, 'fgfi': 20, 'oifi': 10,
    'pfifi': 5, 'rmfi': 10, 'srfi': 10, 'tafi': 5, 'safi': 20,
    'urfi': 20, 'scfi': 1000, 'lufi': 10, 'bcfi': 5, 'nrfi': 1,
    'lgfi': 20, 'brfi': 5, 'lcfi': 1, 'sifi': 5,
    'ni': 1, 'tai': 5,
}
DEF_MULT = 10
COMM = 0.0003


# =====================================================================
# PRE-COMPUTE ALL FACTORS
# =====================================================================
def precompute_factors(NS, ND, C, O, H, L, V, OI):
    """Pre-compute the 5 independent signals + ATR for all commodities."""
    t0 = time.time()
    print("  Pre-computing factors...", flush=True)

    mom5 = np.full((NS, ND), np.nan)
    oi_mom5 = np.full((NS, ND), np.nan)
    vdp_ema = np.full((NS, ND), np.nan)
    donchian = np.full((NS, ND), np.nan)
    body_ratio = np.full((NS, ND), np.nan)
    atr10 = np.full((NS, ND), np.nan)

    # Per-signal vote arrays (+1, -1, 0)
    vote_mom = np.zeros((NS, ND), dtype=np.int8)
    vote_oi = np.zeros((NS, ND), dtype=np.int8)
    vote_vdp = np.zeros((NS, ND), dtype=np.int8)
    vote_donch = np.zeros((NS, ND), dtype=np.int8)
    vote_body = np.zeros((NS, ND), dtype=np.int8)
    confluence = np.zeros((NS, ND), dtype=np.int8)

    # VDP EMA smoothing state per symbol
    vdp_e_state = np.zeros(NS)
    vdp_e_inited = np.zeros(NS, dtype=bool)

    for di in range(20, ND):
        d = di - 1  # Use data up to yesterday

        for si in range(NS):
            c_now = C[si, d]
            if np.isnan(c_now) or c_now <= 0:
                continue

            # ----------------------------------------------------------
            # Signal 1: 5-day price momentum
            # ----------------------------------------------------------
            c_prev5 = C[si, max(0, d - 5)]
            if not np.isnan(c_prev5) and c_prev5 > 0:
                m5 = (c_now - c_prev5) / c_prev5
                mom5[si, di] = m5
                if m5 > 0.02:
                    vote_mom[si, di] = 1
                elif m5 < -0.02:
                    vote_mom[si, di] = -1

            # ----------------------------------------------------------
            # Signal 2: 5-day OI momentum
            # ----------------------------------------------------------
            oi_now = OI[si, d]
            if not np.isnan(oi_now) and oi_now > 0:
                oi_prev5 = OI[si, max(0, d - 5)]
                if not np.isnan(oi_prev5) and oi_prev5 > 0:
                    om5 = (oi_now - oi_prev5) / oi_prev5
                    oi_mom5[si, di] = om5
                    if om5 > 0.05:
                        vote_oi[si, di] = 1
                    elif om5 < -0.05:
                        vote_oi[si, di] = -1

            # ----------------------------------------------------------
            # Signal 3: VDP EMA direction (buying pressure)
            # ----------------------------------------------------------
            hl = H[si, d] - L[si, d]
            if not np.isnan(hl) and hl > 0:
                cd = C[si, d]
                hd = H[si, d]
                ld = L[si, d]
                vd = V[si, d]
                if not (np.isnan(cd) or np.isnan(hd) or np.isnan(ld) or np.isnan(vd)):
                    vdp_val = vd * (2 * cd - hd - ld) / hl
                    alpha = 2.0 / 15
                    if vdp_e_inited[si]:
                        vdp_e_state[si] = alpha * vdp_val + (1 - alpha) * vdp_e_state[si]
                    else:
                        vdp_e_state[si] = vdp_val
                        vdp_e_inited[si] = True
                    vdp_ema[si, di] = vdp_e_state[si]

                    # Vote based on VDP EMA direction vs previous
                    if di >= 1 and not np.isnan(vdp_ema[si, di - 1]):
                        vdp_prev = vdp_ema[si, di - 1]
                        vdp_cur = vdp_e_state[si]
                        # Normalize by absolute scale
                        scale = max(abs(vdp_prev), abs(vdp_cur), 1.0)
                        vdp_chg = (vdp_cur - vdp_prev) / scale
                        if vdp_chg > 0.05:
                            vote_vdp[si, di] = 1
                        elif vdp_chg < -0.05:
                            vote_vdp[si, di] = -1

            # ----------------------------------------------------------
            # Signal 4: Donchian position (20-day breakout level)
            # ----------------------------------------------------------
            if di >= 20:
                h20 = H[si, max(0, d - 19):d + 1]
                l20 = L[si, max(0, d - 19):d + 1]
                h20v = h20[~np.isnan(h20)]
                l20v = l20[~np.isnan(l20)]
                if len(h20v) > 0 and len(l20v) > 0:
                    hh = np.max(h20v)
                    ll = np.min(l20v)
                    rng = hh - ll
                    if rng > 0:
                        dc = (c_now - ll) / rng
                        donchian[si, di] = dc
                        if dc > 0.8:
                            vote_donch[si, di] = 1   # Near 20-day high
                        elif dc < 0.2:
                            vote_donch[si, di] = -1   # Near 20-day low

            # ----------------------------------------------------------
            # Signal 5: Body ratio (candle conviction)
            # ----------------------------------------------------------
            if not np.isnan(hl) and hl > 0:
                od = O[si, d]
                if not np.isnan(od):
                    co = c_now - od
                    br = co / hl
                    body_ratio[si, di] = br
                    if br > 0.4:
                        vote_body[si, di] = 1   # Strong bullish candle
                    elif br < -0.4:
                        vote_body[si, di] = -1   # Strong bearish candle

            # ----------------------------------------------------------
            # ATR 10-day (for trail stop)
            # ----------------------------------------------------------
            trs = []
            for dd in range(max(1, d - 9), d + 1):
                hi = H[si, dd]
                lo = L[si, dd]
                pc = C[si, dd - 1]
                if np.isnan(hi) or np.isnan(lo):
                    continue
                tr = hi - lo
                if not np.isnan(pc):
                    tr = max(tr, abs(hi - pc), abs(lo - pc))
                trs.append(tr)
            if trs:
                atr10[si, di] = np.mean(trs)

        # Compute confluence for this day
        confluence[:, di] = (vote_mom[:, di] + vote_oi[:, di] +
                             vote_vdp[:, di] + vote_donch[:, di] +
                             vote_body[:, di])

    print(f"  Factors done ({time.time()-t0:.0f}s)", flush=True)

    return {
        'mom5': mom5, 'oi_mom5': oi_mom5, 'vdp_ema': vdp_ema,
        'donchian': donchian, 'body_ratio': body_ratio, 'atr10': atr10,
        'confluence': confluence,
        'vote_mom': vote_mom, 'vote_oi': vote_oi, 'vote_vdp': vote_vdp,
        'vote_donch': vote_donch, 'vote_body': vote_body,
    }


# =====================================================================
# BACKTEST ENGINE
# =====================================================================
def run_backtest(factors, NS, ND, dates, C, O, H, L, V, syms,
                 min_confluence=3, hold_max=5, trail_atr=2.0,
                 stop_loss=0.05, allow_short=False, use_top_n=1,
                 concentrated=False, min_trades=20):
    """
    Backtest the confluence ensemble strategy.

    Cash tracking:
      - Long entry:  cash -= entry_price * mult * lots * (1 + COMM)
      - Long exit:   cash += exit_price * mult * lots * (1 - COMM)
      - Short entry: cash += entry_price * mult * lots * (1 - COMM)
      - Short exit:  cash -= exit_price * mult * lots * (1 + COMM)
    """
    conf = factors['confluence']
    atr10_arr = factors['atr10']

    cash = float(CASH0)
    trades = []
    positions = []  # list of pos dicts
    peak_equity = float(CASH0)

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # Update peak equity
        # Compute total equity (cash + mark-to-market of positions)
        pos_equity = 0.0
        for pos in positions:
            c_m = C[pos['si'], di]
            if np.isnan(c_m) or c_m <= 0:
                c_m = pos['avg_entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            if pos['dir'] == 1:
                pos_equity += c_m * mult * pos['lots']
            else:
                pos_equity += (2 * pos['avg_entry'] - c_m) * mult * pos['lots']
        total_equity = cash + pos_equity
        if total_equity > peak_equity:
            peak_equity = total_equity

        # ----------------------------------------------------------
        # MANAGE EXISTING POSITIONS: check exits
        # ----------------------------------------------------------
        still_open = []
        for pos in positions:
            c = C[pos['si'], di]
            if np.isnan(c) or c <= 0:
                c = pos['avg_entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            days_held = di - pos['entry_di']

            # PnL calculation
            if pos['dir'] == 1:
                pnl = (c - pos['avg_entry']) * mult * pos['lots']
            else:
                pnl = (pos['avg_entry'] - c) * mult * pos['lots']
            cost_basis = pos['avg_entry'] * mult * pos['lots']
            pnl_pct = pnl / cost_basis * 100 if cost_basis > 0 else 0

            exit_reason = None

            # 1. Fixed stop loss
            if pnl_pct / 100 < -stop_loss:
                exit_reason = 'stop'

            # 2. Trailing stop (after day 1)
            if exit_reason is None and trail_atr > 0 and days_held >= 1:
                atr_val = pos.get('atr', 0)
                if atr_val > 0:
                    if pos['dir'] == 1:
                        trail = pos.get('trail_price', pos['avg_entry'])
                        new_trail = c - trail_atr * atr_val
                        if new_trail > trail:
                            pos['trail_price'] = new_trail
                        if c <= pos['trail_price'] and days_held >= 1:
                            exit_reason = 'trail'
                    else:
                        trail = pos.get('trail_price', pos['avg_entry'])
                        new_trail = c + trail_atr * atr_val
                        if new_trail < trail:
                            pos['trail_price'] = new_trail
                        if c >= pos['trail_price'] and days_held >= 1:
                            exit_reason = 'trail'

            # 3. Signal reversal
            if exit_reason is None and days_held >= 2:
                cur_conf = conf[pos['si'], di]
                if pos['dir'] == 1 and cur_conf < 1:
                    exit_reason = 'reversal'
                elif pos['dir'] == -1 and cur_conf > -1:
                    exit_reason = 'reversal'

            # 4. Time stop
            if exit_reason is None and days_held >= hold_max:
                exit_reason = 'time'

            if exit_reason is not None:
                # Close position: return exit notional minus commission
                # PnL = (exit - entry) * mult * lots * dir
                # Cash += exit_notional - commission
                cost_out = c * mult * pos['lots'] * COMM
                cash += c * mult * pos['lots'] - cost_out

                trades.append({
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'days': days_held,
                    'di': di,
                    'year': year,
                    'sym': pos['sym'],
                    'dir': pos['dir'],
                    'reason': exit_reason,
                    'confluence': pos['confluence_at_entry'],
                })
            else:
                still_open.append(pos)

        positions = still_open

        # ----------------------------------------------------------
        # ENTRY: find top candidates by confluence
        # ----------------------------------------------------------
        n_slots = use_top_n - len(positions)
        if n_slots <= 0:
            continue

        # Already-held symbols
        held_syms = {pos['si'] for pos in positions}

        candidates = []
        for si in range(NS):
            if si in held_syms:
                continue
            c_now = C[si, di]
            if np.isnan(c_now) or c_now <= 0:
                continue

            cf = conf[si, di]
            abs_cf = abs(cf)

            if abs_cf < min_confluence:
                continue

            # Long candidate
            if cf >= min_confluence:
                candidates.append((si, cf, 1, c_now))
            # Short candidate
            if allow_short and cf <= -min_confluence:
                candidates.append((si, -cf, -1, c_now))

        if not candidates:
            continue

        # Sort by confluence strength (descending)
        candidates.sort(key=lambda x: -x[1])

        for si, cf_strength, direction, price in candidates[:n_slots]:
            sym = syms[si]
            mult = MULT.get(sym, DEF_MULT)
            notional = price * mult
            if notional <= 0:
                continue

            # Position sizing
            if concentrated:
                if cf_strength >= 4:
                    alloc = cash  # 100% capital
                elif cf_strength >= 3:
                    alloc = cash * 0.5  # 50% capital
                else:
                    continue  # Should not reach here due to min_confluence filter
            else:
                alloc = cash / use_top_n

            lots = int(alloc / (notional * (1 + COMM)))
            if lots <= 0:
                continue

            cost = notional * lots * (1 + COMM)
            if cost > cash:
                lots = int(cash / (notional * (1 + COMM)))
                if lots <= 0:
                    continue
                cost = notional * lots * (1 + COMM)

            if cost > cash:
                continue

            # Reserve capital (treats cash as margin account)
            cash -= cost

            # Get ATR for trailing stop
            atr_val = atr10_arr[si, di]
            if np.isnan(atr_val):
                atr_val = 0

            positions.append({
                'si': si,
                'sym': sym,
                'avg_entry': price,
                'entry_di': di,
                'lots': lots,
                'dir': direction,
                'atr': atr_val,
                'trail_price': price if direction == 1 else price,
                'confluence_at_entry': cf_strength if direction == 1 else -cf_strength,
            })

    # Close remaining positions at end
    for pos in positions:
        c = C[pos['si'], ND - 1]
        if np.isnan(c) or c <= 0:
            c = pos['avg_entry']
        mult = MULT.get(pos['sym'], DEF_MULT)
        pnl = (c - pos['avg_entry']) * mult * pos['lots'] * pos['dir']
        cost_out = c * mult * pos['lots'] * COMM
        cash += c * mult * pos['lots'] - cost_out
        cost_basis = pos['avg_entry'] * mult * pos['lots']
        pnl_pct = pnl / cost_basis * 100 if cost_basis > 0 else 0
        trades.append({
            'pnl': pnl, 'pnl_pct': pnl_pct,
            'days': ND - 1 - pos['entry_di'],
            'di': ND - 1, 'year': dates[ND-1].year,
            'sym': pos['sym'], 'dir': pos['dir'],
            'reason': 'end', 'confluence': pos['confluence_at_entry'],
        })

    if len(trades) < min_trades:
        return None

    # ----------------------------------------------------------
    # Compute statistics
    # ----------------------------------------------------------
    equity = float(CASH0)
    peak = float(CASH0)
    max_dd = 0.0
    for t in sorted(trades, key=lambda x: x['di']):
        equity += t['pnl']
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    days_total = (dates[ND - 1] - dates[MIN_TRAIN]).days
    yr = max(days_total / 365.25, 0.01)
    ann = ((cash / CASH0) ** (1 / yr) - 1) * 100

    nw = sum(1 for t in trades if t['pnl'] > 0)
    wr = nw / len(trades) * 100
    avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl'] > 0]) if nw > 0 else 0
    avg_loss = np.mean([abs(t['pnl_pct']) for t in trades if t['pnl'] <= 0]) if nw < len(trades) else 0

    # Yearly breakdown
    year_stats = {}
    for t in trades:
        y = t['year']
        if y not in year_stats:
            year_stats[y] = {'trades': 0, 'wins': 0, 'total_pnl': 0.0, 'pnl_pct_sum': 0.0}
        year_stats[y]['trades'] += 1
        if t['pnl'] > 0:
            year_stats[y]['wins'] += 1
        year_stats[y]['total_pnl'] += t['pnl_pct']
        year_stats[y]['pnl_pct_sum'] += t['pnl']

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        r = t['reason']
        if r not in exit_reasons:
            exit_reasons[r] = 0
        exit_reasons[r] += 1

    return {
        'ann': round(ann, 1),
        'n': len(trades),
        'wr': round(wr, 1),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'dd': round(max_dd, 1),
        'final': round(cash, 0),
        'years': year_stats,
        'exit_reasons': exit_reasons,
    }


def make_label(min_confluence, hold_max, trail_atr, stop_loss,
               allow_short, use_top_n, concentrated):
    """Create a human-readable label for the test configuration."""
    parts = [f"conf{min_confluence}", f"h{hold_max}", f"tr{trail_atr}",
             f"sl{stop_loss}"]
    if allow_short:
        parts.append("S")
    parts.append(f"top{use_top_n}")
    if concentrated:
        parts.append("CONC")
    return "_".join(parts)


# =====================================================================
# MAIN
# =====================================================================
if __name__ == '__main__':
    print("=" * 100, flush=True)
    print("  Alpha Futures V20 — Multi-Strategy Ensemble (Confluence Scoring)", flush=True)
    print("=" * 100, flush=True)

    t_start = time.time()

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    print(f"  Data: {NS} commodities, {ND} days", flush=True)

    # Pre-compute all factors
    factors = precompute_factors(NS, ND, C, O, H, L, V, OI)

    # Quick stats on confluence distribution
    conf = factors['confluence']
    print(f"\n  Confluence distribution:", flush=True)
    for v in range(-5, 6):
        cnt = np.sum(conf == v)
        if cnt > 0:
            print(f"    confluence={v:+d}: {cnt:8d} observations", flush=True)

    # ----------------------------------------------------------
    # TEST CONFIGURATIONS
    # ----------------------------------------------------------
    configs = []
    for min_conf in [2, 3, 4]:
        for hm in [3, 5, 7]:
            for ta in [1.5, 2.0, 3.0]:
                for sl in [0.03, 0.05]:
                    for allow_s in [True, False]:
                        for top_n in [1, 3]:
                            configs.append({
                                'min_confluence': min_conf,
                                'hold_max': hm,
                                'trail_atr': ta,
                                'stop_loss': sl,
                                'allow_short': allow_s,
                                'use_top_n': top_n,
                                'concentrated': False,
                            })

    # Concentrated variants
    for min_conf in [3, 4]:
        for hm in [3, 5, 7]:
            for ta in [1.5, 2.0, 3.0]:
                for sl in [0.03, 0.05]:
                    for allow_s in [True, False]:
                        configs.append({
                            'min_confluence': min_conf,
                            'hold_max': hm,
                            'trail_atr': ta,
                            'stop_loss': sl,
                            'allow_short': allow_s,
                            'use_top_n': 1,
                            'concentrated': True,
                        })

    print(f"\n  Testing {len(configs)} configurations...", flush=True)

    results = []
    for ci, cfg in enumerate(configs):
        label = make_label(**cfg)
        r = run_backtest(factors, NS, ND, dates, C, O, H, L, V, syms, **cfg)
        if r is not None:
            r['label'] = label
            r['cfg'] = cfg
            results.append(r)
        if (ci + 1) % 50 == 0:
            print(f"    {ci+1}/{len(configs)} configs tested", flush=True)

    print(f"  {len(results)} configs produced results", flush=True)

    # ----------------------------------------------------------
    # SORT AND DISPLAY TOP 30
    # ----------------------------------------------------------
    results.sort(key=lambda x: -x['ann'])

    print(f"\n{'='*130}", flush=True)
    print(f"  TOP 30 RESULTS", flush=True)
    print(f"  {'#':>3s}  {'Config':<40s}  {'Ann':>8s} {'N':>5s} {'WR':>6s} "
          f"{'AvgW':>7s} {'AvgL':>7s} {'DD':>6s} {'Final':>12s}", flush=True)
    print(f"  {'-'*120}", flush=True)
    for i, r in enumerate(results[:30]):
        print(f"  {i+1:3d}  {r['label']:<40s}  {r['ann']:+8.1f}% {r['n']:5d} "
              f"{r['wr']:5.1f}% {r['avg_win']:+6.2f}% {r['avg_loss']:6.2f}% "
              f"{r['dd']:5.1f}% {r['final']:>12.0f}", flush=True)

    # ----------------------------------------------------------
    # YEARLY BREAKDOWN FOR TOP 5
    # ----------------------------------------------------------
    for i, r in enumerate(results[:5]):
        print(f"\n  Yearly breakdown #{i+1}: {r['label']} "
              f"(Ann={r['ann']:+.1f}%, DD={r['dd']:.1f}%)", flush=True)
        for y in sorted(r['years'].keys()):
            s = r['years'][y]
            wr_y = s['wins'] / max(s['trades'], 1) * 100
            print(f"    {y}: {s['trades']:4d} trades, WR={wr_y:5.1f}%, "
                  f"pnl={s['total_pnl']:+.1f}%", flush=True)

        # Exit reason breakdown
        print(f"    Exit reasons: {r['exit_reasons']}", flush=True)

    # ----------------------------------------------------------
    # BEST PER GROUP
    # ----------------------------------------------------------
    print(f"\n  Best per confluence threshold:", flush=True)
    for mc in [2, 3, 4]:
        best = [r for r in results if r['cfg']['min_confluence'] == mc and not r['cfg']['concentrated']]
        if best:
            b = max(best, key=lambda x: x['ann'])
            print(f"    min_conf={mc}: {b['ann']:+.1f}% (N={b['n']}, WR={b['wr']:.0f}%, "
                  f"DD={b['dd']:.1f}%) -- {b['label']}", flush=True)

    print(f"\n  Best concentrated:", flush=True)
    conc_results = [r for r in results if r['cfg']['concentrated']]
    if conc_results:
        for cr in sorted(conc_results, key=lambda x: -x['ann'])[:5]:
            print(f"    {cr['ann']:+.1f}% (N={cr['n']}, WR={cr['wr']:.0f}%, "
                  f"DD={cr['dd']:.1f}%) -- {cr['label']}", flush=True)
    else:
        print(f"    No concentrated results", flush=True)

    # ----------------------------------------------------------
    # BEST SHORT vs LONG-ONLY
    # ----------------------------------------------------------
    print(f"\n  Best long-only:", flush=True)
    long_only = [r for r in results if not r['cfg']['allow_short'] and not r['cfg']['concentrated']]
    if long_only:
        for lr in sorted(long_only, key=lambda x: -x['ann'])[:3]:
            print(f"    {lr['ann']:+.1f}% (N={lr['n']}, WR={lr['wr']:.0f}%, "
                  f"DD={lr['dd']:.1f}%) -- {lr['label']}", flush=True)

    print(f"\n  Best long+short:", flush=True)
    ls_results = [r for r in results if r['cfg']['allow_short'] and not r['cfg']['concentrated']]
    if ls_results:
        for lr in sorted(ls_results, key=lambda x: -x['ann'])[:3]:
            print(f"    {lr['ann']:+.1f}% (N={lr['n']}, WR={lr['wr']:.0f}%, "
                  f"DD={lr['dd']:.1f}%) -- {lr['label']}", flush=True)

    print(f"\n{'='*100}", flush=True)
    print(f"  Total time: {time.time()-t_start:.0f}s", flush=True)
    print(f"{'='*100}", flush=True)
