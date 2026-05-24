"""
Alpha Futures V29 — Hurst Exponent + Shannon Entropy Regime Gate
================================================================
Core idea: Use fractal market structure to classify regime, then apply
the appropriate signal type for that regime.

REGIME CLASSIFICATION (per-symbol, per-day):
  - H > 0.55 + low entropy  --> TRENDING  --> momentum entry
  - H < 0.45 + high entropy --> CHAOTIC   --> NO TRADE
  - otherwise               --> NEUTRAL   --> VDP delta flip only

ENTRY:
  TRENDING: mom5 direction + VDP EMA confirmation (same sign)
  NEUTRAL:  VDP delta flip (yesterday negative, today positive = long entry)

EXIT: time (3-5 days), signal flip, rotation to stronger candidate
Single position, P1 concentrated, no leverage, long only.

Data: 68 Chinese commodity futures, 2016-2026, daily OHLCV+OI
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

# ============================================================
# CONTRACT MULTIPLIERS (same as v14b)
# ============================================================
MULT = {
    'agfi': 15, 'alfi': 5, 'aufi': 1000, 'bufi': 10, 'cufi': 5,
    'fufi': 10, 'rbfi': 10, 'znfi': 5, 'nifi': 1, 'hcfi': 10,
    'spfi': 10, 'ssfi': 5, 'sffi': 5, 'smfi': 5, 'pbfi': 5,
    'snfi': 1, 'rufi': 10, 'wrfff': 10,
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


# ============================================================
# PRECOMPUTATION FUNCTIONS
# ============================================================

def compute_hurst_rs(price_series, window=100):
    """Compute rolling Hurst exponent via R/S analysis.

    For each window of 'window' days, split into sub-blocks and compute
    the rescaled range statistic, then regress log(R/S) vs log(n) to
    estimate H.

    H > 0.5  : trending (persistent)
    H ~ 0.5  : random walk
    H < 0.5  : mean-reverting (anti-persistent)
    """
    n = len(price_series)
    hurst = np.full(n, np.nan)
    if n < window + 20:
        return hurst

    # Sub-block sizes: need at least 2 blocks per size within the window
    block_sizes = []
    for bs in [8, 10, 12, 16, 20, 25, 33, 50]:
        if window // bs >= 2:
            block_sizes.append(bs)

    if len(block_sizes) < 2:
        return hurst

    log_ns = np.log(np.array(block_sizes, dtype=float))

    for i in range(window, n):
        seg = price_series[i - window:i]
        valid = seg[~np.isnan(seg)]
        if len(valid) < window * 0.8:
            continue

        log_rs_list = []
        for bs in block_sizes:
            nblocks = len(valid) // bs
            if nblocks < 2:
                continue
            rs_vals = []
            for b in range(nblocks):
                block = valid[b * bs:(b + 1) * bs]
                mean_b = np.mean(block)
                cumdev = np.cumsum(block - mean_b)
                if len(cumdev) == 0:
                    continue
                R = np.max(cumdev) - np.min(cumdev)
                S = np.std(block, ddof=1) if bs > 1 else 0.0
                if S > 0 and R > 0:
                    rs_vals.append(R / S)
            if rs_vals:
                log_rs_list.append(np.log(np.mean(rs_vals)))

        if len(log_rs_list) >= 2:
            ln_rs = np.array(log_rs_list)
            valid_sizes = log_ns[:len(ln_rs)]
            mx = np.mean(valid_sizes)
            my = np.mean(ln_rs)
            denom = np.sum((valid_sizes - mx) ** 2)
            if denom > 0:
                slope = np.sum((valid_sizes - mx) * (ln_rs - my)) / denom
                hurst[i] = np.clip(slope, 0.1, 0.9)

    return hurst


def compute_shannon_entropy(price_series, window=50, n_bins=5):
    """Compute rolling Shannon entropy of daily returns.

    Low entropy = returns cluster in few bins = more predictable / trending
    High entropy = returns spread uniformly = chaotic / noisy
    Normalized to [0, 1] where 1 = uniform distribution.
    """
    n = len(price_series)
    entropy = np.full(n, np.nan)
    if n < window + 2:
        return entropy

    # Compute daily returns
    rets = np.full(n, np.nan)
    for i in range(1, n):
        if (not np.isnan(price_series[i]) and not np.isnan(price_series[i - 1])
                and price_series[i - 1] > 0):
            rets[i] = (price_series[i] - price_series[i - 1]) / price_series[i - 1]

    max_entropy = np.log(n_bins)  # Normalization constant

    for i in range(window + 1, n):
        r = rets[i - window:i]
        r = r[~np.isnan(r)]
        if len(r) < window * 0.7:
            continue

        try:
            edges = np.linspace(np.min(r), np.max(r), n_bins + 1)
            counts, _ = np.histogram(r, bins=edges)
            probs = counts / len(r)
            probs = probs[probs > 0]
            H = -np.sum(probs * np.log(probs))
            entropy[i] = H / max_entropy if max_entropy > 0 else 0.5
        except Exception:
            continue

    return entropy


def compute_vdp_ema_series(close, high, low, volume, ema_span=15):
    """Compute VDP (Volume Delta Pressure) EMA for all days.
    VDP = V * (2C - H - L) / (H - L)
    Measures net buying/selling pressure. EMA smoothed.
    """
    n = len(close)
    vdp_ema = np.full(n, np.nan)
    if n < 2:
        return vdp_ema

    alpha = 2.0 / (ema_span + 1)
    running = 0.0
    started = False

    for i in range(n):
        hl = high[i] - low[i]
        if np.isnan(hl) or hl <= 0:
            continue
        c = close[i]
        v = volume[i]
        if np.isnan(c) or np.isnan(v) or v <= 0:
            continue

        vdp = v * (2 * c - high[i] - low[i]) / hl

        if not started:
            running = vdp
            started = True
        else:
            running = alpha * vdp + (1 - alpha) * running
        vdp_ema[i] = running

    return vdp_ema


def compute_oi_flow(oi, price, window=5):
    """OI flow = sign(price_change) * delta(OI).
    Positive OI flow = new money entering in trend direction.
    Returns rolling sum of OI flow over 'window' days.
    """
    n = len(oi)
    flow = np.full(n, np.nan)
    if n < window + 2:
        return flow

    daily_flow = np.full(n, np.nan)
    for i in range(1, n):
        if (np.isnan(oi[i]) or np.isnan(oi[i - 1])
                or np.isnan(price[i]) or np.isnan(price[i - 1])):
            continue
        if price[i - 1] <= 0:
            continue
        d_oi = oi[i] - oi[i - 1]
        d_price = price[i] - price[i - 1]
        daily_flow[i] = np.sign(d_price) * d_oi

    for i in range(window, n):
        vals = daily_flow[i - window + 1:i + 1]
        vals = vals[~np.isnan(vals)]
        if len(vals) >= window * 0.5:
            flow[i] = np.sum(vals)

    return flow


# ============================================================
# ENTRY SCORE
# ============================================================

def compute_entry_score(si, di, C, hurst_arr, entropy_arr,
                        vdp_ema_arr, oi_flow_arr, mom5, regime_arr,
                        hurst_hi, hurst_lo, ent_low, ent_high):
    """Compute entry score for symbol si on day di based on regime.

    Returns:
        positive float = long entry score (higher = better)
        0 = no signal
        np.nan = insufficient data
    """
    h = hurst_arr[si, di]
    e = entropy_arr[si, di]
    m5 = mom5[si, di]
    vd = vdp_ema_arr[si, di]

    # Need at least Hurst, entropy, momentum
    if np.isnan(h) or np.isnan(e) or np.isnan(m5):
        return np.nan

    # Need valid price
    c = C[si, di]
    if np.isnan(c) or c <= 0:
        return np.nan

    # --- REGIME CLASSIFICATION ---
    if h > hurst_hi and e < ent_low:
        regime = 2  # TRENDING
    elif h < hurst_lo and e > ent_high:
        regime = 0  # CHAOTIC -> no trade
    else:
        regime = 1  # NEUTRAL

    regime_arr[si, di] = regime

    # --- CHAOTIC: no trade ---
    if regime == 0:
        return 0

    # --- TRENDING: momentum + VDP confirmation ---
    if regime == 2:
        if m5 <= 0:
            return 0

        score = m5 * 10  # Scale up momentum

        # VDP confirmation
        if not np.isnan(vd):
            if vd > 0:
                score *= 1.5  # VDP confirms buying pressure
            else:
                score *= 0.3  # VDP contradicts

        # OI flow bonus
        of = oi_flow_arr[si, di]
        if not np.isnan(of) and of > 0:
            score *= 1.2

        return np.clip(score, 0, 2)

    # --- NEUTRAL: VDP delta flip only ---
    if regime == 1:
        vd_prev = vdp_ema_arr[si, di - 1] if di > 0 else np.nan
        if np.isnan(vd) or np.isnan(vd_prev):
            return 0

        # Flip: negative -> positive = buying pressure emerging
        if vd_prev < 0 and vd > 0:
            score = 0.3  # Base score for flip
            if m5 > 0:
                score += m5 * 3
            of = oi_flow_arr[si, di]
            if not np.isnan(of) and of > 0:
                score += 0.1
            return np.clip(score, 0, 1.5)

        return 0

    return 0


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(NS, ND, dates, C, O, H, L, V, OI, syms,
                 hurst_arr, entropy_arr, vdp_ema_arr, oi_flow_arr,
                 mom5, regime_arr,
                 name='v29',
                 hold_min=3, hold_max=5,
                 ent_low=0.85, ent_high=0.92,
                 hurst_hi=0.55, hurst_lo=0.45):
    """Run single-position rotation backtest with regime gate."""
    cash = float(CASH0)
    trades = []
    pos = None
    last_exit = {}

    # Local reference for speed
    _score = compute_entry_score

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # === POSITION MANAGEMENT ===
        if pos is not None:
            c = C[pos['si'], di]
            if np.isnan(c) or c <= 0:
                c = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = c * mult * pos['lots']
            pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
            cost_basis = pos['entry'] * mult * pos['lots']
            pnl_pct = pnl / cost_basis * 100 if cost_basis > 0 else 0
            days_held = di - pos['entry_di']

            exit_reason = None

            # 1. Signal flip: momentum reversed + VDP confirms reversal
            if exit_reason is None and days_held >= 2:
                cur_mom = mom5[pos['si'], di]
                cur_vdp = vdp_ema_arr[pos['si'], di]
                if pos['dir'] == 1:
                    if (not np.isnan(cur_mom) and cur_mom < -0.01
                            and not np.isnan(cur_vdp) and cur_vdp < 0):
                        exit_reason = 'flip'
                elif pos['dir'] == -1:
                    if (not np.isnan(cur_mom) and cur_mom > 0.01
                            and not np.isnan(cur_vdp) and cur_vdp > 0):
                        exit_reason = 'flip'

            # 2. Time exit
            if exit_reason is None and days_held >= hold_max:
                exit_reason = 'time'

            # 3. Rotation: check for stronger candidate after hold_min days
            if exit_reason is None and days_held >= hold_min:
                best_score = 0
                best_si = -1
                for sj in range(NS):
                    sc = _score(sj, di, C, hurst_arr, entropy_arr,
                                vdp_ema_arr, oi_flow_arr, mom5, regime_arr,
                                hurst_hi, hurst_lo, ent_low, ent_high)
                    if not np.isnan(sc) and sc > best_score:
                        best_score = sc
                        best_si = sj

                if best_si >= 0 and best_si != pos['si']:
                    cand_regime = regime_arr[best_si, di]
                    if (not np.isnan(cand_regime) and cand_regime >= 1
                            and best_score > pos['entry_score'] * 1.5 + 0.02):
                        exit_reason = 'rotate'

            if exit_reason:
                cost_out = mkt_val * COMM
                cash += mkt_val - cost_out
                trades.append({
                    'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                    'days': days_held, 'di': di, 'year': year,
                    'sym': pos['sym'], 'dir': pos['dir'],
                    'reason': exit_reason
                })
                last_exit[pos['sym']] = di
                pos = None

        # === ENTRY SCAN ===
        if pos is None:
            best_score = 0
            best_si = -1

            for si in range(NS):
                sym = syms[si]
                if sym in last_exit and di - last_exit[sym] < 2:
                    continue

                sc = _score(si, di, C, hurst_arr, entropy_arr,
                            vdp_ema_arr, oi_flow_arr, mom5, regime_arr,
                            hurst_hi, hurst_lo, ent_low, ent_high)
                if not np.isnan(sc) and sc > best_score:
                    best_score = sc
                    best_si = si

            if best_si >= 0 and best_score > 0:
                c = C[best_si, di]
                if np.isnan(c) or c <= 0:
                    pass  # skip, go to next day
                else:
                    sym = syms[best_si]
                    mult = MULT.get(sym, DEF_MULT)
                    notional = c * mult
                    if notional > 0:
                        lots = int(cash / notional)
                        if lots > 0:
                            cost_in = notional * lots * (1 + COMM)
                            if cost_in <= cash:
                                cash -= cost_in
                                pos = {
                                    'si': best_si, 'entry': c, 'entry_di': di,
                                    'lots': lots, 'dir': 1, 'sym': sym,
                                    'entry_score': best_score
                                }

    # Close any remaining position at end
    if pos is not None:
        c = C[pos['si'], ND - 1]
        if np.isnan(c) or c <= 0:
            c = pos['entry']
        mult = MULT.get(pos['sym'], DEF_MULT)
        pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
        cash += c * mult * pos['lots'] * (1 - COMM)
        trades.append({
            'pnl_pct': pnl / (pos['entry'] * mult * pos['lots']) * 100,
            'pnl_abs': pnl, 'days': ND - 1 - pos['entry_di'],
            'di': ND - 1, 'year': dates[ND - 1].year,
            'sym': pos['sym'], 'dir': pos['dir'], 'reason': 'end'
        })

    if len(trades) < 10:
        return None

    # === STATISTICS ===
    equity = float(CASH0)
    peak = float(CASH0)
    max_dd = 0
    for t in sorted(trades, key=lambda x: x['di']):
        equity += t['pnl_abs']
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd

    days_total = (dates[ND - 1] - dates[MIN_TRAIN]).days
    yr = max(days_total / 365.25, 0.01)
    ann = ((cash / CASH0) ** (1 / yr) - 1) * 100

    nw = sum(1 for t in trades if t['pnl_abs'] > 0)
    wr = nw / len(trades) * 100
    avg_pnl = np.mean([t['pnl_pct'] for t in trades])
    avg_days = np.mean([t['days'] for t in trades])
    avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_abs'] > 0]) if nw > 0 else 0
    avg_loss = (np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_abs'] <= 0])
                if nw < len(trades) else 0)

    # Yearly breakdown
    year_stats = {}
    for t in trades:
        y = t['year']
        if y not in year_stats:
            year_stats[y] = {'n': 0, 'w': 0, 'pnl': 0.0}
        year_stats[y]['n'] += 1
        if t['pnl_abs'] > 0:
            year_stats[y]['w'] += 1
        year_stats[y]['pnl'] += t['pnl_pct']

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        r = t['reason']
        if r not in reasons:
            reasons[r] = {'n': 0, 'w': 0, 'pnl': 0.0}
        reasons[r]['n'] += 1
        if t['pnl_abs'] > 0:
            reasons[r]['w'] += 1
        reasons[r]['pnl'] += t['pnl_pct']

    return {
        'name': name, 'ann': round(ann, 1), 'n': len(trades),
        'wr': round(wr, 1), 'dd': round(max_dd, 1),
        'avg_pnl': round(avg_pnl, 3), 'avg_days': round(avg_days, 1),
        'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
        'final': round(cash, 0), 'years': year_stats, 'reasons': reasons
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 95, flush=True)
    print("  Alpha Futures V29 — Hurst + Entropy Regime Gate with VDP/OI Flow", flush=True)
    print("  Regime: TRENDING (H>0.55,low ent) | NEUTRAL | CHAOTIC (H<0.45,high ent)", flush=True)
    print("=" * 95, flush=True)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    print(f"\n  Precomputing factors for {NS} symbols x {ND} days...", flush=True)
    t0 = time.time()

    # Allocate arrays
    hurst_arr = np.full((NS, ND), np.nan)
    entropy_arr = np.full((NS, ND), np.nan)
    vdp_ema_arr = np.full((NS, ND), np.nan)
    oi_flow_arr = np.full((NS, ND), np.nan)
    mom5 = np.full((NS, ND), np.nan)
    regime_arr = np.full((NS, ND), np.nan)

    for si in range(NS):
        if si % 10 == 0:
            print(f"    Symbol {si}/{NS}...", flush=True)

        close_s = C[si]
        high_s = H[si]
        low_s = L[si]
        vol_s = V[si]
        oi_s = OI[si]

        # --- Hurst Exponent (R/S, 100-day window) ---
        hurst_arr[si] = compute_hurst_rs(close_s, window=100)

        # --- Shannon Entropy (50-day window) ---
        entropy_arr[si] = compute_shannon_entropy(close_s, window=50, n_bins=5)

        # --- VDP EMA (15-day span) ---
        vdp_ema_arr[si] = compute_vdp_ema_series(close_s, high_s, low_s, vol_s, ema_span=15)

        # --- OI Flow (5-day rolling sum) ---
        oi_flow_arr[si] = compute_oi_flow(oi_s, close_s, window=5)

        # --- 5-day momentum ---
        for di in range(5, ND):
            c_now = close_s[di]
            c_prev = close_s[di - 5]
            if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                mom5[si, di] = (c_now - c_prev) / c_prev

    print(f"  Factors done ({time.time() - t0:.0f}s)", flush=True)

    # Pre-classify regimes for stats display (baseline thresholds)
    hurst_hi_base = 0.55
    hurst_lo_base = 0.45
    ent_low_base = 0.85
    ent_high_base = 0.92

    regime_counts = {0: 0, 1: 0, 2: 0}
    total_pts = 0
    for si in range(NS):
        for di in range(MIN_TRAIN, ND):
            h = hurst_arr[si, di]
            e = entropy_arr[si, di]
            if np.isnan(h) or np.isnan(e):
                continue
            total_pts += 1
            if h > hurst_hi_base and e < ent_low_base:
                regime_counts[2] += 1
            elif h < hurst_lo_base and e > ent_high_base:
                regime_counts[0] += 1
            else:
                regime_counts[1] += 1

    print(f"\n  Regime distribution (baseline: H_hi={hurst_hi_base}, H_lo={hurst_lo_base}, "
          f"E_lo={ent_low_base}, E_hi={ent_high_base}):", flush=True)
    for k, v in sorted(regime_counts.items()):
        label = {0: 'CHAOTIC', 1: 'NEUTRAL', 2: 'TRENDING'}[k]
        pct = v / max(total_pts, 1) * 100
        print(f"    {label:10s}: {v:8d} ({pct:5.1f}%)", flush=True)

    # ============================================================
    # RUN PARAMETER SWEEP
    # ============================================================
    print(f"\n  Running backtest sweep...", flush=True)

    configs = []
    for hold_max in [3, 4, 5]:
        for hh in [0.52, 0.55, 0.58]:
            for hl in [0.42, 0.45, 0.48]:
                for el in [0.80, 0.85, 0.90]:
                    for eh in [0.90, 0.92, 0.95]:
                        if el >= eh:
                            continue
                        if hh <= hl:
                            continue
                        name = f"Hmx{hold_max}_Hh{hh}_Hl{hl}_El{el}_Eh{eh}"
                        configs.append((hold_max, hh, hl, el, eh, name))

    print(f"  {len(configs)} configurations", flush=True)

    results = []
    t1 = time.time()
    for ci, (hold_max, hh, hl, el, eh, cfg_name) in enumerate(configs):
        if ci % 50 == 0:
            elapsed = time.time() - t1
            print(f"    Config {ci}/{len(configs)} ({len(results)} profitable, "
                  f"{elapsed:.0f}s)...", flush=True)

        # Fresh regime array for each config (regime depends on thresholds)
        regime_arr_cfg = np.full((NS, ND), np.nan)

        r = run_backtest(NS, ND, dates, C, O, H, L, V, OI, syms,
                         hurst_arr, entropy_arr, vdp_ema_arr, oi_flow_arr,
                         mom5, regime_arr_cfg,
                         name=cfg_name,
                         hold_min=max(2, hold_max - 2),
                         hold_max=hold_max,
                         ent_low=el, ent_high=eh,
                         hurst_hi=hh, hurst_lo=hl)
        if r and r['ann'] > 0:
            results.append(r)

    print(f"\n  Sweep done ({time.time() - t1:.0f}s, {len(results)} profitable configs)", flush=True)
    results.sort(key=lambda x: -x['ann'])

    # ============================================================
    # PRINT RESULTS
    # ============================================================
    print(f"\n{'=' * 105}", flush=True)
    print(f"  TOP 30 RESULTS", flush=True)
    print(f"  {'Config':45s} | {'Ann':>8s} {'WR':>6s} {'N':>4s} {'DD':>7s} "
          f"{'AvgW':>7s} {'AvgL':>7s} {'AvgD':>5s}", flush=True)
    print(f"  {'-' * 45}-+-{'-' * 8}-{'-' * 6}-{'-' * 4}-{'-' * 7}-"
          f"{'-' * 7}-{'-' * 7}-{'-' * 5}", flush=True)
    for r in results[:30]:
        print(f"  {r['name']:45s} | {r['ann']:+8.1f}% {r['wr']:5.1f}% {r['n']:4d} "
              f"{r['dd']:6.1f}% {r['avg_win']:+6.2f}% {r['avg_loss']:6.2f}% "
              f"{r['avg_days']:4.1f}d", flush=True)

    # Detailed breakdown for top 5
    for i, r in enumerate(results[:5]):
        print(f"\n  #{i + 1}: {r['name']}  Ann={r['ann']:+.1f}%  WR={r['wr']:.0f}%  "
              f"DD={r['dd']:.1f}%  Final={r['final']:.0f}", flush=True)
        print(f"       AvgWin={r['avg_win']:+.2f}%  AvgLoss={r['avg_loss']:.2f}%  "
              f"AvgDays={r['avg_days']:.1f}", flush=True)

        # Exit reason breakdown
        print(f"       Exit reasons:", flush=True)
        for reason, s in sorted(r['reasons'].items(), key=lambda x: -x[1]['n']):
            rwr = s['w'] / max(s['n'], 1) * 100
            print(f"         {reason:10s}: {s['n']:4d}t  WR={rwr:4.0f}%  "
                  f"pnl={s['pnl']:+.1f}%", flush=True)

        # Yearly breakdown
        print(f"       Yearly:", flush=True)
        for y in sorted(r['years'].keys()):
            s = r['years'][y]
            wr_y = s['w'] / max(s['n'], 1) * 100
            print(f"         {y}: {s['n']:3d}t  WR={wr_y:4.0f}%  "
                  f"pnl={s['pnl']:+.1f}%", flush=True)

    print(f"\n{'=' * 105}", flush=True)
    if results:
        best = results[0]
        print(f"  BEST: {best['name']}  Ann={best['ann']:+.1f}%  WR={best['wr']:.0f}%  "
              f"DD={best['dd']:.1f}%  N={best['n']}", flush=True)
    else:
        print(f"  No profitable configurations found.", flush=True)
    print(f"{'=' * 105}", flush=True)
