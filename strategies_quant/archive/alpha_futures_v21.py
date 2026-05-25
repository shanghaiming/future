"""
Alpha Futures V21 — Kelly Criterion Adaptive Position Sizing
============================================================
Core innovation: Use Kelly criterion to dynamically size positions based on
rolling win rate and payoff ratio, instead of fixed 100% capital per trade.

Key components:
  1. SCORING: mom5 base signal enhanced with OI momentum + VDP direction bonuses
  2. KELLY SIZING: kelly_f = (p*b - q) / b, using rolling 50-trade stats
     Applied as 3/4 Kelly (or configurable fraction) for safety
  3. DRAWDOWN CIRCUIT BREAKER: DD<5% full, 5-15% half, 15-25% quarter, >25% halt
  4. VOL REGIME: 20d realized vol percentile (252d lookback) adjusts position
     Low vol (<30th pct): +50% size; High vol (>70th pct): -50% size
  5. TOP-N ROTATION: rank all commodities, hold top 1 or 3, rebalance after
     minimum hold period

Test grid over: base_signal, kelly_fraction, hold_min, trail_atr, stop_loss,
                top_n, dd_control, vol_regime
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


# ============================================================
# SIGNAL SCORING
# ============================================================

def compute_scores(NS, ND, C, O, H, L, V, OI, mode='composite'):
    """Pre-compute composite scores for all symbols x dates.
    mode: 'mom5', 'mom3', 'composite'
    Returns score array (NS, ND), positive=long, negative=short.
    """
    mom3 = np.full((NS, ND), np.nan)
    mom5 = np.full((NS, ND), np.nan)
    oi_mom5 = np.full((NS, ND), np.nan)
    vdp_ema = np.full((NS, ND), np.nan)
    atr10 = np.full((NS, ND), np.nan)
    vol20 = np.full((NS, ND), np.nan)       # 20-day realized vol
    vol_pct = np.full((NS, ND), np.nan)     # volatility percentile (252d lookback)

    for si in range(NS):
        vdp_e = 0.0
        for di in range(20, ND):
            d = di - 1
            c_now = C[si, d]
            if np.isnan(c_now) or c_now <= 0:
                continue

            # Momentum
            for lag, arr in [(3, mom3), (5, mom5)]:
                c_prev = C[si, max(0, d - lag)]
                if not np.isnan(c_prev) and c_prev > 0:
                    arr[si, di] = (c_now - c_prev) / c_prev

            # OI momentum
            oi_now = OI[si, d]
            if not np.isnan(oi_now) and oi_now > 0:
                oi5 = OI[si, max(0, d - 4)]
                if not np.isnan(oi5) and oi5 > 0:
                    oi_mom5[si, di] = (oi_now - oi5) / oi5

            # VDP EMA (10-day)
            hl = H[si, d] - L[si, d]
            if not np.isnan(hl) and hl > 0:
                cd = C[si, d]; hd = H[si, d]; ld = L[si, d]; vd = V[si, d]
                if not any(np.isnan([cd, hd, ld, vd])):
                    vdp_val = vd * (2 * cd - hd - ld) / hl
                    alpha = 2.0 / 11
                    vdp_e = alpha * vdp_val + (1 - alpha) * vdp_e
                    vdp_ema[si, di] = vdp_e

            # ATR10
            trs = []
            for dd in range(max(1, d - 9), d + 1):
                hi = H[si, dd]; lo = L[si, dd]; pc = C[si, dd - 1]
                if np.isnan(hi) or np.isnan(lo):
                    continue
                tr = hi - lo
                if not np.isnan(pc):
                    tr = max(tr, abs(hi - pc), abs(lo - pc))
                trs.append(tr)
            if trs:
                atr10[si, di] = np.mean(trs)

            # 20-day realized volatility
            rets = []
            for dd in range(max(1, d - 19), d + 1):
                c0 = C[si, dd - 1]; c1 = C[si, dd]
                if not np.isnan(c0) and not np.isnan(c1) and c0 > 0:
                    rets.append((c1 - c0) / c0)
            if len(rets) >= 10:
                vol20[si, di] = np.std(rets, ddof=0) * np.sqrt(252)

    # Volatility percentile (rolling 252-day lookback)
    for si in range(NS):
        for di in range(252, ND):
            window = vol20[si, di - 251:di + 1]
            valid_w = window[~np.isnan(window)]
            if len(valid_w) >= 60:
                current = vol20[si, di]
                if not np.isnan(current):
                    vol_pct[si, di] = np.searchsorted(np.sort(valid_w), current) / len(valid_w)

    # Build composite scores
    score = np.full((NS, ND), np.nan)

    for si in range(NS):
        for di in range(MIN_TRAIN, ND):
            if mode == 'mom5':
                m = mom5[si, di]
                if np.isnan(m):
                    continue
                score[si, di] = np.clip(m * 8, -1, 1)

            elif mode == 'mom3':
                m = mom3[si, di]
                if np.isnan(m):
                    continue
                score[si, di] = np.clip(m * 8, -1, 1)

            elif mode == 'composite':
                m = mom5[si, di]
                if np.isnan(m):
                    continue
                vals = []
                ws = []

                # Base: mom5
                vals.append(np.clip(m * 8, -1, 1))
                ws.append(0.50)

                # OI momentum bonus
                om = oi_mom5[si, di]
                if not np.isnan(om):
                    oi_sc = np.clip(om * 5, -1, 1)
                    # Direction confirmation: OI rising + price up = bullish
                    vals.append(oi_sc)
                    ws.append(0.25)

                # VDP direction bonus
                vd = vdp_ema[si, di]
                if not np.isnan(vd):
                    vdp_sc = np.sign(vd) * min(abs(vd) / 5e6, 1.0)
                    # Interaction: boost when VDP confirms momentum
                    m5v = m * 8
                    if (vdp_sc > 0 and m5v > 0) or (vdp_sc < 0 and m5v < 0):
                        vdp_sc *= 1.3
                    else:
                        vdp_sc *= 0.5
                    vals.append(vdp_sc)
                    ws.append(0.25)

                if not vals:
                    continue
                score[si, di] = sum(v * w for v, w in zip(vals, ws)) / sum(ws)

    return score, atr10, vol_pct


# ============================================================
# KELLY CRITERION TRACKER
# ============================================================

class KellyTracker:
    """Tracks rolling win rate and payoff ratio for Kelly sizing."""

    def __init__(self, lookback=50, kelly_frac=0.75):
        self.lookback = lookback
        self.kelly_frac = kelly_frac
        self.recent_pnls = []  # rolling window of recent trade PnL%

    def record(self, pnl_pct):
        self.recent_pnls.append(pnl_pct)
        if len(self.recent_pnls) > self.lookback:
            self.recent_pnls = self.recent_pnls[-self.lookback:]

    def kelly_fraction(self):
        """Compute Kelly fraction from rolling stats.
        kelly_f = (p * b - q) / b
        where p = WR, q = 1-p, b = avg_win / avg_loss
        Returns fractional Kelly (e.g., 0.75 * kelly_f).
        """
        if len(self.recent_pnls) < 10:
            return 0.20  # Conservative default

        pnls = np.array(self.recent_pnls)
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        n_total = len(pnls)
        n_wins = len(wins)

        p = n_wins / n_total
        q = 1.0 - p

        avg_win = np.mean(wins) if len(wins) > 0 else 0.01
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0.01

        if avg_loss < 1e-8:
            avg_loss = 0.01

        b = avg_win / avg_loss  # payoff ratio

        if b < 1e-8:
            return 0.05  # Very small if no edge detected

        kelly_f = (p * b - q) / b

        # Floor at small positive value, cap at 1.0
        kelly_f = max(0.02, min(1.0, kelly_f))

        return kelly_f * self.kelly_frac


# ============================================================
# BACKTEST ENGINE WITH KELLY SIZING
# ============================================================

def run_backtest(score, atr10, vol_pct, NS, ND, dates, C, O, H, L, V, OI, syms,
                 kelly_frac=0.75, hold_min=3, trail_atr=2.0, stop_loss=0.03,
                 top_n=1, dd_control=True, vol_regime=True,
                 allow_short=True, config_name=''):
    """Run a single backtest configuration with Kelly position sizing.

    Positions: list of dicts, each:
        {'si', 'sym', 'entry', 'entry_di', 'lots', 'dir', 'atr', 'trail_price'}

    For top_n > 1 we hold up to top_n positions simultaneously.
    """
    cash = float(CASH0)
    trades = []
    positions = []  # list of position dicts
    peak_equity = float(CASH0)
    kelly = KellyTracker(lookback=50, kelly_frac=kelly_frac)
    year_equity = {}  # year -> equity at start of year

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year
        if year not in year_equity:
            year_equity[year] = cash

        # Compute current equity (cash + mark-to-market)
        mkt_val = 0.0
        for pos in positions:
            c = C[pos['si'], di]
            if np.isnan(c) or c <= 0:
                c = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val += c * mult * pos['lots']
        equity = cash + mkt_val

        if equity > peak_equity:
            peak_equity = equity

        # ---- DRAWDOWN CIRCUIT BREAKER ----
        dd_mult = 1.0
        if dd_control and peak_equity > 0:
            dd = (peak_equity - equity) / peak_equity
            if dd > 0.25:
                dd_mult = 0.0
            elif dd > 0.15:
                dd_mult = 0.25
            elif dd > 0.05:
                dd_mult = 0.5

        # ---- VOL REGIME DETECTION ----
        vol_mult = 1.0
        if vol_regime:
            # Average vol percentile across all symbols with valid data today
            vp_today = vol_pct[:, di]
            valid_vp = vp_today[~np.isnan(vp_today)]
            if len(valid_vp) > 0:
                avg_vp = np.mean(valid_vp)
                if avg_vp < 0.30:
                    vol_mult = 1.5
                elif avg_vp > 0.70:
                    vol_mult = 0.5

        # ---- MANAGE EXISTING POSITIONS ----
        to_close = []
        for pi, pos in enumerate(positions):
            c = C[pos['si'], di]
            if np.isnan(c) or c <= 0:
                c = pos['entry']

            mult = MULT.get(pos['sym'], DEF_MULT)
            pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
            pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] > 0 else 0
            days_held = di - pos['entry_di']

            exit_reason = None

            # Stop loss
            if pnl_pct / 100 < -stop_loss:
                exit_reason = 'stop'

            # Trailing stop (after minimum hold)
            if exit_reason is None and trail_atr > 0 and days_held >= hold_min:
                atr_v = pos.get('atr', 0)
                if atr_v > 0:
                    trail = pos.get('trail_price', pos['entry'])
                    if pos['dir'] == 1:
                        new_trail = c - trail_atr * atr_v
                        if new_trail > trail:
                            pos['trail_price'] = new_trail
                        if c < trail:
                            exit_reason = 'trail'
                    else:
                        new_trail = c + trail_atr * atr_v
                        if new_trail < trail:
                            pos['trail_price'] = new_trail
                        if c > trail:
                            exit_reason = 'trail'

            # Signal flip exit (after minimum hold)
            if exit_reason is None and days_held >= hold_min:
                cur_sc = score[pos['si'], di]
                if not np.isnan(cur_sc):
                    if pos['dir'] == 1 and cur_sc < -0.15:
                        exit_reason = 'flip'
                    elif pos['dir'] == -1 and cur_sc > 0.15:
                        exit_reason = 'flip'

            # Time-based exit if no other trigger after hold_min + buffer
            if exit_reason is None and days_held >= hold_min + 5:
                exit_reason = 'time'

            if exit_reason:
                to_close.append(pi)
                cash += c * mult * pos['lots'] * (1 - COMM)
                trades.append({
                    'pnl_pct': pnl_pct,
                    'pnl_abs': pnl,
                    'days': days_held,
                    'di': di,
                    'year': year,
                    'sym': pos['sym'],
                    'dir': pos['dir'],
                    'reason': exit_reason,
                })
                kelly.record(pnl_pct)

        # Remove closed positions (reverse order to preserve indices)
        for pi in sorted(to_close, reverse=True):
            positions.pop(pi)

        # ---- ENTRY: TOP-N ROTATION ----
        n_open_slots = top_n - len(positions)
        if n_open_slots > 0 and dd_mult > 0:
            # Score all symbols
            candidates = []
            held_sis = {pos['si'] for pos in positions}

            for si in range(NS):
                if si in held_sis:
                    continue
                sc = score[si, di]
                if np.isnan(sc):
                    continue

                # For rotation: check if this candidate is better than any held position
                # that has been held at least hold_min days
                c_price = C[si, di]
                if np.isnan(c_price) or c_price <= 0:
                    continue

                # Long candidate
                if sc > 0.05:
                    candidates.append((si, sc, 1, c_price))
                # Short candidate
                if allow_short and sc < -0.05:
                    candidates.append((si, -sc, -1, c_price))

            # Sort by score magnitude descending
            candidates.sort(key=lambda x: -x[1])

            # Also consider replacing worst held position if min hold met
            if len(positions) == top_n and top_n > 1:
                # Find worst held position that has met minimum hold
                worst_pos_idx = -1
                worst_sc = 999
                for pi, pos in enumerate(positions):
                    days_held = di - pos['entry_di']
                    if days_held >= hold_min:
                        sc = score[pos['si'], di]
                        if not np.isnan(sc):
                            eff_sc = sc * pos['dir']  # Positive = in-favor
                            if eff_sc < worst_sc:
                                worst_sc = eff_sc
                                worst_pos_idx = pi

                # If best candidate is significantly better than worst held
                if (worst_pos_idx >= 0 and candidates and
                        candidates[0][1] > worst_sc + 0.1):
                    # Close worst position
                    pos = positions[worst_pos_idx]
                    c = C[pos['si'], di]
                    if np.isnan(c) or c <= 0:
                        c = pos['entry']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
                    pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] > 0 else 0
                    days_held = di - pos['entry_di']

                    cash += c * mult * pos['lots'] * (1 - COMM)
                    trades.append({
                        'pnl_pct': pnl_pct,
                        'pnl_abs': pnl,
                        'days': days_held,
                        'di': di,
                        'year': year,
                        'sym': pos['sym'],
                        'dir': pos['dir'],
                        'reason': 'rotate',
                    })
                    kelly.record(pnl_pct)
                    positions.pop(worst_pos_idx)
                    n_open_slots += 1

            # Open new positions
            for si, sc_val, direction, c_price in candidates[:n_open_slots]:
                sym = syms[si]
                mult = MULT.get(sym, DEF_MULT)
                notional = c_price * mult
                if notional <= 0:
                    continue

                # Kelly position sizing
                k_size = kelly.kelly_fraction()
                # Apply drawdown and vol regime multipliers
                effective_size = k_size * dd_mult * vol_mult
                effective_size = max(0.05, min(0.95, effective_size))

                # For top_n positions, divide capital equally
                alloc_per_slot = cash * effective_size / max(1, top_n)
                lots = int(alloc_per_slot / notional)
                if lots <= 0:
                    continue

                cost = notional * lots * (1 + COMM)
                if cost > cash:
                    lots = int(cash / (notional * (1 + COMM)))
                    if lots <= 0:
                        continue
                    cost = notional * lots * (1 + COMM)

                # ATR for trailing stop
                atr_v = atr10[si, di]
                if np.isnan(atr_v):
                    atr_v = 0

                cash -= cost
                if direction == 1:
                    trail_price = c_price - trail_atr * atr_v if atr_v > 0 else c_price * (1 - stop_loss)
                else:
                    trail_price = c_price + trail_atr * atr_v if atr_v > 0 else c_price * (1 + stop_loss)

                positions.append({
                    'si': si,
                    'sym': sym,
                    'entry': c_price,
                    'entry_di': di,
                    'lots': lots,
                    'dir': direction,
                    'atr': atr_v,
                    'trail_price': trail_price,
                })

    # Close remaining positions
    for pos in positions:
        c = C[pos['si'], ND - 1]
        if np.isnan(c) or c <= 0:
            c = pos['entry']
        mult = MULT.get(pos['sym'], DEF_MULT)
        pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
        pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] > 0 else 0
        days_held = ND - 1 - pos['entry_di']
        cash += c * mult * pos['lots'] * (1 - COMM)
        trades.append({
            'pnl_pct': pnl_pct,
            'pnl_abs': pnl,
            'days': days_held,
            'di': ND - 1,
            'year': dates[ND - 1].year,
            'sym': pos['sym'],
            'dir': pos['dir'],
            'reason': 'end',
        })

    if len(trades) < 20:
        return None

    # ---- COMPUTE STATS ----
    equity_curve = float(CASH0)
    peak = float(CASH0)
    max_dd = 0.0
    for t in sorted(trades, key=lambda x: x['di']):
        equity_curve += t['pnl_abs']
        if equity_curve > peak:
            peak = equity_curve
        if peak > 0:
            dd = (peak - equity_curve) / peak * 100
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
    avg_loss = np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_abs'] <= 0]) if nw < len(trades) else 0

    # Per-year breakdown
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
        'name': config_name,
        'ann': round(ann, 1),
        'n': len(trades),
        'wr': round(wr, 1),
        'dd': round(max_dd, 1),
        'avg_pnl': round(avg_pnl, 3),
        'avg_days': round(avg_days, 1),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'final': round(cash, 0),
        'years': year_stats,
        'reasons': reasons,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 100, flush=True)
    print("  Alpha Futures V21 — Kelly Criterion Adaptive Position Sizing", flush=True)
    print("=" * 100, flush=True)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    print("\n  Pre-computing scores and factors...", flush=True)
    t0 = time.time()

    # Pre-compute all three signal modes
    scores = {}
    atr10_cache = None
    vol_pct_cache = None
    for mode in ['mom5', 'mom3', 'composite']:
        print(f"    Computing {mode} scores...", flush=True)
        sc, atr10_cache, vol_pct_cache = compute_scores(NS, ND, C, O, H, L, V, OI, mode=mode)
        scores[mode] = sc
        print(f"    {mode} done ({time.time() - t0:.0f}s)", flush=True)

    print(f"  All factors computed ({time.time() - t0:.0f}s)", flush=True)

    # ============================================================
    # BUILD CONFIGURATION GRID
    # ============================================================
    print("\n  Building configuration grid...", flush=True)

    configs = []

    for base_signal in ['mom5', 'mom3', 'composite']:
        for kelly_fraction in [0.5, 0.75, 1.0]:
            for hold_min in [2, 3]:
                for trail_atr in [2.0, 3.0]:
                    for stop_loss in [0.03, 0.05]:
                        for top_n in [1, 3]:
                            for dd_control in [True, False]:
                                for vol_regime in [True, False]:
                                    cname = (
                                        f"sig={base_signal}_kf={kelly_fraction}"
                                        f"_hm={hold_min}_ta={trail_atr}"
                                        f"_sl={stop_loss}_n={top_n}"
                                        f"_dd={'Y' if dd_control else 'N'}"
                                        f"_vr={'Y' if vol_regime else 'N'}"
                                    )
                                    configs.append({
                                        'name': cname,
                                        'base_signal': base_signal,
                                        'kelly_frac': kelly_fraction,
                                        'hold_min': hold_min,
                                        'trail_atr': trail_atr,
                                        'stop_loss': stop_loss,
                                        'top_n': top_n,
                                        'dd_control': dd_control,
                                        'vol_regime': vol_regime,
                                    })

    print(f"  Total configurations: {len(configs)}", flush=True)

    # ============================================================
    # RUN ALL CONFIGURATIONS
    # ============================================================
    results = []
    for ci, cfg in enumerate(configs):
        if ci % 50 == 0:
            print(f"  Config {ci}/{len(configs)} ({len(results)} profitable)", flush=True)

        sc = scores[cfg['base_signal']]
        r = run_backtest(
            score=sc, atr10=atr10_cache, vol_pct=vol_pct_cache,
            NS=NS, ND=ND, dates=dates, C=C, O=O, H=H, L=L, V=V, OI=OI,
            syms=syms,
            kelly_frac=cfg['kelly_frac'],
            hold_min=cfg['hold_min'],
            trail_atr=cfg['trail_atr'],
            stop_loss=cfg['stop_loss'],
            top_n=cfg['top_n'],
            dd_control=cfg['dd_control'],
            vol_regime=cfg['vol_regime'],
            allow_short=True,
            config_name=cfg['name'],
        )
        if r and r['ann'] > 0:
            results.append(r)

    print(f"\n  Completed ({time.time() - t0:.0f}s, {len(results)} with ann > 0%)", flush=True)

    # ============================================================
    # SORT AND DISPLAY TOP 30
    # ============================================================
    results.sort(key=lambda x: -x['ann'])

    print(f"\n{'=' * 100}", flush=True)
    print(f"  TOP 30 RESULTS", flush=True)
    print(f"  {'Config':68s} | {'Ann':>8s} {'WR':>5s} {'N':>4s} {'DD':>6s} "
          f"{'AvgW':>6s} {'AvgL':>6s} {'AvgD':>5s}", flush=True)
    print(f"  {'-' * 100}", flush=True)
    for r in results[:30]:
        print(f"  {r['name']:68s} | {r['ann']:+8.1f}% {r['wr']:5.1f}% {r['n']:4d} "
              f"{r['dd']:6.1f}% {r['avg_win']:+6.2f}% {r['avg_loss']:6.2f}% "
              f"{r['avg_days']:5.1f}d", flush=True)

    # ============================================================
    # YEARLY BREAKDOWN FOR TOP 5
    # ============================================================
    for i, r in enumerate(results[:5]):
        print(f"\n  #{i + 1}: {r['name']}", flush=True)
        print(f"    Ann={r['ann']:+.1f}% WR={r['wr']:.0f}% DD={r['dd']:.1f}% "
              f"AvgWin={r['avg_win']:+.2f}% AvgLoss={r['avg_loss']:.2f}% "
              f"AvgDays={r['avg_days']:.1f}", flush=True)

        print(f"    Exit reasons:", flush=True)
        for reason, s in sorted(r['reasons'].items(), key=lambda x: -x[1]['n']):
            rwr = s['w'] / max(s['n'], 1) * 100
            print(f"      {reason:10s}: {s['n']:4d}t  WR={rwr:.0f}%  "
                  f"pnl={s['pnl']:+.1f}%", flush=True)

        print(f"    Yearly breakdown:", flush=True)
        for y in sorted(r['years'].keys()):
            s = r['years'][y]
            ywr = s['w'] / max(s['n'], 1) * 100
            print(f"      {y}: {s['n']:3d}t  WR={ywr:.0f}%  "
                  f"pnl={s['pnl']:+.1f}%", flush=True)

    # ============================================================
    # SUMMARY STATISTICS
    # ============================================================
    print(f"\n{'=' * 100}", flush=True)
    if results:
        anns = [r['ann'] for r in results]
        wrs = [r['wr'] for r in results]
        dds = [r['dd'] for r in results]
        print(f"  Summary across {len(results)} profitable configs:", flush=True)
        print(f"    Annual return :  best={max(anns):+.1f}%  median={np.median(anns):+.1f}%  "
              f"mean={np.mean(anns):+.1f}%", flush=True)
        print(f"    Win rate      :  best={max(wrs):.1f}%  median={np.median(wrs):.1f}%  "
              f"mean={np.mean(wrs):.1f}%", flush=True)
        print(f"    Max drawdown  :  best={min(dds):.1f}%  median={np.median(dds):.1f}%  "
              f"mean={np.mean(dds):.1f}%", flush=True)

        # Count by signal type
        for sig in ['mom5', 'mom3', 'composite']:
            sig_res = [r for r in results if f'sig={sig}_' in r['name']]
            if sig_res:
                best_ann = max(r['ann'] for r in sig_res)
                avg_ann = np.mean([r['ann'] for r in sig_res])
                print(f"    {sig:10s}   : {len(sig_res):3d} configs  "
                      f"best={best_ann:+.1f}%  avg={avg_ann:+.1f}%", flush=True)

        # Count by kelly fraction
        for kf in [0.5, 0.75, 1.0]:
            kf_res = [r for r in results if f'_kf={kf}_' in r['name']]
            if kf_res:
                best_ann = max(r['ann'] for r in kf_res)
                avg_ann = np.mean([r['ann'] for r in kf_res])
                print(f"    kelly={kf:.2f}  : {len(kf_res):3d} configs  "
                      f"best={best_ann:+.1f}%  avg={avg_ann:+.1f}%", flush=True)

        # Count by top_n
        for tn in [1, 3]:
            tn_res = [r for r in results if f'_n={tn}_' in r['name']]
            if tn_res:
                best_ann = max(r['ann'] for r in tn_res)
                avg_ann = np.mean([r['ann'] for r in tn_res])
                print(f"    top_n={tn}    : {len(tn_res):3d} configs  "
                      f"best={best_ann:+.1f}%  avg={avg_ann:+.1f}%", flush=True)

    print(f"\n  Total time: {time.time() - t0:.0f}s", flush=True)
    print(f"{'=' * 100}", flush=True)
