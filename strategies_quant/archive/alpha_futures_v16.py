"""
Alpha Futures V16 — Volatility Breakout with OI Confirmation
=============================================================
Core Idea:
  When a commodity has been in LOW VOLATILITY consolidation (narrow ATR for 10+ days)
  AND price breaks above 20-day high (or below 20-day low)
  AND OI is increasing (new money entering)
  Enter with 100% capital (P1 concentrated), no leverage
  Trailing stop at Nx ATR to ride the trend
  Hold 3-10 days

Strategy Variants:
  1. DONCHIAN_BRK   — Simple 20-day high/low breakout
  2. LOWVOL_BRK     — Breakout ONLY after 10-day ATR < 50th pctile (low vol)
  3. LOWVOL_BRK_OI  — Same + require OI 5-day growth > 5%
  4. LOWVOL_BRK_VDP — Same + require VDP EMA confirms direction
  5. LOWVOL_BRK_ALL — All filters combined
  6. RANGE_EXPANSION— Enter when (H-L) > 2x average of last 20 days
  7. OI_SURGE_BRK   — Enter when OI > 2x 20-day avg + price breakout

Parameters swept:
  trail_mult: 1.5, 2.0, 3.0
  stop_loss:  0.03, 0.05
  hold_max:   5, 7, 10

No leverage: lots = cash / (price * multiplier), full notional purchase.
P1 concentrated: all cash in one position.
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

# ============================================================
# Contract multipliers
# ============================================================
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
# Pre-compute all indicators
# ============================================================
def precompute(NS, ND, C, O, H, L, V, OI):
    """Pre-compute all indicators needed by the strategy variants."""
    print("  Pre-computing indicators...", flush=True)
    t0 = time.time()

    atr10      = np.full((NS, ND), np.nan)   # 10-day ATR
    atr10_pctl = np.full((NS, ND), np.nan)   # 10-day ATR percentile rank (60-day lookback)
    hh20       = np.full((NS, ND), np.nan)   # 20-day highest high
    ll20       = np.full((NS, ND), np.nan)   # 20-day lowest low
    range20    = np.full((NS, ND), np.nan)   # today's range H-L
    range20avg = np.full((NS, ND), np.nan)   # 20-day average of (H-L)
    oi_ma20    = np.full((NS, ND), np.nan)   # 20-day OI moving average
    oi_growth5 = np.full((NS, ND), np.nan)   # OI 5-day growth rate
    vdp_ema    = np.full((NS, ND), np.nan)   # VDP EMA (15-day)

    for si in range(NS):
        vdp_e = 0.0
        for di in range(1, ND):
            # Yesterday's close for True Range
            pc = C[si, di - 1]
            hi = H[si, di]
            lo = L[si, di]
            c_now = C[si, di]

            # Skip if critical data missing
            if np.isnan(hi) or np.isnan(lo) or np.isnan(c_now):
                continue

            # --- True Range ---
            tr = hi - lo
            if not np.isnan(pc):
                tr = max(tr, abs(hi - pc), abs(lo - pc))

            # --- 10-day ATR (rolling average of TR) ---
            if di >= 10:
                trs = []
                for dd in range(di - 9, di + 1):
                    h_ = H[si, dd]; l_ = L[si, dd]; p_ = C[si, dd - 1] if dd > 0 else np.nan
                    if np.isnan(h_) or np.isnan(l_): continue
                    t_ = h_ - l_
                    if not np.isnan(p_):
                        t_ = max(t_, abs(h_ - p_), abs(l_ - p_))
                    trs.append(t_)
                if len(trs) >= 5:
                    atr10[si, di] = np.mean(trs)

            # --- ATR percentile rank over 60-day lookback ---
            if di >= 60 and not np.isnan(atr10[si, di]):
                window_atr = atr10[si, di - 59:di + 1]
                valid_atr = window_atr[~np.isnan(window_atr)]
                if len(valid_atr) >= 20:
                    atr10_pctl[si, di] = np.sum(valid_atr <= atr10[si, di]) / len(valid_atr)

            # --- 20-day Donchian channel ---
            if di >= 20:
                h20 = H[si, di - 19:di]  # exclude current day (look at d-1)
                l20 = L[si, di - 19:di]
                h20v = h20[~np.isnan(h20)]
                l20v = l20[~np.isnan(l20)]
                if len(h20v) > 0:
                    hh20[si, di] = np.max(h20v)
                if len(l20v) > 0:
                    ll20[si, di] = np.min(l20v)

            # --- Range and 20-day average range ---
            range20[si, di] = hi - lo
            if di >= 20:
                rngs = []
                for dd in range(di - 19, di + 1):
                    h_ = H[si, dd]; l_ = L[si, dd]
                    if np.isnan(h_) or np.isnan(l_): continue
                    rngs.append(h_ - l_)
                if len(rngs) >= 10:
                    range20avg[si, di] = np.mean(rngs)

            # --- OI indicators ---
            oi_now = OI[si, di]
            if not np.isnan(oi_now) and oi_now > 0:
                # 20-day OI MA
                if di >= 20:
                    oi_window = OI[si, di - 19:di + 1]
                    oi_valid = oi_window[~np.isnan(oi_window)]
                    if len(oi_valid) >= 10:
                        oi_ma20[si, di] = np.mean(oi_valid)
                # 5-day OI growth
                oi_5ago = OI[si, di - 5] if di >= 5 else np.nan
                if not np.isnan(oi_5ago) and oi_5ago > 0:
                    oi_growth5[si, di] = (oi_now - oi_5ago) / oi_5ago

            # --- VDP EMA ---
            vd = V[si, di]
            hl_range = hi - lo
            if not np.isnan(vd) and not np.isnan(hl_range) and hl_range > 0:
                vdp_val = vd * (2 * c_now - hi - lo) / hl_range
                alpha = 2.0 / 15
                vdp_e = alpha * vdp_val + (1 - alpha) * vdp_e
                vdp_ema[si, di] = vdp_e

    print(f"  Indicators done ({time.time() - t0:.1f}s)", flush=True)
    return atr10, atr10_pctl, hh20, ll20, range20, range20avg, oi_ma20, oi_growth5, vdp_ema


# ============================================================
# Signal generators for each variant
# Each returns: signals[si] = list of (di, direction) at close[di-1] info
# Direction: +1 = long, -1 = short
# ============================================================

def signals_donchian_brk(NS, ND, C, H, L, hh20, ll20):
    """Variant 1: Simple 20-day Donchian breakout.
    Buy when close > 20-day high, short when close < 20-day low."""
    name = "DONCHIAN_BRK"
    signals = [[] for _ in range(NS)]
    for si in range(NS):
        for di in range(MIN_TRAIN, ND):
            c = C[si, di - 1]  # signal uses data up to di-1
            if np.isnan(c) or c <= 0:
                continue
            h20 = hh20[si, di]
            l20 = ll20[si, di]
            if np.isnan(h20) or np.isnan(l20):
                continue
            if c > h20:
                signals[si].append((di, 1))
            elif c < l20:
                signals[si].append((di, -1))
    return name, signals


def signals_lowvol_brk(NS, ND, C, H, L, hh20, ll20, atr10_pctl):
    """Variant 2: Breakout only after low volatility (ATR < 50th percentile)."""
    name = "LOWVOL_BRK"
    signals = [[] for _ in range(NS)]
    for si in range(NS):
        for di in range(MIN_TRAIN, ND):
            c = C[si, di - 1]
            if np.isnan(c) or c <= 0:
                continue
            h20 = hh20[si, di]
            l20 = ll20[si, di]
            pctl = atr10_pctl[si, di]
            if np.isnan(h20) or np.isnan(l20) or np.isnan(pctl):
                continue
            if pctl > 0.50:
                continue  # skip if not low volatility
            if c > h20:
                signals[si].append((di, 1))
            elif c < l20:
                signals[si].append((di, -1))
    return name, signals


def signals_lowvol_brk_oi(NS, ND, C, H, L, hh20, ll20, atr10_pctl, oi_growth5):
    """Variant 3: Low-vol breakout + OI 5-day growth > 5%."""
    name = "LOWVOL_BRK_OI"
    signals = [[] for _ in range(NS)]
    for si in range(NS):
        for di in range(MIN_TRAIN, ND):
            c = C[si, di - 1]
            if np.isnan(c) or c <= 0:
                continue
            h20 = hh20[si, di]
            l20 = ll20[si, di]
            pctl = atr10_pctl[si, di]
            oi_g = oi_growth5[si, di]
            if np.isnan(h20) or np.isnan(l20) or np.isnan(pctl):
                continue
            if pctl > 0.50:
                continue
            if np.isnan(oi_g):
                continue
            # For long: OI growing; for short: OI growing (new money entering confirms conviction)
            if c > h20 and oi_g > 0.05:
                signals[si].append((di, 1))
            elif c < l20 and oi_g > 0.05:
                signals[si].append((di, -1))
    return name, signals


def signals_lowvol_brk_vdp(NS, ND, C, H, L, hh20, ll20, atr10_pctl, vdp_ema):
    """Variant 4: Low-vol breakout + VDP EMA confirms direction."""
    name = "LOWVOL_BRK_VDP"
    signals = [[] for _ in range(NS)]
    for si in range(NS):
        for di in range(MIN_TRAIN, ND):
            c = C[si, di - 1]
            if np.isnan(c) or c <= 0:
                continue
            h20 = hh20[si, di]
            l20 = ll20[si, di]
            pctl = atr10_pctl[si, di]
            vdp = vdp_ema[si, di]
            if np.isnan(h20) or np.isnan(l20) or np.isnan(pctl):
                continue
            if pctl > 0.50:
                continue
            if np.isnan(vdp):
                continue
            if c > h20 and vdp > 0:
                signals[si].append((di, 1))
            elif c < l20 and vdp < 0:
                signals[si].append((di, -1))
    return name, signals


def signals_lowvol_brk_all(NS, ND, C, H, L, hh20, ll20, atr10_pctl, oi_growth5, vdp_ema):
    """Variant 5: Low-vol breakout + OI + VDP all confirm."""
    name = "LOWVOL_BRK_ALL"
    signals = [[] for _ in range(NS)]
    for si in range(NS):
        for di in range(MIN_TRAIN, ND):
            c = C[si, di - 1]
            if np.isnan(c) or c <= 0:
                continue
            h20 = hh20[si, di]
            l20 = ll20[si, di]
            pctl = atr10_pctl[si, di]
            oi_g = oi_growth5[si, di]
            vdp = vdp_ema[si, di]
            if np.isnan(h20) or np.isnan(l20) or np.isnan(pctl):
                continue
            if pctl > 0.50:
                continue
            if np.isnan(oi_g) or np.isnan(vdp):
                continue
            if c > h20 and oi_g > 0.05 and vdp > 0:
                signals[si].append((di, 1))
            elif c < l20 and oi_g > 0.05 and vdp < 0:
                signals[si].append((di, -1))
    return name, signals


def signals_range_expansion(NS, ND, C, H, L, range20, range20avg):
    """Variant 6: Range expansion — today's range > 2x the 20-day average range.
    Enter in the direction of the close relative to the day's midpoint."""
    name = "RANGE_EXPANSION"
    signals = [[] for _ in range(NS)]
    for si in range(NS):
        for di in range(MIN_TRAIN, ND):
            # Use yesterday's range for signal
            rng = range20[si, di - 1] if di > 0 else np.nan
            avg = range20avg[si, di - 1] if di > 0 else np.nan
            if np.isnan(rng) or np.isnan(avg) or avg <= 0:
                continue
            if rng > 2.0 * avg:
                # Direction from close position within the day's range
                c_ = C[si, di - 1]
                h_ = H[si, di - 1]
                l_ = L[si, di - 1]
                if np.isnan(c_) or np.isnan(h_) or np.isnan(l_):
                    continue
                mid = (h_ + l_) / 2.0
                if c_ > mid:
                    signals[si].append((di, 1))
                elif c_ < mid:
                    signals[si].append((di, -1))
    return name, signals


def signals_oi_surge_brk(NS, ND, C, H, L, hh20, ll20, OI, oi_ma20):
    """Variant 7: OI surge (> 2x 20-day average) + price breakout."""
    name = "OI_SURGE_BRK"
    signals = [[] for _ in range(NS)]
    for si in range(NS):
        for di in range(MIN_TRAIN, ND):
            c = C[si, di - 1]
            if np.isnan(c) or c <= 0:
                continue
            h20 = hh20[si, di]
            l20 = ll20[si, di]
            oi_now = OI[si, di - 1] if di > 0 else np.nan
            oi_avg = oi_ma20[si, di - 1] if di > 0 else np.nan
            if np.isnan(h20) or np.isnan(l20):
                continue
            if np.isnan(oi_now) or np.isnan(oi_avg) or oi_avg <= 0:
                continue
            oi_ratio = oi_now / oi_avg
            if c > h20 and oi_ratio > 2.0:
                signals[si].append((di, 1))
            elif c < l20 and oi_ratio > 2.0:
                signals[si].append((di, -1))
    return name, signals


# ============================================================
# Backtest engine (P1 concentrated, no leverage)
# ============================================================
def run_backtest(variant_name, signals, NS, ND, dates, C, H, L, OI, atr10,
                 syms, sym_set,
                 trail_mult=3.0, stop_loss=0.05, hold_max=7):
    """Run backtest for one variant + parameter combo.

    Entry: at close[di] using signal from data up to di-1.
    Exit:  at close[di] when stop/trail/time hit.
    No leverage: lots = floor(cash / (price * multiplier)).
    P1: all capital in one position.
    """
    cash = float(CASH0)
    trades = []
    pos = None  # {'si', 'entry', 'entry_di', 'lots', 'dir', 'sym', 'atr', 'trail_price'}

    # Build day-indexed signal lookup for fast access
    sig_by_day = [{} for _ in range(ND)]  # sig_by_day[di][si] = direction
    for si in range(NS):
        for (di, d) in signals[si]:
            if di not in sig_by_day or si not in sig_by_day[di]:
                sig_by_day[di][si] = d

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # === MANAGE EXISTING POSITION ===
        if pos is not None:
            c = C[pos['si'], di]
            if np.isnan(c) or c <= 0:
                # Use previous known price if current is missing
                c = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = c * mult * pos['lots']
            pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
            pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) if pos['entry'] * mult * pos['lots'] > 0 else 0
            days_held = di - pos['entry_di']

            exit_reason = None

            # 1. Fixed stop loss
            if pnl_pct < -stop_loss:
                exit_reason = 'stop'

            # 2. Trailing stop (start after day 1)
            if exit_reason is None and days_held >= 2 and pos.get('atr', 0) > 0:
                atr = pos['atr']
                trail_price = pos.get('trail_price', pos['entry'])
                if pos['dir'] == 1:
                    new_trail = c - trail_mult * atr
                    if new_trail > trail_price:
                        pos['trail_price'] = new_trail
                    if c < pos['trail_price']:
                        exit_reason = 'trail'
                else:
                    new_trail = c + trail_mult * atr
                    if new_trail < trail_price:
                        pos['trail_price'] = new_trail
                    if c > pos['trail_price']:
                        exit_reason = 'trail'

            # 3. Time exit
            if exit_reason is None and days_held >= hold_max:
                exit_reason = 'time'

            if exit_reason:
                cash += mkt_val * (1 - COMM)
                trades.append({
                    'pnl_pct': pnl_pct * 100,
                    'pnl_abs': pnl,
                    'days': days_held,
                    'di': di,
                    'year': year,
                    'sym': pos['sym'],
                    'dir': pos['dir'],
                    'reason': exit_reason,
                })
                pos = None

        # === ENTRY ===
        if pos is None:
            best_si = -1
            best_dir = 0
            best_score = 0  # score = |pnl potential| using momentum as proxy

            for si, direction in sig_by_day[di].items():
                c = C[si, di]
                if np.isnan(c) or c <= 0:
                    continue
                # Score: use closeness to breakout strength as a simple heuristic
                # For simplicity, just pick the first valid signal
                # But prefer commodities with stronger OI growth or higher vol compression
                # Simple: pick the one with lowest ATR percentile (most compressed)
                score = 1.0  # base
                if direction != 0:
                    if best_si < 0:
                        best_si = si
                        best_dir = direction
                        best_score = score

            if best_si >= 0:
                c = C[best_si, di]
                if np.isnan(c) or c <= 0:
                    continue
                sym = syms[best_si]
                mult = MULT.get(sym, DEF_MULT)
                notional_per_lot = c * mult
                if notional_per_lot <= 0:
                    continue

                # P1 concentrated: all cash into one position, no leverage
                lots = int(cash / notional_per_lot)
                if lots <= 0:
                    continue

                cost_in = notional_per_lot * lots * (1 + COMM)
                if cost_in > cash:
                    lots = int(cash / (notional_per_lot * (1 + COMM)))
                    cost_in = notional_per_lot * lots * (1 + COMM)
                if lots <= 0:
                    continue

                # Get ATR for trailing stop
                atr_val = atr10[best_si, di]
                if np.isnan(atr_val) or atr_val <= 0:
                    # Compute on the fly
                    trs = []
                    for dd in range(max(1, di - 10), di + 1):
                        h_ = H[best_si, dd]; l_ = L[best_si, dd]; p_ = C[best_si, dd - 1]
                        if np.isnan(h_) or np.isnan(l_): continue
                        t_ = h_ - l_
                        if not np.isnan(p_):
                            t_ = max(t_, abs(h_ - p_), abs(l_ - p_))
                        trs.append(t_)
                    atr_val = np.mean(trs) if trs else c * 0.02

                cash -= cost_in
                if best_dir == 1:
                    trail_price = c - trail_mult * atr_val
                else:
                    trail_price = c + trail_mult * atr_val

                pos = {
                    'si': best_si,
                    'entry': c,
                    'entry_di': di,
                    'lots': lots,
                    'dir': best_dir,
                    'sym': sym,
                    'atr': atr_val,
                    'trail_price': trail_price,
                }

    # Close remaining position at end
    if pos is not None:
        c = C[pos['si'], ND - 1]
        if np.isnan(c) or c <= 0:
            c = pos['entry']
        mult = MULT.get(pos['sym'], DEF_MULT)
        pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
        cash += c * mult * pos['lots'] * (1 - COMM)
        trades.append({
            'pnl_pct': pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] * mult * pos['lots'] > 0 else 0,
            'pnl_abs': pnl,
            'days': ND - 1 - pos['entry_di'],
            'di': ND - 1,
            'year': dates[ND - 1].year,
            'sym': pos['sym'],
            'dir': pos['dir'],
            'reason': 'end',
        })

    if len(trades) < 10:
        return None

    # === Compute statistics ===
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
    avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_abs'] > 0]) if nw > 0 else 0
    avg_loss = np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_abs'] <= 0]) if nw < len(trades) else 0
    avg_days = np.mean([t['days'] for t in trades])

    year_stats = {}
    for t in trades:
        y = t['year']
        if y not in year_stats:
            year_stats[y] = {'n': 0, 'w': 0, 'pnl': 0.0}
        year_stats[y]['n'] += 1
        if t['pnl_abs'] > 0:
            year_stats[y]['w'] += 1
        year_stats[y]['pnl'] += t['pnl_pct']

    return {
        'name': variant_name,
        'ann': round(ann, 1),
        'n': len(trades),
        'wr': round(wr, 1),
        'dd': round(max_dd, 1),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'avg_days': round(avg_days, 1),
        'final': round(cash, 0),
        'years': year_stats,
        'trail_mult': trail_mult,
        'stop_loss': stop_loss,
        'hold_max': hold_max,
    }


# ============================================================
# Improved entry: rank candidates by OI growth, pick best
# ============================================================
def run_backtest_ranked(variant_name, signals, NS, ND, dates, C, H, L, OI, atr10,
                        syms, sym_set, atr10_pctl, oi_growth5,
                        trail_mult=3.0, stop_loss=0.05, hold_max=7):
    """Same as run_backtest but ranks candidates by ATR compression (lower pctile = better)."""
    cash = float(CASH0)
    trades = []
    pos = None

    # Build day-indexed signal lookup
    sig_by_day = [{} for _ in range(ND)]
    for si in range(NS):
        for (di, d) in signals[si]:
            sig_by_day[di][si] = d

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        # === MANAGE EXISTING POSITION ===
        if pos is not None:
            c = C[pos['si'], di]
            if np.isnan(c) or c <= 0:
                c = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = c * mult * pos['lots']
            pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
            pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) if pos['entry'] * mult * pos['lots'] > 0 else 0
            days_held = di - pos['entry_di']

            exit_reason = None

            if pnl_pct < -stop_loss:
                exit_reason = 'stop'

            if exit_reason is None and days_held >= 2 and pos.get('atr', 0) > 0:
                atr = pos['atr']
                trail_price = pos.get('trail_price', pos['entry'])
                if pos['dir'] == 1:
                    new_trail = c - trail_mult * atr
                    if new_trail > trail_price:
                        pos['trail_price'] = new_trail
                    if c < pos['trail_price']:
                        exit_reason = 'trail'
                else:
                    new_trail = c + trail_mult * atr
                    if new_trail < trail_price:
                        pos['trail_price'] = new_trail
                    if c > pos['trail_price']:
                        exit_reason = 'trail'

            if exit_reason is None and days_held >= hold_max:
                exit_reason = 'time'

            if exit_reason:
                cash += mkt_val * (1 - COMM)
                trades.append({
                    'pnl_pct': pnl_pct * 100,
                    'pnl_abs': pnl,
                    'days': days_held,
                    'di': di,
                    'year': year,
                    'sym': pos['sym'],
                    'dir': pos['dir'],
                    'reason': exit_reason,
                })
                pos = None

        # === ENTRY: rank all candidates, pick best ===
        if pos is None:
            candidates = []
            for si, direction in sig_by_day[di].items():
                c = C[si, di]
                if np.isnan(c) or c <= 0:
                    continue
                sym = syms[si]
                mult = MULT.get(sym, DEF_MULT)
                notional = c * mult
                if notional <= 0:
                    continue
                lots = int(cash / notional)
                if lots <= 0:
                    continue

                # Ranking score: prefer lower ATR percentile (more compressed)
                pctl = atr10_pctl[si, di]
                oi_g = oi_growth5[si, di]
                score = 0
                if not np.isnan(pctl):
                    score -= pctl  # lower pctile = better
                if not np.isnan(oi_g):
                    score += oi_g  # higher OI growth = better
                candidates.append((si, direction, score, lots, notional, sym))

            if candidates:
                # Sort by score descending
                candidates.sort(key=lambda x: -x[2])
                best_si, best_dir, _, best_lots, best_notional, best_sym = candidates[0]

                cost_in = best_notional * best_lots * (1 + COMM)
                if cost_in > cash:
                    best_lots = int(cash / (best_notional * (1 + COMM)))
                    cost_in = best_notional * best_lots * (1 + COMM)
                if best_lots <= 0:
                    continue

                atr_val = atr10[best_si, di]
                if np.isnan(atr_val) or atr_val <= 0:
                    trs = []
                    for dd in range(max(1, di - 10), di + 1):
                        h_ = H[best_si, dd]; l_ = L[best_si, dd - 1]
                        hi_ = H[best_si, dd]; lo_ = L[best_si, dd]
                        if np.isnan(hi_) or np.isnan(lo_): continue
                        t_ = hi_ - lo_
                        if not np.isnan(l_):
                            t_ = max(t_, abs(hi_ - l_), abs(lo_ - l_))
                        trs.append(t_)
                    atr_val = np.mean(trs) if trs else c * 0.02

                cash -= cost_in
                c_entry = C[best_si, di]
                if best_dir == 1:
                    trail_price = c_entry - trail_mult * atr_val
                else:
                    trail_price = c_entry + trail_mult * atr_val

                pos = {
                    'si': best_si,
                    'entry': c_entry,
                    'entry_di': di,
                    'lots': best_lots,
                    'dir': best_dir,
                    'sym': best_sym,
                    'atr': atr_val,
                    'trail_price': trail_price,
                }

    # Close remaining
    if pos is not None:
        c = C[pos['si'], ND - 1]
        if np.isnan(c) or c <= 0:
            c = pos['entry']
        mult = MULT.get(pos['sym'], DEF_MULT)
        pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
        cash += c * mult * pos['lots'] * (1 - COMM)
        trades.append({
            'pnl_pct': pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] * mult * pos['lots'] > 0 else 0,
            'pnl_abs': pnl,
            'days': ND - 1 - pos['entry_di'],
            'di': ND - 1,
            'year': dates[ND - 1].year,
            'sym': pos['sym'],
            'dir': pos['dir'],
            'reason': 'end',
        })

    if len(trades) < 10:
        return None

    # Stats
    equity = float(CASH0); peak = float(CASH0); max_dd = 0
    for t in sorted(trades, key=lambda x: x['di']):
        equity += t['pnl_abs']
        if equity > peak: peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100
            if dd > max_dd: max_dd = dd

    days_total = (dates[ND - 1] - dates[MIN_TRAIN]).days
    yr = max(days_total / 365.25, 0.01)
    ann = ((cash / CASH0) ** (1 / yr) - 1) * 100

    nw = sum(1 for t in trades if t['pnl_abs'] > 0)
    wr = nw / len(trades) * 100
    avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_abs'] > 0]) if nw > 0 else 0
    avg_loss = np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_abs'] <= 0]) if nw < len(trades) else 0
    avg_days = np.mean([t['days'] for t in trades])

    year_stats = {}
    for t in trades:
        y = t['year']
        if y not in year_stats:
            year_stats[y] = {'n': 0, 'w': 0, 'pnl': 0.0}
        year_stats[y]['n'] += 1
        if t['pnl_abs'] > 0: year_stats[y]['w'] += 1
        year_stats[y]['pnl'] += t['pnl_pct']

    return {
        'name': variant_name,
        'ann': round(ann, 1),
        'n': len(trades),
        'wr': round(wr, 1),
        'dd': round(max_dd, 1),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'avg_days': round(avg_days, 1),
        'final': round(cash, 0),
        'years': year_stats,
        'trail_mult': trail_mult,
        'stop_loss': stop_loss,
        'hold_max': hold_max,
    }


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 100, flush=True)
    print("  Alpha Futures V16 — Volatility Breakout with OI Confirmation", flush=True)
    print("=" * 100, flush=True)

    # Load data with OI
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    dm = {d: i for i, d in enumerate(dates)}

    # Pre-compute indicators
    atr10, atr10_pctl, hh20, ll20, range20, range20avg, oi_ma20, oi_growth5, vdp_ema = \
        precompute(NS, ND, C, O, H, L, V, OI)

    # Generate signals for all 7 variants
    print("\n  Computing signals...", flush=True)
    t1 = time.time()

    variants = []
    vname, vsig = signals_donchian_brk(NS, ND, C, H, L, hh20, ll20)
    variants.append((vname, vsig))
    print(f"    {vname} done ({time.time()-t1:.1f}s)", flush=True)

    vname, vsig = signals_lowvol_brk(NS, ND, C, H, L, hh20, ll20, atr10_pctl)
    variants.append((vname, vsig))
    print(f"    {vname} done ({time.time()-t1:.1f}s)", flush=True)

    vname, vsig = signals_lowvol_brk_oi(NS, ND, C, H, L, hh20, ll20, atr10_pctl, oi_growth5)
    variants.append((vname, vsig))
    print(f"    {vname} done ({time.time()-t1:.1f}s)", flush=True)

    vname, vsig = signals_lowvol_brk_vdp(NS, ND, C, H, L, hh20, ll20, atr10_pctl, vdp_ema)
    variants.append((vname, vsig))
    print(f"    {vname} done ({time.time()-t1:.1f}s)", flush=True)

    vname, vsig = signals_lowvol_brk_all(NS, ND, C, H, L, hh20, ll20, atr10_pctl, oi_growth5, vdp_ema)
    variants.append((vname, vsig))
    print(f"    {vname} done ({time.time()-t1:.1f}s)", flush=True)

    vname, vsig = signals_range_expansion(NS, ND, C, H, L, range20, range20avg)
    variants.append((vname, vsig))
    print(f"    {vname} done ({time.time()-t1:.1f}s)", flush=True)

    vname, vsig = signals_oi_surge_brk(NS, ND, C, H, L, hh20, ll20, OI, oi_ma20)
    variants.append((vname, vsig))
    print(f"    {vname} done ({time.time()-t1:.1f}s)", flush=True)

    # Count signals per variant
    for vn, vs in variants:
        total = sum(len(s) for s in vs)
        print(f"    {vn}: {total} signals", flush=True)

    # Sweep parameters
    trail_mults = [1.5, 2.0, 3.0]
    stop_losses = [0.03, 0.05]
    hold_maxes  = [5, 7, 10]

    print(f"\n  Running backtests ({len(variants)} variants x "
          f"{len(trail_mults)} trails x {len(stop_losses)} stops x {len(hold_maxes)} holds = "
          f"{len(variants) * len(trail_mults) * len(stop_losses) * len(hold_maxes)} configs)...", flush=True)

    results = []
    total_configs = len(variants) * len(trail_mults) * len(stop_losses) * len(hold_maxes)
    ci = 0
    for vname, vsig in variants:
        for tm in trail_mults:
            for sl in stop_losses:
                for hm in hold_maxes:
                    ci += 1
                    config_name = f"{vname}_T{tm}_S{sl}_H{hm}"
                    r = run_backtest_ranked(
                        config_name, vsig, NS, ND, dates, C, H, L, OI, atr10,
                        syms, sym_set, atr10_pctl, oi_growth5,
                        trail_mult=tm, stop_loss=sl, hold_max=hm,
                    )
                    if r is not None:
                        results.append(r)
        if ci % 10 == 0:
            print(f"    {ci}/{total_configs} configs done ({len(results)} profitable)", flush=True)

    print(f"\n  Backtesting done ({time.time()-t1:.1f}s, {len(results)} configs with >=10 trades)", flush=True)

    # Sort by annual return
    results.sort(key=lambda x: -x['ann'])

    # === Print TOP 20 ===
    print(f"\n{'='*100}", flush=True)
    print(f"  TOP 20 RESULTS (sorted by annual return)", flush=True)
    print(f"{'='*100}", flush=True)
    hdr = f"  {'#':>3s}  {'Strategy':40s} | {'Ann%':>8s} {'WR%':>6s} {'N':>5s} {'DD%':>6s} {'AvgW%':>7s} {'AvgL%':>7s} {'AvgD':>5s}"
    print(hdr, flush=True)
    print(f"  {'-'*96}", flush=True)
    for i, r in enumerate(results[:20]):
        print(f"  {i+1:3d}  {r['name']:40s} | {r['ann']:+8.1f}% {r['wr']:6.1f}% {r['n']:5d} "
              f"{r['dd']:6.1f}% {r['avg_win']:+7.2f}% {r['avg_loss']:7.2f}% {r['avg_days']:5.1f}d",
              flush=True)

    # === Yearly breakdown for TOP 3 ===
    print(f"\n{'='*100}", flush=True)
    print(f"  YEARLY BREAKDOWN — TOP 3", flush=True)
    print(f"{'='*100}", flush=True)
    for i, r in enumerate(results[:3]):
        print(f"\n  #{i+1}: {r['name']}", flush=True)
        print(f"       Ann={r['ann']:+.1f}%  WR={r['wr']:.1f}%  N={r['n']}  DD={r['dd']:.1f}%  "
              f"AvgWin={r['avg_win']:+.2f}%  AvgLoss={r['avg_loss']:.2f}%  AvgDays={r['avg_days']:.1f}", flush=True)
        print(f"       Trail={r['trail_mult']}x ATR  Stop={r['stop_loss']*100:.0f}%  HoldMax={r['hold_max']}d", flush=True)
        print(f"       {'Year':>6s}  {'N':>4s}  {'WR%':>6s}  {'PnL%':>8s}", flush=True)
        print(f"       {'-'*30}", flush=True)
        for y in sorted(r['years'].keys()):
            s = r['years'][y]
            wr_y = s['w'] / max(s['n'], 1) * 100
            print(f"       {y:6d}  {s['n']:4d}  {wr_y:6.1f}%  {s['pnl']:+8.1f}%", flush=True)

    # === Summary by variant (best of each) ===
    print(f"\n{'='*100}", flush=True)
    print(f"  BEST PER VARIANT", flush=True)
    print(f"{'='*100}", flush=True)
    best_per = {}
    for r in results:
        # Extract base variant name (before _T)
        base = r['name'].rsplit('_T', 1)[0]
        if base not in best_per or r['ann'] > best_per[base]['ann']:
            best_per[base] = r

    print(f"  {'Variant':40s} | {'Ann%':>8s} {'WR%':>6s} {'N':>5s} {'DD%':>6s} {'AvgW%':>7s} {'AvgL%':>7s} {'AvgD':>5s} | {'Config':20s}", flush=True)
    print(f"  {'-'*110}", flush=True)
    for base in ['DONCHIAN_BRK', 'LOWVOL_BRK', 'LOWVOL_BRK_OI', 'LOWVOL_BRK_VDP',
                 'LOWVOL_BRK_ALL', 'RANGE_EXPANSION', 'OI_SURGE_BRK']:
        if base in best_per:
            r = best_per[base]
            cfg = f"T{r['trail_mult']} S{r['stop_loss']} H{r['hold_max']}"
            print(f"  {base:40s} | {r['ann']:+8.1f}% {r['wr']:6.1f}% {r['n']:5d} "
                  f"{r['dd']:6.1f}% {r['avg_win']:+7.2f}% {r['avg_loss']:7.2f}% {r['avg_days']:5.1f}d | {cfg:20s}",
                  flush=True)
        else:
            print(f"  {base:40s} |   (no trades)", flush=True)

    print(f"\n{'='*100}", flush=True)
    above_0  = sum(1 for r in results if r['ann'] > 0)
    above_50 = sum(1 for r in results if r['ann'] > 50)
    above_100 = sum(1 for r in results if r['ann'] > 100)
    above_200 = sum(1 for r in results if r['ann'] > 200)
    print(f"  Summary: {len(results)} configs tested, {above_0} profitable (>0%), "
          f"{above_50} >50%, {above_100} >100%, {above_200} >200%", flush=True)
    if results:
        print(f"  Best: {results[0]['name']} → {results[0]['ann']:+.1f}% annual", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"\n  Total time: {time.time()-t1:.1f}s", flush=True)
