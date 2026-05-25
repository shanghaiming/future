"""
Alpha Futures V22 — Regime-Adaptive Strategy
=============================================
Detect market regime per-commodity using rolling 20-day window:
  - TRENDING:    directional movement |C_5d_ago - C| / (5 * ATR) > threshold
  - MEAN-REVERT: autocorrelation of 5-day returns < threshold
  - QUIET:       ATR / Close < low vol threshold
  - VOLATILE:    ATR / Close > high vol threshold

Then apply the optimal strategy per regime:
  - TRENDING:    momentum (top-5d momentum, wide trail, hold 5-7d)
  - MEAN-REVERT: reversal (biggest 3d loser, tight stop, hold 2-3d)
  - QUIET:       breakout (20-day high breakout, hold 5-10d)
  - VOLATILE:    reduce size (50% position, tight stops)

Test variants:
  1. REGIME_ADAPTIVE  — full regime switching
  2. MOMENTUM_ONLY    — only trade in trending regime
  3. REVERSAL_ONLY    — only trade in mean-reverting regime
  4. TREND_FILTER     — use momentum but only in trending regime
  5. CONFLUENCE_REGIME— require regime + signal alignment

Parameter sweep over thresholds, trail, stop, hold.
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

# Regime labels
REGIME_TRENDING = 0
REGIME_MEAN_REVERT = 1
REGIME_QUIET = 2
REGIME_VOLATILE = 3
REGIME_NEUTRAL = 4  # default when no regime detected


def compute_regimes(NS, ND, C, H, L, adx_thresh=0.3, autocorr_thresh=-0.2,
                    vol_quiet=0.01, vol_volatile=0.03, regime_window=20):
    """Compute per-commodity regime for each day.

    Returns regime[si, di] with values:
      0=TRENDING, 1=MEAN_REVERT, 2=QUIET, 3=VOLATILE, 4=NEUTRAL

    Also returns atr20[si, di] and atr_pct[si, di] for downstream use.
    """
    t0 = time.time()
    regime = np.full((NS, ND), REGIME_NEUTRAL, dtype=np.int8)
    atr20 = np.full((NS, ND), np.nan)
    atr_pct = np.full((NS, ND), np.nan)

    for si in range(NS):
        # Precompute ATR(20)
        tr_arr = np.full(ND, np.nan)
        for di in range(1, ND):
            hi = H[si, di]; lo = L[si, di]; pc = C[si, di - 1]
            if np.isnan(hi) or np.isnan(lo):
                continue
            tr = hi - lo
            if not np.isnan(pc):
                tr = max(tr, abs(hi - pc), abs(lo - pc))
            tr_arr[di] = tr

        # Rolling ATR(20)
        for di in range(regime_window, ND):
            window = tr_arr[di - regime_window + 1:di + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= regime_window // 2:
                atr20[si, di] = np.mean(valid)

        # ATR % of close
        for di in range(regime_window, ND):
            if not np.isnan(atr20[si, di]) and not np.isnan(C[si, di]) and C[si, di] > 0:
                atr_pct[si, di] = atr20[si, di] / C[si, di]

        # Regime detection for each day
        for di in range(regime_window + 5, ND):
            d = di  # use current day (no lookahead since regime is based on past)

            # 1. TRENDING: directional movement over 5 days
            c_now = C[si, d]
            c_5ago = C[si, d - 5] if d >= 5 else np.nan
            atr_val = atr20[si, d]
            if (not np.isnan(c_now) and not np.isnan(c_5ago) and c_5ago > 0
                    and not np.isnan(atr_val) and atr_val > 0):
                dir_move = abs(c_now - c_5ago) / (5 * atr_val)
                if dir_move > adx_thresh:
                    regime[si, d] = REGIME_TRENDING
                    continue

            # 2. VOLATILE: ATR/Close > threshold
            ap = atr_pct[si, d]
            if not np.isnan(ap) and ap > vol_volatile:
                regime[si, d] = REGIME_VOLATILE
                continue

            # 3. MEAN-REVERTING: autocorrelation of 5-day returns
            rets = []
            for j in range(regime_window):
                idx = d - 5 * (j + 1)
                idx_prev = d - 5 * j
                if idx < 0 or idx_prev < 0:
                    break
                c0 = C[si, idx]
                c1 = C[si, idx_prev]
                if not np.isnan(c0) and not np.isnan(c1) and c0 > 0:
                    rets.append((c1 - c0) / c0)
            if len(rets) >= 10:
                rets_arr = np.array(rets)
                mean_r = np.mean(rets_arr)
                var_r = np.var(rets_arr)
                if var_r > 1e-12:
                    autocorr = np.mean((rets_arr[:-1] - mean_r) * (rets_arr[1:] - mean_r)) / var_r
                    if autocorr < autocorr_thresh:
                        regime[si, d] = REGIME_MEAN_REVERT
                        continue

            # 4. QUIET: ATR/Close < threshold (and not already classified)
            if not np.isnan(ap) and ap < vol_quiet:
                regime[si, d] = REGIME_QUIET
                continue

    print(f"  Regime detection done ({time.time()-t0:.1f}s)", flush=True)

    # Print regime distribution
    counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    total = 0
    for si in range(NS):
        for di in range(MIN_TRAIN, ND):
            r = regime[si, di]
            if not np.isnan(C[si, di]) and C[si, di] > 0:
                counts[int(r)] += 1
                total += 1
    names = {0: 'TREND', 1: 'MREVERT', 2: 'QUIET', 3: 'VOLATILE', 4: 'NEUTRAL'}
    if total > 0:
        for k in sorted(counts.keys()):
            print(f"    {names[k]:8s}: {counts[k]:6d} ({counts[k]/total*100:.1f}%)", flush=True)

    return regime, atr20, atr_pct


def compute_signals(NS, ND, C, O, H, L, V):
    """Precompute momentum and breakout signals for all commodities."""
    t0 = time.time()

    mom3 = np.full((NS, ND), np.nan)
    mom5 = np.full((NS, ND), np.nan)
    high20 = np.full((NS, ND), np.nan)
    low20 = np.full((NS, ND), np.nan)

    for si in range(NS):
        for di in range(20, ND):
            c_now = C[si, di]
            if np.isnan(c_now) or c_now <= 0:
                continue

            # Momentum
            for lag, arr in [(3, mom3), (5, mom5)]:
                c_prev = C[si, di - lag]
                if not np.isnan(c_prev) and c_prev > 0:
                    arr[si, di] = (c_now - c_prev) / c_prev

            # 20-day high/low (exclude current day for breakout)
            h20 = H[si, max(0, di - 20):di]
            l20 = L[si, max(0, di - 20):di]
            h20v = h20[~np.isnan(h20)]
            l20v = l20[~np.isnan(l20)]
            if len(h20v) >= 10:
                high20[si, di] = np.max(h20v)
            if len(l20v) >= 10:
                low20[si, di] = np.min(l20v)

    print(f"  Signals computed ({time.time()-t0:.1f}s)", flush=True)
    return mom3, mom5, high20, low20


def run_backtest(variant, score_fn, name, NS, ND, dates, C, O, H, L, V,
                 syms, regime, atr20, mom3, mom5, high20, low20,
                 hold_max=5, trail_atr=2.0, stop_loss=0.05, allow_short=True):
    """Run regime-adaptive backtest.

    Variants:
      'REGIME_ADAPTIVE'  — full regime switching
      'MOMENTUM_ONLY'    — only trade in trending regime
      'REVERSAL_ONLY'    — only trade in mean-reverting regime
      'TREND_FILTER'     — momentum signals but only in trending regime
      'CONFLUENCE_REGIME'— require regime + signal alignment
    """
    cash = float(CASH0)
    trades = []
    pos = None  # {'si', 'entry', 'entry_di', 'lots', 'dir', 'sym', 'atr', 'trail_price'}
    peak_equity = float(CASH0)

    for di in range(MIN_TRAIN, ND):
        year = dates[di].year

        if cash > peak_equity:
            peak_equity = cash

        # === MANAGE POSITION ===
        if pos is not None:
            c = C[pos['si'], di]
            if np.isnan(c) or c <= 0:
                c = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = c * mult * pos['lots']
            pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
            pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] > 0 else 0
            days_held = di - pos['entry_di']

            exit_reason = None

            # Fixed stop loss
            if pnl_pct / 100 < -stop_loss:
                exit_reason = 'stop'

            # Trailing stop
            if exit_reason is None and trail_atr > 0 and days_held >= 2:
                atr_val = pos.get('atr', 0)
                trail_price = pos.get('trail_price', pos['entry'])
                if atr_val > 0:
                    if pos['dir'] == 1:
                        new_trail = c - trail_atr * atr_val
                        if new_trail > trail_price:
                            pos['trail_price'] = new_trail
                        if c < trail_price:
                            exit_reason = 'trail'
                    else:
                        new_trail = c + trail_atr * atr_val
                        if new_trail < trail_price:
                            pos['trail_price'] = new_trail
                        if c > trail_price:
                            exit_reason = 'trail'

            # Signal flip exit
            if exit_reason is None and days_held >= 2:
                cur_score = score_fn(pos['si'], di)
                if not np.isnan(cur_score):
                    if pos['dir'] == 1 and cur_score < -0.15:
                        exit_reason = 'signal_flip'
                    elif pos['dir'] == -1 and cur_score > 0.15:
                        exit_reason = 'signal_flip'

            # Time exit
            if exit_reason is None and days_held >= hold_max:
                exit_reason = 'time'

            if exit_reason:
                cash += mkt_val * (1 - COMM)
                trades.append({
                    'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                    'days': days_held, 'di': di, 'year': year,
                    'sym': pos['sym'], 'dir': pos['dir'],
                    'reason': exit_reason,
                    'regime_at_entry': pos.get('regime_at_entry', REGIME_NEUTRAL),
                })
                pos = None

        # === ENTRY ===
        if pos is None:
            best_si, best_dir, best_sc = -1, 0, 0.0
            best_regime = REGIME_NEUTRAL

            for si in range(NS):
                c = C[si, di]
                if np.isnan(c) or c <= 0:
                    continue

                r = int(regime[si, di])
                sc = score_fn(si, di)
                if np.isnan(sc):
                    continue

                # Variant-specific regime filter
                if variant == 'MOMENTUM_ONLY':
                    if r != REGIME_TRENDING:
                        continue
                elif variant == 'REVERSAL_ONLY':
                    if r != REGIME_MEAN_REVERT:
                        continue
                elif variant == 'TREND_FILTER':
                    if r != REGIME_TRENDING:
                        continue
                elif variant == 'CONFLUENCE_REGIME':
                    # Require regime and signal to agree
                    if r == REGIME_TRENDING and sc <= 0:
                        continue
                    elif r == REGIME_MEAN_REVERT and sc >= 0:
                        continue
                    elif r == REGIME_QUIET:
                        pass  # allow breakout signals
                    elif r not in (REGIME_TRENDING, REGIME_MEAN_REVERT, REGIME_QUIET):
                        continue

                # For REGIME_ADAPTIVE: pick best absolute score regardless of regime
                # but track which regime it came from
                if sc > best_sc:
                    best_sc = sc
                    best_si = si
                    best_dir = 1
                    best_regime = r
                if allow_short and -sc > best_sc:
                    best_sc = -sc
                    best_si = si
                    best_dir = -1
                    best_regime = r

            if best_si >= 0 and best_sc > 0.05:
                c = C[best_si, di]
                if np.isnan(c) or c <= 0:
                    continue

                sym = syms[best_si]
                mult = MULT.get(sym, DEF_MULT)
                notional = c * mult
                if notional <= 0:
                    continue

                # Position sizing: reduce in volatile regime
                alloc = cash * 0.90  # use 90% of cash
                if best_regime == REGIME_VOLATILE:
                    alloc = cash * 0.45  # 50% size in volatile

                lots = int(alloc / notional)
                if lots <= 0:
                    continue

                cost_in = notional * lots * (1 + COMM)
                if cost_in > cash:
                    continue

                # ATR for trailing stop
                atr_val = atr20[best_si, di] if not np.isnan(atr20[best_si, di]) else 0
                if atr_val <= 0:
                    # Fallback: compute simple ATR
                    trs = []
                    for dd in range(max(1, di - 10), di + 1):
                        hi = H[best_si, dd]
                        lo = L[best_si, dd]
                        pc = C[best_si, dd - 1]
                        if np.isnan(hi) or np.isnan(lo):
                            continue
                        tr = hi - lo
                        if not np.isnan(pc):
                            tr = max(tr, abs(hi - pc), abs(lo - pc))
                        trs.append(tr)
                    if trs:
                        atr_val = np.mean(trs)

                cash -= cost_in
                trail_price = c - trail_atr * atr_val if best_dir == 1 else c + trail_atr * atr_val
                pos = {
                    'si': best_si, 'entry': c, 'entry_di': di,
                    'lots': lots, 'dir': best_dir, 'sym': sym,
                    'atr': atr_val, 'trail_price': trail_price,
                    'regime_at_entry': best_regime,
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
            'pnl_pct': pnl / (pos['entry'] * mult * pos['lots']) * 100,
            'pnl_abs': pnl, 'days': ND - 1 - pos['entry_di'],
            'di': ND - 1, 'year': dates[ND - 1].year,
            'sym': pos['sym'], 'dir': pos['dir'], 'reason': 'end',
            'regime_at_entry': pos.get('regime_at_entry', REGIME_NEUTRAL),
        })

    if len(trades) < 20:
        return None

    # Stats
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
    avg_loss = np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_abs'] <= 0]) if nw < len(trades) else 0

    year_stats = {}
    for t in trades:
        y = t['year']
        if y not in year_stats:
            year_stats[y] = {'n': 0, 'w': 0, 'pnl': 0.0}
        year_stats[y]['n'] += 1
        if t['pnl_abs'] > 0:
            year_stats[y]['w'] += 1
        year_stats[y]['pnl'] += t['pnl_pct']

    # Per-regime stats
    regime_stats = {}
    regime_names = {0: 'TREND', 1: 'MREVERT', 2: 'QUIET', 3: 'VOLATILE', 4: 'NEUTRAL'}
    for t in trades:
        r = regime_names.get(int(t.get('regime_at_entry', 4)), 'NEUTRAL')
        if r not in regime_stats:
            regime_stats[r] = {'n': 0, 'w': 0, 'pnl': 0.0}
        regime_stats[r]['n'] += 1
        if t['pnl_abs'] > 0:
            regime_stats[r]['w'] += 1
        regime_stats[r]['pnl'] += t['pnl_pct']

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
        'final': round(cash, 0), 'years': year_stats,
        'reasons': reasons, 'regime_stats': regime_stats,
    }


if __name__ == '__main__':
    print("=" * 95, flush=True)
    print("  Alpha Futures V22 — Regime-Adaptive Strategy", flush=True)
    print("  Trending→Momentum | MeanRevert→Reversal | Quiet→Breakout | Volatile→Reduce Size", flush=True)
    print("=" * 95, flush=True)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    # ------------------------------------------------------------------
    # Regime detection (with default thresholds)
    # ------------------------------------------------------------------
    print("\n  Computing regimes (default thresholds)...", flush=True)
    regime, atr20, atr_pct = compute_regimes(NS, ND, C, H, L)

    # ------------------------------------------------------------------
    # Precompute signals
    # ------------------------------------------------------------------
    print("\n  Computing signals...", flush=True)
    mom3, mom5, high20, low20 = compute_signals(NS, ND, C, O, H, L, V)

    # ------------------------------------------------------------------
    # Scoring functions per regime
    # ------------------------------------------------------------------
    def make_score_momentum(NS, ND, C, O, H, L, V, mom3, mom5, atr20, high20, low20,
                            w_mom5=0.5, w_mom3=0.3, w_breakout=0.2):
        """Score for TRENDING regime: follow momentum."""
        def score(si, di):
            m5 = mom5[si, di]
            if np.isnan(m5):
                return np.nan
            vals = []
            ws = []
            # 5-day momentum (primary)
            vals.append(np.clip(m5 * 8, -1, 1))
            ws.append(w_mom5)
            # 3-day momentum (secondary)
            m3 = mom3[si, di]
            if not np.isnan(m3):
                vals.append(np.clip(m3 * 8, -1, 1))
                ws.append(w_mom3)
            # Breakout bonus: near 20-day high
            h20 = high20[si, di]
            c = C[si, di]
            if not np.isnan(h20) and not np.isnan(c) and h20 > 0:
                prox = (c - h20) / h20 if h20 > 0 else 0
                if prox > 0:
                    vals.append(min(prox * 10, 1.0))
                    ws.append(w_breakout)
            if not vals:
                return np.nan
            return sum(v * w for v, w in zip(vals, ws)) / sum(ws)
        return score

    def make_score_reversal(NS, ND, C, O, H, L, V, mom3, mom5, atr20, high20, low20,
                            w_mom3=0.6, w_atr_pct=0.2, w_vol=0.2):
        """Score for MEAN-REVERTING regime: fade 3-day losers (negative score = bearish reversal)."""
        def score(si, di):
            m3 = mom3[si, di]
            if np.isnan(m3):
                return np.nan
            vals = []
            ws = []
            # Reversal: buy biggest 3-day losers, sell biggest 3-day winners
            vals.append(-np.clip(m3 * 8, -1, 1))  # invert momentum
            ws.append(w_mom3)
            # ATR%: prefer higher vol for mean-revert (more reversion potential)
            ap = atr_pct[si, di] if not np.isnan(atr_pct[si, di]) else 0
            vals.append(np.clip(ap * 30, 0, 1))
            ws.append(w_atr_pct)
            # Volume confirmation: higher vol = more conviction
            v_now = V[si, di]
            if not np.isnan(v_now) and v_now > 0:
                v20 = V[si, max(0, di - 19):di + 1]
                v20v = v20[~np.isnan(v20)]
                if len(v20v) >= 10:
                    vr = v_now / np.mean(v20v)
                    vals.append(np.clip((vr - 1) * 0.5, -0.5, 0.5))
                    ws.append(w_vol)
            if not vals:
                return np.nan
            return sum(v * w for v, w in zip(vals, ws)) / sum(ws)
        return score

    def make_score_breakout(NS, ND, C, O, H, L, V, mom3, mom5, atr20, high20, low20,
                            w_breakout=0.5, w_vol=0.3, w_mom5=0.2):
        """Score for QUIET regime: breakout from 20-day range."""
        def score(si, di):
            c = C[si, di]
            h20 = high20[si, di]
            l20 = low20[si, di]
            if np.isnan(c) or np.isnan(h20) or np.isnan(l20):
                return np.nan
            vals = []
            ws = []
            # Breakout: price above 20-day high
            if h20 > 0:
                prox = (c - h20) / h20
                vals.append(np.clip(prox * 10, -1, 1))
                ws.append(w_breakout)
            # Volume surge on breakout
            v_now = V[si, di]
            if not np.isnan(v_now) and v_now > 0:
                v20 = V[si, max(0, di - 19):di + 1]
                v20v = v20[~np.isnan(v20)]
                if len(v20v) >= 10:
                    vr = v_now / np.mean(v20v)
                    vals.append(np.clip((vr - 1), -1, 1))
                    ws.append(w_vol)
            # Momentum alignment
            m5 = mom5[si, di]
            if not np.isnan(m5):
                vals.append(np.clip(m5 * 5, -1, 1))
                ws.append(w_mom5)
            if not vals:
                return np.nan
            return sum(v * w for v, w in zip(vals, ws)) / sum(ws)
        return score

    def make_regime_adaptive_score(NS, ND, C, O, H, L, V, mom3, mom5, atr20, atr_pct,
                                   high20, low20, regime,
                                   score_mom, score_rev, score_brk):
        """Combined regime-adaptive score: pick the right sub-strategy based on regime."""
        def score(si, di):
            r = int(regime[si, di])
            if r == REGIME_TRENDING:
                return score_mom(si, di)
            elif r == REGIME_MEAN_REVERT:
                return score_rev(si, di)
            elif r == REGIME_QUIET:
                return score_brk(si, di)
            elif r == REGIME_VOLATILE:
                # In volatile regime, use momentum but with reduced conviction
                s = score_mom(si, di)
                if not np.isnan(s):
                    return s * 0.5  # reduced conviction
                return np.nan
            else:
                # NEUTRAL: use momentum as default
                return score_mom(si, di)
        return score

    # ------------------------------------------------------------------
    # Parameter sweep
    # ------------------------------------------------------------------
    print("\n  Building parameter sweep...", flush=True)
    t_sweep = time.time()

    adx_thresholds = [0.2, 0.3, 0.4]
    autocorr_thresholds = [-0.1, -0.2, -0.3]
    vol_thresholds = [(0.01, 0.03), (0.02, 0.03)]
    trail_atrs = [2.0, 3.0]
    stop_losses = [0.03, 0.05]
    hold_maxes = [5, 7]

    variants = ['REGIME_ADAPTIVE', 'MOMENTUM_ONLY', 'REVERSAL_ONLY',
                'TREND_FILTER', 'CONFLUENCE_REGIME']

    results = []
    config_count = 0

    for adx_t in adx_thresholds:
        for ac_t in autocorr_thresholds:
            for vol_quiet, vol_vol in vol_thresholds:
                # Recompute regime for this parameter set
                regime_cfg, _, _ = compute_regimes(NS, ND, C, H, L,
                                                    adx_thresh=adx_t,
                                                    autocorr_thresh=ac_t,
                                                    vol_quiet=vol_quiet,
                                                    vol_volatile=vol_vol)

                # Build scoring functions
                score_mom = make_score_momentum(NS, ND, C, O, H, L, V,
                                                 mom3, mom5, atr20, high20, low20)
                score_rev = make_score_reversal(NS, ND, C, O, H, L, V,
                                                 mom3, mom5, atr20, high20, low20)
                score_brk = make_score_breakout(NS, ND, C, O, H, L, V,
                                                 mom3, mom5, atr20, high20, low20)
                score_adaptive = make_regime_adaptive_score(
                    NS, ND, C, O, H, L, V, mom3, mom5, atr20, atr_pct,
                    high20, low20, regime_cfg,
                    score_mom, score_rev, score_brk)

                for variant in variants:
                    if variant in ('REGIME_ADAPTIVE', 'CONFLUENCE_REGIME'):
                        score_fn = score_adaptive
                    elif variant == 'MOMENTUM_ONLY':
                        score_fn = score_mom
                    elif variant == 'REVERSAL_ONLY':
                        score_fn = score_rev
                    elif variant == 'TREND_FILTER':
                        score_fn = score_mom

                    for trail in trail_atrs:
                        for sl in stop_losses:
                            for hold in hold_maxes:
                                name = (f"{variant[:6]}_A{adx_t}_AC{ac_t}"
                                        f"_VQ{vol_quiet}_VV{vol_vol}"
                                        f"_T{trail}_S{sl}_H{hold}")
                                r = run_backtest(
                                    variant, score_fn, name,
                                    NS, ND, dates, C, O, H, L, V,
                                    syms, regime_cfg, atr20,
                                    mom3, mom5, high20, low20,
                                    hold_max=hold, trail_atr=trail,
                                    stop_loss=sl, allow_short=True)
                                if r and r['ann'] > 0:
                                    results.append(r)
                                config_count += 1

                print(f"  adx={adx_t} ac={ac_t} vq={vol_quiet} vv={vol_vol} "
                      f"({config_count} configs, {len(results)} >0%, "
                      f"{time.time()-t_sweep:.0f}s)", flush=True)

    print(f"\n  Total configs tested: {config_count}", flush=True)
    print(f"  Profitable results: {len(results)}", flush=True)
    print(f"  Sweep time: {time.time()-t_sweep:.0f}s", flush=True)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    results.sort(key=lambda x: -x['ann'])

    print(f"\n{'='*110}", flush=True)
    print(f"  TOP 30 RESULTS (Regime-Adaptive Strategy)", flush=True)
    print(f"  {'Name':<55s} | {'Ann':>8s} {'WR':>5s} {'N':>4s} {'DD':>6s} "
          f"{'AvgW':>6s} {'AvgL':>6s} {'AvgD':>5s}", flush=True)
    print(f"  {'-'*100}", flush=True)
    for r in results[:30]:
        print(f"  {r['name']:<55s} | {r['ann']:+8.1f}% {r['wr']:5.1f}% {r['n']:4d} "
              f"{r['dd']:6.1f}% {r['avg_win']:+6.2f}% {r['avg_loss']:6.2f}% "
              f"{r['avg_days']:5.1f}d", flush=True)

    # ------------------------------------------------------------------
    # Top 5 year-by-year breakdown
    # ------------------------------------------------------------------
    for i, r in enumerate(results[:5]):
        print(f"\n  #{i+1}: {r['name']}", flush=True)
        print(f"    Ann={r['ann']:+.1f}%  WR={r['wr']:.0f}%  DD={r['dd']:.1f}%  "
              f"AvgWin={r['avg_win']:+.2f}%  AvgLoss={r['avg_loss']:.2f}%  "
              f"AvgDays={r['avg_days']:.1f}", flush=True)
        # Per-regime breakdown
        print(f"    Regime breakdown:", flush=True)
        for rn, s in sorted(r['regime_stats'].items(), key=lambda x: -x[1]['n']):
            rwr = s['w'] / max(s['n'], 1) * 100
            print(f"      {rn:8s}: {s['n']:4d}t WR={rwr:.0f}% pnl={s['pnl']:+.1f}%", flush=True)
        # Exit reasons
        print(f"    Exit reasons:", flush=True)
        for reason, s in sorted(r['reasons'].items(), key=lambda x: -x[1]['n']):
            rwr = s['w'] / max(s['n'], 1) * 100
            print(f"      {reason:12s}: {s['n']:4d}t WR={rwr:.0f}% pnl={s['pnl']:+.1f}%", flush=True)
        # Year-by-year
        print(f"    Year-by-year:", flush=True)
        for y in sorted(r['years'].keys()):
            s = r['years'][y]
            wr = s['w'] / max(s['n'], 1) * 100
            print(f"      {y}: {s['n']:3d}t WR={wr:.0f}% pnl={s['pnl']:+.1f}%", flush=True)

    # ------------------------------------------------------------------
    # Summary by variant
    # ------------------------------------------------------------------
    print(f"\n  === BEST PER VARIANT ===", flush=True)
    for variant in variants:
        vresults = [r for r in results if r['name'].startswith(variant[:6])]
        if vresults:
            best = vresults[0]
            print(f"  {variant:<20s}: Ann={best['ann']:+.1f}% WR={best['wr']:.0f}% "
                  f"DD={best['dd']:.1f}% N={best['n']}", flush=True)
        else:
            print(f"  {variant:<20s}: no profitable results", flush=True)

    print(f"\n{'='*95}", flush=True)
    print(f"  Done. Total time: {time.time()-t_sweep:.0f}s", flush=True)
    print(f"{'='*95}", flush=True)
