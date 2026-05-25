"""
Alpha Futures V30 — TTM Squeeze + OI Surge (Vol Compression / Expansion)
=========================================================================
Core idea: TTM Squeeze (BB inside KC) + OI surge as the ONLY entry trigger.

1. Bollinger Bands (20-day, 2 sigma): upper, lower, bandwidth
2. Keltner Channels (20-day EMA, 1.5x ATR): upper, lower
3. SQUEEZE = BB_upper < KC_upper AND BB_lower > KC_lower (vol compression)
4. SQUEEZE_RELEASE = squeeze was on but now off
5. At release: momentum direction = linear regression slope of 20-day close
6. OI surge: OI > 1.5x 20-day average OI

Entry (ALL must be true):
  - Squeeze just released
  - Momentum slope > 0 -> BUY; slope < 0 -> SELL
  - OI surge confirms institutional participation
  - Volume > 1.2x average

Scoring: score = (momentum_slope_normalized) * (1 + squeeze_duration_bonus) * OI_surge_bonus
  - squeeze_duration_bonus: longer squeeze = stronger (cap at 2x)
  - OI_surge_bonus: OI_ratio > 1.5 = 1.3x, > 2.0 = 1.5x

Cross-sectional: pick top-scoring commodity each day.
Exit: hold_max=5, signal flip, rotation to 50%+ better.
Single position, P1 concentrated, no leverage, COMM=0.0003.
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


def main():
    t_start = time.time()
    print("=" * 110)
    print("Alpha Futures V30 — TTM Squeeze + OI Surge")
    print("Vol Compression (BB inside KC) -> Expansion + OI Confirmation")
    print("=" * 110)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    # ========================================
    # PRECOMPUTE ALL SIGNALS
    # ========================================
    print("\n[1/8] Computing Bollinger Bands (20-day, 2 sigma)...", flush=True)
    t0 = time.time()

    bb_upper = np.full((NS, ND), np.nan)
    bb_lower = np.full((NS, ND), np.nan)
    bb_sma = np.full((NS, ND), np.nan)
    bb_bw = np.full((NS, ND), np.nan)  # bandwidth = (upper - lower) / SMA

    for si in range(NS):
        for di in range(20, ND):
            cs = C[si, di - 20:di]
            valid = cs[~np.isnan(cs)]
            if len(valid) < 15:
                continue
            sma = np.mean(valid)
            std = np.std(valid, ddof=0)
            bb_upper[si, di] = sma + 2.0 * std
            bb_lower[si, di] = sma - 2.0 * std
            bb_sma[si, di] = sma
            if sma > 0:
                bb_bw[si, di] = (bb_upper[si, di] - bb_lower[si, di]) / sma

    print(f"  Done ({time.time() - t0:.1f}s)", flush=True)

    # --- Keltner Channels (20-day EMA, 1.5 x ATR) ---
    print("[2/8] Computing Keltner Channels (20-day EMA, 1.5x ATR)...", flush=True)
    t0 = time.time()

    kc_upper = np.full((NS, ND), np.nan)
    kc_lower = np.full((NS, ND), np.nan)
    kc_ema = np.full((NS, ND), np.nan)
    atr20 = np.full((NS, ND), np.nan)

    for si in range(NS):
        # Compute EMA of close (20-day)
        ema_val = np.nan
        alpha_ema = 2.0 / 21.0
        ema_started = False
        # Compute ATR (20-day)
        for di in range(1, ND):
            # ATR component
            hi, lo = H[si, di - 1], L[si, di - 1]
            pc = C[si, di - 2] if di >= 2 else np.nan
            if np.isnan(hi) or np.isnan(lo):
                continue
            tr = hi - lo
            if not np.isnan(pc):
                tr = max(tr, abs(hi - pc), abs(lo - pc))

            # EMA of close
            c_val = C[si, di - 1]
            if np.isnan(c_val):
                continue
            if not ema_started:
                ema_val = c_val
                ema_started = True
            else:
                ema_val = alpha_ema * c_val + (1 - alpha_ema) * ema_val

            if di >= 21:
                # ATR over last 20 bars
                trs = []
                for dd in range(di - 20, di):
                    h2 = H[si, dd]; l2 = L[si, dd]; p2 = C[si, dd - 1]
                    if np.isnan(h2) or np.isnan(l2):
                        continue
                    t_val = h2 - l2
                    if not np.isnan(p2):
                        t_val = max(t_val, abs(h2 - p2), abs(l2 - p2))
                    trs.append(t_val)
                if len(trs) >= 10:
                    atr_val = np.mean(trs)
                    atr20[si, di] = atr_val
                    kc_ema[si, di] = ema_val
                    kc_upper[si, di] = ema_val + 1.5 * atr_val
                    kc_lower[si, di] = ema_val - 1.5 * atr_val

    print(f"  Done ({time.time() - t0:.1f}s)", flush=True)

    # --- SQUEEZE detection ---
    print("[3/8] Computing SQUEEZE and SQUEEZE_RELEASE...", flush=True)
    t0 = time.time()

    squeeze_on = np.zeros((NS, ND), dtype=bool)
    squeeze_release = np.zeros((NS, ND), dtype=bool)
    squeeze_duration = np.zeros((NS, ND), dtype=np.int32)

    for si in range(NS):
        dur = 0
        for di in range(21, ND):
            bbu = bb_upper[si, di]
            bbl = bb_lower[si, di]
            kcu = kc_upper[si, di]
            kcl = kc_lower[si, di]
            if np.isnan(bbu) or np.isnan(bbl) or np.isnan(kcu) or np.isnan(kcl):
                dur = 0
                continue
            # Squeeze: BB inside KC
            is_squeeze = (bbu < kcu) and (bbl > kcl)
            squeeze_on[si, di] = is_squeeze
            if is_squeeze:
                dur += 1
                squeeze_duration[si, di] = dur
            else:
                # Release: was squeezed, now expanding
                if di > 21 and squeeze_on[si, di - 1]:
                    squeeze_release[si, di] = True
                dur = 0

    print(f"  Done ({time.time() - t0:.1f}s)", flush=True)

    # --- Momentum slope (linear regression of 20-day close) ---
    print("[4/8] Computing momentum slope (20-day linear regression)...", flush=True)
    t0 = time.time()

    mom_slope = np.full((NS, ND), np.nan)

    for si in range(NS):
        for di in range(21, ND):
            cs = C[si, di - 20:di]
            valid_mask = ~np.isnan(cs)
            valid = cs[valid_mask]
            if len(valid) < 12:
                continue
            n = len(valid)
            x = np.arange(n, dtype=float)
            y = valid
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            ss_xx = np.sum((x - x_mean) ** 2)
            ss_xy = np.sum((x - x_mean) * (y - y_mean))
            if ss_xx > 0:
                slope = ss_xy / ss_xx
                # Normalize by mean price to get relative slope
                if y_mean > 0:
                    mom_slope[si, di] = slope / y_mean

    print(f"  Done ({time.time() - t0:.1f}s)", flush=True)

    # --- OI surge (OI > 1.5x 20-day average) ---
    print("[5/8] Computing OI surge and OI ratio...", flush=True)
    t0 = time.time()

    oi_ratio = np.full((NS, ND), np.nan)
    oi_surge = np.zeros((NS, ND), dtype=bool)

    for si in range(NS):
        for di in range(21, ND):
            oi_now = OI[si, di - 1]
            if np.isnan(oi_now) or oi_now <= 0:
                continue
            oi_window = OI[si, di - 21:di - 1]
            oi_valid = oi_window[~np.isnan(oi_window)]
            if len(oi_valid) < 10:
                continue
            avg_oi = np.mean(oi_valid)
            if avg_oi > 0:
                oi_ratio[si, di] = oi_now / avg_oi
                if oi_ratio[si, di] >= 1.5:
                    oi_surge[si, di] = True

    print(f"  Done ({time.time() - t0:.1f}s)", flush=True)

    # --- Volume ratio (vs 20-day average) ---
    print("[6/8] Computing volume ratio...", flush=True)
    t0 = time.time()

    vol_ratio = np.full((NS, ND), np.nan)

    for si in range(NS):
        for di in range(21, ND):
            v_now = V[si, di - 1]
            if np.isnan(v_now) or v_now <= 0:
                continue
            v_window = V[si, di - 21:di - 1]
            v_valid = v_window[~np.isnan(v_window)]
            if len(v_valid) < 10:
                continue
            avg_vol = np.mean(v_valid)
            if avg_vol > 0:
                vol_ratio[si, di] = v_now / avg_vol

    print(f"  Done ({time.time() - t0:.1f}s)", flush=True)

    # --- Normalize momentum slope cross-sectionally ---
    print("[7/8] Normalizing momentum slope...", flush=True)
    t0 = time.time()

    mom_slope_norm = np.full((NS, ND), np.nan)
    for di in range(21, ND):
        vals = mom_slope[:, di]
        valid_mask = ~np.isnan(vals)
        if valid_mask.sum() < 5:
            continue
        v = vals[valid_mask]
        std = np.std(v)
        if std > 0:
            mean_v = np.mean(v)
            mom_slope_norm[valid_mask, di] = (v - mean_v) / std

    print(f"  Done ({time.time() - t0:.1f}s)", flush=True)

    # --- Precompute squeeze_duration at release (for scoring) ---
    print("[8/8] Precomputing squeeze duration at release...", flush=True)
    t0 = time.time()

    release_duration = np.zeros((NS, ND), dtype=np.float64)
    for si in range(NS):
        for di in range(22, ND):
            if squeeze_release[si, di]:
                # Duration of the squeeze that just ended = look back at yesterday's duration
                release_duration[si, di] = float(squeeze_duration[si, di - 1])

    print(f"  Done ({time.time() - t0:.1f}s)", flush=True)

    # ========================================
    # SCORING FUNCTION
    # ========================================
    def score_ttm_squeeze(si, di):
        """TTM Squeeze release + OI surge + volume confirmation."""
        if di < 22:
            return np.nan

        # Must be a squeeze release
        if not squeeze_release[si, di]:
            return np.nan

        # Momentum direction from linear regression slope
        ms = mom_slope[si, di]
        if np.isnan(ms):
            return np.nan
        # Must have a clear direction
        if abs(ms) < 1e-8:
            return np.nan

        # Normalized momentum slope
        msn = mom_slope_norm[si, di]
        if np.isnan(msn):
            msn = 0.0

        # OI surge required
        oir = oi_ratio[si, di]
        if np.isnan(oir):
            return np.nan
        if oir < 1.5:
            return np.nan

        # Volume confirmation required: > 1.2x average
        vr = vol_ratio[si, di]
        if np.isnan(vr) or vr < 1.2:
            return np.nan

        # --- Build score ---
        # Base: normalized momentum slope
        score = msn

        # Squeeze duration bonus: longer squeeze = stronger breakout
        # Cap at 2x bonus
        dur = release_duration[si, di]
        dur_bonus = min(dur / 10.0, 2.0)
        score *= (1.0 + dur_bonus)

        # OI surge bonus
        if oir >= 2.0:
            score *= 1.5
        elif oir >= 1.5:
            score *= 1.3

        # Volume bonus
        if vr >= 2.0:
            score *= 1.2
        elif vr >= 1.5:
            score *= 1.1

        return score

    # ========================================
    # BACKTEST ENGINE
    # ========================================
    def run_backtest(score_fn, name, hold_max=5, trail_atr=2.0,
                     stop_loss=0.05, allow_short=True):
        """Single position, P1 concentrated, rotation-based backtest."""
        cash = float(CASH0)
        trades = []
        pos = None
        last_exit = {}

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
                pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] > 0 else 0
                days_held = di - pos['entry_di']

                exit_reason = None

                # 1. Fixed stop loss
                if pnl_pct / 100 < -stop_loss:
                    exit_reason = 'stop'

                # 2. Trailing stop
                if exit_reason is None and trail_atr > 0:
                    atr = pos.get('atr', 0)
                    if atr > 0:
                        trail_price = pos.get('trail_price', pos['entry'])
                        if pos['dir'] == 1:
                            new_trail = c - trail_atr * atr
                            if new_trail > trail_price:
                                pos['trail_price'] = new_trail
                            if c < trail_price and days_held >= 2:
                                exit_reason = 'trail'
                        else:
                            new_trail = c + trail_atr * atr
                            if new_trail < trail_price:
                                pos['trail_price'] = new_trail
                            if c > trail_price and days_held >= 2:
                                exit_reason = 'trail'

                # 3. Signal flip: if score direction opposes position
                if exit_reason is None and days_held >= 2:
                    cur_score = score_fn(pos['si'], di)
                    if not np.isnan(cur_score):
                        if pos['dir'] == 1 and cur_score < -0.02:
                            exit_reason = 'signal_flip'
                        elif pos['dir'] == -1 and cur_score > 0.02:
                            exit_reason = 'signal_flip'

                # 4. Time exit
                if exit_reason is None and days_held >= hold_max:
                    exit_reason = 'time'

                # 5. Rotation: exit if a 50%+ better candidate exists
                if exit_reason is None and days_held >= 2:
                    best_si, best_dir, best_sc = -1, 0, 0.0
                    for sj in range(NS):
                        sc = score_fn(sj, di)
                        if np.isnan(sc):
                            continue
                        if sc > best_sc:
                            best_sc = sc
                            best_si = sj
                            best_dir = 1
                        if allow_short and -sc > best_sc:
                            best_sc = -sc
                            best_si = sj
                            best_dir = -1

                    cur_sc = abs(score_fn(pos['si'], di))
                    if not np.isnan(cur_sc):
                        # Rotate if new candidate is 50%+ better
                        if best_sc > cur_sc * 1.5 + 0.05 and best_si != pos['si']:
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

            # === ENTRY: pick top-scoring commodity ===
            if pos is None:
                best_si, best_dir, best_sc = -1, 0, 0.0
                for si in range(NS):
                    sc = score_fn(si, di)
                    if np.isnan(sc):
                        continue
                    sym = syms[si]
                    # Skip if just exited this symbol
                    if sym in last_exit and di - last_exit[sym] < 1:
                        continue
                    if sc > best_sc:
                        best_sc = sc
                        best_si = si
                        best_dir = 1
                    if allow_short and -sc > best_sc:
                        best_sc = -sc
                        best_si = si
                        best_dir = -1

                if best_si >= 0 and best_sc > 0:
                    c = C[best_si, di]
                    if np.isnan(c) or c <= 0:
                        pass  # skip, no valid price
                    else:
                        sym = syms[best_si]
                        mult = MULT.get(sym, DEF_MULT)
                        notional = c * mult
                        if notional > 0:
                            lots = int(cash / notional)
                            if lots > 0:
                                cost_in = notional * lots * (1 + COMM)
                                if cost_in <= cash:
                                    # Compute ATR for trailing stop
                                    atr_val = 0.0
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
                                    trail_price = (c - trail_atr * atr_val
                                                   if best_dir == 1
                                                   else c + trail_atr * atr_val)
                                    pos = {
                                        'si': best_si, 'entry': c, 'entry_di': di,
                                        'lots': lots, 'dir': best_dir, 'sym': sym,
                                        'atr': atr_val, 'trail_price': trail_price
                                    }

        # Close remaining position
        if pos is not None:
            c = C[pos['si'], ND - 1]
            if np.isnan(c) or c <= 0:
                c = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
            cash += c * mult * pos['lots'] * (1 - COMM)
            trades.append({
                'pnl_pct': pnl / (pos['entry'] * mult * pos['lots']) * 100
                if pos['entry'] > 0 else 0,
                'pnl_abs': pnl, 'days': di - pos['entry_di'],
                'di': ND - 1, 'year': dates[ND - 1].year,
                'sym': pos['sym'], 'dir': pos['dir'], 'reason': 'end'
            })

        if len(trades) < 5:
            return None

        # --- Compute statistics ---
        equity = float(CASH0)
        peak = float(CASH0)
        max_dd = 0.0
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
        avg_win = (np.mean([t['pnl_pct'] for t in trades if t['pnl_abs'] > 0])
                   if nw > 0 else 0)
        avg_loss = (np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_abs'] <= 0])
                    if nw < len(trades) else 0)

        reasons = {}
        for t in trades:
            r = t['reason']
            if r not in reasons:
                reasons[r] = {'n': 0, 'w': 0, 'pnl': 0.0}
            reasons[r]['n'] += 1
            if t['pnl_abs'] > 0:
                reasons[r]['w'] += 1
            reasons[r]['pnl'] += t['pnl_pct']

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
            'name': name, 'ann': round(ann, 1), 'n': len(trades),
            'wr': round(wr, 1), 'dd': round(max_dd, 1),
            'avg_pnl': round(avg_pnl, 3), 'avg_days': round(avg_days, 1),
            'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
            'cash': round(cash, 0), 'reasons': reasons, 'yearly': year_stats,
        }

    # ========================================
    # RUN CONFIGURATIONS
    # ========================================
    print("\n[Backtest] Running configurations...", flush=True)
    results = []

    configs = [
        # (score_fn, name, hold_max, trail_atr, stop_loss)
        (score_ttm_squeeze, "TTM_H5_T2_S5", 5, 2.0, 0.05),
        (score_ttm_squeeze, "TTM_H5_T3_S5", 5, 3.0, 0.05),
        (score_ttm_squeeze, "TTM_H5_T2_S3", 5, 2.0, 0.03),
        (score_ttm_squeeze, "TTM_H5_T3_S3", 5, 3.0, 0.03),
        (score_ttm_squeeze, "TTM_H5_T0_S5", 5, 0.0, 0.05),
        (score_ttm_squeeze, "TTM_H3_T2_S5", 3, 2.0, 0.05),
        (score_ttm_squeeze, "TTM_H3_T3_S5", 3, 3.0, 0.05),
        (score_ttm_squeeze, "TTM_H3_T0_S5", 3, 0.0, 0.05),
        (score_ttm_squeeze, "TTM_H7_T2_S5", 7, 2.0, 0.05),
        (score_ttm_squeeze, "TTM_H7_T3_S5", 7, 3.0, 0.05),
        (score_ttm_squeeze, "TTM_H7_T0_S5", 7, 0.0, 0.05),
        (score_ttm_squeeze, "TTM_H10_T2_S5", 10, 2.0, 0.05),
        (score_ttm_squeeze, "TTM_H10_T3_S5", 10, 3.0, 0.05),
    ]

    for fn, name, hm, ta, sl in configs:
        r = run_backtest(fn, name, hold_max=hm, trail_atr=ta, stop_loss=sl)
        if r:
            results.append(r)
            print(f"  {r['name']:25s} | Ann {r['ann']:+8.1f}% | WR {r['wr']:5.1f}% | "
                  f"N {r['n']:4d} | DD {r['dd']:6.1f}% | AvgW {r['avg_win']:+6.2f}% | "
                  f"AvgL {r['avg_loss']:6.2f}% | AvgD {r['avg_days']:5.1f}", flush=True)
            parts = []
            for reason, stats in sorted(r['reasons'].items()):
                rwr = stats['w'] / stats['n'] * 100 if stats['n'] > 0 else 0
                parts.append(f"{reason}:{stats['n']}({rwr:.0f}%)pnl={stats['pnl']:+.0f}%")
            print(f"  {'':25s} | {' | '.join(parts)}", flush=True)
        else:
            print(f"  {name:25s} | No trades", flush=True)

    if not results:
        print("\n  No profitable configurations found.")
        return

    results.sort(key=lambda x: -x['ann'])

    # ========================================
    # PRINT FULL RESULTS
    # ========================================
    print(f"\n{'=' * 110}")
    print(f"  TOP RESULTS (sorted by annualized return)")
    print(f"{'=' * 110}")
    print(f"  {'Strategy':25s} | {'Ann':>8s} {'WR':>5s} {'N':>4s} {'DD':>6s} "
          f"{'AvgW':>7s} {'AvgL':>6s} {'AvgD':>5s}", flush=True)
    print(f"  {'-' * 100}")
    for r in results:
        print(f"  {r['name']:25s} | {r['ann']:+8.1f}% {r['wr']:5.1f}% {r['n']:4d} "
              f"{r['dd']:6.1f}% {r['avg_win']:+7.2f}% {r['avg_loss']:6.2f}% "
              f"{r['avg_days']:5.1f}d", flush=True)

    # Detailed breakdown for top result
    best = results[0]
    print(f"\n{'=' * 110}")
    print(f"  BEST: {best['name']}  |  Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  "
          f"N={best['n']}  DD={best['dd']:.1f}%  AvgDays={best['avg_days']:.1f}")
    print(f"  AvgWin={best['avg_win']:+.2f}%  AvgLoss={best['avg_loss']:.2f}%  "
          f"Final={best['cash']:.0f}")
    print(f"{'=' * 110}")

    # Exit reason breakdown
    print(f"\n  EXIT REASON BREAKDOWN:", flush=True)
    for reason, stats in sorted(best['reasons'].items(), key=lambda x: -x[1]['n']):
        rwr = stats['w'] / stats['n'] * 100 if stats['n'] > 0 else 0
        print(f"    {reason:12s}: {stats['n']:4d} trades  WR={rwr:5.1f}%  "
              f"PnL={stats['pnl']:+.1f}%", flush=True)

    # Yearly breakdown
    print(f"\n  YEARLY BREAKDOWN:", flush=True)
    for y in sorted(best['yearly'].keys()):
        ys = best['yearly'][y]
        ywr = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
        print(f"    {y}: {ys['n']:3d} trades  WR={ywr:5.1f}%  PnL={ys['pnl']:+.1f}%", flush=True)

    # Also show yearly for top 5 configs
    if len(results) >= 2:
        print(f"\n{'=' * 110}")
        print(f"  YEARLY BREAKDOWN FOR TOP {min(5, len(results))} CONFIGS:", flush=True)
        for rank, r in enumerate(results[:5], 1):
            print(f"\n  #{rank}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.1f}%, "
                  f"DD={r['dd']:.1f}%)", flush=True)
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                ywr = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:3d}t  WR={ywr:5.1f}%  PnL={ys['pnl']:+.1f}%", flush=True)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 110}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"{'=' * 110}")


if __name__ == '__main__':
    main()
