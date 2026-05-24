"""
Alpha Futures V32 — Energy Conservation Reversal Strategy
==========================================================
Core Physics Analogy: Markets obey an energy conservation law.
When kinetic energy (momentum) and potential energy (distance from mean)
are BOTH at extremes, the market MUST reverse — energy must be released.

Energy Components:
  KE (Kinetic Energy)   = mom5^2 * volume — squared momentum weighted by volume
                          High KE = strong directional force that must exhaust
  PE (Potential Energy)  = -(close - SMA20)^2 / SMA20 — distance from mean
                          High |PE| = stretched rubber band ready to snap back
  OI Energy             = OI_change_rate * close_change_rate — when both same
                          direction, energy is building (new positions opening)
  Vol Energy            = ATR5/ATR20 ratio — compression/expansion state
                          Low vol = coiled spring; high vol = energy release

Total Energy = 0.3*KE + 0.3*PE + 0.2*OI + 0.2*Vol (z-scored over 60 days)

SIGNAL:
  BULLISH REVERSAL: Total Energy < -2sigma (extremely negative = oversold)
                    AND VDP flips positive (institutional buying starts)
  BEARISH REVERSAL: Total Energy > +2sigma (extremely positive = overbought)
                    AND VDP flips negative (institutional selling starts)

RANKING: Score = -sign(Energy_z) * |z| * VDP_direction
         Pick the commodity with the most extreme energy reversal signal.

Design principles from 345+ strategy analysis:
  - VDP flip is the most reliable entry trigger
  - OI provides futures-unique edge (money flow)
  - Vol compression precedes expansion (universal edge)
  - Simple engine + good signals > complex engine

Single position, P1 concentrated, no leverage, COMM=0.0003
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
    print("=" * 120)
    print("Alpha Futures V32 — Energy Conservation Reversal Strategy")
    print("KE (momentum^2 * vol) + PE (mean-reversion) + OI Energy + Vol Energy")
    print("Extreme energy states must release → reversal signals")
    print("=" * 120)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    # ================================================================
    # PHASE 1: PRECOMPUTE ALL ENERGY COMPONENTS
    # ================================================================
    print("\n[Phase 1] Computing energy components...", flush=True)
    t0 = time.time()

    # --- 1. Momentum 5 (needed for KE) ---
    mom5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            d = di - 1
            c_now = C[si, d]
            c_prev = C[si, d - 5]
            if np.isnan(c_now) or np.isnan(c_prev) or c_prev <= 0:
                continue
            mom5[si, di] = (c_now - c_prev) / c_prev

    # --- 2. SMA 20 (needed for PE) ---
    sma20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            window = C[si, di - 20:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= 10:
                sma20[si, di] = np.mean(valid)

    # --- 3. ATR 5 and ATR 20 (needed for Vol Energy) ---
    atr5 = np.full((NS, ND), np.nan)
    atr20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            # ATR 5
            trs5 = []
            for dd in range(di - 5, di):
                hi, lo = H[si, dd], L[si, dd]
                pc = C[si, dd - 1] if dd > 0 else np.nan
                if np.isnan(hi) or np.isnan(lo):
                    continue
                tr = hi - lo
                if not np.isnan(pc):
                    tr = max(tr, abs(hi - pc), abs(lo - pc))
                trs5.append(tr)
            if trs5:
                atr5[si, di] = np.mean(trs5)

            # ATR 20
            trs20 = []
            for dd in range(di - 20, di):
                hi, lo = H[si, dd], L[si, dd]
                pc = C[si, dd - 1] if dd > 0 else np.nan
                if np.isnan(hi) or np.isnan(lo):
                    continue
                tr = hi - lo
                if not np.isnan(pc):
                    tr = max(tr, abs(hi - pc), abs(lo - pc))
                trs20.append(tr)
            if trs20:
                atr20[si, di] = np.mean(trs20)

    # --- 4. VDP EMA and delta (for signal confirmation) ---
    vdp_ema = np.full((NS, ND), np.nan)
    vdp_prev = np.full((NS, ND), np.nan)
    for si in range(NS):
        vdp_e = 0.0
        alpha = 2.0 / 15
        for di in range(1, ND):
            d = di - 1
            cd, hd, ld, vd = C[si, d], H[si, d], L[si, d], V[si, d]
            if any(np.isnan([cd, hd, ld, vd])) or hd == ld:
                continue
            vdp_val = vd * (2 * cd - hd - ld) / (hd - ld)
            prev_e = vdp_e
            vdp_e = alpha * vdp_val + (1 - alpha) * vdp_e
            vdp_ema[si, di] = vdp_e
            vdp_prev[si, di] = prev_e

    # --- 5. OI change rate and close change rate (for OI Energy) ---
    oi_change_rate = np.full((NS, ND), np.nan)
    close_change_rate = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            d = di - 1
            pc = C[si, d]
            pp = C[si, d - 1] if d > 0 else np.nan
            oc = OI[si, d]
            op = OI[si, d - 1] if d > 0 else np.nan
            if np.isnan(pc) or np.isnan(pp) or pp <= 0:
                continue
            close_change_rate[si, di] = (pc - pp) / pp
            if not np.isnan(oc) and not np.isnan(op) and op > 0:
                oi_change_rate[si, di] = (oc - op) / op

    print(f"  Raw components done ({time.time()-t0:.1f}s)", flush=True)

    # ================================================================
    # PHASE 2: COMPUTE ENERGY COMPONENTS (per symbol per day)
    # ================================================================
    print("[Phase 2] Computing energy fields...", flush=True)

    # Energy arrays (unnormalized)
    KE = np.full((NS, ND), np.nan)       # Kinetic Energy
    PE = np.full((NS, ND), np.nan)       # Potential Energy
    OI_E = np.full((NS, ND), np.nan)     # OI Energy
    VOL_E = np.full((NS, ND), np.nan)    # Vol Energy

    for si in range(NS):
        for di in range(21, ND):
            d = di - 1

            # Kinetic Energy = mom5^2 * volume
            m5 = mom5[si, di]
            vol = V[si, d]
            if not np.isnan(m5) and not np.isnan(vol) and vol > 0:
                # Normalize volume by 20-day avg to make cross-commodity comparable
                vol_window = V[si, max(0, d - 19):d + 1]
                vol_valid = vol_window[~np.isnan(vol_window)]
                if len(vol_valid) >= 5:
                    vol_avg = np.mean(vol_valid)
                    if vol_avg > 0:
                        rel_vol = vol / vol_avg
                        KE[si, di] = m5 ** 2 * rel_vol
                    else:
                        KE[si, di] = m5 ** 2
                else:
                    KE[si, di] = m5 ** 2

            # Potential Energy = -(close - SMA20)^2 / SMA20
            c = C[si, d]
            sma = sma20[si, di]
            if not np.isnan(c) and not np.isnan(sma) and sma > 0:
                PE[si, di] = -((c - sma) ** 2) / sma

            # OI Energy = OI_change_rate * close_change_rate
            # Positive when both same direction (energy building)
            # Negative when they diverge (energy dissipating)
            ocr = oi_change_rate[si, di]
            ccr = close_change_rate[si, di]
            if not np.isnan(ocr) and not np.isnan(ccr):
                OI_E[si, di] = ocr * ccr

            # Vol Energy = ATR5 / ATR20 ratio
            a5 = atr5[si, di]
            a20 = atr20[si, di]
            if not np.isnan(a5) and not np.isnan(a20) and a20 > 0:
                # Ratio centered around 1.0: values < 1 = compression, > 1 = expansion
                VOL_E[si, di] = a5 / a20 - 1.0

    print(f"  Energy fields done ({time.time()-t0:.1f}s)", flush=True)

    # ================================================================
    # PHASE 3: NORMALIZE AND COMPUTE TOTAL ENERGY (z-score over 60 days)
    # ================================================================
    print("[Phase 3] Computing total energy with 60-day rolling z-score...", flush=True)

    ROLL = 60  # rolling window for normalization
    total_energy = np.full((NS, ND), np.nan)
    energy_z = np.full((NS, ND), np.nan)

    for si in range(NS):
        for di in range(ROLL + 21, ND):
            ke, pe, oi_e, vol_e = KE[si, di], PE[si, di], OI_E[si, di], VOL_E[si, di]

            # Need at least KE and PE to compute total energy
            if np.isnan(ke) or np.isnan(pe):
                continue

            # Compute z-scores for each component over rolling window
            def zscore_val(arr, si_idx, di_idx, window):
                """Compute z-score of current value relative to rolling window."""
                cur = arr[si_idx, di_idx]
                if np.isnan(cur):
                    return np.nan
                w = arr[si_idx, di_idx - window:di_idx]
                wv = w[~np.isnan(w)]
                if len(wv) < 10:
                    return np.nan
                mean = np.mean(wv)
                std = np.std(wv)
                if std < 1e-12:
                    return 0.0
                return (cur - mean) / std

            z_ke = zscore_val(KE, si, di, ROLL)
            z_pe = zscore_val(PE, si, di, ROLL)
            z_oi = zscore_val(OI_E, si, di, ROLL)
            z_vol = zscore_val(VOL_E, si, di, ROLL)

            # Weighted total energy z-score
            # KE=0.3, PE=0.3, OI=0.2, Vol=0.2
            e_total = 0.0
            weight_sum = 0.0

            if not np.isnan(z_ke):
                e_total += 0.3 * z_ke
                weight_sum += 0.3
            if not np.isnan(z_pe):
                e_total += 0.3 * z_pe
                weight_sum += 0.3
            if not np.isnan(z_oi):
                e_total += 0.2 * z_oi
                weight_sum += 0.2
            if not np.isnan(z_vol):
                e_total += 0.2 * z_vol
                weight_sum += 0.2

            if weight_sum > 0:
                # Normalize by actual weight used (in case some components are missing)
                total_energy[si, di] = e_total
                energy_z[si, di] = e_total / weight_sum * (0.3 + 0.3 + 0.2 + 0.2)

    print(f"  Total energy done ({time.time()-t0:.1f}s)", flush=True)

    # ================================================================
    # PHASE 4: SCORING FUNCTION
    # ================================================================
    print("[Phase 4] Defining scoring functions...", flush=True)

    def score_energy_reversal(si, di, threshold=2.0):
        """
        Energy reversal scoring.
        Returns positive score for bullish reversal, negative for bearish.
        Score magnitude = how extreme the energy state is.
        """
        if di < ROLL + 21:
            return np.nan

        ez = energy_z[si, di]
        if np.isnan(ez):
            return np.nan

        # VDP direction for confirmation
        vdp_cur = vdp_ema[si, di]
        vdp_prv = vdp_prev[si, di]
        if np.isnan(vdp_cur) or np.isnan(vdp_prv):
            return np.nan

        vdp_direction = 1 if vdp_cur > 0 else -1
        vdp_flipped = False
        if vdp_prv <= 0 and vdp_cur > 0:
            vdp_flipped = True  # bullish flip
        elif vdp_prv >= 0 and vdp_cur < 0:
            vdp_flipped = True  # bearish flip

        # Check for extreme energy states
        if ez < -threshold:
            # Extremely negative energy = oversold → bullish reversal
            # Score = -sign(z) * |z| * VDP_direction
            # When z < -threshold: -sign(z) = +1, so score is positive * VDP
            score = -np.sign(ez) * abs(ez) * vdp_direction

            # Bonus for VDP flip (strongest signal)
            if vdp_flipped and vdp_direction > 0:
                score *= 1.5

            return score

        elif ez > threshold:
            # Extremely positive energy = overbought → bearish reversal
            # -sign(z) = -1, so score is negative * VDP
            score = -np.sign(ez) * abs(ez) * vdp_direction

            # Bonus for VDP flip
            if vdp_flipped and vdp_direction < 0:
                score *= 1.5

            return score

        return np.nan

    def score_energy_reversal_relaxed(si, di, threshold=1.5):
        """Relaxed threshold version — more trades."""
        return score_energy_reversal(si, di, threshold)

    def score_energy_strict(si, di, threshold=2.5):
        """Strict threshold — fewer but higher quality trades."""
        return score_energy_reversal(si, di, threshold)

    def score_energy_oi_boost(si, di, threshold=2.0):
        """Energy reversal with OI confirmation boost."""
        base = score_energy_reversal(si, di, threshold)
        if np.isnan(base):
            return np.nan

        # OI energy z-score bonus
        z_oi_val = np.nan
        w = OI_E[si, di - ROLL:di]
        wv = w[~np.isnan(w)]
        cur = OI_E[si, di]
        if not np.isnan(cur) and len(wv) >= 10:
            mean = np.mean(wv)
            std = np.std(wv)
            if std > 1e-12:
                z_oi_val = (cur - mean) / std

        # If OI energy confirms the reversal direction, boost
        if not np.isnan(z_oi_val):
            if base > 0 and z_oi_val > 0:
                base *= 1.3  # OI building in reversal direction
            elif base < 0 and z_oi_val < 0:
                base *= 1.3

        return base

    def score_energy_momentum(si, di, threshold=1.5):
        """Energy reversal + momentum confirmation (hybrid)."""
        base = score_energy_reversal(si, di, threshold)
        if np.isnan(base):
            return np.nan

        # Add momentum as a secondary factor
        m5 = mom5[si, di]
        if not np.isnan(m5):
            # For bullish reversal, want to see momentum starting to turn
            if base > 0 and m5 > -0.02:
                base *= 1.2
            elif base < 0 and m5 < 0.02:
                base *= 1.2

        return base

    # ================================================================
    # PHASE 5: BACKTEST ENGINE
    # ================================================================

    def run_backtest(score_fn, name, hold_max=5, threshold=2.0,
                     allow_short=True, trail_atr=2.5, stop_loss=0.05):
        """
        Single-position rotation backtest.
        P1 concentrated, no leverage.
        """
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

                # 1. Stop loss
                if pnl_pct / 100 < -stop_loss:
                    exit_reason = 'stop'

                # 2. Trailing stop
                if exit_reason is None and trail_atr > 0 and days_held >= 2:
                    atr = pos.get('atr', 0)
                    if atr > 0:
                        trail_price = pos.get('trail_price', pos['entry'])
                        if pos['dir'] == 1:
                            new_trail = c - trail_atr * atr
                            if new_trail > trail_price:
                                pos['trail_price'] = new_trail
                            if c < trail_price:
                                exit_reason = 'trail'
                        else:
                            new_trail = c + trail_atr * atr
                            if new_trail < trail_price:
                                pos['trail_price'] = new_trail
                            if c > trail_price:
                                exit_reason = 'trail'

                # 3. Signal flip
                if exit_reason is None and days_held >= 2:
                    cur_score = score_fn(pos['si'], di, threshold)
                    if not np.isnan(cur_score):
                        if pos['dir'] == 1 and cur_score < -0.02:
                            exit_reason = 'signal_flip'
                        elif pos['dir'] == -1 and cur_score > 0.02:
                            exit_reason = 'signal_flip'

                # 4. Time exit
                if exit_reason is None and days_held >= hold_max:
                    exit_reason = 'time'

                # 5. Rotation — switch to a stronger signal
                if exit_reason is None and days_held >= 2:
                    best_si, best_dir, best_sc = -1, 0, 0
                    for sj in range(NS):
                        sc = score_fn(sj, di, threshold)
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

                    if best_si >= 0 and best_si != pos['si']:
                        cur_sc = abs(score_fn(pos['si'], di, threshold))
                        if np.isnan(cur_sc):
                            cur_sc = 0
                        if best_sc > cur_sc * 1.5 + 0.05:
                            exit_reason = 'rotate'

                if exit_reason:
                    cost_out = mkt_val * COMM
                    cash += mkt_val - cost_out
                    trades.append({
                        'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                        'days': days_held, 'di': di, 'year': year,
                        'sym': pos['sym'], 'dir': pos['dir'],
                        'reason': exit_reason,
                    })
                    last_exit[pos['sym']] = di
                    pos = None

            # === OPEN NEW POSITION ===
            if pos is None:
                best_si, best_dir, best_sc = -1, 0, 0
                for si in range(NS):
                    sc = score_fn(si, di, threshold)
                    if np.isnan(sc):
                        continue
                    sym = syms[si]
                    # Avoid immediate re-entry
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

                if best_si >= 0 and best_sc > 0.1:
                    c = C[best_si, di]
                    if np.isnan(c) or c <= 0:
                        continue

                    sym = syms[best_si]
                    mult = MULT.get(sym, DEF_MULT)
                    notional = c * mult
                    if notional <= 0:
                        continue

                    lots = int(cash / notional)
                    if lots <= 0:
                        continue

                    cost_in = notional * lots * (1 + COMM)
                    if cost_in > cash:
                        lots = int(cash / (notional * (1 + COMM)))
                    if lots <= 0:
                        continue
                    cost_in = notional * lots * (1 + COMM)

                    # Compute ATR for trailing stop
                    atr_val = 0
                    trs = []
                    for dd in range(max(1, di - 14), di + 1):
                        hi, lo = H[best_si, dd], L[best_si, dd]
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
                    if best_dir == 1:
                        tp = c - trail_atr * atr_val
                    else:
                        tp = c + trail_atr * atr_val
                    pos = {
                        'si': best_si, 'entry': c, 'entry_di': di,
                        'lots': lots, 'dir': best_dir, 'sym': sym,
                        'atr': atr_val, 'trail_price': tp,
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
                'pnl_pct': pnl / (pos['entry'] * mult * pos['lots']) * 100,
                'pnl_abs': pnl, 'days': ND - 1 - pos['entry_di'],
                'di': ND - 1, 'year': dates[ND - 1].year,
                'sym': pos['sym'], 'dir': pos['dir'], 'reason': 'end',
            })

        if len(trades) < 10:
            return None

        # === COMPUTE STATISTICS ===
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

        long_trades = [t for t in trades if t['dir'] == 1]
        short_trades = [t for t in trades if t['dir'] == -1]
        long_wr = sum(1 for t in long_trades if t['pnl_abs'] > 0) / max(len(long_trades), 1) * 100
        short_wr = sum(1 for t in short_trades if t['pnl_abs'] > 0) / max(len(short_trades), 1) * 100

        return {
            'name': name, 'ann': round(ann, 1), 'n': len(trades),
            'wr': round(wr, 1), 'dd': round(max_dd, 1),
            'avg_pnl': round(avg_pnl, 3), 'avg_days': round(avg_days, 1),
            'avg_win': round(avg_win, 3), 'avg_loss': round(avg_loss, 3),
            'cash': round(cash, 0), 'reasons': reasons, 'yearly': year_stats,
            'n_long': len(long_trades), 'n_short': len(short_trades),
            'long_wr': round(long_wr, 1), 'short_wr': round(short_wr, 1),
        }

    # ================================================================
    # PHASE 6: RUN ALL CONFIGURATIONS
    # ================================================================
    print(f"\n[Phase 5] Running backtests...", flush=True)
    results = []

    configs = [
        # (score_fn, name, hold_max, threshold, allow_short, trail_atr, stop_loss)
        # --- Core energy reversal ---
        (score_energy_reversal, "ENERGY_T2.0_H3_TRL2.5", 3, 2.0, True, 2.5, 0.05),
        (score_energy_reversal, "ENERGY_T2.0_H5_TRL2.5", 5, 2.0, True, 2.5, 0.05),
        (score_energy_reversal, "ENERGY_T2.0_H5_TRL2.0", 5, 2.0, True, 2.0, 0.05),
        (score_energy_reversal, "ENERGY_T2.0_H5_TRL3.0", 5, 2.0, True, 3.0, 0.05),
        (score_energy_reversal, "ENERGY_T2.0_H7_TRL2.5", 7, 2.0, True, 2.5, 0.05),
        (score_energy_reversal, "ENERGY_T2.0_H3_TRL2.0", 3, 2.0, True, 2.0, 0.05),
        (score_energy_reversal, "ENERGY_T2.0_H3_TRL3.0", 3, 2.0, True, 3.0, 0.05),

        # --- Relaxed threshold (more trades) ---
        (score_energy_reversal_relaxed, "ENERGY_T1.5_H3_TRL2.5", 3, 1.5, True, 2.5, 0.05),
        (score_energy_reversal_relaxed, "ENERGY_T1.5_H5_TRL2.5", 5, 1.5, True, 2.5, 0.05),
        (score_energy_reversal_relaxed, "ENERGY_T1.5_H5_TRL2.0", 5, 1.5, True, 2.0, 0.05),
        (score_energy_reversal_relaxed, "ENERGY_T1.5_H3_TRL2.0", 3, 1.5, True, 2.0, 0.05),
        (score_energy_reversal_relaxed, "ENERGY_T1.5_H7_TRL2.5", 7, 1.5, True, 2.5, 0.05),

        # --- Strict threshold (fewer, higher quality) ---
        (score_energy_strict, "ENERGY_T2.5_H5_TRL2.5", 5, 2.5, True, 2.5, 0.05),
        (score_energy_strict, "ENERGY_T2.5_H3_TRL2.5", 3, 2.5, True, 2.5, 0.05),
        (score_energy_strict, "ENERGY_T2.5_H7_TRL2.5", 7, 2.5, True, 2.5, 0.05),
        (score_energy_strict, "ENERGY_T2.5_H5_TRL2.0", 5, 2.5, True, 2.0, 0.05),
        (score_energy_strict, "ENERGY_T2.5_H5_TRL3.0", 5, 2.5, True, 3.0, 0.05),

        # --- OI boost ---
        (score_energy_oi_boost, "ENERGY_OI_T2.0_H3_TRL2.5", 3, 2.0, True, 2.5, 0.05),
        (score_energy_oi_boost, "ENERGY_OI_T2.0_H5_TRL2.5", 5, 2.0, True, 2.5, 0.05),
        (score_energy_oi_boost, "ENERGY_OI_T2.0_H5_TRL2.0", 5, 2.0, True, 2.0, 0.05),

        # --- Momentum hybrid ---
        (score_energy_momentum, "ENERGY_MOM_T1.5_H3_TRL2.5", 3, 1.5, True, 2.5, 0.05),
        (score_energy_momentum, "ENERGY_MOM_T1.5_H5_TRL2.5", 5, 1.5, True, 2.5, 0.05),
        (score_energy_momentum, "ENERGY_MOM_T2.0_H3_TRL2.5", 3, 2.0, True, 2.5, 0.05),
        (score_energy_momentum, "ENERGY_MOM_T2.0_H5_TRL2.5", 5, 2.0, True, 2.5, 0.05),

        # --- Long only ---
        (score_energy_reversal, "ENERGY_LONG_T2.0_H5_TRL2.5", 5, 2.0, False, 2.5, 0.05),
        (score_energy_reversal, "ENERGY_LONG_T1.5_H5_TRL2.5", 5, 1.5, False, 2.5, 0.05),
        (score_energy_reversal, "ENERGY_LONG_T2.0_H3_TRL2.5", 3, 2.0, False, 2.5, 0.05),
        (score_energy_reversal, "ENERGY_LONG_T1.5_H3_TRL2.5", 3, 1.5, False, 2.5, 0.05),

        # --- No trailing (pure signal exit) ---
        (score_energy_reversal, "ENERGY_T2.0_H5_NOTRL", 5, 2.0, True, 0.0, 0.05),
        (score_energy_reversal, "ENERGY_T1.5_H5_NOTRL", 5, 1.5, True, 0.0, 0.05),
        (score_energy_reversal, "ENERGY_T2.0_H3_NOTRL", 3, 2.0, True, 0.0, 0.05),
    ]

    for fn, name, hm, thr, ashort, ta, sl in configs:
        r = run_backtest(fn, name, hold_max=hm, threshold=thr,
                         allow_short=ashort, trail_atr=ta, stop_loss=sl)
        if r:
            results.append(r)
            print(f"  {r['name']:40s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"N {r['n']:4d} (L{r['n_long']}/S{r['n_short']}) | "
                  f"DD {r['dd']:6.1f}% | AvgW {r['avg_win']:+.2f}% | "
                  f"AvgL {r['avg_loss']:.2f}% | AvgD {r['avg_days']:.1f}")
            parts = []
            for reason, stats in sorted(r['reasons'].items()):
                rwr = stats['w'] / stats['n'] * 100 if stats['n'] > 0 else 0
                parts.append(f"{reason}:{stats['n']}({rwr:.0f}%)pnl={stats['pnl']:+.0f}%")
            print(f"  {'':40s} | {' | '.join(parts)}")

    # ================================================================
    # PHASE 7: SUMMARY
    # ================================================================
    if not results:
        print("\nNo profitable configurations found!")
        elapsed = time.time() - t_start
        print(f"\nTotal time: {elapsed:.1f}s")
        return

    results.sort(key=lambda x: -x['ann'])

    print(f"\n{'=' * 120}")
    print(f"TOP 15 BY ANNUAL RETURN")
    print(f"{'=' * 120}")
    for r in results[:15]:
        ratio = r['ann'] / max(r['dd'], 1)
        print(f"  {r['name']:40s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
              f"N {r['n']:4d} | DD {r['dd']:6.1f}% | R/A {ratio:.2f} | "
              f"LWR {r['long_wr']:.0f}% / SWR {r['short_wr']:.0f}%")

    # Top by risk-adjusted
    print(f"\n--- TOP 10 BY RISK-ADJUSTED (Ann/DD) ---")
    by_ra = sorted(results, key=lambda x: -x['ann'] / max(x['dd'], 1))
    for r in by_ra[:10]:
        ratio = r['ann'] / max(r['dd'], 1)
        print(f"  {r['name']:40s} | Ann {r['ann']:+7.1f}% | DD {r['dd']:6.1f}% | "
              f"R/A {ratio:.2f} | WR {r['wr']:5.1f}% | N {r['n']}")

    # Top by win rate
    print(f"\n--- TOP 10 BY WIN RATE ---")
    by_wr = sorted(results, key=lambda x: -x['wr'])
    for r in by_wr[:10]:
        print(f"  {r['name']:40s} | WR {r['wr']:5.1f}% | Ann {r['ann']:+7.1f}% | "
              f"N {r['n']:4d} | DD {r['dd']:6.1f}%")

    # Yearly breakdown for top 5
    print(f"\n--- YEARLY BREAKDOWN (Top 5 by Ann) ---")
    for r in results[:5]:
        print(f"\n  {r['name']} (Ann {r['ann']:+.1f}%, WR {r['wr']:.1f}%, DD {r['dd']:.1f}%):")
        for y in sorted(r['yearly'].keys()):
            ys = r['yearly'][y]
            wr = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
            print(f"    {y}: {ys['n']:3d} trades, WR {wr:5.1f}%, PnL {ys['pnl']:+.1f}%")

    # Exit analysis for top 3
    print(f"\n--- EXIT ANALYSIS (Top 3) ---")
    for r in results[:3]:
        print(f"\n  {r['name']}:")
        for reason, stats in sorted(r['reasons'].items()):
            rwr = stats['w'] / stats['n'] * 100 if stats['n'] > 0 else 0
            avg = stats['pnl'] / stats['n'] if stats['n'] > 0 else 0
            print(f"    {reason:12s}: {stats['n']:4d} trades, WR {rwr:5.1f}%, "
                  f"Total {stats['pnl']:+.1f}%, Avg {avg:+.3f}%")

    # Long vs Short analysis for top 3
    print(f"\n--- LONG vs SHORT ANALYSIS (Top 3) ---")
    for r in results[:3]:
        print(f"  {r['name']}: Long {r['n_long']}t (WR {r['long_wr']:.1f}%) | "
              f"Short {r['n_short']}t (WR {r['short_wr']:.1f}%)")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
