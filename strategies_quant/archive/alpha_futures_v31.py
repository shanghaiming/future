"""
Alpha Futures V31 — Wyckoff Accumulation + OI Footprint Strategy
================================================================
A genuinely new approach combining classical Wyckoff accumulation theory
with Open Interest analysis as an institutional footprint detector.

Wyckoff Phases Detected:
  Phase A - Selling Climax (SC): High-volume selloff near lows
  Phase B - Base/Consolidation: Low volatility range, OI building
  Phase C - Spring: Brief dip below base then recovery (shakeout)
  Phase D - Breakout: Price exits base with volume confirmation

OI as Institutional Footprint:
  - OI rising while price flat = new positions being built (smart money)
  - OI_5d_change > 0 during base = money flowing in
  - Rising OI at breakout = conviction, not just short covering

Scoring (each phase contributes points):
  +2  Selling climax detected in last 30 days
  +1  Currently in base (low volatility range, 10+ days within 5%)
  +3  Spring detected in last 5 days (dip below base low + recovery)
  +4  Breakout from base today (close above base high)
  +2  OI rising during base phase (money flowing in)
  +1  Volume surge at breakout (>1.5x average)

Direction: breakout above base = long, breakout below = short
Cross-sectional: pick highest scoring symbol each day
Exit: hold 3-7 days, time exit, signal flip, rotation
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
    print("Alpha Futures V31 — Wyckoff Accumulation + OI Footprint Strategy")
    print("Phases: Selling Climax -> Base -> Spring -> Breakout (OI confirmed)")
    print("=" * 110)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    # ==================================================================
    # PRECOMPUTE WYCKOFF PHASES + OI SIGNALS
    # ==================================================================
    print("[Signals] Computing Wyckoff phases + OI footprint...", flush=True)
    t0 = time.time()

    # --- Precompute rolling statistics ---
    # 20-day volume average
    vol_avg20 = np.full((NS, ND), np.nan)
    # 20-day low of close
    low20 = np.full((NS, ND), np.nan)
    # 20-day high of close
    high20 = np.full((NS, ND), np.nan)
    # 5-day OI change ratio
    oi_chg5 = np.full((NS, ND), np.nan)
    # OI average over base period (20-day)
    oi_avg20 = np.full((NS, ND), np.nan)

    for si in range(NS):
        for di in range(21, ND):
            # 20-day volume average
            vw = V[si, di-20:di]
            vv = vw[~np.isnan(vw)]
            if len(vv) >= 10:
                vol_avg20[si, di] = np.mean(vv)

            # 20-day high/low of close
            cw = C[si, di-20:di]
            cc = cw[~np.isnan(cw)]
            if len(cc) >= 10:
                low20[si, di] = np.min(cc)
                high20[si, di] = np.max(cc)

        for di in range(6, ND):
            # 5-day OI change
            d = di - 1
            oi_now = OI[si, d]
            oi_5ago = OI[si, max(0, d - 5)]
            if not np.isnan(oi_now) and not np.isnan(oi_5ago) and oi_5ago > 0:
                oi_chg5[si, di] = (oi_now - oi_5ago) / oi_5ago

        for di in range(21, ND):
            # 20-day OI average
            ow = OI[si, di-20:di]
            ov = ow[~np.isnan(ow)]
            if len(ov) >= 10:
                oi_avg20[si, di] = np.mean(ov)

    # --- Phase A: Selling Climax ---
    # close near 20-day low + volume > 2x 20-day average
    # "near" means close is within bottom 10% of 20-day range
    sc_detected = np.zeros((NS, ND), dtype=bool)   # selling climax flag
    sc_day = np.full((NS, ND), -1, dtype=int)       # most recent SC day

    for si in range(NS):
        last_sc = -1
        for di in range(21, ND):
            d = di - 1  # index into price arrays (di is signal index, offset by 1)
            c = C[si, d]
            v = V[si, d]
            va = vol_avg20[si, di]
            l20 = low20[si, di]
            h20 = high20[si, di]

            if np.isnan(c) or np.isnan(v) or np.isnan(va) or np.isnan(l20) or np.isnan(h20):
                sc_day[si, di] = last_sc
                continue
            if va <= 0 or h20 <= l20:
                sc_day[si, di] = last_sc
                continue

            # Close near 20-day low: within bottom 10% of range
            range20 = h20 - l20
            near_low = (c - l20) / range20 < 0.10 if range20 > 0 else False

            # Volume surge: > 2x average
            vol_surge = v > 2.0 * va

            if near_low and vol_surge:
                sc_detected[si, di] = True
                last_sc = di
            sc_day[si, di] = last_sc

    # --- Phase B: Base / Consolidation ---
    # close within 5% range for 10+ consecutive days
    # Track: base_low, base_high, base_start, base_length
    base_active = np.zeros((NS, ND), dtype=bool)
    base_low = np.full((NS, ND), np.nan)    # low of the base
    base_high = np.full((NS, ND), np.nan)   # high of the base
    base_len = np.zeros((NS, ND), dtype=int) # length of base in days

    for si in range(NS):
        for di in range(21, ND):
            # Check if last 10+ days have close within 5% range
            # Use a sliding window approach
            d = di - 1
            c = C[si, d]
            if np.isnan(c) or c <= 0:
                continue

            # Look back up to 30 days to find the longest contiguous base
            best_len = 0
            best_lo = np.nan
            best_hi = np.nan

            for start_offset in range(0, 25):  # try different start points
                window_start = d - start_offset
                if window_start < 1:
                    break
                cw = C[si, window_start:d+1]
                cvalid = cw[~np.isnan(cw)]
                if len(cvalid) < 10:
                    continue

                wlo = np.min(cvalid)
                whi = np.max(cvalid)
                wmid = (wlo + whi) / 2

                if wmid <= 0:
                    continue

                # Range as % of midpoint
                range_pct = (whi - wlo) / wmid

                if range_pct <= 0.05 and len(cvalid) >= 10:
                    if len(cvalid) > best_len:
                        best_len = len(cvalid)
                        best_lo = wlo
                        best_hi = whi
                    break  # found from this start, try shorter window

            if best_len >= 10:
                base_active[si, di] = True
                base_low[si, di] = best_lo
                base_high[si, di] = best_hi
                base_len[si, di] = best_len

    # --- Phase C: Spring ---
    # close dips below base low then recovers within 2 days
    # A "shakeout" of weak holders before the real move
    spring_detected = np.zeros((NS, ND), dtype=bool)

    for si in range(NS):
        for di in range(2, ND):
            d = di - 1
            # Look for spring pattern: 2 days ago (or yesterday) dipped below base low,
            # today recovered back above base low
            if not base_active[si, max(0, di - 3)]:
                continue

            bl = base_low[si, max(0, di - 3)]
            if np.isnan(bl):
                continue

            c_today = C[si, d]
            if np.isnan(c_today):
                continue

            # Check if any of last 2 days dipped below base low
            dipped = False
            for lookback in range(1, 3):
                dd = d - lookback
                if dd < 0:
                    continue
                cl = C[si, dd]
                if not np.isnan(cl) and cl < bl:
                    dipped = True
                    break

            # Today recovered above base low
            if dipped and c_today > bl:
                spring_detected[si, di] = True

    # --- Phase D: Breakout ---
    # close above base high with volume > 1.5x average
    breakout_up = np.zeros((NS, ND), dtype=bool)
    breakout_dn = np.zeros((NS, ND), dtype=bool)
    breakout_vol = np.zeros((NS, ND), dtype=bool)

    for si in range(NS):
        for di in range(21, ND):
            d = di - 1
            c = C[si, d]
            v = V[si, d]
            va = vol_avg20[si, di]

            if np.isnan(c) or np.isnan(v) or np.isnan(va) or va <= 0:
                continue

            # Check if there was a base recently (within last 5 days)
            had_base = False
            bh = np.nan
            bl = np.nan
            for lookback in range(0, 6):
                bdi = di - lookback
                if bdi < 0:
                    break
                if base_active[si, bdi]:
                    had_base = True
                    bh = base_high[si, bdi]
                    bl = base_low[si, bdi]
                    break

            if not had_base or np.isnan(bh) or np.isnan(bl):
                continue

            # Breakout up: close above base high
            if c > bh:
                breakout_up[si, di] = True
                if v > 1.5 * va:
                    breakout_vol[si, di] = True

            # Breakout down: close below base low (for shorts)
            if c < bl:
                breakout_dn[si, di] = True
                if v > 1.5 * va:
                    breakout_vol[si, di] = True

    # --- OI Accumulation Signal ---
    # OI rising while price in base = institutional position building
    oi_accumulating = np.zeros((NS, ND), dtype=bool)

    for si in range(NS):
        for di in range(21, ND):
            d = di - 1
            if not base_active[si, di]:
                continue

            # Check if OI has been rising during base
            # Compare current OI to OI at base start
            oi_now = OI[si, d]
            blen = base_len[si, di]
            oi_base_start = OI[si, max(0, d - blen)]

            if np.isnan(oi_now) or np.isnan(oi_base_start) or oi_base_start <= 0:
                continue

            if oi_now > oi_base_start:
                oi_accumulating[si, di] = True

    # Also compute OI 5d change for the base phase
    oi_rising_5d = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(6, ND):
            oc = oi_chg5[si, di]
            if not np.isnan(oc) and oc > 0:
                oi_rising_5d[si, di] = True

    print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

    # ==================================================================
    # WYCKOFF SCORING FUNCTION
    # ==================================================================
    def score_wyckoff(si, di):
        """Score based on Wyckoff accumulation phases + OI footprint.

        Returns (score, direction):
            score: positive number, higher = stronger setup
            direction: +1 for long, -1 for short
        """
        if di < 21:
            return np.nan, 0

        score = 0.0
        direction = 0

        # Phase A: Selling climax in last 30 days (+2)
        scd = sc_day[si, di]
        if scd > 0 and (di - scd) <= 30:
            score += 2

        # Phase B: Currently in base (+1)
        if base_active[si, di]:
            score += 1

        # Phase C: Spring in last 5 days (+3)
        for lookback in range(0, 6):
            if spring_detected[si, di - lookback]:
                score += 3
                break

        # Phase D: Breakout today (+4)
        breakout_today = False
        if breakout_up[si, di]:
            score += 4
            direction = 1
            breakout_today = True
        elif breakout_dn[si, di]:
            score += 4
            direction = -1
            breakout_today = True

        # If no breakout, infer direction from context
        if not breakout_today:
            # If we had a selling climax + base + spring, expect upside breakout
            if scd > 0 and (di - scd) <= 30 and base_active[si, di]:
                direction = 1
            # Spring alone implies upside
            for lookback in range(0, 6):
                if spring_detected[si, di - lookback]:
                    direction = 1
                    break

        # OI rising during base (+2)
        if oi_accumulating[si, di]:
            score += 2

        # OI 5d change > 0 during base (+2)
        if base_active[si, di] and oi_rising_5d[si, di]:
            score += 2

        # Volume surge at breakout (+1)
        if breakout_today and breakout_vol[si, di]:
            score += 1

        # Minimum score threshold
        if score < 3:
            return np.nan, 0

        # Convert to signed score
        if direction == 0:
            return np.nan, 0

        signed_score = score * direction

        # Normalize to [-2, 2] range (max raw score is 14)
        signed_score = np.clip(signed_score / 7.0, -2.0, 2.0)

        return signed_score, direction

    # ==================================================================
    # VARIANTS: Additional scoring functions for comparison
    # ==================================================================
    def score_wyckoff_relaxed(si, di):
        """Relaxed version: lower thresholds, more trades."""
        if di < 21:
            return np.nan, 0

        score = 0.0
        direction = 0

        # Selling climax in last 45 days (+2)
        scd = sc_day[si, di]
        if scd > 0 and (di - scd) <= 45:
            score += 2

        # Base (wider: 8% range, 7+ days)
        # We use the precomputed base_active but also check a relaxed base
        d = di - 1
        c_now = C[si, d]
        if not np.isnan(c_now) and c_now > 0 and di >= 8:
            cw = C[si, max(0, d-14):d+1]
            cvalid = cw[~np.isnan(cw)]
            if len(cvalid) >= 7:
                wlo = np.min(cvalid)
                whi = np.max(cvalid)
                wmid = (wlo + whi) / 2
                if wmid > 0 and (whi - wlo) / wmid <= 0.08:
                    score += 1
                    # Track base bounds for this relaxed base
                    bl_r = wlo
                    bh_r = whi

                    # Spring (relaxed: within 7 days)
                    for lb in range(0, 8):
                        if spring_detected[si, di - lb]:
                            score += 3
                            break

                    # Breakout from relaxed base
                    if c_now > bh_r:
                        score += 4
                        direction = 1
                    elif c_now < bl_r:
                        score += 4
                        direction = -1

        # OI accumulation
        if oi_accumulating[si, di]:
            score += 2

        if oi_rising_5d[si, di]:
            score += 2

        # Volume at breakout
        if breakout_vol[si, di]:
            score += 1

        if score < 2:
            return np.nan, 0

        if direction == 0:
            # Infer from spring or SC
            for lb in range(0, 8):
                if spring_detected[si, di - lb]:
                    direction = 1
                    break
            if direction == 0 and scd > 0 and (di - scd) <= 45:
                direction = 1

        if direction == 0:
            return np.nan, 0

        signed = score * direction
        signed = np.clip(signed / 7.0, -2.0, 2.0)
        return signed, direction

    def score_wyckoff_oi_pure(si, di):
        """Pure OI-focused Wyckoff: requires OI confirmation for all phases."""
        if di < 21:
            return np.nan, 0

        score = 0.0
        direction = 0

        # SC + high OI (panic + new positions = smart money absorbing)
        scd = sc_day[si, di]
        if scd > 0 and (di - scd) <= 30:
            d = di - 1
            oi_sc = OI[si, scd - 1] if scd > 0 else np.nan
            oi_now = OI[si, d]
            if not np.isnan(oi_sc) and not np.isnan(oi_now) and oi_sc > 0:
                # OI increased during/after SC = absorption
                if oi_now >= oi_sc:
                    score += 3  # Enhanced SC score
                else:
                    score += 1

        # Base + OI rising = strong base
        if base_active[si, di] and oi_accumulating[si, di]:
            score += 2
        elif base_active[si, di]:
            score += 0.5

        # Spring + OI rising = true shakeout
        for lb in range(0, 6):
            if spring_detected[si, di - lb]:
                if oi_rising_5d[si, di]:
                    score += 4  # Enhanced spring
                else:
                    score += 2
                break

        # Breakout + OI rising = conviction
        if breakout_up[si, di]:
            direction = 1
            if oi_rising_5d[si, di]:
                score += 5  # Conviction breakout
            else:
                score += 3
        elif breakout_dn[si, di]:
            direction = -1
            if oi_rising_5d[si, di]:
                score += 5
            else:
                score += 3

        if score < 3:
            return np.nan, 0
        if direction == 0:
            for lb in range(0, 6):
                if spring_detected[si, di - lb]:
                    direction = 1
                    break
        if direction == 0:
            return np.nan, 0

        signed = score * direction
        signed = np.clip(signed / 10.0, -2.0, 2.0)
        return signed, direction

    def score_spring_only(si, di):
        """Trade springs only — the highest-conviction Wyckoff signal."""
        if di < 21:
            return np.nan, 0

        score = 0.0
        direction = 0

        # Must have spring in last 3 days
        has_spring = False
        for lb in range(0, 4):
            if spring_detected[si, di - lb]:
                has_spring = True
                break

        if not has_spring:
            return np.nan, 0

        # Base context
        if base_active[si, di]:
            score += 2

        # Selling climax before spring
        scd = sc_day[si, di]
        if scd > 0 and (di - scd) <= 30:
            score += 2

        # OI confirmation
        if oi_rising_5d[si, di]:
            score += 3

        if oi_accumulating[si, di]:
            score += 2

        # Volume on spring day
        d = di - 1
        v_now = V[si, d]
        va = vol_avg20[si, di]
        if not np.isnan(v_now) and not np.isnan(va) and va > 0:
            if v_now > 1.5 * va:
                score += 1

        # Springs are always bullish (shakeout before markup)
        direction = 1

        if score < 3:
            return np.nan, 0

        signed = score * direction
        signed = np.clip(signed / 10.0, -2.0, 2.0)
        return signed, direction

    # ==================================================================
    # BACKTEST ENGINE (single position, P1 concentrated)
    # ==================================================================
    def run_backtest(score_fn, name, hold_min=3, hold_max=7,
                     allow_short=True, rotation_threshold=1.5):
        """Run backtest with Wyckoff scoring.

        Args:
            score_fn: function(si, di) -> (signed_score, direction)
            name: strategy name
            hold_min: minimum hold days before exit signals checked
            hold_max: maximum hold days (time exit)
            allow_short: whether to allow short trades
            rotation_threshold: score multiple for rotation exit
        """
        cash = float(CASH0)
        trades = []
        pos = None
        last_exit = {}

        for di in range(MIN_TRAIN, ND):
            year = dates[di].year

            # --- Manage existing position ---
            if pos is not None:
                c = C[pos['si'], di]
                if np.isnan(c) or c <= 0:
                    c = pos['entry']
                mult = MULT.get(pos['sym'], DEF_MULT)
                mkt_val = c * mult * pos['lots']
                pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
                pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100 \
                    if pos['entry'] > 0 else 0
                days_held = di - pos['entry_di']

                exit_reason = None

                # 1. Time exit
                if days_held >= hold_max:
                    exit_reason = 'time'

                # 2. Signal flip (only after hold_min)
                if exit_reason is None and days_held >= hold_min:
                    cur_score, cur_dir = score_fn(pos['si'], di)
                    if not np.isnan(cur_score):
                        if pos['dir'] == 1 and cur_dir == -1 and cur_score < -0.1:
                            exit_reason = 'signal_flip'
                        elif pos['dir'] == -1 and cur_dir == 1 and cur_score > 0.1:
                            exit_reason = 'signal_flip'

                # 3. Rotation: better candidate available
                if exit_reason is None and days_held >= hold_min:
                    best_si, best_dir, best_sc = -1, 0, 0.0
                    for sj in range(NS):
                        sc, sd = score_fn(sj, di)
                        if np.isnan(sc): continue
                        abs_sc = abs(sc)
                        if abs_sc > best_sc:
                            best_sc = abs_sc
                            best_si = sj
                            best_dir = sd
                    cur_sc = abs(score_fn(pos['si'], di)[0]) \
                        if not np.isnan(score_fn(pos['si'], di)[0]) else 0
                    if best_sc > cur_sc * rotation_threshold + 0.05 and best_si != pos['si']:
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

            # --- Open new position ---
            if pos is None:
                best_si, best_dir, best_sc = -1, 0, 0.0
                for si in range(NS):
                    sc, sd = score_fn(si, di)
                    if np.isnan(sc): continue
                    sym = syms[si]
                    # Cooldown: don't re-enter same symbol same day
                    if sym in last_exit and di - last_exit[sym] < 1:
                        continue
                    abs_sc = abs(sc)
                    if abs_sc > best_sc:
                        best_sc = abs_sc
                        best_si = si
                        best_dir = sd

                if best_si >= 0 and best_sc > 0 and best_dir != 0:
                    # Respect allow_short
                    if not allow_short and best_dir == -1:
                        pass  # skip this, try to find a long
                    else:
                        c = C[best_si, di]
                        if np.isnan(c) or c <= 0:
                            pass
                        else:
                            sym = syms[best_si]
                            mult = MULT.get(sym, DEF_MULT)
                            notional = c * mult
                            if notional <= 0:
                                pass
                            else:
                                lots = int(cash / notional)
                                if lots <= 0:
                                    pass
                                else:
                                    cost_in = notional * lots * (1 + COMM)
                                    if cost_in > cash:
                                        lots = int(cash / (notional * (1 + COMM)))
                                    if lots > 0:
                                        cost_in = notional * lots * (1 + COMM)
                                        cash -= cost_in
                                        pos = {
                                            'si': best_si,
                                            'entry': c,
                                            'entry_di': di,
                                            'lots': lots,
                                            'dir': best_dir,
                                            'sym': sym,
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
                'pnl_pct': pnl / (pos['entry'] * mult * pos['lots']) * 100,
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

        # --- Compute stats ---
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
        avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_abs'] > 0]) \
            if nw > 0 else 0
        avg_loss = np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_abs'] <= 0]) \
            if nw < len(trades) else 0

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

        # Direction breakdown
        long_trades = [t for t in trades if t['dir'] == 1]
        short_trades = [t for t in trades if t['dir'] == -1]
        long_wr = sum(1 for t in long_trades if t['pnl_abs'] > 0) / max(len(long_trades), 1) * 100
        short_wr = sum(1 for t in short_trades if t['pnl_abs'] > 0) / max(len(short_trades), 1) * 100
        long_pnl = sum(t['pnl_pct'] for t in long_trades)
        short_pnl = sum(t['pnl_pct'] for t in short_trades)

        return {
            'name': name,
            'ann': round(ann, 1),
            'n': len(trades),
            'wr': round(wr, 1),
            'dd': round(max_dd, 1),
            'avg_pnl': round(avg_pnl, 3),
            'avg_days': round(avg_days, 1),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'cash': round(cash, 0),
            'reasons': reasons,
            'yearly': year_stats,
            'long_n': len(long_trades),
            'long_wr': round(long_wr, 1),
            'long_pnl': round(long_pnl, 1),
            'short_n': len(short_trades),
            'short_wr': round(short_wr, 1),
            'short_pnl': round(short_pnl, 1),
        }

    # ==================================================================
    # RUN ALL CONFIGS
    # ==================================================================
    results = []
    configs = [
        # (score_fn, name, hold_min, hold_max, allow_short, rotation_threshold)

        # --- Core Wyckoff strategy ---
        (score_wyckoff, "WYCKOFF_H3-7", 3, 7, True, 1.5),
        (score_wyckoff, "WYCKOFF_H3-5", 3, 5, True, 1.5),
        (score_wyckoff, "WYCKOFF_H5-7", 5, 7, True, 1.5),
        (score_wyckoff, "WYCKOFF_H3-7_NOROT", 3, 7, True, 99.0),   # no rotation
        (score_wyckoff, "WYCKOFF_LONG_ONLY", 3, 7, False, 1.5),

        # --- Relaxed Wyckoff (more trades) ---
        (score_wyckoff_relaxed, "WYCK_RELAX_H3-7", 3, 7, True, 1.5),
        (score_wyckoff_relaxed, "WYCK_RELAX_H3-5", 3, 5, True, 1.5),
        (score_wyckoff_relaxed, "WYCK_RELAX_H5-7", 5, 7, True, 1.5),
        (score_wyckoff_relaxed, "WYCK_RELAX_LONG", 3, 7, False, 1.5),

        # --- OI-focused Wyckoff ---
        (score_wyckoff_oi_pure, "WYCK_OI_H3-7", 3, 7, True, 1.5),
        (score_wyckoff_oi_pure, "WYCK_OI_H3-5", 3, 5, True, 1.5),
        (score_wyckoff_oi_pure, "WYCK_OI_H5-7", 5, 7, True, 1.5),
        (score_wyckoff_oi_pure, "WYCK_OI_LONG", 3, 7, False, 1.5),

        # --- Spring-only (highest conviction) ---
        (score_spring_only, "SPRING_ONLY_H3-7", 3, 7, True, 1.5),
        (score_spring_only, "SPRING_ONLY_H3-5", 3, 5, True, 1.5),
        (score_spring_only, "SPRING_ONLY_H5-7", 5, 7, True, 1.5),
        (score_spring_only, "SPRING_ONLY_LONG", 3, 7, False, 1.5),
    ]

    for fn, name, hmin, hmax, ashort, rot_thresh in configs:
        r = run_backtest(fn, name, hold_min=hmin, hold_max=hmax,
                         allow_short=ashort, rotation_threshold=rot_thresh)
        if r:
            results.append(r)
            print(f"  {r['name']:35s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"N {r['n']:4d} | DD {r['dd']:6.1f}% | AvgW {r['avg_win']:+.2f}% | "
                  f"AvgL {r['avg_loss']:.2f}% | AvgD {r['avg_days']:.1f}")
            print(f"  {'':35s} | Long: {r['long_n']}t WR {r['long_wr']:.0f}% "
                  f"PnL {r['long_pnl']:+.0f}% | Short: {r['short_n']}t "
                  f"WR {r['short_wr']:.0f}% PnL {r['short_pnl']:+.0f}%")
            # Exit breakdown
            parts = []
            for reason, stats in sorted(r['reasons'].items()):
                rwr = stats['w'] / stats['n'] * 100 if stats['n'] > 0 else 0
                parts.append(f"{reason}:{stats['n']}({rwr:.0f}%)pnl={stats['pnl']:+.0f}%")
            print(f"  {'':35s} | {' | '.join(parts)}")
        else:
            print(f"  {name:35s} | TOO FEW TRADES")

    # ==================================================================
    # YEARLY BREAKDOWN FOR TOP RESULTS
    # ==================================================================
    results.sort(key=lambda x: -x['ann'])

    print(f"\n{'=' * 110}")
    print("YEARLY BREAKDOWN (Top 5)")
    print(f"{'=' * 110}")
    for r in results[:5]:
        print(f"\n  {r['name']} — Ann {r['ann']:+.1f}% WR {r['wr']:.1f}% "
              f"N {r['n']} DD {r['dd']:.1f}%")
        for y in sorted(r['yearly'].keys()):
            ys = r['yearly'][y]
            ywr = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
            print(f"    {y}: {ys['n']:3d}t  WR {ywr:5.1f}%  PnL {ys['pnl']:+.1f}%")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print(f"\n{'=' * 110}")
    print("SUMMARY — All configs ranked by annualized return")
    print(f"{'=' * 110}")
    print(f"  {'Strategy':35s} | {'Ann':>7s} | {'WR':>5s} | {'N':>4s} | "
          f"{'DD':>6s} | {'AvgW':>7s} | {'AvgL':>6s} | {'AvgD':>5s} | "
          f"{'Long':>8s} | {'Short':>8s}")
    print(f"  {'-' * 105}")
    for r in results:
        print(f"  {r['name']:35s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:4d} | {r['dd']:6.1f}% | {r['avg_win']:+7.2f}% | "
              f"{r['avg_loss']:6.2f}% | {r['avg_days']:5.1f} | "
              f"L{r['long_n']:3d}/{r['long_wr']:.0f}% | "
              f"S{r['short_n']:3d}/{r['short_wr']:.0f}%")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
