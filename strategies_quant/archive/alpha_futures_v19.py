"""
Alpha Futures V19 — Optimized 2-Day Momentum Rotation
======================================================
CORE IDEA:
  From prior testing, MOM5 (5-day momentum ranking) with 1-day hold achieved +88.7% annual.
  User rejected 1-day hold as too short. This strategy optimizes for 2-DAY HOLD.

  The math: 120 trades/year x 1.6% per trade = 600% annual
  To get 1.6% per 2-day trade with tight stops:
    - Use VERY TIGHT stop loss (1.0-1.5%) to minimize losses
    - Use trailing take-profit to let winners run to 2-4%
    - This creates natural asymmetry: losses small (1%), wins larger (2-4%)

STRATEGY VARIANTS (all P1 concentrated, 2-day hold):
  1. MOM5_TIGHT:    Pure 5-day momentum ranking, tight 1% stop, trail at 1.5 ATR
  2. MOM3_TIGHT:    3-day momentum, same stops
  3. MOM_OI_TIGHT:  Score = mom5 * (1 + 0.5*oi_mom5), tight stops
  4. MOM_VDP_TIGHT: Score = mom5 * (1 + 0.5*vdp_direction), tight stops
  5. MOM_ALL_TIGHT: Score = mom5 + 0.3*oi_mom5 + 0.2*vdp + 0.1*vol_ratio
  6. HIGHVOL_MOM:   Only trade commodities with ATR/Price > 2% (high vol = bigger moves)
  7. TOP3_PORT:     Hold top 3 momentum commodities, equal weight, rebalance every 2 days
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


if __name__ == '__main__':
    print("=" * 100, flush=True)
    print("  Alpha Futures V19 — Optimized 2-Day Momentum Rotation", flush=True)
    print("  Target: 120 trades/yr x 1.6%/trade = ~600% annual", flush=True)
    print("=" * 100, flush=True)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    print("\n  Precomputing factors...", flush=True)
    t0 = time.time()

    # ------------------------------------------------------------------
    # Pre-compute factors [si, di] using data up to di-1
    # ------------------------------------------------------------------
    mom3     = np.full((NS, ND), np.nan)   # 3-day momentum
    mom5     = np.full((NS, ND), np.nan)   # 5-day momentum
    oi_mom5  = np.full((NS, ND), np.nan)   # OI 5-day change
    vdp_ema  = np.full((NS, ND), np.nan)   # EMA(VDP, 15-day)
    vol_ratio = np.full((NS, ND), np.nan)  # volume ratio
    atr10    = np.full((NS, ND), np.nan)   # ATR(10)
    atr_pct  = np.full((NS, ND), np.nan)   # ATR/Price for high-vol filter

    for si in range(NS):
        vdp_e = 0.0
        for di in range(25, ND):
            d = di - 1  # yesterday's index
            c_now = C[si, d]
            if np.isnan(c_now) or c_now <= 0:
                continue

            # --- 3-day momentum: (C[di-1] - C[di-4]) / C[di-4] ---
            c_m3 = C[si, d - 3]
            if not np.isnan(c_m3) and c_m3 > 0:
                mom3[si, di] = (c_now - c_m3) / c_m3

            # --- 5-day momentum: (C[di-1] - C[di-6]) / C[di-6] ---
            c_m5 = C[si, d - 5]
            if not np.isnan(c_m5) and c_m5 > 0:
                mom5[si, di] = (c_now - c_m5) / c_m5

            # --- OI 5-day momentum: (OI[di-1] - OI[di-6]) / OI[di-6] ---
            oi_now = OI[si, d]
            if not np.isnan(oi_now) and oi_now > 0:
                oi_5 = OI[si, d - 5]
                if not np.isnan(oi_5) and oi_5 > 0:
                    oi_mom5[si, di] = (oi_now - oi_5) / oi_5

            # --- VDP EMA (15-day) ---
            #   VDP = V * (2*C - H - L) / (H - L)
            hl = H[si, d] - L[si, d]
            if not np.isnan(hl) and hl > 0:
                cd = C[si, d]; hd = H[si, d]; ld = L[si, d]; vd = V[si, d]
                if not any(np.isnan([cd, hd, ld, vd])):
                    vdp_val = vd * (2 * cd - hd - ld) / hl
                    alpha = 2.0 / 16  # 15-day EMA
                    vdp_e = alpha * vdp_val + (1 - alpha) * vdp_e
                    vdp_ema[si, di] = vdp_e

            # --- Volume ratio: V[di-1] / mean(V[di-21:di-1]) ---
            v_now = V[si, d]
            if not np.isnan(v_now) and v_now > 0:
                v20 = V[si, max(0, d - 20):d]
                v20v = v20[~np.isnan(v20)]
                if len(v20v) >= 10:
                    vol_ratio[si, di] = v_now / np.mean(v20v)

            # --- ATR(10) ---
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
                atr_pct[si, di] = atr10[si, di] / c_now  # ATR / Price

    print(f"  Factors done ({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------------
    # Scoring functions — one per variant
    # ------------------------------------------------------------------

    def score_mom5(si, di):
        """Variant 1: Pure 5-day momentum."""
        return mom5[si, di]

    def score_mom3(si, di):
        """Variant 2: Pure 3-day momentum."""
        return mom3[si, di]

    def score_mom_oi(si, di):
        """Variant 3: mom5 * (1 + 0.5 * oi_mom5)."""
        m = mom5[si, di]
        if np.isnan(m):
            return np.nan
        o = oi_mom5[si, di]
        if np.isnan(o):
            return m
        return m * (1 + 0.5 * o)

    def score_mom_vdp(si, di):
        """Variant 4: mom5 * (1 + 0.5 * sign(vdp_ema))."""
        m = mom5[si, di]
        if np.isnan(m):
            return np.nan
        v = vdp_ema[si, di]
        if np.isnan(v):
            return m
        return m * (1 + 0.5 * np.sign(v))

    def score_mom_all(si, di):
        """Variant 5: mom5 + 0.3*oi_mom5 + 0.2*vdp_dir + 0.1*vol_ratio."""
        m = mom5[si, di]
        if np.isnan(m):
            return np.nan
        sc = m
        o = oi_mom5[si, di]
        if not np.isnan(o):
            sc += 0.3 * o
        v = vdp_ema[si, di]
        if not np.isnan(v):
            # normalise vdp direction: sign * min(abs/5e6, 1)
            vdp_dir = np.sign(v) * min(abs(v) / 5e6, 1.0)
            sc += 0.2 * vdp_dir
        vr = vol_ratio[si, di]
        if not np.isnan(vr):
            sc += 0.1 * (vr - 1.0)
        return sc

    def score_highvol_mom(si, di):
        """Variant 6: Only allow commodities with ATR/Price > 2%, then use mom5."""
        ap = atr_pct[si, di]
        if np.isnan(ap) or ap < 0.02:
            return np.nan
        return mom5[si, di]

    # Variant 7 (TOP3_PORT) is handled specially in the backtest loop.

    SCORERS = {
        'MOM5_TIGHT':    score_mom5,
        'MOM3_TIGHT':    score_mom3,
        'MOM_OI_TIGHT':  score_mom_oi,
        'MOM_VDP_TIGHT': score_mom_vdp,
        'MOM_ALL_TIGHT': score_mom_all,
        'HIGHVOL_MOM':   score_highvol_mom,
    }

    # ------------------------------------------------------------------
    # Backtest engine — concentrated P1, 2-day hold
    # ------------------------------------------------------------------

    def run_backtest(variant_name, score_fn, stop_loss=0.01, trail_atr=1.5,
                     hold_max=2, allow_short=True, min_score=0.02,
                     top3=False):
        """
        Run a single parameter combination.

        When top3=True, holds top 3 ranked commodities equal-weight, rebalancing
        every hold_max days (Variant 7).  Otherwise concentrated P1.
        """
        cash = float(CASH0)
        trades = []

        if top3:
            # --- TOP3_PORT path ---
            positions = {}  # sym_index -> {'entry', 'lots', 'dir', 'entry_di', 'atr'}
            rebalance_day = MIN_TRAIN  # next rebalance day

            for di in range(MIN_TRAIN, ND):
                year = dates[di].year

                # Rebalance?
                if di >= rebalance_day:
                    # Close all current positions first
                    for si, pos in positions.items():
                        c = C[si, di]
                        if np.isnan(c) or c <= 0:
                            c = pos['entry']
                        mult = MULT.get(syms[si], DEF_MULT)
                        pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
                        pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] > 0 else 0
                        cash += c * mult * pos['lots'] * (1 - COMM)
                        days_held = di - pos['entry_di']
                        trades.append({
                            'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                            'days': days_held, 'di': di, 'year': year,
                            'sym': syms[si], 'dir': pos['dir'],
                            'reason': 'rebalance'
                        })
                    positions.clear()

                    # Rank all commodities
                    scored = []
                    for si in range(NS):
                        sc = score_fn(si, di)
                        if np.isnan(sc):
                            continue
                        scored.append((si, sc))
                    if not scored:
                        rebalance_day = di + 1
                        continue

                    # Sort descending for long, ascending for short
                    scored.sort(key=lambda x: -x[1])
                    candidates = []
                    # Long top 3
                    for si, sc in scored:
                        if sc >= min_score and len(candidates) < 3:
                            candidates.append((si, 1, sc))
                    # Short bottom 3 (if allow_short)
                    if allow_short:
                        scored_rev = sorted(scored, key=lambda x: x[1])
                        for si, sc in scored_rev:
                            if sc <= -min_score and not any(s == si for s, _, _ in candidates):
                                if len(candidates) < 3:
                                    candidates.append((si, -1, sc))

                    if not candidates:
                        rebalance_day = di + 1
                        continue

                    alloc_each = cash / len(candidates)
                    for si, direction, sc in candidates:
                        c = C[si, di]
                        if np.isnan(c) or c <= 0:
                            continue
                        sym = syms[si]
                        mult = MULT.get(sym, DEF_MULT)
                        notional = c * mult
                        if notional <= 0:
                            continue
                        lots = int(alloc_each / notional)
                        if lots <= 0:
                            continue
                        cost = notional * lots * (1 + COMM)
                        if cost > cash:
                            continue
                        cash -= cost
                        # ATR
                        atr_val = atr10[si, di] if not np.isnan(atr10[si, di]) else 0
                        positions[si] = {
                            'entry': c, 'lots': lots, 'dir': direction,
                            'entry_di': di, 'atr': atr_val
                        }

                    rebalance_day = di + hold_max
                else:
                    # Not rebalancing — check stop / trail for each position
                    to_close = []
                    for si, pos in positions.items():
                        c = C[si, di]
                        if np.isnan(c) or c <= 0:
                            c = pos['entry']
                        mult = MULT.get(syms[si], DEF_MULT)
                        pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
                        pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] > 0 else 0
                        days_held = di - pos['entry_di']
                        exit_reason = None

                        # Stop loss
                        if pnl_pct / 100 < -stop_loss:
                            exit_reason = 'stop'
                        # Trail
                        if exit_reason is None and trail_atr > 0 and days_held >= 1:
                            atr_v = pos.get('atr', 0)
                            if atr_v > 0:
                                if pos['dir'] == 1:
                                    trail_price = c - trail_atr * atr_v
                                    if trail_price > pos.get('trail_h', pos['entry']):
                                        pos['trail_h'] = trail_price
                                    if c < pos.get('trail_h', pos['entry'] - trail_atr * atr_v):
                                        exit_reason = 'trail'
                                else:
                                    trail_price = c + trail_atr * atr_v
                                    if trail_price < pos.get('trail_l', pos['entry']):
                                        pos['trail_l'] = trail_price
                                    if c > pos.get('trail_l', pos['entry'] + trail_atr * atr_v):
                                        exit_reason = 'trail'

                        if exit_reason:
                            cash += c * mult * pos['lots'] * (1 - COMM)
                            trades.append({
                                'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                                'days': days_held, 'di': di, 'year': year,
                                'sym': syms[si], 'dir': pos['dir'],
                                'reason': exit_reason
                            })
                            to_close.append(si)
                    for si in to_close:
                        del positions[si]

            # Close remaining
            for si, pos in positions.items():
                c = C[si, ND - 1]
                if np.isnan(c) or c <= 0:
                    c = pos['entry']
                mult = MULT.get(syms[si], DEF_MULT)
                pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
                cash += c * mult * pos['lots'] * (1 - COMM)
                trades.append({
                    'pnl_pct': pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] > 0 else 0,
                    'pnl_abs': pnl,
                    'days': ND - 1 - pos['entry_di'],
                    'di': ND - 1, 'year': dates[ND - 1].year,
                    'sym': syms[si], 'dir': pos['dir'], 'reason': 'end'
                })

        else:
            # --- Concentrated P1 path ---
            pos = None  # single position dict
            for di in range(MIN_TRAIN, ND):
                year = dates[di].year

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

                    # Tight stop loss
                    if pnl_pct / 100 < -stop_loss:
                        exit_reason = 'stop'

                    # Trailing stop (ATR-based)
                    if exit_reason is None and trail_atr > 0 and days_held >= 1:
                        atr_v = pos.get('atr', 0)
                        if atr_v > 0:
                            if pos['dir'] == 1:
                                new_trail = c - trail_atr * atr_v
                                old_trail = pos.get('trail_price', pos['entry'] - trail_atr * atr_v)
                                if new_trail > old_trail:
                                    pos['trail_price'] = new_trail
                                tp = pos.get('trail_price', old_trail)
                                if c < tp:
                                    exit_reason = 'trail'
                            else:
                                new_trail = c + trail_atr * atr_v
                                old_trail = pos.get('trail_price', pos['entry'] + trail_atr * atr_v)
                                if new_trail < old_trail:
                                    pos['trail_price'] = new_trail
                                tp = pos.get('trail_price', old_trail)
                                if c > tp:
                                    exit_reason = 'trail'

                    # Time exit
                    if exit_reason is None and days_held >= hold_max:
                        exit_reason = 'time'

                    # Signal flip exit (strong opposing signal)
                    if exit_reason is None and days_held >= 1:
                        cur_score = score_fn(pos['si'], di)
                        if not np.isnan(cur_score):
                            if pos['dir'] == 1 and cur_score < -min_score:
                                exit_reason = 'flip'
                            elif pos['dir'] == -1 and cur_score > min_score:
                                exit_reason = 'flip'

                    if exit_reason:
                        cash += mkt_val * (1 - COMM)
                        trades.append({
                            'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                            'days': days_held, 'di': di, 'year': year,
                            'sym': pos['sym'], 'dir': pos['dir'],
                            'reason': exit_reason
                        })
                        pos = None

                # === ENTRY ===
                if pos is None:
                    best_si, best_dir, best_sc = -1, 0, 0.0
                    for si in range(NS):
                        sc = score_fn(si, di)
                        if np.isnan(sc):
                            continue
                        if sc > best_sc:
                            best_sc = sc; best_si = si; best_dir = 1
                        if allow_short and -sc > best_sc:
                            best_sc = -sc; best_si = si; best_dir = -1

                    if best_si >= 0 and best_sc >= min_score:
                        c = C[best_si, di]
                        if np.isnan(c) or c <= 0:
                            pass  # skip
                        else:
                            sym = syms[best_si]
                            mult = MULT.get(sym, DEF_MULT)
                            notional = c * mult
                            if notional > 0:
                                lots = int(cash * 0.95 / notional)
                                if lots > 0:
                                    cost = notional * lots * (1 + COMM)
                                    if cost <= cash:
                                        atr_val = atr10[best_si, di] if not np.isnan(atr10[best_si, di]) else 0
                                        cash -= cost
                                        trail_price = (c - trail_atr * atr_val) if best_dir == 1 else (c + trail_atr * atr_val)
                                        pos = {
                                            'si': best_si, 'entry': c, 'entry_di': di,
                                            'lots': lots, 'dir': best_dir, 'sym': sym,
                                            'atr': atr_val, 'trail_price': trail_price
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
                    'pnl_pct': pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] > 0 else 0,
                    'pnl_abs': pnl,
                    'days': ND - 1 - pos['entry_di'],
                    'di': ND - 1, 'year': dates[ND - 1].year,
                    'sym': pos['sym'], 'dir': pos['dir'], 'reason': 'end'
                })

        # === STATS ===
        if len(trades) < 10:
            return None

        # Max drawdown via equity curve
        equity = float(CASH0); peak = float(CASH0); max_dd = 0
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

        reasons = {}
        for t in trades:
            r = t['reason']
            if r not in reasons:
                reasons[r] = {'n': 0, 'w': 0, 'pnl': 0.0}
            reasons[r]['n'] += 1
            if t['pnl_abs'] > 0:
                reasons[r]['w'] += 1
            reasons[r]['pnl'] += t['pnl_pct']

        short_tag = 'S' if allow_short else 'L'
        return {
            'name': f"{variant_name}_SL{stop_loss:.3f}_TR{trail_atr:.1f}_H{hold_max}_{short_tag}_MS{min_score:.2f}",
            'ann': round(ann, 1), 'n': len(trades),
            'wr': round(wr, 1), 'dd': round(max_dd, 1),
            'avg_pnl': round(avg_pnl, 3), 'avg_days': round(avg_days, 1),
            'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
            'final': round(cash, 0), 'years': year_stats, 'reasons': reasons,
            'variant': variant_name
        }

    # ------------------------------------------------------------------
    # Build parameter grid
    # ------------------------------------------------------------------
    configs = []

    for variant_name, score_fn in SCORERS.items():
        for stop_loss in [0.01, 0.015, 0.02, 0.03]:
            for trail_atr in [1.0, 1.5, 2.0, 0.0]:
                for hold_max in [2, 3]:
                    for allow_short in [True, False]:
                        for min_score in [0.01, 0.02, 0.03]:
                            configs.append((variant_name, score_fn, stop_loss,
                                            trail_atr, hold_max, allow_short,
                                            min_score, False))

    # Variant 7: TOP3_PORT
    for stop_loss in [0.01, 0.015, 0.02, 0.03]:
        for trail_atr in [1.0, 1.5, 2.0, 0.0]:
            for hold_max in [2, 3]:
                for allow_short in [True, False]:
                    for min_score in [0.01, 0.02, 0.03]:
                        configs.append(('TOP3_PORT', score_mom5, stop_loss,
                                        trail_atr, hold_max, allow_short,
                                        min_score, True))

    total = len(configs)
    print(f"  Total configs: {total}", flush=True)

    # ------------------------------------------------------------------
    # Run all configs
    # ------------------------------------------------------------------
    results = []
    t1 = time.time()
    for ci, (vname, sfn, sl, tr, hm, short, ms, top3) in enumerate(configs):
        if ci % 200 == 0:
            print(f"  Config {ci}/{total} ({len(results)} profitable, {time.time()-t1:.0f}s)", flush=True)
        r = run_backtest(vname, sfn, stop_loss=sl, trail_atr=tr,
                         hold_max=hm, allow_short=short, min_score=ms,
                         top3=top3)
        if r and r['ann'] > 5:
            results.append(r)

    print(f"\n  Done ({time.time() - t1:.0f}s, {len(results)} configs > 5%)", flush=True)
    results.sort(key=lambda x: -x['ann'])

    # ------------------------------------------------------------------
    # Print TOP 30
    # ------------------------------------------------------------------
    print(f"\n{'=' * 110}", flush=True)
    print(f"  TOP 30 RESULTS (sorted by annualised return)", flush=True)
    print(f"  {'Strategy':65s} | {'Ann':>8s} {'WR':>5s} {'N':>4s} {'DD':>6s} "
          f"{'AvgW':>6s} {'AvgL':>6s} {'AvgD':>5s} {'AvgP':>6s}", flush=True)
    print(f"  {'-' * 105}", flush=True)
    for r in results[:30]:
        print(f"  {r['name']:65s} | {r['ann']:+8.1f}% {r['wr']:5.1f}% {r['n']:4d} "
              f"{r['dd']:6.1f}% {r['avg_win']:+6.2f}% {r['avg_loss']:6.2f}% "
              f"{r['avg_days']:5.1f}d {r['avg_pnl']:+6.3f}%", flush=True)

    # ------------------------------------------------------------------
    # Yearly breakdown for TOP 5
    # ------------------------------------------------------------------
    for i, r in enumerate(results[:5]):
        print(f"\n  #{i + 1}: {r['name']}", flush=True)
        print(f"       Ann={r['ann']:+.1f}%  WR={r['wr']:.0f}%  DD={r['dd']:.1f}%  "
              f"AvgWin={r['avg_win']:+.2f}%  AvgLoss={r['avg_loss']:.2f}%  "
              f"AvgDays={r['avg_days']:.1f}", flush=True)

        print(f"       Exit reasons:", flush=True)
        for reason, s in sorted(r['reasons'].items(), key=lambda x: -x[1]['n']):
            rwr = s['w'] / max(s['n'], 1) * 100
            print(f"         {reason:12s}: {s['n']:4d}t  WR={rwr:.0f}%  pnl={s['pnl']:+.1f}%", flush=True)

        print(f"       Yearly breakdown:", flush=True)
        for y in sorted(r['years'].keys()):
            s = r['years'][y]
            wr_y = s['w'] / max(s['n'], 1) * 100
            print(f"         {y}: {s['n']:3d}t  WR={wr_y:.0f}%  avg_pnl={s['pnl']/max(s['n'],1):+.2f}%", flush=True)

    # ------------------------------------------------------------------
    # Summary by variant
    # ------------------------------------------------------------------
    print(f"\n  Best per variant:", flush=True)
    best_per_variant = {}
    for r in results:
        v = r['variant']
        if v not in best_per_variant or r['ann'] > best_per_variant[v]['ann']:
            best_per_variant[v] = r
    for vname in list(SCORERS.keys()) + ['TOP3_PORT']:
        if vname in best_per_variant:
            r = best_per_variant[vname]
            print(f"    {vname:20s} -> Ann={r['ann']:+.1f}%  WR={r['wr']:.0f}%  "
                  f"DD={r['dd']:.1f}%  N={r['n']}  AvgWin={r['avg_win']:+.2f}%  AvgLoss={r['avg_loss']:.2f}%", flush=True)

    print(f"\n{'=' * 110}", flush=True)
    print(f"  Target: 600%+ annual, 50%+ WR, tight stops, 2-day hold", flush=True)
    if results and results[0]['ann'] >= 600:
        print(f"  >>> TARGET ACHIEVED: {results[0]['ann']:+.1f}% <<<", flush=True)
    elif results:
        print(f"  Best: {results[0]['ann']:+.1f}% -- gap to 600%: {600 - results[0]['ann']:.0f}%", flush=True)
    print(f"{'=' * 110}", flush=True)
    print(f"\nDone! Total time: {time.time() - t0:.0f}s", flush=True)
