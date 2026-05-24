"""
Alpha Futures V28 — Market Essence Strategy
============================================
After studying ALL 345 strategies, synthesized 5 core market truths:

1. VOL COMPRESSION -> EXPANSION (most universal edge across all asset classes)
   - BB_WIDTH at historical low = imminent breakout
   - ATR fast/slow ratio < 0.7 = coiling
   - Mechanism: low vol -> position buildup -> vol expansion

2. VDP DELTA FLIP (most reliable entry signal)
   - VDP = V * (2C - H - L) / (H - L) approximates order flow
   - Delta flip from neg to pos = institutional buying starts
   - Flip events are discrete, not continuous -> harder to overfit

3. OI + PRICE = MONEY FLOW (futures unique)
   - Rising price + rising OI = new longs entering (strong)
   - Rising price + falling OI = shorts covering (weak)
   - OI extreme -> reversion ("物极必反")

4. TREND QUALITY > TREND DIRECTION
   - KER (efficiency ratio) = |net| / sum(|daily|), measures "clean" trends
   - High quality trends persist; choppy trends don't

5. SIMPLE ENGINE > COMPLEX ENGINE
   - v14b's success: good scoring + rotation, not fancy stops
   - Simple v7c engine + better factors = best approach

DESIGN:
  Phase 1: Identify vol compression (ATR ratio + BB width)
  Phase 2: Wait for VDP delta flip (direction confirmation)
  Phase 3: Score with OI momentum + trend quality
  Phase 4: Cross-sectional ranking, pick top 1-3
  Phase 5: Simple rotation engine from v14b
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
    print("Alpha Futures V28 — Market Essence Strategy")
    print("Vol Compression + VDP Delta Flip + OI Flow + Trend Quality")
    print("=" * 110)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    # ========================================
    # PRECOMPUTE ALL SIGNALS
    # ========================================
    print("[Signals] Computing all signals...", flush=True)
    t0 = time.time()

    # --- 1. ATR (10-day) ---
    atr10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(11, ND):
            trs = []
            for dd in range(di - 10, di):
                hi, lo = H[si, dd], L[si, dd]
                pc = C[si, dd - 1]
                if np.isnan(hi) or np.isnan(lo): continue
                tr = hi - lo
                if not np.isnan(pc):
                    tr = max(tr, abs(hi - pc), abs(lo - pc))
                trs.append(tr)
            if trs:
                atr10[si, di] = np.mean(trs)

    # --- 2. ATR fast (5-day) and slow (30-day) ratio ---
    atr5 = np.full((NS, ND), np.nan)
    atr30 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(31, ND):
            # Fast ATR (5)
            trs5 = []
            for dd in range(di - 5, di):
                hi, lo = H[si, dd], L[si, dd]
                pc = C[si, dd - 1]
                if np.isnan(hi) or np.isnan(lo): continue
                tr = hi - lo
                if not np.isnan(pc): tr = max(tr, abs(hi - pc), abs(lo - pc))
                trs5.append(tr)
            if trs5: atr5[si, di] = np.mean(trs5)
            # Slow ATR (30)
            trs30 = []
            for dd in range(di - 30, di):
                hi, lo = H[si, dd], L[si, dd]
                pc = C[si, dd - 1]
                if np.isnan(hi) or np.isnan(lo): continue
                tr = hi - lo
                if not np.isnan(pc): tr = max(tr, abs(hi - pc), abs(lo - pc))
                trs30.append(tr)
            if trs30: atr30[si, di] = np.mean(trs30)

    # ATR ratio (vol compression when < 0.7)
    atr_ratio = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(31, ND):
            a5 = atr5[si, di]
            a30 = atr30[si, di]
            if not np.isnan(a5) and not np.isnan(a30) and a30 > 0:
                atr_ratio[si, di] = a5 / a30

    # ATR ratio percentile (cross-sectional rank)
    atr_ratio_pct = np.full((NS, ND), np.nan)
    for di in range(31, ND):
        vals = atr_ratio[:, di]
        valid = ~np.isnan(vals)
        if valid.sum() > 5:
            ranked = np.argsort(np.argsort(vals[valid])) / valid.sum()
            atr_ratio_pct[valid, di] = ranked

    # --- 3. BB Width (20-day) ---
    bb_width = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            cs = C[si, di-20:di]
            valid = cs[~np.isnan(cs)]
            if len(valid) < 15: continue
            sma = np.mean(valid)
            std = np.std(valid)
            if sma > 0 and std > 0:
                bb_width[si, di] = (4 * std) / sma  # normalized BB width

    # BB width percentile (historical)
    bb_width_pct = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(60, ND):
            bw_hist = bb_width[si, max(0, di-60):di]
            valid = bw_hist[~np.isnan(bw_hist)]
            if len(valid) < 10: continue
            cur = bb_width[si, di]
            if np.isnan(cur): continue
            bb_width_pct[si, di] = np.sum(valid < cur) / len(valid)

    # --- 4. VDP EMA (15-day) and delta flip ---
    vdp_ema = np.full((NS, ND), np.nan)
    vdp_prev = np.full((NS, ND), np.nan)  # previous VDP for flip detection
    for si in range(NS):
        vdp_e = 0.0
        alpha = 2.0 / 15
        for di in range(1, ND):
            d = di - 1
            cd, hd, ld, vd = C[si, d], H[si, d], L[si, d], V[si, d]
            if any(np.isnan([cd, hd, ld, vd])) or hd == ld: continue
            vdp_val = vd * (2 * cd - hd - ld) / (hd - ld)
            prev_e = vdp_e
            vdp_e = alpha * vdp_val + (1 - alpha) * vdp_e
            vdp_ema[si, di] = vdp_e
            vdp_prev[si, di] = prev_e

    # VDP flip: from negative to positive (bullish) or positive to negative (bearish)
    vdp_flip_bull = np.zeros((NS, ND), dtype=bool)
    vdp_flip_bear = np.zeros((NS, ND), dtype=bool)
    for si in range(NS):
        for di in range(2, ND):
            cur = vdp_ema[si, di]
            prev = vdp_prev[si, di]
            if np.isnan(cur) or np.isnan(prev): continue
            if prev <= 0 and cur > 0: vdp_flip_bull[si, di] = True
            if prev >= 0 and cur < 0: vdp_flip_bear[si, di] = True

    # --- 5. Momentum 5/10 ---
    mom5 = np.full((NS, ND), np.nan)
    mom10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(11, ND):
            d = di - 1
            c_now = C[si, d]
            if np.isnan(c_now) or c_now <= 0: continue
            c5 = C[si, max(0, d - 5)]
            c10 = C[si, max(0, d - 10)]
            if not np.isnan(c5) and c5 > 0:
                mom5[si, di] = (c_now - c5) / c5
            if not np.isnan(c10) and c10 > 0:
                mom10[si, di] = (c_now - c10) / c10

    # --- 6. KER (Kaufman Efficiency Ratio, 10-day) ---
    ker10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(11, ND):
            cs = C[si, di-11:di]
            valid = cs[~np.isnan(cs)]
            if len(valid) < 6: continue
            net = abs(valid[-1] - valid[0])
            path = np.sum(np.abs(np.diff(valid)))
            if path > 0:
                ker10[si, di] = net / path

    # --- 7. OI Momentum 5 ---
    oi_mom5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(6, ND):
            d = di - 1
            oi_now = OI[si, d]
            if np.isnan(oi_now) or oi_now <= 0: continue
            oi5 = OI[si, max(0, d - 5)]
            if not np.isnan(oi5) and oi5 > 0:
                oi_mom5[si, di] = (oi_now - oi5) / oi5

    # --- 8. OI-Price divergence (money flow signal) ---
    # positive: price up + OI up (new longs, strong)
    # negative: price up + OI down (short covering, weak)
    oi_flow = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(2, ND):
            d = di - 1
            pc = C[si, d]
            pp = C[si, max(0, d - 1)]
            oc = OI[si, d]
            op = OI[si, max(0, d - 1)]
            if np.isnan(pc) or np.isnan(pp) or np.isnan(oc) or np.isnan(op): continue
            if pp <= 0 or op <= 0: continue
            price_chg = (pc - pp) / pp
            oi_chg = (oc - op) / op
            # Strong bull: price up + OI up (new longs)
            # Strong bear: price down + OI up (new shorts)
            # Weak bull: price up + OI down (short covering)
            # Weak bear: price down + OI down (long liquidation)
            if price_chg > 0 and oi_chg > 0:
                oi_flow[si, di] = abs(price_chg) * abs(oi_chg)  # strong bull
            elif price_chg < 0 and oi_chg > 0:
                oi_flow[si, di] = -abs(price_chg) * abs(oi_chg)  # strong bear
            elif price_chg > 0 and oi_chg < 0:
                oi_flow[si, di] = -abs(price_chg) * abs(oi_chg) * 0.5  # weak bull
            elif price_chg < 0 and oi_chg < 0:
                oi_flow[si, di] = abs(price_chg) * abs(oi_chg) * 0.5  # weak bear

    # OI flow EMA (10-day)
    oi_flow_ema = np.full((NS, ND), np.nan)
    for si in range(NS):
        flow_e = 0.0
        for di in range(1, ND):
            fv = oi_flow[si, di]
            if np.isnan(fv): continue
            alpha = 2.0 / 10
            flow_e = alpha * fv + (1 - alpha) * flow_e
            oi_flow_ema[si, di] = flow_e

    # --- 9. Body ratio (candle quality) ---
    body_ratio = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            d = di - 1
            o, c, h, l = O[si, d], C[si, d], H[si, d], L[si, d]
            if any(np.isnan([o, c, h, l])): continue
            hl = h - l
            if hl <= 0: continue
            body_ratio[si, di] = abs(c - o) / hl

    # --- 10. RSI 14 ---
    rsi14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        gains, losses = [], []
        for di in range(2, ND):
            d = di - 1
            c0 = C[si, d]
            c1 = C[si, d - 1]
            if np.isnan(c0) or np.isnan(c1): continue
            chg = c0 - c1
            gains.append(max(chg, 0))
            losses.append(max(-chg, 0))
            if len(gains) >= 14:
                avg_g = np.mean(gains[-14:])
                avg_l = np.mean(losses[-14:])
                if avg_l > 0:
                    rsi14[si, di] = 100 - 100 / (1 + avg_g / avg_l)
                else:
                    rsi14[si, di] = 100

    # --- 11. Volume surge (relative to 20-day avg) ---
    vol_surge = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            vs = V[si, di-21:di]
            valid = vs[~np.isnan(vs)]
            if len(valid) < 10: continue
            avg = np.mean(valid[:-1])  # exclude current
            cur = valid[-1]
            if avg > 0:
                vol_surge[si, di] = cur / avg

    print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

    # ========================================
    # SCORING FUNCTIONS
    # ========================================

    def score_compression_entry(si, di):
        """Vol compression + VDP flip + OI flow — the core strategy"""
        if di < 31: return np.nan

        # --- Compression signals ---
        ar = atr_ratio[si, di]
        bbp = bb_width_pct[si, di]
        if np.isnan(ar) or np.isnan(bbp): return np.nan

        # Compression score (0-2): ATR ratio low + BB width at low percentile
        comp_score = 0.0
        if ar < 0.7: comp_score += 1.0
        elif ar < 0.85: comp_score += 0.5

        if bbp < 0.2: comp_score += 1.0
        elif bbp < 0.35: comp_score += 0.5

        # --- Direction from momentum + VDP ---
        m5 = mom5[si, di]
        vdp = vdp_ema[si, di]
        if np.isnan(m5): return np.nan

        # Direction: momentum sign
        direction = np.clip(m5 * 8, -1, 1)

        # VDP confirmation
        if not np.isnan(vdp):
            if (direction > 0 and vdp > 0) or (direction < 0 and vdp < 0):
                direction *= 1.3
            else:
                direction *= 0.3

        # VDP flip bonus (biggest edge)
        if direction > 0 and vdp_flip_bull[si, di]:
            direction *= 1.5
        elif direction < 0 and vdp_flip_bear[si, di]:
            direction *= 1.5

        # --- OI flow confirmation ---
        ofe = oi_flow_ema[si, di]
        if not np.isnan(ofe):
            if (direction > 0 and ofe > 0) or (direction < 0 and ofe < 0):
                direction *= 1.2
            else:
                direction *= 0.5

        # --- Trend quality filter ---
        ker = ker10[si, di]
        if not np.isnan(ker):
            if ker < 0.15:  # very choppy, skip
                return np.nan
            elif ker > 0.4:  # clean trend
                direction *= 1.2

        # Final score: direction * (1 + compression bonus)
        score = direction * (1 + comp_score * 0.5)
        return np.clip(score, -2, 2)

    def score_momentum_rotation(si, di):
        """v14b-style scoring: momentum + VDP — proven baseline"""
        if di < 11: return np.nan
        m5 = mom5[si, di]
        vdp = vdp_ema[si, di]
        if np.isnan(m5): return np.nan
        score = np.clip(m5 * 8, -1, 1)
        if not np.isnan(vdp):
            if (m5 > 0 and vdp > 0) or (m5 < 0 and vdp < 0):
                score *= 1.3
            else:
                score *= 0.3
        return score

    def score_oi_flow(si, di):
        """Pure OI flow + momentum — futures unique edge"""
        if di < 11: return np.nan
        m5 = mom5[si, di]
        ofe = oi_flow_ema[si, di]
        oi5 = oi_mom5[si, di]
        if np.isnan(m5): return np.nan

        score = np.clip(m5 * 8, -1, 1)

        # OI flow confirms direction
        if not np.isnan(ofe):
            if (score > 0 and ofe > 0) or (score < 0 and ofe < 0):
                score *= 1.3
            else:
                score *= 0.4

        # OI momentum amplifies
        if not np.isnan(oi5):
            if (score > 0 and oi5 > 0) or (score < 0 and oi5 < 0):
                score *= 1.2
            else:
                score *= 0.5

        return score

    def score_quality_trend(si, di):
        """High KER trends only — trend quality filter"""
        if di < 11: return np.nan
        m5 = mom5[si, di]
        ker = ker10[si, di]
        rsi = rsi14[si, di]
        vdp = vdp_ema[si, di]
        if np.isnan(m5): return np.nan

        # Must have decent trend quality
        if not np.isnan(ker) and ker < 0.2:
            return np.nan

        score = np.clip(m5 * 8, -1, 1)

        # KER bonus for clean trends
        if not np.isnan(ker):
            if ker > 0.5: score *= 1.4
            elif ker > 0.35: score *= 1.2

        # VDP confirmation
        if not np.isnan(vdp):
            if (score > 0 and vdp > 0) or (score < 0 and vdp < 0):
                score *= 1.3
            else:
                score *= 0.3

        # RSI filter: don't buy overbought, don't sell oversold
        if not np.isnan(rsi):
            if score > 0 and rsi > 75: score *= 0.5
            if score < 0 and rsi < 25: score *= 0.5

        return score

    def score_compression_flow(si, di):
        """Compression + OI flow (no VDP) — tests OI edge independently"""
        if di < 31: return np.nan
        ar = atr_ratio[si, di]
        m5 = mom5[si, di]
        ofe = oi_flow_ema[si, di]
        oi5 = oi_mom5[si, di]
        if np.isnan(m5): return np.nan

        # Must be in compression
        if not np.isnan(ar) and ar > 0.85: return np.nan

        score = np.clip(m5 * 8, -1, 1)

        # OI flow
        if not np.isnan(ofe):
            if (score > 0 and ofe > 0) or (score < 0 and ofe < 0):
                score *= 1.5
            else:
                score *= 0.3

        # OI momentum
        if not np.isnan(oi5):
            if (score > 0 and oi5 > 0) or (score < 0 and oi5 < 0):
                score *= 1.3

        return score

    def score_vol_surge_momentum(si, di):
        """Volume surge + momentum — institutional activity detection"""
        if di < 21: return np.nan
        m5 = mom5[si, di]
        vs = vol_surge[si, di]
        vdp = vdp_ema[si, di]
        br = body_ratio[si, di]
        if np.isnan(m5): return np.nan

        score = np.clip(m5 * 8, -1, 1)

        # Volume surge bonus
        if not np.isnan(vs) and vs > 1.5:
            score *= 1.3
            if vs > 2.0: score *= 1.2

        # Strong body (institutional candle)
        if not np.isnan(br) and br > 0.7:
            score *= 1.2

        # VDP confirmation
        if not np.isnan(vdp):
            if (score > 0 and vdp > 0) or (score < 0 and vdp < 0):
                score *= 1.2

        return score

    # ========================================
    # BACKTEST ENGINE (v14b-style with rotation)
    # ========================================
    def run_backtest(score_fn, name, hold_max=3, trail_atr=3.0,
                     stop_loss=0.05, allow_short=True, top_n=1):
        cash = float(CASH0)
        trades = []
        positions = []  # list of dicts for multi-position
        last_exit = {}

        for di in range(MIN_TRAIN, ND):
            year = dates[di].year

            # --- Manage existing positions ---
            for pi, pos in enumerate(positions):
                if pos is None: continue
                c = C[pos['si'], di]
                if np.isnan(c) or c <= 0: c = pos['entry']
                mult = MULT.get(pos['sym'], DEF_MULT)
                mkt_val = c * mult * pos['lots']
                pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
                pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] > 0 else 0
                days_held = di - pos['entry_di']

                exit_reason = None

                # Stop loss
                if pnl_pct / 100 < -stop_loss:
                    exit_reason = 'stop'

                # Trailing stop
                if exit_reason is None and trail_atr > 0:
                    atr = pos.get('atr', 0)
                    if atr > 0:
                        trail_price = pos.get('trail_price', pos['entry'])
                        if pos['dir'] == 1:
                            new_trail = c - trail_atr * atr
                            if new_trail > trail_price: pos['trail_price'] = new_trail
                            if c < trail_price and days_held >= 2: exit_reason = 'trail'
                        else:
                            new_trail = c + trail_atr * atr
                            if new_trail < trail_price: pos['trail_price'] = new_trail
                            if c > trail_price and days_held >= 2: exit_reason = 'trail'

                # Score exit
                if exit_reason is None and days_held >= 2:
                    cur_score = score_fn(pos['si'], di)
                    if not np.isnan(cur_score):
                        if pos['dir'] == 1 and cur_score < -0.02: exit_reason = 'signal_flip'
                        elif pos['dir'] == -1 and cur_score > 0.02: exit_reason = 'signal_flip'

                # Time exit
                if exit_reason is None and days_held >= hold_max:
                    exit_reason = 'time'

                # Rotation (only for single-position mode)
                if exit_reason is None and top_n == 1 and days_held >= 2:
                    best_si, best_dir, best_sc = -1, 0, 0
                    for sj in range(NS):
                        sc = score_fn(sj, di)
                        if np.isnan(sc): continue
                        if sc > best_sc: best_sc = sc; best_si = sj; best_dir = 1
                        if allow_short and -sc > best_sc: best_sc = -sc; best_si = sj; best_dir = -1
                    cur_sc = abs(score_fn(pos['si'], di)) if not np.isnan(score_fn(pos['si'], di)) else 0
                    if best_sc > cur_sc * 1.5 + 0.05 and best_si != pos['si']:
                        exit_reason = 'rotate'

                if exit_reason:
                    cost_out = mkt_val * COMM
                    cash += mkt_val - cost_out
                    trades.append({
                        'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                        'days': days_held, 'di': di, 'year': year,
                        'sym': pos['sym'], 'dir': pos['dir'], 'reason': exit_reason
                    })
                    last_exit[pos['sym']] = di
                    positions[pi] = None

            # Clean up closed positions
            positions = [p for p in positions if p is not None]

            # --- Open new positions ---
            if len(positions) < top_n:
                # Score all symbols
                scored = []
                for si in range(NS):
                    sc = score_fn(si, di)
                    if np.isnan(sc): continue
                    sym = syms[si]
                    if sym in last_exit and di - last_exit[sym] < 1: continue
                    if sc > 0:
                        scored.append((si, 1, sc, sym))
                    if allow_short and -sc > 0:
                        scored.append((si, -1, -sc, sym))

                scored.sort(key=lambda x: -x[2])

                occupied_si = {p['si'] for p in positions}
                opened = 0
                for best_si, best_dir, best_sc, best_sym in scored:
                    if opened >= top_n - len(positions): break
                    if best_si in occupied_si: continue

                    c = C[best_si, di]
                    if np.isnan(c) or c <= 0: continue
                    mult = MULT.get(best_sym, DEF_MULT)
                    notional = c * mult
                    if notional <= 0: continue

                    # Allocate capital per position
                    alloc = cash / max(top_n - len(positions), 1)
                    lots = int(alloc / notional)
                    if lots <= 0: continue
                    cost_in = notional * lots * (1 + COMM)
                    if cost_in > alloc: lots = int(alloc / (notional * (1 + COMM)))
                    if lots <= 0: continue
                    cost_in = notional * lots * (1 + COMM)

                    atr_val = 0
                    trs = []
                    for dd in range(max(1, di - 10), di + 1):
                        hi, lo, pc = H[best_si, dd], L[best_si, dd], C[best_si, dd - 1]
                        if np.isnan(hi) or np.isnan(lo): continue
                        tr = hi - lo
                        if not np.isnan(pc): tr = max(tr, abs(hi - pc), abs(lo - pc))
                        trs.append(tr)
                    if trs: atr_val = np.mean(trs)

                    cash -= cost_in
                    trail_price = c - trail_atr * atr_val if best_dir == 1 else c + trail_atr * atr_val
                    positions.append({
                        'si': best_si, 'entry': c, 'entry_di': di,
                        'lots': lots, 'dir': best_dir, 'sym': best_sym,
                        'atr': atr_val, 'trail_price': trail_price
                    })
                    occupied_si.add(best_si)
                    opened += 1

        # Close remaining positions
        for pos in positions:
            if pos is None: continue
            c = C[pos['si'], ND - 1]
            if np.isnan(c) or c <= 0: c = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
            cash += c * mult * pos['lots'] * (1 - COMM)
            trades.append({
                'pnl_pct': pnl / (pos['entry'] * mult * pos['lots']) * 100,
                'pnl_abs': pnl, 'days': ND - 1 - pos['entry_di'],
                'di': ND - 1, 'year': dates[ND - 1].year,
                'sym': pos['sym'], 'dir': pos['dir'], 'reason': 'end'
            })

        if len(trades) < 10: return None

        equity = float(CASH0)
        peak = float(CASH0)
        max_dd = 0
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
        avg_pnl = np.mean([t['pnl_pct'] for t in trades])
        avg_days = np.mean([t['days'] for t in trades])
        avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_abs'] > 0]) if nw > 0 else 0
        avg_loss = np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_abs'] <= 0]) if nw < len(trades) else 0

        reasons = {}
        for t in trades:
            r = t['reason']
            if r not in reasons: reasons[r] = {'n': 0, 'w': 0, 'pnl': 0.0}
            reasons[r]['n'] += 1
            if t['pnl_abs'] > 0: reasons[r]['w'] += 1
            reasons[r]['pnl'] += t['pnl_pct']

        year_stats = {}
        for t in trades:
            y = t['year']
            if y not in year_stats: year_stats[y] = {'n': 0, 'w': 0, 'pnl': 0.0}
            year_stats[y]['n'] += 1
            if t['pnl_abs'] > 0: year_stats[y]['w'] += 1
            year_stats[y]['pnl'] += t['pnl_pct']

        return {
            'name': name, 'ann': round(ann, 1), 'n': len(trades),
            'wr': round(wr, 1), 'dd': round(max_dd, 1),
            'avg_pnl': round(avg_pnl, 3), 'avg_days': round(avg_days, 1),
            'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
            'cash': round(cash, 0), 'reasons': reasons, 'yearly': year_stats,
        }

    # ========================================
    # RUN ALL CONFIGS
    # ========================================
    results = []
    configs = [
        # (score_fn, name, hold_max, trail_atr, stop_loss, top_n)
        # --- Baseline: v14b reproduction ---
        (score_momentum_rotation, "BASELINE_v14b", 3, 3.0, 0.05, 1),

        # --- Compression Entry ---
        (score_compression_entry, "COMP_VDP_OI_H3", 3, 3.0, 0.05, 1),
        (score_compression_entry, "COMP_VDP_OI_H3_T2", 3, 2.0, 0.05, 1),
        (score_compression_entry, "COMP_VDP_OI_H5", 5, 3.0, 0.05, 1),
        (score_compression_entry, "COMP_VDP_OI_H5_T2", 5, 2.0, 0.05, 1),
        (score_compression_entry, "COMP_VDP_OI_H7", 7, 3.0, 0.05, 1),

        # --- OI Flow ---
        (score_oi_flow, "OI_FLOW_H3", 3, 3.0, 0.05, 1),
        (score_oi_flow, "OI_FLOW_H5", 5, 3.0, 0.05, 1),
        (score_oi_flow, "OI_FLOW_H3_T2", 3, 2.0, 0.05, 1),

        # --- Quality Trend ---
        (score_quality_trend, "QUALITY_H3", 3, 3.0, 0.05, 1),
        (score_quality_trend, "QUALITY_H5", 5, 3.0, 0.05, 1),
        (score_quality_trend, "QUALITY_H5_T2", 5, 2.0, 0.05, 1),

        # --- Compression + OI Flow (no VDP) ---
        (score_compression_flow, "COMP_OI_H3", 3, 3.0, 0.05, 1),
        (score_compression_flow, "COMP_OI_H5", 5, 3.0, 0.05, 1),

        # --- Volume Surge ---
        (score_vol_surge_momentum, "VOL_SURGE_H3", 3, 3.0, 0.05, 1),
        (score_vol_surge_momentum, "VOL_SURGE_H5", 5, 3.0, 0.05, 1),

        # --- Multi-position variants ---
        (score_compression_entry, "COMP_VDP_OI_N2", 3, 3.0, 0.05, 2),
        (score_compression_entry, "COMP_VDP_OI_N3", 3, 3.0, 0.05, 3),
        (score_momentum_rotation, "BASELINE_N2", 3, 3.0, 0.05, 2),
        (score_momentum_rotation, "BASELINE_N3", 3, 3.0, 0.05, 3),
        (score_quality_trend, "QUALITY_N2", 3, 3.0, 0.05, 2),

        # --- No trail (v27 showed NO_TRAIL was slightly better) ---
        (score_compression_entry, "COMP_VDP_OI_NOTRL", 3, 0.0, 0.05, 1),
        (score_quality_trend, "QUALITY_NOTRL", 3, 0.0, 0.05, 1),
        (score_momentum_rotation, "BASELINE_NOTRL", 3, 0.0, 0.05, 1),
    ]

    for fn, name, hm, ta, sl, tn in configs:
        r = run_backtest(fn, name, hold_max=hm, trail_atr=ta,
                         stop_loss=sl, top_n=tn)
        if r:
            results.append(r)
            print(f"  {r['name']:35s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"N {r['n']:4d} | DD {r['dd']:6.1f}% | AvgW {r['avg_win']:+.2f}% | "
                  f"AvgL {r['avg_loss']:.2f}% | AvgD {r['avg_days']:.1f}")
            parts = []
            for reason, stats in sorted(r['reasons'].items()):
                wr = stats['w'] / stats['n'] * 100 if stats['n'] > 0 else 0
                parts.append(f"{reason}:{stats['n']}({wr:.0f}%)pnl={stats['pnl']:+.0f}%")
            print(f"  {'':35s} | {' | '.join(parts)}")

    # Yearly for top configs
    results.sort(key=lambda x: -x['ann'])
    print(f"\n--- YEARLY BREAKDOWN (Top 5) ---")
    for r in results[:5]:
        print(f"\n  {r['name']}:")
        for y in sorted(r['yearly'].keys()):
            ys = r['yearly'][y]
            wr = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
            print(f"    {y}: {ys['n']:3d}t WR {wr:5.1f}% PnL {ys['pnl']:+.1f}%")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
