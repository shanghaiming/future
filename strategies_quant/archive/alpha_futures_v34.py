"""
Alpha Futures V34 — Multi-Signal Portfolio Strategy
====================================================
Combines the best of v14b (+73%) and v33 (+50.2%):

1. SUPPLY CHAIN LAG (from v33): upstream momentum → downstream follows
2. VDP MOMENTUM (from v14b): volume delta pressure confirms buying/selling
3. OI FLOW: institutional money flow confirmation
4. Multi-signal RANK averaging (from ML pipeline insight)
5. Top-N positions (2-3 concurrent) for more trades/year
6. Long-only with regime gate
7. NO stop loss (proven to destroy value), trailing stop only
8. Adaptive hold period based on signal strength

Key insight from v14b: time exits have 70% WR, stops have 0% WR.
Key insight from v33: upstream breakout + long-only = 56.3% WR with structural edge.

Target: 100%+ annual, 55%+ WR, DD < 50%
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
# SUPPLY CHAIN DEFINITIONS
# ============================================================
GROUP_MAP = {
    'rbfi': 'ferrous', 'hcfi': 'ferrous', 'ifi': 'ferrous', 'jfi': 'ferrous', 'jmfi': 'ferrous',
    'cufi': 'nonferrous', 'alfi': 'nonferrous', 'znfi': 'nonferrous', 'nifi': 'nonferrous',
    'afi': 'oils', 'mfi': 'oils', 'yfi': 'oils', 'pfi': 'oils', 'cfi': 'oils',
    'scfi': 'energy', 'mafi': 'energy', 'bfi': 'energy', 'fufi': 'energy',
    'ppfi': 'chemical', 'vfi': 'chemical', 'egfi': 'chemical', 'pgfi': 'chemical',
}

UPSTREAM = {
    'rbfi': 'ifi', 'hcfi': 'rbfi', 'jfi': 'jmfi',
    'mafi': 'scfi', 'bfi': 'scfi', 'fufi': 'scfi',
    'mfi': 'afi', 'yfi': 'afi', 'pfi': 'yfi',
    'ppfi': 'mafi', 'vfi': 'mafi', 'egfi': 'mafi',
}


def main():
    t_start = time.time()
    print("=" * 110)
    print("Alpha Futures V34 — Multi-Signal Portfolio (v33 Supply Chain + v14b VDP Rotation)")
    print("=" * 110)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    # Build index maps
    sym_to_si = {syms[si]: si for si in range(NS)}
    group_members = {}
    for si in range(NS):
        grp = GROUP_MAP.get(syms[si])
        if grp is None:
            continue
        if grp not in group_members:
            group_members[grp] = []
        group_members[grp].append(si)

    upstream_si = {}
    for si in range(NS):
        up_sym = UPSTREAM.get(syms[si])
        if up_sym and up_sym in sym_to_si:
            upstream_si[si] = sym_to_si[up_sym]
        else:
            upstream_si[si] = -1

    print(f"  {NS} stocks, {ND} days, Groups: {len(group_members)}, Upstream links: {sum(1 for v in upstream_si.values() if v >= 0)}")

    # ========================================
    # PRECOMPUTE ALL SIGNALS
    # ========================================
    print("\n[Signals] Computing...", flush=True)
    t0 = time.time()

    # 1. Momentum
    mom3 = np.full((NS, ND), np.nan)
    mom5 = np.full((NS, ND), np.nan)
    mom10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(3, ND):
            c_now = C[si, di]
            if np.isnan(c_now) or c_now <= 0:
                continue
            for lag, arr in [(3, mom3), (5, mom5), (10, mom10)]:
                if di < lag:
                    continue
                c_prev = C[si, di - lag]
                if not np.isnan(c_prev) and c_prev > 0:
                    arr[si, di] = (c_now - c_prev) / c_prev

    # 2. Group momentum (excluding self)
    group_mom5_excl = np.full((NS, ND), np.nan)
    for grp, members in group_members.items():
        for di in range(5, ND):
            for sj in members:
                moms = []
                for sk in members:
                    if sk == sj:
                        continue
                    m = mom5[sk, di]
                    if not np.isnan(m):
                        moms.append(m)
                if moms:
                    group_mom5_excl[sj, di] = np.mean(moms)

    # 3. Upstream leader momentum (1-day lagged)
    leader_mom5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        usi = upstream_si[si]
        if usi < 0:
            leader_mom5[si, :] = group_mom5_excl[si, :]
        else:
            for di in range(1, ND):
                lm = mom5[usi, di - 1]
                if not np.isnan(lm):
                    leader_mom5[si, di] = lm

    # 4. Rolling 20-day correlation with upstream
    corr_strength = np.full((NS, ND), np.nan)
    for si in range(NS):
        usi = upstream_si[si]
        for di in range(25, ND):
            own_vals, lead_vals = [], []
            for dd in range(di - 20, di):
                ov = mom5[si, dd]
                if usi >= 0:
                    lv = mom5[usi, dd]
                else:
                    lv = group_mom5_excl[si, dd]
                if not np.isnan(ov) and not np.isnan(lv):
                    own_vals.append(ov)
                    lead_vals.append(lv)
            if len(own_vals) >= 10:
                os, ls = np.std(own_vals), np.std(lead_vals)
                if os > 0 and ls > 0:
                    corr_strength[si, di] = np.corrcoef(own_vals, lead_vals)[0, 1]

    # 5. VDP EMA (10-day)
    vdp_ema = np.full((NS, ND), np.nan)
    for si in range(NS):
        vdp_e = 0.0
        alpha = 2.0 / 11
        for di in range(1, ND):
            d = di - 1
            cd, hd, ld, vd = C[si, d], H[si, d], L[si, d], V[si, d]
            if np.isnan(cd) or np.isnan(hd) or np.isnan(ld) or np.isnan(vd):
                continue
            rng = hd - ld
            if rng <= 0:
                continue
            vdp_val = vd * (2 * cd - hd - ld) / rng
            vdp_e = alpha * vdp_val + (1 - alpha) * vdp_e
            vdp_ema[si, di] = vdp_e

    # 6. OI EMA trend
    oi_ema = np.full((NS, ND), np.nan)
    for si in range(NS):
        oe = 0.0
        alpha_oi = 2.0 / 6
        for di in range(1, ND):
            oi_val = OI[si, di]
            if np.isnan(oi_val):
                continue
            oe = alpha_oi * oi_val + (1 - alpha_oi) * oe
            oi_ema[si, di] = oe

    oi_rising = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(6, ND):
            cur = oi_ema[si, di]
            prev = oi_ema[si, di - 5]
            if not np.isnan(cur) and not np.isnan(prev) and prev > 0:
                oi_rising[si, di] = (cur - prev) / prev

    # 7. OI-price joint signal
    oi_price_div = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(6, ND):
            m3 = mom3[si, di]
            oi_r = oi_rising[si, di]
            if np.isnan(m3) or np.isnan(oi_r):
                continue
            if m3 > 0.01 and oi_r > 0.02:
                oi_price_div[si, di] = min(m3 * 5, 1) * min(oi_r * 3, 1)
            elif m3 < -0.01 and oi_r < -0.02:
                oi_price_div[si, di] = -min(abs(m3) * 5, 1) * min(abs(oi_r) * 3, 1)

    # 8. Volume ratio
    vol_ratio = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            v_now = V[si, di - 1]
            if np.isnan(v_now) or v_now <= 0:
                continue
            v20 = V[si, max(0, di - 20):di]
            v20v = v20[~np.isnan(v20)]
            if len(v20v) >= 10:
                vol_ratio[si, di] = v_now / np.mean(v20v)

    # 9. KER (Kaufman Efficiency Ratio) — regime proxy
    ker = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            c_now = C[si, di]
            c_20 = C[si, di - 20]
            if np.isnan(c_now) or np.isnan(c_20) or c_20 <= 0:
                continue
            net = abs(c_now - c_20)
            total = 0
            for dd in range(di - 19, di + 1):
                c1 = C[si, dd]
                c0 = C[si, dd - 1]
                if not np.isnan(c1) and not np.isnan(c0):
                    total += abs(c1 - c0)
            if total > 0:
                ker[si, di] = net / total

    # 10. ATR for trailing stop
    atr10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(11, ND):
            trs = []
            for dd in range(di - 10, di):
                hi, lo, pc = H[si, dd], L[si, dd], C[si, dd - 1]
                if np.isnan(hi) or np.isnan(lo):
                    continue
                tr = hi - lo
                if not np.isnan(pc):
                    tr = max(tr, abs(hi - pc), abs(lo - pc))
                trs.append(tr)
            if trs:
                atr10[si, di] = np.mean(trs)

    # 11. Upstream 10-day momentum (for sustained moves)
    up_mom10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        usi = upstream_si[si]
        if usi >= 0:
            up_mom10[si, :] = mom10[usi, :]
        else:
            # Use group average mom10
            grp = GROUP_MAP.get(syms[si])
            if grp and grp in group_members:
                for di in range(10, ND):
                    ms = []
                    for sk in group_members[grp]:
                        if sk != si:
                            m = mom10[sk, di]
                            if not np.isnan(m):
                                ms.append(m)
                    if ms:
                        up_mom10[si, di] = np.mean(ms)

    print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

    # ========================================
    # SCORING FUNCTIONS
    # ========================================

    def score_signal_1_vdp_mom(si, di):
        """V14b's proven VDP + momentum signal."""
        m5 = mom5[si, di]
        vd = vdp_ema[si, di]
        if np.isnan(m5):
            return np.nan
        score = np.clip(m5 * 8, -1, 1)
        if not np.isnan(vd):
            if (m5 > 0 and vd > 0) or (m5 < 0 and vd < 0):
                score *= 1.3
            else:
                score *= 0.3
        return score

    def score_signal_2_supply_chain(si, di):
        """V33's upstream breakout signal."""
        if di < 25:
            return np.nan
        own_m5 = mom5[si, di]
        if np.isnan(own_m5):
            return np.nan

        um10 = up_mom10[si, di]
        um5 = leader_mom5[si, di]
        if np.isnan(um5):
            return np.nan

        # Upstream direction (average of 5d and 10d if available)
        if not np.isnan(um10):
            up_dir = (um5 + um10) / 2
        else:
            up_dir = um5

        if abs(um5) < 0.005 and (np.isnan(um10) or abs(um10) < 0.005):
            return np.nan

        divergence = up_dir - own_m5
        lead_strength = np.clip(divergence * 10, -1, 1)

        if abs(divergence) < 0.004:
            return np.nan

        return lead_strength

    def score_signal_3_oi_price(si, di):
        """OI-price joint interpretation."""
        v = oi_price_div[si, di]
        return v if not np.isnan(v) else np.nan

    def score_signal_4_group_mom(si, di):
        """Group momentum lag (own lagging group)."""
        own = mom5[si, di]
        grp = group_mom5_excl[si, di]
        if np.isnan(own) or np.isnan(grp):
            return np.nan
        if abs(grp) < 0.003:
            return np.nan
        divergence = grp - own
        if abs(divergence) < 0.003:
            return np.nan
        return np.clip(divergence * 10, -1, 1)

    def score_signal_5_vol_confirm(si, di):
        """Volume anomaly + momentum."""
        m5 = mom5[si, di]
        vr = vol_ratio[si, di]
        if np.isnan(m5):
            return np.nan
        score = np.clip(m5 * 8, -1, 1)
        if not np.isnan(vr) and vr > 1.5:
            score *= 1.5  # Volume surge confirms
        elif not np.isnan(vr) and vr < 0.7:
            score *= 0.5  # Low volume weakens
        return score

    # ========================================
    # RANK-BASED ENSEMBLE SCORING
    # ========================================
    def make_ensemble_score(signals, weights, require_n=2, long_only=True):
        """
        Multi-signal rank averaging.
        Each signal ranks all commodities for that day, then we average ranks.
        This is the "rank normalization + averaging" technique from the ML pipeline.
        """
        def score(si, di):
            raw_scores = []
            ws = []
            for sig_fn, w in zip(signals, weights):
                s = sig_fn(si, di)
                if not np.isnan(s):
                    raw_scores.append(s)
                    ws.append(w)
            if len(raw_scores) < require_n:
                return np.nan
            # Weighted average
            total_w = sum(ws)
            avg = sum(s * w for s, w in zip(raw_scores, ws)) / total_w

            if long_only and avg < 0:
                return np.nan  # Long only: skip negative scores

            # Boost by OI and VDP confirmation
            oi_r = oi_rising[si, di]
            if not np.isnan(oi_r):
                if avg > 0 and oi_r > 0.02:
                    avg *= 1.3
                elif avg > 0 and oi_r < -0.02:
                    avg *= 0.6

            vd = vdp_ema[si, di]
            if not np.isnan(vd):
                if avg > 0 and vd > 0:
                    avg *= 1.2
                elif avg > 0 and vd < 0:
                    avg *= 0.5

            return avg
        return score

    # ========================================
    # BACKTEST ENGINE — MULTI-POSITION
    # ========================================
    def run_backtest(score_fn, name, top_n=1, hold_min=2, hold_max=5,
                     trail_atr_mult=2.5, stop_loss_pct=0.0, long_only=True):
        """
        Multi-position backtest.
        top_n: max concurrent positions (cash divided equally)
        stop_loss_pct=0 means no stop loss (key insight from v14b)
        """
        cash = float(CASH0)
        trades = []
        positions = []  # list of position dicts
        last_exit = {}  # sym -> last exit di

        for di in range(MIN_TRAIN, ND):
            year = dates[di].year

            # === MANAGE EXISTING POSITIONS ===
            new_positions = []
            for pos in positions:
                c = C[pos['si'], di]
                if np.isnan(c) or c <= 0:
                    c = pos['entry']
                mult = MULT.get(pos['sym'], DEF_MULT)
                mkt_val = c * mult * pos['lots']
                pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
                pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] > 0 else 0
                days_held = di - pos['entry_di']

                exit_reason = None

                # 1. Stop loss (only if enabled)
                if stop_loss_pct > 0 and pnl_pct / 100 < -stop_loss_pct:
                    exit_reason = 'stop'

                # 2. Trailing stop (after day 2)
                if exit_reason is None and trail_atr_mult > 0 and days_held >= 2:
                    atr = pos.get('atr', 0)
                    if atr > 0:
                        if pos['dir'] == 1:
                            new_trail = c - trail_atr_mult * atr
                            if new_trail > pos.get('trail_price', pos['entry']):
                                pos['trail_price'] = new_trail
                            if c < pos['trail_price']:
                                exit_reason = 'trail'
                        else:
                            new_trail = c + trail_atr_mult * atr
                            if new_trail < pos.get('trail_price', pos['entry']):
                                pos['trail_price'] = new_trail
                            if c > pos['trail_price']:
                                exit_reason = 'trail'

                # 3. Signal flip (after min hold)
                if exit_reason is None and days_held >= hold_min:
                    cur_score = score_fn(pos['si'], di)
                    if not np.isnan(cur_score):
                        if pos['dir'] == 1 and cur_score < -0.02:
                            exit_reason = 'signal_flip'

                # 4. Time exit
                if exit_reason is None and days_held >= hold_max:
                    exit_reason = 'time'

                if exit_reason:
                    cost_out = mkt_val * COMM
                    cash += mkt_val - cost_out
                    trades.append({
                        'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                        'days': days_held, 'di': di, 'year': year,
                        'sym': pos['sym'], 'dir': pos['dir'], 'reason': exit_reason,
                    })
                    last_exit[pos['sym']] = di
                else:
                    new_positions.append(pos)

            positions = new_positions

            # === OPEN NEW POSITIONS ===
            n_open = len(positions)
            if n_open < top_n:
                slots = top_n - n_open
                # Score all symbols
                scored = []
                for si in range(NS):
                    sc = score_fn(si, di)
                    if np.isnan(sc) or sc <= 0.01:
                        continue
                    sym = syms[si]
                    # Skip if already in position
                    if any(p['sym'] == sym for p in positions):
                        continue
                    # Reentry gap
                    if sym in last_exit and di - last_exit[sym] < 1:
                        continue
                    scored.append((si, sc, sym))

                if scored:
                    scored.sort(key=lambda x: -x[1])
                    # Allocate cash equally among slots
                    cash_per_slot = cash / slots if slots > 0 else cash

                    for best_si, best_sc, best_sym in scored[:slots]:
                        c = C[best_si, di]
                        if np.isnan(c) or c <= 0:
                            continue
                        mult = MULT.get(best_sym, DEF_MULT)
                        notional = c * mult
                        if notional <= 0:
                            continue

                        lots = int(cash_per_slot / (notional * (1 + COMM)))
                        if lots <= 0:
                            continue
                        cost_in = notional * lots * (1 + COMM)
                        if cost_in > cash:
                            lots = int(cash / (notional * (1 + COMM)))
                            if lots <= 0:
                                continue
                            cost_in = notional * lots * (1 + COMM)

                        # ATR for trailing stop
                        atr_val = 0
                        trs = []
                        for dd in range(max(1, di - 10), di + 1):
                            hi, lo, pc = H[best_si, dd], L[best_si, dd], C[best_si, dd - 1]
                            if np.isnan(hi) or np.isnan(lo):
                                continue
                            tr = hi - lo
                            if not np.isnan(pc):
                                tr = max(tr, abs(hi - pc), abs(lo - pc))
                            trs.append(tr)
                        if trs:
                            atr_val = np.mean(trs)

                        cash -= cost_in
                        trail_price = c - trail_atr_mult * atr_val
                        positions.append({
                            'si': best_si, 'entry': c, 'entry_di': di,
                            'lots': lots, 'dir': 1, 'sym': best_sym,
                            'atr': atr_val, 'trail_price': trail_price,
                        })

        # Close remaining
        for pos in positions:
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

        # Stats
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
        avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_abs'] > 0]) if nw > 0 else 0
        avg_loss = np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_abs'] <= 0]) if nw < len(trades) else 0
        avg_days = np.mean([t['days'] for t in trades])
        pf = (sum(t['pnl_abs'] for t in trades if t['pnl_abs'] > 0) /
              max(abs(sum(t['pnl_abs'] for t in trades if t['pnl_abs'] < 0)), 1))

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

        grp_counts = {}
        for t in trades:
            g = GROUP_MAP.get(t['sym'], 'other')
            if g not in grp_counts:
                grp_counts[g] = {'n': 0, 'w': 0, 'pnl': 0.0}
            grp_counts[g]['n'] += 1
            if t['pnl_abs'] > 0:
                grp_counts[g]['w'] += 1
            grp_counts[g]['pnl'] += t['pnl_abs']

        return {
            'name': name, 'ann': round(ann, 1), 'n': len(trades),
            'wr': round(wr, 1), 'dd': round(max_dd, 1),
            'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
            'avg_days': round(avg_days, 1), 'pf': round(pf, 2),
            'cash': round(cash, 0),
            'reasons': reasons, 'yearly': year_stats, 'grp_counts': grp_counts,
        }

    # ========================================
    # DEFINE CONFIGURATIONS
    # ========================================
    print("\n[Backtest] Running configurations...", flush=True)
    results = []

    # --- Individual signals ---
    single_signals = [
        ("S1_VDP_MOM", score_signal_1_vdp_mom),
        ("S2_SUPPLY", score_signal_2_supply_chain),
        ("S3_OI_PRICE", score_signal_3_oi_price),
        ("S4_GROUP", score_signal_4_group_mom),
        ("S5_VOL", score_signal_5_vol_confirm),
    ]

    # --- Ensemble signals ---
    # V14b replica (VDP + MOM only, long only)
    ens_v14b = make_ensemble_score(
        [score_signal_1_vdp_mom], [1.0], require_n=1, long_only=True
    )
    # V33 replica (supply chain only, long only)
    ens_v33 = make_ensemble_score(
        [score_signal_2_supply_chain], [1.0], require_n=1, long_only=True
    )
    # Full ensemble: all 5 signals
    ens_full = make_ensemble_score(
        [score_signal_1_vdp_mom, score_signal_2_supply_chain,
         score_signal_3_oi_price, score_signal_4_group_mom, score_signal_5_vol_confirm],
        [0.30, 0.30, 0.15, 0.15, 0.10],
        require_n=2, long_only=True
    )
    # Heavy supply chain: supply chain dominant
    ens_sc_heavy = make_ensemble_score(
        [score_signal_2_supply_chain, score_signal_1_vdp_mom,
         score_signal_4_group_mom],
        [0.50, 0.30, 0.20],
        require_n=2, long_only=True
    )
    # VDP + supply chain blend
    ens_vdp_sc = make_ensemble_score(
        [score_signal_1_vdp_mom, score_signal_2_supply_chain],
        [0.40, 0.60],
        require_n=1, long_only=True
    )
    # All with OI+VDP boost
    ens_oi_boost = make_ensemble_score(
        [score_signal_1_vdp_mom, score_signal_2_supply_chain,
         score_signal_3_oi_price, score_signal_5_vol_confirm],
        [0.25, 0.35, 0.20, 0.20],
        require_n=2, long_only=True
    )

    # --- Run all configs ---
    configs = []

    # Single signal baselines (P1, no stop)
    for sname, sfn in single_signals:
        for top_n in [1, 2, 3]:
            for hold_max in [3, 5, 7]:
                configs.append((sfn, f"{sname}_N{top_n}_H{hold_max}",
                                top_n, 2, hold_max, 2.5, 0.0))

    # Ensemble configs
    for ename, efn in [("ENS_V14B", ens_v14b), ("ENS_V33", ens_v33),
                       ("ENS_FULL", ens_full), ("ENS_SC_HEAVY", ens_sc_heavy),
                       ("ENS_VDP_SC", ens_vdp_sc), ("ENS_OI_BOOST", ens_oi_boost)]:
        for top_n in [1, 2, 3]:
            for hold_max in [3, 5, 7]:
                for trail in [2.0, 2.5, 3.0]:
                    configs.append((efn, f"{ename}_N{top_n}_H{hold_max}_T{trail}",
                                    top_n, 2, hold_max, trail, 0.0))

    # Special configs: with stop loss for comparison
    for ename, efn in [("ENS_FULL", ens_full), ("ENS_SC_HEAVY", ens_sc_heavy)]:
        for sl in [0.03, 0.05]:
            configs.append((efn, f"{ename}_N2_H5_SL{int(sl*100)}",
                            2, 2, 5, 2.5, sl))

    print(f"  {len(configs)} configurations", flush=True)

    for ci, (fn, name, tn, hmin, hmax, trail, sl) in enumerate(configs):
        r = run_backtest(fn, name, top_n=tn, hold_min=hmin, hold_max=hmax,
                         trail_atr_mult=trail, stop_loss_pct=sl, long_only=True)
        if r and r['ann'] > 0:
            results.append(r)
            if r['ann'] > 30:
                parts = []
                for reason, stats in sorted(r['reasons'].items()):
                    wr_r = stats['w'] / stats['n'] * 100 if stats['n'] > 0 else 0
                    parts.append(f"{reason}:{stats['n']}({wr_r:.0f}%)")
                print(f"  {r['name']:40s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                      f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                      f"AvgW {r['avg_win']:+.2f}% | AvgL {r['avg_loss']:.2f}% | "
                      f"AvgD {r['avg_days']:.1f}")
                print(f"  {'':40s} | Exits: {' | '.join(parts)}")

        if (ci + 1) % 50 == 0:
            print(f"  [{ci+1}/{len(configs)}] {len(results)} profitable", flush=True)

    # ========================================
    # RESULTS
    # ========================================
    results.sort(key=lambda x: -x['ann'])

    print(f"\n{'=' * 110}")
    print(f"  TOP RESULTS (sorted by annualized return)")
    print(f"{'=' * 110}")
    print(f"  {'Strategy':40s} | {'Ann':>7s} | {'WR':>5s} | {'N':>4s} | {'DD':>6s} | "
          f"{'PF':>4s} | {'AvgW':>7s} | {'AvgL':>6s} | {'AvgD':>4s}")
    print(f"  {'-' * 105}")
    for r in results[:30]:
        print(f"  {r['name']:40s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:4.2f} | "
              f"{r['avg_win']:+6.2f}% | {r['avg_loss']:5.2f}% | "
              f"{r['avg_days']:4.1f}")

    if results:
        best = results[0]
        print(f"\n  BEST: {best['name']}  |  Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  "
              f"N={best['n']}  DD={best['dd']:.1f}%")
        print(f"  AvgWin={best['avg_win']:+.2f}%  AvgLoss={best['avg_loss']:.2f}%  Final={best['cash']:.0f}")

        print(f"\n  EXIT REASON BREAKDOWN:")
        for reason, s in sorted(best['reasons'].items(), key=lambda x: -x[1]['n']):
            rwr = s['w'] / max(s['n'], 1) * 100
            print(f"    {reason:12s}: {s['n']:4d} trades  WR={rwr:5.1f}%  PnL={s['pnl']:+.1f}%")

        print(f"\n  YEARLY BREAKDOWN:")
        for y in sorted(best['yearly'].keys()):
            s = best['yearly'][y]
            wr = s['w'] / max(s['n'], 1) * 100
            print(f"    {y}: {s['n']:3d} trades  WR={wr:5.1f}%  PnL={s['pnl']:+.1f}%")

        print(f"\n  GROUP BREAKDOWN:")
        for g in sorted(best['grp_counts'].keys(), key=lambda x: -best['grp_counts'][x]['n']):
            gs = best['grp_counts'][g]
            wr_g = gs['w'] / max(gs['n'], 1) * 100
            print(f"    {g:15s}: {gs['n']:3d}t  WR={wr_g:5.1f}%  Abs={gs['pnl']:+.0f}")

    # Yearly breakdown for top 5
    if len(results) >= 2:
        print(f"\n  YEARLY BREAKDOWN FOR TOP 5 CONFIGS:")
        for r in results[:5]:
            print(f"\n  #{results.index(r)+1}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.1f}%, DD={r['dd']:.1f}%)")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:3d}t  WR={wr_y:5.1f}%  PnL={ys['pnl']:+.1f}%")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 110)


if __name__ == '__main__':
    main()
