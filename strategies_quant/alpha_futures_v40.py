"""
Alpha Futures V40 — Open Interest (OI) Institutional Flow Strategy
===================================================================
Core insight: OI changes combined with price direction reveal institutional intent:
  OI↑ + Price↑ = New longs entering (bullish) → follow long
  OI↓ + Price↑ = Short covering → fading move, don't chase
  OI↑ + Price↓ = New shorts entering (bearish) → avoid or short
  OI↓ + Price↓ = Long liquidation → possible bottom

Signals:
  S1: OI Surge — OI >> 20-day average signals major new positions
  S2: OI-Price Divergence — OI↑ + Price↑ with rolling correlation
  S3: OI Exhaustion — OI stalls after rapid rise → reversal
  S4: VDP + OI Confirmation — buying pressure + new longs = double confirmation
  S5: OI Surge + Group Momentum Lag — group signal only when OI confirms

Backtest: single position per symbol, long only, walk-forward validated.
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

GROUP_MAP = {
    'rbfi': 'ferrous', 'hcfi': 'ferrous', 'ifi': 'ferrous', 'jfi': 'ferrous', 'jmfi': 'ferrous',
    'cufi': 'nonferrous', 'alfi': 'nonferrous', 'znfi': 'nonferrous', 'nifi': 'nonferrous',
    'afi': 'oils', 'mfi': 'oils', 'yfi': 'oils', 'pfi': 'oils', 'cfi': 'oils',
    'scfi': 'energy', 'mafi': 'energy', 'bfi': 'energy', 'fufi': 'energy',
    'ppfi': 'chemical', 'vfi': 'chemical', 'egfi': 'chemical', 'pgfi': 'chemical',
}


def main():
    t_start = time.time()
    print("=" * 120)
    print("Alpha Futures V40 — Open Interest (OI) Institutional Flow Strategy")
    print("Core: OI changes + price direction reveal institutional intent. Long only.")
    print("=" * 120)

    # ========================================
    # LOAD DATA
    # ========================================
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    # Check OI coverage
    oi_valid = np.sum(~np.isnan(OI))
    oi_total = OI.size
    print(f"  OI coverage: {oi_valid}/{oi_total} cells ({oi_valid/max(oi_total,1)*100:.1f}%)", flush=True)

    sym_to_si = {syms[si]: si for si in range(NS)}

    # Group membership
    group_members = {}
    for si in range(NS):
        grp = GROUP_MAP.get(syms[si])
        if grp is None:
            continue
        if grp not in group_members:
            group_members[grp] = []
        group_members[grp].append(si)

    print(f"  {NS} stocks, {ND} days, Groups: {len(group_members)}")

    # ========================================
    # PRECOMPUTE ALL SIGNALS
    # ========================================
    print("\n[Signals] Computing OI-based signals...", flush=True)
    t0 = time.time()

    # --- ATR (10-day) ---
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

    # --- OI % change at multiple lookbacks ---
    oi_pct = {}
    for lb in [3, 5, 10, 20]:
        op = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(lb, ND):
                oi_now = OI[si, di]
                oi_prev = OI[si, di - lb]
                if (not np.isnan(oi_now) and not np.isnan(oi_prev)
                        and oi_prev > 0):
                    op[si, di] = (oi_now - oi_prev) / oi_prev
        oi_pct[lb] = op

    # --- Price % change at multiple lookbacks ---
    pr_pct = {}
    for lb in [3, 5, 10, 20]:
        pp = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(lb, ND):
                c_now = C[si, di]
                c_prev = C[si, di - lb]
                if (not np.isnan(c_now) and not np.isnan(c_prev)
                        and c_prev > 0):
                    pp[si, di] = (c_now - c_prev) / c_prev
        pr_pct[lb] = pp

    # --- Daily OI change and price change (for rolling correlation) ---
    oi_daily = np.full((NS, ND), np.nan)
    pr_daily = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            oi_now = OI[si, di]
            oi_prev = OI[si, di - 1]
            c_now = C[si, di]
            c_prev = C[si, di - 1]
            if (not np.isnan(oi_now) and not np.isnan(oi_prev)
                    and oi_prev > 0):
                oi_daily[si, di] = (oi_now - oi_prev) / oi_prev
            if (not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0):
                pr_daily[si, di] = (c_now - c_prev) / c_prev

    # --- 20-day rolling correlation between OI change and price change ---
    oi_price_corr = np.full((NS, ND), np.nan)
    corr_win = 20
    for si in range(NS):
        for di in range(corr_win, ND):
            ods = oi_daily[si, di - corr_win + 1:di + 1]
            pds = pr_daily[si, di - corr_win + 1:di + 1]
            mask = ~np.isnan(ods) & ~np.isnan(pds)
            n_valid = np.sum(mask)
            if n_valid >= 10:
                ov = ods[mask]
                pv = pds[mask]
                ov_m = ov - np.mean(ov)
                pv_m = pv - np.mean(pv)
                denom = np.sqrt(np.sum(ov_m ** 2) * np.sum(pv_m ** 2))
                if denom > 0:
                    oi_price_corr[si, di] = np.sum(ov_m * pv_m) / denom

    # --- OI 20-day rolling mean and surge ratio ---
    oi_20_avg = np.full((NS, ND), np.nan)
    oi_surge = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            window = OI[si, di - 19:di + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= 10:
                avg = np.mean(valid)
                oi_20_avg[si, di] = avg
                cur_oi = OI[si, di]
                if not np.isnan(cur_oi) and avg > 0:
                    oi_surge[si, di] = cur_oi / avg

    # --- OI exhaustion: OI rising 5+ days then stalls ---
    # Measure: 5-day OI growth rate vs 10-day OI growth rate
    oi_exhaustion = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            oi_5 = oi_pct[5][si, di] if not np.isnan(oi_pct[5][si, di]) else 0
            oi_10 = oi_pct[10][si, di] if not np.isnan(oi_pct[10][si, di]) else 0
            # Exhaustion: OI rose fast over 10 days but slowed in last 5
            if oi_10 > 0.05 and oi_5 < oi_10 * 0.3:
                oi_exhaustion[si, di] = oi_10  # magnitude of prior OI build

    # --- VDP (Volume Delta Pressure) EMA ---
    vdp_ema = np.full((NS, ND), np.nan)
    for si in range(NS):
        vdp_e = 0.0
        alpha_vdp = 2.0 / 11
        for di in range(1, ND):
            d = di - 1
            cd, hd, ld, vd = C[si, d], H[si, d], L[si, d], V[si, d]
            if np.isnan(cd) or np.isnan(hd) or np.isnan(ld) or np.isnan(vd):
                continue
            rng = hd - ld
            if rng <= 0:
                continue
            vdp_val = vd * (2 * cd - hd - ld) / rng
            vdp_e = alpha_vdp * vdp_val + (1 - alpha_vdp) * vdp_e
            vdp_ema[si, di] = vdp_e

    # --- OI EMA trend (rising indicator) ---
    oi_ema = np.full((NS, ND), np.nan)
    oi_rising = np.full((NS, ND), np.nan)
    for si in range(NS):
        oe = 0.0
        alpha_oi = 2.0 / 6
        for di in range(1, ND):
            oi_val = OI[si, di]
            if np.isnan(oi_val):
                continue
            oe = alpha_oi * oi_val + (1 - alpha_oi) * oe
            oi_ema[si, di] = oe
        for di in range(6, ND):
            cur = oi_ema[si, di]
            prev = oi_ema[si, di - 5]
            if not np.isnan(cur) and not np.isnan(prev) and prev > 0:
                oi_rising[si, di] = (cur - prev) / prev

    # --- Group momentum (5-day, excluding self) ---
    mom5 = pr_pct[5]
    grp_mom5 = np.full((NS, ND), np.nan)
    for grp, members in group_members.items():
        for di in range(5, ND):
            for sj in members:
                ms = []
                for sk in members:
                    if sk == sj:
                        continue
                    m = mom5[sk, di]
                    if not np.isnan(m):
                        ms.append(m)
                if ms:
                    grp_mom5[sj, di] = np.mean(ms)

    print(f"  Done ({time.time()-t0:.1f}s)", flush=True)

    # ========================================
    # SCORING FUNCTIONS
    # ========================================

    def make_oi_surge_score(surge_thresh=1.5, lookback=5, scale=10.0,
                            require_price_up=True, price_lb=5):
        """S1: OI Surge — major new institutional positions.
        When OI >> 20-day average AND price is up, institutions are buying."""
        def score(si, di):
            surge = oi_surge[si, di]
            if np.isnan(surge) or surge < surge_thresh:
                return np.nan

            pc = pr_pct[price_lb][si, di] if not np.isnan(pr_pct[price_lb][si, di]) else 0
            if require_price_up and pc <= 0:
                return np.nan

            # Score scales with how much OI exceeds threshold
            sc = (surge - 1.0) * scale
            sc = min(sc, 1.0)

            # Boost if price confirming direction
            if pc > 0:
                sc *= (1 + min(pc * 5, 0.5))

            return sc
        return score

    def make_oi_price_div_score(oi_thresh=0.05, lookback=5, corr_weight=0.3,
                                scale=10.0):
        """S2: OI-Price Divergence — OI↑ + Price↑ with correlation confirmation.
        Strongest when OI and price are rising together (institutional consensus)."""
        def score(si, di):
            oi_chg = oi_pct[lookback][si, di]
            pr_chg = pr_pct[lookback][si, di]
            if np.isnan(oi_chg) or np.isnan(pr_chg):
                return np.nan

            # Need OI rising AND price rising
            if oi_chg < oi_thresh or pr_chg <= 0:
                return np.nan

            sc = oi_chg * scale
            sc = min(sc, 1.0)

            # Correlation bonus: if OI and price are positively correlated, stronger signal
            corr = oi_price_corr[si, di]
            if not np.isnan(corr):
                if corr > 0.3:
                    sc *= (1 + corr_weight)
                elif corr < -0.3:
                    sc *= (1 - corr_weight * 0.5)

            return sc
        return score

    def make_oi_exhaustion_score(exhaust_thresh=0.5, scale=5.0):
        """S3: OI Exhaustion (Reversion) — OI rose rapidly then stalled.
        The move is running on momentum without new institutional support.
        We look for SHORT signals or simply AVOID (score=0).
        Since we are long-only, this is used as a filter: returns nan when exhausted."""
        def score(si, di):
            exc = oi_exhaustion[si, di]
            if np.isnan(exc):
                return np.nan
            # Exhaustion detected — do NOT enter long
            return np.nan
        return score

    def make_vdp_oi_score(scale=8.0, oi_rise_thresh=0.01):
        """S4: VDP + OI Combined — buying pressure confirmed by new longs.
        VDP > 0 (buying) AND OI rising (new longs) = double confirmation."""
        def score(si, di):
            vd = vdp_ema[si, di]
            if np.isnan(vd) or vd <= 0:
                return np.nan

            oi_r = oi_rising[si, di]
            if np.isnan(oi_r) or oi_r < oi_rise_thresh:
                return np.nan

            # Score based on VDP magnitude (normalized)
            sc = min(abs(vd) / 5e6, 1.0) * scale / 10.0

            # Boost by OI rise magnitude
            if oi_r > 0.05:
                sc *= 1.3

            return sc
        return score

    def make_oi_group_lag_score(mom_lag=5, min_lag=0.003, scale=10.0,
                                oi_rise_thresh=0.01):
        """S5: OI Surge + Group Momentum Lag — group signal only when OI confirms.
        Commodity lags its supply-chain group, but only take the signal when OI is rising
        (institutional money confirms the catch-up)."""
        def score(si, di):
            own = mom5[si, di]
            grp = grp_mom5[si, di]
            if np.isnan(own) or np.isnan(grp):
                return np.nan

            divergence = grp - own
            if abs(divergence) < min_lag:
                return np.nan
            if divergence <= 0:
                return np.nan

            # OI must be rising to confirm
            oi_r = oi_rising[si, di]
            if np.isnan(oi_r) or oi_r < oi_rise_thresh:
                return np.nan

            sc = np.clip(divergence * scale, -1, 1)

            # OI surge bonus
            surge = oi_surge[si, di]
            if not np.isnan(surge) and surge > 1.5:
                sc *= 1.3

            return sc
        return score

    # ========================================
    # BACKTEST ENGINE
    # ========================================
    def run_backtest(score_fn, name, top_n=1, hold_min=2, hold_max=3,
                     trail_atr_mult=2.5, wf_split_year=None):
        """Multi-position backtest with optional walk-forward split."""
        cash = float(CASH0)
        trades = []
        positions = []
        last_exit = {}

        for di in range(MIN_TRAIN, ND):
            year = dates[di].year

            if wf_split_year is not None and year < wf_split_year:
                continue

            # Manage existing positions
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

                # Trailing stop
                if trail_atr_mult > 0 and days_held >= 2:
                    atr = pos.get('atr', 0)
                    if atr > 0 and pos['dir'] == 1:
                        new_trail = c - trail_atr_mult * atr
                        if new_trail > pos.get('trail_price', pos['entry']):
                            pos['trail_price'] = new_trail
                        if c < pos['trail_price']:
                            exit_reason = 'trail'

                # Signal flip
                if exit_reason is None and days_held >= hold_min:
                    cur_score = score_fn(pos['si'], di)
                    if not np.isnan(cur_score) and cur_score < -0.01:
                        exit_reason = 'signal_flip'

                # Time exit
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

            # Open new positions
            n_open = len(positions)
            if n_open < top_n:
                slots = top_n - n_open
                scored = []
                for si in range(NS):
                    sc = score_fn(si, di)
                    if np.isnan(sc) or sc <= 0.01:
                        continue
                    sym = syms[si]
                    if any(p['sym'] == sym for p in positions):
                        continue
                    scored.append((si, sc, sym))

                if scored:
                    scored.sort(key=lambda x: -x[1])
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

                        atr_val = atr10[best_si, di] if not np.isnan(atr10[best_si, di]) else 0
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

        if len(trades) < 5:
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
        if wf_split_year:
            first_test_di = None
            for di in range(MIN_TRAIN, ND):
                if dates[di].year >= wf_split_year:
                    first_test_di = di
                    break
            if first_test_di:
                days_total = (dates[ND - 1] - dates[first_test_di]).days
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
    # PARAMETER SWEEP — ~250 configs
    # ========================================
    print("\n[Backtest] Running parameter sweep...", flush=True)
    results = []
    configs = []

    # --- S1: Pure OI Surge ---
    for surge_t in [1.3, 1.5, 2.0, 2.5]:
        for lb in [5, 10, 20]:
            for hold in [3, 5, 7]:
                for top_n in [1, 3]:
                    configs.append((
                        make_oi_surge_score(surge_thresh=surge_t, lookback=lb),
                        f"S1_SURGE{surge_t}_LB{lb}_H{hold}_N{top_n}",
                        top_n, 2, hold, 2.5, None
                    ))

    # --- S2: OI-Price Divergence ---
    for oi_thresh in [0.02, 0.05, 0.10]:
        for lb in [5, 10, 20]:
            for cw in [0.2, 0.4]:
                for hold in [3, 5, 7]:
                    configs.append((
                        make_oi_price_div_score(oi_thresh=oi_thresh, lookback=lb,
                                                corr_weight=cw),
                        f"S2_DIV_OI{oi_thresh*100:.0f}_LB{lb}_CW{cw*10:.0f}_H{hold}",
                        1, 2, hold, 2.5, None
                    ))

    # --- S4: VDP + OI Combined ---
    for oi_rt in [0.01, 0.03, 0.05]:
        for hold in [3, 5, 7]:
            for trail in [2.0, 3.0, 4.0]:
                configs.append((
                    make_vdp_oi_score(oi_rise_thresh=oi_rt),
                    f"S4_VDP_OI{oi_rt*100:.0f}_H{hold}_TR{trail:.0f}",
                    1, 2, hold, trail, None
                ))

    # --- S5: OI Surge + Group Momentum Lag ---
    for min_lag in [0.002, 0.003, 0.005]:
        for oi_rt in [0.01, 0.03]:
            for hold in [3, 5, 7]:
                configs.append((
                    make_oi_group_lag_score(min_lag=min_lag, oi_rise_thresh=oi_rt),
                    f"S5_GLAG_ML{min_lag*1000:.0f}_OI{oi_rt*100:.0f}_H{hold}",
                    1, 2, hold, 2.5, None
                ))

    # --- S1 + S4 Combined: OI Surge with VDP filter ---
    for surge_t in [1.5, 2.0]:
        for hold in [3, 5, 7]:
            def make_combined_s1_s4(surge_t, hold):
                oi_surge_fn = make_oi_surge_score(surge_thresh=surge_t)
                def score(si, di):
                    # OI surge signal
                    sc1 = oi_surge_fn(si, di)
                    if np.isnan(sc1):
                        return np.nan
                    # VDP must be positive
                    vd = vdp_ema[si, di]
                    if np.isnan(vd) or vd <= 0:
                        return np.nan
                    return sc1 * 1.2  # boost for double confirmation
                return score
            configs.append((
                make_combined_s1_s4(surge_t, hold),
                f"S1S4_SURGE{surge_t}_VDP_H{hold}",
                1, 2, hold, 2.5, None
            ))

    # --- Walk-forward for best-looking config families ---
    for surge_t in [1.5, 2.0]:
        for hold in [3, 5]:
            for wf_year in [2023, 2024]:
                configs.append((
                    make_oi_surge_score(surge_thresh=surge_t),
                    f"S1_SURGE{surge_t}_H{hold}_WF{wf_year}",
                    1, 2, hold, 2.5, wf_year
                ))

    for oi_thresh in [0.02, 0.05]:
        for hold in [3, 5]:
            for wf_year in [2023, 2024]:
                configs.append((
                    make_oi_price_div_score(oi_thresh=oi_thresh),
                    f"S2_DIV_OI{oi_thresh*100:.0f}_H{hold}_WF{wf_year}",
                    1, 2, hold, 2.5, wf_year
                ))

    for oi_rt in [0.01, 0.03]:
        for hold in [3, 5]:
            for wf_year in [2023, 2024]:
                configs.append((
                    make_vdp_oi_score(oi_rise_thresh=oi_rt),
                    f"S4_VDP_OI{oi_rt*100:.0f}_H{hold}_WF{wf_year}",
                    1, 2, hold, 2.5, wf_year
                ))

    for min_lag in [0.003, 0.005]:
        for hold in [3, 5]:
            for wf_year in [2023, 2024]:
                configs.append((
                    make_oi_group_lag_score(min_lag=min_lag),
                    f"S5_GLAG_ML{min_lag*1000:.0f}_H{hold}_WF{wf_year}",
                    1, 2, hold, 2.5, wf_year
                ))

    print(f"  {len(configs)} configurations", flush=True)

    for ci, (fn, name, tn, hmin, hmax, trail, wf) in enumerate(configs):
        r = run_backtest(fn, name, top_n=tn, hold_min=hmin, hold_max=hmax,
                         trail_atr_mult=trail, wf_split_year=wf)
        if r and r['ann'] > -50:
            results.append(r)
            if r['ann'] > 20:
                parts = []
                for reason, stats in sorted(r['reasons'].items()):
                    wr_r = stats['w'] / stats['n'] * 100 if stats['n'] > 0 else 0
                    parts.append(f"{reason}:{stats['n']}({wr_r:.0f}%)")
                print(f"  {r['name']:50s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                      f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                      f"AvgW {r['avg_win']:+.2f}% | AvgL {r['avg_loss']:.2f}% | AvgD {r['avg_days']:.1f}")
                print(f"  {'':50s} | Exits: {' | '.join(parts)}")

        if (ci + 1) % 50 == 0:
            print(f"  [{ci+1}/{len(configs)}] {len(results)} results so far", flush=True)

    # ========================================
    # RESULTS
    # ========================================
    results.sort(key=lambda x: -x['ann'])

    wf_results = [r for r in results if '_WF' in r['name']]
    full_results = [r for r in results if '_WF' not in r['name']]

    print(f"\n{'=' * 120}")
    print(f"  TOP 20 FULL-PERIOD RESULTS")
    print(f"{'=' * 120}")
    print(f"  {'Strategy':50s} | {'Ann':>7s} | {'WR':>5s} | {'N':>4s} | {'DD':>6s} | "
          f"{'PF':>4s} | {'AvgW':>7s} | {'AvgL':>6s} | {'AvgD':>4s}")
    print(f"  {'-' * 120}")
    for r in full_results[:20]:
        print(f"  {r['name']:50s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:4.2f} | "
              f"{r['avg_win']:+6.2f}% | {r['avg_loss']:5.2f}% | "
              f"{r['avg_days']:4.1f}")

    if wf_results:
        wf_results.sort(key=lambda x: -x['ann'])
        print(f"\n  WALK-FORWARD RESULTS (out-of-sample)")
        print(f"  {'-' * 120}")
        for r in wf_results[:10]:
            print(f"  {r['name']:50s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
                  f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:4.2f}")

    if full_results:
        best = full_results[0]
        print(f"\n  BEST: {best['name']}")
        print(f"  Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  "
              f"N={best['n']}  DD={best['dd']:.1f}%  PF={best['pf']:.2f}  Final={best['cash']:.0f}")

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

    # Yearly for top 5
    if len(full_results) >= 2:
        print(f"\n  YEARLY BREAKDOWN FOR TOP 5:")
        for rank, r in enumerate(full_results[:5]):
            print(f"\n  #{rank+1}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.1f}%, DD={r['dd']:.1f}%)")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:3d}t  WR={wr_y:5.1f}%  PnL={ys['pnl']:+.1f}%")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 120)


if __name__ == '__main__':
    main()
