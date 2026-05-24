"""
Alpha Futures V47 — Opening Range Breakout + OI Confirmation
=============================================================
Core idea: When price breaks above yesterday's high (or below yesterday's low)
with rising OI, it signals institutional conviction behind the breakout.
Follow the breakout.

5 signals:
  1. Yesterday-High Breakout — close > prev_high AND OI rising
  2. N-day Breakout — close > nday_high AND OI rising (test N = 5, 10, 20)
  3. Range Expansion Breakout — large range + close > prev_high
  4. Body Ratio Breakout — strong body candle + close > prev_high + OI rising
  5. Breakout + Momentum — close > prev_high + mom5 > 0 + OI rising

Configs (~200-250): signals x N-day x hold x trail x top_n x OI on/off
Walk-forward validation on best (2023, 2024).
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

MULT = {'agfi': 15, 'alfi': 5, 'aufi': 1000, 'bufi': 10, 'cufi': 5, 'fufi': 10,
        'rbfi': 10, 'znfi': 5, 'nifi': 1, 'hcfi': 10, 'spfi': 10, 'ssfi': 5,
        'sffi': 5, 'smfi': 5, 'pbfi': 5, 'snfi': 1, 'rufi': 10, 'wrffi': 10,
        'afi': 10, 'bfi': 10, 'bbfi': 500, 'cffi': 5, 'cfi': 10, 'csfi': 10,
        'ebfi': 5, 'egfi': 10, 'fbfi': 500, 'ifi': 100, 'jfi': 100, 'jmfi': 60,
        'lfi': 5, 'mfi': 10, 'pgfi': 20, 'ppfi': 5, 'vfi': 5, 'yfi': 10,
        'pfi': 10, 'jdfi': 5, 'lhfi': 16, 'pkfi': 5, 'rrfi': 20, 'lrfi': 20,
        'jrfi': 20, 'pmfi': 20, 'whfi': 20, 'rsfi': 20, 'cjfi': 10, 'mafi': 10,
        'apfi': 10, 'cyfi': 5, 'fgfi': 20, 'oifi': 10, 'pfifi': 5, 'rmfi': 10,
        'srfi': 10, 'tafi': 5, 'safi': 20, 'urfi': 20, 'scfi': 1000, 'lufi': 10,
        'bcfi': 5, 'nrfi': 1, 'lgfi': 20, 'brfi': 5, 'lcfi': 1, 'sifi': 5,
        'ni': 1, 'tai': 5}
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
    print("=" * 130)
    print("Alpha Futures V47 — Opening Range Breakout + OI Confirmation")
    print("Breakout with rising OI = institutional conviction. Follow the breakout.")
    print("=" * 130)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    print(f"  {NS} commodities, {ND} days")

    # ========================================
    # PRECOMPUTE
    # ========================================
    print("\n[Signals] Computing range, OI, ATR, momentum, N-day highs...", flush=True)
    t0 = time.time()

    # 1. Yesterday's range: prev_high, prev_low
    prev_high = np.full((NS, ND), np.nan)
    prev_low = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            h_prev = H[si, di - 1]
            l_prev = L[si, di - 1]
            if not np.isnan(h_prev):
                prev_high[si, di] = h_prev
            if not np.isnan(l_prev):
                prev_low[si, di] = l_prev

    # 2. Range size as percentage of close
    range_pct = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            ph = prev_high[si, di]
            pl = prev_low[si, di]
            pc = C[si, di - 1]
            if not np.isnan(ph) and not np.isnan(pl) and not np.isnan(pc) and pc > 0:
                range_pct[si, di] = (ph - pl) / pc * 100

    # Average range_pct over 20 days
    avg_range_pct_20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            vals = range_pct[si, di - 20:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 10:
                avg_range_pct_20[si, di] = np.mean(valid)

    # 3. OI change (day-over-day)
    oi_chg = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            oi_now = OI[si, di]
            oi_prev = OI[si, di - 1]
            if not np.isnan(oi_now) and not np.isnan(oi_prev) and oi_prev > 0:
                oi_chg[si, di] = (oi_now - oi_prev) / oi_prev

    # 4. OI EMA (5-day) and OI rising flag
    oi_ema = np.full((NS, ND), np.nan)
    oi_rising = np.full((NS, ND), False)
    for si in range(NS):
        oe = 0.0
        alpha_oi = 2.0 / 6  # 5-day EMA
        first_val = True
        for di in range(ND):
            oi_val = OI[si, di]
            if np.isnan(oi_val):
                continue
            if first_val:
                oe = oi_val
                first_val = False
            else:
                oe = alpha_oi * oi_val + (1 - alpha_oi) * oe
            oi_ema[si, di] = oe
        # OI rising: current EMA > EMA 5 days ago by >1%
        for di in range(6, ND):
            cur = oi_ema[si, di]
            prev = oi_ema[si, di - 5]
            if not np.isnan(cur) and not np.isnan(prev) and prev > 0:
                oi_rising[si, di] = (cur - prev) / prev > 0.01

    # 5. ATR (10-day)
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

    # 6. N-day high/low
    nday_high = {}
    nday_low = {}
    for n in [5, 10, 20]:
        nh = np.full((NS, ND), np.nan)
        nl = np.full((NS, ND), np.nan)
        for si in range(NS):
            for di in range(n, ND):
                hw = H[si, di - n:di]
                lw = L[si, di - n:di]
                hv = hw[~np.isnan(hw)]
                lv = lw[~np.isnan(lw)]
                if len(hv) >= n * 0.5:
                    nh[si, di] = np.max(hv)
                if len(lv) >= n * 0.5:
                    nl[si, di] = np.min(lv)
        nday_high[n] = nh
        nday_low[n] = nl

    # 7. 5-day momentum
    mom5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            c_now = C[si, di]
            c_prev = C[si, di - 5]
            if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                mom5[si, di] = (c_now - c_prev) / c_prev

    # 8. Body ratio: |C - O| / (H - L)
    body_ratio = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            c_val = C[si, di]
            o_val = O[si, di]
            h_val = H[si, di]
            l_val = L[si, di]
            if np.isnan(c_val) or np.isnan(o_val) or np.isnan(h_val) or np.isnan(l_val):
                continue
            rng = h_val - l_val
            if rng > 0:
                body_ratio[si, di] = abs(c_val - o_val) / rng

    print(f"  All signals computed ({time.time() - t0:.1f}s)", flush=True)

    # ========================================
    # SCORING FUNCTIONS (5 signals)
    # ========================================

    def make_signal1_prev_break(use_oi=True, scale=10.0):
        """Signal 1: Yesterday-High Breakout.
        Close > prev_high. If use_oi, also require OI rising.
        Score = breakout margin * scale.
        """
        def score(si, di):
            c_val = C[si, di]
            ph = prev_high[si, di]
            if np.isnan(c_val) or np.isnan(ph) or ph <= 0:
                return np.nan
            if c_val <= ph:
                return np.nan
            if use_oi and not oi_rising[si, di]:
                return np.nan
            margin = (c_val - ph) / ph
            return np.clip(margin * scale, 0, 1)
        return score

    def make_signal2_nday_break(n=10, use_oi=True, scale=10.0):
        """Signal 2: N-day Breakout.
        Close > nday_high[n]. If use_oi, also require OI rising.
        Score = breakout margin * scale.
        """
        nh = nday_high[n]
        def score(si, di):
            c_val = C[si, di]
            nh_val = nh[si, di]
            if np.isnan(c_val) or np.isnan(nh_val) or nh_val <= 0:
                return np.nan
            if c_val <= nh_val:
                return np.nan
            if use_oi and not oi_rising[si, di]:
                return np.nan
            margin = (c_val - nh_val) / nh_val
            return np.clip(margin * scale, 0, 1)
        return score

    def make_signal3_range_expand(use_oi=True, scale=10.0):
        """Signal 3: Range Expansion Breakout.
        Range > 2x average range AND close > prev_high.
        If use_oi, also require OI rising.
        """
        def score(si, di):
            c_val = C[si, di]
            ph = prev_high[si, di]
            rp = range_pct[si, di]
            arp = avg_range_pct_20[si, di]
            if np.isnan(c_val) or np.isnan(ph) or ph <= 0:
                return np.nan
            if c_val <= ph:
                return np.nan
            if np.isnan(rp) or np.isnan(arp) or arp <= 0:
                return np.nan
            if rp <= 2.0 * arp:
                return np.nan
            if use_oi and not oi_rising[si, di]:
                return np.nan
            margin = (c_val - ph) / ph
            expansion = rp / arp
            return np.clip(margin * expansion * scale, 0, 1)
        return score

    def make_signal4_body_break(use_oi=True, body_threshold=0.7, scale=10.0):
        """Signal 4: Body Ratio Breakout.
        Body ratio > threshold AND close > prev_high.
        If use_oi, also require OI rising.
        """
        def score(si, di):
            c_val = C[si, di]
            ph = prev_high[si, di]
            br = body_ratio[si, di]
            if np.isnan(c_val) or np.isnan(ph) or ph <= 0:
                return np.nan
            if c_val <= ph:
                return np.nan
            if np.isnan(br) or br < body_threshold:
                return np.nan
            if use_oi and not oi_rising[si, di]:
                return np.nan
            margin = (c_val - ph) / ph
            return np.clip(margin * br * scale, 0, 1)
        return score

    def make_signal5_mom_break(use_oi=True, scale=10.0):
        """Signal 5: Breakout + Momentum Confirmation.
        Close > prev_high AND mom5 > 0.
        If use_oi, also require OI rising.
        """
        def score(si, di):
            c_val = C[si, di]
            ph = prev_high[si, di]
            m5 = mom5[si, di]
            if np.isnan(c_val) or np.isnan(ph) or ph <= 0:
                return np.nan
            if c_val <= ph:
                return np.nan
            if np.isnan(m5) or m5 <= 0:
                return np.nan
            if use_oi and not oi_rising[si, di]:
                return np.nan
            margin = (c_val - ph) / ph
            return np.clip(margin * (1 + m5 * 5) * scale, 0, 1)
        return score

    # ========================================
    # BACKTEST ENGINE
    # ========================================
    def run_backtest(score_fn, name, top_n=1, hold_max=5,
                     trail_atr_mult=3.0, use_prev_low_exit=False,
                     use_signal_flip=False, wf_split_year=None):
        """Single position per symbol, long only, cash/(price*mult) lots."""
        cash = float(CASH0)
        trades = []
        positions = []

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

                # Below yesterday's low exit (conservative)
                if exit_reason is None and use_prev_low_exit and days_held >= 1:
                    pl = prev_low[pos['si'], di]
                    if not np.isnan(pl) and c < pl:
                        exit_reason = 'prev_low'

                # Signal flip: close < prev_low on any subsequent day
                if exit_reason is None and use_signal_flip and days_held >= 2:
                    pl = prev_low[pos['si'], di]
                    if not np.isnan(pl) and c < pl:
                        exit_reason = 'sig_flip'

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
                        trail_price = c - trail_atr_mult * atr_val if atr_val > 0 else c * 0.97
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
        if wf_split_year:
            first_test_di = None
            for d in range(MIN_TRAIN, ND):
                if dates[d].year >= wf_split_year:
                    first_test_di = d
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
    # PARAMETER SWEEP
    # ========================================
    print("\n[Backtest] Building configurations...", flush=True)
    configs = []

    # Signal 1: Yesterday-High Breakout
    for use_oi in [True, False]:
        for hold in [3, 5, 7]:
            for trail in [2.0, 3.0, 4.0]:
                for tn in [1, 3]:
                    oi_tag = 'OI' if use_oi else 'noOI'
                    configs.append((
                        make_signal1_prev_break(use_oi=use_oi),
                        f"S1_PrevBrk_{oi_tag}_H{hold}_TR{trail:.0f}_N{tn}",
                        tn, hold, trail, False, False, None
                    ))

    # Signal 2: N-day Breakout
    for n in [5, 10, 20]:
        for use_oi in [True, False]:
            for hold in [3, 5, 7]:
                for trail in [2.0, 3.0, 4.0]:
                    for tn in [1, 3]:
                        oi_tag = 'OI' if use_oi else 'noOI'
                        configs.append((
                            make_signal2_nday_break(n=n, use_oi=use_oi),
                            f"S2_N{n}Brk_{oi_tag}_H{hold}_TR{trail:.0f}_N{tn}",
                            tn, hold, trail, False, False, None
                        ))

    # Signal 3: Range Expansion Breakout
    for use_oi in [True, False]:
        for hold in [3, 5, 7]:
            for trail in [2.0, 3.0, 4.0]:
                for tn in [1, 3]:
                    oi_tag = 'OI' if use_oi else 'noOI'
                    configs.append((
                        make_signal3_range_expand(use_oi=use_oi),
                        f"S3_RangeExp_{oi_tag}_H{hold}_TR{trail:.0f}_N{tn}",
                        tn, hold, trail, False, False, None
                    ))

    # Signal 4: Body Ratio Breakout
    for use_oi in [True, False]:
        for hold in [3, 5, 7]:
            for trail in [2.0, 3.0, 4.0]:
                for tn in [1, 3]:
                    oi_tag = 'OI' if use_oi else 'noOI'
                    configs.append((
                        make_signal4_body_break(use_oi=use_oi),
                        f"S4_BodyBrk_{oi_tag}_H{hold}_TR{trail:.0f}_N{tn}",
                        tn, hold, trail, False, False, None
                    ))

    # Signal 5: Breakout + Momentum
    for use_oi in [True, False]:
        for hold in [3, 5, 7]:
            for trail in [2.0, 3.0, 4.0]:
                for tn in [1, 3]:
                    oi_tag = 'OI' if use_oi else 'noOI'
                    configs.append((
                        make_signal5_mom_break(use_oi=use_oi),
                        f"S5_MomBrk_{oi_tag}_H{hold}_TR{trail:.0f}_N{tn}",
                        tn, hold, trail, False, False, None
                    ))

    print(f"  {len(configs)} full-period configurations", flush=True)

    # Walk-forward configs for best parameter combos
    wf_configs = []
    for sig_num in [1, 2, 3, 4, 5]:
        for n in [5, 10, 20] if sig_num == 2 else [10]:
            for use_oi in [True, False]:
                for hold in [3, 5]:
                    for trail in [2.0, 3.0]:
                        for tn in [1, 3]:
                            for wf_year in [2023, 2024]:
                                oi_tag = 'OI' if use_oi else 'noOI'
                                if sig_num == 1:
                                    fn = make_signal1_prev_break(use_oi=use_oi)
                                    label = f"S1_PrevBrk_{oi_tag}"
                                elif sig_num == 2:
                                    fn = make_signal2_nday_break(n=n, use_oi=use_oi)
                                    label = f"S2_N{n}Brk_{oi_tag}"
                                elif sig_num == 3:
                                    fn = make_signal3_range_expand(use_oi=use_oi)
                                    label = f"S3_RangeExp_{oi_tag}"
                                elif sig_num == 4:
                                    fn = make_signal4_body_break(use_oi=use_oi)
                                    label = f"S4_BodyBrk_{oi_tag}"
                                else:
                                    fn = make_signal5_mom_break(use_oi=use_oi)
                                    label = f"S5_MomBrk_{oi_tag}"
                                wf_configs.append((
                                    fn,
                                    f"{label}_H{hold}_TR{trail:.0f}_N{tn}_WF{wf_year}",
                                    tn, hold, trail, False, False, wf_year
                                ))

    all_configs = configs + wf_configs
    print(f"  {len(all_configs)} total configurations ({len(configs)} full + {len(wf_configs)} WF)", flush=True)

    # ========================================
    # RUN BACKTESTS
    # ========================================
    print("\n[Backtest] Running...", flush=True)
    results = []
    t_backtest_start = time.time()

    for ci, (fn, name, tn, hmax, trail, use_pl, use_sf, wf) in enumerate(all_configs):
        r = run_backtest(fn, name, top_n=tn, hold_max=hmax,
                         trail_atr_mult=trail,
                         use_prev_low_exit=use_pl,
                         use_signal_flip=use_sf,
                         wf_split_year=wf)
        if r and r['ann'] > 0:
            results.append(r)
            if r['ann'] > 50:
                parts = []
                for reason, stats in sorted(r['reasons'].items()):
                    wr_r = stats['w'] / stats['n'] * 100 if stats['n'] > 0 else 0
                    parts.append(f"{reason}:{stats['n']}({wr_r:.0f}%)")
                print(f"  {r['name']:55s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                      f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                      f"AvgW {r['avg_win']:+.2f}% | AvgL {r['avg_loss']:.2f}% | AvgD {r['avg_days']:.1f}")
                print(f"  {'':55s} | Exits: {' | '.join(parts)}")

        if (ci + 1) % 50 == 0:
            elapsed = time.time() - t_backtest_start
            rate = (ci + 1) / elapsed
            eta = (len(all_configs) - ci - 1) / rate
            print(f"  [{ci + 1}/{len(all_configs)}] {len(results)} profitable "
                  f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)

    print(f"  Backtests done ({time.time() - t_backtest_start:.1f}s)", flush=True)

    # ========================================
    # RESULTS
    # ========================================
    results.sort(key=lambda x: -x['ann'])

    wf_results = [r for r in results if '_WF' in r['name']]
    full_results = [r for r in results if '_WF' not in r['name']]

    # --- Top 20 full-period ---
    print(f"\n{'=' * 130}")
    print(f"  TOP 20 FULL-PERIOD RESULTS")
    print(f"{'=' * 130}")
    print(f"  {'Strategy':55s} | {'Ann':>7s} | {'WR':>5s} | {'N':>4s} | {'DD':>6s} | "
          f"{'PF':>4s} | {'AvgW':>7s} | {'AvgL':>6s} | {'AvgD':>4s}")
    print(f"  {'-' * 130}")
    for r in full_results[:20]:
        print(f"  {r['name']:55s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:4.2f} | "
              f"{r['avg_win']:+6.2f}% | {r['avg_loss']:5.2f}% | "
              f"{r['avg_days']:4.1f}")

    # --- Top 10 walk-forward ---
    if wf_results:
        wf_results.sort(key=lambda x: -x['ann'])
        print(f"\n  TOP 10 WALK-FORWARD RESULTS (out-of-sample)")
        print(f"  {'-' * 130}")
        for r in wf_results[:10]:
            print(f"  {r['name']:55s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
                  f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:4.2f}")

    # --- Best per signal ---
    print(f"\n  BEST PER SIGNAL:")
    for sig_num in [1, 2, 3, 4, 5]:
        sig_results = [r for r in full_results if r['name'].startswith(f'S{sig_num}_')]
        if sig_results:
            best = sig_results[0]
            print(f"    Signal {sig_num}: {best['name']:55s} | "
                  f"Ann {best['ann']:+7.1f}% | WR {best['wr']:5.1f}% | "
                  f"N {best['n']:4d} | DD {best['dd']:6.1f}% | PF {best['pf']:4.2f}")

    # --- OI effect comparison ---
    print(f"\n  OI CONFIRMATION EFFECT:")
    for sig_num in [1, 2, 3, 4, 5]:
        oi_results = [r for r in full_results if r['name'].startswith(f'S{sig_num}_') and '_OI_' in r['name']]
        nooi_results = [r for r in full_results if r['name'].startswith(f'S{sig_num}_') and '_noOI_' in r['name']]
        oi_best_ann = oi_results[0]['ann'] if oi_results else float('nan')
        nooi_best_ann = nooi_results[0]['ann'] if nooi_results else float('nan')
        oi_avg = np.mean([r['ann'] for r in oi_results[:5]]) if oi_results else float('nan')
        nooi_avg = np.mean([r['ann'] for r in nooi_results[:5]]) if nooi_results else float('nan')
        print(f"    Signal {sig_num}: OI_best={oi_best_ann:+.1f}% vs noOI_best={nooi_best_ann:+.1f}% | "
              f"OI_top5avg={oi_avg:+.1f}% vs noOI_top5avg={nooi_avg:+.1f}%")

    # --- Best config detail ---
    if full_results:
        best = full_results[0]
        print(f"\n{'=' * 130}")
        print(f"  BEST CONFIG DETAIL")
        print(f"  {best['name']}")
        print(f"  Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  N={best['n']}  "
              f"DD={best['dd']:.1f}%  PF={best['pf']:.2f}")
        print(f"  AvgWin={best['avg_win']:+.2f}%  AvgLoss={best['avg_loss']:.2f}%  "
              f"AvgDays={best['avg_days']:.1f}  Final={best['cash']:.0f}")
        print(f"{'=' * 130}")

        print(f"\n  EXIT REASON BREAKDOWN:")
        for reason, s in sorted(best['reasons'].items(), key=lambda x: -x[1]['n']):
            rwr = s['w'] / max(s['n'], 1) * 100
            print(f"    {reason:12s}: {s['n']:4d} trades  WR={rwr:5.1f}%  PnL={s['pnl']:+.1f}%")

        print(f"\n  YEARLY BREAKDOWN:")
        for y in sorted(best['yearly'].keys()):
            s = best['yearly'][y]
            wr_y = s['w'] / max(s['n'], 1) * 100
            print(f"    {y}: {s['n']:3d} trades  WR={wr_y:5.1f}%  PnL={s['pnl']:+.1f}%")

        print(f"\n  GROUP BREAKDOWN:")
        for g in sorted(best['grp_counts'].keys(), key=lambda x: -best['grp_counts'][x]['n']):
            gs = best['grp_counts'][g]
            wr_g = gs['w'] / max(gs['n'], 1) * 100
            print(f"    {g:15s}: {gs['n']:3d}t  WR={wr_g:5.1f}%  Abs={gs['pnl']:+.0f}")

    # --- Yearly for top 5 ---
    if len(full_results) >= 2:
        print(f"\n  YEARLY BREAKDOWN FOR TOP 5:")
        for idx, r in enumerate(full_results[:5]):
            print(f"\n  #{idx + 1}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.1f}%, DD={r['dd']:.1f}%)")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:3d}t  WR={wr_y:5.1f}%  PnL={ys['pnl']:+.1f}%")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
