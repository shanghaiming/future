"""
Alpha Futures V35 — Signature-Based Features Strategy
======================================================
Based on arXiv 2503.00603: path signatures capture geometric features of price
paths (V-shape reversal, double top, triangle consolidation) with 15-20% better
accuracy than manual technical indicators.

Path Signatures: iterated integrals that uniquely characterize path geometry.
For a 2D path (t, price), signatures at various levels capture:
  - Level 1: trend direction and magnitude
  - Level 2: path curvature / area under curve
  - Level 3: acceleration, higher-order shape features

Simplified practical implementation uses 8 signature-derived features computed
over a rolling window:
  sig_1: total return (net displacement)
  sig_2: cumulative area (path curvature measure)
  sig_3: total path length (volatility measure)
  sig_4: max drawdown within window
  sig_5: path efficiency (KER = Kaufman Efficiency Ratio)
  sig_6: curvature (sum of return-difference squared, acceleration)
  sig_7: sign flips (direction change count)
  sig_8: largest single day concentration

These features identify high-probability entry patterns:
  - V-shape recovery (downtrend + sharp reversal)
  - Breakout acceleration (increasing momentum)
  - Exhaustion avoidance (single-day dominated moves rejected)

Signal combination:
  Primary: signature pattern score
  Confirm: VDP direction, OI rising
  Filter: volume above average

Data: alpha_v2.load_all_data(load_oi=True)
Backtest: single position, long only, trailing stop, time exit, signal flip exit.
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


# ============================================================
# PATH SIGNATURE COMPUTATION
# ============================================================

def compute_signature_features(C, NS, ND, window):
    """
    Compute 8 path signature features for all stocks over a rolling window.

    Returns dict of 8 arrays, each shape (NS, ND).
    """
    # Daily log returns
    ret = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            c1 = C[si, di]
            c0 = C[si, di - 1]
            if not np.isnan(c1) and not np.isnan(c0) and c0 > 0:
                ret[si, di] = (c1 - c0) / c0

    # Signature features
    sig1_net_ret = np.full((NS, ND), np.nan)       # total return over window
    sig2_cum_area = np.full((NS, ND), np.nan)       # sum of daily returns (curvature)
    sig3_path_len = np.full((NS, ND), np.nan)       # sum of |daily returns| (volatility)
    sig4_max_dd = np.full((NS, ND), np.nan)         # max drawdown within window
    sig5_efficiency = np.full((NS, ND), np.nan)      # |net_return| / sum(|returns|) = KER
    sig6_curvature = np.full((NS, ND), np.nan)       # sum of (ret[t] - ret[t-1])^2
    sig7_sign_flips = np.full((NS, ND), np.nan)      # count of direction changes
    sig8_concentration = np.full((NS, ND), np.nan)   # max(|ret|) / sum(|ret|)

    w = window
    for si in range(NS):
        for di in range(w, ND):
            # Check we have enough valid data in the window
            r_slice = ret[si, di - w + 1: di + 1]
            valid = r_slice[~np.isnan(r_slice)]
            if len(valid) < w - 2:
                continue

            # Fill nan with 0 for computation (small gaps)
            r = np.where(np.isnan(r_slice), 0.0, r_slice)

            # sig1: net return
            net = np.sum(r)
            sig1_net_ret[si, di] = net

            # sig2: cumulative area (sum of returns, measures curvature)
            sig2_cum_area[si, di] = np.sum(r)

            # sig3: total path length (sum of |returns|)
            abs_sum = np.sum(np.abs(r))
            sig3_path_len[si, di] = abs_sum

            # sig4: max drawdown within window (based on cumulative return)
            cum = np.cumsum(r)
            running_max = np.maximum.accumulate(cum)
            # Drawdown at each point relative to running peak
            dd = cum - running_max
            # Max drawdown is the most negative value
            if len(dd) > 0:
                sig4_max_dd[si, di] = np.min(dd)

            # sig5: efficiency = |net| / sum(|r|)  (KER analog)
            if abs_sum > 0:
                sig5_efficiency[si, di] = abs(net) / abs_sum
            else:
                sig5_efficiency[si, di] = 0.0

            # sig6: curvature = sum of (r[t] - r[t-1])^2
            if len(r) >= 2:
                diffs = np.diff(r)
                sig6_curvature[si, di] = np.sum(diffs ** 2)

            # sig7: sign flips
            if len(r) >= 2:
                signs = np.sign(r)
                # Skip zeros
                nonzero_signs = signs[signs != 0]
                if len(nonzero_signs) >= 2:
                    flips = np.sum(np.diff(nonzero_signs) != 0)
                    sig7_sign_flips[si, di] = flips
                else:
                    sig7_sign_flips[si, di] = 0

            # sig8: concentration = max(|r|) / sum(|r|)
            if abs_sum > 0:
                sig8_concentration[si, di] = np.max(np.abs(r)) / abs_sum
            else:
                sig8_concentration[si, di] = 0.0

    return {
        'net_ret': sig1_net_ret,
        'cum_area': sig2_cum_area,
        'path_len': sig3_path_len,
        'max_dd': sig4_max_dd,
        'efficiency': sig5_efficiency,
        'curvature': sig6_curvature,
        'sign_flips': sig7_sign_flips,
        'concentration': sig8_concentration,
    }


def main():
    t_start = time.time()
    print("=" * 120)
    print("Alpha Futures V35 — Path Signature Features Strategy")
    print("Core: 8 signature features capture geometric path shape for pattern recognition")
    print("=" * 120)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)

    sym_to_si = {syms[si]: si for si in range(NS)}
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
    # PRECOMPUTE ALL FEATURES
    # ========================================
    print("\n[Signals] Computing signature features...", flush=True)
    t0 = time.time()

    # Compute signature features at multiple windows
    sig_data = {}
    for win in [5, 10, 15]:
        print(f"  Window={win}...", end="", flush=True)
        sig_data[win] = compute_signature_features(C, NS, ND, win)
        print(" done", flush=True)

    # VDP EMA
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

    # OI EMA trend
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

    # Volume moving average (20-day)
    vol_ma = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            vs = V[si, di - 19: di + 1]
            valid = vs[~np.isnan(vs)]
            if len(valid) >= 10:
                vol_ma[si, di] = np.mean(valid)

    # ATR (10-day)
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

    print(f"  All features computed ({time.time()-t0:.1f}s)", flush=True)

    # ========================================
    # SCORING FUNCTIONS
    # ========================================

    def make_signature_score(window=10, pattern='combined', require_vdp=False,
                             require_oi=False, use_group_rank=False,
                             min_eff=0.0, max_conc=1.0):
        """
        Build a signature-based scoring function.

        pattern: 'vshape', 'breakout', 'momentum', 'combined', 'broad'
          - vshape:    V-shape recovery (dip then sharp reversal)
          - breakout:  Smooth directional move with high efficiency
          - momentum:  Positive trend with moderate features
          - combined:  Weighted sum of all pattern scores
          - broad:     Looser thresholds for more trades

        min_eff: minimum efficiency (KER) threshold
        max_conc: maximum single-day concentration allowed
        """
        sig = sig_data[window]

        def score(si, di):
            net = sig['net_ret'][si, di]
            path_len = sig['path_len'][si, di]
            max_dd = sig['max_dd'][si, di]
            eff = sig['efficiency'][si, di]
            curv = sig['curvature'][si, di]
            flips = sig['sign_flips'][si, di]
            conc = sig['concentration'][si, di]

            # Need valid data
            if np.isnan(net) or np.isnan(path_len) or path_len <= 0:
                return np.nan
            if np.isnan(eff):
                return np.nan

            # Pre-filters
            if eff < min_eff:
                return np.nan
            if not np.isnan(conc) and conc > max_conc:
                return np.nan

            sc = 0.0

            if pattern == 'vshape':
                # === V-Shape Recovery Pattern ===
                # Requires: path dipped then recovered strongly
                if np.isnan(max_dd):
                    return np.nan
                if max_dd > -0.005:
                    return np.nan  # must have a meaningful dip
                if net <= 0:
                    return np.nan  # must recover to positive

                # V-shape strength: ratio of recovery to dip
                dip = abs(max_dd)
                recovery = net
                # V-shape quality: recovered from the dip
                v_ratio = recovery / dip if dip > 0 else 0
                sc = v_ratio * 0.5

                # Bonus: fewer sign flips (clean reversal, not whipsawed)
                if not np.isnan(flips):
                    flip_ratio = flips / max(window - 1, 1)
                    if flip_ratio < 0.3:
                        sc += 0.2  # clean reversal
                    elif flip_ratio > 0.6:
                        sc -= 0.15  # choppy

                # Bonus: low curvature (smooth path shape)
                if not np.isnan(curv) and curv < 0.003:
                    sc += 0.1

                # Size: scale by dip magnitude (deeper dips = stronger signal)
                sc += min(dip * 5, 0.3)

            elif pattern == 'breakout':
                # === Breakout Acceleration Pattern ===
                # Requires: positive, efficient, smooth directional move
                if net <= 0.002:
                    return np.nan
                if eff < 0.25:
                    return np.nan  # must be reasonably efficient

                # Core: net return * efficiency (strong trend)
                sc = net * 8.0 + eff * 0.5

                # Smooth move bonus (low curvature)
                if not np.isnan(curv):
                    if curv < 0.001:
                        sc += 0.3  # very smooth
                    elif curv < 0.005:
                        sc += 0.15  # reasonably smooth
                    elif curv > 0.015:
                        sc -= 0.2  # choppy

                # Low concentration bonus (distributed move, not one-day wonder)
                if not np.isnan(conc) and conc < 0.25:
                    sc += 0.15

                # Moderate sign flips
                if not np.isnan(flips) and flips < window * 0.3:
                    sc += 0.1

            elif pattern == 'momentum':
                # === Simple Momentum Pattern ===
                # Requires: positive net return with sufficient volatility
                if net <= 0:
                    return np.nan

                # Base: net return
                sc = net * 6.0

                # Efficiency boost
                sc += eff * 0.3

                # Not too many flips
                if not np.isnan(flips) and flips < window * 0.4:
                    sc += 0.15

                # Not exhausted
                if not np.isnan(conc) and conc < 0.35:
                    sc += 0.1

            elif pattern == 'broad':
                # === Broad Pattern (loose thresholds) ===
                # Accept any positive net return and score by feature quality
                if net <= 0:
                    return np.nan
                if eff < 0.1:
                    return np.nan

                sc = 0.1  # base score for positive return

                # Feature bonuses
                sc += min(net * 3, 0.4)  # trend
                sc += min(eff * 0.2, 0.2)  # efficiency

                if not np.isnan(conc):
                    if conc < 0.3:
                        sc += 0.1
                    elif conc > 0.5:
                        sc -= 0.1

                if not np.isnan(flips):
                    if flips < window * 0.3:
                        sc += 0.1
                    elif flips > window * 0.7:
                        sc -= 0.1

                # V-shape bonus if present
                if not np.isnan(max_dd) and max_dd < -0.005 and net > 0.005:
                    sc += 0.15

            else:  # 'combined'
                # === Combined Pattern Score ===
                # Weighted sum of all pattern detectors
                if net <= 0:
                    return np.nan

                # Base momentum component
                sc = net * 5.0

                # Efficiency component
                sc += eff * 0.4

                # V-shape bonus
                if not np.isnan(max_dd) and max_dd < -0.005 and net > 0.005:
                    dip = abs(max_dd)
                    v_ratio = net / dip if dip > 0 else 0
                    sc += v_ratio * 0.25
                    if not np.isnan(flips) and flips < window * 0.4:
                        sc += 0.1

                # Smooth move bonus
                if not np.isnan(curv):
                    if curv < 0.003:
                        sc += 0.15
                    elif curv > 0.015:
                        sc -= 0.1

                # Distribution bonus (anti-exhaustion)
                if not np.isnan(conc):
                    if conc < 0.3:
                        sc += 0.1
                    elif conc > 0.5:
                        sc -= 0.15

                # Direction consistency
                if not np.isnan(flips):
                    if flips < window * 0.3:
                        sc += 0.1
                    elif flips > window * 0.6:
                        sc -= 0.1

            # Optional: VDP confirmation
            if require_vdp:
                vd = vdp_ema[si, di]
                if np.isnan(vd):
                    return np.nan
                if vd < 0:
                    return np.nan  # VDP must be positive for long
                sc *= min(1.0 + abs(vd) / 5e6, 1.5)

            # Optional: OI confirmation
            if require_oi:
                oi_r = oi_rising[si, di]
                if not np.isnan(oi_r):
                    if oi_r > 0.01:
                        sc *= 1.3
                    elif oi_r < -0.02:
                        sc *= 0.5

            # Optional: group-relative ranking
            if use_group_rank:
                grp = GROUP_MAP.get(syms[si])
                if grp and grp in group_members:
                    group_scores = []
                    for sj in group_members[grp]:
                        sj_net = sig['net_ret'][sj, di]
                        if not np.isnan(sj_net):
                            group_scores.append(sj_net)
                    if group_scores:
                        grp_avg = np.mean(group_scores)
                        # Bonus for being above group average
                        if net > grp_avg:
                            sc *= 1.0 + min((net - grp_avg) * 10, 0.5)
                        else:
                            sc *= 0.8  # penalty for lagging group

            # Must be positive to trade
            if sc <= 0.01:
                return np.nan

            # Clip to [-1, 1] range
            return np.clip(sc, -1.0, 1.0)

        return score

    # ========================================
    # BACKTEST ENGINE
    # ========================================
    def run_backtest(score_fn, name, top_n=1, hold_min=2, hold_max=5,
                     trail_atr_mult=2.5, wf_split_year=None):
        """
        Multi-position backtest with optional walk-forward split.
        Single position per symbol, long only, trailing stop, time exit, signal flip exit.
        """
        cash = float(CASH0)
        trades = []
        positions = []
        last_exit = {}

        for di in range(MIN_TRAIN, ND):
            year = dates[di].year

            # Walk-forward: skip training period
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

                # Trailing stop (after at least 2 days)
                if trail_atr_mult > 0 and days_held >= 2:
                    atr = pos.get('atr', 0)
                    if atr > 0 and pos['dir'] == 1:
                        new_trail = c - trail_atr_mult * atr
                        if new_trail > pos.get('trail_price', pos['entry']):
                            pos['trail_price'] = new_trail
                        if c < pos['trail_price']:
                            exit_reason = 'trail'

                # Signal flip (after min hold)
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

        # Close remaining positions
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

        # Compute equity curve and max drawdown
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
    # PARAMETER SWEEP
    # ========================================
    print("\n[Backtest] Running parameter sweep...", flush=True)
    results = []
    configs = []

    # === Pattern x Window x Hold ===
    for pat in ['vshape', 'breakout', 'momentum', 'combined', 'broad']:
        for win in [5, 10, 15]:
            for hold in [3, 5, 7]:
                configs.append((
                    make_signature_score(window=win, pattern=pat),
                    f"{pat[:4].upper()}_W{win}_H{hold}",
                    1, 2, hold, 2.5, None
                ))

    # === VDP confirmation (best patterns) ===
    for pat in ['vshape', 'breakout', 'combined']:
        for win in [5, 10, 15]:
            for hold in [3, 5, 7]:
                configs.append((
                    make_signature_score(window=win, pattern=pat, require_vdp=True),
                    f"{pat[:4].upper()}_W{win}_H{hold}_VDP",
                    1, 2, hold, 2.5, None
                ))

    # === OI confirmation ===
    for pat in ['vshape', 'breakout', 'combined']:
        for win in [5, 10, 15]:
            for hold in [3, 5, 7]:
                configs.append((
                    make_signature_score(window=win, pattern=pat, require_oi=True),
                    f"{pat[:4].upper()}_W{win}_H{hold}_OI",
                    1, 2, hold, 2.5, None
                ))

    # === Group relative ranking ===
    for pat in ['vshape', 'breakout', 'combined']:
        for win in [5, 10, 15]:
            for hold in [3, 5, 7]:
                configs.append((
                    make_signature_score(window=win, pattern=pat, use_group_rank=True),
                    f"{pat[:4].upper()}_W{win}_H{hold}_GRP",
                    1, 2, hold, 2.5, None
                ))

    # === VDP + OI combined ===
    for pat in ['vshape', 'breakout', 'combined']:
        for win in [5, 10, 15]:
            for hold in [3, 5, 7]:
                configs.append((
                    make_signature_score(window=win, pattern=pat,
                                         require_vdp=True, require_oi=True),
                    f"{pat[:4].upper()}_W{win}_H{hold}_VDP_OI",
                    1, 2, hold, 2.5, None
                ))

    # === All filters: VDP + OI + Group Rank ===
    for pat in ['vshape', 'breakout', 'combined']:
        for win in [5, 10, 15]:
            for hold in [3, 5, 7]:
                configs.append((
                    make_signature_score(window=win, pattern=pat,
                                         require_vdp=True, require_oi=True,
                                         use_group_rank=True),
                    f"{pat[:4].upper()}_W{win}_H{hold}_ALL",
                    1, 2, hold, 2.5, None
                ))

    # === Trail ATR sweep for best patterns ===
    for pat in ['vshape', 'breakout', 'combined']:
        for win in [10]:
            for hold in [3, 5, 7]:
                for trail in [2.0, 3.0]:
                    configs.append((
                        make_signature_score(window=win, pattern=pat),
                        f"{pat[:4].upper()}_W{win}_H{hold}_TR{trail*10:.0f}",
                        1, 2, hold, trail, None
                    ))

    # === Min efficiency filter ===
    for pat in ['combined', 'breakout']:
        for win in [10]:
            for hold in [3, 5, 7]:
                for min_e in [0.2, 0.3]:
                    configs.append((
                        make_signature_score(window=win, pattern=pat, min_eff=min_e),
                        f"{pat[:4].upper()}_W{win}_H{hold}_ME{min_e*100:.0f}",
                        1, 2, hold, 2.5, None
                    ))

    # === Max concentration filter ===
    for pat in ['combined', 'broad']:
        for win in [10]:
            for hold in [3, 5, 7]:
                for max_c in [0.4, 0.5]:
                    configs.append((
                        make_signature_score(window=win, pattern=pat, max_conc=max_c),
                        f"{pat[:4].upper()}_W{win}_H{hold}_MC{max_c*100:.0f}",
                        1, 2, hold, 2.5, None
                    ))

    # === Walk-forward validation ===
    for pat in ['vshape', 'breakout', 'combined', 'momentum']:
        for win in [5, 10, 15]:
            for hold in [3, 5, 7]:
                for wf_year in [2023, 2025]:
                    configs.append((
                        make_signature_score(window=win, pattern=pat),
                        f"{pat[:4].upper()}_W{win}_H{hold}_WF{wf_year}",
                        1, 2, hold, 2.5, wf_year
                    ))

    # === Walk-forward with all filters ===
    for pat in ['vshape', 'breakout', 'combined']:
        for win in [10]:
            for hold in [3, 5]:
                for wf_year in [2023, 2025]:
                    configs.append((
                        make_signature_score(window=win, pattern=pat,
                                             require_vdp=True, require_oi=True,
                                             use_group_rank=True),
                        f"{pat[:4].upper()}_W{win}_H{hold}_ALL_WF{wf_year}",
                        1, 2, hold, 2.5, wf_year
                    ))

    # === Top-N=2 variants ===
    for pat in ['vshape', 'breakout', 'combined']:
        for win in [10]:
            for hold in [3, 5, 7]:
                configs.append((
                    make_signature_score(window=win, pattern=pat),
                    f"{pat[:4].upper()}_W{win}_H{hold}_N2",
                    2, 2, hold, 2.5, None
                ))

    print(f"  {len(configs)} configurations", flush=True)

    for ci, (fn, name, tn, hmin, hmax, trail, wf) in enumerate(configs):
        r = run_backtest(fn, name, top_n=tn, hold_min=hmin, hold_max=hmax,
                         trail_atr_mult=trail, wf_split_year=wf)
        if r and r['ann'] > 0:
            results.append(r)
            if r['ann'] > 50:
                parts = []
                for reason, stats in sorted(r['reasons'].items()):
                    wr_r = stats['w'] / stats['n'] * 100 if stats['n'] > 0 else 0
                    parts.append(f"{reason}:{stats['n']}({wr_r:.0f}%)")
                print(f"  {r['name']:45s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                      f"N {r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                      f"AvgW {r['avg_win']:+.2f}% | AvgL {r['avg_loss']:.2f}% | AvgD {r['avg_days']:.1f}")
                print(f"  {'':45s} | Exits: {' | '.join(parts)}")

        if (ci + 1) % 50 == 0:
            print(f"  [{ci+1}/{len(configs)}] {len(results)} profitable", flush=True)

    # ========================================
    # RESULTS
    # ========================================
    results.sort(key=lambda x: -x['ann'])

    # Separate walk-forward results
    wf_results = [r for r in results if '_WF' in r['name']]
    full_results = [r for r in results if '_WF' not in r['name']]

    print(f"\n{'=' * 120}")
    print(f"  TOP 20 FULL-PERIOD RESULTS (sorted by annual return)")
    print(f"{'=' * 120}")
    print(f"  {'Strategy':45s} | {'Ann':>7s} | {'WR':>5s} | {'N':>4s} | {'DD':>6s} | "
          f"{'PF':>4s} | {'AvgW':>7s} | {'AvgL':>6s} | {'AvgD':>4s}")
    print(f"  {'-' * 120}")
    for r in full_results[:20]:
        print(f"  {r['name']:45s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:4.2f} | "
              f"{r['avg_win']:+6.2f}% | {r['avg_loss']:5.2f}% | "
              f"{r['avg_days']:4.1f}")

    # Walk-forward results
    if wf_results:
        wf_results.sort(key=lambda x: -x['ann'])
        print(f"\n  WALK-FORWARD RESULTS (out-of-sample)")
        print(f"  {'-' * 120}")
        for r in wf_results[:20]:
            print(f"  {r['name']:45s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
                  f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:4.2f}")

    # Best config details
    if full_results:
        best = full_results[0]
        print(f"\n  BEST: {best['name']}  |  Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  "
              f"N={best['n']}  DD={best['dd']:.1f}%")
        print(f"  AvgWin={best['avg_win']:+.2f}%  AvgLoss={best['avg_loss']:.2f}%  "
              f"PF={best['pf']:.2f}  Final={best['cash']:.0f}")

        print(f"\n  EXIT REASON BREAKDOWN:")
        for reason, s in sorted(best['reasons'].items(), key=lambda x: -x[1]['n']):
            rwr = s['w'] / max(s['n'], 1) * 100
            print(f"    {reason:12s}: {s['n']:4d} trades  WR={rwr:5.1f}%  PnL={s['pnl']:+.1f}%")

        print(f"\n  YEARLY BREAKDOWN (BEST):")
        for y in sorted(best['yearly'].keys()):
            s = best['yearly'][y]
            wr = s['w'] / max(s['n'], 1) * 100
            print(f"    {y}: {s['n']:3d} trades  WR={wr:5.1f}%  PnL={s['pnl']:+.1f}%")

        print(f"\n  GROUP BREAKDOWN (BEST):")
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
