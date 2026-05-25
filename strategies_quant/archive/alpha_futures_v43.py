"""
Alpha Futures V43 — Momentum Spike Mean Reversion
==================================================
Core idea: When a commodity has an extreme 1-2 day move (2+ sigma),
bet on short-term reversal. Large moves overextend due to forced
liquidation, margin calls, or emotional trading. The next 3-5 days
typically see partial reversion.

Completely orthogonal to v34b (group momentum) and v39 (pair trading).

Signal types:
  S1: Single-day spike reversal (z-score of daily return)
  S2: Two-day cumulative spike reversal
  S3: Spike + volume confirmation (exhaustion)
  S4: Spike + group context (spike against group trend)

Exit rules:
  - Time exit: hold 2/3/5 days
  - Take profit: reversion target based on spike magnitude
  - Stop loss: price continues in spike direction by 1 sigma
  - No trailing stop (mean reversion does not benefit from trailing)

Configs: signal_type x spike_threshold x hold_days x top_n
Walk-forward validation on best configs (2023, 2024).
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
    print("=" * 130)
    print("Alpha Futures V43 — Momentum Spike Mean Reversion")
    print("Extreme 1-2 day moves -> bet on short-term reversal")
    print("=" * 130)

    NS, ND, dates, C, O, H, L, V, syms, sym_set = load_all_data(load_oi=False)
    sym_to_si = {syms[si]: si for si in range(NS)}

    # Build group membership
    group_members = {}
    for si in range(NS):
        grp = GROUP_MAP.get(syms[si])
        if grp is None:
            continue
        if grp not in group_members:
            group_members[grp] = []
        group_members[grp].append(si)

    print(f"  {NS} commodities, {ND} days, {len(group_members)} groups")

    # ============================================================
    # PRECOMPUTE ALL SIGNALS
    # ============================================================
    print("\n[Signals] Computing all signals...", flush=True)
    t0 = time.time()

    # 1) Daily returns
    ret = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            c_now = C[si, di]
            c_prev = C[si, di - 1]
            if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                ret[si, di] = (c_now - c_prev) / c_prev

    # 2) Rolling 20-day mean and std of daily returns
    ROLL = 20
    ret_mean = np.full((NS, ND), np.nan)
    ret_std = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ROLL, ND):
            window = ret[si, di - ROLL:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= ROLL * 0.7:
                ret_mean[si, di] = np.mean(valid)
                ret_std[si, di] = np.std(valid, ddof=1)

    # 3) Z-score of daily return
    z_ret = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ROLL, ND):
            if not np.isnan(ret[si, di]) and not np.isnan(ret_mean[si, di]) and not np.isnan(ret_std[si, di]):
                if ret_std[si, di] > 1e-10:
                    z_ret[si, di] = (ret[si, di] - ret_mean[si, di]) / ret_std[si, di]

    # 4) 2-day cumulative return
    ret2 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(2, ND):
            c_now = C[si, di]
            c_prev2 = C[si, di - 2]
            if not np.isnan(c_now) and not np.isnan(c_prev2) and c_prev2 > 0:
                ret2[si, di] = (c_now - c_prev2) / c_prev2

    # 5) Rolling mean and std of 2-day returns
    ret2_mean = np.full((NS, ND), np.nan)
    ret2_std = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ROLL + 2, ND):
            window = ret2[si, di - ROLL:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= ROLL * 0.7:
                ret2_mean[si, di] = np.mean(valid)
                ret2_std[si, di] = np.std(valid, ddof=1)

    # 2-day z-score
    z_ret2 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ROLL + 2, ND):
            if not np.isnan(ret2[si, di]) and not np.isnan(ret2_mean[si, di]) and not np.isnan(ret2_std[si, di]):
                if ret2_std[si, di] > 1e-10:
                    z_ret2[si, di] = (ret2[si, di] - ret2_mean[si, di]) / ret2_std[si, di]

    # 6) Volume rolling mean
    vol_mean20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ROLL, ND):
            window = V[si, di - ROLL:di]
            valid = window[~np.isnan(window)]
            if len(valid) >= ROLL * 0.7:
                vol_mean20[si, di] = np.mean(valid)

    # 7) Group momentum (5-day) for signal 4
    mom5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            c_now = C[si, di]
            c_prev = C[si, di - 5]
            if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0:
                mom5[si, di] = (c_now - c_prev) / c_prev

    grp_mom5 = np.full((NS, ND), np.nan)
    for grp, members in group_members.items():
        for di in range(5, ND):
            ms = []
            for sj in members:
                m = mom5[sj, di]
                if not np.isnan(m):
                    ms.append(m)
            if ms:
                avg_m = np.mean(ms)
                for sj in members:
                    grp_mom5[sj, di] = avg_m

    # 8) ATR for stop loss
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

    print(f"  All signals computed ({time.time()-t0:.1f}s)", flush=True)

    # ============================================================
    # SIGNAL GENERATORS (return direction and strength per si, di)
    # ============================================================

    def signal_s1(si, di, threshold):
        """Signal 1: Single-day spike reversal."""
        z = z_ret[si, di]
        if np.isnan(z):
            return 0, 0.0
        if z > threshold:
            return -1, abs(z)   # extreme up -> short
        if z < -threshold:
            return 1, abs(z)    # extreme down -> long
        return 0, 0.0

    def signal_s2(si, di, threshold):
        """Signal 2: Two-day cumulative spike reversal."""
        z = z_ret2[si, di]
        if np.isnan(z):
            return 0, 0.0
        if z > threshold:
            return -1, abs(z)
        if z < -threshold:
            return 1, abs(z)
        return 0, 0.0

    def signal_s3(si, di, threshold):
        """Signal 3: Spike + volume confirmation (exhaustion)."""
        z = z_ret[si, di]
        if np.isnan(z):
            return 0, 0.0
        v = V[si, di]
        vm = vol_mean20[si, di]
        if np.isnan(v) or np.isnan(vm) or vm <= 0:
            return 0, 0.0
        vol_ratio = v / vm
        if abs(z) > threshold and vol_ratio > 2.0:
            if z > threshold:
                return -1, abs(z) * vol_ratio
            if z < -threshold:
                return 1, abs(z) * vol_ratio
        return 0, 0.0

    def signal_s4(si, di, threshold):
        """Signal 4: Spike + group context."""
        z = z_ret[si, di]
        if np.isnan(z):
            return 0, 0.0
        gm = grp_mom5[si, di]
        # Spike against group trend -> stronger reversal signal
        if z > threshold:
            if np.isnan(gm) or gm < 0:
                return -1, abs(z) * (1.0 + abs(min(gm, 0)) * 10)
            return 0, 0.0
        if z < -threshold:
            if np.isnan(gm) or gm > 0:
                return 1, abs(z) * (1.0 + abs(max(gm, 0)) * 10)
            return 0, 0.0
        return 0, 0.0

    SIGNAL_FNS = {1: signal_s1, 2: signal_s2, 3: signal_s3, 4: signal_s4}
    SIGNAL_NAMES = {1: 'S1_1d', 2: 'S2_2d', 3: 'S3_vol', 4: 'S4_grp'}

    # ============================================================
    # BACKTEST ENGINE
    # ============================================================
    def run_backtest(signal_type, threshold, hold_days, top_n,
                     tp_frac=0.5, sl_sigma=1.0,
                     wf_split_year=None):
        """
        Mean-reversion backtest for spike reversal.
        tp_frac: take profit as fraction of spike magnitude
        sl_sigma: stop loss if price continues by this many sigma
        """
        sig_fn = SIGNAL_FNS[signal_type]
        name = (f"{SIGNAL_NAMES[signal_type]}_Z{threshold:.1f}_"
                f"H{hold_days}_N{top_n}_TP{tp_frac:.1f}_SL{sl_sigma:.1f}")
        if wf_split_year:
            name += f"_WF{wf_split_year}"

        cash = float(CASH0)
        trades = []
        positions = []

        for di in range(MIN_TRAIN, ND):
            year = dates[di].year
            if wf_split_year is not None and year < wf_split_year:
                continue

            # --- Manage existing positions ---
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

                # Take profit: unrealized PnL exceeds tp_frac of spike magnitude
                # spike mag stored as abs(ret) at entry
                if pos.get('spike_mag') and pos['spike_mag'] > 0:
                    tp_target = pos['spike_mag'] * tp_frac
                    reversion_pct = pnl_pct / 100.0
                    if pos['dir'] == 1:
                        # Long: profiting when price goes up after spike down
                        if reversion_pct >= tp_target:
                            exit_reason = 'tp'
                    else:
                        # Short: profiting when price goes down after spike up
                        if reversion_pct >= tp_target:
                            exit_reason = 'tp'

                # Stop loss: price continues in spike direction by sl_sigma * ret_std
                if exit_reason is None and pos.get('entry_std') and pos['entry_std'] > 0:
                    adverse_move = (c - pos['entry']) / pos['entry']
                    if pos['dir'] == 1:
                        # Long: price continues downward
                        if adverse_move < -(pos['entry_std'] * sl_sigma):
                            exit_reason = 'stop'
                    else:
                        # Short: price continues upward
                        if adverse_move > (pos['entry_std'] * sl_sigma):
                            exit_reason = 'stop'

                # Time exit
                if exit_reason is None and days_held >= hold_days:
                    exit_reason = 'time'

                if exit_reason:
                    cost_out = mkt_val * COMM
                    cash += mkt_val - cost_out
                    trades.append({
                        'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                        'days': days_held, 'di': di, 'year': year,
                        'sym': pos['sym'], 'dir': pos['dir'],
                        'reason': exit_reason, 'signal': signal_type,
                    })
                else:
                    new_positions.append(pos)

            positions = new_positions

            # --- Open new positions ---
            n_open = len(positions)
            if n_open < top_n:
                slots = top_n - n_open
                scored = []
                for si in range(NS):
                    direction, strength = sig_fn(si, di, threshold)
                    if direction == 0 or strength <= 0:
                        continue
                    sym = syms[si]
                    if any(p['sym'] == sym for p in positions):
                        continue
                    scored.append((si, direction, strength, sym))

                if scored:
                    scored.sort(key=lambda x: -x[2])
                    cash_per_slot = cash / slots if slots > 0 else cash

                    for best_si, best_dir, best_str, best_sym in scored[:slots]:
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

                        # Store spike magnitude for TP and std for SL
                        spike_mag = abs(ret[best_si, di]) if not np.isnan(ret[best_si, di]) else 0
                        entry_std = ret_std[best_si, di] if not np.isnan(ret_std[best_si, di]) else 0

                        cash -= cost_in
                        positions.append({
                            'si': best_si, 'entry': c, 'entry_di': di,
                            'lots': lots, 'dir': best_dir, 'sym': best_sym,
                            'spike_mag': spike_mag, 'entry_std': entry_std,
                        })

        # Close remaining positions at end
        for pos in positions:
            c = C[pos['si'], ND - 1]
            if np.isnan(c) or c <= 0:
                c = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
            cash += c * mult * pos['lots'] * (1 - COMM)
            trades.append({
                'pnl_pct': pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] > 0 else 0,
                'pnl_abs': pnl, 'days': ND - 1 - pos['entry_di'],
                'di': ND - 1, 'year': dates[ND - 1].year,
                'sym': pos['sym'], 'dir': pos['dir'],
                'reason': 'end', 'signal': signal_type,
            })

        if len(trades) < 5:
            return None

        # Stats
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

        # Direction breakdown
        n_long = sum(1 for t in trades if t['dir'] == 1)
        n_short = sum(1 for t in trades if t['dir'] == -1)
        nw_long = sum(1 for t in trades if t['dir'] == 1 and t['pnl_abs'] > 0)
        nw_short = sum(1 for t in trades if t['dir'] == -1 and t['pnl_abs'] > 0)
        pnl_long = sum(t['pnl_abs'] for t in trades if t['dir'] == 1)
        pnl_short = sum(t['pnl_abs'] for t in trades if t['dir'] == -1)

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
                year_stats[y] = {'n': 0, 'w': 0, 'pnl': 0.0, 'abs': 0.0}
            year_stats[y]['n'] += 1
            if t['pnl_abs'] > 0:
                year_stats[y]['w'] += 1
            year_stats[y]['pnl'] += t['pnl_pct']
            year_stats[y]['abs'] += t['pnl_abs']

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
            'n_long': n_long, 'n_short': n_short,
            'wr_long': round(nw_long / max(n_long, 1) * 100, 1),
            'wr_short': round(nw_short / max(n_short, 1) * 100, 1),
            'pnl_long': round(pnl_long, 0),
            'pnl_short': round(pnl_short, 0),
            'signal_type': signal_type, 'threshold': threshold,
            'hold_days': hold_days, 'top_n': top_n,
            'tp_frac': tp_frac, 'sl_sigma': sl_sigma,
            'reasons': reasons, 'yearly': year_stats, 'grp_counts': grp_counts,
        }

    # ============================================================
    # PARAMETER SWEEP
    # ============================================================
    print("\n[Backtest] Running parameter sweep...", flush=True)
    results = []
    configs = []

    # Main sweep: signal_type x threshold x hold_days x top_n
    for sig_type in [1, 2, 3, 4]:
        for threshold in [1.5, 2.0, 2.5, 3.0]:
            for hold in [2, 3, 5]:
                for top_n in [1, 3]:
                    for tp_frac in [0.5]:
                        for sl_sigma in [1.0]:
                            configs.append({
                                'signal_type': sig_type,
                                'threshold': threshold,
                                'hold_days': hold,
                                'top_n': top_n,
                                'tp_frac': tp_frac,
                                'sl_sigma': sl_sigma,
                                'wf': None,
                            })

    # Extended: add variation on TP and SL for best signal types
    for sig_type in [1, 2, 3]:
        for threshold in [2.0, 2.5]:
            for hold in [3, 5]:
                for tp_frac in [0.3, 0.7]:
                    for sl_sigma in [0.5, 1.5]:
                        configs.append({
                            'signal_type': sig_type,
                            'threshold': threshold,
                            'hold_days': hold,
                            'top_n': 1,
                            'tp_frac': tp_frac,
                            'sl_sigma': sl_sigma,
                            'wf': None,
                        })

    # Walk-forward configs on promising ranges
    for sig_type in [1, 2, 3, 4]:
        for threshold in [1.5, 2.0, 2.5]:
            for hold in [2, 3, 5]:
                for wf_year in [2023, 2024]:
                    configs.append({
                        'signal_type': sig_type,
                        'threshold': threshold,
                        'hold_days': hold,
                        'top_n': 1,
                        'tp_frac': 0.5,
                        'sl_sigma': 1.0,
                        'wf': wf_year,
                    })
                # Also top_n=3
                for wf_year in [2023, 2024]:
                    configs.append({
                        'signal_type': sig_type,
                        'threshold': threshold,
                        'hold_days': hold,
                        'top_n': 3,
                        'tp_frac': 0.5,
                        'sl_sigma': 1.0,
                        'wf': wf_year,
                    })

    print(f"  {len(configs)} configurations", flush=True)

    for ci, cfg in enumerate(configs):
        r = run_backtest(
            signal_type=cfg['signal_type'],
            threshold=cfg['threshold'],
            hold_days=cfg['hold_days'],
            top_n=cfg['top_n'],
            tp_frac=cfg['tp_frac'],
            sl_sigma=cfg['sl_sigma'],
            wf_split_year=cfg['wf'],
        )
        if r and r['ann'] > 0:
            results.append(r)
            if r['ann'] > 30 and not cfg['wf']:
                print(f"  {r['name']:55s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                      f"N={r['n']:4d} | DD {r['dd']:6.1f}% | PF {r['pf']:4.2f} | "
                      f"L={r['n_long']}/S={r['n_short']} | "
                      f"AvgW {r['avg_win']:+.2f}% | AvgL {r['avg_loss']:.2f}%")

        if (ci + 1) % 50 == 0:
            print(f"  [{ci+1}/{len(configs)}] {len(results)} profitable", flush=True)

    # ============================================================
    # RESULTS
    # ============================================================
    results.sort(key=lambda x: -x['ann'])

    wf_results = [r for r in results if '_WF' in r['name']]
    full_results = [r for r in results if '_WF' not in r['name']]

    # --- Top 20 full-period ---
    print(f"\n{'=' * 130}")
    print(f"  TOP 20 FULL-PERIOD RESULTS")
    print(f"{'=' * 130}")
    print(f"  {'Config':55s} | {'Ann':>7s} | {'WR':>5s} | {'N':>4s} | {'DD':>6s} | "
          f"{'PF':>4s} | {'L/S':>7s} | {'AvgW':>7s} | {'AvgL':>6s} | {'AvgD':>4s}")
    print(f"  {'-' * 130}")
    for r in full_results[:20]:
        print(f"  {r['name']:55s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
              f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:4.2f} | "
              f"{r['n_long']:3d}/{r['n_short']:<3d} | {r['avg_win']:+6.2f}% | "
              f"{r['avg_loss']:5.2f}% | {r['avg_days']:4.1f}")

    # --- Walk-forward top 10 ---
    if wf_results:
        wf_results.sort(key=lambda x: -x['ann'])
        print(f"\n{'=' * 130}")
        print(f"  TOP 10 WALK-FORWARD RESULTS (out-of-sample)")
        print(f"{'=' * 130}")
        print(f"  {'Config':65s} | {'Ann':>7s} | {'WR':>5s} | {'N':>4s} | {'DD':>6s} | "
              f"{'PF':>4s}")
        print(f"  {'-' * 110}")
        for r in wf_results[:10]:
            print(f"  {r['name']:65s} | {r['ann']:+7.1f}% | {r['wr']:5.1f}% | "
                  f"{r['n']:4d} | {r['dd']:6.1f}% | {r['pf']:4.2f}")

    # --- Best config detail ---
    if full_results:
        best = full_results[0]
        print(f"\n{'=' * 130}")
        print(f"  BEST CONFIG: {best['name']}")
        print(f"  Ann={best['ann']:+.1f}%  WR={best['wr']:.1f}%  N={best['n']}  "
              f"DD={best['dd']:.1f}%  PF={best['pf']:.2f}  Final={best['cash']:.0f}")
        print(f"{'=' * 130}")

        print(f"\n  DIRECTION BREAKDOWN:")
        print(f"    Long:  {best['n_long']:4d} trades  WR={best['wr_long']:5.1f}%  PnL={best['pnl_long']:+.0f}")
        print(f"    Short: {best['n_short']:4d} trades  WR={best['wr_short']:5.1f}%  PnL={best['pnl_short']:+.0f}")

        print(f"\n  EXIT REASON BREAKDOWN:")
        for reason, s in sorted(best['reasons'].items(), key=lambda x: -x[1]['n']):
            rwr = s['w'] / max(s['n'], 1) * 100
            print(f"    {reason:12s}: {s['n']:4d} trades  WR={rwr:5.1f}%  PnL={s['pnl']:+.1f}%")

        print(f"\n  YEARLY BREAKDOWN:")
        for y in sorted(best['yearly'].keys()):
            s = best['yearly'][y]
            wr_y = s['w'] / max(s['n'], 1) * 100
            print(f"    {y}: {s['n']:3d} trades  WR={wr_y:5.1f}%  PnL%={s['pnl']:+.1f}%  Abs={s['abs']:+.0f}")

        print(f"\n  GROUP BREAKDOWN:")
        for g in sorted(best['grp_counts'].keys(), key=lambda x: -best['grp_counts'][x]['n']):
            gs = best['grp_counts'][g]
            wr_g = gs['w'] / max(gs['n'], 1) * 100
            print(f"    {g:15s}: {gs['n']:3d}t  WR={wr_g:5.1f}%  Abs={gs['pnl']:+.0f}")

    # --- Per signal type summary ---
    print(f"\n  PER SIGNAL TYPE (best config per type):")
    best_per_sig = {}
    for r in full_results:
        st = r['signal_type']
        if st not in best_per_sig or r['ann'] > best_per_sig[st]['ann']:
            best_per_sig[st] = r
    for st in sorted(best_per_sig.keys()):
        r = best_per_sig[st]
        sig_name = SIGNAL_NAMES[st]
        print(f"    {sig_name:8s}: Ann={r['ann']:+.1f}%  WR={r['wr']:.1f}%  "
              f"N={r['n']:3d}  DD={r['dd']:.1f}%  PF={r['pf']:.2f}  "
              f"Long={r['n_long']}/Short={r['n_short']}  "
              f"AvgW={r['avg_win']:+.2f}%  AvgL={r['avg_loss']:.2f}%")

    # --- Yearly for top 5 ---
    if len(full_results) >= 2:
        print(f"\n  YEARLY BREAKDOWN FOR TOP 5:")
        for idx, r in enumerate(full_results[:5]):
            print(f"\n  #{idx+1}: {r['name']} (Ann={r['ann']:+.1f}%, WR={r['wr']:.1f}%, DD={r['dd']:.1f}%)")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr_y = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:3d}t  WR={wr_y:5.1f}%  PnL%={ys['pnl']:+.1f}%  Abs={ys['abs']:+.0f}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
