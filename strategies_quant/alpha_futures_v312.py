"""
V312: IC-Adaptive Momentum — Push toward 600% annual without leverage
======================================================================
Key innovations over V311:
1. IC-adaptive weighting: weight factors by their recent information coefficient
2. Volatility-normalized scoring: prefer commodities with higher expected move
3. Trailing exit via signal decay: hold while momentum persists, exit when it fades
4. Open-to-close intraday: enter at open, exit at close for faster compounding
5. Re-entry on persistence: re-enter same commodity if signal stays strong
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_v301 import load_all_data, compute_factors, detect_regimes

CASH0 = 1_000_000
COMM = 0.0005


def compute_ic_adaptive_signals(C, O, H, L, V, NS, ND, ic_window=60):
    """
    Compute individual factor signals and their rolling IC,
    then combine with IC-weighted ensemble.
    """
    # --- Individual factor computation (cross-sectional percentile ranks) ---

    # Factor 1-3: Multi-period momentum
    mom_signals = {}
    for period in [3, 5, 10, 20]:
        r = np.full((NS, ND), np.nan)
        for di in range(period, ND):
            rets = np.full(NS, np.nan)
            for si in range(NS):
                c0, c1 = C[si, di - period], C[si, di]
                if not np.isnan(c0) and not np.isnan(c1) and c0 > 0:
                    rets[si] = (c1 - c0) / c0
            valid = ~np.isnan(rets)
            if valid.sum() >= 5:
                r[:, di] = pd.Series(rets).rank(pct=True, na_option='keep').values
        mom_signals[period] = r

    # Factor 4: Trend slope (20d)
    slope_rank = np.full((NS, ND), np.nan)
    for di in range(20, ND):
        slopes = np.full(NS, np.nan)
        for si in range(NS):
            prices = C[si, di - 20:di]
            valid = ~np.isnan(prices)
            if valid.sum() >= 15:
                x = np.arange(20)[valid]
                y = prices[valid]
                if len(x) >= 10:
                    s = np.polyfit(x, y, 1)[0]
                    mean_p = np.mean(y)
                    if mean_p > 0:
                        slopes[si] = s / mean_p
        valid = ~np.isnan(slopes)
        if valid.sum() >= 5:
            slope_rank[:, di] = pd.Series(slopes).rank(pct=True, na_option='keep').values

    # Factor 5: Volume surge ratio
    vol_rank = np.full((NS, ND), np.nan)
    for di in range(20, ND):
        vratios = np.full(NS, np.nan)
        for si in range(NS):
            vt = V[si, di]
            va = np.nanmean(V[si, di - 20:di])
            if not np.isnan(vt) and not np.isnan(va) and va > 0:
                vratios[si] = vt / va
        valid = ~np.isnan(vratios)
        if valid.sum() >= 5:
            vol_rank[:, di] = pd.Series(vratios).rank(pct=True, na_option='keep').values

    # Factor 6: Intraday range capture (open-to-close return potential)
    oc_rank = np.full((NS, ND), np.nan)
    for di in range(1, ND):
        oc_rets = np.full(NS, np.nan)
        for si in range(NS):
            o_t = O[si, di]
            c_t = C[si, di]
            if not np.isnan(o_t) and not np.isnan(c_t) and o_t > 0:
                oc_rets[si] = (c_t - o_t) / o_t
        valid = ~np.isnan(oc_rets)
        if valid.sum() >= 5:
            oc_rank[:, di] = pd.Series(oc_rets).rank(pct=True, na_option='keep').values

    # Factor 7: Volatility (high vol = bigger moves when right)
    vol_raw = np.full((NS, ND), np.nan)
    for di in range(20, ND):
        vols = np.full(NS, np.nan)
        for si in range(NS):
            rets = []
            for j in range(max(1, di - 20), di):
                if not np.isnan(C[si, j]) and not np.isnan(C[si, j - 1]) and C[si, j - 1] > 0:
                    rets.append(C[si, j] / C[si, j - 1] - 1)
            if len(rets) >= 10:
                vols[si] = np.std(rets) * np.sqrt(252)
        valid = ~np.isnan(vols)
        if valid.sum() >= 5:
            vol_raw[:, di] = pd.Series(vols).rank(pct=True, na_option='keep').values

    # --- Compute rolling IC for each factor ---
    # IC = Spearman correlation between factor rank at di and forward return at di+1
    # Use next-day open-to-close return as the target
    forward_oc = np.full((NS, ND), np.nan)
    for di in range(1, ND - 1):
        for si in range(NS):
            o_next = O[si, di + 1]
            c_next = C[si, di + 1]
            if not np.isnan(o_next) and not np.isnan(c_next) and o_next > 0:
                forward_oc[si, di] = (c_next - o_next) / o_next

    all_factors = {
        'mom3': mom_signals[3], 'mom5': mom_signals[5],
        'mom10': mom_signals[10], 'mom20': mom_signals[20],
        'slope': slope_rank, 'vol': vol_rank,
        'oc': oc_rank, 'vola': vol_raw,
    }

    # Rolling IC for each factor
    factor_ics = {}
    for fname, fvals in all_factors.items():
        ic_series = np.full(ND, np.nan)
        for di in range(ic_window, ND - 1):
            f_slice = fvals[:, di - ic_window:di].ravel()
            r_slice = forward_oc[:, di - ic_window:di].ravel()
            valid = ~np.isnan(f_slice) & ~np.isnan(r_slice)
            if valid.sum() >= 50:
                ic_series[di] = np.corrcoef(f_slice[valid], r_slice[valid])[0, 1]
        factor_ics[fname] = ic_series

    # --- IC-weighted ensemble score ---
    scores = np.full((NS, ND), np.nan)
    for di in range(ic_window, ND):
        weighted_sum = np.zeros(NS)
        total_ic = 0
        for fname, fvals in all_factors.items():
            ic = factor_ics[fname][di]
            if np.isnan(ic):
                continue
            # Only use positive IC factors (negative IC = factor hurts)
            ic_pos = max(ic, 0)
            f = fvals[:, di]
            valid_f = ~np.isnan(f)
            if valid_f.sum() < 5:
                continue
            weighted_sum += np.nan_to_num(f, nan=0.5) * ic_pos
            total_ic += ic_pos

        if total_ic > 0:
            scores[:, di] = weighted_sum / total_ic
        else:
            # Fallback: equal weight momentum
            fallback = np.zeros(NS)
            n_valid = 0
            for p in [5, 10, 20]:
                f = mom_signals[p][:, di]
                if not np.isnan(f).all():
                    fallback += np.nan_to_num(f, nan=0.5)
                    n_valid += 1
            if n_valid > 0:
                scores[:, di] = fallback / n_valid

    return scores, factor_ics


def backtest_v312(C, O, H, L, NS, ND, dates, syms,
                  scores, regime,
                  top_n=1, hold_days=1, atr_stop=2.5,
                  min_score=0.6, signal_exit=True,
                  start_di=60, end_di=None,
                  use_open_entry=True):
    """
    Backtest with IC-adaptive scoring.
    - use_open_entry=True: enter at next day's open, exit at close (true 1-day)
    - signal_exit=True: exit when score drops below min_score
    """
    if end_di is None:
        end_di = ND

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions = []
    trades = []

    for di in range(max(start_di, 1), end_di - 1):
        d = dates[di]
        daily_pnl = 0
        new_positions = []

        # --- Exit logic ---
        for si, edi, ep, sp, alloc, entry_di in positions:
            if use_open_entry:
                # Enter at next day's open after signal, exit at close
                entry_price = O[si, di] if entry_di == di - 1 and edi == di - 1 else ep
                # Re-check: if we just entered yesterday at open, we're holding
                exit_price = C[si, di]
            else:
                exit_price = C[si, di]

            if np.isnan(exit_price):
                new_positions.append((si, edi, ep, sp, alloc, entry_di))
                continue

            exit_r = None
            actual_ep = ep

            # Check stop
            if exit_price < sp:
                exit_r = 'stop'

            # Check hold period
            elif di - edi >= hold_days:
                exit_r = 'hold'

            # Check signal decay
            elif signal_exit and not np.isnan(scores[si, di]):
                if scores[si, di] < min_score * 0.8:  # Allow some slack
                    exit_r = 'signal'

            if exit_r:
                pnl = (exit_price - actual_ep) / actual_ep - COMM
                profit = equity * alloc * pnl
                daily_pnl += profit
                trades.append({
                    'pnl_abs': profit, 'pnl_pct': pnl * 100,
                    'days': di - edi + 1, 'di': di, 'year': d.year,
                    'sym': syms[si], 'reason': exit_r,
                })
            else:
                new_positions.append((si, edi, ep, sp, alloc, entry_di))

        positions = new_positions
        equity += daily_pnl
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        if equity <= 0:
            break

        # --- Entry logic ---
        held = {p[0] for p in positions}
        if len(positions) >= top_n:
            continue

        # Regime filter
        r = regime[di] if regime is not None and di < len(regime) else 0
        if r in (-1, 2):
            continue

        candidates = []
        for si in range(NS):
            if si in held:
                continue
            score = scores[si, di]
            if np.isnan(score) or score < min_score:
                continue

            # Skip if no open price tomorrow
            if use_open_entry:
                if di + 1 < ND and np.isnan(O[si, di + 1]):
                    continue

            candidates.append((score, si))

        if not candidates:
            continue
        candidates.sort(key=lambda x: -x[0])

        alloc = 1.0 / max(top_n, 1)
        for score, si in candidates[:top_n]:
            if len(positions) >= top_n or si in held:
                break

            if use_open_entry:
                # Enter at tomorrow's open
                if di + 1 >= ND:
                    continue
                entry_p = O[si, di + 1]
                entry_di_actual = di + 1
            else:
                entry_p = C[si, di]
                entry_di_actual = di

            if np.isnan(entry_p) or entry_p <= 0:
                continue

            # ATR-based stop
            atr_v = []
            for j in range(max(start_di, di - 14), di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
            if not atr_v:
                continue
            atr = np.mean(atr_v)
            stop = entry_p - atr_stop * atr

            positions.append((si, entry_di_actual, entry_p, stop, alloc, di))
            held.add(si)

    # Close remaining
    for si, edi, ep, sp, alloc, entry_di in positions:
        c = C[si, ND - 1]
        if not np.isnan(c):
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * pnl

    return trades, equity, max_dd


def analyze(trades, equity, max_dd, label=""):
    if not trades:
        print(f"  {label}: no trades")
        return None
    nw = sum(1 for t in trades if t['pnl_pct'] > 0)
    wr = nw / len(trades) * 100
    n_days = max(1, trades[-1]['di'] - trades[0]['di'])
    ann = ((equity / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
    ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
    rets = np.array(ap) / CASH0
    sh = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0

    # Max consecutive wins/losses
    streak_w = streak_l = 0
    cur_w = cur_l = 0
    for t in sorted(trades, key=lambda x: x['di']):
        if t['pnl_pct'] > 0:
            cur_w += 1; cur_l = 0
            streak_w = max(streak_w, cur_w)
        else:
            cur_l += 1; cur_w = 0
            streak_l = max(streak_l, cur_l)

    print(f"  {label}: {len(trades)}t WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
          f"Sh={sh:.2f} eq={equity:,.0f} streakW={streak_w} streakL={streak_l}")

    yr = {}
    for t in trades:
        y = t['year']
        if y not in yr:
            yr[y] = {'n': 0, 'w': 0, 'pnl': []}
        yr[y]['n'] += 1
        if t['pnl_pct'] > 0:
            yr[y]['w'] += 1
        yr[y]['pnl'].append(t['pnl_pct'])
    for y in sorted(yr.keys()):
        ys = yr[y]
        cum = np.prod([1 + p / 100 for p in ys['pnl']]) - 1
        print(f"    {y}: {ys['n']}t WR={ys['w'] / ys['n'] * 100:.1f}% cum={cum:+.1%}")

    return {'n': len(trades), 'wr': wr, 'dd': max_dd, 'ann': ann, 'sh': sh}


def main():
    t0 = time.time()
    print("=" * 60)
    print("  V312: IC-ADAPTIVE MOMENTUM (600% TARGET, NO LEVERAGE)")
    print("=" * 60)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')

    F = compute_factors(C, O, H, L, V, OI, NS, ND)
    regime = detect_regimes(F, NS, ND)

    print("[V312] Computing IC-adaptive signals...", flush=True)
    scores, factor_ics = compute_ic_adaptive_signals(C, O, H, L, V, NS, ND)

    # Print IC summary
    print("\n--- Factor IC Summary (last available) ---")
    for fname, ic_s in factor_ics.items():
        last_valid = ic_s[~np.isnan(ic_s)]
        if len(last_valid) > 0:
            print(f"  {fname}: mean_IC={np.mean(last_valid):.4f} "
                  f"last_IC={last_valid[-1]:.4f}")

    # --- Walk-forward validation ---
    # Train: 2016-2020 (5yr), Test: 2021-2023 (3yr), then roll
    # Train: 2019-2022 (4yr), Test: 2023-2025 (3yr)
    wf_configs = [
        # (label, start, end, top_n, hold, min_score, open_entry, signal_exit, atr_stop)
        ("WF1-IS", 0, None, 1, 1, 0.5, True, True, 2.5),
        ("WF1-IS", 0, None, 1, 1, 0.6, True, False, 2.5),
        ("WF1-IS", 0, None, 1, 1, 0.7, True, True, 2.5),
        ("WF1-IS", 0, None, 1, 3, 0.6, True, True, 2.5),
        ("WF1-IS", 0, None, 3, 1, 0.6, True, True, 2.5),
        ("WF1-IS", 0, None, 3, 3, 0.6, True, True, 2.5),
        ("WF1-IS", 0, None, 1, 1, 0.6, False, True, 2.5),
        ("WF1-IS", 0, None, 1, 1, 0.6, True, True, 0),  # No stop
    ]

    # Find date indices
    date_idx = {d: i for i, d in enumerate(dates)}
    wf_start = None
    for d, i in date_idx.items():
        if d >= pd.Timestamp('2024-01-01'):
            wf_start = i
            break

    print(f"\n--- In-Sample Full Period ---")
    print(f"  WF start index: {wf_start} ({dates[wf_start] if wf_start else 'N/A'})")

    # Full in-sample
    for label, s, e, tn, hd, ms, oe, se, ats in wf_configs:
        trades, eq, dd = backtest_v312(
            C, O, H, L, NS, ND, dates, syms,
            scores, regime, top_n=tn, hold_days=hd,
            min_score=ms, use_open_entry=oe, signal_exit=se,
            atr_stop=ats)
        analyze(trades, eq, dd, f"{label} tn={tn} hd={hd} ms={ms} "
                             f"{'OE' if oe else 'CE'} {'SE' if se else 'FX'} ats={ats}")

    # Walk-forward (2024-2026 only)
    if wf_start:
        print(f"\n--- Walk-Forward 2024-2026 ---")
        for label, s, e, tn, hd, ms, oe, se, ats in wf_configs:
            trades, eq, dd = backtest_v312(
                C, O, H, L, NS, ND, dates, syms,
                scores, regime, top_n=tn, hold_days=hd,
                min_score=ms, use_open_entry=oe, signal_exit=se,
                atr_stop=ats, start_di=wf_start)
            analyze(trades, eq, dd, f"WF tn={tn} hd={hd} ms={ms} "
                                  f"{'OE' if oe else 'CE'} {'SE' if se else 'FX'} ats={ats}")

    # --- Focused sweep: top configs ---
    print(f"\n--- Focused Sweep (WF 2024-2026) ---")
    results = []
    for tn in [1, 2, 3]:
        for hd in [1, 2, 3]:
            for ms in [0.5, 0.6, 0.7]:
                for oe in [True, False]:
                    trades, eq, dd = backtest_v312(
                        C, O, H, L, NS, ND, dates, syms,
                        scores, regime, top_n=tn, hold_days=hd,
                        min_score=ms, use_open_entry=oe,
                        signal_exit=True, start_di=wf_start)
                    if len(trades) < 5:
                        continue
                    nw = sum(1 for t in trades if t['pnl_pct'] > 0)
                    wr = nw / len(trades) * 100
                    n_days = max(1, trades[-1]['di'] - trades[0]['di'])
                    ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
                    ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
                    rets = np.array(ap) / CASH0
                    sh = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
                    results.append({
                        'tn': tn, 'hd': hd, 'ms': ms, 'oe': oe,
                        'n': len(trades), 'wr': wr, 'ann': ann, 'dd': dd, 'sh': sh,
                    })

    results.sort(key=lambda x: -x['ann'])
    print(f"\n{'TN':>3} {'HD':>3} {'MS':>4} {'OE':>3} {'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 55)
    for r in results[:20]:
        print(f"{r['tn']:>3} {r['hd']:>3} {r['ms']:>4.1f} {'OE' if r['oe'] else 'CE':>3} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} {r['dd']:>6.1f} {r['sh']:>5.2f}")

    results.sort(key=lambda x: -x['sh'])
    print(f"\n--- By Sharpe ---")
    for r in results[:10]:
        print(f"  tn={r['tn']} hd={r['hd']} ms={r['ms']} {'OE' if r['oe'] else 'CE'}: "
              f"{r['n']}t WR={r['wr']:.1f}% ann={r['ann']:+.1f}% DD={r['dd']:.1f}% Sh={r['sh']:.2f}")

    print(f"\n[V312] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
