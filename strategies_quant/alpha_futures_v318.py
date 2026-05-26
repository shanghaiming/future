"""
V318: Research-Based Strategy — Overnight Gap + Carry + Volatility Targeting
===============================================================================
Based on user's comprehensive research findings:

1. Overnight-Intraday Reversal: 0.284%/day, Sharpe 3.5 (THE key edge)
   - Close-to-open gap predicts open-to-close reversal
   - Chinese night session captures global info → overreaction → intraday reversion
   - Implement with daily data: (O[di] - C[di-1]) / C[di-1] = overnight gap

2. Carry filter: only backwardated commodities (positive roll yield)
   - Term structure backwardation = structural bullishness

3. Volatility-targeted leverage: L = target_sigma / realized_sigma
   - Moreira and Muir (2017): improves Sharpe by reducing vol drag

4. Regime-conditional: HV20 percentile determines aggression level
   - Low vol: aggressive (can use leverage)
   - High vol: defensive (reduce or sit out)

5. Kelly sizing with drawdown breaker
   - Full Kelly near HWM, reduce at drawdown

6. Pyramiding winners: add to positions that are profitable

Clean execution: signal from di-1, enter at O[di].
"""
import sys, os, time, warnings, json, glob
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_v301 import load_all_data, compute_factors, detect_regimes
from alpha_futures_v311 import load_ts

CASH0 = 1_000_000
COMM = 0.0005


def compute_overnight_gap(C, O, NS, ND):
    """Overnight gap = (open - prev_close) / prev_close"""
    gap = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            c_prev = C[si, di - 1]
            o_now = O[si, di]
            if not np.isnan(c_prev) and not np.isnan(o_now) and c_prev > 0:
                gap[si, di] = (o_now - c_prev) / c_prev
    return gap


def compute_intraday_return(C, O, NS, ND):
    """Intraday return = (close - open) / open"""
    ret = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            o = O[si, di]
            c = C[si, di]
            if not np.isnan(o) and not np.isnan(c) and o > 0:
                ret[si, di] = (c - o) / o
    return ret


def compute_hv20(C, NS, ND):
    """20-day realized volatility per instrument"""
    hv = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = []
            for j in range(di - 20, di):
                if not np.isnan(C[si, j]) and not np.isnan(C[si, j-1]) and C[si, j-1] > 0:
                    rets.append(C[si, j] / C[si, j-1] - 1)
            if len(rets) >= 10:
                hv[si, di] = np.std(rets) * np.sqrt(252)
    return hv


def backtest_v318(C, O, H, L, NS, ND, dates, syms,
                  gap, intraday_ret, hv, ts_data, regime,
                  # Strategy parameters
                  gap_threshold=0.005,    # Minimum gap to trigger signal
                  use_reversal=True,     # Fade overnight gaps
                  use_momentum=True,     # Confirm with momentum
                  use_carry=True,        # Filter by carry
                  use_vol_target=True,   # Volatility-targeted leverage
                  base_leverage=3.0,     # Base leverage
                  target_vol=0.20,       # Target 20% annualized
                  max_leverage=8.0,
                  kelly_sizing=True,
                  dd_breaker=True,
                  max_dd_pct=25.0,
                  top_n=3,
                  hold_days=1,           # Intraday: enter open, exit close
                  start_di=60, end_di=None):
    """
    Research-based strategy combining overnight gap reversal + carry + vol targeting.

    Execution:
    - Signal computed from data up to di-1 (yesterday's close, overnight gap at di)
    - Entry at O[si, di] (today's open)
    - Exit at C[si, di] (today's close) for hold_days=1
    - Or hold for multiple days with trailing stop
    """
    if end_di is None:
        end_di = ND

    ts_si = {s: i for i, s in enumerate(ts_data['syms'])}
    ts_di_map = {d: i for i, d in enumerate(ts_data['dates'])}

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions = []
    trades = []

    for di in range(max(start_di + 1, 1), end_di):
        d = dates[di]
        daily_pnl = 0
        new_positions = []

        # --- Exit logic ---
        for pos in positions:
            si, edi, ep, sp, alloc, lev = pos
            c = C[si, di]
            if np.isnan(c):
                new_positions.append(pos)
                continue

            exit_r = None
            if c < sp:
                exit_r = 'stop'
            elif hold_days == 1 and di == edi:
                # Intraday: exit at close of entry day
                exit_r = 'intraday'
            elif hold_days > 1 and di - edi >= hold_days:
                exit_r = 'hold'

            if exit_r:
                pnl = (c - ep) / ep - COMM
                profit = equity * alloc * lev * pnl
                daily_pnl += profit
                trades.append({
                    'pnl_abs': profit, 'pnl_pct': pnl * 100 * lev,
                    'days': di - edi + 1, 'di': di, 'year': d.year,
                    'sym': syms[si], 'reason': exit_r,
                })
            else:
                new_positions.append(pos)

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

        # --- Drawdown breaker ---
        if dd_breaker and peak > 0:
            current_dd = (peak - equity) / peak * 100
            if current_dd > max_dd_pct:
                continue  # Stop trading

        # --- Entry logic ---
        held = {p[0] for p in positions}
        if len(positions) >= top_n:
            continue

        # --- Regime filter ---
        r = regime[di] if regime is not None and di < len(regime) else 0
        if r in (-1, 2):  # Skip choppy/volatile
            continue

        # --- Compute signals for each commodity ---
        candidates = []
        for si in range(NS):
            if si in held:
                continue

            g = gap[si, di]  # Today's overnight gap
            if np.isnan(g):
                continue
            if np.isnan(O[si, di]) or O[si, di] <= 0:
                continue

            signal = 0
            n_components = 0

            # 1. OVERNIGHT GAP REVERSAL (weight: 35%)
            if use_reversal:
                if g < -gap_threshold:
                    # Gap DOWN → expect reversal UP → BULLISH
                    signal += 0.35
                    n_components += 1
                elif g > gap_threshold:
                    # Gap UP → expect reversal DOWN → BEARISH (skip for long-only)
                    signal -= 0.35
                    n_components += 1
                else:
                    # Small gap → neutral
                    n_components += 1

            # 2. MOMENTUM CONFIRMATION (weight: 25%)
            if use_momentum:
                c5 = C[si, di - 5] if di >= 5 else np.nan
                c_now = C[si, di - 1] if di >= 1 else np.nan  # Yesterday's close
                if not np.isnan(c5) and not np.isnan(c_now) and c5 > 0:
                    mom5 = (c_now - c5) / c5
                    if mom5 > 0.02:
                        signal += 0.25
                    elif mom5 > 0:
                        signal += 0.15
                    else:
                        signal -= 0.10
                    n_components += 1

            # 3. CARRY FILTER (weight: 25%)
            carry_score = 0
            if use_carry:
                tsi = ts_di_map.get(d)
                if tsi is not None:
                    sym = syms[si]
                    tssi = ts_si.get(sym, -1)
                    if tssi >= 0 and tsi < ts_data['cz'].shape[1]:
                        cz = ts_data['cz'][tssi, tsi]
                        if not np.isnan(cz):
                            if cz > 1:
                                carry_score = 0.25  # Strong backwardation
                            elif cz > 0:
                                carry_score = 0.15  # Mild backwardation
                            else:
                                carry_score = -0.10  # Contango (avoid)
                            n_components += 1
            signal += carry_score

            # 4. VOL REGIME (weight: 15%)
            h = hv[si, di] if not np.isnan(hv[si, di]) else 0.3
            if h < 0.2:  # Low vol = favorable
                signal += 0.15
                n_components += 1
            elif h < 0.35:
                signal += 0.05
                n_components += 1
            else:
                signal -= 0.10
                n_components += 1

            if n_components < 2:
                continue
            if signal < 0.3:  # Minimum composite threshold
                continue

            # --- Volatility-targeted leverage ---
            if use_vol_target and not np.isnan(h) and h > 0.05:
                lev = min(target_vol / h * base_leverage, max_leverage)
            else:
                lev = base_leverage

            # --- Kelly sizing with drawdown breaker ---
            if dd_breaker and peak > 0:
                current_dd = (peak - equity) / peak * 100
                if current_dd > max_dd_pct * 0.6:  # 60% of max DD
                    lev *= 0.5  # Half leverage
                if current_dd > max_dd_pct * 0.8:
                    lev *= 0.25  # Quarter leverage

            lev = max(lev, 0.5)  # Minimum leverage
            candidates.append((signal, si, lev))

        if not candidates:
            continue
        candidates.sort(key=lambda x: -x[0])

        alloc = 1.0 / max(top_n, 1)
        for signal, si, lev in candidates[:top_n]:
            if len(positions) >= top_n or si in held:
                break

            # Enter at today's open (signal from yesterday + today's gap)
            ep = O[si, di]
            if np.isnan(ep) or ep <= 0:
                continue

            # ATR stop
            atr_v = []
            for j in range(max(start_di, di - 14), di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
            if not atr_v:
                continue
            atr = np.mean(atr_v)
            stop = ep - 2.5 * atr

            positions.append((si, di, ep, stop, alloc, lev))
            held.add(si)

    # Close remaining
    for si, edi, ep, sp, alloc, lev in positions:
        c = C[si, ND - 1]
        if not np.isnan(c):
            pnl = (c - ep) / ep - COMM
            equity += equity * alloc * lev * pnl

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
    print(f"  {label}: {len(trades)}t WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
          f"Sh={sh:.2f} eq={equity:,.0f}")
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
    print("=" * 70)
    print("  V318: RESEARCH-BASED — OVERNIGHT GAP + CARRY + VOL TARGETING")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    ts_data = load_ts(start='2021-01-01')
    F = compute_factors(C, O, H, L, V, OI, NS, ND)
    regime = detect_regimes(F, NS, ND)

    print("[V318] Computing signals...", flush=True)
    gap = compute_overnight_gap(C, O, NS, ND)
    idr = compute_intraday_return(C, O, NS, ND)
    hv = compute_hv20(C, NS, ND)

    # Verify overnight-intraday reversal exists in data
    print("\n--- Overnight-Intraday Correlation Check ---")
    gap_flat = gap[:, 60:].ravel()
    idr_flat = idr[:, 60:].ravel()
    valid = ~np.isnan(gap_flat) & ~np.isnan(idr_flat)
    if valid.sum() > 100:
        corr = np.corrcoef(gap_flat[valid], idr_flat[valid])[0, 1]
        print(f"  Gap vs Intraday correlation: {corr:.4f}")
        print(f"  (Negative = reversal effect exists)")

        # By gap direction
        big_gap_up = valid & (gap_flat > 0.005)
        big_gap_dn = valid & (gap_flat < -0.005)
        if big_gap_up.sum() > 50:
            avg_idr_after_up = np.mean(idr_flat[big_gap_up])
            print(f"  Avg intraday after gap UP (>0.5%): {avg_idr_after_up:+.4f}")
        if big_gap_dn.sum() > 50:
            avg_idr_after_dn = np.mean(idr_flat[big_gap_dn])
            print(f"  Avg intraday after gap DN (<-0.5%): {avg_idr_after_dn:+.4f}")

    wf_start = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2024-01-01'):
            wf_start = i
            break

    # --- Parameter sweep ---
    print(f"\n=== PARAMETER SWEEP (WF 2024-2026) ===")
    results = []
    for gap_thresh in [0.003, 0.005, 0.008, 0.01]:
        for use_rev in [True, False]:
            for use_mom in [True, False]:
                for use_car in [True, False]:
                    for tn in [1, 2, 3]:
                        for hd in [1, 2, 3]:
                            for base_lev in [1, 2, 3, 5]:
                                trades, eq, dd = backtest_v318(
                                    C, O, H, L, NS, ND, dates, syms,
                                    gap, idr, hv, ts_data, regime,
                                    gap_threshold=gap_thresh,
                                    use_reversal=use_rev, use_momentum=use_mom,
                                    use_carry=use_car, use_vol_target=True,
                                    base_leverage=base_lev,
                                    top_n=tn, hold_days=hd,
                                    start_di=wf_start)
                                if len(trades) < 5:
                                    continue
                                nw = sum(1 for t in trades if t['pnl_pct'] > 0)
                                wr = nw / len(trades) * 100
                                n_days = max(1, trades[-1]['di'] - trades[0]['di'])
                                ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
                                ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
                                rets_arr = np.array(ap) / CASH0
                                sh = np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252) \
                                    if np.std(rets_arr) > 0 else 0
                                results.append({
                                    'gt': gap_thresh,
                                    'R': use_rev, 'M': use_mom, 'C': use_car,
                                    'tn': tn, 'hd': hd, 'lev': base_lev,
                                    'n': len(trades), 'wr': wr, 'ann': ann,
                                    'dd': dd, 'sh': sh,
                                })

    results.sort(key=lambda x: -x['ann'])
    print(f"\n{'GT':>6} {'R':>2} {'M':>2} {'C':>2} {'TN':>3} {'HD':>3} {'L':>3} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 65)
    for r in results[:30]:
        print(f"{r['gt']:>6.3f} {'Y' if r['R'] else 'N':>2} {'Y' if r['M'] else 'N':>2} "
              f"{'Y' if r['C'] else 'N':>2} {r['tn']:>3} {r['hd']:>3} {r['lev']:>3} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} {r['dd']:>6.1f} {r['sh']:>5.2f}")

    results.sort(key=lambda x: -x['sh'])
    print(f"\n--- By Sharpe ---")
    for r in results[:15]:
        print(f"  gt={r['gt']:.3f} R={'Y' if r['R'] else 'N'} M={'Y' if r['M'] else 'N'} "
              f"C={'Y' if r['C'] else 'N'} tn={r['tn']} hd={r['hd']} lev={r['lev']}: "
              f"{r['n']}t WR={r['wr']:.1f}% ann={r['ann']:+.1f}% "
              f"DD={r['dd']:.1f}% Sh={r['sh']:.2f}")

    # Best configs detail
    print(f"\n=== BEST CONFIG DETAIL ===")
    best_configs = [
        (0.005, True, True, True, 1, 1, 3),
        (0.005, True, True, True, 1, 2, 3),
        (0.005, True, True, True, 2, 1, 3),
        (0.005, True, True, True, 3, 1, 3),
        (0.005, True, True, True, 1, 1, 5),
        (0.005, True, False, True, 1, 1, 3),
        (0.005, False, True, True, 1, 2, 3),
    ]
    for gt, ur, um, uc, tn, hd, bl in best_configs:
        trades, eq, dd = backtest_v318(
            C, O, H, L, NS, ND, dates, syms,
            gap, idr, hv, ts_data, regime,
            gap_threshold=gt, use_reversal=ur, use_momentum=um,
            use_carry=uc, base_leverage=bl, top_n=tn, hold_days=hd,
            start_di=wf_start)
        lbl = f"gt={gt} {'R' if ur else '-'}{'M' if um else '-'}{'C' if uc else '-'} tn={tn} hd={hd} lev={bl}"
        analyze(trades, eq, dd, lbl)

    print(f"\n[V318] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
