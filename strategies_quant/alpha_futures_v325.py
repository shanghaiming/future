"""
V325: Multi-Strategy Portfolio — Diversified Alpha Extraction
=============================================================
Run multiple independent strategies simultaneously:
  A: Short-term mean reversion (combo oversold, hd=2)
  B: Medium-term mean reversion (combo oversold, hd=5)
  C: Momentum with pullback (mom rank + 1d pullback)
  D: Cross-sectional carry + momentum (from V318)

Each gets capital allocation. Portfolio-level DD breaker.
Test 5+ years (2019-2026) and full 10-year (2016-2026).
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
from collections import defaultdict
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_v301 import load_all_data
from alpha_futures_v311 import compute_all_signals

CASH0 = 1_000_000
COMM = 0.0005


def compute_all_v325_signals(C, O, H, L, V, NS, ND):
    """Compute all signals needed for multi-strategy portfolio."""
    # Consecutive down days
    consec_dn = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        consec = 0
        for di in range(1, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                if C[si, di] < C[si, di-1]:
                    consec += 1
                else:
                    consec = 0
            else:
                consec = 0
            consec_dn[si, di] = consec

    # 5d return
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di-5]) and C[si, di-5] > 0:
                ret_5d[si, di] = C[si, di] / C[si, di-5] - 1

    # Overnight gap
    gap = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if not np.isnan(O[si, di]) and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                gap[si, di] = O[si, di] / C[si, di-1] - 1

    # Combo oversold rank
    combo_rank = np.full((NS, ND), np.nan)
    for di in range(ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            s = 0.0
            s += min(consec_dn[si, di] / 5.0, 1.0) * 0.4
            if not np.isnan(ret_5d[si, di]):
                s += min(max(-ret_5d[si, di] / 0.1, 0), 1.0) * 0.4
            if not np.isnan(gap[si, di]):
                s += min(max(-gap[si, di] / 0.03, 0), 1.0) * 0.2
            scores[si] = s
        valid = ~np.isnan(scores)
        if valid.sum() >= 5:
            combo_rank[:, di] = pd.Series(scores).rank(pct=True, na_option='keep').values

    # Momentum scores
    mom_scores = compute_all_signals(C, O, H, L, V, NS, ND)

    # 20d vol
    vol_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            rets = []
            for j in range(di - 20, di):
                if not np.isnan(C[si, j]) and not np.isnan(C[si, j-1]) and C[si, j-1] > 0:
                    rets.append(C[si, j] / C[si, j-1] - 1)
            if len(rets) >= 10:
                vol_20d[si, di] = np.std(rets) * np.sqrt(252)

    return combo_rank, consec_dn, mom_scores, vol_20d


def backtest_portfolio(C, O, H, L, NS, ND, dates, syms,
                       combo_rank, consec_dn, mom_scores, vol_20d,
                       strategies,  # list of (name, signal_type, top_n, hold_days, alloc_pct)
                       atr_stop=2.5,
                       dd_breaker=None,
                       leverage=1.0,
                       start_di=60, end_di=None):
    """
    Multi-strategy portfolio backtest.
    Each strategy independently selects positions from its allocated capital.
    """
    if end_di is None:
        end_di = ND - 1

    total_alloc = sum(s[4] for s in strategies)
    if total_alloc <= 0:
        return [], CASH0, 0

    equity = CASH0
    peak = equity
    max_dd = 0.0
    # Each strategy has its own position list: [(si, edi, ep, sp, alloc)]
    positions = {s[0]: [] for s in strategies}
    trades = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0

        # Exit all strategy positions
        for strat_name, strat_positions in positions.items():
            new_pos = []
            for si, edi, ep, sp, alloc in strat_positions:
                c = C[si, di]
                if np.isnan(c):
                    new_pos.append((si, edi, ep, sp, alloc))
                    continue
                exit_r = None
                if c < sp:
                    exit_r = 'stop'
                # Find hold_days for this strategy
                for s in strategies:
                    if s[0] == strat_name:
                        if di >= edi + s[3]:
                            exit_r = 'hold'
                        break
                if exit_r:
                    pnl = (c - ep) / ep - COMM
                    profit = equity * alloc * leverage * pnl
                    daily_pnl += profit
                    trades.append({
                        'pnl_abs': profit, 'pnl_pct': pnl * 100 * leverage,
                        'days': di - edi + 1, 'di': di, 'year': d.year,
                        'sym': syms[si], 'reason': exit_r, 'strat': strat_name,
                    })
                else:
                    new_pos.append((si, edi, ep, sp, alloc))
            positions[strat_name] = new_pos

        equity += daily_pnl
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        if equity <= 0:
            break

        # Drawdown breaker
        pos_mult = 1.0
        if dd_breaker and peak > 0:
            current_dd = (peak - equity) / peak
            if current_dd > dd_breaker:
                pos_mult = max(0.1, 1.0 - current_dd / 0.5)

        # Entry for each strategy
        all_held = set()
        for p_list in positions.values():
            for p in p_list:
                all_held.add(p[0])

        for strat_name, sig_type, tn, hd, alloc_pct in strategies:
            if len(positions[strat_name]) >= tn:
                continue

            alloc = alloc_pct / total_alloc * pos_mult

            # Select candidates based on signal type
            candidates = []
            for si in range(NS):
                if si in all_held:
                    continue
                if di + 1 >= ND or np.isnan(O[si, di + 1]):
                    continue

                if sig_type == 'mr_short':
                    # Short-term mean reversion: combo rank, prefer high oversold
                    r = combo_rank[si, di]
                    if np.isnan(r) or r < 0.7:
                        continue
                    candidates.append((r, si))

                elif sig_type == 'mr_med':
                    # Medium-term: combo rank but require consec >= 2
                    r = combo_rank[si, di]
                    if np.isnan(r) or r < 0.7:
                        continue
                    if consec_dn[si, di] < 2:
                        continue
                    candidates.append((r, si))

                elif sig_type == 'mom':
                    # Momentum with 1d pullback
                    m = mom_scores[si, di]
                    if np.isnan(m) or m < 0.5:
                        continue
                    if consec_dn[si, di] < 1:
                        continue
                    candidates.append((m, si))

                elif sig_type == 'mr_deep':
                    # Deep oversold: consec >= 4
                    cd = consec_dn[si, di]
                    if cd < 4:
                        continue
                    candidates.append((cd, si))

            if not candidates:
                continue
            candidates.sort(key=lambda x: -x[0])

            for score, si in candidates[:tn]:
                if len(positions[strat_name]) >= tn or si in all_held:
                    break
                ep = O[si, di + 1]
                if np.isnan(ep) or ep <= 0:
                    continue
                atr_v = []
                for j in range(max(start_di, di - 14), di):
                    hh, ll, cc = H[si, j], L[si, j], C[si, j]
                    if not any(np.isnan([hh, ll, cc])):
                        atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
                if not atr_v:
                    continue
                atr = np.mean(atr_v)
                positions[strat_name].append((si, di + 1, ep, ep - atr_stop * atr, alloc))
                all_held.add(si)

    # Close remaining
    for strat_name, strat_positions in positions.items():
        for si, edi, ep, sp, alloc in strat_positions:
            c = C[si, ND - 1]
            if not np.isnan(c) and c > 0:
                pnl = (c - ep) / ep - COMM
                equity += equity * alloc * leverage * pnl

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
        print(f"    {y}: {ys['n']}t WR={ys['w']/ys['n']*100:.1f}% cum={cum:+.1%}")

    return {'n': len(trades), 'wr': wr, 'dd': max_dd, 'ann': ann, 'sh': sh, 'eq': equity}


def main():
    t0 = time.time()
    print("=" * 70)
    print("  V325: MULTI-STRATEGY PORTFOLIO")
    print("  Diversified alpha from independent strategies")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    print(f"  {NS} sym, {ND} days, "
          f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    print("[V325] Computing signals...", flush=True)
    combo_rank, consec_dn, mom_scores, vol_20d = \
        compute_all_v325_signals(C, O, H, L, V, NS, ND)
    print("  Done.", flush=True)

    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2019-01-01'):
            bt_2019 = i
            break

    # === Portfolio compositions ===
    print("\n" + "=" * 70)
    print("  PORTFOLIO COMPOSITION SWEEP (2019-2026)")
    print("=" * 70)

    portfolios = [
        # (name, [(strat_name, signal, top_n, hold_days, alloc), ...])
        ("MR only (1x)", [
            ("mr_short", "mr_short", 1, 2, 1.0),
        ]),
        ("MR only (2x)", [
            ("mr_med", "mr_med", 2, 5, 1.0),
        ]),
        ("MR short+med", [
            ("short", "mr_short", 1, 2, 0.5),
            ("med", "mr_med", 1, 5, 0.5),
        ]),
        ("MR 3-leg", [
            ("short", "mr_short", 1, 2, 0.35),
            ("med", "mr_med", 1, 5, 0.35),
            ("deep", "mr_deep", 1, 5, 0.30),
        ]),
        ("MR+Mom", [
            ("mr", "mr_short", 1, 2, 0.6),
            ("mom", "mom", 1, 2, 0.4),
        ]),
        ("MR+Mom (equal)", [
            ("mr", "mr_med", 1, 5, 0.5),
            ("mom", "mom", 1, 2, 0.5),
        ]),
        ("3-strat equal", [
            ("mr_short", "mr_short", 1, 2, 0.33),
            ("mr_med", "mr_med", 1, 5, 0.33),
            ("mom", "mom", 1, 2, 0.34),
        ]),
        ("4-strat", [
            ("mr_short", "mr_short", 1, 2, 0.25),
            ("mr_med", "mr_med", 1, 5, 0.25),
            ("mr_deep", "mr_deep", 1, 5, 0.25),
            ("mom", "mom", 1, 2, 0.25),
        ]),
        ("MR deep heavy", [
            ("deep", "mr_deep", 1, 5, 0.5),
            ("med", "mr_med", 1, 5, 0.3),
            ("short", "mr_short", 1, 2, 0.2),
        ]),
        ("MR multi-pos", [
            ("mr1", "mr_short", 2, 2, 0.5),
            ("mr2", "mr_med", 2, 5, 0.5),
        ]),
        ("MR all 3-pos", [
            ("mr", "mr_med", 3, 5, 1.0),
        ]),
    ]

    for lev in [1, 2, 3]:
        print(f"\n  --- Leverage {lev}x ---")
        for p_name, strats in portfolios:
            trades, eq, dd = backtest_portfolio(
                C, O, H, L, NS, ND, dates, syms,
                combo_rank, consec_dn, mom_scores, vol_20d,
                strategies=strats, leverage=lev,
                start_di=bt_2019)
            if len(trades) < 5:
                continue
            nw = sum(1 for t in trades if t['pnl_pct'] > 0)
            wr = nw / len(trades) * 100
            n_days = max(1, trades[-1]['di'] - trades[0]['di'])
            ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
            ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
            rets_arr = np.array(ap) / CASH0
            sh = np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252) if np.std(rets_arr) > 0 else 0
            print(f"    {p_name:>20}: {len(trades):>4}t WR={wr:.1f}% "
                  f"ann={ann:+.1f}% DD={dd:.1f}% Sh={sh:.2f}")

    # === Best portfolios with yearly detail ===
    print("\n" + "=" * 70)
    print("  BEST PORTFOLIOS — YEARLY (2019-2026)")
    print("=" * 70)

    best_portfolios = [
        ("MR 3-leg lev=1", [
            ("short", "mr_short", 1, 2, 0.35),
            ("med", "mr_med", 1, 5, 0.35),
            ("deep", "mr_deep", 1, 5, 0.30),
        ], 1),
        ("MR short+med lev=1", [
            ("short", "mr_short", 1, 2, 0.5),
            ("med", "mr_med", 1, 5, 0.5),
        ], 1),
        ("MR short+med lev=2", [
            ("short", "mr_short", 1, 2, 0.5),
            ("med", "mr_med", 1, 5, 0.5),
        ], 2),
        ("MR deep heavy lev=1", [
            ("deep", "mr_deep", 1, 5, 0.5),
            ("med", "mr_med", 1, 5, 0.3),
            ("short", "mr_short", 1, 2, 0.2),
        ], 1),
        ("MR all 3-pos lev=1", [
            ("mr", "mr_med", 3, 5, 1.0),
        ], 1),
    ]

    for p_name, strats, lev in best_portfolios:
        trades, eq, dd = backtest_portfolio(
            C, O, H, L, NS, ND, dates, syms,
            combo_rank, consec_dn, mom_scores, vol_20d,
            strategies=strats, leverage=lev,
            start_di=bt_2019)
        print(f"\n  --- {p_name} ---")
        analyze(trades, eq, dd, p_name)

        # Per-strategy breakdown
        for sname in [s[0] for s in strats]:
            st = [t for t in trades if t.get('strat') == sname]
            if st:
                nw = sum(1 for t in st if t['pnl_pct'] > 0)
                wr = nw / len(st) * 100
                n_days = max(1, st[-1]['di'] - st[0]['di'])
                ann_s = ((1 + sum(t['pnl_pct']/100 for t in st) / len(st)) ** 252 - 1) * 100 if st else 0
                print(f"      {sname:>10}: {len(st)}t WR={wr:.1f}%")

    # === Full 10-year ===
    print("\n" + "=" * 70)
    print("  FULL 2016-2026 (10 years)")
    print("=" * 70)

    for p_name, strats, lev in best_portfolios[:3]:
        trades, eq, dd = backtest_portfolio(
            C, O, H, L, NS, ND, dates, syms,
            combo_rank, consec_dn, mom_scores, vol_20d,
            strategies=strats, leverage=lev,
            start_di=60)
        print(f"\n  FULL {p_name}")
        analyze(trades, eq, dd, f"full {p_name}")

    # === DD breaker variant ===
    print("\n" + "=" * 70)
    print("  DD BREAKER VARIANTS (2019-2026)")
    print("=" * 70)

    strats_best = [
        ("short", "mr_short", 1, 2, 0.5),
        ("med", "mr_med", 1, 5, 0.5),
    ]

    for lev in [1, 2, 3]:
        for db in [None, 0.10, 0.15, 0.20]:
            trades, eq, dd = backtest_portfolio(
                C, O, H, L, NS, ND, dates, syms,
                combo_rank, consec_dn, mom_scores, vol_20d,
                strategies=strats_best, leverage=lev,
                dd_breaker=db, start_di=bt_2019)
            if len(trades) < 5:
                continue
            nw = sum(1 for t in trades if t['pnl_pct'] > 0)
            wr = nw / len(trades) * 100
            n_days = max(1, trades[-1]['di'] - trades[0]['di'])
            ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
            ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
            rets_arr = np.array(ap) / CASH0
            sh = np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252) if np.std(rets_arr) > 0 else 0
            db_s = f"dd={db:.0%}" if db else "no dd"
            print(f"  {db_s:>8} lev={lev}: {len(trades)}t WR={wr:.1f}% "
                  f"ann={ann:+.1f}% DD={dd:.1f}% Sh={sh:.2f}")

    print(f"\n[V325] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
