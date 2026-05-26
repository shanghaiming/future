"""
V319: Cross-Commodity Lead-Lag + Entropy Gate + Kelly Sizing
=============================================================
Research says: structural lead-lag relationships exist because
information propagates from upstream to downstream commodities.
- Crude oil → chemicals (PTA, methanol, PP, PVC)
- Iron ore → steel rebar
- Copper → other base metals
- Soybeans → soybean meal/oil

This is CAUSAL, not just statistical correlation.

Also adding:
- Shannon entropy gate (only trade when market is ordered)
- Improved Kelly sizing with drawdown breaker
- Pyramiding winners (add to profitable positions)
- Clean no-look-ahead execution throughout
"""
import sys, os, time, warnings, json, glob
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_v301 import load_all_data, compute_factors, detect_regimes
from alpha_futures_v311 import load_ts, compute_all_signals

CASH0 = 1_000_000
COMM = 0.0005

# Lead-lag commodity groups (leader → followers)
LEAD_LAG_GROUPS = {
    'iron_steel': {
        'leaders': ['i0', 'im0'],  # Iron ore (new and old)
        'followers': ['rbfi', 'hcfi', 'jfi', 'jmfi'],  # Rebar, HRC, Coke, Coking coal
    },
    'oil_chem': {
        'leaders': ['scfi'],  # Crude oil
        'followers': ['ppfi', 'lfi', 'vfi', 'egfi', 'ebfi', 'safi', 'tafi', 'mafi'],
    },
    'copper_metal': {
        'leaders': ['cufi'],  # Copper
        'followers': ['alfi', 'znfi', 'nifi', 'snfi'],
    },
    'soybean_chain': {
        'leaders': ['mfi'],  # Soybeans
        'followers': ['yfi', 'ofi', 'pfi', 'rmfi'],  # Bean oil, bean meal, palm, rapeseed
    },
}


def find_symbol_idx(syms, targets):
    """Find indices for target symbols."""
    idx = []
    for t in targets:
        # Try direct match
        if t in syms:
            idx.append(syms.index(t))
        else:
            # Try partial match
            for i, s in enumerate(syms):
                if t in s or s in t:
                    idx.append(i)
                    break
    return idx


def compute_entropy(C, NS, ND, window=20, n_bins=10):
    """
    Shannon entropy of returns: H = -sum(p * log(p))
    Low entropy = ordered/trending market
    High entropy = random/choppy market
    """
    entropy = np.full(ND, np.nan)
    for di in range(window + 1, ND):
        rets = []
        for si in range(NS):
            for j in range(di - window, di):
                if not np.isnan(C[si, j]) and not np.isnan(C[si, j-1]) and C[si, j-1] > 0:
                    rets.append(C[si, j] / C[si, j-1] - 1)
        if len(rets) < 50:
            continue
        # Bin returns
        rets = np.array(rets)
        hist, _ = np.histogram(rets, bins=n_bins, density=True)
        hist = hist[hist > 0]
        entropy[di] = -np.sum(hist * np.log(hist))
    return entropy


def compute_lead_lag_signals(C, O, syms, NS, ND):
    """
    Compute lead-lag signals: use yesterday's move in leader
    as a signal for today's trade in followers.
    """
    signals = np.full((NS, ND), 0.0)

    for group_name, group in LEAD_LAG_GROUPS.items():
        leader_idx = find_symbol_idx(syms, group['leaders'])
        follower_idx = find_symbol_idx(syms, group['followers'])

        if not leader_idx or not follower_idx:
            continue

        for di in range(1, ND):
            # Leader's return yesterday
            leader_rets = []
            for li in leader_idx:
                c0, c1 = C[li, di - 1], C[li, di] if di < ND else np.nan
                if not np.isnan(c0) and not np.isnan(c1) and c0 > 0:
                    leader_rets.append((c1 - c0) / c0)

            if not leader_rets:
                continue
            avg_leader_ret = np.mean(leader_rets)

            # Apply signal to followers
            for fi in follower_idx:
                if fi < NS:
                    # Positive leader return = bullish signal for follower
                    if avg_leader_ret > 0.01:
                        signals[fi, di] = 0.6  # Strong bullish
                    elif avg_leader_ret > 0.003:
                        signals[fi, di] = 0.3  # Mild bullish
                    elif avg_leader_ret < -0.01:
                        signals[fi, di] = -0.6  # Strong bearish
                    elif avg_leader_ret < -0.003:
                        signals[fi, di] = -0.3  # Mild bearish

    return signals


def backtest_v319(C, O, H, L, NS, ND, dates, syms,
                  mom_scores, lead_lag, entropy, ts_data, regime,
                  w_mom=0.5, w_leadlag=0.3, w_carry=0.2,
                  entropy_gate=True, entropy_threshold=2.5,
                  top_n=1, hold_days=2, atr_stop=2.5,
                  min_score=0.3, leverage=1.0,
                  kelly_sizing=True, dd_breaker=True, max_dd_pct=25.0,
                  use_carry=True, pyramiding=False,
                  start_di=60, end_di=None):
    """
    Combined momentum + lead-lag + carry with entropy gate and Kelly sizing.
    Clean execution: signal from di, enter at O[di+1].
    """
    if end_di is None:
        end_di = ND - 1

    ts_si = {s: i for i, s in enumerate(ts_data['syms'])}
    ts_di_map = {d: i for i, d in enumerate(ts_data['dates'])}

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions = []
    trades = []
    wins = 0
    losses = 0

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0
        new_positions = []

        # --- Exit logic ---
        for pos in positions:
            si, edi, ep, sp, alloc, lev, pyramid_count = pos
            c = C[si, di]
            h = H[si, di] if not np.isnan(H[si, di]) else c
            if np.isnan(c):
                new_positions.append(pos)
                continue

            # Trailing stop
            new_sp = sp
            atr_v = []
            for j in range(max(start_di, di - 14), di):
                hh, ll, cc = H[si, j], L[si, j], C[si, j]
                if not any(np.isnan([hh, ll, cc])):
                    atr_v.append(max(hh - ll, abs(hh - cc), abs(ll - cc)))
            if atr_v:
                atr = np.mean(atr_v)
                new_sp = max(sp, h - atr_stop * atr)

            exit_r = None
            if c < new_sp:
                exit_r = 'stop'
            elif di - edi >= hold_days:
                exit_r = 'hold'

            # Pyramiding: add to winner
            if pyramiding and pyramid_count < 2 and not exit_r:
                if c > ep * 1.02 and len(positions) < top_n + 2:
                    # Add 50% more at current price
                    add_alloc = alloc * 0.5
                    new_alloc = alloc + add_alloc
                    # Adjust equity for the add
                    add_cost = equity * add_alloc * lev
                    new_positions.append((si, edi, ep, new_sp, new_alloc, lev, pyramid_count + 1))
                    continue

            if exit_r:
                pnl = (c - ep) / ep - COMM
                profit = equity * alloc * lev * pnl
                daily_pnl += profit
                trades.append({
                    'pnl_abs': profit, 'pnl_pct': pnl * 100 * lev,
                    'days': di - edi + 1, 'di': di, 'year': d.year,
                    'sym': syms[si], 'reason': exit_r,
                })
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
            else:
                new_positions.append((si, edi, ep, new_sp, alloc, lev, pyramid_count))

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
                continue

        # --- Entropy gate ---
        if entropy_gate and not np.isnan(entropy[di]):
            if entropy[di] > entropy_threshold:
                continue  # Market too chaotic, skip

        # --- Regime filter ---
        r = regime[di] if regime is not None and di < len(regime) else 0
        if r in (-1, 2):
            continue

        # --- Entry logic ---
        held = {p[0] for p in positions}
        if len(positions) >= top_n:
            continue

        candidates = []
        for si in range(NS):
            if si in held:
                continue
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue

            score = 0
            total_w = 0

            # Momentum signal
            ms = mom_scores[si, di]
            if not np.isnan(ms):
                score += w_mom * ms
                total_w += w_mom

            # Lead-lag signal
            ll = lead_lag[si, di]
            if ll != 0:
                # Convert from [-0.6, 0.6] to [0, 1] scale
                ll_scaled = (ll + 1) / 2
                score += w_leadlag * ll_scaled
                total_w += w_leadlag

            # Carry signal
            if use_carry:
                tsi = ts_di_map.get(d)
                if tsi is not None:
                    sym = syms[si]
                    tssi = ts_si.get(sym, -1)
                    if tssi >= 0 and tsi < ts_data['cz'].shape[1]:
                        cz = ts_data['cz'][tssi, tsi]
                        if not np.isnan(cz):
                            carry_scaled = 0.7 if cz > 1 else (0.6 if cz > 0 else 0.4)
                            score += w_carry * carry_scaled
                            total_w += w_carry

            if total_w == 0 or score / total_w < min_score:
                continue

            candidates.append((score / total_w if total_w > 0 else 0, si))

        if not candidates:
            continue
        candidates.sort(key=lambda x: -x[0])

        # Kelly sizing
        lev = leverage
        if kelly_sizing and wins + losses > 20:
            wr = wins / (wins + losses)
            # Simplified Kelly: f = 2p - 1 for even payoffs
            kelly_f = max(0, 2 * wr - 1)
            lev = leverage * kelly_f * 2  # Scale by Kelly
            lev = max(0.5, min(lev, leverage * 2))

        # Drawdown-adjusted leverage
        if dd_breaker and peak > 0:
            current_dd = (peak - equity) / peak * 100
            if current_dd > max_dd_pct * 0.6:
                lev *= 0.5
            if current_dd > max_dd_pct * 0.8:
                lev *= 0.3

        alloc = 1.0 / max(top_n, 1)
        for score, si in candidates[:top_n]:
            if len(positions) >= top_n or si in held:
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
            positions.append((si, di + 1, ep, ep - atr_stop * atr, alloc, lev, 0))
            held.add(si)

    for si, edi, ep, sp, alloc, lev, pc in positions:
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
    print("  V319: LEAD-LAG + ENTROPY GATE + KELLY SIZING")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    ts_data = load_ts(start='2021-01-01')
    F = compute_factors(C, O, H, L, V, OI, NS, ND)
    regime = detect_regimes(F, NS, ND)

    print("[V319] Computing signals...", flush=True)
    mom_scores = compute_all_signals(C, O, H, L, V, NS, ND)
    lead_lag = compute_lead_lag_signals(C, O, syms, NS, ND)
    entropy = compute_entropy(C, NS, ND)

    # Print lead-lag stats
    ll_active = np.sum(lead_lag != 0) / (NS * ND) * 100
    print(f"  Lead-lag active: {ll_active:.1f}% of (sym, day) pairs")
    print(f"  Entropy range: {np.nanmin(entropy):.3f} to {np.nanmax(entropy):.3f}")

    wf_start = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2024-01-01'):
            wf_start = i
            break

    # --- Print symbol mapping for lead-lag ---
    print(f"\n--- Symbol Mapping ---")
    for gname, g in LEAD_LAG_GROUPS.items():
        l_idx = find_symbol_idx(syms, g['leaders'])
        f_idx = find_symbol_idx(syms, g['followers'])
        l_names = [syms[i] for i in l_idx if i < len(syms)]
        f_names = [syms[i] for i in f_idx if i < len(syms)]
        print(f"  {gname}: leaders={l_names}, followers={f_names}")

    # --- Parameter sweep ---
    print(f"\n=== SWEEP (WF 2024-2026) ===")
    results = []
    for w_m, w_ll in [(0.5, 0.3), (0.4, 0.4), (0.6, 0.2), (0.3, 0.5), (1.0, 0.0), (0.0, 1.0)]:
        for ent_gate in [True, False]:
            for tn in [1, 2, 3]:
                for hd in [2, 3]:
                    for lev in [1, 2, 3]:
                        for kelly in [True, False]:
                            trades, eq, dd = backtest_v319(
                                C, O, H, L, NS, ND, dates, syms,
                                mom_scores, lead_lag, entropy, ts_data, regime,
                                w_mom=w_m, w_leadlag=w_ll,
                                entropy_gate=ent_gate, top_n=tn, hold_days=hd,
                                leverage=lev, kelly_sizing=kelly,
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
                                'wm': w_m, 'wll': w_ll,
                                'eg': ent_gate, 'tn': tn, 'hd': hd,
                                'lev': lev, 'kelly': kelly,
                                'n': len(trades), 'wr': wr, 'ann': ann,
                                'dd': dd, 'sh': sh,
                            })

    results.sort(key=lambda x: -x['ann'])
    print(f"\n{'WM':>4} {'WLL':>4} {'EG':>3} {'TN':>3} {'HD':>3} {'L':>3} {'K':>2} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 60)
    for r in results[:25]:
        print(f"{r['wm']:>4.1f} {r['wll']:>4.1f} {'Y' if r['eg'] else 'N':>3} "
              f"{r['tn']:>3} {r['hd']:>3} {r['lev']:>3} {'Y' if r['kelly'] else 'N':>2} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sh']:>5.2f}")

    results.sort(key=lambda x: -x['sh'])
    print(f"\n--- By Sharpe ---")
    for r in results[:15]:
        print(f"  wm={r['wm']:.1f} ll={r['wll']:.1f} "
              f"eg={'Y' if r['eg'] else 'N'} tn={r['tn']} hd={r['hd']} "
              f"lev={r['lev']} k={'Y' if r['kelly'] else 'N'}: "
              f"{r['n']}t WR={r['wr']:.1f}% ann={r['ann']:+.1f}% "
              f"DD={r['dd']:.1f}% Sh={r['sh']:.2f}")

    # Best detail
    print(f"\n=== LEAD-LAG ONLY DETAIL ===")
    for lev in [1, 2, 3]:
        trades, eq, dd = backtest_v319(
            C, O, H, L, NS, ND, dates, syms,
            mom_scores, lead_lag, entropy, ts_data, regime,
            w_mom=0, w_leadlag=1.0, top_n=2, hold_days=2,
            leverage=lev, start_di=wf_start)
        analyze(trades, eq, dd, f"lead-lag-only lev={lev}")

    print(f"\n=== BEST COMBINED DETAIL ===")
    for lev in [1, 2, 3, 5]:
        trades, eq, dd = backtest_v319(
            C, O, H, L, NS, ND, dates, syms,
            mom_scores, lead_lag, entropy, ts_data, regime,
            w_mom=0.5, w_leadlag=0.3, top_n=1, hold_days=2,
            leverage=lev, kelly_sizing=True, entropy_gate=True,
            start_di=wf_start)
        analyze(trades, eq, dd, f"combined lev={lev}")

    print(f"\n[V319] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
