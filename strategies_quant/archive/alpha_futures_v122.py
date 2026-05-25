"""
Alpha Futures V122 -- Z-Score Method Comparison (Next-Open Execution)
=====================================================================
FIND THE BEST Z-SCORE METHOD for futures momentum signals.

Base signal: ROC(5) > threshold, tested with 12 different Z-score methods.
All signals use NEXT-OPEN execution: signal at di, enter O[di+1], exit C[di+1].

Methods tested:
  A) Z-score of today's return vs 20-day return distribution (champion baseline)
  B) Z-score of today's return vs 10-day return distribution
  C) Z-score of today's return vs 60-day return distribution
  D) Z-score of 5-day return vs 60-day ROC(5) distribution
  E) Z-score of today's range vs 20-day range distribution
  F) Z-score of today's volume vs 20-day volume distribution
  G) Z-score of ROC(5) vs cross-sectional distribution (relative strength)
  H) Z-score of ROC(5) vs within-commodity 252-day distribution
  I) Combined Z-score: average of methods A + D
  J) Double Z-score: BOTH Z-return > threshold AND Z-roc5 > threshold
  K) Rank-based signal (no Z-score): rank jump from >40 to <10
  L) Percentile-based: top 10th pct of own 252d + top 50th pct cross-sectional
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0

# ============================================================
# CONSTANTS
# ============================================================
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


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0:
        return -100.0
    yrs = n_days / 252
    return (final / initial) ** (1.0 / yrs) * 100 - 100


def rolling_zscore(arr, window):
    """Compute rolling Z-score: (value - rolling_mean) / rolling_std.
    arr shape: (ND,). Returns shape (ND,) with NaN where insufficient data."""
    n = len(arr)
    z = np.full(n, np.nan)
    for i in range(window, n):
        w = arr[i - window:i]
        valid = w[~np.isnan(w)]
        if len(valid) >= max(window // 2, 5):
            m = np.mean(valid)
            s = np.std(valid, ddof=0)
            if s > 1e-10 and not np.isnan(arr[i]):
                z[i] = (arr[i] - m) / s
    return z


def main():
    print("=" * 140)
    print("Alpha Futures V122 -- Z-Score Method Comparison (Next-Open Execution)")
    print("=" * 140)

    # ── Load data ────────────────────────────────────────────────────
    print("\n[Data] Loading...", flush=True)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    years = sorted(set(d.year for d in dates))
    print(f"  {NS} commodities, {ND} days, years: {years}")

    # ================================================================
    # PRECOMPUTE BASE METRICS
    # ================================================================
    print("\n[Signals] Precomputing base metrics...", flush=True)
    t0 = time.time()

    # Daily returns: ret1[si, di] = (C[di] - C[di-1]) / C[di-1]
    ret1 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            cn = C[si, di]
            cp = C[si, di - 1]
            if not np.isnan(cn) and not np.isnan(cp) and cp > 0:
                ret1[si, di] = (cn - cp) / cp

    # ROC(5): roc5[si, di] = (C[di] - C[di-5]) / C[di-5]
    roc5 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            cn = C[si, di]
            cp = C[si, di - 5]
            if not np.isnan(cn) and not np.isnan(cp) and cp > 0:
                roc5[si, di] = (cn - cp) / cp

    # Range: (H - L) / C
    daily_range = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            h = H[si, di]
            l = L[si, di]
            c = C[si, di]
            if not np.isnan(h) and not np.isnan(l) and not np.isnan(c) and c > 0:
                daily_range[si, di] = (h - l) / c

    print(f"  ret1, roc5, range computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # COMPUTE Z-SCORES FOR EACH METHOD
    # ================================================================
    print("[Signals] Computing Z-score methods...", flush=True)

    # ── A) Z-score of today's return vs 20-day return distribution ──
    z_ret20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        z_ret20[si] = rolling_zscore(ret1[si], 20)
    print(f"  A) Z-ret vs 20d done ({time.time()-t0:.1f}s)")

    # ── B) Z-score of today's return vs 10-day return distribution ──
    z_ret10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        z_ret10[si] = rolling_zscore(ret1[si], 10)
    print(f"  B) Z-ret vs 10d done ({time.time()-t0:.1f}s)")

    # ── C) Z-score of today's return vs 60-day return distribution ──
    z_ret60 = np.full((NS, ND), np.nan)
    for si in range(NS):
        z_ret60[si] = rolling_zscore(ret1[si], 60)
    print(f"  C) Z-ret vs 60d done ({time.time()-t0:.1f}s)")

    # ── D) Z-score of ROC(5) vs 60-day ROC(5) distribution ──
    z_roc5_60 = np.full((NS, ND), np.nan)
    for si in range(NS):
        z_roc5_60[si] = rolling_zscore(roc5[si], 60)
    print(f"  D) Z-roc5 vs 60d done ({time.time()-t0:.1f}s)")

    # ── E) Z-score of today's range vs 20-day range distribution ──
    z_range20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        z_range20[si] = rolling_zscore(daily_range[si], 20)
    print(f"  E) Z-range vs 20d done ({time.time()-t0:.1f}s)")

    # ── F) Z-score of today's volume vs 20-day volume distribution ──
    z_vol20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        z_vol20[si] = rolling_zscore(V[si], 20)
    print(f"  F) Z-vol vs 20d done ({time.time()-t0:.1f}s)")

    # ── G) Z-score of ROC(5) vs cross-sectional distribution ──
    z_cross = np.full((NS, ND), np.nan)
    for di in range(ND):
        vals = roc5[:, di]
        valid = vals[~np.isnan(vals)]
        if len(valid) >= 10:
            m = np.mean(valid)
            s = np.std(valid, ddof=0)
            if s > 1e-10:
                for si in range(NS):
                    if not np.isnan(vals[si]):
                        z_cross[si, di] = (vals[si] - m) / s
    print(f"  G) Z-cross-sectional done ({time.time()-t0:.1f}s)")

    # ── H) Z-score of ROC(5) vs within-commodity 252-day distribution ──
    z_roc5_252 = np.full((NS, ND), np.nan)
    for si in range(NS):
        z_roc5_252[si] = rolling_zscore(roc5[si], 252)
    print(f"  H) Z-roc5 vs 252d done ({time.time()-t0:.1f}s)")

    # ── I) Combined: average of Z-ret20 and Z-roc5_60 ──
    z_combined = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            a = z_ret20[si, di]
            d = z_roc5_60[si, di]
            if not np.isnan(a) and not np.isnan(d):
                z_combined[si, di] = (a + d) / 2
    print(f"  I) Combined Z done ({time.time()-t0:.1f}s)")

    # ── K) Rank-based: compute daily cross-sectional rank of ROC(5) ──
    roc5_rank = np.full((NS, ND), np.nan)       # current rank
    roc5_rank_prev = np.full((NS, ND), np.nan)  # rank 5 days ago
    for di in range(ND):
        vals = roc5[:, di]
        valid_idx = [si for si in range(NS) if not np.isnan(vals[si])]
        if len(valid_idx) >= 20:
            sorted_idx = sorted(valid_idx, key=lambda si: vals[si], reverse=True)
            for rank, si in enumerate(sorted_idx):
                roc5_rank[si, di] = rank + 1  # 1 = best
    # Rank 5 days ago
    for di in range(5, ND):
        roc5_rank_prev[:, di] = roc5_rank[:, di - 5]
    print(f"  K) Rank done ({time.time()-t0:.1f}s)")

    # ── L) Percentile-based ──
    # L1: within-commodity percentile of ROC(5) over 252 days
    pct_within = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(252, ND):
            w = roc5[si, di - 252:di]
            valid = w[~np.isnan(w)]
            if len(valid) >= 50 and not np.isnan(roc5[si, di]):
                pct_within[si, di] = np.mean(valid < roc5[si, di])

    # L2: cross-sectional percentile of ROC(5)
    pct_cross = np.full((NS, ND), np.nan)
    for di in range(ND):
        vals = roc5[:, di]
        valid = vals[~np.isnan(vals)]
        if len(valid) >= 20:
            for si in range(NS):
                if not np.isnan(vals[si]):
                    pct_cross[si, di] = np.mean(valid < vals[si])
    print(f"  L) Percentile done ({time.time()-t0:.1f}s)")

    print(f"  All Z-score methods computed ({time.time()-t0:.1f}s)")

    # ================================================================
    # BACKTEST ENGINE
    # ================================================================
    def run_backtest(method, z_threshold, roc5_threshold=0.01, hold=1,
                     top_n=1, wf_test_year=None):
        """
        Generic backtest for Z-score methods.
        Signal at day di: ROC(5) > roc5_threshold AND Z-method > z_threshold
        Entry: O[si, di+1] (next-open)
        Exit: C[si, di+1+hold-1] = C[si, di+hold]
        """
        # Date boundaries
        if wf_test_year is not None:
            test_start_di = None
            test_end_di = None
            for di in range(ND):
                if dates[di].year == wf_test_year and test_start_di is None:
                    test_start_di = di
                if dates[di].year == wf_test_year + 1 and test_end_di is None:
                    test_end_di = di
            if test_start_di is None:
                return None
            if test_end_di is None:
                test_end_di = ND
            start_di = MIN_TRAIN
            end_di = test_end_di
        else:
            test_start_di = MIN_TRAIN
            start_di = MIN_TRAIN
            end_di = ND
            test_end_di = ND

        if end_di < start_di + hold + 2:
            return None

        cash = float(CASH0)
        positions = []
        trades = []

        # Select Z-score array for method
        # For method J, we need both z_ret20 and z_roc5_60
        z_arr = None
        if method == 'A': z_arr = z_ret20
        elif method == 'B': z_arr = z_ret10
        elif method == 'C': z_arr = z_ret60
        elif method == 'D': z_arr = z_roc5_60
        elif method == 'E': z_arr = z_range20
        elif method == 'F': z_arr = z_vol20
        elif method == 'G': z_arr = z_cross
        elif method == 'H': z_arr = z_roc5_252
        elif method == 'I': z_arr = z_combined
        # J, K, L handled specially below

        for di in range(start_di, end_di - hold - 1):
            # Reset cash at test window start (WF mode)
            if wf_test_year is not None and di == test_start_di:
                cash = float(CASH0)
                positions = []

            # Close positions whose hold period is up
            closed = []
            for pos in positions:
                if di >= pos['exit_di']:
                    exit_price = C[pos['si'], di]
                    if np.isnan(exit_price) or exit_price <= 0:
                        exit_price = pos['entry_price']
                    mult = MULT.get(pos['sym'], DEF_MULT)
                    mkt_val = exit_price * mult * abs(pos['lots'])
                    cash += mkt_val - mkt_val * COMM
                    pnl = (exit_price - pos['entry_price']) * mult * pos['lots']
                    invested = pos['entry_price'] * mult * abs(pos['lots'])
                    pnl_pct = pnl / invested * 100 if invested > 0 else 0
                    trades.append({
                        'pnl_pct': pnl_pct,
                        'entry_di': pos['entry_di'],
                        'exit_di': di,
                        'year': dates[di].year if di < ND else dates[-1].year,
                    })
                    closed.append(pos)
            for pos in closed:
                positions.remove(pos)

            # Generate signals at day di
            candidates = []

            if method in ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'):
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    r5 = roc5[si, di]
                    z = z_arr[si, di]
                    if np.isnan(r5) or np.isnan(z):
                        continue
                    if r5 > roc5_threshold and z > z_threshold:
                        candidates.append((si, z, syms[si]))

            elif method == 'J':
                # Double Z: BOTH Z-ret20 > threshold AND Z-roc5_60 > threshold
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    r5 = roc5[si, di]
                    za = z_ret20[si, di]
                    zd = z_roc5_60[si, di]
                    if np.isnan(r5) or np.isnan(za) or np.isnan(zd):
                        continue
                    if r5 > roc5_threshold and za > z_threshold and zd > z_threshold:
                        score = min(za, zd)
                        candidates.append((si, score, syms[si]))

            elif method == 'K':
                # Rank-based: rank jumped from >40 to <10
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    r5 = roc5[si, di]
                    rk_now = roc5_rank[si, di]
                    rk_prev = roc5_rank_prev[si, di]
                    if np.isnan(r5) or np.isnan(rk_now) or np.isnan(rk_prev):
                        continue
                    if r5 > roc5_threshold and rk_prev > 40 and rk_now < 10:
                        score = rk_prev - rk_now  # bigger jump = better
                        candidates.append((si, score, syms[si]))

            elif method == 'L':
                # Percentile: top 10th pct own 252d AND top 50th pct cross-sectional
                for si in range(NS):
                    if any(p['si'] == si for p in positions):
                        continue
                    r5 = roc5[si, di]
                    pw = pct_within[si, di]
                    pc = pct_cross[si, di]
                    if np.isnan(r5) or np.isnan(pw) or np.isnan(pc):
                        continue
                    if r5 > roc5_threshold and pw > 0.90 and pc > 0.50:
                        score = pw + pc
                        candidates.append((si, score, syms[si]))

            if not candidates:
                continue

            # Sort by score descending
            candidates.sort(key=lambda x: -x[1])

            # Entry at next open: O[si, di+1], exit at C[si, di+1+hold-1]
            entry_di = di + 1
            exit_di_target = entry_di + hold - 1

            if entry_di >= end_di or exit_di_target >= end_di:
                continue

            n_slots = top_n - len(positions)
            for si, score, sym in candidates[:max(0, n_slots)]:
                price = O[si, entry_di]
                if np.isnan(price) or price <= 0:
                    continue
                mult = MULT.get(sym, DEF_MULT)
                notional = price * mult
                lots = int(cash / (notional * (1 + COMM)))
                if lots <= 0:
                    continue
                cost_in = notional * lots * (1 + COMM)
                if cost_in > cash:
                    lots = int(cash * 0.95 / (notional * (1 + COMM)))
                    cost_in = notional * lots * (1 + COMM) if lots > 0 else 0
                if lots <= 0 or cost_in <= 0 or cost_in > cash:
                    continue

                cash -= cost_in
                positions.append({
                    'si': si, 'entry_price': price, 'entry_di': entry_di,
                    'exit_di': exit_di_target + 1,  # close when di >= exit_di
                    'lots': lots, 'sym': sym,
                })

        # Close remaining positions
        for pos in positions:
            ae = end_di - 1 if end_di < ND else ND - 1
            exit_price = C[pos['si'], ae]
            if np.isnan(exit_price) or exit_price <= 0:
                exit_price = pos['entry_price']
            mult = MULT.get(pos['sym'], DEF_MULT)
            mkt_val = exit_price * mult * abs(pos['lots'])
            cash += mkt_val - mkt_val * COMM

        # Results
        wf_mode = wf_test_year is not None
        n_days_test = (test_end_di - test_start_di) if wf_mode else (end_di - start_di)
        ann = annual_return(cash, CASH0, n_days_test)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        n_trades = len(trades)
        avg_pnl = np.mean([t['pnl_pct'] for t in trades]) if trades else 0

        # Max drawdown
        eq = float(CASH0)
        peak = eq
        mdd = 0.0
        for t in trades:
            eq *= (1 + t['pnl_pct'] / 100)
            if eq > peak:
                peak = eq
            dd = (eq - peak) / peak * 100
            if dd < mdd:
                mdd = dd

        return {
            'ann': ann, 'wr': wr, 'n': n_trades, 'avg_pnl': avg_pnl,
            'final_cash': cash, 'n_days': n_days_test, 'mdd': mdd,
        }

    # ================================================================
    # BUILD CONFIGURATIONS
    # ================================================================
    print("\n[Sweep] Building configurations...", flush=True)
    configs = []
    cid = 0

    # A) Z-ret vs 20d (champion baseline)
    for thresh in [1.0, 1.5, 2.0]:
        cid += 1
        configs.append({
            'id': cid, 'method': 'A', 'z_threshold': thresh,
            'roc5_threshold': 0.01, 'hold': 1, 'top_n': 1,
            'label': f"A_Zret20_T{thresh}",
            'desc': f"Z-ret vs 20d, thr={thresh}",
        })

    # B) Z-ret vs 10d
    for thresh in [1.0, 1.5, 2.0]:
        cid += 1
        configs.append({
            'id': cid, 'method': 'B', 'z_threshold': thresh,
            'roc5_threshold': 0.01, 'hold': 1, 'top_n': 1,
            'label': f"B_Zret10_T{thresh}",
            'desc': f"Z-ret vs 10d, thr={thresh}",
        })

    # C) Z-ret vs 60d
    for thresh in [1.0, 1.5, 2.0]:
        cid += 1
        configs.append({
            'id': cid, 'method': 'C', 'z_threshold': thresh,
            'roc5_threshold': 0.01, 'hold': 1, 'top_n': 1,
            'label': f"C_Zret60_T{thresh}",
            'desc': f"Z-ret vs 60d, thr={thresh}",
        })

    # D) Z-roc5 vs 60d
    for thresh in [0.5, 1.0, 1.5, 2.0]:
        cid += 1
        configs.append({
            'id': cid, 'method': 'D', 'z_threshold': thresh,
            'roc5_threshold': 0.01, 'hold': 1, 'top_n': 1,
            'label': f"D_Zroc5_60d_T{thresh}",
            'desc': f"Z-roc5 vs 60d, thr={thresh}",
        })

    # E) Z-range vs 20d
    for thresh in [0.5, 1.0, 1.5]:
        cid += 1
        configs.append({
            'id': cid, 'method': 'E', 'z_threshold': thresh,
            'roc5_threshold': 0.01, 'hold': 1, 'top_n': 1,
            'label': f"E_Zrange20_T{thresh}",
            'desc': f"Z-range vs 20d, thr={thresh}",
        })

    # F) Z-vol vs 20d
    for thresh in [1.0, 1.5, 2.0]:
        cid += 1
        configs.append({
            'id': cid, 'method': 'F', 'z_threshold': thresh,
            'roc5_threshold': 0.01, 'hold': 1, 'top_n': 1,
            'label': f"F_Zvol20_T{thresh}",
            'desc': f"Z-vol vs 20d, thr={thresh}",
        })

    # G) Z-cross-sectional ROC(5)
    for thresh in [0.5, 1.0, 1.5]:
        cid += 1
        configs.append({
            'id': cid, 'method': 'G', 'z_threshold': thresh,
            'roc5_threshold': 0.01, 'hold': 1, 'top_n': 1,
            'label': f"G_Zcross_T{thresh}",
            'desc': f"Z-cross-sectional, thr={thresh}",
        })

    # H) Z-roc5 vs 252d
    for thresh in [0.5, 1.0, 1.5]:
        cid += 1
        configs.append({
            'id': cid, 'method': 'H', 'z_threshold': thresh,
            'roc5_threshold': 0.01, 'hold': 1, 'top_n': 1,
            'label': f"H_Zroc5_252d_T{thresh}",
            'desc': f"Z-roc5 vs 252d, thr={thresh}",
        })

    # I) Combined Z (avg of A + D)
    for thresh in [0.75, 1.0, 1.5]:
        cid += 1
        configs.append({
            'id': cid, 'method': 'I', 'z_threshold': thresh,
            'roc5_threshold': 0.01, 'hold': 1, 'top_n': 1,
            'label': f"I_Combined_T{thresh}",
            'desc': f"Combined Z (A+D)/2, thr={thresh}",
        })

    # J) Double Z (both A and D must pass)
    for thresh in [0.5, 1.0, 1.5]:
        cid += 1
        configs.append({
            'id': cid, 'method': 'J', 'z_threshold': thresh,
            'roc5_threshold': 0.01, 'hold': 1, 'top_n': 1,
            'label': f"J_DoubleZ_T{thresh}",
            'desc': f"Double Z (A AND D), thr={thresh}",
        })

    # K) Rank-based (no Z-score)
    cid += 1
    configs.append({
        'id': cid, 'method': 'K', 'z_threshold': 0,
        'roc5_threshold': 0.01, 'hold': 1, 'top_n': 1,
        'label': "K_RankJump",
        'desc': "Rank jumped >40 to <10",
    })

    # L) Percentile-based
    cid += 1
    configs.append({
        'id': cid, 'method': 'L', 'z_threshold': 0,
        'roc5_threshold': 0.01, 'hold': 1, 'top_n': 1,
        'label': "L_Percentile",
        'desc': "Top 10% own + Top 50% cross",
    })

    print(f"  Total configs: {len(configs)}")

    # ================================================================
    # RUN FULL-PERIOD BACKTEST
    # ================================================================
    print("\n[Backtest] Running full-period sweep...", flush=True)
    results = []
    for i, cfg in enumerate(configs):
        r = run_backtest(
            cfg['method'], cfg['z_threshold'],
            roc5_threshold=cfg['roc5_threshold'],
            hold=cfg['hold'], top_n=cfg['top_n'],
        )
        if r:
            r['config'] = cfg
            r['label'] = cfg['label']
            r['desc'] = cfg['desc']
            results.append(r)
        if (i + 1) % 5 == 0 or i == len(configs) - 1:
            print(f"  ... {i+1}/{len(configs)} done", flush=True)

    results.sort(key=lambda x: -x['ann'])

    # ================================================================
    # FULL-PERIOD RESULTS
    # ================================================================
    print(f"\n{'=' * 140}")
    print("  FULL-PERIOD RESULTS (All methods sorted by annual return)")
    print(f"{'=' * 140}")
    print(f"  {'#':>3} | {'Method':<20} | {'Label':<25} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | {'Final':>12}")
    print("-" * 140)
    for i, r in enumerate(results):
        print(f"  {i+1:>3} | {r['config']['method']:>20} | {r['label']:<25} | {r['ann']:>+8.1f}% | {r['wr']:>5.1f}% | {r['n']:>5} | {r['avg_pnl']:>+6.3f}% | {r['mdd']:>6.1f}% | {r['final_cash']:>11.0f}")

    # ================================================================
    # BEST PER METHOD
    # ================================================================
    print(f"\n{'=' * 140}")
    print("  BEST CONFIG PER METHOD")
    print(f"{'=' * 140}")
    print(f"  {'Method':<6} | {'Description':<35} | {'Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | Label")
    print("-" * 140)

    best_per_method = {}
    for r in results:
        m = r['config']['method']
        if m not in best_per_method or r['ann'] > best_per_method[m]['ann']:
            best_per_method[m] = r

    method_names = {
        'A': 'Z-ret vs 20d (CHAMPION)',
        'B': 'Z-ret vs 10d',
        'C': 'Z-ret vs 60d',
        'D': 'Z-roc5 vs 60d',
        'E': 'Z-range vs 20d',
        'F': 'Z-vol vs 20d',
        'G': 'Z-cross-sectional',
        'H': 'Z-roc5 vs 252d',
        'I': 'Combined (A+D)/2',
        'J': 'Double Z (A AND D)',
        'K': 'Rank jump',
        'L': 'Percentile',
    }

    for m in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
        if m in best_per_method:
            b = best_per_method[m]
            print(f"  {m:<6} | {method_names[m]:<35} | {b['ann']:>+8.1f}% | {b['wr']:>5.1f}% | {b['n']:>5} | {b['avg_pnl']:>+6.3f}% | {b['mdd']:>6.1f}% | {b['label']}")

    # ================================================================
    # WALK-FORWARD (Best per method + top 5 overall)
    # ================================================================
    wf_years = [2020, 2021, 2022, 2023, 2024, 2025]

    # Collect configs for WF: best per method + top 5 overall
    wf_configs = []
    seen_methods = set()
    for r in results[:5]:
        if r['config'] not in [w for w in wf_configs]:
            wf_configs.append(r['config'])
            seen_methods.add(r['config']['method'])
    for m in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
        if m in best_per_method and m not in seen_methods:
            wf_configs.append(best_per_method[m]['config'])
            seen_methods.add(m)

    print(f"\n{'=' * 160}")
    print(f"  WALK-FORWARD ({len(wf_configs)} configs)")
    print(f"{'=' * 160}")

    header = f"  {'#':>3} | {'Label':<25} | {'Avg':>8} |"
    for yr in wf_years:
        header += f" {yr:>8} |"
    header += f" {'Pos':>4} | {'AvgMDD':>7}"
    print(header)
    print("-" * 160)

    wf_rows = []
    for i, cfg in enumerate(wf_configs):
        wf_row = {'label': cfg['label'], 'method': cfg['method'],
                  'windows': {}, 'mdd': {}}
        for yr in wf_years:
            wr = run_backtest(
                cfg['method'], cfg['z_threshold'],
                roc5_threshold=cfg['roc5_threshold'],
                hold=cfg['hold'], top_n=cfg['top_n'],
                wf_test_year=yr,
            )
            if wr:
                wf_row['windows'][yr] = wr['ann']
                wf_row['mdd'][yr] = wr['mdd']
        wf_rows.append(wf_row)

        vals = [wf_row['windows'].get(yr, 0) for yr in wf_years]
        avg = np.mean(vals) if vals else 0
        pos = sum(1 for v in vals if v > 0)
        avg_mdd = np.mean(list(wf_row['mdd'].values())) if wf_row['mdd'] else 0

        row_str = f"  {i+1:>3} | {wf_row['label']:<25} | {avg:>+7.1f}% |"
        for v in vals:
            row_str += f" {v:>+7.1f}% |"
        row_str += f" {pos}/6 | {avg_mdd:>6.1f}%"
        print(row_str)

    # ================================================================
    # ANALYSIS: WINDOW COMPARISON (10d vs 20d vs 60d)
    # ================================================================
    print(f"\n{'=' * 140}")
    print("  ANALYSIS: Z-SCORE WINDOW COMPARISON (10d vs 20d vs 60d for return Z-score)")
    print(f"{'=' * 140}")
    print(f"  {'Window':>8} | {'Best Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | WF Avg | WF Pos/6")
    print("-" * 140)

    for m, label in [('B', '10-day'), ('A', '20-day'), ('C', '60-day')]:
        if m in best_per_method:
            b = best_per_method[m]
            # Find WF for this method
            wf_match = [w for w in wf_rows if w['method'] == m]
            if wf_match:
                wm = wf_match[0]
                wf_avg = np.mean([wm['windows'].get(yr, 0) for yr in wf_years])
                wf_pos = sum(1 for yr in wf_years if wm['windows'].get(yr, 0) > 0)
            else:
                wf_avg = 0
                wf_pos = 0
            print(f"  {label:>8} | {b['ann']:>+8.1f}% | {b['wr']:>5.1f}% | {b['n']:>5} | {b['avg_pnl']:>+6.3f}% | {b['mdd']:>6.1f}% | {wf_avg:>+7.1f}% | {wf_pos}/6")

    # ================================================================
    # ANALYSIS: CROSS-SECTIONAL vs TIME-SERIES
    # ================================================================
    print(f"\n{'=' * 140}")
    print("  ANALYSIS: CROSS-SECTIONAL vs TIME-SERIES Z-SCORE")
    print(f"{'=' * 140}")
    print(f"  {'Type':<30} | {'Best Ann':>9} | {'WR':>6} | {'N':>5} | {'AvgPnL':>7} | {'MDD':>7} | Method")
    print("-" * 140)

    ts_methods = {'A': 'Time-series (20d)', 'B': 'Time-series (10d)', 'C': 'Time-series (60d)',
                  'D': 'TS-ROC5 (60d)', 'H': 'TS-ROC5 (252d)'}
    cs_methods = {'G': 'Cross-sectional', 'K': 'Rank-based', 'L': 'Percentile'}
    combo_methods = {'I': 'Combined (A+D)/2', 'J': 'Double Z (A AND D)'}

    for methods_dict, category in [(ts_methods, 'TIME-SERIES'), (cs_methods, 'CROSS-SECTIONAL'),
                                    (combo_methods, 'COMBINED')]:
        print(f"\n  --- {category} ---")
        for m, desc in methods_dict.items():
            if m in best_per_method:
                b = best_per_method[m]
                print(f"  {desc:<30} | {b['ann']:>+8.1f}% | {b['wr']:>5.1f}% | {b['n']:>5} | {b['avg_pnl']:>+6.3f}% | {b['mdd']:>6.1f}% | {m}")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print(f"\n{'=' * 140}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 140}")

    if results:
        champion = results[0]
        print(f"\n  BEST METHOD OVERALL: {champion['label']} ({method_names[champion['config']['method']]})")
        print(f"    Annual Return: {champion['ann']:+.1f}%")
        print(f"    Win Rate:      {champion['wr']:.1f}%")
        print(f"    Trades:        {champion['n']}")
        print(f"    Avg PnL:       {champion['avg_pnl']:+.3f}%")
        print(f"    Max Drawdown:  {champion['mdd']:.1f}%")

        # Does it beat +306.2%?
        if champion['ann'] > 306.2:
            print(f"\n  *** YES! Beats champion +306.2% by {champion['ann'] - 306.2:.1f}pp ***")
        else:
            print(f"\n  Does NOT beat champion +306.2% (gap: {306.2 - champion['ann']:.1f}pp)")

        # WF summary
        if wf_rows:
            best_wf = max(wf_rows, key=lambda w: np.mean([w['windows'].get(yr, 0) for yr in wf_years]))
            wf_avg = np.mean([best_wf['windows'].get(yr, 0) for yr in wf_years])
            wf_pos = sum(1 for yr in wf_years if best_wf['windows'].get(yr, 0) > 0)
            print(f"\n  BEST WF: {best_wf['label']} avg={wf_avg:+.1f}% positive={wf_pos}/6")

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed: {elapsed:.1f}s")
    print("=" * 140)


if __name__ == '__main__':
    main()
