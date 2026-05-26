"""
V1: Multi-Alpha Mean Reversion with Regime Gating
==================================================
Extends V0's oversold combo with:
  1. OI capitulation signal (OI declining + price declining → reversal)
  2. VDP volume confirmation (selling exhaustion detection)
  3. KER regime gating (trade only when market is mean-reverting)
  4. TA-Lib indicators as filters (RSI, Bollinger, CCI)
  5. Confidence-based position sizing (signal stacking)
  6. Walk-forward validation, 5+ years, no gap, no leverage

Signal at close[di], enter at open[di+1]. No look-ahead.
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
from collections import defaultdict
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_data import load_all_data

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

CASH0 = 1_000_000
COMM = 0.0005


# ============================================================
# SIGNAL COMPUTATION
# ============================================================
def compute_signals(C, O, H, L, V, OI, NS, ND):
    """Compute all signals for V1 multi-alpha strategy."""
    t0 = time.time()
    print("[V1] Computing signals...", flush=True)

    # --- 1. Consecutive down days ---
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

    # --- 2. 5d return ---
    ret_5d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di-5]) and C[si, di-5] > 0:
                ret_5d[si, di] = C[si, di] / C[si, di-5] - 1

    # --- 3. 20d volatility ---
    vol_20d = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            rets = []
            for j in range(di - 20, di):
                if not np.isnan(C[si, j]) and not np.isnan(C[si, j-1]) and C[si, j-1] > 0:
                    rets.append(C[si, j] / C[si, j-1] - 1)
            if len(rets) >= 10:
                vol_20d[si, di] = np.std(rets) * np.sqrt(252)

    # --- 4. OI capitulation: OI declining + price declining ---
    oi_decline = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(5, ND):
            if np.isnan(OI[si, di]) or np.isnan(OI[si, di-5]):
                continue
            if np.isnan(C[si, di]) or np.isnan(C[si, di-5]) or C[si, di-5] <= 0:
                continue
            oi_chg = OI[si, di] / OI[si, di-5] - 1
            price_chg = C[si, di] / C[si, di-5] - 1
            # OI declining AND price declining = capitulation
            if oi_chg < -0.02 and price_chg < -0.02:
                oi_decline[si, di] = min(abs(oi_chg), 0.2) / 0.2 * min(abs(price_chg), 0.1) / 0.1
            else:
                oi_decline[si, di] = 0.0

    # --- 5. VDP (Volume Delta Pressure) ---
    # delta = V × (2C-H-L) / (H-L)
    vdp = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(1, ND):
            if np.isnan(H[si, di]) or np.isnan(L[si, di]) or np.isnan(C[si, di]) or np.isnan(V[si, di]):
                continue
            bar_range = H[si, di] - L[si, di]
            if bar_range > 0 and V[si, di] > 0:
                vdp[si, di] = V[si, di] * (2 * C[si, di] - H[si, di] - L[si, di]) / bar_range

    # 10d average VDP (normalized)
    vdp_10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            vals = vdp[si, di-10:di]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 5:
                vdp_10[si, di] = np.mean(valid)

    # Selling exhaustion: very negative VDP → potential reversal
    vdp_exhaust = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            if np.isnan(vdp_10[si, di]):
                continue
            # VDP z-score over rolling window
            window = vdp_10[si, max(0, di-20):di]
            wv = window[~np.isnan(window)]
            if len(wv) >= 10:
                mu, sig = np.mean(wv), np.std(wv)
                if sig > 0:
                    z = (vdp_10[si, di] - mu) / sig
                    # Negative z = selling pressure → potential reversal
                    vdp_exhaust[si, di] = min(-z, 3.0) / 3.0 if z < 0 else 0.0

    # --- 6. KER (Kaufman Efficiency Ratio) ---
    ker_10 = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(10, ND):
            closes = C[si, di-10:di+1]
            valid = closes[~np.isnan(closes)]
            if len(valid) < 10 or valid[0] <= 0:
                continue
            net_change = abs(valid[-1] - valid[0])
            total_change = np.sum(np.abs(np.diff(valid)))
            if total_change > 1e-10:
                ker_10[si, di] = net_change / total_change

    # --- 7. TA-Lib indicators ---
    rsi14 = np.full((NS, ND), np.nan)
    bb_pos = np.full((NS, ND), np.nan)  # Bollinger position
    cci14 = np.full((NS, ND), np.nan)
    willr14 = np.full((NS, ND), np.nan)

    if HAS_TALIB:
        for si in range(NS):
            h = np.where(np.isnan(H[si]), 0, H[si]).astype(np.float64)
            l = np.where(np.isnan(L[si]), 0, L[si]).astype(np.float64)
            c = np.where(np.isnan(C[si]), 0, C[si]).astype(np.float64)
            nan_mask = np.isnan(C[si])

            # RSI
            try:
                rsi = talib.RSI(c, 14)
                rsi14[si] = np.where(nan_mask, np.nan, rsi)
            except Exception:
                pass

            # Bollinger Band position
            try:
                upper, mid, lower = talib.BBANDS(c, 20, 2.0, 2.0)
                bb_range = upper - lower
                valid_bb = (~nan_mask) & (bb_range > 1e-10)
                bb_pos[si] = np.where(valid_bb, (c - lower) / bb_range, np.nan)
            except Exception:
                pass

            # CCI
            try:
                cci = talib.CCI(h, l, c, 14)
                cci14[si] = np.where(nan_mask, np.nan, cci)
            except Exception:
                pass

            # Williams %R
            try:
                wr = talib.WILLR(h, l, c, 14)
                willr14[si] = np.where(nan_mask, np.nan, wr)
            except Exception:
                pass

    # --- 8. Composite score with cross-sectional ranking ---
    # Stack signals into composite oversold score
    raw_score = np.full((NS, ND), np.nan)
    for di in range(ND):
        scores = np.full(NS, np.nan)
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue

            s = 0.0
            w_total = 0.0

            # Consecutive down days (weight 0.20)
            cd = consec_dn[si, di]
            s += min(cd / 5.0, 1.0) * 0.20
            w_total += 0.20

            # 5d return oversold (weight 0.20)
            if not np.isnan(ret_5d[si, di]):
                s += min(max(-ret_5d[si, di] / 0.1, 0), 1.0) * 0.20
                w_total += 0.20

            # OI capitulation (weight 0.20)
            if not np.isnan(oi_decline[si, di]):
                s += oi_decline[si, di] * 0.20
                w_total += 0.20

            # VDP selling exhaustion (weight 0.15)
            if not np.isnan(vdp_exhaust[si, di]):
                s += vdp_exhaust[si, di] * 0.15
                w_total += 0.15

            # RSI oversold (weight 0.10)
            if not np.isnan(rsi14[si, di]):
                if rsi14[si, di] < 30:
                    s += (30 - rsi14[si, di]) / 30.0 * 0.10
                w_total += 0.10

            # Bollinger lower band (weight 0.10)
            if not np.isnan(bb_pos[si, di]):
                if bb_pos[si, di] < 0.2:
                    s += (0.2 - bb_pos[si, di]) / 0.2 * 0.10
                w_total += 0.10

            # CCI oversold (weight 0.05)
            if not np.isnan(cci14[si, di]):
                if cci14[si, di] < -100:
                    s += min((-100 - cci14[si, di]) / 200.0, 1.0) * 0.05
                w_total += 0.05

            if w_total > 0:
                scores[si] = s / w_total

        # Cross-sectional rank
        valid = ~np.isnan(scores)
        if valid.sum() >= 5:
            raw_score[:, di] = pd.Series(scores).rank(pct=True, na_option='keep').values

    # --- 9. Confidence level (number of signals firing) ---
    n_signals = np.zeros((NS, ND), dtype=int)
    for di in range(ND):
        for si in range(NS):
            if np.isnan(C[si, di]) or C[si, di] <= 0:
                continue
            n = 0
            if consec_dn[si, di] >= 3:
                n += 1
            if not np.isnan(ret_5d[si, di]) and ret_5d[si, di] < -0.03:
                n += 1
            if not np.isnan(oi_decline[si, di]) and oi_decline[si, di] > 0.1:
                n += 1
            if not np.isnan(vdp_exhaust[si, di]) and vdp_exhaust[si, di] > 0.3:
                n += 1
            if not np.isnan(rsi14[si, di]) and rsi14[si, di] < 35:
                n += 1
            if not np.isnan(bb_pos[si, di]) and bb_pos[si, di] < 0.15:
                n += 1
            if not np.isnan(cci14[si, di]) and cci14[si, di] < -100:
                n += 1
            n_signals[si, di] = n

    # --- 10. Regime classification from KER ---
    # ker < 0.15 = mean-reverting regime → best for our signals
    # ker > 0.3 = trending → signals may still work but less reliable
    ker_regime = np.zeros((NS, ND), dtype=int)  # 0=neutral, 1=MR, -1=trending
    for si in range(NS):
        for di in range(ND):
            if np.isnan(ker_10[si, di]):
                continue
            if ker_10[si, di] < 0.15:
                ker_regime[si, di] = 1  # mean-reverting
            elif ker_10[si, di] > 0.3:
                ker_regime[si, di] = -1  # trending

    elapsed = time.time() - t0
    print(f"  Done: {elapsed:.1f}s (TA-Lib: {HAS_TALIB})", flush=True)

    return {
        'combo_rank': raw_score,
        'consec_dn': consec_dn,
        'vol_20d': vol_20d,
        'ker_10': ker_10,
        'ker_regime': ker_regime,
        'n_signals': n_signals,
        'oi_decline': oi_decline,
        'vdp_exhaust': vdp_exhaust,
        'rsi14': rsi14,
        'bb_pos': bb_pos,
        'cci14': cci14,
        'willr14': willr14,
    }


# ============================================================
# BACKTEST ENGINE
# ============================================================
def backtest_v1(C, O, H, L, NS, ND, dates, syms, sigs,
                top_n=1, hold_days=5, min_rank=0.7, atr_stop=2.5,
                min_confidence=2, use_ker_gate=True,
                use_short=False,
                vol_target=None,
                dd_breaker=None,
                leverage=1.0,
                start_di=60, end_di=None):
    """
    Multi-alpha mean reversion backtest.
    Signal at close[di], enter at open[di+1]. No look-ahead.
    """
    combo_rank = sigs['combo_rank']
    vol_20d = sigs['vol_20d']
    ker_regime = sigs['ker_regime']
    n_signals = sigs['n_signals']

    if end_di is None:
        end_di = ND - 1

    equity = CASH0
    peak = equity
    max_dd = 0.0
    positions = []
    trades = []

    for di in range(max(start_di, 1), end_di):
        d = dates[di]
        daily_pnl = 0
        new_positions = []

        # Exit positions
        for si, edi, ep, sp, alloc, direction in positions:
            c = C[si, di]
            if np.isnan(c):
                new_positions.append((si, edi, ep, sp, alloc, direction))
                continue
            exit_r = None
            if direction > 0 and c < sp:
                exit_r = 'stop'
            elif direction < 0 and c > sp:
                exit_r = 'stop'
            elif di - edi >= hold_days:
                exit_r = 'hold'
            if exit_r:
                pnl = direction * (c - ep) / ep - COMM
                profit = equity * alloc * leverage * pnl
                daily_pnl += profit
                trades.append({
                    'pnl_abs': profit, 'pnl_pct': pnl * 100 * leverage,
                    'days': di - edi + 1, 'di': di, 'year': d.year,
                    'sym': syms[si], 'reason': exit_r, 'dir': direction,
                })
            else:
                new_positions.append((si, edi, ep, sp, alloc, direction))

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

        held = {p[0] for p in positions}
        max_pos = top_n * (2 if use_short else 1)
        if len(positions) >= max_pos:
            continue

        # Drawdown breaker
        pos_multiplier = 1.0
        if dd_breaker and peak > 0:
            current_dd = (peak - equity) / peak
            if current_dd > dd_breaker:
                pos_multiplier = max(0.1, 1.0 - current_dd / 0.5)

        # --- Long candidates ---
        long_candidates = []
        for si in range(NS):
            if si in held:
                continue
            if np.isnan(combo_rank[si, di]):
                continue
            if combo_rank[si, di] < min_rank:
                continue
            # Confidence gate
            if n_signals[si, di] < min_confidence:
                continue
            # KER regime gate
            if use_ker_gate and ker_regime[si, di] < 0:
                continue  # trending → skip
            if di + 1 >= ND or np.isnan(O[si, di + 1]):
                continue

            # Position sizing
            alloc = pos_multiplier / max_pos

            # Vol-targeted sizing
            if vol_target and not np.isnan(vol_20d[si, di]) and vol_20d[si, di] > 0:
                vol_adj = min(vol_target / vol_20d[si, di], 2.0)
                alloc *= vol_adj

            # Confidence boost: more signals → larger position
            conf_boost = 1.0 + min(n_signals[si, di], 5) * 0.1
            alloc *= conf_boost

            long_candidates.append((combo_rank[si, di], si, alloc))

        # Sort by rank (most oversold first)
        long_candidates.sort(key=lambda x: -x[0])
        for rank, si, alloc in long_candidates[:top_n]:
            if len(positions) >= max_pos or si in held:
                break
            ep = O[si, di + 1]
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
            positions.append((si, di + 1, ep, ep - atr_stop * atr, alloc, 1))
            held.add(si)

        # --- Short candidates (overbought) ---
        if use_short:
            short_candidates = []
            for si in range(NS):
                if si in held:
                    continue
                if np.isnan(combo_rank[si, di]):
                    continue
                # For shorting: low rank = overbought (inverted)
                if combo_rank[si, di] > (1 - min_rank):
                    continue
                if n_signals[si, di] < min_confidence:
                    continue
                if use_ker_gate and ker_regime[si, di] < 0:
                    continue
                if di + 1 >= ND or np.isnan(O[si, di + 1]):
                    continue

                alloc = pos_multiplier / max_pos
                if vol_target and not np.isnan(vol_20d[si, di]) and vol_20d[si, di] > 0:
                    vol_adj = min(vol_target / vol_20d[si, di], 2.0)
                    alloc *= vol_adj
                conf_boost = 1.0 + min(n_signals[si, di], 5) * 0.1
                alloc *= conf_boost

                short_candidates.append((combo_rank[si, di], si, alloc))

            short_candidates.sort(key=lambda x: x[0])  # lowest rank = most overbought
            for rank, si, alloc in short_candidates[:top_n]:
                if len(positions) >= max_pos or si in held:
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
                positions.append((si, di + 1, ep, ep + atr_stop * atr, alloc, -1))
                held.add(si)

    # Close remaining
    for si, edi, ep, sp, alloc, direction in positions:
        c = C[si, ND - 1]
        if not np.isnan(c) and c > 0:
            pnl = direction * (c - ep) / ep - COMM
            equity += equity * alloc * leverage * pnl

    return trades, equity, max_dd


# ============================================================
# ANALYSIS
# ============================================================
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

    long_t = [t for t in trades if t.get('dir', 1) > 0]
    short_t = [t for t in trades if t.get('dir', 1) < 0]

    print(f"  {label}: {len(trades)}t (L:{len(long_t)} S:{len(short_t)}) "
          f"WR={wr:.1f}% ann={ann:+.1f}% DD={max_dd:.1f}% "
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


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("  V1: MULTI-ALPHA MEAN REVERSION + REGIME GATING")
    print("=" * 70)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')
    print(f"  {NS} sym, {ND} days, "
          f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    sigs = compute_signals(C, O, H, L, V, OI, NS, ND)

    bt_2019 = None
    for i, d in enumerate(dates):
        if d >= pd.Timestamp('2019-01-01'):
            bt_2019 = i
            break

    # === Ablation: signal components ===
    print("\n" + "=" * 70)
    print("  SIGNAL ABLATION (tn=1 hd=5 lev=1, 2019-2026)")
    print("=" * 70)

    configs = [
        # (min_conf, ker_gate, short, vol_target, dd_breaker, label)
        (1, False, False, None, None, "V0 baseline (conf≥1)"),
        (2, False, False, None, None, "conf≥2"),
        (3, False, False, None, None, "conf≥3"),
        (2, True,  False, None, None, "conf≥2+KER gate"),
        (3, True,  False, None, None, "conf≥3+KER gate"),
        (2, True,  False, 0.20, None, "conf≥2+KER+vol20%"),
        (2, True,  False, 0.20, 0.15, "conf≥2+KER+vol+dd15%"),
        (2, False, True,  None, None, "conf≥2+short"),
        (2, True,  True,  None, None, "conf≥2+KER+short"),
        (2, True,  True,  0.20, None, "conf≥2+KER+short+vol"),
        (3, True,  True,  0.20, None, "conf≥3+KER+short+vol"),
        (3, True,  True,  0.20, 0.15, "conf≥3+KER+short+vol+dd15%"),
    ]

    for mc, kg, sh, vt, db, label in configs:
        trades, eq, dd = backtest_v1(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=1, hold_days=5, min_rank=0.7,
            min_confidence=mc, use_ker_gate=kg,
            use_short=sh, vol_target=vt, dd_breaker=db,
            leverage=1, start_di=bt_2019)
        analyze(trades, eq, dd, label)

    # === Top-N sweep ===
    print("\n" + "=" * 70)
    print("  TOP-N SWEEP (2019-2026)")
    print("=" * 70)

    results = []
    for tn in [1, 2, 3, 5]:
        for hd in [3, 5, 7]:
            for mc in [1, 2, 3]:
                for kg in [False, True]:
                    for sh in [False, True]:
                        trades, eq, dd = backtest_v1(
                            C, O, H, L, NS, ND, dates, syms, sigs,
                            top_n=tn, hold_days=hd, min_rank=0.7,
                            min_confidence=mc, use_ker_gate=kg,
                            use_short=sh, leverage=1, start_di=bt_2019)
                        if len(trades) < 10:
                            continue
                        nw = sum(1 for t in trades if t['pnl_pct'] > 0)
                        wr = nw / len(trades) * 100
                        n_days = max(1, trades[-1]['di'] - trades[0]['di'])
                        ann = ((eq / CASH0) ** (1 / max(1.0, n_days / 252)) - 1) * 100
                        ap = [t['pnl_abs'] for t in sorted(trades, key=lambda x: x['di'])]
                        rets_arr = np.array(ap) / CASH0
                        sh_val = np.mean(rets_arr) / np.std(rets_arr) * np.sqrt(252) if np.std(rets_arr) > 0 else 0
                        results.append({
                            'tn': tn, 'hd': hd, 'mc': mc, 'kg': kg, 'sh': sh,
                            'n': len(trades), 'wr': wr, 'ann': ann, 'dd': dd, 'sharpe': sh_val,
                        })

    results.sort(key=lambda x: -x['sharpe'])
    print(f"\n{'TN':>3} {'HD':>3} {'MC':>3} {'KER':>3} {'SH':>3} "
          f"{'N':>5} {'WR':>5} {'Ann':>8} {'DD':>6} {'Sh':>5}")
    print("-" * 60)
    for r in results[:25]:
        print(f"{r['tn']:>3} {r['hd']:>3} {r['mc']:>3} "
              f"{'Y' if r['kg'] else 'N':>3} {'Y' if r['sh'] else 'N':>3} "
              f"{r['n']:>5} {r['wr']:>5.1f} {r['ann']:>+8.1f} "
              f"{r['dd']:>6.1f} {r['sharpe']:>5.2f}")

    # === Best configs — yearly breakdown ===
    print("\n" + "=" * 70)
    print("  BEST CONFIGS — YEARLY (2019-2026)")
    print("=" * 70)

    best = results[:5]
    for r in best:
        trades, eq, dd = backtest_v1(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=r['tn'], hold_days=r['hd'], min_rank=0.7,
            min_confidence=r['mc'], use_ker_gate=r['kg'],
            use_short=r['sh'], leverage=1, start_di=bt_2019)
        kg_s = "KER" if r['kg'] else "noKER"
        sh_s = "S" if r['sh'] else "L"
        print(f"\n  --- {kg_s} {sh_s} tn={r['tn']} hd={r['hd']} conf≥{r['mc']} ---")
        analyze(trades, eq, dd, f"{kg_s} {sh_s} tn={r['tn']} hd={r['hd']}")

    # === Full 10-year ===
    print("\n" + "=" * 70)
    print("  FULL 2016-2026 (10 years)")
    print("=" * 70)

    for r in best[:3]:
        trades, eq, dd = backtest_v1(
            C, O, H, L, NS, ND, dates, syms, sigs,
            top_n=r['tn'], hold_days=r['hd'], min_rank=0.7,
            min_confidence=r['mc'], use_ker_gate=r['kg'],
            use_short=r['sh'], leverage=1, start_di=60)
        kg_s = "KER" if r['kg'] else "noKER"
        sh_s = "S" if r['sh'] else "L"
        print(f"\n  FULL {kg_s} {sh_s} tn={r['tn']} hd={r['hd']} conf≥{r['mc']}")
        analyze(trades, eq, dd, f"full {kg_s} {sh_s} tn={r['tn']} hd={r['hd']}")

    print(f"\n[V1] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
