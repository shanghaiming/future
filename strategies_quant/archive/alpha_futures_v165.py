"""
Alpha Futures V165 — Mean Reversion Overlay on Momentum
==============================================================================
V157 champion gives +267%/-26% WF with V121 + Union momentum signals.

V165 explores MEAN REVERSION overlay to:
1. Provide diversification (negative correlation with momentum)
2. Catch short-term reversals that momentum misses
3. Act as natural hedging during momentum drawdowns

MR Signals (all LONG — buy oversold, expect bounce):
A. RSI oversold bounce:   RSI(14)<30 AND next-day open > prev close
B. Bollinger Band touch:  Close < BB_lower(20,2) AND next-day open > close
C. Consecutive down days: 3+ down days AND ROC5 < -2%
D. Gap down fill:         OV_GAP < -1% AND ID_RET > 0 (intraday recovery)
E. VWAP reversion:        Price far below VWAP (approx via vol-weighted typical)

Test modes:
- MR-only:      Trade mean reversion alone
- MR-overlay:   Add MR to V121+Union portfolio
- MR-replace:   Replace OV/ID + FF signals with MR in Union
- MR-defense:   Only activate MR during momentum drawdowns (>5% DD)

Kitchen Sink sizing, regime 0.5-1.5, max_corr=0.5, no SL.
"""
import sys, os, time, warnings
import numpy as np
import talib
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


def annual_return(final, initial, n_days):
    if final <= 0 or initial <= 0 or n_days <= 0: return -100.0
    return (final / initial) ** (1.0 / (n_days / 252)) * 100 - 100


def main():
    print("=" * 130)
    print("  V165 — Mean Reversion Overlay on V121+Union Momentum")
    print("=" * 130)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    print(f"  {NS} commodities, {ND} days")

    # ===================== PRECOMPUTE =====================
    print("\n[Precompute]...", flush=True)
    t0 = time.time()

    # --- Core indicators ---
    RET = np.full((NS, ND), np.nan)
    ROC5 = np.full((NS, ND), np.nan)
    ROC10 = np.full((NS, ND), np.nan)
    ROC20 = np.full((NS, ND), np.nan)
    RSI14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100
        ROC5[si] = talib.ROC(c, timeperiod=5)
        ROC10[si] = talib.ROC(c, timeperiod=10)
        ROC20[si] = talib.ROC(c, timeperiod=20)
        RSI14[si] = talib.RSI(c, timeperiod=14)

    ATR14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        ATR14[si] = talib.ATR(H[si].astype(np.float64), L[si].astype(np.float64),
                               C[si].astype(np.float64), timeperiod=14)

    # ZSCORE of daily returns (20-day rolling)
    ZSCORE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = RET[si, di-20:di]
            v = rets[~np.isnan(rets)]
            if len(v) < 10: continue
            s = np.std(v, ddof=1)
            if s > 0 and not np.isnan(RET[si, di]):
                ZSCORE[si, di] = (RET[si, di] - np.mean(v)) / s

    # Overnight gap and intraday return
    OV_GAP = np.full((NS, ND), np.nan)
    ID_RET = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            o, c = O[si, di], C[si, di]
            if not np.isnan(o) and not np.isnan(c):
                if di > 0 and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                    OV_GAP[si, di] = (o - C[si, di-1]) / C[si, di-1] * 100
                if o > 0: ID_RET[si, di] = (c - o) / o * 100

    # --- Bollinger Bands (20, 2) ---
    BB_UPPER = np.full((NS, ND), np.nan)
    BB_LOWER = np.full((NS, ND), np.nan)
    BB_MID   = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        bb_up, bb_mid, bb_lo = talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        BB_UPPER[si] = bb_up
        BB_MID[si]   = bb_mid
        BB_LOWER[si] = bb_lo

    # --- Consecutive down days counter ---
    CONSEC_DOWN = np.zeros((NS, ND), dtype=int)
    for si in range(NS):
        count = 0
        for di in range(1, ND):
            r = RET[si, di]
            if not np.isnan(r) and r < 0:
                count += 1
            else:
                count = 0
            CONSEC_DOWN[si, di] = count

    # --- Approximate VWAP (volume-weighted typical price, rolling 20-day) ---
    # Typical price = (H + L + C) / 3
    # VWAP deviation: % distance of close below VWAP
    VWAP_DEV = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(20, ND):
            h_w = H[si, di-19:di+1]; l_w = L[si, di-19:di+1]
            c_w = C[si, di-19:di+1]; v_w = V[si, di-19:di+1]
            valid = ~(np.isnan(h_w) | np.isnan(l_w) | np.isnan(c_w) | np.isnan(v_w))
            if np.sum(valid) < 10: continue
            h_v = h_w[valid]; l_v = l_w[valid]; c_v = c_w[valid]; v_v = v_w[valid]
            total_vol = np.sum(v_v)
            if total_vol <= 0: continue
            typ = (h_v + l_v + c_v) / 3.0
            vwap = np.sum(typ * v_v) / total_vol
            if vwap > 0 and not np.isnan(c_v[-1]):
                VWAP_DEV[si, di] = (c_v[-1] - vwap) / vwap * 100  # negative = below VWAP

    print(f"  Core indicators done ({time.time()-t0:.1f}s)")

    # --- Regime indicators ---
    print("  Computing regime indicators...", flush=True)

    BREADTH = np.full(ND, np.nan)
    for di in range(5, ND):
        pos_count = 0; total = 0
        for si in range(NS):
            r = ROC5[si, di]
            if not np.isnan(r):
                total += 1
                if r > 0: pos_count += 1
        if total > 0:
            BREADTH[di] = pos_count / total

    MKT_RET = np.full(ND, np.nan)
    for di in range(ND):
        rets_day = RET[:, di]
        valid = rets_day[~np.isnan(rets_day)]
        if len(valid) > 10:
            MKT_RET[di] = np.mean(valid)

    MKT_VOL = np.full(ND, np.nan)
    for di in range(20, ND):
        window = MKT_RET[di-20:di]
        valid = window[~np.isnan(window)]
        if len(valid) >= 10:
            MKT_VOL[di] = np.std(valid, ddof=1)

    valid_vols = MKT_VOL[~np.isnan(MKT_VOL)]
    VOL_MEDIAN = np.median(valid_vols) if len(valid_vols) > 0 else 1.0
    print(f"  Market vol median: {VOL_MEDIAN:.4f}%")
    print(f"  All precompute done ({time.time()-t0:.1f}s)")

    # ===================== SIGNAL DEFINITIONS =====================

    # --- MOMENTUM SIGNALS (unchanged from V157) ---

    def sig_v121(di, edi):
        """V121: ROC(5)>1% + Z>1.5 + ROC improving"""
        c = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((roc * zs, s, ep, 'v121'))
        return c

    def sig_ov_id(di, edi):
        """Overnight gap + intraday continuation"""
        c = []
        for s in range(NS):
            ov = OV_GAP[s, di]; idr = ID_RET[s, di]; roc = ROC5[s, di]
            if any(np.isnan(x) for x in [ov, idr, roc]): continue
            if ov <= 0.3 or idr <= 0.3 or roc <= 1.0: continue
            zs = ZSCORE[s, di]
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            z_bonus = zs if not np.isnan(zs) and zs > 1.0 else 1.0
            c.append(((ov + idr) * roc * z_bonus * 2, s, ep, 'ov_id'))
        return c

    def sig_final_flag(di, edi):
        """Final flag: 20-day breakout in tight range"""
        c = []
        for s in range(NS):
            roc20 = ROC20[s, di]
            if np.isnan(roc20) or roc20 <= 5.0 or di < 6: continue
            h5 = H[s, di-4:di+1]; l5 = L[s, di-4:di+1]
            if any(np.isnan(x) for x in h5) or any(np.isnan(x) for x in l5): continue
            r5 = np.max(h5) - np.min(l5)
            atr = ATR14[s, di]
            if np.isnan(atr) or atr <= 0 or r5 > atr * 3.0: continue
            h4 = np.max(H[s, di-4:di])
            cp = C[s, di]
            if np.isnan(cp) or cp <= h4: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((roc20 * (cp - h4) / atr, s, ep, 'ff'))
        return c

    def sig_union(di, edi):
        """Union: V121 + OV_ID + FF combined scores"""
        all_sigs = {}
        for item in sig_v121(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc * 3
            all_sigs[s][2].append('v121')
        for item in sig_ov_id(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc * 2
            all_sigs[s][2].append('ov_id')
        for item in sig_final_flag(di, edi):
            sc, s, ep, st = item
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += sc
            all_sigs[s][2].append('ff')
        return [(sc, s, ep, '+'.join(sigs)) for s, (sc, ep, sigs) in all_sigs.items()]

    # --- MEAN REVERSION SIGNALS (all LONG) ---

    def sig_mr_rsi(di, edi):
        """MR-A: RSI(14) < 30 oversold + next-day bounce confirmation.
        Score = (30 - RSI) / 30, stronger signal when more oversold."""
        c = []
        for s in range(NS):
            rsi = RSI14[s, di]
            if np.isnan(rsi) or rsi >= 30: continue
            # Bounce confirmation: next-day open > previous close
            if edi >= ND: continue
            next_open = O[s, edi]
            prev_close = C[s, di]
            if np.isnan(next_open) or np.isnan(prev_close) or prev_close <= 0: continue
            if next_open <= prev_close: continue  # no bounce
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = (30 - rsi) / 30.0  # higher score = more oversold
            c.append((score, s, ep, 'mr_rsi'))
        return c

    def sig_mr_bollinger(di, edi):
        """MR-B: Close < BB_lower(20,2) AND next-day open > close (recovery).
        Score = (BB_mid - close) / ATR, normalized deviation below bands."""
        c = []
        for s in range(NS):
            cp = C[s, di]; bb_lo = BB_LOWER[s, di]; bb_mid = BB_MID[s, di]
            if np.isnan(cp) or np.isnan(bb_lo) or np.isnan(bb_mid): continue
            if cp >= bb_lo: continue  # not below lower band
            # Recovery confirmation: next-day open > close
            if edi >= ND: continue
            next_open = O[s, edi]
            if np.isnan(next_open) or next_open <= cp: continue
            atr = ATR14[s, di]
            if np.isnan(atr) or atr <= 0: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = (bb_mid - cp) / atr  # how far below bands in ATR units
            c.append((score, s, ep, 'mr_bb'))
        return c

    def sig_mr_consecutive_down(di, edi):
        """MR-C: 3+ consecutive down days AND ROC5 < -2% (oversold).
        Score = consecutive_days * |ROC5|, stronger with more down days."""
        c = []
        for s in range(NS):
            cd = CONSEC_DOWN[s, di]
            roc = ROC5[s, di]
            if cd < 3: continue
            if np.isnan(roc) or roc >= -2.0: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = cd * abs(roc)
            c.append((score, s, ep, 'mr_cd'))
        return c

    def sig_mr_gap_fill(di, edi):
        """MR-D: OV_GAP < -1% (gap down) AND ID_RET > 0 (intraday recovery).
        Gap is being filled intraday = bullish reversal.
        Score = |OV_GAP| * ID_RET."""
        c = []
        for s in range(NS):
            ov = OV_GAP[s, di]; idr = ID_RET[s, di]
            if np.isnan(ov) or np.isnan(idr): continue
            if ov >= -1.0 or idr <= 0: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = abs(ov) * idr
            c.append((score, s, ep, 'mr_gap'))
        return c

    def sig_mr_vwap(di, edi):
        """MR-E: Price far below VWAP (VWAP_DEV < -1.5%), mean reversion expected.
        Score = |VWAP_DEV|, stronger when further below."""
        c = []
        for s in range(NS):
            dev = VWAP_DEV[s, di]
            if np.isnan(dev) or dev >= -1.5: continue
            # Additional confirmation: RSI < 45 (not already rallying)
            rsi = RSI14[s, di]
            if not np.isnan(rsi) and rsi > 45: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = abs(dev)
            c.append((score, s, ep, 'mr_vwap'))
        return c

    def sig_mr_combined(di, edi):
        """Combined MR: Aggregate all MR signals with voting.
        A symbol qualifies if it triggers 2+ MR sub-signals.
        Score = sum of sub-scores * vote_count."""
        all_mr = {}
        for sig_func in [sig_mr_rsi, sig_mr_bollinger, sig_mr_consecutive_down,
                         sig_mr_gap_fill, sig_mr_vwap]:
            for sc, s, ep, tag in sig_func(di, edi):
                if s not in all_mr:
                    all_mr[s] = [0.0, ep, 0]
                all_mr[s][0] += sc
                all_mr[s][2] += 1
        # Filter: require 2+ sub-signals for combined
        results = []
        for s, (total_sc, ep, votes) in all_mr.items():
            if votes >= 2:
                results.append((total_sc * votes, s, ep, f'mr_x{votes}'))
        return results

    # MR signal dispatcher
    MR_FUNCS = {
        'rsi': sig_mr_rsi,
        'bollinger': sig_mr_bollinger,
        'consecutive_down': sig_mr_consecutive_down,
        'gap_fill': sig_mr_gap_fill,
        'vwap': sig_mr_vwap,
        'combined': sig_mr_combined,
    }

    # ===================== HELPER: Correlation =====================
    def get_corr(si_a, si_b, di, window=20):
        start_idx = max(0, di - window)
        ret_a = RET[si_a, start_idx:di]
        ret_b = RET[si_b, start_idx:di]
        valid = ~(np.isnan(ret_a) | np.isnan(ret_b))
        n_valid = np.sum(valid)
        if n_valid < 8:
            return 0.5
        ra = ret_a[valid]; rb = ret_b[valid]
        if np.std(ra) == 0 or np.std(rb) == 0:
            return 0.5
        c = np.corrcoef(ra, rb)[0, 1]
        return c if not np.isnan(c) else 0.5

    # ===================== HELPER: Compute composite regime score =====================
    def compute_composite(di, daily_eq, high_water, perf_window=20):
        scores = []
        bth = BREADTH[di]
        if not np.isnan(bth):
            scores.append(np.clip((bth - 0.4) / (0.7 - 0.4), 0, 1))
        vol = MKT_VOL[di]
        if not np.isnan(vol) and VOL_MEDIAN > 0:
            vol_ratio = vol / VOL_MEDIAN
            scores.append(np.clip((1.5 - vol_ratio) / (1.5 - 0.8), 0, 1))
        if len(daily_eq) >= perf_window:
            eq_window = np.array(daily_eq[-perf_window:])
            x = np.arange(perf_window)
            try:
                slope = np.polyfit(x, eq_window, 1)[0]
                eq_mean = np.mean(eq_window)
                norm_slope = slope / eq_mean * 100 if eq_mean > 0 else 0
                eq_rets = np.diff(eq_window) / eq_window[:-1] * 100
                eq_rets = eq_rets[np.isfinite(eq_rets)]
                eq_std = np.std(eq_rets) if len(eq_rets) > 5 else 1.0
                z = norm_slope / eq_std if eq_std > 0 else 0
                scores.append(np.clip((z + 1.0) / 2.0, 0, 1))
            except Exception:
                pass
        if high_water > 0:
            cur_dd = (daily_eq[-1] - high_water) / high_water
        else:
            cur_dd = 0
        scores.append(np.clip(1.0 + cur_dd / 0.3, 0, 1))
        return np.mean(scores) if scores else 0.5

    # ===================== HELPER: DD-based sizing =====================
    def dd_size(pv, high_water, tiers):
        if high_water <= 0:
            return tiers[0][1]
        dd = (pv - high_water) / high_water
        for dd_thresh, size_frac in tiers:
            if dd >= -dd_thresh:
                return size_frac
        return tiers[-1][1]

    # ===================== HELPER: WR-adaptive sizing =====================
    def wr_size(trades, window=20):
        if len(trades) < window:
            return 1.0
        recent = trades[-window:]
        wr = np.mean([1 if t > 0 else 0 for t in recent])
        if wr > 0.65:
            return 1.3
        elif wr >= 0.50:
            return 1.0
        else:
            return 0.5

    # ===================== UNIFIED BACKTEST ENGINE =====================
    def backtest(start_di=MIN_TRAIN, end_di=None,
                 mr_method='combined',    # 'rsi','bollinger','consecutive_down','gap_fill','vwap','combined'
                 mr_weight=1.0,           # relative weight of MR vs momentum signals
                 mr_mode='overlay',       # 'only','overlay','replace','defense'
                 max_corr=0.5,
                 dd_tiers=None,
                 regime_lo=0.5, regime_hi=1.5,
                 sl_pct=0.0,
                 hold=1, top_n=3):
        if end_di is None: end_di = ND
        if dd_tiers is None:
            dd_tiers = [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)]

        cash = float(CASH0)
        positions = []
        trades = []
        daily_eq = []
        high_water = float(CASH0)

        for di in range(start_di, end_di - 1):
            # Mark-to-market
            pv = cash
            for p in positions:
                cp = C[p['si'], di]
                if not np.isnan(cp) and cp > 0:
                    m = MULT.get(p['sym'], DEF_MULT)
                    pv += cp * m * p['lots'] - cp * m * abs(p['lots']) * COMM
            daily_eq.append(pv)
            if pv > high_water:
                high_water = pv

            # --- Stop-loss check ---
            if sl_pct > 0:
                cl_early = []
                for p in positions:
                    cp = C[p['si'], di]
                    if np.isnan(cp) or cp <= 0: continue
                    m = MULT.get(p['sym'], DEF_MULT)
                    unrealized = (cp - p['entry_price']) * m * p['lots']
                    invested = p['entry_price'] * m * abs(p['lots'])
                    if invested > 0:
                        loss_pct = unrealized / invested
                        if loss_pct < -sl_pct:
                            cash += cp * m * abs(p['lots']) * (1 - COMM)
                            pnl_pct = unrealized / invested * 100
                            trades.append(pnl_pct)
                            cl_early.append(p)
                for p in cl_early: positions.remove(p)

            # Close positions past hold period
            cl = []
            for p in positions:
                if di - p['entry_di'] >= p['hold_days']:
                    ep = C[p['si'], di]
                    if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                    m = MULT.get(p['sym'], DEF_MULT)
                    pnl = (ep - p['entry_price']) * m * p['lots']
                    inv = p['entry_price'] * m * abs(p['lots'])
                    pp = pnl / inv * 100 if inv > 0 else 0
                    cash += ep * m * abs(p['lots']) * (1 - COMM)
                    trades.append(pp)
                    cl.append(p)
            for p in cl: positions.remove(p)

            # --- Kitchen Sink sizing ---
            dd_sz = dd_size(pv, high_water, dd_tiers)
            wr_mult_val = wr_size(trades, window=20)
            composite = compute_composite(di, daily_eq, high_water)
            regime_mult = regime_lo + composite * (regime_hi - regime_lo)
            pos_size = dd_sz * wr_mult_val * regime_mult
            pos_size = max(0.05, min(0.99, pos_size))

            # --- Enter positions ---
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue

            held_si = set(p['si'] for p in positions)

            # Get momentum signals
            cands_v121 = sig_v121(di, edi)
            cands_union = sig_union(di, edi)
            cands_v121.sort(key=lambda x: -x[0])
            cands_union.sort(key=lambda x: -x[0])

            # Get MR signals
            mr_func = MR_FUNCS.get(mr_method, sig_mr_combined)
            cands_mr = mr_func(di, edi)
            cands_mr.sort(key=lambda x: -x[0])

            # Check drawdown for defense mode
            in_drawdown = False
            if high_water > 0:
                cur_dd_pct = (pv - high_water) / high_water
                if cur_dd_pct < -0.05:  # >5% DD
                    in_drawdown = True

            entries = []

            if mr_mode == 'only':
                # MR-only: only trade mean reversion signals
                for sc, s, ep, tag in cands_mr:
                    if s not in held_si:
                        entries.append((sc, s, ep, tag, pos_size * mr_weight))
                        if len(entries) >= top_n: break

            elif mr_mode == 'overlay':
                # Overlay: trade both momentum and MR
                # Momentum entries
                best_v121 = None
                for c in cands_v121:
                    if c[1] not in held_si:
                        best_v121 = c; break
                best_union = None
                for c in cands_union:
                    if c[1] not in held_si:
                        best_union = c; break

                if best_v121 and best_union:
                    if best_v121[1] == best_union[1]:
                        entries.append((best_v121[0], best_v121[1], best_v121[2],
                                        'v121+union', pos_size * 1.5))
                    else:
                        corr = get_corr(best_v121[1], best_union[1], di)
                        if corr < max_corr:
                            entries.append((best_v121[0], best_v121[1], best_v121[2],
                                            'v121', pos_size))
                            entries.append((best_union[0], best_union[1], best_union[2],
                                            'union', pos_size))
                        else:
                            best = best_v121 if best_v121[0] >= best_union[0] else best_union
                            entries.append((best[0], best[1], best[2], 'best', pos_size))
                elif best_v121:
                    entries.append((best_v121[0], best_v121[1], best_v121[2], 'v121', pos_size))
                elif best_union:
                    entries.append((best_union[0], best_union[1], best_union[2], 'union', pos_size))

                # Add MR entries (separate from momentum, with weight)
                used_si = set(e[1] for e in entries)
                for sc, s, ep, tag in cands_mr:
                    if s not in held_si and s not in used_si:
                        entries.append((sc, s, ep, tag, pos_size * mr_weight))
                        if len(entries) >= top_n: break

            elif mr_mode == 'replace':
                # Replace: V121 stays, but replace OV/ID + FF in Union with MR
                all_sigs = {}
                # Keep V121
                for item in sig_v121(di, edi):
                    sc, s, ep, st = item
                    if s not in all_sigs: all_sigs[s] = [0, ep, []]
                    all_sigs[s][0] += sc * 3
                    all_sigs[s][2].append('v121')
                # Replace OV/ID and FF with MR
                for sc, s, ep, tag in cands_mr:
                    if s not in all_sigs: all_sigs[s] = [0, ep, []]
                    all_sigs[s][0] += sc * mr_weight * 2  # same weight as OV/ID
                    all_sigs[s][2].append(tag)
                union_mr = [(sc, s, ep, '+'.join(sigs))
                            for s, (sc, ep, sigs) in all_sigs.items()]
                union_mr.sort(key=lambda x: -x[0])

                # Best V121 and best Union-MR
                best_v121 = None
                for c in cands_v121:
                    if c[1] not in held_si:
                        best_v121 = c; break
                best_union_mr = None
                for c in union_mr:
                    if c[1] not in held_si:
                        best_union_mr = c; break

                if best_v121 and best_union_mr:
                    if best_v121[1] == best_union_mr[1]:
                        entries.append((best_v121[0], best_v121[1], best_v121[2],
                                        'v121+union_mr', pos_size * 1.5))
                    else:
                        corr = get_corr(best_v121[1], best_union_mr[1], di)
                        if corr < max_corr:
                            entries.append((best_v121[0], best_v121[1], best_v121[2],
                                            'v121', pos_size))
                            entries.append((best_union_mr[0], best_union_mr[1], best_union_mr[2],
                                            'union_mr', pos_size))
                        else:
                            best = best_v121 if best_v121[0] >= best_union_mr[0] else best_union_mr
                            entries.append((best[0], best[1], best[2], 'best', pos_size))
                elif best_v121:
                    entries.append((best_v121[0], best_v121[1], best_v121[2], 'v121', pos_size))
                elif best_union_mr:
                    entries.append((best_union_mr[0], best_union_mr[1], best_union_mr[2],
                                    'union_mr', pos_size))

            elif mr_mode == 'defense':
                # Defense: normal momentum, activate MR only during >5% drawdowns
                if in_drawdown:
                    # During drawdowns: add MR signals alongside momentum
                    best_v121 = None
                    for c in cands_v121:
                        if c[1] not in held_si:
                            best_v121 = c; break
                    best_union = None
                    for c in cands_union:
                        if c[1] not in held_si:
                            best_union = c; break

                    if best_v121:
                        entries.append((best_v121[0], best_v121[1], best_v121[2],
                                        'v121', pos_size))
                    if best_union:
                        entries.append((best_union[0], best_union[1], best_union[2],
                                        'union', pos_size))

                    used_si = set(e[1] for e in entries)
                    for sc, s, ep, tag in cands_mr:
                        if s not in held_si and s not in used_si:
                            entries.append((sc, s, ep, tag, pos_size * mr_weight))
                            if len(entries) >= top_n: break
                else:
                    # Normal: pure momentum
                    best_v121 = None
                    for c in cands_v121:
                        if c[1] not in held_si:
                            best_v121 = c; break
                    best_union = None
                    for c in cands_union:
                        if c[1] not in held_si:
                            best_union = c; break

                    if best_v121 and best_union:
                        if best_v121[1] == best_union[1]:
                            entries.append((best_v121[0], best_v121[1], best_v121[2],
                                            'v121+union', pos_size * 1.5))
                        else:
                            corr = get_corr(best_v121[1], best_union[1], di)
                            if corr < max_corr:
                                entries.append((best_v121[0], best_v121[1], best_v121[2],
                                                'v121', pos_size))
                                entries.append((best_union[0], best_union[1], best_union[2],
                                                'union', pos_size))
                            else:
                                best = best_v121 if best_v121[0] >= best_union[0] else best_union
                                entries.append((best[0], best[1], best[2], 'best', pos_size))
                    elif best_v121:
                        entries.append((best_v121[0], best_v121[1], best_v121[2], 'v121', pos_size))
                    elif best_union:
                        entries.append((best_union[0], best_union[1], best_union[2], 'union', pos_size))

            # Execute entries
            cash_snapshot = cash
            n_planned = len(entries)
            for sc, s, pr, sig_str, pct in entries:
                if s in set(p['si'] for p in positions): continue
                if len(positions) >= top_n: break
                cap = cash_snapshot * pct / max(n_planned, 1)
                sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                ct = max(1, int(cap / (pr * m * (1 + COMM))))
                ci = pr * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                    ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                if ct <= 0 or ci <= 0 or ci > cash: continue
                cash -= ci
                positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                  'lots': ct, 'dir': 1, 'sym': sym, 'hold_days': hold,
                                  'sig': sig_str, 'score': sc})

        # Close remaining
        for p in positions:
            ae = end_di - 1
            ep = C[p['si'], min(ae, ND-1)]
            if np.isnan(ep) or ep <= 0: ep = p['entry_price']
            m = MULT.get(p['sym'], DEF_MULT)
            cash += ep * m * abs(p['lots']) * (1 - COMM)

        nd = end_di - start_di
        ann = annual_return(cash, CASH0, nd)
        wr = np.mean([1 if t > 0 else 0 for t in trades]) * 100 if trades else 0
        nt = len(trades)
        if daily_eq:
            eq = np.array(daily_eq); pk = np.maximum.accumulate(eq)
            mdd = np.min((eq - pk) / pk * 100)
            r = np.diff(eq) / eq[:-1]
            r = np.where(np.isfinite(r), r, 0)
            sh = np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0
        else:
            mdd = 0; sh = 0
        return {'ann': ann, 'wr': wr, 'n': nt, 'mdd': mdd, 'sharpe': sh, 'final': cash}

    # ===================== PRINTING HELPERS =====================
    def pr(r, label=""):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  {label:95s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:.2f} | N={r['n']:4d}")

    def walk_forward(mr_method='combined', mr_weight=1.0, mr_mode='overlay',
                     max_corr=0.5, dd_tiers=None, regime_lo=0.5, regime_hi=1.5,
                     sl_pct=0.0, hold=1, top_n=3, label=""):
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue
            r = backtest(start_di=ys, end_di=ye, mr_method=mr_method,
                         mr_weight=mr_weight, mr_mode=mr_mode,
                         max_corr=max_corr, dd_tiers=dd_tiers,
                         regime_lo=regime_lo, regime_hi=regime_hi,
                         sl_pct=sl_pct, hold=hold, top_n=top_n)
            res[yr] = r
        return res

    def print_wf(wf_res, label=""):
        pos = sum(1 for r in wf_res.values() if r['ann'] > 0)
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"    {label}")
        print(f"      {pos}/6 pos | Avg={avg_ann:>+7.0f}% | WorstWfMDD={worst_mdd:>5.0f}%")
        print(f"      {ws}")

    # ===================== CONFIG DEFINITIONS =====================
    DD_TIERS = {
        'aggro100': [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)],
        'aggro90':  [(0, 0.90), (0.10, 0.80), (0.20, 0.60), (0.30, 0.40)],
        'champ':    [(0, 0.70), (0.10, 0.60), (0.20, 0.40), (0.30, 0.20)],
    }
    MR_METHODS = ['rsi', 'bollinger', 'consecutive_down', 'gap_fill', 'vwap', 'combined']
    MR_WEIGHTS = [0.5, 1.0, 2.0]
    MR_MODES = ['only', 'overlay', 'replace', 'defense']
    TOP_NS = [3, 4, 5]

    # ===================== SECTION 0: MOMENTUM BASELINE =====================
    print("\n" + "=" * 130)
    print("  SECTION 0: MOMENTUM BASELINE (V121 + Union, no MR)")
    print("=" * 130)

    # Pure momentum baseline (mr_mode='overlay' with no MR matches = pure momentum)
    baseline_results = []
    for dd_name, dd_t in DD_TIERS.items():
        for tn in TOP_NS:
            label = f"BASELINE mom-only DD{dd_name:7s} top{tn} reg0.5-1.5 noSL"
            r = backtest(mr_method='combined', mr_weight=0.0, mr_mode='overlay',
                         max_corr=0.5, dd_tiers=dd_t, regime_lo=0.5, regime_hi=1.5,
                         sl_pct=0.0, hold=1, top_n=tn)
            pr(r, label)
            baseline_results.append((r['ann'], r['mdd'], r['sharpe'], r['n'],
                                     label, {'mr_method': 'combined', 'mr_weight': 0.0,
                                             'mr_mode': 'overlay',
                                             'dd': dd_name, 'top_n': tn}))

    # ===================== SECTION 1: MR-ONLY (standalone mean reversion) =====================
    print("\n" + "=" * 130)
    print("  SECTION 1: MR-ONLY — Mean reversion signals standalone")
    print("  Test each MR method alone, no momentum")
    print("=" * 130)

    mr_only_results = []

    for mr_m in MR_METHODS:
        for dd_name, dd_t in DD_TIERS.items():
            for tn in TOP_NS:
                label = f"MR-only {mr_m:16s} DD{dd_name:7s} top{tn}"
                r = backtest(mr_method=mr_m, mr_weight=1.0, mr_mode='only',
                             max_corr=0.5, dd_tiers=dd_t, regime_lo=0.5, regime_hi=1.5,
                             sl_pct=0.0, hold=1, top_n=tn)
                pr(r, label)
                mr_only_results.append((r['ann'], r['mdd'], r['sharpe'], r['n'],
                                        label, {'mr_method': mr_m, 'dd': dd_name,
                                                'top_n': tn}))

    # ===================== SECTION 2: MR OVERLAY =====================
    print("\n" + "=" * 130)
    print("  SECTION 2: MR OVERLAY — Momentum + Mean reversion together")
    print("  Add MR signals on top of V121 + Union")
    print("=" * 130)

    overlay_results = []

    for mr_m in MR_METHODS:
        for mr_w in MR_WEIGHTS:
            for dd_name, dd_t in DD_TIERS.items():
                for tn in TOP_NS:
                    label = f"overlay {mr_m:16s} w={mr_w:.1f} DD{dd_name:7s} top{tn}"
                    r = backtest(mr_method=mr_m, mr_weight=mr_w, mr_mode='overlay',
                                 max_corr=0.5, dd_tiers=dd_t, regime_lo=0.5, regime_hi=1.5,
                                 sl_pct=0.0, hold=1, top_n=tn)
                    pr(r, label)
                    overlay_results.append((r['ann'], r['mdd'], r['sharpe'], r['n'],
                                           label, {'mr_method': mr_m, 'mr_weight': mr_w,
                                                   'dd': dd_name, 'top_n': tn}))

    # ===================== SECTION 3: MR REPLACE =====================
    print("\n" + "=" * 130)
    print("  SECTION 3: MR REPLACE — Replace OV/ID + FF with MR in Union")
    print("=" * 130)

    replace_results = []

    for mr_m in MR_METHODS:
        for mr_w in MR_WEIGHTS:
            for dd_name, dd_t in DD_TIERS.items():
                for tn in TOP_NS:
                    label = f"replace {mr_m:16s} w={mr_w:.1f} DD{dd_name:7s} top{tn}"
                    r = backtest(mr_method=mr_m, mr_weight=mr_w, mr_mode='replace',
                                 max_corr=0.5, dd_tiers=dd_t, regime_lo=0.5, regime_hi=1.5,
                                 sl_pct=0.0, hold=1, top_n=tn)
                    pr(r, label)
                    replace_results.append((r['ann'], r['mdd'], r['sharpe'], r['n'],
                                            label, {'mr_method': mr_m, 'mr_weight': mr_w,
                                                    'dd': dd_name, 'top_n': tn}))

    # ===================== SECTION 4: MR DEFENSE =====================
    print("\n" + "=" * 130)
    print("  SECTION 4: MR DEFENSE — Activate MR only during momentum drawdowns (>5% DD)")
    print("=" * 130)

    defense_results = []

    for mr_m in MR_METHODS:
        for mr_w in MR_WEIGHTS:
            for dd_name, dd_t in DD_TIERS.items():
                for tn in TOP_NS:
                    label = f"defense {mr_m:16s} w={mr_w:.1f} DD{dd_name:7s} top{tn}"
                    r = backtest(mr_method=mr_m, mr_weight=mr_w, mr_mode='defense',
                                 max_corr=0.5, dd_tiers=dd_t, regime_lo=0.5, regime_hi=1.5,
                                 sl_pct=0.0, hold=1, top_n=tn)
                    pr(r, label)
                    defense_results.append((r['ann'], r['mdd'], r['sharpe'], r['n'],
                                            label, {'mr_method': mr_m, 'mr_weight': mr_w,
                                                    'dd': dd_name, 'top_n': tn}))

    # ===================== SECTION 5: TOP 20 FULL-PERIOD ACROSS ALL MODES =====================
    print("\n" + "=" * 130)
    print("  SECTION 5: TOP 20 FULL-PERIOD RESULTS (all modes)")
    print("=" * 130)

    all_results = ([(ann, mdd, sh, n, lbl, cfg) for ann, mdd, sh, n, lbl, cfg in baseline_results] +
                   [(ann, mdd, sh, n, lbl, cfg) for ann, mdd, sh, n, lbl, cfg in mr_only_results] +
                   [(ann, mdd, sh, n, lbl, cfg) for ann, mdd, sh, n, lbl, cfg in overlay_results] +
                   [(ann, mdd, sh, n, lbl, cfg) for ann, mdd, sh, n, lbl, cfg in replace_results] +
                   [(ann, mdd, sh, n, lbl, cfg) for ann, mdd, sh, n, lbl, cfg in defense_results])

    all_results.sort(key=lambda x: -x[0])
    for i, (ann, mdd, sh, n, label, cfg) in enumerate(all_results[:20]):
        ratio = abs(ann / mdd) if mdd != 0 else 0
        print(f"  #{i+1:2d} | Ann={ann:+8.1f}% | MDD={mdd:6.1f}% | Sh={sh:4.2f} | R/M={ratio:.2f} | N={n:4d} | {label}")

    # ===================== SECTION 6: BEST PER MR METHOD (full-period) =====================
    print("\n" + "=" * 130)
    print("  SECTION 6: BEST PER MR METHOD (full-period, all modes)")
    print("=" * 130)

    for mr_m in MR_METHODS:
        mr_specific = [(ann, mdd, sh, n, lbl, cfg)
                       for ann, mdd, sh, n, lbl, cfg in all_results
                       if cfg.get('mr_method') == mr_m]
        if not mr_specific:
            print(f"\n  {mr_m}: no results")
            continue
        mr_specific.sort(key=lambda x: -x[0])
        best = mr_specific[0]
        ann, mdd, sh, n, label, cfg = best
        ratio = abs(ann / mdd) if mdd != 0 else 0
        print(f"\n  {mr_m:16s} best: Ann={ann:+8.1f}% | MDD={mdd:6.1f}% | R/M={ratio:.2f} | {label}")

    # ===================== SECTION 7: BEST PER MR MODE (full-period) =====================
    print("\n" + "=" * 130)
    print("  SECTION 7: BEST PER MR MODE (full-period)")
    print("=" * 130)

    for mode in MR_MODES:
        mode_results = [(ann, mdd, sh, n, lbl, cfg)
                        for ann, mdd, sh, n, lbl, cfg in all_results
                        if mode in lbl]
        if not mode_results:
            print(f"\n  {mode}: no results")
            continue
        mode_results.sort(key=lambda x: -x[0])
        best = mode_results[0]
        ann, mdd, sh, n, label, cfg = best
        ratio = abs(ann / mdd) if mdd != 0 else 0
        print(f"\n  {mode:10s} best: Ann={ann:+8.1f}% | MDD={mdd:6.1f}% | R/M={ratio:.2f} | {label}")

    # ===================== SECTION 8: MOMENTUM vs OVERLAY COMPARISON =====================
    print("\n" + "=" * 130)
    print("  SECTION 8: MOMENTUM vs OVERLAY COMPARISON")
    print("  Show best momentum baseline vs best overlay (same DD tier + top_n)")
    print("=" * 130)

    for dd_name in DD_TIERS:
        for tn in TOP_NS:
            # Find baseline
            baseline_ann = None
            for ann, mdd, sh, n, lbl, cfg in baseline_results:
                if cfg.get('dd') == dd_name and cfg.get('top_n') == tn:
                    baseline_ann = ann
                    print(f"  BASELINE  DD{dd_name:7s} top{tn}: Ann={ann:+8.1f}%")
                    break
            # Find best overlay for same config
            best_overlay = None
            for ann, mdd, sh, n, lbl, cfg in overlay_results:
                if cfg.get('dd') == dd_name and cfg.get('top_n') == tn:
                    if best_overlay is None or ann > best_overlay[0]:
                        best_overlay = (ann, mdd, sh, n, lbl, cfg)
            if best_overlay and baseline_ann is not None:
                ann, mdd, sh, n, lbl, cfg = best_overlay
                delta = ann - baseline_ann
                print(f"  BEST OVLY DD{dd_name:7s} top{tn}: Ann={ann:+8.1f}% (delta={delta:+.0f}%) MDD={mdd:6.1f}% | {lbl}")

    # ===================== SECTION 9: WALK-FORWARD VALIDATION =====================
    print("\n" + "=" * 130)
    print("  SECTION 9: WALK-FORWARD VALIDATION — TOP 30 by full-period ann")
    print("=" * 130)

    wf_all = {}
    # Exclude baselines from WF grid (they are run separately below)
    non_baseline = [(ann, mdd, sh, n, lbl, cfg)
                    for ann, mdd, sh, n, lbl, cfg in all_results
                    if 'BASELINE' not in lbl]
    for i, (ann, mdd, sh, n, label, cfg) in enumerate(non_baseline[:30]):
        dd_t = DD_TIERS.get(cfg.get('dd', 'aggro100'), DD_TIERS['aggro100'])
        wf_res = walk_forward(mr_method=cfg.get('mr_method', 'combined'),
                              mr_weight=cfg.get('mr_weight', 1.0),
                              mr_mode=cfg.get('mr_mode', 'overlay'),
                              max_corr=0.5, dd_tiers=dd_t,
                              regime_lo=0.5, regime_hi=1.5,
                              sl_pct=0.0, hold=1, top_n=cfg.get('top_n', 3),
                              label=label)
        wf_all[label] = (wf_res, cfg)
        print_wf(wf_res, label)

    # Also WF for baseline (momentum-only with aggro100, top 3-5)
    print("\n  --- Walk-forward baselines (momentum-only) ---")
    for dd_name, dd_t in DD_TIERS.items():
        for tn in TOP_NS:
            label = f"BASELINE WF DD{dd_name} top{tn}"
            # Run momentum-only via overlay with 0 weight
            wf_res = walk_forward(mr_method='combined', mr_weight=0.0, mr_mode='overlay',
                                  max_corr=0.5, dd_tiers=dd_t, regime_lo=0.5, regime_hi=1.5,
                                  sl_pct=0.0, hold=1, top_n=tn, label=label)
            wf_all[label] = (wf_res, {'mr_method': 'combined', 'mr_weight': 0.0,
                                       'mr_mode': 'overlay', 'dd': dd_name, 'top_n': tn})
            print_wf(wf_res, label)

    # ===================== SECTION 10: TOP 10 BY WF AVERAGE =====================
    print("\n" + "=" * 130)
    print("  SECTION 10: TOP 10 BY WF AVERAGE")
    print("=" * 130)

    wf_ranked = []
    for label, (wf_res, cfg) in wf_all.items():
        avg_ann = np.mean([r['ann'] for r in wf_res.values()])
        worst_mdd = min(r['mdd'] for r in wf_res.values())
        best_ann = max(r['ann'] for r in wf_res.values())
        pos = sum(1 for r in wf_res.values() if r['ann'] > 0)
        total_n = sum(r['n'] for r in wf_res.values())
        avg_wr = np.mean([r['wr'] for r in wf_res.values()])
        wf_ranked.append((avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res))

    wf_ranked.sort(key=lambda x: -x[0])

    for i, (avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res) in enumerate(wf_ranked[:10]):
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  #{i+1} WF AVG={avg_ann:+.0f}% | WorstMDD={worst_mdd:.0f}% | {pos}/6 pos | TotalTrades={total_n} | AvgWR={avg_wr:.1f}%")
        print(f"     {label}")
        print(f"     {ws}")

    # ===================== SECTION 11: BEST RISK-ADJUSTED (WF) =====================
    print("\n" + "=" * 130)
    print("  SECTION 11: BEST RISK-ADJUSTED (WF avg / |worst MDD|)")
    print("=" * 130)

    wf_ra = sorted(wf_ranked, key=lambda x: -abs(x[0] / x[1]) if x[1] != 0 else 0)
    for i, (avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res) in enumerate(wf_ra[:10]):
        ratio = abs(avg_ann / worst_mdd) if worst_mdd != 0 else 0
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  #{i+1} WF AVG={avg_ann:+.0f}% | WorstMDD={worst_mdd:.0f}% | R/M={ratio:.2f} | {pos}/6 pos")
        print(f"     {label}")
        print(f"     {ws}")

    # ===================== SECTION 12: MR METHOD COMPARISON (WF) =====================
    print("\n" + "=" * 130)
    print("  SECTION 12: BEST PER MR METHOD (WF avg)")
    print("=" * 130)

    for mr_m in MR_METHODS + ['baseline']:
        mr_results = [(avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr)
                      for avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr in wf_ranked]
        if mr_m == 'baseline':
            mr_results = [x for x in mr_results if 'BASELINE' in x[6]]
        else:
            mr_results = [x for x in mr_results if x[7].get('mr_method') == mr_m and 'BASELINE' not in x[6]]
        if not mr_results:
            print(f"\n  {mr_m:16s}: no WF results")
            continue
        mr_results.sort(key=lambda x: -x[0])
        best = mr_results[0]
        avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res = best
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  {mr_m:16s} best: WF AVG={avg_ann:+.0f}% | WorstMDD={worst_mdd:.0f}% | {pos}/6 pos")
        print(f"     {label}")
        print(f"     {ws}")

    # ===================== SECTION 13: BEST PER MR MODE (WF) =====================
    print("\n" + "=" * 130)
    print("  SECTION 13: BEST PER MR MODE (WF avg)")
    print("=" * 130)

    for mode in MR_MODES:
        mode_results = [(avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr)
                        for avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr in wf_ranked
                        if mode in lbl]
        if not mode_results:
            print(f"\n  {mode:10s}: no WF results")
            continue
        mode_results.sort(key=lambda x: -x[0])
        best = mode_results[0]
        avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res = best
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"\n  {mode:10s} best: WF AVG={avg_ann:+.0f}% | WorstMDD={worst_mdd:.0f}% | {pos}/6 pos")
        print(f"     {label}")
        print(f"     {ws}")

    # ===================== SECTION 14: IMPROVEMENT OVER BASELINE =====================
    print("\n" + "=" * 130)
    print("  SECTION 14: IMPROVEMENT OVER MOMENTUM BASELINE")
    print("  Show configs that beat their matching baseline in WF avg")
    print("=" * 130)

    # Collect baselines
    baselines_wf = {}
    for avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr in wf_ranked:
        if 'BASELINE' in lbl:
            key = f"{cfg.get('dd', 'unknown')}_top{cfg.get('top_n', 3)}"
            baselines_wf[key] = avg

    improvements = []
    for avg, wmdd, bann, pos, tn, awr, lbl, cfg, wfr in wf_ranked:
        if 'BASELINE' in lbl: continue
        key = f"{cfg.get('dd', 'unknown')}_top{cfg.get('top_n', 3)}"
        base_avg = baselines_wf.get(key)
        if base_avg is not None:
            delta = avg - base_avg
            improvements.append((delta, avg, base_avg, wmdd, key, lbl, cfg, wfr))

    improvements.sort(key=lambda x: -x[0])
    print("\n  --- Top 15 improvements over baseline ---")
    for i, (delta, avg, base_avg, wmdd, key, lbl, cfg, wfr) in enumerate(improvements[:15]):
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wfr.items())])
        print(f"\n  #{i+1} delta={delta:+.0f}% | MR WF={avg:+.0f}% | Base WF={base_avg:+.0f}% | Key={key}")
        print(f"     {lbl}")
        print(f"     {ws}")

    print("\n  --- Top 15 worst vs baseline ---")
    worst_improvements = sorted(improvements, key=lambda x: x[0])
    for i, (delta, avg, base_avg, wmdd, key, lbl, cfg, wfr) in enumerate(worst_improvements[:15]):
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wfr.items())])
        print(f"\n  #{i+1} delta={delta:+.0f}% | MR WF={avg:+.0f}% | Base WF={base_avg:+.0f}% | Key={key}")
        print(f"     {lbl}")
        print(f"     {ws}")

    # ===================== SECTION 15: MR SIGNAL FREQUENCY ANALYSIS =====================
    print("\n" + "=" * 130)
    print("  SECTION 15: MR SIGNAL FREQUENCY (how often each MR triggers)")
    print("=" * 130)

    for mr_m in MR_METHODS:
        mr_func = MR_FUNCS[mr_m]
        total_days = 0
        total_signals = 0
        for di in range(MIN_TRAIN, ND - 1):
            edi = di + 1
            if edi >= ND: continue
            total_days += 1
            sigs = mr_func(di, edi)
            if sigs:
                total_signals += 1
        freq = total_signals / max(total_days, 1) * 100
        print(f"  {mr_m:16s}: triggers on {total_signals:5d}/{total_days} days ({freq:.1f}%)")

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 130)
    print("  FINAL SUMMARY")
    print("=" * 130)

    print(f"\n  Baseline (V121+Union momentum): best WF from baselines above")
    print(f"  V165 target: improve on momentum via mean reversion overlay")

    if wf_ranked:
        # Best overall
        best = wf_ranked[0]
        avg_ann, worst_mdd, best_ann, pos, total_n, avg_wr, label, cfg, wf_res = best
        print(f"\n  Best V165 overall: {label}")
        print(f"    WF AVG={avg_ann:+.0f}% | WorstMDD={worst_mdd:.0f}% | {pos}/6 pos")
        ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                         for yr, r in sorted(wf_res.items())])
        print(f"    {ws}")

        # Best overlay that improves on baseline
        if improvements and improvements[0][0] > 0:
            delta, avg, base_avg, wmdd, key, lbl, cfg, wfr = improvements[0]
            print(f"\n  Best improvement over baseline: delta={delta:+.0f}%")
            print(f"    {lbl}")
            ws = " | ".join([f"{yr}:{r['ann']:+.0f}%/{r['mdd']:.0f}%"
                             for yr, r in sorted(wfr.items())])
            print(f"    {ws}")

        # Summary by MR mode
        print(f"\n  --- Best per MR mode (WF avg) ---")
        for mode in MR_MODES:
            mode_results = [x for x in wf_ranked if mode in x[6]]
            if mode_results:
                mode_results.sort(key=lambda x: -x[0])
                best_mode = mode_results[0]
                print(f"    {mode:10s}: WF AVG={best_mode[0]:+.0f}% | WorstMDD={best_mode[1]:.0f}% | {best_mode[6]}")

        # Did MR help?
        positive_improvements = sum(1 for d, *_ in improvements if d > 0)
        negative_improvements = sum(1 for d, *_ in improvements if d <= 0)
        avg_delta = np.mean([d for d, *_ in improvements]) if improvements else 0
        print(f"\n  MR impact: {positive_improvements} configs improved, {negative_improvements} degraded")
        print(f"  Average delta over baseline: {avg_delta:+.0f}%")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
