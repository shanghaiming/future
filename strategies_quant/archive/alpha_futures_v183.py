"""
Alpha Futures V183 — Cross-Commodity Group Momentum (Supply Chain)
==============================================================================
Data: 4 black-chain futures — rbfi, hcfi, jfi, jmfi
  rbfi = rebar (rebar/螺纹钢)  — downstream
  hcfi = hot coil (热卷)       — downstream
  jfi  = coke (焦炭)           — upstream
  jmfi = coking coal (焦煤)    — upstream

Baseline to beat: R/M = 11.41 (from V178)

Strategy concept:
  When an entire supply chain group shows coordinated momentum, individual
  members are more likely to continue. We test four factors:

  Factor 1 — GROUP MOMENTUM ALIGNMENT:
    % of group members with positive 5d ROC. When >60% are up, the group
    is bullish. Require alignment before entering any individual position.

  Factor 2 — RELATIVE STRENGTH RANK:
    20-day return rank within the group. Top performer gets highest signal.
    Favor the strongest member in a coordinated move.

  Factor 3 — SUPPLY CHAIN LEAD-LAG:
    Upstream (jfi, jmfi) leads downstream (rbfi, hcfi). When upstream
    momentum is positive AND downstream hasn't caught up yet, buy
    downstream. Tests various lead-lag windows (1-5 days).

  Factor 4 — GROUP MEAN REVERSION (bonus):
    When the group is oversold (<30% positive ROC) and individual shows
    bullish reversal patterns, buy the most oversold member.

Signal: Long only, buy when group alignment is strong AND individual
        momentum is positive.
Walk-forward: Train 2019-2023, Test 2024-2026.
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
    print("  V183 — Cross-Commodity Group Momentum (Supply Chain)")
    print("  Black chain: rbfi(螺纹) hcfi(热卷) jfi(焦炭) jmfi(焦煤)")
    print("  Baseline R/M to beat: 11.41")
    print("=" * 130)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    print(f"  {NS} commodities, {ND} days")

    # ===================== SYMBOL GROUP SETUP =====================
    sym_to_si = {s: i for i, s in enumerate(syms)}

    # Supply chain classification
    UPSTREAM = {'jfi', 'jmfi'}    # coke, coking coal (原材料)
    DOWNSTREAM = {'rbfi', 'hcfi'}  # rebar, hot coil (成品)
    ALL_GROUP = set(syms)          # only black chain in this dataset

    upstream_si = [sym_to_si[s] for s in UPSTREAM if s in sym_to_si]
    downstream_si = [sym_to_si[s] for s in DOWNSTREAM if s in sym_to_si]
    all_si = list(range(NS))

    print(f"  Upstream: {[syms[si] for si in upstream_si]}")
    print(f"  Downstream: {[syms[si] for si in downstream_si]}")

    # ===================== PRECOMPUTE =====================
    print("\n[Precompute]...", flush=True)
    t0 = time.time()

    RET = np.full((NS, ND), np.nan)
    ROC5 = np.full((NS, ND), np.nan)
    ROC10 = np.full((NS, ND), np.nan)
    ROC20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100
        ROC5[si] = talib.ROC(c, timeperiod=5)
        ROC10[si] = talib.ROC(c, timeperiod=10)
        ROC20[si] = talib.ROC(c, timeperiod=20)

    ATR14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        ATR14[si] = talib.ATR(H[si].astype(np.float64), L[si].astype(np.float64),
                               C[si].astype(np.float64), timeperiod=14)

    ATR_NORM = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            atr = ATR14[si, di]
            cp = C[si, di]
            if not np.isnan(atr) and not np.isnan(cp) and cp > 0:
                ATR_NORM[si, di] = atr / cp * 100

    ZSCORE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = RET[si, di-20:di]
            v = rets[~np.isnan(rets)]
            if len(v) < 10: continue
            s = np.std(v, ddof=1)
            if s > 0 and not np.isnan(RET[si, di]):
                ZSCORE[si, di] = (RET[si, di] - np.mean(v)) / s

    # ===================== GROUP-LEVEL FACTORS =====================
    print("  Computing group-level factors...", flush=True)

    # Factor 1: Group Momentum Alignment — % of group with positive ROC5
    GROUP_ALIGN = np.full(ND, np.nan)
    for di in range(5, ND):
        pos_count = 0; total = 0
        for si in all_si:
            r = ROC5[si, di]
            if not np.isnan(r):
                total += 1
                if r > 0: pos_count += 1
        if total > 0:
            GROUP_ALIGN[di] = pos_count / total

    # Factor 1b: Group average ROC (momentum magnitude)
    GROUP_AVG_ROC5 = np.full(ND, np.nan)
    for di in range(5, ND):
        rocs = [ROC5[si, di] for si in all_si if not np.isnan(ROC5[si, di])]
        if len(rocs) > 0:
            GROUP_AVG_ROC5[di] = np.mean(rocs)

    # Factor 1c: Group average ROC20 (longer-term trend)
    GROUP_AVG_ROC20 = np.full(ND, np.nan)
    for di in range(20, ND):
        rocs = [ROC20[si, di] for si in all_si if not np.isnan(ROC20[si, di])]
        if len(rocs) > 0:
            GROUP_AVG_ROC20[di] = np.mean(rocs)

    # Factor 2: Relative Strength Rank within group (20d return rank)
    RS_RANK = np.full((NS, ND), np.nan)
    for di in range(20, ND):
        rets = {}
        for si in all_si:
            r = ROC20[si, di]
            if not np.isnan(r):
                rets[si] = r
        if len(rets) < 2: continue
        sorted_si = sorted(rets.keys(), key=lambda s: rets[s])
        n = len(sorted_si)
        for rank, si in enumerate(sorted_si):
            RS_RANK[si, di] = (rank + 1) / n  # 1/n (weakest) to 1.0 (strongest)

    # Factor 3: Supply Chain Lead-Lag
    # Upstream average ROC vs Downstream average ROC
    UPSTREAM_AVG_ROC5 = np.full(ND, np.nan)
    DOWNSTREAM_AVG_ROC5 = np.full(ND, np.nan)
    for di in range(5, ND):
        up_rocs = [ROC5[si, di] for si in upstream_si if not np.isnan(ROC5[si, di])]
        dn_rocs = [ROC5[si, di] for si in downstream_si if not np.isnan(ROC5[si, di])]
        if up_rocs: UPSTREAM_AVG_ROC5[di] = np.mean(up_rocs)
        if dn_rocs: DOWNSTREAM_AVG_ROC5[di] = np.mean(dn_rocs)

    # Lead-lag indicator: upstream positive + downstream lagging → buy downstream
    # Positive value means upstream leads (downstream hasn't caught up yet)
    LEAD_LAG = np.full(ND, np.nan)
    for di in range(5, ND):
        up = UPSTREAM_AVG_ROC5[di]
        dn = DOWNSTREAM_AVG_ROC5[di]
        if not np.isnan(up) and not np.isnan(dn):
            LEAD_LAG[di] = up - dn

    # Factor 3b: Lagged upstream momentum (upstream ROC from N days ago)
    # When upstream was positive N days ago, downstream should rally now
    UPSTREAM_LAG = {}
    for lag in [1, 2, 3, 5]:
        UPSTREAM_LAG[lag] = np.full(ND, np.nan)
        for di in range(5 + lag, ND):
            up_rocs = [ROC5[si, di-lag] for si in upstream_si if not np.isnan(ROC5[si, di-lag])]
            if up_rocs:
                UPSTREAM_LAG[lag][di] = np.mean(up_rocs)

    # Factor 4: Group oversold (mean reversion opportunity)
    GROUP_OVERSOLD = np.full(ND, np.nan)
    for di in range(5, ND):
        align = GROUP_ALIGN[di]
        if not np.isnan(align):
            GROUP_OVERSOLD[di] = 1.0 if align < 0.3 else 0.0

    # ===================== REGIME INDICATORS =====================
    print("  Computing regime indicators...", flush=True)

    MKT_RET = np.full(ND, np.nan)
    for di in range(ND):
        rets_day = RET[:, di]
        valid = rets_day[~np.isnan(rets_day)]
        if len(valid) > 1:
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
    print(f"  Precompute done ({time.time()-t0:.1f}s)")

    # ===================== SIGNAL DEFINITIONS =====================
    def sig_v121_baseline(di, edi):
        """Original V121 long signal: ROC(5)>1% + Z>1.5 + ROC improving"""
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

    def sig_group_aligned(di, edi, align_threshold=0.6):
        """Factor 1: Group alignment + individual momentum.
        Buy when group alignment > threshold AND individual ROC5 > 0 AND ROC improving."""
        c = []
        ga = GROUP_ALIGN[di]
        if np.isnan(ga) or ga < align_threshold: return c
        for s in range(NS):
            roc = ROC5[s, di]
            if np.isnan(roc) or roc <= 0.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue
            # Boost score by group alignment strength
            score = roc * ga
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((score, s, ep, f'grp_align_{align_threshold:.0%}'))
        return c

    def sig_group_aligned_strict(di, edi, align_threshold=0.75):
        """Factor 1 strict: Need 75%+ alignment + ROC5 > 1% + ROC improving."""
        c = []
        ga = GROUP_ALIGN[di]
        if np.isnan(ga) or ga < align_threshold: return c
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.0: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue
            score = roc * zs * ga
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((score, s, ep, f'grp_strict_{align_threshold:.0%}'))
        return c

    def sig_rs_rank(di, edi, rs_min=0.5, align_threshold=0.6):
        """Factor 2: Relative strength rank within group.
        Buy the strongest performer when group is aligned."""
        c = []
        ga = GROUP_ALIGN[di]
        if np.isnan(ga) or ga < align_threshold: return c
        for s in range(NS):
            rs = RS_RANK[s, di]
            roc = ROC5[s, di]
            if np.isnan(rs) or np.isnan(roc) or rs < rs_min or roc <= 0: continue
            # Score = RS rank * ROC * group alignment
            score = rs * roc * ga
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((score, s, ep, f'rs_{rs_min:.1f}_ga{align_threshold:.0%}'))
        return c

    def sig_lead_lag(di, edi, lag=2, upstream_min=1.0):
        """Factor 3: Supply chain lead-lag.
        When upstream was positive N days ago and downstream hasn't caught up,
        buy downstream. The idea: raw materials lead finished goods."""
        c = []
        # Check if upstream was bullish 'lag' days ago
        up_lag = UPSTREAM_LAG.get(lag, UPSTREAM_LAG[2])
        up_val = up_lag[di]
        if np.isnan(up_val) or up_val < upstream_min: return c
        # Only buy downstream that hasn't fully responded yet
        for s in downstream_si:
            roc = ROC5[s, di]
            if np.isnan(roc) or roc <= 0: continue
            # Score boosted by upstream lead strength
            score = roc * up_val
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((score, s, ep, f'leadlag_l{lag}_u{upstream_min:.0f}'))
        return c

    def sig_lead_lag_all(di, edi, lag=2, upstream_min=1.0):
        """Factor 3 variant: Lead-lag signal on ALL symbols when upstream leads.
        Upstream momentum predicts the whole group, not just downstream."""
        c = []
        up_lag = UPSTREAM_LAG.get(lag, UPSTREAM_LAG[2])
        up_val = up_lag[di]
        if np.isnan(up_val) or up_val < upstream_min: return c
        # Current group alignment must be positive
        ga = GROUP_ALIGN[di]
        if np.isnan(ga) or ga < 0.5: return c
        for s in range(NS):
            roc = ROC5[s, di]
            if np.isnan(roc) or roc <= 0: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue
            # Upstream symbols get extra boost, downstream gets lead-lag bonus
            if s in upstream_si:
                score = roc * up_val * 1.5
            else:
                score = roc * up_val
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((score, s, ep, f'leadlag_all_l{lag}_u{upstream_min:.0f}'))
        return c

    def sig_group_oversold(di, edi):
        """Factor 4: Group mean reversion.
        When group is oversold (<30% positive) and individual shows reversal,
        buy the most oversold member with positive ROC today."""
        c = []
        oversold = GROUP_OVERSOLD[di]
        if np.isnan(oversold) or oversold < 1: return c
        for s in range(NS):
            roc = ROC5[s, di]
            if np.isnan(roc) or roc <= 0.5: continue
            # Must have been declining (ROC5 was negative recently)
            roc_prev = ROC5[s, di-1] if di > 0 else np.nan
            if np.isnan(roc_prev) or roc_prev >= 0: continue
            # This is a genuine reversal from oversold
            score = roc * abs(roc_prev)  # stronger reversal = higher score
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((score, s, ep, 'grp_oversold'))
        return c

    def sig_combo_group(di, edi, align_threshold=0.6, rs_weight=0.3,
                        lag=2, upstream_min=1.0, use_leadlag=True):
        """COMBO: Group alignment + RS rank + lead-lag (optional).
        Multi-factor score combining all group factors."""
        c = []
        ga = GROUP_ALIGN[di]
        if np.isnan(ga) or ga < align_threshold: return c

        up_lag = UPSTREAM_LAG.get(lag, UPSTREAM_LAG[2])
        up_val = up_lag[di]
        lead_lag_active = (not np.isnan(up_val)) and up_val >= upstream_min

        for s in range(NS):
            roc = ROC5[s, di]
            if np.isnan(roc) or roc <= 0.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue

            # Base score: ROC * group alignment
            score = roc * ga

            # RS rank bonus
            rs = RS_RANK[s, di]
            if not np.isnan(rs):
                score += roc * rs * rs_weight

            # Lead-lag bonus: downstream gets bonus when upstream leads
            if use_leadlag and lead_lag_active and s in downstream_si:
                score *= 1.5

            # Upstream bonus: upstream is the leader, gets boosted when leading
            if use_leadlag and lead_lag_active and s in upstream_si:
                score *= 1.3

            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((score, s, ep, f'combo_a{align_threshold:.0%}_l{lag}'))
        return c

    def sig_combo_zscore(di, edi, align_threshold=0.6, z_min=1.0,
                         lag=2, upstream_min=1.0):
        """COMBO+Z: Group alignment + Z-score + lead-lag.
        Strongest filter: need both group alignment AND individual statistical significance."""
        c = []
        ga = GROUP_ALIGN[di]
        if np.isnan(ga) or ga < align_threshold: return c

        up_lag = UPSTREAM_LAG.get(lag, UPSTREAM_LAG[2])
        up_val = up_lag[di]
        lead_lag_active = (not np.isnan(up_val)) and up_val >= upstream_min

        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 0.5 or zs <= z_min: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue

            score = roc * zs * ga

            # RS rank bonus
            rs = RS_RANK[s, di]
            if not np.isnan(rs):
                score *= (0.7 + 0.3 * rs)

            # Lead-lag bonus
            if lead_lag_active:
                if s in downstream_si:
                    score *= 1.5
                elif s in upstream_si:
                    score *= 1.2

            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((score, s, ep, f'combo_z_a{align_threshold:.0%}_z{z_min}'))
        return c

    # ===================== HELPERS =====================
    def compute_composite(di, daily_eq, high_water, perf_window=20):
        scores = []
        # Market breadth (group alignment as proxy)
        bth = GROUP_ALIGN[di]
        if not np.isnan(bth):
            scores.append(np.clip((bth - 0.4) / (0.8 - 0.4), 0, 1))
        # Volatility regime
        vol = MKT_VOL[di]
        if not np.isnan(vol) and VOL_MEDIAN > 0:
            vol_ratio = vol / VOL_MEDIAN
            scores.append(np.clip((1.5 - vol_ratio) / (1.5 - 0.8), 0, 1))
        # Equity curve trend
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
        # Drawdown factor
        if high_water > 0:
            cur_dd = (daily_eq[-1] - high_water) / high_water
        else:
            cur_dd = 0
        scores.append(np.clip(1.0 + cur_dd / 0.3, 0, 1))
        return np.mean(scores) if scores else 0.5

    def dd_size(pv, high_water, tiers):
        if high_water <= 0: return tiers[0][1]
        dd = (pv - high_water) / high_water
        for dd_thresh, size_frac in tiers:
            if dd >= -dd_thresh: return size_frac
        return tiers[-1][1]

    # ===================== BACKTEST ENGINE =====================
    def backtest(start_di=MIN_TRAIN, end_di=None,
                 atr_norm_max=10.0, dd_tiers=None,
                 regime_lo=0.5, regime_hi=1.5,
                 top_n=3, hold=1,
                 signal_fn=None,
                 short_mode='long_only'):
        if end_di is None: end_di = ND
        if dd_tiers is None:
            dd_tiers = [(0, 1.00), (0.10, 0.90), (0.20, 0.70), (0.30, 0.50)]
        if signal_fn is None:
            signal_fn = sig_v121_baseline

        cash = float(CASH0)
        positions = []
        trades = []
        daily_eq = []
        high_water = float(CASH0)
        trade_pnls = []

        for di in range(start_di, end_di - 1):
            # Mark-to-market
            pv = cash
            for p in positions:
                cp = C[p['si'], di]
                if not np.isnan(cp) and cp > 0:
                    m = MULT.get(p['sym'], DEF_MULT)
                    unrealized = (cp - p['entry_price']) * m * p['lots']
                    pv += p['entry_price'] * m * abs(p['lots']) + unrealized - cp * m * abs(p['lots']) * COMM
            daily_eq.append(pv)
            if pv > high_water:
                high_water = pv

            # --- Exit logic: fixed hold ---
            cl = []
            for p in positions:
                days_held = di - p['entry_di']
                if days_held >= hold:
                    cp = C[p['si'], di]
                    if np.isnan(cp) or cp <= 0: continue
                    m = MULT.get(p['sym'], DEF_MULT)
                    pnl = (cp - p['entry_price']) * m * p['lots']
                    inv = p['entry_price'] * m * abs(p['lots'])
                    pp = pnl / inv * 100 if inv > 0 else 0
                    cash += cp * m * abs(p['lots']) * (1 - COMM)
                    trades.append(pp)
                    trade_pnls.append(pnl)
                    cl.append(p)
            for p in cl: positions.remove(p)

            # --- Position sizing ---
            dd_sz = dd_size(pv, high_water, dd_tiers)
            composite = compute_composite(di, daily_eq, high_water)
            regime_mult = regime_lo + composite * (regime_hi - regime_lo)
            pos_size = max(0.05, min(0.99, dd_sz * regime_mult))

            # --- Enter positions ---
            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue

            held_si = set(p['si'] for p in positions)

            # Long signals from the chosen signal function
            cands = signal_fn(di, edi)
            cands_f = [c for c in cands
                       if not np.isnan(ATR_NORM[c[1], di]) and ATR_NORM[c[1], di] < atr_norm_max]
            cands_f.sort(key=lambda x: -x[0])

            for sc, s, pr, sig_str in cands_f:
                if s in held_si: continue
                if len(positions) >= top_n: break
                cap = cash * pos_size / max(1, top_n - len(positions))
                sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                ct = max(1, int(cap / (pr * m * (1 + COMM))))
                ci = pr * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                    ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                if ct <= 0 or ci <= 0 or ci > cash: continue
                cash -= ci
                positions.append({
                    'si': s, 'entry_price': pr, 'entry_di': edi,
                    'lots': ct, 'sym': sym, 'sig': sig_str, 'score': sc
                })
                held_si.add(s)

        # Close remaining positions
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

        avg_pnl = np.mean(trade_pnls) if trade_pnls else 0

        return {'ann': ann, 'wr': wr, 'n': nt, 'mdd': mdd, 'sharpe': sh,
                'final': cash, 'avg_pnl': avg_pnl}

    # ===================== PRINTING HELPERS =====================
    def pr(r, label=""):
        ratio = abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0
        print(f"  {label:85s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f} | R/M={ratio:6.2f} | N={r['n']:4d} | AvgPnL={r['avg_pnl']:>8.0f}")

    def walk_forward(signal_fn, label="", hold=1, top_n=3, **kwargs):
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue
            r = backtest(start_di=ys, end_di=ye, signal_fn=signal_fn,
                         hold=hold, top_n=top_n, **kwargs)
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

    def _parse_label_to_fn(lbl):
        """Parse a result label back to (signal_fn, hold) for walk-forward/test."""
        if lbl.startswith('combo_z_'):
            a_th = 0.5 if 'a0.5' in lbl else (0.6 if 'a0.6' in lbl else 0.75)
            z_m = 0.5 if 'z0.5' in lbl else (1.0 if 'z1.0' in lbl else 1.5)
            h = 2 if 'h2' in lbl else 1
            fn = lambda di, edi, a=a_th, z=z_m: sig_combo_zscore(di, edi, align_threshold=a, z_min=z)
        elif lbl.startswith('combo_'):
            a_th = 0.5 if 'a50' in lbl else (0.6 if 'a60' in lbl else 0.75)
            lag = 2 if 'l2' in lbl else (3 if 'l3' in lbl else 5)
            h = int(lbl.split('_')[-1].replace('h', '')) if 'h' in lbl.split('_')[-1] else 1
            fn = lambda di, edi, a=a_th, l=lag: sig_combo_group(di, edi, align_threshold=a, lag=l)
        elif lbl.startswith('leadlag_all'):
            lag = 2 if 'l2' in lbl else 3
            up = 0.5 if 'u0.5' in lbl else 1.0
            fn = lambda di, edi, l=lag, u=up: sig_lead_lag_all(di, edi, lag=l, upstream_min=u)
            h = 2
        elif lbl.startswith('leadlag_'):
            # Label format: leadlag_l{lag}_u{up_min}_h{hold}
            parts = lbl.split('_')
            lag = int(parts[1].replace('l', ''))
            up = float(parts[2].replace('u', ''))
            h = int(parts[3].replace('h', ''))
            fn = lambda di, edi, l=lag, u=up: sig_lead_lag(di, edi, lag=l, upstream_min=u)
        elif lbl.startswith('align_'):
            a_s = lbl.split('_')[1]
            a_th = float(a_s.replace('%', '')) / 100
            h = int(lbl.split('_')[-1].replace('h', ''))
            fn = lambda di, edi, a=a_th: sig_group_aligned(di, edi, align_threshold=a)
        elif lbl.startswith('strict_'):
            a_s = lbl.split('_')[1]
            a_th = float(a_s.replace('%', '')) / 100
            fn = lambda di, edi, a=a_th: sig_group_aligned_strict(di, edi, align_threshold=a)
            h = 1
        elif lbl.startswith('rs_'):
            parts = lbl.split('_')
            rs_min = float(parts[1])
            a_s = parts[2].replace('a', '').replace('%', '')
            a_th = float(a_s) / 100
            h = int(parts[3].replace('h', ''))
            fn = lambda di, edi, r=rs_min, a=a_th: sig_rs_rank(di, edi, rs_min=r, align_threshold=a)
        elif lbl.startswith('oversold'):
            h = int(lbl.split('_')[-1].replace('h', ''))
            fn = sig_group_oversold
        else:
            return None, 1
        return fn, h

    all_results = []

    # ===================== SECTION 0: BASELINE (V121) =====================
    print("\n" + "=" * 130)
    print("  SECTION 0: BASELINE — V121 signal (no group factors)")
    print("=" * 130)

    r_base = backtest(signal_fn=sig_v121_baseline, hold=1, top_n=3)
    pr(r_base, "BASELINE: V121, hold=1, top_n=3")
    base_rm = abs(r_base['ann'] / r_base['mdd']) if r_base['mdd'] != 0 else 0
    all_results.append({**r_base, 'label': 'baseline_v121'})

    # ===================== SECTION 1: GROUP ALIGNMENT =====================
    print("\n" + "=" * 130)
    print("  SECTION 1: Group Momentum Alignment (% of group with positive ROC5)")
    print("  Hypothesis: Group alignment filters out false signals")
    print("=" * 130)

    for align_th in [0.5, 0.6, 0.75, 1.0]:
        for h in [1, 2, 3]:
            fn = lambda di, edi, a=align_th: sig_group_aligned(di, edi, align_threshold=a)
            r = backtest(signal_fn=fn, hold=h, top_n=3)
            pr(r, f"GROUP_ALIGN >{align_th:.0%}, hold={h}")
            all_results.append({**r, 'label': f'align_{align_th:.0%}_h{h}'})

    # Strict alignment with Z-score
    for align_th in [0.75, 1.0]:
        fn = lambda di, edi, a=align_th: sig_group_aligned_strict(di, edi, align_threshold=a)
        r = backtest(signal_fn=fn, hold=1, top_n=3)
        pr(r, f"GROUP_STRICT >{align_th:.0%} + Z>1.0")
        all_results.append({**r, 'label': f'strict_{align_th:.0%}'})

    # ===================== SECTION 2: RELATIVE STRENGTH RANK =====================
    print("\n" + "=" * 130)
    print("  SECTION 2: Relative Strength Rank within Group")
    print("  Hypothesis: In a bullish group, strongest member continues")
    print("=" * 130)

    for rs_min in [0.3, 0.5, 0.75]:
        for align_th in [0.5, 0.6]:
            fn = lambda di, edi, r=rs_min, a=align_th: sig_rs_rank(di, edi, rs_min=r, align_threshold=a)
            for h in [1, 2]:
                r = backtest(signal_fn=fn, hold=h, top_n=2)
                pr(r, f"RS_RANK >{rs_min:.1f} + ALIGN >{align_th:.0%}, hold={h}, top_n=2")
                all_results.append({**r, 'label': f'rs_{rs_min:.1f}_a{align_th:.0%}_h{h}'})

    # ===================== SECTION 3: SUPPLY CHAIN LEAD-LAG =====================
    print("\n" + "=" * 130)
    print("  SECTION 3: Supply Chain Lead-Lag (upstream -> downstream)")
    print("  Hypothesis: Iron ore/coke leads steel/hot coil")
    print("=" * 130)

    # Lead-lag: only buy downstream when upstream led
    for lag in [1, 2, 3, 5]:
        for up_min in [0.5, 1.0, 2.0]:
            for h in [1, 2, 3]:
                fn = lambda di, edi, l=lag, u=up_min: sig_lead_lag(di, edi, lag=l, upstream_min=u)
                r = backtest(signal_fn=fn, hold=h, top_n=2)
                pr(r, f"LEAD_LAG lag={lag}d up>{up_min:.0f}%, hold={h}")
                all_results.append({**r, 'label': f'leadlag_l{lag}_u{up_min:.0f}_h{h}'})

    # Lead-lag: all symbols (upstream also buys when it leads)
    for lag in [2, 3]:
        for up_min in [0.5, 1.0]:
            fn = lambda di, edi, l=lag, u=up_min: sig_lead_lag_all(di, edi, lag=l, upstream_min=u)
            r = backtest(signal_fn=fn, hold=2, top_n=3)
            pr(r, f"LEAD_LAG_ALL lag={lag}d up>{up_min:.0f}%, hold=2")
            all_results.append({**r, 'label': f'leadlag_all_l{lag}_u{up_min:.0f}_h2'})

    # ===================== SECTION 4: GROUP OVERSOLD MEAN REVERSION =====================
    print("\n" + "=" * 130)
    print("  SECTION 4: Group Oversold Mean Reversion")
    print("  Hypothesis: Buy reversals when entire group is oversold")
    print("=" * 130)

    for h in [1, 2, 3]:
        r = backtest(signal_fn=sig_group_oversold, hold=h, top_n=2)
        pr(r, f"GROUP_OVERSOLD, hold={h}")
        all_results.append({**r, 'label': f'oversold_h{h}'})

    # ===================== SECTION 5: COMBO — ALL FACTORS =====================
    print("\n" + "=" * 130)
    print("  SECTION 5: COMBO — Group Alignment + RS Rank + Lead-Lag")
    print("  Hypothesis: Multi-factor score captures best opportunities")
    print("=" * 130)

    for align_th in [0.5, 0.6, 0.75]:
        for lag in [2, 3]:
            for h in [1, 2, 3]:
                fn = lambda di, edi, a=align_th, l=lag: sig_combo_group(
                    di, edi, align_threshold=a, lag=l)
                r = backtest(signal_fn=fn, hold=h, top_n=3)
                pr(r, f"COMBO align>{align_th:.0%} lag={lag}d, hold={h}")
                all_results.append({**r, 'label': f'combo_a{align_th:.0%}_l{lag}_h{h}'})

    # ===================== SECTION 6: COMBO + Z-SCORE =====================
    print("\n" + "=" * 130)
    print("  SECTION 6: COMBO+Z — Alignment + Z-score + Lead-Lag")
    print("  Hypothesis: Statistical significance + group context = best signal")
    print("=" * 130)

    for align_th in [0.5, 0.6]:
        for z_min in [0.5, 1.0, 1.5]:
            for lag in [2, 3]:
                fn = lambda di, edi, a=align_th, z=z_min, l=lag: sig_combo_zscore(
                    di, edi, align_threshold=a, z_min=z, lag=l)
                r = backtest(signal_fn=fn, hold=1, top_n=3)
                pr(r, f"COMBO_Z align>{align_th:.0%} Z>{z_min} lag={lag}d")
                all_results.append({**r, 'label': f'combo_z_a{align_th:.0%}_z{z_min}_l{lag}'})

    # COMBO+Z with hold=2
    for align_th in [0.5, 0.6]:
        for z_min in [0.5, 1.0]:
            fn = lambda di, edi, a=align_th, z=z_min: sig_combo_zscore(
                di, edi, align_threshold=a, z_min=z)
            r = backtest(signal_fn=fn, hold=2, top_n=3)
            pr(r, f"COMBO_Z align>{align_th:.0%} Z>{z_min}, hold=2")
            all_results.append({**r, 'label': f'combo_z_a{align_th:.0%}_z{z_min}_h2'})

    # ===================== SECTION 7: TOP_N VARIANTS =====================
    print("\n" + "=" * 130)
    print("  SECTION 7: Top-N position variants on best signals")
    print("=" * 130)

    for tn in [1, 2, 3]:
        for h in [1, 2]:
            fn = lambda di, edi: sig_combo_group(di, edi, align_threshold=0.6, lag=2)
            r = backtest(signal_fn=fn, hold=h, top_n=tn)
            pr(r, f"COMBO align>60% lag=2d, hold={h}, top_n={tn}")
            all_results.append({**r, 'label': f'combo_a60_l2_h{h}_tn{tn}'})

    # ===================== SECTION 8: ATR NORM FILTER SENSITIVITY =====================
    print("\n" + "=" * 130)
    print("  SECTION 8: ATR Normalization Filter Sensitivity")
    print("=" * 130)

    for atr_max in [5.0, 7.0, 10.0, 15.0]:
        fn = lambda di, edi: sig_combo_group(di, edi, align_threshold=0.6, lag=2)
        r = backtest(signal_fn=fn, hold=2, top_n=3, atr_norm_max=atr_max)
        pr(r, f"COMBO align>60% lag=2d, hold=2, ATR<{atr_max:.0f}%")
        all_results.append({**r, 'label': f'combo_a60_l2_atr{atr_max:.0f}'})

    # ===================== WALK-FORWARD =====================
    print("\n" + "=" * 130)
    print("  WALK-FORWARD ANALYSIS")
    print("=" * 130)

    print(f"\n  Baseline R/M = {base_rm:.2f}")

    # Walk-forward for baseline
    print(f"\n  Walk-forward: BASELINE V121")
    wf_base = walk_forward(sig_v121_baseline, hold=1)
    print_wf(wf_base, "BASELINE V121")

    # Top 5 by R/M
    ranked = sorted(all_results, key=lambda x: abs(x.get('ann', 0) / x.get('mdd', -1)) if x.get('mdd', 0) != 0 else 0, reverse=True)
    top5 = [r for r in ranked if r.get('label') != 'baseline_v121'][:5]

    # Walk-forward for top configs
    for r in top5:
        lbl = r['label']
        print(f"\n  Walk-forward: {lbl}")
        fn, h = _parse_label_to_fn(lbl)
        if fn is None:
            continue

        wf = walk_forward(fn, hold=h)
        print_wf(wf, lbl)

    # ===================== TRAIN/TEST SPLIT =====================
    print("\n" + "=" * 130)
    print("  TRAIN/TEST SPLIT: Train 2019-2023, Test 2024-2026")
    print("=" * 130)

    # Find train/test boundaries
    train_start = train_end = test_start = test_end = None
    for di in range(ND):
        if dates[di].year == 2019 and train_start is None: train_start = di
        if dates[di].year == 2023: train_end = di + 1
        if dates[di].year == 2024 and test_start is None: test_start = di
        if dates[di].year == 2026: test_end = di + 1
    if test_end is None: test_end = ND

    print(f"  Train: day {train_start}-{train_end} | Test: day {test_start}-{test_end}")

    # Evaluate top configs on train then test
    print(f"\n  {'Config':40s} | {'Train Ann':>9s} | {'Train R/M':>9s} | {'Test Ann':>9s} | {'Test R/M':>9s} | {'Test N':>6s}")
    print(f"  {'-'*40}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*6}")

    for r in ranked[:10]:
        lbl = r['label']
        fn, h = _parse_label_to_fn(lbl)
        if fn is None:
            continue

        r_train = backtest(start_di=train_start, end_di=train_end, signal_fn=fn, hold=h)
        r_test = backtest(start_di=test_start, end_di=test_end, signal_fn=fn, hold=h)
        tr_rm = abs(r_train['ann'] / r_train['mdd']) if r_train['mdd'] != 0 else 0
        te_rm = abs(r_test['ann'] / r_test['mdd']) if r_test['mdd'] != 0 else 0
        print(f"  {lbl:40s} | {r_train['ann']:>+8.0f}% | {tr_rm:>9.2f} | {r_test['ann']:>+8.0f}% | {te_rm:>9.2f} | {r_test['n']:>6d}")

    # ===================== FINAL SUMMARY =====================
    print("\n" + "=" * 130)
    print("  V183 FINAL SUMMARY: Cross-Commodity Group Momentum")
    print("=" * 130)

    print(f"\n  {'Config':45s} | {'Ann':>7s} | {'MDD':>5s} | {'R/M':>6s} | {'WR':>5s} | {'N':>4s} | {'Sh':>5s} | vs Base")
    print(f"  {'-'*45}-+-{'-'*7}-+-{'-'*5}-+-{'-'*6}-+-{'-'*5}-+-{'-'*4}-+-{'-'*5}-+-{'-'*8}")
    print(f"  {'BASELINE V121 (hold=1)':45s} | {r_base['ann']:>+7.0f}% | {r_base['mdd']:>5.0f}% | {base_rm:>6.2f} | {r_base['wr']:>5.1f}% | {r_base['n']:>4d} | {r_base['sharpe']:>5.2f} |    ---")

    for r in ranked[:20]:
        ann = r['ann']; mdd = r['mdd']
        rm = abs(ann / mdd) if mdd != 0 else 0
        delta = rm - base_rm
        marker = " ***" if delta > 2.0 else (" **" if delta > 1.0 else (" *" if delta > 0 else ""))
        print(f"  {r['label']:45s} | {ann:>+7.0f}% | {mdd:>5.0f}% | {rm:>6.2f} | {r['wr']:>5.1f}% | {r['n']:>4d} | {r['sharpe']:>5.2f} | {delta:>+8.2f}{marker}")

    # Factor contribution summary
    print(f"\n  FACTOR CONTRIBUTION (avg R/M by category):")
    categories = {
        'Group Alignment': [r for r in all_results if r['label'].startswith('align_')],
        'Group Strict (Z)': [r for r in all_results if r['label'].startswith('strict_')],
        'Relative Strength': [r for r in all_results if r['label'].startswith('rs_')],
        'Lead-Lag (downstream only)': [r for r in all_results if r['label'].startswith('leadlag_l')],
        'Lead-Lag (all symbols)': [r for r in all_results if r['label'].startswith('leadlag_all')],
        'Group Oversold': [r for r in all_results if r['label'].startswith('oversold')],
        'Combo (Align+RS+LeadLag)': [r for r in all_results if r['label'].startswith('combo_a')],
        'Combo+Z (all factors)': [r for r in all_results if r['label'].startswith('combo_z_')],
    }
    for cat_name, cat_results in categories.items():
        if not cat_results: continue
        avg_rm = np.mean([abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0 for r in cat_results])
        best_rm = max(abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0 for r in cat_results)
        best_label = max(cat_results, key=lambda r: abs(r['ann'] / r['mdd']) if r['mdd'] != 0 else 0)['label']
        n_configs = len(cat_results)
        print(f"    {cat_name:40s} | Avg R/M={avg_rm:>6.2f} | Best R/M={best_rm:>6.2f} | {n_configs} configs")
        print(f"    {'':40s} | Best: {best_label}")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
