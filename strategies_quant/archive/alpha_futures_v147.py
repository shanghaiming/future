"""
Alpha Futures V147 — SIGNAL ENHANCEMENT VARIANTS
==============================================================================
Goal: Find STRONGER signals with higher WR or better avg win/loss to
naturally reduce MDD while maintaining or improving returns.

Test signal variants:
  A) Momentum Quality Score (body_ratio + volume confirmation)
  B) Multi-Timeframe Momentum (ROC alignment across 3/5/10/20)
  C) Overnight Confirmation + Intraday Continuation (tighter thresholds + strong close)
  D) OI Confirmation as ranking bonus
  E) Volatility-Adjusted Entry (skip unusual vol spikes)
  F) Trend Strength Gate (ADX filter)

For each: standalone (50% sizing) + 50/50 with V121 + walk-forward.
Baselines: V121 = +123% @50% sizing, Union = +141% @50% sizing
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
    print("  V147 — SIGNAL ENHANCEMENT VARIANTS")
    print("=" * 130)
    t_start = time.time()
    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(load_oi=True)
    print(f"  {NS} commodities, {ND} days")

    print("\n[Precompute]...", flush=True)
    t0 = time.time()

    # --- Basic indicators ---
    RET = np.full((NS, ND), np.nan)
    ROC3 = np.full((NS, ND), np.nan)
    ROC5 = np.full((NS, ND), np.nan)
    ROC10 = np.full((NS, ND), np.nan)
    ROC20 = np.full((NS, ND), np.nan)
    for si in range(NS):
        c = C[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(c[di]) and not np.isnan(c[di-1]) and c[di-1] > 0:
                RET[si, di] = (c[di] / c[di-1] - 1) * 100
        ROC3[si] = talib.ROC(c, timeperiod=3)
        ROC5[si] = talib.ROC(c, timeperiod=5)
        ROC10[si] = talib.ROC(c, timeperiod=10)
        ROC20[si] = talib.ROC(c, timeperiod=20)

    ATR14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        ATR14[si] = talib.ATR(H[si].astype(np.float64), L[si].astype(np.float64),
                               C[si].astype(np.float64), timeperiod=14)

    ZSCORE = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(21, ND):
            rets = RET[si, di-20:di]
            v = rets[~np.isnan(rets)]
            if len(v) < 10: continue
            s = np.std(v, ddof=1)
            if s > 0 and not np.isnan(RET[si, di]):
                ZSCORE[si, di] = (RET[si, di] - np.mean(v)) / s

    OV_GAP = np.full((NS, ND), np.nan)
    ID_RET = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            o, c = O[si, di], C[si, di]
            if not np.isnan(o) and not np.isnan(c):
                if di > 0 and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                    OV_GAP[si, di] = (o - C[si, di-1]) / C[si, di-1] * 100
                if o > 0: ID_RET[si, di] = (c - o) / o * 100

    # --- NEW: Body ratio (close-open) / (high-low) ---
    BODY_RATIO = np.full((NS, ND), np.nan)
    for si in range(NS):
        for di in range(ND):
            o, c, h, l = O[si, di], C[si, di], H[si, di], L[si, di]
            if any(np.isnan(x) for x in [o, c, h, l]): continue
            rng = h - l
            if rng > 0:
                BODY_RATIO[si, di] = abs(c - o) / rng

    # --- NEW: Volume ratio vs 20d avg ---
    VOL_RATIO = np.full((NS, ND), np.nan)
    for si in range(NS):
        v = V[si].astype(np.float64)
        vma = talib.MA(v, timeperiod=20)
        for di in range(ND):
            if not np.isnan(vma[di]) and vma[di] > 0 and not np.isnan(v[di]):
                VOL_RATIO[si, di] = v[di] / vma[di]

    # --- NEW: ATR percentile (60d median) ---
    ATR_MED60 = np.full((NS, ND), np.nan)
    for si in range(NS):
        atr = ATR14[si]
        for di in range(60, ND):
            window = atr[di-59:di+1]
            valid = window[~np.isnan(window)]
            if len(valid) >= 20:
                ATR_MED60[si, di] = np.median(valid)

    # --- NEW: OI change percentile (vs 20d) ---
    OI_CHANGE = np.full((NS, ND), np.nan)
    OI_PCTILE = np.full((NS, ND), np.nan)
    for si in range(NS):
        oi = OI[si].astype(np.float64)
        for di in range(1, ND):
            if not np.isnan(oi[di]) and not np.isnan(oi[di-1]) and oi[di-1] > 0:
                OI_CHANGE[si, di] = (oi[di] - oi[di-1]) / oi[di-1] * 100
        for di in range(20, ND):
            window = OI_CHANGE[si, di-19:di+1]
            valid = window[~np.isnan(window)]
            if len(valid) >= 10:
                OI_PCTILE[si, di] = np.sum(valid < OI_CHANGE[si, di]) / len(valid) if not np.isnan(OI_CHANGE[si, di]) else np.nan

    # --- NEW: ADX ---
    ADX14 = np.full((NS, ND), np.nan)
    for si in range(NS):
        ADX14[si] = talib.ADX(H[si].astype(np.float64), L[si].astype(np.float64),
                                C[si].astype(np.float64), timeperiod=14)

    print(f"  Precompute done ({time.time()-t0:.1f}s)")

    # ===================== SIGNAL DEFINITIONS =====================

    # --- BASELINE: V121 ---
    def sig_v121(di, edi):
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

    # --- BASELINE: Union ---
    def sig_union(di, edi):
        all_sigs = {}
        # V121 component
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += roc * zs * 3
            all_sigs[s][2].append('v121')
        # OV/ID component
        for s in range(NS):
            ov = OV_GAP[s, di]; idr = ID_RET[s, di]; roc = ROC5[s, di]
            if any(np.isnan(x) for x in [ov, idr, roc]): continue
            if ov <= 0.3 or idr <= 0.3 or roc <= 1.0: continue
            zs = ZSCORE[s, di]
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            z_bonus = zs if not np.isnan(zs) and zs > 1.0 else 1.0
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += (ov + idr) * roc * z_bonus * 2 * 2
            all_sigs[s][2].append('ov_id')
        # Final Flag component
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
            if s not in all_sigs: all_sigs[s] = [0, ep, []]
            all_sigs[s][0] += roc20 * (cp - h4) / atr
            all_sigs[s][2].append('ff')
        return [(sc, s, ep, '+'.join(sigs)) for s, (sc, ep, sigs) in all_sigs.items()]

    # ---- A) MOMENTUM QUALITY SCORE ----
    # score = ROC5 * (1 + body_ratio) * (1 + volume_ratio_vs_20d_avg)
    # Only take signals where score > 2.0
    def sig_momentum_quality(di, edi, score_thresh=2.0):
        c = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue
            br = BODY_RATIO[s, di]
            vr = VOL_RATIO[s, di]
            if np.isnan(br): br = 0.5
            if np.isnan(vr): vr = 1.0
            quality = roc / 100.0 * (1 + br) * (1 + vr)  # roc is in %
            quality_score = quality * 100  # scale back to meaningful range
            if quality_score < score_thresh: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((quality_score * zs, s, ep, 'mq'))
        return c

    # ---- B) MULTI-TIMEFRAME MOMENTUM ----
    # ROC(3)>0 AND ROC(5)>1% AND ROC(10)>0 AND ROC(20)>0
    def sig_multi_tf(di, edi):
        c = []
        for s in range(NS):
            roc3 = ROC3[s, di]; roc5 = ROC5[s, di]
            roc10 = ROC10[s, di]; roc20 = ROC20[s, di]
            if any(np.isnan(x) for x in [roc3, roc5, roc10, roc20]): continue
            if roc3 <= 0 or roc5 <= 1.0 or roc10 <= 0 or roc20 <= 0: continue
            zs = ZSCORE[s, di]
            if np.isnan(zs) or zs <= 1.0: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            # Score: sum of aligned ROCs weighted by z-score
            score = (roc3 + roc5 * 2 + roc10 + roc20) * zs
            c.append((score, s, ep, 'mtf'))
        return c

    # ---- C) OVERNIGHT + INTRADAY TIGHTER ----
    # OV>0.5% AND ID>0.5% AND ROC(5)>1.5% + close in top 25% of day's range
    def sig_ov_id_tight(di, edi):
        c = []
        for s in range(NS):
            ov = OV_GAP[s, di]; idr = ID_RET[s, di]; roc = ROC5[s, di]
            if any(np.isnan(x) for x in [ov, idr, roc]): continue
            if ov <= 0.5 or idr <= 0.5 or roc <= 1.5: continue
            # Strong close: close in top 25% of day's range
            o, cp, h, l = O[s, di], C[s, di], H[s, di], L[s, di]
            if any(np.isnan(x) for x in [o, cp, h, l]): continue
            rng = h - l
            if rng <= 0: continue
            close_position = (cp - l) / rng  # 0=bottom, 1=top
            if close_position < 0.75: continue  # must be in top 25%
            zs = ZSCORE[s, di]
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            z_bonus = zs if not np.isnan(zs) and zs > 1.0 else 1.0
            score = (ov + idr) * roc * z_bonus * close_position
            c.append((score, s, ep, 'ovid_t'))
        return c

    # ---- D) OI RANKING BONUS ----
    # base V121 signal, but score boosted by OI change percentile
    # score = base_score * (1 + 0.3 * OI_pctile)
    def sig_oi_boosted(di, edi):
        c = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            base = roc * zs
            oi_pct = OI_PCTILE[s, di]
            if np.isnan(oi_pct): oi_pct = 0.5
            score = base * (1 + 0.3 * oi_pct)
            c.append((score, s, ep, 'oi_b'))
        return c

    # ---- E) VOLATILITY-ADJUSTED ENTRY ----
    # Only enter when ATR(14) < 1.5 * median_ATR(60)
    def sig_vol_adj(di, edi, atr_mult=1.5):
        c = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue
            # Volatility gate
            atr = ATR14[s, di]; med_atr = ATR_MED60[s, di]
            if np.isnan(atr) or np.isnan(med_atr) or med_atr <= 0: continue
            if atr > atr_mult * med_atr: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((roc * zs, s, ep, 'vadj'))
        return c

    # ---- F) TREND STRENGTH GATE (ADX) ----
    # ADX > 25: size=60%, ADX 20-25: size=45%, ADX < 20: skip
    # Returns (score, s, ep, sig, size_mult)
    def sig_adx_gate(di, edi, adx_strong=25, adx_min=20):
        c = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]; adx = ADX14[s, di]
            if np.isnan(roc) or np.isnan(zs) or np.isnan(adx): continue
            if roc <= 1.0 or zs <= 1.5: continue
            if adx < adx_min: continue  # No trend, skip
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            size_mult = 0.60 if adx >= adx_strong else 0.45
            c.append((roc * zs, s, ep, 'adx_g', size_mult))
        return c

    # ===================== BACKTEST ENGINE =====================
    def backtest(signal_func, hold=1, top_n=1, start_di=MIN_TRAIN, end_di=None,
                 size_frac=0.50, adx_sizing=False):
        """Standard backtest with configurable sizing.
        adx_sizing: if True, use per-signal size_mult from ADX gate."""
        if end_di is None: end_di = ND
        cash = float(CASH0)
        positions = []; trades = []; daily_eq = []
        for di in range(start_di, end_di - 1):
            pv = cash
            for p in positions:
                cp = C[p['si'], di]
                if not np.isnan(cp) and cp > 0:
                    m = MULT.get(p['sym'], DEF_MULT)
                    pv += cp * m * p['lots'] - cp * m * abs(p['lots']) * COMM
            daily_eq.append(pv)

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
                    trades.append({'pnl_pct': pp, 'sig': p.get('sig', '')})
                    cl.append(p)
            for p in cl: positions.remove(p)

            if len(positions) >= top_n: continue
            edi = di + 1
            if edi >= end_di: continue
            cands = signal_func(di, edi)
            if not cands: continue
            cands.sort(key=lambda x: -x[0])
            ns = top_n - len(positions)

            for item in cands[:ns]:
                if len(item) == 3:
                    sc, s, pr = item; sig = ''; sm = size_frac
                elif len(item) == 4:
                    sc, s, pr, sig = item; sm = size_frac
                elif len(item) == 5:
                    sc, s, pr, sig, sm = item
                    if not adx_sizing: sm = size_frac
                else:
                    continue
                sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                cap = cash * sm / max(1, ns)
                ct = max(1, int(cap / (pr * m * (1 + COMM))))
                ci = pr * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                    ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                if ct <= 0 or ci <= 0 or ci > cash: continue
                cash -= ci
                positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                  'lots': ct, 'dir': 1, 'sym': sym,
                                  'hold_days': hold, 'sig': sig})

        for p in positions:
            ae = end_di - 1
            ep = C[p['si'], min(ae, ND-1)]
            if np.isnan(ep) or ep <= 0: ep = p['entry_price']
            m = MULT.get(p['sym'], DEF_MULT)
            cash += ep * m * abs(p['lots']) * (1 - COMM)

        nd = end_di - start_di
        ann = annual_return(cash, CASH0, nd)
        wr = np.mean([1 if t['pnl_pct'] > 0 else 0 for t in trades]) * 100 if trades else 0
        nt = len(trades)
        ap = np.mean([t['pnl_pct'] for t in trades]) if trades else 0
        wins = [t for t in trades if t['pnl_pct'] > 0]
        losses = [t for t in trades if t['pnl_pct'] <= 0]
        avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
        wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        if daily_eq:
            eq = np.array(daily_eq); pk = np.maximum.accumulate(eq)
            mdd = np.min((eq - pk) / pk * 100)
            r = np.diff(eq) / eq[:-1]
            r = np.where(np.isfinite(r), r, 0)
            sh = np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0
        else: mdd = 0; sh = 0
        return {'ann': ann, 'wr': wr, 'n': nt, 'avg_pnl': ap, 'mdd': mdd, 'sharpe': sh,
                'final': cash, 'avg_win': avg_win, 'avg_loss': avg_loss, 'wl_ratio': wl_ratio}

    # ===================== PORTFOLIO BACKTEST (50/50) =====================
    def backtest_portfolio(sig_A, sig_B, start_di=MIN_TRAIN, end_di=None,
                           size_frac_A=0.50, size_frac_B=0.50,
                           adx_sizing_A=False, adx_sizing_B=False):
        """Run two sub-strategies independently, combine equity 50/50."""
        if end_di is None: end_di = ND

        def run_sub(sig_func, size_frac, adx_sizing):
            cash = float(CASH0); positions = []; daily_eq = []
            for di in range(start_di, end_di - 1):
                pv = cash
                for p in positions:
                    cp = C[p['si'], di]
                    if not np.isnan(cp) and cp > 0:
                        m = MULT.get(p['sym'], DEF_MULT)
                        pv += cp * m * p['lots'] - cp * m * abs(p['lots']) * COMM
                daily_eq.append(pv)

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
                        cl.append(p)
                for p in cl: positions.remove(p)

                if len(positions) >= 1: continue
                edi = di + 1
                if edi >= end_di: continue
                cands = sig_func(di, edi)
                if not cands: continue
                cands.sort(key=lambda x: -x[0])
                item = cands[0]
                if len(item) == 3:
                    sc, s, pr = item; sig = ''; sm = size_frac
                elif len(item) == 4:
                    sc, s, pr, sig = item; sm = size_frac
                elif len(item) == 5:
                    sc, s, pr, sig, sm = item
                    if not adx_sizing: sm = size_frac
                else:
                    continue
                sym = syms[s]; m = MULT.get(sym, DEF_MULT)
                cap = cash * sm
                ct = max(1, int(cap / (pr * m * (1 + COMM))))
                ci = pr * m * ct * (1 + COMM)
                if ci > cash:
                    ct = int(cash * 0.9 / (pr * m * (1 + COMM)))
                    ci = pr * m * ct * (1 + COMM) if ct > 0 else 0
                if ct <= 0 or ci <= 0 or ci > cash: continue
                cash -= ci
                positions.append({'si': s, 'entry_price': pr, 'entry_di': edi,
                                  'lots': ct, 'dir': 1, 'sym': sym,
                                  'hold_days': 1, 'sig': sig})

            for p in positions:
                ep = C[p['si'], min(end_di-1, ND-1)]
                if np.isnan(ep) or ep <= 0: ep = p['entry_price']
                m = MULT.get(p['sym'], DEF_MULT)
                cash += ep * m * abs(p['lots']) * (1 - COMM)
            return np.array(daily_eq)

        eq_A = run_sub(sig_A, size_frac_A, adx_sizing_A)
        eq_B = run_sub(sig_B, size_frac_B, adx_sizing_B)

        ml = min(len(eq_A), len(eq_B))
        if ml <= 1:
            return {'ann': -100.0, 'mdd': 0, 'sharpe': 0, 'final': CASH0, 'wr': 0, 'n': 0,
                    'avg_win': 0, 'avg_loss': 0, 'wl_ratio': 0}

        ret_A = np.diff(eq_A[:ml]) / eq_A[:ml-1]
        ret_B = np.diff(eq_B[:ml]) / eq_B[:ml-1]
        ret_A = np.where(np.isfinite(ret_A), ret_A, 0)
        ret_B = np.where(np.isfinite(ret_B), ret_B, 0)

        combined = 0.5 * ret_A + 0.5 * ret_B
        eq = np.zeros(ml)
        eq[0] = float(CASH0)
        for i in range(ml - 1):
            eq[i+1] = eq[i] * (1 + combined[i])

        final = eq[-1]
        nd = ml
        ann = annual_return(final, CASH0, nd)
        pk = np.maximum.accumulate(eq)
        mdd = np.min((eq - pk) / pk * 100)
        sh = np.mean(combined) / np.std(combined) * np.sqrt(252) if np.std(combined) > 0 else 0
        return {'ann': ann, 'mdd': mdd, 'sharpe': sh, 'final': final, 'wr': 0, 'n': 0,
                'avg_win': 0, 'avg_loss': 0, 'wl_ratio': 0}

    # ===================== HELPERS =====================
    def pr(r, label=""):
        print(f"  {label:72s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | "
              f"Sh={r['sharpe']:4.2f} | WR={r['wr']:5.1f}% | N={r['n']:4d} | "
              f"W/L={r['wl_ratio']:4.2f}")

    def pr_short(r, label=""):
        print(f"  {label:72s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    def wf(signal_func, hold=1, topn=1, size_frac=0.50, adx_sizing=False):
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue
            r = backtest(signal_func, hold=hold, top_n=topn, start_di=ys, end_di=ye,
                         size_frac=size_frac, adx_sizing=adx_sizing)
            res[yr] = r['ann']
        return res

    def wf_portfolio(sig_A, sig_B, size_frac=0.50, adx_sizing_A=False, adx_sizing_B=False):
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue
            r = backtest_portfolio(sig_A, sig_B, start_di=ys, end_di=ye,
                                   size_frac_A=size_frac, size_frac_B=size_frac,
                                   adx_sizing_A=adx_sizing_A, adx_sizing_B=adx_sizing_B)
            res[yr] = r['ann']
        return res

    def print_wf(label, wf_res):
        pos = sum(1 for v in wf_res.values() if v > 0)
        avg = np.mean(list(wf_res.values())) if wf_res else 0
        ws = " | ".join([f"{yr}:{v:+.0f}%" for yr, v in sorted(wf_res.items())])
        print(f"  {label:72s} | {pos}/6 | Avg={avg:>+7.0f}% | {ws}")

    # ===================== SECTION 1: BASELINES =====================
    print("\n" + "=" * 130)
    print("  SECTION 1: BASELINES (50% sizing)")
    print("=" * 130)

    r_v121 = backtest(sig_v121, size_frac=0.50)
    pr(r_v121, "V121 baseline")

    r_union = backtest(sig_union, size_frac=0.50)
    pr(r_union, "Union baseline")

    wf_v121 = wf(sig_v121, size_frac=0.50)
    print_wf("V121 WF", wf_v121)

    wf_union = wf(sig_union, size_frac=0.50)
    print_wf("Union WF", wf_union)

    # ===================== SECTION 2: MOMENTUM QUALITY SCORE (A) =====================
    print("\n" + "=" * 130)
    print("  SECTION 2: MOMENTUM QUALITY SCORE (body_ratio + volume confirmation)")
    print("=" * 130)

    configs_a = [
        (1.5, "A: MQ thresh=1.5"),
        (2.0, "A: MQ thresh=2.0"),
        (2.5, "A: MQ thresh=2.5"),
        (3.0, "A: MQ thresh=3.0"),
    ]
    results_a = []
    for thresh, label in configs_a:
        r = backtest(lambda di, edi, t=thresh: sig_momentum_quality(di, edi, t), size_frac=0.50)
        r['desc'] = label; results_a.append(r)
        pr(r, label)

    # Best MQ vs V121 portfolio
    print(f"\n  --- MQ / V121 50/50 Portfolios ---")
    best_a = max(results_a, key=lambda x: x['ann'])
    best_a_thresh = 2.0  # default
    for thresh, label in configs_a:
        if results_a[configs_a.index((thresh, label))] == best_a:
            best_a_thresh = thresh
            break

    for thresh, label in configs_a:
        r_port = backtest_portfolio(lambda di, edi, t=thresh: sig_momentum_quality(di, edi, t),
                                    sig_v121, size_frac_A=0.50, size_frac_B=0.50)
        r_port['desc'] = f"MQ({thresh})/V121 50/50"
        pr_short(r_port, r_port['desc'])

    # ===================== SECTION 3: MULTI-TIMEFRAME MOMENTUM (B) =====================
    print("\n" + "=" * 130)
    print("  SECTION 3: MULTI-TIMEFRAME MOMENTUM (ROC3>0 + ROC5>1% + ROC10>0 + ROC20>0)")
    print("=" * 130)

    r_mtf = backtest(sig_multi_tf, size_frac=0.50)
    r_mtf['desc'] = "B: Multi-TF momentum"
    pr(r_mtf, r_mtf['desc'])

    print_wf("B: Multi-TF WF", wf(sig_multi_tf, size_frac=0.50))

    r_mtf_port = backtest_portfolio(sig_multi_tf, sig_v121, size_frac_A=0.50, size_frac_B=0.50)
    pr_short(r_mtf_port, "MTF/V121 50/50")

    # ===================== SECTION 4: OVERNIGHT + INTRADAY TIGHTER (C) =====================
    print("\n" + "=" * 130)
    print("  SECTION 4: OV/ID TIGHTER (OV>0.5% + ID>0.5% + ROC5>1.5% + top-25% close)")
    print("=" * 130)

    r_ovtight = backtest(sig_ov_id_tight, size_frac=0.50)
    r_ovtight['desc'] = "C: OV/ID tight"
    pr(r_ovtight, r_ovtight['desc'])

    print_wf("C: OV/ID tight WF", wf(sig_ov_id_tight, size_frac=0.50))

    r_ovtight_port = backtest_portfolio(sig_ov_id_tight, sig_v121, size_frac_A=0.50, size_frac_B=0.50)
    pr_short(r_ovtight_port, "OVTight/V121 50/50")

    # ===================== SECTION 5: OI RANKING BONUS (D) =====================
    print("\n" + "=" * 130)
    print("  SECTION 5: OI RANKING BONUS (score * (1 + 0.3 * OI_pctile))")
    print("=" * 130)

    r_oi = backtest(sig_oi_boosted, size_frac=0.50)
    r_oi['desc'] = "D: OI boosted"
    pr(r_oi, r_oi['desc'])

    print_wf("D: OI boosted WF", wf(sig_oi_boosted, size_frac=0.50))

    r_oi_port = backtest_portfolio(sig_oi_boosted, sig_v121, size_frac_A=0.50, size_frac_B=0.50)
    pr_short(r_oi_port, "OIBosted/V121 50/50")

    # ===================== SECTION 6: VOLATILITY-ADJUSTED ENTRY (E) =====================
    print("\n" + "=" * 130)
    print("  SECTION 6: VOLATILITY-ADJUSTED ENTRY (skip when ATR > N * median_ATR)")
    print("=" * 130)

    configs_e = [
        (1.5, "E: VolAdj ATR<1.5*median"),
        (2.0, "E: VolAdj ATR<2.0*median"),
        (1.2, "E: VolAdj ATR<1.2*median"),
    ]
    results_e = []
    for mult, label in configs_e:
        r = backtest(lambda di, edi, m=mult: sig_vol_adj(di, edi, m), size_frac=0.50)
        r['desc'] = label; results_e.append(r)
        pr(r, label)

    print(f"\n  --- VolAdj / V121 50/50 Portfolios ---")
    for mult, label in configs_e:
        r_port = backtest_portfolio(lambda di, edi, m=mult: sig_vol_adj(di, edi, m),
                                    sig_v121, size_frac_A=0.50, size_frac_B=0.50)
        pr_short(r_port, f"VolAdj({mult})/V121 50/50")

    # ===================== SECTION 7: ADX TREND STRENGTH GATE (F) =====================
    print("\n" + "=" * 130)
    print("  SECTION 7: ADX TREND STRENGTH GATE (ADX>20 required, sizing by ADX level)")
    print("=" * 130)

    r_adx = backtest(sig_adx_gate, size_frac=0.50, adx_sizing=True)
    r_adx['desc'] = "F: ADX gate (dynamic sizing)"
    pr(r_adx, r_adx['desc'])

    # Also test with fixed sizing for fair comparison
    r_adx_fixed = backtest(sig_adx_gate, size_frac=0.50, adx_sizing=False)
    r_adx_fixed['desc'] = "F: ADX gate (fixed 50% sizing)"
    pr(r_adx_fixed, r_adx_fixed['desc'])

    print_wf("F: ADX gate WF (dynamic)", wf(sig_adx_gate, size_frac=0.50, adx_sizing=True))
    print_wf("F: ADX gate WF (fixed)", wf(sig_adx_gate, size_frac=0.50, adx_sizing=False))

    r_adx_port = backtest_portfolio(sig_adx_gate, sig_v121,
                                     size_frac_A=0.50, size_frac_B=0.50,
                                     adx_sizing_A=True, adx_sizing_B=False)
    pr_short(r_adx_port, "ADX/V121 50/50 (ADX dynamic)")

    # ===================== SECTION 8: COMBINED VARIANTS =====================
    print("\n" + "=" * 130)
    print("  SECTION 8: COMBINED VARIANTS (best ideas layered)")
    print("=" * 130)

    # Combo 1: MQ + VolAdj
    def sig_mq_voladj(di, edi):
        c = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue
            atr = ATR14[s, di]; med_atr = ATR_MED60[s, di]
            if np.isnan(atr) or np.isnan(med_atr) or med_atr <= 0: continue
            if atr > 1.5 * med_atr: continue
            br = BODY_RATIO[s, di]; vr = VOL_RATIO[s, di]
            if np.isnan(br): br = 0.5
            if np.isnan(vr): vr = 1.0
            quality = roc / 100.0 * (1 + br) * (1 + vr) * 100
            if quality < 2.0: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((quality * zs, s, ep, 'mq_va'))
        return c

    # Combo 2: Multi-TF + ADX
    def sig_mtf_adx(di, edi):
        c = []
        for s in range(NS):
            roc3 = ROC3[s, di]; roc5 = ROC5[s, di]
            roc10 = ROC10[s, di]; roc20 = ROC20[s, di]; adx = ADX14[s, di]
            if any(np.isnan(x) for x in [roc3, roc5, roc10, roc20, adx]): continue
            if roc3 <= 0 or roc5 <= 1.0 or roc10 <= 0 or roc20 <= 0: continue
            if adx < 20: continue
            zs = ZSCORE[s, di]
            if np.isnan(zs) or zs <= 1.0: continue
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            score = (roc3 + roc5 * 2 + roc10 + roc20) * zs
            sm = 0.60 if adx >= 25 else 0.45
            c.append((score, s, ep, 'mtf_adx', sm))
        return c

    # Combo 3: V121 + OI bonus + VolAdj
    def sig_v121_oi_vol(di, edi):
        c = []
        for s in range(NS):
            roc = ROC5[s, di]; zs = ZSCORE[s, di]
            if np.isnan(roc) or np.isnan(zs) or roc <= 1.0 or zs <= 1.5: continue
            rp = ROC5[s, di-1] if di > 0 else np.nan
            if not np.isnan(rp) and roc <= rp: continue
            atr = ATR14[s, di]; med_atr = ATR_MED60[s, di]
            if not np.isnan(atr) and not np.isnan(med_atr) and med_atr > 0:
                if atr > 1.5 * med_atr: continue
            base = roc * zs
            oi_pct = OI_PCTILE[s, di]
            if np.isnan(oi_pct): oi_pct = 0.5
            score = base * (1 + 0.3 * oi_pct)
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((score, s, ep, 'v121_oi_v'))
        return c

    # Combo 4: Multi-TF + OI bonus + VolAdj
    def sig_mtf_oi_vol(di, edi):
        c = []
        for s in range(NS):
            roc3 = ROC3[s, di]; roc5 = ROC5[s, di]
            roc10 = ROC10[s, di]; roc20 = ROC20[s, di]
            if any(np.isnan(x) for x in [roc3, roc5, roc10, roc20]): continue
            if roc3 <= 0 or roc5 <= 1.0 or roc10 <= 0 or roc20 <= 0: continue
            zs = ZSCORE[s, di]
            if np.isnan(zs) or zs <= 1.0: continue
            atr = ATR14[s, di]; med_atr = ATR_MED60[s, di]
            if not np.isnan(atr) and not np.isnan(med_atr) and med_atr > 0:
                if atr > 1.5 * med_atr: continue
            base = (roc3 + roc5 * 2 + roc10 + roc20) * zs
            oi_pct = OI_PCTILE[s, di]
            if np.isnan(oi_pct): oi_pct = 0.5
            score = base * (1 + 0.3 * oi_pct)
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((score, s, ep, 'mtf_oi_v'))
        return c

    # Combo 5: MQ + Multi-TF + ADX (kitchen sink without vol filter)
    def sig_mq_mtf_adx(di, edi):
        c = []
        for s in range(NS):
            roc3 = ROC3[s, di]; roc5 = ROC5[s, di]
            roc10 = ROC10[s, di]; roc20 = ROC20[s, di]; adx = ADX14[s, di]
            if any(np.isnan(x) for x in [roc3, roc5, roc10, roc20, adx]): continue
            if roc3 <= 0 or roc5 <= 1.0 or roc10 <= 0 or roc20 <= 0: continue
            if adx < 20: continue
            zs = ZSCORE[s, di]
            if np.isnan(zs) or zs <= 1.0: continue
            br = BODY_RATIO[s, di]; vr = VOL_RATIO[s, di]
            if np.isnan(br): br = 0.5
            if np.isnan(vr): vr = 1.0
            quality = roc5 / 100.0 * (1 + br) * (1 + vr) * 100
            if quality < 1.5: continue
            base = (roc3 + roc5 * 2 + roc10 + roc20) * zs * quality
            sm = 0.60 if adx >= 25 else 0.45
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((base, s, ep, 'mq_mtf_adx', sm))
        return c

    # Combo 6: Full kitchen sink (MQ + MTF + ADX + VolAdj + OI)
    def sig_kitchen_sink(di, edi):
        c = []
        for s in range(NS):
            roc3 = ROC3[s, di]; roc5 = ROC5[s, di]
            roc10 = ROC10[s, di]; roc20 = ROC20[s, di]; adx = ADX14[s, di]
            if any(np.isnan(x) for x in [roc3, roc5, roc10, roc20, adx]): continue
            if roc3 <= 0 or roc5 <= 1.0 or roc10 <= 0 or roc20 <= 0: continue
            if adx < 20: continue
            zs = ZSCORE[s, di]
            if np.isnan(zs) or zs <= 1.0: continue
            atr = ATR14[s, di]; med_atr = ATR_MED60[s, di]
            if not np.isnan(atr) and not np.isnan(med_atr) and med_atr > 0:
                if atr > 1.5 * med_atr: continue
            br = BODY_RATIO[s, di]; vr = VOL_RATIO[s, di]
            if np.isnan(br): br = 0.5
            if np.isnan(vr): vr = 1.0
            quality = roc5 / 100.0 * (1 + br) * (1 + vr) * 100
            if quality < 1.5: continue
            base = (roc3 + roc5 * 2 + roc10 + roc20) * zs * quality
            oi_pct = OI_PCTILE[s, di]
            if np.isnan(oi_pct): oi_pct = 0.5
            score = base * (1 + 0.3 * oi_pct)
            sm = 0.60 if adx >= 25 else 0.45
            ep = O[s, edi]
            if np.isnan(ep) or ep <= 0: continue
            c.append((score, s, ep, 'kitchen', sm))
        return c

    combo_configs = [
        (sig_mq_voladj, False, "Combo 1: MQ + VolAdj"),
        (sig_mtf_adx, True, "Combo 2: Multi-TF + ADX"),
        (sig_v121_oi_vol, False, "Combo 3: V121 + OI + VolAdj"),
        (sig_mtf_oi_vol, False, "Combo 4: Multi-TF + OI + VolAdj"),
        (sig_mq_mtf_adx, True, "Combo 5: MQ + Multi-TF + ADX"),
        (sig_kitchen_sink, True, "Combo 6: Kitchen Sink (all)"),
    ]

    combo_results = []
    for sig_func, adx_sz, label in combo_configs:
        r = backtest(sig_func, size_frac=0.50, adx_sizing=adx_sz)
        r['desc'] = label; combo_results.append(r)
        pr(r, label)

    # ===================== SECTION 9: ALL PORTFOLIO COMBOS =====================
    print("\n" + "=" * 130)
    print("  SECTION 9: PORTFOLIO COMBOS (each variant / V121 50/50)")
    print("=" * 130)

    port_configs = [
        (sig_momentum_quality, False, "MQ(2.0)/V121 50/50"),
        (sig_multi_tf, False, "MultiTF/V121 50/50"),
        (sig_ov_id_tight, False, "OVTight/V121 50/50"),
        (sig_oi_boosted, False, "OIBosted/V121 50/50"),
        (sig_vol_adj, False, "VolAdj(1.5)/V121 50/50"),
        (sig_adx_gate, True, "ADX/V121 50/50"),
        (sig_mq_voladj, False, "MQ+VolAdj/V121 50/50"),
        (sig_mtf_adx, True, "MTF+ADX/V121 50/50"),
        (sig_v121_oi_vol, False, "V121+OI+Vol/V121 50/50"),
        (sig_mtf_oi_vol, False, "MTF+OI+Vol/V121 50/50"),
        (sig_mq_mtf_adx, True, "MQ+MTF+ADX/V121 50/50"),
        (sig_kitchen_sink, True, "Kitchen/V121 50/50"),
    ]

    port_results = []
    for sig_func, adx_sz, label in port_configs:
        r = backtest_portfolio(sig_func, sig_v121, size_frac_A=0.50, size_frac_B=0.50,
                               adx_sizing_A=adx_sz, adx_sizing_B=False)
        r['desc'] = label; port_results.append(r)
        pr_short(r, label)

    # ===================== SECTION 10: WALK-FORWARD FOR BEST VARIANTS =====================
    print("\n" + "=" * 130)
    print("  SECTION 10: WALK-FORWARD VALIDATION (per-year with MDD)")
    print("=" * 130)

    # Collect all standalone results
    all_standalone = [r_v121, r_union] + results_a + [r_mtf, r_ovtight, r_oi] + results_e
    all_standalone += [r_adx, r_adx_fixed] + combo_results
    reasonable = [r for r in all_standalone if r['ann'] > 0 and r.get('desc', '')]
    reasonable.sort(key=lambda x: -x['ann'])

    print(f"\n  --- Top 15 standalone by return ---")
    for i, r in enumerate(reasonable[:15]):
        desc = r.get('desc', '')
        print(f"  #{i+1}: {desc:72s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | "
              f"Sh={r['sharpe']:4.2f} | WR={r['wr']:5.1f}% | W/L={r['wl_ratio']:4.2f}")

    # Best standalone by Return/MDD ratio
    by_ratio = [(r, abs(r['ann']/r['mdd']) if r['mdd'] != 0 else 0)
                for r in all_standalone if r.get('desc', '') and r['mdd'] < 0]
    by_ratio.sort(key=lambda x: -x[1])
    print(f"\n  --- Top 10 standalone by Ann/MDD ratio ---")
    for i, (r, ratio) in enumerate(by_ratio[:10]):
        desc = r.get('desc', '')
        print(f"  #{i+1}: {desc:72s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Ratio={ratio:.2f}")

    # WF for top standalone signals
    print(f"\n  --- Walk-Forward: Standalone ---")
    wf_targets = [
        (sig_v121, False, "V121 baseline"),
        (sig_union, False, "Union baseline"),
        (sig_momentum_quality, False, "MQ thresh=2.0"),
        (sig_multi_tf, False, "Multi-TF"),
        (sig_ov_id_tight, False, "OV/ID tight"),
        (sig_oi_boosted, False, "OI boosted"),
        (sig_vol_adj, False, "VolAdj 1.5x"),
        (sig_adx_gate, True, "ADX gate"),
        (sig_mq_voladj, False, "MQ+VolAdj"),
        (sig_mtf_adx, True, "MTF+ADX"),
        (sig_kitchen_sink, True, "Kitchen Sink"),
    ]
    for sig_func, adx_sz, label in wf_targets:
        w = wf(sig_func, size_frac=0.50, adx_sizing=adx_sz)
        print_wf(label, w)

    # WF for top portfolios
    print(f"\n  --- Walk-Forward: Portfolios ---")
    port_wf_targets = [
        (sig_momentum_quality, False, "MQ/V121 50/50"),
        (sig_multi_tf, False, "MTF/V121 50/50"),
        (sig_oi_boosted, False, "OI/V121 50/50"),
        (sig_adx_gate, True, "ADX/V121 50/50"),
        (sig_mtf_adx, True, "MTF+ADX/V121 50/50"),
        (sig_kitchen_sink, True, "Kitchen/V121 50/50"),
    ]
    for sig_func, adx_sz, label in port_wf_targets:
        w = wf_portfolio(sig_func, sig_v121, size_frac=0.50, adx_sizing_A=adx_sz, adx_sizing_B=False)
        print_wf(label, w)

    # ===================== SECTION 11: PER-YEAR MDD ANALYSIS =====================
    print("\n" + "=" * 130)
    print("  SECTION 11: PER-YEAR MDD ANALYSIS (best variants)")
    print("=" * 130)

    def wf_mdd(signal_func, adx_sizing=False):
        res = {}
        for yr in [2020, 2021, 2022, 2023, 2024, 2025]:
            ys = ye = None
            for di in range(ND):
                if dates[di].year == yr and ys is None: ys = di
                if dates[di].year == yr: ye = di + 1
            if ys is None: continue
            r = backtest(signal_func, start_di=ys, end_di=ye, size_frac=0.50, adx_sizing=adx_sizing)
            res[yr] = (r['ann'], r['mdd'])
        return res

    mdd_targets = [
        (sig_v121, False, "V121"),
        (sig_union, False, "Union"),
        (sig_multi_tf, False, "Multi-TF"),
        (sig_adx_gate, True, "ADX gate"),
        (sig_kitchen_sink, True, "Kitchen Sink"),
        (sig_mtf_adx, True, "MTF+ADX"),
    ]
    for sig_func, adx_sz, label in mdd_targets:
        wf_r = wf_mdd(sig_func, adx_sizing=adx_sz)
        ws = " | ".join([f"{yr}:A={a:+.0f}%,M={m:.0f}%"
                         for yr, (a, m) in sorted(wf_r.items())])
        print(f"  {label:20s} | {ws}")

    # ===================== COMPREHENSIVE SUMMARY =====================
    print("\n" + "=" * 130)
    print("  COMPREHENSIVE SUMMARY")
    print("=" * 130)

    print(f"\n  --- Baselines ---")
    pr(r_v121, "V121 baseline (50%)")
    pr(r_union, "Union baseline (50%)")

    print(f"\n  --- Best Standalone by Return (top 5) ---")
    for i, r in enumerate(reasonable[:5]):
        desc = r.get('desc', '')
        print(f"  #{i+1}: {desc:72s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | "
              f"Sh={r['sharpe']:4.2f} | WR={r['wr']:5.1f}% | W/L={r['wl_ratio']:4.2f}")

    print(f"\n  --- Best Portfolio by Return (top 5) ---")
    port_sorted = sorted([r for r in port_results if r.get('desc', '')], key=lambda x: -x['ann'])
    for i, r in enumerate(port_sorted[:5]):
        desc = r.get('desc', '')
        print(f"  #{i+1}: {desc:72s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Sh={r['sharpe']:4.2f}")

    print(f"\n  --- Best by Ann/MDD ratio (standalone, top 5) ---")
    for i, (r, ratio) in enumerate(by_ratio[:5]):
        desc = r.get('desc', '')
        print(f"  #{i+1}: {desc:72s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Ratio={ratio:.2f}")

    port_by_ratio = [(r, abs(r['ann']/r['mdd']) if r['mdd'] != 0 else 0)
                     for r in port_results if r.get('desc', '') and r['mdd'] < 0]
    port_by_ratio.sort(key=lambda x: -x[1])
    print(f"\n  --- Best Portfolio by Ann/MDD ratio (top 5) ---")
    for i, (r, ratio) in enumerate(port_by_ratio[:5]):
        desc = r.get('desc', '')
        print(f"  #{i+1}: {desc:72s} | Ann={r['ann']:+8.1f}% | MDD={r['mdd']:6.1f}% | Ratio={ratio:.2f}")

    print(f"\n  Elapsed: {time.time()-t_start:.0f}s")
    print("=" * 130)


if __name__ == '__main__':
    main()
