"""
Alpha Futures V26 — Pure Time Exit (无杠杆, 纯日线)
====================================================
核心发现: 所有策略中, 只有时间退出是稳定盈利的
  - v14b: 时间退出 81% WR, 止损 0% WR, 轮换亏损
  - v23: 时间退出 71% WR, 止损 0% WR
  - v25: 轮换全部亏损

策略: 简单到极致
  1. 每天/每N天, 用VDP+MOM评分选最强品种
  2. 全仓买入, 持有N天
  3. N天后卖出, 重新选最强品种
  4. 无止损, 无止盈, 无轮换, 无trailing stop
  5. 仅在极端情况(跌>10%)时止损

数学:
  - 3天持有: 83 trades/year, 需要 (1+r)^83 = 7, r = 2.36%
  - 如果WR=65%, avg_win=4%, avg_loss=1.5%:
    net = 0.65*4 - 0.35*1.5 = 2.075% → 接近2.36%目标

约束: 不做gap, 不做日内, 无杠杆
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpha_v2 import load_all_data, MIN_TRAIN, CASH0, compute_vdp

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


def main():
    t_start = time.time()
    print("=" * 120)
    print("Alpha Futures V26 — Pure Time Exit Strategy")
    print("=" * 120)

    NS, ND, dates, C, O, H, L, V, OI, syms, sym_set = load_all_data(
        max_stocks=500, load_oi=True
    )

    # Precompute signals for all stocks
    print("[Signals] Precomputing...", flush=True)
    t0 = time.time()

    # Store precomputed signals per stock
    signals = {}
    for si in range(NS):
        c, h, l, v, oi = C[si], H[si], L[si], V[si], OI[si]
        if np.sum(~np.isnan(c)) < 60: continue

        # Momentums
        mom3 = np.full(ND, np.nan); mom5 = np.full(ND, np.nan)
        mom10 = np.full(ND, np.nan); mom20 = np.full(ND, np.nan)
        for i in range(3, ND):
            if not np.isnan(c[i]) and not np.isnan(c[i-3]):
                mom3[i] = (c[i] - c[i-3]) / c[i-3]
        for i in range(5, ND):
            if not np.isnan(c[i]) and not np.isnan(c[i-5]):
                mom5[i] = (c[i] - c[i-5]) / c[i-5]
        for i in range(10, ND):
            if not np.isnan(c[i]) and not np.isnan(c[i-10]):
                mom10[i] = (c[i] - c[i-10]) / c[i-10]
        for i in range(20, ND):
            if not np.isnan(c[i]) and not np.isnan(c[i-20]):
                mom20[i] = (c[i] - c[i-20]) / c[i-20]

        # VDP + EMA
        vdp = compute_vdp(c, h, l, v)
        vdp_ema = np.full(ND, np.nan)
        vdp_ema[0] = vdp[0]
        a = 2.0 / 16
        for i in range(1, ND):
            if not np.isnan(vdp[i]):
                vdp_ema[i] = a * vdp[i] + (1-a) * (vdp_ema[i-1] if not np.isnan(vdp_ema[i-1]) else vdp[i])

        # OI momentum
        oi_mom5 = np.full(ND, np.nan)
        for i in range(5, ND):
            if not np.isnan(oi[i]) and oi[i-5] > 0 and not np.isnan(oi[i-5]):
                oi_mom5[i] = (oi[i] - oi[i-5]) / oi[i-5]

        # EMAs
        ema50 = np.full(ND, np.nan)
        al = 2.0 / 51
        start = None
        for i in range(ND):
            if not np.isnan(c[i]):
                if start is None: ema50[i] = c[i]; start = i
                else: ema50[i] = al * c[i] + (1-al) * ema50[i-1]

        sma200 = np.full(ND, np.nan)
        for i in range(199, ND):
            w = c[i-199:i+1]; v2 = w[~np.isnan(w)]
            if len(v2) >= 100: sma200[i] = np.mean(v2)

        signals[si] = {
            'mom3': mom3, 'mom5': mom5, 'mom10': mom10, 'mom20': mom20,
            'vdp_ema': vdp_ema, 'oi_mom5': oi_mom5,
            'ema50': ema50, 'sma200': sma200,
        }

    print(f"  Done in {time.time()-t0:.1f}s, {len(signals)} stocks", flush=True)

    # === SCORING FUNCTIONS ===
    def score(si, di, mode='vdp_mom5'):
        """Score a single stock on a single day."""
        if si not in signals: return np.nan
        s = signals[si]
        c = C[si, di]
        if np.isnan(c) or c <= 0: return np.nan

        if mode == 'mom5':
            m = s['mom5'][di]
            return m if not np.isnan(m) and m > 0 else np.nan

        elif mode == 'mom3':
            m = s['mom3'][di]
            return m if not np.isnan(m) and m > 0 else np.nan

        elif mode == 'mom10':
            m = s['mom10'][di]
            return m if not np.isnan(m) and m > 0 else np.nan

        elif mode == 'vdp_mom5':
            m = s['mom5'][di]
            if np.isnan(m) or m <= 0: return np.nan
            sc = m
            vdp = s['vdp_ema'][di]
            if not np.isnan(vdp):
                sc *= (1.5 if vdp > 0 else 0.5)
            oi_m = s['oi_mom5'][di]
            if not np.isnan(oi_m):
                sc *= (1.3 if oi_m > 0 else 0.7)
            return sc

        elif mode == 'vdp_mom3':
            m = s['mom3'][di]
            if np.isnan(m) or m <= 0: return np.nan
            sc = m
            vdp = s['vdp_ema'][di]
            if not np.isnan(vdp):
                sc *= (1.5 if vdp > 0 else 0.5)
            oi_m = s['oi_mom5'][di]
            if not np.isnan(oi_m):
                sc *= (1.3 if oi_m > 0 else 0.7)
            return sc

        elif mode == 'vdp_mom10':
            m = s['mom10'][di]
            if np.isnan(m) or m <= 0: return np.nan
            sc = m
            vdp = s['vdp_ema'][di]
            if not np.isnan(vdp):
                sc *= (1.5 if vdp > 0 else 0.5)
            oi_m = s['oi_mom5'][di]
            if not np.isnan(oi_m):
                sc *= (1.3 if oi_m > 0 else 0.7)
            return sc

        elif mode == 'vdp_mom5_trend':
            m = s['mom5'][di]
            if np.isnan(m) or m <= 0: return np.nan
            sma200 = s['sma200'][di]
            if not np.isnan(sma200) and c < sma200: return np.nan  # hard filter
            sc = m
            vdp = s['vdp_ema'][di]
            if not np.isnan(vdp):
                sc *= (1.5 if vdp > 0 else 0.5)
            oi_m = s['oi_mom5'][di]
            if not np.isnan(oi_m):
                sc *= (1.3 if oi_m > 0 else 0.7)
            return sc

        elif mode == 'vdp_mom3_trend':
            m = s['mom3'][di]
            if np.isnan(m) or m <= 0: return np.nan
            sma200 = s['sma200'][di]
            if not np.isnan(sma200) and c < sma200: return np.nan
            sc = m
            vdp = s['vdp_ema'][di]
            if not np.isnan(vdp):
                sc *= (1.5 if vdp > 0 else 0.5)
            oi_m = s['oi_mom5'][di]
            if not np.isnan(oi_m):
                sc *= (1.3 if oi_m > 0 else 0.7)
            return sc

        return np.nan

    # === BACKTEST: PURE TIME EXIT ===
    def run_pure_time(score_mode, hold_days, rebalance='fixed'):
        """Pure time exit strategy.

        rebalance modes:
        - 'fixed': enter, hold exactly N days, exit
        - 'rolling': every day, if no position, enter top-ranked. Hold for N days.
        """
        cash = float(CASH0)
        trades = []
        pos = None
        next_exit_di = -1

        for di in range(MIN_TRAIN, ND):
            year = dates[di].year

            # EXIT: if holding and reached exit day
            if pos is not None and di >= next_exit_di:
                c = C[pos['si'], di]
                if np.isnan(c) or c <= 0: c = pos['entry']
                mult = MULT.get(pos['sym'], DEF_MULT)
                mkt_val = c * mult * pos['lots']
                pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
                pnl_pct = pnl / (pos['entry'] * mult * pos['lots']) * 100 if pos['entry'] > 0 else 0

                # Safety stop: if loss > 15%, exit
                if pnl_pct / 100 < -0.15:
                    cost_out = mkt_val * COMM
                    cash += mkt_val - cost_out
                    trades.append({
                        'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                        'days': di - pos['entry_di'], 'di': di, 'year': year,
                        'sym': pos['sym'], 'dir': pos['dir'], 'reason': 'safety_stop',
                    })
                    pos = None
                    next_exit_di = -1
                    continue

                cost_out = mkt_val * COMM
                cash += mkt_val - cost_out
                trades.append({
                    'pnl_pct': pnl_pct, 'pnl_abs': pnl,
                    'days': di - pos['entry_di'], 'di': di, 'year': year,
                    'sym': pos['sym'], 'dir': pos['dir'], 'reason': 'time',
                })
                pos = None
                next_exit_di = -1

            # ENTRY: if no position, enter top-ranked
            if pos is None:
                best_si, best_sc = -1, 0
                for si in range(NS):
                    sc = score(si, di, score_mode)
                    if np.isnan(sc): continue
                    if sc > best_sc:
                        best_sc = sc; best_si = si

                if best_si >= 0 and best_sc > 0:
                    c = C[best_si, di]
                    if np.isnan(c) or c <= 0: continue

                    sym = syms[best_si]
                    mult = MULT.get(sym, DEF_MULT)
                    notional = c * mult
                    if notional <= 0: continue

                    lots = int(cash / notional)
                    if lots <= 0: continue

                    cost_in = notional * lots * (1 + COMM)
                    if cost_in > cash: continue

                    cash -= cost_in
                    pos = {
                        'si': best_si, 'entry': c, 'entry_di': di,
                        'lots': lots, 'dir': 1, 'sym': sym,
                    }
                    next_exit_di = di + hold_days

        # Close remaining
        if pos is not None:
            c = C[pos['si'], ND-1]
            if np.isnan(c) or c <= 0: c = pos['entry']
            mult = MULT.get(pos['sym'], DEF_MULT)
            pnl = (c - pos['entry']) * mult * pos['lots'] * pos['dir']
            cash += c * mult * pos['lots'] * (1 - COMM)
            trades.append({
                'pnl_pct': pnl / (pos['entry'] * mult * pos['lots']) * 100,
                'pnl_abs': pnl, 'days': ND-1 - pos['entry_di'],
                'di': ND-1, 'year': dates[ND-1].year,
                'sym': pos['sym'], 'dir': pos['dir'], 'reason': 'end',
            })

        if len(trades) < 20:
            return None

        # Stats
        equity = float(CASH0); peak = float(CASH0); max_dd = 0
        for t in sorted(trades, key=lambda x: x['di']):
            equity += t['pnl_abs']
            if equity > peak: peak = equity
            if peak > 0:
                dd = (peak - equity) / peak * 100
                if dd > max_dd: max_dd = dd

        days_total = (dates[ND-1] - dates[MIN_TRAIN]).days
        yr = max(days_total / 365.25, 0.01)
        ann = ((cash / CASH0) ** (1 / yr) - 1) * 100

        nw = sum(1 for t in trades if t['pnl_abs'] > 0)
        wr = nw / len(trades) * 100
        avg_pnl = np.mean([t['pnl_pct'] for t in trades])
        avg_days = np.mean([t['days'] for t in trades])
        avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_abs'] > 0]) if nw > 0 else 0
        avg_loss = np.mean([abs(t['pnl_pct']) for t in trades if t['pnl_abs'] <= 0]) if nw < len(trades) else 0

        year_stats = {}
        for t in trades:
            y = t['year']
            if y not in year_stats:
                year_stats[y] = {'n': 0, 'w': 0, 'pnl': 0.0}
            year_stats[y]['n'] += 1
            if t['pnl_abs'] > 0: year_stats[y]['w'] += 1
            year_stats[y]['pnl'] += t['pnl_pct']

        return {
            'name': f"{score_mode}_H{hold_days}",
            'ann': round(ann, 1), 'n': len(trades),
            'wr': round(wr, 1), 'dd': round(max_dd, 1),
            'avg_pnl': round(avg_pnl, 3), 'avg_days': round(avg_days, 1),
            'avg_win': round(avg_win, 3), 'avg_loss': round(avg_loss, 3),
            'cash': round(cash, 0), 'yearly': year_stats,
            'wlr': round(avg_win / max(avg_loss, 0.01), 2),
        }

    # === RUN ALL CONFIGS ===
    results = []

    modes = ['mom3', 'mom5', 'mom10',
             'vdp_mom3', 'vdp_mom5', 'vdp_mom10',
             'vdp_mom3_trend', 'vdp_mom5_trend']
    hold_days_list = [2, 3, 4, 5, 7, 10, 15]

    for mode in modes:
        for hd in hold_days_list:
            r = run_pure_time(mode, hd)
            if r:
                results.append(r)
                print(f"  {r['name']:30s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                      f"N {r['n']:4d} | DD {r['dd']:6.1f}% | AvgPnl {r['avg_pnl']:+.3f}% | "
                      f"AvgD {r['avg_days']:.1f} | W/L {r['avg_win']:.2f}/{r['avg_loss']:.2f} | "
                      f"WLR {r['wlr']:.2f}")

    # === SUMMARY ===
    print(f"\n{'='*120}")
    print(f"TOTAL: {len(results)} configs")
    print(f"{'='*120}")

    if results:
        results.sort(key=lambda x: -x['ann'])

        print(f"\nTOP 20 BY ANNUAL RETURN:")
        for r in results[:20]:
            print(f"  {r['name']:30s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"N {r['n']:4d} | DD {r['dd']:6.1f}% | WLR {r['wlr']:.2f}")

        print(f"\n--- TOP 5 BY WIN RATE ---")
        by_wr = sorted(results, key=lambda x: -x['wr'])
        for r in by_wr[:5]:
            print(f"  {r['name']:30s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"N {r['n']:4d} | DD {r['dd']:6.1f}%")

        print(f"\n--- YEARLY BREAKDOWN (Top 5) ---")
        for r in results[:5]:
            print(f"\n  {r['name']}:")
            for y in sorted(r['yearly'].keys()):
                ys = r['yearly'][y]
                wr = ys['w'] / ys['n'] * 100 if ys['n'] > 0 else 0
                print(f"    {y}: {ys['n']:3d} trades, WR {wr:5.1f}%, PnL {ys['pnl']:+.1f}%")

        # Hold period analysis for best scoring mode
        print(f"\n--- HOLD PERIOD ANALYSIS (vdp_mom5) ---")
        for r in sorted([r for r in results if 'vdp_mom5' in r['name'] and 'trend' not in r['name']],
                        key=lambda x: int(x['name'].split('_H')[1])):
            print(f"  {r['name']:30s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"DD {r['dd']:6.1f}% | AvgPnl {r['avg_pnl']:+.3f}%")

        # Scoring mode comparison at optimal hold
        print(f"\n--- SCORING MODE COMPARISON (H=2) ---")
        for r in sorted([r for r in results if r['name'].endswith('_H2')],
                        key=lambda x: -x['ann']):
            print(f"  {r['name']:30s} | Ann {r['ann']:+7.1f}% | WR {r['wr']:5.1f}% | "
                  f"DD {r['dd']:6.1f}% | AvgW {r['avg_win']:.3f}% | AvgL {r['avg_loss']:.3f}%")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
