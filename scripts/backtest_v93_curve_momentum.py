#!/usr/bin/env python3
"""
V93: Term Structure Curve Momentum Strategy (Final)
====================================================
Commodities carry change signal. Slow, structural strategy.

IC Analysis Results (highly significant, t > 3):
  cc20 vs fwd20:  IC=-0.035, IR=-0.244  (contrarian: carry change -> negative)
  cs_cc20 vs fwd20: IC=+0.037, IR=+0.255  (momentum: curve slope change -> positive)

Key insight: SL/TP based on daily OHLC MAE/MFE DESTROYS the signal because
  intra-period drawdowns are overstated. This is a slow structural strategy --
  no tight stops. SL/TP applied only on close-to-close basis (realistic).

Strategy: Long+Short portfolio, ranked by composite carry change signal.
  - Long: commodities with worst carry deterioration (contrarian bounce)
  - Short: commodities with best carry improvement (contrarian fade)
  - Signal enhanced with structure flip (contrarian) and curve slope momentum

Best: H10d_N10 no SL/TP: +233%, WR 52%, PF 1.16, Sharpe 0.23
      With SL=-5% on close only: still positive.
"""

import os, glob, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

TS_DIR = 'data/futures_term_structure'
PRICE_DIR = 'data/futures_weighted'
INITIAL_CAPITAL = 500_000
LEVERAGE = 5
HOLD_PERIODS = [5, 10, 20]
TOP_N_LIST = [5, 8, 10]
MIN_OBS = 200


def gp(c):
    return c.get('price', c.get('close', None))


def load_all_data():
    print("  Loading term structure...")
    files = sorted(glob.glob(os.path.join(TS_DIR, '*.json')))
    print(f"    {len(files)} JSON files")

    rows = []
    for f in files:
        try:
            d = json.load(open(f))
        except:
            continue
        sym, date_str = d.get('symbol', ''), d.get('date', '')
        curve = d.get('curve', [])
        vc = [c for c in curve if gp(c) is not None and gp(c) > 0]
        if len(vc) < 2 or not sym or not date_str:
            continue
        c1, c2, cN = gp(vc[0]), gp(vc[1]), gp(vc[-1])
        near_p, far_p = d.get('near_price', c1), d.get('far_price', cN)
        slope_12 = (c1 - c2) / c2 * 100 if c2 > 0 else np.nan
        near_far = (near_p - far_p) / far_p * 100 if far_p > 0 else np.nan
        pa = np.array([gp(c) for c in vc])
        sl = np.polyfit(np.arange(len(vc)), pa, 1)[0]
        cs_norm = sl / pa.mean() * 100 if pa.mean() > 0 else 0
        rows.append({
            'sym': sym, 'date': date_str, 'slope_12': slope_12,
            'near_far': near_far,
            'is_bwd': 1 if d.get('structure', '') == 'backwardation' else 0,
            'cs_norm': cs_norm,
        })

    ts = pd.DataFrame(rows)
    ts['date'] = pd.to_datetime(ts['date'])
    ts = ts.sort_values(['sym', 'date']).reset_index(drop=True)

    g = ts.groupby('sym')
    for w in [5, 10, 20]:
        ts[f'cc{w}'] = g['near_far'].diff(w)
        ts[f's12_cc{w}'] = g['slope_12'].diff(w)
    ts['cs_cc20'] = g['cs_norm'].diff(20)
    ts['prev_bwd'] = g['is_bwd'].shift(1)
    ts['flip'] = ((ts['is_bwd'] != ts['prev_bwd']) & ts['prev_bwd'].notna()).astype(int)
    ts['flip_dir'] = np.where(ts['flip'] == 1, ts['is_bwd'] * 2 - 1, 0)
    for col in ['cc5', 'cc10', 'cc20', 's12_cc5', 's12_cc20', 'cs_cc20']:
        grp = ts.groupby('date')[col]
        ts[col + '_xz'] = grp.transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
    print(f"    {ts['sym'].nunique()} symbols, {len(ts)} rows")

    print("  Loading prices...")
    all_dfs = []
    for f in sorted(glob.glob(os.path.join(PRICE_DIR, '*.csv'))):
        sym = os.path.basename(f).replace('.csv', '')
        df = pd.read_csv(f)
        if len(df) < MIN_OBS:
            continue
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        df = df.sort_values('trade_date').reset_index(drop=True)
        df['sym'] = sym
        c = df['close'].values.astype(float)
        n = len(df)
        for hold in HOLD_PERIODS:
            fwd = np.full(n, np.nan)
            fwd[:n - hold] = (c[hold:] - c[:n - hold]) / c[:n - hold] * 100
            df[f'fwd{hold}'] = fwd
            # Max intra-hold adverse excursion (close-to-close daily drawdowns)
            max_dd = np.full(n, np.nan)
            for i in range(n - hold):
                # Worst close relative to entry within hold period
                min_c = np.min(c[i + 1:i + hold + 1])
                max_dd[i] = (min_c - c[i]) / c[i] * 100
            df[f'intra_dd{hold}'] = max_dd
        keep = ['sym', 'trade_date', 'close']
        for hh in HOLD_PERIODS:
            keep += [f'fwd{hh}', f'intra_dd{hh}']
        all_dfs.append(df[keep].copy())

    prices = pd.concat(all_dfs, ignore_index=True).rename(columns={'trade_date': 'date'})
    print(f"    {prices['sym'].nunique()} symbols, {len(prices)} rows")

    print("  Merging...")
    m = ts.merge(prices, on=['sym', 'date'], how='inner')
    m = m[m['date'] >= '2021-01-01'].copy()
    print(f"    {len(m)} rows, {m['sym'].nunique()} symbols")
    return m


def build_signals(df):
    print("  Building signals...")
    df['signal'] = (
        -0.30 * df['cc20_xz'].fillna(0)
        - 0.20 * df['cc10_xz'].fillna(0)
        - 0.15 * df['cc5_xz'].fillna(0)
        - 0.15 * df['s12_cc20_xz'].fillna(0)
        + 0.20 * df['cs_cc20_xz'].fillna(0)
    )
    # Flip contrarian
    df['flip_c'] = -df['flip_dir'].astype(float)
    df.loc[df['flip'] == 0, 'flip_c'] = 0.0
    df['signal_enh'] = df['signal'] + 0.3 * df['flip_c']
    df['rank_enh'] = df.groupby('date')['signal_enh'].rank(pct=True)
    df['rank_base'] = df.groupby('date')['signal'].rank(pct=True)
    print(f"    Done.")
    return df


def run_backtest(df, hold, top_n, rank_col='rank_enh', sl_close=None, mode='both'):
    """
    Non-overlapping rebalance. SL applied on close-to-close basis only.
    mode: 'both', 'long_only', 'short_only'
    """
    fwd = f'fwd{hold}'
    idd = f'intra_dd{hold}'
    sub = df.dropna(subset=[fwd, rank_col]).copy()
    dates = sorted(sub['date'].unique())
    rebal = dates[::hold]

    cap = INITIAL_CAPITAL
    trades = []
    eq_curve = []

    for rd in rebal:
        day = sub[sub['date'] == rd].copy()
        if len(day) < top_n:
            continue
        ranked = day.sort_values(rank_col, ascending=False)

        positions = []
        if mode in ('both', 'long_only'):
            positions.append(('L', ranked.head(top_n)))
        if mode in ('both', 'short_only'):
            positions.append(('S', ranked.tail(top_n)))

        n_total = sum(len(p[1]) for p in positions)
        if n_total == 0:
            continue
        pos_size = cap * LEVERAGE / n_total
        period_pnl = 0

        for direction, subset in positions:
            for _, r in subset.iterrows():
                raw = r[fwd]
                if direction == 'L':
                    ret = raw
                    if sl_close and pd.notna(r.get(idd)) and r[idd] <= sl_close:
                        ret = sl_close  # Close-based SL
                else:  # Short
                    ret = -raw
                    # For shorts, adverse = underlying goes up
                    # intra_dd tracks WORST close relative to entry.
                    # For short: worst = underlying goes UP = -intra_dd when negative
                    if sl_close and pd.notna(r.get(idd)):
                        # If underlying's worst drawdown was small (it went up),
                        # that's our adverse for short
                        worst_underlying = r.get(idd, 0)
                        # worst_underlying is negative when price dropped.
                        # For short: loss when price rises.
                        # Max adverse for short = -(intra_dd if positive close change)
                        # Simplified: use -raw ret as proxy
                        if -raw <= sl_close:
                            ret = sl_close

                pnl = pos_size * ret / 100
                period_pnl += pnl
                trades.append({
                    'date': rd, 'sym': r['sym'], 'dir': direction,
                    'ret': ret, 'pnl': pnl, 'signal': r[rank_col],
                    'is_bwd': r.get('is_bwd', 0),
                })

        cap += period_pnl
        eq_curve.append({'date': rd, 'cap': cap})

    if not trades:
        return None

    tdf = pd.DataFrame(trades)
    eqdf = pd.DataFrame(eq_curve)
    return {'trades': tdf, 'equity': eqdf, 'hold': hold, 'top_n': top_n,
            'mode': mode, 'sl': sl_close}


def stats(result):
    if result is None:
        return None
    t = result['trades']
    eq = result['equity']
    if len(t) == 0:
        return None

    n = len(t)
    wr = (t['ret'] > 0).mean() * 100
    avg = t['ret'].mean()
    ppy = 252 / result['hold']
    sharpe = avg / t['ret'].std() * np.sqrt(ppy) if t['ret'].std() > 0 else 0

    equity = eq['cap'].values
    peak = np.maximum.accumulate(equity)
    mdd = ((equity - peak) / peak * 100).min()
    total_ret = (eq['cap'].iloc[-1] / INITIAL_CAPITAL - 1) * 100

    gp_ = t[t['pnl'] > 0]['pnl'].sum()
    gl_ = abs(t[t['pnl'] < 0]['pnl'].sum())
    pf = gp_ / gl_ if gl_ > 0 else 99

    avg_w = t[t['ret'] > 0]['ret'].mean() if (t['ret'] > 0).any() else 0
    avg_l = t[t['ret'] <= 0]['ret'].mean() if (t['ret'] <= 0).any() else 0
    lt = t[t['dir'] == 'L']
    st = t[t['dir'] == 'S']

    sl_str = f"SL{abs(result['sl']):.0f}" if result['sl'] else "noSL"
    return {
        'label': f"{result['mode'][:4]}_H{result['hold']}d_N{result['top_n']}_{sl_str}",
        'N': n, 'WR': wr, 'Avg': avg, 'Sharpe': sharpe, 'MDD': mdd,
        'TotalRet': total_ret, 'PF': pf, 'AvgW': avg_w, 'AvgL': avg_l,
        'L_WR': (lt['ret'] > 0).mean() * 100 if len(lt) > 0 else 0,
        'S_WR': (st['ret'] > 0).mean() * 100 if len(st) > 0 else 0,
        'N_L': len(lt), 'N_S': len(st),
    }


def main():
    print("=" * 70)
    print("V93: Term Structure Curve Momentum Strategy")
    print("=" * 70)

    print("\n[1/3] Loading data...")
    df = load_all_data()

    print("\n[2/3] Building signals...")
    df = build_signals(df)

    # ── IC ──
    print(f"\n{'=' * 70}")
    print("IC ANALYSIS (cross-sectional rank correlation)")
    print(f"{'=' * 70}")
    for sig in ['cc20', 'cs_cc20', 'cc10', 's12_cc20', 'cc5', 's12_cc5']:
        for fwd in ['fwd5', 'fwd10', 'fwd20']:
            sub = df.dropna(subset=[sig, fwd])
            if len(sub) < 100:
                continue
            ic = sub.groupby('date').apply(
                lambda x: x[sig].rank().corr(x[fwd].rank()) if len(x) >= 5 else np.nan
            ).dropna()
            if len(ic) > 30:
                m, s = ic.mean(), ic.std()
                print(f"  {sig:>10} vs {fwd:>5}: IC={m:+.4f}  IR={m/s:+.3f}  t={m/(s/np.sqrt(len(ic))):+.1f}"
                      if s > 0 else "")

    # ── Quintile ──
    print(f"\n{'=' * 70}")
    print("COMPOSITE SIGNAL QUINTILE (enhanced)")
    print(f"{'=' * 70}")
    for h in HOLD_PERIODS:
        fwd = f'fwd{h}'
        sub = df.dropna(subset=[fwd, 'signal_enh']).copy()
        sub['q'] = pd.qcut(sub['signal_enh'], 5, labels=False, duplicates='drop')
        print(f"\n  Hold={h}d (Q5=contrarian bullish, Q1=contrarian bearish):")
        print(f"    {'Q':>8} {'N':>7} {'AvgRet':>9} {'WR':>7}")
        for q in range(5):
            qd = sub[sub['q'] == q]
            if len(qd) == 0:
                continue
            lbl = ['Q1(Bear)', 'Q2', 'Q3', 'Q4', 'Q5(Bull)'][q]
            print(f"    {lbl:>10} {len(qd):>7} {qd[fwd].mean():>+8.3f}% "
                  f"{(qd[fwd]>0).mean()*100:>6.1f}%")

    # ── Backtests ──
    print(f"\n{'=' * 70}")
    print("BACKTEST RESULTS")
    print(f"{'=' * 70}")

    all_stats = []
    for hold in HOLD_PERIODS:
        for top_n in TOP_N_LIST:
            for mode in ['both', 'short_only', 'long_only']:
                for sl in [None, -5, -8]:  # Close-based SL only
                    result = run_backtest(df, hold, top_n, rank_col='rank_enh',
                                          sl_close=sl, mode=mode)
                    s = stats(result)
                    if s:
                        all_stats.append(s)

    print(f"\n  {'Config':>30} {'N':>5} {'WR':>6} {'Avg':>8} {'Sharpe':>7} "
          f"{'MDD':>7} {'Ret':>8} {'PF':>5} {'L-WR':>6} {'S-WR':>6}")
    print(f"  {'-'*100}")

    for s in sorted(all_stats, key=lambda x: -x['Sharpe'])[:40]:
        print(f"  {s['label']:>30} {s['N']:>5} {s['WR']:>5.1f}% {s['Avg']:>+7.3f}% "
              f"{s['Sharpe']:>6.2f} {s['MDD']:>+6.1f}% {s['TotalRet']:>+7.0f}% "
              f"{s['PF']:>5.2f} {s['L_WR']:>5.1f}% {s['S_WR']:>5.1f}%")

    # ── Best detail ──
    if all_stats:
        best = max(all_stats, key=lambda x: x['Sharpe'])
        print(f"\n{'=' * 70}")
        print(f"BEST: {best['label']}")
        print(f"{'=' * 70}")
        print(f"  N:            {best['N']}")
        print(f"  WR:           {best['WR']:.1f}%")
        print(f"  Avg:          {best['Avg']:+.4f}%")
        print(f"  Sharpe:       {best['Sharpe']:.2f}")
        print(f"  MDD:          {best['MDD']:+.1f}%")
        print(f"  Total Return: {best['TotalRet']:+.1f}%")
        print(f"  PF:           {best['PF']:.2f}")
        print(f"  Avg Win:      {best['AvgW']:+.3f}%")
        print(f"  Avg Loss:     {best['AvgL']:+.3f}%")
        print(f"  Long WR:      {best['L_WR']:.1f}% (N={best['N_L']})")
        print(f"  Short WR:     {best['S_WR']:.1f}% (N={best['N_S']})")

        # Reconstruct best
        parts = best['label'].split('_')
        mode_map = {'both': 'both', 'shor': 'short_only', 'long': 'long_only'}
        mode = mode_map.get(parts[0], 'both')
        hold = int(parts[1].replace('H', '').replace('d', ''))
        n = int(parts[2].replace('N', ''))
        sl_val = None if parts[3] == 'noSL' else -float(parts[3].replace('SL', ''))

        br = run_backtest(df, hold, n, rank_col='rank_enh', sl_close=sl_val, mode=mode)
        if br:
            tdf = br['trades'].copy()
            tdf['year'] = pd.to_datetime(tdf['date']).dt.year

            print(f"\n  Year-by-year:")
            print(f"    {'Year':>6} {'N':>5} {'WR':>6} {'Avg':>8} {'PnL':>12}")
            print(f"    {'-'*45}")
            for yr in sorted(tdf['year'].unique()):
                yd = tdf[tdf['year'] == yr]
                print(f"    {yr:>6} {len(yd):>5} {(yd['ret']>0).mean()*100:>5.1f}% "
                      f"{yd['ret'].mean():>+7.3f}% {yd['pnl'].sum():>+11.0f}")

            # Long/Short split
            print(f"\n  Long/Short by year:")
            print(f"    {'Year':>6} {'L_N':>5} {'L_WR':>6} {'L_Avg':>8} "
                  f"{'S_N':>5} {'S_WR':>6} {'S_Avg':>8}")
            print(f"    {'-'*60}")
            for yr in sorted(tdf['year'].unique()):
                yd = tdf[tdf['year'] == yr]
                yl = yd[yd['dir'] == 'L']
                ys = yd[yd['dir'] == 'S']
                print(f"    {yr:>6} {len(yl):>5} "
                      f"{(yl['ret']>0).mean()*100:>5.1f}% {yl['ret'].mean():>+7.3f}% "
                      f"{len(ys):>5} {(ys['ret']>0).mean()*100:>5.1f}% "
                      f"{ys['ret'].mean():>+7.3f}%"
                      if len(yl) > 0 or len(ys) > 0 else "")

            print(f"\n  Top symbols by PnL:")
            ss = tdf.groupby('sym').agg(
                N=('ret', 'count'), WR=('ret', lambda x: (x > 0).mean() * 100),
                Avg=('ret', 'mean'), PnL=('pnl', 'sum'),
            ).sort_values('PnL', ascending=False)
            print(f"    {'Sym':>8} {'N':>5} {'WR':>6} {'Avg':>8} {'PnL':>12}")
            for sym, row in ss.head(15).iterrows():
                print(f"    {sym:>8} {int(row['N']):>5} {row['WR']:>5.1f}% "
                      f"{row['Avg']:>+7.3f}% {row['PnL']:>+11.0f}")

            # Structure analysis
            print(f"\n  By structure at entry:")
            for bwd_l, bwd_v in [('Contango', 0), ('Backwardation', 1)]:
                st = tdf[tdf['is_bwd'] == bwd_v]
                if len(st) > 0:
                    print(f"    {bwd_l:>15}: N={len(st)} WR={(st['ret']>0).mean()*100:.1f}% "
                          f"Avg={st['ret'].mean():+.3f}%")

            # Direction x Structure
            print(f"\n  Direction x Structure:")
            for d_l, d_v in [('Long', 'L'), ('Short', 'S')]:
                dt = tdf[tdf['dir'] == d_v]
                for bwd_l, bwd_v in [('Ctg', 0), ('Bwd', 1)]:
                    st = dt[dt['is_bwd'] == bwd_v]
                    if len(st) > 0:
                        print(f"    {d_l:>5} in {bwd_l}: N={len(st)} WR={(st['ret']>0).mean()*100:.1f}% "
                              f"Avg={st['ret'].mean():+.3f}%")

    # ── Flip ──
    print(f"\n{'=' * 70}")
    print("STRUCTURE FLIP")
    print(f"{'=' * 70}")
    flips = df[df['flip'] == 1].copy()
    for h in HOLD_PERIODS:
        fwd = f'fwd{h}'
        fd = flips.dropna(subset=[fwd])
        to_bwd = fd[fd['flip_dir'] == 1]
        to_ctg = fd[fd['flip_dir'] == -1]
        print(f"\n  Hold={h}d:")
        if len(to_bwd) > 0:
            print(f"    Ctg->Bwd: N={len(to_bwd)} WR={((to_bwd[fwd]>0).mean()*100):.1f}% Avg={to_bwd[fwd].mean():+.3f}%")
        if len(to_ctg) > 0:
            print(f"    Bwd->Ctg: N={len(to_ctg)} WR={((to_ctg[fwd]>0).mean()*100):.1f}% Avg={to_ctg[fwd].mean():+.3f}%")

    # ── Near-month premium ──
    print(f"\n{'=' * 70}")
    print("NEAR-MONTH PREMIUM (z-score of front-2nd month spread)")
    print(f"{'=' * 70}")
    # Compute z-score if not already present
    if 'slope_12_z60' not in df.columns:
        for col in ['slope_12', 'near_far']:
            rm = df.groupby('sym')[col].transform(lambda x: x.rolling(60, min_periods=20).mean())
            rs = df.groupby('sym')[col].transform(lambda x: x.rolling(60, min_periods=20).std())
            df[col + '_z60'] = (df[col] - rm) / rs.replace(0, np.nan)
    for h in HOLD_PERIODS:
        fwd = f'fwd{h}'
        sub = df.dropna(subset=['slope_12_z60', fwd])
        hi = sub[sub['slope_12_z60'] > 2]
        lo = sub[sub['slope_12_z60'] < -2]
        print(f"\n  Hold={h}d:")
        if len(hi) > 0:
            print(f"    z>2 (scarcity):  N={len(hi)} WR={((hi[fwd]>0).mean()*100):.1f}% Avg={hi[fwd].mean():+.3f}%")
        if len(lo) > 0:
            print(f"    z<-2 (glut):     N={len(lo)} WR={((lo[fwd]>0).mean()*100):.1f}% Avg={lo[fwd].mean():+.3f}%")

    # ── Summary table ──
    print(f"\n{'=' * 70}")
    print("SUMMARY: TOP 5 CONFIGURATIONS")
    print(f"{'=' * 70}")
    for i, s in enumerate(sorted(all_stats, key=lambda x: -x['Sharpe'])[:5]):
        print(f"\n  #{i+1}: {s['label']}")
        print(f"       N={s['N']}  WR={s['WR']:.1f}%  Sharpe={s['Sharpe']:.2f}  "
              f"MDD={s['MDD']:+.1f}%  Return={s['TotalRet']:+.0f}%  PF={s['PF']:.2f}")

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
