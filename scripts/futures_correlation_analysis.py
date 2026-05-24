#!/usr/bin/env python3
"""
Futures Correlation Analysis System v2
=========================================
Finds pair trading opportunities via rolling correlation analysis,
correlation break detection, and convergence backtesting.

Key design:
- Uses a 3-year lookback window for pair selection (more relevant regime)
- Computes rolling 60-day pairwise Pearson correlations
- Detects correlation breaks via z-score vs 1-year rolling mean
- Backtests convergence trades on the price spread (ratio-based)
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')

# ─── Configuration ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'futures_weighted'

ROLLING_WINDOW = 60       # 60-day rolling correlation
LOOKBACK_1Y = 250         # ~1 year of trading days for rolling mean/std
CORR_BREAK_ZSCORE = -2.0  # z-score threshold for correlation break
HOLD_DAYS = 7             # hold period for backtest
MIN_DATA_DAYS = 500       # minimum data days to include a symbol
TOP_N = 20                # top N pairs to display
RECENT_YEARS = 3          # years of recent data for pair selection

# Sector mapping
SECTOR_MAP = {
    'agfi': 'Metals', 'alfi': 'Metals', 'aufi': 'Metals', 'bufi': 'Metals',
    'cufi': 'Metals', 'nifi': 'Metals', 'pbfi': 'Metals', 'snfi': 'Metals',
    'znfi': 'Metals', 'ssfi': 'Metals', 'bcfi': 'Metals',
    'rbfi': 'Ferrous', 'hcfi': 'Ferrous', 'ifi': 'Ferrous',
    'jfi': 'Ferrous', 'jmfi': 'Ferrous', 'sffi': 'Ferrous', 'smfi': 'Ferrous',
    'scfi': 'Energy', 'fufi': 'Energy', 'pgfi': 'Energy',
    'lufi': 'Energy', 'nrfi': 'Energy', 'brfi': 'Energy', 'tafi': 'Energy',
    'ebfi': 'Chemical', 'egfi': 'Chemical', 'lfi': 'Chemical',
    'ppfi': 'Chemical', 'vfi': 'Chemical', 'mafi': 'Chemical',
    'fgfi': 'Chemical', 'safi': 'Chemical', 'urfi': 'Chemical',
    'afi': 'Agri', 'bfi': 'Agri', 'cfi': 'Agri', 'csfi': 'Agri',
    'mfi': 'Agri', 'yfi': 'Agri', 'pfi': 'Agri', 'oifi': 'Agri',
    'rmfi': 'Agri', 'srfi': 'Agri', 'whfi': 'Agri', 'pmfi': 'Agri',
    'rrfi': 'Agri', 'rsfi': 'Agri', 'jrfi': 'Agri', 'lrfi': 'Agri',
    'cjfi': 'Agri', 'apfi': 'Agri', 'jdfi': 'Agri', 'pkfi': 'Agri',
    'cyfi': 'Agri', 'cffi': 'Agri', 'lhfi': 'Agri',
    'spfi': 'Softs', 'rufi': 'Softs', 'lcfi': 'Softs', 'sifi': 'Softs',
    'bbfi': 'Softs', 'fbfi': 'Softs', 'rifi': 'Softs', 'lgfi': 'Softs',
    'adfi': 'Softs', 'aofi': 'Softs', 'plfi': 'Softs', 'ptfi': 'Softs',
    'opfi': 'Softs', 'psfi': 'Softs', 'prfi': 'Softs', 'pxfi': 'Softs',
    'shfi': 'Softs', 'ecfi': 'Softs', 'wrfi': 'Softs',
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load Data & Compute Log Returns
# ══════════════════════════════════════════════════════════════════════════════

def load_all_futures(data_dir: Path, min_days: int = MIN_DATA_DAYS):
    print("=" * 90)
    print(" STEP 1: Loading futures data and computing log returns")
    print("=" * 90)

    all_returns = {}
    price_data = {}
    valid_symbols = []

    for csv_path in sorted(data_dir.glob('*.csv')):
        symbol = csv_path.stem
        df = pd.read_csv(csv_path)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed', errors='coerce')
        df = df.dropna(subset=['trade_date']).sort_values('trade_date').reset_index(drop=True)
        df = df[(df['close'] > 0) & (df['vol'] > 0)]

        if len(df) < min_days:
            continue

        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df = df.dropna(subset=['log_ret'])
        df = df.drop_duplicates(subset='trade_date', keep='last')

        s_ret = df.set_index('trade_date')['log_ret']
        s_ret.index = s_ret.index.normalize()
        all_returns[symbol] = s_ret

        s_prc = df.set_index('trade_date')['close']
        s_prc.index = s_prc.index.normalize()
        price_data[symbol] = s_prc
        valid_symbols.append(symbol)

    returns_df = pd.DataFrame(all_returns).dropna(how='all').sort_index()
    prices_df = pd.DataFrame(price_data).dropna(how='all').sort_index()

    print(f"  Symbols loaded: {len(valid_symbols)}")
    print(f"  Date range: {returns_df.index[0].date()} to {returns_df.index[-1].date()}")
    print(f"  Total trading days: {len(returns_df)}")

    sector_counts = defaultdict(int)
    for s in valid_symbols:
        sector_counts[SECTOR_MAP.get(s, 'Unknown')] += 1
    print(f"  Sectors: {dict(sorted(sector_counts.items()))}")

    return returns_df, prices_df, valid_symbols


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Rolling 60-Day Pairwise Correlations
# ══════════════════════════════════════════════════════════════════════════════

def compute_rolling_correlations(returns_df, window=ROLLING_WINDOW):
    print(f"\n{'=' * 90}")
    print(f" STEP 2: Computing rolling {window}-day pairwise correlations")
    print(f"{'=' * 90}")

    symbols = returns_df.columns.tolist()
    n_pairs = len(symbols) * (len(symbols) - 1) // 2
    print(f"  Symbols: {len(symbols)}, Possible pairs: {n_pairs}")

    pair_corr_dict = {}
    for i, sym_a in enumerate(symbols):
        for sym_b in symbols[i+1:]:
            s1 = returns_df[sym_a].dropna()
            s2 = returns_df[sym_b].dropna()
            combined = pd.DataFrame({'a': s1, 'b': s2}).dropna()
            if len(combined) < window + 100:
                continue
            rc = combined['a'].rolling(window).corr(combined['b']).dropna()
            if len(rc) < 100:
                continue
            pair_corr_dict[(sym_a, sym_b)] = rc

    print(f"  Computed rolling correlations for {len(pair_corr_dict)} pairs")
    return pair_corr_dict


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Pair Statistics & Break Detection
# ══════════════════════════════════════════════════════════════════════════════

def compute_pair_statistics(pair_corr_dict, returns_df, recent_years=RECENT_YEARS):
    """Compute pair-level statistics using both full history and recent window."""
    print(f"\n{'=' * 90}")
    print(f" STEP 3: Computing pair statistics and detecting correlation breaks")
    print(f"{'=' * 90}")

    # Define recent window cutoff
    last_date = returns_df.index[-1]
    recent_cutoff = last_date - pd.DateOffset(years=recent_years)

    pair_stats = []
    break_signals = []

    for pair_key, rolling_corr in pair_corr_dict.items():
        sym_a, sym_b = pair_key

        # 1-year rolling mean/std for z-score
        corr_mean_1y = rolling_corr.rolling(LOOKBACK_1Y, min_periods=120).mean()
        corr_std_1y = rolling_corr.rolling(LOOKBACK_1Y, min_periods=120).std()
        corr_std_1y = corr_std_1y.replace(0, np.nan)

        # Current values
        curr_corr = rolling_corr.iloc[-1]
        curr_mean_1y = corr_mean_1y.iloc[-1]
        curr_std_1y = corr_std_1y.iloc[-1]

        if pd.isna(curr_mean_1y) or pd.isna(curr_std_1y) or curr_std_1y == 0:
            continue

        curr_zscore = (curr_corr - curr_mean_1y) / curr_std_1y

        # Full history stats
        full_mean = rolling_corr.mean()
        full_std = rolling_corr.std()

        # Recent 3-year stats (more relevant for pair trading)
        recent_corr = rolling_corr[rolling_corr.index >= recent_cutoff]
        if len(recent_corr) < 100:
            continue
        recent_mean = recent_corr.mean()
        recent_std = recent_corr.std()
        recent_stability = recent_mean / recent_std if recent_std > 0 else 0

        # Count breaks: zscore < -2
        zscores = ((rolling_corr - corr_mean_1y) / corr_std_1y).dropna()
        n_breaks = (zscores < CORR_BREAK_ZSCORE).sum()
        break_pct = n_breaks / len(zscores) * 100 if len(zscores) > 0 else 0

        # Time spent in high-correlation regime (>0.5) in recent window
        high_corr_pct_recent = (recent_corr > 0.5).sum() / len(recent_corr) * 100

        # Maximum drawdown of correlation in recent window
        cummax = recent_corr.expanding().max()
        corr_dd = (recent_corr - cummax).min()

        pair_stats.append({
            'sym_a': sym_a, 'sym_b': sym_b,
            'sector_a': SECTOR_MAP.get(sym_a, 'Unknown'),
            'sector_b': SECTOR_MAP.get(sym_b, 'Unknown'),
            'full_avg_corr': full_mean,
            'full_std_corr': full_std,
            'recent_avg_corr': recent_mean,
            'recent_std_corr': recent_std,
            'recent_stability': recent_stability,
            'curr_corr_60d': curr_corr,
            'curr_corr_mean_1y': curr_mean_1y,
            'curr_zscore': curr_zscore,
            'n_breaks': n_breaks,
            'break_pct': break_pct,
            'high_corr_pct_recent': high_corr_pct_recent,
            'corr_dd_recent': corr_dd,
            'n_observations': len(rolling_corr),
        })

        if curr_zscore < CORR_BREAK_ZSCORE:
            break_signals.append({
                'sym_a': sym_a, 'sym_b': sym_b,
                'sector_a': SECTOR_MAP.get(sym_a, 'Unknown'),
                'sector_b': SECTOR_MAP.get(sym_b, 'Unknown'),
                'curr_corr_60d': curr_corr,
                'recent_avg_corr': recent_mean,
                'full_avg_corr': full_mean,
                'recent_std_corr': recent_std,
                'curr_zscore': curr_zscore,
                'curr_corr_mean_1y': curr_mean_1y,
                'break_severity': abs(curr_zscore),
            })

    pair_stats_df = pd.DataFrame(pair_stats)
    break_signals_df = pd.DataFrame(break_signals)

    print(f"  Pairs analyzed: {len(pair_stats_df)}")
    print(f"  Current correlation breaks (zscore < {CORR_BREAK_ZSCORE}): {len(break_signals_df)}")
    print(f"  Recent-avg-corr > 0.5 pairs: {len(pair_stats_df[pair_stats_df['recent_avg_corr'] > 0.5])}")
    print(f"  Recent-avg-corr > 0.7 pairs: {len(pair_stats_df[pair_stats_df['recent_avg_corr'] > 0.7])}")

    return pair_stats_df, break_signals_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Display Results
# ══════════════════════════════════════════════════════════════════════════════

def display_top_correlations(pair_stats_df):
    print(f"\n{'=' * 90}")
    print(f" TOP 40 POSITIVE CORRELATIONS (recent {RECENT_YEARS}-year average)")
    print(f"{'=' * 90}")
    cols = ['sym_a', 'sym_b', 'sector_a', 'sector_b', 'recent_avg_corr',
            'recent_std_corr', 'curr_corr_60d', 'high_corr_pct_recent']
    top = pair_stats_df.nlargest(40, 'recent_avg_corr')
    print(top[cols].to_string(index=False))

    print(f"\n{'=' * 90}")
    print(f" TOP 20 NEGATIVE CORRELATIONS (recent {RECENT_YEARS}-year average)")
    print(f"{'=' * 90}")
    bot = pair_stats_df.nsmallest(20, 'recent_avg_corr')
    print(bot[cols].to_string(index=False))


def display_stable_pairs(pair_stats_df, top_n=TOP_N):
    print(f"\n{'=' * 90}")
    print(f" TOP {top_n} MOST STABLE PAIRS (recent avg_corr / recent_std, min avg>0.4)")
    print(f"{'=' * 90}")
    mask = (pair_stats_df['recent_avg_corr'] > 0.25) & (pair_stats_df['n_observations'] > 500)
    stable = pair_stats_df[mask].nlargest(top_n, 'recent_stability')
    if len(stable) == 0:
        mask = pair_stats_df['recent_avg_corr'] > 0.2
        stable = pair_stats_df[mask].nlargest(top_n, 'recent_stability')
    cols = ['sym_a', 'sym_b', 'sector_a', 'sector_b', 'recent_avg_corr',
            'recent_std_corr', 'recent_stability', 'curr_corr_60d', 'curr_zscore',
            'n_breaks', 'high_corr_pct_recent']
    print(stable[cols].to_string(index=False))


def display_break_pairs(break_signals_df, pair_stats_df, top_n=TOP_N):
    print(f"\n{'=' * 90}")
    print(f" TOP {top_n} CURRENT CORRELATION BREAKS")
    print(f" (zscore < {CORR_BREAK_ZSCORE}, sorted by severity)")
    print(f"{'=' * 90}")
    if len(break_signals_df) == 0:
        print("  No current breaks.")
        near = pair_stats_df.nsmallest(top_n, 'curr_zscore')
        cols = ['sym_a', 'sym_b', 'sector_a', 'sector_b',
                'recent_avg_corr', 'curr_corr_60d', 'curr_zscore']
        print(f"\n  Nearest to break:")
        print(near[cols].to_string(index=False))
        return

    cols = ['sym_a', 'sym_b', 'sector_a', 'sector_b',
            'recent_avg_corr', 'curr_corr_60d', 'curr_zscore', 'break_severity']
    print(break_signals_df.nlargest(top_n, 'break_severity')[cols].to_string(index=False))

    # High-corr pairs that are breaking (most actionable)
    high_corr_breaks = break_signals_df[break_signals_df['recent_avg_corr'] > 0.5]
    if len(high_corr_breaks) > 0:
        print(f"\n  {'=' * 80}")
        print(f"  ACTIONABLE BREAKS: High-corr pairs (>0.5 recent avg) now breaking")
        print(f"  {'=' * 80}")
        print(high_corr_breaks.nlargest(min(top_n, len(high_corr_breaks)), 'break_severity')[
            ['sym_a', 'sym_b', 'recent_avg_corr', 'curr_corr_60d', 'curr_zscore', 'break_severity']
        ].to_string(index=False))

    # Most frequent breakers
    print(f"\n  TOP {top_n} MOST FREQUENT HISTORICAL BREAKERS")
    freq = pair_stats_df.nlargest(top_n, 'break_pct')
    cols2 = ['sym_a', 'sym_b', 'sector_a', 'sector_b',
             'recent_avg_corr', 'n_breaks', 'break_pct', 'curr_zscore']
    print(freq[cols2].to_string(index=False))


def display_sector_correlations(pair_stats_df):
    print(f"\n{'=' * 90}")
    print(f" SECTOR-LEVEL CORRELATION (recent {RECENT_YEARS}-year)")
    print(f"{'=' * 90}")

    pair_stats_df['_sp'] = pair_stats_df.apply(
        lambda r: tuple(sorted([r['sector_a'], r['sector_b']])), axis=1)

    rows = []
    for sp, g in pair_stats_df.groupby('_sp'):
        rows.append({
            'sector_pair': f"{sp[0]}-{sp[1]}",
            'avg_corr': g['recent_avg_corr'].mean(),
            'median_corr': g['recent_avg_corr'].median(),
            'n_pairs': len(g),
            'avg_curr_corr': g['curr_corr_60d'].mean(),
        })
    print(pd.DataFrame(rows).sort_values('avg_corr', ascending=False).to_string(index=False))

    print(f"\n  WITHIN-SECTOR:")
    for sec in sorted(pair_stats_df['sector_a'].unique()):
        w = pair_stats_df[(pair_stats_df['sector_a'] == sec) & (pair_stats_df['sector_b'] == sec)]
        if len(w) == 0:
            continue
        print(f"    {sec:12s}: avg={w['recent_avg_corr'].mean():.3f}  "
              f"med={w['recent_avg_corr'].median():.3f}  "
              f"n={len(w):3d}  "
              f"curr={w['curr_corr_60d'].mean():.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Correlation Regression Analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyze_correlation_regressions(pair_corr_dict, returns_df, pair_stats_df,
                                    hold_days_list=[5, 10, 20]):
    print(f"\n{'=' * 90}")
    print(f" STEP 5: CORRELATION REGRESSION ANALYSIS")
    print(f"{'=' * 90}")

    # Focus on pairs with decent recent correlation
    focus = pair_stats_df[pair_stats_df['recent_avg_corr'] > 0.3].nlargest(100, 'recent_avg_corr')

    all_reg = []
    for _, row in focus.iterrows():
        pk = (row['sym_a'], row['sym_b'])
        if pk not in pair_corr_dict:
            continue
        rc = pair_corr_dict[pk]
        if len(rc) < LOOKBACK_1Y + 50:
            continue

        sym_a, sym_b = pk
        ra = returns_df[sym_a]
        rb = returns_df[sym_b]

        # Build aligned combined dataframe
        combined = pd.DataFrame({'a': ra, 'b': rb, 'rc': rc}).dropna()
        if len(combined) < LOOKBACK_1Y + 50:
            continue

        cm = combined['rc'].rolling(LOOKBACK_1Y, min_periods=120).mean()
        cs = combined['rc'].rolling(LOOKBACK_1Y, min_periods=120).std().replace(0, np.nan)
        zs = ((combined['rc'] - cm) / cs).dropna()

        break_idx = zs[zs < CORR_BREAK_ZSCORE].index

        for bi in break_idx:
            try:
                loc = combined.index.get_loc(bi)
            except KeyError:
                continue
            if loc < 20:
                continue
            corr_at_break = combined['rc'].iloc[loc]

            # Determine direction from past 20 days BEFORE the break
            perf_a = combined['a'].iloc[loc-20:loc].sum()
            perf_b = combined['b'].iloc[loc-20:loc].sum()

            for hd in hold_days_list:
                if loc + hd >= len(combined):
                    continue
                corr_after = combined['rc'].iloc[loc + hd]
                corr_delta = corr_after - corr_at_break

                # Forward returns
                fwd_a = combined['a'].iloc[loc+1:loc+1+hd].sum()
                fwd_b = combined['b'].iloc[loc+1:loc+1+hd].sum()

                # Long the underperformer (before break), short the overperformer
                if perf_a < perf_b:
                    trade_pnl = fwd_a - fwd_b
                else:
                    trade_pnl = fwd_b - fwd_a

                all_reg.append({
                    'pair': f"{sym_a}/{sym_b}",
                    'sector_pair': f"{row['sector_a']}-{row['sector_b']}",
                    'break_date': bi,
                    'hold_days': hd,
                    'corr_at_break': corr_at_break,
                    'corr_after': corr_after,
                    'corr_delta': corr_delta,
                    'regressed': corr_delta > 0,
                    'zscore_at_break': zs.loc[bi],
                    'trade_pnl': trade_pnl,
                    'recent_avg_corr': row['recent_avg_corr'],
                })

    if not all_reg:
        print("  No regression events found (thresholds too strict).")
        return pd.DataFrame()

    reg_df = pd.DataFrame(all_reg)
    print(f"  Total break events analyzed: {len(reg_df)}")
    print(f"  Unique pairs: {reg_df['pair'].nunique()}")

    # Regression rate by hold period
    print(f"\n  --- REGRESSION & CONVERGENCE BY HOLD PERIOD ---")
    for hd, grp in reg_df.groupby('hold_days'):
        reg_rate = grp['regressed'].mean()
        conv_pnl = grp['trade_pnl'].mean()
        wr = (grp['trade_pnl'] > 0).mean()
        print(f"    Hold {hd:2d}d: n={len(grp):5d}  corr_regression={reg_rate:.1%}  "
              f"convergence_wr={wr:.1%}  avg_pnl={conv_pnl:+.4f}")

    # By break severity
    print(f"\n  --- BY BREAK SEVERITY (all hold periods) ---")
    reg_df['sev'] = pd.cut(reg_df['zscore_at_break'],
                            bins=[-10, -4, -3, -2.5, -2],
                            labels=['<-4', '-4~-3', '-3~-2.5', '-2.5~-2'])
    sev = reg_df.groupby('sev', observed=True).agg(
        n=('regressed', 'count'),
        reg_rate=('regressed', 'mean'),
        avg_corr_delta=('corr_delta', 'mean'),
        conv_wr=('trade_pnl', lambda x: (x > 0).mean()),
        avg_pnl=('trade_pnl', 'mean'),
    )
    print(sev.to_string())

    # By recent correlation level
    print(f"\n  --- BY RECENT AVG CORRELATION ---")
    reg_df['corr_bucket'] = pd.cut(reg_df['recent_avg_corr'],
                                    bins=[0.2, 0.3, 0.4, 0.5, 1.0],
                                    labels=['0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5+'])
    cb = reg_df.groupby('corr_bucket', observed=True).agg(
        n=('regressed', 'count'),
        reg_rate=('regressed', 'mean'),
        conv_wr=('trade_pnl', lambda x: (x > 0).mean()),
        avg_pnl=('trade_pnl', 'mean'),
    )
    print(cb.to_string())

    return reg_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Backtest Pair Convergence Strategy
# ══════════════════════════════════════════════════════════════════════════════

def backtest_pair_convergence(returns_df, prices_df, pair_stats_df, pair_corr_dict,
                               hold_days=HOLD_DAYS):
    """Backtest convergence after correlation breaks using spread-based entry."""
    print(f"\n{'=' * 90}")
    print(f" STEP 6: BACKTEST - Pair Convergence After Correlation Breaks")
    print(f"{'=' * 90}")
    print(f"  Strategy:")
    print(f"    1. Select pairs with recent_avg_corr > 0.3 (top 100 by avg corr)")
    print(f"    2. Entry signal: 60d corr drops more than 1.5 std below 1y rolling mean")
    print(f"    3. Direction: long the underperformer (past 20d), short the overperformer")
    print(f"    4. Hold {hold_days} trading days")
    print(f"    5. P&L = long return - short return (log return basis)")

    # Select pairs
    eligible = pair_stats_df[pair_stats_df['recent_avg_corr'] > 0.3].nlargest(100, 'recent_avg_corr')
    if len(eligible) == 0:
        eligible = pair_stats_df.nlargest(100, 'recent_avg_corr')
    print(f"\n  Eligible pairs: {len(eligible)}")

    all_trades = []
    cooldown_days = hold_days * 2

    for _, row in eligible.iterrows():
        sym_a, sym_b = row['sym_a'], row['sym_b']
        pk = (sym_a, sym_b)

        if pk not in pair_corr_dict:
            continue
        rc = pair_corr_dict[pk]

        # Get aligned returns
        ra = returns_df[sym_a].dropna()
        rb = returns_df[sym_b].dropna()
        combined = pd.DataFrame({'a': ra, 'b': rb}).dropna()
        if len(combined) < ROLLING_WINDOW + hold_days + 100:
            continue

        # Rolling 60d correlation
        combined['roll_corr'] = combined['a'].rolling(ROLLING_WINDOW).corr(combined['b'])

        # 1y rolling mean of the 60d correlation
        combined['corr_mean'] = combined['roll_corr'].rolling(LOOKBACK_1Y, min_periods=120).mean()
        combined['corr_std'] = combined['roll_corr'].rolling(LOOKBACK_1Y, min_periods=120).std()

        # Break signal: 60d corr drops more than 1 std below its 1y mean
        # AND recent_avg_corr was decent (already filtered in eligible)
        combined['corr_zscore'] = (combined['roll_corr'] - combined['corr_mean']) / combined['corr_std']
        combined['break_signal'] = combined['corr_zscore'] < -1.5

        # Require correlation was recently "normal" (above mean - 0.5 std)
        combined['was_normal'] = combined['corr_zscore'].shift(5) > -0.5

        combined['entry'] = combined['break_signal'] & combined['was_normal']
        combined = combined.dropna(subset=['entry'])

        entries = combined[combined['entry']].index
        last_exit_date = None

        for entry_date in entries:
            # Cooldown
            if last_exit_date is not None and entry_date < last_exit_date + pd.Timedelta(days=cooldown_days):
                continue

            loc = combined.index.get_loc(entry_date)
            if loc < 20:
                continue

            # Determine direction: long the underperformer over past 20 days
            perf_a = combined['a'].iloc[loc-20:loc].sum()
            perf_b = combined['b'].iloc[loc-20:loc].sum()

            if perf_a <= perf_b:
                # A underperformed -> long A, short B
                trade_pnl_series = combined['a'].iloc[loc+1:loc+1+hold_days] - \
                                   combined['b'].iloc[loc+1:loc+1+hold_days]
            else:
                # B underperformed -> long B, short A
                trade_pnl_series = combined['b'].iloc[loc+1:loc+1+hold_days] - \
                                   combined['a'].iloc[loc+1:loc+1+hold_days]

            if len(trade_pnl_series) == 0:
                continue

            total_pnl = trade_pnl_series.sum()
            actual_hold = len(trade_pnl_series)

            exit_loc = min(loc + hold_days, len(combined) - 1)
            last_exit_date = combined.index[exit_loc]

            all_trades.append({
                'pair': f"{sym_a}/{sym_b}",
                'long': sym_a if perf_a <= perf_b else sym_b,
                'short': sym_b if perf_a <= perf_b else sym_a,
                'entry_date': entry_date,
                'exit_date': combined.index[exit_loc],
                'hold_days': actual_hold,
                'pnl': total_pnl,
                'annualized': total_pnl / actual_hold * 252 if actual_hold > 0 else 0,
                'entry_corr': combined['roll_corr'].iloc[loc],
                'entry_zscore': combined['corr_zscore'].iloc[loc],
                'recent_avg_corr': row['recent_avg_corr'],
                'sector_pair': f"{row['sector_a']}-{row['sector_b']}",
            })

    if not all_trades:
        print("\n  No trades generated.")
        return pd.DataFrame()

    trades_df = pd.DataFrame(all_trades)
    wr = (trades_df['pnl'] > 0).mean()
    avg_pnl = trades_df['pnl'].mean()
    med_pnl = trades_df['pnl'].median()
    avg_ann = trades_df['annualized'].mean()
    sharpe = trades_df['pnl'].mean() / trades_df['pnl'].std() * np.sqrt(252 / hold_days) \
             if trades_df['pnl'].std() > 0 else 0

    # Cumulative P&L curve stats
    trades_sorted = trades_df.sort_values('entry_date')
    cum_pnl = trades_sorted['pnl'].cumsum()
    max_dd = (cum_pnl.expanding().max() - cum_pnl).max()

    print(f"\n  --- OVERALL BACKTEST RESULTS ---")
    print(f"  Total trades:          {len(trades_df)}")
    print(f"  Win rate:              {wr:.1%}")
    print(f"  Avg P&L per trade:     {avg_pnl:+.4f} ({avg_pnl*100:+.2f}%)")
    print(f"  Median P&L per trade:  {med_pnl:+.4f} ({med_pnl*100:+.2f}%)")
    print(f"  Avg annualized return: {avg_ann:+.2%}")
    print(f"  Sharpe ratio (proxy):  {sharpe:+.2f}")
    print(f"  Cumulative P&L:        {cum_pnl.iloc[-1]:+.4f}")
    print(f"  Max drawdown:          {max_dd:+.4f}")
    print(f"  Best trade:            {trades_df['pnl'].max():+.4f} ({trades_df.loc[trades_df['pnl'].idxmax(), 'pair']})")
    print(f"  Worst trade:           {trades_df['pnl'].min():+.4f} ({trades_df.loc[trades_df['pnl'].idxmin(), 'pair']})")

    # By sector pair
    print(f"\n  --- RESULTS BY SECTOR PAIR (min 10 trades) ---")
    sp = trades_df.groupby('sector_pair').agg(
        n=('pnl', 'count'),
        wr=('pnl', lambda x: (x > 0).mean()),
        avg_pnl=('pnl', 'mean'),
        total_pnl=('pnl', 'sum'),
        avg_ann=('annualized', 'mean'),
    ).sort_values('total_pnl', ascending=False)
    sp = sp[sp['n'] >= 10]
    print(sp.to_string())

    # Top profitable pairs
    print(f"\n  --- TOP 20 MOST PROFITABLE PAIRS ---")
    pr = trades_df.groupby('pair').agg(
        n=('pnl', 'count'),
        wr=('pnl', lambda x: (x > 0).mean()),
        avg_pnl=('pnl', 'mean'),
        total_pnl=('pnl', 'sum'),
        avg_ann=('annualized', 'mean'),
    ).sort_values('total_pnl', ascending=False)
    print(pr.head(20).to_string())

    # P&L by year
    print(f"\n  --- P&L BY YEAR ---")
    trades_sorted['year'] = trades_sorted['entry_date'].dt.year
    yearly = trades_sorted.groupby('year').agg(
        n=('pnl', 'count'),
        wr=('pnl', lambda x: (x > 0).mean()),
        avg_pnl=('pnl', 'mean'),
        total_pnl=('pnl', 'sum'),
    )
    print(yearly.to_string())

    # Recent trades
    print(f"\n  --- LAST 20 TRADES ---")
    cols = ['pair', 'long', 'short', 'entry_date', 'hold_days',
            'pnl', 'entry_corr', 'entry_zscore', 'recent_avg_corr']
    print(trades_df.nlargest(20, 'entry_date')[cols].to_string(index=False))

    return trades_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Additional Analysis - Current Opportunity Scan
# ══════════════════════════════════════════════════════════════════════════════

def scan_current_opportunities(pair_stats_df, break_signals_df, returns_df, prices_df):
    """Scan for current actionable pair trading opportunities."""
    print(f"\n{'=' * 90}")
    print(f" STEP 7: CURRENT OPPORTUNITY SCAN")
    print(f"{'=' * 90}")

    # High-corr pairs with current correlation significantly below recent average
    actionable = pair_stats_df[
        (pair_stats_df['recent_avg_corr'] > 0.35) &
        (pair_stats_df['curr_corr_60d'] < pair_stats_df['recent_avg_corr'] - 0.15)
    ].copy()

    if len(actionable) == 0:
        # Relax criteria
        actionable = pair_stats_df[
            (pair_stats_df['recent_avg_corr'] > 0.3) &
            (pair_stats_df['curr_corr_60d'] < pair_stats_df['recent_avg_corr'] - 0.10)
        ].copy()

    print(f"  Pairs with recent avg_corr > 0.5 and current 60d corr significantly below avg: {len(actionable)}")

    if len(actionable) == 0:
        print("  No actionable divergence opportunities found at current time.")
        return

    # For each actionable pair, determine direction
    last_date = returns_df.index[-1]
    lookback = min(20, len(returns_df) - 1)
    recent_start = returns_df.index[-lookback]

    print(f"\n  {'=' * 80}")
    print(f"  ACTIONABLE PAIR TRADES (as of {last_date.date()})")
    print(f"  {'=' * 80}")

    opportunities = []
    for _, row in actionable.nlargest(30, 'recent_avg_corr').iterrows():
        sym_a, sym_b = row['sym_a'], row['sym_b']
        if sym_a not in returns_df.columns or sym_b not in returns_df.columns:
            continue

        ra = returns_df[sym_a].loc[recent_start:last_date].dropna()
        rb = returns_df[sym_b].loc[recent_start:last_date].dropna()
        common = ra.index.intersection(rb.index)
        if len(common) < 5:
            continue

        perf_a = ra.loc[common].sum()
        perf_b = rb.loc[common].sum()
        corr_gap = row['recent_avg_corr'] - row['curr_corr_60d']

        long_sym = sym_a if perf_a < perf_b else sym_b
        short_sym = sym_b if perf_a < perf_b else sym_a

        opportunities.append({
            'pair': f"{sym_a}/{sym_b}",
            'long': long_sym,
            'short': short_sym,
            'sector': f"{row['sector_a']}-{row['sector_b']}",
            'recent_avg_corr': row['recent_avg_corr'],
            'curr_corr_60d': row['curr_corr_60d'],
            'corr_gap': corr_gap,
            'curr_zscore': row['curr_zscore'],
            'perf_a_20d': perf_a,
            'perf_b_20d': perf_b,
        })

    if opportunities:
        opp_df = pd.DataFrame(opportunities).sort_values('corr_gap', ascending=False)
        print(opp_df.to_string(index=False))
    else:
        print("  No opportunities with sufficient recent data.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: Spread Z-Score Backtest (Standard Pairs Trading)
# ══════════════════════════════════════════════════════════════════════════════

def backtest_spread_zscore(returns_df, prices_df, pair_stats_df,
                            spread_window=20, entry_z=2.0, exit_z=0.5,
                            max_hold=15, recent_years=3):
    """
    Standard pairs trading using price spread z-score.
    For each correlated pair:
      - Compute price ratio (A/B)
      - Compute rolling z-score of the ratio over `spread_window` days
      - When z > entry_z: short A, long B (ratio too high, expect mean reversion)
      - When z < -entry_z: long A, short B (ratio too low, expect mean reversion)
      - Exit when z crosses exit_z toward 0, or max_hold days reached
    """
    print(f"\n{'=' * 90}")
    print(f" STEP 8: SPREAD Z-SCORE BACKTEST (Standard Pairs Trading)")
    print(f"{'=' * 90}")
    print(f"  Parameters:")
    print(f"    Pairs: top 30 by recent {recent_years}Y avg corr (min 0.25)")
    print(f"    Spread = log(price_A) - log(price_B)")
    print(f"    Z-score window: {spread_window} days")
    print(f"    Entry: |z| > {entry_z}")
    print(f"    Exit: |z| < {exit_z} or {max_hold} days max")
    print(f"    Position: log-return based (long underperformer, short outperformer)")

    last_date = returns_df.index[-1]
    recent_cutoff = last_date - pd.DateOffset(years=recent_years)

    # Select pairs
    eligible = pair_stats_df[
        (pair_stats_df['recent_avg_corr'] > 0.25)
    ].nlargest(30, 'recent_avg_corr')

    if len(eligible) == 0:
        eligible = pair_stats_df.nlargest(30, 'recent_avg_corr')

    print(f"\n  Eligible pairs: {len(eligible)}")

    all_trades = []

    for _, row in eligible.iterrows():
        sym_a, sym_b = row['sym_a'], row['sym_b']
        if sym_a not in prices_df.columns or sym_b not in prices_df.columns:
            continue

        pa = prices_df[sym_a].dropna()
        pb = prices_df[sym_b].dropna()

        # Use log prices for spread
        log_pa = np.log(pa)
        log_pb = np.log(pb)

        combined = pd.DataFrame({'log_a': log_pa, 'log_b': log_pb}).dropna()
        if len(combined) < spread_window + max_hold + 50:
            continue

        # Compute spread
        combined['spread'] = combined['log_a'] - combined['log_b']

        # Rolling z-score of spread
        spread_mean = combined['spread'].rolling(spread_window).mean()
        spread_std = combined['spread'].rolling(spread_window).std()
        combined['z'] = (combined['spread'] - spread_mean) / spread_std.replace(0, np.nan)
        combined = combined.dropna(subset=['z'])

        if len(combined) < 100:
            continue

        # Get returns
        ra = returns_df[sym_a].dropna()
        rb = returns_df[sym_b].dropna()
        combined['ret_a'] = ra.reindex(combined.index)
        combined['ret_b'] = rb.reindex(combined.index)
        combined = combined.dropna(subset=['ret_a', 'ret_b'])

        if len(combined) < 100:
            continue

        # Generate trades
        position = 0  # 0 = flat, 1 = long spread (long A, short B), -1 = short spread
        entry_idx = None
        entry_z_val = 0

        for i in range(len(combined)):
            z = combined['z'].iloc[i]

            if position == 0:
                # No position - check for entry
                if z > entry_z:
                    position = -1  # short spread: short A, long B (expect spread to decrease)
                    entry_idx = i
                    entry_z_val = z
                elif z < -entry_z:
                    position = 1  # long spread: long A, short B (expect spread to increase)
                    entry_idx = i
                    entry_z_val = z

            elif position != 0:
                # In position - check for exit
                days_held = i - entry_idx
                should_exit = False

                if position == 1 and z > -exit_z:
                    should_exit = True
                elif position == -1 and z < exit_z:
                    should_exit = True
                elif days_held >= max_hold:
                    should_exit = True

                if should_exit and days_held > 0:
                    # Calculate P&L
                    if position == 1:
                        # Long A, short B
                        pnl = combined['ret_a'].iloc[entry_idx+1:i+1].sum() - \
                              combined['ret_b'].iloc[entry_idx+1:i+1].sum()
                    else:
                        # Short A, long B
                        pnl = combined['ret_b'].iloc[entry_idx+1:i+1].sum() - \
                              combined['ret_a'].iloc[entry_idx+1:i+1].sum()

                    all_trades.append({
                        'pair': f"{sym_a}/{sym_b}",
                        'sector_pair': f"{row['sector_a']}-{row['sector_b']}",
                        'entry_date': combined.index[entry_idx],
                        'exit_date': combined.index[i],
                        'hold_days': days_held,
                        'direction': 'long_spread' if position == 1 else 'short_spread',
                        'pnl': pnl,
                        'annualized': pnl / days_held * 252 if days_held > 0 else 0,
                        'entry_z': entry_z_val,
                        'exit_z': z,
                        'recent_avg_corr': row['recent_avg_corr'],
                    })
                    position = 0
                    entry_idx = None

        # Close any open position at end
        if position != 0 and entry_idx is not None:
            days_held = len(combined) - 1 - entry_idx
            if days_held > 0:
                if position == 1:
                    pnl = combined['ret_a'].iloc[entry_idx+1:].sum() - \
                          combined['ret_b'].iloc[entry_idx+1:].sum()
                else:
                    pnl = combined['ret_b'].iloc[entry_idx+1:].sum() - \
                          combined['ret_a'].iloc[entry_idx+1:].sum()

                all_trades.append({
                    'pair': f"{sym_a}/{sym_b}",
                    'sector_pair': f"{row['sector_a']}-{row['sector_b']}",
                    'entry_date': combined.index[entry_idx],
                    'exit_date': combined.index[-1],
                    'hold_days': days_held,
                    'direction': 'long_spread' if position == 1 else 'short_spread',
                    'pnl': pnl,
                    'annualized': pnl / days_held * 252 if days_held > 0 else 0,
                    'entry_z': entry_z_val,
                    'exit_z': combined['z'].iloc[-1],
                    'recent_avg_corr': row['recent_avg_corr'],
                })

    if not all_trades:
        print("\n  No trades generated.")
        return pd.DataFrame()

    trades_df = pd.DataFrame(all_trades)

    # Filter to recent years only for cleaner results
    trades_recent = trades_df[trades_df['entry_date'] >= recent_cutoff].copy()
    trades_full = trades_df.copy()

    for label, tdf in [("FULL HISTORY", trades_full), (f"RECENT {recent_years}Y", trades_recent)]:
        if len(tdf) == 0:
            continue
        wr = (tdf['pnl'] > 0).mean()
        avg_pnl = tdf['pnl'].mean()
        med_pnl = tdf['pnl'].median()
        avg_hold = tdf['hold_days'].mean()
        avg_ann = tdf['annualized'].mean()
        sharpe = tdf['pnl'].mean() / tdf['pnl'].std() * np.sqrt(252 / avg_hold) \
                 if tdf['pnl'].std() > 0 else 0
        cum_pnl = tdf.sort_values('entry_date')['pnl'].cumsum()
        max_dd = (cum_pnl.expanding().max() - cum_pnl).max()

        print(f"\n  --- {label} RESULTS ---")
        print(f"  Total trades:          {len(tdf)}")
        print(f"  Win rate:              {wr:.1%}")
        print(f"  Avg P&L per trade:     {avg_pnl:+.5f} ({avg_pnl*100:+.3f}%)")
        print(f"  Median P&L per trade:  {med_pnl:+.5f} ({med_pnl*100:+.3f}%)")
        print(f"  Avg hold days:         {avg_hold:.1f}")
        print(f"  Avg annualized return: {avg_ann:+.2%}")
        print(f"  Sharpe ratio (proxy):  {sharpe:+.2f}")
        print(f"  Cumulative P&L:        {cum_pnl.iloc[-1]:+.4f}")
        print(f"  Max drawdown:          {max_dd:+.4f}")

    # By pair (full history)
    print(f"\n  --- RESULTS BY PAIR (full history) ---")
    pr = trades_full.groupby('pair').agg(
        n=('pnl', 'count'),
        wr=('pnl', lambda x: (x > 0).mean()),
        avg_pnl=('pnl', 'mean'),
        total_pnl=('pnl', 'sum'),
        avg_hold=('hold_days', 'mean'),
    ).sort_values('total_pnl', ascending=False)
    print(pr.to_string())

    # By direction
    print(f"\n  --- BY DIRECTION ---")
    dr = trades_full.groupby('direction').agg(
        n=('pnl', 'count'),
        wr=('pnl', lambda x: (x > 0).mean()),
        avg_pnl=('pnl', 'mean'),
        total_pnl=('pnl', 'sum'),
    )
    print(dr.to_string())

    # By year
    print(f"\n  --- P&L BY YEAR ---")
    trades_full['year'] = trades_full['entry_date'].dt.year
    yearly = trades_full.groupby('year').agg(
        n=('pnl', 'count'),
        wr=('pnl', lambda x: (x > 0).mean()),
        avg_pnl=('pnl', 'mean'),
        total_pnl=('pnl', 'sum'),
    )
    print(yearly.to_string())

    # By entry z-score magnitude
    print(f"\n  --- BY ENTRY |Z| MAGNITUDE ---")
    trades_full['entry_z_abs'] = trades_full['entry_z'].abs()
    trades_full['z_bucket'] = pd.cut(trades_full['entry_z_abs'],
                                      bins=[2.0, 2.5, 3.0, 4.0, 20],
                                      labels=['2.0-2.5', '2.5-3.0', '3.0-4.0', '4.0+'])
    zb = trades_full.groupby('z_bucket', observed=True).agg(
        n=('pnl', 'count'),
        wr=('pnl', lambda x: (x > 0).mean()),
        avg_pnl=('pnl', 'mean'),
        total_pnl=('pnl', 'sum'),
    )
    print(zb.to_string())

    # Top recent trades
    print(f"\n  --- LAST 20 TRADES ---")
    cols = ['pair', 'direction', 'entry_date', 'hold_days', 'pnl', 'entry_z', 'exit_z']
    print(trades_full.nlargest(20, 'entry_date')[cols].to_string(index=False))

    return trades_full


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 90)
    print(" FUTURES CORRELATION ANALYSIS SYSTEM v2")
    print(" Pair Trading Opportunity Scanner")
    print("=" * 90)

    # Step 1
    returns_df, prices_df, valid_symbols = load_all_futures(DATA_DIR)

    # Step 2
    pair_corr_dict = compute_rolling_correlations(returns_df)

    # Step 3
    pair_stats_df, break_signals_df = compute_pair_statistics(pair_corr_dict, returns_df)

    # Step 4: Display
    display_top_correlations(pair_stats_df)
    display_stable_pairs(pair_stats_df)
    display_break_pairs(break_signals_df, pair_stats_df)
    display_sector_correlations(pair_stats_df)

    # Step 5
    reg_df = analyze_correlation_regressions(pair_corr_dict, returns_df, pair_stats_df)

    # Step 6
    trades_df = backtest_pair_convergence(returns_df, prices_df, pair_stats_df, pair_corr_dict)

    # Step 7
    scan_current_opportunities(pair_stats_df, break_signals_df, returns_df, prices_df)

    # Step 8: Spread z-score backtest
    spread_trades = backtest_spread_zscore(returns_df, prices_df, pair_stats_df)

    # Final summary
    print(f"\n{'=' * 90}")
    print(f" FINAL SUMMARY")
    print(f"{'=' * 90}")
    print(f"  Symbols analyzed:          {len(valid_symbols)}")
    print(f"  Pairs with statistics:     {len(pair_stats_df)}")
    rec05 = len(pair_stats_df[pair_stats_df['recent_avg_corr'] > 0.5])
    rec07 = len(pair_stats_df[pair_stats_df['recent_avg_corr'] > 0.7])
    print(f"  Recent avg_corr > 0.5:    {rec05} pairs")
    print(f"  Recent avg_corr > 0.7:    {rec07} pairs")
    print(f"  Current breaks:            {len(break_signals_df)}")

    best = pair_stats_df.nlargest(1, 'recent_stability').iloc[0]
    print(f"  Most stable pair:          {best['sym_a']}/{best['sym_b']} "
          f"(avg={best['recent_avg_corr']:.3f}, stability={best['recent_stability']:.2f})")

    if len(trades_df) > 0:
        print(f"  Corr-break backtest:       {len(trades_df)} trades, "
              f"WR={(trades_df['pnl'] > 0).mean():.1%}, "
              f"cumPnL={trades_df['pnl'].sum():+.4f}")

    if len(spread_trades) > 0:
        print(f"  Spread z-score backtest:   {len(spread_trades)} trades, "
              f"WR={(spread_trades['pnl'] > 0).mean():.1%}, "
              f"cumPnL={spread_trades['pnl'].sum():+.4f}")

    print(f"\n{'=' * 90}")
    print(f" Analysis complete.")
    print(f"{'=' * 90}")

    return pair_stats_df, break_signals_df, trades_df, spread_trades


if __name__ == '__main__':
    pair_stats_df, break_signals_df, trades_df, spread_trades = main()
