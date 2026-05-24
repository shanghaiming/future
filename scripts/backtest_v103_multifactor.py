#!/usr/bin/env python3
"""
Multi-Factor Ensemble Strategy for Chinese Commodity Futures
=============================================================
Factors:
  - Momentum: 5d, 10d, 20d returns
  - OI change: 5d OI change %
  - POI signal: sign(price_chg) * sign(OI_chg) * |OI_chg|
  - Term structure: spread_pct percentile rank
  - Volatility: 20d realized vol (inverse)
  - Volume ratio: 5d avg vol / 20d avg vol

Ensemble methods:
  - Equal weight rank combination
  - Optimized weights (training period IC-based)
  - Conditional: regime-dependent factor weights

Walk-forward: train 2021-2023, validate 2024, test 2025-2026
Risk: -2% SL, +5% TP per position
"""

import os
import warnings
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings('ignore')

DATA_DIR = os.path.expanduser('~/home/futures_platform/data/futures_weighted/')
TS_DIR = os.path.expanduser('~/home/futures_platform/data/futures_term_structure/')

# =============================================================================
# 1. Load and prepare data
# =============================================================================

def load_all_futures():
    """Load all commodity futures daily data into a single panel."""
    frames = []
    for f in sorted(os.listdir(DATA_DIR)):
        if not f.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(DATA_DIR, f))
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        df['symbol'] = f.replace('.csv', '')
        frames.append(df[['symbol', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'oi']])
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
    return panel


def load_term_structure():
    """Load term structure spread data from JSON files."""
    spread_data = []
    if not os.path.isdir(TS_DIR):
        print("  [WARN] Term structure directory not found, skipping TS factor.")
        return pd.DataFrame()

    for f in sorted(os.listdir(TS_DIR)):
        if not f.endswith('.json'):
            continue
        # Only use lowercase-named files (the "fi" weighted series)
        prefix = f.split('_')[0]
        if prefix != prefix.lower():
            continue
        try:
            with open(os.path.join(TS_DIR, f)) as fh:
                d = json.load(fh)
            date_str = d.get('date', '')
            if not date_str:
                continue
            spread_data.append({
                'symbol': d.get('symbol', prefix),
                'trade_date': pd.to_datetime(date_str),
                'structure': d.get('structure', ''),
                'spread_pct': float(d.get('total_spread_pct', 0)),
                'near_price': float(d.get('near_price', 0)),
                'far_price': float(d.get('far_price', 0)),
            })
        except Exception:
            continue

    if not spread_data:
        return pd.DataFrame()
    ts_df = pd.DataFrame(spread_data)
    ts_df = ts_df.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
    return ts_df


# =============================================================================
# 2. Factor computation
# =============================================================================

def compute_factors(panel, ts_df):
    """Compute all factors for each symbol on each date."""
    # Work per-symbol
    factor_frames = []

    symbols = panel['symbol'].unique()
    for sym in sorted(symbols):
        g = panel[panel['symbol'] == sym].copy().sort_values('trade_date').reset_index(drop=True)
        if len(g) < 60:
            continue

        close = g['close'].values
        oi = g['oi'].values
        vol = g['vol'].values
        high = g['high'].values
        low = g['low'].values

        df = g[['symbol', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'oi']].copy()

        # --- Momentum factors ---
        df['ret_5d'] = g['close'].pct_change(5)
        df['ret_10d'] = g['close'].pct_change(10)
        df['ret_20d'] = g['close'].pct_change(20)

        # --- OI change ---
        df['oi_chg_5d'] = g['oi'].pct_change(5)

        # --- POI signal ---
        price_chg_5d = g['close'].diff(5)
        oi_chg_5d = g['oi'].diff(5)
        df['poi_signal'] = (
            np.sign(price_chg_5d) * np.sign(oi_chg_5d) * np.abs(g['oi'].pct_change(5))
        )

        # --- Volatility (20d realized) ---
        rets = g['close'].pct_change()
        df['realized_vol_20d'] = rets.rolling(20).std() * np.sqrt(252)

        # --- Volume ratio ---
        vol_5d = g['vol'].rolling(5).mean()
        vol_20d = g['vol'].rolling(20).mean()
        df['vol_ratio'] = vol_5d / vol_20d

        factor_frames.append(df)

    factors = pd.concat(factor_frames, ignore_index=True)

    # --- Term structure: merge spread data ---
    if not ts_df.empty:
        ts_agg = ts_df[['symbol', 'trade_date', 'spread_pct', 'structure']].copy()
        factors = factors.merge(ts_agg, on=['symbol', 'trade_date'], how='left')
        # Forward-fill TS data within each symbol (TS may not be daily)
        factors = factors.sort_values(['symbol', 'trade_date'])
        factors['spread_pct'] = factors.groupby('symbol')['spread_pct'].ffill(limit=10)
    else:
        factors['spread_pct'] = np.nan
        factors['structure'] = ''

    return factors


def cross_sectional_ranks(factors, factor_cols, date_col='trade_date'):
    """Rank each factor cross-sectionally on each date. Rank 1=worst, N=best."""
    rank_frames = []
    dates = sorted(factors[date_col].unique())

    for dt in dates:
        day = factors[factors[date_col] == dt].copy()
        if len(day) < 10:
            continue

        for col in factor_cols:
            valid = day[col].notna()
            if valid.sum() < 5:
                day.loc[valid, f'{col}_rank'] = np.nan
                continue
            # Rank: higher value = higher rank (bullish signal for momentum-like factors)
            ranks = day.loc[valid, col].rank(method='average', pct=True)
            day.loc[valid, f'{col}_rank'] = ranks

        rank_frames.append(day)

    return pd.concat(rank_frames, ignore_index=True)


# =============================================================================
# 3. Composite score weighting schemes
# =============================================================================

def equal_weight_score(df, factor_cols):
    """Equal weight: simple average of percentile ranks."""
    rank_cols = [f'{c}_rank' for c in factor_cols]
    return df[rank_cols].mean(axis=1)


def compute_ic_weights(factors, factor_cols, train_start, train_end):
    """Compute IC-based weights from training period.

    IC = rank correlation of factor with forward 5d return.
    Weight proportional to mean(|IC|) over training period.
    """
    rank_cols = [f'{c}_rank' for c in factor_cols]
    train = factors[
        (factors['trade_date'] >= pd.Timestamp(train_start)) &
        (factors['trade_date'] <= pd.Timestamp(train_end))
    ].copy()

    # Compute forward 5d return as target
    train = train.sort_values(['symbol', 'trade_date'])
    train['fwd_ret_5d'] = train.groupby('symbol')['close'].pct_change(5).shift(-5)

    # Monthly IC
    train['month'] = train['trade_date'].dt.to_period('M')
    ic_per_factor = {}

    for col in factor_cols:
        rank_col = f'{col}_rank'
        ics = []
        for month, grp in train.groupby('month'):
            valid = grp[[rank_col, 'fwd_ret_5d']].dropna()
            if len(valid) < 10:
                continue
            from scipy.stats import spearmanr
            ic, _ = spearmanr(valid[rank_col], valid['fwd_ret_5d'])
            ics.append(ic)
        if ics:
            ic_per_factor[col] = np.mean(np.abs(ics))
        else:
            ic_per_factor[col] = 0.0

    # Normalize weights
    total = sum(ic_per_factor.values())
    if total == 0:
        weights = {c: 1.0 / len(factor_cols) for c in factor_cols}
    else:
        weights = {c: v / total for c, v in ic_per_factor.items()}

    return weights, ic_per_factor


def optimized_weight_score(df, factor_cols, weights):
    """Weighted sum of rank percentiles using IC-derived weights."""
    score = pd.Series(0.0, index=df.index)
    for col in factor_cols:
        rank_col = f'{col}_rank'
        w = weights.get(col, 0)
        score += w * df[rank_col].fillna(0.5)
    return score


def regime_detection(factors, date):
    """Determine market regime based on cross-sectional average vol.

    Note: realized_vol_20d has been negated for ranking purposes,
    so we must negate it back to get the actual vol level.
    """
    day = factors[factors['trade_date'] == date]
    if len(day) == 0:
        return 'normal'
    # Negate back to get actual vol (column was negated for ranking: low vol = high rank)
    actual_vol = -day['realized_vol_20d'].mean()
    if pd.isna(actual_vol):
        return 'normal'
    # Regime thresholds based on annualized vol
    if actual_vol > 0.30:
        return 'high_vol'
    elif actual_vol < 0.15:
        return 'low_vol'
    else:
        return 'normal'


def conditional_weight_score(df, factors, factor_cols, date):
    """Use different factor weights depending on regime."""
    regime = regime_detection(factors, date)

    # Regime-specific weights
    if regime == 'high_vol':
        # High vol: rely more on POI, vol ratio (mean reversion cues)
        reg_weights = {
            'ret_5d': 0.05, 'ret_10d': 0.05, 'ret_20d': 0.05,
            'oi_chg_5d': 0.10, 'poi_signal': 0.30,
            'spread_pct': 0.15, 'realized_vol_20d': 0.15, 'vol_ratio': 0.15,
        }
    elif regime == 'low_vol':
        # Low vol: momentum works better
        reg_weights = {
            'ret_5d': 0.15, 'ret_10d': 0.15, 'ret_20d': 0.15,
            'oi_chg_5d': 0.10, 'poi_signal': 0.15,
            'spread_pct': 0.10, 'realized_vol_20d': 0.10, 'vol_ratio': 0.10,
        }
    else:
        # Normal: balanced
        reg_weights = {c: 1.0 / len(factor_cols) for c in factor_cols}

    score = pd.Series(0.0, index=df.index)
    for col in factor_cols:
        rank_col = f'{col}_rank'
        w = reg_weights.get(col, 1.0 / len(factor_cols))
        score += w * df[rank_col].fillna(0.5)
    return score


# =============================================================================
# 4. Backtesting engine
# =============================================================================

def backtest_strategy(panel, factors, factor_cols, scheme, K, rebalance_days,
                      start_date, end_date, weights=None, sl_pct=-0.02, tp_pct=0.05):
    """
    Run a long-short backtest.

    Parameters
    ----------
    scheme : str
        'equal', 'optimized', or 'conditional'
    K : int
        Number of positions (long top K, short bottom K)
    rebalance_days : int
        Rebalance frequency in trading days
    start_date, end_date : str
        Period boundaries
    weights : dict or None
        Factor weights for 'optimized' scheme
    sl_pct, tp_pct : float
        Stop-loss and take-profit thresholds

    Returns
    -------
    dict with performance metrics and equity curve
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # Get trading dates in range
    dates_in_range = sorted(factors[
        (factors['trade_date'] >= start) & (factors['trade_date'] <= end)
    ]['trade_date'].unique())

    if len(dates_in_range) < 20:
        return None

    # Generate rebalance dates
    rebalance_dates = dates_in_range[::rebalance_days]

    # Prepare price lookup: symbol -> {date: close}
    price_lookup = {}
    for sym in panel['symbol'].unique():
        sym_data = panel[panel['symbol'] == sym][['trade_date', 'close']].set_index('trade_date')
        price_lookup[sym] = sym_data['close'].to_dict()

    equity = 1.0
    equity_curve = []
    positions = {}  # symbol -> {'side': 'long'/'short', 'entry_price': float, 'prev_price': float}
    all_trades = []
    regime_counts = defaultdict(int)

    # Precompute previous-day close lookup for daily MTM
    all_dates_sorted = sorted(panel['trade_date'].unique())

    for i, dt in enumerate(dates_in_range):
        day_data = factors[factors['trade_date'] == dt].copy()

        # --- Daily MTM: compute PnL from yesterday's close to today's close ---
        daily_pnl = 0.0
        to_close_sltp = []

        for sym, pos in list(positions.items()):
            if sym not in price_lookup or dt not in price_lookup[sym]:
                continue
            cur_price = price_lookup[sym][dt]
            if pd.isna(cur_price) or cur_price <= 0:
                continue
            prev_price = pos['prev_price']
            if prev_price is None or np.isnan(prev_price) or prev_price <= 0:
                pos['prev_price'] = cur_price
                continue

            if pos['side'] == 'long':
                day_ret = (cur_price - prev_price) / prev_price
                total_ret = (cur_price - pos['entry_price']) / pos['entry_price']
            else:
                day_ret = (prev_price - cur_price) / prev_price
                total_ret = (pos['entry_price'] - cur_price) / pos['entry_price']

            daily_pnl += day_ret

            # Check SL/TP based on total return from entry
            if total_ret <= sl_pct or total_ret >= tp_pct:
                to_close_sltp.append(sym)
                all_trades.append({
                    'date': dt, 'symbol': sym, 'side': pos['side'],
                    'entry': pos['entry_price'], 'exit': cur_price,
                    'ret': total_ret,
                    'exit_reason': 'SL' if total_ret <= sl_pct else 'TP'
                })
            else:
                pos['prev_price'] = cur_price

        # Close SL/TP positions
        for sym in to_close_sltp:
            del positions[sym]

        # Apply PnL: equal weight per position slot (2*K total slots)
        pos_weight = 1.0 / (2 * K)
        equity += daily_pnl * pos_weight

        # --- Rebalance if needed ---
        if dt in rebalance_dates:
            # Close remaining positions, record final trade
            for sym, pos in positions.items():
                if sym in price_lookup and dt in price_lookup[sym]:
                    cur_price = price_lookup[sym][dt]
                    if pd.isna(cur_price) or cur_price <= 0:
                        continue
                    entry = pos['entry_price']
                    if pos['side'] == 'long':
                        ret = (cur_price - entry) / entry
                    else:
                        ret = (entry - cur_price) / entry
                    all_trades.append({
                        'date': dt, 'symbol': sym, 'side': pos['side'],
                        'entry': entry, 'exit': cur_price, 'ret': ret,
                        'exit_reason': 'rebalance'
                    })
            positions = {}

            if len(day_data) < K * 2:
                equity_curve.append({'date': dt, 'equity': equity})
                continue

            # Compute composite scores
            if scheme == 'equal':
                day_data['score'] = equal_weight_score(day_data, factor_cols)
            elif scheme == 'optimized':
                day_data['score'] = optimized_weight_score(day_data, factor_cols, weights)
            elif scheme == 'conditional':
                day_data['score'] = conditional_weight_score(day_data, factors, factor_cols, dt)
                regime_counts[regime_detection(factors, dt)] += 1
            else:
                raise ValueError(f"Unknown scheme: {scheme}")

            # Select top K and bottom K (by score)
            valid_scores = day_data.dropna(subset=['score'])
            if len(valid_scores) < K * 2:
                equity_curve.append({'date': dt, 'equity': equity})
                continue

            sorted_data = valid_scores.sort_values('score', ascending=False)
            long_candidates = sorted_data.head(K)
            short_candidates = sorted_data.tail(K)

            # Open new positions
            for _, row in long_candidates.iterrows():
                sym = row['symbol']
                if sym in price_lookup and dt in price_lookup[sym]:
                    price = price_lookup[sym][dt]
                    if not pd.isna(price) and price > 0:
                        positions[sym] = {'side': 'long', 'entry_price': price, 'prev_price': price}

            for _, row in short_candidates.iterrows():
                sym = row['symbol']
                if sym in price_lookup and dt in price_lookup[sym]:
                    price = price_lookup[sym][dt]
                    if not pd.isna(price) and price > 0:
                        positions[sym] = {'side': 'short', 'entry_price': price, 'prev_price': price}

        equity_curve.append({'date': dt, 'equity': equity})

    # Compute performance metrics
    ec = pd.DataFrame(equity_curve)
    if len(ec) < 2:
        return None

    ec['daily_ret'] = ec['equity'].pct_change().fillna(0)
    total_ret = ec['equity'].iloc[-1] / ec['equity'].iloc[0] - 1

    # Annualized return and vol
    n_days = len(ec)
    ann_ret = (1 + total_ret) ** (252 / max(n_days, 1)) - 1
    ann_vol = ec['daily_ret'].std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    ec['peak'] = ec['equity'].cummax()
    ec['dd'] = (ec['equity'] - ec['peak']) / ec['peak']
    max_dd = ec['dd'].min()

    # Win rate: count trades that exited with positive return
    if all_trades:
        trade_rets = [t['ret'] for t in all_trades]
        sl_tp_trades = [t['ret'] for t in all_trades if t['exit_reason'] in ('SL', 'TP')]
        # Win rate over all trades (positive return = win)
        wins = sum(1 for r in trade_rets if r > 0)
        win_rate = wins / len(trade_rets) if trade_rets else 0
        avg_trade = np.mean(trade_rets)
        n_trades = len(all_trades)
        # SL/TP specific stats
        n_sl = sum(1 for t in all_trades if t['exit_reason'] == 'SL')
        n_tp = sum(1 for t in all_trades if t['exit_reason'] == 'TP')
        sl_tp_win = sum(1 for r in sl_tp_trades if r > 0) / len(sl_tp_trades) if sl_tp_trades else 0
    else:
        win_rate = 0
        avg_trade = 0
        n_trades = 0
        n_sl = 0
        n_tp = 0
        sl_tp_win = 0

    return {
        'scheme': scheme, 'K': K, 'rebalance_days': rebalance_days,
        'total_return': total_ret, 'ann_return': ann_ret, 'ann_vol': ann_vol,
        'sharpe': sharpe, 'max_dd': max_dd, 'win_rate': win_rate,
        'avg_trade': avg_trade, 'n_trades': n_trades,
        'n_sl': n_sl, 'n_tp': n_tp, 'sl_tp_win': sl_tp_win,
        'equity_curve': ec, 'regime_counts': dict(regime_counts),
        'period': f"{start_date} to {end_date}",
    }


# =============================================================================
# 5. Factor correlation analysis
# =============================================================================

def print_factor_correlation(factors, factor_cols, period_start, period_end):
    """Print factor correlation matrix for a given period."""
    sub = factors[
        (factors['trade_date'] >= pd.Timestamp(period_start)) &
        (factors['trade_date'] <= pd.Timestamp(period_end))
    ]
    corr = sub[factor_cols].corr()

    print("\n" + "=" * 80)
    print(f"FACTOR CORRELATION MATRIX ({period_start} to {period_end})")
    print("=" * 80)

    # Print formatted correlation matrix
    names = [c[:12] for c in factor_cols]
    header = f"{'Factor':<14}" + "".join(f"{n:>12}" for n in names)
    print(header)
    print("-" * len(header))

    for i, col in enumerate(factor_cols):
        row_str = f"{names[i]:<14}"
        for j, col2 in enumerate(factor_cols):
            val = corr.iloc[i, j]
            row_str += f"{val:>12.3f}"
        print(row_str)

    return corr


# =============================================================================
# 6. Main execution
# =============================================================================

def main():
    print("=" * 80)
    print("MULTI-FACTOR ENSEMBLE STRATEGY BACKTEST")
    print("Chinese Commodity Futures")
    print("=" * 80)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Load data ---
    print("\n[1/7] Loading futures data...")
    panel = load_all_futures()
    print(f"  Loaded {len(panel):,} rows, {panel['symbol'].nunique()} commodities")
    print(f"  Date range: {panel['trade_date'].min().date()} to {panel['trade_date'].max().date()}")

    print("\n[2/7] Loading term structure data...")
    ts_df = load_term_structure()
    if not ts_df.empty:
        print(f"  Loaded {len(ts_df):,} term structure records")
        print(f"  Symbols: {ts_df['symbol'].nunique()}, dates: {ts_df['trade_date'].nunique()}")
    else:
        print("  No term structure data loaded")

    # --- Compute factors ---
    print("\n[3/7] Computing factors...")
    factors = compute_factors(panel, ts_df)
    print(f"  Factor panel: {len(factors):,} rows")

    FACTOR_COLS = [
        'ret_5d', 'ret_10d', 'ret_20d',
        'oi_chg_5d', 'poi_signal',
        'spread_pct',
        'realized_vol_20d', 'vol_ratio',
    ]

    # For realized_vol_20d and spread_pct: invert direction for rank meaning
    # Higher vol = riskier, so we want LOW vol -> higher rank (better)
    # Higher spread (contango) = carry cost for longs, so LOW spread -> higher rank
    # We handle this by negating before ranking
    factors['realized_vol_20d'] = -factors['realized_vol_20d']
    factors['spread_pct'] = -factors['spread_pct']

    print("\n  Factor summary (training period 2021-2023):")
    train_factors = factors[
        (factors['trade_date'] >= '2021-01-01') & (factors['trade_date'] <= '2023-12-31')
    ]
    for col in FACTOR_COLS:
        orig_col = col.replace('realized_vol_20d', '-realized_vol_20d').replace('spread_pct', '-spread_pct')
        vals = train_factors[col].dropna()
        print(f"    {col:<20s} mean={vals.mean():>10.4f}  std={vals.std():>10.4f}  "
              f"min={vals.min():>10.4f}  max={vals.max():>10.4f}  n={len(vals):>6d}")

    # --- Factor correlation ---
    # Use original (non-negated) for display
    display_cols = list(FACTOR_COLS)
    display_factors = factors.copy()
    display_factors['realized_vol_20d'] = -display_factors['realized_vol_20d']
    display_factors['spread_pct'] = -display_factors['spread_pct']
    corr = print_factor_correlation(display_factors, display_cols, '2021-01-01', '2023-12-31')

    # --- Cross-sectional ranking ---
    print("\n[4/7] Computing cross-sectional ranks...")
    ranked = cross_sectional_ranks(factors, FACTOR_COLS)
    factors = ranked  # replace with ranked version
    print(f"  Ranked panel: {len(factors):,} rows")

    # --- Compute IC-based weights ---
    print("\n[5/7] Computing IC-based factor weights (training: 2021-2023)...")
    try:
        ic_weights, ic_values = compute_ic_weights(factors, FACTOR_COLS, '2021-01-01', '2023-12-31')
        print("\n  Factor IC Analysis (mean |IC| over monthly periods):")
        print(f"  {'Factor':<20s} {'Mean|IC|':>10s} {'Weight':>10s}")
        print("  " + "-" * 42)
        for col in FACTOR_COLS:
            print(f"  {col:<20s} {ic_values.get(col, 0):>10.4f} {ic_weights.get(col, 0):>10.4f}")
    except ImportError:
        print("  [WARN] scipy not available, using equal weights for optimized scheme")
        ic_weights = {c: 1.0 / len(FACTOR_COLS) for c in FACTOR_COLS}

    # --- Run backtests ---
    print("\n[6/7] Running backtests...")

    schemes = ['equal', 'optimized', 'conditional']
    K_values = [5, 10, 15]
    rebalance_values = [5, 10]

    periods = {
        'train': ('2021-01-01', '2023-12-31'),
        'validate': ('2024-01-01', '2024-12-31'),
        'test': ('2025-01-01', '2026-05-21'),
    }

    all_results = []

    total_configs = len(schemes) * len(K_values) * len(rebalance_values) * len(periods)
    config_idx = 0

    for scheme in schemes:
        for K in K_values:
            for rebal in rebalance_values:
                for period_name, (pstart, pend) in periods.items():
                    config_idx += 1
                    if config_idx % 20 == 0 or config_idx == total_configs:
                        print(f"  Config {config_idx}/{total_configs}...")

                    result = backtest_strategy(
                        panel, factors, FACTOR_COLS,
                        scheme=scheme, K=K, rebalance_days=rebal,
                        start_date=pstart, end_date=pend,
                        weights=ic_weights,
                        sl_pct=-0.02, tp_pct=0.05,
                    )
                    if result:
                        result['period_name'] = period_name
                        all_results.append(result)

    # --- Display results ---
    print("\n[7/7] Results")
    print("=" * 120)

    # Build results DataFrame
    res_df = pd.DataFrame([{
        'scheme': r['scheme'],
        'K': r['K'],
        'rebal': r['rebalance_days'],
        'period': r['period_name'],
        'total_ret': r['total_return'],
        'ann_ret': r['ann_return'],
        'ann_vol': r['ann_vol'],
        'sharpe': r['sharpe'],
        'max_dd': r['max_dd'],
        'win_rate': r['win_rate'],
        'avg_trade': r['avg_trade'],
        'n_trades': r['n_trades'],
    } for r in all_results])

    # --- Table 1: Train period results ---
    for period_name in ['train', 'validate', 'test']:
        sub = res_df[res_df['period'] == period_name].copy()
        if sub.empty:
            continue
        pstart, pend = periods[period_name]
        print(f"\n{'=' * 120}")
        print(f"  PERIOD: {period_name.upper()} ({pstart} to {pend})")
        print(f"{'=' * 120}")
        print(f"  {'Scheme':<14s} {'K':>3s} {'Rebal':>5s} | "
              f"{'TotalRet':>9s} {'AnnRet':>8s} {'AnnVol':>8s} {'Sharpe':>7s} "
              f"{'MaxDD':>8s} {'WinRate':>8s} {'AvgTrade':>9s} {'#Trades':>8s}")
        print("  " + "-" * 108)

        sub = sub.sort_values('sharpe', ascending=False)
        for _, row in sub.iterrows():
            print(f"  {row['scheme']:<14s} {int(row['K']):>3d} {int(row['rebal']):>5d} | "
                  f"{row['total_ret']:>8.2%} {row['ann_ret']:>7.2%} {row['ann_vol']:>7.2%} "
                  f"{row['sharpe']:>7.3f} {row['max_dd']:>7.2%} "
                  f"{row['win_rate']:>7.1%} {row['avg_trade']:>8.4f} {int(row['n_trades']):>8d}")

    # --- Best configurations (by test Sharpe) ---
    print(f"\n{'=' * 120}")
    print("  TOP 10 CONFIGURATIONS BY TEST PERIOD SHARPE")
    print(f"{'=' * 120}")

    test_results = res_df[res_df['period'] == 'test'].copy()
    if not test_results.empty:
        top10 = test_results.nlargest(10, 'sharpe')
        print(f"  {'Scheme':<14s} {'K':>3s} {'Rebal':>5s} | "
              f"{'TotalRet':>9s} {'AnnRet':>8s} {'AnnVol':>8s} {'Sharpe':>7s} "
              f"{'MaxDD':>8s} {'WinRate':>8s} {'#Trades':>8s}")
        print("  " + "-" * 100)
        for _, row in top10.iterrows():
            print(f"  {row['scheme']:<14s} {int(row['K']):>3d} {int(row['rebal']):>5d} | "
                  f"{row['total_ret']:>8.2%} {row['ann_ret']:>7.2%} {row['ann_vol']:>7.2%} "
                  f"{row['sharpe']:>7.3f} {row['max_dd']:>7.2%} "
                  f"{row['win_rate']:>7.1%} {int(row['n_trades']):>8d}")

    # --- Walk-forward analysis ---
    print(f"\n{'=' * 120}")
    print("  WALK-FORWARD ANALYSIS")
    print(f"{'=' * 120}")
    print("  Comparing best config from training/validation performance on test data")

    # Find best config per scheme from validation period
    for scheme in schemes:
        val_sub = res_df[(res_df['period'] == 'validate') & (res_df['scheme'] == scheme)]
        if val_sub.empty:
            continue
        best_val = val_sub.nlargest(1, 'sharpe').iloc[0]
        best_K = int(best_val['K'])
        best_rebal = int(best_val['rebal'])

        # Look up test result for this config
        test_match = res_df[
            (res_df['period'] == 'test') &
            (res_df['scheme'] == scheme) &
            (res_df['K'] == best_K) &
            (res_df['rebal'] == best_rebal)
        ]
        train_match = res_df[
            (res_df['period'] == 'train') &
            (res_df['scheme'] == scheme) &
            (res_df['K'] == best_K) &
            (res_df['rebal'] == best_rebal)
        ]

        print(f"\n  Scheme: {scheme.upper()}")
        print(f"    Best validation config: K={best_K}, rebalance={best_rebal}d")
        if not train_match.empty:
            tr = train_match.iloc[0]
            print(f"    Train:   Sharpe={tr['sharpe']:.3f}  Ret={tr['total_ret']:.2%}  MDD={tr['max_dd']:.2%}  WR={tr['win_rate']:.1%}")
        vr = best_val
        print(f"    Valid:   Sharpe={vr['sharpe']:.3f}  Ret={vr['total_ret']:.2%}  MDD={vr['max_dd']:.2%}  WR={vr['win_rate']:.1%}")
        if not test_match.empty:
            te = test_match.iloc[0]
            print(f"    Test:    Sharpe={te['sharpe']:.3f}  Ret={te['total_ret']:.2%}  MDD={te['max_dd']:.2%}  WR={te['win_rate']:.1%}")
            # Decay ratio
            if vr['sharpe'] > 0:
                decay = te['sharpe'] / vr['sharpe']
                print(f"    Sharpe decay (test/val): {decay:.2f}")
        else:
            print(f"    Test:    No data")

    # --- Composite weights summary ---
    print(f"\n{'=' * 120}")
    print("  COMPOSITE SCORE WEIGHTS SUMMARY")
    print(f"{'=' * 120}")
    print(f"  {'Factor':<20s} {'IC-Weight':>10s} {'Equal-Wt':>10s} {'Cond-HighVol':>13s} {'Cond-LowVol':>13s} {'Cond-Normal':>13s}")
    print("  " + "-" * 80)
    cond_high = {'ret_5d': 0.05, 'ret_10d': 0.05, 'ret_20d': 0.05,
                 'oi_chg_5d': 0.10, 'poi_signal': 0.30,
                 'spread_pct': 0.15, 'realized_vol_20d': 0.15, 'vol_ratio': 0.15}
    cond_low = {'ret_5d': 0.15, 'ret_10d': 0.15, 'ret_20d': 0.15,
                'oi_chg_5d': 0.10, 'poi_signal': 0.15,
                'spread_pct': 0.10, 'realized_vol_20d': 0.10, 'vol_ratio': 0.10}
    ew = 1.0 / len(FACTOR_COLS)
    for col in FACTOR_COLS:
        print(f"  {col:<20s} {ic_weights.get(col, 0):>10.4f} {ew:>10.4f} "
              f"{cond_high.get(col, 0):>13.4f} {cond_low.get(col, 0):>13.4f} {ew:>13.4f}")

    # --- Regime analysis ---
    print(f"\n{'=' * 120}")
    print("  REGIME DISTRIBUTION (Conditional scheme)")
    print(f"{'=' * 120}")
    cond_results = [r for r in all_results if r['scheme'] == 'conditional' and r.get('regime_counts')]
    if cond_results:
        total_regime = defaultdict(int)
        for r in cond_results:
            for regime, count in r['regime_counts'].items():
                total_regime[regime] += count
        total = sum(total_regime.values())
        if total > 0:
            for regime in ['high_vol', 'normal', 'low_vol']:
                count = total_regime.get(regime, 0)
                print(f"    {regime:<15s}: {count:>6d} ({count/total:>6.1%})")

    # --- Equity curve comparison (best per scheme) ---
    print(f"\n{'=' * 120}")
    print("  EQUITY CURVE SNAPSHOT (Test period, best per scheme)")
    print(f"{'=' * 120}")
    for scheme in schemes:
        scheme_test = [r for r in all_results
                       if r['scheme'] == scheme and r['period_name'] == 'test']
        if not scheme_test:
            continue
        best = max(scheme_test, key=lambda x: x['sharpe'])
        ec = best['equity_curve']
        # Print quarterly snapshots
        ec['quarter'] = ec['date'].dt.to_period('Q')
        quarterly = ec.groupby('quarter')['equity'].last()
        print(f"\n  {scheme.upper()} (K={best['K']}, rebal={best['rebalance_days']}d):")
        for q, eq in quarterly.items():
            ret_q = (eq - 1.0)
            print(f"    {q}: equity={eq:.4f} (return={ret_q:+.2%})")

    # --- Final summary ---
    print(f"\n{'=' * 120}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 120}")

    # Overall best test Sharpe
    test_all = [r for r in all_results if r['period_name'] == 'test']
    if test_all:
        best_overall = max(test_all, key=lambda x: x['sharpe'])
        print(f"\n  Best test configuration:")
        print(f"    Scheme:       {best_overall['scheme']}")
        print(f"    K:            {best_overall['K']}")
        print(f"    Rebalance:    {best_overall['rebalance_days']} days")
        print(f"    Total Return: {best_overall['total_return']:.2%}")
        print(f"    Ann Return:   {best_overall['ann_return']:.2%}")
        print(f"    Ann Vol:      {best_overall['ann_vol']:.2%}")
        print(f"    Sharpe:       {best_overall['sharpe']:.3f}")
        print(f"    Max Drawdown: {best_overall['max_dd']:.2%}")
        print(f"    Win Rate:     {best_overall['win_rate']:.1%}")
        print(f"    Avg Trade:    {best_overall['avg_trade']:.4f}")
        print(f"    # Trades:     {best_overall['n_trades']}")

    # Best per scheme on test
    print(f"\n  Best by scheme (test period):")
    print(f"  {'Scheme':<14s} {'K':>3s} {'Reb':>4s} {'Sharpe':>7s} {'AnnRet':>8s} {'MaxDD':>8s} {'WR':>7s}")
    print("  " + "-" * 55)
    for scheme in schemes:
        s_test = [r for r in test_all if r['scheme'] == scheme]
        if s_test:
            best_s = max(s_test, key=lambda x: x['sharpe'])
            print(f"  {scheme:<14s} {best_s['K']:>3d} {best_s['rebalance_days']:>4d} "
                  f"{best_s['sharpe']:>7.3f} {best_s['ann_return']:>7.2%} "
                  f"{best_s['max_dd']:>7.2%} {best_s['win_rate']:>6.1%}")

    print(f"\n{'=' * 120}")
    print("  BACKTEST COMPLETE")
    print(f"{'=' * 120}")


if __name__ == '__main__':
    main()
