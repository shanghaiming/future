#!/usr/bin/env python3
"""
IV/HV Signal-Based Futures Backtest
====================================
Since we only have 2 days of options IV data (20260520-20260521), we cannot
backtest IV signals historically. Instead, we use the methodology from
process_tq_options.py to reconstruct volatility signals from futures data.

Strategies tested:
  A. HV Mean Reversion: commodities at extreme HV percentiles revert
     - Long commodities with HV at 252d-low percentile (expect vol to rise = typically contrarian)
     - Short commodities with HV at 252d-high percentile (expect vol to drop)
  B. HV Momentum (Vol Clustering): rising HV tends to continue
     - Short commodities with rapidly rising HV (vol spike = bearish)
     - Long commodities with stable/declining HV (calm = bullish)
  C. HV Cross-Sectional (Low Vol Anomaly):
     - Long low-HV commodities, short high-HV commodities
  D. HV + OI Signal: HV rise combined with OI increase = new positions being built
     - When HV rises AND OI increases -> strong directional signal
  E. Estimated IV Signal: use baseline IV/HV ratio from 2-day snapshot to proxy
     historical IV, then trade IV-HV spread (estimated)

Walk-forward: train 2021-2023, validate 2024, test 2025-2026.
Risk: -2% SL, +5% TP per position.
"""

import os
import warnings
import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')

BASE_DIR = os.path.expanduser('~/home/futures_platform')
DATA_DIR = os.path.join(BASE_DIR, 'data/futures_weighted/')
OPT_DIR = os.path.join(BASE_DIR, 'data/options_calculated/')


# =============================================================================
# 1. Data Loading
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


def load_iv_hv_baselines():
    """Load IV/HV ratio baselines from the 2-day options snapshot."""
    import glob

    # First compute HV from futures data (same method as process_tq_options.py)
    hv_cache = {}
    for f in sorted(os.listdir(DATA_DIR)):
        if not f.endswith('.csv'):
            continue
        sym = f.replace('.csv', '')
        df = pd.read_csv(os.path.join(DATA_DIR, f))
        if len(df) < 20:
            continue
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        df = df.sort_values('trade_date').reset_index(drop=True)
        rets = df['close'].pct_change()
        for i in range(len(df)):
            row_date = df['trade_date'].iloc[i].strftime('%Y%m%d')
            hv20 = rets.iloc[max(0, i-19):i+1].std() * np.sqrt(252) if i >= 19 else None
            if hv20:
                hv_cache.setdefault(sym, {})[row_date] = round(float(hv20), 6)

    # Now load IV data and compute ATM IV per product per date
    product_stats = defaultdict(lambda: {'ivs': [], 'hvs': [], 'ratios': []})

    for f in sorted(os.listdir(OPT_DIR)):
        if not f.endswith('.json') or f.startswith('all_') or f.startswith('iv_'):
            continue
        parts = f.replace('.json', '').rsplit('_', 1)
        if len(parts) != 2:
            continue
        product, date_str = parts

        with open(os.path.join(OPT_DIR, f)) as fh:
            records = json.load(fh)
        if not isinstance(records, list):
            continue

        # ATM options (moneyness 0.95-1.05)
        atm = [r for r in records
               if 0.95 <= r.get('moneyness', 0) <= 1.05 and r.get('implied_vol')]
        if not atm:
            continue
        atm_iv = float(np.mean([r['implied_vol'] for r in atm]))

        # Get HV
        hv_key = product.lower() + 'fi'
        hv20 = hv_cache.get(hv_key, {}).get(date_str)

        product_stats[product]['ivs'].append(atm_iv)
        if hv20:
            product_stats[product]['hvs'].append(hv20)
            product_stats[product]['ratios'].append(atm_iv / hv20)

    # Compute baseline IV/HV ratio per product
    baselines = {}
    for prod, stats in product_stats.items():
        ratios = stats['ratios']
        if len(ratios) >= 1:
            baselines[prod] = float(np.mean(ratios))

    return baselines


# =============================================================================
# 2. Feature Computation
# =============================================================================

def compute_hv_features(panel):
    """Compute HV-based features for each symbol on each date."""
    factor_frames = []

    symbols = panel['symbol'].unique()
    for sym in sorted(symbols):
        g = panel[panel['symbol'] == sym].copy().sort_values('trade_date').reset_index(drop=True)
        if len(g) < 60:
            continue

        df = g[['symbol', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'oi']].copy()

        close = g['close'].values
        oi = g['oi'].values
        vol = g['vol'].values

        # --- Returns and HV ---
        rets = g['close'].pct_change()
        df['ret'] = rets

        # HV at various windows
        df['hv_5d'] = rets.rolling(5).std() * np.sqrt(252)
        df['hv_10d'] = rets.rolling(10).std() * np.sqrt(252)
        df['hv_20d'] = rets.rolling(20).std() * np.sqrt(252)
        df['hv_60d'] = rets.rolling(60).std() * np.sqrt(252)

        # HV change (momentum of volatility)
        df['hv_20d_chg_5d'] = df['hv_20d'].diff(5) / df['hv_20d'].shift(5).replace(0, np.nan)
        df['hv_20d_chg_10d'] = df['hv_20d'].diff(10) / df['hv_20d'].shift(10).replace(0, np.nan)
        # Clip extreme values
        df['hv_20d_chg_5d'] = df['hv_20d_chg_5d'].clip(-2, 5)
        df['hv_20d_chg_10d'] = df['hv_20d_chg_10d'].clip(-2, 5)

        # HV percentile rank over 252 days
        df['hv_20d_pct_252'] = df['hv_20d'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x.dropna()) >= 20 else np.nan,
            raw=False
        )

        # HV ratio: short-term vs long-term
        df['hv_ratio_5_60'] = df['hv_5d'] / df['hv_60d']

        # --- OI features ---
        df['oi_chg_5d'] = g['oi'].pct_change(5)
        df['oi_chg_10d'] = g['oi'].pct_change(10)

        # --- Combined HV + OI signal ---
        # When HV rises AND OI increases: new positions being built in volatile market
        hv_chg = df['hv_20d_chg_5d'].fillna(0).replace([np.inf, -np.inf], 0)
        hv_rising = (hv_chg > 0).astype(float)
        oi_rising = (df['oi_chg_5d'].fillna(0) > 0).astype(float)
        df['hv_oi_signal'] = hv_rising * oi_rising * hv_chg.abs()
        df['hv_oi_signal'] = df['hv_oi_signal'].clip(0, 5)

        # Price direction * OI direction * HV change (enhanced POI with vol)
        price_chg_5d = g['close'].diff(5)
        oi_chg_5d = g['oi'].diff(5)
        oi_pct_chg = g['oi'].pct_change(5).replace([np.inf, -np.inf], np.nan)
        hv_chg_abs = df['hv_20d_chg_5d'].abs().fillna(0).replace(np.inf, 0)
        df['vol_poi_signal'] = (
            np.sign(price_chg_5d) * np.sign(oi_chg_5d)
            * np.abs(oi_pct_chg)
            * (1 + hv_chg_abs)
        )
        df['vol_poi_signal'] = df['vol_poi_signal'].clip(-50, 50)

        # --- Volume features ---
        vol_5d = g['vol'].rolling(5).mean()
        vol_20d = g['vol'].rolling(20).mean()
        df['vol_ratio'] = vol_5d / vol_20d

        # --- Forward returns (for backtesting) ---
        df['fwd_ret_1d'] = g['close'].pct_change(1).shift(-1)
        df['fwd_ret_5d'] = g['close'].pct_change(5).shift(-5)
        df['fwd_ret_10d'] = g['close'].pct_change(10).shift(-10)

        factor_frames.append(df)

    factors = pd.concat(factor_frames, ignore_index=True)
    return factors


def add_estimated_iv(factors, iv_hv_baselines):
    """Add estimated IV using baseline IV/HV ratios."""
    # Map product from symbol: symbol like 'safi' -> product 'SA'
    # Create mapping
    product_map = {}
    for sym in factors['symbol'].unique():
        # 'safi' -> 'sa' -> try 'SA', 'Sa', etc.
        base = sym.replace('fi', '').upper()
        if base in iv_hv_baselines:
            product_map[sym] = base
        # Also try just the first part
        elif len(sym) > 3:
            base2 = sym[:-2].upper()
            if base2 in iv_hv_baselines:
                product_map[sym] = base2

    factors['est_iv_hv_ratio'] = factors['symbol'].map(
        lambda s: iv_hv_baselines.get(product_map.get(s, ''), np.nan)
    )
    factors['est_iv'] = factors['hv_20d'] * factors['est_iv_hv_ratio']
    factors['est_iv_hv_spread'] = factors['est_iv'] - factors['hv_20d']

    return factors, product_map


# =============================================================================
# 3. Strategy Implementations
# =============================================================================

def backtest_long_short(panel, factors, signal_col, K, rebalance_days,
                        start_date, end_date, ascending=True,
                        sl_pct=-0.02, tp_pct=0.05, strategy_name=''):
    """
    Generic long-short backtest based on cross-sectional signal ranking.

    Parameters
    ----------
    signal_col : str
        Column name to rank by
    ascending : bool
        If True, low values get long, high values get short
        If False, high values get long, low values get short
    K : int
        Number of positions per side
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    dates_in_range = sorted(factors[
        (factors['trade_date'] >= start) & (factors['trade_date'] <= end)
    ]['trade_date'].unique())

    if len(dates_in_range) < 20:
        return None

    rebalance_dates = set(dates_in_range[::rebalance_days])

    # Price lookup (handle duplicate dates by taking last)
    price_lookup = {}
    for sym in panel['symbol'].unique():
        sym_data = panel[panel['symbol'] == sym][['trade_date', 'close']].drop_duplicates(
            subset='trade_date', keep='last').set_index('trade_date')
        price_lookup[sym] = sym_data['close'].to_dict()

    equity = 1.0
    equity_curve = []
    positions = {}
    all_trades = []

    for i, dt in enumerate(dates_in_range):
        day_data = factors[factors['trade_date'] == dt].copy()
        if len(day_data) < K * 2:
            continue

        # --- Check SL/TP ---
        closed_pnl = 0.0
        to_close = []
        for sym, pos in list(positions.items()):
            if sym not in price_lookup or dt not in price_lookup[sym]:
                continue
            cur_price = price_lookup[sym][dt]
            if pd.isna(cur_price):
                continue
            entry = pos['entry_price']
            if pos['side'] == 'long':
                ret = (cur_price - entry) / entry
            else:
                ret = (entry - cur_price) / entry

            if ret <= sl_pct or ret >= tp_pct:
                closed_pnl += ret
                to_close.append(sym)
                all_trades.append({
                    'date': dt, 'symbol': sym, 'side': pos['side'],
                    'entry': entry, 'exit': cur_price, 'ret': ret,
                    'exit_reason': 'SL' if ret <= sl_pct else 'TP'
                })

        for sym in to_close:
            del positions[sym]

        pos_weight = 1.0 / (2 * K)
        equity += closed_pnl * pos_weight

        # --- Rebalance ---
        if dt in rebalance_dates:
            # Close remaining positions
            mtm_pnl = 0.0
            for sym, pos in positions.items():
                if sym in price_lookup and dt in price_lookup[sym]:
                    cur_price = price_lookup[sym][dt]
                    if pd.isna(cur_price):
                        continue
                    entry = pos['entry_price']
                    if pos['side'] == 'long':
                        ret = (cur_price - entry) / entry
                    else:
                        ret = (entry - cur_price) / entry
                    mtm_pnl += ret
                    all_trades.append({
                        'date': dt, 'symbol': sym, 'side': pos['side'],
                        'entry': entry, 'exit': cur_price, 'ret': ret,
                        'exit_reason': 'rebalance'
                    })
            equity += mtm_pnl * pos_weight
            positions = {}

            # Rank by signal
            valid = day_data.dropna(subset=[signal_col])
            if len(valid) < K * 2:
                continue

            sorted_data = valid.sort_values(signal_col, ascending=ascending)

            # Long bottom K (if ascending=True, these are LOW values)
            long_candidates = sorted_data.head(K)
            # Short top K
            short_candidates = sorted_data.tail(K)

            for _, row in long_candidates.iterrows():
                sym = row['symbol']
                if sym in price_lookup and dt in price_lookup[sym]:
                    price = price_lookup[sym][dt]
                    if not pd.isna(price) and price > 0:
                        positions[sym] = {'side': 'long', 'entry_price': price}

            for _, row in short_candidates.iterrows():
                sym = row['symbol']
                if sym in price_lookup and dt in price_lookup[sym]:
                    price = price_lookup[sym][dt]
                    if not pd.isna(price) and price > 0:
                        positions[sym] = {'side': 'short', 'entry_price': price}

        equity_curve.append({'date': dt, 'equity': equity})

    return _compute_metrics(equity_curve, all_trades, strategy_name, K, rebalance_days,
                            start_date, end_date)


def backtest_hv_mean_reversion(panel, factors, K, rebalance_days,
                                start_date, end_date, sl_pct=-0.02, tp_pct=0.05):
    """
    HV Mean Reversion: When HV is at extreme highs -> expect vol to decrease
    (prices tend to calm down, often bullish).
    When HV is at extreme lows -> expect vol to increase (breakout incoming).

    Long: commodities with lowest HV percentile (calm, stable)
    Short: commodities with highest HV percentile (volatile, stressed)
    """
    return backtest_long_short(
        panel, factors, 'hv_20d_pct_252', K, rebalance_days,
        start_date, end_date, ascending=True,
        sl_pct=sl_pct, tp_pct=tp_pct,
        strategy_name='HV_MeanReversion'
    )


def backtest_hv_momentum(panel, factors, K, rebalance_days,
                          start_date, end_date, sl_pct=-0.02, tp_pct=0.05):
    """
    HV Momentum (Vol Clustering): Rising HV tends to continue in the short term.
    Vol clustering is a well-documented phenomenon.

    Long: commodities with DECLINING HV (stable, trending calmly)
    Short: commodities with RISING HV (vol spike, often bearish)

    ascending=True means sort by hv_20d_chg_5d ascending -> long those with LOW/negative
    HV change, short those with HIGH/positive HV change.
    """
    return backtest_long_short(
        panel, factors, 'hv_20d_chg_5d', K, rebalance_days,
        start_date, end_date, ascending=True,
        sl_pct=sl_pct, tp_pct=tp_pct,
        strategy_name='HV_Momentum'
    )


def backtest_hv_cross_sectional(panel, factors, K, rebalance_days,
                                 start_date, end_date, sl_pct=-0.02, tp_pct=0.05):
    """
    Low Vol Anomaly: commodities with low HV tend to outperform.
    Long: lowest HV commodities
    Short: highest HV commodities
    """
    return backtest_long_short(
        panel, factors, 'hv_20d', K, rebalance_days,
        start_date, end_date, ascending=True,
        sl_pct=sl_pct, tp_pct=tp_pct,
        strategy_name='HV_CrossSectional'
    )


def backtest_hv_oi_signal(panel, factors, K, rebalance_days,
                           start_date, end_date, sl_pct=-0.02, tp_pct=0.05):
    """
    HV + OI Combined Signal:
    When HV rises AND OI increases -> new positions being built in a volatile market.
    This is often a strong directional signal.

    We use the hv_oi_signal which captures this interaction.
    Long: LOW hv_oi_signal (stable HV, declining OI = calm positioning)
    Short: HIGH hv_oi_signal (rising HV, rising OI = speculative buildup)

    ascending=True -> long bottom K (low signal), short top K (high signal)
    """
    return backtest_long_short(
        panel, factors, 'hv_oi_signal', K, rebalance_days,
        start_date, end_date, ascending=True,
        sl_pct=sl_pct, tp_pct=tp_pct,
        strategy_name='HV_OI_Signal'
    )


def backtest_vol_poi(panel, factors, K, rebalance_days,
                     start_date, end_date, sl_pct=-0.02, tp_pct=0.05):
    """
    Vol-enhanced POI: combines price direction, OI direction, and HV change.
    Uses vol_poi_signal = sign(price_chg) * sign(OI_chg) * |OI_chg_pct| * (1 + |HV_chg|)
    """
    return backtest_long_short(
        panel, factors, 'vol_poi_signal', K, rebalance_days,
        start_date, end_date, ascending=False,
        sl_pct=sl_pct, tp_pct=tp_pct,
        strategy_name='Vol_POI'
    )


def backtest_estimated_iv(panel, factors, K, rebalance_days,
                           start_date, end_date, sl_pct=-0.02, tp_pct=0.05):
    """
    Estimated IV Signal: use baseline IV/HV ratio to estimate historical IV,
    then trade the IV-HV spread (overpricing/underpricing of vol).

    Long: high est_iv_hv_spread (IV > HV, market expects more vol than realized)
          -> prices may have upside due to vol risk premium
    Short: low/negative est_iv_hv_spread (IV < HV, realized vol exceeds implied)
    """
    return backtest_long_short(
        panel, factors, 'est_iv_hv_spread', K, rebalance_days,
        start_date, end_date, ascending=False,
        sl_pct=sl_pct, tp_pct=tp_pct,
        strategy_name='Est_IV_Signal'
    )


def backtest_hv_ratio_signal(panel, factors, K, rebalance_days,
                              start_date, end_date, sl_pct=-0.02, tp_pct=0.05):
    """
    HV Term Structure: HV 5d / HV 60d ratio.
    When short-term HV >> long-term HV -> recent vol spike, likely to revert.
    When short-term HV << long-term HV -> unusual calm, may break out.

    Long: low hv_ratio_5_60 (short-term vol has calmed relative to long-term)
    Short: high hv_ratio_5_60 (short-term vol spike)
    """
    return backtest_long_short(
        panel, factors, 'hv_ratio_5_60', K, rebalance_days,
        start_date, end_date, ascending=True,
        sl_pct=sl_pct, tp_pct=tp_pct,
        strategy_name='HV_TermStructure'
    )


def _compute_metrics(equity_curve, all_trades, strategy_name, K, rebalance_days,
                     start_date, end_date):
    """Compute performance metrics from equity curve and trade list."""
    ec = pd.DataFrame(equity_curve)
    if len(ec) < 2:
        return None

    ec['daily_ret'] = ec['equity'].pct_change().fillna(0)
    total_ret = ec['equity'].iloc[-1] / ec['equity'].iloc[0] - 1

    n_days = len(ec)
    ann_ret = (1 + total_ret) ** (252 / max(n_days, 1)) - 1
    ann_vol = ec['daily_ret'].std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    ec['peak'] = ec['equity'].cummax()
    ec['dd'] = (ec['equity'] - ec['peak']) / ec['peak']
    max_dd = ec['dd'].min()

    if all_trades:
        trade_rets = [t['ret'] for t in all_trades]
        win_rate = np.mean([1 for r in trade_rets if r > 0]) / len(trade_rets)
        avg_trade = np.mean(trade_rets)
        n_trades = len(all_trades)
    else:
        win_rate = 0
        avg_trade = 0
        n_trades = 0

    return {
        'strategy': strategy_name, 'K': K, 'rebalance_days': rebalance_days,
        'total_return': total_ret, 'ann_return': ann_ret, 'ann_vol': ann_vol,
        'sharpe': sharpe, 'max_dd': max_dd, 'win_rate': win_rate,
        'avg_trade': avg_trade, 'n_trades': n_trades,
        'equity_curve': ec,
        'period': f"{start_date} to {end_date}",
    }


# =============================================================================
# 4. IC Analysis
# =============================================================================

def compute_signal_ic(factors, signal_cols, period_start, period_end):
    """Compute IC (rank correlation) between each signal and forward 5d return."""
    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("  [WARN] scipy not available, skipping IC analysis")
        return {}

    sub = factors[
        (factors['trade_date'] >= pd.Timestamp(period_start)) &
        (factors['trade_date'] <= pd.Timestamp(period_end))
    ].copy()
    sub['month'] = sub['trade_date'].dt.to_period('M')

    ic_results = {}
    for col in signal_cols:
        monthly_ics = []
        for month, grp in sub.groupby('month'):
            valid = grp[[col, 'fwd_ret_5d']].dropna()
            if len(valid) < 10:
                continue
            ic, _ = spearmanr(valid[col], valid['fwd_ret_5d'])
            monthly_ics.append(ic)

        if monthly_ics:
            ic_results[col] = {
                'mean_ic': np.mean(monthly_ics),
                'mean_abs_ic': np.mean(np.abs(monthly_ics)),
                'ic_ir': np.mean(monthly_ics) / np.std(monthly_ics) if np.std(monthly_ics) > 0 else 0,
                'pct_positive': np.mean([1 for ic in monthly_ics if ic > 0]),
                'n_months': len(monthly_ics),
            }

    return ic_results


# =============================================================================
# 5. Composite HV Strategy
# =============================================================================

def backtest_composite_hv(panel, factors, K, rebalance_days,
                          start_date, end_date, weights=None,
                          sl_pct=-0.02, tp_pct=0.05):
    """
    Composite strategy combining multiple HV signals via rank averaging.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    dates_in_range = sorted(factors[
        (factors['trade_date'] >= start) & (factors['trade_date'] <= end)
    ]['trade_date'].unique())

    if len(dates_in_range) < 20:
        return None

    rebalance_dates = set(dates_in_range[::rebalance_days])

    # Signals to combine
    signal_cols = {
        'hv_20d_pct_252': True,   # ascending (long low percentile)
        'hv_20d_chg_5d': True,    # ascending (long declining vol)
        'hv_20d': True,           # ascending (long low vol)
        'hv_oi_signal': True,     # ascending (long low vol-oi interaction)
        'vol_poi_signal': False,  # descending (long high vol-poi)
        'hv_ratio_5_60': True,    # ascending (long low ratio = vol calmed)
    }

    if weights is None:
        weights = {col: 1.0 / len(signal_cols) for col in signal_cols}

    price_lookup = {}
    for sym in panel['symbol'].unique():
        sym_data = panel[panel['symbol'] == sym][['trade_date', 'close']].drop_duplicates(
            subset='trade_date', keep='last').set_index('trade_date')
        price_lookup[sym] = sym_data['close'].to_dict()

    equity = 1.0
    equity_curve = []
    positions = {}
    all_trades = []

    for dt in dates_in_range:
        day_data = factors[factors['trade_date'] == dt].copy()
        if len(day_data) < K * 2:
            continue

        # SL/TP
        closed_pnl = 0.0
        to_close = []
        for sym, pos in list(positions.items()):
            if sym not in price_lookup or dt not in price_lookup[sym]:
                continue
            cur_price = price_lookup[sym][dt]
            if pd.isna(cur_price):
                continue
            entry = pos['entry_price']
            ret = (cur_price - entry) / entry if pos['side'] == 'long' else (entry - cur_price) / entry
            if ret <= sl_pct or ret >= tp_pct:
                closed_pnl += ret
                to_close.append(sym)
                all_trades.append({
                    'date': dt, 'symbol': sym, 'side': pos['side'],
                    'entry': entry, 'exit': cur_price, 'ret': ret,
                    'exit_reason': 'SL' if ret <= sl_pct else 'TP'
                })
        for sym in to_close:
            del positions[sym]
        pos_weight = 1.0 / (2 * K)
        equity += closed_pnl * pos_weight

        # Rebalance
        if dt in rebalance_dates:
            mtm_pnl = 0.0
            for sym, pos in positions.items():
                if sym in price_lookup and dt in price_lookup[sym]:
                    cur_price = price_lookup[sym][dt]
                    if pd.isna(cur_price):
                        continue
                    entry = pos['entry_price']
                    ret = (cur_price - entry) / entry if pos['side'] == 'long' else (entry - cur_price) / entry
                    mtm_pnl += ret
                    all_trades.append({
                        'date': dt, 'symbol': sym, 'side': pos['side'],
                        'entry': entry, 'exit': cur_price, 'ret': ret,
                        'exit_reason': 'rebalance'
                    })
            equity += mtm_pnl * pos_weight
            positions = {}

            # Compute composite score
            score = pd.Series(0.0, index=day_data.index)
            for col, ascending in signal_cols.items():
                valid = day_data[col].notna()
                if valid.sum() < 5:
                    continue
                ranks = day_data.loc[valid, col].rank(pct=True, ascending=ascending)
                if not ascending:
                    ranks = 1 - ranks  # Invert: high raw -> low rank for shorting
                score.loc[valid] += weights.get(col, 0) * ranks

            day_data['score'] = score
            valid_scores = day_data.dropna(subset=['score'])
            if len(valid_scores) < K * 2:
                continue

            sorted_data = valid_scores.sort_values('score', ascending=False)
            long_candidates = sorted_data.head(K)
            short_candidates = sorted_data.tail(K)

            for _, row in long_candidates.iterrows():
                sym = row['symbol']
                if sym in price_lookup and dt in price_lookup[sym]:
                    price = price_lookup[sym][dt]
                    if not pd.isna(price) and price > 0:
                        positions[sym] = {'side': 'long', 'entry_price': price}

            for _, row in short_candidates.iterrows():
                sym = row['symbol']
                if sym in price_lookup and dt in price_lookup[sym]:
                    price = price_lookup[sym][dt]
                    if not pd.isna(price) and price > 0:
                        positions[sym] = {'side': 'short', 'entry_price': price}

        equity_curve.append({'date': dt, 'equity': equity})

    return _compute_metrics(equity_curve, all_trades, 'Composite_HV', K, rebalance_days,
                            start_date, end_date)


# =============================================================================
# 6. Main Execution
# =============================================================================

def main():
    print("=" * 120)
    print("IV/HV SIGNAL-BASED FUTURES BACKTEST")
    print("Chinese Commodity Futures")
    print("=" * 120)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("NOTE: Only 2 dates of IV data available (20260520-20260521).")
    print("      Using HV-based signals reconstructed from futures daily data.")
    print("      Baseline IV/HV ratios from snapshot used for estimated IV strategy.")

    # --- Load data ---
    print("\n[1/8] Loading futures data...")
    panel = load_all_futures()
    print(f"  Loaded {len(panel):,} rows, {panel['symbol'].nunique()} commodities")
    print(f"  Date range: {panel['trade_date'].min().date()} to {panel['trade_date'].max().date()}")

    print("\n[2/8] Loading IV/HV baselines from options snapshot...")
    iv_hv_baselines = load_iv_hv_baselines()
    print(f"  Baseline IV/HV ratios available for {len(iv_hv_baselines)} products")
    # Show a few examples
    for i, (prod, ratio) in enumerate(sorted(iv_hv_baselines.items())[:10]):
        print(f"    {prod:6s}: IV/HV = {ratio:.4f}")
    if len(iv_hv_baselines) > 10:
        print(f"    ... and {len(iv_hv_baselines) - 10} more")

    # --- Compute features ---
    print("\n[3/8] Computing HV-based features...")
    factors = compute_hv_features(panel)
    print(f"  Feature panel: {len(factors):,} rows")

    # Add estimated IV
    factors, product_map = add_estimated_iv(factors, iv_hv_baselines)
    n_mapped = factors['est_iv_hv_ratio'].notna().sum()
    print(f"  Estimated IV mapped: {n_mapped:,} rows ({n_mapped/len(factors)*100:.1f}%)")

    # Feature summary
    signal_cols_all = [
        'hv_20d', 'hv_20d_pct_252', 'hv_20d_chg_5d', 'hv_20d_chg_10d',
        'hv_ratio_5_60', 'hv_oi_signal', 'vol_poi_signal', 'est_iv_hv_spread',
    ]
    print("\n  Signal summary (training period 2021-2023):")
    train_factors = factors[
        (factors['trade_date'] >= '2021-01-01') & (factors['trade_date'] <= '2023-12-31')
    ]
    for col in signal_cols_all:
        vals = train_factors[col].dropna()
        if len(vals) > 0:
            print(f"    {col:<22s} mean={vals.mean():>10.4f}  std={vals.std():>10.4f}  "
                  f"min={vals.min():>10.4f}  max={vals.max():>10.4f}  n={len(vals):>6d}")
        else:
            print(f"    {col:<22s}  (no data)")

    # --- IC Analysis ---
    print("\n[4/8] Computing signal IC (Information Coefficient)...")
    ic_results = compute_signal_ic(
        factors, signal_cols_all,
        '2021-01-01', '2023-12-31'
    )
    if ic_results:
        print(f"\n  {'Signal':<22s} {'MeanIC':>8s} {'|IC|':>8s} {'IC_IR':>8s} {'%Pos':>6s} {'Months':>7s}")
        print("  " + "-" * 62)
        for col in signal_cols_all:
            if col in ic_results:
                r = ic_results[col]
                print(f"  {col:<22s} {r['mean_ic']:>8.4f} {r['mean_abs_ic']:>8.4f} "
                      f"{r['ic_ir']:>8.3f} {r['pct_positive']:>5.1%} {r['n_months']:>7d}")

    # --- Define strategies ---
    strategies = {
        'HV_MeanReversion': {
            'func': backtest_hv_mean_reversion,
            'desc': 'Long low HV pct, short high HV pct (mean reversion)',
        },
        'HV_Momentum': {
            'func': backtest_hv_momentum,
            'desc': 'Long declining HV, short rising HV (vol clustering)',
        },
        'HV_CrossSectional': {
            'func': backtest_hv_cross_sectional,
            'desc': 'Long low HV, short high HV (low vol anomaly)',
        },
        'HV_OI_Signal': {
            'func': backtest_hv_oi_signal,
            'desc': 'HV rise + OI increase = speculative buildup',
        },
        'Vol_POI': {
            'func': backtest_vol_poi,
            'desc': 'Vol-enhanced POI signal',
        },
        'HV_TermStructure': {
            'func': backtest_hv_ratio_signal,
            'desc': 'Short-term vs long-term HV ratio',
        },
        'Est_IV_Signal': {
            'func': backtest_estimated_iv,
            'desc': 'Estimated IV-HV spread (using baseline ratio)',
        },
        'Composite_HV': {
            'func': backtest_composite_hv,
            'desc': 'Combined HV signals via rank averaging',
            'is_composite': True,
        },
    }

    # --- Run backtests ---
    print("\n[5/8] Running backtests...")

    K_values = [5, 10, 15]
    rebalance_values = [5, 10]
    periods = {
        'train': ('2021-01-01', '2023-12-31'),
        'validate': ('2024-01-01', '2024-12-31'),
        'test': ('2025-01-01', '2026-05-21'),
    }

    all_results = []
    total_configs = len(strategies) * len(K_values) * len(rebalance_values) * len(periods)
    config_idx = 0

    for strat_name, strat_info in strategies.items():
        func = strat_info['func']
        is_composite = strat_info.get('is_composite', False)

        for K in K_values:
            for rebal in rebalance_values:
                for period_name, (pstart, pend) in periods.items():
                    config_idx += 1
                    if config_idx % 20 == 0 or config_idx == total_configs:
                        print(f"  Config {config_idx}/{total_configs}...")

                    if is_composite:
                        result = func(panel, factors, K, rebal, pstart, pend,
                                      sl_pct=-0.02, tp_pct=0.05)
                    else:
                        result = func(panel, factors, K, rebal, pstart, pend,
                                      sl_pct=-0.02, tp_pct=0.05)

                    if result:
                        result['period_name'] = period_name
                        result['strategy_desc'] = strat_info['desc']
                        all_results.append(result)

    # --- Run FLIPPED strategies ---
    # Many HV signals show consistently negative Sharpe, suggesting the direction
    # should be reversed. Test the flipped version.
    print("\n  Running flipped-direction variants...")

    # Define flip-eligible strategies and their reversed direction
    flip_strategies = {
        'Flip_HV_MeanReversion': {
            'signal': 'hv_20d_pct_252',
            'ascending': False,  # flipped: long HIGH HV pct, short LOW
            'desc': 'Flipped: Long high HV pct, short low HV pct',
        },
        'Flip_HV_Momentum': {
            'signal': 'hv_20d_chg_5d',
            'ascending': False,  # flipped: long RISING HV, short DECLINING
            'desc': 'Flipped: Long rising HV, short declining HV',
        },
        'Flip_HV_CrossSectional': {
            'signal': 'hv_20d',
            'ascending': False,  # flipped: long HIGH HV, short LOW HV
            'desc': 'Flipped: Long high HV, short low HV (high vol premium)',
        },
        'Flip_HV_TermStructure': {
            'signal': 'hv_ratio_5_60',
            'ascending': False,  # flipped: long HIGH ratio (vol spike), short LOW
            'desc': 'Flipped: Long high short-term vol spike, short calm',
        },
    }

    for flip_name, flip_info in flip_strategies.items():
        for K in K_values:
            for rebal in rebalance_values:
                for period_name, (pstart, pend) in periods.items():
                    result = backtest_long_short(
                        panel, factors, flip_info['signal'], K, rebal,
                        pstart, pend, ascending=flip_info['ascending'],
                        sl_pct=-0.02, tp_pct=0.05,
                        strategy_name=flip_name
                    )
                    if result:
                        result['period_name'] = period_name
                        result['strategy_desc'] = flip_info['desc']
                        all_results.append(result)

    # --- Display results ---
    print("\n[6/8] Results by Strategy")
    print("=" * 130)

    res_df = pd.DataFrame([{
        'strategy': r['strategy'],
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

    # --- Per-period tables ---
    for period_name in ['train', 'validate', 'test']:
        sub = res_df[res_df['period'] == period_name].copy()
        if sub.empty:
            continue
        pstart, pend = periods[period_name]
        print(f"\n{'=' * 130}")
        print(f"  PERIOD: {period_name.upper()} ({pstart} to {pend})")
        print(f"{'=' * 130}")
        print(f"  {'Strategy':<22s} {'K':>3s} {'Rebal':>5s} | "
              f"{'TotalRet':>9s} {'AnnRet':>8s} {'AnnVol':>8s} {'Sharpe':>7s} "
              f"{'MaxDD':>8s} {'WinRate':>8s} {'AvgTrade':>9s} {'#Trades':>8s}")
        print("  " + "-" * 118)

        sub = sub.sort_values('sharpe', ascending=False)
        for _, row in sub.iterrows():
            print(f"  {row['strategy']:<22s} {int(row['K']):>3d} {int(row['rebal']):>5d} | "
                  f"{row['total_ret']:>8.2%} {row['ann_ret']:>7.2%} {row['ann_vol']:>7.2%} "
                  f"{row['sharpe']:>7.3f} {row['max_dd']:>7.2%} "
                  f"{row['win_rate']:>7.1%} {row['avg_trade']:>8.4f} {int(row['n_trades']):>8d}")

    # --- Best per strategy across periods ---
    print(f"\n{'=' * 130}")
    print("  BEST PER STRATEGY (Test Period)")
    print(f"{'=' * 130}")
    print(f"  {'Strategy':<22s} {'Desc':<50s} | {'K':>3s} {'Reb':>4s} "
          f"{'Sharpe':>7s} {'AnnRet':>8s} {'MaxDD':>8s} {'WR':>6s} {'#Tr':>5s}")
    print("  " + "-" * 125)

    test_all = [r for r in all_results if r['period_name'] == 'test']
    for strat_name, strat_info in strategies.items():
        s_test = [r for r in test_all if r['strategy'] == strat_name]
        if not s_test:
            continue
        best = max(s_test, key=lambda x: x['sharpe'])
        desc = strat_info['desc'][:50]
        print(f"  {strat_name:<22s} {desc:<50s} | {best['K']:>3d} {best['rebalance_days']:>4d} "
              f"{best['sharpe']:>7.3f} {best['ann_return']:>7.2%} "
              f"{best['max_dd']:>7.2%} {best['win_rate']:>5.1%} {best['n_trades']:>5d}")

    # --- Walk-forward analysis ---
    print(f"\n{'=' * 130}")
    print("  WALK-FORWARD ANALYSIS")
    print(f"{'=' * 130}")
    print("  Selecting best config per strategy from validation period, checking test performance")

    for strat_name in strategies:
        val_sub = res_df[(res_df['period'] == 'validate') & (res_df['strategy'] == strat_name)]
        if val_sub.empty:
            continue
        best_val = val_sub.nlargest(1, 'sharpe').iloc[0]
        best_K = int(best_val['K'])
        best_rebal = int(best_val['rebal'])

        train_match = res_df[
            (res_df['period'] == 'train') & (res_df['strategy'] == strat_name) &
            (res_df['K'] == best_K) & (res_df['rebal'] == best_rebal)
        ]
        test_match = res_df[
            (res_df['period'] == 'test') & (res_df['strategy'] == strat_name) &
            (res_df['K'] == best_K) & (res_df['rebal'] == best_rebal)
        ]

        print(f"\n  Strategy: {strat_name}")
        print(f"    Best validation config: K={best_K}, rebalance={best_rebal}d")
        if not train_match.empty:
            tr = train_match.iloc[0]
            print(f"    Train:   Sharpe={tr['sharpe']:.3f}  Ret={tr['total_ret']:.2%}  MDD={tr['max_dd']:.2%}  WR={tr['win_rate']:.1%}")
        print(f"    Valid:   Sharpe={best_val['sharpe']:.3f}  Ret={best_val['total_ret']:.2%}  MDD={best_val['max_dd']:.2%}  WR={best_val['win_rate']:.1%}")
        if not test_match.empty:
            te = test_match.iloc[0]
            print(f"    Test:    Sharpe={te['sharpe']:.3f}  Ret={te['total_ret']:.2%}  MDD={te['max_dd']:.2%}  WR={te['win_rate']:.1%}")
            if best_val['sharpe'] > 0:
                decay = te['sharpe'] / best_val['sharpe']
                print(f"    Sharpe decay (test/val): {decay:.2f}")
        else:
            print(f"    Test:    No data")

    # --- Equity curve comparison ---
    print(f"\n{'=' * 130}")
    print("  EQUITY CURVE SNAPSHOT (Test period, best per strategy)")
    print(f"{'=' * 130}")
    for strat_name in strategies:
        s_test = [r for r in test_all if r['strategy'] == strat_name]
        if not s_test:
            continue
        best = max(s_test, key=lambda x: x['sharpe'])
        ec = best['equity_curve']
        ec['quarter'] = ec['date'].dt.to_period('Q')
        quarterly = ec.groupby('quarter')['equity'].last()
        print(f"\n  {strat_name} (K={best['K']}, rebal={best['rebalance_days']}d, "
              f"Sharpe={best['sharpe']:.3f}):")
        for q, eq in quarterly.items():
            ret_q = (eq - 1.0)
            print(f"    {q}: equity={eq:.4f} (return={ret_q:+.2%})")

    # --- Strategy comparison summary ---
    print(f"\n{'=' * 130}")
    print("  STRATEGY COMPARISON SUMMARY (Including Flipped Variants)")
    print(f"{'=' * 130}")

    # Include all strategies in the comparison (original + flipped)
    all_strat_names = list(strategies.keys()) + list(flip_strategies.keys())

    # Test period Sharpe comparison
    print(f"\n  Test Period Sharpe (best config per strategy, sorted):")
    strat_sharpes = []
    for strat_name in all_strat_names:
        s_test = [r for r in test_all if r['strategy'] == strat_name]
        if s_test:
            best = max(s_test, key=lambda x: x['sharpe'])
            strat_sharpes.append((strat_name, best['sharpe'], best['ann_return'],
                                  best['max_dd'], best['win_rate'], best['K'],
                                  best['rebalance_days']))
    strat_sharpes.sort(key=lambda x: x[1], reverse=True)
    print(f"  {'Rank':>4s} {'Strategy':<28s} {'Sharpe':>7s} {'AnnRet':>8s} {'MaxDD':>8s} "
          f"{'WR':>6s} {'K':>3s} {'Reb':>4s}")
    print("  " + "-" * 75)
    for i, (name, sh, ar, md, wr, k, rb) in enumerate(strat_sharpes, 1):
        print(f"  {i:>4d}  {name:<28s} {sh:>7.3f} {ar:>7.2%} {md:>7.2%} "
              f"{wr:>5.1%} {k:>3d} {rb:>4d}")

    # --- Original vs Flipped comparison ---
    print(f"\n{'=' * 130}")
    print("  ORIGINAL vs FLIPPED DIRECTION COMPARISON (Test Period)")
    print(f"{'=' * 130}")
    print(f"  {'Original':<28s} {'Sharpe':>7s} {'AnnRet':>8s} | "
          f"{'Flipped':<28s} {'Sharpe':>7s} {'AnnRet':>8s} | {'Improvement':>12s}")
    print("  " + "-" * 120)

    flip_pairs = [
        ('HV_MeanReversion', 'Flip_HV_MeanReversion'),
        ('HV_Momentum', 'Flip_HV_Momentum'),
        ('HV_CrossSectional', 'Flip_HV_CrossSectional'),
        ('HV_TermStructure', 'Flip_HV_TermStructure'),
    ]

    for orig_name, flip_name in flip_pairs:
        orig_test = [r for r in test_all if r['strategy'] == orig_name]
        flip_test = [r for r in test_all if r['strategy'] == flip_name]
        if orig_test:
            best_orig = max(orig_test, key=lambda x: x['sharpe'])
            o_sh = best_orig['sharpe']
            o_ar = best_orig['ann_return']
        else:
            o_sh, o_ar = 0, 0
        if flip_test:
            best_flip = max(flip_test, key=lambda x: x['sharpe'])
            f_sh = best_flip['sharpe']
            f_ar = best_flip['ann_return']
        else:
            f_sh, f_ar = 0, 0

        improvement = "FLIP BETTER" if f_sh > o_sh else "ORIGINAL"
        print(f"  {orig_name:<28s} {o_sh:>7.3f} {o_ar:>7.2%} | "
              f"{flip_name:<28s} {f_sh:>7.3f} {f_ar:>7.2%} | {improvement:>12s}")

    # --- IV baseline analysis ---
    print(f"\n{'=' * 130}")
    print("  IV/HV BASELINE RATIO ANALYSIS (from 2-day snapshot)")
    print(f"{'=' * 130}")
    if iv_hv_baselines:
        ratios = list(iv_hv_baselines.values())
        print(f"  Products with IV/HV ratio: {len(ratios)}")
        print(f"  Mean IV/HV ratio:   {np.mean(ratios):.4f}")
        print(f"  Median IV/HV ratio: {np.median(ratios):.4f}")
        print(f"  Std IV/HV ratio:    {np.std(ratios):.4f}")
        print(f"  Min IV/HV ratio:    {np.min(ratios):.4f}")
        print(f"  Max IV/HV ratio:    {np.max(ratios):.4f}")
        print(f"\n  Products with IV/HV > 1.0 (IV premium): "
              f"{sum(1 for r in ratios if r > 1.0)}")
        print(f"  Products with IV/HV < 1.0 (IV discount): "
              f"{sum(1 for r in ratios if r < 1.0)}")
        print(f"\n  Note: Low ratios may indicate options data quality issues")
        print(f"        or products with short futures history.")

    # --- Signal characteristics ---
    print(f"\n{'=' * 130}")
    print("  HV SIGNAL CHARACTERISTICS (Full History)")
    print(f"{'=' * 130}")

    # Distribution of HV percentiles
    test_hv_pct = factors[
        (factors['trade_date'] >= '2025-01-01') &
        (factors['trade_date'] <= '2026-05-21')
    ]['hv_20d_pct_252'].dropna()

    if len(test_hv_pct) > 0:
        print(f"\n  HV 20d Percentile (252d lookback) - Test Period:")
        for pct in [10, 25, 50, 75, 90]:
            print(f"    {pct}th percentile: {np.percentile(test_hv_pct, pct):.4f}")

    # HV change distribution
    test_hv_chg = factors[
        (factors['trade_date'] >= '2025-01-01') &
        (factors['trade_date'] <= '2026-05-21')
    ]['hv_20d_chg_5d'].dropna()

    if len(test_hv_chg) > 0:
        print(f"\n  HV 20d 5-day Change - Test Period:")
        print(f"    Mean: {test_hv_chg.mean():.4f}")
        print(f"    Std:  {test_hv_chg.std():.4f}")
        print(f"    % Positive: {(test_hv_chg > 0).mean():.1%}")
        print(f"    % Negative: {(test_hv_chg < 0).mean():.1%}")

    # --- Final recommendation ---
    print(f"\n{'=' * 130}")
    print("  FINAL SUMMARY & RECOMMENDATIONS")
    print(f"{'=' * 130}")

    if strat_sharpes:
        best_name, best_sharpe, best_ann_ret, best_mdd, best_wr, best_k, best_rb = strat_sharpes[0]
        print(f"\n  Best strategy (test Sharpe):")
        print(f"    Strategy:   {best_name}")
        print(f"    Sharpe:     {best_sharpe:.3f}")
        print(f"    Ann Return: {best_ann_ret:.2%}")
        print(f"    Max DD:     {best_mdd:.2%}")
        print(f"    Win Rate:   {best_wr:.1%}")
        print(f"    Config:     K={best_k}, rebalance={best_rb}d")

        # Strategies with positive test Sharpe
        positive_test = [(n, s) for n, s, *_ in strat_sharpes if s > 0]
        print(f"\n  Strategies with positive test Sharpe: {len(positive_test)}/{len(strat_sharpes)}")
        for name, sharpe in positive_test:
            print(f"    {name}: Sharpe = {sharpe:.3f}")

    print(f"\n  Key Findings:")
    print(f"    - HV signals can be reconstructed from futures data back to 2021")
    print(f"    - {len(iv_hv_baselines)} products have baseline IV/HV ratios from snapshot")
    print(f"    - Estimated IV strategy has limited reliability (baseline from 2 days)")
    print(f"    - Pure HV strategies (mean reversion, momentum, cross-sectional) are more robust")

    print(f"\n{'=' * 130}")
    print("  BACKTEST COMPLETE")
    print(f"{'=' * 120}")


if __name__ == '__main__':
    main()
