#!/usr/bin/env python3
"""
Seasonality-Based Strategy for Chinese Commodity Futures v101
=============================================================
Strategies tested:
  1. Monthly seasonality: rank commodities by historical avg monthly returns
  2. Day-of-week seasonality: certain commodities move on certain weekdays
  3. Pre-holiday effect: returns before major Chinese holidays
  4. Month-start / month-end effect
  5. Combination: seasonality filter + momentum confirmation

Walk-forward: train 2021-2023, validate 2024, test 2025-2026.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.expanduser('~/home/futures_platform/data/futures_weighted')

# ──────────────────────────── Data Loading ────────────────────────────

def load_all_data(data_dir=DATA_DIR):
    """Load all commodity futures data, standardize date format."""
    all_data = {}
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith('.csv'):
            continue
        symbol = f.replace('.csv', '')
        df = pd.read_csv(os.path.join(data_dir, f))
        # Handle mixed date formats
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        df = df.sort_values('trade_date').reset_index(drop=True)
        # Filter to 2021+
        df = df[df['trade_date'] >= '2021-01-01'].copy()
        if len(df) < 200:
            continue
        df['return'] = df['close'].pct_change()
        df['month'] = df['trade_date'].dt.month
        df['year'] = df['trade_date'].dt.year
        df['dayofweek'] = df['trade_date'].dt.dayofweek  # 0=Mon..4=Fri
        df['day'] = df['trade_date'].dt.day
        df['week_of_year'] = df['trade_date'].dt.isocalendar().week.astype(int)
        all_data[symbol] = df
    return all_data


# ──────────────────────────── Chinese Holidays ────────────────────────────

def build_chinese_holidays():
    """
    Build a set of Chinese holiday dates (major ones) for 2021-2026.
    Returns dict mapping date -> holiday_name.
    """
    holidays = {}
    # New Year (元旦)
    for y in range(2021, 2027):
        holidays[pd.Timestamp(f'{y}-01-01')] = 'new_year'
    # Spring Festival (Chinese New Year) -- approximate start of week-long holiday
    spring_dates = {
        2021: '2021-02-11',  # CNY eve
        2022: '2022-01-31',
        2023: '2023-01-21',
        2024: '2024-02-09',
        2025: '2025-01-28',
        2026: '2026-02-16',
    }
    for y, d in spring_dates.items():
        holidays[pd.Timestamp(d)] = 'spring_festival'
    # Qingming (Tomb Sweeping) -- ~April 4-5
    for y in range(2021, 2027):
        holidays[pd.Timestamp(f'{y}-04-04')] = 'qingming'
    # Labor Day -- May 1
    for y in range(2021, 2027):
        holidays[pd.Timestamp(f'{y}-05-01')] = 'labor_day'
    # Dragon Boat -- ~June (varies)
    dragon_dates = {
        2021: '2021-06-14',
        2022: '2022-06-03',
        2023: '2023-06-22',
        2024: '2024-06-10',
        2025: '2025-05-31',
        2026: '2026-06-19',
    }
    for y, d in dragon_dates.items():
        holidays[pd.Timestamp(d)] = 'dragon_boat'
    # Mid-Autumn
    mid_autumn = {
        2021: '2021-09-21',
        2022: '2022-09-10',
        2023: '2023-09-29',
        2024: '2024-09-17',
        2025: '2025-10-06',
        2026: '2026-09-25',
    }
    for y, d in mid_autumn.items():
        holidays[pd.Timestamp(d)] = 'mid_autumn'
    # National Day -- Oct 1
    for y in range(2021, 2027):
        holidays[pd.Timestamp(f'{y}-10-01')] = 'national_day'

    return holidays


def get_pre_holiday_dates(holidays, trading_dates, days_before=3):
    """For each holiday, find the N trading days immediately before it."""
    trading_set = sorted(trading_dates)
    pre_hol_dates = set()
    for hol_date in holidays.keys():
        prior = [d for d in trading_set if d < hol_date]
        for d in prior[-days_before:]:
            pre_hol_dates.add(d)
    return pre_hol_dates


# ──────────────────────────── Seasonality Calculations ────────────────────────────

def calc_monthly_seasonality(all_data, train_end='2023-12-31'):
    """
    For each commodity, calculate average monthly return during training period.
    Returns: dict[symbol] -> DataFrame with month, avg_return, std, sharpe, win_rate, count
    """
    seasonality = {}
    for sym, df in all_data.items():
        train = df[df['trade_date'] <= train_end].copy()
        if len(train) < 100:
            continue
        # Calculate monthly returns: first-close-of-month to last-close-of-month
        train['ym'] = train['trade_date'].dt.to_period('M')
        monthly = train.groupby('ym').agg(
            first_close=('close', 'first'),
            last_close=('close', 'last'),
            first_date=('trade_date', 'first'),
            last_date=('trade_date', 'last'),
        )
        monthly['monthly_return'] = (monthly['last_close'] / monthly['first_close'] - 1)
        monthly['month'] = monthly.index.month

        # Average by calendar month
        month_stats = monthly.groupby('month').agg(
            avg_return=('monthly_return', 'mean'),
            std_return=('monthly_return', 'std'),
            count=('monthly_return', 'count'),
            win_rate=('monthly_return', lambda x: (x > 0).mean()),
        ).reset_index()
        month_stats['sharpe'] = month_stats['avg_return'] / month_stats['std_return'].replace(0, np.nan)

        seasonality[sym] = month_stats
    return seasonality


def calc_dow_seasonality(all_data, train_end='2023-12-31'):
    """
    Calculate average daily return by day-of-week for each commodity.
    Returns: dict[symbol] -> DataFrame with dow, avg_return, t_stat
    """
    dow_seasonality = {}
    for sym, df in all_data.items():
        train = df[(df['trade_date'] <= train_end) & (df['return'].notna())].copy()
        if len(train) < 100:
            continue
        dow_stats = train.groupby('dayofweek').agg(
            avg_return=('return', 'mean'),
            std_return=('return', 'std'),
            count=('return', 'count'),
        ).reset_index()
        dow_stats['t_stat'] = dow_stats['avg_return'] / (dow_stats['std_return'] / np.sqrt(dow_stats['count']))
        dow_seasonality[sym] = dow_stats
    return dow_seasonality


def calc_preholiday_effect(all_data, holidays, train_end='2023-12-31', days_before=3):
    """
    Calculate average return on days before holidays vs normal days.
    Returns: dict[symbol] -> {pre_holiday_avg, normal_avg, diff, t_stat}
    """
    results = {}
    for sym, df in all_data.items():
        train = df[(df['trade_date'] <= train_end) & (df['return'].notna())].copy()
        if len(train) < 100:
            continue
        pre_hol = get_pre_holiday_dates(holidays, train['trade_date'].tolist(), days_before)
        train['is_pre_holiday'] = train['trade_date'].isin(pre_hol)

        pre_ret = train[train['is_pre_holiday']]['return']
        norm_ret = train[~train['is_pre_holiday']]['return']

        if len(pre_ret) < 10:
            continue

        results[sym] = {
            'pre_holiday_avg': pre_ret.mean(),
            'pre_holiday_std': pre_ret.std(),
            'pre_holiday_count': len(pre_ret),
            'normal_avg': norm_ret.mean(),
            'normal_std': norm_ret.std(),
            'normal_count': len(norm_ret),
            'diff': pre_ret.mean() - norm_ret.mean(),
            'pre_holiday_wr': (pre_ret > 0).mean(),
            'normal_wr': (norm_ret > 0).mean(),
        }
    return results


def calc_month_start_end_effect(all_data, train_end='2023-12-31', start_days=3, end_days=3):
    """
    Calculate returns at month-start (first N trading days) and month-end (last N trading days).
    """
    results = {}
    for sym, df in all_data.items():
        train = df[(df['trade_date'] <= train_end) & (df['return'].notna())].copy()
        if len(train) < 100:
            continue

        train['ym'] = train['trade_date'].dt.to_period('M')
        # Month start: first N trading days of each month
        train['day_in_month'] = train.groupby('ym').cumcount()
        month_size = train.groupby('ym')['trade_date'].transform('count')
        train['days_from_end'] = month_size - 1 - train['day_in_month']

        start_ret = train[train['day_in_month'] < start_days]['return']
        end_ret = train[train['days_from_end'] < end_days]['return']
        mid_ret = train[(train['day_in_month'] >= start_days) & (train['days_from_end'] >= end_days)]['return']

        if len(start_ret) < 10 or len(end_ret) < 10:
            continue

        results[sym] = {
            'month_start_avg': start_ret.mean(),
            'month_start_wr': (start_ret > 0).mean(),
            'month_start_count': len(start_ret),
            'month_end_avg': end_ret.mean(),
            'month_end_wr': (end_ret > 0).mean(),
            'month_end_count': len(end_ret),
            'mid_avg': mid_ret.mean(),
            'mid_wr': (mid_ret > 0).mean(),
            'start_minus_mid': start_ret.mean() - mid_ret.mean(),
            'end_minus_mid': end_ret.mean() - mid_ret.mean(),
        }
    return results


# ──────────────────────────── Strategy Backtesting Engine ────────────────────────────

def _get_trading_dates(all_data, start=None, end=None):
    """Build a union of all trading dates across all commodities."""
    all_dates = set()
    for sym, df in all_data.items():
        dates = df['trade_date']
        if start:
            dates = dates[dates >= start]
        if end:
            dates = dates[dates <= end]
        all_dates.update(dates.tolist())
    return sorted(all_dates)


def backtest_monthly_seasonality(all_data, seasonality_data, K=10,
                                  test_start='2024-01-01', test_end='2026-12-31',
                                  long_only=False):
    """
    Monthly seasonality strategy.
    At start of each month, rank commodities by historical avg return for that month.
    Go long top K, short bottom K.
    Equal-weight portfolio.
    """
    # Build look-up: sym -> {month: avg_return}
    lookup = {}
    for sym, stats in seasonality_data.items():
        lookup[sym] = dict(zip(stats['month'], stats['avg_return']))

    # Get test period months using union of all trading dates
    trading_dates = pd.Series(_get_trading_dates(all_data, test_start, test_end))
    months = sorted(trading_dates.dt.to_period('M').unique())

    monthly_returns_long = []
    monthly_returns_short = []
    monthly_returns_combined = []
    holdings_records = []

    for m in months:
        month_num = m.month
        # Score each commodity
        scores = {}
        for sym, month_avgs in lookup.items():
            if month_num in month_avgs:
                scores[sym] = month_avgs[month_num]

        if len(scores) < 2 * K:
            continue

        # Rank
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        long_syms = [r[0] for r in ranked[:K]]
        short_syms = [r[0] for r in ranked[-K:]] if not long_only else []

        # Calculate actual returns for this month
        long_rets = []
        short_rets = []
        for sym in long_syms:
            df = all_data.get(sym)
            if df is None:
                continue
            month_data = df[df['trade_date'].dt.to_period('M') == m]
            if len(month_data) < 5:
                continue
            ret = month_data['close'].iloc[-1] / month_data['close'].iloc[0] - 1
            long_rets.append(ret)

        for sym in short_syms:
            df = all_data.get(sym)
            if df is None:
                continue
            month_data = df[df['trade_date'].dt.to_period('M') == m]
            if len(month_data) < 5:
                continue
            ret = month_data['close'].iloc[-1] / month_data['close'].iloc[0] - 1
            short_rets.append(ret)

        if not long_rets:
            continue

        avg_long = np.mean(long_rets) if long_rets else 0
        avg_short = np.mean(short_rets) if short_rets else 0
        combined = avg_long + (-avg_short if short_rets else 0)

        monthly_returns_long.append(avg_long)
        monthly_returns_short.append(-avg_short if short_rets else 0)
        monthly_returns_combined.append(combined)

        holdings_records.append({
            'month': str(m),
            'long': long_syms,
            'short': short_syms,
            'long_ret': avg_long,
            'short_ret': -avg_short if short_rets else 0,
            'combined_ret': combined,
        })

    return {
        'long_returns': monthly_returns_long,
        'short_returns': monthly_returns_short,
        'combined_returns': monthly_returns_combined,
        'holdings': holdings_records,
    }


def backtest_dow_seasonality(all_data, dow_data, test_start='2024-01-01',
                              test_end='2026-12-31'):
    """
    Day-of-week strategy. Each day, go long commodities with strongest
    positive DoW pattern for today's weekday, short those with strongest negative.
    """
    # Build lookup
    lookup = {}
    for sym, stats in dow_data.items():
        lookup[sym] = dict(zip(stats['dayofweek'], stats['avg_return']))

    daily_rets = []
    test_dates = _get_trading_dates(all_data, test_start, test_end)

    K = 10

    for date in test_dates:
        date_ts = pd.Timestamp(date)
        dow = date_ts.dayofweek

        # Score commodities
        scores = {}
        for sym, dow_avgs in lookup.items():
            if dow in dow_avgs:
                scores[sym] = dow_avgs[dow]

        if len(scores) < 2 * K:
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        long_syms = [r[0] for r in ranked[:K]]
        short_syms = [r[0] for r in ranked[-K:]]

        # Get today's returns
        long_rets = []
        short_rets = []
        for sym in long_syms:
            df = all_data.get(sym)
            if df is None:
                continue
            row = df[df['trade_date'] == date_ts]
            if len(row) == 0 or pd.isna(row['return'].values[0]):
                continue
            long_rets.append(row['return'].values[0])

        for sym in short_syms:
            df = all_data.get(sym)
            if df is None:
                continue
            row = df[df['trade_date'] == date_ts]
            if len(row) == 0 or pd.isna(row['return'].values[0]):
                continue
            short_rets.append(-row['return'].values[0])

        if long_rets or short_rets:
            combined = np.mean(long_rets) if long_rets else 0
            combined += np.mean(short_rets) if short_rets else 0
            daily_rets.append(combined)

    return daily_rets


def backtest_preholiday(all_data, holidays, preholiday_data,
                         test_start='2024-01-01', test_end='2026-12-31',
                         top_n=15, days_before=3):
    """
    Pre-holiday effect strategy.
    On pre-holiday days, go long commodities with strongest historical pre-holiday premium.
    """
    # Rank by pre-holiday premium
    ranked = sorted(preholiday_data.items(), key=lambda x: x[1]['diff'], reverse=True)
    long_syms = [r[0] for r in ranked[:top_n] if r[1]['diff'] > 0]
    short_syms = [r[0] for r in ranked[-top_n:] if r[1]['diff'] < 0]

    # Get test trading dates
    all_trading_dates = _get_trading_dates(all_data, test_start, test_end)
    pre_hol_dates = get_pre_holiday_dates(holidays, all_trading_dates, days_before)

    pre_hol_rets = []
    normal_rets = []

    for date_ts in pre_hol_dates:
        long_r = []
        short_r = []
        for sym in long_syms:
            df = all_data.get(sym)
            if df is None:
                continue
            row = df[df['trade_date'] == date_ts]
            if len(row) == 0 or pd.isna(row['return'].values[0]):
                continue
            long_r.append(row['return'].values[0])
        for sym in short_syms:
            df = all_data.get(sym)
            if df is None:
                continue
            row = df[df['trade_date'] == date_ts]
            if len(row) == 0 or pd.isna(row['return'].values[0]):
                continue
            short_r.append(-row['return'].values[0])

        if long_r or short_r:
            combined = np.mean(long_r) if long_r else 0
            combined += np.mean(short_r) if short_r else 0
            pre_hol_rets.append(combined)

    # Normal days for comparison
    normal_dates = set(all_trading_dates) - pre_hol_dates
    for date_ts in list(normal_dates)[:len(pre_hol_dates) * 3]:
        long_r = []
        short_r = []
        for sym in long_syms:
            df = all_data.get(sym)
            if df is None:
                continue
            row = df[df['trade_date'] == date_ts]
            if len(row) == 0 or pd.isna(row['return'].values[0]):
                continue
            long_r.append(row['return'].values[0])
        for sym in short_syms:
            df = all_data.get(sym)
            if df is None:
                continue
            row = df[df['trade_date'] == date_ts]
            if len(row) == 0 or pd.isna(row['return'].values[0]):
                continue
            short_r.append(-row['return'].values[0])
        if long_r or short_r:
            combined = np.mean(long_r) if long_r else 0
            combined += np.mean(short_r) if short_r else 0
            normal_rets.append(combined)

    return {
        'pre_holiday_returns': pre_hol_rets,
        'normal_returns': normal_rets,
        'long_syms': long_syms,
        'short_syms': short_syms,
    }


def backtest_month_start_end(all_data, mse_data, test_start='2024-01-01',
                              test_end='2026-12-31', top_n=15,
                              start_days=3, end_days=3):
    """
    Month-start/end effect strategy.
    Go long at month-start for commodities with strong start-of-month pattern,
    go long at month-end for commodities with strong end-of-month pattern.
    """
    # Rank
    start_ranked = sorted(mse_data.items(), key=lambda x: x[1]['start_minus_mid'], reverse=True)
    end_ranked = sorted(mse_data.items(), key=lambda x: x[1]['end_minus_mid'], reverse=True)

    start_long = [r[0] for r in start_ranked[:top_n] if r[1]['start_minus_mid'] > 0]
    end_long = [r[0] for r in end_ranked[:top_n] if r[1]['end_minus_mid'] > 0]

    all_dates = _get_trading_dates(all_data, test_start, test_end)

    start_rets = []
    end_rets = []
    mid_rets = []

    for date_ts in all_dates:
        date_ts = pd.Timestamp(date_ts)
        # Find position in month
        month_start = date_ts.replace(day=1)
        # Get trading day index in month
        month_dates = [d for d in all_dates if pd.Timestamp(d).to_period('M') == date_ts.to_period('M')]
        day_idx = month_dates.index(date_ts) if date_ts in month_dates else -1
        days_from_end = len(month_dates) - 1 - day_idx

        is_start = day_idx < start_days
        is_end = days_from_end < end_days

        if is_start:
            syms = start_long
        elif is_end:
            syms = end_long
        else:
            # mid-month baseline -- use both sets
            syms = list(set(start_long + end_long))
            target = mid_rets
            rlist = []
            for sym in syms:
                df = all_data.get(sym)
                if df is None:
                    continue
                row = df[df['trade_date'] == date_ts]
                if len(row) == 0 or pd.isna(row['return'].values[0]):
                    continue
                rlist.append(row['return'].values[0])
            if rlist:
                mid_rets.append(np.mean(rlist))
            continue

        rlist = []
        for sym in syms:
            df = all_data.get(sym)
            if df is None:
                continue
            row = df[df['trade_date'] == date_ts]
            if len(row) == 0 or pd.isna(row['return'].values[0]):
                continue
            rlist.append(row['return'].values[0])
        if rlist:
            if is_start:
                start_rets.append(np.mean(rlist))
            else:
                end_rets.append(np.mean(rlist))

    return {
        'month_start_returns': start_rets,
        'month_end_returns': end_rets,
        'mid_returns': mid_rets,
        'start_syms': start_long,
        'end_syms': end_long,
    }


def backtest_seasonality_plus_momentum(all_data, seasonality_data, K=10,
                                        test_start='2024-01-01',
                                        test_end='2026-12-31'):
    """
    Combination: Seasonality filter + momentum confirmation.
    Only take seasonal long trades when 20-day momentum > 0,
    only take seasonal short trades when 20-day momentum < 0.
    """
    # Precompute momentum for each commodity
    momentum = {}
    for sym, df in all_data.items():
        df2 = df.copy()
        df2['mom_20'] = df2['close'] / df2['close'].shift(20) - 1
        momentum[sym] = df2[['trade_date', 'mom_20']].set_index('trade_date')['mom_20']

    # Seasonality lookup
    lookup = {}
    for sym, stats in seasonality_data.items():
        lookup[sym] = dict(zip(stats['month'], stats['avg_return']))

    trading_dates = pd.Series(_get_trading_dates(all_data, test_start, test_end))
    months = sorted(trading_dates.dt.to_period('M').unique())

    monthly_returns = []

    for m in months:
        month_num = m.month
        scores = {}
        for sym, month_avgs in lookup.items():
            if month_num in month_avgs:
                scores[sym] = month_avgs[month_num]

        if len(scores) < 2 * K:
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        long_candidates = [r[0] for r in ranked[:K]]
        short_candidates = [r[0] for r in ranked[-K:]]

        # Get first trading date of month for momentum signal
        month_dates = trading_dates[trading_dates.dt.to_period('M') == m]
        if len(month_dates) == 0:
            continue
        signal_date = month_dates.iloc[0]

        # Filter with momentum
        filtered_long = []
        for sym in long_candidates:
            mom_series = momentum.get(sym)
            if mom_series is None:
                continue
            try:
                mom_val = mom_series.loc[mom_series.index <= signal_date].iloc[-1]
                if not pd.isna(mom_val) and mom_val > 0:
                    filtered_long.append(sym)
            except (IndexError, KeyError):
                continue

        filtered_short = []
        for sym in short_candidates:
            mom_series = momentum.get(sym)
            if mom_series is None:
                continue
            try:
                mom_val = mom_series.loc[mom_series.index <= signal_date].iloc[-1]
                if not pd.isna(mom_val) and mom_val < 0:
                    filtered_short.append(sym)
            except (IndexError, KeyError):
                continue

        # Calculate returns
        long_rets = []
        for sym in filtered_long:
            df = all_data.get(sym)
            if df is None:
                continue
            month_data = df[df['trade_date'].dt.to_period('M') == m]
            if len(month_data) < 5:
                continue
            ret = month_data['close'].iloc[-1] / month_data['close'].iloc[0] - 1
            long_rets.append(ret)

        short_rets = []
        for sym in filtered_short:
            df = all_data.get(sym)
            if df is None:
                continue
            month_data = df[df['trade_date'].dt.to_period('M') == m]
            if len(month_data) < 5:
                continue
            ret = -(month_data['close'].iloc[-1] / month_data['close'].iloc[0] - 1)
            short_rets.append(ret)

        all_rets = long_rets + short_rets
        if all_rets:
            monthly_returns.append(np.mean(all_rets))
        else:
            monthly_returns.append(0)

    return monthly_returns


# ──────────────────────────── Performance Metrics ────────────────────────────

def calc_performance(returns, label="Strategy", is_daily=False):
    """Calculate key performance metrics from a return series."""
    if not returns or len(returns) == 0:
        return {}

    rets = np.array(returns)
    ann_factor = 252 if is_daily else 12

    total_return = np.prod(1 + rets) - 1
    ann_return = (1 + total_return) ** (ann_factor / len(rets)) - 1 if len(rets) > 0 else 0
    ann_vol = np.std(rets) * np.sqrt(ann_factor)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Win rate
    wr = np.mean(rets > 0)

    # Max drawdown
    cum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    mdd = np.min(dd)

    # Calmar
    calmar = ann_return / abs(mdd) if mdd != 0 else 0

    # Sortino (downside deviation)
    neg_rets = rets[rets < 0]
    downside_vol = np.std(neg_rets) * np.sqrt(ann_factor) if len(neg_rets) > 0 else ann_vol
    sortino = ann_return / downside_vol if downside_vol > 0 else 0

    # Avg win / avg loss
    avg_win = np.mean(rets[rets > 0]) if np.any(rets > 0) else 0
    avg_loss = abs(np.mean(rets[rets < 0])) if np.any(rets < 0) else 0
    profit_factor = (avg_win * np.sum(rets > 0)) / (avg_loss * np.sum(rets < 0)) if avg_loss > 0 and np.sum(rets < 0) > 0 else float('inf')

    return {
        'label': label,
        'total_return': total_return,
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'win_rate': wr,
        'max_drawdown': mdd,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'n_periods': len(rets),
    }


def print_performance(metrics):
    """Pretty print performance metrics."""
    if not metrics:
        print("  No data available.")
        return
    print(f"  {metrics['label']}")
    print(f"    Total Return:     {metrics['total_return']:.2%}")
    print(f"    Annual Return:    {metrics['ann_return']:.2%}")
    print(f"    Annual Vol:       {metrics['ann_vol']:.2%}")
    print(f"    Sharpe:           {metrics['sharpe']:.3f}")
    print(f"    Sortino:          {metrics['sortino']:.3f}")
    print(f"    Calmar:           {metrics['calmar']:.3f}")
    print(f"    Win Rate:         {metrics['win_rate']:.2%}")
    print(f"    Max Drawdown:     {metrics['max_drawdown']:.2%}")
    print(f"    Avg Win:          {metrics['avg_win']:.4f}")
    print(f"    Avg Loss:         {metrics['avg_loss']:.4f}")
    print(f"    Profit Factor:    {metrics['profit_factor']:.3f}")
    print(f"    Periods:          {metrics['n_periods']}")


# ──────────────────────────── Walk-Forward Engine ────────────────────────────

def walk_forward_monthly_seasonality(all_data, K=10):
    """
    Walk-forward test for monthly seasonality.
    Fold 1: train 2021-2023, test 2024
    Fold 2: train 2021-2024, test 2025-2026
    """
    folds = [
        ('2021-2023 -> 2024', '2023-12-31', '2024-01-01', '2024-12-31'),
        ('2021-2024 -> 2025-2026', '2024-12-31', '2025-01-01', '2026-12-31'),
    ]

    all_oos_returns = []

    for fold_name, train_end, test_start, test_end in folds:
        print(f"\n  Walk-Forward Fold: {fold_name}")
        seasonality = calc_monthly_seasonality(all_data, train_end)
        result = backtest_monthly_seasonality(all_data, seasonality, K=K,
                                               test_start=test_start, test_end=test_end)
        metrics = calc_performance(result['combined_returns'],
                                    label=f"  {fold_name} (K={K})", is_daily=False)
        print_performance(metrics)
        all_oos_returns.extend(result['combined_returns'])

    # Combined OOS
    combined_metrics = calc_performance(all_oos_returns,
                                         label=f"  Combined OOS (K={K})", is_daily=False)
    print(f"\n  {'='*50}")
    print_performance(combined_metrics)
    return combined_metrics


# ──────────────────────────── Main ────────────────────────────

def main():
    print("=" * 80)
    print("  SEASONALITY-BASED STRATEGY FOR CHINESE COMMODITY FUTURES v101")
    print("=" * 80)

    # Load data
    print("\n[1] Loading data...")
    all_data = load_all_data()
    print(f"    Loaded {len(all_data)} commodities")

    holidays = build_chinese_holidays()
    print(f"    Chinese holidays defined: {len(holidays)}")

    # ─── Monthly Seasonality Table ───
    print("\n" + "=" * 80)
    print("  [2] MONTHLY SEASONALITY TABLE (Training: 2021-2023)")
    print("=" * 80)

    seasonality = calc_monthly_seasonality(all_data, train_end='2023-12-31')

    # Print top seasonal commodities for each month
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    print(f"\n  {'Month':<6} {'Top 5 Bullish':<55} {'Top 5 Bearish':<55}")
    print("  " + "-" * 116)

    for m in range(1, 13):
        month_scores = {}
        for sym, stats in seasonality.items():
            row = stats[stats['month'] == m]
            if len(row) > 0:
                month_scores[sym] = row['avg_return'].values[0]

        ranked = sorted(month_scores.items(), key=lambda x: x[1], reverse=True)
        top5 = ranked[:5]
        bot5 = ranked[-5:]

        top_str = ", ".join([f"{s}({r:.1%})" for s, r in top5])
        bot_str = ", ".join([f"{s}({r:.1%})" for s, r in bot5])
        print(f"  {month_names[m-1]:<6} {top_str:<55} {bot_str:<55}")

    # ─── Strategy 1: Monthly Seasonality ───
    print("\n" + "=" * 80)
    print("  [3] STRATEGY 1: MONTHLY SEASONALITY (Long+Short)")
    print("=" * 80)

    for K in [5, 10, 15]:
        print(f"\n  --- K = {K} (top/bottom {K} commodities) ---")

        # Train 2021-2023, Test 2024-2026
        result = backtest_monthly_seasonality(all_data, seasonality, K=K,
                                               test_start='2024-01-01',
                                               test_end='2026-12-31')

        # Long leg
        m_long = calc_performance(result['long_returns'],
                                   label=f"  Long Leg (K={K})", is_daily=False)
        print_performance(m_long)

        # Short leg
        m_short = calc_performance(result['short_returns'],
                                    label=f"  Short Leg (K={K})", is_daily=False)
        print_performance(m_short)

        # Combined
        m_comb = calc_performance(result['combined_returns'],
                                   label=f"  Combined L+S (K={K})", is_daily=False)
        print_performance(m_comb)

    # ─── Strategy 2: Day-of-Week Seasonality ───
    print("\n" + "=" * 80)
    print("  [4] STRATEGY 2: DAY-OF-WEEK SEASONALITY")
    print("=" * 80)

    # Print DoW table for top commodities
    dow_data = calc_dow_seasonality(all_data, train_end='2023-12-31')

    print("\n  Average daily return by weekday (selected commodities):")
    print(f"  {'Symbol':<10} {'Mon':>8} {'Tue':>8} {'Wed':>8} {'Thu':>8} {'Fri':>8} {'Best':>6} {'Worst':>6}")
    print("  " + "-" * 74)

    # Show commodities with strongest DoW pattern
    dow_strength = {}
    for sym, stats in dow_data.items():
        rets = stats.set_index('dayofweek')['avg_return']
        if len(rets) >= 5:
            dow_strength[sym] = rets.max() - rets.min()

    top_dow = sorted(dow_strength.items(), key=lambda x: x[1], reverse=True)[:15]

    for sym, _ in top_dow:
        stats = dow_data[sym].set_index('dayofweek')
        vals = [stats.loc[d, 'avg_return'] * 100 if d in stats.index else 0 for d in range(5)]
        best = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'][np.argmax(vals)]
        worst = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'][np.argmin(vals)]
        print(f"  {sym:<10} {vals[0]:>7.3f}% {vals[1]:>7.3f}% {vals[2]:>7.3f}% {vals[3]:>7.3f}% {vals[4]:>7.3f}% {best:>6} {worst:>6}")

    # Backtest DoW
    print("\n  Day-of-Week Strategy Backtest (2024-2026):")
    dow_rets = backtest_dow_seasonality(all_data, dow_data,
                                         test_start='2024-01-01',
                                         test_end='2026-12-31')
    dow_metrics = calc_performance(dow_rets, label="  DoW Strategy", is_daily=True)
    print_performance(dow_metrics)

    # ─── Strategy 3: Pre-Holiday Effect ───
    print("\n" + "=" * 80)
    print("  [5] STRATEGY 3: PRE-HOLIDAY EFFECT")
    print("=" * 80)

    prehol = calc_preholiday_effect(all_data, holidays, train_end='2023-12-31', days_before=3)

    # Show top pre-holiday commodities
    print("\n  Top 15 commodities with strongest pre-holiday premium:")
    print(f"  {'Symbol':<10} {'Pre-Hol Avg':>12} {'Normal Avg':>12} {'Diff':>10} {'Pre WR':>8} {'Norm WR':>8}")
    print("  " + "-" * 66)
    ranked_prehol = sorted(prehol.items(), key=lambda x: x[1]['diff'], reverse=True)
    for sym, v in ranked_prehol[:15]:
        print(f"  {sym:<10} {v['pre_holiday_avg']*100:>11.3f}% {v['normal_avg']*100:>11.3f}% "
              f"{v['diff']*100:>9.3f}% {v['pre_holiday_wr']:>7.1%} {v['normal_wr']:>7.1%}")

    print("\n  Bottom 5 (negative pre-holiday effect):")
    for sym, v in ranked_prehol[-5:]:
        print(f"  {sym:<10} {v['pre_holiday_avg']*100:>11.3f}% {v['normal_avg']*100:>11.3f}% "
              f"{v['diff']*100:>9.3f}% {v['pre_holiday_wr']:>7.1%} {v['normal_wr']:>7.1%}")

    # Backtest
    ph_result = backtest_preholiday(all_data, holidays, prehol,
                                     test_start='2024-01-01',
                                     test_end='2026-12-31',
                                     top_n=15, days_before=3)
    ph_metrics = calc_performance(ph_result['pre_holiday_returns'],
                                   label="  Pre-Holiday Strategy", is_daily=True)
    print_performance(ph_metrics)

    norm_metrics = calc_performance(ph_result['normal_returns'],
                                     label="  Normal Days (comparison)", is_daily=True)
    print_performance(norm_metrics)

    # ─── Strategy 4: Month-Start / Month-End Effect ───
    print("\n" + "=" * 80)
    print("  [6] STRATEGY 4: MONTH-START / MONTH-END EFFECT")
    print("=" * 80)

    mse_data = calc_month_start_end_effect(all_data, train_end='2023-12-31',
                                            start_days=3, end_days=3)

    print("\n  Top 15 commodities by month-start premium:")
    print(f"  {'Symbol':<10} {'Start Avg':>10} {'End Avg':>10} {'Mid Avg':>10} {'Start-Mid':>10} {'End-Mid':>10}")
    print("  " + "-" * 64)
    ranked_start = sorted(mse_data.items(), key=lambda x: x[1]['start_minus_mid'], reverse=True)
    for sym, v in ranked_start[:15]:
        print(f"  {sym:<10} {v['month_start_avg']*100:>9.3f}% {v['month_end_avg']*100:>9.3f}% "
              f"{v['mid_avg']*100:>9.3f}% {v['start_minus_mid']*100:>9.3f}% {v['end_minus_mid']*100:>9.3f}%")

    # Backtest
    mse_result = backtest_month_start_end(all_data, mse_data,
                                           test_start='2024-01-01',
                                           test_end='2026-12-31',
                                           top_n=15, start_days=3, end_days=3)

    mse_start = calc_performance(mse_result['month_start_returns'],
                                  label="  Month-Start Days", is_daily=True)
    mse_end = calc_performance(mse_result['month_end_returns'],
                                label="  Month-End Days", is_daily=True)
    mse_mid = calc_performance(mse_result['mid_returns'],
                                label="  Mid-Month (comparison)", is_daily=True)
    print_performance(mse_start)
    print_performance(mse_end)
    print_performance(mse_mid)

    # ─── Strategy 5: Seasonality + Momentum Combination ───
    print("\n" + "=" * 80)
    print("  [7] STRATEGY 5: SEASONALITY + MOMENTUM COMBINATION")
    print("=" * 80)

    for K in [5, 10, 15]:
        print(f"\n  --- K = {K} ---")

        # Pure seasonality for comparison
        pure = backtest_monthly_seasonality(all_data, seasonality, K=K,
                                            test_start='2024-01-01',
                                            test_end='2026-12-31')
        pure_m = calc_performance(pure['combined_returns'],
                                   label=f"  Pure Seasonality (K={K})", is_daily=False)
        print_performance(pure_m)

        # Seasonality + Momentum
        combo = backtest_seasonality_plus_momentum(all_data, seasonality, K=K,
                                                    test_start='2024-01-01',
                                                    test_end='2026-12-31')
        combo_m = calc_performance(combo,
                                    label=f"  Seasonality+Momentum (K={K})", is_daily=False)
        print_performance(combo_m)

        # Improvement
        if pure_m['sharpe'] != 0:
            improvement = (combo_m['sharpe'] - pure_m['sharpe']) / abs(pure_m['sharpe']) * 100
            print(f"    Sharpe improvement: {improvement:+.1f}%")

    # ─── Walk-Forward Analysis ───
    print("\n" + "=" * 80)
    print("  [8] WALK-FORWARD ANALYSIS (Monthly Seasonality)")
    print("=" * 80)

    for K in [5, 10, 15]:
        print(f"\n  ===== K = {K} =====")
        walk_forward_monthly_seasonality(all_data, K=K)

    # ─── Walk-Forward for Combo ───
    print("\n" + "=" * 80)
    print("  [9] WALK-FORWARD: SEASONALITY + MOMENTUM")
    print("=" * 80)

    for K in [5, 10, 15]:
        print(f"\n  ===== K = {K} =====")
        folds = [
            ('2021-2023 -> 2024', '2023-12-31', '2024-01-01', '2024-12-31'),
            ('2021-2024 -> 2025-2026', '2024-12-31', '2025-01-01', '2026-12-31'),
        ]
        all_oos = []
        for fold_name, train_end, test_start, test_end in folds:
            print(f"\n  Fold: {fold_name}")
            seas = calc_monthly_seasonality(all_data, train_end)
            combo = backtest_seasonality_plus_momentum(all_data, seas, K=K,
                                                        test_start=test_start,
                                                        test_end=test_end)
            m = calc_performance(combo, label=f"  {fold_name}", is_daily=False)
            print_performance(m)
            all_oos.extend(combo)

        cm = calc_performance(all_oos, label=f"  Combined OOS (K={K})", is_daily=False)
        print(f"\n  {'='*50}")
        print_performance(cm)

    # ─── Summary ───
    print("\n" + "=" * 80)
    print("  [10] SUMMARY: BEST CONFIGURATIONS")
    print("=" * 80)

    all_results = []

    # Test all K values for main strategies
    for K in [5, 10, 15]:
        # Monthly seasonality combined
        r = backtest_monthly_seasonality(all_data, seasonality, K=K,
                                          test_start='2024-01-01',
                                          test_end='2026-12-31')
        m = calc_performance(r['combined_returns'],
                              label=f"Monthly Seas. K={K}", is_daily=False)
        all_results.append(m)

        # Combo
        combo = backtest_seasonality_plus_momentum(all_data, seasonality, K=K,
                                                    test_start='2024-01-01',
                                                    test_end='2026-12-31')
        m2 = calc_performance(combo,
                               label=f"Seas.+Mom. K={K}", is_daily=False)
        all_results.append(m2)

    # DoW
    all_results.append(dow_metrics)

    # Pre-holiday
    all_results.append(ph_metrics)

    # Month start/end
    all_results.append(mse_start)
    all_results.append(mse_end)

    # Sort by Sharpe
    all_results.sort(key=lambda x: x.get('sharpe', -999), reverse=True)

    print(f"\n  {'Rank':<5} {'Strategy':<25} {'AnnRet':>8} {'Sharpe':>8} {'WR':>6} {'MDD':>8} {'PF':>6}")
    print("  " + "-" * 70)
    for i, m in enumerate(all_results, 1):
        print(f"  {i:<5} {m['label']:<25} {m['ann_return']:>7.1%} {m['sharpe']:>7.3f} "
              f"{m['win_rate']:>5.1%} {m['max_drawdown']:>7.1%} {m['profit_factor']:>5.2f}")

    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
