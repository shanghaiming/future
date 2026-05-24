#!/usr/bin/env python3
"""
波动率分析模块
- 历史波动率计算
- 波动率锥 (Volatility Cone)
- 波动率偏斜/微笑 (Skew/Smile)
- 已实现波动率 vs 隐含波动率
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.expanduser("~/home/futures_platform")
DATA_DIR = os.path.join(BASE_DIR, "data")


def calculate_historical_volatility(prices, window=20, annualize=True):
    """
    计算历史波动率
    
    Parameters:
    -----------
    prices : pd.Series
        价格序列
    window : int
        计算窗口（交易日）
    annualize : bool
        是否年化（乘以sqrt(252)）
    
    Returns:
    --------
    pd.Series : 波动率序列
    """
    log_returns = np.log(prices / prices.shift(1))
    vol = log_returns.rolling(window=window).std()
    
    if annualize:
        vol = vol * np.sqrt(252)
    
    return vol


def calculate_volatility_cone(prices, windows=[5, 10, 20, 60, 120]):
    """
    计算波动率锥
    显示不同时间窗口的历史波动率分布
    
    Returns:
    --------
    dict : 各窗口的波动率统计
    """
    results = {}
    
    for window in windows:
        vol = calculate_historical_volatility(prices, window=window)
        vol_clean = vol.dropna()
        
        if len(vol_clean) == 0:
            continue
        
        results[f'{window}d'] = {
            'current': vol_clean.iloc[-1],
            'min': vol_clean.min(),
            'max': vol_clean.max(),
            'median': vol_clean.median(),
            'p25': vol_clean.quantile(0.25),
            'p75': vol_clean.quantile(0.75),
            'p10': vol_clean.quantile(0.10),
            'p90': vol_clean.quantile(0.90),
        }
    
    return results


def calculate_realized_volatility(prices, window=20):
    """
    计算已实现波动率（Realized Volatility）
    日内高频数据的平方和开根号
    这里用日数据近似
    """
    log_returns = np.log(prices / prices.shift(1))
    rv = np.sqrt(np.sum(log_returns.tail(window) ** 2))
    return rv * np.sqrt(252 / window)  # 年化


def calculate_parkinson_volatility(high, low, window=20):
    """
    Parkinson波动率（用高低价计算，比收盘价波动率更精确）
    """
    hl_ratio = np.log(high / low)
    parkinson_var = (hl_ratio ** 2) / (4 * np.log(2))
    
    vol = parkinson_var.rolling(window=window).mean()
    return np.sqrt(vol) * np.sqrt(252)


def calculate_garman_klass_volatility(open_p, high, low, close, window=20):
    """
    Garman-Klass波动率（用OHLC，比Parkinson更精确）
    """
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_p) ** 2
    
    gk_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    vol = gk_var.rolling(window=window).mean()
    return np.sqrt(vol) * np.sqrt(252)


def calculate_volatility_regime(volatility_series, lookback=252):
    """
    判断波动率状态（高/中/低）
    """
    if len(volatility_series) < lookback:
        lookback = len(volatility_series)
    
    recent_vol = volatility_series.iloc[-lookback:]
    current = volatility_series.iloc[-1]
    
    p33 = recent_vol.quantile(0.33)
    p67 = recent_vol.quantile(0.67)
    
    if current > p67:
        return 'high'
    elif current < p33:
        return 'low'
    else:
        return 'medium'


def calculate_volatility_persistence(volatility_series):
    """
    计算波动率聚集性（GARCH效应）
    用自相关系数衡量
    """
    from statsmodels.tsa.stattools import acf
    
    vol_clean = volatility_series.dropna()
    if len(vol_clean) < 60:
        return None
    
    autocorr = acf(vol_clean ** 2, nlags=5, fft=True)
    return {
        'lag1': autocorr[1],
        'lag5': autocorr[5] if len(autocorr) > 5 else None,
        'persistence': 'high' if autocorr[1] > 0.1 else 'low'
    }


def analyze_symbol_volatility(symbol, data_type="futures_weighted"):
    """
    综合分析单个品种的波动率特征
    """
    path = os.path.join(DATA_DIR, data_type, f"{symbol.lower()}.csv")
    if not os.path.exists(path):
        return None
    
    df = pd.read_csv(path)
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df = df.sort_values('trade_date')
    
    if len(df) < 60:
        return None
    
    prices = df['close']
    
    # 各种波动率
    hv_20 = calculate_historical_volatility(prices, 20)
    hv_60 = calculate_historical_volatility(prices, 60)
    
    # 波动率锥
    cone = calculate_volatility_cone(prices)
    
    # 已实现波动率
    rv = calculate_realized_volatility(prices)
    
    # 波动率状态
    regime = calculate_volatility_regime(hv_20.dropna())
    
    # Parkinson波动率（如果有高低价数据）
    parkinson = None
    if 'high' in df.columns and 'low' in df.columns:
        parkinson = calculate_parkinson_volatility(df['high'], df['low'], 20).iloc[-1]
    
    # Garman-Klass（如果有OHLC）
    gk = None
    if all(c in df.columns for c in ['open', 'high', 'low', 'close']):
        gk = calculate_garman_klass_volatility(
            df['open'], df['high'], df['low'], df['close'], 20
        ).iloc[-1]
    
    return {
        'symbol': symbol,
        'date': df['trade_date'].iloc[-1].strftime('%Y-%m-%d'),
        'current_price': prices.iloc[-1],
        'hv_20d': hv_20.iloc[-1],
        'hv_60d': hv_60.iloc[-1],
        'realized_vol': rv,
        'parkinson_vol': parkinson,
        'gk_vol': gk,
        'vol_regime': regime,
        'vol_cone': cone,
        'price_change_20d': (prices.iloc[-1] / prices.iloc[-20] - 1) * 100 if len(prices) >= 20 else None,
    }


def get_all_volatility_analysis():
    """分析所有品种的波动率"""
    weighted_dir = os.path.join(DATA_DIR, "futures_weighted")
    results = []
    
    for f in sorted(os.listdir(weighted_dir)):
        if not f.endswith('.csv'):
            continue
        symbol = f.replace('.csv', '')
        analysis = analyze_symbol_volatility(symbol)
        if analysis:
            results.append(analysis)
    
    # 按20日波动率排序
    results.sort(key=lambda x: x['hv_20d'] if x['hv_20d'] is not None else 0, reverse=True)
    return results


if __name__ == '__main__':
    results = get_all_volatility_analysis()
    print(f"波动率分析完成，共{len(results)}个品种\n")
    
    print("波动率最高（短期）:")
    for r in results[:5]:
        print(f"  {r['symbol']}: HV20={r['hv_20d']:.1%}, HV60={r['hv_60d']:.1%}, 状态={r['vol_regime']}")
    
    print("\n波动率最低（短期）:")
    for r in results[-5:]:
        print(f"  {r['symbol']}: HV20={r['hv_20d']:.1%}, HV60={r['hv_60d']:.1%}, 状态={r['vol_regime']}")
