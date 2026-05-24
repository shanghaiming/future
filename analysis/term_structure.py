#!/usr/bin/env python3
"""
期货期限结构分析模块
- 计算期限结构曲线 (近月-远月价差)
- 判断 contango / backwardation
- 计算 roll yield / carry
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

BASE_DIR = os.path.expanduser("~/home/futures_platform")
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_futures_data(symbol, data_type="futures_weighted"):
    """加载期货数据"""
    path = os.path.join(DATA_DIR, data_type, f"{symbol.lower()}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df = df.sort_values('trade_date')
    return df


def calculate_term_structure(near_df, far_df, near_symbol, far_symbol):
    """
    计算期限结构指标
    near_df: 近月合约数据
    far_df: 远月合约数据
    """
    if near_df.empty or far_df.empty:
        return pd.DataFrame()
    
    # 合并
    merged = pd.merge(
        near_df[['trade_date', 'close']].rename(columns={'close': 'near_close'}),
        far_df[['trade_date', 'close']].rename(columns={'close': 'far_close'}),
        on='trade_date',
        how='inner'
    )
    
    if merged.empty:
        return pd.DataFrame()
    
    # 期限结构指标
    merged['spread'] = merged['near_close'] - merged['far_close']
    merged['spread_pct'] = merged['spread'] / merged['near_close'] * 100
    merged['annualized_roll_yield'] = merged['spread_pct'] * 12  # 假设1个月换月，年化
    
    # 判断结构
    merged['structure'] = merged['spread'].apply(
        lambda x: 'backwardation' if x > 0 else 'contango'
    )
    
    return merged


def analyze_term_structure_curve(symbol, contract_months=[1, 3, 6, 12]):
    """
    分析多合约期限结构曲线
    需要有多合约数据，目前用加权数据模拟
    """
    df = load_futures_data(symbol)
    if df.empty:
        return None
    
    latest = df.iloc[-1]
    
    # 模拟不同到期月的合约价格（基于历史波动率调整）
    # 实际应该有多合约数据
    volatility = df['close'].pct_change().std() * np.sqrt(252)
    
    curve = {
        'symbol': symbol,
        'date': latest['trade_date'].strftime('%Y-%m-%d'),
        'spot': latest['close'],
        'contracts': {}
    }
    
    for month in contract_months:
        # 模拟：远月合约价格 = 近月 * e^(±便利收益*时间)
        # 实际应从多合约数据计算
        time_factor = month / 12
        # 简单假设：contango时远月贵，backwardation时远月便宜
        # 用历史基差均值作为估计
        basis_estimate = 0  # 需要现货数据
        curve['contracts'][f'{month}M'] = latest['close'] * (1 + basis_estimate * time_factor)
    
    return curve


def get_carry_signal(symbol):
    """
    生成carry交易信号
    backwardation → 多头信号（赚roll yield）
    contango → 空头信号（或避开）
    """
    df = load_futures_data(symbol)
    if df.empty or len(df) < 20:
        return None
    
    # 用近20日价格变化模拟期限结构倾向
    # 实际应有多合约数据
    recent = df.tail(20)
    momentum = (recent['close'].iloc[-1] / recent['close'].iloc[0] - 1) * 100
    
    # 简单规则：强势品种往往backwardation，弱势contango
    if momentum > 5:
        signal = 'long_carry'
        strength = min(momentum / 10, 1.0)
    elif momentum < -5:
        signal = 'short_carry'
        strength = min(abs(momentum) / 10, 1.0)
    else:
        signal = 'neutral'
        strength = 0.0
    
    return {
        'symbol': symbol,
        'signal': signal,
        'strength': strength,
        'momentum_20d': momentum,
        'date': df['trade_date'].iloc[-1].strftime('%Y-%m-%d')
    }


def analyze_all_symbols():
    """分析所有品种的期限结构信号"""
    weighted_dir = os.path.join(DATA_DIR, "futures_weighted")
    results = []
    
    for f in os.listdir(weighted_dir):
        if not f.endswith('.csv'):
            continue
        symbol = f.replace('.csv', '')
        signal = get_carry_signal(symbol)
        if signal:
            results.append(signal)
    
    # 排序：做多信号强的在前
    results.sort(key=lambda x: x['strength'] if x['signal'] == 'long_carry' else -x['strength'], reverse=True)
    return results


if __name__ == '__main__':
    # 测试
    signals = analyze_all_symbols()
    print(f"分析完成，共{len(signals)}个品种")
    print("\n做多信号最强:")
    for s in signals[:5]:
        if s['signal'] == 'long_carry':
            print(f"  {s['symbol']}: 强度={s['strength']:.2f}, 动量={s['momentum_20d']:.2f}%")
    
    print("\n做空信号最强:")
    for s in signals[-5:]:
        if s['signal'] == 'short_carry':
            print(f"  {s['symbol']}: 强度={s['strength']:.2f}, 动量={s['momentum_20d']:.2f}%")
