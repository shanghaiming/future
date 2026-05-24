"""
数据加载模块 - 本地CSV数据调用
替代 mindgo_api 的本地实现
"""

import pandas as pd
import numpy as np
import os
from typing import List, Optional, Dict, Union
import glob

# 数据目录路径 — 指向期货数据
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
DAILY_DATA_DIR = os.path.join(DATA_DIR, 'futures_weighted')
WEEKLY_DATA_DIR = os.path.join(DATA_DIR, 'futures_weighted')
MIN5_DATA_DIR = os.path.join(DATA_DIR, 'futures_weighted')
MIN30_DATA_DIR = os.path.join(DATA_DIR, 'futures_weighted')

def list_available_symbols(frequency: str = 'daily') -> List[str]:
    """
    列出指定频率下可用的股票代码
    
    Args:
        frequency: 数据频率，可选 'daily', 'weekly', '5min', '30min'
    
    Returns:
        股票代码列表（不带文件扩展名）
    """
    if frequency == 'daily':
        data_dir = DAILY_DATA_DIR
    elif frequency == 'weekly':
        data_dir = WEEKLY_DATA_DIR
    elif frequency == '5min':
        data_dir = MIN5_DATA_DIR
    elif frequency == '30min':
        data_dir = MIN30_DATA_DIR
    else:
        raise ValueError(f"不支持的频率: {frequency}，支持 'daily', 'weekly', '5min', '30min'")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    symbols = [os.path.basename(f).replace('.csv', '') for f in csv_files]
    return sorted(symbols)

def load_stock_data(symbol: str, 
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   frequency: str = 'daily',
                   fields: Optional[List[str]] = None) -> pd.DataFrame:
    """
    加载单个股票的历史数据
    
    Args:
        symbol: 股票代码，如 '000001.SZ'
        start_date: 开始日期，格式 'YYYYMMDD' 或 'YYYY-MM-DD'，可选
        end_date: 结束日期，格式 'YYYYMMDD' 或 'YYYY-MM-DD'，可选
        frequency: 数据频率，可选 'daily', 'weekly', '5min', '30min'
        fields: 需要加载的字段列表，可选
    
    Returns:
        pandas DataFrame，索引为 datetime，包含OHLC等数据
    """
    if frequency == 'daily':
        data_dir = DAILY_DATA_DIR
    elif frequency == 'weekly':
        data_dir = WEEKLY_DATA_DIR
    elif frequency == '5min':
        data_dir = MIN5_DATA_DIR
    elif frequency == '30min':
        data_dir = MIN30_DATA_DIR
    else:
        raise ValueError(f"不支持的频率: {frequency}")
    
    file_path = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.exists(file_path):
        available = list_available_symbols(frequency)
        raise FileNotFoundError(
            f"数据文件不存在: {file_path}\n"
            f"可用股票代码 ({frequency}): {available[:10]}{'...' if len(available) > 10 else ''}"
        )
    
    # 读取CSV
    df = pd.read_csv(file_path)
    
    # 转换日期列
    if 'trade_date' in df.columns:
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
        df.set_index('trade_date', inplace=True)
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    
    # 重命名列以匹配标准OHLC格式
    column_mapping = {
        'open': 'open',
        'high': 'high', 
        'low': 'low',
        'close': 'close',
        'vol': 'volume',
        'amount': 'amount',
        'pct_chg': 'pct_change'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    # 选择指定字段
    if fields:
        available_fields = [col for col in fields if col in df.columns]
        df = df[available_fields]
    
    # 按日期筛选
    if start_date:
        start_date = pd.to_datetime(start_date)
        df = df[df.index >= start_date]
    if end_date:
        end_date = pd.to_datetime(end_date)
        df = df[df.index <= end_date]
    
    # 按日期排序
    df.sort_index(inplace=True)
    
    return df

def load_multi_stock_data(symbols: List[str],
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         frequency: str = 'daily',
                         fields: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    加载多个股票的历史数据
    
    Args:
        symbols: 股票代码列表
        start_date: 开始日期，可选
        end_date: 结束日期，可选  
        frequency: 数据频率，可选
        fields: 需要加载的字段列表，可选
    
    Returns:
        字典，键为股票代码，值为DataFrame
    """
    result = {}
    for symbol in symbols:
        try:
            df = load_stock_data(symbol, start_date, end_date, frequency, fields)
            result[symbol] = df
        except Exception as e:
            print(f"警告: 加载股票 {symbol} 失败: {e}")
            continue
    
    return result

# 为兼容性添加其他可能需要的函数
def get_all_securities(types: List[str] = ['stock'], date: Optional[str] = None) -> pd.DataFrame:
    """
    模拟 jqdatasdk 的 get_all_securities 函数
    返回本地可用的股票列表
    
    Args:
        types: 证券类型列表，目前仅支持 ['stock']
        date: 日期，可选
    
    Returns:
        DataFrame 包含 display_name, name, start_date, end_date 等列
    """
    if 'stock' not in types:
        raise ValueError("目前仅支持 'stock' 类型")
    
    symbols = list_available_symbols('daily')
    
    data = []
    for symbol in symbols:
        # 简单示例，实际可以从CSV中获取更多信息
        data.append({
            'display_name': symbol,
            'name': symbol,
            'start_date': '2000-01-01',  # 占位符
            'end_date': '2025-12-31',    # 占位符
            'type': 'stock'
        })
    
    return pd.DataFrame(data)

def __getattr__(name):
    """
    拦截对 mindgo_api 中其他函数的调用
    """
    raise AttributeError(
        f"模块 'data_loader' 没有属性 '{name}'。\n"
        "注意: 此模块用于替代 mindgo_api，但仅实现了部分核心函数。\n"
        "如需调用本地数据，请使用 load_stock_data, list_available_symbols 等函数。\n"
        "原 mindgo_api 函数需重写为本地数据调用。"
    )