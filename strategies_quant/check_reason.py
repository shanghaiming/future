#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from core.base_strategy import BaseStrategy
from datetime import datetime, timedelta
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import cluster
import matplotlib.dates as mdates
from numpy.lib.stride_tricks import sliding_window_view
import argparse
import datetime
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.cluster import hierarchy
from scipy.stats import linregress
import matplotlib.pyplot as plt
import argparse
import datetime
from matplotlib.dates import DateFormatter
from matplotlib.patches import Rectangle
from scipy.signal import find_peaks
from scipy import stats, signal, cluster
from matplotlib import gridspec
from matplotlib.ticker import FuncFormatter, MaxNLocator
from core.data_loader import *
import logging 

# ================== 用户自定义函数 ==================
def get_stock_list():
    """获取指定日期的股票列表"""
    table = get_all_securities(ty='stock', date="20181212")
    all_stocks = table.index.tolist()
    return all_stocks

def get_stock_data(stock, start_date, end_date):
    """获取单只股票在指定日期范围内的数据"""
   
    data = get_price(
                    securities=stock,  # 注意参数名是复数但支持单个代码
                    end_date=end_date,  # 结束日期设为今天
                    start_date=start_date,  # 结束日期设为今天
                    fre_step='1d',           # 日线频率
                    fields=['open','high','low','close','volume', 'turnover_rate'],
                    fq='pre',                # 前复权
                    #bar_count=none,           # 获取250根K线
                    skip_paused=True         # 跳过停牌日
                ).sort_index()  # 清除证券代码索引层级
    return data
    
def tsma_fast(data, n):
    """极速TSMA实现（参数n为周期数）"""
    data = np.asarray(data)
    length = len(data)
    if n <= 0 or n > length:
        return np.full_like(data, np.nan)
    
    # 生成滑动窗口视图（倒序）
    windows = sliding_window_view(data, n)[:, ::-1]
    m = len(windows)
    
    # 预计算绝对时间索引矩阵
    i_indices = np.arange(n-1, n-1 + m)  # 窗口结束的原始索引
    x = i_indices[:, None] - np.arange(n)
    
    # 批量计算核心项
    y_sum = windows.sum(axis=1)
    x_sum = x.sum(axis=1)
    xx_sum = (x ** 2).sum(axis=1)
    xy_sum = (x * windows).sum(axis=1)
    
    denominator = xx_sum - (x_sum ** 2) / n
    numerator = xy_sum - (y_sum * x_sum) / n
    k = np.divide(numerator, denominator, where=denominator != 0)
    b = (y_sum / n) - k * (x_sum / n)
    
    tsma = k * i_indices + b + k
    
    result = np.full_like(data, np.nan)
    result[n-1 : n-1 + m] = np.where(denominator != 0, tsma, np.nan)
    return result
 
def calculate_tsma_cross(stock_data):
    """计算TSMA5上穿TSMA8的信号"""
    
    
    # 计算移动平均线
    stock_data['tsma5'] = tsma_fast(stock_data['close'], 5)
    stock_data['tsma8'] = tsma_fast(stock_data['close'], 8)
    
    # 识别金叉信号
    stock_data['signal'] = (stock_data['tsma5'] > stock_data['tsma8']) & \
                           (stock_data['tsma5'].shift(1) <= stock_data['tsma8'].shift(1))
    
    return stock_data
def detect_price_ranges(close_prices, index, window=20, cluster_threshold=1.0, min_range_length=10):
    """
    检测股价区间并识别趋势
    
    参数:
        close_prices: 收盘价序列 (np.array)
        index: 对应的日期索引 (DatetimeIndex)
        window: 趋势检测窗口大小
        cluster_threshold: 聚类边界阈值
        min_range_length: 最小震荡区间长度
        
    返回:
        dict: 包含分析结果
    """
    results = {
        'trend_direction': None, 
        'is_range': False, 
        'support': None, 
        'resistance': None,
        'range_start': None,
        'range_end': None
    }
    
    # 1. 趋势检测（线性回归斜率）
    slopes = []
    for i in range(len(close_prices) - window):
        y = close_prices[i:i+window]
        x = np.arange(len(y))
        slope = stats.linregress(x, y).slope
        slopes.append(slope)
    
    avg_slope = np.mean(slopes)
    
    # 根据斜率判断趋势方向
    if avg_slope > 0.05:
        results['trend_direction'] = '上升趋势'
    elif avg_slope < -0.05:
        results['trend_direction'] = '下降趋势'
    else:
        results['trend_direction'] = '震荡趋势'
    
    # 2. 区间识别（局部极值点）
    max_indices = signal.argrelextrema(close_prices, np.greater, order=5)[0]
    min_indices = signal.argrelextrema(close_prices, np.less, order=5)[0]
    
    resistance = close_prices[max_indices]
    support = close_prices[min_indices]
    
    # 3. 边界聚类分析
    if len(resistance) > 3 and len(support) > 3:
        # 聚类分析找出主要支撑/阻力位
        resistance_centers = cluster.vq.kmeans(resistance.reshape(-1,1), 2)[0]
        support_centers = cluster.vq.kmeans(support.reshape(-1,1), 2)[0]
        
        resistance_range = np.abs(resistance_centers[0] - resistance_centers[1])
        support_range = np.abs(support_centers[0] - support_centers[1])
        
        # 判断是否形成有效区间
        if resistance_range < cluster_threshold and support_range < cluster_threshold:
            support_level = float(min(support_centers))  # 转换为float
            resistance_level = float(max(resistance_centers))  # 转换为float
            
            # 检查区间长度是否足够
            if len(close_prices) >= min_range_length:
                results['is_range'] = True
                results['support'] = support_level
                results['resistance'] = resistance_level
                results['range_start'] = index[0]
                results['range_end'] = index[-1]
    
    return results
    
    
def detect_high_low_points(df, window=5):
    """
    可靠检测高点和低点极点，并在原始DataFrame中标记相关信息
    
    :param df: 包含价格数据的DataFrame（必须有'high'和'low'列）
    :param window: 检测窗口大小（数据点）
    :return: 增强后的DataFrame，包含新列：
        - 'is_high': 是否是高点
        - 'is_low': 是否是低点
        - 'is_extreme_high': 是否是极高点（比相邻高点都高）
        - 'is_extreme_low': 是否是极低点（比相邻低点都低）
        - 'prev_high_index': 前一个高点的索引位置
        - 'prev_low_index': 前一个低点的索引位置
        - 'prev_high_price': 前一个高点的价格
        - 'prev_low_price': 前一个低点的价格
    """
    # 确保必要的列存在
    if 'high' not in df.columns or 'low' not in df.columns:
        raise ValueError("DataFrame必须包含'high'和'low'列")
    
    # 获取价格序列
    high_prices = df['high'].values
    low_prices = df['low'].values
    
    # 检测高点（局部最大值）
    high_indices = argrelextrema(high_prices, np.greater, order=window)[0]
    # 检测低点（局部最小值）
    low_indices = argrelextrema(low_prices, np.less, order=window)[0]
    
    # 创建标记列
    df = df.copy()
    df['is_high'] = False
    df['is_low'] = False
    df['is_extreme_high'] = False
    df['is_extreme_low'] = False
    
    # 标记高点和低点
    df.loc[df.index[high_indices], 'is_high'] = True
    df.loc[df.index[low_indices], 'is_low'] = True
    
    # 检测极高点
    if len(high_indices) >= 3:
        for i in range(1, len(high_indices) - 1):
            prev_idx = high_indices[i-1]
            current_idx = high_indices[i]
            next_idx = high_indices[i+1]
            
            current_high = high_prices[current_idx]
            prev_high = high_prices[prev_idx]
            next_high = high_prices[next_idx]
            
            if current_high > prev_high and current_high > next_high:
                df.loc[df.index[current_idx], 'is_extreme_high'] = True
    
    # 检测极低点
    if len(low_indices) >= 3:
        for i in range(1, len(low_indices) - 1):
            prev_idx = low_indices[i-1]
            current_idx = low_indices[i]
            next_idx = low_indices[i+1]
            
            current_low = low_prices[current_idx]
            prev_low = low_prices[prev_idx]
            next_low = low_prices[next_idx]
            
            if current_low < prev_low and current_low < next_low:
                df.loc[df.index[current_idx], 'is_extreme_low'] = True
    
    # 创建前一个高点和低点的信息列
    df['prev_high_index'] = np.nan
    df['prev_high_price'] = np.nan
    df['prev_low_index'] = np.nan
    df['prev_low_price'] = np.nan
    
    # 初始化前一个高点和低点信息
    prev_high_index = None
    prev_high_price = None
    prev_low_index = None
    prev_low_price = None
    
    # 遍历DataFrame，记录前一个高点和低点信息
    for i, (index, row) in enumerate(df.iterrows()):
        # 记录前一个高点和低点信息
        if prev_high_index is not None:
            df.at[index, 'prev_high_index'] = prev_high_index
            df.at[index, 'prev_high_price'] = prev_high_price
        
        if prev_low_index is not None:
            df.at[index, 'prev_low_index'] = prev_low_index
            df.at[index, 'prev_low_price'] = prev_low_price
        
        # 如果当前点是高点，更新前一个高点信息
        if row['is_high']:
            prev_high_index = index
            prev_high_price = row['high']
        
        # 如果当前点是低点，更新前一个低点信息
        if row['is_low']:
            prev_low_index = index
            prev_low_price = row['low']
    
    return df
 


def detect_ma_trend_change(ma_series, window=None, strictness=3):
    """
    检测均线序列是否出现过极大值（局部峰值），要求峰值大于左右各3个点
    
    参数:
    ma_series -- 均线值序列（Pandas Series），按时间顺序排列（最近的在最后）
    window -- 检测窗口大小（最近的N个数据点，None表示使用全部数据）
    strictness -- 峰值严格程度，即要求大于左右几个点（默认为3）
    
    返回:
    has_extreme -- 布尔值，表示是否出现过极大值
    extreme_indices -- 所有极大值点在原始序列中的索引位置列表
    """
    # 转换为NumPy数组
    ma_values = ma_series.values
    
    # 如果指定了检测窗口，只取最近的数据
    if window is not None and len(ma_values) > window:
        start_idx = len(ma_values) - window
        ma_values = ma_values[-window:]
    else:
        start_idx = 0
    
    # 计算需要的最小数据量
    min_points = 2 * strictness + 1
    
    # 检查数据量是否足够
    if len(ma_values) < min_points:
        return False, []
    
    # 使用argrelextrema查找局部极大值
    # order=strictness表示每个极大值点必须大于左右各strictness个点
    max_indices = argrelextrema(ma_values, np.greater, order=strictness)[0]
    
    # 映射回原始索引
    extreme_indices = [int(idx + start_idx) for idx in max_indices]
    has_extreme = len(extreme_indices) > 0
    
    return has_extreme, extreme_indices
 

    
def detect_cross_events(df, fast_col='tsma5', slow_col='tsma8', window_size=13, start_index=None, end_index=None):
    """
    检测指定窗口内是否发生过快速均线上穿慢速均线后又下穿慢速均线
    
    参数:
    df : DataFrame - 包含均线数据的DataFrame
    fast_col : str - 快速均线列名（默认'tsma5'）
    slow_col : str - 慢速均线列名（默认'tsma8'）
    window_size : int - 可选，检测窗口大小（单位：交易日）
    start_index : int - 可选，窗口起始位置索引
    end_index : int - 可选，窗口结束位置索引（包含）
    
    返回:
    int - 指定窗口内发生金叉后死叉的次数
    
    说明:
    1. 优先使用 start_index/end_index 指定的窗口
    2. 若未指定索引窗口，则使用 window_size 创建最近N日窗口
    3. 若均未指定，默认检查整个DataFrame
    """
    # 确定分析窗口
    if start_index is not None and end_index is not None:
        # 确保索引在有效范围内
        start = max(0, min(start_index, end_index))
        end = min(len(df)-1, max(start_index, end_index))
        window_data = df.iloc[start:end+1]
    elif window_size is not None:
        window_size = min(len(df), max(1, window_size))
        window_data = df.iloc[-window_size:]
    else:
        window_data = df
    
    # 初始化状态变量
    golden_cross = False
    event_count = 0
    
    # 单次遍历检测金叉后死叉的事件
    for i in range(1, len(window_data)):
        # 获取当前和前一日数据
        prev_fast = window_data[fast_col].iloc[i-1]
        prev_slow = window_data[slow_col].iloc[i-1]
        curr_fast = window_data[fast_col].iloc[i]
        curr_slow = window_data[slow_col].iloc[i]
        
        # 检查金叉：当日快速线 > 慢速线 且 前一日快速线 <= 慢速线
        if curr_fast > curr_slow and prev_fast <= prev_slow:
            golden_cross = True
            golden_index = i
        
        # 检查死叉：当日快速线 < 慢速线 且 前一日快速线 >= 慢速线
        if curr_fast < curr_slow and prev_fast >= prev_slow:
            # 如果之前有金叉且死叉在金叉之后（至少间隔1天）
            if golden_cross and i > golden_index:
                event_count += 1
                # 重置金叉状态
                golden_cross = False
    
    return event_count
    
def detect_cross_events_r(df, fast_col='tsma5', slow_col='tsma8', window_size=None, start_index=None, end_index=None):
    """
    检测指定窗口内是否发生过快速均线下穿慢速均线后又上穿慢速均线（死叉后金叉）
    
    参数:
    df : DataFrame - 包含均线数据的DataFrame
    fast_col : str - 快速均线列名（默认'tsma5'）
    slow_col : str - 慢速均线列名（默认'tsma8'）
    window_size : int - 可选，检测窗口大小（单位：交易日）
    start_index : int - 可选，窗口起始位置索引
    end_index : int - 可选，窗口结束位置索引（包含）
    
    返回:
    int - 指定窗口内发生死叉后金叉的次数
    """
    # 确定分析窗口
    if start_index is not None and end_index is not None:
        # 确保索引在有效范围内
        start = max(0, min(start_index, end_index))
        end = min(len(df)-1, max(start_index, end_index))
        window_data = df.iloc[start:end+1]
    elif window_size is not None:
        window_size = min(len(df), max(1, window_size))
        window_data = df.iloc[-window_size:]
    else:
        window_data = df
    
    # 初始化状态变量
    event_count = 0
    dead_cross_occurred = False  # 记录是否发生了死叉
    
    # 遍历检测死叉后金叉的事件
    for i in range(1, len(window_data)):
        # 获取当前和前一日数据
        prev_fast = window_data[fast_col].iloc[i-1]
        prev_slow = window_data[slow_col].iloc[i-1]
        curr_fast = window_data[fast_col].iloc[i]
        curr_slow = window_data[slow_col].iloc[i]
        
        # 1. 先检查死叉：当日快速线 < 慢速线 且 前一日快速线 >= 慢速线
        if curr_fast < curr_slow and prev_fast >= prev_slow:
            dead_cross_occurred = True  # 标记死叉已发生
        
        # 2. 再检查金叉：当日快速线 > 慢速线 且 前一日快速线 <= 慢速线
        if curr_fast > curr_slow and prev_fast <= prev_slow:
            # 如果之前发生了死叉
            if dead_cross_occurred:
                event_count += 1  # 计数死叉后金叉事件
                dead_cross_occurred = False  # 重置死叉标记
    
    return event_count
def tsma_fast(data, n):
    """极速TSMA实现（参数n为周期数）"""
    data = np.asarray(data)
    length = len(data)
    if n <= 0 or n > length:
        return np.full_like(data, np.nan)
    
    # 生成滑动窗口视图（倒序）
    windows = sliding_window_view(data, n)[:, ::-1]
    m = len(windows)
    
    # 预计算绝对时间索引矩阵
    i_indices = np.arange(n-1, n-1 + m)  # 窗口结束的原始索引
    x = i_indices[:, None] - np.arange(n)
    
    # 批量计算核心项
    y_sum = windows.sum(axis=1)
    x_sum = x.sum(axis=1)
    xx_sum = (x ** 2).sum(axis=1)
    xy_sum = (x * windows).sum(axis=1)
    
    denominator = xx_sum - (x_sum ** 2) / n
    numerator = xy_sum - (y_sum * x_sum) / n
    k = np.divide(numerator, denominator, where=denominator != 0)
    b = (y_sum / n) - k * (x_sum / n)
    
    tsma = k * i_indices + b + k
    
    result = np.full_like(data, np.nan)
    result[n-1 : n-1 + m] = np.where(denominator != 0, tsma, np.nan)
    return result


    
def calculate_williams_r(data, period=5, high_col='high', low_col='low', close_col='close'):
    """
    计算威廉指标（Williams %R），适配DataFrame输入
    
    参数：
    data : DataFrame
        必须包含最高价、最低价、收盘价列（列名可自定义）
    period : int
        计算周期（默认14）
    high_col, low_col, close_col : str
        指定最高价、最低价、收盘价的列名
    
    返回：
    Series
        Williams %R值，与原始数据索引对齐
    """
    # 确保输入是DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("输入数据必须是Pandas DataFrame")
    
    # 检查必要的列是否存在
    required_cols = [high_col, low_col, close_col]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise KeyError(f"缺失必要列: {missing_cols}")
    
    # 计算滚动窗口内的最高价和最低价
    highest_high = data[high_col].rolling(window=period, min_periods=period).max()
    lowest_low = data[low_col].rolling(window=period, min_periods=period).min()
    
    # 计算Williams %R
    numerator = highest_high - data[close_col]
    denominator = highest_high - lowest_low
    williams_r1 = (numerator / denominator) * -100
    
    # 处理分母为零的情况（设为NaN）
    williams_r1 = williams_r1.where(denominator != 0, float('nan'))
    williams_r =williams_r1.ewm(span=period, adjust=False).mean()
    
    return williams_r.rename(f'Williams_%R_{period}')    
    
def calculate_indicators(df):
    df['LC'] = df['close'].shift(1)
    df['CLOSE_LC'] = df['close'] - df['LC']
    # 计算VR指标
    av = df['volume'].where(df['close'] > df['LC'], 0)
    bv = df['volume'].where(df['close'] <= df['LC'], 0)
    df['vr'] = 100 * av.rolling(26).sum()/bv.rolling(26).sum()  
    
    
    
    
    
    # 计算MACD
    fast_ema = df['close'].ewm(span=5, adjust=False).mean()
    slow_ema = df['close'].ewm(span=13, adjust=False).mean()
    df['diff'] = fast_ema - slow_ema
    df['dea'] = df['diff'].ewm(span=8, adjust=False).mean()
    df['macd'] = 2 * (df['diff'] - df['dea'])
    # 计算布林带
    df['sma'] = df['close'].rolling(20).mean()  # 中轨
    df['std'] = df['close'].rolling(20).std()    # 标准差
    df['upper'] = df['sma'] + (df['std'] * 2)              # 上轨
    df['lower'] = df['sma'] - (df['std'] * 2) 
    
    #计算tsma
    df['tsma5'] = tsma_fast(df['close'], 5)
    df['tsma8'] = tsma_fast(df['close'], 8)
    df['tsma13'] = tsma_fast(df['close'], 13)
    df['tsma34'] = tsma_fast(df['close'], 34)
    # 计算ER（效率比率）
    change = df['close'].diff(14).abs()
    volatility = df['close'].diff().abs().rolling(14).sum()
    df['er'] = change / (volatility + 1e-7)
    # 计算均线
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma8'] = df['close'].rolling(8).mean()
    df['ma89'] = df['close'].rolling(89).mean()
    df['ma55'] = df['close'].rolling(55).mean()
    df['ma34'] = df['close'].rolling(34).mean()
    #神奇九转
    # 计算连续上涨条件
    df['A1'] = df['close'] > df['close'].shift(4)
    df['NT'] = df['A1'].astype(int).groupby((~df['A1']).cumsum()).cumcount()
    
    # 计算连续下跌条件
    df['B1'] = df['close'] < df['close'].shift(4)
    df['NT0'] = df['B1'].astype(int).groupby((~df['B1']).cumsum()).cumcount()
    
    # 初始化标记列
    df['up_mark'] = 0
    df['down_mark'] = 0

    # 处理上涨标记
    for i in range(len(df)):
        # 九转结构成立
        if df['NT'].iloc[i] == 9:
            start = max(0, i - 8)
            for j in range(start, i + 1):
                df['up_mark'].iloc[j] = j - start + 1
        
        # 最后一个K线且未完成结构
        if i == len(df) - 1 and 5 <= df['NT'].iloc[i] <= 8:
            nt = df['NT'].iloc[i]
            start = max(0, i - nt + 1)
            for j in range(start, i + 1):
                df['up_mark'].iloc[j] = j - start + 1

    # 处理下跌标记
    for i in range(len(df)):
        # 九转结构成立
        if df['NT0'].iloc[i] == 9:
            start = max(0, i - 8)
            for j in range(start, i + 1):
                df['down_mark'].iloc[j] = j - start + 1
        
        # 最后一个K线且未完成结构
        if i == len(df) - 1 and 5 <= df['NT0'].iloc[i] <= 8:
            nt = df['NT0'].iloc[i]
            start = max(0, i - nt + 1)
            for j in range(start, i + 1):
                df['down_mark'].iloc[j] = j - start + 1

    # 清理中间列
    
    # 清理中间列
    return df.drop(['LC','CLOSE_LC'], axis=1)

def detect_single_divergence(df, indicator_col='macd', window_size=30):
    """
    单一指标顶背离检测函数
    参数：
    df - 包含以下列的DataFrame:
         - high: 价格高点
         - close: 收盘价
         - [indicator_col]: 指标列（如macd/vr）
    indicator_col - 要检测的指标列名（默认'macd'）
    window_size - 检测窗口大小（默认14）
    
    返回：
    包含divergence_status列的DataFrame，取值：
    'normal' - 正常状态
    'tbd' - 待确认背离
    'confirmed' - 确认背离
    """
    df = df.copy()
    df['divergence_status'] = 'normal'
    original_index = df.index
    df = df.reset_index(drop=True)
    
    current_status = 'normal'
    reference_high = None
    reference_indicator = None
    
    for i in range(1, len(df)):
        current_high = df.at[i, 'high']
        current_close = df.at[i, 'close']
        current_indicator = df.at[i, indicator_col]
        
        # 状态转移逻辑
        if current_status == 'normal':
            start_idx = max(0, i - window_size)
            window = df.iloc[start_idx:i]
            
            if not window.empty:
                prev_high_idx = window['high'].idxmax()
                prev_high = window.at[prev_high_idx, 'high']
                prev_indicator = window.at[prev_high_idx, indicator_col]
                
                # 价格创新高但指标未新高
                if (current_high > prev_high) and (current_indicator < prev_indicator):
                    current_status = 'tbd'
                    reference_high = prev_high
                    reference_indicator = prev_indicator
        
        elif current_status == 'tbd':
            # 情况1：价格创新高
            if current_high > reference_high:
                if current_indicator >= reference_indicator:
                    current_status = 'normal'
            
            # 情况2：价格下跌
            if current_close < df.at[i-1, 'close']:
                if current_indicator < reference_indicator:
                    current_status = 'confirmed'
                else:
                    current_status = 'normal'
        
        elif current_status == 'confirmed':
            # 清除条件：价格连续2日上涨 或 指标连续2日增加
            clear_cond = False
            if i >= 2:
                price_up = (current_close > df.at[i-1, 'close']) and \
                          (df.at[i-1, 'close'] > df.at[i-2, 'close'])
                indicator_up = (current_indicator > df.at[i-1, indicator_col]) and \
                              (df.at[i-1, indicator_col] > df.at[i-2, indicator_col])
                clear_cond = price_up or indicator_up
            
            if clear_cond:
                current_status = 'normal'
                reference_high = None
                reference_indicator = None
        
        df.at[i, 'divergence_status'] = current_status
    
    df.index = original_index
    return df

def detect_pivot_points(df, distance_threshold=7, height_threshold=0.007):
    """检测关键枢轴点（尖峰后的第一根异性K棒），合并相邻且高度相近的峰值"""
    close = df['close'].values
    open_ = df['open'].values
    high = df['high'].values
    low = df['low'].values
    
    # 1. 检测上涨尖峰（局部高点）
    up_peaks = argrelextrema(close, np.greater, order=3)[0]
    
    # 2. 检测下跌极点（局部低点）
    down_peaks = argrelextrema(close, np.less, order=3)[0]
    
    # 3. 合并相邻且高度相近的峰值
    merged_up_peaks = []
    merged_down_peaks = []
    
    # 合并上涨尖峰
    i = 0
    while i < len(up_peaks):
        current_peak = up_peaks[i]
        group = [current_peak]
        j = i + 1
        
        # 寻找相邻的峰值
        while j < len(up_peaks) and (up_peaks[j] - current_peak) <= distance_threshold:
            group.append(up_peaks[j])
            j += 1
        
        # 在组内找到最高点
        highest_idx = group[0]
        for idx in group:
            if high[idx] > high[highest_idx]:
                highest_idx = idx
        
        # 检查高度是否相近
        min_height = min(high[idx] for idx in group)
        max_height = max(high[idx] for idx in group)
        height_diff = (max_height - min_height) / min_height
        
        if height_diff <= height_threshold:
            merged_up_peaks.append(highest_idx)
        else:
            # 高度差异大，保留所有点
            merged_up_peaks.extend(group)
        
        i = j
    
    # 合并下跌极点
    i = 0
    while i < len(down_peaks):
        current_peak = down_peaks[i]
        group = [current_peak]
        j = i + 1
        
        # 寻找相邻的峰值
        while j < len(down_peaks) and (down_peaks[j] - current_peak) <= distance_threshold:
            group.append(down_peaks[j])
            j += 1
        
        # 在组内找到最低点
        lowest_idx = group[0]
        for idx in group:
            if low[idx] < low[lowest_idx]:
                lowest_idx = idx
        
        # 检查高度是否相近
        min_height = min(low[idx] for idx in group)
        max_height = max(low[idx] for idx in group)
        height_diff = (max_height - min_height) / min_height
        
        if height_diff <= height_threshold:
            merged_down_peaks.append(lowest_idx)
        else:
            # 高度差异大，保留所有点
            merged_down_peaks.extend(group)
        
        i = j
    
    # 4. 检测中枢棒（尖峰后第一根反向K棒）
    pivot_bars = []
    
    # 处理上涨尖峰
    for peak_idx in merged_up_peaks:
        # 寻找尖峰后第一根阴棒（收盘<开盘）
        for i in range(peak_idx + 1, min(peak_idx + 10, len(df))):
            if close[i] < open_[i]:  # 阴线
                pivot_bar = {
                    'index': i,
                    'type': 'bearish_pivot',
                    'peak_index': peak_idx,
                    'price': (high[i] + low[i]) / 2,
                    'high': high[i],
                    'low': low[i],
                    'date': df.index[i]
                }
                pivot_bars.append(pivot_bar)
                break
    
    # 处理下跌尖峰
    for trough_idx in merged_down_peaks:
        # 寻找尖峰后第一根阳棒（收盘>开盘）
        for i in range(trough_idx + 1, min(trough_idx + 10, len(df))):
            if close[i] > open_[i]:  # 阳线
                pivot_bar = {
                    'index': i,
                    'type': 'bullish_pivot',
                    'peak_index': trough_idx,
                    'price': (high[i] + low[i]) / 2,
                    'high': high[i],
                    'low': low[i],
                    'date': df.index[i]
                }
                pivot_bars.append(pivot_bar)
                break
    
    # 对枢轴点按索引排序
    pivot_bars.sort(key=lambda x: x['index'])
    
    return merged_up_peaks, merged_down_peaks, pivot_bars
   

def cluster_pivot_points(pivot_bars, max_distance=5):
    """聚类相邻的BP和AP，识别震荡区间作为中枢"""
    if len(pivot_bars) < 2:
        return [], pivot_bars
    
    # 按索引排序枢轴点
    pivot_bars = sorted(pivot_bars, key=lambda x: x['index'])
    
    # 提取索引用于聚类
    indices = np.array([p['index'] for p in pivot_bars]).reshape(-1,1)
    if len(pivot_bars) < 2:
        return [], pivot_bars
    
    # 按索引排序枢轴点
    pivot_bars = sorted(pivot_bars, key=lambda x: x['index'])
    
    # 提取索引用于聚类
    indices = np.array([p['index'] for p in pivot_bars]).reshape(-1, 1)
    
    # 使用层次聚类
    clusters = cluster.hierarchy.fclusterdata(
        indices, 
        t=max_distance, 
        criterion='distance', 
        metric='euclidean', 
        method='single'
    )
    
    # 组织聚类结果
    clustered_zones = []
    non_clustered = []
    
    for cluster_id in np.unique(clusters):
        cluster_points = [p for i, p in enumerate(pivot_bars) if clusters[i] == cluster_id]
        
        # 如果聚类中只有一个点，则视为非震荡中枢
        if len(cluster_points) == 1:
            non_clustered.append(cluster_points[0])
            continue
        
        # 创建震荡中枢
        start_idx = min(p['index'] for p in cluster_points)
        end_idx = max(p['index'] for p in cluster_points)
        high = max(p['high'] for p in cluster_points)
        low = min(p['low'] for p in cluster_points)
        center = (high + low) / 2
        
        # 确定震荡中枢类型
        bullish_count = sum(1 for p in cluster_points if p['type'] == 'bullish_pivot')
        bearish_count = sum(1 for p in cluster_points if p['type'] == 'bearish_pivot')
        pivot_type = 'bullish' if bullish_count > bearish_count else 'bearish'
        
        clustered_zones.append({
            'start_idx': start_idx,
            'end_idx': end_idx,
            'high': high,
            'low': low,
            'center': center,
            'points': cluster_points,
            'type': pivot_type + '_zone',  # 如 'bullish_zone'
            'index': (start_idx + end_idx) // 2,  # 中枢中心位置
            'price': center,
            'date': cluster_points[len(cluster_points)//2]['date']  # 取中间点的日期
        })
    
    return clustered_zones, non_clustered

def create_zone(cluster_points):
    """从一组点创建震荡区间"""
    start_idx = min(p['index'] for p in cluster_points)
    end_idx = max(p['index'] for p in cluster_points)
    high = max(p['high'] for p in cluster_points)
    low = min(p['low'] for p in cluster_points)
    center = (high + low) / 2
    
    # 确定震荡中枢类型
    bullish_count = sum(1 for p in cluster_points if p['type'] == 'bullish_pivot')
    bearish_count = sum(1 for p in cluster_points if p['type'] == 'bearish_pivot')
    pivot_type = 'bullish' if bullish_count > bearish_count else 'bearish'
    
    return {
        'start_idx': start_idx,
        'end_idx': end_idx,
        'high': high,
        'low': low,
        'center': center,
        'points': cluster_points,
        'type': pivot_type + '_zone',
        'index': (start_idx + end_idx) // 2,
        'price': center,
        'date': cluster_points[len(cluster_points)//2]['date']
    }
    
    
def create_unified_pivots(clustered_zones, non_clustered):
    """创建统一的中枢列表（包括震荡中枢和非震荡中枢）并按时间排序"""
    # 转换非震荡中枢格式
    single_pivots = []
    for pivot in non_clustered:
        single_pivots.append({
            'type': pivot['type'] + '_pivot',
            'index': pivot['index'],
            'price': pivot['price'],
            'date': pivot['date'],
            'is_zone': False
        })
    
    # 转换震荡中枢格式
    zone_pivots = []
    for zone in clustered_zones:
        zone_pivots.append({
            'type': zone['type'],
            'index': zone['index'],
            'price': zone['price'],
            'date': zone['date'],
            'is_zone': True,
            'zone_data': zone  # 保留原始数据用于绘图
        })
    
    # 合并并排序
    all_pivots = single_pivots + zone_pivots
    all_pivots = sorted(all_pivots, key=lambda x: x['index'])
    
    return all_pivots





def detect_subwaves_by_slope(df, start_idx, end_idx, min_wave_length=2, slope_threshold=0.005, wave_number_start=1):
    """
    改进子波检测：使用更灵敏的极值点检测方法，确保波数连续
    """
    # 检查波段长度
    if end_idx <= start_idx or end_idx - start_idx < min_wave_length * 2:
        direction = 'up' if df['close'].iloc[end_idx] > df['close'].iloc[start_idx] else 'down'
        return [{
            'start': start_idx,
            'end': end_idx,
            'direction': direction,
            'wave_number': wave_number_start
        }]
    
    # 提取波段数据
    segment = df.iloc[start_idx:end_idx+1]
    prices = segment['close'].values
    
    # 1. 确定整体方向
    overall_direction = 1 if prices[-1] > prices[0] else -1
    
    # 2. 寻找波段内的主要转折点（改进的极值点检测）
    peaks = []
    troughs = []
    
    # 使用更灵敏的极值点检测
    for i in range(1, len(prices)-1):
        # 检测局部高点：比前2根和后2根都高
        if i >= 2 and i < len(prices)-2:
            if prices[i] > max(prices[i-2], prices[i-1]) and prices[i] > max(prices[i+1], prices[i+2]):
                peaks.append(i)
        
        # 检测局部低点：比前2根和后2根都低
        if i >= 2 and i < len(prices)-2:
            if prices[i] < min(prices[i-2], prices[i-1]) and prices[i] < min(prices[i+1], prices[i+2]):
                troughs.append(i)
    
    # 3. 添加更严格的转折点筛选
    filtered_points = []
    
    # 确保转折点之间有足够的价格变化
    for point in sorted(peaks + troughs):
        if not filtered_points:
            filtered_points.append(point)
            continue
            
        last_point = filtered_points[-1]
        price_change = abs(prices[point] - prices[last_point]) / prices[last_point]
        
        # 只保留价格变化超过阈值的转折点
        if price_change > slope_threshold:
            filtered_points.append(point)
    
    # 4. 合并起点、终点和过滤后的极值点
    all_points = sorted([0] + filtered_points + [len(prices)-1])
    
    # 5. 创建子波（只保留与整体方向一致的波段）
    subwaves = []
    wave_number = wave_number_start
    
    for i in range(1, len(all_points)):
        start_i = all_points[i-1]
        end_i = all_points[i]
        
        # 确保波段有最小长度
        if end_i - start_i < min_wave_length:
            continue
            
        # 计算波段方向
        start_price = prices[start_i]
        end_price = prices[end_i]
        direction = 'up' if end_price > start_price else 'down'
        
        # 只保留与整体方向一致的波段
        if (overall_direction == 1 and direction == 'up') or (overall_direction == -1 and direction == 'down'):
            # 计算波段斜率
            slope = (end_price - start_price) / (end_i - start_i) if (end_i - start_i) > 0 else 0
            
            # 只保留斜率超过阈值的波段
            if abs(slope) > slope_threshold:
                subwaves.append({
                    'start': start_idx + start_i,
                    'end': start_idx + end_i,
                    'direction': direction,
                    'wave_number': wave_number,
                    'slope': slope
                })
                wave_number += 1
    
    # 6. 如果没有检测到子波，返回整个波段
    if not subwaves:
        direction = 'up' if prices[-1] > prices[0] else 'down'
        return [{
            'start': start_idx,
            'end': end_idx,
            'direction': direction,
            'wave_number': wave_number_start
        }]
    
    return subwaves

def calculate_wave_structure(pivots, df):
    """为中枢之间的运动分配波数（严格遵循波浪理论）"""
    if len(pivots) < 2:
        return pivots, []
    
    # 初始化波浪计数
    wave_count = 1
    wave_moves = []  # 存储波浪运动的详细信息
    trend_direction = None  # 当前趋势方向
    
    # 第一个中枢没有运动
    pivots[0]['wave'] = 0
    
    # 遍历中枢之间的运动
    for i in range(1, len(pivots)):
        prev_pivot = pivots[i-1]
        curr_pivot = pivots[i]
        
        # 确定运动方向
        if curr_pivot['price'] > prev_pivot['price']:
            move_direction = 'up'
        else:
            move_direction = 'down'
        
        # 确定运动类型
        if i == 1:
            # 第一个运动总是推动波
            move_type = 'impulse'
            wave_count = 1
            trend_direction = move_direction  # 设置趋势方向
        else:
            prev_move = wave_moves[-1]
            
            # 规则1: 连续两个调整波 -> 视为反转
            if prev_move['move_type'] == 'correction' and move_direction == prev_move['direction']:
                move_type = 'impulse'
                wave_count = 1
                trend_direction = move_direction  # 反转趋势
            
            # 规则2: 调整波->推动波->调整波 且价格突破
            elif (
                len(wave_moves) >= 2 and 
                wave_moves[-2]['move_type'] == 'correction' and 
                prev_move['move_type'] == 'impulse'
            ):
                first_correction = wave_moves[-2]  # 第一个调整波
                
                # 在上升趋势中：第二个调整波（下跌）必须创更低低点
                if trend_direction == 'up' and move_direction == 'down':
                    if curr_pivot['price'] < first_correction['end_pivot']['price']:
                        move_type = 'impulse'  # 视为反转
                        wave_count = 1
                        trend_direction = 'down'  # 更新为下跌趋势
                    else:
                        move_type = 'correction'
                
                # 在下跌趋势中：第二个调整波（上涨）必须创更高高点
                elif trend_direction == 'down' and move_direction == 'up':
                    if curr_pivot['price'] > first_correction['end_pivot']['price']:
                        move_type = 'impulse'  # 视为反转
                        wave_count = 1
                        trend_direction = 'up'  # 更新为上升趋势
                    else:
                        move_type = 'correction'
                else:
                    # 其他情况按正常规则处理
                    if move_direction == trend_direction:
                        move_type = 'impulse'
                        wave_count += 1
                    else:
                        move_type = 'correction'
            
            # 正常情况：同方向为推动波，反方向为调整波
            else:
                if move_direction == trend_direction:
                    move_type = 'impulse'
                    wave_count += 1
                else:
                    move_type = 'correction'
        
        # 检测子波
        subwaves = detect_subwaves_by_slope(df, prev_pivot['index'], curr_pivot['index'])
        
        # 记录波浪运动
        move_info = {
            'start_pivot': prev_pivot,
            'end_pivot': curr_pivot,
            'direction': move_direction,
            'wave_number': wave_count if move_type == 'impulse' else 0,
            'move_type': move_type,
            'subwaves': subwaves,
            'trend_direction': trend_direction
        }
        wave_moves.append(move_info)
        
        # 为当前中枢标记波数
        curr_pivot['wave'] = wave_count if move_type == 'impulse' else 0
    
    return pivots, wave_moves
 
def detect_trend_direction(wave_moves):
    """基于波浪运动识别市场主要趋势方向"""
    if len(wave_moves) == 0:
        return None
    return wave_moves[-1]['trend_direction']

    

def identify_measurement_moves(wave_moves, df):
    """识别波浪运动的测量目标"""
    measurements = []
    
    for move in wave_moves:
        # 只对推动波进行测量
        if move['move_type'] != 'impulse':
            continue
            
        start_pivot = move['start_pivot']
        end_pivot = move['end_pivot']
        wave_number = move['wave_number']
        direction = move['direction']
        
        # 计算运动距离
        distance = abs(end_pivot['price'] - start_pivot['price'])
        
        # 预测目标位
        if direction == 'up':
            target = end_pivot['price'] + distance
        else:
            target = end_pivot['price'] - distance
        
        # 找到实际突破点
        start_idx = end_pivot['index'] + 1
        if start_idx < len(df):
            # 对于上升趋势，寻找突破高点
            if direction == 'up':
                # 找到从突破点开始的新高
                high_prices = df['high'].iloc[start_idx:]
                if len(high_prices) > 0:
                    # 获取突破点的索引位置
                    breakout_idx = high_prices.idxmax()
                    # 确保是整数索引
                    if isinstance(breakout_idx, pd.Timestamp):
                        # 如果是时间戳，转换为整数位置
                        breakout_idx = df.index.get_loc(breakout_idx)
                    breakout_price = high_prices.max()
                else:
                    breakout_idx = start_idx
                    breakout_price = df['high'].iloc[start_idx]
            # 对于下降趋势，寻找突破低点
            else:
                low_prices = df['low'].iloc[start_idx:]
                if len(low_prices) > 0:
                    breakout_idx = low_prices.idxmin()
                    # 确保是整数索引
                    if isinstance(breakout_idx, pd.Timestamp):
                        # 如果是时间戳，转换为整数位置
                        breakout_idx = df.index.get_loc(breakout_idx)
                    breakout_price = low_prices.min()
                else:
                    breakout_idx = start_idx
                    breakout_price = df['low'].iloc[start_idx]
        else:
            breakout_idx = len(df) - 1
            breakout_price = df['close'].iloc[-1]
        
        measurements.append({
            'start_pivot': start_pivot,
            'end_pivot': end_pivot,
            'distance': distance,
            'direction': direction,
            'target_price': target,
            'breakout_index': breakout_idx,
            'breakout_price': breakout_price,
            'wave_number': wave_number
        })
    
    return measurements   

def select(stock, date):
    try:
        errors = []  # 创建空列表收集错误信息
        price_data = stock[stock.index <= date]
        # 基础检查
        if len(price_data) < 100:            
            errors.append("数据不够"  )    
        # 计算20日平均换手率
        price_data['turnover_ma20'] = price_data['turnover_rate'].rolling(20).mean()
        if price_data['turnover_ma20'].iloc[-1] < 1.28:
            errors.append( "换手不够" )       
        # 计算技术指标
        hist = calculate_indicators(price_data)        
        # TSMA 条件筛选
        tsma_cond1 = hist['tsma5'].iloc[-1] < hist['tsma8'].iloc[-1]
        tsma_cond2 = hist['tsma5'].iloc[-2] > hist['tsma8'].iloc[-2]
        mean_diff = ((hist['tsma5'].tail(3) - hist['tsma8'].tail(3)).abs() / hist['tsma5'].tail(3)).mean()
        tsma_cond4 = hist['tsma5'].iloc[-1] < hist['tsma5'].iloc[-2]        
        # 使用与原始代码完全相同的条件
        tsma_cond3 = (mean_diff < 0.01)        
        if tsma_cond1 or tsma_cond2 or tsma_cond3 or tsma_cond4:
            errors.append("tsma不够")        
        # 神奇九转筛选
        if hist['up_mark'].iloc[-1] > 4  :
            errors.append( "九转不够"  )     
        # 布林带筛选
        boll_cond1 = (price_data['high'].iloc[-1] >= hist['upper'].iloc[-1]) and \
                     (price_data['close'].iloc[-1] < hist['upper'].iloc[-1])
        boll_cond2 = (price_data['high'].iloc[-1] >= hist['sma'].iloc[-1]) and \
                     (price_data['close'].iloc[-1] <= hist['sma'].iloc[-1])
        if boll_cond1 or boll_cond2:
            errors.append( "布林不够"  )      
        # 影线筛选 (防止除零错误)
        price_diff = abs(price_data['open'].iloc[-1] - price_data['close'].iloc[-1])
        if price_diff == 0:
            shadow_ratio = float('inf')
        else:
            shadow_ratio = (price_data['high'].iloc[-1] - price_data['close'].iloc[-1]) / price_diff
        if shadow_ratio > 1:
            errors.append( "影线" )       
        # WR 指标筛选
        hist['wr5'] = calculate_williams_r(hist, period=5)
        hist['wr55'] = calculate_williams_r(hist, period=55)
        wr_cond1 = (hist['wr55'].iloc[-1] < hist['wr55'].iloc[-2]) and \
                   (hist['wr55'].iloc[-1] > -50)
        wr_cond2 = (hist['wr55'].iloc[-1] > -50) and \
                   (hist['wr55'].iloc[-1] > hist['wr5'].iloc[-1])
        if wr_cond1 or wr_cond2:
            errors.append("wr"  )      
        # VR 范围筛选
        if hist['vr'].iloc[-1] < 100 or hist['vr'].iloc[-1] > 250:
            errors.append("vr"    )    
        # 背离检查
        for col in ['vr', 'wr5']:  # None表示默认指标
            result_df = detect_single_divergence(hist, indicator_col=col)
            if result_df['divergence_status'].iloc[-1] == "confirmed":
                errors.append("背离"   )    
        # 均线极性筛选
        has_change_55, _ = detect_ma_trend_change(hist['ma55'], 50)
        has_change_34, _ = detect_ma_trend_change(hist['ma34'], 30)
        if has_change_55 or has_change_34:
            errors.append("均线"      )  
        # 均线交叉筛选
        if detect_cross_events(hist, window_size=13) >= 1:
            errors.append( "均线交叉"  )      
        # 高低点检测
        hist = detect_high_low_points(hist, window=5)
        if 'prev_high_price' not in hist.columns or len(hist[hist['is_high'] == True]) == 0:
            errors.append( "高低点"     )       
        if abs(price_data['close'].iloc[-1]/hist['prev_high_price'].iloc[-1]) > 0.9:
            errors.append( "价格" )
        #89日均线
        if (abs(hist['prev_high_price'].iloc[-1] - hist['ma89'].iloc[-1]))/hist['ma89'].iloc[-1] < 0.02 and (hist['prev_high_price'].iloc[-1]/hist['prev_low_price'].iloc[-1] >1.1):
            errors.append( "89均线"  )          
        # 价格区间分析
        analysis = detect_price_ranges(
            hist['close'].values, 
            hist.index, 
            window=10, 
            cluster_threshold=1.9
        )
        if analysis['is_range']:
            if len(hist[hist['is_high'] == True]) > 0 and len(hist[hist['is_low'] == True]) > 0:
                high_idx = hist.index.get_loc(hist[hist['is_high'] == True].index[-1])
                low_idx = hist.index.get_loc(hist[hist['is_low'] == True].index[-1])
                if high_idx > low_idx:
                    if hist['close'].iloc[-1]/analysis['resistance'] > 1.05:
                        errors.append( "支持")
                else:
                    if analysis['support']/hist['close'].iloc[-1] > 1.05:
                        errors.append( "阻力")
            else:
                errors.append( "分析")
        
        # 波浪结构分析
        up_peaks, down_peaks, pivot_bars = detect_pivot_points(hist)
        clustered_zones, non_clustered = cluster_pivot_points(pivot_bars)
        all_pivots = create_unified_pivots(clustered_zones, non_clustered)
        all_pivots, wave_moves = calculate_wave_structure(all_pivots, hist)
        # 基于波浪运动识别趋势方向
        trend_direction = detect_trend_direction(wave_moves)        
        # 识别测量运动
        measurements = identify_measurement_moves(wave_moves, hist)
        t_price = pd.DataFrame(measurements)        
        if len(wave_moves) > 0:
            last_wave = wave_moves[-1]
            if (last_wave['direction'] == "down" and 
                last_wave['subwaves'][0]['wave_number'] == 1 and 
                last_wave['start_pivot']['type'] == "bearish_pivot_pivot"):
                errors.append( "zone不够" )           
        if (price_data['close'].iloc[-1] -  t_price['target_price'].iloc[-1])/price_data['close'].iloc[-1] > 0.1:
            errors.append( "target"  ) 
        if errors:
            return "，".join(errors)  # 用逗号分隔所有错误
    except Exception as e:
        # 实际使用时可记录日志
        print(f"股票 数据获取失败: {str(e)}")
        return( "error")
          
                   
            
            

            
# ==================================================

def main(start_date, end_date):
    # 结果容器
    rejection_stats = []
    daily_rejections = {}
    stock_list = get_stock_list()
    # 1. 日期循环
    current_date = datetime.datetime.strptime(start_date, "%Y%m%d")
    end_date = datetime.datetime.strptime(end_date, "%Y%m%d")
    
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        tdays_list = get_trade_days(start_date,end_date)
        if current_date not in tdays_list:
            print(f"非交易日, 跳过{current_date}")
            current_date += timedelta(days=1)
            continue
        print(f"\n处理日期: {date_str}")
        
        # 2. 获取当日股票列表
        
        if not stock_list:
            current_date += timedelta(days=1)
            continue
        
        # 3. 计算每只股票的5日涨幅
        stock_returns = {}
        for stock in stock_list:
            try:
                # 获取过去5个交易日数据
                start_dt = (current_date - timedelta(days=20)).strftime("%Y%m%d")
                stock_data = get_stock_data(stock, start_dt, date_str)
                
                
                if len(stock_data) < 5:
                    continue
                
                # 计算5日涨幅
                start_price = stock_data.iloc[-5]['close']
                end_price = stock_data.iloc[-1]['close']
                returns = (end_price - start_price) / start_price
                stock_returns[stock] = returns
            except Exception as e:
                print(f"股票 {stock} 数据获取失败: {str(e)}")
                continue
        
        # 筛选正涨幅Top100
        positive_returns = {k: v for k, v in stock_returns.items() if v > 0}
        if not positive_returns:
            current_date += timedelta(days=1)
            continue
            
        sorted_stocks = sorted(positive_returns.items(), key=lambda x: x[1], reverse=True)
        top_100 = [s[0] for s in sorted_stocks[:min(100, len(sorted_stocks))]]
        print(f"Top100股票数量: {len(top_100)}")
        
        # 4. 股票循环
        daily_rejections[date_str] = 0
        
        for stock in top_100:
            # 5. 获取足够历史数据计算金叉
            try:
                print(stock)
                # 获取过去60天数据（足够计算移动平均）
                start_dt = (current_date - timedelta(days=200)).strftime("%Y%m%d")
                stock_data = get_stock_data(stock, start_dt, date_str)
                #print(stock_data)
                if len(stock_data) < 8:  # 至少需要8天计算TSMA8
                    continue
                
                # 计算金叉信号
                stock_data = calculate_tsma_cross(stock_data)
                #print(stock_data)
                # 找到最近的金叉日期
                crossovers = stock_data[stock_data['signal']]
                
                if crossovers.empty:
                    continue
                #print(crossovers.iloc[-1])
                    
                crossover_date = crossovers.iloc[-1].name
                
                
                # 6. 金叉后日期循环
                # 获取金叉后的日期序列
                post_crossover = stock_data[stock_data.index >= crossover_date]
                if len(post_crossover) == 0:
                    continue
                
                # 取金叉日及之后5个交易日
                decision_days = post_crossover.index
                if len(decision_days) > 1:
                    decision_days = decision_days[:1]
                
                # 7. 对每个决策日调用筛选函数
                for decision_date in decision_days:
                    reason = select(stock_data, decision_date)
                    print(reason)
                    if reason:
                        # 8. 记录结果
                        rejection_stats.append({
                            'ts_code': stock,
                            'decision_date': decision_date,
                            'analysis_date': date_str,
                            'crossover_date': crossover_date,
                            'reason': reason
                        })
                        daily_rejections[date_str] += 1
            except Exception as e:
                print(f"股票 {stock} 处理失败: {str(e)}")
                continue
        
        current_date += timedelta(days=1)
    
    # 结果分析
    if not rejection_stats:
        print("\n没有未通过筛选的记录")
        return
        
    df = pd.DataFrame(rejection_stats)
    
    # 可视化
    plt.figure(figsize=(15, 10))
    
    # 1. 未通过原因分布
    plt.subplot(2, 2, 1)
    reason_counts = df['reason'].value_counts()
    reason_counts.plot(kind='bar', color='skyblue')
    plt.title('未通过原因分布')
    plt.ylabel('次数')
    plt.xticks(rotation=45)
    
    # 2. 按股票统计
    plt.subplot(2, 2, 2)
    stock_rejections = df.groupby('ts_code')['reason'].count().nlargest(20)
    stock_rejections.plot(kind='bar', color='salmon')
    plt.title('股票未通过次数TOP20')
    plt.ylabel('未通过次数')
    
    # 3. 时间趋势
    plt.subplot(2, 2, 3)
    daily_counts = pd.Series(daily_rejections)
    daily_counts.plot(kind='line', marker='o', color='green')
    plt.title('每日未通过次数趋势')
    plt.ylabel('次数')
    plt.grid(True)
    
    # 4. 原因时间分布
    plt.subplot(2, 2, 4)
    for reason in reason_counts.index[:3]:  # 显示前3个原因
        reason_df = df[df['reason'] == reason]
        reason_daily = reason_df.groupby('decision_date').size()
        reason_daily.plot(label=reason)
    plt.title('主要原因时间分布')
    plt.ylabel('次数')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('rejection_analysis.png')
    plt.show()
    
    # 输出详细结果
    df.to_csv('rejection_details.csv', index=False)
    
    # 打印摘要统计
    print("\n===== 统计摘要 =====")
    print(f"总未通过次数: {len(df)}")
    print(f"涉及股票数量: {df['ts_code'].nunique()}")
    print(f"时间范围: {df['decision_date'].min()} 至 {df['decision_date'].max()}")
    print("\n主要未通过原因:")
    print(reason_counts.head(10))
    
    return df

# 使用示例
if __name__ == "__main__":
    start_date = "20240331"
    end_date = "20250331"
    results = main(start_date, end_date)

# In[ ]:






class CheckReasonStrategy(BaseStrategy):
    """基于check_reason的策略"""
    
    def __init__(self, data, params=None):
        super().__init__(data, params)
        # 初始化代码
        self.name = "CheckReasonStrategy"
        self.description = "基于check_reason的策略"
        
    def calculate_signals(self, df):
        """计算交易信号"""
        # 策略逻辑
        return df
        
    def generate_signals(self, df):
        """生成交易信号"""
        # 信号生成逻辑
        return df
