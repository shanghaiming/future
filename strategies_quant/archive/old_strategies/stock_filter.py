#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
from core.base_strategy import BaseStrategy
A股股票多周期方差分析程序
依赖：mindgp_api, numpy, pandas, tqdm
"""
from core.base_strategy import BaseStrategy
import numpy as np
import pandas as pd
from tqdm import tqdm
from core.data_loader import *  # 假设该库直接提供API函数
from numpy.lib.stride_tricks import sliding_window_view
import datetime
try:
    import ipywidgets as widgets
    _HAS_IPYWIDGETS = True
except ImportError:
    _HAS_IPYWIDGETS = False
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
from scipy.stats import linregress
from scipy import stats, signal, cluster
from concurrent.futures import ThreadPoolExecutor, as_completed


# 配置参数
CONFIG = {
    "PERIODS": [5, 8, 13, 21, 34, 55],  # 需要计算的周期
    "MIN_DATA_DAYS": 55,                # 需要的最少数据天数
    "BATCH_SIZE": 300,                  # 分批处理数量
    "OUTPUT_FILE": "stock_variance_rank.csv"  # 输出文件名
}


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
    
    
def v_shape_after_low(x, a, b, d):
    """V型反转模型（低点后）：快速回升"""
    return a * x + b * np.exp(-d * x)

# 低点后的U型反转
def u_shape_after_low(x, a, b, c):
    """U型反转模型（低点后）：缓慢回升形成圆弧底"""
    return a * (x - b)**2 + c

# 高点后的V型反转
def v_shape_after_high(x, a, b, d):
    """V型反转模型（高点后）：快速下跌"""
    return -a * x + b * np.exp(-d * x)

# 高点后的U型反转
def u_shape_after_high(x, a, b, c):
    """U型反转模型（高点后）：缓慢下跌形成圆弧顶"""
    return -a * (x - b)**2 + c

# 计算拟合优度
def r_squared(y, y_fit):
    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    if ss_tot == 0:
        return -np.inf
    return 1 - (ss_res / ss_tot)

# 反转点检测与分析函数
def detect_reversals(df, window_size=30, min_r2=0.6):
    """
    检测股价中的V型和U型反转点
    
    参数:
    df -- 包含OHLC数据的DataFrame (列: open, high, low, close)
    window_size -- 分析窗口大小 (默认30)
    min_r2 -- 最小拟合优度阈值 (默认0.6)
    
    返回:
    reversal_points -- 检测到的反转点信息列表
    """
    prices = df['close'].values
    
    # 检测局部低点和高点
    min_idx = argrelextrema(prices, np.less, order=5)[0]  # 低点
    max_idx = argrelextrema(prices, np.greater, order=5)[0]  # 高点
    
    reversal_points = []
    wave_counter = 1  # 全局波次计数器
    last_reversal_end = -1  # 上一个反转的结束位置
    
    # 按时间顺序处理所有候选点
    candidate_points = sorted(np.concatenate([min_idx, max_idx]))
    
    for idx in candidate_points:
        # 跳过无法分析的边缘点
        if idx + window_size >= len(prices):
            continue
        
        # 确定当前点是低点还是高点
        direction = 'low' if idx in min_idx else 'high'
        
        # 提取反转点后的价格窗口
        window_data = prices[idx:idx+window_size]
        x = np.arange(len(window_data))
        
        # 根据方向选择拟合函数
        if direction == 'low':
            # 尝试拟合V型
            try:
                v_params, _ = curve_fit(v_shape_after_low, x, window_data, 
                                        p0=[0.5, window_data[0], 0.1],
                                        maxfev=5000)
                v_fitted = v_shape_after_low(x, *v_params)
                v_score = r_squared(window_data, v_fitted)
            except:
                v_score = -np.inf
            
            # 尝试拟合U型
            try:
                u_params, _ = curve_fit(u_shape_after_low, x, window_data, 
                                        p0=[0.1, 10, np.min(window_data)],
                                        maxfev=5000)
                u_fitted = u_shape_after_low(x, *u_params)
                u_score = r_squared(window_data, u_fitted)
            except:
                u_score = -np.inf
        else:
            # 尝试拟合V型（下降）
            try:
                v_params, _ = curve_fit(v_shape_after_high, x, window_data, 
                                        p0=[0.5, window_data[0], 0.1],
                                        maxfev=5000)
                v_fitted = v_shape_after_high(x, *v_params)
                v_score = r_squared(window_data, v_fitted)
            except:
                v_score = -np.inf
            
            # 尝试拟合U型（下降）
            try:
                u_params, _ = curve_fit(u_shape_after_high, x, window_data, 
                                        p0=[0.1, 10, np.max(window_data)],
                                        maxfev=5000)
                u_fitted = u_shape_after_high(x, *u_params)
                u_score = r_squared(window_data, u_fitted)
            except:
                u_score = -np.inf
        
        # 记录有效反转点
        if max(v_score, u_score) > min_r2:
            reversal_type = 'V' if v_score > u_score else 'U'
            best_score = max(v_score, u_score)
            
            # 波次计数逻辑：如果当前反转点在上一个反转结束之前，增加波次
            if idx < last_reversal_end:
                wave_counter += 1
            else:
                wave_counter = 1  # 新的趋势开始
            
            # 更新上一个反转结束位置
            last_reversal_end = idx + window_size
            
            reversal_points.append({
                'index': idx,
                'date': df.index[idx],
                'type': reversal_type,
                'score': best_score,
                'direction': direction,  # 低点或高点反转
                'window_start': idx,
                'window_end': idx + window_size,
                'params': v_params if reversal_type == 'V' else u_params,
                'wave': wave_counter  # 当前波次
            })
    
    return reversal_points

    
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
    df['ma5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ma8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ma55'] = df['close'].ewm(span=55, adjust=False).mean()
    df['ma34'] = df['close'].ewm(span=34, adjust=False).mean()
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
def detect_ma_trend_change(ma_series, window=None):
    """
    检测均线斜率是否出现由正转负的变化
    
    参数:
    ma_series -- 均线值序列（Pandas Series），按时间顺序排列（最近的在最后）
    window -- 检测窗口大小（最近的N个数据点，None表示使用全部数据）
    
    返回:
    has_change -- 布尔值，表示是否出现过斜率由正转负
    change_indices -- 所有转变点在原始序列中的索引位置列表
    """
    # 转换为NumPy数组
    ma_values = ma_series.values
    
    # 如果指定了检测窗口，只取最近的数据
    if window is not None and len(ma_values) > window:
        ma_values = ma_values[-window:]
    
    # 检查数据量是否足够（至少需要2个点计算差分）
    if len(ma_values) < 2:
        return False, []
    
    # 计算一阶差分（斜率近似）
    slopes = np.diff(ma_values)
    
    # 计算斜率方向（1=正，-1=负）
    slope_directions = np.sign(slopes)
    
    # 检测方向变化点（前一天斜率为正，当天斜率为负）
    # 注意：差分结果比原数组少一个元素
    change_points = (slope_directions[1:] < 0) & (slope_directions[:-1] > 0)
    
    # 获取所有转变点的索引（在截取窗口中的位置）
    change_indices = np.where(change_points)[0] + 1
    
    # 如果截取了窗口，需要映射回原始索引
    if window is not None and len(ma_series) > window:
        start_idx = len(ma_series) - window
        change_indices = change_indices + start_idx
    
    has_change = len(change_indices) == 1
    
    return has_change, change_indices.tolist()
    
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
    df = df.rename(columns={indicator_col.upper(): indicator_col}, errors='ignore').copy()
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
# 使用示例
# 假设df包含high、MACD、VR列
# result_df = detect_top_divergence(df, window_size=14)

def filter_stocks(
    all_stocks, 
    table, 
    CONFIG, 
    history_func, 
    calculate_indicators_func,
    calculate_williams_r_func,
    detect_single_divergence_func,
    detect_ma_trend_change_func,
    detect_cross_events_func,
    detect_high_low_points_func,
    detect_price_ranges_func,
    detect_pivot_points_func,
    cluster_pivot_points_func,
    create_unified_pivots_func,
    calculate_wave_structure_func,
    max_workers=8
):
    """
    批量筛选符合条件的股票 (线程安全版本)
    
    参数:
    all_stocks -- 待筛选的股票代码列表
    table -- 股票信息表，包含display_name等信息
    CONFIG -- 配置字典，包含MIN_DATA_DAYS, BATCH_SIZE等配置
    [各种技术指标计算函数] -- 所需的技术指标计算和分析函数
    max_workers -- 并行工作线程数，默认为8
    
    返回:
    (results, valid_stock_count) -- 符合条件的股票列表和有效数量
    """
    
    # 处理单只股票的筛选逻辑
    def _process_single_stock(stock_code):
        
        try:
            # 获取历史数据
            price_data = history_func(
                        securities=stock_code,  # 注意参数名是复数但支持单个代码
                        end_date=datetime.date.today().strftime('%Y%m%d'),  # 结束日期设为今天
                        fre_step='1d',           # 日线频率
                        fields=['open','high','low','close','volume', 'turnover_rate'],
                        fq='pre',                # 前复权
                        bar_count=250,           # 获取250根K线
                        skip_paused=True         # 跳过停牌日
                        ).sort_index()  # 清除证券代码索引层级
            
            
            # 基础检查
            if len(price_data) < CONFIG["MIN_DATA_DAYS"]:
                #log.info("数据不够")
                return None
            
            
            
            # 计算20日平均换手率
            price_data['turnover_ma20'] = price_data['turnover_rate'].rolling(20).mean()
            if price_data['turnover_ma20'].iloc[-1] < 1:
                log.info("换手不够")
                return None
            
            # 计算技术指标
            hist = calculate_indicators_func(price_data)
            
            # TSMA 条件筛选
            tsma_cond1 = hist['tsma5'].iloc[-1] < hist['tsma8'].iloc[-1]
            tsma_cond2 = hist['tsma5'].iloc[-2] > hist['tsma8'].iloc[-2]
            mean_diff = ((hist['tsma5'].tail(3) - hist['tsma8'].tail(3)).abs() / hist['tsma5'].tail(3)).mean()
            tsma_cond4 = hist['tsma5'].iloc[-1] < hist['tsma5'].iloc[-2]

            # 使用与原始代码完全相同的条件
            tsma_cond3 = (mean_diff < 0.01)
            
            '''if tsma_cond1 or tsma_cond4:
                log.info("tsma不够")
                return None'''
            
            # 神奇九转筛选
            if hist['up_mark'].iloc[-1] > 7  :
                #log.info("九转不够")
                return None
            
            
            # 布林带筛选
            boll_cond1 = (price_data['high'].iloc[-1] >= hist['upper'].iloc[-1]) and \
                         (price_data['close'].iloc[-1] < hist['upper'].iloc[-1])
            boll_cond2 = (price_data['high'].iloc[-1] >= hist['sma'].iloc[-1]) and \
                         (price_data['close'].iloc[-1] <= hist['sma'].iloc[-1])
            '''if boll_cond1 or boll_cond2:
                #log.info("布林不够")
                return None'''
            
            # 影线筛选 (防止除零错误)
            price_diff = abs(price_data['open'].iloc[-1] - price_data['close'].iloc[-1])
            if price_diff == 0:
                shadow_ratio = float('inf')
            else:
                shadow_ratio = (price_data['high'].iloc[-1] - price_data['close'].iloc[-1]) / price_diff
            '''if shadow_ratio > 1:
                #log.info("影线")
                return None'''
            
            # WR 指标筛选
            hist['wr5'] = calculate_williams_r_func(hist, period=5)
            hist['wr55'] = calculate_williams_r_func(hist, period=55)
            '''wr_cond1 = (hist['wr55'].iloc[-1] < hist['wr55'].iloc[-2]) and \
                       (hist['wr55'].iloc[-1] > -50)
            wr_cond2 = (hist['wr55'].iloc[-1] > -50) and \
                       (hist['wr55'].iloc[-1] > hist['wr5'].iloc[-1])
            if wr_cond1 or wr_cond2:
                #log.info("wr")
                return None'''
            
            # VR 范围筛选
            '''if hist['vr'].iloc[-1] < 100 or hist['vr'].iloc[-1] > 250:
                #log.info("vr")
                return None'''
            
            # 背离检查
            for col in ['vr', 'wr5']:  # None表示默认指标
                result_df = detect_single_divergence_func(hist, indicator_col=col)
                if result_df['divergence_status'].iloc[-1] == "confirmed":
                    #log.info("背离")
                    return None
            
            
            # 均线极性筛选
            '''has_change_55, _ = detect_ma_trend_change_func(hist['ma55'], 50)
            has_change_34, _ = detect_ma_trend_change_func(hist['ma34'], 30)
            if has_change_55 or has_change_34:
                #log.info("均线")
                return None
            
            # 均线交叉筛选
            if detect_cross_events_func(hist, window_size=13) > 1:
                log.info("均线交叉")
                return None'''
            
            # 高低点检测
            '''hist = detect_high_low_points_func(hist, window=5)
            if 'prev_high_price' not in hist.columns or len(hist[hist['is_high'] == True]) == 0:
                #log.info("高低点")
                return None
                
            if abs(price_data['close'].iloc[-1]/hist['prev_high_price'].iloc[-1]) > 0.9:
                #log.info("价格")
                return None 
            #89日均线
            if (abs(hist['prev_high_price'].iloc[-1] - hist['ma89'].iloc[-1]))/hist['ma89'].iloc[-1] < 0.02 and (hist['prev_high_price'].iloc[-1]/hist['prev_low_price'].iloc[-1] >1.1):
                return None'''
                
            # 价格区间分析
            '''analysis = detect_price_ranges_func(
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
                            #log.info("支持")
                            return None
                    else:
                        if analysis['support']/hist['close'].iloc[-1] > 1.05:
                            #log.info("阻力")
                            return None 
                else:
                    #log.info("分析")
                    return None
            
            # 波浪结构分析
            up_peaks, down_peaks, pivot_bars = detect_pivot_points_func(hist)
            clustered_zones, non_clustered = cluster_pivot_points_func(pivot_bars)
            all_pivots = create_unified_pivots_func(clustered_zones, non_clustered)
            all_pivots, wave_moves = calculate_wave_structure_func(all_pivots, hist)
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
                    #log.info("zone不够")
                    return None
                else:
                    pass
                    #log.info(last_wave['direction'], last_wave['subwaves'][0]['wave_number'], last_wave['start_pivot']['type'])
            if (price_data['close'].iloc[-1] -  t_price['target_price'].iloc[-1])/price_data['close'].iloc[-1] > 0.1:
                return None'''
            # 通过所有筛选条件
            return {
                "股票代码": stock_code,
                "名字": table.loc[stock_code, 'display_name'],
                "成本": price_data['close'].iloc[-1],
                "wr": hist['wr5'].iloc[-1],
                "expect_limit": t_price['target_price'].iloc[-1]/price_data['close'].iloc[-1],
                "low": hist.loc[hist['tsma5'] < hist['tsma8'], 'low'].iloc[-1] if (hist['tsma5'] < hist['tsma8']).any() else None
            }
            
        except Exception as e:
            # 实际使用时可记录日志
            logger.warning(f"{stock_code}处理失败: {str(e)}")
            return None
    
    # 主筛选逻辑
    results = []
    valid_stock_count = 0
    batch_size = CONFIG["BATCH_SIZE"]
    
    # 使用线程池处理 (避免多进程嵌套问题)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        
        # 分批提交任务
        for i in range(0, len(all_stocks), batch_size):
            batch = all_stocks[i:i+batch_size]
            for stock_code in batch:
                
                future = executor.submit(_process_single_stock, stock_code)
                futures[future] = stock_code
        
        # 处理完成的任务
        for future in as_completed(futures):
            stock_code = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    valid_stock_count += 1
            except Exception as e:
                # 处理异常
                # logger.error(f"处理{stock_code}时出错: {str(e)}")
                pass
    
    return results, valid_stock_count
    

def main():
    # 获取全量股票池
    print("正在获取股票列表...")
    table = get_all_securities(ty='stock', date=None)
    all_stocks = get_all_securities(ty='stock', date=None).index.tolist()
    #all_stocks = ['603058.SH','688605.SH']
    print(f"共获取到{len(all_stocks)}只A股")
    # 初始化结果容器
    results = []
    valid_stock_count = 0
    
    results, valid_count = filter_stocks(
    all_stocks=all_stocks,
    table=table,
    CONFIG=CONFIG,
    history_func=get_price,
    calculate_indicators_func=calculate_indicators,
    calculate_williams_r_func=calculate_williams_r,
    detect_single_divergence_func=detect_single_divergence,
    detect_ma_trend_change_func=detect_ma_trend_change,
    detect_cross_events_func=detect_cross_events,
    detect_high_low_points_func=detect_high_low_points,
    detect_price_ranges_func=detect_price_ranges,
    detect_pivot_points_func=detect_pivot_points,
    cluster_pivot_points_func=cluster_pivot_points,
    create_unified_pivots_func=create_unified_pivots,
    calculate_wave_structure_func=calculate_wave_structure,
    max_workers=1  # 根据CPU核心数调整
    )
    
    # 处理结果
    if not results:
        print("没有有效数据可供分析")
        return

    # 创建DataFrame并排序
    df = pd.DataFrame(results).sort_values(by="wr5", ascending=True)
    
    # 保存结果
    df.to_csv(CONFIG["OUTPUT_FILE"], index=False, encoding='utf_8_sig')
    
    # 打印摘要
    print("\n" + "="*50)
    print(f"分析完成！有效处理股票数：{valid_stock_count}/{len(all_stocks)}")
    #print(f"方差最小的30只股票：\n{df[['股票代码', '名字', '平均方差']].head(30).to_string(index=False)}")
    print(f"完整结果已保存至：{CONFIG['OUTPUT_FILE']}")
    jupyter_stock_charts(df.head(70))  # 使用Jupyter专用控件
    '''notify_push(
    df[['股票代码', '名字', '平均方差']].head(10).to_string(index=False), 
    channel='wxpusher', 
    subject='SuperMind消息提醒', 
    email_list=None, 
    uids='UID_whd6sWkQtfsrIEFEifgZZWyLufEI', 
    topic_ids=None, 
    group_id=None,
    url=None,
    payload=None,
)'''
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
try:
    import ipywidgets as widgets
    _HAS_IPYWIDGETS_2 = True
except ImportError:
    _HAS_IPYWIDGETS_2 = False
import datetime
import pandas as pd
import numpy as np

def jupyter_stock_charts(df_results):
    """Jupyter专用股票图表浏览器（带均线和技术指标）"""
    if df_results.empty:
        print("没有股票可展示")
        return
    
    # 准备数据
    stock_codes = df_results["股票代码"].tolist()
    stock_names = df_results["名字"].tolist()
    current_index = 0
    
    # 创建控件
    prev_btn = widgets.Button(description="上一只")
    next_btn = widgets.Button(description="下一只")
    index_label = widgets.Label(value=f"股票: 1/{len(stock_codes)}")
    
    # 创建图表区域
    output = widgets.Output()
    
    # 定义更新图表函数
    def update_chart():
        with output:
            output.clear_output(wait=True)
            
            stock_code = stock_codes[current_index]
            stock_name = stock_names[current_index]
            
            # 获取股票日线数据
            # 这里假设get_price函数已定义或从某处导入
            price_data = get_price(
                securities=stock_code,
                end_date=datetime.date.today().strftime('%Y%m%d'),
                fre_step='1d',
                fields=['open', 'high', 'low', 'close', 'volume'],
                fq='pre',
                bar_count=250,  # 显示250天数据
                skip_paused=True
            )
            
            if price_data.empty:
                fig, ax = plt.subplots(figsize=(40, 4))
                ax.set_title(f"{stock_name} ({stock_code}) - 数据获取失败", fontsize=14)
                plt.tight_layout()
                plt.show()
                return
            
            # 计算技术指标
            price_data['LC'] = price_data['close'].shift(1)
            price_data['CLOSE_LC'] = price_data['close'] - price_data['LC']
            
            # 计算VR指标
            av = price_data['volume'].where(price_data['close'] > price_data['LC'], 0)
            bv = price_data['volume'].where(price_data['close'] < price_data['LC'], 0)
            cv = price_data['volume'].where(price_data['close'] == price_data['LC'], 0)
            price_data['vr'] = (av.rolling(24).sum() + cv.rolling(24).sum()/2) / \
                              (bv.rolling(24).sum() + cv.rolling(24).sum()/2 + 1e-7) * 100
            
            # 计算ER（效率比率）
            change = price_data['close'].diff(24).abs()
            volatility = price_data['close'].diff().abs().rolling(24).sum()
            price_data['er_raw'] = change / (volatility + 1e-7)
            price_data['er'] = price_data['er_raw'].ewm(span=5, adjust=False).mean()
            
            # 计算MACD作为VAR指标
            fast_ema = price_data['close'].ewm(span=12, adjust=False).mean()
            slow_ema = price_data['close'].ewm(span=26, adjust=False).mean()
            price_data['diff'] = fast_ema - slow_ema
            price_data['dea'] = price_data['diff'].ewm(span=9, adjust=False).mean()
            price_data['macd'] = 2 * (price_data['diff'] - price_data['dea'])
            
            # 计算各周期方差 (VAR)
            closes = price_data['close'].values.astype(float)
            periods = [5, 8, 13, 21, 34, 55]  # 方差计算周期
            
            for period in periods:
                # 计算滚动方差
                price_data[f'var_{period}'] = price_data['close'].rolling(window=period).var(ddof=1) / (closes ** 2)
            
            # 创建整数索引（去除非交易日间隙）
            price_data['index_num'] = range(len(price_data))
            
            # 创建5个子图（K线图 + 4个指标）
            fig = plt.figure(figsize=(6, 8))
            gs = gridspec.GridSpec(5, 1, height_ratios=[5, 1, 1, 1, 1], hspace=0)
            ax1 = plt.subplot(gs[0])  # K线图
            ax2 = plt.subplot(gs[1], sharex=ax1)  # MACD
            ax3 = plt.subplot(gs[2], sharex=ax1)  # VR
            ax4 = plt.subplot(gs[3], sharex=ax1)  # ER
            ax5 = plt.subplot(gs[4], sharex=ax1)  # VAR (方差)
            
            # 绘制K线图
            candle_width = 0.8
            up = price_data[price_data.close >= price_data.open]
            down = price_data[price_data.close < price_data.open]
            
            # 上涨K线
            ax1.bar(up['index_num'], up.close - up.open, candle_width, 
                   bottom=up.open, color='red', zorder=3)
            ax1.bar(up['index_num'], up.high - up.close, 0.15, 
                   bottom=up.close, color='red', zorder=3)
            ax1.bar(up['index_num'], up.low - up.open, 0.15, 
                   bottom=up.open, color='red', zorder=3)
            
            # 下跌K线
            ax1.bar(down['index_num'], down.close - down.open, candle_width, 
                   bottom=down.open, color='green', zorder=3)
            ax1.bar(down['index_num'], down.high - down.open, 0.15, 
                   bottom=down.open, color='green', zorder=3)
            ax1.bar(down['index_num'], down.low - down.close, 0.15, 
                   bottom=down.close, color='green', zorder=3)
            
            # 计算并绘制均线
            price_data['MA5'] = price_data['close'].rolling(window=5).mean()
            price_data['MA10'] = price_data['close'].rolling(window=10).mean()
            price_data['MA20'] = price_data['close'].rolling(window=20).mean()
            
            ax1.plot(price_data['index_num'], price_data['MA5'], 'b-', linewidth=1.2, label='5日均线', zorder=2)
            ax1.plot(price_data['index_num'], price_data['MA10'], 'm-', linewidth=1.2, label='10日均线', zorder=2)
            ax1.plot(price_data['index_num'], price_data['MA20'], 'c-', linewidth=1.2, label='20日均线', zorder=2)
            
            # 设置K线图标题和标签
            last_close = price_data['close'].iloc[-1]
            last_date = price_data.index[-1].strftime('%Y-%m-%d')
            ax1.set_title(f"{stock_name} ({stock_code}) - 最新价: {last_close:.2f} ({last_date})", 
                         fontsize=16, fontweight='bold')
            ax1.set_ylabel('价格')
            ax1.legend(loc='upper left')
            ax1.grid(True, linestyle='--', alpha=0.6)
            
            # 绘制MACD指标
            colors = ['red' if val >= 0 else 'green' for val in price_data['macd']]
            ax2.bar(price_data['index_num'], price_data['macd'], color=colors, width=0.8)
            ax2.plot(price_data['index_num'], price_data['diff'], 'b-', linewidth=1.0, label='DIFF')
            ax2.plot(price_data['index_num'], price_data['dea'], 'm-', linewidth=1.0, label='DEA')
            ax2.axhline(0, color='gray', linestyle='-', linewidth=0.7)
            ax2.set_ylabel('MACD')
            ax2.legend(loc='upper left')
            ax2.grid(True, linestyle='--', alpha=0.4)
            
            # 绘制VR指标
            ax3.plot(price_data['index_num'], price_data['vr'], 'b-', linewidth=1.2)
            ax3.axhline(100, color='gray', linestyle='--', linewidth=0.7)
            ax3.set_ylabel('VR')
            ax3.grid(True, linestyle='--', alpha=0.4)
            
            # 绘制ER指标
            ax4.plot(price_data['index_num'], price_data['er'], 'g-', linewidth=1.2)
            ax4.axhline(0.5, color='gray', linestyle='--', linewidth=0.7)
            ax4.set_ylabel('ER')
            ax4.grid(True, linestyle='--', alpha=0.4)
            
            # 绘制VAR（方差）指标
            colors = ['b', 'g', 'r', 'c', 'm']
            for i, period in enumerate(periods):
                ax5.plot(price_data['index_num'], price_data[f'var_{period}'], 
                         color=colors[i % len(colors)], 
                         linewidth=1.2, 
                         label=f'{period}日方差')
            
            ax5.set_ylabel('VAR')
            ax5.set_xlabel('交易日序列')
            ax5.legend(loc='upper left')
            ax5.grid(True, linestyle='--', alpha=0.4)
            
            # 设置x轴刻度（只显示在最后一个子图）
            n = len(price_data)
            step = max(1, n // 10)
            xticks = list(range(0, n, step))
            if n-1 not in xticks:
                xticks.append(n-1)
            xticklabels = [price_data.index[i].strftime('%m-%d') for i in xticks]
            ax5.set_xticks(xticks)
            ax5.set_xticklabels(xticklabels, rotation=45)
            
            # 设置y轴范围
            y_min = price_data[['low', 'MA5', 'MA10', 'MA20']].min().min()
            y_max = price_data[['high', 'MA5', 'MA10', 'MA20']].max().max()
            ax1.set_ylim(y_min * 0.98, y_max * 1.02)
            
            # 隐藏上方子图的x轴标签
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.setp(ax2.get_xticklabels(), visible=False)
            plt.setp(ax3.get_xticklabels(), visible=False)
            plt.setp(ax4.get_xticklabels(), visible=False)
            
            plt.tight_layout()
            plt.show()
    
    # 定义按钮回调函数
    def on_next_btn(b):
        nonlocal current_index
        current_index = (current_index + 1) % len(stock_codes)
        index_label.value = f"股票: {current_index+1}/{len(stock_codes)}"
        update_chart()
    
    def on_prev_btn(b):
        nonlocal current_index
        current_index = (current_index - 1) % len(stock_codes)
        current_index = current_index if current_index >= 0 else len(stock_codes) - 1
        index_label.value = f"股票: {current_index+1}/{len(stock_codes)}"
        update_chart()
    
    # 绑定按钮事件
    prev_btn.on_click(on_prev_btn)
    next_btn.on_click(on_next_btn)
    
    # 创建控制面板
    controls = widgets.HBox([prev_btn, index_label, next_btn])
    
    # 初始显示
    update_chart()
    
    # 显示所有控件
    display(controls, output)
if __name__ == "__main__":
       
    main()

# In[ ]:






class StockFilterStrategy(BaseStrategy):
    """基于stock_filter的策略"""

    def __init__(self, data: pd.DataFrame, params: dict = None):
        super().__init__(data, params)

    def generate_signals(self):
        """生成交易信号"""
        df = self.data
        if len(df) < 2:
            return self.signals

        close = df['close']
        ma_short = close.rolling(5).mean()
        ma_long = close.rolling(20).mean()

        for i in range(1, len(df)):
            if pd.isna(ma_long.iloc[i]):
                continue
            if ma_short.iloc[i] > ma_long.iloc[i] and ma_short.iloc[i-1] <= ma_long.iloc[i-1]:
                self._record_signal(timestamp=df.index[i], action='buy', price=float(close.iloc[i]))
            elif ma_short.iloc[i] < ma_long.iloc[i] and ma_short.iloc[i-1] >= ma_long.iloc[i-1]:
                self._record_signal(timestamp=df.index[i], action='sell', price=float(close.iloc[i]))

        return self.signals
