#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
from core.base_strategy import BaseStrategy
A股期货多周期方差分析程序
依赖：mindgp_api, numpy, pandas, tqdm
"""
from core.base_strategy import BaseStrategy
import numpy as np
import pandas as pd
from tqdm import tqdm
from core.data_loader import *  # 假设该库直接提供API函数
from numpy.lib.stride_tricks import sliding_window_view
import datetime
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
try:
    import ipywidgets as widgets
    _HAS_IPYWIDGETS = True
except ImportError:
    _HAS_IPYWIDGETS = False
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
from scipy.stats import linregress
from scipy import stats, signal, cluster
from sklearn.cluster import KMeans

# 配置参数
CONFIG = {
    "PERIODS": [5, 8, 13, 21, 34, 55],  # 需要计算的周期
    "MIN_DATA_DAYS": 55,                # 需要的最少数据天数
    "BATCH_SIZE": 20
}

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
    bv = df['volume'].where(df['close'] < df['LC'], 0)
    cv = df['volume'].where(df['close'] == df['LC'], 0)
    df['vr'] = (av.rolling(24).sum() + cv.rolling(24).sum()/2) / \
              (bv.rolling(24).sum() + cv.rolling(24).sum()/2 + 1e-7) * 100
    
    # 计算MACD
    fast_ema = df['close'].ewm(span=5, adjust=False).mean()
    slow_ema = df['close'].ewm(span=13, adjust=False).mean()
    df['diff'] = fast_ema - slow_ema
    df['dea'] = df['diff'].ewm(span=8, adjust=False).mean()
    df['macd'] = 2 * (df['diff'] - df['dea'])
    #计算tsma
    df['tsma5'] = tsma_fast(df['close'], 5)
    df['tsma8'] = tsma_fast(df['close'], 8)
    df['tsma13'] = tsma_fast(df['close'], 13)
        
    
    # 清理中间列
    return df.drop(['LC','CLOSE_LC'], axis=1)


    
def detect_bottom_divergence(df, indicator_col='macd', window_size=30):
    """
    单一指标底背离检测函数
    参数：
    df - 包含以下列的DataFrame:
         - low: 价格低点
         - close: 收盘价
         - [indicator_col]: 指标列（如macd/vr）
    indicator_col - 要检测的指标列名（默认'macd'）
    window_size - 检测窗口大小（默认30）
    
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
    reference_low = None
    reference_indicator = None
    
    for i in range(1, len(df)):
        current_low = df.at[i, 'low']
        current_close = df.at[i, 'close']
        current_indicator = df.at[i, indicator_col]
        
        # 状态转移逻辑
        if current_status == 'normal':
            start_idx = max(0, i - window_size)
            window = df.iloc[start_idx:i]
            
            if not window.empty:
                prev_low_idx = window['low'].idxmin()
                prev_low = window.at[prev_low_idx, 'low']
                prev_indicator = window.at[prev_low_idx, indicator_col]
                
                # 价格创新低但指标未新低 (底背离条件)
                if (current_low < prev_low) and (current_indicator > prev_indicator):
                    current_status = 'tbd'
                    reference_low = prev_low
                    reference_indicator = prev_indicator
        
        elif current_status == 'tbd':
            # 情况1：价格继续创新低
            if current_low < reference_low:
                # 如果指标也创新低，则破坏背离条件
                if current_indicator <= reference_indicator:
                    current_status = 'normal'
            
            # 情况2：价格上涨（确认背离信号）
            if current_close > df.at[i-1, 'close']:
                if current_indicator > reference_indicator:
                    current_status = 'confirmed'
                else:
                    current_status = 'normal'
        
        elif current_status == 'confirmed':
            # 清除条件：价格连续2日下跌 或 指标连续2日下降
            clear_cond = False
            if i >= 2:
                price_down = (current_close < df.at[i-1, 'close']) and \
                            (df.at[i-1, 'close'] < df.at[i-2, 'close'])
                indicator_down = (current_indicator < df.at[i-1, indicator_col]) and \
                               (df.at[i-1, indicator_col] < df.at[i-2, indicator_col])
                clear_cond = price_down or indicator_down
            
            if clear_cond:
                current_status = 'normal'
                reference_low = None
                reference_indicator = None
        
        df.at[i, 'divergence_status'] = current_status
    
    df.index = original_index
    return df

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



def get_dominant_contracts():
    """获取主力合约列表并缓存到本地文件（无os模块版本）"""
    cache_file = "主力合约.csv"
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    
    try:
        # 尝试读取缓存文件
        df = pd.read_csv(cache_file, header=None)
        
        # 检查文件格式：第一行应为日期，后面是合约代码
        if len(df) > 1 and df.iloc[0, 0] == today_str:
            print("读取缓存的主力合约...")
            return df.iloc[1:, 0].tolist()
        else:
            print(f"缓存已过期或格式无效，需要重新获取")
    except:
        print("未找到缓存文件或读取失败，需要重新获取")
    
    # 重新获取主力合约
    print("正在获取合约列表...")
    table = get_all_securities(ty='commodity_futures', date=None)
    all_futures = table.index.tolist()
    
    # 提取唯一品种代码（前两位字母）
    unique_letter_pairs = set(
        code[:2] for code in all_futures 
        if len(code) == 6 and code[:2].isalpha() and code[2:].isdigit()
    )
    letter_pairs = sorted(unique_letter_pairs)
    
    # 创建空列表保存主力合约
    dominant_contracts_list = []
    
    print("开始获取主力合约...")
    # 为每个品种代码获取主力合约并保存到列表
    for symbol in letter_pairs:
        try:
            # 调用主力合约函数并添加到列表
            dominant_contract = get_futures_dominate(symbol, date=None, seq=0)
            dominant_contracts_list.append(dominant_contract)
            print(f"成功获取 {symbol} 的主力合约: {dominant_contract}")
        except Exception as e:
            print(f"获取 {symbol} 的主力合约时出错: {str(e)}")
            # 如果需要保留位置，可以添加None
            # dominant_contracts_list.append(None)
    
    count = len(dominant_contracts_list)
    print(f"共获取到{count}只期货主力合约")
    
    # 保存到CSV文件（第一行存储日期，后面存储合约代码）
   
    write_file(cache_file, today_str + '\n')  # 第一行写入当天日期
    for contract in dominant_contracts_list:
        write_file(cache_file, contract + '\n', append=True)
    
    print(f"主力合约已保存至: {cache_file}")
    
    return dominant_contracts_list


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


class KLineCenterAnalyzer:
    def __init__(self, window=5, shadow_ratio=0.7, min_zone_separation=0.02, 
                 max_zone_width_ratio=0.1, entity_size_threshold=0.05):
        self.window = window
        self.shadow_ratio = shadow_ratio
        self.min_zone_separation = min_zone_separation
        self.max_zone_width_ratio = max_zone_width_ratio
        self.entity_size_threshold = entity_size_threshold
    
    def is_bullish(self, open_price, close_price):
        """判断是否为阳线"""
        return close_price >= open_price
    
    def calculate_shadow_ratio(self, open_price, high_price, low_price, close_price):
        """计算影线比例"""
        body_size = abs(close_price - open_price)
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        
        if body_size == 0:
            return upper_shadow, lower_shadow, 0
        
        return upper_shadow, lower_shadow, max(upper_shadow, lower_shadow) / body_size
    
    def find_reversal_points(self, df):
        """寻找K线反转点"""
        reversal_points = []
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        for i in range(1, len(opens)-1):
            # 检查颜色反转
            prev_bullish = self.is_bullish(opens[i-1], closes[i-1])
            curr_bullish = self.is_bullish(opens[i], closes[i])
            next_bullish = self.is_bullish(opens[i+1], closes[i+1])
            
            # 颜色反转点
            if (prev_bullish != curr_bullish) or (curr_bullish != next_bullish):
                reversal_points.append(('color_reversal', i, closes[i]))
            
            # 检查影线反转
            _, _, ratio_curr = self.calculate_shadow_ratio(
                opens[i], highs[i], lows[i], closes[i])
            
            # 影线比例超过阈值，可能是反转点
            if ratio_curr > self.shadow_ratio:
                upper_shadow, lower_shadow, _ = self.calculate_shadow_ratio(
                    opens[i], highs[i], lows[i], closes[i])
                
                if upper_shadow > lower_shadow:
                    reversal_points.append(('upper_shadow', i, highs[i]))
                else:
                    reversal_points.append(('lower_shadow', i, lows[i]))
        
        return reversal_points
    
    def find_volume_price_points(self, df):
        """寻找成交量极值点对应的K线价格"""
        volumes = df['volume'].values
        closes = df['close'].values
        
        # 寻找成交量局部高点
        volume_high_idx = argrelextrema(volumes, np.greater, order=self.window)[0]
        
        volume_points = []
        for idx in volume_high_idx:
            volume_points.append(('volume_price', idx, closes[idx]))
        
        return volume_points
    
    def cluster_prices(self, points, n_clusters=3):
        """对价格点进行聚类，返回聚类中心"""
        if not points:
            return []
            
        prices = np.array([point[2] for point in points]).reshape(-1, 1)
        
        if len(prices) <= n_clusters:
            return sorted([p[0] for p in prices])
        
        kmeans = KMeans(n_clusters=min(n_clusters, len(prices)), random_state=0).fit(prices)
        centers = sorted([center[0] for center in kmeans.cluster_centers_])
        return centers
    
    def find_centers(self, df):
        """找出中枢水平线"""
        reversal_points = self.find_reversal_points(df)
        volume_points = self.find_volume_price_points(df)
        
        all_points = reversal_points + volume_points
        
        if not all_points:
            return [], [], []
        
        center_lines = self.cluster_prices(all_points, 3)
        
        return center_lines, reversal_points, volume_points
    
    def calculate_center_zones(self, df, center_lines):
        """计算中枢区域，确保不重叠"""
        center_zones = []
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        price_range = np.max(highs) - np.min(lows)
        min_separation = price_range * self.min_zone_separation
        max_zone_width = price_range * self.max_zone_width_ratio
        
        entity_sizes = np.abs(closes - opens)
        entity_size_threshold = np.percentile(entity_sizes, 95)
        
        for center_line in center_lines:
            crossing_k_lines = []
            for i in range(len(df)):
                if lows[i] <= center_line <= highs[i]:
                    entity_high = max(opens[i], closes[i])
                    entity_low = min(opens[i], closes[i])
                    entity_size = entity_high - entity_low
                    
                    if entity_size <= entity_size_threshold:
                        crossing_k_lines.append((entity_high, entity_low))
            
            if crossing_k_lines:
                entity_highs = [h for h, l in crossing_k_lines]
                entity_lows = [l for h, l in crossing_k_lines]
                
                zone_high = np.mean(entity_highs)
                zone_low = np.mean(entity_lows)
                
                min_zone_height = price_range * 0.005
                if zone_high - zone_low < min_zone_height:
                    mid_point = (zone_high + zone_low) / 2
                    zone_high = mid_point + min_zone_height / 2
                    zone_low = mid_point - min_zone_height / 2
                
                if zone_high - zone_low > max_zone_width:
                    mid_point = (zone_high + zone_low) / 2
                    zone_high = mid_point + max_zone_width / 2
                    zone_low = mid_point - max_zone_width / 2
                
                overlap_found = False
                for i, (existing_center, existing_low, existing_high, existing_count) in enumerate(center_zones):
                    if not (zone_high < existing_low or zone_low > existing_high):
                        # 发现重叠，合并区域
                        merged_low = min(zone_low, existing_low)
                        merged_high = max(zone_high, existing_high)
                        
                        if merged_high - merged_low > max_zone_width:
                            mid_point = (merged_low + merged_high) / 2
                            merged_low = mid_point - max_zone_width / 2
                            merged_high = mid_point + max_zone_width / 2
                        
                        merged_center = (center_line + existing_center) / 2
                        merged_count = len(crossing_k_lines) + existing_count
                        
                        center_zones[i] = (merged_center, merged_low, merged_high, merged_count)
                        overlap_found = True
                        break
                
                if not overlap_found:
                    center_zones.append((center_line, zone_low, zone_high, len(crossing_k_lines)))
        
        # 确保区域之间有最小分离距离，防止重叠
        center_zones.sort(key=lambda x: x[0])
        
        non_overlapping_zones = []
        for zone in center_zones:
            center_line, zone_low, zone_high, count = zone
            
            if not non_overlapping_zones:
                non_overlapping_zones.append(zone)
                continue
                
            # 检查是否与已有区域重叠
            overlaps = False
            for existing_zone in non_overlapping_zones:
                existing_center, existing_low, existing_high, existing_count = existing_zone
                
                if not (zone_high < existing_low or zone_low > existing_high):
                    # 有重叠，跳过这个区域
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping_zones.append(zone)
        
        # 按中心线排序
        non_overlapping_zones.sort(key=lambda x: x[0])
        
        return non_overlapping_zones


class MomentumEnergyAnalyzer:
    def __init__(self, df, symbol="未知标的"):
        self.df = df.copy()
        self.symbol = symbol
        self.center_analyzer = KLineCenterAnalyzer()
        self.calculate_all_indicators()
    
    def calculate_all_indicators(self):
        """一次性计算所有指标"""
        # 1. 计算基础技术指标
        self.calculate_basic_indicators()
        
        # 2. 计算中枢
        self.calculate_centers()
        
        # 3. 计算各种势能
        self.calculate_energy_components()
        
        # 4. 生成信号
        self.generate_signals()
    
    def calculate_basic_indicators(self):
        """计算基础技术指标"""
        # 价格变化
        self.df['price_change'] = self.df['close'].pct_change()
        self.df['log_return'] = np.log(self.df['close'] / self.df['close'].shift(1))
        
        # K线特征
        self.df['body_size'] = abs(self.df['close'] - self.df['open']) / self.df['open']
        self.df['total_range'] = (self.df['high'] - self.df['low']) / self.df['open']
        self.df['upper_shadow'] = (self.df['high'] - np.maximum(self.df['open'], self.df['close'])) / self.df['open']
        self.df['lower_shadow'] = (np.minimum(self.df['open'], self.df['close']) - self.df['low']) / self.df['open']
        self.df['price_position'] = (self.df['close'] - self.df['low']) / (self.df['high'] - self.df['low'])
        
        # 移动平均
        self.df['ma5'] = self.df['close'].rolling(5).mean()
        self.df['ma20'] = self.df['close'].rolling(20).mean()
        self.df['volume_ma5'] = self.df['volume'].rolling(5).mean()
    
    def calculate_centers(self):
        """计算中枢区域"""
        # 使用整个数据集计算中枢
        center_lines, reversal_points, volume_points = self.center_analyzer.find_centers(self.df)
        self.center_zones = self.center_analyzer.calculate_center_zones(self.df, center_lines)
        
        '''print(f"检测到 {len(self.center_zones)} 个中枢区域:")
        for i, (center, low, high, count) in enumerate(self.center_zones):
            print(f"  中枢{i+1}: {center:.3f} [{low:.3f} - {high:.3f}], 穿越K线: {count}")'''
    
    def calculate_center_energy(self):
        """计算基于中枢的势能 - 修正极性"""
        center_energy = []
        
        for i in range(len(self.df)):
            price = self.df['close'].iloc[i]
            total_energy = 0
            
            for center, zone_low, zone_high, strength in self.center_zones:
                # 计算强度权重
                zone_strength = np.log1p(strength)
                
                if price > zone_high:
                    # 在上方 - 正势能（上涨动力）- 修正极性
                    distance_ratio = (price - center) / (zone_high - center) if (zone_high - center) > 0 else 0
                    energy = distance_ratio * zone_strength  # 改为正数
                elif price < zone_low:
                    # 在下方 - 负势能（下跌压力）- 修正极性
                    distance_ratio = (center - price) / (center - zone_low) if (center - zone_low) > 0 else 0
                    energy = -distance_ratio * zone_strength  # 改为负数
                else:
                    # 在中枢内 - 根据位置决定
                    position = (price - zone_low) / (zone_high - zone_low) if (zone_high - zone_low) > 0 else 0.5
                    # 中枢上半部：轻微正势能；下半部：轻微负势能
                    energy = (position - 0.5) * 0.5 * zone_strength
                
                total_energy += energy
            
            center_energy.append(total_energy)
        
        return center_energy
    
    def calculate_breakout_energy(self):
        """计算突破势能"""
        breakout_energy = [0] * len(self.df)
        
        for i in range(1, len(self.df)):
            current = self.df.iloc[i]
            prev = self.df.iloc[i-1]
            
            # 突破前高
            if current['close'] > prev['high'] and current['volume'] > prev['volume']:
                breakout_energy[i] = current['log_return'] * current['volume'] * 2
            # 跌破前低
            elif current['close'] < prev['low'] and current['volume'] > prev['volume']:
                breakout_energy[i] = current['log_return'] * current['volume'] * 2
            # 普通情况
            else:
                breakout_energy[i] = current['log_return'] * current['volume']
        
        return breakout_energy
    
    def calculate_energy_components(self):
        """计算各种势能成分"""
        
        # 1. 基础动能势能
        self.df['kinetic_energy'] = self.df['log_return'] * self.df['volume']
        self.df['potential_energy'] = (self.df['price_position'] - 0.5) * self.df['volume'] * 2
        self.df['mechanical_energy'] = self.df['kinetic_energy'] + self.df['potential_energy']
        self.df['cumulative_mechanical'] = self.df['mechanical_energy'].cumsum()
        
        # 2. 压力支撑势能
        def pressure_support_energy(row):
            pressure = row['upper_shadow'] * row['volume']
            support = row['lower_shadow'] * row['volume']
            if row['close'] > row['open']:
                return support - pressure * 0.5
            else:
                return pressure - support * 0.5
        
        self.df['pressure_support_energy'] = self.df.apply(pressure_support_energy, axis=1)
        self.df['cumulative_pressure_support'] = self.df['pressure_support_energy'].cumsum()
        
        # 3. 趋势势能
        self.df['trend_strength'] = (self.df['ma5'] - self.df['ma20']) / self.df['ma20'] * 100
        self.df['trend_energy'] = self.df['trend_strength'] * self.df['volume'] / self.df['volume_ma5']
        self.df['cumulative_trend'] = self.df['trend_energy'].cumsum()
        
        # 4. 中枢势能（新增）
        self.df['center_energy'] = self.calculate_center_energy()
        self.df['cumulative_center'] = self.df['center_energy'].cumsum()
        
        # 5. 突破势能（保留）
        self.df['breakout_energy'] = self.calculate_breakout_energy()
        self.df['cumulative_breakout'] = self.df['breakout_energy'].cumsum()
        
        # 6. 综合势能（加权组合）
        # 标准化各成分
        energy_components = [
            'cumulative_mechanical', 
            'cumulative_pressure_support', 
            'cumulative_trend', 
            'cumulative_center',
            'cumulative_breakout'
        ]
        
        for col in energy_components:
            if self.df[col].std() > 0:
                self.df[f'normalized_{col}'] = (self.df[col] - self.df[col].mean()) / self.df[col].std()
            else:
                self.df[f'normalized_{col}'] = 0
        
        # 权重分配（中枢势能和突破势能占重要地位）
        weights = {
            'mechanical': 0.2,
            'pressure_support': 0.15, 
            'trend': 0.15,
            'center': 0.3,      # 中枢势能权重最大
            'breakout': 0.2     # 突破势能重要地位
        }
        
        self.df['comprehensive_energy'] = (
            weights['mechanical'] * self.df['normalized_cumulative_mechanical'] +
            weights['pressure_support'] * self.df['normalized_cumulative_pressure_support'] +
            weights['trend'] * self.df['normalized_cumulative_trend'] +
            weights['center'] * self.df['normalized_cumulative_center'] +
            weights['breakout'] * self.df['normalized_cumulative_breakout']
        )
        
        # 势能动量
        self.df['energy_momentum'] = self.df['comprehensive_energy'].diff(3)
    
    def generate_signals(self):
        """生成交易信号"""
        # 势能水平
        self.df['energy_level'] = pd.cut(
            self.df['comprehensive_energy'], 
            bins=[-np.inf, -1, -0.5, 0.5, 1, np.inf],
            labels=['极低', '较低', '中性', '较高', '极高']
        )
        
        # 势能趋势
        self.df['energy_trend'] = np.where(
            self.df['energy_momentum'] > 0.1, '加速上升',
            np.where(self.df['energy_momentum'] < -0.1, '加速下降', '平稳')
        )
        
        # 交易信号（修正逻辑：高势能看跌，低势能看涨）
        conditions = [
            (self.df['energy_level'].isin(['极高', '较高'])) & (self.df['energy_trend'] == '加速上升'),
            (self.df['energy_level'].isin(['极低', '较低'])) & (self.df['energy_trend'] == '加速下降'),
            (self.df['energy_level'].isin(['较高'])) & (self.df['energy_trend'] == '平稳'),
            (self.df['energy_level'].isin(['较低'])) & (self.df['energy_trend'] == '平稳')
        ]
        choices = ['强烈看跌', '强烈看涨', '温和看跌', '温和看涨']
        self.df['trading_signal'] = np.select(conditions, choices, default='中性')
    

class ContractScorer:
    """合约评分器 - 用于多合约筛选"""
    
    def __init__(self):
        self.min_data_length = 50  # 最小数据长度要求
        
    def score_contract(self, symbol, hist_data):
        """对单个合约进行评分，返回方向和分数"""
        try:
            if len(hist_data) < self.min_data_length:
                return {'direction': 'neutral', 'score': 0, 'reason': '数据不足'}
                
            # 运行势能分析
            analyzer = MomentumEnergyAnalyzer(hist_data, symbol)
            
            # 获取最近10个周期的信号
            recent_signals = analyzer.df['trading_signal'].tail(10)
            
            # 计算看涨和看跌信号数量
            bull_signals = recent_signals.isin(['强烈看涨', '温和看涨']).sum()
            bear_signals = recent_signals.isin(['强烈看跌', '温和看跌']).sum()
            
            # 计算势能指标
            current_energy = analyzer.df['comprehensive_energy'].iloc[-1]
            energy_momentum = analyzer.df['energy_momentum'].iloc[-1]
            
            # 计算趋势指标
            ma5 = analyzer.df['ma5'].iloc[-1]
            ma20 = analyzer.df['ma20'].iloc[-1]
            trend_strength = (ma5 - ma20) / ma20 * 100
            
            # 综合评分
            long_score = (
                bull_signals * 10 +  # 看涨信号权重
                max(0, current_energy) * 5 +  # 正势能
                max(0, energy_momentum) * 3 +  # 正势能动量
                max(0, trend_strength) * 2     # 正趋势
            )
            
            short_score = (
                bear_signals * 10 +  # 看跌信号权重
                max(0, -current_energy) * 5 +  # 负势能
                max(0, -energy_momentum) * 3 +  # 负势能动量
                max(0, -trend_strength) * 2     # 负趋势
            )
            
            # 确定方向和分数
            if long_score > short_score and long_score > 15:  # 阈值过滤
                return {
                    'direction': 'long', 
                    'score': long_score,
                    'details': {
                        'bull_signals': bull_signals,
                        'current_energy': current_energy,
                        'trend_strength': trend_strength
                    }
                }
            elif short_score > long_score and short_score > 15:  # 阈值过滤
                return {
                    'direction': 'short', 
                    'score': short_score,
                    'details': {
                        'bear_signals': bear_signals,
                        'current_energy': current_energy,
                        'trend_strength': trend_strength
                    }
                }
            else:
                return {'direction': 'neutral', 'score': 0, 'reason': '信号不足'}
                
        except Exception as e:
            log.warn(f"合约 {symbol} 评分失败: {str(e)}")
            return {'direction': 'neutral', 'score': 0, 'reason': f'评分异常: {str(e)}'}


 
def main():
    # 获取全量期货池
    print("正在获取合约列表...")
    
    table = get_all_securities(ty='commodity_futures', date=None)
    #all_futures = get_all_securities(ty='commodity_futures', date=None).index.tolist()
    dominant_contracts_list = get_dominant_contracts()
    print(f"共获取到{len(dominant_contracts_list)}只期货")
    # 初始化结果容器
    results = []
    valid_stock_count = 0
    print(datetime.date.today().strftime('%Y%m%d'))
    # 使用进度条处理
    with tqdm(total=len(dominant_contracts_list), desc="计算进度") as pbar:
        for i in range(0, len(dominant_contracts_list), CONFIG["BATCH_SIZE"]):
            batch = dominant_contracts_list[i:i+CONFIG["BATCH_SIZE"]]
            
            for stock_code in batch:
                try:
                    # 获取历史行情（假设返回DataFrame）
                    price_data = get_price_future(
                        symbol_list=stock_code,  # 注意参数名是复数但支持单个代码
                        end_date=datetime.datetime.now().strftime('%Y%m%d %H:%M'),  # 结束日期设为今天
                        start_date = None,
                        fre_step='15m',           # 日线频率
                        fields=['open','high','low','close','volume'],
                        fq='pre',                # 前复权
                        bar_count=250           # 获取250根K线
                        ).sort_index()  # 清除证券代码索引层级
                    
                    # 检查数据有效性
                    if len(price_data) < CONFIG["MIN_DATA_DAYS"]:
                        pbar.update(1)
                        continue
                    
                    #计算指标
                    hist = calculate_indicators(price_data)                   
  
                    '''                  
                    result_df = detect_single_divergence(hist)
                    result_df1 = detect_bottom_divergence(hist)
                    if result_df['divergence_status'].iloc[-1] != "confirmed" and result_df1['divergence_status'].iloc[-1] != "confirmed":  
                        pbar.update(1)
                        continue  
                    '''
                    scorer = ContractScorer()
                    score_result = scorer.score_contract(stock_code, hist)
                    if score_result['direction'] == 'neutral' or score_result['score'] < 60:
                        pbar.update(1)
                        continue
                                                        
                    
                    # 记录结果
                    results.append({
                        "期货代码": stock_code,
                        "名字": table.loc[stock_code, 'display_name'],
                        "分数": score_result['score']

                    })
                    valid_stock_count += 1
                    
                except Exception as e:
                    print(f"\n{stock_code}处理失败: {str(e)}")
                finally:
                    pbar.update(1)
    
    # 处理结果
    if not results:
        print("没有有效数据可供分析")
        return

    # 创建DataFrame并排序
    df = pd.DataFrame(results).sort_values(by="分数", ascending=True)
    df = pd.DataFrame(results)
    
    
    # 打印摘要
    print("\n" + "="*50)
    print(f"分析完成！有效处理合约数：{valid_stock_count}/{len(dominant_contracts_list)}")
    print(f"选出的合约：\n{df[['期货代码', '名字']].head(30).to_string(index=False)}")
    jupyter_stock_charts(df.head(70))  # 使用Jupyter专用控件
    

import matplotlib.gridspec as gridspec


def jupyter_stock_charts(df_results):
    """Jupyter专用期货合约图表浏览器（带均线和技术指标）"""
    if df_results.empty:
        print("没有期货合约可展示")
        return
    
    # 准备数据
    stock_codes = df_results["期货代码"].tolist()
    stock_names = df_results["名字"].tolist()
    current_index = 0
    
    # 创建控件
    prev_btn = widgets.Button(description="上一只")
    next_btn = widgets.Button(description="下一只")
    index_label = widgets.Label(value=f"期货合约: 1/{len(stock_codes)}")
    
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
            price_data = get_price_future(
                    symbol_list=stock_code,  # 注意参数名是复数但支持单个代码
                    end_date=datetime.datetime.now().strftime('%Y%m%d %H:%M') ,  # 结束日期设为今天
                    start_date = None,
                    fre_step='15m',           # 日线频率
                    fields=['open','high','low','close','volume'],
                    fq='pre',                # 前复权
                    bar_count=100           # 获取250根K线
                    ).sort_index()  # 清除证券代码索引层级
            
            if price_data.empty:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.set_title(f"{stock_name} ({stock_code}) - 数据获取失败", fontsize=14)
                plt.tight_layout()
                plt.show()
                return
            
            # 计算技术指标
            # 计算MACD
            fast_ema = price_data['close'].ewm(span=12, adjust=False).mean()
            slow_ema = price_data['close'].ewm(span=26, adjust=False).mean()
            price_data['diff'] = fast_ema - slow_ema
            price_data['dea'] = price_data['diff'].ewm(span=9, adjust=False).mean()
            price_data['macd'] = 2 * (price_data['diff'] - price_data['dea'])
            
            # 计算TSMA指标（这里使用普通SMA代替，您可以根据需要修改为真实TSMA计算）
            price_data['tsma5'] = tsma_fast(price_data['close'], 5)
            price_data['tsma13'] = tsma_fast(price_data['close'], 13)
            
            # 创建整数索引（去除非交易日间隙）
            price_data['index_num'] = range(len(price_data))
            
            # 创建3个子图（K线图 + 2个指标）
            fig = plt.figure(figsize=(12, 8))
            gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0)
            ax1 = plt.subplot(gs[0])  # K线图
            ax2 = plt.subplot(gs[1], sharex=ax1)  # MACD
            ax3 = plt.subplot(gs[2], sharex=ax1)  # TSMA
            
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
            
            ax1.plot(price_data['index_num'], price_data['MA5'], 'b-', linewidth=1.5, label='5日均线', zorder=2)
            ax1.plot(price_data['index_num'], price_data['MA10'], 'm-', linewidth=1.5, label='10日均线', zorder=2)
            ax1.plot(price_data['index_num'], price_data['MA20'], 'c-', linewidth=1.5, label='20日均线', zorder=2)
            
            # 设置K线图标题和标签
            last_close = price_data['close'].iloc[-1]
            last_date = price_data.index[-1].strftime('%Y-%m-%d %H:%M')
            ax1.set_title(f"{stock_name} ({stock_code}) - 最新价: {last_close:.2f} ({last_date})", 
                         fontsize=16, fontweight='bold')
            ax1.set_ylabel('价格')
            ax1.legend(loc='upper left')
            ax1.grid(True, linestyle='--', alpha=0.6)
            
            # 绘制MACD指标
            colors = ['red' if val >= 0 else 'green' for val in price_data['macd']]
            ax2.bar(price_data['index_num'], price_data['macd'], color=colors, width=0.8)
            ax2.plot(price_data['index_num'], price_data['diff'], 'b-', linewidth=1.2, label='DIFF')
            ax2.plot(price_data['index_num'], price_data['dea'], 'm-', linewidth=1.2, label='DEA')
            ax2.axhline(0, color='gray', linestyle='-', linewidth=0.7)
            ax2.set_ylabel('MACD')
            ax2.legend(loc='upper left')
            ax2.grid(True, linestyle='--', alpha=0.4)
            
            # 绘制TSMA指标
            ax3.plot(price_data['index_num'], price_data['tsma5'], 'b-', linewidth=1.5, label='5日TSMA')
            ax3.plot(price_data['index_num'], price_data['tsma13'], 'r-', linewidth=1.5, label='13日TSMA')
            ax3.set_ylabel('TSMA')
            ax3.set_xlabel('交易日序列')
            ax3.legend(loc='upper left')
            ax3.grid(True, linestyle='--', alpha=0.4)
            
            # 设置x轴刻度（只显示在最后一个子图）
            n = len(price_data)
            step = max(1, n // 10)
            xticks = list(range(0, n, step))
            if n-1 not in xticks:
                xticks.append(n-1)
            xticklabels = [price_data.index[i].strftime('%m-%d %H:%M') for i in xticks]
            ax3.set_xticks(xticks)
            ax3.set_xticklabels(xticklabels, rotation=45)
            
            # 设置y轴范围
            y_min = price_data[['low', 'MA5', 'MA10', 'MA20']].min().min()
            y_max = price_data[['high', 'MA5', 'MA10', 'MA20']].max().max()
            ax1.set_ylim(y_min * 0.999, y_max * 1.001)
            
            # 隐藏上方子图的x轴标签
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.setp(ax2.get_xticklabels(), visible=False)
            
            plt.tight_layout()
            plt.show()
    
    # 定义按钮回调函数
    def on_next_btn(b):
        nonlocal current_index
        current_index = (current_index + 1) % len(stock_codes)
        index_label.value = f"期货合约: {current_index+1}/{len(stock_codes)}"
        update_chart()
    
    def on_prev_btn(b):
        nonlocal current_index
        current_index = (current_index - 1) % len(stock_codes)
        current_index = current_index if current_index >= 0 else len(stock_codes) - 1
        index_label.value = f"期货合约: {current_index+1}/{len(stock_codes)}"
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






class FutureFiltetV2Strategy(BaseStrategy):
    """基于future_filtet_v2的策略"""

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
