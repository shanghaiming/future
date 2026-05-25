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
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False   # 用来正常显示负号
# 配置参数
CONFIG = {
    "PERIODS": [5, 8, 13, 21, 34, 55],  # 需要计算的周期
    "MIN_DATA_DAYS": 55,                # 需要的最少数据天数
    "BATCH_SIZE": 100,                  # 分批处理数量
    "OUTPUT_FILE": "output/selected_stocks"  # 输出文件名
}


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
            'mechanical': 0.3,
            'pressure_support': 0.25, 
            'trend': 0.25,
            'center': 0.0,      # 中枢势能权重最大
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
    

def filter_stocks(
    all_stocks, 
    table, 
    CONFIG, 
    history_func, 
    calculate_williams_r_func,
    MomentumEnergyAnalyzer,
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
        if stock_code.startswith("8") or stock_code.startswith("9"):
            return None
        
        try:
            # 获取历史数据
            time = datetime.date.today()
            price_data = history_func(
                        stock=stock_code,  # 注意参数名是复数但支持单个代码
                        end_date=time.strftime('%Y%m%d'),  # 结束日期设为今天
                        start_date= (time - datetime.timedelta(days=200)).strftime('%Y%m%d')
                        )
            
            
            # 基础检查
            if len(price_data) < CONFIG["MIN_DATA_DAYS"]:
                print(f"数据不够{stock_code}")
                return None
            hist = price_data
            hist['wr5'] = calculate_williams_r_func(hist, period=5)
            df = hist
            # 运行分析
            analyzer = MomentumEnergyAnalyzer(df)
            #analyzer.calculate_all_indicators()
            signal = analyzer.df['trading_signal']
            
            
            if signal[-1] != '强烈看涨':
                return None
            count = signal.tail(10).isin(['强烈看涨', '温和看涨']).sum()

            
            
            # 通过所有筛选条件
            return {
                "股票代码": stock_code,
                "名字": table.loc[stock_code, 'name'],
                "成本": price_data['close'].iloc[-1],
                "wr5": hist['wr5'].iloc[-1],
                "count": count
            }
            
        except Exception as e:
            # 实际使用时可记录日志
            print(f"{stock_code}处理失败: {str(e)}")
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
                if stock_code.startswith("8"):
                    continue
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
                print(f"处理{stock_code}时出错: {str(e)}")
                pass
    
    return results, valid_stock_count


    

def get_stock_data(stock, start_date, end_date):
    """获取单只股票在指定日期范围内的数据"""
    
    file_path = fr"E:\stock\csv_version\analysis_results\{stock}_analysis.csv"
    df = pd.read_csv(file_path, index_col='trade_date', parse_dates=True)
    filtered_df = df.loc[start_date:end_date]
    
    return filtered_df

def main():
    # 获取全量股票池
    print("正在获取股票列表...")
    file_path = fr"E:\stock\csv_version\stocks_list.csv"
    df = pd.read_csv(file_path, index_col='ts_code')
    #df = df[df['list_date'] < 20181212].sort_values('list_date', ascending=False)
    table = df
    all_stocks = df.index.to_list()
    #all_stocks = ['603058.SH','688605.SH']
    print(f"共获取到{len(all_stocks)}只A股")
    # 初始化结果容器
    results = []
    valid_stock_count = 0
    
    results, valid_count = filter_stocks(
    all_stocks=all_stocks,
    table=table,
    CONFIG=CONFIG,
    history_func=get_stock_data,
    calculate_williams_r_func=calculate_williams_r,
    MomentumEnergyAnalyzer = MomentumEnergyAnalyzer,
    max_workers=8  # 根据CPU核心数调整
    )
    
    # 处理结果
    if not results:
        print("没有有效数据可供分析")
        return

    # 创建DataFrame并排序
    df = pd.DataFrame(results).sort_values(by="count", ascending=False)
    
    # 保存结果
    #df.to_csv(CONFIG["OUTPUT_FILE"], index=False, encoding='utf_8_sig')
    
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
    """Jupyter专用股票图表浏览器（带线性回归通道和技术指标）"""
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
    time = datetime.date.today()
    
    # 定义更新图表函数
    def update_chart():
        with output:
            output.clear_output(wait=True)
            
            stock_code = stock_codes[current_index]
            stock_name = stock_names[current_index]
            
            # 获取股票日线数据
            price_data = get_stock_data(
                        stock=stock_code,
                        end_date=time.strftime('%Y%m%d'),
                        start_date= (time - datetime.timedelta(days=400)).strftime('%Y%m%d')
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
            
            # 计算MACD
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
            
            # 计算威廉指标WR5和WR55
            price_data['wr5'] = calculate_williams_r(price_data, period=5)
            price_data['wr55'] = calculate_williams_r(price_data, period=55)
            
            # 创建整数索引（去除非交易日间隙）
            price_data['index_num'] = range(len(price_data))
            
            # 计算线性回归通道
            close_prices = price_data['close'].values
            high_prices = price_data['high'].values
            low_prices = price_data['low'].values
            
            try:
                analyzer = AdaptiveRegressionChannel()
                regression_results = analyzer.get_adaptive_regression(close_prices, high_prices, low_prices)
                regression = regression_results['regression']
                start_info = regression_results['start_info']
                all_high_points = regression_results['all_high_points']
                all_low_points = regression_results['all_low_points']
                
                # 获取回归通道数据
                start_idx = regression['start_index']
                end_idx = regression['end_index']
                regression_line = regression['regression_line']
                upper_band = regression['upper_band']
                lower_band = regression['lower_band']
                
                has_regression = True
            except Exception as e:
                print(f"计算回归通道时出错: {e}")
                has_regression = False
            
            # 创建4个子图（K线图 + 2个指标）
            fig = plt.figure(figsize=(14, 8))
            gs = gridspec.GridSpec(4, 1, height_ratios=[5, 2, 2, 2], hspace=0.1)
            ax1 = plt.subplot(gs[0])  # K线图
            ax2 = plt.subplot(gs[1], sharex=ax1)  # MACD和VR合并
            ax3 = plt.subplot(gs[2], sharex=ax1)  # VAR (方差)
            ax4 = plt.subplot(gs[3], sharex=ax1)  # WR (威廉指标)
            
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
            
            # 绘制成交量（在K线图上，使用次坐标轴）
            ax1_vol = ax1.twinx()  # 创建成交量次坐标轴
            # 计算成交量颜色（与K线颜色一致）
            vol_colors = ['red' if close >= open else 'green' 
                         for close, open in zip(price_data['close'], price_data['open'])]
            # 绘制成交量柱状图，设置透明度
            ax1_vol.bar(price_data['index_num'], price_data['volume'], 
                       color=vol_colors, alpha=0.3, width=0.8, zorder=1)
            ax1_vol.set_ylabel('成交量', fontsize=10)
            # 设置成交量坐标轴在右侧
            ax1_vol.yaxis.set_label_position("right")
            ax1_vol.yaxis.tick_right()
            # 计算并绘制均线
            price_data['MA5'] = price_data['close'].rolling(window=5).mean()
            price_data['MA10'] = price_data['close'].rolling(window=10).mean()
            price_data['MA20'] = price_data['close'].rolling(window=20).mean()
            
            ax1.plot(price_data['index_num'], price_data['MA5'], 'b-', linewidth=1.2, label='5日均线', zorder=2)
            ax1.plot(price_data['index_num'], price_data['MA10'], 'm-', linewidth=1.2, label='10日均线', zorder=2)
            ax1.plot(price_data['index_num'], price_data['MA20'], 'c-', linewidth=1.2, label='20日均线', zorder=2)
            
            # 绘制线性回归通道（如果计算成功）
            if has_regression:
                # 创建回归通道的x坐标
                x_reg = np.arange(start_idx, end_idx + 1)
                
                # 绘制回归线和通道
                ax1.plot(x_reg, regression_line, color='#3498db', linewidth=2.5, label='回归线', zorder=4)
                ax1.plot(x_reg, upper_band, color='#e74c3c', linestyle='--', linewidth=2, label='上轨', zorder=4)
                ax1.plot(x_reg, lower_band, color='#2ecc71', linestyle='--', linewidth=2, label='下轨', zorder=4)
                
                # 填充通道区域
                ax1.fill_between(x_reg, upper_band, lower_band, alpha=0.1, color='#95a5a6', label='回归通道', zorder=3)
                
                # 标记极值点
                high_indices = [i for i, _ in all_high_points]
                high_values = [v for _, v in all_high_points]
                ax1.scatter(high_indices, high_values, color='#e74c3c', marker='v', s=60, 
                           label=f'高点({len(all_high_points)}个)', zorder=5, alpha=0.7)
                
                low_indices = [i for i, _ in all_low_points]
                low_values = [v for _, v in all_low_points]
                ax1.scatter(low_indices, low_values, color='#2ecc71', marker='^', s=60, 
                           label=f'低点({len(all_low_points)}个)', zorder=5, alpha=0.7)
                
                # 标记回归起点
                ax1.scatter([start_info['start_index']], [start_info['start_price']], 
                           color='#3498db', marker='*', s=200, label='回归起点', zorder=6)
                
                # 添加垂直线标记回归起点位置
                ax1.axvline(x=start_info['start_index'], color='#9b59b6', linestyle=':', alpha=0.8, 
                           linewidth=2, label='起点位置', zorder=4)
                
                # 在价格图左上角添加回归信息文本框
                current_gap = abs(close_prices[-1] - lower_band[-1])
                relative_gap = current_gap / close_prices[-1]
                
                stats_text = (
                    f"斜率: {regression['slope']:.6f}\n"
                    f"R²: {regression['r_squared']:.4f}\n"
                    f"质量: {start_info['quality_score']:.4f}\n"
                    f"下轨: {lower_band[-1]:.2f}\n"
                    f"距下轨: {current_gap:.2f}({relative_gap:.2%})"
                )
                ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                                                          facecolor="lightblue", 
                                                          alpha=0.8))
            
            # 设置K线图标题和标签
            last_close = price_data['close'].iloc[-1]
            last_date = price_data.index[-1].strftime('%Y-%m-%d')
            
            title = f"{stock_name} ({stock_code}) - 最新价: {last_close:.2f} ({last_date})"
            ax1.set_title(title, fontsize=16, fontweight='bold')
            ax1.set_ylabel('价格')
            ax1.legend(loc='upper left', fontsize=8)
            ax1.grid(True, linestyle='--', alpha=0.6)
            
            # 绘制MACD和VR指标（合并到一个图，双Y轴）
            # 创建第二个Y轴
            ax2_vr = ax2.twinx()
            
            # 绘制MACD（主Y轴）
            colors_macd = ['red' if val >= 0 else 'green' for val in price_data['macd']]
            ax2.bar(price_data['index_num'], price_data['macd'], color=colors_macd, width=0.8, alpha=0.7, label='MACD')
            ax2.plot(price_data['index_num'], price_data['diff'], 'blue', linewidth=1.5, label='DIFF')
            ax2.plot(price_data['index_num'], price_data['dea'], 'orange', linewidth=1.5, label='DEA')
            ax2.axhline(0, color='gray', linestyle='-', linewidth=0.7)
            ax2.set_ylabel('MACD', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            
            # 绘制VR（次Y轴）
            ax2_vr.plot(price_data['index_num'], price_data['vr'], 'purple', linewidth=1.5, label='VR')
            ax2_vr.axhline(100, color='gray', linestyle='--', linewidth=0.7)
            ax2_vr.set_ylabel('VR', color='purple')
            ax2_vr.tick_params(axis='y', labelcolor='purple')
            
            # 添加图例
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_vr.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
            
            ax2.grid(True, linestyle='--', alpha=0.4)
            ax2.set_title('MACD 和 VR 指标', fontsize=12)
            
            # 绘制VAR（方差）指标
            colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
            for i, period in enumerate(periods):
                ax3.plot(price_data['index_num'], price_data[f'var_{period}'], 
                         color=colors[i % len(colors)], 
                         linewidth=1.2, 
                         label=f'{period}日方差')
            
            ax3.set_ylabel('VAR')
            ax3.legend(loc='upper left', fontsize=8)
            ax3.grid(True, linestyle='--', alpha=0.4)
            ax3.set_title('方差指标', fontsize=12)
            
            # 绘制威廉指标WR（WR5和WR55在同一图上）
            ax4.plot(price_data['index_num'], price_data['wr5'], 'blue', linewidth=1.5, label='WR5')
            ax4.plot(price_data['index_num'], price_data['wr55'], 'red', linewidth=1.5, label='WR55')
            # 添加威廉指标参考线
            ax4.axhline(-20, color='gray', linestyle='--', linewidth=0.7, alpha=0.7)
            ax4.axhline(-80, color='gray', linestyle='--', linewidth=0.7, alpha=0.7)
            ax4.set_ylabel('WR')
            ax4.set_xlabel('交易日序列')
            ax4.legend(loc='upper left', fontsize=8)
            ax4.grid(True, linestyle='--', alpha=0.4)
            ax4.set_title('威廉指标', fontsize=12)
            # 设置WR y轴范围（威廉指标通常在0到-100之间）
            ax4.set_ylim(-100, 0)
            
            # 设置x轴刻度（只显示在最后一个子图）
            n = len(price_data)
            step = max(1, n // 10)
            xticks = list(range(0, n, step))
            if n-1 not in xticks:
                xticks.append(n-1)
            xticklabels = [price_data.index[i].strftime('%m-%d') for i in xticks]
            ax4.set_xticks(xticks)
            ax4.set_xticklabels(xticklabels, rotation=45)
            
            # 设置K线图y轴范围
            y_min = price_data[['low', 'MA5', 'MA10', 'MA20']].min().min()
            y_max = price_data[['high', 'MA5', 'MA10', 'MA20']].max().max()
            
            # 如果有回归通道，调整y轴范围以包含回归通道
            if has_regression:
                y_min = min(y_min, np.min(lower_band))
                y_max = max(y_max, np.max(upper_band))
            
            ax1.set_ylim(y_min * 0.98, y_max * 1.02)
            
            # 隐藏上方子图的x轴标签
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.setp(ax2.get_xticklabels(), visible=False)
            plt.setp(ax3.get_xticklabels(), visible=False)
            
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






class StategyMomentumStrategy(BaseStrategy):
    """基于stategy_momentum的策略"""

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
