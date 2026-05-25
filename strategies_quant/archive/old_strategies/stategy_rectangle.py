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
from numpy.lib.stride_tricks import sliding_window_view
import datetime
try:
    import ipywidgets as widgets
    _HAS_IPYWIDGETS = True
except ImportError:
    _HAS_IPYWIDGETS = False
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from scipy.signal import argrelextrema

from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from sklearn.cluster import DBSCAN
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
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



class SpatiotemporalBoxDetector:
    """
    时空矩形箱体检测器 - 同时考虑空间位置和时间连续性
    """
    
    def __init__(self, data, lookback_period=100):
        """
        初始化
        
        参数:
        data: K线数据 (DataFrame，包含'open', 'high', 'low', 'close'列)
        lookback_period: 回看周期，只分析最近这段时间的数据
        """
        self.data = data.copy()
        self.lookback_period = lookback_period
        self.boxes = []
        
        # 确保数据按时间排序
        self.data = self.data.sort_index()
        
        # 只保留最近的数据
        if len(self.data) > lookback_period:
            self.recent_data = self.data.iloc[-lookback_period:]
        else:
            self.recent_data = self.data
    
    def find_pivot_points(self, window=5):
        """
        寻找关键高低点(pivot points)
        """
        data = self.recent_data
        
        # 寻找局部高点
        high_indices = argrelextrema(data['high'].values, np.greater, order=window)[0]
        high_pivots = [(data.index[i], data['high'].iloc[i]) for i in high_indices]
        
        # 寻找局部低点
        low_indices = argrelextrema(data['low'].values, np.less, order=window)[0]
        low_pivots = [(data.index[i], data['low'].iloc[i]) for i in low_indices]
        
        return high_pivots, low_pivots
    
    def cluster_prices_with_dbscan(self, price_points, eps_percent=0.02, min_samples=2):
        """
        使用DBSCAN聚类算法识别价格水平
        """
        if len(price_points) < min_samples:
            return []
        
        # 提取价格
        prices = np.array([p[1] for p in price_points]).reshape(-1, 1)
        
        # 计算eps（基于价格百分比）
        avg_price = np.mean(prices)
        eps = avg_price * eps_percent
        
        # 使用DBSCAN聚类
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(prices)
        
        # 找出每个簇的中心点
        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:  # 噪声点（突出的高低点）
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(price_points[i])
        
        # 计算每个簇的平均价格和强度
        cluster_levels = []
        for label, points in clusters.items():
            cluster_prices = [p[1] for p in points]
            cluster_mean = np.mean(cluster_prices)
            cluster_std = np.std(cluster_prices)
            
            # 计算簇的强度（点数）
            strength = len(points)
            
            cluster_levels.append({
                'price': cluster_mean,
                'points': points,
                'strength': strength,
                'std': cluster_std
            })
        
        # 按强度排序
        cluster_levels.sort(key=lambda x: x['strength'], reverse=True)
        return cluster_levels
    
    def find_continuous_box(self, min_cluster_strength=2, max_height_percent=0.15, include_last_n=5):
        """
        寻找时间上连续的矩形箱体
        """
        self.boxes = []
        data = self.recent_data
        
        # 寻找高低点
        high_pivots, low_pivots = self.find_pivot_points(window=5)
        
        # 聚类高低点
        resistance_levels = self.cluster_prices_with_dbscan(high_pivots)
        support_levels = self.cluster_prices_with_dbscan(low_pivots)
        
        # 过滤强度不足的簇
        resistance_levels = [r for r in resistance_levels if r['strength'] >= min_cluster_strength]
        support_levels = [s for s in support_levels if s['strength'] >= min_cluster_strength]
        
        # 如果没有足够的簇，直接返回
        if not resistance_levels or not support_levels:
            return self.boxes
        
        # 寻找可能的箱体组合
        for resistance in resistance_levels[:3]:  # 取前3个最强的阻力位
            for support in support_levels[:3]:    # 取前3个最强的支撑位
                # 计算箱体高度
                height = resistance['price'] - support['price']
                avg_price = (resistance['price'] + support['price']) / 2
                height_percent = height / avg_price
                
                # 检查高度是否合理
                if height_percent > max_height_percent:
                    continue
                
                # 确定箱体的时间范围
                resistance_dates = [p[0] for p in resistance['points']]
                support_dates = [p[0] for p in support['points']]
                all_dates = resistance_dates + support_dates
                
                start_date = min(all_dates)
                end_date = max(all_dates)
                
                # 确保包含最新数据
                if end_date < data.index[-1]:
                    end_date = data.index[-1]
                
                # 检查最后几根K线是否在箱体内
                last_n_data = data.iloc[-include_last_n:]
                all_in_box = all(
                    (support['price'] <= last_n_data['low'].iloc[i]) and 
                    (last_n_data['high'].iloc[i] <= resistance['price'])
                    for i in range(len(last_n_data))
                )
                
                if not all_in_box:
                    continue
                
                # 检查时间连续性 - 是否有明显的中断
                if not self._check_time_continuity(data, start_date, end_date, resistance['price'], support['price']):
                    continue
                
                # 检查是否有突出的高低点突破箱体
                if self._has_breakouts(data, start_date, end_date, resistance['price'], support['price']):
                    continue
                
                # 计算质量评分
                quality = self._calculate_box_quality(resistance, support, height_percent)
                
                self.boxes.append({
                    'start': start_date,
                    'end': end_date,
                    'resistance': resistance['price'],
                    'support': support['price'],
                    'height': height,
                    'height_percent': height_percent,
                    'quality': quality,
                    'resistance_strength': resistance['strength'],
                    'support_strength': support['strength'],
                    'resistance_std': resistance['std'],
                    'support_std': support['std']
                })
        
        # 按质量排序
        self.boxes.sort(key=lambda x: x['quality'], reverse=True)
        
        # 只保留最佳结果
        self.boxes = self.boxes[:1]
        
        return self.boxes
    
    def _check_time_continuity(self, data, start_date, end_date, resistance, support):
        """
        检查时间连续性 - 确保箱体在时间上是连续的
        """
        # 获取箱体时间范围内的数据
        box_data = data.loc[start_date:end_date]
        
        # 计算在箱体内的K线比例
        in_box_count = 0
        for i in range(len(box_data)):
            high = box_data['high'].iloc[i]
            low = box_data['low'].iloc[i]
            
            # 检查K线是否在箱体内
            if support <= low and high <= resistance:
                in_box_count += 1
        
        # 计算在箱体内的比例
        in_box_ratio = in_box_count / len(box_data)
        
        # 如果大部分K线都在箱体内，说明连续性良好
        return in_box_ratio >= 0.7  # 70%的K线在箱体内
    
    def _has_breakouts(self, data, start_date, end_date, resistance, support):
        """
        检查是否有突出的高低点突破箱体
        """
        # 获取箱体时间范围内的数据
        box_data = data.loc[start_date:end_date]
        
        # 检查是否有明显的高点突破
        max_high = box_data['high'].max()
        if max_high > resistance * 1.02:  # 超过阻力线2%
            return True
        
        # 检查是否有明显的低点突破
        min_low = box_data['low'].min()
        if min_low < support * 0.98:  # 低于支撑线2%
            return True
        
        # 检查是否有连续多个K线突破箱体
        breakout_count = 0
        for i in range(len(box_data)):
            high = box_data['high'].iloc[i]
            low = box_data['low'].iloc[i]
            
            # 检查是否有突破
            if high > resistance * 1.01 or low < support * 0.99:
                breakout_count += 1
            else:
                breakout_count = 0
            
            # 如果有连续3个K线突破，认为是明显中断
            if breakout_count >= 3:
                return True
        
        return False
    
    def _calculate_box_quality(self, resistance, support, height_percent):
        """
        计算箱体质量评分
        """
        # 簇强度越高，质量越高
        resistance_strength_score = min(resistance['strength'] / 5, 1.0)
        support_strength_score = min(support['strength'] / 5, 1.0)
        
        # 簇标准差越小，质量越高（价格更集中）
        resistance_std_score = 1.0 - min(resistance['std'] / (resistance['price'] * 0.01), 1.0)
        support_std_score = 1.0 - min(support['std'] / (support['price'] * 0.01), 1.0)
        
        # 高度百分比适中（1%-15%）
        height_score = 1.0 - abs(height_percent - 0.08) / 0.08
        
        # 综合评分
        quality = (
            resistance_strength_score * 0.25 +
            support_strength_score * 0.25 +
            resistance_std_score * 0.15 +
            support_std_score * 0.15 +
            height_score * 0.2
        )
        
        return min(max(quality, 0), 1.0)
    
    def get_latest_box(self):
        """
        获取最新的箱体
        """
        if not self.boxes:
            return None
        return self.boxes[0]
    
    def plot_with_boxes(self, title="K线图与时空矩形箱体识别"):
        """
        绘制K线图并标记矩形箱体区域
        """
        if not self.boxes:
            print("未检测到符合条件的时空连续矩形箱体")
            return
        
        # 创建图表
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        
        # 准备数据
        data = self.recent_data
        dates = data.index
        opens = data['open']
        highs = data['high']
        lows = data['low']
        closes = data['close']
        
        # 计算涨跌颜色
        colors = ['red' if close >= open else 'green' 
                 for open, close in zip(opens, closes)]
        
        # 绘制K线
        for i, (date, open_price, high, low, close, color) in enumerate(
                zip(dates, opens, highs, lows, closes, colors)):
            
            # 绘制影线
            ax.plot([date, date], [low, high], color='black', linewidth=1)
            
            # 绘制K线实体
            body_height = abs(close - open_price)
            if body_height > 0:
                ax.bar(date, body_height, bottom=min(open_price, close), 
                       color=color, width=0.8, alpha=0.7)
        
        # 绘制矩形箱体
        for i, box in enumerate(self.boxes):
            start_date = box['start']
            end_date = box['end']
            resistance = box['resistance']
            support = box['support']
            
            # 绘制矩形
            rect = plt.Rectangle(
                (mdates.date2num(start_date), support),
                mdates.date2num(end_date) - mdates.date2num(start_date),
                box['height'],
                linewidth=2, edgecolor='purple', facecolor='yellow', alpha=0.2
            )
            ax.add_patch(rect)
            
            # 绘制支撑线和阻力线
            ax.hlines(resistance, start_date, end_date, colors='red', 
                      linestyles='-', linewidth=2, label='阻力线')
            ax.hlines(support, start_date, end_date, colors='blue', 
                      linestyles='-', linewidth=2, label='支撑线')
            
            # 标注信息
            mid_date = start_date + (end_date - start_date) / 2
            ax.text(mid_date, resistance, 
                    f'阻力: {resistance:.2f}\n强度: {box["resistance_strength"]}', 
                    ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
            
            ax.text(mid_date, support, 
                    f'支撑: {support:.2f}\n强度: {box["support_strength"]}', 
                    ha='center', va='top', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
            
            # 标记最后几根K线
            last_n = 5
            if len(data) > last_n:
                last_dates = data.index[-last_n:]
                for date in last_dates:
                    ax.axvline(x=date, color='orange', linestyle=':', alpha=0.7, linewidth=1)
        
        # 设置图表属性
        ax.set_title(f'{title} (最近{self.lookback_period}期数据)', fontsize=16)
        ax.set_ylabel('价格', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 自动调整x轴日期显示
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.show()
    
    def print_detection_results(self):
        """
        打印检测结果
        """
        if not self.boxes:
            print("未检测到符合条件的时空连续矩形箱体")
            return
        
        print(f"在最近{self.lookback_period}期数据中检测到时空连续矩形箱体:")
        print("=" * 60)
        
        for i, box in enumerate(self.boxes):
            print(f"时空连续矩形箱体 (质量评分: {box['quality']:.2f}):")
            print(f"  时间范围: {box['start']} 到 {box['end']}")
            print(f"  支撑位: {box['support']:.4f} (强度: {box['support_strength']}, 标准差: {box['support_std']:.4f})")
            print(f"  阻力位: {box['resistance']:.4f} (强度: {box['resistance_strength']}, 标准差: {box['resistance_std']:.4f})")
            print(f"  箱体高度: {box['height']:.4f} ({box['height_percent']*100:.2f}%)")
            print()



def filter_stocks(
    all_stocks, 
    table, 
    CONFIG, 
    history_func, 
    calculate_williams_r_func,
    Analyzer,
    max_workers=8
):
    """
    批量筛选符合条件的股票 (线程安全版本)
    返回价格与线性回归带下轨差距绝对值最小的100只股票，且线性回归斜率为正
    
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
                        stock=stock_code,
                        end_date=time.strftime('%Y%m%d'),
                        start_date= (time - datetime.timedelta(days=200)).strftime('%Y%m%d')
                        )
            
            # 基础检查
            if len(price_data) < CONFIG["MIN_DATA_DAYS"]:
                print(f"数据不够{stock_code}")
                return None
            df = price_data
            # 创建检测器实例
            detector = Analyzer(df, lookback_period=100)
            
            # 检测时空连续矩形箱体
            boxes = detector.find_continuous_box(
                min_cluster_strength=2,    # 最小簇强度
                max_height_percent=0.15,   # 最大箱体高度百分比
                include_last_n=5           # 必须包含最后5根K线
            )

            if not boxes:
                return None
            
            df['wr5'] = calculate_williams_r_func(df, period=5)
            wr5_value = df['wr5'].iloc[-1]
        
        
            return {
                "股票代码": stock_code,
                "名字": table.loc[stock_code, 'name'],
                "wr5":wr5_value
            }
            
        except Exception as e:
            # 实际使用时可记录日志
            print(f"{stock_code}处理失败: {str(e)}")
            return None
    
    # 主筛选逻辑
    results = []
    valid_stock_count = 0
    batch_size = CONFIG["BATCH_SIZE"]
    
    # 使用线程池处理
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
                    
                    # 实时显示进度
                    if valid_stock_count % 10 == 0:
                        print(f"已处理 {valid_stock_count} 只有效股票...")
                        
            except Exception as e:
                print(f"处理{stock_code}时出错: {str(e)}")
                pass
    
    # 筛选和排序
    if results:
        # 创建DataFrame
        df_results = pd.DataFrame(results)
        
        # 按相对差距排序（差距最小的在前）
        df_sorted = df_results.sort_values(by="wr5", ascending=True)
        
        # 取前100只
        top_100 = df_sorted.head(100)
        
        print(f"\n筛选完成！共找到 {len(top_100)} 只符合条件的股票")

        
        return top_100.to_dict('records'), valid_stock_count
    else:
        print("没有找到符合条件的股票")
        return [], 0
    

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
    valid_count = 0
    
    results, valid_count = filter_stocks(
    all_stocks=all_stocks,
    table=table,
    CONFIG=CONFIG,
    history_func=get_stock_data,
    calculate_williams_r_func=calculate_williams_r,
    Analyzer = SpatiotemporalBoxDetector,
    max_workers=8  # 根据CPU核心数调整
    )
    
    # 处理结果
    if not results:
        print("没有有效数据可供分析")
        return

    # 创建DataFrame并排序
    df = pd.DataFrame(results).sort_values(by="wr5", ascending=True)
    
    # 保存结果
    #df.to_csv(CONFIG["OUTPUT_FILE"], index=False, encoding='utf_8_sig')
    
    # 打印摘要
    print("\n" + "="*50)
    print(f"分析完成！有效处理股票数：{valid_count}/{len(all_stocks)}")
    
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
    """Jupyter专用股票图表浏览器（带线性回归通道、技术指标和矩形箱体检测）"""
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
            
            # 检测矩形箱体
            try:
                box_detector = SpatiotemporalBoxDetector(price_data, lookback_period=100)
                boxes = box_detector.find_continuous_box(
                    min_cluster_strength=2,
                    max_height_percent=0.15,
                    include_last_n=5
                )
                has_box = len(boxes) > 0
            except Exception as e:
                print(f"检测矩形箱体时出错: {e}")
                has_box = False
                boxes = []
            
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
            
            # 绘制矩形箱体（如果检测到）
            if has_box:
                for i, box in enumerate(boxes):
                    start_date = box['start']
                    end_date = box['end']
                    resistance = box['resistance']
                    support = box['support']
                    
                    # 找到对应的索引位置
                    start_idx = price_data.index.get_loc(start_date)
                    end_idx = price_data.index.get_loc(end_date)
                    
                    # 绘制矩形
                    rect = plt.Rectangle(
                        (start_idx, support),
                        end_idx - start_idx,
                        box['height'],
                        linewidth=2, edgecolor='purple', facecolor='yellow', alpha=0.2, zorder=4
                    )
                    ax1.add_patch(rect)
                    
                    # 绘制支撑线和阻力线
                    ax1.hlines(resistance, start_idx, end_idx, colors='red', 
                              linestyles='-', linewidth=2, label='箱体阻力', zorder=4)
                    ax1.hlines(support, start_idx, end_idx, colors='blue', 
                              linestyles='-', linewidth=2, label='箱体支撑', zorder=4)
                    
                    # 标注信息
                    mid_idx = start_idx + (end_idx - start_idx) / 2
                    ax1.text(mid_idx, resistance, 
                            f'阻力: {resistance:.2f}\n质量: {box["quality"]:.2f}', 
                            ha='center', va='bottom', fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7), zorder=5)
                    
                    ax1.text(mid_idx, support, 
                            f'支撑: {support:.2f}\n强度: {box["support_strength"]}', 
                            ha='center', va='top', fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7), zorder=5)
            
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
            if has_box:
                title += " - 检测到矩形箱体"
            ax1.set_title(title, fontsize=16, fontweight='bold')
            ax1.set_ylabel('价格')
            
            # 合并图例（避免重复）
            handles, labels = ax1.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax1.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=8)
            
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








class StategyRectangleStrategy(BaseStrategy):
    """基于stategy_rectangle的策略"""

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
