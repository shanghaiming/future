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
                    
                    # 筛选掉tsma向下的期货
                    if (hist['tsma5'].iloc[-1] < hist['tsma8'].iloc[-1]) and (hist['tsma5'].iloc[-2] < hist['tsma8'].iloc[-2]):
                        pbar.update(1)
                        continue
                    if (hist['tsma5'].iloc[-1] > hist['tsma8'].iloc[-1]) and (hist['tsma5'].iloc[-2] > hist['tsma8'].iloc[-2]):
                        pbar.update(1)
                        continue
                    
                    

                    result_df = detect_single_divergence(hist)
                    
                    if result_df['divergence_status'].iloc[-1] == "confirmed":  
                        pbar.update(1)
                        continue
                    
                    result_df_vr = detect_single_divergence(hist, indicator_col='macd')
                    
                    if result_df_vr['divergence_status'].iloc[-1] == "confirmed":  
                        pbar.update(1)
                        continue                       
                                       
                    
                    # 记录结果
                    results.append({
                        "期货代码": stock_code,
                        "名字": table.loc[stock_code, 'display_name'],
                        "tsma5": hist['tsma5'],
                        "tsma5": hist['tsma5'],
                        "tsma13": hist['tsma13'],
                        "macd": hist['macd'],
                        "diff": hist['diff'],
                        "dea": hist['dea'],
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
    #df = pd.DataFrame(results).sort_values(by="平均方差", ascending=True)
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






class FutureFilterStrategy(BaseStrategy):
    """基于future_filter的策略"""

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
