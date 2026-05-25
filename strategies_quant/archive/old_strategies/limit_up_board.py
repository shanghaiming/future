#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
from core.base_strategy import BaseStrategy
A股股票多周期方差分析程序
依赖：mindgp_api, numpy, pandas, tqdm
"""

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
# 配置参数
CONFIG = {
    "PERIODS": [5, 8, 13, 21, 34, 55],  # 需要计算的周期
    "MIN_DATA_DAYS": 55,                # 需要的最少数据天数
    "BATCH_SIZE": 300,                  # 分批处理数量
    "OUTPUT_FILE": "打板.csv"  # 输出文件名
}
import ipywidgets as widgets
def calculate_indicators(df, M=55, N=34, 
                        window_llv=10, window_hhv=25,
                        ema_period=4):
    

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
      
       
    return df.drop([ 'A1', 'B1', 'NT', 'NT0'], axis=1)




def main():
    # 获取全量股票池
    print("正在获取股票列表...")
    table = get_all_securities(ty='stock', date=None)
    all_stocks = get_all_securities(ty='stock', date=None).index.tolist()
    print(f"共获取到{len(all_stocks)}只A股")
    # 初始化结果容器
    results = []
    valid_stock_count = 0
    
    # 使用进度条处理
    with tqdm(total=len(all_stocks), desc="计算进度") as pbar:
        for i in range(0, len(all_stocks), CONFIG["BATCH_SIZE"]):
            batch = all_stocks[i:i+CONFIG["BATCH_SIZE"]]
            
            for stock_code in batch:
                try:
                    if "ST" in table.loc[stock_code, 'display_name']:
                        pbar.update(1)
                        continue 
                    
                    # 获取历史行情（假设返回DataFrame）
                    price_data = get_price(
                        securities=stock_code,  # 注意参数名是复数但支持单个代码
                        end_date=datetime.date.today().strftime('%Y%m%d'),  # 结束日期设为今天
                        fre_step='1d',           # 日线频率
                        fields=['open','high','low','close','volume', 'turnover_rate', 'high_limit','quote_rate'],
                        fq='pre',                # 前复权
                        bar_count=250,           # 获取250根K线
                        skip_paused=True         # 跳过停牌日
                        ).sort_index()  # 清除证券代码索引层级
                    
                    # 检查数据有效性
                    if len(price_data) < CONFIG["MIN_DATA_DAYS"]:
                        pbar.update(1)
                        continue
                    if price_data['close'].iloc[-1] != price_data['high_limit'].iloc[-1]:
                        pbar.update(1)
                        continue                    
                    #计算指标
                    hist = calculate_indicators(price_data) 
                    if hist['up_mark'].iloc[-1] > 6:
                        print(table.loc[stock_code, 'display_name'], hist['up_mark'].iloc[-1], "日线" )
                        pbar.update(1)
                        continue
                    price_data_120 = get_price(
                        securities=stock_code,  # 注意参数名是复数但支持单个代码
                        end_date=datetime.date.today().strftime('%Y%m%d'),  # 结束日期设为今天
                        fre_step='120m',           # 120min频率
                        fields=['open','high','low','close'],
                        fq='pre',                # 前复权
                        bar_count=250,           # 获取250根K线
                        skip_paused=True         # 跳过停牌日
                        ).sort_index()  # 清除证券代码索引层级                    
                                     
                    #计算指标
                    hist_120 = calculate_indicators(price_data_120) 
                    if hist_120['up_mark'].iloc[-1] > 6:
                        print(table.loc[stock_code, 'display_name'], hist_120['up_mark'].iloc[-1], "120min")
                        pbar.update(1)
                        continue
                    price_data_60 = get_price(
                        securities=stock_code,  # 注意参数名是复数但支持单个代码
                        end_date=datetime.date.today().strftime('%Y%m%d'),  # 结束日期设为今天
                        fre_step='60m',           # 60min频率
                        fields=['open','high','low','close'],
                        fq='pre',                # 前复权
                        bar_count=250,           # 获取250根K线
                        skip_paused=True         # 跳过停牌日
                        ).sort_index()  # 清除证券代码索引层级                    
                                     
                    #计算指标
                    hist_60 = calculate_indicators(price_data_60) 
                    if hist_60['up_mark'].iloc[-1] > 6:
                        print(table.loc[stock_code, 'display_name'], hist_120['up_mark'].iloc[-1], "60min")
                        pbar.update(1)
                        continue
                    

                    
                    results.append({
                        "股票代码": stock_code,
                        "名字": table.loc[stock_code, 'display_name'],
                        "涨幅": price_data['quote_rate'].iloc[-1]
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
    df = pd.DataFrame(results).sort_values(by="涨幅", ascending=False)
    
    # 保存结果
    df.to_csv(CONFIG["OUTPUT_FILE"], index=False, encoding='utf_8_sig')
    
    # 打印摘要
    print("\n" + "="*50)
    print(f"分析完成！有效处理股票数：{valid_stock_count}/{len(all_stocks)}")
    print(f"筛选涨停板股票：\n{df[['股票代码', '名字', '涨幅']].head(70).to_string(index=False)}")
    print(f"完整结果已保存至：{CONFIG['OUTPUT_FILE']}")
    # 调用交互式图表浏览器
    jupyter_stock_charts(df.head(70))  # 使用Jupyter专用控件


def jupyter_stock_charts(df_results):
    """Jupyter专用股票图表浏览器（带均线）"""
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
    fig, ax = plt.subplots(figsize=(14, 6))
    plt.close(fig)  # 避免自动显示
    output = widgets.Output()
    
    # 定义更新图表函数
    def update_chart():
        with output:
            output.clear_output(wait=True)
            ax.clear()
            
            stock_code = stock_codes[current_index]
            stock_name = stock_names[current_index]
            
            # 获取股票日线数据
            price_data = get_price(
                securities=stock_code,
                end_date=datetime.date.today().strftime('%Y%m%d'),
                fre_step='1d',
                fields=['open', 'high', 'low', 'close'],
                fq='pre',
                bar_count=120,  # 显示120天数据
                skip_paused=True
            )
            
            if price_data.empty:
                ax.set_title(f"{stock_name} ({stock_code}) - 数据获取失败", fontsize=14)
                display(fig)
                return
            
            # 计算均线
            price_data['MA5'] = price_data['close'].rolling(window=5).mean()
            price_data['MA10'] = price_data['close'].rolling(window=10).mean()
            price_data['MA20'] = price_data['close'].rolling(window=20).mean()
            
            # 创建整数索引（去除非交易日间隙）
            price_data['index_num'] = range(len(price_data))
            
            # 绘制K线图
            candle_width = 0.8
            up = price_data[price_data.close >= price_data.open]
            down = price_data[price_data.close < price_data.open]
            
            # 上涨K线
            ax.bar(up['index_num'], up.close - up.open, candle_width, 
                   bottom=up.open, color='red', zorder=3)
            ax.bar(up['index_num'], up.high - up.close, 0.15, 
                   bottom=up.close, color='red', zorder=3)
            ax.bar(up['index_num'], up.low - up.open, 0.15, 
                   bottom=up.open, color='red', zorder=3)
            
            # 下跌K线
            ax.bar(down['index_num'], down.close - down.open, candle_width, 
                   bottom=down.open, color='green', zorder=3)
            ax.bar(down['index_num'], down.high - down.open, 0.15, 
                   bottom=down.open, color='green', zorder=3)
            ax.bar(down['index_num'], down.low - down.close, 0.15, 
                   bottom=down.close, color='green', zorder=3)
            
            # 绘制均线
            ax.plot(price_data['index_num'], price_data['MA5'], 'b-', linewidth=0.8, label='5日均线', zorder=2)
            ax.plot(price_data['index_num'], price_data['MA10'], 'm-', linewidth=0.8, label='10日均线', zorder=2)
            ax.plot(price_data['index_num'], price_data['MA20'], 'c-', linewidth=0.8, label='20日均线', zorder=2)
            
            # 设置标题和标签
            last_close = price_data['close'].iloc[-1]
            last_date = price_data.index[-1].strftime('%Y-%m-%d')
            ax.set_title(f"{stock_name} ({stock_code}) - 最新价: {last_close:.2f} ({last_date})", 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('交易日序列')
            ax.set_ylabel('价格')
            ax.legend(loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # 设置x轴刻度
            n = len(price_data)
            step = max(1, n // 20)
            xticks = list(range(0, n, step))
            if n-1 not in xticks:
                xticks.append(n-1)
            xticklabels = [price_data.index[i].strftime('%m-%d') for i in xticks]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            
            # 设置y轴范围
            y_min = price_data[['low', 'MA5', 'MA10', 'MA20']].min().min()
            y_max = price_data[['high', 'MA5', 'MA10', 'MA20']].max().max()
            ax.set_ylim(y_min * 0.98, y_max * 1.02)
            
            display(fig)
    
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






class LimitUpBoardStrategy(BaseStrategy):
    """基于limit_up_board的策略"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化代码
        self.name = "LimitUpBoardStrategy"
        self.description = "基于limit_up_board的策略"
        
    def calculate_signals(self, df):
        """计算交易信号"""
        # 策略逻辑
        return df
        
    def generate_signals(self, df):
        """生成交易信号"""
        # 信号生成逻辑
        return df
