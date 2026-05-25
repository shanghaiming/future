#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import time
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
    
    #均线
    df['ma5'] = df['close'].iloc[-5:].mean()
    df['ma55'] = df['close'].iloc[-55:].mean()
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
    pending_stocks = []
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
                    if price_data['close'].iloc[-2] == price_data['high_limit'].iloc[-2]:
                        pbar.update(1)
                        continue                    
                    #计算指标
                    hist = calculate_indicators(price_data) 
                    if hist['up_mark'].iloc[-2] > 6:
                        print(table.loc[stock_code, 'display_name'], hist['up_mark'].iloc[-2], "日线" )
                        pbar.update(1)
                        continue
                    if hist['ma5'].iloc[-2] > hist['ma55'].iloc[-2]:
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
                    if hist_120['up_mark'].iloc[-2] > 6:
                        print(table.loc[stock_code, 'display_name'], hist_120['up_mark'].iloc[-2], "120min")
                        pbar.update(1)
                        continue
                    
                    

                    
                    pending_stocks.append({
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
    if not pending_stocks:
        print("没有有效数据可供分析")
        return

    # 创建DataFrame并排序
    df = pd.DataFrame(pending_stocks).sort_values(by="涨幅", ascending=False)

    
    # 保存结果
    df.to_csv(CONFIG["OUTPUT_FILE"], index=False, encoding='utf_8_sig')
    
    # 打印摘要
    print("\n" + "="*50)
    print(f"分析完成！待处理股票数：{valid_stock_count}/{len(all_stocks)}")
    
    # 调用交互式图表浏览器
    



if __name__ == "__main__":
    # 执行主程序获取筛选结果
    #main_results = main()
    # 从本地文件读取主程序结果
    
    try:
        main_results = pd.read_csv(CONFIG["OUTPUT_FILE"], encoding='utf_8_sig')
        print(f"已从本地文件加载结果: {CONFIG['OUTPUT_FILE']}")
    except FileNotFoundError:
        print(f"错误: 未找到结果文件 {CONFIG['OUTPUT_FILE']}")
        exit(1)
    
    
    
    
    # 持续运行的同花顺问财匹配逻辑
    # 初始化变量（放在循环之前）
    import pandas as pd
    import time
    from datetime import datetime
    
    # 初始化数据结构
    matched_stocks = []          # 存储所有匹配的股票 (code, name)
    last_matched_stocks = set()  # 记录上一次循环的匹配股票集合
    current_date = datetime.now().strftime("%Y-%m-%d")  # 获取当前日期
    
    # 定义CSV文件名（固定文件名）
    output_filename = "matched_stocks.csv"
    
    try:
        while True:
            try:
                # 获取问财结果（伪代码，实际需替换为真实查询）
                iwencai_results = query_iwencai(
                    "沪深最高涨幅大于8%；排除创业板；排除科创板;"
                    "3日内没有涨停或者首板;换手率大于8%小于16%;主力净量大于0.5"
                )
            
                new_stocks_found = False
                current_matched_codes = set()  # 当前轮次匹配的股票代码
                
                # 检查匹配的股票
                for _, row in iwencai_results.iterrows():
                    stock_code = row['股票代码']
                    if stock_code in main_results['股票代码'].values:
                        stock_name = row['股票简称']
                        current_matched_codes.add(stock_code)
                        
                        # 如果是新股票则添加到列表
                        if stock_code not in last_matched_stocks:
                            print(f"新增匹配股票: {stock_name}({stock_code})")
                            matched_stocks.append((stock_code, stock_name))
                            new_stocks_found = True
                
                # 检测变化并打印更新
                if new_stocks_found or current_matched_codes != last_matched_stocks:
                    if matched_stocks:
                        print("\n=== 当前匹配股票列表（已更新）===")
                        for idx, (code, name) in enumerate(matched_stocks, 1):
                            status = " [新增]" if code in (current_matched_codes - last_matched_stocks) else ""
                            print(f"{idx}. {name}({code}){status}")
                        print("=============================\n")
                    
                    last_matched_stocks = current_matched_codes.copy()
                
                time.sleep(60)  # 每分钟查询一次
            
            except Exception as e:
                print(f"查询出错: {str(e)}")
                time.sleep(30)
    
    except KeyboardInterrupt:
        # 用户中断程序时执行保存操作
        print("\n程序被中断，正在保存结果...")
        
        # 创建包含日期列的DataFrame
        result_df = pd.DataFrame({
            "股票代码": [code for code, _ in matched_stocks],
            "股票名称": [name for _, name in matched_stocks],
            "日期": current_date  # 添加日期列
        })
        
        # 保存到CSV（追加模式）
        try:
            # 尝试读取现有文件
            existing_df = pd.read_csv(output_filename)
            # 合并新旧数据
            combined_df = pd.concat([existing_df, result_df], ignore_index=True)
            # 去除重复项
            combined_df = combined_df.drop_duplicates(subset=["股票代码", "日期"])
            # 保存合并后的数据
            combined_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
            print(f"已追加匹配股票到: {output_filename}")
            
        except FileNotFoundError:
            # 如果文件不存在，则创建新文件
            result_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
            print(f"已创建匹配股票文件: {output_filename}")
        
        print(f"共保存 {len(matched_stocks)} 只股票")

# In[ ]:






class RushBuyStrategy(BaseStrategy):
    """基于rush_buy的策略"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化代码
        self.name = "RushBuyStrategy"
        self.description = "基于rush_buy的策略"
        
    def calculate_signals(self, df):
        """计算交易信号"""
        # 策略逻辑
        return df
        
    def generate_signals(self, df):
        """生成交易信号"""
        # 信号生成逻辑
        return df
