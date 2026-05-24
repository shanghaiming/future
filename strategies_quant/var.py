#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
A股股票多周期方差分析程序
依赖：mindgp_api, numpy, pandas, tqdm
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from core.base_strategy import BaseStrategy
from core.data_loader import *  # 假设该库直接提供API函数
from numpy.lib.stride_tricks import sliding_window_view
# 配置参数
CONFIG = {
    "PERIODS": [5, 8, 13, 21, 34, 55],  # 需要计算的周期
    "MIN_DATA_DAYS": 55,                # 需要的最少数据天数
    "BATCH_SIZE": 300,                  # 分批处理数量
    "OUTPUT_FILE": "stock_variance_rank.csv"  # 输出文件名
}
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
    # 计算ER（效率比率）
    change = df['close'].diff(14).abs()
    volatility = df['close'].diff().abs().rolling(14).sum()
    df['er'] = change / (volatility + 1e-7)
    
    # 清理中间列
    return df.drop(['LC','CLOSE_LC'], axis=1)

import pandas as pd

import pandas as pd
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
def detect_single_divergence(df, indicator_col='macd', window_size=14):
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
                    # 获取历史行情（假设返回DataFrame）
                    price_data = get_price(
                        securities=stock_code,  # 注意参数名是复数但支持单个代码
                        end_date='20250517',  # 结束日期设为今天
                        fre_step='1d',           # 日线频率
                        fields=['open','high','low','close','volume', 'turnover_rate'],
                        fq='pre',                # 前复权
                        bar_count=250,           # 获取250根K线
                        skip_paused=True         # 跳过停牌日
                        ).sort_index()  # 清除证券代码索引层级
                    
                    # 检查数据有效性
                    if len(price_data) < CONFIG["MIN_DATA_DAYS"]:
                        pbar.update(1)
                        continue
                    # 计算20日平均换手率
                    price_data['turnover_ma20'] = price_data['turnover_rate'].rolling(20).mean()
        
                    # 筛选最新20日均值 <1 的股票
                    if price_data['turnover_ma20'].iloc[-1] < 1:  # 取最后一天的20日均值
                        pbar.update(1)
                        continue
                    hist = calculate_indicators(price_data)
                    
                    result_df = detect_single_divergence(hist)
                    
                    if result_df['divergence_status'].iloc[-1] == "confirmed":  
                        pbar.update(1)
                        continue
                        
                        
                    closes = price_data['close'].values
                    
                    # 计算各周期方差
                    variances = []
                    for period in CONFIG["PERIODS"]:
                        period_data = closes[-period:]
                        variances.append(np.var(period_data, ddof=1)/(closes*closes))  # 样本方差
                    
                    # 记录结果
                    results.append({
                        "股票代码": stock_code,
                        "名字": table.loc[stock_code, 'display_name'],
                        "平均方差": np.mean(variances),
                        "5日方差": variances[0],
                        "8日方差": variances[1],
                        "13日方差": variances[2],
                        "21日方差": variances[3],
                        "34日方差": variances[4],
                        "55日方差": variances[5]
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
    df = pd.DataFrame(results).sort_values(by="平均方差", ascending=True)
    
    # 保存结果
    df.to_csv(CONFIG["OUTPUT_FILE"], index=False, encoding='utf_8_sig')
    
    # 打印摘要
    print("\n" + "="*50)
    print(f"分析完成！有效处理股票数：{valid_stock_count}/{len(all_stocks)}")
    print(f"方差最小的30只股票：\n{df[['股票代码', '名字', '平均方差']].head(30).to_string(index=False)}")
    print(f"完整结果已保存至：{CONFIG['OUTPUT_FILE']}")
    notify_push(
    df[['股票代码', '名字', '平均方差']].head(10).to_string(index=False), 
    channel='wxpusher', 
    subject='SuperMind消息提醒', 
    email_list=None, 
    uids='UID_whd6sWkQtfsrIEFEifgZZWyLufEI', 
    topic_ids=None, 
    group_id=None,
    url=None,
    payload=None,
)

if __name__ == "__main__":
       
    main()

# In[ ]:




# In[ ]:




# In[ ]:






class VarStrategy(BaseStrategy):
    """基于var的策略"""

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
