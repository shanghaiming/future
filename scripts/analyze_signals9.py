import pandas as pd, os, numpy as np

dir = os.path.expanduser('~/home/futures_platform/data/futures_weighted')

# 计算信号频率
total_days = 0
total_signals = 0
for f in sorted(os.listdir(dir))[:40]:
    if not f.endswith('.csv'): continue
    df = pd.read_csv(os.path.join(dir, f))
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    
    df['vol_ma20'] = df['vol'].rolling(20).mean()
    df['vol_ratio'] = df['vol'] / df['vol_ma20']
    
    mask = (df['rsi'] > 60) & (df['rsi'].shift(1) <= 60) & (df['vol_ratio'] > 2.0)
    signals = mask.sum()
    days = len(df) - 60
    
    total_signals += signals
    total_days += days
    print(f'{f}: {signals}次 / {days}天 = {signals/days*100:.2f}%')

print(f'\n总计: {total_signals}次 / {total_days}天 = {total_signals/total_days*100:.2f}%')
print(f'80品种估计: {total_signals * 2}次 / {total_days}天')
print(f'每天信号数: {total_signals * 2 / total_days * 80:.1f}')
