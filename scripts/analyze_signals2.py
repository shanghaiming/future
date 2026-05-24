import pandas as pd, os

dir = os.path.expanduser('~/home/futures_platform/data/futures_weighted')

# 测试不同条件的信号频率和胜率
conditions = [
    ('趋势>=3 + 价格<MA5', lambda df: (df['trend_score'] >= 3) & (df['close'] < df['ma_5'])),
    ('趋势>=3 + 价格<MA5 + RSI<40', lambda df: (df['trend_score'] >= 3) & (df['close'] < df['ma_5']) & (df['rsi'] < 40)),
    ('趋势>=3 + 价格<MA5 + RSI<45', lambda df: (df['trend_score'] >= 3) & (df['close'] < df['ma_5']) & (df['rsi'] < 45)),
    ('趋势>=3 + 价格<MA10', lambda df: (df['trend_score'] >= 3) & (df['close'] < df['ma_10'])),
    ('趋势>=2 + 价格<MA5', lambda df: (df['trend_score'] >= 2) & (df['close'] < df['ma_5'])),
    ('趋势>=2 + 价格<MA10', lambda df: (df['trend_score'] >= 2) & (df['close'] < df['ma_10'])),
    ('趋势>=2 + 价格<MA10 + RSI<50', lambda df: (df['trend_score'] >= 2) & (df['close'] < df['ma_10']) & (df['rsi'] < 50)),
]

for name, cond in conditions:
    signals = []
    for f in sorted(os.listdir(dir))[:30]:
        if not f.endswith('.csv'): continue
        df = pd.read_csv(os.path.join(dir, f))
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        for p in [5, 10, 20, 60]:
            df[f'ma_{p}'] = df['close'].rolling(p).mean()
        df['trend_score'] = sum((df['close'] > df[f'ma_{p}']).astype(int) for p in [5, 10, 20, 60])
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        
        mask = cond(df)
        for i in range(60, len(df)-5):
            if mask.iloc[i]:
                future_ret = (df['close'].iloc[i+5] - df['close'].iloc[i]) / df['close'].iloc[i]
                signals.append(future_ret)
    
    if signals:
        r = pd.Series(signals)
        print(f'{name}: 信号={len(r)} 胜率={(r>0).mean():.1%} 平均={r.mean():.2%}')
    else:
        print(f'{name}: 无信号')
