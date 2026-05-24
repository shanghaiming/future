import pandas as pd, os, numpy as np

dir = os.path.expanduser('~/home/futures_platform/data/futures_weighted')

# 测试突破策略
conditions = [
    ('价格突破MA20 + VOL>1.5', lambda df: (df['close'] > df['ma_20']) & (df['close'].shift(1) <= df['ma_20'].shift(1)) & (df['vol_ratio'] > 1.5)),
    ('价格突破MA20 + VOL>2.0', lambda df: (df['close'] > df['ma_20']) & (df['close'].shift(1) <= df['ma_20'].shift(1)) & (df['vol_ratio'] > 2.0)),
    ('价格突破MA20 + VOL>1.5 + 趋势分>=3', lambda df: (df['close'] > df['ma_20']) & (df['close'].shift(1) <= df['ma_20'].shift(1)) & (df['vol_ratio'] > 1.5) & (df['trend_score'] >= 3)),
    ('价格突破MA10 + VOL>1.5', lambda df: (df['close'] > df['ma_10']) & (df['close'].shift(1) <= df['ma_10'].shift(1)) & (df['vol_ratio'] > 1.5)),
    ('RSI突破50 + VOL>1.5', lambda df: (df['rsi'] > 50) & (df['rsi'].shift(1) <= 50) & (df['vol_ratio'] > 1.5)),
    ('RSI突破60 + VOL>1.5', lambda df: (df['rsi'] > 60) & (df['rsi'].shift(1) <= 60) & (df['vol_ratio'] > 1.5)),
]

for name, cond in conditions:
    signals = []
    for f in sorted(os.listdir(dir))[:40]:
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
        
        df['vol_ma20'] = df['vol'].rolling(20).mean()
        df['vol_ratio'] = df['vol'] / df['vol_ma20']
        
        mask = cond(df)
        for i in range(60, len(df)-10):
            if mask.iloc[i]:
                future_ret = (df['close'].iloc[i+10] - df['close'].iloc[i]) / df['close'].iloc[i]
                signals.append(future_ret)
    
    if signals:
        r = pd.Series(signals)
        print(f'{name}: 信号={len(r)} 胜率={(r>0).mean():.1%} 平均={r.mean():.2%}')
    else:
        print(f'{name}: 无信号')
