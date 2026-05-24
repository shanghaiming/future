import pandas as pd, os, numpy as np

dir = os.path.expanduser('~/home/futures_platform/data/futures_weighted')

# 测试组合条件
conditions = [
    ('偏离>0.5% + RSI<40', lambda df: (df['trend_score'] >= 2) & (df['close'] < df['ma_5']) & (df['ma5_dev'] < -0.005) & (df['rsi'] < 40)),
    ('偏离>0.5% + RSI<45', lambda df: (df['trend_score'] >= 2) & (df['close'] < df['ma_5']) & (df['ma5_dev'] < -0.005) & (df['rsi'] < 45)),
    ('偏离>0.5% + VOL>1.0', lambda df: (df['trend_score'] >= 2) & (df['close'] < df['ma_5']) & (df['ma5_dev'] < -0.005) & (df['vol_ratio'] > 1.0)),
    ('偏离>0.5% + VOL>1.2', lambda df: (df['trend_score'] >= 2) & (df['close'] < df['ma_5']) & (df['ma5_dev'] < -0.005) & (df['vol_ratio'] > 1.2)),
    ('偏离>0.5% + RSI<40 + VOL>1.0', lambda df: (df['trend_score'] >= 2) & (df['close'] < df['ma_5']) & (df['ma5_dev'] < -0.005) & (df['rsi'] < 40) & (df['vol_ratio'] > 1.0)),
    ('偏离>1.0% + RSI<40', lambda df: (df['trend_score'] >= 2) & (df['close'] < df['ma_5']) & (df['ma5_dev'] < -0.01) & (df['rsi'] < 40)),
    ('偏离>1.0% + VOL>1.0', lambda df: (df['trend_score'] >= 2) & (df['close'] < df['ma_5']) & (df['ma5_dev'] < -0.01) & (df['vol_ratio'] > 1.0)),
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
        df['ma5_dev'] = (df['close'] - df['ma_5']) / df['ma_5']
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        
        df['vol_ma20'] = df['vol'].rolling(20).mean()
        df['vol_ratio'] = df['vol'] / df['vol_ma20']
        
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
