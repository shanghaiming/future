import pandas as pd, os

dir = os.path.expanduser('~/home/futures_platform/data/futures_weighted')
count = 0
for f in sorted(os.listdir(dir))[:10]:
    if not f.endswith('.csv'):
        continue
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
    mask = (df['trend_score'] >= 3) & (df['close'] < df['ma_5']) & (df['rsi'] < 35) & (df['vol_ratio'] > 0.8)
    c = mask.sum()
    count += c
    print(f'{f}: {c}次')
print(f'10个品种总计: {count}次')
print(f'估计80品种2年: {count * 8 * 2}次')
