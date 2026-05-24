#!/usr/bin/env python3
"""
期货期限结构历史回补脚本
用akshare/新浪拉各合约历史日线，逐日构建期限结构快照
"""

import os
for k in list(os.environ.keys()):
    if 'proxy' in k.lower():
        del os.environ[k]

import sys
import time
import json
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR = os.path.expanduser("~/home/futures_platform")
DATA_DIR = os.path.join(BASE_DIR, "data")
TERM_DIR = os.path.join(DATA_DIR, "futures_term_structure")
os.makedirs(TERM_DIR, exist_ok=True)

import akshare as ak

# 品种 → 新浪代码前缀 + 显示名
PRODUCT_MAP = {
    'rbfi': ('RB', '螺纹钢'), 'hcfi': ('HC', '热卷'), 'ifi': ('I', '铁矿石'),
    'jfi': ('J', '焦炭'), 'jmfi': ('JM', '焦煤'), 'sffi': ('SF', '硅铁'),
    'smfi': ('SM', '锰硅'), 'aufi': ('AU', '黄金'), 'agfi': ('AG', '白银'),
    'cufi': ('CU', '沪铜'), 'alfi': ('AL', '沪铝'), 'znfi': ('ZN', '沪锌'),
    'pbfi': ('PB', '沪铅'), 'nifi': ('NI', '沪镍'), 'snfi': ('SN', '沪锡'),
    'ssfi': ('SS', '不锈钢'), 'bufi': ('BU', '沥青'), 'rufi': ('RU', '橡胶'),
    'fufi': ('FU', '燃油'), 'spfi': ('SP', '纸浆'), 'pgfi': ('PG', 'LPG'),
    'ebfi': ('EB', '苯乙烯'), 'egfi': ('EG', '乙二醇'), 'mafi': ('MA', '甲醇'),
    'tafi': ('TA', 'PTA'), 'ppfi': ('PP', '聚丙烯'), 'lfi': ('L', '塑料'),
    'vfi': ('V', 'PVC'), 'yfi': ('Y', '豆油'), 'pfi': ('P', '棕榈油'),
    'cfi': ('C', '玉米'), 'csfi': ('CS', '淀粉'), 'afi': ('A', '豆一'),
    'bfi': ('B', '豆二'), 'mfi': ('M', '豆粕'), 'yfi': ('Y', '豆油'),
    'rmfi': ('RM', '菜粕'), 'oifi': ('OI', '菜油'), 'apfi': ('AP', '苹果'),
    'cjfi': ('CJ', '红枣'), 'srfi': ('SR', '白糖'), 'cffi': ('CF', '棉花'),
    'pffi': ('PF', '短纤'), 'safi': ('SA', '纯碱'), 'fgfi': ('FG', '玻璃'),
    'urfi': ('UR', '尿素'), 'shfi': ('SH', '烧碱'), 'pkfi': ('PK', '花生'),
    'lrfi': ('LR', '晚稻'), 'rrfi': ('RR', '粳米'), 'prfi': ('PR', '瓶片'),
    'pxfi': ('PX', 'PX'), 'bcfi': ('BC', '国际铜'), 'scfi': ('SC', '原油'),
    'nrfi': ('NR', '20号胶'), 'aofi': ('AO', '氧化铝'), 'brfi': ('BR', '丁二烯'),
    'ecfi': ('EC', '集运'), 'lcfi': ('LC', '碳酸锂'), 'sifi': ('SI', '工业硅'),
    'pdfi': ('PD', '钯'), 'ptfi': ('PT', '铂'), 'psfi': ('PS', '镁'),
    'opfi': ('OP', '胶版纸'), 'adfi': ('AD', '合成氨'), 'lgfi': ('LG', '液化气'),
    'lufi': ('LU', '低硫燃油'), 'fbfi': ('FB', '纤维板'), 'bbfi': ('BB', '胶合板'),
    'whfi': ('WH', '强麦'), 'pmfi': ('PM', '普麦'), 'rifi': ('RI', '早稻'),
    'jrfi': ('JR', '粳稻'), 'rsfi': ('RS', '菜籽'), 'zcfi': ('ZC', '动力煤'),
    'plfi': ('PL', '棕榈'), 'bzi': ('BZ', '苯乙烯(旧)'), 'lhfi': ('LH', '生猪'),
    'jd': ('JD', '鸡蛋'),
}


def generate_contract_codes(prefix, start_year=24, end_year=27):
    """生成品种的所有合约代码（如RB2401~RB2712）"""
    codes = []
    for yy in range(start_year, end_year + 1):
        for mm in range(1, 13):
            codes.append(f"{prefix}{yy}{mm:02d}")
    return codes


def fetch_contract_history(prefix, contract_code):
    """拉单个合约的历史日线"""
    try:
        df = ak.futures_zh_daily_sina(symbol=contract_code)
        if df.empty:
            return None
        df['date'] = pd.to_datetime(df['date'])
        return df[['date', 'close', 'volume', 'hold']].rename(columns={
            'close': 'price', 'volume': 'vol', 'hold': 'oi'
        })
    except:
        return None


def backfill_product(symbol, prefix, name, start_date='2024-01-01'):
    """回补单个品种的期限结构历史"""
    # 已有日期
    existing_dates = set()
    for f in os.listdir(TERM_DIR):
        if f.startswith(symbol + '_') and f.endswith('.json'):
            d = f.replace(symbol + '_', '').replace('.json', '')
            existing_dates.add(d)

    # 生成合约代码列表
    contracts = generate_contract_codes(prefix)
    
    print(f"[{symbol}] 拉取{len(contracts)}个合约历史...")
    
    # 拉所有合约数据
    all_data = {}
    for i, code in enumerate(contracts):
        df = fetch_contract_history(prefix, code)
        if df is not None and len(df) > 0:
            all_data[code] = df
        if (i + 1) % 20 == 0:
            print(f"  [{symbol}] 已拉{i+1}/{len(contracts)}个合约, 有数据:{len(all_data)}")
            time.sleep(0.3)  # 限流
    
    if not all_data:
        print(f"[{symbol}] 无数据")
        return 0
    
    # 构建日期 → 合约价格表
    all_dates = set()
    for df in all_data.values():
        for d in df['date']:
            all_dates.add(d.strftime('%Y%m%d'))
    
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    target_dates = sorted([d for d in all_dates if d >= start_dt.strftime('%Y%m%d')])
    new_count = 0
    
    for date_str in target_dates:
        if date_str in existing_dates:
            continue
        
        date_dt = pd.Timestamp(date_str)
        # 找当日有数据的合约
        day_contracts = []
        for code, df in all_data.items():
            row = df[df['date'] == date_dt]
            if len(row) > 0 and row.iloc[0]['price'] > 0:
                price = float(row.iloc[0]['price'])
                vol = int(row.iloc[0]['vol']) if pd.notna(row.iloc[0]['vol']) else 0
                # 解析合约年月
                num_part = code[len(prefix):]
                try:
                    yy = int(num_part[:2])
                    mm = int(num_part[2:])
                except:
                    continue
                day_contracts.append({
                    'symbol': code, 'name': f'{name}{yy}{mm:02d}',
                    'price': price, 'year': 2000 + yy, 'month': mm, 'volume': vol
                })
        
        if len(day_contracts) < 2:
            continue
        
        # 按年月排序，取前4个
        day_contracts.sort(key=lambda x: (x['year'], x['month']))
        curve = day_contracts[:4]
        near, far = curve[0], curve[-1]
        spread = far['price'] - near['price']
        spread_pct = (spread / near['price'] * 100) if near['price'] > 0 else 0
        
        record = {
            'symbol': symbol, 'name': name,
            'date': f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}",
            'structure': 'contango' if spread > 0 else 'backwardation',
            'curve': curve,
            'near_contract': near['symbol'], 'near_price': near['price'],
            'far_contract': far['symbol'], 'far_price': far['price'],
            'total_spread': spread, 'total_spread_pct': round(spread_pct, 4),
        }
        
        out_path = os.path.join(TERM_DIR, f"{symbol}_{date_str}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        new_count += 1
    
    print(f"[{symbol}] 完成, 新增{new_count}天")
    return new_count


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="期限结构历史回补")
    parser.add_argument("--symbol", default=None, help="只跑某个品种(如rbfi)")
    parser.add_argument("--start-date", default="2024-01-01", help="回补起始日")
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"期货期限结构历史回补")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"起始日: {args.start_date}")
    print("=" * 60)
    
    if args.symbol:
        # 单品种
        if args.symbol in PRODUCT_MAP:
            prefix, name = PRODUCT_MAP[args.symbol]
            backfill_product(args.symbol, prefix, name, args.start_date)
        else:
            print(f"未知品种: {args.symbol}")
    else:
        # 全部品种
        total_new = 0
        for i, (symbol, (prefix, name)) in enumerate(PRODUCT_MAP.items()):
            print(f"\n--- {i+1}/{len(PRODUCT_MAP)} {symbol} ({name}) ---")
            n = backfill_product(symbol, prefix, name, args.start_date)
            total_new += n
        
        print(f"\n=== 全部完成 === 总新增: {total_new}天")
