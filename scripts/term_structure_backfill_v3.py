#!/usr/bin/env python3
"""
期货期限结构历史回补 v3 (5年, 全商品, 新浪接口)
用akshare/新浪拉各合约历史日线，逐日构建期限结构快照
"""

import os
for k in list(os.environ.keys()):
    if 'proxy' in k.lower():
        del os.environ[k]

import sys
import time
import json
import subprocess
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.expanduser("~/home/futures_platform")
DATA_DIR = os.path.join(BASE_DIR, "data")
TERM_DIR = os.path.join(DATA_DIR, "futures_term_structure")
os.makedirs(TERM_DIR, exist_ok=True)

# 全商品品种配置: symbol -> (exchange, contract_prefix, name, year_digits)
# year_digits: 4=标准格式(RB2609), 3=CZCE格式(AP501)
PRODUCTS = {
    # === SHFE 黑色 ===
    'rbfi': ('SHFE', 'RB', '螺纹钢', 4),
    'hcfi': ('SHFE', 'HC', '热卷', 4),
    'ifi':  ('SHFE', 'I',  '铁矿石', 4),
    'jfi':  ('SHFE', 'J',  '焦炭', 4),
    'jmfi': ('SHFE', 'JM', '焦煤', 4),
    'sffi': ('SHFE', 'SF', '硅铁', 4),
    'smfi': ('SHFE', 'SM', '锰硅', 4),
    'wrfi': ('SHFE', 'WR', '线材', 4),
    # === SHFE 有色 ===
    'cufi': ('SHFE', 'CU', '沪铜', 4),
    'alfi': ('SHFE', 'AL', '沪铝', 4),
    'znfi': ('SHFE', 'ZN', '沪锌', 4),
    'pbfi': ('SHFE', 'PB', '沪铅', 4),
    'nifi': ('SHFE', 'NI', '沪镍', 4),
    'snfi': ('SHFE', 'SN', '沪锡', 4),
    'ssfi': ('SHFE', 'SS', '不锈钢', 4),
    'aofi': ('SHFE', 'AO', '氧化铝', 4),
    # === SHFE 贵金属 ===
    'aufi': ('SHFE', 'AU', '黄金', 4),
    'agfi': ('SHFE', 'AG', '白银', 4),
    # === SHFE 能源化工 ===
    'rufi': ('SHFE', 'RU', '橡胶', 4),
    'bufi': ('SHFE', 'BU', '沥青', 4),
    'fufi': ('SHFE', 'FU', '燃油', 4),
    'spfi': ('SHFE', 'SP', '纸浆', 4),
    'brfi': ('SHFE', 'BR', '丁二烯', 4),
    # === INE ===
    'scfi': ('INE', 'SC', '原油', 4),
    'nrfi': ('INE', 'NR', '20号胶', 4),
    'bcfi': ('INE', 'BC', '国际铜', 4),
    'lufi': ('INE', 'LU', '低硫燃油', 4),
    # === DCE 黑色/化工 ===
    'jfi2':  ('DCE', 'j',  '焦炭DCE', 4),
    'jmfi2': ('DCE', 'jm', '焦煤DCE', 4),
    'ifi2':  ('DCE', 'i',  '铁矿石DCE', 4),
    'egfi':  ('DCE', 'eg', '乙二醇', 4),
    'pgfi':  ('DCE', 'pg', 'LPG', 4),
    'ebfi':  ('DCE', 'eb', '苯乙烯', 4),
    'ppfi':  ('DCE', 'pp', '聚丙烯', 4),
    'lfi':   ('DCE', 'l',  '塑料', 4),
    'vfi':   ('DCE', 'v',  'PVC', 4),
    # === DCE 农产品 ===
    'mfi':  ('DCE', 'm',  '豆粕', 4),
    'yfi':  ('DCE', 'y',  '豆油', 4),
    'pfi':  ('DCE', 'p',  '棕榈油', 4),
    'cfi':  ('DCE', 'c',  '玉米', 4),
    'csfi': ('DCE', 'cs', '淀粉', 4),
    'afi':  ('DCE', 'a',  '豆一', 4),
    'bfi':  ('DCE', 'b',  '豆二', 4),
    'jdfi': ('DCE', 'jd', '鸡蛋', 4),
    'lhfi': ('DCE', 'lh', '生猪', 4),
    'fbfi': ('DCE', 'fb', '纤维板', 4),
    'bbfi': ('DCE', 'bb', '胶合板', 4),
    # === CZCE ===
    'apfi': ('CZCE', 'AP', '苹果', 3),
    'cffi': ('CZCE', 'CF', '棉花', 3),
    'cjfi': ('CZCE', 'CJ', '红枣', 3),
    'cyfi': ('CZCE', 'CY', '棉纱', 3),
    'fgfi': ('CZCE', 'FG', '玻璃', 3),
    'mafi': ('CZCE', 'MA', '甲醇', 3),
    'oifi': ('CZCE', 'OI', '菜油', 3),
    'pffi': ('CZCE', 'PF', '短纤', 3),
    'pkfi': ('CZCE', 'PK', '花生', 3),
    'plfi': ('CZCE', 'PL', '涤纶短纤', 3),
    'prfi': ('CZCE', 'PR', '瓶片', 3),
    'pxfi': ('CZCE', 'PX', 'PX', 3),
    'rmfi': ('CZCE', 'RM', '菜粕', 3),
    'rsfi': ('CZCE', 'RS', '菜籽', 3),
    'safi': ('CZCE', 'SA', '纯碱', 3),
    'sffi': ('CZCE', 'SF', '硅铁CZCE', 3),
    'shfi': ('CZCE', 'SH', '烧碱', 3),
    'smfi': ('CZCE', 'SM', '锰硅CZCE', 3),
    'srfi': ('CZCE', 'SR', '白糖', 3),
    'tafi': ('CZCE', 'TA', 'PTA', 3),
    'urfi': ('CZCE', 'UR', '尿素', 3),
    'whfi': ('CZCE', 'WH', '强麦', 3),
    'zcfi': ('CZCE', 'ZC', '动力煤', 3),
    'jrfi': ('CZCE', 'JR', '粳稻', 3),
    'lrfi': ('CZCE', 'LR', '晚稻', 3),
    'pmfi': ('CZCE', 'PM', '普麦', 3),
    'rrfi': ('CZCE', 'RR', '粳米', 3),
    # === GFEX ===
    'sifi': ('GFEX', 'si', '工业硅', 4),
    'lcfi': ('GFEX', 'lc', '碳酸锂', 4),
    'pdfi': ('GFEX', 'pd', '钯', 4),
    'psfi': ('GFEX', 'ps', '镁', 4),
    'ptfi': ('GFEX', 'pt', '铂', 4),
}


def parse_contract_year_month(num_part, year_digits):
    """解析合约代码的年月部分"""
    try:
        if year_digits == 4:
            # RB2609 -> 2026年9月
            yy = int(num_part[:2]) + 2000
            mm = int(num_part[2:])
        elif year_digits == 3:
            # AP501 -> 2025年1月, AP110 -> 2021年10月
            yy = int(num_part[0]) + 2020
            mm = int(num_part[1:3])
        else:
            return None, None
        if 1 <= mm <= 12 and 2020 <= yy <= 2030:
            return yy, mm
    except:
        pass
    return None, None


def generate_contract_codes(prefix, year_digits, start_year=21, end_year=27):
    """生成品种的所有合约代码"""
    codes = []
    for yy in range(start_year, end_year + 1):
        for mm in range(1, 13):
            if year_digits == 4:
                codes.append(f"{prefix}{yy}{mm:02d}")
            elif year_digits == 3:
                # CZCE: 21->1, 25->5, 30->0(不对, 2030->A0?)
                single_y = yy - 2020
                codes.append(f"{prefix}{single_y}{mm:02d}")
    return codes


def backfill_product(symbol, exchange, prefix, name, year_digits, start_date='2021-01-01'):
    """回补单个品种的期限结构历史"""
    import akshare as ak

    existing_dates = set()
    for f in os.listdir(TERM_DIR):
        if f.startswith(symbol + '_') and f.endswith('.json'):
            existing_dates.add(f.replace(symbol + '_', '').replace('.json', ''))

    contracts = generate_contract_codes(prefix, year_digits)
    print(f"  [{symbol}] 拉取{len(contracts)}个合约...")

    all_data = {}
    for i, code in enumerate(contracts):
        try:
            df = ak.futures_zh_daily_sina(symbol=code)
            if df is not None and len(df) > 0:
                df['date_str'] = df['date'].astype(str).str.replace('-', '')
                all_data[code] = df
        except:
            pass
        if (i + 1) % 24 == 0:
            time.sleep(1)  # 限流：每24个合约停1秒

    if not all_data:
        print(f"  [{symbol}] 无数据")
        return 0

    # 按日期汇总
    start_str = start_date.replace('-', '')
    contracts_by_date = {}
    for code, df in all_data.items():
        num_part = code[len(prefix):]
        yy, mm = parse_contract_year_month(num_part, year_digits)
        if yy is None:
            continue

        for _, row in df.iterrows():
            d = str(row.get('date_str', ''))
            if not d or d < start_str:
                continue
            price = float(row['close']) if pd.notna(row.get('close')) else 0
            vol = int(row['volume']) if pd.notna(row.get('volume', 0)) else 0
            if price > 0:
                if d not in contracts_by_date:
                    contracts_by_date[d] = []
                contracts_by_date[d].append({
                    'symbol': code.upper(), 'name': f'{name}{yy-2000:02d}{mm:02d}',
                    'price': price, 'year': yy, 'month': mm, 'volume': vol
                })

    # 构建期限结构
    new_count = 0
    for date_str in sorted(contracts_by_date.keys()):
        if date_str in existing_dates:
            continue
        day_contracts = contracts_by_date[date_str]
        if len(day_contracts) < 2:
            continue

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

    print(f"  [{symbol}] +{new_count}天 (共{len(contracts_by_date)}交易日, {len(all_data)}合约有数据)")
    return new_count


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="期限结构历史回补 v3 (全商品, 5年)")
    parser.add_argument("--symbol", default=None, help="只跑某个品种(如rbfi)")
    parser.add_argument("--start-date", default="2021-01-01", help="回补起始日")
    args = parser.parse_args()

    print("=" * 60)
    print(f"期货期限结构历史回补 v3 (全商品, 5年)")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"起始日: {args.start_date}")
    print(f"品种数: {len(PRODUCTS)}")
    print("=" * 60)

    total_new = 0
    success = fail = 0

    for i, (symbol, (exchange, prefix, name, year_digits)) in enumerate(PRODUCTS.items()):
        if args.symbol and symbol != args.symbol:
            continue

        print(f"\n--- {i+1}/{len(PRODUCTS)} {symbol} ({name}) [{exchange}] ---")
        try:
            n = backfill_product(symbol, exchange, prefix, name, year_digits, args.start_date)
            total_new += n
            if n > 0:
                success += 1
            else:
                fail += 1
        except Exception as e:
            fail += 1
            print(f"  [{symbol}] ERROR: {str(e)[:80]}")

    print(f"\n=== 完成 === 成功:{success} 失败:{fail} 总新增:{total_new}天")
