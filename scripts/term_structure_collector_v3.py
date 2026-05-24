#!/usr/bin/env python3
"""
期货期限结构数据采集 v3
- 用akshare实时接口获取多合约数据（支持大部分品种）
- 用futures_zh_daily_sina获取历史数据（支持新品种）
- 遍历所有80个品种
"""

import os
for k in list(os.environ.keys()):
    if 'proxy' in k.lower():
        del os.environ[k]

import sys
import time
import json
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.expanduser("~/home/futures_platform")
DATA_DIR = os.path.join(BASE_DIR, "data")
TERM_DIR = os.path.join(DATA_DIR, "futures_term_structure")
os.makedirs(TERM_DIR, exist_ok=True)

import akshare as ak

# 品种名称映射（akshare实时接口用）
NAME_MAP = {
    'agfi': '白银', 'alfi': '沪铝', 'aofi': '豆一', 'apfi': '鲜苹果',
    'aufi': '黄金', 'bbfi': '胶合板', 'bcfi': '国际铜', 'bfi': '豆二',
    'brfi': '丁二烯橡胶', 'bufi': '沥青', 'bzfi': '苯乙烯', 'cffi': '棉纱',
    'cfi': '棉花', 'cjfi': '红枣', 'csfi': '玉米淀粉', 'cyfi': '棉纱',
    'ebfi': '苯乙烯', 'egfi': '乙二醇', 'fbfi': '纤维板', 'fgfi': '玻璃',
    'fufi': '燃油', 'hcfi': '热轧卷板', 'ifi': '铁矿石', 'jdfi': '鸡蛋',
    'jfi': '焦炭', 'jmfi': '焦煤', 'lcfi': '碳酸锂', 'lfi': '塑料',
    'lgfi': '硅铁', 'lhfi': '生猪', 'lufi': '低硫燃料油', 
    'mfi': '豆粕', 'nifi': '沪镍', 'nrfi': '20号胶', 'oifi': '菜油',
    'pbfi': '沪铅', 'pffi': '短纤', 'pfi': '棕榈', 'pgfi': '液化石油气',
    'pkfi': '花生', 'ptfi': 'PTA', 'rbfi': '螺纹钢', 'rifi': '早籼稻',
    'rmfi': '菜粕', 'rrfi': '粳米', 'rsfi': '菜籽', 'rufi': '橡胶',
    'safi': '纯碱', 'scfi': '原油', 'shfi': '烧碱', 'sifi': '工业硅',
    'smfi': '锰硅', 'snfi': '沪锡', 'spfi': '纸浆', 'srfi': '白糖',
    'ssfi': '不锈钢', 'tafi': 'PTA', 'urfi': '尿素', 'vfi': 'PVC',
    'whfi': '强麦', 'wrfi': '线材', 'yfi': '豆油', 'znfi': '沪锌',
    'cufi': '沪铜', 'pfi': '棕榈油', 'afi': '豆一', 'jrfi': '粳稻',
    'lrfi': '晚籼稻', 'sffi': '硅铁',
}

# 需要用历史接口的品种（新浪代码）
HISTORY_SYMBOLS = {
    'ecfi': 'EC', 'pdfi': 'PR', 'prfi': 'PR', 'psfi': 'PS', 'pxfi': 'PX', 
    'adfi': 'AD', 'mafi': 'MA', 'opfi': 'OP', 'zcfi': 'ZC',
}


def fetch_via_realtime(symbol, name):
    """用实时接口获取期限结构"""
    try:
        df = ak.futures_zh_realtime(symbol=name)
        if df.empty:
            return None
        
        contracts = []
        for _, row in df.iterrows():
            sym = str(row['symbol'])
            if sym.endswith('0'):
                continue
            if len(sym) >= 4:
                month_code = sym[-4:]
                try:
                    year = int(month_code[:2])
                    month = int(month_code[2:])
                    price = float(row['trade']) if pd.notna(row['trade']) else 0
                    if price > 0:
                        contracts.append({
                            'symbol': sym, 'name': row['name'],
                            'price': price, 'year': year, 'month': month,
                            'volume': int(row['volume']) if pd.notna(row['volume']) else 0,
                        })
                except:
                    continue
        
        if len(contracts) < 2:
            return None
        
        contracts.sort(key=lambda x: (x['year'], x['month']))
        curve = contracts[:4]
        near, far = curve[0], curve[-1]
        spread = far['price'] - near['price']
        spread_pct = (spread / near['price'] * 100) if near['price'] > 0 else 0
        
        return {
            'symbol': symbol, 'name': name,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'structure': 'contango' if spread > 0 else 'backwardation',
            'curve': curve,
            'near_contract': near['symbol'], 'near_price': near['price'],
            'far_contract': far['symbol'], 'far_price': far['price'],
            'total_spread': spread, 'total_spread_pct': spread_pct,
        }
    except Exception as e:
        return None


def fetch_via_history(symbol, base_code):
    """用历史接口获取期限结构（新品种）"""
    try:
        contracts = []
        # 尝试获取近月合约（25年6-12月，26年1-12月）
        for year in [25, 26]:
            for month in range(1, 13):
                code = f"{base_code}{year}{month:02d}"
                try:
                    df = ak.futures_zh_daily_sina(symbol=code)
                    if len(df) > 0:
                        latest = df.iloc[-1]
                        contracts.append({
                            'symbol': code, 'name': code,
                            'price': float(latest['close']),
                            'year': year, 'month': month,
                            'volume': 0,
                        })
                except:
                    continue
        
        if len(contracts) < 2:
            return None
        
        contracts.sort(key=lambda x: (x['year'], x['month']))
        curve = contracts[:4]
        near, far = curve[0], curve[-1]
        spread = far['price'] - near['price']
        spread_pct = (spread / near['price'] * 100) if near['price'] > 0 else 0
        
        return {
            'symbol': symbol, 'name': base_code,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'structure': 'contango' if spread > 0 else 'backwardation',
            'curve': curve,
            'near_contract': near['symbol'], 'near_price': near['price'],
            'far_contract': far['symbol'], 'far_price': far['price'],
            'total_spread': spread, 'total_spread_pct': spread_pct,
        }
    except Exception as e:
        return None


def main():
    futures_dir = os.path.join(DATA_DIR, 'futures_weighted')
    symbols = sorted([f.replace('.csv', '') for f in os.listdir(futures_dir) if f.endswith('.csv')])
    
    print(f"开始采集 {len(symbols)} 个品种的期限结构...")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    success = 0
    fail = 0
    
    for symbol in symbols:
        if symbol in HISTORY_SYMBOLS:
            # 用历史接口
            print(f"采集 {symbol} (历史接口)...", end=' ')
            result = fetch_via_history(symbol, HISTORY_SYMBOLS[symbol])
            method = "历史"
        else:
            # 用实时接口
            name = NAME_MAP.get(symbol)
            if not name:
                print(f"  {symbol}: 未找到映射，跳过")
                fail += 1
                continue
            print(f"采集 {symbol} ({name})...", end=' ')
            result = fetch_via_realtime(symbol, name)
            method = "实时"
        
        if result:
            filepath = os.path.join(TERM_DIR, f"{symbol}_{result['date'].replace('-', '')}.json")
            with open(filepath, 'w') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✓ {result['structure']} {result['total_spread_pct']:+.2f}% ({method})")
            success += 1
        else:
            print("✗ 无数据")
            fail += 1
        
        time.sleep(0.3)
    
    print("-" * 60)
    print(f"完成: 成功 {success} 个, 失败 {fail} 个")


if __name__ == '__main__':
    main()
