#!/usr/bin/env python3
"""
期货期限结构数据采集 v2
- 从akshare新浪接口获取多合约实时行情
- 支持所有80个品种
- 保存到 ~/home/futures_platform/data/futures_term_structure/
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

# 品种名称映射（akshare用中文名）
SYMBOL_NAME_MAP = {
    'cufi': '沪铜', 'alfi': '沪铝', 'znfi': '沪锌', 'pbfi': '沪铅',
    'nifi': '沪镍', 'snfi': '沪锡', 'aufi': '沪金', 'agfi': '沪银',
    'rbfi': '螺纹', 'hcfi': '热卷', 'ifi': '铁矿', 'jfi': '焦炭',
    'jmfi': '焦煤', 'fgfi': '玻璃', 'safi': '纯碱', 'cufi': '沪铜',
    'scfi': '原油', 'lufi': '低硫燃油', 'fufi': '燃油', 'pgfi': 'LPG',
    'bifi': '沥青', 'mfi': '豆粕', 'ofi': '豆油', 'yfi': '棕榈',
    'pfi': '豆油', 'rmfi': '菜粕', 'oi': '菜油', 'cfi': '棉花',
    'srfi': '白糖', 'cyfi': '棉纱', 'apfi': '苹果', 'cjfi': '红枣',
    'jd': '鸡蛋', 'lhfi': '生猪', 'csfi': '玉米淀粉', 'mfi': '豆粕',
    'rrfi': '粳米', 'whfi': '强麦', 'pmfi': '普麦', 'lrfi': '晚稻',
    'jr': '粳稻', 'ri': '早稻', 'rsfi': '菜籽', 'aofi': '豆一',
    'befi': '豆二', 'cufi': '沪铜', 'znfi': '沪锌', 'al': '沪铝',
    'pb': '沪铅', 'ni': '沪镍', 'sn': '沪锡', 'au': '沪金',
    'ag': '沪银', 'rb': '螺纹', 'hc': '热卷', 'i': '铁矿',
    'j': '焦炭', 'jm': '焦煤', 'fg': '玻璃', 'sa': '纯碱',
    'sc': '原油', 'lu': '低硫燃油', 'fu': '燃油', 'pg': 'LPG',
    'bu': '沥青', 'm': '豆粕', 'y': '棕榈', 'p': '豆油',
    'rm': '菜粕', 'oi': '菜油', 'cf': '棉花', 'sr': '白糖',
    'cy': '棉纱', 'ap': '苹果', 'cj': '红枣', 'jd': '鸡蛋',
    'lh': '生猪', 'cs': '玉米淀粉', 'rr': '粳米', 'wh': '强麦',
    'pm': '普麦', 'lr': '晚稻', 'rs': '菜籽', 'a': '豆一',
    'b': '豆二', 'eb': '苯乙烯', 'eg': '乙二醇', 'ma': '甲醇',
    'ta': 'PTA', 'pf': '短纤', 'pg': 'LPG', 'pp': '聚丙烯',
    'l': '塑料', 'v': 'PVC', 'eg': '乙二醇', 'eb': '苯乙烯',
    'ur': '尿素', 'sp': '纸浆', 'sc': '原油', 'nr': '20号胶',
    'br': '丁二烯橡胶', 'ru': '天然橡胶', 'cu': '沪铜', 'al': '沪铝',
    'zn': '沪锌', 'pb': '沪铅', 'ni': '沪镍', 'sn': '沪锡',
    'au': '沪金', 'ag': '沪银', 'rb': '螺纹', 'hc': '热卷',
    'i': '铁矿', 'j': '焦炭', 'jm': '焦煤', 'fg': '玻璃',
    'sa': '纯碱', 'sc': '原油', 'lu': '低硫燃油', 'fu': '燃油',
    'pg': 'LPG', 'bu': '沥青', 'm': '豆粕', 'y': '棕榈',
    'p': '豆油', 'rm': '菜粕', 'oi': '菜油', 'cf': '棉花',
    'sr': '白糖', 'cy': '棉纱', 'ap': '苹果', 'cj': '红枣',
    'jd': '鸡蛋', 'lh': '生猪', 'cs': '玉米淀粉', 'rr': '粳米',
    'wh': '强麦', 'pm': '普麦', 'lr': '晚稻', 'rs': '菜籽',
    'a': '豆一', 'b': '豆二', 'eb': '苯乙烯', 'eg': '乙二醇',
    'ma': '甲醇', 'ta': 'PTA', 'pf': '短纤', 'pp': '聚丙烯',
    'l': '塑料', 'v': 'PVC', 'ur': '尿素', 'sp': '纸浆',
    'nr': '20号胶', 'br': '丁二烯橡胶', 'ru': '天然橡胶',
    'ec': '集运指数', 'lc': '碳酸锂', 'si': '工业硅',
    'sh': '烧碱', 'px': '对二甲苯', 'pr': '瓶片',
    'br': '丁二烯橡胶', 'bc': '国际铜', 'ao': '氧化铝',
    'lc': '碳酸锂', 'si': '工业硅', 'ec': '集运指数',
    'sh': '烧碱', 'px': '对二甲苯', 'pr': '瓶片',
    'bb': '胶合板', 'fb': '纤维板', 'wr': '线材',
    'ss': '不锈钢', 'bc': '国际铜', 'ao': '氧化铝',
}

# 去重并建立反向映射
UNIQUE_NAMES = {}
for code, name in SYMBOL_NAME_MAP.items():
    if name not in UNIQUE_NAMES.values():
        UNIQUE_NAMES[code] = name

import akshare as ak

def fetch_term_structure(symbol, name):
    """获取单个品种的期限结构"""
    try:
        df = ak.futures_zh_realtime(symbol=name)
        if df.empty:
            return None
        
        # 过滤掉连续合约，只保留具体月份合约
        contracts = []
        for _, row in df.iterrows():
            sym = str(row['symbol'])
            # 跳过连续合约（以0结尾）
            if sym.endswith('0'):
                continue
            # 提取合约月份
            if len(sym) >= 4:
                month_code = sym[-4:]  # 如 2606
                try:
                    year = int(month_code[:2])
                    month = int(month_code[2:])
                    contracts.append({
                        'symbol': sym,
                        'name': row['name'],
                        'price': float(row['trade']) if pd.notna(row['trade']) else 0,
                        'year': year,
                        'month': month,
                        'volume': int(row['volume']) if pd.notna(row['volume']) else 0,
                    })
                except:
                    continue
        
        if len(contracts) < 2:
            return None
        
        # 按年月排序
        contracts.sort(key=lambda x: (x['year'], x['month']))
        
        # 取前4个合约
        curve = contracts[:4]
        
        near = curve[0]
        far = curve[-1]
        
        spread = far['price'] - near['price']
        spread_pct = (spread / near['price'] * 100) if near['price'] > 0 else 0
        
        return {
            'symbol': symbol,
            'name': name,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'structure': 'contango' if spread > 0 else 'backwardation',
            'curve': curve,
            'near_contract': near['symbol'],
            'near_price': near['price'],
            'far_contract': far['symbol'],
            'far_price': far['price'],
            'total_spread': spread,
            'total_spread_pct': spread_pct,
        }
    except Exception as e:
        print(f"  {symbol}({name}): 错误 - {str(e)[:60]}")
        return None


def main():
    # 获取所有期货品种
    futures_dir = os.path.join(DATA_DIR, 'futures_weighted')
    symbols = sorted([f.replace('.csv', '') for f in os.listdir(futures_dir) if f.endswith('.csv')])
    
    print(f"开始采集 {len(symbols)} 个品种的期限结构...")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    success = 0
    fail = 0
    
    for symbol in symbols:
        # 查找对应的中文名
        name = None
        for code, cn in UNIQUE_NAMES.items():
            if code == symbol or code + 'fi' == symbol or symbol.startswith(code):
                name = cn
                break
        
        if not name:
            # 尝试直接用小写代码
            base = symbol.replace('fi', '')
            if base in UNIQUE_NAMES:
                name = UNIQUE_NAMES[base]
            else:
                print(f"  {symbol}: 未找到映射，跳过")
                fail += 1
                continue
        
        print(f"采集 {symbol} ({name})...", end=' ')
        result = fetch_term_structure(symbol, name)
        
        if result:
            filepath = os.path.join(TERM_DIR, f"{symbol}_{result['date'].replace('-', '')}.json")
            with open(filepath, 'w') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✓ {result['structure']} {result['total_spread_pct']:+.2f}%")
            success += 1
        else:
            print("✗ 无数据")
            fail += 1
        
        time.sleep(0.5)  # 限速
    
    print("-" * 60)
    print(f"完成: 成功 {success} 个, 失败 {fail} 个")


if __name__ == '__main__':
    main()
