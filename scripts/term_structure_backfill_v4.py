#!/usr/bin/env python3
"""
期货期限结构历史回补 v4 (5年, tqsdk全品种)
用tqsdk拉各合约K线，逐日构建期限结构快照
逐品种spawn子进程，避免连接断开
"""

import os
import sys
import time
import json
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = os.path.expanduser("~/home/futures_platform")
DATA_DIR = os.path.join(BASE_DIR, "data")
TERM_DIR = os.path.join(DATA_DIR, "futures_term_structure")
os.makedirs(TERM_DIR, exist_ok=True)

TQ_ACCOUNT = '18844561230'
TQ_PASSWORD = 'zxcvbnm0717'

# 全商品品种: symbol -> (exchange, product_id, name)
PRODUCTS = {
    # === SHFE 黑色 ===
    'rbfi': ('SHFE', 'rb', '螺纹钢'),
    'hcfi': ('SHFE', 'hc', '热卷'),
    'ifi':  ('SHFE', 'i',  '铁矿石'),
    'jfi':  ('SHFE', 'j',  '焦炭'),
    'jmfi': ('SHFE', 'jm', '焦煤'),
    'sffi': ('SHFE', 'sf', '硅铁'),
    'smfi': ('SHFE', 'sm', '锰硅'),
    'wrfi': ('SHFE', 'wr', '线材'),
    # === SHFE 有色 ===
    'cufi': ('SHFE', 'cu', '沪铜'),
    'alfi': ('SHFE', 'al', '沪铝'),
    'znfi': ('SHFE', 'zn', '沪锌'),
    'pbfi': ('SHFE', 'pb', '沪铅'),
    'nifi': ('SHFE', 'ni', '沪镍'),
    'snfi': ('SHFE', 'sn', '沪锡'),
    'ssfi': ('SHFE', 'ss', '不锈钢'),
    'aofi': ('SHFE', 'ao', '氧化铝'),
    # === SHFE 贵金属 ===
    'aufi': ('SHFE', 'au', '黄金'),
    'agfi': ('SHFE', 'ag', '白银'),
    # === SHFE 能源化工 ===
    'rufi': ('SHFE', 'ru', '橡胶'),
    'bufi': ('SHFE', 'bu', '沥青'),
    'fufi': ('SHFE', 'fu', '燃油'),
    'spfi': ('SHFE', 'sp', '纸浆'),
    'brfi': ('SHFE', 'br', '丁二烯'),
    # === DCE 化工 ===
    'egfi': ('DCE', 'eg', '乙二醇'),
    'pgfi': ('DCE', 'pg', 'LPG'),
    'ebfi': ('DCE', 'eb', '苯乙烯'),
    'ppfi': ('DCE', 'pp', '聚丙烯'),
    'lfi':  ('DCE', 'l',  '塑料'),
    'vfi':  ('DCE', 'v',  'PVC'),
    # === DCE 农产品 ===
    'mfi':  ('DCE', 'm',  '豆粕'),
    'yfi':  ('DCE', 'y',  '豆油'),
    'pfi':  ('DCE', 'p',  '棕榈油'),
    'cfi':  ('DCE', 'c',  '玉米'),
    'csfi': ('DCE', 'cs', '淀粉'),
    'afi':  ('DCE', 'a',  '豆一'),
    'bfi':  ('DCE', 'b',  '豆二'),
    'jdfi': ('DCE', 'jd', '鸡蛋'),
    'lhfi': ('DCE', 'lh', '生猪'),
    'fbfi': ('DCE', 'fb', '纤维板'),
    'bbfi': ('DCE', 'bb', '胶合板'),
    # === DCE 黑色(实际在DCE) ===
    'ifi2':  ('DCE', 'i',  '铁矿石DCE'),
    'jfi2':  ('DCE', 'j',  '焦炭DCE'),
    'jmfi2': ('DCE', 'jm', '焦煤DCE'),
    # === CZCE ===
    'apfi': ('CZCE', 'AP', '苹果'),
    'cffi': ('CZCE', 'CF', '棉花'),
    'cjfi': ('CZCE', 'CJ', '红枣'),
    'cyfi': ('CZCE', 'CY', '棉纱'),
    'fgfi': ('CZCE', 'FG', '玻璃'),
    'mafi': ('CZCE', 'MA', '甲醇'),
    'oifi': ('CZCE', 'OI', '菜油'),
    'pffi': ('CZCE', 'PF', '短纤'),
    'pkfi': ('CZCE', 'PK', '花生'),
    'prfi': ('CZCE', 'PR', '瓶片'),
    'pxfi': ('CZCE', 'PX', 'PX'),
    'rmfi': ('CZCE', 'RM', '菜粕'),
    'rsfi': ('CZCE', 'RS', '菜籽'),
    'safi': ('CZCE', 'SA', '纯碱'),
    'sffi2':('CZCE', 'SF', '硅铁CZCE'),
    'shfi': ('CZCE', 'SH', '烧碱'),
    'smfi2':('CZCE', 'SM', '锰硅CZCE'),
    'srfi': ('CZCE', 'SR', '白糖'),
    'tafi': ('CZCE', 'TA', 'PTA'),
    'urfi': ('CZCE', 'UR', '尿素'),
    'whfi': ('CZCE', 'WH', '强麦'),
    'zcfi': ('CZCE', 'ZC', '动力煤'),
    'jrfi': ('CZCE', 'JR', '粳稻'),
    'lrfi': ('CZCE', 'LR', '晚稻'),
    'pmfi': ('CZCE', 'PM', '普麦'),
    'rrfi': ('CZCE', 'RR', '粳米'),
    # === INE ===
    'scfi': ('INE', 'sc', '原油'),
    'nrfi': ('INE', 'nr', '20号胶'),
    'bcfi': ('INE', 'bc', '国际铜'),
    'lufi': ('INE', 'lu', '低硫燃油'),
    # === GFEX ===
    'sifi': ('GFEX', 'si', '工业硅'),
    'lcfi': ('GFEX', 'lc', '碳酸锂'),
    'pdfi': ('GFEX', 'pd', '钯'),
    'psfi': ('GFEX', 'ps', '镁'),
    'ptfi': ('GFEX', 'pt', '铂'),
}


def parse_czce_contract(local_sym, product_id):
    """解析CZCE合约年月: AP410->2024年10月, AP501->2025年1月"""
    num_part = local_sym[len(product_id):]
    try:
        if len(num_part) == 3:
            yy = int(num_part[0]) + 2020
            mm = int(num_part[1:3])
        elif len(num_part) == 4:
            yy = int(num_part[:2]) + 2000
            mm = int(num_part[2:])
        else:
            return None, None
        if 1 <= mm <= 12 and 2020 <= yy <= 2030:
            return yy, mm
    except:
        pass
    return None, None


def parse_standard_contract(local_sym, product_id):
    """解析标准合约年月: rb2609->2026年9月"""
    num_part = local_sym[len(product_id):]
    try:
        if len(num_part) == 4:
            yy = int(num_part[:2]) + 2000
            mm = int(num_part[2:])
            if 1 <= mm <= 12 and 2020 <= yy <= 2030:
                return yy, mm
    except:
        pass
    return None, None


def fetch_product_term_structure(exchange, product_id, symbol, name, start_date='2021-01-01'):
    """用tqsdk拉单个品种全部合约K线，构建期限结构"""
    from tqsdk import TqApi, TqAuth

    existing_dates = set()
    for f in os.listdir(TERM_DIR):
        if f.startswith(symbol + '_') and f.endswith('.json'):
            existing_dates.add(f.replace(symbol + '_', '').replace('.json', ''))

    api = TqApi(auth=TqAuth(TQ_ACCOUNT, TQ_PASSWORD))
    try:
        # 查找该品种所有期货合约
        all_quotes = api.query_quotes(ins_class='FUTURE', exchange_id=exchange, product_id=product_id)
        if not all_quotes:
            print(f"EMPTY:{symbol}", file=sys.stderr)
            return 0

        # 过滤：只保留5年内的合约(2021+)
        filtered = []
        for q in all_quotes:
            local = q.split('.')[-1] if '.' in q else q
            num = local[len(product_id):]
            if exchange == 'CZCE':
                if len(num) == 3:
                    yy = int(num[0]) + 2020
                elif len(num) == 4:
                    yy = int(num[:2]) + 2000
                else:
                    continue
            else:
                if len(num) >= 4:
                    yy = int(num[:2]) + 2000
                else:
                    continue
            if yy >= 2021:
                filtered.append(q)

        if not filtered:
            print(f"EMPTY:{symbol}", file=sys.stderr)
            return 0

        print(f"  [{symbol}] {len(filtered)}个合约", file=sys.stderr)

        # 批量拉K线，每批8个
        all_klines = {}
        batch_size = 8
        for batch_start in range(0, len(filtered), batch_size):
            batch = filtered[batch_start:batch_start + batch_size]
            kline_objs = {}
            for sym in batch:
                try:
                    kl = api.get_kline_serial(sym, 86400, 1200)
                    kline_objs[sym] = kl
                except:
                    pass

            # 等待数据到达
            deadline = time.time() + 30
            while time.time() < deadline:
                try:
                    api.wait_update(deadline=time.time() + 5)
                except:
                    break
                valid_count = sum(1 for kl in kline_objs.values() if len(kl[kl['close'] > 0]) > 10)
                if valid_count >= len(kline_objs) * 0.5:
                    break

            for sym, kl in kline_objs.items():
                valid = kl[kl['close'] > 0]
                if len(valid) > 0:
                    all_klines[sym] = valid

            print(f"  [{symbol}] {min(batch_start+batch_size, len(filtered))}/{len(filtered)} 有数据:{len(all_klines)}", file=sys.stderr)

        if not all_klines:
            print(f"NODATA:{symbol}", file=sys.stderr)
            return 0

        # 按日期汇总合约价格
        start_str = start_date.replace('-', '')
        contracts_by_date = {}

        for sym, df in all_klines.items():
            parts = sym.split('.')
            if len(parts) < 2:
                continue
            local_sym = parts[1]

            if exchange == 'CZCE':
                yy, mm = parse_czce_contract(local_sym, product_id)
            else:
                yy, mm = parse_standard_contract(local_sym, product_id)
            if yy is None:
                continue

            for _, row in df.iterrows():
                ts = row['datetime']
                if pd.isna(ts) or ts == 0:
                    continue
                try:
                    dt = datetime.fromtimestamp(ts / 1e9)
                    d = dt.strftime('%Y%m%d')
                except:
                    continue
                if d < start_str:
                    continue
                price = float(row['close']) if pd.notna(row['close']) else 0
                vol = int(row['volume']) if pd.notna(row.get('volume', 0)) else 0
                if price > 0:
                    if d not in contracts_by_date:
                        contracts_by_date[d] = []
                    contracts_by_date[d].append({
                        'symbol': local_sym.upper(), 'name': f'{name}{yy-2000:02d}{mm:02d}',
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

        print(f"OK:{symbol}:{new_count}", file=sys.stderr)
        return new_count

    except Exception as e:
        print(f"FAIL:{symbol}:{str(e)[:80]}", file=sys.stderr)
        return 0
    finally:
        try:
            api.close()
        except:
            pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="期限结构历史回补 v4 (tqsdk, 5年)")
    parser.add_argument("mode", nargs="?", default="all", help="all | _fetch_one(内部)")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--exchange", default=None)
    parser.add_argument("--product-id", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--start-date", default="2021-01-01")
    args = parser.parse_args()

    if args.mode == "_fetch_one":
        fetch_product_term_structure(args.exchange, args.product_id, args.symbol, args.name or args.symbol, args.start_date)
        sys.exit(0)

    print("=" * 60)
    print(f"期货期限结构历史回补 v4 (tqsdk, 5年)")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"起始日: {args.start_date}")
    print(f"品种数: {len(PRODUCTS)}")
    print("=" * 60)

    success = fail = skip = 0
    total_new = 0

    for i, (symbol, (exchange, product_id, name)) in enumerate(PRODUCTS.items()):
        if args.symbol and symbol != args.symbol:
            continue

        print(f"\n--- {i+1}/{len(PRODUCTS)} {symbol} ({name}) [{exchange}] ---")

        cmd = [sys.executable, "-u", os.path.abspath(__file__),
               "_fetch_one",
               "--symbol", symbol,
               "--exchange", exchange,
               "--product-id", product_id,
               "--name", name,
               "--start-date", args.start_date]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            output = (result.stderr or "").strip().split("\n")
            status_line = ""
            for line in reversed(output):
                if line.startswith(("OK:", "FAIL:", "NODATA:", "EMPTY:")):
                    status_line = line
                    break

            if status_line.startswith("OK:"):
                parts = status_line.split(":")
                n = int(parts[2]) if len(parts) > 2 else 0
                success += 1
                total_new += n
                print(f"  {symbol}: +{n}天")
            elif status_line.startswith("NODATA:") or status_line.startswith("EMPTY:"):
                skip += 1
                print(f"  {symbol}: 无数据")
            else:
                fail += 1
                err = status_line or "unknown"
                print(f"  {symbol}: FAIL {err[:60]}")

        except subprocess.TimeoutExpired:
            fail += 1
            print(f"  {symbol}: 超时")
        except Exception as e:
            fail += 1
            print(f"  {symbol}: 异常 {str(e)[:60]}")

        time.sleep(0.5)

    print(f"\n=== 完成 === 成功:{success} 跳过:{skip} 失败:{fail} 总新增:{total_new}天")
