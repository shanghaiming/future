#!/usr/bin/env python3
"""
期货期限结构每日快照 (tqsdk)
用get_quote批量获取当前价格，只存当日，速度快
逐品种独立进程
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime

BASE_DIR = os.path.expanduser("~/home/futures_platform")
TERM_DIR = os.path.join(BASE_DIR, "data", "futures_term_structure")
os.makedirs(TERM_DIR, exist_ok=True)

TQ_ACCOUNT = '18844561230'
TQ_PASSWORD = 'zxcvbnm0717'

PRODUCTS = {
    'rbfi': ('SHFE', 'rb', '螺纹钢'), 'hcfi': ('SHFE', 'hc', '热卷'),
    'ifi':  ('SHFE', 'i',  '铁矿石'), 'jfi':  ('SHFE', 'j',  '焦炭'),
    'jmfi': ('SHFE', 'jm', '焦煤'),   'sffi': ('SHFE', 'sf', '硅铁'),
    'smfi': ('SHFE', 'sm', '锰硅'),   'wrfi': ('SHFE', 'wr', '线材'),
    'cufi': ('SHFE', 'cu', '沪铜'),   'alfi': ('SHFE', 'al', '沪铝'),
    'znfi': ('SHFE', 'zn', '沪锌'),   'pbfi': ('SHFE', 'pb', '沪铅'),
    'nifi': ('SHFE', 'ni', '沪镍'),   'snfi': ('SHFE', 'sn', '沪锡'),
    'ssfi': ('SHFE', 'ss', '不锈钢'), 'aofi': ('SHFE', 'ao', '氧化铝'),
    'aufi': ('SHFE', 'au', '黄金'),   'agfi': ('SHFE', 'ag', '白银'),
    'rufi': ('SHFE', 'ru', '橡胶'),   'bufi': ('SHFE', 'bu', '沥青'),
    'fufi': ('SHFE', 'fu', '燃油'),   'spfi': ('SHFE', 'sp', '纸浆'),
    'brfi': ('SHFE', 'br', '丁二烯'),
    'egfi': ('DCE', 'eg', '乙二醇'),  'pgfi': ('DCE', 'pg', 'LPG'),
    'ebfi': ('DCE', 'eb', '苯乙烯'),  'ppfi': ('DCE', 'pp', '聚丙烯'),
    'lfi':  ('DCE', 'l',  '塑料'),    'vfi':  ('DCE', 'v',  'PVC'),
    'mfi':  ('DCE', 'm',  '豆粕'),    'yfi':  ('DCE', 'y',  '豆油'),
    'pfi':  ('DCE', 'p',  '棕榈油'),  'cfi':  ('DCE', 'c',  '玉米'),
    'csfi': ('DCE', 'cs', '淀粉'),    'afi':  ('DCE', 'a',  '豆一'),
    'bfi':  ('DCE', 'b',  '豆二'),    'jdfi': ('DCE', 'jd', '鸡蛋'),
    'lhfi': ('DCE', 'lh', '生猪'),    'fbfi': ('DCE', 'fb', '纤维板'),
    'bbfi': ('DCE', 'bb', '胶合板'),
    'apfi': ('CZCE', 'AP', '苹果'),   'cffi': ('CZCE', 'CF', '棉花'),
    'cjfi': ('CZCE', 'CJ', '红枣'),   'cyfi': ('CZCE', 'CY', '棉纱'),
    'fgfi': ('CZCE', 'FG', '玻璃'),   'mafi': ('CZCE', 'MA', '甲醇'),
    'oifi': ('CZCE', 'OI', '菜油'),   'pffi': ('CZCE', 'PF', '短纤'),
    'pkfi': ('CZCE', 'PK', '花生'),   'prfi': ('CZCE', 'PR', '瓶片'),
    'pxfi': ('CZCE', 'PX', 'PX'),     'rmfi': ('CZCE', 'RM', '菜粕'),
    'rsfi': ('CZCE', 'RS', '菜籽'),   'safi': ('CZCE', 'SA', '纯碱'),
    'sffi2':('CZCE', 'SF', '硅铁CZCE'), 'shfi': ('CZCE', 'SH', '烧碱'),
    'smfi2':('CZCE', 'SM', '锰硅CZCE'), 'srfi': ('CZCE', 'SR', '白糖'),
    'tafi': ('CZCE', 'TA', 'PTA'),    'urfi': ('CZCE', 'UR', '尿素'),
    'whfi': ('CZCE', 'WH', '强麦'),   'zcfi': ('CZCE', 'ZC', '动力煤'),
    'jrfi': ('CZCE', 'JR', '粳稻'),   'lrfi': ('CZCE', 'LR', '晚稻'),
    'pmfi': ('CZCE', 'PM', '普麦'),   'rrfi': ('CZCE', 'RR', '粳米'),
    'scfi': ('INE', 'sc', '原油'),    'nrfi': ('INE', 'nr', '20号胶'),
    'bcfi': ('INE', 'bc', '国际铜'),  'lufi': ('INE', 'lu', '低硫燃油'),
    'sifi': ('GFEX', 'si', '工业硅'), 'lcfi': ('GFEX', 'lc', '碳酸锂'),
    'pdfi': ('GFEX', 'pd', '钯'),     'psfi': ('GFEX', 'ps', '镁'),
    'ptfi': ('GFEX', 'pt', '铂'),
}


def fetch_daily_snapshot(exchange, product_id, symbol, name):
    """用tqsdk获取当日快照"""
    from tqsdk import TqApi, TqAuth

    today = datetime.now().strftime('%Y%m%d')
    today_fmt = datetime.now().strftime('%Y-%m-%d')
    out_path = os.path.join(TERM_DIR, f"{symbol}_{today}.json")
    if os.path.exists(out_path):
        print(f"SKIP:{symbol}", file=sys.stderr)
        return True

    api = TqApi(auth=TqAuth(TQ_ACCOUNT, TQ_PASSWORD))
    try:
        all_quotes = api.query_quotes(ins_class='FUTURE', exchange_id=exchange, product_id=product_id)
        if not all_quotes:
            print(f"EMPTY:{symbol}", file=sys.stderr)
            return False

        # 批量订阅行情
        quote_list = api.get_quote_list(all_quotes)
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                api.wait_update(deadline=time.time() + 5)
            except:
                break
            # 检查是否有有效数据
            valid = 0
            for q in quote_list:
                if q.last_price > 0 and q.last_price != float('nan'):
                    valid += 1
            if valid > len(all_quotes) * 0.3:
                break

        # 构建期限结构
        day_contracts = []
        for q in quote_list:
            if q.last_price <= 0 or q.last_price != q.last_price:  # NaN check
                continue
            local_sym = str(q.underlying_symbol).split('.')[-1] if hasattr(q, 'underlying_symbol') else ''
            sym_str = all_quotes[quote_list.index(q)] if quote_list.index(q) < len(all_quotes) else ''
            # 从合约代码解析年月
            parts = sym_str.split('.')
            local = parts[1] if len(parts) > 1 else sym_str
            num = local[len(product_id):]
            yy = mm = None
            try:
                if exchange == 'CZCE':
                    if len(num) == 3:
                        yy = int(num[0]) + 2020
                        mm = int(num[1:3])
                    elif len(num) == 4:
                        yy = int(num[:2]) + 2000
                        mm = int(num[2:])
                else:
                    if len(num) >= 4:
                        yy = int(num[:2]) + 2000
                        mm = int(num[2:4])
            except:
                continue
            if yy is None or mm is None or not (1 <= mm <= 12):
                continue

            day_contracts.append({
                'symbol': local.upper(), 'name': f'{name}{yy-2000:02d}{mm:02d}',
                'price': float(q.last_price), 'year': yy, 'month': mm,
                'volume': int(q.volume) if hasattr(q, 'volume') and q.volume == q.volume else 0,
                'open_interest': int(q.open_interest) if hasattr(q, 'open_interest') and q.open_interest == q.open_interest else 0,
            })

        if len(day_contracts) < 2:
            print(f"NODATA:{symbol}", file=sys.stderr)
            return False

        day_contracts.sort(key=lambda x: (x['year'], x['month']))
        curve = day_contracts[:4]
        near, far = curve[0], curve[-1]
        spread = far['price'] - near['price']
        spread_pct = (spread / near['price'] * 100) if near['price'] > 0 else 0

        record = {
            'symbol': symbol, 'name': name,
            'date': today_fmt,
            'structure': 'contango' if spread > 0 else 'backwardation',
            'curve': curve,
            'near_contract': near['symbol'], 'near_price': near['price'],
            'far_contract': far['symbol'], 'far_price': far['price'],
            'total_spread': spread, 'total_spread_pct': round(spread_pct, 4),
        }

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print(f"OK:{symbol}:{len(day_contracts)}", file=sys.stderr)
        return True

    except Exception as e:
        print(f"FAIL:{symbol}:{str(e)[:80]}", file=sys.stderr)
        return False
    finally:
        try:
            api.close()
        except:
            pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="all")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--exchange", default=None)
    parser.add_argument("--product-id", default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    if args.mode == "_fetch_one":
        fetch_daily_snapshot(args.exchange, args.product_id, args.symbol, args.name or args.symbol)
        sys.exit(0)

    print(f"期限结构每日快照 {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    success = fail = skip = 0
    for i, (symbol, (exchange, product_id, name)) in enumerate(PRODUCTS.items()):
        if args.symbol and symbol != args.symbol:
            continue

        cmd = [sys.executable, "-u", os.path.abspath(__file__),
               "_fetch_one", "--symbol", symbol,
               "--exchange", exchange, "--product-id", product_id,
               "--name", name]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            output = (result.stderr or "").strip().split("\n")
            status = ""
            for line in reversed(output):
                if line.startswith(("OK:", "FAIL:", "NODATA:", "EMPTY:", "SKIP:")):
                    status = line
                    break

            if status.startswith("OK:"):
                success += 1
                print(f"  {symbol}: OK")
            elif status.startswith("SKIP:"):
                skip += 1
            elif status.startswith("NODATA:") or status.startswith("EMPTY:"):
                skip += 1
                print(f"  {symbol}: 无数据")
            else:
                fail += 1
                print(f"  {symbol}: FAIL")
        except subprocess.TimeoutExpired:
            fail += 1
            print(f"  {symbol}: 超时")

        time.sleep(0.3)

    print(f"完成 成功:{success} 跳过:{skip} 失败:{fail}")

    # 自动压缩30天前的JSON
    _compress_old_files(TS_DIR, days=30)


def _compress_old_files(data_dir, days=30):
    """压缩N天前的JSON为gzip，节省空间"""
    import gzip
    from datetime import timedelta
    cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
    compressed = 0
    saved_mb = 0
    for fname in os.listdir(data_dir):
        if not fname.endswith('.json'):
            continue
        parts = fname.rsplit('_', 1)
        if len(parts) != 2:
            continue
        date_str = parts[1].replace('.json', '')
        if date_str >= cutoff:
            continue
        gz_path = os.path.join(data_dir, fname + '.gz')
        if os.path.exists(gz_path):
            continue
        src = os.path.join(data_dir, fname)
        try:
            with open(src, 'rb') as f_in:
                with gzip.open(gz_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            orig_size = os.path.getsize(src)
            gz_size = os.path.getsize(gz_path)
            os.remove(src)
            saved_mb += (orig_size - gz_size) / 1024 / 1024
            compressed += 1
        except:
            pass
    if compressed > 0:
        print(f"压缩旧文件: {compressed}个, 节省{saved_mb:.1f}MB")
