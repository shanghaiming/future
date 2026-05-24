#!/usr/bin/env python3
"""
天勤量化期权数据采集脚本 v2
采集真实期权行情（全部合约），盘后计算IV/Greeks
逐品种独立进程，避免tqsdk连接断开
"""

import json
import os
import sys
import subprocess
import time
from datetime import datetime

import numpy as np
from scipy.stats import norm

from tqsdk import TqApi, TqAuth

TQ_ACCOUNT = '18844561230'
TQ_PASSWORD = 'zxcvbnm0717'

DATA_DIR = os.path.expanduser("~/home/futures_platform/data/tq_options")
os.makedirs(DATA_DIR, exist_ok=True)

# 所有期权品种
TARGET_PREFIXES = {
    'CZCE': ['AP', 'CF', 'CJ', 'FG', 'MA', 'OI', 'PF', 'PK', 'PL', 'PR', 'PX', 'RM', 'SA', 'SF', 'SH', 'SM', 'SR', 'TA', 'UR', 'ZC'],
    'DCE': ['a', 'b', 'bz', 'c', 'cs', 'eb', 'eg', 'i', 'jd', 'jm', 'l', 'lg', 'lh', 'm', 'p', 'pg', 'pp', 'v', 'y'],
    'GFEX': ['lc', 'pd', 'ps', 'pt', 'si'],
    'INE': ['bc', 'nr', 'sc'],
    'SHFE': ['ad', 'ag', 'al', 'ao', 'au', 'br', 'bu', 'cu', 'fu', 'ni', 'op', 'pb', 'rb', 'ru', 'sn', 'sp', 'zn'],
}


def bsm_call_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bsm_put_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(d1)


def calculate_iv(S, K, T, r, market_price, option_type='C'):
    if T <= 0 or market_price <= 0:
        return 0
    sigma_low, sigma_high = 0.001, 5.0
    for _ in range(100):
        sigma_mid = (sigma_low + sigma_high) / 2
        price = bsm_call_price(S, K, T, r, sigma_mid) if option_type == 'C' else bsm_put_price(S, K, T, r, sigma_mid)
        if abs(price - market_price) < 0.01:
            return sigma_mid
        if price > market_price:
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid
    return (sigma_low + sigma_high) / 2


def calculate_greeks(S, K, T, r, sigma, option_type='C'):
    if T <= 0 or sigma <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if option_type == 'C' else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    theta += (-r * K * np.exp(-r * T) * norm.cdf(d2) if option_type == 'C' else r * K * np.exp(-r * T) * norm.cdf(-d2))
    theta = theta / 365
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    rho = (K * T * np.exp(-r * T) * norm.cdf(d2) / 100 if option_type == 'C' else -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100)
    return {'delta': round(delta, 4), 'gamma': round(gamma, 6), 'theta': round(theta, 4), 'vega': round(vega, 4), 'rho': round(rho, 4)}


def fetch_product_options(exchange, prefix):
    """采集单个品种的全部期权合约（独立tqsdk连接，批量订阅）"""
    api = TqApi(auth=TqAuth(TQ_ACCOUNT, TQ_PASSWORD))

    try:
        all_quotes = api.query_quotes(ins_class='OPTION')
        filtered = [q for q in all_quotes if q.startswith(f'{exchange}.{prefix}')]

        if not filtered:
            print(f"EMPTY:{exchange}.{prefix}", file=sys.stderr)
            return []

        # 批量订阅所有合约报价
        quote_list = api.get_quote_list(filtered)

        # 等待数据到达
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                api.wait_update(deadline=time.time() + 5)
            except:
                break
            # 检查是否有有效数据
            valid_count = sum(1 for q in quote_list if q.last_price > 0 or q.volume > 0 or q.open_interest > 0)
            if valid_count > len(filtered) * 0.3:
                break

        # 再获取所有标的合约
        underlying_symbols = set()
        for q in quote_list:
            usym = getattr(q, 'underlying_symbol', None)
            if usym:
                underlying_symbols.add(usym)

        underlying_cache = {}
        if underlying_symbols:
            ulist = api.get_quote_list(list(underlying_symbols))
            deadline2 = time.time() + 10
            while time.time() < deadline2:
                try:
                    api.wait_update(deadline=time.time() + 2)
                except:
                    break
                if sum(1 for u in ulist if u.last_price > 0) > 0:
                    break
            for u in ulist:
                underlying_cache[u.instrument_id] = u.last_price if u.last_price > 0 else 0

        results = []
        for q in quote_list:
            try:
                sym = q.instrument_id
                underlying_sym = getattr(q, 'underlying_symbol', None)
                S = underlying_cache.get(underlying_sym, 0)

                K = q.strike_price
                expire_ts = q.expire_datetime
                now_ts = datetime.now().timestamp()
                T = max((expire_ts - now_ts) / (365 * 24 * 3600), 0.001)
                r = 0.02
                market_price = q.last_price
                option_class = getattr(q, 'option_class', 'CALL')
                option_type = 'C' if option_class == 'CALL' else 'P'

                iv = 0
                greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
                if S > 0 and K > 0 and market_price > 0 and T > 0.001:
                    iv = calculate_iv(S, K, T, r, market_price, option_type)
                    greeks = calculate_greeks(S, K, T, r, iv, option_type)

                data = {
                    'symbol': sym,
                    'instrument_name': q.instrument_name,
                    'exchange': exchange,
                    'product': prefix,
                    'option_class': option_class,
                    'strike_price': K,
                    'underlying_symbol': underlying_sym,
                    'underlying_price': S,
                    'last_price': market_price,
                    'bid_price1': q.bid_price1,
                    'ask_price1': q.ask_price1,
                    'bid_vol1': q.bid_volume1,
                    'ask_vol1': q.ask_volume1,
                    'volume': q.volume,
                    'open_interest': q.open_interest,
                    'settlement': getattr(q, 'settlement', 0),
                    'pre_settlement': getattr(q, 'pre_settlement', 0),
                    'implied_volatility': round(iv * 100, 2),
                    'delta': greeks['delta'],
                    'gamma': greeks['gamma'],
                    'theta': greeks['theta'],
                    'vega': greeks['vega'],
                    'rho': greeks['rho'],
                    'expire_datetime': expire_ts,
                }
                results.append(data)

            except Exception as e:
                continue

        print(f"OK:{exchange}.{prefix}:{len(results)}", file=sys.stderr)
        return results

    except Exception as e:
        print(f"FAIL:{exchange}.{prefix}:{str(e)[:80]}", file=sys.stderr)
        return []
    finally:
        try:
            api.close()
        except:
            pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="天勤期权数据采集 v2")
    parser.add_argument("mode", nargs="?", default="all", help="all | _fetch_one(内部)")
    parser.add_argument("--exchange", default=None)
    parser.add_argument("--prefix", default=None)
    args = parser.parse_args()

    if args.mode == "_fetch_one":
        # 单品种采集
        results = fetch_product_options(args.exchange, args.prefix)
        out_file = os.path.join(DATA_DIR, f"{args.exchange}_{args.prefix}_{datetime.now().strftime('%Y%m%d')}.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        sys.exit(0)

    # 主进程：逐品种spawn子进程
    print("=" * 60)
    print(f"天勤期权数据采集 v2 (逐品种独立连接)")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    today = datetime.now().strftime('%Y%m%d')
    success = fail = skip = 0
    total_contracts = 0

    for exchange, prefixes in TARGET_PREFIXES.items():
        for prefix in prefixes:
            out_file = os.path.join(DATA_DIR, f"{exchange}_{prefix}_{today}.json")

            cmd = [sys.executable, "-u", os.path.abspath(__file__),
                   "_fetch_one",
                   "--exchange", exchange,
                   "--prefix", prefix]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                output = (result.stderr or "").strip().split("\n")[-1]

                if output.startswith("OK:"):
                    parts = output.split(":")
                    n = int(parts[2]) if len(parts) > 2 else 0
                    success += 1
                    total_contracts += n
                    print(f"  {exchange}.{prefix}: {n}个合约")
                elif output.startswith("EMPTY:"):
                    skip += 1
                    print(f"  {exchange}.{prefix}: 无合约")
                else:
                    fail += 1
                    err = output if output else (result.stderr or "").strip().split("\n")[-1]
                    print(f"  {exchange}.{prefix}: FAIL {err[:60]}")

            except subprocess.TimeoutExpired:
                fail += 1
                print(f"  {exchange}.{prefix}: 超时")
            except Exception as e:
                fail += 1
                print(f"  {exchange}.{prefix}: 异常 {str(e)[:60]}")

            time.sleep(0.3)

    # 合并所有品种为一个总文件
    print()
    print(f"=== 完成 === 成功:{success} 跳过:{skip} 失败:{fail} 总合约:{total_contracts}")

    # 自动压缩7天前的品种JSON
    _compress_old_files(DATA_DIR, days=7)


def _compress_old_files(data_dir, days=7):
    """压缩N天前的品种JSON为gzip，节省空间"""
    import gzip
    from datetime import timedelta
    cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
    compressed = 0
    saved_mb = 0
    for fname in os.listdir(data_dir):
        if not fname.endswith('.json'):
            continue
        # 提取日期
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
