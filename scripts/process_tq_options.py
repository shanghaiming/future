#!/usr/bin/env python3
"""
处理TQ期权原始数据，转换为看板可用的格式
- 使用交易所提供的IV/Greeks
- 计算moneyness、days_to_expiry、HV等辅助指标
- 输出到 data/options_calculated/
"""
import os, json, glob, math, time, calendar
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from collections import defaultdict

BASE_DIR = os.path.expanduser("~/home/futures_platform")
TQ_OPT_DIR = os.path.join(BASE_DIR, "data/tq_options")
FUT_DIR = os.path.join(BASE_DIR, "data/futures_weighted")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/options_calculated")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Skip aggregated files
SKIP_PREFIXES = ('all_options_', 'options_')


def load_futures_hv():
    """加载期货数据计算历史波动率"""
    hv_cache = {}  # symbol -> {date_str -> hv20, hv60}
    if not os.path.isdir(FUT_DIR):
        return hv_cache
    for f in sorted(glob.glob(os.path.join(FUT_DIR, '*.csv'))):
        sym = os.path.basename(f).replace('.csv', '')
        try:
            df = pd.read_csv(f)
            if len(df) < 20:
                continue
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
            df = df.sort_values('trade_date').reset_index(drop=True)
            rets = df['close'].pct_change()
            for i in range(len(df)):
                row_date = df['trade_date'].iloc[i].strftime('%Y%m%d')
                hv20 = rets.iloc[max(0, i-19):i+1].std() * np.sqrt(252) if i >= 19 else None
                hv60 = rets.iloc[max(0, i-59):i+1].std() * np.sqrt(252) if i >= 59 else None
                hv_cache.setdefault(sym, {})[row_date] = {
                    'hv_20': round(float(hv20), 6) if hv20 else None,
                    'hv_60': round(float(hv60), 6) if hv60 else None,
                }
        except Exception as e:
            print(f"  Warning: failed to load {sym}: {e}")
    return hv_cache


def safe_float(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return float(v)


def process_file(fpath, hv_cache):
    """处理单个TQ期权文件，返回清洗后的记录列表"""
    fname = os.path.basename(fpath)

    # 提取product和date
    # 格式: EXCHANGE_PRODUCT_YYYYMMDD.json (e.g., CZCE_SA_20260520.json)
    parts = fname.replace('.json', '').rsplit('_', 1)
    if len(parts) != 2:
        return [], '', ''
    prefix, date_str = parts
    # Extract product from prefix (e.g., CZCE.SA -> SA, DCE.b -> b)
    product = prefix.split('_', 1)[-1] if '_' in prefix else prefix

    try:
        trade_date = datetime.strptime(date_str, '%Y%m%d')
    except ValueError:
        return [], '', ''

    with open(fpath) as f:
        records = json.load(f)
    if not isinstance(records, list):
        return [], '', ''

    results = []
    for r in records:
        # 必须有基础字段
        strike = safe_float(r.get('strike_price'))
        underlying = safe_float(r.get('underlying_price'))
        option_class = r.get('option_class', '').upper()
        iv_raw = safe_float(r.get('implied_volatility'))

        if not strike or not underlying or not iv_raw:
            continue

        # IV转换: TQ提供百分比(如26.15)或小数(如0.2615)
        iv_val = iv_raw
        if iv_val > 1.5:
            iv_val = iv_val / 100.0
        if iv_val <= 0.001 or iv_val > 2.0:
            continue

        # 确定市场价格
        last = safe_float(r.get('last_price'))
        bid = safe_float(r.get('bid_price1'))
        ask = safe_float(r.get('ask_price1'))
        pre_sett = safe_float(r.get('pre_settlement'))
        sett = safe_float(r.get('settlement'))

        market_price = None
        price_source = ''
        if last and last > 0:
            market_price = last
            price_source = 'last'
        elif bid and ask and bid > 0 and ask > 0:
            market_price = (bid + ask) / 2.0
            price_source = 'mid'
        elif sett and sett > 0:
            market_price = sett
            price_source = 'settlement'
        elif pre_sett and pre_sett > 0:
            market_price = pre_sett
            price_source = 'pre_settlement'

        if not market_price or market_price <= 0:
            continue

        # 到期日 - expire_datetime is UTC timestamp
        expire_ts = safe_float(r.get('expire_datetime'))
        if not expire_ts:
            continue

        expire_dt = datetime.fromtimestamp(expire_ts, tz=timezone.utc)
        days_to_expiry = max(1, (expire_dt - trade_date.replace(tzinfo=timezone.utc)).days)
        T_years = days_to_expiry / 365.0

        # Moneyness
        moneyness = round(strike / underlying, 4)

        # Greeks
        delta = safe_float(r.get('delta'))
        gamma = safe_float(r.get('gamma'))
        theta = safe_float(r.get('theta'))
        vega = safe_float(r.get('vega'))
        rho = safe_float(r.get('rho'))

        # HV from futures data - try product+fi, then product
        hv_key = product.lower() + 'fi'
        hv_info = hv_cache.get(hv_key, {}).get(date_str)
        if not hv_info:
            hv_info = hv_cache.get(product.lower(), {}).get(date_str)
        if not hv_info:
            hv_info = {}
        hv_20 = hv_info.get('hv_20') if hv_info else None
        hv_60 = hv_info.get('hv_60') if hv_info else None

        # IV/HV ratio
        iv_hv_ratio = round(iv_val / hv_20, 4) if hv_20 and hv_20 > 0 else None

        rec = {
            'symbol': r.get('symbol', ''),
            'product': product,
            'date': date_str,
            'option_type': option_class,
            'strike': strike,
            'underlying_price': underlying,
            'moneyness': moneyness,
            'market_price': round(market_price, 4),
            'price_source': price_source,
            'implied_vol': round(iv_val, 6),
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho,
            'days_to_expiry': days_to_expiry,
            'T_years': round(T_years, 6),
            'expiry_date': expire_dt.strftime('%Y-%m-%d'),
            'volume': int(r.get('volume', 0) or 0),
            'open_interest': int(r.get('open_interest', 0) or 0),
            'hv_20': hv_20,
            'hv_60': hv_60,
            'iv_hv_ratio': iv_hv_ratio,
        }
        results.append(rec)

    return results, product, date_str


def build_iv_summary(all_records):
    """构建IV汇总统计"""
    by_key = defaultdict(list)
    for r in all_records:
        key = f"{r['product']}_{r['date']}"
        by_key[key].append(r)

    summary = []
    for key, recs in sorted(by_key.items()):
        product = recs[0]['product']
        date = recs[0]['date']

        # ATM options (moneyness 0.95-1.05)
        atm = [r for r in recs if 0.95 <= r['moneyness'] <= 1.05 and r['implied_vol']]
        atm_ivs = [r['implied_vol'] for r in atm]
        atm_iv = float(np.mean(atm_ivs)) if atm_ivs else None

        # OTM put (moneyness <= 0.90)
        otm_puts = [r for r in recs if r['moneyness'] <= 0.90 and r['option_type'] == 'PUT' and r['implied_vol']]
        otm_put_iv = float(np.mean([r['implied_vol'] for r in otm_puts])) if otm_puts else None

        # OTM call (moneyness >= 1.10)
        otm_calls = [r for r in recs if r['moneyness'] >= 1.10 and r['option_type'] == 'CALL' and r['implied_vol']]
        otm_call_iv = float(np.mean([r['implied_vol'] for r in otm_calls])) if otm_calls else None

        skew = (otm_put_iv - otm_call_iv) if (otm_put_iv and otm_call_iv) else None

        all_ivs = [r['implied_vol'] for r in recs if r['implied_vol']]
        underlying_price = recs[0].get('underlying_price')
        hv_20 = recs[0].get('hv_20')
        iv_hv_ratio = round(atm_iv / hv_20, 4) if (atm_iv and hv_20 and hv_20 > 0) else None

        summary.append({
            'product': product,
            'date': date,
            'underlying_price': underlying_price,
            'atm_iv': round(atm_iv, 6) if atm_iv else None,
            'otm_put_iv': round(otm_put_iv, 6) if otm_put_iv else None,
            'otm_call_iv': round(otm_call_iv, 6) if otm_call_iv else None,
            'skew': round(skew, 6) if skew else None,
            'iv_min': round(float(min(all_ivs)), 6) if all_ivs else None,
            'iv_max': round(float(max(all_ivs)), 6) if all_ivs else None,
            'iv_median': round(float(np.median(all_ivs)), 6) if all_ivs else None,
            'hv_20': hv_20,
            'iv_hv_ratio': iv_hv_ratio,
            'n_contracts': len(recs),
            'n_with_iv': len(all_ivs),
        })

    summary.sort(key=lambda x: x.get('atm_iv') or 0, reverse=True)
    return summary


def main():
    print("加载期货数据计算HV...")
    t0 = time.time()
    hv_cache = load_futures_hv()
    print(f"  HV: {len(hv_cache)} 品种, {time.time()-t0:.1f}s")

    # Get all files, skip aggregated ones
    files = sorted(glob.glob(os.path.join(TQ_OPT_DIR, '*.json')))
    files = [f for f in files if not any(os.path.basename(f).startswith(p) for p in SKIP_PREFIXES)]
    # Also skip 0-byte files
    files = [f for f in files if os.path.getsize(f) > 0]
    print(f"\n处理 {len(files)} 个期权文件...")

    all_records = []
    by_product_date = defaultdict(list)
    seen = set()  # (product, date) pairs

    for i, fpath in enumerate(files):
        recs, product, date_str = process_file(fpath, hv_cache)
        if not recs:
            continue

        key = f"{product}_{date_str}"
        if key in seen:
            continue  # deduplicate
        seen.add(key)

        all_records.extend(recs)
        by_product_date[key] = recs

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(files)} done, {len(all_records)} valid records")

    print(f"\n总计: {len(all_records)} 条有效记录")
    dates = set()
    products = set()
    for key in by_product_date:
        parts = key.split('_', 1)
        products.add(parts[0])
        dates.add(parts[1])
    print(f"  {len(products)} 品种, {len(dates)} 日期: {sorted(dates)}")

    # Save per product_date
    for key, recs in by_product_date.items():
        out_path = os.path.join(OUTPUT_DIR, f"{key}.json")
        with open(out_path, 'w') as f:
            json.dump(recs, f, ensure_ascii=False)

    # Save all
    print("保存 all_options_with_iv.json ...")
    with open(os.path.join(OUTPUT_DIR, 'all_options_with_iv.json'), 'w') as f:
        json.dump(all_records, f, ensure_ascii=False)

    # Save IV summary
    print("生成 IV汇总...")
    iv_summary = build_iv_summary(all_records)
    with open(os.path.join(OUTPUT_DIR, 'iv_summary.json'), 'w') as f:
        json.dump(iv_summary, f, ensure_ascii=False)
    print(f"  {len(iv_summary)} 条汇总")

    print(f"\n完成! {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
