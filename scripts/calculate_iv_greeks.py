#!/usr/bin/env python3
"""
从TQ期权原始数据计算真实IV和Greeks
- 使用Newton-Raphson反推隐含波动率
- BSM模型计算Delta/Gamma/Theta/Vega/Rho
- 从期货日线数据计算历史波动率(HV)
"""
import os, json, glob, math, time
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timedelta

BASE_DIR = os.path.expanduser("~/home/futures_platform")
TQ_OPT_DIR = os.path.join(BASE_DIR, "data/tq_options")
FUT_DIR = os.path.join(BASE_DIR, "data/futures_weighted")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/options_calculated")

RISK_FREE_RATE = 0.02  # 无风险利率近似


# ============ BSM 模型 ============

def bsm_d1(S, K, T, r, sigma):
    """计算d1"""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def bsm_d2(S, K, T, r, sigma):
    """计算d2"""
    d1 = bsm_d1(S, K, T, r, sigma)
    if d1 is None:
        return None
    return d1 - sigma * math.sqrt(T)


def bsm_call_price(S, K, T, r, sigma):
    """BSM看涨期权价格"""
    d1 = bsm_d1(S, K, T, r, sigma)
    d2 = bsm_d2(S, K, T, r, sigma)
    if d1 is None or d2 is None:
        return None
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bsm_put_price(S, K, T, r, sigma):
    """BSM看跌期权价格"""
    d1 = bsm_d1(S, K, T, r, sigma)
    d2 = bsm_d2(S, K, T, r, sigma)
    if d1 is None or d2 is None:
        return None
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bsm_price(S, K, T, r, sigma, option_type):
    """根据类型计算价格"""
    if option_type in ('CALL', 'c', 'C'):
        return bsm_call_price(S, K, T, r, sigma)
    else:
        return bsm_put_price(S, K, T, r, sigma)


def implied_vol_newton(S, K, T, r, market_price, option_type,
                       max_iter=100, tol=1e-8, vol_lower=0.001, vol_upper=5.0):
    """Newton-Raphson法求解隐含波动率"""
    if market_price is None or market_price <= 0:
        return None
    if T <= 0 or S <= 0 or K <= 0:
        return None

    # 初始猜测
    sigma = 0.3  # 从30%开始

    for i in range(max_iter):
        price = bsm_price(S, K, T, r, sigma, option_type)
        if price is None:
            return None

        diff = price - market_price
        if abs(diff) < tol:
            return sigma

        # Vega = S * sqrt(T) * N'(d1)
        d1 = bsm_d1(S, K, T, r, sigma)
        if d1 is None:
            return None
        vega = S * math.sqrt(T) * norm.pdf(d1)
        if vega < 1e-12:
            return None

        sigma_new = sigma - diff / vega
        # 限制范围
        sigma_new = max(vol_lower, min(vol_upper, sigma_new))

        # 检查收敛
        if abs(sigma_new - sigma) < tol:
            return sigma_new
        sigma = sigma_new

    # 未收敛，尝试二分法
    return implied_vol_bisect(S, K, T, r, market_price, option_type, vol_lower, vol_upper)


def implied_vol_bisect(S, K, T, r, market_price, option_type,
                       vol_lower=0.001, vol_upper=5.0, max_iter=100, tol=1e-8):
    """二分法求解隐含波动率"""
    lo, hi = vol_lower, vol_upper

    price_lo = bsm_price(S, K, T, r, lo, option_type)
    price_hi = bsm_price(S, K, T, r, hi, option_type)
    if price_lo is None or price_hi is None:
        return None

    # 检查价格是否在范围内
    target = market_price
    # Call: 价格随vol单调递增; Put同理
    if price_lo > target or price_hi < target:
        return None

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        price_mid = bsm_price(S, K, T, r, mid, option_type)
        if price_mid is None:
            return None
        if abs(price_mid - target) < tol:
            return mid
        if price_mid < target:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2


def calc_greeks(S, K, T, r, sigma, option_type):
    """计算全部Greeks"""
    d1 = bsm_d1(S, K, T, r, sigma)
    d2 = bsm_d2(S, K, T, r, sigma)
    if d1 is None or d2 is None:
        return {}

    sqrtT = math.sqrt(T)
    exp_rT = math.exp(-r * T)
    pdf_d1 = norm.pdf(d1)

    # Delta
    if option_type in ('CALL', 'c', 'C'):
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # Gamma
    gamma = pdf_d1 / (S * sigma * sqrtT)

    # Theta (per calendar day)
    if option_type in ('CALL', 'c', 'C'):
        theta = (-S * pdf_d1 * sigma / (2 * sqrtT)
                 - r * K * exp_rT * norm.cdf(d2))
    else:
        theta = (-S * pdf_d1 * sigma / (2 * sqrtT)
                 + r * K * exp_rT * norm.cdf(-d2))
    theta_per_day = theta / 365.0

    # Vega (per 1% change in vol)
    vega = S * sqrtT * pdf_d1 / 100.0

    # Rho (per 1% change in rate)
    if option_type in ('CALL', 'c', 'C'):
        rho = K * T * exp_rT * norm.cdf(d2) / 100.0
    else:
        rho = -K * T * exp_rT * norm.cdf(-d2) / 100.0

    return {
        'delta': round(delta, 6),
        'gamma': round(gamma, 6),
        'theta': round(theta_per_day, 6),
        'vega': round(vega, 6),
        'rho': round(rho, 6),
    }


# ============ 历史波动率 ============

def load_futures_data():
    """加载期货日线数据"""
    all_data = {}
    for f in sorted(glob.glob(os.path.join(FUT_DIR, '*.csv'))):
        sym = os.path.basename(f).replace('.csv', '')
        try:
            df = pd.read_csv(f)
            if len(df) < 20:
                continue
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='mixed')
            df = df.sort_values('trade_date').reset_index(drop=True)
            all_data[sym] = df
        except:
            continue
    return all_data


def calc_hv(df, window=20):
    """计算历史波动率 (年化)"""
    if len(df) < window + 1:
        return None
    returns = df['close'].pct_change().dropna().tail(window)
    if len(returns) < window:
        return None
    return float(returns.std() * math.sqrt(252))


def calc_parkinson_hv(df, window=20):
    """Parkinson波动率 (使用日内高低价)"""
    if len(df) < window:
        return None
    sub = df.tail(window)
    hl_ratio = np.log(sub['high'].values / sub['low'].values)
    # Parkinson factor = 1/(4*ln2) ≈ 0.3611
    factor = 1.0 / (4.0 * math.log(2))
    variance = factor * np.sum(hl_ratio ** 2) / len(sub)
    return float(math.sqrt(variance * 252))


# ============ TQ产品代码 → 期货品种代码映射 ============

def tq_product_to_symbol(product):
    """TQ期权product → futures_weighted里的symbol
    e.g. 'm' → 'mfi', 'au' → 'aufi', 'AP' → 'apfi'
    """
    product_lower = product.lower()
    # 直接映射 (futures_weighted里的小写fi后缀)
    sym = f"{product_lower}fi"
    return sym


# ============ 主处理逻辑 ============

def is_valid_price(val):
    """检查价格是否有效"""
    if val is None:
        return False
    s = str(val)
    if s in ('NaN', 'nan', '', 'None', '0', '0.0'):
        return False
    try:
        return float(val) > 0
    except:
        return False


def get_best_price(opt):
    """获取最佳可用价格 (优先mid > last > settlement)"""
    bid = opt.get('bid_price1')
    ask = opt.get('ask_price1')
    last = opt.get('last_price')
    settle = opt.get('settlement')

    # 优先使用mid price (bid/ask均值)
    if is_valid_price(bid) and is_valid_price(ask):
        return (float(bid) + float(ask)) / 2.0, 'mid'

    # 其次用last_price
    if is_valid_price(last) and float(last) > 0:
        return float(last), 'last'

    # 再用settlement
    if is_valid_price(settle) and float(settle) > 0:
        return float(settle), 'settle'

    # 只有一个方向的价格时使用
    if is_valid_price(bid):
        return float(bid), 'bid_only'
    if is_valid_price(ask):
        return float(ask), 'ask_only'

    return None, None


def process_tq_options(date_filter=None):
    """处理TQ期权数据，计算IV和Greeks"""
    print("加载TQ期权原始数据...")
    files = sorted(glob.glob(os.path.join(TQ_OPT_DIR, '*.json')))
    print(f"  找到 {len(files)} 个文件")

    # 加载期货数据计算HV
    print("加载期货日线数据...")
    fut_data = load_futures_data()
    print(f"  找到 {len(fut_data)} 个品种")

    all_results = []
    stats = {'total': 0, 'valid_price': 0, 'iv_success': 0, 'iv_fail': 0}

    for fpath in files:
        fname = os.path.basename(fpath)
        # 解析日期
        parts = fname.replace('.json', '').split('_')
        if len(parts) >= 3:
            file_date = parts[-1]
            product = parts[1]
        elif fname.startswith('all_options_'):
            file_date = fname.replace('all_options_', '').replace('.json', '')
            product = 'ALL'
        else:
            continue

        if date_filter and file_date != date_filter:
            continue

        try:
            with open(fpath) as fp:
                data = json.load(fp)
        except:
            continue
        if not isinstance(data, list):
            continue

        # 计算HV
        fut_sym = tq_product_to_symbol(product)
        hv_20 = hv_60 = parkinson_20 = None
        if fut_sym in fut_data:
            df = fut_data[fut_sym]
            hv_20 = calc_hv(df, 20)
            hv_60 = calc_hv(df, 60)
            parkinson_20 = calc_parkinson_hv(df, 20)

        file_date_dt = datetime.strptime(file_date, '%Y%m%d')

        for opt in data:
            stats['total'] += 1
            S = opt.get('underlying_price', 0)
            K = opt.get('strike_price', 0)
            option_type = opt.get('option_class', '')
            exp_ts = opt.get('expire_datetime', 0)
            vol = opt.get('volume', 0)
            oi = opt.get('open_interest', 0)

            if not S or not K or not exp_ts or not option_type:
                continue
            S = float(S)
            K = float(K)
            if S <= 0 or K <= 0:
                continue

            # 计算到期时间(年)
            exp_dt = datetime.fromtimestamp(float(exp_ts))
            T = (exp_dt - file_date_dt).total_seconds() / (365.25 * 24 * 3600)
            if T <= 1.0 / 365.0:  # 不到1天的跳过
                continue

            # 获取价格
            price, price_source = get_best_price(opt)
            if price is None or price <= 0:
                continue
            stats['valid_price'] += 1

            # 计算IV
            iv = implied_vol_newton(S, K, T, RISK_FREE_RATE, price, option_type)
            if iv is None:
                stats['iv_fail'] += 1
                continue
            stats['iv_success'] += 1

            # 计算Greeks
            greeks = calc_greeks(S, K, T, RISK_FREE_RATE, iv, option_type)
            if not greeks:
                continue

            # Moneyness
            moneyness = K / S
            # 到期天数
            days_to_exp = max(1, int((exp_dt - file_date_dt).days))

            result = {
                'symbol': opt.get('symbol', ''),
                'product': product,
                'date': file_date,
                'option_type': option_type,
                'strike': K,
                'underlying_price': S,
                'moneyness': round(moneyness, 4),
                'expiry_date': exp_dt.strftime('%Y-%m-%d'),
                'days_to_expiry': days_to_exp,
                'T_years': round(T, 6),
                'market_price': round(price, 4),
                'price_source': price_source,
                'implied_vol': round(iv, 6),
                'volume': vol,
                'open_interest': oi,
                'hv_20': round(hv_20, 6) if hv_20 else None,
                'hv_60': round(hv_60, 6) if hv_60 else None,
                'parkinson_20': round(parkinson_20, 6) if parkinson_20 else None,
                'iv_hv_ratio': round(iv / hv_20, 4) if hv_20 and hv_20 > 0 else None,
                # Greeks
                'delta': greeks['delta'],
                'gamma': greeks['gamma'],
                'theta': greeks['theta'],
                'vega': greeks['vega'],
                'rho': greeks['rho'],
            }
            all_results.append(result)

    print(f"\n处理统计:")
    print(f"  总期权数: {stats['total']:,}")
    print(f"  有效价格: {stats['valid_price']:,} ({stats['valid_price']/max(stats['total'],1)*100:.1f}%)")
    print(f"  IV计算成功: {stats['iv_success']:,} ({stats['iv_success']/max(stats['valid_price'],1)*100:.1f}%)")
    print(f"  IV计算失败: {stats['iv_fail']:,}")

    return all_results


def generate_summary(results):
    """生成各品种IV汇总"""
    if not results:
        return []

    df = pd.DataFrame(results)
    summaries = []

    for (product, date), grp in df.groupby(['product', 'date']):
        # ATM IV (moneyness 0.95-1.05)
        atm = grp[(grp['moneyness'] >= 0.95) & (grp['moneyness'] <= 1.05)]
        atm_iv = atm['implied_vol'].mean() if len(atm) > 0 else None

        # OTM Put IV (moneyness < 0.90)
        otm_put = grp[(grp['option_type'] == 'PUT') & (grp['moneyness'] <= 0.90)]
        otm_put_iv = otm_put['implied_vol'].mean() if len(otm_put) > 0 else None

        # OTM Call IV (moneyness > 1.10)
        otm_call = grp[(grp['option_type'] == 'CALL') & (grp['moneyness'] >= 1.10)]
        otm_call_iv = otm_call['implied_vol'].mean() if len(otm_call) > 0 else None

        # Skew = OTM Put IV - OTM Call IV
        skew = (otm_put_iv - otm_call_iv) if otm_put_iv and otm_call_iv else None

        # IV rank (percentile vs HV)
        hv20 = grp['hv_20'].iloc[0] if 'hv_20' in grp.columns and len(grp) > 0 else None
        iv_hv = grp['iv_hv_ratio'].iloc[0] if 'iv_hv_ratio' in grp.columns and len(grp) > 0 else None

        summaries.append({
            'product': product,
            'date': date,
            'n_contracts': len(grp),
            'n_with_iv': len(grp[grp['implied_vol'].notna()]),
            'underlying_price': float(grp['underlying_price'].iloc[0]) if len(grp) > 0 else None,
            'atm_iv': round(float(atm_iv), 6) if atm_iv else None,
            'otm_put_iv': round(float(otm_put_iv), 6) if otm_put_iv else None,
            'otm_call_iv': round(float(otm_call_iv), 6) if otm_call_iv else None,
            'skew': round(float(skew), 6) if skew else None,
            'hv_20': round(float(hv20), 6) if hv20 else None,
            'iv_hv_ratio': round(float(iv_hv), 4) if iv_hv else None,
            'iv_min': round(float(grp['implied_vol'].min()), 6),
            'iv_max': round(float(grp['implied_vol'].max()), 6),
            'iv_median': round(float(grp['implied_vol'].median()), 6),
        })

    summaries.sort(key=lambda x: x.get('atm_iv') or 0, reverse=True)
    return summaries


def save_results(results, summaries, output_dir):
    """保存计算结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 按product+date分组保存
    from collections import defaultdict
    by_product_date = defaultdict(list)
    for r in results:
        key = f"{r['product']}_{r['date']}"
        by_product_date[key].append(r)

    for key, items in by_product_date.items():
        outpath = os.path.join(output_dir, f"{key}.json")
        with open(outpath, 'w') as f:
            json.dump(items, f, ensure_ascii=False)

    # 保存汇总
    summary_path = os.path.join(output_dir, "iv_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    # 保存全量数据
    all_path = os.path.join(output_dir, "all_options_with_iv.json")
    with open(all_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False)

    print(f"\n保存完成:")
    print(f"  分品种文件: {len(by_product_date)} 个")
    print(f"  汇总文件: {summary_path}")
    print(f"  全量文件: {all_path}")
    print(f"  总记录数: {len(results):,}")


def print_iv_report(summaries):
    """打印IV分析报告"""
    if not summaries:
        print("无汇总数据")
        return

    print("\n" + "=" * 80)
    print("IV分析报告 (按ATM IV降序)")
    print("=" * 80)
    print(f"{'品种':>6} {'合约数':>6} {'ATM IV':>8} {'Skew':>8} {'HV20':>8} {'IV/HV':>6} {'IV范围':>16}")
    print("-" * 80)

    for s in summaries[:30]:
        atm = f"{s['atm_iv']*100:.1f}%" if s['atm_iv'] else '-'
        skew = f"{s['skew']*100:.1f}%" if s['skew'] else '-'
        hv = f"{s['hv_20']*100:.1f}%" if s['hv_20'] else '-'
        ratio = f"{s['iv_hv_ratio']:.2f}" if s['iv_hv_ratio'] else '-'
        iv_range = f"{s['iv_min']*100:.1f}%-{s['iv_max']*100:.1f}%"
        print(f"{s['product']:>6} {s['n_with_iv']:>6} {atm:>8} {skew:>8} {hv:>8} {ratio:>6} {iv_range:>16}")

    print(f"\n共 {len(summaries)} 个品种")


if __name__ == '__main__':
    t0 = time.time()
    print("开始计算IV和Greeks...")
    print(f"TQ期权数据目录: {TQ_OPT_DIR}")

    results = process_tq_options()
    summaries = generate_summary(results)
    save_results(results, summaries, OUTPUT_DIR)
    print_iv_report(summaries)

    elapsed = time.time() - t0
    print(f"\n耗时: {elapsed:.1f}s")
