"""
V320: 深入市场结构分析 — 理解中国期货市场本质
================================================
不再盲目堆因子。先搞清楚：
1. 各品种的收益分布特征（偏度、峰度、自相关）
2. 价格涨跌停的影响
3. 夜盘vs日盘的收益结构
4. 持仓量(OI)变化与价格的因果关系
5. 连续涨/跌的天数分布
6. 品种间的相关结构（谁领导谁）
7. 动量在不同市场状态下的表现差异
8. 真正的alpha来源在哪里
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
from collections import defaultdict
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_futures_v301 import load_all_data


def analyze_return_distribution(C, O, H, L, V, OI, NS, ND, dates, syms):
    """分析每个品种的收益分布特征"""
    print("\n" + "="*80)
    print("  1. 各品种收益分布特征")
    print("="*80)

    results = []
    for si in range(NS):
        rets = []
        oc_rets = []  # open-to-close (intraday)
        co_rets = []  # close-to-open (overnight)
        for di in range(1, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                r = C[si, di] / C[si, di-1] - 1
                rets.append(r)

                # Intraday vs overnight decomposition
                o = O[si, di]
                if not np.isnan(o) and o > 0 and not np.isnan(C[si, di]):
                    oc = C[si, di] / o - 1  # intraday
                    co = o / C[si, di-1] - 1  # overnight
                    oc_rets.append(oc)
                    co_rets.append(co)

        if len(rets) < 200:
            continue

        rets = np.array(rets)
        oc_rets = np.array(oc_rets) if oc_rets else np.array([0])
        co_rets = np.array(co_rets) if co_rets else np.array([0])

        # Autocorrelation at lags 1, 2, 3, 5
        autocorr = []
        for lag in [1, 2, 3, 5]:
            if len(rets) > lag + 10:
                ac = np.corrcoef(rets[:-lag], rets[lag:])[0, 1]
                autocorr.append(ac)
            else:
                autocorr.append(0)

        # Positive day % and consecutive wins/losses
        pos_pct = np.mean(rets > 0) * 100

        # Max consecutive up/down
        max_consec_up = 0
        max_consec_dn = 0
        cur_up = cur_dn = 0
        for r in rets:
            if r > 0:
                cur_up += 1; cur_dn = 0
                max_consec_up = max(max_consec_up, cur_up)
            else:
                cur_dn += 1; cur_up = 0
                max_consec_dn = max(max_consec_dn, cur_dn)

        # Overnight vs intraday variance
        vol_total = np.std(rets) * np.sqrt(252)
        vol_intraday = np.std(oc_rets) * np.sqrt(252)
        vol_overnight = np.std(co_rets) * np.sqrt(252)

        # Skewness and kurtosis
        from scipy import stats as sp_stats
        skew = sp_stats.skew(rets)
        kurt = sp_stats.kurtosis(rets)

        # Average daily volume
        vol_avg = np.nanmean(V[si, :])

        results.append({
            'sym': syms[si],
            'mean_ret': np.mean(rets) * 252 * 100,
            'vol': vol_total,
            'vol_intra': vol_intraday,
            'vol_overnight': vol_overnight,
            'overnight_pct': vol_overnight / vol_total * 100 if vol_total > 0 else 0,
            'skew': skew,
            'kurt': kurt,
            'ac1': autocorr[0],
            'ac2': autocorr[1],
            'ac5': autocorr[4] if len(autocorr) > 4 else 0,
            'pos_pct': pos_pct,
            'max_up': max_consec_up,
            'max_dn': max_consec_dn,
            'vol_avg': vol_avg,
        })

    df = pd.DataFrame(results)
    df = df.sort_values('vol', ascending=False)

    print(f"\n{'Sym':>6} {'AnnRet':>7} {'Vol':>6} {'VolOC':>6} {'VolCO':>6} {'CO%':>4} "
          f"{'Skew':>6} {'Kurt':>5} {'AC1':>6} {'AC5':>6} {'Pos%':>5} {'MaxUp':>5}")
    print("-" * 90)
    for _, row in df.iterrows():
        print(f"{row['sym']:>6} {row['mean_ret']:>+7.1f} {row['vol']:>6.2f} "
              f"{row['vol_intra']:>6.2f} {row['vol_overnight']:>6.2f} "
              f"{row['overnight_pct']:>4.0f} "
              f"{row['skew']:>+6.3f} {row['kurt']:>5.1f} "
              f"{row['ac1']:>+6.3f} {row['ac5']:>+6.3f} "
              f"{row['pos_pct']:>5.1f} {row['max_up']:>5.0f}")

    return df


def analyze_autocorrelation_structure(C, NS, ND, dates, syms):
    """分析自相关结构：动量效应在哪些品种、哪些时间尺度上存在"""
    print("\n" + "="*80)
    print("  2. 自相关结构 — 动量效应的真实来源")
    print("="*80)

    # Cross-commodity average autocorrelation at different lags
    lags = list(range(1, 21))
    avg_ac = []
    for lag in lags:
        acs = []
        for si in range(NS):
            rets = []
            for di in range(1, ND):
                if not np.isnan(C[si, di]) and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                    rets.append(C[si, di] / C[si, di-1] - 1)
            rets = np.array(rets)
            if len(rets) > lag + 50:
                ac = np.corrcoef(rets[:-lag], rets[lag:])[0, 1]
                acs.append(ac)
        avg_ac.append(np.mean(acs) if acs else 0)

    print(f"\n  Lag-wise average autocorrelation across all commodities:")
    for i, lag in enumerate(lags):
        bar = "+" * max(0, int(avg_ac[i] * 500)) if avg_ac[i] > 0 else "-" * max(0, int(-avg_ac[i] * 500))
        print(f"  Lag {lag:>2}d: {avg_ac[i]:>+7.4f}  {bar}")

    # Which commodities have strongest AC1 (predictability)?
    print(f"\n  Top-10 most predictable (highest AC1):")
    ac1_by_sym = []
    for si in range(NS):
        rets = []
        for di in range(1, ND):
            if not np.isnan(C[si, di]) and not np.isnan(C[si, di-1]) and C[si, di-1] > 0:
                rets.append(C[si, di] / C[si, di-1] - 1)
        if len(rets) > 100:
            ac1 = np.corrcoef(rets[:-1], rets[1:])[0, 1]
            ac1_by_sym.append((ac1, syms[si]))
    ac1_by_sym.sort(key=lambda x: -x[0])
    for ac, sym in ac1_by_sym[:10]:
        print(f"    {sym}: AC1={ac:+.4f}")

    print(f"\n  Top-10 most negative AC1 (mean-reverting):")
    for ac, sym in ac1_by_sym[-10:]:
        print(f"    {sym}: AC1={ac:+.4f}")

    return avg_ac


def analyze_overnight_intraday(C, O, NS, ND, dates, syms):
    """分析隔夜vs日内的收益结构"""
    print("\n" + "="*80)
    print("  3. 隔夜 vs 日内收益分解")
    print("="*80)

    co_rets_all = []  # close-to-open (overnight)
    oc_rets_all = []  # open-to-close (intraday)
    cc_rets_all = []  # close-to-close (full day)

    for si in range(NS):
        for di in range(1, ND):
            c_prev = C[si, di-1]
            o_now = O[si, di]
            c_now = C[si, di]
            if np.isnan(c_prev) or np.isnan(o_now) or np.isnan(c_now):
                continue
            if c_prev <= 0 or o_now <= 0:
                continue

            co = o_now / c_prev - 1  # overnight
            oc = c_now / o_now - 1   # intraday
            cc = c_now / c_prev - 1  # full day

            co_rets_all.append((syms[si], di, co))
            oc_rets_all.append((syms[si], di, oc))
            cc_rets_all.append((syms[si], di, cc))

    co_vals = np.array([r[2] for r in co_rets_all])
    oc_vals = np.array([r[2] for r in oc_rets_all])
    cc_vals = np.array([r[2] for r in cc_rets_all])

    print(f"\n  Overall statistics:")
    print(f"    Overnight (CO): mean={np.mean(co_vals)*252*100:+.1f}% ann, "
          f"std={np.std(co_vals)*np.sqrt(252):.2f}, "
          f"positive={np.mean(co_vals>0)*100:.1f}%")
    print(f"    Intraday  (OC): mean={np.mean(oc_vals)*252*100:+.1f}% ann, "
          f"std={np.std(oc_vals)*np.sqrt(252):.2f}, "
          f"positive={np.mean(oc_vals>0)*100:.1f}%")
    print(f"    Full day  (CC): mean={np.mean(cc_vals)*252*100:+.1f}% ann, "
          f"std={np.std(cc_vals)*np.sqrt(252):.2f}, "
          f"positive={np.mean(cc_vals>0)*100:.1f}%")

    # CO vs OC correlation
    corr = np.corrcoef(co_vals, oc_vals)[0, 1]
    print(f"\n    CO-OC correlation: {corr:+.4f} (negative = reversal)")

    # By year
    print(f"\n  By year:")
    for year in sorted(set(d.year for d in dates)):
        yr_co = [r[2] for r in co_rets_all if dates[r[1]].year == year]
        yr_oc = [r[2] for r in oc_rets_all if dates[r[1]].year == year]
        if yr_co and yr_oc:
            print(f"    {year}: CO={np.mean(yr_co)*252*100:+.1f}% OC={np.mean(yr_oc)*252*100:+.1f}% "
                  f"CO_pos={np.mean(np.array(yr_co)>0)*100:.1f}% OC_pos={np.mean(np.array(yr_oc)>0)*100:.1f}%")

    # Big gap analysis
    print(f"\n  Big gap analysis:")
    for threshold in [0.005, 0.01, 0.02]:
        big_up = co_vals > threshold
        big_dn = co_vals < -threshold
        if big_up.sum() > 20:
            avg_oc_after_up = np.mean(oc_vals[big_up])
            pos_after_up = np.mean(oc_vals[big_up] > 0) * 100
            print(f"    Gap UP >{threshold*100:.1f}%: n={big_up.sum()}, "
                  f"avg intraday={avg_oc_after_up:+.4f} ({pos_after_up:.1f}% positive)")
        if big_dn.sum() > 20:
            avg_oc_after_dn = np.mean(oc_vals[big_dn])
            pos_after_dn = np.mean(oc_vals[big_dn] > 0) * 100
            print(f"    Gap DN <{-threshold*100:.1f}%: n={big_dn.sum()}, "
                  f"avg intraday={avg_oc_after_dn:+.4f} ({pos_after_dn:.1f}% positive)")


def analyze_oi_price_relationship(C, V, OI, NS, ND, dates, syms):
    """持仓量与价格的关系 — 资金流向分析"""
    print("\n" + "="*80)
    print("  4. 持仓量(OI)与价格关系 — 资金流向")
    print("="*80)

    # OI up + price up = new longs entering (bullish)
    # OI up + price down = new shorts entering (bearish)
    # OI down + price up = shorts covering (less bullish)
    # OI down + price down = longs exiting (less bearish)

    categories = {'oi_up_p_up': [], 'oi_up_p_dn': [], 'oi_dn_p_up': [], 'oi_dn_p_dn': []}

    for si in range(NS):
        for di in range(5, ND):
            oi_now = OI[si, di]
            oi_5 = OI[si, di-5]
            c_now = C[si, di]
            c_5 = C[si, di-5]

            if np.isnan(oi_now) or np.isnan(oi_5) or np.isnan(c_now) or np.isnan(c_5):
                continue
            if c_5 <= 0:
                continue

            ret = (c_now - c_5) / c_5
            oi_chg = oi_now - oi_5

            # Next 5-day return (forward looking, for prediction)
            if di + 5 < ND and not np.isnan(C[si, di+5]) and c_now > 0:
                fwd_ret = (C[si, di+5] - c_now) / c_now

                if oi_chg > 0 and ret > 0:
                    categories['oi_up_p_up'].append(fwd_ret)
                elif oi_chg > 0 and ret <= 0:
                    categories['oi_up_p_dn'].append(fwd_ret)
                elif oi_chg <= 0 and ret > 0:
                    categories['oi_dn_p_up'].append(fwd_ret)
                else:
                    categories['oi_dn_p_dn'].append(fwd_ret)

    print(f"\n  Forward 5-day returns by OI/Price category:")
    for cat, rets in categories.items():
        if rets:
            rets = np.array(rets)
            pos_pct = np.mean(rets > 0) * 100
            avg = np.mean(rets) * 252 * 100 / 5
            print(f"    {cat:>15}: n={len(rets):>6}, "
                  f"avg fwd={avg:+.2f}% ann, positive={pos_pct:.1f}%")


def analyze_consecutive_moves(C, NS, ND, dates, syms):
    """连续涨跌后的收益 — 动量/反转的真实表现"""
    print("\n" + "="*80)
    print("  5. 连续涨跌后的未来收益 — 动量的真实alpha")
    print("="*80)

    for consec_days in [2, 3, 4, 5]:
        up_next = []
        dn_next = []

        for si in range(NS):
            consec_up = 0
            consec_dn = 0
            for di in range(1, ND):
                if np.isnan(C[si, di]) or np.isnan(C[si, di-1]) or C[si, di-1] <= 0:
                    consec_up = consec_dn = 0
                    continue

                ret = C[si, di] / C[si, di-1] - 1

                if ret > 0:
                    consec_up += 1
                    consec_dn = 0
                else:
                    consec_dn += 1
                    consec_up = 0

                # After N consecutive days, look at next day
                if consec_up == consec_days and di + 1 < ND:
                    if not np.isnan(C[si, di+1]) and C[si, di] > 0:
                        fwd = C[si, di+1] / C[si, di] - 1
                        up_next.append(fwd)

                if consec_dn == consec_days and di + 1 < ND:
                    if not np.isnan(C[si, di+1]) and C[si, di] > 0:
                        fwd = C[si, di+1] / C[si, di] - 1
                        dn_next.append(fwd)

        if up_next:
            up_arr = np.array(up_next)
            print(f"\n  After {consec_days} consecutive UP days:")
            print(f"    n={len(up_arr)}, avg next day={np.mean(up_arr)*100:+.4f}%, "
                  f"positive={np.mean(up_arr>0)*100:.1f}%")
        if dn_next:
            dn_arr = np.array(dn_next)
            print(f"  After {consec_days} consecutive DOWN days:")
            print(f"    n={len(dn_arr)}, avg next day={np.mean(dn_arr)*100:+.4f}%, "
                  f"positive={np.mean(dn_arr>0)*100:.1f}%")


def analyze_momentum_by_quantile(C, NS, ND, dates, syms):
    """横截面动量的真实预测力 — 按momentum十分位看未来收益"""
    print("\n" + "="*80)
    print("  7. 横截面动量的真实预测力")
    print("="*80)

    for mom_period in [5, 10, 20]:
        fwd_rets_by_decile = defaultdict(list)

        for di in range(mom_period + 1, ND - 5):
            # Compute cross-sectional momentum
            mom = np.full(NS, np.nan)
            for si in range(NS):
                c0 = C[si, di - mom_period]
                c1 = C[si, di]
                if not np.isnan(c0) and not np.isnan(c1) and c0 > 0:
                    mom[si] = (c1 - c0) / c0

            valid = ~np.isnan(mom)
            if valid.sum() < 10:
                continue

            # Rank into deciles
            ranks = pd.Series(mom).rank(pct=True, na_option='keep')

            # Forward 5-day return
            for si in range(NS):
                if np.isnan(ranks[si]):
                    continue
                if di + 5 >= ND:
                    continue
                if np.isnan(C[si, di+5]) or np.isnan(C[si, di]) or C[si, di] <= 0:
                    continue
                fwd = (C[si, di+5] - C[si, di]) / C[si, di]
                decile = int(ranks[si] * 10)  # 0-9
                decile = min(decile, 9)
                fwd_rets_by_decile[decile].append(fwd)

        print(f"\n  Momentum period = {mom_period}d, Forward = 5d:")
        print(f"  {'Decile':>8} {'N':>7} {'Avg Fwd':>10} {'Ann':>8} {'Pos%':>6}")
        print(f"  {'-'*45}")
        for d in sorted(fwd_rets_by_decile.keys()):
            rets = np.array(fwd_rets_by_decile[d])
            if len(rets) > 50:
                print(f"  {'Q'+str(d+1):>8} {len(rets):>7} {np.mean(rets)*100:>+10.4f}% "
                      f"{np.mean(rets)*252/5*100:>+8.1f}% {np.mean(rets>0)*100:>6.1f}%")

        # Long top decile, short bottom decile
        if 0 in fwd_rets_by_decile and 9 in fwd_rets_by_decile:
            top = np.array(fwd_rets_by_decile[9])
            bot = np.array(fwd_rets_by_decile[0])
            spread = np.mean(top) - np.mean(bot)
            print(f"  → Top-Bottom spread: {spread*252/5*100:+.1f}% annualized")


def main():
    t0 = time.time()
    print("=" * 80)
    print("  V320: 深入市场结构分析 — 理解中国期货市场本质")
    print("=" * 80)

    C, O, H, L, V, OI, NS, ND, dates, syms = load_all_data(start='2016-01-01')

    print(f"  {NS} commodities, {ND} days, "
          f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    df = analyze_return_distribution(C, O, H, L, V, OI, NS, ND, dates, syms)
    avg_ac = analyze_autocorrelation_structure(C, NS, ND, dates, syms)
    analyze_overnight_intraday(C, O, NS, ND, dates, syms)
    analyze_oi_price_relationship(C, V, OI, NS, ND, dates, syms)
    analyze_consecutive_moves(C, NS, ND, dates, syms)
    analyze_momentum_by_quantile(C, NS, ND, dates, syms)

    print(f"\n[V320] Done. {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
