#!/usr/bin/env python3
"""
期权分析模块
- 隐含波动率计算（BSM模型）
- 希腊字母计算（Delta/Gamma/Theta/Vega/Rho）
- 波动率曲面构建
- 期权期限结构
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# 尝试导入外部库
try:
    from py_vollib.black_scholes_merton.implied_volatility import implied_volatility as iv_bsm
    VOLLIB_AVAILABLE = True
except ImportError:
    VOLLIB_AVAILABLE = False

try:
    import QuantLib as ql
    QUANTLIB_AVAILABLE = True
except ImportError:
    QUANTLIB_AVAILABLE = False

BASE_DIR = os.path.expanduser("~/home/futures_platform")
DATA_DIR = os.path.join(BASE_DIR, "data")


def bsm_price(S, K, T, r, q, sigma, flag):
    """BSM定价公式"""
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if flag == 'c' else max(0, K - S)
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if flag == 'c':
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    return price


def bsm_iv(S, K, T, r, q, market_price, flag, tol=1e-6, max_iter=100):
    """用二分法计算隐含波动率"""
    if market_price <= 0:
        return None
    
    # 边界检查
    intrinsic = max(0, S - K) if flag == 'c' else max(0, K - S)
    if market_price < intrinsic:
        return None
    
    sigma_low, sigma_high = 0.001, 5.0
    
    for _ in range(max_iter):
        sigma_mid = (sigma_low + sigma_high) / 2
        price_mid = bsm_price(S, K, T, r, q, sigma_mid, flag)
        
        if abs(price_mid - market_price) < tol:
            return sigma_mid
        
        if price_mid < market_price:
            sigma_low = sigma_mid
        else:
            sigma_high = sigma_mid
    
    return sigma_mid


def bsm_greeks(S, K, T, r, q, sigma, flag):
    """计算BSM希腊字母"""
    if T <= 0 or sigma <= 0:
        return None
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    nd1 = norm.pdf(d1)
    
    if flag == 'c':
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (-S * np.exp(-q * T) * nd1 * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)
                 + q * S * np.exp(-q * T) * norm.cdf(d1))
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (-S * np.exp(-q * T) * nd1 * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)
                 - q * S * np.exp(-q * T) * norm.cdf(-d1))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    gamma = np.exp(-q * T) * nd1 / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * nd1 * np.sqrt(T)
    
    # Theta转日度，Vega/Rho转每1%
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta / 365,
        'vega': vega / 100,
        'rho': rho / 100
    }


class OptionsAnalyzer:
    """期权分析器"""
    
    def __init__(self, risk_free_rate=0.025):
        self.r = risk_free_rate
    
    def calculate_european_iv(self, S, K, T, price, flag='c', q=0.0):
        """计算欧式期权隐含波动率"""
        return bsm_iv(S, K, T, self.r, q, price, flag)
    
    def calculate_european_greeks(self, S, K, T, sigma, flag='c', q=0.0):
        """计算欧式期权希腊字母"""
        return bsm_greeks(S, K, T, self.r, q, sigma, flag)
    
    def calculate_american_greeks(self, S, K, T, r, sigma, flag='c', q=0.0):
        """用QuantLib计算美式期权希腊字母"""
        if not QUANTLIB_AVAILABLE:
            return None
        
        try:
            calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
            day_count = ql.Actual365Fixed()
            
            settlement = ql.Date.todaysDate()
            maturity = settlement + int(T * 365)
            
            spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
            flat_ts = ql.YieldTermStructureHandle(
                ql.FlatForward(settlement, r, day_count))
            dividend_ts = ql.YieldTermStructureHandle(
                ql.FlatForward(settlement, q, day_count))
            flat_vol = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(settlement, calendar, sigma, day_count))
            
            payoff = ql.PlainVanillaPayoff(
                ql.Option.Call if flag == 'c' else ql.Option.Put, K)
            exercise = ql.AmericanExercise(settlement, maturity)
            
            bsm_process = ql.BlackScholesMertonProcess(
                spot_handle, dividend_ts, flat_ts, flat_vol)
            
            option = ql.VanillaOption(payoff, exercise)
            option.setPricingEngine(ql.BinomialVanillaEngine(bsm_process, "crr", 100))
            
            greeks = {
                'price': option.NPV(),
                'delta': option.delta(),
                'gamma': option.gamma(),
            }
            # Theta和Rho美式期权可能不支持
            try:
                greeks['theta'] = option.theta()
            except:
                greeks['theta'] = None
            try:
                greeks['rho'] = option.rho()
            except:
                greeks['rho'] = None
            try:
                greeks['vega'] = option.vega()
            except:
                # 数值计算vega
                eps = 0.001
                option_up = ql.VanillaOption(payoff, exercise)
                vol_up = ql.BlackVolTermStructureHandle(
                    ql.BlackConstantVol(settlement, calendar, sigma + eps, day_count))
                process_up = ql.BlackScholesMertonProcess(
                    spot_handle, dividend_ts, flat_ts, vol_up)
                option_up.setPricingEngine(ql.BinomialVanillaEngine(process_up, "crr", 100))
                greeks['vega'] = (option_up.NPV() - option.NPV()) / eps
            
            return greeks
            
        except Exception as e:
            return None
    
    def build_volatility_surface(self, S, strikes, maturities, prices, flag='c', q=0.0):
        """构建波动率曲面"""
        surface = pd.DataFrame(
            index=[f'{k:.1f}' for k in strikes],
            columns=[f'{m:.2f}Y' for m in maturities]
        )
        
        for K in strikes:
            for T in maturities:
                price = prices.get((K, T))
                if price:
                    iv = self.calculate_european_iv(S, K, T, price, flag, q)
                    surface.loc[f'{K:.1f}', f'{T:.2f}Y'] = iv
        
        return surface
    
    def analyze_volatility_skew(self, S, strikes, prices, T, flag='c', q=0.0):
        """分析波动率偏斜"""
        ivs = {}
        for K in strikes:
            price = prices.get(K)
            if price:
                iv = self.calculate_european_iv(S, K, T, price, flag, q)
                if iv:
                    ivs[K] = iv
        
        if not ivs:
            return None
        
        atm_strike = min(strikes, key=lambda x: abs(x - S))
        atm_iv = ivs.get(atm_strike, np.mean(list(ivs.values())))
        
        # 偏斜指标
        otm_puts = {k: v for k, v in ivs.items() if k < S}
        otm_calls = {k: v for k, v in ivs.items() if k > S}
        
        skew = {
            'atm_iv': atm_iv,
            'atm_strike': atm_strike,
            'max_iv': max(ivs.values()),
            'min_iv': min(ivs.values()),
            'iv_range': max(ivs.values()) - min(ivs.values()),
            'put_skew': np.mean(list(otm_puts.values())) - atm_iv if otm_puts else None,
            'call_skew': np.mean(list(otm_calls.values())) - atm_iv if otm_calls else None,
            'ivs': ivs
        }
        
        return skew
    
    def calculate_term_structure(self, S, K, T_list, prices, flag='c', q=0.0):
        """计算波动率期限结构"""
        term_structure = {}
        for T in T_list:
            price = prices.get(T)
            if price:
                iv = self.calculate_european_iv(S, K, T, price, flag, q)
                if iv:
                    term_structure[T] = iv
        return term_structure


def demo():
    """演示"""
    analyzer = OptionsAnalyzer()
    
    S, K, T, price = 100.0, 100.0, 30/365, 2.5
    
    print("=" * 60)
    print("期权分析演示")
    print("=" * 60)
    
    # 欧式
    print("\n【欧式期权】")
    iv = analyzer.calculate_european_iv(S, K, T, price, 'c')
    print(f"隐含波动率: {iv:.4f} ({iv*100:.2f}%)")
    
    greeks = analyzer.calculate_european_greeks(S, K, T, iv, 'c')
    print(f"Delta: {greeks['delta']:.4f}")
    print(f"Gamma: {greeks['gamma']:.4f}")
    print(f"Theta(日): {greeks['theta']:.4f}")
    print(f"Vega(1%): {greeks['vega']:.4f}")
    print(f"Rho(1%): {greeks['rho']:.4f}")
    
    # 美式
    if QUANTLIB_AVAILABLE:
        print("\n【美式期权（QuantLib）】")
        american = analyzer.calculate_american_greeks(S, K, T, 0.025, iv, 'c')
        if american:
            print(f"价格: {american['price']:.4f}")
            print(f"Delta: {american['delta']:.4f}")
            print(f"Gamma: {american['gamma']:.4f}")
            print(f"Theta: {american['theta']:.4f}")
            print(f"Vega: {american['vega']:.4f}")
    
    # 偏斜
    print("\n【波动率偏斜】")
    strikes = [90, 95, 100, 105, 110]
    prices_dict = {90: 0.5, 95: 1.2, 100: 2.5, 105: 1.8, 110: 0.8}
    skew = analyzer.analyze_volatility_skew(S, strikes, prices_dict, T, 'c')
    if skew:
        print(f"ATM IV: {skew['atm_iv']:.4f}")
        print(f"Put偏斜: {skew['put_skew']:.4f}" if skew['put_skew'] else "Put偏斜: N/A")


if __name__ == '__main__':
    demo()
