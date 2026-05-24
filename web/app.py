#!/usr/bin/env python3
"""
期货分析平台 Web 服务 v3
- 期限结构：点击品种显示单个曲线
- 波动率曲面：改进展示
- 期权T型报价
- 明确标注模拟IV
"""

import os
import sys
import json
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, jsonify, render_template_string, request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analysis'))

app = Flask(__name__)

BASE_DIR = os.path.expanduser("~/home/futures_platform")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ============ 通用样式 ============
COMMON_HEAD = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>期货分析平台</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0e1a; color: #e0e0e0; line-height: 1.6; }
        .header { background: linear-gradient(135deg, #1a1f3a 0%, #0d1120 100%); padding: 20px 40px; border-bottom: 1px solid #2a3f5f; }
        .header h1 { font-size: 24px; color: #4fc3f7; }
        .header .subtitle { color: #78909c; font-size: 14px; }
        .container { padding: 20px 40px; }
        .nav { display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }
        .nav a { color: #4fc3f7; text-decoration: none; padding: 8px 16px; border-radius: 4px; }
        .nav a:hover, .nav a.active { background: #1e3a5f; }
        .card { background: #111827; border: 1px solid #1e3a5f; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        .card h3 { color: #4fc3f7; margin-bottom: 15px; font-size: 16px; }
        .card h3 .badge { background: #1e3a5f; color: #4fc3f7; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-left: 10px; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th { text-align: left; padding: 8px; color: #78909c; border-bottom: 1px solid #1e3a5f; }
        td { padding: 8px; border-bottom: 1px solid #1a2332; }
        tr:hover { background: #1a2332; }
        .up { color: #ef5350; }
        .down { color: #26a69a; }
        .neutral { color: #78909c; }
        .chart { width: 100%; height: 400px; margin: 20px 0; display: block; }
        .chart-large { width: 100%; height: 500px; margin: 20px 0; display: block; }
        .tabs { display: flex; gap: 10px; margin-bottom: 15px; border-bottom: 1px solid #1e3a5f; }
        .tab { padding: 10px 20px; cursor: pointer; color: #78909c; }
        .tab.active { color: #4fc3f7; border-bottom: 2px solid #4fc3f7; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .symbol-list { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 15px; max-height: 200px; overflow-y: auto; }
        .symbol-tag { padding: 4px 12px; background: #1a2332; border-radius: 4px; cursor: pointer; font-size: 12px; }
        .symbol-tag:hover { background: #1e3a5f; }
        .symbol-tag.active { background: #4fc3f7; color: #0a0e1a; }
        .warning { background: #332200; border: 1px solid #665500; color: #ffcc00; padding: 10px; border-radius: 4px; margin-bottom: 15px; font-size: 13px; }
        .itm { background: rgba(239, 83, 80, 0.15); }
        .otm { background: rgba(38, 166, 154, 0.15); }
        .atm { background: rgba(255, 193, 7, 0.2); }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 900px) { .grid-2 { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="header">
        <h1>期货分析平台</h1>
        <div class="subtitle">期限结构 | 波动率分析 | 期权希腊字母</div>
    </div>
    <div class="container">
        <div class="nav">
            <a href="/" class="{{ 'active' if active_page == 'home' else '' }}">总览</a>
            <a href="/term_structure" class="{{ 'active' if active_page == 'term' else '' }}">期限结构</a>
            <a href="/volatility" class="{{ 'active' if active_page == 'vol' else '' }}">波动率</a>
            <a href="/options" class="{{ 'active' if active_page == 'options' else '' }}">期权分析</a>
            <a href="/data_status" class="{{ 'active' if active_page == 'data' else '' }}">数据状态</a>
        </div>
"""

COMMON_FOOT = """
    </div>
    <script>
        function showTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
            
            // 延迟初始化图表（确保容器可见）
            setTimeout(function() {
                if (tabId === 'tab-greeks') initGreeksCharts();
                if (tabId === 'tab-surface') initSurfaceChart();
                if (tabId === 'tab-termiv') initTermIvChart();
                
                // 触发resize
                if (window.charts) {
                    Object.values(window.charts).forEach(function(c) {
                        if (c && c.resize) c.resize();
                    });
                }
            }, 200);
        }
    </script>
</body>
</html>
"""

# ============ 数据加载 ============

def load_term_structure_data():
    """加载期限结构数据"""
    files = glob.glob(os.path.join(DATA_DIR, "futures_term_structure", "*.json"))
    data = []
    for f in sorted(files):
        try:
            with open(f) as fp:
                d = json.load(fp)
                if isinstance(d, dict):
                    data.append(d)
        except:
            continue
    return data

def load_futures_data():
    """加载期货数据"""
    futures_dir = os.path.join(DATA_DIR, "futures_weighted")
    data = []
    for f in sorted(os.listdir(futures_dir)):
        if not f.endswith('.csv'):
            continue
        symbol = f.replace('.csv', '')
        try:
            df = pd.read_csv(os.path.join(futures_dir, f))
            if len(df) < 20:
                continue
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date')
            close = df['close'].values
            hv_20 = np.std(np.log(close[1:] / close[:-1])) * np.sqrt(252) * 100 if len(close) > 20 else 0
            hv_60 = np.std(np.log(close[1:] / close[:-1])) * np.sqrt(252) * 100 if len(close) > 60 else 0
            returns = np.log(close[-20:] / close[-21:-1])
            rv = np.std(returns) * np.sqrt(252) * 100 if len(returns) > 1 else 0
            data.append({
                'symbol': symbol, 'last_price': close[-1],
                'hv_20': hv_20, 'hv_60': hv_60, 'rv': rv,
                'date': df['trade_date'].iloc[-1].strftime('%Y-%m-%d'),
            })
        except:
            continue
    return data

def load_options_data():
    """加载天勤量化真实期权数据"""
    tq_dir = os.path.join(DATA_DIR, "tq_options")
    files = glob.glob(os.path.join(tq_dir, "*.json"))
    
    all_options = []
    for f in sorted(files, reverse=True):
        try:
            with open(f) as fp:
                data = json.load(fp)
                if isinstance(data, list):
                    all_options.extend(data)
        except:
            continue
    
    return all_options

# ============ 首页 ============

@app.route("/")
def home():
    vol_data = load_futures_data()
    term_data = load_term_structure_data()
    options_data = load_options_data()
    
    vol_sorted = sorted(vol_data, key=lambda x: x.get('hv_20', 0), reverse=True)
    top_vol = vol_sorted[:5]
    bottom_vol = vol_sorted[-5:]
    
    carry_signals = [t for t in term_data if abs(t.get('total_spread_pct', 0)) > 5]
    carry_signals = sorted(carry_signals, key=lambda x: abs(x.get('total_spread_pct', 0)), reverse=True)[:5]
    
    top_options = sorted(options_data, key=lambda x: x.get('hv_20', 0), reverse=True)[:5] if options_data else []
    
    # 获取数据日期
    data_date = 'N/A'
    if vol_data:
        data_date = vol_data[0].get('date', 'N/A')
    elif term_data:
        data_date = term_data[0].get('date', 'N/A')
    
    html = COMMON_HEAD + """
        <div class="grid-2">
            <div class="card">
                <h3>平台状态</h3>
                <table>
                    <tr><td>期货品种</td><td>""" + str(len(vol_data)) + """</td></tr>
                    <tr><td>期限结构</td><td>""" + str(len(term_data)) + """</td></tr>
                    <tr><td>期权合约</td><td>""" + str(len(options_data)) + """</td></tr>
                    <tr><td>数据日期</td><td>""" + data_date + """</td></tr>
                </table>
            </div>
            <div class="card">
                <h3>波动率最高 <span class="badge">TOP 5</span></h3>
                <table><tr><th>品种</th><th>HV20</th><th>HV60</th></tr>
                {% for item in top_vol %}
                <tr><td>{{ item.symbol }}</td><td class="up">{{ "%.1f"|format(item.hv_20) }}%</td><td>{{ "%.1f"|format(item.hv_60) }}%</td></tr>
                {% endfor %}
                </table>
            </div>
            <div class="card">
                <h3>波动率最低 <span class="badge">BOTTOM 5</span></h3>
                <table><tr><th>品种</th><th>HV20</th><th>HV60</th></tr>
                {% for item in bottom_vol %}
                <tr><td>{{ item.symbol }}</td><td class="down">{{ "%.1f"|format(item.hv_20) }}%</td><td>{{ "%.1f"|format(item.hv_60) }}%</td></tr>
                {% endfor %}
                </table>
            </div>
            <div class="card">
                <h3>期限结构Carry信号</h3>
                <table><tr><th>品种</th><th>结构</th><th>价差</th></tr>
                {% for item in carry_signals %}
                <tr><td>{{ item.symbol }}</td><td class="{{ 'up' if item.structure == 'contango' else 'down' }}">{{ item.structure }}</td><td>{{ "%.2f"|format(item.total_spread_pct) }}%</td></tr>
                {% endfor %}
                </table>
            </div>
        </div>
    """ + COMMON_FOOT
    
    return render_template_string(html, active_page='home', top_vol=top_vol, bottom_vol=bottom_vol,
                                  carry_signals=carry_signals)

# ============ 期限结构页（点击显示单个品种） ============

@app.route("/term_structure")
def term_structure():
    data = load_term_structure_data()
    
    # 获取所有品种列表
    symbols = []
    for d in data:
        curve = d.get('curve', [])
        if len(curve) >= 2:
            symbols.append({
                'symbol': d['symbol'],
                'name': d.get('name', d['symbol']),
                'structure': d.get('structure', 'unknown'),
                'spread_pct': d.get('total_spread_pct', 0),
                'near': curve[0]['symbol'] if curve else '',
                'far': curve[-1]['symbol'] if curve else '',
            })
    
    symbols.sort(key=lambda x: abs(x['spread_pct']), reverse=True)
    
    # 第一个品种的数据用于默认显示
    default_symbol = symbols[0]['symbol'] if symbols else ''
    default_data = next((d for d in data if d['symbol'] == default_symbol), None)
    
    html = COMMON_HEAD + """
        <div class="card">
            <h3>期限结构分析 <span class="badge">点击品种查看曲线</span></h3>
            <div class="symbol-list" id="symbolList">
                {% for s in symbols %}
                <span class="symbol-tag {{ 'active' if s.symbol == default_symbol else '' }}" 
                      onclick="showSymbol('{{ s.symbol }}')">
                    {{ s.symbol }} {{ "%.1f"|format(s.spread_pct) }}%
                </span>
                {% endfor %}
            </div>
            <div id="termChart" class="chart-large"></div>
            <div id="termInfo" style="margin-top:15px;"></div>
        </div>
        
        <div class="card">
            <h3>全品种期限结构</h3>
            <table>
                <tr><th>品种</th><th>结构</th><th>价差%</th><th>近月</th><th>近月价</th><th>远月</th><th>远月价</th></tr>
                {% for d in data %}
                {% set curve = d.curve if d.curve else [] %}
                {% if curve|length >= 2 %}
                <tr style="cursor:pointer" onclick="showSymbol('{{ d.symbol }}')">
                    <td><b>{{ d.symbol }}</b></td>
                    <td class="{{ 'up' if d.structure == 'contango' else 'down' }}">{{ d.structure }}</td>
                    <td>{{ "%.2f"|format(d.total_spread_pct) }}%</td>
                    <td>{{ curve[0].symbol }}</td>
                    <td>{{ curve[0].price }}</td>
                    <td>{{ curve[-1].symbol }}</td>
                    <td>{{ curve[-1].price }}</td>
                </tr>
                {% endif %}
                {% endfor %}
            </table>
        </div>
        
        <script>
            var termData = {{ data|tojson }};
            var defaultSymbol = '{{ default_symbol }}';
            
            function showSymbol(symbol) {
                // 更新标签状态
                document.querySelectorAll('.symbol-tag').forEach(t => t.classList.remove('active'));
                event.target.classList.add('active');
                
                // 找到数据
                var d = termData.find(x => x.symbol === symbol);
                if (!d || !d.curve || d.curve.length < 2) return;
                
                // 更新图表
                var chart = echarts.init(document.getElementById('termChart'));
                var xData = d.curve.map(c => c.symbol);
                var yData = d.curve.map(c => c.price);
                
                chart.setOption({
                    title: { text: symbol + ' 期限结构 (' + d.structure + ')', left: 'center', textStyle: { color: '#e0e0e0' } },
                    tooltip: { trigger: 'axis' },
                    grid: { left: '10%', right: '10%', bottom: '15%' },
                    xAxis: { type: 'category', data: xData, axisLabel: { color: '#78909c', rotate: 30 } },
                    yAxis: { type: 'value', axisLabel: { color: '#78909c' }, splitLine: { lineStyle: { color: '#1e3a5f' } } },
                    series: [{
                        data: yData,
                        type: 'line',
                        smooth: true,
                        symbol: 'circle',
                        symbolSize: 10,
                        lineStyle: { width: 3, color: d.structure === 'contango' ? '#ef5350' : '#26a69a' },
                        itemStyle: { color: d.structure === 'contango' ? '#ef5350' : '#26a69a' },
                        areaStyle: {
                            color: {
                                type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
                                colorStops: [
                                    { offset: 0, color: d.structure === 'contango' ? 'rgba(239,83,80,0.3)' : 'rgba(38,166,154,0.3)' },
                                    { offset: 1, color: 'transparent' }
                                ]
                            }
                        },
                        label: { show: true, position: 'top', color: '#e0e0e0', formatter: '{c}' }
                    }]
                });
                
                // 更新信息
                var info = document.getElementById('termInfo');
                info.innerHTML = '<b>' + symbol + '</b> | 结构: <span style="color:' + 
                    (d.structure === 'contango' ? '#ef5350' : '#26a69a') + '">' + d.structure + 
                    '</span> | 价差: ' + d.total_spread_pct.toFixed(2) + '% | 日期: ' + d.date;
            }
            
            // 默认显示第一个
            if (defaultSymbol) {
                setTimeout(() => showSymbol(defaultSymbol), 100);
            }
        </script>
    """ + COMMON_FOOT
    
    return render_template_string(html, active_page='term', symbols=symbols, 
                                  data=data, default_symbol=default_symbol)

# ============ 波动率页 ============

@app.route("/volatility")
def volatility():
    data = load_futures_data()
    data.sort(key=lambda x: x.get('hv_20', 0), reverse=True)
    
    # 波动率分布数据
    vol_ranges = {'<10%': 0, '10-20%': 0, '20-30%': 0, '30-40%': 0, '>40%': 0}
    for d in data:
        hv = d.get('hv_20', 0)
        if hv < 10: vol_ranges['<10%'] += 1
        elif hv < 20: vol_ranges['10-20%'] += 1
        elif hv < 30: vol_ranges['20-30%'] += 1
        elif hv < 40: vol_ranges['30-40%'] += 1
        else: vol_ranges['>40%'] += 1
    
    html = COMMON_HEAD + """
        <div class="card">
            <h3>波动率分布</h3>
            <div id="volDistChart" class="chart"></div>
        </div>
        <div class="card">
            <h3>全品种波动率排名 <span class="badge">{{ data|length }}个品种</span></h3>
            <table>
                <tr><th>排名</th><th>品种</th><th>最新价</th><th>HV20</th><th>HV60</th><th>RV20</th><th>状态</th></tr>
                {% for item in data %}
                <tr>
                    <td>{{ loop.index }}</td>
                    <td><b>{{ item.symbol }}</b></td>
                    <td>{{ "%.2f"|format(item.last_price) }}</td>
                    <td class="{{ 'up' if item.hv_20 > 30 else 'neutral' }}">{{ "%.1f"|format(item.hv_20) }}%</td>
                    <td>{{ "%.1f"|format(item.hv_60) }}%</td>
                    <td>{{ "%.1f"|format(item.rv) }}%</td>
                    <td class="{{ 'up' if item.hv_20 > item.hv_60 else 'down' }}">{{ "高" if item.hv_20 > item.hv_60 else "低" }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        <script>
            var chart = echarts.init(document.getElementById('volDistChart'));
            chart.setOption({
                title: { text: '波动率分布', left: 'center', textStyle: { color: '#e0e0e0' } },
                tooltip: { trigger: 'axis' },
                xAxis: { type: 'category', data: {{ vol_ranges.keys()|list|tojson }}, axisLabel: { color: '#78909c' } },
                yAxis: { type: 'value', axisLabel: { color: '#78909c' }, splitLine: { lineStyle: { color: '#1e3a5f' } } },
                series: [{
                    data: {{ vol_ranges.values()|list|tojson }},
                    type: 'bar',
                    itemStyle: {
                        color: {
                            type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
                            colorStops: [
                                { offset: 0, color: '#4fc3f7' },
                                { offset: 1, color: '#1e3a5f' }
                            ]
                        }
                    },
                    label: { show: true, position: 'top', color: '#e0e0e0' }
                }]
            });
        </script>
    """ + COMMON_FOOT
    
    return render_template_string(html, active_page='vol', data=data, vol_ranges=vol_ranges)

# ============ 期权分析页 ============

@app.route("/options")
def options():
    data = load_options_data()
    
    # 按品种分组
    products = {}
    for d in data:
        product = d.get('product', '')
        if product not in products:
            products[product] = []
        products[product].append(d)
    
    # 选择第一个品种展示
    sample_product = list(products.keys())[0] if products else ''
    sample_options = products.get(sample_product, [])
    
    # T型报价
    t_quote = []
    spot_price = 0
    
    if sample_options:
        spot_price = sample_options[0].get('underlying_price', 0)
        
        strikes = {}
        for opt in sample_options:
            k = opt.get('strike_price', 0)
            if k not in strikes:
                strikes[k] = {'call': None, 'put': None}
            if opt.get('option_class') == 'CALL':
                strikes[k]['call'] = opt
            else:
                strikes[k]['put'] = opt
        
        for strike in sorted(strikes.keys()):
            call = strikes[strike]['call']
            put = strikes[strike]['put']
            moneyness = spot_price / strike if strike > 0 else 1
            
            t_quote.append({
                'strike': strike,
                'moneyness': moneyness,
                'call_price': call.get('last_price', 0) if call else 0,
                'call_iv': call.get('implied_volatility', 0) if call else 0,
                'call_delta': call.get('delta', 0) if call else 0,
                'call_gamma': call.get('gamma', 0) if call else 0,
                'call_vega': call.get('vega', 0) if call else 0,
                'put_price': put.get('last_price', 0) if put else 0,
                'put_iv': put.get('implied_volatility', 0) if put else 0,
                'put_delta': put.get('delta', 0) if put else 0,
                'put_gamma': put.get('gamma', 0) if put else 0,
                'put_vega': put.get('vega', 0) if put else 0,
            })
    
    # IV排名
    iv_ranking = []
    for product, opts in products.items():
        if opts:
            max_iv_opt = max(opts, key=lambda x: x.get('implied_volatility', 0))
            iv_ranking.append({
                'symbol': product,
                'iv': max_iv_opt.get('implied_volatility', 0),
                'strike': max_iv_opt.get('strike_price', 0),
                'option_class': max_iv_opt.get('option_class', ''),
            })
    iv_ranking.sort(key=lambda x: x['iv'], reverse=True)
    
    # IV曲面数据 - 按品种分组，取Call的IV
    surface_data = []
    for product, opts in products.items():
        calls = [o for o in opts if o.get('option_class') == 'CALL']
        for c in calls:
            if spot_price > 0:
                moneyness = (c.get('strike_price', 0) / spot_price - 1) * 100
                surface_data.append({
                    'product': product,
                    'moneyness': round(moneyness, 1),
                    'iv': c.get('implied_volatility', 0),
                    'strike': c.get('strike_price', 0)
                })
    
    # IV期限结构 - 按品种取不同行权价的IV
    term_iv_data = {}
    for product, opts in products.items():
        calls = sorted([o for o in opts if o.get('option_class') == 'CALL'], 
                      key=lambda x: x.get('strike_price', 0))
        if calls:
            # 取低、中、高三个行权价的IV
            n = len(calls)
            if n >= 3:
                ivs = [calls[0].get('implied_volatility', 0),
                       calls[n//2].get('implied_volatility', 0),
                       calls[-1].get('implied_volatility', 0)]
            else:
                ivs = [c.get('implied_volatility', 0) for c in calls]
            term_iv_data[product] = ivs
    
    # 希腊字母曲线数据
    greeks_data = {
        'strikes': [r['strike'] for r in t_quote],
        'call_delta': [r['call_delta'] for r in t_quote],
        'put_delta': [r['put_delta'] for r in t_quote],
        'call_gamma': [r['call_gamma'] for r in t_quote],
        'put_gamma': [r['put_gamma'] for r in t_quote],
        'call_vega': [r['call_vega'] for r in t_quote],
        'put_vega': [r['put_vega'] for r in t_quote],
    }
    
    html = COMMON_HEAD + """
        <div class="warning" style="background:#1a3a1a;border-color:#2e7d32;color:#81c784">
            <b>数据来源：</b>天勤量化(TqSdk)真实期权行情 | IV通过BSM模型从市场价格反推
        </div>
        
        <div class="tabs">
            <div class="tab active" onclick="showTab('tab-tquote')">T型报价</div>
            <div class="tab" onclick="showTab('tab-surface')">IV曲面</div>
            <div class="tab" onclick="showTab('tab-termiv')">IV期限结构</div>
            <div class="tab" onclick="showTab('tab-greeks')">希腊字母</div>
            <div class="tab" onclick="showTab('tab-ranking')">IV排名</div>
        </div>
        
        <div id="tab-tquote" class="tab-content active">
            <div class="card">
                <h3>T型报价 <span class="badge">{{ sample_product }} 标的={{ "%.2f"|format(spot_price) }}</span></h3>
                <table>
                    <tr>
                        <th colspan="5" style="text-align:center;color:#ef5350">CALL (买权)</th>
                        <th style="text-align:center;background:#1e3a5f">行权价</th>
                        <th colspan="5" style="text-align:center;color:#26a69a">PUT (卖权)</th>
                    </tr>
                    <tr>
                        <th>价格</th><th>IV%</th><th>Delta</th><th>Gamma</th><th>Vega</th>
                        <th style="text-align:center;background:#1e3a5f">K</th>
                        <th>价格</th><th>IV%</th><th>Delta</th><th>Gamma</th><th>Vega</th>
                    </tr>
                    {% for row in t_quote %}
                    <tr class="{{ 'atm' if 0.97 < row.moneyness < 1.03 else ('itm' if row.moneyness >= 1.03 else 'otm') }}">
                        <td>{{ "%.1f"|format(row.call_price) }}</td>
                        <td>{{ "%.1f"|format(row.call_iv) }}%</td>
                        <td>{{ "%.3f"|format(row.call_delta) }}</td>
                        <td>{{ "%.4f"|format(row.call_gamma) }}</td>
                        <td>{{ "%.3f"|format(row.call_vega) }}</td>
                        <td style="text-align:center;background:#1e3a5f;font-weight:bold">{{ "%.0f"|format(row.strike) }}</td>
                        <td>{{ "%.1f"|format(row.put_price) }}</td>
                        <td>{{ "%.1f"|format(row.put_iv) }}%</td>
                        <td>{{ "%.3f"|format(row.put_delta) }}</td>
                        <td>{{ "%.4f"|format(row.put_gamma) }}</td>
                        <td>{{ "%.3f"|format(row.put_vega) }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
        </div>
        
        <div id="tab-surface" class="tab-content">
            <div class="card">
                <h3>IV波动率曲面 <span class="badge">点击品种查看</span></h3>
                <div class="symbol-list" id="surfaceProductList"></div>
                <div id="surfaceChart" class="chart-large"></div>
            </div>
        </div>
        
        <div id="tab-termiv" class="tab-content">
            <div class="card">
                <h3>IV期限结构 <span class="badge">真实数据</span></h3>
                <div id="termIvChart" class="chart-large"></div>
            </div>
        </div>
        
        <div id="tab-greeks" class="tab-content">
            <div class="card">
                <h3>Delta曲线 <span class="badge">{{ sample_product }}</span></h3>
                <div id="deltaChart" class="chart"></div>
            </div>
            <div class="card">
                <h3>Gamma曲线 <span class="badge">{{ sample_product }}</span></h3>
                <div id="gammaChart" class="chart"></div>
            </div>
            <div class="card">
                <h3>Vega曲线 <span class="badge">{{ sample_product }}</span></h3>
                <div id="vegaChart" class="chart"></div>
            </div>
        </div>
        
        <div id="tab-ranking" class="tab-content">
            <div class="card">
                <h3>IV排名 <span class="badge">真实数据</span></h3>
                <table>
                    <tr><th>排名</th><th>品种</th><th>最高IV</th><th>合约</th><th>行权价</th></tr>
                    {% for item in iv_ranking %}
                    <tr>
                        <td>{{ loop.index }}</td>
                        <td><b>{{ item.symbol }}</b></td>
                        <td class="up">{{ "%.1f"|format(item.iv) }}%</td>
                        <td>{{ item.option_class }}</td>
                        <td>{{ "%.0f"|format(item.strike) }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
        </div>
        
        <script>
            // 图表实例存储
            var charts = {};
            
            // 初始化希腊字母图表（默认tab可见）
            function initGreeksCharts() {
                if (charts.delta) return; // 已初始化
                
                var greeksData = {{ greeks_data|tojson }};
                
                // 设置图表容器宽度
                document.querySelectorAll('.chart, .chart-large').forEach(function(el) {
                    el.style.width = '100%';
                });
                
                // Delta
                charts.delta = echarts.init(document.getElementById('deltaChart'));
                charts.delta.setOption({
                    title: { text: 'Delta', left: 'center', textStyle: { color: '#e0e0e0' } },
                    tooltip: { trigger: 'axis' },
                    legend: { data: ['Call Delta', 'Put Delta'], top: 30, textStyle: { color: '#78909c' } },
                    xAxis: { type: 'category', data: greeksData.strikes.map(k => k.toFixed(0)), name: '行权价', nameTextStyle: { color: '#78909c' }, axisLabel: { color: '#78909c' } },
                    yAxis: { type: 'value', name: 'Delta', axisLabel: { color: '#78909c' }, splitLine: { lineStyle: { color: '#1e3a5f' } } },
                    series: [
                        { name: 'Call Delta', type: 'line', data: greeksData.call_delta, smooth: true, lineStyle: { color: '#ef5350', width: 2 } },
                        { name: 'Put Delta', type: 'line', data: greeksData.put_delta, smooth: true, lineStyle: { color: '#26a69a', width: 2 } }
                    ]
                });
                
                // Gamma
                charts.gamma = echarts.init(document.getElementById('gammaChart'));
                charts.gamma.setOption({
                    title: { text: 'Gamma', left: 'center', textStyle: { color: '#e0e0e0' } },
                    tooltip: { trigger: 'axis' },
                    legend: { data: ['Call Gamma', 'Put Gamma'], top: 30, textStyle: { color: '#78909c' } },
                    xAxis: { type: 'category', data: greeksData.strikes.map(k => k.toFixed(0)), name: '行权价', nameTextStyle: { color: '#78909c' }, axisLabel: { color: '#78909c' } },
                    yAxis: { type: 'value', name: 'Gamma', axisLabel: { color: '#78909c' }, splitLine: { lineStyle: { color: '#1e3a5f' } } },
                    series: [
                        { name: 'Call Gamma', type: 'line', data: greeksData.call_gamma, smooth: true, lineStyle: { color: '#ef5350', width: 2 } },
                        { name: 'Put Gamma', type: 'line', data: greeksData.put_gamma, smooth: true, lineStyle: { color: '#26a69a', width: 2 } }
                    ]
                });
                
                // Vega
                charts.vega = echarts.init(document.getElementById('vegaChart'));
                charts.vega.setOption({
                    title: { text: 'Vega', left: 'center', textStyle: { color: '#e0e0e0' } },
                    tooltip: { trigger: 'axis' },
                    legend: { data: ['Call Vega', 'Put Vega'], top: 30, textStyle: { color: '#78909c' } },
                    xAxis: { type: 'category', data: greeksData.strikes.map(k => k.toFixed(0)), name: '行权价', nameTextStyle: { color: '#78909c' }, axisLabel: { color: '#78909c' } },
                    yAxis: { type: 'value', name: 'Vega', axisLabel: { color: '#78909c' }, splitLine: { lineStyle: { color: '#1e3a5f' } } },
                    series: [
                        { name: 'Call Vega', type: 'line', data: greeksData.call_vega, smooth: true, lineStyle: { color: '#ef5350', width: 2 } },
                        { name: 'Put Vega', type: 'line', data: greeksData.put_vega, smooth: true, lineStyle: { color: '#26a69a', width: 2 } }
                    ]
                });
            }
            
            // 初始化IV曲面
            function initSurfaceChart() {
                if (charts.surface) return;
                
                var surfaceData = {{ surface_data|tojson }};
                if (surfaceData.length === 0) return;
                
                var products = [...new Set(surfaceData.map(d => d.product))];
                var currentProduct = products[0];
                
                // 创建品种选择标签
                var listEl = document.getElementById('surfaceProductList');
                listEl.innerHTML = '';
                products.forEach(function(prod) {
                    var tag = document.createElement('span');
                    tag.className = 'symbol-tag' + (prod === currentProduct ? ' active' : '');
                    tag.textContent = prod;
                    tag.onclick = function() {
                        document.querySelectorAll('#surfaceProductList .symbol-tag').forEach(t => t.classList.remove('active'));
                        this.classList.add('active');
                        currentProduct = prod;
                        updateSurfaceChart();
                    };
                    listEl.appendChild(tag);
                });
                
                charts.surface = echarts.init(document.getElementById('surfaceChart'));
                
                function updateSurfaceChart() {
                    var prodData = surfaceData.filter(d => d.product === currentProduct);
                    var calls = prodData.filter(d => d.iv > 0);
                    
                    // 按行权价排序
                    calls.sort((a, b) => a.strike - b.strike);
                    
                    var strikes = calls.map(d => d.strike);
                    var ivs = calls.map(d => d.iv);
                    var moneyness = calls.map(d => d.moneyness);
                    
                    charts.surface.setOption({
                        title: { 
                            text: currentProduct + ' IV微笑曲线', 
                            left: 'center', 
                            textStyle: { color: '#e0e0e0', fontSize: 16 } 
                        },
                        tooltip: { 
                            trigger: 'axis',
                            formatter: function(params) {
                                var p = params[0];
                                return '行权价: ' + p.name + '<br>IV: ' + p.value.toFixed(1) + '%<br>偏移: ' + moneyness[p.dataIndex].toFixed(1) + '%';
                            }
                        },
                        grid: { left: '10%', right: '10%', bottom: '15%' },
                        xAxis: { 
                            type: 'category', 
                            data: strikes.map(k => k.toFixed(0)),
                            name: '行权价', 
                            nameTextStyle: { color: '#78909c' }, 
                            axisLabel: { color: '#78909c', rotate: 45 }
                        },
                        yAxis: { 
                            type: 'value', 
                            name: 'IV%', 
                            nameTextStyle: { color: '#78909c' }, 
                            axisLabel: { color: '#78909c' }, 
                            splitLine: { lineStyle: { color: '#1e3a5f' } } 
                        },
                        series: [{
                            name: 'IV',
                            type: 'line',
                            data: ivs,
                            smooth: true,
                            symbol: 'circle',
                            symbolSize: 10,
                            lineStyle: { width: 3, color: '#4fc3f7' },
                            itemStyle: { color: '#4fc3f7' },
                            areaStyle: {
                                color: {
                                    type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
                                    colorStops: [
                                        { offset: 0, color: 'rgba(79,195,247,0.3)' },
                                        { offset: 1, color: 'transparent' }
                                    ]
                                }
                            }
                        }]
                    });
                }
                
                updateSurfaceChart();
            }
            
            // 初始化IV期限结构
            function initTermIvChart() {
                if (charts.termIv) return;
                
                var termIvData = {{ term_iv_data|tojson }};
                if (Object.keys(termIvData).length > 0) {
                    charts.termIv = echarts.init(document.getElementById('termIvChart'));
                    var series = [];
                    var colors = ['#4fc3f7', '#ef5350', '#26a69a', '#ffeb3b', '#ab47bc'];
                    var idx = 0;
                    
                    for (var sym in termIvData) {
                        if (idx >= 15) break;
                        series.push({
                            name: sym,
                            type: 'line',
                            data: termIvData[sym],
                            smooth: true,
                            lineStyle: { width: 2 },
                            itemStyle: { color: colors[idx % colors.length] }
                        });
                        idx++;
                    }
                    
                    charts.termIv.setOption({
                        title: { text: 'IV期限结构 (低-中-高行权价)', left: 'center', textStyle: { color: '#e0e0e0' } },
                        tooltip: { trigger: 'axis' },
                        legend: { data: Object.keys(termIvData).slice(0, 15), top: 30, textStyle: { color: '#78909c' }, type: 'scroll' },
                        xAxis: { type: 'category', data: ['低行权价', '中行权价', '高行权价'], axisLabel: { color: '#78909c' } },
                        yAxis: { type: 'value', name: 'IV%', axisLabel: { color: '#78909c' }, splitLine: { lineStyle: { color: '#1e3a5f' } } },
                        series: series
                    });
                }
            }
            
            // 页面加载时初始化希腊字母图表（默认可见）
            initGreeksCharts();
            
            // 窗口大小改变时重绘
            window.addEventListener('resize', function() {
                for (var key in charts) {
                    if (charts[key]) charts[key].resize();
                }
            });
        </script>
    """ + COMMON_FOOT
    
    return render_template_string(html, active_page='options', t_quote=t_quote,
                                  spot_price=spot_price, sample_product=sample_product,
                                  iv_ranking=iv_ranking, surface_data=surface_data,
                                  term_iv_data=term_iv_data, greeks_data=greeks_data)

# ============ 数据状态页 ============

@app.route("/data_status")
def data_status():
    futures_dir = os.path.join(DATA_DIR, "futures_weighted")
    futures_files = sorted([f for f in os.listdir(futures_dir) if f.endswith('.csv')])
    
    term_files = sorted(glob.glob(os.path.join(DATA_DIR, "futures_term_structure", "*.json")))
    
    options_files = sorted(glob.glob(os.path.join(DATA_DIR, "options", "*_options.json")))
    
    futures_info = []
    for f in futures_files:
        symbol = f.replace('.csv', '')
        try:
            df = pd.read_csv(os.path.join(futures_dir, f))
            futures_info.append({
                'symbol': symbol,
                'records': len(df),
                'date_range': f"{df['trade_date'].min()} ~ {df['trade_date'].max()}" if 'trade_date' in df.columns else 'N/A',
            })
        except:
            futures_info.append({'symbol': symbol, 'records': 0, 'date_range': 'Error'})
    
    html = COMMON_HEAD + """
        <div class="tabs">
            <div class="tab active" onclick="showTab('tab-futures')">期货数据 ({{ futures_info|length }})</div>
            <div class="tab" onclick="showTab('tab-term')">期限结构 ({{ term_files|length }})</div>
            <div class="tab" onclick="showTab('tab-options')">期权数据 ({{ options_files|length }})</div>
        </div>
        
        <div id="tab-futures" class="tab-content active">
            <div class="card">
                <table>
                    <tr><th>品种</th><th>记录数</th><th>日期范围</th></tr>
                    {% for item in futures_info %}
                    <tr><td>{{ item.symbol }}</td><td>{{ item.records }}</td><td>{{ item.date_range }}</td></tr>
                    {% endfor %}
                </table>
            </div>
        </div>
        
        <div id="tab-term" class="tab-content">
            <div class="card">
                <table>
                    <tr><th>文件名</th></tr>
                    {% for f in term_files %}
                    <tr><td>{{ f.split('/')[-1] }}</td></tr>
                    {% endfor %}
                </table>
            </div>
        </div>
        
        <div id="tab-options" class="tab-content">
            <div class="card">
                <table>
                    <tr><th>文件名</th></tr>
                    {% for f in options_files %}
                    <tr><td>{{ f.split('/')[-1] }}</td></tr>
                    {% endfor %}
                </table>
            </div>
        </div>
    """ + COMMON_FOOT
    
    return render_template_string(html, active_page='data', futures_info=futures_info,
                                  term_files=term_files, options_files=options_files)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
