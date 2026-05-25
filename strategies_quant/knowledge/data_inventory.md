# 数据资源清单

## 1. 主力连续合约 (futures_daily/)
- **74个品种**，命名格式：AG0(银), RB0(螺纹), CU0(铜)等
- 字段：ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount, oi
- **价格可交易**，但OI只有最近几天有值
- 覆盖：大部分从2005年开始，最新到2026-05-25

## 2. 加权指数 (futures_weighted/)  
- **73个品种**，命名格式：agfi, rbfi, cufi等
- 字段：ts_code, trade_date, open, high, low, close, vol, amount, oi
- **OI完整**（2016-2026全覆盖），但价格不可交易
- data_loader自动合并：主力价格 + 加权OI

## 3. 期限结构 (futures_term_structure/)
- **82,182个JSON文件**，按品种+日期组织
- 每个文件包含：structure(contango/backwardation), curve(远期曲线), spread, spread_pct
- 覆盖：rbfi有1303天数据（~5年），其他品种类似
- 关键因子：近远月价差、contango/backwardation状态、曲线形状

## 4. 期权数据 (tq_options/)
- **207个文件**，覆盖66个品种
- 每条记录包含：strike, IV, delta, gamma, theta, vega, OI, volume
- 用途：波动率分析、Greeks信号、IV-HV偏离
- 注意：很多IV/Greeks为0（数据质量问题）

## 5. 计算期权 (options_calculated/)
- 236个文件，每文件约43条记录
- 可能是加工后的期权数据

## 数据加载方式
```python
from core.data_loader import load_stock_data, list_available_symbols
# symbol用加权命名(agfi)，实际加载主力合约价格+加权OI
df = load_stock_data('agfi')  # 加载AG0.csv价格，补充加权OI
syms = list_available_symbols('daily')  # 返回加权命名列表
```

## Symbol映射
| 主力合约 | 加权指数 | 品种 |
|----------|----------|------|
| AG0 | agfi | 白银 |
| RB0 | rbfi | 螺纹钢 |
| CU0 | cufi | 铜 |
| I0 | ifi | 铁矿石 |
| AU0 | aufi | 黄金 |
| J0 | jfi | 焦炭 |
| JM0 | jmfi | 焦煤 |
| HC0 | hcfi | 热卷 |
| M0 | mfi | 豆粕 |
| C0 | cfi | 玉米 |
