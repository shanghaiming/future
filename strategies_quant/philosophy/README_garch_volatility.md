# GARCH Volatility Strategy

基于GARCH(1,1)的波动率均值回归策略，利用波动率的聚类效应和均值回归特性进行交易。

## 核心原理

### 1. GARCH(1,1) 模型
```
σ²_t = ω + α * r_{t-1}^2 + β * σ_{t-1}^2
```
其中：
- σ²_t：t时刻的条件方差
- ω：常数项（长期平均方差）
- α：ARCH项（新息冲击的影响）
- β：GARCH项（波动率持续性）
- α + β < 1 保证平稳性（均值回归）

### 2. 交易逻辑
- **高波动状态**（条件波动率/无条件波动率 > 1.5）：预期波动率收缩，卖出
- **低波动状态**（条件波动率/无条件波动率 < 0.7）：预期波动率扩张，买入（考虑向上的趋势偏移）
- **正常状态**：持有

### 3. 关键特性
- **仓位管理**：仓位大小与条件波动率成反比（低波动=大仓位）
- **风险控制**：使用GARCH预测波动率作为ATR-like跟踪止损
- **过滤机制**：通过结构张力指标过滤信号，避免在无明显方向的市场中交易

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| garch_window | 126 | GARCH参数估计的滚动窗口大小（约6个月） |
| high_vol_threshold | 1.5 | 高波动率阈值 |
| low_vol_threshold | 0.7 | 低波动率阈值 |
| max_position | 1.0 | 最大仓位比例 |
| position_scale | 0.5 | 仓位缩放因子 |
| atr_multiplier | 2.0 | ATR跟踪止损倍数 |
| atr_period | 14 | ATR计算周期 |
| tension_threshold | 0.3 | 结构张力阈值 |

## 使用方法

```python
from strategies.philosophy.garch_volatility_strategy import GARCHVolatilityStrategy
import pandas as pd

# 准备数据（必须有open, high, low, close列）
data = pd.read_csv('your_data.csv', index_col='timestamp', parse_dates=True)

# 创建策略
strategy = GARCHVolatilityStrategy(data)

# 生成信号
signals = strategy.generate_signals()

# 查看信号摘要
summary = strategy.get_signals_summary()
print(f"信号摘要: {summary}")

# 获取策略指标
metrics = strategy.get_strategy_metrics()
print(f"策略指标: {metrics}")
```

## 策略特点

### 优点
1. **理论基础扎实**：基于金融时间序列的波动率聚类和均值回归特性
2. **参数稳健**：使用滚动窗口自适应参数，适应不同市场状态
3. **风险可控**：基于GARCH预测的波动率做止损，动态调整风险
4. **过滤噪声**：结构张力过滤避免在震荡市场中的无效交易

### 注意事项
1. **数据要求**：需要至少126天的数据用于初始GARCH估计
2. **计算复杂度**：滚动窗口的最大似然估计计算成本较高
3. **参数敏感性**：波动率阈值需要根据具体品种调整
4. **趋势配合**：纯波动率策略需要配合方向信号使用效果更佳

## 回测建议

1. **参数调优**：调整`garch_window`以适应不同品种
2. **品种选择**：适合波动率有明显聚类效应的品种
3. **组合使用**：建议与其他趋势策略组合，作为仓位管理工具
4. **风险控制**：严格使用G预测的波动率作为止损

## 数学原理

波动率的均值回归特性源于GARCH模型的平稳性条件：
- 当α + β接近1时，波动率持续性高
- 当α + β接近0时，波动率变化快
- 半衰期公式：τ = -ln2 / ln(α + β)

本策略利用这一特性，在波动率极高时卖出（预期收缩），在波动率极低时买入（预期扩张）。