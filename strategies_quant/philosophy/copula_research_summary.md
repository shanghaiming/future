# Copula Strategy Research Summary

## 研究背景
基于概率论中的 Copula 理论，研究如何利用尾部依赖检测来识别市场状态变化。

## 核心理论要点

### 1. Copula 理论基础
- **定义**: Copula 是将多个随机变量的边缘分布连接到联合分布的函数
- **Sklar 定理**: 任何联合分布 F(x₁,...,xₙ) = C(F₁(x₁),...,Fₙ(xₙ))
- **优势**: 捕获非线性依赖关系，超越 Pearson 相关性的局限

### 2. 尾部依赖概念
- **上尾部依赖 λᵤ**: P(U₂ > F₂⁻¹(q) | U₁ > F₁⁻¹(q)), q→1
  - 两策略同时在极端盈利区的概率
  - 对应牛市环境

- **下尾部依赖 λₗ**: P(U₂ ≤ F₂⁻¹(q) | U₁ ≤ F₂⁻¹(q)), q→0  
  - 两策略同时亏损的概率
  - 对应熊市环境

### 3. 主要 Copula 类型
| Copula 类型 | 依赖特征 | 适用场景 |
|------------|---------|---------|
| Gaussian | 对称，无尾部依赖 | 正常市场 |
| Clayton | 下尾依赖强 | 熊市保护 |
| Gumbel | 上尾依赖强 | 牛市加速 |
| Frank | 对称但无尾部 | 中性市场 |
| Student-t | 对称，对称尾依赖 | 极端行情 |

## 策略实现

### 策略设计
1. **检测对象**: 个股收益与市场指数的尾部依赖关系
2. **牛市信号**: 上尾部依赖增强 → Gumbel 参数上升
3. **熊市信号**: 下尾部依赖增强 → Clayton 参数上升
4. **风险控制**: ATR trailing stop，根据 regime 调整止损宽松度

### 关键算法
1. **经验 CDF**: 将收益率转换为均匀分布 [0,1]
2. **Kendall's τ**: 衡量秩相关性，稳健于非线性关系
3. **尾部依赖估计**: 通过条件概率计算尾部依赖强度
4. **Copula 参数估计**: 
   - Clayton: θ = 2τ/(1-τ)
   - Gumbel: θ = 1/(1-τ)

### 参数设置
- copula_window: 252天（约1年）
- tau_window: 63天（约3个月）
- tail_threshold: 0.1（尾部依赖阈值）
- position_size_factor: 0.3（基础仓位）
- atr_multiplier: 2.0（ATR 倍数）

### Regime 判断逻辑
```python
if upper_strength > threshold and upper_strength > |lower_strength|:
    return 'bull'  # 牛市 regime
elif lower_strength > threshold and lower_strength > |upper_strength|:
    return 'bear'  # 熊市 regime
else:
    return 'neutral'  # 中性
```

## 实现特点

### 1. 鲁棒性设计
- 提供 copulae 包的简化实现作为 fallback
- 使用经验 CDF 避免分布假设错误
- 处理缺失值和数据长度不足的情况

### 2. 动态风险控制
- 牛市：宽松止损（ATR × 2.0）
- 熊市：紧止损（ATR × 1.4）
- 中性：标准止损

### 3. 仓位管理
- 基于尾部依赖强度动态调整仓位
- 牛市：增加仓位（1 + tail_strength × 2）
- 熊市：减少仓位（1 - tail_strength）

## 测试结果
使用模拟数据测试生成：
- 买入信号：牛市 regime 检测到
- 尾部依赖：下 0.360，上 0.406
- Kendall τ：0.221（中等正相关）
- 建议仓位：54.33%
- 止损价：12.49

## 理论意义

### 1. 克服 Pearson 相关性局限
- 传统相关性只捕捉线性关系
- Copula 能捕获极端事件的同时发生
- 更适合金融市场的肥尾特性

### 2. 市场状态量化
- 将抽象的市场状态（牛市/熊市）数学化
- 通过尾部依赖强度量化状态转换
- 提供概率化的 regime 判断

### 3. 策略组合优化
- 检测策略间的尾部依赖
- 避免组合中同时出现极端亏损
- 实现真正的风险分散

## 未来改进方向
1. 引入 Vine Copula 处理高维情况
2. 结合极值理论（EVT）改进尾部估计
3. 添加宏观因子作为外部输入
4. 实现更复杂的时序依赖建模

## 文件位置
策略实现：/Users/chengming/.openclaw/workspace/quant_trade-main/strategies/philosophy/copula_dependency_strategy.py