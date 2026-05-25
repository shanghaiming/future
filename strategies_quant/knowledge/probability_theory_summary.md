# 概率论与统计推断 — 期货量化交易实战摘要

---

## 1. 核心概率概念与交易映射

### 条件概率与贝叶斯推断

- **条件概率**: P(A|B) = P(A intersect B) / P(B)
- **贝叶斯定理**: P(H|D) = P(D|H) * P(H) / P(D) -- 观测数据更新假设概率
- 先验 = 交易前信念; 后验 = 看到数据后的更新信念

**交易应用 -- 信号可靠性评估**:

```
先验: RSI金叉默认胜率 (如55%)
似然: 当前市场状态下该信号的历史表现
后验: 更新后的胜率估计
```

**Beta分布共轭先验 -- 动态止损**:

```python
def bayesian_stop_loss(prior_win_rate, recent_wins, recent_losses, base_stop):
    alpha = prior_win_rate * 100 + recent_wins
    beta = (1 - prior_win_rate) * 100 + recent_losses
    posterior_mean = alpha / (alpha + beta)
    stop_multiplier = max(0.5, min(2.0, posterior_mean / prior_win_rate))
    return base_stop * stop_multiplier
```

### 凯利公式 -- 最优仓位

```python
def kelly_fraction(win_prob, avg_win, avg_loss):
    b = avg_win / avg_loss  # 盈亏比
    f = (win_prob * b - (1 - win_prob)) / b
    return max(0, f)  # 实际使用: kelly * 0.5 (半凯利)
```

---

## 2. 统计分布与收益分析

### 常用分布

| 分布 | 特点 | 交易用途 |
|------|------|---------|
| 正态分布 N(mu, sigma^2) | 对称, 薄尾 | 基准比较 (金融收益通常偏离) |
| 学生t分布 | 肥尾, 自由度越低尾部越厚 | 收益率建模 (比正态更准确) |
| 广义Pareto (GPD) | 专门拟合尾部 | 极端风险VaR/TVaR计算 |
| 广义极值 (GEV) | 3种类型: Gumbel/Frechet/Weibull | 极端事件建模 (金融收益属于Frechet) |
| Beta分布 | 有界 [0,1] | 胜率估计的贝叶斯先验/后验 |

### 肥尾效应 -- 关键结论

金融收益率有显著肥尾 (峰度 > 3) 和偏度。A股实证: GEV形状参数 xi 约等于 0.3-0.5, 远超正态的 xi=0。这意味着正态VaR会低估20-50%的极端风险。

**GPD尾部风险估计**:

```python
from scipy.stats import genpareto

def estimate_tail_risk(returns, confidence=0.99):
    threshold = np.percentile(returns, 5)
    tail_data = threshold - returns[returns < threshold]
    params = genpareto.fit(tail_data)
    evt_var = genpareto.ppf(confidence, *params)
    return evt_var
```

### 波动率建模

- 历史波动率: sigma = std(returns) * sqrt(252)
- EWMA: sigma_t^2 = lambda * sigma_{t-1}^2 + (1-lambda) * r_{t-1}^2, lambda=0.94
- GARCH(1,1): sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2

---

## 3. 假设检验与策略验证

### 策略显著性检验

```python
from scipy import stats

def strategy_significance(returns, benchmark_returns):
    excess = returns - benchmark_returns
    t_stat, p_value = stats.ttest_1samp(excess, 0)
    sharpe = excess.mean() / excess.std() * np.sqrt(252)
    return p_value < 0.05 and sharpe > 1.0
```

### 过拟合检测方法

1. **样本外测试**: 训练60% / 验证20% / 测试20%
2. **Walk-forward验证**: 滚动窗口训练和测试
3. **Bootstrap**: 对收益序列重采样1000次, 检查表现分布
4. **CSCV**: 数据分N段, 取所有组合做交叉验证

### 多重检验修正

- 测试N个策略时, Bonferroni修正: p_threshold = 0.05/N
- Benjamini-Hochberg (FDR): 控制假发现率, 更适合策略筛选
- 最小样本量: 交易次数 > 30; 月度收益 > 60个月 (5年)

### Bootstrap策略评估

```python
def bootstrap_strategy(returns, B=1000):
    n = len(returns)
    sharpes = []
    for _ in range(B):
        sample = np.random.choice(returns, size=n, replace=True)
        sharpe = sample.mean() / sample.std() * np.sqrt(252)
        sharpes.append(sharpe)
    ci_low, ci_high = np.percentile(sharpes, [2.5, 97.5])
    return np.mean(sharpes), ci_low, ci_high
```

---

## 4. 随机过程与时间序列

### 几何布朗运动 (GBM)

dS = mu*S*dt + sigma*S*dW, W是维纳过程

**蒙特卡洛模拟**:

```python
def monte_carlo_price(S0, mu, sigma, T, n_sims=10000):
    dt = 1 / 252
    steps = int(T * 252)
    prices = np.zeros((n_sims, steps))
    prices[:, 0] = S0
    for t in range(1, steps):
        z = np.random.standard_normal(n_sims)
        prices[:, t] = prices[:, t-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        )
    return prices
```

### 均值回归检测

- ADF检验: 判断价格序列是否平稳
- Hurst指数: H < 0.5 均值回归, H = 0.5 随机游走, H > 0.5 趋势

### 鞅论与交易

- 价格是鞅: E[X_{n+1} | past] = X_n -- 任何策略期望收益=0 (有效市场)
- 上鞅 (Submartingale): E[X_{n+1} | past] >= X_n -- 趋势的数学定义
- Doob分解: 价格 = 随机噪声(鞅) + 可预测趋势(A)
- 检验鞅假设: 用自回归 R^2, 若显著>0则拒绝鞅假设, 存在可利用规律

---

## 5. 贝叶斯方法与信号融合

### 多信号融合

每个信号给出独立的 P(上涨|信号), 用朴素贝叶斯融合:

```python
def naive_bayes_fusion(signal_probs, prior_up=0.5):
    """signal_probs: 各信号的 P(up|signal_i)"""
    odds = prior_up / (1 - prior_up)
    for p in signal_probs:
        odds *= p / (1 - p)
    return odds / (1 + odds)  # 融合后概率
```

### 信息论 -- 特征选择与市场状态检测

- **熵**: H(X) = -Sum p(x) log p(x) -- 不确定性度量
- **互信息**: I(X;Y) = H(X) - H(X|Y) -- 共享信息量
- **KL散度**: D_KL(P||Q) = Sum P(x) log(P(x)/Q(x)) -- 分布差异

**互信息做特征选择**:

```python
from sklearn.feature_selection import mutual_info_regression
mi_scores = mutual_info_regression(features_df, future_returns)
```

**KL散度检测市场状态变化**:

```python
from scipy.stats import entropy

def detect_regime_change(recent_returns, historical_dist):
    recent_hist, _ = np.histogram(recent_returns, bins=50, density=True)
    hist_hist, _ = np.histogram(historical_dist, bins=50, density=True)
    return entropy(recent_hist + 1e-10, hist_hist + 1e-10)
```

### 信号独立性 -- 组合策略的前提

策略组合的关键不是"哪些好", 而是"哪些独立"。互信息比率 I(X;Y)/H(X) < 0.3 才允许组合。

```python
from scipy.stats import kendalltau

def strategy_independence_test(returns_A, returns_B):
    tau, p_tau = kendalltau(returns_A, returns_B)
    is_independent = p_tau > 0.05 and abs(tau) < 0.1
    return is_independent, tau, p_tau
```

---

## 6. 核方法与非参数估计

### 核密度估计 (KDE)

f(x) = (1/nh) * Sum K((x - x_i)/h), 不假设数据分布, 让数据自己说话。

### 核回归 (Nadaraya-Watson)

m(x) = Sum w_i * y_i, w_i = K((x-x_i)/h) / Sum K((x-x_j)/h)
本质: 对每个查询点用核函数加权附近样本 -- 局部加权平均。
带宽自适应: h = ATR * multiplier, 波动大时窗口宽(平滑噪音), 波动小时窗口窄(灵敏信号)。

### Epanechnikov核的最优性

MSE最优核, 有限支撑 (|u|>1时K(u)=0), 远端点权重为0, 不会被极端值污染。

---

## 7. 稳健统计与异常值处理

### L1 vs L2损失

- L2 (OLS): 对异常值极度敏感 (平方放大偏差)
- L1 (绝对值): 对异常值鲁棒 (线性增长)
- Huber: delta以下用L2 (保持效率), delta以上用L1 (保持鲁棒)

### IRLS (迭代重加权最小二乘)

beta^(t+1) = (X'W^t X)^{-1} X'W^t y, 权重W随残差自适应调整。通常3次迭代足够。对期货涨跌停等极端bar的残差自动降权。

---

## 8. 序贯分析与信号检测

### SPRT (序贯概率比检验)

每个时间步计算似然比 Lambda_n, 超过上限则入场, 低于下限则不入场, 否则继续观察。自适应决策时间, 不需要预先确定样本量。

### CUSUM控制图

S_t = max(0, S_{t-1} + (x_t - mu_0 - k)), 信号: S_t > h。累积偏差, 比单点翻转更稳定。

### 变点检测

- 波动率regime切换 = 方差变点
- 趋势开始/结束 = 均值变点

---

## 9. Copula与多策略依赖结构

### 核心结论

Pearson相关只度量线性关系, 两个策略可以 rho=0 但完全非线性依赖。Copula分离了边缘分布和依赖结构。

### 策略独立性检验

```python
def strategy_similarity_matrix(returns_dict):
    strategies = list(returns_dict.keys())
    n = len(strategies)
    tau_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            tau, _ = kendalltau(returns_dict[strategies[i]], returns_dict[strategies[j]])
            tau_matrix[i, j] = tau_matrix[j, i] = abs(tau)
    return tau_matrix
```

### 组合数学

独立信号一致的贝叶斯后验:
- P(真信号|信号A and 信号B) = P(A and B|真信号) * P(真信号) / P(A and B)
- 独立时分子 = P(A|真) * P(B|真) * P(真) -- 后验概率大幅提升
- 好策略+好策略 != 好组合, 独立策略+独立策略 = 好组合

---

## 10. 蒙特卡洛方法与鲁棒性验证

### Walk-Forward框架

```python
def walk_forward_validation(data, strategy_class, train_years=2, test_years=1):
    results = []
    start = 0
    total_days = (train_years + test_years) * 252
    while start + total_days <= len(data):
        train = data[start : start + train_years * 252]
        test = data[start + train_years * 252 : start + total_days]
        best_params = optimize(strategy_class, train)
        test_result = backtest(strategy_class, test, best_params)
        results.append(test_result)
        start += 252
    return results
```

### 随机矩阵理论 (RMT) 去噪

经验相关矩阵的大部分特征值是噪声。Marchenko-Pastur定律界定噪声范围, lambda > lambda_+ 的特征值包含真实信号。

```python
from sklearn.covariance import LedoitWolf

lw = LedoitWolf().fit(returns)
clean_cov = lw.covariance_  # 收缩估计, 实战替代RMT
```

---

## 11. Kalman滤波 -- 自适应估计器

Kalman滤波是贝叶斯推断的递归实现: 每步做一次先验->后验更新。EMA是它的退化特例(固定增益), ATR止损是它的简化版。

### Kalman趋势滤波 (替代EMA)

```python
def kalman_trend_filter(prices, q_ratio=0.01, r_ratio=1.0):
    """状态: [price, velocity], 观测: price"""
    F = np.array([[1, 1], [0, 1]])
    H = np.array([[1, 0]])
    Q = np.diag([q_ratio * 0.25, q_ratio])
    R = np.array([[r_ratio]])
    x0 = np.array([prices[0], 0.0])
    P0 = np.eye(2)
    kf = KalmanFilter(F, H, Q, R, x0, P0)
    filtered, velocity, uncertainty = [], [], []
    for p in prices:
        state, cov = kf.step(p)
        filtered.append(state[0])
        velocity.append(state[1])
        uncertainty.append(np.sqrt(cov[0, 0]))
    return filtered, velocity, uncertainty
```

**核心优势**: Kalman增益 K_t 每步自适应 -- 波动率高时K下降(更信任预测), 波动率低时K上升(更信任观测)。

### Regime判定

- |velocity| > theta 且不确定性小 --> 趋势regime
- |velocity| < theta --> 震荡regime
- 不确定性大 --> 不确定, 减仓

---

## 12. HMM (隐马尔可夫模型) -- 市场状态检测

HMM是概率化的Hurst: 不问"有没有趋势", 而是问"处于bull状态的概率是多少"。

核心三问题: 评估(Forward算法), 解码(Viterbi算法), 学习(Baum-Welch/EM)。

推荐发射变量: (收益率, 归一化波动率, 成交量百分位排名), 使用Student-t发射分布适配肥尾。

HMM适合做regime gate (状态门控), 不适合做信号发生器。

---

## 13. Wasserstein距离 -- 分布漂移检测

唯一在分布支撑集不重叠时仍能提供有意义梯度的度量。

**训练/测试分布漂移检测**:

```python
from scipy.stats import wasserstein_distance

def detect_distribution_shift(train_returns, test_returns, window=60):
    baseline = []
    for i in range(window, len(train_returns) - window):
        w1 = wasserstein_distance(
            train_returns[i-window:i], train_returns[i:i+window]
        )
        baseline.append(w1)
    threshold = np.percentile(baseline, 95)
    train_tail = train_returns[-window:]
    shifts = []
    for i in range(window, len(test_returns)):
        w1 = wasserstein_distance(train_tail, test_returns[i-window:i])
        shifts.append(w1 > threshold)
    return shifts, threshold
```

---

## 14. 极值理论 (EVT) -- 尾部风险

**Fisher-Tippett定理**: 极值的极限分布只有3种, 金融收益属于Frechet (幂律尾部)。

**POT方法 (Peaks Over Threshold)**: 对超过高阈值的极端值用GPD建模。

ATR止损的EVT优化: 如果xi=0.3 (典型值), 止损倍数k需要比正态假设大20-30%。动态k = k_base * (1 + xi * percentile_rank(ATR))。

---

## 15. 成交量微观结构 -- VDP公式

**量效方程**: Volume_Efficiency(t) = Volume(t) * Directional_Quality(t)

VDP (最优): delta = V * (2C - H - L) / (H - L), 范围 [-1, +1]
- C=H (涨停): delta = +V (100%买入方向)
- C=L (跌停): delta = -V (100%卖出方向)
- C=(H+L)/2 (中点): delta = 0 (中性)

方向性质量层次: VDP > PVS > FlowStrength > OBV > Raw Volume, 每步降级都丢失不可恢复的信息。

---

## 16. 波动率的正确角色

波动率是二阶统计量, 不应作为独立alpha源, 正确角色:
1. 止损引擎 (ATR trailing stop)
2. 仓位管理器 (波动率倒数加权, 风险平价)
3. 方向信号的修饰符, 而非信号本身

ATR比率 (fast/slow) 是最干净的波动率度量:
- Ratio < 阈值 = 压缩 (Squeeze)
- Ratio 穿越回1.0以上 = 扩张触发 (Expansion)

---

## 17. 实战公式速查表

| 公式 | 用途 | Python |
|------|------|--------|
| 夏普比率 = mean/sigma * sqrt(252) | 策略评估 | `excess.mean()/excess.std()*252**0.5` |
| VaR (历史) = percentile(returns, 5) | 95% VaR | `np.percentile(returns, 5)` |
| EWMA波动率 | 实时波动率 | `ewm(span=20).var()` |
| Hurst指数 | 趋势/均值回归判定 | R/S分析或小波方差回归 |
| KL散度 | 状态变化检测 | `scipy.stats.entropy(p+eps, q+eps)` |
| Kendall tau | 策略独立性 | `scipy.stats.kendalltau(a, b)` |
| 互信息 | 特征选择 | `sklearn.mutual_info_regression` |
| Wasserstein W1 | 分布漂移 | `scipy.stats.wasserstein_distance` |
| Kalman滤波 | 自适应趋势估计 | 自定义 KalmanFilter 类 |
| GPD尾部 | 极端VaR | `scipy.stats.genpareto.fit` |

---

## 18. 核心原则总结

1. **独立性 > 复杂性**: 两个独立的好策略组合远胜于两个相关的优秀策略
2. **秩 > 矩**: 非参数方法 (percentile/rank) 在有涨跌停的市场中全面优于参数方法 (z-score/mean)
3. **波动率做防守, 方向做进攻**: 波动率是守门员不是前锋
4. **贝叶斯思维**: 每个新数据点应更新信念, 而非推翻或盲从
5. **统计显著性是底线**: 策略必须通过多重检验修正和样本外验证
6. **肥尾是常态**: 永远不要用正态分布假设做风险管理, 用Student-t或EVT
