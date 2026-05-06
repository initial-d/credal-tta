# Credal-TTA 非金融数据实现分析报告

## 执行摘要

本报告分析了 Credal-TTA 项目对于非金融领域数据（电力、天气、能源等）的实现质量、代码正确性以及论文适用性。

**总体评估：⚠️ 存在关键问题**

---

## 1. 数据加载实现分析

### 1.1 UCI Electricity（电力需求数据）

**文件**: `credal_tta/utils/data_loader.py` (Line 143-175)

**实现状态**: ✅ 基本正确，但有改进空间

```python
def load_uci_electricity(
    customer_id: int = 0,
    data_path: str = "data/electricity.npy"
) -> Tuple[np.ndarray, int]:
    """返回: (时间序列, shift_point)"""
    try:
        data = np.load(data_path)
        return data[customer_id], 0  # ✅ 正确返回 Tuple
    except FileNotFoundError:
        # 生成合成数据作为后备
        T = 26304  # ~3年小时数据
        t = np.arange(T)
        daily = np.sin(2 * np.pi * t / 24)
        weekly = 0.5 * np.sin(2 * np.pi * t / (24 * 7))
        trend = 0.0001 * t
        noise = np.random.normal(0, 0.2, T)
        return 5 + daily + weekly + trend + noise, 0
```

**优点**:
- ✅ 返回类型正确 (Tuple[np.ndarray, int])
- ✅ 提供合理的合成数据后备方案
- ✅ 包含日周期、周周期和趋势成分

**问题**:
- ⚠️ `shift_point=0` 表示未知变化点，但实际电力数据可能有季节性变化
- ⚠️ 合成数据过于简单，缺乏真实电力数据的复杂性（如突发事件、节假日效应）

**建议**:
```python
# 可以添加更真实的变化点检测
# 例如使用统计方法自动检测潜在的regime shift
```

---

### 1.2 NOAA Weather（天气数据）

**文件**: `credal_tta/utils/data_loader.py` (Line 244-265)

**实现状态**: ✅ 正确

```python
def load_noaa_weather(
    station_id: int = 0,
    data_path: str = "data/noaa_weather.npy"
) -> Tuple[np.ndarray, int]:
    """
    冬季风暴 Uri 数据 (2021年2月)
    返回: (小时温度, 风暴开始点)
    """
    try:
        data = np.load(data_path)
        return data[station_id], 1080  # ~45天 * 24小时
    except FileNotFoundError:
        # 合成风暴数据
        T = 2400
        pre_storm = np.random.normal(10, 2, 1080)
        storm = np.linspace(10, -15, 48)  # 48小时急剧降温
        post_storm = np.random.normal(-5, 5, T - 1128)
        return np.concatenate([pre_storm, storm, post_storm]), 1080
```

**优点**:
- ✅ 明确的变化点（风暴开始）
- ✅ 合成数据模拟了真实的极端天气事件
- ✅ 温度变化幅度合理（10°C → -15°C）

---

### 1.3 ETTm1（能源变压器数据）

**文件**: `credal_tta/utils/data_loader.py` (Line 267-288)

**实现状态**: ✅ 正确

```python
def load_ettm1(
    data_path: str = "data/ETTm1.csv"
) -> Tuple[np.ndarray, int]:
    """
    ETTm1 能源变压器数据
    返回: (15分钟油温, 异常点)
    """
    try:
        import pandas as pd
        df = pd.read_csv(data_path)
        oil_temp = df['OT'].values  # 油温列
        return oil_temp, 20000  # 约2017-07-18异常
    except (FileNotFoundError, ImportError):
        # 合成异常数据
        T = 40000
        normal = np.random.normal(50, 2, 20000)
        anomaly = np.random.normal(62, 5, T - 20000)  # +24%均值, +150%方差
        return np.concatenate([normal, anomaly]), 20000
```

**优点**:
- ✅ 使用真实的 ETTm1 数据集（时序预测基准）
- ✅ 合成数据模拟了显著的异常（均值和方差同时变化）

---

## 2. 实验代码分析

### 2.1 Electricity 实验 (`experiments/electricity.py`)

**实现状态**: ⚠️ 存在问题

#### 问题 1: 超参数不一致

```python
# Line 172-177
adapter = CredalTTA(
    model=base_model,
    K=3,
    lambda_reset=3,      # ⚠️ 与论文不符
    W_max=512,
    L_min=192,           # ⚠️ 与论文不符
    smoothing_alpha=0.1,
    sigma_noise=0.5
)
```

**论文中的标准超参数**:
- `lambda_reset = 1.3` (Eq. 8)
- `L_min = 10` (最小上下文长度)

**当前设置的影响**:
- `lambda_reset=3` 过高 → 检测灵敏度降低，可能漏检regime shift
- `L_min=192` 过高 → 重置后需要更长时间恢复，增加 recovery time

#### 问题 2: 数据加载正确性

```python
# Line 112
data, _ = load_uci_electricity(customer_id=series_id)
```

✅ **已修复**: `load_uci_electricity` 现在正确返回 `Tuple[np.ndarray, int]`

#### 问题 3: 诊断信息不足

```python
# Line 179
credal_preds, diagnostics = adapter.predict_sequence(data, return_diagnostics=True)

# Line 182-186: 添加了诊断输出
num_resets = sum(1 for d in diagnostics if d.get('reset_occurred', False))
reset_times = [d['t'] for d in diagnostics if d.get('reset_occurred', False)]
max_diameter = max([d.get('diameter', 0) for d in diagnostics])
max_ratio = max([d.get('ratio', 0) for d in diagnostics])
print(f"  DEBUG: Resets={num_resets}, Times={reset_times[:10]}, MaxDiam={max_diameter:.4f}, MaxRatio={max_ratio:.4f}")
```

✅ **良好实践**: 添加了详细的诊断输出，有助于调试

---

### 2.2 Cross-Domain 实验 (`experiments/cross_domain.py`)

**实现状态**: ✅ 基本正确

```python
datasets = [
    ('Finance', 'S&P500', load_sp500_crisis, ChronosWrapper, {}),
    ('Finance', 'Bitcoin', load_bitcoin_crash, PatchTSTWrapper, {}),
    ('Demand', 'Electricity', load_uci_electricity, MoiraiWrapper, {}),  # ✅
    ('Sensor', 'NOAA', load_noaa_weather, ChronosWrapper, {}),           # ✅
    ('Energy', 'ETTm1', load_ettm1, ChronosWrapper, {}),                 # ✅
]
```

**优点**:
- ✅ 覆盖5个领域的数据集
- ✅ 使用不同的基础模型（Chronos, Moirai, PatchTST）
- ✅ 统一的评估流程

**问题**:
- ⚠️ 所有数据集使用相同的超参数 `K=3, lambda_reset=1.3`
- ⚠️ 没有针对不同领域调整参数（可能不是最优）

---

### 2.3 Multivariate ETTh1 实验 (`experiments/multivariate_etth1.py`)

**实现状态**: ❌ 存在严重问题

#### 问题 1: Regime Shift 检测失效

```python
# Line 44-50: MultivariateDiagonalHCA
def run_credal_multivariate(data, K=3, lambda_reset=1.3, diagonal=True):
    T, d = data.shape
    hca = MultivariateDiagonalHCA(d=d, K=K, lambda_reset=lambda_reset)
    preds = np.zeros_like(data)
    context = []

    for t in range(T):
        x_t = data[t]
        result = hca.update(x_t)

        if result['regime_shift']:  # ❌ 永远不会触发！
            context = context[-10:] if len(context) > 10 else context
```

**根本原因**: `credal_tta/core/hca_multivariate.py` (Line 38)

```python
def update(self, x_t: np.ndarray) -> Dict:
    """Update with d-dimensional observation"""
    # ... Bayesian update ...
    
    diameter = self._compute_diameter()
    
    self.t += 1
    
    return {
        'regime_shift': False,  # ❌ 硬编码为 False！
        'diameter': diameter
    }
```

**影响**:
- ❌ 多变量 HCA 完全失去了 regime shift 检测能力
- ❌ 实验结果不可信：Credal-TTA 退化为固定窗口方法
- ❌ 论文中关于多变量扩展的声明（Appendix B）无法验证

#### 问题 2: 缺少完整的 Bayesian 更新逻辑

```python
# Line 41-49: 简化的 Kalman 更新
for k in range(self.K):
    for j in range(self.d):
        prior_var = self.stds[k][j] ** 2
        obs_var = 0.1  # ⚠️ 固定观测噪声
        
        K_gain = prior_var / (prior_var + obs_var)
        self.means[k][j] += K_gain * (x_t[j] - self.means[k][j])
        self.stds[k][j] = np.sqrt((1 - K_gain) * prior_var)
```

**问题**:
- ⚠️ `obs_var=0.1` 固定值，未考虑不同维度的噪声差异
- ⚠️ 缺少 credal set 的极值分布更新逻辑
- ⚠️ 没有实现论文中的对角协方差近似（Eq. W2 距离）

---

### 2.4 Synthetic 实验 (`experiments/synthetic.py`)

**实现状态**: ✅ 正确

```python
# Line 105-112
adapter = CredalTTA(
    model=base_model,
    K=3,
    lambda_reset=3.5,    # ⚠️ 比论文高
    W_max=512,
    L_min=64,            # ⚠️ 比论文高
    smoothing_alpha=1.0
)
```

**优点**:
- ✅ 完整的对比实验（Standard, Variance-Trigger, Credal-TTA）
- ✅ 多次运行取平均（num_runs=10）
- ✅ 使用合成数据（SinFreq, StepMean）便于控制

**问题**:
- ⚠️ 超参数与论文不一致（可能是针对合成数据的调优）

---

## 3. 核心算法分析

### 3.1 单变量 HCA (`credal_tta/core/hca.py`)

**实现状态**: ✅ 正确且完整

```python
# Line 204: Regime shift 检测逻辑
regime_shift = self.smoothed_ratio > self.lambda_reset
```

**优点**:
- ✅ 完整实现了论文算法（Algorithm 1）
- ✅ Burn-in health check（R1-W2 响应）
- ✅ 自适应阈值估计
- ✅ 详细的诊断信息

---

### 3.2 多变量 HCA (`credal_tta/core/hca_multivariate.py`)

**实现状态**: ❌ 不完整

**缺失功能**:
1. ❌ Regime shift 检测（硬编码为 False）
2. ❌ Contraction ratio 计算
3. ❌ 完整的 credal set 更新逻辑
4. ⚠️ 对角协方差近似实现过于简化

**需要修复**:
```python
def update(self, x_t: np.ndarray) -> Dict:
    # ... 现有的 Bayesian 更新 ...
    
    # ✅ 应该添加:
    curr_diameter = self._compute_diameter()
    ratio = curr_diameter / (self.prev_diameter + epsilon)
    regime_shift = ratio > self.lambda_reset
    
    self.prev_diameter = curr_diameter
    
    return {
        'regime_shift': regime_shift,  # ✅ 修复
        'diameter': curr_diameter,
        'ratio': ratio
    }
```

---

## 4. 论文适用性评估

### 4.1 数据集覆盖

| 领域 | 数据集 | 论文提及 | 代码实现 | 状态 |
|------|--------|----------|----------|------|
| Finance | S&P 500 | ✅ | ✅ | ✅ 正确 |
| Finance | Bitcoin | ✅ | ✅ | ✅ 正确 |
| Demand | UCI Electricity | ✅ | ✅ | ⚠️ 超参数问题 |
| Sensor | NOAA Weather | ✅ | ✅ | ✅ 正确 |
| Energy | ETTm1 | ✅ | ✅ | ❌ 多变量问题 |

---

### 4.2 实验可重现性

| 实验 | 论文表格 | 代码文件 | 可重现性 |
|------|----------|----------|----------|
| 合成数据 | Table 2 | `synthetic.py` | ⚠️ 超参数不同 |
| 电力数据 | Table 3 | `electricity.py` | ⚠️ 超参数不同 |
| 跨领域 | Table 3 | `cross_domain.py` | ✅ 基本可重现 |
| 多变量 | Appendix B | `multivariate_etth1.py` | ❌ 算法缺陷 |

---

### 4.3 论文声明验证

#### ✅ 已验证的声明:
1. **单变量 regime shift 检测**: HCA 正确实现了基于 credal set diameter 的检测
2. **跨领域泛化**: 代码支持多个领域的数据集
3. **Burn-in health check**: 实现了 R1-W2 响应中的改进

#### ❌ 无法验证的声明:
1. **多变量扩展 (Appendix B)**: 
   - 论文声称: "对角协方差近似将复杂度从 O(K²d³) 降至 O(K²d)"
   - 代码问题: `regime_shift` 硬编码为 False，算法失效
   
2. **超参数一致性**:
   - 论文: `lambda_reset=1.3, L_min=10`
   - 代码: `lambda_reset=3, L_min=192` (electricity.py)
   - 影响: 实验结果可能与论文不一致

---

## 5. 预期实验效果

### 5.1 电力数据 (UCI Electricity)

**理论预期** (基于论文 Table 3):
- **MAE**: Credal-TTA 应比 Standard 降低 5-10%
- **Recovery Time**: 应减少 30-50%
- **Regime Shift 检测**: 应检测到季节性变化、节假日效应

**当前代码的实际效果**:
- ⚠️ 由于 `lambda_reset=3` 过高，检测灵敏度降低
- ⚠️ `L_min=192` 过高，恢复时间可能增加
- ⚠️ 可能无法达到论文声称的性能

**建议修复**:
```python
adapter = CredalTTA(
    model=base_model,
    K=3,
    lambda_reset=1.3,  # ✅ 改为论文值
    W_max=512,
    L_min=10,          # ✅ 改为论文值
    smoothing_alpha=1.0,
    sigma_noise=None   # ✅ 自动估计
)
```

---

### 5.2 天气数据 (NOAA Weather)

**理论预期**:
- **极端事件检测**: 应快速检测到冬季风暴 Uri
- **Recovery Time**: 风暴后 < 24 小时恢复
- **MAE**: 风暴期间误差应显著低于 Standard

**代码实现**: ✅ 正确，预期效果良好

---

### 5.3 多变量数据 (ETTm1)

**理论预期** (基于论文 Appendix B):
- **对角近似**: 应保持与完整协方差相近的性能
- **计算效率**: 延迟应 < 1ms per step
- **异常检测**: 应检测到 2017-07-18 的油温异常

**当前代码的实际效果**:
- ❌ **完全失效**: `regime_shift=False` 导致算法退化
- ❌ **无法验证论文声明**: 实验结果不可信
- ❌ **需要完全重写**: 必须实现完整的多变量 HCA

---

## 6. 关键问题总结

### 🔴 严重问题 (必须修复)

1. **多变量 HCA 算法缺陷**
   - 文件: `credal_tta/core/hca_multivariate.py`
   - 问题: `regime_shift` 硬编码为 False
   - 影响: 多变量实验完全失效
   - 优先级: **最高**

2. **超参数不一致**
   - 文件: `experiments/electricity.py`, `experiments/synthetic.py`
   - 问题: `lambda_reset` 和 `L_min` 与论文不符
   - 影响: 实验结果无法重现论文
   - 优先级: **高**

### ⚠️ 中等问题 (建议修复)

3. **电力数据变化点未知**
   - 文件: `credal_tta/utils/data_loader.py`
   - 问题: `shift_point=0` 表示未知
   - 影响: 无法准确评估 recovery time
   - 建议: 添加自动变化点检测

4. **合成数据过于简单**
   - 文件: `credal_tta/utils/data_loader.py`
   - 问题: 后备合成数据缺乏真实复杂性
   - 影响: 测试覆盖不足
   - 建议: 增强合成数据的真实性

### ✅ 良好实践

5. **单变量 HCA 实现完整**
   - 文件: `credal_tta/core/hca.py`
   - 状态: ✅ 正确实现论文算法

6. **数据加载接口统一**
   - 文件: `credal_tta/utils/data_loader.py`
   - 状态: ✅ 所有加载器返回 `Tuple[np.ndarray, int]`

---

## 7. 修复建议

### 7.1 立即修复 (P0)

```python
# 1. 修复 credal_tta/core/hca_multivariate.py
def update(self, x_t: np.ndarray) -> Dict:
    # ... 现有更新逻辑 ...
    
    curr_diameter = self._compute_diameter()
    
    # ✅ 添加 contraction ratio 计算
    if self.prev_diameter > 0:
        ratio = curr_diameter / (self.prev_diameter + 1e-8)
    else:
        ratio = 1.0
    
    # ✅ 添加 regime shift 检测
    regime_shift = ratio > self.lambda_reset
    
    self.prev_diameter = curr_diameter
    
    return {
        'regime_shift': regime_shift,  # ✅ 修复
        'diameter': curr_diameter,
        'ratio': ratio
    }
```

```python
# 2. 修复 experiments/electricity.py 超参数
adapter = CredalTTA(
    model=base_model,
    K=3,
    lambda_reset=1.3,  # ✅ 改为论文值
    W_max=512,
    L_min=10,          # ✅ 改为论文值
    smoothing_alpha=1.0,
    sigma_noise=None
)
```

### 7.2 后续改进 (P1)

```python
# 3. 增强电力数据加载器
def load_uci_electricity(customer_id: int = 0, detect_shifts: bool = True):
    data = np.load(data_path)[customer_id]
    
    if detect_shifts:
        # 使用统计方法检测变化点
        shift_points = detect_change_points(data)
        return data, shift_points[0] if shift_points else 0
    else:
        return data, 0
```

---

## 8. 结论

### 8.1 总体评价

**代码质量**: ⚠️ **中等偏下**
- 单变量实现: ✅ 优秀
- 多变量实现: ❌ 不可用
- 数据加载: ✅ 良好
- 实验脚本: ⚠️ 超参数问题

### 8.2 论文适用性

**可用于论文的部分**:
- ✅ 单变量 regime shift 检测（金融、天气数据）
- ✅ 跨领域泛化实验
- ✅ 合成数据基准测试

**不可用于论文的部分**:
- ❌ 多变量扩展（Appendix B）
- ⚠️ 电力数据实验（超参数不一致）

### 8.3 修复优先级

1. **P0 (必须修复)**: 多变量 HCA 的 `regime_shift` 检测
2. **P0 (必须修复)**: 统一超参数为论文值
3. **P1 (建议修复)**: 电力数据变化点检测
4. **P2 (可选)**: 增强合成数据真实性

### 8.4 预期修复后效果

修复后，项目应能够:
- ✅ 完整重现论文中的所有实验
- ✅ 验证多变量扩展的有效性
- ✅ 在电力、天气、能源等领域达到论文声称的性能
- ✅ 提供可靠的跨领域泛化能力

---

## 附录: 快速修复脚本

```bash
# 1. 备份原文件
cp credal_tta/core/hca_multivariate.py credal_tta/core/hca_multivariate.py.bak
cp experiments/electricity.py experiments/electricity.py.bak

# 2. 应用修复 (需要手动编辑上述文件)

# 3. 验证修复
python experiments/multivariate_etth1.py  # 应该看到 regime_shift=True
python experiments/electricity.py --num_series 5  # 快速测试

# 4. 完整测试
python experiments/cross_domain.py
```

---

**报告生成时间**: 2026-05-06  
**分析者**: Claude ([REDACTED])  
**项目版本**: commit 4c33eb9
