# Development Log

> 开发日志 - 记录每次改进、测试效果和下一步计划
> 使用 `/log` 命令添加新记录，`/devlog` 命令查看历史

---

## 2025-12-03 05:30

**提交状态**: ⏳ 未提交

### 改进内容
- 实现真实期权数据的波动率曲面校准
- 验证 EP Hessian 在金融应用中的正确性
- 比较不同 B-spline 网格配置的拟合效果
- 编译 Cython EP 模块 (algo4_cython_simple, symm_sparse_adjlist_cpp)
- 完成 PDE+B-spline V(w) 的 Hessian 计算

### 波动率曲面校准

#### 数据
- SPY 期权数据: `UnderlyingOptionsEODQuotes_2025-02-06.csv`
- 标的价格: $6081.76
- 有效期权数: 5,405

#### EP Hessian 验证
- EP vs 数值 Hessian 误差: **10⁻⁸ ~ 10⁻¹⁴**
- 测试配置: 20 参数 (5×4), 100 数据点
- 结构完全匹配，验证 EP 在真实金融问题上正确

#### B-spline 网格配置比较

| 配置 | 参数数 | 总 RMSE | 短期 RMSE |
|------|--------|---------|-----------|
| Uniform (5×4) | 42 | 0.0156 | 0.0164 |
| **Dense Short-T (5×6)** | 56 | **0.0148** | **0.0150** |
| Dense M + Short-T (7×6) | 72 | 0.0149 | 0.0152 |
| Weighted Short-T (5×4) | 42 | 0.0153 | 0.0158 |

#### 关键发现
- Dense Short-T 配置最优，短期 RMSE 降低 8.5%
- 增加 moneyness 节点无显著改善
- 加权方法效果有限，问题根源是节点分布

#### PDE+B-spline Hessian 计算

| 指标 | 值 |
|------|-----|
| 期权价格 (PDE) | 10.4501 |
| BS 解析价格 | 10.4506 |
| 价格误差 | 0.0005 |
| Hessian 维度 | 10×10 |
| 计算时间 (Cython) | 84.6s |
| 稀疏性 | 36% |
| 条件数 | 1.25 |

### 生成的文件
- `calibrate_vol_surface_fast.py` - 快速校准脚本
- `calibrate_vol_surface_ep.py` - EP Hessian 验证脚本
- `calibrate_vol_surface_improved.py` - 网格配置比较脚本
- `compute_pde_hessian.py` - PDE Hessian 计算脚本
- `vol_surface.png`, `vol_smiles.png` - 波动率曲面可视化
- `hessian_comparison.png` - EP vs 数值 Hessian 比较
- `vol_smiles_comparison.png`, `rmse_comparison.png` - 配置比较
- `pde_hessian.png` - PDE Hessian 可视化

### Insights
- **EP 在真实金融问题上的精度**: EP vs 数值 Hessian 误差达到 10⁻⁸ ~ 10⁻¹⁴，远超金融应用需求
- **B-spline 节点分布的重要性**: 短期期权波动率曲率更陡，需要更密的时间节点来捕捉，加权方法无法替代合理的节点分布
- **PDE Hessian 的稀疏结构**: B-spline 紧支撑特性导致 36% 稀疏性，中心区域（ATM）对波动率最敏感
- **Hessian 不定性**: 期权价格对波动率的二阶导数可正可负，反映了期权价值函数的凸性变化

### 下一步计划
1. ~~EP vs Taylor vs Bumping2 性能基准测试~~ (已完成基础测试)
2. ~~Newton-Raphson 二阶优化~~ (跳过)
3. ~~PDE+B-spline V(w) 的 Hessian 计算~~ (已完成)

---

## 2025-12-03 04:00

**提交状态**: ✅ 已提交 (commit: 30487e2)

### 改进内容
- 将 Taylor 代码从 `jpm_practicum` 提取到 `aad_edge_pushing/aad/taylor.py`
- 修复 `aad_edge_pushing` 的 EP bug：
  - `use_tape()` 兼容性问题（arithmetic.py, transcendental.py, special.py, engine.py）
  - Creating 阶段同变量因子2问题（engine.py 第458-461行）

### 测试效果
- EP vs Taylor 精度完全匹配（误差 <1e-15）
- 测试通过函数：
  - `f(x) = sum(xᵢ²)` → Hessian = 2I ✓
  - `f(x) = sqrt(1 + Σxᵢ²)` ✓
  - `f(x) = exp(Σxᵢ²)` ✓
  - `f(x) = log(1 + Σxᵢ²)` ✓
  - `f(x) = (Σxᵢ²)²` ✓
  - `f(x) = x³` via x*x*x ✓

### Insights
- **Bug 根因**：`x*x` 的 cross term `{("cross", (0,1)): 1.0}` 两个 parent 都是同一个 x
- `frozenset({id(x), id(x)})` 坍缩为 singleton `frozenset({id(x)})`
- 需要 factor 2 补偿：`(a+a)² = 2a²`
- **重要**：Pushing 阶段和 Creating 阶段**都需要**处理同变量情况

### 下一步计划
1. 分析 Taylor, EP, Bumping2 在 PDE+B-spline 运行速度差异的原因
2. 使用真实数据 `UnderlyingOptionsEODQuotes_2025-02-06` 进行 calibration
3. 绘制波动率曲面和实际 Hessian 图

---

## 2025-12-03 02:30

### B-spline PDE模型验证

#### 测试环境
- Grid: M=20, N=10
- B-spline configs: 4x4 到 12x12

#### 精度验证

对角线和交叉项Hessian与Bumping2完全匹配：
- 对角线最大相对误差: < 0.01%
- 交叉项最大相对误差: < 0.01%

#### 性能对比

| 参数数 | EP (s) | Bumping2 diag | Bumping2 full (est) | vs diag | vs full |
|--------|--------|---------------|---------------------|---------|---------|
| 16 | 1.31 | 0.18 | 3.01 | 0.14x | 2.3x |
| 36 | 1.48 | 0.50 | 18.27 | 0.34x | **12.3x** |
| 64 | 1.70 | 1.07 | 69.20 | 0.63x | **40.6x** |
| 100 | 1.91 | 1.90 | 190.91 | 1.0x | **100x** |
| 144 | 1.86 | 3.13 | 453.00 | **1.69x** | **244x** |

**关键发现**：
- EP时间几乎恒定 (~1.5-1.9s)，与参数数量关系不大
- Crossover点（vs Bumping2 diag）：约100参数
- 完整Hessian优势巨大：144参数时**244x faster**
- B-spline compact support自动剔除无效参数（144→110）

---

## 2025-12-03 02:00

### EP精度问题修复 - Pushing阶段同变量因子2

#### 问题描述

EP在复合函数（如 `sqrt(1 + Σxᵢ²)`）上有~3-12%的误差。

#### 根本原因

在Pushing阶段处理单例 `W[{y}]` 时，当两个parent指向**同一个变量**时（例如 `y = x * x`），cross term缺少因子2。

**数学推导**：
对于 `y = x * x`，存储的local partials是 `[(x, x.val), (x, x.val)]`。
```
(∂y/∂x)² = (a₀ + a₁)² = a₀² + 2*a₀*a₁ + a₁²
```
原代码计算 `a₀² + a₀*a₁ + a₁² = 3x²`，但正确值是 `4x²`。

#### 修复位置

`engine.py` 第 386-393 行：
```python
# 原代码
W_var_pairs[new_key] += w * a_r * a_s

# 修复后
if id(p_r) == id(p_s):
    W_var_pairs[new_key] += 2.0 * w * a_r * a_s  # 二项式展开因子2
else:
    W_var_pairs[new_key] += w * a_r * a_s
```

#### 修复后验证

| 测试函数 | 修复前误差 | 修复后误差 |
|---------|-----------|-----------|
| sqrt(1+Σx²) | 3.40e-02 | **6.92e-07** |
| exp(-0.5*Σx²) | ~10% | **2.60e-07** |
| exp(-sqrt(log(1+Σx²))) | ~12% | **4.09e-07** |

**精度现在与Taylor完全一致！**

#### 性能对比（修复后）

| 维度 | EP (ms) | Taylor (ms) | EP加速 |
|------|---------|-------------|--------|
| 5 | 0.34 | 20.68 | 60.8x |
| 10 | 0.74 | 153.69 | 206.9x |
| 20 | 1.93 | 1171.93 | 606.9x |
| 36 | 5.15 | 7585.23 | **1472.4x** |

**结论**：EP现在在所有复合函数上精度与Taylor一致，且速度优势保持不变。

---

## 2025-12-03 01:00

### 完整对比测试：EP vs Taylor vs Bumping2

#### 测试环境

创建 `test_ep_taylor_bumping_comparison.py`，系统对比三种方法的精度和速度。

#### 精度对比

| 测试函数 | Taylor误差 | EP误差 | 说明 |
|---------|-----------|--------|------|
| f(x) = Σxᵢ² | 1.02e-05 | 1.02e-05 | ✓ 两者都准确 |
| f(x) = x₀x₁+x₂x₃ | 1.55e-05 | 1.55e-05 | ✓ 交叉项准确 |
| exp+sqrt+log | 6.22e-05 | **1.23e-01** | ⚠ EP有精度问题 |

**EP精度问题分析**：
- `sqrt(1 + Σxᵢ²)` 单独测试有 ~3% 误差
- 可能原因：复合函数的二阶导数链式法则处理不完善
- 我们的稀疏EP实现（用于B-spline）已验证正确（误差<0.02%）

#### 速度对比

| 维度 | Taylor (ms) | EP (ms) | Bump2 (ms) | EP vs Bump | EP vs Taylor |
|------|------------|---------|------------|------------|--------------|
| 5 | 0.34 | 0.54 | 0.44 | 0.82x | 0.63x |
| **10** | 2.05 | **1.13** | 2.63 | **2.32x** | **1.81x** |
| 20 | 14.35 | 2.92 | 17.88 | 6.12x | 4.91x |
| 36 | 78.79 | 8.42 | 96.41 | **11.45x** | **9.36x** |
| 64 | 427.28 | 28.96 | 517.83 | **17.88x** | **14.75x** |

**Crossover点**：n ≈ 10

#### 复杂度分析

```
方法对比:
┌─────────────────┬────────────────┬────────────────────┐
│     Method      │ Function Evals │   Complexity       │
├─────────────────┼────────────────┼────────────────────┤
│ Taylor          │  n(n+1)/2      │  O(n²) passes      │
│ Edge-Pushing    │  1             │  O(1) passes       │
│ Bumping2 (FD)   │  ~4n(n+1)/2    │  O(n²) evals       │
└─────────────────┴────────────────┴────────────────────┘

实例 (n=225):
- Taylor:  25,425 次前向传播
- EP:      1 次反向传播
- Bumping2: ~101,700 次函数求值
```

#### 推荐使用场景

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| n < 10 | Taylor | 更快，精度最高 |
| n = 10-30 | EP | 速度优势开始显现 |
| n > 30 | EP | 10-20x faster than Taylor |
| B-spline PDE | EP | **1248x** faster than Bumping2 |

#### 可能的精度改进方向

1. 检查 `_second_locals` 对复合函数的处理
2. 检查 Pushing stage 的链式法则累积
3. 参考我们的稀疏EP实现（已验证正确）

---

## 2025-12-03 00:30

### JPM repo EP Bug修复 + 重新验证

#### 发现并修复的Bug

**Bug 1: use_tape() 兼容性问题**
- 问题：`arithmetic.py`, `transcendental.py`, `special.py`, `engine.py` 使用 `from .tape import global_tape`
- 原因：Python 模块导入在加载时绑定，`use_tape()` 替换模块变量后不生效
- 修复：改用 `from . import tape as tape_mod`，访问 `tape_mod.global_tape`

**Bug 2: 自乘二阶导数缺失**
- 问题：`x * x` 返回 Hessian=1 而不是 2
- 原因：`edge_push_hessian` 的 Creating stage 没有处理 parent_ids 相同的情况
- 修复：在 `engine.py` 第 448-455 行添加 factor=2 当 `parent_ids[u] == parent_ids[v]`

#### 修复后的性能对比

| 维度 | Taylor (ms) | EP (ms) | EP加速 |
|------|------------|---------|--------|
| 5 | 0.38 | 0.61 | 0.63x (Taylor faster) |
| **10** | **2.25** | **1.24** | **1.81x** |
| 15 | 6.89 | 2.06 | 3.34x |
| 20 | 15.43 | 3.19 | 4.83x |
| 36 | 81.81 | 8.75 | **9.35x** |
| 64 | 440.91 | 30.01 | **14.69x** |
| 100 | 1657.22 | 88.06 | **18.82x** |

**Crossover点**：n ≈ 10 (比修复前更早！)

#### 正确性验证

| 函数 | Taylor vs FD | EP vs FD | 匹配 |
|------|-------------|----------|------|
| f(x) = Σxᵢ² | 1.02e-05 | 1.02e-05 | ✓ |
| exp+sqrt+log | 7.75e-05 | 9.45e-02 | 部分 |

#### 结论更新

| 场景 | 推荐方法 | EP加速 |
|------|---------|--------|
| n < 10 | Taylor | - |
| n = 10-36 | EP | 2-9x |
| n = 64-100 | EP | 15-19x |
| B-spline (225) | EP | **1248x** |

**EP优势更明显**：修复bug后，EP在n≥10时就开始领先，而非之前的n≥20。

---

## 2025-12-02 23:30

### Taylor Expansion vs Edge-Pushing 初步对比

#### 对比环境

**代码来源**：https://github.com/xuenailao/JPM-Practicum-AAD

#### 发现的问题

**JPM repo EP实现有bug**：
- 对于 f(x) = Σxᵢ², Hessian应为2I
- Taylor返回2I ✓
- EP返回I ✗（差一倍）
- 原因：自乘（x*x）的二阶导数处理有误 + use_tape()兼容性问题

→ 已在后续记录中修复

#### 创建的文件

- `test_taylor_vs_ep.py` - 基础对比测试
- `test_taylor_vs_ep_detailed.py` - 详细分析（含有限差分验证）

---

## 2025-12-02 22:15

### 后续工作计划

#### 当前状态 ✅

性能优化已完成：
- 稀疏优化：60x
- Cython优化：+3x = 190x total
- 大规模验证：up to **1248x** at 225 params
- EP现在比Bumping2对角线都快（5.5x）

#### 近期任务（1-2周）

**优先级1：真实期权数据应用** ⭐⭐⭐
- 数据：`UnderlyingOptionsEODQuotes_2025-02-06`
- 目标：用EP加速B-spline volatility surface校准
- 任务：加载数据 → 设置校准目标 → EP计算Hessian → Newton优化 → 验证曲面质量

**优先级2：Taylor Expansion对比** ⭐⭐
- 代码：https://github.com/xuenailao/JPM-Practicum-AAD
- 目标：对比EP vs Taylor expansion在B-spline模型上的表现
- 关键：精度、速度、内存对比，确定最优使用场景

**优先级3：技术报告** ⭐⭐
- 文档化优化历程（60x → 190x → 1248x）
- 实际应用案例
- 方法对比和建议

#### 中长期计划

**1-3月**：
- 更复杂PDE模型（Heston, Jump-diffusion, Multi-asset）
- 生产系统开发（API, 缓存, 监控）

**3-6月**：
- 学术发表（Journal of Computational Finance）
- 开源包发布（`aad-edge-pushing`）

#### 推荐行动

Week 1-2：真实数据应用
Week 3-4：Taylor对比
Week 5-6：技术报告

---

## 2025-12-02 22:00

### 大规模验证完成 - 1248x极限加速！

#### 测试配置

**规模**：100-225参数（实际active参数：90-140，因B-spline compact support）
**环境**：Grid M=20, N=10, Cython优化启用

#### 性能结果

| 参数配置 | 活跃参数 | EP (Cython) | Bumping2 Full (估计) | **加速** |
|---------|---------|-------------|---------------------|---------|
| 10×10 (100) | 90 | 1.05s | 218s | **207x** |
| 12×12 (144) | 110 | 1.01s | 524s | **519x** |
| 15×15 (225) | 140 | **1.18s** | **1470s** | **1248x** |

#### 关键发现

**1. EP时间几乎恒定**（~1秒）- O(1)复杂度！
**2. EP现在比Bumping2对角线还快**（5.5x at 225 params）
**3. B-spline compact support自动剔除无效参数**（90/100, 110/144, 140/225）
**4. 极限加速**：225参数时达到**1248x**！

#### 结论

性能已远超目标。EP适用于：
- **完整Hessian**：始终使用EP（200-1200x faster）
- **对角线Hessian**（参数≥50）：EP仍更快（up to 5.5x）

**下一步**：准备技术报告，测试真实数据应用

---

## 2025-12-02 21:25

### Cython优化实现 - 额外3.15x加速！

#### 实施内容

尝试实现OpenMP并行化以进一步加速algo4_sparse。遇到技术限制（需要.pxd文件才能实现true nogil），但成功实现Cython + C++ vector优化版本。

**创建文件**：
- `algo4_sparse_openmp.pyx` - Cython优化的稀疏algo4
- `setup_sparse_openmp.py` - 编译配置
- `test_cython_speedup.py` - 性能测试

#### 性能结果

**测试配置**: 6×6 = 36 parameters, M=20, N=10

| 实现 | 时间 | 加速 |
|------|------|------|
| Python sparse | 1.277s | 1.0x (baseline) |
| **Cython sparse** | **0.406s** | **3.15x** |

**总加速倍数**：
- 稀疏优化: 60x (vs naive algo4)
- Cython优化: 3.15x (vs Python sparse)
- **总计: ~190x** (60 × 3.15)

#### 技术总结

**尝试的方法**：
1. ✗ True OpenMP并行化 - 需要.pxd文件支持cimport，过于复杂
2. ✓ Cython + C++ vector优化 - 成功实现，提供3x额外加速

**关键优化点**：
- 使用C++ `vector<pair<int, pair<int, double>>>` 存储更新
- Cython编译的C-level循环（比Python快）
- 与`symm_sparse_adjlist_cpp` C++矩阵配合

**技术教训**：
- 没有.pxd文件时无法使用`cimport`和`nogil`
- 但Cython的C++ vector + C-level循环仍能提供显著加速
- 稀疏优化（60x）比并行化（3x）更重要

#### 结论

**当前状态**：
- Python sparse: 1.28s/iteration
- Cython sparse: 0.41s/iteration
- **性能已足够好，无需继续优化**

**EP vs Bumping2（使用Cython版本）**：
- 36参数：Cython EP 0.41s vs Bumping2 full ~20s → **~50x faster**
- 64参数：Cython EP ~0.42s vs Bumping2 full ~78s → **~186x faster**

---

## 2025-12-02 20:00

### 代码清理 + 大规模验证完成

#### 1. 代码清理

**删除废弃文件**：
- 8个过时的C++ OpenMP实现（124K行代码）
- 15个临时文件（.pkl, .txt, LaTeX编译产物）
- 归档4个调试测试脚本到 `archive/testing/`

**保留核心文件**：
- `algo4_sparse.py` - 稀疏优化主实现
- `test_ep_vs_bumping_sparse.py` - 核心对比测试
- `test_ep_scaling.py` - 规模测试

**结果**：代码库更简洁，易于维护

#### 2. 测试框架改进

**新增CLI参数**：
```bash
python test_ep_vs_bumping_sparse.py --configs "4x4,6x6,8x8" --grid-m 20 --grid-n 10
```

参数说明：
- `--grid-m`: 空间网格点数（默认20）
- `--grid-n`: 时间步数（默认10）
- `--configs`: B-spline配置（默认"4x4,6x6"）

#### 3. 大规模验证结果

**测试配置**: M=20, N=10, S0=100, K=100, T=1, r=0.05

| n_params | EP (完整Hessian) | Bumping2 Full (估计) | **EP加速** |
|----------|------------------|---------------------|-----------|
| 16 (4×4) | 1.44s | 3.31s | 2.3x |
| 36 (6×6) | 1.70s | 20.26s | 11.9x |
| 64 (8×8) | 1.94s | 77.91s | **40.2x** |
| 100 (10×10) | ~2.15s | ~210s (估计) | **~98x** |

**关键发现**：
1. **EP时间几乎恒定**：~2秒完成完整Hessian，与参数数量关系不大
2. **Bumping2时间O(n²)增长**：需要4×n(n+1)/2次PDE求解
3. **Crossover点**：约50参数时EP开始比Bumping2对角线更快
4. **完整Hessian优势明显**：64参数时EP快40倍

#### 4. 对角线 vs 完整 Hessian

**重要区别**：
- **对角线Hessian**: 只有 n 个元素（∂²V/∂wᵢ²）
  - 用途：单参数敏感性分析
  - Bumping2需要：2n+1次PDE求解

- **完整Hessian**: n×n 个元素（包括交叉导数 ∂²V/∂wᵢ∂wⱼ）
  - 用途：牛顿优化、不确定性量化、参数相互影响分析
  - Bumping2需要：~4×n(n+1)/2次PDE求解
  - **EP一次反向传播得到全部**

#### 5. B-spline紧支集观察

10×10配置中：
- 定义了100个系数参数
- **实际使用90个**（一行w5,*未被使用）
- 原因：B-spline紧支集性质，某些基函数在当前网格上支集为空
- EP正确识别实际依赖关系 ✓

#### 6. 下一步

**已完成**：
- ✅ 代码清理（-124K行）
- ✅ 测试框架CLI化
- ✅ 验证到64参数（40x加速）

**可选改进**：
- 并行化实现（OpenMP）预期额外2-4x加速
- 更高精度网格测试（M=50, N=25）
- 稀疏Hessian存储（200+参数）

---

## 2025-12-02 05:30

### 稀疏优化原理详解

#### 1. 问题背景

**原始 algo4 的瓶颈**：
```
PDE tape: ~11K nodes
W 矩阵: 11K × 11K 的稠密操作
但我们只需要: 16×16 的输入 Hessian
```

#### 2. 核心思想

只追踪**与输入参数相关**的 W 矩阵条目：

```
原始: W[i,j] for all i,j ∈ {0, ..., 11K}
稀疏: W[i,j] where i OR j 是输入或输入的"后代"
```

#### 3. 两个关键优化

**优化 1: 找出"相关节点"** (`_find_relevant_nodes`)

```python
def _find_relevant_nodes(nodes, var_to_idx, input_indices):
    relevant = set(input_indices)  # 输入节点一定相关

    # 前向遍历：从输入向输出传播
    for node in nodes:
        for parent, _ in node.parents:
            if parent_idx in relevant:
                relevant.add(node_idx)  # 父节点相关 → 当前节点也相关
                break

    return relevant
```

**效果**：11K 节点 → ~3K 相关节点

**优化 2: 稀疏 W 矩阵存储** (`SparseWMatrix`)

```python
class SparseWMatrix:
    # 不存储 11K×11K 稠密矩阵
    # 只存储有值的 (i,j) 对
    _data: Dict[int, Dict[int, float]]  # W[i][j] = value
```

#### 4. 修改后的算法流程

```
┌─────────────────────────────────────────────────────────┐
│  原始 algo4                    │  稀疏 algo4            │
├─────────────────────────────────────────────────────────┤
│  for node in all_nodes:        │  for node in all_nodes:│
│    pushing_stage(W, node)      │    if node in relevant:│
│    creating_stage(W, node)     │      pushing(W, node)  │
│    adjoint_stage(vbar, node)   │      creating(W, node) │
│                                │      adjoint(vbar)     │
└─────────────────────────────────────────────────────────┘
```

#### 5. Pushing Stage 优化

**原始**：
```python
for all neighbors p of i:
    for all predecessors j, k of i:
        W[j,k] += d_j * d_k * W[p,i]
```

**稀疏**：
```python
for all neighbors p of i:
    if p not in relevant: continue  # 跳过不相关
    for all predecessors j, k of i:
        if j not in relevant: continue
        if k not in relevant: continue
        W[j,k] += d_j * d_k * W[p,i]
```

#### 6. 性能对比

| 阶段 | 原始 | 稀疏 | 说明 |
|------|------|------|------|
| 节点遍历 | 11K | 3K | 跳过不相关节点 |
| W 条目 | O(n²) | O(稀疏) | 只存有值条目 |
| Pushing | O(E×d²) | O(E_rel×d²) | E_rel << E |

#### 7. 图示

```
输入层 (16个 w 系数)
    ↓
  [相关节点] ← 只追踪这些
    ↓
  [相关节点]
    ↓
  [不相关节点] ← 跳过！
    ↓
  [相关节点]
    ↓
输出 (price)
```

**结果**：60x 加速 (60s → 1s)，Hessian 完全一致 (diff < 1e-14)

#### 8. 完整 Hessian 验证

验证 EP 计算的完整 Hessian（不仅是对角线）与 Bumping2 匹配：

| 指标 | 结果 |
|------|------|
| 最大绝对误差 | 6.85e-03 |
| **最大相对误差** | **0.083%** |
| 对角线误差 | < 0.02% |
| 非对角误差 | < 0.1% |

**结论**：EP 计算的完整 Hessian 与有限差分高度吻合

#### 9. 计算图分析：相关节点判断示例

**示例函数**：`f(x) = 5x_{-2}(x_{-1} + x_0)`

**计算图节点**：
```
v_{-2} = x_{-2}  (输入)
v_{-1} = x_{-1}  (输入)
v_0 = x_0        (输入)
v_1 = 5v_{-2}    (中间节点)
v_2 = v_{-1} + v_0  (中间节点)
v_3 = v_2 v_1    (输出)
```

**相关节点判断过程**：

```python
# 步骤 1: 初始化
relevant = {v_{-2}, v_{-1}, v_0}  # 3个输入节点

# 步骤 2: 前向遍历
# 处理 v_1 = 5v_{-2}
#   父节点: v_{-2} ∈ relevant? ✓
relevant.add(v_1)  # relevant = {v_{-2}, v_{-1}, v_0, v_1}

# 处理 v_2 = v_{-1} + v_0
#   父节点: v_{-1} ∈ relevant? ✓
relevant.add(v_2)  # relevant = {v_{-2}, v_{-1}, v_0, v_1, v_2}

# 处理 v_3 = v_2 · v_1
#   父节点: v_2 ∈ relevant? ✓
relevant.add(v_3)  # relevant = {v_{-2}, v_{-1}, v_0, v_1, v_2, v_3}

# 结果: 所有 6 个节点都相关
```

**为什么所有节点都相关？**

因为所有节点都在输入→输出的依赖路径上：
```
路径 1: x_{-2} → v_1 → v_3  (x_{-2} 影响输出)
路径 2: x_{-1} → v_2 → v_3  (x_{-1} 影响输出)
路径 3: x_0 → v_2 → v_3     (x_0 影响输出)
```

**反例：不相关节点**

如果添加 `v_4 = 2 × 3 = 6` (常数计算)：
- 父节点都是常数，不在 `relevant` 中
- v_4 不会被添加到 `relevant`
- 原因：`∂²v_3/∂x_i∂x_j` 不需要通过 v_4 传播（∂v_4/∂x_i = 0）

**PDE 场景应用**：

在 B-spline PDE 中：
```
输入：16 个系数 w[i,j]
  ↓ 相关
σ(S,t) = Σw[i,j]·B_i(S)·B_j(t)  ← 依赖输入
  ↓ 相关
三对角矩阵元素 a, b, c          ← 依赖 σ
  ↓ 不相关
dt = T/N (常数)                 ← 不依赖输入！
dx = (S_max-S_min)/M (常数)     ← 不依赖输入！
  ↓ 相关
V 的时间步进                    ← 依赖 a,b,c
  ↓
输出：price
```

**性能提升**：
- 总节点：11,000
- 不相关节点：~8,000 (73%) ← 常数计算、固定网格计算
- 相关节点：~3,000 (27%)
- **只处理 3,000 节点 → 60x 加速**

---

## 2025-12-02 05:00

### 大规模测试：EP 优势随参数数量增长

**测试配置**: M=20, N=10, S0=100, K=100, T=1, r=0.05

| n_params | EP (s) | Bumping2 full (est) | **EP 加速** |
|----------|--------|---------------------|-------------|
| 16 | 1.60 | 3.50 | 2.2x |
| 36 | 1.71 | 20.15 | 11.8x |
| 64 | 1.96 | 77.05 | **39.4x** |
| 100 | 2.17 | 209.83 | **96.5x** |

**关键洞察**：
- EP 时间 ~O(1)：主要是 tape 构建，与参数数量关系不大
- Bumping2 时间 ~O(n²)：需要 4×n(n+1)/2 次 PDE 求解
- **100 参数时 EP 快近 100 倍！**

**结论**：Edge-Pushing 在高维参数空间有显著优势

---

## 2025-12-02 04:30

### 集成：algo4_sparse 到主 PDE 求解器

**变更**：`pde_aad_bspline_2d.py` 中 `solve_pde_with_aad()` 现在使用 `algo4_sparse`

**替换**：
```python
# 旧代码
from algo4_cython_simple import algo4_cython_simple
hessian = algo4_cython_simple(price_var, coeff_advars_flat)

# 新代码
from algo4_sparse import algo4_sparse
hessian = algo4_sparse(price_var, coeff_advars_flat, sort_inputs=True)
```

**结果**：
- 主求解器自动使用稀疏优化
- 1.38s (16 params) vs 原来 ~60s
- 与 Bumping2 对角线误差 < 0.02%

---

## 2025-12-02 04:00

### 修复：EP vs Bumping2 对角线匹配问题

**问题**：EP 和 Bumping2 的 Hessian 对角线不匹配

**原因分析**：
- EP 输入按 tape 出现顺序：`w0,0, w0,1, w0,2, w1,0, ..., w0,3, w1,3, w2,3, w3,3`
- Bumping2 按行主序迭代：`w0,0, w0,1, w0,2, w0,3, w1,0, ...`
- 顺序不同导致 Hessian 矩阵元素错位

**解决方案**：
- 添加 `sort_inputs_rowmajor()` 辅助函数
- `algo4_sparse()` 新增 `sort_inputs=True` 参数
- 将输入按 `(i, j)` 索引排序为行主序

**结果**：
- 对角线匹配，最大相对误差 **< 0.02%** ✓
- 与有限差分 Bumping2 完美吻合

---

## 2025-12-02 03:30

### 重大突破：稀疏 algo4 优化成功！

**优化思路**：只追踪与 B-spline 系数输入相关的 W 矩阵条目

**实现**：`algo4_sparse.py`
- 识别真正的输入参数（只有 w* 命名的系数）
- 前向遍历找出"相关节点"（从输入可达的节点）
- pushing/creating stage 只处理相关节点

**性能对比** (M=20, N=10, 16 params):

| 版本 | algo4 时间 | 加速 |
|------|-----------|------|
| 原始 algo4 | ~60s | 基准 |
| **稀疏 algo4** | **~1s** | **60× 加速** |

**正确性验证**：稀疏 vs 原始 Hessian 差异 < 1e-14 ✓

### EP vs Bumping2 (使用稀疏优化后)

| n_params | EP 总时间 | Bumping2 full (est) | EP 优势 |
|----------|----------|---------------------|---------|
| 16 | 1.39s | 3.08s | **2.2× 更快** |
| 36 | 1.57s | 18.47s | **11.7× 更快** |

**关键点**：
- EP 提供**完整 Hessian**（所有 n² 个元素）
- Bumping2 对角线只有 n 个元素
- 对比完整 Hessian 时，EP 已经显著更快

### 下一步
- 清理测试文件
- 将稀疏优化集成到主 PDE 求解器

---

## 2025-12-02 02:30

### 公平对比测试结果：EP vs 纯数值 Bumping2（优化前）

**测试配置**: M=20, N=10, S0=100, K=100, T=1, r=0.05

| n_params | EP (s) | Bumping2 diag (s) | Bumping2 full (est) | EP/Bump |
|----------|--------|-------------------|---------------------|---------|
| 16 (4×4) | 58.37 | 0.20 | 3.24 | **298× 慢** |
| 36 (6×6) | 63.19 | 0.51 | 18.54 | **124× 慢** |

### 关键发现

**EP 在当前实现下远慢于纯数值 Bumping2**

原因分析：
1. **algo4 pushing stage 是瓶颈**：PDE 图结构导致某些节点被引用上千次
2. **W 矩阵稠密化**：推送操作复杂度爆炸
3. EP 的理论优势 O(1) PDE 求解被 algo4 的 O(E×d²) 吃掉

### 结论

EP 的优势在于**避免重复构建 tape**，而不是 Hessian 计算本身更快。
当对比纯数值 Bumping2 时，EP 没有优势。

### 下一步

需要根本性优化 algo4：
1. **图簇化**：合并 PDE 时间步为黑盒
2. **稀疏子图**：只追踪与输入相关的路径
3. **并行化**：OpenMP pushing stage

---

## 2025-12-02 01:30

### 重要更正：性能对比方法论问题

**问题发现**：之前的 "4.8× 加速" 结论存在方法论问题

**原始测试代码中 Bumping2 的实现**：
```python
for i in range(n_params):
    for j in range(i, n_params):
        r_pp = solver.solve_pde_with_aad(100.0, coeffs_pp, compute_hessian=False, ...)
        # 每次都调用 solve_pde_with_aad，虽然 compute_hessian=False
        # 但仍然会构建完整的 AAD tape！
```

**两种 Bumping2 实现的区别**：
| 实现方式 | 说明 | 性能 |
|---------|------|------|
| `solve_pde_with_aad(compute_hessian=False)` | 仍构建 AAD tape | 慢 (~81s) |
| `Bumping2BSpline2D._solve_pde_simple()` | 纯数值求解，无 tape | 快 (~1-10s) |

**结论**：
- EP 17.10s vs Bumping2(with tape) 81.32s → EP 快 4.8×（**不公平对比**）
- EP 8-13s vs Bumping2(pure numerical) 1-10s → EP 可能更慢（**公平对比**）

### 根本问题
所谓 "加速" 主要来自避免重复构建 AAD tape，而不是 Edge-Pushing 算法本身比纯数值 Bumping2 更快。

### 下一步
- 明确定义公平对比基准
- 分析 EP 在大规模问题上的真实优势

---

## 2025-12-02 00:30

### 改进内容
- 修复 Cython 编译问题（重写 `symm_sparse_adjlist_cpp.pyx`，使用内联类型定义）
- 成功编译 `algo4_cython_simple` 和 `symm_sparse_adjlist_cpp` 模块
- 为 `pde_aad_bspline_2d.py` 添加 Cython→Pure Python fallback 机制

### 测试效果
- EP vs Bumping2 Hessian 精度验证通过（最大相对误差 0.05%）
- **16 params (4×4), M=10, N=10**：EP 17.10s vs Bumping2(with tape) 81.32s
- 注意：此 Bumping2 使用 `solve_pde_with_aad(compute_hessian=False)`，非公平对比

### 性能分析
- 发现 Edge-Pushing 瓶颈：PDE 图结构导致 W 矩阵变稠密
- algo4 在简单图上 ~44ms (11K nodes)，在 PDE 图上 ~24s (9K nodes)
- 原因：PDE 图中某节点被引用 1040 次，pushing stage 复杂度高
- Cython vs Pure Python: 24.5s vs 39.3s (1.6× 加速)

### 下一步计划
- 优化 Edge-Pushing pushing stage（减少邻居遍历）
- 考虑稀疏子图策略（只追踪与输入相关的路径）

---

## 2025-12-01 21:45

### 改进内容
- 修复 B-spline 系数公式 bug（`n = k + degree + 1` → `n = degree + k - 1`）
- 应用数值精度优化（var.py 使用 np.float64，arithmetic.py 使用 np.square）
- 验证稀疏 Edge-Pushing 优化（get_active_parameters + 条件 Hessian 计算）

### 测试效果
- EP vs BS 理论值差异从 3.15% 降至 0.077%
- sigma(t=0) 和 sigma(t=1) 都正确返回 0.2
- B-spline 形状 (6,6) 符合预期

### 下一步计划
- 运行完整 Hessian 测试验证 Edge-Pushing
- 性能基准测试（稀疏 vs 全量）

---

## 2025-12-01 17:30

### 改进内容
- 初始化 GitHub 仓库 (xuenailao/AAD-Edgepushing-Bspline)
- 上传核心代码 (aad_edge_pushing/)
- 配置自动版本控制 pipeline (.claude/git-auto-push.sh)
- 创建开发日志系统 (DEVLOG.md + /log + /devlog 命令)

### 测试效果
- Stop hook 自动 commit + push 功能正常
- 只提交核心文件，排除测试文件

### 下一步计划
- 继续开发 AAD Edge Pushing 算法
- 完善 2D B-spline PDE 求解器

---
