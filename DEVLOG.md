# Development Log

> 开发日志 - 记录每次改进、测试效果和下一步计划
> 使用 `/log` 命令添加新记录，`/devlog` 命令查看历史

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
