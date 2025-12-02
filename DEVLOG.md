# Development Log

> 开发日志 - 记录每次改进、测试效果和下一步计划
> 使用 `/log` 命令添加新记录，`/devlog` 命令查看历史

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
