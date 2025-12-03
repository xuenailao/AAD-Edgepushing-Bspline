# 文件版本管理与归档

帮我管理项目文件的版本和归档。支持以下子命令：

## 使用方式

- `/archive tree` - 查看文件结构树
- `/archive review <文件路径>` - 审查文件版本历史
- `/archive move <文件路径>` - 归档旧版本文件
- `/archive clean` - 批量清理临时文件

---

## 1. 查看文件结构树 (`tree`)

显示当前项目的文件结构，标注：
- 最后修改时间
- 文件大小
- 是否被 git 跟踪

**执行**：
```bash
# 显示 aad_edge_pushing/ 的结构（3层深度）
tree -h -L 3 aad_edge_pushing/

# 列出被 git 跟踪的文件
git ls-files aad_edge_pushing/
```

---

## 2. 审查文件版本 (`review <文件路径>`)

对指定文件显示：
- Git 历史记录（最近5次提交）
- DEVLOG.md 中涉及该文件的条目
- archive/ 中的归档版本（如果存在）
- 文件信息（大小、修改时间）

**执行**：
```bash
# Git 历史
git log --follow --oneline -5 -- <文件路径>

# 文件信息
ls -lh <文件路径>

# 检查归档版本
ls -lh archive/<文件路径>_* 2>/dev/null || echo "无归档版本"
```

然后搜索 DEVLOG.md 中包含该文件名的条目。

---

## 3. 归档旧版本 (`move <文件路径>`)

将指定文件归档到 archive/ 文件夹：

**步骤**：
1. 检查文件是否存在
2. 创建时间戳：`YYYYMMDD_HHMMSS`
3. 创建目录结构：`mkdir -p archive/$(dirname <文件路径>)`
4. 复制文件：`cp <文件路径> archive/<文件路径_时间戳>`
5. 在 `archive/ARCHIVE_LOG.md` 中记录：
   ```markdown
   ## YYYY-MM-DD HH:MM:SS
   **归档文件**: <文件路径>
   **归档位置**: archive/<文件路径_时间戳>
   **原因**: [询问用户]
   ---
   ```
6. 询问是否删除原文件（默认保留）

**注意**：archive/ 目录在 .gitignore 中已排除，不会推送到 GitHub

---

## 4. 批量清理 (`clean`)

扫描并清理临时文件：

**扫描模式**：
```bash
# 查找未跟踪文件
git ls-files --others --exclude-standard

# 查找可疑模式
find . -name "*_old.*" -o -name "*_backup.*" -o -name "*.bak" -o -name "*.tmp"
```

**处理流程**：
1. 列出所有候选文件
2. 分类显示：
   - 测试文件：`test_*.py`（非测试目录中）
   - 备份文件：`*_old.*`, `*_backup.*`, `*.bak`
   - 临时文件：`*.tmp`, `*.log`
   - 编译产物：`*.pyc`, `*.so`（如果未被 .gitignore 排除）
3. 询问用户每个文件的处理方式：
   - **归档**（移到 archive/）
   - **删除**
   - **保留**
4. 执行用户选择的操作

---

## 归档目录结构示例

```
archive/
├── ARCHIVE_LOG.md              # 归档操作日志
├── aad_edge_pushing/
│   ├── aad/
│   │   └── tape_20241201_143022.py
│   └── edge_pushing/
│       └── algo3_20241130_091500.pyx
└── test_scripts/
    └── test_cython_speedup_20241125_103045.py
```

---

## 实现说明

当用户运行 `/archive <子命令>` 时：

1. 解析子命令参数
2. 根据子命令执行相应操作
3. 对于 `move` 和 `clean`，在执行前显示将要进行的操作并要求确认
4. 所有归档操作都记录到 `archive/ARCHIVE_LOG.md`
5. 归档前检查是否有未提交的改动，建议先 commit

## 注意事项

- 归档不会修改 git 历史
- archive/ 目录已在 .gitignore 中排除
- 建议定期运行 `/archive clean`
- 归档前应先 commit 当前改动
