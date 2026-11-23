# 代码改动可见性约定

本仓库的改动必须"看得见"。任何一次代码修改都要在回复里给出 diff，并且提交后由钩子再打印一次。
这套机制由两个文件和一个 git 钩子组成，已固化在本仓库。

## 第一步：安装钩子（只需一次）

```bash
python3 tools/install_diff_hooks.py
```

输出 `installed .git/hooks/post-commit` 即完成。钩子不随仓库分发，新克隆的仓库要重跑一次。

## 组成部分

| 文件 | 作用 |
|---|---|
| `tools/pic_diff.py` | 打印 worktree / staged / 最近 N 个提交的逐文件 diff，并按 source 与 docs 分组 |
| `tools/install_diff_hooks.py` | 安装 `post-commit` 钩子（幂等，可重复执行） |
| `.git/hooks/post-commit`（由上面脚本生成） | 每次提交后自动运行 `tools/pic_diff.py --last` |

## 用法

```bash
# 未提交的改动
python3 tools/pic_diff.py

# 只看统计
python3 tools/pic_diff.py --stat

# 最近一个提交 / 最近三个提交
python3 tools/pic_diff.py --last
python3 tools/pic_diff.py --last 3

# 只看某几个文件（默认截断 400 行/文件）
python3 tools/pic_diff.py --files vllm/v1/pic/runtime.py --max-lines 200
```

输出会把 `vllm/`、`tests/`、`tools/` 归为 **source**，其余归为 **docs/tools only**，
这样审阅时先看代码再看文档。

## 约定

1. **改代码用编辑工具**（`edit` / `write`），不要用脚本批量写源码——工具调用会在界面上直接显示 diff。
2. 每轮代码改动结束前运行一次 `python3 tools/pic_diff.py`，把输出贴进回复。
3. 提交信息里说明"改了什么 + 为什么"，diff 由钩子自动打印。
4. 文档类批量生成（README、stage plan）可以用脚本，但涉及代码文件时同样要先给出 diff。

## 安装（新克隆的仓库）

```bash
python3 tools/install_diff_hooks.py
```

输出 `installed .git/hooks/post-commit` 即完成。

## 为什么不是 GUI 插件

Harness 的界面插件需要客户端插件包（`@deepseek-ai/dsh-client-*`）与开发构建环境；
当前工作区没有该 SDK，也没有 `pnpm run dev:web` 在跑，因此客户端插件无法在本环境编译和热加载。
上面的方案不依赖 SDK，但把"每次改动必须有 diff"变成了仓库级约束，效果等价于一个常驻的提示。
