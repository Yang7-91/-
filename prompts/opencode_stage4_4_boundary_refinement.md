# OpenCode Prompt — Stage 4.4 Temporal Boundary Refinement Protocol + Scaffold

你是 OpenCode，当前角色是：Stage 4.4 工程实现者。

## 一、本轮目标

实现 Stage 4.4.0：Temporal Boundary Refinement Protocol + Scaffold。

本轮只做工程骨架，不做复杂算法，不做正式调参。

目标：

1. 读取 Stage 4.2 frozen candidate cache；
2. 定义 boundary refinement 输出 schema；
3. 实现 BR-0 identity baseline；
4. 实现 validate-boundary-refinement；
5. 实现 replay/evaluate 最小链路；
6. 证明 BR-0 与 frozen baseline 指标一致；
7. 保存 run config、summary、logs；
8. 不访问 Heldout，不调用 Qwen/vLLM。

## 二、当前冻结事实

- Dataset：aic_highlight_dev_v1.1
- Frozen upstream：Stage 4.2 cache_build_a
- Source split：Dev183 + Hard248
- Heldout392：LOCKED
- Final Stage 4.3 selection result：
  - SEL-1：NO_ELIGIBLE_CONFIGURATION
  - SEL-2：NO_ELIGIBLE_CONFIGURATION
  - SEL-3：Optional Frozen Prune / DEV_ONLY_SIGNAL
- 下一阶段优先方向：Boundary Refinement

## 三、禁止事项

禁止：

- 访问 Heldout；
- 调用 Qwen；
- 调用 vLLM；
- 下载新模型；
- 下载新数据；
- 修改 frozen candidate cache；
- 修改 Stage 3 frozen predictions；
- 修改 Prompt；
- 修改 parser；
- 修改 merge 逻辑；
- 使用 Audit36 调参；
- 编写 video_id / audit_id 特定规则；
- 同时实现 frame selection、bbox、SAM、tracking；
- 大范围重构；
- reset / clean / stash / rebase / force push。

## 四、允许事项

允许：

- 新增 stage4_4 相关协议文件；
- 新增 boundary_refinement 模块；
- 新增 CLI；
- 新增测试；
- 新增 docs；
- 新增小型 machine-readable summary；
- 使用 Dev role 做最小验证；
- 使用 Hard role 只做 identity sanity，不做调参；
- commit / push 前必须向用户汇报 diff。

## 五、建议文件结构

优先在现有项目结构内最小新增：

```text
configs/
  stage4_4_boundary_protocol.json

src 或 aic_video_highlight/
  boundary_refinement/
    __init__.py
    schema.py
    identity.py
    validate.py
    replay.py
    evaluate.py
    cli.py

tests/
  test_boundary_refinement_schema.py
  test_boundary_refinement_identity.py
  test_boundary_refinement_validate.py

docs/
  stage4_4_boundary_refinement_protocol.md
```

如果仓库现有结构不同，遵循现有结构，不要硬建冲突目录。

六、BR-0 Identity Baseline

BR-0 行为：

输入 frozen merged candidates；
输出相同 video_id；
输出相同 candidate_id；
start_sec 不变；
end_sec 不变；
score 不变；
reason 不变；
source_chunk 不变；
增加 method metadata，例如 method=BR-0-identity；
不新增候选；
不删除候选；
不改变顺序。

BR-0 目标：

replay 后 segment 数与 frozen baseline 完全一致；
evaluation 指标与 frozen baseline 完全一致；
若不一致，必须停止并定位原因。

七、Boundary 输出 schema 要求

每条 refined candidate 至少包含：

video_id
original_candidate_id
refined_candidate_id
start_sec
end_sec
original_start_sec
original_end_sec
score
reason
source_chunk
method
provenance

约束：

0 <= start_sec < end_sec <= video duration；
refined interval 必须来自某个 frozen candidate；
BR-0 中 start/end 必须完全等于 original start/end；
后续非 identity 方法允许修改 start/end，但不得改变事件身份字段；
不允许 NaN / Inf；
不允许重复 candidate id；
不允许跨 video 引用 candidate。

八、Validator 要求

validate-boundary-refinement 必须检查：

输入 cache semantic hash 是否匹配；
video_id 是否存在于 cache；
original_candidate_id 是否存在；
输出数量是否合法；
start/end 是否在 video duration 内；
start < end；
BR-0 是否完全 identity；
是否访问 Heldout；
是否包含 reference / metric / audit / human label leakage；
是否包含不允许字段；
输出是否 canonical / deterministic。

九、CLI 建议

先搜索现有 CLI 风格：

rg -n "candidate_cache|selection|validate-selection|replay|evaluate|assess|stage4_3|argparse|click|typer" .

不要盲扫整个仓库。

建议新增或复用 CLI：

python -m aic_video_highlight.boundary_refinement.cli identity \
  --cache-dir <cache_build_a> \
  --role dev \
  --output-dir <out_dir>/dev/br0

python -m aic_video_highlight.boundary_refinement.cli validate \
  --cache-dir <cache_build_a> \
  --input <out_dir>/dev/br0/refined_candidates.jsonl

python -m aic_video_highlight.boundary_refinement.cli replay \
  --input <out_dir>/dev/br0/refined_candidates.jsonl \
  --output <out_dir>/dev/br0/replayed_predictions.jsonl

python -m aic_video_highlight.boundary_refinement.cli evaluate \
  --predictions <out_dir>/dev/br0/replayed_predictions.jsonl \
  --references <frozen_reference_source> \
  --output <out_dir>/dev/br0/evaluation.json

具体命令以仓库已有结构为准，不要臆造不可运行命令。

十、测试要求

至少新增测试覆盖：

BR-0 不改变 start/end；
BR-0 不改变 candidate 顺序；
validator 能拒绝非法时间；
validator 能拒绝未知 original_candidate_id；
validator 能拒绝 Heldout；
validator 能拒绝 NaN/Inf；
replay 输出数量与输入一致；
identity evaluation 与 frozen baseline 一致。

运行：

pytest tests/test_boundary_refinement*.py
pytest

如果全仓测试太慢，先跑定向测试，再说明全仓是否运行。

十一、Git 规则

开始前：

git status --short --branch
git remote -v
git log --oneline --decorate -5

修改后：

git diff --stat
git diff --check
git diff
pytest tests/test_boundary_refinement*.py

未经用户确认，不要 push。

如果本轮明确要求 push 到 GitHub，则只在以下条件全部满足后 push：

tests pass；
diff 可解释；
不包含数据、模型、cache、token；
git status clean after commit；
remote 是用户确认的仓库。

Commit message 建议：

docs: add stage4.4 boundary refinement plan

如果同时实现 scaffold：

feat: add stage4.4 boundary refinement scaffold

十二、最终汇报格式

请按以下格式汇报：

当前仓库
路径
branch
HEAD
remote
本轮新增文件
文件列表
每个文件用途
Stage 4.4 scaffold 状态
是否只做 protocol / schema / BR-0
是否未访问 Heldout
是否未调用 Qwen/vLLM
测试结果
定向测试
全仓测试，如有
Git 状态
diff stat
commit hash
push 是否成功
下一步
如果 scaffold 完成：下一轮进入 BR-1/BR-2 简单边界规则实验
如果 scaffold 未完成：列出阻塞点和最小修复建议
