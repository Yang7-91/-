# AIC-VideoHighlight 下一阶段计划：Stage 4.4 Temporal Boundary Refinement

日期：2026-09-08

## 1. 当前项目状态

项目已经完成 Stage 1 到 Stage 4.3 的主要时间候选实验链路。

目前已经具备：

- Qwen/Qwen3.5-4B + high_recall_retrieval_v0 的稳定粗召回基线；
- aic_highlight_dev_v1.1 开发数据集；
- Dev183 + Hard248 共 431 条 frozen v0 结果；
- Frozen Candidate Cache，可在不调用 Qwen/vLLM 的情况下重放候选；
- Lightweight Candidate Selection 实验结论。

当前最重要结论是：粗召回 Recall 较高，但候选区间普遍偏宽，prediction coverage 长期接近全片。Selection 方向收益有限，下一阶段应优先验证时间边界精修。

## 2. 为什么进入 Stage 4.4

Stage 4.1 人工审计显示，Boundary-involved 是主要错误家族之一，其中 over-wide 是最常见问题。Stage 4.3 的 lightweight selection 只带来极小收益，score 阈值方案还会明显伤害 Recall。

因此下一阶段重点不应继续筛候选，而应研究：

> 在正确事件已经进入候选池的前提下，能否缩短过宽时间边界，同时尽量保护 Recall。

## 3. Stage 4.4 的目标

Stage 4.4 的目标是建立一个可复现、可验证、可消融的 Temporal Boundary Refinement baseline。

核心问题：

1. 能否减少候选区间过宽导致的 over-prediction；
2. 能否提升 Precision、F1 或 temporal IoU；
3. 能否在 Recall guardrail 下避免误删高光事件；
4. 能否保护 near-full positive controls；
5. 能否保持 frozen candidate cache 不变，只在下游修改 start/end。

## 4. 输入与约束

唯一上游输入：

- Stage 4.2 frozen candidate cache：cache_build_a；
- Stage 3 frozen predictions / weak references，仅用于开发评估；
- SEL-3 fixed rules 只能作为可选输入，不得重新调参。

禁止：

- 访问 Heldout392；
- 重新调用 Qwen/vLLM 生成候选；
- 修改 frozen candidate cache；
- 修改 Prompt、parser、merge；
- 使用 Audit36 调参；
- 编写 video_id / audit_id 特定规则；
- 同时开发 frame selection、bbox、SAM、tracking。

## 5. 建议阶段拆分

### Stage 4.4.0：Protocol + Scaffold

只做协议和工程骨架：

- 定义 boundary refinement 输入输出 schema；
- 定义 validator；
- 定义 identity baseline BR-0；
- 定义 replay / evaluate / assess 流程；
- 定义 Recall guardrail；
- 定义 promotion gate；
- 不运行正式调参。

### Stage 4.4.1：Identity Replay

实现 BR-0：

- 输入 frozen candidates；
- 输出完全相同 start/end；
- 证明 evaluation 与 frozen baseline 完全一致；
- 作为后续 boundary 方法的控制组。

### Stage 4.4.2：Simple Boundary Baselines

在 Dev role 上测试轻量规则，例如：

- 固定比例收缩；
- 固定秒数裁边；
- 仅对过长候选收缩；
- 保留 near-full positive controls；
- 保留多事件候选保护。

每次只改变一个变量。

### Stage 4.4.3：Hard Stress

只有 Dev 上通过 Recall guardrail 和 promotion gate 的配置才能进入 Hard229。

Hard 结果只做压力验证，不能反向调参。

### Stage 4.4.4：Freeze

若存在有效方法，则冻结：

- 代码；
- 配置；
- protocol；
- metrics；
- decision rule；
- 输出目录；
- semantic hash。

如果无方法通过 gate，也要记录负结果。

## 6. 验收标准

Stage 4.4 成功不等于一定提升指标。最小成功标准是：

1. BR-0 identity baseline 能完全重放 frozen baseline；
2. 所有 boundary 输出均通过 schema 和合法区间检查；
3. 没有访问 Heldout；
4. 没有调用 Qwen/vLLM；
5. 所有实验均可从 frozen cache 复现；
6. 所有指标明确标注为 weak-reference development metrics；
7. 若方法失败，能明确说明失败原因。

## 7. 当前建议

下一轮 OpenCode 开发不应直接上复杂模型，也不应继续调 score selection。

推荐先执行：

> Stage 4.4.0 — Boundary Refinement Protocol + Engineering Scaffold

即先把协议、schema、validator、BR-0 identity control 和最小 CLI 建起来，再进入具体边界算法。
