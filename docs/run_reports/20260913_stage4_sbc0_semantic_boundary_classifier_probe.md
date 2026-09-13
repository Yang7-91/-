# Stage 4-new / SBC-0 — Semantic Boundary Classifier Separability Probe Report

日期：2026-09-13
结论：**SBC-0 = NOT_ACTIONABLE**（schema 可用，但语义信号对 oracle 边界动作不可分；不进入 BR-3）

## 1. 背景：BHD-0.1 推翻 STOP 并给出新方向

- BHD-0 原报告的 STOP_STAGE4_TEMPORAL 由实现 bug（搜索空间无 TRIM、aggregation 基数不一致、重叠重复计数）造成，已被 BHD-0.1 supersede；
- BHD-0.1 修正后：**local boundary optional oracle 有显著 headroom**（±1s：ΔF1 +0.126；±5s：ΔF1 +0.254、ΔtIoU +0.322），**shot snap oracle 也有 headroom**（ΔF1 +0.054）；
- 但 BR-2 低层视觉 evidence 与 oracle 动作的 AUC-like 仅 0.53-0.58 < 0.65 —— 边界有上界、现有低层信号不可兑现；
- 因此推荐方向为 **TRY_SEMANTIC_BOUNDARY_CLASSIFIER**：用 VLM 的语义理解判断边界动作是否可分。

## 2. SBC-0 定位（diagnostic only）

本阶段**不生成任何 deployable refined candidates / predictions**，只回答一个问题：**在固定 prompt + 局部 clip 输入下，Qwen/Qwen3.5-4B 的边界动作判断（TRIM/KEEP/EXPAND）与 BHD-0.1 oracle 标签是否可分。** 所有 oracle 标签来自 weak-reference，本身为诊断用弱标签。

## 3. 数据与样本

- 数据：Dev-Tune166（cache_build_a、dev_tune_166 role manifest、166 视频全部就绪）；
- oracle labels：BHD-0.1 `boundary_oracle_labels.jsonl`（F1-optimal optional oracle @±5s）；
- 样本：**180**（left 90 / right 90；TRIM 60 / KEEP 60 / EXPAND 60；99 个视频），分层采样、seed 20260911、零跳过；
- 每样本输入：边界 ±4s 局部 clip（ffmpeg 提取，fps 2.0）+ 固定文本上下文（side / 边界在 clip 内的位置 / 候选时长 / 内外方向说明）；**不包含任何 weak-reference 信息**。

## 4. 模型与 Prompt（运行前冻结）

- 模型：**Qwen/Qwen3.5-4B**（vLLM 0.28.0，OpenAI 兼容 API，`/root/autodl-tmp/datasets` 本地媒体路径）；
- temperature 0.0、top_p 1.0、max_tokens 512、**enable_thinking=false**；
- prompt：`prompts/stage4_sbc0_semantic_boundary_classifier_v1.md`（v1 冻结，运行后未改）；
- qwen_calls = 180、vllm_calls = 180、parse_failures = **0**（修复后）。

### 运行过程记录（审计）

第一次 classify 运行 180/180 parse 失败：根因是 CLI 漏传 `enable_thinking=false`，Qwen3.5 默认 thinking 模式把 512 token 全部耗尽（finish_reason=length），JSON 永不输出。修复为显式 `enable_thinking=false`（与 Stage 1 项目惯例一致）并增加 raw_response 记录后重跑，180/180 成功。**Prompt 未修改**，该修复不构成 prompt tuning。

## 5. 结果（Dev-Tune166，180 samples）

| 指标 | 值 |
|---|---|
| schema_success_rate | **1.000** ✓ |
| accuracy | 0.378 |
| macro_F1 | **0.350**（阈值 0.45 ✗） |
| left AUC-like | **0.509**（阈值 0.65 ✗） |
| right AUC-like | **0.519**（阈值 0.65 ✗） |

Per-class：TRIM P 0.333 / R 0.517；KEEP P 0.435 / R 0.500；EXPAND P 0.389 / R **0.117**

Confusion matrix（行=oracle，列=预测）：

| | TRIM | KEEP | EXPAND |
|---|---|---|---|
| TRIM | 31 | 23 | 6 |
| KEEP | 25 | 30 | 5 |
| EXPAND | 37 | 16 | 7 |

- 模型行为特征：confidence 普遍为 0.0（极度保守）、预测大量 KEEP、几乎不预测 EXPAND（60 个 EXPAND oracle 中仅 7 个被预测为 EXPAND，37 个被吞为 TRIM）；
- **decision = NOT_ACTIONABLE**（三项规则：schema ✓ / AUC ✗ / macro-F1 ✗）。

## 6. 结论

1. **schema 完全可控**（1.0），模型能遵循固定 JSON 协议并生成可解析输出；
2. **但语义可分性不成立**：left/right AUC-like ≈ 0.51-0.52，接近随机 —— Qwen3.5-4B 在"4s 局部 clip + 通用 prompt"下对边界动作的判断与 weak-reference oracle 几乎无关；
3. 综合 BHD-0.1 与 SBC-0：**边界有上界（oracle ΔF1 +0.126~+0.254），但低层证据与现成 4B VLM 都读不出该信号**；
4. 按协议规则 B：**本轮不进入 BR-3**。SBC-0 结果为负，但完成了关键的路线排除。

## 7. 隔离性

- Hard：未运行；Heldout392：未访问；Audit36：未用于调参
- 未训练/微调模型；未下载新模型；frozen cache / Stage 3 frozen predictions：未修改
- 本轮不产生 deployable 方法；oracle 标签与评估均为诊断性质、非官方指标
- 视频、模型、outputs 未上传 GitHub（仅代码与报告）

## 8. 测试

- 定向 pytest：`tests/test_semantic_boundary_classifier.py` **12 passed**
- 全仓 pytest：**177 passed / 1 skipped**
- 输出目录：`/root/autodl-tmp/outputs/stage4_sbc0_semantic_boundary_classifier_20260913_140130/`（samples / predictions / evaluation / summaries，未上传）

## 9. 下一步建议

SBC-0 = NOT_ACTIONABLE 后，按预案选择：

1. **SBA-1 — Shot Boundary Snap Deployable**（首选低成本方向）：BHD-0.1 显示 shot snap oracle ΔF1 +0.054，用 deployable 协议（无 weak-reference）在 Dev-Tune166 验证可实现比例；
2. **Stage 5 spatial 方向**：temporal 边界在当前框架下已三次证伪（BR-1/SABR-1.1/BR-2 + SBC-0 语义探针），可接受 temporal mainline = Frozen Retrieval v0，把精力转向 spatial 定位维度；
3. 不建议在没有新语义信号来源（如更强 VLM、微调、或多模态对齐数据）前继续 temporal 边界分类路线。
