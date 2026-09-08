# Stage 4.4.1 — BR-0 Real Cache Identity Replay Report

日期：2026-09-08
结论：**Stage 4.4.1 = PASS**（identity replay 完全等价，可进入 Stage 4.4.2）

## 1. Role manifests 恢复

- 本地来源：`C:\Users\p0220\Desktop\AIC_Stage4_Role_Manifests_Recovery\canonical\`（源自 Stage 4.3 `00_PROTOCOL` 冻结目录，逐字节复制）
- 规范文件名（4 个，同目录）：
  - `dev_tune_166.json`（166 records）
  - `hard_stress_229.json`（229 records）
  - `audit36_diagnostic.json`（36 records，本轮未运行，仅目录校验）
  - `stage4_3_role_manifest_summary.json`
- Semantic SHA-256（协议校验目标）全部匹配：
  - dev_tune = `bceb8203…368ec71` ✓
  - hard_stress = `81d734ee…4405687` ✓
  - audit_diagnostic = `98fd987b…e115e1780c` ✓
  - summary = `6eccf825…11b23e0` ✓
- AutoDL 落位：`/root/autodl-tmp/outputs/stage4_3_formal_431/FORMAL_431/protocol/role_manifests/`
- `validate-roles`：**PASS**（`{dev_tune: 166, hard_stress: 229, audit_diagnostic: 36}`，pairwise_disjoint=true，union=431，summary hash 与协议冻结值精确一致）

## 2. Stage 3 frozen outputs

- Dev：`/root/autodl-tmp/AIC-VideoHighlight/outputs/stage3_dev_full_baseline_v1/results/predictions.jsonl`（183 行，run_config 完整）
- Hard：`/root/autodl-tmp/AIC-VideoHighlight/outputs/stage3_hard_full_baseline_v1/results/predictions.jsonl`（248 行，run_config 完整）
- 未做任何修改，只读使用

## 3. Cache 状态

- 路径：`/root/autodl-tmp/outputs/stage4_2_formal_431/FORMAL_431/cache_build_a`
- record_count = 431；raw = 1293；merged = 1277；split = {dev: 183, hard: 248}
- global_semantic_sha256 = `4b515a6d6fb47073413c686214c3fa5f97293655a3824241eb3305b9e7753246`
- 项目 validator（`run_candidate_cache.py validate`）：**PASS**

## 4. 执行环境

- 代码：`main @ 644b81d`（含本轮 duration 容差修复，见第 7 节）
- 输出目录：`/root/autodl-tmp/outputs/stage4_4_1_br0_identity_20260908_201456`

## 5. BR-0 identity replay 结果

### Dev role（dev_tune_166）

| 步骤 | 结果 |
|---|---|
| refine（BR-0） | **PASS**：393 candidates，`IDENTITY=393, REFINE=0, IDENTITY_FALLBACK=0`，refinement hash `e7b85a5f…2416f52` |
| validate-refinement | **PASS**（exit 0，hash 一致） |
| replay | **PASS**：166 records / 393 segments |
| evaluate | **PASS**：mean_recall `0.9444833053…`，tiou `0.5442217072…`，f1 `0.6665055259…`，precision `0.5736344361…`，coverage `0.9099025603…` |
| SEL-0 baseline 链 | PASS（select→replay→evaluate） |
| assess | recall guardrail **全过**（deltas 全 0）；promotion gate `pass=false`（唯一项 `min_delta_mean_temporal_iou: 0.0 < 0.01`）—— **identity 控制组的预期行为**，promotion gate 为 BR-1 改进方法设计 |

### Hard role（hard_stress_229）

| 步骤 | 结果 |
|---|---|
| refine（BR-0） | **PASS**：803 candidates，`IDENTITY=803, REFINE=0, IDENTITY_FALLBACK=0`，hash `07c5388f…92e33b18` |
| validate-refinement | **PASS** |
| replay | **PASS**：229 records / 803 segments |
| evaluate | **PASS**：mean_recall `0.9564025398…`，tiou `0.2932398590…`，f1 `0.4186983644…` |
| SEL-0 baseline 链 | PASS |
| assess（hard_stress_gate） | **pass=true** |

### Identity 证明

1. BR-0 决策：1196/1196 候选全部 `IDENTITY`，0 个 REFINE/FALLBACK；
2. BR-0 replay 输出与 SEL-0 replay 输出**字节级完全一致**：
   - dev：SHA-256 `40b3efe503ae0d3b…`（166 行）
   - hard：SHA-256 `73b05f068c3e1186…`（229 行）
3. BR-0 与 SEL-0 的 evaluation aggregate **逐位相等**（dev/hard 均为 true）；
4. recall guardrail 所有 delta = 0。

## 6. 隔离性

- Heldout392 accessed = **false**
- Qwen calls = **0**；vLLM calls = **0**；GPU 推理 = 0
- frozen cache 未修改；Stage 3 frozen outputs 未修改；role manifests 未修改
- 未做 BR-1；未做调参；无 video_id/audit_id 特定规则

## 7. 过程中的最小代码修复（已测试 + 已 commit）

- `644b81d fix(stage4.4): tolerate 6-decimal rounding of frozen candidate ends vs full-precision durations`
- 根因：frozen cache 的 merged candidates `end_sec` 为上游 Stage 1 固有的 6 位小数舍入值，与全精度 `duration_sec` 相差最多 5e-7 秒；59/1277 候选触发严格 `end<=duration` 检查失败。
- 修复：新增 `_DURATION_TOLERANCE_SEC = 1e-6`（远小于任何实际时间粒度，仅消除浮点表示噪声），用于 refine 前置校验与 event-identity guard。
- 测试：定向 26 passed；全仓 71 passed / 1 skipped。
- 备注：dev promotion gate 的 `pass=false` 为 identity 控制组的预期结果（gate 要求 tiou 提升 ≥ 0.01），不构成 Stage 4.4.1 失败；hard gate PASS。

## 8. 关键产物路径

- 输出目录：`/root/autodl-tmp/outputs/stage4_4_1_br0_identity_20260908_201456`
- dev：`refinement_result.json` / `replayed_predictions.jsonl` / `br0_evaluation.json` / `sel0_evaluation.json` / `assessment.json`
- hard：同结构
- 汇总：`summary.json`；日志：`logs/*.log`（14 个）
- 云端产物体积 2.5M，未上传 GitHub（仅本报告）

## 9. 下一步

Stage 4.4.1 = **PASS**，下一轮进入 **Stage 4.4.2 Simple Boundary Baselines**：

- 先跑 Dev-Tune166；
- BR-1 类轻量规则：固定秒数裁边、固定比例收缩、long-only shrink；
- 保留 near-full positive controls 与多事件候选保护；
- Recall guardrail（min_mean_recall 0.94 等）+ promotion gate（Δtiou ≥ 0.01）按协议执行；
- 不直接进 Hard；不使用 Audit36 调参。
