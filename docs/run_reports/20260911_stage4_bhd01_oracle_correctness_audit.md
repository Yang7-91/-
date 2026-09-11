# Stage 4-new / BHD-0.1 — Oracle Correctness & Metric Consistency Audit Report

日期：2026-09-11
结论：**BHD-0 original report is superseded by BHD-0.1.** 修正后 Recommendation = **TRY_SEMANTIC_BOUNDARY_CLASSIFIER**（不再 STOP）。

## 1. 为什么需要 BHD-0.1 审计

BHD-0 报告存在四处必须解释的异常：①optional oracle 的 ΔF1/ΔtIoU 为负（与 upper-bound 语义矛盾）；②action 分布 EXPAND-only 但 coverage 反降 31%；③forced diagnostic 与 upper bound 混用；④baseline coverage 单位漂移（11.07s vs 0.9099）。

## 2. 代码审计结果：确认 4 个实现级 bug + 1 个评估语义缺陷

| # | Bug | 影响 |
|---|---|---|
| 1 | oracle 搜索空间 `ds,de ≥ 0` 只允许向外扩张，**TRIM 完全不在空间内** | BHD-0 实际是 forced-expand diagnostic，不是 optional upper bound |
| 2 | aggregation 基数不一致：oracle 用 per-candidate mean（393 基数），baseline 用 per-video mean（166 基数） | ΔF1/Δcoverage 负值是 aggregation artifact，不是真实上界 |
| 3 | coverage 单位混用（秒 vs ratio） | 与 Stage 4.4 evaluate（ratio）不可比 |
| 4 | right_delta 符号与约定相反（`end - new_end`） | BR-2 detail 字段符号错误 |
| 5 | oracle 变体互相重叠时 merged 指标重复计数（recall/f1/tIoU 可 > 1） | 修复：评估前 merge overlapping predicted intervals（与部署语义一致） |

## 3. 修复内容

- `boundary_headroom_diagnostic.py` 重写：**视频级贪心 oracle**（与 Stage 4.4 evaluate 相同的 merged per-video 语义）+ **identity-inclusive 搜索（TRIM/KEEP/EXPAND 双向）** + **merge overlapping predictions** + forced/optional 显式分离（`forced_diagnostic` / `upper_bound` / `identity_included` 字段）+ 三目标（f1 / temporal_iou / precision_under_recall_guard）；
- `br2_asymmetric_boundary.py`：right_delta 统一为带符号约定（`refined_end - original_end`），validator 同步；
- baseline consistency 修复后 **与 Stage 4.4 evaluate 输出完全一致**（见下）。

## 4. BR-0 Baseline consistency（修复后）

P 0.5736 / R 0.9445 / F1 0.6665 / tIoU 0.5442 / **coverage_ratio 0.9099**（coverage_seconds 10.96）
—— 与 Stage 4.4.1 / SABR / BR-2 轮的 evaluate 输出逐位一致。**PASS**。

## 5. Sanity Tests（全部通过）

1. **identity dominance**：optional oracle（含 identity）F1/tIoU ≥ baseline（贪心构造保证）—— PASS；
2. forced diagnostic 标记 `forced_diagnostic=true, upper_bound=false` —— PASS；
3. 动作符号约定六情形（left/right × trim/expand）—— PASS（含 BR-2 right_delta 符号修正）；
4. coverage direction sanity（EXPAND-only 时 coverage 不降）—— PASS；
5. baseline consistency（merged 语义 == evaluate_weak_references）—— PASS；
6. summary schema 分离 optional/forced —— PASS。

## 6. 修正后的 Dev-Tune166 Oracle 结果（全部 NON_DEPLOYABLE / USES_WEAK_REFERENCE）

### Optional oracle upper bounds（identity-inclusive，贪心）

| 方向 | best 配置 | ΔP | ΔR | ΔF1 | ΔtIoU | Δcoverage | headroom |
|---|---|---|---|---|---|---|---|
| **Local boundary** | F1 @ ±5s | **+0.312** | **+0.032** | **+0.254** | **+0.322** | -0.321 | **TRUE** |
| Local（±1s） | F1 @ ±1s | +0.148 | +0.019 | +0.126 | +0.153 | -0.196 | TRUE |
| Local（recall guard @ ±1s） | P-guard | +0.157 | -0.013 | +0.112 | +0.135 | -0.221 | TRUE |
| **Shot snap** | F1 @ ±5s | +0.053 | +0.028 | **+0.054** | +0.063 | -0.056 | **TRUE** |
| Subshot split | 3 components | +0.053 | -0.014 | +0.037 | +0.043 | -0.088 | FALSE（ΔR 略破 guard） |

### Forced diagnostics（强制执行，可为负，非 upper bound）

- local forced F1@±5s：ΔF1 **-0.176**（强制所有候选移动明显变差）；
- shot forced snap@±5s：ΔF1 **+0.152** 但 recall 语义受重叠影响、解释需谨慎；
- subshot forced@3：ΔF1 +0.035 / ΔR -0.019。

## 7. 修正后的核心洞察

1. **BHD-0 的"STOP"结论被推翻**：修正 aggregation 并把 TRIM 纳入搜索后，local boundary oracle 在 ±1s 内即可 ΔF1 +0.126（ΔP +0.148、ΔR +0.019）—— **边界精修存在真实、可观的 headroom**；
2. **frozen candidates 普遍偏宽**：oracle 的 F1 最优解把 coverage 从 0.910 降到 0.589-0.714，同时 recall 反而略升 —— "裁掉的是冗余而非高光"；
3. **BR-2 失败的真正原因定论**：不是没有空间，而是**低层视觉证据（histogram/edge/contrast 与 core 相似度）对 oracle 动作的区分度不足**（AUC-like 0.53-0.58 < 0.65；oracle label 分布 TRIM 142-209 / KEEP / EXPAND 150-179，动作空间本活跃）；
4. **shot boundary snap 也有独立 headroom**（ΔF1 +0.054），可作为低成本 deployable 尝试方向。

## 8. 最终 Recommendation（审计后）

> **TRY_SEMANTIC_BOUNDARY_CLASSIFIER**

决策路径：local oracle 有 headroom（规则 B 前半）→ 但 BR-2 证据 AUC < 0.65（规则 B）→ 需要更强的**语义级**边界信号才能兑现上界。次选：TRY_SHOT_BOUNDARY_SNAP（shot snap oracle ΔF1 +0.054 为独立可验证的低成本方向，可在后续轮次以 deployable 协议验证）。

## 9. 隔离性

- Hard：未运行；Heldout392：未访问；Audit36：未使用；Qwen：0 次；vLLM：0 次；训练：无；frozen cache：未修改
- oracle 结果全部标记 NON_DEPLOYABLE，不构成提交方法或官方成绩

## 10. 测试

- 定向 pytest：**16 passed**（含 6 项新 sanity tests）
- 全仓 pytest：**165 passed / 1 skipped**
- 输出：`/root/autodl-tmp/outputs/stage4_bhd01_oracle_audit_20260911_155423/`（未上传 GitHub）

## 11. 下一步建议

1. **SBA-0 / Semantic Boundary Classifier 探索**（下一轮）：用与高光边界相关的语义信号（如 Qwen 视觉描述中的场景转换提及、音频/字幕辅助信号）替换低层证据，先做与 BHD-0.1 相同的 separability 诊断，AUC ≥ 0.65 才进入 deployable 方法；
2. **Shot-boundary snap deployable 实验**（低成本备选）：SBA-1，Dev-Tune166 上以贪心 snap 的 deployable 协议验证 ΔF1 +0.054 的可实现比例；
3. 任何新方法在 Dev gate 通过前不触碰 Hard/Heldout。
