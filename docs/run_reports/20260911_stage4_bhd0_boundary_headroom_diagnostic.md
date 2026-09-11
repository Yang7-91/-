# Stage 4-new / BHD-0 — Boundary Headroom & Signal Separability Diagnostic Report

日期：2026-09-11
结论：**Recommendation = STOP_STAGE4_TEMPORAL**（边界平移 / shot 对齐 / split 三方向在 Dev-Tune166 上均无理论 headroom）

## 1. 为什么不继续调 BR-2

BR-2 的失败是结构性而非参数性：92-99% 候选被 persistence/confidence 门锁在 identity，触发的少量 EXPAND 对 aggregate 无影响。在没回答"**边界精修理论上到底有多少空间**"之前，任何新的阈值组合都只是在噪声里搜索。BHD-0 因此先做上界诊断。

## 2. BHD-0 目的与数据

- 目的：测量 ①local boundary 平移 oracle 上界；②shot-boundary snap oracle 上界；③subshot split oracle 上界；④BR-2 局部证据对 oracle 动作的区分度。
- 数据：Dev-Tune166（166 视频 / 393 candidates），cache_build_a（hash 校验通过），Stage 3 frozen predictions 作 weak-reference 评估源。
- **所有 oracle 结果标记 ORACLE_DIAGNOSTIC_ONLY / NON_DEPLOYABLE / USES_WEAK_REFERENCE**；不生成正式 refined candidates / predictions；非官方指标。

## 3. BR-0 Baseline

P 0.5731 / R 0.9459 / F1 0.6660 / tIoU 0.5438 / coverage 11.07s（本诊断复算，与 Stage 4.4.1 评估一致）

## 4. Local Boundary Oracle（±shift 联合最优，weak-reference 选择）

| max_shift | ΔP | ΔR | ΔF1 | ΔtIoU |
|---|---|---|---|---|
| 5.0s（best） | **+0.0464** | **-0.1460** | **-0.0103** | -0.0074 |

- **has_meaningful_headroom = false**：oracle 把 coverage 砍掉 31%（11.07s → 7.68s）换 precision +4.6%，但 recall 损失 -14.6%，F1/tIoU 反而下降；
- 解释：frozen candidates 的 coverage 已接近 recall 最优（R=0.946），任何收缩必然裁掉真实高光；而扩张降低 precision。**Pareto 前沿上不存在"既升 precision 又保 recall"的纯边界平移改进**；
- 这从上界层面解释了 BR-1（模型输出）、SABR-1.1（对称收缩）、BR-2（保守动作）全部失败的根本原因。

## 5. Shot Boundary Oracle

- 每个 candidate 边界 snap 到最近 shot edge（±shift）后评估，best（5s）：ΔP -0.0058 / ΔR -0.3274 / **ΔF1 -0.1345** → **无 headroom**；
- candidate/reference 边界到最近 shot edge 的距离统计见 `candidate_boundary_alignment.jsonl`：weak-reference 高光边界与 shot boundary 的对齐度不足，snap 反而远离 reference。

## 6. Subshot Split Oracle

- best（3 components）：ΔP -0.0599 / ΔR **-0.5526** / **ΔF1 -0.2676** → **无 headroom**；
- 说明问题不是"多事件 candidate 需要 split"—— split 只会让 prediction 更远离 reference 覆盖。

## 7. BR-2 Evidence Separability

- Oracle label 分布（±5s oracle）：left TRIM 0 / KEEP 232 / EXPAND 161；right TRIM 0 / KEEP 214 / EXPAND 179 —— **oracle 自己从不选择 TRIM**；
- AUC-like：left expand-vs-rest 0.532，right expand-vs-rest 0.593（max 0.593 < 0.65）→ **evidence_is_actionable = false**；
- BR-2 不动作的原因定论：低层视觉证据与"该扩/该缩"的 oracle 标签之间几乎不可分，即使提高灵敏度也只会引入错误动作。

## 8. 最终 Recommendation

> **STOP_STAGE4_TEMPORAL**

理由（决策规则 A 命中）：local boundary oracle 的 best ΔF1 = -0.010 < +0.005 且 best ΔtIoU = -0.007 < +0.005 —— 单纯边界移动的理论空间不存在；shot snap 与 split 的上界更差；BR-2 证据不可分。三条正交失败路线（BR-1 / SABR-1.1 / BR-2）+ 三项 oracle 上界诊断一致指向同一结论。

## 9. 隔离性

- Hard：未运行；Heldout392：未访问；Audit36：未使用
- Qwen：0 次；vLLM：0 次；训练：无
- frozen cache / Stage 3 frozen predictions：未修改（只读）
- 本轮不产生 deployable 方法；oracle 结果不可作为提交或官方成绩

## 10. 实现与测试

- 新增：`configs/stage4_bhd0_boundary_headroom_diagnostic.json`、`src/.../boundary_headroom_diagnostic.py`、`scripts/run_boundary_headroom_diagnostic.py`、`tests/test_boundary_headroom_diagnostic.py`
- 定向 pytest：**11 passed**；全仓：**161 passed / 1 skipped**
- 输出：`/root/autodl-tmp/outputs/stage4_bhd0_boundary_headroom_20260911_150547/`（bhd0_summary.json 等 8 个产物，未上传 GitHub）

## 11. 下一步建议

1. **temporal mainline 正式冻结 = Frozen Retrieval v0**（BR-0 / SEL-0 identity），Stage 4 temporal 关闭；
2. 若未来要重启 boundary 方向，前提是引入与高光边界真正相关的信号（如语义级边界分类器）—— 但 BHD-0 显示即使 oracle 搜索也没有可兑现的 headroom，重启的优先级应很低；
3. 研发精力建议转向：(a) spatial 定位维度（Stage 5 方向）；(b) 更高 recall 的检索源（上游 Stage 1/2 的候选覆盖）；(c) 提交策略与评测协议准备。
