# Stage 4-new / SBA-1 — Conservative Shot-Boundary Snap (Dev) Report

日期：2026-09-13
结论：**SBA-1 Dev result = NEGATIVE_RESULT / NO_PROMOTION_TO_HARD**

## 1. 为什么选择 SBA-1（而不是 BR-3 / 继续调参）

- SBC-0（语义边界分类器探针）为 NOT_ACTIONABLE：Qwen3.5-4B 的动作判断与 oracle 标签 AUC ≈ 0.51-0.52，按协议不进入 BR-3；
- BHD-0.1 显示 **shot snap oracle 存在独立 headroom**（ΔF1 +0.054），且该方向**不需要模型、不依赖语义信号**；
- 因此本轮把 shot snap 从 oracle 转为 **deployable**（无 weak-reference 决策），验证该 headroom 的可兑现比例 —— 这是进入 Stage 5 spatial 前对 temporal 的最后一轮低成本验证。

## 2. SBA-1 与 BHD-0.1 shot oracle 的关系

| | BHD-0.1 shot snap oracle | SBA-1 |
|---|---|---|
| snap 目标选择 | weak-reference 选最优（oracle） | **仅吸附最近 shot boundary（deployable）** |
| 定位 | ORACLE_DIAGNOSTIC_ONLY / NON_DEPLOYABLE | **deployable, uses_weak_reference_for_decision = false** |
| 性质 | 上界 | 可实现比例验证 |

## 3. 数据与检测

- Dev-Tune166（166 视频 / 393 candidates），视频齐备（复用已校验下载）；cache_build_a、dev_tune_166、frozen predictions 均沿用已验证 hash；
- shot detection：BHD-0.1 同款确定性方法（OpenCV 内容差分，stride 0.5s，median+3·1.4826·MAD 与 p85 双阈值，min shot len 1.0s）；166 视频共检测 **235** 个 shot boundary（每配置独立检测，结果一致）。

## 4. 预注册配置（运行后未改）

| 参数 | C1 | C2 | C3 |
|---|---|---|---|
| snap_window_sec | 1.0 | 2.0 | 3.0 |
| expand | 允许 ≤1.0s | 允许 ≤1.5s | **禁止（inward-only）** |
| max_trim_each_side | 1.0 | 2.0 | 2.0 |
| change ratio cap | 0.15 | 0.20 | 0.20 |
| min overlap | 0.85 | 0.80 | 0.80 |
| min candidate duration | 6.0 | 8.0 | 10.0 |

## 5. 结果（Dev-Tune166，weak-reference development metrics，非官方指标）

BR-0 baseline（本轮重算，与 Stage 4.4.1/BR-2/BHD-0.1 逐位一致）：P 0.573634 / R 0.944483 / F1 0.666506 / tIoU 0.544222 / coverage 0.909903；R=0: 1；R<0.5: 9

| 配置 | snap/identity/fallback | P (Δ) | R (Δ) | F1 (Δ) | tIoU (Δ) | cov (Δ) | 新R=0 | 新R<0.5 | Gate |
|---|---|---|---|---|---|---|---|---|---|
| C1 | 39 / 60 / 294 | 0.5752 (+0.0016) | 0.9411 (-0.0034) | 0.6661 (-0.0004) | 0.5435 (-0.0007) | 0.9039 (-0.0060) | 0 | 0 | **FAIL** |
| C2 | 30 / 27 / 336 | 0.5770 (+0.0033) | 0.9409 (-0.0035) | 0.6669 (+0.0004) | 0.5446 (+0.0004) | 0.9019 (-0.0080) | 0 | 0 | **FAIL** |
| C3 | 10 / 24 / 359 | 0.5759 (+0.0023) | 0.9423 (-0.0022) | 0.6669 (+0.0004) | 0.5446 (+0.0004) | 0.9050 (-0.0049) | 0 | 0 | **FAIL** |

失败项均为：`precision_delta_min`（需 ≥+0.005，实际 +0.0016~+0.0033）、`f1_or_tiou_delta_min`（需 ≥+0.003，实际 -0.0007~+0.0004）。

### Action 分布

| 配置 | left TRIM/KEEP/EXPAND | right TRIM/KEEP/EXPAND |
|---|---|---|
| C1 | 12 / 373 / 8 | 9 / 374 / 10 |
| C2 | 9 / 380 / 4 | 11 / 375 / 7 |
| C3 | 6 / 387 / 0 | 5 / 388 / 0 |

## 6. 失败机制分析

1. **可兑现比例过低**：fallback 率 75-91%（主因：min_candidate_duration 门槛筛掉大量短候选 + snap window 内无 shot boundary + guard 拒绝），实际 snap 仅 10-39/393（3-10%）；
2. **shot boundary 与 weak-reference 高光边界对齐度差**：BHD-0.1 已测出 candidate/reference 边界到最近 shot edge 的距离普遍较大 —— oracle 是在**全部可选 snap 点中按 weak-reference 择优**才拿到 +0.054，deployable 的"最近邻"规则无法复现同样的择优；
3. 方向正确（precision ↑ / coverage ↓ / F1·tIoU 微升）且 **Recall tail 零损伤**（新 R=0 = 0、新 R<0.5 = 0）—— 保守设计目标全部达成，但幅度比 gate 低一个数量级。

## 7. 结论与下一步

**SBA-1 Dev result = NEGATIVE_RESULT / NO_PROMOTION_TO_HARD。**

- Stage 4 temporal 至此完成 **五轮独立负结果**：BR-1（模型输出崩）、SABR-1.1（对称收缩伤 recall）、BR-2（低层证据不动作）、SBC-0（语义分类不可分）、SBA-1（shot snap 不可兑现）；
- BHD-0.1 的上界（local oracle ΔF1 +0.126~+0.254）在**当前可用信号源**（低层视觉 / 现成 4B VLM / shot detection）下均无法兑现；
- 建议：**正式关闭 Stage 4 temporal，mainline 冻结 = Frozen Retrieval v0**（BR-0 / SEL-0 identity）；下一优先级转向 **Stage 5 spatial** 维度或提交策略准备；不重启 temporal 边界方向，除非引入新的信号源（更强的视觉模型 / 微调 / 多模态对齐数据）。

## 8. 隔离性

- Hard：未运行；Heldout392：未访问；Audit36：未使用
- Qwen：0 次；vLLM：0 次；训练：无；oracle 决策：无（deployable 版本不使用 weak-reference 选择边界）
- frozen cache / Stage 3 frozen predictions：未修改；无 video_id/audit_id 特定规则；视频与云端产物未上传 GitHub

## 9. 实现与测试

- 新增：`configs/stage4_sba1_shot_boundary_snap_protocol.json`、`src/.../shot_boundary_snap.py`、`scripts/run_shot_boundary_snap.py`、`tests/test_shot_boundary_snap.py`；validator 增加 SBA-1 分支（BR-0/BR-1/SABR-1.1/BR-2 语义未改）；
- 定向 pytest：**17 passed**；全仓：**194 passed / 1 skipped**；
- 三配置 refinement 全部通过正式 validator；replay/evaluate 复用 Stage 4.4 链路；
- 输出：`/root/autodl-tmp/outputs/stage4_sba1_shot_boundary_snap_dev_20260913_144637/`（未上传）。
