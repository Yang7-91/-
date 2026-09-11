# Stage 4-new / BR-2 — Asymmetric Local Evidence Boundary Refinement (Dev) Report

日期：2026-09-11
结论：**BR-2 Dev result = NEGATIVE_RESULT / NO_PROMOTION_TO_HARD**

## 1. 为什么不继续 SABR-1.1，以及 BR-2 的动机

SABR-1.1 的失败机制是**双向对称收缩**：默认向内裁边会把与 weak-reference 重叠的边缘内容裁掉（recall -0.022~-0.025，precision 仅 +0.003）。BR-2 换一个正交假设：

> 边界错误可能是**单侧**的。对左/右边界分别建立局部 evidence curve，独立判断 TRIM / KEEP / EXPAND；证据不足（persistence 不足或 confidence 低）则 KEEP；全 candidate 级 guard 失败则 fallback identity。

与旧路线的区别：BR-1 让模型输出时间戳（Hard 崩）；SABR-1.1 对称裁边（recall 崩）；DTL-1 无证据外扩（precision 崩）。BR-2 的每一步动作都必须有**连续 K 个局部采样点**的证据支持，且 EXPAND 首次被纳入受控动作空间。

## 2. 数据（Dev-Tune166）

- 视频：**166/166 齐全**（SABR 轮已补下载，来源 `ayushsdev/qvhighlights-videos` 公开 mirror，166/166 sha256 与 frozen dev manifest 精确一致）；
- cache：`cache_build_a`（global hash 校验通过）；role manifest：dev_tune_166（hash `bceb8203…`）；
- 未下载新数据、未触碰 Heldout。

## 3. 预注册配置（协议冻结，事后未改）

| 参数 | C1 | C2 | C3 |
|---|---|---|---|
| boundary_window_sec | 4.0 | 5.0 | 6.0 |
| sample_stride_sec | 0.5 | 0.5 | 0.75 |
| persistence_k | 3 | 3 | 4 |
| action_confidence_min | 0.70 | 0.65 | 0.70 |
| max_trim / max_expand_each_side | 2.0 / 1.0 | 3.0 / 1.5 | 3.0 / 2.0 |
| max_total_boundary_change_ratio | 0.20 | 0.25 | 0.25 |
| min_parent_overlap_ratio | 0.80 | 0.75 | 0.70 |
| min_refined_duration_sec | 5.0 | 4.0 | 5.0 |

## 4. 结果（Dev-Tune166，weak-reference development metrics，非官方指标）

BR-0 baseline：P 0.5736 / R 0.9445 / F1 0.6665 / tIoU 0.5442 / coverage 0.9099；R=0: 1；R<0.5: 9

| 配置 | refine / identity / fallback | P (Δ) | R (Δ) | F1 (Δ) | tIoU (Δ) | cov (Δ) | 新R=0 | 新R<0.5 | Gate |
|---|---|---|---|---|---|---|---|---|---|
| C1 | 9 / 170 / 214 | 0.5736 (-0.0001) | 0.9445 (+0.0000) | 0.6664 (-0.0001) | 0.5442 (-0.0001) | 0.9101 (+0.0002) | 0 | 0 | **FAIL** |
| C2 | 5 / 144 / 244 | 0.5735 (-0.0001) | 0.9445 (+0.0000) | 0.6664 (-0.0001) | 0.5441 (-0.0001) | 0.9101 (+0.0002) | 0 | 0 | **FAIL** |
| C3 | 3 / 100 / 290 | 0.5736 (+0.0000) | 0.9445 (+0.0000) | 0.6665 (+0.0000) | 0.5442 (+0.0000) | 0.9099 (+0.0000) | 0 | 0 | **FAIL** |

Promotion gate（预注册）：recall Δ ≥ -0.01、新 R=0 = 0、新 R<0.5 ≤ 1、precision Δ ≥ +0.005、F1 或 tIoU Δ ≥ +0.003、coverage Δ ≤ +0.005。三配置失败项均为：precision_delta_min、f1_or_tiou_delta_min。

### Action 分布

| 配置 | left TRIM/KEEP/EXPAND | right TRIM/KEEP/EXPAND | fallback 比例 |
|---|---|---|---|
| C1 | 0 / 392 / 1 | 0 / 385 / 8 | 214/393 = 54% |
| C2 | 0 / 393 / 0 | 0 / 388 / 5 | 244/393 = 62% |
| C3 | 0 / 393 / 0 | 0 / 390 / 3 | 290/393 = 74% |

## 5. 失败机制（负结果分析）

1. **保守到几乎不动作**：persistence_k 连续证据 + action_confidence_min 门把 92-99% 的候选锁在 identity/fallback，REFINE 仅 3-9 个（<2%）；
2. **触发的动作全部是 EXPAND（TRIM = 0）**：candidate 外部紧邻内容的信号强度超过内部 core 中位数的场景极少；
3. **少量 EXPAND 的改动量被 margin/cap 压到 ~0.5-1.5s**，对 aggregate 指标的影响在小数第 4 位以后（C3 甚至完全归零）；
4. 结论：**局部帧级证据（边缘/直方图/运动代理与 core 余弦相似度）在候选边界处的判别力不足以支撑可靠的边界移动** —— 这与 SABR-1.1 的失败互为印证：当前无模型特征对 QVHighlights 9x16 clip 的高光边界没有足够的定位能力。

## 6. 隔离性

- Heldout392：未访问；Hard：未运行；Audit36：未用于调参（配置全部预注册，运行后未修改）
- Qwen：0 次；vLLM：0 次；训练：无；oracle 边界选择：无
- frozen cache / Stage 3 frozen predictions：未修改；无 video_id/audit_id 特定规则
- 视频与云端产物未上传 GitHub

## 7. 实现与测试

- 新增 `configs/stage4_br2_asymmetric_boundary_protocol.json`、`src/.../br2_asymmetric_boundary.py`、`scripts/run_br2_asymmetric_boundary.py`、`tests/test_br2_asymmetric_boundary.py`；
- validator 最小扩展：BR-2 分支允许有限 EXPAND（per-side expand/trim cap、change ratio、parent overlap、duration 下限、视频边界），BR-0/BR-1/SABR-1.1 语义未改；
- 定向 pytest：**14 passed**（BR-2）；全仓：**150 passed / 1 skipped**；
- 三配置 refinement 全部通过正式 validator；replay/evaluate 复用 Stage 4.4 链路。

## 8. 结论与下一步

**BR-2 Dev result = NEGATIVE_RESULT / NO_PROMOTION_TO_HARD。**

- BR-2 达成了"不伤 Recall tail"的设计目标（新 R=0 = 0、recall delta ≈ 0），但代价是几乎不动作，无法产生 gate 要求的提升；
- 至此 boundary 方向已有三次系统性负结果：SABR-1.1（对称收缩伤 recall）、BR-2（非对称保守动作无效）、BR-1（模型输出崩溃）—— **在当前无模型特征与 weak-reference 体系下，candidate 边界的进一步精修缺乏可靠信号来源**；
- 建议 temporal mainline 正式冻结于 **Frozen Retrieval v0**（BR-0 / SEL-0 identity），下一轮优先级应转向：①对高光边界的信号上限做一次诊断性分析（纯离线，无调参）；②或转向其他正交维度；③或接受当前 temporal 结果进入后续阶段规划。
