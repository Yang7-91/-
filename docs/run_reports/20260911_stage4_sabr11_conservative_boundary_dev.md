# Stage 4-new / SABR-1.1 — Conservative Boundary Proposal (Dev) Report

日期：2026-09-11
结论：**SABR-1.1 Dev result = NEGATIVE_RESULT / NO_PROMOTION_TO_HARD**

## 1. 方法动机

SABR-1（Shot-aware Saliency Anchored Boundary Refinement）不走旧路线：不让 Qwen 直接输出时间戳（BR-1 已崩）、不做 score threshold selection（SEL 系已证伪）、不做外扩式增补（DTL-1 已证伪）。SABR-1.1 在 frozen candidate **内部**用 1s/2s bin 视觉显著性信号找高显著核心区，只裁掉头部/尾部的连续低显著 run，并施加多重保守保护；任何 guard 失败即 fallback identity。

## 2. 数据准备（Dev-Tune166）

- 166 个 Dev-Tune 视频：云端已有 5 个，其余 161 个用仓库脚本 `download_baseline_videos.py` 从公开 mirror `ayushsdev/qvhighlights-videos`（hf-mirror endpoint）下载；
- **166/166 size + sha256 与 frozen dev manifest 精确一致**（`DEV166_ALL_HASH_OK`）；
- 落位 `/root/autodl-tmp/datasets/videos/dev/`；video_id → 文件映射用 Stage 3 frozen dev.jsonl；
- 磁盘充足（50G 盘剩 30G）；Heldout 未触碰。

## 3. 预注册配置（协议冻结，事后未改）

| 参数 | C1（保守） | C2（中等） | C3（2s bin） |
|---|---|---|---|
| bin_sec | 1.0 | 1.0 | 2.0 |
| core_quantile | 0.80 | 0.75 | 0.75 |
| margin_sec | 1.0 | 1.0 | 2.0 |
| max_trim_each_side_sec | 3.0 | 4.0 | 4.0 |
| max_total_shrink_ratio | 0.25 | 0.30 | 0.30 |
| min_parent_overlap_ratio | 0.75 | 0.70 | 0.70 |
| min_refined_duration_sec | 5.0 | 4.0 | 6.0 |
| min_peak_prominence | 0.15 | 0.12 | 0.12 |

## 4. 结果（Dev-Tune166，weak-reference development metrics，非官方指标）

BR-0 baseline：P 0.5736 / R 0.9445 / F1 0.6665 / tIoU 0.5442 / coverage 0.9099；R=0: 1；R<0.5: 9

| 配置 | trimmed / fallback / identity | P (Δ) | R (Δ) | F1 (Δ) | tIoU (Δ) | coverage (Δ) | 新R=0 | 新R<0.5 | Gate |
|---|---|---|---|---|---|---|---|---|---|
| C1 | 31 / 354 / 8 | 0.5753 (+0.0017) | 0.9226 (**-0.0219**) | 0.6592 (-0.0073) | 0.5336 (-0.0106) | 0.8871 (-0.0228) | 0 | 0 | **FAIL** |
| C2 | 35 / 346 / 12 | 0.5771 (+0.0034) | 0.9200 (**-0.0245**) | 0.6590 (-0.0075) | 0.5332 (-0.0110) | 0.8835 (-0.0264) | 0 | 1 | **FAIL** |
| C3 | 25 / 357 / 11 | 0.5730 (-0.0007) | 0.9228 (**-0.0217**) | 0.6585 (-0.0080) | 0.5325 (-0.0117) | 0.8880 (-0.0219) | 0 | 1 | **FAIL** |

Promotion gate（预注册）：recall Δ ≥ -0.01、新 R=0 = 0、新 R<0.5 ≤ 1、precision Δ ≥ +0.01、F1 或 tIoU Δ ≥ +0.005、coverage Δ ≤ -0.01。三个配置的失败项均含：recall_delta_min、precision_delta_min、f1_or_tiou_delta_min。

## 5. 失败机制

1. **Recall 损失集中且不成比例**：fallback 率高达 90%（354-357/393），实际仅裁 25-35 个候选（8-9%），但 mean recall 就掉 2.2-2.5% —— 说明被裁掉的"低显著边缘"里包含大量真实高光内容；
2. **1s bin 显著性信号与 weak-reference 高光边界相关性弱**：precision 收益（+0.003 量级）远低于 recall 代价，F1/tIoU 净收益为负；
3. C3 的 2s bin + 更大 margin 没有改善（更平滑的信号反而丢失边界定位精度）。

## 6. 隔离性

- Heldout392：未访问；Hard：未运行；Audit36：未用于调参（配置全部预注册于协议，运行后未修改）
- Qwen：0 次；vLLM：0 次；训练：无
- frozen cache / Stage 3 frozen predictions：未修改；无 video_id/audit_id 特定规则
- 视频与云端产物未上传 GitHub

## 7. 实现与测试

- 新增 `src/.../sabr_boundary.py`（propose_conservative_boundary / build_sabr11_result）、CLI `propose` 子命令、协议 `boundary_proposal` 段（C1/C2/C3 + gate）、validator 的 SABR 最小兼容分支（不外扩 + 收缩帽校验，未改 BR-0/BR-1 语义）；
- 过程修复：CLI import 缺失、result semantic hash 补齐、canonical JSON 写入（commit `7e52bd5` / `7c134bf` / `ae7531d`）；
- 定向 pytest：**22 passed**（saliency 10 + boundary 12）；全仓：**133 passed / 1 skipped**；
- 每个配置的 refinement 均通过正式 validator（schema/hash/不外扩/收缩帽），replay/evaluate 复用 Stage 4.4 链路。

## 8. 结论与下一步

**SABR-1.1 Dev result = NEGATIVE_RESULT / NO_PROMOTION_TO_HARD。**

- 三个预注册配置均未通过 gate，按协议冻结负结果，不调参重试、不生成"新版本协议"变相重跑；
- 当前证据不支持继续在 1-2s bin 显著性方向投入：信号粒度与 weak-reference 高光边界的对齐不足是根本机制；
- temporal mainline 维持 **Frozen Retrieval v0**（BR-0 / SEL-0 identity）；
- 下一步建议（供下一轮讨论，非本轮执行）：
  1. 若继续 boundary 方向，需先量化 bin 信号与高光边界的对齐度（诊断性分析，非调参）；
  2. 或转向完全不同的正交信号（如 shot boundary 检测做边界对齐，而非 saliency 收缩）；
  3. 或接受 retrieval v0 边界，把精力转向下游 spatial/其他维度。
