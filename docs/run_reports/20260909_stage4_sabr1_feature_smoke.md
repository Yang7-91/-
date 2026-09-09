# Stage 4-new / SABR-1.0 — Feature Extraction Smoke Report

日期：2026-09-09
结论：**SABR-1.0 feature smoke = PASS**（5 视频 / 9 candidates / 48 bins / 0 errors）

## 1. 为什么不重复 SEL-3 / BR-1 / DTL-1

队友上一轮完整 Stage 4 的三条路线均已证伪或收益不足：

- SEL-3（reason-aware fixed rules）仅极小收益，不足以 promotion；
- BR-1（让 Qwen 直接输出精修时间戳）在 Hard 上造成 Recall tail 崩溃；
- DTL-1 提高 Recall 但显著降低 Precision，F1/tIoU 无有效净收益。

SABR-1 换一条正交路线：**不让模型输出时间戳、不做 score threshold**，而是在 frozen candidate 内部提取低成本确定性视觉显著性信号（动作峰、镜头变化、清晰度、对比度），用信号锚定高光核心区后再做保守边界重构；低置信样本 fallback identity，避免 Recall tail 崩溃。

## 2. 本轮范围

只做 **SABR-1.0 Protocol + Feature Extraction Smoke**：建立协议、特征模块、CLI 与测试，对少量 Dev 样本提取 1s bin 级显著性特征。**不改变任何 candidate 边界，不做任何指标提升声明。**

## 3. 视频获取（云端下载，无需用户上传）

- 用户本地无 Dev v1.1 视频；本轮在 AutoDL 用**仓库已有下载脚本** `scripts/download_baseline_videos.py` + 临时 dev5 manifest 下载（临时 manifest 不入库）；
- 下载来源：公开 HuggingFace mirror `ayushsdev/qvhighlights-videos`（与 Stage 1 Baseline20 同一来源，endpoint `https://hf-mirror.com`）；
- 下载后严格校验：**5/5 size + sha256 与 frozen dev manifest 精确一致**：

| 文件 | 大小 | sha256 前缀 |
|---|---|---|
| 0Yf4z13YlrY_210.0_360.0.mp4 | 4,995,082 | `b3785be8…` |
| 0Yf4z13YlrY_60.0_210.0.mp4 | 5,687,889 | `2a3ad70d…` |
| 0nQgqIJbCZw_360.0_510.0.mp4 | 9,185,670 | `821346dd…` |
| 0nQgqIJbCZw_660.0_810.0.mp4 | 8,488,493 | `4e5ab813…` |
| 2b9txcAt4e0_60.0_210.0.mp4 | 10,398,297 | `adce5408…` |

- 落位：`/root/autodl-tmp/datasets/videos/dev/`（`DEV5_DOWNLOAD_AND_HASH_OK`）

## 4. Smoke 结果

| 项 | 值 |
|---|---|
| cache_dir | `/root/autodl-tmp/outputs/stage4_2_formal_431/FORMAL_431/cache_build_a`（global hash 校验通过） |
| role_manifest | `…/stage4_3_formal_431/FORMAL_431/protocol/role_manifests/dev_tune_166.json`（hash `bceb8203…`） |
| video_dir | `/root/autodl-tmp/datasets/videos/dev`（+ `--video-manifest` 用 Stage 3 frozen dev.jsonl 映射 video_id → 文件） |
| limit / bin_sec | 5 / 1.0s |
| 样本数 | **5**（dev_tune 前 5 个可用样本，skipped_missing = 0） |
| candidate 数 | **9** |
| bin 数 | **48** |
| 错误 | 0 |
| 输出目录 | `/root/autodl-tmp/outputs/stage4_sabr1_feature_smoke_20260909_211732` |

特征质量：6 个信号全部有区分度（`frame_difference` 39/48 bins 非零，其余 5 信号 48/48 非零）；所有值 finite；`saliency_score`（robust z-score + rank 归一后固定权重加权）范围 [-11.38, 10.91]。

## 5. 新增文件

- `configs/stage4_sabr1_protocol.json` —— SABR-1 协议（frozen_upstream / feature_extraction / 固定权重 / safety flags / future gate）
- `src/aic_video_highlight/highlight_retrieval/saliency_anchor.py` —— 特征模块（bin 切分、6 信号、robust 归一化、summary；只用 OpenCV/NumPy/标准库，deterministic）
- `scripts/run_saliency_anchor.py` —— CLI（`extract` 子命令，--video-manifest 支持 video_id → 文件映射）
- `tests/test_saliency_anchor.py` —— 10 个测试（确定性、bin 合法性、序列化、缺失视频报错、NaN/Inf 防护、常数序列归一化、CLI help、协议 safety flags、schema）
- 本报告

## 6. 测试结果

- 定向：`pytest tests/test_saliency_anchor.py` → **10 passed**（本地 Windows 与云端均跑过）
- 全仓：**121 passed / 1 skipped**（无既有失败；1 skip 为既有用例）

## 7. 隔离性

- Heldout392 访问：**否**
- Qwen 调用：**0**；vLLM 调用：**0**；模型训练：**无**
- frozen candidate cache：未修改（只读）；Stage 3 frozen predictions：未修改
- 未做调参（saliency 权重固定于协议，未参考任何 weak-reference/audit 信号）
- 无 video_id / audit_id 特定规则
- 视频未上传 GitHub；云端产物不入库

## 8. 下一步

SABR-1.1 — Conservative Boundary Proposal：

- 只在 Dev-Tune166 上跑；
- 先启用 identity fallback（低置信 → 原边界）；
- 用本模块 bin 级信号在 candidate 内部提出保守边界候选；
- 按 SABR-1 协议 future gate 评估（recall drop ≥ -0.01、无新零 recall、precision Δ ≥ +0.01、F1/tIoU Δ ≥ +0.005）；
- 不直接跑 Hard；不使用 Audit36 调参；不调用 Qwen/vLLM。
