# Stage 4.4.1 — Artifact Missing Report: cache_build_a 不可用

日期：2026-09-08
阶段：Stage 4.4.1 — BR-0 Real Cache Identity Replay
结论：**本轮无法执行 BR-0 真实 cache identity replay；未使用任何 synthetic 替代。**

## 1. 执行摘要

Stage 4.4.0 的代码链路（BR-0 identity / validate-refinement / replay / evaluate / assess / CLI / tests）已在仓库中就绪并通过 26 个定向测试。但 Stage 4.4.1 要求在**真实** Stage 4.2 frozen cache（`cache_build_a`）上运行 identity replay，而当前可访问的全部环境中均不存在该产物，亦不存在可用于离线复现 cache 的 Stage 3 frozen Dev/Hard 输出。

因此本轮按协议停止，不运行任何替代实验，不重跑 Qwen，不访问 Heldout。

## 2. 已检查的环境与路径

### 2.1 Windows 本地（D:\CDUT\AIC）

- 完整项目代码仓库（main @ cc7444a），无 outputs 数据（`outputs/` 仅有 `.gitkeep`）；
- 本地无 `FORMAL_431/`、无 `cache_build_a`、无 Stage 3 frozen outputs。

### 2.2 AutoDL 实例（connect.cqa1.seetacloud.com:21821）

`/root/autodl-tmp` 下仅有：

```
AIC-VideoHighlight/            （代码仓库，outputs/ 为空）
aic_video_highlight_outputs/   （仅 Stage 1 产物，见下）
datasets/  hf-cache/  models/  TempSamp-R1-main*
```

`aic_video_highlight_outputs/` 现有内容：

- `stage1_baseline20_20260907_194441/`（Stage 1 Baseline20 粗召回）
- `stage1_highlight_retrieval_smoke_20260907/`（Stage 1 smoke）
- `logs/`、`.last_baseline20_run`

### 2.3 精准路径检查（全部缺失）

以下 6 个候选路径均不存在：

- `/root/autodl-tmp/outputs/stage4_2_formal_431/FORMAL_431/cache_build_a`
- `/root/autodl-tmp/AIC-VideoHighlight-run/outputs/stage4_2_formal_431/FORMAL_431/cache_build_a`
- `/root/autodl-tmp/AIC-VideoHighlight/outputs/stage4_2_formal_431/FORMAL_431/cache_build_a`
- `/root/autodl-tmp/outputs/FORMAL_431/cache_build_a`
- `/root/autodl-tmp/AIC-VideoHighlight-run/FORMAL_431/cache_build_a`
- `/root/autodl-tmp/AIC-VideoHighlight/FORMAL_431/cache_build_a`

### 2.4 限制深度搜索（0 命中）

- `find /root/autodl-tmp`（排除 .venv / hf-cache / models / datasets / data）：**无任何 `cache_manifest.json`**
- `find /root -maxdepth 4`（排除 .venv）：**无任何 `cache_manifest.json`**

### 2.5 Stage 3 frozen Dev/Hard source（全部缺失）

以下 6 个路径均不存在：

- `/root/autodl-tmp/AIC-VideoHighlight-run/outputs/stage3_dev_full_baseline_v1`
- `/root/autodl-tmp/AIC-VideoHighlight-run/outputs/stage3_hard_full_baseline_v1`
- `/root/autodl-tmp/AIC-VideoHighlight/outputs/stage3_dev_full_baseline_v1`
- `/root/autodl-tmp/AIC-VideoHighlight/outputs/stage3_hard_full_baseline_v1`
- `/root/autodl-tmp/outputs/stage3_dev_full_baseline_v1`
- `/root/autodl-tmp/outputs/stage3_hard_full_baseline_v1`

## 3. 缺失产物清单

| 产物 | 说明 | 状态 |
|---|---|---|
| `cache_build_a` 目录 | Stage 4.2 frozen candidate cache（431 records / raw 1293 / merged 1277 / `global_semantic_sha256 = 4b515a6d6fb47073413c686214c3fa5f97293655a3824241eb3305b9e7753246`） | **缺失** |
| `stage3_dev_full_baseline_v1` | Dev183 frozen predictions/raw/run_config（cache 复现所需 source） | **缺失** |
| `stage3_hard_full_baseline_v1` | Hard248 frozen predictions/raw/run_config（cache 复现所需 source） | **缺失** |

## 4. 为什么不能在本机恢复

1. `candidate_cache` / `boundary_refinement` 模块对 cache 做 `EXPECTED_CACHE_GLOBAL_HASH`（`4b515a6d…7753246`）硬校验，synthetic 或部分数据无法通过校验；
2. Stage 4.2 cache 虽然设计上可离线复现（identity replay、Qwen/vLLM 调用 = 0），但复现输入是 Stage 3 frozen Dev/Hard outputs —— 它们同样缺失；
3. 重跑 Stage 3 需要调用 Qwen/vLLM 与 aic_highlight_dev_v1.1 数据，本轮协议明确禁止。

## 5. 需要用户提供的恢复路径（二选一）

### 路径 A（推荐，最快）：直接恢复 `cache_build_a` 目录

从以下任一来源找回 Stage 4.2 正式实验的 `FORMAL_431/cache_build_a` 整个目录（含 `cache_manifest.json` 与各 role 记录）：

- 旧 AutoDL 实例（若 Stage 4.2/4.3 曾在其他实例运行，检查其 `/root/autodl-tmp` 下的 outputs）；
- AutoDL 网盘 / 云备份；
- 本地硬盘或其他备份介质。

恢复后放到本实例，例如：

```
/root/autodl-tmp/outputs/stage4_2_formal_431/FORMAL_431/cache_build_a
```

随后的验收（本轮已预备好）：`cache_manifest.json` 中 `record_count=431`、raw=1293、merged=1277、`global_semantic_sha256=4b515a6d…7753246`。

### 路径 B：恢复 Stage 3 frozen Dev/Hard outputs 后离线重建 cache

提供 `stage3_dev_full_baseline_v1` 与 `stage3_hard_full_baseline_v1` 两个目录（frozen predictions / raw / run_config）。然后用仓库已有 Stage 4.2 exporter（`scripts/run_candidate_cache.py`）离线重建 cache（零模型调用），并要求重建结果的 global hash 与预期一致；不一致则停止。

## 6. 隔离性声明

- Heldout392：未访问；
- Qwen 调用：0；vLLM 调用：0；
- frozen cache / Stage 3 frozen predictions：未修改（目标产物本身缺失，无从修改）；
- 本轮未运行任何 synthetic/替代实验冒充正式结果。

## 7. 下一步

1. 用户按路径 A 或 B 提供产物；
2. 恢复并验收 hash 后，下一轮执行 Stage 4.4.1：`refine (BR-0, dev/hard) → validate-refinement → replay → evaluate → assess`，输出 identity replay 报告；
3. 在 Stage 4.4.1 通过之前，不进入 Stage 4.4.2（BR-1 简单边界规则）。
