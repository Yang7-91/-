# Stage 4.4.1 — Recovery Progress Report（BR-0 未执行，等待 role manifests）

日期：2026-09-08
阶段：Stage 4.4.1 — BR-0 Real Cache Identity Replay
状态：**RECOVERY PARTIAL — 唯一缺口 role manifests，等待用户补充后继续**

## 1. 结论

用户提供的恢复包已定位并完成大部分恢复工作。`cache_build_a` 已上传 AutoDL 并通过项目正式 validator（全部数字与 Stage 4.2 冻结基线精确一致）。**BR-0 正式 replay 因缺少 role manifests 而未执行**；未使用任何 synthetic 替代。

## 2. 恢复产物识别

- 本地恢复目录：`C:\Users\p0220\Desktop\AIC_Stage4_4_1_Recovery_Artifacts_20260908\AIC_Stage4_4_1_Recovery_Artifacts\`
- **类型 A**（直接包含 cache_build_a）+ 附带 Stage 3 frozen Dev/Hard outputs（Path B 备用源）
- 附 `RECOVERY_MANIFEST.md`：声明 432/432 文件 SHA-256 拷贝一致、Qwen/vLLM/GPU 调用 = 0、无 Heldout/视频/模型

## 3. 已完成的恢复与校验（云端 AutoDL）

| 步骤 | 结果 |
|---|---|
| 云端代码同步 | `/root/autodl-tmp/AIC-VideoHighlight` main = `eadaa16`（含全部 Stage 4.4 文件），git clean |
| cache 上传 | 本地 tar 打包（358KB，规避 PowerShell zip 反斜杠路径问题）→ scp → 解压 |
| cache 落位 | `/root/autodl-tmp/outputs/stage4_2_formal_431/FORMAL_431/cache_build_a`，432 files / 2.6M，CACHE_LAYOUT_OK |
| manifest 校验 | record_count=431，global_semantic_sha256=`4b515a6d…7753246`（与预期精确一致），complete_source_export=true |
| **项目 validator** | **PASS**：`run_candidate_cache.py validate` 输出 `record_count=431, raw=1293, merged=1277, split={dev:183, hard:248}, global hash=4b515a6d…`，exit 0 |
| Stage 3 frozen outputs 上传 | `stage3_dev_full_baseline_v1`（183 predictions）+ `stage3_hard_full_baseline_v1`（248 predictions）tar 包已上传 `/root/autodl-tmp/recovery_artifacts/`，待解压使用 |
| CLI 参数探明 | refine/validate-refinement/replay/evaluate/assess 真实参数（--cache-dir/--role-manifest/--protocol/--refiner 等），与任务模板不同处已记录 |

## 4. 唯一缺口（阻塞项）

**role manifests 目录**（3 个 JSON，必须原样、同一目录）：

| 文件 | 冻结 hash（protocol 校验目标） |
|---|---|
| `dev_tune.json`（166 records） | `bceb82034bcc5eed42a5099d7f54dd0fa776fe240969dfada231d255a368ec71` |
| `hard_stress.json`（229 records） | `81d734ee28ec38c096b908a573249c62713d358fb03ea135e5b55e1ae4405687` |
| `audit_diagnostic.json`（36 records，本轮不运行但目录校验必需） | `98fd987b7b56bdc18dd58a0c809e77adb28b93e322fd08a5246c04e115e1780c` |
| 目录 summary hash | `6eccf82509295980cb16270a306fad251941b6a09f9f0178335eeabd811b23e0` |

- 预期来源：`D:/CDUT/硕士/2026-09 AIC/实验记录/Stage4_候选筛选与时间边界精修/02_Frozen_Candidate_Cache_Identity_Replay/FORMAL_431/` 内与 `cache_build_a` 同级的 role manifests；
- 无法重建替代：`build-roles` 需要 Audit36 membership 源文件（同样缺失），且协议 hash 校验要求逐字节一致；
- 已搜索范围：恢复包、本地 `D:\CDUT`、`D:\` 中文目录、OneDrive —— 均未找到。

## 5. 隔离性声明

- Heldout392：未访问；Qwen 调用 = 0；vLLM 调用 = 0；GPU 实验 = 0
- frozen cache 未修改（validator 为只读校验）；未做 BR-1；未做任何调参
- 未运行 synthetic/替代实验

## 6. 用户补充文件后的执行计划（已就绪，无需再探路）

1. 上传 role manifests → `/root/autodl-tmp/outputs/stage4_2_formal_431/FORMAL_431/role_manifests/`
2. `validate-roles`（目录级校验 + hash 匹配）
3. 解压 Stage 3 frozen outputs
4. Dev/Hard × {BR-0 refine → validate-refinement → replay → evaluate}（frozen-predictions 用 Stage 3 predictions.jsonl）
5. 生成 SEL-0 baseline evaluation（dev/hard，供 assess 对照）
6. assess（dev_recall_guardrail / dev_promotion_gate / hard_stress_gate）
7. 输出 `docs/run_reports/20260908_stage4_4_1_br0_identity_replay.md` 并 push
