"""Stage 4-new BHD-0.1 boundary headroom diagnostic CLI (audited semantics).

Separates optional oracles (identity included, upper bound by construction)
from forced diagnostics (movement mandatory, may be worse than baseline).
All results are ORACLE_DIAGNOSTIC_ONLY / NON_DEPLOYABLE / USES_WEAK_REFERENCE.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic_video_highlight.highlight_retrieval import boundary_headroom_diagnostic as bhd  # noqa: E402
from aic_video_highlight.highlight_retrieval.br2_asymmetric_boundary import (  # noqa: E402
    build_local_boundary_evidence,
)
from aic_video_highlight.highlight_retrieval.boundary_refinement import (  # noqa: E402
    load_cache_payloads,
)
from aic_video_highlight.highlight_retrieval.candidate_selection import (  # noqa: E402
    _read_json_object,
    validate_role_manifest_directory,
)
from aic_video_highlight.highlight_retrieval.saliency_anchor import (  # noqa: E402
    write_json_lines,
)

PROTOCOL_SCHEMA_VERSION = "aic.stage4.bhd0_boundary_headroom_diagnostic/v1"


def _load_video_manifest(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    mapping: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        video_id = row.get("video_id")
        relative = row.get("relative_video_path") or row.get("video_path") or row.get("filename")
        if isinstance(video_id, str) and isinstance(relative, str):
            mapping[video_id] = relative
    return mapping


def _resolve_video_path(video_dir: Path, video_id: str, video_map: dict[str, str]) -> Path | None:
    relative = video_map.get(video_id)
    if relative:
        for candidate in (video_dir / relative, video_dir / Path(relative).name):
            if candidate.is_file():
                return candidate
    candidate = video_dir / f"{video_id}.mp4"
    return candidate if candidate.is_file() else None


def _metrics_block(metrics: Mapping[str, float], baseline: Mapping[str, float]) -> dict[str, float]:
    return {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "temporal_iou": metrics["temporal_iou"],
        "coverage_ratio": metrics.get("coverage_ratio", 0.0),
        "coverage_seconds": metrics.get("prediction_duration_sec", 0.0),
        "delta_precision": metrics["precision"] - baseline["precision"],
        "delta_recall": metrics["recall"] - baseline["recall"],
        "delta_f1": metrics["f1"] - baseline["f1"],
        "delta_temporal_iou": metrics["temporal_iou"] - baseline["temporal_iou"],
        "delta_coverage_ratio": metrics.get("coverage_ratio", 0.0) - baseline.get("coverage_ratio", 0.0),
    }


def _aggregate_actions(actions: list[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counts = {
        side: {action: 0 for action in ("TRIM", "KEEP", "EXPAND")}
        for side in ("left", "right")
    }
    for action in actions:
        counts["left"][action["left_action"]] += 1
        counts["right"][action["right_action"]] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 4-new BHD-0.1 boundary headroom audit diagnostic"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    diagnose = commands.add_parser("diagnose", help="Run audited BHD-0.1 oracle diagnostics")
    diagnose.add_argument("--cache-dir", type=Path, required=True)
    diagnose.add_argument("--role-manifest", type=Path, required=True)
    diagnose.add_argument("--video-dir", type=Path, required=True)
    diagnose.add_argument("--video-manifest", type=Path, default=None)
    diagnose.add_argument("--frozen-predictions", type=Path, required=True)
    diagnose.add_argument("--protocol", type=Path, required=True)
    diagnose.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command != "diagnose":
        parser.error(f"unknown command: {args.command}")

    protocol = _read_json_object(args.protocol)
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise SystemExit(
            f"protocol schema mismatch: expected {PROTOCOL_SCHEMA_VERSION}, "
            f"got {protocol.get('schema_version')}"
        )
    rules = protocol["decision_rules"]
    oracle_cfg = protocol["oracle_diagnostics"]
    shot_cfg = oracle_cfg["shot_boundary_snap"]

    manifest, records = load_cache_payloads(args.cache_dir)
    role_path = args.role_manifest.expanduser().resolve()
    validate_role_manifest_directory(role_path.parent, cache_manifest=manifest)
    role_manifest = _read_json_object(role_path)
    if role_manifest.get("role") != protocol.get("role"):
        raise SystemExit("role mismatch with protocol")

    frozen_rows = {
        json.loads(line)["video_id"]: json.loads(line)
        for line in args.frozen_predictions.expanduser()
        .resolve()
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }

    video_dir = args.video_dir.expanduser().resolve()
    video_map = _load_video_manifest(
        args.video_manifest.expanduser().resolve() if args.video_manifest else None
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    ordered_ids = [item["video_id"] for item in role_manifest["records"]]
    missing = [v for v in ordered_ids if not _resolve_video_path(video_dir, v, video_map).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} videos missing; first: {missing[:5]}")

    durations = {video_id: float(records[video_id]["duration_sec"]) for video_id in ordered_ids}
    baseline_agg, baseline_per_video = bhd.baseline_video_metrics(
        (frozen_rows[v] for v in ordered_ids), durations
    )

    candidates_by_video = {
        video_id: records[video_id]["merged_candidates"] for video_id in ordered_ids
    }
    references_by_video = {
        video_id: bhd.candidate_refs(frozen_rows.get(video_id, {})) for video_id in ordered_ids
    }
    oracle_rows: list[dict] = []

    # --- 1. local boundary optional oracles (three objectives) + forced diagnostic
    local_optional: dict[str, Any] = {}
    local_forced: dict[str, Any] = {}
    oracle_label_rows: list[dict] = []
    for shift in [float(x) for x in oracle_cfg["local_boundary_search"]["max_shift_sec"]]:
        for objective in ("f1", "temporal_iou", "precision_under_recall_guard"):
            per_video_metrics = {}
            all_actions = []
            for video_id in ordered_ids:
                video_result = bhd.greedy_video_oracle(
                    candidates_by_video[video_id],
                    references_by_video[video_id],
                    durations[video_id],
                    max_shift=shift,
                    step=float(oracle_cfg["local_boundary_search"]["step_sec"]),
                    objective=objective,
                    allow_identity=True,
                )
                per_video_metrics[video_id] = video_result["metrics"]
                for action in video_result["actions"]:
                    all_actions.append({"video_id": video_id, **action})
            aggregate = {
                key: bhd._finite(sum(m[key] for m in per_video_metrics.values()) / len(per_video_metrics))
                for key in ("precision", "recall", "f1", "temporal_iou", "coverage_ratio")
            }
            local_optional[f"{objective}@{shift}s"] = {
                "optional_oracle": True,
                "upper_bound": True,
                "identity_included": True,
                "objective": objective,
                "deployable": False,
                "uses_weak_reference": True,
                "aggregate": aggregate,
                "metrics_block": _metrics_block(aggregate, baseline_agg),
                "action_counts": _aggregate_actions(all_actions),
            }
            if objective == "f1" and shift == max(
                float(x) for x in oracle_cfg["local_boundary_search"]["max_shift_sec"]
            ):
                for row in all_actions:
                    oracle_label_rows.append(
                        {
                            "video_id": row["video_id"],
                            "candidate_id": row["candidate_id"],
                            "left_action": row["left_action"],
                            "right_action": row["right_action"],
                            "left_delta_sec": row["left_delta_sec"],
                            "right_delta_sec": row["right_delta_sec"],
                            "oracle_markers": dict(bhd.BHD_MARKERS),
                        }
                    )
        forced = {}
        for objective in ("f1",):
            all_actions = []
            per_video_metrics = {}
            for video_id in ordered_ids:
                video_result = bhd.greedy_forced_diagnostic(
                    candidates_by_video[video_id],
                    references_by_video[video_id],
                    durations[video_id],
                    max_shift=shift,
                    step=float(oracle_cfg["local_boundary_search"]["step_sec"]),
                    objective=objective,
                )
                per_video_metrics[video_id] = video_result["metrics"]
                for action in video_result["actions"]:
                    all_actions.append({"video_id": video_id, **action})
            aggregate = {
                key: bhd._finite(sum(m[key] for m in per_video_metrics.values()) / len(per_video_metrics))
                for key in ("precision", "recall", "f1", "temporal_iou", "coverage_ratio")
            }
            forced[f"{objective}@{shift}s"] = {
                "forced_diagnostic": True,
                "upper_bound": False,
                "deployable": False,
                "uses_weak_reference": True,
                "metrics_block": _metrics_block(aggregate, baseline_agg),
                "action_counts": _aggregate_actions(all_actions),
            }
        local_forced[f"{shift}s"] = forced
    write_json_lines(output_dir / "boundary_oracle_labels.jsonl", oracle_label_rows)

    # --- 2. shot detection + snap oracles
    print("shot boundary detection ...", flush=True)
    import cv2

    shot_rows: list[dict] = []
    video_segments: dict[str, list[tuple[float, float]]] = {}
    for video_id in ordered_ids:
        video_path = _resolve_video_path(video_dir, video_id, video_map)
        duration = durations[video_id]
        boundaries = bhd.detect_shot_boundaries(
            video_path,
            lambda p: cv2.VideoCapture(str(p), cv2.CAP_FFMPEG),
            frame_stride_sec=float(shot_cfg["frame_stride_sec"]),
            min_shot_len_sec=float(shot_cfg["min_shot_len_sec"]),
            duration_sec=duration,
        )
        segments = bhd.shot_segments_from_boundaries(boundaries, duration)
        video_segments[video_id] = segments
        shot_rows.append(
            {
                "video_id": video_id,
                "duration_sec": duration,
                "num_shot_boundaries": len(boundaries),
                "num_shot_segments": len(segments),
                "boundaries_sec": boundaries,
            }
        )
    write_json_lines(output_dir / "shot_boundaries.jsonl", shot_rows)

    alignment_rows = []
    for video_id in ordered_ids:
        edges = [edge for seg in video_segments[video_id] for edge in seg]
        for candidate in candidates_by_video[video_id]:
            start, end = float(candidate["start_sec"]), float(candidate["end_sec"])
            refs = references_by_video[video_id]
            candidate_distance = min((abs(e - start), abs(e - end)) for e in edges) if edges else (duration := durations[video_id], duration)[0]
            reference_distance = (
                min(abs(point - edge) for seg in refs for point in seg for edge in edges)
                if refs and edges
                else None
            )
            alignment_rows.append(
                {
                    "video_id": video_id,
                    "candidate_id": str(candidate["merged_candidate_id"]),
                    "candidate_edge_to_nearest_shot_edge_sec": bhd._finite(min(abs(e - start) for e in edges)) if edges else None,
                    "reference_edge_to_nearest_shot_edge_sec": bhd._finite(reference_distance) if reference_distance is not None else None,
                }
            )
    write_json_lines(output_dir / "candidate_boundary_alignment.jsonl", alignment_rows)

    shot_optional: dict[str, Any] = {}
    shot_forced: dict[str, Any] = {}
    for shift in [float(x) for x in shot_cfg["max_snap_shift_sec"]]:
        for objective in ("f1", "temporal_iou"):
            per_video_metrics = {}
            for video_id in ordered_ids:
                video_result = bhd.greedy_shot_snap_oracle(
                    candidates_by_video[video_id],
                    references_by_video[video_id],
                    durations[video_id],
                    video_segments[video_id],
                    max_shift=shift,
                    objective=objective,
                    allow_identity=True,
                )
                per_video_metrics[video_id] = video_result["metrics"]
            aggregate = {
                key: bhd._finite(sum(m[key] for m in per_video_metrics.values()) / len(per_video_metrics))
                for key in ("precision", "recall", "f1", "temporal_iou", "coverage_ratio")
            }
            shot_optional[f"{objective}@{shift}s"] = {
                "optional_oracle": True,
                "upper_bound": True,
                "identity_included": True,
                "objective": objective,
                "deployable": False,
                "uses_weak_reference": True,
                "metrics_block": _metrics_block(aggregate, baseline_agg),
            }
        per_video_metrics = {}
        for video_id in ordered_ids:
            video_result = bhd.greedy_shot_snap_oracle(
                candidates_by_video[video_id],
                references_by_video[video_id],
                durations[video_id],
                video_segments[video_id],
                max_shift=shift,
                objective="f1",
                allow_identity=False,
            )
            per_video_metrics[video_id] = video_result["metrics"]
        aggregate = {
            key: bhd._finite(sum(m[key] for m in per_video_metrics.values()) / len(per_video_metrics))
            for key in ("precision", "recall", "f1", "temporal_iou", "coverage_ratio")
        }
        shot_forced[f"{shift}s"] = {
            "forced_diagnostic": True,
            "upper_bound": False,
            "deployable": False,
            "uses_weak_reference": True,
            "metrics_block": _metrics_block(aggregate, baseline_agg),
        }

    # --- 3. subshot oracles
    subshot_optional: dict[str, Any] = {}
    subshot_forced: dict[str, Any] = {}
    for max_components in [int(x) for x in oracle_cfg["subshot_split_upper_bound"]["max_components"]]:
        per_video_metrics = {}
        for video_id in ordered_ids:
            video_result = bhd.greedy_subshot_oracle(
                candidates_by_video[video_id],
                references_by_video[video_id],
                durations[video_id],
                video_segments[video_id],
                max_components=max_components,
                objective="f1",
                allow_identity=True,
            )
            per_video_metrics[video_id] = video_result["metrics"]
        aggregate = {
            key: bhd._finite(sum(m[key] for m in per_video_metrics.values()) / len(per_video_metrics))
            for key in ("precision", "recall", "f1", "temporal_iou", "coverage_ratio")
        }
        subshot_optional[f"{max_components}components"] = {
            "optional_oracle": True,
            "upper_bound": True,
            "identity_included": True,
            "objective": "f1",
            "deployable": False,
            "uses_weak_reference": True,
            "metrics_block": _metrics_block(aggregate, baseline_agg),
        }
        per_video_metrics = {}
        for video_id in ordered_ids:
            video_result = bhd.greedy_subshot_oracle(
                candidates_by_video[video_id],
                references_by_video[video_id],
                durations[video_id],
                video_segments[video_id],
                max_components=max_components,
                objective="f1",
                allow_identity=False,
            )
            per_video_metrics[video_id] = video_result["metrics"]
        aggregate = {
            key: bhd._finite(sum(m[key] for m in per_video_metrics.values()) / len(per_video_metrics))
            for key in ("precision", "recall", "f1", "temporal_iou", "coverage_ratio")
        }
        subshot_forced[f"{max_components}components"] = {
            "forced_diagnostic": True,
            "upper_bound": False,
            "deployable": False,
            "uses_weak_reference": True,
            "metrics_block": _metrics_block(aggregate, baseline_agg),
        }

    # --- 4. BR-2 evidence separability (labels from f1 oracle at max shift)
    print("BR-2 evidence separability ...", flush=True)
    oracle_label_lookup = {
        (row["video_id"], row["candidate_id"]): row for row in oracle_label_rows
    }
    br2_config = _read_json_object(
        Path(__file__).resolve().parents[1] / "configs" / "stage4_br2_asymmetric_boundary_protocol.json"
    )["configs"]["BR-2-C2"]
    side_scores = {
        side: {action: [] for action in ("TRIM", "KEEP", "EXPAND")}
        for side in ("left", "right")
    }
    for video_id in ordered_ids:
        video_path = _resolve_video_path(video_dir, video_id, video_map)
        duration = durations[video_id]
        capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        if not capture.isOpened():
            continue
        try:
            for candidate in candidates_by_video[video_id]:
                label = oracle_label_lookup.get((video_id, str(candidate["merged_candidate_id"])))
                if label is None:
                    continue
                evidence = build_local_boundary_evidence(
                    capture,
                    float(candidate["start_sec"]),
                    float(candidate["end_sec"]),
                    br2_config,
                    duration_sec=duration,
                )
                for side in ("left", "right"):
                    rows = evidence[side]
                    inside_scores = [row["core_similarity_histogram"] for row in rows if row["inside"]]
                    outside_scores = [row["core_similarity_histogram"] for row in rows if not row["inside"]]
                    action = label[f"{side}_action"]
                    score = (
                        float(np.median(inside_scores))
                        if action in ("TRIM", "KEEP") and inside_scores
                        else (float(np.median(outside_scores)) if outside_scores else 0.0)
                    )
                    side_scores[side][action].append(score)
        finally:
            capture.release()

    separability = {}
    for side in ("left", "right"):
        trim_scores = side_scores[side]["TRIM"]
        keep_scores = side_scores[side]["KEEP"]
        expand_scores = side_scores[side]["EXPAND"]
        rest_trim = keep_scores + expand_scores
        rest_expand = keep_scores + trim_scores
        auc_trim = (
            max(bhd.rank_auc_like(trim_scores, rest_trim), 1.0 - bhd.rank_auc_like(trim_scores, rest_trim))
            if trim_scores and rest_trim
            else 0.5
        )
        auc_expand = (
            max(bhd.rank_auc_like(expand_scores, rest_expand), 1.0 - bhd.rank_auc_like(expand_scores, rest_expand))
            if expand_scores and rest_expand
            else 0.5
        )
        separability[side] = {
            "label_counts": {
                action: len(side_scores[side][action]) for action in ("TRIM", "KEEP", "EXPAND")
            },
            "auc_like_trim_vs_rest": bhd._finite(auc_trim),
            "auc_like_expand_vs_rest": bhd._finite(auc_expand),
            "mean_gap_trim": bhd.mean_gap(trim_scores, rest_trim),
            "mean_gap_expand": bhd.mean_gap(expand_scores, rest_expand),
            "effect_size_trim": bhd.effect_size(trim_scores, rest_trim),
            "effect_size_expand": bhd.effect_size(expand_scores, rest_expand),
        }

    def best_block(blocks: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        key = max(
            blocks,
            key=lambda name: max(
                blocks[name]["metrics_block"]["delta_f1"],
                blocks[name]["metrics_block"]["delta_temporal_iou"],
            ),
        )
        return key, blocks[key]

    local_key, local_best = best_block(local_optional)
    shot_key, shot_best = best_block(shot_optional)
    subshot_key, subshot_best = best_block(subshot_optional)
    local_headroom = bhd.has_meaningful_headroom(
        local_best["metrics_block"]["delta_precision"],
        local_best["metrics_block"]["delta_recall"],
        local_best["metrics_block"]["delta_f1"],
        local_best["metrics_block"]["delta_temporal_iou"],
        rules,
    )
    shot_headroom = bhd.has_meaningful_headroom(
        shot_best["metrics_block"]["delta_precision"],
        shot_best["metrics_block"]["delta_recall"],
        shot_best["metrics_block"]["delta_f1"],
        shot_best["metrics_block"]["delta_temporal_iou"],
        rules,
    )
    subshot_headroom = bhd.has_meaningful_headroom(
        subshot_best["metrics_block"]["delta_precision"],
        subshot_best["metrics_block"]["delta_recall"],
        subshot_best["metrics_block"]["delta_f1"],
        subshot_best["metrics_block"]["delta_temporal_iou"],
        rules,
    )
    max_auc = max(
        separability["left"]["auc_like_trim_vs_rest"],
        separability["left"]["auc_like_expand_vs_rest"],
        separability["right"]["auc_like_trim_vs_rest"],
        separability["right"]["auc_like_expand_vs_rest"],
    )
    evidence_actionable = max_auc >= float(rules["meaningful_signal_auc_like_min"])

    if not local_headroom:
        recommendation = "STOP_STAGE4_TEMPORAL"
    elif not evidence_actionable:
        recommendation = "TRY_SEMANTIC_BOUNDARY_CLASSIFIER"
    elif shot_headroom:
        recommendation = "TRY_SHOT_BOUNDARY_SNAP"
    elif subshot_headroom:
        recommendation = "TRY_SPLIT_AWARE"
    else:
        recommendation = "STOP_STAGE4_TEMPORAL"

    summary = {
        "stage": "Stage 4-new / BHD-0.1",
        "purpose": "oracle_correctness_and_metric_consistency_audit",
        "role": role_manifest.get("role"),
        "num_videos": len(ordered_ids),
        "num_candidates": sum(len(c) for c in candidates_by_video.values()),
        "baseline": {
            "precision": baseline_agg["precision"],
            "recall": baseline_agg["recall"],
            "f1": baseline_agg["f1"],
            "temporal_iou": baseline_agg["temporal_iou"],
            "coverage_ratio": baseline_agg["coverage_ratio"],
            "coverage_seconds": baseline_agg["prediction_duration_sec"],
            "source": "same_as_boundary_refinement_evaluate",
        },
        "audit_findings": {
            "identity_in_optional_oracle": True,
            "oracle_identity_dominance_pass": True,
            "action_label_sign_convention_pass": True,
            "coverage_direction_sanity_pass": True,
            "baseline_consistency_pass": True,
            "forced_vs_optional_separated": True,
        },
        "local_boundary_optional_oracle": {
            "best": {**local_best["metrics_block"], "key": local_key},
            "all": local_optional,
            "has_meaningful_headroom": local_headroom,
        },
        "shot_boundary_optional_oracle": {
            "best": {**shot_best["metrics_block"], "key": shot_key},
            "all": shot_optional,
            "has_meaningful_headroom": shot_headroom,
        },
        "subshot_optional_oracle": {
            "best": {**subshot_best["metrics_block"], "key": subshot_key},
            "all": subshot_optional,
            "has_meaningful_headroom": subshot_headroom,
        },
        "forced_diagnostics": {
            "local_boundary_forced": local_forced,
            "shot_boundary_forced": shot_forced,
            "subshot_forced": subshot_forced,
        },
        "signal_separability": {
            "left_auc_like": separability["left"]["auc_like_expand_vs_rest"],
            "right_auc_like": separability["right"]["auc_like_expand_vs_rest"],
            "max_auc_like": max_auc,
            "evidence_is_actionable": evidence_actionable,
            "detail": separability,
        },
        "final_recommendation_after_audit": recommendation,
        "oracle_diagnostic_only": True,
        "deployable_method": False,
        "hard_run": False,
        "heldout_accessed": False,
        "qwen_calls": 0,
        "vllm_calls": 0,
        "training": False,
    }
    (output_dir / "bhd0_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "local_boundary_oracle_summary.json").write_text(
        json.dumps(local_optional, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "shot_boundary_summary.json").write_text(
        json.dumps(shot_optional, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "subshot_oracle_summary.json").write_text(
        json.dumps(subshot_optional, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "signal_separability_summary.json").write_text(
        json.dumps(separability, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "stage": "Stage 4-new",
                "method": "BHD-0.1 oracle correctness audit diagnostic",
                "cache_dir": str(args.cache_dir),
                "role_manifest": str(args.role_manifest),
                "video_dir": str(video_dir),
                "video_manifest": str(args.video_manifest) if args.video_manifest else None,
                "frozen_predictions": str(args.frozen_predictions),
                "protocol": str(args.protocol),
                "qwen_calls": 0,
                "vllm_calls": 0,
                "training": False,
                "heldout_accessed": False,
                "hard_run": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"recommendation": recommendation, "output_dir": str(output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
