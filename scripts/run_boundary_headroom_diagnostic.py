"""Stage 4-new BHD-0 boundary headroom and signal separability diagnostic CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
        candidate = video_dir / relative
        if candidate.is_file():
            return candidate
        candidate = video_dir / Path(relative).name
        if candidate.is_file():
            return candidate
    return video_dir / f"{video_id}.mp4"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 4-new BHD-0 boundary headroom diagnostic (oracle, non-deployable)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    diagnose = commands.add_parser("diagnose", help="Run all BHD-0 oracle diagnostics")
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

    baseline_agg = bhd.baseline_aggregate(frozen_rows.get(v, {}) for v in ordered_ids)

    candidates_by_video: dict[str, list] = {}
    references_by_video: dict[str, list] = {}
    for video_id in ordered_ids:
        record = records[video_id]
        candidates_by_video[video_id] = record["merged_candidates"]
        references_by_video[video_id] = bhd._candidate_refs(frozen_rows.get(video_id, {}))

    print("[1/4] local boundary oracle ...", flush=True)
    local_oracle = bhd.run_local_boundary_oracle(
        candidates_by_video,
        references_by_video,
        [float(x) for x in oracle_cfg["local_boundary_search"]["max_shift_sec"]],
        float(oracle_cfg["local_boundary_search"]["step_sec"]),
    )
    write_json_lines(
        output_dir / "boundary_oracle_labels.jsonl", local_oracle["oracle_labels"]
    )
    local_summary = {}
    for shift, payload in local_oracle["levels"].items():
        aggregate = payload["aggregate"]
        local_summary[shift] = {
            "delta_precision": aggregate["precision"] - baseline_agg["precision"],
            "delta_recall": aggregate["recall"] - baseline_agg["recall"],
            "delta_f1": aggregate["f1"] - baseline_agg["f1"],
            "delta_temporal_iou": aggregate["temporal_iou"] - baseline_agg["temporal_iou"],
            "delta_coverage": aggregate["prediction_duration_sec"] - baseline_agg["prediction_duration_sec"],
            "action_counts": payload["action_counts"],
            "aggregate": aggregate,
        }

    print("[2/4] shot boundary detection ...", flush=True)
    import cv2

    shot_rows: list[dict] = []
    shot_oracle_by_shift: dict[str, dict] = {
        str(shift): {"rows": [], "per_candidate": {}} for shift in shot_cfg["max_snap_shift_sec"]
    }
    candidate_alignment_rows: list[dict] = []
    for video_id in ordered_ids:
        video_path = _resolve_video_path(video_dir, video_id, video_map)
        record = records[video_id]
        duration = float(record["duration_sec"])
        boundaries = bhd.detect_shot_boundaries(
            video_path,
            lambda p: cv2.VideoCapture(str(p), cv2.CAP_FFMPEG),
            frame_stride_sec=float(shot_cfg["frame_stride_sec"]),
            min_shot_len_sec=float(shot_cfg["min_shot_len_sec"]),
            duration_sec=duration,
        )
        segments = bhd.shot_segments_from_boundaries(boundaries, duration)
        shot_rows.append(
            {
                "video_id": video_id,
                "duration_sec": duration,
                "num_shot_boundaries": len(boundaries),
                "num_shot_segments": len(segments),
                "boundaries_sec": boundaries,
            }
        )
        refs = references_by_video[video_id]
        for candidate in record["merged_candidates"]:
            candidate_id = str(candidate["merged_candidate_id"])
            start = float(candidate["start_sec"])
            end = float(candidate["end_sec"])
            nearest_edge = min(
                (abs(e - start) for s in segments for e in s), default=duration
            )
            nearest_edge = min(
                nearest_edge, min((abs(e - end) for s in segments for e in s), default=duration)
            )
            ref_nearest = None
            if refs:
                ref_points = [point for seg in refs for point in seg]
                ref_nearest = min(abs(point - e) for point in ref_points for e in [x for s in segments for x in s]) if segments else None
            candidate_alignment_rows.append(
                {
                    "video_id": video_id,
                    "candidate_id": candidate_id,
                    "candidate_edge_to_nearest_shot_edge_sec": bhd._finite(nearest_edge),
                    "reference_edge_to_nearest_shot_edge_sec": bhd._finite(ref_nearest) if ref_nearest is not None else None,
                    "num_shots_inside_candidate": sum(
                        1 for lo, hi in segments if lo >= start - 1e-9 and hi <= end + 1e-9
                    ),
                }
            )
            for shift in shot_cfg["max_snap_shift_sec"]:
                snap = bhd.snap_oracle_to_shot_boundary(
                    candidate, refs, segments, float(shift)
                )
                if snap.get("snapped"):
                    shot_oracle_by_shift[str(shift)]["per_candidate"][candidate_id] = snap

    print("[3/4] shot + subshot oracle aggregation ...", flush=True)
    shot_summary = {}
    for shift, payload in shot_oracle_by_shift.items():
        rows = list(payload["per_candidate"].values())
        aggregate = bhd.aggregate_oracle_metrics(rows) if rows else {}
        shot_summary[str(shift)] = {
            "snapped_candidates": len(rows),
            "aggregate": aggregate,
            "delta_f1": (aggregate.get("f1", 0.0) - baseline_agg["f1"]) if rows else 0.0,
            "delta_temporal_iou": (aggregate.get("temporal_iou", 0.0) - baseline_agg["temporal_iou"]) if rows else 0.0,
            "delta_precision": (aggregate.get("precision", 0.0) - baseline_agg["precision"]) if rows else 0.0,
            "delta_recall": (aggregate.get("recall", 0.0) - baseline_agg["recall"]) if rows else 0.0,
        }

    subshot_summary = {}
    for max_components in oracle_cfg["subshot_split_upper_bound"]["max_components"]:
        rows = []
        for video_id in ordered_ids:
            segments = bhd.shot_segments_from_boundaries(
                shot_rows[[r["video_id"] for r in shot_rows].index(video_id)]["boundaries_sec"],
                float(records[video_id]["duration_sec"]),
            )
            refs = references_by_video[video_id]
            for candidate in records[video_id]["merged_candidates"]:
                oracle = bhd.oracle_subshot_split_upper_bound(
                    candidate, refs, segments, int(max_components)
                )
                if oracle.get("metrics"):
                    oracle["video_id"] = video_id
                    rows.append(oracle)
        aggregate = bhd.aggregate_oracle_metrics(rows) if rows else {}
        subshot_summary[str(max_components)] = {
            "aggregate": aggregate,
            "delta_f1": (aggregate.get("f1", 0.0) - baseline_agg["f1"]) if rows else 0.0,
            "delta_temporal_iou": (aggregate.get("temporal_iou", 0.0) - baseline_agg["temporal_iou"]) if rows else 0.0,
            "delta_precision": (aggregate.get("precision", 0.0) - baseline_agg["precision"]) if rows else 0.0,
            "delta_recall": (aggregate.get("recall", 0.0) - baseline_agg["recall"]) if rows else 0.0,
        }

    print("[4/4] BR-2 evidence separability ...", flush=True)
    max_shift_for_labels = max(
        float(x) for x in oracle_cfg["local_boundary_search"]["max_shift_sec"]
    )
    oracle_label_lookup = {
        (row["video_id"], row["candidate_id"]): row
        for row in local_oracle["oracle_labels"]
    }
    side_scores: dict[str, dict[str, list[float]]] = {
        "left": {"TRIM": [], "KEEP": [], "EXPAND": []},
        "right": {"TRIM": [], "KEEP": [], "EXPAND": []},
    }
    br2_config = json.loads(
        (Path(__file__).resolve().parents[1] / "configs" / "stage4_br2_asymmetric_boundary_protocol.json").read_text(
            encoding="utf-8"
        )
    )["configs"]["BR-2-C2"]
    for video_id in ordered_ids:
        video_path = _resolve_video_path(video_dir, video_id, video_map)
        record = records[video_id]
        duration = float(record["duration_sec"])
        import cv2

        capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        if not capture.isOpened():
            continue
        try:
            for candidate in record["merged_candidates"]:
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
                    inside_scores = [
                        row["core_similarity_histogram"] for row in rows if row["inside"]
                    ]
                    outside_scores = [
                        row["core_similarity_histogram"] for row in rows if not row["inside"]
                    ]
                    score = float(np.median(inside_scores)) if inside_scores else 0.0
                    outside_strength = float(np.median(outside_scores)) if outside_scores else 0.0
                    action = label[f"{side}_action"]
                    side_scores[side][action].append(
                        score if action != "EXPAND" else outside_strength
                    )
        finally:
            capture.release()

    separability = {}
    for side in ("left", "right"):
        trim_scores = side_scores[side]["TRIM"]
        keep_scores = side_scores[side]["KEEP"]
        expand_scores = side_scores[side]["EXPAND"]
        auc_trim = max(
            bhd.rank_auc_like(trim_scores, keep_scores + expand_scores),
            1.0 - bhd.rank_auc_like(trim_scores, keep_scores + expand_scores),
        ) if trim_scores and (keep_scores or expand_scores) else 0.5
        auc_expand = max(
            bhd.rank_auc_like(expand_scores, keep_scores + trim_scores),
            1.0 - bhd.rank_auc_like(expand_scores, keep_scores + trim_scores),
        ) if expand_scores and (keep_scores or trim_scores) else 0.5
        separability[side] = {
            "label_counts": {
                action: len(side_scores[side][action]) for action in ("TRIM", "KEEP", "EXPAND")
            },
            "auc_like_trim_vs_rest": bhd._finite(auc_trim),
            "auc_like_expand_vs_rest": bhd._finite(auc_expand),
            "mean_gap_trim": bhd.mean_gap(trim_scores, keep_scores + expand_scores),
            "mean_gap_expand": bhd.mean_gap(expand_scores, keep_scores + trim_scores),
            "effect_size_trim": bhd.effect_size(trim_scores, keep_scores + expand_scores),
            "effect_size_expand": bhd.effect_size(expand_scores, keep_scores + trim_scores),
        }

    best_local_shift = max(
        local_summary,
        key=lambda s: max(local_summary[s]["delta_f1"], local_summary[s]["delta_temporal_iou"]),
    )
    best_local = local_summary[best_local_shift]
    best_shot_shift = max(
        shot_summary, key=lambda s: max(shot_summary[s]["delta_f1"], shot_summary[s]["delta_temporal_iou"])
    )
    best_shot = shot_summary[best_shot_shift]
    best_components = max(
        subshot_summary, key=lambda s: max(subshot_summary[s]["delta_f1"], subshot_summary[s]["delta_temporal_iou"])
    )
    best_subshot = subshot_summary[best_components]

    local_headroom = bhd.has_meaningful_headroom(
        best_local["delta_precision"],
        best_local["delta_recall"],
        best_local["delta_f1"],
        best_local["delta_temporal_iou"],
        rules,
    )
    shot_headroom = bhd.has_meaningful_headroom(
        best_shot["delta_precision"],
        best_shot["delta_recall"],
        best_shot["delta_f1"],
        best_shot["delta_temporal_iou"],
        rules,
    )
    subshot_headroom = bhd.has_meaningful_headroom(
        best_subshot["delta_precision"],
        best_subshot["delta_recall"],
        best_subshot["delta_f1"],
        best_subshot["delta_temporal_iou"],
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
        "stage": "Stage 4-new / BHD-0",
        "role": role_manifest.get("role"),
        "num_videos": len(ordered_ids),
        "num_candidates": sum(len(c) for c in candidates_by_video.values()),
        "baseline": {
            "method": "BR-0 / Frozen Retrieval v0",
            "precision": baseline_agg["precision"],
            "recall": baseline_agg["recall"],
            "f1": baseline_agg["f1"],
            "temporal_iou": baseline_agg["temporal_iou"],
            "coverage": baseline_agg["prediction_duration_sec"],
        },
        "local_boundary_oracle": {
            "best_max_shift_sec": float(best_local_shift),
            "best_delta_precision": best_local["delta_precision"],
            "best_delta_recall": best_local["delta_recall"],
            "best_delta_f1": best_local["delta_f1"],
            "best_delta_temporal_iou": best_local["delta_temporal_iou"],
            "best_delta_coverage": best_local["delta_coverage"],
            "has_meaningful_headroom": local_headroom,
            "oracle_markers": dict(bhd.BHD0_MARKERS),
        },
        "shot_boundary_oracle": {
            "best_max_shift_sec": float(best_shot_shift),
            "best_delta_precision": best_shot["delta_precision"],
            "best_delta_recall": best_shot["delta_recall"],
            "best_delta_f1": best_shot["delta_f1"],
            "best_delta_temporal_iou": best_shot["delta_temporal_iou"],
            "has_meaningful_headroom": shot_headroom,
            "oracle_markers": dict(bhd.BHD0_MARKERS),
        },
        "subshot_oracle": {
            "best_max_components": int(best_components),
            "best_delta_precision": best_subshot["delta_precision"],
            "best_delta_recall": best_subshot["delta_recall"],
            "best_delta_f1": best_subshot["delta_f1"],
            "best_delta_temporal_iou": best_subshot["delta_temporal_iou"],
            "has_meaningful_headroom": subshot_headroom,
            "oracle_markers": dict(bhd.BHD0_MARKERS),
        },
        "signal_separability": {
            "left_auc_like": separability["left"]["auc_like_trim_vs_rest"],
            "right_auc_like": separability["right"]["auc_like_trim_vs_rest"],
            "max_auc_like": max_auc,
            "evidence_is_actionable": evidence_actionable,
            "detail": separability,
        },
        "recommendation": recommendation,
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
        json.dumps(local_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "shot_boundary_summary.json").write_text(
        json.dumps(shot_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "subshot_oracle_summary.json").write_text(
        json.dumps(subshot_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "signal_separability_summary.json").write_text(
        json.dumps(separability, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_json_lines(output_dir / "shot_boundaries.jsonl", shot_rows)
    write_json_lines(output_dir / "candidate_boundary_alignment.jsonl", candidate_alignment_rows)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "stage": "Stage 4-new",
                "method": "BHD-0 boundary headroom diagnostic",
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
