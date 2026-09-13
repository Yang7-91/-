"""Stage 4-new SBA-1 conservative shot-boundary snap CLI.

Deployable rule-based boundary refinement: snap frozen candidate boundaries to
nearby detected shot boundaries.  No weak reference, model, or training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic_video_highlight.highlight_retrieval.boundary_headroom_diagnostic import (  # noqa: E402
    detect_shot_boundaries,
)
from aic_video_highlight.highlight_retrieval.boundary_refinement import (  # noqa: E402
    load_cache_payloads,
    validate_boundary_refinement_payload,
)
from aic_video_highlight.highlight_retrieval.candidate_selection import (  # noqa: E402
    _read_json_object,
    canonical_json_bytes,
    validate_role_manifest_directory,
)
from aic_video_highlight.highlight_retrieval.saliency_anchor import (  # noqa: E402
    write_json_lines,
)
from aic_video_highlight.highlight_retrieval.shot_boundary_snap import (  # noqa: E402
    run_sba1_for_role,
)

PROTOCOL_SCHEMA_VERSION = "aic.stage4.sba1_shot_boundary_snap/v1"


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 4-new SBA-1 conservative shot-boundary snap (deployable)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    propose = commands.add_parser("propose", help="Propose SBA-1 snaps on dev_tune")
    propose.add_argument("--cache-dir", type=Path, required=True)
    propose.add_argument("--role-manifest", type=Path, required=True)
    propose.add_argument("--video-dir", type=Path, required=True)
    propose.add_argument("--video-manifest", type=Path, default=None)
    propose.add_argument("--protocol", type=Path, required=True)
    propose.add_argument("--config-name", required=True)
    propose.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command != "propose":
        parser.error(f"unknown command: {args.command}")

    protocol = _read_json_object(args.protocol)
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise SystemExit(
            f"protocol schema mismatch: expected {PROTOCOL_SCHEMA_VERSION}, "
            f"got {protocol.get('schema_version')}"
        )
    config = protocol["configs"].get(args.config_name)
    if not isinstance(config, dict):
        raise SystemExit(
            f"unknown SBA-1 config: {args.config_name}; available: {sorted(protocol['configs'])}"
        )

    manifest, records = load_cache_payloads(args.cache_dir)
    role_path = args.role_manifest.expanduser().resolve()
    validate_role_manifest_directory(role_path.parent, cache_manifest=manifest)
    role_manifest = _read_json_object(role_path)
    if role_manifest.get("role") != protocol.get("role"):
        raise SystemExit(
            f"role mismatch: protocol requires {protocol.get('role')!r}, "
            f"got {role_manifest.get('role')!r}"
        )

    video_dir = args.video_dir.expanduser().resolve()
    video_map = _load_video_manifest(
        args.video_manifest.expanduser().resolve() if args.video_manifest else None
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import cv2

    shot_cfg = protocol["shot_detection"]
    ordered_ids = [item["video_id"] for item in role_manifest["records"]]
    missing = [
        video_id
        for video_id in ordered_ids
        if _resolve_video_path(video_dir, video_id, video_map) is None
    ]
    if missing:
        raise SystemExit(
            f"{len(missing)} dev_tune videos missing from {video_dir}; first few: {missing[:5]}"
        )

    shot_rows: list[dict] = []
    shot_boundaries_by_video: dict[str, list[float]] = {}
    for video_id in ordered_ids:
        video_path = _resolve_video_path(video_dir, video_id, video_map)
        duration = float(records[video_id]["duration_sec"])
        boundaries = detect_shot_boundaries(
            video_path,
            lambda p: cv2.VideoCapture(str(p), cv2.CAP_FFMPEG),
            frame_stride_sec=float(shot_cfg["frame_stride_sec"]),
            min_shot_len_sec=float(shot_cfg["min_shot_len_sec"]),
            duration_sec=duration,
        )
        shot_boundaries_by_video[video_id] = boundaries
        shot_rows.append(
            {
                "video_id": video_id,
                "duration_sec": duration,
                "num_shot_boundaries": len(boundaries),
                "boundaries_sec": boundaries,
            }
        )

    result = run_sba1_for_role(
        manifest,
        role_manifest,
        records,
        args.config_name,
        config,
        shot_boundaries_by_video,
    )
    validate_boundary_refinement_payload(result, manifest, role_manifest, records)

    (output_dir / "refinement_result.json").write_bytes(canonical_json_bytes(result))

    flat_rows = []
    for record_item in result["records"]:
        video_id = record_item["video_id"]
        cache_record = records[video_id]
        for candidate, refinement in zip(
            cache_record["merged_candidates"],
            record_item["candidate_refinements"],
            strict=True,
        ):
            flat_rows.append(
                {
                    "video_id": video_id,
                    "original_candidate_id": candidate["merged_candidate_id"],
                    "refined_candidate_id": f"{args.config_name}:{candidate['merged_candidate_id']}",
                    "original_start_sec": refinement["original_start_sec"],
                    "original_end_sec": refinement["original_end_sec"],
                    "start_sec": refinement["refined_start_sec"],
                    "end_sec": refinement["refined_end_sec"],
                    "score": candidate["score"],
                    "reason": candidate.get("reason", ""),
                    "source_chunk": candidate.get("source_chunk"),
                    "method": args.config_name,
                    "decision": refinement["decision"],
                    "decision_detail": refinement["decision_detail"],
                    "provenance": {
                        "qwen_calls": 0,
                        "vllm_calls": 0,
                        "training": False,
                        "heldout_access": False,
                        "uses_weak_reference_for_decision": False,
                        "oracle": False,
                    },
                }
            )
    write_json_lines(output_dir / "refined_candidates.jsonl", flat_rows)
    write_json_lines(output_dir / "shot_boundaries.jsonl", shot_rows)
    write_json_lines(output_dir / "candidate_decisions.jsonl", flat_rows)

    left_actions = {"TRIM": 0, "KEEP": 0, "EXPAND": 0}
    right_actions = {"TRIM": 0, "KEEP": 0, "EXPAND": 0}
    for row in flat_rows:
        detail = row["decision_detail"]
        left_actions[detail.get("left_action", "KEEP")] += 1
        right_actions[detail.get("right_action", "KEEP")] += 1

    proposal_summary = {
        "config_name": args.config_name,
        "role": role_manifest.get("role"),
        "decision_counts": result["decision_counts"],
        "input_candidate_count": result["input_candidate_count"],
        "left_actions": left_actions,
        "right_actions": right_actions,
        "video_count": len(ordered_ids),
        "total_shot_boundaries": sum(len(v) for v in shot_boundaries_by_video.values()),
        "heldout_accessed": False,
        "hard_run": False,
        "qwen_calls": 0,
        "vllm_calls": 0,
        "training": False,
        "uses_weak_reference_for_decision": False,
    }
    (output_dir / "proposal_summary.json").write_text(
        json.dumps(proposal_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "stage": "Stage 4-new",
                "method": "SBA-1 conservative shot-boundary snap (deployable)",
                "config_name": args.config_name,
                "config": config,
                "cache_dir": str(args.cache_dir),
                "role_manifest": str(args.role_manifest),
                "video_dir": str(video_dir),
                "video_manifest": str(args.video_manifest) if args.video_manifest else None,
                "protocol": str(args.protocol),
                "qwen_calls": 0,
                "vllm_calls": 0,
                "training": False,
                "heldout_accessed": False,
                "hard_run": False,
                "uses_weak_reference_for_decision": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(proposal_summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
