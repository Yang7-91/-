"""Stage 4-new BR-2 asymmetric local-evidence boundary refinement CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic_video_highlight.highlight_retrieval.br2_asymmetric_boundary import (  # noqa: E402
    propose_br2_boundary,
    run_br2_for_role,
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

PROTOCOL_SCHEMA_VERSION = "aic.stage4.br2_asymmetric_boundary/v1"


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
    candidate = video_dir / f"{video_id}.mp4"
    return candidate if candidate.is_file() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 4-new BR-2 asymmetric local-evidence boundary refinement"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    propose = commands.add_parser(
        "propose", help="Propose BR-2 asymmetric boundaries on dev_tune"
    )
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
            f"unknown BR-2 config: {args.config_name}; available: {sorted(protocol['configs'])}"
        )

    manifest, records = load_cache_payloads(args.cache_dir)
    role_path = args.role_manifest.expanduser().resolve()
    validate_role_manifest_directory(role_path.parent, cache_manifest=manifest)
    role_manifest = _read_json_object(role_path)
    if role_manifest.get("role") != protocol.get("role"):
        raise SystemExit(
            f"role mismatch: protocol requires {protocol.get('role')!r}, got {role_manifest.get('role')!r}"
        )

    video_dir = args.video_dir.expanduser().resolve()
    video_map = _load_video_manifest(
        args.video_manifest.expanduser().resolve() if args.video_manifest else None
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import cv2

    ordered_ids = [item["video_id"] for item in role_manifest["records"]]
    missing = [
        video_id
        for video_id in ordered_ids
        if _resolve_video_path(video_dir, video_id, video_map) is None
    ]
    if missing:
        raise SystemExit(
            f"{len(missing)} dev_tune videos missing from {video_dir}; "
            f"first few: {missing[:5]}"
        )

    evidence_rows: list[dict] = []

    def propose_fn(video_id: str, candidate: dict, record: dict):
        video_path = _resolve_video_path(video_dir, video_id, video_map)
        capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        if not capture.isOpened():
            raise SystemExit(f"OpenCV could not decode video: {video_path}")
        try:
            rows, refinement = propose_br2_boundary(
                capture,
                candidate,
                config,
                video_id=video_id,
                duration_sec=float(record["duration_sec"]),
            )
        finally:
            capture.release()
        evidence_rows.extend(rows)
        return rows, refinement

    result = run_br2_for_role(
        manifest, role_manifest, records, args.config_name, config, propose_fn
    )
    validate_boundary_refinement_payload(result, manifest, role_manifest, records)

    result_path = output_dir / "refinement_result.json"
    result_path.write_bytes(canonical_json_bytes(result))

    flat_rows = []
    for record_item in result["records"]:
        video_id = record_item["video_id"]
        cache_record = records[video_id]
        for candidate, refinement in zip(
            cache_record["merged_candidates"],
            record_item["candidate_refinements"],
            strict=True,
        ):
            detail = refinement["decision_detail"]
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
                    "decision_detail": detail,
                    "provenance": {
                        "qwen_calls": 0,
                        "vllm_calls": 0,
                        "training": False,
                        "heldout_access": False,
                        "oracle": False,
                    },
                }
            )
    write_json_lines(output_dir / "refined_candidates.jsonl", flat_rows)
    write_json_lines(output_dir / "boundary_evidence.jsonl", evidence_rows)
    write_json_lines(output_dir / "candidate_decisions.jsonl", flat_rows)

    left_actions = {"TRIM": 0, "KEEP": 0, "EXPAND": 0}
    right_actions = {"TRIM": 0, "KEEP": 0, "EXPAND": 0}
    for row in flat_rows:
        detail = row["decision_detail"]
        if row["decision"] == "REFINE":
            left_actions[detail["left_action"]] += 1
            right_actions[detail["right_action"]] += 1
        else:
            left_actions["KEEP"] += 1
            right_actions["KEEP"] += 1

    proposal_summary = {
        "config_name": args.config_name,
        "role": role_manifest.get("role"),
        "decision_counts": result["decision_counts"],
        "input_candidate_count": result["input_candidate_count"],
        "left_actions": left_actions,
        "right_actions": right_actions,
        "video_count": len(ordered_ids),
        "heldout_accessed": False,
        "hard_run": False,
        "qwen_calls": 0,
        "vllm_calls": 0,
        "training": False,
    }
    (output_dir / "proposal_summary.json").write_text(
        json.dumps(proposal_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "stage": "Stage 4-new",
                "method": "BR-2 asymmetric local-evidence boundary refinement",
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
