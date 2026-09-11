"""Stage 4-new SABR-1.0 saliency feature extraction smoke CLI.

Reads the frozen Stage 4.2 cache and a Stage 4.3 role manifest, extracts
per-bin visual saliency signals inside frozen candidate intervals for a small
sample of Dev videos, and writes feature summaries.  This CLI never changes
candidate boundaries, never emits refined candidates or predictions, and never
calls any model.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic_video_highlight.highlight_retrieval.boundary_refinement import (  # noqa: E402
    load_cache_payloads,
)
from aic_video_highlight.highlight_retrieval.candidate_selection import (  # noqa: E402
    _read_json_object,
    validate_role_manifest_directory,
)
from aic_video_highlight.highlight_retrieval.saliency_anchor import (  # noqa: E402
    SALIENCY_ANCHOR_SCHEMA_VERSION,
    SaliencyExtractionError,
    extract_saliency_bins,
    normalize_saliency_bins,
    summarize_candidate_saliency,
    write_json_lines,
)
from aic_video_highlight.highlight_retrieval.sabr_boundary import (  # noqa: E402
    build_sabr11_result,
)
from aic_video_highlight.highlight_retrieval.boundary_refinement import (  # noqa: E402
    validate_boundary_refinement_payload,
)

PROTOCOL_SCHEMA_VERSION = "aic.stage4.sabr1_protocol/v1"


def _load_protocol(path: Path, *, boundary_phase: bool = False) -> dict:
    protocol = _read_json_object(path)
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise SystemExit(
            f"protocol schema mismatch: expected {PROTOCOL_SCHEMA_VERSION}, "
            f"got {protocol.get('schema_version')}"
        )
    safety = protocol.get("safety", {})
    if boundary_phase:
        proposal = protocol.get("boundary_proposal")
        if not isinstance(proposal, dict) or proposal.get("changes_boundaries") is not True:
            raise SystemExit("protocol does not declare a boundary proposal phase")
        if safety.get("no_heldout") is not True or safety.get("no_qwen") is not True:
            raise SystemExit("protocol safety flags violate SABR isolation rules")
        return protocol
    if not (
        safety.get("this_phase_changes_boundaries") is False
        and safety.get("no_heldout") is True
        and safety.get("no_qwen") is True
        and safety.get("no_vllm") is True
    ):
        raise SystemExit("protocol safety flags do not permit this extraction phase")
    return protocol


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


def _load_video_manifest(path: Path | None) -> dict[str, str]:
    """Load optional video_id -> relative path mapping from a JSONL manifest."""
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


def _cmd_propose(args) -> int:
    protocol = _load_protocol(args.protocol, boundary_phase=True)
    proposal = protocol["boundary_proposal"]
    config_name = args.config_name
    config = proposal["configs"].get(config_name)
    if not isinstance(config, dict):
        raise SystemExit(
            f"unknown SABR-1.1 config: {config_name}; "
            f"available: {sorted(proposal['configs'])}"
        )

    manifest, records = load_cache_payloads(args.cache_dir)
    role_path = args.role_manifest.expanduser().resolve()
    validate_role_manifest_directory(role_path.parent, cache_manifest=manifest)
    role_manifest = _read_json_object(role_path)
    role_name = role_manifest.get("role")
    if role_name not in proposal.get("allowed_roles", []):
        raise SystemExit(
            f"role {role_name!r} is not allowed for SABR-1.1; "
            f"allowed: {proposal.get('allowed_roles')}"
        )

    video_dir = args.video_dir.expanduser().resolve()
    video_map = _load_video_manifest(
        args.video_manifest.expanduser().resolve() if args.video_manifest else None
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ordered_ids = [item["video_id"] for item in role_manifest["records"]]
    missing = [
        video_id
        for video_id in ordered_ids
        if _resolve_video_path(video_dir, video_id, video_map) is None
    ]
    if missing:
        raise SystemExit(
            f"{len(missing)} dev_tune videos missing from {video_dir}; "
            f"first few: {missing[:5]} (download them before proposing boundaries)"
        )

    fb_config = dict(config)
    fb_config.update(proposal.get("identity_fallback", {}))
    all_bins: list[dict] = []

    def propose_fn(video_id: str, candidate: dict, record: dict):
        video_path = _resolve_video_path(video_dir, video_id, video_map)
        bins = normalize_saliency_bins(
            extract_saliency_bins(
                video_path,
                float(candidate["start_sec"]),
                float(candidate["end_sec"]),
                float(config["bin_sec"]),
                video_id=video_id,
                candidate_id=str(candidate["merged_candidate_id"]),
            )
        )
        all_bins.extend(bins)
        refinement = propose_conservative_boundary(
            candidate,
            bins,
            fb_config,
            duration_sec=float(record["duration_sec"]),
        )
        return bins, refinement

    result = build_sabr11_result(
        manifest, role_manifest, records, config_name, fb_config, propose_fn
    )
    validate_boundary_refinement_payload(result, manifest, role_manifest, records)

    result_path = output_dir / "refinement_result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
    )

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
                    "refined_candidate_id": f"{config_name}:{candidate['merged_candidate_id']}",
                    "original_start_sec": refinement["original_start_sec"],
                    "original_end_sec": refinement["original_end_sec"],
                    "start_sec": refinement["refined_start_sec"],
                    "end_sec": refinement["refined_end_sec"],
                    "score": candidate["score"],
                    "reason": candidate.get("reason", ""),
                    "source_chunk": candidate.get("source_chunk"),
                    "method": config_name,
                    "decision": refinement["decision"],
                    "decision_detail": detail,
                    "provenance": {
                        "qwen_calls": 0,
                        "vllm_calls": 0,
                        "training": False,
                        "heldout_access": False,
                    },
                }
            )
    write_json_lines(output_dir / "refined_candidates.jsonl", flat_rows)
    write_json_lines(output_dir / "saliency_bins.jsonl", all_bins)
    write_json_lines(output_dir / "candidate_decisions.jsonl", flat_rows)

    proposal_summary = {
        "config_name": config_name,
        "role": role_name,
        "refiner_name": result["refiner_name"],
        "decision_counts": result["decision_counts"],
        "input_candidate_count": result["input_candidate_count"],
        "num_trimmed": result["decision_counts"]["REFINE"],
        "num_fallback_identity": result["decision_counts"]["IDENTITY_FALLBACK"],
        "num_identity": result["decision_counts"]["IDENTITY"],
        "bin_count": len(all_bins),
        "video_count": len(ordered_ids),
        "heldout_accessed": False,
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
                "method": "SABR-1.1 conservative boundary proposal",
                "config_name": config_name,
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
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(proposal_summary))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 4-new SABR-1.0 saliency feature extraction smoke"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract", help="Extract saliency bins for a small Dev sample")
    extract.add_argument("--cache-dir", type=Path, required=True)
    extract.add_argument("--role-manifest", type=Path, required=True)
    extract.add_argument("--video-dir", type=Path, required=True)
    extract.add_argument("--video-manifest", type=Path, default=None)
    extract.add_argument("--protocol", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--limit", type=int, default=5)
    extract.add_argument("--bin-sec", type=float, default=1.0)
    propose = commands.add_parser(
        "propose", help="Propose conservative SABR-1.1 boundaries on dev_tune"
    )
    propose.add_argument("--cache-dir", type=Path, required=True)
    propose.add_argument("--role-manifest", type=Path, required=True)
    propose.add_argument("--video-dir", type=Path, required=True)
    propose.add_argument("--video-manifest", type=Path, default=None)
    propose.add_argument("--protocol", type=Path, required=True)
    propose.add_argument("--config-name", required=True)
    propose.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "extract":
        return _cmd_extract(args)
    if args.command == "propose":
        return _cmd_propose(args)
    parser.error(f"unknown command: {args.command}")
    return 2

def _cmd_extract(args) -> int:
    protocol = _load_protocol(args.protocol)
    bin_sec = float(args.bin_sec)
    if bin_sec <= 0:
        raise SystemExit("--bin-sec must be positive")

    manifest, records = load_cache_payloads(args.cache_dir)
    role_path = args.role_manifest.expanduser().resolve()
    validate_role_manifest_directory(role_path.parent, cache_manifest=manifest)
    role_manifest = _read_json_object(role_path)
    if role_manifest.get("role") != "dev_tune":
        raise SystemExit("SABR-1.0 smoke only permits the dev_tune role")

    video_dir = args.video_dir.expanduser().resolve()
    video_map = _load_video_manifest(
        args.video_manifest.expanduser().resolve() if args.video_manifest else None
    )
    output_dir = args.output_dir.expanduser().resolve()
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    ordered_ids = [item["video_id"] for item in role_manifest["records"]]
    selected: list[str] = []
    skipped_missing: list[str] = []
    for video_id in ordered_ids:
        if len(selected) >= args.limit:
            break
        if _resolve_video_path(video_dir, video_id, video_map) is not None:
            selected.append(video_id)
        else:
            skipped_missing.append(video_id)

    all_bins: list[dict] = []
    summaries: list[dict] = []
    errors: list[dict] = []
    for video_id in selected:
        video_path = _resolve_video_path(video_dir, video_id, video_map)
        record = records[video_id]
        for candidate in record["merged_candidates"]:
            candidate_id = candidate["merged_candidate_id"]
            try:
                bins = extract_saliency_bins(
                    video_path,
                    float(candidate["start_sec"]),
                    float(candidate["end_sec"]),
                    bin_sec,
                    video_id=video_id,
                    candidate_id=candidate_id,
                )
            except SaliencyExtractionError as exc:
                errors.append({"video_id": video_id, "candidate_id": candidate_id, "error": str(exc)})
                continue
            bins = normalize_saliency_bins(bins)
            all_bins.extend(bins)
            summaries.append(summarize_candidate_saliency(video_id, candidate_id, bins))

    extraction_summary = {
        "extraction_schema_version": SALIENCY_ANCHOR_SCHEMA_VERSION,
        "protocol_schema_version": protocol["schema_version"],
        "source_cache_global_hash": manifest["global_semantic_sha256"],
        "role": role_manifest["role"],
        "role_manifest_hash": role_manifest["semantic_sha256"],
        "requested_limit": args.limit,
        "selected_video_count": len(selected),
        "selected_video_ids": selected,
        "skipped_missing_videos": skipped_missing,
        "candidate_count": len(summaries),
        "bin_count": len(all_bins),
        "error_count": len(errors),
        "errors": errors,
        "bin_sec": bin_sec,
        "this_phase_changes_boundaries": False,
        "deterministic": True,
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    write_json_lines(output_dir / "saliency_bins.jsonl", all_bins)
    (output_dir / "candidate_saliency_summary.json").write_text(
        json.dumps(
            {
                "extraction_schema_version": SALIENCY_ANCHOR_SCHEMA_VERSION,
                "summaries": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "stage": "Stage 4-new",
                "method": "SABR-1.0 feature smoke",
                "cache_dir": str(args.cache_dir),
                "role_manifest": str(args.role_manifest),
                "video_dir": str(video_dir),
                "video_manifest": str(args.video_manifest) if args.video_manifest else None,
                "protocol": str(args.protocol),
                "protocol_schema_version": protocol["schema_version"],
                "limit": args.limit,
                "bin_sec": bin_sec,
                "qwen_calls": 0,
                "vllm_calls": 0,
                "heldout_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "extraction_summary.json").write_text(
        json.dumps(extraction_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_video_count": len(selected),
                "skipped_missing_videos": len(skipped_missing),
                "candidate_count": len(summaries),
                "bin_count": len(all_bins),
                "error_count": len(errors),
                "output_dir": str(output_dir),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
