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

PROTOCOL_SCHEMA_VERSION = "aic.stage4.sabr1_protocol/v1"


def _load_protocol(path: Path) -> dict:
    protocol = _read_json_object(path)
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise SystemExit(
            f"protocol schema mismatch: expected {PROTOCOL_SCHEMA_VERSION}, "
            f"got {protocol.get('schema_version')}"
        )
    safety = protocol.get("safety", {})
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
    args = parser.parse_args(argv)

    if args.command != "extract":
        parser.error(f"unknown command: {args.command}")

    protocol = _load_protocol(args.protocol)
    bin_sec = float(args.bin_sec)
    if bin_sec <= 0:
        parser.error("--bin-sec must be positive")

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
