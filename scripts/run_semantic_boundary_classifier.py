"""Stage 4-new SBC-0 semantic boundary classifier separability probe CLI.

build-samples -> classify (Qwen via vLLM, fixed prompt) -> evaluate.
Diagnostic only: never writes deployable predictions or touches frozen data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic_video_highlight.highlight_retrieval.boundary_refinement import (  # noqa: E402
    load_cache_payloads,
)
from aic_video_highlight.highlight_retrieval.candidate_selection import (  # noqa: E402
    _read_json_object,
    validate_role_manifest_directory,
)
from aic_video_highlight.highlight_retrieval.qwen_vllm_client import (  # noqa: E402
    QwenVLLMClient,
)
from aic_video_highlight.highlight_retrieval.saliency_anchor import (  # noqa: E402
    write_json_lines,
)
from aic_video_highlight.highlight_retrieval.semantic_boundary_classifier import (  # noqa: E402
    SBC0_SCHEMA_VERSION,
    BoundarySideSample,
    SemanticBoundaryParseError,
    SemanticBoundaryPrediction,
    build_boundary_clip_bounds,
    build_stratified_boundary_samples,
    compute_sbc0_metrics,
    load_oracle_boundary_labels,
    parse_semantic_boundary_response,
    render_boundary_prompt,
    sample_to_row,
)
from aic_video_highlight.highlight_retrieval.video_clip import (  # noqa: E402
    extract_video_clip,
)

PROTOCOL_SCHEMA_VERSION = "aic.stage4.sbc0_semantic_boundary_classifier_probe/v1"


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


def _sample_from_row(row: dict) -> BoundarySideSample:
    return BoundarySideSample(
        sample_id=row["sample_id"],
        video_id=row["video_id"],
        candidate_id=row["candidate_id"],
        side=row["side"],
        oracle_action=row["oracle_action"],
        candidate_start_sec=float(row["candidate_start_sec"]),
        candidate_end_sec=float(row["candidate_end_sec"]),
        boundary_sec=float(row["boundary_sec"]),
        candidate_duration_sec=float(row["candidate_duration_sec"]),
    )


def _load_protocol(path: Path) -> dict:
    protocol = _read_json_object(path)
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise SystemExit(
            f"protocol schema mismatch: expected {PROTOCOL_SCHEMA_VERSION}, "
            f"got {protocol.get('schema_version')}"
        )
    return protocol


def _cmd_build_samples(args) -> int:
    protocol = _load_protocol(args.protocol)
    manifest, records = load_cache_payloads(args.cache_dir)
    role_path = args.role_manifest.expanduser().resolve()
    validate_role_manifest_directory(role_path.parent, cache_manifest=manifest)
    role_manifest = _read_json_object(role_path)
    if role_manifest.get("role") != protocol.get("role"):
        raise SystemExit("role mismatch with protocol")

    labels = load_oracle_boundary_labels(args.oracle_labels)
    samples, report = build_stratified_boundary_samples(
        labels,
        records,
        max_samples_total=args.max_samples_total,
        max_per_label_per_side=args.max_per_label_per_side,
        seed=args.seed,
    )
    write_json_lines(args.output, [sample_to_row(s) for s in samples])
    summary_path = args.output.parent / "sbc0_sample_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "oracle_labels": str(args.oracle_labels),
                "role_manifest": str(args.role_manifest),
                "diagnostic_only": True,
                "deployable_method": False,
                **report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report))
    return 0


def _cmd_classify(args) -> int:
    protocol = _load_protocol(args.protocol)
    rows = [
        json.loads(line)
        for line in args.samples.expanduser().resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    samples = [_sample_from_row(row) for row in rows]
    template = args.prompt.expanduser().resolve().read_text(encoding="utf-8")
    video_dir = args.video_dir.expanduser().resolve()
    video_map = _load_video_manifest(
        args.video_manifest.expanduser().resolve() if args.video_manifest else None
    )
    durations = {}
    if args.video_manifest:
        for line in args.video_manifest.expanduser().resolve().read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if isinstance(item.get("duration_sec"), (int, float)):
                    durations[item["video_id"]] = float(item["duration_sec"])
    clip_dir = args.clip_dir.expanduser().resolve()
    clip_dir.mkdir(parents=True, exist_ok=True)

    client = QwenVLLMClient(base_url=args.api_base, model=args.model, timeout_sec=args.timeout_sec)
    if not client.health_check():
        raise SystemExit(f"model {args.model!r} is not visible at {args.api_base}")

    predictions: list[dict] = []
    qwen_calls = 0
    vllm_calls = 0
    parse_failures = 0
    for sample in samples:
        video_path = _resolve_video_path(video_dir, sample.video_id, video_map)
        if video_path is None:
            predictions.append(
                {
                    "sample_id": sample.sample_id,
                    "predicted_action": None,
                    "confidence": None,
                    "rationale_short": None,
                    "parse_ok": False,
                    "error": "video_missing",
                }
            )
            continue
        duration = durations.get(sample.video_id)
        if duration is None:
            try:
                import cv2

                capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                capture.release()
                duration = frames / fps if fps > 0 else 0.0
            except Exception:
                duration = 0.0
        clip_bounds = build_boundary_clip_bounds(
            sample, float(duration), float(protocol["local_context"]["boundary_window_sec"])
        )
        clip_path = clip_dir / f"{sample.side}_{sample.sample_id.replace('|', '_')}.mp4"
        error = None
        payload = None
        try:
            extract_video_clip(
                video_path,
                clip_path,
                start_sec=clip_bounds["clip_start_sec"],
                end_sec=clip_bounds["clip_end_sec"],
            )
        except Exception as exc:  # clip failure is recorded, not silently skipped
            error = f"clip_error: {type(exc).__name__}: {exc}"
        if error is None:
            prompt = render_boundary_prompt(template, sample, clip_bounds)
            qwen_calls += 1
            vllm_calls += 1
            try:
                response = client.analyze_video(
                    clip_path,
                    prompt,
                    max_new_tokens=int(protocol["model"]["max_tokens"]),
                    temperature=float(protocol["model"]["temperature"]),
                    coarse_fps=float(protocol["local_context"]["fps"]),
                )
                payload = parse_semantic_boundary_response(response.content)
            except SemanticBoundaryParseError as exc:
                error = f"parse_error: {exc}"
                parse_failures += 1
            except Exception as exc:  # transport or server failure
                error = f"model_error: {type(exc).__name__}: {exc}"
        predictions.append(
            {
                "sample_id": sample.sample_id,
                "video_id": sample.video_id,
                "side": sample.side,
                "oracle_action": sample.oracle_action,
                "predicted_action": payload["action"] if payload else None,
                "confidence": payload["confidence"] if payload else None,
                "rationale_short": payload["rationale_short"] if payload else None,
                "parse_ok": payload is not None,
                "error": error,
            }
        )
    write_json_lines(args.output, predictions)
    summary = {
        "model": args.model,
        "api_base": args.api_base,
        "temperature": float(protocol["model"]["temperature"]),
        "prompt_version": protocol["prompt"]["version"],
        "prompt_path": str(args.prompt),
        "num_samples": len(samples),
        "qwen_calls": qwen_calls,
        "vllm_calls": vllm_calls,
        "parse_failures": parse_failures,
        "diagnostic_only": True,
        "deployable_method": False,
        "heldout_accessed": False,
    }
    (args.output.parent / "sbc0_classify_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary))
    return 0


def _cmd_evaluate(args) -> int:
    protocol = _load_protocol(args.protocol)
    sample_rows = [
        json.loads(line)
        for line in args.samples.expanduser().resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    prediction_rows = {}
    for line in args.predictions.expanduser().resolve().read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            prediction_rows[row["sample_id"]] = row
    samples = [_sample_from_row(row) for row in sample_rows]
    predictions = [
        SemanticBoundaryPrediction(
            sample_id=s.sample_id,
            predicted_action=prediction_rows.get(s.sample_id, {}).get("predicted_action"),
            confidence=prediction_rows.get(s.sample_id, {}).get("confidence"),
            rationale_short=prediction_rows.get(s.sample_id, {}).get("rationale_short"),
            parse_ok=bool(prediction_rows.get(s.sample_id, {}).get("parse_ok")),
            error=prediction_rows.get(s.sample_id, {}).get("error"),
        )
        for s in samples
    ]
    evaluation = compute_sbc0_metrics(samples, predictions, protocol)
    evaluation["oracle_source"] = "BHD-0.1 boundary_oracle_labels.jsonl"
    evaluation["prompt_version"] = protocol["prompt"]["version"]
    evaluation["model"] = protocol["model"]["name"]
    evaluation["qwen_calls"] = sum(
        1
        for row in prediction_rows.values()
        if row.get("error") is None or str(row.get("error", "")).startswith("parse_error")
    )
    evaluation["vllm_calls"] = evaluation["qwen_calls"]
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().resolve().write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: evaluation[k] for k in ("decision", "schema_success_rate", "accuracy", "macro_f1")}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 4-new SBC-0 semantic boundary classifier separability probe"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-samples", help="Build stratified boundary-side samples")
    build.add_argument("--oracle-labels", type=Path, required=True)
    build.add_argument("--cache-dir", type=Path, required=True)
    build.add_argument("--role-manifest", type=Path, required=True)
    build.add_argument("--video-manifest", type=Path, default=None)
    build.add_argument("--protocol", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--max-samples-total", type=int, default=180)
    build.add_argument("--max-per-label-per-side", type=int, default=30)
    build.add_argument("--seed", type=int, default=20260911)

    classify = commands.add_parser("classify", help="Call the VLM for every sample")
    classify.add_argument("--samples", type=Path, required=True)
    classify.add_argument("--video-dir", type=Path, required=True)
    classify.add_argument("--video-manifest", type=Path, default=None)
    classify.add_argument("--prompt", type=Path, required=True)
    classify.add_argument("--protocol", type=Path, required=True)
    classify.add_argument("--model", default="Qwen/Qwen3.5-4B")
    classify.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    classify.add_argument("--clip-dir", type=Path, required=True)
    classify.add_argument("--timeout-sec", type=float, default=120.0)
    classify.add_argument("--output", type=Path, required=True)

    evaluate = commands.add_parser("evaluate", help="Compute SBC-0 separability metrics")
    evaluate.add_argument("--samples", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--protocol", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    run = commands.add_parser("run", help="build-samples then classify then evaluate")
    run.add_argument("--oracle-labels", type=Path, required=True)
    run.add_argument("--cache-dir", type=Path, required=True)
    run.add_argument("--role-manifest", type=Path, required=True)
    run.add_argument("--video-dir", type=Path, required=True)
    run.add_argument("--video-manifest", type=Path, default=None)
    run.add_argument("--prompt", type=Path, required=True)
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--model", default="Qwen/Qwen3.5-4B")
    run.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    run.add_argument("--clip-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "build-samples":
        return _cmd_build_samples(args)
    if args.command == "classify":
        return _cmd_classify(args)
    if args.command == "evaluate":
        return _cmd_evaluate(args)
    if args.command == "run":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        build_args = argparse.Namespace(
            oracle_labels=args.oracle_labels,
            cache_dir=args.cache_dir,
            role_manifest=args.role_manifest,
            protocol=args.protocol,
            output=args.output_dir / "sbc0_samples.jsonl",
            max_samples_total=180,
            max_per_label_per_side=30,
            seed=20260911,
        )
        classify_args = argparse.Namespace(
            samples=build_args.output,
            video_dir=args.video_dir,
            video_manifest=args.video_manifest,
            prompt=args.prompt,
            protocol=args.protocol,
            model=args.model,
            api_base=args.api_base,
            clip_dir=args.clip_dir,
            timeout_sec=120.0,
            output=args.output_dir / "sbc0_predictions.jsonl",
        )
        evaluate_args = argparse.Namespace(
            samples=build_args.output,
            predictions=classify_args.output,
            protocol=args.protocol,
            output=args.output_dir / "sbc0_evaluation.json",
        )
        _cmd_build_samples(build_args)
        _cmd_classify(classify_args)
        _cmd_evaluate(evaluate_args)
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
