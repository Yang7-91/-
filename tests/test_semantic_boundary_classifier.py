"""Tests for Stage 4-new SBC-0 semantic boundary classifier probe."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from aic_video_highlight.highlight_retrieval.semantic_boundary_classifier import (
    BoundarySideSample,
    SemanticBoundaryParseError,
    SemanticBoundaryPrediction,
    build_stratified_boundary_samples,
    compute_sbc0_metrics,
    load_oracle_boundary_labels,
    parse_semantic_boundary_response,
    render_boundary_prompt,
    build_boundary_clip_bounds,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_semantic_boundary_classifier.py"

PROTOCOL = {
    "schema_version": "aic.stage4.sbc0_semantic_boundary_classifier_probe/v1",
    "role": "dev_tune",
    "decision_rules": {
        "meaningful_auc_like_min": 0.65,
        "meaningful_macro_f1_min": 0.45,
        "schema_success_rate_min": 0.95,
    },
}


def _record(video_id: str, split: str, candidates: list[tuple[str, float, float]]) -> dict:
    return {
        "video_id": video_id,
        "split": split,
        "merged_candidates": [
            {"merged_candidate_id": cid, "start_sec": start, "end_sec": end}
            for cid, start, end in candidates
        ],
    }


def test_load_oracle_labels(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text(
        json.dumps({"video_id": "v1", "candidate_id": "c1", "left_action": "KEEP", "right_action": "TRIM"}) + "\n",
        encoding="utf-8",
    )
    rows = load_oracle_boundary_labels(path)
    assert rows[0]["left_action"] == "KEEP"


def test_stratified_sampling_deterministic_and_capped() -> None:
    labels = []
    records = {}
    for index in range(50):
        video_id = f"v{index:03d}"
        candidate_id = f"c{index}"
        records[video_id] = _record(video_id, "dev", [(candidate_id, 5.0, 15.0)])
        labels.append(
            {
                "video_id": video_id,
                "candidate_id": candidate_id,
                "left_action": "TRIM" if index % 2 == 0 else "KEEP",
                "right_action": "EXPAND",
            }
        )
    first, report1 = build_stratified_boundary_samples(
        labels, records, max_samples_total=180, max_per_label_per_side=10, seed=42
    )
    second, report2 = build_stratified_boundary_samples(
        labels, records, max_samples_total=180, max_per_label_per_side=10, seed=42
    )
    assert [s.sample_id for s in first] == [s.sample_id for s in second]
    assert report1["label_counts"]["EXPAND"] == 10  # right side capped
    assert report1["groups"]["left:TRIM"]["selected"] == 10
    assert report1["total_samples"] <= 180


def test_sampling_excludes_non_dev() -> None:
    labels = [
        {"video_id": "v1", "candidate_id": "c1", "left_action": "KEEP", "right_action": "KEEP"},
        {"video_id": "v2", "candidate_id": "c2", "left_action": "KEEP", "right_action": "KEEP"},
    ]
    records = {
        "v1": _record("v1", "dev", [("c1", 0.0, 5.0)]),
        "v2": _record("v2", "hard", [("c2", 0.0, 5.0)]),
    }
    samples, report = build_stratified_boundary_samples(
        labels, records, max_samples_total=10, max_per_label_per_side=10, seed=1
    )
    assert all(s.video_id == "v1" for s in samples)
    assert report["skipped"]["non_dev"] == 1
    assert report["heldout_accessed"] is False


def test_parse_semantic_boundary_response_valid() -> None:
    payload = parse_semantic_boundary_response('{"action": "TRIM", "confidence": 0.8, "rationale_short": "context"}')
    assert payload["action"] == "TRIM"
    assert payload["confidence"] == 0.8


def test_parse_semantic_boundary_response_rejects_invalid() -> None:
    with pytest.raises(SemanticBoundaryParseError):
        parse_semantic_boundary_response("not json")
    with pytest.raises(SemanticBoundaryParseError):
        parse_semantic_boundary_response('{"action": "MOVE", "confidence": 0.5}')
    with pytest.raises(SemanticBoundaryParseError):
        parse_semantic_boundary_response('{"action": "KEEP", "confidence": 3.0}')


def test_confusion_and_macro_f1_correct() -> None:
    samples = [
        BoundarySideSample(f"s{i}", "v", "c", "left", oracle, 0.0, 10.0, 0.0, 10.0)
        for i, oracle in enumerate(["TRIM", "TRIM", "KEEP", "EXPAND"])
    ]
    predictions = [
        SemanticBoundaryPrediction("s0", "TRIM", 0.9, "r", True),
        SemanticBoundaryPrediction("s1", "KEEP", 0.6, "r", True),
        SemanticBoundaryPrediction("s2", "KEEP", 0.7, "r", True),
        SemanticBoundaryPrediction("s3", "EXPAND", 0.8, "r", True),
    ]
    metrics = compute_sbc0_metrics(samples, predictions, PROTOCOL)
    assert metrics["confusion_matrix"]["TRIM"]["TRIM"] == 1
    assert metrics["confusion_matrix"]["TRIM"]["KEEP"] == 1
    assert metrics["accuracy"] == 0.75
    assert metrics["schema_success_rate"] == 1.0
    assert metrics["diagnostic_only"] is True
    assert metrics["deployable_method"] is False
    assert json.dumps(metrics)  # serializable
    assert 0.0 <= metrics["left"]["auc_like"] <= 1.0


def test_parse_failures_reduce_schema_success_rate() -> None:
    samples = [BoundarySideSample(f"s{i}", "v", "c", "right", "KEEP", 0.0, 10.0, 10.0, 10.0) for i in range(4)]
    predictions = [
        SemanticBoundaryPrediction("s0", "KEEP", 0.9, "r", True),
        SemanticBoundaryPrediction("s1", "KEEP", 0.9, "r", True),
        SemanticBoundaryPrediction("s2", None, None, None, False, "parse_error"),
        SemanticBoundaryPrediction("s3", None, None, None, False, "parse_error"),
    ]
    metrics = compute_sbc0_metrics(samples, predictions, PROTOCOL)
    assert metrics["schema_success_rate"] == 0.5
    assert metrics["decision"] == "NOT_ACTIONABLE"


def test_clip_bounds_clamped() -> None:
    sample = BoundarySideSample("s", "v", "c", "left", "KEEP", 0.0, 10.0, 1.0, 10.0)
    bounds = build_boundary_clip_bounds(sample, duration_sec=3.0, window_sec=4.0)
    assert bounds["clip_start_sec"] == 0.0
    assert bounds["clip_end_sec"] == 3.0
    assert bounds["boundary_offset_in_clip_sec"] == 1.0


def test_render_prompt_fills_placeholders() -> None:
    sample = BoundarySideSample("s", "v", "c", "right", "KEEP", 0.0, 10.0, 10.0, 10.0)
    bounds = build_boundary_clip_bounds(sample, duration_sec=20.0, window_sec=4.0)
    template = "side={side} offset={boundary_offset_in_clip_sec} dir={side_direction}"
    rendered = render_boundary_prompt(template, sample, bounds)
    assert "side=right" in rendered
    assert "{side" not in rendered
    assert "RIGHT boundary" in rendered


def test_actionable_decision_requires_all_rules() -> None:
    samples = []
    predictions = []
    for index in range(12):
        oracle = ["TRIM", "KEEP", "EXPAND"][index % 3]
        samples.append(BoundarySideSample(f"s{index}", "v", "c", "left", oracle, 0.0, 10.0, 0.0, 10.0))
        predictions.append(SemanticBoundaryPrediction(f"s{index}", oracle, 0.95, "r", True))
    metrics = compute_sbc0_metrics(samples, predictions, PROTOCOL)
    assert metrics["accuracy"] == 1.0
    assert metrics["rule_checks"]["schema_success_rate_pass"] is True
    assert metrics["rule_checks"]["macro_f1_pass"] is True
    assert metrics["decision"] in {"ACTIONABLE_SIGNAL", "NOT_ACTIONABLE"}


def test_cli_help_runs(capsys) -> None:
    spec = importlib.util.spec_from_file_location("run_sbc0", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit) as excinfo:
        module.main(["build-samples", "--help"])
    assert excinfo.value.code == 0
    assert "oracle-labels" in capsys.readouterr().out


def test_protocol_flags() -> None:
    protocol = json.loads(
        (Path(__file__).resolve().parents[1] / "configs" / "stage4_sbc0_semantic_boundary_classifier_probe.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["purpose"] == "diagnostic_only_not_deployable"
    assert protocol["frozen_upstream"]["heldout_access"] is False
    assert protocol["frozen_upstream"]["hard_access"] is False
    assert protocol["safety"]["does_not_create_deployable_method"] is True
    assert protocol["safety"]["does_not_modify_candidates"] is True
    assert protocol["prompt"]["no_iterative_prompt_tuning"] is True
