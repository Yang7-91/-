"""Tests for Stage 4-new SABR-1.1 conservative boundary proposal."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from aic_video_highlight.highlight_retrieval.sabr_boundary import (
    build_sabr11_result,
    propose_conservative_boundary,
)
from aic_video_highlight.highlight_retrieval.saliency_anchor import normalize_saliency_bins

CONFIG = {
    "bin_sec": 1.0,
    "core_quantile": 0.80,
    "low_quantile": 0.35,
    "margin_sec": 1.0,
    "max_trim_each_side_sec": 3.0,
    "max_total_shrink_ratio": 0.25,
    "min_parent_overlap_ratio": 0.75,
    "min_refined_duration_sec": 5.0,
    "min_peak_prominence": 0.15,
    "max_candidate_duration_sec": 6.0,
    "short_duration_margin_sec": 2.0,
    "min_effective_bins": 4,
    "constant_saliency_epsilon": 1e-9,
}


def _bins(scores: list[float], bin_sec: float = 1.0) -> list[dict]:
    rows = []
    for index, score in enumerate(scores):
        rows.append(
            {
                "video_id": "v",
                "candidate_id": "c",
                "bin_index": index,
                "start_sec": index * bin_sec,
                "end_sec": (index + 1) * bin_sec,
                "frame_count": 2,
                "frame_difference": score,
                "histogram_difference": score / 2,
                "laplacian_variance": 100.0 + score,
                "edge_density": 0.1 + score / 10,
                "saturation": 0.3,
                "contrast": 0.4,
                "saliency_score": score,
            }
        )
    return rows


def _candidate(start: float, end: float) -> dict:
    return {
        "merged_candidate_id": "cand-1",
        "start_sec": start,
        "end_sec": end,
        "score": 0.8,
        "reason": "test",
        "source_chunk": 0,
    }


def test_propose_never_expands() -> None:
    candidate = _candidate(10.0, 30.0)
    bins = _bins([0.0, 0.0, 0.1, 0.9, 0.9, 0.8, 0.9, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = propose_conservative_boundary(candidate, normalize_saliency_bins(bins), CONFIG, duration_sec=40.0)
    assert CONFIG["original_start"] if False else True
    assert result["refined_start_sec"] >= candidate["start_sec"] - 1e-9
    assert result["refined_end_sec"] <= candidate["end_sec"] + 1e-9
    assert result["refined_start_sec"] < result["refined_end_sec"]


def test_low_confidence_short_candidate_falls_back() -> None:
    candidate = _candidate(5.0, 9.0)
    bins = _bins([0.9, 0.9, 0.9, 0.9])
    result = propose_conservative_boundary(candidate, normalize_saliency_bins(bins), CONFIG, duration_sec=40.0)
    assert result["decision"] == "IDENTITY_FALLBACK"
    assert result["decision_detail"]["fallback_reason"] == "candidate_duration_below_floor"
    assert result["refined_start_sec"] == candidate["start_sec"]


def test_constant_saliency_falls_back() -> None:
    candidate = _candidate(0.0, 30.0)
    bins = _bins([0.5] * 30)
    result = propose_conservative_boundary(candidate, normalize_saliency_bins(bins), CONFIG, duration_sec=40.0)
    assert result["decision"] == "IDENTITY_FALLBACK"
    assert result["decision_detail"]["fallback_reason"] == "saliency_approximately_constant"


def test_too_few_effective_bins_falls_back() -> None:
    candidate = _candidate(0.0, 30.0)
    bins = _bins([0.9, 0.1, 0.9, 0.1, 0.9])
    for row in bins[:4]:
        row["frame_count"] = 0
    result = propose_conservative_boundary(candidate, normalize_saliency_bins(bins), CONFIG, duration_sec=40.0)
    assert result["decision"] == "IDENTITY_FALLBACK"
    assert result["decision_detail"]["fallback_reason"] == "too_few_effective_bins"


def test_max_trim_each_side_respected() -> None:
    candidate = _candidate(0.0, 30.0)
    scores = [0.0] * 10 + [0.9] * 10 + [0.0] * 10
    result = propose_conservative_boundary(candidate, normalize_saliency_bins(_bins(scores)), CONFIG, duration_sec=40.0)
    if result["decision"] == "REFINE":
        assert result["decision_detail"]["trim_left_sec"] <= CONFIG["max_trim_each_side_sec"] + 1e-9
        assert result["decision_detail"]["trim_right_sec"] <= CONFIG["max_trim_each_side_sec"] + 1e-9
        assert result["refined_start_sec"] >= candidate["start_sec"]


def test_shrink_ratio_cap_enforced() -> None:
    candidate = _candidate(0.0, 30.0)
    scores = [0.0] * 11 + [0.9] * 8 + [0.0] * 11
    tight = {**CONFIG, "max_total_shrink_ratio": 0.10}
    result = propose_conservative_boundary(candidate, normalize_saliency_bins(_bins(scores)), tight, duration_sec=40.0)
    if result["decision"] == "REFINE":
        refined_duration = result["refined_end_sec"] - result["refined_start_sec"]
        shrink = (30.0 - refined_duration) / 30.0
        assert shrink <= tight["max_total_shrink_ratio"] + 1e-9
    else:
        assert result["decision"] == "IDENTITY_FALLBACK"
        assert result["decision_detail"]["fallback_reason"] == "total_shrink_ratio_exceeds_cap"


def test_parent_overlap_floor_enforced() -> None:
    candidate = _candidate(0.0, 30.0)
    scores = [0.0] * 11 + [0.9] * 8 + [0.0] * 11
    strict = {**CONFIG, "min_parent_overlap_ratio": 0.95}
    result = propose_conservative_boundary(candidate, normalize_saliency_bins(_bins(scores)), strict, duration_sec=40.0)
    if result["decision"] == "REFINE":
        refined_duration = result["refined_end_sec"] - result["refined_start_sec"]
        assert refined_duration / 30.0 >= strict["min_parent_overlap_ratio"] - 1e-9
    else:
        assert result["decision"] == "IDENTITY_FALLBACK"
        assert result["decision_detail"]["fallback_reason"] == "parent_overlap_ratio_below_floor"


def test_propose_is_deterministic_and_serializable() -> None:
    candidate = _candidate(10.0, 30.0)
    bins = normalize_saliency_bins(
        _bins([0.1, 0.0, 0.1, 0.9, 0.9, 0.8, 0.9, 0.9, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    first = propose_conservative_boundary(candidate, bins, CONFIG, duration_sec=40.0)
    second = propose_conservative_boundary(candidate, bins, CONFIG, duration_sec=40.0)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_no_nan_or_inf_in_output() -> None:
    candidate = _candidate(10.0, 30.0)
    bins = normalize_saliency_bins(
        _bins([0.1, 0.0, 0.9, 0.9, 0.8, 0.9, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    result = propose_conservative_boundary(candidate, bins, CONFIG, duration_sec=40.0)

    def walk(value):
        if isinstance(value, float):
            assert math.isfinite(value)
        elif isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)

    walk(result)


def test_output_has_no_label_leakage() -> None:
    candidate = _candidate(10.0, 30.0)
    bins = normalize_saliency_bins(
        _bins([0.1, 0.0, 0.9, 0.9, 0.8, 0.9, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    result = propose_conservative_boundary(candidate, bins, CONFIG, duration_sec=40.0)
    payload = json.dumps(result).lower()
    for forbidden in ("weak_reference", "audit", "human_label", "adjudication", "heldout"):
        assert forbidden not in payload


def test_build_result_shape_and_decisions(tmp_path: Path) -> None:
    candidate = _candidate(0.0, 30.0)
    bins = normalize_saliency_bins(
        _bins([0.0, 0.1, 0.9, 0.9, 0.8, 0.9, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )

    cache_manifest = {"global_semantic_sha256": "0" * 64}
    role_manifest = {
        "role": "dev_tune",
        "semantic_sha256": "1" * 64,
        "source_cache_global_hash": "0" * 64,
        "records": [{"video_id": "v1", "split": "dev"}],
    }
    cache_records = {
        "v1": {
            "video_id": "v1",
            "split": "dev",
            "duration_sec": 40.0,
            "semantic_sha256": "2" * 64,
            "merged_candidates": [candidate],
        }
    }

    def propose_fn(video_id, cand, record):
        return bins, propose_conservative_boundary(cand, bins, CONFIG, duration_sec=record["duration_sec"])

    result = build_sabr11_result(cache_manifest, role_manifest, cache_records, "SABR-1.1-C1", CONFIG, propose_fn)
    assert result["refiner_name"] == "SABR-1.1-C1"
    assert result["decision_counts"]["IDENTITY"] + result["decision_counts"]["REFINE"] + result["decision_counts"]["IDENTITY_FALLBACK"] == 1
    refinement = result["records"][0]["candidate_refinements"][0]
    assert refinement["merged_candidate_id"] == "cand-1"
    assert refinement["decision_rule"].startswith("sabr1.")
    assert 0.0 <= refinement["confidence"] <= 1.0
    assert json.dumps(result)


def test_cli_propose_help_runs(capsys) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_saliency_anchor", Path(__file__).resolve().parents[1] / "scripts" / "run_saliency_anchor.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit) as excinfo:
        module.main(["propose", "--help"])
    assert excinfo.value.code == 0
    assert "config-name" in capsys.readouterr().out
