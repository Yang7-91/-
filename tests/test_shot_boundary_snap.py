"""Tests for Stage 4-new SBA-1 conservative shot-boundary snap."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from aic_video_highlight.highlight_retrieval.shot_boundary_snap import (
    find_nearest_shot_boundary,
    propose_shot_boundary_snap,
    run_sba1_for_role,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_shot_boundary_snap.py"

BASE_CONFIG = {
    "snap_window_sec": 2.0,
    "allow_trim": True,
    "allow_expand": True,
    "prefer_nearest": True,
    "max_trim_each_side_sec": 2.0,
    "max_expand_each_side_sec": 1.5,
    "max_total_boundary_change_ratio": 0.20,
    "min_parent_overlap_ratio": 0.80,
    "min_refined_duration_sec": 5.0,
    "min_candidate_duration_sec": 4.0,
}


def _candidate(start: float, end: float) -> dict:
    return {
        "merged_candidate_id": "cand-1",
        "start_sec": start,
        "end_sec": end,
        "score": 0.8,
        "reason": "r",
        "source_chunk": 0,
    }


def _propose(candidate: dict, shots: list[float], config: dict | None = None, duration: float = 60.0):
    config = {**BASE_CONFIG, **(config or {})}
    return propose_shot_boundary_snap(
        candidate, shots, config, duration_sec=duration, video_id="v", candidate_id="cand-1"
    )


def test_nearest_selection_deterministic() -> None:
    decision = find_nearest_shot_boundary(10.0, [9.0, 10.5, 11.2], 2.0, "left", BASE_CONFIG)
    assert decision.target_sec == 10.5
    assert decision.action == "TRIM"
    again = find_nearest_shot_boundary(10.0, [9.0, 10.5, 11.2], 2.0, "left", BASE_CONFIG)
    assert (decision.target_sec, decision.action) == (again.target_sec, again.action)


def test_no_boundary_in_window_keeps() -> None:
    detail, refinement = _propose(_candidate(10.0, 20.0), [0.5, 50.0])
    assert refinement["decision"] == "IDENTITY"
    assert detail["left_action"] == "KEEP"
    assert detail["right_action"] == "KEEP"


def test_allow_expand_false_blocks_outward_snap() -> None:
    config = {**BASE_CONFIG, "allow_expand": False, "inward_only": True, "max_expand_each_side_sec": 0.0}
    decision = find_nearest_shot_boundary(10.0, [9.0], 2.0, "left", config)
    assert decision.action == "KEEP"  # outward blocked
    decision2 = find_nearest_shot_boundary(10.0, [10.5], 2.0, "left", config)
    assert decision2.action == "TRIM"


def test_prefer_inner_when_tie() -> None:
    tie_inner = {**BASE_CONFIG, "prefer_inner_when_tie": True}
    decision = find_nearest_shot_boundary(10.0, [9.5, 10.5], 2.0, "left", tie_inner)
    assert decision.target_sec == 10.5  # inward for left
    tie_outer = {**BASE_CONFIG, "prefer_inner_when_tie": False}
    decision2 = find_nearest_shot_boundary(10.0, [9.5, 10.5], 2.0, "left", tie_outer)
    assert decision2.target_sec == 9.5


def test_left_actions_signs() -> None:
    _, refinement = _propose(_candidate(10.0, 20.0), [10.8])
    assert refinement["decision_detail"]["left_action"] == "TRIM"
    assert refinement["refined_start_sec"] > 10.0
    _, refinement2 = _propose(_candidate(10.0, 20.0), [9.4])
    assert refinement2["decision_detail"]["left_action"] == "EXPAND"
    assert refinement2["refined_start_sec"] < 10.0


def test_right_actions_signs() -> None:
    _, refinement = _propose(_candidate(10.0, 20.0), [19.4])
    assert refinement["decision_detail"]["right_action"] == "TRIM"
    assert refinement["refined_end_sec"] < 20.0
    _, refinement2 = _propose(_candidate(10.0, 20.0), [20.6])
    assert refinement2["decision_detail"]["right_action"] == "EXPAND"
    assert refinement2["refined_end_sec"] > 20.0


def test_max_trim_cap_falls_back() -> None:
    # shot at distance 1.5 but trim cap 1.0 -> fallback identity
    _, refinement = _propose(_candidate(10.0, 20.0), [11.5], {"max_trim_each_side_sec": 1.0})
    assert refinement["decision"] == "IDENTITY_FALLBACK"
    assert refinement["decision_detail"]["fallback_reason"] == "trim_exceeds_cap"


def test_max_expand_cap_falls_back() -> None:
    _, refinement = _propose(_candidate(10.0, 20.0), [8.6], {"max_expand_each_side_sec": 1.0})
    assert refinement["decision"] == "IDENTITY_FALLBACK"
    assert refinement["decision_detail"]["fallback_reason"] == "expand_exceeds_cap"


def test_total_change_ratio_cap_falls_back() -> None:
    _, refinement = _propose(
        _candidate(10.0, 20.0),
        [8.8, 21.2],
        {"max_total_boundary_change_ratio": 0.10},
    )
    assert refinement["decision"] == "IDENTITY_FALLBACK"
    assert refinement["decision_detail"]["fallback_reason"] == "total_boundary_change_ratio_exceeds_cap"


def test_parent_overlap_floor_falls_back() -> None:
    _, refinement = _propose(
        _candidate(10.0, 16.0),
        [12.0],
        {"min_parent_overlap_ratio": 0.90, "max_trim_each_side_sec": 2.0},
    )
    assert refinement["decision"] == "IDENTITY_FALLBACK"
    assert refinement["decision_detail"]["fallback_reason"] in {
        "parent_overlap_below_floor",
        "refined_duration_below_minimum",
    }


def test_min_refined_duration_falls_back() -> None:
    _, refinement = _propose(
        _candidate(10.0, 15.0),
        [12.0],
        {"min_refined_duration_sec": 4.0, "max_trim_each_side_sec": 2.0},
    )
    assert refinement["decision"] == "IDENTITY_FALLBACK"


def test_candidate_too_short_falls_back() -> None:
    _, refinement = _propose(_candidate(10.0, 13.0), [11.0], {"min_candidate_duration_sec": 6.0})
    assert refinement["decision"] == "IDENTITY_FALLBACK"
    assert refinement["decision_detail"]["fallback_reason"] == "candidate_below_min_duration"


def test_refined_interval_stays_in_bounds() -> None:
    _, refinement = _propose(_candidate(0.5, 8.0), [0.0, 8.5], {"snap_window_sec": 1.0})
    if refinement["decision"] == "REFINE":
        assert refinement["refined_start_sec"] >= 0.0
        assert refinement["refined_end_sec"] <= 60.0
    else:
        assert refinement["decision"] == "IDENTITY_FALLBACK"


def test_no_nan_inf_and_no_leakage() -> None:
    detail, refinement = _propose(_candidate(10.0, 20.0), [9.2, 20.8])
    payload = json.dumps({"detail": detail, "refinement": refinement})

    def walk(value):
        if isinstance(value, float):
            assert math.isfinite(value)
        elif isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)

    walk({"detail": detail, "refinement": refinement})
    lowered = payload.lower()
    for forbidden in ("weak_reference", "audit", "human", "adjudication", "oracle_label", "heldout"):
        assert forbidden not in lowered


def test_build_result_via_run_sba1_for_role() -> None:
    candidate = _candidate(10.0, 20.0)
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
            "duration_sec": 60.0,
            "semantic_sha256": "2" * 64,
            "merged_candidates": [candidate],
        }
    }
    result = run_sba1_for_role(
        cache_manifest, role_manifest, cache_records, "SBA-1-C1", BASE_CONFIG, {"v1": [10.6]}
    )
    refinement = result["records"][0]["candidate_refinements"][0]
    assert refinement["decision"] in {"REFINE", "IDENTITY", "IDENTITY_FALLBACK"}
    assert refinement["decision_rule"].startswith("sba1.")
    assert json.dumps(result)


def test_cli_help_runs(capsys) -> None:
    spec = importlib.util.spec_from_file_location("run_sba1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit) as excinfo:
        module.main(["propose", "--help"])
    assert excinfo.value.code == 0
    assert "config-name" in capsys.readouterr().out


def test_protocol_deployable_flags() -> None:
    protocol = json.loads(
        (Path(__file__).resolve().parents[1] / "configs" / "stage4_sba1_shot_boundary_snap_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["principles"]["deployable"] is True
    assert protocol["principles"]["uses_weak_reference_for_decision"] is False
    assert protocol["principles"]["no_model_calls"] is True
    assert protocol["frozen_upstream"]["qwen_calls_allowed"] is False
    assert protocol["frozen_upstream"]["vllm_calls_allowed"] is False
    assert protocol["frozen_upstream"]["heldout_access"] is False
    assert protocol["frozen_upstream"]["training_allowed"] is False
    assert protocol["safety"]["does_not_modify_cache"] is True
    assert set(protocol["configs"]) == {"SBA-1-C1", "SBA-1-C2", "SBA-1-C3"}
