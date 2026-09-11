"""Tests for Stage 4-new BR-2 asymmetric local-evidence boundary refinement."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from aic_video_highlight.highlight_retrieval.br2_asymmetric_boundary import (
    apply_asymmetric_boundary_action,
    decide_left_boundary_action,
    decide_right_boundary_action,
    propose_br2_boundary,
    run_br2_for_role,
    _side_decision,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_br2_asymmetric_boundary.py"

CONFIG = {
    "boundary_window_sec": 4.0,
    "sample_stride_sec": 0.5,
    "persistence_k": 3,
    "action_confidence_min": 0.70,
    "trim_margin_sec": 0.5,
    "expand_margin_sec": 0.5,
    "max_trim_each_side_sec": 2.0,
    "max_expand_each_side_sec": 1.0,
    "max_total_boundary_change_ratio": 0.20,
    "min_parent_overlap_ratio": 0.80,
    "min_refined_duration_sec": 5.0,
}


def _samples(side: str, start: float, end: float, rows: list[tuple[float, bool, float]]) -> list[dict]:
    return [
        {"t": t, "side": side, "inside": inside, "core_similarity_histogram": sim}
        for t, inside, sim in rows
    ]


def test_keep_keep_yields_identity() -> None:
    left = {"action": "KEEP", "confidence": 0.0, "supported": 0, "num_samples": 10}
    right = {"action": "KEEP", "confidence": 0.0, "supported": 0, "num_samples": 10}
    result = apply_asymmetric_boundary_action(10.0, 30.0, left, right, CONFIG, video_id="v", candidate_id="c")
    assert result["decision"] == "IDENTITY"
    assert result["refined_start_sec"] == 10.0
    assert result["refined_end_sec"] == 30.0


def test_left_trim_does_not_touch_right() -> None:
    left = {"action": "TRIM", "confidence": 0.9, "supported": 4, "num_samples": 10}
    right = {"action": "KEEP", "confidence": 0.0, "supported": 0, "num_samples": 10}
    result = apply_asymmetric_boundary_action(10.0, 30.0, left, right, CONFIG, video_id="v", candidate_id="c")
    assert result["decision"] == "REFINE"
    detail = result["decision_detail"]
    assert detail["left_action"] == "TRIM"
    assert detail["right_action"] == "KEEP"
    assert result["refined_start_sec"] > 10.0
    assert result["refined_end_sec"] == 30.0


def test_right_trim_does_not_touch_left() -> None:
    left = {"action": "KEEP", "confidence": 0.0, "supported": 0, "num_samples": 10}
    right = {"action": "TRIM", "confidence": 0.9, "supported": 4, "num_samples": 10}
    result = apply_asymmetric_boundary_action(10.0, 30.0, left, right, CONFIG, video_id="v", candidate_id="c")
    assert result["decision"] == "REFINE"
    assert result["refined_start_sec"] == 10.0
    assert result["refined_end_sec"] < 30.0


def test_expand_respects_per_side_cap() -> None:
    left = {"action": "EXPAND", "confidence": 0.95, "supported": 20, "num_samples": 30}
    right = {"action": "KEEP", "confidence": 0.0, "supported": 0, "num_samples": 10}
    result = apply_asymmetric_boundary_action(10.0, 30.0, left, right, CONFIG, video_id="v", candidate_id="c")
    if result["decision"] == "REFINE":
        assert (10.0 - result["refined_start_sec"]) <= CONFIG["max_expand_each_side_sec"] + 1e-9
    else:
        assert result["decision"] == "IDENTITY_FALLBACK"


def test_total_change_ratio_cap_enforced() -> None:
    left = {"action": "TRIM", "confidence": 0.95, "supported": 10, "num_samples": 20}
    right = {"action": "TRIM", "confidence": 0.95, "supported": 10, "num_samples": 20}
    result = apply_asymmetric_boundary_action(10.0, 30.0, left, right, CONFIG, video_id="v", candidate_id="c")
    if result["decision"] == "REFINE":
        original = 20.0
        change = abs(result["decision_detail"]["left_delta_sec"]) + abs(
            result["decision_detail"]["right_delta_sec"]
        )
        assert change / original <= CONFIG["max_total_boundary_change_ratio"] + 1e-9
    else:
        assert result["decision"] == "IDENTITY_FALLBACK"


def test_min_refined_duration_enforced() -> None:
    left = {"action": "TRIM", "confidence": 0.95, "supported": 10, "num_samples": 20}
    right = {"action": "TRIM", "confidence": 0.95, "supported": 10, "num_samples": 20}
    short = {**CONFIG, "min_refined_duration_sec": 19.0}
    result = apply_asymmetric_boundary_action(10.0, 30.0, left, right, short, video_id="v", candidate_id="c")
    assert result["decision"] == "IDENTITY_FALLBACK"


def test_low_confidence_becomes_keep_and_identity() -> None:
    left = {"action": "TRIM", "confidence": 0.5, "supported": 4, "num_samples": 10}
    right = {"action": "KEEP", "confidence": 0.0, "supported": 0, "num_samples": 10}
    result = apply_asymmetric_boundary_action(10.0, 30.0, left, right, CONFIG, video_id="v", candidate_id="c")
    assert result["decision"] == "IDENTITY"
    assert result["decision_detail"]["left_action"] == "KEEP"


def test_persistence_k_governs_action() -> None:
    samples = _samples(
        "left",
        10.0,
        30.0,
        [(10.25, True, 0.0), (10.75, True, 0.0), (11.25, True, 0.0)]
        + [(11.75 + i * 0.5, True, 0.9) for i in range(8)]
        + [(9.75 - i * 0.5, False, 0.5) for i in range(6)],
    )
    strong_config = {**CONFIG, "persistence_k": 3}
    decision = _side_decision(samples, True, strong_config)
    assert decision["action"] == "TRIM"
    weak_config = {**CONFIG, "persistence_k": 5}
    decision2 = _side_decision(samples, True, weak_config)
    assert decision2["action"] == "KEEP"


def test_constant_evidence_yields_keep() -> None:
    samples = _samples(
        "left",
        10.0,
        30.0,
        [(t, True, 0.0) for t in (10.25, 10.75, 11.25, 11.75, 12.25, 13.0, 13.5, 14.0)]
        + [(t, False, 0.0) for t in (9.75, 9.25, 8.75, 8.25, 7.75, 7.25)],
    )
    decision = _side_decision(samples, True, CONFIG)
    assert decision["action"] == "KEEP"


def test_refined_interval_stays_in_bounds() -> None:
    left = {"action": "EXPAND", "confidence": 0.95, "supported": 6, "num_samples": 20}
    right = {"action": "EXPAND", "confidence": 0.95, "supported": 6, "num_samples": 20}
    wide = {**CONFIG, "max_expand_each_side_sec": 12.0, "max_total_boundary_change_ratio": 1.0}
    result = apply_asymmetric_boundary_action(2.0, 8.0, left, right, wide, video_id="v", candidate_id="c")
    if result["decision"] == "REFINE":
        assert result["refined_start_sec"] >= 0.0
    else:
        assert result["decision"] == "IDENTITY_FALLBACK"


def test_no_nan_inf_and_no_leakage() -> None:
    left = {"action": "TRIM", "confidence": 0.9, "supported": 4, "num_samples": 10}
    right = {"action": "EXPAND", "confidence": 0.8, "supported": 4, "num_samples": 10}
    result = apply_asymmetric_boundary_action(10.0, 30.0, left, right, CONFIG, video_id="v", candidate_id="c")

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
    payload = json.dumps(result).lower()
    for forbidden in ("weak_reference", "audit", "human_label", "adjudication", "heldout"):
        assert forbidden not in payload


def test_cli_help_runs(capsys) -> None:
    spec = importlib.util.spec_from_file_location("run_br2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit) as excinfo:
        module.main(["propose", "--help"])
    assert excinfo.value.code == 0
    assert "config-name" in capsys.readouterr().out


def test_protocol_safety_flags() -> None:
    protocol = json.loads(
        (Path(__file__).resolve().parents[1] / "configs" / "stage4_br2_asymmetric_boundary_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["principles"]["no_model_calls"] is True
    assert protocol["principles"]["no_oracle_boundary_selection"] is True
    assert protocol["frozen_upstream"]["heldout_access"] is False
    assert protocol["frozen_upstream"]["qwen_calls_allowed"] is False
    assert protocol["frozen_upstream"]["vllm_calls_allowed"] is False
    assert protocol["frozen_upstream"]["training_allowed"] is False
    assert set(protocol["configs"]) == {"BR-2-C1", "BR-2-C2", "BR-2-C3"}


def test_build_result_via_run_br2_for_role(tmp_path: Path) -> None:
    candidate = {"merged_candidate_id": "cand-1", "start_sec": 10.0, "end_sec": 30.0, "score": 0.8, "reason": "r", "source_chunk": 0}
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
        rows, refinement = propose_br2_boundary(
            _FakeCapture(), cand, CONFIG, video_id=video_id, duration_sec=record["duration_sec"]
        )
        return rows, refinement

    result = run_br2_for_role(cache_manifest, role_manifest, cache_records, "BR-2-C1", CONFIG, propose_fn)
    assert result["refiner_name"] == "BR-2-C1"
    refinement = result["records"][0]["candidate_refinements"][0]
    assert refinement["decision"] in {"IDENTITY", "REFINE", "IDENTITY_FALLBACK"}
    assert json.dumps(result)


class _FakeCapture:
    """Deterministic fake capture: returns a fixed frame for every seek."""

    def isOpened(self):  # noqa: N802
        return True

    def set(self, prop, value):  # noqa: N803
        return True

    def read(self):
        import numpy as np

        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        return True, frame

    def release(self):
        return None
