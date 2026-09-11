"""Tests for Stage 4-new BHD-0 boundary headroom diagnostics."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from aic_video_highlight.highlight_retrieval import boundary_headroom_diagnostic as bhd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_boundary_headroom_diagnostic.py"


def test_oracle_markers_present_and_non_deployable() -> None:
    candidate = {"merged_candidate_id": "c", "start_sec": 5.0, "end_sec": 15.0}
    result = bhd.run_local_boundary_oracle_for_candidate(
        candidate, [(6.0, 9.0)], max_shift_sec=2.0, step_sec=0.5
    )
    assert result["oracle_markers"]["oracle_diagnostic_only"] is True
    assert result["oracle_markers"]["non_deployable"] is True
    assert result["oracle_markers"]["uses_weak_reference"] is True


def test_oracle_does_not_emit_refined_candidate_payload() -> None:
    candidate = {"merged_candidate_id": "c", "start_sec": 5.0, "end_sec": 15.0}
    result = bhd.run_local_boundary_oracle_for_candidate(
        candidate, [(6.0, 9.0)], max_shift_sec=1.0, step_sec=0.5
    )
    payload = json.dumps(result)
    assert "refined_candidates" not in payload
    assert "replayed_predictions" not in payload


def test_oracle_search_finite_outputs() -> None:
    candidate = {"merged_candidate_id": "c", "start_sec": 5.0, "end_sec": 15.0}
    result = bhd.run_local_boundary_oracle_for_candidate(
        candidate, [(6.0, 9.0)], max_shift_sec=3.0, step_sec=0.5
    )

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


def test_shot_detection_deterministic(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    path = tmp_path / "shot_test.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (64, 48))
    for index in range(16):
        if index < 8:
            frame = np.full((48, 64, 3), 30, dtype=np.uint8)
        else:
            frame = np.full((48, 64, 3), 220, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    def factory(p):
        return cv2.VideoCapture(str(p), cv2.CAP_FFMPEG)

    first = bhd.detect_shot_boundaries(path, factory, frame_stride_sec=0.5, min_shot_len_sec=1.0, duration_sec=4.0)
    second = bhd.detect_shot_boundaries(path, factory, frame_stride_sec=0.5, min_shot_len_sec=1.0, duration_sec=4.0)
    assert first == second


def test_shot_segments_non_overlapping() -> None:
    segments = bhd.shot_segments_from_boundaries([2.0, 5.0, 5.5, 9.0], 12.0)
    for (_, end_a), (start_b, _) in zip(segments, segments[1:]):
        assert end_a <= start_b + 1e-9
    assert segments[0][0] == 0.0
    assert segments[-1][1] == 12.0


def test_subshot_oracle_marked_non_deployable() -> None:
    candidate = {"merged_candidate_id": "c", "start_sec": 0.0, "end_sec": 10.0}
    result = bhd.oracle_subshot_split_upper_bound(
        candidate, [(0.0, 2.0), (6.0, 9.0)], [(0.0, 3.0), (3.0, 6.0), (6.0, 10.0)], max_components=2
    )
    assert result["oracle_markers"]["non_deployable"] is True
    assert result["oracle_markers"]["uses_weak_reference"] is True


def test_signal_separability_summary_serializable() -> None:
    auc = bhd.rank_auc_like([0.9, 0.8, 0.85], [0.1, 0.2, 0.15])
    payload = {
        "auc_like": auc,
        "mean_gap": bhd.mean_gap([0.9, 0.8], [0.1, 0.2]),
        "effect_size": bhd.effect_size([0.9, 0.8], [0.1, 0.2]),
    }
    assert json.dumps(payload)
    assert auc > 0.9


def test_rank_auc_like_no_data_defaults() -> None:
    assert bhd.rank_auc_like([], [0.1]) == 0.5
    assert bhd.mean_gap([], []) == 0.0


def test_headroom_rules() -> None:
    rules = {
        "meaningful_oracle_precision_delta": 0.01,
        "meaningful_oracle_recall_delta_min": -0.01,
        "meaningful_oracle_f1_or_tiou_delta": 0.005,
    }
    assert bhd.has_meaningful_headroom(0.02, 0.0, 0.006, 0.0, rules) is True
    assert bhd.has_meaningful_headroom(0.003, -0.001, 0.001, 0.0, rules) is False
    assert bhd.has_meaningful_headroom(0.02, -0.05, 0.01, 0.0, rules) is False


def test_cli_help_runs(capsys) -> None:
    spec = importlib.util.spec_from_file_location("run_bhd0", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit) as excinfo:
        module.main(["diagnose", "--help"])
    assert excinfo.value.code == 0
    assert "cache-dir" in capsys.readouterr().out


def test_protocol_marks_diagnostic_only() -> None:
    protocol = json.loads(
        (Path(__file__).resolve().parents[1] / "configs" / "stage4_bhd0_boundary_headroom_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["purpose"] == "diagnostic_only_not_deployable"
    assert protocol["oracle_diagnostics"]["uses_weak_reference"] is True
    assert protocol["oracle_diagnostics"]["deployable"] is False
    assert protocol["safety"]["no_hard"] is True
    assert protocol["safety"]["no_heldout"] is True
    assert protocol["safety"]["does_not_create_deployable_method"] is True
    assert protocol["frozen_upstream"]["qwen_calls_allowed"] is False
    assert protocol["frozen_upstream"]["vllm_calls_allowed"] is False
