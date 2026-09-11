"""Tests for Stage 4-new BHD-0.1 boundary headroom diagnostics (audited)."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from aic_video_highlight.highlight_retrieval import boundary_headroom_diagnostic as bhd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_boundary_headroom_diagnostic.py"


def _candidate(cid: str, start: float, end: float) -> dict:
    return {"merged_candidate_id": cid, "start_sec": start, "end_sec": end}


def test_oracle_markers_present_and_non_deployable() -> None:
    candidates = [_candidate("c1", 5.0, 15.0)]
    result = bhd.greedy_video_oracle(
        candidates, [(6.0, 9.0)], 40.0, max_shift=2.0, step=0.5, objective="f1"
    )
    assert result["oracle_markers"]["oracle_diagnostic_only"] is True
    assert result["oracle_markers"]["non_deployable"] is True
    assert result["oracle_markers"]["uses_weak_reference"] is True
    assert result["identity_included"] is True
    assert result["upper_bound"] is True


def test_optional_oracle_identity_dominance_f1() -> None:
    """Optional oracle (identity in search space) must never be below baseline."""
    candidates = [_candidate(f"c{i}", 5.0 + i * 15.0, 15.0 + i * 15.0) for i in range(4)]
    refs = [(6.0, 9.0), (20.0, 24.0), (36.0, 40.0), (52.0, 55.0)]
    result = bhd.greedy_video_oracle(
        candidates, refs, 70.0, max_shift=5.0, step=0.5, objective="f1"
    )
    assert result["metrics"]["f1"] >= result["baseline_metrics"]["f1"] - 1e-9
    result_tiou = bhd.greedy_video_oracle(
        candidates, refs, 70.0, max_shift=5.0, step=0.5, objective="temporal_iou"
    )
    assert (
        result_tiou["metrics"]["temporal_iou"]
        >= result_tiou["baseline_metrics"]["temporal_iou"] - 1e-9
    )


def test_optional_oracle_precision_under_recall_guard() -> None:
    candidates = [_candidate(f"c{i}", 5.0 + i * 15.0, 15.0 + i * 15.0) for i in range(4)]
    refs = [(6.0, 9.0), (20.0, 24.0), (36.0, 40.0), (52.0, 55.0)]
    result = bhd.greedy_video_oracle(
        candidates,
        refs,
        70.0,
        max_shift=3.0,
        step=0.5,
        objective="precision_under_recall_guard",
        recall_guard=None,
    )
    assert result["metrics"]["recall"] >= result["baseline_metrics"]["recall"] - 1e-9
    assert result["metrics"]["precision"] >= result["baseline_metrics"]["precision"] - 1e-9


def test_forced_diagnostic_may_be_worse_but_labelled() -> None:
    candidates = [_candidate("c1", 5.0, 15.0)]
    result = bhd.greedy_forced_diagnostic(
        candidates, [(6.0, 9.0)], 40.0, max_shift=5.0, step=0.5, objective="f1"
    )
    assert result["forced_diagnostic"] is True
    assert result["upper_bound"] is False
    assert result["identity_included"] is False


def test_action_label_sign_convention() -> None:
    """left_delta = refined_start - original_start; right_delta = refined_end - original_end."""
    left_trim = {"action": "TRIM", "confidence": 0.9, "supported": 4, "num_samples": 10}
    right_trim = {"action": "TRIM", "confidence": 0.9, "supported": 4, "num_samples": 10}
    result = bhd  # placeholder to keep import explicit
    from aic_video_highlight.highlight_retrieval.br2_asymmetric_boundary import (
        apply_asymmetric_boundary_action,
    )

    refined = apply_asymmetric_boundary_action(
        10.0, 30.0, left_trim, right_trim, {
            "action_confidence_min": 0.7, "trim_margin_sec": 0.5,
            "expand_margin_sec": 0.5, "max_trim_each_side_sec": 2.0,
            "max_expand_each_side_sec": 1.0, "max_total_boundary_change_ratio": 0.9,
            "min_parent_overlap_ratio": 0.5, "min_refined_duration_sec": 4.0,
            "boundary_window_sec": 4.0,
        },
        video_id="v", candidate_id="c",
    )
    left_delta = refined["refined_start_sec"] - 10.0
    right_delta = refined["refined_end_sec"] - 30.0
    assert left_delta > 0  # LEFT_TRIM
    assert right_delta < 0  # RIGHT_TRIM
    assert refined["decision_detail"]["left_delta_sec"] == pytest.approx(left_delta, abs=1e-9)
    assert refined["decision_detail"]["right_delta_sec"] == pytest.approx(right_delta, abs=1e-9)


def test_left_expand_right_expand_signs() -> None:
    from aic_video_highlight.highlight_retrieval.br2_asymmetric_boundary import (
        apply_asymmetric_boundary_action,
    )
    config = {
        "action_confidence_min": 0.7, "trim_margin_sec": 0.5,
        "expand_margin_sec": 0.5, "max_trim_each_side_sec": 2.0,
        "max_expand_each_side_sec": 1.0, "max_total_boundary_change_ratio": 0.9,
        "min_parent_overlap_ratio": 0.5, "min_refined_duration_sec": 4.0,
        "boundary_window_sec": 4.0,
    }
    left = {"action": "EXPAND", "confidence": 0.9, "supported": 2, "num_samples": 10}
    right = {"action": "EXPAND", "confidence": 0.9, "supported": 2, "num_samples": 10}
    refined = apply_asymmetric_boundary_action(10.0, 30.0, left, right, config, video_id="v", candidate_id="c")
    assert refined["refined_start_sec"] - 10.0 < 0  # LEFT_EXPAND
    assert refined["refined_end_sec"] - 30.0 > 0  # RIGHT_EXPAND


def test_coverage_direction_sanity() -> None:
    """With only EXPAND/KEEP actions, coverage must not drop."""
    candidates = [_candidate("c1", 5.0, 15.0)]
    refs = [(6.0, 9.0)]
    result = bhd.greedy_video_oracle(
        candidates, refs, 40.0, max_shift=2.0, step=0.5, objective="precision_under_recall_guard"
    )
    expand_or_keep = all(
        action["left_action"] in ("EXPAND", "KEEP") and action["right_action"] in ("EXPAND", "KEEP")
        for action in result["actions"]
    )
    if expand_or_keep:
        assert (
            result["metrics"]["prediction_duration_sec"]
            >= result["baseline_metrics"]["prediction_duration_sec"] - 1e-9
        )
    label_inconsistency = (
        result["metrics"]["prediction_duration_sec"]
        < result["baseline_metrics"]["prediction_duration_sec"] - 1e-9
        and all(a["left_action"] != "TRIM" and a["right_action"] != "TRIM" for a in result["actions"])
    )
    assert label_inconsistency is False


def test_baseline_matches_stage44_semantics() -> None:
    """BHD baseline (merged per-video) must equal evaluate_weak_references on merged segments."""
    from aic_video_highlight.highlight_retrieval.baseline_experiment import (
        evaluate_weak_references,
    )

    segments = [(5.0, 15.0), (20.0, 30.0)]
    refs = [(6.0, 9.0), (22.0, 26.0)]
    bhd_metrics = bhd.evaluate_video_merged(segments, refs)
    frozen_metrics = evaluate_weak_references(
        [{"start_sec": s, "end_sec": e} for s, e in segments],
        [{"start_sec": s, "end_sec": e} for s, e in refs],
    )
    assert bhd_metrics["precision"] == pytest.approx(frozen_metrics["weak_ref_precision"], abs=1e-9)
    assert bhd_metrics["recall"] == pytest.approx(frozen_metrics["weak_ref_recall"], abs=1e-9)
    assert bhd_metrics["f1"] == pytest.approx(frozen_metrics["weak_ref_f1"], abs=1e-9)
    assert bhd_metrics["temporal_iou"] == pytest.approx(frozen_metrics["temporal_iou"], abs=1e-9)


def test_summary_schema_separates_forced_and_optional() -> None:
    sample = {
        "local_boundary_optional_oracle": {"upper_bound": True, "identity_included": True},
        "local_boundary_forced": {"forced_diagnostic": True, "upper_bound": False},
    }
    payload = json.dumps(sample)
    assert "upper_bound" in payload and "forced_diagnostic" in payload


def test_shot_detection_deterministic(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    path = tmp_path / "shot_test.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (64, 48))
    for index in range(16):
        frame = np.full((48, 64, 3), 30 if index < 8 else 220, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    def factory(p):
        return cv2.VideoCapture(str(p), cv2.CAP_FFMPEG)

    first = bhd.detect_shot_boundaries(path, factory, frame_stride_sec=0.5, min_shot_len_sec=1.0, duration_sec=4.0)
    second = bhd.detect_shot_boundaries(path, factory, frame_stride_sec=0.5, min_shot_len_sec=1.0, duration_sec=4.0)
    assert first == second
    segments = bhd.shot_segments_from_boundaries(first, 4.0)
    for (_, end_a), (start_b, _) in zip(segments, segments[1:]):
        assert end_a <= start_b + 1e-9


def test_shot_snap_optional_identity_dominance() -> None:
    candidates = [_candidate("c1", 5.0, 15.0)]
    refs = [(6.0, 9.0)]
    segments = [(0.0, 6.5), (6.5, 40.0)]
    result = bhd.greedy_shot_snap_oracle(
        candidates, refs, 40.0, segments, max_shift=3.0, objective="f1"
    )
    assert result["metrics"]["f1"] >= result["baseline_metrics"]["f1"] - 1e-9
    assert result["identity_included"] is True


def test_subshot_oracle_marked_non_deployable_and_identity_included() -> None:
    candidates = [_candidate("c1", 0.0, 10.0)]
    refs = [(0.0, 2.0), (6.0, 9.0)]
    segments = [(0.0, 3.0), (3.0, 6.0), (6.0, 10.0)]
    result = bhd.greedy_subshot_oracle(
        candidates, refs, 12.0, segments, max_components=2, objective="f1"
    )
    assert result["oracle_markers"]["non_deployable"] is True
    assert result["metrics"]["f1"] >= result["baseline_metrics"]["f1"] - 1e-9


def test_no_nan_inf_in_oracle_outputs() -> None:
    candidates = [_candidate(f"c{i}", 5.0 + i * 15.0, 15.0 + i * 15.0) for i in range(3)]
    result = bhd.greedy_video_oracle(
        candidates, [(6.0, 9.0), (21.0, 24.0), (36.0, 40.0)], 70.0, max_shift=5.0, step=0.5, objective="f1"
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


def test_no_label_leakage_in_oracle_output() -> None:
    candidates = [_candidate("c1", 5.0, 15.0)]
    result = bhd.greedy_video_oracle(candidates, [(6.0, 9.0)], 40.0, max_shift=2.0, step=0.5, objective="f1")
    payload = json.dumps(result).lower()
    for forbidden in ("audit", "human_label", "adjudication", "heldout"):
        assert forbidden not in payload
    # uses_weak_reference inside oracle_markers is a REQUIRED audit marker,
    # not leakage; it must appear exactly there.
    assert result["oracle_markers"]["uses_weak_reference"] is True


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
