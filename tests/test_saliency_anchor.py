"""Tests for Stage 4-new SABR-1.0 saliency anchor feature extraction."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
spec = importlib.util.spec_from_file_location(
    "run_saliency_anchor", Path(__file__).resolve().parents[1] / "scripts" / "run_saliency_anchor.py"
)
run_saliency_anchor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_saliency_anchor)

from aic_video_highlight.highlight_retrieval.saliency_anchor import (  # noqa: E402
    SIGNAL_NAMES,
    SaliencyExtractionError,
    extract_saliency_bins,
    normalize_saliency_bins,
    summarize_candidate_saliency,
)

cv2 = pytest.importorskip("cv2")


def _write_test_video(path: Path, seconds: float = 2.0, fps: int = 4) -> Path:
    """Create a small deterministic synthetic video."""
    width, height = 64, 48
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened()
    total = int(seconds * fps)
    for index in range(total):
        frame = np.full((height, width, 3), 128, dtype=np.uint8)
        value = 40 + (index * 37) % 160
        frame[:, : width // 2] = (value, value // 2, 255 - value)
        frame[: height // 2, width // 2 :] = (255 - value, value, value // 3)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture()
def test_video(tmp_path: Path) -> Path:
    return _write_test_video(tmp_path / "qvh_test_9x16.mp4")


def _all_finite(bins: list[dict]) -> bool:
    for item in bins:
        for key, value in item.items():
            if isinstance(value, float) and not math.isfinite(value):
                return False
    return True


def test_bin_splitting_is_deterministic(test_video: Path) -> None:
    first = extract_saliency_bins(test_video, 0.0, 2.0, 1.0, video_id="v", candidate_id="c")
    second = extract_saliency_bins(test_video, 0.0, 2.0, 1.0, video_id="v", candidate_id="c")
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_bin_start_end_legal_and_serializable(test_video: Path) -> None:
    bins = extract_saliency_bins(test_video, 0.25, 1.75, 1.0, video_id="v", candidate_id="c")
    assert bins, "expected non-empty bins"
    for index, item in enumerate(bins):
        assert item["bin_index"] == index
        assert 0.0 <= item["start_sec"] < item["end_sec"] <= 1.75 + 1e-9
        assert json.dumps(item)  # serializable
        assert set(SIGNAL_NAMES).issubset(item.keys())
        assert "saliency_score" in item


def test_missing_video_path_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(SaliencyExtractionError) as excinfo:
        extract_saliency_bins(tmp_path / "does_not_exist.mp4", 0.0, 1.0)
    assert "not found" in str(excinfo.value)


def test_invalid_interval_raises(tmp_path: Path) -> None:
    video = _write_test_video(tmp_path / "tiny.mp4", seconds=1.0)
    with pytest.raises(SaliencyExtractionError):
        extract_saliency_bins(video, 1.0, 1.0)


def test_normalize_handles_constant_sequences() -> None:
    bins = [
        {
            "bin_index": i,
            "start_sec": float(i),
            "end_sec": float(i + 1),
            "frame_count": 2,
            "frame_difference": 0.5,
            "histogram_difference": 0.1,
            "laplacian_variance": 100.0,
            "edge_density": 0.2,
            "saturation": 0.3,
            "contrast": 0.4,
            "saliency_score": 0.0,
        }
        for i in range(3)
    ]
    normalized = normalize_saliency_bins(bins)
    assert len(normalized) == 3
    assert _all_finite(normalized)
    scores = [item["saliency_score"] for item in normalized]
    assert json.dumps(scores)
    assert all(math.isfinite(s) for s in scores)


def test_normalize_never_produces_nan_or_inf(test_video: Path) -> None:
    bins = extract_saliency_bins(test_video, 0.0, 2.0, 0.5, video_id="v", candidate_id="c")
    normalized = normalize_saliency_bins(bins)
    assert _all_finite(normalized)


def test_summary_shape(test_video: Path) -> None:
    bins = normalize_saliency_bins(
        extract_saliency_bins(test_video, 0.0, 2.0, 1.0, video_id="v", candidate_id="c")
    )
    summary = summarize_candidate_saliency("v", "c", bins)
    assert summary["video_id"] == "v"
    assert summary["candidate_id"] == "c"
    assert summary["bin_count"] == len(bins)
    assert 0.0 <= summary["peak_bin"]["saliency_score"] <= max(
        b["saliency_score"] for b in bins
    )
    assert summary["bin_level_signals_present"] is True


def test_cli_help_runs(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        run_saliency_anchor.main(["--help"])
    assert excinfo.value.code == 0
    assert "extract" in capsys.readouterr().out


def test_cli_extract_rejects_bad_protocol(tmp_path: Path, test_video: Path) -> None:
    bad_protocol = tmp_path / "bad_protocol.json"
    bad_protocol.write_text(json.dumps({"schema_version": "wrong/v0"}), encoding="utf-8")
    with pytest.raises(SystemExit):
        run_saliency_anchor.main(
            [
                "extract",
                "--cache-dir",
                str(tmp_path),
                "--role-manifest",
                str(tmp_path / "role.json"),
                "--video-dir",
                str(tmp_path),
                "--protocol",
                str(bad_protocol),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )


def test_protocol_file_safety_flags() -> None:
    protocol = json.loads(
        (Path(__file__).resolve().parents[1] / "configs" / "stage4_sabr1_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["safety"]["this_phase_changes_boundaries"] is False
    assert protocol["safety"]["no_heldout"] is True
    assert protocol["safety"]["no_qwen"] is True
    assert protocol["safety"]["no_vllm"] is True
    assert protocol["frozen_upstream"]["heldout_access"] is False
    assert protocol["frozen_upstream"]["training_allowed"] is False
    assert protocol["feature_extraction"]["deterministic"] is True
