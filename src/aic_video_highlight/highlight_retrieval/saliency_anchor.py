"""Stage 4-new SABR-1 saliency feature extraction over frozen candidates.

Extracts deterministic, low-cost visual saliency signals inside the frozen
candidate intervals of the Stage 4.2 cache using only OpenCV / NumPy and the
standard library.  This phase never changes candidate boundaries, never reads
references or Heldout data, and never calls any model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SALIENCY_ANCHOR_SCHEMA_VERSION = "aic.stage4.sabr1_saliency/v1"

SIGNAL_NAMES = (
    "frame_difference",
    "histogram_difference",
    "laplacian_variance",
    "edge_density",
    "saturation",
    "contrast",
)

SIGNAL_WEIGHTS = {
    "frame_difference": 0.30,
    "histogram_difference": 0.20,
    "laplacian_variance": 0.15,
    "edge_density": 0.10,
    "saturation": 0.10,
    "contrast": 0.15,
}

_FRAMES_PER_BIN = 2


class SaliencyExtractionError(RuntimeError):
    """Raised when saliency extraction cannot proceed for a video."""


def _finite(value: Any) -> float:
    """Coerce to a finite float; non-finite values become 0.0."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def compute_frame_difference(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    """Mean absolute per-pixel difference between consecutive frames in [0, 1]."""
    if prev_gray is None or gray is None or prev_gray.shape != gray.shape:
        return 0.0
    diff = _cv2_absdiff(prev_gray, gray)
    return _finite(float(np.mean(diff)) / 255.0)


def compute_histogram_difference(prev_hist: np.ndarray, hist: np.ndarray) -> float:
    """Bhattacharyya distance between consecutive grayscale histograms in [0, 1]."""
    if prev_hist is None or hist is None:
        return 0.0
    distance = _cv2_compareHist(prev_hist, hist)
    return _finite(max(0.0, min(1.0, float(distance))))


def compute_laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian response (sharpness proxy)."""
    if gray is None:
        return 0.0
    lap = _cv2_laplacian(gray)
    return _finite(float(lap.var()))


def compute_edge_density(gray: np.ndarray) -> float:
    """Fraction of pixels flagged as edges by Canny."""
    if gray is None or gray.size == 0:
        return 0.0
    edges = _cv2_canny(gray)
    return _finite(float(np.count_nonzero(edges)) / float(edges.size))


def compute_saturation_contrast(frame_bgr: np.ndarray, gray: np.ndarray) -> tuple[float, float]:
    """Mean HSV saturation in [0, 1] and grayscale contrast (std / 127.5)."""
    if frame_bgr is None or gray is None:
        return 0.0, 0.0
    hsv = _cv2_cvt_hsv(frame_bgr)
    saturation = _finite(float(np.mean(hsv[:, :, 1])) / 255.0)
    contrast = _finite(float(np.std(gray)) / 127.5)
    return saturation, contrast


def _iter_bin_frames(
    capture: Any,
    start_sec: float,
    end_sec: float,
    bin_sec: float,
    frames_per_bin: int,
) -> list[dict[str, Any]]:
    """Deterministically sample frames per bin using absolute-time seeks."""
    import cv2

    if not capture.isOpened():
        raise SaliencyExtractionError("video capture is not open")
    bins: list[dict[str, Any]] = []
    total = max(0.0, end_sec - start_sec)
    if total <= 0.0:
        return bins
    bin_count = int(math.ceil(total / bin_sec))
    bin_count = max(bin_count, 1)
    prev_gray: np.ndarray | None = None
    prev_hist: np.ndarray | None = None
    for bin_index in range(bin_count):
        bin_start = start_sec + bin_index * bin_sec
        bin_end = min(start_sec + (bin_index + 1) * bin_sec, end_sec)
        offsets = [
            bin_start + (i + 0.5) * (bin_end - bin_start) / frames_per_bin
            for i in range(frames_per_bin)
        ]
        frame_difference = 0.0
        histogram_difference = 0.0
        laplacian_variance = 0.0
        edge_density = 0.0
        saturation = 0.0
        contrast = 0.0
        frame_count = 0
        for offset in offsets:
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, offset) * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            if frame_count == 0:
                if prev_gray is not None:
                    frame_difference = compute_frame_difference(prev_gray, gray)
                if prev_hist is not None:
                    histogram_difference = compute_histogram_difference(prev_hist, hist)
            laplacian_variance += compute_laplacian_variance(gray)
            edge_density += compute_edge_density(gray)
            sat, con = compute_saturation_contrast(frame, gray)
            saturation += sat
            contrast += con
            frame_count += 1
            prev_gray = gray
            prev_hist = hist
        if frame_count > 0:
            laplacian_variance /= frame_count
            edge_density /= frame_count
            saturation /= frame_count
            contrast /= frame_count
        bins.append(
            {
                "bin_index": bin_index,
                "start_sec": _finite(bin_start),
                "end_sec": _finite(bin_end),
                "frame_count": frame_count,
                "frame_difference": _finite(frame_difference),
                "histogram_difference": _finite(histogram_difference),
                "laplacian_variance": _finite(laplacian_variance),
                "edge_density": _finite(edge_density),
                "saturation": _finite(saturation),
                "contrast": _finite(contrast),
            }
        )
    return bins


def extract_saliency_bins(
    video_path: Path | str,
    start_sec: float,
    end_sec: float,
    bin_sec: float = 1.0,
    *,
    video_id: str = "",
    candidate_id: str = "",
    frames_per_bin: int = _FRAMES_PER_BIN,
) -> list[dict[str, Any]]:
    """Extract per-bin saliency signals for one candidate interval."""
    import cv2

    if bin_sec <= 0:
        raise SaliencyExtractionError("bin_sec must be positive")
    if frames_per_bin <= 0:
        raise SaliencyExtractionError("frames_per_bin must be positive")
    start = _finite(start_sec)
    end = _finite(end_sec)
    if start < 0 or end <= start:
        raise SaliencyExtractionError(
            f"invalid candidate interval [{start_sec}, {end_sec}] for {video_path}"
        )
    path = Path(video_path)
    if not path.is_file():
        raise SaliencyExtractionError(f"video file not found: {path}")
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise SaliencyExtractionError(f"OpenCV could not decode video: {path}")
    try:
        raw_bins = _iter_bin_frames(capture, start, end, float(bin_sec), int(frames_per_bin))
    finally:
        capture.release()
    bins = []
    for item in raw_bins:
        entry = {"video_id": video_id, "candidate_id": candidate_id}
        entry.update(item)
        entry["saliency_score"] = 0.0
        bins.append(entry)
    return bins


def _robust_z(values: list[float]) -> list[float]:
    """Robust z-score via median and MAD; constant sequences map to zeros."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return []
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    if mad <= 0.0:
        return [0.0 for _ in values]
    scale = 1.4826 * mad
    return [_finite((v - median) / scale) for v in values]


def _dense_rank(values: list[float]) -> list[float]:
    """Dense rank normalized to [0, 1]; constant sequences map to 0.5."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return []
    order = np.argsort(array, kind="stable")
    ranks = np.empty(array.size, dtype=float)
    current = 0.0
    previous = None
    for position in order:
        value = float(array[position])
        if previous is not None and value != previous:
            current += 1.0
        ranks[position] = current
        previous = value
    top = ranks.max()
    if top <= 0.0:
        return [0.5 for _ in values]
    return [_finite(r / top) for r in ranks]


def normalize_saliency_bins(bins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply per-candidate robust z-score and rank normalization, then score.

    Score weights are fixed in the protocol and are never tuned on
    weak-reference labels.
    """
    if not bins:
        return []
    normalized: dict[str, list[float]] = {}
    ranks: dict[str, list[float]] = {}
    for signal in SIGNAL_NAMES:
        values = [_finite(b.get(signal)) for b in bins]
        normalized[signal] = _robust_z(values)
        ranks[signal] = _dense_rank(values)
    scored: list[dict[str, Any]] = []
    for index, item in enumerate(bins):
        entry = dict(item)
        entry["saliency_score"] = _finite(
            sum(
                SIGNAL_WEIGHTS[signal]
                * 0.5
                * (normalized[signal][index] + ranks[signal][index])
                for signal in SIGNAL_NAMES
            )
        )
        scored.append(entry)
    return scored


def summarize_candidate_saliency(
    video_id: str,
    candidate_id: str,
    bins: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize one candidate's saliency bins without changing boundaries."""
    scores = [_finite(b.get("saliency_score")) for b in bins]
    if scores:
        peak_index = max(range(len(scores)), key=lambda i: scores[i])
        peak = {
            "bin_index": bins[peak_index].get("bin_index"),
            "start_sec": _finite(bins[peak_index].get("start_sec")),
            "end_sec": _finite(bins[peak_index].get("end_sec")),
            "saliency_score": scores[peak_index],
        }
        mean_score = _finite(sum(scores) / len(scores))
    else:
        peak = {"bin_index": None, "start_sec": None, "end_sec": None, "saliency_score": 0.0}
        mean_score = 0.0
    return {
        "video_id": video_id,
        "candidate_id": candidate_id,
        "bin_count": len(bins),
        "mean_saliency_score": mean_score,
        "peak_bin": peak,
        "bin_level_signals_present": bool(bins) and all(
            all(signal in b for signal in SIGNAL_NAMES) for b in bins
        ),
    }


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def write_json_lines(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as canonical JSONL."""
    payload = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
    Path(path).write_text(payload + ("\n" if rows else ""), encoding="utf-8")


def _cv2_absdiff(prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.absdiff(prev_gray, gray)


def _cv2_compareHist(prev_hist: np.ndarray, hist: np.ndarray) -> float:
    import cv2

    return float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))


def _cv2_laplacian(gray: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.Laplacian(gray, cv2.CV_64F)


def _cv2_canny(gray: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.Canny(gray, 100, 200)


def _cv2_cvt_hsv(frame_bgr: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
