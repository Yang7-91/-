"""Stage 4-new BHD-0 boundary headroom and signal separability diagnostics.

Diagnostic only: every oracle in this module uses weak references to measure
the *theoretical* headroom of boundary refinement.  Oracle outputs are never
deployable methods and never become formal predictions.  No model is called.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np

BHD0_MARKERS = {
    "oracle_diagnostic_only": True,
    "non_deployable": True,
    "uses_weak_reference": True,
}

_EPS = 1e-6


def _finite(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _intervals_iou_metrics(
    predicted: list[tuple[float, float]], references: list[tuple[float, float]]
) -> dict[str, float]:
    """Duration-based P/R/F1/tIoU for small interval sets (no model needed)."""
    predicted = [(s, e) for s, e in predicted if e > s]
    references = [(s, e) for s, e in references if e > s]
    pred_dur = sum(e - s for s, e in predicted)
    ref_dur = sum(e - s for s, e in references)
    intersection = 0.0
    for ps, pe in predicted:
        for rs, re_ in references:
            intersection += max(0.0, min(pe, re_) - max(ps, rs))
    union = pred_dur + ref_dur - intersection
    precision = intersection / pred_dur if pred_dur > _EPS else 0.0
    recall = intersection / ref_dur if ref_dur > _EPS else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > _EPS else 0.0
    iou = intersection / union if union > _EPS else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "temporal_iou": iou,
        "prediction_duration_sec": pred_dur,
        "reference_duration_sec": ref_dur,
    }


def _candidate_refs(prediction_row: Mapping[str, Any]) -> list[tuple[float, float]]:
    return [
        (_finite(seg["start_sec"]), _finite(seg["end_sec"]))
        for seg in prediction_row.get("weak_reference_segments", [])
        if _finite(seg.get("end_sec", 0.0)) > _finite(seg.get("start_sec", 0.0))
    ]


def run_local_boundary_oracle_for_candidate(
    candidate: Mapping[str, Any],
    references: list[tuple[float, float]],
    max_shift_sec: float,
    step_sec: float,
) -> dict[str, Any]:
    """Exhaustive local start/end shift search for one candidate (oracle)."""
    start = _finite(candidate["start_sec"])
    end = _finite(candidate["end_sec"])
    shifts = np.arange(0.0, max_shift_sec + _EPS, step_sec)
    best = None
    for ds in shifts:
        for de in shifts:
            new_start = start - ds
            new_end = end + de
            if new_start < 0 or new_end <= new_start:
                continue
            metrics = _intervals_iou_metrics([(new_start, new_end)], references)
            key = (metrics["f1"], metrics["temporal_iou"], metrics["precision"])
            if best is None or key > best[0]:
                best = (key, new_start, new_end, metrics)
    if best is None:
        new_start, new_end = start, end
        metrics = _intervals_iou_metrics([(start, end)], references)
    else:
        _, new_start, new_end, metrics = best
    left_delta = new_start - start
    right_delta = end - new_end
    left_action = "EXPAND" if left_delta < -_EPS else ("TRIM" if left_delta > _EPS else "KEEP")
    right_action = "EXPAND" if right_delta < -_EPS else ("TRIM" if right_delta > _EPS else "KEEP")
    return {
        "oracle_start_sec": new_start,
        "oracle_end_sec": new_end,
        "left_delta_sec": left_delta,
        "right_delta_sec": right_delta,
        "left_action": left_action,
        "right_action": right_action,
        "metrics": metrics,
        "oracle_markers": dict(BHD0_MARKERS),
    }


def aggregate_oracle_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Mean per-candidate oracle metrics (mirrors BR-0 aggregate semantics)."""
    if not rows:
        return {}
    keys = ("precision", "recall", "f1", "temporal_iou", "prediction_duration_sec", "reference_duration_sec")
    return {key: _finite(sum(row["metrics"][key] for row in rows) / len(rows)) for key in keys}


def baseline_aggregate(prediction_rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    aggregate: dict[str, list[float]] = {}
    for row in prediction_rows:
        refs = _candidate_refs(row)
        predicted = [
            (_finite(seg["start_sec"]), _finite(seg["end_sec"]))
            for seg in row.get("merged_prediction_segments", [])
        ]
        metrics = _intervals_iou_metrics(predicted, refs)
        for key, value in metrics.items():
            aggregate.setdefault(key, []).append(value)
    return {key: _finite(sum(values) / len(values)) for key, values in aggregate.items()}


def run_local_boundary_oracle(
    candidates_by_video: Mapping[str, list[Mapping[str, Any]]],
    references_by_video: Mapping[str, list[tuple[float, float]]],
    max_shift_levels: list[float],
    step_sec: float,
) -> dict[str, Any]:
    """Aggregate local-boundary oracle metrics per max_shift level."""
    levels: dict[str, Any] = {}
    labels: list[dict[str, Any]] = []
    for max_shift in max_shift_levels:
        rows = []
        for video_id, candidates in candidates_by_video.items():
            refs = references_by_video.get(video_id, [])
            for candidate in candidates:
                result = run_local_boundary_oracle_for_candidate(
                    candidate, refs, max_shift, step_sec
                )
                result["video_id"] = video_id
                result["candidate_id"] = candidate.get("merged_candidate_id")
                result["max_shift_sec"] = max_shift
                rows.append(result)
                labels.append(
                    {
                        "video_id": video_id,
                        "candidate_id": candidate.get("merged_candidate_id"),
                        "max_shift_sec": max_shift,
                        "left_action": result["left_action"],
                        "right_action": result["right_action"],
                        "oracle_markers": dict(BHD0_MARKERS),
                    }
                )
        aggregate = aggregate_oracle_metrics(rows)
        action_counts = {
            side: {
                action: sum(1 for row in rows if row[f"{side}_action"] == action)
                for action in ("TRIM", "KEEP", "EXPAND")
            }
            for side in ("left", "right")
        }
        levels[str(max_shift)] = {
            "aggregate": aggregate,
            "action_counts": action_counts,
            "oracle_markers": dict(BHD0_MARKERS),
        }
    labels_out = [row for row in labels if row["max_shift_sec"] == max(max_shift_levels)]
    return {"levels": levels, "oracle_labels": labels_out}


def detect_shot_boundaries(
    video_path: Any,
    capture_factory,
    *,
    frame_stride_sec: float = 0.5,
    min_shot_len_sec: float = 1.0,
    duration_sec: float,
) -> list[float]:
    """Deterministic histogram+pixel difference shot boundary detection."""
    import cv2

    capture = capture_factory(video_path)
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    diffs: list[tuple[float, float]] = []
    previous: np.ndarray | None = None
    previous_hist: np.ndarray | None = None
    t = 0.0
    try:
        while t <= duration_sec + _EPS:
            capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            small = cv2.resize(frame, (64, 48), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            cv2.normalize(hist, hist)
            if previous is not None:
                pixel = float(np.mean(cv2.absdiff(previous, gray))) / 255.0
                hist_diff = (
                    float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
                    if previous_hist is not None
                    else 0.0
                )
                diffs.append((t, 0.5 * pixel + 0.5 * hist_diff))
            previous = gray
            previous_hist = hist
            t += frame_stride_sec
    finally:
        capture.release()
    if not diffs:
        return []
    values = np.asarray([value for _, value in diffs], dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    adaptive = median + 3.0 * 1.4826 * mad
    percentile = float(np.percentile(values, 85))
    threshold = max(adaptive, percentile, 1e-6)
    raw = [t for t, value in diffs if value > threshold]
    boundaries: list[float] = []
    for t in raw:
        if boundaries and t - boundaries[-1] < min_shot_len_sec:
            continue
        boundaries.append(t)
    return boundaries


def shot_segments_from_boundaries(
    boundaries: list[float], duration_sec: float
) -> list[tuple[float, float]]:
    """Split [0, duration] into non-overlapping shot segments."""
    edges = [0.0] + sorted(boundaries) + [duration_sec]
    segments = []
    for lo, hi in zip(edges, edges[1:]):
        if hi - lo > _EPS:
            segments.append((lo, hi))
    return segments


def snap_oracle_to_shot_boundary(
    candidate: Mapping[str, Any],
    references: list[tuple[float, float]],
    shot_segments: list[tuple[float, float]],
    max_shift_sec: float,
) -> dict[str, Any]:
    """Snap candidate start/end to nearest shot edges within shift (oracle)."""
    start = _finite(candidate["start_sec"])
    end = _finite(candidate["end_sec"])
    edges = sorted({edge for segment in shot_segments for edge in segment})

    def near_edges(point: float) -> list[float]:
        window = [point] + [
            edge for edge in edges if abs(edge - point) <= max_shift_sec + _EPS
        ]
        return sorted(set(window))

    best = None
    for new_start in near_edges(start):
        for new_end in near_edges(end):
            if new_end <= new_start:
                continue
            metrics = _intervals_iou_metrics([(new_start, new_end)], references)
            key = (metrics["f1"], metrics["temporal_iou"])
            if best is None or key > best[0]:
                best = (key, new_start, new_end, metrics)
    if best is None:
        return {"snapped": False}
    _, new_start, new_end, metrics = best
    return {
        "snapped": True,
        "oracle_start_sec": new_start,
        "oracle_end_sec": new_end,
        "metrics": metrics,
        "start_snap_distance_sec": abs(new_start - start),
        "end_snap_distance_sec": abs(new_end - end),
        "oracle_markers": dict(BHD0_MARKERS),
    }


def oracle_subshot_split_upper_bound(
    candidate: Mapping[str, Any],
    references: list[tuple[float, float]],
    shot_segments: list[tuple[float, float]],
    max_components: int,
) -> dict[str, Any]:
    """Best contiguous subshot component selection (oracle, non-deployable)."""
    start = _finite(candidate["start_sec"])
    end = _finite(candidate["end_sec"])
    inside = [
        (max(start, lo), min(end, hi))
        for lo, hi in shot_segments
        if min(end, hi) - max(start, lo) > _EPS
    ]
    if not inside:
        inside = [(start, end)]
    best = None
    n = len(inside)
    for count in range(1, min(max_components, n) + 1):
        for i in range(n - count + 1):
            chosen = inside[i : i + count]
            metrics = _intervals_iou_metrics(list(chosen), references)
            key = (metrics["f1"], metrics["temporal_iou"])
            if best is None or key > best[0]:
                best = (key, chosen, metrics)
    if best is None:
        return {"components": 0, "oracle_markers": dict(BHD0_MARKERS)}
    _, chosen, metrics = best
    return {
        "components": len(chosen),
        "chosen_segments": [
            {"start_sec": _finite(s), "end_sec": _finite(e)} for s, e in chosen
        ],
        "metrics": metrics,
        "oracle_markers": dict(BHD0_MARKERS),
    }


def rank_auc_like(positive_scores: list[float], negative_scores: list[float]) -> float:
    """Rank-based AUC-like separability in [0, 1] (0.5 = no separation)."""
    if not positive_scores or not negative_scores:
        return 0.5
    wins = 0.0
    for p in positive_scores:
        for n in negative_scores:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return _finite(wins / (len(positive_scores) * len(negative_scores)))


def mean_gap(positive_scores: list[float], negative_scores: list[float]) -> float:
    if not positive_scores or not negative_scores:
        return 0.0
    return _finite(sum(positive_scores) / len(positive_scores) - sum(negative_scores) / len(negative_scores))


def effect_size(positive_scores: list[float], negative_scores: list[float]) -> float:
    """Cohen's d between the two score groups."""
    if len(positive_scores) < 2 or len(negative_scores) < 2:
        return 0.0
    a = np.asarray(positive_scores, dtype=float)
    b = np.asarray(negative_scores, dtype=float)
    pooled = math.sqrt((a.var() + b.var()) / 2.0)
    if pooled <= _EPS:
        return 0.0
    return _finite((a.mean() - b.mean()) / pooled)


def has_meaningful_headroom(
    delta_precision: float,
    delta_recall: float,
    delta_f1: float,
    delta_tiou: float,
    rules: Mapping[str, Any],
) -> bool:
    return (
        delta_precision >= _finite(rules["meaningful_oracle_precision_delta"])
        and delta_recall >= _finite(rules["meaningful_oracle_recall_delta_min"])
        and max(delta_f1, delta_tiou) >= _finite(rules["meaningful_oracle_f1_or_tiou_delta"])
    )
