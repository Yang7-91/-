"""Stage 4-new BHD-0.1 boundary headroom diagnostics (audited semantics).

All oracle searches are video-level greedy walks over the identity-inclusive
variant space, evaluated with the same merged per-video metric semantics as
``run_boundary_refinement.py evaluate``.  Optional oracles therefore satisfy
identity dominance (aggregate metric >= BR-0 baseline) by construction, while
forced diagnostics are clearly labelled and may be worse than baseline.

Everything here is ORACLE_DIAGNOSTIC_ONLY / NON_DEPLOYABLE / USES_WEAK_REFERENCE.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Mapping

import numpy as np

BHD_MARKERS = {
    "oracle_diagnostic_only": True,
    "non_deployable": True,
    "uses_weak_reference": True,
}

_EPS = 1e-9


def _finite(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _all_finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(v) for v in values)


def evaluate_video_merged(
    segments: list[tuple[float, float]], references: list[tuple[float, float]]
) -> dict[str, float]:
    """Merged per-video duration metrics (same semantics as Stage 4.4 evaluate)."""
    predicted = [(s, e) for s, e in segments if e > s]
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
        "intersection_sec": intersection,
    }


def candidate_refs(prediction_row: Mapping[str, Any]) -> list[tuple[float, float]]:
    return [
        (_finite(seg["start_sec"]), _finite(seg["end_sec"]))
        for seg in prediction_row.get("weak_reference_segments", [])
        if _finite(seg.get("end_sec", 0.0)) > _finite(seg.get("start_sec", 0.0))
    ]


def baseline_video_metrics(
    prediction_rows: Iterable[Mapping[str, Any]],
    durations_by_id: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Per-video merged baseline metrics (matches Stage 4.4 evaluate semantics)."""
    per_video: dict[str, dict[str, float]] = {}
    for row in prediction_rows:
        video_id = row["video_id"]
        refs = candidate_refs(row)
        segments = [
            (_finite(seg["start_sec"]), _finite(seg["end_sec"]))
            for seg in row.get("merged_prediction_segments", [])
        ]
        metrics = evaluate_video_merged(segments, refs)
        duration = _finite(durations_by_id.get(video_id, 0.0))
        metrics["coverage_ratio"] = (
            metrics["prediction_duration_sec"] / duration if duration > _EPS else 0.0
        )
        per_video[video_id] = metrics
    keys = (
        "precision",
        "recall",
        "f1",
        "temporal_iou",
        "coverage_ratio",
        "prediction_duration_sec",
        "reference_duration_sec",
    )
    aggregate = {
        key: _finite(sum(m[key] for m in per_video.values()) / len(per_video))
        for key in keys
        if per_video
    }
    return aggregate, per_video


def _variant_bounds(
    start: float, end: float, max_shift: float, step: float
) -> list[tuple[float, float]]:
    """All (start, end) variants with both boundaries shifted bidirectionally."""
    offsets = np.arange(-max_shift, max_shift + _EPS, step)
    variants = []
    for ds in offsets:
        for de in offsets:
            variants.append((start + float(ds), end + float(de)))
    return variants


OBJECTIVES = ("f1", "temporal_iou", "precision_under_recall_guard")


def _objective_value(metrics: Mapping[str, float], objective: str) -> float:
    if objective == "f1":
        return metrics["f1"]
    if objective == "temporal_iou":
        return metrics["temporal_iou"]
    if objective == "precision_under_recall_guard":
        return metrics["precision"]
    raise ValueError(f"unknown objective: {objective}")


def greedy_video_oracle(
    candidates: list[Mapping[str, Any]],
    references: list[tuple[float, float]],
    video_duration: float,
    *,
    max_shift: float,
    step: float,
    objective: str,
    recall_guard: float | None = None,
    allow_identity: bool = True,
) -> dict[str, Any]:
    """Greedy per-candidate boundary search evaluated at video level.

    ``allow_identity=True`` (optional oracle) keeps the current segment unless
    a variant strictly improves the video-level objective, which guarantees the
    final aggregate is never worse than baseline for that objective.
    ``allow_identity=False`` (forced diagnostic) must replace every candidate
    with its locally best variant and may be worse than baseline.
    """
    segments = [(_finite(c["start_sec"]), _finite(c["end_sec"])) for c in candidates]
    base_metrics = evaluate_video_merged(segments, references)
    guard_recall = base_metrics["recall"] if recall_guard is None else recall_guard

    def video_metrics(trial: list[tuple[float, float]]) -> dict[str, float]:
        metrics = evaluate_video_merged(trial, references)
        metrics["coverage_ratio"] = (
            metrics["prediction_duration_sec"] / video_duration if video_duration > _EPS else 0.0
        )
        return metrics

    current = list(segments)
    current_metrics = base_metrics
    actions: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        start, end = segments[index]
        variants = [
            (s, e)
            for s, e in _variant_bounds(start, end, max_shift, step)
            if 0.0 <= s < e <= video_duration + _EPS
        ]
        best_choice = None
        if allow_identity:
            best_choice = (current[index], current_metrics)
        for variant in variants:
            trial = _replace(current, index, variant)
            metrics = video_metrics(trial)
            if recall_guard is not None and metrics["recall"] < guard_recall - _EPS:
                continue
            value = _objective_value(metrics, objective)
            best_value = (
                _objective_value(best_choice[1], objective) if best_choice is not None else float("-inf")
            )
            if value > best_value + _EPS:
                best_choice = (variant, metrics)
        if best_choice is None:
            actions.append(
                {
                    "candidate_id": candidate.get("merged_candidate_id"),
                    "left_delta_sec": 0.0,
                    "right_delta_sec": 0.0,
                    "left_action": "KEEP",
                    "right_action": "KEEP",
                }
            )
            continue
        chosen, _ = best_choice
        current[index] = chosen
        left_delta = chosen[0] - start
        right_delta = chosen[1] - end
        actions.append(
            {
                "candidate_id": candidate.get("merged_candidate_id"),
                "left_delta_sec": left_delta,
                "right_delta_sec": right_delta,
                "left_action": "TRIM" if left_delta > _EPS else ("EXPAND" if left_delta < -_EPS else "KEEP"),
                "right_action": "TRIM" if right_delta < -_EPS else ("EXPAND" if right_delta > _EPS else "KEEP"),
            }
        )
        current_metrics = video_metrics(current)
    final_metrics = video_metrics(current)
    if not allow_identity:
        pass
    return {
        "refined_segments": current,
        "metrics": final_metrics,
        "actions": actions,
        "baseline_metrics": base_metrics,
        "identity_included": allow_identity,
        "forced_diagnostic": not allow_identity,
        "upper_bound": allow_identity,
        "oracle_markers": dict(BHD_MARKERS),
    }


def _replace(segments: list[tuple[float, float]], index: int, variant: tuple[float, float]) -> list[tuple[float, float]]:
    trial = list(segments)
    trial[index] = variant
    return trial


def greedy_forced_diagnostic(
    candidates: list[Mapping[str, Any]],
    references: list[tuple[float, float]],
    video_duration: float,
    *,
    max_shift: float,
    step: float,
    objective: str,
) -> dict[str, Any]:
    """Forced variant of the greedy oracle: every candidate must move."""
    result = greedy_video_oracle(
        candidates,
        references,
        video_duration,
        max_shift=max_shift,
        step=step,
        objective=objective,
        allow_identity=False,
    )
    result["forced_diagnostic"] = True
    result["upper_bound"] = False
    return result


def detect_shot_boundaries(
    video_path: Any,
    capture_factory: Callable[[Any], Any],
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
    previous_gray: np.ndarray | None = None
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
            if previous_gray is not None:
                pixel = float(np.mean(cv2.absdiff(previous_gray, gray))) / 255.0
                hist_diff = (
                    float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
                    if previous_hist is not None
                    else 0.0
                )
                diffs.append((t, 0.5 * pixel + 0.5 * hist_diff))
            previous_gray = gray
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


def shot_segments_from_boundaries(boundaries: list[float], duration_sec: float) -> list[tuple[float, float]]:
    edges = [0.0] + sorted(boundaries) + [duration_sec]
    return [(lo, hi) for lo, hi in zip(edges, edges[1:]) if hi - lo > _EPS]


def shot_snap_choices(
    start: float,
    end: float,
    shot_segments: list[tuple[float, float]],
    max_shift: float,
) -> list[tuple[float, float]]:
    """Identity plus nearest-shot-edge snap variants within the shift budget."""
    edges = sorted({edge for segment in shot_segments for edge in segment})
    choices = [(start, end)]
    for new_start in [start] + [e for e in edges if 0 < abs(e - start) <= max_shift + _EPS]:
        for new_end in [end] + [e for e in edges if 0 < abs(e - end) <= max_shift + _EPS]:
            if new_end > new_start and (new_start, new_end) not in choices:
                choices.append((new_start, new_end))
    return choices


def greedy_shot_snap_oracle(
    candidates: list[Mapping[str, Any]],
    references: list[tuple[float, float]],
    video_duration: float,
    shot_segments: list[tuple[float, float]],
    *,
    max_shift: float,
    objective: str,
    allow_identity: bool = True,
) -> dict[str, Any]:
    def snap_variants(start: float, end: float) -> list[tuple[float, float]]:
        return [
            (s, e)
            for s, e in shot_snap_choices(start, end, shot_segments, max_shift)
            if 0.0 <= s < e <= video_duration + _EPS
        ]

    return _greedy_over_variant_fn(
        candidates,
        references,
        video_duration,
        objective,
        allow_identity,
        lambda index, start, end: snap_variants(start, end),
    )


def greedy_subshot_oracle(
    candidates: list[Mapping[str, Any]],
    references: list[tuple[float, float]],
    video_duration: float,
    shot_segments: list[tuple[float, float]],
    *,
    max_components: int,
    objective: str,
    allow_identity: bool = True,
) -> dict[str, Any]:
    def subshot_variants(index: int, start: float, end: float) -> list[tuple[float, float]]:
        del index
        inside = [
            (max(start, lo), min(end, hi))
            for lo, hi in shot_segments
            if min(end, hi) - max(start, lo) > _EPS
        ]
        if not inside:
            return []
        variants: list[tuple[float, float]] = []
        n = len(inside)
        for count in range(1, min(max_components, n) + 1):
            for i in range(n - count + 1):
                chosen = inside[i : i + count]
                variants.append((chosen[0][0], chosen[-1][1]))
        return [(s, e) for s, e in variants if 0.0 <= s < e <= video_duration + _EPS]

    return _greedy_over_variant_fn(
        candidates,
        references,
        video_duration,
        objective,
        allow_identity,
        subshot_variants,
    )


def _greedy_over_variant_fn(
    candidates: list[Mapping[str, Any]],
    references: list[tuple[float, float]],
    video_duration: float,
    objective: str,
    allow_identity: bool,
    variants_fn: Callable[[int, float, float], list[tuple[float, float]]],
) -> dict[str, Any]:
    segments = [(_finite(c["start_sec"]), _finite(c["end_sec"])) for c in candidates]
    base_metrics = evaluate_video_merged(segments, references)

    def video_metrics(trial: list[tuple[float, float]]) -> dict[str, float]:
        metrics = evaluate_video_merged(trial, references)
        metrics["coverage_ratio"] = (
            metrics["prediction_duration_sec"] / video_duration if video_duration > _EPS else 0.0
        )
        return metrics

    current = list(segments)
    actions: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        start, end = segments[index]
        variants = [
            v for v in variants_fn(index, start, end) if (v != current[index] or not allow_identity)
        ]
        best_choice = (current[index], video_metrics(current)) if allow_identity else None
        for variant in variants:
            trial = _replace(current, index, variant)
            metrics = video_metrics(trial)
            value = _objective_value(metrics, objective)
            if best_choice is None or value > _objective_value(best_choice[1], objective) + _EPS:
                best_choice = (variant, metrics)
        if best_choice is None:
            actions.append(
                {
                    "candidate_id": candidate.get("merged_candidate_id"),
                    "left_delta_sec": 0.0,
                    "right_delta_sec": 0.0,
                    "left_action": "KEEP",
                    "right_action": "KEEP",
                }
            )
            continue
        chosen, _ = best_choice
        current[index] = chosen
        left_delta = chosen[0] - start
        right_delta = chosen[1] - end
        actions.append(
            {
                "candidate_id": candidate.get("merged_candidate_id"),
                "left_delta_sec": left_delta,
                "right_delta_sec": right_delta,
                "left_action": "TRIM" if left_delta > _EPS else ("EXPAND" if left_delta < -_EPS else "KEEP"),
                "right_action": "TRIM" if right_delta < -_EPS else ("EXPAND" if right_delta > _EPS else "KEEP"),
            }
        )
    final_metrics = video_metrics(current)
    return {
        "refined_segments": current,
        "metrics": final_metrics,
        "actions": actions,
        "baseline_metrics": base_metrics,
        "identity_included": allow_identity,
        "forced_diagnostic": not allow_identity,
        "upper_bound": allow_identity,
        "oracle_markers": dict(BHD_MARKERS),
    }


def rank_auc_like(positive_scores: list[float], negative_scores: list[float]) -> float:
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
