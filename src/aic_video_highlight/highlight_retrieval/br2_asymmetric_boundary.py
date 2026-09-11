"""Stage 4-new BR-2 asymmetric local-evidence boundary refinement.

For each frozen candidate the left and right boundaries are judged
independently against a local evidence curve sampled around the boundary.
Each side chooses one of TRIM / KEEP / EXPAND; both sides KEEP means identity.
Any guard failure falls back to identity for the whole candidate.  Only
OpenCV / NumPy / the standard library are used; no model is called and no
reference, audit, or Heldout information is read.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .candidate_selection import semantic_sha256

BR2_RESULT_SCHEMA_VERSION = "aic.boundary-refinement-result/v1"
BR2_REFINER_VERSION = "aic.br2-asymmetric-boundary/v1"
BR2_REFINER_PREFIX = "BR-2"

_FALLBACK_SHORT_CANDIDATE = "candidate_too_short_for_windows"
_FALLBACK_VIDEO_UNREADABLE = "video_unreadable"
_FALLBACK_EVIDENCE_CONSTANT = "local_evidence_constant"
_FALLBACK_NONFINITE = "nonfinite_evidence_values"
_FALLBACK_ACTION_CONFLICT = "unexplainable_action_conflict"
_FALLBACK_GUARD = "safety_guard_failure"


def _finite(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _sample_frame_signals(
    capture: Any,
    t_sec: float,
    *,
    prev_state: dict[str, Any],
) -> dict[str, float] | None:
    """Read one frame near ``t_sec`` and compute local evidence signals."""
    import cv2

    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_sec) * 1000.0)
    ok, frame = capture.read()
    if not ok or frame is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (64, 48), interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([small], [0], None, [32], [0, 256])
    cv2.normalize(hist, hist)
    previous_small = prev_state.get("small")
    frame_difference = 0.0
    if previous_small is not None and previous_small.shape == small.shape:
        frame_difference = float(np.mean(cv2.absdiff(previous_small, small))) / 255.0
    histogram_difference = 0.0
    if prev_state.get("hist") is not None:
        histogram_difference = max(
            0.0, min(1.0, float(cv2.compareHist(prev_state["hist"], hist, cv2.HISTCMP_BHATTACHARYYA)))
        )
    edges = cv2.Canny(small, 100, 200)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    contrast = float(np.std(gray)) / 127.5
    local_motion_proxy = frame_difference
    prev_state["small"] = small
    prev_state["hist"] = hist
    return {
        "frame_difference": _finite(frame_difference),
        "histogram_difference": _finite(histogram_difference),
        "edge_density": _finite(edge_density),
        "contrast": _finite(contrast),
        "local_motion_proxy": _finite(local_motion_proxy),
        "core_similarity_histogram": 0.0,
    }


def _sample_side(
    capture: Any,
    side: str,
    start_sec: float,
    end_sec: float,
    window_sec: float,
    stride_sec: float,
    duration_sec: float,
) -> list[dict[str, Any]]:
    """Sample evidence points around one boundary (outside then inside)."""
    if side == "left":
        lo = max(0.0, start_sec - window_sec)
        hi = min(duration_sec, start_sec + window_sec)
    else:
        lo = max(0.0, end_sec - window_sec)
        hi = min(duration_sec, end_sec + window_sec)
    if hi <= lo:
        return []
    times: list[float] = []
    t = lo
    while t <= hi + 1e-9:
        times.append(round(t, 6))
        t += stride_sec
    prev_state: dict[str, Any] = {}
    samples: list[dict[str, Any]] = []
    for t in times:
        signals = _sample_frame_signals(capture, t, prev_state=prev_state)
        if signals is None:
            continue
        inside = lo <= t < hi and (
            (t >= start_sec) if side == "left" else (t <= end_sec)
        )
        samples.append({"t": t, "side": side, "inside": bool(inside), **signals})
    return samples


def _candidate_core_reference(
    capture: Any, start_sec: float, end_sec: float
) -> dict[str, float] | None:
    """Average signals over the deterministic middle 30% of the candidate."""
    duration = end_sec - start_sec
    core_start = start_sec + duration * 0.35
    core_end = start_sec + duration * 0.65
    stride = max(0.25, (core_end - core_start) / 8.0)
    prev_state: dict[str, Any] = {}
    rows: list[dict[str, float]] = []
    t = core_start
    while t <= core_end + 1e-9:
        signals = _sample_frame_signals(capture, t, prev_state=prev_state)
        if signals is not None:
            rows.append(signals)
        t += stride
    if not rows:
        return None
    reference = {}
    for key in rows[0]:
        reference[key] = _finite(sum(row[key] for row in rows) / len(rows))
    return reference


def _cosine_similarity(a: Mapping[str, float], b: Mapping[str, float], keys: list[str]) -> float:
    va = np.asarray([_finite(a.get(k)) for k in keys], dtype=float)
    vb = np.asarray([_finite(b.get(k)) for k in keys], dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if not math.isfinite(denom) or denom <= 0.0:
        return 0.0
    return _finite(float(np.dot(va, vb) / denom))


def _robust_z(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return []
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    if mad <= 0.0:
        return [0.0 for _ in values]
    return [_finite((v - median) / (1.4826 * mad)) for v in values]


def build_local_boundary_evidence(
    capture: Any,
    start_sec: float,
    end_sec: float,
    config: Mapping[str, Any],
    *,
    duration_sec: float,
) -> dict[str, Any]:
    """Build left/right local evidence curves plus core reference."""
    window = _finite(config["boundary_window_sec"])
    stride = _finite(config["sample_stride_sec"])
    left = _sample_side(capture, "left", start_sec, end_sec, window, stride, duration_sec)
    right = _sample_side(capture, "right", start_sec, end_sec, window, stride, duration_sec)
    core = _candidate_core_reference(capture, start_sec, end_sec)
    keys = [
        "frame_difference",
        "histogram_difference",
        "edge_density",
        "contrast",
        "local_motion_proxy",
    ]
    for samples in (left, right):
        for row in samples:
            if core is not None:
                row["core_similarity_histogram"] = _cosine_similarity(row, core, keys)
            else:
                row["core_similarity_histogram"] = 0.0
    return {"left": left, "right": right, "core_reference": core}


def _side_decision(
    samples: list[dict[str, Any]],
    inside_flag: bool,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Decide TRIM/KEEP/EXPAND for one side from its evidence curve.

    ``inside_flag`` is True for the left side (inside means t >= start) and
    False for the right side (inside means t <= end).
    """
    inside_rows = [row for row in samples if row["inside"]]
    outside_rows = [row for row in samples if not row["inside"]]
    if not inside_rows or not outside_rows:
        return {"action": "KEEP", "confidence": 0.0, "delta_sec": 0.0, "supported": 0}

    core_scores = np.asarray([row["core_similarity_histogram"] for row in inside_rows], dtype=float)
    if core_scores.size == 0 or not np.isfinite(core_scores).all():
        return {"action": "KEEP", "confidence": 0.0, "delta_sec": 0.0, "supported": 0}
    core_level = float(np.median(core_scores))
    outside_scores = np.asarray(
        [row["core_similarity_histogram"] for row in outside_rows], dtype=float
    )
    if float(np.max(np.abs(core_scores))) <= 1e-9 and float(np.max(np.abs(outside_scores))) <= 1e-9:
        return {"action": "KEEP", "confidence": 0.0, "delta_sec": 0.0, "supported": 0}
    persistence_k = int(config["persistence_k"])
    if inside_flag:
        ordered = inside_rows
    else:
        ordered = list(reversed(inside_rows))

    weak_run = 0
    for row in ordered:
        if row["core_similarity_histogram"] < core_level * 0.5:
            weak_run += 1
        else:
            break

    outside_ordered = outside_rows if inside_flag else list(reversed(outside_rows))
    strong_run = 0
    for row in outside_ordered:
        if row["core_similarity_histogram"] >= core_level:
            strong_run += 1
        else:
            break

    trim_supported = weak_run >= persistence_k
    expand_supported = strong_run >= persistence_k
    if trim_supported and expand_supported:
        return {"action": "KEEP", "confidence": 0.0, "delta_sec": 0.0, "supported": 0}
    if trim_supported:
        confidence = min(1.0, weak_run / max(1.0, float(len(ordered))))
        return {"action": "TRIM", "confidence": confidence, "delta_sec": 0.0, "supported": weak_run}
    if expand_supported:
        confidence = min(1.0, strong_run / max(1.0, float(len(outside_ordered))))
        return {
            "action": "EXPAND",
            "confidence": confidence,
            "delta_sec": 0.0,
            "supported": strong_run,
        }
    return {"action": "KEEP", "confidence": 0.0, "delta_sec": 0.0, "supported": 0}


def decide_left_boundary_action(evidence: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    return _side_decision(list(evidence["left"]), True, config)


def decide_right_boundary_action(evidence: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    return _side_decision(list(evidence["right"]), False, config)


def apply_asymmetric_boundary_action(
    start_sec: float,
    end_sec: float,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    video_id: str,
    candidate_id: str,
) -> dict[str, Any]:
    """Apply per-side actions with all safety guards; fallback on any failure."""
    original_start = _finite(start_sec)
    original_end = _finite(end_sec)

    def identity(action: str = "identity", reason: str | None = None) -> dict[str, Any]:
        return {
            "decision": "IDENTITY_FALLBACK" if action == "fallback_identity" else "IDENTITY",
            "decision_rule": "br2.identity" if action == "identity" else "br2.fallback",
            "confidence": 1.0,
            "refined_start_sec": original_start,
            "refined_end_sec": original_end,
            "boundary_reason": "br2 identity" if action == "identity" else f"br2 fallback: {reason}",
            "decision_detail": {
                "action": action,
                "fallback_reason": reason,
                "left_action": "KEEP" if action in ("identity", "fallback_identity") else left.get("action", "KEEP"),
                "right_action": "KEEP" if action in ("identity", "fallback_identity") else right.get("action", "KEEP"),
                "left_delta_sec": 0.0,
                "right_delta_sec": 0.0,
                "left_confidence": _finite(left.get("confidence", 0.0)),
                "right_confidence": _finite(right.get("confidence", 0.0)),
                "num_left_samples": _finite(left.get("num_samples", 0.0)),
                "num_right_samples": _finite(right.get("num_samples", 0.0)),
            },
        }

    if not _all_finite_local([original_start, original_end]):
        return identity("fallback_identity", "nonfinite_original_interval")
    original_duration = original_end - original_start
    if original_duration <= 0:
        return identity("fallback_identity", "nonpositive_duration")

    min_duration_for_windows = 2 * _finite(config["boundary_window_sec"])
    if original_duration < min_duration_for_windows * 0.5:
        return identity("fallback_identity", _FALLBACK_SHORT_CANDIDATE)

    left_action = left.get("action", "KEEP")
    right_action = right.get("action", "KEEP")
    confidence_min = _finite(config["action_confidence_min"])
    if left.get("confidence", 0.0) < confidence_min:
        left_action = "KEEP"
    if right.get("confidence", 0.0) < confidence_min:
        right_action = "KEEP"
    if left_action == "KEEP" and right_action == "KEEP":
        return identity()

    max_trim = _finite(config["max_trim_each_side_sec"])
    max_expand = _finite(config["max_expand_each_side_sec"])
    trim_margin = _finite(config["trim_margin_sec"])
    expand_margin = _finite(config["expand_margin_sec"])

    left_delta = 0.0
    if left_action == "TRIM":
        left_delta = min(max_trim, trim_margin + max(0.0, left.get("supported", 0) - 1) * 0.5)
        left_delta = min(left_delta, original_duration / 2.0)
    elif left_action == "EXPAND":
        left_delta = -min(max_expand, expand_margin + max(0.0, left.get("supported", 0) - 1) * 0.5)

    right_delta = 0.0
    if right_action == "TRIM":
        trim_amount = min(max_trim, trim_margin + max(0.0, right.get("supported", 0) - 1) * 0.5)
        trim_amount = min(trim_amount, original_duration / 2.0)
        right_delta = -trim_amount
    elif right_action == "EXPAND":
        right_delta = min(max_expand, expand_margin + max(0.0, right.get("supported", 0) - 1) * 0.5)

    refined_start = original_start + left_delta
    refined_end = original_end + right_delta
    refined_duration = refined_end - refined_start
    change_ratio = (abs(left_delta) + abs(right_delta)) / original_duration
    max_change = _finite(config["max_total_boundary_change_ratio"])

    if not _all_finite_local([refined_start, refined_end, refined_duration]):
        return identity("fallback_identity", _FALLBACK_NONFINITE)
    if refined_start < 0.0:
        return identity("fallback_identity", "refined_start_below_zero")
    if refined_duration < _finite(config["min_refined_duration_sec"]):
        return identity("fallback_identity", "refined_duration_below_minimum")
    if refined_start >= refined_end:
        return identity("fallback_identity", _FALLBACK_GUARD)
    if change_ratio > max_change:
        return identity("fallback_identity", "total_boundary_change_ratio_exceeds_cap")
    if refined_duration / original_duration < _finite(config["min_parent_overlap_ratio"]):
        return identity("fallback_identity", "parent_overlap_ratio_below_floor")
    if left_delta > max_trim + 1e-9 or -right_delta > max_trim + 1e-9:
        return identity("fallback_identity", "trim_exceeds_cap")
    if -left_delta > max_expand + 1e-9 or right_delta > max_expand + 1e-9:
        return identity("fallback_identity", "expand_exceeds_cap")
    if abs(left_delta) < 1e-9 and abs(right_delta) < 1e-9:
        return identity()

    return {
        "decision": "REFINE",
        "decision_rule": "br2.asymmetric_refine",
        "confidence": round(min(1.0, max(0.0, min(left.get("confidence", 0.0), right.get("confidence", 0.0)) or max(left.get("confidence", 0.0), right.get("confidence", 0.0)))), 6),
        "refined_start_sec": refined_start,
        "refined_end_sec": refined_end,
        "boundary_reason": "br2 asymmetric local-evidence refine",
        "decision_detail": {
            "action": "asymmetric_refine",
            "fallback_reason": None,
            "left_action": left_action,
            "right_action": right_action,
            "left_delta_sec": left_delta,
            "right_delta_sec": right_delta,
            "left_confidence": _finite(left.get("confidence", 0.0)),
            "right_confidence": _finite(right.get("confidence", 0.0)),
            "num_left_samples": _finite(left.get("num_samples", 0.0)),
            "num_right_samples": _finite(right.get("num_samples", 0.0)),
            "original_duration_sec": original_duration,
            "refined_duration_sec": refined_duration,
            "change_ratio": change_ratio,
        },
    }


def _all_finite_local(values: list[float]) -> bool:
    return all(math.isfinite(v) for v in values)


def propose_br2_boundary(
    capture: Any,
    candidate: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    video_id: str,
    duration_sec: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Propose one asymmetric refinement; returns (evidence rows, refinement)."""
    start = _finite(candidate["start_sec"])
    end = _finite(candidate["end_sec"])
    evidence = build_local_boundary_evidence(
        capture, start, end, config, duration_sec=duration_sec
    )
    left = decide_left_boundary_action(evidence, config)
    right = decide_right_boundary_action(evidence, config)
    left["num_samples"] = len(evidence["left"])
    right["num_samples"] = len(evidence["right"])
    refinement = apply_asymmetric_boundary_action(
        start,
        end,
        left,
        right,
        config,
        video_id=video_id,
        candidate_id=str(candidate["merged_candidate_id"]),
    )
    rows = list(evidence["left"]) + list(evidence["right"])
    for row in rows:
        row["video_id"] = video_id
        row["candidate_id"] = str(candidate["merged_candidate_id"])
    return rows, refinement


def run_br2_for_role(
    cache_manifest: Mapping[str, Any],
    role_manifest: Mapping[str, Any],
    cache_records_by_id: Mapping[str, Mapping[str, Any]],
    config_name: str,
    config: Mapping[str, Any],
    propose_fn,
) -> dict[str, Any]:
    """Build a boundary-refinement-result compatible payload for one BR-2 config."""
    cache_hash = cache_manifest.get("global_semantic_sha256")
    if cache_hash != role_manifest.get("source_cache_global_hash"):
        raise ValueError("role manifest does not match source cache")
    if not config_name.startswith(BR2_REFINER_PREFIX):
        raise ValueError(f"unexpected BR-2 config name: {config_name}")
    config_payload = dict(config)
    config_payload["config_name"] = config_name
    config_hash = semantic_sha256(config_payload)

    records: list[dict[str, Any]] = []
    candidate_total = 0
    decision_counts = {"IDENTITY": 0, "REFINE": 0, "IDENTITY_FALLBACK": 0}
    for identity in role_manifest["records"]:
        video_id = identity["video_id"]
        record = cache_records_by_id.get(video_id)
        if record is None:
            raise ValueError(f"role video is missing from cache: {video_id}")
        if record.get("split") != identity["split"]:
            raise ValueError(f"role/cache split mismatch for {video_id}")
        candidates = record.get("merged_candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"cache merged_candidates missing for {video_id}")
        refinements: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_id = str(candidate["merged_candidate_id"])
            _evidence_rows, refinement = propose_fn(video_id, candidate, record)
            refinement["merged_candidate_id"] = candidate_id
            refinement["original_start_sec"] = _finite(candidate["start_sec"])
            refinement["original_end_sec"] = _finite(candidate["end_sec"])
            refinement["parent_candidate_semantic_sha256"] = semantic_sha256(
                {"video_id": video_id, **candidate}
            )
            refinements.append(refinement)
            candidate_total += 1
            decision_counts[refinement["decision"]] += 1
        video_record = {
            "video_id": video_id,
            "split": identity["split"],
            "source_record_semantic_sha256": record["semantic_sha256"],
            "candidate_refinements": refinements,
        }
        video_record["video_refinement_semantic_sha256"] = semantic_sha256(video_record)
        records.append(video_record)

    result: dict[str, Any] = {
        "boundary_result_schema_version": BR2_RESULT_SCHEMA_VERSION,
        "source_cache_global_hash": cache_hash,
        "role": role_manifest["role"],
        "role_manifest_hash": role_manifest["semantic_sha256"],
        "refiner_name": config_name,
        "refiner_version": BR2_REFINER_VERSION,
        "refiner_config": config_payload,
        "refiner_config_hash": config_hash,
        "record_count": len(records),
        "input_candidate_count": candidate_total,
        "decision_counts": decision_counts,
        "records": records,
    }
    from .boundary_refinement import _check_result_leakage

    _check_result_leakage(result)
    result["boundary_refinement_semantic_hash"] = semantic_sha256(result)
    return result
