"""Stage 4-new SBA-1 conservative shot-boundary snap (deployable).

Snaps each frozen candidate boundary to a nearby detected shot boundary using
only video-derived signals and preregistered rules.  No weak reference, oracle,
model call, training, or frozen-artifact modification is involved; every guard
failure falls back to identity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .boundary_headroom_diagnostic import (
    detect_shot_boundaries,
    shot_segments_from_boundaries,
)
from .candidate_selection import semantic_sha256

SBA1_RESULT_SCHEMA_VERSION = "aic.boundary-refinement-result/v1"
SBA1_REFINER_VERSION = "aic.sba1-shot-boundary-snap/v1"
SBA1_REFINER_PREFIX = "SBA-1"

_EPS = 1e-9

_FALLBACK_SHORT = "candidate_below_min_duration"
_FALLBACK_GUARD = "safety_guard_failure"
_FALLBACK_NONFINITE = "nonfinite_values"


@dataclass(frozen=True, slots=True)
class ShotBoundary:
    time_sec: float


@dataclass(frozen=True, slots=True)
class ShotSegment:
    start_sec: float
    end_sec: float


@dataclass(frozen=True, slots=True)
class ShotSnapDecision:
    action: str
    target_sec: float | None
    nearest_distance_sec: float | None
    delta_sec: float


def _finite(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def find_nearest_shot_boundary(
    boundary_sec: float,
    shot_boundaries: list[float],
    snap_window_sec: float,
    side: str,
    config: Mapping[str, Any],
) -> ShotSnapDecision:
    """Pick the snap target for one side using preregistered rules only."""
    allow_trim = bool(config.get("allow_trim", True))
    allow_expand = bool(config.get("allow_expand", True))
    inward_only = bool(config.get("inward_only", False)) or not allow_expand
    prefer_inner_tie = bool(config.get("prefer_inner_when_tie", True))

    candidates: list[tuple[float, float, int]] = []
    for shot in shot_boundaries:
        distance = abs(shot - boundary_sec)
        if distance > snap_window_sec + _EPS:
            continue
        if distance <= _EPS:
            continue
        if side == "left":
            inward = shot > boundary_sec
        else:
            inward = shot < boundary_sec
        if inward and not allow_trim:
            continue
        if (not inward) and inward_only:
            continue
        if (not inward) and not allow_expand:
            continue
        inward_priority = 0 if (inward and prefer_inner_tie) else 1
        candidates.append((distance, inward_priority, shot))  # type: ignore[arg-type]
    if not candidates:
        return ShotSnapDecision("KEEP", None, None, 0.0)
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    distance, _priority, shot = candidates[0]
    delta = shot - boundary_sec
    if side == "left":
        action = "EXPAND" if delta < -_EPS else ("TRIM" if delta > _EPS else "KEEP")
    else:
        action = "TRIM" if delta < -_EPS else ("EXPAND" if delta > _EPS else "KEEP")
    return ShotSnapDecision(action, shot, distance, delta)


def snap_one_boundary(
    boundary_sec: float,
    shot_boundaries: list[float],
    side: str,
    config: Mapping[str, Any],
) -> ShotSnapDecision:
    return find_nearest_shot_boundary(
        boundary_sec,
        shot_boundaries,
        _finite(config["snap_window_sec"]),
        side,
        config,
    )


def propose_shot_boundary_snap(
    candidate: Mapping[str, Any],
    shot_boundaries: list[float],
    config: Mapping[str, Any],
    *,
    duration_sec: float,
    video_id: str,
    candidate_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Propose one snap refinement; returns (decision_detail, refinement)."""
    original_start = _finite(candidate["start_sec"])
    original_end = _finite(candidate["end_sec"])
    original_duration = original_end - original_start
    num_inside = sum(
        1 for shot in shot_boundaries if original_start - _EPS <= shot <= original_end + _EPS
    )
    detail: dict[str, Any] = {
        "action": "identity",
        "fallback_reason": None,
        "left_delta_sec": 0.0,
        "right_delta_sec": 0.0,
        "left_action": "KEEP",
        "right_action": "KEEP",
        "left_nearest_shot_distance_sec": None,
        "right_nearest_shot_distance_sec": None,
        "num_shot_boundaries_in_candidate": num_inside,
    }
    fallback = {
        "decision": "IDENTITY_FALLBACK",
        "decision_rule": "sba1.fallback",
        "confidence": 1.0,
        "refined_start_sec": original_start,
        "refined_end_sec": original_end,
        "boundary_reason": "sba1 fallback",
        "decision_detail": detail,
    }
    identity = {
        "decision": "IDENTITY",
        "decision_rule": "sba1.identity",
        "confidence": 1.0,
        "refined_start_sec": original_start,
        "refined_end_sec": original_end,
        "boundary_reason": "sba1 identity",
        "decision_detail": detail,
    }
    if not all(math.isfinite(v) for v in (original_start, original_end, duration_sec)):
        detail["action"] = "fallback_identity"
        detail["fallback_reason"] = _FALLBACK_NONFINITE
        return detail, fallback
    if original_duration <= 0:
        detail["action"] = "fallback_identity"
        detail["fallback_reason"] = _FALLBACK_NONFINITE
        return detail, fallback
    if original_duration < _finite(config["min_candidate_duration_sec"]):
        detail["action"] = "fallback_identity"
        detail["fallback_reason"] = _FALLBACK_SHORT
        return detail, fallback

    left = snap_one_boundary(original_start, shot_boundaries, "left", config)
    right = snap_one_boundary(original_end, shot_boundaries, "right", config)
    detail["left_nearest_shot_distance_sec"] = left.nearest_distance_sec
    detail["right_nearest_shot_distance_sec"] = right.nearest_distance_sec
    if left.action == "KEEP" and right.action == "KEEP":
        return detail, identity

    refined_start = left.target_sec if left.target_sec is not None else original_start
    refined_end = right.target_sec if right.target_sec is not None else original_end
    if not all(math.isfinite(v) for v in (refined_start, refined_end)):
        detail["action"] = "fallback_identity"
        detail["fallback_reason"] = _FALLBACK_NONFINITE
        return detail, fallback
    left_delta = refined_start - original_start
    right_delta = refined_end - original_end
    refined_duration = refined_end - refined_start
    change_ratio = (abs(left_delta) + abs(right_delta)) / original_duration
    max_trim = _finite(config["max_trim_each_side_sec"])
    max_expand = _finite(config["max_expand_each_side_sec"])

    def fail(reason: str) -> tuple[dict[str, Any], dict[str, Any]]:
        detail["action"] = "fallback_identity"
        detail["fallback_reason"] = reason
        detail["left_delta_sec"] = 0.0
        detail["right_delta_sec"] = 0.0
        return detail, fallback

    if refined_start < -_EPS or refined_end > duration_sec + _EPS:
        return fail("refined_outside_video")
    if refined_start >= refined_end:
        return fail("invalid_interval")
    if refined_duration < _finite(config["min_refined_duration_sec"]) - _EPS:
        return fail("refined_duration_below_minimum")
    if refined_duration / original_duration < _finite(config["min_parent_overlap_ratio"]) - _EPS:
        return fail("parent_overlap_below_floor")
    if change_ratio > _finite(config["max_total_boundary_change_ratio"]) + _EPS:
        return fail("total_boundary_change_ratio_exceeds_cap")
    if left_delta > max_trim + _EPS or -right_delta > max_trim + _EPS:
        return fail("trim_exceeds_cap")
    if -left_delta > max_expand + _EPS or right_delta > max_expand + _EPS:
        return fail("expand_exceeds_cap")
    if abs(left_delta) <= _EPS and abs(right_delta) <= _EPS:
        return detail, identity

    detail["action"] = "shot_snap"
    detail["left_delta_sec"] = left_delta
    detail["right_delta_sec"] = right_delta
    detail["left_action"] = "TRIM" if left_delta > _EPS else ("EXPAND" if left_delta < -_EPS else "KEEP")
    detail["right_action"] = "TRIM" if right_delta < -_EPS else ("EXPAND" if right_delta > _EPS else "KEEP")
    refinement = {
        "decision": "REFINE",
        "decision_rule": "sba1.snap",
        "confidence": 1.0,
        "refined_start_sec": refined_start,
        "refined_end_sec": refined_end,
        "boundary_reason": "sba1 shot-boundary snap",
        "decision_detail": detail,
    }
    return detail, refinement


def run_sba1_for_role(
    cache_manifest: Mapping[str, Any],
    role_manifest: Mapping[str, Any],
    cache_records_by_id: Mapping[str, Mapping[str, Any]],
    config_name: str,
    config: Mapping[str, Any],
    shot_boundaries_by_video: Mapping[str, list[float]],
) -> dict[str, Any]:
    """Build a boundary-refinement-result compatible payload for one config."""
    cache_hash = cache_manifest.get("global_semantic_sha256")
    if cache_hash != role_manifest.get("source_cache_global_hash"):
        raise ValueError("role manifest does not match source cache")
    if not config_name.startswith(SBA1_REFINER_PREFIX):
        raise ValueError(f"unexpected SBA-1 config name: {config_name}")
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
        boundaries = shot_boundaries_by_video.get(video_id, [])
        refinements: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_id = str(candidate["merged_candidate_id"])
            _detail, refinement = propose_shot_boundary_snap(
                candidate,
                boundaries,
                config,
                duration_sec=float(record["duration_sec"]),
                video_id=video_id,
                candidate_id=candidate_id,
            )
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
        "boundary_result_schema_version": SBA1_RESULT_SCHEMA_VERSION,
        "source_cache_global_hash": cache_hash,
        "role": role_manifest["role"],
        "role_manifest_hash": role_manifest["semantic_sha256"],
        "refiner_name": config_name,
        "refiner_version": SBA1_REFINER_VERSION,
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
