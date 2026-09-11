"""Stage 4-new SABR-1.1 conservative boundary proposal over saliency bins.

Proposes trimmed boundaries inside frozen merged candidates using bin-level
saliency features from :mod:`saliency_anchor`.  The proposal never expands a
candidate, never splits or merges candidates, and falls back to identity
whenever any conservative guard fails.  No model is called and no reference,
audit, or Heldout information is read.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .candidate_selection import semantic_sha256
from .saliency_anchor import SIGNAL_WEIGHTS

SABR_REFINER_VERSION = "aic.sabr1-boundary/v1"
SABR_REFINER_PREFIX = "SABR-1.1"
SABR_RESULT_SCHEMA_VERSION = "aic.boundary-refinement-result/v1"

_FALLBACK_SHORT_DURATION = "candidate_duration_below_floor"
_FALLBACK_TOO_FEW_BINS = "too_few_effective_bins"
_FALLBACK_CONSTANT_SALIENCY = "saliency_approximately_constant"
_FALLBACK_WEAK_PEAK = "peak_prominence_below_threshold"
_FALLBACK_NO_CORE = "no_core_component_found"
_FALLBACK_MIN_REFINED_DURATION = "refined_duration_below_minimum"
_FALLBACK_SHRINK_RATIO = "total_shrink_ratio_exceeds_cap"
_FALLBACK_PARENT_OVERLAP = "parent_overlap_ratio_below_floor"
_FALLBACK_NONFINITE = "nonfinite_signal_values"


def _finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def _all_finite(values: list[float]) -> bool:
    return all(math.isfinite(v) for v in values)


def propose_conservative_boundary(
    candidate: Mapping[str, Any],
    bins: list[dict[str, Any]],
    config: Mapping[str, Any],
    *,
    duration_sec: float,
) -> dict[str, Any]:
    """Propose one conservative trimmed interval for one candidate.

    Returns a refinement record compatible with the Stage 4.4 boundary
    refinement result schema.  Any guard failure yields an identity decision
    with a recorded fallback reason; the candidate interval is never expanded.
    """
    original_start = _finite(candidate["start_sec"])
    original_end = _finite(candidate["end_sec"])
    identity = {
        "decision": "IDENTITY",
        "decision_rule": "sabr1.identity",
        "confidence": 1.0,
        "refined_start_sec": original_start,
        "refined_end_sec": original_end,
        "decision_detail": {
            "action": "identity",
            "fallback_reason": None,
            "trim_left_sec": 0.0,
            "trim_right_sec": 0.0,
            "peak_prominence": 0.0,
            "num_bins": len(bins),
        },
    }

    if not _all_finite([original_start, original_end, duration_sec]):
        raise ValueError(
            f"nonfinite candidate interval or duration: start={original_start} end={original_end}"
        )
    original_duration = original_end - original_start
    if original_duration <= 0:
        raise ValueError(f"nonpositive candidate duration: {original_duration}")

    bin_sec = _finite(config["bin_sec"])
    max_candidate_duration = _finite(config["max_candidate_duration_sec"])
    short_duration_margin = _finite(config["short_duration_margin_sec"])
    if original_duration <= max_candidate_duration or original_duration <= 2 * bin_sec + short_duration_margin:
        return _fallback(identity, _FALLBACK_SHORT_DURATION, original_start, original_end, len(bins))

    min_effective_bins = int(config["min_effective_bins"])
    effective = [b for b in bins if int(b.get("frame_count", 0)) > 0]
    if len(effective) < min_effective_bins:
        return _fallback(identity, _FALLBACK_TOO_FEW_BINS, original_start, original_end, len(bins))

    scores = [_finite(b.get("saliency_score")) for b in effective]
    if not _all_finite(scores):
        return _fallback(identity, _FALLBACK_NONFINITE, original_start, original_end, len(bins))
    if max(scores) - min(scores) <= _finite(config["constant_saliency_epsilon"]):
        return _fallback(identity, _FALLBACK_CONSTANT_SALIENCY, original_start, original_end, len(bins))

    array = np.asarray(scores, dtype=float)
    core_threshold = float(np.quantile(array, _finite(config["core_quantile"])))
    median_score = float(np.median(array))
    peak_prominence = float(array.max() - median_score)
    if peak_prominence < _finite(config["min_peak_prominence"]):
        return _fallback(identity, _FALLBACK_WEAK_PEAK, original_start, original_end, len(bins))

    above = [value >= core_threshold for value in scores]
    try:
        core_first = above.index(True)
        core_last = len(above) - 1 - above[::-1].index(True)
    except ValueError:
        return _fallback(identity, _FALLBACK_NO_CORE, original_start, original_end, len(bins))

    leading_bins = core_first
    trailing_bins = len(above) - 1 - core_last
    if leading_bins <= 0 and trailing_bins <= 0:
        return identity

    margin_sec = _finite(config["margin_sec"])
    max_trim_each_side = _finite(config["max_trim_each_side_sec"])
    trim_left = max(0.0, leading_bins * bin_sec - margin_sec)
    trim_right = max(0.0, trailing_bins * bin_sec - margin_sec)
    trim_left = min(trim_left, max_trim_each_side, original_duration / 2.0)
    trim_right = min(trim_right, max_trim_each_side, original_duration / 2.0)

    refined_start = original_start + trim_left
    refined_end = original_end - trim_right
    refined_duration = refined_end - refined_start

    min_refined_duration = _finite(config["min_refined_duration_sec"])
    max_total_shrink_ratio = _finite(config["max_total_shrink_ratio"])
    min_parent_overlap = _finite(config["min_parent_overlap_ratio"])

    if refined_duration < min_refined_duration:
        return _fallback(identity, _FALLBACK_MIN_REFINED_DURATION, original_start, original_end, len(bins))
    if (original_duration - refined_duration) / original_duration > max_total_shrink_ratio:
        return _fallback(identity, _FALLBACK_SHRINK_RATIO, original_start, original_end, len(bins))
    if refined_duration / original_duration < min_parent_overlap:
        return _fallback(identity, _FALLBACK_PARENT_OVERLAP, original_start, original_end, len(bins))
    if not _all_finite([refined_start, refined_end]):
        return _fallback(identity, _FALLBACK_NONFINITE, original_start, original_end, len(bins))
    if refined_start < original_start or refined_end > original_end or refined_start >= refined_end:
        return _fallback(identity, _FALLBACK_NONFINITE, original_start, original_end, len(bins))

    if trim_left <= 0.0 and trim_right <= 0.0:
        return identity

    return {
        "decision": "REFINE",
        "decision_rule": "sabr1.trim",
        "confidence": round(min(1.0, max(0.0, peak_prominence)), 6),
        "refined_start_sec": refined_start,
        "refined_end_sec": refined_end,
        "decision_detail": {
            "action": "trim",
            "fallback_reason": None,
            "trim_left_sec": trim_left,
            "trim_right_sec": trim_right,
            "peak_prominence": peak_prominence,
            "core_threshold": core_threshold,
            "leading_low_bins": leading_bins,
            "trailing_low_bins": trailing_bins,
            "num_bins": len(bins),
        },
    }


def _boundary_reason(refinement: Mapping[str, Any]) -> str:
    decision = refinement["decision"]
    if decision == "REFINE":
        return "sabr1 saliency-guided trim"
    if decision == "IDENTITY":
        return "sabr1 identity"
    reason = refinement["decision_detail"].get("fallback_reason")
    return f"sabr1 fallback: {reason}"


def _fallback(
    identity: dict[str, Any],
    reason: str,
    original_start: float,
    original_end: float,
    num_bins: int,
) -> dict[str, Any]:
    return {
        "decision": "IDENTITY_FALLBACK",
        "decision_rule": "sabr1.fallback",
        "confidence": 1.0,
        "refined_start_sec": original_start,
        "refined_end_sec": original_end,
        "decision_detail": {
            "action": "fallback_identity",
            "fallback_reason": reason,
            "trim_left_sec": 0.0,
            "trim_right_sec": 0.0,
            "peak_prominence": 0.0,
            "num_bins": num_bins,
        },
    }


def build_sabr11_result(
    cache_manifest: Mapping[str, Any],
    role_manifest: Mapping[str, Any],
    cache_records_by_id: Mapping[str, Mapping[str, Any]],
    config_name: str,
    config: Mapping[str, Any],
    propose_fn,
) -> dict[str, Any]:
    """Build a boundary-refinement-result compatible payload for one config.

    ``propose_fn(video_id, candidate, record)`` must return
    ``(bins, refinement)``; bin extraction is delegated so the caller controls
    video access.  The result keeps score/reason/source projections to the
    frozen cache via the shared replay layer.
    """
    cache_hash = cache_manifest.get("global_semantic_sha256")
    if cache_hash != role_manifest.get("source_cache_global_hash"):
        raise ValueError("role manifest does not match source cache")
    if not config_name.startswith(SABR_REFINER_PREFIX):
        raise ValueError(f"unexpected SABR config name: {config_name}")
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
            bins, refinement = propose_fn(video_id, candidate, record)
            refinement["merged_candidate_id"] = candidate_id
            refinement["original_start_sec"] = _finite(candidate["start_sec"])
            refinement["original_end_sec"] = _finite(candidate["end_sec"])
            refinement["boundary_reason"] = _boundary_reason(refinement)
            refinement["decision_detail"]["candidate_duration_sec"] = _finite(
                candidate["end_sec"]
            ) - _finite(candidate["start_sec"])
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
        "boundary_result_schema_version": SABR_RESULT_SCHEMA_VERSION,
        "source_cache_global_hash": cache_hash,
        "role": role_manifest["role"],
        "role_manifest_hash": role_manifest["semantic_sha256"],
        "refiner_name": config_name,
        "refiner_version": SABR_REFINER_VERSION,
        "refiner_config": config_payload,
        "refiner_config_hash": config_hash,
        "record_count": len(records),
        "input_candidate_count": candidate_total,
        "decision_counts": decision_counts,
        "records": records,
    }
    return result
