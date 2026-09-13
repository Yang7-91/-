"""Stage 4-new SBC-0 semantic boundary classifier separability probe.

Diagnostic only.  Boundary-side samples are built from the audited BHD-0.1
oracle labels (weak-reference derived).  A fixed prompt asks a VLM to predict
TRIM / KEEP / EXPAND for one boundary of a coarse candidate from a local clip.
Nothing here creates deployable predictions or modifies frozen artifacts.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

SBC0_SCHEMA_VERSION = "aic.stage4.sbc0_semantic_boundary_classifier_probe/v1"
SBC0_ACTIONS = ("TRIM", "KEEP", "EXPAND")
SBC0_SIDES = ("left", "right")


@dataclass(frozen=True, slots=True)
class BoundarySideSample:
    sample_id: str
    video_id: str
    candidate_id: str
    side: str
    oracle_action: str
    candidate_start_sec: float
    candidate_end_sec: float
    boundary_sec: float
    candidate_duration_sec: float


@dataclass(frozen=True, slots=True)
class SemanticBoundaryPrediction:
    sample_id: str
    predicted_action: str | None
    confidence: float | None
    rationale_short: str | None
    parse_ok: bool
    error: str | None = None


class SemanticBoundaryParseError(ValueError):
    """Raised when a model response does not match the required JSON schema."""


def load_oracle_boundary_labels(path: Any) -> list[dict[str, Any]]:
    """Load BHD-0.1 boundary_oracle_labels.jsonl rows."""
    from pathlib import Path

    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_stratified_boundary_samples(
    labels: Iterable[Mapping[str, Any]],
    cache_records_by_id: Mapping[str, Mapping[str, Any]],
    *,
    max_samples_total: int,
    max_per_label_per_side: int,
    seed: int,
) -> tuple[list[BoundarySideSample], dict[str, Any]]:
    """Deterministically build stratified boundary-side samples from labels.

    Only ``dev`` split records are eligible; any missing candidate or non-dev
    row is skipped and counted.  Sampling is grouped by (side, oracle action)
    with per-group caps and a fixed seed.
    """
    grouped: dict[tuple[str, str], list[tuple[str, str, float, float]]] = {}
    skipped = {"missing_video": 0, "missing_candidate": 0, "non_dev": 0, "unknown_action": 0}
    for row in labels:
        video_id = row.get("video_id")
        candidate_id = row.get("candidate_id")
        record = cache_records_by_id.get(str(video_id))
        if record is None:
            skipped["missing_video"] += 1
            continue
        if record.get("split") != "dev":
            skipped["non_dev"] += 1
            continue
        candidate = next(
            (c for c in record.get("merged_candidates", []) if str(c.get("merged_candidate_id")) == str(candidate_id)),
            None,
        )
        if candidate is None:
            skipped["missing_candidate"] += 1
            continue
        start = float(candidate["start_sec"])
        end = float(candidate["end_sec"])
        for side in SBC0_SIDES:
            action = row.get(f"{side}_action")
            if action not in SBC0_ACTIONS:
                skipped["unknown_action"] += 1
                continue
            boundary = start if side == "left" else end
            grouped.setdefault((side, action), []).append(
                (str(video_id), str(candidate_id), boundary, end - start)
            )
    rng = random.Random(seed)
    selected: list[BoundarySideSample] = []
    group_report: dict[str, Any] = {}
    for side in SBC0_SIDES:
        for action in SBC0_ACTIONS:
            members = sorted(grouped.get((side, action), []))
            cap = min(len(members), max_per_label_per_side)
            chosen = members if cap == len(members) else sorted(rng.sample(members, cap))
            for video_id, candidate_id, boundary, duration in chosen:
                selected.append(
                    BoundarySideSample(
                        sample_id=f"{video_id}|{candidate_id}|{side}",
                        video_id=video_id,
                        candidate_id=candidate_id,
                        side=side,
                        oracle_action=action,
                        candidate_start_sec=0.0,
                        candidate_end_sec=0.0,
                        boundary_sec=boundary,
                        candidate_duration_sec=duration,
                    )
                )
            group_report[f"{side}:{action}"] = {"available": len(members), "selected": cap}
    if len(selected) > max_samples_total:
        selected = selected[:max_samples_total]
    selected = _fill_candidate_bounds(selected, cache_records_by_id)
    report = {
        "total_samples": len(selected),
        "left_count": sum(1 for s in selected if s.side == "left"),
        "right_count": sum(1 for s in selected if s.side == "right"),
        "label_counts": {
            action: sum(1 for s in selected if s.oracle_action == action) for action in SBC0_ACTIONS
        },
        "video_count": len({s.video_id for s in selected}),
        "groups": group_report,
        "skipped": skipped,
        "seed": seed,
        "heldout_accessed": False,
    }
    return selected, report


def _fill_candidate_bounds(
    samples: list[BoundarySideSample],
    cache_records_by_id: Mapping[str, Mapping[str, Any]],
) -> list[BoundarySideSample]:
    fixed = []
    for sample in samples:
        record = cache_records_by_id[sample.video_id]
        candidate = next(
            c for c in record["merged_candidates"] if str(c["merged_candidate_id"]) == sample.candidate_id
        )
        fixed.append(
            BoundarySideSample(
                sample_id=sample.sample_id,
                video_id=sample.video_id,
                candidate_id=sample.candidate_id,
                side=sample.side,
                oracle_action=sample.oracle_action,
                candidate_start_sec=float(candidate["start_sec"]),
                candidate_end_sec=float(candidate["end_sec"]),
                boundary_sec=sample.boundary_sec,
                candidate_duration_sec=sample.candidate_duration_sec,
            )
        )
    return fixed


def build_boundary_clip_bounds(
    sample: BoundarySideSample, duration_sec: float, window_sec: float
) -> dict[str, float]:
    """Clip window around the boundary, clamped to the video duration."""
    clip_start = max(0.0, sample.boundary_sec - window_sec)
    clip_end = min(duration_sec, sample.boundary_sec + window_sec)
    if clip_end <= clip_start:
        raise ValueError(f"empty clip window for sample {sample.sample_id}")
    return {
        "clip_start_sec": clip_start,
        "clip_end_sec": clip_end,
        "boundary_sec": sample.boundary_sec,
        "boundary_offset_in_clip_sec": sample.boundary_sec - clip_start,
        "clip_duration_sec": clip_end - clip_start,
    }


def render_boundary_prompt(
    template: str,
    sample: BoundarySideSample,
    clip_bounds: Mapping[str, float],
) -> str:
    """Fill the fixed prompt template with deterministic sample context."""
    if sample.side == "left":
        direction = (
            "For this LEFT boundary, moving inward means moving later (to the right), "
            "and moving outward means moving earlier (to the left)."
        )
    else:
        direction = (
            "For this RIGHT boundary, moving inward means moving earlier (to the left), "
            "and moving outward means moving later (to the right)."
        )
    return (
        template.replace("{side}", sample.side)
        .replace("{boundary_offset_in_clip_sec}", f"{clip_bounds['boundary_offset_in_clip_sec']:.2f}")
        .replace("{clip_duration_sec}", f"{clip_bounds['clip_duration_sec']:.2f}")
        .replace("{candidate_duration_sec}", f"{sample.candidate_duration_sec:.2f}")
        .replace("{side_direction}", direction)
    )


def parse_semantic_boundary_response(content: str) -> dict[str, Any]:
    """Parse the model response as strict JSON with the fixed schema."""
    if not isinstance(content, str) or not content.strip():
        raise SemanticBoundaryParseError("empty response")
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SemanticBoundaryParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SemanticBoundaryParseError("response is not a JSON object")
    action = payload.get("action")
    if action not in SBC0_ACTIONS:
        raise SemanticBoundaryParseError(f"invalid action: {action!r}")
    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)):
        raise SemanticBoundaryParseError(f"invalid confidence: {confidence!r}")
    if not 0.0 <= float(confidence) <= 1.0:
        raise SemanticBoundaryParseError(f"confidence out of range: {confidence!r}")
    rationale = payload.get("rationale_short", "")
    if not isinstance(rationale, str):
        raise SemanticBoundaryParseError("rationale_short must be a string")
    return {
        "action": action,
        "confidence": float(confidence),
        "rationale_short": rationale[:200],
    }


def _one_vs_rest_auc(score_pos: list[float], score_neg: list[float]) -> float:
    if not score_pos or not score_neg:
        return 0.5
    wins = 0.0
    for p in score_pos:
        for n in score_neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(score_pos) * len(score_neg))


def _macro_f1(confusion: Mapping[str, Mapping[str, int]]) -> float:
    f1s = []
    for label in SBC0_ACTIONS:
        tp = confusion.get(label, {}).get(label, 0)
        fn = sum(confusion.get(label, {}).get(other, 0) for other in SBC0_ACTIONS if other != label)
        fp = sum(confusion.get(other, {}).get(label, 0) for other in SBC0_ACTIONS if other != label)
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        f1s.append((precision, recall, f1))
    if not f1s:
        return 0.0
    return sum(f1 for _, _, f1 in f1s) / len(f1s)


def compute_sbc0_metrics(
    samples: list[BoundarySideSample],
    predictions: list[SemanticBoundaryPrediction],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Confusion, accuracy, macro-F1, per-class P/R, per-side AUC-like."""
    sample_ids = {s.sample_id for s in samples}
    prediction_by_id = {p.sample_id: p for p in predictions if p.sample_id in sample_ids}
    confusion = {label: {other: 0 for other in SBC0_ACTIONS} for label in SBC0_ACTIONS}
    per_class_correct: dict[str, int] = {label: 0 for label in SBC0_ACTIONS}
    per_class_total: dict[str, int] = {label: 0 for label in SBC0_ACTIONS}
    side_instances: dict[str, list[tuple[str, str, float]]] = {side: [] for side in SBC0_SIDES}
    parsed = 0
    for sample in samples:
        prediction = prediction_by_id.get(sample.sample_id)
        if prediction is None or not prediction.parse_ok or prediction.predicted_action is None:
            continue
        parsed += 1
        confusion[sample.oracle_action][prediction.predicted_action] += 1
        per_class_total[sample.oracle_action] += 1
        if prediction.predicted_action == sample.oracle_action:
            per_class_correct[sample.oracle_action] += 1
        side_instances[sample.side].append(
            (sample.oracle_action, prediction.predicted_action, float(prediction.confidence or 0.0))
        )
    total = len(samples)
    correct = sum(per_class_correct.values())
    per_class = {}
    for label in SBC0_ACTIONS:
        tp = confusion[label][label]
        fn = sum(confusion[label][other] for other in SBC0_ACTIONS if other != label)
        fp = sum(confusion[other][label] for other in SBC0_ACTIONS if other != label)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        per_class[label] = {
            "support": per_class_total[label],
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0,
        }
    side_metrics = {}
    for side in SBC0_SIDES:
        aucs = []
        instances = side_instances[side]
        for label in SBC0_ACTIONS:
            positives = [
                confidence if predicted == label else 0.0
                for oracle, predicted, confidence in instances
                if oracle == label
            ]
            negatives = [
                confidence if predicted == label else 0.0
                for oracle, predicted, confidence in instances
                if oracle != label
            ]
            if not positives or not negatives:
                continue
            auc = _one_vs_rest_auc(positives, negatives)
            aucs.append(max(auc, 1.0 - auc))
        side_metrics[side] = {
            "support": len(instances),
            "auc_like": sum(aucs) / len(aucs) if aucs else 0.5,
        }
    rules = protocol["decision_rules"]
    schema_success_rate = parsed / total if total else 0.0
    macro_f1 = _macro_f1(confusion)
    max_auc = max(side_metrics["left"]["auc_like"], side_metrics["right"]["auc_like"])
    actionable = (
        schema_success_rate >= float(rules["schema_success_rate_min"])
        and max_auc >= float(rules["meaningful_auc_like_min"])
        and macro_f1 >= float(rules["meaningful_macro_f1_min"])
    )
    return {
        "stage": "Stage 4-new / SBC-0",
        "schema_version": SBC0_SCHEMA_VERSION,
        "diagnostic_only": True,
        "deployable_method": False,
        "num_samples": total,
        "num_parsed": parsed,
        "schema_success_rate": schema_success_rate,
        "accuracy": correct / parsed if parsed else 0.0,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "left": {"accuracy": _side_accuracy("left", samples, prediction_by_id), "macro_f1": None, "auc_like": side_metrics["left"]["auc_like"]},
        "right": {"accuracy": _side_accuracy("right", samples, prediction_by_id), "macro_f1": None, "auc_like": side_metrics["right"]["auc_like"]},
        "confusion_matrix": confusion,
        "decision": "ACTIONABLE_SIGNAL" if actionable else "NOT_ACTIONABLE",
        "rule_checks": {
            "schema_success_rate_pass": schema_success_rate >= float(rules["schema_success_rate_min"]),
            "auc_like_pass": max_auc >= float(rules["meaningful_auc_like_min"]),
            "macro_f1_pass": macro_f1 >= float(rules["meaningful_macro_f1_min"]),
        },
        "heldout_accessed": False,
        "hard_run": False,
        "training": False,
    }


def _side_accuracy(
    side: str,
    samples: list[BoundarySideSample],
    prediction_by_id: Mapping[str, SemanticBoundaryPrediction],
) -> float:
    correct = 0
    total = 0
    for sample in samples:
        if sample.side != side:
            continue
        prediction = prediction_by_id.get(sample.sample_id)
        if prediction is None or not prediction.parse_ok:
            continue
        total += 1
        if prediction.predicted_action == sample.oracle_action:
            correct += 1
    return correct / total if total else 0.0


def sample_to_row(sample: BoundarySideSample) -> dict[str, Any]:
    row = asdict(sample)
    row["diagnostic_only"] = True
    row["deployable_method"] = False
    row["heldout_accessed"] = False
    return row
