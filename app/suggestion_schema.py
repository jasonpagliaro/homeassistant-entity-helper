from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

SUGGESTION_SCHEMA_VERSION = "haev.automation.suggestion.v1"
CONCEPT_SCHEMA_VERSION = "haev.automation.concept.v2"
ALLOWED_RISK_LEVELS = {"low", "medium", "high"}
ALLOWED_TARGET_KINDS = {"new_automation", "existing_automation"}
ENTITY_ID_RE = re.compile(r"\b[a-z0-9_]+\.[a-z0-9_]+\b")


def extract_entity_ids(raw_value: Any) -> set[str]:
    if isinstance(raw_value, str):
        return set(match.group(0) for match in ENTITY_ID_RE.finditer(raw_value.lower()))
    if isinstance(raw_value, dict):
        values: set[str] = set()
        for item in raw_value.values():
            values.update(extract_entity_ids(item))
        return values
    if isinstance(raw_value, list):
        values = set()
        for item in raw_value:
            values.update(extract_entity_ids(item))
        return values
    return set()


def _as_score(raw_value: Any) -> float:
    if isinstance(raw_value, (int, float)):
        return max(0.0, min(1.0, float(raw_value)))
    return 0.0


def _clean_string_list(raw_value: Any, *, limit: int = 40) -> list[str]:
    if not isinstance(raw_value, list):
        return []
    cleaned: list[str] = []
    for item in raw_value:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if value and value not in cleaned:
            cleaned.append(value)
        if len(cleaned) >= limit:
            break
    return cleaned


def _is_automation_entity_id(value: str) -> bool:
    return bool(value.startswith("automation.") and "." in value)


def _calibration_components(item: dict[str, Any]) -> dict[str, float]:
    confidence = _as_score(item.get("confidence"))
    risk_level = str(item.get("risk_level", item.get("risk", "medium"))).strip().lower()
    risk_multiplier = {"low": 1.0, "medium": 0.93, "high": 0.82}.get(risk_level, 0.93)

    involved_entities = item.get("involved_entities")
    prerequisites = item.get("prerequisites")
    verification_outline = item.get("verification_outline")
    involved_count = len(involved_entities) if isinstance(involved_entities, list) else 0
    prerequisites_count = len(prerequisites) if isinstance(prerequisites, list) else 0
    verification_count = len(verification_outline) if isinstance(verification_outline, list) else 0

    # More moving parts means more room for implementation drift.
    complexity = min(1.0, (max(0, involved_count - 2) / 8.0) + (prerequisites_count / 10.0))
    complexity_factor = max(0.75, 1.0 - 0.20 * complexity)

    # Better articulated verification/prereqs increases trust in scores.
    evidence = min(1.0, (verification_count / 4.0) + (prerequisites_count / 6.0))
    evidence_factor = 0.85 + 0.15 * evidence

    confidence_factor = 0.65 + 0.35 * confidence
    calibration_multiplier = min(0.98, confidence_factor * complexity_factor * evidence_factor * risk_multiplier)

    return {
        "confidence": confidence,
        "risk_multiplier": risk_multiplier,
        "complexity": complexity,
        "complexity_factor": complexity_factor,
        "evidence": evidence,
        "evidence_factor": evidence_factor,
        "confidence_factor": confidence_factor,
        "calibration_multiplier": calibration_multiplier,
    }


def _calibrate_metric(raw_value: Any, calibration_multiplier: float) -> float:
    return _as_score(_as_score(raw_value) * calibration_multiplier)


# Legacy validator retained for backward compatibility with older data/tests.
def validate_suggestion_payload(
    payload: Any,
    known_entity_ids: set[str],
    max_patch_ops: int = 12,
) -> tuple[list[dict[str, Any]], list[str]]:
    if isinstance(payload, dict) and isinstance(payload.get("suggestions"), list):
        candidates = payload.get("suggestions")
    elif isinstance(payload, list):
        candidates = payload
    else:
        return [], ["response_missing_suggestions_array"]

    valid: list[dict[str, Any]] = []
    errors: list[str] = []

    for idx, item in enumerate(candidates):
        if not isinstance(item, dict):
            errors.append(f"item_{idx}:not_object")
            continue

        target_entity_id = str(item.get("target_entity_id", "")).strip().lower()
        if not target_entity_id.startswith("automation."):
            errors.append(f"item_{idx}:invalid_target_entity_id")
            continue

        summary = item.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            errors.append(f"item_{idx}:missing_summary")
            continue
        summary = summary.strip()

        raw_confidence = item.get("confidence")
        confidence: float
        if isinstance(raw_confidence, (int, float)):
            confidence = max(0.0, min(1.0, float(raw_confidence)))
        else:
            confidence = 0.0

        risk_level = str(item.get("risk_level", "")).strip().lower()
        if risk_level not in ALLOWED_RISK_LEVELS:
            errors.append(f"item_{idx}:invalid_risk_level")
            continue

        proposed_patch = item.get("proposed_patch")
        if not isinstance(proposed_patch, list) or not proposed_patch:
            errors.append(f"item_{idx}:missing_patch")
            continue
        if len(proposed_patch) > max_patch_ops:
            errors.append(f"item_{idx}:patch_too_large")
            continue
        if not all(isinstance(op, dict) for op in proposed_patch):
            errors.append(f"item_{idx}:patch_item_not_object")
            continue

        verification_steps = item.get("verification_steps")
        if not isinstance(verification_steps, list) or not verification_steps:
            errors.append(f"item_{idx}:missing_verification_steps")
            continue
        if not all(isinstance(step, str) and step.strip() for step in verification_steps):
            errors.append(f"item_{idx}:invalid_verification_step")
            continue
        if len(verification_steps) > 12:
            errors.append(f"item_{idx}:too_many_verification_steps")
            continue

        referenced_entities = extract_entity_ids(proposed_patch)
        unknown_entities = sorted(entity for entity in referenced_entities if entity not in known_entity_ids)
        if unknown_entities:
            errors.append(f"item_{idx}:unknown_entities:{','.join(unknown_entities[:5])}")
            continue

        valid.append(
            {
                "schema_version": SUGGESTION_SCHEMA_VERSION,
                "target_entity_id": target_entity_id,
                "summary": summary,
                "confidence": confidence,
                "risk_level": risk_level,
                "proposed_patch": proposed_patch,
                "verification_steps": [step.strip() for step in verification_steps],
            }
        )

    return valid, errors


def validate_concept_suggestion_payload(
    payload: Any,
    known_entity_ids: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    if isinstance(payload, dict) and isinstance(payload.get("suggestions"), list):
        candidates = payload.get("suggestions")
    elif isinstance(payload, list):
        candidates = payload
    else:
        return [], ["response_missing_suggestions_array"]

    valid: list[dict[str, Any]] = []
    errors: list[str] = []

    for idx, item in enumerate(candidates):
        if not isinstance(item, dict):
            errors.append(f"item_{idx}:not_object")
            continue

        title = str(item.get("title", "")).strip()
        summary = str(item.get("summary", "")).strip()
        if not title:
            errors.append(f"item_{idx}:missing_title")
            continue
        if not summary:
            errors.append(f"item_{idx}:missing_summary")
            continue

        concept_type = str(item.get("concept_type", "general")).strip().lower() or "general"
        target_kind = str(item.get("target_kind", "new_automation")).strip().lower() or "new_automation"
        if target_kind not in ALLOWED_TARGET_KINDS:
            errors.append(f"item_{idx}:invalid_target_kind")
            continue

        target_entity_id = str(item.get("target_entity_id", "")).strip().lower()
        if target_kind == "existing_automation":
            if not _is_automation_entity_id(target_entity_id):
                errors.append(f"item_{idx}:invalid_target_entity_id")
                continue
            if known_entity_ids and target_entity_id not in known_entity_ids:
                errors.append(f"item_{idx}:unknown_target_entity")
                continue
        else:
            target_entity_id = ""

        involved_entities = [entity.lower() for entity in _clean_string_list(item.get("involved_entities"), limit=30)]
        if known_entity_ids:
            involved_entities = [entity for entity in involved_entities if entity in known_entity_ids]

        risk_level = str(item.get("risk", item.get("risk_level", "medium"))).strip().lower() or "medium"
        if risk_level not in ALLOWED_RISK_LEVELS:
            risk_level = "medium"

        concept_payload = {
            "schema_version": CONCEPT_SCHEMA_VERSION,
            "title": title,
            "summary": summary,
            "concept_type": concept_type,
            "target_kind": target_kind,
            "target_entity_id": target_entity_id or None,
            "involved_entities": involved_entities,
            "impact_score": _as_score(item.get("impact", item.get("impact_score"))),
            "feasibility_score": _as_score(item.get("feasibility", item.get("feasibility_score"))),
            "novelty_score": _as_score(item.get("novelty", item.get("novelty_score"))),
            "confidence": _as_score(item.get("confidence")),
            "risk_level": risk_level,
            "prerequisites": _clean_string_list(item.get("prerequisites"), limit=15),
            "verification_outline": _clean_string_list(item.get("verification_outline"), limit=15),
            "rationale": str(item.get("rationale", "")).strip(),
        }
        valid.append(concept_payload)

    return valid, errors


def rank_concept_suggestions(
    concepts: list[dict[str, Any]],
    *,
    mode: str,
    top_k: int,
) -> list[dict[str, Any]]:
    if not concepts:
        return []

    normalized_mode = mode.strip().lower() or "standard"
    if normalized_mode not in {"standard", "surprise", "obscure"}:
        normalized_mode = "standard"

    if normalized_mode == "obscure":
        weights = {
            "impact": 0.25,
            "feasibility": 0.15,
            "confidence": 0.10,
            "novelty": 0.50,
        }
        novelty_floor = 0.55
    else:
        weights = {
            "impact": 0.45,
            "feasibility": 0.35,
            "confidence": 0.10,
            "novelty": 0.10,
        }
        novelty_floor = 0.0

    scored: list[dict[str, Any]] = []
    for item in concepts:
        calibration = _calibration_components(item)
        confidence = calibration["confidence"]
        impact = _calibrate_metric(item.get("impact_score"), calibration["calibration_multiplier"])
        feasibility = _calibrate_metric(item.get("feasibility_score"), calibration["calibration_multiplier"])
        novelty = _calibrate_metric(item.get("novelty_score"), calibration["calibration_multiplier"])

        if novelty < novelty_floor:
            continue

        ranking_score = (
            impact * weights["impact"]
            + feasibility * weights["feasibility"]
            + confidence * weights["confidence"]
            + novelty * weights["novelty"]
        )
        scored_item = dict(item)
        scored_item["ranking_score"] = round(ranking_score, 6)
        scored_item["ranking_breakdown"] = {
            "weights": weights,
            "raw_scores": {
                "impact": _as_score(item.get("impact_score")),
                "feasibility": _as_score(item.get("feasibility_score")),
                "confidence": confidence,
                "novelty": _as_score(item.get("novelty_score")),
            },
            "adjusted_scores": {
                "impact": impact,
                "feasibility": feasibility,
                "confidence": confidence,
                "novelty": novelty,
            },
            "calibration": calibration,
            "mode": normalized_mode,
        }
        scored_item["impact_score"] = impact
        scored_item["feasibility_score"] = feasibility
        scored_item["novelty_score"] = novelty
        scored.append(scored_item)

    scored.sort(
        key=lambda item: (
            float(item.get("ranking_score", 0.0)),
            float(item.get("novelty_score", 0.0)),
            float(item.get("impact_score", 0.0)),
            str(item.get("title", "")).lower(),
        ),
        reverse=True,
    )

    if normalized_mode == "surprise":
        per_type_cap = max(1, top_k // 3)
        counts: dict[str, int] = defaultdict(int)
        diverse: list[dict[str, Any]] = []
        overflow: list[dict[str, Any]] = []
        for item in scored:
            concept_type = str(item.get("concept_type", "general")).strip().lower() or "general"
            if counts[concept_type] < per_type_cap:
                diverse.append(item)
                counts[concept_type] += 1
            else:
                overflow.append(item)
        scored = diverse + overflow

    limit = max(1, min(25, top_k))
    return scored[:limit]
