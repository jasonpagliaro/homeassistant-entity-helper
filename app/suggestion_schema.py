from __future__ import annotations

import re
from typing import Any

SUGGESTION_SCHEMA_VERSION = "haev.automation.suggestion.v1"
ALLOWED_RISK_LEVELS = {"low", "medium", "high"}
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
