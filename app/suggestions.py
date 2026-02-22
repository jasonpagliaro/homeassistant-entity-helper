from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from app.models import EntitySnapshot

SUGGESTION_POLICY_VERSION = "v1"
ACTIONABLE_SUGGESTION_DOMAINS = {"sensor", "binary_sensor", "lock"}
WORKFLOW_FIXABLE_ISSUE_CODES = {
    "missing_area",
    "missing_area_enrichment_unavailable",
    "missing_friendly_name",
    "generic_friendly_name",
    "missing_sensor_semantic_type",
    "missing_binary_sensor_device_class",
    "missing_labels",
}
MANUAL_ISSUE_GUIDANCE = {
    "invalid_entity_id_shape": "Rename or recreate this entity in Home Assistant so it has a valid domain.object_id format.",
    "unhealthy_state": "Check integration health in Home Assistant and verify the backing device is online.",
    "missing_device_linkage": "Use Home Assistant device/entity registry tools to link this entity to a device.",
}
GENERIC_FRIENDLY_NAME_PATTERNS = [
    re.compile(r"^(?:entity|device|sensor|switch|light|binary sensor)\s*\d*$", re.IGNORECASE),
    re.compile(r"^[a-z]+(?:_[a-z0-9]+)+$", re.IGNORECASE),
]


def parse_json_or_default(raw_json: str | None, default: Any) -> Any:
    if not raw_json:
        return default
    try:
        return json.loads(raw_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


@dataclass
class SuggestionEvaluation:
    readiness_status: str
    missing_fields: list[str]
    issues: list[dict[str, Any]]
    semantic_type: dict[str, Any]
    source_metadata: dict[str, Any]


def is_supported_entity(snapshot: EntitySnapshot) -> bool:
    domain = (snapshot.domain or "").strip().lower()
    return bool(domain in ACTIONABLE_SUGGESTION_DOMAINS and "." in snapshot.entity_id)


def _metadata_value(metadata: dict[str, Any], key: str) -> Any:
    value = metadata.get(key)
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return value


def _has_label_payload(labels: dict[str, Any]) -> bool:
    for key in ("ids", "names"):
        value = labels.get(key)
        if isinstance(value, list) and any(isinstance(item, str) and item.strip() for item in value):
            return True
    return False


def _is_generic_friendly_name(name: str | None) -> bool:
    if not name:
        return False
    normalized = name.strip()
    if not normalized:
        return False
    for pattern in GENERIC_FRIENDLY_NAME_PATTERNS:
        if pattern.match(normalized):
            return True
    return False


def evaluate_entity_snapshot(snapshot: EntitySnapshot, area_required: bool = True) -> SuggestionEvaluation:
    attributes = parse_json_or_default(snapshot.attributes_json, {})
    metadata = parse_json_or_default(snapshot.metadata_json, {})
    labels = parse_json_or_default(snapshot.labels_json, {})

    missing_fields: list[str] = []
    issues: list[dict[str, Any]] = []

    domain = (snapshot.domain or "").strip().lower()
    semantic_type = {
        "domain": domain or None,
        "device_class": _metadata_value(metadata, "attribute_device_class")
        or (attributes.get("device_class") if isinstance(attributes, dict) else None),
        "state_class": _metadata_value(metadata, "attribute_state_class")
        or (attributes.get("state_class") if isinstance(attributes, dict) else None),
        "entity_category": _metadata_value(metadata, "entity_category")
        or (attributes.get("entity_category") if isinstance(attributes, dict) else None),
    }

    if "." not in snapshot.entity_id or not domain:
        missing_fields.append("entity_id_domain")
        issues.append(
            {
                "severity": "blocker",
                "code": "invalid_entity_id_shape",
                "field": "entity_id",
                "message": "Entity ID is missing a valid domain prefix.",
            }
        )

    if not (snapshot.area_id or snapshot.area_name):
        missing_fields.append("area")
        if area_required:
            issues.append(
                {
                    "severity": "blocker",
                    "code": "missing_area",
                    "field": "area",
                    "message": "Entity is missing area mapping.",
                }
            )
        else:
            issues.append(
                {
                    "severity": "warning",
                    "code": "missing_area_enrichment_unavailable",
                    "field": "area",
                    "message": (
                        "Entity is missing area mapping, but area checks were downgraded "
                        "because this run had no area/device enrichment context."
                    ),
                }
            )

    if not (snapshot.friendly_name or "").strip():
        missing_fields.append("friendly_name")
        issues.append(
            {
                "severity": "blocker",
                "code": "missing_friendly_name",
                "field": "friendly_name",
                "message": "Entity is missing friendly_name.",
            }
        )

    if domain == "sensor" and not (semantic_type["device_class"] or semantic_type["state_class"]):
        missing_fields.append("semantic_type")
        issues.append(
            {
                "severity": "blocker",
                "code": "missing_sensor_semantic_type",
                "field": "semantic_type",
                "message": "Sensor must include device_class or state_class.",
            }
        )

    if domain == "binary_sensor" and not semantic_type["device_class"]:
        missing_fields.append("semantic_type")
        issues.append(
            {
                "severity": "blocker",
                "code": "missing_binary_sensor_device_class",
                "field": "semantic_type",
                "message": "Binary sensor should include a device_class.",
            }
        )

    if domain == "climate":
        hvac_mode = None
        hvac_modes: list[Any] = []
        if isinstance(attributes, dict):
            hvac_mode = attributes.get("hvac_mode")
            hvac_modes_raw = attributes.get("hvac_modes")
            if isinstance(hvac_modes_raw, list):
                hvac_modes = hvac_modes_raw
        if not hvac_mode and not hvac_modes:
            missing_fields.append("climate_hvac_mode")
            issues.append(
                {
                    "severity": "blocker",
                    "code": "missing_climate_hvac_mode",
                    "field": "attributes.hvac_mode",
                    "message": "Climate entities need hvac_mode or hvac_modes for automation intent.",
                }
            )

    has_blocker = any(issue.get("severity") == "blocker" for issue in issues)
    if not has_blocker:
        state_value = (snapshot.state or "").strip().lower()
        if state_value in {"unknown", "unavailable"}:
            issues.append(
                {
                    "severity": "warning",
                    "code": "unhealthy_state",
                    "field": "state",
                    "message": "Entity currently reports unknown/unavailable state.",
                }
            )

        if not _has_label_payload(labels if isinstance(labels, dict) else {}):
            issues.append(
                {
                    "severity": "warning",
                    "code": "missing_labels",
                    "field": "labels",
                    "message": "Entity has no labels.",
                }
            )

        if not ((snapshot.device_id or "").strip() or (snapshot.device_name or "").strip()):
            issues.append(
                {
                    "severity": "warning",
                    "code": "missing_device_linkage",
                    "field": "device",
                    "message": "Entity is not linked to a known device.",
                }
            )

        if _is_generic_friendly_name(snapshot.friendly_name):
            issues.append(
                {
                    "severity": "warning",
                    "code": "generic_friendly_name",
                    "field": "friendly_name",
                    "message": "Friendly name looks generic and may need review.",
                }
            )

    has_warning = any(issue.get("severity") == "warning" for issue in issues)
    if has_blocker:
        readiness_status = "blocked"
    elif has_warning:
        readiness_status = "needs_review"
    else:
        readiness_status = "ready"

    source_metadata = {
        "entity_id": snapshot.entity_id,
        "domain": snapshot.domain,
        "friendly_name": snapshot.friendly_name,
        "area_id": snapshot.area_id,
        "area_name": snapshot.area_name,
        "device_id": snapshot.device_id,
        "device_name": snapshot.device_name,
        "location_name": snapshot.location_name,
    }

    deduped_missing_fields = sorted(set(missing_fields))
    return SuggestionEvaluation(
        readiness_status=readiness_status,
        missing_fields=deduped_missing_fields,
        issues=issues,
        semantic_type=semantic_type,
        source_metadata=source_metadata,
    )


def build_llm_suggestion_payload(
    snapshot: EntitySnapshot,
    evaluation: SuggestionEvaluation,
) -> dict[str, Any]:
    attributes = parse_json_or_default(snapshot.attributes_json, {})
    metadata = parse_json_or_default(snapshot.metadata_json, {})
    labels = parse_json_or_default(snapshot.labels_json, {})
    return {
        "policy_version": SUGGESTION_POLICY_VERSION,
        "entity": {
            "entity_id": snapshot.entity_id,
            "domain": snapshot.domain,
            "state": snapshot.state,
            "friendly_name": snapshot.friendly_name,
            "area_id": snapshot.area_id,
            "area_name": snapshot.area_name,
            "device_id": snapshot.device_id,
            "device_name": snapshot.device_name,
            "location_name": snapshot.location_name,
            "semantic_type": evaluation.semantic_type,
            "labels": labels if isinstance(labels, dict) else {},
        },
        "issues": evaluation.issues,
        "missing_fields": evaluation.missing_fields,
        "attributes": attributes if isinstance(attributes, dict) else {},
        "metadata": metadata if isinstance(metadata, dict) else {},
    }


def _issue_code(issue: dict[str, Any]) -> str:
    raw_code = issue.get("code")
    if not isinstance(raw_code, str):
        return ""
    return raw_code.strip()


def split_issues_by_fixability(
    issues: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fixable: list[dict[str, Any]] = []
    manual_only: list[dict[str, Any]] = []
    for issue in issues:
        code = _issue_code(issue)
        if code in WORKFLOW_FIXABLE_ISSUE_CODES:
            fixable.append(issue)
        else:
            manual_only.append(issue)
    return fixable, manual_only


def is_fixable_issue_code(code: str) -> bool:
    return code in WORKFLOW_FIXABLE_ISSUE_CODES


def manual_guidance_for_issue(issue: dict[str, Any]) -> str:
    code = _issue_code(issue)
    if code in MANUAL_ISSUE_GUIDANCE:
        return MANUAL_ISSUE_GUIDANCE[code]
    return "Review this issue directly in Home Assistant and rerun suggestions after correcting it."
