from __future__ import annotations

import json
from typing import Any

import yaml  # type: ignore[import-untyped]

FLOW_METADATA_VERSION = 1
FLOW_VARIABLE_KEY = "_haev_flow"


def merge_automation_documents(
    base_document: dict[str, Any] | None,
    updated_document: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(base_document or {})
    merged.update(updated_document)
    return merged


def normalize_automation_document(
    raw_payload: dict[str, Any],
    *,
    base_document: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(raw_payload, dict):
        raise ValueError("Automation payload must decode to an object.")

    normalized = merge_automation_documents(base_document, raw_payload)

    alias = str(normalized.get("alias", "")).strip()
    if not alias:
        raise ValueError("Automation payload must include alias.")

    description = normalized.get("description")
    if description is None:
        description = ""
    if not isinstance(description, str):
        raise ValueError("Automation description must be a string.")

    trigger = normalized.get("trigger")
    if trigger is None:
        trigger = normalized.get("triggers")
    if not isinstance(trigger, list) or not trigger:
        raise ValueError("Automation payload must include at least one trigger.")

    condition = normalized.get("condition")
    if condition is None:
        condition = normalized.get("conditions")
    if condition is None:
        condition = []
    if not isinstance(condition, list):
        raise ValueError("Automation condition must be a list.")

    action = normalized.get("action")
    if action is None:
        action = normalized.get("actions")
    if not isinstance(action, list) or not action:
        raise ValueError("Automation payload must include at least one action.")

    mode = normalized.get("mode")
    if not isinstance(mode, str) or not mode.strip():
        mode = "single"

    variables = normalized.get("variables")
    if variables is not None and not isinstance(variables, dict):
        raise ValueError("Automation variables must be an object when present.")

    normalized["alias"] = alias
    normalized["description"] = description.strip()
    normalized["trigger"] = trigger
    normalized["condition"] = condition
    normalized["action"] = action
    normalized["mode"] = mode.strip()
    if "triggers" in normalized:
        del normalized["triggers"]
    if "conditions" in normalized:
        del normalized["conditions"]
    if "actions" in normalized:
        del normalized["actions"]
    return normalized


def yaml_from_automation_document(payload: dict[str, Any]) -> str:
    yaml_text = yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=False,
        default_flow_style=False,
    )
    return yaml_text.strip() + "\n"


def parse_automation_yaml_text(
    yaml_text: str,
    *,
    base_document: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Automation YAML must decode to an object.")
    return normalize_automation_document(parsed, base_document=base_document)


def parse_automation_json_text(
    raw_json: str,
    *,
    base_document: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid automation JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Automation JSON must decode to an object.")
    return normalize_automation_document(parsed, base_document=base_document)
