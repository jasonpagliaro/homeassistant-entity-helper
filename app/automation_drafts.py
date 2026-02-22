from __future__ import annotations

import json
import os
from typing import Any

import yaml  # type: ignore[import-untyped]

from app.models import EntitySnapshot

DEFAULT_DRAFT_LIMIT = 50

TEMPLATE_CATALOG: dict[str, dict[str, str]] = {
    "motion_light_auto_off": {
        "title": "Motion Light Auto Off",
        "description": "Turn lights on from motion and off after inactivity.",
    },
    "temperature_alert": {
        "title": "Temperature Alert",
        "description": "Notify when temperature crosses thresholds.",
    },
    "humidity_alert": {
        "title": "Humidity Alert",
        "description": "Notify when humidity exceeds threshold.",
    },
    "door_open_alert": {
        "title": "Door Open Alert",
        "description": "Notify when a door/window remains open.",
    },
    "lock_night_reminder": {
        "title": "Night Lock Reminder",
        "description": "Notify when locks are still unlocked at night.",
    },
}


def parse_json_or_default(raw_json: str | None, default: Any) -> Any:
    if not raw_json:
        return default
    try:
        return json.loads(raw_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def draft_limit_from_env() -> int:
    raw_limit = (os.getenv("HEV_AUTOMATION_DRAFT_MAX_ITEMS") or str(DEFAULT_DRAFT_LIMIT)).strip()
    try:
        value = int(raw_limit)
    except ValueError:
        return DEFAULT_DRAFT_LIMIT
    return max(1, value)


def pick_template_id(domain: str, semantic_type: dict[str, Any]) -> str | None:
    device_class = str(semantic_type.get("device_class") or "").strip().lower()
    if domain == "binary_sensor" and device_class in {"motion", "presence", "occupancy"}:
        return "motion_light_auto_off"
    if domain == "sensor" and device_class == "temperature":
        return "temperature_alert"
    if domain == "sensor" and device_class == "humidity":
        return "humidity_alert"
    if domain == "binary_sensor" and device_class in {"door", "window", "opening", "garage_door"}:
        return "door_open_alert"
    if domain == "lock":
        return "lock_night_reminder"
    return None


def build_draft_prompt_payload(
    entity_snapshot: EntitySnapshot,
    semantic_type: dict[str, Any],
    template_id: str,
    peer_snapshots: list[EntitySnapshot],
) -> dict[str, Any]:
    template = TEMPLATE_CATALOG[template_id]
    peer_entities: list[dict[str, Any]] = []
    for peer in peer_snapshots:
        peer_entities.append(
            {
                "entity_id": peer.entity_id,
                "domain": peer.domain,
                "friendly_name": peer.friendly_name,
                "area_name": peer.area_name,
                "device_name": peer.device_name,
            }
        )

    return {
        "template_id": template_id,
        "template_title": template["title"],
        "template_description": template["description"],
        "entity": {
            "entity_id": entity_snapshot.entity_id,
            "domain": entity_snapshot.domain,
            "state": entity_snapshot.state,
            "friendly_name": entity_snapshot.friendly_name,
            "area_name": entity_snapshot.area_name,
            "device_name": entity_snapshot.device_name,
            "semantic_type": semantic_type,
        },
        "peers_in_same_area": peer_entities[:20],
        "constraints": {
            "home_assistant_format": "automation",
            "must_include_trigger_and_action": True,
            "safe_defaults": True,
        },
    }


def build_automation_structured_payload(
    llm_response: dict[str, Any],
    template_id: str,
    entity_id: str,
) -> dict[str, Any]:
    return {
        "alias": llm_response["alias"],
        "description": llm_response["description"],
        "trigger": llm_response["trigger"],
        "condition": llm_response["condition"],
        "action": llm_response["action"],
        "mode": "single",
        "metadata": {
            "generated_by": "ha_entity_vault",
            "template_id": template_id,
            "source_entity_id": entity_id,
        },
    }


def render_automation_yaml(structured_payload: dict[str, Any]) -> str:
    yaml_text = yaml.safe_dump(
        structured_payload,
        sort_keys=False,
        allow_unicode=False,
        default_flow_style=False,
    )
    return yaml_text.strip() + "\n"
